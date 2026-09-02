"""
用于方程（17）的中继以及混合直连/中继资源分配。

原始的 :mod:dm_jcr.resource_allocation 模块处理直连任务，即一个任务在单个边缘节点上消耗带宽、CPU频率和发射功率。而中继计算任务则不同：它同时消耗中继无人机和最终计算节点上的资源。

本模块新增三项功能：

中继任务的上下文表示，以及原始/投影后的中继资源分配表示；

联合投影机制，使直连任务和中继任务共享每个节点的带宽、CPU和发射功率总预算；

仅中继场景和混合直连/中继场景的评估，依据方程（8）、（12）、（13）、（17）以及方程（27）中的比例归一化方法。

带宽核算假设

论文并未明确说明四个中继跳如何共享每个节点的单一带宽预算。本实现采用透明的逐跳核算规则：

车辆 → 中继无人机、中继无人机 → 计算节点，以及中继无人机 → 车辆的带宽，均计入中继无人机的带宽消耗；

计算节点 → 中继无人机的带宽，计入计算节点的带宽消耗。

中继无人机分配有一个发射功率值。该值在其两条出站链路上重复使用，因为方程（13）对两次中继传输使用同一个 p_u 变量。各跳按顺序进行评估，因此该值仅对无人机的瞬时功率预算计一次。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

import numpy as np

from dm_jcr.channel import achievable_rate_bps
from dm_jcr.relay_model import RelayLinkResources, RelayTaskMetrics, evaluate_relay_task
from dm_jcr.resource_allocation import (
    DirectTaskContext,
    FeasibleTaskAllocation,
    NodeResourceCapacity,
    ObjectiveNormalization,
    ObjectiveWeights,
    RawTaskAllocation,
    TaskAllocationEvaluation,
    evaluate_direct_resource_strategy,
    proportional_normalize,
)
from dm_jcr.task_model import DEFAULT_CPU_ENERGY_COEFFICIENT, ComputationTask

_EPSILON = 1.0e-12


def _clean_identifier(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value.strip()


def _finite_non_negative(name: str, value: float) -> float:
    value = float(value)
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} 必须为非负有限数值")
    return value


def _finite_positive(name: str, value: float) -> float:
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} 必须为大于 0 的有限数值")
    return value


def _index_capacities(
    capacities: Iterable[NodeResourceCapacity],
) -> dict[str, NodeResourceCapacity]:
    indexed: dict[str, NodeResourceCapacity] = {}
    for capacity in capacities:
        if capacity.node_id in indexed:
            raise ValueError(f"节点 {capacity.node_id!r} 的容量配置重复")
        indexed[capacity.node_id] = capacity
    if not indexed:
        raise ValueError("至少需要一个节点容量配置")
    return indexed


@dataclass(frozen=True)
class RelayTaskContext:
    """一个中继任务的固定环境与映射信息。

    ``relay_uav_id`` 和 ``compute_node_id`` 在资源分配前固定，四个信道增益表示
    四跳通信。车辆发射功率属于环境输入；中继无人机和计算节点的功率由分配策略生成。
    """

    task_id: str
    relay_uav_id: str
    compute_node_id: str
    task: ComputationTask

    vehicle_to_uav_channel_gain: float
    uav_to_node_channel_gain: float
    node_to_uav_channel_gain: float
    uav_to_vehicle_channel_gain: float

    vehicle_transmit_power_w: float
    noise_psd_w_hz: float
    forwarding_cycles_per_bit: float

    vehicle_to_uav_interference_power_w: float = 0.0
    uav_to_node_interference_power_w: float = 0.0
    node_to_uav_interference_power_w: float = 0.0
    uav_to_vehicle_interference_power_w: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _clean_identifier("task_id", self.task_id))
        object.__setattr__(
            self,
            "relay_uav_id",
            _clean_identifier("relay_uav_id", self.relay_uav_id),
        )
        object.__setattr__(
            self,
            "compute_node_id",
            _clean_identifier("compute_node_id", self.compute_node_id),
        )
        if self.relay_uav_id == self.compute_node_id:
            raise ValueError("relay_uav_id 与 compute_node_id 必须不同")
        if not isinstance(self.task, ComputationTask):
            raise TypeError("task 必须是 ComputationTask")

        for name in (
            "vehicle_to_uav_channel_gain",
            "uav_to_node_channel_gain",
            "node_to_uav_channel_gain",
            "uav_to_vehicle_channel_gain",
        ):
            object.__setattr__(self, name, _finite_positive(name, getattr(self, name)))

        object.__setattr__(
            self,
            "vehicle_transmit_power_w",
            _finite_non_negative(
                "vehicle_transmit_power_w", self.vehicle_transmit_power_w
            ),
        )
        object.__setattr__(
            self,
            "noise_psd_w_hz",
            _finite_positive("noise_psd_w_hz", self.noise_psd_w_hz),
        )
        object.__setattr__(
            self,
            "forwarding_cycles_per_bit",
            _finite_non_negative(
                "forwarding_cycles_per_bit", self.forwarding_cycles_per_bit
            ),
        )

        for name in (
            "vehicle_to_uav_interference_power_w",
            "uav_to_node_interference_power_w",
            "node_to_uav_interference_power_w",
            "uav_to_vehicle_interference_power_w",
        ):
            object.__setattr__(
                self, name, _finite_non_negative(name, getattr(self, name))
            )


@dataclass(frozen=True)
class RawRelayTaskAllocation:
    """一个中继计算任务的无约束非负分数。

    四个带宽分数与对应节点上的其他带宽请求共同归一化；CPU 与功率分数分别独立
    归一化，与公式（27）一致。
    """

    task_id: str
    relay_uav_id: str
    compute_node_id: str

    vehicle_to_uav_bandwidth_score: float
    uav_to_node_bandwidth_score: float
    node_to_uav_bandwidth_score: float
    uav_to_vehicle_bandwidth_score: float

    relay_cpu_score: float
    compute_cpu_score: float
    relay_power_score: float
    compute_node_power_score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _clean_identifier("task_id", self.task_id))
        object.__setattr__(
            self,
            "relay_uav_id",
            _clean_identifier("relay_uav_id", self.relay_uav_id),
        )
        object.__setattr__(
            self,
            "compute_node_id",
            _clean_identifier("compute_node_id", self.compute_node_id),
        )
        if self.relay_uav_id == self.compute_node_id:
            raise ValueError("relay_uav_id 与 compute_node_id 必须不同")

        for name in (
            "vehicle_to_uav_bandwidth_score",
            "uav_to_node_bandwidth_score",
            "node_to_uav_bandwidth_score",
            "uav_to_vehicle_bandwidth_score",
            "relay_cpu_score",
            "compute_cpu_score",
            "relay_power_score",
            "compute_node_power_score",
        ):
            object.__setattr__(
                self, name, _finite_non_negative(name, getattr(self, name))
            )


@dataclass(frozen=True)
class FeasibleRelayTaskAllocation:
    """投影后分配给一个中继任务的物理资源。"""

    task_id: str
    relay_uav_id: str
    compute_node_id: str

    vehicle_to_uav_bandwidth_hz: float
    uav_to_node_bandwidth_hz: float
    node_to_uav_bandwidth_hz: float
    uav_to_vehicle_bandwidth_hz: float

    relay_cpu_frequency_hz: float
    compute_cpu_frequency_hz: float
    relay_transmit_power_w: float
    compute_node_transmit_power_w: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _clean_identifier("task_id", self.task_id))
        object.__setattr__(
            self,
            "relay_uav_id",
            _clean_identifier("relay_uav_id", self.relay_uav_id),
        )
        object.__setattr__(
            self,
            "compute_node_id",
            _clean_identifier("compute_node_id", self.compute_node_id),
        )
        if self.relay_uav_id == self.compute_node_id:
            raise ValueError("relay_uav_id 与 compute_node_id 必须不同")

        for name in (
            "vehicle_to_uav_bandwidth_hz",
            "uav_to_node_bandwidth_hz",
            "node_to_uav_bandwidth_hz",
            "uav_to_vehicle_bandwidth_hz",
            "relay_cpu_frequency_hz",
            "compute_cpu_frequency_hz",
        ):
            object.__setattr__(self, name, _finite_positive(name, getattr(self, name)))

        for name in (
            "relay_transmit_power_w",
            "compute_node_transmit_power_w",
        ):
            object.__setattr__(
                self, name, _finite_non_negative(name, getattr(self, name))
            )

    @property
    def relay_bandwidth_total_hz(self) -> float:
        """该任务计入中继无人机的带宽总量。"""

        return float(
            self.vehicle_to_uav_bandwidth_hz
            + self.uav_to_node_bandwidth_hz
            + self.uav_to_vehicle_bandwidth_hz
        )

    @property
    def compute_node_bandwidth_total_hz(self) -> float:
        """计入最终计算节点的带宽总量。"""

        return float(self.node_to_uav_bandwidth_hz)


@dataclass(frozen=True)
class ProjectedJointResourceStrategy:
    """一个时隙内投影后的直连与中继分配。"""

    direct_allocations: tuple[FeasibleTaskAllocation, ...]
    relay_allocations: tuple[FeasibleRelayTaskAllocation, ...]

    def __post_init__(self) -> None:
        direct_ids = [item.task_id for item in self.direct_allocations]
        relay_ids = [item.task_id for item in self.relay_allocations]
        all_ids = direct_ids + relay_ids
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("直连与中继任务的 task_id 必须唯一")
        if not all_ids:
            raise ValueError("至少需要一个直连或中继分配")


@dataclass(frozen=True)
class RelayTaskAllocationEvaluation:
    """一个中继任务的速率、指标和归一化目标项。"""

    task_id: str
    relay_uav_id: str
    compute_node_id: str
    allocation: FeasibleRelayTaskAllocation

    vehicle_to_uav_rate_bps: float
    uav_to_node_rate_bps: float
    node_to_uav_rate_bps: float
    uav_to_vehicle_rate_bps: float

    metrics: RelayTaskMetrics
    normalized_latency: float
    normalized_energy: float


@dataclass(frozen=True)
class RelayAllocationEvaluation:
    """纯中继策略的目标值与可行性报告。"""

    tasks: tuple[RelayTaskAllocationEvaluation, ...]
    mean_normalized_latency: float
    mean_normalized_energy: float
    weighted_objective: float
    resource_constraints_satisfied: bool
    deadline_constraints_satisfied: bool

    @property
    def feasible(self) -> bool:
        return (
            self.resource_constraints_satisfied
            and self.deadline_constraints_satisfied
        )


@dataclass(frozen=True)
class SystemAllocationEvaluation:
    """直连与中继任务的统一公式（17）评价。"""

    direct_tasks: tuple[TaskAllocationEvaluation, ...]
    relay_tasks: tuple[RelayTaskAllocationEvaluation, ...]
    mean_normalized_latency: float
    mean_normalized_energy: float
    weighted_objective: float
    resource_constraints_satisfied: bool
    deadline_constraints_satisfied: bool

    @property
    def feasible(self) -> bool:
        return (
            self.resource_constraints_satisfied
            and self.deadline_constraints_satisfied
        )

    @property
    def task_count(self) -> int:
        return len(self.direct_tasks) + len(self.relay_tasks)


@dataclass(frozen=True)
class _ResourceSlot:
    owner_kind: str
    owner_index: int
    field_name: str
    score: float


def _append_slot(
    slots_by_node: dict[str, list[_ResourceSlot]],
    *,
    node_id: str,
    owner_kind: str,
    owner_index: int,
    field_name: str,
    score: float,
) -> None:
    slots_by_node.setdefault(node_id, []).append(
        _ResourceSlot(
            owner_kind=owner_kind,
            owner_index=owner_index,
            field_name=field_name,
            score=score,
        )
    )


def _assign_normalized_slots(
    slots_by_node: dict[str, list[_ResourceSlot]],
    capacity_by_node: dict[str, NodeResourceCapacity],
    capacity_attribute: str,
    direct_values: list[dict[str, float]],
    relay_values: list[dict[str, float]],
) -> None:
    for node_id, slots in slots_by_node.items():
        capacity = capacity_by_node[node_id]
        normalized = proportional_normalize(
            (slot.score for slot in slots),
            getattr(capacity, capacity_attribute),
        )
        for slot, value in zip(slots, normalized, strict=True):
            target = (
                direct_values[slot.owner_index]
                if slot.owner_kind == "direct"
                else relay_values[slot.owner_index]
            )
            target[slot.field_name] = float(value)


def project_joint_resource_strategy(
    direct_raw_allocations: Iterable[RawTaskAllocation],
    relay_raw_allocations: Iterable[RawRelayTaskAllocation],
    capacities: Iterable[NodeResourceCapacity],
) -> ProjectedJointResourceStrategy:
    """将直连和中继分数投影到各节点的共享容量上。

    节点上的全部带宽请求共同执行一次比例归一化，全部 CPU 请求和全部发射功率
    请求再分别归一化，因此直连与中继任务竞争相同的物理资源。
    """

    direct_raw = tuple(direct_raw_allocations)
    relay_raw = tuple(relay_raw_allocations)
    if not direct_raw and not relay_raw:
        raise ValueError("至少需要一个直连或中继原始分配")

    capacity_by_node = _index_capacities(capacities)
    all_task_ids = [item.task_id for item in direct_raw] + [
        item.task_id for item in relay_raw
    ]
    if len(set(all_task_ids)) != len(all_task_ids):
        raise ValueError("直连与中继任务的 task_id 必须唯一")

    direct_values: list[dict[str, float]] = [dict() for _ in direct_raw]
    relay_values: list[dict[str, float]] = [dict() for _ in relay_raw]

    bandwidth_slots: dict[str, list[_ResourceSlot]] = {}
    cpu_slots: dict[str, list[_ResourceSlot]] = {}
    power_slots: dict[str, list[_ResourceSlot]] = {}

    for index, item in enumerate(direct_raw):
        if item.node_id not in capacity_by_node:
            raise ValueError(f"缺少节点 {item.node_id!r} 的容量配置")
        _append_slot(
            bandwidth_slots,
            node_id=item.node_id,
            owner_kind="direct",
            owner_index=index,
            field_name="bandwidth_hz",
            score=item.bandwidth_score,
        )
        _append_slot(
            cpu_slots,
            node_id=item.node_id,
            owner_kind="direct",
            owner_index=index,
            field_name="cpu_frequency_hz",
            score=item.cpu_score,
        )
        _append_slot(
            power_slots,
            node_id=item.node_id,
            owner_kind="direct",
            owner_index=index,
            field_name="node_transmit_power_w",
            score=item.power_score,
        )

    for index, item in enumerate(relay_raw):
        for node_id in (item.relay_uav_id, item.compute_node_id):
            if node_id not in capacity_by_node:
                raise ValueError(f"缺少节点 {node_id!r} 的容量配置")

        for field_name, score in (
            (
                "vehicle_to_uav_bandwidth_hz",
                item.vehicle_to_uav_bandwidth_score,
            ),
            ("uav_to_node_bandwidth_hz", item.uav_to_node_bandwidth_score),
            (
                "uav_to_vehicle_bandwidth_hz",
                item.uav_to_vehicle_bandwidth_score,
            ),
        ):
            _append_slot(
                bandwidth_slots,
                node_id=item.relay_uav_id,
                owner_kind="relay",
                owner_index=index,
                field_name=field_name,
                score=score,
            )

        _append_slot(
            bandwidth_slots,
            node_id=item.compute_node_id,
            owner_kind="relay",
            owner_index=index,
            field_name="node_to_uav_bandwidth_hz",
            score=item.node_to_uav_bandwidth_score,
        )
        _append_slot(
            cpu_slots,
            node_id=item.relay_uav_id,
            owner_kind="relay",
            owner_index=index,
            field_name="relay_cpu_frequency_hz",
            score=item.relay_cpu_score,
        )
        _append_slot(
            cpu_slots,
            node_id=item.compute_node_id,
            owner_kind="relay",
            owner_index=index,
            field_name="compute_cpu_frequency_hz",
            score=item.compute_cpu_score,
        )
        _append_slot(
            power_slots,
            node_id=item.relay_uav_id,
            owner_kind="relay",
            owner_index=index,
            field_name="relay_transmit_power_w",
            score=item.relay_power_score,
        )
        _append_slot(
            power_slots,
            node_id=item.compute_node_id,
            owner_kind="relay",
            owner_index=index,
            field_name="compute_node_transmit_power_w",
            score=item.compute_node_power_score,
        )

    _assign_normalized_slots(
        bandwidth_slots,
        capacity_by_node,
        "total_bandwidth_hz",
        direct_values,
        relay_values,
    )
    _assign_normalized_slots(
        cpu_slots,
        capacity_by_node,
        "total_cpu_frequency_hz",
        direct_values,
        relay_values,
    )
    _assign_normalized_slots(
        power_slots,
        capacity_by_node,
        "total_transmit_power_w",
        direct_values,
        relay_values,
    )

    direct_allocations = tuple(
        FeasibleTaskAllocation(
            task_id=item.task_id,
            node_id=item.node_id,
            bandwidth_hz=direct_values[index]["bandwidth_hz"],
            cpu_frequency_hz=direct_values[index]["cpu_frequency_hz"],
            node_transmit_power_w=direct_values[index]["node_transmit_power_w"],
        )
        for index, item in enumerate(direct_raw)
    )
    relay_allocations = tuple(
        FeasibleRelayTaskAllocation(
            task_id=item.task_id,
            relay_uav_id=item.relay_uav_id,
            compute_node_id=item.compute_node_id,
            vehicle_to_uav_bandwidth_hz=relay_values[index][
                "vehicle_to_uav_bandwidth_hz"
            ],
            uav_to_node_bandwidth_hz=relay_values[index][
                "uav_to_node_bandwidth_hz"
            ],
            node_to_uav_bandwidth_hz=relay_values[index][
                "node_to_uav_bandwidth_hz"
            ],
            uav_to_vehicle_bandwidth_hz=relay_values[index][
                "uav_to_vehicle_bandwidth_hz"
            ],
            relay_cpu_frequency_hz=relay_values[index][
                "relay_cpu_frequency_hz"
            ],
            compute_cpu_frequency_hz=relay_values[index][
                "compute_cpu_frequency_hz"
            ],
            relay_transmit_power_w=relay_values[index][
                "relay_transmit_power_w"
            ],
            compute_node_transmit_power_w=relay_values[index][
                "compute_node_transmit_power_w"
            ],
        )
        for index, item in enumerate(relay_raw)
    )

    return ProjectedJointResourceStrategy(
        direct_allocations=direct_allocations,
        relay_allocations=relay_allocations,
    )


def joint_resource_totals(
    strategy: ProjectedJointResourceStrategy,
) -> dict[str, tuple[float, float, float]]:
    """返回各节点的 ``(带宽, CPU, 功率)`` 总量以供检查。"""

    totals: dict[str, np.ndarray] = {}

    def add(node_id: str, bandwidth: float, cpu: float, power: float) -> None:
        totals.setdefault(node_id, np.zeros(3, dtype=np.float64))
        totals[node_id] += np.array([bandwidth, cpu, power], dtype=np.float64)

    for item in strategy.direct_allocations:
        add(
            item.node_id,
            item.bandwidth_hz,
            item.cpu_frequency_hz,
            item.node_transmit_power_w,
        )

    for item in strategy.relay_allocations:
        add(
            item.relay_uav_id,
            item.relay_bandwidth_total_hz,
            item.relay_cpu_frequency_hz,
            item.relay_transmit_power_w,
        )
        add(
            item.compute_node_id,
            item.compute_node_bandwidth_total_hz,
            item.compute_cpu_frequency_hz,
            item.compute_node_transmit_power_w,
        )

    return {
        node_id: (float(values[0]), float(values[1]), float(values[2]))
        for node_id, values in totals.items()
    }


def verify_joint_resource_constraints(
    strategy: ProjectedJointResourceStrategy,
    capacities: Iterable[NodeResourceCapacity],
    *,
    tolerance: float = 1.0e-8,
) -> bool:
    """检查混合直连/中继场景的带宽、CPU 和功率约束。"""

    tolerance = _finite_non_negative("tolerance", tolerance)
    capacity_by_node = _index_capacities(capacities)
    totals = joint_resource_totals(strategy)

    for node_id, values in totals.items():
        capacity = capacity_by_node.get(node_id)
        if capacity is None:
            return False
        limits = (
            capacity.total_bandwidth_hz,
            capacity.total_cpu_frequency_hz,
            capacity.total_transmit_power_w,
        )
        if any(value < -tolerance for value in values):
            return False
        if any(value - limit > tolerance for value, limit in zip(values, limits)):
            return False
    return True


def _relay_context_and_allocation_maps(
    contexts: Iterable[RelayTaskContext],
    allocations: Iterable[FeasibleRelayTaskAllocation],
) -> tuple[
    tuple[RelayTaskContext, ...],
    tuple[FeasibleRelayTaskAllocation, ...],
    dict[str, RelayTaskContext],
    dict[str, FeasibleRelayTaskAllocation],
]:
    context_items = tuple(contexts)
    allocation_items = tuple(allocations)
    if not context_items:
        raise ValueError("至少需要一个中继任务上下文")

    context_by_task = {item.task_id: item for item in context_items}
    allocation_by_task = {item.task_id: item for item in allocation_items}
    if len(context_by_task) != len(context_items):
        raise ValueError("中继上下文中的 task_id 必须唯一")
    if len(allocation_by_task) != len(allocation_items):
        raise ValueError("中继分配中的 task_id 必须唯一")
    if context_by_task.keys() != allocation_by_task.keys():
        raise ValueError("中继上下文与分配必须包含相同任务")

    return context_items, allocation_items, context_by_task, allocation_by_task


def evaluate_relay_resource_strategy(
    contexts: Iterable[RelayTaskContext],
    allocations: Iterable[FeasibleRelayTaskAllocation],
    capacities: Iterable[NodeResourceCapacity],
    normalization: ObjectiveNormalization,
    weights: ObjectiveWeights = ObjectiveWeights(),
    *,
    energy_coefficient: float = DEFAULT_CPU_ENERGY_COEFFICIENT,
) -> RelayAllocationEvaluation:
    """评价投影后的纯中继策略。

    先通过公式（8）计算四个速率，再由
    :func:`dm_jcr.relay_model.evaluate_relay_task` 计算公式（12）至（13）中的
    七个时延分量和七个能耗分量。
    """

    (
        context_items,
        allocation_items,
        _context_by_task,
        allocation_by_task,
    ) = _relay_context_and_allocation_maps(contexts, allocations)

    task_results: list[RelayTaskAllocationEvaluation] = []
    for context in context_items:
        allocation = allocation_by_task[context.task_id]
        if allocation.relay_uav_id != context.relay_uav_id:
            raise ValueError(
                f"任务 {context.task_id!r} 使用中继 {context.relay_uav_id!r}，"
                f"而不是 {allocation.relay_uav_id!r}"
            )
        if allocation.compute_node_id != context.compute_node_id:
            raise ValueError(
                f"任务 {context.task_id!r} 使用计算节点 "
                f"{context.compute_node_id!r}，而不是 {allocation.compute_node_id!r}"
            )

        vehicle_to_uav_rate = achievable_rate_bps(
            bandwidth_hz=allocation.vehicle_to_uav_bandwidth_hz,
            transmit_power_w=context.vehicle_transmit_power_w,
            channel_gain=context.vehicle_to_uav_channel_gain,
            noise_psd_w_hz=context.noise_psd_w_hz,
            interference_power_w=context.vehicle_to_uav_interference_power_w,
        )
        uav_to_node_rate = achievable_rate_bps(
            bandwidth_hz=allocation.uav_to_node_bandwidth_hz,
            transmit_power_w=allocation.relay_transmit_power_w,
            channel_gain=context.uav_to_node_channel_gain,
            noise_psd_w_hz=context.noise_psd_w_hz,
            interference_power_w=context.uav_to_node_interference_power_w,
        )
        node_to_uav_rate = achievable_rate_bps(
            bandwidth_hz=allocation.node_to_uav_bandwidth_hz,
            transmit_power_w=allocation.compute_node_transmit_power_w,
            channel_gain=context.node_to_uav_channel_gain,
            noise_psd_w_hz=context.noise_psd_w_hz,
            interference_power_w=context.node_to_uav_interference_power_w,
        )
        uav_to_vehicle_rate = achievable_rate_bps(
            bandwidth_hz=allocation.uav_to_vehicle_bandwidth_hz,
            transmit_power_w=allocation.relay_transmit_power_w,
            channel_gain=context.uav_to_vehicle_channel_gain,
            noise_psd_w_hz=context.noise_psd_w_hz,
            interference_power_w=context.uav_to_vehicle_interference_power_w,
        )

        rates = (
            vehicle_to_uav_rate,
            uav_to_node_rate,
            node_to_uav_rate,
            uav_to_vehicle_rate,
        )
        if any(rate <= _EPSILON for rate in rates):
            raise ValueError("已分配的中继资源产生了零通信速率")

        metrics = evaluate_relay_task(
            task=context.task,
            resources=RelayLinkResources(
                vehicle_to_uav_rate_bps=vehicle_to_uav_rate,
                uav_to_node_rate_bps=uav_to_node_rate,
                node_to_uav_rate_bps=node_to_uav_rate,
                uav_to_vehicle_rate_bps=uav_to_vehicle_rate,
                uav_cpu_frequency_hz=allocation.relay_cpu_frequency_hz,
                node_cpu_frequency_hz=allocation.compute_cpu_frequency_hz,
                vehicle_tx_power_w=context.vehicle_transmit_power_w,
                uav_tx_power_w=allocation.relay_transmit_power_w,
                node_tx_power_w=allocation.compute_node_transmit_power_w,
                forwarding_cycles_per_bit=context.forwarding_cycles_per_bit,
            ),
            energy_coefficient=energy_coefficient,
        )
        latency_reference = (
            normalization.latency_reference_s
            if normalization.latency_reference_s is not None
            else context.task.max_latency_s
        )
        task_results.append(
            RelayTaskAllocationEvaluation(
                task_id=context.task_id,
                relay_uav_id=context.relay_uav_id,
                compute_node_id=context.compute_node_id,
                allocation=allocation,
                vehicle_to_uav_rate_bps=float(vehicle_to_uav_rate),
                uav_to_node_rate_bps=float(uav_to_node_rate),
                node_to_uav_rate_bps=float(node_to_uav_rate),
                uav_to_vehicle_rate_bps=float(uav_to_vehicle_rate),
                metrics=metrics,
                normalized_latency=float(
                    metrics.total_latency_s / latency_reference
                ),
                normalized_energy=float(
                    metrics.total_energy_j / normalization.energy_reference_j
                ),
            )
        )

    mean_latency = float(
        np.mean([item.normalized_latency for item in task_results])
    )
    mean_energy = float(np.mean([item.normalized_energy for item in task_results]))
    objective = weights.latency * mean_latency + weights.energy * mean_energy
    strategy = ProjectedJointResourceStrategy(
        direct_allocations=(),
        relay_allocations=allocation_items,
    )

    return RelayAllocationEvaluation(
        tasks=tuple(task_results),
        mean_normalized_latency=mean_latency,
        mean_normalized_energy=mean_energy,
        weighted_objective=float(objective),
        resource_constraints_satisfied=verify_joint_resource_constraints(
            strategy, capacities
        ),
        deadline_constraints_satisfied=all(
            item.metrics.deadline_satisfied for item in task_results
        ),
    )


def evaluate_joint_resource_strategy(
    direct_contexts: Iterable[DirectTaskContext],
    relay_contexts: Iterable[RelayTaskContext],
    strategy: ProjectedJointResourceStrategy,
    capacities: Iterable[NodeResourceCapacity],
    normalization: ObjectiveNormalization,
    weights: ObjectiveWeights = ObjectiveWeights(),
    *,
    energy_coefficient: float = DEFAULT_CPU_ENERGY_COEFFICIENT,
) -> SystemAllocationEvaluation:
    """在同一个公式（17）目标中评价直连与中继任务。

    最终时延与能耗均值按所有任务计算，而不是先按两类任务分别计算。因此，除非
    调用方改变任务集合，否则一个直连任务与一个中继任务在目标中权重相同。
    """

    direct_context_items = tuple(direct_contexts)
    relay_context_items = tuple(relay_contexts)
    capacity_items = tuple(capacities)
    if not direct_context_items and not relay_context_items:
        raise ValueError("至少需要一个直连或中继任务上下文")

    context_task_ids = [item.task_id for item in direct_context_items] + [
        item.task_id for item in relay_context_items
    ]
    if len(set(context_task_ids)) != len(context_task_ids):
        raise ValueError("直连与中继上下文中的 task_id 必须唯一")

    direct_results: tuple[TaskAllocationEvaluation, ...] = ()
    relay_results: tuple[RelayTaskAllocationEvaluation, ...] = ()

    if direct_context_items:
        direct_evaluation = evaluate_direct_resource_strategy(
            contexts=direct_context_items,
            allocations=strategy.direct_allocations,
            capacities=capacity_items,
            normalization=normalization,
            weights=weights,
            energy_coefficient=energy_coefficient,
        )
        direct_results = direct_evaluation.tasks
    elif strategy.direct_allocations:
        raise ValueError("提供了直连分配，但未提供直连上下文")

    if relay_context_items:
        relay_evaluation = evaluate_relay_resource_strategy(
            contexts=relay_context_items,
            allocations=strategy.relay_allocations,
            capacities=capacity_items,
            normalization=normalization,
            weights=weights,
            energy_coefficient=energy_coefficient,
        )
        relay_results = relay_evaluation.tasks
    elif strategy.relay_allocations:
        raise ValueError("提供了中继分配，但未提供中继上下文")

    normalized_latencies = [item.normalized_latency for item in direct_results]
    normalized_latencies.extend(item.normalized_latency for item in relay_results)
    normalized_energies = [item.normalized_energy for item in direct_results]
    normalized_energies.extend(item.normalized_energy for item in relay_results)

    mean_latency = float(np.mean(normalized_latencies))
    mean_energy = float(np.mean(normalized_energies))
    objective = weights.latency * mean_latency + weights.energy * mean_energy

    return SystemAllocationEvaluation(
        direct_tasks=direct_results,
        relay_tasks=relay_results,
        mean_normalized_latency=mean_latency,
        mean_normalized_energy=mean_energy,
        weighted_objective=float(objective),
        resource_constraints_satisfied=verify_joint_resource_constraints(
            strategy, capacity_items
        ),
        deadline_constraints_satisfied=(
            all(item.metrics.meets_deadline for item in direct_results)
            and all(item.metrics.deadline_satisfied for item in relay_results)
        ),
    )


__all__ = [
    "FeasibleRelayTaskAllocation",
    "ProjectedJointResourceStrategy",
    "RawRelayTaskAllocation",
    "RelayAllocationEvaluation",
    "RelayTaskAllocationEvaluation",
    "RelayTaskContext",
    "SystemAllocationEvaluation",
    "evaluate_joint_resource_strategy",
    "evaluate_relay_resource_strategy",
    "joint_resource_totals",
    "project_joint_resource_strategy",
    "verify_joint_resource_constraints",
]
