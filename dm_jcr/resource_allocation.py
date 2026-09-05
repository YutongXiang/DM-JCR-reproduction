"""用于方程（17）和（27）的联合通信/计算资源分配。

本模块中，由 :mod:dm_jcr.offloading 生成的任务到节点的映射关系是固定的。对于每个已映射的任务，一种分配策略会分配三种连续资源：

带宽 b；

边缘CPU频率 f；

边缘节点发射功率 p。

论文采用比例归一化方法（方程（27）），使得每个节点上的资源分配均满足问题（17）中的总带宽、CPU和功率约束。本模块实现该投影操作，并利用已有的信道模型和任务模型，对直连计算、中继计算及无人机辅助车车通信三类任务进行联合评价。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Iterable, Mapping

import numpy as np

from dm_jcr.channel import achievable_rate_bps
from dm_jcr.task_model import (
    DEFAULT_CPU_ENERGY_COEFFICIENT,
    ComputationTask,
    DirectLinkResources,
    DirectTaskEvaluation,
    evaluate_direct_task,
)

_EPSILON = 1.0e-12


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


def _clean_identifier(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value.strip()


@dataclass(frozen=True)
class NodeResourceCapacity:
    """一个时隙内某无人机或路侧单元可分配的资源总量。"""

    node_id: str
    total_bandwidth_hz: float
    total_cpu_frequency_hz: float
    total_transmit_power_w: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _clean_identifier("node_id", self.node_id))
        object.__setattr__(
            self,
            "total_bandwidth_hz",
            _finite_positive("total_bandwidth_hz", self.total_bandwidth_hz),
        )
        object.__setattr__(
            self,
            "total_cpu_frequency_hz",
            _finite_positive(
                "total_cpu_frequency_hz", self.total_cpu_frequency_hz
            ),
        )
        object.__setattr__(
            self,
            "total_transmit_power_w",
            _finite_positive(
                "total_transmit_power_w", self.total_transmit_power_w
            ),
        )


@dataclass(frozen=True)
class DirectTaskContext:
    """一个直连任务的固定环境与映射信息。

    ``node_id`` 是公式（10）至（11）的输出。车辆发射功率和信道状态属于环境
    输入；带宽、边缘 CPU 频率和边缘节点发射功率由分配策略生成。
    """

    task_id: str
    node_id: str
    task: ComputationTask
    uplink_channel_gain: float
    downlink_channel_gain: float
    vehicle_transmit_power_w: float
    noise_psd_w_hz: float
    uplink_interference_power_w: float = 0.0
    downlink_interference_power_w: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _clean_identifier("task_id", self.task_id))
        object.__setattr__(self, "node_id", _clean_identifier("node_id", self.node_id))
        if not isinstance(self.task, ComputationTask):
            raise TypeError("task 必须是 ComputationTask")
        for name in (
            "uplink_channel_gain",
            "downlink_channel_gain",
            "vehicle_transmit_power_w",
            "uplink_interference_power_w",
            "downlink_interference_power_w",
        ):
            object.__setattr__(
                self, name, _finite_non_negative(name, getattr(self, name))
            )
        object.__setattr__(
            self,
            "noise_psd_w_hz",
            _finite_positive("noise_psd_w_hz", self.noise_psd_w_hz),
        )
        if self.uplink_channel_gain == 0.0:
            raise ValueError("uplink_channel_gain 必须大于 0")
        if self.downlink_channel_gain == 0.0:
            raise ValueError("downlink_channel_gain 必须大于 0")


@dataclass(frozen=True)
class RawTaskAllocation:
    """优化器或生成器输出的无约束非负分数。"""

    task_id: str
    node_id: str
    bandwidth_score: float
    cpu_score: float
    power_score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _clean_identifier("task_id", self.task_id))
        object.__setattr__(self, "node_id", _clean_identifier("node_id", self.node_id))
        for name in ("bandwidth_score", "cpu_score", "power_score"):
            object.__setattr__(
                self, name, _finite_non_negative(name, getattr(self, name))
            )


@dataclass(frozen=True)
class FeasibleTaskAllocation:
    """满足节点资源总量约束的投影后资源分配。"""

    task_id: str
    node_id: str
    bandwidth_hz: float
    cpu_frequency_hz: float
    node_transmit_power_w: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _clean_identifier("task_id", self.task_id))
        object.__setattr__(self, "node_id", _clean_identifier("node_id", self.node_id))
        object.__setattr__(
            self,
            "bandwidth_hz",
            _finite_positive("bandwidth_hz", self.bandwidth_hz),
        )
        object.__setattr__(
            self,
            "cpu_frequency_hz",
            _finite_positive("cpu_frequency_hz", self.cpu_frequency_hz),
        )
        object.__setattr__(
            self,
            "node_transmit_power_w",
            _finite_non_negative(
                "node_transmit_power_w", self.node_transmit_power_w
            ),
        )


@dataclass(frozen=True)
class ObjectiveWeights:
    """公式（17）中的权重 ``rho`` 和 ``xi``。"""

    latency: float = 0.5
    energy: float = 0.5

    def __post_init__(self) -> None:
        latency = _finite_non_negative("latency", self.latency)
        energy = _finite_non_negative("energy", self.energy)
        if not isclose(latency + energy, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("时延权重与能耗权重之和必须为 1")
        object.__setattr__(self, "latency", latency)
        object.__setattr__(self, "energy", energy)


@dataclass(frozen=True)
class ObjectiveNormalization:
    """加权目标函数使用的显式归一化常数。

    论文说明时延和能耗需要归一化，但未给出具体公式。本实现默认以各任务时限
    作为时延参考值，并由调用方提供能耗参考值。复现数据集级归一化方案时也可
    提供统一的时延参考值。
    """

    energy_reference_j: float
    latency_reference_s: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "energy_reference_j",
            _finite_positive("energy_reference_j", self.energy_reference_j),
        )
        if self.latency_reference_s is not None:
            object.__setattr__(
                self,
                "latency_reference_s",
                _finite_positive(
                    "latency_reference_s", self.latency_reference_s
                ),
            )


@dataclass(frozen=True)
class TaskAllocationEvaluation:
    """一个任务在资源投影后的评价结果。"""

    task_id: str
    node_id: str
    allocation: FeasibleTaskAllocation
    uplink_rate_bps: float
    downlink_rate_bps: float
    metrics: DirectTaskEvaluation
    normalized_latency: float
    normalized_energy: float


@dataclass(frozen=True)
class JointAllocationEvaluation:
    """一个时隙的公式（17）目标值与可行性报告。"""

    tasks: tuple[TaskAllocationEvaluation, ...]
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


def proportional_normalize(
    scores: Iterable[float],
    total_resource: float,
) -> np.ndarray:
    """应用公式（27）的比例归一化。

    对非零分数向量 ``gamma``，返回
    ``total_resource / sum(gamma) * gamma``。全零向量不含比例信息，因此采用
    对称且确定的平均分配；负数、无穷值及空输入均会被拒绝。
    """

    total = _finite_positive("total_resource", total_resource)
    vector = np.asarray(tuple(scores), dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("scores 必须是非空一维序列")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError("scores 必须只包含非负有限数值")

    score_sum = float(vector.sum())
    if score_sum <= _EPSILON:
        return np.full(vector.shape, total / vector.size, dtype=np.float64)
    return vector * (total / score_sum)


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


def project_resource_strategy(
    raw_allocations: Iterable[RawTaskAllocation],
    capacities: Iterable[NodeResourceCapacity],
) -> tuple[FeasibleTaskAllocation, ...]:
    """将原始策略投影到公式（17）的全部资源约束上。

    各节点的带宽、CPU 和功率分别独立投影，并严格按照公式（27）保留相对分数。
    """

    raw = tuple(raw_allocations)
    if not raw:
        raise ValueError("至少需要一个原始分配")
    capacity_by_node = _index_capacities(capacities)

    task_ids = [item.task_id for item in raw]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_id 必须唯一")

    result: list[FeasibleTaskAllocation | None] = [None] * len(raw)
    indices_by_node: dict[str, list[int]] = {}
    for index, item in enumerate(raw):
        if item.node_id not in capacity_by_node:
            raise ValueError(f"缺少节点 {item.node_id!r} 的容量配置")
        indices_by_node.setdefault(item.node_id, []).append(index)

    for node_id, indices in indices_by_node.items():
        capacity = capacity_by_node[node_id]
        bandwidth = proportional_normalize(
            (raw[i].bandwidth_score for i in indices),
            capacity.total_bandwidth_hz,
        )
        cpu = proportional_normalize(
            (raw[i].cpu_score for i in indices),
            capacity.total_cpu_frequency_hz,
        )
        power = proportional_normalize(
            (raw[i].power_score for i in indices),
            capacity.total_transmit_power_w,
        )

        for local_index, raw_index in enumerate(indices):
            item = raw[raw_index]
            result[raw_index] = FeasibleTaskAllocation(
                task_id=item.task_id,
                node_id=item.node_id,
                bandwidth_hz=float(bandwidth[local_index]),
                cpu_frequency_hz=float(cpu[local_index]),
                node_transmit_power_w=float(power[local_index]),
            )

    return tuple(item for item in result if item is not None)


def verify_resource_constraints(
    allocations: Iterable[FeasibleTaskAllocation],
    capacities: Iterable[NodeResourceCapacity],
    *,
    tolerance: float = 1.0e-8,
) -> bool:
    """返回每个节点的三类资源总量是否均满足公式（17）。"""

    tolerance = _finite_non_negative("tolerance", tolerance)
    capacity_by_node = _index_capacities(capacities)
    totals = {
        node_id: np.zeros(3, dtype=np.float64) for node_id in capacity_by_node
    }
    for item in allocations:
        if item.node_id not in capacity_by_node:
            return False
        totals[item.node_id] += np.array(
            [
                item.bandwidth_hz,
                item.cpu_frequency_hz,
                item.node_transmit_power_w,
            ],
            dtype=np.float64,
        )

    for node_id, values in totals.items():
        capacity = capacity_by_node[node_id]
        limits = np.array(
            [
                capacity.total_bandwidth_hz,
                capacity.total_cpu_frequency_hz,
                capacity.total_transmit_power_w,
            ],
            dtype=np.float64,
        )
        if np.any(values < -tolerance) or np.any(values - limits > tolerance):
            return False
    return True


def evaluate_direct_resource_strategy(
    contexts: Iterable[DirectTaskContext],
    allocations: Iterable[FeasibleTaskAllocation],
    capacities: Iterable[NodeResourceCapacity],
    normalization: ObjectiveNormalization,
    weights: ObjectiveWeights = ObjectiveWeights(),
    *,
    energy_coefficient: float = DEFAULT_CPU_ENERGY_COEFFICIENT,
) -> JointAllocationEvaluation:
    """按照公式（17）评价投影后的直连策略。

    本函数通过公式（8）计算速率，通过公式（12）至（13）计算任务时延与能耗，
    再计算归一化指标和加权目标 ``J``。不可行性不会被惩罚项掩盖；资源约束和
    时限约束分别报告，与原约束问题保持一致。
    """

    context_items = tuple(contexts)
    allocation_items = tuple(allocations)
    if not context_items:
        raise ValueError("至少需要一个任务上下文")

    context_by_task = {item.task_id: item for item in context_items}
    allocation_by_task = {item.task_id: item for item in allocation_items}
    if len(context_by_task) != len(context_items):
        raise ValueError("上下文中的 task_id 必须唯一")
    if len(allocation_by_task) != len(allocation_items):
        raise ValueError("分配中的 task_id 必须唯一")
    if context_by_task.keys() != allocation_by_task.keys():
        raise ValueError("上下文与分配必须包含相同任务")

    task_results: list[TaskAllocationEvaluation] = []
    for context in context_items:
        allocation = allocation_by_task[context.task_id]
        if allocation.node_id != context.node_id:
            raise ValueError(
                f"任务 {context.task_id!r} 映射至 {context.node_id!r}，"
                f"而不是 {allocation.node_id!r}"
            )
        if allocation.bandwidth_hz <= 0.0:
            raise ValueError("投影后的带宽必须大于 0")
        if allocation.cpu_frequency_hz <= 0.0:
            raise ValueError("投影后的 CPU 频率必须大于 0")
        if allocation.node_transmit_power_w < 0.0:
            raise ValueError("投影后的节点发射功率不能为负数")

        uplink_rate = achievable_rate_bps(
            bandwidth_hz=allocation.bandwidth_hz,
            transmit_power_w=context.vehicle_transmit_power_w,
            channel_gain=context.uplink_channel_gain,
            noise_psd_w_hz=context.noise_psd_w_hz,
            interference_power_w=context.uplink_interference_power_w,
        )
        downlink_rate = achievable_rate_bps(
            bandwidth_hz=allocation.bandwidth_hz,
            transmit_power_w=allocation.node_transmit_power_w,
            channel_gain=context.downlink_channel_gain,
            noise_psd_w_hz=context.noise_psd_w_hz,
            interference_power_w=context.downlink_interference_power_w,
        )
        if uplink_rate <= 0.0 or downlink_rate <= 0.0:
            raise ValueError("已分配资源产生了零通信速率")

        metrics = evaluate_direct_task(
            task=context.task,
            resources=DirectLinkResources(
                uplink_rate_bps=uplink_rate,
                downlink_rate_bps=downlink_rate,
                cpu_frequency_hz=allocation.cpu_frequency_hz,
                vehicle_tx_power_w=context.vehicle_transmit_power_w,
                node_tx_power_w=allocation.node_transmit_power_w,
            ),
            energy_coefficient=energy_coefficient,
        )
        latency_reference = (
            normalization.latency_reference_s
            if normalization.latency_reference_s is not None
            else context.task.max_latency_s
        )
        task_results.append(
            TaskAllocationEvaluation(
                task_id=context.task_id,
                node_id=context.node_id,
                allocation=allocation,
                uplink_rate_bps=float(uplink_rate),
                downlink_rate_bps=float(downlink_rate),
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
    mean_energy = float(
        np.mean([item.normalized_energy for item in task_results])
    )
    objective = weights.latency * mean_latency + weights.energy * mean_energy

    return JointAllocationEvaluation(
        tasks=tuple(task_results),
        mean_normalized_latency=mean_latency,
        mean_normalized_energy=mean_energy,
        weighted_objective=float(objective),
        resource_constraints_satisfied=verify_resource_constraints(
            allocation_items, capacities
        ),
        deadline_constraints_satisfied=all(
            item.metrics.meets_deadline for item in task_results
        ),
    )


def allocations_by_task(
    allocations: Iterable[FeasibleTaskAllocation],
) -> Mapping[str, FeasibleTaskAllocation]:
    """构建只读式任务索引，并拒绝重复的任务标识。"""

    result: dict[str, FeasibleTaskAllocation] = {}
    for allocation in allocations:
        if allocation.task_id in result:
            raise ValueError(f"task_id {allocation.task_id!r} 重复")
        result[allocation.task_id] = allocation
    return result


__all__ = [
    "DirectTaskContext",
    "FeasibleTaskAllocation",
    "JointAllocationEvaluation",
    "NodeResourceCapacity",
    "ObjectiveNormalization",
    "ObjectiveWeights",
    "RawTaskAllocation",
    "TaskAllocationEvaluation",
    "allocations_by_task",
    "evaluate_direct_resource_strategy",
    "project_resource_strategy",
    "proportional_normalize",
    "verify_resource_constraints",
]

# ---------------------------------------------------------------------------
# 完整支持公式（17）：直连计算、中继计算，以及公式（14）至（15）的
# 无人机辅助车车通信任务。
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class V2VRelayTaskContext:
    """一个无人机辅助车车通信任务的固定环境。

    该任务类型对应公式（17）中的 ``m_{v,v*,k}=1``。数据通过一架中继无人机
    从一辆车传至另一辆车，不执行边缘节点计算。
    """

    task_id: str
    source_vehicle_id: str
    target_vehicle_id: str
    relay_uav_id: str
    data_bits: float
    max_latency_s: float
    vehicle_to_uav_channel_gain: float
    uav_to_vehicle_channel_gain: float
    source_vehicle_transmit_power_w: float
    noise_psd_w_hz: float
    forwarding_cycles_per_bit: float
    vehicle_to_uav_interference_power_w: float = 0.0
    uav_to_vehicle_interference_power_w: float = 0.0

    def __post_init__(self) -> None:
        for name in ("task_id", "source_vehicle_id", "target_vehicle_id", "relay_uav_id"):
            object.__setattr__(self, name, _clean_identifier(name, getattr(self, name)))
        if self.source_vehicle_id == self.target_vehicle_id:
            raise ValueError("source_vehicle_id 与 target_vehicle_id 必须不同")
        for name in (
            "data_bits", "max_latency_s", "vehicle_to_uav_channel_gain",
            "uav_to_vehicle_channel_gain", "noise_psd_w_hz",
        ):
            object.__setattr__(self, name, _finite_positive(name, getattr(self, name)))
        for name in (
            "source_vehicle_transmit_power_w", "forwarding_cycles_per_bit",
            "vehicle_to_uav_interference_power_w", "uav_to_vehicle_interference_power_w",
        ):
            object.__setattr__(self, name, _finite_non_negative(name, getattr(self, name)))


@dataclass(frozen=True)
class RawV2VRelayAllocation:
    """公式（14）至（15）使用的无约束分数。"""

    task_id: str
    relay_uav_id: str
    vehicle_to_uav_bandwidth_score: float
    uav_to_vehicle_bandwidth_score: float
    relay_cpu_score: float
    relay_power_score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _clean_identifier("task_id", self.task_id))
        object.__setattr__(self, "relay_uav_id", _clean_identifier("relay_uav_id", self.relay_uav_id))
        for name in (
            "vehicle_to_uav_bandwidth_score", "uav_to_vehicle_bandwidth_score",
            "relay_cpu_score", "relay_power_score",
        ):
            object.__setattr__(self, name, _finite_non_negative(name, getattr(self, name)))


@dataclass(frozen=True)
class FeasibleV2VRelayAllocation:
    """无人机辅助车车任务投影后的物理资源。"""

    task_id: str
    relay_uav_id: str
    vehicle_to_uav_bandwidth_hz: float
    uav_to_vehicle_bandwidth_hz: float
    relay_cpu_frequency_hz: float
    relay_transmit_power_w: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _clean_identifier("task_id", self.task_id))
        object.__setattr__(self, "relay_uav_id", _clean_identifier("relay_uav_id", self.relay_uav_id))
        for name in (
            "vehicle_to_uav_bandwidth_hz", "uav_to_vehicle_bandwidth_hz",
            "relay_cpu_frequency_hz",
        ):
            object.__setattr__(self, name, _finite_positive(name, getattr(self, name)))
        object.__setattr__(self, "relay_transmit_power_w", _finite_non_negative(
            "relay_transmit_power_w", self.relay_transmit_power_w
        ))

    @property
    def relay_bandwidth_total_hz(self) -> float:
        return float(self.vehicle_to_uav_bandwidth_hz + self.uav_to_vehicle_bandwidth_hz)


@dataclass(frozen=True)
class V2VRelayTaskMetrics:
    vehicle_to_uav_time_s: float
    forwarding_time_s: float
    uav_to_vehicle_time_s: float
    total_latency_s: float
    vehicle_to_uav_energy_j: float
    forwarding_energy_j: float
    uav_to_vehicle_energy_j: float
    total_energy_j: float
    deadline_satisfied: bool


@dataclass(frozen=True)
class V2VRelayTaskEvaluation:
    task_id: str
    relay_uav_id: str
    allocation: FeasibleV2VRelayAllocation
    vehicle_to_uav_rate_bps: float
    uav_to_vehicle_rate_bps: float
    metrics: V2VRelayTaskMetrics
    normalized_latency: float
    normalized_energy: float


@dataclass(frozen=True)
class Equation17ProjectedStrategy:
    """包含公式（17）全部任务类型的一项投影策略。"""

    direct_allocations: tuple[FeasibleTaskAllocation, ...]
    relay_computation_allocations: tuple[object, ...]
    v2v_relay_allocations: tuple[FeasibleV2VRelayAllocation, ...]

    def __post_init__(self) -> None:
        ids = [x.task_id for x in self.direct_allocations]
        ids += [x.task_id for x in self.relay_computation_allocations]
        ids += [x.task_id for x in self.v2v_relay_allocations]
        if not ids:
            raise ValueError("至少需要一个分配")
        if len(ids) != len(set(ids)):
            raise ValueError("所有任务类型中的 task_id 必须唯一")


@dataclass(frozen=True)
class Equation17Evaluation:
    direct_tasks: tuple[TaskAllocationEvaluation, ...]
    relay_computation_tasks: tuple[object, ...]
    v2v_relay_tasks: tuple[V2VRelayTaskEvaluation, ...]
    mean_normalized_latency: float
    mean_normalized_energy: float
    weighted_objective: float
    resource_constraints_satisfied: bool
    deadline_constraints_satisfied: bool

    @property
    def feasible(self) -> bool:
        return self.resource_constraints_satisfied and self.deadline_constraints_satisfied

    @property
    def task_count(self) -> int:
        return len(self.direct_tasks) + len(self.relay_computation_tasks) + len(self.v2v_relay_tasks)


def evaluate_v2v_relay_task(
    context: V2VRelayTaskContext,
    allocation: FeasibleV2VRelayAllocation,
    *,
    energy_coefficient: float = DEFAULT_CPU_ENERGY_COEFFICIENT,
) -> V2VRelayTaskMetrics:
    """为一个车车中继任务评价公式（14）和（15）。"""

    if context.task_id != allocation.task_id or context.relay_uav_id != allocation.relay_uav_id:
        raise ValueError("车车任务上下文与分配映射不匹配")
    energy_coefficient = _finite_non_negative("energy_coefficient", energy_coefficient)

    r_vu = achievable_rate_bps(
        bandwidth_hz=allocation.vehicle_to_uav_bandwidth_hz,
        transmit_power_w=context.source_vehicle_transmit_power_w,
        channel_gain=context.vehicle_to_uav_channel_gain,
        noise_psd_w_hz=context.noise_psd_w_hz,
        interference_power_w=context.vehicle_to_uav_interference_power_w,
    )
    r_uv = achievable_rate_bps(
        bandwidth_hz=allocation.uav_to_vehicle_bandwidth_hz,
        transmit_power_w=allocation.relay_transmit_power_w,
        channel_gain=context.uav_to_vehicle_channel_gain,
        noise_psd_w_hz=context.noise_psd_w_hz,
        interference_power_w=context.uav_to_vehicle_interference_power_w,
    )
    if r_vu <= _EPSILON or r_uv <= _EPSILON:
        raise ValueError("车车中继速率必须大于零")

    t_vu = context.data_bits / r_vu
    forwarding_cycles = context.forwarding_cycles_per_bit * context.data_bits
    t_forward = forwarding_cycles / allocation.relay_cpu_frequency_hz
    t_uv = context.data_bits / r_uv
    total_t = t_vu + t_forward + t_uv

    e_vu = context.source_vehicle_transmit_power_w * t_vu
    e_forward = energy_coefficient * allocation.relay_cpu_frequency_hz**2 * forwarding_cycles
    e_uv = allocation.relay_transmit_power_w * t_uv
    total_e = e_vu + e_forward + e_uv
    return V2VRelayTaskMetrics(
        vehicle_to_uav_time_s=float(t_vu),
        forwarding_time_s=float(t_forward),
        uav_to_vehicle_time_s=float(t_uv),
        total_latency_s=float(total_t),
        vehicle_to_uav_energy_j=float(e_vu),
        forwarding_energy_j=float(e_forward),
        uav_to_vehicle_energy_j=float(e_uv),
        total_energy_j=float(total_e),
        deadline_satisfied=bool(total_t <= context.max_latency_s),
    )


def project_equation17_strategy(
    direct_raw_allocations: Iterable[RawTaskAllocation],
    relay_computation_raw_allocations: Iterable[object],
    v2v_relay_raw_allocations: Iterable[RawV2VRelayAllocation],
    capacities: Iterable[NodeResourceCapacity],
) -> Equation17ProjectedStrategy:
    """将公式（17）的所有任务类型投影到共享节点预算上。

    中继计算分配对象为 :mod:`dm_jcr.relay_resource_allocation` 中的
    ``RawRelayTaskAllocation``。局部导入既避免循环依赖，也保留直连接口的向后兼容性。
    """

    from dm_jcr.relay_resource_allocation import FeasibleRelayTaskAllocation, RawRelayTaskAllocation

    direct = tuple(direct_raw_allocations)
    relay = tuple(relay_computation_raw_allocations)
    v2v = tuple(v2v_relay_raw_allocations)
    if any(not isinstance(x, RawRelayTaskAllocation) for x in relay):
        raise TypeError("relay_computation_raw_allocations 必须包含 RawRelayTaskAllocation")
    all_ids = [x.task_id for x in direct] + [x.task_id for x in relay] + [x.task_id for x in v2v]
    if not all_ids:
        raise ValueError("至少需要一个原始分配")
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("所有任务类型中的 task_id 必须唯一")

    capacity_by_node = _index_capacities(capacities)
    direct_values = [dict() for _ in direct]
    relay_values = [dict() for _ in relay]
    v2v_values = [dict() for _ in v2v]
    slots = {"bandwidth": {}, "cpu": {}, "power": {}}

    def add(kind: str, node: str, owner: str, index: int, field: str, score: float) -> None:
        if node not in capacity_by_node:
            raise ValueError(f"缺少节点 {node!r} 的容量配置")
        slots[kind].setdefault(node, []).append((owner, index, field, score))

    for i, x in enumerate(direct):
        add("bandwidth", x.node_id, "d", i, "bandwidth_hz", x.bandwidth_score)
        add("cpu", x.node_id, "d", i, "cpu_frequency_hz", x.cpu_score)
        add("power", x.node_id, "d", i, "node_transmit_power_w", x.power_score)
    for i, x in enumerate(relay):
        add("bandwidth", x.relay_uav_id, "r", i, "vehicle_to_uav_bandwidth_hz", x.vehicle_to_uav_bandwidth_score)
        add("bandwidth", x.relay_uav_id, "r", i, "uav_to_node_bandwidth_hz", x.uav_to_node_bandwidth_score)
        add("bandwidth", x.relay_uav_id, "r", i, "uav_to_vehicle_bandwidth_hz", x.uav_to_vehicle_bandwidth_score)
        add("bandwidth", x.compute_node_id, "r", i, "node_to_uav_bandwidth_hz", x.node_to_uav_bandwidth_score)
        add("cpu", x.relay_uav_id, "r", i, "relay_cpu_frequency_hz", x.relay_cpu_score)
        add("cpu", x.compute_node_id, "r", i, "compute_cpu_frequency_hz", x.compute_cpu_score)
        add("power", x.relay_uav_id, "r", i, "relay_transmit_power_w", x.relay_power_score)
        add("power", x.compute_node_id, "r", i, "compute_node_transmit_power_w", x.compute_node_power_score)
    for i, x in enumerate(v2v):
        add("bandwidth", x.relay_uav_id, "v", i, "vehicle_to_uav_bandwidth_hz", x.vehicle_to_uav_bandwidth_score)
        add("bandwidth", x.relay_uav_id, "v", i, "uav_to_vehicle_bandwidth_hz", x.uav_to_vehicle_bandwidth_score)
        add("cpu", x.relay_uav_id, "v", i, "relay_cpu_frequency_hz", x.relay_cpu_score)
        add("power", x.relay_uav_id, "v", i, "relay_transmit_power_w", x.relay_power_score)

    attr = {"bandwidth": "total_bandwidth_hz", "cpu": "total_cpu_frequency_hz", "power": "total_transmit_power_w"}
    targets = {"d": direct_values, "r": relay_values, "v": v2v_values}
    for kind, by_node in slots.items():
        for node, node_slots in by_node.items():
            values = proportional_normalize((s[3] for s in node_slots), getattr(capacity_by_node[node], attr[kind]))
            for (owner, index, field, _), value in zip(node_slots, values, strict=True):
                targets[owner][index][field] = float(value)

    direct_out = tuple(FeasibleTaskAllocation(x.task_id, x.node_id, **direct_values[i]) for i, x in enumerate(direct))
    relay_out = tuple(FeasibleRelayTaskAllocation(
        task_id=x.task_id, relay_uav_id=x.relay_uav_id, compute_node_id=x.compute_node_id, **relay_values[i]
    ) for i, x in enumerate(relay))
    v2v_out = tuple(FeasibleV2VRelayAllocation(
        task_id=x.task_id, relay_uav_id=x.relay_uav_id, **v2v_values[i]
    ) for i, x in enumerate(v2v))
    return Equation17ProjectedStrategy(direct_out, relay_out, v2v_out)


def equation17_resource_totals(strategy: Equation17ProjectedStrategy) -> dict[str, tuple[float, float, float]]:
    totals: dict[str, np.ndarray] = {}
    def add(node: str, b: float, f: float, p: float) -> None:
        totals.setdefault(node, np.zeros(3, dtype=np.float64))
        totals[node] += (b, f, p)
    for x in strategy.direct_allocations:
        add(x.node_id, x.bandwidth_hz, x.cpu_frequency_hz, x.node_transmit_power_w)
    for x in strategy.relay_computation_allocations:
        add(x.relay_uav_id, x.relay_bandwidth_total_hz, x.relay_cpu_frequency_hz, x.relay_transmit_power_w)
        add(x.compute_node_id, x.compute_node_bandwidth_total_hz, x.compute_cpu_frequency_hz, x.compute_node_transmit_power_w)
    for x in strategy.v2v_relay_allocations:
        add(x.relay_uav_id, x.relay_bandwidth_total_hz, x.relay_cpu_frequency_hz, x.relay_transmit_power_w)
    return {k: tuple(float(v) for v in values) for k, values in totals.items()}


def verify_equation17_resource_constraints(
    strategy: Equation17ProjectedStrategy,
    capacities: Iterable[NodeResourceCapacity],
    *, tolerance: float = 1e-8,
    relative_tolerance: float = 1e-12,
) -> bool:
    tolerance = _finite_non_negative("tolerance", tolerance)
    relative_tolerance = _finite_non_negative("relative_tolerance", relative_tolerance)
    cap = _index_capacities(capacities)
    for node, values in equation17_resource_totals(strategy).items():
        if node not in cap:
            return False
        limits = (cap[node].total_bandwidth_hz, cap[node].total_cpu_frequency_hz, cap[node].total_transmit_power_w)
        if any(
            value < -tolerance
            or value - limit > max(tolerance, relative_tolerance * limit)
            for value, limit in zip(values, limits)
        ):
            return False
    return True


def evaluate_equation17_strategy(
    direct_contexts: Iterable[DirectTaskContext],
    relay_computation_contexts: Iterable[object],
    v2v_relay_contexts: Iterable[V2VRelayTaskContext],
    strategy: Equation17ProjectedStrategy,
    capacities: Iterable[NodeResourceCapacity],
    normalization: ObjectiveNormalization,
    weights: ObjectiveWeights = ObjectiveWeights(),
    *, energy_coefficient: float = DEFAULT_CPU_ENERGY_COEFFICIENT,
) -> Equation17Evaluation:
    """评价公式（17）中出现的每一项任务。"""

    from dm_jcr.relay_resource_allocation import RelayTaskContext, evaluate_relay_resource_strategy

    dctx, rctx, vctx = tuple(direct_contexts), tuple(relay_computation_contexts), tuple(v2v_relay_contexts)
    if any(not isinstance(x, RelayTaskContext) for x in rctx):
        raise TypeError("relay_computation_contexts 必须包含 RelayTaskContext")
    context_ids = [x.task_id for x in dctx] + [x.task_id for x in rctx] + [x.task_id for x in vctx]
    if not context_ids:
        raise ValueError("至少需要一个任务上下文")
    if len(context_ids) != len(set(context_ids)):
        raise ValueError("所有上下文类型中的 task_id 必须唯一")
    capacities = tuple(capacities)

    direct_results = ()
    if dctx:
        direct_results = evaluate_direct_resource_strategy(
            dctx, strategy.direct_allocations, capacities, normalization, weights,
            energy_coefficient=energy_coefficient,
        ).tasks
    elif strategy.direct_allocations:
        raise ValueError("提供了直连分配，但未提供对应上下文")

    relay_results = ()
    if rctx:
        relay_results = evaluate_relay_resource_strategy(
            rctx, strategy.relay_computation_allocations, capacities, normalization, weights,
            energy_coefficient=energy_coefficient,
        ).tasks
    elif strategy.relay_computation_allocations:
        raise ValueError("提供了中继计算分配，但未提供对应上下文")

    vmap = {x.task_id: x for x in strategy.v2v_relay_allocations}
    if len(vmap) != len(strategy.v2v_relay_allocations) or set(vmap) != {x.task_id for x in vctx}:
        if vctx or strategy.v2v_relay_allocations:
            raise ValueError("车车任务上下文与分配必须包含相同任务")
    v2v_results = []
    for context in vctx:
        allocation = vmap[context.task_id]
        metrics = evaluate_v2v_relay_task(context, allocation, energy_coefficient=energy_coefficient)
        r_vu = achievable_rate_bps(
            allocation.vehicle_to_uav_bandwidth_hz, context.source_vehicle_transmit_power_w,
            context.vehicle_to_uav_channel_gain, context.noise_psd_w_hz,
            context.vehicle_to_uav_interference_power_w,
        )
        r_uv = achievable_rate_bps(
            allocation.uav_to_vehicle_bandwidth_hz, allocation.relay_transmit_power_w,
            context.uav_to_vehicle_channel_gain, context.noise_psd_w_hz,
            context.uav_to_vehicle_interference_power_w,
        )
        lat_ref = normalization.latency_reference_s or context.max_latency_s
        v2v_results.append(V2VRelayTaskEvaluation(
            task_id=context.task_id, relay_uav_id=context.relay_uav_id, allocation=allocation,
            vehicle_to_uav_rate_bps=float(r_vu), uav_to_vehicle_rate_bps=float(r_uv), metrics=metrics,
            normalized_latency=float(metrics.total_latency_s / lat_ref),
            normalized_energy=float(metrics.total_energy_j / normalization.energy_reference_j),
        ))

    latencies = [x.normalized_latency for x in direct_results]
    latencies += [x.normalized_latency for x in relay_results]
    latencies += [x.normalized_latency for x in v2v_results]
    energies = [x.normalized_energy for x in direct_results]
    energies += [x.normalized_energy for x in relay_results]
    energies += [x.normalized_energy for x in v2v_results]
    mean_t, mean_e = float(np.mean(latencies)), float(np.mean(energies))
    return Equation17Evaluation(
        direct_tasks=tuple(direct_results),
        relay_computation_tasks=tuple(relay_results),
        v2v_relay_tasks=tuple(v2v_results),
        mean_normalized_latency=mean_t,
        mean_normalized_energy=mean_e,
        weighted_objective=float(weights.latency * mean_t + weights.energy * mean_e),
        resource_constraints_satisfied=verify_equation17_resource_constraints(strategy, capacities),
        deadline_constraints_satisfied=(
            all(x.metrics.meets_deadline for x in direct_results)
            and all(x.metrics.deadline_satisfied for x in relay_results)
            and all(x.metrics.deadline_satisfied for x in v2v_results)
        ),
    )


__all__ += [
    "Equation17Evaluation", "Equation17ProjectedStrategy",
    "FeasibleV2VRelayAllocation", "RawV2VRelayAllocation",
    "V2VRelayTaskContext", "V2VRelayTaskEvaluation", "V2VRelayTaskMetrics",
    "equation17_resource_totals", "evaluate_equation17_strategy",
    "evaluate_v2v_relay_task", "project_equation17_strategy",
    "verify_equation17_resource_constraints",
]
