"""在三类原始资源分配对象与固定尺寸策略张量之间进行转换。"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from dm_jcr.environment import TaskNodeMapping
from dm_jcr.relay_resource_allocation import RawRelayTaskAllocation
from dm_jcr.resource_allocation import RawTaskAllocation, RawV2VRelayAllocation


# 八个槽位在所有任务模式下保持相同语义。无关槽位补零。
STRATEGY_FIELDS: tuple[str, ...] = (
    "vehicle_to_node_or_uav_bandwidth_score",
    "uav_to_compute_node_bandwidth_score",
    "compute_node_to_uav_bandwidth_score",
    "uav_to_vehicle_bandwidth_score",
    "relay_or_direct_cpu_score",
    "compute_node_cpu_score",
    "relay_or_direct_power_score",
    "compute_node_power_score",
)


@dataclass(frozen=True)
class StrategyTensorSpec:
    """扩散模型低维输出策略的固定尺寸。"""

    max_tasks: int

    def __post_init__(self) -> None:
        if not isinstance(self.max_tasks, int) or self.max_tasks <= 0:
            raise ValueError("max_tasks 必须是正整数")

    @property
    def score_width(self) -> int:
        return len(STRATEGY_FIELDS)


def strategy_spec_from_config(config: Mapping[str, Any]) -> StrategyTensorSpec:
    """从统一配置读取扩散模型策略张量的任务上限。"""

    return StrategyTensorSpec(max_tasks=config["assumptions"]["tensor"]["max_tasks"])


@dataclass(frozen=True)
class EncodedRawStrategy:
    """补零后的原始策略分数，以及恢复对象所需的固定映射元数据。"""

    values: NDArray[np.float64]
    task_mask: NDArray[np.bool_]
    mappings: tuple[TaskNodeMapping, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        mask = np.asarray(self.task_mask, dtype=np.bool_)
        if values.ndim != 2 or values.shape[1] != len(STRATEGY_FIELDS):
            raise ValueError(
                f"values 必须具有形状 (max_tasks, {len(STRATEGY_FIELDS)})"
            )
        if mask.shape != (values.shape[0],):
            raise ValueError("task_mask 的长度必须等于 values 的第一维")
        if len(self.mappings) != int(mask.sum()):
            raise ValueError("mappings 数量必须等于 task_mask 中的有效任务数")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("策略张量必须只包含有限非负数")
        if np.any(values[~mask] != 0.0):
            raise ValueError("padding 行必须全部为 0")
        object.__setattr__(self, "values", values.copy())
        object.__setattr__(self, "task_mask", mask.copy())

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(mapping.task_id for mapping in self.mappings)

    def flat_vector(self) -> NDArray[np.float64]:
        """返回扩散模型编码器可使用的一维低维策略表示。"""

        return self.values.ravel().copy()


def _unique_allocations(
    direct: tuple[RawTaskAllocation, ...],
    relay: tuple[RawRelayTaskAllocation, ...],
    v2v: tuple[RawV2VRelayAllocation, ...],
) -> dict[str, RawTaskAllocation | RawRelayTaskAllocation | RawV2VRelayAllocation]:
    result: dict[
        str, RawTaskAllocation | RawRelayTaskAllocation | RawV2VRelayAllocation
    ] = {}
    for allocation in (*direct, *relay, *v2v):
        if allocation.task_id in result:
            raise ValueError(f"重复的 task_id：{allocation.task_id!r}")
        result[allocation.task_id] = allocation
    return result


def encode_raw_strategy(
    direct_allocations: Iterable[RawTaskAllocation],
    relay_allocations: Iterable[RawRelayTaskAllocation],
    v2v_allocations: Iterable[RawV2VRelayAllocation],
    mappings: Iterable[TaskNodeMapping],
    spec: StrategyTensorSpec,
) -> EncodedRawStrategy:
    """按环境任务映射顺序编码三类原始资源分配策略。"""

    direct = tuple(direct_allocations)
    relay = tuple(relay_allocations)
    v2v = tuple(v2v_allocations)
    mapping_items = tuple(mappings)
    if len(mapping_items) > spec.max_tasks:
        raise ValueError(
            f"任务数 {len(mapping_items)} 超过策略张量上限 {spec.max_tasks}"
        )
    mapping_ids = [mapping.task_id for mapping in mapping_items]
    if len(mapping_ids) != len(set(mapping_ids)):
        raise ValueError("mappings 中的 task_id 必须唯一")

    allocation_by_task = _unique_allocations(direct, relay, v2v)
    if set(mapping_ids) != set(allocation_by_task):
        raise ValueError("资源分配与任务映射必须包含相同的 task_id")

    values = np.zeros((spec.max_tasks, spec.score_width), dtype=np.float64)
    mask = np.zeros(spec.max_tasks, dtype=np.bool_)
    for index, mapping in enumerate(mapping_items):
        allocation = allocation_by_task[mapping.task_id]
        mask[index] = True
        if mapping.mode == "direct":
            if not isinstance(allocation, RawTaskAllocation):
                raise TypeError(f"任务 {mapping.task_id!r} 需要 RawTaskAllocation")
            if allocation.node_id != mapping.compute_node_id:
                raise ValueError(f"任务 {mapping.task_id!r} 的直接计算节点不匹配")
            values[index, (0, 4, 6)] = (
                allocation.bandwidth_score,
                allocation.cpu_score,
                allocation.power_score,
            )
        elif mapping.mode == "relay_computation":
            if not isinstance(allocation, RawRelayTaskAllocation):
                raise TypeError(
                    f"任务 {mapping.task_id!r} 需要 RawRelayTaskAllocation"
                )
            if (
                allocation.relay_uav_id != mapping.relay_uav_id
                or allocation.compute_node_id != mapping.compute_node_id
            ):
                raise ValueError(f"任务 {mapping.task_id!r} 的中继计算节点不匹配")
            values[index] = (
                allocation.vehicle_to_uav_bandwidth_score,
                allocation.uav_to_node_bandwidth_score,
                allocation.node_to_uav_bandwidth_score,
                allocation.uav_to_vehicle_bandwidth_score,
                allocation.relay_cpu_score,
                allocation.compute_cpu_score,
                allocation.relay_power_score,
                allocation.compute_node_power_score,
            )
        else:
            if not isinstance(allocation, RawV2VRelayAllocation):
                raise TypeError(f"任务 {mapping.task_id!r} 需要 RawV2VRelayAllocation")
            if allocation.relay_uav_id != mapping.relay_uav_id:
                raise ValueError(f"任务 {mapping.task_id!r} 的 V2V 中继节点不匹配")
            values[index, (0, 3, 4, 6)] = (
                allocation.vehicle_to_uav_bandwidth_score,
                allocation.uav_to_vehicle_bandwidth_score,
                allocation.relay_cpu_score,
                allocation.relay_power_score,
            )

    return EncodedRawStrategy(
        values=values,
        task_mask=mask,
        mappings=mapping_items,
    )


def decode_raw_strategy(
    encoded: EncodedRawStrategy,
) -> tuple[
    tuple[RawTaskAllocation, ...],
    tuple[RawRelayTaskAllocation, ...],
    tuple[RawV2VRelayAllocation, ...],
]:
    """按照编码时保存的任务映射恢复三类原始资源分配对象。"""

    direct: list[RawTaskAllocation] = []
    relay: list[RawRelayTaskAllocation] = []
    v2v: list[RawV2VRelayAllocation] = []
    for index, mapping in enumerate(encoded.mappings):
        row = encoded.values[index]
        if mapping.mode == "direct":
            direct.append(
                RawTaskAllocation(
                    task_id=mapping.task_id,
                    node_id=mapping.compute_node_id or "",
                    bandwidth_score=row[0],
                    cpu_score=row[4],
                    power_score=row[6],
                )
            )
        elif mapping.mode == "relay_computation":
            relay.append(
                RawRelayTaskAllocation(
                    task_id=mapping.task_id,
                    relay_uav_id=mapping.relay_uav_id or "",
                    compute_node_id=mapping.compute_node_id or "",
                    vehicle_to_uav_bandwidth_score=row[0],
                    uav_to_node_bandwidth_score=row[1],
                    node_to_uav_bandwidth_score=row[2],
                    uav_to_vehicle_bandwidth_score=row[3],
                    relay_cpu_score=row[4],
                    compute_cpu_score=row[5],
                    relay_power_score=row[6],
                    compute_node_power_score=row[7],
                )
            )
        else:
            v2v.append(
                RawV2VRelayAllocation(
                    task_id=mapping.task_id,
                    relay_uav_id=mapping.relay_uav_id or "",
                    vehicle_to_uav_bandwidth_score=row[0],
                    uav_to_vehicle_bandwidth_score=row[3],
                    relay_cpu_score=row[4],
                    relay_power_score=row[6],
                )
            )
    return tuple(direct), tuple(relay), tuple(v2v)


__all__ = [
    "EncodedRawStrategy",
    "STRATEGY_FIELDS",
    "StrategyTensorSpec",
    "decode_raw_strategy",
    "encode_raw_strategy",
    "strategy_spec_from_config",
]
