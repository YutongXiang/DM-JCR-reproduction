"""定义论文公式（16）的环境快照及其固定尺寸张量编码。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


NodeType = Literal["vehicle", "uav", "rsu"]
TaskMode = Literal["direct", "relay_computation", "v2v_relay"]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

NODE_TYPES: tuple[NodeType, ...] = ("vehicle", "uav", "rsu")
TASK_MODES: tuple[TaskMode, ...] = (
    "direct",
    "relay_computation",
    "v2v_relay",
)


def _identifier(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value.strip()


def _non_negative(name: str, value: float) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} 必须是有限非负数")
    return result


def _positive(name: str, value: float) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} 必须是有限正数")
    return result


def _vector3(name: str, value: ArrayLike) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 必须是包含三个有限数值的向量")
    return result.copy()


@dataclass(frozen=True)
class EnvironmentNode:
    """公式（16）中单个节点的位置、速度和剩余资源状态。"""

    node_id: str
    node_type: NodeType
    position_m: FloatArray
    velocity_mps: FloatArray
    remaining_bandwidth_hz: float = 0.0
    remaining_cpu_frequency_hz: float = 0.0
    remaining_transmit_power_w: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _identifier("node_id", self.node_id))
        if self.node_type not in NODE_TYPES:
            raise ValueError(f"不支持的 node_type：{self.node_type!r}")
        object.__setattr__(self, "position_m", _vector3("position_m", self.position_m))
        object.__setattr__(
            self, "velocity_mps", _vector3("velocity_mps", self.velocity_mps)
        )
        for name in (
            "remaining_bandwidth_hz",
            "remaining_cpu_frequency_hz",
            "remaining_transmit_power_w",
        ):
            object.__setattr__(self, name, _non_negative(name, getattr(self, name)))


@dataclass(frozen=True)
class EnvironmentLink:
    """公式（16）中一条有向链路的信道、遮挡和拓扑状态。"""

    source_node_id: str
    target_node_id: str
    channel_gain: float
    blocked: bool
    available: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_node_id",
            _identifier("source_node_id", self.source_node_id),
        )
        object.__setattr__(
            self,
            "target_node_id",
            _identifier("target_node_id", self.target_node_id),
        )
        if self.source_node_id == self.target_node_id:
            raise ValueError("链路两端节点必须不同")
        object.__setattr__(self, "channel_gain", _non_negative("channel_gain", self.channel_gain))
        object.__setattr__(self, "blocked", bool(self.blocked))
        object.__setattr__(self, "available", bool(self.available))
        if not self.available and self.channel_gain != 0.0:
            raise ValueError("不可用链路的 channel_gain 必须为 0")


@dataclass(frozen=True)
class EnvironmentTask:
    """公式（16）任务队列 Q 中的一项计算或转发任务。"""

    task_id: str
    source_vehicle_id: str
    input_bits: float
    cpu_cycles: float
    max_latency_s: float
    output_ratio: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _identifier("task_id", self.task_id))
        object.__setattr__(
            self,
            "source_vehicle_id",
            _identifier("source_vehicle_id", self.source_vehicle_id),
        )
        object.__setattr__(self, "input_bits", _non_negative("input_bits", self.input_bits))
        object.__setattr__(self, "cpu_cycles", _non_negative("cpu_cycles", self.cpu_cycles))
        object.__setattr__(
            self, "max_latency_s", _positive("max_latency_s", self.max_latency_s)
        )
        object.__setattr__(
            self, "output_ratio", _non_negative("output_ratio", self.output_ratio)
        )


@dataclass(frozen=True)
class TaskNodeMapping:
    """公式（16）中的任务—节点映射 W，同时标记任务执行模式。"""

    task_id: str
    mode: TaskMode
    compute_node_id: str | None = None
    relay_uav_id: str | None = None
    target_vehicle_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _identifier("task_id", self.task_id))
        if self.mode not in TASK_MODES:
            raise ValueError(f"不支持的任务模式：{self.mode!r}")
        for name in ("compute_node_id", "relay_uav_id", "target_vehicle_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _identifier(name, value))

        if self.mode == "direct":
            valid = (
                self.compute_node_id is not None
                and self.relay_uav_id is None
                and self.target_vehicle_id is None
            )
        elif self.mode == "relay_computation":
            valid = (
                self.compute_node_id is not None
                and self.relay_uav_id is not None
                and self.target_vehicle_id is None
            )
        else:
            valid = (
                self.compute_node_id is None
                and self.relay_uav_id is not None
                and self.target_vehicle_id is not None
            )
        if not valid:
            raise ValueError(f"任务 {self.task_id!r} 的节点字段与模式 {self.mode!r} 不匹配")


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """一个时隙内完整的公式（16）环境信息。"""

    time_slot: int
    nodes: tuple[EnvironmentNode, ...]
    links: tuple[EnvironmentLink, ...]
    tasks: tuple[EnvironmentTask, ...]
    mappings: tuple[TaskNodeMapping, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.time_slot, int) or self.time_slot < 0:
            raise ValueError("time_slot 必须是非负整数")
        if not self.nodes:
            raise ValueError("环境快照至少需要一个节点")
        node_by_id = {node.node_id: node for node in self.nodes}
        if len(node_by_id) != len(self.nodes):
            raise ValueError("node_id 必须唯一")

        task_by_id = {task.task_id: task for task in self.tasks}
        mapping_by_task = {mapping.task_id: mapping for mapping in self.mappings}
        if len(task_by_id) != len(self.tasks):
            raise ValueError("task_id 必须唯一")
        if len(mapping_by_task) != len(self.mappings):
            raise ValueError("每个任务只能有一条映射")
        if set(task_by_id) != set(mapping_by_task):
            raise ValueError("tasks 与 mappings 必须包含相同的 task_id")

        link_pairs: set[tuple[str, str]] = set()
        for link in self.links:
            pair = (link.source_node_id, link.target_node_id)
            if pair in link_pairs:
                raise ValueError(f"重复链路：{pair!r}")
            link_pairs.add(pair)
            if pair[0] not in node_by_id or pair[1] not in node_by_id:
                raise ValueError(f"链路引用了未知节点：{pair!r}")

        for task in self.tasks:
            source = node_by_id.get(task.source_vehicle_id)
            if source is None or source.node_type != "vehicle":
                raise ValueError(f"任务 {task.task_id!r} 的源节点必须是车辆")
            mapping = mapping_by_task[task.task_id]
            if mapping.compute_node_id is not None:
                compute = node_by_id.get(mapping.compute_node_id)
                if compute is None or compute.node_type not in ("uav", "rsu"):
                    raise ValueError(f"任务 {task.task_id!r} 的计算节点必须是 UAV 或 RSU")
            if mapping.relay_uav_id is not None:
                relay = node_by_id.get(mapping.relay_uav_id)
                if relay is None or relay.node_type != "uav":
                    raise ValueError(f"任务 {task.task_id!r} 的中继节点必须是 UAV")
            if mapping.target_vehicle_id is not None:
                target = node_by_id.get(mapping.target_vehicle_id)
                if target is None or target.node_type != "vehicle":
                    raise ValueError(f"任务 {task.task_id!r} 的目标节点必须是车辆")
                if target.node_id == task.source_vehicle_id:
                    raise ValueError("V2V 任务的源车辆和目标车辆必须不同")


@dataclass(frozen=True)
class EnvironmentTensorSpec:
    """固定尺寸及数值缩放规则；这些规则由统一配置文件提供。"""

    max_vehicles: int
    max_uavs: int
    max_rsus: int
    max_tasks: int
    position_scale_m: float
    speed_scale_mps: float
    bandwidth_scale_hz: float
    cpu_scale_hz: float
    power_scale_w: float
    task_bits_scale: float
    task_cycles_scale: float
    latency_scale_s: float
    channel_gain_scale: float

    def __post_init__(self) -> None:
        for name in ("max_vehicles", "max_uavs", "max_rsus", "max_tasks"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        for name in (
            "position_scale_m",
            "speed_scale_mps",
            "bandwidth_scale_hz",
            "cpu_scale_hz",
            "power_scale_w",
            "task_bits_scale",
            "task_cycles_scale",
            "latency_scale_s",
            "channel_gain_scale",
        ):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))

    @property
    def max_nodes(self) -> int:
        return self.max_vehicles + self.max_uavs + self.max_rsus


@dataclass(frozen=True)
class EncodedEnvironment:
    """公式（16）各分量的补零张量及其索引元数据。"""

    node_features: FloatArray
    channel_gains: FloatArray
    blockage: FloatArray
    topology: FloatArray
    task_features: FloatArray
    task_node_indices: IntArray
    node_mask: BoolArray
    task_mask: BoolArray
    node_ids: tuple[str, ...]
    task_ids: tuple[str, ...]

    def flat_vector(self) -> FloatArray:
        """返回可直接输入 MLP 或数据预处理器的一维浮点向量。"""

        max_nodes = self.node_features.shape[0]
        indices = self.task_node_indices.astype(np.float64)
        present = indices >= 0
        indices[present] /= max(1, max_nodes - 1)
        return np.concatenate(
            (
                self.node_features.ravel(),
                self.channel_gains.ravel(),
                self.blockage.ravel(),
                self.topology.ravel(),
                self.task_features.ravel(),
                indices.ravel(),
                self.node_mask.astype(np.float64),
                self.task_mask.astype(np.float64),
            )
        )


def tensor_spec_from_config(config: Mapping[str, Any]) -> EnvironmentTensorSpec:
    """从统一配置读取论文规模上限和复现采用的张量缩放因子。"""

    system = config["paper"]["system"]
    tensor = config["assumptions"]["tensor"]
    return EnvironmentTensorSpec(
        max_vehicles=system["max_vehicles"],
        max_uavs=system["max_uavs"],
        max_rsus=system["max_rsus"],
        max_tasks=tensor["max_tasks"],
        position_scale_m=tensor["position_scale_m"],
        speed_scale_mps=tensor["speed_scale_mps"],
        bandwidth_scale_hz=tensor["bandwidth_scale_hz"],
        cpu_scale_hz=tensor["cpu_scale_hz"],
        power_scale_w=tensor["power_scale_w"],
        task_bits_scale=tensor["task_bits_scale"],
        task_cycles_scale=tensor["task_cycles_scale"],
        latency_scale_s=tensor["latency_scale_s"],
        channel_gain_scale=tensor["channel_gain_scale"],
    )


def encode_environment(
    snapshot: EnvironmentSnapshot,
    spec: EnvironmentTensorSpec,
) -> EncodedEnvironment:
    """把一个公式（16）环境快照编码成固定尺寸、带掩码的张量。"""

    grouped_nodes = tuple(
        node
        for node_type in NODE_TYPES
        for node in sorted(
            (item for item in snapshot.nodes if item.node_type == node_type),
            key=lambda item: item.node_id,
        )
    )
    counts = {
        node_type: sum(node.node_type == node_type for node in grouped_nodes)
        for node_type in NODE_TYPES
    }
    limits = {
        "vehicle": spec.max_vehicles,
        "uav": spec.max_uavs,
        "rsu": spec.max_rsus,
    }
    for node_type, count in counts.items():
        if count > limits[node_type]:
            raise ValueError(f"{node_type} 节点数 {count} 超过张量上限 {limits[node_type]}")
    if len(snapshot.tasks) > spec.max_tasks:
        raise ValueError(
            f"任务数 {len(snapshot.tasks)} 超过张量上限 {spec.max_tasks}"
        )

    node_ids = tuple(node.node_id for node in grouped_nodes)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    task_ids = tuple(task.task_id for task in snapshot.tasks)
    mapping_by_task = {mapping.task_id: mapping for mapping in snapshot.mappings}

    node_features = np.zeros((spec.max_nodes, 12), dtype=np.float64)
    node_mask = np.zeros(spec.max_nodes, dtype=np.bool_)
    for index, node in enumerate(grouped_nodes):
        node_mask[index] = True
        node_features[index, NODE_TYPES.index(node.node_type)] = 1.0
        node_features[index, 3:6] = node.position_m / spec.position_scale_m
        node_features[index, 6:9] = node.velocity_mps / spec.speed_scale_mps
        node_features[index, 9:] = (
            node.remaining_bandwidth_hz / spec.bandwidth_scale_hz,
            node.remaining_cpu_frequency_hz / spec.cpu_scale_hz,
            node.remaining_transmit_power_w / spec.power_scale_w,
        )

    matrix_shape = (spec.max_nodes, spec.max_nodes)
    channel_gains = np.zeros(matrix_shape, dtype=np.float64)
    blockage = np.zeros(matrix_shape, dtype=np.float64)
    topology = np.zeros(matrix_shape, dtype=np.float64)
    for link in snapshot.links:
        source = node_index[link.source_node_id]
        target = node_index[link.target_node_id]
        channel_gains[source, target] = link.channel_gain / spec.channel_gain_scale
        blockage[source, target] = float(link.blocked)
        topology[source, target] = float(link.available)

    task_features = np.zeros((spec.max_tasks, 7), dtype=np.float64)
    task_node_indices = np.full((spec.max_tasks, 4), -1, dtype=np.int64)
    task_mask = np.zeros(spec.max_tasks, dtype=np.bool_)
    for index, task in enumerate(snapshot.tasks):
        mapping = mapping_by_task[task.task_id]
        task_mask[index] = True
        task_features[index, :4] = (
            task.input_bits / spec.task_bits_scale,
            task.cpu_cycles / spec.task_cycles_scale,
            task.max_latency_s / spec.latency_scale_s,
            task.output_ratio,
        )
        task_features[index, 4 + TASK_MODES.index(mapping.mode)] = 1.0
        task_node_indices[index] = (
            node_index[task.source_vehicle_id],
            node_index[mapping.compute_node_id]
            if mapping.compute_node_id is not None
            else -1,
            node_index[mapping.relay_uav_id]
            if mapping.relay_uav_id is not None
            else -1,
            node_index[mapping.target_vehicle_id]
            if mapping.target_vehicle_id is not None
            else -1,
        )

    return EncodedEnvironment(
        node_features=node_features,
        channel_gains=channel_gains,
        blockage=blockage,
        topology=topology,
        task_features=task_features,
        task_node_indices=task_node_indices,
        node_mask=node_mask,
        task_mask=task_mask,
        node_ids=node_ids,
        task_ids=task_ids,
    )


__all__ = [
    "EncodedEnvironment",
    "EnvironmentLink",
    "EnvironmentNode",
    "EnvironmentSnapshot",
    "EnvironmentTask",
    "EnvironmentTensorSpec",
    "NODE_TYPES",
    "TASK_MODES",
    "TaskNodeMapping",
    "encode_environment",
    "tensor_spec_from_config",
]
