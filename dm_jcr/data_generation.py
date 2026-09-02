"""生成条件环境 E 与高质量资源策略 x0 的扩散模型数据集。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from math import pi
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from dm_jcr.channel import calculate_sinr, noise_psd_w_per_hz
from dm_jcr.environment import (
    EncodedEnvironment,
    EnvironmentLink,
    EnvironmentNode,
    EnvironmentSnapshot,
    EnvironmentTask,
    EnvironmentTensorSpec,
    TaskNodeMapping,
    encode_environment,
    tensor_spec_from_config,
)
from dm_jcr.offloading import (
    NodeCandidateState,
    UtilityWeights,
    select_offloading_node,
)
from dm_jcr.random_channel import sample_random_channel
from dm_jcr.relay_resource_allocation import (
    RawRelayTaskAllocation,
    RelayTaskContext,
)
from dm_jcr.resource_allocation import (
    DirectTaskContext,
    Equation17Evaluation,
    NodeResourceCapacity,
    ObjectiveNormalization,
    ObjectiveWeights,
    RawTaskAllocation,
    RawV2VRelayAllocation,
    V2VRelayTaskContext,
    evaluate_equation17_strategy,
    project_equation17_strategy,
)
from dm_jcr.strategy_codec import (
    EncodedRawStrategy,
    StrategyTensorSpec,
    decode_raw_strategy,
    encode_raw_strategy,
    strategy_spec_from_config,
)
from dm_jcr.task_model import ComputationTask


FloatArray = NDArray[np.float64]
SplitName = Literal["train", "validation", "test"]


def _float_pair(name: str, value: Any, *, positive: bool = False) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} 必须是包含两个数值的范围")
    low, high = float(value[0]), float(value[1])
    lower_bound = 0.0 if positive else -np.inf
    if not np.isfinite((low, high)).all() or low < lower_bound or high < low:
        raise ValueError(f"{name} 不是有效的递增范围")
    if positive and low <= 0.0:
        raise ValueError(f"{name} 的下限必须大于 0")
    return low, high


def _positive_vector2(name: str, value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} 必须包含两个数值")
    result = (float(value[0]), float(value[1]))
    if not np.isfinite(result).all() or any(item <= 0.0 for item in result):
        raise ValueError(f"{name} 必须包含两个有限正数")
    return result


def _int_pair(name: str, value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} 必须是包含两个整数的范围")
    low, high = value
    if not isinstance(low, int) or not isinstance(high, int) or low <= 0 or high < low:
        raise ValueError(f"{name} 不是有效的正整数递增范围")
    return low, high


@dataclass(frozen=True)
class ScenarioGenerationSpec:
    """单时隙随机环境的规模、物理量范围和固定复现假设。"""

    vehicle_count_range: tuple[int, int]
    uav_count_range: tuple[int, int]
    rsu_count_range: tuple[int, int]
    task_count_range: tuple[int, int]
    area_size_m: tuple[float, float]
    uav_altitude_range_m: tuple[float, float]
    vehicle_speed_range_mps: tuple[float, float]
    uav_speed_range_mps: tuple[float, float]
    remaining_resource_fraction_range: tuple[float, float]
    uav_bandwidth_range_hz: tuple[float, float]
    rsu_bandwidth_range_hz: tuple[float, float]
    uav_cpu_range_hz: tuple[float, float]
    rsu_cpu_range_hz: tuple[float, float]
    uav_power_range_w: tuple[float, float]
    rsu_power_range_w: tuple[float, float]
    task_input_bits_range: tuple[float, float]
    task_cpu_cycles_range: tuple[float, float]
    task_latency_range_s: tuple[float, float]
    output_ratio_range: tuple[float, float]
    mode_probabilities: tuple[float, float, float]
    ensure_all_task_modes: bool
    vehicle_transmit_power_w: float
    forwarding_cycles_per_bit: float
    nominal_bandwidth_hz: float
    noise_psd_w_hz: float
    vehicle_antenna_gain_db: float
    uav_antenna_gain_db: float
    rsu_antenna_gain_db: float
    wavelength_m: float
    blockage_probability: float
    rician_k_factor_db: float
    los_loss_range_db: tuple[float, float]
    blocked_loss_range_db: tuple[float, float]
    offloading_weights: UtilityWeights

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.mode_probabilities, dtype=np.float64)
        if probabilities.shape != (3,) or np.any(probabilities < 0.0):
            raise ValueError("mode_probabilities 必须包含三个非负概率")
        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("mode_probabilities 之和必须为 1")
        if self.vehicle_count_range[0] < 2:
            raise ValueError("生成 V2V 任务至少需要两辆车")
        if self.uav_count_range[0] < 1 or self.rsu_count_range[0] < 1:
            raise ValueError("生成器至少需要一个 UAV 和一个 RSU")
        if self.task_count_range[1] > self.vehicle_count_range[1]:
            raise ValueError("任务上限不能超过车辆上限，以保证每辆源车至多一个任务")
        if self.remaining_resource_fraction_range[1] > 1.0:
            raise ValueError("remaining_resource_fraction_range 不能超过 1")
        if self.ensure_all_task_modes and self.task_count_range[0] < 3:
            raise ValueError("确保三类任务时，任务数量下限至少为 3")
        for name in (
            "vehicle_transmit_power_w",
            "forwarding_cycles_per_bit",
            "nominal_bandwidth_hz",
            "noise_psd_w_hz",
            "wavelength_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} 必须是有限正数")
        if not 0.0 <= self.blockage_probability <= 1.0:
            raise ValueError("blockage_probability 必须位于 [0, 1]")


@dataclass(frozen=True)
class StrategySearchSpec:
    """利用现有公式（17）评价器搜索标签策略的参数。"""

    candidate_count: int
    log_score_std: float
    minimum_score: float
    maximum_score: float
    deadline_violation_penalty: float
    require_feasible_label: bool = True
    maximum_scenario_attempts: int = 20

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_count, int) or self.candidate_count <= 0:
            raise ValueError("candidate_count 必须是正整数")
        if self.log_score_std < 0.0:
            raise ValueError("log_score_std 不能为负")
        if self.minimum_score <= 0.0 or self.maximum_score < self.minimum_score:
            raise ValueError("策略分数范围无效")
        if self.deadline_violation_penalty < 0.0:
            raise ValueError("deadline_violation_penalty 不能为负")
        if not isinstance(self.maximum_scenario_attempts, int) or self.maximum_scenario_attempts <= 0:
            raise ValueError("maximum_scenario_attempts 必须是正整数")


@dataclass(frozen=True)
class GeneratedScenario:
    """一个环境快照及公式（17）评价所需的三类任务上下文。"""

    snapshot: EnvironmentSnapshot
    capacities: tuple[NodeResourceCapacity, ...]
    direct_contexts: tuple[DirectTaskContext, ...]
    relay_contexts: tuple[RelayTaskContext, ...]
    v2v_contexts: tuple[V2VRelayTaskContext, ...]

    @property
    def task_count(self) -> int:
        return len(self.snapshot.tasks)


@dataclass(frozen=True)
class StrategySearchResult:
    """候选搜索返回的标签、评价值以及相对于全 1 基线的质量信息。"""

    strategy: EncodedRawStrategy
    evaluation: Equation17Evaluation
    penalized_objective: float
    baseline_objective: float
    baseline_penalized_objective: float
    evaluated_candidates: int


@dataclass(frozen=True)
class DiffusionSample:
    """一对可供条件扩散模型使用的环境 E 和目标策略 x0。"""

    environment: EncodedEnvironment
    strategy: EncodedRawStrategy
    objective: float
    penalized_objective: float
    baseline_objective: float
    baseline_penalized_objective: float
    feasible: bool
    evaluated_candidates: int

    @property
    def E(self) -> FloatArray:
        return self.environment.flat_vector()

    @property
    def x0(self) -> FloatArray:
        return self.strategy.values.copy()


def scenario_spec_from_config(config: Mapping[str, Any]) -> ScenarioGenerationSpec:
    """从统一配置构建随机环境生成规则。"""

    values = config["assumptions"]["dataset_generation"]
    channel = config["assumptions"]["channel"]
    paper_channel = config["paper"]["channel"]
    task = config["paper"]["task"]
    offloading = config["paper"]["offloading"]
    probabilities = tuple(float(x) for x in values["task_mode_probabilities"])
    if len(probabilities) != 3:
        raise ValueError("task_mode_probabilities 必须包含三个概率")
    return ScenarioGenerationSpec(
        vehicle_count_range=_int_pair("vehicle_count_range", values["vehicle_count_range"]),
        uav_count_range=_int_pair("uav_count_range", values["uav_count_range"]),
        rsu_count_range=_int_pair("rsu_count_range", values["rsu_count_range"]),
        task_count_range=_int_pair("task_count_range", values["task_count_range"]),
        area_size_m=_positive_vector2("area_size_m", values["area_size_m"]),
        uav_altitude_range_m=_float_pair("uav_altitude_range_m", values["uav_altitude_range_m"], positive=True),
        vehicle_speed_range_mps=_float_pair("vehicle_speed_range_mps", values["vehicle_speed_range_mps"]),
        uav_speed_range_mps=_float_pair("uav_speed_range_mps", values["uav_speed_range_mps"]),
        remaining_resource_fraction_range=_float_pair("remaining_resource_fraction_range", values["remaining_resource_fraction_range"], positive=True),
        uav_bandwidth_range_hz=_float_pair("uav_bandwidth_range_hz", values["uav_bandwidth_range_hz"], positive=True),
        rsu_bandwidth_range_hz=_float_pair("rsu_bandwidth_range_hz", values["rsu_bandwidth_range_hz"], positive=True),
        uav_cpu_range_hz=_float_pair("uav_cpu_range_hz", values["uav_cpu_range_hz"], positive=True),
        rsu_cpu_range_hz=_float_pair("rsu_cpu_range_hz", values["rsu_cpu_range_hz"], positive=True),
        uav_power_range_w=_float_pair("uav_power_range_w", values["uav_power_range_w"], positive=True),
        rsu_power_range_w=_float_pair("rsu_power_range_w", values["rsu_power_range_w"], positive=True),
        task_input_bits_range=_float_pair("task_input_bits_range", values["task_input_bits_range"], positive=True),
        task_cpu_cycles_range=_float_pair("task_cpu_cycles_range", values["task_cpu_cycles_range"], positive=True),
        task_latency_range_s=_float_pair("task_latency_range_s", values["task_latency_range_s"], positive=True),
        output_ratio_range=_float_pair("output_ratio_range", task["output_ratio_range"]),
        mode_probabilities=probabilities,  # type: ignore[arg-type]
        ensure_all_task_modes=bool(values["ensure_all_task_modes"]),
        vehicle_transmit_power_w=float(values["vehicle_transmit_power_w"]),
        forwarding_cycles_per_bit=float(values["forwarding_cycles_per_bit"]),
        nominal_bandwidth_hz=float(values["nominal_bandwidth_hz"]),
        noise_psd_w_hz=noise_psd_w_per_hz(
            channel["thermal_noise_density_dbm_hz"],
            channel["receiver_noise_figure_db"],
        ),
        vehicle_antenna_gain_db=float(values["vehicle_antenna_gain_db"]),
        uav_antenna_gain_db=float(values["uav_antenna_gain_db"]),
        rsu_antenna_gain_db=float(values["rsu_antenna_gain_db"]),
        wavelength_m=float(paper_channel["wavelength_m"]),
        blockage_probability=float(paper_channel["blockage_probability"]),
        rician_k_factor_db=float(channel["rician_k_factor_db"]),
        los_loss_range_db=_float_pair("los_additional_loss_db", channel["los_additional_loss_db"]),
        blocked_loss_range_db=_float_pair("blocked_additional_loss_db", channel["blocked_additional_loss_db"]),
        offloading_weights=UtilityWeights(
            link_quality=offloading["link_quality_weight"],
            cpu_load=offloading["cpu_load_weight"],
            bandwidth_load=offloading["bandwidth_load_weight"],
            power_load=offloading["power_load_weight"],
            latency_risk=offloading["latency_risk_weight"],
        ),
    )


def search_spec_from_config(config: Mapping[str, Any]) -> StrategySearchSpec:
    """从统一配置读取标签策略搜索参数。"""

    values = config["assumptions"]["strategy_search"]
    return StrategySearchSpec(**values)


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(*bounds))


def _count(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    return int(rng.integers(bounds[0], bounds[1] + 1))


def _velocity(rng: np.random.Generator, speed_range: tuple[float, float]) -> FloatArray:
    speed = _uniform(rng, speed_range)
    angle = rng.uniform(0.0, 2.0 * pi)
    return np.array([speed * np.cos(angle), speed * np.sin(angle), 0.0])


def _antenna_gain(node: EnvironmentNode, spec: ScenarioGenerationSpec) -> float:
    return {
        "vehicle": spec.vehicle_antenna_gain_db,
        "uav": spec.uav_antenna_gain_db,
        "rsu": spec.rsu_antenna_gain_db,
    }[node.node_type]


def _sample_nodes(
    rng: np.random.Generator,
    spec: ScenarioGenerationSpec,
) -> tuple[tuple[EnvironmentNode, ...], tuple[NodeResourceCapacity, ...]]:
    width, height = spec.area_size_m
    nodes: list[EnvironmentNode] = []
    capacities: list[NodeResourceCapacity] = []
    counts = {
        "vehicle": _count(rng, spec.vehicle_count_range),
        "uav": _count(rng, spec.uav_count_range),
        "rsu": _count(rng, spec.rsu_count_range),
    }
    for node_type in ("vehicle", "uav", "rsu"):
        for index in range(counts[node_type]):
            node_id = f"{node_type}-{index + 1}"
            altitude = (
                _uniform(rng, spec.uav_altitude_range_m)
                if node_type == "uav"
                else 0.0
            )
            position = np.array([rng.uniform(0, width), rng.uniform(0, height), altitude])
            if node_type == "vehicle":
                velocity = _velocity(rng, spec.vehicle_speed_range_mps)
                resources = (0.0, 0.0, 0.0)
            else:
                velocity = (
                    _velocity(rng, spec.uav_speed_range_mps)
                    if node_type == "uav"
                    else np.zeros(3)
                )
                bandwidth = _uniform(
                    rng,
                    spec.uav_bandwidth_range_hz if node_type == "uav" else spec.rsu_bandwidth_range_hz,
                )
                cpu = _uniform(
                    rng,
                    spec.uav_cpu_range_hz if node_type == "uav" else spec.rsu_cpu_range_hz,
                )
                power = _uniform(
                    rng,
                    spec.uav_power_range_w if node_type == "uav" else spec.rsu_power_range_w,
                )
                fraction = _uniform(rng, spec.remaining_resource_fraction_range)
                resources = (bandwidth * fraction, cpu * fraction, power * fraction)
                capacities.append(NodeResourceCapacity(node_id, *resources))
            nodes.append(EnvironmentNode(node_id, node_type, position, velocity, *resources))
    return tuple(nodes), tuple(capacities)


def _sample_links(
    rng: np.random.Generator,
    nodes: tuple[EnvironmentNode, ...],
    spec: ScenarioGenerationSpec,
) -> tuple[tuple[EnvironmentLink, ...], dict[tuple[str, str], float]]:
    links: list[EnvironmentLink] = []
    gains: dict[tuple[str, str], float] = {}
    for source in nodes:
        for target in nodes:
            if source.node_id == target.node_id:
                continue
            channel = sample_random_channel(
                rng=rng,
                vehicle_position=source.position_m,
                node_position=target.position_m,
                vehicle_gain_db=_antenna_gain(source, spec),
                node_gain_db=_antenna_gain(target, spec),
                wavelength_m=spec.wavelength_m,
                blockage_probability=spec.blockage_probability,
                rician_k_factor_db=spec.rician_k_factor_db,
                los_loss_range_db=spec.los_loss_range_db,
                blocked_loss_range_db=spec.blocked_loss_range_db,
            )
            gain = max(float(channel.channel_gain), np.finfo(np.float64).tiny)
            gains[(source.node_id, target.node_id)] = gain
            links.append(EnvironmentLink(source.node_id, target.node_id, gain, channel.blocked))
    return tuple(links), gains


def _candidate_state(
    task: ComputationTask,
    node: EnvironmentNode,
    channel_gain: float,
    assigned_count: int,
    spec: ScenarioGenerationSpec,
) -> NodeCandidateState:
    snr = calculate_sinr(
        spec.nominal_bandwidth_hz,
        spec.vehicle_transmit_power_w,
        channel_gain,
        spec.noise_psd_w_hz,
    )
    share = assigned_count + 1
    bandwidth_load = min(node.remaining_bandwidth_hz * 1.5, share * spec.nominal_bandwidth_hz)
    cpu_load = min(node.remaining_cpu_frequency_hz * 1.5, share * task.cpu_cycles / task.max_latency_s)
    power_load = min(node.remaining_transmit_power_w * 1.5, share * node.remaining_transmit_power_w / 4.0)
    rate = spec.nominal_bandwidth_hz * np.log2(1.0 + snr)
    predicted_latency = task.input_bits / max(rate, np.finfo(float).tiny)
    predicted_latency += task.cpu_cycles / max(node.remaining_cpu_frequency_hz, np.finfo(float).tiny)
    return NodeCandidateState(
        node.node_id,
        snr,
        cpu_load,
        node.remaining_cpu_frequency_hz,
        bandwidth_load,
        node.remaining_bandwidth_hz,
        power_load,
        node.remaining_transmit_power_w,
        predicted_latency,
        task.max_latency_s,
    )


def _task_modes(
    rng: np.random.Generator,
    count: int,
    spec: ScenarioGenerationSpec,
) -> tuple[str, ...]:
    fixed = ["direct", "relay_computation", "v2v_relay"] if spec.ensure_all_task_modes else []
    remaining = rng.choice(
        ("direct", "relay_computation", "v2v_relay"),
        size=count - len(fixed),
        p=spec.mode_probabilities,
    ).tolist()
    modes = fixed + remaining
    rng.shuffle(modes)
    return tuple(modes)


def generate_scenario(
    rng: np.random.Generator,
    time_slot: int,
    spec: ScenarioGenerationSpec,
) -> GeneratedScenario:
    """生成一个完整随机时隙，并构造三类公式（17）任务上下文。"""

    nodes, capacities = _sample_nodes(rng, spec)
    links, gains = _sample_links(rng, nodes, spec)
    node_by_id = {node.node_id: node for node in nodes}
    vehicles = tuple(node for node in nodes if node.node_type == "vehicle")
    uavs = tuple(node for node in nodes if node.node_type == "uav")
    rsus = tuple(node for node in nodes if node.node_type == "rsu")
    compute_nodes = (*uavs, *rsus)
    task_count = min(_count(rng, spec.task_count_range), len(vehicles))
    sources = tuple(rng.choice(vehicles, size=task_count, replace=False))
    modes = _task_modes(rng, task_count, spec)
    assigned_count = {node.node_id: 0 for node in compute_nodes}

    tasks: list[EnvironmentTask] = []
    mappings: list[TaskNodeMapping] = []
    direct_contexts: list[DirectTaskContext] = []
    relay_contexts: list[RelayTaskContext] = []
    v2v_contexts: list[V2VRelayTaskContext] = []

    for index, (source, mode) in enumerate(zip(sources, modes, strict=True)):
        task_id = f"slot-{time_slot}/task-{index + 1}"
        input_bits = _uniform(rng, spec.task_input_bits_range)
        cpu_cycles = _uniform(rng, spec.task_cpu_cycles_range)
        deadline = _uniform(rng, spec.task_latency_range_s)
        output_ratio = _uniform(rng, spec.output_ratio_range)
        computation = ComputationTask(input_bits, cpu_cycles, deadline, output_ratio)

        if mode == "direct":
            candidates = tuple(
                _candidate_state(
                    computation,
                    node,
                    gains[(source.node_id, node.node_id)],
                    assigned_count[node.node_id],
                    spec,
                )
                for node in compute_nodes
            )
            selected_id = select_offloading_node(candidates, spec.offloading_weights).selected_node_id
            assigned_count[selected_id] += 1
            mappings.append(TaskNodeMapping(task_id, "direct", compute_node_id=selected_id))
            direct_contexts.append(
                DirectTaskContext(
                    task_id,
                    selected_id,
                    computation,
                    gains[(source.node_id, selected_id)],
                    gains[(selected_id, source.node_id)],
                    spec.vehicle_transmit_power_w,
                    spec.noise_psd_w_hz,
                )
            )
        elif mode == "relay_computation":
            relay = max(
                uavs,
                key=lambda node: gains[(source.node_id, node.node_id)] * gains[(node.node_id, source.node_id)],
            )
            candidates = tuple(
                _candidate_state(
                    computation,
                    node,
                    gains[(relay.node_id, node.node_id)],
                    assigned_count[node.node_id],
                    spec,
                )
                for node in rsus
            )
            compute_id = select_offloading_node(candidates, spec.offloading_weights).selected_node_id
            assigned_count[relay.node_id] += 1
            assigned_count[compute_id] += 1
            mappings.append(
                TaskNodeMapping(
                    task_id,
                    "relay_computation",
                    compute_node_id=compute_id,
                    relay_uav_id=relay.node_id,
                )
            )
            relay_contexts.append(
                RelayTaskContext(
                    task_id,
                    relay.node_id,
                    compute_id,
                    computation,
                    gains[(source.node_id, relay.node_id)],
                    gains[(relay.node_id, compute_id)],
                    gains[(compute_id, relay.node_id)],
                    gains[(relay.node_id, source.node_id)],
                    spec.vehicle_transmit_power_w,
                    spec.noise_psd_w_hz,
                    spec.forwarding_cycles_per_bit,
                )
            )
        else:
            targets = tuple(vehicle for vehicle in vehicles if vehicle.node_id != source.node_id)
            target = targets[int(rng.integers(0, len(targets)))]
            relay = max(
                uavs,
                key=lambda node: gains[(source.node_id, node.node_id)] * gains[(node.node_id, target.node_id)],
            )
            assigned_count[relay.node_id] += 1
            mappings.append(
                TaskNodeMapping(
                    task_id,
                    "v2v_relay",
                    relay_uav_id=relay.node_id,
                    target_vehicle_id=target.node_id,
                )
            )
            v2v_contexts.append(
                V2VRelayTaskContext(
                    task_id,
                    source.node_id,
                    target.node_id,
                    relay.node_id,
                    input_bits,
                    deadline,
                    gains[(source.node_id, relay.node_id)],
                    gains[(relay.node_id, target.node_id)],
                    spec.vehicle_transmit_power_w,
                    spec.noise_psd_w_hz,
                    spec.forwarding_cycles_per_bit,
                )
            )
            cpu_cycles = input_bits * spec.forwarding_cycles_per_bit
            output_ratio = 0.0

        tasks.append(
            EnvironmentTask(task_id, source.node_id, input_bits, cpu_cycles, deadline, output_ratio)
        )

    snapshot = EnvironmentSnapshot(time_slot, nodes, links, tuple(tasks), tuple(mappings))
    return GeneratedScenario(
        snapshot,
        capacities,
        tuple(direct_contexts),
        tuple(relay_contexts),
        tuple(v2v_contexts),
    )


def _active_slots(mapping: TaskNodeMapping) -> tuple[int, ...]:
    if mapping.mode == "direct":
        return (0, 4, 6)
    if mapping.mode == "relay_computation":
        return tuple(range(8))
    return (0, 3, 4, 6)


def _candidate_strategy(
    rng: np.random.Generator,
    mappings: tuple[TaskNodeMapping, ...],
    tensor_spec: StrategyTensorSpec,
    search_spec: StrategySearchSpec,
    center: FloatArray | None,
) -> EncodedRawStrategy:
    values = np.zeros((tensor_spec.max_tasks, tensor_spec.score_width))
    mask = np.zeros(tensor_spec.max_tasks, dtype=np.bool_)
    for index, mapping in enumerate(mappings):
        mask[index] = True
        slots = _active_slots(mapping)
        if center is None:
            scores = np.ones(len(slots))
        else:
            base = np.log(np.maximum(center[index, slots], search_spec.minimum_score))
            scores = np.exp(base + rng.normal(0.0, search_spec.log_score_std, len(slots)))
        values[index, slots] = np.clip(
            scores,
            search_spec.minimum_score,
            search_spec.maximum_score,
        )
    return EncodedRawStrategy(values, mask, mappings)


def _deadline_violations(evaluation: Equation17Evaluation) -> int:
    violations = sum(not item.metrics.meets_deadline for item in evaluation.direct_tasks)
    violations += sum(not item.metrics.deadline_satisfied for item in evaluation.relay_computation_tasks)
    violations += sum(not item.metrics.deadline_satisfied for item in evaluation.v2v_relay_tasks)
    return violations


def _evaluate_candidate(
    encoded: EncodedRawStrategy,
    scenario: GeneratedScenario,
    normalization: ObjectiveNormalization,
    weights: ObjectiveWeights,
    energy_coefficient: float,
    deadline_penalty: float,
) -> tuple[Equation17Evaluation, float]:
    direct, relay, v2v = decode_raw_strategy(encoded)
    projected = project_equation17_strategy(direct, relay, v2v, scenario.capacities)
    evaluation = evaluate_equation17_strategy(
        scenario.direct_contexts,
        scenario.relay_contexts,
        scenario.v2v_contexts,
        projected,
        scenario.capacities,
        normalization,
        weights,
        energy_coefficient=energy_coefficient,
    )
    penalty = deadline_penalty * _deadline_violations(evaluation) / scenario.task_count
    return evaluation, float(evaluation.weighted_objective + penalty)


def search_high_quality_strategy(
    rng: np.random.Generator,
    scenario: GeneratedScenario,
    tensor_spec: StrategyTensorSpec,
    search_spec: StrategySearchSpec,
    normalization: ObjectiveNormalization,
    weights: ObjectiveWeights,
    energy_coefficient: float,
) -> StrategySearchResult:
    """搜索并返回不劣于全 1 基线的高质量公式（17）策略。"""

    mappings = scenario.snapshot.mappings
    baseline = _candidate_strategy(rng, mappings, tensor_spec, search_spec, None)
    baseline_evaluation, baseline_penalized = _evaluate_candidate(
        baseline,
        scenario,
        normalization,
        weights,
        energy_coefficient,
        search_spec.deadline_violation_penalty,
    )
    best = baseline
    best_evaluation = baseline_evaluation
    best_penalized = baseline_penalized
    for index in range(1, search_spec.candidate_count):
        center = best.values if index % 2 == 0 else baseline.values
        candidate = _candidate_strategy(rng, mappings, tensor_spec, search_spec, center)
        evaluation, penalized = _evaluate_candidate(
            candidate,
            scenario,
            normalization,
            weights,
            energy_coefficient,
            search_spec.deadline_violation_penalty,
        )
        if penalized < best_penalized:
            best, best_evaluation, best_penalized = candidate, evaluation, penalized
    return StrategySearchResult(
        best,
        best_evaluation,
        best_penalized,
        baseline_evaluation.weighted_objective,
        baseline_penalized,
        search_spec.candidate_count,
    )


def generate_diffusion_sample(
    rng: np.random.Generator,
    time_slot: int,
    scenario_spec: ScenarioGenerationSpec,
    environment_tensor_spec: EnvironmentTensorSpec,
    strategy_tensor_spec: StrategyTensorSpec,
    search_spec: StrategySearchSpec,
    normalization: ObjectiveNormalization,
    weights: ObjectiveWeights,
    energy_coefficient: float,
) -> DiffusionSample:
    """生成一个环境、搜索其标签策略，并返回成对的 E 与 x0。"""

    for attempt in range(search_spec.maximum_scenario_attempts):
        scenario = generate_scenario(
            rng,
            time_slot * search_spec.maximum_scenario_attempts + attempt,
            scenario_spec,
        )
        result = search_high_quality_strategy(
            rng,
            scenario,
            strategy_tensor_spec,
            search_spec,
            normalization,
            weights,
            energy_coefficient,
        )
        if result.evaluation.feasible or not search_spec.require_feasible_label:
            break
    else:
        raise RuntimeError(
            f"在 {search_spec.maximum_scenario_attempts} 个环境中均未搜索到可行标签；"
            "请增加候选数或放宽数据生成范围"
        )
    return DiffusionSample(
        encode_environment(scenario.snapshot, environment_tensor_spec),
        result.strategy,
        result.evaluation.weighted_objective,
        result.penalized_objective,
        result.baseline_objective,
        result.baseline_penalized_objective,
        result.evaluation.feasible,
        result.evaluated_candidates,
    )


def _dataset_arrays(samples: list[DiffusionSample]) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("数据集至少需要一个样本")
    return {
        "E": np.stack([sample.E for sample in samples]),
        "x0": np.stack([sample.x0 for sample in samples]),
        "node_features": np.stack(
            [sample.environment.node_features for sample in samples]
        ),
        "channel_gains": np.stack(
            [sample.environment.channel_gains for sample in samples]
        ),
        "blockage": np.stack([sample.environment.blockage for sample in samples]),
        "topology": np.stack([sample.environment.topology for sample in samples]),
        "task_features": np.stack(
            [sample.environment.task_features for sample in samples]
        ),
        "task_node_indices": np.stack(
            [sample.environment.task_node_indices for sample in samples]
        ),
        "node_mask": np.stack([sample.environment.node_mask for sample in samples]),
        "task_mask": np.stack([sample.environment.task_mask for sample in samples]),
        "objective": np.asarray([sample.objective for sample in samples]),
        "penalized_objective": np.asarray([sample.penalized_objective for sample in samples]),
        "baseline_objective": np.asarray([sample.baseline_objective for sample in samples]),
        "baseline_penalized_objective": np.asarray(
            [sample.baseline_penalized_objective for sample in samples]
        ),
        "feasible": np.asarray([sample.feasible for sample in samples]),
        "evaluated_candidates": np.asarray([sample.evaluated_candidates for sample in samples]),
    }


def generate_and_save_dataset(
    config: Mapping[str, Any],
    output_directory: str | Path,
    split_sizes: Mapping[SplitName, int],
    *,
    random_seed: int,
    candidate_count: int | None = None,
) -> dict[str, Any]:
    """生成三个互斥数据划分，保存压缩 NPZ 和 JSON 清单。"""

    scenario_spec = scenario_spec_from_config(config)
    environment_spec = tensor_spec_from_config(config)
    strategy_spec = strategy_spec_from_config(config)
    search_spec = search_spec_from_config(config)
    if candidate_count is not None:
        search_spec = StrategySearchSpec(
            candidate_count,
            search_spec.log_score_std,
            search_spec.minimum_score,
            search_spec.maximum_score,
            search_spec.deadline_violation_penalty,
            search_spec.require_feasible_label,
            search_spec.maximum_scenario_attempts,
        )
    normalization = ObjectiveNormalization(
        energy_reference_j=config["assumptions"]["objective"]["energy_reference_j"]
    )
    weights = ObjectiveWeights(
        config["paper"]["objective"]["latency_weight"],
        config["paper"]["objective"]["energy_weight"],
    )
    energy_coefficient = float(config["paper"]["task"]["cpu_energy_coefficient"])
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    seed_sequence = np.random.SeedSequence(random_seed)
    child_seeds = seed_sequence.spawn(3)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "random_seed": random_seed,
        "candidate_count": search_spec.candidate_count,
        "environment_width": None,
        "strategy_shape": [strategy_spec.max_tasks, strategy_spec.score_width],
        "splits": {},
        "label_semantics": "搜索得到的最佳候选策略，不保证是全局最优解",
    }
    for split_index, split in enumerate(("train", "validation", "test")):
        size = int(split_sizes[split])
        if size <= 0:
            raise ValueError(f"{split} 样本数必须是正整数")
        rng = np.random.default_rng(child_seeds[split_index])
        samples = [
            generate_diffusion_sample(
                rng,
                sample_index,
                scenario_spec,
                environment_spec,
                strategy_spec,
                search_spec,
                normalization,
                weights,
                energy_coefficient,
            )
            for sample_index in range(size)
        ]
        arrays = _dataset_arrays(samples)
        path = output / f"{split}.npz"
        np.savez_compressed(path, **arrays)
        manifest["environment_width"] = int(arrays["E"].shape[1])
        manifest["splits"][split] = {
            "samples": size,
            "file": path.name,
            "mean_objective": float(arrays["objective"].mean()),
            "mean_baseline_objective": float(arrays["baseline_objective"].mean()),
            "mean_penalized_improvement": float(
                (arrays["baseline_penalized_objective"] - arrays["penalized_objective"]).mean()
            ),
            "feasible_ratio": float(arrays["feasible"].mean()),
        }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


__all__ = [
    "DiffusionSample",
    "GeneratedScenario",
    "ScenarioGenerationSpec",
    "StrategySearchResult",
    "StrategySearchSpec",
    "generate_and_save_dataset",
    "generate_diffusion_sample",
    "generate_scenario",
    "scenario_spec_from_config",
    "search_high_quality_strategy",
    "search_spec_from_config",
]
