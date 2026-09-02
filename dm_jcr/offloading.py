"""实现论文公式（9）至（11）的规则式任务—节点映射。

论文先用效用函数评价每个候选无人机或路侧单元，再把任务映射到效用最大的
节点。该映射是后续资源分配问题的输入，并非由扩散模型生成。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log1p
from typing import Iterable


def _finite_non_negative(name: str, value: float) -> float:
    """返回满足 ``value >= 0`` 的有限浮点数。"""
    value = float(value)
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} 必须为非负有限数值")
    return value


def _finite_positive(name: str, value: float) -> float:
    """返回满足 ``value > 0`` 的有限浮点数。"""
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} 必须为大于 0 的有限数值")
    return value


@dataclass(frozen=True)
class UtilityWeights:
    r"""公式（9）中的权重 :math:`\psi_1,\ldots,\psi_5`。

    论文未要求五个权重之和为一；其大小控制链路质量、当前资源负载和时限风险
    的相对重要程度。
    """

    link_quality: float = 1.0
    cpu_load: float = 1.0
    bandwidth_load: float = 1.0
    power_load: float = 1.0
    latency_risk: float = 1.0

    def __post_init__(self) -> None:
        names = (
            "link_quality",
            "cpu_load",
            "bandwidth_load",
            "power_load",
            "latency_risk",
        )
        for name in names:
            object.__setattr__(
                self,
                name,
                _finite_non_negative(name, getattr(self, name)),
            )

        if all(getattr(self, name) == 0.0 for name in names):
            raise ValueError("至少一个效用权重必须为正数")


@dataclass(frozen=True)
class NodeCandidateState:
    """评价一个候选无人机或路侧单元所需的信息。

    ``estimated_*_load`` 表示接收任务后的预期负载。此处允许数值超过对应上限，
    并给予大于一的惩罚；公式（17）会进一步执行硬可行性约束。
    """

    node_id: str
    snr: float

    estimated_cpu_load_hz: float
    maximum_cpu_hz: float

    estimated_bandwidth_load_hz: float
    maximum_bandwidth_hz: float

    estimated_power_load_w: float
    maximum_power_w: float

    predicted_latency_s: float
    maximum_latency_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("node_id 必须是非空字符串")
        object.__setattr__(self, "node_id", self.node_id.strip())

        non_negative = (
            "snr",
            "estimated_cpu_load_hz",
            "estimated_bandwidth_load_hz",
            "estimated_power_load_w",
            "predicted_latency_s",
        )
        positive = (
            "maximum_cpu_hz",
            "maximum_bandwidth_hz",
            "maximum_power_w",
            "maximum_latency_s",
        )

        for name in non_negative:
            object.__setattr__(
                self,
                name,
                _finite_non_negative(name, getattr(self, name)),
            )
        for name in positive:
            object.__setattr__(
                self,
                name,
                _finite_positive(name, getattr(self, name)),
            )


@dataclass(frozen=True)
class UtilityBreakdown:
    """组成公式（9）的奖励项与惩罚项。"""

    link_quality_reward: float
    cpu_load_penalty: float
    bandwidth_load_penalty: float
    power_load_penalty: float
    latency_risk_penalty: float
    total_utility: float


@dataclass(frozen=True)
class OffloadingDecision:
    """公式（10）和（11）的结果，可选用迟滞机制。"""

    selected_node_id: str
    selected_utility: float
    utilities: dict[str, float]
    assignment_indicators: dict[str, int]
    previous_node_id: str | None = None
    switched: bool = False


def calculate_node_utility(
    candidate: NodeCandidateState,
    weights: UtilityWeights,
) -> UtilityBreakdown:
    r"""为一个任务—候选节点对计算公式（9）。

    实现的效用函数为

    .. math::

       U = \psi_1\log(1+\mathrm{SNR})
           -\psi_2 L_{cpu}/F^{max}
           -\psi_3 L_{bw}/B^{max}
           -\psi_4 L_{pw}/P^{max}
           -\psi_5 \max(0, \hat T-T^{max})/T^{max}.

    ``math.log1p`` 使用自然对数。论文中的 ``log`` 未注明底数；改变底数只会
    缩放第一项，因此可吸收到 :math:`\psi_1` 中。
    """
    link_quality_reward = weights.link_quality * log1p(candidate.snr)
    cpu_load_penalty = weights.cpu_load * (
        candidate.estimated_cpu_load_hz / candidate.maximum_cpu_hz
    )
    bandwidth_load_penalty = weights.bandwidth_load * (
        candidate.estimated_bandwidth_load_hz
        / candidate.maximum_bandwidth_hz
    )
    power_load_penalty = weights.power_load * (
        candidate.estimated_power_load_w / candidate.maximum_power_w
    )

    excess_latency_s = max(
        0.0,
        candidate.predicted_latency_s - candidate.maximum_latency_s,
    )
    latency_risk_penalty = weights.latency_risk * (
        excess_latency_s / candidate.maximum_latency_s
    )

    total_utility = (
        link_quality_reward
        - cpu_load_penalty
        - bandwidth_load_penalty
        - power_load_penalty
        - latency_risk_penalty
    )

    return UtilityBreakdown(
        link_quality_reward=float(link_quality_reward),
        cpu_load_penalty=float(cpu_load_penalty),
        bandwidth_load_penalty=float(bandwidth_load_penalty),
        power_load_penalty=float(power_load_penalty),
        latency_risk_penalty=float(latency_risk_penalty),
        total_utility=float(total_utility),
    )


def _candidate_tuple(
    candidates: Iterable[NodeCandidateState],
) -> tuple[NodeCandidateState, ...]:
    """将候选节点具体化，并拒绝空列表或重复节点标识。"""
    result = tuple(candidates)
    if not result:
        raise ValueError("至少需要一个候选节点")

    node_ids = [candidate.node_id for candidate in result]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("候选节点标识必须唯一")
    return result


def calculate_all_utilities(
    candidates: Iterable[NodeCandidateState],
    weights: UtilityWeights,
) -> dict[str, float]:
    """按输入顺序为每个候选节点计算公式（9）。"""
    candidate_tuple = _candidate_tuple(candidates)
    return {
        candidate.node_id: calculate_node_utility(
            candidate,
            weights,
        ).total_utility
        for candidate in candidate_tuple
    }


def build_assignment_indicators(
    candidate_node_ids: Iterable[str],
    selected_node_id: str,
) -> dict[str, int]:
    """构造公式（11）中的二进制指示变量。"""
    node_ids = tuple(candidate_node_ids)
    if not node_ids:
        raise ValueError("至少需要一个候选节点标识")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("候选节点标识必须唯一")
    if selected_node_id not in node_ids:
        raise ValueError("selected_node_id 必须属于候选节点")

    return {
        node_id: int(node_id == selected_node_id)
        for node_id in node_ids
    }


def select_offloading_node(
    candidates: Iterable[NodeCandidateState],
    weights: UtilityWeights,
) -> OffloadingDecision:
    """按照公式（10）选择效用最大的节点。

    多个候选节点效用完全相同时，选择输入顺序中的第一个，使平局处理具有确定性。
    """
    candidate_tuple = _candidate_tuple(candidates)
    utilities = calculate_all_utilities(candidate_tuple, weights)
    selected_node_id = max(utilities, key=utilities.__getitem__)
    indicators = build_assignment_indicators(
        utilities.keys(),
        selected_node_id,
    )

    return OffloadingDecision(
        selected_node_id=selected_node_id,
        selected_utility=utilities[selected_node_id],
        utilities=utilities,
        assignment_indicators=indicators,
    )


def select_node_with_hysteresis(
    candidates: Iterable[NodeCandidateState],
    weights: UtilityWeights,
    *,
    previous_node_id: str | None,
    switching_threshold: float,
) -> OffloadingDecision:
    """在公式（9）至（11）之后应用论文的任务级迟滞机制。

    只有新最优效用严格超过原节点效用 ``switching_threshold`` 以上时，任务才会
    切换节点。若原节点已不在候选集中，则直接选择当前最优节点。
    """
    threshold = _finite_non_negative(
        "switching_threshold",
        switching_threshold,
    )
    candidate_tuple = _candidate_tuple(candidates)
    utilities = calculate_all_utilities(candidate_tuple, weights)
    best_node_id = max(utilities, key=utilities.__getitem__)

    if previous_node_id is None or previous_node_id not in utilities:
        selected_node_id = best_node_id
    else:
        utility_gain = (
            utilities[best_node_id] - utilities[previous_node_id]
        )
        selected_node_id = (
            best_node_id
            if utility_gain > threshold
            else previous_node_id
        )

    indicators = build_assignment_indicators(
        utilities.keys(),
        selected_node_id,
    )
    switched = (
        previous_node_id is not None
        and selected_node_id != previous_node_id
    )

    return OffloadingDecision(
        selected_node_id=selected_node_id,
        selected_utility=utilities[selected_node_id],
        utilities=utilities,
        assignment_indicators=indicators,
        previous_node_id=previous_node_id,
        switched=switched,
    )


__all__ = [
    "NodeCandidateState",
    "OffloadingDecision",
    "UtilityBreakdown",
    "UtilityWeights",
    "build_assignment_indicators",
    "calculate_all_utilities",
    "calculate_node_utility",
    "select_node_with_hysteresis",
    "select_offloading_node",
]
