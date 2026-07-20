"""Rule-based task-to-node mapping from equations (9)-(11).

The paper first evaluates every candidate UAV/RSU with a utility function,
then maps a task to the node with the largest utility.  This mapping is an
input to the later resource-allocation problem; it is not itself generated
by the diffusion model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log1p
from typing import Iterable


def _finite_non_negative(name: str, value: float) -> float:
    """Return a finite float constrained to ``value >= 0``."""
    value = float(value)
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _finite_positive(name: str, value: float) -> float:
    """Return a finite float constrained to ``value > 0``."""
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and greater than 0")
    return value


@dataclass(frozen=True)
class UtilityWeights:
    r"""Weights :math:`\psi_1,\ldots,\psi_5` in equation (9).

    The paper does not require these five values to sum to one.  Their scale
    controls the relative importance of link quality, current resource load
    and deadline risk.
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
            raise ValueError("at least one utility weight must be positive")


@dataclass(frozen=True)
class NodeCandidateState:
    """Information needed to evaluate one candidate UAV or RSU.

    ``estimated_*_load`` is the load expected after accepting the task.
    Values larger than the corresponding maximum are allowed here and receive
    a penalty larger than one.  Hard feasibility constraints will be enforced
    later when equation (17) is implemented.
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
            raise ValueError("node_id must be a non-empty string")
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
    """Reward and penalty terms that compose equation (9)."""

    link_quality_reward: float
    cpu_load_penalty: float
    bandwidth_load_penalty: float
    power_load_penalty: float
    latency_risk_penalty: float
    total_utility: float


@dataclass(frozen=True)
class OffloadingDecision:
    """Result of equations (10), (11), optionally with hysteresis."""

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
    r"""Calculate equation (9) for one task-node candidate pair.

    The implemented utility is

    .. math::

       U = \psi_1\log(1+\mathrm{SNR})
           -\psi_2 L_{cpu}/F^{max}
           -\psi_3 L_{bw}/B^{max}
           -\psi_4 L_{pw}/P^{max}
           -\psi_5 \max(0, \hat T-T^{max})/T^{max}.

    ``math.log1p`` is the natural logarithm.  The paper writes ``log`` without
    specifying its base; changing the base only rescales the first term and
    can therefore be absorbed into :math:`\psi_1`.
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
    """Materialize candidates and reject empty or duplicate node IDs."""
    result = tuple(candidates)
    if not result:
        raise ValueError("at least one candidate node is required")

    node_ids = [candidate.node_id for candidate in result]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("candidate node IDs must be unique")
    return result


def calculate_all_utilities(
    candidates: Iterable[NodeCandidateState],
    weights: UtilityWeights,
) -> dict[str, float]:
    """Calculate equation (9) for every candidate in input order."""
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
    """Build the binary indicators in equation (11)."""
    node_ids = tuple(candidate_node_ids)
    if not node_ids:
        raise ValueError("at least one candidate node ID is required")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("candidate node IDs must be unique")
    if selected_node_id not in node_ids:
        raise ValueError("selected_node_id must be one of the candidates")

    return {
        node_id: int(node_id == selected_node_id)
        for node_id in node_ids
    }


def select_offloading_node(
    candidates: Iterable[NodeCandidateState],
    weights: UtilityWeights,
) -> OffloadingDecision:
    """Select the maximum-utility node in equation (10).

    If multiple candidates have exactly the same utility, the first candidate
    in the supplied order is selected.  This makes tie handling deterministic.
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
    """Apply the paper's task-level hysteresis after equations (9)-(11).

    A task changes nodes only when the new best utility exceeds the previous
    node's utility by strictly more than ``switching_threshold``.  If the old
    node is no longer a candidate, the current best node is selected directly.
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
