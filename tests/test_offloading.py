"""Tests for equations (9)-(11) and the hysteresis rule."""

from math import exp

import pytest

from dm_jcr.offloading import (
    NodeCandidateState,
    UtilityWeights,
    build_assignment_indicators,
    calculate_node_utility,
    select_node_with_hysteresis,
    select_offloading_node,
)


def make_candidate(
    node_id: str,
    *,
    snr: float = 10.0,
    cpu_ratio: float = 0.2,
    bandwidth_ratio: float = 0.2,
    power_ratio: float = 0.2,
    predicted_latency_s: float = 0.8,
    maximum_latency_s: float = 1.0,
) -> NodeCandidateState:
    """Construct a candidate from easy-to-read utilization ratios."""
    return NodeCandidateState(
        node_id=node_id,
        snr=snr,
        estimated_cpu_load_hz=cpu_ratio * 10.0e9,
        maximum_cpu_hz=10.0e9,
        estimated_bandwidth_load_hz=bandwidth_ratio * 20.0e6,
        maximum_bandwidth_hz=20.0e6,
        estimated_power_load_w=power_ratio * 10.0,
        maximum_power_w=10.0,
        predicted_latency_s=predicted_latency_s,
        maximum_latency_s=maximum_latency_s,
    )


def test_equation_9_term_by_term() -> None:
    """Verify every reward and penalty in equation (9)."""
    candidate = make_candidate(
        "rsu-1",
        snr=exp(2.0) - 1.0,
        cpu_ratio=0.2,
        bandwidth_ratio=0.3,
        power_ratio=0.4,
        predicted_latency_s=1.5,
        maximum_latency_s=1.0,
    )
    weights = UtilityWeights(
        link_quality=1.0,
        cpu_load=2.0,
        bandwidth_load=3.0,
        power_load=4.0,
        latency_risk=5.0,
    )

    result = calculate_node_utility(candidate, weights)

    assert result.link_quality_reward == pytest.approx(2.0)
    assert result.cpu_load_penalty == pytest.approx(0.4)
    assert result.bandwidth_load_penalty == pytest.approx(0.9)
    assert result.power_load_penalty == pytest.approx(1.6)
    assert result.latency_risk_penalty == pytest.approx(2.5)
    assert result.total_utility == pytest.approx(-3.4)


def test_no_latency_penalty_before_deadline() -> None:
    candidate = make_candidate(
        "uav-1",
        predicted_latency_s=0.9,
        maximum_latency_s=1.0,
    )
    result = calculate_node_utility(candidate, UtilityWeights())
    assert result.latency_risk_penalty == pytest.approx(0.0)


def test_better_snr_increases_utility() -> None:
    weights = UtilityWeights()
    low = calculate_node_utility(
        make_candidate("low", snr=1.0),
        weights,
    )
    high = calculate_node_utility(
        make_candidate("high", snr=100.0),
        weights,
    )
    assert high.total_utility > low.total_utility


def test_higher_resource_load_reduces_utility() -> None:
    weights = UtilityWeights()
    lightly_loaded = calculate_node_utility(
        make_candidate(
            "light",
            cpu_ratio=0.1,
            bandwidth_ratio=0.1,
            power_ratio=0.1,
        ),
        weights,
    )
    heavily_loaded = calculate_node_utility(
        make_candidate(
            "heavy",
            cpu_ratio=0.9,
            bandwidth_ratio=0.9,
            power_ratio=0.9,
        ),
        weights,
    )
    assert lightly_loaded.total_utility > heavily_loaded.total_utility


def test_equation_10_selects_highest_utility_node() -> None:
    candidates = [
        make_candidate("rsu-1", snr=2.0),
        make_candidate("uav-1", snr=30.0),
        make_candidate("rsu-2", snr=5.0),
    ]

    decision = select_offloading_node(candidates, UtilityWeights())

    assert decision.selected_node_id == "uav-1"
    assert decision.selected_utility == pytest.approx(
        max(decision.utilities.values())
    )


def test_equation_10_tie_uses_first_candidate() -> None:
    candidates = [
        make_candidate("rsu-first"),
        make_candidate("rsu-second"),
    ]
    decision = select_offloading_node(candidates, UtilityWeights())
    assert decision.selected_node_id == "rsu-first"


def test_equation_11_builds_one_hot_indicators() -> None:
    indicators = build_assignment_indicators(
        ["uav-1", "rsu-1", "rsu-2"],
        selected_node_id="rsu-1",
    )
    assert indicators == {"uav-1": 0, "rsu-1": 1, "rsu-2": 0}
    assert sum(indicators.values()) == 1


def test_hysteresis_keeps_previous_node_for_small_gain() -> None:
    candidates = [
        make_candidate("old", snr=10.0),
        make_candidate("new", snr=10.5),
    ]
    decision = select_node_with_hysteresis(
        candidates,
        UtilityWeights(),
        previous_node_id="old",
        switching_threshold=0.1,
    )
    assert decision.selected_node_id == "old"
    assert decision.switched is False


def test_hysteresis_switches_for_large_gain() -> None:
    candidates = [
        make_candidate("old", snr=1.0),
        make_candidate("new", snr=100.0),
    ]
    decision = select_node_with_hysteresis(
        candidates,
        UtilityWeights(),
        previous_node_id="old",
        switching_threshold=0.1,
    )
    assert decision.selected_node_id == "new"
    assert decision.switched is True


def test_hysteresis_selects_best_if_previous_node_disappears() -> None:
    candidates = [
        make_candidate("uav-1", snr=2.0),
        make_candidate("rsu-1", snr=20.0),
    ]
    decision = select_node_with_hysteresis(
        candidates,
        UtilityWeights(),
        previous_node_id="offline-node",
        switching_threshold=100.0,
    )
    assert decision.selected_node_id == "rsu-1"
    assert decision.switched is True


def test_empty_candidate_collection_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        select_offloading_node([], UtilityWeights())


def test_duplicate_candidate_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        select_offloading_node(
            [make_candidate("same"), make_candidate("same")],
            UtilityWeights(),
        )


def test_selected_indicator_node_must_exist() -> None:
    with pytest.raises(ValueError, match="one of the candidates"):
        build_assignment_indicators(["uav-1"], "rsu-1")


@pytest.mark.parametrize(
    "bad_weight",
    [-1.0, float("inf"), float("nan")],
)
def test_invalid_weight_is_rejected(bad_weight: float) -> None:
    with pytest.raises(ValueError):
        UtilityWeights(link_quality=bad_weight)


def test_all_zero_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        UtilityWeights(0.0, 0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "bad_threshold",
    [-1.0, float("inf"), float("nan")],
)
def test_invalid_hysteresis_threshold_is_rejected(
    bad_threshold: float,
) -> None:
    with pytest.raises(ValueError):
        select_node_with_hysteresis(
            [make_candidate("rsu-1")],
            UtilityWeights(),
            previous_node_id=None,
            switching_threshold=bad_threshold,
        )
