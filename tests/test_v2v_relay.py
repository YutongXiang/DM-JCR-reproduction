"""Tests for UAV-assisted V2V equations (14) and (15)."""

import pytest

from dm_jcr.task_model import kilobytes_to_bits
from dm_jcr.v2v_relay import (
    V2VRelayResources,
    V2VTask,
    evaluate_v2v_relay,
    v2v_relay_energy_j,
    v2v_relay_latency_s,
)


@pytest.fixture
def task() -> V2VTask:
    """A 100 KB V2V message with a 0.1-second deadline."""
    return V2VTask(
        data_bits=kilobytes_to_bits(100.0),
        max_latency_s=0.1,
    )


@pytest.fixture
def resources() -> V2VRelayResources:
    """Resources chosen to make every term easy to calculate by hand."""
    return V2VRelayResources(
        source_to_uav_rate_bps=20.0e6,
        uav_to_destination_rate_bps=40.0e6,
        uav_cpu_frequency_hz=1.0e9,
        source_vehicle_tx_power_w=0.2,
        uav_tx_power_w=0.5,
        forwarding_cycles_per_bit=0.1,
    )


def test_equation_14_latency_terms(
    task: V2VTask,
    resources: V2VRelayResources,
) -> None:
    result = evaluate_v2v_relay(task, resources)
    assert result.source_to_uav_time_s == pytest.approx(0.04)
    assert result.uav_to_destination_time_s == pytest.approx(0.02)
    assert result.forwarding_processing_time_s == pytest.approx(0.00008)
    assert result.total_latency_s == pytest.approx(0.06008)


def test_equation_15_energy_terms(
    task: V2VTask,
    resources: V2VRelayResources,
) -> None:
    result = evaluate_v2v_relay(task, resources)
    assert result.source_to_uav_energy_j == pytest.approx(0.008)
    assert result.uav_to_destination_energy_j == pytest.approx(0.01)
    assert result.forwarding_processing_energy_j == pytest.approx(0.00004)
    assert result.total_energy_j == pytest.approx(0.01804)


def test_wrapper_functions_match_detailed_result(
    task: V2VTask,
    resources: V2VRelayResources,
) -> None:
    detailed = evaluate_v2v_relay(task, resources)
    assert v2v_relay_latency_s(task, resources) == pytest.approx(
        detailed.total_latency_s
    )
    assert v2v_relay_energy_j(task, resources) == pytest.approx(
        detailed.total_energy_j
    )


def test_deadline_satisfied(
    task: V2VTask,
    resources: V2VRelayResources,
) -> None:
    assert evaluate_v2v_relay(task, resources).meets_deadline is True


def test_deadline_missed(resources: V2VRelayResources) -> None:
    short_deadline_task = V2VTask(
        data_bits=kilobytes_to_bits(100.0),
        max_latency_s=0.05,
    )
    result = evaluate_v2v_relay(short_deadline_task, resources)
    assert result.total_latency_s == pytest.approx(0.06008)
    assert result.meets_deadline is False


def test_zero_data_has_zero_cost(resources: V2VRelayResources) -> None:
    empty_task = V2VTask(data_bits=0.0, max_latency_s=0.1)
    result = evaluate_v2v_relay(empty_task, resources)
    assert result.total_latency_s == pytest.approx(0.0)
    assert result.total_energy_j == pytest.approx(0.0)


def test_more_forwarding_cycles_increase_latency_and_energy(
    task: V2VTask,
) -> None:
    low_phi = V2VRelayResources(
        20.0e6, 40.0e6, 1.0e9, 0.2, 0.5, 0.01
    )
    high_phi = V2VRelayResources(
        20.0e6, 40.0e6, 1.0e9, 0.2, 0.5, 0.2
    )
    low = evaluate_v2v_relay(task, low_phi)
    high = evaluate_v2v_relay(task, high_phi)
    assert high.total_latency_s > low.total_latency_s
    assert high.total_energy_j > low.total_energy_j


def test_higher_uav_frequency_trades_latency_for_energy(
    task: V2VTask,
) -> None:
    slow = V2VRelayResources(
        20.0e6, 40.0e6, 1.0e9, 0.2, 0.5, 0.1
    )
    fast = V2VRelayResources(
        20.0e6, 40.0e6, 2.0e9, 0.2, 0.5, 0.1
    )
    slow_result = evaluate_v2v_relay(task, slow)
    fast_result = evaluate_v2v_relay(task, fast)
    assert fast_result.total_latency_s < slow_result.total_latency_s
    assert fast_result.total_energy_j > slow_result.total_energy_j


@pytest.mark.parametrize(
    ("data_bits", "max_latency_s"),
    [
        (-1.0, 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
        (float("nan"), 1.0),
        (1.0, float("inf")),
    ],
)
def test_invalid_task_values_are_rejected(
    data_bits: float,
    max_latency_s: float,
) -> None:
    with pytest.raises(ValueError):
        V2VTask(data_bits=data_bits, max_latency_s=max_latency_s)


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_to_uav_rate_bps": 0.0},
        {"uav_to_destination_rate_bps": -1.0},
        {"uav_cpu_frequency_hz": 0.0},
        {"source_vehicle_tx_power_w": -0.1},
        {"uav_tx_power_w": -0.1},
        {"forwarding_cycles_per_bit": -0.1},
    ],
)
def test_invalid_resource_values_are_rejected(
    overrides: dict[str, float],
) -> None:
    values = {
        "source_to_uav_rate_bps": 20.0e6,
        "uav_to_destination_rate_bps": 40.0e6,
        "uav_cpu_frequency_hz": 1.0e9,
        "source_vehicle_tx_power_w": 0.2,
        "uav_tx_power_w": 0.5,
        "forwarding_cycles_per_bit": 0.1,
    }
    values.update(overrides)
    with pytest.raises(ValueError):
        V2VRelayResources(**values)


@pytest.mark.parametrize(
    "bad_coefficient",
    [-1.0, float("nan"), float("inf")],
)
def test_invalid_energy_coefficient_is_rejected(
    task: V2VTask,
    resources: V2VRelayResources,
    bad_coefficient: float,
) -> None:
    with pytest.raises(ValueError):
        evaluate_v2v_relay(
            task,
            resources,
            energy_coefficient=bad_coefficient,
        )
