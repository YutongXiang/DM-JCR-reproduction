"""Tests for equations (17) and (27)."""

import numpy as np
import pytest

from dm_jcr.channel import achievable_rate_bps
from dm_jcr.resource_allocation import (
    DirectTaskContext,
    FeasibleTaskAllocation,
    NodeResourceCapacity,
    ObjectiveNormalization,
    ObjectiveWeights,
    RawTaskAllocation,
    evaluate_direct_resource_strategy,
    project_resource_strategy,
    proportional_normalize,
    verify_resource_constraints,
)
from dm_jcr.task_model import ComputationTask


def test_proportional_normalize_matches_equation_27() -> None:
    result = proportional_normalize([1.0, 3.0], total_resource=40.0)
    assert result == pytest.approx([10.0, 30.0])
    assert float(result.sum()) == pytest.approx(40.0)


def test_zero_scores_use_equal_split() -> None:
    result = proportional_normalize([0.0, 0.0, 0.0], total_resource=12.0)
    assert result == pytest.approx([4.0, 4.0, 4.0])


def test_projection_enforces_three_resource_totals_per_node() -> None:
    capacities = [
        NodeResourceCapacity("uav-1", 100.0, 200.0, 30.0),
        NodeResourceCapacity("rsu-1", 80.0, 300.0, 50.0),
    ]
    raw = [
        RawTaskAllocation("t1", "uav-1", 1.0, 3.0, 1.0),
        RawTaskAllocation("t2", "uav-1", 3.0, 1.0, 2.0),
        RawTaskAllocation("t3", "rsu-1", 7.0, 9.0, 5.0),
    ]

    projected = project_resource_strategy(raw, capacities)
    assert verify_resource_constraints(projected, capacities)

    uav = [item for item in projected if item.node_id == "uav-1"]
    assert sum(item.bandwidth_hz for item in uav) == pytest.approx(100.0)
    assert sum(item.cpu_frequency_hz for item in uav) == pytest.approx(200.0)
    assert sum(item.node_transmit_power_w for item in uav) == pytest.approx(30.0)
    assert uav[0].bandwidth_hz == pytest.approx(25.0)
    assert uav[1].bandwidth_hz == pytest.approx(75.0)

    rsu = [item for item in projected if item.node_id == "rsu-1"]
    assert rsu[0].bandwidth_hz == pytest.approx(80.0)
    assert rsu[0].cpu_frequency_hz == pytest.approx(300.0)
    assert rsu[0].node_transmit_power_w == pytest.approx(50.0)


def _example_context(max_latency_s: float = 10.0) -> DirectTaskContext:
    return DirectTaskContext(
        task_id="task-1",
        node_id="rsu-1",
        task=ComputationTask(
            input_bits=8.0e5,
            cpu_cycles=1.0e8,
            max_latency_s=max_latency_s,
            output_ratio=0.1,
        ),
        uplink_channel_gain=1.0e-8,
        downlink_channel_gain=2.0e-8,
        vehicle_transmit_power_w=0.2,
        noise_psd_w_hz=1.0e-15,
    )


def test_evaluation_connects_channel_task_and_objective_models() -> None:
    capacity = NodeResourceCapacity(
        "rsu-1",
        total_bandwidth_hz=2.0e6,
        total_cpu_frequency_hz=2.0e9,
        total_transmit_power_w=1.0,
    )
    allocation = FeasibleTaskAllocation(
        task_id="task-1",
        node_id="rsu-1",
        bandwidth_hz=2.0e6,
        cpu_frequency_hz=2.0e9,
        node_transmit_power_w=1.0,
    )
    context = _example_context()

    result = evaluate_direct_resource_strategy(
        contexts=[context],
        allocations=[allocation],
        capacities=[capacity],
        normalization=ObjectiveNormalization(energy_reference_j=10.0),
        weights=ObjectiveWeights(latency=0.5, energy=0.5),
    )

    expected_up = achievable_rate_bps(
        bandwidth_hz=2.0e6,
        transmit_power_w=0.2,
        channel_gain=1.0e-8,
        noise_psd_w_hz=1.0e-15,
    )
    expected_down = achievable_rate_bps(
        bandwidth_hz=2.0e6,
        transmit_power_w=1.0,
        channel_gain=2.0e-8,
        noise_psd_w_hz=1.0e-15,
    )

    task_result = result.tasks[0]
    assert task_result.uplink_rate_bps == pytest.approx(expected_up)
    assert task_result.downlink_rate_bps == pytest.approx(expected_down)
    assert result.weighted_objective == pytest.approx(
        0.5 * task_result.normalized_latency
        + 0.5 * task_result.normalized_energy
    )
    assert result.resource_constraints_satisfied
    assert result.deadline_constraints_satisfied
    assert result.feasible


def test_deadline_constraint_is_reported_without_changing_objective() -> None:
    context = _example_context(max_latency_s=1.0e-6)
    capacity = NodeResourceCapacity("rsu-1", 1.0e6, 1.0e9, 1.0)
    allocation = FeasibleTaskAllocation(
        "task-1", "rsu-1", 1.0e6, 1.0e9, 1.0
    )

    result = evaluate_direct_resource_strategy(
        contexts=[context],
        allocations=[allocation],
        capacities=[capacity],
        normalization=ObjectiveNormalization(energy_reference_j=10.0),
    )

    assert not result.deadline_constraints_satisfied
    assert not result.feasible
    assert np.isfinite(result.weighted_objective)


def test_projection_rejects_missing_node_capacity() -> None:
    raw = [RawTaskAllocation("task-1", "missing", 1.0, 1.0, 1.0)]
    unrelated_capacity = NodeResourceCapacity("other", 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="missing capacity"):
        project_resource_strategy(raw, [unrelated_capacity])


def test_objective_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        ObjectiveWeights(latency=0.7, energy=0.7)
