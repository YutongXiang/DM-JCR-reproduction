"""论文公式（17）中直连与中继混合资源分配的单元测试。"""

import pytest

from dm_jcr.channel import achievable_rate_bps
from dm_jcr.relay_model import RelayLinkResources, evaluate_relay_task
from dm_jcr.relay_resource_allocation import (
    FeasibleRelayTaskAllocation,
    ProjectedJointResourceStrategy,
    RawRelayTaskAllocation,
    RelayTaskContext,
    evaluate_joint_resource_strategy,
    evaluate_relay_resource_strategy,
    joint_resource_totals,
    project_joint_resource_strategy,
    verify_joint_resource_constraints,
)
from dm_jcr.resource_allocation import (
    DirectTaskContext,
    FeasibleTaskAllocation,
    NodeResourceCapacity,
    ObjectiveNormalization,
    RawTaskAllocation,
    project_resource_strategy,
)
from dm_jcr.task_model import ComputationTask


def _capacities() -> tuple[NodeResourceCapacity, ...]:
    return (
        NodeResourceCapacity("uav-1", 120.0, 60.0, 30.0),
        NodeResourceCapacity("rsu-1", 100.0, 300.0, 50.0),
    )


def _relay_raw(task_id: str = "relay-task") -> RawRelayTaskAllocation:
    return RawRelayTaskAllocation(
        task_id=task_id,
        relay_uav_id="uav-1",
        compute_node_id="rsu-1",
        vehicle_to_uav_bandwidth_score=1.0,
        uav_to_node_bandwidth_score=1.0,
        node_to_uav_bandwidth_score=3.0,
        uav_to_vehicle_bandwidth_score=2.0,
        relay_cpu_score=2.0,
        compute_cpu_score=1.0,
        relay_power_score=1.0,
        compute_node_power_score=3.0,
    )


def test_joint_projection_shares_node_budgets_between_task_types() -> None:
    direct_raw = [RawTaskAllocation("direct-task", "rsu-1", 1.0, 2.0, 1.0)]
    strategy = project_joint_resource_strategy(
        direct_raw_allocations=direct_raw,
        relay_raw_allocations=[_relay_raw()],
        capacities=_capacities(),
    )

    direct = strategy.direct_allocations[0]
    relay = strategy.relay_allocations[0]

    # 无人机带宽分数按 1:1:2 分配 120 单位的资源池。
    assert relay.vehicle_to_uav_bandwidth_hz == pytest.approx(30.0)
    assert relay.uav_to_node_bandwidth_hz == pytest.approx(30.0)
    assert relay.uav_to_vehicle_bandwidth_hz == pytest.approx(60.0)
    assert relay.relay_cpu_frequency_hz == pytest.approx(60.0)
    assert relay.relay_transmit_power_w == pytest.approx(30.0)

    # 路侧单元上的直连与中继请求竞争相同资源池。
    assert direct.bandwidth_hz == pytest.approx(25.0)
    assert relay.node_to_uav_bandwidth_hz == pytest.approx(75.0)
    assert direct.cpu_frequency_hz == pytest.approx(200.0)
    assert relay.compute_cpu_frequency_hz == pytest.approx(100.0)
    assert direct.node_transmit_power_w == pytest.approx(12.5)
    assert relay.compute_node_transmit_power_w == pytest.approx(37.5)

    assert verify_joint_resource_constraints(strategy, _capacities())
    assert joint_resource_totals(strategy) == pytest.approx(
        {
            "uav-1": (120.0, 60.0, 30.0),
            "rsu-1": (100.0, 300.0, 50.0),
        }
    )


def test_direct_only_joint_projection_matches_existing_projection() -> None:
    capacities = [NodeResourceCapacity("rsu-1", 100.0, 300.0, 50.0)]
    raw = [
        RawTaskAllocation("t1", "rsu-1", 1.0, 3.0, 1.0),
        RawTaskAllocation("t2", "rsu-1", 3.0, 1.0, 2.0),
    ]

    old_projection = project_resource_strategy(raw, capacities)
    joint_projection = project_joint_resource_strategy(raw, [], capacities)

    assert joint_projection.direct_allocations == old_projection
    assert joint_projection.relay_allocations == ()


def _relay_context(max_latency_s: float = 1.0) -> RelayTaskContext:
    return RelayTaskContext(
        task_id="relay-task",
        relay_uav_id="uav-1",
        compute_node_id="rsu-1",
        task=ComputationTask(
            input_bits=8.0e5,
            cpu_cycles=2.0e8,
            max_latency_s=max_latency_s,
            output_ratio=0.1,
        ),
        vehicle_to_uav_channel_gain=2.0e-9,
        uav_to_node_channel_gain=5.0e-9,
        node_to_uav_channel_gain=4.0e-9,
        uav_to_vehicle_channel_gain=3.0e-9,
        vehicle_transmit_power_w=0.2,
        noise_psd_w_hz=4.0e-21,
        forwarding_cycles_per_bit=0.1,
    )


def _relay_allocation() -> FeasibleRelayTaskAllocation:
    return FeasibleRelayTaskAllocation(
        task_id="relay-task",
        relay_uav_id="uav-1",
        compute_node_id="rsu-1",
        vehicle_to_uav_bandwidth_hz=20.0e6,
        uav_to_node_bandwidth_hz=30.0e6,
        node_to_uav_bandwidth_hz=25.0e6,
        uav_to_vehicle_bandwidth_hz=10.0e6,
        relay_cpu_frequency_hz=1.0e9,
        compute_cpu_frequency_hz=2.0e9,
        relay_transmit_power_w=20.0,
        compute_node_transmit_power_w=30.0,
    )


def _relay_eval_capacities() -> tuple[NodeResourceCapacity, ...]:
    return (
        NodeResourceCapacity("uav-1", 60.0e6, 1.0e9, 20.0),
        NodeResourceCapacity("rsu-1", 25.0e6, 2.0e9, 30.0),
    )


def test_relay_evaluation_matches_channel_and_relay_models() -> None:
    context = _relay_context()
    allocation = _relay_allocation()

    evaluation = evaluate_relay_resource_strategy(
        contexts=[context],
        allocations=[allocation],
        capacities=_relay_eval_capacities(),
        normalization=ObjectiveNormalization(energy_reference_j=10.0),
    )
    result = evaluation.tasks[0]

    expected_v2u = achievable_rate_bps(
        allocation.vehicle_to_uav_bandwidth_hz,
        context.vehicle_transmit_power_w,
        context.vehicle_to_uav_channel_gain,
        context.noise_psd_w_hz,
    )
    expected_u2n = achievable_rate_bps(
        allocation.uav_to_node_bandwidth_hz,
        allocation.relay_transmit_power_w,
        context.uav_to_node_channel_gain,
        context.noise_psd_w_hz,
    )
    expected_n2u = achievable_rate_bps(
        allocation.node_to_uav_bandwidth_hz,
        allocation.compute_node_transmit_power_w,
        context.node_to_uav_channel_gain,
        context.noise_psd_w_hz,
    )
    expected_u2v = achievable_rate_bps(
        allocation.uav_to_vehicle_bandwidth_hz,
        allocation.relay_transmit_power_w,
        context.uav_to_vehicle_channel_gain,
        context.noise_psd_w_hz,
    )
    expected_metrics = evaluate_relay_task(
        context.task,
        RelayLinkResources(
            vehicle_to_uav_rate_bps=expected_v2u,
            uav_to_node_rate_bps=expected_u2n,
            node_to_uav_rate_bps=expected_n2u,
            uav_to_vehicle_rate_bps=expected_u2v,
            uav_cpu_frequency_hz=allocation.relay_cpu_frequency_hz,
            node_cpu_frequency_hz=allocation.compute_cpu_frequency_hz,
            vehicle_tx_power_w=context.vehicle_transmit_power_w,
            uav_tx_power_w=allocation.relay_transmit_power_w,
            node_tx_power_w=allocation.compute_node_transmit_power_w,
            forwarding_cycles_per_bit=context.forwarding_cycles_per_bit,
        ),
    )

    assert result.vehicle_to_uav_rate_bps == pytest.approx(expected_v2u)
    assert result.uav_to_node_rate_bps == pytest.approx(expected_u2n)
    assert result.node_to_uav_rate_bps == pytest.approx(expected_n2u)
    assert result.uav_to_vehicle_rate_bps == pytest.approx(expected_u2v)
    assert result.metrics.total_latency_s == pytest.approx(
        expected_metrics.total_latency_s
    )
    assert result.metrics.total_energy_j == pytest.approx(
        expected_metrics.total_energy_j
    )
    assert evaluation.feasible


def test_joint_evaluation_averages_all_tasks_together() -> None:
    capacities = (
        NodeResourceCapacity("uav-1", 100.0e6, 1.0e9, 20.0),
        NodeResourceCapacity("rsu-1", 100.0e6, 3.0e9, 50.0),
    )
    direct_context = DirectTaskContext(
        task_id="direct-task",
        node_id="rsu-1",
        task=ComputationTask(4.0e5, 1.0e8, 1.0, 0.1),
        uplink_channel_gain=2.0e-9,
        downlink_channel_gain=3.0e-9,
        vehicle_transmit_power_w=0.2,
        noise_psd_w_hz=4.0e-21,
    )
    relay_context = _relay_context(max_latency_s=1.0)
    strategy = project_joint_resource_strategy(
        direct_raw_allocations=[
            RawTaskAllocation("direct-task", "rsu-1", 1.0, 1.0, 1.0)
        ],
        relay_raw_allocations=[_relay_raw()],
        capacities=capacities,
    )

    evaluation = evaluate_joint_resource_strategy(
        direct_contexts=[direct_context],
        relay_contexts=[relay_context],
        strategy=strategy,
        capacities=capacities,
        normalization=ObjectiveNormalization(energy_reference_j=10.0),
    )

    direct = evaluation.direct_tasks[0]
    relay = evaluation.relay_tasks[0]
    expected_latency = (direct.normalized_latency + relay.normalized_latency) / 2
    expected_energy = (direct.normalized_energy + relay.normalized_energy) / 2

    assert evaluation.task_count == 2
    assert evaluation.mean_normalized_latency == pytest.approx(expected_latency)
    assert evaluation.mean_normalized_energy == pytest.approx(expected_energy)
    assert evaluation.weighted_objective == pytest.approx(
        0.5 * expected_latency + 0.5 * expected_energy
    )
    assert evaluation.feasible


def test_relay_deadline_violation_is_reported() -> None:
    context = _relay_context(max_latency_s=1.0e-9)
    evaluation = evaluate_relay_resource_strategy(
        contexts=[context],
        allocations=[_relay_allocation()],
        capacities=_relay_eval_capacities(),
        normalization=ObjectiveNormalization(energy_reference_j=10.0),
    )

    assert not evaluation.deadline_constraints_satisfied
    assert not evaluation.feasible


def test_joint_projection_rejects_missing_compute_node_capacity() -> None:
    capacities = [NodeResourceCapacity("uav-1", 1.0, 1.0, 1.0)]
    with pytest.raises(ValueError, match="缺少节点"):
        project_joint_resource_strategy([], [_relay_raw()], capacities)


def test_duplicate_task_ids_across_direct_and_relay_are_rejected() -> None:
    direct = [RawTaskAllocation("same", "rsu-1", 1.0, 1.0, 1.0)]
    with pytest.raises(ValueError, match="必须唯一"):
        project_joint_resource_strategy(direct, [_relay_raw("same")], _capacities())


def test_relay_and_compute_nodes_must_be_different() -> None:
    with pytest.raises(ValueError, match="必须不同"):
        RawRelayTaskAllocation(
            task_id="task",
            relay_uav_id="uav-1",
            compute_node_id="uav-1",
            vehicle_to_uav_bandwidth_score=1.0,
            uav_to_node_bandwidth_score=1.0,
            node_to_uav_bandwidth_score=1.0,
            uav_to_vehicle_bandwidth_score=1.0,
            relay_cpu_score=1.0,
            compute_cpu_score=1.0,
            relay_power_score=1.0,
            compute_node_power_score=1.0,
        )


def test_manual_joint_constraint_check_detects_uav_bandwidth_overflow() -> None:
    strategy = ProjectedJointResourceStrategy(
        direct_allocations=(),
        relay_allocations=(
            FeasibleRelayTaskAllocation(
                task_id="relay-task",
                relay_uav_id="uav-1",
                compute_node_id="rsu-1",
                vehicle_to_uav_bandwidth_hz=30.0,
                uav_to_node_bandwidth_hz=30.0,
                node_to_uav_bandwidth_hz=20.0,
                uav_to_vehicle_bandwidth_hz=50.0,
                relay_cpu_frequency_hz=10.0,
                compute_cpu_frequency_hz=20.0,
                relay_transmit_power_w=5.0,
                compute_node_transmit_power_w=5.0,
            ),
        ),
    )
    capacities = (
        NodeResourceCapacity("uav-1", 100.0, 10.0, 5.0),
        NodeResourceCapacity("rsu-1", 20.0, 20.0, 5.0),
    )

    assert not verify_joint_resource_constraints(strategy, capacities)
