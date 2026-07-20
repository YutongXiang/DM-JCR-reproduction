import pytest

from dm_jcr.relay_model import (
    RelayLinkResources,
    evaluate_relay_task,
    forwarding_cpu_cycles,
    relay_task_energy_j,
    relay_task_latency_s,
)
from dm_jcr.task_model import (
    ComputationTask,
    kilobytes_to_bits,
)


def make_relay_task():
    return ComputationTask(
        input_bits=kilobytes_to_bits(100.0),
        cpu_cycles=1.0e9,
        max_latency_s=0.6,
        output_ratio=0.1,
    )


def make_relay_resources():
    return RelayLinkResources(
        vehicle_to_uav_rate_bps=20.0e6,
        uav_to_node_rate_bps=50.0e6,
        node_to_uav_rate_bps=50.0e6,
        uav_to_vehicle_rate_bps=40.0e6,
        uav_cpu_frequency_hz=1.5e9,
        node_cpu_frequency_hz=2.0e9,
        vehicle_tx_power_w=0.2,
        uav_tx_power_w=0.5,
        node_tx_power_w=0.5,
        forwarding_cycles_per_bit=0.1,
    )


def test_forwarding_cpu_cycles():
    cycles = forwarding_cpu_cycles(
        data_bits=800_000.0,
        forwarding_cycles_per_bit=0.1,
    )

    assert cycles == pytest.approx(80_000.0)


def test_relay_latency_components():
    metrics = evaluate_relay_task(
        task=make_relay_task(),
        resources=make_relay_resources(),
    )

    assert metrics.vehicle_to_uav_time_s == (
        pytest.approx(0.04)
    )

    assert metrics.uav_to_node_time_s == (
        pytest.approx(0.016)
    )

    assert metrics.input_forwarding_time_s == (
        pytest.approx(80_000.0 / 1.5e9)
    )

    assert metrics.task_computation_time_s == (
        pytest.approx(0.5)
    )

    assert metrics.output_forwarding_time_s == (
        pytest.approx(8_000.0 / 1.5e9)
    )

    assert metrics.node_to_uav_time_s == (
        pytest.approx(0.0016)
    )

    assert metrics.uav_to_vehicle_time_s == (
        pytest.approx(0.002)
    )

    assert metrics.total_latency_s == (
        pytest.approx(0.5596586666666667)
    )


def test_relay_energy_components():
    metrics = evaluate_relay_task(
        task=make_relay_task(),
        resources=make_relay_resources(),
    )

    assert metrics.vehicle_to_uav_energy_j == (
        pytest.approx(0.008)
    )

    assert metrics.uav_to_node_energy_j == (
        pytest.approx(0.008)
    )

    assert metrics.input_forwarding_energy_j == (
        pytest.approx(0.00009)
    )

    assert metrics.task_computation_energy_j == (
        pytest.approx(2.0)
    )

    assert metrics.output_forwarding_energy_j == (
        pytest.approx(0.000009)
    )

    assert metrics.node_to_uav_energy_j == (
        pytest.approx(0.0008)
    )

    assert metrics.uav_to_vehicle_energy_j == (
        pytest.approx(0.001)
    )

    assert metrics.total_energy_j == (
        pytest.approx(2.017899)
    )


def test_relay_wrapper_functions():
    task = make_relay_task()
    resources = make_relay_resources()

    latency = relay_task_latency_s(
        task=task,
        resources=resources,
    )

    energy = relay_task_energy_j(
        task=task,
        resources=resources,
    )

    assert latency == pytest.approx(
        0.5596586666666667
    )

    assert energy == pytest.approx(2.017899)


def test_relay_deadline_is_satisfied():
    metrics = evaluate_relay_task(
        task=make_relay_task(),
        resources=make_relay_resources(),
    )

    assert metrics.deadline_satisfied is True


def test_larger_forwarding_ratio_increases_cost():
    task = make_relay_task()

    low_phi = RelayLinkResources(
        vehicle_to_uav_rate_bps=20.0e6,
        uav_to_node_rate_bps=50.0e6,
        node_to_uav_rate_bps=50.0e6,
        uav_to_vehicle_rate_bps=40.0e6,
        uav_cpu_frequency_hz=1.5e9,
        node_cpu_frequency_hz=2.0e9,
        vehicle_tx_power_w=0.2,
        uav_tx_power_w=0.5,
        node_tx_power_w=0.5,
        forwarding_cycles_per_bit=0.01,
    )

    high_phi = RelayLinkResources(
        vehicle_to_uav_rate_bps=20.0e6,
        uav_to_node_rate_bps=50.0e6,
        node_to_uav_rate_bps=50.0e6,
        uav_to_vehicle_rate_bps=40.0e6,
        uav_cpu_frequency_hz=1.5e9,
        node_cpu_frequency_hz=2.0e9,
        vehicle_tx_power_w=0.2,
        uav_tx_power_w=0.5,
        node_tx_power_w=0.5,
        forwarding_cycles_per_bit=0.2,
    )

    low_metrics = evaluate_relay_task(
        task=task,
        resources=low_phi,
    )

    high_metrics = evaluate_relay_task(
        task=task,
        resources=high_phi,
    )

    assert (
        high_metrics.total_latency_s
        > low_metrics.total_latency_s
    )

    assert (
        high_metrics.total_energy_j
        > low_metrics.total_energy_j
    )
