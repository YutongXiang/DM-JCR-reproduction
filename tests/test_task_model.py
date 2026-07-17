"""Tests for the direct-link task model.

Run this file from the project root with:

    python -m pytest tests/test_task_model.py -q
"""

import pytest

from dm_jcr.task_model import (
    DEFAULT_CPU_ENERGY_COEFFICIENT,
    ComputationTask,
    DirectLinkResources,
    computation_time_s,
    direct_task_energy_j,
    direct_task_latency_s,
    dynamic_cpu_energy_j,
    evaluate_direct_task,
    kilobytes_to_bits,
    transmission_time_s,
)


@pytest.fixture
def example_task() -> ComputationTask:
    """Create a task with values that are easy to calculate manually.

    D = 8,000,000 bit
    C = 2,000,000,000 cycle
    mu = 0.1
    T_max = 3.5 s

    Therefore, the output size is:

        mu * D = 800,000 bit
    """
    return ComputationTask(
        input_bits=8.0e6,
        cpu_cycles=2.0e9,
        max_latency_s=3.5,
        output_ratio=0.1,
    )


@pytest.fixture
def example_resources() -> DirectLinkResources:
    """Create one direct-link resource allocation.

    Uplink rate:
        r_vn = 4 Mbit/s

    Downlink rate:
        r_nv = 2 Mbit/s

    Edge CPU frequency:
        f_n = 2 GHz

    Transmission powers:
        p_v = 0.2 W
        p_n = 0.5 W
    """
    return DirectLinkResources(
        uplink_rate_bps=4.0e6,
        downlink_rate_bps=2.0e6,
        cpu_frequency_hz=2.0e9,
        vehicle_tx_power_w=0.2,
        node_tx_power_w=0.5,
    )


def test_kilobytes_to_bits() -> None:
    """1 KB should equal 8000 bit in the decimal-KB convention."""
    assert kilobytes_to_bits(1.0) == pytest.approx(8000.0)
    assert kilobytes_to_bits(0.0) == pytest.approx(0.0)
    assert kilobytes_to_bits(1000.0) == pytest.approx(8.0e6)


def test_task_output_bits(
    example_task: ComputationTask,
) -> None:
    """The result-data size should be mu times the input size."""
    expected_output_bits = 0.1 * 8.0e6

    assert example_task.output_bits == pytest.approx(
        expected_output_bits
    )
    assert example_task.output_bits == pytest.approx(8.0e5)


def test_transmission_time() -> None:
    """Transmission time should follow T = D / r."""
    data_bits = 8.0e6
    rate_bps = 4.0e6

    result = transmission_time_s(
        data_bits=data_bits,
        rate_bps=rate_bps,
    )

    assert result == pytest.approx(2.0)


def test_zero_data_has_zero_transmission_time() -> None:
    """Transmitting zero bits should require zero seconds."""
    result = transmission_time_s(
        data_bits=0.0,
        rate_bps=4.0e6,
    )

    assert result == pytest.approx(0.0)


def test_computation_time() -> None:
    """Computation time should follow T = C / f."""
    cpu_cycles = 2.0e9
    cpu_frequency_hz = 2.0e9

    result = computation_time_s(
        cpu_cycles=cpu_cycles,
        cpu_frequency_hz=cpu_frequency_hz,
    )

    assert result == pytest.approx(1.0)


def test_dynamic_cpu_energy() -> None:
    """CPU energy should follow E = zeta * f^2 * C."""
    cpu_cycles = 2.0e9
    cpu_frequency_hz = 2.0e9
    zeta = DEFAULT_CPU_ENERGY_COEFFICIENT

    result = dynamic_cpu_energy_j(
        cpu_cycles=cpu_cycles,
        cpu_frequency_hz=cpu_frequency_hz,
        energy_coefficient=zeta,
    )

    expected = (
        zeta
        * cpu_frequency_hz**2
        * cpu_cycles
    )

    assert expected == pytest.approx(4.0)
    assert result == pytest.approx(expected)
    assert result == pytest.approx(4.0)


def test_direct_task_latency(
    example_task: ComputationTask,
    example_resources: DirectLinkResources,
) -> None:
    """Verify the direct branch of equation (12).

    Upload:
        D / r_vn = 8e6 / 4e6 = 2.0 s

    Computation:
        C / f_n = 2e9 / 2e9 = 1.0 s

    Download:
        mu*D / r_nv = 0.8e6 / 2e6 = 0.4 s

    Total:
        2.0 + 1.0 + 0.4 = 3.4 s
    """
    result = direct_task_latency_s(
        task=example_task,
        resources=example_resources,
    )

    assert result == pytest.approx(3.4)


def test_direct_task_energy(
    example_task: ComputationTask,
    example_resources: DirectLinkResources,
) -> None:
    """Verify the direct branch of equation (13).

    Vehicle upload energy:
        p_v * D/r_vn = 0.2 * 2.0 = 0.4 J

    Edge computation energy:
        zeta * f_n^2 * C = 4.0 J

    Node download energy:
        p_n * mu*D/r_nv = 0.5 * 0.4 = 0.2 J

    Total:
        0.4 + 4.0 + 0.2 = 4.6 J
    """
    result = direct_task_energy_j(
        task=example_task,
        resources=example_resources,
    )

    assert result == pytest.approx(4.6)


def test_evaluate_direct_task(
    example_task: ComputationTask,
    example_resources: DirectLinkResources,
) -> None:
    """Check all latency and energy components together."""
    result = evaluate_direct_task(
        task=example_task,
        resources=example_resources,
    )

    # Equation (12): latency breakdown.
    assert result.upload_latency_s == pytest.approx(2.0)
    assert result.computation_latency_s == pytest.approx(1.0)
    assert result.download_latency_s == pytest.approx(0.4)
    assert result.total_latency_s == pytest.approx(3.4)

    # Equation (13): energy breakdown.
    assert result.upload_energy_j == pytest.approx(0.4)
    assert result.computation_energy_j == pytest.approx(4.0)
    assert result.download_energy_j == pytest.approx(0.2)
    assert result.total_energy_j == pytest.approx(4.6)

    # The deadline is 3.5 s, while total latency is 3.4 s.
    assert result.meets_deadline is True


def test_evaluation_totals_match_individual_functions(
    example_task: ComputationTask,
    example_resources: DirectLinkResources,
) -> None:
    """The detailed interface should agree with the total functions."""
    evaluation = evaluate_direct_task(
        task=example_task,
        resources=example_resources,
    )

    expected_latency = direct_task_latency_s(
        task=example_task,
        resources=example_resources,
    )
    expected_energy = direct_task_energy_j(
        task=example_task,
        resources=example_resources,
    )

    assert evaluation.total_latency_s == pytest.approx(
        expected_latency
    )
    assert evaluation.total_energy_j == pytest.approx(
        expected_energy
    )


def test_task_misses_deadline(
    example_resources: DirectLinkResources,
) -> None:
    """A 3.4-second task should miss a 3.0-second deadline."""
    task = ComputationTask(
        input_bits=8.0e6,
        cpu_cycles=2.0e9,
        max_latency_s=3.0,
        output_ratio=0.1,
    )

    result = evaluate_direct_task(
        task=task,
        resources=example_resources,
    )

    assert result.total_latency_s == pytest.approx(3.4)
    assert result.meets_deadline is False


def test_task_exactly_meets_deadline(
    example_resources: DirectLinkResources,
) -> None:
    """Latency equal to the deadline should be feasible."""
    task = ComputationTask(
        input_bits=8.0e6,
        cpu_cycles=2.0e9,
        max_latency_s=3.4,
        output_ratio=0.1,
    )

    result = evaluate_direct_task(
        task=task,
        resources=example_resources,
    )

    assert result.total_latency_s == pytest.approx(3.4)
    assert result.meets_deadline is True


def test_higher_cpu_frequency_reduces_latency(
    example_task: ComputationTask,
    example_resources: DirectLinkResources,
) -> None:
    """A higher CPU frequency should reduce computation latency."""
    faster_resources = DirectLinkResources(
        uplink_rate_bps=example_resources.uplink_rate_bps,
        downlink_rate_bps=example_resources.downlink_rate_bps,
        cpu_frequency_hz=4.0e9,
        vehicle_tx_power_w=example_resources.vehicle_tx_power_w,
        node_tx_power_w=example_resources.node_tx_power_w,
    )

    original_latency = direct_task_latency_s(
        task=example_task,
        resources=example_resources,
    )
    faster_latency = direct_task_latency_s(
        task=example_task,
        resources=faster_resources,
    )

    assert faster_latency < original_latency

    # New computation time:
    # 2e9 / 4e9 = 0.5 s
    #
    # New total:
    # 2.0 + 0.5 + 0.4 = 2.9 s
    assert faster_latency == pytest.approx(2.9)


def test_higher_cpu_frequency_increases_energy(
    example_task: ComputationTask,
    example_resources: DirectLinkResources,
) -> None:
    """A higher CPU frequency should increase dynamic CPU energy."""
    faster_resources = DirectLinkResources(
        uplink_rate_bps=example_resources.uplink_rate_bps,
        downlink_rate_bps=example_resources.downlink_rate_bps,
        cpu_frequency_hz=4.0e9,
        vehicle_tx_power_w=example_resources.vehicle_tx_power_w,
        node_tx_power_w=example_resources.node_tx_power_w,
    )

    original_energy = direct_task_energy_j(
        task=example_task,
        resources=example_resources,
    )
    faster_energy = direct_task_energy_j(
        task=example_task,
        resources=faster_resources,
    )

    assert faster_energy > original_energy

    # Doubling f makes the CPU-energy term four times larger:
    #
    # original CPU energy = 4 J
    # new CPU energy = 16 J
    #
    # Total = 0.4 + 16 + 0.2 = 16.6 J
    assert faster_energy == pytest.approx(16.6)


@pytest.mark.parametrize(
    ("data_bits", "rate_bps"),
    [
        (-1.0, 1.0e6),
        (1.0e6, 0.0),
        (1.0e6, -1.0),
        (float("inf"), 1.0e6),
        (1.0e6, float("nan")),
    ],
)
def test_transmission_time_rejects_invalid_parameters(
    data_bits: float,
    rate_bps: float,
) -> None:
    """Invalid data sizes and rates should raise ValueError."""
    with pytest.raises(ValueError):
        transmission_time_s(
            data_bits=data_bits,
            rate_bps=rate_bps,
        )


@pytest.mark.parametrize(
    ("cpu_cycles", "cpu_frequency_hz"),
    [
        (-1.0, 2.0e9),
        (2.0e9, 0.0),
        (2.0e9, -1.0),
        (float("nan"), 2.0e9),
        (2.0e9, float("inf")),
    ],
)
def test_computation_time_rejects_invalid_parameters(
    cpu_cycles: float,
    cpu_frequency_hz: float,
) -> None:
    """Invalid cycle counts and CPU frequencies should fail."""
    with pytest.raises(ValueError):
        computation_time_s(
            cpu_cycles=cpu_cycles,
            cpu_frequency_hz=cpu_frequency_hz,
        )


@pytest.mark.parametrize(
    ("input_bits", "cpu_cycles", "max_latency_s", "output_ratio"),
    [
        (-1.0, 1.0e9, 1.0, 0.1),
        (1.0e6, -1.0, 1.0, 0.1),
        (1.0e6, 1.0e9, 0.0, 0.1),
        (1.0e6, 1.0e9, -1.0, 0.1),
        (1.0e6, 1.0e9, 1.0, -0.1),
    ],
)
def test_computation_task_rejects_invalid_parameters(
    input_bits: float,
    cpu_cycles: float,
    max_latency_s: float,
    output_ratio: float,
) -> None:
    """ComputationTask should reject physically invalid values."""
    with pytest.raises(ValueError):
        ComputationTask(
            input_bits=input_bits,
            cpu_cycles=cpu_cycles,
            max_latency_s=max_latency_s,
            output_ratio=output_ratio,
        )


@pytest.mark.parametrize(
    (
        "uplink_rate_bps",
        "downlink_rate_bps",
        "cpu_frequency_hz",
        "vehicle_tx_power_w",
        "node_tx_power_w",
    ),
    [
        (0.0, 2.0e6, 2.0e9, 0.2, 0.5),
        (4.0e6, 0.0, 2.0e9, 0.2, 0.5),
        (4.0e6, 2.0e6, 0.0, 0.2, 0.5),
        (4.0e6, 2.0e6, 2.0e9, -0.2, 0.5),
        (4.0e6, 2.0e6, 2.0e9, 0.2, -0.5),
    ],
)
def test_direct_resources_reject_invalid_parameters(
    uplink_rate_bps: float,
    downlink_rate_bps: float,
    cpu_frequency_hz: float,
    vehicle_tx_power_w: float,
    node_tx_power_w: float,
) -> None:
    """DirectLinkResources should reject invalid resources."""
    with pytest.raises(ValueError):
        DirectLinkResources(
            uplink_rate_bps=uplink_rate_bps,
            downlink_rate_bps=downlink_rate_bps,
            cpu_frequency_hz=cpu_frequency_hz,
            vehicle_tx_power_w=vehicle_tx_power_w,
            node_tx_power_w=node_tx_power_w,
        )