"""
公式(12)、(13) 的直接链路任务时延与能耗计算函数。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


# The value used in the paper's computation-energy model.
DEFAULT_CPU_ENERGY_COEFFICIENT = 5.0e-28


def _validate_finite(
    name: str,
    value: float,
    *,
    strictly_positive: bool,
) -> float:
    """Convert a value to float and validate its range.

    Parameters
    ----------
    name:
        Parameter name used in error messages.
    value:
        Numeric value to validate.
    strictly_positive:
        If True, require value > 0.
        Otherwise, require value >= 0.

    Returns
    -------
    float
        The validated floating-point value.
    """
    value = float(value)

    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")

    if strictly_positive:
        if value <= 0.0:
            raise ValueError(f"{name} must be greater than 0, got {value!r}")
    elif value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")

    return value


def kilobytes_to_bits(size_kb: float) -> float:
    """Convert decimal kilobytes to bits.

    This reproduction uses:

        1 KB = 1000 Byte
        1 Byte = 8 bit

    Therefore:

        number of bits = size_kb * 1000 * 8
    """
    size_kb = _validate_finite(
        "size_kb",
        size_kb,
        strictly_positive=False,
    )
    return size_kb * 1000.0 * 8.0


@dataclass(frozen=True)
class ComputationTask:
    """Description of one computation task.

    Parameters
    ----------
    input_bits:
        Input task size D, in bits.
    cpu_cycles:
        Total number of CPU cycles C required to execute the task.
    max_latency_s:
        Maximum allowed latency T_max, in seconds.
    output_ratio:
        Output/input data-size ratio mu. The output size is mu * D.
    """

    input_bits: float
    cpu_cycles: float
    max_latency_s: float
    output_ratio: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_bits",
            _validate_finite(
                "input_bits",
                self.input_bits,
                strictly_positive=False,
            ),
        )
        object.__setattr__(
            self,
            "cpu_cycles",
            _validate_finite(
                "cpu_cycles",
                self.cpu_cycles,
                strictly_positive=False,
            ),
        )
        object.__setattr__(
            self,
            "max_latency_s",
            _validate_finite(
                "max_latency_s",
                self.max_latency_s,
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "output_ratio",
            _validate_finite(
                "output_ratio",
                self.output_ratio,
                strictly_positive=False,
            ),
        )

    @property
    def output_bits(self) -> float:
        """Return the result-data size mu * D, in bits."""
        return self.output_ratio * self.input_bits


@dataclass(frozen=True)
class DirectLinkResources:
    """Communication and computing resources for direct execution.

    Parameters
    ----------
    uplink_rate_bps:
        Vehicle-to-node rate r_vn, in bit/s.
    downlink_rate_bps:
        Node-to-vehicle rate r_nv, in bit/s.
    cpu_frequency_hz:
        CPU frequency f_n allocated by the edge node, in cycle/s.
    vehicle_tx_power_w:
        Vehicle transmission power p_v, in watts.
    node_tx_power_w:
        Edge-node transmission power p_n, in watts.
    """

    uplink_rate_bps: float
    downlink_rate_bps: float
    cpu_frequency_hz: float
    vehicle_tx_power_w: float
    node_tx_power_w: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "uplink_rate_bps",
            _validate_finite(
                "uplink_rate_bps",
                self.uplink_rate_bps,
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "downlink_rate_bps",
            _validate_finite(
                "downlink_rate_bps",
                self.downlink_rate_bps,
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "cpu_frequency_hz",
            _validate_finite(
                "cpu_frequency_hz",
                self.cpu_frequency_hz,
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "vehicle_tx_power_w",
            _validate_finite(
                "vehicle_tx_power_w",
                self.vehicle_tx_power_w,
                strictly_positive=False,
            ),
        )
        object.__setattr__(
            self,
            "node_tx_power_w",
            _validate_finite(
                "node_tx_power_w",
                self.node_tx_power_w,
                strictly_positive=False,
            ),
        )


@dataclass(frozen=True)
class DirectTaskEvaluation:
    """Detailed result of evaluating one direct-link task."""

    upload_latency_s: float
    computation_latency_s: float
    download_latency_s: float
    total_latency_s: float

    upload_energy_j: float
    computation_energy_j: float
    download_energy_j: float
    total_energy_j: float

    meets_deadline: bool


def transmission_time_s(
    data_bits: float,
    rate_bps: float,
) -> float:
    """Calculate transmission time using T = D / r.

    Parameters
    ----------
    data_bits:
        Data size D, in bits.
    rate_bps:
        Transmission rate r, in bit/s.

    Returns
    -------
    float
        Transmission time in seconds.
    """
    data_bits = _validate_finite(
        "data_bits",
        data_bits,
        strictly_positive=False,
    )
    rate_bps = _validate_finite(
        "rate_bps",
        rate_bps,
        strictly_positive=True,
    )

    return data_bits / rate_bps


def computation_time_s(
    cpu_cycles: float,
    cpu_frequency_hz: float,
) -> float:
    """Calculate computation time using T = C / f.

    Although the parameter name contains ``hz``, the intended unit is
    CPU cycles per second.

    Parameters
    ----------
    cpu_cycles:
        Required CPU cycles C.
    cpu_frequency_hz:
        Allocated CPU frequency f, in cycle/s.

    Returns
    -------
    float
        Computation time in seconds.
    """
    cpu_cycles = _validate_finite(
        "cpu_cycles",
        cpu_cycles,
        strictly_positive=False,
    )
    cpu_frequency_hz = _validate_finite(
        "cpu_frequency_hz",
        cpu_frequency_hz,
        strictly_positive=True,
    )

    return cpu_cycles / cpu_frequency_hz


def dynamic_cpu_energy_j(
    cpu_cycles: float,
    cpu_frequency_hz: float,
    energy_coefficient: float = DEFAULT_CPU_ENERGY_COEFFICIENT,
) -> float:
    """Calculate dynamic CPU energy using E = zeta * f^2 * C.

    Parameters
    ----------
    cpu_cycles:
        Required CPU cycles C.
    cpu_frequency_hz:
        CPU frequency f, in cycle/s.
    energy_coefficient:
        Hardware-dependent coefficient zeta.

    Returns
    -------
    float
        Dynamic CPU energy in joules.
    """
    cpu_cycles = _validate_finite(
        "cpu_cycles",
        cpu_cycles,
        strictly_positive=False,
    )
    cpu_frequency_hz = _validate_finite(
        "cpu_frequency_hz",
        cpu_frequency_hz,
        strictly_positive=True,
    )
    energy_coefficient = _validate_finite(
        "energy_coefficient",
        energy_coefficient,
        strictly_positive=False,
    )

    return (
        energy_coefficient
        * cpu_frequency_hz**2
        * cpu_cycles
    )


def direct_task_latency_s(
    task: ComputationTask,
    resources: DirectLinkResources,
) -> float:
    """Calculate direct-link total latency from equation (12).

    The direct-link branch is:

        T_direct
            = D / r_vn
            + C / f_n
            + mu * D / r_nv

    Returns
    -------
    float
        Total task latency in seconds.
    """
    upload_latency_s = transmission_time_s(
        task.input_bits,
        resources.uplink_rate_bps,
    )

    computation_latency_s = computation_time_s(
        task.cpu_cycles,
        resources.cpu_frequency_hz,
    )

    download_latency_s = transmission_time_s(
        task.output_bits,
        resources.downlink_rate_bps,
    )

    return (
        upload_latency_s
        + computation_latency_s
        + download_latency_s
    )


def direct_task_energy_j(
    task: ComputationTask,
    resources: DirectLinkResources,
    energy_coefficient: float = DEFAULT_CPU_ENERGY_COEFFICIENT,
) -> float:
    """Calculate direct-link total energy from equation (13).

    The direct-link branch is:

        E_direct
            = p_v * D / r_vn
            + zeta * f_n^2 * C
            + p_n * mu * D / r_nv

    Returns
    -------
    float
        Total energy consumption in joules.
    """
    upload_latency_s = transmission_time_s(
        task.input_bits,
        resources.uplink_rate_bps,
    )
    upload_energy_j = (
        resources.vehicle_tx_power_w
        * upload_latency_s
    )

    computation_energy_j = dynamic_cpu_energy_j(
        task.cpu_cycles,
        resources.cpu_frequency_hz,
        energy_coefficient,
    )

    download_latency_s = transmission_time_s(
        task.output_bits,
        resources.downlink_rate_bps,
    )
    download_energy_j = (
        resources.node_tx_power_w
        * download_latency_s
    )

    return (
        upload_energy_j
        + computation_energy_j
        + download_energy_j
    )


def evaluate_direct_task(
    task: ComputationTask,
    resources: DirectLinkResources,
    energy_coefficient: float = DEFAULT_CPU_ENERGY_COEFFICIENT,
    deadline_tolerance_s: float = 1.0e-12,
) -> DirectTaskEvaluation:
    """Evaluate latency, energy and deadline feasibility together.

    Parameters
    ----------
    task:
        Computation task to evaluate.
    resources:
        Direct-link communication and computing resources.
    energy_coefficient:
        CPU energy coefficient zeta.
    deadline_tolerance_s:
        Small numerical tolerance used when comparing the calculated
        latency with T_max.

    Returns
    -------
    DirectTaskEvaluation
        Detailed latency and energy breakdown.
    """
    energy_coefficient = _validate_finite(
        "energy_coefficient",
        energy_coefficient,
        strictly_positive=False,
    )
    deadline_tolerance_s = _validate_finite(
        "deadline_tolerance_s",
        deadline_tolerance_s,
        strictly_positive=False,
    )

    # Equation (12): latency components.
    upload_latency_s = transmission_time_s(
        task.input_bits,
        resources.uplink_rate_bps,
    )
    computation_latency_s = computation_time_s(
        task.cpu_cycles,
        resources.cpu_frequency_hz,
    )
    download_latency_s = transmission_time_s(
        task.output_bits,
        resources.downlink_rate_bps,
    )

    total_latency_s = (
        upload_latency_s
        + computation_latency_s
        + download_latency_s
    )

    # Equation (13): energy components.
    upload_energy_j = (
        resources.vehicle_tx_power_w
        * upload_latency_s
    )

    computation_energy_j = dynamic_cpu_energy_j(
        task.cpu_cycles,
        resources.cpu_frequency_hz,
        energy_coefficient,
    )

    download_energy_j = (
        resources.node_tx_power_w
        * download_latency_s
    )

    total_energy_j = (
        upload_energy_j
        + computation_energy_j
        + download_energy_j
    )

    meets_deadline = (
        total_latency_s
        <= task.max_latency_s + deadline_tolerance_s
    )

    return DirectTaskEvaluation(
        upload_latency_s=upload_latency_s,
        computation_latency_s=computation_latency_s,
        download_latency_s=download_latency_s,
        total_latency_s=total_latency_s,
        upload_energy_j=upload_energy_j,
        computation_energy_j=computation_energy_j,
        download_energy_j=download_energy_j,
        total_energy_j=total_energy_j,
        meets_deadline=meets_deadline,
    )


__all__ = [
    "DEFAULT_CPU_ENERGY_COEFFICIENT",
    "ComputationTask",
    "DirectLinkResources",
    "DirectTaskEvaluation",
    "kilobytes_to_bits",
    "transmission_time_s",
    "computation_time_s",
    "dynamic_cpu_energy_j",
    "direct_task_latency_s",
    "direct_task_energy_j",
    "evaluate_direct_task",
]