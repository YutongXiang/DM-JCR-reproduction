"""UAV-assisted vehicle-to-vehicle forwarding, equations (14) and (15).

Unlike equations (12) and (13), this model describes a V2V communication
task. The UAV receives data from the source vehicle, performs forwarding
processing such as decoding/error correction, and transmits the same data to
the destination vehicle. No edge-node task computation is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from dm_jcr.relay_model import forwarding_cpu_cycles
from dm_jcr.task_model import (
    DEFAULT_CPU_ENERGY_COEFFICIENT,
    computation_time_s,
    dynamic_cpu_energy_j,
    transmission_time_s,
)


def _validate_value(
    name: str,
    value: float,
    *,
    strictly_positive: bool,
) -> float:
    """Validate a finite scalar used by the V2V relay model."""
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and value <= 0.0:
        raise ValueError(f"{name} must be greater than 0")
    if not strictly_positive and value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class V2VTask:
    """Data-forwarding task sent from one vehicle to another.

    ``data_bits`` is :math:`D_{v,k}^t`; ``max_latency_s`` is the maximum
    tolerable latency :math:`T_{v,k}^{max,t}`.
    """

    data_bits: float
    max_latency_s: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "data_bits",
            _validate_value(
                "data_bits",
                self.data_bits,
                strictly_positive=False,
            ),
        )
        object.__setattr__(
            self,
            "max_latency_s",
            _validate_value(
                "max_latency_s",
                self.max_latency_s,
                strictly_positive=True,
            ),
        )


@dataclass(frozen=True)
class V2VRelayResources:
    """Communication and computing resources for UAV-assisted V2V relay."""

    source_to_uav_rate_bps: float
    uav_to_destination_rate_bps: float
    uav_cpu_frequency_hz: float

    source_vehicle_tx_power_w: float
    uav_tx_power_w: float

    forwarding_cycles_per_bit: float

    def __post_init__(self) -> None:
        positive = (
            "source_to_uav_rate_bps",
            "uav_to_destination_rate_bps",
            "uav_cpu_frequency_hz",
        )
        non_negative = (
            "source_vehicle_tx_power_w",
            "uav_tx_power_w",
            "forwarding_cycles_per_bit",
        )

        for name in positive:
            object.__setattr__(
                self,
                name,
                _validate_value(
                    name,
                    getattr(self, name),
                    strictly_positive=True,
                ),
            )
        for name in non_negative:
            object.__setattr__(
                self,
                name,
                _validate_value(
                    name,
                    getattr(self, name),
                    strictly_positive=False,
                ),
            )


@dataclass(frozen=True)
class V2VRelayMetrics:
    """Term-by-term latency and energy results for equations (14), (15)."""

    source_to_uav_time_s: float
    uav_to_destination_time_s: float
    forwarding_processing_time_s: float
    total_latency_s: float

    source_to_uav_energy_j: float
    uav_to_destination_energy_j: float
    forwarding_processing_energy_j: float
    total_energy_j: float

    meets_deadline: bool


def evaluate_v2v_relay(
    task: V2VTask,
    resources: V2VRelayResources,
    energy_coefficient: float = DEFAULT_CPU_ENERGY_COEFFICIENT,
    deadline_tolerance_s: float = 1.0e-12,
) -> V2VRelayMetrics:
    r"""Evaluate equations (14) and (15) term by term.

    Equation (14):

    .. math:: T = D/r_{v,u} + D/r_{u,v^*} + \phi D/f_u.

    Equation (15):

    .. math:: E = p_vD/r_{v,u} + p_uD/r_{u,v^*} + \zeta f_u^2\phi D.
    """
    energy_coefficient = _validate_value(
        "energy_coefficient",
        energy_coefficient,
        strictly_positive=False,
    )
    deadline_tolerance_s = _validate_value(
        "deadline_tolerance_s",
        deadline_tolerance_s,
        strictly_positive=False,
    )

    forwarding_cycles = forwarding_cpu_cycles(
        data_bits=task.data_bits,
        forwarding_cycles_per_bit=(
            resources.forwarding_cycles_per_bit
        ),
    )

    source_to_uav_time_s = transmission_time_s(
        data_bits=task.data_bits,
        rate_bps=resources.source_to_uav_rate_bps,
    )
    uav_to_destination_time_s = transmission_time_s(
        data_bits=task.data_bits,
        rate_bps=resources.uav_to_destination_rate_bps,
    )
    forwarding_processing_time_s = computation_time_s(
        cpu_cycles=forwarding_cycles,
        cpu_frequency_hz=resources.uav_cpu_frequency_hz,
    )

    total_latency_s = (
        source_to_uav_time_s
        + uav_to_destination_time_s
        + forwarding_processing_time_s
    )

    source_to_uav_energy_j = (
        resources.source_vehicle_tx_power_w
        * source_to_uav_time_s
    )
    uav_to_destination_energy_j = (
        resources.uav_tx_power_w
        * uav_to_destination_time_s
    )
    forwarding_processing_energy_j = dynamic_cpu_energy_j(
        cpu_cycles=forwarding_cycles,
        cpu_frequency_hz=resources.uav_cpu_frequency_hz,
        energy_coefficient=energy_coefficient,
    )

    total_energy_j = (
        source_to_uav_energy_j
        + uav_to_destination_energy_j
        + forwarding_processing_energy_j
    )

    return V2VRelayMetrics(
        source_to_uav_time_s=float(source_to_uav_time_s),
        uav_to_destination_time_s=float(uav_to_destination_time_s),
        forwarding_processing_time_s=float(
            forwarding_processing_time_s
        ),
        total_latency_s=float(total_latency_s),
        source_to_uav_energy_j=float(source_to_uav_energy_j),
        uav_to_destination_energy_j=float(
            uav_to_destination_energy_j
        ),
        forwarding_processing_energy_j=float(
            forwarding_processing_energy_j
        ),
        total_energy_j=float(total_energy_j),
        meets_deadline=(
            total_latency_s
            <= task.max_latency_s + deadline_tolerance_s
        ),
    )


def v2v_relay_latency_s(
    task: V2VTask,
    resources: V2VRelayResources,
) -> float:
    """Return only the total latency from equation (14)."""
    return evaluate_v2v_relay(task, resources).total_latency_s


def v2v_relay_energy_j(
    task: V2VTask,
    resources: V2VRelayResources,
    energy_coefficient: float = DEFAULT_CPU_ENERGY_COEFFICIENT,
) -> float:
    """Return only the total energy from equation (15)."""
    return evaluate_v2v_relay(
        task,
        resources,
        energy_coefficient=energy_coefficient,
    ).total_energy_j


__all__ = [
    "V2VRelayMetrics",
    "V2VRelayResources",
    "V2VTask",
    "evaluate_v2v_relay",
    "v2v_relay_energy_j",
    "v2v_relay_latency_s",
]
