"""
公式(4) 和 (5) 的移动模型实现。
Mobility models used by the DM-JCR system simulator.

This module implements the position evolution equations from the paper:

    l_v^(t+1) = l_v^t + sp_v^t * delta_t + omega_v^t    (4)
    l_u^(t+1) = l_u^t + sp_u^t * delta_t + omega_u^t    (5)

Vehicles are constrained to the ground plane (z = 0), while UAVs move in
three-dimensional space.  All positions are measured in metres, velocities
in metres per second, and time-slot lengths in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


Vector3: TypeAlias = NDArray[np.float64]


def _as_vector3(name: str, value: ArrayLike) -> Vector3:
    """Validate and copy a finite three-dimensional vector."""
    vector = np.asarray(value, dtype=np.float64)

    if vector.shape != (3,):
        raise ValueError(
            f"{name} must have shape (3,), got {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")

    return vector.copy()


def _validate_positive(name: str, value: float) -> float:
    """Validate a finite scalar that must be greater than zero."""
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and greater than 0")
    return value


@dataclass(frozen=True)
class MobilityState:
    """Position and velocity of one vehicle or UAV.

    Parameters
    ----------
    position_m:
        Current position ``[x, y, z]`` in metres.
    velocity_mps:
        Current velocity ``[vx, vy, vz]`` in metres per second.

    Notes
    -----
    Input arrays are copied so that updating a state does not modify the
    arrays supplied by the caller.
    """

    position_m: Vector3
    velocity_mps: Vector3

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_m",
            _as_vector3("position_m", self.position_m),
        )
        object.__setattr__(
            self,
            "velocity_mps",
            _as_vector3("velocity_mps", self.velocity_mps),
        )


@dataclass(frozen=True)
class SimulationBounds:
    """Axis-aligned boundary of the simulated region.

    A position outside the region is clipped independently along x, y and z.
    Equal lower and upper bounds are allowed, for example ``z_min = z_max = 0``
    for a purely two-dimensional road region.
    """

    minimum_m: Vector3
    maximum_m: Vector3

    def __post_init__(self) -> None:
        minimum = _as_vector3("minimum_m", self.minimum_m)
        maximum = _as_vector3("maximum_m", self.maximum_m)

        if np.any(maximum < minimum):
            raise ValueError(
                "maximum_m must be greater than or equal to minimum_m "
                "on every axis"
            )

        object.__setattr__(self, "minimum_m", minimum)
        object.__setattr__(self, "maximum_m", maximum)


def sample_mobility_noise(
    std_m: float | ArrayLike,
    *,
    rng: np.random.Generator | None = None,
    horizontal_only: bool = False,
) -> Vector3:
    """Sample the mobility-noise vector omega.

    The paper introduces mobility noise but does not specify its exact
    distribution.  This reproduction uses independent zero-mean Gaussian
    noise on each axis:

        omega ~ Normal(0, sigma^2)

    Parameters
    ----------
    std_m:
        Standard deviation in metres.  It may be one scalar for all axes or
        a vector ``[sigma_x, sigma_y, sigma_z]``.
    rng:
        NumPy random-number generator.  Pass ``np.random.default_rng(seed)``
        to make an experiment reproducible.
    horizontal_only:
        If true, force the z component of the noise to zero.  This is useful
        for ground vehicles.
    """
    standard_deviation = np.asarray(std_m, dtype=np.float64)

    if standard_deviation.ndim == 0:
        standard_deviation = np.full(3, float(standard_deviation))
    elif standard_deviation.shape == (3,):
        standard_deviation = standard_deviation.copy()
    else:
        raise ValueError("std_m must be a scalar or have shape (3,)")

    if (
        not np.all(np.isfinite(standard_deviation))
        or np.any(standard_deviation < 0.0)
    ):
        raise ValueError("std_m must contain finite, non-negative values")

    if horizontal_only:
        standard_deviation[2] = 0.0

    generator = rng if rng is not None else np.random.default_rng()
    return generator.normal(
        loc=0.0,
        scale=standard_deviation,
        size=3,
    ).astype(np.float64)


def clip_position_to_area(
    position_m: ArrayLike,
    bounds: SimulationBounds,
) -> Vector3:
    """Clip a three-dimensional position to the simulation region."""
    position = _as_vector3("position_m", position_m)
    return np.clip(
        position,
        bounds.minimum_m,
        bounds.maximum_m,
    )


def update_position(
    position_m: ArrayLike,
    velocity_mps: ArrayLike,
    time_step_s: float,
    *,
    noise_m: ArrayLike | None = None,
    bounds: SimulationBounds | None = None,
) -> Vector3:
    """Apply the common position-update rule in equations (4) and (5).

    The implemented equation is:

        next_position = position + velocity * time_step + noise

    This function is intentionally independent of node type.  Use
    :func:`update_vehicle_position` or :func:`update_uav_position` when the
    ground-plane or UAV-specific behavior is required.
    """
    position = _as_vector3("position_m", position_m)
    velocity = _as_vector3("velocity_mps", velocity_mps)
    time_step = _validate_positive("time_step_s", time_step_s)

    if noise_m is None:
        noise = np.zeros(3, dtype=np.float64)
    else:
        noise = _as_vector3("noise_m", noise_m)

    next_position = position + velocity * time_step + noise

    if bounds is not None:
        next_position = clip_position_to_area(next_position, bounds)

    return next_position


def update_vehicle_position(
    state: MobilityState,
    time_step_s: float,
    *,
    noise_m: ArrayLike | None = None,
    bounds: SimulationBounds | None = None,
) -> MobilityState:
    """Advance a ground vehicle by one time slot using equation (4).

    The vehicle's z-position and vertical velocity are always forced to zero,
    even if a non-zero z component is accidentally supplied in the state or
    noise vector.
    """
    if bounds is not None and not (
        bounds.minimum_m[2] <= 0.0 <= bounds.maximum_m[2]
    ):
        raise ValueError("vehicle bounds must include the ground plane z = 0")

    position = state.position_m.copy()
    velocity = state.velocity_mps.copy()
    position[2] = 0.0
    velocity[2] = 0.0

    if noise_m is None:
        noise = np.zeros(3, dtype=np.float64)
    else:
        noise = _as_vector3("noise_m", noise_m)
        noise[2] = 0.0

    next_position = update_position(
        position,
        velocity,
        time_step_s,
        noise_m=noise,
        bounds=bounds,
    )
    next_position[2] = 0.0

    return MobilityState(
        position_m=next_position,
        velocity_mps=velocity,
    )


def update_uav_position(
    state: MobilityState,
    time_step_s: float,
    *,
    noise_m: ArrayLike | None = None,
    bounds: SimulationBounds | None = None,
) -> MobilityState:
    """Advance a UAV by one time slot in 3D using equation (5)."""
    next_position = update_position(
        state.position_m,
        state.velocity_mps,
        time_step_s,
        noise_m=noise_m,
        bounds=bounds,
    )

    return MobilityState(
        position_m=next_position,
        velocity_mps=state.velocity_mps,
    )


__all__ = [
    "MobilityState",
    "SimulationBounds",
    "clip_position_to_area",
    "sample_mobility_noise",
    "update_position",
    "update_uav_position",
    "update_vehicle_position",
]

