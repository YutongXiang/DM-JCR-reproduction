"""Tests for equations (4) and (5) in dm_jcr.mobility."""

import numpy as np
import pytest

from dm_jcr.mobility import (
    MobilityState,
    SimulationBounds,
    clip_position_to_area,
    sample_mobility_noise,
    update_position,
    update_uav_position,
    update_vehicle_position,
)


def test_update_position_without_noise() -> None:
    """Verify l_next = l + velocity * delta_t."""
    result = update_position(
        position_m=[10.0, 20.0, 30.0],
        velocity_mps=[4.0, -2.0, 1.0],
        time_step_s=0.5,
    )

    np.testing.assert_allclose(result, [12.0, 19.0, 30.5])


def test_update_position_includes_mobility_noise() -> None:
    """Verify that omega is added after the deterministic displacement."""
    result = update_position(
        position_m=[10.0, 20.0, 30.0],
        velocity_mps=[4.0, -2.0, 1.0],
        time_step_s=0.5,
        noise_m=[0.2, -0.3, 0.5],
    )

    np.testing.assert_allclose(result, [12.2, 18.7, 31.0])


def test_zero_velocity_and_noise_keep_position_fixed() -> None:
    """A stationary node should remain at its current position."""
    result = update_position(
        position_m=[100.0, 200.0, 50.0],
        velocity_mps=[0.0, 0.0, 0.0],
        time_step_s=1.0,
        noise_m=[0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(result, [100.0, 200.0, 50.0])


def test_vehicle_is_constrained_to_ground_plane() -> None:
    """A vehicle must keep z = 0 in equation (4)."""
    state = MobilityState(
        position_m=[10.0, 20.0, 5.0],
        velocity_mps=[3.0, 4.0, 2.0],
    )

    result = update_vehicle_position(
        state,
        time_step_s=2.0,
        noise_m=[0.5, -0.5, 10.0],
    )

    np.testing.assert_allclose(result.position_m, [16.5, 27.5, 0.0])
    np.testing.assert_allclose(result.velocity_mps, [3.0, 4.0, 0.0])


def test_uav_moves_in_three_dimensions() -> None:
    """A UAV may change altitude in equation (5)."""
    state = MobilityState(
        position_m=[100.0, 200.0, 80.0],
        velocity_mps=[5.0, -2.0, 3.0],
    )

    result = update_uav_position(
        state,
        time_step_s=2.0,
        noise_m=[1.0, 0.5, -0.5],
    )

    np.testing.assert_allclose(
        result.position_m,
        [111.0, 196.5, 85.5],
    )
    np.testing.assert_allclose(result.velocity_mps, state.velocity_mps)


def test_position_is_clipped_to_simulation_bounds() -> None:
    """Nodes that move outside the area should be placed on its boundary."""
    bounds = SimulationBounds(
        minimum_m=[0.0, 0.0, 20.0],
        maximum_m=[1000.0, 1000.0, 120.0],
    )

    result = clip_position_to_area(
        position_m=[-10.0, 1100.0, 150.0],
        bounds=bounds,
    )

    np.testing.assert_allclose(result, [0.0, 1000.0, 120.0])


def test_update_uav_position_applies_bounds() -> None:
    """Boundary handling should also work during a UAV update."""
    state = MobilityState(
        position_m=[990.0, 500.0, 119.0],
        velocity_mps=[20.0, 0.0, 5.0],
    )
    bounds = SimulationBounds(
        minimum_m=[0.0, 0.0, 20.0],
        maximum_m=[1000.0, 1000.0, 120.0],
    )

    result = update_uav_position(
        state,
        time_step_s=1.0,
        bounds=bounds,
    )

    np.testing.assert_allclose(result.position_m, [1000.0, 500.0, 120.0])


def test_noise_sampling_is_reproducible_with_a_seed() -> None:
    """The same seed must produce the same mobility-noise sequence."""
    first_rng = np.random.default_rng(2026)
    second_rng = np.random.default_rng(2026)

    first = sample_mobility_noise(0.5, rng=first_rng)
    second = sample_mobility_noise(0.5, rng=second_rng)

    np.testing.assert_allclose(first, second)


def test_horizontal_noise_has_zero_vertical_component() -> None:
    """Ground-vehicle mobility noise should not change altitude."""
    noise = sample_mobility_noise(
        [1.0, 1.0, 5.0],
        rng=np.random.default_rng(7),
        horizontal_only=True,
    )

    assert noise[2] == pytest.approx(0.0)


def test_update_does_not_modify_caller_arrays() -> None:
    """Position updates should return new arrays rather than mutate inputs."""
    position = np.array([10.0, 20.0, 30.0])
    velocity = np.array([1.0, 2.0, 3.0])
    original_position = position.copy()
    original_velocity = velocity.copy()

    update_position(position, velocity, time_step_s=1.0)

    np.testing.assert_array_equal(position, original_position)
    np.testing.assert_array_equal(velocity, original_velocity)


@pytest.mark.parametrize("time_step_s", [0.0, -1.0, np.inf, np.nan])
def test_invalid_time_step_is_rejected(time_step_s: float) -> None:
    """A simulation time slot must have a finite positive duration."""
    with pytest.raises(ValueError):
        update_position(
            position_m=[0.0, 0.0, 0.0],
            velocity_mps=[1.0, 0.0, 0.0],
            time_step_s=time_step_s,
        )


@pytest.mark.parametrize(
    "bad_vector",
    [
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, np.nan, 3.0],
        [1.0, np.inf, 3.0],
    ],
)
def test_invalid_position_vector_is_rejected(bad_vector: list[float]) -> None:
    """Mobility vectors must contain exactly three finite values."""
    with pytest.raises(ValueError):
        MobilityState(
            position_m=bad_vector,
            velocity_mps=[0.0, 0.0, 0.0],
        )


def test_invalid_bounds_are_rejected() -> None:
    """Every upper boundary must be at least its lower boundary."""
    with pytest.raises(ValueError):
        SimulationBounds(
            minimum_m=[0.0, 0.0, 20.0],
            maximum_m=[1000.0, 1000.0, 10.0],
        )


def test_vehicle_bounds_must_include_ground_plane() -> None:
    """Vehicle movement is invalid if the configured area excludes z = 0."""
    state = MobilityState(
        position_m=[0.0, 0.0, 0.0],
        velocity_mps=[1.0, 0.0, 0.0],
    )
    uav_only_bounds = SimulationBounds(
        minimum_m=[0.0, 0.0, 20.0],
        maximum_m=[1000.0, 1000.0, 120.0],
    )

    with pytest.raises(ValueError, match="ground plane"):
        update_vehicle_position(
            state,
            time_step_s=1.0,
            bounds=uav_only_bounds,
        )


@pytest.mark.parametrize(
    "std_m",
    [
        -1.0,
        np.inf,
        [1.0, -1.0, 1.0],
        [1.0, 2.0],
    ],
)
def test_invalid_noise_standard_deviation_is_rejected(std_m: object) -> None:
    """Noise standard deviations must be finite and non-negative."""
    with pytest.raises(ValueError):
        sample_mobility_noise(std_m)
