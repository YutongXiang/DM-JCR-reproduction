import numpy as np
import pytest

from dm_jcr.channel import (
    achievable_rate_bps,
    calculate_sinr,
    db_to_linear,
    dbm_to_watts,
    distance_3d,
    free_space_channel_gain,
    noise_psd_w_per_hz,
)


def test_db_to_linear():
    assert db_to_linear(0.0) == pytest.approx(1.0)
    assert db_to_linear(10.0) == pytest.approx(10.0)


def test_dbm_to_watts():
    assert dbm_to_watts(30.0) == pytest.approx(1.0)
    assert dbm_to_watts(0.0) == pytest.approx(0.001)

    # 23 dBm 约等于 0.2 W
    assert dbm_to_watts(23.0) == pytest.approx(
        0.199526,
        rel=1e-5,
    )


def test_noise_psd_conversion():
    expected = 10.0 ** ((-174.0 - 30.0) / 10.0)

    assert noise_psd_w_per_hz() == pytest.approx(
        expected
    )


def test_distance_3d():
    point_a = np.array([0.0, 0.0, 0.0])
    point_b = np.array([3.0, 4.0, 12.0])

    assert distance_3d(
        point_a,
        point_b,
    ) == pytest.approx(13.0)


def test_unavailable_link_has_zero_gain():
    gain = free_space_channel_gain(
        vehicle_position=np.array([0.0, 0.0, 0.0]),
        node_position=np.array([100.0, 0.0, 0.0]),
        vehicle_gain_db=5.0,
        node_gain_db=5.0,
        link_available=False,
    )

    assert gain == 0.0


def test_channel_gain_matches_formula():
    distance_m = 100.0
    wavelength_m = 0.1

    gain = free_space_channel_gain(
        vehicle_position=np.array([0.0, 0.0, 0.0]),
        node_position=np.array(
            [distance_m, 0.0, 0.0]
        ),
        vehicle_gain_db=5.0,
        node_gain_db=5.0,
        wavelength_m=wavelength_m,
        shadow_factor=1.0,
        fading_power_gain=1.0,
    )

    expected = (
        db_to_linear(5.0)
        * db_to_linear(5.0)
        * wavelength_m**2
        / (4.0 * np.pi * distance_m) ** 2
    )

    assert gain == pytest.approx(expected)


def test_channel_gain_obeys_inverse_square_law():
    common_arguments = {
        "vehicle_position": np.array(
            [0.0, 0.0, 0.0]
        ),
        "vehicle_gain_db": 5.0,
        "node_gain_db": 5.0,
    }

    gain_100m = free_space_channel_gain(
        node_position=np.array([100.0, 0.0, 0.0]),
        **common_arguments,
    )

    gain_200m = free_space_channel_gain(
        node_position=np.array([200.0, 0.0, 0.0]),
        **common_arguments,
    )

    assert gain_100m / gain_200m == pytest.approx(
        4.0
    )


def test_sinr_matches_formula():
    bandwidth_hz = 10.0e6
    transmit_power_w = 0.2
    channel_gain = 1.0e-7
    noise_psd = 4.0e-21
    interference_w = 1.0e-10

    actual = calculate_sinr(
        bandwidth_hz=bandwidth_hz,
        transmit_power_w=transmit_power_w,
        channel_gain=channel_gain,
        noise_psd_w_hz=noise_psd,
        interference_power_w=interference_w,
    )

    expected = (
        transmit_power_w * channel_gain
        / (
            noise_psd * bandwidth_hz
            + interference_w
        )
    )

    assert actual == pytest.approx(expected)


def test_achievable_rate_matches_equation_8():
    bandwidth_hz = 10.0e6
    transmit_power_w = 0.2
    channel_gain = 1.0e-7
    noise_psd = 4.0e-21
    interference_w = 1.0e-10

    sinr = (
        transmit_power_w * channel_gain
        / (
            noise_psd * bandwidth_hz
            + interference_w
        )
    )

    expected_rate = (
        bandwidth_hz * np.log2(1.0 + sinr)
    )

    actual_rate = achievable_rate_bps(
        bandwidth_hz=bandwidth_hz,
        transmit_power_w=transmit_power_w,
        channel_gain=channel_gain,
        noise_psd_w_hz=noise_psd,
        interference_power_w=interference_w,
    )

    assert actual_rate == pytest.approx(expected_rate)


def test_more_interference_reduces_rate():
    common_arguments = {
        "bandwidth_hz": 10.0e6,
        "transmit_power_w": 0.2,
        "channel_gain": 1.0e-7,
        "noise_psd_w_hz": 4.0e-21,
    }

    rate_without_interference = achievable_rate_bps(
        interference_power_w=0.0,
        **common_arguments,
    )

    rate_with_interference = achievable_rate_bps(
        interference_power_w=1.0e-9,
        **common_arguments,
    )

    assert (
        rate_with_interference
        < rate_without_interference
    )


def test_zero_channel_gain_produces_zero_rate():
    rate = achievable_rate_bps(
        bandwidth_hz=10.0e6,
        transmit_power_w=0.2,
        channel_gain=0.0,
        noise_psd_w_hz=4.0e-21,
    )

    assert rate == pytest.approx(0.0)