import numpy as np
import pytest

from dm_jcr.channel import noise_psd_w_per_hz
from dm_jcr.random_channel import (
    sample_blockage,
    sample_random_channel,
    sample_rayleigh_power_gain,
    sample_rician_power_gain,
    sample_shadow_factor,
    sample_wireless_link,
    shadow_factor_from_loss_db,
)


def test_blockage_probability_is_about_03():
    rng = np.random.default_rng(1)

    blocked = sample_blockage(
        rng=rng,
        blockage_probability=0.3,
        size=100_000,
    )

    assert blocked.mean() == pytest.approx(
        0.3,
        abs=0.01,
    )


def test_rayleigh_mean_power_is_one():
    rng = np.random.default_rng(2)

    gain = sample_rayleigh_power_gain(
        rng=rng,
        size=100_000,
    )

    assert gain.mean() == pytest.approx(
        1.0,
        abs=0.02,
    )


def test_rician_mean_power_is_one():
    rng = np.random.default_rng(3)

    gain = sample_rician_power_gain(
        rng=rng,
        k_factor_db=6.0,
        size=100_000,
    )

    assert gain.mean() == pytest.approx(
        1.0,
        abs=0.02,
    )


def test_shadow_loss_conversion():
    assert shadow_factor_from_loss_db(
        0.0
    ) == pytest.approx(1.0)

    assert shadow_factor_from_loss_db(
        10.0
    ) == pytest.approx(0.1)

    assert shadow_factor_from_loss_db(
        20.0
    ) == pytest.approx(0.01)


def test_blocked_shadow_is_stronger():
    rng = np.random.default_rng(4)

    los_factor = sample_shadow_factor(
        rng=rng,
        blocked=False,
    )

    blocked_factor = sample_shadow_factor(
        rng=rng,
        blocked=True,
    )

    assert 0.5 <= los_factor <= 1.0
    assert 0.001 <= blocked_factor <= 0.1
    assert blocked_factor < los_factor


def test_forced_los_uses_rician():
    sample = sample_random_channel(
        rng=np.random.default_rng(5),
        vehicle_position=np.array(
            [0.0, 0.0, 0.0]
        ),
        node_position=np.array(
            [100.0, 0.0, 100.0]
        ),
        vehicle_gain_db=5.0,
        node_gain_db=5.0,
        blockage_probability=0.0,
    )

    assert sample.blocked is False
    assert sample.fading_model == "rician"
    assert sample.channel_gain > 0.0


def test_forced_blockage_uses_rayleigh():
    sample = sample_random_channel(
        rng=np.random.default_rng(6),
        vehicle_position=np.array(
            [0.0, 0.0, 0.0]
        ),
        node_position=np.array(
            [100.0, 0.0, 100.0]
        ),
        vehicle_gain_db=5.0,
        node_gain_db=5.0,
        blockage_probability=1.0,
    )

    assert sample.blocked is True
    assert sample.fading_model == "rayleigh"
    assert sample.shadow_factor <= 0.1


def test_random_link_is_reproducible():
    arguments = {
        "vehicle_position": np.array(
            [0.0, 0.0, 0.0]
        ),
        "node_position": np.array(
            [100.0, 0.0, 100.0]
        ),
        "vehicle_gain_db": 5.0,
        "node_gain_db": 5.0,
        "bandwidth_hz": 10.0e6,
        "transmit_power_w": 0.2,
        "noise_psd_w_hz":
            noise_psd_w_per_hz(),
    }

    sample_a = sample_wireless_link(
        rng=np.random.default_rng(42),
        **arguments,
    )

    sample_b = sample_wireless_link(
        rng=np.random.default_rng(42),
        **arguments,
    )

    assert sample_a == sample_b


def test_random_link_produces_rate():
    sample = sample_wireless_link(
        rng=np.random.default_rng(7),
        vehicle_position=np.array(
            [0.0, 0.0, 0.0]
        ),
        node_position=np.array(
            [100.0, 0.0, 100.0]
        ),
        vehicle_gain_db=5.0,
        node_gain_db=5.0,
        bandwidth_hz=10.0e6,
        transmit_power_w=0.2,
        noise_psd_w_hz=noise_psd_w_per_hz(),
        blockage_probability=0.0,
    )

    assert sample.channel.channel_gain > 0.0
    assert sample.sinr > 0.0
    assert sample.rate_bps > 0.0
    assert sample.rate_mbps > 0.0


def test_unavailable_link_has_zero_rate():
    sample = sample_wireless_link(
        rng=np.random.default_rng(8),
        vehicle_position=np.array(
            [0.0, 0.0, 0.0]
        ),
        node_position=np.array(
            [100.0, 0.0, 100.0]
        ),
        vehicle_gain_db=5.0,
        node_gain_db=5.0,
        bandwidth_hz=10.0e6,
        transmit_power_w=0.2,
        noise_psd_w_hz=noise_psd_w_per_hz(),
        link_available=False,
    )

    assert sample.channel.channel_gain == 0.0
    assert sample.sinr == 0.0
    assert sample.rate_bps == 0.0