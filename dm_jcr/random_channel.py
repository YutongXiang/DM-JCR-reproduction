"""
将公式(7)接入到公式(8)的端到端计算函数。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dm_jcr.channel import (
    achievable_rate_bps,
    calculate_sinr,
    distance_3d,
    free_space_channel_gain,
)


@dataclass(frozen=True)
class ChannelSample:
    """
    保存一次公式 (7) 随机信道采样结果。
    """

    distance_m: float
    blocked: bool
    shadow_factor: float
    fading_power_gain: float
    fading_model: str
    link_available: bool
    channel_gain: float


@dataclass(frozen=True)
class WirelessLinkSample:
    """
    保存公式 (7)、(8) 的端到端计算结果。
    """

    channel: ChannelSample
    sinr: float
    rate_bps: float

    @property
    def rate_mbps(self) -> float:
        return float(self.rate_bps / 1.0e6)


def sample_blockage(
    rng: np.random.Generator,
    blockage_probability: float = 0.3,
    size: int | tuple[int, ...] | None = None,
) -> bool | np.ndarray:
    """
    生成建筑遮挡状态。

    True 表示建筑遮挡/NLoS；
    False 表示无遮挡/LoS。
    """
    if not 0.0 <= blockage_probability <= 1.0:
        raise ValueError(
            "blockage_probability 必须位于 [0, 1]"
        )

    blocked = (
        rng.random(size=size)
        < blockage_probability
    )

    if size is None:
        return bool(blocked)

    return np.asarray(blocked, dtype=bool)


def sample_rayleigh_power_gain(
    rng: np.random.Generator,
    size: int | tuple[int, ...] | None = None,
) -> float | np.ndarray:
    """
    生成归一化 Rayleigh 功率增益。

    Rayleigh 幅度的平方服从均值为1的指数分布。
    """
    power_gain = rng.exponential(
        scale=1.0,
        size=size,
    )

    if size is None:
        return float(power_gain)

    return np.asarray(
        power_gain,
        dtype=np.float64,
    )


def sample_rician_power_gain(
    rng: np.random.Generator,
    k_factor_db: float = 6.0,
    size: int | tuple[int, ...] | None = None,
) -> float | np.ndarray:
    """
    生成归一化 Rician 功率增益。

    由直射分量和复高斯散射分量共同构成。
    """
    if k_factor_db < 0:
        raise ValueError("Rician K 因子不能为负")

    k_linear = 10.0 ** (k_factor_db / 10.0)

    los_amplitude = np.sqrt(
        k_linear / (k_linear + 1.0)
    )

    scatter_std = np.sqrt(
        1.0 / (2.0 * (k_linear + 1.0))
    )

    shape = () if size is None else size

    real_part = (
        los_amplitude
        + scatter_std * rng.normal(size=shape)
    )

    imaginary_part = (
        scatter_std * rng.normal(size=shape)
    )

    power_gain = (
        real_part**2 + imaginary_part**2
    )

    if size is None:
        return float(power_gain)

    return np.asarray(
        power_gain,
        dtype=np.float64,
    )


def shadow_factor_from_loss_db(
    loss_db: float | np.ndarray,
) -> float | np.ndarray:
    """
    将附加损耗从 dB 转换为线性衰减因子。

        shadow = 10^(-loss_db / 10)
    """
    loss = np.asarray(
        loss_db,
        dtype=np.float64,
    )

    if np.any(loss < 0):
        raise ValueError("附加损耗不能为负")

    factor = 10.0 ** (-loss / 10.0)

    if factor.ndim == 0:
        return float(factor)

    return factor


def _validate_loss_range(
    loss_range_db: tuple[float, float],
    parameter_name: str,
) -> None:
    """
    检查附加损耗范围是否合法。
    """
    low, high = loss_range_db

    if low < 0 or high < 0:
        raise ValueError(
            f"{parameter_name} 不能包含负数"
        )

    if low > high:
        raise ValueError(
            f"{parameter_name} 下限不能大于上限"
        )


def sample_shadow_factor(
    rng: np.random.Generator,
    blocked: bool,
    los_loss_range_db: tuple[float, float] = (
        0.0,
        3.0,
    ),
    blocked_loss_range_db: tuple[float, float] = (
        10.0,
        30.0,
    ),
) -> float:
    """
    根据遮挡状态生成附加路径损耗。
    """
    _validate_loss_range(
        los_loss_range_db,
        "los_loss_range_db",
    )

    _validate_loss_range(
        blocked_loss_range_db,
        "blocked_loss_range_db",
    )

    if blocked:
        low, high = blocked_loss_range_db
    else:
        low, high = los_loss_range_db

    loss_db = rng.uniform(low, high)

    return float(
        shadow_factor_from_loss_db(loss_db)
    )


def sample_random_channel(
    rng: np.random.Generator,
    vehicle_position: np.ndarray,
    node_position: np.ndarray,
    vehicle_gain_db: float,
    node_gain_db: float,
    wavelength_m: float = 0.1,
    blockage_probability: float = 0.3,
    rician_k_factor_db: float = 6.0,
    los_loss_range_db: tuple[float, float] = (
        0.0,
        3.0,
    ),
    blocked_loss_range_db: tuple[float, float] = (
        10.0,
        30.0,
    ),
    link_available: bool = True,
) -> ChannelSample:
    """
    完整采样论文公式 (7)。

    无遮挡：Rician 衰落；
    建筑遮挡：Rayleigh 衰落。
    """
    distance_m = distance_3d(
        vehicle_position,
        node_position,
    )

    blocked = sample_blockage(
        rng=rng,
        blockage_probability=blockage_probability,
    )

    shadow_factor = sample_shadow_factor(
        rng=rng,
        blocked=blocked,
        los_loss_range_db=los_loss_range_db,
        blocked_loss_range_db=(
            blocked_loss_range_db
        ),
    )

    if blocked:
        fading_model = "rayleigh"
        fading_power_gain = (
            sample_rayleigh_power_gain(rng)
        )
    else:
        fading_model = "rician"
        fading_power_gain = (
            sample_rician_power_gain(
                rng=rng,
                k_factor_db=rician_k_factor_db,
            )
        )

    channel_gain = free_space_channel_gain(
        vehicle_position=vehicle_position,
        node_position=node_position,
        vehicle_gain_db=vehicle_gain_db,
        node_gain_db=node_gain_db,
        wavelength_m=wavelength_m,
        shadow_factor=shadow_factor,
        fading_power_gain=fading_power_gain,
        link_available=link_available,
    )

    return ChannelSample(
        distance_m=distance_m,
        blocked=blocked,
        shadow_factor=shadow_factor,
        fading_power_gain=fading_power_gain,
        fading_model=fading_model,
        link_available=link_available,
        channel_gain=channel_gain,
    )


def sample_wireless_link(
    rng: np.random.Generator,
    vehicle_position: np.ndarray,
    node_position: np.ndarray,
    vehicle_gain_db: float,
    node_gain_db: float,
    bandwidth_hz: float,
    transmit_power_w: float,
    noise_psd_w_hz: float,
    interference_power_w: float = 0.0,
    wavelength_m: float = 0.1,
    blockage_probability: float = 0.3,
    rician_k_factor_db: float = 6.0,
    los_loss_range_db: tuple[float, float] = (
        0.0,
        3.0,
    ),
    blocked_loss_range_db: tuple[float, float] = (
        10.0,
        30.0,
    ),
    link_available: bool = True,
) -> WirelessLinkSample:
    """
    完整计算随机信道和传输速率。

    流程：
        遮挡
        -> Rayleigh/Rician
        -> 公式 (7) 信道增益
        -> SINR
        -> 公式 (8) 传输速率
    """
    channel = sample_random_channel(
        rng=rng,
        vehicle_position=vehicle_position,
        node_position=node_position,
        vehicle_gain_db=vehicle_gain_db,
        node_gain_db=node_gain_db,
        wavelength_m=wavelength_m,
        blockage_probability=blockage_probability,
        rician_k_factor_db=rician_k_factor_db,
        los_loss_range_db=los_loss_range_db,
        blocked_loss_range_db=(
            blocked_loss_range_db
        ),
        link_available=link_available,
    )

    sinr = calculate_sinr(
        bandwidth_hz=bandwidth_hz,
        transmit_power_w=transmit_power_w,
        channel_gain=channel.channel_gain,
        noise_psd_w_hz=noise_psd_w_hz,
        interference_power_w=interference_power_w,
    )

    rate_bps = achievable_rate_bps(
        bandwidth_hz=bandwidth_hz,
        transmit_power_w=transmit_power_w,
        channel_gain=channel.channel_gain,
        noise_psd_w_hz=noise_psd_w_hz,
        interference_power_w=interference_power_w,
    )

    return WirelessLinkSample(
        channel=channel,
        sinr=sinr,
        rate_bps=rate_bps,
    )