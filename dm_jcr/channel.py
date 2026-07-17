"""
对应论文公式(6)、(7)、(8) 的信道模型计算函数。
"""

from __future__ import annotations

import numpy as np


def db_to_linear(value_db: float) -> float:
    """
    将 dB 或 dBi 转换成线性倍数。

    例如：
        0 dB  -> 1
        10 dB -> 10
    """
    return float(10.0 ** (value_db / 10.0))


def dbm_to_watts(power_dbm: float) -> float:
    """
    将功率从 dBm 转换为 W。

    转换关系：
        P(W) = 10^((P(dBm) - 30) / 10)
    """
    return float(10.0 ** ((power_dbm - 30.0) / 10.0))


def noise_psd_w_per_hz(
    noise_density_dbm_hz: float = -174.0,
    noise_figure_db: float = 0.0,
) -> float:
    """
    将噪声功率谱密度从 dBm/Hz 转换为 W/Hz。

    noise_density_dbm_hz:
        热噪声功率谱密度，默认取 -174 dBm/Hz。

    noise_figure_db:
        接收机噪声系数，论文没有给出，当前默认取 0 dB。
    """
    if noise_figure_db < 0:
        raise ValueError("noise_figure_db 不能为负")

    effective_noise_dbm_hz = (
        noise_density_dbm_hz + noise_figure_db
    )

    return dbm_to_watts(effective_noise_dbm_hz)


def distance_3d(
    position_a: np.ndarray,
    position_b: np.ndarray,
) -> float:
    """
    计算两个节点之间的三维欧氏距离，单位为米。

    对应论文公式 (6)。
    """
    position_a = np.asarray(position_a, dtype=np.float64)
    position_b = np.asarray(position_b, dtype=np.float64)

    if position_a.shape != (3,) or position_b.shape != (3,):
        raise ValueError(
            "节点坐标必须是形如 [x, y, z] 的三维向量"
        )

    return float(np.linalg.norm(position_a - position_b))


def free_space_channel_gain(
    vehicle_position: np.ndarray,
    node_position: np.ndarray,
    vehicle_gain_db: float,
    node_gain_db: float,
    wavelength_m: float = 0.1,
    shadow_factor: float = 1.0,
    fading_power_gain: float = 1.0,
    link_available: bool = True,
) -> float:
    """
    计算车辆到 UAV/RSU 的信道功率增益。

    对应论文公式 (7)：

        h = G_v * G_n * lambda^2 / (4*pi*d)^2
            * shadow * fading * indicator
    """
    if not link_available:
        return 0.0

    if wavelength_m <= 0:
        raise ValueError("通信波长必须大于 0")

    if not 0.0 <= shadow_factor <= 1.0:
        raise ValueError(
            "shadow_factor 必须位于 [0, 1]"
        )

    if fading_power_gain < 0:
        raise ValueError(
            "fading_power_gain 不能为负"
        )

    distance_m = distance_3d(
        vehicle_position,
        node_position,
    )

    if distance_m <= 0:
        raise ValueError(
            "两个通信节点不能位于完全相同的位置"
        )

    vehicle_gain = db_to_linear(vehicle_gain_db)
    node_gain = db_to_linear(node_gain_db)

    free_space_gain = (
        vehicle_gain
        * node_gain
        * wavelength_m**2
        / (4.0 * np.pi * distance_m) ** 2
    )

    return float(
        free_space_gain
        * shadow_factor
        * fading_power_gain
    )


def calculate_sinr(
    bandwidth_hz: float,
    transmit_power_w: float,
    channel_gain: float,
    noise_psd_w_hz: float,
    interference_power_w: float = 0.0,
) -> float:
    """
    计算信噪干扰比 SINR。

    对应论文公式 (8) 中对数内部的分数：

        SINR = p*h / (N0*b + I)
    """
    if bandwidth_hz <= 0:
        raise ValueError("带宽必须大于 0")

    if transmit_power_w < 0:
        raise ValueError("发射功率不能为负")

    if channel_gain < 0:
        raise ValueError("信道增益不能为负")

    if noise_psd_w_hz <= 0:
        raise ValueError("噪声功率谱密度必须大于 0")

    if interference_power_w < 0:
        raise ValueError("干扰功率不能为负")

    signal_power_w = transmit_power_w * channel_gain
    noise_power_w = noise_psd_w_hz * bandwidth_hz

    denominator_w = (
        noise_power_w + interference_power_w
    )

    return float(signal_power_w / denominator_w)


def achievable_rate_bps(
    bandwidth_hz: float,
    transmit_power_w: float,
    channel_gain: float,
    noise_psd_w_hz: float,
    interference_power_w: float = 0.0,
) -> float:
    """
    计算可达传输速率，单位为 bit/s。

    对应论文公式 (8)：

        r = b * log2(1 + SINR)
    """
    sinr = calculate_sinr(
        bandwidth_hz=bandwidth_hz,
        transmit_power_w=transmit_power_w,
        channel_gain=channel_gain,
        noise_psd_w_hz=noise_psd_w_hz,
        interference_power_w=interference_power_w,
    )

    return float(
        bandwidth_hz * np.log2(1.0 + sinr)
    )