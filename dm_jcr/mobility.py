"""DM-JCR 系统仿真器使用的移动模型，对应论文公式（4）和（5）。

本模块实现以下位置演化公式：

    l_v^(t+1) = l_v^t + sp_v^t * delta_t + omega_v^t    (4)
    l_u^(t+1) = l_u^t + sp_u^t * delta_t + omega_u^t    (5)

车辆被约束在地面平面（z = 0），无人机可在三维空间中移动。位置单位为米，
速度单位为米每秒，时隙长度单位为秒。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


Vector3: TypeAlias = NDArray[np.float64]


def _as_vector3(name: str, value: ArrayLike) -> Vector3:
    """校验并复制一个元素有限的三维向量。"""
    vector = np.asarray(value, dtype=np.float64)

    if vector.shape != (3,):
        raise ValueError(
            f"{name} 的形状必须为 (3,)，实际为 {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} 只能包含有限数值")

    return vector.copy()


def _validate_positive(name: str, value: float) -> float:
    """校验一个必须大于零的有限标量。"""
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} 必须为大于 0 的有限数值")
    return value


@dataclass(frozen=True)
class MobilityState:
    """一辆车辆或一架无人机的位置与速度。

    position_m:
        当前位置 ``[x, y, z]``，单位为米。
    velocity_mps:
        当前速度 ``[vx, vy, vz]``，单位为米每秒。

    输入数组会被复制，因此状态更新不会修改调用方传入的数组。
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
    """与坐标轴对齐的仿真区域边界。

    区域外的位置会沿 x、y、z 三个方向分别截断。上下界允许相等，例如纯二维
    道路区域可设置 ``z_min = z_max = 0``。
    """

    minimum_m: Vector3
    maximum_m: Vector3

    def __post_init__(self) -> None:
        minimum = _as_vector3("minimum_m", self.minimum_m)
        maximum = _as_vector3("maximum_m", self.maximum_m)

        if np.any(maximum < minimum):
            raise ValueError(
                "maximum_m 在每个坐标轴上都必须大于或等于 minimum_m"
            )

        object.__setattr__(self, "minimum_m", minimum)
        object.__setattr__(self, "maximum_m", maximum)


def sample_mobility_noise(
    std_m: float | ArrayLike,
    *,
    rng: np.random.Generator | None = None,
    horizontal_only: bool = False,
) -> Vector3:
    """采样移动噪声向量 omega。

    论文引入了移动噪声，但未说明具体分布。本复现在各坐标轴上采用相互独立的
    零均值高斯噪声：

        omega ~ Normal(0, sigma^2)

    std_m:
        标准差，单位为米。可以是所有坐标轴共用的标量，也可以是向量
        ``[sigma_x, sigma_y, sigma_z]``。
    rng:
        NumPy 随机数生成器。传入 ``np.random.default_rng(seed)`` 可确保实验可复现。
    horizontal_only:
        为真时强制噪声的 z 分量为零，适用于地面车辆。
    """
    standard_deviation = np.asarray(std_m, dtype=np.float64)

    if standard_deviation.ndim == 0:
        standard_deviation = np.full(3, float(standard_deviation))
    elif standard_deviation.shape == (3,):
        standard_deviation = standard_deviation.copy()
    else:
        raise ValueError("std_m 必须是标量或形状为 (3,) 的向量")

    if (
        not np.all(np.isfinite(standard_deviation))
        or np.any(standard_deviation < 0.0)
    ):
        raise ValueError("std_m 必须只包含非负有限数值")

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
    """将三维位置截断到仿真区域内。"""
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
    """应用公式（4）和（5）共用的位置更新规则。

    实现的公式为：

        next_position = position + velocity * time_step + noise

    本函数与节点类型无关。需要地面约束或无人机专用行为时，请使用
    :func:`update_vehicle_position` 或 :func:`update_uav_position`。
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
    """按照公式（4）将地面车辆推进一个时隙。

    即使状态或噪声向量误传了非零 z 分量，车辆的 z 坐标和垂直速度也始终
    强制为零。
    """
    if bounds is not None and not (
        bounds.minimum_m[2] <= 0.0 <= bounds.maximum_m[2]
    ):
        raise ValueError("车辆活动边界必须包含地面平面 z = 0")

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
    """按照公式（5）在三维空间中将无人机推进一个时隙。"""
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
