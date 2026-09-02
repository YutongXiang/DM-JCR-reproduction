"""
公式(12)、(13) 的直接链路任务时延与能耗计算函数。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


# 论文计算能耗模型采用的系数。
DEFAULT_CPU_ENERGY_COEFFICIENT = 5.0e-28


def _validate_finite(
    name: str,
    value: float,
    *,
    strictly_positive: bool,
) -> float:
    """将数值转换为浮点数并校验范围。

    name:
        错误信息中使用的参数名。
    value:
        待校验数值。
    strictly_positive:
        为真时要求 value > 0，否则要求 value >= 0。

    返回校验后的浮点数。
    """
    value = float(value)

    if not isfinite(value):
        raise ValueError(f"{name} 必须为有限数值，实际为 {value!r}")

    if strictly_positive:
        if value <= 0.0:
            raise ValueError(f"{name} 必须大于 0，实际为 {value!r}")
    elif value < 0.0:
        raise ValueError(f"{name} 必须为非负数，实际为 {value!r}")

    return value


def kilobytes_to_bits(size_kb: float) -> float:
    """将十进制千字节换算为比特。

    本复现采用：

        1 KB = 1000 Byte
        1 Byte = 8 bit

    因此：

        比特数 = size_kb * 1000 * 8
    """
    size_kb = _validate_finite(
        "size_kb",
        size_kb,
        strictly_positive=False,
    )
    return size_kb * 1000.0 * 8.0


@dataclass(frozen=True)
class ComputationTask:
    """一个计算任务的描述。

    input_bits:
        输入任务大小 D，单位为比特。
    cpu_cycles:
        执行任务所需的 CPU 总周期数 C。
    max_latency_s:
        最大允许时延 T_max，单位为秒。
    output_ratio:
        输出与输入的数据量比 mu，输出大小为 mu * D。
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
        """返回结果数据大小 mu * D，单位为比特。"""
        return self.output_ratio * self.input_bits


@dataclass(frozen=True)
class DirectLinkResources:
    """任务直连执行所使用的通信与计算资源。

    uplink_rate_bps:
        车辆到节点的速率 r_vn，单位为 bit/s。
    downlink_rate_bps:
        节点到车辆的速率 r_nv，单位为 bit/s。
    cpu_frequency_hz:
        边缘节点分配的 CPU 频率 f_n，单位为 cycle/s。
    vehicle_tx_power_w:
        车辆发射功率 p_v，单位为瓦。
    node_tx_power_w:
        边缘节点发射功率 p_n，单位为瓦。
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
    """一个直连任务的详细评价结果。"""

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
    """按照 T = D / r 计算传输时间。

    data_bits:
        数据大小 D，单位为比特。
    rate_bps:
        传输速率 r，单位为 bit/s。

    返回以秒为单位的传输时间。
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
    """按照 T = C / f 计算执行时间。

    虽然参数名含有 ``hz``，实际单位是 CPU 每秒周期数。

    cpu_cycles:
        所需 CPU 周期数 C。
    cpu_frequency_hz:
        分配的 CPU 频率 f，单位为 cycle/s。

    返回以秒为单位的执行时间。
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
    """按照 E = zeta * f^2 * C 计算 CPU 动态能耗。

    cpu_cycles:
        所需 CPU 周期数 C。
    cpu_frequency_hz:
        CPU 频率 f，单位为 cycle/s。
    energy_coefficient:
        与硬件有关的系数 zeta。

    返回以焦耳为单位的 CPU 动态能耗。
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
    """按照公式（12）计算直连任务的总时延。

    直连分支为：

        T_direct
            = D / r_vn
            + C / f_n
            + mu * D / r_nv

    返回以秒为单位的任务总时延。
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
    """按照公式（13）计算直连任务的总能耗。

    直连分支为：

        E_direct
            = p_v * D / r_vn
            + zeta * f_n^2 * C
            + p_n * mu * D / r_nv

    返回以焦耳为单位的总能耗。
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
    """同时评价时延、能耗和时限可行性。

    task:
        待评价的计算任务。
    resources:
        直连通信与计算资源。
    energy_coefficient:
        CPU 能耗系数 zeta。
    deadline_tolerance_s:
        将计算时延与 T_max 比较时采用的数值容差。

    返回详细的时延与能耗分解结果。
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

    # 公式（12）：时延分量。
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

    # 公式（13）：能耗分量。
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
