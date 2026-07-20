"""
公式(12)、(13) 的 UAV 中继任务时延与能耗计算函数。
"""
from __future__ import annotations

from dataclasses import dataclass

from dm_jcr.task_model import (
    DEFAULT_CPU_ENERGY_COEFFICIENT,
    ComputationTask,
    computation_time_s,
    dynamic_cpu_energy_j,
    transmission_time_s,
)


@dataclass(frozen=True)
class RelayLinkResources:
    """
    描述 UAV 中继任务使用的通信和计算资源。
    """

    vehicle_to_uav_rate_bps: float
    uav_to_node_rate_bps: float
    node_to_uav_rate_bps: float
    uav_to_vehicle_rate_bps: float

    uav_cpu_frequency_hz: float
    node_cpu_frequency_hz: float

    vehicle_tx_power_w: float
    uav_tx_power_w: float
    node_tx_power_w: float

    forwarding_cycles_per_bit: float

    def __post_init__(self) -> None:
        rates = {
            "车辆到 UAV 速率":
                self.vehicle_to_uav_rate_bps,
            "UAV 到节点速率":
                self.uav_to_node_rate_bps,
            "节点到 UAV 速率":
                self.node_to_uav_rate_bps,
            "UAV 到车辆速率":
                self.uav_to_vehicle_rate_bps,
        }

        for name, value in rates.items():
            if value <= 0:
                raise ValueError(
                    f"{name}必须大于 0"
                )

        if self.uav_cpu_frequency_hz <= 0:
            raise ValueError(
                "UAV CPU 频率必须大于 0"
            )

        if self.node_cpu_frequency_hz <= 0:
            raise ValueError(
                "节点 CPU 频率必须大于 0"
            )

        powers = {
            "车辆发射功率":
                self.vehicle_tx_power_w,
            "UAV 发射功率":
                self.uav_tx_power_w,
            "节点发射功率":
                self.node_tx_power_w,
        }

        for name, value in powers.items():
            if value < 0:
                raise ValueError(
                    f"{name}不能为负"
                )

        if self.forwarding_cycles_per_bit < 0:
            raise ValueError(
                "转发处理系数不能为负"
            )


@dataclass(frozen=True)
class RelayTaskMetrics:
    """
    保存 UAV 中继任务的时延与能耗分解。
    """

    vehicle_to_uav_time_s: float
    uav_to_node_time_s: float
    input_forwarding_time_s: float
    task_computation_time_s: float
    output_forwarding_time_s: float
    node_to_uav_time_s: float
    uav_to_vehicle_time_s: float
    total_latency_s: float

    vehicle_to_uav_energy_j: float
    uav_to_node_energy_j: float
    input_forwarding_energy_j: float
    task_computation_energy_j: float
    output_forwarding_energy_j: float
    node_to_uav_energy_j: float
    uav_to_vehicle_energy_j: float
    total_energy_j: float

    deadline_satisfied: bool


def forwarding_cpu_cycles(
    data_bits: float,
    forwarding_cycles_per_bit: float,
) -> float:
    """
    计算 UAV 处理转发数据所需的 CPU 周期。

        C_forward = phi * D
    """
    if data_bits < 0:
        raise ValueError("数据量不能为负")

    if forwarding_cycles_per_bit < 0:
        raise ValueError(
            "每 bit 转发周期数不能为负"
        )

    return float(
        data_bits * forwarding_cycles_per_bit
    )


def evaluate_relay_task(
    task: ComputationTask,
    resources: RelayLinkResources,
    energy_coefficient: float = (
        DEFAULT_CPU_ENERGY_COEFFICIENT
    ),
) -> RelayTaskMetrics:
    """
    计算 UAV 中继任务的完整时延和能耗。

    对应论文公式 (12)、(13) 的 relay 分支。
    """
    input_forwarding_cycles = forwarding_cpu_cycles(
        data_bits=task.input_bits,
        forwarding_cycles_per_bit=(
            resources.forwarding_cycles_per_bit
        ),
    )

    output_forwarding_cycles = forwarding_cpu_cycles(
        data_bits=task.output_bits,
        forwarding_cycles_per_bit=(
            resources.forwarding_cycles_per_bit
        ),
    )

    # ---------- 时延部分 ----------

    vehicle_to_uav_time = transmission_time_s(
        data_bits=task.input_bits,
        rate_bps=(
            resources.vehicle_to_uav_rate_bps
        ),
    )

    uav_to_node_time = transmission_time_s(
        data_bits=task.input_bits,
        rate_bps=resources.uav_to_node_rate_bps,
    )

    input_forwarding_time = computation_time_s(
        cpu_cycles=input_forwarding_cycles,
        cpu_frequency_hz=(
            resources.uav_cpu_frequency_hz
        ),
    )

    task_computation_time = computation_time_s(
        cpu_cycles=task.cpu_cycles,
        cpu_frequency_hz=(
            resources.node_cpu_frequency_hz
        ),
    )

    output_forwarding_time = computation_time_s(
        cpu_cycles=output_forwarding_cycles,
        cpu_frequency_hz=(
            resources.uav_cpu_frequency_hz
        ),
    )

    node_to_uav_time = transmission_time_s(
        data_bits=task.output_bits,
        rate_bps=resources.node_to_uav_rate_bps,
    )

    uav_to_vehicle_time = transmission_time_s(
        data_bits=task.output_bits,
        rate_bps=(
            resources.uav_to_vehicle_rate_bps
        ),
    )

    total_latency = sum(
        [
            vehicle_to_uav_time,
            uav_to_node_time,
            input_forwarding_time,
            task_computation_time,
            output_forwarding_time,
            node_to_uav_time,
            uav_to_vehicle_time,
        ]
    )

    # ---------- 能耗部分 ----------

    vehicle_to_uav_energy = (
        resources.vehicle_tx_power_w
        * vehicle_to_uav_time
    )

    uav_to_node_energy = (
        resources.uav_tx_power_w
        * uav_to_node_time
    )

    input_forwarding_energy = dynamic_cpu_energy_j(
        cpu_cycles=input_forwarding_cycles,
        cpu_frequency_hz=(
            resources.uav_cpu_frequency_hz
        ),
        energy_coefficient=(
            energy_coefficient
        ),
    )

    task_computation_energy = dynamic_cpu_energy_j(
        cpu_cycles=task.cpu_cycles,
        cpu_frequency_hz=(
            resources.node_cpu_frequency_hz
        ),
        energy_coefficient=(
            energy_coefficient
        ),
    )

    output_forwarding_energy = dynamic_cpu_energy_j(
        cpu_cycles=output_forwarding_cycles,
        cpu_frequency_hz=(
            resources.uav_cpu_frequency_hz
        ),
        energy_coefficient=(
            energy_coefficient
        ),
    )

    node_to_uav_energy = (
        resources.node_tx_power_w
        * node_to_uav_time
    )

    uav_to_vehicle_energy = (
        resources.uav_tx_power_w
        * uav_to_vehicle_time
    )

    total_energy = sum(
        [
            vehicle_to_uav_energy,
            uav_to_node_energy,
            input_forwarding_energy,
            task_computation_energy,
            output_forwarding_energy,
            node_to_uav_energy,
            uav_to_vehicle_energy,
        ]
    )

    return RelayTaskMetrics(
        vehicle_to_uav_time_s=float(
            vehicle_to_uav_time
        ),
        uav_to_node_time_s=float(
            uav_to_node_time
        ),
        input_forwarding_time_s=float(
            input_forwarding_time
        ),
        task_computation_time_s=float(
            task_computation_time
        ),
        output_forwarding_time_s=float(
            output_forwarding_time
        ),
        node_to_uav_time_s=float(
            node_to_uav_time
        ),
        uav_to_vehicle_time_s=float(
            uav_to_vehicle_time
        ),
        total_latency_s=float(total_latency),

        vehicle_to_uav_energy_j=float(
            vehicle_to_uav_energy
        ),
        uav_to_node_energy_j=float(
            uav_to_node_energy
        ),
        input_forwarding_energy_j=float(
            input_forwarding_energy
        ),
        task_computation_energy_j=float(
            task_computation_energy
        ),
        output_forwarding_energy_j=float(
            output_forwarding_energy
        ),
        node_to_uav_energy_j=float(
            node_to_uav_energy
        ),
        uav_to_vehicle_energy_j=float(
            uav_to_vehicle_energy
        ),
        total_energy_j=float(total_energy),

        deadline_satisfied=(
            total_latency <= task.max_latency_s
        ),
    )


def relay_task_latency_s(
    task: ComputationTask,
    resources: RelayLinkResources,
) -> float:
    """
    返回 UAV 中继任务的总时延。
    """
    metrics = evaluate_relay_task(
        task=task,
        resources=resources,
    )

    return metrics.total_latency_s


def relay_task_energy_j(
    task: ComputationTask,
    resources: RelayLinkResources,
    energy_coefficient: float = (
        DEFAULT_CPU_ENERGY_COEFFICIENT
    ),
) -> float:
    """
    返回 UAV 中继任务的总能耗。
    """
    metrics = evaluate_relay_task(
        task=task,
        resources=resources,
        energy_coefficient=(
            energy_coefficient
        ),
    )

    return metrics.total_energy_j
