"""用 PyTorch 可微运算实现论文公式（8）、（12）、（13）、（17）和（27）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from dm_jcr.channel import noise_psd_w_per_hz


@dataclass(frozen=True)
class ObjectiveTensorSpec:
    """从归一化环境张量恢复物理量并计算公式（17）所需的常量。"""

    bandwidth_scale_hz: float
    cpu_scale_hz: float
    power_scale_w: float
    task_bits_scale: float
    task_cycles_scale: float
    latency_scale_s: float
    channel_gain_scale: float
    vehicle_transmit_power_w: float
    noise_psd_w_hz: float
    forwarding_cycles_per_bit: float
    energy_coefficient: float
    energy_reference_j: float
    latency_weight: float
    energy_weight: float


@dataclass(frozen=True)
class DifferentiableObjective:
    """一个 batch 的公式（17）损失组成。"""

    weighted_objective: Tensor
    mean_normalized_latency: Tensor
    mean_normalized_energy: Tensor
    deadline_satisfied_ratio: Tensor


def objective_tensor_spec_from_config(config: dict[str, Any]) -> ObjectiveTensorSpec:
    tensor = config["assumptions"]["tensor"]
    generation = config["assumptions"]["dataset_generation"]
    channel = config["assumptions"]["channel"]
    objective = config["paper"]["objective"]
    return ObjectiveTensorSpec(
        bandwidth_scale_hz=tensor["bandwidth_scale_hz"],
        cpu_scale_hz=tensor["cpu_scale_hz"],
        power_scale_w=tensor["power_scale_w"],
        task_bits_scale=tensor["task_bits_scale"],
        task_cycles_scale=tensor["task_cycles_scale"],
        latency_scale_s=tensor["latency_scale_s"],
        channel_gain_scale=tensor["channel_gain_scale"],
        vehicle_transmit_power_w=generation["vehicle_transmit_power_w"],
        noise_psd_w_hz=noise_psd_w_per_hz(
            channel["thermal_noise_density_dbm_hz"],
            channel["receiver_noise_figure_db"],
        ),
        forwarding_cycles_per_bit=generation["forwarding_cycles_per_bit"],
        energy_coefficient=config["paper"]["task"]["cpu_energy_coefficient"],
        energy_reference_j=config["assumptions"]["objective"]["energy_reference_j"],
        latency_weight=objective["latency_weight"],
        energy_weight=objective["energy_weight"],
    )


def strategy_slot_mask(task_features: Tensor, task_mask: Tensor) -> Tensor:
    """返回三类任务在八槽位策略中的有效位置。"""

    direct = task_features[..., 4].bool() & task_mask.bool()
    relay = task_features[..., 5].bool() & task_mask.bool()
    v2v = task_features[..., 6].bool() & task_mask.bool()
    result = torch.zeros((*task_mask.shape, 8), device=task_mask.device, dtype=torch.bool)
    result[..., 0] = direct | relay | v2v
    result[..., 1] = relay
    result[..., 2] = relay
    result[..., 3] = relay | v2v
    result[..., 4] = direct | relay | v2v
    result[..., 5] = relay
    result[..., 6] = direct | relay | v2v
    result[..., 7] = relay
    return result


def _gather_nodes(values: Tensor, indices: Tensor) -> Tensor:
    safe = indices.clamp(min=0)
    return torch.gather(values, 1, safe.unsqueeze(-1).expand(-1, -1, values.shape[-1]))


def _gather_matrix(values: Tensor, rows: Tensor, columns: Tensor) -> Tensor:
    batch = torch.arange(values.shape[0], device=values.device).unsqueeze(1)
    return values[batch, rows.clamp(min=0), columns.clamp(min=0)]


def _scatter_claim(denominator: Tensor, node_indices: Tensor, claim: Tensor) -> Tensor:
    return denominator.scatter_add(1, node_indices.clamp(min=0), claim)


def _allocated(
    capacity: Tensor,
    denominator: Tensor,
    node_indices: Tensor,
    score: Tensor,
    active: Tensor,
    epsilon: float,
) -> Tensor:
    safe = node_indices.clamp(min=0)
    node_capacity = torch.gather(capacity, 1, safe)
    node_total = torch.gather(denominator, 1, safe).clamp_min(epsilon)
    return node_capacity * score * active / node_total


def _rate(
    bandwidth: Tensor,
    power: Tensor,
    gain: Tensor,
    noise_psd: float,
    epsilon: float,
) -> Tensor:
    bandwidth = bandwidth.clamp_min(epsilon)
    sinr = power.clamp_min(0.0) * gain.clamp_min(0.0) / (
        noise_psd * bandwidth
    ).clamp_min(epsilon)
    return (bandwidth * torch.log2(1.0 + sinr).clamp_min(epsilon)).clamp_min(
        epsilon
    )


def evaluate_strategy_tensor(
    scores: Tensor,
    node_features: Tensor,
    channel_gains: Tensor,
    task_features: Tensor,
    task_node_indices: Tensor,
    task_mask: Tensor,
    spec: ObjectiveTensorSpec,
    *,
    epsilon: float = 1.0e-30,
) -> DifferentiableObjective:
    """对模型输出分数执行公式（27）并计算可反向传播的公式（17）目标。"""

    if scores.ndim != 3 or scores.shape[-1] != 8:
        raise ValueError("scores 必须具有形状 (batch, tasks, 8)")
    task_count = scores.shape[1]
    task_features = task_features[:, :task_count]
    task_node_indices = task_node_indices[:, :task_count]
    task_mask = task_mask[:, :task_count].bool()
    valid_slots = strategy_slot_mask(task_features, task_mask)
    scores = scores.clamp_min(epsilon) * valid_slots

    node_capacity = torch.stack(
        (
            node_features[..., 9] * spec.bandwidth_scale_hz,
            node_features[..., 10] * spec.cpu_scale_hz,
            node_features[..., 11] * spec.power_scale_w,
        ),
        dim=-1,
    )
    bandwidth_capacity, cpu_capacity, power_capacity = node_capacity.unbind(-1)
    source, compute, relay_node, target = task_node_indices.unbind(-1)
    direct = task_features[..., 4] * task_mask
    relay = task_features[..., 5] * task_mask
    v2v = task_features[..., 6] * task_mask

    zeros = torch.zeros_like(bandwidth_capacity)
    bandwidth_denominator = zeros
    bandwidth_denominator = _scatter_claim(
        bandwidth_denominator, compute, scores[..., 0] * direct
    )
    bandwidth_denominator = _scatter_claim(
        bandwidth_denominator,
        relay_node,
        (scores[..., 0] + scores[..., 1] + scores[..., 3]) * relay
        + (scores[..., 0] + scores[..., 3]) * v2v,
    )
    bandwidth_denominator = _scatter_claim(
        bandwidth_denominator, compute, scores[..., 2] * relay
    )

    cpu_denominator = zeros
    cpu_denominator = _scatter_claim(
        cpu_denominator, compute, scores[..., 4] * direct + scores[..., 5] * relay
    )
    cpu_denominator = _scatter_claim(
        cpu_denominator, relay_node, scores[..., 4] * (relay + v2v)
    )

    power_denominator = zeros
    power_denominator = _scatter_claim(
        power_denominator, compute, scores[..., 6] * direct + scores[..., 7] * relay
    )
    power_denominator = _scatter_claim(
        power_denominator, relay_node, scores[..., 6] * (relay + v2v)
    )

    direct_bandwidth = _allocated(
        bandwidth_capacity, bandwidth_denominator, compute, scores[..., 0], direct, epsilon
    )
    direct_cpu = _allocated(
        cpu_capacity, cpu_denominator, compute, scores[..., 4], direct, epsilon
    )
    direct_power = _allocated(
        power_capacity, power_denominator, compute, scores[..., 6], direct, epsilon
    )
    relay_bandwidths = tuple(
        _allocated(
            bandwidth_capacity,
            bandwidth_denominator,
            relay_node if slot != 2 else compute,
            scores[..., slot],
            relay,
            epsilon,
        )
        for slot in range(4)
    )
    relay_cpu = _allocated(
        cpu_capacity, cpu_denominator, relay_node, scores[..., 4], relay, epsilon
    )
    compute_cpu = _allocated(
        cpu_capacity, cpu_denominator, compute, scores[..., 5], relay, epsilon
    )
    relay_power = _allocated(
        power_capacity, power_denominator, relay_node, scores[..., 6], relay, epsilon
    )
    compute_power = _allocated(
        power_capacity, power_denominator, compute, scores[..., 7], relay, epsilon
    )
    v2v_bandwidth_up = _allocated(
        bandwidth_capacity, bandwidth_denominator, relay_node, scores[..., 0], v2v, epsilon
    )
    v2v_bandwidth_down = _allocated(
        bandwidth_capacity, bandwidth_denominator, relay_node, scores[..., 3], v2v, epsilon
    )
    v2v_cpu = _allocated(
        cpu_capacity, cpu_denominator, relay_node, scores[..., 4], v2v, epsilon
    )
    v2v_power = _allocated(
        power_capacity, power_denominator, relay_node, scores[..., 6], v2v, epsilon
    )

    gains = channel_gains * spec.channel_gain_scale
    data = task_features[..., 0] * spec.task_bits_scale
    cycles = task_features[..., 1] * spec.task_cycles_scale
    deadline = task_features[..., 2] * spec.latency_scale_s
    output_ratio = task_features[..., 3]
    output_data = data * output_ratio
    vehicle_power = torch.as_tensor(
        spec.vehicle_transmit_power_w, device=scores.device, dtype=scores.dtype
    )

    direct_up_rate = _rate(
        direct_bandwidth,
        vehicle_power,
        _gather_matrix(gains, source, compute),
        spec.noise_psd_w_hz,
        epsilon,
    )
    direct_down_rate = _rate(
        direct_bandwidth,
        direct_power,
        _gather_matrix(gains, compute, source),
        spec.noise_psd_w_hz,
        epsilon,
    )
    direct_data = data * direct
    direct_cycles = cycles * direct
    direct_output = output_data * direct
    direct_up_time = direct_data / direct_up_rate
    direct_compute_time = direct_cycles / direct_cpu.clamp_min(epsilon)
    direct_down_time = direct_output / direct_down_rate
    direct_latency = direct_up_time + direct_compute_time + direct_down_time
    direct_energy = (
        vehicle_power * direct_up_time
        + spec.energy_coefficient * direct_cpu.square() * direct_cycles
        + direct_power * direct_down_time
    )

    b_vu, b_un, b_nu, b_uv = relay_bandwidths
    rate_vu = _rate(
        b_vu, vehicle_power, _gather_matrix(gains, source, relay_node), spec.noise_psd_w_hz, epsilon
    )
    rate_un = _rate(
        b_un, relay_power, _gather_matrix(gains, relay_node, compute), spec.noise_psd_w_hz, epsilon
    )
    rate_nu = _rate(
        b_nu, compute_power, _gather_matrix(gains, compute, relay_node), spec.noise_psd_w_hz, epsilon
    )
    rate_uv = _rate(
        b_uv, relay_power, _gather_matrix(gains, relay_node, source), spec.noise_psd_w_hz, epsilon
    )
    relay_data = data * relay
    relay_cycles = cycles * relay
    relay_output = output_data * relay
    input_forward_cycles = relay_data * spec.forwarding_cycles_per_bit
    output_forward_cycles = relay_output * spec.forwarding_cycles_per_bit
    times = (
        relay_data / rate_vu,
        relay_data / rate_un,
        input_forward_cycles / relay_cpu.clamp_min(epsilon),
        relay_cycles / compute_cpu.clamp_min(epsilon),
        output_forward_cycles / relay_cpu.clamp_min(epsilon),
        relay_output / rate_nu,
        relay_output / rate_uv,
    )
    relay_latency = sum(times)
    relay_energy = (
        vehicle_power * times[0]
        + relay_power * times[1]
        + spec.energy_coefficient * relay_cpu.square() * input_forward_cycles
        + spec.energy_coefficient * compute_cpu.square() * relay_cycles
        + spec.energy_coefficient * relay_cpu.square() * output_forward_cycles
        + compute_power * times[5]
        + relay_power * times[6]
    )

    v2v_up_rate = _rate(
        v2v_bandwidth_up,
        vehicle_power,
        _gather_matrix(gains, source, relay_node),
        spec.noise_psd_w_hz,
        epsilon,
    )
    v2v_down_rate = _rate(
        v2v_bandwidth_down,
        v2v_power,
        _gather_matrix(gains, relay_node, target),
        spec.noise_psd_w_hz,
        epsilon,
    )
    v2v_data = data * v2v
    v2v_forward_cycles = v2v_data * spec.forwarding_cycles_per_bit
    v2v_up_time = v2v_data / v2v_up_rate
    v2v_forward_time = v2v_forward_cycles / v2v_cpu.clamp_min(epsilon)
    v2v_down_time = v2v_data / v2v_down_rate
    v2v_latency = v2v_up_time + v2v_forward_time + v2v_down_time
    v2v_energy = (
        vehicle_power * v2v_up_time
        + spec.energy_coefficient * v2v_cpu.square() * v2v_forward_cycles
        + v2v_power * v2v_down_time
    )

    latency = direct * direct_latency + relay * relay_latency + v2v * v2v_latency
    energy = direct * direct_energy + relay * relay_energy + v2v * v2v_energy
    denominator = task_mask.sum(dim=1).clamp_min(1)
    normalized_latency = (latency / deadline.clamp_min(epsilon) * task_mask).sum(dim=1) / denominator
    normalized_energy = (energy / spec.energy_reference_j * task_mask).sum(dim=1) / denominator
    weighted = spec.latency_weight * normalized_latency + spec.energy_weight * normalized_energy
    satisfied = ((latency <= deadline) & task_mask).sum(dim=1) / denominator
    return DifferentiableObjective(
        weighted,
        normalized_latency,
        normalized_energy,
        satisfied,
    )


__all__ = [
    "DifferentiableObjective",
    "ObjectiveTensorSpec",
    "evaluate_strategy_tensor",
    "objective_tensor_spec_from_config",
    "strategy_slot_mask",
]
