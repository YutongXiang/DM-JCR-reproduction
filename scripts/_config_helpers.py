"""为配置驱动的检查脚本提供共用对象构造函数。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dm_jcr.config import DEFAULT_CONFIG_PATH, get_experiment, load_config
from dm_jcr.relay_resource_allocation import RawRelayTaskAllocation, RelayTaskContext
from dm_jcr.resource_allocation import (
    DirectTaskContext,
    NodeResourceCapacity,
    ObjectiveNormalization,
    ObjectiveWeights,
    RawTaskAllocation,
    RawV2VRelayAllocation,
    V2VRelayTaskContext,
)
from dm_jcr.task_model import ComputationTask


class ChineseArgumentParser(argparse.ArgumentParser):
    """使用中文标题、帮助项和错误前缀的命令行解析器。"""

    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs["add_help"] = False
        super().__init__(*args, **kwargs)
        self._positionals.title = "位置参数"
        self._optionals.title = "选项"
        self.add_argument(
            "-h",
            "--help",
            action="help",
            default=argparse.SUPPRESS,
            help="显示帮助信息并退出",
        )

    def format_help(self) -> str:
        """将 argparse 固定生成的英文区段标题替换为中文。"""

        return super().format_help().replace("usage:", "用法：", 1)

    def format_usage(self) -> str:
        """返回采用中文前缀的用法说明。"""

        return super().format_usage().replace("usage:", "用法：", 1)

    def error(self, message: str) -> None:
        """以中文错误前缀报告无效命令行参数。"""

        self.print_usage()
        self.exit(2, f"{self.prog}：参数错误：{message}\n")


def load_script_config(
    experiment_name: str,
    description: str,
    argv: Sequence[str] | None = None,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    parser = ChineseArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="TOML 配置文件路径",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    return config, get_experiment(config, experiment_name)


def capacities(records: Sequence[Mapping[str, Any]]) -> tuple[NodeResourceCapacity, ...]:
    return tuple(NodeResourceCapacity(**record) for record in records)


def computation_task(record: Mapping[str, Any]) -> ComputationTask:
    return ComputationTask(
        input_bits=record["input_bits"],
        cpu_cycles=record["cpu_cycles"],
        max_latency_s=record["max_latency_s"],
        output_ratio=record["output_ratio"],
    )


def direct_context(record: Mapping[str, Any]) -> DirectTaskContext:
    return DirectTaskContext(
        task_id=record["task_id"],
        node_id=record["node_id"],
        task=computation_task(record),
        uplink_channel_gain=record["uplink_channel_gain"],
        downlink_channel_gain=record["downlink_channel_gain"],
        vehicle_transmit_power_w=record["vehicle_transmit_power_w"],
        noise_psd_w_hz=record["noise_psd_w_hz"],
    )


def raw_direct(record: Mapping[str, Any]) -> RawTaskAllocation:
    return RawTaskAllocation(
        task_id=record["task_id"],
        node_id=record["node_id"],
        bandwidth_score=record["bandwidth_score"],
        cpu_score=record["cpu_score"],
        power_score=record["power_score"],
    )


def relay_context(record: Mapping[str, Any]) -> RelayTaskContext:
    return RelayTaskContext(
        task_id=record["task_id"],
        relay_uav_id=record["relay_uav_id"],
        compute_node_id=record["compute_node_id"],
        task=computation_task(record),
        vehicle_to_uav_channel_gain=record["vehicle_to_uav_channel_gain"],
        uav_to_node_channel_gain=record["uav_to_node_channel_gain"],
        node_to_uav_channel_gain=record["node_to_uav_channel_gain"],
        uav_to_vehicle_channel_gain=record["uav_to_vehicle_channel_gain"],
        vehicle_transmit_power_w=record["vehicle_transmit_power_w"],
        noise_psd_w_hz=record["noise_psd_w_hz"],
        forwarding_cycles_per_bit=record["forwarding_cycles_per_bit"],
    )


def raw_relay(record: Mapping[str, Any]) -> RawRelayTaskAllocation:
    fields = (
        "task_id", "relay_uav_id", "compute_node_id",
        "vehicle_to_uav_bandwidth_score", "uav_to_node_bandwidth_score",
        "node_to_uav_bandwidth_score", "uav_to_vehicle_bandwidth_score",
        "relay_cpu_score", "compute_cpu_score", "relay_power_score",
        "compute_node_power_score",
    )
    return RawRelayTaskAllocation(**{field: record[field] for field in fields})


def v2v_context(record: Mapping[str, Any]) -> V2VRelayTaskContext:
    fields = (
        "task_id", "source_vehicle_id", "target_vehicle_id", "relay_uav_id",
        "data_bits", "max_latency_s", "vehicle_to_uav_channel_gain",
        "uav_to_vehicle_channel_gain", "source_vehicle_transmit_power_w",
        "noise_psd_w_hz", "forwarding_cycles_per_bit",
    )
    return V2VRelayTaskContext(**{field: record[field] for field in fields})


def raw_v2v(record: Mapping[str, Any]) -> RawV2VRelayAllocation:
    fields = (
        "task_id", "relay_uav_id", "vehicle_to_uav_bandwidth_score",
        "uav_to_vehicle_bandwidth_score", "relay_cpu_score", "relay_power_score",
    )
    return RawV2VRelayAllocation(**{field: record[field] for field in fields})


def objective_settings(
    config: Mapping[str, Any],
) -> tuple[ObjectiveNormalization, ObjectiveWeights, float]:
    assumed = config["assumptions"]["objective"]
    paper = config["paper"]["objective"]
    normalization = ObjectiveNormalization(
        energy_reference_j=assumed["energy_reference_j"]
    )
    weights = ObjectiveWeights(
        latency=paper["latency_weight"],
        energy=paper["energy_weight"],
    )
    energy_coefficient = float(config["paper"]["task"]["cpu_energy_coefficient"])
    return normalization, weights, energy_coefficient
