"""复现论文图 3的地面直连与 UAV 中继传输时延对照实验。

在项目根目录运行：

    python -m experiments.experiment_figure3_link_latency

实验固定任务与三个节点的位置，仅重复采样论文公式（7）中的遮挡和随机衰落，
再用公式（8）计算双向直连及四跳 UAV 中继的纯通信时延。计算时延不包含在内。
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dm_jcr.channel import achievable_rate_bps, noise_psd_w_per_hz
from dm_jcr.config import DEFAULT_CONFIG_PATH, get_experiment, load_config
from dm_jcr.random_channel import ChannelSample, sample_random_channel
from dm_jcr.task_model import transmission_time_s
from scripts._config_helpers import ChineseArgumentParser


@dataclass(frozen=True)
class Figure3Config:
    """图 3实验中固定不变的配置。"""

    trials: int
    random_seed: int
    input_bits: float
    output_ratio: float
    bandwidth_hz: float
    vehicle_transmit_power_w: float
    uav_transmit_power_w: float
    node_transmit_power_w: float
    vehicle_antenna_gain_db: float
    uav_antenna_gain_db: float
    node_antenna_gain_db: float
    wavelength_m: float
    noise_psd_w_hz: float
    interference_power_w: float
    direct_blockage_probability: float
    direct_force_blocked: bool
    relay_force_los: bool
    rician_k_factor_db: float
    los_loss_range_db: tuple[float, float]
    blocked_loss_range_db: tuple[float, float]
    vehicle_position_m: tuple[float, float, float]
    uav_position_m: tuple[float, float, float]
    node_position_m: tuple[float, float, float]
    output_csv: Path
    output_summary: Path
    output_figure: Path

    def __post_init__(self) -> None:
        if self.trials <= 0:
            raise ValueError("trials 必须是正整数")
        for name in (
            "input_bits",
            "bandwidth_hz",
            "vehicle_transmit_power_w",
            "uav_transmit_power_w",
            "node_transmit_power_w",
            "wavelength_m",
            "noise_psd_w_hz",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} 必须大于 0")
        if not 0.0 <= self.output_ratio <= 1.0:
            raise ValueError("output_ratio 必须位于 [0, 1]")
        if not 0.0 <= self.direct_blockage_probability <= 1.0:
            raise ValueError("direct_blockage_probability 必须位于 [0, 1]")

    @classmethod
    def from_mapping(
        cls,
        root: Mapping[str, Any],
        experiment: Mapping[str, Any],
    ) -> "Figure3Config":
        """从统一 TOML 配置构造图 3实验参数。"""

        paper_channel = root["paper"]["channel"]
        assumed_channel = root["assumptions"]["channel"]
        bytes_per_kilobyte = root["assumptions"]["units"]["bytes_per_kilobyte"]

        def vector3(value: Sequence[float]) -> tuple[float, float, float]:
            if len(value) != 3:
                raise ValueError("节点位置必须包含三个数值")
            return (float(value[0]), float(value[1]), float(value[2]))

        return cls(
            trials=int(experiment["trials"]),
            random_seed=int(experiment["random_seed"]),
            input_bits=float(experiment["task_input_kb"]) * bytes_per_kilobyte * 8.0,
            output_ratio=float(experiment["output_ratio"]),
            bandwidth_hz=float(experiment["bandwidth_hz"]),
            vehicle_transmit_power_w=float(experiment["vehicle_transmit_power_w"]),
            uav_transmit_power_w=float(experiment["uav_transmit_power_w"]),
            node_transmit_power_w=float(experiment["node_transmit_power_w"]),
            vehicle_antenna_gain_db=float(experiment["vehicle_antenna_gain_db"]),
            uav_antenna_gain_db=float(experiment["uav_antenna_gain_db"]),
            node_antenna_gain_db=float(experiment["node_antenna_gain_db"]),
            wavelength_m=float(paper_channel["wavelength_m"]),
            noise_psd_w_hz=noise_psd_w_per_hz(
                assumed_channel["thermal_noise_density_dbm_hz"],
                assumed_channel["receiver_noise_figure_db"],
            ),
            interference_power_w=float(assumed_channel["default_interference_power_w"]),
            direct_blockage_probability=float(paper_channel["blockage_probability"]),
            direct_force_blocked=bool(experiment["direct_force_blocked"]),
            relay_force_los=bool(experiment["relay_force_los"]),
            rician_k_factor_db=float(assumed_channel["rician_k_factor_db"]),
            los_loss_range_db=tuple(float(x) for x in assumed_channel["los_additional_loss_db"]),
            blocked_loss_range_db=tuple(
                float(x) for x in assumed_channel["blocked_additional_loss_db"]
            ),
            vehicle_position_m=vector3(experiment["vehicle_position_m"]),
            uav_position_m=vector3(experiment["uav_position_m"]),
            node_position_m=vector3(experiment["node_position_m"]),
            output_csv=Path(experiment["output_csv"]),
            output_summary=Path(experiment["output_summary"]),
            output_figure=Path(experiment["output_figure"]),
        )


@dataclass(frozen=True)
class Figure3Trial:
    """一次信道采样得到的图 3原始结果。"""

    trial: int
    direct_blocked: bool
    direct_gain: float
    vehicle_to_uav_gain: float
    uav_to_node_gain: float
    direct_upload_rate_mbps: float
    direct_download_rate_mbps: float
    relay_vehicle_to_uav_rate_mbps: float
    relay_uav_to_node_rate_mbps: float
    relay_node_to_uav_rate_mbps: float
    relay_uav_to_vehicle_rate_mbps: float
    direct_latency_ms: float
    relay_latency_ms: float


def _channel(
    rng: np.random.Generator,
    first_position: tuple[float, float, float],
    second_position: tuple[float, float, float],
    first_gain_db: float,
    second_gain_db: float,
    blockage_probability: float,
    forced_blocked: bool | None,
    config: Figure3Config,
) -> ChannelSample:
    return sample_random_channel(
        rng,
        np.asarray(first_position, dtype=np.float64),
        np.asarray(second_position, dtype=np.float64),
        first_gain_db,
        second_gain_db,
        wavelength_m=config.wavelength_m,
        blockage_probability=blockage_probability,
        rician_k_factor_db=config.rician_k_factor_db,
        los_loss_range_db=config.los_loss_range_db,
        blocked_loss_range_db=config.blocked_loss_range_db,
        forced_blocked=forced_blocked,
    )


def _rate(config: Figure3Config, power_w: float, channel_gain: float) -> float:
    return achievable_rate_bps(
        config.bandwidth_hz,
        power_w,
        channel_gain,
        config.noise_psd_w_hz,
        config.interference_power_w,
    )


def run_experiment(config: Figure3Config) -> list[Figure3Trial]:
    """运行全部独立信道采样并返回逐次实验结果。"""

    rng = np.random.default_rng(config.random_seed)
    output_bits = config.input_bits * config.output_ratio
    records: list[Figure3Trial] = []
    for trial in range(1, config.trials + 1):
        # 同一物理链路的上下行共享信道增益，发射功率分别计算。
        direct = _channel(
            rng,
            config.vehicle_position_m,
            config.node_position_m,
            config.vehicle_antenna_gain_db,
            config.node_antenna_gain_db,
            config.direct_blockage_probability,
            True if config.direct_force_blocked else None,
            config,
        )
        vehicle_uav = _channel(
            rng,
            config.vehicle_position_m,
            config.uav_position_m,
            config.vehicle_antenna_gain_db,
            config.uav_antenna_gain_db,
            0.0,
            False if config.relay_force_los else None,
            config,
        )
        uav_node = _channel(
            rng,
            config.uav_position_m,
            config.node_position_m,
            config.uav_antenna_gain_db,
            config.node_antenna_gain_db,
            0.0,
            False if config.relay_force_los else None,
            config,
        )

        direct_upload = _rate(config, config.vehicle_transmit_power_w, direct.channel_gain)
        direct_download = _rate(config, config.node_transmit_power_w, direct.channel_gain)
        vehicle_to_uav = _rate(
            config, config.vehicle_transmit_power_w, vehicle_uav.channel_gain
        )
        uav_to_vehicle = _rate(config, config.uav_transmit_power_w, vehicle_uav.channel_gain)
        uav_to_node = _rate(config, config.uav_transmit_power_w, uav_node.channel_gain)
        node_to_uav = _rate(config, config.node_transmit_power_w, uav_node.channel_gain)

        direct_latency = transmission_time_s(config.input_bits, direct_upload)
        direct_latency += transmission_time_s(output_bits, direct_download)
        relay_latency = transmission_time_s(config.input_bits, vehicle_to_uav)
        relay_latency += transmission_time_s(config.input_bits, uav_to_node)
        relay_latency += transmission_time_s(output_bits, node_to_uav)
        relay_latency += transmission_time_s(output_bits, uav_to_vehicle)

        records.append(
            Figure3Trial(
                trial,
                direct.blocked,
                direct.channel_gain,
                vehicle_uav.channel_gain,
                uav_node.channel_gain,
                direct_upload / 1.0e6,
                direct_download / 1.0e6,
                vehicle_to_uav / 1.0e6,
                uav_to_node / 1.0e6,
                node_to_uav / 1.0e6,
                uav_to_vehicle / 1.0e6,
                direct_latency * 1.0e3,
                relay_latency * 1.0e3,
            )
        )
    return records


def summarize(records: Sequence[Figure3Trial]) -> dict[str, float | int]:
    """汇总图 3需要的均值、波动和中继改善比例。"""

    if not records:
        raise ValueError("至少需要一条实验记录")
    direct = np.asarray([item.direct_latency_ms for item in records])
    relay = np.asarray([item.relay_latency_ms for item in records])
    blocked = np.asarray([item.direct_blocked for item in records], dtype=np.float64)
    return {
        "trials": len(records),
        "direct_latency_mean_ms": float(direct.mean()),
        "direct_latency_std_ms": float(direct.std()),
        "relay_latency_mean_ms": float(relay.mean()),
        "relay_latency_std_ms": float(relay.std()),
        "direct_blocked_ratio": float(blocked.mean()),
        "relay_faster_ratio": float((relay < direct).mean()),
    }


def save_results(
    config: Figure3Config,
    records: Sequence[Figure3Trial],
    summary: Mapping[str, float | int],
) -> None:
    """保存可复核的原始数据、汇总信息和论文风格折线图。"""

    for path in (config.output_csv, config.output_summary, config.output_figure):
        path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(asdict(records[0]).keys())
    with config.output_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(item) for item in records)

    payload = {
        "summary": dict(summary),
        "assumptions": {
            "latency_scope": "仅计算双向传输时延，不包含计算和转发处理时延",
            "reciprocal_channel_gain": True,
            "direct_blockage_probability": config.direct_blockage_probability,
            "direct_force_blocked": config.direct_force_blocked,
            "relay_force_los": config.relay_force_los,
            "input_bits": config.input_bits,
            "output_ratio": config.output_ratio,
            "bandwidth_hz": config.bandwidth_hz,
            "random_seed": config.random_seed,
        },
    }
    config.output_summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    trials = [item.trial for item in records]
    direct = [item.direct_latency_ms for item in records]
    relay = [item.relay_latency_ms for item in records]
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    axis.plot(trials, direct, linewidth=1.35, label="Direct link")
    axis.plot(trials, relay, linewidth=1.35, label="UAV relay")
    axis.set_xlabel("Experiment index")
    axis.set_ylabel("Transmission latency (ms)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(config.output_figure, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = ChineseArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="TOML 配置文件路径",
    )
    args = parser.parse_args()
    root = load_config(args.config)
    config = Figure3Config.from_mapping(
        root, get_experiment(root, "figure3_link_latency")
    )
    records = run_experiment(config)
    summary = summarize(records)
    save_results(config, records, summary)
    print(
        "图 3实验完成："
        f"直连平均时延={summary['direct_latency_mean_ms']:.3f} ms，"
        f"UAV 中继平均时延={summary['relay_latency_mean_ms']:.3f} ms，"
        f"中继更快比例={summary['relay_faster_ratio']:.1%}"
    )
    print(f"原始数据：{config.output_csv.resolve()}")
    print(f"汇总信息：{config.output_summary.resolve()}")
    print(f"图像：{config.output_figure.resolve()}")


if __name__ == "__main__":
    main()
