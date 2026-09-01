"""在多时隙信道实验中集成论文公式（4）至（8）。

在项目根目录运行：

    python -m scripts.check_dynamic_channel

输出文件：

    outputs/dynamic_channel.csv
    outputs/dynamic_channel_trend.png
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from dm_jcr.channel import noise_psd_w_per_hz

from dm_jcr.mobility import (
    MobilityState,
    SimulationBounds,
    sample_mobility_noise,
    update_uav_position,
    update_vehicle_position,
)
from dm_jcr.random_channel import sample_wireless_link
from scripts._config_helpers import load_script_config


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters that remain fixed during the experiment."""

    number_of_slots: int
    time_step_s: float
    random_seed: int
    vehicle_mobility_noise_std_m: float
    uav_mobility_noise_std_m: float
    vehicle_gain_db: float
    uav_gain_db: float
    wavelength_m: float
    bandwidth_hz: float
    transmit_power_w: float
    noise_psd_w_hz: float
    interference_power_w: float
    blockage_probability: float
    rician_k_factor_db: float
    los_loss_range_db: tuple[float, float]
    blocked_loss_range_db: tuple[float, float]
    link_available: bool
    smoothing_window: int
    initial_vehicle_position_m: tuple[float, float, float]
    initial_vehicle_velocity_mps: tuple[float, float, float]
    initial_uav_position_m: tuple[float, float, float]
    initial_uav_velocity_mps: tuple[float, float, float]
    vehicle_bounds_minimum_m: tuple[float, float, float]
    vehicle_bounds_maximum_m: tuple[float, float, float]
    uav_bounds_minimum_m: tuple[float, float, float]
    uav_bounds_maximum_m: tuple[float, float, float]
    output_csv: Path
    output_figure: Path

    def __post_init__(self) -> None:
        if self.number_of_slots <= 0:
            raise ValueError("number_of_slots must be greater than 0")
        if self.time_step_s <= 0.0:
            raise ValueError("time_step_s must be greater than 0")
        if not 0.0 <= self.blockage_probability <= 1.0:
            raise ValueError("blockage_probability must be in [0, 1]")
        if self.smoothing_window <= 0:
            raise ValueError("smoothing_window must be greater than 0")

    @classmethod
    def from_mapping(
        cls,
        root: Mapping[str, Any],
        experiment: Mapping[str, Any],
    ) -> "ExperimentConfig":
        """Construct the experiment solely from the TOML configuration."""

        paper_channel = root["paper"]["channel"]
        assumed_channel = root["assumptions"]["channel"]
        vehicle = experiment["initial_vehicle"]
        uav = experiment["initial_uav"]
        vehicle_bounds = experiment["vehicle_bounds"]
        uav_bounds = experiment["uav_bounds"]

        def vector3(value: Sequence[float]) -> tuple[float, float, float]:
            if len(value) != 3:
                raise ValueError("configured mobility vectors must have length 3")
            return (float(value[0]), float(value[1]), float(value[2]))

        return cls(
            number_of_slots=experiment["number_of_slots"],
            time_step_s=experiment["time_step_s"],
            random_seed=experiment["random_seed"],
            vehicle_mobility_noise_std_m=experiment[
                "vehicle_mobility_noise_std_m"
            ],
            uav_mobility_noise_std_m=experiment["uav_mobility_noise_std_m"],
            vehicle_gain_db=experiment["vehicle_gain_db"],
            uav_gain_db=experiment["uav_gain_db"],
            wavelength_m=paper_channel["wavelength_m"],
            bandwidth_hz=experiment["bandwidth_hz"],
            transmit_power_w=experiment["transmit_power_w"],
            noise_psd_w_hz=noise_psd_w_per_hz(
                assumed_channel["thermal_noise_density_dbm_hz"],
                assumed_channel["receiver_noise_figure_db"],
            ),
            interference_power_w=assumed_channel[
                "default_interference_power_w"
            ],
            blockage_probability=paper_channel["blockage_probability"],
            rician_k_factor_db=assumed_channel["rician_k_factor_db"],
            los_loss_range_db=tuple(assumed_channel["los_additional_loss_db"]),
            blocked_loss_range_db=tuple(
                assumed_channel["blocked_additional_loss_db"]
            ),
            link_available=assumed_channel["default_link_available"],
            smoothing_window=experiment["smoothing_window"],
            initial_vehicle_position_m=vector3(vehicle["position_m"]),
            initial_vehicle_velocity_mps=vector3(vehicle["velocity_mps"]),
            initial_uav_position_m=vector3(uav["position_m"]),
            initial_uav_velocity_mps=vector3(uav["velocity_mps"]),
            vehicle_bounds_minimum_m=vector3(vehicle_bounds["minimum_m"]),
            vehicle_bounds_maximum_m=vector3(vehicle_bounds["maximum_m"]),
            uav_bounds_minimum_m=vector3(uav_bounds["minimum_m"]),
            uav_bounds_maximum_m=vector3(uav_bounds["maximum_m"]),
            output_csv=Path(experiment["output_csv"]),
            output_figure=Path(experiment["output_figure"]),
        )


@dataclass(frozen=True)
class SlotRecord:
    """Values recorded from one simulation time slot."""

    slot: int
    time_s: float

    vehicle_x_m: float
    vehicle_y_m: float
    vehicle_z_m: float

    uav_x_m: float
    uav_y_m: float
    uav_z_m: float

    distance_m: float
    blocked: bool
    fading_model: str
    shadow_factor: float
    fading_power_gain: float
    channel_gain: float
    channel_gain_db: float
    sinr: float
    rate_bps: float
    rate_mbps: float


def linear_to_db(value: float, floor: float = 1.0e-300) -> float:
    """Convert a positive linear power ratio to decibels."""
    return 10.0 * np.log10(max(float(value), floor))


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Return a centered moving average without zero-padding artifacts."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")

    window = min(int(window), len(values))
    if window <= 1:
        return values.copy()

    left = window // 2
    right = window - left
    result = np.empty_like(values)

    for index in range(len(values)):
        start = max(0, index - left)
        stop = min(len(values), index + right)
        result[index] = np.mean(values[start:stop])

    return result


def initial_states(config: ExperimentConfig) -> tuple[MobilityState, MobilityState]:
    """Create one ground vehicle and one UAV for the demonstration.

    The vehicle initially approaches the UAV and later moves away from it.
    Consequently, the distance curve should first decrease and then increase.
    """
    vehicle = MobilityState(
        position_m=np.array(config.initial_vehicle_position_m),
        velocity_mps=np.array(config.initial_vehicle_velocity_mps),
    )
    uav = MobilityState(
        position_m=np.array(config.initial_uav_position_m),
        velocity_mps=np.array(config.initial_uav_velocity_mps),
    )
    return vehicle, uav


def run_experiment(config: ExperimentConfig) -> list[SlotRecord]:
    """Run the mobility-channel-rate pipeline for all time slots."""
    rng = np.random.default_rng(config.random_seed)
    vehicle, uav = initial_states(config)

    vehicle_bounds = SimulationBounds(
        minimum_m=np.array(config.vehicle_bounds_minimum_m),
        maximum_m=np.array(config.vehicle_bounds_maximum_m),
    )
    uav_bounds = SimulationBounds(
        minimum_m=np.array(config.uav_bounds_minimum_m),
        maximum_m=np.array(config.uav_bounds_maximum_m),
    )

    records: list[SlotRecord] = []

    for slot in range(config.number_of_slots):
        # Equations (4) and (5): update vehicle and UAV positions.
        vehicle_noise = sample_mobility_noise(
            config.vehicle_mobility_noise_std_m,
            rng=rng,
            horizontal_only=True,
        )
        uav_noise = sample_mobility_noise(
            config.uav_mobility_noise_std_m,
            rng=rng,
        )

        vehicle = update_vehicle_position(
            vehicle,
            config.time_step_s,
            noise_m=vehicle_noise,
            bounds=vehicle_bounds,
        )
        uav = update_uav_position(
            uav,
            config.time_step_s,
            noise_m=uav_noise,
            bounds=uav_bounds,
        )

        # Equations (6)-(8): distance, random channel and achievable rate.
        link = sample_wireless_link(
            rng=rng,
            vehicle_position=vehicle.position_m,
            node_position=uav.position_m,
            vehicle_gain_db=config.vehicle_gain_db,
            node_gain_db=config.uav_gain_db,
            bandwidth_hz=config.bandwidth_hz,
            transmit_power_w=config.transmit_power_w,
            noise_psd_w_hz=config.noise_psd_w_hz,
            interference_power_w=config.interference_power_w,
            wavelength_m=config.wavelength_m,
            blockage_probability=config.blockage_probability,
            rician_k_factor_db=config.rician_k_factor_db,
            los_loss_range_db=config.los_loss_range_db,
            blocked_loss_range_db=config.blocked_loss_range_db,
            link_available=config.link_available,
        )

        channel = link.channel
        records.append(
            SlotRecord(
                slot=slot,
                time_s=(slot + 1) * config.time_step_s,
                vehicle_x_m=float(vehicle.position_m[0]),
                vehicle_y_m=float(vehicle.position_m[1]),
                vehicle_z_m=float(vehicle.position_m[2]),
                uav_x_m=float(uav.position_m[0]),
                uav_y_m=float(uav.position_m[1]),
                uav_z_m=float(uav.position_m[2]),
                distance_m=channel.distance_m,
                blocked=channel.blocked,
                fading_model=channel.fading_model,
                shadow_factor=channel.shadow_factor,
                fading_power_gain=channel.fading_power_gain,
                channel_gain=channel.channel_gain,
                channel_gain_db=linear_to_db(channel.channel_gain),
                sinr=link.sinr,
                rate_bps=link.rate_bps,
                rate_mbps=link.rate_mbps,
            )
        )

    return records


def save_records_csv(records: list[SlotRecord], output_path: Path) -> None:
    """Save all per-slot values so later experiments can reuse them."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    field_names = list(SlotRecord.__dataclass_fields__)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {name: getattr(record, name) for name in field_names}
            )


def plot_results(
    records: list[SlotRecord],
    config: ExperimentConfig,
    output_path: Path,
) -> None:
    """Plot distance, gain, rate and blockage state over time."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    time_s = np.array([record.time_s for record in records])
    distance_m = np.array([record.distance_m for record in records])
    gain_db = np.array([record.channel_gain_db for record in records])
    rate_mbps = np.array([record.rate_mbps for record in records])
    blocked = np.array([record.blocked for record in records], dtype=int)

    smooth_gain_db = moving_average(gain_db, config.smoothing_window)
    smooth_rate_mbps = moving_average(rate_mbps, config.smoothing_window)

    figure, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(10.0, 11.0),
        sharex=True,
    )

    axes[0].plot(time_s, distance_m, color="#1f77b4", linewidth=2.0)
    axes[0].set_ylabel("Distance (m)")
    axes[0].set_title("DM-JCR dynamic mobility and wireless channel")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        time_s,
        gain_db,
        color="#9ecae1",
        linewidth=1.0,
        label="instantaneous",
    )
    axes[1].plot(
        time_s,
        smooth_gain_db,
        color="#08519c",
        linewidth=2.0,
        label=f"{config.smoothing_window}-slot mean",
    )
    axes[1].set_ylabel("Channel gain (dB)")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    axes[2].plot(
        time_s,
        rate_mbps,
        color="#a1d99b",
        linewidth=1.0,
        label="instantaneous",
    )
    axes[2].plot(
        time_s,
        smooth_rate_mbps,
        color="#238b45",
        linewidth=2.0,
        label=f"{config.smoothing_window}-slot mean",
    )
    axes[2].set_ylabel("Rate (Mbit/s)")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.25)

    axes[3].step(
        time_s,
        blocked,
        where="mid",
        color="#d62728",
        linewidth=1.5,
    )
    axes[3].set_yticks([0, 1], labels=["LoS", "blocked"])
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Link state")
    axes[3].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def print_summary(records: list[SlotRecord], config: ExperimentConfig) -> None:
    """Print numerical checks for the expected physical trends."""
    distance_m = np.array([record.distance_m for record in records])
    rate_mbps = np.array([record.rate_mbps for record in records])
    blocked = np.array([record.blocked for record in records], dtype=bool)

    smooth_rate = moving_average(rate_mbps, config.smoothing_window)
    distance_rate_correlation = float(
        np.corrcoef(distance_m, smooth_rate)[0, 1]
    )

    los_rate = rate_mbps[~blocked]
    blocked_rate = rate_mbps[blocked]

    print("Dynamic channel experiment completed")
    print(f"Slots: {len(records)}")
    print(
        "Distance range: "
        f"{distance_m.min():.2f} m to {distance_m.max():.2f} m"
    )
    print(f"Observed blockage ratio: {blocked.mean():.3f}")
    print(f"Mean LoS rate: {los_rate.mean():.3f} Mbit/s")
    print(f"Mean blocked rate: {blocked_rate.mean():.3f} Mbit/s")
    print(
        "Correlation(distance, smoothed rate): "
        f"{distance_rate_correlation:.3f}"
    )
    print("Expected: blocked rate < LoS rate")
    print("Expected: distance-rate correlation is usually negative")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the experiment and write its two output files."""
    root, experiment = load_script_config(
        "dynamic_channel",
        __doc__ or "Dynamic channel check",
        argv,
    )
    config = ExperimentConfig.from_mapping(root, experiment)
    records = run_experiment(config)

    csv_path = config.output_csv
    figure_path = config.output_figure

    save_records_csv(records, csv_path)
    plot_results(records, config, figure_path)
    print_summary(records, config)

    print(f"CSV saved to: {csv_path.resolve()}")
    print(f"Figure saved to: {figure_path.resolve()}")


if __name__ == "__main__":
    main()
