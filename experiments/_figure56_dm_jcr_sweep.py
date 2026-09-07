"""为论文图5、图6和图7提供共享的DM-JCR任务规模扫描与结果保存功能。"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Literal, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dm_jcr.data_generation import GeneratedScenario, generate_scenario, scenario_spec_from_config
from dm_jcr.diffusion_training import generate_resource_strategy, load_bundle
from dm_jcr.environment import EncodedEnvironment, encode_environment, tensor_spec_from_config
from dm_jcr.resource_allocation import (
    ObjectiveNormalization,
    ObjectiveWeights,
    evaluate_equation17_strategy,
    project_equation17_strategy,
)
from dm_jcr.strategy_codec import EncodedRawStrategy, decode_raw_strategy


MetricName = Literal["latency", "energy", "weighted"]


@dataclass(frozen=True)
class TaskSweepConfig:
    """图5、图6或图7一次独立DM-JCR任务规模扫描的配置。"""

    checkpoint: Path
    task_counts: tuple[int, ...]
    trials_per_point: int
    random_seed: int
    denoising_steps: int
    inference_batch_size: int
    device: str
    output_csv: Path
    output_summary: Path
    output_figure: Path

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TaskSweepConfig":
        return cls(
            checkpoint=Path(values["checkpoint"]),
            task_counts=tuple(int(value) for value in values["task_counts"]),
            trials_per_point=int(values["trials_per_point"]),
            random_seed=int(values["random_seed"]),
            denoising_steps=int(values["denoising_steps"]),
            inference_batch_size=int(values["inference_batch_size"]),
            device=str(values["device"]),
            output_csv=Path(values["output_csv"]),
            output_summary=Path(values["output_summary"]),
            output_figure=Path(values["output_figure"]),
        )

    def validate(self, maximum_tasks: int) -> None:
        if not self.task_counts or any(value <= 0 for value in self.task_counts):
            raise ValueError("task_counts必须包含正整数")
        if tuple(sorted(set(self.task_counts))) != self.task_counts:
            raise ValueError("task_counts必须严格递增且不能重复")
        if self.task_counts[-1] > maximum_tasks:
            raise ValueError(f"任务数{self.task_counts[-1]}超过模型容量{maximum_tasks}")
        if self.trials_per_point <= 0 or self.inference_batch_size <= 0:
            raise ValueError("trials_per_point和inference_batch_size必须为正数")
        if self.denoising_steps <= 0:
            raise ValueError("denoising_steps必须为正数")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device必须为auto、cpu或cuda")


@dataclass(frozen=True)
class SweepRecord:
    """一次随机场景上的DM-JCR最终策略评价结果。"""

    task_count: int
    trial: int
    scenario_seed: int
    inference_seed: int
    average_latency_s: float
    average_energy_j: float
    normalized_latency: float
    normalized_energy: float
    weighted_normalized_latency: float
    weighted_normalized_energy: float
    weighted_objective: float
    resource_feasible: bool
    deadline_feasible: bool


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求使用CUDA，但当前PyTorch无法使用CUDA")
    return torch.device(name)


def _absolute_means(evaluation: Any) -> tuple[float, float]:
    task_results = (
        *evaluation.direct_tasks,
        *evaluation.relay_computation_tasks,
        *evaluation.v2v_relay_tasks,
    )
    latencies = [float(result.metrics.total_latency_s) for result in task_results]
    energies = [float(result.metrics.total_energy_j) for result in task_results]
    if not latencies:
        raise ValueError("场景中没有可评价任务")
    return float(np.mean(latencies)), float(np.mean(energies))


def _make_scenarios(
    root: Mapping[str, Any], config: TaskSweepConfig
) -> tuple[list[GeneratedScenario], list[EncodedEnvironment], list[tuple[int, int, int]]]:
    base_spec = scenario_spec_from_config(root)
    tensor_spec = tensor_spec_from_config(root)
    seed_sequence = np.random.SeedSequence(config.random_seed)
    children = seed_sequence.spawn(len(config.task_counts) * config.trials_per_point)
    scenarios: list[GeneratedScenario] = []
    encoded: list[EncodedEnvironment] = []
    metadata: list[tuple[int, int, int]] = []
    index = 0
    for task_count in config.task_counts:
        fixed_spec = replace(base_spec, task_count_range=(task_count, task_count))
        for trial in range(config.trials_per_point):
            scenario_seed = int(children[index].generate_state(1, dtype=np.uint32)[0])
            scenario = generate_scenario(np.random.default_rng(scenario_seed), index, fixed_spec)
            scenarios.append(scenario)
            encoded.append(encode_environment(scenario.snapshot, tensor_spec))
            metadata.append((task_count, trial, scenario_seed))
            index += 1
    return scenarios, encoded, metadata


def run_sweep(root: Mapping[str, Any], config: TaskSweepConfig) -> list[SweepRecord]:
    """生成固定任务数场景，批量执行200步DM-JCR推理并按公式（17）评价。"""

    device = _device(config.device)
    bundle, history = load_bundle(config.checkpoint, device)
    config.validate(bundle.model_spec.model_max_tasks)
    if config.denoising_steps > bundle.model_spec.denoising_steps:
        raise ValueError("实验去噪步数超过检查点支持的扩散调度长度")
    training_steps = history.get("training_denoising_steps", [])
    if not training_steps or int(training_steps[-1]) != config.denoising_steps:
        raise ValueError("检查点训练去噪步数与图5/图6实验设置不一致")

    scenarios, encoded, metadata = _make_scenarios(root, config)
    normalization = ObjectiveNormalization(root["assumptions"]["objective"]["energy_reference_j"])
    weights = ObjectiveWeights(
        root["paper"]["objective"]["latency_weight"],
        root["paper"]["objective"]["energy_weight"],
    )
    coefficient = float(root["paper"]["task"]["cpu_energy_coefficient"])
    records: list[SweepRecord] = []

    for start in range(0, len(scenarios), config.inference_batch_size):
        stop = min(start + config.inference_batch_size, len(scenarios))
        batch = encoded[start:stop]
        inference_seed = config.random_seed + 1_000_003 + start
        strategies = generate_resource_strategy(
            bundle,
            np.stack([item.flat_vector() for item in batch]),
            np.stack([item.task_features for item in batch]),
            np.stack([item.task_mask for item in batch]),
            device,
            random_seed=inference_seed,
            denoising_steps=config.denoising_steps,
        )
        for offset, scores in enumerate(strategies):
            global_index = start + offset
            scenario = scenarios[global_index]
            environment = encoded[global_index]
            task_count, trial, scenario_seed = metadata[global_index]
            raw = EncodedRawStrategy(scores, environment.task_mask, scenario.snapshot.mappings)
            direct, relay, v2v = decode_raw_strategy(raw)
            projected = project_equation17_strategy(direct, relay, v2v, scenario.capacities)
            evaluation = evaluate_equation17_strategy(
                scenario.direct_contexts,
                scenario.relay_contexts,
                scenario.v2v_contexts,
                projected,
                scenario.capacities,
                normalization,
                weights,
                energy_coefficient=coefficient,
            )
            average_latency, average_energy = _absolute_means(evaluation)
            records.append(
                SweepRecord(
                    task_count,
                    trial,
                    scenario_seed,
                    inference_seed,
                    average_latency,
                    average_energy,
                    evaluation.mean_normalized_latency,
                    evaluation.mean_normalized_energy,
                    weights.latency * evaluation.mean_normalized_latency,
                    weights.energy * evaluation.mean_normalized_energy,
                    evaluation.weighted_objective,
                    evaluation.resource_constraints_satisfied,
                    evaluation.deadline_constraints_satisfied,
                )
            )
        print(f"DM-JCR推理进度：{stop}/{len(scenarios)}", flush=True)
    return records


def save_metric_results(
    config: TaskSweepConfig,
    records: list[SweepRecord],
    metric: MetricName,
    figure_number: int,
) -> dict[str, Any]:
    """保存逐场景数据、按任务数聚合的统计信息以及论文风格曲线。"""

    for path in (config.output_csv, config.output_summary, config.output_figure):
        path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(SweepRecord.__dataclass_fields__)
    with config.output_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({name: getattr(record, name) for name in fields} for record in records)

    attributes = {
        "latency": "weighted_normalized_latency",
        "energy": "weighted_normalized_energy",
        "weighted": "weighted_objective",
    }
    attribute = attributes[metric]
    points = []
    for task_count in config.task_counts:
        group = [record for record in records if record.task_count == task_count]
        values = np.asarray([getattr(record, attribute) for record in group], dtype=np.float64)
        points.append(
            {
                "task_count": task_count,
                "samples": len(group),
                "mean": float(values.mean()),
                "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "standard_error": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
                "resource_feasible_ratio": float(np.mean([item.resource_feasible for item in group])),
                "deadline_feasible_ratio": float(np.mean([item.deadline_feasible for item in group])),
            }
        )
    summary = {
        "figure": figure_number,
        "method": "DM-JCR",
        "metric": metric,
        "denoising_steps": config.denoising_steps,
        "trials_per_point": config.trials_per_point,
        "random_seed": config.random_seed,
        "checkpoint": str(config.checkpoint),
        "paper_disclosure": {
            "task_range": "论文图中覆盖10至150个任务",
            "trials_per_point": "论文未公开每个横轴点的独立重复次数，使用配置中的复现假设",
            "scope": "只生成DM-JCR曲线，不包含Random、Greedy、SAC和DDPG基线",
            "normalization": "论文未公开归一化公式，使用统一配置中声明的复现归一化假设",
            "component_relation": "图5和图6分别绘制乘以0.5权重后的归一化分项，图7为两者之和",
        },
        "points": points,
    }
    config.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    x = np.asarray([point["task_count"] for point in points])
    y = np.asarray([point["mean"] for point in points])
    labels = {
        "latency": "Normalized Average Latency",
        "energy": "Normalized Average Energy Consumption",
        "weighted": "Normalized Average Weighted Indicator",
    }
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    axis.plot(x, y, color="crimson", marker="o", linewidth=1.6, label="DM-JCR(proposed)")
    axis.set_xlabel("Number of Computational Tasks")
    axis.set_ylabel(labels[metric])
    axis.set_xticks(x)
    axis.grid(alpha=0.35, linestyle="--")
    axis.legend()
    figure.tight_layout()
    figure.savefig(config.output_figure, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return summary


__all__ = ["TaskSweepConfig", "run_sweep", "save_metric_results"]
