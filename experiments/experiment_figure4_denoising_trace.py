"""按论文建议的200步实用设置记录 DM-JCR 的确定性生成轨迹。

论文正式图4使用1000步，并指出约200步可兼顾指标和开销；本脚本采用后者。
其余要求保持为28×28单通道策略表示、确定性去噪、时延与能耗权重均为0.5。
横轴“Generating Epoch”按正文解释为已经完成的去噪步骤，而不是训练 epoch。

在项目根目录运行：

    python -m experiments.experiment_figure4_denoising_trace
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dm_jcr.config import DEFAULT_CONFIG_PATH, get_experiment, load_config
from dm_jcr.diffusion_training import (
    StrategyGenerationTrace,
    generate_resource_strategy_trace,
    load_bundle,
    load_diffusion_split,
)
from scripts._config_helpers import ChineseArgumentParser


SelectionRule = Literal["raw", "best_so_far"]


@dataclass(frozen=True)
class Figure4Config:
    """论文图 4实验配置。"""

    checkpoint: Path
    dataset: Path
    samples: int
    random_seed: int
    denoising_steps: int
    selection_rule: SelectionRule
    latency_normalization: str
    energy_reference_j: float
    device: str
    output_csv: Path
    output_summary: Path
    output_figure: Path

    def __post_init__(self) -> None:
        if self.samples <= 0:
            raise ValueError("samples 必须是正整数")
        if self.denoising_steps <= 0:
            raise ValueError("denoising_steps 必须是正整数")
        if self.selection_rule not in ("raw", "best_so_far"):
            raise ValueError("selection_rule 必须是 raw 或 best_so_far")
        if self.latency_normalization != "per_task_deadline":
            raise ValueError("当前实现只支持按任务截止期归一化时延")
        if self.energy_reference_j <= 0.0:
            raise ValueError("energy_reference_j 必须大于 0")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device 必须是 auto、cpu 或 cuda")

    @classmethod
    def from_mapping(
        cls,
        root: Mapping[str, Any],
        values: Mapping[str, Any],
    ) -> "Figure4Config":
        objective = root["assumptions"]["objective"]
        return cls(
            checkpoint=Path(values["checkpoint"]),
            dataset=Path(values["dataset"]),
            samples=int(values["samples"]),
            random_seed=int(values["random_seed"]),
            denoising_steps=int(values["denoising_steps"]),
            selection_rule=values["selection_rule"],
            latency_normalization=str(objective["latency_normalization"]),
            energy_reference_j=float(objective["energy_reference_j"]),
            device=str(values["device"]),
            output_csv=Path(values["output_csv"]),
            output_summary=Path(values["output_summary"]),
            output_figure=Path(values["output_figure"]),
        )


def validate_paper_requirements(
    root: Mapping[str, Any],
    config: Figure4Config,
    trace_model_spec: Any,
    trace_objective_spec: Any,
    history: Mapping[str, Any],
) -> None:
    """校验论文结构要求和200步低成本实验设置。"""

    paper_diffusion = root["paper"]["diffusion"]
    objective = root["paper"]["objective"]
    paper_steps = int(paper_diffusion["denoising_steps"])
    experiment_steps = 200
    if trace_model_spec.denoising_steps != paper_steps:
        raise ValueError(f"模型必须保留论文规定的 {paper_steps} 级扩散调度")
    if config.denoising_steps != experiment_steps:
        raise ValueError(f"图4低成本复现实验固定使用 {experiment_steps} 个去噪步骤")
    if trace_model_spec.image_size != 28 or trace_model_spec.image_channels != 1:
        raise ValueError("论文图 4要求使用28×28单通道扩散张量")
    if tuple(trace_model_spec.feature_channels) != (64, 128, 256):
        raise ValueError("论文图 4要求U-Net使用64、128、256三级特征通道")
    if paper_diffusion["deterministic_reverse_sampling"] is not True:
        raise ValueError("论文图 4要求采用公式（19）的确定性反向采样")
    if objective["latency_weight"] != 0.5 or objective["energy_weight"] != 0.5:
        raise ValueError("论文图 4要求时延和能耗权重ρ、ξ均为0.5")
    if trace_objective_spec.latency_weight != 0.5 or trace_objective_spec.energy_weight != 0.5:
        raise ValueError("检查点中的目标权重与论文图 4不一致")
    training_steps = history.get("training_denoising_steps", [])
    if not training_steps or int(training_steps[-1]) != experiment_steps:
        raise ValueError("检查点必须使用200个去噪步骤训练")


def select_trace(trace: StrategyGenerationTrace, rule: SelectionRule) -> tuple[np.ndarray, ...]:
    """选择原始轨迹或论文所述只接受更优方案的当前最优轨迹。"""

    if rule == "raw":
        return (
            trace.raw_weighted_objective,
            trace.raw_normalized_latency,
            trace.raw_normalized_energy,
            trace.raw_deadline_satisfied_ratio,
        )
    return (
        trace.best_weighted_objective,
        trace.best_normalized_latency,
        trace.best_normalized_energy,
        trace.best_deadline_satisfied_ratio,
    )


def save_results(config: Figure4Config, trace: StrategyGenerationTrace) -> dict[str, Any]:
    """保存逐样本原始轨迹、聚合统计、实验元数据和论文图 4曲线。"""

    for path in (config.output_csv, config.output_summary, config.output_figure):
        path.parent.mkdir(parents=True, exist_ok=True)
    selected = select_trace(trace, config.selection_rule)
    weighted, latency, energy, satisfied = selected
    with config.output_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "generation_step",
                "diffusion_timestep",
                "sample_index",
                "category",
                "raw_weighted_objective",
                "raw_normalized_latency",
                "raw_normalized_energy",
                "raw_deadline_satisfied_ratio",
                "selected_weighted_objective",
                "selected_normalized_latency",
                "selected_normalized_energy",
                "selected_deadline_satisfied_ratio",
            )
        )
        for step_index, generation_step in enumerate(trace.generation_steps):
            for sample_index, category in enumerate(trace.categories):
                writer.writerow(
                    (
                        int(generation_step),
                        int(trace.diffusion_timesteps[step_index]),
                        sample_index,
                        int(category),
                        trace.raw_weighted_objective[step_index, sample_index],
                        trace.raw_normalized_latency[step_index, sample_index],
                        trace.raw_normalized_energy[step_index, sample_index],
                        trace.raw_deadline_satisfied_ratio[step_index, sample_index],
                        weighted[step_index, sample_index],
                        latency[step_index, sample_index],
                        energy[step_index, sample_index],
                        satisfied[step_index, sample_index],
                    )
                )

    mean_weighted = weighted.mean(axis=1)
    summary: dict[str, Any] = {
        "samples": int(weighted.shape[1]),
        "denoising_steps": int(trace.generation_steps[-1]),
        "selection_rule": config.selection_rule,
        "initial_weighted_objective": float(mean_weighted[0]),
        "final_weighted_objective": float(mean_weighted[-1]),
        "minimum_weighted_objective": float(mean_weighted.min()),
        "relative_improvement": float(
            (mean_weighted[0] - mean_weighted[-1]) / max(abs(mean_weighted[0]), 1.0e-12)
        ),
        "paper_requirements": {
            "image_shape": [1, 28, 28],
            "paper_full_denoising_steps": 1000,
            "experiment_denoising_steps": 200,
            "reproduction_scope": "论文建议的200步低成本趋势复现，不是1000步完整曲线",
            "latency_weight": 0.5,
            "energy_weight": 0.5,
            "deterministic_reverse_sampling": True,
            "x_axis_interpretation": "已经完成的去噪步骤（论文横轴标为Generating Epoch）",
        },
        "normalization_disclosure": {
            "paper": "论文仅说明对实验数据归一化，未公开具体公式",
            "latency": config.latency_normalization,
            "energy_reference_j": config.energy_reference_j,
        },
        "random_seed": config.random_seed,
        "checkpoint": str(config.checkpoint),
        "dataset": str(config.dataset),
    }
    config.output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    axis.plot(trace.generation_steps, mean_weighted, color="red", linewidth=1.5, label="DM-JCR")
    axis.set_xlabel("Generating Epoch")
    axis.set_ylabel("Normalized Average Weighted Indicator")
    axis.grid(alpha=0.35, linestyle="--")
    axis.legend()
    figure.tight_layout()
    figure.savefig(config.output_figure, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return summary


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求使用 CUDA，但当前环境不可用")
    return torch.device(name)


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
    config = Figure4Config.from_mapping(
        root, get_experiment(root, "figure4_denoising_trace")
    )
    device = _device(config.device)
    bundle, history = load_bundle(config.checkpoint, device)
    validate_paper_requirements(
        root, config, bundle.model_spec, bundle.objective_spec, history
    )
    data = load_diffusion_split(config.dataset)
    if data["E"].shape[0] < config.samples:
        raise ValueError(
            f"测试集只有 {data['E'].shape[0]} 个样本，少于配置要求的 {config.samples} 个"
        )
    count = config.samples
    trace = generate_resource_strategy_trace(
        bundle,
        data["E"][:count],
        data["node_features"][:count],
        data["channel_gains"][:count],
        data["task_features"][:count],
        data["task_node_indices"][:count],
        data["task_mask"][:count],
        device,
        random_seed=config.random_seed,
        denoising_steps=config.denoising_steps,
    )
    summary = save_results(config, trace)
    print(
        "图 4实验完成："
        f"初始指标={summary['initial_weighted_objective']:.6f}，"
        f"最终指标={summary['final_weighted_objective']:.6f}，"
        f"相对改善={summary['relative_improvement']:.2%}"
    )
    print(f"原始轨迹：{config.output_csv.resolve()}")
    print(f"汇总信息：{config.output_summary.resolve()}")
    print(f"图像：{config.output_figure.resolve()}")


if __name__ == "__main__":
    main()
