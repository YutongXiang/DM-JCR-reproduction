"""验证论文图 4去噪轨迹实验的严格条件与结果输出。"""

from dataclasses import replace
import json

import numpy as np
import pytest

from dm_jcr.config import get_experiment, load_config
from dm_jcr.diffusion import diffusion_spec_from_config
from dm_jcr.diffusion_objective import objective_tensor_spec_from_config
from dm_jcr.diffusion_training import StrategyGenerationTrace
from experiments.experiment_figure4_denoising_trace import (
    Figure4Config,
    save_results,
    select_trace,
    validate_paper_requirements,
)


def _config() -> Figure4Config:
    root = load_config()
    return Figure4Config.from_mapping(
        root, get_experiment(root, "figure4_denoising_trace")
    )


def _trace() -> StrategyGenerationTrace:
    raw = np.array([[0.7, 0.8], [0.5, 0.9], [0.4, 0.6]])
    best = np.minimum.accumulate(raw, axis=0)
    return StrategyGenerationTrace(
        generation_steps=np.array([0, 1, 2]),
        diffusion_timesteps=np.array([2, 1, 0]),
        categories=np.array([0, 1]),
        raw_weighted_objective=raw,
        raw_normalized_latency=raw + 0.1,
        raw_normalized_energy=raw - 0.1,
        raw_deadline_satisfied_ratio=np.full_like(raw, 0.5),
        best_weighted_objective=best,
        best_normalized_latency=best + 0.1,
        best_normalized_energy=best - 0.1,
        best_deadline_satisfied_ratio=np.full_like(best, 0.5),
    )


def test_figure4_config_matches_recommended_200_step_setting() -> None:
    root = load_config()
    config = _config()
    validate_paper_requirements(
        root,
        config,
        diffusion_spec_from_config(root),
        objective_tensor_spec_from_config(root),
        {"training_denoising_steps": [200.0]},
    )

    assert config.denoising_steps == 200
    assert config.selection_rule == "raw"


def test_figure4_rejects_checkpoint_not_trained_with_200_steps() -> None:
    root = load_config()
    with pytest.raises(ValueError, match="200"):
        validate_paper_requirements(
            root,
            _config(),
            diffusion_spec_from_config(root),
            objective_tensor_spec_from_config(root),
            {"training_denoising_steps": [1000.0]},
        )


def test_best_so_far_selection_and_output_files(tmp_path) -> None:
    trace = _trace()
    weighted, *_ = select_trace(trace, "best_so_far")
    np.testing.assert_array_equal(weighted, np.minimum.accumulate(weighted, axis=0))
    config = replace(
        _config(),
        output_csv=tmp_path / "trace.csv",
        output_summary=tmp_path / "summary.json",
        output_figure=tmp_path / "figure.png",
    )

    summary = save_results(config, trace)

    assert config.output_csv.is_file()
    assert config.output_figure.is_file()
    assert summary["initial_weighted_objective"] == 0.75
    assert summary["final_weighted_objective"] == 0.5
    saved = json.loads(config.output_summary.read_text(encoding="utf-8"))
    assert saved["paper_requirements"]["image_shape"] == [1, 28, 28]
