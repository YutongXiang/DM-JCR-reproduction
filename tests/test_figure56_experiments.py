"""验证图5、图6和图7的配置、聚合统计及输出文件。"""

import json

import pytest

from dm_jcr.config import get_experiment, load_config
from experiments._figure56_dm_jcr_sweep import (
    SweepRecord,
    TaskSweepConfig,
    save_metric_results,
)


def _records() -> list[SweepRecord]:
    return [
        SweepRecord(10, 0, 1, 11, 2.0, 4.0, 0.2, 0.4, 0.1, 0.2, 0.3, True, True),
        SweepRecord(10, 1, 2, 11, 4.0, 6.0, 0.4, 0.6, 0.2, 0.3, 0.5, True, False),
        SweepRecord(30, 0, 3, 12, 6.0, 8.0, 0.6, 0.8, 0.3, 0.4, 0.7, True, True),
        SweepRecord(30, 1, 4, 12, 8.0, 10.0, 0.8, 1.0, 0.4, 0.5, 0.9, True, True),
    ]


def test_figure5_figure6_and_figure7_use_paired_task_sweeps() -> None:
    root = load_config()
    figure5 = TaskSweepConfig.from_mapping(get_experiment(root, "figure5_task_latency"))
    figure6 = TaskSweepConfig.from_mapping(get_experiment(root, "figure6_task_energy"))
    figure7 = TaskSweepConfig.from_mapping(get_experiment(root, "figure7_weighted_indicator"))

    assert figure5.task_counts == tuple(range(10, 151, 10))
    assert figure5.task_counts == figure6.task_counts
    assert figure5.task_counts == figure7.task_counts
    assert figure5.trials_per_point == figure6.trials_per_point == figure7.trials_per_point
    assert figure5.random_seed == figure6.random_seed == figure7.random_seed
    assert figure5.denoising_steps == figure6.denoising_steps == figure7.denoising_steps == 200
    figure5.validate(150)
    figure6.validate(150)
    figure7.validate(150)


def test_metric_result_writer_generates_csv_json_and_png(tmp_path) -> None:
    config = TaskSweepConfig(
        checkpoint=tmp_path / "model.pt",
        task_counts=(10, 30),
        trials_per_point=2,
        random_seed=7,
        denoising_steps=200,
        inference_batch_size=2,
        device="cpu",
        output_csv=tmp_path / "values.csv",
        output_summary=tmp_path / "summary.json",
        output_figure=tmp_path / "figure.png",
    )

    summary = save_metric_results(config, _records(), "latency", 5)

    assert summary["points"][0]["mean"] == pytest.approx(0.15)
    assert summary["points"][0]["deadline_feasible_ratio"] == 0.5
    assert config.output_csv.exists()
    assert config.output_figure.exists()
    assert json.loads(config.output_summary.read_text(encoding="utf-8"))["figure"] == 5
