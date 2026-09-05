"""验证论文图 3链路时延实验。"""

from dataclasses import replace
import json

import numpy as np

from dm_jcr.config import get_experiment, load_config
from experiments.experiment_figure3_link_latency import (
    Figure3Config,
    run_experiment,
    save_results,
    summarize,
)


def _config() -> Figure3Config:
    root = load_config()
    return Figure3Config.from_mapping(
        root, get_experiment(root, "figure3_link_latency")
    )


def test_figure3_experiment_is_reproducible_and_finite() -> None:
    config = replace(_config(), trials=8)

    first = run_experiment(config)
    second = run_experiment(config)

    assert first == second
    assert len(first) == 8
    values = np.asarray(
        [[item.direct_latency_ms, item.relay_latency_ms] for item in first]
    )
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)
    summary = summarize(first)
    assert summary["direct_blocked_ratio"] == 1.0
    assert 0.0 <= summary["relay_faster_ratio"] <= 1.0


def test_figure3_experiment_writes_all_outputs(tmp_path) -> None:
    config = replace(
        _config(),
        trials=4,
        output_csv=tmp_path / "trials.csv",
        output_summary=tmp_path / "summary.json",
        output_figure=tmp_path / "figure.png",
    )
    records = run_experiment(config)
    summary = summarize(records)

    save_results(config, records, summary)

    assert config.output_csv.is_file()
    assert config.output_figure.is_file()
    payload = json.loads(config.output_summary.read_text(encoding="utf-8"))
    assert payload["summary"]["trials"] == 4
    assert payload["assumptions"]["latency_scope"].startswith("仅计算")
