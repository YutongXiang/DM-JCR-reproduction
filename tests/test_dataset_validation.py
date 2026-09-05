"""验证训练前数据集检查能够接受合法数据并拒绝损坏数据。"""

from __future__ import annotations

import numpy as np

from dm_jcr.config import load_config
from dm_jcr.data_generation import generate_and_save_dataset
from dm_jcr.dataset_validation import validate_diffusion_dataset


def _small_dataset(tmp_path):
    config = load_config()
    generate_and_save_dataset(
        config,
        tmp_path,
        {"train": 2, "validation": 1, "test": 1},
        random_seed=73,
        candidate_count=2,
    )
    return config


def test_structural_profile_accepts_current_generator_output(tmp_path) -> None:
    config = _small_dataset(tmp_path)

    report = validate_diffusion_dataset(config, tmp_path, profile="structural")

    assert report["valid"]
    assert not report["formal_training_ready"]
    assert report["issues"] == []


def test_formal_profile_rejects_too_few_samples(tmp_path) -> None:
    config = _small_dataset(tmp_path)

    report = validate_diffusion_dataset(config, tmp_path, profile="formal")

    assert not report["valid"]
    assert any("正式训练至少需要" in issue["message"] for issue in report["issues"])


def test_validator_detects_modified_objective(tmp_path) -> None:
    config = _small_dataset(tmp_path)
    path = tmp_path / "train.npz"
    with np.load(path) as source:
        arrays = {name: source[name].copy() for name in source.files}
    arrays["objective"][0] += 1.0
    np.savez_compressed(path, **arrays)

    report = validate_diffusion_dataset(config, tmp_path, profile="structural")

    assert not report["valid"]
    assert any("重算目标值" in issue["message"] for issue in report["issues"])
