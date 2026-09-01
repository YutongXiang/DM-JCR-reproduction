"""实验脚本统一配置来源的单元测试。"""

from pathlib import Path

import pytest

from dm_jcr.config import (
    ConfigError,
    DEFAULT_CONFIG_PATH,
    get_experiment,
    load_config,
)
from scripts._config_helpers import (
    capacities,
    direct_context,
    objective_settings,
    raw_direct,
    raw_relay,
    raw_v2v,
    relay_context,
    v2v_context,
)
from scripts.check_dynamic_channel import ExperimentConfig


def test_default_config_separates_paper_values_and_assumptions() -> None:
    config = load_config()

    assert DEFAULT_CONFIG_PATH.name == "reproduction.toml"
    assert config["paper"]["channel"]["blockage_probability"] == pytest.approx(0.3)
    assert config["assumptions"]["channel"]["rician_k_factor_db"] == pytest.approx(6.0)
    assert config["assumptions"]["mobility"]["noise_distribution"] == (
        "zero_mean_independent_gaussian"
    )


def test_every_configured_experiment_can_construct_runtime_objects() -> None:
    config = load_config()

    direct = get_experiment(config, "direct_resource")
    assert len(capacities(direct["capacities"])) == 1
    assert all(direct_context(item) for item in direct["tasks"])
    assert all(raw_direct(item) for item in direct["tasks"])

    relay = get_experiment(config, "relay_resource")
    assert len(capacities(relay["capacities"])) == 2
    assert all(relay_context(item) for item in relay["relay_tasks"])
    assert all(raw_relay(item) for item in relay["relay_tasks"])

    equation17 = get_experiment(config, "equation17")
    assert all(v2v_context(item) for item in equation17["v2v_tasks"])
    assert all(raw_v2v(item) for item in equation17["v2v_tasks"])

    normalization, weights, energy_coefficient = objective_settings(config)
    assert normalization.energy_reference_j == pytest.approx(10.0)
    assert weights.latency == pytest.approx(0.5)
    assert energy_coefficient == pytest.approx(5.0e-28)


def test_dynamic_channel_config_is_fully_loaded_from_toml() -> None:
    config = load_config()
    dynamic = ExperimentConfig.from_mapping(
        config,
        get_experiment(config, "dynamic_channel"),
    )

    assert dynamic.random_seed == 2026
    assert dynamic.initial_uav_position_m == pytest.approx((600.0, 100.0, 100.0))
    assert dynamic.rician_k_factor_db == pytest.approx(6.0)
    assert dynamic.output_csv == Path("outputs/dynamic_channel.csv")


def test_unsupported_schema_is_rejected(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.toml"
    bad_config.write_text(
        "[metadata]\nschema_version=2\n[paper]\n[assumptions]\n[experiments]\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="schema_version"):
        load_config(bad_config)
