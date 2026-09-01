"""加载 DM-JCR 实验统一使用的 TOML 配置。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import tomllib
from typing import Any


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "reproduction.toml"
)


class ConfigError(ValueError):
    """Raised when the reproduction configuration is malformed."""


def _validate_supported_assumptions(data: Mapping[str, Any]) -> None:
    """Reject policy edits that the current implementation cannot honor."""

    assumptions = data["assumptions"]
    supported = {
        ("units", "bytes_per_kilobyte"): 1000,
        ("mobility", "noise_distribution"): (
            "zero_mean_independent_gaussian"
        ),
        ("mobility", "boundary_policy"): "clip",
        ("objective", "latency_normalization"): "per_task_deadline",
        ("relay_accounting", "bandwidth_policy"): (
            "charge_v2u_u2n_u2v_to_relay_and_n2u_to_compute_node"
        ),
        ("relay_accounting", "relay_power_policy"): (
            "reuse_one_relay_power_for_both_outbound_hops"
        ),
    }
    for (section, key), expected in supported.items():
        actual = assumptions.get(section, {}).get(key)
        if actual != expected:
            raise ConfigError(
                f"unsupported assumptions.{section}.{key}={actual!r}; "
                f"current implementation requires {expected!r}"
            )


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read and minimally validate a DM-JCR TOML configuration."""

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")

    with config_path.open("rb") as file:
        data = tomllib.load(file)

    version = data.get("metadata", {}).get("schema_version")
    if version != 1:
        raise ConfigError(
            f"unsupported configuration schema_version {version!r}; expected 1"
        )
    for section in ("paper", "assumptions", "experiments"):
        if not isinstance(data.get(section), Mapping):
            raise ConfigError(f"missing or invalid [{section}] section")
    _validate_supported_assumptions(data)
    return data


def get_experiment(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return one named experiment section with a useful validation error."""

    experiments = config.get("experiments")
    if not isinstance(experiments, Mapping):
        raise ConfigError("missing or invalid [experiments] section")
    experiment = experiments.get(name)
    if not isinstance(experiment, Mapping):
        raise ConfigError(f"missing or invalid [experiments.{name}] section")
    return experiment


__all__ = [
    "ConfigError",
    "DEFAULT_CONFIG_PATH",
    "get_experiment",
    "load_config",
]
