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
    """复现配置格式错误时抛出的异常。"""


def _validate_supported_assumptions(data: Mapping[str, Any]) -> None:
    """拒绝当前实现尚不支持的复现策略。"""

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
                f"不支持 assumptions.{section}.{key}={actual!r}；"
                f"当前实现要求该值为 {expected!r}"
            )


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """读取 DM-JCR TOML 配置并完成必要校验。"""

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"未找到配置文件：{config_path}")

    with config_path.open("rb") as file:
        data = tomllib.load(file)

    version = data.get("metadata", {}).get("schema_version")
    if version != 1:
        raise ConfigError(
            f"不支持配置 schema_version={version!r}；应为 1"
        )
    for section in ("paper", "assumptions", "experiments"):
        if not isinstance(data.get(section), Mapping):
            raise ConfigError(f"缺少 [{section}] 配置节或其格式无效")
    _validate_supported_assumptions(data)
    return data


def get_experiment(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """返回指定实验配置节，并在配置无效时给出明确错误。"""

    experiments = config.get("experiments")
    if not isinstance(experiments, Mapping):
        raise ConfigError("缺少 [experiments] 配置节或其格式无效")
    experiment = experiments.get(name)
    if not isinstance(experiment, Mapping):
        raise ConfigError(
            f"缺少 [experiments.{name}] 配置节或其格式无效"
        )
    return experiment


__all__ = [
    "ConfigError",
    "DEFAULT_CONFIG_PATH",
    "get_experiment",
    "load_config",
]
