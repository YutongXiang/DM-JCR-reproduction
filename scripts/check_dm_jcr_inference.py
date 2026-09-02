"""端到端检查 DM-JCR 推理、策略解码、公式（27）投影及公式（17）评价。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dm_jcr.config import DEFAULT_CONFIG_PATH, load_config
from dm_jcr.data_generation import generate_scenario, scenario_spec_from_config
from dm_jcr.diffusion_training import generate_resource_strategy, load_bundle
from dm_jcr.environment import encode_environment, tensor_spec_from_config
from dm_jcr.resource_allocation import (
    ObjectiveNormalization,
    ObjectiveWeights,
    evaluate_equation17_strategy,
    project_equation17_strategy,
)
from dm_jcr.strategy_codec import EncodedRawStrategy, decode_raw_strategy
from scripts._config_helpers import ChineseArgumentParser


def main() -> None:
    parser = ChineseArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="TOML 配置文件路径")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/dm_jcr.pt"), help="模型检查点路径")
    parser.add_argument("--denoising-steps", type=int, help="反向去噪步数")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="运行设备")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    config = load_config(args.config)
    bundle, _ = load_bundle(args.checkpoint, device)
    scenario = generate_scenario(
        np.random.default_rng(args.seed),
        0,
        scenario_spec_from_config(config),
    )
    environment = encode_environment(scenario.snapshot, tensor_spec_from_config(config))
    scores = generate_resource_strategy(
        bundle,
        environment.flat_vector()[None, :],
        environment.task_features[None, ...],
        environment.task_mask[None, ...],
        device,
        random_seed=args.seed,
        denoising_steps=args.denoising_steps,
    )[0]
    encoded = EncodedRawStrategy(scores, environment.task_mask, scenario.snapshot.mappings)
    direct, relay, v2v = decode_raw_strategy(encoded)
    projected = project_equation17_strategy(direct, relay, v2v, scenario.capacities)
    result = evaluate_equation17_strategy(
        scenario.direct_contexts,
        scenario.relay_contexts,
        scenario.v2v_contexts,
        projected,
        scenario.capacities,
        ObjectiveNormalization(config["assumptions"]["objective"]["energy_reference_j"]),
        ObjectiveWeights(
            config["paper"]["objective"]["latency_weight"],
            config["paper"]["objective"]["energy_weight"],
        ),
        energy_coefficient=config["paper"]["task"]["cpu_energy_coefficient"],
    )
    print(f"任务数={result.task_count}")
    print(f"平均归一化时延={result.mean_normalized_latency:.6f}")
    print(f"平均归一化能耗={result.mean_normalized_energy:.6f}")
    print(f"J={result.weighted_objective:.6f}")
    print(f"是否可行={result.feasible}")


if __name__ == "__main__":
    main()
