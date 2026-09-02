"""加载 DM-JCR 检查点，为数据集环境生成非负资源分配策略。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dm_jcr.diffusion_training import (
    generate_resource_strategy,
    load_bundle,
    load_diffusion_split,
)
from scripts._config_helpers import ChineseArgumentParser


def main() -> None:
    parser = ChineseArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/dm_jcr.pt"), help="模型检查点路径")
    parser.add_argument("--dataset", type=Path, default=Path("outputs/diffusion_dataset/test.npz"), help="输入数据集路径")
    parser.add_argument("--output", type=Path, default=Path("outputs/generated_strategy.npy"), help="生成策略的输出路径")
    parser.add_argument("--samples", type=int, default=1, help="要生成的策略数量")
    parser.add_argument("--denoising-steps", type=int, help="反向去噪步数")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="运行设备")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples 必须是正整数")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    bundle, _ = load_bundle(args.checkpoint, device)
    data = load_diffusion_split(args.dataset)
    count = min(args.samples, data["E"].shape[0])
    strategies = generate_resource_strategy(
        bundle,
        data["E"][:count],
        data["task_features"][:count],
        data["task_mask"][:count],
        device,
        random_seed=args.seed,
        denoising_steps=args.denoising_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, strategies)
    print(f"已生成 {count} 个策略，形状={strategies.shape}")
    print(f"输出文件：{args.output.resolve()}")


if __name__ == "__main__":
    main()
