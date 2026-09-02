"""按照论文算法 1训练环境分类器、升降维网络和 DM-JCR 扩散网络。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from dm_jcr.config import DEFAULT_CONFIG_PATH, load_config
from dm_jcr.diffusion import diffusion_spec_from_config
from dm_jcr.diffusion_training import (
    create_bundle,
    fit_environment_preprocessor,
    load_diffusion_split,
    save_bundle,
    train_diffusion_model,
    train_environment_classifier,
    train_strategy_autoencoder,
    training_spec_from_config,
)
from scripts._config_helpers import ChineseArgumentParser


def _positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return result


def main() -> None:
    parser = ChineseArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="TOML 配置文件路径")
    parser.add_argument("--dataset", type=Path, default=Path("outputs/diffusion_dataset/train.npz"), help="训练数据集路径")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/dm_jcr.pt"), help="模型检查点输出路径")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="训练设备")
    parser.add_argument("--classifier-epochs", type=_positive, help="环境分类器训练轮数")
    parser.add_argument("--autoencoder-epochs", type=_positive, help="升降维网络训练轮数")
    parser.add_argument("--diffusion-epochs", type=_positive, help="扩散网络训练轮数")
    parser.add_argument("--batch-size", type=_positive, help="批大小")
    parser.add_argument("--denoising-steps", type=_positive, help="训练采用的反向去噪步数")
    parser.add_argument("--limit-samples", type=_positive, help="限制读取的训练样本数")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    args = parser.parse_args()

    config = load_config(args.config)
    model_spec = diffusion_spec_from_config(config)
    training_spec = training_spec_from_config(config)
    training_spec = replace(
        training_spec,
        classifier_epochs=args.classifier_epochs or training_spec.classifier_epochs,
        autoencoder_epochs=args.autoencoder_epochs or training_spec.autoencoder_epochs,
        diffusion_epochs=args.diffusion_epochs or training_spec.diffusion_epochs,
        batch_size=args.batch_size or training_spec.batch_size,
    )
    steps = args.denoising_steps or model_spec.denoising_steps
    if steps > model_spec.denoising_steps:
        parser.error("--denoising-steps 不能超过论文配置的总扩散步数")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)

    data = load_diffusion_split(args.dataset)
    if args.limit_samples is not None:
        data = {name: values[: args.limit_samples] for name, values in data.items()}
    preprocessor, categories = fit_environment_preprocessor(
        data["E"],
        model_spec.environment_categories,
        args.seed,
        training_spec.kmeans_n_init,
    )
    normalized_environment = preprocessor.transform(data["E"])
    bundle = create_bundle(config, data["E"].shape[1], preprocessor).to(device)

    print(f"设备：{device}；样本数：{len(categories)}；反向去噪步数：{steps}")
    classifier_history = train_environment_classifier(
        bundle.classifier,
        normalized_environment,
        categories,
        training_spec,
        device,
    )
    print(f"环境分类器最终损失：{classifier_history[-1]:.6f}")
    autoencoder_history = train_strategy_autoencoder(
        bundle.autoencoder,
        data["task_features"],
        data["task_mask"],
        bundle.model_spec,
        training_spec,
        device,
        args.seed,
    )
    print(f"升降维网络最终重建损失：{autoencoder_history[-1]:.6f}")
    diffusion_history = train_diffusion_model(
        bundle,
        data,
        categories,
        training_spec,
        device,
        args.seed,
        denoising_steps=steps,
    )
    print(f"公式（24）最终损失：{diffusion_history[-1]:.6f}")
    save_bundle(
        bundle,
        args.checkpoint,
        {
            "classifier": classifier_history,
            "autoencoder": autoencoder_history,
            "diffusion": diffusion_history,
            "training_denoising_steps": [float(steps)],
        },
    )
    print(f"检查点已保存：{args.checkpoint.resolve()}")


if __name__ == "__main__":
    main()
