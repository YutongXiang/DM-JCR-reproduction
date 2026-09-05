"""按照论文算法1训练环境分类器、升降维网络和DM-JCR扩散网络。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Callable

import torch

from dm_jcr.config import DEFAULT_CONFIG_PATH, load_config
from dm_jcr.diffusion import diffusion_spec_from_config
from dm_jcr.diffusion_training import (
    DMJCRBundle,
    create_bundle,
    fit_environment_preprocessor,
    load_diffusion_split,
    load_training_bundle,
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


def _configure_accelerator(device: torch.device, allow_tf32: bool) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def _restore_rng(training_state: dict[str, Any] | None, device: torch.device) -> None:
    if not training_state:
        return
    if training_state.get("torch_rng_state") is not None:
        torch.set_rng_state(training_state["torch_rng_state"].cpu())
    cuda_states = training_state.get("cuda_rng_state_all")
    if device.type == "cuda" and cuda_states is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


def _training_state(
    *, phase: str, completed: dict[str, int], targets: dict[str, int],
    runtime: dict[str, Any] | None, steps: int, training_spec: Any,
    dataset: Path, samples: int, seed: int, device: torch.device,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": phase,
        "completed_epochs": dict(completed),
        "target_epochs": dict(targets),
        "runtime": runtime,
        "denoising_steps": steps,
        "batch_size": training_spec.batch_size,
        "amp_dtype": training_spec.amp_dtype,
        "dataset": str(dataset.resolve()),
        "dataset_samples": samples,
        "seed": seed,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = ChineseArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="TOML配置文件路径")
    parser.add_argument("--dataset", type=Path, default=Path("outputs/diffusion_dataset/train.npz"), help="训练数据集路径")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/dm_jcr.pt"), help="模型检查点路径")
    parser.add_argument("--resume", action="store_true", help="从--checkpoint指定的检查点断点续训")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="训练设备")
    parser.add_argument("--classifier-epochs", type=_positive, help="环境分类器总训练轮数")
    parser.add_argument("--autoencoder-epochs", type=_positive, help="升降维网络总训练轮数")
    parser.add_argument("--diffusion-epochs", type=_positive, help="扩散网络总训练轮数")
    parser.add_argument("--batch-size", type=_positive, help="批量大小；RTX 5090建议从64开始")
    parser.add_argument("--denoising-steps", type=_positive, help="训练采用的反向去噪步数")
    parser.add_argument("--checkpoint-every", type=_positive, help="每隔多少轮保存一次检查点")
    parser.add_argument("--workers", type=int, help="DataLoader工作进程数；内存数据通常使用0")
    parser.add_argument("--amp", choices=("off", "bf16", "fp16"), help="自动混合精度类型")
    parser.add_argument("--no-tf32", action="store_true", help="禁用TF32")
    parser.add_argument("--no-fused-optimizer", action="store_true", help="禁用fused Adam")
    parser.add_argument("--no-channels-last", action="store_true", help="禁用channels-last内存格式")
    parser.add_argument("--limit-samples", type=_positive, help="限制读取的训练样本数，仅用于试跑")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    args = parser.parse_args()

    config = load_config(args.config)
    model_spec = diffusion_spec_from_config(config)
    training_spec = training_spec_from_config(config)
    if args.workers is not None and args.workers < 0:
        parser.error("--workers不能小于0")
    training_spec = replace(
        training_spec,
        classifier_epochs=args.classifier_epochs or training_spec.classifier_epochs,
        autoencoder_epochs=args.autoencoder_epochs or training_spec.autoencoder_epochs,
        diffusion_epochs=args.diffusion_epochs or training_spec.diffusion_epochs,
        batch_size=args.batch_size or training_spec.batch_size,
        checkpoint_every_epochs=args.checkpoint_every or training_spec.checkpoint_every_epochs,
        dataloader_workers=training_spec.dataloader_workers if args.workers is None else args.workers,
        amp_dtype=args.amp or training_spec.amp_dtype,
        allow_tf32=training_spec.allow_tf32 and not args.no_tf32,
        fused_optimizer=training_spec.fused_optimizer and not args.no_fused_optimizer,
        channels_last=training_spec.channels_last and not args.no_channels_last,
    )
    steps = args.denoising_steps or model_spec.denoising_steps
    if steps > model_spec.denoising_steps:
        parser.error("--denoising-steps不能超过模型配置的总扩散步数")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("当前PyTorch无法使用CUDA")
    if device.type == "cuda" and training_spec.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("当前GPU或PyTorch不支持BF16，请改用--amp fp16")
    _configure_accelerator(device, training_spec.allow_tf32)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    data = load_diffusion_split(args.dataset)
    if args.limit_samples is not None:
        data = {name: values[: args.limit_samples] for name, values in data.items()}
    samples = data["E"].shape[0]
    targets = {
        "classifier": training_spec.classifier_epochs,
        "autoencoder": training_spec.autoencoder_epochs,
        "diffusion": training_spec.diffusion_epochs,
    }
    completed = {name: 0 for name in targets}
    histories: dict[str, list[float]] = {
        "classifier": [], "autoencoder": [], "diffusion": [],
        "training_denoising_steps": [float(steps)],
    }
    saved_state: dict[str, Any] | None = None

    if args.resume:
        if not args.checkpoint.exists():
            parser.error(f"续训检查点不存在：{args.checkpoint}")
        bundle, loaded_history, saved_state = load_training_bundle(args.checkpoint, device)
        if saved_state is None:
            parser.error("该检查点只含推理权重，不含断点续训状态")
        if asdict(bundle.model_spec) != asdict(model_spec):
            parser.error("检查点模型结构与当前配置不一致")
        if saved_state.get("denoising_steps") != steps:
            parser.error("检查点去噪步数与当前--denoising-steps不一致")
        if saved_state.get("batch_size") != training_spec.batch_size:
            parser.error("精确续训必须保持与检查点相同的batch-size")
        if saved_state.get("seed") != args.seed:
            parser.error("精确续训必须保持与检查点相同的seed")
        if saved_state.get("amp_dtype") != training_spec.amp_dtype:
            parser.error("精确续训必须保持与检查点相同的AMP类型")
        if saved_state.get("dataset_samples") != samples:
            parser.error("检查点使用的样本数与当前数据集不一致")
        completed.update(saved_state.get("completed_epochs", {}))
        for name in histories:
            if name in loaded_history:
                histories[name] = list(loaded_history[name])
        categories = bundle.preprocessor.categories(data["E"])
        _restore_rng(saved_state, device)
        print(f"已恢复检查点：阶段={saved_state['phase']}，进度={completed}")
    else:
        preprocessor, categories = fit_environment_preprocessor(
            data["E"], model_spec.environment_categories, args.seed, training_spec.kmeans_n_init
        )
        bundle = create_bundle(config, data["E"].shape[1], preprocessor).to(device)

    normalized_environment = bundle.preprocessor.transform(data["E"])
    checkpoint_interval = training_spec.checkpoint_every_epochs
    run_started = time.monotonic()

    def save_state(phase: str, phase_runtime: dict[str, Any] | None) -> None:
        save_bundle(
            bundle, args.checkpoint, histories,
            _training_state(
                phase=phase, completed=completed, targets=targets, runtime=phase_runtime,
                steps=steps, training_spec=training_spec, dataset=args.dataset,
                samples=samples, seed=args.seed, device=device,
            ),
        )

    def callback(phase: str, target: int) -> Callable[[int, list[float], dict[str, Any]], None]:
        epoch_started = time.monotonic()

        def on_epoch(epoch: int, losses: list[float], phase_runtime: dict[str, Any]) -> None:
            nonlocal epoch_started
            duration = time.monotonic() - epoch_started
            epoch_started = time.monotonic()
            histories[phase] = list(losses)
            completed[phase] = epoch
            remaining = max(0, target - epoch) * duration
            print(f"{phase}：{epoch}/{target}，损失={losses[-1]:.6f}，本轮={duration / 60:.1f}分钟，阶段剩余约={remaining / 3600:.1f}小时", flush=True)
            if epoch % checkpoint_interval == 0 or epoch == target:
                save_state(phase, phase_runtime)
                print(f"检查点已保存：{args.checkpoint.resolve()}", flush=True)

        return on_epoch

    print(
        f"设备={device}；样本={samples}；去噪步数={steps}；batch={training_spec.batch_size}；"
        f"AMP={training_spec.amp_dtype}；TF32={training_spec.allow_tf32}；fused Adam={training_spec.fused_optimizer}"
    )
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(0)}；PyTorch={torch.__version__}；CUDA运行时={torch.version.cuda}")
    if not args.resume:
        save_state("classifier", None)

    resume_phase = saved_state.get("phase") if saved_state else "classifier"
    resume_runtime = saved_state.get("runtime") if saved_state else None
    try:
        if completed["classifier"] < targets["classifier"]:
            histories["classifier"] = train_environment_classifier(
                bundle.classifier, normalized_environment, categories, training_spec, device,
                start_epoch=completed["classifier"], history=histories["classifier"],
                runtime_state=resume_runtime if resume_phase == "classifier" else None,
                epoch_callback=callback("classifier", targets["classifier"]),
            )
        if resume_phase not in ("autoencoder", "diffusion"):
            save_state("autoencoder", None)
        if completed["autoencoder"] < targets["autoencoder"]:
            histories["autoencoder"] = train_strategy_autoencoder(
                bundle.autoencoder, data["task_features"], data["task_mask"],
                bundle.model_spec, training_spec, device, args.seed,
                start_epoch=completed["autoencoder"], history=histories["autoencoder"],
                runtime_state=resume_runtime if resume_phase == "autoencoder" else None,
                epoch_callback=callback("autoencoder", targets["autoencoder"]),
            )
        if resume_phase != "diffusion":
            save_state("diffusion", None)
        if completed["diffusion"] < targets["diffusion"]:
            histories["diffusion"] = train_diffusion_model(
                bundle, data, categories, training_spec, device, args.seed,
                denoising_steps=steps, start_epoch=completed["diffusion"],
                history=histories["diffusion"],
                runtime_state=resume_runtime if resume_phase == "diffusion" else None,
                epoch_callback=callback("diffusion", targets["diffusion"]),
            )
        save_state("complete", None)
    except KeyboardInterrupt:
        print("训练已中断；最近一个完整轮次的原子检查点仍然有效，可加--resume继续。")
        raise SystemExit(130)

    print(f"训练完成，总耗时={(time.monotonic() - run_started) / 3600:.2f}小时")
    print(f"最终检查点：{args.checkpoint.resolve()}")


if __name__ == "__main__":
    main()
