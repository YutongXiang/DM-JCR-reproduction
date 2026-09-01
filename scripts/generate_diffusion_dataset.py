"""生成条件扩散模型使用的环境 E 与高质量策略 x0 数据集。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dm_jcr.config import DEFAULT_CONFIG_PATH, get_experiment, load_config
from dm_jcr.data_generation import generate_and_save_dataset


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-samples", type=_positive_int)
    parser.add_argument("--validation-samples", type=_positive_int)
    parser.add_argument("--test-samples", type=_positive_int)
    parser.add_argument("--candidates", type=_positive_int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    experiment = get_experiment(config, "dataset_generation")
    output = args.output or Path(experiment["output_directory"])
    split_sizes = {
        "train": args.train_samples or experiment["train_samples"],
        "validation": args.validation_samples or experiment["validation_samples"],
        "test": args.test_samples or experiment["test_samples"],
    }
    seed = args.seed if args.seed is not None else experiment["random_seed"]
    manifest = generate_and_save_dataset(
        config,
        output,
        split_sizes,
        random_seed=seed,
        candidate_count=args.candidates,
    )

    print(f"数据集已保存到：{output.resolve()}")
    print(
        f"E 宽度={manifest['environment_width']}，"
        f"x0 形状={tuple(manifest['strategy_shape'])}"
    )
    for name, information in manifest["splits"].items():
        print(
            f"{name}: samples={information['samples']} "
            f"mean_J={information['mean_objective']:.6f} "
            f"baseline_J={information['mean_baseline_objective']:.6f} "
            f"feasible={information['feasible_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
