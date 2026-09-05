"""检查生成的数据集是否合法，以及是否达到正式训练门槛。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from dm_jcr.config import DEFAULT_CONFIG_PATH, get_experiment, load_config
from dm_jcr.dataset_validation import validate_diffusion_dataset
from scripts._config_helpers import ChineseArgumentParser


def main(argv: Sequence[str] | None = None) -> None:
    parser = ChineseArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="TOML配置文件路径")
    parser.add_argument("--dataset", type=Path, help="数据集目录")
    parser.add_argument("--profile", choices=("structural", "formal"), help="结构检查或正式训练检查")
    parser.add_argument("--report", type=Path, help="JSON检查报告路径")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    settings = get_experiment(config, "dataset_validation")
    directory = args.dataset or Path(settings["dataset_directory"])
    profile = args.profile or settings["profile"]
    report_path = args.report or Path(settings["report"])
    result = validate_diffusion_dataset(config, directory, profile=profile)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"数据集：{result['dataset_directory']}")
    for split, summary in result["summaries"].items():
        print(
            f"{split}: 样本={summary['samples']}，"
            f"任务={summary['task_count_min']}–{summary['task_count_max']}，"
            f"总体可行={summary['feasible_ratio']:.3f}，"
            f"单任务按时={summary['mean_task_deadline_satisfied_ratio']:.3f}"
        )
    for issue in result["issues"]:
        print(f"[错误] {issue['split']}: {issue['message']}")
    print(f"检查报告：{report_path.resolve()}")
    if result["formal_training_ready"]:
        print("结论：数据集合法，可以用于正式训练。")
    elif result["valid"]:
        print("结论：数据集结构合法，但本次未执行正式训练门槛检查。")
    else:
        print("结论：数据集不满足当前检查要求，不能用于正式训练。")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
