"""复现论文图7中DM-JCR归一化平均加权指标随任务数量变化的曲线。

在项目根目录运行：python -m experiments.experiment_figure7_weighted_indicator
"""

from pathlib import Path

from dm_jcr.config import DEFAULT_CONFIG_PATH, get_experiment, load_config
from experiments._figure56_dm_jcr_sweep import TaskSweepConfig, run_sweep, save_metric_results
from scripts._config_helpers import ChineseArgumentParser


def main() -> None:
    parser = ChineseArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="TOML配置文件路径")
    args = parser.parse_args()
    root = load_config(args.config)
    config = TaskSweepConfig.from_mapping(get_experiment(root, "figure7_weighted_indicator"))
    summary = save_metric_results(config, run_sweep(root, config), "weighted", 7)
    print(f"图7完成：{len(summary['points'])}个任务规模点")
    print(f"原始数据：{config.output_csv.resolve()}")
    print(f"汇总信息：{config.output_summary.resolve()}")
    print(f"图像：{config.output_figure.resolve()}")


if __name__ == "__main__":
    main()
