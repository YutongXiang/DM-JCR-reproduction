"""生成论文图3至图7与当前复现图的逐行并排对比图。

在项目根目录运行：

    python -m experiments.experiment_compare_paper_figures
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from dm_jcr.config import DEFAULT_CONFIG_PATH, get_experiment, load_config
from scripts._config_helpers import ChineseArgumentParser


@dataclass(frozen=True)
class ComparisonConfig:
    """论文原图与复现图的路径和排列配置。"""

    figure_numbers: tuple[int, ...]
    paper_figures: tuple[Path, ...]
    reproduction_figures: tuple[Path, ...]
    output_figure: Path

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ComparisonConfig":
        return cls(
            figure_numbers=tuple(int(value) for value in values["figure_numbers"]),
            paper_figures=tuple(Path(value) for value in values["paper_figures"]),
            reproduction_figures=tuple(Path(value) for value in values["reproduction_figures"]),
            output_figure=Path(values["output_figure"]),
        )

    def validate(self) -> None:
        """检查图号数量、输入文件和输出扩展名。"""

        count = len(self.figure_numbers)
        if count == 0:
            raise ValueError("figure_numbers不能为空")
        if len(self.paper_figures) != count or len(self.reproduction_figures) != count:
            raise ValueError("图号、论文原图和复现图的数量必须一致")
        missing = [
            path
            for path in (*self.paper_figures, *self.reproduction_figures)
            if not path.is_file()
        ]
        if missing:
            names = "、".join(str(path) for path in missing)
            raise FileNotFoundError(f"以下对比图输入不存在：{names}")
        if self.output_figure.suffix.lower() != ".png":
            raise ValueError("output_figure必须是PNG文件")


def generate_comparison(config: ComparisonConfig) -> None:
    """将每张论文原图和对应复现图排列在同一行。"""

    config.validate()
    rows = len(config.figure_numbers)
    figure, axes = plt.subplots(rows, 2, figsize=(13.6, 4.9 * rows), squeeze=False)
    for row, (number, paper_path, reproduced_path) in enumerate(
        zip(
            config.figure_numbers,
            config.paper_figures,
            config.reproduction_figures,
            strict=True,
        )
    ):
        pairs = (
            (paper_path, f"Figure {number} - Paper"),
            (reproduced_path, f"Figure {number} - Reproduction"),
        )
        for column, (path, title) in enumerate(pairs):
            axis = axes[row, column]
            axis.imshow(mpimg.imread(path))
            axis.set_title(title, fontsize=14, pad=10)
            axis.set_axis_off()

    figure.subplots_adjust(left=0.01, right=0.99, top=0.985, bottom=0.01, wspace=0.035, hspace=0.12)
    config.output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(config.output_figure, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = ChineseArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="TOML配置文件路径")
    args = parser.parse_args()
    root = load_config(args.config)
    config = ComparisonConfig.from_mapping(
        get_experiment(root, "paper_reproduction_comparison")
    )
    generate_comparison(config)
    print(f"论文原图与复现图的对比图已保存到：{config.output_figure.resolve()}")


if __name__ == "__main__":
    main()
