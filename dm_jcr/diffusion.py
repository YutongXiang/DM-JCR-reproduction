"""按照论文公式（1）—（3）、（19）实现 DM-JCR 条件扩散网络。"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class DiffusionModelSpec:
    """论文明确结构与未公开复现参数组成的扩散模型规格。"""

    denoising_steps: int
    image_size: int
    image_channels: int
    feature_channels: tuple[int, int, int]
    beta_start: float
    beta_end: float
    environment_categories: int
    classifier_hidden_sizes: tuple[int, int]
    classifier_dropout: float
    autoencoder_hidden_sizes: tuple[int, int]
    model_max_tasks: int
    strategy_fields: int
    time_embedding_size: int
    loss_delta: float
    random_strategy_range: tuple[float, float]

    def __post_init__(self) -> None:
        if self.denoising_steps <= 0 or self.image_size <= 0 or self.image_channels != 1:
            raise ValueError("扩散步数和图像尺寸必须为正，论文 U-Net 输入通道必须为 1")
        if len(self.feature_channels) != 3 or any(value <= 0 for value in self.feature_channels):
            raise ValueError("feature_channels 必须包含三个正整数")
        if not 0.0 < self.beta_start < self.beta_end < 1.0:
            raise ValueError("beta 必须满足 0 < beta_start < beta_end < 1")
        if self.environment_categories <= 1:
            raise ValueError("environment_categories 必须大于 1")
        if not 0.0 <= self.classifier_dropout < 1.0:
            raise ValueError("classifier_dropout 必须位于 [0, 1)")
        if self.strategy_width > self.high_dimension:
            raise ValueError("论文要求先升维，因此低维策略宽度不能超过 28×28")
        if self.time_embedding_size <= 0 or self.time_embedding_size % 2 != 0:
            raise ValueError("time_embedding_size 必须是正偶数")
        if self.loss_delta <= 0.0:
            raise ValueError("loss_delta 必须为正")
        low, high = self.random_strategy_range
        if low <= 0.0 or high < low:
            raise ValueError("random_strategy_range 必须是正数递增范围")

    @property
    def strategy_width(self) -> int:
        return self.model_max_tasks * self.strategy_fields

    @property
    def high_dimension(self) -> int:
        return self.image_size * self.image_size * self.image_channels


def diffusion_spec_from_config(config: dict[str, Any]) -> DiffusionModelSpec:
    """读取论文结构和明确标注的复现假设。"""

    paper = config["paper"]["diffusion"]
    assumed = config["assumptions"]["diffusion"]
    required_paper_values = {
        "image_size": 28,
        "image_channels": 1,
        "convolution_kernel_size": 3,
        "pooling_kernel_size": 2,
        "upsampling_kernel_size": 2,
        "optimizer": "adam",
    }
    for name, expected in required_paper_values.items():
        if paper[name] != expected:
            raise ValueError(
                f"paper.diffusion.{name}={paper[name]!r} 与论文结构 {expected!r} 不一致"
            )
    if paper["deterministic_reverse_sampling"] is not True:
        raise ValueError("当前实现严格采用论文公式（19）的确定性反向采样")
    if assumed["beta_schedule"] != "linear":
        raise ValueError("当前复现仅实现配置中声明的 linear beta 调度")
    if assumed["positive_output_transform"] != "softplus":
        raise ValueError("当前复现要求使用 softplus 生成非负资源分数")
    if assumed["environment_preprocessing"] != "per_feature_standardization":
        raise ValueError("当前复现要求对环境条件执行逐特征标准化")
    if assumed["condition_injection"] != "time_and_class_embedding_added_per_unet_block":
        raise ValueError("当前复现仅支持配置中声明的时间步与类别联合条件注入")
    return DiffusionModelSpec(
        denoising_steps=paper["denoising_steps"],
        image_size=paper["image_size"],
        image_channels=paper["image_channels"],
        feature_channels=tuple(paper["unet_feature_channels"]),
        beta_start=assumed["beta_start"],
        beta_end=assumed["beta_end"],
        environment_categories=assumed["environment_categories"],
        classifier_hidden_sizes=tuple(assumed["classifier_hidden_sizes"]),
        classifier_dropout=assumed["classifier_dropout"],
        autoencoder_hidden_sizes=tuple(assumed["autoencoder_hidden_sizes"]),
        model_max_tasks=assumed["model_max_tasks"],
        strategy_fields=8,
        time_embedding_size=assumed["time_embedding_size"],
        loss_delta=assumed["loss_delta"],
        random_strategy_range=tuple(assumed["random_strategy_range"]),
    )


class SinusoidalTimeEmbedding(nn.Module):
    """为未指定具体实现的时间步条件提供标准正弦嵌入。"""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, timesteps: Tensor) -> Tensor:
        half = self.dimension // 2
        scale = -log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=torch.float32) * scale
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class ConditionedDoubleConv(nn.Module):
    """图 2的两次 3×3 卷积，并注入时间步和环境类别 prompt。"""

    def __init__(self, input_channels: int, output_channels: int, condition_size: int) -> None:
        super().__init__()
        self.first = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.second = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.condition = nn.Linear(condition_size, output_channels)

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        value = F.relu(self.first(value))
        value = value + self.condition(condition).unsqueeze(-1).unsqueeze(-1)
        return F.relu(self.second(value))


class ConditionalUNet(nn.Module):
    """论文图 2所示 28×28、两下采样和两上采样 U-Net。"""

    def __init__(self, spec: DiffusionModelSpec) -> None:
        super().__init__()
        c1, c2, c3 = spec.feature_channels
        embedding = spec.time_embedding_size
        self.time_embedding = SinusoidalTimeEmbedding(embedding)
        self.time_projection = nn.Sequential(
            nn.Linear(embedding, embedding),
            nn.ReLU(),
            nn.Linear(embedding, embedding),
        )
        self.class_embedding = nn.Embedding(spec.environment_categories, embedding)
        self.down1 = ConditionedDoubleConv(spec.image_channels, c1, embedding)
        self.down2 = ConditionedDoubleConv(c1, c2, embedding)
        self.bottom = ConditionedDoubleConv(c2, c3, embedding)
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.decode1 = ConditionedDoubleConv(c2 + c2, c2, embedding)
        self.up2 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.decode2 = ConditionedDoubleConv(c1 + c1, c1, embedding)
        self.output = nn.Conv2d(c1, spec.image_channels, 1)

    def forward(self, value: Tensor, timesteps: Tensor, categories: Tensor) -> Tensor:
        condition = self.time_projection(self.time_embedding(timesteps))
        condition = condition + self.class_embedding(categories)
        skip1 = self.down1(value, condition)
        skip2 = self.down2(self.pool(skip1), condition)
        value = self.bottom(self.pool(skip2), condition)
        value = self.up1(value)
        value = self.decode1(torch.cat((value, skip2), dim=1), condition)
        value = self.up2(value)
        value = self.decode2(torch.cat((value, skip1), dim=1), condition)
        return self.output(value)


class StrategyAutoencoder(nn.Module):
    """论文所述对称全连接编码器—解码器。"""

    def __init__(self, spec: DiffusionModelSpec) -> None:
        super().__init__()
        h1, h2 = spec.autoencoder_hidden_sizes
        self.strategy_width = spec.strategy_width
        self.image_size = spec.image_size
        self.decoder = nn.Sequential(
            nn.Linear(spec.strategy_width, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, spec.high_dimension),
        )
        self.encoder = nn.Sequential(
            nn.Linear(spec.high_dimension, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, spec.strategy_width),
        )

    def expand(self, strategy: Tensor) -> Tensor:
        return self.decoder(strategy).reshape(-1, 1, self.image_size, self.image_size)

    def reduce(self, image: Tensor) -> Tensor:
        return self.encoder(image.flatten(1))

    def forward(self, strategy: Tensor) -> Tensor:
        return self.reduce(self.expand(strategy))


class EnvironmentClassifier(nn.Module):
    """论文所述两隐藏层、Dropout、Softmax 环境分类器。"""

    def __init__(self, input_size: int, spec: DiffusionModelSpec) -> None:
        super().__init__()
        h1, h2 = spec.classifier_hidden_sizes
        self.network = nn.Sequential(
            nn.Linear(input_size, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(spec.classifier_dropout),
            nn.Linear(h2, spec.environment_categories),
        )

    def forward(self, environment: Tensor) -> Tensor:
        return self.network(environment)

    def probabilities(self, environment: Tensor) -> Tensor:
        return F.softmax(self(environment), dim=-1)


class DeterministicDiffusion(nn.Module):
    """公式（2）的直接加噪和公式（19）的无随机项反向采样。"""

    def __init__(self, network: ConditionalUNet, spec: DiffusionModelSpec) -> None:
        super().__init__()
        self.network = network
        self.spec = spec
        beta = torch.linspace(spec.beta_start, spec.beta_end, spec.denoising_steps)
        beta = torch.cat((torch.zeros(1), beta))
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def add_noise(self, clean: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """实现论文公式（2）。"""

        alpha_bar = self.alpha_bar[timesteps].reshape(-1, 1, 1, 1)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    def reverse_step(
        self,
        value: Tensor,
        timesteps: Tensor,
        categories: Tensor,
        *,
        gradient_checkpointing: bool = False,
    ) -> Tensor:
        """实现论文公式（19），明确不加入 σ_a z。"""

        predicted_noise = (
            checkpoint(
                self.network,
                value,
                timesteps,
                categories,
                use_reentrant=False,
            )
            if gradient_checkpointing and torch.is_grad_enabled()
            else self.network(value, timesteps, categories)
        )
        alpha = self.alpha[timesteps].reshape(-1, 1, 1, 1)
        beta = self.beta[timesteps].reshape(-1, 1, 1, 1)
        alpha_bar = self.alpha_bar[timesteps].reshape(-1, 1, 1, 1)
        return (value - beta / (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()

    def sample(
        self,
        initial: Tensor,
        categories: Tensor,
        *,
        steps: int | None = None,
        gradient_checkpointing: bool = False,
    ) -> Tensor:
        """从 x_A 到 x_0 执行论文算法 1的确定性逐步去噪。"""

        final_step = self.spec.denoising_steps if steps is None else steps
        if not 1 <= final_step <= self.spec.denoising_steps:
            raise ValueError("steps 必须位于 [1, denoising_steps]")
        value = initial
        for step in range(final_step, 0, -1):
            timesteps = torch.full(
                (value.shape[0],), step, device=value.device, dtype=torch.long
            )
            value = self.reverse_step(
                value,
                timesteps,
                categories,
                gradient_checkpointing=gradient_checkpointing,
            )
        return value


__all__ = [
    "ConditionalUNet",
    "DeterministicDiffusion",
    "DiffusionModelSpec",
    "EnvironmentClassifier",
    "StrategyAutoencoder",
    "diffusion_spec_from_config",
]
