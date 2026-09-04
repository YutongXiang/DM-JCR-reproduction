"""实现论文算法 1的环境分类、升降维训练、扩散训练与推理流程。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any
import warnings

import numpy as np

# 避免 joblib 在部分 Windows 环境调用缺失的 WMIC 来探测物理核心数。
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from sklearn.cluster import KMeans
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from dm_jcr.diffusion import (
    ConditionalUNet,
    DeterministicDiffusion,
    DiffusionModelSpec,
    EnvironmentClassifier,
    StrategyAutoencoder,
    diffusion_spec_from_config,
)
from dm_jcr.diffusion_objective import (
    compact_strategy_slot_mask,
    expand_compact_strategy,
    ObjectiveTensorSpec,
    evaluate_strategy_tensor,
    objective_tensor_spec_from_config,
)


@dataclass(frozen=True)
class DiffusionTrainingSpec:
    """论文未公开但运行训练所必需的优化器参数。"""

    classifier_learning_rate: float
    autoencoder_learning_rate: float
    diffusion_learning_rate: float
    classifier_epochs: int
    autoencoder_epochs: int
    diffusion_epochs: int
    batch_size: int
    gradient_clip_norm: float
    gradient_checkpointing: bool
    kmeans_n_init: int


@dataclass(frozen=True)
class EnvironmentPreprocessor:
    """环境标准化参数和 K-Means 类别中心。"""

    mean: np.ndarray
    scale: np.ndarray
    cluster_centers: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def categories(self, values: np.ndarray) -> np.ndarray:
        normalized = self.transform(values)
        distances = ((normalized[:, None, :] - self.cluster_centers[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1).astype(np.int64)


@dataclass
class DMJCRBundle:
    """训练和推理所需的环境分类器、自动编码器及扩散网络。"""

    model_spec: DiffusionModelSpec
    objective_spec: ObjectiveTensorSpec
    preprocessor: EnvironmentPreprocessor
    classifier: EnvironmentClassifier
    autoencoder: StrategyAutoencoder
    diffusion: DeterministicDiffusion

    def to(self, device: torch.device | str) -> "DMJCRBundle":
        self.classifier.to(device)
        self.autoencoder.to(device)
        self.diffusion.to(device)
        return self


def training_spec_from_config(config: dict[str, Any]) -> DiffusionTrainingSpec:
    assumed = config["assumptions"]["diffusion"]
    return DiffusionTrainingSpec(
        classifier_learning_rate=assumed["classifier_learning_rate"],
        autoencoder_learning_rate=assumed["autoencoder_learning_rate"],
        diffusion_learning_rate=assumed["diffusion_learning_rate"],
        classifier_epochs=assumed["classifier_epochs"],
        autoencoder_epochs=assumed["autoencoder_epochs"],
        diffusion_epochs=assumed["diffusion_epochs"],
        batch_size=assumed["batch_size"],
        gradient_clip_norm=assumed["gradient_clip_norm"],
        gradient_checkpointing=assumed["gradient_checkpointing"],
        kmeans_n_init=assumed["kmeans_n_init"],
    )


def load_diffusion_split(path: str | Path) -> dict[str, np.ndarray]:
    """加载并校验数据生成器保存的结构化 NPZ。"""

    required = {
        "E",
        "node_features",
        "channel_gains",
        "task_features",
        "task_node_indices",
        "task_mask",
    }
    with np.load(path) as source:
        missing = required - set(source.files)
        if missing:
            raise ValueError(f"扩散数据集缺少字段：{sorted(missing)}")
        return {name: source[name].copy() for name in required}


def fit_environment_preprocessor(
    environments: np.ndarray,
    categories: int,
    random_seed: int,
    n_init: int = 10,
) -> tuple[EnvironmentPreprocessor, np.ndarray]:
    """按照论文流程标准化环境数据并用 K-Means 划分类别。"""

    if environments.ndim != 2 or environments.shape[0] < categories:
        raise ValueError("环境样本数必须不少于 K-Means 类别数")
    mean = environments.mean(axis=0)
    scale = environments.std(axis=0)
    scale[scale < 1.0e-12] = 1.0
    normalized = ((environments - mean) / scale).astype(np.float32)
    if n_init <= 0:
        raise ValueError("n_init 必须是正整数")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not find the number of physical cores.*",
            category=UserWarning,
        )
        kmeans = KMeans(
            n_clusters=categories,
            random_state=random_seed,
            n_init=n_init,
        )
        labels = kmeans.fit_predict(normalized).astype(np.int64)
    return (
        EnvironmentPreprocessor(mean, scale, kmeans.cluster_centers_.astype(np.float32)),
        labels,
    )


def _loader(
    data: dict[str, np.ndarray],
    normalized_environment: np.ndarray,
    categories: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(normalized_environment).float(),
        torch.from_numpy(categories).long(),
        torch.from_numpy(data["node_features"]).float(),
        torch.from_numpy(data["channel_gains"]).float(),
        torch.from_numpy(data["task_features"]).float(),
        torch.from_numpy(data["task_node_indices"]).long(),
        torch.from_numpy(data["task_mask"]).bool(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_environment_classifier(
    classifier: EnvironmentClassifier,
    environments: np.ndarray,
    categories: np.ndarray,
    spec: DiffusionTrainingSpec,
    device: torch.device,
) -> list[float]:
    """训练论文的两隐藏层 MLP 环境分类器。"""

    dataset = TensorDataset(
        torch.from_numpy(environments).float(), torch.from_numpy(categories).long()
    )
    loader = DataLoader(dataset, batch_size=spec.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=spec.classifier_learning_rate)
    losses: list[float] = []
    classifier.train()
    for _ in range(spec.classifier_epochs):
        total = 0.0
        count = 0
        for environment, category in loader:
            environment, category = environment.to(device), category.to(device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(classifier(environment), category)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * environment.shape[0]
            count += environment.shape[0]
        losses.append(total / count)
    return losses


def train_strategy_autoencoder(
    autoencoder: StrategyAutoencoder,
    task_features: np.ndarray,
    task_mask: np.ndarray,
    model_spec: DiffusionModelSpec,
    training_spec: DiffusionTrainingSpec,
    device: torch.device,
    random_seed: int,
) -> list[float]:
    """按论文描述使用随机低维资源方案训练对称升降维网络。"""

    features = torch.from_numpy(task_features[:, : model_spec.model_max_tasks]).float()
    masks = torch.from_numpy(task_mask[:, : model_spec.model_max_tasks]).bool()
    loader = DataLoader(TensorDataset(features, masks), batch_size=training_spec.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(
        autoencoder.parameters(), lr=training_spec.autoencoder_learning_rate
    )
    generator = torch.Generator(device=device).manual_seed(random_seed)
    low, high = model_spec.random_strategy_range
    losses: list[float] = []
    autoencoder.train()
    for _ in range(training_spec.autoencoder_epochs):
        total = 0.0
        count = 0
        for batch_features, batch_mask in loader:
            batch_features, batch_mask = batch_features.to(device), batch_mask.to(device)
            slot_mask = compact_strategy_slot_mask(
                batch_mask, model_spec.strategy_fields
            ).flatten(1)
            random_strategy = torch.rand(
                slot_mask.shape,
                generator=generator,
                device=device,
            ) * (high - low) + low
            random_strategy = random_strategy * slot_mask
            optimizer.zero_grad()
            reconstruction = autoencoder(random_strategy)
            loss = ((reconstruction - random_strategy) ** 2 * slot_mask).sum()
            loss = loss / slot_mask.sum().clamp_min(1)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * batch_features.shape[0]
            count += batch_features.shape[0]
        losses.append(total / count)
    return losses


def train_diffusion_model(
    bundle: DMJCRBundle,
    data: dict[str, np.ndarray],
    categories: np.ndarray,
    training_spec: DiffusionTrainingSpec,
    device: torch.device,
    random_seed: int,
    *,
    denoising_steps: int | None = None,
) -> list[float]:
    """按算法 1反向传播公式（24）的 δJ²，而非专家标签噪声损失。"""

    steps = denoising_steps or bundle.model_spec.denoising_steps
    normalized = bundle.preprocessor.transform(data["E"])
    loader = _loader(data, normalized, categories, training_spec.batch_size, True)
    for parameter in bundle.classifier.parameters():
        parameter.requires_grad_(False)
    for parameter in bundle.autoencoder.parameters():
        parameter.requires_grad_(False)
    bundle.autoencoder.eval()
    bundle.diffusion.train()
    optimizer = torch.optim.Adam(
        bundle.diffusion.parameters(), lr=training_spec.diffusion_learning_rate
    )
    generator = torch.Generator(device=device).manual_seed(random_seed)
    low, high = bundle.model_spec.random_strategy_range
    losses: list[float] = []
    for _ in range(training_spec.diffusion_epochs):
        total = 0.0
        count = 0
        for _, category, nodes, gains, tasks, indices, task_mask in loader:
            category = category.to(device)
            nodes, gains = nodes.to(device), gains.to(device)
            tasks, indices, task_mask = tasks.to(device), indices.to(device), task_mask.to(device)
            model_tasks = tasks[:, : bundle.model_spec.model_max_tasks]
            model_mask = task_mask[:, : bundle.model_spec.model_max_tasks]
            compact_slots = compact_strategy_slot_mask(
                model_mask, bundle.model_spec.strategy_fields
            )
            random_strategy = torch.rand(
                (tasks.shape[0], bundle.model_spec.strategy_width),
                generator=generator,
                device=device,
            ) * (high - low) + low
            random_strategy = random_strategy * compact_slots.flatten(1)
            with torch.no_grad():
                clean_high = bundle.autoencoder.expand(random_strategy)
            noise = torch.randn(
                clean_high.shape, generator=generator, device=device, dtype=clean_high.dtype
            )
            timestep = torch.full(
                (clean_high.shape[0],), steps, device=device, dtype=torch.long
            )
            noisy = bundle.diffusion.add_noise(clean_high, noise, timestep)
            optimizer.zero_grad()
            denoised = bundle.diffusion.sample(
                noisy,
                category,
                steps=steps,
                gradient_checkpointing=training_spec.gradient_checkpointing,
            )
            reduced = nn.functional.softplus(bundle.autoencoder.reduce(denoised))
            compact_scores = reduced.reshape(
                -1,
                bundle.model_spec.model_max_tasks,
                bundle.model_spec.strategy_fields,
            ) * compact_slots
            scores = expand_compact_strategy(compact_scores, model_tasks, model_mask)
            objective = evaluate_strategy_tensor(
                scores,
                nodes,
                gains,
                tasks,
                indices,
                task_mask,
                bundle.objective_spec,
            )
            loss = bundle.model_spec.loss_delta * objective.weighted_objective.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("扩散训练产生了非有限的公式（24）损失")
            loss.backward()
            nn.utils.clip_grad_norm_(
                bundle.diffusion.parameters(), training_spec.gradient_clip_norm
            )
            optimizer.step()
            total += float(loss.detach()) * tasks.shape[0]
            count += tasks.shape[0]
        losses.append(total / count)
    return losses


@torch.no_grad()
def generate_resource_strategy(
    bundle: DMJCRBundle,
    environment: np.ndarray,
    task_features: np.ndarray,
    task_mask: np.ndarray,
    device: torch.device,
    *,
    random_seed: int,
    denoising_steps: int | None = None,
) -> np.ndarray:
    """执行论文推理阶段：分类 prompt、确定性去噪、降维并输出非负策略。"""

    steps = denoising_steps or bundle.model_spec.denoising_steps
    normalized = torch.from_numpy(bundle.preprocessor.transform(environment)).float().to(device)
    features = torch.from_numpy(task_features[:, : bundle.model_spec.model_max_tasks]).float().to(device)
    masks = torch.from_numpy(task_mask[:, : bundle.model_spec.model_max_tasks]).bool().to(device)
    if np.any(task_mask[:, bundle.model_spec.model_max_tasks :]):
        raise ValueError("有效任务数超过 assumptions.diffusion.model_max_tasks")
    bundle.classifier.eval()
    bundle.autoencoder.eval()
    bundle.diffusion.eval()
    categories = bundle.classifier(normalized).argmax(dim=1)
    compact_slots = compact_strategy_slot_mask(masks, bundle.model_spec.strategy_fields)
    generator = torch.Generator(device=device).manual_seed(random_seed)
    low, high = bundle.model_spec.random_strategy_range
    random_strategy = torch.rand(
        (environment.shape[0], bundle.model_spec.strategy_width),
        generator=generator,
        device=device,
    ) * (high - low) + low
    random_strategy = random_strategy * compact_slots.flatten(1)
    clean_high = bundle.autoencoder.expand(random_strategy)
    noise = torch.randn(
        clean_high.shape, generator=generator, device=device, dtype=clean_high.dtype
    )
    timestep = torch.full((environment.shape[0],), steps, device=device, dtype=torch.long)
    noisy = bundle.diffusion.add_noise(clean_high, noise, timestep)
    denoised = bundle.diffusion.sample(noisy, categories, steps=steps)
    low_strategy = nn.functional.softplus(bundle.autoencoder.reduce(denoised))
    compact_strategy = low_strategy.reshape(
        -1,
        bundle.model_spec.model_max_tasks,
        bundle.model_spec.strategy_fields,
    ) * compact_slots
    low_strategy = expand_compact_strategy(compact_strategy, features, masks)
    output = np.zeros((environment.shape[0], task_features.shape[1], 8), dtype=np.float32)
    output[:, : bundle.model_spec.model_max_tasks] = low_strategy.cpu().numpy()
    return output


def create_bundle(
    config: dict[str, Any],
    environment_width: int,
    preprocessor: EnvironmentPreprocessor,
) -> DMJCRBundle:
    model_spec = diffusion_spec_from_config(config)
    classifier = EnvironmentClassifier(environment_width, model_spec)
    autoencoder = StrategyAutoencoder(model_spec)
    network = ConditionalUNet(model_spec)
    return DMJCRBundle(
        model_spec,
        objective_tensor_spec_from_config(config),
        preprocessor,
        classifier,
        autoencoder,
        DeterministicDiffusion(network, model_spec),
    )


def save_bundle(
    bundle: DMJCRBundle,
    path: str | Path,
    history: dict[str, list[float]],
) -> None:
    """保存可独立推理的模型、预处理参数和训练历史。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_spec": asdict(bundle.model_spec),
            "objective_spec": asdict(bundle.objective_spec),
            "preprocessor": {
                "mean": bundle.preprocessor.mean,
                "scale": bundle.preprocessor.scale,
                "cluster_centers": bundle.preprocessor.cluster_centers,
            },
            "classifier": bundle.classifier.state_dict(),
            "autoencoder": bundle.autoencoder.state_dict(),
            "diffusion": bundle.diffusion.state_dict(),
            "history": history,
        },
        destination,
    )


def load_bundle(path: str | Path, device: torch.device) -> tuple[DMJCRBundle, dict[str, Any]]:
    """恢复训练检查点。"""

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_spec = DiffusionModelSpec(**checkpoint["model_spec"])
    objective_spec = ObjectiveTensorSpec(**checkpoint["objective_spec"])
    preprocess = checkpoint["preprocessor"]
    preprocessor = EnvironmentPreprocessor(
        preprocess["mean"], preprocess["scale"], preprocess["cluster_centers"]
    )
    classifier = EnvironmentClassifier(preprocessor.mean.shape[0], model_spec)
    autoencoder = StrategyAutoencoder(model_spec)
    diffusion = DeterministicDiffusion(ConditionalUNet(model_spec), model_spec)
    classifier.load_state_dict(checkpoint["classifier"])
    autoencoder.load_state_dict(checkpoint["autoencoder"])
    diffusion.load_state_dict(checkpoint["diffusion"])
    bundle = DMJCRBundle(
        model_spec, objective_spec, preprocessor, classifier, autoencoder, diffusion
    ).to(device)
    return bundle, checkpoint.get("history", {})


__all__ = [
    "DMJCRBundle",
    "DiffusionTrainingSpec",
    "EnvironmentPreprocessor",
    "create_bundle",
    "fit_environment_preprocessor",
    "generate_resource_strategy",
    "load_bundle",
    "load_diffusion_split",
    "save_bundle",
    "train_diffusion_model",
    "train_environment_classifier",
    "train_strategy_autoencoder",
    "training_spec_from_config",
]
