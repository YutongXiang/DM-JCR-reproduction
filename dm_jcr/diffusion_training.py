"""实现论文算法 1的环境分类、升降维训练、扩散训练与推理流程。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import nullcontext
import os
from pathlib import Path
from typing import Any, Callable
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
    amp_dtype: str = "off"
    allow_tf32: bool = True
    fused_optimizer: bool = True
    channels_last: bool = True
    pin_memory: bool = True
    dataloader_workers: int = 0
    persistent_workers: bool = False
    checkpoint_every_epochs: int = 1


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


@dataclass(frozen=True)
class StrategyGenerationTrace:
    """反向扩散每一步的原始指标及只接受更优方案后的当前最优指标。"""

    generation_steps: np.ndarray
    diffusion_timesteps: np.ndarray
    categories: np.ndarray
    raw_weighted_objective: np.ndarray
    raw_normalized_latency: np.ndarray
    raw_normalized_energy: np.ndarray
    raw_deadline_satisfied_ratio: np.ndarray
    best_weighted_objective: np.ndarray
    best_normalized_latency: np.ndarray
    best_normalized_energy: np.ndarray
    best_deadline_satisfied_ratio: np.ndarray


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
        amp_dtype=assumed.get("amp_dtype", "off"),
        allow_tf32=assumed.get("allow_tf32", True),
        fused_optimizer=assumed.get("fused_optimizer", True),
        channels_last=assumed.get("channels_last", True),
        pin_memory=assumed.get("pin_memory", True),
        dataloader_workers=assumed.get("dataloader_workers", 0),
        persistent_workers=assumed.get("persistent_workers", False),
        checkpoint_every_epochs=assumed.get("checkpoint_every_epochs", 1),
    )


EpochCallback = Callable[[int, list[float], dict[str, Any]], None]


def _loader_options(spec: DiffusionTrainingSpec, device: torch.device) -> dict[str, Any]:
    workers = max(0, spec.dataloader_workers)
    return {
        "num_workers": workers,
        "pin_memory": spec.pin_memory and device.type == "cuda",
        "persistent_workers": spec.persistent_workers and workers > 0,
    }


def _optimizer(parameters: Any, learning_rate: float, spec: DiffusionTrainingSpec, device: torch.device) -> torch.optim.Optimizer:
    fused = spec.fused_optimizer and device.type == "cuda"
    try:
        return torch.optim.Adam(parameters, lr=learning_rate, fused=fused)
    except TypeError:
        return torch.optim.Adam(parameters, lr=learning_rate)


def _amp_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _grad_scaler(device: torch.device, amp_dtype: str) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == "fp16")


def _restore_runtime(
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    generator: torch.Generator | None,
    runtime_state: dict[str, Any] | None,
) -> None:
    if not runtime_state:
        return
    if runtime_state.get("optimizer") is not None:
        optimizer.load_state_dict(runtime_state["optimizer"])
    if runtime_state.get("scaler") is not None:
        scaler.load_state_dict(runtime_state["scaler"])
    if generator is not None and runtime_state.get("generator") is not None:
        generator.set_state(runtime_state["generator"])


def _runtime_state(
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    return {
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "generator": None if generator is None else generator.get_state(),
    }


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
    spec: DiffusionTrainingSpec,
    device: torch.device,
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
    return DataLoader(
        dataset,
        batch_size=spec.batch_size,
        shuffle=shuffle,
        **_loader_options(spec, device),
    )


def train_environment_classifier(
    classifier: EnvironmentClassifier,
    environments: np.ndarray,
    categories: np.ndarray,
    spec: DiffusionTrainingSpec,
    device: torch.device,
    *,
    start_epoch: int = 0,
    history: list[float] | None = None,
    runtime_state: dict[str, Any] | None = None,
    epoch_callback: EpochCallback | None = None,
) -> list[float]:
    """训练论文的两隐藏层 MLP 环境分类器。"""

    dataset = TensorDataset(
        torch.from_numpy(environments).float(), torch.from_numpy(categories).long()
    )
    loader = DataLoader(
        dataset,
        batch_size=spec.batch_size,
        shuffle=True,
        **_loader_options(spec, device),
    )
    optimizer = _optimizer(classifier.parameters(), spec.classifier_learning_rate, spec, device)
    scaler = _grad_scaler(device, spec.amp_dtype)
    _restore_runtime(optimizer, scaler, None, runtime_state)
    losses = list(history or [])
    classifier.train()
    for epoch in range(start_epoch, spec.classifier_epochs):
        total = 0.0
        count = 0
        for environment, category in loader:
            environment = environment.to(device, non_blocking=True)
            category = category.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(device, spec.amp_dtype):
                loss = nn.functional.cross_entropy(classifier(environment), category)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * environment.shape[0]
            count += environment.shape[0]
        losses.append(total / count)
        if epoch_callback is not None:
            epoch_callback(epoch + 1, losses, _runtime_state(optimizer, scaler))
    return losses


def train_strategy_autoencoder(
    autoencoder: StrategyAutoencoder,
    task_features: np.ndarray,
    task_mask: np.ndarray,
    model_spec: DiffusionModelSpec,
    training_spec: DiffusionTrainingSpec,
    device: torch.device,
    random_seed: int,
    *,
    start_epoch: int = 0,
    history: list[float] | None = None,
    runtime_state: dict[str, Any] | None = None,
    epoch_callback: EpochCallback | None = None,
) -> list[float]:
    """按论文描述使用随机低维资源方案训练对称升降维网络。"""

    features = torch.from_numpy(task_features[:, : model_spec.model_max_tasks]).float()
    masks = torch.from_numpy(task_mask[:, : model_spec.model_max_tasks]).bool()
    loader = DataLoader(
        TensorDataset(features, masks),
        batch_size=training_spec.batch_size,
        shuffle=True,
        **_loader_options(training_spec, device),
    )
    optimizer = _optimizer(
        autoencoder.parameters(), training_spec.autoencoder_learning_rate, training_spec, device
    )
    scaler = _grad_scaler(device, training_spec.amp_dtype)
    generator = torch.Generator(device=device).manual_seed(random_seed)
    _restore_runtime(optimizer, scaler, generator, runtime_state)
    low, high = model_spec.random_strategy_range
    losses = list(history or [])
    autoencoder.train()
    for epoch in range(start_epoch, training_spec.autoencoder_epochs):
        total = 0.0
        count = 0
        for batch_features, batch_mask in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            slot_mask = compact_strategy_slot_mask(
                batch_mask, model_spec.strategy_fields
            ).flatten(1)
            random_strategy = torch.rand(
                slot_mask.shape,
                generator=generator,
                device=device,
            ) * (high - low) + low
            random_strategy = random_strategy * slot_mask
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(device, training_spec.amp_dtype):
                reconstruction = autoencoder(random_strategy)
                loss = ((reconstruction - random_strategy) ** 2 * slot_mask).sum()
                loss = loss / slot_mask.sum().clamp_min(1)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * batch_features.shape[0]
            count += batch_features.shape[0]
        losses.append(total / count)
        if epoch_callback is not None:
            epoch_callback(epoch + 1, losses, _runtime_state(optimizer, scaler, generator))
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
    start_epoch: int = 0,
    history: list[float] | None = None,
    runtime_state: dict[str, Any] | None = None,
    epoch_callback: EpochCallback | None = None,
) -> list[float]:
    """按算法 1反向传播公式（24）的 δJ²，而非专家标签噪声损失。"""

    steps = denoising_steps or bundle.model_spec.denoising_steps
    normalized = bundle.preprocessor.transform(data["E"])
    loader = _loader(data, normalized, categories, training_spec, device, True)
    for parameter in bundle.classifier.parameters():
        parameter.requires_grad_(False)
    for parameter in bundle.autoencoder.parameters():
        parameter.requires_grad_(False)
    bundle.autoencoder.eval()
    bundle.diffusion.train()
    optimizer = _optimizer(
        bundle.diffusion.parameters(),
        training_spec.diffusion_learning_rate,
        training_spec,
        device,
    )
    scaler = _grad_scaler(device, training_spec.amp_dtype)
    generator = torch.Generator(device=device).manual_seed(random_seed)
    _restore_runtime(optimizer, scaler, generator, runtime_state)
    low, high = bundle.model_spec.random_strategy_range
    losses = list(history or [])
    use_channels_last = training_spec.channels_last and device.type == "cuda"
    if use_channels_last:
        bundle.diffusion.network.to(memory_format=torch.channels_last)
    for epoch in range(start_epoch, training_spec.diffusion_epochs):
        total = 0.0
        count = 0
        for _, category, nodes, gains, tasks, indices, task_mask in loader:
            category = category.to(device, non_blocking=True)
            nodes = nodes.to(device, non_blocking=True)
            gains = gains.to(device, non_blocking=True)
            tasks = tasks.to(device, non_blocking=True)
            indices = indices.to(device, non_blocking=True)
            task_mask = task_mask.to(device, non_blocking=True)
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
            if use_channels_last:
                clean_high = clean_high.contiguous(memory_format=torch.channels_last)
            noise = torch.randn(
                clean_high.shape, generator=generator, device=device, dtype=clean_high.dtype
            )
            timestep = torch.full(
                (clean_high.shape[0],), steps, device=device, dtype=torch.long
            )
            noisy = bundle.diffusion.add_noise(clean_high, noise, timestep)
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(device, training_spec.amp_dtype):
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
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                bundle.diffusion.parameters(), training_spec.gradient_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * tasks.shape[0]
            count += tasks.shape[0]
        losses.append(total / count)
        if epoch_callback is not None:
            epoch_callback(epoch + 1, losses, _runtime_state(optimizer, scaler, generator))
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


@torch.no_grad()
def generate_resource_strategy_trace(
    bundle: DMJCRBundle,
    environment: np.ndarray,
    node_features: np.ndarray,
    channel_gains: np.ndarray,
    task_features: np.ndarray,
    task_node_indices: np.ndarray,
    task_mask: np.ndarray,
    device: torch.device,
    *,
    random_seed: int,
    denoising_steps: int | None = None,
) -> StrategyGenerationTrace:
    """按论文图 4记录从初始噪声策略到最终策略的逐步评价轨迹。"""

    steps = denoising_steps or bundle.model_spec.denoising_steps
    if not 1 <= steps <= bundle.model_spec.denoising_steps:
        raise ValueError("denoising_steps 必须位于模型支持的扩散步数范围内")
    batch_size = environment.shape[0]
    arrays = (node_features, channel_gains, task_features, task_node_indices, task_mask)
    if any(item.shape[0] != batch_size for item in arrays):
        raise ValueError("环境、节点、信道和任务张量的 batch 大小必须一致")
    if np.any(task_mask[:, bundle.model_spec.model_max_tasks :]):
        raise ValueError("有效任务数超过 assumptions.diffusion.model_max_tasks")

    normalized = torch.from_numpy(bundle.preprocessor.transform(environment)).float().to(device)
    nodes = torch.from_numpy(node_features).float().to(device)
    gains = torch.from_numpy(channel_gains).float().to(device)
    tasks = torch.from_numpy(task_features).float().to(device)
    indices = torch.from_numpy(task_node_indices).long().to(device)
    masks = torch.from_numpy(task_mask).bool().to(device)
    model_tasks = tasks[:, : bundle.model_spec.model_max_tasks]
    model_masks = masks[:, : bundle.model_spec.model_max_tasks]

    bundle.classifier.eval()
    bundle.autoencoder.eval()
    bundle.diffusion.eval()
    categories = bundle.classifier(normalized).argmax(dim=1)
    compact_slots = compact_strategy_slot_mask(
        model_masks, bundle.model_spec.strategy_fields
    )
    generator = torch.Generator(device=device).manual_seed(random_seed)
    low, high = bundle.model_spec.random_strategy_range
    random_strategy = torch.rand(
        (batch_size, bundle.model_spec.strategy_width),
        generator=generator,
        device=device,
    ) * (high - low) + low
    random_strategy = random_strategy * compact_slots.flatten(1)
    clean_high = bundle.autoencoder.expand(random_strategy)
    noise = torch.randn(
        clean_high.shape,
        generator=generator,
        device=device,
        dtype=clean_high.dtype,
    )
    timestep = torch.full((batch_size,), steps, device=device, dtype=torch.long)
    value = bundle.diffusion.add_noise(clean_high, noise, timestep)

    raw_weighted: list[np.ndarray] = []
    raw_latency: list[np.ndarray] = []
    raw_energy: list[np.ndarray] = []
    raw_satisfied: list[np.ndarray] = []
    best_weighted: list[np.ndarray] = []
    best_latency: list[np.ndarray] = []
    best_energy: list[np.ndarray] = []
    best_satisfied: list[np.ndarray] = []
    current_best: tuple[Tensor, Tensor, Tensor, Tensor] | None = None

    def record(current: Tensor) -> None:
        nonlocal current_best
        reduced = nn.functional.softplus(bundle.autoencoder.reduce(current))
        compact = reduced.reshape(
            batch_size,
            bundle.model_spec.model_max_tasks,
            bundle.model_spec.strategy_fields,
        ) * compact_slots
        scores = expand_compact_strategy(compact, model_tasks, model_masks)
        metrics = evaluate_strategy_tensor(
            scores,
            nodes,
            gains,
            tasks,
            indices,
            masks,
            bundle.objective_spec,
        )
        raw_values = (
            metrics.weighted_objective,
            metrics.mean_normalized_latency,
            metrics.mean_normalized_energy,
            metrics.deadline_satisfied_ratio,
        )
        if current_best is None:
            current_best = tuple(item.clone() for item in raw_values)  # type: ignore[assignment]
        else:
            improved = raw_values[0] < current_best[0]
            current_best = tuple(
                torch.where(improved, new, old)
                for new, old in zip(raw_values, current_best, strict=True)
            )  # type: ignore[assignment]
        targets = (raw_weighted, raw_latency, raw_energy, raw_satisfied)
        best_targets = (best_weighted, best_latency, best_energy, best_satisfied)
        for target, item in zip(targets, raw_values, strict=True):
            target.append(item.cpu().numpy().copy())
        for target, item in zip(best_targets, current_best, strict=True):
            target.append(item.cpu().numpy().copy())

    # 横轴0对应论文算法1中的初始表示x_A，随后记录每个确定性反向步骤。
    record(value)
    diffusion_timesteps = [steps]
    for step in range(steps, 0, -1):
        timestep = torch.full((batch_size,), step, device=device, dtype=torch.long)
        value = bundle.diffusion.reverse_step(value, timestep, categories)
        record(value)
        diffusion_timesteps.append(step - 1)

    return StrategyGenerationTrace(
        generation_steps=np.arange(steps + 1, dtype=np.int64),
        diffusion_timesteps=np.asarray(diffusion_timesteps, dtype=np.int64),
        categories=categories.cpu().numpy().copy(),
        raw_weighted_objective=np.stack(raw_weighted),
        raw_normalized_latency=np.stack(raw_latency),
        raw_normalized_energy=np.stack(raw_energy),
        raw_deadline_satisfied_ratio=np.stack(raw_satisfied),
        best_weighted_objective=np.stack(best_weighted),
        best_normalized_latency=np.stack(best_latency),
        best_normalized_energy=np.stack(best_energy),
        best_deadline_satisfied_ratio=np.stack(best_satisfied),
    )


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
    training_state: dict[str, Any] | None = None,
) -> None:
    """保存可独立推理的模型、预处理参数和训练历史。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(
        {
            "checkpoint_version": 2,
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
            "training_state": training_state,
        },
        temporary,
    )
    os.replace(temporary, destination)


def load_bundle(path: str | Path, device: torch.device) -> tuple[DMJCRBundle, dict[str, Any]]:
    """恢复训练检查点。"""

    bundle, history, _ = load_training_bundle(path, device)
    return bundle, history


def load_training_bundle(
    path: str | Path,
    device: torch.device,
) -> tuple[DMJCRBundle, dict[str, Any], dict[str, Any] | None]:
    """恢复模型，并在存在时返回可用于断点续训的完整运行状态。"""

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
    return bundle, checkpoint.get("history", {}), checkpoint.get("training_state")


__all__ = [
    "DMJCRBundle",
    "DiffusionTrainingSpec",
    "EnvironmentPreprocessor",
    "StrategyGenerationTrace",
    "create_bundle",
    "fit_environment_preprocessor",
    "generate_resource_strategy",
    "generate_resource_strategy_trace",
    "load_bundle",
    "load_training_bundle",
    "load_diffusion_split",
    "save_bundle",
    "train_diffusion_model",
    "train_environment_classifier",
    "train_strategy_autoencoder",
    "training_spec_from_config",
]
