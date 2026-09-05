"""验证环境分类预处理、模型检查点和论文推理阶段。"""

from dataclasses import replace

import numpy as np
import torch

from dm_jcr.config import load_config
from dm_jcr.data_generation import generate_scenario, scenario_spec_from_config
from dm_jcr.diffusion import (
    ConditionalUNet,
    DeterministicDiffusion,
    EnvironmentClassifier,
    StrategyAutoencoder,
    diffusion_spec_from_config,
)
from dm_jcr.diffusion_objective import objective_tensor_spec_from_config
from dm_jcr.diffusion_training import (
    DMJCRBundle,
    EnvironmentPreprocessor,
    fit_environment_preprocessor,
    generate_resource_strategy,
    generate_resource_strategy_trace,
    load_bundle,
    load_training_bundle,
    save_bundle,
    train_environment_classifier,
    training_spec_from_config,
)
from dm_jcr.environment import encode_environment, tensor_spec_from_config


def _bundle(environment_width: int, model_max_tasks: int = 3) -> DMJCRBundle:
    config = load_config()
    spec = replace(
        diffusion_spec_from_config(config),
        denoising_steps=2,
        feature_channels=(4, 8, 16),
        environment_categories=2,
        classifier_hidden_sizes=(8, 4),
        autoencoder_hidden_sizes=(16, 32),
        model_max_tasks=model_max_tasks,
        time_embedding_size=8,
    )
    preprocessor = EnvironmentPreprocessor(
        np.zeros(environment_width),
        np.ones(environment_width),
        np.zeros((2, environment_width), dtype=np.float32),
    )
    return DMJCRBundle(
        spec,
        objective_tensor_spec_from_config(config),
        preprocessor,
        EnvironmentClassifier(environment_width, spec),
        StrategyAutoencoder(spec),
        DeterministicDiffusion(ConditionalUNet(spec), spec),
    )


def test_kmeans_environment_categories_are_reproducible() -> None:
    environments = np.array(
        [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]], dtype=np.float64
    )
    first, labels = fit_environment_preprocessor(environments, 2, 9)
    second, second_labels = fit_environment_preprocessor(environments, 2, 9)

    np.testing.assert_array_equal(labels, second_labels)
    np.testing.assert_allclose(first.cluster_centers, second.cluster_centers)
    np.testing.assert_array_equal(first.categories(environments), labels)


def test_checkpoint_round_trip_and_inference_mask(tmp_path) -> None:
    bundle = _bundle(5)
    checkpoint = tmp_path / "model.pt"
    save_bundle(bundle, checkpoint, {"diffusion": [1.0]})
    loaded, history = load_bundle(checkpoint, torch.device("cpu"))

    environment = np.zeros((1, 5), dtype=np.float64)
    task_features = np.zeros((1, 5, 7), dtype=np.float64)
    task_mask = np.zeros((1, 5), dtype=np.bool_)
    task_features[0, 0, 4] = 1.0
    task_features[0, 1, 6] = 1.0
    task_mask[0, :2] = True
    strategy = generate_resource_strategy(
        loaded,
        environment,
        task_features,
        task_mask,
        torch.device("cpu"),
        random_seed=3,
        denoising_steps=1,
    )

    assert history == {"diffusion": [1.0]}
    assert strategy.shape == (1, 5, 8)
    assert np.all(np.isfinite(strategy))
    assert np.all(strategy >= 0.0)
    np.testing.assert_array_equal(strategy[0, 0] > 0, [1, 0, 0, 0, 1, 0, 1, 0])
    np.testing.assert_array_equal(strategy[0, 1] > 0, [1, 0, 0, 1, 1, 0, 1, 0])
    np.testing.assert_array_equal(strategy[0, 2:], 0.0)


def test_checkpoint_round_trip_preserves_resume_state_atomically(tmp_path) -> None:
    bundle = _bundle(5)
    checkpoint = tmp_path / "resume.pt"
    training_state = {
        "phase": "diffusion",
        "completed_epochs": {"classifier": 1, "autoencoder": 1, "diffusion": 2},
        "runtime": {"optimizer": {"state": {}, "param_groups": []}},
    }

    save_bundle(bundle, checkpoint, {"diffusion": [2.0, 1.0]}, training_state)
    _, history, restored = load_training_bundle(checkpoint, torch.device("cpu"))

    assert history == {"diffusion": [2.0, 1.0]}
    assert restored == training_state
    assert not (tmp_path / "resume.pt.tmp").exists()


def test_classifier_training_can_resume_from_epoch_runtime() -> None:
    config = load_config()
    model_spec = replace(
        diffusion_spec_from_config(config),
        environment_categories=2,
        classifier_hidden_sizes=(8, 4),
    )
    training_spec = replace(
        training_spec_from_config(config),
        classifier_epochs=1,
        batch_size=2,
        amp_dtype="off",
        fused_optimizer=False,
        pin_memory=False,
    )
    environments = np.array(
        [[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]], dtype=np.float32
    )
    categories = np.array([0, 0, 1, 1], dtype=np.int64)
    classifier = EnvironmentClassifier(2, model_spec)
    captured = {}

    def remember(epoch, losses, runtime):
        captured.update(epoch=epoch, losses=list(losses), runtime=runtime)

    first = train_environment_classifier(
        classifier,
        environments,
        categories,
        training_spec,
        torch.device("cpu"),
        epoch_callback=remember,
    )
    second = train_environment_classifier(
        classifier,
        environments,
        categories,
        replace(training_spec, classifier_epochs=2),
        torch.device("cpu"),
        start_epoch=1,
        history=first,
        runtime_state=captured["runtime"],
    )

    assert captured["epoch"] == 1
    assert len(second) == 2
    assert all(np.isfinite(second))


def test_inference_supports_150_compact_tasks() -> None:
    bundle = _bundle(5, model_max_tasks=150)
    environment = np.zeros((1, 5), dtype=np.float64)
    task_features = np.zeros((1, 150, 7), dtype=np.float64)
    task_mask = np.ones((1, 150), dtype=np.bool_)
    for index in range(150):
        task_features[0, index, 4 + index % 3] = 1.0

    strategy = generate_resource_strategy(
        bundle,
        environment,
        task_features,
        task_mask,
        torch.device("cpu"),
        random_seed=5,
        denoising_steps=1,
    )

    assert strategy.shape == (1, 150, 8)
    assert np.all(np.isfinite(strategy))
    assert np.all(strategy >= 0.0)


def test_generation_trace_records_initial_and_each_reverse_step() -> None:
    config = load_config()
    scenario_spec = replace(
        scenario_spec_from_config(config),
        task_count_range=(3, 3),
    )
    scenario = generate_scenario(np.random.default_rng(29), 0, scenario_spec)
    encoded = encode_environment(scenario.snapshot, tensor_spec_from_config(config))
    environment = encoded.flat_vector()[None, :]
    bundle = _bundle(environment.shape[1], model_max_tasks=3)

    trace = generate_resource_strategy_trace(
        bundle,
        environment,
        encoded.node_features[None, ...],
        encoded.channel_gains[None, ...],
        encoded.task_features[None, ...],
        encoded.task_node_indices[None, ...],
        encoded.task_mask[None, ...],
        torch.device("cpu"),
        random_seed=29,
        denoising_steps=2,
    )

    np.testing.assert_array_equal(trace.generation_steps, [0, 1, 2])
    np.testing.assert_array_equal(trace.diffusion_timesteps, [2, 1, 0])
    assert trace.raw_weighted_objective.shape == (3, 1)
    assert np.all(np.isfinite(trace.raw_weighted_objective))
    assert np.all(np.diff(trace.best_weighted_objective[:, 0]) <= 0.0)
