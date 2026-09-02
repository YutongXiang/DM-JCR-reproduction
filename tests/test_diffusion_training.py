"""验证环境分类预处理、模型检查点和论文推理阶段。"""

from dataclasses import replace

import numpy as np
import torch

from dm_jcr.config import load_config
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
    load_bundle,
    save_bundle,
)


def _bundle(environment_width: int) -> DMJCRBundle:
    config = load_config()
    spec = replace(
        diffusion_spec_from_config(config),
        denoising_steps=2,
        feature_channels=(4, 8, 16),
        environment_categories=2,
        classifier_hidden_sizes=(8, 4),
        autoencoder_hidden_sizes=(16, 32),
        model_max_tasks=3,
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
