"""验证论文扩散公式、网络尺寸、确定性采样和可微公式（17）目标。"""

from dataclasses import replace

import numpy as np
import torch
from torch import nn

from dm_jcr.config import load_config
from dm_jcr.data_generation import generate_scenario, scenario_spec_from_config
from dm_jcr.diffusion import (
    ConditionalUNet,
    DeterministicDiffusion,
    EnvironmentClassifier,
    StrategyAutoencoder,
    diffusion_spec_from_config,
)
from dm_jcr.diffusion_objective import (
    evaluate_strategy_tensor,
    objective_tensor_spec_from_config,
)
from dm_jcr.environment import encode_environment, tensor_spec_from_config
from dm_jcr.resource_allocation import (
    ObjectiveNormalization,
    ObjectiveWeights,
    evaluate_equation17_strategy,
    project_equation17_strategy,
)
from dm_jcr.strategy_codec import EncodedRawStrategy, decode_raw_strategy


class _ConstantNoise(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, sample, timesteps, categories):
        return torch.full_like(sample, self.value)


def _small_spec():
    return replace(
        diffusion_spec_from_config(load_config()),
        denoising_steps=4,
        feature_channels=(4, 8, 16),
        environment_categories=3,
        classifier_hidden_sizes=(8, 4),
        autoencoder_hidden_sizes=(16, 32),
        model_max_tasks=3,
        time_embedding_size=8,
    )


def test_forward_noise_matches_equation_2() -> None:
    spec = _small_spec()
    diffusion = DeterministicDiffusion(_ConstantNoise(0.0), spec)
    clean = torch.full((2, 1, 28, 28), 2.0)
    noise = torch.full_like(clean, 3.0)
    timesteps = torch.tensor([1, 4])

    actual = diffusion.add_noise(clean, noise, timesteps)
    alpha_bar = diffusion.alpha_bar[timesteps].reshape(2, 1, 1, 1)
    expected = alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    torch.testing.assert_close(actual, expected)


def test_reverse_step_matches_paper_equation_19_without_random_term() -> None:
    spec = _small_spec()
    diffusion = DeterministicDiffusion(_ConstantNoise(0.25), spec)
    value = torch.ones((1, 1, 28, 28))
    timestep = torch.tensor([3])

    actual = diffusion.reverse_step(value, timestep, torch.tensor([1]))
    alpha = diffusion.alpha[3]
    beta = diffusion.beta[3]
    alpha_bar = diffusion.alpha_bar[3]
    expected = (value - beta / torch.sqrt(1.0 - alpha_bar) * 0.25) / torch.sqrt(alpha)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        diffusion.sample(value, torch.tensor([1]), steps=3),
        diffusion.sample(value, torch.tensor([1]), steps=3),
    )


def test_paper_networks_have_expected_shapes() -> None:
    spec = _small_spec()
    unet = ConditionalUNet(spec)
    autoencoder = StrategyAutoencoder(spec)
    classifier = EnvironmentClassifier(12, spec)

    image = torch.randn(2, 1, 28, 28)
    assert unet(image, torch.tensor([1, 2]), torch.tensor([0, 2])).shape == image.shape
    low = torch.rand(2, spec.strategy_width)
    assert autoencoder.expand(low).shape == image.shape
    assert autoencoder.reduce(image).shape == low.shape
    probabilities = classifier.probabilities(torch.randn(2, 12))
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))


def test_differentiable_objective_matches_existing_equation_17_evaluator() -> None:
    config = load_config()
    rng = np.random.default_rng(31)
    scenario = generate_scenario(rng, 0, scenario_spec_from_config(config))
    environment = encode_environment(scenario.snapshot, tensor_spec_from_config(config))
    task_count = scenario.task_count
    values = np.zeros((150, 8), dtype=np.float64)
    mask = np.zeros(150, dtype=np.bool_)
    for index, mapping in enumerate(scenario.snapshot.mappings):
        mask[index] = True
        if mapping.mode == "direct":
            values[index, (0, 4, 6)] = 1.0
        elif mapping.mode == "relay_computation":
            values[index] = 1.0
        else:
            values[index, (0, 3, 4, 6)] = 1.0
    encoded = EncodedRawStrategy(values, mask, scenario.snapshot.mappings)
    direct, relay, v2v = decode_raw_strategy(encoded)
    projected = project_equation17_strategy(direct, relay, v2v, scenario.capacities)
    expected = evaluate_equation17_strategy(
        scenario.direct_contexts,
        scenario.relay_contexts,
        scenario.v2v_contexts,
        projected,
        scenario.capacities,
        ObjectiveNormalization(config["assumptions"]["objective"]["energy_reference_j"]),
        ObjectiveWeights(0.5, 0.5),
        energy_coefficient=config["paper"]["task"]["cpu_energy_coefficient"],
    )

    scores = torch.tensor(values[:task_count], dtype=torch.float64).unsqueeze(0)
    scores.requires_grad_()
    actual = evaluate_strategy_tensor(
        scores,
        torch.tensor(environment.node_features, dtype=torch.float64).unsqueeze(0),
        torch.tensor(environment.channel_gains, dtype=torch.float64).unsqueeze(0),
        torch.tensor(environment.task_features, dtype=torch.float64).unsqueeze(0),
        torch.tensor(environment.task_node_indices, dtype=torch.long).unsqueeze(0),
        torch.tensor(environment.task_mask, dtype=torch.bool).unsqueeze(0),
        objective_tensor_spec_from_config(config),
    )

    np.testing.assert_allclose(
        actual.weighted_objective.detach().numpy(),
        [expected.weighted_objective],
        rtol=1e-8,
    )
    actual.weighted_objective.sum().backward()
    assert scores.grad is not None
    assert torch.all(torch.isfinite(scores.grad))
