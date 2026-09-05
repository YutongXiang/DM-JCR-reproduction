"""验证环境、标签策略和数据集划分的一体化生成流程。"""

from dataclasses import replace
import json

import numpy as np

from dm_jcr.config import load_config
from dm_jcr.data_generation import (
    StrategySearchSpec,
    generate_and_save_dataset,
    generate_scenario,
    scenario_spec_from_config,
    search_high_quality_strategy,
)
from dm_jcr.environment import encode_environment, tensor_spec_from_config
from dm_jcr.resource_allocation import ObjectiveNormalization, ObjectiveWeights
from dm_jcr.strategy_codec import strategy_spec_from_config


def _objective_settings(config):
    return (
        ObjectiveNormalization(config["assumptions"]["objective"]["energy_reference_j"]),
        ObjectiveWeights(
            config["paper"]["objective"]["latency_weight"],
            config["paper"]["objective"]["energy_weight"],
        ),
        config["paper"]["task"]["cpu_energy_coefficient"],
    )


def test_generated_scenario_contains_all_task_modes_and_valid_relations() -> None:
    config = load_config()
    scenario = generate_scenario(
        np.random.default_rng(7),
        3,
        scenario_spec_from_config(config),
    )

    modes = {mapping.mode for mapping in scenario.snapshot.mappings}
    assert modes == {"direct", "relay_computation", "v2v_relay"}
    assert scenario.task_count == (
        len(scenario.direct_contexts)
        + len(scenario.relay_contexts)
        + len(scenario.v2v_contexts)
    )
    assert len(scenario.snapshot.links) == (
        len(scenario.snapshot.nodes) * (len(scenario.snapshot.nodes) - 1)
    )
    assert all(capacity.node_id.startswith(("uav-", "rsu-")) for capacity in scenario.capacities)

    encoded = encode_environment(scenario.snapshot, tensor_spec_from_config(config))
    assert encoded.task_mask.sum() == scenario.task_count
    assert np.all(np.isfinite(encoded.flat_vector()))


def test_generator_supports_more_tasks_than_vehicles() -> None:
    config = load_config()
    spec = scenario_spec_from_config(config)
    fixed = replace(
        spec,
        vehicle_count_range=(4, 4),
        task_count_range=(150, 150),
    )

    scenario = generate_scenario(np.random.default_rng(17), 0, fixed)

    assert len(scenario.snapshot.nodes) >= 4
    assert scenario.task_count == 150
    assert len({task.source_vehicle_id for task in scenario.snapshot.tasks}) <= 4


def test_strategy_search_returns_candidate_not_worse_than_baseline() -> None:
    config = load_config()
    rng = np.random.default_rng(11)
    scenario = generate_scenario(rng, 0, scenario_spec_from_config(config))
    normalization, weights, coefficient = _objective_settings(config)
    result = search_high_quality_strategy(
        rng,
        scenario,
        strategy_spec_from_config(config),
        StrategySearchSpec(6, 0.5, 0.05, 20.0, 2.0, False, 1),
        normalization,
        weights,
        coefficient,
    )

    assert result.evaluated_candidates == 6
    assert result.penalized_objective <= result.baseline_penalized_objective
    assert result.strategy.task_mask.sum() == scenario.task_count
    assert np.all(result.strategy.values >= 0.0)


def test_dataset_writer_creates_reproducible_disjoint_split_files(tmp_path) -> None:
    config = load_config()
    sizes = {"train": 2, "validation": 1, "test": 1}
    first = tmp_path / "first"
    second = tmp_path / "second"

    manifest = generate_and_save_dataset(
        config,
        first,
        sizes,
        random_seed=19,
        candidate_count=2,
    )
    generate_and_save_dataset(
        config,
        second,
        sizes,
        random_seed=19,
        candidate_count=2,
    )

    assert json.loads((first / "manifest.json").read_text(encoding="utf-8")) == manifest
    for split, size in sizes.items():
        with np.load(first / f"{split}.npz") as left, np.load(second / f"{split}.npz") as right:
            assert left["E"].shape[0] == size
            assert left["x0"].shape[1:] == (150, 8)
            assert left["node_features"].shape[1:] == (135, 12)
            assert left["channel_gains"].shape[1:] == (135, 135)
            assert left["task_features"].shape[1:] == (150, 7)
            assert left["task_node_indices"].shape[1:] == (150, 4)
            np.testing.assert_array_equal(left["E"], right["E"])
            np.testing.assert_array_equal(left["x0"], right["x0"])
            assert np.all(left["penalized_objective"] >= left["objective"])
            assert np.all(left["resource_feasible"])
            assert np.all((0.0 <= left["deadline_satisfied_ratio"]) & (left["deadline_satisfied_ratio"] <= 1.0))
            assert np.all(
                left["penalized_objective"] <= left["baseline_penalized_objective"]
            )
