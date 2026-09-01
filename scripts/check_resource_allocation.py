"""使用统一配置对论文公式（17）和（27）进行小型端到端检查。"""

from collections.abc import Sequence

from dm_jcr.resource_allocation import (
    evaluate_direct_resource_strategy,
    project_resource_strategy,
)
from scripts._config_helpers import (
    capacities,
    direct_context,
    load_script_config,
    objective_settings,
    raw_direct,
)


def main(argv: Sequence[str] | None = None) -> None:
    config, experiment = load_script_config(
        "direct_resource",
        __doc__ or "Direct resource-allocation check",
        argv,
    )
    capacity_items = capacities(experiment["capacities"])
    contexts = tuple(direct_context(item) for item in experiment["tasks"])
    raw = tuple(raw_direct(item) for item in experiment["tasks"])
    normalization, weights, energy_coefficient = objective_settings(config)

    allocation = project_resource_strategy(raw, capacity_items)
    evaluation = evaluate_direct_resource_strategy(
        contexts,
        allocation,
        capacity_items,
        normalization,
        weights,
        energy_coefficient=energy_coefficient,
    )

    for item in evaluation.tasks:
        print(
            item.task_id,
            f"bandwidth={item.allocation.bandwidth_hz / 1e6:.2f} MHz",
            f"cpu={item.allocation.cpu_frequency_hz / 1e9:.2f} GHz",
            f"power={item.allocation.node_transmit_power_w:.2f} W",
            f"latency={item.metrics.total_latency_s:.6f} s",
            f"energy={item.metrics.total_energy_j:.6f} J",
        )
    print(f"J={evaluation.weighted_objective:.6f}")
    print(f"feasible={evaluation.feasible}")


if __name__ == "__main__":
    main()
