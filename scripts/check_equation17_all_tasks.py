"""使用统一配置端到端检查论文公式（17）中的全部任务类型。"""

from collections.abc import Sequence

from dm_jcr.resource_allocation import (
    equation17_resource_totals,
    evaluate_equation17_strategy,
    project_equation17_strategy,
)
from scripts._config_helpers import (
    capacities,
    direct_context,
    load_script_config,
    objective_settings,
    raw_direct,
    raw_relay,
    raw_v2v,
    relay_context,
    v2v_context,
)


def main(argv: Sequence[str] | None = None) -> None:
    config, experiment = load_script_config(
        "equation17",
        __doc__ or "Equation-17 check",
        argv,
    )
    capacity_items = capacities(experiment["capacities"])
    direct_records = experiment["direct_tasks"]
    relay_records = experiment["relay_tasks"]
    v2v_records = experiment["v2v_tasks"]
    normalization, weights, energy_coefficient = objective_settings(config)

    strategy = project_equation17_strategy(
        tuple(raw_direct(item) for item in direct_records),
        tuple(raw_relay(item) for item in relay_records),
        tuple(raw_v2v(item) for item in v2v_records),
        capacity_items,
    )
    result = evaluate_equation17_strategy(
        tuple(direct_context(item) for item in direct_records),
        tuple(relay_context(item) for item in relay_records),
        tuple(v2v_context(item) for item in v2v_records),
        strategy,
        capacity_items,
        normalization,
        weights,
        energy_coefficient=energy_coefficient,
    )

    print("=== Per-node totals ===")
    for node, (bandwidth, cpu, power) in equation17_resource_totals(strategy).items():
        print(
            f"{node}: bandwidth={bandwidth / 1e6:.2f} MHz "
            f"cpu={cpu / 1e9:.2f} GHz power={power:.2f} W"
        )
    print("\n=== Task types ===")
    for item in result.direct_tasks:
        print(
            f"direct {item.task_id}: latency={item.metrics.total_latency_s:.6f}s "
            f"energy={item.metrics.total_energy_j:.6f}J"
        )
    for item in result.relay_computation_tasks:
        print(
            f"relay-compute {item.task_id}: "
            f"latency={item.metrics.total_latency_s:.6f}s "
            f"energy={item.metrics.total_energy_j:.6f}J"
        )
    for item in result.v2v_relay_tasks:
        print(
            f"v2v-relay {item.task_id}: latency={item.metrics.total_latency_s:.6f}s "
            f"energy={item.metrics.total_energy_j:.6f}J"
        )
    print(f"\nmean_normalized_latency={result.mean_normalized_latency:.6f}")
    print(f"mean_normalized_energy={result.mean_normalized_energy:.6f}")
    print(f"J={result.weighted_objective:.6f}")
    print(f"feasible={result.feasible}")


if __name__ == "__main__":
    main()
