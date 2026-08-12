"""End-to-end check for every task type in equation (17)."""
from dm_jcr.relay_resource_allocation import RelayTaskContext, RawRelayTaskAllocation
from dm_jcr.resource_allocation import (
    DirectTaskContext, NodeResourceCapacity, ObjectiveNormalization,
    RawTaskAllocation, RawV2VRelayAllocation, V2VRelayTaskContext,
    equation17_resource_totals, evaluate_equation17_strategy,
    project_equation17_strategy,
)
from dm_jcr.task_model import ComputationTask


def task(bits, cycles, deadline, ratio=0.1):
    return ComputationTask(bits, cycles, deadline, ratio)


def main() -> None:
    capacities = (
        NodeResourceCapacity("uav-1", 120e6, 2e9, 30.0),
        NodeResourceCapacity("rsu-1", 100e6, 3e9, 50.0),
    )
    strategy = project_equation17_strategy(
        (RawTaskAllocation("direct", "rsu-1", 1, 2, 1),),
        (RawRelayTaskAllocation("relay-compute", "uav-1", "rsu-1", 1, 2, 2, 1, 1, 1, 1, 2),),
        (RawV2VRelayAllocation("v2v-relay", "uav-1", 1, 1, 1, 1),),
        capacities,
    )
    result = evaluate_equation17_strategy(
        (DirectTaskContext("direct", "rsu-1", task(8e5, 2e8, 1.0), 2e-9, 3e-9, 0.2, 4e-21),),
        (RelayTaskContext(
            "relay-compute", "uav-1", "rsu-1", task(8e5, 2e8, 2.0),
            3e-9, 4e-9, 4e-9, 3e-9, 0.2, 4e-21, 0.1,
        ),),
        (V2VRelayTaskContext(
            "v2v-relay", "vehicle-1", "vehicle-2", "uav-1",
            4e5, 1.0, 3e-9, 3e-9, 0.2, 4e-21, 0.1,
        ),),
        strategy, capacities, ObjectiveNormalization(energy_reference_j=10.0),
    )

    print("=== Per-node totals ===")
    for node, (b, f, p) in equation17_resource_totals(strategy).items():
        print(f"{node}: bandwidth={b/1e6:.2f} MHz cpu={f/1e9:.2f} GHz power={p:.2f} W")
    print("\n=== Task types ===")
    for x in result.direct_tasks:
        print(f"direct {x.task_id}: latency={x.metrics.total_latency_s:.6f}s energy={x.metrics.total_energy_j:.6f}J")
    for x in result.relay_computation_tasks:
        print(f"relay-compute {x.task_id}: latency={x.metrics.total_latency_s:.6f}s energy={x.metrics.total_energy_j:.6f}J")
    for x in result.v2v_relay_tasks:
        print(f"v2v-relay {x.task_id}: latency={x.metrics.total_latency_s:.6f}s energy={x.metrics.total_energy_j:.6f}J")
    print(f"\nmean_normalized_latency={result.mean_normalized_latency:.6f}")
    print(f"mean_normalized_energy={result.mean_normalized_energy:.6f}")
    print(f"J={result.weighted_objective:.6f}")
    print(f"feasible={result.feasible}")


if __name__ == "__main__":
    main()
