"""End-to-end check for mixed direct and UAV-relayed task allocation.

Run from the repository root:

    python -m scripts.check_relay_resource_allocation

The script demonstrates the complete numerical path:
raw scores -> joint projection -> four-hop relay rates -> latency/energy -> J.
It is an executable example, not a unit test or an optimizer.
"""

from dm_jcr.relay_resource_allocation import (
    RawRelayTaskAllocation,
    RelayTaskContext,
    evaluate_joint_resource_strategy,
    joint_resource_totals,
    project_joint_resource_strategy,
)
from dm_jcr.resource_allocation import (
    DirectTaskContext,
    NodeResourceCapacity,
    ObjectiveNormalization,
    RawTaskAllocation,
)
from dm_jcr.task_model import ComputationTask


def main() -> None:
    capacities = (
        NodeResourceCapacity(
            node_id="uav-1",
            total_bandwidth_hz=120.0e6,
            total_cpu_frequency_hz=1.5e9,
            total_transmit_power_w=30.0,
        ),
        NodeResourceCapacity(
            node_id="rsu-1",
            total_bandwidth_hz=100.0e6,
            total_cpu_frequency_hz=3.0e9,
            total_transmit_power_w=50.0,
        ),
    )

    direct_contexts = (
        DirectTaskContext(
            task_id="vehicle-1/direct-task",
            node_id="rsu-1",
            task=ComputationTask(6.0e5, 1.5e8, 0.8, 0.1),
            uplink_channel_gain=2.0e-9,
            downlink_channel_gain=3.0e-9,
            vehicle_transmit_power_w=0.2,
            noise_psd_w_hz=4.0e-21,
        ),
    )
    relay_contexts = (
        RelayTaskContext(
            task_id="vehicle-2/relay-task",
            relay_uav_id="uav-1",
            compute_node_id="rsu-1",
            task=ComputationTask(8.0e5, 2.0e8, 1.0, 0.1),
            vehicle_to_uav_channel_gain=3.0e-9,
            uav_to_node_channel_gain=6.0e-9,
            node_to_uav_channel_gain=5.0e-9,
            uav_to_vehicle_channel_gain=4.0e-9,
            vehicle_transmit_power_w=0.2,
            noise_psd_w_hz=4.0e-21,
            forwarding_cycles_per_bit=0.1,
        ),
    )

    direct_raw = (
        RawTaskAllocation(
            task_id="vehicle-1/direct-task",
            node_id="rsu-1",
            bandwidth_score=1.0,
            cpu_score=2.0,
            power_score=1.0,
        ),
    )
    relay_raw = (
        RawRelayTaskAllocation(
            task_id="vehicle-2/relay-task",
            relay_uav_id="uav-1",
            compute_node_id="rsu-1",
            vehicle_to_uav_bandwidth_score=1.0,
            uav_to_node_bandwidth_score=2.0,
            node_to_uav_bandwidth_score=2.0,
            uav_to_vehicle_bandwidth_score=1.0,
            relay_cpu_score=1.0,
            compute_cpu_score=1.0,
            relay_power_score=1.0,
            compute_node_power_score=2.0,
        ),
    )

    strategy = project_joint_resource_strategy(
        direct_raw_allocations=direct_raw,
        relay_raw_allocations=relay_raw,
        capacities=capacities,
    )
    evaluation = evaluate_joint_resource_strategy(
        direct_contexts=direct_contexts,
        relay_contexts=relay_contexts,
        strategy=strategy,
        capacities=capacities,
        normalization=ObjectiveNormalization(energy_reference_j=10.0),
    )

    print("=== Direct task ===")
    for item in evaluation.direct_tasks:
        print(
            item.task_id,
            f"node={item.node_id}",
            f"bandwidth={item.allocation.bandwidth_hz / 1e6:.2f} MHz",
            f"cpu={item.allocation.cpu_frequency_hz / 1e9:.2f} GHz",
            f"power={item.allocation.node_transmit_power_w:.2f} W",
            f"latency={item.metrics.total_latency_s:.6f} s",
            f"energy={item.metrics.total_energy_j:.6f} J",
        )

    print("\n=== Relay task ===")
    for item in evaluation.relay_tasks:
        allocation = item.allocation
        print(
            item.task_id,
            f"path={item.relay_uav_id}->{item.compute_node_id}",
        )
        print(
            "  bandwidth:",
            f"V2U={allocation.vehicle_to_uav_bandwidth_hz / 1e6:.2f} MHz",
            f"U2N={allocation.uav_to_node_bandwidth_hz / 1e6:.2f} MHz",
            f"N2U={allocation.node_to_uav_bandwidth_hz / 1e6:.2f} MHz",
            f"U2V={allocation.uav_to_vehicle_bandwidth_hz / 1e6:.2f} MHz",
        )
        print(
            "  compute/power:",
            f"relay_cpu={allocation.relay_cpu_frequency_hz / 1e9:.2f} GHz",
            f"node_cpu={allocation.compute_cpu_frequency_hz / 1e9:.2f} GHz",
            f"relay_power={allocation.relay_transmit_power_w:.2f} W",
            f"node_power={allocation.compute_node_transmit_power_w:.2f} W",
        )
        print(
            "  result:",
            f"latency={item.metrics.total_latency_s:.6f} s",
            f"energy={item.metrics.total_energy_j:.6f} J",
        )

    print("\n=== Per-node totals ===")
    for node_id, (bandwidth, cpu, power) in joint_resource_totals(strategy).items():
        print(
            node_id,
            f"bandwidth={bandwidth / 1e6:.2f} MHz",
            f"cpu={cpu / 1e9:.2f} GHz",
            f"power={power:.2f} W",
        )

    print("\n=== Joint objective ===")
    print(f"mean_normalized_latency={evaluation.mean_normalized_latency:.6f}")
    print(f"mean_normalized_energy={evaluation.mean_normalized_energy:.6f}")
    print(f"J={evaluation.weighted_objective:.6f}")
    print(f"feasible={evaluation.feasible}")


if __name__ == "__main__":
    main()
