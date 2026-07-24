"""Small end-to-end check for equations (17) and (27)."""

from dm_jcr.resource_allocation import (
    DirectTaskContext,
    NodeResourceCapacity,
    ObjectiveNormalization,
    RawTaskAllocation,
    evaluate_direct_resource_strategy,
    project_resource_strategy,
)
from dm_jcr.task_model import ComputationTask


def main() -> None:
    capacity = NodeResourceCapacity(
        node_id="rsu-1",
        total_bandwidth_hz=100.0e6,
        total_cpu_frequency_hz=3.0e9,
        total_transmit_power_w=50.0,
    )
    contexts = [
        DirectTaskContext(
            task_id="vehicle-1/task-1",
            node_id="rsu-1",
            task=ComputationTask(8.0e5, 2.0e8, 0.5, 0.1),
            uplink_channel_gain=2.0e-9,
            downlink_channel_gain=3.0e-9,
            vehicle_transmit_power_w=0.2,
            noise_psd_w_hz=4.0e-21,
        ),
        DirectTaskContext(
            task_id="vehicle-2/task-1",
            node_id="rsu-1",
            task=ComputationTask(1.6e6, 4.0e8, 0.8, 0.2),
            uplink_channel_gain=1.0e-9,
            downlink_channel_gain=2.0e-9,
            vehicle_transmit_power_w=0.2,
            noise_psd_w_hz=4.0e-21,
        ),
    ]
    raw = [
        RawTaskAllocation("vehicle-1/task-1", "rsu-1", 1.0, 2.0, 1.0),
        RawTaskAllocation("vehicle-2/task-1", "rsu-1", 3.0, 1.0, 2.0),
    ]

    allocation = project_resource_strategy(raw, [capacity])
    evaluation = evaluate_direct_resource_strategy(
        contexts,
        allocation,
        [capacity],
        ObjectiveNormalization(energy_reference_j=10.0),
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
