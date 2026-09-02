"""使用统一配置检查直连与无人机中继混合资源分配。

在仓库根目录运行：

    python -m scripts.check_relay_resource_allocation

本脚本演示完整数值链路：原始分数 → 联合投影 → 四跳中继速率 → 时延/能耗
→ 目标值 J。它是可执行示例，不是单元测试或优化器。
"""

from collections.abc import Sequence

from dm_jcr.relay_resource_allocation import (
    evaluate_joint_resource_strategy,
    joint_resource_totals,
    project_joint_resource_strategy,
)
from scripts._config_helpers import (
    capacities,
    direct_context,
    load_script_config,
    objective_settings,
    raw_direct,
    raw_relay,
    relay_context,
)


def main(argv: Sequence[str] | None = None) -> None:
    config, experiment = load_script_config(
        "relay_resource",
        __doc__ or "中继资源分配检查",
        argv,
    )
    capacity_items = capacities(experiment["capacities"])
    direct_contexts = tuple(
        direct_context(item) for item in experiment["direct_tasks"]
    )
    relay_contexts = tuple(
        relay_context(item) for item in experiment["relay_tasks"]
    )
    direct_raw = tuple(raw_direct(item) for item in experiment["direct_tasks"])
    relay_raw = tuple(raw_relay(item) for item in experiment["relay_tasks"])
    normalization, weights, energy_coefficient = objective_settings(config)

    strategy = project_joint_resource_strategy(
        direct_raw_allocations=direct_raw,
        relay_raw_allocations=relay_raw,
        capacities=capacity_items,
    )
    evaluation = evaluate_joint_resource_strategy(
        direct_contexts=direct_contexts,
        relay_contexts=relay_contexts,
        strategy=strategy,
        capacities=capacity_items,
        normalization=normalization,
        weights=weights,
        energy_coefficient=energy_coefficient,
    )

    print("=== 直连任务 ===")
    for item in evaluation.direct_tasks:
        print(
            item.task_id,
            f"节点={item.node_id}",
            f"带宽={item.allocation.bandwidth_hz / 1e6:.2f} MHz",
            f"CPU={item.allocation.cpu_frequency_hz / 1e9:.2f} GHz",
            f"功率={item.allocation.node_transmit_power_w:.2f} W",
            f"时延={item.metrics.total_latency_s:.6f} s",
            f"能耗={item.metrics.total_energy_j:.6f} J",
        )

    print("\n=== 中继任务 ===")
    for item in evaluation.relay_tasks:
        allocation = item.allocation
        print(
            item.task_id,
            f"路径={item.relay_uav_id}->{item.compute_node_id}",
        )
        print(
            "  带宽：",
            f"V2U={allocation.vehicle_to_uav_bandwidth_hz / 1e6:.2f} MHz",
            f"U2N={allocation.uav_to_node_bandwidth_hz / 1e6:.2f} MHz",
            f"N2U={allocation.node_to_uav_bandwidth_hz / 1e6:.2f} MHz",
            f"U2V={allocation.uav_to_vehicle_bandwidth_hz / 1e6:.2f} MHz",
        )
        print(
            "  计算/功率：",
            f"中继CPU={allocation.relay_cpu_frequency_hz / 1e9:.2f} GHz",
            f"节点CPU={allocation.compute_cpu_frequency_hz / 1e9:.2f} GHz",
            f"中继功率={allocation.relay_transmit_power_w:.2f} W",
            f"节点功率={allocation.compute_node_transmit_power_w:.2f} W",
        )
        print(
            "  结果：",
            f"时延={item.metrics.total_latency_s:.6f} s",
            f"能耗={item.metrics.total_energy_j:.6f} J",
        )

    print("\n=== 各节点资源总量 ===")
    for node_id, (bandwidth, cpu, power) in joint_resource_totals(strategy).items():
        print(
            node_id,
            f"带宽={bandwidth / 1e6:.2f} MHz",
            f"CPU={cpu / 1e9:.2f} GHz",
            f"功率={power:.2f} W",
        )

    print("\n=== 联合目标 ===")
    print(f"平均归一化时延={evaluation.mean_normalized_latency:.6f}")
    print(f"平均归一化能耗={evaluation.mean_normalized_energy:.6f}")
    print(f"J={evaluation.weighted_objective:.6f}")
    print(f"是否可行={evaluation.feasible}")


if __name__ == "__main__":
    main()
