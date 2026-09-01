"""验证公式（16）环境快照及固定尺寸张量编码。"""

import numpy as np
import pytest

from dm_jcr.config import load_config
from dm_jcr.environment import (
    EnvironmentLink,
    EnvironmentNode,
    EnvironmentSnapshot,
    EnvironmentTask,
    EnvironmentTensorSpec,
    TaskNodeMapping,
    encode_environment,
    tensor_spec_from_config,
)


def _node(node_id: str, node_type: str, x: float = 0.0) -> EnvironmentNode:
    return EnvironmentNode(
        node_id=node_id,
        node_type=node_type,
        position_m=np.array([x, 0.0, 100.0 if node_type == "uav" else 0.0]),
        velocity_mps=np.array([10.0, 0.0, 0.0]) if node_type == "vehicle" else np.zeros(3),
        remaining_bandwidth_hz=50.0,
        remaining_cpu_frequency_hz=60.0,
        remaining_transmit_power_w=70.0,
    )


def _snapshot() -> EnvironmentSnapshot:
    nodes = (
        _node("rsu-1", "rsu", 300.0),
        _node("vehicle-2", "vehicle", 100.0),
        _node("uav-1", "uav", 200.0),
        _node("vehicle-1", "vehicle"),
    )
    tasks = (
        EnvironmentTask("direct", "vehicle-1", 100.0, 200.0, 0.5, 0.1),
        EnvironmentTask("relay", "vehicle-2", 200.0, 300.0, 1.0, 0.2),
        EnvironmentTask("v2v", "vehicle-1", 300.0, 0.0, 0.25, 0.0),
    )
    mappings = (
        TaskNodeMapping("direct", "direct", compute_node_id="rsu-1"),
        TaskNodeMapping(
            "relay",
            "relay_computation",
            compute_node_id="rsu-1",
            relay_uav_id="uav-1",
        ),
        TaskNodeMapping(
            "v2v",
            "v2v_relay",
            relay_uav_id="uav-1",
            target_vehicle_id="vehicle-2",
        ),
    )
    links = (
        EnvironmentLink("vehicle-1", "uav-1", 2.0, blocked=False),
        EnvironmentLink("uav-1", "rsu-1", 1.0, blocked=True),
        EnvironmentLink("vehicle-2", "rsu-1", 0.0, blocked=False, available=False),
    )
    return EnvironmentSnapshot(4, nodes, links, tasks, mappings)


def _spec(**overrides: object) -> EnvironmentTensorSpec:
    values = {
        "max_vehicles": 2,
        "max_uavs": 1,
        "max_rsus": 1,
        "max_tasks": 4,
        "position_scale_m": 100.0,
        "speed_scale_mps": 10.0,
        "bandwidth_scale_hz": 10.0,
        "cpu_scale_hz": 20.0,
        "power_scale_w": 35.0,
        "task_bits_scale": 100.0,
        "task_cycles_scale": 100.0,
        "latency_scale_s": 0.5,
        "channel_gain_scale": 2.0,
    }
    values.update(overrides)
    return EnvironmentTensorSpec(**values)


def test_encode_environment_uses_stable_node_order_and_masks() -> None:
    encoded = encode_environment(_snapshot(), _spec())

    assert encoded.node_ids == ("vehicle-1", "vehicle-2", "uav-1", "rsu-1")
    assert encoded.task_ids == ("direct", "relay", "v2v")
    assert encoded.node_features.shape == (4, 12)
    assert encoded.task_features.shape == (4, 7)
    assert encoded.channel_gains.shape == (4, 4)
    assert encoded.node_mask.tolist() == [True, True, True, True]
    assert encoded.task_mask.tolist() == [True, True, True, False]

    np.testing.assert_array_equal(encoded.node_features[0, :3], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(encoded.node_features[1, 3:10], [1, 0, 0, 1, 0, 0, 5])
    np.testing.assert_array_equal(encoded.task_features[:, 4:], [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])


def test_encode_environment_preserves_equation_16_relations() -> None:
    encoded = encode_environment(_snapshot(), _spec())

    # 节点索引依次为 vehicle-1、vehicle-2、uav-1、rsu-1。
    np.testing.assert_array_equal(encoded.task_node_indices[:3], [
        [0, 3, -1, -1],
        [1, 3, 2, -1],
        [0, -1, 2, 1],
    ])
    assert encoded.channel_gains[0, 2] == 1.0
    assert encoded.blockage[2, 3] == 1.0
    assert encoded.topology[1, 3] == 0.0
    assert encoded.flat_vector().ndim == 1
    assert np.all(np.isfinite(encoded.flat_vector()))


def test_snapshot_rejects_invalid_references_and_roles() -> None:
    with pytest.raises(ValueError, match="未知节点"):
        EnvironmentSnapshot(
            0,
            (_node("vehicle-1", "vehicle"),),
            (EnvironmentLink("vehicle-1", "uav-missing", 1.0, False),),
            (),
            (),
        )

    with pytest.raises(ValueError, match="必须是 UAV"):
        EnvironmentSnapshot(
            0,
            (_node("vehicle-1", "vehicle"), _node("rsu-1", "rsu")),
            (),
            (EnvironmentTask("task", "vehicle-1", 1, 0, 1, 0),),
            (TaskNodeMapping("task", "v2v_relay", relay_uav_id="rsu-1", target_vehicle_id="vehicle-1"),),
        )


def test_encoder_rejects_capacity_overflow() -> None:
    with pytest.raises(ValueError, match="vehicle 节点数"):
        encode_environment(_snapshot(), _spec(max_vehicles=1))


def test_tensor_spec_is_loaded_from_unified_config() -> None:
    spec = tensor_spec_from_config(load_config())

    assert (spec.max_vehicles, spec.max_uavs, spec.max_rsus) == (100, 15, 20)
    assert spec.max_tasks == 150
    assert spec.max_nodes == 135
