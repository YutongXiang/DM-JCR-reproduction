"""验证三类原始资源策略与统一低维张量之间的双向转换。"""

import numpy as np
import pytest

from dm_jcr.config import load_config
from dm_jcr.environment import TaskNodeMapping
from dm_jcr.relay_resource_allocation import RawRelayTaskAllocation
from dm_jcr.resource_allocation import RawTaskAllocation, RawV2VRelayAllocation
from dm_jcr.strategy_codec import (
    EncodedRawStrategy,
    StrategyTensorSpec,
    decode_raw_strategy,
    encode_raw_strategy,
    strategy_spec_from_config,
)


def _allocations():
    direct = RawTaskAllocation("direct", "rsu-1", 1, 2, 3)
    relay = RawRelayTaskAllocation(
        "relay", "uav-1", "rsu-1", 1, 2, 3, 4, 5, 6, 7, 8
    )
    v2v = RawV2VRelayAllocation("v2v", "uav-1", 9, 10, 11, 12)
    return direct, relay, v2v


def _mappings() -> tuple[TaskNodeMapping, ...]:
    return (
        TaskNodeMapping("v2v", "v2v_relay", relay_uav_id="uav-1", target_vehicle_id="vehicle-2"),
        TaskNodeMapping("direct", "direct", compute_node_id="rsu-1"),
        TaskNodeMapping("relay", "relay_computation", compute_node_id="rsu-1", relay_uav_id="uav-1"),
    )


def test_encode_raw_strategy_uses_one_semantic_layout_for_all_modes() -> None:
    direct, relay, v2v = _allocations()
    encoded = encode_raw_strategy((direct,), (relay,), (v2v,), _mappings(), StrategyTensorSpec(5))

    np.testing.assert_array_equal(encoded.values[0], [9, 0, 0, 10, 11, 0, 12, 0])
    np.testing.assert_array_equal(encoded.values[1], [1, 0, 0, 0, 2, 0, 3, 0])
    np.testing.assert_array_equal(encoded.values[2], [1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(encoded.values[3:], 0)
    assert encoded.task_mask.tolist() == [True, True, True, False, False]
    assert encoded.task_ids == ("v2v", "direct", "relay")


def test_strategy_round_trip_recovers_original_allocations() -> None:
    direct, relay, v2v = _allocations()
    encoded = encode_raw_strategy((direct,), (relay,), (v2v,), _mappings(), StrategyTensorSpec(3))

    decoded_direct, decoded_relay, decoded_v2v = decode_raw_strategy(encoded)

    assert decoded_direct == (direct,)
    assert decoded_relay == (relay,)
    assert decoded_v2v == (v2v,)
    assert encoded.flat_vector().shape == (24,)


def test_strategy_encoder_rejects_inconsistent_mapping() -> None:
    direct, relay, v2v = _allocations()
    wrong = (TaskNodeMapping("direct", "direct", compute_node_id="uav-1"),)

    with pytest.raises(ValueError, match="相同的 task_id"):
        encode_raw_strategy((direct,), (relay,), (), wrong, StrategyTensorSpec(3))

    with pytest.raises(ValueError, match="直接计算节点不匹配"):
        encode_raw_strategy((direct,), (), (), wrong, StrategyTensorSpec(3))

    with pytest.raises(ValueError, match="超过策略张量上限"):
        encode_raw_strategy((direct,), (relay,), (v2v,), _mappings(), StrategyTensorSpec(2))


def test_encoded_strategy_rejects_nonzero_padding() -> None:
    values = np.zeros((2, 8))
    values[1, 0] = 1.0
    mapping = TaskNodeMapping("direct", "direct", compute_node_id="rsu-1")

    with pytest.raises(ValueError, match="padding"):
        EncodedRawStrategy(values, np.array([True, False]), (mapping,))


def test_strategy_spec_is_loaded_from_unified_config() -> None:
    assert strategy_spec_from_config(load_config()).max_tasks == 150
