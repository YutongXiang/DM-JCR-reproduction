"""验证扩散训练数据集的结构、语义、可行性和配置兼容性。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from dm_jcr.diffusion_objective import evaluate_strategy_tensor, objective_tensor_spec_from_config
from dm_jcr.environment import tensor_spec_from_config
from dm_jcr.strategy_codec import strategy_spec_from_config


SPLITS = ("train", "validation", "test")
REQUIRED_FIELDS = {
    "E", "x0", "node_features", "channel_gains", "blockage", "topology",
    "task_features", "task_node_indices", "node_mask", "task_mask", "objective",
    "penalized_objective", "baseline_objective", "baseline_penalized_objective",
    "resource_feasible", "deadline_feasible", "deadline_satisfied_ratio", "feasible",
    "evaluated_candidates",
}


def config_sha256(config: Mapping[str, Any]) -> str:
    """返回与数据生成清单一致的配置指纹。"""

    payload = json.dumps(
        {
            "paper": config["paper"],
            "assumptions": {
                name: config["assumptions"][name]
                for name in (
                    "channel", "objective", "relay_accounting", "tensor",
                    "dataset_generation", "strategy_search",
                )
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _flat_environment(
    data: Mapping[str, np.ndarray], start: int, stop: int,
) -> np.ndarray:
    indices = data["task_node_indices"][start:stop].astype(np.float64, copy=True)
    present = indices >= 0
    max_nodes = data["node_features"].shape[1]
    indices[present] /= max(1, max_nodes - 1)
    parts = (
        data["node_features"][start:stop].reshape(len(indices), -1),
        data["channel_gains"][start:stop].reshape(len(indices), -1),
        data["blockage"][start:stop].reshape(len(indices), -1),
        data["topology"][start:stop].reshape(len(indices), -1),
        data["task_features"][start:stop].reshape(len(indices), -1),
        indices.reshape(len(indices), -1),
        data["node_mask"][start:stop].astype(np.float64),
        data["task_mask"][start:stop].astype(np.float64),
    )
    return np.concatenate(parts, axis=1)


def _add(issues: list[dict[str, str]], split: str, message: str) -> None:
    issues.append({"level": "error", "split": split, "message": message})


def _shape_errors(
    data: Mapping[str, np.ndarray], sample_count: int, max_nodes: int,
    max_tasks: int, environment_width: int,
) -> list[str]:
    expected = {
        "E": (sample_count, environment_width),
        "x0": (sample_count, max_tasks, 8),
        "node_features": (sample_count, max_nodes, 12),
        "channel_gains": (sample_count, max_nodes, max_nodes),
        "blockage": (sample_count, max_nodes, max_nodes),
        "topology": (sample_count, max_nodes, max_nodes),
        "task_features": (sample_count, max_tasks, 7),
        "task_node_indices": (sample_count, max_tasks, 4),
        "node_mask": (sample_count, max_nodes),
        "task_mask": (sample_count, max_tasks),
    }
    expected.update({name: (sample_count,) for name in REQUIRED_FIELDS - set(expected)})
    return [
        f"{name}形状应为{shape}，实际为{data[name].shape}"
        for name, shape in expected.items() if data[name].shape != shape
    ]


def _semantic_errors(data: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    node_mask = data["node_mask"].astype(bool)
    task_mask = data["task_mask"].astype(bool)
    nodes, tasks = data["node_features"], data["task_features"]
    indices, x0 = data["task_node_indices"], data["x0"]
    contiguous_nodes = np.maximum.accumulate(node_mask[:, ::-1], axis=1)[:, ::-1]
    contiguous_tasks = np.maximum.accumulate(task_mask[:, ::-1], axis=1)[:, ::-1]
    if not np.array_equal(node_mask, contiguous_nodes):
        errors.append("node_mask的有效节点必须从索引0开始连续排列")
    if not np.array_equal(task_mask, contiguous_tasks):
        errors.append("task_mask的有效任务必须从索引0开始连续排列")
    if np.any(nodes[~node_mask] != 0.0) or np.any(tasks[~task_mask] != 0.0):
        errors.append("补零区域包含非零节点或任务特征")
    if np.any(indices[~task_mask] != -1) or np.any(x0[~task_mask] != 0.0):
        errors.append("无效任务区域的节点索引应为-1且策略应为0")
    active_nodes, active_tasks = nodes[node_mask], tasks[task_mask]
    if active_nodes.size and not np.allclose(active_nodes[:, :3].sum(axis=1), 1.0):
        errors.append("有效节点的类型独热编码不合法")
    if active_tasks.size and not np.allclose(active_tasks[:, 4:7].sum(axis=1), 1.0):
        errors.append("有效任务的模式独热编码不合法")
    if np.any(indices[task_mask] < -1) or np.any(indices[task_mask] >= nodes.shape[1]):
        errors.append("任务节点索引超出合法范围")
    for sample in range(len(task_mask)):
        valid = task_mask[sample]
        if not np.any(valid):
            errors.append(f"样本{sample}没有有效任务")
            continue
        count = int(node_mask[sample].sum())
        idx = indices[sample, valid]
        modes = np.argmax(tasks[sample, valid, 4:7], axis=1)
        if np.any(idx[idx >= 0] >= count):
            errors.append(f"样本{sample}的任务引用了补零节点")
            continue
        if np.any(np.argmax(nodes[sample, idx[:, 0], :3], axis=1) != 0):
            errors.append(f"样本{sample}存在非车辆任务源")
        direct, relay, v2v = modes == 0, modes == 1, modes == 2
        if np.any(idx[direct, 1] < 0) or np.any(idx[direct, 2:] != -1):
            errors.append(f"样本{sample}的直连任务节点映射不合法")
        if np.any(idx[relay, 1:3] < 0) or np.any(idx[relay, 3] != -1):
            errors.append(f"样本{sample}的中继计算任务节点映射不合法")
        if np.any(idx[v2v, 1] != -1) or np.any(idx[v2v, 2:] < 0):
            errors.append(f"样本{sample}的V2V任务节点映射不合法")
        node_types = np.argmax(nodes[sample, :count, :3], axis=1)
        if np.any(direct) and np.any(~np.isin(node_types[idx[direct, 1]], (1, 2))):
            errors.append(f"样本{sample}的直连计算节点不是UAV或RSU")
        if np.any(relay) and (
            np.any(node_types[idx[relay, 1]] != 2)
            or np.any(node_types[idx[relay, 2]] != 1)
        ):
            errors.append(f"样本{sample}的中继计算节点类型不合法")
        if np.any(v2v) and (
            np.any(node_types[idx[v2v, 2]] != 1)
            or np.any(node_types[idx[v2v, 3]] != 0)
        ):
            errors.append(f"样本{sample}的V2V节点类型不合法")
        topology = data["topology"][sample]
        gains = data["channel_gains"][sample]
        expected_topology = np.ones((count, count), dtype=topology.dtype)
        np.fill_diagonal(expected_topology, 0.0)
        if not np.array_equal(topology[:count, :count], expected_topology):
            errors.append(f"样本{sample}的有效节点拓扑不是完整有向图")
        if np.any(gains[:count, :count][~expected_topology.astype(bool)] != 0.0):
            errors.append(f"样本{sample}的信道增益对角线应为0")
        if np.any(gains[:count, :count][expected_topology.astype(bool)] <= 0.0):
            errors.append(f"样本{sample}存在非正的有效链路增益")
        if np.any(topology[count:, :] != 0.0) or np.any(topology[:, count:] != 0.0):
            errors.append(f"样本{sample}的拓扑补零区域非零")
    valid_slots = np.zeros_like(x0, dtype=bool)
    modes = tasks[..., 4:7].argmax(axis=-1)
    valid_slots[..., 0] = task_mask
    valid_slots[..., 3] = task_mask & (modes != 0)
    valid_slots[..., 4] = task_mask
    valid_slots[..., 6] = task_mask
    for slot in (1, 2, 5, 7):
        valid_slots[..., slot] = task_mask & (modes == 1)
    if np.any(x0[~valid_slots] != 0.0) or np.any(x0[valid_slots] <= 0.0):
        errors.append("x0的有效槽位必须为正，未使用槽位必须为0")
    search = config["assumptions"]["strategy_search"]
    if np.any(x0[valid_slots] < search["minimum_score"]) or np.any(
        x0[valid_slots] > search["maximum_score"]
    ):
        errors.append("x0超出配置的候选策略分数范围")
    return errors


def _recompute_objective(
    data: Mapping[str, np.ndarray], config: Mapping[str, Any], batch_size: int,
) -> np.ndarray:
    values: list[np.ndarray] = []
    spec = objective_tensor_spec_from_config(dict(config))
    for start in range(0, len(data["E"]), batch_size):
        stop = min(start + batch_size, len(data["E"]))
        objective = evaluate_strategy_tensor(
            torch.as_tensor(data["x0"][start:stop], dtype=torch.float64),
            torch.as_tensor(data["node_features"][start:stop], dtype=torch.float64),
            torch.as_tensor(data["channel_gains"][start:stop], dtype=torch.float64),
            torch.as_tensor(data["task_features"][start:stop], dtype=torch.float64),
            torch.as_tensor(data["task_node_indices"][start:stop]),
            torch.as_tensor(data["task_mask"][start:stop]),
            spec,
        ).weighted_objective
        values.append(objective.detach().cpu().numpy())
    return np.concatenate(values)


def validate_diffusion_dataset(
    config: Mapping[str, Any], directory: str | Path, *, profile: str = "formal",
) -> dict[str, Any]:
    """检查数据集；formal还要求样本规模、全可行标签和任务规模覆盖度。"""

    if profile not in {"structural", "formal"}:
        raise ValueError("profile必须是structural或formal")
    directory = Path(directory)
    settings = config["experiments"]["dataset_validation"]
    tensor, strategy = tensor_spec_from_config(config), strategy_spec_from_config(config)
    max_nodes, max_tasks = tensor.max_nodes, strategy.max_tasks
    environment_width = (
        max_nodes * 12 + 3 * max_nodes * max_nodes + max_tasks * 7
        + max_tasks * 4 + max_nodes + max_tasks
    )
    issues: list[dict[str, str]] = []
    summaries: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    manifest_path = directory / "manifest.json"
    manifest: dict[str, Any] = {}
    if not manifest_path.is_file():
        _add(issues, "dataset", "缺少manifest.json")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            _add(issues, "dataset", f"manifest.json无法读取：{error}")
    if manifest:
        if manifest.get("schema_version") != 2:
            _add(issues, "dataset", "数据集模式版本不是2，请用当前生成器重新生成")
        if manifest.get("config_sha256") != config_sha256(config):
            _add(issues, "dataset", "数据集配置指纹与当前配置不一致")
        if manifest.get("environment_width") != environment_width:
            _add(issues, "dataset", "清单中的环境宽度与当前编码不一致")
        if manifest.get("strategy_shape") != [max_tasks, 8]:
            _add(issues, "dataset", "清单中的策略形状与当前编码不一致")
    minimums = {
        "train": int(settings["minimum_train_samples"]),
        "validation": int(settings["minimum_validation_samples"]),
        "test": int(settings["minimum_test_samples"]),
    }
    for split in SPLITS:
        path = directory / f"{split}.npz"
        if not path.is_file():
            _add(issues, split, f"缺少{path.name}")
            continue
        try:
            with np.load(path, allow_pickle=False) as source:
                missing = REQUIRED_FIELDS - set(source.files)
                if missing:
                    _add(issues, split, f"缺少字段：{sorted(missing)}")
                    continue
                data = {name: source[name] for name in REQUIRED_FIELDS}
        except (OSError, ValueError) as error:
            _add(issues, split, f"NPZ无法读取：{error}")
            continue
        count = len(data["E"])
        if count == 0:
            _add(issues, split, "数据划分不能为空")
            continue
        for message in _shape_errors(data, count, max_nodes, max_tasks, environment_width):
            _add(issues, split, message)
        if any(issue["split"] == split for issue in issues):
            continue
        numeric = [value for value in data.values() if np.issubdtype(value.dtype, np.number)]
        if any(not np.all(np.isfinite(value)) for value in numeric):
            _add(issues, split, "存在NaN或无穷大")
            continue
        for message in _semantic_errors(data, config):
            _add(issues, split, message)
        batch_size = int(settings["objective_batch_size"])
        if batch_size <= 0:
            raise ValueError("objective_batch_size必须为正整数")
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            if not np.allclose(
                _flat_environment(data, start, stop), data["E"][start:stop],
                atol=1e-12, rtol=1e-12,
            ):
                _add(issues, split, f"E与结构化环境字段不一致，错误位于样本{start}附近")
                break
        recomputed = _recompute_objective(data, config, batch_size)
        if not np.allclose(
            recomputed, data["objective"],
            atol=float(settings["objective_absolute_tolerance"]),
            rtol=float(settings["objective_relative_tolerance"]),
        ):
            maximum = float(np.max(np.abs(recomputed - data["objective"])))
            _add(issues, split, f"重算目标值与存储值不一致，最大误差={maximum:.3e}")
        if np.any(data["penalized_objective"] > data["baseline_penalized_objective"] + 1e-10):
            _add(issues, split, "候选策略的惩罚目标劣于基线")
        task_counts = data["task_mask"].sum(axis=1)
        summaries[split] = {
            "samples": count,
            "task_count_min": int(task_counts.min()),
            "task_count_max": int(task_counts.max()),
            "mean_objective": float(data["objective"].mean()),
            "feasible_ratio": float(data["feasible"].mean()),
            "resource_feasible_ratio": float(data["resource_feasible"].mean()),
            "deadline_feasible_ratio": float(data["deadline_feasible"].mean()),
            "mean_task_deadline_satisfied_ratio": float(data["deadline_satisfied_ratio"].mean()),
        }
        if manifest and manifest.get("splits", {}).get(split, {}).get("samples") != count:
            _add(issues, split, "实际样本数与清单不一致")
        if manifest and np.any(data["evaluated_candidates"] != manifest.get("candidate_count")):
            _add(issues, split, "候选策略计数与清单不一致")
        if profile == "formal":
            if count < minimums[split]:
                _add(issues, split, f"正式训练至少需要{minimums[split]}条样本，当前只有{count}条")
            if not np.all(data["resource_feasible"] & data["deadline_feasible"] & data["feasible"]):
                _add(issues, split, "正式数据包含不可行标签")
            if split == "train":
                low, high = config["assumptions"]["dataset_generation"]["task_count_range"]
                edges = np.linspace(low, high + 1, 5, dtype=int)
                minimum = max(1, int(np.ceil(count * float(settings["minimum_bin_fraction"]))))
                for left, right in zip(edges[:-1], edges[1:], strict=True):
                    members = int(((task_counts >= left) & (task_counts < right)).sum())
                    if members < minimum:
                        _add(issues, split, f"任务数区间[{left},{right - 1}]仅有{members}条，至少需要{minimum}条")
        for row in data["E"]:
            digest = hashlib.sha256(np.ascontiguousarray(row).view(np.uint8)).hexdigest()
            if digest in hashes:
                _add(issues, split, f"发现与{hashes[digest]}重复的环境样本")
                break
            hashes[digest] = split
    valid = not issues
    return {
        "profile": profile,
        "dataset_directory": str(directory.resolve()),
        "valid": valid,
        "formal_training_ready": bool(valid and profile == "formal"),
        "summaries": summaries,
        "issues": issues,
    }


__all__ = ["config_sha256", "validate_diffusion_dataset"]
