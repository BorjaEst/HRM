"""Trace Tree implementation for batched simulation rollouts.

This module provides a clean, minimal implementation of the Trace Tree
Pattern with explicit handling of dense data and static metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from numbers import Number
from typing import Any, Iterable, Iterator, List, Optional

import numpy as np


@dataclass
class TraceConfig:
    """Configuration for trace construction and validation.

    Attributes:
        strict: Whether to raise errors on invariant violations.
        global_paths: Paths treated as global (no batch axis enforced).
        rebase_time_on_slice: Whether to rebase time to zero on slices.
    """

    strict: bool = True
    global_paths: set[str] = field(default_factory=set)
    rebase_time_on_slice: bool = True


@dataclass
class TraceNode:
    """Node in a trace tree with dense data and static metadata."""

    data: dict[str, np.ndarray] = field(default_factory=dict)
    meta_static: dict[str, Any] = field(default_factory=dict)
    children: dict[str, "TraceNode"] = field(default_factory=dict)
    _buffers: dict[str, list[np.ndarray]] = field(default_factory=dict, repr=False)

    def child(self, name: str) -> "TraceNode":
        """Return (or create) a child node by name."""

        if name not in self.children:
            self.children[name] = TraceNode()
        return self.children[name]

    def append_data(self, key: str, value: np.ndarray) -> None:
        """Append a dense value to the buffer for a given key."""

        self._buffers.setdefault(key, []).append(value)

    def finalize(self, *, length: int, strict: bool) -> None:
        """Stack buffered data into dense arrays and recurse into children."""

        for key, values in self._buffers.items():
            if strict and len(values) != length:
                raise ValueError("Incomplete data buffer for key " f"{key!r}: expected {length}, got {len(values)}")
            self.data[key] = np.stack(values, axis=0)
        self._buffers.clear()

        for child in self.children.values():
            child.finalize(length=length, strict=strict)

    def slice_time(self, t0: int, t1: int, *, rebase: bool) -> "TraceNode":
        """Return a time-sliced copy of this node."""

        sliced = TraceNode()

        for key, arr in self.data.items():
            sliced.data[key] = arr[t0:t1]

        sliced.meta_static = dict(self.meta_static)

        for name, child in self.children.items():
            sliced.children[name] = child.slice_time(t0, t1, rebase=rebase)

        return sliced


@dataclass
class TraceTree:
    """Trace tree builder and container for a rollout."""

    config: TraceConfig = field(default_factory=TraceConfig)
    root: TraceNode = field(default_factory=TraceNode)
    batch_size: Optional[int] = None
    length: int = 0

    def append(self, state: Any) -> None:
        """Append a state snapshot into the trace tree."""

        t = self.length
        _append_state(node=self.root, state=state, t=t, path=(), tree=self)
        self.length += 1

    def finalize(self) -> None:
        """Finalize the trace by stacking buffered data into arrays."""

        self.root.finalize(length=self.length, strict=self.config.strict)

    def slice_time(self, t0: int, t1: int) -> "TraceTree":
        """Return a time-sliced copy of the trace."""

        t0 = max(0, t0)
        t1 = min(self.length, t1)
        sliced = TraceTree(config=self.config)
        sliced.root = self.root.slice_time(t0, t1, rebase=self.config.rebase_time_on_slice)
        sliced.length = max(0, t1 - t0)
        sliced.batch_size = self.batch_size
        return sliced

    def node(self, path: str) -> TraceNode:
        """Return the TraceNode at the given slash-delimited path."""
        return _get_node(self.root, path)

    def get(self, path: str) -> np.ndarray:
        """Return a dense array at a slash-delimited path.

        Args:
            path: Slash-delimited path to a dense array or indexed child.

        Returns:
            Dense NumPy array stored at the requested path.
        """
        node_path, key = _split_path(path)
        node = _get_node(self.root, node_path)
        return _resolve_dense_value(node, key, strict=self.config.strict)

    def export_dense_tree(self) -> Any:
        """Export dense trace data as a nested dict/list pytree."""
        return _export_dense_node(self.root, strict=self.config.strict)

    def export_meta_tree(self) -> dict[str, Any]:
        """Export static metadata as a nested dict tree."""
        return _export_meta_node(self.root, strict=self.config.strict)

    def export(self, *, flatten: bool = False, sep: str = "/") -> Any:
        """Export dense trace data, optionally flattening to a path map."""
        dense = self.export_dense_tree()
        if not flatten:
            return dense
        return flatten_pytree(dense, sep=sep)

    def get_meta(self) -> dict[str, Any]:
        """Return root metadata dictionary, if available."""
        meta = self.root.meta_static.get("meta")
        return dict(meta) if isinstance(meta, dict) else {}

    def get_environments(self) -> list[Any]:
        """Return environments stored in the trace metadata."""
        envs = self.root.meta_static.get("environments")
        return list(envs) if envs is not None else []

    def get_visited(self) -> Any:
        """Return visited masks stored in the trace metadata."""
        return self.root.meta_static.get("visited")

    def get_world(self, env_idx: int) -> Any:
        """Return the World object for a selected environment index."""
        envs = self.get_environments()
        if not envs:
            raise ValueError("TraceTree has no environments")
        if not (0 <= env_idx < len(envs)):
            raise IndexError(f"env_idx {env_idx} out of range [0, {len(envs)})")
        return envs[env_idx]

    def n_freq(self, base_path: str) -> int:
        """Return number of indexed children at a multiscale path."""
        node = self.node(base_path)
        if _is_indexed_children(node.children):
            return len(_sorted_indices(node.children.keys()))
        if _is_indexed_data(node.data):
            return len(_sorted_indices(node.data.keys()))
        return 0

    def validate_env_idx(self, env_idx: int) -> int:
        """Validate and return environment index."""
        batch_size = int(self.batch_size or 0)
        if not (0 <= env_idx < batch_size):
            raise IndexError(f"env_idx {env_idx} out of range [0, {batch_size})")
        return env_idx

    def validate_freq_idx(self, base_path: str, freq_idx: int) -> int:
        """Validate and return frequency index."""
        n_freq = self.n_freq(base_path)
        if not (0 <= freq_idx < n_freq):
            raise IndexError(f"freq_idx {freq_idx} out of range [0, {n_freq})")
        return freq_idx


def _append_state(*, node: TraceNode, state: Any, t: int, path: tuple[str, ...], tree: TraceTree) -> None:
    """Append a state tree into the trace tree (recursive)."""

    if _is_dataclass_instance(state):
        for field_info in fields(state):
            name = field_info.name
            value = getattr(state, name)
            _append_value(node=node, name=name, value=value, t=t, path=path, tree=tree)
        return

    raise TypeError("State must be a dataclass instance. " f"Got {type(state)!r} at path {"/".join(path) or "<root>"}.")


def _append_value(*, node: TraceNode, name: str, value: Any, t: int, path: tuple[str, ...], tree: TraceTree) -> None:
    """Append a single field value into the trace tree."""

    current_path = path + (name,)

    if _is_dataclass_instance(value):
        child = node.child(name)
        _append_state(node=child, state=value, t=t, path=current_path, tree=tree)
        return

    if _is_list_of_numeric_arrays(value):
        container = node.child(name)
        container.meta_static.setdefault("length", len(value))
        for idx, elem in enumerate(value):
            arr = _to_numeric_array(elem)
            _append_dense(node=container, key=str(idx), arr=arr, t=t, path=current_path + (str(idx),), tree=tree)
        return

    arr = _to_numeric_array(value)
    if arr is not None:
        _append_dense(node=node, key=name, arr=arr, t=t, path=current_path + (name,), tree=tree)
        return

    _record_static_meta(node, name, value, strict=tree.config.strict)


def _append_dense(*, node: TraceNode, key: str, arr: np.ndarray, t: int, path: tuple[str, ...], tree: TraceTree) -> None:
    """Append a dense numeric array with validation."""

    if arr.dtype == object:
        if tree.config.strict:
            raise ValueError(f"Irregular value for key {key!r} at t={t}")
        return

    _validate_batch(arr, path=path, tree=tree)
    _validate_shape(node, key, arr, path=path, tree=tree)
    node.append_data(key, arr)


def _validate_batch(arr: np.ndarray, *, path: tuple[str, ...], tree: TraceTree) -> None:
    """Validate or set the batch size for batched arrays."""

    if arr.ndim == 0:
        return

    full_path = "/".join(path)
    if full_path in tree.config.global_paths:
        return

    if tree.batch_size is None:
        tree.batch_size = int(arr.shape[0])
        return

    if int(arr.shape[0]) != tree.batch_size:
        _raise_or_event(tree, ValueError("Batch size mismatch at " f"{full_path}: expected {tree.batch_size}, got {arr.shape[0]}"))


def _validate_shape(node: TraceNode, key: str, arr: np.ndarray, *, path: tuple[str, ...], tree: TraceTree) -> None:
    """Validate shape and dtype stability for a dense key."""

    if key not in node._buffers or not node._buffers[key]:
        return

    prev = node._buffers[key][-1]
    if prev.shape != arr.shape or prev.dtype != arr.dtype:
        full_path = "/".join(path)
        _raise_or_event(tree, ValueError("Shape/dtype mismatch at " f"{full_path}: expected {prev.shape}/{prev.dtype}, " f"got {arr.shape}/{arr.dtype}"))


def _raise_or_event(tree: TraceTree, exc: Exception) -> None:
    """Raise in strict mode or swallow in lenient mode."""

    if tree.config.strict:
        raise exc


def _to_numeric_array(value: Any) -> Optional[np.ndarray]:
    """Convert a value to a numeric NumPy array if possible."""

    if value is None:
        return None

    if isinstance(value, Number):
        return np.asarray(value)

    if isinstance(value, np.ndarray):
        return value

    if hasattr(value, "detach") and callable(value.detach):
        try:
            return value.detach().cpu().numpy()
        except Exception:
            return None

    try:
        arr = np.asarray(value)
    except Exception:
        return None

    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        return None

    return arr


def _is_list_of_numeric_arrays(value: Any) -> bool:
    """Return True if value is a non-empty list/tuple of numeric arrays."""

    if not isinstance(value, (list, tuple)):
        return False
    if not value:
        return False
    return all(_to_numeric_array(item) is not None for item in value)


def _is_dataclass_instance(value: Any) -> bool:
    """Return True for dataclass instances (not classes)."""

    return is_dataclass(value) and not isinstance(value, type)


def _iter_path_parts(path: str) -> Iterable[str]:
    """Yield path parts for non-empty segments."""
    for part in path.split("/"):
        if part:
            yield part


def _split_path(path: str) -> tuple[str, str]:
    """Split a path into node path and final key."""
    parts = list(_iter_path_parts(path))
    if len(parts) < 2:
        raise ValueError("Path must include a node and data key")
    node_path = "/".join(parts[:-1])
    return node_path, parts[-1]


def _get_node(root: TraceNode, path: str) -> TraceNode:
    """Return the TraceNode for a slash-delimited path."""
    current = root
    for part in _iter_path_parts(path):
        if part not in current.children:
            raise ValueError(f"Trace path '{path}' is missing '{part}'")
        current = current.children[part]
    return current


def _resolve_dense_value(node: TraceNode, key: str, *, strict: bool) -> np.ndarray:
    """Resolve a dense value from a node by key."""
    has_child = key in node.children
    has_data = key in node.data
    if has_child and has_data and strict:
        raise ValueError(f"Dense key '{key}' conflicts with child node")
    if has_data:
        return node.data[key]
    if has_child:
        raise ValueError(f"No dense value at path ending '{key}'")
    raise ValueError(f"Dense key '{key}' missing at node")


def _export_dense_node(node: TraceNode, *, strict: bool) -> Any:
    """Export dense data for a node into a nested dict/list pytree."""
    _validate_child_data_conflicts(node, strict=strict)
    if _is_indexed_data(node.data) and not node.children:
        indices = _sorted_indices(node.data.keys())
        if strict and indices != list(range(len(indices))):
            raise ValueError("Indexed data keys missing sequential indices")
        return [node.data[str(idx)] for idx in indices]
    if _is_indexed_children(node.children) and not node.data:
        indices = _sorted_indices(node.children.keys())
        if strict and indices != list(range(len(indices))):
            raise ValueError("Indexed children missing sequential indices")
        return [_export_dense_child(node.children[str(idx)], strict=strict) for idx in indices]

    exported: dict[str, Any] = {key: value for key, value in node.data.items()}
    for name, child in node.children.items():
        if name in exported:
            if strict:
                raise ValueError(f"Child/data conflict at key '{name}'")
            continue
        exported[name] = _export_dense_child(child, strict=strict)
    return exported


def _export_dense_child(node: TraceNode, *, strict: bool) -> Any:
    """Export a child node."""
    return _export_dense_node(node, strict=strict)


def _export_meta_node(node: TraceNode, *, strict: bool) -> dict[str, Any]:
    """Export static metadata into a nested dict tree with __meta__ keys."""
    _validate_child_data_conflicts(node, strict=strict)
    exported: dict[str, Any] = {}
    if node.meta_static:
        exported["__meta__"] = dict(node.meta_static)
    for name, child in node.children.items():
        if name in node.data:
            if strict:
                raise ValueError(f"Child/data conflict at key '{name}'")
            continue
        exported[name] = _export_meta_node(child, strict=strict)
    return exported


def flatten_pytree(tree: Any, *, sep: str = "/") -> dict[str, np.ndarray]:
    """Flatten a nested dict/list pytree into a path map."""
    flattened: dict[str, np.ndarray] = {}

    def _walk(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "__meta__":
                    continue
                next_prefix = f"{prefix}{sep}{key}" if prefix else key
                _walk(child, next_prefix)
            return
        if isinstance(value, list):
            for idx, child in enumerate(value):
                next_prefix = f"{prefix}{sep}{idx}" if prefix else str(idx)
                _walk(child, next_prefix)
            return
        flattened[prefix] = value

    _walk(tree, "")
    return flattened


def _is_indexed_children(children: dict[str, TraceNode]) -> bool:
    """Return True if all children keys are numeric strings."""
    if not children:
        return False
    return all(key.isdigit() for key in children.keys())


def _is_indexed_data(data: dict[str, np.ndarray]) -> bool:
    """Return True if all data keys are numeric strings."""
    if not data:
        return False
    return all(key.isdigit() for key in data.keys())


def _sorted_indices(keys: Iterable[str]) -> list[int]:
    """Return numeric indices sorted ascending."""
    indices = []
    for key in keys:
        try:
            indices.append(int(key))
        except ValueError:
            continue
    return sorted(indices)


def _validate_child_data_conflicts(node: TraceNode, *, strict: bool) -> None:
    """Validate that no child name conflicts with dense data keys."""
    conflicts = set(node.children.keys()) & set(node.data.keys())
    if conflicts and strict:
        joined = ", ".join(sorted(conflicts))
        raise ValueError(f"Child/data conflict at keys: {joined}")


def _record_static_meta(node: TraceNode, key: str, value: Any, *, strict: bool) -> None:
    """Record static metadata, rejecting changes in strict mode."""
    if key in node.meta_static:
        if node.meta_static[key] != value and strict:
            raise ValueError(f"Metadata value changed for key '{key}'")
        return
    node.meta_static[key] = value


def iter_nodes(root: TraceNode) -> Iterator[TraceNode]:
    """Depth-first iteration over trace nodes."""

    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(list(node.children.values())))
