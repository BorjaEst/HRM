"""Trace Tree implementation for batched simulation rollouts.

This module provides a clean, minimal implementation of the Trace Tree
Pattern with explicit handling of dense data and static metadata.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from numbers import Number
from typing import Any, Iterable, Iterator, Optional

import numpy as np


# =================================================================================================
@dataclass
class TraceConfig:
    """Configuration for trace construction and validation.

    Attributes:
        global_paths: Segment paths treated as global (no batch axis enforced).
        rebase_time_on_slice: Whether to rebase time to zero on slices.
    """

    global_paths: set[tuple[str, ...]] = field(default_factory=set)
    rebase_time_on_slice: bool = True


# =================================================================================================
@dataclass
class TraceNode:
    """Node in a trace tree with dense data and static metadata."""

    data: dict[str, np.ndarray] = field(default_factory=dict)
    meta_static: dict[str, Any] = field(default_factory=dict)
    children: dict[str, "TraceNode"] = field(default_factory=dict)
    _buffers: dict[str, list[np.ndarray]] = field(default_factory=dict, repr=False)
    _container_kind: Optional[str] = field(default=None, repr=False)
    _kinds_by_key: dict[str, str] = field(default_factory=dict, repr=False)
    _schema_keys: Optional[tuple[str, ...]] = field(default=None, repr=False)
    _list_len: Optional[int] = field(default=None, repr=False)

    def child(  # ---------------------------------------------------------------------------------
        self, name: str,
    ) -> TraceNode:  # fmt: skip
        """Return (or create) a child node by name."""
        if name not in self.children:
            self.children[name] = TraceNode()
        return self.children[name]

    def append_data(  # ---------------------------------------------------------------------------
        self, key: str, value: np.ndarray,
    ) -> None:  # fmt: skip
        """Append a dense value to the buffer for a given key."""
        self._buffers.setdefault(key, []).append(value)

    def finalize(  # ------------------------------------------------------------------------------
        self, *, length: int,
    ) -> None:  # fmt: skip
        """Stack buffered data into dense arrays and recurse into children."""
        for key, values in self._buffers.items():
            if len(values) != length:
                err_msg = f"Incomplete data buffer for key {key!r}: expected {length}, got {len(values)}"
                raise ValueError(err_msg)
            self.data[key] = np.stack(values, axis=0)
        self._buffers.clear()
        for child in self.children.values():
            child.finalize(length=length)

    def slice_time(  # ----------------------------------------------------------------------------
        self, t0: int, t1: int, *, rebase: bool,
    ) -> "TraceNode":  # fmt: skip
        """Return a time-sliced copy of this node."""
        sliced = TraceNode()
        for key, arr in self.data.items():
            sliced.data[key] = arr[t0:t1]
        sliced.meta_static = dict(self.meta_static)
        sliced._container_kind = self._container_kind
        sliced._kinds_by_key = dict(self._kinds_by_key)
        sliced._schema_keys = self._schema_keys
        sliced._list_len = self._list_len
        for name, child in self.children.items():
            sliced.children[name] = child.slice_time(t0, t1, rebase=rebase)
        return sliced


# =================================================================================================
@dataclass
class TraceTree:
    """Trace tree builder and container for a rollout."""

    config: TraceConfig = field(default_factory=TraceConfig)
    root: TraceNode = field(default_factory=TraceNode)
    batch_size: Optional[int] = None
    length: int = 0
    leaf_signatures: dict[tuple[str, ...], tuple[tuple[int, ...], np.dtype]] = field(
        default_factory=dict,
    )

    def append(  # --------------------------------------------------------------------------------
        self, state: Any,
    ) -> None:  # fmt: skip
        """Append a state snapshot into the trace tree."""
        _append_node(node=self.root, value=state, path=(), tree=self)
        self.length += 1

    def finalize(  # ------------------------------------------------------------------------------
        self,
    ) -> None:  # fmt: skip
        """Finalize the trace by stacking buffered data into arrays."""
        self.root.finalize(length=self.length)

    def slice_time(  # ----------------------------------------------------------------------------
        self, t0: int, t1: int,
    ) -> "TraceTree":  # fmt: skip
        """Return a time-sliced copy of the trace."""
        t0 = max(0, t0)
        t1 = min(self.length, t1)
        sliced = TraceTree(config=self.config)
        sliced.root = self.root.slice_time(t0, t1, rebase=self.config.rebase_time_on_slice)
        sliced.length = max(0, t1 - t0)
        sliced.batch_size = self.batch_size
        sliced.leaf_signatures = dict(self.leaf_signatures)
        return sliced

    def node(  # ----------------------------------------------------------------------------------
        self, path: str,
    ) -> TraceNode:  # fmt: skip
        """Return the TraceNode at the given slash-delimited path."""
        return _get_node(self.root, path)

    def get(  # -----------------------------------------------------------------------------------
        self, path: str,
    ) -> np.ndarray:  # fmt: skip
        """Return a dense array at a slash-delimited path.

        Args:
            path: Slash-delimited path to a dense array or indexed child.

        Returns:
            Dense NumPy array stored at the requested path.
        """
        node_path, key = _split_path(path)
        node = _get_node(self.root, node_path)
        return _resolve_dense_value(node, key)

    def export_dense_tree(  # ---------------------------------------------------------------------
        self,
    ) -> Any:  # fmt: skip
        """Export dense trace data as a nested dict/list pytree."""
        return _export_dense_node(self.root)

    def export_meta_tree(  # ----------------------------------------------------------------------
        self,
    ) -> Any:  # fmt: skip
        """Export static metadata as a nested dict/list tree."""
        return _export_meta_node(self.root)

    def export(  # --------------------------------------------------------------------------------
        self, *, flatten: bool = False, sep: str = "/",
    ) -> Any:  # fmt: skip
        """Export dense trace data, optionally flattening to a path map."""
        dense = self.export_dense_tree()
        if not flatten:
            return dense
        return flatten_pytree(dense, sep=sep)

    def get_meta(  # ------------------------------------------------------------------------------
        self,
    ) -> dict[str, Any]:  # fmt: skip
        """Return root metadata dictionary, if available."""
        meta = self.root.meta_static.get("meta")
        return dict(meta) if isinstance(meta, dict) else {}

    def get_environments(  # ----------------------------------------------------------------------
        self,
    ) -> list[Any]:  # fmt: skip
        """Return environments stored in the trace metadata."""
        envs = self.root.meta_static.get("environments")
        return list(envs) if envs is not None else []

    def get_visited(  # ---------------------------------------------------------------------------
        self,
    ) -> Any:  # fmt: skip
        """Return visited masks stored in the trace metadata."""
        return self.root.meta_static.get("visited")

    def get_world(  # -----------------------------------------------------------------------------
        self, env_idx: int,
    ) -> Any:  # fmt: skip
        """Return the World object for a selected environment index."""
        envs = self.get_environments()
        if not envs:
            raise ValueError("TraceTree has no environments")
        if not (0 <= env_idx < len(envs)):
            raise IndexError(f"env_idx {env_idx} out of range [0, {len(envs)})")
        return envs[env_idx]

    def n_freq(  # --------------------------------------------------------------------------------
        self, base_path: str,
    ) -> int:  # fmt: skip
        """Return number of indexed children at a multiscale path."""
        node = self.node(base_path)
        if node._container_kind == "list" and node._list_len is not None:
            return node._list_len
        if node.children and all(key.isdigit() for key in node.children.keys()):
            return len(node.children)
        if node.data and all(key.isdigit() for key in node.data.keys()):
            return len(node.data)
        return 0

    def validate_env_idx(  # ----------------------------------------------------------------------
        self, env_idx: int,
    ) -> int:  # fmt: skip
        """Validate and return environment index."""
        batch_size = int(self.batch_size or 0)
        if not (0 <= env_idx < batch_size):
            raise IndexError(f"env_idx {env_idx} out of range [0, {batch_size})")
        return env_idx

    def validate_freq_idx(  # ---------------------------------------------------------------------
        self, base_path: str, freq_idx: int,
    ) -> int:  # fmt: skip
        """Validate and return frequency index."""
        n_freq = self.n_freq(base_path)
        if not (0 <= freq_idx < n_freq):
            raise IndexError(f"freq_idx {freq_idx} out of range [0, {n_freq})")
        return freq_idx


# =================================================================================================
def _append_node(  # -----------------------------------------------------------------------------
    *, node: TraceNode, value: Any, path: tuple[str, ...], tree: TraceTree,
) -> None:  # fmt: skip
    """Append a container node (mapping/list) into the trace tree."""
    if isinstance(value, MappingABC):
        if node._container_kind is None:
            node._container_kind = "dict"
        elif node._container_kind != "dict":
            err_msg = f"Container kind changed at {_path_str(path)}: {node._container_kind} -> dict"
            raise ValueError(err_msg)
        keys = tuple(sorted(str(k) for k in value.keys()))
        if node._schema_keys is None:
            node._schema_keys = keys
        elif node._schema_keys != keys:
            err_msg = f"Schema keys changed at {_path_str(path)}: {node._schema_keys} -> {keys}"
            raise ValueError(err_msg)
        for key, child in value.items():
            _append_entry(node, str(key), child, path + (str(key),), tree)
        return
    if isinstance(value, (list, tuple)):
        if node._container_kind is None:
            node._container_kind = "list"
        elif node._container_kind != "list":
            err_msg = f"Container kind changed at {_path_str(path)}: {node._container_kind} -> list"
            raise ValueError(err_msg)
        length = len(value)
        if node._list_len is None:
            node._list_len = length
        elif node._list_len != length:
            err_msg = f"List length changed at {_path_str(path)}: {node._list_len} -> {length}"
            raise ValueError(err_msg)
        for idx, child in enumerate(value):
            _append_entry(node, str(idx), child, path + (str(idx),), tree)
        return
    raise TypeError(
        "State must be a mapping or list/tuple container. "
        f"Got {type(value)!r} at path {_path_str(path)}."
    )


# =================================================================================================
def _append_entry(  # ----------------------------------------------------------------------------
    node: TraceNode, key: str, value: Any, path: tuple[str, ...], tree: TraceTree,
) -> None:  # fmt: skip
    """Append a single key/value under a container node."""
    is_mapping = isinstance(value, MappingABC)
    is_list = isinstance(value, (list, tuple))
    arr = None if (is_mapping or is_list) else _to_numeric_array(value)
    if is_mapping or is_list:
        entry_kind = "container"
    elif arr is not None:
        entry_kind = "leaf_numeric"
    else:
        entry_kind = "leaf_meta"

    prev_kind = node._kinds_by_key.get(key)
    if prev_kind is None:
        node._kinds_by_key[key] = entry_kind
    elif prev_kind != entry_kind:
        raise ValueError(f"Key kind changed at {_path_str(path)}: {prev_kind} -> {entry_kind}")

    if is_mapping or is_list:
        child = node.child(key)
        _append_node(node=child, value=value, path=path, tree=tree)
        return
    if arr is not None:
        _append_leaf_numeric(node=node, key=key, arr=arr, path=path, tree=tree)
        return
    _record_static_meta(node, key, value)


# =================================================================================================
def _append_leaf_numeric(  # ---------------------------------------------------------------------
    *, node: TraceNode, key: str, arr: np.ndarray, path: tuple[str, ...], tree: TraceTree,
) -> None:  # fmt: skip
    """Append a numeric leaf with batch and signature invariants."""
    if arr.dtype == object:
        raise ValueError(f"Irregular value for key {key!r} at {_path_str(path)}")
    is_global = _is_global_path(path, global_paths=tree.config.global_paths)
    if not is_global:
        _enforce_batch_axis(arr, path=path, tree=tree)
    _enforce_signature(arr, path=path, tree=tree, is_global=is_global)
    node.append_data(key, arr)


# =================================================================================================
def _enforce_batch_axis(  # ----------------------------------------------------------------------
    arr: np.ndarray, *, path: tuple[str, ...], tree: TraceTree,
) -> None:  # fmt: skip
    """Enforce batch-axis stability for non-global numeric leaves."""
    if arr.ndim == 0:
        return
    if tree.batch_size is None:
        tree.batch_size = int(arr.shape[0])
        return
    if int(arr.shape[0]) != tree.batch_size:
        raise ValueError(
            f"Batch size mismatch at {_path_str(path)}: "
            f"expected {tree.batch_size}, got {arr.shape[0]}"
        )


# =================================================================================================
def _enforce_signature(  # -----------------------------------------------------------------------
    arr: np.ndarray, *, path: tuple[str, ...], tree: TraceTree, is_global: bool = False,
) -> None:  # fmt: skip
    """Enforce first-seen shape/dtype signature for numeric leaves."""
    if is_global:
        shape_sig = tuple(arr.shape)
    elif arr.ndim == 0:
        shape_sig = ()
    else:
        shape_sig = tuple(arr.shape[1:])
    sig = (shape_sig, arr.dtype)
    prev = tree.leaf_signatures.get(path)
    if prev is None:
        tree.leaf_signatures[path] = sig
        return
    if prev != sig:
        raise ValueError(f"Shape/dtype mismatch at {_path_str(path)}: expected {prev}, got {sig}")


# =================================================================================================
def _is_global_path(  # --------------------------------------------------------------------------
    path: tuple[str, ...], *, global_paths: set[tuple[str, ...]],
) -> bool:  # fmt: skip
    """Return True if any prefix of path is marked as global."""
    if not global_paths:
        return False
    for i in range(1, len(path) + 1):
        if path[:i] in global_paths:
            return True
    return False


# =================================================================================================
def _to_numeric_array(  # -------------------------------------------------------------------------
    value: Any,
) -> Optional[np.ndarray]:  # fmt: skip
    """Convert a value to a numeric NumPy array if possible."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return None
    if isinstance(value, Number):
        return np.asarray(value)
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and callable(value.detach):
        try:
            return value.detach().cpu().numpy()  # type: ignore
        except Exception:
            return None
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        return None
    return arr


# =================================================================================================
def _path_str(  # ---------------------------------------------------------------------------------
    path: tuple[str, ...],
) -> str:  # fmt: skip
    """Return a readable path string for error messages."""
    return "/".join(path) or "<root>"


# =================================================================================================
def _iter_path_parts(  # --------------------------------------------------------------------------
    path: str,
) -> Iterable[str]:  # fmt: skip
    """Yield path parts for non-empty segments."""
    for part in path.split("/"):
        if part:
            yield part


# =================================================================================================
def _split_path(  # -------------------------------------------------------------------------------
    path: str,
) -> tuple[str, str]:  # fmt: skip
    """Split a path into node path and final key."""
    parts = list(_iter_path_parts(path))
    if len(parts) < 2:
        raise ValueError("Path must include a node and data key")
    node_path = "/".join(parts[:-1])
    return node_path, parts[-1]


# =================================================================================================
def _get_node(  # ---------------------------------------------------------------------------------
    root: TraceNode, path: str,
) -> TraceNode:  # fmt: skip
    """Return the TraceNode for a slash-delimited path."""
    current = root
    for part in _iter_path_parts(path):
        if part not in current.children:
            raise ValueError(f"Trace path '{path}' is missing '{part}'")
        current = current.children[part]
    return current


# =================================================================================================
def _resolve_dense_value(  # ----------------------------------------------------------------------
    node: TraceNode, key: str,
) -> np.ndarray:  # fmt: skip
    """Resolve a dense value from a node by key."""
    has_child = key in node.children
    has_data = key in node.data
    if has_child and has_data:
        raise ValueError(f"Dense key '{key}' conflicts with child node")
    if has_data:
        return node.data[key]
    if has_child:
        raise ValueError(f"No dense value at path ending '{key}'")
    raise ValueError(f"Dense key '{key}' missing at node")


# =================================================================================================
def _export_dense_node(  # ------------------------------------------------------------------------
    node: TraceNode,
) -> Any:  # fmt: skip
    """Export dense data for a node into a nested dict/list pytree."""
    if node._container_kind == "list":
        if node._list_len is None:
            raise ValueError("List container missing length metadata")
        exported_list: list[Any] = []
        for idx in range(node._list_len):
            key = str(idx)
            if key in node.data:
                exported_list.append(node.data[key])
            elif key in node.children:
                exported_list.append(_export_dense_node(node.children[key]))
            else:
                exported_list.append(None)
        return exported_list

    exported: dict[str, Any] = {key: value for key, value in node.data.items()}
    for name, child in node.children.items():
        if name in exported:
            raise ValueError(f"Child/data conflict at key '{name}'")
        exported[name] = _export_dense_node(child)
    return exported


# =================================================================================================
def _export_meta_node(  # -------------------------------------------------------------------------
    node: TraceNode,
) -> Any:  # fmt: skip
    """Export static metadata into a nested dict/list tree."""
    if node._container_kind == "list":
        if node._list_len is None:
            raise ValueError("List container missing length metadata")
        exported_list: list[Any] = []
        for idx in range(node._list_len):
            key = str(idx)
            if key in node.children:
                exported_list.append(_export_meta_node(node.children[key]))
            elif key in node.meta_static:
                exported_list.append(node.meta_static[key])
            else:
                exported_list.append(None)
        return exported_list

    exported: dict[str, Any] = {}
    keys = set(node.meta_static.keys()) | set(node.children.keys()) | set(node.data.keys())
    for name in sorted(keys):
        if name in node.children:
            exported[name] = _export_meta_node(node.children[name])
            continue
        if name in node.meta_static:
            exported[name] = node.meta_static[name]
            continue
        exported[name] = None
    return exported


# =================================================================================================
def flatten_pytree(  # ----------------------------------------------------------------------------
    tree: Any, *, sep: str = "/",
) -> dict[str, np.ndarray]:  # fmt: skip
    """Flatten a nested dict/list pytree into a path map."""
    flattened: dict[str, np.ndarray] = {}

    def _walk(value: Any, prefix: str) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            for key, child in value.items():
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


def _record_static_meta(  # -----------------------------------------------------------------------
    node: TraceNode, key: str, value: Any,
) -> None:  # fmt: skip
    """Record static metadata (first-write wins)."""
    if key in node.meta_static:
        return
    node.meta_static[key] = value


# =================================================================================================
def iter_nodes(  # --------------------------------------------------------------------------------
    root: TraceNode,
) -> Iterator[TraceNode]:  # fmt: skip
    """Depth-first iteration over trace nodes."""
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(list(node.children.values())))
