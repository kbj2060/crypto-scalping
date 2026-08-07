from __future__ import annotations

import threading
from functools import wraps
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar


P = ParamSpec("P")
R = TypeVar("R")

_registry_guard = threading.Lock()
_path_locks: dict[str, threading.RLock] = {}


def duckdb_path_lock(path: str | Path) -> threading.RLock:
    key = str(Path(path).expanduser().resolve())
    with _registry_guard:
        lock = _path_locks.get(key)
        if lock is None:
            lock = threading.RLock()
            _path_locks[key] = lock
        return lock


def serialized_duckdb_access(
    path_getter: Callable[P, str | Path],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Serialize connections to one DuckDB file across all in-process threads."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            path = path_getter(*args, **kwargs)
            with duckdb_path_lock(path):
                return function(*args, **kwargs)

        return wrapped

    return decorate
