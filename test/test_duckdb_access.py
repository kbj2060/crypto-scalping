from __future__ import annotations

import concurrent.futures
import threading
import time

from trading_bot_modules.duckdb_access import serialized_duckdb_access


def test_same_duckdb_path_is_serialized_across_threads(tmp_path):
    active = 0
    max_active = 0
    guard = threading.Lock()

    @serialized_duckdb_access(lambda path, _value: path)
    def write(path, value):
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.002)
        with guard:
            active -= 1
        return value

    path = tmp_path / "microstructure.duckdb"
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda value: write(path, value), range(30)))

    assert results == list(range(30))
    assert max_active == 1
