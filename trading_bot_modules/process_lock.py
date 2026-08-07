from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import TextIO


def acquire_trading_bot_process_lock(
    *,
    journal_path: str | Path,
    lock_path: str | Path | None = None,
) -> TextIO:
    """Acquire the single-process lock and keep it held through the returned handle."""
    journal = Path(journal_path).resolve()
    target = Path(lock_path).resolve() if lock_path else journal.with_suffix(".lock")
    target.parent.mkdir(parents=True, exist_ok=True)

    lock_fh = target.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_fh.seek(0)
        owner = lock_fh.read().strip() or "unknown"
        lock_fh.close()
        raise RuntimeError(f"trading_bot process lock already held: lock={target} owner={owner}")

    lock_fh.seek(0)
    lock_fh.truncate()
    lock_fh.write(str(os.getpid()))
    lock_fh.flush()
    return lock_fh
