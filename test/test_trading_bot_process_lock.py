from __future__ import annotations

import os

import pytest

pytest.importorskip("fcntl")

from trading_bot_modules.process_lock import acquire_trading_bot_process_lock


def test_process_lock_is_exclusive_and_records_owner(tmp_path):
    journal = tmp_path / "trade_journal.jsonl"
    lock_path = tmp_path / "bot.lock"

    first = acquire_trading_bot_process_lock(journal_path=journal, lock_path=lock_path)
    try:
        assert lock_path.read_text(encoding="utf-8") == str(os.getpid())
        with pytest.raises(RuntimeError, match="process lock already held"):
            acquire_trading_bot_process_lock(journal_path=journal, lock_path=lock_path)
    finally:
        first.close()


def test_process_lock_defaults_next_to_journal(tmp_path):
    journal = tmp_path / "trade_journal.jsonl"
    handle = acquire_trading_bot_process_lock(journal_path=journal)
    try:
        assert journal.with_suffix(".lock").exists()
    finally:
        handle.close()
