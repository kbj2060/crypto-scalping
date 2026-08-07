from __future__ import annotations

import json

import pytest

from trading_bot_modules import live_io


def test_atomic_write_replaces_complete_json(tmp_path):
    target = tmp_path / "state.json"
    target.write_text('{"revision": 1}', encoding="utf-8")

    live_io._atomic_write_json(str(target), {"revision": 2, "position": "LONG"})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "revision": 2,
        "position": "LONG",
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_preserves_previous_file_when_replace_fails(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    target.write_text('{"revision": 1}', encoding="utf-8")

    def fail_replace(_source, _target):
        raise OSError("injected replace failure")

    monkeypatch.setattr(live_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        live_io._atomic_write_json(str(target), {"revision": 2})

    assert json.loads(target.read_text(encoding="utf-8")) == {"revision": 1}
    assert not list(tmp_path.glob("*.tmp"))
