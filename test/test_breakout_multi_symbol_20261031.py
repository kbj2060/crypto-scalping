"""2026-10-31 추세 전환 워커: 한 프로세스가 심볼 목록을 차례로 -- 상태 파일은 코인별 옛 이름, 심볼은 _fetch 까지 전달된다."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import live_eth_breakout_detector_20260911 as M  # noqa: E402
import live_eth_breakout_detector_worker_20260911 as W  # noqa: E402


def test_symbol_reaches_fetch_and_state_files_are_per_coin(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(M, "_fetch", lambda limit=0, symbol=None: seen.append(symbol) or "d")
    monkeypatch.setattr(M, "compute_signals", lambda d: {"timestamp": "2026-10-31T00:00:00", "close": 1.0})
    monkeypatch.setattr(W, "get_signals", M.get_signals)
    monkeypatch.setattr(W, "ROOT", tmp_path)
    for sym in ("ETHUSDT", "SOLUSDT"):
        W.cycle(sym)
    assert seen == ["ETHUSDT", "SOLUSDT"]
    assert json.loads((tmp_path / "data/live/eth_breakout_detector_state.json").read_text())["ok"]
    assert (tmp_path / "data/live/sol_breakout_detector_state.json").exists()
