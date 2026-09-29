"""2026-09-30: 추세 전환 워커 _fetch 는 형성 중인 봉(종료 시각이 미래)을 버린다 -- 마지막 행 = 방금 마감된 봉."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import live_eth_breakout_detector_20260911 as M  # noqa: E402


def test_fetch_drops_forming_bar(monkeypatch):
    now = int(time.time() * 1000) // 300000 * 300000          # 지금 열린 5분봉 시작
    row = lambda ot: [ot, "1", "1", "1", "1", "1", ot + 299999, "10", "5", "1", "5", "0"]
    rows = [row(now - 300000 * k) for k in range(3, -1, -1)]    # 마지막 = 지금 형성 중인 봉

    class R:
        def raise_for_status(self): pass
        def json(self): return rows
    monkeypatch.setattr(M.requests, "get", lambda *a, **k: R())
    d = M._fetch(limit=4)
    assert len(d) == 3 and int(d["ot"].iloc[-1]) == now - 300000, d[["ot", "ct"]]
