"""ops_watchdog -- Pi 수집 노드 감시(2026-10-12): 원격 프로브(mtime) → Check 변환, 노드 불통 = BLOCKED."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "ops"))
import ops_watchdog as w  # noqa: E402
import remote_fresh_probe as p  # noqa: E402


def _node(now: float) -> Path:
    root = Path(tempfile.mkdtemp())
    (root / "data/live/orderflow/bt/BTCUSDT").mkdir(parents=True)
    (root / "data/live/orderflow/bt/HYPEUSDT").mkdir(parents=True)
    fresh = root / "data/live/orderflow/bt/BTCUSDT/2026-10-12T13.bt"
    stale = root / "data/live/orderflow/bt/HYPEUSDT/2026-10-12T10.bt"
    fresh.write_text("x"), stale.write_text("x")
    os.utime(stale, (now - 20 * 60, now - 20 * 60))                     # 20분 묵음
    db = root / "data/live/hl_ctx_btc.duckdb"
    db.write_text("x"), os.utime(db, (now - 3600, now - 3600))
    Path(str(db) + ".wal").write_text("x")                               # 본체는 1시간 전, .wal 은 지금 = 살아 있다
    d = lambda name, path, kind="dir": {"name": name, "kind": kind, "path": path, "glob": "*", "table": "t", "ts": "ts",  # noqa: E731
                                        "warn": 5, "critical": 15}
    (root / "data/live/multicoin_collectors.json").write_text(json.dumps({"collectors": {
        "a": {"started_at": now - 86400, "fresh": [d("bt_btc", "data/live/orderflow/bt/BTCUSDT"),
                                                   d("bt_hype", "data/live/orderflow/bt/HYPEUSDT"),
                                                   d("hl_ctx_btc", "data/live/hl_ctx_btc.duckdb", "duckdb"),
                                                   d("gone", "data/live/orderflow/bt/XRPUSDT")]},
        "b": {"started_at": now - 60, "grace_minutes": 15, "fresh": [d("new", "data/live/nothing_yet")]},
    }}))
    return root


def test_probe_and_checks():
    now = time.time()
    rows = p.probe(_node(now), now)
    by = {r["name"]: r for r in rows}
    assert by["bt_btc"]["age_minutes"] < 1 and by["bt_hype"]["age_minutes"] >= 19
    assert by["hl_ctx_btc"]["age_minutes"] < 1                           # .wal mtime 이 본체보다 우선(최신)
    assert by["gone"]["age_minutes"] is None and by["new"]["grace"] is True
    st = {c.component: c.status for c in w.pi_checks_from_rows(rows, None)}
    assert st == {"pi_reachable": "OK", "pi_bt_btc": "OK", "pi_bt_hype": "CRITICAL",
                  "pi_hl_ctx_btc": "OK", "pi_gone": "BLOCKED"}, st          # 유예 중인 "new" 는 빠진다


def test_unreachable_node_is_blocked():
    cs = w.pi_checks_from_rows(None, "ssh rc=255: Connection timed out")
    assert [(c.component, c.status) for c in cs] == [("pi_reachable", "BLOCKED")]


def test_probe_matches_local_multicoin_rules():
    """같은 매니페스트를 서버 쪽 check_multicoin_collectors(dir 항목)로 판정해도 같은 상태여야 한다."""
    now = time.time()
    root = _node(now)
    w.ROOT, w.MULTICOIN_MANIFEST = root, root / "data/live/multicoin_collectors.json"
    local = {c.component.removeprefix("multicoin_"): c.status for c in w.check_multicoin_collectors()
             if not c.component.endswith("hl_ctx_btc")}                   # duckdb 항목은 로컬이 DB 를 연다 -- 비교 밖
    remote = {c.component.removeprefix("pi_"): c.status for c in w.pi_checks_from_rows(p.probe(root, now), None)
              if c.component not in ("pi_reachable", "pi_hl_ctx_btc")}
    assert local == remote, (local, remote)
