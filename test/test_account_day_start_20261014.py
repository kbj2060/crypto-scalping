"""계좌 카드 «오늘 손익» 기준(2026-10-14): KST 날이 바뀐 뒤 첫 조회만 기록 · 같은 날엔 안 덮음 · 실패 응답은 None.
python -m pytest -q test/test_account_day_start_20261014.py"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.server import account_day_start  # noqa: E402

KST = timezone(timedelta(hours=9))
ok = lambda w, u: {"ok": True, "balance": {"wallet": w, "unrealized": u}}  # noqa: E731


def test_first_call_of_day_records_then_keeps(tmp_path):
    p = tmp_path / "ds.json"
    t1 = datetime(2026, 10, 14, 0, 3, tzinfo=KST).timestamp()
    a = account_day_start(ok(1000, 50), now=t1, path=p)
    assert a["date"] == "2026-10-14" and a["equity"] == 1050 and a["late_s"] == 180
    b = account_day_start(ok(900, 0), now=t1 + 3600, path=p)       # 같은 날 = 안 덮음
    assert b["equity"] == 1050
    c = account_day_start(ok(900, 10), now=t1 + 86400, path=p)     # 다음 날 = 새로
    assert c["date"] == "2026-10-15" and c["equity"] == 910


def test_failed_payload_is_none(tmp_path):
    p = tmp_path / "ds.json"
    assert account_day_start({"ok": False, "balance": {"wallet": 1}}, now=0, path=p) is None
    assert account_day_start({"ok": True}, now=0, path=p) is None
    assert not p.exists()
