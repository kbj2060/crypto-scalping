"""worker_payload 의 계약 고정. 2026-09-12 통합(4종 -> 1개) 때 실제로 틀렸던 세 지점을 잡는다.

프레임워크 없이 `python3 test/test_worker_payload_contract_20260912.py` 로 돈다.
"""
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "dashboard" / "server.py").read_text(encoding="utf-8")
chunk = src[src.index("def worker_payload("):src.index("def extreme_detector_payload()")]

STATE: dict = {}
now = datetime.now(timezone.utc)
def _age_min(raw):
    if not raw:
        return None
    dt = datetime.fromisoformat(str(raw))
    return round((now - dt).total_seconds() / 60.0, 1)

ns = {"Path": Path, "Any": object, "load_json": lambda p: STATE.get(p), "_age_min": _age_min}
exec(compile(chunk, "worker_payload", "exec"), ns)
wp = ns["worker_payload"]

EXTRAS = {"grade": None, "history": []}
fresh = (now - timedelta(minutes=1)).isoformat()
stale = (now - timedelta(minutes=999)).isoformat()

# 1) stale 은 워커가 쓴 값을 보여준다 -- extra_missing 으로 덮으면 history 가 빈 배열이 된다.
STATE["p"] = {"ok": True, "updated_utc": stale, "history": [1, 2]}
out = wp("p", 15.0, extra_missing=EXTRAS)
assert out["error"] == "worker_stale" and out["history"] == [1, 2], out

# 2) stamp_available=False 면 available 을 찍지 않는다(워커 값을 통과).
STATE["p"] = {"ok": True, "updated_utc": fresh, "available": False}
assert wp("p", 15.0)["available"] is False
assert wp("p", 15.0, stamp_available=True)["available"] is True

# 3) bare_missing 계열의 ok=False 는 워커 상태를 얹지 않는다.
STATE["p"] = {"ok": False, "generated_at": fresh, "margin": 0.3}
bare = wp("p", 15.0, ts_field="generated_at", require_ok=True, bare_missing=True)
assert "margin" not in bare and bare["error"] == "worker_state_missing", bare
rich = wp("p", 15.0, ts_field="generated_at", require_ok=True)
assert rich["margin"] == 0.3 and rich["error"] == "worker_fetch_failed", rich

# 4) 상태 파일이 없으면 결측 + extras.
STATE.pop("p", None)
assert wp("p", 15.0, extra_missing=EXTRAS) == {
    "available": False, "error": "worker_state_missing", "tone": "neutral",
    "subText": "데이터 없음", **EXTRAS}

print("worker_payload 계약 4건 통과")
