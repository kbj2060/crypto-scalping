"""⑤ 예측 줄 «15분 폭»·«앞 1시간» + 손절·익절 꼬리표 «15분 N%» (2026-11-15, 시안 C · 배포 대기 A10).

값은 서버 크론 scripts/live_risk_radar_20261115.py 가 매분 쓰는 data/live/risk_radar_latest.json 을 그대로 읽는다(화면·서버는 계산 안 함).
꼬리표 확률 표(touch_k → coins.x.touch)는 동결 계수 dashboard/risk_radar_params_20261116.json 에서 붙인다.
"""
from __future__ import annotations

import calendar
import re
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "data/live/risk_radar_latest.json"
PARAMS = ROOT / "dashboard/risk_radar_params_20261116.json"   # 워커와 같은 계수(B2)
STALE_S = 180                                 # 워커가 3분 넘게 못 쓰면 화면은 «지연»


def risk_radar_payload(latest: Path = LATEST, params: Path = PARAMS, now: float | None = None) -> dict:
    if not latest.exists():
        return {"ok": False, "reason": "기록 대기"}
    d = json.loads(latest.read_text())
    nm = str(d.get("params") or "")                                   # 워커가 쓴 계수 파일 = 꼬리표 표도 같은 파일(되돌릴 때 두 파일이 어긋나지 않게 · 검증 11-16)
    if params == PARAMS and re.fullmatch(r"risk_radar_params_\d{8}[a-z]?\.json", nm) and (ROOT / "dashboard" / nm).exists():
        params = ROOT / "dashboard" / nm
    P = json.loads(params.read_text())
    age = (now or time.time()) - calendar.timegm(time.strptime(d["updated"], "%Y-%m-%dT%H:%M:%SZ"))
    coins = {c.lower(): {**o, "touch": P["coins"].get(c, {}).get("touch")} for c, o in d.get("coins", {}).items()}
    return {"ok": True, "updated": d["updated"], "age_s": round(age, 1), "stale": age > STALE_S, "touch_k": P.get("touch_k"), "coins": coins}


if __name__ == "__main__":   # 자체점검: 파일 없음 → 대기 · 신선/지연 판정 · 코인 키 소문자 · 표 붙음
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        lt, pr = Path(td) / "l.json", Path(td) / "p.json"
        assert not risk_radar_payload(lt, pr)["ok"]
        pr.write_text(json.dumps({"touch_k": [0.25, 0.5], "coins": {"ETH": {"touch": [0.7, 0.5]}}}))
        lt.write_text(json.dumps({"updated": "2026-11-15T00:00:00Z", "coins": {"ETH": {"sigma15_pct": 0.3}}}))
        t0 = calendar.timegm((2026, 11, 15, 0, 0, 0))
        a, b = risk_radar_payload(lt, pr, now=t0 + 30), risk_radar_payload(lt, pr, now=t0 + 400)
        assert a["ok"] and not a["stale"] and abs(a["age_s"] - 30) < 1e-6 and b["stale"] and a["coins"]["eth"]["touch"] == [0.7, 0.5]
    print("selftest OK")
