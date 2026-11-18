"""ETH «레벨 5분» 기울기(지지·저항 접근·돌파 뒤 5분 안 어느 쪽 0.1σ 먼저 · 배포 대기 A19)의 최신값 (2026-11-18).

차트 표식과 ⑤ 예측 줄 «레벨 5분» 칸이 읽는다. 값은 요청 때 계산하지 않는다 -- 상시 워커(scripts/live_sr_lean_20261118.py)가
data/live/sr_lean/ethusdt.jsonl 에 쌓은 줄의 꼬리 256KB(약 12일)와 생존 표시 파일만 읽는다(stale = 워커가 마지막으로 닫은 체결 초가 45초 넘게 묵음).
화면에는 점수 양끝 10%(top = 사건 쪽 먼저 · bot = 반대쪽 먼저)만 보낸다 -- 가운데 80% 는 동전이라 건수만.
연구: docs/experiments/sr_ml_rl_20261118.md (2023 확인 AUC .531 · 상위 989/1,752 = 56.4% 대 하위 1,269/2,804 = 45.3%). 표시 전용 · 매매 근거 아님.

    python dashboard/sr_lean.py --selftest
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "data" / "live" / "sr_lean"
TAIL_BYTES, SHOW_S, STALE_S = 262144, 48 * 3600, 45


def sr_lean_payload(d: Path = DIR, now: float | None = None) -> dict:
    """{"ok", "now", "stale", "warm_until", "pct": {top, bot}, "ev": [{a, d, side, kind, look, L, P, grp, y, hit_s}], "mid48", "rec": {n, hit, recent}}."""
    now = time.time() if now is None else now
    try:
        st = json.loads((d / "ethusdt_state.json").read_text())
    except (OSError, ValueError):
        return {"ok": False, "reason": "워커 기록 없음"}
    ev: dict = {}; late = 0
    log = d / "ethusdt.jsonl"
    if log.exists():
        with log.open("rb") as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - TAIL_BYTES))
            lines = f.read().decode("utf-8", "replace").splitlines()[1 if size > TAIL_BYTES else 0:]   # 잘린 첫 줄은 버린다
        for ln in lines:
            try:
                r = json.loads(ln)
                if r["k"] == "ev" and r.get("lag", 0) > 0:
                    late += 1                              # 사건 초가 «알 수 있던 시각»보다 앞선 것(체결 없는 초에 직전가로 문턱을 넘은 접근) -- 늦게 나온 판정은 안 그린다
                elif r["k"] == "ev":
                    ev[r["id"]] = {k: r[k] for k in ("a", "d", "side", "kind", "look", "L", "P", "grp")} | {"y": None, "hit_s": None}
                elif r["k"] == "res" and r["id"] in ev:
                    ev[r["id"]].update(y=r["y"], hit_s=r["hit_s"])
            except (ValueError, KeyError, TypeError):
                continue                                   # 깨진 줄은 건너뛴다
    seen, shown = set(), []
    for e in sorted(ev.values(), key=lambda e: (e["a"], e["look"])):   # 같은 초·같은 쪽·같은 판정은 하나로(가까운 거리 것)
        if e["grp"] != "mid" and (e["a"], e["side"], e["grp"]) not in seen:
            seen.add((e["a"], e["side"], e["grp"])); shown.append(e)
    done = [e for e in shown if e["y"] is not None]
    ok = [int((e["y"] == 1) == (e["grp"] == "top")) for e in done]
    rate = st.get("rate") or {"top": [989, 1752], "bot": [1269, 2804]}
    return {"ok": True, "now": now, "stale": now - (st.get("done") or 0) > STALE_S, "warm_until": st.get("warm_until"),
            "pct": {"top": round(100 * rate["top"][0] / rate["top"][1]), "bot": round(100 * (1 - rate["bot"][0] / rate["bot"][1]))},
            "ev": [e for e in shown if e["a"] >= now - SHOW_S], "late": late, "mid48": sum(1 for e in ev.values() if e["grp"] == "mid" and e["a"] >= now - SHOW_S),
            "rec": {"n": len(ok), "hit": sum(ok), "recent": [[e["d"] * 1000, o] for e, o in zip(done[-30:], ok[-30:])]}}


def _selftest() -> None:
    d = Path(tempfile.mkdtemp()); now = 2_000_000_000
    assert sr_lean_payload(d, now)["ok"] is False
    (d / "ethusdt_state.json").write_text(json.dumps({"ts": now - 5, "done": now - 2, "warm_until": now - 100, "rate": {"top": [989, 1752], "bot": [1269, 2804]}}))
    assert sr_lean_payload(d, now) == {"ok": True, "now": now, "stale": False, "warm_until": now - 100, "pct": {"top": 56, "bot": 55}, "ev": [], "late": 0, "mid48": 0, "rec": {"n": 0, "hit": 0, "recent": []}}
    ev = lambda a, look, side, grp: dict(k="ev", id=f"{a}_{look}_brk_{side}", a=a, d=a + 5, side=side, kind="brk", look=look, L=1800.0, s1h=.004, P=1801.0, p=.6, grp=grp)
    rows = [ev(now - 900, 60, 1, "top"), ev(now - 900, 240, 1, "top"), ev(now - 800, 60, -1, "mid"), ev(now - 700, 60, -1, "bot"), ev(now - 100, 60, 1, "bot"), dict(ev(now - 50, 60, 1, "top"), lag=7),
            dict(k="res", id=f"{now - 900}_60_brk_1", y=1, hit_s=12, r=3.0), dict(k="res", id=f"{now - 700}_60_brk_-1", y=1, hit_s=40, r=2.0), dict(k="skip", a=1, why="warm")]
    (d / "ethusdt.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n{깨진 줄\n")
    o = sr_lean_payload(d, now)
    assert [(e["a"], e["look"], e["grp"], e["y"]) for e in o["ev"]] == [(now - 900, 60, "top", 1), (now - 700, 60, "bot", 1), (now - 100, 60, "bot", None)], o["ev"]   # 같은 초·쪽 중복은 하나 · 가운데는 안 보냄
    assert o["mid48"] == 1 and o["late"] == 1 and o["rec"] == {"n": 2, "hit": 1, "recent": [[(now - 895) * 1000, 1], [(now - 695) * 1000, 0]]}, o["rec"]   # top·y=1 적중 · bot·y=1 빗나감
    assert sr_lean_payload(d, now + 60)["stale"] is True
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
