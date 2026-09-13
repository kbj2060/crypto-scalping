#!/usr/bin/env python3
"""**예산 사다리 섀도우 기록기** — 주문을 내지 않는다 (2026-09-14, 사용자 요청 «1단계»).

사다리(`exit_fraction_required`)는 지금 **화면 표시 전용**이다. 자동 집행으로 올릴지 판단하려면
«실제로 얼마나 자주·언제·얼마나 크게 발동하는가»를 먼저 알아야 한다. 이 러너가 그걸만 기록한다.

## ⭐로직을 재구현하지 않는다 — 배포된 엔드포인트를 그대로 부른다
`required_fraction` 을 만드는 `effective_cap` / `remaining_hold` / `planning_hold` 는 전부
`dashboard/server.py::make_app()` 안의 **클로저**라 import 이 안 된다. 베껴 쓰면 배포본과 조용히
갈라진다 — 2026-09-14 세션에서 고친 결함 셋이 **전부 그 부류**였다(청산 카드가 상한을 상수로 쓰던 것 ·
하네스가 지평을 48봉으로 쓰던 것 · 하네스 사다리가 문턱을 25배로 쓰던 것).
⇒ `/api/manual-exit/preview` 를 호출해 **화면이 보는 바로 그 숫자**를 받아 적는다. 사본이 없으니
   갈라질 수 없다. 대가는 대시보드가 떠 있어야 한다는 것뿐이고, 못 붙으면 그 틱을 건너뛴다.

## 기록만 한다
주문 경로(`run_exit`)를 import 조차 하지 않는다. 이 파일이 실수로도 주문을 낼 수 없게 하려는 것이다.

## 읽는 법
    python3 scripts/live_eth_exit_ladder_shadow_20260914.py            # 기록 루프
    python3 scripts/live_eth_exit_ladder_shadow_20260914.py --report   # 지금까지 요약
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "live" / "exit_ladder_shadow.jsonl"
BASE = "http://127.0.0.1:8787"
SLEEP_SEC = 60.0
# 🔴시뮬과 **같은 하한**을 쓴다(research_fresh_forward_random_entry_stack_20260914 의 사다리 절단).
# 1% 미만은 격자·수수료에 묻혀 실제로는 못 닫는다 -- 여기서 «발동»으로 세면 빈도가 부풀려진다.
MIN_FRACTION = 0.01
# 기록할 필드. plan.risk 아래에서 그대로 꺼낸다(가공하지 않는다 -- 가공은 --report 에서).
RISK_FIELDS = ("required_fraction", "allowed_notional", "excess_notional", "current_notional",
               "effective_x", "applied_binding", "leverage", "safe_mae_pct", "hold_min")


def _get(path: str, timeout: float = 25.0):
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None      # 대시보드 재기동 중이면 그 틱은 건너뛴다


def sample(side: str) -> dict | None:
    """한 측면의 현재 사다리 상태. 포지션이 없으면 None."""
    b = _get(f"/api/manual-exit/preview?side={side}")
    if not b or not b.get("ok"):
        return None                              # no_position 포함
    plan = b.get("plan") or {}
    risk = plan.get("risk") or {}
    if not risk.get("available"):
        return {"side": side, "available": False,
                "reason": risk.get("reason"), "unrealized_pnl": plan.get("unrealized_pnl")}
    return {"side": side, "available": True,
            **{k: risk.get(k) for k in RISK_FIELDS},
            "hold_planned_min": plan.get("hold_planned_min"),
            "hold_remaining_min": plan.get("hold_remaining_min"),
            "unrealized_pnl": plan.get("unrealized_pnl"),
            "quantity": plan.get("quantity"), "price": plan.get("price")}


def record_once(out: pathlib.Path) -> int:
    """두 측면을 찍어 JSONL 에 append. 기록한 줄 수를 돌려준다."""
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for side in ("LONG", "SHORT"):
            s = sample(side)
            if s is None:
                continue                          # 포지션 없음 -- 적을 게 없다
            f.write(json.dumps({"ts": now, **s}, ensure_ascii=False) + "\n")
            n += 1
    return n


def report(out: pathlib.Path) -> int:
    if not out.exists():
        print(f"기록 없음: {out}"); return 0
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    live = [r for r in rows if r.get("available")]
    print(f"표본 {len(rows):,}틱 (포지션 있던 틱 {len(live):,})")
    if not live:
        print("포지션이 있던 틱이 없다 -- 사다리를 평가할 표본이 아직 없다.")
        return 0
    t0, t1 = live[0]["ts"][:16], live[-1]["ts"][:16]
    fired = [r for r in live if (r.get("required_fraction") or 0.0) >= MIN_FRACTION]
    hours = max((datetime.fromisoformat(live[-1]["ts"]) -
                 datetime.fromisoformat(live[0]["ts"])).total_seconds() / 3600.0, 1e-9)
    print(f"구간 {t0} ~ {t1} ({hours:.1f}시간)")
    print(f"발동(>= {100*MIN_FRACTION:.0f}%) {len(fired)}틱 "
          f"= 포지션 있던 틱의 {100*len(fired)/len(live):.1f}% · 시간당 {len(fired)/hours:.2f}회")
    if fired:
        fr = sorted(r["required_fraction"] for r in fired)
        q = lambda p: fr[min(len(fr) - 1, int(p * len(fr)))]          # noqa: E731
        print(f"요구 비율: 중앙 {100*q(.5):.1f}% · 90분위 {100*q(.9):.1f}% · 최대 {100*fr[-1]:.1f}%")
        print("묶은 상한:", dict(Counter(r.get("applied_binding") for r in fired)))
        print("측면:", dict(Counter(r["side"] for r in fired)))
    print("\n⚠️틱 수는 «연속 발동»을 세므로 «닫아야 할 사건 수»가 아니다 -- 한 번 역행하면 닫을 "
          "때까지 매 틱 걸린다. 사건 수로 보려면 연속 구간을 하나로 묶어야 한다(아래).")
    runs, prev = 0, False
    for r in live:
        f = (r.get("required_fraction") or 0.0) >= MIN_FRACTION
        runs += f and not prev
        prev = f
    print(f"연속 구간으로 묶으면 **{runs}건** · 시간당 {runs/hours:.2f}건 · 하루 {24*runs/hours:.1f}건")
    return 0


def _self_check() -> None:
    """발동 판정과 구간 묶기 -- 이 둘이 리포트의 결론을 만든다."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "x.jsonl"
        # 0.5% 는 하한 미만이라 발동이 아니다. 연속 두 틱은 **한 건**으로 묶여야 한다.
        seq = [0.0, 0.005, 0.03, 0.04, 0.0, 0.0, 0.02]
        with p.open("w", encoding="utf-8") as f:
            for i, v in enumerate(seq):
                f.write(json.dumps({"ts": f"2026-09-14T00:{i:02d}:00+00:00", "side": "LONG",
                                    "available": True, "required_fraction": v,
                                    "applied_binding": "equity"}) + "\n")
        rows = [json.loads(l) for l in p.read_text().splitlines()]
        fired = [r for r in rows if r["required_fraction"] >= MIN_FRACTION]
        assert len(fired) == 3, f"발동 틱 {len(fired)} != 3 (0.5% 를 셌나?)"
        runs, prev = 0, False
        for r in rows:
            f = r["required_fraction"] >= MIN_FRACTION
            runs += f and not prev
            prev = f
        assert runs == 2, f"연속 구간 {runs} != 2 (0.03·0.04 는 한 건, 0.02 가 두 번째)"
    # 🔴집행 모듈이 로드되지 않았는지 **런타임으로** 확인한다. 소스에서 금지 문자열을 찾는
    # 방식은 **점검 코드 자신이 걸린다**(첫 판에서 실제로 그랬다) -- 금지어를 적으려면 그 금지어를
    # 파일에 써야 하기 때문이다. 모듈이 실제로 import 됐는지는 그런 자기참조가 없다.
    loaded = [m for m in sys.modules if "manual_peg_execute" in m]
    assert not loaded, f"집행 모듈이 로드됐다: {loaded} -- 섀도우가 아니다"
    print("통과 — 발동 하한 · 연속 구간 묶기 · 집행 모듈 미로드")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="지금까지 기록을 요약한다")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--once", action="store_true", help="한 번만 찍고 끝낸다(점검용)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if a.self_check:
        _self_check(); return 0
    if a.report:
        return report(out)
    if a.once:
        print(f"{record_once(out)}줄 기록 -> {out}"); return 0
    print(f"사다리 섀도우 시작 · {SLEEP_SEC:.0f}초 간격 · {out}  (주문은 내지 않는다)", flush=True)
    while True:
        try:
            record_once(out)
        except Exception as exc:                  # noqa: BLE001 -- 기록기가 죽으면 안 된다
            print(f"tick 실패: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        time.sleep(SLEEP_SEC)


if __name__ == "__main__":
    raise SystemExit(main())
