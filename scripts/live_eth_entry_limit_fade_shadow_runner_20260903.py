#!/usr/bin/env python3
"""ETH 지정가 페이드 진입 **섀도우 러너 (v3)** -- 가상 주문/체결만 기록. 주문을 내지 않는다.

⚠️⚠️**2026-09-03 전수조사로 v1/v2가 철회됐다.** 라벨(`trail_out`)이 **체결 봉 자체부터**
평가해 **체결 이전 고가**를 진입 후 이익으로 크레딧했다 -- 지정가가 직전 종가보다 3 ATR
아래라 그 봉의 고가는 거의 확실히 체결 이전이고, 실측 유리폭이 중앙 1.76 ATR·82.3%가
ARM(1.0) 초과라 **진입 즉시 무장**됐다. 1분봉 해상으로 확증한 결과 전체 후보 팔의
**PF 2.86 → 0.95**, 평균 +20.34 → −1.01bp였다.
전문: `docs/experiments/eth_entry_intrabar_fill_bar_credit_artifact_20260903.md`

**이 전략에는 현재 확립된 엣지가 없다.** 정직한 라벨에서 ①트리거가 무작위 봉보다 못하고
(VAL +1.48 vs +2.87 · OOS +1.95 vs +5.45) ②독립 일수가 42~45일뿐이라 일 단위 군집 부트스트랩
CI 하한이 VAL/OOS에서 음수이며 ③dtype 하나에 VAL 부호가 뒤집힌다.

⭐**그래서 이 러너의 목적은 성과 시연이 아니라 "정직한 기반 위에서 전진 데이터를 모으는 것"**
이다. VAL/OOS/HOLDOUT은 하루에만 수십 번 재사용돼 더 이상 어떤 판정 근거도 되지 못한다.
신선한 표본만이 남은 셋(페이드 방향성 · arm1만>양팔 · 모델 선별력의 시드 안정성)이 진짜인지
가릴 수 있다.

## v2 → v3 변경 셋

  ① **L3 정직 라벨로 학습** -- 1분봉으로 체결 시점을 특정하고 그 이후 구간만 크레딧
  ② **arm1(신호방향)만 제출** -- 역방향 팔이 아티팩트의 최대 수혜자였다(정직 라벨에서 OOS −11.33bp)
  ③ **청산도 L3 규약** -- 체결 봉은 `post_fill_range()`가 낸 **사후 구간만** 기여한다.
     학습 라벨과 같은 규약이라 학습/추론 정합이 맞는다(v1/v2 섀도우는 L1이라 어긋나 있었다).

## 이 러너가 실제로 재려는 것

  ①**체결 모델** -- 백테스트는 "저가 ≤ 지정가면 지정가 체결"을 가정했다. 3 ATR 떨어진
    지정가는 가격이 그쪽으로 강하게 이동해야 닿으므로 역선택이 구조적으로 의심된다.
  ②**HGB vs TabPFN** -- 동결 판정이 순열검정 OOS p=0.0518로 경계선이었다. 소진된 창의
    p값 대신 전진 데이터로 끝낸다.
  ③라이브 피쳐/트리거가 연구와 같은 빈도로 나오는가 (실측 기준선 아래 참조)

## ⭐체결 판정 두 가지를 나란히 기록한다 (사용자 결정, 2026-09-03)

  A `fill_5m`  -- 백테스트 그대로. 5분봉 저가 ≤ 지정가(매수) / 고가 ≥ 지정가(매도).
                 **포지션은 이 기준으로 연다** -- 동결 수치와 직접 비교 가능해야 하므로.
  B `fill_1m`  -- 1분봉 해상도로 다시 본다. `through`(지정가를 **뚫음** = 확실 체결) /
                 `touch_only`(봉 극값이 지정가와 같음 = 큐 위치에 달림) / `none`으로 분류.

⚠️**maker 워커의 큐 규칙은 이식할 수 없다.** 그건 터치(L1)에 붙이는 주문용이고, 우리 지정가는
3 ATR(약 0.8%) 떨어져 있어 `/fapi/v1/depth?limit=1000`(약 $10 범위)의 **바깥**이다 -- 제출
시점에 앞선 큐를 관측할 방법이 없다. 그래서 관측 가능한 축(뚫음 vs 스침)으로 대체한다.
`touch_only` 비율이 높게 나오면 그때 스트림 기반 큐 추정을 설계한다.

## 배리어 판정 = 완결 봉 고가/저가

`sim_exit`/`trail_out`과 같은 컨벤션이며 순서도 같다 -- 봉마다 (1)불리한 쪽 스톱 판정,
(2)유리한 쪽 best 갱신, (3)무장, (4)트레일 조임. 순서를 바꾸면 낙관 편향이다.
마크가격 한 점만 보면 wick 스톱을 놓쳐 손실이 통째로 사라진다(V자반등 섀도우 전례:
원장 9건 전부 양수 +69bp = 계측 아티팩트).

## 실측 기준선 (2026-09-03, 최근 2000봉)

  트리거 발동 77.3건/일 · **채점 대상 봉 41.2봉/일(전체봉의 14.3%)** · 제출 팔 154.7개/일
  체결률 추정 11.9% (연구 HOLDOUT 실측 체결 18.42건/일 ÷ 제출 154.7)
  → TabPFN 동시채점 비용 = 41.2회/일 × 69초 = **하루 약 47분 GPU**

## 사전등록 -- 모델 전환 규칙 (2026-09-03, 착수 전 고정)

섀도우 체결 **200건 이상** 누적 **그리고** 같은 후보 봉 기준 페어드 부트스트랩
(TabPFN − HGB) 95% CI **하한 > 0** 이면 TabPFN으로 전환한다. 미달이면 HGB 유지.
이 규칙은 데이터를 보기 전에 고정됐다.

Usage:
    python scripts/live_eth_entry_limit_fade_shadow_runner_20260903.py            # 1회
    python scripts/live_eth_entry_limit_fade_shadow_runner_20260903.py --loop     # 상시(300초)
    python scripts/live_eth_entry_limit_fade_shadow_runner_20260903.py --report   # 원장 요약
    python ... --loop --tabpfn                                                    # 대조 채점 포함
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_SIG = importlib.import_module("live_eth_entry_limit_fade_signal_20260903")

STATE = ROOT / "data/live/entry_limit_fade_v4_shadow_state.json"
KL1M = "https://fapi.binance.com/fapi/v1/klines"
LOOP_SECONDS = 300                       # 5분봉 1회. 60초로 줄이면 같은 봉을 4중 채점한다.
COST_SOURCE = "assumed_10bp"             # peg-maker 실측(>=2026-09-04) 반영 시 갱신


def log(m: str) -> None:
    print(f"[entry-shadow {datetime.now(timezone.utc):%m-%d %H:%M:%S}] {m}", flush=True)


def load_state() -> dict[str, Any]:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:                                          # noqa: BLE001
            log("⚠️상태 파싱 실패 -- 새로 시작")
    return {"pending": [], "positions": [], "ledger": [], "skipped": 0,
            "started_utc": datetime.now(timezone.utc).isoformat(), "version": 1}


def save_state(s: dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, ensure_ascii=False, indent=2, default=str))
    tmp.replace(STATE)                                             # 원자적 교체


def fetch_1m(start_utc: str, end_utc: str) -> list[dict]:
    """체결 판정 B용 1분봉. 실패해도 A 판정에는 영향이 없다."""
    try:
        s_ms = int(pd.Timestamp(start_utc).timestamp() * 1000)
        e_ms = int(pd.Timestamp(end_utc).timestamp() * 1000) + 60_000
        r = requests.get(KL1M, params={"symbol": "ETHUSDT", "interval": "1m",
                                       "startTime": s_ms, "endTime": e_ms, "limit": 1000},
                         timeout=15)
        r.raise_for_status()
        return [{"t": str(pd.to_datetime(k[0], unit="ms")), "h": float(k[2]), "l": float(k[3])}
                for k in r.json()]
    except Exception:                                              # noqa: BLE001
        return []


def classify_1m(sd: int, lim: float, bars1m: list[dict]) -> tuple[str, str | None]:
    """`through`(뚫음, 확실 체결) / `touch_only`(스침, 큐 의존) / `none`."""
    touch_t = None
    for b in bars1m:
        if sd > 0:                                                 # 아래 매수
            if b["l"] < lim:
                return "through", b["t"]
            if b["l"] == lim and touch_t is None:
                touch_t = b["t"]
        else:                                                      # 위 매도
            if b["h"] > lim:
                return "through", b["t"]
            if b["h"] == lim and touch_t is None:
                touch_t = b["t"]
    return ("touch_only", touch_t) if touch_t else ("none", None)


def post_fill_range(sd: int, lim: float, bars1m: list[dict]) -> tuple[float, float] | None:
    """⭐**L3 규약**: 체결 봉에서 지정가에 처음 닿은 분을 찾고, **그 다음 분부터**의
    (고가, 저가)만 돌려준다. 학습 라벨(L3)과 같은 규약이라 학습/추론 정합이 맞는다.

    ⚠️왜 필요한가: v1/v2는 체결 봉 **전체**를 청산에 썼는데, 지정가가 직전 종가보다 3 ATR
    아래라 그 봉의 고가는 **거의 확실히 체결 이전**이다(실측 유리폭 중앙 1.76 ATR, 82.3%가
    ARM 초과). 그 결과 진입 즉시 무장돼 전체 후보 PF가 2.86으로 부풀었고, 정직하게 고치면
    0.95였다. 섀도우가 이 규약을 안 따르면 학습과 채점이 어긋난다.

    체결이 봉의 마지막 분이면 사후 구간이 없으므로 (lim, lim)을 돌려준다 -- 그 봉은 아무것도
    기여하지 않는다는 뜻이다(실측상 체결의 36.8%가 마지막 2분에 일어난다)."""
    k0 = None
    for k, b in enumerate(bars1m):
        if (b["l"] <= lim) if sd > 0 else (b["h"] >= lim):
            k0 = k; break
    if k0 is None:
        return None
    post = bars1m[k0 + 1:]
    if not post:
        return lim, lim
    return max(b["h"] for b in post), min(b["l"] for b in post)


def expire_and_fill(s: dict[str, Any], bars: list[dict], pol: dict) -> None:
    """대기 중 지정가를 새 완결 봉에 대해 체결/만료 처리한다."""
    # ⭐섀도우는 슬롯을 걸지 않는다 -- 원장에 fi/ei가 있으므로 어떤 슬롯 정책이든 사후
    # 재구성할 수 있다. 배치 시점에 버리면 그 정보는 영영 못 얻는다.
    wait, slots = int(pol["wait_bars"]), 10**6
    keep = []
    for o in s["pending"]:
        todo = [b for b in bars if b["timestamp_utc"] > o["last_bar_utc"]]
        filled_bar = None
        for b in todo:
            o["bars_waited"] = int(o.get("bars_waited", 0)) + 1
            o["last_bar_utc"] = b["timestamp_utc"]
            hit = (b["low"] <= o["limit"]) if o["sd"] > 0 else (b["high"] >= o["limit"])
            if hit:
                filled_bar = b
                break
            if o["bars_waited"] >= wait:
                break
        if filled_bar is not None:
            b1 = fetch_1m(filled_bar["timestamp_utc"], filled_bar["timestamp_utc"])
            basis_b, t1 = classify_1m(o["sd"], o["limit"], b1)
            pf_ = post_fill_range(o["sd"], o["limit"], b1)
            if len(s["positions"]) >= slots:
                s["skipped"] += 1                                  # slotN 의미: 슬롯 없으면 미체결
                log(f"  [슬롯없음] {o['signal']} arm{o['arm']} 건너뜀 ({s['skipped']}건 누적)")
                continue
            s["positions"].append({**o, "entry": o["limit"], "entry_bar_utc": filled_bar["timestamp_utc"],
                                   "stop": o["limit"] - o["sd"] * pol["exit"]["sl_atr"] * o["atr_abs"],
                                   "best": o["limit"], "armed": False, "bars_held": 0,
                                   "fill_1m_basis": basis_b, "fill_1m_utc": t1,
                                   "n_1m_bars": len(b1),
                                   # ⭐L3: 체결 봉은 **사후 구간만** 청산에 기여한다
                                   "post_fill_high": (pf_[0] if pf_ else None),
                                   "post_fill_low": (pf_[1] if pf_ else None)})
            log(f"  [가상체결] {o['signal']} arm{o['arm']} sd{o['sd']:+d} @{o['limit']:.2f} "
                f"({o['bars_waited']}봉 대기, 1m판정 {basis_b})")
        elif o["bars_waited"] >= wait:
            s.setdefault("cancelled", 0)
            s["cancelled"] += 1
        else:
            keep.append(o)
    s["pending"] = keep


def manage(s: dict[str, Any], bars: list[dict], pol: dict) -> None:
    """⭐`trail_out`과 **같은 순서**: 불리한 쪽 스톱 → best → 무장 → 트레일."""
    ex, keep = pol["exit"], []
    for p in s["positions"]:
        sgn, a = p["sd"], p["atr_abs"]
        todo = [b for b in bars if b["timestamp_utc"] > p["entry_bar_utc"]
                and b["timestamp_utc"] > p.get("last_bar_utc", "")]
        # ⭐L3 규약: 아직 체결 봉을 평가하지 않았다면, 그 봉의 **사후 구간**을 먼저 처리한다.
        # (v1/v2는 체결 봉 전체를 썼고 그게 미래참조였다. 사후 구간만이 실제로 우리 것이다.)
        if not p.get("post_fill_done") and p.get("post_fill_high") is not None:
            todo = [{"timestamp_utc": p["entry_bar_utc"] + "#post",
                     "open": p["entry"], "high": p["post_fill_high"],
                     "low": p["post_fill_low"], "close": p["entry"]}] + todo
            p["post_fill_done"] = True
        closed = False
        for b in todo:
            adv = b["low"] if sgn > 0 else b["high"]
            if (adv <= p["stop"]) if sgn > 0 else (adv >= p["stop"]):
                _close(s, p, p["stop"], "stop", b["timestamp_utc"]); closed = True; break
            fav = b["high"] if sgn > 0 else b["low"]
            if sgn * (fav - p["best"]) > 0:
                p["best"] = fav
            if not p["armed"] and sgn * (p["best"] - p["entry"]) >= ex["arm_atr"] * a:
                p["armed"] = True
            if p["armed"]:
                ns = p["best"] - sgn * ex["trail_atr"] * a
                if sgn * (ns - p["stop"]) > 0:
                    if sgn * (ns - b["close"]) > 0:                   # 걸 수 없는 스톱(2026-09-07): 새 스톱이 그 봉 종가보다 유리한 쪽이면 거래소가 거부하는 자리다
                    # ("Order would immediately trigger"). 그 봉 종가에 즉시 청산한다.
                        _close(s, p, b["close"], "stop_infeasible", b["timestamp_utc"]); closed = True; break
                    p["stop"] = ns
            if not b["timestamp_utc"].endswith("#post"):
                p["bars_held"] = int(p.get("bars_held", 0)) + 1
                p["last_bar_utc"] = b["timestamp_utc"]
            if p["bars_held"] >= int(p["horizon"]):
                _close(s, p, b["close"], "timeout", b["timestamp_utc"]); closed = True; break
        if not closed:
            keep.append(p)
    s["positions"] = keep


def _close(s: dict[str, Any], p: dict[str, Any], px: float, reason: str, bar_utc: str) -> None:
    mv = p["sd"] * (px - p["entry"]) / p["entry"]
    net = (mv - float(p["cost_roundtrip"])) * float(p["notional"])
    s["ledger"].append({
        "signal": p["signal"], "side": p["side"], "arm": p["arm"], "sd": p["sd"],
        "placed_bar_utc": p["placed_bar_utc"], "entry_bar_utc": p["entry_bar_utc"],
        "exit_utc": bar_utc, "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "entry": p["entry"], "exit": px, "atr_pct": p["atr_pct"],
        "bars_waited": p.get("bars_waited"), "bars_held": p.get("bars_held"),
        "pnl_bp": round(net * 1e4, 3), "reason": reason,
        "pred_hgb": p.get("pred_hgb"), "pass_hgb": p.get("pass_hgb"),
        "gate_p90": p.get("gate_p90"), "atr_pct_at_entry": p.get("atr_pct"),
        "pred_tabpfn": p.get("pred_tabpfn"),
        "fill_5m_basis": "bar_low_high", "fill_1m_basis": p.get("fill_1m_basis"),
        "fill_1m_utc": p.get("fill_1m_utc"), "exit_basis": "L3_post_fill_then_bar_high_low",
        "cost_source": COST_SOURCE})
    log(f"  [청산] {p['signal']} arm{p['arm']} {net*1e4:+.2f}bp ({reason}, {p.get('bars_held')}봉, "
        f"1m판정 {p.get('fill_1m_basis')})")


def place(s: dict[str, Any], out: dict[str, Any], pol: dict) -> None:
    """τ를 넘긴 후보만 가상 제출. 슬롯은 **체결 시점**에 판정한다(slotN 의미)."""
    ts, close = out["last_closed_bar_utc"], out["close"]
    if any(o["placed_bar_utc"] == ts for o in s["pending"]):
        return                                                     # 같은 봉 중복 제출 방지
    n = 0
    for c in out["candidates"]:
        if not c.get("pass_tau"):
            continue
        s["pending"].append({**c, "placed_bar_utc": ts, "last_bar_utc": ts, "bars_waited": 0,
                             "atr_abs": float(c["atr_pct"]) * close,
                             "notional": float(pol["margin_fraction"]) * float(pol["leverage"]),
                             "cost_roundtrip": float(pol["cost_roundtrip"]),
                             "placed_utc": datetime.now(timezone.utc).isoformat()})
        n += 1
    if n:
        log(f"  [가상제출] {n}개 (후보 {len(out['candidates'])}개 중 τ통과)")


def report(s: dict[str, Any]) -> None:
    led = s["ledger"]
    log("=== 진입 섀도우 원장 ===")
    start = pd.Timestamp(s.get("started_utc"))
    days = max((pd.Timestamp.now(tz="UTC") - start).total_seconds() / 86400, 1e-9)
    log(f"  가동 {days:.2f}일 · 대기 {len(s['pending'])} · 포지션 {len(s['positions'])} · "
        f"취소 {s.get('cancelled', 0)} · 슬롯없음 {s.get('skipped', 0)}")
    if not led:
        log("  원장 비어있음"); return
    pnl = np.array([t["pnl_bp"] for t in led], float)
    w = pnl > 0
    log(f"  체결 {len(pnl):,}건 ({len(pnl)/days:.2f}건/일) · 기대값 {pnl.mean():+.2f}bp · "
        f"누적 {pnl.sum():+.0f}bp · 승률 {w.mean()*100:.1f}%")
    b1 = pd.Series([t.get("fill_1m_basis") for t in led]).value_counts()
    log(f"  ⭐1분봉 체결판정: " + " · ".join(f"{k} {v}({v/len(led)*100:.0f}%)" for k, v in b1.items()))
    thr = [t["pnl_bp"] for t in led if t.get("fill_1m_basis") == "through"]
    if thr and len(thr) < len(led):
        log(f"     through만: {np.mean(thr):+.2f}bp (n={len(thr)}) vs 전체 {pnl.mean():+.2f}bp "
            f"← 격차가 크면 5분봉 가정이 낙관적이라는 뜻")
    tp = [(t["pred_tabpfn"], t["pnl_bp"]) for t in led if t.get("pred_tabpfn") is not None]
    log(f"  TabPFN 채점 {len(tp)}/{len(led)}건 · 사전등록 전환선 200건")
    log(f"  [동결 대조 v3·진단] VAL +10.58 · OOS +37.06 · HOLDOUT +32.75bp "
        f"(무필터 arm1: +4.01/+6.02/−0.99) · 비용출처 {COST_SOURCE}")
    log("  ⚠️위 수치는 소진된 창의 진단이다. 이 러너의 목적은 전진 데이터 수집이다.")


def cycle(s: dict[str, Any], use_tabpfn: bool) -> None:
    out = _SIG.compute_entry_signal(score_tabpfn=use_tabpfn)
    if not out.get("warmed_up"):
        log(f"⚠️신호 이상: {out.get('error')} -- 이번 사이클 건너뜀"); return
    _, CARD = _SIG._art()
    pol = CARD["policy"]
    bars = out.get("bars") or []
    expire_and_fill(s, bars, pol)
    manage(s, bars, pol)
    place(s, out, pol)
    npass = sum(1 for c in out["candidates"] if c.get("pass_tau"))
    log(f"봉 {out['last_closed_bar_utc'][:16]} close={out['close']:.2f} · 후보 {len(out['candidates'])} "
        f"(τ통과 {npass}) · 대기 {len(s['pending'])} · 포지션 {len(s['positions'])} · "
        f"원장 {len(s['ledger'])}건")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--tabpfn", action="store_true", help="TabPFN 대조 채점(하루 약 47분 GPU)")
    args = ap.parse_args()
    s = load_state()
    if args.report:
        report(s); return 0
    _, CARD = _SIG._art()
    pol = CARD["policy"]
    log(f"⚠️섀도우 모드 -- 주문을 내지 않습니다. v4(L3·arm1·HGB+변동성게이트) "
        f"depth {pol['depth_atr']}×ATR · 대기 {pol['wait_bars']}봉 · 주기 {LOOP_SECONDS}초 "
        f"· 청산규약 L3")
    log(f"  ⭐전부 제출·전부 기록 (필터는 사후 적용): HGB τ {pol['hgb_tau']*1e4:+.2f}bp · "
        f"변동성 p90 {pol['vol_threshold_atr_pct']:.6f}")
    if args.tabpfn:
        log(f"TabPFN 컨텍스트 구축 중...")
        log(f"  멤버 {_SIG.build_tabpfn()}개 상주")
    if not args.loop:
        cycle(s, args.tabpfn); save_state(s); report(s); return 0
    while True:
        try:
            cycle(s, args.tabpfn); save_state(s)
        except KeyboardInterrupt:
            save_state(s); log("중단"); return 0
        except Exception as e:                                      # noqa: BLE001
            log(f"⚠️사이클 예외: {type(e).__name__}: {e}")
        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
