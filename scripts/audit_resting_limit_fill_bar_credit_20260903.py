#!/usr/bin/env python3
"""오늘 이식한 지정가 상태기계가 **체결 봉 크레딧 결함**을 갖는지 실측 검증.

## 왜

2026-09-03 다른 세션에서 **진입 모델 v1/v2가 라벨 결함으로 무효** 판정됐다
(`docs/experiments/eth_entry_intrabar_fill_bar_credit_artifact_20260903.md`):

> `trail_out(...)`가 **체결 봉 `f` 자체부터** 평가한다. 지정가는 직전 종가보다 3 ATR 아래에
> 걸리므로 그 봉의 고가는 **체결 이전** 시점이다. 그걸 "진입 후 유리한 움직임"으로 크레딧하면
> 포지션이 없던 시점의 가격을 쓰는 것이다.
> 실측: 체결 봉 유리폭 중앙 **1.76 ATR**, 82.3%가 ARM(1.0)을 넘는다.
> HOLDOUT +78.16 -> **−2.99**(부호 반전).

⚠️**오늘 XRP·BTC 지정가 이식은 `research_eth_resting_limit_entry_20260902.py::run()`을
import해서 썼다.** 그 함수가 같은 결함을 갖는다면 오늘 결과도 전부 무효다.

코드를 읽으면 체결 봉에서 `continue`로 OPEN 관리를 건너뛰는 것처럼 보이지만,
**읽기로 판단하지 않는다** -- 오늘 하루 반복 확인된 규율이다. 실측한다.

## 검정

같은 데이터·같은 셀로 두 변형을 돌린다:

  · **A. 현행**(import한 `run`) -- 체결 봉에서 `continue`
  · **B. 결함 재현**(체결 봉부터 관리) -- `peak`를 체결 봉의 고/저로 초기화하고 그 봉부터 무장 판정

두 결과가 **다르면** 현행은 체결 봉을 제외하는 정직한 컨벤션(L1)이고, B가 결함본이 낸 수치다.
두 결과가 **같으면** 현행도 결함본이며 오늘 결과는 무효다.

부수로 **체결 봉의 유리폭 분포**를 낸다(ETH 실측 중앙 1.76 ATR과 대조).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

_R = importlib.util.spec_from_file_location(
    "restport", ROOT / "scripts/research_xrp_btc_resting_limit_entry_20260903.py")
_r = importlib.util.module_from_spec(_R)
_R.loader.exec_module(_r)
_E = _r._e

from live_evidence_signal_dashboard_20260823 import compute_signals   # noqa: E402

OUT = ROOT / "data/research/resting_limit_fill_bar_credit_audit_20260903.json"
NOTIONAL = 0.30 * 3.0


def log(m): print(f"[fillbar] {m}", flush=True)


def run_buggy(trig_bottom, trig_top, atr, o, h, l, c, lo_i, hi_i, *, depth, wait,
              sl, arm, trail, horizon, cost, flip=False):
    """⚠️**결함 재현본**: 체결 봉부터 트레일링을 관리한다(peak를 체결 봉 고/저로 초기화).
    `research_eth_entry_b6_expand_20260903.py`가 하던 것과 같은 컨벤션."""
    acct_cost = cost * NOTIONAL
    rets, excursions = [], []
    state = 0
    p_limit = p_side = p_exp = p_atr = 0.0
    entry = stop = peak = 0.0
    o_side = 0; o_exp = 0; armed = False
    n = len(c)
    for i in range(lo_i, min(hi_i + 400, n - 1)):
        if state == 2:
            done = False
            if o_side > 0:
                if l[i] <= stop:
                    rets.append(float(stop / entry - 1.0) * NOTIONAL - acct_cost); done = True
                else:
                    if h[i] > peak:
                        peak = h[i]
                        if not armed and (peak - entry) / entry >= arm * p_atr:
                            armed = True
                        if armed:
                            stop = max(stop, peak * (1.0 - trail * p_atr))
            else:
                if h[i] >= stop:
                    rets.append(float(1.0 - stop / entry) * NOTIONAL - acct_cost); done = True
                else:
                    if l[i] < peak:
                        peak = l[i]
                        if not armed and (entry - peak) / entry >= arm * p_atr:
                            armed = True
                        if armed:
                            stop = min(stop, peak * (1.0 + trail * p_atr))
            if not done and i >= o_exp:
                mv = (c[i] / entry - 1.0) if o_side > 0 else (1.0 - c[i] / entry)
                rets.append(float(mv) * NOTIONAL - acct_cost); done = True
            if done:
                state = 0
            else:
                continue
        if state == 1:
            hit = (l[i] <= p_limit) if p_side > 0 else (h[i] >= p_limit)
            if hit:
                entry = p_limit; o_side = int(p_side); armed = False
                stop = entry * (1.0 - sl * p_atr) if o_side > 0 else entry * (1.0 + sl * p_atr)
                # ⚠️여기가 결함: 체결 봉의 고/저를 peak로 삼는다(= 체결 이전 가격을 크레딧)
                peak = h[i] if o_side > 0 else l[i]
                exc = ((peak - entry) / entry if o_side > 0 else (entry - peak) / entry) / p_atr
                excursions.append(float(exc))
                if (peak - entry) / entry >= arm * p_atr if o_side > 0 \
                        else (entry - peak) / entry >= arm * p_atr:
                    armed = True
                    stop = (max(stop, peak * (1.0 - trail * p_atr)) if o_side > 0
                            else min(stop, peak * (1.0 + trail * p_atr)))
                o_exp = i + horizon - 1; state = 2
                continue
            if i >= p_exp:
                state = 0
            else:
                continue
        if state == 0 and lo_i <= i < hi_i:
            a = atr[i]
            if not (np.isfinite(a) and a > 0):
                continue
            side = 1 if trig_bottom[i] else (-1 if trig_top[i] else 0)
            if side == 0:
                continue
            if flip:
                side = -side
            if depth == 0.0:
                entry = o[i + 1]; o_side = side; armed = False
                stop = entry * (1.0 - sl * a) if side > 0 else entry * (1.0 + sl * a)
                peak = entry; p_atr = a; o_exp = i + 1 + horizon - 1; state = 2
            else:
                p_limit = c[i] * (1.0 - depth * a) if side > 0 else c[i] * (1.0 + depth * a)
                p_side = side; p_atr = a; p_exp = i + wait; state = 1
    return np.array(rets, dtype=float), np.array(excursions, dtype=float)


def main() -> int:
    t0 = time.time()
    rep = {"note": "현행 import 상태기계 vs 체결봉 크레딧 결함 재현본", "assets": {}}
    # 오늘 결과에서 가장 유력했던 셀들
    TARGETS = {
        "XRP": [("short_term_return_z", 3.0, 6), ("taker_delta_z_climax", 3.0, 3)],
        "BTC": [("short_term_return_z", 4.0, 3)],
    }
    for asset, cfg in _r.ASSETS.items():
        if asset not in TARGETS:
            continue
        raw = pd.read_csv(cfg["klines"], parse_dates=["timestamp"])
        raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        partner = pd.read_csv(cfg["partner"], usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
        funding = _r.load_funding(cfg["funding"])
        for d in (raw, partner):
            d["timestamp"] = d["timestamp"].astype("datetime64[ns]")
        frame = compute_signals(raw, btc_df=partner, funding_df=funding)
        frame["timestamp"] = frame["timestamp"].astype("datetime64[ns]")
        ts = frame["timestamp"]
        o, h, l, c = (frame[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        atr = frame["atr_pct"].to_numpy(float)
        idx = pd.DatetimeIndex(ts)
        W = {"VAL": (int(idx.searchsorted(_r.VAL_START)), int(idx.searchsorted(_r.OOS_START))),
             "OOS": (int(idx.searchsorted(_r.OOS_START)), int(idx.searchsorted(_r.HOLDOUT_START))),
             "HOLDOUT": (int(idx.searchsorted(_r.HOLDOUT_START)), len(idx))}
        gate = json.loads(cfg["gate"].read_text())["signals"]
        log(""); log("#" * 72); log(asset); log("#" * 72)
        res = {}
        for sname, depth, wait in TARGETS[asset]:
            gname = {"taker_delta_z_climax": "taker_delta_climax"}.get(sname, sname)
            v = gate.get(gname, {})
            g1 = v.get("genuine_arm_ge_1") or []
            if not g1:
                continue
            b = max(g1, key=lambda x: min(x["val_fwd_bp"], x["oos_fwd_bp"]))
            sp = {"sl": b["sl"], "arm": b["arm"], "trail": b["trail"], "horizon": v["horizon_bars"]}
            tb = frame[f"bottom_{sname}"].fillna(False).to_numpy(bool)
            tt = frame[f"top_{sname}"].fillna(False).to_numpy(bool)
            log("")
            log(f"=== {sname}  d={depth} w={wait}  셀 {sp} ===")
            row = {"depth": depth, "wait": wait, "cell": sp}
            for wn, (lo, hi) in W.items():
                a_ = _E.stats(_E.run(tb, tt, atr, o, h, l, c, lo, hi, depth=depth, wait=wait,
                                     cost=0.0010, flip=False, **sp))
                bg, exc = run_buggy(tb, tt, atr, o, h, l, c, lo, hi, depth=depth, wait=wait,
                                    cost=0.0010, flip=False, **sp)
                b_ = _E.stats(bg)
                row[wn] = {"current": a_, "buggy": b_,
                           "fill_excursion_atr": {
                               "median": float(np.median(exc)) if len(exc) else None,
                               "mean": float(exc.mean()) if len(exc) else None,
                               "frac_ge_arm": float((exc >= sp["arm"]).mean()) if len(exc) else None,
                               "n": int(len(exc))}}
                e = row[wn]["fill_excursion_atr"]
                log(f"  {wn:<8} 현행 {a_['mean_bp']:>+8.2f}bp (n={a_['n']:<4})   "
                    f"결함재현 {b_['mean_bp']:>+8.2f}bp (n={b_['n']:<4})   "
                    f"차 {b_['mean_bp'] - a_['mean_bp']:>+8.2f}")
                if e["n"]:
                    log(f"           체결봉 유리폭 중앙 {e['median']:.2f} ATR / 평균 {e['mean']:.2f} ATR"
                        f" / ARM({sp['arm']}) 초과 {e['frac_ge_arm']*100:.1f}%")
            same = all(abs(row[w]["current"]["mean_bp"] - row[w]["buggy"]["mean_bp"]) < 1e-6
                       for w in W)
            log(f"  ⇒ {'❌**현행도 결함본과 동일 -- 오늘 결과 무효**' if same else '✅현행은 체결봉을 제외한다(L1 정직 컨벤션)'}")
            row["identical_to_buggy"] = bool(same)
            res[sname] = row
        rep["assets"][asset] = res

    log(""); log("=" * 74)
    log("종합")
    log("=" * 74)
    bad = 0
    for asset, res in rep["assets"].items():
        for sname, row in res.items():
            bad += row["identical_to_buggy"]
            log(f"  {asset} {sname:<24} {'❌결함 동일' if row['identical_to_buggy'] else '✅정직(체결봉 제외)'}")
    log("")
    log(f"⇒ 결함을 가진 대상: **{bad}종**")
    rep["n_buggy"] = bad
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
