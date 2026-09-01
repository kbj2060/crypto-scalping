#!/usr/bin/env python3
"""라벨 1이 **진입가 기준으로 실제 먹을 수 있는 폭**을 측정 -- 수수료도 못 버는 양성 비율.

## 사용자 지적 (2026-09-02)

"익절가가 너무 가까워. 지금 보니까 수수료도 못 벌고 나오고 있는 걸 라벨 1로 모델한테
가르치고 있잖아."

## 이 지적이 겨누는 구멍

9-9에서 **절대 bp 하한**을 이미 스윕했다(FLOOR 0/10/20/30/40). 그런데 그 하한은
`fast_move_bp = (fast_close_max - low[i]) / close`, 즉 **앵커 기준**이다
(`..._absolute_bp_floor_sweep_20260901.py::side_fields` 100~116행에서 확인).
**진입가(open[i+1])와의 갭을 전혀 반영하지 않는다.**

라벨 양성의 소진 중앙값이 40%(2026-09-02 실측)이므로 앵커 기준 30bp짜리 양성은 진입 후
18bp만 남는다. 소진이 100%면 0이 남는다. 즉 **9-9의 하한은 거래 가능성을 보증하지 못했다.**

## 무엇을 재나

라벨 1(양성)에 대해 세 가지를 bp로 낸다. 전부 **진입가 open[i+1] 기준**:

  A. `capturable_bp` = (fast_close_max - entry)/entry -- 30분 창에서 **종가 기준 최대 이익**
     (라벨이 성공이라 부른 그 다리에서 실제로 먹을 수 있었던 최대치)
  B. `capturable_high_bp` = (fast_high_max - entry)/entry -- 고가 기준(리밋 익절 상한, 낙관)
  C. `target_remaining_bp` = 목표(앵커+1.5×ATR)까지 남은 거리

그리고 **왕복비용 10bp 대비 비율**을 낸다. 사용자 주장이 맞다면 A가 10bp 미만인 양성이
무시 못 할 비율이어야 한다.

부수: 앵커 기준(9-9가 쓴 `fast_move_bp`)과 나란히 찍어 **두 지표가 얼마나 다른지** 보인다.

⚠️HOLDOUT 미터치. 라이브 코드 변경 없음.

Run on the server via handoff.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_s1 = _load("s1_cap", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
_vs = _s1._vs
FAST_BARS, ATR_MULT, COST_BP = _s1.FAST_BARS_FIXED, 1.5, 10.0
DEPLOYED = {"atr_mult": 1.50, "t_sustain": 0.20, "full_bars": 12}
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
OUT = ROOT / "data/research/eth_v_rebound_capturable_bp_20260902/report.json"


def log(m): print(f"[cap] {m}", flush=True)


def qs(a, name):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    d = {"n": int(len(a))}
    for p in (5, 10, 25, 50, 75, 90):
        d[f"p{p}"] = round(float(np.percentile(a, p)), 2)
    d["mean"] = round(float(a.mean()), 2)
    log(f"    {name:26s} p10 {d['p10']:>8.2f}  p25 {d['p25']:>8.2f}  "
        f"중앙 {d['p50']:>8.2f}  p75 {d['p75']:>8.2f}  평균 {d['mean']:>8.2f}")
    return d


def main() -> int:
    _s1.VAL_END = OOS_END
    log("building frame ...")
    sig, feat, eth = _s1.build_sig()
    sb = _s1.label_param(sig, True, ambig="drop", anchor="wick", **DEPLOYED)
    st = _s1.label_param(sig, False, ambig="drop", anchor="wick", **DEPLOYED)
    long = _s1.long_frame_for(sig, feat, sb, st)
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                     np.where(long["timestamp"] < VAL_END, "VAL", "OOS"))
    assert long["timestamp"].max() < OOS_END, "HOLDOUT 누출"

    ts_pos = {t: i for i, t in enumerate(sig["timestamp"].dt.tz_localize(None).to_numpy())}
    long["pos"] = [ts_pos.get(np.datetime64(t.tz_localize(None)), -1) for t in long["timestamp"]]
    long = long.loc[long["pos"] >= 0].reset_index(drop=True)
    if not len(long):
        raise RuntimeError("타임스탬프 매칭 0건")

    close, high, low = (sig[c].to_numpy() for c in ("close", "high", "low"))
    op, atr = sig["open"].to_numpy(), sig["atr"].to_numpy()
    pre_atr = _vs.shifted_at(atr, -1)
    fmax = _vs.fwd_window(close, 1, FAST_BARS, "max")
    fmin = _vs.fwd_window(close, 1, FAST_BARS, "min")
    hmax = _vs.fwd_window(high, 1, FAST_BARS, "max")
    lmin = _vs.fwd_window(low, 1, FAST_BARS, "min")

    i = long["pos"].to_numpy().astype(int)
    dn = long["is_downside"].to_numpy() == 1
    sgn = np.where(dn, 1.0, -1.0)
    anchor = np.where(dn, low[i], high[i])
    entry = op[np.minimum(i + 1, len(op) - 1)]
    target = anchor + sgn * ATR_MULT * pre_atr[i]

    long["capturable_bp"] = sgn * (np.where(dn, fmax[i], fmin[i]) - entry) / entry * 1e4
    long["capturable_high_bp"] = sgn * (np.where(dn, hmax[i], lmin[i]) - entry) / entry * 1e4
    long["target_remaining_bp"] = sgn * (target - entry) / entry * 1e4
    long["anchor_move_bp"] = sgn * (np.where(dn, fmax[i], fmin[i]) - anchor) / close[i] * 1e4
    long["atr_bp"] = pre_atr[i] / close[i] * 1e4

    report = {"signal": "v_rebound_capturable_bp_from_entry", "asset": "ETHUSDT",
              "scope": {"cost_bp": COST_BP,
                        "note": "9-9의 bp 하한은 anchor_move_bp 기준 -- 진입가 갭 미반영",
                        "holdout_touched": False, "live_code_changed": False}, "splits": {}}

    for spn in ("TRAIN", "VAL", "OOS"):
        s = long.loc[(long["split"] == spn) & (long["label"] == 1)]
        if len(s) < 100:
            continue
        log("")
        log(f"===== {spn}  라벨 1 (양성) {len(s):,}건 =====")
        d = {"n": int(len(s))}
        d["atr_bp"] = qs(s["atr_bp"], "ATR (bp)")
        d["anchor_move_bp"] = qs(s["anchor_move_bp"], "앵커기준 이동 (9-9 하한)")
        d["capturable_bp"] = qs(s["capturable_bp"], "⭐진입기준 먹을수있는폭")
        d["capturable_high_bp"] = qs(s["capturable_high_bp"], "  (고가기준, 낙관)")
        d["target_remaining_bp"] = qs(s["target_remaining_bp"], "목표까지 남은거리")

        cap = s["capturable_bp"].to_numpy()
        anc = s["anchor_move_bp"].to_numpy()
        log("")
        log(f"    ⭐왕복비용 {COST_BP}bp 대비 (라벨 1 중):")
        for thr, nm in ((0, "≤0 (진입 즉시 손실)"), (COST_BP, f"<{COST_BP:.0f}bp (수수료 미만)"),
                        (2 * COST_BP, "<20bp"), (3 * COST_BP, "<30bp")):
            f_cap = float((cap < thr).mean()) * 100 if thr > 0 else float((cap <= 0).mean()) * 100
            f_anc = float((anc < thr).mean()) * 100 if thr > 0 else float((anc <= 0).mean()) * 100
            d[f"pct_capturable_below_{thr:.0f}"] = round(f_cap, 1)
            d[f"pct_anchor_below_{thr:.0f}"] = round(f_anc, 1)
            log(f"      {nm:24s} 진입기준 **{f_cap:5.1f}%**   (앵커기준으로는 {f_anc:4.1f}%)")
        report["splits"][spn] = d

    log("")
    log("=== 해석 기준 ===")
    log("  앵커기준으로는 거의 전부 비용 위인데 진입기준으로 큰 비율이 비용 미만이면,")
    log("  9-9의 bp 하한이 거래 가능성을 보증하지 못했다는 뜻이다(사용자 지적이 맞음).")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
