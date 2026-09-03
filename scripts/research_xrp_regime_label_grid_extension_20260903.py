#!/usr/bin/env python3
"""XRP 레짐 라벨 격자 **경계 확장** + 교차자산 파트너 감사.

## 왜 -- 승자가 두 축 모두 격자 상단 경계에 있다

2026-09-03 XRP 레짐 Phase 2에서 `S48_K6`이 뽑혔는데:

    SCALES    = (6, 12, 24, 48)   -> 승자 S48  = **상단 경계**
    DEBOUNCES = (1, 3, 6)         -> 승자 K6   = **상단 경계**

호메로스 README **5.6절**(격자 경계 규칙)은 경계에서 최선이 나오면 그 방향으로 격자를
넓히라고 못박는다. 게다가 `DEBOUNCES`는 ETH 스크립트에서 그대로 import한 것이고,
K=12를 뺀 근거도 **ETH의 Phase 1에서 관측된 lock-up**이지 XRP에서 확인한 게 아니다.
XRP 문서 자신이 "디바운스가 스케일보다 중요"라고 적었는데, 그건 정확히 "K를 더 봤어야
한다"는 신호다.

## 확장

    SCALES    (6, 12, 24, 48)  ->  (6, 12, 24, 48, 96, 192)
    DEBOUNCES (1, 3, 6)        ->  (1, 3, 6, 9, 12)

K=12를 되살려 XRP에서 실제로 lock-up이 나는지 직접 본다(라벨 전환 횟수로 확인).

## 곁들여 -- 교차자산 파트너 슬롯

XRP의 파트너가 BTC로 고정돼 있는데, 이건 "BTC 원본이 ETH를 넣었으니 XRP는 BTC"라는
**기계적 상속**이지 측정된 선택이 아니다. 2026-09-03 진단에서 XRP~ETH 동시상관 0.6683이
XRP~BTC 0.6399를 이겼다(6개 지연 중 5개에서 ETH 우세). 같은 격자를 **파트너 BTC / ETH**
두 번 돌려 증거신호 조건부 lift가 갈리는지 본다.

⚠️Phase 2(참 라벨의 조건화 가치)만 본다. Phase 3(학습가능성)과 3b(실배포형태)는 여기서
승자가 바뀌면 그때 다시 돌린다 -- ETH/BTC/XRP 모두 "Phase 2 승자 != Phase 3 승자"였으므로
Phase 2만으로 결론내면 안 된다.
⚠️HOLDOUT 미터치(VAL/OOS 창만).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_S = importlib.util.spec_from_file_location(
    "xrpreg", ROOT / "scripts/research_xrp_regime_label_conditional_lift_20260903.py")
_x = importlib.util.module_from_spec(_S)
_S.loader.exec_module(_x)

from features.elite import RegimeEngine                                    # noqa: E402
from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER, compute_signals  # noqa: E402
from research_eth_regime_label_conditional_lift_20260902 import N_NULL, seg_lift    # noqa: E402
from research_eth_regime_scalping_label_geometry_20260902 import (        # noqa: E402
    _debounce, efficiency_ratio, scaled_label,
)

SCALES_EXT = (6, 12, 24, 48, 96, 192)
DEBOUNCES_EXT = (1, 3, 6, 9, 12)
PARTNERS = {"BTC": _x.PARTNER_KLINES, "ETH": _x.ETH_KLINES}
OUT = ROOT / "data/research/xrp_regime_label_grid_extension_20260903.json"


def log(m): print(f"[xrp-grid] {m}", flush=True)


def build_variants(close):
    """XRP Phase 2와 **동일한** 라벨 생성 규칙, 격자만 넓힌다."""
    rate1 = float((efficiency_ratio(close, 24) >= 0.20).mean())
    rate2 = float((efficiency_ratio(close, 48) >= 0.16).mean())
    variants, meta = {}, {}
    for s in SCALES_EXT:
        t1 = float(efficiency_ratio(close, s).quantile(1.0 - rate1))
        t2 = float(efficiency_ratio(close, 2 * s).quantile(1.0 - rate2))
        y0 = scaled_label(close, s, t1, t2)
        for k in DEBOUNCES_EXT:
            y = y0 if k == 1 else _debounce(y0, k)
            name = f"S{s}_K{k}"
            variants[name] = y
            # lock-up 진단: 라벨 전환 횟수와 chop 비중
            flips = int((np.diff(y) != 0).sum())
            meta[name] = {"label_flip_rate": flips / max(1, len(y) - 1),
                          "chop_share": float((y == 2).mean()),
                          "bull_share": float((y == 0).mean()),
                          "bear_share": float((y == 1).mean())}
    return variants, meta


def evaluate(frame, variants, pivot_pos, windows, rng):
    rows = []
    for vname, y in variants.items():
        chop_all = (y == 2)
        for wname, wmask in windows.items():
            seg = chop_all & wmask
            for sname, _ in SIGNAL_ORDER:
                for side in ("bottom", "top"):
                    sig = frame[f"{side}_{sname}"].fillna(False).to_numpy()
                    l_all, n_all = seg_lift(sig, pivot_pos[side], wmask)
                    l_chop, n_chop = seg_lift(sig, pivot_pos[side], seg)
                    if not (np.isfinite(l_all) and np.isfinite(l_chop)) or l_all <= 0:
                        continue
                    imp = l_chop / l_all - 1.0
                    null = []
                    for _ in range(N_NULL):
                        lb, _n = seg_lift(sig, pivot_pos[side],
                                          np.roll(chop_all, int(rng.integers(1, len(y)))) & wmask)
                        if np.isfinite(lb):
                            null.append(lb / l_all - 1.0)
                    p95 = float(np.percentile(null, 95)) if len(null) >= 50 else float("nan")
                    rows.append({"variant": vname, "window": wname, "signal": sname, "side": side,
                                 "n_all": n_all, "n_chop": n_chop,
                                 "improvement": imp,
                                 "beats_null95": bool(np.isfinite(p95) and imp > p95)})
    return pd.DataFrame(rows)


def summarize(df, variants):
    out = {}
    for v in variants:
        sub = df[df["variant"] == v]
        p = sub.pivot_table(index=["signal", "side"], columns="window", values="improvement")
        if "VAL" not in p or "OOS" not in p:
            continue
        both = int(((p["VAL"] > 0) & (p["OOS"] > 0)).sum())
        out[v] = {"both_positive": both, "n_cells": int(len(p)),
                  "mean_val": float(p["VAL"].mean()), "mean_oos": float(p["OOS"].mean()),
                  "beats_null": int(sub["beats_null95"].sum())}
    return out


def main() -> int:
    t0 = time.time()
    raw = pd.read_csv(_x.XRP_KLINES, parse_dates=["timestamp"])
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    pivots = _x.build_xrp_pivots()
    funding = _x.load_xrp_funding_z()
    log(f"XRP klines {len(raw):,} | pivots {len(pivots):,}")
    log(f"격자 확장: SCALES {SCALES_EXT}  x  DEBOUNCES {DEBOUNCES_EXT} = "
        f"{len(SCALES_EXT)*len(DEBOUNCES_EXT)}종")

    rep = {"scales": list(SCALES_EXT), "debounces": list(DEBOUNCES_EXT),
           "prev_winner": "S48_K6", "prev_grid": {"scales": [6, 12, 24, 48], "debounces": [1, 3, 6]},
           "holdout_touched": False, "partners": {}}

    for pname, ppath in PARTNERS.items():
        log("")
        log("#" * 66)
        log(f"교차자산 파트너 = {pname}")
        log("#" * 66)
        partner = pd.read_csv(ppath, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
        frame = compute_signals(raw, btc_df=partner, funding_df=funding)
        ts, close = frame["timestamp"], frame["close"]
        windows = {"VAL": ((ts >= _x.EV_VAL_START) & (ts <= _x.EV_VAL_END)).to_numpy(),
                   "OOS": ((ts >= _x.EV_OOS_START) & (ts <= _x.EV_OOS_END)).to_numpy()}
        windows["POOLED"] = windows["VAL"] | windows["OOS"]
        pivot_pos = {s: frame.index[frame["timestamp"].isin(
            pivots.loc[pivots["pivot_type"] == s, "timestamp"])].to_numpy()
            for s in ("bottom", "top")}

        variants, meta = build_variants(close)
        # REF도 같이 (기준선)
        ref = frame.copy()
        ref["mtf_trend_1h"] = close.ewm(span=12, adjust=False).mean().pct_change().fillna(0.0)
        lab = RegimeEngine().compute(ref)
        y_ref = np.full(len(frame), 2, dtype=int)
        y_ref[lab["regime_bull"].to_numpy() > 0] = 0
        y_ref[lab["regime_bear"].to_numpy() > 0] = 1
        variants["REF_RegimeEngine"] = y_ref
        meta["REF_RegimeEngine"] = {
            "label_flip_rate": float((np.diff(y_ref) != 0).sum()) / max(1, len(y_ref) - 1),
            "chop_share": float((y_ref == 2).mean()),
            "bull_share": float((y_ref == 0).mean()),
            "bear_share": float((y_ref == 1).mean())}

        rng = np.random.default_rng(20260903)
        df = evaluate(frame, variants, pivot_pos, windows, rng)
        summ = summarize(df, variants)

        ranked = sorted(summ.items(), key=lambda kv: (-kv[1]["both_positive"], -kv[1]["mean_oos"]))
        log("")
        log(f"{'라벨':<18} {'양쪽창양수':>10} {'meanVAL':>9} {'meanOOS':>9} "
            f"{'라벨전환율':>10} {'chop비중':>9}")
        for v, s in ranked[:14]:
            m = meta[v]
            mark = " ⭐" if v == "S48_K6" else ""
            log(f"{v:<18} {s['both_positive']:>4}/{s['n_cells']:<5} {s['mean_val']:>+9.4f} "
                f"{s['mean_oos']:>+9.4f} {m['label_flip_rate']:>10.4f} "
                f"{m['chop_share']:>9.3f}{mark}")

        top = ranked[0][0]
        prev = summ.get("S48_K6")
        log("")
        log(f"  ⇒ 확장 격자 1위: **{top}** "
            f"(양쪽창 {summ[top]['both_positive']}/{summ[top]['n_cells']}, "
            f"OOS {summ[top]['mean_oos']:+.4f})")
        if prev:
            log(f"     기존 승자 S48_K6: 양쪽창 {prev['both_positive']}/{prev['n_cells']}, "
                f"OOS {prev['mean_oos']:+.4f}")
        # 경계 재점검
        if top.startswith("S"):
            s_ = int(top.split("_")[0][1:]); k_ = int(top.split("_K")[1])
            at_edge = []
            if s_ == SCALES_EXT[-1]: at_edge.append("S 상단")
            if s_ == SCALES_EXT[0]: at_edge.append("S 하단")
            if k_ == DEBOUNCES_EXT[-1]: at_edge.append("K 상단")
            if k_ == DEBOUNCES_EXT[0]: at_edge.append("K 하단")
            log(f"     경계 점검: {'⚠️여전히 경계(' + ', '.join(at_edge) + ') -- 더 넓혀야 함' if at_edge else '✅내부값 -- 격자 충분'}")
        rep["partners"][pname] = {"summary": summ, "meta": meta, "top": top,
                                  "top_at_edge": bool(at_edge) if top.startswith("S") else None}

    log("")
    log("=== 파트너 비교 (같은 라벨, 파트너만 다름) ===")
    b, e = rep["partners"]["BTC"]["summary"], rep["partners"]["ETH"]["summary"]
    common = sorted(set(b) & set(e), key=lambda v: -max(b[v]["both_positive"], e[v]["both_positive"]))
    log(f"{'라벨':<18} {'BTC 양쪽창':>11} {'ETH 양쪽창':>11}  {'BTC OOS':>9} {'ETH OOS':>9}  승")
    wins = {"BTC": 0, "ETH": 0}
    for v in common[:12]:
        w = "BTC" if b[v]["both_positive"] > e[v]["both_positive"] else (
            "ETH" if e[v]["both_positive"] > b[v]["both_positive"] else "=")
        if w in wins:
            wins[w] += 1
        log(f"{v:<18} {b[v]['both_positive']:>6}/{b[v]['n_cells']:<4} "
            f"{e[v]['both_positive']:>6}/{e[v]['n_cells']:<4}  "
            f"{b[v]['mean_oos']:>+9.4f} {e[v]['mean_oos']:>+9.4f}  {w}")
    log(f"  ⇒ 상위 12 라벨 중 BTC 우세 {wins['BTC']} / ETH 우세 {wins['ETH']}")
    rep["partner_head_to_head"] = wins

    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
