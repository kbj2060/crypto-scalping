#!/usr/bin/env python3
"""XRP 레짐 확장격자 후보(S96_K9 등)의 **Phase 3 + 3b** -- 소진된 창을 건드리지 않는 분할로.

## 왜

2026-09-03 격자 확장에서 배포본 `S48_K6`이 크게 밀렸다(Phase 2, 증거신호 VAL/OOS 창):

    S96_K9   양쪽창 15/16   mean OOS +0.1686
    S96_K12  양쪽창 15/16   mean OOS +0.1656
    S192_K1  양쪽창 14/16   mean OOS +0.1775
    S48_K6   양쪽창 10/16   mean OOS +0.0652   <- 현재 배포

⚠️그러나 **Phase 2 승자 != Phase 3 승자**가 이 저장소의 반복 관측이다. ETH·BTC·XRP 셋 다
"후보 라벨은 학습이 더 어렵지만 실배포형태(예측-chop 게이팅)에서는 이긴다"였다.
Phase 2만 보고 라벨을 바꾸면 안 된다.

## ⚠️창 규율 -- 원본 Phase 3의 OOS는 이미 소진됐다

`research_xrp_regime_s48k6_label_train_20260903.py`는 **2026-07-01~2026-08-01**을
"FIRST OOS LOOK"으로 1회 소진했다(S48_K6 채택 근거). 새 라벨로 그 창을 다시 보면
같은 창에 대한 두 번째 모델선택이 되어 선택편향이 들어간다.

⇒ 여기서는 그 창을 **아예 건드리지 않는다.**

    학습:   2024-01-01 ~ 2025-08-31   (증거신호 VAL 시작 직전까지)
    평가:   2025-09-01 ~ 2026-03-31   (증거신호 VAL+OOS = 학습 밖)
    미사용: 2026-04-01 이후 전부       (HOLDOUT + 소진된 Phase3 창)

원본보다 학습기간이 짧아 절대 bal_acc는 낮게 나온다. **라벨 간 비교**가 목적이므로
모든 라벨에 같은 분할을 쓴다.

## 산출

  Phase 3  -- 학습가능성: bal_acc / chop recall·precision / pred_flip
  Phase 3b -- 실배포형태: 예측-chop 게이팅 시 증거신호 조건부 lift 개선(VAL/OOS 분리)
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sklearn.ensemble import HistGradientBoostingClassifier                 # noqa: E402

_S = importlib.util.spec_from_file_location(
    "xrpp3", ROOT / "scripts/research_xrp_regime_s48k6_label_train_20260903.py")
_p = importlib.util.module_from_spec(_S)
_S.loader.exec_module(_p)

from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER, compute_signals  # noqa: E402
from research_eth_regime_label_conditional_lift_20260902 import N_NULL, seg_lift   # noqa: E402
from research_eth_regime_scalping_label_geometry_20260902 import (           # noqa: E402
    _debounce, efficiency_ratio, scaled_label,
)

OUT = ROOT / "data/research/xrp_regime_extended_label_phase3_clean_20260903.json"

FIT_START = pd.Timestamp("2024-01-01T00:00:00")
FIT_END = pd.Timestamp("2025-08-31T23:55:00")      # 증거신호 VAL 시작 직전
EVAL_START = pd.Timestamp("2025-09-01T00:00:00")
EVAL_END = pd.Timestamp("2026-03-31T23:55:00")     # HOLDOUT 직전
UNTOUCHED_FROM = pd.Timestamp("2026-04-01T00:00:00")

# 확장격자 상위 + 현 배포 + REF
CANDIDATES = [(48, 6), (96, 9), (96, 12), (192, 3), (192, 1)]
DEPLOYED = (48, 6)


def log(m): print(f"[p3clean] {m}", flush=True)


def make_label(close: pd.Series, fit_mask: np.ndarray, scale: int, k: int) -> np.ndarray:
    """원본 `s24k3_label`과 **동일한** 규칙. 임계는 학습구간에서만 뽑는다."""
    c_fit = close[fit_mask]
    rate1 = float((efficiency_ratio(c_fit, 24) >= 0.20).mean())
    rate2 = float((efficiency_ratio(c_fit, 48) >= 0.16).mean())
    t1 = float(efficiency_ratio(c_fit, scale).quantile(1.0 - rate1))
    t2 = float(efficiency_ratio(c_fit, 2 * scale).quantile(1.0 - rate2))
    y0 = scaled_label(close, scale, t1, t2)
    return y0 if k == 1 else _debounce(y0, k)


def main() -> int:
    t0 = time.time()
    _p._assert_xrp_canon()
    payload = joblib.load(_p.GBM3_MODEL_PATH)
    feat_cols, medians = payload["feature_cols"], payload["feature_medians"]
    df = _p.load_btc_frame(feat_cols)          # 이름만 btc -- XRP canonical을 읽는다(가드 통과함)
    ts = df["timestamp"]

    fit = ((ts >= FIT_START) & (ts <= FIT_END)).to_numpy()
    ev = ((ts >= EVAL_START) & (ts <= EVAL_END)).to_numpy()
    untouched = (ts >= UNTOUCHED_FROM).to_numpy()
    log(f"XRP canonical {len(df):,}행 | 학습 {int(fit.sum()):,} / 평가 {int(ev.sum()):,} "
        f"| 미사용(2026-04+) {int(untouched.sum()):,}")
    log("⚠️소진된 Phase3 창(2026-07~08)과 HOLDOUT은 이 스크립트에서 한 번도 읽지 않는다")

    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    for c in feat_cols:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(medians.get(c, 0.0))

    close = df["close"]
    labels = {"REF_RegimeEngine": _p.deployed_label(df)}
    for s, k in CANDIDATES:
        labels[f"S{s}_K{k}"] = make_label(close, fit, s, k)

    log("")
    log(f"{'라벨':<18} {'bal_acc':>8} {'chop_R':>8} {'chop_P':>8} {'bull_R':>8} "
        f"{'bear_R':>8} {'pred_flip':>10}")
    results, gate_pred = {}, {}
    for lname, y in labels.items():
        m = HistGradientBoostingClassifier(random_state=_p.SEED, **_p.GBM3_HP).fit(
            x.loc[fit, feat_cols], y[fit])
        pred_ev = m.predict(x.loc[ev, feat_cols])
        r = _p.evaluate(y[ev], pred_ev)
        results[lname] = r
        gate_pred[lname] = pd.DataFrame({"timestamp": ts, "pred": m.predict(x[feat_cols])})
        mark = " ⭐배포" if lname == f"S{DEPLOYED[0]}_K{DEPLOYED[1]}" else ""
        log(f"{lname:<18} {r['balanced_accuracy']:>8.4f} {r['chop_recall']:>8.4f} "
            f"{r['chop_precision']:>8.4f} {r['bull_recall']:>8.4f} {r['bear_recall']:>8.4f} "
            f"{r['flip_rate']:>10.4f}{mark}")

    # ---------- Phase 3b: 예측-chop 게이팅 ----------
    log("")
    log("Phase 3b -- 예측-chop 게이팅 시 증거신호 조건부 lift 개선")
    raw = pd.read_csv(_p.XRP_KLINES, parse_dates=["timestamp"])
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    partner = pd.read_csv(_p.PARTNER_KLINES, usecols=["timestamp", "high", "low"],
                          parse_dates=["timestamp"])
    frame = compute_signals(raw, btc_df=partner, funding_df=_p.load_xrp_funding_z())
    pivots = _p.build_xrp_pivots()
    ts_e = frame["timestamp"]
    windows = {"VAL": ((ts_e >= _p.EV_VAL_START) & (ts_e <= _p.EV_VAL_END)).to_numpy(),
               "OOS": ((ts_e >= _p.EV_OOS_START) & (ts_e <= _p.EV_OOS_END)).to_numpy()}
    pivot_pos = {s: frame.index[frame["timestamp"].isin(
        pivots.loc[pivots["pivot_type"] == s, "timestamp"])].to_numpy() for s in ("bottom", "top")}

    rng = np.random.default_rng(20260903)
    gate = {}
    for lname, pf in gate_pred.items():
        merged = frame[["timestamp"]].merge(pf, on="timestamp", how="left")
        chop = (merged["pred"].to_numpy() == 2)
        rows = []
        for wname, wmask in windows.items():
            seg = chop & wmask
            for sname, _ in SIGNAL_ORDER:
                for side in ("bottom", "top"):
                    sig = frame[f"{side}_{sname}"].fillna(False).to_numpy()
                    l_all, _n = seg_lift(sig, pivot_pos[side], wmask)
                    l_chop, n_chop = seg_lift(sig, pivot_pos[side], seg)
                    if not (np.isfinite(l_all) and np.isfinite(l_chop)) or l_all <= 0:
                        continue
                    rows.append({"window": wname, "signal": sname, "side": side,
                                 "improvement": l_chop / l_all - 1.0, "n_chop": n_chop})
        d = pd.DataFrame(rows)
        if not len(d):
            gate[lname] = {"error": "no cells"}
            continue
        p = d.pivot_table(index=["signal", "side"], columns="window", values="improvement")
        both = int(((p.get("VAL", pd.Series(dtype=float)) > 0)
                    & (p.get("OOS", pd.Series(dtype=float)) > 0)).sum()) if "VAL" in p and "OOS" in p else 0
        gate[lname] = {"both_positive": both, "n_cells": int(len(p)),
                       "mean_val": float(p["VAL"].mean()) if "VAL" in p else float("nan"),
                       "mean_oos": float(p["OOS"].mean()) if "OOS" in p else float("nan"),
                       "pred_chop_share": float(chop.mean())}

    log("")
    log(f"{'라벨':<18} {'양쪽창양수':>10} {'meanVAL':>9} {'meanOOS':>9} {'예측chop비중':>12}")
    for lname in labels:
        g = gate.get(lname, {})
        if "error" in g:
            log(f"{lname:<18} {g['error']}"); continue
        mark = " ⭐배포" if lname == f"S{DEPLOYED[0]}_K{DEPLOYED[1]}" else ""
        log(f"{lname:<18} {g['both_positive']:>4}/{g['n_cells']:<5} {g['mean_val']:>+9.4f} "
            f"{g['mean_oos']:>+9.4f} {g['pred_chop_share']:>12.3f}{mark}")

    dep = f"S{DEPLOYED[0]}_K{DEPLOYED[1]}"
    ranked = sorted((k for k in gate if "error" not in gate[k]),
                    key=lambda k: (-gate[k]["both_positive"], -gate[k]["mean_oos"]))
    top = ranked[0]
    log("")
    log(f"⇒ Phase 3b 1위: **{top}** (양쪽창 {gate[top]['both_positive']}/{gate[top]['n_cells']}, "
        f"OOS {gate[top]['mean_oos']:+.4f})")
    log(f"   현재 배포 {dep}: 양쪽창 {gate[dep]['both_positive']}/{gate[dep]['n_cells']}, "
        f"OOS {gate[dep]['mean_oos']:+.4f}")
    log(f"   ⇒ {'⚠️**배포 라벨을 바꿀 근거가 있다**' if top != dep else '✅현행 유지'}")

    rep = {"fit": [str(FIT_START), str(FIT_END)], "eval": [str(EVAL_START), str(EVAL_END)],
           "untouched_from": str(UNTOUCHED_FROM),
           "spent_phase3_window_touched": False, "holdout_touched": False,
           "candidates": [f"S{s}_K{k}" for s, k in CANDIDATES], "deployed": dep,
           "phase3": results, "phase3b": gate, "top": top,
           "change_recommended": bool(top != dep),
           "runtime_sec": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
