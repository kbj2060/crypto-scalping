#!/usr/bin/env python3
"""XRP Phase 3 + 3b -- is the S48_K6 candidate label learnable, and does it make a better gate once
modelled? Counterpart of research_eth_regime_s12k3_label_train_20260902.py + ..._predicted_gate_lift.

XRP LINEAGE
  Phase 1 (research_btc_regime_scalping_label_geometry_20260902.py): transition-edge axis closed on
    BTC too -- 0/16 cells cleared zero (ETH: 1/16, itself chance).
  Phase 2 (research_xrp_regime_label_conditional_lift_20260903.py): S48_K6 selected. Among
    adequately-sampled cells (n_chop>=150) it is 7/13 both-window-positive, all seven of them
    large-sample, and it has the highest OOS null-beating count (6/16) of any variant.
    ⭐BTC's winner is the ORIGINAL RegimeEngine scale (S=24) plus a K=3 confirm -- ETH's S=12 scores
    only 3/10 on BTC. Shortening the scale is not what BTC wanted; the debounce is. This is exactly
    why the grid was re-screened rather than ported (btc_v_rebound_feeder_gap_threshold_screen_
    20260901 precedent: ETH parameters did not carry over and Kalman had to be dropped on BTC).

⚠️ PHASE 3 SPENDS BTC'S FIRST OOS LOOK (2026-07-01~2026-08-01, 9,141 bars -- BTC's canonical feature
file ends 2026-08-01 17:40, so it is ~32d vs ETH's ~50d). Unlike ETH's window this one has NOT been
consumed by prior regime rounds, so it is comparatively fresh -- all the more reason this is the
only look taken here. Research/dev score, not a promotion claim.

Checks carried over from the ETH study: chop precision/recall are the PRIMARY metrics (not bal_acc,
which direction-proxy features can carry), a direction-proxy ablation quantifies that, and the
reference label is retrained by this same script so the comparison is like-for-like. Phase 3b then
measures the quantity that actually ships -- PREDICTED-chop gated conditional lift.

## ⚠️XRP 포팅 (2026-09-03)

BTC Phase 3의 자산 상수만 바꾼 포팅. **라벨은 XRP 자신의 Phase 2 승자 S48_K6**이다.
ETH S12_K3 / BTC S24_K3 — 세 자산이 전부 다르고, BTC 승자는 XRP에서 3/16으로 거의 최하위였다.
경로/펀딩/피벗은 XRP Phase 2 모듈에서 import하므로 그쪽 수정이 여기에도 반영된다.
교차자산 파트너 슬롯엔 BTC를 넣는다(`btc_df`는 인자 이름일 뿐).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import OOS_END as EV_OOS_END  # noqa: E402
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    OOS_START as EV_OOS_START, VAL_END as EV_VAL_END, VAL_START as EV_VAL_START,
)
from features.elite import RegimeEngine  # noqa: E402
from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER, compute_signals  # noqa: E402
from research_xrp_regime_label_conditional_lift_20260903 import (  # noqa: E402
    XRP_KLINES, PARTNER_KLINES, build_xrp_pivots, load_xrp_funding_z,
)
from research_eth_regime_label_conditional_lift_20260902 import seg_lift, K_HORIZON  # noqa: E402
from research_eth_regime_scalping_label_geometry_20260902 import (  # noqa: E402
    _debounce, _run_lengths, efficiency_ratio, scaled_label,
)
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402

# ⚠️⚠️2026-09-03: 여기가 **BTC 캐노니컬을 가리키고 있었다**(변수명만 XRP_CANON).
# 1차 Phase 3 결과(bal_acc 0.8459, 게이트 9/16)는 전부 BTC 데이터 위에서 나온 것이라 무효였다.
# TRAIN 262,656 + OOS 9,216 = 271,872가 XRP 1year 파일(224,245행)로는 불가능해서 발각.
# 동결 컨텍스트에서 잡았던 것과 **같은 오염 계열**이다 -- 포팅은 변수명을 바꿔도 경로를 남긴다.
XRP_CANON = ROOT / "data/splits/year_oos/xrp_features_2024_2026.csv"
EXPECTED_CANON_ROWS = 272_490      # 자산 오염 가드
GBM3_MODEL_PATH = ROOT / "tmp/eth_regime_gbm3_independent_20260826/model.joblib"  # feature_cols source
GBM3_HP = dict(max_depth=10, learning_rate=0.04, max_iter=400, l2_regularization=2.0)
SEED = 7529
TRAIN_START = pd.Timestamp("2024-01-01T00:00:00")
TRAIN_END = pd.Timestamp("2026-06-30T23:55:00")
OOS_START = pd.Timestamp("2026-07-01T00:00:00")
OOS_END = pd.Timestamp("2026-08-01T23:55:00")
SCALE, DEBOUNCE_K = 48, 6   # ⭐XRP Phase2 승자 S48_K6 (ETH S12_K3 / BTC S48_K6과 전부 다름)
CLASSES3 = ["bull", "bear", "chop"]
DIRECTION_PROXIES = ["vwap_dist_24", "kalman_velocity", "rsi", "hma_slope", "mtf_trend_4h"]
OUT_DIR = ROOT / "tmp/xrp_regime_s48k6_label_train_20260903"


def load_btc_frame(feat_cols: list[str]) -> pd.DataFrame:
    need = ["timestamp", "open", "high", "low", "close", "volume"]
    have = set(pd.read_csv(XRP_CANON, nrows=1).columns)
    use = need + [c for c in feat_cols if c in have and c not in need]
    df = pd.read_csv(XRP_CANON, usecols=use, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    df = df[(df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= OOS_END)].reset_index(drop=True)
    return _with_raw_state12(df)


def _assert_xrp_canon() -> None:
    """⭐자산 오염 가드 -- 다른 자산 캐노니컬을 읽으면 여기서 죽는다."""
    n = sum(1 for _ in open(XRP_CANON)) - 1
    if abs(n - EXPECTED_CANON_ROWS) > 200:
        raise RuntimeError(f"{XRP_CANON.name}: {n:,}행 != XRP 기대치 {EXPECTED_CANON_ROWS:,} "
                           f"-- 다른 자산 데이터일 가능성")


def deployed_label(df: pd.DataFrame) -> np.ndarray:
    d = df.copy()
    if "mtf_trend_1h" not in d.columns:
        d["mtf_trend_1h"] = d["close"].ewm(span=12, adjust=False).mean().pct_change().fillna(0.0)
    lab = RegimeEngine().compute(d)
    y = np.full(len(df), 2, dtype=int)
    y[lab["regime_bull"].to_numpy() > 0] = 0
    y[lab["regime_bear"].to_numpy() > 0] = 1
    return y


def s24k3_label(df: pd.DataFrame, train_mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Thresholds percentile-matched on TRAIN ONLY to RegimeEngine's own firing rates there."""
    close = df["close"]
    r1 = float((efficiency_ratio(close, 24)[train_mask] >= 0.20).mean())
    r2 = float((efficiency_ratio(close, 48)[train_mask] >= 0.16).mean())
    t1 = float(efficiency_ratio(close, SCALE)[train_mask].quantile(1.0 - r1))
    t2 = float(efficiency_ratio(close, 2 * SCALE)[train_mask].quantile(1.0 - r2))
    return _debounce(scaled_label(close, SCALE, t1, t2), DEBOUNCE_K), t1, t2


def evaluate(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    runs = _run_lengths(pred)
    out = {"balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
           "flip_rate": round(float(np.mean(pred[1:] != pred[:-1])), 4),
           "median_run_bars": float(np.median(runs)) if runs else 0.0}
    for i, n in enumerate(CLASSES3):
        sup, prd = cm[i].sum(), cm[:, i].sum()
        out[f"{n}_recall"] = round(float(cm[i, i] / sup), 4) if sup else None
        out[f"{n}_precision"] = round(float(cm[i, i] / prd), 4) if prd else None
    return out


def main() -> None:
    _assert_xrp_canon()
    payload = joblib.load(GBM3_MODEL_PATH)
    feat_cols, medians = payload["feature_cols"], payload["feature_medians"]
    print(f"feature_cols from the ETH GBM3 artifact: {len(feat_cols)} (BTC canonical carries the same set)")
    print("⚠️ Phase 3 spends BTC's FIRST OOS look on 2026-07-01~2026-08-01.\n")

    df = load_btc_frame(feat_cols)
    ts = df["timestamp"]
    tr = ((ts >= TRAIN_START) & (ts <= TRAIN_END)).to_numpy()
    oos = ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()
    miss = [c for c in feat_cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"missing features after _with_raw_state12: {miss}")
    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    for c in feat_cols:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(medians.get(c, 0.0))

    y_new, t1, t2 = s24k3_label(df, tr)
    labels = {"REF_RegimeEngine": deployed_label(df), f"S{SCALE}_K{DEBOUNCE_K}": y_new}
    print(f"TRAIN {int(tr.sum()):,} / OOS {int(oos.sum()):,} bars | S48_K6 T1={t1:.6f} T2={t2:.6f}")
    for n, y in labels.items():
        print(f"  {n:18s} TRAIN shares " +
              " ".join(f"{c}={np.mean(y[tr]==i):.3f}" for i, c in enumerate(CLASSES3)))

    feature_sets = {"full136": feat_cols, "ablated": [c for c in feat_cols if c not in DIRECTION_PROXIES]}
    results, gate_models = {}, {}
    print()
    for lname, y in labels.items():
        for fname, cols in feature_sets.items():
            m = HistGradientBoostingClassifier(random_state=SEED, **GBM3_HP).fit(x.loc[tr, cols], y[tr])
            r = evaluate(y[oos], m.predict(x.loc[oos, cols]))
            results[f"{lname}__{fname}"] = r
            print(f"{lname:18s} {fname:8s} bal_acc={r['balanced_accuracy']:.4f} "
                  f"chop_R={r['chop_recall']:.4f} chop_P={r['chop_precision']:.4f} "
                  f"bull_R={r['bull_recall']:.4f} bear_R={r['bear_recall']:.4f} "
                  f"pred_flip={r['flip_rate']:.4f}")
            if fname == "full136":
                gate_models[lname] = pd.DataFrame({"timestamp": ts, "pred": m.predict(x[cols])})

    print("\n=== direction-proxy ablation (full136 -> ablated) ===")
    for lname in labels:
        a, b = results[f"{lname}__full136"], results[f"{lname}__ablated"]
        print(f"  {lname:18s} bal_acc {a['balanced_accuracy']:.4f} -> {b['balanced_accuracy']:.4f} "
              f"({b['balanced_accuracy']-a['balanced_accuracy']:+.4f}) | chop_P "
              f"{a['chop_precision']:.4f} -> {b['chop_precision']:.4f} "
              f"({b['chop_precision']-a['chop_precision']:+.4f})")

    # ---- Phase 3b: predicted-chop gate on the BTC evidence window (no extra OOS spend) ----
    raw = pd.read_csv(XRP_KLINES, parse_dates=["timestamp"])
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    partner = pd.read_csv(PARTNER_KLINES, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    frame = compute_signals(raw, btc_df=partner, funding_df=load_xrp_funding_z())
    pivots = build_xrp_pivots()
    ts_e = frame["timestamp"]
    windows = {"VAL": ((ts_e >= EV_VAL_START) & (ts_e <= EV_VAL_END)).to_numpy(),
               "OOS": ((ts_e >= EV_OOS_START) & (ts_e <= EV_OOS_END)).to_numpy()}
    windows["POOLED"] = windows["VAL"] | windows["OOS"]
    pivot_pos = {s: frame.index[frame["timestamp"].isin(
        pivots.loc[pivots["pivot_type"] == s, "timestamp"])].to_numpy() for s in ("bottom", "top")}

    rows = []
    for lname, pf in gate_models.items():
        merged = frame[["timestamp"]].merge(pf, on="timestamp", how="left")
        gate = (merged["pred"] == 2).fillna(False).to_numpy()
        for wname, wmask in windows.items():
            for sname, _ in SIGNAL_ORDER:
                for side in ("bottom", "top"):
                    sig = frame[f"{side}_{sname}"].fillna(False).to_numpy()
                    l_all, _ = seg_lift(sig, pivot_pos[side], wmask)
                    l_g, n_g = seg_lift(sig, pivot_pos[side], gate & wmask)
                    if not (np.isfinite(l_all) and np.isfinite(l_g)) or l_all <= 0:
                        continue
                    rows.append({"label": lname, "window": wname, "ss": f"{sname}.{side}",
                                 "n_gated": n_g, "improvement": round(l_g / l_all - 1.0, 4)})
    gl = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gl.to_csv(OUT_DIR / "predicted_gate_lift.csv", index=False)
    (OUT_DIR / "report.json").write_text(json.dumps(
        {"config": {"scale": SCALE, "debounce_k": DEBOUNCE_K, "T1": t1, "T2": t2, "seed": SEED,
                    "train": f"{TRAIN_START}~{TRAIN_END}", "oos": f"{OOS_START}~{OOS_END}"},
         "oos_disclosure": "BTC's first OOS look; 9,141 bars (~32d); research/dev score only.",
         "results": results}, indent=2, ensure_ascii=False))

    pd.set_option("display.width", 200)
    print("\n=== Phase 3b: PREDICTED-chop gated lift improvement (what actually ships) ===")
    print(gl.groupby(["label", "window"]).agg(
        cells=("improvement", "size"), mean=("improvement", "mean"),
        pos=("improvement", lambda s: int((s > 0).sum()))).round(4).to_string())
    print("\n=== both-window-positive cells ===")
    for lname in labels:
        p = gl[gl["label"] == lname].pivot_table(index="ss", columns="window", values="improvement")
        if "VAL" in p and "OOS" in p:
            print(f"  {lname:18s} {int(((p['VAL']>0)&(p['OOS']>0)).sum())}/{len(p)}"
                  f"  | mean VAL {p['VAL'].mean():+.4f} OOS {p['OOS'].mean():+.4f}")
    print(f"\nWrote {OUT_DIR}/")


if __name__ == "__main__":
    main()
