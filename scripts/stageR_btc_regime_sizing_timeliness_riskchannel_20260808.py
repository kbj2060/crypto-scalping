"""Stage R — is the sizing overlay's broken risk channel just DETECTION LAG?  (2026-08-08)

Contract: docs/experiments/btc_regime_sizing_timeliness_riskchannel_20260808.json

The czz_trend sizing overlay was downgraded to NOT-SUPPORTED because its risk channel points the
wrong way: variance ratio bear/bull = 0.881, i.e. bear-labelled trades are LESS volatile, so
downsizing bear cannot cut drawdown through risk.  But the detector it reads (czz4 on 5m) has a
theta=4% detection lag of 985 minutes -- about 16.4 hours.  A label that late attaches "bear" to
the calm tail of a down-wave while the volatile early part is still labelled "bull", which would
produce exactly the ratio that was measured.

So before anything is re-sized or re-measured for PnL, this asks the one question that decides
whether the overlay has a mechanism at all:

    does a FASTER theta=4% detector flip the variance ratio bear/bull above 1?

Stage 0 (docs/experiments/btc_regime_bar_timeframe_scale_20260808.json) measured that coarse grids
detect 4% turns earlier -- 5m 985 min, 15m 892.5, 30m 812.5, 1h 847.5 -- at a cost of only
1-2pp agreement.  Those same grids are the arms here.  Nothing is learned, nothing is sized, no
PnL is read: trades are simply relabelled by each detector at their entry-signal bar and the risk
channel is measured.  Failing this closes the overlay for a MECHANISM reason rather than an
effect-size one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from chart_btc_jm_regime_verification_20260808 import causal_zigzag  # noqa: E402
from refine_btc_regime_classifier_theta005_20260808 import PANEL_PATH  # noqa: E402
from stage0_btc_regime_bar_timeframe_scale_20260808 import resample_close  # noqa: E402

OUT_DIR = ROOT / "tmp/btc_regime_sizing_timeliness_20260808"
THETA = 0.04
BAR_SIZES = {"5m": 1, "15m": 3, "30m": 6, "1h": 12}
LEDGERS = {
    "single_slot": {
        "validation": ROOT / "tmp/btc_regime_sizing_overlay_20260808/validation_ledger_identity.csv",
        "oos": ROOT / "tmp/btc_regime_sizing_overlay_20260808/oos_ledger_identity.csv",
    },
    "n3_m1.5": {
        "validation": ROOT / "tmp/btc_multislot_margin_resweep_20260808/validation_ledger_n3_m1.5.csv",
        "oos": ROOT / "tmp/btc_multislot_margin_resweep_20260808/oos_ledger_n3_m1.5.csv",
    },
}
VAR_RATIO_BAR, BF_P_BAR, MIN_BEAR_OCCUPANCY = 1.0, 0.10, 0.15
INCUMBENT_RATIO = 0.881  # what the downgrade audit measured on 5m czz4


def risk_channel(ret: np.ndarray, regime: np.ndarray) -> dict:
    bear, bull = ret[regime == -1], ret[regime == 1]
    n_b, n_u = len(bear), len(bull)
    out: dict = {"n_bear": n_b, "n_bull": n_u, "n_total": len(ret),
                 "bear_occupancy": round(n_b / max(len(ret), 1), 3)}
    if n_b < 5 or n_u < 5:
        return out | {"insufficient": True}
    vb, vu = float(np.var(bear, ddof=1)), float(np.var(bull, ddof=1))
    out["var_ratio_bear_over_bull"] = round(vb / vu, 3) if vu > 0 else None
    out["brown_forsythe_p"] = round(float(stats.levene(bear, bull, center="median").pvalue), 3)
    t, p = stats.ttest_ind(bear, bull, equal_var=False)
    pooled = np.sqrt((vb * (n_b - 1) + vu * (n_u - 1)) / max(n_b + n_u - 2, 1))
    out |= {"mean_bear_pct": round(float(bear.mean()) * 100, 3),
            "mean_bull_pct": round(float(bull.mean()) * 100, 3),
            "welch_t": round(float(t), 2), "welch_p": round(float(p), 3),
            "cohens_d": round(float((bear.mean() - bull.mean()) / pooled), 3) if pooled > 0 else None}
    neg = ret < 0
    out["bear_share_of_losses"] = (round(float(ret[neg & (regime == -1)].sum() / ret[neg].sum()), 3)
                                   if neg.any() and ret[neg].sum() != 0 else None)
    out["bear_share_of_trades"] = round(n_b / len(ret), 3)
    return out


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH, low_memory=False, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts, close = panel["timestamp"], panel["close"].to_numpy(dtype=np.float64)

    states = {}
    for name, step in BAR_SIZES.items():
        c_close, idx_map = resample_close(ts, close, step)
        st = causal_zigzag(c_close, threshold=THETA)
        s5 = st[np.clip(idx_map, 0, len(st) - 1)]
        s5[idx_map < 0] = 0
        states[name] = pd.DataFrame({"timestamp": ts, "regime": s5.astype(np.int8)})
        print(json.dumps({"detector": name,
                          "bear_bar_share": round(float((s5 == -1).mean()), 3),
                          "bull_bar_share": round(float((s5 == 1).mean()), 3)}), flush=True)

    results: dict = {}
    for led_name, splits in LEDGERS.items():
        for split, path in splits.items():
            if not path.exists():
                print(json.dumps({"missing_ledger": str(path)}), flush=True)
                continue
            led = pd.read_csv(path)
            led["entry_timestamp"] = pd.to_datetime(led["entry_timestamp"])
            led = led.sort_values("entry_timestamp").reset_index(drop=True)
            ret = led["trade_return"].to_numpy(dtype=np.float64)
            for det, st in states.items():
                merged = pd.merge_asof(led[["entry_timestamp"]], st.rename(columns={"timestamp": "entry_timestamp"}),
                                       on="entry_timestamp", direction="backward",
                                       tolerance=pd.Timedelta("10min"))
                reg = merged["regime"].fillna(0).to_numpy(dtype=np.int64)
                key = f"{led_name}|{split}|{det}"
                results[key] = risk_channel(ret, reg)
                r = results[key]
                print(f"  {key:28} n {r['n_total']:4}  bear {r['n_bear']:3} ({r['bear_occupancy']:.2f})  "
                      f"varratio {r.get('var_ratio_bear_over_bull')}  BF p {r.get('brown_forsythe_p')}  "
                      f"meanB {r.get('mean_bear_pct')} vs {r.get('mean_bull_pct')}", flush=True)

    passing = []
    for key, r in results.items():
        if "|oos|" not in key or r.get("insufficient") or r.get("var_ratio_bear_over_bull") is None:
            continue
        if (r["var_ratio_bear_over_bull"] > VAR_RATIO_BAR and r["brown_forsythe_p"] < BF_P_BAR
                and r["bear_occupancy"] >= MIN_BEAR_OCCUPANCY):
            passing.append({"cell": key, **{k: r[k] for k in
                                            ("var_ratio_bear_over_bull", "brown_forsythe_p", "bear_occupancy")}})

    verdict = {
        "gate": f"OOS var_ratio > {VAR_RATIO_BAR} AND Brown-Forsythe p < {BF_P_BAR} AND bear occupancy >= {MIN_BEAR_OCCUPANCY}",
        "incumbent_5m_ratio_from_downgrade_audit": INCUMBENT_RATIO,
        "passing_cells": passing,
        "lag_artifact_hypothesis_supported": bool(passing),
        "meaning": ("faster detection flips the risk channel; the overlay reopens on a real mechanism "
                    "and a SECOND stage may then measure sizing/PnL"
                    if passing else
                    "the risk channel stays inverted at every detector speed, so the overlay's premise "
                    "is dead independently of lag — close for a MECHANISM reason, no PnL read spent"),
    }
    out = {"contract": "docs/experiments/btc_regime_sizing_timeliness_riskchannel_20260808.json",
           "theta": THETA, "kind": "no-learning relabelling; no sizing applied, no PnL read",
           "results": results, "verdict": verdict}
    (OUT_DIR / "stageR.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(verdict, indent=2), flush=True)
    print(f"wrote {OUT_DIR / 'stageR.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
