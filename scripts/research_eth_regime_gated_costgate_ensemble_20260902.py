#!/usr/bin/env python3
"""PART A -- does regime gating survive a COST GATE (PnL), not just a lift improvement?
2026-09-02, user: "그렇게 하고 증거신호들과 어떻게 앙상블을 해야할지 연구해줘. pnl 위주로".

WHY THIS IS THE RIGHT NEXT TEST. The Phase 1-3b regime study measured CONDITIONAL LIFT -- "does the
signal hit a swing pivot more often when the regime says chop". That is not PnL. This repo has a
direct precedent for the two coming apart: the 2026-08-27 chop-conditional study found +29~37% lift
improvements and then FAILED its cost gate 0/6 (loss reduction was real, but it never beat
max(always_long, always_short)). So the S12_K3 gate's +9.8% lift is a claim about signal quality,
not about money, until it clears this.

METHOD -- the established economics-gate harness, reused, with only the gate added
  * fires CSVs / simulate_single_position / MARGIN 0.30 x LEV 3.0 / ROUNDTRIP 10bp
  * SL(6) x ARM(4) x Trail(4) = 96 combos, purged_decision_mask VAL/OOS
  * ⭐DIRECTION-FLIP CONTROL ON THE WHOLE GRID -- not one config. README ss5.8; skipping this is
    exactly how fib_extension_exhaustion's economics claim survived a month before being withdrawn.
  * "genuine" = both windows positive AND beats its own flip in both AND the flip is negative in both

ARMS PER SIGNAL
  all         -- every fire (the deployed baseline)
  chop        -- only fires where the DEPLOYED S12_K3 model predicts chop (what would actually ship)
  nonchop     -- the complement, as a sanity control: if chop-gating is real, this should be worse

⚠️ HOLDOUT (>=2026-04-01) is excluded entirely -- all 8 signals have already spent theirs, so this
is a VAL/OOS research/dev score and cannot be a promotion claim.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import joblib
import numpy as np
import pandas as pd

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402

KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
REGIME_ARTIFACT = ROOT / "tmp/eth_regime_s12k3_20260902/model.joblib"
REGIME_CACHE = ROOT / "tmp/eth_regime_gated_costgate_20260902/regime_pred.parquet"
OUT_DIR = ROOT / "tmp/eth_regime_gated_costgate_20260902"

MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.001
SL_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ARM_GRID = [0.5, 1.0, 1.5, 2.0]
TRAIL_GRID = [0.1, 0.2, 0.3, 0.5]
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")
MIN_WINDOW_N = 30

# fib_extension_exhaustion deliberately EXCLUDED: its economics claim was withdrawn 2026-09-01
# (0 genuine combos on the full-grid flip audit) and it carries a 23.0pp cost-erosion gap.
SIGNALS = {
    "taker_delta_z_climax": {"fires": "data/labels/eth_5m_taker_delta_climax_metalabel_20260829/eth_5m_taker_delta_climax_metalabel_features.csv", "horizon": 24},
    "short_term_return_z": {"fires": "data/labels/eth_5m_short_term_return_z_metalabel_20260829/eth_5m_short_term_return_z_metalabel_features.csv", "horizon": 12},
    "liquidity_sweep": {"fires": "data/labels/eth_5m_liquidity_sweep_topdown_metalabel_20260830/eth_5m_liquidity_sweep_topdown_metalabel_features_H30_GAP12_K4.0.csv", "horizon": 30},
    "orthogonal_combo": {"fires": "data/labels/eth_5m_orthogonal_combo_metalabel_20260830/eth_5m_orthogonal_combo_metalabel_features_H24_GAP12_ALLFIRES.csv", "horizon": 24},
    "smt_divergence": {"fires": "data/labels/eth_5m_smt_divergence_metalabel_20260831/eth_5m_smt_divergence_metalabel_features.csv", "horizon": 72},
}


def log(m: str) -> None:
    print(f"[regime_gate_pnl] {m}", flush=True)


def build_regime_pred() -> pd.DataFrame:
    """timestamp -> predicted regime (0 bull / 1 bear / 2 chop) from the DEPLOYED S12_K3 artifact."""
    if REGIME_CACHE.exists():
        return pd.read_parquet(REGIME_CACHE)
    from research_eth_regime_s12k3_label_train_20260902 import load_frame
    from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: F401
    payload = joblib.load(REGIME_ARTIFACT)
    cols, med = payload["feature_cols"], payload["feature_medians"]
    df = load_frame()
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    for c in cols:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(med.get(c, 0.0))
    out = pd.DataFrame({"timestamp": df["timestamp"], "regime": payload["model"].predict(x)})
    REGIME_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(REGIME_CACHE, index=False)
    log(f"regime predictions cached: {len(out):,} bars, chop share {float((out.regime==2).mean()):.3f}")
    return out


def run_grid(ts, o, h, l, c, dec, sc, atr, horizon, vm, om) -> pd.DataFrame:
    tp = np.full(len(dec), 999.0)
    rows = []
    for sl in SL_GRID:
        for arm in ARM_GRID:
            for tr in TRAIL_GRID:
                row, ok = {"sl": sl, "arm": arm, "trail": tr}, True
                for wn, mask in (("val", vm), ("oos", om)):
                    res = simulate_single_position(
                        timestamps=ts, open_px=o, high=h, low=l, close=c,
                        decision_indices=dec[mask], scores=sc[mask], tp_moves=tp[mask],
                        sl_moves=(sl * atr)[mask], upper_threshold=1.0, lower_threshold=-1.0,
                        horizon_bars=horizon, margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE,
                        roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
                        arm_moves=(arm * atr)[mask], trail_moves=(tr * atr)[mask])
                    led = res.ledger
                    n = int(len(led))
                    avg = float(led["trade_return"].mean() * 1e4) if n else float("nan")
                    tot = float(led["trade_return"].sum() * 1e4) if n else 0.0
                    pwr = float((led["trade_return"] > 0).mean()) if n else float("nan")
                    row[f"{wn}_n"], row[f"{wn}_avg_bp"] = n, round(avg, 3)
                    row[f"{wn}_total_bp"], row[f"{wn}_profit_wr"] = round(tot, 1), round(pwr, 4)
                    if not (n > 0 and avg > 0):
                        ok = False
                row["both_positive"] = ok
                rows.append(row)
    return pd.DataFrame(rows)


def genuine_from(real: pd.DataFrame, flip: pd.DataFrame) -> list[dict]:
    fmap = {(r.sl, r.arm, r.trail): r for r in flip.itertuples()}
    out = []
    for r in real[real["both_positive"]].itertuples():
        f = fmap[(r.sl, r.arm, r.trail)]
        if (r.val_avg_bp - f.val_avg_bp) > 0 and (r.oos_avg_bp - f.oos_avg_bp) > 0 \
           and f.val_avg_bp < 0 and f.oos_avg_bp < 0:
            out.append({"sl": r.sl, "arm": r.arm, "trail": r.trail,
                        "val_bp": r.val_avg_bp, "oos_bp": r.oos_avg_bp,
                        "val_total_bp": r.val_total_bp, "oos_total_bp": r.oos_total_bp,
                        "val_n": int(r.val_n), "oos_n": int(r.oos_n),
                        "val_pwr": r.val_profit_wr, "oos_pwr": r.oos_profit_wr})
    return out


def main() -> int:
    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    log(f"klines {len(klines):,} bars {klines.timestamp.min()} ~ {klines.timestamp.max()}")
    reg = build_regime_pred()
    reg_map = dict(zip(reg["timestamp"], reg["regime"]))

    ts = klines["timestamp"]
    o, h, l, c = (klines[k].to_numpy() for k in ("open", "high", "low", "close"))
    results = {}
    for name, cfg in SIGNALS.items():
        fires = pd.read_csv(ROOT / cfg["fires"], parse_dates=["timestamp"])
        fires = fires.loc[fires["timestamp"] < HOLDOUT_START].sort_values("pos").reset_index(drop=True)
        fires["regime"] = fires["timestamp"].map(reg_map)
        cov = float(fires["regime"].notna().mean())
        horizon = cfg["horizon"]
        ev = purged_decision_mask(ts, start=VAL_START, end=OOS_START, horizon_bars=horizon)
        eo = purged_decision_mask(ts, start=OOS_START, end=HOLDOUT_START, horizon_bars=horizon)
        vset, oset = set(np.flatnonzero(ev).tolist()), set(np.flatnonzero(eo).tolist())
        log(f"\n=== {name} (H={horizon}) fires={len(fires)} regime coverage={cov:.3f} "
            f"chop share={float((fires.regime==2).mean()):.3f} ===")

        arms = {"all": np.ones(len(fires), bool),
                "chop": (fires["regime"] == 2).to_numpy(),
                "nonchop": fires["regime"].isin([0, 1]).to_numpy()}
        sig_out = {}
        for aname, keep in arms.items():
            dec = fires.loc[keep, "pos"].to_numpy(np.int64)
            sc = np.where(fires.loc[keep, "side"].to_numpy() == "bottom", 1.0, -1.0)
            atr = fires.loc[keep, "atr_pct"].to_numpy(float)
            vm = np.array([d in vset for d in dec]); om = np.array([d in oset for d in dec])
            if vm.sum() < MIN_WINDOW_N or om.sum() < MIN_WINDOW_N:
                log(f"  {aname:8s}: 표본부족 val={vm.sum()} oos={om.sum()} -- 스킵"); continue
            real = run_grid(ts, o, h, l, c, dec, sc, atr, horizon, vm, om)
            flip = run_grid(ts, o, h, l, c, dec, -sc, atr, horizon, vm, om)
            gen = genuine_from(real, flip)
            best = max(gen, key=lambda g: min(g["val_bp"], g["oos_bp"])) if gen else None
            sig_out[aname] = {"n": int(keep.sum()), "val_n": int(vm.sum()), "oos_n": int(om.sum()),
                              "n_both_positive": int(real["both_positive"].sum()),
                              "n_genuine": len(gen), "best": best}
            log(f"  {aname:8s}: n={keep.sum():5d} val={vm.sum():4d} oos={om.sum():4d} | "
                f"양수 {int(real['both_positive'].sum()):2d}/96 진짜 {len(gen):2d}"
                + (f" | best VAL {best['val_bp']:+6.2f}bp OOS {best['oos_bp']:+6.2f}bp "
                   f"(tot {best['val_total_bp']:+.0f}/{best['oos_total_bp']:+.0f}, "
                   f"SL{best['sl']}/ARM{best['arm']}/Tr{best['trail']})" if best else " | 진짜 없음"))
        results[name] = sig_out

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "part_a_regime_gate.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    log(f"\n=== PART A 요약: 진짜 조합 수 (all -> chop) ===")
    for n, r in results.items():
        a = r.get("all", {}).get("n_genuine", 0); ch = r.get("chop", {}).get("n_genuine", 0)
        nc = r.get("nonchop", {}).get("n_genuine", 0)
        log(f"  {n:24s} all={a:2d}  chop={ch:2d}  nonchop={nc:2d}")
    log(f"Wrote {OUT_DIR/'part_a_regime_gate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
