#!/usr/bin/env python3
"""Does the DEPLOYED bull/bear/chop regime classifier (live_regime_gbm3_signal_20260826.py,
tmp/eth_regime_gbm3_independent_20260826/model.joblib, OOS bal_acc 0.9189) rescue the
trend-continuation trade that docs/experiments/eth_trend_continuation_at_evidence_signal_fires_
20260831.md closed as "real but uneconomic"?

That closure found: the union of the 8 evidence-signal fires shows a real continuation tilt
(96/96 trailing-stop grid cells beat random, 9/9 quarters mostly positive) but the per-trade edge
(~+2.2bp avg) is smaller than the trail width needed to execute it -- the grid's optimum keeps
receding to sub-spread trail widths (0.5bp) with no interior peak, and a physically-floored trail
(>=5bp) is barely breakeven at 10bp cost. The unconditional population mixes trend and chop bars;
if the edge is concentrated in genuine trend regime, restricting to it could raise the per-trade
edge enough for a physically realistic (wide) trail to clear costs -- this script tests that
directly rather than assuming it.

Regime probabilities reused VERBATIM from research_eth_evidence_signal_regime_chop_conditional_
20260827.py's build_regime_frame() (same _with_raw_state12 + GBM3 model, same
training_features_{2025,2026_rebuilt}.csv source) -- not re-derived. Coverage is 2025-01-01 to
2026-08-19 only (the training_features files' range), which still fully covers this lineage's
VAL(2025-09..12)/OOS(2026-01..03) window; the 2024 TRAIN-diagnostic quarters used elsewhere in
this lineage are NOT available here and are excluded, not silently zero-filled.

⚠️ CAVEAT carried over unchanged from the source script: VAL+OOS is INSIDE the regime model's own
TRAIN range (2024-01-01..2026-06-30) -- its bull/bear/chop split there is in-sample and may be
optimistic. Flagged, not hidden -- diagnostic framing, not a promotion claim.

⚠️ Precedent this script deliberately does NOT assume the direction of: research_eth_autocorr_
regime_gate_kalman_demarker_20260831.py hard-gated 2 REVERSAL signals on a mean-reversion
autocorrelation regime and found the OPPOSITE of the hypothesis (12/12 comparisons stronger in
momentum regime, not mean-reversion regime) -- regime conditioning in this repo has already
surprised once. Every claim below is measured, not assumed from the "trend regime should favor
continuation" intuition.

Four parts:
  A) raw continuation lift by regime bucket (bull/bear/chop argmax, and trend=bull|bear vs chop)
  B) the Phase1 script's decisive pure-direction test (exactly-one-of-{extend,revert} occurred),
     rerun WITH bull/bear/chop probs added as GBM features -- does regime fix the dead 0.49-0.52
     AUC, or is it still uninformative?
  C) economics: the confirmed cell (SL3.5/ARM0.5/Trail0.1) AND a physically-floored cell, each
     restricted to fires inside a regime bucket, vs a random-entry baseline drawn from the SAME
     bucket (rules out "regime bars just have different vol/drift" as a confound)
  D) sample-size disclosure -- how much does each regime filter cut the population by
"""
from __future__ import annotations

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
from live_evidence_signal_dashboard_20260823 import SIGNAL_ORDER, compute_signals  # noqa: E402
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,
    build_indicator_frame,
)
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402

START = pd.Timestamp("2025-01-01")   # regime-data floor, NOT the lineage's usual 2024-01-01
VAL_START, OOS_START, HOLDOUT_START = (pd.Timestamp(x) for x in ("2025-09-01", "2026-01-01", "2026-04-01"))
GAP, H, SL, ARM, TRAIL = 12, 24, 3.5, 0.5, 0.1        # the confirmed cell
FLOOR_ARM, FLOOR_TRAIL = 0.5, 0.20                     # physically-defensible (~5bp trail) cell
MARGIN, LEV, COST = 0.30, 3.0, 0.001
RANDOM_SEED = 20260831
REGIME_TRAIN_PATHS = [
    ROOT / "data/splits/year_oos/training_features_2025.csv",
    ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
]
REGIME_MODEL_PATH = ROOT / "tmp/eth_regime_gbm3_independent_20260826/model.joblib"


def build_regime_frame() -> pd.DataFrame:
    frames = [pd.read_csv(p, parse_dates=["timestamp"]) for p in REGIME_TRAIN_PATHS]
    raw = pd.concat(frames, ignore_index=True).sort_values("timestamp") \
        .drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    feats = _with_raw_state12(raw)
    payload = joblib.load(REGIME_MODEL_PATH)
    cols = payload["feature_cols"]
    med = pd.Series(payload["feature_medians"])
    for c in cols:
        if c not in feats.columns:
            feats[c] = med.get(c, 0.0)
    x = feats[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    proba = payload["model"].predict_proba(x)
    classes = list(payload["classes"])
    out = pd.DataFrame({"timestamp": feats["timestamp"].reset_index(drop=True)})
    for i, name in enumerate(classes):
        out[f"{name}_prob"] = proba[:, i]
    out["regime_label"] = out[[f"{c}_prob" for c in classes]].idxmax(axis=1).str.replace("_prob", "", regex=False)
    return out


def load_kl(name: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / f"binance_data/klines/{name}/{name}-5m-api.csv", parse_dates=["timestamp"])
    return df.loc[df["timestamp"] >= START - pd.Timedelta(days=10)].reset_index(drop=True)


def forward_extremes(close, high, low, h):
    fh = pd.Series(high).shift(-1).rolling(h, min_periods=h).max().shift(-(h - 1)).to_numpy()
    fl = pd.Series(low).shift(-1).rolling(h, min_periods=h).min().shift(-(h - 1)).to_numpy()
    return (fh - close) / close, (close - fl) / close


def main() -> int:
    print("building regime frame (GBM3 bull/bear/chop, verbatim reuse)...")
    regime = build_regime_frame()
    print(f"  regime coverage: {regime['timestamp'].min()} .. {regime['timestamp'].max()} "
          f"({len(regime)} bars)")
    print("  label mix:", regime["regime_label"].value_counts(normalize=True).round(3).to_dict())

    eth, btc = load_kl("ETHUSDT"), load_kl("BTCUSDT")
    sig = compute_signals(eth, btc, None)
    sig = sig.loc[sig["timestamp"] >= START].reset_index(drop=True)
    kl = eth.loc[eth["timestamp"] >= START].reset_index(drop=True)
    ind = build_indicator_frame(eth)
    ind = ind.loc[ind["timestamp"] >= START].reset_index(drop=True)
    assert (kl["timestamp"].to_numpy() == sig["timestamp"].to_numpy()).all()
    assert (kl["timestamp"].to_numpy() == ind["timestamp"].to_numpy()).all()

    merged = kl[["timestamp"]].merge(regime, on="timestamp", how="left")
    print(f"  regime NaN after merge (gap bars): {merged['regime_label'].isna().sum()} / {len(merged)}")
    regime_label = merged["regime_label"].to_numpy()
    trend_prob = 1.0 - merged["chop_prob"].to_numpy()

    close, high, low = kl["close"].to_numpy(), kl["high"].to_numpy(), kl["low"].to_numpy()
    atr_pct = ind["atr_pct"].to_numpy()
    names = [n for n, _ in SIGNAL_ORDER]
    bot = np.zeros(len(sig), bool); top = np.zeros(len(sig), bool)
    for n in names:
        bot |= sig[f"bottom_{n}"].to_numpy(); top |= sig[f"top_{n}"].to_numpy()
    rows = []
    for side, m in (("bottom", bot), ("top", top)):
        last = -10**9
        for i in np.flatnonzero(m):
            if i - last < GAP:
                continue
            last = i; rows.append((i, side))
    ev = pd.DataFrame(rows, columns=["pos", "side"]).sort_values("pos").reset_index(drop=True)
    iu = ev["pos"].to_numpy(); isb = (ev["side"] == "bottom").to_numpy()
    print(f"  cluster-anchored candidates: {len(ev)}  "
          f"(regime known at fire: {(~pd.isna(regime_label[iu])).sum()})")

    # ================= Part A: raw lift by regime bucket =================
    print("\n=== [A] raw continuation lift by regime bucket (vs population baseline WITHIN that "
          "bucket) ===")
    print(f"{'bucket':<12}{'H':>4}{'n':>7}{'P(cont|fire)':>14}{'P(cont|bucket)':>16}{'lift':>7}")
    buckets = {
        "bull": regime_label == "bull", "bear": regime_label == "bear", "chop": regime_label == "chop",
        "trend(bull|bear)": (regime_label == "bull") | (regime_label == "bear"),
    }
    for h in (3, 6, 12, 24):
        up, dn = forward_extremes(close, high, low, h)
        cont_at_i = np.where(isb, dn[iu], up[iu])
        for bname, bmask in buckets.items():
            base_v = bmask & ~np.isnan(dn)
            base = ((np.where(True, dn, dn)[base_v] >= 2.0 * atr_pct[base_v]).mean() * 0.5 +
                    (up[base_v] >= 2.0 * atr_pct[base_v]).mean() * 0.5)  # symmetric baseline
            fire_v = bmask[iu] & ~np.isnan(cont_at_i)
            if fire_v.sum() < 30:
                continue
            pf = (cont_at_i[fire_v] >= 2.0 * atr_pct[iu][fire_v]).mean()
            print(f"{bname:<12}{h:>4}{int(fire_v.sum()):>7}{100*pf:>13.1f}%{100*base:>15.1f}%{pf/max(base,1e-9):>7.2f}x")

    # ================= Part B: does regime fix the dead pure-direction AUC? =================
    print("\n=== [B] pure-direction test (exactly one of extend/revert happened), Tier0 vs "
          "Tier0+regime ===")
    Xbase = ind.iloc[iu][[c for c in FEATURE_COLUMNS if c != "is_bottom"]].copy()
    Xbase["is_bottom"] = isb.astype(int)
    reg_at_i = merged.iloc[iu][["bull_prob", "bear_prob", "chop_prob"]].fillna(1.0 / 3).reset_index(drop=True)
    Xreg = pd.concat([Xbase.reset_index(drop=True), reg_at_i], axis=1)
    t = kl["timestamp"].iloc[iu].to_numpy()
    print(f"{'H':>4}{'K':>6}{'n_decisive':>11}{'Tier0 VAL':>11}{'Tier0 OOS':>11}"
          f"{'+regime VAL':>12}{'+regime OOS':>12}")
    for h in (3, 6, 12, 24, 48):
        up, dn = forward_extremes(close, high, low, h)
        cont = np.where(isb, dn[iu], up[iu]); rev = np.where(isb, up[iu], dn[iu])
        ok = ~np.isnan(cont) & ~np.isnan(rev) & (atr_pct[iu] > 0)
        k = float(np.median(cont[ok] / atr_pct[iu][ok]))
        E, Rv = cont >= k * atr_pct[iu], rev >= k * atr_pct[iu]
        dec = ok & (E ^ Rv)
        y = E[dec].astype(int)
        td = t[dec]
        tr = td < VAL_START; va = (td >= VAL_START) & (td < OOS_START); oo = (td >= OOS_START) & (td < HOLDOUT_START)
        aucs = {}
        for tag, X in (("base", Xbase[dec]), ("reg", Xreg[dec])):
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
                                                 l2_regularization=1.0, early_stopping=True, random_state=0)
            clf.fit(X[tr], y[tr])
            aucs[tag] = (roc_auc_score(y[va], clf.predict_proba(X[va])[:, 1]),
                        roc_auc_score(y[oo], clf.predict_proba(X[oo])[:, 1]))
        print(f"{h:>4}{k:>6.2f}{int(dec.sum()):>11}{aucs['base'][0]:>11.4f}{aucs['base'][1]:>11.4f}"
              f"{aucs['reg'][0]:>12.4f}{aucs['reg'][1]:>12.4f}")

    # ================= Part C: economics by regime bucket =================
    ts = kl["timestamp"]
    o = kl["open"].to_numpy(); c_ = kl["close"].to_numpy()
    dec_all = iu.astype(np.int64)
    scores_all = np.where(isb, -1.0, 1.0)
    print("\n=== [C] economics by regime bucket -- CONFIRMED cell (SL3.5/ARM0.5/Trail0.1) ===")
    print(f"{'bucket':<18}{'split':>6}{'n':>6}{'CONT bp':>10}{'RAND bp':>10}{'diff':>8}")

    rng = np.random.default_rng(RANDOM_SEED)
    valid_rand = np.flatnonzero(~np.isnan(atr_pct) & (atr_pct > 0))
    valid_rand = valid_rand[(valid_rand > 900) & (valid_rand < len(kl) - 60)]

    def econ(dec, sc, arm, trail, w):
        s, e = (VAL_START, OOS_START) if w == "val" else (OOS_START, HOLDOUT_START)
        el = set(np.flatnonzero(purged_decision_mask(ts, start=s, end=e, horizon_bars=H)).tolist())
        m = np.array([d in el for d in dec])
        if m.sum() == 0:
            return 0, float("nan")
        a = atr_pct[dec][m]
        r = simulate_single_position(timestamps=ts, open_px=o, high=high, low=low, close=c_,
            decision_indices=dec[m], scores=sc[m], tp_moves=np.full(int(m.sum()), 999.0),
            sl_moves=SL * a, upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=H,
            margin_fraction=MARGIN, leverage=LEV, roundtrip_cost_rate=COST,
            arm_moves=arm * a, trail_moves=trail * a)
        return len(r.ledger), (r.ledger["trade_return"].mean() * 1e4 if len(r.ledger) else float("nan"))

    def econ_for(arm, trail, label):
        for bname, bmask in {**buckets, "ALL(unconditional)": np.ones(len(kl), bool)}.items():
            fire_ok = bmask[dec_all]
            d = dec_all[fire_ok]; sc = scores_all[fire_ok]
            rnd_pool = valid_rand[bmask[valid_rand]] if bname != "ALL(unconditional)" else valid_rand
            if len(rnd_pool) < len(d):
                continue
            rd = np.sort(rng.choice(rnd_pool, size=len(d), replace=False))
            rsc = rng.choice([-1.0, 1.0], size=len(rd))
            for w in ("val", "oos"):
                n, bp = econ(d, sc, arm, trail, w)
                _, rbp = econ(rd, rsc, arm, trail, w)
                print(f"{bname:<18}{w:>6}{n:>6}{bp:>+10.2f}{rbp:>+10.2f}{bp-rbp:>+8.2f}")

    econ_for(ARM, TRAIL, "confirmed")
    print(f"\n=== [C2] economics by regime bucket -- PHYSICALLY-FLOORED cell "
          f"(ARM={FLOOR_ARM}/Trail={FLOOR_TRAIL}, trail~{FLOOR_TRAIL*0.258*100:.1f}bp median) ===")
    print(f"{'bucket':<18}{'split':>6}{'n':>6}{'CONT bp':>10}{'RAND bp':>10}{'diff':>8}")
    econ_for(FLOOR_ARM, FLOOR_TRAIL, "floored")

    print("\n=== [D] sample-size disclosure -- population remaining after each regime filter ===")
    for bname, bmask in buckets.items():
        f = bmask[dec_all]
        print(f"  {bname:<18} {int(f.sum())}/{len(dec_all)} candidates ({100*f.mean():.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
