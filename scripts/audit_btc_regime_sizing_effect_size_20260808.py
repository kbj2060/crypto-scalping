"""Effect-size re-examination of the ADOPTED BTC czz_trend regime sizing overlay (2026-08-08).

Trigger: the ETH transfer test showed a paired time-block bootstrap can report P=0.979 for a
per-trade difference of t=0.32 -- the bootstrap measures SIGN CONSISTENCY across blocks, not
effect size. BTC's czz_trend was adopted on P=0.739 over 61 trades, so the same scrutiny is owed.

The BTC claim is a RISK claim (MDD -10.34 -> -6.63), so a mean-return test alone would be unfair.
The overlay downsizes bear-regime entries and upsizes bull-regime ones; for that to reduce
drawdown for a REASON rather than by luck, bear-regime trades must be genuinely worse and/or
riskier than bull-regime ones. Tests, on the IDENTITY ledger (VAL+OOS combined, the same 59-61
trade population the adoption used):

  1 MEAN      Welch t-test, bear-regime vs bull-regime per-trade return
  2 RISK      Brown-Forsythe (Levene, median-centred) on return dispersion -- the risk-relevant
              test for an MDD claim; plus the variance ratio
  3 TAIL      worst-quartile and worst-5 trade comparison by regime
  4 DD-SHARE  each regime's share of the identity equity path's drawdown (sum of negative
              returns), versus its share of trades -- does bear actually own the drawdowns?
  5 PERMUTE   the decisive one: keep the SAME multiplier multiset the overlay applies, but assign
              it to trades at random (matched to the regime's trade counts), R=20000, and locate
              the real czz_trend map's ledger-level MDD in that null. This asks "how special is
              THIS labelling?" rather than "is some reweighting better than none?".
              First-order caveat recorded in the output: this permutes sizing on the identity
              trade sequence, whereas the true overlay also perturbs exits through the exit head,
              so it measures the map's information content, not the full replay.

No re-tuning, no new adoption; this only re-scores evidence already collected.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
LEDGER_DIR = ROOT / "tmp/btc_regime_sizing_overlay_20260808"
STATES_PATH = ROOT / "data/research/btc_jm_regime_states_20260808.parquet"
OUT_PATH = LEDGER_DIR / "effect_size_audit.json"
MULT = {0: 0.5, 1: 1.0, 2: 1.5}
R_PERM, SEED = 20000, 903174


def mdd_of(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    return float((curve / np.maximum(peak, 1e-12) - 1.0).min() * 100.0)


def main() -> int:
    st = pd.read_parquet(STATES_PATH)[["timestamp", "czz4"]]
    st["timestamp"] = pd.to_datetime(st["timestamp"])
    st = st.sort_values("timestamp").reset_index(drop=True)

    frames = []
    for split in ("validation", "oos"):
        d = pd.read_csv(LEDGER_DIR / f"{split}_ledger_identity.csv")
        d["split"] = split
        frames.append(d)
    led = pd.concat(frames, ignore_index=True)
    led["t"] = pd.to_datetime(led["entry_timestamp"])
    led = led.sort_values("t").reset_index(drop=True)
    led = pd.merge_asof(led, st.rename(columns={"timestamp": "t"}), on="t",
                        direction="backward", tolerance=pd.Timedelta("10min"))
    led["czz4"] = led["czz4"].fillna(1).astype(int)
    r = led["trade_return"].to_numpy(dtype=float)
    reg = led["czz4"].to_numpy()
    bear, bull = r[reg == 0], r[reg == 2]
    out: dict = {"n_trades": int(len(led)),
                 "regime_trade_counts": {k: int((reg == v).sum()) for k, v in (("bear", 0), ("chop", 1), ("bull", 2))},
                 "identity_mdd_ledger_level": round(mdd_of(r), 2)}

    # 1 MEAN
    t_m, p_m = stats.ttest_ind(bear, bull, equal_var=False)
    pooled_sd = np.sqrt((bear.var(ddof=1) + bull.var(ddof=1)) / 2)
    out["1_mean"] = {"bear_n": len(bear), "bear_mean": round(float(bear.mean()), 4),
                     "bull_n": len(bull), "bull_mean": round(float(bull.mean()), 4),
                     "diff": round(float(bear.mean() - bull.mean()), 4),
                     "welch_t": round(float(t_m), 3), "p": round(float(p_m), 4),
                     "cohens_d": round(float((bear.mean() - bull.mean()) / pooled_sd), 3)}

    # 2 RISK (the relevant axis for an MDD claim)
    lev_s, lev_p = stats.levene(bear, bull, center="median")
    out["2_risk"] = {"bear_sd": round(float(bear.std(ddof=1)), 4), "bull_sd": round(float(bull.std(ddof=1)), 4),
                     "variance_ratio_bear_over_bull": round(float(bear.var(ddof=1) / max(bull.var(ddof=1), 1e-12)), 3),
                     "brown_forsythe_stat": round(float(lev_s), 3), "p": round(float(lev_p), 4),
                     "interpretation": "bear must be RISKIER than bull for downsizing bear to reduce drawdown for a reason"}

    # 3 TAIL
    out["3_tail"] = {"bear_worst5_sum": round(float(np.sort(bear)[:5].sum()), 4),
                     "bull_worst5_sum": round(float(np.sort(bull)[:5].sum()), 4),
                     "bear_q25": round(float(np.quantile(bear, 0.25)), 4),
                     "bull_q25": round(float(np.quantile(bull, 0.25)), 4)}

    # 4 DD-SHARE
    neg = np.minimum(r, 0.0)
    tot_neg = neg.sum()
    out["4_drawdown_share"] = {
        "bear_share_of_negative_return": round(float(neg[reg == 0].sum() / tot_neg), 3) if tot_neg < 0 else None,
        "bear_share_of_trades": round(float((reg == 0).mean()), 3),
        "bull_share_of_negative_return": round(float(neg[reg == 2].sum() / tot_neg), 3) if tot_neg < 0 else None,
        "bull_share_of_trades": round(float((reg == 2).mean()), 3),
        "note": "if bear's share of losses is not above its share of trades, it does not own the drawdowns"}

    # 5 PERMUTATION on the multiplier labelling
    mult_real = np.array([MULT[int(v)] for v in reg], dtype=float)
    real_mdd = mdd_of(r * mult_real)
    rng = np.random.default_rng(SEED)
    null = np.empty(R_PERM)
    for i in range(R_PERM):
        null[i] = mdd_of(r * rng.permutation(mult_real))
    pct = float((null < real_mdd).mean())
    out["5_permutation_multiplier_labelling"] = {
        "real_czz_trend_ledger_mdd": round(real_mdd, 2),
        "identity_mdd": round(mdd_of(r), 2),
        "null_mean_mdd": round(float(null.mean()), 2),
        "null_p05_p95": [round(float(np.percentile(null, 5)), 2), round(float(np.percentile(null, 95)), 2)],
        "fraction_of_random_labellings_WORSE_than_real": round(pct, 3),
        "R": R_PERM,
        "caveat": "permutes sizing on the identity trade sequence; the true overlay also perturbs exits via the exit head, so this measures the MAP's information content, not the full replay",
        "reading": "a real regime effect should put the actual map in the good tail (fraction >> 0.5)"}

    # verdict against an effect-size gate
    gate = {
        "mean_effect_significant_p05": bool(out["1_mean"]["p"] < 0.05),
        "risk_effect_significant_p05": bool(out["2_risk"]["p"] < 0.05),
        "bear_owns_drawdowns": bool((out["4_drawdown_share"]["bear_share_of_negative_return"] or 0)
                                    > out["4_drawdown_share"]["bear_share_of_trades"]),
        "map_beats_random_labelling_p90": bool(pct >= 0.90),
    }
    gate["any_mechanism_evidence"] = bool(any(gate.values()))
    out["effect_size_gate"] = gate
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
