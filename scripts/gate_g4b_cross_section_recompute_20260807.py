"""Gate G4b, cross-sectional part recomputed correctly from the labels the main G4b run saved.

The first run conditioned label agreement on bars where ALL 60 assets were simultaneously non-CASH,
which selects a vanishing and extreme subset of bars once the asset count is large and pushes the
agreement statistic toward 1 for reasons that have nothing to do with redundancy. This recomputes
it pairwise (each pair evaluated only where that pair is jointly active) and adds two further
measures, because the pooled sample-size claim rests entirely on this number:

- effective rank of the LABEL correlation matrix (labels mapped to -1/0/+1), which is what pooling
  actually duplicates, rather than the 5m return correlation used before.
- the classic correlated-observations count `m / (1 + (m-1) * rho_bar)`, as an independent check on
  the participation-ratio estimate.

Reads data/panel/tripbarrier/*.parquet (written by
scripts/gate_g4b_btc_panel_pooling_sample_size_20260807.py) so nothing is recomputed from bars.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data/panel/tripbarrier"
PANEL_DIR = ROOT / "data/panel/features"
OUT_DIR = ROOT / "tmp/btc_gate_g4b_panel_pooling_20260807"


def _effective_rank(C: np.ndarray) -> float:
    eig = np.clip(np.linalg.eigvalsh(C), 0, None)
    return float(eig.sum() ** 2 / (eig ** 2).sum())


def main() -> int:
    files = sorted(LABEL_DIR.glob("*.parquet"))
    lab, ret = {}, {}
    for f in files:
        sym = f.stem
        d = pd.read_parquet(f, columns=["timestamp", "trade_outcome_action"])
        lab[sym] = pd.Series(d["trade_outcome_action"].to_numpy(), index=pd.DatetimeIndex(d["timestamp"]))
        c = pd.read_parquet(PANEL_DIR / f"{sym}.parquet", columns=["timestamp", "close"])
        close = c["close"].to_numpy(dtype=np.float64)
        lr = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(max(close[0], 1e-9)))
        ret[sym] = pd.Series(lr, index=pd.DatetimeIndex(c["timestamp"]))

    syms = sorted(lab)
    A = pd.concat([lab[s].rename(s) for s in syms], axis=1, join="inner")
    R = pd.concat([ret[s].rename(s) for s in syms], axis=1, join="inner")
    S = len(syms)
    print(f"{S} symbols, {len(A)} common bars")

    Aa = A.to_numpy().T
    act = Aa != 0
    agree = np.full((S, S), np.nan)
    both_n = np.zeros((S, S))
    for i in range(S):
        for j in range(S):
            b = act[i] & act[j]
            both_n[i, j] = b.sum()
            if b.any():
                agree[i, j] = float((Aa[i, b] == Aa[j, b]).mean())
    mask = ~np.eye(S, dtype=bool)
    off = agree[mask]
    off = off[np.isfinite(off)]

    # signed label correlation: what pooling actually duplicates
    signed = np.where(Aa == 1, 1.0, np.where(Aa == 2, -1.0, 0.0))
    C_lab = np.corrcoef(signed)
    C_ret = np.corrcoef(R.to_numpy(dtype=np.float64).T)
    er_lab, er_ret = _effective_rank(C_lab), _effective_rank(C_ret)
    rho_lab = float(C_lab[mask].mean())
    rho_ret = float(C_ret[mask].mean())
    m_eff_lab = S / (1.0 + (S - 1) * rho_lab)
    m_eff_ret = S / (1.0 + (S - 1) * rho_ret)

    summary = json.loads((OUT_DIR / "g4b_summary.json").read_text(encoding="utf-8"))
    naive_s4 = summary["pooled"]["naive_sum_stride4"]
    btc_s4 = summary["pooled"]["btc_effective_n_stride4"]

    print("\n=== pairwise label agreement (corrected: conditioned per pair) ===")
    print(f"  mean {off.mean():.3f}  p10 {np.percentile(off,10):.3f}  "
          f"median {np.median(off):.3f}  p90 {np.percentile(off,90):.3f}  max {off.max():.3f}")
    print(f"  median bars per pair used: {np.median(both_n[mask]):.0f}")
    print("\n=== redundancy estimators ===")
    print(f"  signed-label corr: mean off-diag {rho_lab:.3f}  effective_rank {er_lab:.2f}/{S}  "
          f"m/(1+(m-1)rho) = {m_eff_lab:.2f}")
    print(f"  5m return corr:    mean off-diag {rho_ret:.3f}  effective_rank {er_ret:.2f}/{S}  "
          f"m/(1+(m-1)rho) = {m_eff_ret:.2f}")

    print("\n=== pooled effective sample size (stride 4) ===")
    print(f"  BTC alone: {btc_s4:.0f}")
    print(f"  naive sum over {S} assets: {naive_s4:.0f}  ({naive_s4/btc_s4:.1f}x)")
    for name, m_eff in (("label effective_rank", er_lab), ("label m/(1+(m-1)rho)", m_eff_lab),
                        ("return effective_rank", er_ret), ("return m/(1+(m-1)rho)", m_eff_ret)):
        val = naive_s4 * m_eff / S
        print(f"  corrected by {name:<26}: {val:>9.0f}  ({val/btc_s4:.1f}x over BTC alone)")

    payload = {
        "n_symbols": S, "n_common_bars": int(len(A)),
        "pairwise_label_agreement": {
            "mean": float(off.mean()), "p10": float(np.percentile(off, 10)),
            "median": float(np.median(off)), "p90": float(np.percentile(off, 90)),
            "max": float(off.max()),
            "median_bars_per_pair": float(np.median(both_n[mask])),
        },
        "redundancy": {
            "label_corr_mean": rho_lab, "label_effective_rank": er_lab, "label_m_eff": m_eff_lab,
            "return_corr_mean": rho_ret, "return_effective_rank": er_ret, "return_m_eff": m_eff_ret,
        },
        "pooled_stride4": {
            "btc_alone": btc_s4, "naive_sum": naive_s4,
            "corrected_label_effective_rank": naive_s4 * er_lab / S,
            "corrected_label_m_eff": naive_s4 * m_eff_lab / S,
            "corrected_return_effective_rank": naive_s4 * er_ret / S,
            "corrected_return_m_eff": naive_s4 * m_eff_ret / S,
        },
    }
    (OUT_DIR / "g4b_cross_section_corrected.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_DIR}/g4b_cross_section_corrected.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
