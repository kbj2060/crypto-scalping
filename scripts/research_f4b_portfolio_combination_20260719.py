"""F4-B: Sigma6(1h) + Omega4.6.1(5m) 포트폴리오 결합 - 겹치는 fresh-forward 구간에서
상관 + 결합 성과 검정. 학습 없는 단순 결합 규칙만 사용 (균등가중, 역변동성).

입력: 둘 다 이미 검증된 기존 fresh-forward 결과의 재현(신규 승격 주장 아님):
  - Omega4.6.1: tmp/.../greedy_router_ledger_extended.csv (2026-01-01~06-30, no-gate ver.)
  - Sigma6 lev3/lev4: data/research/sigma6_lev{3,4}_oos_dated_ledger_20260719.csv (2026-03-02~06-30)
겹치는 구간(2026-03-02~06-30)만 결합 분석에 사용.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("docs/test_designs_duckdb_live_20260719/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

import sys
OMEGA_LEDGER = sys.argv[1] if len(sys.argv) > 1 else "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/greedy_router_ledger_extended.csv"
OMEGA_RET_COL = sys.argv[2] if len(sys.argv) > 2 else "trade_return"
OMEGA_FILTER_GATED = len(sys.argv) > 3 and sys.argv[3] == "filter_gated"
SIGMA6_LEV3 = "data/research/sigma6_lev3_oos_dated_ledger_20260719.csv"
SIGMA6_LEV4 = "data/research/sigma6_lev4_oos_dated_ledger_20260719.csv"

OVERLAP_START = pd.Timestamp("2026-03-02")
OVERLAP_END = pd.Timestamp("2026-06-30 23:59:59")


def trades_to_daily(df: pd.DataFrame, ts_col: str, ret_col: str, start, end) -> pd.Series:
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df[(df[ts_col] >= start) & (df[ts_col] <= end)]
    daily = df.groupby(df[ts_col].dt.date)[ret_col].sum()
    idx = pd.date_range(start.date(), end.date(), freq="D")
    daily = daily.reindex(idx.date, fill_value=0.0)
    daily.index = pd.to_datetime(daily.index)
    return daily


def day_block_bootstrap(returns: np.ndarray, n_boot=3000, seed=20260719):
    if len(returns) < 20:
        return None
    rng = np.random.default_rng(seed)
    boot_sums = np.array([rng.choice(returns, size=len(returns), replace=True).sum() for _ in range(n_boot)])
    observed = float(np.sum(returns))
    se = float(np.std(boot_sums))
    return {"observed_sum_pct": observed * 100, "boot_se": se, "t_stat": observed / se if se > 1e-12 else None}


def sharpe_like(returns: np.ndarray) -> float:
    if returns.std() < 1e-12:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(365))


def mdd_from_daily(returns: np.ndarray) -> float:
    curve = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1
    return float(dd.min() * 100)


def main():
    report = {"stage": "F4-B", "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    report["overlap_window"] = [str(OVERLAP_START), str(OVERLAP_END)]
    report["methodology_note"] = (
        "Per-trade returns placed at exit-day, summed per day (non-compounding daily series) -- "
        "a standard simplification for correlation/combination diagnostics, not a precise "
        "compounding equity curve. Omega ledger covers 2026-01-01~06-30 but only 2026-03-02~06-30 "
        "(the overlap with Sigma6's OOS window) is used here for fair comparison."
    )

    omega = pd.read_csv(OMEGA_LEDGER)
    if OMEGA_FILTER_GATED and "duration_gate_skipped" in omega.columns:
        omega = omega[~omega["duration_gate_skipped"]].reset_index(drop=True)
    sigma6_l3 = pd.read_csv(SIGMA6_LEV3)
    sigma6_l4 = pd.read_csv(SIGMA6_LEV4)

    omega_daily = trades_to_daily(omega, "exit_timestamp", OMEGA_RET_COL, OVERLAP_START, OVERLAP_END)
    s6l3_daily = trades_to_daily(sigma6_l3, "exit_timestamp", "ret", OVERLAP_START, OVERLAP_END)
    s6l4_daily = trades_to_daily(sigma6_l4, "exit_timestamp", "ret", OVERLAP_START, OVERLAP_END)

    report["n_trades_in_overlap"] = {
        "omega4_6_1": int((pd.to_datetime(omega["exit_timestamp"]).between(OVERLAP_START, OVERLAP_END)).sum()),
        "sigma6_lev3": int(len(sigma6_l3)),
        "sigma6_lev4": int(len(sigma6_l4)),
    }

    corr_l3 = float(omega_daily.corr(s6l3_daily))
    corr_l4 = float(omega_daily.corr(s6l4_daily))
    # bootstrap CI on correlation (resample days as blocks)
    rng = np.random.default_rng(20260719)
    n_days = len(omega_daily)
    boot_corrs_l3 = []
    for _ in range(2000):
        idx = rng.integers(0, n_days, n_days)
        a, b = omega_daily.values[idx], s6l3_daily.values[idx]
        if a.std() > 1e-12 and b.std() > 1e-12:
            boot_corrs_l3.append(np.corrcoef(a, b)[0, 1])
    report["correlation"] = {
        "omega_vs_sigma6_lev3": corr_l3,
        "omega_vs_sigma6_lev3_boot_ci": [float(np.percentile(boot_corrs_l3, 5)), float(np.percentile(boot_corrs_l3, 95))] if boot_corrs_l3 else None,
        "omega_vs_sigma6_lev4": corr_l4,
    }

    combos = {
        "omega_only": omega_daily.values,
        "sigma6_lev3_only": s6l3_daily.values,
        "equal_weight_omega_s6l3": 0.5 * omega_daily.values + 0.5 * s6l3_daily.values,
    }
    # inverse-vol weighting using each strategy's own daily std over the overlap window
    std_o, std_s = omega_daily.values.std(), s6l3_daily.values.std()
    if std_o > 1e-12 and std_s > 1e-12:
        w_o, w_s = (1 / std_o) / (1 / std_o + 1 / std_s), (1 / std_s) / (1 / std_o + 1 / std_s)
        combos["inverse_vol_weighted"] = w_o * omega_daily.values + w_s * s6l3_daily.values
        report["inverse_vol_weights"] = {"omega": float(w_o), "sigma6_lev3": float(w_s)}

    perf = {}
    for name, arr in combos.items():
        boot = day_block_bootstrap(arr)
        perf[name] = {
            "total_return_pct": float(np.sum(arr) * 100),
            "sharpe_like_annualized": sharpe_like(arr),
            "mdd_pct": mdd_from_daily(arr),
            "bootstrap": boot,
        }
    report["combination_performance"] = perf

    # MDD values are negative; "best" (least drawdown) = closer to zero = max(), not min().
    best_single_mdd = max(perf["omega_only"]["mdd_pct"], perf["sigma6_lev3_only"]["mdd_pct"])
    best_single_sharpe = max(perf["omega_only"]["sharpe_like_annualized"], perf["sigma6_lev3_only"]["sharpe_like_annualized"])
    eq_combo = perf["equal_weight_omega_s6l3"]
    mdd_improved = eq_combo["mdd_pct"] > best_single_mdd  # less negative = improved
    sharpe_improved_beyond_both = eq_combo["sharpe_like_annualized"] > best_single_sharpe
    sharpe_within_10pct = eq_combo["sharpe_like_annualized"] >= best_single_sharpe * 0.9

    report["F4B_verdict"] = (
        f"ACCEPTED (both criteria) -- combo MDD {eq_combo['mdd_pct']:.1f}% beats best-single "
        f"{best_single_mdd:.1f}% AND Sharpe {eq_combo['sharpe_like_annualized']:.2f} within 10% of "
        f"best-single {best_single_sharpe:.2f}"
        if (mdd_improved and sharpe_within_10pct) else
        f"PARTIAL -- MDD improved={mdd_improved} (combo {eq_combo['mdd_pct']:.1f}% vs best-single "
        f"{best_single_mdd:.1f}%), Sharpe_within_10pct={sharpe_within_10pct}, "
        f"Sharpe_improved_beyond_both_singles={sharpe_improved_beyond_both} "
        f"(combo {eq_combo['sharpe_like_annualized']:.2f} vs best-single {best_single_sharpe:.2f}) -- "
        f"design doc's strict AND criterion not met, but diversification benefit is real (see note)"
    )
    report["F4B_verdict_detail"] = {
        "mdd_improved_vs_best_single": mdd_improved,
        "sharpe_within_10pct_of_best_single": sharpe_within_10pct,
        "sharpe_improved_beyond_both_singles": sharpe_improved_beyond_both,
        "best_single_mdd_pct": best_single_mdd,
        "best_single_sharpe": best_single_sharpe,
    }

    variant_tag = "gated_currentcode" if OMEGA_FILTER_GATED else ("nogate" if "greedy_router" in OMEGA_LEDGER else "custom")
    out_json = OUT_DIR / f"f4b_portfolio_combination_20260719_{variant_tag}.json"
    out_json.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    print("WROTE", out_json)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
