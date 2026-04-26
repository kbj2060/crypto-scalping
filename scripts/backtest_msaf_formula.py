#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.optimize_duckdb_quant_formula import load_merged
except ModuleNotFoundError:
    from optimize_duckdb_quant_formula import load_merged

from ensemble.msaf_formula import MSAFConfig, MSAFEngine


@dataclass
class SimResult:
    pnl_pct: float
    mdd_pct: float
    sharpe: float
    trades: int
    win_rate_pct: float
    avg_abs_pos: float
    equity_final: float
    params: dict


def calc_mdd(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    dd = eq / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min()) * 100.0


def calc_sharpe(eq: np.ndarray, bars_per_year: int = 365 * 24 * 12) -> float:
    if len(eq) < 10:
        return 0.0
    r = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    s = float(np.std(r))
    if s < 1e-12:
        return 0.0
    return float(np.mean(r) / s * math.sqrt(bars_per_year))


def _cfg_from_params(p: dict) -> MSAFConfig:
    # Grouped weights: 3 groups -> 9 effective weights (fixed internal ratio)
    theta_dir = float(p["theta_direction"])
    theta_struct = float(p["theta_structure"])
    theta_risk = float(p["theta_risk"])
    # direction: flow/obi/nif
    w_flow = theta_dir * 0.40
    w_obi = theta_dir * 0.33
    w_nif = theta_dir * 0.27
    # structure: abs/tox/vpin
    w_abs = theta_struct * 0.45
    w_tox = theta_struct * 0.35
    w_vpin = theta_struct * 0.20
    # risk: liq/aft/eai
    w_liq = theta_risk * 0.19
    w_aft = theta_risk * 0.50
    w_eai = theta_risk * 0.31

    return MSAFConfig(
        w_flow=float(w_flow),
        w_obi=float(w_obi),
        w_nif=float(w_nif),
        w_abs=float(w_abs),
        w_tox=float(w_tox),
        w_vpin=float(w_vpin),
        w_tasd=float(p["w_tasd"]),
        use_tasd=bool(p["use_tasd"]),
        w_liq=float(w_liq),
        w_aft=float(w_aft),
        gamma=float(p["gamma"]),
        beta=float(p["beta"]),
        w_eai=float(w_eai),
        eai_apply_min=float(p["eai_apply_min"]),
        k_stale=float(p["k_stale"]),
        stale_half_life_sec=float(p["stale_half_life_sec"]),
        stale_force_close_sec=float(p["stale_force_close_sec"]),
        k_align=float(p["k_align"]),
        regime_hot_th=float(p["regime_hot_th"]),
        regime_cold_th=float(p["regime_cold_th"]),
        hot_fas_mult=float(p["hot_fas_mult"]),
        hot_sis_mult=float(p["hot_sis_mult"]),
        hot_lcs_mult=float(p["hot_lcs_mult"]),
        hot_roa_mult=float(p["hot_roa_mult"]),
        cold_fas_mult=float(p["cold_fas_mult"]),
        cold_sis_mult=float(p["cold_sis_mult"]),
        cold_lcs_mult=float(p["cold_lcs_mult"]),
        cold_roa_mult=float(p["cold_roa_mult"]),
        c_kelly=float(p["c_kelly"]),
        base_size=float(p["base_size"]),
        min_abs_hat_for_trade=float(p["min_abs_hat_for_trade"]),
        overheat_alpha=float(p["overheat_alpha"]),
        overheat_window=int(p["overheat_window"]),
    )


def _sample_params(rng: np.random.Generator) -> dict:
    return {
        # compressed params (5 core): theta_direction, theta_structure, theta_risk, gamma, beta
        "theta_direction": float(rng.uniform(0.30, 1.20)),
        "theta_structure": float(rng.uniform(0.20, 1.20)),
        "theta_risk": float(rng.uniform(0.20, 1.20)),
        "gamma": float(rng.uniform(0.05, 0.35)),
        "beta": float(rng.uniform(0.6, 2.6)),
        # optional secondary controls
        "w_tasd": float(rng.uniform(0.00, 0.20)),
        "use_tasd": bool(rng.random() >= 0.15),
        "eai_apply_min": float(rng.uniform(0.20, 0.85)),
        "k_stale": float(rng.uniform(0.25, 0.95)),
        "stale_half_life_sec": float(rng.uniform(30.0, 120.0)),
        "stale_force_close_sec": float(rng.uniform(120.0, 360.0)),
        "k_align": float(rng.uniform(0.00, 0.40)),
        "regime_hot_th": float(rng.uniform(1.2, 2.2)),
        "regime_cold_th": float(rng.uniform(-2.2, -1.2)),
        "hot_fas_mult": float(rng.uniform(0.65, 1.10)),
        "hot_sis_mult": float(rng.uniform(0.90, 1.45)),
        "hot_lcs_mult": float(rng.uniform(0.85, 1.25)),
        "hot_roa_mult": float(rng.uniform(0.90, 1.45)),
        "cold_fas_mult": float(rng.uniform(0.70, 1.20)),
        "cold_sis_mult": float(rng.uniform(0.70, 1.20)),
        "cold_lcs_mult": float(rng.uniform(0.90, 1.50)),
        "cold_roa_mult": float(rng.uniform(0.80, 1.30)),
        "c_kelly": float(rng.uniform(1.1, 3.2)),
        "base_size": float(rng.uniform(0.5, 1.5)),
        "min_abs_hat_for_trade": float(rng.uniform(0.00, 0.35)),
        "overheat_alpha": float(rng.uniform(0.35, 0.85)),
        "overheat_window": int(rng.integers(64, 145)),
        "entry_abs": float(rng.uniform(0.04, 0.45)),
        "exit_abs": float(rng.uniform(0.01, 0.30)),
        "min_hold": int(rng.integers(1, 16)),
        "tox_veto": float(rng.uniform(0.70, 0.95)),
        "qc_veto": float(rng.uniform(0.65, 0.95)),
        "aft_veto": float(rng.uniform(0.70, 0.95)),
    }


def _default_params() -> dict:
    return {
        "theta_direction": 0.75,  # maps to flow/obi/nif ~= 0.30/0.25/0.20
        "theta_structure": 0.44,  # maps to abs/tox/vpin ~= 0.20/0.15/0.09
        "theta_risk": 0.80,       # maps to liq/aft/eai ~= 0.15/0.40/0.25
        "gamma": 0.15,
        "beta": 1.50,
        "w_tasd": 0.08,
        "use_tasd": True,
        "eai_apply_min": 0.50,
        "k_stale": 0.80,
        "stale_half_life_sec": 60.0,
        "stale_force_close_sec": 120.0,
        "k_align": 0.20,
        "regime_hot_th": 1.5,
        "regime_cold_th": -1.5,
        "hot_fas_mult": 0.85,
        "hot_sis_mult": 1.15,
        "hot_lcs_mult": 1.00,
        "hot_roa_mult": 1.20,
        "cold_fas_mult": 0.95,
        "cold_sis_mult": 0.90,
        "cold_lcs_mult": 1.20,
        "cold_roa_mult": 1.05,
        "c_kelly": 2.0,
        "base_size": 1.0,
        "min_abs_hat_for_trade": 0.05,
        "overheat_alpha": 0.60,
        "overheat_window": 96,
        "entry_abs": 0.12,
        "exit_abs": 0.05,
        "min_hold": 4,
        "tox_veto": 0.90,
        "qc_veto": 0.90,
        "aft_veto": 0.90,
    }


def run_msaf_sim(
    m: pd.DataFrame,
    p: dict,
    fee_bps: float,
    slip_bps: float,
    leverage: float,
) -> SimResult:
    cfg = _cfg_from_params(p)
    eng = MSAFEngine(cfg)

    close = m["close"].to_numpy(np.float64)
    rets = np.zeros(len(m), dtype=np.float64)
    if len(m) > 1:
        rets[:-1] = close[1:] / np.maximum(close[:-1], 1e-12) - 1.0

    fee = float(fee_bps) / 10_000.0
    slip = float(slip_bps) / 10_000.0
    lev = float(max(leverage, 0.0))
    entry_abs = float(p["entry_abs"])
    exit_abs = float(min(p["exit_abs"], entry_abs))
    min_hold = int(max(1, p["min_hold"]))

    pos = 0.0
    hold = 0
    pnl_bars: list[float] = []
    pos_hist: list[float] = []
    eq = 1.0
    eq_curve = [eq]
    trades = 0
    trade_pnls: list[float] = []
    trade_open_eq = eq
    ts_ns = pd.to_datetime(m["ts"], utc=True).astype("int64").to_numpy(np.int64)

    for i, row in enumerate(m.itertuples(index=False)):
        stale_seconds = 0.0
        if i > 0:
            gap_sec = float((ts_ns[i] - ts_ns[i - 1]) / 1e9)
            stale_seconds = max(0.0, gap_sec - 300.0)
        out = eng.compute(
            {
                "taker_buy_ratio": float(row.taker_buy_ratio),
                "obi": float(row.obi),
                "nif_whale": float(row.nif_whale),
                "shadow_absorption_score": float(row.shadow_absorption_score),
                "shadow_toxicity_score": float(row.shadow_toxicity_score),
                "shadow_queue_collapse": float(row.shadow_queue_collapse),
                "eai": float(row.eai),
                "oi_delta_pct": float(row.oi_delta_pct),
                "funding_rate": float(row.funding_rate),
                "short_usd_1m": float(row.short_usd_1m),
                "long_usd_1m": float(row.long_usd_1m),
                "shadow_aftershock_prob": float(row.shadow_aftershock_prob),
                "data_stale": 0.0,
                "stale_seconds": stale_seconds,
            }
        )

        desired = float(out["size_signed"])
        # veto zone
        if (
            float(row.shadow_toxicity_score) >= float(p["tox_veto"])
            and float(row.shadow_queue_collapse) >= float(p["qc_veto"])
        ) or float(row.shadow_aftershock_prob) >= float(p["aft_veto"]):
            desired = 0.0

        # hysteresis
        if pos == 0.0:
            if abs(desired) < entry_abs:
                desired = 0.0
        else:
            if hold < min_hold:
                desired = pos
            elif abs(desired) < exit_abs:
                desired = 0.0

        prev_pos = pos
        pos = float(np.clip(desired, -1.0, 1.0))
        turn = abs(pos - prev_pos)

        if prev_pos == 0.0 and pos != 0.0:
            trades += 1
            trade_open_eq = eq
            hold = 0
        elif pos != 0.0:
            hold += 1
        else:
            hold = 0

        bar_pnl = lev * prev_pos * rets[i] - lev * turn * (fee + slip)
        eq *= (1.0 + bar_pnl)
        pnl_bars.append(float(bar_pnl))
        pos_hist.append(float(prev_pos))
        eq_curve.append(eq)

        if prev_pos != 0.0 and pos == 0.0:
            trade_pnls.append(eq / max(trade_open_eq, 1e-12) - 1.0)

    eq_arr = np.asarray(eq_curve, dtype=np.float64)
    pnl_pct = (eq_arr[-1] - 1.0) * 100.0
    mdd_pct = calc_mdd(eq_arr)
    sharpe = calc_sharpe(eq_arr)
    wins = sum(1 for x in trade_pnls if x > 0.0)
    win_rate = (100.0 * wins / len(trade_pnls)) if trade_pnls else 0.0
    avg_abs_pos = float(np.mean(np.abs(np.asarray(pos_hist, dtype=np.float64)))) if pos_hist else 0.0

    return SimResult(
        pnl_pct=float(pnl_pct),
        mdd_pct=float(mdd_pct),
        sharpe=float(sharpe),
        trades=int(trades),
        win_rate_pct=float(win_rate),
        avg_abs_pos=float(avg_abs_pos),
        equity_final=float(eq_arr[-1]),
        params=dict(p),
    )


def objective(r: SimResult) -> float:
    return float(r.pnl_pct - 0.55 * abs(min(0.0, r.mdd_pct)) + 0.08 * r.sharpe - 0.02 * max(0, 8 - r.trades))


def objective_with_l2(r: SimResult, p: dict, lam: float = 0.07) -> float:
    l2 = (
        float(p["theta_direction"]) ** 2
        + float(p["theta_structure"]) ** 2
        + float(p["theta_risk"]) ** 2
        + float(p["gamma"]) ** 2
        + float(p["beta"]) ** 2
    )
    return float(objective(r) - lam * l2)


def _median_params(params_list: list[dict]) -> dict:
    if not params_list:
        return _default_params()
    keys = params_list[0].keys()
    out: dict = {}
    for k in keys:
        vals = [p[k] for p in params_list]
        if isinstance(vals[0], bool):
            out[k] = bool(sum(1 for v in vals if bool(v)) >= (len(vals) / 2))
        elif isinstance(vals[0], int) and not isinstance(vals[0], bool):
            out[k] = int(round(float(np.median(vals))))
        else:
            out[k] = float(np.median([float(v) for v in vals]))
    return out


def tune_on_train(
    train_df: pd.DataFrame,
    trials: int,
    fee_bps: float,
    slip_bps: float,
    leverage: float,
    seed: int,
) -> SimResult:
    rng = np.random.default_rng(seed)
    records: list[tuple[float, SimResult, dict]] = []
    base_p = _default_params()
    base_r = run_msaf_sim(train_df, base_p, fee_bps, slip_bps, leverage)
    records.append((objective_with_l2(base_r, base_p), base_r, base_p))
    for _ in range(max(1, trials)):
        p = _sample_params(rng)
        r = run_msaf_sim(train_df, p, fee_bps, slip_bps, leverage)
        records.append((objective_with_l2(r, p), r, p))
    records.sort(key=lambda x: x[0], reverse=True)
    top_n = max(3, int(math.ceil(len(records) * 0.05)))
    top_params = [x[2] for x in records[:top_n]]
    med_p = _median_params(top_params)
    return run_msaf_sim(train_df, med_p, fee_bps, slip_bps, leverage)


def tune_robust_train_val(
    train_df: pd.DataFrame,
    trials: int,
    fee_bps: float,
    slip_bps: float,
    leverage: float,
    seed: int,
    val_ratio: float = 0.3,
) -> SimResult:
    n = len(train_df)
    cut = int(max(150, min(n - 50, round(n * (1.0 - val_ratio)))))
    sub_train = train_df.iloc[:cut].reset_index(drop=True)
    sub_val = train_df.iloc[cut:].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    best_train = run_msaf_sim(sub_train, _default_params(), fee_bps, slip_bps, leverage)
    best_val = run_msaf_sim(sub_val, best_train.params, fee_bps, slip_bps, leverage)
    best_obj = (
        best_val.pnl_pct
        - 0.70 * abs(min(0.0, best_val.mdd_pct))
        + 0.06 * best_val.sharpe
        - 0.35 * abs(best_train.pnl_pct - best_val.pnl_pct)
        - 0.03 * max(0, 6 - best_val.trades)
    )

    for _ in range(max(1, trials)):
        p = _sample_params(rng)
        tr = run_msaf_sim(sub_train, p, fee_bps, slip_bps, leverage)
        va = run_msaf_sim(sub_val, p, fee_bps, slip_bps, leverage)
        obj = (
            va.pnl_pct
            - 0.70 * abs(min(0.0, va.mdd_pct))
            + 0.06 * va.sharpe
            - 0.35 * abs(tr.pnl_pct - va.pnl_pct)
            - 0.03 * max(0, 6 - va.trades)
        )
        if obj > best_obj:
            best_obj = obj
            best_train = tr
            best_val = va

    # Return params chosen by robust objective, evaluated on full train window.
    full_train = run_msaf_sim(train_df, best_train.params, fee_bps, slip_bps, leverage)
    return full_train


def split_train_test(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(max(200, min(n - 50, round(n * train_ratio))))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def walkforward_check(
    df: pd.DataFrame,
    folds: int,
    trials_per_fold: int,
    fee_bps: float,
    slip_bps: float,
    leverage: float,
    seed: int,
    embargo_bars: int = 12,
) -> list[dict]:
    n = len(df)
    out: list[dict] = []
    if n < 400:
        return out
    fold_span = max(80, n // (folds + 1))
    for k in range(1, folds + 1):
        tr_end = fold_span * k
        te_end = min(n, tr_end + fold_span)
        tr_eff_end = max(0, tr_end - embargo_bars)
        te_eff_start = min(n, tr_end + embargo_bars)
        train = df.iloc[:tr_eff_end].reset_index(drop=True)
        test = df.iloc[te_eff_start:te_end].reset_index(drop=True)
        if len(train) < 200 or len(test) < 50:
            continue
        best_train = tune_on_train(train, trials_per_fold, fee_bps, slip_bps, leverage, seed + k * 17)
        test_res = run_msaf_sim(test, best_train.params, fee_bps, slip_bps, leverage)
        out.append(
            {
                "fold": k,
                "train_rows": len(train),
                "test_rows": len(test),
                "train": asdict(best_train),
                "test": asdict(test_res),
                "overfit_gap_pnl_pct": float(best_train.pnl_pct - test_res.pnl_pct),
                "overfit_ratio": float((test_res.pnl_pct / best_train.pnl_pct) if abs(best_train.pnl_pct) > 1e-9 else 0.0),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-csv", default="data/training_features_5m.csv")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--robust-trials", type=int, default=500)
    ap.add_argument("--wf-folds", type=int, default=3)
    ap.add_argument("--wf-trials", type=int, default=60)
    ap.add_argument("--embargo-bars", type=int, default=12)
    ap.add_argument("--fee-bps", type=float, default=2.0)
    ap.add_argument("--slip-bps", type=float, default=1.0)
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/ensemble/metrics/msaf_v1_overfit_check.json")
    args = ap.parse_args()

    df = load_merged(args.price_csv, days=int(args.days))
    if len(df) < 300:
        raise RuntimeError(f"not enough merged rows: {len(df)}")

    train_df, test_df = split_train_test(df, float(args.train_ratio))

    default_p = _default_params()
    default_train = run_msaf_sim(train_df, default_p, args.fee_bps, args.slip_bps, args.leverage)
    default_test = run_msaf_sim(test_df, default_p, args.fee_bps, args.slip_bps, args.leverage)

    tuned_train = tune_on_train(
        train_df,
        trials=int(args.trials),
        fee_bps=float(args.fee_bps),
        slip_bps=float(args.slip_bps),
        leverage=float(args.leverage),
        seed=int(args.seed),
    )
    tuned_test = run_msaf_sim(test_df, tuned_train.params, args.fee_bps, args.slip_bps, args.leverage)

    robust_train = tune_robust_train_val(
        train_df,
        trials=int(args.robust_trials),
        fee_bps=float(args.fee_bps),
        slip_bps=float(args.slip_bps),
        leverage=float(args.leverage),
        seed=int(args.seed) + 777,
    )
    robust_test = run_msaf_sim(test_df, robust_train.params, args.fee_bps, args.slip_bps, args.leverage)

    wf = walkforward_check(
        df,
        folds=int(args.wf_folds),
        trials_per_fold=int(args.wf_trials),
        fee_bps=float(args.fee_bps),
        slip_bps=float(args.slip_bps),
        leverage=float(args.leverage),
        seed=int(args.seed),
        embargo_bars=int(args.embargo_bars),
    )

    cost_scenarios = []
    for fb, sb in [(2.0, 1.0), (4.0, 2.0), (5.0, 3.0), (6.0, 4.0)]:
        r = run_msaf_sim(test_df, tuned_train.params, fb, sb, args.leverage)
        cost_scenarios.append(
            {
                "fee_bps": fb,
                "slip_bps": sb,
                "pnl_pct": r.pnl_pct,
                "mdd_pct": r.mdd_pct,
                "sharpe": r.sharpe,
                "trades": r.trades,
            }
        )

    result = {
        "meta": {
            "rows_total": len(df),
            "rows_train": len(train_df),
            "rows_test": len(test_df),
            "time_min": str(df["ts"].min()),
            "time_max": str(df["ts"].max()),
            "price_csv": args.price_csv,
            "days": int(args.days),
            "fee_bps": float(args.fee_bps),
            "slip_bps": float(args.slip_bps),
            "leverage": float(args.leverage),
            "trials": int(args.trials),
            "embargo_bars": int(args.embargo_bars),
        },
        "default": {
            "train": asdict(default_train),
            "test": asdict(default_test),
            "overfit_gap_pnl_pct": float(default_train.pnl_pct - default_test.pnl_pct),
            "overfit_ratio": float((default_test.pnl_pct / default_train.pnl_pct) if abs(default_train.pnl_pct) > 1e-9 else 0.0),
        },
        "tuned": {
            "train": asdict(tuned_train),
            "test": asdict(tuned_test),
            "overfit_gap_pnl_pct": float(tuned_train.pnl_pct - tuned_test.pnl_pct),
            "overfit_ratio": float((tuned_test.pnl_pct / tuned_train.pnl_pct) if abs(tuned_train.pnl_pct) > 1e-9 else 0.0),
        },
        "robust_tuned": {
            "train": asdict(robust_train),
            "test": asdict(robust_test),
            "overfit_gap_pnl_pct": float(robust_train.pnl_pct - robust_test.pnl_pct),
            "overfit_ratio": float((robust_test.pnl_pct / robust_train.pnl_pct) if abs(robust_train.pnl_pct) > 1e-9 else 0.0),
        },
        "walkforward": wf,
        "cost_scenarios_on_tuned_test": cost_scenarios,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
