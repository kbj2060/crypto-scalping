#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_param_ensemble import _ensemble_backtest
from scripts.backtest_msaf_formula import run_msaf_sim


def _clamp(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.clip(lower=lo, upper=hi)


def _to_num(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def load_mapped_2025(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    num_cols = [
        "open", "high", "low", "close", "volume", "quote_volume",
        "taker_buy_quote", "smart_money_flow", "oi_change_rate", "last_funding_rate",
        "net_taker_ratio", "ofti", "volatility_z", "squeeze_power",
        "trade_intensity", "big_trade_ratio", "amihud_illiquidity_z",
        "evt_tail_flag", "jump_z",
    ]
    _to_num(df, num_cols)
    df = df.dropna(subset=["open", "high", "low", "close", "volume", "quote_volume"]).copy()

    # Proxy mapping from rl_training_data -> microstructure-style features.
    qv = df["quote_volume"].replace(0.0, np.nan).ffill().bfill().fillna(1.0)
    tbr = (df["taker_buy_quote"] / qv).replace([np.inf, -np.inf], np.nan).fillna(0.5)
    df["taker_buy_ratio"] = _clamp(tbr, 0.0, 1.0)

    ofti_scale = float(df["ofti"].abs().quantile(0.99) + 1e-8)
    smf_scale = float(df["smart_money_flow"].abs().quantile(0.99) + 1e-8)
    sq_scale = float(df["squeeze_power"].abs().quantile(0.95) + 1e-8)

    df["obi"] = np.tanh(df["ofti"] / ofti_scale)
    df["nif_whale"] = _clamp(df["smart_money_flow"] / smf_scale, -1.0, 1.0)
    df["funding_rate"] = df["last_funding_rate"].fillna(0.0)
    df["oi_delta_pct"] = df["oi_change_rate"].fillna(0.0)

    # Structure proxies
    ti_z = ((df["trade_intensity"] - df["trade_intensity"].rolling(96, min_periods=16).mean()) /
            (df["trade_intensity"].rolling(96, min_periods=16).std().replace(0, np.nan))).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    btr = df["big_trade_ratio"].fillna(0.0)
    volz = df["volatility_z"].fillna(0.0)
    amz = df["amihud_illiquidity_z"].fillna(0.0)
    jump = df["jump_z"].fillna(0.0).abs()
    tail = df["evt_tail_flag"].fillna(0.0)

    abs_score = 1.0 / (1.0 + np.exp(-(0.8 * ti_z + 0.5 * btr - 0.4 * df["net_taker_ratio"].abs().fillna(0.0))))
    tox_score = 1.0 / (1.0 + np.exp(-(0.8 * volz + 0.6 * amz + 0.3 * jump)))
    qc_score = 1.0 / (1.0 + np.exp(-(0.7 * volz + 0.6 * jump + 0.8 * tail)))
    aft_prob = _clamp(0.35 * tail + 0.65 * (1.0 / (1.0 + np.exp(-(jump - 1.5)))), 0.0, 1.0)

    df["shadow_absorption_score"] = _clamp(abs_score, 0.0, 1.0)
    df["shadow_toxicity_score"] = _clamp(tox_score, 0.0, 1.0)
    df["shadow_queue_collapse"] = _clamp(qc_score, 0.0, 1.0)
    df["shadow_aftershock_prob"] = aft_prob

    # eai proxy
    sq_n = np.tanh(df["squeeze_power"].fillna(0.0) / sq_scale)
    df["eai"] = _clamp(0.5 * (sq_n.abs()) + 0.3 * _clamp((ti_z + 3.0) / 6.0, 0.0, 1.0) + 0.2 * _clamp((volz + 3.0) / 6.0, 0.0, 1.0), 0.0, 1.0)

    # liquidation proxies (no raw liquidation stream in this dataset)
    liq_mag = (df["oi_delta_pct"].abs() * qv).fillna(0.0)
    df["long_usd_1m"] = np.where(df["oi_delta_pct"] > 0, liq_mag, 0.0)
    df["short_usd_1m"] = np.where(df["oi_delta_pct"] < 0, liq_mag, 0.0)

    keep = [
        "ts", "open", "high", "low", "close", "volume", "quote_volume",
        "obi", "taker_buy_ratio", "nif_whale", "funding_rate", "oi_delta_pct",
        "shadow_absorption_score", "shadow_toxicity_score", "shadow_queue_collapse",
        "shadow_aftershock_prob", "eai", "long_usd_1m", "short_usd_1m",
    ]
    out = df[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return out


def _load_param_sets() -> tuple[list[dict], list[dict]]:
    bal = json.load(open(ROOT / "data/ensemble/metrics/param_ensemble_result.json", "r", encoding="utf-8"))
    low = json.load(open(ROOT / "data/ensemble/metrics/param_ensemble_lowfreq_highpnl.json", "r", encoding="utf-8"))
    bal_params = [x["params"] for x in bal.get("top_params", [])][:10]
    low_params = [x["params"] for x in low.get("top10_singles", [])][:10]
    return bal_params, low_params


def _default_msaf_params() -> dict:
    return {
        "theta_direction": 0.75,
        "theta_structure": 0.44,
        "theta_risk": 0.8,
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
        "tox_veto": 0.9,
        "qc_veto": 0.9,
        "aft_veto": 0.9,
    }


def _as_report_row(name: str, r: dict) -> dict:
    return {
        "formula": name,
        "pnl_pct": float(r["pnl_pct"]),
        "mdd_pct": float(r["mdd_pct"]),
        "trades": int(r["trades"]),
        "win_rate": float(r["win_rate"]),
        "sharpe": float(r["sharpe"]),
        "equity": float(r["equity"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/splits/year_oos/rl_training_2025_m7.csv")
    ap.add_argument("--fee-bps", type=float, default=2.0)
    ap.add_argument("--slip-bps", type=float, default=1.0)
    ap.add_argument("--lev", type=float, default=1.0)
    ap.add_argument("--out", default="data/ensemble/metrics/quant_formulas_2025_rl_compare.json")
    args = ap.parse_args()

    m = load_mapped_2025(args.csv)
    bal_params, low_params = _load_param_sets()

    res = {
        "meta": {
            "rows": int(len(m)),
            "start": str(m["ts"].min()),
            "end": str(m["ts"].max()),
            "fee_bps": float(args.fee_bps),
            "slip_bps": float(args.slip_bps),
            "lev": float(args.lev),
            "source_csv": args.csv,
            "mapping": "rl_training_2025_m7 -> microstructure proxy",
        },
        "results": [],
    }

    # Ensemble formulas
    if bal_params:
        r = _ensemble_backtest(m, bal_params, min_votes=6, exit_on_hold=True)
        res["results"].append(_as_report_row("BALANCED_ENSEMBLE_TOP10_VOTE6", r))
        r1 = _ensemble_backtest(m, [bal_params[0]], min_votes=1, exit_on_hold=True)
        res["results"].append(_as_report_row("BALANCED_SINGLE_BEST", r1))
    if low_params:
        r = _ensemble_backtest(m, low_params, min_votes=7, exit_on_hold=True)
        res["results"].append(_as_report_row("LOWFREQ_ENSEMBLE_TOP10_VOTE7", r))
        r1 = _ensemble_backtest(m, [low_params[0]], min_votes=1, exit_on_hold=True)
        res["results"].append(_as_report_row("LOWFREQ_SINGLE_BEST", r1))

    # MSAF formulas
    msaf_default = run_msaf_sim(m, _default_msaf_params(), fee_bps=args.fee_bps, slip_bps=args.slip_bps, leverage=args.lev)
    res["results"].append(
        {
            "formula": "MSAF_DEFAULT",
            "pnl_pct": msaf_default.pnl_pct,
            "mdd_pct": msaf_default.mdd_pct,
            "trades": msaf_default.trades,
            "win_rate": msaf_default.win_rate_pct,
            "sharpe": msaf_default.sharpe,
            "equity": msaf_default.equity_final,
        }
    )

    msaf_tuned_path = ROOT / "data/ensemble/metrics/msaf_best_candidate_7d.json"
    if msaf_tuned_path.exists():
        msaf_tuned = json.load(open(msaf_tuned_path, "r", encoding="utf-8")).get("params", {})
        required = {"theta_direction", "theta_structure", "theta_risk", "gamma", "beta"}
        if isinstance(msaf_tuned, dict) and msaf_tuned and required.issubset(msaf_tuned.keys()):
            rr = run_msaf_sim(m, msaf_tuned, fee_bps=args.fee_bps, slip_bps=args.slip_bps, leverage=args.lev)
            res["results"].append(
                {
                    "formula": "MSAF_TUNED_7D",
                    "pnl_pct": rr.pnl_pct,
                    "mdd_pct": rr.mdd_pct,
                    "trades": rr.trades,
                    "win_rate": rr.win_rate_pct,
                    "sharpe": rr.sharpe,
                    "equity": rr.equity_final,
                }
            )

    res["results"] = sorted(res["results"], key=lambda x: x["pnl_pct"], reverse=True)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
