#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    pnl_pct: float
    mdd_pct: float
    trades: int
    wr_pct: float
    avg_base_lev: float
    avg_new_exposure: float
    avg_leverage_mult: float
    avg_fraction_mult: float


def _safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return x


def _regime_name(row: pd.Series) -> str:
    if _safe_float(row.get("regime_bull", 0.0)) >= 0.5:
        return "bull"
    if _safe_float(row.get("regime_bear", 0.0)) >= 0.5:
        return "bear"
    if _safe_float(row.get("regime_chop", 0.0)) >= 0.5:
        return "chop"
    if _safe_float(row.get("regime_whipsaw", 0.0)) >= 0.5:
        return "whipsaw"
    return "normal"


def _load_trade_payload(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return list((((payload or {}).get("extra", {}) or {}).get("trades", [])) or [])


def _load_frame(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def _policy(row: pd.Series, side: str, base_lev: float, policy_name: str, equity: float, peak: float) -> tuple[float, float, float]:
    regime = _regime_name(row)
    conf = _safe_float(row.get("m7_confidence", 0.0))
    qwidth = _safe_float(row.get("m7_qwidth", 0.0))
    vol_z = abs(_safe_float(row.get("volatility_z", 0.0)))
    smf = _safe_float(row.get("smart_money_flow", 0.0))
    whale = _safe_float(row.get("whale_conviction", 0.0))
    funding_div = _safe_float(row.get("funding_price_divergence", 0.0))
    side_sign = 1.0 if side == "LONG" else -1.0
    aligned = (side == "LONG" and regime == "bull") or (side == "SHORT" and regime == "bear")
    flow = side_sign * (0.55 * smf + 0.35 * whale + 0.10 * funding_div)
    good = aligned and conf > 0.58 and qwidth < 0.0065 and vol_z < 1.0 and flow > 0.0
    great = aligned and conf > 0.68 and qwidth < 0.0055 and vol_z < 0.85 and flow > 0.03
    bad = regime in {"chop", "whipsaw"} or conf < 0.42 or qwidth > 0.010 or vol_z > 1.6
    very_bad = regime == "whipsaw" or qwidth > 0.014 or vol_z > 2.3
    drawdown = 1.0 - (equity / max(peak, 1e-8))
    quality_core = (
        1.35 * conf
        - 0.85 * min(qwidth / 0.010, 2.0)
        - 0.55 * min(vol_z / 1.8, 2.0)
        + 0.45 * (1.0 if aligned else -0.35)
        + 0.60 * np.tanh(flow / 0.08)
    )
    stability_core = (
        1.10
        - 0.90 * min(qwidth / 0.011, 2.0)
        - 0.75 * min(vol_z / 1.9, 2.0)
        - 0.45 * (1.0 if regime in {"chop", "whipsaw"} else 0.0)
        - 0.55 * drawdown
    )
    convex_gain = max(0.0, quality_core - 0.55)
    stability = 1.0 / (1.0 + np.exp(-stability_core))
    quality = np.tanh(quality_core)

    fraction_mult = 1.0
    leverage_mult = 1.0

    if policy_name == "baseline":
        pass
    elif policy_name == "convex_confidence":
        if very_bad:
            fraction_mult *= 0.78
            leverage_mult *= 0.92
        elif bad:
            fraction_mult *= 0.90
            leverage_mult *= 0.97
        elif great:
            fraction_mult *= 1.03
            leverage_mult *= 1.18
        elif good:
            fraction_mult *= 1.01
            leverage_mult *= 1.10
    elif policy_name == "drawdown_controlled":
        if very_bad:
            fraction_mult *= 0.80
            leverage_mult *= 0.90
        elif bad:
            fraction_mult *= 0.92
            leverage_mult *= 0.96
        elif great:
            fraction_mult *= 1.02
            leverage_mult *= 1.14
        elif good:
            fraction_mult *= 1.01
            leverage_mult *= 1.08
        if drawdown >= 0.025:
            fraction_mult *= 0.95
        if drawdown >= 0.04:
            fraction_mult *= 0.92
            leverage_mult *= 0.97
    elif policy_name == "barbell_quality":
        if very_bad:
            fraction_mult *= 0.74
            leverage_mult *= 0.92
        elif bad:
            fraction_mult *= 0.88
            leverage_mult *= 0.98
        elif great:
            fraction_mult *= 1.05
            leverage_mult *= 1.22
        elif good:
            fraction_mult *= 1.02
            leverage_mult *= 1.12
    elif policy_name == "soft_plus":
        if very_bad:
            fraction_mult *= 0.90
            leverage_mult *= 0.96
        elif bad:
            fraction_mult *= 0.97
        elif great:
            fraction_mult *= 1.02
            leverage_mult *= 1.10
        elif good:
            leverage_mult *= 1.05
    elif policy_name == "alpha_ramp":
        if very_bad:
            fraction_mult *= 0.93
            leverage_mult *= 0.98
        elif bad:
            fraction_mult *= 0.98
        elif great:
            fraction_mult *= 1.04
            leverage_mult *= 1.28
        elif good:
            fraction_mult *= 1.02
            leverage_mult *= 1.14
        elif aligned and conf > 0.56 and qwidth < 0.008:
            leverage_mult *= 1.06
    elif policy_name == "trend_pyramidal":
        if very_bad:
            fraction_mult *= 0.95
            leverage_mult *= 0.99
        elif bad:
            fraction_mult *= 0.99
        elif great:
            fraction_mult *= 1.05
            leverage_mult *= 1.32
        elif good:
            fraction_mult *= 1.03
            leverage_mult *= 1.18
        elif aligned and flow > 0.02 and conf > 0.60:
            fraction_mult *= 1.01
            leverage_mult *= 1.10
    elif policy_name == "triple_ramp":
        if very_bad:
            fraction_mult *= 0.96
            leverage_mult *= 0.98
        elif bad:
            fraction_mult *= 0.995
        elif great:
            fraction_mult *= 1.08
            leverage_mult *= 2.10
        elif good:
            fraction_mult *= 1.04
            leverage_mult *= 1.55
        elif aligned and flow > 0.03 and conf > 0.62 and qwidth < 0.0075:
            fraction_mult *= 1.02
            leverage_mult *= 1.25
    elif policy_name == "triple_selective":
        if very_bad:
            fraction_mult *= 0.97
            leverage_mult *= 0.99
        elif bad:
            fraction_mult *= 1.00
        elif great:
            fraction_mult *= 1.10
            leverage_mult *= 2.55
        elif good:
            fraction_mult *= 1.05
            leverage_mult *= 1.75
        elif aligned and flow > 0.05 and conf > 0.70 and qwidth < 0.0065 and vol_z < 0.8:
            fraction_mult *= 1.03
            leverage_mult *= 1.35
    elif policy_name == "triple_convex":
        if very_bad:
            fraction_mult *= 0.94
            leverage_mult *= 0.97
        elif bad:
            fraction_mult *= 0.99
        elif great:
            fraction_mult *= 1.12
            leverage_mult *= 2.85
        elif good:
            fraction_mult *= 1.06
            leverage_mult *= 1.95
        elif aligned and flow > 0.02 and conf > 0.66:
            fraction_mult *= 1.02
            leverage_mult *= 1.28
    elif policy_name == "manifold_kelly":
        fraction_mult *= float(np.clip(0.82 + 0.46 * stability + 0.12 * max(quality, 0.0), 0.72, 1.18))
        leverage_mult *= float(np.clip(0.92 + 0.36 * max(quality, 0.0) + 0.22 * convex_gain**2, 0.90, 1.95))
        if very_bad:
            fraction_mult *= 0.94
            leverage_mult *= 0.95
    elif policy_name == "curvature_tensor":
        tensor_score = (
            0.90 * max(quality, 0.0) ** 2
            + 0.55 * stability
            + 0.35 * max(flow, 0.0)
            + 0.20 * (1.0 if aligned else 0.0)
        )
        fraction_mult *= float(np.clip(0.78 + 0.42 * stability + 0.18 * max(quality, 0.0), 0.70, 1.16))
        leverage_mult *= float(np.clip(0.94 + 0.22 * tensor_score + 0.18 * convex_gain**2, 0.92, 2.25))
        if bad:
            fraction_mult *= 0.97
    elif policy_name == "integral_kelly":
        integral_score = (
            0.70 * conf
            + 0.55 * np.tanh(flow / 0.06)
            + 0.35 * (1.0 if aligned else -0.2)
            - 0.50 * min(qwidth / 0.009, 2.0)
            - 0.35 * min(vol_z / 1.5, 2.0)
        )
        positive_band = max(0.0, integral_score - 0.35)
        fraction_mult *= float(np.clip(0.84 + 0.30 * stability + 0.16 * positive_band, 0.75, 1.14))
        leverage_mult *= float(np.clip(0.96 + 0.28 * positive_band + 0.32 * positive_band**2, 0.94, 2.60))
        if very_bad:
            leverage_mult *= 0.94
    elif policy_name == "flow_impulse":
        if very_bad:
            fraction_mult *= 0.99
            leverage_mult *= 0.98
        elif bad:
            fraction_mult *= 0.995
        strong_base = base_lev > 0.78
        elite_base = base_lev > 0.90
        if flow > 0.075 and elite_base:
            leverage_mult *= 1.85
            fraction_mult *= 1.02
        elif flow > 0.045 and strong_base and conf > 0.55:
            leverage_mult *= 1.35
            fraction_mult *= 1.01
        elif flow > 0.025 and aligned and base_lev > 0.65:
            leverage_mult *= 1.12
    elif policy_name == "alpha_focus":
        if very_bad:
            fraction_mult *= 0.985
            leverage_mult *= 0.98
        elif bad:
            fraction_mult *= 0.998
        strong_base = base_lev > 0.72
        elite_combo = flow > 0.05 and conf > 0.62 and base_lev > 0.82
        if flow > 0.09 and base_lev > 0.92:
            leverage_mult *= 2.05
            fraction_mult *= 1.02
        elif elite_combo:
            leverage_mult *= 1.48
            fraction_mult *= 1.01
        elif strong_base and flow > 0.03 and (aligned or conf > 0.70):
            leverage_mult *= 1.18
        if flow < -0.04 and not aligned:
            fraction_mult *= 0.99
    else:
        raise ValueError(policy_name)

    new_exposure = float(np.clip(base_lev * fraction_mult * leverage_mult, 0.03, 3.0))
    return new_exposure, float(fraction_mult), float(leverage_mult)


def evaluate(trades: list[dict], df: pd.DataFrame, policy_name: str) -> dict:
    ts_map = {pd.Timestamp(ts): i for i, ts in enumerate(df["timestamp"])}
    balance = 1.0
    eq_curve = [balance]
    wins = 0
    base_levs: list[float] = []
    new_exposures: list[float] = []
    lev_mults: list[float] = []
    frac_mults: list[float] = []
    out_rows: list[dict] = []
    peak = balance

    for tr in trades:
        exit_ts = pd.Timestamp(tr["ts"])
        hold_bars = int(tr.get("hold_bars", 0) or 0)
        entry_ts = exit_ts - pd.Timedelta(minutes=5 * max(hold_bars, 0))
        entry_idx = ts_map.get(entry_ts)
        if entry_idx is None:
            maybe = df.index[df["timestamp"] <= entry_ts]
            entry_idx = int(maybe[-1]) if len(maybe) else 0
        row = df.iloc[entry_idx]

        side = str(tr.get("side", "LONG") or "LONG").upper()
        base_lev = max(_safe_float(tr.get("lev", 0.0)), 1e-8)
        base_pnl = _safe_float(tr.get("pnl_frac", 0.0))
        unit_pnl = base_pnl / base_lev

        new_exposure, frac_mult, lev_mult = _policy(row, side, base_lev, policy_name, balance, peak)
        new_pnl = unit_pnl * new_exposure

        balance *= 1.0 + new_pnl
        peak = max(peak, balance)
        eq_curve.append(balance)
        wins += int(new_pnl > 0.0)

        base_levs.append(base_lev)
        new_exposures.append(new_exposure)
        lev_mults.append(lev_mult)
        frac_mults.append(frac_mult)
        out_rows.append(
            {
                "ts": str(exit_ts),
                "side": side,
                "base_lev": base_lev,
                "new_exposure": new_exposure,
                "fraction_mult": frac_mult,
                "leverage_mult": lev_mult,
                "base_pnl_frac": base_pnl,
                "new_pnl_frac": new_pnl,
                "regime": _regime_name(row),
                "m7_confidence": _safe_float(row.get("m7_confidence", 0.0)),
                "m7_qwidth": _safe_float(row.get("m7_qwidth", 0.0)),
                "volatility_z": _safe_float(row.get("volatility_z", 0.0)),
            }
        )

    eq = np.asarray(eq_curve, dtype=float)
    peaks = np.maximum.accumulate(eq)
    drawdown = (eq / np.maximum(peaks, 1e-12)) - 1.0
    metrics = Metrics(
        pnl_pct=float((balance - 1.0) * 100.0),
        mdd_pct=float(drawdown.min() * 100.0),
        trades=len(trades),
        wr_pct=float((wins / max(len(trades), 1)) * 100.0),
        avg_base_lev=float(np.mean(base_levs) if base_levs else 0.0),
        avg_new_exposure=float(np.mean(new_exposures) if new_exposures else 0.0),
        avg_leverage_mult=float(np.mean(lev_mults) if lev_mults else 0.0),
        avg_fraction_mult=float(np.mean(frac_mults) if frac_mults else 0.0),
    )
    return {"policy": policy_name, "metrics": asdict(metrics), "trades_sample": out_rows[:10]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-json", required=True)
    ap.add_argument("--csv-path", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    trades = _load_trade_payload(args.baseline_json)
    df = _load_frame(args.csv_path)
    policies = [
        "baseline",
        "soft_plus",
        "alpha_ramp",
        "trend_pyramidal",
        "triple_ramp",
        "triple_selective",
        "triple_convex",
        "manifold_kelly",
        "curvature_tensor",
        "integral_kelly",
        "flow_impulse",
        "alpha_focus",
        "convex_confidence",
        "drawdown_controlled",
        "barbell_quality",
    ]
    payload = {
        "baseline_json": args.baseline_json,
        "csv_path": args.csv_path,
        "trades": len(trades),
        "results": [evaluate(trades, df, p) for p in policies],
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload["results"], ensure_ascii=False, indent=2))
    print(f"\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()
