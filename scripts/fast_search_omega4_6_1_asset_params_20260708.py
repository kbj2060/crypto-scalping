#!/usr/bin/env python3
"""Fast ledger-level parameter search for SOL/BTC Omega4.6.1 replicas.

This is a candidate-selection search over existing real-stack risk ledgers.
It recomputes trade_return after a final scale-map using saved
net_per_notional and margin_fraction. Because notional/leverage can be inputs
to the learned exit head, this is not a promotion-grade replay; use it to pick
asset-specific component/quality/scale/duration candidates before exact replay.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASSET_DATES = {"sol": "20260707", "btc": "20260708"}
LEVERAGE_CAP = 5.0
NOTIONAL_CAP = 1.8


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _compound_metrics(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0, "avg_notional": 0.0, "max_leverage": 0.0, "max_notional": 0.0}
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    for ret in ledger["trade_return"].to_numpy(dtype=np.float64):
        cash *= 1.0 + float(ret)
        wins += int(ret > 0.0)
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(len(ledger)),
        "wr": float(wins / len(ledger)),
        "avg_notional": float(ledger["notional"].mean()),
        "max_leverage": float(ledger["leverage"].max()),
        "max_notional": float(ledger["notional"].max()),
    }


def _risk_dirs(asset: str) -> list[Path]:
    date = ASSET_DATES[asset]
    base = ROOT / "tmp/causal_regen_20260516"
    return sorted(
        d
        for d in base.glob(f"{asset}_omega4_2_trade_risk_sidecar_{date}_*_q*_{date}*")
        if d.is_dir()
        and (d / "validation_selected_risk_replayed_trade_ledger.csv").exists()
        and (d / "oos_selected_risk_replayed_trade_ledger.csv").exists()
    )


def _parse(path: Path) -> tuple[str, str]:
    m = re.search(r"_(h48qual|zig075)_q(\d{3})_", path.as_posix())
    if not m:
        raise RuntimeError(f"cannot parse candidate from {path}")
    return m.group(1), f"q{m.group(2)}"


def _load_ledger(path: Path, features_path: Path) -> pd.DataFrame:
    ledger = pd.read_csv(path, parse_dates=["entry_timestamp"])
    feats = pd.read_csv(features_path, usecols=["timestamp", "ou_halflife"], parse_dates=["timestamp"])
    out = ledger.merge(feats.rename(columns={"timestamp": "entry_timestamp"}), on="entry_timestamp", how="left", validate="one_to_one")
    if out["ou_halflife"].isna().any():
        raise RuntimeError(f"{path}: ou_halflife merge produced NaN")
    return out


def _apply_scale(ledger: pd.DataFrame, *, long_scale: float, short_scale: float) -> pd.DataFrame:
    out = ledger.copy()
    side = pd.to_numeric(out["side"], errors="raise").to_numpy(dtype=np.int64)
    margin = pd.to_numeric(out["margin_fraction"], errors="raise").to_numpy(dtype=np.float64)
    lev = pd.to_numeric(out["leverage"], errors="raise").to_numpy(dtype=np.float64)
    scale = np.where(side > 0, float(long_scale), np.where(side < 0, float(short_scale), 1.0))
    lev2 = np.minimum(lev * scale, LEVERAGE_CAP)
    notional = np.minimum(margin * lev2, NOTIONAL_CAP)
    lev2 = np.where(margin > 0.0, notional / np.maximum(margin, 1e-12), lev2)
    out["leverage"] = lev2
    out["notional"] = notional
    out["trade_return"] = pd.to_numeric(out["net_per_notional"], errors="raise").to_numpy(dtype=np.float64) * notional
    return out


def _duration_variants(ledger: pd.DataFrame, *, min_trade_ratio: float) -> list[tuple[float, pd.DataFrame]]:
    out = [(0.0, ledger)]
    if ledger.empty:
        return out
    floor = max(1, int(np.floor(len(ledger) * min_trade_ratio)))
    for q in np.arange(0.05, 0.85, 0.05):
        th = float(np.quantile(ledger["ou_halflife"].to_numpy(dtype=np.float64), q))
        gated = ledger.loc[ledger["ou_halflife"] > th].reset_index(drop=True)
        if len(gated) >= floor:
            out.append((th, gated))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", choices=sorted(ASSET_DATES), required=True)
    ap.add_argument("--scale-grid", default="0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0")
    ap.add_argument("--max-validation-mdd-abs", type=float, default=30.0)
    ap.add_argument("--min-trade-ratio", type=float, default=0.50)
    ap.add_argument("--min-validation-trades", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    asset = args.asset
    date = ASSET_DATES[asset]
    out_dir = args.out_dir or ROOT / "tmp/causal_regen_20260516" / f"{asset}_omega4_6_1_fast_param_search_{date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    val_features = ROOT / f"data/splits/year_oos/{asset}_features_2025.csv"
    oos_features = ROOT / f"data/splits/year_oos/{asset}_features_2026.csv"
    scales = [float(x) for x in str(args.scale_grid).split(",") if x.strip()]

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for d in _risk_dirs(asset):
        component, tag = _parse(d)
        val = _load_ledger(d / "validation_selected_risk_replayed_trade_ledger.csv", val_features)
        for long_scale in scales:
            for short_scale in scales:
                scaled = _apply_scale(val, long_scale=long_scale, short_scale=short_scale)
                for duration_threshold, gated in _duration_variants(scaled, min_trade_ratio=float(args.min_trade_ratio)):
                    metrics = _compound_metrics(gated)
                    eligible = (
                        metrics["trades"] >= int(args.min_validation_trades)
                        and metrics["mdd"] >= -abs(float(args.max_validation_mdd_abs))
                    )
                    row = {
                        "asset": asset,
                        "component": component,
                        "quality_tag": tag,
                        "quality_threshold": int(tag[1:]) / 100.0,
                        "long_scale": float(long_scale),
                        "short_scale": float(short_scale),
                        "duration_threshold": float(duration_threshold),
                        "validation": metrics,
                        "eligible": bool(eligible),
                        "risk_dir": str(d),
                    }
                    rows.append(row)
                    if eligible and (best is None or metrics["pnl"] > best["validation"]["pnl"]):
                        best = row
    if best is None:
        raise RuntimeError("no eligible candidate")

    oos = _load_ledger(Path(best["risk_dir"]) / "oos_selected_risk_replayed_trade_ledger.csv", oos_features)
    oos_scaled = _apply_scale(oos, long_scale=float(best["long_scale"]), short_scale=float(best["short_scale"]))
    oos_gated = oos_scaled.loc[oos_scaled["ou_halflife"] > float(best["duration_threshold"])].reset_index(drop=True)
    oos_frozen = oos_gated.loc[oos_gated["entry_timestamp"] < pd.Timestamp("2026-04-01")].reset_index(drop=True)
    report = {
        "method": "fast_ledger_level_omega4_6_1_asset_param_search",
        "promotion_grade": False,
        "caveat": "Uses existing real-stack ledgers and recomputes scale-map PnL from net_per_notional; exact replay is still needed if selected notional/leverage changes exit-head timing.",
        "asset": asset,
        "search_space": {
            "risk_dirs": [str(d) for d in _risk_dirs(asset)],
            "scale_grid": scales,
            "duration_quantiles": "0.05..0.80 by 0.05 plus no-gate",
            "leverage_cap": LEVERAGE_CAP,
            "notional_cap": NOTIONAL_CAP,
            "max_validation_mdd_abs": float(args.max_validation_mdd_abs),
            "min_trade_ratio": float(args.min_trade_ratio),
            "min_validation_trades": int(args.min_validation_trades),
        },
        "selected": best,
        "oos_one_shot": _compound_metrics(oos_gated),
        "oos_frozen_q1_2026": _compound_metrics(oos_frozen),
        "candidate_count": len(rows),
    }
    pd.DataFrame(rows).to_csv(out_dir / "candidate_grid.csv", index=False)
    oos_gated.to_csv(out_dir / "selected_oos_gated_ledger.csv", index=False)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
