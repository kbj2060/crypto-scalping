#!/usr/bin/env python3
"""VAL-only parameter search for SOL/BTC Omega4.6.1 replication candidates.

This uses already-trained per-asset parent bundles and risk sidecars, then
replays the real ATR/exit-head/risk stack while searching:
- component/quality tag, via existing risk_sidecar.pkl artifacts
- final long/short leverage scale-map
- entry duration gate threshold on ou_halflife

OOS is replayed once for the selected VAL configuration.
"""
from __future__ import annotations

import argparse
import importlib
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402

LEVERAGE_CAP = 5.0
NOTIONAL_CAP = 1.8
ATR_KWARGS = {
    "atr_window": 192,
    "tp_mult": 12.0,
    "sl_mult": 6.0,
    "min_tp": 0.075,
    "min_sl": 0.040,
    "max_tp": 0.22,
    "max_sl": 0.12,
}

ASSET_DATES = {"sol": "20260707", "btc": "20260708"}


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
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0}
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
    }


def _scaled_margin_leverage(
    dec: pd.DataFrame,
    base_margin: np.ndarray,
    base_leverage: np.ndarray,
    *,
    long_scale: float,
    short_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    scale = np.where(side > 0, float(long_scale), np.where(side < 0, float(short_scale), 1.0))
    leverage = np.minimum(base_leverage * scale, LEVERAGE_CAP)
    notional = np.minimum(base_margin * leverage, NOTIONAL_CAP)
    with np.errstate(divide="ignore", invalid="ignore"):
        leverage = np.where(base_margin > 0.0, notional / np.maximum(base_margin, 1e-12), leverage)
    return base_margin, leverage


def _candidate_key(path: Path) -> tuple[str, str]:
    m = re.search(r"_(h48qual|zig075)_q(\d{3})_", path.as_posix())
    if not m:
        raise RuntimeError(f"cannot parse component/qtag from {path}")
    return m.group(1), f"q{m.group(2)}"


def _parent_dir(asset: str, component: str) -> Path:
    date = ASSET_DATES[asset]
    return ROOT / "tmp/causal_regen_20260516" / f"{asset}_omega4_3head_parent72_loose_entry_quality_{date}_{component}_{date}"


def _risk_sidecar_dirs(asset: str) -> list[Path]:
    date = ASSET_DATES[asset]
    base = ROOT / "tmp/causal_regen_20260516"
    dirs = sorted(base.glob(f"{asset}_omega4_2_trade_risk_sidecar_{date}_*_q*_{date}"))
    return [d for d in dirs if (d / "risk_sidecar.pkl").exists()]


def _duration_candidates(ledger: pd.DataFrame, *, min_trade_ratio: float) -> list[tuple[float, pd.DataFrame]]:
    out: list[tuple[float, pd.DataFrame]] = [(0.0, ledger)]
    if ledger.empty:
        return out
    trade_floor = max(1, int(np.floor(len(ledger) * float(min_trade_ratio))))
    thresholds = sorted(set(float(np.quantile(ledger["ou_halflife"].to_numpy(dtype=np.float64), q)) for q in np.arange(0.05, 0.85, 0.05)))
    for th in thresholds:
        gated = ledger.loc[ledger["ou_halflife"] > th].reset_index(drop=True)
        if len(gated) >= trade_floor:
            out.append((th, gated))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", choices=sorted(ASSET_DATES), required=True)
    ap.add_argument("--scale-grid", default="0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0")
    ap.add_argument("--component", choices=["h48qual", "zig075"], default=None)
    ap.add_argument("--quality-tag", default=None, help="Optional qXXX tag filter, e.g. q070")
    ap.add_argument("--max-validation-mdd-abs", type=float, default=30.0)
    ap.add_argument("--min-trade-ratio", type=float, default=0.50)
    ap.add_argument("--exit-threshold", type=float, default=0.95)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    asset = args.asset
    date = ASSET_DATES[asset]
    out_dir = args.out_dir or ROOT / "tmp/causal_regen_20260516" / f"{asset}_omega4_6_1_param_search_{date}"
    out_dir.mkdir(parents=True, exist_ok=True)

    omega = importlib.import_module(f"train_eval_omega1_2_tabm_diffusion_risk_{asset}_{date}")
    omega4 = importlib.import_module(f"train_eval_omega4_3head_parent72_loose_entry_quality_{asset}_{date}")
    sidecar = importlib.import_module(f"train_eval_omega4_2_risk_sidecar_{asset}_{date}")

    device = parent._device(str(args.device))
    scale_values = [float(x) for x in str(args.scale_grid).split(",") if x.strip()]

    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=omega4.LABEL_DIR,
        quality_mode="same_as_direction",
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    fee, slip = omega._load_fee_slip()

    val_ou = frames["val_raw"][["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp"})
    oos_ou = frames["oos_raw"][["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp"})

    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    risk_dirs = []
    for d in _risk_sidecar_dirs(asset):
        c, t = _candidate_key(d)
        if args.component is not None and c != args.component:
            continue
        if args.quality_tag is not None and t != str(args.quality_tag):
            continue
        risk_dirs.append(d)
    if not risk_dirs:
        raise RuntimeError("no risk sidecar dirs match the requested filters")

    for risk_dir in risk_dirs:
        component, tag = _candidate_key(risk_dir)
        pred_dir = _parent_dir(asset, component)
        bundle_path = pred_dir / "true_3head_tabm_bundle.pt"
        if not bundle_path.exists():
            raise FileNotFoundError(bundle_path)
        print(f"stage=candidate asset={asset} component={component} tag={tag}", flush=True)

        bundle = torch.load(bundle_path, map_location=device, weights_only=False)
        models: dict[str, Any] = bundle["models"]
        base_cols = list(bundle["base_cols"])
        loaded = parent._load_payloads(models, device=device)
        with open(risk_dir / "risk_sidecar.pkl", "rb") as f:
            pkl = pickle.load(f)

        val_src = sidecar._load_precomputed_prediction(pred_dir, "validation", tag, frames["val_raw"])
        oos_src = sidecar._load_precomputed_prediction(pred_dir, "oos", tag, frames["oos_raw"])
        x_val = parent._base_input(frames["val_raw"], base_cols)
        x_oos = parent._base_input(frames["oos_raw"], base_cols)
        val_dec_base = parent._to_decisions(val_src, oof=True)
        oos_dec_base = parent._to_decisions(oos_src, oof=False)
        val_dec, _ = atr_eval._apply_atr_safety_sltp(val_dec_base, frames["val_raw"], **ATR_KWARGS)
        oos_dec, _ = atr_eval._apply_atr_safety_sltp(oos_dec_base, frames["oos_raw"], **ATR_KWARGS)
        val_atr = atr_eval._atr_pct(frames["val_raw"], 192)
        oos_atr = atr_eval._atr_pct(frames["oos_raw"], 192)

        val_features = sidecar._risk_feature_frame(frames["val_raw"], val_src, val_dec, base_cols, atr_pct=val_atr, feature_mode=pkl["risk_feature_mode"])
        oos_features = sidecar._risk_feature_frame(frames["oos_raw"], oos_src, oos_dec, base_cols, atr_pct=oos_atr, feature_mode=pkl["risk_feature_mode"])
        x_val_all, _ = sidecar._feature_matrix(val_features, pkl["feature_columns"])
        x_oos_all, _ = sidecar._feature_matrix(oos_features, pkl["feature_columns"])
        val_side_all = pd.to_numeric(val_dec["side"], errors="raise").to_numpy(dtype=np.int64)
        oos_side_all = pd.to_numeric(oos_dec["side"], errors="raise").to_numpy(dtype=np.int64)
        if pkl["side_split_model"]:
            val_score = sidecar._predict_side_split_models(pkl["model"], x_val_all, val_side_all)
            oos_score = sidecar._predict_side_split_models(pkl["model"], x_oos_all, oos_side_all)
        else:
            val_score = np.asarray(pkl["model"].predict(x_val_all), dtype=np.float64)
            oos_score = np.asarray(pkl["model"].predict(x_oos_all), dtype=np.float64)
        mapping = pkl["selected_mapping"]
        val_base_margin = sidecar._risk_margins(val_dec, val_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS})
        oos_base_margin = sidecar._risk_margins(oos_dec, oos_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS})
        if pkl["dynamic_leverage"]:
            val_base_leverage = sidecar._risk_leverage(val_dec, val_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS})
            oos_base_leverage = sidecar._risk_leverage(oos_dec, oos_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS})
        else:
            val_base_leverage = np.ones(len(val_dec), dtype=np.float64)
            oos_base_leverage = np.ones(len(oos_dec), dtype=np.float64)

        def replay(dec: pd.DataFrame, frame: pd.DataFrame, x: pd.DataFrame, margin: np.ndarray, leverage: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
            return sidecar._replay_with_risk(
                frame,
                x,
                dec,
                loaded,
                risk_margin_fraction=margin,
                risk_leverage=leverage,
                exit_threshold=float(args.exit_threshold),
                fee=fee,
                slip=slip,
                cost_mult=float(args.cost_mult),
                notional_scaled_sltp=False,
                exit_sizing_input_mode="actual",
                device=device,
            )

        for long_scale in scale_values:
            for short_scale in scale_values:
                val_margin, val_leverage = _scaled_margin_leverage(
                    val_dec,
                    val_base_margin,
                    val_base_leverage,
                    long_scale=long_scale,
                    short_scale=short_scale,
                )
                _, val_ledger = replay(val_dec, frames["val_raw"], x_val, val_margin, val_leverage)
                val_ledger = val_ledger.copy()
                val_ledger["entry_timestamp"] = pd.to_datetime(val_ledger["entry_timestamp"])
                val_ledger = val_ledger.merge(val_ou, on="entry_timestamp", how="left", validate="one_to_one")
                if val_ledger["ou_halflife"].isna().any():
                    raise RuntimeError("VAL ou_halflife merge produced NaN")
                for duration_threshold, gated in _duration_candidates(val_ledger, min_trade_ratio=float(args.min_trade_ratio)):
                    val_m = _compound_metrics(gated)
                    eligible = int(val_m["trades"]) > 0 and float(val_m["mdd"]) >= -abs(float(args.max_validation_mdd_abs))
                    row = {
                        "asset": asset,
                        "component": component,
                        "quality_tag": tag,
                        "quality_threshold": int(tag[1:]) / 100.0,
                        "long_scale": float(long_scale),
                        "short_scale": float(short_scale),
                        "duration_threshold": float(duration_threshold),
                        "validation": val_m,
                        "eligible": bool(eligible),
                        "risk_dir": str(risk_dir),
                        "parent_dir": str(pred_dir),
                    }
                    candidates.append(row)
                    if eligible and (best is None or float(val_m["pnl"]) > float(best["validation"]["pnl"])):
                        best = row

        if best and best["component"] == component and best["quality_tag"] == tag:
            best["bundle_path"] = str(bundle_path)

    if best is None:
        raise RuntimeError("no eligible candidate found")

    print(f"stage=selected {best}", flush=True)

    # Replay OOS once for the selected candidate.
    component = str(best["component"])
    tag = str(best["quality_tag"])
    pred_dir = _parent_dir(asset, component)
    risk_dir = Path(str(best["risk_dir"]))
    bundle = torch.load(pred_dir / "true_3head_tabm_bundle.pt", map_location=device, weights_only=False)
    models = bundle["models"]
    base_cols = list(bundle["base_cols"])
    loaded = parent._load_payloads(models, device=device)
    with open(risk_dir / "risk_sidecar.pkl", "rb") as f:
        pkl = pickle.load(f)
    oos_src = sidecar._load_precomputed_prediction(pred_dir, "oos", tag, frames["oos_raw"])
    x_oos = parent._base_input(frames["oos_raw"], base_cols)
    oos_dec_base = parent._to_decisions(oos_src, oof=False)
    oos_dec, _ = atr_eval._apply_atr_safety_sltp(oos_dec_base, frames["oos_raw"], **ATR_KWARGS)
    oos_atr = atr_eval._atr_pct(frames["oos_raw"], 192)
    oos_features = sidecar._risk_feature_frame(frames["oos_raw"], oos_src, oos_dec, base_cols, atr_pct=oos_atr, feature_mode=pkl["risk_feature_mode"])
    x_oos_all, _ = sidecar._feature_matrix(oos_features, pkl["feature_columns"])
    oos_side_all = pd.to_numeric(oos_dec["side"], errors="raise").to_numpy(dtype=np.int64)
    if pkl["side_split_model"]:
        oos_score = sidecar._predict_side_split_models(pkl["model"], x_oos_all, oos_side_all)
    else:
        oos_score = np.asarray(pkl["model"].predict(x_oos_all), dtype=np.float64)
    mapping = pkl["selected_mapping"]
    oos_base_margin = sidecar._risk_margins(oos_dec, oos_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS})
    oos_base_leverage = sidecar._risk_leverage(oos_dec, oos_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS}) if pkl["dynamic_leverage"] else np.ones(len(oos_dec), dtype=np.float64)
    oos_margin, oos_leverage = _scaled_margin_leverage(
        oos_dec,
        oos_base_margin,
        oos_base_leverage,
        long_scale=float(best["long_scale"]),
        short_scale=float(best["short_scale"]),
    )
    _, oos_ledger = sidecar._replay_with_risk(
        frames["oos_raw"],
        x_oos,
        oos_dec,
        loaded,
        risk_margin_fraction=oos_margin,
        risk_leverage=oos_leverage,
        exit_threshold=float(args.exit_threshold),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        notional_scaled_sltp=False,
        exit_sizing_input_mode="actual",
        device=device,
    )
    oos_ledger = oos_ledger.copy()
    oos_ledger["entry_timestamp"] = pd.to_datetime(oos_ledger["entry_timestamp"])
    oos_ledger = oos_ledger.merge(oos_ou, on="entry_timestamp", how="left", validate="one_to_one")
    oos_gated = oos_ledger.loc[oos_ledger["ou_halflife"] > float(best["duration_threshold"])].reset_index(drop=True)
    oos_frozen = oos_gated.loc[oos_gated["entry_timestamp"] < pd.Timestamp("2026-04-01")].reset_index(drop=True)

    report = {
        "method": "omega4_6_1_asset_param_search_val_only",
        "asset": asset,
        "selection_objective": "max validation pnl with validation MDD >= -max_validation_mdd_abs and duration-gated trades >= min_trade_ratio per scale/component candidate",
        "search_space": {
            "risk_sidecar_dirs": [str(d) for d in _risk_sidecar_dirs(asset)],
            "scale_grid": scale_values,
            "duration_quantiles": "0.05..0.80 by 0.05 plus no-gate",
            "exit_threshold": float(args.exit_threshold),
            "cost_mult": float(args.cost_mult),
            "leverage_cap": LEVERAGE_CAP,
            "notional_cap": NOTIONAL_CAP,
            "max_validation_mdd_abs": float(args.max_validation_mdd_abs),
            "min_trade_ratio": float(args.min_trade_ratio),
        },
        "selected": best,
        "oos_one_shot": _compound_metrics(oos_gated),
        "oos_frozen_q1_2026": _compound_metrics(oos_frozen),
        "candidate_count": len(candidates),
    }
    pd.DataFrame(candidates).to_csv(out_dir / "candidate_grid.csv", index=False)
    oos_gated.to_csv(out_dir / "selected_oos_gated_ledger.csv", index=False)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
