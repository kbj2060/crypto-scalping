#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_2_tp_runner_cash_sleeve_20260615 as base  # noqa: E402
import train_eval_omega1_2_3_cash_sleeve_upgrade_20260615 as upgrade  # noqa: E402
from trading_bot_modules.omega1_2_1_live import Omega121Decision  # noqa: E402
from trading_bot_modules.omega1_2_3_cash_sleeve import (  # noqa: E402
    ACTION_CASH,
    ACTION_LONG,
    ACTION_SHORT,
    Omega123CashSleeveAdapter,
)


DEFAULT_BUNDLE = (
    ROOT
    / "data/ensemble/supervised/omega1_2_3_ev_hgb_cash_sleeve_20260615/ev_hgb_cash_sleeve.joblib"
)
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_3_ev_hgb_cash_sleeve_parity_20260616"


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


def _action_from_ev(long_ev: float, short_ev: float, ev_min: float) -> tuple[int, int]:
    if max(float(long_ev), float(short_ev)) <= float(ev_min):
        return ACTION_CASH, 0
    if float(long_ev) >= float(short_ev):
        return ACTION_LONG, 1
    return ACTION_SHORT, -1


def _router_from_state(row: pd.Series) -> str:
    vals = {
        "bull": float(row.get("tabm_router_bull", 0.0) or 0.0),
        "bear": float(row.get("tabm_router_bear", 0.0) or 0.0),
        "chop_expert": float(row.get("tabm_router_chop_expert", 0.0) or 0.0),
    }
    best = max(vals.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0.0 else ""


def _primary_decision(dec_row: pd.Series, state_row: pd.Series) -> Omega121Decision:
    trace = {
        "router_expert": _router_from_state(state_row),
        "router_confidence": float(state_row.get("tabm_router_confidence", 0.0) or 0.0),
        "router_margin": float(state_row.get("tabm_router_margin", 0.0) or 0.0),
        "direction_proba": {
            "cash": float(state_row.get("tabm_dir_p_cash", 0.0) or 0.0),
            "long": float(state_row.get("tabm_dir_p_long", 0.0) or 0.0),
            "short": float(state_row.get("tabm_dir_p_short", 0.0) or 0.0),
        },
        "quality_proba": {
            "cash": float(state_row.get("tabm_quality_p_cash", 0.0) or 0.0),
            "long": float(state_row.get("tabm_quality_p_long", 0.0) or 0.0),
            "short": float(state_row.get("tabm_quality_p_short", 0.0) or 0.0),
        },
        "quality_for_action": float(state_row.get("tabm_quality_for_action", dec_row.get("quality_score", 0.0)) or 0.0),
    }
    return Omega121Decision(
        action=int(dec_row.get("action", 0) or 0),
        side=int(dec_row.get("side", 0) or 0),
        notional_exposure=float(dec_row.get("notional_exposure", 0.0) or 0.0),
        leverage=float(dec_row.get("leverage", 1.0) or 1.0),
        position_fraction=float(dec_row.get("position_fraction", 0.0) or 0.0),
        take_profit=float(dec_row.get("take_profit", 0.0) or 0.0),
        stop_loss=float(dec_row.get("stop_loss", 0.0) or 0.0),
        max_hold_bars=int(dec_row.get("max_hold_bars", 0) or 0),
        cooldown_bars=int(dec_row.get("cooldown_bars", 0) or 0),
        quality_score=float(dec_row.get("quality_score", 0.0) or 0.0),
        confidence=float(dec_row.get("confidence", 0.0) or 0.0),
        router_expert=str(dec_row.get("router_expert", "") or ""),
        trace=trace,
    )


def _load_payload(split: str) -> dict[str, Any]:
    payload = base.legacy_runner._build()[split]
    return {
        "frame": payload["frame"].reset_index(drop=True),
        "dec": payload["dec"].reset_index(drop=True),
        "state": payload["state"].reset_index(drop=True),
        "fee": payload["fee"],
        "slip": payload["slip"],
    }


def _reconstruct_live_feature_matrix(payload: dict[str, Any]) -> pd.DataFrame:
    frame = payload["frame"].reset_index(drop=True)
    dec = payload["dec"].reset_index(drop=True)
    active = base._active(dec)
    out = payload["state"].copy().reset_index(drop=True)
    out["primary_is_cash"] = (~active).astype(float)
    out["primary_active_roll_12"] = pd.Series(active.astype(float)).rolling(12, min_periods=1).mean().to_numpy(dtype=np.float64)
    out["primary_active_roll_48"] = pd.Series(active.astype(float)).rolling(48, min_periods=1).mean().to_numpy(dtype=np.float64)

    primary_cash_streak = np.zeros(len(out), dtype=np.float64)
    time_since_primary_exit = np.zeros(len(out), dtype=np.float64)
    last_primary_active_len = np.zeros(len(out), dtype=np.float64)
    last_primary_side = np.zeros(len(out), dtype=np.float64)
    cur_cash = 0
    cur_active = 0
    prev_active_len = 0
    prev_side = 0
    sides = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    for i, is_active in enumerate(active):
        if bool(is_active):
            cur_active += 1
            cur_cash = 0
            prev_side = int(sides[i])
        else:
            if i > 0 and bool(active[i - 1]):
                prev_active_len = cur_active
                cur_active = 0
            cur_cash += 1
        primary_cash_streak[i] = cur_cash
        time_since_primary_exit[i] = cur_cash
        last_primary_active_len[i] = prev_active_len
        last_primary_side[i] = prev_side

    close = pd.to_numeric(frame["close"], errors="raise").reset_index(drop=True)
    high = pd.to_numeric(frame["high"], errors="raise").reset_index(drop=True)
    low = pd.to_numeric(frame["low"], errors="raise").reset_index(drop=True)
    ret1 = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rng = ((high - low) / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["primary_cash_streak"] = np.tanh(primary_cash_streak / 144.0)
    out["cash_ret_sum_12"] = ret1.rolling(12, min_periods=1).sum().to_numpy(dtype=np.float64)
    out["cash_ret_sum_48"] = ret1.rolling(48, min_periods=1).sum().to_numpy(dtype=np.float64)
    out["cash_ret_vol_12"] = ret1.rolling(12, min_periods=2).std().fillna(0.0).to_numpy(dtype=np.float64)
    out["cash_ret_vol_48"] = ret1.rolling(48, min_periods=2).std().fillna(0.0).to_numpy(dtype=np.float64)
    out["cash_range_ratio_12_48"] = (
        rng.rolling(12, min_periods=1).mean() / rng.rolling(48, min_periods=1).mean().replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0).to_numpy(dtype=np.float64)

    probs = out[["tabm_dir_p_cash", "tabm_dir_p_long", "tabm_dir_p_short"]].clip(1e-9, 1.0)
    out["tabm_dir_entropy"] = (-(probs * np.log(probs)).sum(axis=1) / np.log(3.0)).to_numpy(dtype=np.float64)
    out["tabm_long_short_gap"] = (out["tabm_dir_p_long"] - out["tabm_dir_p_short"]).to_numpy(dtype=np.float64)
    out["tabm_abs_side_gap"] = np.abs(out["tabm_long_short_gap"]).to_numpy(dtype=np.float64)
    out["tabm_quality_side_gap"] = (out["tabm_quality_p_long"] - out["tabm_quality_p_short"]).to_numpy(dtype=np.float64)
    out["tabm_quality_abs_gap"] = np.abs(out["tabm_quality_side_gap"]).to_numpy(dtype=np.float64)
    out["time_since_primary_exit"] = np.tanh(time_since_primary_exit / 144.0)
    out["last_primary_active_len"] = np.tanh(last_primary_active_len / 288.0)
    out["last_primary_side"] = last_primary_side
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _predict_actions(bundle: dict[str, Any], x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    calibration = dict(bundle.get("calibration") or {})
    long_ev = bundle["long_model"].predict(x.to_numpy(dtype=np.float64)).astype(np.float64) - float(
        calibration.get("long_abs_residual_offset", 0.0) or 0.0
    )
    short_ev = bundle["short_model"].predict(x.to_numpy(dtype=np.float64)).astype(np.float64) - float(
        calibration.get("short_abs_residual_offset", 0.0) or 0.0
    )
    ev_min = float(bundle["ev_min"])
    best_long = long_ev >= short_ev
    best_ev = np.maximum(long_ev, short_ev)
    action = np.where(best_ev > ev_min, np.where(best_long, ACTION_LONG, ACTION_SHORT), ACTION_CASH).astype(np.int64)
    side = np.where(action == ACTION_LONG, 1, np.where(action == ACTION_SHORT, -1, 0)).astype(np.int64)
    return action, side, long_ev, short_ev


def run_fast(split: str, bundle_path: Path, max_rows: int) -> dict[str, Any]:
    payload = _load_payload(split)
    if max_rows > 0:
        for key in ("frame", "dec", "state"):
            payload[key] = payload[key].iloc[: int(max_rows)].reset_index(drop=True)
    reference = upgrade._enhanced_features(payload)
    reconstructed = _reconstruct_live_feature_matrix(payload)
    bundle = joblib.load(bundle_path)
    feature_cols = list(bundle["feature_cols"])
    active = base._active(payload["dec"])
    cash_mask = ~active
    ref_x = reference.loc[cash_mask, feature_cols].reset_index(drop=True)
    live_x = reconstructed.loc[cash_mask, feature_cols].reset_index(drop=True)
    feature_abs = np.abs(ref_x.to_numpy(dtype=np.float64) - live_x.to_numpy(dtype=np.float64))
    feature_row_diff = feature_abs.max(axis=1) if len(feature_abs) else np.asarray([], dtype=np.float64)

    ref_action, ref_side, ref_long, ref_short = _predict_actions(bundle, ref_x)
    live_action, live_side, live_long, live_short = _predict_actions(bundle, live_x)
    long_diff = np.abs(ref_long - live_long)
    short_diff = np.abs(ref_short - live_short)
    mismatch_mask = (
        (ref_action != live_action)
        | (ref_side != live_side)
        | (feature_row_diff > 1e-12)
        | (long_diff > 1e-12)
        | (short_diff > 1e-12)
    )
    cash_index = np.flatnonzero(cash_mask)
    rows = pd.DataFrame(
        {
            "i": cash_index.astype(np.int64),
            "timestamp": payload["frame"].loc[cash_mask, "timestamp"].to_numpy() if "timestamp" in payload["frame"].columns else "",
            "reference_action": ref_action,
            "live_action": live_action,
            "reference_side": ref_side,
            "live_side": live_side,
            "reference_long_ev": ref_long,
            "live_long_ev": live_long,
            "reference_short_ev": ref_short,
            "live_short_ev": live_short,
            "feature_max_abs_diff": feature_row_diff,
            "long_ev_abs_diff": long_diff,
            "short_ev_abs_diff": short_diff,
        }
    )
    mismatches = rows.loc[mismatch_mask].copy()
    out_dir = OUT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_dir / "cash_sleeve_parity_rows.csv", index=False)
    mismatches.to_csv(out_dir / "mismatches.csv", index=False)
    report = {
        "mode": "fast_vectorized_full",
        "split": split,
        "bundle_path": str(bundle_path),
        "rows": int(len(payload["frame"])),
        "cash_rows": int(cash_mask.sum()),
        "compared_rows": int(len(rows)),
        "action_mismatches": int(((ref_action != live_action) | (ref_side != live_side)).sum()),
        "total_mismatches": int(len(mismatches)),
        "feature_count": int(len(feature_cols)),
        "feature_max_abs_diff": float(feature_row_diff.max() if len(feature_row_diff) else 0.0),
        "long_ev_max_abs_diff": float(long_diff.max() if len(long_diff) else 0.0),
        "short_ev_max_abs_diff": float(short_diff.max() if len(short_diff) else 0.0),
        "status": "pass" if len(mismatches) == 0 else "fail",
        "outputs": {
            "rows": str(out_dir / "cash_sleeve_parity_rows.csv"),
            "mismatches": str(out_dir / "mismatches.csv"),
            "report": str(out_dir / "report.json"),
        },
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True, default=_json_default)
    return report


def run_direct(split: str, bundle_path: Path, max_rows: int) -> dict[str, Any]:
    payload = _load_payload(split)
    feature_df = upgrade._enhanced_features(payload)
    bundle = joblib.load(bundle_path)
    feature_cols = list(bundle["feature_cols"])
    ev_min = float(bundle["ev_min"])
    adapter = Omega123CashSleeveAdapter(bundle_path)
    active = base._active(payload["dec"])
    limit = len(payload["frame"]) if max_rows <= 0 else min(int(max_rows), len(payload["frame"]))

    rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    long_diffs: list[float] = []
    short_diffs: list[float] = []
    feature_diffs: list[float] = []
    cash_rows = 0
    compared_rows = 0
    for i in range(limit):
        primary = _primary_decision(payload["dec"].iloc[i], payload["state"].iloc[i])
        adapter.observe_primary(bool(active[i]), int(primary.side))
        if bool(active[i]):
            continue
        cash_rows += 1
        if i >= len(feature_df) - 1:
            continue
        reference_x = feature_df.loc[[i], feature_cols]
        live_features = adapter._history_features(payload["frame"].iloc[: i + 1].reset_index(drop=True), primary)
        live_x = pd.DataFrame([{c: float(live_features[c]) for c in feature_cols}], columns=feature_cols)
        feature_diff = float(np.max(np.abs(reference_x.to_numpy(dtype=np.float64) - live_x.to_numpy(dtype=np.float64))))
        calibration = dict(bundle.get("calibration") or {})
        reference_long = float(bundle["long_model"].predict(reference_x.to_numpy(dtype=np.float64))[0]) - float(
            calibration.get("long_abs_residual_offset", 0.0) or 0.0
        )
        reference_short = float(bundle["short_model"].predict(reference_x.to_numpy(dtype=np.float64))[0]) - float(
            calibration.get("short_abs_residual_offset", 0.0) or 0.0
        )
        ref_action, ref_side = _action_from_ev(reference_long, reference_short, ev_min)
        live_dec = adapter.decide_latest(payload["frame"].iloc[: i + 1], primary)
        long_diff = abs(float(live_dec.long_ev) - reference_long)
        short_diff = abs(float(live_dec.short_ev) - reference_short)
        compared_rows += 1
        long_diffs.append(long_diff)
        short_diffs.append(short_diff)
        feature_diffs.append(feature_diff)
        row = {
            "i": int(i),
            "timestamp": payload["frame"].iloc[i].get("timestamp"),
            "reference_action": int(ref_action),
            "live_action": int(live_dec.action),
            "reference_side": int(ref_side),
            "live_side": int(live_dec.side),
            "reference_long_ev": reference_long,
            "live_long_ev": float(live_dec.long_ev),
            "reference_short_ev": reference_short,
            "live_short_ev": float(live_dec.short_ev),
            "feature_max_abs_diff": feature_diff,
            "long_ev_abs_diff": long_diff,
            "short_ev_abs_diff": short_diff,
        }
        rows.append(row)
        if (
            int(ref_action) != int(live_dec.action)
            or int(ref_side) != int(live_dec.side)
            or feature_diff > 1e-12
            or long_diff > 1e-12
            or short_diff > 1e-12
        ):
            mismatches.append(row)

    out_dir = OUT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "cash_sleeve_parity_rows.csv", index=False)
    pd.DataFrame(mismatches).to_csv(out_dir / "mismatches.csv", index=False)
    report = {
        "split": split,
        "bundle_path": str(bundle_path),
        "rows": int(limit),
        "cash_rows": int(cash_rows),
        "compared_rows": int(compared_rows),
        "action_mismatches": int(
            sum(1 for r in mismatches if r["reference_action"] != r["live_action"] or r["reference_side"] != r["live_side"])
        ),
        "total_mismatches": int(len(mismatches)),
        "feature_count": int(len(feature_cols)),
        "feature_max_abs_diff": float(max(feature_diffs) if feature_diffs else 0.0),
        "long_ev_max_abs_diff": float(max(long_diffs) if long_diffs else 0.0),
        "short_ev_max_abs_diff": float(max(short_diffs) if short_diffs else 0.0),
        "status": "pass" if not mismatches else "fail",
        "outputs": {
            "rows": str(out_dir / "cash_sleeve_parity_rows.csv"),
            "mismatches": str(out_dir / "mismatches.csv"),
            "report": str(out_dir / "report.json"),
        },
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True, default=_json_default)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["validation", "oos"], required=True)
    ap.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--mode", choices=["fast", "direct"], default="fast")
    args = ap.parse_args()
    report = run_direct(args.split, args.bundle_path, args.max_rows) if args.mode == "direct" else run_fast(args.split, args.bundle_path, args.max_rows)
    print(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
