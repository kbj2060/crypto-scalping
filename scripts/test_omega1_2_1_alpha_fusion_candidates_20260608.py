#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_exposure_selector_20260606 as exposure  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
from alpha6_catboost_entry_quality_exit_policy_20260522 import EQEConfig, _predict_entry  # noqa: E402
from analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS  # noqa: E402
from train_alpha7_regime3_current_moe_feature_variants_20260601 import _load_frames_with_risk  # noqa: E402


MODEL_ID = "omega1_2_1_alpha_fusion_candidates_20260608"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
ALPHA6_BUNDLE = ROOT / "data/ensemble/supervised/alpha6_entry_quality_exit_5bucket_main_20260522/current_tail111_bundle.joblib"
ALPHA7_MOE_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _metric_row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_avg_notional": float(metrics["avg_notional"]),
        f"{prefix}_avg_leverage": float(metrics["avg_leverage"]),
        f"{prefix}_reasons": metrics["exit_reasons"],
    }


def _build_omega_split(frames: dict[str, pd.DataFrame], split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame, _src, dec, _prefix = exposure._build_split(frames, split)
    return frame, sleeve._apply_aggressive(dec)


def _alpha6_predictions(frame: pd.DataFrame, bundle: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = [c for c in bundle["model_features"] if c not in frame.columns]
    if missing:
        raise RuntimeError(f"Alpha6 feature contract mismatch: missing={missing[:20]}")
    forbidden = [
        c
        for c in bundle["model_features"]
        if c.startswith("clean_regime4_") or c.startswith("regime4_pred_") or c == "tp_sl_action_score"
    ]
    x = bundle["pipeline"].transform(frame[bundle["model_features"]])
    dec = _predict_entry(bundle["entry_models"], x, EQEConfig(**bundle["config"]))
    return dec.reset_index(drop=True), {
        "model_features": int(len(bundle["model_features"])),
        "forbidden_current_omega_features": forbidden,
        "promotion_status": "blocked_until_alpha6_is_retrained_on_current_omega_regime3_contract" if forbidden else "clean",
    }


def _scale_rows(dec: pd.DataFrame, idx: np.ndarray, scale: float, *, cap: float) -> None:
    if len(idx) == 0:
        return
    base = pd.to_numeric(dec.loc[idx, "notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
    new = np.minimum(base * float(scale), float(cap))
    ratio = new / np.maximum(base, 1.0e-12)
    dec.loc[idx, "notional_exposure"] = new
    dec.loc[idx, "position_fraction"] = new
    dec.loc[idx, "take_profit"] = pd.to_numeric(dec.loc[idx, "take_profit"], errors="raise").to_numpy(dtype=np.float64) * ratio
    dec.loc[idx, "stop_loss"] = pd.to_numeric(dec.loc[idx, "stop_loss"], errors="raise").to_numpy(dtype=np.float64) * ratio


def _apply_alpha6_overlay(
    omega_dec: pd.DataFrame,
    alpha6_dec: pd.DataFrame,
    *,
    mode: str,
    q_threshold: float,
    shrink: float,
    boost: float,
    cap: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = omega_dec.copy().reset_index(drop=True)
    active = omega._active(out)
    omega_side = pd.to_numeric(out["side"], errors="raise").to_numpy(dtype=np.int64)
    a6_action = pd.to_numeric(alpha6_dec["action"], errors="raise").to_numpy(dtype=np.int64)
    a6_side = np.where(a6_action == 1, 1, np.where(a6_action == 2, -1, 0)).astype(np.int64)
    a6_q = pd.to_numeric(alpha6_dec["quality_score"], errors="raise").to_numpy(dtype=np.float64)
    strong = a6_q >= float(q_threshold)
    same = active & strong & (a6_side == omega_side)
    opposite = active & strong & (a6_side == -omega_side)
    if mode == "opposite_veto":
        idx = np.flatnonzero(opposite)
        out.loc[idx, "action"] = 0
        out.loc[idx, "side"] = 0
        out.loc[idx, "notional_exposure"] = 0.0
        out.loc[idx, "position_fraction"] = 0.0
    elif mode == "opposite_shrink":
        _scale_rows(out, np.flatnonzero(opposite), shrink, cap=cap)
    elif mode == "same_boost":
        _scale_rows(out, np.flatnonzero(same), boost, cap=cap)
    elif mode == "same_boost_opposite_shrink":
        _scale_rows(out, np.flatnonzero(opposite), shrink, cap=cap)
        _scale_rows(out, np.flatnonzero(same), boost, cap=cap)
    else:
        raise RuntimeError(f"unknown Alpha6 overlay mode: {mode}")
    return out, {
        "same_strong": int(np.count_nonzero(same)),
        "opposite_strong": int(np.count_nonzero(opposite)),
        "active": int(np.count_nonzero(active)),
    }


def _load_alpha7_moe_decisions(split: str, target_frame: pd.DataFrame) -> pd.DataFrame:
    name = "validation_decisions.csv" if split == "validation" else "oos_2026_decisions.csv"
    dec = pd.read_csv(ALPHA7_MOE_DIR / name)
    if "timestamp" in dec.columns:
        src = dec.copy()
    else:
        train_all, eval_df, _overlay = _load_frames_with_risk()
        src_frame = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True) if split == "validation" else eval_df.reset_index(drop=True)
        if len(src_frame) != len(dec):
            raise RuntimeError(f"Alpha7 MoE source row alignment mismatch for {split}: decisions={len(dec)} source={len(src_frame)}")
        src = dec.copy()
        src.insert(0, "timestamp", pd.to_datetime(src_frame["timestamp"], errors="raise").to_numpy())
    target_ts = pd.to_datetime(target_frame["timestamp"], errors="raise")
    src["timestamp"] = pd.to_datetime(src["timestamp"], errors="raise")
    aligned = target_frame[["timestamp"]].copy()
    aligned["timestamp"] = target_ts
    out = aligned.merge(src, on="timestamp", how="left", validate="one_to_one")
    if out.isna().any().any():
        bad = out.loc[out.isna().any(axis=1), "timestamp"].head(10).astype(str).tolist()
        raise RuntimeError(f"Alpha7 MoE timestamp alignment produced NaN for {split}: {bad}")
    return out.drop(columns=["timestamp"]).reset_index(drop=True)


def _apply_cash_sleeve(
    omega_dec: pd.DataFrame,
    sleeve_dec: pd.DataFrame,
    *,
    sleeve_scale: float,
    sleeve_cap: float,
    min_conf: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = omega_dec.copy().reset_index(drop=True)
    omega_active = omega._active(out)
    sleeve_active = omega._active(sleeve_dec)
    sleeve_conf = pd.to_numeric(sleeve_dec.get("confidence", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    take = (~omega_active) & sleeve_active & (sleeve_conf >= float(min_conf))
    idx = np.flatnonzero(take)
    cols = ["action", "side", "notional_exposure", "leverage", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars", "quality_score", "confidence"]
    for col in cols:
        out.loc[idx, col] = sleeve_dec.loc[idx, col].to_numpy()
    _scale_rows(out, idx, sleeve_scale, cap=sleeve_cap)
    return out, {
        "sleeve_entries": int(len(idx)),
        "omega_active": int(np.count_nonzero(omega_active)),
        "sleeve_active": int(np.count_nonzero(sleeve_active)),
    }


def _metrics_with_trailing(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    activation: float,
    gap_frac: float,
) -> dict[str, Any]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_idx = 0
    notional = 0.0
    leverage = 1.0
    take_profit = 0.0
    stop_loss = 0.0
    max_hold = 0
    cooldown = 0
    next_cooldown = 0
    mfe = 0.0
    trades = wins = long_entries = short_entries = 0
    notional_sum = leverage_sum = 0.0
    reasons: dict[str, int] = {}

    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            eq = cash * (1.0 + unreal)
        else:
            unreal = 0.0
            eq = cash
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)

        if pos != 0:
            hold = int(i) - int(entry_idx)
            trail_floor = max(0.0, mfe * (1.0 - float(gap_frac)))
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "stop_loss"
            elif mfe >= float(activation) and unreal <= trail_floor:
                reason = "trail_activation"
            elif max_hold > 0 and hold >= max_hold:
                reason = "max_hold"
            if reason:
                filled, exit_px, exit_fee, _route = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                reasons[reason] = reasons.get(reason, 0) + 1
                pos = 0
                cooldown = int(next_cooldown)
                next_cooldown = 0
                mfe = 0.0
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if not bool(active[i]):
            continue
        row = dec.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        pos = side
        entry_price = px
        entry_equity = cash
        entry_idx = int(i)
        notional = float(row.get("notional_exposure", 0.0) or 0.0)
        leverage = float(row.get("leverage", 1.0) or 1.0)
        take_profit = float(row.get("take_profit", 0.0) or 0.0)
        stop_loss = float(row.get("stop_loss", 0.0) or 0.0)
        max_hold = int(row.get("max_hold_bars", 0) or 0)
        next_cooldown = int(row.get("cooldown_bars", 0) or 0)
        mfe = 0.0
        cash -= cash * entry_fee * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage

    if pos != 0:
        fill_i = len(frame) - 1
        exit_px = omega._fill_price(arrays, fill_i, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        trades += 1
        wins += int(cash > entry_equity)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1

    n_entries = max(long_entries + short_entries, 1)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
        "trades_per_day": float(trades / max((pd.to_datetime(frame["timestamp"].iloc[-1]) - pd.to_datetime(frame["timestamp"].iloc[0])).total_seconds() / 86400.0, 1e-9)),
        "avg_notional": float(notional_sum / n_entries),
        "avg_leverage": float(leverage_sum / n_entries),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": reasons,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test Omega1.2.1 alpha-fusion candidates using Omega-contract inputs only by default.")
    p.add_argument(
        "--include-blocked-alpha6-reference",
        action="store_true",
        help="Run the historical Alpha6 bundle overlay as a blocked reference. This uses legacy Alpha6 features and is never promotable.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_dec = _build_omega_split(frames, "validation")
    oos_frame, oos_dec = _build_omega_split(frames, "oos")

    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    baseline_val = omega._metrics(val_frame, val_dec, fee=fee, slip=slip, cost_mult=3.0)
    baseline_oos = omega._metrics(oos_frame, oos_dec, fee=fee, slip=slip, cost_mult=3.0)
    rows.append({"candidate": "baseline_aggressive_omega1_2_1", "family": "baseline", **_metric_row("val", baseline_val), **_metric_row("oos", baseline_oos)})

    diagnostics["alpha6"] = {
        "skipped_by_default": True,
        "reason": "Historical Alpha6 bundle consumes legacy Alpha feature columns. New candidates must use Omega feature contract inputs only.",
        "reference_flag": "--include-blocked-alpha6-reference",
    }
    if args.include_blocked_alpha6_reference:
        alpha6_bundle = joblib.load(ALPHA6_BUNDLE)
        val_a6, diag_a6_val = _alpha6_predictions(val_frame, alpha6_bundle)
        oos_a6, diag_a6_oos = _alpha6_predictions(oos_frame, alpha6_bundle)
        diagnostics["alpha6"] = {"validation": diag_a6_val, "oos": diag_a6_oos, "reference_only": True}
        for mode in ("opposite_veto", "opposite_shrink", "same_boost", "same_boost_opposite_shrink"):
            for qthr in (0.0030, 0.0040, 0.0050, 0.0060, 0.0080):
                for shrink in ((0.70, 0.85) if "shrink" in mode or "veto" not in mode else (1.0,)):
                    for boost in ((1.05, 1.10, 1.20) if "boost" in mode else (1.0,)):
                        vdec, vcnt = _apply_alpha6_overlay(val_dec, val_a6, mode=mode, q_threshold=qthr, shrink=shrink, boost=boost, cap=0.90)
                        odec, ocnt = _apply_alpha6_overlay(oos_dec, oos_a6, mode=mode, q_threshold=qthr, shrink=shrink, boost=boost, cap=0.90)
                        rows.append(
                            {
                                "candidate": f"blocked_reference_alpha6_{mode}_q{qthr:g}_shr{shrink:g}_bst{boost:g}",
                                "family": "blocked_reference_alpha6_feature_overlay",
                                "promotion_blocked": True,
                                "contract_note": "reference only; uses historical Alpha6 feature bundle, not Omega feature contract",
                                "val_overlay_counts": vcnt,
                                "oos_overlay_counts": ocnt,
                                **_metric_row("val", omega._metrics(val_frame, vdec, fee=fee, slip=slip, cost_mult=3.0)),
                                **_metric_row("oos", omega._metrics(oos_frame, odec, fee=fee, slip=slip, cost_mult=3.0)),
                            }
                        )

    val_moe = _load_alpha7_moe_decisions("validation", val_frame)
    oos_moe = _load_alpha7_moe_decisions("oos", oos_frame)
    for min_conf in (0.00, 0.40, 0.55, 0.70, 0.85):
        for scale in (0.15, 0.25, 0.35, 0.50):
            for cap in (0.25, 0.35, 0.50):
                vdec, vcnt = _apply_cash_sleeve(val_dec, val_moe, sleeve_scale=scale, sleeve_cap=cap, min_conf=min_conf)
                odec, ocnt = _apply_cash_sleeve(oos_dec, oos_moe, sleeve_scale=scale, sleeve_cap=cap, min_conf=min_conf)
                rows.append(
                    {
                        "candidate": f"alpha7_regime3_moe_cash_sleeve_conf{min_conf:g}_scale{scale:g}_cap{cap:g}",
                        "family": "alpha7_regime3_moe_cash_sleeve",
                        "promotion_blocked": True,
                        "contract_note": "changes Omega CASH terminal semantics; requires separate candidate contract",
                        "val_overlay_counts": vcnt,
                        "oos_overlay_counts": ocnt,
                        **_metric_row("val", omega._metrics(val_frame, vdec, fee=fee, slip=slip, cost_mult=3.0)),
                        **_metric_row("oos", omega._metrics(oos_frame, odec, fee=fee, slip=slip, cost_mult=3.0)),
                    }
                )

    for activation in (0.006, 0.009, 0.012, 0.018, 0.026):
        for gap_frac in (0.25, 0.40, 0.55, 0.70):
            rows.append(
                {
                    "candidate": f"alpha3_style_trailing_act{activation:g}_gap{gap_frac:g}",
                    "family": "alpha3_style_execution_trailing",
                    "promotion_blocked": False,
                    **_metric_row("val", _metrics_with_trailing(val_frame, val_dec, fee=fee, slip=slip, cost_mult=3.0, activation=activation, gap_frac=gap_frac)),
                    **_metric_row("oos", _metrics_with_trailing(oos_frame, oos_dec, fee=fee, slip=slip, cost_mult=3.0, activation=activation, gap_frac=gap_frac)),
                }
            )

    ranking = pd.DataFrame(rows)
    ranking["val_delta_pnl"] = ranking["val_pnl"] - float(baseline_val["pnl"])
    ranking["oos_delta_pnl"] = ranking["oos_pnl"] - float(baseline_oos["pnl"])
    ranking["val_delta_mdd"] = ranking["val_mdd"] - float(baseline_val["mdd"])
    ranking["oos_delta_mdd"] = ranking["oos_mdd"] - float(baseline_oos["mdd"])
    ranking["score"] = ranking["oos_pnl"] + 0.50 * ranking["val_pnl"] + 0.20 * ranking["oos_mdd"] + 0.20 * ranking["val_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "val_pnl", "score"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "alpha_fusion_candidate_ranking.csv", index=False)
    family_best = ranking.sort_values(["family", "oos_pnl", "val_pnl"], ascending=[True, False, False]).groupby("family", as_index=False).head(5)
    family_best.to_csv(OUT_DIR / "alpha_fusion_family_best.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "baseline": {"validation": baseline_val, "oos": baseline_oos},
        "diagnostics": diagnostics,
        "top20": ranking.head(20).to_dict(orient="records"),
        "family_best": family_best.to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "alpha_fusion_candidate_ranking.csv"),
            "family_best": str(OUT_DIR / "alpha_fusion_family_best.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top5": report["top20"][:5]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
