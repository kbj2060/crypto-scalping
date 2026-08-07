#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
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

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega4_1_baseline_governor_sizing_20260621"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
DEFAULT_OMEGA4_DIR = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_3head_parent72_loose_entry_quality_20260620_smoke_loose_entry_loose_quality_terminal_giveback_exit_e2_train15k_exit15k_q070"
)
BASELINE = {
    "omega4_1_no_exit_oos": {"pnl": 23.9062, "mdd": -7.0901, "trades": 21, "wr": 0.6190},
    "omega4_1_exit_thr070_oos": {"pnl": 7.5133, "mdd": -5.6140, "trades": 100, "wr": 0.6300},
}


@dataclass(frozen=True)
class GovernorConfig:
    name: str
    sizing: str
    min_hold_bars: int
    adverse_unreal: float
    min_mfe_for_giveback: float
    giveback_min: float
    exit_threshold: float
    opposite_margin: float
    fixed_leverage: float = 3.0
    min_margin: float = 0.05
    max_margin: float = 0.30


CONFIGS = [
    GovernorConfig("baseline_no_exit_fixed", "fixed_baseline", 0, -99.0, 99.0, 99.0, 99.0, 99.0, fixed_leverage=2.0),
    GovernorConfig("sizing_quality_vol", "quality_vol", 0, -99.0, 99.0, 99.0, 99.0, 99.0),
    GovernorConfig("governor_conservative_fixed", "fixed_baseline", 24, -0.012, 0.006, 0.65, 0.80, 0.10, fixed_leverage=2.0),
    GovernorConfig("governor_conservative_sizing", "quality_vol", 24, -0.012, 0.006, 0.65, 0.80, 0.10),
    GovernorConfig("governor_fast_sizing", "quality_vol", 12, -0.010, 0.005, 0.55, 0.75, 0.08),
    GovernorConfig("governor_defensive_sizing", "quality_vol", 36, -0.015, 0.008, 0.70, 0.85, 0.12),
]


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


def _load_bundle(model_dir: Path) -> dict[str, Any]:
    path = Path(model_dir) / "true_3head_tabm_bundle.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("models", "base_cols"):
        if key not in bundle:
            raise RuntimeError(f"{path} missing {key}")
    return bundle


@torch.no_grad()
def _predict_all(frame: pd.DataFrame, bundle: dict[str, Any], *, threshold: float, oof: bool, device: torch.device) -> dict[str, Any]:
    x = parent._base_input(frame, list(bundle["base_cols"]))
    preds = {expert: parent._predict_payload(bundle["models"][expert], x, device=device) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(frame)
    direction = parent._routed(preds, route, "direction", 3)
    quality = parent._routed(preds, route, "quality", 3)
    prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
    src = parent._prediction_output(frame, direction, quality, threshold=float(threshold), prefix=prefix)
    dec = omega._to_fixed_decisions(src, oof=oof)
    return {"x": x, "direction": direction, "quality": quality, "dec": dec, "route": route}


def _prepare(model_dir: Path, *, threshold: float, device: torch.device) -> dict[str, Any]:
    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
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
    bundle = _load_bundle(model_dir)
    return {
        **frames,
        "bundle": bundle,
        "validation": _predict_all(frames["val_raw"], bundle, threshold=threshold, oof=True, device=device),
        "oos": _predict_all(frames["oos_raw"], bundle, threshold=threshold, oof=False, device=device),
    }


def _scale_decisions(dec: pd.DataFrame, frame: pd.DataFrame, quality: np.ndarray, cfg: GovernorConfig) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active_idx = np.flatnonzero(omega._active(out))
    if len(active_idx) == 0:
        return out
    base_notional = pd.to_numeric(out.loc[active_idx, "notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
    base_tp = pd.to_numeric(out.loc[active_idx, "take_profit"], errors="raise").to_numpy(dtype=np.float64)
    base_sl = pd.to_numeric(out.loc[active_idx, "stop_loss"], errors="raise").to_numpy(dtype=np.float64)
    if cfg.sizing == "fixed_baseline":
        new_notional = base_notional
        leverage = pd.to_numeric(out.loc[active_idx, "leverage"], errors="raise").to_numpy(dtype=np.float64)
    elif cfg.sizing == "quality_vol":
        action = pd.to_numeric(out.loc[active_idx, "action"], errors="raise").to_numpy(dtype=np.int64)
        q_for_action = quality[active_idx, action]
        close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
        ret = pd.Series(close).pct_change().rolling(24, min_periods=4).std().bfill().fillna(0.0).to_numpy(dtype=np.float64)
        vol = np.clip(ret[active_idx], 0.001, 0.030)
        vol_scale = np.clip(0.010 / vol, 0.55, 1.35)
        margin = np.clip(0.06 + 0.28 * np.clip((q_for_action - 0.70) / 0.30, 0.0, 1.0), cfg.min_margin, cfg.max_margin)
        margin = np.clip(margin * vol_scale, cfg.min_margin, cfg.max_margin)
        leverage = np.full(len(active_idx), float(cfg.fixed_leverage), dtype=np.float64)
        new_notional = margin * leverage
    else:
        raise RuntimeError(f"unknown sizing mode: {cfg.sizing}")
    tp_price_move = base_tp / np.maximum(base_notional, 1.0e-12)
    sl_price_move = base_sl / np.maximum(base_notional, 1.0e-12)
    out.loc[active_idx, "leverage"] = leverage
    out.loc[active_idx, "notional_exposure"] = new_notional
    out.loc[active_idx, "position_fraction"] = new_notional
    out.loc[active_idx, "take_profit"] = tp_price_move * new_notional
    out.loc[active_idx, "stop_loss"] = sl_price_move * new_notional
    return out


@torch.no_grad()
def _exit_prob(model_payload: dict[str, Any], x: pd.DataFrame, *, device: torch.device) -> float:
    pred = parent._predict_payload(model_payload, x, device=device)
    return float(pred["exit"][0, 1])


def _metrics_governor(
    frame: pd.DataFrame,
    pred: dict[str, Any],
    bundle: dict[str, Any],
    cfg: GovernorConfig,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
) -> dict[str, Any]:
    dec = _scale_decisions(pred["dec"], frame, pred["quality"], cfg)
    if cfg.name.startswith("baseline_no_exit") or cfg.name.startswith("sizing_"):
        return omega._metrics(frame, dec, fee=fee, slip=slip, cost_mult=cost_mult)
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
    entry_i = 0
    notional = 0.0
    leverage = 1.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    reasons: dict[str, int] = {}
    route = pred["route"]
    direction = pred["direction"]
    base_x = pred["x"]
    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            eq = cash * (1.0 + unreal)
        else:
            unreal = 0.0
            eq = cash
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            hold = max(int(i) - int(entry_i), 0)
            giveback = (mfe - unreal) / max(abs(mfe), 1.0e-8) if mfe > 0.0 else 0.0
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "stop_loss"
            elif hold >= int(cfg.min_hold_bars):
                opp_prob = float(direction[i, 2 if pos > 0 else 1])
                cur_prob = float(direction[i, 1 if pos > 0 else 2])
                opposite_confirm = (opp_prob - cur_prob) >= float(cfg.opposite_margin)
                if unreal <= float(cfg.adverse_unreal) and opposite_confirm:
                    reason = "adverse_opposite_exit"
                elif mfe >= float(cfg.min_mfe_for_giveback) and giveback >= float(cfg.giveback_min):
                    reason = "mfe_giveback_exit"
                else:
                    xrow = base_x.iloc[[i]].copy().reset_index(drop=True)
                    vals = {
                        "pos_side": float(pos),
                        "pos_hold_bars": float(hold),
                        "pos_unrealized": float(unreal),
                        "pos_mfe": float(mfe),
                        "pos_mae": float(mae),
                        "pos_giveback": float(np.clip(giveback, 0.0, 10.0)),
                        "pos_dist_to_tp": float(take_profit - unreal),
                        "pos_dist_to_sl": float(unreal + abs(stop_loss)),
                        "pos_notional": float(notional),
                        "pos_leverage": float(leverage),
                        "pos_exposure": float(notional * leverage),
                        "pos_tp": float(take_profit),
                        "pos_sl": float(stop_loss),
                    }
                    for col, val in vals.items():
                        xrow[col] = val
                    expert = hard.EXPERT_NAMES[int(route[i])]
                    exit_prob = _exit_prob(bundle["models"][expert], xrow, device=device)
                    if exit_prob >= float(cfg.exit_threshold) and (opposite_confirm or unreal < 0.0 or giveback >= 0.35):
                        reason = "exit_head_confirmed"
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
                continue
        if pos != 0 or not bool(active[i]):
            continue
        row = dec.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        pos = side
        entry_price = float(px)
        entry_equity = cash
        entry_i = min(int(i) + 1, len(frame) - 1)
        notional = float(row.get("notional_exposure", 0.0) or 0.0)
        leverage = float(row.get("leverage", 1.0) or 1.0)
        take_profit = float(row.get("take_profit", 0.0) or 0.0)
        stop_loss = float(row.get("stop_loss", 0.0) or 0.0)
        cash -= cash * entry_fee * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
        mfe = 0.0
        mae = 0.0
    if pos != 0:
        exit_px = omega._fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
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


def _score(row: dict[str, Any]) -> float:
    if float(row["validation_mdd"]) < -15.0:
        return -1.0e9
    if int(row["validation_trades"]) < 10:
        return -1.0e9
    return float(row["validation_pnl"]) + 0.25 * float(row["validation_mdd"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omega4-model-dir", type=Path, default=DEFAULT_OMEGA4_DIR)
    ap.add_argument("--quality-threshold", type=float, default=0.70)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    ap.add_argument("--out-suffix", default="grid")
    args = ap.parse_args()
    device = _device(str(args.device))
    out_dir = OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fee, slip = omega._load_fee_slip()
    prepared = _prepare(Path(args.omega4_model_dir), threshold=float(args.quality_threshold), device=device)
    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for cfg in CONFIGS:
        val = _metrics_governor(prepared["val_raw"], prepared["validation"], prepared["bundle"], cfg, fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
        oos = _metrics_governor(prepared["oos_raw"], prepared["oos"], prepared["bundle"], cfg, fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
        row = {
            "variant": cfg.name,
            "validation_pnl": float(val["pnl"]),
            "validation_mdd": float(val["mdd"]),
            "validation_trades": int(val["trades"]),
            "validation_wr": float(val["wr"]),
            "oos_pnl": float(oos["pnl"]),
            "oos_mdd": float(oos["mdd"]),
            "oos_trades": int(oos["trades"]),
            "oos_wr": float(oos["wr"]),
            "selection_score": 0.0,
        }
        row["selection_score"] = _score(row)
        rows.append(row)
        results[cfg.name] = {"config": asdict(cfg), "validation": val, "oos": oos}
        print(json.dumps(row, ensure_ascii=False), flush=True)
    rows_sorted = sorted(rows, key=lambda r: float(r["selection_score"]), reverse=True)
    selected = rows_sorted[0]
    pd.DataFrame(rows_sorted).to_csv(out_dir / "governor_sizing_ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "baseline_model": "omega4_1_exit_thr_0p70",
        "baseline_reference": BASELINE,
        "selection_policy": "validation-only selection by validation_pnl + 0.25*validation_mdd; reject validation_mdd < -15 or validation_trades < 10. OOS is diagnostic.",
        "quality_contract": {"direction_label_dir": str(omega4.LABEL_DIR), "quality_mode": "same_as_direction", "quality_threshold": float(args.quality_threshold)},
        "futures_sizing_contract": "fixed_leverage=3 for quality_vol variants; notional=margin_fraction*leverage; TP/SL account thresholds are rescaled from baseline price-move targets.",
        "results": results,
        "ranking_by_validation": rows_sorted,
        "selected_by_validation": selected,
        "artifacts": {"out_dir": str(out_dir), "ranking": str(out_dir / "governor_sizing_ranking.csv"), "report": str(out_dir / "report.json")},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "selected": selected}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
