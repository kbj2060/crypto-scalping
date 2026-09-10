#!/usr/bin/env python3
"""RESEARCH ONLY -- exploratory sweep of ETH live Omega4.6.1 exit logic.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py,
or .env. Uses the FROZEN h48qual/zig075 parent bundles + risk sidecars (same artifacts the
live adapter loads) and the FROZEN, already-generated OOF/held-out prediction CSVs that were
used to originally certify Omega4.6.1 -- no retraining.

Two experiments, both causal bar-by-bar walks (fresh-forward in the CLAUDE.md sense: each bar
only uses that bar own row + already-closed history; TP/SL/exit-head/trailing-stop are
evaluated in strict forward order per component, matching the certified _replay_with_risk
loop in train_eval_omega4_2_risk_sidecar_20260622.py):

  A) EXIT_THRESHOLD sweep -- same entry/sizing/TP/SL, only the exit-head probability gate varies.
  B) Trailing-stop / profit-giveback forced exit layered ON TOP of the existing SL/TP/exit-head
     checks (does not replace any of them). Once MFE (price-move since entry) reaches
     activate_frac * take_profit, the rule arms; once armed, if current unrealized profit drops
     to <= retain_frac * peak_MFE, force an exit immediately (bypasses the exit-head threshold,
     but SL/TP are still checked first exactly as in the baseline).

Windows: VAL = 2025-10-01..2025-12-31 (see NOTE below), OOS = 2026-01-01..2026-03-31.
NOTE / ASSUMPTION: the canonical CLAUDE.md VAL window is 2025-09-01..12-31, but the frozen OOF
prediction CSVs for this model only exist for 2025-10-01 onward (2025-09 was inside the parent
model own TRAIN split, so using it as VAL here would be leaky). VAL is therefore
2025-10-01..2025-12-31, one month short of canonical on the start side. Flagged explicitly in
the report; not silently fixed.
"""

from __future__ import annotations

import json
import pickle
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
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

WIDE24_2025 = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2025_regime3_current_sensitive_hmm_wide24.csv"
BASE_2025 = ROOT / "data/splits/year_oos/training_features_2025.csv"
WIDE24_2026 = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv"
BASE_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
EXT_PRED_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706"

VAL_START, VAL_END = "2025-10-01", "2025-12-31"
OOS_START, OOS_END = "2026-01-01", "2026-03-31"

COMPONENTS = {
    "h48qual": {
        "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/true_3head_tabm_bundle.pt",
        "sidecar_pkl": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_h48qual_q050_precomputed_20260630/risk_sidecar.pkl",
        "q_tag": "q050",
        "quality_threshold": 0.50,
        "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0,
        "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12,
    },
    "zig075": {
        "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/true_3head_tabm_bundle.pt",
        "sidecar_pkl": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_zig075_q075_precomputed_20260630/risk_sidecar.pkl",
        "q_tag": "q075",
        "quality_threshold": 0.75,
        "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0,
        "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12,
    },
}
COST_MULT = 1.0
DEVICE = parent._device("cpu")
BASELINE_EXIT_THRESHOLD = 0.95


def load_frame(start: str, end: str, *, base_csv: Path, wide24_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(base_csv, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    overlay = pd.read_csv(wide24_csv, low_memory=False)
    overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
    cols = [c for c in overlay.columns if c != "timestamp"]
    frame = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)].reset_index(drop=True)
    return frame


@torch.no_grad()
def replay_exit_variant(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded_models: dict[str, tuple],
    *,
    risk_margin_fraction: np.ndarray,
    risk_leverage: np.ndarray,
    exit_threshold: float,
    fee: float,
    slip: float,
    cost_mult: float,
    notional_scaled_sltp: bool,
    device: torch.device,
    trailing_activate_frac: float | None = None,
    trailing_retain_frac: float | None = None,
    trailing_trail_frac: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Causal bar-by-bar replay. Structurally identical to
    train_eval_omega4_2_risk_sidecar_20260622._replay_with_risk (same TP/SL/exit-head order,
    same fill/cost model), with one addition: an optional trailing-stop / profit-giveback forced
    exit, checked after SL/TP and before (in place of) the exit-head threshold once armed.
    fresh_forward_bar_by_bar=true (single forward pass, i in increasing order, only row i and
    already-closed prior bars used at bar i); no saved ledger is used as input.
    """
    # `trailing_retain_frac` is the original PROPORTIONAL giveback rule (exit once profit falls to
    # retain_frac * peak MFE). `trailing_trail_frac` added 2026-08-07 is the FIXED-DISTANCE rule
    # carried over from the BTC gate-G1 result (exit once profit falls trail_frac * |stop_loss|
    # below peak MFE) -- the two are mutually exclusive, and leaving the new argument at None
    # reproduces this script's original behaviour exactly.
    trailing_enabled = trailing_activate_frac is not None and (
        trailing_retain_frac is not None or trailing_trail_frac is not None)
    if trailing_retain_frac is not None and trailing_trail_frac is not None:
        raise ValueError("pass either trailing_retain_frac (proportional) or trailing_trail_frac (fixed distance)")
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
    entry_signal_i = 0
    notional = 0.0
    leverage = 1.0
    margin_fraction = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    armed = False
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    margin_sum = 0.0
    reasons: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    route = hard._route_id(frame)
    from train_eval_omega1_2_tabm_diffusion_risk_20260603 import _try_execution as omega_try_execution, _fill_price as omega_fill_price
    import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit

    base_np, exit_runtime, pos_idx = rs._prepare_exit_runtime(base_x, loaded_models)

    for i in range(0, len(frame) - 2):
        if pos != 0:
            move = price_exit._price_move(arrays, int(i), side=pos, entry_price=float(entry_price), slip_eff=slip_eff)
            mfe = max(mfe, move)
            mae = min(mae, move)
        else:
            move = 0.0

        if pos != 0:
            reason = ""
            exit_prob = 0.0
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            elif trailing_enabled and (not armed) and mfe >= float(trailing_activate_frac) * take_profit and take_profit > 0.0:
                armed = True
            if not reason and trailing_enabled and armed and mfe > 0.0:
                if trailing_retain_frac is not None:
                    if move <= float(trailing_retain_frac) * mfe:
                        reason = "trailing_stop"
                elif move <= mfe - float(trailing_trail_frac) * abs(stop_loss):
                    reason = "trailing_stop"
            if not reason:
                hold = max(int(i) - int(entry_i), 0)
                giveback = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(route[i])]
                prob = rs._predict_exit_prob_one(
                    base_np, exit_runtime, pos_idx, row_i=int(i), expert=expert,
                    pos_values=[
                        float(pos), float(hold), float(move), float(mfe), float(mae),
                        float(np.clip(giveback, 0.0, 10.0)), float(take_profit - move), float(move + abs(stop_loss)),
                        float(notional), float(leverage), float(notional * leverage), float(take_profit), float(stop_loss),
                    ],
                    device=device,
                )
                exit_prob = float(prob)
                if prob >= float(exit_threshold):
                    reason = "exit_head"
            if reason:
                filled, exit_px, exit_fee, _route = omega_try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
                trades += 1
                win = int(cash > entry_equity)
                wins += win
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append({
                    "entry_signal_i": int(entry_signal_i), "entry_i": int(entry_i), "exit_i": int(i),
                    "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                    "exit_timestamp": str(frame["timestamp"].iloc[int(i)]), "side": int(pos), "reason": reason,
                    "win": int(win), "raw_exit_price_move": float(raw_exit), "mfe_price_move": float(mfe),
                    "mae_price_move": float(mae), "trade_return": float(trade_return),
                    "net_per_notional": float(trade_return / max(notional, 1.0e-12)), "notional": float(notional),
                    "margin_fraction": float(margin_fraction), "leverage": float(leverage),
                    "exit_prob": float(exit_prob), "take_profit": float(take_profit), "stop_loss": float(stop_loss),
                })
                pos = 0
                armed = False
                continue
        eq = cash if pos == 0 else cash * (1.0 + move * notional)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1.0e-12) - 1.0)
        if pos != 0 or not bool(active[i]):
            continue
        row = dec.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, fee_paid, _route = omega_try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        row_leverage = float(risk_leverage[int(i)])
        row_margin = float(risk_margin_fraction[int(i)])
        row_notional = row_margin * row_leverage
        if row_notional <= 0.0:
            continue
        pos = side
        entry_price = float(px)
        entry_equity = cash
        entry_i = min(int(i) + 1, len(frame) - 1)
        entry_signal_i = int(i)
        leverage = row_leverage
        margin_fraction = row_margin
        notional = row_notional
        base_tp = float(row.get("take_profit", 0.0) or 0.0)
        base_sl = float(row.get("stop_loss", 0.0) or 0.0)
        if bool(notional_scaled_sltp):
            take_profit = base_tp * row_notional
            stop_loss = base_sl * row_notional
        else:
            take_profit = base_tp
            stop_loss = base_sl
        cash -= cash * fee_paid * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
        margin_sum += margin_fraction
        mfe = 0.0
        mae = 0.0
        armed = False

    if pos != 0:
        exit_px = omega_fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
        trades += 1
        win = int(cash > entry_equity)
        wins += win
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append({
            "entry_signal_i": int(entry_signal_i), "entry_i": int(entry_i), "exit_i": int(len(frame) - 1),
            "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]), "exit_timestamp": str(frame["timestamp"].iloc[-1]),
            "side": int(pos), "reason": "forced_end", "win": int(win), "raw_exit_price_move": float(raw_exit),
            "mfe_price_move": float(mfe), "mae_price_move": float(mae), "trade_return": float(trade_return),
            "net_per_notional": float(trade_return / max(notional, 1.0e-12)), "notional": float(notional),
            "margin_fraction": float(margin_fraction), "leverage": float(leverage), "exit_prob": 0.0,
            "take_profit": float(take_profit), "stop_loss": float(stop_loss),
        })

    n_entries = max(long_entries + short_entries, 1)
    ledger = pd.DataFrame(rows)
    hold_bars = (ledger["exit_i"] - ledger["entry_i"]).clip(lower=0) if len(ledger) else pd.Series(dtype=float)
    return (
        {
            "pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades),
            "wr": float(wins / trades) if trades else 0.0, "trades_per_day": float(trades / rs._duration_days(frame)),
            "avg_notional": float(notional_sum / n_entries), "avg_leverage": float(leverage_sum / n_entries),
            "avg_hold_bars": float(hold_bars.mean()) if len(hold_bars) else 0.0,
            "max_trade_pnl": float(ledger["trade_return"].max() * 100.0) if len(ledger) else 0.0,
            "p95_trade_pnl": float(ledger["trade_return"].quantile(0.95) * 100.0) if len(ledger) else 0.0,
            "long_entries": int(long_entries), "short_entries": int(short_entries), "exit_reasons": reasons,
        },
        ledger,
    )


def prep_component(name: str, cfg: dict, frame: pd.DataFrame, pred_csv: Path, *, oof: bool) -> dict[str, Any]:
    bundle = torch.load(cfg["bundle"], map_location="cpu", weights_only=False)
    base_cols = bundle["base_cols"]
    models = bundle["models"]

    src_raw = pd.read_csv(pred_csv)
    # pandas>=3.0 defaults CSV text columns to StringDtype, but the frozen
    # train_eval_omega4_2_risk_sidecar_20260622._risk_feature_frame() checks `dtype == object`
    # (written against pandas 2.x). Cast back to plain object here (harness-only fix, does not
    # touch the frozen sidecar/parent scripts) so router_expert one-hot expansion still fires.
    for c in src_raw.columns:
        if str(src_raw[c].dtype).lower().startswith("str"):
            src_raw[c] = src_raw[c].astype(object)
    src_raw["timestamp"] = pd.to_datetime(src_raw["timestamp"])
    keep_ts = set(src_raw["timestamp"])
    frame = frame[frame["timestamp"].isin(keep_ts)].reset_index(drop=True)
    src = src_raw[src_raw["timestamp"].isin(set(frame["timestamp"]))].reset_index(drop=True)
    if len(src) != len(frame) or not src["timestamp"].equals(frame["timestamp"]):
        raise RuntimeError(f"{name}: prediction/frame timestamp mismatch ({len(src)} vs {len(frame)})")

    x = parent._base_input(frame, base_cols)
    dec_base = parent._to_decisions(src, oof=oof)
    dec, _atr_diag = atr_eval._apply_atr_safety_sltp(
        dec_base, frame, atr_window=cfg["atr_window"], tp_mult=cfg["tp_mult"], sl_mult=cfg["sl_mult"],
        min_tp=cfg["min_tp"], min_sl=cfg["min_sl"], max_tp=cfg["max_tp"], max_sl=cfg["max_sl"],
    )
    atr_pct = atr_eval._atr_pct(frame, cfg["atr_window"])
    fee, slip = omega._load_fee_slip()
    loaded = parent._load_payloads(models, device=DEVICE)

    with open(cfg["sidecar_pkl"], "rb") as f:
        pkl = pickle.load(f)

    features = rs._risk_feature_frame(frame, src, dec, base_cols, atr_pct=atr_pct, feature_mode=pkl["risk_feature_mode"])
    x_all, _ = rs._feature_matrix(features, pkl["feature_columns"])
    side_all = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    score = rs._predict_side_split_models(pkl["model"], x_all, side_all) if pkl["side_split_model"] else np.asarray(pkl["model"].predict(x_all), dtype=np.float64)

    mapping = pkl["selected_mapping"]
    margin_kwargs = {k: mapping[k] for k in rs.MARGIN_CFG_KEYS}
    margin = rs._risk_margins(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **margin_kwargs)
    leverage = None
    if pkl["dynamic_leverage"]:
        lev_kwargs = {k: mapping[k] for k in rs.LEVERAGE_CFG_KEYS}
        leverage = rs._risk_leverage(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **lev_kwargs)

    return dict(
        frame=frame, x=x, dec=dec, loaded=loaded, margin=margin, leverage=leverage,
        fee=fee, slip=slip, notional_scaled_sltp=pkl["notional_scaled_sltp"],
    )


def run_grid(prepped: dict[str, dict[str, Any]], *, exit_thresholds: list[float],
             trailing_grid: list[tuple[float, float]] | None = None) -> pd.DataFrame:
    out = []
    for name, p in prepped.items():
        for et in exit_thresholds:
            m, _ledger = replay_exit_variant(
                p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
                exit_threshold=et, fee=p["fee"], slip=p["slip"], cost_mult=COST_MULT,
                notional_scaled_sltp=p["notional_scaled_sltp"], device=DEVICE,
            )
            out.append({"component": name, "experiment": "A_exit_threshold", "exit_threshold": et,
                        "trailing_activate": None, "trailing_retain": None, **m, "exit_reasons": json.dumps(m["exit_reasons"])})
        if trailing_grid:
            for act, ret in trailing_grid:
                m, _ledger = replay_exit_variant(
                    p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
                    exit_threshold=BASELINE_EXIT_THRESHOLD, fee=p["fee"], slip=p["slip"], cost_mult=COST_MULT,
                    notional_scaled_sltp=p["notional_scaled_sltp"], device=DEVICE,
                    trailing_activate_frac=act, trailing_retain_frac=ret,
                )
                out.append({"component": name, "experiment": "B_trailing_stop", "exit_threshold": BASELINE_EXIT_THRESHOLD,
                            "trailing_activate": act, "trailing_retain": ret, **m, "exit_reasons": json.dumps(m["exit_reasons"])})
    return pd.DataFrame(out)


def main() -> int:
    val_frame = load_frame(VAL_START, VAL_END, base_csv=BASE_2025, wide24_csv=WIDE24_2025)
    oos_frame = load_frame(OOS_START, OOS_END, base_csv=BASE_2026, wide24_csv=WIDE24_2026)
    print(f"VAL frame rows={len(val_frame)} range=[{val_frame['timestamp'].min()}, {val_frame['timestamp'].max()}]", flush=True)
    print(f"OOS frame rows={len(oos_frame)} range=[{oos_frame['timestamp'].min()}, {oos_frame['timestamp'].max()}]", flush=True)

    val_prepped = {}
    oos_prepped = {}
    for name, cfg in COMPONENTS.items():
        val_pred = EXT_PRED_DIR / name / f"validation_predictions_{cfg['q_tag']}.csv"
        oos_pred_full = EXT_PRED_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
        print(f"stage=prep component={name} split=VAL", flush=True)
        val_prepped[name] = prep_component(name, cfg, val_frame, val_pred, oof=True)
        print(f"stage=prep component={name} split=OOS", flush=True)
        oos_prepped[name] = prep_component(name, cfg, oos_frame, oos_pred_full, oof=False)

    # --- Experiment A: EXIT_THRESHOLD sweep on VAL ---
    exit_grid = [0.999, 0.99, 0.97, 0.95, 0.90, 0.85, 0.80, 0.70]
    print("stage=experiment_A_val", flush=True)
    val_a = run_grid(val_prepped, exit_thresholds=exit_grid)
    val_a.to_csv(ROOT / "tmp/research_20260721/exit_threshold_sweep_VAL.csv", index=False)
    print(val_a[["component", "exit_threshold", "pnl", "mdd", "trades", "wr", "avg_hold_bars"]].to_string(index=False), flush=True)

    # --- Experiment B: trailing-stop grid on VAL ---
    trailing_grid = [(a, r) for a in (0.4, 0.6, 0.8) for r in (0.4, 0.6, 0.8)]
    print("stage=experiment_B_val", flush=True)
    val_b = run_grid(val_prepped, exit_thresholds=[], trailing_grid=trailing_grid)
    val_b.to_csv(ROOT / "tmp/research_20260721/trailing_stop_sweep_VAL.csv", index=False)
    print(val_b[["component", "trailing_activate", "trailing_retain", "pnl", "mdd", "trades", "wr", "max_trade_pnl", "p95_trade_pnl"]].to_string(index=False), flush=True)

    # --- Baseline reference on OOS (exit_threshold=0.95, no trailing) ---
    print("stage=baseline_oos", flush=True)
    oos_baseline = run_grid(oos_prepped, exit_thresholds=[BASELINE_EXIT_THRESHOLD])
    oos_baseline.to_csv(ROOT / "tmp/research_20260721/baseline_OOS.csv", index=False)
    print(oos_baseline[["component", "exit_threshold", "pnl", "mdd", "trades", "wr"]].to_string(index=False), flush=True)

    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
