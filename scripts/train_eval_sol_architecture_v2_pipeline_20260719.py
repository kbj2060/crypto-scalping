#!/usr/bin/env python3
"""Causal validation replay for the tuned SOL architecture-v2 pipeline.

Pipeline:
  causal TCN direction/state encoder
  side-conditional H24 CatBoost quality heads
  side-conditional return/adverse-excursion outcome heads
  separate position-aware exit classifier
  semi-Markov transition constraints and deterministic futures sizing

All selection in this script is validation-only.  OOS is intentionally not
loaded.  Saved ledgers are outputs, never replay inputs.
"""
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, CatBoostRegressor
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707 as sol  # noqa: E402
import train_eval_sol_architecture_v2_entry_20260719 as entry  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


ENTRY_ARTIFACT = ROOT / "tmp/causal_regen_20260516/sol_architecture_v2_entry_h24_tuned_20260719/tcn_l24_h32.pt"
QUALITY_DIR = ROOT / "tmp/causal_regen_20260516/sol_architecture_v2_horizon_ceiling_20260719"
TB_DIR = ROOT / "tmp/causal_regen_20260516/sol_omega1_2_triple_barrier_labels_hysteresis_rebuild_20260719"
LABEL_DIR = ROOT / "tmp/causal_regen_20260516/sol_zigzag_hysteresis_labels_20260719"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sol_architecture_v2_pipeline_20260719"
FEE_SLIP_COST_MULT = 3.0


@dataclass(frozen=True)
class ReplayCfg:
    direction_threshold: float
    quality_threshold: float
    edge_threshold: float
    entry_confirmation: int
    cooldown: int
    transition_confirmation: int
    min_hold: int
    max_hold: int
    atr_window: int
    tp_mult: float
    sl_mult: float
    min_tp: float
    min_sl: float
    max_tp: float
    max_sl: float
    margin_fraction: float
    leverage: float
    outcome_sizing: bool
    exit_threshold: float | None


def _load_tb() -> pd.DataFrame:
    cols = [
        "timestamp",
        "tb_long_ret_h24_conservative",
        "tb_short_ret_h24_conservative",
        "tb_long_mae_h24_conservative",
        "tb_short_mae_h24_conservative",
    ]
    parts = [pd.read_csv(TB_DIR / f"{split}_triple_barrier_labels.csv", usecols=cols, parse_dates=["timestamp"]) for split in ("train", "validation")]
    return pd.concat(parts, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp", keep="last")


def _predict_tcn(payload: dict[str, Any], frames: pd.DataFrame, validation_start: int, device: torch.device) -> np.ndarray:
    variant = entry.Variant(**payload["variant"])
    model = entry.SoftResidualEntry(len(payload["feature_columns"]), variant).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    raw = frames[payload["feature_columns"]].to_numpy(dtype=np.float32)
    x = ((raw - payload["feature_mean"]) / payload["feature_std"]).astype(np.float32)
    gates = frames[hard.ROUTE_COLS].to_numpy(dtype=np.float32)
    gates /= np.clip(gates.sum(axis=1, keepdims=True), 1.0e-8, None)
    zeros_dir = np.zeros(len(frames), dtype=np.int64)
    zeros_q = np.zeros((len(frames), 2), dtype=np.float32)
    zeros_o = np.zeros((len(frames), 6), dtype=np.float32)
    ds = entry.SequenceRows(x, gates, zeros_dir, zeros_q, zeros_o, seq_len=variant.seq_len, start=validation_start, end=len(frames))
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
    chunks = []
    with torch.no_grad():
        for xb, gb, _yd, _yq, _yo, _idx in loader:
            chunks.append(torch.softmax(model(xb.to(device), gb.to(device))["direction"], dim=1).cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def _train_outcome_models(train: pd.DataFrame, base_cols: list[str], out_dir: Path) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for side in ("long", "short"):
        ret_path = out_dir / f"outcome_{side}_return.cbm"
        adverse_path = out_dir / f"outcome_{side}_adverse_q80.cbm"
        if ret_path.exists() and adverse_path.exists():
            models[f"{side}_return"] = CatBoostRegressor().load_model(str(ret_path))
            models[f"{side}_adverse"] = CatBoostRegressor().load_model(str(adverse_path))
            continue
        ret_col = f"tb_{side}_ret_h24_conservative"
        mae_col = f"tb_{side}_mae_h24_conservative"
        ret = CatBoostRegressor(
            iterations=600,
            depth=7,
            learning_rate=0.03,
            loss_function="RMSE",
            random_seed=260719,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        adverse = CatBoostRegressor(
            iterations=600,
            depth=7,
            learning_rate=0.03,
            loss_function="Quantile:alpha=0.80",
            random_seed=260719 + 1,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        ret.fit(train[base_cols], train[ret_col])
        adverse.fit(train[base_cols], np.abs(train[mae_col].to_numpy(dtype=np.float64)))
        ret.save_model(str(ret_path))
        adverse.save_model(str(adverse_path))
        models[f"{side}_return"] = ret
        models[f"{side}_adverse"] = adverse
    return models


def _train_exit_model(frames: dict[str, Any], base_cols: list[str], out_dir: Path) -> tuple[CatBoostClassifier, list[str], dict[str, Any]]:
    model_path = out_dir / "separate_exit_classifier.cbm"
    pipeline_path = out_dir / "pipeline.pkl"
    if model_path.exists() and pipeline_path.exists():
        with pipeline_path.open("rb") as handle:
            previous = pickle.load(handle)
        model = CatBoostClassifier().load_model(str(model_path))
        return model, list(previous["exit_columns"]), {"reused": True, "source": str(model_path), "best_iteration": int(model.get_best_iteration())}
    fee, slip = sol.omega._load_fee_slip()
    x_exit_raw, y_exit, _route, diag = sol._build_exit_dataset_entry_label_terminal_giveback(
        frames["train_df"],
        frames["s_train_label"],
        fee=fee,
        slip=slip,
        cost_mult=FEE_SLIP_COST_MULT,
        max_samples=30000,
        terminal_window=3,
        adverse_unreal=-0.010,
        min_mfe_for_giveback=0.006,
        giveback_min=0.65,
    )
    x_exit = parent._exit_input_from_position_rows(x_exit_raw, base_cols)
    if model_path.exists():
        model = CatBoostClassifier().load_model(str(model_path))
        return model, list(x_exit.columns), {"rows": len(x_exit), "positive_rate": float(np.mean(y_exit)), "best_iteration": int(model.get_best_iteration()), "label_diag": diag, "reused": True}
    split = int(len(x_exit) * 0.85)
    model = CatBoostClassifier(
        iterations=1000,
        depth=7,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=260720,
        od_type="Iter",
        od_wait=80,
        verbose=100,
        allow_writing_files=False,
        thread_count=-1,
    )
    model.fit(x_exit.iloc[:split], y_exit[:split], eval_set=(x_exit.iloc[split:], y_exit[split:]), use_best_model=True)
    model.save_model(str(model_path))
    return model, list(x_exit.columns), {"rows": len(x_exit), "positive_rate": float(np.mean(y_exit)), "best_iteration": int(model.get_best_iteration()), "label_diag": diag}


def _compound(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cash = peak = 1.0
    mdd = 0.0
    for row in rows:
        cash *= 1.0 + float(row["trade_return"])
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1.0e-12) - 1.0)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(len(rows)),
        "wr": float(np.mean([float(row["trade_return"]) > 0.0 for row in rows])) if rows else 0.0,
    }


def _replay(
    frame: pd.DataFrame,
    direction: np.ndarray,
    quality: np.ndarray,
    ret_pred: np.ndarray,
    adverse_pred: np.ndarray,
    cfg: ReplayCfg,
    *,
    exit_model: CatBoostClassifier | None,
    exit_columns: list[str] | None,
    base_cols: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    if len(frame) != len(direction):
        raise RuntimeError("prediction/frame length mismatch")
    arrays = {col: frame[col].to_numpy(dtype=np.float64) for col in ("open", "high", "low", "close")}
    atr = atr_eval._atr_pct(frame, int(cfg.atr_window))
    fee, slip = sol.omega._load_fee_slip()
    fee_eff = float(fee) * FEE_SLIP_COST_MULT
    slip_eff = float(slip) * FEE_SLIP_COST_MULT
    pos = 0
    entry_i = 0
    entry_price = 0.0
    notional = margin = 0.0
    tp = sl = 0.0
    mfe = 0.0
    mae = 0.0
    rows: list[dict[str, Any]] = []
    cooldown_until = 0
    pending_action = 0
    pending_count = 0
    transition_count = 0
    for i in range(0, len(frame) - 2):
        if pos != 0:
            close_px = arrays["close"][i] * (1.0 - slip_eff if pos > 0 else 1.0 + slip_eff)
            move = (close_px - entry_price) / entry_price if pos > 0 else (entry_price - close_px) / entry_price
            mfe = max(mfe, move)
            mae = min(mae, move)
            hold = i - entry_i
            reason = ""
            if move <= -abs(sl):
                reason = "stop_loss"
            elif move >= tp:
                reason = "take_profit"
            elif hold >= int(cfg.max_hold):
                reason = "max_hold"
            elif hold >= int(cfg.min_hold):
                proposed = int(np.argmax(direction[i]))
                expected = 1 if pos > 0 else 2
                if proposed != expected:
                    transition_count += 1
                    if int(cfg.transition_confirmation) > 0 and transition_count >= int(cfg.transition_confirmation):
                        reason = "state_transition"
                elif exit_model is not None and exit_columns is not None and cfg.exit_threshold is not None:
                    transition_count = 0
                    giveback = (mfe - move) / max(abs(mfe), 1.0e-8) if mfe > 0 else 0.0
                    data = {col: float(frame.iloc[i][col]) if col in base_cols else 0.0 for col in base_cols}
                    data.update(
                        {
                            "pos_side": float(pos),
                            "pos_hold_bars": float(hold),
                            "pos_unrealized": float(move * notional),
                            "pos_mfe": float(mfe * notional),
                            "pos_mae": float(mae * notional),
                            "pos_giveback": float(np.clip(giveback, 0.0, 10.0)),
                            "pos_dist_to_tp": float((tp - move) * notional),
                            "pos_dist_to_sl": float((move + abs(sl)) * notional),
                            "pos_notional": float(notional),
                            "pos_leverage": float(cfg.leverage),
                            "pos_exposure": float(notional),
                            "pos_tp": float(tp * notional),
                            "pos_sl": float(sl * notional),
                        }
                    )
                    xrow = pd.DataFrame([[data[col] for col in exit_columns]], columns=exit_columns)
                    if float(exit_model.predict_proba(xrow)[0, 1]) >= float(cfg.exit_threshold):
                        reason = "exit_head"
            if reason:
                trade_return = move * notional - fee_eff * notional * 2.0
                rows.append(
                    {
                        "entry_timestamp": frame.iloc[entry_i]["timestamp"],
                        "exit_timestamp": frame.iloc[i]["timestamp"],
                        "side": pos,
                        "hold_bars": hold,
                        "margin_fraction": margin,
                        "leverage": cfg.leverage,
                        "notional": notional,
                        "take_profit": tp,
                        "stop_loss": sl,
                        "trade_return": trade_return,
                        "reason": reason,
                    }
                )
                pos = 0
                cooldown_until = i + int(cfg.cooldown)
                pending_action = 0
                pending_count = 0
                transition_count = 0
            continue
        if i < cooldown_until:
            continue
        action = int(np.argmax(direction[i]))
        if action == 0 or float(direction[i, action]) < float(cfg.direction_threshold):
            pending_action = 0
            pending_count = 0
            continue
        side_idx = action - 1
        if float(quality[i, side_idx]) < float(cfg.quality_threshold):
            pending_action = 0
            pending_count = 0
            continue
        if float(ret_pred[i, side_idx]) < float(cfg.edge_threshold):
            pending_action = 0
            pending_count = 0
            continue
        if action != pending_action:
            pending_action = action
            pending_count = 1
        else:
            pending_count += 1
        if pending_count < int(cfg.entry_confirmation):
            continue
        pos = 1 if action == 1 else -1
        entry_i = i + 1
        entry_price = arrays["open"][entry_i] * (1.0 + slip_eff if pos > 0 else 1.0 - slip_eff)
        margin = float(cfg.margin_fraction)
        if cfg.outcome_sizing:
            edge_score = float(np.clip((ret_pred[i, side_idx] - cfg.edge_threshold) / 0.006, 0.0, 1.0))
            quality_score = float(np.clip((quality[i, side_idx] - cfg.quality_threshold) / max(1.0 - cfg.quality_threshold, 1.0e-6), 0.0, 1.0))
            margin *= 0.50 + 0.50 * (0.5 * edge_score + 0.5 * quality_score)
        notional = margin * float(cfg.leverage)
        atr_v = float(atr[i]) if np.isfinite(atr[i]) else 0.0
        tp = float(np.clip(max(cfg.min_tp, atr_v * cfg.tp_mult), cfg.min_tp, cfg.max_tp))
        sl_atr = float(np.clip(max(cfg.min_sl, atr_v * cfg.sl_mult), cfg.min_sl, cfg.max_sl))
        sl = max(sl_atr, float(adverse_pred[i, side_idx]) * 1.10) if cfg.outcome_sizing else sl_atr
        sl = min(sl, cfg.max_sl)
        mfe = 0.0
        mae = 0.0
        pending_action = 0
        pending_count = 0
        transition_count = 0
    if pos != 0:
        i = len(frame) - 1
        close_px = arrays["close"][i] * (1.0 - slip_eff if pos > 0 else 1.0 + slip_eff)
        move = (close_px - entry_price) / entry_price if pos > 0 else (entry_price - close_px) / entry_price
        rows.append(
            {
                "entry_timestamp": frame.iloc[entry_i]["timestamp"],
                "exit_timestamp": frame.iloc[i]["timestamp"],
                "side": pos,
                "hold_bars": i - entry_i,
                "margin_fraction": margin,
                "leverage": cfg.leverage,
                "notional": notional,
                "take_profit": tp,
                "stop_loss": sl,
                "trade_return": move * notional - fee_eff * notional * 2.0,
                "reason": "forced_end",
            }
        )
    ledger = pd.DataFrame(rows)
    return _compound(rows), ledger


def _cfg_dict(cfg: ReplayCfg) -> dict[str, Any]:
    return {name: getattr(cfg, name) for name in cfg.__dataclass_fields__}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(ENTRY_ARTIFACT, map_location="cpu", weights_only=False)
    base_cols = list(payload["feature_columns"])
    frames = sol._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=LABEL_DIR,
        quality_mode="same_as_direction",
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    train = frames["train_raw"].copy()
    validation = frames["val_raw"].copy()
    tb = _load_tb()
    train = train.merge(tb, on="timestamp", how="inner", validate="one_to_one")
    combined = pd.concat([frames["train_raw"], validation], ignore_index=True)
    direction = _predict_tcn(payload, combined, len(frames["train_raw"]), device)
    quality_models = {side: CatBoostClassifier().load_model(str(QUALITY_DIR / f"h24_{side}.cbm")) for side in ("long", "short")}
    quality = np.column_stack([quality_models[side].predict_proba(validation[base_cols])[:, 1] for side in ("long", "short")])
    outcome_models = _train_outcome_models(train, base_cols, OUT_DIR)
    ret_pred = np.column_stack([outcome_models[f"{side}_return"].predict(validation[base_cols]) for side in ("long", "short")])
    adverse_pred = np.column_stack([outcome_models[f"{side}_adverse"].predict(validation[base_cols]) for side in ("long", "short")])
    exit_model, exit_columns, exit_diag = _train_exit_model(frames, base_cols, OUT_DIR)

    base = dict(
        min_hold=24,
        entry_confirmation=12,
        cooldown=24,
        transition_confirmation=12,
        max_hold=96,
        atr_window=192,
        tp_mult=10.0,
        sl_mult=5.0,
        min_tp=0.050,
        min_sl=0.025,
        max_tp=0.18,
        max_sl=0.09,
        margin_fraction=0.20,
        leverage=3.0,
        outcome_sizing=False,
        exit_threshold=None,
    )
    stage1 = []
    for d in (0.34, 0.38, 0.42):
        for q in (0.47, 0.49, 0.50, 0.51, 0.52):
            for edge_threshold in (-0.001, 0.0, 0.0005):
                cfg = ReplayCfg(direction_threshold=d, quality_threshold=q, edge_threshold=edge_threshold, **base)
                metrics, _ = _replay(validation, direction, quality, ret_pred, adverse_pred, cfg, exit_model=None, exit_columns=None, base_cols=base_cols)
                stage1.append({**_cfg_dict(cfg), **metrics})
    stage1.sort(key=lambda row: (row["trades"] >= 20, row["pnl"], row["mdd"]), reverse=True)
    pd.DataFrame(stage1).to_csv(OUT_DIR / "validation_stage1_entry_ranking.csv", index=False)

    stage2 = []
    active_stage1 = [row for row in stage1 if row["trades"] >= 20]
    for seed_row in active_stage1[:6]:
        for entry_confirmation in (6, 12, 24):
            for cooldown in (12, 24, 48):
                for min_hold in (12, 24, 48):
                    for max_hold in (96, 144):
                        for atr_contract in (
                            (8.0, 4.0, 0.040, 0.020),
                            (10.0, 5.0, 0.050, 0.025),
                            (12.0, 6.0, 0.060, 0.030),
                            ):
                            for margin in (0.15, 0.20, 0.25):
                                for outcome_sizing in (False, True):
                                    cfg = ReplayCfg(
                                        direction_threshold=float(seed_row["direction_threshold"]),
                                        quality_threshold=float(seed_row["quality_threshold"]),
                                        edge_threshold=float(seed_row["edge_threshold"]),
                                        entry_confirmation=entry_confirmation,
                                        cooldown=cooldown,
                                        transition_confirmation=12,
                                        min_hold=min_hold,
                                        max_hold=max_hold,
                                        atr_window=192,
                                        tp_mult=atr_contract[0],
                                        sl_mult=atr_contract[1],
                                        min_tp=atr_contract[2],
                                        min_sl=atr_contract[3],
                                        max_tp=0.18,
                                        max_sl=0.09,
                                        margin_fraction=margin,
                                        leverage=3.0,
                                        outcome_sizing=outcome_sizing,
                                        exit_threshold=None,
                                    )
                                    metrics, _ = _replay(validation, direction, quality, ret_pred, adverse_pred, cfg, exit_model=None, exit_columns=None, base_cols=base_cols)
                                    stage2.append({**_cfg_dict(cfg), **metrics})
    unique = pd.DataFrame(stage2).drop_duplicates(subset=list(ReplayCfg.__dataclass_fields__))
    unique = unique[unique["trades"] >= 20].sort_values(["pnl", "mdd"], ascending=[False, False])
    if unique.empty:
        raise RuntimeError("no stage2 candidate met the minimum 20-trade validation gate")
    unique.to_csv(OUT_DIR / "validation_stage2_risk_duration_ranking.csv", index=False)

    final_rows = []
    for _, seed_row in unique.head(8).iterrows():
        for transition_confirmation in (0, 6, 12, 24):
            for exit_threshold in (0.60, 0.70, 0.80, 0.90, None):
                values = {name: seed_row[name] for name in ReplayCfg.__dataclass_fields__}
                values["min_hold"] = int(values["min_hold"])
                values["max_hold"] = int(values["max_hold"])
                values["atr_window"] = int(values["atr_window"])
                values["outcome_sizing"] = bool(values["outcome_sizing"])
                values["transition_confirmation"] = transition_confirmation
                values["exit_threshold"] = exit_threshold
                cfg = ReplayCfg(**values)
                metrics, ledger = _replay(
                    validation,
                    direction,
                    quality,
                    ret_pred,
                    adverse_pred,
                    cfg,
                    exit_model=exit_model if exit_threshold is not None else None,
                    exit_columns=exit_columns,
                    base_cols=base_cols,
                )
                final_rows.append({**_cfg_dict(cfg), **metrics})
                if len(final_rows) == 1:
                    ledger.to_csv(OUT_DIR / "validation_first_final_ledger.csv", index=False)
    final_rows.sort(key=lambda row: (row["pnl"], row["mdd"]), reverse=True)
    pd.DataFrame(final_rows).to_csv(OUT_DIR / "validation_final_ranking.csv", index=False)
    best_cfg = ReplayCfg(**{name: final_rows[0][name] for name in ReplayCfg.__dataclass_fields__})
    best_metrics, best_ledger = _replay(
        validation,
        direction,
        quality,
        ret_pred,
        adverse_pred,
        best_cfg,
        exit_model=exit_model if best_cfg.exit_threshold is not None else None,
        exit_columns=exit_columns,
        base_cols=base_cols,
    )
    best_ledger.to_csv(OUT_DIR / "validation_selected_ledger.csv", index=False)
    with (OUT_DIR / "pipeline.pkl").open("wb") as handle:
        pickle.dump(
            {
                "entry_artifact": str(ENTRY_ARTIFACT),
                "quality_models": quality_models,
                "outcome_models": outcome_models,
                "exit_model": exit_model,
                "exit_columns": exit_columns,
                "base_cols": base_cols,
                "selected_config": _cfg_dict(best_cfg),
            },
            handle,
        )
    report = {
        "selection_scope": "validation_only",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "validation_range": [str(validation.timestamp.iloc[0]), str(validation.timestamp.iloc[-1])],
        "oos_used": False,
        "architecture": "TCN24 direction + conditional H24 quality + outcome heads + separate exit + semi-Markov transitions",
        "exit_model": exit_diag,
        "selected": {"config": _cfg_dict(best_cfg), "validation": best_metrics},
        "candidate_counts": {"stage1": len(stage1), "stage2": len(unique), "final": len(final_rows)},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
