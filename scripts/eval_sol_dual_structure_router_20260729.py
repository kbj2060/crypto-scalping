#!/usr/bin/env python3
"""Validation-select a split-correct SOL ZIGZAG/H24-wide single-slot router."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_sol_20260707 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_sol_20260707 as sidecar  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707 as sol_parent  # noqa: E402


LABEL_DIR = ROOT / "tmp/causal_regen_20260516/sol_dual_zig075_h24wide_splitlocal_20260729/zig075"
OUT_ROOT = ROOT / "tmp/causal_regen_20260516"
SPLIT_TS = pd.Timestamp("2025-09-01")
VAL_END = pd.Timestamp("2026-01-01")
OOS_END = pd.Timestamp("2026-04-01")
EXPERT_NAMES = ("bull", "bear", "chop")
COMPONENT_NAMES = ("zig075", "h24wide")


def active(component: dict) -> np.ndarray:
    values = omega._active(component["dec"])
    return np.asarray(values, dtype=bool)


def masked_components(base: dict[str, dict], variant: dict) -> dict[str, dict]:
    output = {name: copy.copy(component) for name, component in base.items()}
    for name in output:
        output[name]["dec"] = output[name]["dec"].copy()

    kind = variant["kind"]
    if kind == "single":
        keep = variant["component"]
        return {keep: output[keep]}

    z, h = output["zig075"], output["h24wide"]
    za, ha = active(z), active(h)
    zs = pd.to_numeric(z["dec"]["side"], errors="raise").to_numpy(dtype=np.int8)
    hs = pd.to_numeric(h["dec"]["side"], errors="raise").to_numpy(dtype=np.int8)

    if kind == "conflict_cash":
        conflict = za & ha & (zs != hs)
        z["dec"].loc[conflict, "side"] = 0
        h["dec"].loc[conflict, "side"] = 0
    elif kind == "unanimity":
        agree = za & ha & (zs == hs)
        z["dec"].loc[~agree, "side"] = 0
        h["dec"].loc[~agree, "side"] = 0
    elif kind == "regime_anchor":
        route = np.asarray(z["route"], dtype=np.int8)
        assignment = variant["assignment"]
        for route_id, expert in enumerate(EXPERT_NAMES):
            choose = assignment[expert]
            hit = route == route_id
            other = "h24wide" if choose == "zig075" else "zig075"
            output[other]["dec"].loc[hit, "side"] = 0
    elif kind != "priority":
        raise ValueError(f"unknown variant kind: {kind}")
    return output


def make_variants(risk_mode: str) -> list[dict]:
    base_variants = [
        {"name": "single_zig075", "kind": "single", "component": "zig075", "priority": ["zig075"]},
        {"name": "single_h24wide", "kind": "single", "component": "h24wide", "priority": ["h24wide"]},
        {"name": "priority_zig075_first", "kind": "priority", "priority": ["zig075", "h24wide"]},
        {"name": "priority_h24wide_first", "kind": "priority", "priority": ["h24wide", "zig075"]},
        {"name": "conflict_cash", "kind": "conflict_cash", "priority": ["zig075", "h24wide"]},
        {"name": "unanimity_only", "kind": "unanimity", "priority": ["zig075", "h24wide"]},
    ]
    for choices in product(COMPONENT_NAMES, repeat=len(EXPERT_NAMES)):
        assignment = dict(zip(EXPERT_NAMES, choices, strict=True))
        suffix = "_".join(f"{expert}-{assignment[expert]}" for expert in EXPERT_NAMES)
        base_variants.append({
            "name": f"regime_anchor_{suffix}",
            "kind": "regime_anchor",
            "assignment": assignment,
            "priority": ["zig075", "h24wide"],
        })
    variants = []
    for variant in base_variants:
        scales = (1.0,) if variant["kind"] == "single" or risk_mode == "sidecar" else (0.80, 0.90, 0.95, 0.975, 1.0)
        for scale in scales:
            scaled = copy.deepcopy(variant)
            scaled["risk_scale"] = scale
            if scale != 1.0:
                scaled["name"] = f"{scaled['name']}_margin{int(round(scale * 1000)):04d}"
            variants.append(scaled)
    if risk_mode == "sidecar":
        anchor = next(variant for variant in base_variants if variant["name"] == "regime_anchor_bull-h24wide_bear-h24wide_chop-zig075")
        for threshold in (0.00543, 0.00547, 0.00553, 0.00557, 0.005595, 0.005605, 0.00561, 0.005615, 0.005625, 0.00564):
            gated = copy.deepcopy(anchor)
            gated["risk_scale"] = 1.0
            gated["duration_threshold"] = threshold
            gated["name"] = f"{gated['name']}_ougate_{threshold:.6f}"
            variants.append(gated)
        regime_scale_specs = (
            {"bull": 0.50, "bear": 1.0, "chop": 1.0},
            {"bull": 0.75, "bear": 1.0, "chop": 1.0},
            {"bull": 1.0, "bear": 0.50, "chop": 1.0},
            {"bull": 1.0, "bear": 0.75, "chop": 1.0},
            {"bull": 1.0, "bear": 1.0, "chop": 0.50},
            {"bull": 1.0, "bear": 1.0, "chop": 0.75},
            {"bull": 0.75, "bear": 0.75, "chop": 1.0},
        )
        for scale_map in regime_scale_specs:
            scaled = copy.deepcopy(anchor)
            scaled["risk_scale"] = 1.0
            scaled["regime_margin_scale"] = scale_map
            suffix = "_".join(f"{name}{int(round(scale_map[name] * 100)):03d}" for name in EXPERT_NAMES)
            scaled["name"] = f"{scaled['name']}_rscale_{suffix}"
            variants.append(scaled)
        combined_scale_specs = (
            {"bull": 0.00, "bear": 0.40, "chop": 1.0},
            {"bull": 0.00, "bear": 0.50, "chop": 1.0},
            {"bull": 0.00, "bear": 0.60, "chop": 1.0},
            {"bull": 0.25, "bear": 0.40, "chop": 1.0},
            {"bull": 0.25, "bear": 0.50, "chop": 1.0},
        )
        for scale_map in combined_scale_specs:
            scaled = copy.deepcopy(anchor)
            scaled["risk_scale"] = 1.0
            scaled["regime_margin_scale"] = scale_map
            suffix = "_".join(f"{name}{int(round(scale_map[name] * 100)):03d}" for name in EXPERT_NAMES)
            scaled["name"] = f"{scaled['name']}_rscale2_{suffix}"
            variants.append(scaled)
    for variant in variants:
        variant.setdefault("duration_threshold", None)
        variant.setdefault("regime_margin_scale", {name: 1.0 for name in EXPERT_NAMES})
    return variants


def prepare_component(frame: pd.DataFrame, pred_csv: Path, cfg: dict, device: torch.device, *, oof: bool) -> dict:
    bundle = torch.load(cfg["bundle"], map_location="cpu", weights_only=False)
    base_cols, models = bundle["base_cols"], bundle["models"]
    pred = pd.read_csv(pred_csv)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    if not pred["timestamp"].equals(frame["timestamp"]):
        raise RuntimeError(f"prediction timestamp contract mismatch: {pred_csv}")
    x = parent._base_input(frame, base_cols)
    dec_base = parent._to_decisions(pred, oof=oof)
    dec, _ = greedy.atr_eval._apply_atr_safety_sltp(
        dec_base, frame, atr_window=cfg["atr_window"], tp_mult=cfg["tp_mult"], sl_mult=cfg["sl_mult"],
        min_tp=cfg["min_tp"], min_sl=cfg["min_sl"], max_tp=cfg["max_tp"], max_sl=cfg["max_sl"],
    )
    atr = greedy.atr_eval._atr_pct(frame, cfg["atr_window"])
    loaded = parent._load_payloads(models, device=device)
    with open(cfg["sidecar_pkl"], "rb") as handle:
        payload = pickle.load(handle)
    features = sidecar._risk_feature_frame(frame, pred, dec, base_cols, atr_pct=atr, feature_mode=payload["risk_feature_mode"])
    x_all, _ = sidecar._feature_matrix(features, payload["feature_columns"])
    side_all = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    score = sidecar._predict_side_split_models(payload["model"], x_all, side_all) if payload["side_split_model"] else np.asarray(payload["model"].predict(x_all))
    mapping = payload["selected_mapping"]
    margin = sidecar._risk_margins(dec, score, train_q50=payload["train_score_q50"], train_iqr=payload["train_score_iqr"], **{key: mapping[key] for key in sidecar.MARGIN_CFG_KEYS})
    leverage = sidecar._risk_leverage(dec, score, train_q50=payload["train_score_q50"], train_iqr=payload["train_score_iqr"], **{key: mapping[key] for key in sidecar.LEVERAGE_CFG_KEYS}) if payload["dynamic_leverage"] else pd.to_numeric(dec["leverage"], errors="raise").to_numpy(dtype=np.float64)
    base_np, exit_runtime, pos_idx = sidecar._prepare_exit_runtime(x, loaded)
    return {"dec": dec, "atr": atr, "margin": margin, "leverage": leverage, "base_np": base_np, "exit_runtime": exit_runtime, "pos_idx": pos_idx, "route": greedy.hard._route_id(frame), "exit_threshold": cfg["exit_threshold"]}


def prepare(frame: pd.DataFrame, pred_paths: dict[str, Path], cfgs: dict[str, dict], device: torch.device, risk_mode: str, *, oof: bool) -> dict[str, dict]:
    result = {}
    for name in COMPONENT_NAMES:
        result[name] = prepare_component(frame, pred_paths[name], cfgs[name], device, oof=oof)
        if risk_mode == "parent_static":
            base_notional = pd.to_numeric(result[name]["dec"]["notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
            base_leverage = pd.to_numeric(result[name]["dec"]["leverage"], errors="raise").to_numpy(dtype=np.float64)
            result[name]["margin"] = base_notional / np.maximum(base_leverage, 1.0e-12)
            result[name]["leverage"] = base_leverage
            positive = base_notional > 0.0
            take_profit = np.where(positive, 0.026, 0.0)
            stop_loss = np.where(positive, 0.014, 0.0)
            result[name]["dec"]["take_profit"] = np.divide(take_profit, base_notional, out=np.zeros_like(take_profit), where=positive)
            result[name]["dec"]["stop_loss"] = np.divide(stop_loss, base_notional, out=np.zeros_like(stop_loss), where=positive)
            result[name]["exit_threshold"] = 2.0
            result[name]["entry_index_offset"] = 0
        else:
            result[name]["dec"]["max_hold_bars"] = 0
            result[name]["dec"]["cooldown_bars"] = 0
            result[name]["entry_index_offset"] = 1
    return result


@torch.no_grad()
def dual_replay(frame: pd.DataFrame, components: dict[str, dict], priority: tuple[str, ...], *, risk_scale: float, regime_margin_scale: dict[str, float], duration_threshold: float | None, fee: float, slip: float, cost_mult: float, device: torch.device) -> tuple[dict, pd.DataFrame]:
    arrays = {column: pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64) for column in ("open", "high", "low", "close")}
    actives = {name: active(component) for name, component in components.items()}
    fee_eff, slip_eff = float(fee) * cost_mult, float(slip) * cost_mult
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    active_name = ""
    entry_price = entry_equity = 1.0
    entry_i = entry_signal_i = 0
    notional = margin_fraction = leverage = 0.0
    take_profit = stop_loss = 0.0
    max_hold = next_cooldown = cooldown = 0
    mfe = mae = 0.0
    rows: list[dict] = []
    reasons: dict[str, int] = {}
    ou_halflife = pd.to_numeric(frame["ou_halflife"], errors="raise").to_numpy(dtype=np.float64)

    for i in range(0, len(frame) - 2):
        if pos != 0:
            close = float(arrays["close"][i])
            move = (close * (1.0 - slip_eff) - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - close * (1.0 + slip_eff)) / max(entry_price, 1.0e-12)
            mfe, mae = max(mfe, move), min(mae, move)
            equity = cash * (1.0 + move * notional)
        else:
            move, equity = 0.0, cash
        peak = max(peak, equity)
        mdd = min(mdd, equity / max(peak, 1.0e-12) - 1.0)

        if pos != 0:
            component = components[active_name]
            reason = ""
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            elif max_hold > 0 and i - entry_i >= max_hold:
                reason = "max_hold"
            elif float(component["exit_threshold"]) <= 1.0:
                hold = max(i - entry_i, 0)
                giveback = (mfe - move) / max(abs(mfe), 1.0e-8) if mfe > 0.0 else 0.0
                expert = greedy.hard.EXPERT_NAMES[int(component["route"][i])]
                probability = sidecar._predict_exit_prob_one(
                    component["base_np"], component["exit_runtime"], component["pos_idx"], row_i=i, expert=expert,
                    pos_values=[float(pos), float(hold), float(move), float(mfe), float(mae), float(np.clip(giveback, 0.0, 10.0)), float(take_profit - move), float(move + abs(stop_loss)), float(notional), float(leverage), float(notional * leverage), float(take_profit), float(stop_loss)],
                    device=device,
                )
                if probability >= float(component["exit_threshold"]):
                    reason = "exit_head"
            if reason:
                filled, exit_price, exit_fee, _ = omega._try_execution(arrays, i, pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_price - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1.0e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional) - before * exit_fee * notional
                trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append({"entry_signal_i": entry_signal_i, "entry_i": entry_i, "exit_i": i, "entry_timestamp": str(frame["timestamp"].iloc[entry_signal_i]), "exit_timestamp": str(frame["timestamp"].iloc[i]), "side": pos, "source_component": active_name, "reason": reason, "win": int(trade_return > 0.0), "trade_return": float(trade_return), "notional": notional, "margin_fraction": margin_fraction, "leverage": leverage})
                pos, active_name = 0, ""
                cooldown, next_cooldown = next_cooldown, 0
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if duration_threshold is not None and ou_halflife[i] <= float(duration_threshold):
            continue

        for name in priority:
            component = components[name]
            if not bool(actives[name][i]):
                continue
            side = int(component["dec"]["side"].iloc[i])
            if side == 0:
                continue
            route_name = EXPERT_NAMES[int(component["route"][i])]
            row_margin = float(component["margin"][i]) * float(risk_scale) * float(regime_margin_scale[route_name])
            row_leverage = min(float(component["leverage"][i]), greedy.LEVERAGE_CAP)
            row_notional = min(row_margin * row_leverage, greedy.NOTIONAL_CAP)
            if row_margin <= 0.0 or row_notional <= 0.0:
                continue
            filled, price, entry_fee, _ = omega._try_execution(arrays, i, side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
            if not filled:
                continue
            pos, active_name = side, name
            entry_price, entry_equity = float(price), cash
            entry_i = min(i + int(component["entry_index_offset"]), len(frame) - 1)
            entry_signal_i = i
            margin_fraction, leverage, notional = row_margin, row_notional / row_margin, row_notional
            take_profit = float(component["dec"]["take_profit"].iloc[i])
            stop_loss = float(component["dec"]["stop_loss"].iloc[i])
            max_hold = int(component["dec"]["max_hold_bars"].iloc[i])
            next_cooldown = int(component["dec"]["cooldown_bars"].iloc[i])
            cash -= cash * float(entry_fee) * notional
            mfe = mae = 0.0
            break

    if pos != 0:
        exit_price = omega._fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_price - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1.0e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional) - before * fee_eff * notional
        trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append({"entry_signal_i": entry_signal_i, "entry_i": entry_i, "exit_i": len(frame) - 1, "entry_timestamp": str(frame["timestamp"].iloc[entry_signal_i]), "exit_timestamp": str(frame["timestamp"].iloc[-1]), "side": pos, "source_component": active_name, "reason": "forced_end", "win": int(trade_return > 0.0), "trade_return": float(trade_return), "notional": notional, "margin_fraction": margin_fraction, "leverage": leverage})

    ledger = pd.DataFrame(rows)
    result = {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(len(ledger)), "wr": float(ledger["win"].mean()) if len(ledger) else 0.0, "exit_reasons": reasons}
    return result, ledger


def replay_variant(frame: pd.DataFrame, base: dict[str, dict], variant: dict, *, fee: float, slip: float, cost_mult: float, device: torch.device) -> tuple[dict, pd.DataFrame]:
    components = masked_components(base, variant)
    greedy.PRIORITY = tuple(variant["priority"])
    greedy.SCALE_MAP = {f"{name}_{side}": 1.0 for name in COMPONENT_NAMES for side in ("L", "S")}
    return dual_replay(frame, components, tuple(variant["priority"]), risk_scale=float(variant["risk_scale"]), regime_margin_scale=variant["regime_margin_scale"], duration_threshold=variant["duration_threshold"], fee=fee, slip=slip, cost_mult=cost_mult, device=device)


def selection_key(row: dict) -> tuple[float, float, int]:
    result = row["validation"]
    return (float(result["pnl"]), float(result["mdd"]), -int(result["trades"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zig-parent-dir", type=Path, required=True)
    parser.add_argument("--zig-risk-dir", type=Path, required=True)
    parser.add_argument("--zig-tag", required=True)
    parser.add_argument("--h24-parent-dir", type=Path, required=True)
    parser.add_argument("--h24-risk-dir", type=Path, required=True)
    parser.add_argument("--h24-tag", required=True)
    parser.add_argument("--risk-mode", choices=("sidecar", "parent_static"), required=True)
    parser.add_argument("--only-variant-name", default="")
    parser.add_argument("--variant-prefix", default="")
    parser.add_argument("--out-suffix", default="")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    suffix = f"_{args.out_suffix.strip()}" if args.out_suffix.strip() else ""
    out_dir = OUT_ROOT / f"sol_dual_structure_router_{args.risk_mode}_{args.zig_tag}_{args.h24_tag}_20260729{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    parent.SPLIT_TS = SPLIT_TS
    omega.SPLIT_TS = SPLIT_TS
    greedy.omega = omega
    greedy.sidecar = sidecar
    frames = sol_parent._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=LABEL_DIR,
        quality_mode="same_as_direction",
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    val_frame = frames["val_raw"].loc[frames["val_raw"]["timestamp"] < VAL_END].reset_index(drop=True)
    oos_frame = frames["oos_raw"].loc[frames["oos_raw"]["timestamp"] < OOS_END].reset_index(drop=True)
    if val_frame.empty or oos_frame.empty:
        raise RuntimeError("validation or OOS frame is empty")
    device = parent._device(args.device)
    fee, slip = omega._load_fee_slip()
    cfgs = {
        "zig075": {"bundle": args.zig_parent_dir / "true_3head_tabm_bundle.pt", "sidecar_pkl": args.zig_risk_dir / "risk_sidecar.pkl", "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0, "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12, "exit_threshold": 0.95},
        "h24wide": {"bundle": args.h24_parent_dir / "true_3head_tabm_bundle.pt", "sidecar_pkl": args.h24_risk_dir / "risk_sidecar.pkl", "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0, "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12, "exit_threshold": 0.95},
    }
    val_preds = {"zig075": args.zig_parent_dir / f"validation_predictions_{args.zig_tag}.csv", "h24wide": args.h24_parent_dir / f"validation_predictions_{args.h24_tag}.csv"}
    oos_preds = {"zig075": args.zig_parent_dir / f"oos_predictions_{args.zig_tag}.csv", "h24wide": args.h24_parent_dir / f"oos_predictions_{args.h24_tag}.csv"}

    print("stage=prepare_validation_components", flush=True)
    cost_mult = 3.0
    val_components = prepare(val_frame, val_preds, cfgs, device, args.risk_mode, oof=True)
    validation_rows = []
    variants = make_variants(args.risk_mode)
    if args.only_variant_name:
        variants = [variant for variant in variants if variant["name"] == args.only_variant_name]
        if len(variants) != 1:
            raise RuntimeError(f"expected exactly one --only-variant-name match, got {len(variants)}")
    if args.variant_prefix:
        variants = [variant for variant in variants if variant["name"].startswith(args.variant_prefix)]
        if not variants:
            raise RuntimeError("--variant-prefix matched no variants")
    for index, variant in enumerate(variants, start=1):
        result, _ = replay_variant(val_frame, val_components, variant, fee=fee, slip=slip, cost_mult=cost_mult, device=device)
        row = {"variant": variant, "validation": result}
        validation_rows.append(row)
        print(f"validation={index}/{len(variants)} {variant['name']} {result}", flush=True)

    published_baseline = {"validation_pnl": 23.45, "validation_mdd": -7.69, "oos_pnl": 7.66, "oos_mdd": -12.52}
    beats_published = [row for row in validation_rows if float(row["validation"]["pnl"]) > published_baseline["validation_pnl"] and float(row["validation"]["mdd"]) > published_baseline["validation_mdd"]]
    single_rows = [row for row in validation_rows if row["variant"]["kind"] == "single"]
    if single_rows:
        best_single_pnl = max(float(row["validation"]["pnl"]) for row in single_rows)
        best_single_mdd = max(float(row["validation"]["mdd"]) for row in single_rows)
        dominates_singles = [row for row in validation_rows if float(row["validation"]["pnl"]) >= best_single_pnl and float(row["validation"]["mdd"]) >= best_single_mdd]
    else:
        dominates_singles = []
    selection_pool = beats_published or dominates_singles or validation_rows
    selected = max(selection_pool, key=selection_key)

    print(f"selected_on_validation={selected['variant']['name']}", flush=True)
    if args.validation_only:
        search_report = {
            "model_id": f"sol_dual_structure_router_validation_search_{args.zig_tag}_{args.h24_tag}_20260729",
            "selected_on_validation": selected,
            "validation_candidates": sorted(validation_rows, key=selection_key, reverse=True),
            "oos_executed": False,
            "oos_used_for_selection": False,
            "fresh_forward_bar_by_bar": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
        }
        (out_dir / "validation_search_report.json").write_text(json.dumps(search_report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(search_report, indent=2), flush=True)
        return 0
    print("stage=prepare_frozen_oos_components", flush=True)
    oos_components = prepare(oos_frame, oos_preds, cfgs, device, args.risk_mode, oof=False)
    oos_result, oos_ledger = replay_variant(oos_frame, oos_components, selected["variant"], fee=fee, slip=slip, cost_mult=cost_mult, device=device)
    _, val_ledger = replay_variant(val_frame, val_components, selected["variant"], fee=fee, slip=slip, cost_mult=cost_mult, device=device)
    val_ledger.to_csv(out_dir / "selected_validation_trade_ledger_diagnostic.csv", index=False)
    oos_ledger.to_csv(out_dir / "selected_oos_trade_ledger_diagnostic.csv", index=False)

    report = {
        "model_id": f"sol_dual_structure_router_{args.risk_mode}_{args.zig_tag}_{args.h24_tag}_20260729",
        "components": {
            "zig075": {"out_dir": str(args.zig_risk_dir), "parent_dir": str(args.zig_parent_dir), "prediction_tag": args.zig_tag},
            "h24wide": {"out_dir": str(args.h24_risk_dir), "parent_dir": str(args.h24_parent_dir), "prediction_tag": args.h24_tag},
        },
        "selection_rule": "validation PnL first among variants that beat the published baseline on both PnL and MDD; otherwise among variants that dominate both best singles; otherwise validation PnL then MDD",
        "published_baseline": published_baseline,
        "selected_on_validation": selected,
        "frozen_oos": oos_result,
        "validation_candidates": sorted(validation_rows, key=selection_key, reverse=True),
        "evaluation_contract": {
            "validation_range": [str(val_frame["timestamp"].iloc[0]), str(val_frame["timestamp"].iloc[-1])],
            "oos_range": [str(oos_frame["timestamp"].iloc[0]), str(oos_frame["timestamp"].iloc[-1])],
            "fresh_forward_bar_by_bar": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "single_shared_position_slot": True,
            "oos_used_for_selection": False,
            "diagnostic_ledgers_written_after_selection": True,
        },
        "component_contract": {
            "zig075_prediction_tag": args.zig_tag,
            "h24wide_prediction_tag": args.h24_tag,
            "risk_scale": 1.0,
            "risk_mode": args.risk_mode,
            "cost_multiplier": cost_mult,
            "leverage_cap": greedy.LEVERAGE_CAP,
            "notional_cap": greedy.NOTIONAL_CAP,
            "notional_formula": "margin_fraction * leverage",
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "frozen_oos": oos_result}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
