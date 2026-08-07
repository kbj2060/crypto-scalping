#!/usr/bin/env python3
"""Two-component greedy router replay for SOL/BTC, exit30k recipe (2026-07-22).

Fork of replay_omega4_6_1_two_component_router_assets_20260708.py: only CONFIGS
is repointed at the exit30k parent/sidecar artifacts (max-exit-samples 30000,
matching ETH's own h48qual/zig075 training recipe -- the 07-08 version used the
default 12000, see docs/model_contracts/sol_btc_data_freshness_audit_20260721.md
and this session's investigation). Everything else (greedy router logic,
duration-gate selection, fresh-forward VAL-then-OOS discipline) is unchanged.
"""
from __future__ import annotations

import argparse
import importlib
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
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

ASSET_DATES = {"sol": "20260707", "btc": "20260708"}
LEVERAGE_CAP = 5.0
NOTIONAL_CAP = 1.8
PRIORITY = ("h48qual", "zig075")

CONFIGS = {
    "sol": {
        "h48qual": {
            "parent_dir": "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_h48qual_exit30k_20260722",
            "risk_dir": "tmp/causal_regen_20260516/sol_omega4_2_trade_risk_sidecar_20260707_adaptive_squeeze_h48qual_exit30k_q045_20260722",
            "tag": "q045",
            "long_scale": 1.0,
            "short_scale": 1.0,
        },
        "zig075": {
            "parent_dir": "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_exit30k_20260722",
            "risk_dir": "tmp/causal_regen_20260516/sol_omega4_2_trade_risk_sidecar_20260707_adaptive_squeeze_exit30k_q070_20260722",
            "tag": "q070",
            "long_scale": 1.0,
            "short_scale": 1.75,
        },
    },
    "btc": {
        "h48qual": {
            "parent_dir": "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_h48qual_exit30k_20260722",
            "risk_dir": "tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48qual_exit30k_q055_20260722",
            "tag": "q055",
            "long_scale": 0.5,
            "short_scale": 2.5,
        },
        "zig075": {
            "parent_dir": "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_zig075_exit30k_20260722",
            "risk_dir": "tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_zig075_exit30k_q065_20260722",
            "tag": "q065",
            "long_scale": 2.5,
            "short_scale": 2.75,
        },
    },
}


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


def _load_frames(asset: str) -> dict[str, Any]:
    date = ASSET_DATES[asset]
    omega4 = importlib.import_module(f"train_eval_omega4_3head_parent72_loose_entry_quality_{asset}_{date}")
    return omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=omega4.LABEL_DIR,
        quality_mode="same_as_direction",
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )


def _prepare_component(asset: str, split: str, frame: pd.DataFrame, cfg: dict[str, Any], *, device: torch.device) -> dict[str, Any]:
    date = ASSET_DATES[asset]
    sidecar = importlib.import_module(f"train_eval_omega4_2_risk_sidecar_{asset}_{date}")
    bundle = torch.load(ROOT / cfg["parent_dir"] / "true_3head_tabm_bundle.pt", map_location=device, weights_only=False)
    base_cols = list(bundle["base_cols"])
    models = bundle["models"]
    loaded = parent._load_payloads(models, device=device)
    pred = sidecar._load_precomputed_prediction(ROOT / cfg["parent_dir"], split, cfg["tag"], frame)
    x = parent._base_input(frame, base_cols)
    dec_base = parent._to_decisions(pred, oof=(split != "oos"))
    dec, _ = atr_eval._apply_atr_safety_sltp(
        dec_base,
        frame,
        atr_window=192,
        tp_mult=12.0,
        sl_mult=6.0,
        min_tp=0.075,
        min_sl=0.040,
        max_tp=0.22,
        max_sl=0.12,
    )
    atr = atr_eval._atr_pct(frame, 192)
    with open(ROOT / cfg["risk_dir"] / "risk_sidecar.pkl", "rb") as f:
        pkl = pickle.load(f)
    features = sidecar._risk_feature_frame(frame, pred, dec, base_cols, atr_pct=atr, feature_mode=pkl["risk_feature_mode"])
    x_all, _ = sidecar._feature_matrix(features, pkl["feature_columns"])
    side_all = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    score = sidecar._predict_side_split_models(pkl["model"], x_all, side_all) if pkl["side_split_model"] else np.asarray(pkl["model"].predict(x_all), dtype=np.float64)
    mapping = pkl["selected_mapping"]
    margin = sidecar._risk_margins(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS})
    leverage = sidecar._risk_leverage(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS}) if pkl["dynamic_leverage"] else np.ones(len(dec))
    base_np, exit_runtime, pos_idx = sidecar._prepare_exit_runtime(x, loaded)
    return {
        "dec": dec,
        "margin": margin,
        "leverage": leverage,
        "base_np": base_np,
        "exit_runtime": exit_runtime,
        "pos_idx": pos_idx,
        "route": hard._route_id(frame),
        "exit_threshold": 0.95,
        "long_scale": float(cfg["long_scale"]),
        "short_scale": float(cfg["short_scale"]),
        "sidecar": sidecar,
    }


@torch.no_grad()
def _greedy_replay(frame: pd.DataFrame, components: dict[str, Any], *, fee: float, slip: float, cost_mult: float, device: torch.device) -> pd.DataFrame:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    n = len(frame)
    cash = 1.0
    pos = 0
    active_comp: str | None = None
    entry_price = entry_equity = 1.0
    entry_i = entry_signal_i = 0
    notional = leverage_v = margin_fraction = 0.0
    take_profit = stop_loss = 0.0
    mfe = mae = 0.0
    rows: list[dict[str, Any]] = []

    active_masks = {name: pd.to_numeric(comp["dec"]["side"], errors="raise").to_numpy(dtype=np.int64) != 0 for name, comp in components.items()}

    for i in range(0, n - 2):
        if pos != 0:
            comp = components[str(active_comp)]
            sidecar = comp["sidecar"]
            close_px = arrays["close"][i] * (1 - slip_eff if pos > 0 else 1 + slip_eff)
            move = (close_px - entry_price) / entry_price if pos > 0 else (entry_price - close_px) / entry_price
            mfe = max(mfe, move)
            mae = min(mae, move)
            reason = ""
            exit_prob = 0.0
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            else:
                hold = max(i - entry_i, 0)
                giveback = (mfe - move) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(comp["route"][i])]
                exit_prob = sidecar._predict_exit_prob_one(
                    comp["base_np"],
                    comp["exit_runtime"],
                    comp["pos_idx"],
                    row_i=int(i),
                    expert=expert,
                    pos_values=[
                        float(pos),
                        float(hold),
                        float(move),
                        float(mfe),
                        float(mae),
                        float(np.clip(giveback, 0.0, 10.0)),
                        float(take_profit - move),
                        float(move + abs(stop_loss)),
                        float(notional),
                        float(leverage_v),
                        float(notional * leverage_v),
                        float(take_profit),
                        float(stop_loss),
                    ],
                    device=device,
                )
                if exit_prob >= float(comp["exit_threshold"]):
                    reason = "exit_head"
            if reason:
                before = cash
                cash = cash * (1.0 + move * notional)
                cash -= before * fee_eff * notional
                rows.append(
                    {
                        "entry_signal_i": entry_signal_i,
                        "entry_i": entry_i,
                        "exit_i": i,
                        "entry_timestamp": frame["timestamp"].iloc[entry_signal_i],
                        "exit_timestamp": frame["timestamp"].iloc[i],
                        "source_component": active_comp,
                        "side": int(pos),
                        "reason": reason,
                        "trade_return": float(cash / max(entry_equity, 1e-12) - 1.0),
                        "notional": float(notional),
                        "margin_fraction": float(margin_fraction),
                        "leverage": float(leverage_v),
                        "exit_prob": float(exit_prob),
                    }
                )
                pos = 0
                active_comp = None
            continue

        for name in PRIORITY:
            comp = components[name]
            if not active_masks[name][i]:
                continue
            side = int(comp["dec"]["side"].iloc[i])
            margin = float(comp["margin"][i])
            leverage = float(comp["leverage"][i])
            if margin <= 0.0 or leverage <= 0.0:
                continue
            scale = comp["long_scale"] if side > 0 else comp["short_scale"]
            leverage = min(leverage * scale, LEVERAGE_CAP)
            row_notional = min(margin * leverage, NOTIONAL_CAP)
            leverage = row_notional / max(margin, 1e-12)
            if row_notional <= 0.0:
                continue
            entry_px = arrays["open"][min(i + 1, n - 1)] * (1 + slip_eff if side > 0 else 1 - slip_eff)
            pos = side
            active_comp = name
            entry_i = min(i + 1, n - 1)
            entry_signal_i = i
            entry_price = float(entry_px)
            entry_equity = cash
            margin_fraction = margin
            leverage_v = leverage
            notional = row_notional
            take_profit = float(comp["dec"]["take_profit"].iloc[i])
            stop_loss = float(comp["dec"]["stop_loss"].iloc[i])
            cash -= cash * fee_eff * notional
            mfe = mae = 0.0
            break
    return pd.DataFrame(rows)


def _duration_search(ledger: pd.DataFrame) -> dict[str, Any]:
    candidates = [{"threshold": 0.0, "quantile": None, "validation": _compound_metrics(ledger), "eligible": True}]
    if len(ledger) == 0:
        return {"selected": candidates[0], "candidates": candidates}
    floor = max(1, int(np.floor(len(ledger) * 0.50)))
    for q in np.arange(0.05, 0.85, 0.05):
        th = float(np.quantile(ledger["ou_halflife"].to_numpy(dtype=np.float64), q))
        gated = ledger.loc[ledger["ou_halflife"] > th].reset_index(drop=True)
        m = _compound_metrics(gated)
        candidates.append({"threshold": th, "quantile": float(q), "validation": m, "eligible": int(m["trades"]) >= floor and float(m["mdd"]) >= -30.0})
    selected = max((c for c in candidates if c["eligible"]), key=lambda c: float(c["validation"]["pnl"]))
    return {"selected": selected, "candidates": candidates}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", choices=sorted(ASSET_DATES), required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    asset = args.asset
    date = ASSET_DATES[asset]
    out_dir = args.out_dir or ROOT / "tmp/causal_regen_20260516" / f"{asset}_omega4_6_1_two_component_router_exit30k_20260722"
    out_dir.mkdir(parents=True, exist_ok=True)
    omega = importlib.import_module(f"train_eval_omega1_2_tabm_diffusion_risk_{asset}_{date}")
    device = parent._device(str(args.device))
    frames = _load_frames(asset)
    fee, slip = omega._load_fee_slip()

    reports: dict[str, Any] = {}
    for split, frame_key in (("validation", "val_raw"), ("oos", "oos_raw")):
        frame = frames[frame_key]
        components = {
            name: _prepare_component(asset, split, frame, cfg, device=device)
            for name, cfg in CONFIGS[asset].items()
        }
        ledger = _greedy_replay(frame, components, fee=fee, slip=slip, cost_mult=3.0, device=device)
        ledger.to_csv(out_dir / f"{split}_router_ledger.csv", index=False)
        reports[split] = {
            "no_duration_gate": _compound_metrics(ledger),
            "source_counts": ledger["source_component"].value_counts().to_dict() if not ledger.empty else {},
        }
        if split == "validation":
            val_for_gate = ledger.merge(
                frame[["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp"}),
                on="entry_timestamp",
                how="left",
                validate="one_to_one",
            )
            duration = _duration_search(val_for_gate)
            selected_threshold = float(duration["selected"]["threshold"])
            reports["duration_gate"] = duration
            reports["selected_duration_threshold"] = selected_threshold
            gated = val_for_gate.loc[val_for_gate["ou_halflife"] > selected_threshold].reset_index(drop=True)
            reports[split]["with_duration_gate"] = _compound_metrics(gated)
        else:
            threshold = float(reports["selected_duration_threshold"])
            oos_for_gate = ledger.merge(
                frame[["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp"}),
                on="entry_timestamp",
                how="left",
                validate="one_to_one",
            )
            gated = oos_for_gate.loc[oos_for_gate["ou_halflife"] > threshold].reset_index(drop=True)
            q1 = gated.loc[pd.to_datetime(gated["entry_timestamp"]) < pd.Timestamp("2026-04-01")].reset_index(drop=True)
            reports[split]["with_duration_gate"] = _compound_metrics(gated)
            reports[split]["with_duration_gate_q1_2026"] = _compound_metrics(q1)

    report = {
        "method": "asset_two_component_greedy_router_exact_replay_exit30k",
        "asset": asset,
        "priority": list(PRIORITY),
        "component_configs": CONFIGS[asset],
        **reports,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
