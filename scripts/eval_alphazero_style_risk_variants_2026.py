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
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, FEATURE_COLS  # noqa: E402
from scripts.eval_hf_entry_overlay_grid import _audit  # noqa: E402
from scripts.train_eval_alphazero_style_governor_2026 import (  # noqa: E402
    AZExitModel,
    ENTRY_ACTIONS,
    EXIT_ACTIONS,
    DEFAULT_MODEL_OUT,
    PolicyValueNet,
    PVBundle,
    _entry_modified_decisions,
    _monthly,
    _predict_pv,
)
from scripts.train_eval_dsac_replacement_heads_2026 import (  # noqa: E402
    DEFAULT_EVAL_CSV,
    DEFAULT_EXIT_BUNDLE,
    DEFAULT_POLICY,
    DEFAULT_SELECTION,
    DEFAULT_TRAIN_CSV,
    _load_selected,
    _read,
)
from scripts.train_eval_hf_no_limit_exit_governor import (  # noqa: E402
    MODEL_COLS,
    _base_frame,
    _compact,
    backtest_no_limit_exit,
)


DEFAULT_REPORT = ROOT / "data/ensemble/reports/alphazero_style_risk_variants_2026.json"


def _load_bundle(model_path: Path, key: str, actions: tuple[str, ...], device: str) -> PVBundle:
    payload = torch.load(model_path, map_location=device, weights_only=False)
    section = payload[key]
    hidden = int(section["state_dict"]["trunk.0.weight"].shape[0])
    net = PolicyValueNet(len(section["feature_cols"]), len(actions), hidden_dim=hidden).to(device)
    net.load_state_dict(section["state_dict"])
    return PVBundle(
        net=net,
        mean=np.asarray(section["mean"], dtype=np.float32),
        std=np.asarray(section["std"], dtype=np.float32),
        feature_cols=list(section["feature_cols"]),
        actions=actions,
    )


def _value_adjusted_decisions(
    base_dec: pd.DataFrame,
    probs: np.ndarray,
    values: np.ndarray,
    *,
    min_keep_prob: float,
    value_floor: float,
    value_ceiling: float,
    mode: str,
) -> pd.DataFrame:
    out = _entry_modified_decisions(base_dec, probs, min_keep_prob=float(min_keep_prob))
    values = np.asarray(values, dtype=np.float64)
    active = (out["action"].astype(int).to_numpy() != ACTION_CASH) & (out["side"].astype(int).to_numpy() != 0)
    lev = pd.to_numeric(out["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    notional = pd.to_numeric(out["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    if mode == "block":
        notional = np.where(active & (values < float(value_floor)), 0.0, notional)
    elif mode == "soft":
        span = max(float(value_ceiling) - float(value_floor), 1e-6)
        scale = np.clip((values - float(value_floor)) / span, 0.0, 1.0)
        # Keep high-value decisions untouched, fade low-value decisions instead of adding another hard gate.
        notional = np.where(active, notional * scale, notional)
    elif mode == "convex":
        span = max(float(value_ceiling) - float(value_floor), 1e-6)
        scale = np.clip((values - float(value_floor)) / span, 0.0, 1.15)
        scale = scale * scale
        notional = np.where(active, notional * scale, notional)
    else:
        raise ValueError(f"unknown mode: {mode}")

    block = notional <= 0.05
    out.loc[:, "notional_exposure"] = notional
    out.loc[:, "position_fraction"] = notional / np.maximum(lev, 1e-12)
    out.loc[block, ["action", "side", "notional_exposure", "position_fraction"]] = 0
    out.loc[block, "leverage"] = 1.0
    return out


def _run(
    name: str,
    eval_df: pd.DataFrame,
    policy: dict[str, Any],
    exit_model: Any,
    entry_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    exit_cfg: dict[str, Any],
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    *,
    fee: float,
    slip: float,
    include_monthly: bool = False,
) -> dict[str, Any]:
    bt = backtest_no_limit_exit(
        eval_df,
        policy,
        exit_model,
        entry_config=entry_cfg,
        risk_config=risk_cfg,
        exit_threshold=float(exit_cfg["exit_threshold"]),
        min_exit_age=int(exit_cfg["min_exit_age"]),
        fee=float(fee),
        slip=float(slip),
        precomputed=precomputed,
    )
    row = {"name": name, "eval": _compact(bt)}
    if include_monthly:
        row["monthly"] = _monthly(eval_df, policy, exit_model, entry_cfg, risk_cfg, exit_cfg, precomputed, fee, slip)
    row["score"] = float(row["eval"]["pnl"] + 3.0 * row["eval"]["mdd"])
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Risk-aware AlphaZero-style governor variant search.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_OUT)
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--exit-bundle", type=Path, default=DEFAULT_EXIT_BUNDLE)
    p.add_argument("--selection-report", type=Path, default=DEFAULT_SELECTION)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = "cuda" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    policy = joblib.load(args.policy)
    exit_bundle = joblib.load(args.exit_bundle)
    base_exit_model = exit_bundle["model"] if isinstance(exit_bundle, dict) and "model" in exit_bundle else exit_bundle
    entry_cfg, risk_cfg, exit_cfg = _load_selected(args.selection_report)
    eval_df = _read(args.eval_csv)
    eval_pre = _base_frame(eval_df, policy, entry_cfg)
    eval_feat, eval_dec, eval_close, eval_fill = eval_pre

    entry_bundle = _load_bundle(args.model, "entry", ENTRY_ACTIONS, device)
    exit_bundle_pv = _load_bundle(args.model, "exit", EXIT_ACTIONS, device)
    entry_x = eval_feat.reindex(columns=FEATURE_COLS).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    entry_probs, entry_values = _predict_pv(entry_bundle, entry_x, device)
    az_exit_model = AZExitModel(exit_bundle_pv, device)

    rows: list[dict[str, Any]] = []
    rows.append(_run("baseline_hf_no_limit", eval_df, policy, base_exit_model, entry_cfg, risk_cfg, exit_cfg, eval_pre, fee=args.fee, slip=args.slip))
    entry_floors = (0.0, 0.40, 0.60, 0.70)
    exit_thresholds = (0.45, 0.55)
    value_floors = (0.00, 0.04, 0.08, 0.12, 0.18)
    value_modes = ("block", "soft")

    for floor in entry_floors:
        base_dec = _entry_modified_decisions(eval_dec, entry_probs, min_keep_prob=floor)
        rows.append(_run(f"az_entry_floor{floor:.2f}", eval_df, policy, base_exit_model, entry_cfg, risk_cfg, exit_cfg, (eval_feat, base_dec, eval_close, eval_fill), fee=args.fee, slip=args.slip))
        for ex_th in exit_thresholds:
            rows.append(_run(f"az_entry_floor{floor:.2f}_exit{ex_th:.2f}", eval_df, policy, az_exit_model, entry_cfg, risk_cfg, {"exit_threshold": ex_th, "min_exit_age": exit_cfg["min_exit_age"]}, (eval_feat, base_dec, eval_close, eval_fill), fee=args.fee, slip=args.slip))

    for mode in value_modes:
        for floor in entry_floors:
            for vf in value_floors:
                dec = _value_adjusted_decisions(
                    eval_dec,
                    entry_probs,
                    entry_values,
                    min_keep_prob=floor,
                    value_floor=vf,
                    value_ceiling=0.55,
                    mode=mode,
                )
                for ex_th in exit_thresholds:
                    rows.append(_run(f"az_{mode}_floor{floor:.2f}_vf{vf:.2f}_exit{ex_th:.2f}", eval_df, policy, az_exit_model, entry_cfg, risk_cfg, {"exit_threshold": ex_th, "min_exit_age": exit_cfg["min_exit_age"]}, (eval_feat, dec, eval_close, eval_fill), fee=args.fee, slip=args.slip))

    ranked_pnl = sorted(rows, key=lambda r: float(r["eval"]["pnl"]), reverse=True)
    ranked_score = sorted(rows, key=lambda r: float(r["score"]), reverse=True)
    chosen = []
    for row in ranked_pnl[:3] + ranked_score[:3] + [next(r for r in rows if r["name"] == "baseline_hf_no_limit")]:
        if row["name"] not in {r["name"] for r in chosen}:
            chosen.append(row)

    def reconstruct(name: str) -> tuple[Any, dict[str, Any], pd.DataFrame]:
        if name == "baseline_hf_no_limit":
            return base_exit_model, exit_cfg, eval_dec
        if name.startswith("az_entry_floor"):
            floor = float(name.split("floor", 1)[1].split("_", 1)[0])
            dec = _entry_modified_decisions(eval_dec, entry_probs, min_keep_prob=floor)
            if "_exit" in name:
                th = float(name.rsplit("_exit", 1)[1])
                return az_exit_model, {"exit_threshold": th, "min_exit_age": exit_cfg["min_exit_age"]}, dec
            return base_exit_model, exit_cfg, dec
        parts = name.split("_")
        mode = parts[1]
        floor = float(name.split("_floor", 1)[1].split("_", 1)[0])
        vf = float(name.split("_vf", 1)[1].split("_", 1)[0])
        th = float(name.rsplit("_exit", 1)[1])
        dec = _value_adjusted_decisions(eval_dec, entry_probs, entry_values, min_keep_prob=floor, value_floor=vf, value_ceiling=0.55, mode=mode)
        return az_exit_model, {"exit_threshold": th, "min_exit_age": exit_cfg["min_exit_age"]}, dec

    selected_detail = []
    cost_stress: dict[str, list[dict[str, Any]]] = {}
    for row in chosen:
        exit_model, cfg, dec = reconstruct(row["name"])
        selected_detail.append(_run(row["name"], eval_df, policy, exit_model, entry_cfg, risk_cfg, cfg, (eval_feat, dec, eval_close, eval_fill), fee=args.fee, slip=args.slip, include_monthly=True))
    for mult in (1.0, 2.0, 3.0):
        cost_stress[f"cost_{mult:g}x"] = []
        for row in chosen:
            exit_model, cfg, dec = reconstruct(row["name"])
            cost_stress[f"cost_{mult:g}x"].append(_run(row["name"], eval_df, policy, exit_model, entry_cfg, risk_cfg, cfg, (eval_feat, dec, eval_close, eval_fill), fee=args.fee * mult, slip=args.slip * mult))

    report = {
        "type": "alphazero_style_risk_variants_2026",
        "note": "Post-training search using the AlphaZero-style value head as a causal risk/exposure modifier. No 2026 future returns are used for inference.",
        "model": str(args.model),
        "audit": _audit(args.train_csv, args.eval_csv, policy),
        "entry_value_quantiles": np.quantile(entry_values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).round(6).tolist(),
        "entry_conf_quantiles": np.quantile(entry_probs.max(axis=1), [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).round(6).tolist(),
        "ranked_by_pnl": ranked_pnl[:30],
        "ranked_by_score": ranked_score[:30],
        "selected_detail": selected_detail,
        "cost_stress": cost_stress,
        "decision": {
            "best_pnl_name": ranked_pnl[0]["name"],
            "best_pnl": ranked_pnl[0]["eval"]["pnl"],
            "best_score_name": ranked_score[0]["name"],
            "best_score": ranked_score[0]["score"],
            "baseline_pnl": next(r for r in rows if r["name"] == "baseline_hf_no_limit")["eval"]["pnl"],
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "decision": report["decision"], "top_pnl": ranked_pnl[:5], "top_score": ranked_score[:5]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
