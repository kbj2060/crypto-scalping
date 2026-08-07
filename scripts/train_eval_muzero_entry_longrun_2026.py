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

from ensemble.fully_learned_governor_policy import FEATURE_COLS  # noqa: E402
from scripts.train_eval_alphazero_style_governor_2026 import DEFAULT_MODEL_OUT as DEFAULT_AZ_MODEL  # noqa: E402
from scripts.train_eval_dsac_replacement_heads_2026 import (  # noqa: E402
    DEFAULT_EVAL_CSV,
    DEFAULT_EXIT_BUNDLE,
    DEFAULT_POLICY,
    DEFAULT_SELECTION,
    DEFAULT_TRAIN_CSV,
    _load_selected,
    _read,
)
from scripts.train_eval_hf_no_limit_exit_governor import _base_frame  # noqa: E402
from scripts.train_eval_muzero_style_governor_2026 import (  # noqa: E402
    ENTRY_ACTIONS,
    MZBundle,
    _load_az_exit,
    _make_targets,
    _planned_decisions,
    _plan_scores,
    _run_bt,
    _train_muzero,
)


DEFAULT_MODEL_OUT = ROOT / "data/ensemble/supervised/muzero_style/mz_entry_longrun_governor.pt"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/muzero_entry_longrun_governor_2026.json"


def _range(df: pd.DataFrame) -> list[str]:
    if "timestamp" not in df.columns or df.empty:
        return ["", ""]
    ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    if ts.empty:
        return ["", ""]
    return [str(ts.min()), str(ts.max())]


def _overlap(a: pd.DataFrame, b: pd.DataFrame) -> int:
    if "timestamp" not in a.columns or "timestamp" not in b.columns:
        return 0
    ta = pd.to_datetime(a["timestamp"], errors="coerce").dropna().astype("int64")
    tb = pd.to_datetime(b["timestamp"], errors="coerce").dropna().astype("int64")
    return int(len(set(ta.tolist()) & set(tb.tolist())))


def _planner_grid(mode: str) -> list[dict[str, float | int]]:
    if mode == "small":
        gammas = (0.70,)
        priors = (0.16,)
        depths = (1,)
        score_floors = (0.0,)
        conf_floors = (0.0, 0.35)
        value_floors = (-0.05, 0.05)
    else:
        gammas = (0.55, 0.70, 0.85)
        priors = (0.08, 0.16, 0.24)
        depths = (1, 2)
        score_floors = (-0.05, 0.0, 0.08, 0.16)
        conf_floors = (0.0, 0.35)
        value_floors = (-0.05, 0.05)
    return [
        {"gamma": float(g), "prior_weight": float(p), "depth": int(d), "score_floor": float(sf), "confidence_floor": float(cf), "value_floor": float(vf)}
        for g in gammas
        for p in priors
        for d in depths
        for sf in score_floors
        for cf in conf_floors
        for vf in value_floors
    ]


def _decision_name(cfg: dict[str, float | int], suffix: str) -> str:
    return (
        f"mzlr_g{float(cfg['gamma']):.2f}_p{float(cfg['prior_weight']):.2f}_d{int(cfg['depth'])}"
        f"_sf{float(cfg['score_floor']):.2f}_cf{float(cfg['confidence_floor']):.2f}_vf{float(cfg['value_floor']):.2f}_{suffix}"
    )


def _precompute_plan(
    bundle: MZBundle,
    feat: pd.DataFrame,
    *,
    device: str,
    grid: list[dict[str, float | int]],
) -> dict[tuple[float, float, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    x = feat.reindex(columns=bundle.feature_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    out = {}
    for cfg in grid:
        key = (float(cfg["gamma"]), float(cfg["prior_weight"]), int(cfg["depth"]))
        if key not in out:
            out[key] = _plan_scores(bundle, x, device=device, gamma=key[0], prior_weight=key[1], depth=key[2])
    return out


def _decisions_from_plan(
    base_dec: pd.DataFrame,
    cache: dict[tuple[float, float, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    cfg: dict[str, float | int],
) -> pd.DataFrame:
    scores, probs, vals = cache[(float(cfg["gamma"]), float(cfg["prior_weight"]), int(cfg["depth"]))]
    return _planned_decisions(
        base_dec,
        scores,
        probs,
        vals,
        score_floor=float(cfg["score_floor"]),
        confidence_floor=float(cfg["confidence_floor"]),
        value_floor=float(cfg["value_floor"]),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Long-run MuZero entry training with 2025 validation selection and 2026 final eval.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--exit-bundle", type=Path, default=DEFAULT_EXIT_BUNDLE)
    p.add_argument("--selection-report", type=Path, default=DEFAULT_SELECTION)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--az-model", type=Path, default=DEFAULT_AZ_MODEL)
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--split-date", type=str, default="2025-11-01")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--search-horizon", type=int, default=144)
    p.add_argument("--dynamics-step", type=int, default=12)
    p.add_argument("--samples", type=int, default=90000)
    p.add_argument("--temperature", type=float, default=0.012)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid-mode", choices=["small", "full"], default="full")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = "cuda" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    policy = joblib.load(args.policy)
    exit_bundle = joblib.load(args.exit_bundle)
    base_exit_model = exit_bundle["model"] if isinstance(exit_bundle, dict) and "model" in exit_bundle else exit_bundle
    az_exit_model = _load_az_exit(args.az_model, device)
    entry_cfg, risk_cfg, exit_cfg = _load_selected(args.selection_report)

    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    split_ts = pd.Timestamp(args.split_date)
    train_ts = pd.to_datetime(train_all["timestamp"], errors="coerce")
    train_core = train_all.loc[train_ts < split_ts].reset_index(drop=True)
    val_df = train_all.loc[train_ts >= split_ts].reset_index(drop=True)
    if len(train_core) < 1000 or len(val_df) < 1000:
        raise RuntimeError(f"bad train/validation split: train_core={len(train_core)} val={len(val_df)}")

    train_pre = _base_frame(train_core, policy, entry_cfg)
    train_feat, train_dec, _, _ = train_pre
    x, x_next, pi, value, reward, label_meta = _make_targets(
        train_core,
        train_dec,
        train_feat.reindex(columns=FEATURE_COLS),
        search_horizon=int(args.search_horizon),
        dynamics_step=int(args.dynamics_step),
        fee=float(args.fee),
        slip=float(args.slip),
        temperature=float(args.temperature),
        max_samples=int(args.samples),
        seed=int(args.seed),
    )
    net, mean, std, train_meta = _train_muzero(
        x,
        x_next,
        pi,
        value,
        reward,
        hidden_dim=int(args.hidden_dim),
        latent_dim=int(args.latent_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=device,
        seed=int(args.seed),
    )
    bundle = MZBundle(net, mean, std, list(FEATURE_COLS), ENTRY_ACTIONS)
    grid = _planner_grid(args.grid_mode)

    val_pre = _base_frame(val_df, policy, entry_cfg)
    val_feat, val_dec, val_close, val_fill = val_pre
    val_cache = _precompute_plan(bundle, val_feat, device=device, grid=grid)
    val_rows = [
        _run_bt("val_baseline_hf_no_limit", val_df, policy, base_exit_model, entry_cfg, risk_cfg, exit_cfg, val_pre, fee=args.fee, slip=args.slip),
    ]
    for cfg in grid:
        dec = _decisions_from_plan(val_dec, val_cache, cfg)
        val_rows.append(
            _run_bt(
                _decision_name(cfg, "azexit0.45"),
                val_df,
                policy,
                az_exit_model,
                entry_cfg,
                risk_cfg,
                {"exit_threshold": 0.45, "min_exit_age": exit_cfg["min_exit_age"]},
                (val_feat, dec, val_close, val_fill),
                fee=args.fee,
                slip=args.slip,
            )
        )
    val_ranked = sorted(val_rows, key=lambda r: float(r["eval"]["pnl"] + 3.0 * r["eval"]["mdd"]), reverse=True)
    selected = next(r for r in val_ranked if r["name"] != "val_baseline_hf_no_limit")
    selected_cfg = next(cfg for cfg in grid if _decision_name(cfg, "azexit0.45") == selected["name"])

    eval_pre = _base_frame(eval_df, policy, entry_cfg)
    eval_feat, eval_dec, eval_close, eval_fill = eval_pre
    eval_cache = _precompute_plan(bundle, eval_feat, device=device, grid=[selected_cfg])
    eval_selected_dec = _decisions_from_plan(eval_dec, eval_cache, selected_cfg)
    eval_selected_pre = (eval_feat, eval_selected_dec, eval_close, eval_fill)
    final_rows = [
        _run_bt("baseline_hf_no_limit", eval_df, policy, base_exit_model, entry_cfg, risk_cfg, exit_cfg, eval_pre, fee=args.fee, slip=args.slip, monthly=True),
        _run_bt(
            "selected_mz_longrun_entry_az_exit0.45",
            eval_df,
            policy,
            az_exit_model,
            entry_cfg,
            risk_cfg,
            {"exit_threshold": 0.45, "min_exit_age": exit_cfg["min_exit_age"]},
            eval_selected_pre,
            fee=args.fee,
            slip=args.slip,
            monthly=True,
        ),
    ]
    cost_stress: dict[str, list[dict[str, Any]]] = {}
    for mult in (1.0, 2.0, 3.0):
        cost_stress[f"cost_{mult:g}x"] = [
            _run_bt("baseline_hf_no_limit", eval_df, policy, base_exit_model, entry_cfg, risk_cfg, exit_cfg, eval_pre, fee=args.fee * mult, slip=args.slip * mult),
            _run_bt(
                "selected_mz_longrun_entry_az_exit0.45",
                eval_df,
                policy,
                az_exit_model,
                entry_cfg,
                risk_cfg,
                {"exit_threshold": 0.45, "min_exit_age": exit_cfg["min_exit_age"]},
                eval_selected_pre,
                fee=args.fee * mult,
                slip=args.slip * mult,
            ),
        ]

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "type": "muzero_entry_longrun_governor",
            "state_dict": net.state_dict(),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "feature_cols": list(FEATURE_COLS),
            "actions": list(ENTRY_ACTIONS),
            "train_meta": train_meta,
            "label_meta": label_meta,
            "selected_planner": selected_cfg,
            "selection_metric": "validation pnl + 3*mdd",
        },
        args.model_out,
    )
    report = {
        "type": "muzero_entry_longrun_governor_2026",
        "note": "Long-run MuZero-style entry training. Planner params are selected on late-2025 validation, then evaluated once on 2026 with AlphaZero-style exit fixed at 0.45.",
        "model_out": str(args.model_out),
        "audit": {
            "train_csv": str(args.train_csv),
            "eval_csv": str(args.eval_csv),
            "train_core_rows": int(len(train_core)),
            "validation_rows": int(len(val_df)),
            "eval_rows": int(len(eval_df)),
            "train_core_range": _range(train_core),
            "validation_range": _range(val_df),
            "eval_range": _range(eval_df),
            "train_validation_overlap": _overlap(train_core, val_df),
            "train_eval_overlap": _overlap(train_core, eval_df),
            "validation_eval_overlap": _overlap(val_df, eval_df),
        },
        "label_meta": label_meta,
        "train_meta": train_meta,
        "grid_mode": args.grid_mode,
        "validation_ranked_by_score": val_ranked[:30],
        "selected_validation": selected,
        "selected_planner": selected_cfg,
        "final_2026": final_rows,
        "cost_stress": cost_stress,
        "decision": {
            "selected_validation_name": selected["name"],
            "selected_validation_pnl": selected["eval"]["pnl"],
            "selected_2026_pnl": next(r for r in final_rows if r["name"] == "selected_mz_longrun_entry_az_exit0.45")["eval"]["pnl"],
            "selected_2026_mdd": next(r for r in final_rows if r["name"] == "selected_mz_longrun_entry_az_exit0.45")["eval"]["mdd"],
            "baseline_2026_pnl": next(r for r in final_rows if r["name"] == "baseline_hf_no_limit")["eval"]["pnl"],
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "model": str(args.model_out), "decision": report["decision"], "selected_planner": selected_cfg}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
