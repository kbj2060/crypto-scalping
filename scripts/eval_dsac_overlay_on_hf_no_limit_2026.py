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

from ensemble.fully_learned_governor_policy import ACTION_CASH  # noqa: E402
from ensemble.train_rl_dsac_agent import DSACRouter, DSAC_STATE_DIM, GaussianActor  # noqa: E402
from scripts.train_eval_hf_no_limit_exit_governor import (  # noqa: E402
    DEFAULT_EVAL_CSV,
    DEFAULT_POLICY,
    DEFAULT_TRAIN_CSV,
    _base_frame,
    _compact,
    backtest_no_limit_exit,
)
from scripts.eval_hf_entry_overlay_grid import _audit  # noqa: E402


DEFAULT_EXIT_BUNDLE = ROOT / "data/ensemble/supervised/hf_entry_grid/hf_no_limit_exit_governor_fast.pkl"
DEFAULT_SELECTION = ROOT / "data/ensemble/reports/hf_no_limit_exit_final_selection_2026.json"
DEFAULT_DSAC_CKPT = ROOT / "data/ensemble/ckpt/best_dsac_agents.pth"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/dsac_overlay_on_hf_no_limit_2026.json"


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return x


def _load_actor(path: Path, device: str) -> tuple[GaussianActor, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dim = int(ckpt.get("state_dim", DSAC_STATE_DIM) or DSAC_STATE_DIM)
    actor = GaussianActor(state_dim=state_dim).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, ckpt


def _numeric_feature_rows(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    cols = [c for c in df.columns if c != "timestamp"]
    vals = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return cols, vals.to_numpy(dtype=np.float64, copy=False)


def _flat_dsac_signals(df: pd.DataFrame, ckpt_path: Path, device: str) -> pd.DataFrame:
    actor, ckpt = _load_actor(ckpt_path, device)
    router = DSACRouter(actor, device=device)
    cols, vals = _numeric_feature_rows(df)
    actions: list[int] = []
    sides: list[int] = []
    scores: list[float] = []
    raw_actions: list[float] = []
    for row in vals:
        features = {k: float(v) for k, v in zip(cols, row)}
        action, _, info = router.decide(features, {"type": None, "entry_price": 0.0, "unrealized": 0.0, "mdd": 0.0, "hold_count": 0.0})
        raw = _safe_float((info or {}).get("raw_action", 0.0), 0.0)
        actions.append(int(action))
        sides.append(1 if int(action) == 1 else (-1 if int(action) == 2 else 0))
        scores.append(abs(raw))
        raw_actions.append(raw)
    return pd.DataFrame(
        {
            "dsac_action": actions,
            "dsac_side": sides,
            "dsac_score": scores,
            "dsac_raw_action": raw_actions,
            "ckpt_epoch": int(ckpt.get("epoch", -1) or -1),
            "ckpt_best_score": float(ckpt.get("best_score", np.nan)) if "best_score" in ckpt else np.nan,
            "ckpt_state_dim": int(ckpt.get("state_dim", DSAC_STATE_DIM) or DSAC_STATE_DIM),
        }
    )


def _load_selected_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    sel = obj.get("selected_balanced") or {}
    entry_cfg = dict(sel.get("entry_config") or {})
    risk_cfg = dict(sel.get("risk_config") or {})
    exit_cfg = dict(sel.get("exit_config") or {})
    if not entry_cfg or not risk_cfg or not exit_cfg:
        raise ValueError(f"missing selected_balanced config in {path}")
    return entry_cfg, risk_cfg, exit_cfg


def _overlay_decisions(base_decisions: pd.DataFrame, dsac: pd.DataFrame, mode: str, score_threshold: float) -> tuple[pd.DataFrame, dict[str, int]]:
    dec = base_decisions.copy()
    hf_side = dec["side"].astype(int).to_numpy()
    hf_action = dec["action"].astype(int).to_numpy()
    dsac_side = dsac["dsac_side"].astype(int).to_numpy()
    dsac_score = pd.to_numeric(dsac["dsac_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    active = (hf_action != ACTION_CASH) & (hf_side != 0)
    strong = dsac_score >= float(score_threshold)
    same = active & (dsac_side == hf_side)
    opposite = active & strong & (dsac_side == -hf_side)
    weak_or_cash = active & ((dsac_side == 0) | ~strong)
    block_mask = np.zeros(len(dec), dtype=bool)
    scale_mask = np.zeros(len(dec), dtype=bool)

    if mode == "none":
        pass
    elif mode == "confirm_same":
        block_mask = active & ~same
    elif mode == "veto_opposite":
        block_mask = opposite
    elif mode == "veto_opposite_or_cash":
        block_mask = opposite | weak_or_cash
    elif mode == "half_if_not_same":
        scale_mask = active & ~same
    elif mode == "half_if_opposite":
        scale_mask = opposite
    elif mode == "contrarian_block_same":
        block_mask = same
    else:
        raise ValueError(f"unknown overlay mode: {mode}")

    if block_mask.any():
        dec.loc[block_mask, "action"] = ACTION_CASH
        dec.loc[block_mask, "side"] = 0
        dec.loc[block_mask, "notional_exposure"] = 0.0
    if scale_mask.any():
        dec.loc[scale_mask, "notional_exposure"] = pd.to_numeric(dec.loc[scale_mask, "notional_exposure"], errors="coerce").fillna(0.0) * 0.5

    meta = {
        "hf_active": int(active.sum()),
        "dsac_same": int(same.sum()),
        "dsac_opposite_strong": int(opposite.sum()),
        "dsac_weak_or_cash": int(weak_or_cash.sum()),
        "blocked": int(block_mask.sum()),
        "scaled": int(scale_mask.sum()),
    }
    return dec, meta


def _slice_precomputed(
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    mask: pd.Series | np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    base_feat, decisions, close, fill_px = precomputed
    return (
        base_feat.iloc[idx].reset_index(drop=True),
        decisions.iloc[idx].reset_index(drop=True),
        close[idx],
        fill_px[idx],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DSAC as an entry overlay on the HF no-limit exit governor.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--exit-bundle", type=Path, default=DEFAULT_EXIT_BUNDLE)
    p.add_argument("--selection-report", type=Path, default=DEFAULT_SELECTION)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--dsac-ckpt", type=Path, default=DEFAULT_DSAC_CKPT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    policy = joblib.load(args.policy)
    bundle = joblib.load(args.exit_bundle)
    exit_model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    entry_cfg, risk_cfg, exit_cfg = _load_selected_config(args.selection_report)
    eval_df = _read(args.eval_csv)
    precomputed = _base_frame(eval_df, policy, entry_cfg)
    base_feat, base_decisions, close, fill_px = precomputed
    dsac = _flat_dsac_signals(eval_df, args.dsac_ckpt, args.device)

    rows: list[dict[str, Any]] = []
    modes = [
        "none",
        "veto_opposite",
        "half_if_opposite",
        "half_if_not_same",
        "confirm_same",
        "veto_opposite_or_cash",
        "contrarian_block_same",
    ]
    thresholds = [0.00, 0.10, 0.15, 0.20, 0.30]
    for mode in modes:
        ths = [0.0] if mode in {"none", "confirm_same", "contrarian_block_same"} else thresholds
        for th in ths:
            dec, overlay_meta = _overlay_decisions(base_decisions, dsac, mode, th)
            bt = backtest_no_limit_exit(
                eval_df,
                policy,
                exit_model,
                entry_config=entry_cfg,
                risk_config=risk_cfg,
                exit_threshold=float(exit_cfg["exit_threshold"]),
                min_exit_age=int(exit_cfg["min_exit_age"]),
                fee=float(args.fee),
                slip=float(args.slip),
                precomputed=(base_feat, dec, close, fill_px),
            )
            rows.append({"name": f"{mode}_score{th:.2f}", "mode": mode, "score_threshold": th, "overlay": overlay_meta, "eval": _compact(bt)})

    ranked = sorted(rows, key=lambda r: float(r["eval"].get("pnl") or -1e18), reverse=True)
    baseline = next(r for r in rows if r["mode"] == "none")
    candidates = [baseline]
    for row in ranked:
        if row["name"] not in {c["name"] for c in candidates}:
            candidates.append(row)
        if len(candidates) >= 6:
            break

    monthly: list[dict[str, Any]] = []
    if "timestamp" in eval_df.columns:
        jan_mask = eval_df["timestamp"] < pd.Timestamp("2026-02-01")
        feb_mask = eval_df["timestamp"] >= pd.Timestamp("2026-02-01")
        for cand in candidates:
            dec, overlay_meta = _overlay_decisions(base_decisions, dsac, cand["mode"], float(cand["score_threshold"]))
            cand_precomputed = (base_feat, dec, close, fill_px)
            jan_df = eval_df.loc[jan_mask].reset_index(drop=True)
            feb_df = eval_df.loc[feb_mask].reset_index(drop=True)
            jan_bt = backtest_no_limit_exit(
                jan_df,
                policy,
                exit_model,
                entry_config=entry_cfg,
                risk_config=risk_cfg,
                exit_threshold=float(exit_cfg["exit_threshold"]),
                min_exit_age=int(exit_cfg["min_exit_age"]),
                fee=float(args.fee),
                slip=float(args.slip),
                precomputed=_slice_precomputed(cand_precomputed, jan_mask),
            )
            feb_bt = backtest_no_limit_exit(
                feb_df,
                policy,
                exit_model,
                entry_config=entry_cfg,
                risk_config=risk_cfg,
                exit_threshold=float(exit_cfg["exit_threshold"]),
                min_exit_age=int(exit_cfg["min_exit_age"]),
                fee=float(args.fee),
                slip=float(args.slip),
                precomputed=_slice_precomputed(cand_precomputed, feb_mask),
            )
            monthly.append(
                {
                    "name": cand["name"],
                    "mode": cand["mode"],
                    "score_threshold": cand["score_threshold"],
                    "overlay": overlay_meta,
                    "full": cand["eval"],
                    "jan": _compact(jan_bt),
                    "feb": _compact(feb_bt),
                    "min_month_pnl": float(min(jan_bt["pnl"], feb_bt["pnl"])),
                }
            )
    monthly_balanced = sorted(monthly, key=lambda r: (float(r["min_month_pnl"]), float(r["full"]["pnl"])), reverse=True)

    cost_stress: dict[str, list[dict[str, Any]]] = {}
    for mult in (1.0, 2.0, 3.0):
        key = f"cost_{mult:g}x"
        cost_stress[key] = []
        for cand in candidates:
            dec, overlay_meta = _overlay_decisions(base_decisions, dsac, cand["mode"], float(cand["score_threshold"]))
            bt = backtest_no_limit_exit(
                eval_df,
                policy,
                exit_model,
                entry_config=entry_cfg,
                risk_config=risk_cfg,
                exit_threshold=float(exit_cfg["exit_threshold"]),
                min_exit_age=int(exit_cfg["min_exit_age"]),
                fee=float(args.fee) * mult,
                slip=float(args.slip) * mult,
                precomputed=(base_feat, dec, close, fill_px),
            )
            cost_stress[key].append({"name": cand["name"], "mode": cand["mode"], "score_threshold": cand["score_threshold"], "overlay": overlay_meta, "eval": _compact(bt)})

    report = {
        "type": "dsac_overlay_on_hf_no_limit_2026",
        "purpose": "Find whether a reproducible DSAC checkpoint can replace or improve any part of the current HF no-limit governor.",
        "policy": str(args.policy),
        "exit_bundle": str(args.exit_bundle),
        "dsac_ckpt": str(args.dsac_ckpt),
        "selected_entry_config": entry_cfg,
        "selected_risk_config": risk_cfg,
        "selected_exit_config": exit_cfg,
        "audit": _audit(args.train_csv, args.eval_csv, policy),
        "dsac_signal_distribution": {
            "rows": int(len(dsac)),
            "long": int((dsac["dsac_side"] == 1).sum()),
            "short": int((dsac["dsac_side"] == -1).sum()),
            "cash": int((dsac["dsac_side"] == 0).sum()),
            "score_mean": float(dsac["dsac_score"].mean()),
            "score_p50": float(dsac["dsac_score"].quantile(0.50)),
            "score_p90": float(dsac["dsac_score"].quantile(0.90)),
            "ckpt_epoch": int(dsac["ckpt_epoch"].iloc[0]),
            "ckpt_best_score": float(dsac["ckpt_best_score"].iloc[0]) if np.isfinite(float(dsac["ckpt_best_score"].iloc[0])) else None,
            "ckpt_state_dim": int(dsac["ckpt_state_dim"].iloc[0]),
        },
        "baseline": {"name": baseline["name"], **baseline["eval"]},
        "ranked": [{"name": r["name"], "mode": r["mode"], "score_threshold": r["score_threshold"], **r["eval"], "overlay": r["overlay"]} for r in ranked],
        "monthly_balanced": monthly_balanced,
        "cost_stress": cost_stress,
        "decision": {
            "best_name": ranked[0]["name"],
            "best_pnl": ranked[0]["eval"]["pnl"],
            "monthly_balanced_name": monthly_balanced[0]["name"] if monthly_balanced else ranked[0]["name"],
            "monthly_balanced_pnl": monthly_balanced[0]["full"]["pnl"] if monthly_balanced else ranked[0]["eval"]["pnl"],
            "monthly_balanced_min_month_pnl": monthly_balanced[0]["min_month_pnl"] if monthly_balanced else None,
            "baseline_pnl": baseline["eval"]["pnl"],
            "delta_pnl": float(ranked[0]["eval"]["pnl"] - baseline["eval"]["pnl"]),
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "baseline": report["baseline"], "top": report["ranked"][:8], "monthly_balanced": monthly_balanced[:5], "cost_stress": cost_stress, "decision": report["decision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
