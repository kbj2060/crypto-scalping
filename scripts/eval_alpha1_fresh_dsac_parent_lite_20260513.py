#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    ACTION_LONG,
    ACTION_SHORT,
    predict_policy_frame,
)
from ensemble.train_rl_dsac_agent import DSACRouter, DSAC_STATE_DIM, GaussianActor  # noqa: E402
from scripts import eval_alpha1_rl_exit_and_sizing_20260513 as alpha1  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha1_fresh_dsac_parent_lite_20260513"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_PARENT = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_margin110_20260511/v13_clean_regime_margin110.pkl"
DEFAULT_JACKPOT = ROOT / "data/ensemble/supervised/hf_v13_jackpot_runner_v21_2_20260511/v21_2_jackpot_runner.pkl"
DEFAULT_V27 = ROOT / "data/ensemble/supervised/hf_v13_deep_alpha_candidate_expansion_v27_20260511/v27_deep_alpha_candidate_expansion.pt"
DEFAULT_DSAC = ROOT / "data/ensemble/ckpt/alpha1_dsac_parent_lite_20260513/best_dsac_agents.pth"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/alpha1_fresh_dsac_parent_lite_20260513"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/alpha1_fresh_dsac_parent_lite_20260513_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/alpha1_fresh_dsac_parent_lite_20260513_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/alpha1_fresh_dsac_parent_lite_20260513_grid.csv"


@dataclass(frozen=True)
class DsacParentConfig:
    name: str
    action_th: float
    notional_scale: float
    max_entry_notional: float
    quality_floor: float


def _grid() -> list[DsacParentConfig]:
    out: list[DsacParentConfig] = []
    for th in (0.12, 0.16, 0.20, 0.24, 0.28, 0.32):
        for scale in (0.75, 1.00, 1.25):
            for cap in (2.00, 2.75):
                out.append(
                    DsacParentConfig(
                        name=f"dsac_parent_th{th:.2f}_s{scale:.2f}_cap{cap:.2f}",
                        action_th=float(th),
                        notional_scale=float(scale),
                        max_entry_notional=float(cap),
                        quality_floor=0.0,
                    )
                )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train-fresh DSAC parent-lite plugged into alpha1 stack.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=DEFAULT_JACKPOT)
    p.add_argument("--v27-model", type=Path, default=DEFAULT_V27)
    p.add_argument("--dsac-model", type=Path, default=DEFAULT_DSAC)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def _device(requested: str) -> str:
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _load_dsac_actor(path: Path, device: str) -> tuple[GaussianActor, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dim = int(ckpt.get("state_dim", DSAC_STATE_DIM))
    actor = GaussianActor(state_dim=state_dim, hidden_dim=256).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, dict(ckpt.get("meta", {})) | {
        "epoch": int(ckpt.get("epoch", -1)),
        "best_pnl": float(ckpt.get("best_pnl", np.nan)),
        "best_score": float(ckpt.get("best_score", np.nan)),
        "state_dim": state_dim,
    }


def _row_features(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k == "timestamp":
            continue
        out[str(k)] = v
    return out


def _predict_raw_actions(df: pd.DataFrame, actor: GaussianActor, device: str, batch_size: int = 8192) -> np.ndarray:
    router = DSACRouter(actor, device=device)
    states = np.zeros((len(df), DSAC_STATE_DIM), dtype=np.float32)
    for i in range(len(df)):
        states[i] = router._build_compact_state(_row_features(df.iloc[i]), {})
    raw_parts: list[np.ndarray] = []
    actor.eval()
    with torch.no_grad():
        for s in range(0, len(states), int(batch_size)):
            xb = torch.from_numpy(states[s : s + int(batch_size)]).to(device)
            raw_parts.append(actor.deterministic(xb).squeeze(-1).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(raw_parts) if raw_parts else np.zeros(0, dtype=np.float32)


def _active_defaults(base_dec: pd.DataFrame) -> dict[str, float]:
    active = (base_dec["action"].astype(int).to_numpy() != ACTION_CASH) & (base_dec["side"].astype(int).to_numpy() != 0)
    src = base_dec.loc[active].copy()

    def med(col: str, fallback: float) -> float:
        if col not in src.columns or src.empty:
            return float(fallback)
        vals = pd.to_numeric(src[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        return float(vals.median()) if len(vals) else float(fallback)

    return {
        "notional_exposure": med("notional_exposure", 0.8625),
        "leverage": med("leverage", 2.0),
        "take_profit": med("take_profit", 0.03),
        "stop_loss": med("stop_loss", 0.014),
        "max_hold_bars": med("max_hold_bars", 48.0),
        "cooldown_bars": med("cooldown_bars", 6.0),
    }


def _dsac_parent_decisions(base_dec: pd.DataFrame, raw: np.ndarray, cfg: DsacParentConfig) -> pd.DataFrame:
    out = base_dec.copy()
    defaults = _active_defaults(base_dec)
    raw = np.asarray(raw, dtype=np.float64)
    side = np.where(raw >= float(cfg.action_th), 1, np.where(raw <= -float(cfg.action_th), -1, 0)).astype(np.int64)
    active = side != 0
    abs_raw = np.clip(np.abs(raw), float(cfg.action_th), 1.0)
    strength = np.where(active, abs_raw / max(float(cfg.action_th), 1e-12), 0.0)
    strength = np.clip(strength, 1.0, 2.0)

    base_notional = pd.to_numeric(out.get("notional_exposure", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    base_leverage = pd.to_numeric(out.get("leverage", defaults["leverage"]), errors="coerce").fillna(defaults["leverage"]).to_numpy(dtype=np.float64)
    notional = np.where(base_notional > 0.0, base_notional, defaults["notional_exposure"])
    notional = np.minimum(notional * float(cfg.notional_scale) * strength, float(cfg.max_entry_notional))
    leverage = np.where(base_leverage > 0.0, base_leverage, defaults["leverage"])
    leverage = np.maximum(leverage, 1.0)

    out.loc[:, "side"] = side
    out.loc[:, "action"] = np.where(side > 0, ACTION_LONG, np.where(side < 0, ACTION_SHORT, ACTION_CASH)).astype(np.int64)
    out.loc[:, "notional_exposure"] = np.where(active, notional, 0.0)
    out.loc[:, "leverage"] = np.where(active, leverage, 1.0)
    out.loc[:, "position_fraction"] = np.where(active, out["notional_exposure"] / np.maximum(out["leverage"], 1e-12), 0.0)
    for col, fallback in (
        ("take_profit", defaults["take_profit"]),
        ("stop_loss", defaults["stop_loss"]),
        ("max_hold_bars", defaults["max_hold_bars"]),
        ("cooldown_bars", defaults["cooldown_bars"]),
    ):
        values = pd.to_numeric(out[col], errors="coerce").fillna(float(fallback)).to_numpy(dtype=np.float64)
        values = np.where(values > 0.0, values, float(fallback))
        out.loc[:, col] = np.where(active, values, 0.0)
    out.loc[:, "max_hold_bars"] = out["max_hold_bars"].round().astype(np.int64)
    out.loc[:, "cooldown_bars"] = out["cooldown_bars"].round().astype(np.int64)
    out.loc[:, "quality_score"] = np.abs(raw)
    out.loc[:, "confidence"] = np.clip(np.abs(raw), 0.0, 1.0)
    return out


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.35 * c2["pnl"] + 0.15 * c3["pnl"] - 0.25 * abs(c1["mdd"]))


def main() -> int:
    args = parse_args()
    if not args.dsac_model.exists():
        raise FileNotFoundError(f"missing fresh DSAC model: {args.dsac_model}")
    device = _device(args.device)
    print(f"[{MODEL_ID}] loading alpha1 stack and fresh DSAC actor device={device}", flush=True)
    parent = joblib.load(args.parent_model)
    jackpot_payload = joblib.load(args.jackpot_model)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = alpha1.v31._load_v27(args.v27_model)
    actor, dsac_meta = _load_dsac_actor(args.dsac_model, device)

    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    base = dict(parent["config"])
    fee = float(base.get("fee", 0.0005))
    slip = float(base.get("slip", 0.0002))

    parent_audit = _audit_contract(train_all, eval_df, list(parent.get("feature_cols") or []))
    train_dec_base = predict_policy_frame(parent, train, close=_close(train))
    val_dec_base = predict_policy_frame(parent, val, close=_close(val))
    eval_dec_base = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    print(f"[{MODEL_ID}] predicting DSAC raw actions", flush=True)
    train_raw = _predict_raw_actions(train, actor, device)
    val_raw = _predict_raw_actions(val, actor, device)
    eval_raw = _predict_raw_actions(eval_df, actor, device)
    val_q = alpha1.v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = alpha1.v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])

    grid_rows: list[dict[str, Any]] = []
    selected: DsacParentConfig | None = None
    best_score = -1e18
    for cfg in _grid():
        val_dec = _dsac_parent_decisions(val_dec_base, val_raw, cfg)
        v1 = alpha1.backtest_alpha1(val, parent, jackpot_model, add_cfg, val_q, fee=fee, slip=slip, cost_mult=1.0, decisions=val_dec)
        v2 = alpha1.backtest_alpha1(val, parent, jackpot_model, add_cfg, val_q, fee=fee, slip=slip, cost_mult=2.0, decisions=val_dec)
        v3 = alpha1.backtest_alpha1(val, parent, jackpot_model, add_cfg, val_q, fee=fee, slip=slip, cost_mult=3.0, decisions=val_dec)
        score = _score(v1, v2, v3)
        row = {
            **asdict(cfg),
            "selection_score": score,
            "val_cost1_pnl": v1["pnl"],
            "val_cost1_mdd": v1["mdd"],
            "val_cost1_trades": v1["trades"],
            "val_cost2_pnl": v2["pnl"],
            "val_cost3_pnl": v3["pnl"],
        }
        grid_rows.append(row)
        if score > best_score:
            best_score = score
            selected = cfg
    assert selected is not None
    print(f"[{MODEL_ID}] selected {selected.name}", flush=True)

    experiments: list[dict[str, Any]] = []
    for name, decisions in (
        ("alpha1_original_parent", eval_dec_base),
        (f"alpha1_fresh_dsac_parent_lite::{selected.name}", _dsac_parent_decisions(eval_dec_base, eval_raw, selected)),
    ):
        metrics = {
            f"cost{mult}": alpha1.backtest_alpha1(
                eval_df,
                parent,
                jackpot_model,
                add_cfg,
                eval_q,
                fee=fee,
                slip=slip,
                cost_mult=float(mult),
                decisions=decisions,
            )
            for mult in (1, 2, 3)
        }
        experiments.append({"name": name, "metrics": metrics, "score": _score(metrics["cost1"], metrics["cost2"], metrics["cost3"])})
        print(
            f"[{MODEL_ID}] {name} cost1={metrics['cost1']['pnl']:.2f} "
            f"mdd={metrics['cost1']['mdd']:.2f} cost2={metrics['cost2']['pnl']:.2f} cost3={metrics['cost3']['pnl']:.2f}",
            flush=True,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grid_rows).sort_values("selection_score", ascending=False).to_csv(args.grid_out, index=False)
    manifest = {
        "model_id": MODEL_ID,
        "selected_config": asdict(selected),
        "dsac_model": str(args.dsac_model),
        "dsac_meta": dsac_meta,
        "train_raw_summary": {
            "mean": float(np.mean(train_raw)),
            "std": float(np.std(train_raw)),
            "p95_abs": float(np.quantile(np.abs(train_raw), 0.95)),
        },
        "val_raw_summary": {
            "mean": float(np.mean(val_raw)),
            "std": float(np.std(val_raw)),
            "p95_abs": float(np.quantile(np.abs(val_raw), 0.95)),
        },
        "eval_raw_summary": {
            "mean": float(np.mean(eval_raw)),
            "std": float(np.std(eval_raw)),
            "p95_abs": float(np.quantile(np.abs(eval_raw), 0.95)),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    best = max(experiments, key=lambda x: x["score"])
    blocking = list(parent_audit.get("blocking", []))
    warnings = list(parent_audit.get("warnings", []))
    if best["name"] != "alpha1_original_parent" and best["metrics"]["cost1"]["pnl"] <= alpha1.ALPHA1_BASELINE["cost1"]["pnl"]:
        warnings.append("fresh_dsac_parent_lite_did_not_beat_alpha1_cost1")
    if best["metrics"]["cost2"]["pnl"] <= 0.0:
        warnings.append("best_cost2_not_survived")
    if best["metrics"]["cost3"]["pnl"] <= 0.0:
        warnings.append("best_cost3_not_survived")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and best["name"] != "alpha1_original_parent" and best["metrics"]["cost1"]["pnl"] > alpha1.ALPHA1_BASELINE["cost1"]["pnl"] else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "fresh_dsac_training_required": True,
        "alpha1_stack": "V21.2 jackpot + frozen V27 deep scout + V31 exit + deep notional 2.0",
        "selected_config": asdict(selected),
        "parent_audit": parent_audit,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Fresh DSAC parent-lite replaces only alpha1 parent entry direction. Tactical fields are inherited from the alpha1 parent distribution; DSAC action threshold, notional scale, and cap are selected on 2025Q4, then evaluated on fixed 2026 OOS.",
        "dsac_meta": dsac_meta,
        "selected_config": asdict(selected),
        "experiments": experiments,
        "audit": audit,
        "artifacts": {
            "model": str(args.dsac_model),
            "manifest": str(args.out_dir / "manifest.json"),
            "report": str(args.report_out),
            "audit": str(args.audit_out),
            "grid": str(args.grid_out),
        },
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "best": best}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
