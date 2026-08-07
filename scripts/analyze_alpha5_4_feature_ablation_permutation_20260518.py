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

from ensemble.dueling_dqn_per_full_architecture import ActionSpace, ConditionedDQNTrainer, ConditionedDQNTrainerConfig  # noqa: E402
from scripts.train_eval_alpha5_3_hmm_dqn_router_parent_20260517 import CLEAN4_PREFIX, DEFAULT_EVAL, DEFAULT_TRAIN, ROUTER_COLS  # noqa: E402
from scripts.train_eval_alpha5_4_single_conditioned_dqn_20260518 import _metrics, _regime_matrix, _transform_market  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default, _read  # noqa: E402


DEFAULT_MODEL_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_4_single_conditioned_dqn_loop16_evalfix_h045_gate100_20260518"
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/alpha5_4_feature_ablation_permutation_20260518.json"


def _load_trainer(model_dir: Path, device: str | None = None) -> ConditionedDQNTrainer:
    ckpt_path = model_dir / "single_conditioned_dqn_best.pt"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "single_conditioned_dqn_agent.pt"
    ckpt = torch.load(ckpt_path, map_location=device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = ConditionedDQNTrainerConfig(**ckpt["cfg"])
    trainer = ConditionedDQNTrainer(cfg, device=device)
    trainer.online.load_state_dict(ckpt["model_state_dict"])
    trainer.target.load_state_dict(ckpt.get("target_state_dict", ckpt["model_state_dict"]))
    trainer.online.eval()
    trainer.target.eval()
    return trainer


def _groups(cols: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[str]] = {
        "future_regime_pred": [c for c in cols if c.startswith("regime4_pred_")],
        "clean4_factor_core": [
            c
            for c in cols
            if c
            in {
                f"{CLEAN4_PREFIX}factor_trend",
                f"{CLEAN4_PREFIX}factor_vol",
                f"{CLEAN4_PREFIX}factor_flow",
                f"{CLEAN4_PREFIX}factor_liquidity",
                f"{CLEAN4_PREFIX}factor_crowding",
                f"{CLEAN4_PREFIX}trend_bias",
                f"{CLEAN4_PREFIX}directional_bias",
            }
        ],
        "clean4_semantic_probs": [
            c
            for c in cols
            if c
            in {
                f"{CLEAN4_PREFIX}bull_prob",
                f"{CLEAN4_PREFIX}bear_prob",
                f"{CLEAN4_PREFIX}chop_prob",
                f"{CLEAN4_PREFIX}whipsaw_prob",
                f"{CLEAN4_PREFIX}trend_prob",
                f"{CLEAN4_PREFIX}micro_prob",
                f"{CLEAN4_PREFIX}range_prob",
            }
        ],
        "clean4_conf_uncertainty": [
            c
            for c in cols
            if c
            in {
                f"{CLEAN4_PREFIX}confidence",
                f"{CLEAN4_PREFIX}entropy",
                f"{CLEAN4_PREFIX}margin",
                f"{CLEAN4_PREFIX}instability_prob",
            }
        ],
        "clean4_risk_transition": [
            c
            for c in cols
            if c
            in {
                f"{CLEAN4_PREFIX}risk_off_prob",
                f"{CLEAN4_PREFIX}transition_risk",
            }
        ],
        "ai_m7": [
            c
            for c in cols
            if c.startswith("ai_")
            or c.startswith("m7_")
            or c.startswith("patchtst_")
            or c.startswith("dlinear_")
            or c.startswith("tide_")
        ],
        "trend_momentum": [
            c
            for c in cols
            if any(k in c for k in ("trend", "mom", "rsi", "breakout", "squeeze"))
            and not c.startswith(CLEAN4_PREFIX)
            and not c.startswith("regime4_pred_")
        ],
        "flow_liquidity_crowding": [
            c
            for c in cols
            if any(k in c for k in ("flow", "liquidity", "taker", "whale", "crowding", "funding", "trade_intensity", "ofi"))
            and not c.startswith(CLEAN4_PREFIX)
            and not c.startswith("regime4_pred_")
        ],
        "volatility_risk": [
            c
            for c in cols
            if any(k in c for k in ("vol", "risk", "jump", "evt", "garch", "rogers", "amihud"))
            and not c.startswith(CLEAN4_PREFIX)
            and not c.startswith("regime4_pred_")
        ],
        "tp_sl_action_score": [c for c in cols if c == "tp_sl_action_score"],
    }
    return {name: [cols.index(c) for c in members if c in cols] for name, members in groups.items()}


def _compact(metrics: dict[str, Any]) -> dict[str, float]:
    return {k: float(metrics[k]) for k in ("pnl", "mdd", "trades_per_day", "wr")}


def _score_delta(base: dict[str, Any], changed: dict[str, Any]) -> dict[str, float]:
    return {
        "delta_pnl": float(changed["pnl"] - base["pnl"]),
        "delta_mdd": float(changed["mdd"] - base["mdd"]),
        "delta_trades_per_day": float(changed["trades_per_day"] - base["trades_per_day"]),
        "delta_wr": float(changed["wr"] - base["wr"]),
        "impact_pnl_loss": float(base["pnl"] - changed["pnl"]),
    }


def _evaluate(
    df: pd.DataFrame,
    market: np.ndarray,
    regime: np.ndarray,
    trainer: ConditionedDQNTrainer,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    return _metrics(
        df,
        market,
        regime,
        trainer,
        fee=float(cfg["fee"]),
        slip=float(cfg["slip"]),
        unit_exposure=float(cfg["unit_exposure"]),
        min_hold_bars=int(cfg["min_hold_bars"]),
        hard_min_hold_bars=int(cfg["hard_min_hold_bars"]),
        max_hold_bars=int(cfg["max_hold_bars"]),
        entry_edge_threshold=float(cfg["entry_edge_threshold"]),
        name=str(cfg.get("name", "importance")),
        log_every=0,
    )


def _permutation_matrix(market: np.ndarray, indices: list[int], rng: np.random.Generator) -> np.ndarray:
    mutated = market.copy()
    for idx in indices:
        mutated[:, idx] = mutated[rng.permutation(len(mutated)), idx]
    return mutated


def _zero_matrix(market: np.ndarray, indices: list[int]) -> np.ndarray:
    mutated = market.copy()
    mutated[:, indices] = 0.0
    return mutated


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alpha5.4 feature group ablation and permutation analysis.")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--split", choices=("oos", "selection"), default="oos")
    p.add_argument("--seed", type=int, default=5418)
    p.add_argument("--top-individual", type=int, default=25)
    p.add_argument("--cost-mode", choices=("cost1", "all"), default="cost1")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    trainer = _load_trainer(args.model_dir)
    scaler_pack = joblib.load(args.model_dir / "single_conditioned_dqn_scaler.joblib")
    market_cols = list(scaler_pack["market_cols"])
    scaler = scaler_pack["scaler"]
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    if args.split == "selection":
        frame = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    else:
        frame = eval_df.reset_index(drop=True)
    market = _transform_market(frame, market_cols, scaler)
    regime = _regime_matrix(frame)
    summary = json.loads((args.model_dir / "alpha5_4_single_conditioned_dqn_summary.json").read_text())
    cfg = {
        "fee": 0.0005,
        "slip": 0.0002,
        "unit_exposure": float(summary["config"]["unit_exposure"]),
        "min_hold_bars": int(summary["config"]["min_hold_bars"]),
        "hard_min_hold_bars": int(summary["config"]["hard_min_hold_bars"]),
        "max_hold_bars": int(summary["config"]["max_hold_bars"]),
        "entry_edge_threshold": float(summary["config"]["entry_edge_threshold"]),
        "name": args.split,
    }
    baseline_all = _evaluate(frame, market, regime, trainer, cfg)
    baseline = baseline_all["cost1"]
    group_indices = _groups(market_cols)
    group_rows = []
    for name, indices in group_indices.items():
        if not indices:
            continue
        ablated = _evaluate(frame, _zero_matrix(market, indices), regime, trainer, cfg)["cost1"]
        permuted = _evaluate(frame, _permutation_matrix(market, indices, rng), regime, trainer, cfg)["cost1"]
        group_rows.append(
            {
                "group": name,
                "n_features": len(indices),
                "features": [market_cols[i] for i in indices],
                "ablation": {**_compact(ablated), **_score_delta(baseline, ablated)},
                "permutation": {**_compact(permuted), **_score_delta(baseline, permuted)},
            }
        )
    group_rows.sort(key=lambda row: max(row["ablation"]["impact_pnl_loss"], row["permutation"]["impact_pnl_loss"]), reverse=True)

    individual_candidates = sorted(
        {idx for indices in group_indices.values() for idx in indices}
        | set(range(min(len(market_cols), int(args.top_individual)))),
    )
    individual_rows = []
    for idx in individual_candidates:
        name = market_cols[idx]
        permuted = _evaluate(frame, _permutation_matrix(market, [idx], rng), regime, trainer, cfg)["cost1"]
        individual_rows.append({"feature": name, "permutation": {**_compact(permuted), **_score_delta(baseline, permuted)}})
    individual_rows.sort(key=lambda row: row["permutation"]["impact_pnl_loss"], reverse=True)

    report = {
        "model_dir": str(args.model_dir),
        "split": args.split,
        "market_feature_count": len(market_cols),
        "baseline_cost1": _compact(baseline),
        "baseline_all_costs": {k: _compact(v) for k, v in baseline_all.items()},
        "group_importance": group_rows,
        "individual_permutation_importance": individual_rows,
        "notes": {
            "ablation": "Selected normalized columns are set to 0.0, i.e. train mean after z-score scaling.",
            "permutation": "Selected normalized columns are shuffled independently across the evaluated split.",
            "impact_pnl_loss": "Positive means removing/shuffling the feature made PnL worse; negative means it improved PnL.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"stage": "complete", "out": str(args.out), "baseline_cost1": report["baseline_cost1"], "top_groups": group_rows[:5], "top_features": individual_rows[:10]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
