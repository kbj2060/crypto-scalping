#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    _combine_primary_fallback,
    _load_best_scale_runtime,
    _predict_scaled,
    _read,
)
from scripts.rebuild_alpha7_v2_only_high_turnover_20260526 import _rename_clean4_v2  # noqa: E402
from scripts.sweep_alpha8_origin_scaled_combo_20260529 import OfficialCost3  # noqa: E402
from scripts.train_eval_alpha7_directional_dsac_router_20260529 import (  # noqa: E402
    EVAL_CSV,
    FORBIDDEN_PREFIXES,
    TRAIN_CSV,
    _apply_norm,
    _fit_norm,
)
from scripts.train_eval_alpha8_dsac_iqn_risk_selector_20260529 import _state_frame  # noqa: E402
from scripts.train_eval_alpha8_primary_autoreg_dsac_risk_20260529 import (  # noqa: E402
    ACTION_DIM,
    TP_BUCKETS,
    SL_BUCKETS,
    HOLD_BUCKETS,
    MULT_BUCKETS,
    CAP_BUCKETS,
    _build_dataset,
    _compose_decisions,
    _fixed_action_id,
    _fixed_decisions,
    _policy_actions,
    _score,
    _seed_everything,
    _train_autoreg_dsac,
    _usage,
    _active,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha8_highcap_autoreg_dsac_risk_20260529"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


def _assert_clean(df: pd.DataFrame, *, name: str) -> None:
    bad = [c for c in df.columns if str(c).startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise RuntimeError(f"{name} contains forbidden legacy regime columns: {bad[:20]}")


def _metrics_rows(evaluator: OfficialCost3, splits: list[tuple[str, pd.DataFrame, dict[str, pd.DataFrame]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, frame, variants in splits:
        for variant, dec in variants.items():
            rows.append({"split": split, "variant": variant, **evaluator(frame, dec)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--bc-coef", type=float, default=0.08)
    ap.add_argument("--samples-per-row", type=int, default=96)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    _seed_everything(310529)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")

    baseline = get_live_baseline()
    train_all = _rename_clean4_v2(_read(TRAIN_CSV))
    eval_df = _rename_clean4_v2(_read(EVAL_CSV))
    _assert_clean(train_all, name="train_all")
    _assert_clean(eval_df, name="eval")
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    primary = joblib.load(baseline.primary_parent)
    fallback = joblib.load(baseline.fallback_parent)
    primary_rt = _load_best_scale_runtime(baseline.primary_summary)
    fallback_rt = _load_best_scale_runtime(baseline.fallback_summary)
    p_train = _predict_scaled(primary, train_df, primary_rt).reset_index(drop=True)
    p_val = _predict_scaled(primary, val_df, primary_rt).reset_index(drop=True)
    p_eval = _predict_scaled(primary, eval_df, primary_rt).reset_index(drop=True)
    f_train = _predict_scaled(fallback, train_df, fallback_rt).reset_index(drop=True)
    f_val = _predict_scaled(fallback, val_df, fallback_rt).reset_index(drop=True)
    f_eval = _predict_scaled(fallback, eval_df, fallback_rt).reset_index(drop=True)
    combo_train = _combine_primary_fallback(p_train, f_train).reset_index(drop=True)
    combo_val = _combine_primary_fallback(p_val, f_val).reset_index(drop=True)
    combo_eval = _combine_primary_fallback(p_eval, f_eval).reset_index(drop=True)

    s_train = _state_frame(train_df, p_train, f_train, combo_train)
    s_val = _state_frame(val_df, p_val, f_val, combo_val)
    s_eval = _state_frame(eval_df, p_eval, f_eval, combo_eval)
    norm = _fit_norm(s_train)
    x_train = _apply_norm(s_train, norm)
    x_val = _apply_norm(s_val, norm)
    x_eval = _apply_norm(s_eval, norm)

    evaluator = OfficialCost3()
    dataset, data_diag = _build_dataset(
        train_df,
        x_train,
        combo_train,
        fee=float(evaluator.fee),
        slip=float(evaluator.slip),
        cost_mult=float(args.cost_mult),
        samples_per_row=int(args.samples_per_row),
    )
    print(
        json.dumps(
            {
                "stage": "train_start",
                "model_id": MODEL_ID,
                "device": str(device),
                "state_dim": int(x_train.shape[1]),
                "action_dim": int(ACTION_DIM),
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "oos_rows": int(len(eval_df)),
                "data_diag": data_diag,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    actor, train_diag = _train_autoreg_dsac(
        dataset,
        state_dim=int(x_train.shape[1]),
        device=device,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        bc_coef=float(args.bc_coef),
    )
    a_train = _policy_actions(actor, x_train, device=device)
    a_val = _policy_actions(actor, x_val, device=device)
    a_eval = _policy_actions(actor, x_eval, device=device)

    dsac_train = _compose_decisions(combo_train, a_train)
    dsac_val = _compose_decisions(combo_val, a_val)
    dsac_eval = _compose_decisions(combo_eval, a_eval)

    fixed52_id = _fixed_action_id(0.200, 5.0, 0.75, 1.10, 7.5)
    fixed54_id = _fixed_action_id(0.200, 5.0, 0.75, 1.20, 7.5)
    fixed55_id = _fixed_action_id(0.200, 5.0, 0.75, 1.20, 10.0)
    fixed60_id = _fixed_action_id(0.200, 5.0, 0.75, 1.75, 7.5)

    variants = {
        "baseline_combo": (combo_train, combo_val, combo_eval),
        "fixed_52_highwr": (_fixed_decisions(combo_train, fixed52_id), _fixed_decisions(combo_val, fixed52_id), _fixed_decisions(combo_eval, fixed52_id)),
        "fixed_54_highcap": (_fixed_decisions(combo_train, fixed54_id), _fixed_decisions(combo_val, fixed54_id), _fixed_decisions(combo_eval, fixed54_id)),
        "fixed_55_highcap": (_fixed_decisions(combo_train, fixed55_id), _fixed_decisions(combo_val, fixed55_id), _fixed_decisions(combo_eval, fixed55_id)),
        "fixed_60_aggressive": (_fixed_decisions(combo_train, fixed60_id), _fixed_decisions(combo_val, fixed60_id), _fixed_decisions(combo_eval, fixed60_id)),
        "autoreg_dsac": (dsac_train, dsac_val, dsac_eval),
    }
    grid = _metrics_rows(
        evaluator,
        [
            ("train", train_df, {k: v[0] for k, v in variants.items()}),
            ("val", val_df, {k: v[1] for k, v in variants.items()}),
            ("oos", eval_df, {k: v[2] for k, v in variants.items()}),
        ],
    )
    grid["selection_score"] = grid.apply(_score, axis=1)
    grid_path = OUT_DIR / "grid.csv"
    grid.to_csv(grid_path, index=False)

    val_rank = grid[(grid["split"] == "val") & (grid["variant"] != "baseline_combo")].sort_values("selection_score", ascending=False)
    selected_variant = str(val_rank.iloc[0]["variant"])
    selected_oos = grid[(grid["split"] == "oos") & (grid["variant"] == selected_variant)].iloc[0].to_dict()
    fixed54_oos = grid[(grid["split"] == "oos") & (grid["variant"] == "fixed_54_highcap")].iloc[0].to_dict()

    model_path = OUT_DIR / "alpha8_highcap_autoreg_dsac_risk.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "selected_variant": selected_variant,
            "state_dim": int(x_train.shape[1]),
            "action_dim": int(ACTION_DIM),
            "state_columns": list(norm["columns"]),
            "state_normalizer": norm,
            "buckets": {
                "tp": TP_BUCKETS,
                "sl": SL_BUCKETS,
                "hold": HOLD_BUCKETS,
                "mult": MULT_BUCKETS,
                "cap": CAP_BUCKETS,
            },
            "actor_state_dict": actor.state_dict(),
        },
        model_path,
    )
    summary = {
        "model_id": MODEL_ID,
        "design": "Reproduces alpha8_wr50_high_cap_research 54/55 context: Alpha7 primary+fallback combo owns direction; Autoregressive DSAC replaces only TP/SL/hold/mult/cap risk bucket heads.",
        "live_wired": False,
        "selection_basis": "2025Q4 validation official Cost3 score; 2026 OOS is reported only.",
        "baseline_model_id": baseline.model_id,
        "allowed_regime_surfaces": ["clean_regime4_state24_sticky090_v2_*", "regime4_pred_*"],
        "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
        "forbidden_prefix_count": 0,
        "fixed_54_highcap_bucket": {"tp": 0.2, "sl": 5.0, "hold": 0.75, "mult": 1.2, "cap": 7.5},
        "training": {
            "device": str(device),
            "state_dim": int(x_train.shape[1]),
            "action_dim": int(ACTION_DIM),
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "bc_coef": float(args.bc_coef),
            "samples_per_row": int(args.samples_per_row),
            "reward_label": "full_trade_net_pnl_after_cost",
            "reward_accounting": "qty=entry_notional/entry_fill; exit_notional=qty*exit_fill; net=gross_pnl-entry_fee-exit_fee",
            "dataset_diagnostics": data_diag,
            "train_diag": train_diag,
            "action_usage": {
                "train": _usage(a_train, _active(combo_train)),
                "val": _usage(a_val, _active(combo_val)),
                "oos": _usage(a_eval, _active(combo_eval)),
            },
        },
        "selected": {
            "variant": selected_variant,
            "val": grid[(grid["split"] == "val") & (grid["variant"] == selected_variant)].iloc[0].to_dict(),
            "oos": selected_oos,
            "delta_vs_fixed_54_highcap_oos_pnl": float(selected_oos["pnl"]) - float(fixed54_oos["pnl"]),
        },
        "fixed_54_highcap_oos": fixed54_oos,
        "artifacts": {
            "summary": str(OUT_DIR / "summary.json"),
            "grid": str(grid_path),
            "model": str(model_path),
        },
        "audit": {
            "feature_contract_fail_fast": True,
            "legacy_compat_alias": False,
            "selection_uses_2026": False,
            "official_accounting": "OfficialCost3",
        },
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    print(json.dumps({"summary": str(summary_path), "selected": summary["selected"], "fixed_54_highcap_oos": fixed54_oos}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
