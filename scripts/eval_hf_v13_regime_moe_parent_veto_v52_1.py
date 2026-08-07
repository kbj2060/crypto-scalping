#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, ACTION_SHORT, FullyLearnedGovernorConfig, build_training_set, predict_policy_frame  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import eval_hf_v13_v31_conservative_limit_sniper_v46 as v46  # noqa: E402
from scripts.eval_hf_v13_regime_moe_action_quality_v52 import (  # noqa: E402
    DEFAULT_EVAL,
    DEFAULT_JACKPOT,
    DEFAULT_PARENT,
    DEFAULT_TRAIN,
    DEFAULT_V27,
    _feature_audit,
    _grid as _moe_grid,
    _load_pickle,
    _score,
    _train_moe_bundle,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "hf_v13_regime_moe_parent_veto_v52_1_20260513"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_regime_moe_parent_veto_v52_1_20260513"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_regime_moe_parent_veto_v52_1_20260513_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_regime_moe_parent_veto_v52_1_20260513_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_regime_moe_parent_veto_v52_1_20260513_grid.csv"


@dataclass(frozen=True)
class VetoConfig:
    name: str
    moe_name: str
    same_min: float
    opp_max: float
    cash_max: float


def _veto_grid(moe_names: list[str]) -> list[VetoConfig]:
    out: list[VetoConfig] = []
    for moe_name in moe_names:
        for same_min in (0.30, 0.36, 0.42):
            for opp_max in (0.38, 0.46):
                out.append(VetoConfig(f"{moe_name}_same{same_min:.2f}_opp{opp_max:.2f}", moe_name, same_min, opp_max, 0.62))
        out.append(VetoConfig(f"{moe_name}_loose_cash", moe_name, 0.28, 0.52, 0.70))
    return out


def _apply_veto(base_dec: pd.DataFrame, moe_bundle: dict[str, Any], frame: pd.DataFrame, cfg: VetoConfig) -> pd.DataFrame:
    out = base_dec.copy()
    feature_cols = list(moe_bundle.get("feature_cols") or [])
    x = frame.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan).copy()
    if "side_hint" in x.columns:
        x["side_hint"] = 0.0
    proba = moe_bundle["action_model"].predict_proba(x)
    classes = np.asarray(moe_bundle["action_model"].classes_, dtype=int)
    idx = {int(c): j for j, c in enumerate(classes)}
    p_cash = proba[:, idx.get(ACTION_CASH, 0)]
    p_long = proba[:, idx.get(ACTION_LONG, 0)] if ACTION_LONG in idx else np.zeros(len(out))
    p_short = proba[:, idx.get(ACTION_SHORT, 0)] if ACTION_SHORT in idx else np.zeros(len(out))
    base_side = out["side"].to_numpy(dtype=int)
    same = np.where(base_side > 0, p_long, np.where(base_side < 0, p_short, 1.0))
    opp = np.where(base_side > 0, p_short, np.where(base_side < 0, p_long, 0.0))
    trade = base_side != 0
    veto = trade & ((same < float(cfg.same_min)) | (opp > float(cfg.opp_max)) | (p_cash > float(cfg.cash_max)))
    out.loc[veto, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[veto, "leverage"] = 1.0
    out["moe_veto"] = veto.astype(int)
    out["moe_same_prob"] = same
    out["moe_opp_prob"] = opp
    out["moe_cash_prob"] = p_cash
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V52.1 defensive MoE parent veto layer.")
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=DEFAULT_JACKPOT)
    p.add_argument("--v27-model", type=Path, default=DEFAULT_V27)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=2053)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    base_bundle = _load_pickle(args.parent_model)
    jackpot_payload = _load_pickle(args.jackpot_model)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(args.v27_model)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    split_ts = pd.Timestamp("2025-10-01")
    train = train_all[train_all["timestamp"] < split_ts].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= split_ts].reset_index(drop=True)
    feature_cols = list(base_bundle.get("feature_cols") or [])
    feature_audit = _feature_audit(feature_cols, train_all, eval_df)
    parent_audit = _audit_contract(train_all, eval_df, feature_cols)
    if feature_audit["status"] != "pass":
        raise RuntimeError(f"feature audit failed: {feature_audit}")

    print(f"[{MODEL_ID}] labels and MoE training", flush=True)
    full_cfg = FullyLearnedGovernorConfig(**dict(base_bundle.get("config", {})))
    x_train, y_train, training_meta = build_training_set(train, cfg=full_cfg, stride_bars=int(args.stride), batch_size=int(args.batch_size), feature_cols=feature_cols)
    moe_bundles: dict[str, dict[str, Any]] = {}
    for i, moe_cfg in enumerate(_moe_grid()):
        moe_bundles[moe_cfg.name] = _train_moe_bundle(base_bundle=base_bundle, x=x_train, y=y_train, cfg=moe_cfg, random_state=int(args.seed + i * 101))

    print(f"[{MODEL_ID}] frozen parent/V27 predictions", flush=True)
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    base_val_dec = predict_policy_frame(base_bundle, val, close=_close(val))
    base_eval_dec = predict_policy_frame(base_bundle, eval_df, close=_close(eval_df))
    fee = float(dict(base_bundle["config"])["fee"])
    slip = float(dict(base_bundle["config"])["slip"])
    overlay = v46._base_overlay()
    baseline = {
        f"cost{mult}": v31.backtest(eval_df, base_bundle, jackpot_model, add_cfg, eval_q, overlay, fee=fee, slip=slip, cost_mult=float(mult), decisions=base_eval_dec)
        for mult in (1, 2, 3)
    }

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    print(f"[{MODEL_ID}] validation grid", flush=True)
    for cfg in _veto_grid(list(moe_bundles.keys())):
        moe_bundle = moe_bundles[cfg.moe_name]
        val_dec = _apply_veto(base_val_dec, moe_bundle, val, cfg)
        v1 = v31.backtest(val, base_bundle, jackpot_model, add_cfg, val_q, overlay, fee=fee, slip=slip, cost_mult=1.0, decisions=val_dec)
        v2 = v31.backtest(val, base_bundle, jackpot_model, add_cfg, val_q, overlay, fee=fee, slip=slip, cost_mult=2.0, decisions=val_dec)
        v3 = v31.backtest(val, base_bundle, jackpot_model, add_cfg, val_q, overlay, fee=fee, slip=slip, cost_mult=3.0, decisions=val_dec)
        row = {
            "config": asdict(cfg),
            "validation_cost1": v1,
            "validation_cost2": v2,
            "validation_cost3": v3,
            "selection_score": _score(v1, v2, v3),
            "veto_count": int(val_dec["moe_veto"].sum()),
        }
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    assert best is not None
    selected = VetoConfig(**best["config"])
    selected_moe = moe_bundles[selected.moe_name]
    eval_dec = _apply_veto(base_eval_dec, selected_moe, eval_df, selected)
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        r = v31.backtest(eval_df, base_bundle, jackpot_model, add_cfg, eval_q, overlay, fee=fee, slip=slip, cost_mult=float(mult), decisions=eval_dec, record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            ledger_path = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            ledger.to_csv(ledger_path, index=False)
            ledgers["cost1"] = str(ledger_path)
        metrics[f"cost{mult}"] = r

    model_path = args.out_dir / "v52_1_regime_moe_parent_veto.pkl"
    joblib.dump({"base_parent_model": str(args.parent_model), "moe_bundle": selected_moe, "selected_config": asdict(selected)}, model_path)
    pd.DataFrame(
        [
            {
                **{f"cfg_{k}": v for k, v in row["config"].items()},
                "veto_count": row["veto_count"],
                "selection_score": row["selection_score"],
                "val_cost1_pnl": row["validation_cost1"]["pnl"],
                "val_cost1_mdd": row["validation_cost1"]["mdd"],
                "val_cost1_trades": row["validation_cost1"]["trades"],
                "val_cost2_pnl": row["validation_cost2"]["pnl"],
                "val_cost3_pnl": row["validation_cost3"]["pnl"],
            }
            for row in rows
        ]
    ).to_csv(args.grid_out, index=False)

    blocking: list[str] = []
    warnings: list[str] = []
    if parent_audit["status"] != "pass":
        blocking.extend(parent_audit.get("blocking", []))
    warnings.extend(parent_audit.get("warnings", []))
    for mult in (1, 2, 3):
        if metrics[f"cost{mult}"]["pnl"] <= baseline[f"cost{mult}"]["pnl"]:
            warnings.append(f"oos_cost{mult}_did_not_beat_v31")
    verdict = "candidate_recheck" if not blocking and not any(w.startswith("oos_cost") for w in warnings) else "iterate"
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": verdict,
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS after validation selection",
        "feature_audit": feature_audit,
        "parent_contract_audit": parent_audit,
        "baseline_recomputed_v31": baseline,
        "selected_config": asdict(selected),
        "eval_veto_count": int(eval_dec["moe_veto"].sum()),
        "metrics": metrics,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Defensive clean-regime MoE veto. Base V31 parent remains the entry owner; MoE cannot create trades and can only veto parent LONG/SHORT when same-side confidence is weak, opposite-side confidence is high, or cash probability is high. V27 can still scout after parent veto.",
        "training_meta": training_meta,
        "selected_config": asdict(selected),
        "selection_result": best,
        "baseline_recomputed_v31": baseline,
        "metrics": metrics,
        "audit": audit,
        "artifacts": {"model": str(model_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers},
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "model": str(model_path), "selected": asdict(selected), "metrics": metrics, "verdict": verdict}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
