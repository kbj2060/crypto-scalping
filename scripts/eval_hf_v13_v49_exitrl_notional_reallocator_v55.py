#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import train_eval_hf_v13_frozen_v27_offline_rl_exit_overlay_v33 as v33  # noqa: E402
from scripts.eval_hf_v13_v31_rl_surrounding_v49_v50_v51 import (  # noqa: E402
    ClosePolicyNet,
    TorchClosePolicy,
    _feature_audit,
    _numeric_cols,
    _patch_v33_state,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "hf_v13_v49_exitrl_notional_reallocator_v55_20260513"
DEFAULT_PARENT = v31.DEFAULT_PARENT
DEFAULT_JACKPOT = v31.DEFAULT_JACKPOT
DEFAULT_V27 = v31.DEFAULT_V27
DEFAULT_V49 = ROOT / "data/ensemble/supervised/hf_v13_v31_rl_surrounding_v49_v50_v51_20260512/v49_exit_rl_raw_all.pkl"
DEFAULT_TRAIN = v31.DEFAULT_TRAIN
DEFAULT_EVAL = v31.DEFAULT_EVAL
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_v49_exitrl_notional_reallocator_v55_20260513"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_v49_exitrl_notional_reallocator_v55_20260513_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_v49_exitrl_notional_reallocator_v55_20260513_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_v49_exitrl_notional_reallocator_v55_20260513_grid.csv"

V31_BASELINE = {
    "cost1": {"pnl": 277.0679629973942, "mdd": -31.74},
    "cost2": {"pnl": 112.79326141840412, "mdd": -31.46},
    "cost3": {"pnl": 20.933695032758784, "mdd": -43.09},
}


@dataclass(frozen=True)
class V55Config:
    name: str
    parent_mult: float
    parent_cap: float
    deep_mult: float
    tpsl_power: float
    addon_frac: float
    addon_total_mult: float
    addon_cap: float


def _configs() -> list[V55Config]:
    rows: list[V55Config] = [
        V55Config("v55_identity_v49", 1.0, 2.75, 1.0, 1.0, 0.20, 1.35, 2.75),
    ]
    i = 0
    for parent_mult in (1.05, 1.10, 1.20, 1.35):
        for deep_mult in (1.0, 1.15, 1.30):
            for tpsl_power in (0.75, 1.00):
                rows.append(V55Config(f"v55_bal_{i}", parent_mult, 4.14, deep_mult, tpsl_power, 0.20, 1.35, 4.14))
                i += 1
    for deep_mult in (1.15, 1.30, 1.50):
        rows.append(V55Config(f"v55_deep_only_{deep_mult:.2f}", 1.0, 2.75, deep_mult, 1.0, 0.20, 1.35, 2.75))
    for addon_frac, total in ((0.35, 1.60), (0.50, 1.85), (0.65, 2.10)):
        rows.append(V55Config(f"v55_winner_add_{addon_frac:.2f}", 1.0, 2.75, 1.0, 1.0, addon_frac, total, 4.14))
    return rows


def _install_pickle_aliases() -> None:
    # V49 was produced by a script run as __main__; expose the same symbols so
    # old joblib artifacts remain readable without retraining or rewriting.
    import __main__

    setattr(__main__, "TorchClosePolicy", TorchClosePolicy)
    setattr(__main__, "ClosePolicyNet", ClosePolicyNet)


def _load_v49(path: Path) -> dict[str, Any]:
    _install_pickle_aliases()
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "policy" not in obj or "selected_config" not in obj:
        raise TypeError(f"{path} is not a V49 exit-RL policy payload")
    return obj


def _scale_decisions(decisions: pd.DataFrame, cfg: V55Config) -> pd.DataFrame:
    out = decisions.copy()
    trade = (out["action"].to_numpy(dtype=np.int64) != ACTION_CASH) & (out["side"].to_numpy(dtype=np.int64) != 0)
    old_notional = out["notional_exposure"].to_numpy(dtype=np.float64)
    new_notional = old_notional.copy()
    new_notional[trade] = np.minimum(old_notional[trade] * float(cfg.parent_mult), float(cfg.parent_cap))
    scale = np.divide(new_notional, np.maximum(old_notional, 1e-12), out=np.ones_like(new_notional), where=old_notional > 0)
    out["notional_exposure"] = new_notional
    out["leverage"] = np.minimum(np.maximum(out["leverage"].to_numpy(dtype=np.float64), new_notional), 5.0)
    out["position_fraction"] = np.divide(new_notional, np.maximum(out["leverage"].to_numpy(dtype=np.float64), 1e-12))
    tpsl_scale = np.power(scale, float(cfg.tpsl_power))
    out["take_profit"] = out["take_profit"].to_numpy(dtype=np.float64) * tpsl_scale
    out["stop_loss"] = out["stop_loss"].to_numpy(dtype=np.float64) * tpsl_scale
    cash = ~trade
    out.loc[cash, ["side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[cash, "leverage"] = 1.0
    return out


def _scale_overlay(overlay: v33.OverlayConfig, cfg: V55Config) -> v33.OverlayConfig:
    if cfg.deep_mult == 1.0:
        return replace(overlay, name=f"{overlay.name}_{cfg.name}")
    scale = float(cfg.deep_mult) ** float(cfg.tpsl_power)
    return replace(
        overlay,
        name=f"{overlay.name}_{cfg.name}_deepx{cfg.deep_mult:.2f}",
        notional=float(overlay.notional) * float(cfg.deep_mult),
        base_tp=float(overlay.base_tp) * scale,
        base_sl=float(overlay.base_sl) * scale,
    )


def _scale_add_cfg(add_cfg: CostRunnerConfig, cfg: V55Config) -> CostRunnerConfig:
    return replace(
        add_cfg,
        name=f"{add_cfg.name}_{cfg.name}",
        full_add_frac=float(cfg.addon_frac),
        half_add_frac=0.0,
        max_total_mult=float(cfg.addon_total_mult),
        max_entry_notional=float(cfg.addon_cap),
    )


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    target_bonus = 0.25 * max(float(c1["pnl"]) - V31_BASELINE["cost1"]["pnl"], 0.0)
    mdd_penalty = 0.55 * max(abs(float(c1["mdd"])) - 45.0, 0.0)
    return float(c1["pnl"] + 0.45 * c2["pnl"] + 0.25 * c3["pnl"] - 0.30 * abs(c1["mdd"]) - mdd_penalty + target_bonus)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V55 V31 + V49 Exit-RL with conservative notional reallocation.")
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=DEFAULT_JACKPOT)
    p.add_argument("--v27-model", type=Path, default=DEFAULT_V27)
    p.add_argument("--v49-policy", type=Path, default=DEFAULT_V49)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[{MODEL_ID}] loading frozen stack and V49 policy", flush=True)
    parent = joblib.load(args.parent_model)
    jackpot_payload = joblib.load(args.jackpot_model)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg0 = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(args.v27_model)
    v49_payload = _load_v49(args.v49_policy)
    policy = v49_payload["policy"]
    v49_base_cfg = v33.OverlayConfig(**dict(v49_payload["selected_config"]))
    base = dict(parent["config"])
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    raw_cols = list(v49_payload.get("feature_cols") or _numeric_cols(train_all, eval_df))
    parent_audit = _audit_contract(train_all, eval_df, list(parent.get("feature_cols") or []))
    state_audit = _feature_audit(raw_cols, train_all, eval_df)
    if parent_audit["status"] != "pass" or state_audit["status"] != "pass":
        print(f"[{MODEL_ID}] audit precheck failed; no backtest", flush=True)
    print(f"[{MODEL_ID}] predicting decisions/utilities", flush=True)
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    val_dec0 = predict_policy_frame(parent, val, close=_close(val))
    eval_dec0 = predict_policy_frame(parent, eval_df, close=_close(eval_df))

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    with _patch_v33_state(raw_cols):
        for cfg in _configs():
            val_dec = _scale_decisions(val_dec0, cfg)
            add_cfg = _scale_add_cfg(add_cfg0, cfg)
            overlay = _scale_overlay(v49_base_cfg, cfg)
            print(f"[{MODEL_ID}] validation cfg={cfg.name} overlay={overlay.name}", flush=True)
            v1 = v33.backtest(val, parent, jackpot_model, add_cfg, val_q, policy, overlay, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=1.0, decisions=val_dec)
            v2 = v33.backtest(val, parent, jackpot_model, add_cfg, val_q, policy, overlay, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=2.0, decisions=val_dec)
            v3 = v33.backtest(val, parent, jackpot_model, add_cfg, val_q, policy, overlay, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=3.0, decisions=val_dec)
            row = {"config": asdict(cfg), "overlay": asdict(overlay), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
            rows.append(row)
            if best is None or row["selection_score"] > best["selection_score"]:
                best = row
        assert best is not None
        selected_cfg = V55Config(**best["config"])
        selected_overlay = v33.OverlayConfig(**best["overlay"])
        eval_dec = _scale_decisions(eval_dec0, selected_cfg)
        selected_add = _scale_add_cfg(add_cfg0, selected_cfg)
        metrics: dict[str, Any] = {}
        ledgers: dict[str, str] = {}
        for mult in (1, 2, 3):
            r = v33.backtest(eval_df, parent, jackpot_model, selected_add, eval_q, policy, selected_overlay, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=float(mult), decisions=eval_dec, record=(mult == 1))
            if mult == 1:
                ledger = pd.DataFrame(r.pop("trade_records", []))
                lp = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
                lp.parent.mkdir(parents=True, exist_ok=True)
                ledger.to_csv(lp, index=False)
                ledgers["cost1"] = str(lp)
            metrics[f"cost{mult}"] = r

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "v55_v49_exitrl_notional_reallocator_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "selected_config": asdict(selected_cfg),
                "selected_overlay": asdict(selected_overlay),
                "v49_policy": str(args.v49_policy),
                "parent_model": str(args.parent_model),
                "jackpot_model": str(args.jackpot_model),
                "v27_model": str(args.v27_model),
                "raw_state_feature_count": int(len(raw_cols)),
            },
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                **{f"cfg_{k}": v for k, v in r["config"].items()},
                **{f"overlay_{k}": v for k, v in r["overlay"].items()},
                "selection_score": r["selection_score"],
                "val_pnl": r["validation_cost1"]["pnl"],
                "val_mdd": r["validation_cost1"]["mdd"],
                "val_trades": r["validation_cost1"]["trades"],
                "val_c2_pnl": r["validation_cost2"]["pnl"],
                "val_c3_pnl": r["validation_cost3"]["pnl"],
            }
            for r in rows
        ]
    ).to_csv(args.grid_out, index=False)
    blocking = list(parent_audit.get("blocking", [])) + [f"state:{x}" for x in state_audit.get("blocking", [])]
    warnings = list(parent_audit.get("warnings", [])) + [f"state:{x}" for x in state_audit.get("warnings", [])]
    if metrics["cost1"]["pnl"] <= V31_BASELINE["cost1"]["pnl"]:
        warnings.append("oos_cost1_did_not_beat_v31")
    if metrics["cost1"]["pnl"] <= 500.0:
        warnings.append("target_pnl_500_not_reached")
    if metrics["cost2"]["pnl"] <= 0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0:
        warnings.append("cost3_not_survived")
    verdict = "promote" if not blocking and metrics["cost1"]["pnl"] > 500.0 and metrics["cost2"]["pnl"] > 0.0 else "iterate"
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": verdict,
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "policy": "v49_exitrl_notional_reallocator_v55",
        "entry_owner_frozen": True,
        "v49_exit_policy_reused": True,
        "direction_changed": False,
        "parent_audit": parent_audit,
        "state_feature_audit": state_audit,
        "selected_config": asdict(selected_cfg),
        "selected_overlay": asdict(selected_overlay),
        "metrics": metrics,
        "baseline_v31": V31_BASELINE,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "V55 keeps V31 parent and V27 entries frozen, reuses the audited V49 raw-all Exit-RL close/hold policy, then searches conservative parent/deep/add-on notional reallocations using 2025 Q4 only.",
        "selected_config": asdict(selected_cfg),
        "selected_overlay": asdict(selected_overlay),
        "selection_result": best,
        "metrics": metrics,
        "audit": audit,
        "artifacts": {"manifest": str(manifest_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers},
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "manifest": str(manifest_path), "selected_config": asdict(selected_cfg), "selected_overlay": asdict(selected_overlay), "metrics": metrics, "verdict": verdict}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
