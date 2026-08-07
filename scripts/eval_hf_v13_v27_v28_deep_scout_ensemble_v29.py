#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import predict_policy_frame
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _read
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import (
    DeepAlphaConfig,
    DeepAlphaTCN,
    V21_2_COST1,
    V21_2_COST2,
    V21_2_COST3,
    _grid as v27_grid,
    _json_default,
    _score,
    backtest,
)
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig


MODEL_ID = "hf_v13_v27_v28_deep_scout_ensemble_v29_20260511"
DEFAULT_PARENT = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_margin110_20260511/v13_clean_regime_margin110.pkl"
DEFAULT_JACKPOT = ROOT / "data/ensemble/supervised/hf_v13_jackpot_runner_v21_2_20260511/v21_2_jackpot_runner.pkl"
DEFAULT_V27 = ROOT / "data/ensemble/supervised/hf_v13_deep_alpha_candidate_expansion_v27_20260511/v27_deep_alpha_candidate_expansion.pt"
DEFAULT_V28 = ROOT / "data/ensemble/supervised/hf_v13_margin110_deep_residual_scout_v28_20260511/v28_margin110_deep_residual_scout.pt"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_v27_v28_deep_scout_ensemble_v29_20260511"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_v27_v28_deep_scout_ensemble_v29_20260511_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_v27_v28_deep_scout_ensemble_v29_20260511_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_v27_v28_deep_scout_ensemble_v29_20260511_grid.csv"

V27_COST1 = 226.82447187089713
V27_COST2 = 123.11659362616143
V27_COST3 = 14.22783363158393
V28_COST1 = 179.57754230121466
V28_COST2 = 106.6560393877917
V28_COST3 = 20.918985030784377


def _load_deep_artifact(path: Path) -> tuple[dict[str, Any], DeepAlphaTCN]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    seq_cols = list(payload["seq_cols"])
    model = DeepAlphaTCN(len(seq_cols))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return payload, model


def _seq_at(df: pd.DataFrame, idx: int, cols: list[str], seq_len: int = 72) -> np.ndarray:
    start = max(0, idx - seq_len + 1)
    arr = df.loc[start:idx, cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    if len(arr) < seq_len:
        arr = np.vstack([np.zeros((seq_len - len(arr), len(cols)), dtype=np.float32), arr])
    return arr[-seq_len:]


def _apply_norm(seqs: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    return ((seqs - norm["mean"][None, None, :]) / norm["std"][None, None, :]).astype(np.float32)


def _predict_artifact(model: DeepAlphaTCN, payload: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    seq_cols = list(payload["seq_cols"])
    seqs = np.stack([_seq_at(df, i, seq_cols) for i in range(len(df))]).astype(np.float32)
    x = _apply_norm(seqs, payload["norm"])
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            out.append(model(torch.from_numpy(x[start : start + 512])).numpy())
    return np.vstack(out).astype(np.float32)


def _combine(q27: np.ndarray, q28: np.ndarray, mode: str) -> np.ndarray:
    if mode == "v27_only":
        return q27.copy()
    if mode == "v28_only":
        return q28.copy()
    if mode == "avg":
        return ((q27 + q28) * 0.5).astype(np.float32)
    if mode == "v27_weighted":
        return (q27 * 0.70 + q28 * 0.30).astype(np.float32)
    if mode == "v28_weighted":
        return (q27 * 0.30 + q28 * 0.70).astype(np.float32)
    side27 = np.where(q27[:, 0] >= q27[:, 1], 1, -1)
    side28 = np.where(q28[:, 0] >= q28[:, 1], 1, -1)
    if mode == "consensus":
        q = ((q27 + q28) * 0.5).astype(np.float32)
        q[side27 != side28] = 0.0
        return q
    if mode == "max_edge":
        e27 = np.max(q27, axis=1)
        e28 = np.max(q28, axis=1)
        use28 = e28 > e27
        q = q27.copy()
        q[use28] = q28[use28]
        return q.astype(np.float32)
    raise ValueError(f"unknown ensemble mode: {mode}")


def _ensemble_grid() -> list[tuple[str, DeepAlphaConfig]]:
    base = v27_grid()
    selected_like = [
        DeepAlphaConfig("v29_slow_high_edge_n0.8", 0.010, 0.0040, 0.8, 0.045, 0.022, 48, 12),
        DeepAlphaConfig("v29_slow_high_edge_n1.0", 0.010, 0.0040, 1.0, 0.045, 0.022, 48, 12),
        DeepAlphaConfig("v29_slow_high_edge_n1.2", 0.010, 0.0040, 1.2, 0.045, 0.022, 48, 12),
        DeepAlphaConfig("v29_precision_n1.2", 0.014, 0.0050, 1.2, 0.050, 0.022, 48, 12),
    ]
    modes = ["v27_only", "v28_only", "avg", "v27_weighted", "v28_weighted", "consensus", "max_edge"]
    rows: list[tuple[str, DeepAlphaConfig]] = []
    for mode in modes:
        for cfg in selected_like:
            rows.append((mode, cfg))
    for mode in ("avg", "consensus", "max_edge"):
        for cfg in base:
            rows.append((mode, cfg))
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V29 ensemble of V27 and V28 deep scout signals.")
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=DEFAULT_JACKPOT)
    p.add_argument("--v27-model", type=Path, default=DEFAULT_V27)
    p.add_argument("--v28-model", type=Path, default=DEFAULT_V28)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bundle = joblib.load(args.parent_model)
    jackpot_payload = joblib.load(args.jackpot_model)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    base = dict(bundle["config"])
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_audit = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))

    v27_payload, v27_model = _load_deep_artifact(args.v27_model)
    v28_payload, v28_model = _load_deep_artifact(args.v28_model)
    val27 = _predict_artifact(v27_model, v27_payload, val)
    val28 = _predict_artifact(v28_model, v28_payload, val)
    eval27 = _predict_artifact(v27_model, v27_payload, eval_df)
    eval28 = _predict_artifact(v28_model, v28_payload, eval_df)
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))

    forbidden_cols = []
    for payload in (v27_payload, v28_payload):
        forbidden_cols.extend([c for c in payload.get("seq_cols", []) if any(tok in c.lower() for tok in ("future", "target", "label", "leak", "hdbscan", "hmm", "regime_v2"))])

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    q_cache: dict[str, np.ndarray] = {}
    for mode, cfg in _ensemble_grid():
        if mode not in q_cache:
            q_cache[mode] = _combine(val27, val28, mode)
        q = q_cache[mode]
        v1 = backtest(val, bundle, jackpot_model, add_cfg, q, cfg, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=1.0, decisions=val_dec)
        v2 = backtest(val, bundle, jackpot_model, add_cfg, q, cfg, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=2.0, decisions=val_dec)
        v3 = backtest(val, bundle, jackpot_model, add_cfg, q, cfg, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=3.0, decisions=val_dec)
        row = {"mode": mode, "config": asdict(cfg), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    assert best is not None
    selected_mode = str(best["mode"])
    selected_cfg = DeepAlphaConfig(**best["config"])
    eval_q = _combine(eval27, eval28, selected_mode)

    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        r = backtest(eval_df, bundle, jackpot_model, add_cfg, eval_q, selected_cfg, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=float(mult), decisions=eval_dec, record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            lp = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            lp.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(lp, index=False)
            ledgers["cost1"] = str(lp)
        metrics[f"cost{mult}"] = r

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "v29_ensemble_manifest.json"
    manifest = {
        "model_id": MODEL_ID,
        "selected_mode": selected_mode,
        "selected_config": asdict(selected_cfg),
        "v27_model": str(args.v27_model),
        "v28_model": str(args.v28_model),
        "v27_seq_cols": v27_payload.get("seq_cols", []),
        "v28_seq_cols": v28_payload.get("seq_cols", []),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "mode": r["mode"],
                **{f"cfg_{k}": v for k, v in r["config"].items()},
                "score": r["selection_score"],
                "val_pnl": r["validation_cost1"]["pnl"],
                "val_mdd": r["validation_cost1"]["mdd"],
                "val_trades": r["validation_cost1"]["trades"],
                "val_deep_entries": r["validation_cost1"].get("deep_entries", 0),
                "val_c2_pnl": r["validation_cost2"]["pnl"],
                "val_c3_pnl": r["validation_cost3"]["pnl"],
            }
            for r in rows
        ]
    ).to_csv(args.grid_out, index=False)

    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit.get("blocking", []))
    if forbidden_cols:
        blocking.append(f"forbidden_deep_sequence_columns={sorted(set(forbidden_cols))}")
    warnings.extend(feature_audit.get("warnings", []))
    if metrics["cost1"]["pnl"] <= V27_COST1:
        warnings.append("oos_cost1_did_not_beat_v27")
    if metrics["cost1"]["pnl"] <= V28_COST1:
        warnings.append("oos_cost1_did_not_beat_v28")
    if metrics["cost2"]["pnl"] <= 0.0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0.0:
        warnings.append("cost3_not_survived")
    verdict = "promote" if not blocking and metrics["cost1"]["pnl"] > V27_COST1 and metrics["cost2"]["pnl"] > 0.0 and metrics["cost3"]["pnl"] > 0.0 else "iterate"
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": verdict,
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "policy": "v27_v28_deep_scout_ensemble_v29",
        "parent_model": str(args.parent_model),
        "jackpot_model": str(args.jackpot_model),
        "v27_model": str(args.v27_model),
        "v28_model": str(args.v28_model),
        "v21_2_parent_preserved": True,
        "deep_sleeve_only_when_parent_cash": True,
        "selected_mode": selected_mode,
        "selected_config": asdict(selected_cfg),
        "feature_audit": feature_audit,
        "metrics": metrics,
        "baselines": {
            "v21_2": {"cost1": V21_2_COST1, "cost2": V21_2_COST2, "cost3": V21_2_COST3},
            "v27": {"cost1": V27_COST1, "cost2": V27_COST2, "cost3": V27_COST3},
            "v28": {"cost1": V28_COST1, "cost2": V28_COST2, "cost3": V28_COST3},
        },
    }
    report = {
        "model_id": MODEL_ID,
        "design": "V29 combines the trained V27 and V28 deep scout models as an ensemble signal. The V21.2 jackpot parent is preserved; deep scout entries are allowed only when the parent is CASH. Ensemble mode and deep scout contract are selected on 2025 Q4 only, then evaluated on fixed 2026 OOS.",
        "selected_mode": selected_mode,
        "selected_config": asdict(selected_cfg),
        "selection_result": best,
        "metrics": metrics,
        "audit": audit,
        "artifacts": {"manifest": str(manifest_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers},
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "manifest": str(manifest_path), "selected_mode": selected_mode, "selected": asdict(selected_cfg), "metrics": metrics, "verdict": verdict}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
