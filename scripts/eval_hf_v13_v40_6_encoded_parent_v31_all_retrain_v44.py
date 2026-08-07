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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import build_training_set, predict_policy_frame, prepare_features  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts import train_eval_hf_v13_jackpot_runner_v21_2 as v21  # noqa: E402
from scripts.eval_hf_v13_v40_6_full_v31_stack_retrain import (  # noqa: E402
    DEFAULT_EVAL,
    DEFAULT_PARENT,
    DEFAULT_PARENT_REPORT,
    DEFAULT_TRAIN,
    _build_v40_6_frames,
    _load_bundle,
    _projection_targets,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import (  # noqa: E402
    _audit_contract,
    _close,
    _feature_cols,
    _json_default,
    _read,
)
from scripts.train_eval_hf_v13_multitrack_foundation_parent_v40 import _parent_cfg  # noqa: E402


MODEL_ID = "hf_v13_v40_6_encoded_parent_v31_all_retrain_v44_20260512"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_v40_6_encoded_parent_v31_all_retrain_v44_20260512"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_v40_6_encoded_parent_v31_all_retrain_v44_20260512_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_v40_6_encoded_parent_v31_all_retrain_v44_20260512_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_v40_6_encoded_parent_v31_all_retrain_v44_20260512_grid.csv"


def _encoded_seq_cols(df: pd.DataFrame) -> list[str]:
    cols = list(v27._seq_cols(df))
    for c in df.columns:
        lc = c.lower()
        if (c.startswith("macro_factor_") or c.startswith("micro_factor_")) and c not in cols:
            cols.append(c)
        if len(cols) >= 96:
            break
    bad = [
        c
        for c in cols
        if any(tok in c.lower() for tok in ("target", "label", "future", "cash_after", "pnl_after", "regime_v2", "hdb", "hmm"))
    ]
    if bad:
        raise RuntimeError(f"forbidden sequence columns selected: {bad}")
    return cols


def _train_deep_scout(ds: dict[str, np.ndarray], norm: dict[str, np.ndarray], *, epochs: int, batch_size: int) -> v27.DeepAlphaTCN:
    x = v27._apply_norm(ds["seq"], norm)
    y = ds["target"].astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = v27.DeepAlphaTCN(x.shape[-1]).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=int(batch_size), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    for epoch in range(int(epochs)):
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu())
            n += 1
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == int(epochs):
            print(
                json.dumps(
                    {"stage": "deep_scout_train", "epoch": epoch + 1, "epochs": int(epochs), "loss": total / max(n, 1)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return model.cpu().eval()


def _predict_deep(model: v27.DeepAlphaTCN, df: pd.DataFrame, seq_cols: list[str], norm: dict[str, np.ndarray]) -> np.ndarray:
    seqs = np.stack([v31._seq_at(df, i, seq_cols) for i in range(len(df))]).astype(np.float32)
    x = v27._apply_norm(seqs, norm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            pred = model(torch.from_numpy(x[start : start + 512]).to(device))
            out.append(pred.detach().cpu().numpy())
    model.cpu()
    return np.vstack(out).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V31 stack with encoded v40.6 parent, retrained V21 runner, retrained deep scout, and reselected V31 overlay.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--parent-report", type=Path, default=DEFAULT_PARENT_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--train-stride", type=int, default=48)
    p.add_argument("--embed-batch", type=int, default=8)
    p.add_argument("--deep-stride", type=int, default=3)
    p.add_argument("--deep-epochs", type=int, default=40)
    p.add_argument("--deep-batch-size", type=int, default=128)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[{MODEL_ID}] loading data and encoded parent", flush=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    split_ts = pd.Timestamp("2025-10-01")
    train_df = train_all[train_all["timestamp"] < split_ts].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= split_ts].reset_index(drop=True)
    parent_bundle = _load_bundle(args.parent_model)
    with args.parent_report.open(encoding="utf-8") as f:
        parent_report = json.load(f)

    feature_cols = _feature_cols(train_all, eval_df)
    parent_cfg = _parent_cfg()
    print(f"[{MODEL_ID}] rebuilding encoded v40.6 feature frames", flush=True)
    x_train, y, training_meta = build_training_set(
        train_df,
        cfg=parent_cfg,
        stride_bars=int(args.train_stride),
        batch_size=512,
        feature_cols=feature_cols,
    )
    train_idx_sample = np.arange(
        0,
        max(0, len(train_df) - parent_cfg.max_train_horizon_bars - 1),
        max(1, int(args.train_stride)),
        dtype=np.int64,
    )
    if len(train_idx_sample) != len(x_train):
        raise RuntimeError(f"train sample mismatch: {len(train_idx_sample)} vs {len(x_train)}")
    proj_targets = _projection_targets(y)
    train_feat = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=feature_cols)
    val_feat = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    eval_feat = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    train_full, val_full, eval_full, encoding_meta = _build_v40_6_frames(
        args=args,
        parent_report=parent_report,
        train_df=train_df,
        val_df=val_df,
        eval_df=eval_df,
        train_feat=train_feat,
        val_feat=val_feat,
        eval_feat=eval_feat,
        train_idx_sample=train_idx_sample,
        proj_targets=proj_targets,
    )

    base = dict(parent_bundle.get("config", {}))
    fee = float(base.get("fee", parent_cfg.fee))
    slip = float(base.get("slip", parent_cfg.slip))
    print(f"[{MODEL_ID}] predicting encoded parent decisions", flush=True)
    val_dec = predict_policy_frame(parent_bundle, val_full, close=_close(val_full))
    eval_dec = predict_policy_frame(parent_bundle, eval_full, close=_close(eval_full))

    print(f"[{MODEL_ID}] retraining V21.2 runner on encoded parent frame", flush=True)
    runner_model = v21._fit_cost_runner(train_full, parent_bundle, fee=fee, slip=slip)

    print(f"[{MODEL_ID}] retraining V27-style deep scout on encoded frame", flush=True)
    seq_cols = _encoded_seq_cols(train_full)
    train_ds = v27._build_train_set(train_full, seq_cols, fee=fee, slip=slip, stride=int(args.deep_stride))
    norm = v27._normalizer(train_ds["seq"])
    deep_model = _train_deep_scout(train_ds, norm, epochs=int(args.deep_epochs), batch_size=int(args.deep_batch_size))
    val_q = _predict_deep(deep_model, val_full, seq_cols, norm)
    eval_q = _predict_deep(deep_model, eval_full, seq_cols, norm)

    print(f"[{MODEL_ID}] joint-selecting runner config and V31 overlay on 2025 Q4", flush=True)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for add_cfg in v21._grid():
        for overlay in v31._grid():
            v1 = v31.backtest(val_full, parent_bundle, runner_model, add_cfg, val_q, overlay, fee=fee, slip=slip, cost_mult=1.0, decisions=val_dec)
            v2 = v31.backtest(val_full, parent_bundle, runner_model, add_cfg, val_q, overlay, fee=fee, slip=slip, cost_mult=2.0, decisions=val_dec)
            v3 = v31.backtest(val_full, parent_bundle, runner_model, add_cfg, val_q, overlay, fee=fee, slip=slip, cost_mult=3.0, decisions=val_dec)
            score = v31._score(v1, v2, v3)
            row = {
                "runner_config": asdict(add_cfg),
                "overlay_config": asdict(overlay),
                "runner": add_cfg.name,
                "overlay": overlay.name,
                "selection_score": score,
                "validation_cost1": v1,
                "validation_cost2": v2,
                "validation_cost3": v3,
            }
            rows.append(row)
            if best is None or score > best["selection_score"]:
                best = row
    assert best is not None
    selected_add = v21.CostRunnerConfig(**best["runner_config"])
    selected_overlay = v31.OverlayConfig(**best["overlay_config"])

    print(f"[{MODEL_ID}] evaluating fixed 2026 OOS", flush=True)
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        r = v31.backtest(
            eval_full,
            parent_bundle,
            runner_model,
            selected_add,
            eval_q,
            selected_overlay,
            fee=fee,
            slip=slip,
            cost_mult=float(mult),
            decisions=eval_dec,
            record=(mult == 1),
        )
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            lp = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            lp.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(lp, index=False)
            ledgers["cost1"] = str(lp)
        metrics[f"cost{mult}"] = r

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runner_path = args.out_dir / "v44_retrained_v21_2_runner.pkl"
    scout_path = args.out_dir / "v44_retrained_deep_scout.pt"
    manifest_path = args.out_dir / "v44_encoded_parent_v31_all_retrain_manifest.json"
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "base_parent": str(args.parent_model),
            "cost_runner": runner_model,
            "selected_config": asdict(selected_add),
        },
        runner_path,
    )
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": deep_model.state_dict(),
            "seq_cols": seq_cols,
            "norm": norm,
            "selected_overlay": asdict(selected_overlay),
            "selected_runner_config": asdict(selected_add),
            "parent_model": str(args.parent_model),
            "runner_model": str(runner_path),
        },
        scout_path,
    )

    pd.DataFrame(
        [
            {
                "runner": r["runner"],
                "overlay": r["overlay"],
                "selection_score": r["selection_score"],
                "val_pnl": r["validation_cost1"]["pnl"],
                "val_mdd": r["validation_cost1"]["mdd"],
                "val_trades": r["validation_cost1"]["trades"],
                "val_deep_entries": r["validation_cost1"].get("deep_entries", 0),
                "val_c2_pnl": r["validation_cost2"]["pnl"],
                "val_c3_pnl": r["validation_cost3"]["pnl"],
                "val_adds": r["validation_cost1"].get("runner_actions", {}).get("v21_add_on", 0),
                "val_rejects": r["validation_cost1"].get("runner_actions", {}).get("v21_reject", 0),
            }
            for r in rows
        ]
    ).sort_values("selection_score", ascending=False).to_csv(args.grid_out, index=False)

    feature_audit_cols = [c for c in list(parent_bundle.get("feature_cols") or []) if not c.startswith("macro_factor_") and not c.startswith("micro_factor_")]
    feature_audit = _audit_contract(train_all, eval_df, feature_audit_cols)
    forbidden_seq_cols = [
        c
        for c in seq_cols
        if any(tok in c.lower() for tok in ("target", "label", "future", "cash_after", "pnl_after", "regime_v2", "hdb", "hmm"))
    ]
    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit.get("status") != "pass":
        blocking.extend(feature_audit.get("blocking", []))
    if forbidden_seq_cols:
        blocking.append(f"forbidden_sequence_columns={forbidden_seq_cols}")
    warnings.extend(feature_audit.get("warnings", []))
    if metrics["cost2"]["pnl"] <= 0.0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0.0:
        warnings.append("cost3_not_survived")
    baseline_cost1 = 67.3873915423753
    baseline_v40_6_no_deep = 133.36501388253345
    verdict = "promote" if not blocking and metrics["cost1"]["pnl"] > baseline_v40_6_no_deep and metrics["cost2"]["pnl"] > 0.0 and metrics["cost3"]["pnl"] > 0.0 else "reject"
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": verdict,
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31 joint runner/overlay selection",
        "oos_window": "2026 fixed OOS only after selection",
        "parent": "v40_6_target_aware_encoded_parent",
        "parent_retrained_in_this_script": False,
        "parent_artifact_reused": str(args.parent_model),
        "v21_2_runner_retrained": True,
        "deep_scout_retrained": True,
        "v31_overlay_reselected": True,
        "feature_audit": feature_audit,
        "forbidden_sequence_columns": forbidden_seq_cols,
        "seq_feature_count": len(seq_cols),
        "train_snapshot_count": int(len(train_ds["target"])),
        "deep_target_mean": np.mean(train_ds["target"], axis=0),
        "runner_meta": {
            k: v
            for k, v in runner_model.items()
            if k
            not in {
                "regressor",
                "q10_regressor",
                "q90_regressor",
                "classifier",
                "jackpot_classifier",
                "bad_classifier",
                "cost3_classifier",
                "feature_cols",
            }
        },
        "baseline_frozen_v27_stack_cost1_pnl": baseline_cost1,
        "baseline_v40_6_no_deep_cost1_pnl": baseline_v40_6_no_deep,
    }
    manifest = {
        "model_id": MODEL_ID,
        "parent_model": str(args.parent_model),
        "parent_report": str(args.parent_report),
        "runner_model": str(runner_path),
        "deep_scout_model": str(scout_path),
        "selected_runner_config": asdict(selected_add),
        "selected_overlay": asdict(selected_overlay),
        "seq_cols": seq_cols,
        "metrics": metrics,
        "audit_status": audit["status"],
        "verdict": verdict,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    report = {
        "model_id": MODEL_ID,
        "design": "Previous V31 stack with parent swapped to the encoded V40.6 main parent. Around that parent, V21.2 cost-stressed runner and V27-style deep scout are retrained on encoded frames, V31 exit overlay is reselected on 2025 Q4, and fixed 2026 OOS is evaluated after selection.",
        "parent_model": str(args.parent_model),
        "parent_report": str(args.parent_report),
        "encoding_meta": encoding_meta,
        "training_meta": training_meta,
        "runner_model": str(runner_path),
        "deep_scout_model": str(scout_path),
        "selected_runner_config": asdict(selected_add),
        "selected_overlay": asdict(selected_overlay),
        "selection_result": best,
        "metrics": metrics,
        "audit": audit,
        "artifacts": {
            "manifest": str(manifest_path),
            "runner_model": str(runner_path),
            "deep_scout_model": str(scout_path),
            "report": str(args.report_out),
            "audit": str(args.audit_out),
            "grid": str(args.grid_out),
            "ledgers": ledgers,
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "audit": str(args.audit_out),
                "grid": str(args.grid_out),
                "runner_model": str(runner_path),
                "deep_scout_model": str(scout_path),
                "selected_runner": asdict(selected_add),
                "selected_overlay": asdict(selected_overlay),
                "metrics": metrics,
                "verdict": verdict,
                "audit_status": audit["status"],
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
