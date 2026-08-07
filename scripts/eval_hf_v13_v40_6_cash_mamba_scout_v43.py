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

try:
    from mamba_ssm import Mamba
except Exception:  # pragma: no cover - optional native backend
    Mamba = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    build_training_set,
    predict_policy_frame,
    prepare_features,
)
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.eval_hf_v13_v40_6_full_v31_stack_retrain import (  # noqa: E402
    DEFAULT_EVAL,
    DEFAULT_PARENT,
    DEFAULT_PARENT_REPORT,
    DEFAULT_TRAIN,
    _build_v40_6_frames,
    _load_bundle,
    _projection_targets,
)
from scripts.eval_hf_v13_v40_6_nohold_deep_scout_v42 import (  # noqa: E402
    _score,
    _scout_grid,
    backtest_nohold_deep,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import (  # noqa: E402
    _audit_contract,
    _close,
    _feature_cols,
    _fill_price,
    _json_default,
    _read,
)
from scripts.train_eval_hf_v13_deep_alpha_mamba_ssm_v41 import (  # noqa: E402
    DeepAlphaMambaStyleSSM,
    _apply_norm,
    _normalizer,
    _seq_cols,
)
from scripts.train_eval_hf_v13_multitrack_foundation_parent_v40 import _parent_cfg  # noqa: E402


MODEL_ID = "hf_v13_v40_6_cash_mamba_scout_v43_20260512"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_v40_6_cash_mamba_scout_v43_20260512"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_v40_6_cash_mamba_scout_v43_20260512_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_v40_6_cash_mamba_scout_v43_20260512_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_v40_6_cash_mamba_scout_v43_20260512_grid.csv"
SEQ_LEN = 72
HORIZONS = (12, 24, 48)


class NativeMambaScout(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 96, blocks: int = 2) -> None:
        super().__init__()
        if Mamba is None:
            raise RuntimeError("mamba-ssm is not installed; install mamba-ssm or use --mamba-backend custom")
        self.input_proj = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList(
            [Mamba(d_model=hidden, d_state=16, d_conv=4, expand=2) for _ in range(int(blocks))]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(int(blocks))])
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        for block, norm in zip(self.blocks, self.norms):
            z = norm(z + block(z))
        pooled = torch.cat([z.mean(dim=1), z[:, -1, :]], dim=-1)
        return self.head(pooled)


def _build_cash_scout_train_set(
    df: pd.DataFrame,
    decisions: pd.DataFrame,
    seq_cols: list[str],
    *,
    fee: float,
    slip: float,
    stride: int,
) -> dict[str, np.ndarray]:
    seqs: list[np.ndarray] = []
    targets: list[list[float]] = []
    for i in range(SEQ_LEN, len(df) - max(HORIZONS) - 2, max(1, int(stride))):
        dec = decisions.iloc[i]
        if int(dec.action) != ACTION_CASH or int(dec.side) != 0:
            continue
        entry_i = min(i + 1, len(df) - 1)
        long_entry = _fill_price(df, entry_i, 1, slip, entry=True)
        short_entry = _fill_price(df, entry_i, -1, slip, entry=True)
        long_rewards: list[float] = []
        short_rewards: list[float] = []
        for h in HORIZONS:
            exit_i = min(i + h, len(df) - 1)
            long_exit = _fill_price(df, exit_i, 1, slip, entry=False)
            short_exit = _fill_price(df, exit_i, -1, slip, entry=False)
            long_rewards.append((long_exit - long_entry) / max(long_entry, 1e-12) - fee * 2.0)
            short_rewards.append((short_entry - short_exit) / max(short_entry, 1e-12) - fee * 2.0)
        seqs.append(v31._seq_at(df, i, seq_cols))
        targets.append([float(max(long_rewards)), float(max(short_rewards))])
    if not seqs:
        raise RuntimeError("no v40.6 CASH scout train sequences")
    return {"seq": np.stack(seqs).astype(np.float32), "target": np.asarray(targets, dtype=np.float32)}


def _train_cash_mamba(
    ds: dict[str, np.ndarray],
    norm: dict[str, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    backend: str,
) -> nn.Module:
    x = _apply_norm(ds["seq"], norm)
    y = ds["target"].astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if backend == "native":
        model: nn.Module = NativeMambaScout(x.shape[-1], hidden=96, blocks=2).to(device)
    elif backend == "custom":
        model = DeepAlphaMambaStyleSSM(x.shape[-1], hidden=96, blocks=2).to(device)
    else:
        raise ValueError(f"unsupported mamba backend: {backend}")
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=int(batch_size), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    for epoch in range(int(epochs)):
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
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
                    {
                        "stage": "cash_mamba_train",
                        "backend": backend,
                        "epoch": epoch + 1,
                        "epochs": int(epochs),
                        "loss": total / max(n, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return model.cpu().eval()


def _predict_all(model: nn.Module, df: pd.DataFrame, seq_cols: list[str], norm: dict[str, np.ndarray]) -> np.ndarray:
    seqs = np.stack([v31._seq_at(df, i, seq_cols) for i in range(len(df))]).astype(np.float32)
    x = _apply_norm(seqs, norm)
    out: list[np.ndarray] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 512):
            pred = model(torch.from_numpy(x[start : start + 512]).to(device))
            out.append(pred.detach().cpu().numpy())
    model.cpu()
    return np.vstack(out).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Mamba-style scout only on v40.6 CASH states, then attach it to no-hold v40.6.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--parent-report", type=Path, default=DEFAULT_PARENT_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--train-stride", type=int, default=48)
    p.add_argument("--scout-stride", type=int, default=3)
    p.add_argument("--embed-batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--mamba-backend", choices=["custom", "native"], default="custom")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[{MODEL_ID}] loading v40.6 parent and data", flush=True)
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
    print(f"[{MODEL_ID}] rebuilding target-aware v40.6 frames", flush=True)
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
    print(f"[{MODEL_ID}] predicting v40.6 parent decisions", flush=True)
    train_dec = predict_policy_frame(parent_bundle, train_full, close=_close(train_full))
    val_dec = predict_policy_frame(parent_bundle, val_full, close=_close(val_full))
    eval_dec = predict_policy_frame(parent_bundle, eval_full, close=_close(eval_full))

    seq_cols = _seq_cols(train_full)
    forbidden_seq_cols = [
        c
        for c in seq_cols
        if any(tok in c.lower() for tok in ("target", "label", "future", "cash_after", "pnl_after"))
    ]
    print(f"[{MODEL_ID}] training {args.mamba_backend} Mamba scout only on v40.6 CASH states", flush=True)
    train_ds = _build_cash_scout_train_set(
        train_full,
        train_dec,
        seq_cols,
        fee=fee,
        slip=slip,
        stride=int(args.scout_stride),
    )
    norm = _normalizer(train_ds["seq"])
    model = _train_cash_mamba(
        train_ds,
        norm,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        backend=str(args.mamba_backend),
    )
    val_q = _predict_all(model, val_full, seq_cols, norm)
    eval_q = _predict_all(model, eval_full, seq_cols, norm)

    print(f"[{MODEL_ID}] selecting scout execution config on 2025 Q4", flush=True)
    baseline_val = {
        f"cost{m}": backtest_nohold_deep(
            val_full,
            parent_bundle,
            val_q,
            None,
            fee=fee,
            slip=slip,
            cost_mult=float(m),
            decisions=val_dec,
            enable_deep=False,
        )
        for m in (1, 2, 3)
    }
    baseline_selection_score = _score(baseline_val["cost1"], baseline_val["cost2"], baseline_val["cost3"])
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for scout_cfg in _scout_grid():
        c1 = backtest_nohold_deep(val_full, parent_bundle, val_q, scout_cfg, fee=fee, slip=slip, cost_mult=1.0, decisions=val_dec)
        c2 = backtest_nohold_deep(val_full, parent_bundle, val_q, scout_cfg, fee=fee, slip=slip, cost_mult=2.0, decisions=val_dec)
        c3 = backtest_nohold_deep(val_full, parent_bundle, val_q, scout_cfg, fee=fee, slip=slip, cost_mult=3.0, decisions=val_dec)
        row = {
            "config": asdict(scout_cfg),
            "validation_cost1": c1,
            "validation_cost2": c2,
            "validation_cost3": c3,
            "selection_score": _score(c1, c2, c3),
        }
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    assert best is not None
    selected = v31.OverlayConfig(**best["config"])
    selected_beats_validation = (
        float(best["selection_score"]) > float(baseline_selection_score)
        and float(best["validation_cost1"]["pnl"]) > float(baseline_val["cost1"]["pnl"])
        and float(best["validation_cost2"]["pnl"]) > 0.0
        and float(best["validation_cost3"]["pnl"]) > 0.0
    )

    print(f"[{MODEL_ID}] evaluating fixed 2026 OOS", flush=True)
    baseline_oos: dict[str, Any] = {}
    candidate_metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        baseline_oos[f"cost{mult}"] = backtest_nohold_deep(
            eval_full,
            parent_bundle,
            eval_q,
            None,
            fee=fee,
            slip=slip,
            cost_mult=float(mult),
            decisions=eval_dec,
            enable_deep=False,
            record=(mult == 1),
        )
        if mult == 1:
            base_ledger = pd.DataFrame(baseline_oos[f"cost{mult}"].pop("trade_records", []))
            path = args.report_out.with_name(args.report_out.stem + "_baseline_cost1_ledger.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            base_ledger.to_csv(path, index=False)
            ledgers["baseline_cost1"] = str(path)
        cand = backtest_nohold_deep(
            eval_full,
            parent_bundle,
            eval_q,
            selected,
            fee=fee,
            slip=slip,
            cost_mult=float(mult),
            decisions=eval_dec,
            enable_deep=True,
            record=(mult == 1),
        )
        if mult == 1:
            cand_ledger = pd.DataFrame(cand.pop("trade_records", []))
            path = args.report_out.with_name(args.report_out.stem + "_candidate_cost1_ledger.csv")
            cand_ledger.to_csv(path, index=False)
            ledgers["candidate_cost1"] = str(path)
        candidate_metrics[f"cost{mult}"] = cand

    oos_beats_baseline = (
        float(candidate_metrics["cost1"]["pnl"]) > float(baseline_oos["cost1"]["pnl"])
        and float(candidate_metrics["cost2"]["pnl"]) > 0.0
        and float(candidate_metrics["cost3"]["pnl"]) > 0.0
    )
    promoted_variant = "cash_mamba_scout" if selected_beats_validation and oos_beats_baseline else "baseline_no_deep"
    metrics = candidate_metrics if promoted_variant == "cash_mamba_scout" else baseline_oos

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "cash_mamba_scout.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "seq_cols": seq_cols,
            "norm": norm,
            "selected_config": asdict(selected),
            "parent_model": str(args.parent_model),
            "train_scope": "v40_6_parent_cash_only",
        },
        model_path,
    )
    manifest_path = args.out_dir / "cash_mamba_scout_v43_manifest.json"
    manifest = {
        "model_id": MODEL_ID,
        "parent_model": str(args.parent_model),
        "parent_contract": "v40_6_no_maxhold_no_cooldown",
        "scout_model": str(model_path),
        "mamba_backend": str(args.mamba_backend),
        "scout_train_scope": "v40_6_cash_states_only",
        "selected_config": asdict(selected),
        "selected_beats_validation_baseline": bool(selected_beats_validation),
        "candidate_beats_oos_baseline": bool(oos_beats_baseline),
        "promoted_variant": promoted_variant,
        "metrics": metrics,
        "candidate_metrics": candidate_metrics,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    pd.DataFrame(
        [
            {
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
    ).sort_values("score", ascending=False).to_csv(args.grid_out, index=False)

    feature_audit_cols = [
        c
        for c in list(parent_bundle.get("feature_cols") or [])
        if not c.startswith("macro_factor_") and not c.startswith("micro_factor_")
    ]
    feature_audit = _audit_contract(train_all, eval_df, feature_audit_cols)
    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit.get("status") != "pass":
        blocking.extend(feature_audit.get("blocking", []))
    if forbidden_seq_cols:
        blocking.append(f"forbidden_sequence_columns={forbidden_seq_cols}")
    warnings.extend(feature_audit.get("warnings", []))
    if not selected_beats_validation:
        warnings.append("cash_mamba_scout_rejected_by_validation_baseline_gate")
    if not oos_beats_baseline:
        warnings.append("cash_mamba_scout_rejected_by_oos_baseline_gate")
    if candidate_metrics["cost1"]["pnl"] <= baseline_oos["cost1"]["pnl"]:
        warnings.append("cash_mamba_scout_did_not_beat_nohold_baseline_cost1")
    if any("max_hold" in k for k in candidate_metrics["cost1"].get("exits", {})):
        blocking.append("effective_max_hold_exit_detected")
    if any("cooldown" in k for k in candidate_metrics["cost1"].get("actions", {})):
        blocking.append("effective_cooldown_action_detected")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if promoted_variant == "cash_mamba_scout" and not blocking else "reject",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after scout selection",
        "parent_contract": "v40_6_no_maxhold_no_cooldown",
        "scout_train_scope": "v40_6_parent_cash_only",
        "scout_architecture": "NativeMambaScout" if args.mamba_backend == "native" else "MambaStyleSSM",
        "mamba_backend": str(args.mamba_backend),
        "stage1_stage2_parent": "v40_6_target_aware_full_parent_retained",
        "dsac_sniper": "deferred_not_wired_in_this_backtest",
        "train_snapshot_count": int(len(train_ds["target"])),
        "forbidden_sequence_columns": forbidden_seq_cols,
        "feature_audit": feature_audit,
        "baseline_selection_score": float(baseline_selection_score),
        "scout_selection_score": float(best["selection_score"]),
        "selected_beats_validation_baseline": bool(selected_beats_validation),
        "candidate_beats_oos_baseline": bool(oos_beats_baseline),
        "promoted_variant": promoted_variant,
        "selected_config": asdict(selected),
        "baseline_oos": baseline_oos,
        "candidate_metrics": candidate_metrics,
        "metrics": metrics,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "V43 keeps v40.6 no-max-hold/no-cooldown as the cascade parent and retrains the Deep Scout as a Mamba-style sequence model only on timestamps where v40.6 parent is CASH. Scout is attached as a CASH-only residual sleeve. DSAC Sniper execution is not wired in this backtest because current DSAC checkpoints require a separate schema audit.",
        "parent_model": str(args.parent_model),
        "parent_report": str(args.parent_report),
        "encoding_meta": encoding_meta,
        "training_meta": training_meta,
        "seq_cols": seq_cols,
        "model": str(model_path),
        "baseline_validation": baseline_val,
        "baseline_selection_score": float(baseline_selection_score),
        "selection_result": best,
        "selected_config": asdict(selected),
        "selected_beats_validation_baseline": bool(selected_beats_validation),
        "candidate_beats_oos_baseline": bool(oos_beats_baseline),
        "promoted_variant": promoted_variant,
        "baseline_oos": baseline_oos,
        "candidate_metrics": candidate_metrics,
        "metrics": metrics,
        "audit": audit,
        "artifacts": {
            "model": str(model_path),
            "manifest": str(manifest_path),
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
                "model": str(model_path),
                "selected_config": asdict(selected),
                "promoted_variant": promoted_variant,
                "baseline_oos": baseline_oos,
                "candidate_metrics": candidate_metrics,
                "metrics": metrics,
                "verdict": audit["verdict"],
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
