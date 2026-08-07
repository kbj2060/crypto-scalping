#!/usr/bin/env python3
from __future__ import annotations

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

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, ACTION_SHORT, predict_policy_frame, prepare_features  # noqa: E402
from scripts import eval_alpha1_rl_exit_and_sizing_20260513 as alpha1  # noqa: E402
from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as candidate  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _feature_cols, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_deep_entry_parent_lite_v38 import DeepEntryParentLite, _apply_norm, _normalizer  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha1_teacher_constrained_deep_parent_verify_20260513"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha1_teacher_constrained_deep_parent_20260513_verify.json"
SEEDS = (20260513, 20260514, 20260515)
EPOCHS = 35


def _train_with_seed(seq: np.ndarray, action: np.ndarray, quality: np.ndarray, notional: np.ndarray, *, n_buckets: int, seed: int) -> tuple[DeepEntryParentLite, dict[str, Any]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    norm = _normalizer(seq)
    x = _apply_norm(seq, norm)
    device = candidate._device()
    model = DeepEntryParentLite(x.shape[-1], notional_classes=int(n_buckets)).to(device)
    counts = np.bincount(action, minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights[0] *= 0.25
    weights = weights / max(float(weights.mean()), 1e-6)
    ce_action = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    ce_size = nn.CrossEntropyLoss()
    huber = nn.SmoothL1Loss()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(action.astype(np.int64)), torch.from_numpy(quality.astype(np.float32)), torch.from_numpy(notional.astype(np.int64))),
        batch_size=256,
        shuffle=True,
        generator=generator,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    model.train()
    for ep in range(EPOCHS):
        loss_sum = 0.0
        for xb, ab, qb, nb in loader:
            xb, ab, qb, nb = xb.to(device), ab.to(device), qb.to(device), nb.to(device)
            logits, qhat, nlogits = model(xb)
            active = ab != ACTION_CASH
            loss = ce_action(logits, ab) + 1.2 * huber(qhat, qb)
            if torch.any(active):
                loss = loss + 0.25 * ce_size(nlogits[active], nb[active])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.detach().cpu())
        if ep in {0, EPOCHS - 1} or (ep + 1) % 10 == 0:
            print(f"[{MODEL_ID}] seed={seed} epoch={ep+1} loss={loss_sum/max(len(loader),1):.5f}", flush=True)
    return model.cpu().eval(), {"norm": norm, "label_counts": {str(i): int(v) for i, v in enumerate(counts)}, "epochs": int(EPOCHS), "seed": int(seed)}


def _metrics(df: pd.DataFrame, q: np.ndarray, decisions: pd.DataFrame, parent: dict[str, Any], jackpot_model: dict[str, Any], add_cfg: CostRunnerConfig, base: dict[str, Any]) -> dict[str, Any]:
    return {
        f"cost{mult}": alpha1.backtest_alpha1(df, parent, jackpot_model, add_cfg, q, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=float(mult), decisions=decisions)
        for mult in (1, 2, 3)
    }


def _slice_period(df: pd.DataFrame, q: np.ndarray, decisions: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    ts = pd.to_datetime(df["timestamp"])
    mask = (ts >= pd.Timestamp(start)) & (ts < pd.Timestamp(end))
    idx = np.flatnonzero(mask.to_numpy())
    return df.iloc[idx].reset_index(drop=True), q[idx], decisions.iloc[idx].reset_index(drop=True)


def main() -> int:
    print(f"[{MODEL_ID}] loading baseline artifacts", flush=True)
    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    base = dict(parent["config"])
    buckets = tuple(base.get("notional_buckets", (0.23, 0.368, 0.575, 0.8625, 1.2075, 1.6675, 2.3, 3.105, 4.14)))

    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_cols = _feature_cols(train_all, eval_df)
    parent_audit = _audit_contract(train_all, eval_df, list(parent.get("feature_cols") or []))

    train_dec = predict_policy_frame(parent, train, close=_close(train))
    val_dec = predict_policy_frame(parent, val, close=_close(val))
    eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))

    train_features = prepare_features(train, side_hint=0, close=_close(train), feature_cols=feature_cols)
    train_seq = candidate._seq_tensor(train_features, np.arange(len(train), dtype=np.int64), feature_cols)
    y_action = train_dec["action"].astype(int).to_numpy(dtype=np.int64)
    y_quality = pd.to_numeric(train_dec["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y_notional = candidate._bucket_labels(train_dec, buckets)

    print(f"[{MODEL_ID}] computing V27 q", flush=True)
    val_q = candidate._predict_v27_fast(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = candidate._predict_v27_fast(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    alpha1_eval = _metrics(eval_df, eval_q, eval_dec, parent, jackpot_model, add_cfg, base)
    monthly_alpha1: dict[str, Any] = {}
    for label, start, end in (("2026-01", "2026-01-01", "2026-02-01"), ("2026-02", "2026-02-01", "2026-03-01")):
        mdf, mq, mdec = _slice_period(eval_df, eval_q, eval_dec, start, end)
        monthly_alpha1[label] = _metrics(mdf, mq, mdec, parent, jackpot_model, add_cfg, base)

    runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        model, meta = _train_with_seed(train_seq, y_action, y_quality, y_notional, n_buckets=len(buckets), seed=seed)
        val_features = prepare_features(val, side_hint=0, close=_close(val), feature_cols=feature_cols)
        eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
        val_pred = candidate._predict_deep(model, val_features, feature_cols, meta["norm"])
        eval_pred = candidate._predict_deep(model, eval_features, feature_cols, meta["norm"])

        grid_rows: list[dict[str, Any]] = []
        selected = None
        best_score = -1e18
        for rt in candidate._grid():
            dec = candidate._constrained_decisions(val_dec, val_pred, buckets, rt)
            vm = _metrics(val, val_q, dec, parent, jackpot_model, add_cfg, base)
            score = candidate._score(vm["cost1"], vm["cost2"], vm["cost3"])
            row = {**asdict(rt), "score": score, "val_pnl": vm["cost1"]["pnl"], "val_mdd": vm["cost1"]["mdd"], "val_c2_pnl": vm["cost2"]["pnl"], "val_c3_pnl": vm["cost3"]["pnl"], "val_trades": vm["cost1"]["trades"], "val_deep_entries": vm["cost1"]["deep_entries"]}
            grid_rows.append(row)
            if score > best_score:
                best_score = score
                selected = rt
        assert selected is not None
        eval_dec2 = candidate._constrained_decisions(eval_dec, eval_pred, buckets, selected)
        eval_metrics = _metrics(eval_df, eval_q, eval_dec2, parent, jackpot_model, add_cfg, base)
        monthly: dict[str, Any] = {}
        for label, start, end in (("2026-01", "2026-01-01", "2026-02-01"), ("2026-02", "2026-02-01", "2026-03-01")):
            mdf, mq, mdec = _slice_period(eval_df, eval_q, eval_dec2, start, end)
            monthly[label] = _metrics(mdf, mq, mdec, parent, jackpot_model, add_cfg, base)
        print(
            f"[{MODEL_ID}] seed={seed} selected={selected.name} val={grid_rows[0]['val_pnl']:.2f} "
            f"oos={eval_metrics['cost1']['pnl']:.2f} c2={eval_metrics['cost2']['pnl']:.2f} c3={eval_metrics['cost3']['pnl']:.2f}",
            flush=True,
        )
        runs.append({"seed": seed, "selected_config": asdict(selected), "best_val": max(grid_rows, key=lambda r: r["score"]), "eval": eval_metrics, "monthly": monthly, "grid": sorted(grid_rows, key=lambda r: r["score"], reverse=True)[:5]})

    cost1 = np.asarray([r["eval"]["cost1"]["pnl"] for r in runs], dtype=float)
    cost2 = np.asarray([r["eval"]["cost2"]["pnl"] for r in runs], dtype=float)
    cost3 = np.asarray([r["eval"]["cost3"]["pnl"] for r in runs], dtype=float)
    report = {
        "model_id": MODEL_ID,
        "baseline_alpha1": alpha1_eval,
        "monthly_alpha1": monthly_alpha1,
        "runs": runs,
        "aggregate": {
            "seeds": list(SEEDS),
            "cost1_mean": float(cost1.mean()),
            "cost1_min": float(cost1.min()),
            "cost1_max": float(cost1.max()),
            "cost2_mean": float(cost2.mean()),
            "cost3_mean": float(cost3.mean()),
            "beats_alpha1_cost1_count": int(np.sum(cost1 > float(alpha1_eval["cost1"]["pnl"]))),
            "beats_alpha1_cost2_count": int(np.sum(cost2 > float(alpha1_eval["cost2"]["pnl"]))),
            "beats_alpha1_cost3_count": int(np.sum(cost3 > float(alpha1_eval["cost3"]["pnl"]))),
        },
        "audit": {
            "status": "pass" if not parent_audit.get("blocking") else "fail",
            "blocking": parent_audit.get("blocking", []),
            "warnings": parent_audit.get("warnings", []),
            "selection_uses_2026": False,
            "cash_preserving": True,
            "new_parent_entries_allowed_in_teacher_cash": False,
            "v27_deep_scout_preserved": True,
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "aggregate": report["aggregate"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
