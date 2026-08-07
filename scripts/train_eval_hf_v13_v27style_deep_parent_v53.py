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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    ACTION_LONG,
    ACTION_SHORT,
    FullyLearnedGovernorConfig,
    build_training_set,
    predict_policy_frame,
    prepare_features,
)
from scripts.eval_hf_v13_frozen_v27_rule_exit_overlay_v31 import (  # noqa: E402
    DEFAULT_JACKPOT,
    DEFAULT_PARENT,
    DEFAULT_V27,
    OverlayConfig,
    _grid as v31_overlay_grid,
    _load_v27,
    _predict_all as predict_v27_all,
    _score as v31_score,
    backtest as v31_backtest,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import (  # noqa: E402
    _audit_contract,
    _close,
    _feature_cols,
    _read,
)
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "hf_v13_v27style_deep_parent_v53_20260513"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_v27style_deep_parent_v53_20260513"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_v27style_deep_parent_v53_20260513_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_v27style_deep_parent_v53_20260513_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_v27style_deep_parent_v53_20260513_grid.csv"

SEQ_LEN = 72
V31_COST1 = 277.0679629973942
V31_COST2 = 112.79326141840412
V31_COST3 = 20.933695032758784


@dataclass(frozen=True)
class GateConfig:
    name: str
    confidence: float
    quality_floor: float
    margin: float
    cash_bias: float = 0.0


class V27StyleDeepParent(nn.Module):
    """Action-only parent encoder inspired by V27, with context-preserving readout."""

    def __init__(self, input_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Conv1d(hidden, hidden, 3, padding=8, dilation=8),
            nn.GELU(),
        )
        self.attn = nn.Sequential(nn.Conv1d(hidden, hidden // 2, 1), nn.Tanh(), nn.Conv1d(hidden // 2, 1, 1))
        self.readout = nn.Sequential(nn.Linear(hidden * 3, 128), nn.GELU(), nn.Dropout(0.10))
        self.action_head = nn.Linear(128, 3)
        self.quality_head = nn.Linear(128, 1)

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(seq.transpose(1, 2))
        avg = torch.mean(h, dim=-1)
        last = h[:, :, -1]
        w = torch.softmax(self.attn(h), dim=-1)
        attn = torch.sum(h * w, dim=-1)
        z = self.readout(torch.cat([avg, last, attn], dim=1))
        return self.action_head(z), self.quality_head(z).squeeze(-1)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _parent_cfg(base: dict[str, Any]) -> FullyLearnedGovernorConfig:
    cfg = dict(base)
    cfg.update(
        {
            "notional_buckets": tuple(cfg.get("notional_buckets", (0.23, 0.368, 0.575, 0.8625, 1.2075, 1.6675, 2.3, 3.105, 4.14))),
            "leverage_buckets": tuple(cfg.get("leverage_buckets", (1.5, 2.0, 3.0, 4.0, 5.0))),
            "take_profit_buckets": tuple(cfg.get("take_profit_buckets", (0.007, 0.011, 0.018, 0.030, 0.050, 0.090, 0.180, 0.450, 0.900))),
            "stop_loss_buckets": tuple(cfg.get("stop_loss_buckets", (0.004, 0.006, 0.009, 0.014, 0.022, 0.035, 0.055))),
            "max_hold_buckets": tuple(cfg.get("max_hold_buckets", (6, 12, 24, 48, 96, 192, 288))),
            "cooldown_buckets": tuple(cfg.get("cooldown_buckets", (0, 1, 3, 6, 12, 24, 48))),
        }
    )
    return FullyLearnedGovernorConfig(**cfg)


def _gate_grid() -> list[GateConfig]:
    rows: list[GateConfig] = []
    for conf in (0.44, 0.50, 0.56, 0.62):
        for qf in (-0.005, 0.000, 0.010, 0.020):
            rows.append(GateConfig(f"v53_c{conf:.2f}_q{qf:.3f}_m0.04", conf, qf, 0.04))
    rows.append(GateConfig("v53_precision_c0.66_q0.02_m0.08", 0.66, 0.020, 0.08))
    rows.append(GateConfig("v53_active_c0.42_q-0.01_m0.02", 0.42, -0.010, 0.02))
    return rows


def _seq_tensor(features: pd.DataFrame, indices: np.ndarray, cols: list[str]) -> np.ndarray:
    arr = features.loc[:, cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    pad = np.zeros((SEQ_LEN - 1, arr.shape[1]), dtype=np.float32)
    padded = np.vstack([pad, arr])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=SEQ_LEN, axis=0)
    if windows.shape[1] == arr.shape[1]:
        windows = windows.transpose(0, 2, 1)
    return np.ascontiguousarray(windows[indices])


def _normalizer(seq: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": np.nanmean(seq, axis=(0, 1)).astype(np.float32),
        "std": (np.nanstd(seq, axis=(0, 1)) + 1e-6).astype(np.float32),
    }


def _apply_norm(seq: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    return ((seq - norm["mean"][None, None, :]) / norm["std"][None, None, :]).astype(np.float32)


def _fit_model(
    seq: np.ndarray,
    y: dict[str, np.ndarray],
    norm: dict[str, np.ndarray],
    *,
    epochs: int,
    seed: int,
) -> V27StyleDeepParent:
    torch.manual_seed(int(seed))
    x = _apply_norm(seq, norm)
    action = np.asarray(y["action"], dtype=np.int64)
    quality = np.asarray(y["quality"], dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V27StyleDeepParent(x.shape[-1]).to(device)
    counts = np.bincount(action, minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights[0] *= 0.35
    weights = weights / max(weights.mean(), 1e-6)
    ce = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    huber = nn.SmoothL1Loss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(action), torch.from_numpy(quality)), batch_size=256, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    model.train()
    for ep in range(int(epochs)):
        loss_sum = 0.0
        for xb, ab, qb in loader:
            xb, ab, qb = xb.to(device), ab.to(device), qb.to(device)
            logits, qhat = model(xb)
            loss = ce(logits, ab) + 2.0 * huber(qhat, qb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.detach().cpu())
        if ep == 0 or (ep + 1) % 20 == 0 or ep + 1 == int(epochs):
            print(f"[{MODEL_ID}] epoch {ep + 1}/{epochs} loss={loss_sum / max(len(loader), 1):.6f}", flush=True)
    return model.cpu().eval()


def _predict_all(model: V27StyleDeepParent, features: pd.DataFrame, cols: list[str], norm: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    seq = _seq_tensor(features, np.arange(len(features), dtype=np.int64), cols)
    x = _apply_norm(seq, norm)
    probs: list[np.ndarray] = []
    qvals: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            logits, qhat = model(torch.from_numpy(x[start : start + 1024]))
            probs.append(torch.softmax(logits, dim=1).numpy())
            qvals.append(qhat.numpy())
    return {"action_proba": np.vstack(probs), "quality": np.concatenate(qvals)}


def _override_decisions(base_dec: pd.DataFrame, pred: dict[str, np.ndarray], gate: GateConfig) -> pd.DataFrame:
    dec = base_dec.copy()
    proba = pred["action_proba"]
    q = pred["quality"]
    p_cash = proba[:, ACTION_CASH] + float(gate.cash_bias)
    p_long = proba[:, ACTION_LONG]
    p_short = proba[:, ACTION_SHORT]
    side_score = np.maximum(p_long, p_short)
    margin = np.abs(p_long - p_short)
    side = np.where(p_long >= p_short, 1, -1)
    action = np.where(side > 0, ACTION_LONG, ACTION_SHORT)
    trade = (side_score >= float(gate.confidence)) & (q >= float(gate.quality_floor)) & (margin >= float(gate.margin)) & (side_score > p_cash)
    dec["action"] = np.where(trade, action, ACTION_CASH).astype(np.int64)
    dec["side"] = np.where(trade, side, 0).astype(np.int64)
    dec["quality_score"] = q.astype(np.float64)
    dec["confidence"] = np.maximum.reduce([p_cash, p_long, p_short]).astype(np.float64)
    cash = ~trade
    dec.loc[cash, ["side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    dec.loc[cash, "leverage"] = 1.0
    return dec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V53 V27-style deep action-only parent replacement for V31 stack.")
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=DEFAULT_JACKPOT)
    p.add_argument("--v27-model", type=Path, default=DEFAULT_V27)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--seed", type=int, default=2053)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[{MODEL_ID}] loading data/models", flush=True)
    parent = joblib.load(args.parent_model)
    jackpot_payload = joblib.load(args.jackpot_model)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = _load_v27(args.v27_model)
    base_cfg = dict(parent["config"])
    cfg = _parent_cfg(base_cfg)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_cols = _feature_cols(train_all, eval_df)
    audit_base = _audit_contract(train_all, eval_df, feature_cols)
    print(f"[{MODEL_ID}] building parent labels", flush=True)
    _, y, meta = build_training_set(train, cfg=cfg, stride_bars=12, batch_size=512, feature_cols=feature_cols)
    valid = np.arange(0, max(0, len(train) - cfg.max_train_horizon_bars - 1), 12, dtype=np.int64)
    train_features = prepare_features(train, side_hint=0, close=_close(train), feature_cols=feature_cols)
    val_features = prepare_features(val, side_hint=0, close=_close(val), feature_cols=feature_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    print(f"[{MODEL_ID}] building V27-style parent sequence tensor rows={len(valid)} cols={len(feature_cols)}", flush=True)
    seq = _seq_tensor(train_features, valid, feature_cols)
    norm = _normalizer(seq)
    print(f"[{MODEL_ID}] training deep action-only parent on {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}", flush=True)
    model = _fit_model(seq, y, norm, epochs=args.epochs, seed=args.seed)
    print(f"[{MODEL_ID}] predicting parent gates and frozen V27 scout", flush=True)
    val_pred = _predict_all(model, val_features, feature_cols, norm)
    eval_pred = _predict_all(model, eval_features, feature_cols, norm)
    val_base_dec = predict_policy_frame(parent, val, close=_close(val))
    eval_base_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    val_v27_q = predict_v27_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_v27_q = predict_v27_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for gate in _gate_grid():
        val_dec = _override_decisions(val_base_dec, val_pred, gate)
        for overlay in v31_overlay_grid():
            print(f"[{MODEL_ID}] validation gate={gate.name} overlay={overlay.name}", flush=True)
            v1 = v31_backtest(val, parent, jackpot_model, add_cfg, val_v27_q, overlay, fee=float(base_cfg["fee"]), slip=float(base_cfg["slip"]), cost_mult=1.0, decisions=val_dec)
            v2 = v31_backtest(val, parent, jackpot_model, add_cfg, val_v27_q, overlay, fee=float(base_cfg["fee"]), slip=float(base_cfg["slip"]), cost_mult=2.0, decisions=val_dec)
            v3 = v31_backtest(val, parent, jackpot_model, add_cfg, val_v27_q, overlay, fee=float(base_cfg["fee"]), slip=float(base_cfg["slip"]), cost_mult=3.0, decisions=val_dec)
            row = {
                "gate": asdict(gate),
                "overlay": asdict(overlay),
                "validation_cost1": v1,
                "validation_cost2": v2,
                "validation_cost3": v3,
                "selection_score": v31_score(v1, v2, v3),
            }
            rows.append(row)
            if best is None or row["selection_score"] > best["selection_score"]:
                best = row
    assert best is not None
    selected_gate = GateConfig(**best["gate"])
    selected_overlay = OverlayConfig(**best["overlay"])
    eval_dec = _override_decisions(eval_base_dec, eval_pred, selected_gate)
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        r = v31_backtest(eval_df, parent, jackpot_model, add_cfg, eval_v27_q, selected_overlay, fee=float(base_cfg["fee"]), slip=float(base_cfg["slip"]), cost_mult=float(mult), decisions=eval_dec, record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            lp = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            lp.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(lp, index=False)
            ledgers["cost1"] = str(lp)
        metrics[f"cost{mult}"] = r
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "v53_v27style_deep_parent.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "norm": norm,
            "seq_len": SEQ_LEN,
            "selected_gate": asdict(selected_gate),
            "selected_overlay": asdict(selected_overlay),
            "parent_model": str(args.parent_model),
            "jackpot_model": str(args.jackpot_model),
            "v27_model": str(args.v27_model),
            "training_meta": meta,
        },
        model_path,
    )
    pd.DataFrame(
        [
            {
                **{f"gate_{k}": v for k, v in r["gate"].items()},
                **{f"overlay_{k}": v for k, v in r["overlay"].items()},
                "selection_score": r["selection_score"],
                "val_pnl": r["validation_cost1"]["pnl"],
                "val_mdd": r["validation_cost1"]["mdd"],
                "val_trades": r["validation_cost1"]["trades"],
                "val_deep_entries": r["validation_cost1"].get("deep_entries", 0),
                "val_cost2_pnl": r["validation_cost2"]["pnl"],
                "val_cost3_pnl": r["validation_cost3"]["pnl"],
            }
            for r in rows
        ]
    ).to_csv(args.grid_out, index=False)
    blocking = list(audit_base.get("blocking", []))
    warnings = list(audit_base.get("warnings", []))
    if metrics["cost1"]["pnl"] <= V31_COST1:
        warnings.append("oos_cost1_did_not_beat_v31")
    if metrics["cost2"]["pnl"] <= 0.0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0.0:
        warnings.append("cost3_not_survived")
    verdict = "promote" if not blocking and metrics["cost1"]["pnl"] > V31_COST1 and metrics["cost2"]["pnl"] > 0 and metrics["cost3"]["pnl"] > 0 else "iterate"
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": verdict,
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "policy": "v27style_deep_action_parent_replacement_for_v31_stack",
        "parent_bucket_heads_reused": True,
        "v21_2_preserved": True,
        "frozen_v27_residual_scout_preserved": True,
        "feature_audit": audit_base,
        "selected_gate": asdict(selected_gate),
        "selected_overlay": asdict(selected_overlay),
        "metrics": metrics,
        "baseline_v31": {"cost1": V31_COST1, "cost2": V31_COST2, "cost3": V31_COST3},
    }
    report = {
        "model_id": MODEL_ID,
        "design": "V27-style action-only deep parent. A causal TCN reads 72 bars of parent features and combines global average, last-step, and attention readouts. It replaces only the parent action/quality decision. Existing parent bucket heads provide notional/leverage/TP/SL/hold/cooldown; V21.2 add-on and frozen V27 residual scout are preserved.",
        "split_policy": "Train action labels on 2025 Jan-Sep; select gate/overlay on 2025 Q4; evaluate fixed 2026 OOS after selection.",
        "feature_count": len(feature_cols),
        "training_meta": meta,
        "label_distribution": {k: pd.Series(v).value_counts().sort_index().to_dict() for k, v in y.items() if k != "quality"},
        "selected_gate": asdict(selected_gate),
        "selected_overlay": asdict(selected_overlay),
        "selection_result": best,
        "metrics": metrics,
        "audit": audit,
        "artifacts": {
            "model": str(model_path),
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
                "model": str(model_path),
                "selected_gate": asdict(selected_gate),
                "selected_overlay": asdict(selected_overlay),
                "metrics": metrics,
                "verdict": verdict,
            },
            ensure_ascii=False,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
