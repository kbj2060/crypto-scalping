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
    _bucket_or_default_batch,
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
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _feature_cols, _fill_price, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "hf_v13_v27style_utility_parent_v53_1_20260513"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_v27style_utility_parent_v53_1_20260513"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_v27style_utility_parent_v53_1_20260513_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_v27style_utility_parent_v53_1_20260513_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_v27style_utility_parent_v53_1_20260513_grid.csv"

SEQ_LEN = 72
HORIZONS = (12, 24, 48, 96)
V31_COST1 = 277.0679629973942
V31_COST2 = 112.79326141840412
V31_COST3 = 20.933695032758784


@dataclass(frozen=True)
class UtilityGate:
    name: str
    edge_th: float
    margin_th: float
    min_quality: float


class V27StyleUtilityParent(nn.Module):
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
        self.head = nn.Sequential(nn.Linear(hidden * 3, 128), nn.GELU(), nn.Dropout(0.10), nn.Linear(128, 2))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.net(seq.transpose(1, 2))
        avg = torch.mean(h, dim=-1)
        last = h[:, :, -1]
        w = torch.softmax(self.attn(h), dim=-1)
        attn = torch.sum(h * w, dim=-1)
        return self.head(torch.cat([avg, last, attn], dim=1))


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
    for key, default in {
        "notional_buckets": (0.23, 0.368, 0.575, 0.8625, 1.2075, 1.6675, 2.3, 3.105, 4.14),
        "leverage_buckets": (1.5, 2.0, 3.0, 4.0, 5.0),
        "take_profit_buckets": (0.007, 0.011, 0.018, 0.030, 0.050, 0.090, 0.180, 0.450, 0.900),
        "stop_loss_buckets": (0.004, 0.006, 0.009, 0.014, 0.022, 0.035, 0.055),
        "max_hold_buckets": (6, 12, 24, 48, 96, 192, 288),
        "cooldown_buckets": (0, 1, 3, 6, 12, 24, 48),
    }.items():
        cfg[key] = tuple(cfg.get(key, default))
    return FullyLearnedGovernorConfig(**cfg)


def _gate_grid() -> list[UtilityGate]:
    rows: list[UtilityGate] = []
    for edge in (0.002, 0.004, 0.006, 0.008, 0.010, 0.012):
        for margin in (0.0005, 0.0015, 0.0030, 0.0050):
            rows.append(UtilityGate(f"v53_1_e{edge:.3f}_m{margin:.4f}", edge, margin, -99.0))
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
    return {"mean": np.nanmean(seq, axis=(0, 1)).astype(np.float32), "std": (np.nanstd(seq, axis=(0, 1)) + 1e-6).astype(np.float32)}


def _apply_norm(seq: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    return ((seq - norm["mean"][None, None, :]) / norm["std"][None, None, :]).astype(np.float32)


def _build_utility_targets(df: pd.DataFrame, feature_frame: pd.DataFrame, cols: list[str], *, fee: float, slip: float, stride: int) -> dict[str, np.ndarray]:
    close = _close(df)
    idx = np.arange(SEQ_LEN, max(SEQ_LEN, len(df) - max(HORIZONS) - 2), int(stride), dtype=np.int64)
    seq = _seq_tensor(feature_frame, idx, cols)
    target = np.zeros((len(idx), 2), dtype=np.float32)
    for j, i in enumerate(idx):
        entry_i = min(i + 1, len(df) - 1)
        le = _fill_price(df, entry_i, 1, slip, entry=True)
        se = _fill_price(df, entry_i, -1, slip, entry=True)
        long_rewards: list[float] = []
        short_rewards: list[float] = []
        for h in HORIZONS:
            exit_i = min(i + h, len(df) - 1)
            lx = _fill_price(df, exit_i, 1, slip, entry=False)
            sx = _fill_price(df, exit_i, -1, slip, entry=False)
            long_rewards.append((lx - le) / max(le, 1e-12) - 2.0 * fee)
            short_rewards.append((se - sx) / max(se, 1e-12) - 2.0 * fee)
        target[j, 0] = float(max(long_rewards))
        target[j, 1] = float(max(short_rewards))
    return {"idx": idx, "seq": seq, "target": target}


def _fit_model(ds: dict[str, np.ndarray], norm: dict[str, np.ndarray], *, epochs: int, seed: int) -> V27StyleUtilityParent:
    torch.manual_seed(int(seed))
    x = _apply_norm(ds["seq"], norm)
    y = ds["target"].astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V27StyleUtilityParent(x.shape[-1]).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    for ep in range(int(epochs)):
        loss_sum = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.detach().cpu())
        if ep == 0 or (ep + 1) % 20 == 0 or ep + 1 == int(epochs):
            print(f"[{MODEL_ID}] epoch {ep + 1}/{epochs} loss={loss_sum / max(len(loader), 1):.6f}", flush=True)
    return model.cpu().eval()


def _predict_all(model: V27StyleUtilityParent, features: pd.DataFrame, cols: list[str], norm: dict[str, np.ndarray]) -> np.ndarray:
    seq = _seq_tensor(features, np.arange(len(features), dtype=np.int64), cols)
    x = _apply_norm(seq, norm)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            out.append(model(torch.from_numpy(x[start : start + 1024])).numpy())
    return np.vstack(out).astype(np.float32)


def _bucket_frame(parent: dict[str, Any], frame: pd.DataFrame, side: np.ndarray) -> pd.DataFrame:
    cfg = FullyLearnedGovernorConfig(**dict(parent.get("config", {})))
    feature_cols = list(parent.get("feature_cols") or [])
    close = _close(frame)
    if feature_cols and set(feature_cols).issubset(frame.columns):
        x = frame.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan).copy()
        if "side_hint" in x.columns:
            x["side_hint"] = 0.0
    else:
        x = prepare_features(frame, side_hint=0, close=close, feature_cols=feature_cols)
    x_side = x.copy()
    x_side["side_hint"] = np.asarray(side, dtype=np.float64)
    notional, c1 = _bucket_or_default_batch(parent, "notional", x_side, cfg.notional_buckets)
    leverage, c2 = _bucket_or_default_batch(parent, "leverage", x_side, cfg.leverage_buckets)
    take_profit, c3 = _bucket_or_default_batch(parent, "take_profit", x_side, cfg.take_profit_buckets)
    stop_loss, c4 = _bucket_or_default_batch(parent, "stop_loss", x_side, cfg.stop_loss_buckets)
    max_hold, c5 = _bucket_or_default_batch(parent, "max_hold", x_side, tuple(float(v) for v in cfg.max_hold_buckets))
    cooldown, c6 = _bucket_or_default_batch(parent, "cooldown", x_side, tuple(float(v) for v in cfg.cooldown_buckets))
    leverage = np.clip(leverage, min(cfg.leverage_buckets), max(cfg.leverage_buckets))
    notional = np.clip(notional, min(cfg.notional_buckets), max(cfg.notional_buckets))
    fraction = np.clip(notional / np.maximum(leverage, 1e-8), 0.0, cfg.max_margin_fraction)
    notional = fraction * leverage
    confidence = np.mean(np.vstack([c1, c2, c3, c4, c5, c6]), axis=0)
    return pd.DataFrame(
        {
            "notional_exposure": notional.astype(np.float64),
            "leverage": leverage.astype(np.float64),
            "position_fraction": fraction.astype(np.float64),
            "take_profit": take_profit.astype(np.float64),
            "stop_loss": stop_loss.astype(np.float64),
            "max_hold_bars": np.rint(max_hold).astype(np.int64),
            "cooldown_bars": np.rint(cooldown).astype(np.int64),
            "bucket_confidence": confidence.astype(np.float64),
        },
        index=frame.index,
    )


def _override_decisions(parent: dict[str, Any], frame: pd.DataFrame, q: np.ndarray, gate: UtilityGate) -> pd.DataFrame:
    base_dec = predict_policy_frame(parent, frame, close=_close(frame))
    dec = base_dec.copy()
    ql = q[:, 0]
    qs = q[:, 1]
    edge = np.maximum(ql, qs)
    margin = np.abs(ql - qs)
    side = np.where(ql >= qs, 1, -1)
    action = np.where(side > 0, ACTION_LONG, ACTION_SHORT)
    trade = (edge >= float(gate.edge_th)) & (margin >= float(gate.margin_th)) & (edge >= float(gate.min_quality))
    dec["action"] = np.where(trade, action, ACTION_CASH).astype(np.int64)
    dec["side"] = np.where(trade, side, 0).astype(np.int64)
    dec["quality_score"] = edge.astype(np.float64)
    dec["confidence"] = margin.astype(np.float64)
    dec["deep_parent_q_long"] = ql.astype(np.float64)
    dec["deep_parent_q_short"] = qs.astype(np.float64)
    bucket_dec = _bucket_frame(parent, frame, side)
    cols = ["notional_exposure", "leverage", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]
    dec.loc[trade, cols] = bucket_dec.loc[trade, cols]
    dec.loc[trade, "confidence"] = np.maximum(dec.loc[trade, "confidence"].to_numpy(dtype=np.float64), bucket_dec.loc[trade, "bucket_confidence"].to_numpy(dtype=np.float64))
    cash = ~trade
    dec.loc[cash, ["side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    dec.loc[cash, "leverage"] = 1.0
    return dec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V53.1 V27-style utility parent replacement for V31 stack.")
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=DEFAULT_JACKPOT)
    p.add_argument("--v27-model", type=Path, default=DEFAULT_V27)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--seed", type=int, default=20531)
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
    train_features = prepare_features(train, side_hint=0, close=_close(train), feature_cols=feature_cols)
    val_features = prepare_features(val, side_hint=0, close=_close(val), feature_cols=feature_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    print(f"[{MODEL_ID}] building utility targets", flush=True)
    ds = _build_utility_targets(train, train_features, feature_cols, fee=float(cfg.fee), slip=float(cfg.slip), stride=3)
    norm = _normalizer(ds["seq"])
    print(f"[{MODEL_ID}] training utility parent rows={len(ds['target'])} cols={len(feature_cols)} on {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}", flush=True)
    model = _fit_model(ds, norm, epochs=args.epochs, seed=args.seed)
    print(f"[{MODEL_ID}] predicting parent utilities and frozen V27 scout", flush=True)
    val_q_parent = _predict_all(model, val_features, feature_cols, norm)
    eval_q_parent = _predict_all(model, eval_features, feature_cols, norm)
    val_v27_q = predict_v27_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_v27_q = predict_v27_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for gate in _gate_grid():
        val_dec = _override_decisions(parent, val, val_q_parent, gate)
        for overlay in v31_overlay_grid():
            print(f"[{MODEL_ID}] validation gate={gate.name} overlay={overlay.name}", flush=True)
            v1 = v31_backtest(val, parent, jackpot_model, add_cfg, val_v27_q, overlay, fee=float(base_cfg["fee"]), slip=float(base_cfg["slip"]), cost_mult=1.0, decisions=val_dec)
            v2 = v31_backtest(val, parent, jackpot_model, add_cfg, val_v27_q, overlay, fee=float(base_cfg["fee"]), slip=float(base_cfg["slip"]), cost_mult=2.0, decisions=val_dec)
            v3 = v31_backtest(val, parent, jackpot_model, add_cfg, val_v27_q, overlay, fee=float(base_cfg["fee"]), slip=float(base_cfg["slip"]), cost_mult=3.0, decisions=val_dec)
            row = {"gate": asdict(gate), "overlay": asdict(overlay), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": v31_score(v1, v2, v3)}
            rows.append(row)
            if best is None or row["selection_score"] > best["selection_score"]:
                best = row
    assert best is not None
    selected_gate = UtilityGate(**best["gate"])
    selected_overlay = OverlayConfig(**best["overlay"])
    eval_dec = _override_decisions(parent, eval_df, eval_q_parent, selected_gate)
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
    model_path = args.out_dir / "v53_1_v27style_utility_parent.pt"
    torch.save({"model_id": MODEL_ID, "state_dict": model.state_dict(), "feature_cols": feature_cols, "norm": norm, "seq_len": SEQ_LEN, "horizons": HORIZONS, "selected_gate": asdict(selected_gate), "selected_overlay": asdict(selected_overlay), "parent_model": str(args.parent_model), "jackpot_model": str(args.jackpot_model), "v27_model": str(args.v27_model)}, model_path)
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
    if metrics["cost2"]["pnl"] <= 0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0:
        warnings.append("cost3_not_survived")
    verdict = "promote" if not blocking and metrics["cost1"]["pnl"] > V31_COST1 and metrics["cost2"]["pnl"] > 0 and metrics["cost3"]["pnl"] > 0 else "iterate"
    audit = {"status": "pass" if not blocking else "fail", "verdict": verdict, "blocking": blocking, "warnings": warnings, "selection_uses_2026": False, "selection_window": "2025-10-01..2025-12-31", "oos_window": "2026 fixed OOS only after selection", "policy": "v27style_two_utility_parent_replacement_for_v31_stack", "cash_is_threshold_not_class": True, "parent_bucket_heads_reused": True, "v21_2_preserved": True, "frozen_v27_residual_scout_preserved": True, "feature_audit": audit_base, "target_mean": np.mean(ds["target"], axis=0), "selected_gate": asdict(selected_gate), "selected_overlay": asdict(selected_overlay), "metrics": metrics, "baseline_v31": {"cost1": V31_COST1, "cost2": V31_COST2, "cost3": V31_COST3}}
    report = {"model_id": MODEL_ID, "design": "V27-style utility parent. It predicts q_long/q_short from 72-bar parent feature sequences; CASH is not a learned class and is derived only by edge/margin thresholds. Existing parent bucket heads provide notional/leverage/TP/SL/hold/cooldown; V21.2 add-on and frozen V27 residual scout are preserved.", "split_policy": "Train utility targets on 2025 Jan-Sep; select gate/overlay on 2025 Q4; evaluate fixed 2026 OOS after selection.", "feature_count": len(feature_cols), "training_rows": int(len(ds["target"])), "target_mean": np.mean(ds["target"], axis=0), "selected_gate": asdict(selected_gate), "selected_overlay": asdict(selected_overlay), "selection_result": best, "metrics": metrics, "audit": audit, "artifacts": {"model": str(model_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers}}
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "model": str(model_path), "selected_gate": asdict(selected_gate), "selected_overlay": asdict(selected_overlay), "metrics": metrics, "verdict": verdict}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
