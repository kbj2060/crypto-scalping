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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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
from scripts import eval_hf_v13_deep_tabular_parent_mdd_20260514 as base  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import eval_hf_v13_v31_frozen_parent_layer_ablation_v45 as v45  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "hf_v13_tft_entry_parent_lite_hgb_tactician_20260514"
OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_tft_entry_parent_lite_hgb_tactician_20260514"
REPORT_OUT = ROOT / "data/ensemble/reports/hf_v13_tft_entry_parent_lite_hgb_tactician_20260514_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/hf_v13_tft_entry_parent_lite_hgb_tactician_20260514_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/hf_v13_tft_entry_parent_lite_hgb_tactician_20260514_grid.csv"
SEQ_LEN = 72


@dataclass(frozen=True)
class Runtime:
    name: str
    confidence: float
    quality_floor: float
    uncertainty_max: float


class EntryDataset(Dataset):
    def __init__(self, x_tab: np.ndarray, x_seq: np.ndarray, action: np.ndarray, quality: np.ndarray) -> None:
        self.x_tab = torch.as_tensor(x_tab, dtype=torch.float32)
        self.x_seq = torch.as_tensor(x_seq, dtype=torch.float32)
        self.action = torch.as_tensor(action, dtype=torch.long)
        self.quality = torch.as_tensor(quality, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x_tab.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_tab[idx], self.x_seq[idx], self.action[idx], self.quality[idx]


class TFTEntryParentLite(nn.Module):
    def __init__(self, n_features: int, hidden: int = 96, n_layers: int = 2) -> None:
        super().__init__()
        self.feature_gate = nn.Sequential(nn.LayerNorm(n_features), nn.Linear(n_features, n_features), nn.Sigmoid())
        self.proj = nn.Linear(n_features, hidden)
        enc = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 4,
            dropout=0.12,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.attn = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.skip = nn.Linear(n_features, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.action_head = nn.Linear(hidden, 3)
        self.quality_head = nn.Linear(hidden, 1)
        self.loss_log_vars = nn.ParameterDict(
            {
                "action": nn.Parameter(torch.zeros(())),
                "quality": nn.Parameter(torch.zeros(())),
            }
        )

    def forward(self, x_tab: torch.Tensor, x_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        seq = x_seq if x_seq.ndim == 3 else x_tab[:, None, :]
        gate = self.feature_gate(seq)
        h = self.encoder(self.proj(seq * gate))
        recency = torch.linspace(0.0, 0.35, h.shape[1], device=h.device).view(1, -1, 1)
        w = torch.softmax(self.attn(h) + recency, dim=1)
        z = torch.sum(h * w, dim=1) + self.skip(x_tab)
        z = self.norm(z)
        return {"action": self.action_head(z), "quality": self.quality_head(z).squeeze(-1)}


def _balanced(model: nn.Module, name: str, term: torch.Tensor) -> torch.Tensor:
    s = model.loss_log_vars[name].clamp(-3.0, 3.0)
    return torch.exp(-s) * term + 0.5 * s


def _loss(model: nn.Module, out: dict[str, torch.Tensor], action: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    active = action != ACTION_CASH
    weights = torch.ones(3, device=action.device)
    weights[ACTION_CASH] = 0.45
    action_loss = F.cross_entropy(out["action"], action, weight=weights)
    q_weight = torch.where(active, torch.tensor(1.0, device=action.device), torch.tensor(0.35, device=action.device))
    q_loss = (F.smooth_l1_loss(out["quality"], quality, reduction="none") * q_weight).mean()
    return _balanced(model, "action", action_loss) + _balanced(model, "quality", q_loss)


def _train(
    model: TFTEntryParentLite,
    train_ds: EntryDataset,
    val_ds: EntryDataset,
    *,
    epochs: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1.2e-4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    best = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, xs, ya, yq in train_loader:
            xb, xs, ya, yq = xb.to(device), xs.to(device), ya.to(device), yq.to(device)
            opt.zero_grad(set_to_none=True)
            loss = _loss(model, model(xb, xs), ya, yq)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
        model.eval()
        vtotal = 0.0
        vcount = 0
        with torch.no_grad():
            for xb, xs, ya, yq in val_loader:
                xb, xs, ya, yq = xb.to(device), xs.to(device), ya.to(device), yq.to(device)
                loss = _loss(model, model(xb, xs), ya, yq)
                vtotal += float(loss.item()) * len(xb)
                vcount += len(xb)
        tr = total / max(count, 1)
        va = vtotal / max(vcount, 1)
        history.append({"epoch": float(epoch), "train_loss": tr, "val_loss": va})
        print(f"[{MODEL_ID}] epoch={epoch:02d} train_loss={tr:.5f} val_loss={va:.5f}", flush=True)
        if va < best:
            best = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    return {"best_val_loss": float(best), "history": history}


def _predict_entry(
    model: TFTEntryParentLite,
    x_tab: np.ndarray,
    x_seq: np.ndarray,
    device: torch.device,
    batch_size: int,
    mc_passes: int = 8,
) -> dict[str, np.ndarray]:
    model.to(device)
    outs: dict[str, list[np.ndarray]] = {"action": [], "quality": [], "uncertainty": []}
    with torch.no_grad():
        for start in range(0, len(x_tab), batch_size):
            end = min(start + batch_size, len(x_tab))
            xb = torch.as_tensor(x_tab[start:end], dtype=torch.float32, device=device)
            xs = torch.as_tensor(x_seq[start:end], dtype=torch.float32, device=device)
            probs: list[torch.Tensor] = []
            qualities: list[torch.Tensor] = []
            for _ in range(max(1, int(mc_passes))):
                model.train(mc_passes > 1)
                pred = model(xb, xs)
                probs.append(torch.softmax(pred["action"].clamp(-7.0, 7.0) / 1.35, dim=-1))
                qualities.append(pred["quality"])
            stack = torch.stack(probs, dim=0)
            outs["action"].append(stack.mean(dim=0).detach().cpu().numpy())
            outs["uncertainty"].append(stack.std(dim=0).mean(dim=1).detach().cpu().numpy())
            outs["quality"].append(torch.stack(qualities, dim=0).mean(dim=0).detach().cpu().numpy())
    model.to("cpu")
    return {k: np.concatenate(v, axis=0) for k, v in outs.items()}


def _hgb_tactician_frame(
    parent: dict[str, Any],
    frame: pd.DataFrame,
    close: np.ndarray,
    side: np.ndarray,
    action: np.ndarray,
    quality: np.ndarray,
    confidence: np.ndarray,
) -> pd.DataFrame:
    cfg = FullyLearnedGovernorConfig(**dict(parent["config"]))
    feature_cols = list(parent.get("feature_cols") or [])
    if set(feature_cols).issubset(frame.columns):
        x = frame.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan).copy()
    else:
        x = prepare_features(frame, side_hint=0, close=close, feature_cols=feature_cols)
    x_side = x.copy()
    if "side_hint" in x_side.columns:
        x_side["side_hint"] = side.astype(np.float64)
    notional, c1 = _bucket_or_default_batch(parent, "notional", x_side, cfg.notional_buckets)
    leverage, c2 = _bucket_or_default_batch(parent, "leverage", x_side, cfg.leverage_buckets)
    tp, c3 = _bucket_or_default_batch(parent, "take_profit", x_side, cfg.take_profit_buckets)
    sl, c4 = _bucket_or_default_batch(parent, "stop_loss", x_side, cfg.stop_loss_buckets)
    mh, c5 = _bucket_or_default_batch(parent, "max_hold", x_side, tuple(float(v) for v in cfg.max_hold_buckets))
    cd, c6 = _bucket_or_default_batch(parent, "cooldown", x_side, tuple(float(v) for v in cfg.cooldown_buckets))
    leverage = np.clip(leverage, min(cfg.leverage_buckets), max(cfg.leverage_buckets))
    notional = np.clip(notional, min(cfg.notional_buckets), max(cfg.notional_buckets))
    fraction = np.clip(notional / np.maximum(leverage, 1e-8), 0.0, cfg.max_margin_fraction)
    notional = fraction * leverage
    active = (action != ACTION_CASH) & (side != 0)
    out = pd.DataFrame(
        {
            "action": action.astype(np.int64),
            "side": side.astype(np.int64),
            "notional_exposure": notional.astype(np.float64),
            "leverage": leverage.astype(np.float64),
            "position_fraction": fraction.astype(np.float64),
            "take_profit": tp.astype(np.float64),
            "stop_loss": sl.astype(np.float64),
            "max_hold_bars": np.rint(mh).astype(np.int64),
            "cooldown_bars": np.rint(cd).astype(np.int64),
            "quality_score": quality.astype(np.float64),
            "confidence": np.mean(np.vstack([confidence, c1, c2, c3, c4, c5, c6]), axis=0).astype(np.float64),
        },
        index=frame.index,
    )
    out.loc[~active, ["side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[~active, "leverage"] = 1.0
    return out


def _decisions_from_entry(parent: dict[str, Any], frame: pd.DataFrame, outputs: dict[str, np.ndarray], rt: Runtime) -> pd.DataFrame:
    proba = outputs["action"]
    pred_action = np.argmax(proba, axis=1).astype(np.int64)
    pred_conf = np.max(proba, axis=1)
    pred_side = np.where(pred_action == ACTION_LONG, 1, np.where(pred_action == ACTION_SHORT, -1, 0)).astype(np.int64)
    quality = np.asarray(outputs["quality"], dtype=np.float64)
    uncertainty = np.asarray(outputs["uncertainty"], dtype=np.float64)
    active = (pred_action != ACTION_CASH) & (pred_side != 0) & (pred_conf >= rt.confidence) & (quality >= rt.quality_floor) & (uncertainty <= rt.uncertainty_max)
    action = np.where(active, pred_action, ACTION_CASH).astype(np.int64)
    side = np.where(active, pred_side, 0).astype(np.int64)
    dec = _hgb_tactician_frame(parent, frame, _close(frame), side, action, quality, pred_conf)
    dec.loc[:, "tft_entry_confidence"] = pred_conf.astype(np.float64)
    dec.loc[:, "tft_entry_uncertainty"] = uncertainty.astype(np.float64)
    return dec


def _runtime_grid() -> list[Runtime]:
    rows: list[Runtime] = []
    for conf in (0.30, 0.38, 0.46, 0.54, 0.62):
        for q in (-0.020, -0.010, 0.000, 0.010):
            for unc in (0.050, 0.070, 0.095):
                rows.append(Runtime(f"tft_entry_hgb_tactician_c{conf:.2f}_q{q:.3f}_u{unc:.3f}", conf, q, unc))
    return rows


def _metrics(df: pd.DataFrame, q: np.ndarray, decisions: pd.DataFrame, parent: dict[str, Any], jackpot_model: dict[str, Any], add_cfg: CostRunnerConfig, variant: v45.LayerVariant, base_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        f"cost{m}": v45.backtest_variant(df, parent, jackpot_model, add_cfg, q, variant, fee=float(base_cfg["fee"]), slip=float(base_cfg["slip"]), cost_mult=float(m), decisions=decisions)
        for m in (1, 2, 3)
    }


def _score(metrics: dict[str, Any]) -> float:
    c1, c2, c3 = metrics["cost1"], metrics["cost2"], metrics["cost3"]
    if int(c1["trades"]) < 20:
        return -1e9 + float(c1["pnl"])
    return float(c1["pnl"] + 0.35 * c2["pnl"] + 0.12 * c3["pnl"] - 4.5 * abs(c1["mdd"]))


def main() -> int:
    p = argparse.ArgumentParser(description="TFT Entry Parent Lite + frozen HGB tactician heads in Alpha1/V31 stack.")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=768)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(20260514)
    np.random.seed(20260514)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"[{MODEL_ID}] device={device} epochs={args.epochs} stride={args.stride}", flush=True)

    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    cfg = FullyLearnedGovernorConfig(**dict(parent["config"]))
    feature_cols = list(parent.get("feature_cols") or [])
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    print(f"[{MODEL_ID}] rows train={len(train_df)} val={len(val_df)} eval={len(eval_df)} features={len(feature_cols)}", flush=True)

    audit_base = _audit_contract(train_all, eval_df, feature_cols)
    train_teacher = predict_policy_frame(parent, train_df, close=_close(train_df))
    val_teacher = predict_policy_frame(parent, val_df, close=_close(val_df))
    eval_teacher = predict_policy_frame(parent, eval_df, close=_close(eval_df))

    idx_train = base._candidate_indices(len(train_df), cfg, int(args.stride))
    train_pre = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=feature_cols)
    x_train_labels = train_pre.iloc[idx_train].reset_index(drop=True)
    x_train_norm, norm = base._normalise_fit(x_train_labels)
    train_full = base._normalise_apply(train_pre, norm)
    x_train_seq = base._sequence_array(train_full, idx_train)
    y_action = train_teacher.iloc[idx_train]["action"].astype(int).to_numpy(dtype=np.int64)
    y_quality = pd.to_numeric(train_teacher.iloc[idx_train]["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    idx_val = base._candidate_indices(len(val_df), cfg, max(3, int(args.stride)))
    val_pre = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    val_full = base._normalise_apply(val_pre, norm)
    x_val_tab = val_full[idx_val]
    x_val_seq = base._sequence_array(val_full, idx_val)
    y_val_action = val_teacher.iloc[idx_val]["action"].astype(int).to_numpy(dtype=np.int64)
    y_val_quality = pd.to_numeric(val_teacher.iloc[idx_val]["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    train_ds = EntryDataset(x_train_norm.to_numpy(dtype=np.float32), x_train_seq, y_action, y_quality)
    val_ds = EntryDataset(x_val_tab, x_val_seq, y_val_action, y_val_quality)
    model = TFTEntryParentLite(len(feature_cols))
    training = _train(model, train_ds, val_ds, epochs=int(args.epochs), device=device, batch_size=int(args.batch_size))

    val_q = v31._predict_all(v27_model, val_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_pre = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    eval_full = base._normalise_apply(eval_pre, norm)
    val_outputs = _predict_entry(model, val_full, base._sequence_array(val_full, np.arange(len(val_df), dtype=np.int64)), device, int(args.batch_size), mc_passes=8)
    eval_outputs = _predict_entry(model, eval_full, base._sequence_array(eval_full, np.arange(len(eval_df), dtype=np.int64)), device, int(args.batch_size), mc_passes=8)

    variant = v45.LayerVariant("alpha1_tft_entry_lite_hgb_tactician", "parent_tft_entry_hgb_tactician", base._overlay_alpha1())
    base_cfg = dict(parent["config"])
    rows: list[dict[str, Any]] = []
    selected: Runtime | None = None
    best_score = -1e18
    grid = _runtime_grid()
    if args.quick:
        grid = [r for r in grid if r.confidence in (0.30, 0.46, 0.62) and r.quality_floor in (-0.01, 0.0) and r.uncertainty_max in (0.070, 0.095)]
    for rt in grid:
        dec = _decisions_from_entry(parent, val_df, val_outputs, rt)
        vm = _metrics(val_df, val_q, dec, parent, jackpot_model, add_cfg, variant, base_cfg)
        score = _score(vm)
        row = {**asdict(rt), "score": score, "val_pnl": vm["cost1"]["pnl"], "val_mdd": vm["cost1"]["mdd"], "val_trades": vm["cost1"]["trades"], "val_cost2_pnl": vm["cost2"]["pnl"], "val_cost3_pnl": vm["cost3"]["pnl"]}
        rows.append(row)
        if score > best_score:
            best_score = score
            selected = rt
            print(f"[{MODEL_ID}] new val best {rt.name} score={score:.2f} pnl={row['val_pnl']:.2f} mdd={row['val_mdd']:.2f}", flush=True)
    assert selected is not None

    baseline_metrics = _metrics(eval_df, eval_q, eval_teacher, parent, jackpot_model, add_cfg, variant, base_cfg)
    best_dec = _decisions_from_entry(parent, eval_df, eval_outputs, selected)
    best_metrics = _metrics(eval_df, eval_q, best_dec, parent, jackpot_model, add_cfg, variant, base_cfg)
    experiments = [
        {"name": "alpha1_hgb_parent_baseline", "metrics": baseline_metrics, "score": _score(baseline_metrics)},
        {"name": f"tft_entry_lite_hgb_tactician::{selected.name}", "selected_runtime": asdict(selected), "metrics": best_metrics, "score": _score(best_metrics)},
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_id": MODEL_ID, "state_dict": model.state_dict(), "feature_cols": feature_cols, "normalizer": norm, "training": training, "selected_runtime": asdict(selected), "config": base_cfg}, OUT_DIR / "tft_entry_parent_lite.pt")
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(GRID_OUT, index=False)

    blocking = list(audit_base.get("blocking", []))
    warnings = list(audit_base.get("warnings", []))
    if best_metrics["cost1"]["pnl"] < baseline_metrics["cost1"]["pnl"]:
        warnings.append("tft_entry_lite_hgb_tactician_cost1_below_alpha1")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and best_metrics["cost1"]["pnl"] > baseline_metrics["cost1"]["pnl"] and best_metrics["cost1"]["mdd"] >= baseline_metrics["cost1"]["mdd"] else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS after selection",
        "design": "TFT Entry Parent Lite predicts only action/quality. Frozen HGB tactician heads produce notional, leverage, TP, SL, max_hold, and cooldown conditioned on TFT side.",
        "base_audit": audit_base,
    }
    report = {
        "model_id": MODEL_ID,
        "selected": max(experiments, key=lambda e: e["score"]),
        "experiments": experiments,
        "artifact_dir": str(OUT_DIR),
        "grid_path": str(GRID_OUT),
        "audit_path": str(AUDIT_OUT),
        "audit": audit,
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] selected={selected.name} cost1={best_metrics['cost1']['pnl']:.2f} mdd={best_metrics['cost1']['mdd']:.2f} cost2={best_metrics['cost2']['pnl']:.2f} cost3={best_metrics['cost3']['pnl']:.2f}", flush=True)
    print(f"[{MODEL_ID}] report={REPORT_OUT}", flush=True)
    print(f"[{MODEL_ID}] audit={AUDIT_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
