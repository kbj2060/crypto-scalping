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

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, ACTION_SHORT, FullyLearnedGovernorConfig, build_training_set, predict_policy_frame, prepare_features  # noqa: E402
from scripts import eval_hf_v13_deep_tabular_parent_mdd_20260514 as base  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import eval_hf_v13_v31_frozen_parent_layer_ablation_v45 as v45  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "hf_v13_tft_oracle_bad_entry_guard_20260514"
OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_tft_oracle_bad_entry_guard_20260514"
REPORT_OUT = ROOT / "data/ensemble/reports/hf_v13_tft_oracle_bad_entry_guard_20260514_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/hf_v13_tft_oracle_bad_entry_guard_20260514_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/hf_v13_tft_oracle_bad_entry_guard_20260514_grid.csv"


@dataclass(frozen=True)
class Runtime:
    name: str
    keep_p: float
    weak_scale: float
    strong_scale: float
    hard: bool


class OracleDataset(Dataset):
    def __init__(self, x_tab: np.ndarray, x_seq: np.ndarray, keep: np.ndarray, weight: np.ndarray) -> None:
        self.x_tab = torch.as_tensor(x_tab, dtype=torch.float32)
        self.x_seq = torch.as_tensor(x_seq, dtype=torch.float32)
        self.keep = torch.as_tensor(keep, dtype=torch.float32)
        self.weight = torch.as_tensor(weight, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x_tab.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_tab[idx], self.x_seq[idx], self.keep[idx], self.weight[idx]


class OracleGuardTFT(nn.Module):
    def __init__(self, n_features: int, hidden: int = 80, n_layers: int = 1) -> None:
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
        self.keep_head = nn.Linear(hidden, 1)

    def forward(self, x_tab: torch.Tensor, x_seq: torch.Tensor) -> torch.Tensor:
        seq = x_seq if x_seq.ndim == 3 else x_tab[:, None, :]
        h = self.encoder(self.proj(seq * self.feature_gate(seq)))
        recency = torch.linspace(0.0, 0.35, h.shape[1], device=h.device).view(1, -1, 1)
        w = torch.softmax(self.attn(h) + recency, dim=1)
        z = self.norm(torch.sum(h * w, dim=1) + self.skip(x_tab))
        return self.keep_head(z).squeeze(-1)


def _make_labels(teacher: pd.DataFrame, gt: dict[str, np.ndarray], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = teacher.iloc[indices.astype(int)]
    teacher_action = d["action"].astype(int).to_numpy(dtype=np.int64)
    teacher_side = d["side"].astype(int).to_numpy(dtype=np.int64)
    active = (teacher_action != ACTION_CASH) & (teacher_side != 0)
    gt_action = np.asarray(gt["action"], dtype=np.int64)[: len(indices)]
    gt_quality = np.asarray(gt["quality"], dtype=np.float64)[: len(indices)]
    same = active & (gt_action == teacher_action)
    keep = (same & (gt_quality > 0.002)).astype(np.float32)
    # Emphasize active parent rows and high quality opportunities. CASH rows remain as low-weight negatives.
    weight = np.where(active, 1.0 + np.clip(np.abs(gt_quality) * 20.0, 0.0, 2.0), 0.12).astype(np.float32)
    return keep, weight, active


def _train(model: OracleGuardTFT, train_ds: OracleDataset, val_ds: OracleDataset, *, epochs: int, batch_size: int, device: torch.device) -> dict[str, Any]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    best = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, xs, y, w in train_loader:
            xb, xs, y, w = xb.to(device), xs.to(device), y.to(device), w.to(device)
            opt.zero_grad(set_to_none=True)
            loss = (F.binary_cross_entropy_with_logits(model(xb, xs), y, reduction="none") * w).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
        model.eval()
        vtotal = 0.0
        vcount = 0
        with torch.no_grad():
            for xb, xs, y, w in val_loader:
                xb, xs, y, w = xb.to(device), xs.to(device), y.to(device), w.to(device)
                loss = (F.binary_cross_entropy_with_logits(model(xb, xs), y, reduction="none") * w).mean()
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


def _predict(model: OracleGuardTFT, x_tab: np.ndarray, x_seq: np.ndarray, *, batch_size: int, device: torch.device) -> np.ndarray:
    model.to(device)
    model.eval()
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_tab), batch_size):
            end = min(start + batch_size, len(x_tab))
            xb = torch.as_tensor(x_tab[start:end], dtype=torch.float32, device=device)
            xs = torch.as_tensor(x_seq[start:end], dtype=torch.float32, device=device)
            outs.append(torch.sigmoid(model(xb, xs)).detach().cpu().numpy())
    model.to("cpu")
    return np.concatenate(outs, axis=0)


def _decisions(teacher: pd.DataFrame, keep_p: np.ndarray, rt: Runtime) -> pd.DataFrame:
    out = teacher.copy()
    active = (out["action"].astype(int).to_numpy() != ACTION_CASH) & (out["side"].astype(int).to_numpy() != 0)
    if rt.hard:
        scale = np.where(keep_p >= rt.keep_p, 1.0, 0.0)
    else:
        scale = np.where(keep_p >= rt.keep_p + 0.20, rt.strong_scale, np.where(keep_p >= rt.keep_p, 1.0, rt.weak_scale * np.clip(keep_p / max(rt.keep_p, 1e-6), 0.0, 1.0)))
    scale = np.clip(scale, 0.0, rt.strong_scale)
    out.loc[active, "notional_exposure"] = out.loc[active, "notional_exposure"].to_numpy(dtype=np.float64) * scale[active]
    out.loc[active, "position_fraction"] = out.loc[active, "position_fraction"].to_numpy(dtype=np.float64) * scale[active]
    zero = active & (scale <= 1e-8)
    out.loc[zero, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[zero, "leverage"] = 1.0
    out.loc[:, "oracle_keep_p"] = keep_p.astype(np.float64)
    out.loc[:, "oracle_scale"] = scale.astype(np.float64)
    return out


def _grid() -> list[Runtime]:
    rows: list[Runtime] = []
    for p in (0.30, 0.40, 0.50, 0.60, 0.70):
        rows.append(Runtime(f"oracle_hard_p{p:.2f}", p, 0.0, 1.0, True))
        for weak in (0.35, 0.50, 0.65):
            rows.append(Runtime(f"oracle_soft_p{p:.2f}_w{weak:.2f}", p, weak, 1.15, False))
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
    return float(c1["pnl"] + 0.35 * c2["pnl"] + 0.15 * c3["pnl"] - 4.5 * abs(c1["mdd"]))


def main() -> int:
    p = argparse.ArgumentParser(description="Oracle-style TFT bad-entry guard: predict whether HGB parent active entry should be kept.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=768)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(20260514)
    np.random.seed(20260514)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

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
    audit_base = _audit_contract(train_all, eval_df, feature_cols)
    print(f"[{MODEL_ID}] rows train={len(train_df)} val={len(val_df)} eval={len(eval_df)} features={len(feature_cols)}", flush=True)

    train_teacher = predict_policy_frame(parent, train_df, close=_close(train_df))
    val_teacher = predict_policy_frame(parent, val_df, close=_close(val_df))
    eval_teacher = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    idx_train = base._candidate_indices(len(train_df), cfg, int(args.stride))
    idx_val = base._candidate_indices(len(val_df), cfg, max(3, int(args.stride)))
    train_pre = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=feature_cols)
    train_x_labels = train_pre.iloc[idx_train].reset_index(drop=True)
    train_x_norm, norm = base._normalise_fit(train_x_labels)
    train_full = base._normalise_apply(train_pre, norm)
    x_train_seq = base._sequence_array(train_full, idx_train)
    _, y_gt_train, train_meta = build_training_set(train_df, cfg=cfg, stride_bars=int(args.stride), batch_size=512, feature_cols=feature_cols)
    keep_train, weight_train, active_train = _make_labels(train_teacher, y_gt_train, idx_train)

    val_pre = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    val_full = base._normalise_apply(val_pre, norm)
    x_val = val_full[idx_val]
    x_val_seq = base._sequence_array(val_full, idx_val)
    _, y_gt_val, val_meta = build_training_set(val_df, cfg=cfg, stride_bars=max(3, int(args.stride)), batch_size=512, feature_cols=feature_cols)
    keep_val, weight_val, active_val = _make_labels(val_teacher, y_gt_val, idx_val)
    keep = min(len(keep_val), len(x_val))
    x_val, x_val_seq = x_val[:keep], x_val_seq[:keep]
    keep_val, weight_val = keep_val[:keep], weight_val[:keep]

    train_ds = OracleDataset(train_x_norm.to_numpy(dtype=np.float32), x_train_seq, keep_train, weight_train)
    val_ds = OracleDataset(x_val, x_val_seq, keep_val, weight_val)
    print(f"[{MODEL_ID}] active_train={int(active_train.sum())} keep_rate_train={float(keep_train[active_train].mean()) if active_train.any() else 0.0:.4f} active_val={int(active_val[:keep].sum())} keep_rate_val={float(keep_val[active_val[:keep]].mean()) if active_val[:keep].any() else 0.0:.4f}", flush=True)
    model = OracleGuardTFT(len(feature_cols))
    training = _train(model, train_ds, val_ds, epochs=int(args.epochs), batch_size=int(args.batch_size), device=device)

    def pred_for(df: pd.DataFrame) -> np.ndarray:
        pre = prepare_features(df, side_hint=0, close=_close(df), feature_cols=feature_cols)
        x = base._normalise_apply(pre, norm)
        return _predict(model, x, base._sequence_array(x, np.arange(len(df), dtype=np.int64)), batch_size=int(args.batch_size), device=device)

    val_keep_p = pred_for(val_df)
    eval_keep_p = pred_for(eval_df)
    val_q = v31._predict_all(v27_model, val_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    variant = v45.LayerVariant("alpha1_oracle_bad_entry_guard", "oracle_bad_entry_guard", base._overlay_alpha1())
    base_cfg = dict(parent["config"])
    grid = _grid()
    if args.quick:
        grid = [r for r in grid if r.keep_p in (0.4, 0.5, 0.6) and (r.hard or r.weak_scale in (0.5,))]
    rows: list[dict[str, Any]] = []
    selected: Runtime | None = None
    best_score = -1e18
    for rt in grid:
        dec = _decisions(val_teacher, val_keep_p, rt)
        vm = _metrics(val_df, val_q, dec, parent, jackpot_model, add_cfg, variant, base_cfg)
        score = _score(vm)
        row = {**asdict(rt), "score": score, "val_pnl": vm["cost1"]["pnl"], "val_mdd": vm["cost1"]["mdd"], "val_trades": vm["cost1"]["trades"], "val_cost2_pnl": vm["cost2"]["pnl"], "val_cost3_pnl": vm["cost3"]["pnl"]}
        rows.append(row)
        if score > best_score:
            selected = rt
            best_score = score
            print(f"[{MODEL_ID}] new val best {rt.name} score={score:.2f} pnl={row['val_pnl']:.2f} mdd={row['val_mdd']:.2f}", flush=True)
    assert selected is not None

    experiments: list[dict[str, Any]] = []
    baseline_metrics = _metrics(eval_df, eval_q, eval_teacher, parent, jackpot_model, add_cfg, variant, base_cfg)
    experiments.append({"name": "alpha1_hgb_parent_baseline", "metrics": baseline_metrics, "score": _score(baseline_metrics)})
    dec = _decisions(eval_teacher, eval_keep_p, selected)
    best_metrics = _metrics(eval_df, eval_q, dec, parent, jackpot_model, add_cfg, variant, base_cfg)
    experiments.append({"name": selected.name, "config": asdict(selected), "metrics": best_metrics, "score": _score(best_metrics)})
    print(f"[{MODEL_ID}] OOS {selected.name} cost1={best_metrics['cost1']['pnl']:.2f} mdd={best_metrics['cost1']['mdd']:.2f} cost2={best_metrics['cost2']['pnl']:.2f} cost3={best_metrics['cost3']['pnl']:.2f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_id": MODEL_ID, "state_dict": model.state_dict(), "feature_cols": feature_cols, "normalizer": norm, "training": training, "selected": asdict(selected)}, OUT_DIR / "oracle_bad_entry_guard.pt")
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(GRID_OUT, index=False)
    blocking = list(audit_base.get("blocking", []))
    warnings = list(audit_base.get("warnings", []))
    if best_metrics["cost1"]["pnl"] < baseline_metrics["cost1"]["pnl"]:
        warnings.append("oracle_guard_cost1_below_alpha1")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and best_metrics["cost1"]["mdd"] > baseline_metrics["cost1"]["mdd"] and best_metrics["cost1"]["pnl"] > 0 else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS after selection",
        "design": "TFT oracle-style guard trained to predict whether HGB parent active entries match future optimal action and positive quality.",
        "train_meta": train_meta,
        "val_meta": val_meta,
        "base_audit": audit_base,
    }
    report = {
        "model_id": MODEL_ID,
        "selected": max(experiments, key=lambda e: e["score"]),
        "experiments": experiments,
        "selected_guard_from_validation": asdict(selected),
        "grid_path": str(GRID_OUT),
        "audit_path": str(AUDIT_OUT),
        "artifact_dir": str(OUT_DIR),
        "audit": audit,
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] selected={max(experiments, key=lambda e: e['score'])['name']} report={REPORT_OUT}", flush=True)
    print(f"[{MODEL_ID}] audit={AUDIT_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
