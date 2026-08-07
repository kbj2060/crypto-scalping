#!/usr/bin/env python3
"""Validation-only SOL architecture search for a causal entry encoder.

The experiment keeps the ETH 102-feature contract but replaces the shared
direction/quality/exit encoder with an entry-only model.  H48 quality is
represented by two side-conditional binary heads and the H48 return/MAE/MFE
targets are learned by side.  Regime probabilities mix small residual expert
adapters; inference never hard-routes to one expert.

This script is research-only.  It never changes the live model path.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707 as sol  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


LABEL_DIR = ROOT / "tmp/causal_regen_20260516/sol_zigzag_hysteresis_labels_20260719"
TB_DIR = ROOT / "tmp/causal_regen_20260516/sol_omega1_2_triple_barrier_labels_hysteresis_rebuild_20260719"
ETH_CONTRACT = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/true_3head_tabm_bundle.pt"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sol_architecture_v2_entry_20260719"
def _target_cols(horizon: int) -> list[str]:
    suffix = f"h{int(horizon)}_conservative"
    return [
        f"tb_long_ret_{suffix}",
        f"tb_short_ret_{suffix}",
        f"tb_long_mae_{suffix}",
        f"tb_short_mae_{suffix}",
        f"tb_long_mfe_{suffix}",
        f"tb_short_mfe_{suffix}",
        f"tb_long_reason_{suffix}",
        f"tb_short_reason_{suffix}",
    ]


@dataclass(frozen=True)
class Variant:
    name: str
    encoder: str
    seq_len: int
    hidden: int
    dropout: float


VARIANTS = (
    Variant("mlp_h96", "mlp", 1, 96, 0.08),
    Variant("mlp_h144", "mlp", 1, 144, 0.08),
    Variant("tcn_l24_h64", "tcn", 24, 64, 0.08),
    Variant("tcn_l48_h64", "tcn", 48, 64, 0.08),
    Variant("tcn_l24_h32", "tcn", 24, 32, 0.15),
)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_tb(horizon: int) -> pd.DataFrame:
    target_cols = _target_cols(horizon)
    parts = []
    for name in ("train", "validation", "oos"):
        path = TB_DIR / f"{name}_triple_barrier_labels.csv"
        usecols = ["timestamp", *target_cols]
        part = pd.read_csv(path, usecols=usecols, parse_dates=["timestamp"], low_memory=False)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return out.reset_index(drop=True)


def _attach_targets(frame: pd.DataFrame, tb: pd.DataFrame, *, horizon: int) -> tuple[pd.DataFrame, int]:
    target_cols = _target_cols(horizon)
    out = frame.merge(tb, on="timestamp", how="left", validate="one_to_one")
    missing_mask = out[target_cols].isna().any(axis=1)
    missing = int(missing_mask.sum())
    if missing:
        print(f"stage=exclude_missing_h48_targets rows={missing}", flush=True)
        out = out.loc[~missing_mask].reset_index(drop=True)
    out["q_long"] = (out[f"tb_long_reason_h{horizon}_conservative"] == "tp").astype(np.float32)
    out["q_short"] = (out[f"tb_short_reason_h{horizon}_conservative"] == "tp").astype(np.float32)
    return out, missing


class SequenceRows(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        gates: np.ndarray,
        y_dir: np.ndarray,
        y_quality: np.ndarray,
        y_outcome: np.ndarray,
        *,
        seq_len: int,
        start: int,
        end: int,
    ) -> None:
        self.x = x
        self.gates = gates
        self.y_dir = y_dir
        self.y_quality = y_quality
        self.y_outcome = y_outcome
        self.seq_len = int(seq_len)
        self.indices = np.arange(max(int(start), self.seq_len - 1), int(end), dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        idx = int(self.indices[item])
        left = idx - self.seq_len + 1
        return (
            torch.from_numpy(self.x[left : idx + 1]),
            torch.from_numpy(self.gates[idx]),
            torch.tensor(self.y_dir[idx], dtype=torch.long),
            torch.from_numpy(self.y_quality[idx]),
            torch.from_numpy(self.y_outcome[idx]),
            torch.tensor(idx, dtype=torch.long),
        )


class MLPEncoder(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x[:, -1])


class CausalTCNEncoder(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(n_features, hidden)
        self.blocks = nn.ModuleList()
        for dilation in (1, 2, 4, 8):
            conv = nn.Conv1d(hidden, hidden, kernel_size=3, dilation=dilation)
            self.blocks.append(nn.ModuleDict({"conv": conv, "norm": nn.LayerNorm(hidden), "drop": nn.Dropout(dropout)}))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x).transpose(1, 2)
        for idx, block in enumerate(self.blocks):
            dilation = 2**idx
            residual = h
            h = torch.nn.functional.pad(h, (2 * dilation, 0))
            h = block["conv"](h).transpose(1, 2)
            h = block["drop"](torch.nn.functional.silu(block["norm"](h))).transpose(1, 2)
            h = h + residual
        return h[:, :, -1]


class SoftResidualEntry(nn.Module):
    def __init__(self, n_features: int, variant: Variant) -> None:
        super().__init__()
        if variant.encoder == "mlp":
            self.encoder = MLPEncoder(n_features, variant.hidden, variant.dropout)
        elif variant.encoder == "tcn":
            self.encoder = CausalTCNEncoder(n_features, variant.hidden, variant.dropout)
        else:
            raise ValueError(variant.encoder)
        self.adapters = nn.ModuleList(
            nn.Sequential(nn.Linear(variant.hidden, variant.hidden), nn.SiLU(), nn.Dropout(variant.dropout))
            for _ in range(3)
        )
        self.direction = nn.Linear(variant.hidden, 3)
        self.quality = nn.Linear(variant.hidden, 2)
        self.outcome = nn.Linear(variant.hidden, 6)

    def forward(self, x: torch.Tensor, gates: torch.Tensor) -> dict[str, torch.Tensor]:
        base = self.encoder(x)
        adapted = torch.stack([base + adapter(base) for adapter in self.adapters], dim=1)
        mixed = (adapted * gates.unsqueeze(-1)).sum(dim=1)
        return {"direction": self.direction(mixed), "quality": self.quality(mixed), "outcome": self.outcome(mixed)}


def _metrics(y_dir: np.ndarray, y_quality: np.ndarray, pred: dict[str, np.ndarray]) -> dict[str, float]:
    direction = pred["direction"]
    quality = pred["quality"]
    result = {
        "direction_bacc": float(balanced_accuracy_score(y_dir, direction.argmax(axis=1))),
        "direction_logloss": float(log_loss(y_dir, direction, labels=[0, 1, 2])),
    }
    for idx, side in enumerate(("long", "short")):
        result[f"quality_{side}_auc"] = float(roc_auc_score(y_quality[:, idx], quality[:, idx]))
    return result


def _proxy_replay(
    pred: dict[str, np.ndarray],
    y_outcome_raw: np.ndarray,
    *,
    quality_threshold: float,
    direction_threshold: float,
    expected_return_threshold: float,
    cooldown_bars: int,
) -> dict[str, Any]:
    d = pred["direction"]
    q = pred["quality"]
    outcome = pred["outcome_raw"]
    cash = peak = 1.0
    mdd = 0.0
    rows: list[float] = []
    next_i = 0
    for i in range(len(d)):
        if i < next_i:
            continue
        action = int(np.argmax(d[i]))
        if action == 0 or float(d[i, action]) < float(direction_threshold):
            continue
        side_idx = action - 1
        if float(q[i, side_idx]) < float(quality_threshold):
            continue
        if float(outcome[i, side_idx]) < float(expected_return_threshold):
            continue
        realized = float(y_outcome_raw[i, side_idx]) - 0.0042
        trade_return = 0.45 * realized
        cash *= 1.0 + trade_return
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1.0e-12) - 1.0)
        rows.append(trade_return)
        next_i = i + int(cooldown_bars)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(len(rows)),
        "wr": float(np.mean(np.asarray(rows) > 0.0)) if rows else 0.0,
    }


def _predict(model: nn.Module, loader: DataLoader, device: torch.device, outcome_mean: np.ndarray, outcome_std: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    indices: list[np.ndarray] = []
    direction: list[np.ndarray] = []
    quality: list[np.ndarray] = []
    outcome: list[np.ndarray] = []
    with torch.no_grad():
        for xb, gb, _yd, _yq, _yo, idx in loader:
            out = model(xb.to(device), gb.to(device))
            indices.append(idx.numpy())
            direction.append(torch.softmax(out["direction"], dim=1).cpu().numpy())
            quality.append(torch.sigmoid(out["quality"]).cpu().numpy())
            outcome.append(out["outcome"].cpu().numpy())
    pred_outcome = np.concatenate(outcome)
    return np.concatenate(indices), {
        "direction": np.concatenate(direction),
        "quality": np.concatenate(quality),
        "outcome": pred_outcome,
        "outcome_raw": pred_outcome * outcome_std + outcome_mean,
    }


def _fit_variant(
    variant: Variant,
    x: np.ndarray,
    gates: np.ndarray,
    y_dir: np.ndarray,
    y_quality: np.ndarray,
    y_outcome: np.ndarray,
    *,
    train_end: int,
    device: torch.device,
    epochs: int,
    seed: int,
    out_dir: Path,
    lr: float,
    direction_loss_weight: float,
    quality_loss_weight: float,
    outcome_loss_weight: float,
) -> dict[str, Any]:
    _seed(seed)
    internal = int(train_end * 0.85)
    train_ds = SequenceRows(x, gates, y_dir, y_quality, y_outcome, seq_len=variant.seq_len, start=0, end=internal)
    early_ds = SequenceRows(x, gates, y_dir, y_quality, y_outcome, seq_len=variant.seq_len, start=internal, end=train_end)
    val_ds = SequenceRows(x, gates, y_dir, y_quality, y_outcome, seq_len=variant.seq_len, start=train_end, end=len(x))
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
    early_loader = DataLoader(early_ds, batch_size=1024, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=0)
    model = SoftResidualEntry(x.shape[1], variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1.0e-3)
    classes = np.array([0, 1, 2])
    dir_weights = torch.tensor(compute_class_weight("balanced", classes=classes, y=y_dir[:internal]), dtype=torch.float32, device=device)
    quality_pos = torch.tensor(
        [(y_quality[:internal, i] == 0).sum() / max((y_quality[:internal, i] == 1).sum(), 1) for i in range(2)],
        dtype=torch.float32,
        device=device,
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    epochs_ran = 0
    for epoch in range(int(epochs)):
        epochs_ran = epoch + 1
        model.train()
        for xb, gb, yd, yq, yo, _idx in train_loader:
            xb, gb, yd, yq, yo = xb.to(device), gb.to(device), yd.to(device), yq.to(device), yo.to(device)
            out = model(xb, gb)
            loss_dir = nn.functional.cross_entropy(out["direction"], yd, weight=dir_weights)
            loss_quality = nn.functional.binary_cross_entropy_with_logits(out["quality"], yq, pos_weight=quality_pos)
            loss_outcome = nn.functional.smooth_l1_loss(out["outcome"], yo)
            loss = direction_loss_weight * loss_dir + quality_loss_weight * loss_quality + outcome_loss_weight * loss_outcome
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, gb, yd, yq, yo, _idx in early_loader:
                xb, gb, yd, yq, yo = xb.to(device), gb.to(device), yd.to(device), yq.to(device), yo.to(device)
                out = model(xb, gb)
                loss = (
                    direction_loss_weight * nn.functional.cross_entropy(out["direction"], yd, weight=dir_weights)
                    + quality_loss_weight * nn.functional.binary_cross_entropy_with_logits(out["quality"], yq, pos_weight=quality_pos)
                    + outcome_loss_weight * nn.functional.smooth_l1_loss(out["outcome"], yo)
                )
                losses.append(float(loss.cpu()))
        current = float(np.mean(losses))
        print(f"variant={variant.name} epoch={epoch + 1} early_loss={current:.6f}", flush=True)
        if current + 1.0e-5 < best_loss:
            best_loss = current
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 6:
                break
    if best_state is None:
        raise RuntimeError(f"{variant.name}: no checkpoint")
    model.load_state_dict(best_state)
    idx, pred = _predict(model, val_loader, device, OUTCOME_MEAN, OUTCOME_STD)
    diagnostics = _metrics(y_dir[idx], y_quality[idx], pred)
    ranking = []
    for q in (0.45, 0.50, 0.55, 0.60, 0.65):
        for d in (0.36, 0.40, 0.44):
            for edge in (-0.001, 0.0, 0.001):
                metrics = _proxy_replay(
                    pred,
                    OUTCOME_RAW[idx],
                    quality_threshold=q,
                    direction_threshold=d,
                    expected_return_threshold=edge,
                    cooldown_bars=48,
                )
                ranking.append({"quality_threshold": q, "direction_threshold": d, "expected_return_threshold": edge, **metrics})
    ranking.sort(key=lambda row: (row["pnl"], row["mdd"]), reverse=True)
    payload = {
        "variant": asdict(variant),
        "state_dict": best_state,
        "feature_columns": BASE_COLS,
        "feature_mean": FEATURE_MEAN,
        "feature_std": FEATURE_STD,
        "outcome_mean": OUTCOME_MEAN,
        "outcome_std": OUTCOME_STD,
        "best_validation_loss": best_loss,
        "epochs_ran": epochs_ran,
        "diagnostics": diagnostics,
        "best_proxy": ranking[0],
    }
    torch.save(payload, out_dir / f"{variant.name}.pt")
    pd.DataFrame(ranking).to_csv(out_dir / f"{variant.name}_validation_proxy_ranking.csv", index=False)
    return {"variant": variant.name, "epochs_ran": epochs_ran, "best_validation_loss": best_loss, **diagnostics, **{f"proxy_{k}": v for k, v in ranking[0].items()}}


BASE_COLS: list[str] = []
FEATURE_MEAN = np.empty(0, dtype=np.float32)
FEATURE_STD = np.empty(0, dtype=np.float32)
OUTCOME_MEAN = np.empty(0, dtype=np.float32)
OUTCOME_STD = np.empty(0, dtype=np.float32)
OUTCOME_RAW = np.empty((0, 6), dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=260719)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--variants", default=",".join(v.name for v in VARIANTS))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--horizon", type=int, choices=[24, 48, 96], default=48)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--direction-loss-weight", type=float, default=1.0)
    ap.add_argument("--quality-loss-weight", type=float, default=0.75)
    ap.add_argument("--outcome-loss-weight", type=float, default=0.25)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    contract = torch.load(ETH_CONTRACT, map_location="cpu", weights_only=False)
    global BASE_COLS, FEATURE_MEAN, FEATURE_STD, OUTCOME_MEAN, OUTCOME_STD, OUTCOME_RAW
    BASE_COLS = list(contract["base_cols"])
    frames = sol._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=LABEL_DIR,
        quality_mode="same_as_direction",
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    train = frames["train_raw"].copy()
    validation = frames["val_raw"].copy()
    train["_split"] = "train"
    validation["_split"] = "validation"
    all_frame = pd.concat([train, validation], ignore_index=True)
    all_frame, missing_target_rows = _attach_targets(all_frame, _load_tb(int(args.horizon)), horizon=int(args.horizon))
    x_raw = all_frame[BASE_COLS].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    train_end = int((all_frame["_split"] == "train").sum())
    if not bool((all_frame["_split"].iloc[:train_end] == "train").all()):
        raise RuntimeError("train/validation ordering changed after target alignment")
    FEATURE_MEAN = x_raw[:train_end].mean(axis=0).astype(np.float32)
    FEATURE_STD = x_raw[:train_end].std(axis=0).astype(np.float32)
    FEATURE_STD[FEATURE_STD < 1.0e-6] = 1.0
    x = ((x_raw - FEATURE_MEAN) / FEATURE_STD).astype(np.float32)
    gates = all_frame[hard.ROUTE_COLS].to_numpy(dtype=np.float32)
    gates /= np.clip(gates.sum(axis=1, keepdims=True), 1.0e-8, None)
    y_dir = all_frame["zigzag_action"].to_numpy(dtype=np.int64)
    y_quality = all_frame[["q_long", "q_short"]].to_numpy(dtype=np.float32)
    target_cols = _target_cols(int(args.horizon))
    OUTCOME_RAW = all_frame[target_cols[:6]].to_numpy(dtype=np.float32)
    OUTCOME_MEAN = OUTCOME_RAW[:train_end].mean(axis=0).astype(np.float32)
    OUTCOME_STD = OUTCOME_RAW[:train_end].std(axis=0).astype(np.float32)
    OUTCOME_STD[OUTCOME_STD < 1.0e-6] = 1.0
    y_outcome = ((OUTCOME_RAW - OUTCOME_MEAN) / OUTCOME_STD).astype(np.float32)
    selected = {name.strip() for name in str(args.variants).split(",") if name.strip()}
    unknown = selected - {variant.name for variant in VARIANTS}
    if unknown:
        raise RuntimeError(f"unknown variants: {sorted(unknown)}")
    rows = []
    for variant in VARIANTS:
        if variant.name not in selected:
            continue
        rows.append(
            _fit_variant(
                variant,
                x,
                gates,
                y_dir,
                y_quality,
                y_outcome,
                train_end=train_end,
                device=device,
                epochs=int(args.epochs),
                seed=int(args.seed),
                out_dir=args.out_dir,
                lr=float(args.lr),
                direction_loss_weight=float(args.direction_loss_weight),
                quality_loss_weight=float(args.quality_loss_weight),
                outcome_loss_weight=float(args.outcome_loss_weight),
            )
        )
    rows.sort(key=lambda row: (row["proxy_pnl"], row["proxy_mdd"]), reverse=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "validation_architecture_ranking.csv", index=False)
    report = {
        "selection_scope": "validation_only",
        "train_range": [str(train.timestamp.iloc[0]), str(train.timestamp.iloc[-1])],
        "validation_range": [str(validation.timestamp.iloc[0]), str(validation.timestamp.iloc[-1])],
        "oos_used": False,
        "feature_contract": str(ETH_CONTRACT),
        "base_feature_count": len(BASE_COLS),
        "target_contract": f"side-conditional H{int(args.horizon)} conservative TP success plus return/MAE/MFE",
        "loss_weights": {
            "direction": float(args.direction_loss_weight),
            "quality": float(args.quality_loss_weight),
            "outcome": float(args.outcome_loss_weight),
        },
        "excluded_missing_h48_target_rows": missing_target_rows,
        "rows": rows,
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
