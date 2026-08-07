#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    _combine_primary_fallback,
    _predict_scaled,
    _read,
)
from scripts.rebuild_alpha7_v2_only_high_turnover_20260526 import _rename_clean4_v2  # noqa: E402
from scripts.sweep_alpha8_origin_scaled_combo_20260529 import OfficialCost3  # noqa: E402
from scripts.train_eval_alpha8_dsac_iqn_risk_selector_20260529 import (  # noqa: E402
    _active,
    _assert_clean,
    _load_scale_runtime_any,
    _state_frame,
    _zero_row,
    _simulate_template,
)
from scripts.train_eval_alpha7_directional_dsac_router_20260529 import _apply_norm, _fit_norm  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha8_bounded_regression_risk_20260529"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

DEFAULT_TRAIN_CSV = (
    ROOT
    / "tmp/causal_regen_20260516/alpha8_clean_hybrid_v1_20260529_inputs"
    / "trade_candidates_2025_alpha8_cleanfunding_m7_regimepred.csv"
)
DEFAULT_EVAL_CSV = (
    ROOT
    / "tmp/causal_regen_20260516/alpha8_clean_hybrid_v1_20260529_inputs"
    / "trade_candidates_2026_alpha8_cleanfunding_m7_regimepred.csv"
)
DEFAULT_PARENT_DIR = ROOT / "tmp/causal_regen_20260516/alpha8_clean_parent_fallback_retrain_20260529"


@dataclass(frozen=True)
class RegressionBounds:
    name: str
    mult_min: float
    mult_max: float
    cap: float
    lev_min: float
    lev_max: float
    tp_min: float
    tp_max: float
    sl_min: float
    sl_max: float
    hold_min: float
    hold_max: float


@dataclass(frozen=True)
class CandidateRisk:
    name: str
    mult: float
    cap: float
    leverage: float
    tp_mult: float
    sl_mult: float
    hold_mult: float
    veto: bool = False


class BoundedRiskRegressor(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


BOUNDS: tuple[RegressionBounds, ...] = (
    RegressionBounds("reg_conservative", 0.55, 1.45, 5.0, 2.0, 5.0, 0.10, 0.26, 3.0, 6.0, 0.50, 0.95),
    RegressionBounds("reg_balanced", 0.55, 2.10, 7.5, 2.0, 5.0, 0.08, 0.35, 2.5, 7.0, 0.40, 1.15),
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scale01(value: float, lo: float, hi: float) -> float:
    return float(np.clip((float(value) - float(lo)) / max(float(hi) - float(lo), 1e-12), 0.0, 1.0))


def _unscale01(value: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return float(lo) + np.clip(value, 0.0, 1.0) * (float(hi) - float(lo))


def _candidate_grid(bounds: RegressionBounds, *, candidate_count: int) -> list[CandidateRisk]:
    rows: list[CandidateRisk] = [CandidateRisk("veto", 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, True)]
    center = CandidateRisk(
        name="center",
        mult=float((bounds.mult_min + bounds.mult_max) * 0.5),
        cap=float(bounds.cap),
        leverage=float((bounds.lev_min + bounds.lev_max) * 0.5),
        tp_mult=float((bounds.tp_min + bounds.tp_max) * 0.5),
        sl_mult=float((bounds.sl_min + bounds.sl_max) * 0.5),
        hold_mult=float((bounds.hold_min + bounds.hold_max) * 0.5),
    )
    rows.append(center)
    seed = 8052901 if bounds.name == "reg_conservative" else 8052902
    rng = np.random.default_rng(seed)
    for i in range(max(0, int(candidate_count) - len(rows))):
        rows.append(
            CandidateRisk(
                name=f"sample_{i:02d}",
                mult=float(rng.uniform(bounds.mult_min, bounds.mult_max)),
                cap=float(bounds.cap),
                leverage=float(rng.uniform(bounds.lev_min, bounds.lev_max)),
                tp_mult=float(rng.uniform(bounds.tp_min, bounds.tp_max)),
                sl_mult=float(rng.uniform(bounds.sl_min, bounds.sl_max)),
                hold_mult=float(rng.uniform(bounds.hold_min, bounds.hold_max)),
            )
        )
    return rows


def _build_targets(
    frame: pd.DataFrame,
    combo: pd.DataFrame,
    states: np.ndarray,
    *,
    bounds: RegressionBounds,
    fee: float,
    slip: float,
    cost_mult: float,
    candidate_count: int,
    max_target_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    active = _active(combo)
    idxs = np.flatnonzero(active & (np.arange(len(frame)) < len(frame) - 3))
    if int(max_target_rows) > 0 and len(idxs) > int(max_target_rows):
        pick = np.linspace(0, len(idxs) - 1, int(max_target_rows)).round().astype(int)
        idxs = idxs[pick]
    candidates = _candidate_grid(bounds, candidate_count=int(candidate_count))
    x_rows: list[np.ndarray] = []
    y_rows: list[list[float]] = []
    weights: list[float] = []
    best_counts = {c.name: 0 for c in candidates}
    best_rewards: list[float] = []

    for i in idxs:
        rewards: list[float] = []
        for candidate in candidates:
            reward, _ = _simulate_template(
                frame,
                int(i),
                combo.iloc[int(i)],
                candidate,  # type: ignore[arg-type]
                fee=fee,
                slip=slip,
                cost_mult=cost_mult,
            )
            rewards.append(float(reward))
        best_id = int(np.argmax(rewards))
        best = candidates[best_id]
        best_counts[best.name] += 1
        best_reward = float(rewards[best_id])
        best_rewards.append(best_reward)
        trade = 0.0 if best.veto else 1.0
        x_rows.append(states[int(i)])
        y_rows.append(
            [
                trade,
                0.0 if best.veto else _scale01(best.mult, bounds.mult_min, bounds.mult_max),
                0.0 if best.veto else _scale01(best.leverage, bounds.lev_min, bounds.lev_max),
                0.0 if best.veto else _scale01(best.tp_mult, bounds.tp_min, bounds.tp_max),
                0.0 if best.veto else _scale01(best.sl_mult, bounds.sl_min, bounds.sl_max),
                0.0 if best.veto else _scale01(best.hold_mult, bounds.hold_min, bounds.hold_max),
            ]
        )
        weights.append(2.0 if trade > 0.5 else 1.0)

    diag = {
        "active_rows": int(len(idxs)),
        "max_target_rows": int(max_target_rows),
        "candidate_count": int(len(candidates)),
        "best_counts": best_counts,
        "best_reward_mean": float(np.mean(best_rewards)) if best_rewards else 0.0,
        "best_reward_median": float(np.median(best_rewards)) if best_rewards else 0.0,
    }
    return (
        np.asarray(x_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
        diag,
    )


def _train_regressor(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[BoundedRiskRegressor, dict[str, Any]]:
    _seed_everything(seed)
    model = BoundedRiskRegressor(state_dim=x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=2e-4)
    sampler = WeightedRandomSampler(torch.from_numpy(sample_weight.astype(np.float64)), len(sample_weight), replacement=True)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(sample_weight))
    dl = DataLoader(ds, batch_size=int(batch_size), sampler=sampler, drop_last=False)
    losses: list[float] = []
    for epoch in range(1, int(epochs) + 1):
        total = 0.0
        n = 0
        model.train()
        for xb, yb, sw in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            sw = sw.to(device)
            pred = model(xb)
            loss_by_head = F.smooth_l1_loss(pred, yb, reduction="none")
            head_weight = torch.tensor([2.5, 1.0, 0.7, 1.0, 1.0, 0.7], device=device, dtype=pred.dtype)
            loss = ((loss_by_head * head_weight).mean(dim=1) * sw).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            n += len(xb)
        losses.append(total / max(n, 1))
        if epoch % 5 == 0 or epoch == int(epochs):
            print(json.dumps({"stage": "regression_progress", "epoch": epoch, "loss": losses[-1]}, ensure_ascii=False), flush=True)
    return model.cpu(), {"epochs": int(epochs), "losses": losses}


def _predict(model: BoundedRiskRegressor, states: np.ndarray, *, device: torch.device) -> np.ndarray:
    model = model.to(device)
    model.eval()
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states), 8192):
            xb = torch.from_numpy(states[start : start + 8192]).to(device)
            outs.append(model(xb).cpu().numpy().astype(np.float32))
    return np.concatenate(outs) if outs else np.zeros((0, 6), dtype=np.float32)


def _compose_regression_decisions(
    combo: pd.DataFrame,
    pred: np.ndarray,
    *,
    bounds: RegressionBounds,
    trade_threshold: float,
) -> pd.DataFrame:
    out = combo.copy().reset_index(drop=True)
    active = _active(out)
    for i in np.flatnonzero(active):
        row = out.iloc[int(i)].copy()
        p = pred[int(i)]
        if float(p[0]) < float(trade_threshold):
            out.iloc[int(i)] = _zero_row(row)
            continue
        mult = float(_unscale01(p[1], bounds.mult_min, bounds.mult_max))
        lev = float(_unscale01(p[2], bounds.lev_min, bounds.lev_max))
        tp_mult = float(_unscale01(p[3], bounds.tp_min, bounds.tp_max))
        sl_mult = float(_unscale01(p[4], bounds.sl_min, bounds.sl_max))
        hold_mult = float(_unscale01(p[5], bounds.hold_min, bounds.hold_max))
        base_notional = float(row.get("notional_exposure", 0.0) or 0.0)
        base_tp = float(row.get("take_profit", 0.0) or 0.0)
        base_sl = abs(float(row.get("stop_loss", 0.0) or 0.0))
        base_hold = max(int(row.get("max_hold_bars", 0) or 0), 1)
        notional = min(max(base_notional * mult, 0.0), float(bounds.cap))
        row.loc["notional_exposure"] = notional
        row.loc["leverage"] = max(lev, 1e-8)
        row.loc["position_fraction"] = float(notional / max(lev, 1e-8))
        row.loc["take_profit"] = float(max(base_tp, 1e-8) * tp_mult)
        row.loc["stop_loss"] = float(max(base_sl, 1e-8) * sl_mult)
        row.loc["max_hold_bars"] = int(max(1, round(base_hold * hold_mult)))
        out.iloc[int(i)] = row
    out.loc[~active] = out.loc[~active].apply(_zero_row, axis=1)
    return out


def _metrics_rows(evaluator: OfficialCost3, splits: list[tuple[str, pd.DataFrame, dict[str, pd.DataFrame]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, frame, variants in splits:
        for variant, dec in variants.items():
            rows.append({"split": split, "variant": variant, **evaluator(frame, dec)})
    return pd.DataFrame(rows)


def _score(row: pd.Series) -> float:
    trades = int(row.get("trades", 0) or 0)
    pnl = float(row.get("pnl", 0.0) or 0.0)
    if trades < 30 or pnl <= 0.0:
        return -1e9 + pnl
    return pnl + 110.0 * float(row.get("wr", 0.0) or 0.0) - 0.45 * abs(float(row.get("mdd", 0.0) or 0.0)) + 0.01 * trades


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    ap.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--primary-parent", type=Path, default=DEFAULT_PARENT_DIR / "primary/parent.pkl")
    ap.add_argument("--primary-summary", type=Path, default=DEFAULT_PARENT_DIR / "primary/summary.json")
    ap.add_argument("--fallback-parent", type=Path, default=DEFAULT_PARENT_DIR / "fallback/parent.pkl")
    ap.add_argument("--fallback-summary", type=Path, default=DEFAULT_PARENT_DIR / "fallback/summary.json")
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--candidate-count", type=int, default=20)
    ap.add_argument("--max-target-rows", type=int, default=2500)
    ap.add_argument("--thresholds", default="0.35,0.45,0.55,0.65")
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")

    train_all = _rename_clean4_v2(_read(args.train_csv))
    eval_df = _rename_clean4_v2(_read(args.eval_csv))
    _assert_clean(train_all, name="train_all")
    _assert_clean(eval_df, name="eval")
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    primary = joblib.load(args.primary_parent)
    fallback = joblib.load(args.fallback_parent)
    primary_rt = _load_scale_runtime_any(args.primary_summary)
    fallback_rt = _load_scale_runtime_any(args.fallback_summary)
    p_train = _predict_scaled(primary, train_df, primary_rt).reset_index(drop=True)
    p_val = _predict_scaled(primary, val_df, primary_rt).reset_index(drop=True)
    p_eval = _predict_scaled(primary, eval_df, primary_rt).reset_index(drop=True)
    f_train = _predict_scaled(fallback, train_df, fallback_rt).reset_index(drop=True)
    f_val = _predict_scaled(fallback, val_df, fallback_rt).reset_index(drop=True)
    f_eval = _predict_scaled(fallback, eval_df, fallback_rt).reset_index(drop=True)
    combo_train = _combine_primary_fallback(p_train, f_train).reset_index(drop=True)
    combo_val = _combine_primary_fallback(p_val, f_val).reset_index(drop=True)
    combo_eval = _combine_primary_fallback(p_eval, f_eval).reset_index(drop=True)

    state_train = _state_frame(train_df, p_train, f_train, combo_train)
    state_val = _state_frame(val_df, p_val, f_val, combo_val)
    state_eval = _state_frame(eval_df, p_eval, f_eval, combo_eval)
    norm = _fit_norm(state_train)
    x_train = _apply_norm(state_train, norm)
    x_val = _apply_norm(state_val, norm)
    x_eval = _apply_norm(state_eval, norm)

    evaluator = OfficialCost3()
    fee = float(evaluator.fee)
    slip = float(evaluator.slip)
    thresholds = [float(x) for x in str(args.thresholds).split(",") if x.strip()]

    variants_train: dict[str, pd.DataFrame] = {"baseline_combo": combo_train}
    variants_val: dict[str, pd.DataFrame] = {"baseline_combo": combo_val}
    variants_eval: dict[str, pd.DataFrame] = {"baseline_combo": combo_eval}
    diagnostics: dict[str, Any] = {}
    saved_models: dict[str, str] = {}

    print(
        json.dumps(
            {
                "stage": "start",
                "device": str(device),
                "state_dim": int(x_train.shape[1]),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "oos_rows": len(eval_df),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for bi, bounds in enumerate(BOUNDS):
        print(
            json.dumps(
                {
                    "stage": "target_start",
                    "bounds": bounds.name,
                    "candidate_count": int(args.candidate_count),
                    "max_target_rows": int(args.max_target_rows),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        x_target, y_target, sample_weight, target_diag = _build_targets(
            train_df,
            combo_train,
            x_train,
            bounds=bounds,
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            candidate_count=int(args.candidate_count),
            max_target_rows=int(args.max_target_rows),
        )
        print(json.dumps({"stage": "target_done", "bounds": bounds.name, "diag": target_diag}, ensure_ascii=False), flush=True)
        model, train_diag = _train_regressor(
            x_target,
            y_target,
            sample_weight,
            device=device,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            seed=9052900 + bi,
        )
        pred_train = _predict(model, x_train, device=device)
        pred_val = _predict(model, x_val, device=device)
        pred_eval = _predict(model, x_eval, device=device)
        for threshold in thresholds:
            name = f"{bounds.name}_thr{threshold:.2f}"
            variants_train[name] = _compose_regression_decisions(combo_train, pred_train, bounds=bounds, trade_threshold=threshold)
            variants_val[name] = _compose_regression_decisions(combo_val, pred_val, bounds=bounds, trade_threshold=threshold)
            variants_eval[name] = _compose_regression_decisions(combo_eval, pred_eval, bounds=bounds, trade_threshold=threshold)
        model_path = out_dir / f"{bounds.name}.pt"
        torch.save(
            {
                "model_id": MODEL_ID,
                "bounds": asdict(bounds),
                "state_columns": list(norm["columns"]),
                "state_normalizer": norm,
                "model_state_dict": model.state_dict(),
                "network": {"hidden_dim": 256, "output": ["trade_score", "mult", "leverage", "tp_mult", "sl_mult", "hold_mult"]},
            },
            model_path,
        )
        saved_models[bounds.name] = str(model_path)
        diagnostics[bounds.name] = {"target": target_diag, "train": train_diag}

    grid = _metrics_rows(
        evaluator,
        [
            ("train", train_df, variants_train),
            ("val", val_df, variants_val),
            ("oos", eval_df, variants_eval),
        ],
    )
    grid["selection_score"] = grid.apply(_score, axis=1)
    grid_path = out_dir / "grid.csv"
    grid.to_csv(grid_path, index=False)

    val_rank = grid[(grid["split"] == "val") & (grid["variant"] != "baseline_combo")].sort_values("selection_score", ascending=False)
    selected = str(val_rank.iloc[0]["variant"]) if len(val_rank) else "none"
    selected_val = grid[(grid["split"] == "val") & (grid["variant"] == selected)].iloc[0].to_dict() if selected != "none" else {}
    selected_oos = grid[(grid["split"] == "oos") & (grid["variant"] == selected)].iloc[0].to_dict() if selected != "none" else {}
    baseline_oos = grid[(grid["split"] == "oos") & (grid["variant"] == "baseline_combo")].iloc[0].to_dict()

    summary = {
        "model_id": MODEL_ID,
        "design": "Bounded regression risk heads. Parent/fallback combo owns entry and direction; regression predicts trade_score, notional multiplier, leverage, TP multiplier, SL multiplier, and hold multiplier inside explicit bounds.",
        "live_wired": False,
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "bounds": [asdict(b) for b in BOUNDS],
        "training": {"device": str(device), "epochs": int(args.epochs), "lr": float(args.lr), "state_dim": int(x_train.shape[1]), "diagnostics": diagnostics},
        "selected": {"variant": selected, "val": selected_val, "oos": selected_oos},
        "baseline_oos": baseline_oos,
        "artifacts": {"grid": str(grid_path), "models": saved_models},
        "audit": {"feature_contract_fail_fast": True, "legacy_compat_alias": False, "live_overwrite": False, "selection_uses_2026": False},
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "selected": summary["selected"], "baseline_oos": baseline_oos}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
