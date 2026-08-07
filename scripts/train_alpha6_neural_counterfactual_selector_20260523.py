#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha6_catboost_entry_quality_exit_policy_20260522 import (  # noqa: E402
    TARGET_BUCKET_TO_HORIZON,
    _exit_close_prob,
    _exit_state_vec,
    _threshold_for_bucket,
)
from scripts.alpha6_catboost_5head_policy_20260522 import _days, _fill_price  # noqa: E402
from scripts.train_alpha6_dsac_ensemble_router_20260523 import (  # noqa: E402
    MODEL_SPECS,
    RouterData,
    _build_base_features,
    _load_router_data,
)


EXPERT_NAMES = [name for name, _ in MODEL_SPECS]


@dataclass
class CandidateTable:
    x: np.ndarray
    y: np.ndarray
    entry_idx: np.ndarray
    expert_idx: np.ndarray
    side: np.ndarray
    pnl_pct: np.ndarray
    mae_pct: np.ndarray
    mfe_pct: np.ndarray
    hold_bars: np.ndarray


class SelectorNet(nn.Module):
    def __init__(self, dim: int, hidden: int = 192, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class OracleClassifierNet(nn.Module):
    def __init__(self, dim: int, classes: int, hidden: int = 192, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _filter_val(data: RouterData) -> RouterData:
    split = data.frame["dataset_split"].astype(str).str.lower().to_numpy()
    mask = split != "train"
    frame = data.frame.loc[mask].reset_index(drop=True)
    preds = [p.loc[mask].reset_index(drop=True) for p in data.preds]
    xs = [np.asarray(x[mask], dtype=np.float64) for x in data.xs]
    base_x, names = _build_base_features(frame, preds)
    return RouterData(
        frame=frame,
        preds=preds,
        xs=xs,
        bundles=data.bundles,
        exit_models=data.exit_models,
        exit_model_ids=[np.zeros(len(frame), dtype=np.int64) for _ in data.preds],
        base_x=base_x,
        base_names=names,
        thresholds=data.thresholds,
        exit_thresholds=data.exit_thresholds,
    )


def _split_points(n: int, purge_bars: int) -> dict[str, tuple[int, int]]:
    train_end = int(n * 0.50)
    calib_end = int(n * 0.75)
    return {
        "meta_train": (0, max(0, train_end - purge_bars)),
        "calib": (train_end, max(train_end, calib_end - purge_bars)),
        "test": (calib_end, n - 2),
        "full_val": (0, n - 2),
    }


def _desired(data: RouterData, expert_idx: int, i: int) -> int:
    row = data.preds[expert_idx].iloc[i]
    return int(row.action) if float(row.quality) >= float(data.thresholds[expert_idx]) else 0


def _candidate_feature(data: RouterData, expert_idx: int, i: int) -> np.ndarray:
    p = data.preds[expert_idx].iloc[i]
    onehot = np.zeros(len(data.preds), dtype=np.float32)
    onehot[expert_idx] = 1.0
    q = float(p.quality)
    thr = float(data.thresholds[expert_idx])
    side = 1.0 if int(p.action) == 1 else -1.0
    cand = np.asarray(
        [
            side,
            float(p.cash_prob),
            float(p.long_prob),
            float(p.short_prob),
            float(p.confidence),
            q,
            q - thr,
            q / max(abs(thr), 1e-9),
            float(p.target_horizon) / 96.0,
            float(p.target_bucket) / 4.0,
            float(data.exit_thresholds[expert_idx]),
        ],
        dtype=np.float32,
    )
    return np.concatenate([data.base_x[i].astype(np.float32), onehot, cand])


def _simulate_candidate(
    data: RouterData,
    expert_idx: int,
    entry_signal_idx: int,
    *,
    fee: float,
    slip: float,
    min_exit_hold: int,
    state_horizon: int,
    end_idx: int,
    max_sim_bars: int,
) -> tuple[float, float, float, int, str]:
    frame = data.frame
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    expert_pred = data.preds[expert_idx]
    row = expert_pred.iloc[entry_signal_idx]
    side = 1 if int(row.action) == 1 else -1
    fill_i = min(entry_signal_idx + 1, len(frame) - 1)
    entry = _fill_price(frame, fill_i, side, slip, entry=True)
    exposure = 0.25
    cash = 1.0 - fee * exposure
    target_horizon = int(np.clip(int(row.target_horizon), 2, state_horizon))
    target_bucket = int(np.clip(int(row.target_bucket), 0, 4))
    hold = 0
    mae = 0.0
    mfe = 0.0
    exit_model = data.bundles[expert_idx]["exit_model"]
    expected = data.bundles[expert_idx].get("expected_return_by_bucket") or {k: 0.01 for k in TARGET_BUCKET_TO_HORIZON}
    exit_meta = data.bundles[expert_idx].get("exit_meta", {})
    end = min(int(end_idx), entry_signal_idx + int(max_sim_bars), len(frame) - 2)
    reason = "end"
    exit_signal_idx = end
    for j in range(entry_signal_idx + 1, end + 1):
        hold += 1
        px = float(close[j])
        raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
        mae = max(mae, max(0.0, -raw * exposure))
        mfe = max(mfe, max(0.0, raw * exposure))
        if hold < int(min_exit_hold):
            continue
        state = _exit_state_vec(
            frame,
            side=side,
            entry_idx=entry_signal_idx,
            current_idx=j,
            entry_px=entry,
            px=px,
            hold=hold,
            horizon=target_horizon,
            mae=mae,
            mfe=mfe,
            target_bucket=target_bucket,
            regime_drift=bool(exit_meta.get("regime_drift", False)),
            capture_ratio=bool(exit_meta.get("capture_ratio", False)),
            expected_return=float(expected.get(target_bucket, 0.01)),
        )
        close_prob = _exit_close_prob(exit_model, data.xs[expert_idx][j], state)
        if close_prob >= _threshold_for_bucket(float(data.exit_thresholds[expert_idx]), target_bucket):
            reason = "exit_model"
            exit_signal_idx = j
            break
    fill_exit_i = min(exit_signal_idx + 1, len(frame) - 1)
    exit_px = _fill_price(frame, fill_exit_i, side, slip, entry=False)
    raw = (exit_px - entry) / max(entry, 1e-12) if side > 0 else (entry - exit_px) / max(entry, 1e-12)
    before_fee_cash = cash * (1.0 + raw * exposure)
    cash = before_fee_cash - cash * fee * exposure
    return float((cash - 1.0) * 100.0), float(mae * 100.0), float(mfe * 100.0), int(hold), reason


def _build_candidates(
    data: RouterData,
    *,
    start: int,
    end: int,
    fee: float,
    slip: float,
    min_exit_hold: int,
    state_horizon: int,
    candidate_stride: int,
    transition_only: bool,
    max_sim_bars: int,
) -> CandidateTable:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    entry_idx: list[int] = []
    expert_idx: list[int] = []
    side: list[int] = []
    pnl_pct: list[float] = []
    mae_pct: list[float] = []
    mfe_pct: list[float] = []
    hold_bars: list[int] = []
    n = len(data.frame)
    lo = max(0, int(start))
    hi = min(int(end), n - 2)
    for i in range(lo, hi, max(1, int(candidate_stride))):
        for mi in range(len(data.preds)):
            desired = _desired(data, mi, i)
            if desired == 0:
                continue
            if transition_only and i > 0 and _desired(data, mi, i - 1) == desired:
                continue
            pnl, mae, mfe, hold, _ = _simulate_candidate(
                data,
                mi,
                i,
                fee=fee,
                slip=slip,
                min_exit_hold=min_exit_hold,
                state_horizon=state_horizon,
                end_idx=hi,
                max_sim_bars=max_sim_bars,
            )
            utility = pnl - 0.35 * mae + 0.05 * min(mfe, 5.0) - 0.002 * hold
            xs.append(_candidate_feature(data, mi, i))
            ys.append(float(utility))
            entry_idx.append(i)
            expert_idx.append(mi)
            side.append(1 if desired == 1 else -1)
            pnl_pct.append(pnl)
            mae_pct.append(mae)
            mfe_pct.append(mfe)
            hold_bars.append(hold)
        if (i - lo) > 0 and (i - lo) % 2000 == 0:
            print(f"[candidates] progress rows={i - lo}/{hi - lo} candidates={len(xs)}", flush=True)
    if not xs:
        raise RuntimeError(f"no candidates in range {start}:{end}")
    return CandidateTable(
        x=np.vstack(xs).astype(np.float32),
        y=np.asarray(ys, dtype=np.float32),
        entry_idx=np.asarray(entry_idx, dtype=np.int64),
        expert_idx=np.asarray(expert_idx, dtype=np.int64),
        side=np.asarray(side, dtype=np.int64),
        pnl_pct=np.asarray(pnl_pct, dtype=np.float32),
        mae_pct=np.asarray(mae_pct, dtype=np.float32),
        mfe_pct=np.asarray(mfe_pct, dtype=np.float32),
        hold_bars=np.asarray(hold_bars, dtype=np.int64),
    )


def _fit_selector(
    train: CandidateTable,
    calib: CandidateTable,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[SelectorNet, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean = train.x.mean(axis=0)
    std = train.x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    tx = torch.tensor((train.x - mean) / std, dtype=torch.float32)
    ty = torch.tensor(train.y, dtype=torch.float32)
    cx = torch.tensor((calib.x - mean) / std, dtype=torch.float32)
    cy = torch.tensor(calib.y, dtype=torch.float32)
    model = SelectorNet(tx.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None
    best_loss = float("inf")
    n = len(tx)
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n)
        model.train()
        losses = []
        for s in range(0, n, batch_size):
            idx = order[s : s + batch_size]
            pred = model(tx[idx])
            loss = F.smooth_l1_loss(pred, ty[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(F.smooth_l1_loss(model(cx), cy).cpu())
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % max(1, epochs // 5) == 0:
            print(f"[train] epoch={epoch} train_loss={np.mean(losses):.6f} calib_loss={val_loss:.6f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    meta = {"mean": mean.tolist(), "std": std.tolist(), "best_calib_loss": best_loss}
    return model, meta


def _oracle_dataset(data: RouterData, table: CandidateTable, *, min_utility: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    idxs: list[int] = []
    for idx in np.unique(table.entry_idx):
        loc = np.flatnonzero(table.entry_idx == idx)
        best_loc = loc[int(np.argmax(table.y[loc]))]
        best_utility = float(table.y[best_loc])
        label = int(table.expert_idx[best_loc]) + 1 if best_utility >= float(min_utility) else 0
        rows.append(data.base_x[int(idx)].astype(np.float32))
        labels.append(label)
        idxs.append(int(idx))
    return np.vstack(rows).astype(np.float32), np.asarray(labels, dtype=np.int64), np.asarray(idxs, dtype=np.int64)


def _fit_oracle_classifier(
    data: RouterData,
    train: CandidateTable,
    calib: CandidateTable,
    *,
    min_utility: float,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[OracleClassifierNet, dict[str, Any]]:
    torch.manual_seed(seed + 17)
    np.random.seed(seed + 17)
    x_train, y_train, _ = _oracle_dataset(data, train, min_utility=min_utility)
    x_calib, y_calib, _ = _oracle_dataset(data, calib, min_utility=min_utility)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    tx = torch.tensor((x_train - mean) / std, dtype=torch.float32)
    ty = torch.tensor(y_train, dtype=torch.long)
    cx = torch.tensor((x_calib - mean) / std, dtype=torch.float32)
    cy = torch.tensor(y_calib, dtype=torch.long)
    classes = len(data.preds) + 1
    counts = np.bincount(y_train, minlength=classes).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    weights[0] *= 0.65
    model = OracleClassifierNet(tx.shape[1], classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None
    best_loss = float("inf")
    w = torch.tensor(weights, dtype=torch.float32)
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(tx))
        model.train()
        losses = []
        for s in range(0, len(tx), batch_size):
            idx = order[s : s + batch_size]
            logits = model(tx[idx])
            loss = F.cross_entropy(logits, ty[idx], weight=w)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(F.cross_entropy(model(cx), cy, weight=w).cpu())
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % max(1, epochs // 5) == 0:
            pred = model(cx).argmax(dim=1)
            acc = float((pred == cy).float().mean().cpu())
            print(f"[clf] epoch={epoch} train_loss={np.mean(losses):.6f} calib_loss={val_loss:.6f} calib_acc={acc:.3f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "best_calib_loss": best_loss,
        "class_counts": counts.tolist(),
        "min_utility": float(min_utility),
    }


def _score_candidates(model: SelectorNet, meta: dict[str, Any], data: RouterData) -> np.ndarray:
    n = len(data.frame)
    m = len(data.preds)
    scores = np.full((n, m), -np.inf, dtype=np.float32)
    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    rows: list[np.ndarray] = []
    locs: list[tuple[int, int]] = []
    for i in range(n - 2):
        for mi in range(m):
            if _desired(data, mi, i) == 0:
                continue
            rows.append(_candidate_feature(data, mi, i))
            locs.append((i, mi))
    if not rows:
        return scores
    x = torch.tensor((np.vstack(rows).astype(np.float32) - mean) / std, dtype=torch.float32)
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, len(x), 8192):
            out.append(model(x[s : s + 8192]).cpu().numpy())
    pred = np.concatenate(out)
    for (i, mi), score in zip(locs, pred):
        scores[i, mi] = float(score)
    return scores


def _backtest_selector(
    data: RouterData,
    scores: np.ndarray,
    *,
    start: int,
    end: int,
    score_threshold: float,
    fee: float,
    slip: float,
    min_exit_hold: int,
    state_horizon: int,
) -> dict[str, Any]:
    frame = data.frame
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    start = max(0, int(start))
    end = min(int(end), len(frame) - 2)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    side = 0
    active = -1
    entry = 0.0
    entry_idx = 0
    entry_equity = 1.0
    hold = 0
    mae = mfe = exposure = 0.0
    target_horizon = int(state_horizon)
    target_bucket = 4
    trades = wins = long_entries = short_entries = exit_model_closes = 0
    exits: dict[str, int] = {}

    def equity(i: int) -> float:
        if side == 0:
            return cash
        px = close[int(np.clip(i, 0, len(close) - 1))]
        raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
        return cash * (1.0 + raw * exposure)

    def enter(i: int, mi: int) -> None:
        nonlocal side, active, entry, entry_idx, entry_equity, hold, mae, mfe, exposure, target_horizon, target_bucket, cash
        row = data.preds[mi].iloc[i]
        fill_i = min(i + 1, len(frame) - 1)
        side = 1 if int(row.action) == 1 else -1
        active = int(mi)
        entry_idx = int(i)
        exposure = 0.25
        target_horizon = int(np.clip(int(row.target_horizon), 2, state_horizon))
        target_bucket = int(np.clip(int(row.target_bucket), 0, 4))
        entry = _fill_price(frame, fill_i, side, slip, entry=True)
        entry_equity = cash
        cash -= cash * fee * exposure
        hold = 0
        mae = mfe = 0.0

    def exit_pos(i: int, reason: str) -> None:
        nonlocal side, active, entry, cash, hold, mae, mfe, exposure, target_horizon, target_bucket
        nonlocal trades, wins, long_entries, short_entries, exit_model_closes
        fill_px = _fill_price(frame, min(i + 1, len(frame) - 1), side, slip, entry=False)
        raw = (fill_px - entry) / max(entry, 1e-12) if side > 0 else (entry - fill_px) / max(entry, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * exposure)
        cash -= before * fee * exposure
        trades += 1
        wins += int(cash > entry_equity)
        long_entries += int(side > 0)
        short_entries += int(side < 0)
        exits[reason] = exits.get(reason, 0) + 1
        side = 0
        active = -1
        entry = 0.0
        hold = 0
        mae = mfe = exposure = 0.0
        target_horizon = int(state_horizon)
        target_bucket = 4

    for i in range(start, end):
        if side != 0:
            hold += 1
            px = float(close[i])
            raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
            mae = max(mae, max(0.0, -raw * exposure))
            mfe = max(mfe, max(0.0, raw * exposure))
            if hold >= int(min_exit_hold):
                bundle = data.bundles[active]
                expected = bundle.get("expected_return_by_bucket") or {k: 0.01 for k in TARGET_BUCKET_TO_HORIZON}
                exit_meta = bundle.get("exit_meta", {})
                state = _exit_state_vec(
                    frame,
                    side=side,
                    entry_idx=entry_idx,
                    current_idx=i,
                    entry_px=entry,
                    px=px,
                    hold=hold,
                    horizon=int(target_horizon),
                    mae=mae,
                    mfe=mfe,
                    target_bucket=target_bucket,
                    regime_drift=bool(exit_meta.get("regime_drift", False)),
                    capture_ratio=bool(exit_meta.get("capture_ratio", False)),
                    expected_return=float(expected.get(target_bucket, 0.01)),
                )
                close_prob = _exit_close_prob(bundle["exit_model"], data.xs[active][i], state)
                if close_prob >= _threshold_for_bucket(float(data.exit_thresholds[active]), target_bucket):
                    exit_model_closes += 1
                    exit_pos(i, "exit_model")
        eq = equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if side == 0:
            mi = int(np.argmax(scores[i]))
            if np.isfinite(scores[i, mi]) and float(scores[i, mi]) >= float(score_threshold):
                enter(i, mi)
    if side != 0:
        exit_pos(end, "end")
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "calmar": float(((cash - 1.0) * 100.0) / max(abs(mdd * 100.0), 1e-12)),
        "trades": int(trades),
        "trades_per_day": float(trades / _days(frame.iloc[start : end + 1])),
        "wr": float(wins / max(trades, 1)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_model_closes": int(exit_model_closes),
        "exits": exits,
    }


def _threshold_grid(scores: np.ndarray, start: int, end: int) -> np.ndarray:
    vals = scores[start:end]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.asarray([0.0])
    qs = np.linspace(0.40, 0.98, 30)
    grid = np.unique(np.quantile(vals, qs))
    return grid.astype(np.float64)


def _classifier_probs(model: OracleClassifierNet, meta: dict[str, Any], data: RouterData) -> np.ndarray:
    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    x = torch.tensor((data.base_x.astype(np.float32) - mean) / std, dtype=torch.float32)
    outs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(x), 8192):
            outs.append(torch.softmax(model(x[s : s + 8192]), dim=1).cpu().numpy())
    return np.vstack(outs).astype(np.float32)


def _backtest_classifier(
    data: RouterData,
    probs: np.ndarray,
    *,
    start: int,
    end: int,
    confidence_threshold: float,
    fee: float,
    slip: float,
    min_exit_hold: int,
    state_horizon: int,
) -> dict[str, Any]:
    scores = np.full((len(data.frame), len(data.preds)), -np.inf, dtype=np.float32)
    for i in range(len(data.frame) - 2):
        cls = int(np.argmax(probs[i, 1:])) + 1
        mi = cls - 1
        if float(probs[i, cls]) >= float(confidence_threshold) and _desired(data, mi, i) != 0:
            scores[i, mi] = float(probs[i, cls])
    return _backtest_selector(
        data,
        scores,
        start=start,
        end=end,
        score_threshold=0.0,
        fee=fee,
        slip=slip,
        min_exit_hold=min_exit_hold,
        state_horizon=state_horizon,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="current_tail111")
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--min-exit-hold", type=int, default=2)
    ap.add_argument("--state-horizon", type=int, default=96)
    ap.add_argument("--max-sim-bars", type=int, default=96)
    ap.add_argument("--purge-bars", type=int, default=96)
    ap.add_argument("--candidate-stride", type=int, default=1)
    ap.add_argument("--all-desired-candidates", action="store_true")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--oracle-min-utility", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/alpha6_neural_cf_selector_20260523")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fee = 0.0004 * float(args.cost_mult)
    slip = 0.00015 * float(args.cost_mult)
    print("[load] CatBoost expert outputs", flush=True)
    data = _filter_val(_load_router_data(args.variant))
    splits = _split_points(len(data.frame), args.purge_bars)
    print(f"[split] {splits}", flush=True)
    tables: dict[str, CandidateTable] = {}
    for name in ("meta_train", "calib", "test"):
        s, e = splits[name]
        print(f"[candidates] {name} rows={s}:{e}", flush=True)
        tables[name] = _build_candidates(
            data,
            start=s,
            end=e,
            fee=fee,
            slip=slip,
            min_exit_hold=args.min_exit_hold,
            state_horizon=args.state_horizon,
            candidate_stride=args.candidate_stride,
            transition_only=not bool(args.all_desired_candidates),
            max_sim_bars=args.max_sim_bars,
        )
        print(
            f"[candidates] {name} n={len(tables[name].x)} y_mean={tables[name].y.mean():.4f} "
            f"pnl_mean={tables[name].pnl_pct.mean():.4f} win={(tables[name].pnl_pct > 0).mean():.3f}",
            flush=True,
        )
    model, meta = _fit_selector(
        tables["meta_train"],
        tables["calib"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    scores = _score_candidates(model, meta, data)
    calib_start, calib_end = splits["calib"]
    test_start, test_end = splits["test"]
    rows = []
    best = None
    for thr in _threshold_grid(scores, calib_start, calib_end):
        bt = _backtest_selector(
            data,
            scores,
            start=calib_start,
            end=calib_end,
            score_threshold=float(thr),
            fee=fee,
            slip=slip,
            min_exit_hold=args.min_exit_hold,
            state_horizon=args.state_horizon,
        )
        row = {"threshold": float(thr), "split": "calib", **bt}
        rows.append(row)
        score = bt["calmar"] if bt["trades"] >= 5 else -1e6 + bt["pnl"]
        if best is None or score > best[0]:
            best = (score, float(thr), bt)
    assert best is not None
    best_thr = float(best[1])
    for split in ("meta_train", "calib", "test", "full_val"):
        s, e = splits[split]
        bt = _backtest_selector(
            data,
            scores,
            start=s,
            end=e,
            score_threshold=best_thr,
            fee=fee,
            slip=slip,
            min_exit_hold=args.min_exit_hold,
            state_horizon=args.state_horizon,
        )
        rows.append({"threshold": best_thr, "split": split, **bt})
        print(f"[bt] split={split} thr={best_thr:.6f} {bt}", flush=True)

    clf, clf_meta = _fit_oracle_classifier(
        data,
        tables["meta_train"],
        tables["calib"],
        min_utility=float(args.oracle_min_utility),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    probs = _classifier_probs(clf, clf_meta, data)
    clf_rows = []
    best_clf = None
    calib_start, calib_end = splits["calib"]
    calib_conf = probs[calib_start:calib_end, 1:].max(axis=1)
    conf_grid = np.unique(np.quantile(calib_conf[np.isfinite(calib_conf)], np.linspace(0.40, 0.98, 30)))
    for conf in conf_grid:
        bt = _backtest_classifier(
            data,
            probs,
            start=calib_start,
            end=calib_end,
            confidence_threshold=float(conf),
            fee=fee,
            slip=slip,
            min_exit_hold=args.min_exit_hold,
            state_horizon=args.state_horizon,
        )
        row = {"confidence_threshold": float(conf), "split": "calib", **bt}
        clf_rows.append(row)
        score = bt["calmar"] if bt["trades"] >= 5 else -1e6 + bt["pnl"]
        if best_clf is None or score > best_clf[0]:
            best_clf = (score, float(conf), bt)
    assert best_clf is not None
    best_conf = float(best_clf[1])
    for split in ("meta_train", "calib", "test", "full_val"):
        s, e = splits[split]
        bt = _backtest_classifier(
            data,
            probs,
            start=s,
            end=e,
            confidence_threshold=best_conf,
            fee=fee,
            slip=slip,
            min_exit_hold=args.min_exit_hold,
            state_horizon=args.state_horizon,
        )
        clf_rows.append({"confidence_threshold": best_conf, "split": split, **bt})
        print(f"[clf_bt] split={split} conf={best_conf:.6f} {bt}", flush=True)

    ranking = pd.DataFrame(rows)
    ranking.to_csv(args.out_dir / "neural_cf_selector_results.csv", index=False)
    pd.DataFrame(clf_rows).to_csv(args.out_dir / "oracle_classifier_results.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "meta": meta, "expert_names": EXPERT_NAMES}, args.out_dir / "selector.pt")
    torch.save({"state_dict": clf.state_dict(), "meta": clf_meta, "expert_names": EXPERT_NAMES}, args.out_dir / "oracle_classifier.pt")
    summary = {
        "best_threshold": best_thr,
        "best_classifier_confidence": best_conf,
        "splits": splits,
        "cost_mult": float(args.cost_mult),
        "candidate_counts": {k: int(len(v.x)) for k, v in tables.items()},
        "best_calib_loss": float(meta["best_calib_loss"]),
        "best_classifier_calib_loss": float(clf_meta["best_calib_loss"]),
        "classifier_class_counts": clf_meta["class_counts"],
        "results": rows,
        "classifier_results": clf_rows,
        "expert_names": EXPERT_NAMES,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"[out] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
