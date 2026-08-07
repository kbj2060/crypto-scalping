#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    _combine_primary_fallback,
    _load_best_scale_runtime,
    _predict_scaled,
    _read,
)
from scripts.rebuild_alpha7_v2_only_high_turnover_20260526 import _rename_clean4_v2  # noqa: E402
from scripts.sweep_alpha8_origin_scaled_combo_20260529 import OfficialCost3  # noqa: E402
from scripts.train_eval_alpha7_directional_dsac_router_20260529 import (  # noqa: E402
    DECISION_COLS,
    EVAL_CSV,
    FORBIDDEN_PREFIXES,
    SOURCE_COLS,
    TRAIN_CSV,
    Actor,
    Critic,
    DatasetBundle,
    _apply_norm,
    _audit_frame_contract,
    _directional_features,
    _fit_norm,
    _policy_action,
    _safe_num,
    _train_dsac_offline,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha8_dsac_iqn_risk_selector_20260529"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

TEACHER_COLS = [
    "teacher_long_edge",
    "teacher_short_edge",
    "teacher_side_margin",
    "teacher_side_disagreement",
    "teacher_quantile_skew",
    "teacher_uncertainty",
    "teacher_tail_warning",
]


@dataclass(frozen=True)
class RiskTemplate:
    name: str
    mult: float
    cap: float
    leverage: float
    tp_mult: float
    sl_mult: float
    hold_mult: float
    veto: bool = False


TEMPLATES: tuple[RiskTemplate, ...] = (
    RiskTemplate("veto", 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, True),
    RiskTemplate("defensive_110_tp0175_sl5", 1.10, 7.5, 5.0, 0.175, 5.0, 0.75),
    RiskTemplate("defensive_110_tp020_sl5", 1.10, 7.5, 5.0, 0.200, 5.0, 0.75),
    RiskTemplate("stable_120_tp0175_sl5", 1.20, 7.5, 5.0, 0.175, 5.0, 0.75),
    RiskTemplate("stable_120_tp020_sl5", 1.20, 7.5, 5.0, 0.200, 5.0, 0.75),
    RiskTemplate("balanced_135_tp018_sl4", 1.35, 7.5, 5.0, 0.180, 4.0, 0.75),
    RiskTemplate("runner_150_tp0175_sl5", 1.50, 7.5, 5.0, 0.175, 5.0, 0.75),
    RiskTemplate("aggressive_175_tp020_sl5", 1.75, 5.0, 5.0, 0.200, 5.0, 0.75),
    RiskTemplate("aggressive_200_tp018_sl4", 2.00, 5.0, 5.0, 0.180, 4.0, 0.75),
    RiskTemplate("micro_100_tp012_sl4", 1.00, 5.0, 5.0, 0.120, 4.0, 0.50),
    RiskTemplate("micro_105_tp012_sl4", 1.05, 5.0, 5.0, 0.120, 4.0, 0.50),
    RiskTemplate("highwr_100_tp011_sl4", 1.00, 5.0, 5.0, 0.110, 4.0, 0.75),
)


class IQNRiskSelector(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_cos: int = 64) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.n_cos = int(n_cos)
        self.state = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.quantile = nn.Sequential(nn.Linear(n_cos, hidden_dim), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        state = self.state(x).unsqueeze(1)
        basis_idx = torch.arange(1, self.n_cos + 1, device=x.device, dtype=x.dtype).view(1, 1, -1)
        tau_basis = torch.cos(math.pi * tau.unsqueeze(-1) * basis_idx)
        tau_emb = self.quantile(tau_basis)
        return self.head(state * tau_emb)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _assert_clean(df: pd.DataFrame, *, name: str) -> None:
    bad = [c for c in df.columns if str(c).startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise RuntimeError(f"{name} contains forbidden legacy regime columns: {bad[:20]}")


def _load_scale_runtime_any(summary_path: Path) -> alpha2.Alpha2Runtime | None:
    rt = _load_best_scale_runtime(summary_path)
    if rt is not None:
        return rt
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best = summary.get("best_by_validation", {})
    raw = best.get("runtime")
    if not raw:
        return None
    return alpha2.Alpha2Runtime(
        name=str(raw["name"]),
        confidence=float(raw["confidence"]),
        parent_notional_scale=float(raw["parent_notional_scale"]),
        max_notional=float(raw["max_notional"]),
    )


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    notional = pd.to_numeric(dec["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return (action != ACTION_CASH) & (side != 0) & (notional > 0.0)


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).to_numpy(dtype=np.float64)


def _state_frame(frame: pd.DataFrame, primary: pd.DataFrame, fallback: pd.DataFrame, combo: pd.DataFrame) -> pd.DataFrame:
    _audit_frame_contract(frame, name="alpha8_risk_selector_state")
    _assert_clean(frame, name="alpha8_risk_selector_state")

    parts: list[pd.DataFrame] = [_directional_features(frame).reset_index(drop=True)]
    parts.append(pd.DataFrame({c: _safe_num(frame, c) for c in SOURCE_COLS}, index=frame.index).reset_index(drop=True))
    for col in TEACHER_COLS:
        if col not in frame.columns:
            raise RuntimeError(f"certified teacher feature missing: {col}")
    parts.append(pd.DataFrame({c: _safe_num(frame, c) for c in TEACHER_COLS}, index=frame.index).reset_index(drop=True))

    for prefix, dec in (("primary", primary), ("fallback", fallback), ("combo", combo)):
        d = pd.DataFrame(index=frame.index)
        for col in DECISION_COLS:
            if col not in dec.columns:
                raise RuntimeError(f"{prefix} decision missing column: {col}")
            d[f"{prefix}_{col}"] = _num(dec, col)
        d[f"{prefix}_rr"] = d[f"{prefix}_take_profit"] / np.maximum(np.abs(d[f"{prefix}_stop_loss"]), 1e-8)
        d[f"{prefix}_margin_fraction"] = d[f"{prefix}_notional_exposure"] / np.maximum(d[f"{prefix}_leverage"], 1e-8)
        parts.append(d.reset_index(drop=True))

    primary_active = _active(primary)
    fallback_active = _active(fallback)
    combo_active = _active(combo)
    origin = np.zeros(len(frame), dtype=np.float64)
    origin[primary_active] = 1.0
    origin[(~primary_active) & fallback_active] = 2.0

    meta = pd.DataFrame(index=frame.index)
    meta["origin_primary"] = (origin == 1.0).astype(float)
    meta["origin_fallback"] = (origin == 2.0).astype(float)
    meta["combo_active"] = combo_active.astype(float)
    meta["primary_fallback_side_agree"] = (_num(primary, "side").astype(int) == _num(fallback, "side").astype(int)).astype(float)
    meta["primary_fallback_side_disagree"] = (
        (_num(primary, "side").astype(int) != _num(fallback, "side").astype(int))
        & (_num(primary, "side").astype(int) != 0)
        & (_num(fallback, "side").astype(int) != 0)
    ).astype(float)
    meta["combo_side_x_confidence"] = _num(combo, "side") * _num(combo, "confidence")
    meta["combo_side_x_quality"] = _num(combo, "side") * _num(combo, "quality_score")
    meta["primary_minus_fallback_conf"] = _num(primary, "confidence") - _num(fallback, "confidence")
    meta["primary_minus_fallback_quality"] = _num(primary, "quality_score") - _num(fallback, "quality_score")
    parts.append(meta.reset_index(drop=True))

    out = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if out.columns.duplicated().any():
        dup = out.columns[out.columns.duplicated()].tolist()
        raise RuntimeError(f"duplicate alpha8 risk selector state columns: {dup[:20]}")
    return out


def _zero_row(row: pd.Series) -> pd.Series:
    out = row.copy()
    for col, value in (
        ("action", 0),
        ("side", 0),
        ("notional_exposure", 0.0),
        ("position_fraction", 0.0),
        ("take_profit", 0.0),
        ("stop_loss", 0.0),
        ("max_hold_bars", 0),
        ("cooldown_bars", 0),
    ):
        out.loc[col] = value
    out.loc["leverage"] = 1.0
    return out


def _apply_template(row: pd.Series, template: RiskTemplate) -> pd.Series:
    if template.veto:
        return _zero_row(row)
    out = row.copy()
    base_notional = float(row.get("notional_exposure", 0.0) or 0.0)
    notional = float(min(max(base_notional * float(template.mult), 0.0), float(template.cap)))
    leverage = float(max(template.leverage, 1e-8))
    base_tp = float(row.get("take_profit", 0.0) or 0.0)
    base_sl = abs(float(row.get("stop_loss", 0.0) or 0.0))
    base_hold = max(int(row.get("max_hold_bars", 0) or 0), 1)
    out.loc["notional_exposure"] = notional
    out.loc["leverage"] = leverage
    out.loc["position_fraction"] = float(notional / leverage)
    out.loc["take_profit"] = float(max(base_tp, 1e-8) * float(template.tp_mult))
    out.loc["stop_loss"] = float(max(base_sl, 1e-8) * float(template.sl_mult))
    out.loc["max_hold_bars"] = int(max(1, round(base_hold * float(template.hold_mult))))
    return out


def _simulate_template(
    frame: pd.DataFrame,
    i: int,
    dec_row: pd.Series,
    template: RiskTemplate,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
) -> tuple[float, dict[str, Any]]:
    dec = _apply_template(dec_row, template)
    action = int(dec.get("action", 0) or 0)
    side = int(dec.get("side", 0) or 0)
    notional = float(dec.get("notional_exposure", 0.0) or 0.0)
    if action == ACTION_CASH or side == 0 or notional <= 0.0:
        return 0.0, {"active": 0, "net": 0.0, "win": 0, "mae": 0.0}

    open_px = _num(frame, "open")
    high = _num(frame, "high")
    low = _num(frame, "low")
    close = _num(frame, "close")
    entry_i = min(int(i) + 1, len(frame) - 1)
    entry = float(open_px[entry_i])
    if entry <= 0.0:
        return 0.0, {"active": 0, "net": 0.0, "win": 0, "mae": 0.0}

    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    entry = entry * (1.0 + slip_eff if side > 0 else 1.0 - slip_eff)
    tp = float(dec.get("take_profit", 0.0) or 0.0)
    sl = abs(float(dec.get("stop_loss", 0.0) or 0.0))
    hold = max(int(dec.get("max_hold_bars", 0) or 0), 1)
    end_i = min(entry_i + hold, len(frame) - 1)

    realized: float | None = None
    mae = 0.0
    mfe = 0.0
    exit_i = end_i
    for j in range(entry_i + 1, end_i + 1):
        if side > 0:
            favorable = (float(high[j]) / max(entry, 1e-12) - 1.0) * notional
            adverse = (float(low[j]) / max(entry, 1e-12) - 1.0) * notional
        else:
            favorable = (entry / max(float(low[j]), 1e-12) - 1.0) * notional
            adverse = (entry / max(float(high[j]), 1e-12) - 1.0) * notional
        mfe = max(mfe, favorable)
        mae = min(mae, adverse)
        if adverse <= -sl:
            realized = -sl
            exit_i = j
            break
        if favorable >= tp:
            realized = tp
            exit_i = j
            break
    if realized is None:
        exit_px = float(close[end_i])
        exit_px = exit_px * (1.0 - slip_eff if side > 0 else 1.0 + slip_eff)
        raw = (exit_px - entry) / max(entry, 1e-12) if side > 0 else (entry - exit_px) / max(entry, 1e-12)
        realized = float(raw) * notional
        exit_i = end_i

    net = float(realized) - 2.0 * fee_eff * notional
    hold_frac = float(max(exit_i - entry_i, 1) / max(hold, 1))
    win = int(net > 0.0)
    reward = 110.0 * net + (0.25 if win else -0.18)
    reward -= 10.0 * max(0.0, -mae)
    reward -= 0.015 * hold_frac
    return float(reward), {"active": 1, "net": net, "win": win, "mae": mae, "mfe": mfe}


def _build_rewards(
    frame: pd.DataFrame,
    combo: pd.DataFrame,
    states: np.ndarray,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
) -> tuple[DatasetBundle, np.ndarray, dict[str, Any]]:
    active = _active(combo)
    idxs = np.flatnonzero(active & (np.arange(len(frame)) < len(frame) - 3))
    rewards_matrix = np.zeros((len(frame), len(TEMPLATES)), dtype=np.float32)
    s_list: list[np.ndarray] = []
    sp_list: list[np.ndarray] = []
    a_list: list[int] = []
    r_list: list[float] = []
    d_list: list[float] = []
    net_stats: dict[str, list[float]] = {t.name: [] for t in TEMPLATES}
    win_stats: dict[str, list[int]] = {t.name: [] for t in TEMPLATES}

    for i in idxs:
        next_i = min(int(i) + 1, len(states) - 1)
        for action_id, template in enumerate(TEMPLATES):
            reward, meta = _simulate_template(frame, int(i), combo.iloc[int(i)], template, fee=fee, slip=slip, cost_mult=cost_mult)
            rewards_matrix[int(i), int(action_id)] = float(reward)
            s_list.append(states[int(i)])
            sp_list.append(states[next_i])
            a_list.append(int(action_id))
            r_list.append(float(reward))
            d_list.append(0.0)
            if int(meta["active"]) == 1:
                net_stats[template.name].append(float(meta["net"]))
                win_stats[template.name].append(int(meta["win"]))

    rewards_np = np.asarray(r_list, dtype=np.float32)
    scale = float(np.nanstd(rewards_np))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    rewards_scaled = np.clip(rewards_np / scale, -8.0, 8.0).astype(np.float32)
    dataset = DatasetBundle(
        states=np.asarray(s_list, dtype=np.float32),
        next_states=np.asarray(sp_list, dtype=np.float32),
        actions=np.asarray(a_list, dtype=np.int64),
        rewards=rewards_scaled,
        dones=np.asarray(d_list, dtype=np.float32),
    )
    diagnostics = {
        "active_rows": int(len(idxs)),
        "reward_scale": float(scale),
        "template_net_mean": {k: float(np.mean(v)) if v else 0.0 for k, v in net_stats.items()},
        "template_win_rate": {k: float(np.mean(v)) if v else 0.0 for k, v in win_stats.items()},
        "template_active_count": {k: int(len(v)) for k, v in net_stats.items()},
        "oracle_best_template_counts": {
            TEMPLATES[int(i)].name: int(v)
            for i, v in enumerate(np.bincount(np.argmax(rewards_matrix[idxs], axis=1), minlength=len(TEMPLATES)))
        },
    }
    return dataset, rewards_matrix, diagnostics


def _sample_tau(batch: int, n_tau: int, device: torch.device, dtype: torch.dtype, *, tail_mix: float, tail_max: float) -> torch.Tensor:
    n_tail = int(round(int(n_tau) * float(np.clip(tail_mix, 0.0, 0.95))))
    n_base = max(1, int(n_tau) - n_tail)
    base = torch.rand((batch, n_base), device=device, dtype=dtype)
    if n_tail <= 0:
        tau = base
    else:
        tail = torch.rand((batch, n_tail), device=device, dtype=dtype) * float(np.clip(tail_max, 0.01, 1.0))
        tau = torch.cat([tail, base], dim=1)
    return tau.clamp(0.001, 0.999)


def _quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor, tau: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    td = target.unsqueeze(1) - pred
    abs_td = td.abs()
    huber = torch.where(abs_td <= 1.0, 0.5 * td.pow(2), abs_td - 0.5)
    weight = (tau.unsqueeze(-1) - (td.detach() < 0.0).float()).abs()
    loss = (weight * huber).mean(dim=(1, 2))
    return (loss * sample_weight).mean()


def _train_iqn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    tau_samples: int,
    tail_tau_mix: float,
    tail_tau_max: float,
    seed: int,
) -> tuple[IQNRiskSelector, dict[str, Any]]:
    _seed_everything(seed)
    model = IQNRiskSelector(state_dim=x_train.shape[1], action_dim=y_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    best_action = np.argmax(y_train, axis=1)
    counts = np.bincount(best_action, minlength=y_train.shape[1]).astype(np.float64)
    sample_weight = np.sqrt(max(float(len(y_train)), 1.0) / np.maximum(counts[best_action], 1.0)).astype(np.float32)
    sample_weight = np.clip(sample_weight, 0.25, np.percentile(sample_weight, 99)).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32)), torch.from_numpy(sample_weight))
    sampler = WeightedRandomSampler(torch.from_numpy(sample_weight.astype(np.float64)), num_samples=len(sample_weight), replacement=True)
    dl = DataLoader(ds, batch_size=int(batch_size), sampler=sampler, drop_last=False)
    losses: list[float] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb, sw in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            sw = sw.to(device)
            tau = _sample_tau(len(xb), int(tau_samples), device, xb.dtype, tail_mix=tail_tau_mix, tail_max=tail_tau_max)
            pred = model(xb, tau)
            loss = _quantile_huber_loss(pred, yb, tau, sw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            n += len(xb)
        losses.append(float(total / max(n, 1)))
        if epoch % 5 == 0 or epoch == int(epochs):
            print(json.dumps({"stage": "iqn_progress", "epoch": epoch, "loss": losses[-1]}, ensure_ascii=False), flush=True)
    return model.cpu(), {
        "epochs": int(epochs),
        "losses": losses,
        "best_action_counts": {TEMPLATES[int(i)].name: int(v) for i, v in enumerate(counts.astype(int))},
        "sample_weight_mean": float(np.mean(sample_weight)),
    }


def _iqn_action(model: IQNRiskSelector, states: np.ndarray, *, device: torch.device, risk_tau: float, num_tau: int) -> np.ndarray:
    model = model.to(device)
    model.eval()
    taus = torch.linspace(0.01, float(risk_tau), int(num_tau), device=device).view(1, -1)
    actions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states), 8192):
            x = torch.from_numpy(states[start : start + 8192]).to(device)
            tau = taus.repeat(len(x), 1)
            q = model(x, tau).mean(dim=1)
            actions.append(torch.argmax(q, dim=-1).cpu().numpy().astype(np.int64))
    return np.concatenate(actions) if actions else np.zeros(0, dtype=np.int64)


def _compose_decisions(combo: pd.DataFrame, actions: np.ndarray) -> pd.DataFrame:
    out = combo.copy().reset_index(drop=True)
    active = _active(out)
    for i in np.flatnonzero(active):
        action_id = int(actions[int(i)])
        if action_id < 0 or action_id >= len(TEMPLATES):
            raise RuntimeError(f"invalid risk template id: {action_id}")
        out.iloc[int(i)] = _apply_template(out.iloc[int(i)], TEMPLATES[action_id])
    out.loc[~active] = out.loc[~active].apply(_zero_row, axis=1)
    return out


def _usage(actions: np.ndarray, active: np.ndarray) -> dict[str, int]:
    return {t.name: int(np.sum(actions[np.asarray(active, dtype=bool)] == i)) for i, t in enumerate(TEMPLATES)}


def _metrics_rows(evaluator: OfficialCost3, splits: list[tuple[str, pd.DataFrame, dict[str, pd.DataFrame]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, frame, variants in splits:
        for variant, dec in variants.items():
            m = evaluator(frame, dec)
            rows.append({"split": split, "variant": variant, **m})
    return pd.DataFrame(rows)


def _score(row: pd.Series) -> float:
    trades = int(row.get("trades", 0) or 0)
    if trades < 30:
        return -1e9 + float(row.get("pnl", 0.0) or 0.0)
    return (
        float(row.get("pnl", 0.0) or 0.0)
        + 130.0 * float(row.get("wr", 0.0) or 0.0)
        - 0.45 * abs(float(row.get("mdd", 0.0) or 0.0))
        + 0.015 * trades
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    ap.add_argument("--eval-csv", type=Path, default=EVAL_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--primary-parent", type=Path, default=None)
    ap.add_argument("--primary-summary", type=Path, default=None)
    ap.add_argument("--fallback-parent", type=Path, default=None)
    ap.add_argument("--fallback-summary", type=Path, default=None)
    ap.add_argument("--dsac-steps", type=int, default=12000)
    ap.add_argument("--dsac-batch-size", type=int, default=1024)
    ap.add_argument("--iqn-epochs", type=int, default=28)
    ap.add_argument("--iqn-batch-size", type=int, default=1024)
    ap.add_argument("--iqn-lr", type=float, default=7e-4)
    ap.add_argument("--iqn-tau-samples", type=int, default=32)
    ap.add_argument("--risk-taus", default="0.15,0.25,0.35,0.50")
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    _seed_everything(80529)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = Path(args.train_csv)
    eval_csv = Path(args.eval_csv)
    if not train_csv.exists():
        raise FileNotFoundError(f"Alpha8 train CSV missing: {train_csv}")
    if not eval_csv.exists():
        raise FileNotFoundError(f"Alpha8 eval CSV missing: {eval_csv}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")

    baseline = get_live_baseline()
    train_all = _rename_clean4_v2(_read(train_csv))
    eval_df = _rename_clean4_v2(_read(eval_csv))
    _assert_clean(train_all, name="train_all")
    _assert_clean(eval_df, name="eval")
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    primary_parent_path = Path(args.primary_parent) if args.primary_parent else baseline.primary_parent
    fallback_parent_path = Path(args.fallback_parent) if args.fallback_parent else baseline.fallback_parent
    primary_summary_path = Path(args.primary_summary) if args.primary_summary else baseline.primary_summary
    fallback_summary_path = Path(args.fallback_summary) if args.fallback_summary else baseline.fallback_summary
    primary = joblib.load(primary_parent_path)
    fallback = joblib.load(fallback_parent_path)
    primary_rt = _load_scale_runtime_any(primary_summary_path)
    fallback_rt = _load_scale_runtime_any(fallback_summary_path)
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
    dsac_data, reward_matrix, data_diag = _build_rewards(
        train_df,
        combo_train,
        x_train,
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
    )
    active_train = _active(combo_train)
    active_val = _active(combo_val)
    active_eval = _active(combo_eval)

    print(
        json.dumps(
            {
                "stage": "train_start",
                "model_id": str(args.model_id),
                "device": str(device),
                "train_csv": str(train_csv),
                "eval_csv": str(eval_csv),
                "state_dim": int(x_train.shape[1]),
                "templates": len(TEMPLATES),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "oos_rows": len(eval_df),
                "data_diag": data_diag,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    dsac = _train_dsac_offline(
        dsac_data,
        state_dim=int(x_train.shape[1]),
        action_dim=len(TEMPLATES),
        device=device,
        steps=int(args.dsac_steps),
        batch_size=int(args.dsac_batch_size),
    )
    dsac_actor: nn.Module = dsac["actor"]
    dsac_train_actions = _policy_action(dsac_actor, x_train, device=device)
    dsac_val_actions = _policy_action(dsac_actor, x_val, device=device)
    dsac_eval_actions = _policy_action(dsac_actor, x_eval, device=device)

    y_iqn = reward_matrix[active_train].astype(np.float32)
    x_iqn = x_train[active_train].astype(np.float32)
    iqn, iqn_diag = _train_iqn(
        x_iqn,
        y_iqn,
        device=device,
        epochs=int(args.iqn_epochs),
        batch_size=int(args.iqn_batch_size),
        lr=float(args.iqn_lr),
        tau_samples=int(args.iqn_tau_samples),
        tail_tau_mix=0.45,
        tail_tau_max=0.25,
        seed=90529,
    )

    variants: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "dsac": (dsac_train_actions, dsac_val_actions, dsac_eval_actions),
    }
    for risk_tau in [float(x) for x in str(args.risk_taus).split(",") if x.strip()]:
        variants[f"iqn_cvar{risk_tau:.2f}"] = (
            _iqn_action(iqn, x_train, device=device, risk_tau=risk_tau, num_tau=32),
            _iqn_action(iqn, x_val, device=device, risk_tau=risk_tau, num_tau=32),
            _iqn_action(iqn, x_eval, device=device, risk_tau=risk_tau, num_tau=32),
        )

    decision_variants_train = {"baseline_combo": combo_train}
    decision_variants_val = {"baseline_combo": combo_val}
    decision_variants_eval = {"baseline_combo": combo_eval}
    action_usage: dict[str, Any] = {}
    for name, (a_train, a_val, a_eval) in variants.items():
        decision_variants_train[name] = _compose_decisions(combo_train, a_train)
        decision_variants_val[name] = _compose_decisions(combo_val, a_val)
        decision_variants_eval[name] = _compose_decisions(combo_eval, a_eval)
        action_usage[name] = {
            "train": _usage(a_train, active_train),
            "val": _usage(a_val, active_val),
            "oos": _usage(a_eval, active_eval),
        }

    grid = _metrics_rows(
        evaluator,
        [
            ("train", train_df, decision_variants_train),
            ("val", val_df, decision_variants_val),
            ("oos", eval_df, decision_variants_eval),
        ],
    )
    grid["selection_score"] = grid.apply(_score, axis=1)
    grid_path = out_dir / "grid.csv"
    grid.to_csv(grid_path, index=False)

    val_rank = grid[(grid["split"] == "val") & (grid["variant"] != "baseline_combo")].sort_values("selection_score", ascending=False)
    selected_variant = str(val_rank.iloc[0]["variant"]) if len(val_rank) else "dsac"
    selected_oos = grid[(grid["split"] == "oos") & (grid["variant"] == selected_variant)].iloc[0].to_dict()
    baseline_oos = grid[(grid["split"] == "oos") & (grid["variant"] == "baseline_combo")].iloc[0].to_dict()

    model_path = out_dir / "alpha8_dsac_iqn_risk_selector.pt"
    torch.save(
        {
            "model_id": str(args.model_id),
            "selected_variant": selected_variant,
            "state_dim": int(x_train.shape[1]),
            "action_dim": len(TEMPLATES),
            "state_columns": list(norm["columns"]),
            "state_normalizer": norm,
            "templates": [asdict(t) for t in TEMPLATES],
            "dsac_actor_state_dict": dsac_actor.state_dict(),
            "iqn_state_dict": iqn.state_dict(),
            "iqn_network": {"hidden_dim": 256, "n_cos": 64},
        },
        model_path,
    )
    (out_dir / "state_columns.json").write_text(json.dumps(list(norm["columns"]), indent=2) + "\n")
    (out_dir / "templates.json").write_text(json.dumps([asdict(t) for t in TEMPLATES], indent=2) + "\n")

    summary = {
        "model_id": str(args.model_id),
        "design": "Alpha7 primary+fallback combo owns direction. DSAC and IQN choose only constrained risk templates: veto, notional multiplier/cap, leverage, TP/SL/hold multipliers.",
        "live_wired": False,
        "selection_basis": "2025Q4 validation official Cost3 score; 2026 OOS is reported only.",
        "baseline_model_id": baseline.model_id,
        "train_csv": str(train_csv),
        "eval_csv": str(eval_csv),
        "parent_artifacts": {
            "primary_parent": str(primary_parent_path),
            "primary_summary": str(primary_summary_path),
            "fallback_parent": str(fallback_parent_path),
            "fallback_summary": str(fallback_summary_path),
        },
        "allowed_regime_surfaces": ["clean_regime4_state24_sticky090_v2_*", "regime4_pred_*"],
        "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
        "forbidden_prefix_count": 0,
        "teacher_features": TEACHER_COLS,
        "templates": [asdict(t) for t in TEMPLATES],
        "training": {
            "device": str(device),
            "state_dim": int(x_train.shape[1]),
            "dsac_steps": int(args.dsac_steps),
            "iqn_epochs": int(args.iqn_epochs),
            "dataset_diagnostics": data_diag,
            "dsac_diag": dsac["train_diag"],
            "iqn_diag": iqn_diag,
            "action_usage": action_usage,
        },
        "selected": {
            "variant": selected_variant,
            "val": grid[(grid["split"] == "val") & (grid["variant"] == selected_variant)].iloc[0].to_dict(),
            "oos": selected_oos,
            "delta_vs_baseline_oos_pnl": float(selected_oos["pnl"]) - float(baseline_oos["pnl"]),
        },
        "baseline_oos": baseline_oos,
        "artifacts": {
            "summary": str(out_dir / "summary.json"),
            "grid": str(grid_path),
            "model": str(model_path),
            "state_columns": str(out_dir / "state_columns.json"),
            "templates": str(out_dir / "templates.json"),
        },
        "audit": {
            "feature_contract_fail_fast": True,
            "legacy_compat_alias": False,
            "selection_uses_2026": False,
            "official_accounting": "OfficialCost3",
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    print(json.dumps({"summary": str(summary_path), "selected": summary["selected"], "baseline_oos": baseline_oos}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
