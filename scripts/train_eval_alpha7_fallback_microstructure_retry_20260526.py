#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
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

from ensemble.fully_learned_governor_policy import ACTION_CASH  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import train_eval_alpha7_meta_fallback_cash_router_20260526 as base  # noqa: E402
from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import (  # noqa: E402
    _compact_costs,
    _metrics,
    _score,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _grid as _runner_grid  # noqa: E402


BASELINE = get_live_baseline()
TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_fallback_microstructure_retry_20260526"
OLD_CLEAN_PREFIX = "clean_regime_2024_unsup_v4_"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _require_cols(df: pd.DataFrame, cols: list[str], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name}: missing required columns: {missing}")


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return (action != ACTION_CASH) & (side != 0)


def _empty_dec_like(template: pd.DataFrame) -> pd.DataFrame:
    dec = template.copy().reset_index(drop=True)
    dec["action"] = 0
    dec["side"] = 0
    dec["notional_exposure"] = 0.0
    dec["leverage"] = 1.0
    dec["position_fraction"] = 0.0
    dec["take_profit"] = 0.0
    dec["stop_loss"] = 0.0
    dec["max_hold_bars"] = 0
    dec["cooldown_bars"] = 0
    dec["quality_score"] = 0.0
    dec["confidence"] = 0.0
    return dec


def _pick_pool_best(primary_dec: pd.DataFrame, candidate_decs: list[pd.DataFrame]) -> pd.DataFrame:
    out = _empty_dec_like(primary_dec)
    cash = ~_active(primary_dec)
    n = len(out)
    if not candidate_decs:
        return out
    q_mat = np.column_stack(
        [pd.to_numeric(d["quality_score"], errors="coerce").fillna(-1e9).to_numpy(dtype=np.float64) for d in candidate_decs]
    )
    for i in range(n):
        if not cash[i]:
            continue
        best_j = -1
        best_q = -1e18
        for j, d in enumerate(candidate_decs):
            row = d.iloc[i]
            if int(pd.to_numeric(row["action"], errors="coerce")) == 0:
                continue
            if int(pd.to_numeric(row["side"], errors="coerce")) == 0:
                continue
            q = float(q_mat[i, j])
            if q > best_q:
                best_q = q
                best_j = j
        if best_j < 0:
            continue
        chosen = candidate_decs[best_j]
        for col in out.columns:
            out.iat[i, out.columns.get_loc(col)] = chosen.iat[i, chosen.columns.get_loc(col)]
    return out


def _rolling_hurst(close: pd.Series, window: int = 48) -> pd.Series:
    ret = pd.to_numeric(close, errors="coerce").ffill().pct_change().fillna(0.0)

    def _rs_h(x: np.ndarray) -> float:
        y = np.asarray(x, dtype=np.float64)
        if len(y) < 8:
            return 0.5
        y = y - y.mean()
        z = np.cumsum(y)
        r = float(np.max(z) - np.min(z))
        s = float(np.std(y))
        if not np.isfinite(r) or not np.isfinite(s) or s <= 1e-12 or r <= 1e-12:
            return 0.5
        h = float(np.log(r / s) / np.log(max(len(y), 2)))
        return float(np.clip(h, 0.0, 1.0))

    return ret.rolling(window, min_periods=max(12, window // 2)).apply(_rs_h, raw=True).fillna(0.5)


def _hawkes_proxy(frame: pd.DataFrame, beta: float) -> tuple[np.ndarray, np.ndarray]:
    volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    tb = pd.to_numeric(frame["taker_buy_base"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ti = pd.to_numeric(frame["trade_intensity"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    buy_ratio = np.clip(tb / np.maximum(volume, 1e-9), 0.0, 1.0)
    sell_ratio = 1.0 - buy_ratio
    evt_b = ti * buy_ratio
    evt_s = ti * sell_ratio

    lam_b = np.zeros_like(evt_b)
    lam_s = np.zeros_like(evt_s)
    for i in range(1, len(evt_b)):
        lam_b[i] = float((1.0 - beta) * evt_b[i] + beta * lam_b[i - 1])
        lam_s[i] = float((1.0 - beta) * evt_s[i] + beta * lam_s[i - 1])
    d_b = np.maximum(lam_b[:-1] - lam_b[1:], 0.0)
    d_s = np.maximum(lam_s[:-1] - lam_s[1:], 0.0)
    ex_long = np.zeros_like(lam_b)
    ex_short = np.zeros_like(lam_s)
    ex_long[1:] = d_s * np.maximum(lam_s[:-1] - lam_b[:-1], 0.0)
    ex_short[1:] = d_b * np.maximum(lam_b[:-1] - lam_s[:-1], 0.0)
    return ex_long, ex_short


class _SeqAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.enc = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dec = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.enc(x)
        z = h[-1]
        rep = z.unsqueeze(1).repeat(1, x.shape[1], 1)
        y, _ = self.dec(rep)
        return self.head(y)


def _build_seq_matrix(frame: pd.DataFrame, cols: list[str], seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    x = frame[cols].copy()
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mu = x.mean(axis=0)
    sd = x.std(axis=0).replace(0.0, 1.0)
    z = ((x - mu) / sd).to_numpy(dtype=np.float32)
    n = len(z)
    seq = np.zeros((n, seq_len, z.shape[1]), dtype=np.float32)
    for i in range(n):
        s = max(0, i - seq_len + 1)
        cur = z[s : i + 1]
        seq[i, -len(cur) :, :] = cur
    return seq, z


def _fit_seq_toxic_score(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    *,
    cols: list[str],
    seq_len: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_seq, _ = _build_seq_matrix(train_frame, cols, seq_len)
    val_seq, _ = _build_seq_matrix(val_frame, cols, seq_len)
    eval_seq, _ = _build_seq_matrix(eval_frame, cols, seq_len)

    model = _SeqAutoEncoder(input_dim=len(cols), hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    ds = TensorDataset(torch.from_numpy(train_seq))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    model.train()
    for _ in range(epochs):
        for (xb,) in dl:
            xb = xb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    def _score(seq: np.ndarray) -> np.ndarray:
        model.eval()
        out = np.zeros((len(seq),), dtype=np.float64)
        with torch.no_grad():
            for s in range(0, len(seq), batch_size):
                e = min(s + batch_size, len(seq))
                xb = torch.from_numpy(seq[s:e]).to(device)
                pred = model(xb)
                err = torch.mean((pred - xb) ** 2, dim=(1, 2))
                out[s:e] = err.detach().cpu().numpy().astype(np.float64)
        return out

    return _score(train_seq), _score(val_seq), _score(eval_seq)


def _apply_hurst_gate(
    frame: pd.DataFrame,
    primary_dec: pd.DataFrame,
    pool_dec: pd.DataFrame,
    *,
    hurst_th: float,
    chop_whipsaw_th: float,
    meanrev_abs_min: float,
) -> pd.DataFrame:
    out = _empty_dec_like(pool_dec)
    out.loc[:, :] = pool_dec.to_numpy()
    hurst = _rolling_hurst(frame["close"], window=48).to_numpy(dtype=np.float64)
    mr = pd.to_numeric(frame["mean_reversion_z"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    chop = pd.to_numeric(frame["clean_regime4_2024_unsup_v1_chop_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    whip = pd.to_numeric(frame["clean_regime4_2024_unsup_v1_whipsaw_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    side = pd.to_numeric(out["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    cash = ~_active(primary_dec)
    active = _active(out)
    mask = (
        cash
        & active
        & (hurst < float(hurst_th))
        & ((chop + whip) >= float(chop_whipsaw_th))
        & (np.abs(mr) >= float(meanrev_abs_min))
        & ((side.astype(np.float64) * np.sign(mr)) <= 0.0)
    )
    kill = active & (~mask)
    out.loc[kill, ["action", "side", "notional_exposure", "quality_score", "confidence"]] = [0, 0, 0.0, 0.0, 0.0]
    out.loc[kill, ["take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = [0.0, 0.0, 0, 0]
    keep = mask
    if np.any(keep):
        out.loc[keep, "take_profit"] = np.minimum(pd.to_numeric(out.loc[keep, "take_profit"], errors="coerce").fillna(0.0), 0.0060)
        out.loc[keep, "stop_loss"] = np.minimum(pd.to_numeric(out.loc[keep, "stop_loss"], errors="coerce").fillna(0.0), 0.0042)
        out.loc[keep, "max_hold_bars"] = np.minimum(pd.to_numeric(out.loc[keep, "max_hold_bars"], errors="coerce").fillna(8).astype(int), 8)
    return out


def _apply_hawkes_gate(
    frame: pd.DataFrame,
    primary_dec: pd.DataFrame,
    pool_dec: pd.DataFrame,
    *,
    beta: float,
    ex_th: float,
) -> pd.DataFrame:
    out = _empty_dec_like(pool_dec)
    out.loc[:, :] = pool_dec.to_numpy()
    ex_long, ex_short = _hawkes_proxy(frame, beta=float(beta))
    side = pd.to_numeric(out["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    cash = ~_active(primary_dec)
    active = _active(out)
    cond = np.where(side > 0, ex_long >= float(ex_th), np.where(side < 0, ex_short >= float(ex_th), False))
    mask = cash & active & cond
    kill = active & (~mask)
    out.loc[kill, ["action", "side", "notional_exposure", "quality_score", "confidence"]] = [0, 0, 0.0, 0.0, 0.0]
    out.loc[kill, ["take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = [0.0, 0.0, 0, 0]
    keep = mask
    if np.any(keep):
        out.loc[keep, "take_profit"] = np.minimum(pd.to_numeric(out.loc[keep, "take_profit"], errors="coerce").fillna(0.0), 0.0052)
        out.loc[keep, "stop_loss"] = np.minimum(pd.to_numeric(out.loc[keep, "stop_loss"], errors="coerce").fillna(0.0), 0.0044)
        out.loc[keep, "max_hold_bars"] = np.minimum(pd.to_numeric(out.loc[keep, "max_hold_bars"], errors="coerce").fillna(10).astype(int), 10)
    return out


def _apply_seq_toxic_gate(
    frame: pd.DataFrame,
    primary_dec: pd.DataFrame,
    pool_dec: pd.DataFrame,
    seq_err: np.ndarray,
    *,
    err_threshold: float,
    flip_prob_min: float,
) -> pd.DataFrame:
    out = _empty_dec_like(pool_dec)
    out.loc[:, :] = pool_dec.to_numpy()
    flip = pd.to_numeric(frame["ai_flow_flip_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    cash = ~_active(primary_dec)
    active = _active(out)
    mask = cash & active & (seq_err <= float(err_threshold)) & (flip >= float(flip_prob_min))
    kill = active & (~mask)
    out.loc[kill, ["action", "side", "notional_exposure", "quality_score", "confidence"]] = [0, 0, 0.0, 0.0, 0.0]
    out.loc[kill, ["take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = [0.0, 0.0, 0, 0]
    keep = mask
    if np.any(keep):
        out.loc[keep, "take_profit"] = np.minimum(pd.to_numeric(out.loc[keep, "take_profit"], errors="coerce").fillna(0.0), 0.0058)
        out.loc[keep, "stop_loss"] = np.minimum(pd.to_numeric(out.loc[keep, "stop_loss"], errors="coerce").fillna(0.0), 0.0046)
        out.loc[keep, "max_hold_bars"] = np.minimum(pd.to_numeric(out.loc[keep, "max_hold_bars"], errors="coerce").fillna(9).astype(int), 9)
    return out


def _apply_directionless_maker_proxy(
    frame: pd.DataFrame,
    primary_dec: pd.DataFrame,
    *,
    beta: float,
    hurst_th: float,
    sig_th: float,
    alpha: float,
    prob_min: float,
    notional: float,
    tp: float,
    sl: float,
    hold: int,
) -> pd.DataFrame:
    out = _empty_dec_like(primary_dec)
    cash = ~_active(primary_dec)
    hurst = _rolling_hurst(frame["close"], window=48).to_numpy(dtype=np.float64)
    ex_long, ex_short = _hawkes_proxy(frame, beta=float(beta))
    vac = pd.to_numeric(frame["liquidity_vacuum"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    flow = pd.to_numeric(frame["smart_money_flow"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    score_long = ex_long - 0.35 * np.maximum(vac, 0.0) + 0.15 * np.maximum(-flow, 0.0)
    score_short = ex_short - 0.35 * np.maximum(vac, 0.0) + 0.15 * np.maximum(flow, 0.0)
    for i in range(len(out)):
        if not cash[i]:
            continue
        if hurst[i] >= float(hurst_th):
            continue
        s = np.array([score_long[i], score_short[i]], dtype=np.float64)
        s = s - np.max(s)
        p = np.exp(s / max(float(alpha), 1e-6))
        p = p / np.maximum(np.sum(p), 1e-12)
        cls = int(np.argmax(p))
        conf = float(np.max(p))
        margin = float(abs(score_long[i] - score_short[i]))
        if conf < float(prob_min) or margin < float(sig_th):
            continue
        side = 1 if cls == 0 else -1
        out.at[i, "action"] = 1
        out.at[i, "side"] = side
        out.at[i, "notional_exposure"] = float(notional)
        out.at[i, "leverage"] = 1.0
        out.at[i, "position_fraction"] = float(min(notional / 5.0, 1.0))
        out.at[i, "take_profit"] = float(tp)
        out.at[i, "stop_loss"] = float(sl)
        out.at[i, "max_hold_bars"] = int(hold)
        out.at[i, "cooldown_bars"] = 0
        out.at[i, "quality_score"] = float(margin)
        out.at[i, "confidence"] = float(conf)
    return out


def _evaluate_combo(
    frame: pd.DataFrame,
    primary_dec: pd.DataFrame,
    fallback_dec: pd.DataFrame,
    *,
    ref_parent: dict[str, Any],
    noop_runner: dict[str, Any],
    noop_cfg: Any,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    combo = base._combine_primary_fallback(primary_dec, fallback_dec)
    return _compact_costs(
        _metrics(
            frame,
            parent_for_features=ref_parent,
            runner=noop_runner,
            runner_cfg=noop_cfg,
            dec=combo,
            fee=fee,
            slip=slip,
        )
    )


@dataclass(frozen=True)
class MethodResult:
    method: str
    params: dict[str, Any]
    selection_score: float
    val_metrics: dict[str, Any]
    oos_metrics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Retry 4 microstructure fallback methods on Alpha7 cash region.")
    ap.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    ap.add_argument("--eval-csv", type=Path, default=EVAL_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=52626)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    _require_cols(
        train_all,
        [
            "timestamp",
            "close",
            "volume",
            "taker_buy_base",
            "trade_intensity",
            "mean_reversion_z",
            "liquidity_vacuum",
            "smart_money_flow",
            "clean_regime4_2024_unsup_v1_chop_prob",
            "clean_regime4_2024_unsup_v1_whipsaw_prob",
            "ai_flow_flip_prob",
            "ai_flow_exhaustion",
            "ai_flow_pressure",
            "ofi_acceleration",
            "net_taker_ratio",
        ],
        name="train_csv",
    )
    _require_cols(
        eval_df,
        [
            "timestamp",
            "close",
            "volume",
            "taker_buy_base",
            "trade_intensity",
            "mean_reversion_z",
            "liquidity_vacuum",
            "smart_money_flow",
            "clean_regime4_2024_unsup_v1_chop_prob",
            "clean_regime4_2024_unsup_v1_whipsaw_prob",
            "ai_flow_flip_prob",
            "ai_flow_exhaustion",
            "ai_flow_pressure",
            "ofi_acceleration",
            "net_taker_ratio",
        ],
        name="eval_csv",
    )

    cutoff = pd.Timestamp("2025-10-01")
    train_df = train_all[train_all["timestamp"] < cutoff].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= cutoff].reset_index(drop=True)

    primary_parent = joblib.load(BASELINE.primary_parent)
    primary_rt = base._load_best_scale_runtime(BASELINE.primary_summary)
    primary_train = base._predict_scaled(primary_parent, train_df, primary_rt)
    primary_val = base._predict_scaled(primary_parent, val_df, primary_rt)
    primary_eval = base._predict_scaled(primary_parent, eval_df, primary_rt)

    candidate_specs: list[base.CandidateSpec] = []
    train_candidate_decs: list[pd.DataFrame] = []
    val_candidate_decs: list[pd.DataFrame] = []
    eval_candidate_decs: list[pd.DataFrame] = []
    for spec in base._candidate_specs():
        if not spec.parent.exists():
            continue
        parent = joblib.load(spec.parent)
        feature_cols = list(parent.get("feature_cols", []))
        if any(str(c).startswith(OLD_CLEAN_PREFIX) for c in feature_cols):
            continue
        rt = base._load_best_scale_runtime(spec.summary)
        candidate_specs.append(spec)
        train_candidate_decs.append(base._predict_scaled(parent, train_df, rt))
        val_candidate_decs.append(base._predict_scaled(parent, val_df, rt))
        eval_candidate_decs.append(base._predict_scaled(parent, eval_df, rt))
    if not candidate_specs:
        raise RuntimeError("no valid fallback candidate for retry")

    pool_train = _pick_pool_best(primary_train, train_candidate_decs)
    pool_val = _pick_pool_best(primary_val, val_candidate_decs)
    pool_eval = _pick_pool_best(primary_eval, eval_candidate_decs)

    ref_parent = base._parent_for_features(list(joblib.load(v31.DEFAULT_PARENT)["feature_cols"]))
    fee = float(joblib.load(v31.DEFAULT_PARENT)["config"]["fee"])
    slip = float(joblib.load(v31.DEFAULT_PARENT)["config"]["slip"])
    noop_runner = joblib.load(v31.DEFAULT_JACKPOT)["cost_runner"]
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")

    current_fb_eval = eval_candidate_decs[0]
    current_fb_val = val_candidate_decs[0]
    baseline_val = _evaluate_combo(
        val_df,
        primary_val,
        current_fb_val,
        ref_parent=ref_parent,
        noop_runner=noop_runner,
        noop_cfg=noop_cfg,
        fee=fee,
        slip=slip,
    )
    baseline_eval = _evaluate_combo(
        eval_df,
        primary_eval,
        current_fb_eval,
        ref_parent=ref_parent,
        noop_runner=noop_runner,
        noop_cfg=noop_cfg,
        fee=fee,
        slip=slip,
    )

    results: list[MethodResult] = []
    ledgers: dict[str, dict[str, Any]] = {}

    # 1) Rough volatility + local Hurst mean-reversion gate
    best: MethodResult | None = None
    for h in (0.40, 0.42, 0.44):
        for cw in (0.50, 0.60, 0.70):
            for mr_abs in (0.35, 0.50, 0.65):
                val_dec = _apply_hurst_gate(val_df, primary_val, pool_val, hurst_th=h, chop_whipsaw_th=cw, meanrev_abs_min=mr_abs)
                eval_dec = _apply_hurst_gate(eval_df, primary_eval, pool_eval, hurst_th=h, chop_whipsaw_th=cw, meanrev_abs_min=mr_abs)
                val_m = _evaluate_combo(
                    val_df,
                    primary_val,
                    val_dec,
                    ref_parent=ref_parent,
                    noop_runner=noop_runner,
                    noop_cfg=noop_cfg,
                    fee=fee,
                    slip=slip,
                )
                sel = float(_score(val_m))
                if best is None or sel > best.selection_score:
                    eval_m = _evaluate_combo(
                        eval_df,
                        primary_eval,
                        eval_dec,
                        ref_parent=ref_parent,
                        noop_runner=noop_runner,
                        noop_cfg=noop_cfg,
                        fee=fee,
                        slip=slip,
                    )
                    best = MethodResult(
                        method="rough_hurst_mr",
                        params={"hurst_th": h, "chop_whipsaw_th": cw, "meanrev_abs_min": mr_abs},
                        selection_score=sel,
                        val_metrics=val_m,
                        oos_metrics=eval_m,
                    )
    assert best is not None
    results.append(best)

    # 2) Hawkes exhaustion gate
    best = None
    for beta in (0.65, 0.78, 0.86):
        for ex_th in (0.003, 0.006, 0.010):
            val_dec = _apply_hawkes_gate(val_df, primary_val, pool_val, beta=beta, ex_th=ex_th)
            eval_dec = _apply_hawkes_gate(eval_df, primary_eval, pool_eval, beta=beta, ex_th=ex_th)
            val_m = _evaluate_combo(
                val_df,
                primary_val,
                val_dec,
                ref_parent=ref_parent,
                noop_runner=noop_runner,
                noop_cfg=noop_cfg,
                fee=fee,
                slip=slip,
            )
            sel = float(_score(val_m))
            if best is None or sel > best.selection_score:
                eval_m = _evaluate_combo(
                    eval_df,
                    primary_eval,
                    eval_dec,
                    ref_parent=ref_parent,
                    noop_runner=noop_runner,
                    noop_cfg=noop_cfg,
                    fee=fee,
                    slip=slip,
                )
                best = MethodResult(
                    method="hawkes_exhaustion",
                    params={"beta": beta, "ex_th": ex_th},
                    selection_score=sel,
                    val_metrics=val_m,
                    oos_metrics=eval_m,
                )
    assert best is not None
    results.append(best)

    # 3) xLSTM-style sequence toxic-flow embedding gate (light autoencoder retry)
    seq_cols = [
        "net_taker_ratio",
        "ofi_acceleration",
        "trade_intensity",
        "liquidity_vacuum",
        "smart_money_flow",
        "ai_flow_exhaustion",
        "ai_flow_pressure",
    ]
    _require_cols(train_df, seq_cols, name="seq_train")
    tr_err, val_err, eval_err = _fit_seq_toxic_score(
        train_df,
        val_df,
        eval_df,
        cols=seq_cols,
        seq_len=24,
        hidden_dim=24,
        epochs=7,
        lr=1e-3,
        batch_size=512,
        seed=int(args.seed),
    )
    cash_tr = ~_active(primary_train)
    err_q_vals = [float(np.quantile(tr_err[cash_tr], q)) for q in (0.50, 0.60, 0.70)]
    best = None
    for err_th in err_q_vals:
        for flip_min in (0.52, 0.58, 0.64):
            val_dec = _apply_seq_toxic_gate(val_df, primary_val, pool_val, val_err, err_threshold=err_th, flip_prob_min=flip_min)
            eval_dec = _apply_seq_toxic_gate(eval_df, primary_eval, pool_eval, eval_err, err_threshold=err_th, flip_prob_min=flip_min)
            val_m = _evaluate_combo(
                val_df,
                primary_val,
                val_dec,
                ref_parent=ref_parent,
                noop_runner=noop_runner,
                noop_cfg=noop_cfg,
                fee=fee,
                slip=slip,
            )
            sel = float(_score(val_m))
            if best is None or sel > best.selection_score:
                eval_m = _evaluate_combo(
                    eval_df,
                    primary_eval,
                    eval_dec,
                    ref_parent=ref_parent,
                    noop_runner=noop_runner,
                    noop_cfg=noop_cfg,
                    fee=fee,
                    slip=slip,
                )
                best = MethodResult(
                    method="xlstm_proxy_toxic_gate",
                    params={"err_threshold": float(err_th), "flip_prob_min": float(flip_min)},
                    selection_score=sel,
                    val_metrics=val_m,
                    oos_metrics=eval_m,
                )
    assert best is not None
    results.append(best)

    # 4) SAC-style directionless maker proxy on primary-cash region
    best = None
    for beta in (0.70, 0.82):
        for alpha in (0.20, 0.30):
            for prob_min in (0.58, 0.64):
                for sig_th in (0.008, 0.012):
                    for notional in (0.9, 1.1, 1.3):
                        val_dec = _apply_directionless_maker_proxy(
                            val_df,
                            primary_val,
                            beta=beta,
                            hurst_th=0.45,
                            sig_th=sig_th,
                            alpha=alpha,
                            prob_min=prob_min,
                            notional=notional,
                            tp=0.0028,
                            sl=0.0030,
                            hold=6,
                        )
                        eval_dec = _apply_directionless_maker_proxy(
                            eval_df,
                            primary_eval,
                            beta=beta,
                            hurst_th=0.45,
                            sig_th=sig_th,
                            alpha=alpha,
                            prob_min=prob_min,
                            notional=notional,
                            tp=0.0028,
                            sl=0.0030,
                            hold=6,
                        )
                        val_m = _evaluate_combo(
                            val_df,
                            primary_val,
                            val_dec,
                            ref_parent=ref_parent,
                            noop_runner=noop_runner,
                            noop_cfg=noop_cfg,
                            fee=fee,
                            slip=slip,
                        )
                        sel = float(_score(val_m))
                        if best is None or sel > best.selection_score:
                            eval_m = _evaluate_combo(
                                eval_df,
                                primary_eval,
                                eval_dec,
                                ref_parent=ref_parent,
                                noop_runner=noop_runner,
                                noop_cfg=noop_cfg,
                                fee=fee,
                                slip=slip,
                            )
                            best = MethodResult(
                                method="directionless_maker_softq_proxy",
                                params={
                                    "beta": beta,
                                    "alpha": alpha,
                                    "prob_min": prob_min,
                                    "sig_th": sig_th,
                                    "notional": notional,
                                    "tp": 0.0028,
                                    "sl": 0.0030,
                                    "hold": 6,
                                },
                                selection_score=sel,
                                val_metrics=val_m,
                                oos_metrics=eval_m,
                            )
    assert best is not None
    results.append(best)

    rows = []
    for r in results:
        rows.append(
            {
                "method": r.method,
                "selection_score": float(r.selection_score),
                "val_cost3_pnl": float(r.val_metrics["cost3"]["pnl"]),
                "val_cost3_mdd": float(r.val_metrics["cost3"]["mdd"]),
                "val_cost3_trades": int(r.val_metrics["cost3"]["trades"]),
                "oos_cost3_pnl": float(r.oos_metrics["cost3"]["pnl"]),
                "oos_cost3_mdd": float(r.oos_metrics["cost3"]["mdd"]),
                "oos_cost3_trades": int(r.oos_metrics["cost3"]["trades"]),
                "oos_cost3_wr": float(r.oos_metrics["cost3"]["wr"]),
                "delta_vs_baseline_oos_cost3_pnl": float(r.oos_metrics["cost3"]["pnl"]) - float(baseline_eval["cost3"]["pnl"]),
                "params": json.dumps(r.params, ensure_ascii=False, default=_json_default),
            }
        )

    ranking = pd.DataFrame(rows).sort_values(["selection_score", "oos_cost3_pnl"], ascending=[False, False]).reset_index(drop=True)
    ranking_path = args.out_dir / "ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    report = {
        "model_id": "alpha7_fallback_microstructure_retry_20260526",
        "design": (
            "Retry 4 sparse-entry microstructure ideas on Alpha7 primary CASH region under the same runtime-native "
            "accounting contract: (1) rough-vol/local-Hurst mean-reversion gate, (2) Hawkes-style exhaustion gate, "
            "(3) xLSTM-style sequence toxic-flow embedding gate via light sequence autoencoder, "
            "(4) SAC-style directionless maker soft-Q proxy."
        ),
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "baseline": {
            "val": baseline_val,
            "oos": baseline_eval,
            "baseline_fallback": candidate_specs[0].name if candidate_specs else None,
        },
        "results": [
            {
                "method": r.method,
                "params": r.params,
                "selection_score": r.selection_score,
                "val_metrics": r.val_metrics,
                "oos_metrics": r.oos_metrics,
            }
            for r in results
        ],
        "artifacts": {
            "ranking_csv": str(ranking_path),
        },
        "audit": {
            "selection_uses_2026": False,
            "selection_window": "2025-10-01..2025-12-31",
            "oos_window": "2026 fixed OOS",
            "fail_fast_contract": True,
            "notes": [
                "Method 3 is a lightweight sequence autoencoder retry using available 5m aggregates (no raw tick replay in this run).",
                "Method 4 is SAC-style soft-Q proxy in discrete maker-like action space under current backtest execution contract.",
            ],
        },
    }
    report_path = args.out_dir / "summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")

    top = ranking.iloc[0].to_dict()
    print(
        json.dumps(
            {
                "report": str(report_path),
                "ranking": str(ranking_path),
                "best_method": top.get("method"),
                "best_oos_cost3_pnl": float(top.get("oos_cost3_pnl", 0.0)),
                "best_delta_vs_baseline": float(top.get("delta_vs_baseline_oos_cost3_pnl", 0.0)),
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
