#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in [_ROOT_DIR, os.path.join(_ROOT_DIR, "ensemble")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.rl_runtime_primitives import OnlineHMMDetector, MultiTimeframeFeatures
from ensemble.train_rl_dsac_agent import (
    DSAC_STATE_DIM,
    DSACCompactTradingEnv,
    DSACRouter,
    GaussianActor,
)
from features.playbook_meta_controller import PlaybookMetaConfig, compute_playbook_meta_controller

ANNUAL_FACTOR_5M = math.sqrt(365 * 24 * 12)
RL_CSV = "data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv"
FEAT_CSV = "data/splits/year_oos/training_features_2026_rebuilt.csv"
CKPT = "data/ensemble/ckpt/best_dsac_agents.pth"
OUT_JSON = "data/ensemble/reports/eval_2026_oos_playbook_controller.json"
FEE, SLIP = 0.0005, 0.0002


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate sparse playbook controller on official 2026 OOS harness")
    p.add_argument("--rl-csv", default=RL_CSV)
    p.add_argument("--feat-csv", default=FEAT_CSV)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--out-json", default=OUT_JSON)
    return p.parse_args()


def _load_2026_df(rl_csv: str, feat_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rl = pd.read_csv(rl_csv)
    rl["timestamp"] = pd.to_datetime(rl["timestamp"], errors="coerce")

    mask_rl = rl["timestamp"].dt.year == 2026
    df26_env = rl.loc[mask_rl].copy().reset_index(drop=True)

    df26_ohlc = df26_env.copy()
    need_ohlc = [c for c in ("open", "high", "low") if c not in df26_ohlc.columns]
    if need_ohlc:
        feat = pd.read_csv(feat_csv, usecols=["timestamp", "open", "high", "low"])
        feat["timestamp"] = pd.to_datetime(feat["timestamp"], errors="coerce")
        df_merged = df26_ohlc.merge(feat, on="timestamp", how="left", suffixes=("", "_feat"))
        for c in ("open", "high", "low"):
            feat_c = f"{c}_feat"
            if c not in df_merged.columns and feat_c in df_merged.columns:
                df_merged[c] = df_merged[feat_c]
        df26_ohlc = df_merged

    for c in ("close", "open", "high", "low"):
        if c in df26_ohlc.columns:
            df26_ohlc[c] = pd.to_numeric(df26_ohlc[c], errors="coerce")
    df26_ohlc = df26_ohlc.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["close", "open", "high", "low"]
    ).reset_index(drop=True)

    print(
        f"[DATA] 2026 env_rows={len(df26_env):,} ohlc_rows={len(df26_ohlc):,} "
        f"range={df26_env['timestamp'].min()} -> {df26_env['timestamp'].max()}"
    )
    return df26_env, df26_ohlc


def _sharpe(eq_curve: list[float]) -> float:
    eq = np.array(eq_curve, dtype=np.float64)
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    if len(rets) < 3 or np.std(rets) < 1e-12:
        return 0.0
    return float(np.mean(rets) / np.std(rets) * ANNUAL_FACTOR_5M)


def _mdd(eq_curve: list[float]) -> float:
    eq = np.array(eq_curve, dtype=np.float64)
    run_max = np.maximum.accumulate(eq)
    dd = eq / np.maximum(run_max, 1e-12) - 1.0
    return float(np.min(dd)) * 100.0


def method1_training_env(df26: pd.DataFrame, actor: GaussianActor, device: str) -> dict:
    print("\n[METHOD 1] DSACCompactTradingEnv")
    hmm = OnlineHMMDetector()
    try:
        df_train = pd.read_csv(RL_CSV, usecols=["timestamp", "log_return", "garch_vol_z", "oi_change_rate"])
        df_train["timestamp"] = pd.to_datetime(df_train["timestamp"], errors="coerce")
        df_train_2024 = df_train[df_train["timestamp"].dt.year < 2025].copy()
    except Exception:
        df_train_2024 = pd.DataFrame()
    if df_train_2024.empty:
        return {
            "method": "training_env",
            "skipped": True,
            "reason": "No pre-2025 rows available for HMM fit in current RL csv.",
        }
    hmm.fit(df_train_2024, n_iter=30)

    mtf = MultiTimeframeFeatures(df26["close"].values.astype(np.float32))
    env = DSACCompactTradingEnv(
        df26,
        initial_balance=10000.0,
        fee=FEE,
        slip=SLIP,
        phase="val",
        hmm_detector=copy.deepcopy(hmm),
        mtf_features=mtf,
    )

    state = env.reset()
    done = False
    eq_curve = [env.initial_balance]
    actor.eval()
    with torch.no_grad():
        for _ in tqdm(range(len(df26)), desc="eval-training-env", unit="step"):
            if done:
                break
            action = float(torch.tanh(actor.forward(torch.FloatTensor(state).unsqueeze(0).to(device))[0]).item())
            state, _, done, _ = env.step(action)
            bal = env.balance * (1.0 + (env.unrealized_pnl if env.pos is not None else 0.0))
            eq_curve.append(max(bal, 1e-8))

    pnl = (env.balance / env.initial_balance - 1.0) * 100.0
    wr = env.win_rate * 100.0
    return {
        "method": "training_env",
        "pnl_pct": round(pnl, 4),
        "wr_pct": round(wr, 2),
        "trades": env.total_trades,
        "sharpe": round(_sharpe(eq_curve), 4),
        "mdd_pct": round(_mdd(eq_curve), 4),
    }


def build_proxy_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def col(name: str, default: float = 0.0) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").fillna(default)
        return pd.Series(default, index=out.index, dtype=np.float64)

    def nz(series: pd.Series, fill: float = 0.0) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(fill)

    close = nz(col("close"), 0.0).clip(lower=1e-9)
    volume = nz(col("volume"), 0.0).clip(lower=1e-9)
    taker_buy_base = nz(col("taker_buy_base"), 0.0)
    net_taker_ratio = nz(col("net_taker_ratio"), 0.0)
    whale_conviction = nz(col("whale_conviction"), 0.0)
    sig_whale = nz(col("sig_whale"), 0.0)
    volatility_z = nz(col("volatility_z"), 0.0)
    amihud_illiquidity_z = nz(col("amihud_illiquidity_z"), 0.0)
    liquidity_vacuum = nz(col("liquidity_vacuum"), 0.0)
    sig_liquidity_trap = nz(col("sig_liquidity_trap"), 0.0)
    sig_volume_confirm = nz(col("sig_volume_confirm"), 0.0)
    evt_tail_flag = nz(col("evt_tail_flag"), 0.0)
    jump_flag = nz(col("jump_flag"), 0.0)
    evt_excess_z = nz(col("evt_excess_z"), 0.0)
    long_squeeze_risk = nz(col("long_squeeze_risk"), 0.0)
    short_squeeze_risk = nz(col("short_squeeze_risk"), 0.0)
    ou_halflife = nz(col("ou_halflife"), 0.0)
    m7_conf = nz(col("m7_confidence"), 0.0)
    m7_qwidth = nz(col("m7_qwidth"), 0.0)
    m7_expected_ret = nz(col("m7_expected_ret"), 0.0)
    m7_mtl_dn = nz(col("m7_mtl_dn"), 0.0)
    m7_mtl_up = nz(col("m7_mtl_up"), 0.0)
    m7_quant_10 = nz(col("m7_quantile_10"), 0.0)
    m7_quant_50 = nz(col("m7_quantile_50"), 0.0)
    m7_quant_90 = nz(col("m7_quantile_90"), 0.0)

    out["signal_bias"] = np.tanh(net_taker_ratio)
    out["nif_whale"] = np.tanh(0.70 * whale_conviction + 0.30 * sig_whale)
    out["taker_buy_ratio"] = (taker_buy_base / volume).clip(0.0, 1.0)

    toxicity = 0.42 * volatility_z.abs() + 0.33 * amihud_illiquidity_z.abs() + 0.25 * liquidity_vacuum.abs()
    out["shadow_toxicity_score"] = toxicity.clip(0.0, 1.0)
    out["shadow_queue_collapse"] = (0.55 * liquidity_vacuum.abs() + 0.45 * sig_liquidity_trap.abs()).clip(0.0, 1.0)
    out["shadow_absorption_score"] = (0.55 * sig_volume_confirm.clip(lower=0.0) + 0.45 * (1.0 - out["shadow_queue_collapse"])).clip(0.0, 1.0)
    out["shadow_regime_conf"] = (0.60 * m7_conf + 0.40 * (1.0 - out["shadow_toxicity_score"])).clip(0.0, 1.0)

    aftershock = (
        0.25 * evt_tail_flag.abs()
        + 0.20 * jump_flag.abs()
        + 0.20 * evt_excess_z.abs().clip(0.0, 3.0) / 3.0
        + 0.20 * long_squeeze_risk.clip(0.0, 1.0)
        + 0.15 * short_squeeze_risk.clip(0.0, 1.0)
    )
    out["shadow_aftershock_prob"] = aftershock.clip(0.0, 1.0)
    out["shadow_decay_half_life"] = ou_halflife.clip(lower=0.0)
    risk_bucket = np.where(out["shadow_aftershock_prob"] >= 0.75, 2, np.where(out["shadow_aftershock_prob"] >= 0.45, 1, 0))
    out["shadow_risk_bucket"] = risk_bucket.astype(np.float64)

    probs = pd.concat([m7_mtl_dn, m7_mtl_up], axis=1).clip(lower=0.0)
    probs_sum = probs.sum(axis=1).replace(0.0, np.nan)
    probs_norm = probs.div(probs_sum, axis=0).fillna(0.5)
    out["mode_prob"] = probs_norm.max(axis=1).clip(0.0, 1.0)
    top2 = np.sort(probs_norm.to_numpy(dtype=np.float64), axis=1)[:, -2:]
    out["mode_spread"] = (top2[:, 1] - top2[:, 0]).clip(0.0, 1.0)
    entropy = -(probs_norm * np.log(np.clip(probs_norm, 1e-9, 1.0))).sum(axis=1) / math.log(2.0)
    out["entropy"] = entropy.clip(0.0, 1.0)

    out["tail_down_prob"] = ((m7_mtl_dn * 0.70) + (m7_quant_10 > 0).astype(float) * 0.30).clip(0.0, 1.0)
    out["tail_up_prob"] = ((m7_mtl_up * 0.70) + (m7_quant_90 > 0).astype(float) * 0.30).clip(0.0, 1.0)
    out["target_gap"] = m7_expected_ret.clip(-0.10, 0.10)
    out["target_gap_delta_1m"] = out["target_gap"].diff().fillna(0.0).clip(-0.05, 0.05)
    out["prob_mom_1m"] = out["mode_prob"].diff().fillna(0.0).clip(-0.50, 0.50)
    out["weighted_target"] = close * (1.0 + out["target_gap"])
    out["m7_qwidth"] = m7_qwidth
    out["m7_confidence"] = m7_conf
    out["close"] = close
    return out


def _closed_loop_core(
    df26: pd.DataFrame,
    actor: GaussianActor,
    device: str,
    cfg: PlaybookMetaConfig | None = None,
) -> dict:
    router = DSACRouter(actor, device=device)
    numeric_cols = [c for c in df26.columns if c != "timestamp"]
    values = df26[numeric_cols].to_numpy(dtype=np.float64)
    open_np = df26["open"].to_numpy(dtype=np.float64)
    close_np = df26["close"].to_numpy(dtype=np.float64)

    balance = 1.0
    pos: str | None = None
    entry_price = 0.0
    cur_lev = 0.0
    hold_count = 0
    trades = wins = 0
    eq_curve = [1.0]
    pending_side: str | None = None
    pending_delay = 0
    pending_size_mult = 1.0
    n = len(df26)
    meta_stats = {
        "skipped": 0,
        "delayed": 0,
        "reduced": 0,
        "boosted": 0,
        "hold_capped": 0,
        "meta_exits": 0,
    }

    def _unr(p: str | None, ep: float, cp: float, lv: float) -> float:
        if p is None or ep <= 0 or lv <= 0:
            return 0.0
        raw = (cp * (1 - SLIP) - ep) / ep if p == "LONG" else (ep - cp * (1 + SLIP)) / ep
        return raw * lv

    def _real(p: str, ep: float, xp: float, lv: float) -> float:
        raw = (xp * (1 - SLIP) - ep) / ep if p == "LONG" else (ep - xp * (1 + SLIP)) / ep
        return raw * lv

    iterator = tqdm(range(n - 1), desc=("eval-closed-loop-base" if cfg is None else f"eval-closed-loop-{cfg.name}"), unit="bar")
    for i in iterator:
        cp = float(close_np[i])
        next_open = float(open_np[i + 1])
        next_close = float(close_np[i + 1])

        if pos is not None:
            hold_count += 1

        unr = _unr(pos, entry_price, cp, cur_lev)
        pos_dict = {
            "type": pos,
            "entry_price": float(entry_price),
            "unrealized": float(unr),
            "mdd": 0.0,
            "hold_norm": float(min(hold_count / 96.0, 1.0)),
            "margin_usage": float(cur_lev if pos else 0.0),
            "hold_count": float(hold_count),
        }
        row = values[i]
        features = {k: float(v) for k, v in zip(numeric_cols, row)}
        action_int, lev, _ = router.decide(features, pos_dict)
        lev = float(np.clip(lev, 0.0, 1.0))
        desired_side = None if action_int == 0 else ("LONG" if action_int == 1 else "SHORT")
        entry_size_mult = 1.0
        exit_now = False

        if cfg is not None:
            controller = compute_playbook_meta_controller(
                {
                    "position": pos or "FLAT",
                    "close": cp,
                    "entry_price": entry_price if pos else 0.0,
                    "hold_bars": hold_count if pos else 0,
                    "micro": {
                        "signal_bias": features.get("signal_bias", 0.0),
                        "nif_whale": features.get("nif_whale", 0.0),
                        "taker_buy_ratio": features.get("taker_buy_ratio", 0.5),
                        "toxicity": features.get("shadow_toxicity_score", 0.0),
                        "queue_collapse": features.get("shadow_queue_collapse", 0.0),
                        "absorption": features.get("shadow_absorption_score", 0.0),
                        "regime_conf": features.get("shadow_regime_conf", 0.5),
                    },
                    "tail": {
                        "aftershock_prob": features.get("shadow_aftershock_prob", 0.0),
                        "decay_half_life": features.get("shadow_decay_half_life", 0.0),
                        "risk_bucket": features.get("shadow_risk_bucket", 0.0),
                    },
                    "poly": {
                        "weighted_target": features.get("weighted_target", cp),
                        "target_gap": features.get("target_gap", 0.0),
                        "target_gap_delta_1m": features.get("target_gap_delta_1m", 0.0),
                        "prob_mom_1m": features.get("prob_mom_1m", 0.0),
                        "mode_prob": features.get("mode_prob", 0.33),
                        "mode_spread": features.get("mode_spread", 0.0),
                        "entropy": features.get("entropy", 1.0),
                        "tail_up_prob": features.get("tail_up_prob", 0.0),
                        "tail_down_prob": features.get("tail_down_prob", 0.0),
                    },
                    "m7": {
                        "confidence": features.get("m7_confidence", 0.5),
                        "qwidth": features.get("m7_qwidth", 0.0),
                    },
                    "dsac": {
                        "desired_side": desired_side or "FLAT",
                        "leverage": lev,
                    },
                },
                cfg,
            )
            if controller.get("skip_entry") and pos is None and desired_side is not None:
                meta_stats["skipped"] += 1
                desired_side = None
                lev = 0.0
            if controller.get("delay_bars", 0) > 0 and pos is None and desired_side is not None:
                pending_side = desired_side
                pending_delay = int(controller["delay_bars"])
                pending_size_mult = float(controller.get("size_mult", 1.0))
                meta_stats["delayed"] += 1
                desired_side = None
                lev = 0.0
            else:
                entry_size_mult = float(controller.get("size_mult", 1.0))

            if entry_size_mult < 0.999:
                meta_stats["reduced"] += 1
            elif entry_size_mult > 1.001:
                meta_stats["boosted"] += 1

            exit_now = bool(controller.get("exit_now", False))
            if exit_now and pos is not None:
                meta_stats["meta_exits"] += 1

        if pending_side is not None and pos is None:
            pending_delay -= 1
            if pending_delay <= 0:
                if desired_side == pending_side:
                    desired_side = pending_side
                    lev = float(np.clip(lev * pending_size_mult, 0.0, 1.0))
                pending_side = None
                pending_delay = 0
                pending_size_mult = 1.0

        if pos is None:
            if desired_side == "LONG" and lev > 0.0:
                eff_lev = float(np.clip(lev * entry_size_mult, 0.0, 1.0))
                if eff_lev > 0.0:
                    pos = "LONG"
                    entry_price = next_open * (1 + SLIP)
                    cur_lev = eff_lev
                    hold_count = 0
                    balance -= balance * FEE * cur_lev
            elif desired_side == "SHORT" and lev > 0.0:
                eff_lev = float(np.clip(lev * entry_size_mult, 0.0, 1.0))
                if eff_lev > 0.0:
                    pos = "SHORT"
                    entry_price = next_open * (1 - SLIP)
                    cur_lev = eff_lev
                    hold_count = 0
                    balance -= balance * FEE * cur_lev
        else:
            should_close = (
                exit_now
                or action_int == 0
                or (action_int == 1 and pos == "SHORT")
                or (action_int == 2 and pos == "LONG")
            )
            if should_close:
                realized = _real(pos, entry_price, next_open, cur_lev)
                balance = balance * (1.0 + realized) - balance * FEE * cur_lev
                trades += 1
                if realized > 0:
                    wins += 1
                pos = None
                entry_price = 0.0
                cur_lev = 0.0
                hold_count = 0
            else:
                delta = abs(lev - cur_lev)
                if delta > 0.05:
                    balance -= balance * FEE * delta
                    cur_lev = lev

        eq = balance * (1 + _unr(pos, entry_price, next_close, cur_lev)) if pos else balance
        eq_curve.append(max(float(eq), 1e-8))

    if pos and entry_price > 0:
        realized = _real(pos, entry_price, float(close_np[-1]), cur_lev)
        balance = balance * (1.0 + realized) - balance * FEE * cur_lev
        trades += 1
        if realized > 0:
            wins += 1

    pnl = (balance - 1.0) * 100.0
    wr = (wins / trades * 100.0) if trades > 0 else 0.0
    result = {
        "method": "closed_loop" if cfg is None else f"closed_loop_playbook::{cfg.name}",
        "pnl_pct": round(pnl, 4),
        "wr_pct": round(wr, 2),
        "trades": trades,
        "sharpe": round(_sharpe(eq_curve), 4),
        "mdd_pct": round(_mdd(eq_curve), 4),
        "meta": meta_stats,
    }
    if cfg is not None:
        result["config"] = asdict(cfg)
        result["delta_pct"] = round(result["pnl_pct"] - 0.0, 4)
    return result


def _coarse_grid() -> list[tuple[str, PlaybookMetaConfig]]:
    seeds = [
        dict(event_k=0.95, hazard_k=1.04, continuation_k=0.95, pullback_k=1.05, size_boost=0.04, size_floor=0.93, hold_base_bars=360, hold_scale=0.07, exit_aggr=0.95, skip_hazard_th=0.93, sparse_event_th=0.84, sparse_hazard_th=0.80, severe_exit_th=0.93, mild_reduce_th=0.87),
        dict(event_k=0.95, hazard_k=1.10, continuation_k=0.95, pullback_k=1.05, size_boost=0.05, size_floor=0.90, hold_base_bars=300, hold_scale=0.07, exit_aggr=0.95, skip_hazard_th=0.93, sparse_event_th=0.82, sparse_hazard_th=0.80, severe_exit_th=0.93, mild_reduce_th=0.87),
        dict(event_k=0.90, hazard_k=1.05, continuation_k=0.95, pullback_k=1.00, size_boost=0.04, size_floor=0.88, hold_base_bars=360, hold_scale=0.06, exit_aggr=0.90, skip_hazard_th=0.94, sparse_event_th=0.84, sparse_hazard_th=0.82, severe_exit_th=0.94, mild_reduce_th=0.88),
        dict(event_k=0.98, hazard_k=1.08, continuation_k=0.95, pullback_k=1.00, size_boost=0.04, size_floor=0.94, hold_base_bars=420, hold_scale=0.05, exit_aggr=0.90, skip_hazard_th=0.95, sparse_event_th=0.86, sparse_hazard_th=0.84, severe_exit_th=0.95, mild_reduce_th=0.90),
    ]
    return [(f"official_{i+1}", PlaybookMetaConfig(**cfg)) for i, cfg in enumerate(seeds)]


def _refine_grid(base: PlaybookMetaConfig) -> list[tuple[str, PlaybookMetaConfig]]:
    grid: list[tuple[str, PlaybookMetaConfig]] = []
    for hazard_k in [max(0.9, base.hazard_k - 0.04), base.hazard_k, base.hazard_k + 0.04]:
        for sparse_event_th in [max(0.75, base.sparse_event_th - 0.02), base.sparse_event_th, min(0.92, base.sparse_event_th + 0.02)]:
            for severe_exit_th in [max(0.88, base.severe_exit_th - 0.02), base.severe_exit_th, min(0.98, base.severe_exit_th + 0.02)]:
                name = f"ref_h{hazard_k:.2f}_se{sparse_event_th:.2f}_sx{severe_exit_th:.2f}"
                cfg = PlaybookMetaConfig(
                    event_k=base.event_k,
                    hazard_k=hazard_k,
                    continuation_k=base.continuation_k,
                    pullback_k=base.pullback_k,
                    size_boost=base.size_boost,
                    size_floor=base.size_floor,
                    size_cap=base.size_cap,
                    delay_scale=base.delay_scale,
                    hold_base_bars=base.hold_base_bars,
                    hold_scale=base.hold_scale,
                    exit_aggr=base.exit_aggr,
                    skip_hazard_th=base.skip_hazard_th,
                    sparse_event_th=sparse_event_th,
                    sparse_hazard_th=base.sparse_hazard_th,
                    severe_exit_th=severe_exit_th,
                    mild_reduce_th=base.mild_reduce_th,
                )
                grid.append((name, cfg))
    uniq: dict[str, PlaybookMetaConfig] = {}
    for name, cfg in grid:
        uniq[name] = cfg
    return list(uniq.items())


def _run_grid(df: pd.DataFrame, actor: GaussianActor, device: str, baseline: dict, grid: list[tuple[str, PlaybookMetaConfig]], desc: str) -> list[dict]:
    results: list[dict] = []
    for name, cfg in tqdm(grid, desc=desc, ncols=110):
        res = _closed_loop_core(df, actor, device, cfg=cfg)
        res["name"] = name
        res["config"] = asdict(cfg)
        res["baseline"] = baseline
        res["delta_pct"] = round(res["pnl_pct"] - baseline["pnl_pct"], 4)
        results.append(res)
    results.sort(key=lambda x: (x["delta_pct"], x["pnl_pct"], x["mdd_pct"], x["sharpe"]), reverse=True)
    return results


def _to_serializable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")
    print(f"[CKPT]   {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    actor = GaussianActor(state_dim=DSAC_STATE_DIM).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    df26_env, df26_ohlc = _load_2026_df(args.rl_csv, args.feat_csv)
    df26_ohlc = build_proxy_frame(df26_ohlc)

    r1 = method1_training_env(df26_env, actor, device)
    baseline = _closed_loop_core(df26_ohlc, actor, device, cfg=None)
    coarse_grid = _coarse_grid()
    coarse = _run_grid(df26_ohlc, actor, device, baseline, coarse_grid, "playbook-official-coarse")
    refine = _run_grid(
        df26_ohlc,
        actor,
        device,
        baseline,
        _refine_grid(PlaybookMetaConfig(**coarse[0]["config"])),
        "playbook-official-refine",
    )

    best = refine[0]
    report = {
        "checkpoint": args.ckpt,
        "checkpoint_best_val_pnl": float(ckpt.get("best_pnl", 0.0)),
        "checkpoint_epoch": int(ckpt.get("epoch", 0)),
        "rl_csv": args.rl_csv,
        "feat_csv": args.feat_csv,
        "data_period": "2026-01-01 ~ 2026-02-28",
        "data_rows": len(df26_env),
        "note": "Official 2026 OOS harness. Method2 baseline matches eval_2026_oos closed-loop rules; controller uses proxy playbook features built from 2026 OOS columns.",
        "results": [r1, baseline],
        "coarse_top3": coarse[:3],
        "refine_top5": refine[:5],
        "best_overall": best,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(report), f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("[SUMMARY] Official 2026 OOS with sparse playbook controller")
    if "pnl_pct" in r1:
        print(f"  Method1 (training env)        : {r1['pnl_pct']:.2f}%")
    else:
        print(f"  Method1 (training env)        : skipped ({r1.get('reason', 'n/a')})")
    print(f"  Method2 baseline (closedloop) : {baseline['pnl_pct']:.2f}%")
    print(f"  Method2 best playbook         : {best['pnl_pct']:.2f}%")
    print(f"  Delta vs baseline             : {best['delta_pct']:+.2f}%p")
    print("=" * 60)
    print(f"[SAVED] {args.out_json}")


if __name__ == "__main__":
    main()
