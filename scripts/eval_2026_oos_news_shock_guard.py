#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from ensemble.train_rl_dsac_agent import DSAC_STATE_DIM, DSACRouter, GaussianActor
from features.news_shock_guard import NewsShockGuardConfig, compute_news_shock_guard

ANNUAL_FACTOR_5M = math.sqrt(365 * 24 * 12)
RL_CSV = "data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv"
FEAT_CSV = "data/splits/year_oos/training_features_2026_rebuilt.csv"
CKPT = "data/ensemble/ckpt/best_dsac_agents.pth"
OUT_JSON = "data/ensemble/reports/eval_2026_oos_news_shock_guard.json"
FEE, SLIP = 0.0005, 0.0002


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate active news shock guard on official 2026 OOS harness")
    p.add_argument("--rl-csv", default=RL_CSV)
    p.add_argument("--feat-csv", default=FEAT_CSV)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--out-json", default=OUT_JSON)
    return p.parse_args()


def _load_2026_df(rl_csv: str, feat_csv: str) -> pd.DataFrame:
    rl = pd.read_csv(rl_csv)
    rl["timestamp"] = pd.to_datetime(rl["timestamp"], errors="coerce")
    df26 = rl.loc[rl["timestamp"].dt.year == 2026].copy().reset_index(drop=True)
    need_ohlc = [c for c in ("open", "high", "low") if c not in df26.columns]
    if need_ohlc:
        feat = pd.read_csv(feat_csv, usecols=["timestamp", "open", "high", "low"])
        feat["timestamp"] = pd.to_datetime(feat["timestamp"], errors="coerce")
        df26 = df26.merge(feat, on="timestamp", how="left", suffixes=("", "_feat"))
        for c in ("open", "high", "low"):
            feat_c = f"{c}_feat"
            if c not in df26.columns and feat_c in df26.columns:
                df26[c] = df26[feat_c]
    for c in ("close", "open", "high", "low"):
        df26[c] = pd.to_numeric(df26[c], errors="coerce")
    df26 = df26.replace([np.inf, -np.inf], np.nan).dropna(subset=["close", "open", "high", "low"]).reset_index(drop=True)
    return df26


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
    m7_expected_ret = nz(col("m7_expected_ret"), 0.0)
    m7_mtl_dn = nz(col("m7_mtl_dn"), 0.0)
    m7_mtl_up = nz(col("m7_mtl_up"), 0.0)

    out["signal_bias"] = np.tanh(net_taker_ratio)
    out["nif_whale"] = np.tanh(0.70 * whale_conviction + 0.30 * sig_whale)
    out["taker_buy_ratio"] = (taker_buy_base / volume).clip(0.0, 1.0)

    toxicity = 0.42 * volatility_z.abs() + 0.33 * amihud_illiquidity_z.abs() + 0.25 * liquidity_vacuum.abs()
    out["shadow_toxicity_score"] = toxicity.clip(0.0, 1.0)
    out["shadow_queue_collapse"] = (0.55 * liquidity_vacuum.abs() + 0.45 * sig_liquidity_trap.abs()).clip(0.0, 1.0)
    out["shadow_absorption_score"] = (0.55 * sig_volume_confirm.clip(lower=0.0) + 0.45 * (1.0 - out["shadow_queue_collapse"])).clip(0.0, 1.0)

    aftershock = (
        0.25 * evt_tail_flag.abs()
        + 0.20 * jump_flag.abs()
        + 0.20 * evt_excess_z.abs().clip(0.0, 3.0) / 3.0
        + 0.20 * long_squeeze_risk.clip(0.0, 1.0)
        + 0.15 * short_squeeze_risk.clip(0.0, 1.0)
    )
    out["shadow_aftershock_prob"] = aftershock.clip(0.0, 1.0)
    out["shadow_decay_half_life"] = ou_halflife.clip(lower=0.0)
    out["shadow_risk_bucket"] = np.where(out["shadow_aftershock_prob"] >= 0.75, 2, np.where(out["shadow_aftershock_prob"] >= 0.45, 1, 0))

    probs = pd.concat([m7_mtl_dn, m7_mtl_up], axis=1).clip(lower=0.0)
    probs_sum = probs.sum(axis=1).replace(0.0, np.nan)
    probs_norm = probs.div(probs_sum, axis=0).fillna(0.5)
    out["mode_prob"] = probs_norm.max(axis=1).clip(0.0, 1.0)
    top2 = np.sort(probs_norm.to_numpy(dtype=np.float64), axis=1)[:, -2:]
    out["mode_spread"] = (top2[:, 1] - top2[:, 0]).clip(0.0, 1.0)
    entropy = -(probs_norm * np.log(np.clip(probs_norm, 1e-9, 1.0))).sum(axis=1) / math.log(2.0)
    out["entropy"] = entropy.clip(0.0, 1.0)

    out["tail_down_prob"] = m7_mtl_dn.clip(0.0, 1.0)
    out["tail_up_prob"] = m7_mtl_up.clip(0.0, 1.0)
    out["target_gap"] = m7_expected_ret.clip(-0.10, 0.10)
    out["target_gap_delta_1m"] = out["target_gap"].diff().fillna(0.0).clip(-0.05, 0.05)
    out["prob_mom_1m"] = out["mode_prob"].diff().fillna(0.0).clip(-0.50, 0.50)
    return out


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


def simulate(df26: pd.DataFrame, actor: GaussianActor, device: str, cfg: NewsShockGuardConfig | None = None) -> dict:
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
    cooldown = 0
    meta = {"trigger_count": 0, "severe_count": 0, "reduce_count": 0, "flat_count": 0, "cooldown_blocks": 0}

    def _unr(p: str | None, ep: float, cp: float, lv: float) -> float:
        if p is None or ep <= 0 or lv <= 0:
            return 0.0
        raw = (cp * (1 - SLIP) - ep) / ep if p == "LONG" else (ep - cp * (1 + SLIP)) / ep
        return raw * lv

    def _real(p: str, ep: float, xp: float, lv: float) -> float:
        raw = (xp * (1 - SLIP) - ep) / ep if p == "LONG" else (ep - xp * (1 + SLIP)) / ep
        return raw * lv

    iterator = tqdm(range(len(df26) - 1), desc=("guard-baseline" if cfg is None else f"guard-{cfg.name}"), unit="bar", ncols=110)
    for i in iterator:
        cp = float(close_np[i])
        next_open = float(open_np[i + 1])
        next_close = float(close_np[i + 1])

        if pos is not None:
            hold_count += 1
        if cooldown > 0:
            cooldown -= 1

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
        features = {k: float(v) for k, v in zip(numeric_cols, values[i])}
        action_int, lev, _ = router.decide(features, pos_dict)
        lev = float(np.clip(lev, 0.0, 1.0))

        if cfg is not None and pos is not None:
            guard = compute_news_shock_guard(pos, features, cfg=cfg)
            if bool(guard["trigger"]):
                meta["trigger_count"] += 1
                if bool(guard["severe"]):
                    meta["severe_count"] += 1
                reduce_mult = float(guard["reduce_mult"])
                cooldown = max(cooldown, int(guard["cooldown_bars"]))
                if reduce_mult <= 0.0:
                    realized = _real(pos, entry_price, next_open, cur_lev)
                    balance = balance * (1.0 + realized) - balance * FEE * cur_lev
                    trades += 1
                    if realized > 0:
                        wins += 1
                    pos = None
                    entry_price = 0.0
                    cur_lev = 0.0
                    hold_count = 0
                    meta["flat_count"] += 1
                else:
                    new_lev = float(np.clip(cur_lev * reduce_mult, 0.0, 1.0))
                    if new_lev + 1e-12 < cur_lev:
                        delta = cur_lev - new_lev
                        balance -= balance * FEE * delta
                        cur_lev = new_lev
                        meta["reduce_count"] += 1

        if pos is None:
            if cooldown > 0:
                if action_int in (1, 2):
                    meta["cooldown_blocks"] += 1
            else:
                if action_int == 1 and lev > 0.0:
                    pos = "LONG"
                    entry_price = next_open * (1 + SLIP)
                    cur_lev = lev
                    hold_count = 0
                    balance -= balance * FEE * cur_lev
                elif action_int == 2 and lev > 0.0:
                    pos = "SHORT"
                    entry_price = next_open * (1 - SLIP)
                    cur_lev = lev
                    hold_count = 0
                    balance -= balance * FEE * cur_lev
        else:
            should_close = (
                action_int == 0
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
    return {
        "pnl_pct": round(pnl, 4),
        "wr_pct": round(wr, 2),
        "trades": trades,
        "sharpe": round(_sharpe(eq_curve), 4),
        "mdd_pct": round(_mdd(eq_curve), 4),
        "meta": meta,
    }


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    actor = GaussianActor(state_dim=DSAC_STATE_DIM).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    df26 = build_proxy_frame(_load_2026_df(args.rl_csv, args.feat_csv))
    baseline = simulate(df26, actor, device, cfg=None)

    grid = [
        NewsShockGuardConfig(shock_trigger_th=0.88, aftershock_trigger_th=0.80, toxicity_trigger_th=0.74, queue_trigger_th=0.78, poly_momentum_trigger=0.008, poly_gap_trigger=0.015, reduce_mult=0.50, severe_reduce_mult=0.0, cooldown_bars=4, severe_cooldown_bars=8),
        NewsShockGuardConfig(shock_trigger_th=0.84, aftershock_trigger_th=0.76, toxicity_trigger_th=0.70, queue_trigger_th=0.74, poly_momentum_trigger=0.007, poly_gap_trigger=0.013, reduce_mult=0.35, severe_reduce_mult=0.0, cooldown_bars=5, severe_cooldown_bars=9),
        NewsShockGuardConfig(shock_trigger_th=0.80, aftershock_trigger_th=0.72, toxicity_trigger_th=0.66, queue_trigger_th=0.70, poly_momentum_trigger=0.006, poly_gap_trigger=0.011, reduce_mult=0.50, severe_reduce_mult=0.20, cooldown_bars=4, severe_cooldown_bars=8),
        NewsShockGuardConfig(shock_trigger_th=0.92, aftershock_trigger_th=0.84, toxicity_trigger_th=0.78, queue_trigger_th=0.82, poly_momentum_trigger=0.010, poly_gap_trigger=0.018, reduce_mult=0.35, severe_reduce_mult=0.0, cooldown_bars=6, severe_cooldown_bars=10),
    ]

    results = []
    for cfg in tqdm(grid, desc="official-news-guard-grid", ncols=110):
        res = simulate(df26, actor, device, cfg=cfg)
        res["config"] = asdict(cfg)
        res["name"] = cfg.name
        res["baseline"] = baseline
        res["delta_pct"] = round(res["pnl_pct"] - baseline["pnl_pct"], 4)
        results.append(res)

    results.sort(key=lambda x: (x["delta_pct"], x["mdd_pct"], x["sharpe"]), reverse=True)
    report = {
        "checkpoint": args.ckpt,
        "checkpoint_best_val_pnl": float(ckpt.get("best_pnl", 0.0)),
        "checkpoint_epoch": int(ckpt.get("epoch", 0)),
        "rl_csv": args.rl_csv,
        "feat_csv": args.feat_csv,
        "data_period": "2026-01-01 ~ 2026-02-28",
        "data_rows": len(df26),
        "note": "Official 2026 OOS harness with active news shock guard only. No passive hold caps.",
        "baseline": baseline,
        "top3": results[:3],
        "all_results": results,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    best = results[0]
    print("=== Official 2026 News Shock Guard ===")
    print(f"baseline={baseline['pnl_pct']:+.4f}% best={best['pnl_pct']:+.4f}% delta={best['delta_pct']:+.4f}%p")
    print(f"best_name={best['name']} mdd={best['mdd_pct']:+.4f}% sharpe={best['sharpe']:.4f} meta={best['meta']}")
    print(f"report={args.out_json}")


if __name__ == "__main__":
    main()
