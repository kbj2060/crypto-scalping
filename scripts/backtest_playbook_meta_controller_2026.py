#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.train_rl_dsac_agent import DSACRouter, DSAC_STATE_DIM, GaussianActor
from features.playbook_meta_controller import PlaybookMetaConfig, compute_playbook_meta_controller

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


ANNUAL_FACTOR_5M = math.sqrt(365 * 24 * 12)


def _safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return float(x)


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), float(lo), float(hi)))


def _resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_actor(ckpt_path: str, device: str) -> GaussianActor:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor = GaussianActor(state_dim=int(ckpt.get("state_dim", DSAC_STATE_DIM))).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def _load_frame(rl_csv: str) -> pd.DataFrame:
    df = pd.read_csv(rl_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").reset_index(drop=True)
    for c in df.columns:
        if c != "timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)
    return df


def _entropy_2(a: float, b: float) -> float:
    p = np.asarray([max(a, 0.0), max(b, 0.0)], dtype=np.float64)
    s = float(p.sum())
    if s <= 1e-12:
        return 1.0
    p = p / s
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum() / np.log(2.0))


def build_proxy_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    vol = np.maximum(out["volume"].to_numpy(dtype=np.float64), 1e-8)
    taker_buy_base = out.get("taker_buy_base", pd.Series(0.0, index=out.index)).to_numpy(dtype=np.float64)
    out["signal_bias"] = np.tanh(out.get("net_taker_ratio", pd.Series(0.0, index=out.index)).to_numpy(dtype=np.float64))
    out["nif_whale"] = np.tanh(
        0.70 * out.get("whale_conviction", pd.Series(0.0, index=out.index)).to_numpy(dtype=np.float64)
        + 0.30 * out.get("sig_whale", pd.Series(0.0, index=out.index)).to_numpy(dtype=np.float64)
    )
    out["taker_buy_ratio"] = np.clip(taker_buy_base / vol, 0.0, 1.0)

    liquidity_vacuum = out.get("liquidity_vacuum", pd.Series(0.0, index=out.index)).to_numpy(dtype=np.float64)
    amihud = out.get("amihud_illiquidity_z", pd.Series(0.0, index=out.index)).to_numpy(dtype=np.float64)
    volz = out.get("volatility_z", pd.Series(0.0, index=out.index)).to_numpy(dtype=np.float64)
    sig_liq = out.get("sig_liquidity_trap", pd.Series(0.0, index=out.index)).to_numpy(dtype=np.float64)
    sig_vol = out.get("sig_volume_confirm", pd.Series(0.0, index=out.index)).to_numpy(dtype=np.float64)

    out["shadow_toxicity_score"] = np.clip(
        1.2
        * (
            0.50 * np.clip(np.abs(volz) / 3.0, 0.0, 1.0)
            + 0.25 * np.clip(amihud / 3.0, 0.0, 1.0)
            + 0.25 * np.clip(liquidity_vacuum, 0.0, 1.0)
        ),
        0.0,
        1.5,
    )
    out["shadow_queue_collapse"] = np.clip(
        0.65 * np.clip(liquidity_vacuum, 0.0, 1.0)
        + 0.20 * np.clip(np.abs(volz) / 3.0, 0.0, 1.0)
        + 0.15 * (1.0 - np.clip(sig_vol, 0.0, 1.0)),
        0.0,
        1.0,
    )
    out["shadow_absorption_score"] = np.clip(
        0.45 * np.clip(sig_liq, 0.0, 1.0)
        + 0.30 * np.clip(sig_vol, 0.0, 1.0)
        + 0.25 * (1.0 - out["shadow_queue_collapse"].to_numpy(dtype=np.float64)),
        0.0,
        1.0,
    )

    regime_cols = [c for c in ["regime_bull", "regime_bear", "regime_chop", "regime_whipsaw", "regime_normal"] if c in out.columns]
    if regime_cols:
        out["shadow_regime_conf"] = out[regime_cols].max(axis=1).clip(0.0, 1.0)
    else:
        out["shadow_regime_conf"] = 0.0

    evt_tail = out.get("evt_tail_flag", pd.Series(0.0, index=out.index)).to_numpy(dtype=np.float64)
    jump_flag = out.get("jump_flag", pd.Series(0.0, index=out.index)).to_numpy(dtype=np.float64)
    evt_excess = out.get("evt_excess_z", pd.Series(0.0, index=out.index)).to_numpy(dtype=np.float64)
    long_sq = out.get("long_squeeze_risk", pd.Series(0.0, index=out.index)).to_numpy(dtype=np.float64)
    short_sq = out.get("short_squeeze_risk", pd.Series(0.0, index=out.index)).to_numpy(dtype=np.float64)
    out["shadow_aftershock_prob"] = np.clip(
        0.35 * np.clip(evt_tail, 0.0, 1.0)
        + 0.20 * np.clip(jump_flag, 0.0, 1.0)
        + 0.20 * np.clip(np.abs(evt_excess) / 3.0, 0.0, 1.0)
        + 0.25 * np.clip(np.maximum(long_sq, short_sq), 0.0, 1.0),
        0.0,
        1.0,
    )
    out["shadow_decay_half_life"] = np.clip(out.get("ou_halflife", pd.Series(0.0, index=out.index)).to_numpy(dtype=np.float64), 0.0, 10.0)
    out["shadow_risk_bucket"] = np.where(
        out["shadow_aftershock_prob"] >= 0.72,
        "high",
        np.where(out["shadow_aftershock_prob"] >= 0.45, "watch", "normal"),
    )

    q_dn = out.get("m7_quant_dn", pd.Series(0.0, index=out.index)).to_numpy(dtype=np.float64)
    q_up = out.get("m7_quant_up", pd.Series(0.0, index=out.index)).to_numpy(dtype=np.float64)
    m_dn = out.get("m7_mtl_dn", pd.Series(0.0, index=out.index)).to_numpy(dtype=np.float64)
    m_up = out.get("m7_mtl_up", pd.Series(0.0, index=out.index)).to_numpy(dtype=np.float64)
    mode_prob = []
    mode_spread = []
    entropy = []
    for a, b in zip(m_dn, m_up):
        arr = sorted([max(a, 0.0), max(b, 0.0)], reverse=True)
        s = max(sum(arr), 1e-12)
        top1 = arr[0] / s
        top2 = arr[1] / s
        mode_prob.append(top1)
        mode_spread.append(top1 - top2)
        entropy.append(_entropy_2(a, b))
    out["mode_prob"] = mode_prob
    out["mode_spread"] = mode_spread
    out["entropy"] = entropy
    out["tail_up_prob"] = np.clip(np.maximum(q_up, m_up), 0.0, 1.0)
    out["tail_down_prob"] = np.clip(np.maximum(q_dn, m_dn), 0.0, 1.0)

    out["target_gap"] = out.get("m7_expected_ret", pd.Series(0.0, index=out.index)).to_numpy(dtype=np.float64)
    out["target_gap_delta_1m"] = out["target_gap"].diff().fillna(0.0)
    out["prob_mom_1m"] = pd.Series(out["mode_prob"]).diff().fillna(0.0)

    return out


def _unrealized(pos: str | None, entry: float, close: float, lev: float, slip: float) -> float:
    if pos is None or entry <= 0 or close <= 0 or lev <= 0:
        return 0.0
    if pos == "LONG":
        raw = (close * (1.0 - slip) - entry) / entry
    else:
        raw = (entry - close * (1.0 + slip)) / entry
    return float(raw * lev)


def _realized(pos: str, entry: float, exit_base: float, lev: float, fee: float, slip: float) -> float:
    if pos == "LONG":
        raw = (exit_base * (1.0 - slip) - entry) / entry
    else:
        raw = (entry - exit_base * (1.0 + slip)) / entry
    return float((raw * lev) - (2.0 * fee * lev))


def _close_trade(eq: float, realized: float, trades: int, wins: int, hold_closed: list[int], hold_count: int):
    eq *= 1.0 + realized
    trades += 1
    if realized > 0:
        wins += 1
    hold_closed.append(hold_count)
    return eq, trades, wins


def simulate_closed_loop(df: pd.DataFrame, actor: GaussianActor, device: str, fee: float, slip: float, cfg: PlaybookMetaConfig | None = None) -> dict:
    router = DSACRouter(actor, device=device)
    n = len(df)
    pos: str | None = None
    entry_price = 0.0
    cur_lev = 0.0
    hold_count = 0
    pos_peak_eq = 1.0
    entry_hold_limit = 0

    pending_side: str | None = None
    pending_lev = 0.0
    pending_delay = 0

    eq = 1.0
    eq_curve = [1.0]
    trades = 0
    wins = 0
    hold_closed: list[int] = []

    skip_count = delayed_count = reduced_count = boosted_count = capped_count = meta_exit_count = 0

    iterator = range(0, n - 1)
    if tqdm is not None:
        iterator = tqdm(iterator, total=n - 1, desc=("baseline-2026" if cfg is None else f"meta-{cfg.name}"), ncols=110)

    for i in iterator:
        row = df.iloc[i]
        next_open = float(df.iloc[i + 1]["open"])
        cp = float(row["close"])

        if pos is not None:
            hold_count += 1
            unr = _unrealized(pos, entry_price, cp, cur_lev, slip)
            pos_peak_eq = max(pos_peak_eq, eq * (1.0 + unr))
            cur_mdd = (eq * (1.0 + unr)) / max(pos_peak_eq, 1e-12) - 1.0
        else:
            unr = 0.0
            cur_mdd = 0.0

        pos_dict = {
            "type": pos,
            "hold_count": hold_count,
            "unrealized": unr,
            "mdd": cur_mdd,
            "margin_usage": cur_lev,
        }
        action_int, leverage, _ = router.decide(row.to_dict(), pos_dict)
        desired_side = None if action_int == 0 else ("LONG" if action_int == 1 else "SHORT")

        if pending_side is not None and pos is None:
            pending_delay -= 1
            if pending_delay <= 0:
                if desired_side == pending_side:
                    pos = pending_side
                    entry_price = next_open * (1.0 + slip if pos == "LONG" else 1.0 - slip)
                    cur_lev = pending_lev
                    hold_count = 0
                    pos_peak_eq = eq
                pending_side = None
                pending_lev = 0.0

        if pos is not None and cfg is not None:
            ctl_live = compute_playbook_meta_controller(pos, row, cfg=cfg)
            live_limit = min(entry_hold_limit if entry_hold_limit > 0 else 10**9, int(ctl_live["max_hold_bars"]))
            if hold_count >= live_limit or float(ctl_live["exit_danger"]) >= float(ctl_live["exit_trigger"]):
                realized = _realized(pos, entry_price, next_open, cur_lev, fee, slip)
                eq, trades, wins = _close_trade(eq, realized, trades, wins, hold_closed, hold_count)
                eq_curve.append(eq)
                pos = None
                entry_price = 0.0
                cur_lev = 0.0
                hold_count = 0
                entry_hold_limit = 0
                meta_exit_count += 1
                continue

        if pos is not None:
            if action_int == 0:
                realized = _realized(pos, entry_price, next_open, cur_lev, fee, slip)
                eq, trades, wins = _close_trade(eq, realized, trades, wins, hold_closed, hold_count)
                eq_curve.append(eq)
                pos = None
                entry_price = 0.0
                cur_lev = 0.0
                hold_count = 0
                entry_hold_limit = 0
            else:
                eq_curve.append(eq * (1.0 + unr))
            continue

        if pos is None and pending_side is None and desired_side in {"LONG", "SHORT"} and leverage > 0:
            if cfg is None:
                pos = desired_side
                entry_price = next_open * (1.0 + slip if pos == "LONG" else 1.0 - slip)
                cur_lev = float(leverage)
                hold_count = 0
                pos_peak_eq = eq
            else:
                ctl = compute_playbook_meta_controller(desired_side, row, cfg=cfg)
                if bool(ctl["skip_entry"]):
                    skip_count += 1
                else:
                    adj_lev = float(leverage) * float(ctl["size_mult"])
                    if ctl["size_mult"] > 1.001:
                        boosted_count += 1
                    elif ctl["size_mult"] < 0.999:
                        reduced_count += 1
                    entry_hold_limit = int(ctl["max_hold_bars"])
                    if entry_hold_limit < 180:
                        capped_count += 1
                    delay = int(ctl["delay_bars"])
                    if delay > 0:
                        pending_side = desired_side
                        pending_lev = adj_lev
                        pending_delay = delay
                        delayed_count += 1
                    else:
                        pos = desired_side
                        entry_price = next_open * (1.0 + slip if pos == "LONG" else 1.0 - slip)
                        cur_lev = adj_lev
                        hold_count = 0
                        pos_peak_eq = eq
            eq_curve.append(eq)
            continue

        eq_curve.append(eq)

    if pos is not None:
        final_close = float(df.iloc[-1]["close"])
        realized = _realized(pos, entry_price, final_close, cur_lev, fee, slip)
        eq, trades, wins = _close_trade(eq, realized, trades, wins, hold_closed, hold_count)
        eq_curve.append(eq)

    rets = np.diff(np.asarray(eq_curve, dtype=np.float64)) / np.maximum(np.asarray(eq_curve[:-1], dtype=np.float64), 1e-12)
    peak = np.maximum.accumulate(np.asarray(eq_curve, dtype=np.float64))
    dd = np.asarray(eq_curve, dtype=np.float64) / np.maximum(peak, 1e-12) - 1.0
    sharpe = float(rets.mean() / (rets.std() + 1e-12) * ANNUAL_FACTOR_5M) if len(rets) > 2 else 0.0
    return {
        "pnl_pct": float(eq - 1.0) * 100.0,
        "mdd_pct": float(dd.min()) * 100.0,
        "sharpe": sharpe,
        "trades": int(trades),
        "wr_pct": float(wins / max(trades, 1)) * 100.0,
        "avg_hold_bars": float(np.mean(hold_closed)) if hold_closed else 0.0,
        "meta": {
            "skipped": int(skip_count),
            "delayed": int(delayed_count),
            "reduced": int(reduced_count),
            "boosted": int(boosted_count),
            "hold_capped": int(capped_count),
            "meta_exits": int(meta_exit_count),
        },
    }


def _coarse_grid() -> list[PlaybookMetaConfig]:
    return [
        PlaybookMetaConfig(event_k=0.90, hazard_k=1.05, continuation_k=0.95, pullback_k=1.00, size_boost=0.04, size_floor=0.88, hold_base_bars=360, hold_scale=0.06, exit_aggr=0.90, skip_hazard_th=0.94, sparse_event_th=0.84, sparse_hazard_th=0.82, severe_exit_th=0.94, mild_reduce_th=0.88),
        PlaybookMetaConfig(event_k=0.95, hazard_k=1.10, continuation_k=0.95, pullback_k=1.05, size_boost=0.05, size_floor=0.90, hold_base_bars=300, hold_scale=0.07, exit_aggr=0.95, skip_hazard_th=0.93, sparse_event_th=0.82, sparse_hazard_th=0.80, severe_exit_th=0.93, mild_reduce_th=0.87),
        PlaybookMetaConfig(event_k=1.00, hazard_k=1.15, continuation_k=1.00, pullback_k=1.05, size_boost=0.06, size_floor=0.92, hold_base_bars=240, hold_scale=0.08, exit_aggr=1.00, skip_hazard_th=0.92, sparse_event_th=0.80, sparse_hazard_th=0.78, severe_exit_th=0.92, mild_reduce_th=0.86),
        PlaybookMetaConfig(event_k=1.05, hazard_k=1.20, continuation_k=1.00, pullback_k=1.10, size_boost=0.06, size_floor=0.90, hold_base_bars=240, hold_scale=0.08, exit_aggr=1.05, skip_hazard_th=0.90, sparse_event_th=0.78, sparse_hazard_th=0.76, severe_exit_th=0.91, mild_reduce_th=0.85),
        PlaybookMetaConfig(event_k=0.98, hazard_k=1.08, continuation_k=0.95, pullback_k=1.00, size_boost=0.04, size_floor=0.94, hold_base_bars=420, hold_scale=0.05, exit_aggr=0.90, skip_hazard_th=0.95, sparse_event_th=0.86, sparse_hazard_th=0.84, severe_exit_th=0.95, mild_reduce_th=0.90),
    ]


def _refine_grid(best: PlaybookMetaConfig) -> list[PlaybookMetaConfig]:
    grid: list[PlaybookMetaConfig] = []
    for dh in (-0.06, 0.0, 0.06):
        for ds in (-0.01, 0.0, 0.01):
            for floor in (max(0.82, best.size_floor - 0.03), best.size_floor, min(0.97, best.size_floor + 0.03)):
                for hold in (max(180, best.hold_base_bars - 60), best.hold_base_bars, best.hold_base_bars + 60):
                    for se in (max(0.76, best.sparse_event_th - 0.02), best.sparse_event_th, min(0.90, best.sparse_event_th + 0.02)):
                        grid.append(
                            replace(
                                best,
                                hazard_k=round(best.hazard_k + dh, 3),
                                size_boost=round(_clip(best.size_boost + ds, 0.03, 0.08), 3),
                                size_floor=round(floor, 3),
                                hold_base_bars=int(hold),
                                sparse_event_th=round(se, 3),
                            )
                        )
    uniq = {cfg.name: cfg for cfg in grid}
    return list(uniq.values())


def _run_grid(df: pd.DataFrame, actor: GaussianActor, device: str, fee: float, slip: float, grid: list[PlaybookMetaConfig], baseline: dict, desc: str) -> list[dict]:
    results = []
    iterator = grid
    if tqdm is not None:
        iterator = tqdm(grid, desc=desc, ncols=110)
    for cfg in iterator:
        res = simulate_closed_loop(df, actor, device, fee, slip, cfg=cfg)
        res["config"] = asdict(cfg)
        res["name"] = cfg.name
        res["baseline"] = baseline
        res["delta_pct"] = float(res["pnl_pct"] - baseline["pnl_pct"])
        res["objective"] = (
            float(res["pnl_pct"] - baseline["pnl_pct"])
            + 0.30 * float(res["mdd_pct"] - baseline["mdd_pct"])
            + 0.10 * float(res["sharpe"] - baseline["sharpe"])
        )
        results.append(res)
    results.sort(key=lambda x: (x["objective"], x["delta_pct"], x["pnl_pct"], x["mdd_pct"], x["sharpe"]), reverse=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Rough then refined 2026 OOS backtest for playbook meta controller.")
    ap.add_argument("--rl-csv", default="data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv")
    ap.add_argument("--ckpt", default="data/ensemble/ckpt/best_dsac_agents.pth")
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.0002)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-json", default="data/ensemble/reports/playbook_meta_controller_2026_full.json")
    args = ap.parse_args()

    device = _resolve_device(args.device)
    df = _load_frame(args.rl_csv)
    df = build_proxy_frame(df)
    actor = _load_actor(args.ckpt, device)

    baseline = simulate_closed_loop(df, actor, device, args.fee, args.slip, cfg=None)
    coarse = _run_grid(df, actor, device, args.fee, args.slip, _coarse_grid(), baseline, desc="playbook2026-coarse")
    refine = _run_grid(df, actor, device, args.fee, args.slip, _refine_grid(PlaybookMetaConfig(**coarse[0]["config"])), baseline, desc="playbook2026-refine")

    summary = {
        "dataset": {
            "rl_csv": args.rl_csv,
            "rows": int(len(df)),
            "start": str(df["timestamp"].iloc[0]),
            "end": str(df["timestamp"].iloc[-1]),
            "note": "Polymarket live fields are unavailable in 2026 OOS csv, so M7 distribution + event/tail features are used as proxies.",
        },
        "baseline": baseline,
        "coarse_top3": coarse[:3],
        "refine_top5": refine[:5],
        "best_overall": refine[0],
        "all_coarse": coarse,
        "all_refine": refine,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    best = refine[0]
    print("=== Playbook Meta Controller 2026 Full OOS ===")
    print(f"rows={summary['dataset']['rows']} period={summary['dataset']['start']} -> {summary['dataset']['end']}")
    print(
        f"baseline={baseline['pnl_pct']:+.4f}% best={best['pnl_pct']:+.4f}% "
        f"delta={best['delta_pct']:+.4f}%p mdd={best['mdd_pct']:+.4f}% sharpe={best['sharpe']:.4f}"
    )


if __name__ == "__main__":
    main()
