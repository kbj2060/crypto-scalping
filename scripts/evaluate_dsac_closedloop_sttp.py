#!/usr/bin/env python3
"""Closed-loop DSAC evaluation with optional ST/TP overlay.

Key differences vs open-loop A/B:
- Policy is queried every bar from the latest simulated position state.
- ST/TP exits are applied inside the same simulation loop.
- OOS A/B comparison uses the exact same closed-loop engine.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from ensemble.train_rl_dsac_agent import DSACRouter, DSAC_STATE_DIM, GaussianActor

ANNUAL_FACTOR_5M = math.sqrt(365 * 24 * 12)


@dataclass
class STTPParams:
    tp_mult: float
    sl_mult: float
    max_hold: int


@dataclass
class SimMetrics:
    pnl_pct: float
    mdd_pct: float
    sharpe: float
    trades: int
    wr_pct: float
    avg_hold_bars: float
    median_hold_bars: float
    tp_hits: int
    sl_hits: int
    timeout_hits: int
    score: float


def _resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _parse_list(s: str, cast):
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(cast(tok))
    return out


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return x


def _load_merged_frame(rl_csv: str, feat_csv: str) -> pd.DataFrame:
    if not os.path.exists(rl_csv):
        raise FileNotFoundError(f"rl csv not found: {rl_csv}")
    if not os.path.exists(feat_csv):
        raise FileNotFoundError(f"feature csv not found: {feat_csv}")

    rdf = pd.read_csv(rl_csv)
    fdf = pd.read_csv(feat_csv, usecols=["timestamp", "open", "high", "low"])

    rdf["timestamp"] = pd.to_datetime(rdf["timestamp"], errors="coerce")
    fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], errors="coerce")

    rdf = rdf.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
    fdf = fdf.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")

    merged = rdf.merge(fdf, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    for c in ("close", "open", "high", "low"):
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=["timestamp", "close", "open", "high", "low"]).reset_index(drop=True)

    hi = np.maximum(merged["high"].to_numpy(dtype=np.float64), merged["low"].to_numpy(dtype=np.float64))
    lo = np.minimum(merged["high"].to_numpy(dtype=np.float64), merged["low"].to_numpy(dtype=np.float64))
    merged["high"] = hi
    merged["low"] = lo
    merged["close"] = np.maximum(merged["close"].to_numpy(dtype=np.float64), 1e-8)
    merged["open"] = np.maximum(merged["open"].to_numpy(dtype=np.float64), 1e-8)
    return merged


def _load_actor(ckpt_path: str, device: str) -> GaussianActor:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "actor" not in ckpt:
        raise KeyError(f"'actor' key missing in checkpoint: {ckpt_path}")
    state_dim = int(ckpt.get("state_dim", DSAC_STATE_DIM))
    actor = GaussianActor(state_dim=state_dim).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


def _vol_proxy(close_np: np.ndarray, span: int = 48) -> np.ndarray:
    lr = np.zeros_like(close_np, dtype=np.float64)
    if len(close_np) > 1:
        c = np.maximum(close_np, 1e-8)
        lr[1:] = np.diff(np.log(c))
    v = pd.Series(lr).ewm(span=max(4, int(span)), adjust=False).std(bias=False).fillna(0.0).to_numpy(dtype=np.float64)
    return np.maximum(v, 1e-8)


def _unrealized(pos: str | None, entry: float, close: float, lev: float, slip: float) -> float:
    if pos is None or entry <= 0 or close <= 0 or lev <= 0:
        return 0.0
    if pos == "LONG":
        raw = (close * (1.0 - slip) - entry) / entry
    else:
        raw = (entry - close * (1.0 + slip)) / entry
    return float(raw * lev)


def _realized(pos: str, entry: float, exit_base: float, lev: float, slip: float) -> float:
    if pos == "LONG":
        raw = (exit_base * (1.0 - slip) - entry) / entry
    else:
        raw = (entry - exit_base * (1.0 + slip)) / entry
    return float(raw * lev)


def _risk_score(m: SimMetrics) -> float:
    return float(m.pnl_pct + 1.5 * m.sharpe - 1.3 * abs(m.mdd_pct) - 0.01 * m.trades)


def _simulate_closed_loop(
    df: pd.DataFrame,
    actor: GaussianActor,
    device: str,
    fee: float,
    slip: float,
    sttp: STTPParams | None = None,
    vol_span: int = 48,
    tp_floor: float = 0.0030,
    sl_floor: float = 0.0020,
) -> SimMetrics:
    n = len(df)
    if n < 3:
        raise RuntimeError(f"not enough rows: {n}")

    router = DSACRouter(actor, device=device)

    numeric_cols = [c for c in df.columns if c != "timestamp"]
    values = df[numeric_cols].to_numpy(dtype=np.float64)
    open_np = df["open"].to_numpy(dtype=np.float64)
    high_np = df["high"].to_numpy(dtype=np.float64)
    low_np = df["low"].to_numpy(dtype=np.float64)
    close_np = df["close"].to_numpy(dtype=np.float64)
    vol_np = _vol_proxy(close_np, span=vol_span)

    balance = 1.0
    pos: str | None = None
    entry_price = 0.0
    cur_lev = 0.0
    hold_count = 0
    pos_peak_eq = 1.0

    trades = 0
    wins = 0
    hold_closed: list[int] = []
    tp_hits = sl_hits = timeout_hits = 0

    eq_curve = [1.0]

    for i in range(0, n - 1):
        cp = float(close_np[i])
        hp = float(high_np[i])
        lp = float(low_np[i])
        next_open = float(open_np[i + 1])
        next_close = float(close_np[i + 1])
        vv = max(float(vol_np[i]), 1e-8)

        if pos is not None:
            hold_count += 1

        # 1) ST/TP + timeout first (closed-loop, conservative intrabar)
        if sttp is not None and pos is not None and entry_price > 0 and cur_lev > 0:
            tp_pct = max(float(tp_floor), float(sttp.tp_mult) * vv)
            sl_pct = max(float(sl_floor), float(sttp.sl_mult) * vv)

            hit_tp = False
            hit_sl = False
            tp_level = sl_level = 0.0
            if pos == "LONG":
                tp_level = entry_price * (1.0 + tp_pct)
                sl_level = entry_price * (1.0 - sl_pct)
                hit_tp = hp >= tp_level
                hit_sl = lp <= sl_level
            else:
                tp_level = entry_price * (1.0 - tp_pct)
                sl_level = entry_price * (1.0 + sl_pct)
                hit_tp = lp <= tp_level
                hit_sl = hp >= sl_level

            sttp_exit = False
            exit_price = cp
            if hit_tp and hit_sl:
                # conservative tie-breaker
                sttp_exit = True
                exit_price = sl_level
                sl_hits += 1
            elif hit_sl:
                sttp_exit = True
                exit_price = sl_level
                sl_hits += 1
            elif hit_tp:
                sttp_exit = True
                exit_price = tp_level
                tp_hits += 1

            if sttp_exit:
                base = balance
                realized = _realized(pos, entry_price, exit_price, cur_lev, slip)
                balance = base * (1.0 + realized)
                balance -= base * fee * cur_lev
                trades += 1
                if realized > 0:
                    wins += 1
                hold_closed.append(max(1, hold_count))
                pos = None
                entry_price = 0.0
                cur_lev = 0.0
                hold_count = 0
                pos_peak_eq = 1.0

            elif hold_count >= int(sttp.max_hold):
                base = balance
                realized = _realized(pos, entry_price, cp, cur_lev, slip)
                balance = base * (1.0 + realized)
                balance -= base * fee * cur_lev
                trades += 1
                if realized > 0:
                    wins += 1
                timeout_hits += 1
                hold_closed.append(max(1, hold_count))
                pos = None
                entry_price = 0.0
                cur_lev = 0.0
                hold_count = 0
                pos_peak_eq = 1.0

        # 2) Build closed-loop policy input from latest simulated position state
        if pos is not None and entry_price > 0 and cur_lev > 0:
            unr = _unrealized(pos, entry_price, cp, cur_lev, slip)
            cur_eq = 1.0 + unr
            pos_peak_eq = max(pos_peak_eq, cur_eq)
            mdd = float(np.clip(min((cur_eq / max(pos_peak_eq, 1e-8)) - 1.0, 0.0) / 0.05, -1.0, 1.0))
        else:
            unr = 0.0
            mdd = 0.0
            pos_peak_eq = 1.0

        pos_dict = {
            "type": pos,
            "entry_price": float(entry_price),
            "unrealized": float(unr),
            "mdd": float(mdd),
            "hold_norm": float(min(hold_count / 96.0, 1.0)),
            "margin_usage": float(cur_lev if pos is not None else 0.0),
        }
        row = values[i]
        features = {k: float(v) for k, v in zip(numeric_cols, row)}
        action_int, lev, _ = router.decide(features, pos_dict)
        lev = float(np.clip(lev, 0.0, 1.0))

        # 3) Execute policy at next bar open (same semantic as training env)
        if pos is None:
            if action_int == 1 and lev > 0.0:
                pos = "LONG"
                entry_price = next_open * (1.0 + slip)
                cur_lev = lev
                hold_count = 0
                pos_peak_eq = 1.0
                balance -= balance * fee * cur_lev
            elif action_int == 2 and lev > 0.0:
                pos = "SHORT"
                entry_price = next_open * (1.0 - slip)
                cur_lev = lev
                hold_count = 0
                pos_peak_eq = 1.0
                balance -= balance * fee * cur_lev
        else:
            should_close = (action_int == 0) or (action_int == 1 and pos == "SHORT") or (action_int == 2 and pos == "LONG")
            if should_close:
                base = balance
                realized = _realized(pos, entry_price, next_open, cur_lev, slip)
                balance = base * (1.0 + realized)
                balance -= base * fee * cur_lev
                trades += 1
                if realized > 0:
                    wins += 1
                hold_closed.append(max(1, hold_count))
                pos = None
                entry_price = 0.0
                cur_lev = 0.0
                hold_count = 0
                pos_peak_eq = 1.0
            else:
                lev_delta = abs(lev - cur_lev)
                if lev_delta > 0.05:
                    balance -= balance * fee * lev_delta
                    cur_lev = lev

        # 4) Mark-to-market at next close
        if pos is not None:
            eq = balance * (1.0 + _unrealized(pos, entry_price, next_close, cur_lev, slip))
        else:
            eq = balance
        eq_curve.append(max(float(eq), 1e-8))

    # terminal close
    if pos is not None and entry_price > 0 and cur_lev > 0:
        base = balance
        realized = _realized(pos, entry_price, float(close_np[-1]), cur_lev, slip)
        balance = base * (1.0 + realized)
        balance -= base * fee * cur_lev
        trades += 1
        if realized > 0:
            wins += 1
        hold_closed.append(max(1, hold_count))
        eq_curve[-1] = max(balance, 1e-8)

    eq = np.asarray(eq_curve, dtype=np.float64)
    run_max = np.maximum.accumulate(eq)
    dd = eq / np.maximum(run_max, 1e-12) - 1.0
    mdd = float(np.min(dd)) if len(dd) else 0.0

    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    if len(rets) < 3 or np.std(rets) < 1e-12:
        sharpe = 0.0
    else:
        sharpe = float(np.mean(rets) / np.std(rets) * ANNUAL_FACTOR_5M)

    wr = float(wins / trades) if trades > 0 else 0.0
    avg_hold = float(np.mean(hold_closed)) if hold_closed else 0.0
    med_hold = float(np.median(hold_closed)) if hold_closed else 0.0
    pnl_pct = float((balance - 1.0) * 100.0)

    out = SimMetrics(
        pnl_pct=pnl_pct,
        mdd_pct=float(mdd * 100.0),
        sharpe=sharpe,
        trades=int(trades),
        wr_pct=float(wr * 100.0),
        avg_hold_bars=avg_hold,
        median_hold_bars=med_hold,
        tp_hits=int(tp_hits),
        sl_hits=int(sl_hits),
        timeout_hits=int(timeout_hits),
        score=0.0,
    )
    out.score = _risk_score(out)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Closed-loop DSAC A/B evaluation with ST/TP")
    p.add_argument("--rl-csv", default="data/rl_training_data_full.csv")
    p.add_argument("--feature-csv", default="data/training_features_5m.csv")
    p.add_argument("--ckpt-path", default="data/ensemble/ckpt/best_dsac_agents.pth")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--vol-span", type=int, default=48)
    p.add_argument("--tp-floor", type=float, default=0.0030)
    p.add_argument("--sl-floor", type=float, default=0.0020)
    p.add_argument("--tp-mults", default="2.0,2.5,3.0")
    p.add_argument("--sl-mults", default="1.0,1.2,1.5")
    p.add_argument("--max-holds", default="48,72")
    p.add_argument("--calib-year", type=int, default=2025)
    p.add_argument("--test-start", default="2026-01-01")
    p.add_argument("--test-end", default="2026-02-28 23:59:59")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--out-json", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = _resolve_device(args.device)

    df = _load_merged_frame(args.rl_csv, args.feature_csv)
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.assign(timestamp=ts).dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    calib_mask = df["timestamp"].dt.year == int(args.calib_year)
    test_mask = (df["timestamp"] >= pd.Timestamp(args.test_start)) & (df["timestamp"] <= pd.Timestamp(args.test_end))
    df_calib = df.loc[calib_mask].reset_index(drop=True)
    df_test = df.loc[test_mask].reset_index(drop=True)
    if len(df_calib) < 1000:
        raise RuntimeError(f"calibration split too small: {len(df_calib)}")
    if len(df_test) < 1000:
        raise RuntimeError(f"test split too small: {len(df_test)}")

    actor = _load_actor(args.ckpt_path, device=device)
    print(f"[DATA] merged_rows={len(df):,} calib_rows={len(df_calib):,} test_rows={len(df_test):,}")
    print(f"[DATA] calib_year={args.calib_year} test={args.test_start} ~ {args.test_end}")
    print(f"[MODEL] ckpt={args.ckpt_path} device={device}")
    print("[MODE] closed-loop evaluation (policy queried every bar)")

    base_calib = _simulate_closed_loop(
        df_calib,
        actor=actor,
        device=device,
        fee=float(args.fee),
        slip=float(args.slip),
        sttp=None,
        vol_span=int(args.vol_span),
        tp_floor=float(args.tp_floor),
        sl_floor=float(args.sl_floor),
    )

    tp_mults = _parse_list(args.tp_mults, float)
    sl_mults = _parse_list(args.sl_mults, float)
    max_holds = _parse_list(args.max_holds, int)
    if not tp_mults or not sl_mults or not max_holds:
        raise ValueError("tp/sl/max_hold grid must be non-empty")

    total = len(tp_mults) * len(sl_mults) * len(max_holds)
    print(f"[STEP] tuning ST/TP grid on calibration split... combos={total}")

    best_params: STTPParams | None = None
    best_metric: SimMetrics | None = None
    leaderboard: list[dict[str, Any]] = []
    k = 0
    for tp_m, sl_m, mh in itertools.product(tp_mults, sl_mults, max_holds):
        k += 1
        params = STTPParams(tp_mult=float(tp_m), sl_mult=float(sl_m), max_hold=int(mh))
        met = _simulate_closed_loop(
            df_calib,
            actor=actor,
            device=device,
            fee=float(args.fee),
            slip=float(args.slip),
            sttp=params,
            vol_span=int(args.vol_span),
            tp_floor=float(args.tp_floor),
            sl_floor=float(args.sl_floor),
        )
        row = {"tp_mult": params.tp_mult, "sl_mult": params.sl_mult, "max_hold": params.max_hold, **asdict(met)}
        leaderboard.append(row)
        if (best_metric is None) or (met.score > best_metric.score):
            best_metric = met
            best_params = params
        if k % max(1, total // 8) == 0 or k == total:
            assert best_metric is not None
            print(
                f"  [{k:>3}/{total}] best score={best_metric.score:.3f} "
                f"pnl={best_metric.pnl_pct:.2f}% mdd={best_metric.mdd_pct:.2f}% "
                f"sharpe={best_metric.sharpe:.2f} tr={best_metric.trades}"
            )

    assert best_params is not None and best_metric is not None
    leaderboard = sorted(leaderboard, key=lambda x: float(x["score"]), reverse=True)
    topk = leaderboard[: max(1, int(args.topk))]

    A = _simulate_closed_loop(
        df_test,
        actor=actor,
        device=device,
        fee=float(args.fee),
        slip=float(args.slip),
        sttp=None,
        vol_span=int(args.vol_span),
        tp_floor=float(args.tp_floor),
        sl_floor=float(args.sl_floor),
    )
    B = _simulate_closed_loop(
        df_test,
        actor=actor,
        device=device,
        fee=float(args.fee),
        slip=float(args.slip),
        sttp=best_params,
        vol_span=int(args.vol_span),
        tp_floor=float(args.tp_floor),
        sl_floor=float(args.sl_floor),
    )

    delta = {
        "pnl_pct": float(B.pnl_pct - A.pnl_pct),
        "mdd_pct": float(B.mdd_pct - A.mdd_pct),
        "sharpe": float(B.sharpe - A.sharpe),
        "trades": int(B.trades - A.trades),
        "wr_pct": float(B.wr_pct - A.wr_pct),
        "avg_hold_bars": float(B.avg_hold_bars - A.avg_hold_bars),
        "median_hold_bars": float(B.median_hold_bars - A.median_hold_bars),
        "tp_hits": int(B.tp_hits - A.tp_hits),
        "sl_hits": int(B.sl_hits - A.sl_hits),
        "timeout_hits": int(B.timeout_hits - A.timeout_hits),
        "score": float(B.score - A.score),
    }

    out = {
        "config": {
            "rl_csv": args.rl_csv,
            "feature_csv": args.feature_csv,
            "ckpt_path": args.ckpt_path,
            "device": device,
            "fee": float(args.fee),
            "slip": float(args.slip),
            "vol_span": int(args.vol_span),
            "tp_floor": float(args.tp_floor),
            "sl_floor": float(args.sl_floor),
            "grid": {"tp_mults": tp_mults, "sl_mults": sl_mults, "max_holds": max_holds},
            "calib_year": int(args.calib_year),
            "test_start": args.test_start,
            "test_end": args.test_end,
            "rows": {"merged": int(len(df)), "calibration": int(len(df_calib)), "test": int(len(df_test))},
            "note": "Closed-loop evaluation: policy is queried each bar; ST/TP exits are applied in-loop.",
        },
        "selected_sttp_params": asdict(best_params),
        "calibration_baseline_A": asdict(base_calib),
        "calibration_best_B": asdict(best_metric),
        "calibration_topk": topk,
        "oos_A_dsac_only": asdict(A),
        "oos_B_dsac_plus_sttp": asdict(B),
        "oos_delta_B_minus_A": delta,
    }

    out_json = args.out_json.strip()
    if not out_json:
        os.makedirs("data/ensemble/metrics", exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_json = f"data/ensemble/metrics/closedloop_dsac_sttp_{stamp}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n== OOS RESULT (closed-loop, 2026-01~02) ==")
    print(
        "A DSAC-only | pnl={:.2f}% mdd={:.2f}% sharpe={:.2f} tr={} wr={:.2f}% hold={:.2f}/{:.2f}".format(
            A.pnl_pct, A.mdd_pct, A.sharpe, A.trades, A.wr_pct, A.avg_hold_bars, A.median_hold_bars
        )
    )
    print(
        "B DSAC+STTP | pnl={:.2f}% mdd={:.2f}% sharpe={:.2f} tr={} wr={:.2f}% hold={:.2f}/{:.2f} tp/sl/to={}/{}/{}".format(
            B.pnl_pct,
            B.mdd_pct,
            B.sharpe,
            B.trades,
            B.wr_pct,
            B.avg_hold_bars,
            B.median_hold_bars,
            B.tp_hits,
            B.sl_hits,
            B.timeout_hits,
        )
    )
    print(
        "DELTA(B-A)  | pnl={:+.2f}% mdd={:+.2f}% sharpe={:+.2f} tr={:+d} wr={:+.2f}% hold={:+.2f}/{:+.2f}".format(
            delta["pnl_pct"],
            delta["mdd_pct"],
            delta["sharpe"],
            delta["trades"],
            delta["wr_pct"],
            delta["avg_hold_bars"],
            delta["median_hold_bars"],
        )
    )
    print(f"selected STTP params: {best_params}")
    print(f"saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

