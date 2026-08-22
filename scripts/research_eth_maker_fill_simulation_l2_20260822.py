#!/usr/bin/env python3
"""ETH maker 체결 시뮬레이션 — 실효 비용(bp/leg) 계측 (2026-08-22).

목적: RDE(레짐 직결 노출) 정책의 breakeven(3.6~3.7bp/leg, taker 7bp에서 사망)을 maker
집행으로 넘을 수 있는지 — passive limit 진입의 실효 비용을 raw L2 + aggTrades로 계측한다.
(배경: docs/experiments/eth_ilias_regime_direct_exposure_seed_stable_direction_20260822.md)

데이터:
- WS-E 격리 파일럿 raw L2: data/research/ws_e_orderbook_raw_pilot.duckdb
  (orderbook_periodic_snapshots_eth_soak_20260719, 상위20레벨, ~10초 연속, 53h)
- Binance aggTrades (data.binance.vision, futures/um/daily, 2026-07-19~21)

체결 규칙(보수적 — Huang/Lehalle/Rosenbaum 2015 큐-리액티브 프레임의 worst-case 단순화):
- 내 가격 p를 '뚫는' 반대측 aggressor 체결(price < p for buy) → 즉시 체결.
- 내 가격 p '에서의' aggressor 체결 → 앞선 큐(진입 시점 해당 레벨 표시 수량)를 먼저 소진한
  뒤에만 체결. 큐 감소는 체결로만 인정(앞선 취소 무시 → 큐 과대추정 = 보수적).
- 호가 크로스(best_ask ≤ p) → 보장 체결(교차 호가는 존재 불가).
- 지연 마진: 스냅샷 시각 +200ms 이후 체결만 인정.
정책: 도착 시 best bid에 join. "static"(가격 고정) / "peg"(시장이 위로 가면 새 best bid로
재호가, 큐 리셋). 타임아웃 시 그 시점 best ask로 taker 폴백.
비용 = (체결가 − 도착시점 mid)/mid + 수수료(maker 2bp / taker 5bp). 폴백 드리프트 포함.

한계(정직성): 53h 단일 창(2026-07-19~21, 저변동 구간일 수 있음), 자기 주문의 시장 영향
무시(소액 가정), 아이스버그/숨은 유동성 무시(체결 과소 = 보수적), 큐 앞 취소 무시(보수적).
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# aggTrades 원본: https://data.binance.vision/data/futures/um/daily/aggTrades/ETHUSDT/
# ETHUSDT-aggTrades-{YYYY-MM-DD}.zip (무료 아카이브, 재다운로드로 재현 가능)
SCRATCH = Path(__file__).resolve().parents[1] / "tmp/eth_maker_fill_simulation_20260822/aggtrades"
DB = ROOT / "data/research/ws_e_orderbook_raw_pilot.duckdb"
TABLE = "orderbook_periodic_snapshots_eth_soak_20260719"
OUT_DIR = ROOT / "tmp/eth_maker_fill_simulation_20260822"

MAKER_FEE_BP = 2.0
TAKER_FEE_BP = 5.0
LATENCY_MS = 200
ARRIVAL_SPACING_S = 300          # 5분 간격 가상 도착
TIMEOUTS_S = [30, 60, 120, 300]
POLICIES = ["static", "peg"]


def load_snapshots() -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        f"select exchange_timestamp, best_bid, best_ask, mid, bids_json, asks_json "
        f"from {TABLE} order by exchange_timestamp"
    ).df()
    con.close()
    df["bids"] = df["bids_json"].apply(json.loads)
    df["asks"] = df["asks_json"].apply(json.loads)
    # 정합성: level1 == best 컬럼
    b0 = np.array([b[0][0] for b in df["bids"]])
    a0 = np.array([a[0][0] for a in df["asks"]])
    assert np.allclose(b0, df["best_bid"]), "bids_json[0] != best_bid"
    assert np.allclose(a0, df["best_ask"]), "asks_json[0] != best_ask"
    return df


def load_trades() -> pd.DataFrame:
    parts = []
    for d in ["2026-07-19", "2026-07-20", "2026-07-21"]:
        parts.append(pd.read_csv(SCRATCH / f"ETHUSDT-aggTrades-{d}.csv",
                                 usecols=["price", "quantity", "transact_time", "is_buyer_maker"]))
    tr = pd.concat(parts, ignore_index=True).sort_values("transact_time").reset_index(drop=True)
    return tr


def level_qty(levels: list, price: float) -> float:
    for p, q in levels:
        if abs(p - price) < 1e-9:
            return float(q)
    return 0.0


def simulate_leg(snap: pd.DataFrame, tr_px: np.ndarray, tr_qty: np.ndarray, tr_ts: np.ndarray,
                 tr_sellagg: np.ndarray, i0: int, side: str, timeout_s: int, policy: str) -> dict:
    """도착 스냅샷 i0에서 side('buy'/'sell') 1 leg 실행. 반환: 비용(bp), 체결경로."""
    ts0 = int(snap["exchange_timestamp"].iloc[i0])
    mid0 = float(snap["mid"].iloc[i0])
    deadline = ts0 + timeout_s * 1000
    buy = side == "buy"

    if buy:
        my_px = float(snap["best_bid"].iloc[i0])
        queue = level_qty(snap["bids"].iloc[i0], my_px)
    else:
        my_px = float(snap["best_ask"].iloc[i0])
        queue = level_qty(snap["asks"].iloc[i0], my_px)
    active_from = ts0 + LATENCY_MS

    j = i0
    n = len(snap)
    while True:
        ts_j = int(snap["exchange_timestamp"].iloc[j])
        seg_end = min(int(snap["exchange_timestamp"].iloc[j + 1]) if j + 1 < n else deadline, deadline)
        # 이 구간의 aggressor 체결 처리
        lo = np.searchsorted(tr_ts, max(active_from, ts_j), side="left")
        hi = np.searchsorted(tr_ts, seg_end, side="right")
        for k in range(lo, hi):
            if buy:
                if not tr_sellagg[k]:
                    continue
                if tr_px[k] < my_px - 1e-9:
                    return {"filled": True, "px": my_px, "mode": "trade_through", "t_ms": int(tr_ts[k] - ts0)}
                if abs(tr_px[k] - my_px) < 1e-9:
                    queue -= tr_qty[k]
                    if queue < -1e-9:
                        return {"filled": True, "px": my_px, "mode": "queue_exhaust", "t_ms": int(tr_ts[k] - ts0)}
            else:
                if tr_sellagg[k]:
                    continue
                if tr_px[k] > my_px + 1e-9:
                    return {"filled": True, "px": my_px, "mode": "trade_through", "t_ms": int(tr_ts[k] - ts0)}
                if abs(tr_px[k] - my_px) < 1e-9:
                    queue -= tr_qty[k]
                    if queue < -1e-9:
                        return {"filled": True, "px": my_px, "mode": "queue_exhaust", "t_ms": int(tr_ts[k] - ts0)}
        if seg_end >= deadline or j + 1 >= n:
            break
        j += 1
        # 다음 스냅샷에서 호가크로스/재호가 판정
        bb = float(snap["best_bid"].iloc[j]); ba = float(snap["best_ask"].iloc[j])
        if buy and ba <= my_px + 1e-9:
            return {"filled": True, "px": my_px, "mode": "quote_cross", "t_ms": int(snap["exchange_timestamp"].iloc[j] - ts0)}
        if not buy and bb >= my_px - 1e-9:
            return {"filled": True, "px": my_px, "mode": "quote_cross", "t_ms": int(snap["exchange_timestamp"].iloc[j] - ts0)}
        if policy == "peg":
            if buy and bb > my_px + 1e-9:      # 시장이 위로 → 추격 재호가(큐 리셋)
                my_px = bb
                queue = level_qty(snap["bids"].iloc[j], my_px)
                active_from = int(snap["exchange_timestamp"].iloc[j]) + LATENCY_MS
            elif not buy and ba < my_px - 1e-9:
                my_px = ba
                queue = level_qty(snap["asks"].iloc[j], my_px)
                active_from = int(snap["exchange_timestamp"].iloc[j]) + LATENCY_MS

    # 타임아웃 → taker 폴백(그 시점 반대측 최우선호가)
    jT = j if j < n else n - 1
    fb_px = float(snap["best_ask"].iloc[jT]) if buy else float(snap["best_bid"].iloc[jT])
    return {"filled": False, "px": fb_px, "mode": "taker_fallback", "t_ms": timeout_s * 1000}


def leg_cost_bp(res: dict, mid0: float, buy: bool) -> float:
    sgn = 1.0 if buy else -1.0
    price_bp = sgn * (res["px"] - mid0) / mid0 * 1e4
    fee = MAKER_FEE_BP if res["filled"] else TAKER_FEE_BP
    return price_bp + fee


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snap = load_snapshots()
    tr = load_trades()
    ts = snap["exchange_timestamp"].to_numpy()
    tr_ts = tr["transact_time"].to_numpy()
    tr_px = tr["price"].to_numpy()
    tr_qty = tr["quantity"].to_numpy()
    tr_sellagg = tr["is_buyer_maker"].to_numpy()  # True = 매도 aggressor(매수측 maker)
    print(f"snapshots={len(snap)} trades={len(tr)}", flush=True)

    # 5분 간격 가상 도착(스냅샷 인덱스). 마지막 타임아웃 여유 확보 위해 끝 300s 제외.
    arrivals = []
    t_cursor = ts[0]
    for i in range(len(ts)):
        if ts[i] >= t_cursor:
            arrivals.append(i)
            t_cursor = ts[i] + ARRIVAL_SPACING_S * 1000
    arrivals = [i for i in arrivals if ts[i] <= ts[-1] - 310 * 1000]
    print(f"arrivals={len(arrivals)}", flush=True)

    # 도착 시점 조건부 분석용: 직전 5분 실현변동성
    mid = snap["mid"].to_numpy()
    vol5 = np.full(len(snap), np.nan)
    for i in range(30, len(snap)):
        seg = mid[i - 30:i + 1]
        vol5[i] = np.std(np.diff(np.log(seg))) if len(seg) > 2 else np.nan

    results = {}
    for policy in POLICIES:
        for T in TIMEOUTS_S:
            rows = []
            for i0 in arrivals:
                mid0 = float(snap["mid"].iloc[i0])
                for side in ("buy", "sell"):
                    r = simulate_leg(snap, tr_px, tr_qty, tr_ts, tr_sellagg, i0, side, T, policy)
                    rows.append({
                        "side": side, "filled": r["filled"], "mode": r["mode"],
                        "t_ms": r["t_ms"], "cost_bp": leg_cost_bp(r, mid0, side == "buy"),
                        "vol5": vol5[i0],
                    })
            df = pd.DataFrame(rows)
            key = f"{policy}_T{T}"
            fb = df[~df.filled]
            fl = df[df.filled]
            hi_vol = df[df.vol5 > np.nanquantile(df.vol5, 0.67)]
            results[key] = {
                "n_legs": len(df),
                "fill_rate": float(df.filled.mean()),
                "cost_bp_mean": float(df.cost_bp.mean()),
                "cost_bp_median": float(df.cost_bp.median()),
                "cost_bp_p90": float(df.cost_bp.quantile(0.9)),
                "filled_cost_bp_mean": float(fl.cost_bp.mean()) if len(fl) else None,
                "fallback_cost_bp_mean": float(fb.cost_bp.mean()) if len(fb) else None,
                "fill_mode_counts": df["mode"].value_counts().to_dict(),
                "highvol_tercile_cost_bp_mean": float(hi_vol.cost_bp.mean()),
                "median_fill_time_s": float(fl.t_ms.median() / 1000) if len(fl) else None,
            }
            print(f"{key}: fill={results[key]['fill_rate']:.3f} "
                  f"mean={results[key]['cost_bp_mean']:.2f}bp "
                  f"(filled {results[key]['filled_cost_bp_mean']:.2f} / "
                  f"fallback {results[key]['fallback_cost_bp_mean'] if results[key]['fallback_cost_bp_mean'] is not None else float('nan'):.2f}) "
                  f"hi-vol {results[key]['highvol_tercile_cost_bp_mean']:.2f}bp", flush=True)

    # taker 베이스라인(도착 즉시 크로스): 반스프레드 + taker fee
    taker_rows = []
    for i0 in arrivals:
        mid0 = float(snap["mid"].iloc[i0])
        taker_rows.append((float(snap["best_ask"].iloc[i0]) - mid0) / mid0 * 1e4 + TAKER_FEE_BP)
        taker_rows.append((mid0 - float(snap["best_bid"].iloc[i0])) / mid0 * 1e4 + TAKER_FEE_BP)
    baseline = {"taker_immediate_cost_bp_mean": float(np.mean(taker_rows)),
                "repo_assumed_cost_bp": 7.0}

    report = {
        "experiment": "eth_maker_fill_simulation_l2_20260822",
        "data": {"snapshots": len(snap), "trades": len(tr), "window": "2026-07-19~21 (53h)",
                 "arrivals": len(arrivals), "sides": ["buy", "sell"]},
        "fees_bp": {"maker": MAKER_FEE_BP, "taker": TAKER_FEE_BP},
        "fill_rule": "conservative: trade strictly through OR queue exhausted by trades at price OR quote cross; queue decremented by trades only (cancellations ignored); latency 200ms",
        "baseline": baseline,
        "results": results,
        "rde_breakeven_targets_bp_per_leg": {"episode_hold_region": 3.65, "maker_goal": 2.0},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(f"report -> {OUT_DIR / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
