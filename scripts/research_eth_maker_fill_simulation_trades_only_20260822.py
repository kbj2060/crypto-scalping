#!/usr/bin/env python3
"""ETH maker 체결 시뮬레이션 v2 — aggTrades 단독(고변동 과거일 재계측용) (2026-08-22).

배경: v1(`research_eth_maker_fill_simulation_l2_20260822.py`)은 raw L2 스냅샷이 있는 53h
저변동 창에서만 가능했다. 고변동 구간 재계측을 위해 L2 없이 aggTrades만으로 시뮬한다
(Binance bookTicker 아카이브는 2023년 중단 — 2026년분 없음, S3 리스팅으로 확인).

bid/ask 재구성: is_buyer_maker=True 체결가 = bid, False = ask (최근값, 교차 시 최신측 기준
1틱 클램프). 파일럿 창 검증: 진실 L2 대비 89% 정확일치/92% 1틱 이내/p90 오차 0.053bp,
체결측 trade age 중앙값 0.38s.

체결 규칙(v1보다 한층 더 보수적 — 큐소진 체결 제외):
- buy at p: 매도 aggressor 체결 price < p → 체결. ask측 체결 price ≤ p(=ask가 p 이하) → 체결.
- 내 가격 '에서의' bid측 체결은 무시(큐 정보 없음 → 전부 미체결 처리 = 보수적).
- peg: bid 재구성값이 내 가격 위로 가면 추격 재호가(지연 200ms 재적용).
- 타임아웃 → 그 시점 ask 재구성값으로 taker 폴백.

사용:
  --mode pilot          : 파일럿 창(07-19~21)에서 v1과 교차검증
  --mode days --days 2026-02-03,2026-03-10 : 지정일 측정(aggTrades 자동 다운로드)
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AGG_DIR = ROOT / "tmp/eth_maker_fill_simulation_20260822/aggtrades"
OUT_DIR = ROOT / "tmp/eth_maker_fill_simulation_20260822"

MAKER_FEE_BP = 2.0
TAKER_FEE_BP = 5.0
LATENCY_MS = 200
ARRIVAL_SPACING_MS = 300_000
TIMEOUTS_S = [60, 120, 300]
TICK = 0.01
URL = "https://data.binance.vision/data/futures/um/daily/aggTrades/ETHUSDT/ETHUSDT-aggTrades-{d}.zip"


def ensure_day(d: str) -> Path:
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    out = AGG_DIR / f"ETHUSDT-aggTrades-{d}.csv"
    if out.exists():
        return out
    print(f"downloading {d} ...", flush=True)
    data = urllib.request.urlopen(URL.format(d=d), timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(AGG_DIR)
    assert out.exists()
    return out


def load_days(days: list[str]) -> pd.DataFrame:
    parts = [pd.read_csv(ensure_day(d), usecols=["price", "quantity", "transact_time", "is_buyer_maker"])
             for d in days]
    return pd.concat(parts, ignore_index=True).sort_values("transact_time").reset_index(drop=True)


class Book:
    """aggTrades 기반 bid/ask 재구성(전처리: 시점별 최신 bid/ask측 체결 인덱스)."""

    def __init__(self, tr: pd.DataFrame):
        self.ts = tr["transact_time"].to_numpy(np.int64)
        self.px = tr["price"].to_numpy(np.float64)
        self.qty = tr["quantity"].to_numpy(np.float64)
        self.bm = tr["is_buyer_maker"].to_numpy(bool)      # True = 매도 aggressor(bid에서 체결)
        self.bid_pos = np.where(self.bm)[0]
        self.ask_pos = np.where(~self.bm)[0]
        self.bid_ts = self.ts[self.bid_pos]
        self.ask_ts = self.ts[self.ask_pos]

    def quotes(self, t: int) -> tuple[float, float] | None:
        bi = np.searchsorted(self.bid_ts, t, side="right") - 1
        ai = np.searchsorted(self.ask_ts, t, side="right") - 1
        if bi < 0 or ai < 0:
            return None
        bb = self.px[self.bid_pos[bi]]
        ba = self.px[self.ask_pos[ai]]
        if bb >= ba:  # 교차(staleness) → 최신측 기준 클램프
            if self.bid_ts[bi] >= self.ask_ts[ai]:
                ba = bb + TICK
            else:
                bb = ba - TICK
        return float(bb), float(ba)


def first_true(mask: np.ndarray) -> int:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else -1


def simulate_leg(book: Book, t0: int, side: str, timeout_s: int, policy: str) -> dict | None:
    q = book.quotes(t0)
    if q is None:
        return None
    bb0, ba0 = q
    mid0 = (bb0 + ba0) / 2.0
    buy = side == "buy"
    my_px = bb0 if buy else ba0
    deadline = t0 + timeout_s * 1000
    active = t0 + LATENCY_MS

    lo_all = np.searchsorted(book.ts, active, side="left")
    hi_all = np.searchsorted(book.ts, deadline, side="right")
    ts = book.ts[lo_all:hi_all]
    px = book.px[lo_all:hi_all]
    bm = book.bm[lo_all:hi_all]

    start = 0
    while True:
        if buy:
            fill_mask = (bm & (px < my_px - 1e-9)) | (~bm & (px <= my_px + 1e-9))
            move_mask = bm & (px > my_px + 1e-9)      # bid가 내 가격 위로 → 추격 조건
        else:
            fill_mask = (~bm & (px > my_px + 1e-9)) | (bm & (px >= my_px - 1e-9))
            move_mask = (~bm) & (px < my_px - 1e-9)
        f = first_true(fill_mask[start:])
        f = -1 if f < 0 else f + start
        if policy == "peg":
            m = first_true(move_mask[start:])
            m = -1 if m < 0 else m + start
        else:
            m = -1
        if f >= 0 and (m < 0 or f <= m):
            price_bp = (1 if buy else -1) * (my_px - mid0) / mid0 * 1e4
            return {"filled": True, "cost_bp": price_bp + MAKER_FEE_BP, "t_ms": int(ts[f] - t0), "mode": "trade"}
        if m < 0:
            break
        # 추격 재호가: m 시점 재구성 호가로 이동(+지연)
        qm = book.quotes(int(ts[m]))
        new_px = qm[0] if buy else qm[1]
        if (buy and new_px > my_px + 1e-9) or (not buy and new_px < my_px - 1e-9):
            my_px = new_px
        nxt = np.searchsorted(ts, ts[m] + LATENCY_MS, side="left")
        if nxt >= len(ts):
            break
        start = int(nxt)

    qT = book.quotes(deadline)
    fb_px = qT[1] if buy else qT[0]
    price_bp = (1 if buy else -1) * (fb_px - mid0) / mid0 * 1e4
    return {"filled": False, "cost_bp": price_bp + TAKER_FEE_BP, "t_ms": timeout_s * 1000, "mode": "taker_fallback"}


def run(tr: pd.DataFrame, tag: str) -> dict:
    book = Book(tr)
    t_start, t_end = int(book.ts[0]), int(book.ts[-1])
    arrivals = list(range(t_start + 600_000, t_end - 310_000, ARRIVAL_SPACING_MS))
    # 도착 직전 5분 실현변동성(체결가 기반)
    out = {}
    for policy in ["static", "peg"]:
        for T in TIMEOUTS_S:
            rows = []
            for t0 in arrivals:
                for side in ("buy", "sell"):
                    r = simulate_leg(book, t0, side, T, policy)
                    if r is not None:
                        r["t0"] = t0
                        rows.append(r)
            df = pd.DataFrame(rows)
            key = f"{policy}_T{T}"
            fl, fb = df[df.filled], df[~df.filled]
            out[key] = {
                "n_legs": len(df), "fill_rate": float(df.filled.mean()),
                "cost_bp_mean": float(df.cost_bp.mean()),
                "cost_bp_median": float(df.cost_bp.median()),
                "cost_bp_p90": float(df.cost_bp.quantile(0.9)),
                "filled_cost_bp_mean": float(fl.cost_bp.mean()) if len(fl) else None,
                "fallback_cost_bp_mean": float(fb.cost_bp.mean()) if len(fb) else None,
            }
            print(f"[{tag}] {key}: fill={out[key]['fill_rate']:.3f} mean={out[key]['cost_bp_mean']:.2f}bp "
                  f"(filled {out[key]['filled_cost_bp_mean']:.2f} / fb {out[key]['fallback_cost_bp_mean'] if len(fb) else float('nan'):.2f})",
                  flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "days"], required=True)
    ap.add_argument("--days", default="")
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "pilot":
        tr = load_days(["2026-07-19", "2026-07-20", "2026-07-21"])
        # 파일럿 실측 창과 정렬(스냅샷 시작/끝 사이만)
        tr = tr[(tr.transact_time >= 1784429195845) & (tr.transact_time <= 1784625671624)].reset_index(drop=True)
        results = {"pilot_window": run(tr, "pilot")}
        out = OUT_DIR / "report_trades_only_pilot_validation.json"
    else:
        days = [d.strip() for d in args.days.split(",") if d.strip()]
        results = {}
        for d in days:
            tr = load_days([d])
            results[d] = run(tr, d)
        out = OUT_DIR / f"report_trades_only_{args.out_tag or 'days'}.json"

    report = {
        "experiment": "eth_maker_fill_simulation_trades_only_20260822",
        "fees_bp": {"maker": MAKER_FEE_BP, "taker": TAKER_FEE_BP},
        "fill_rule": "trades-only ultra-conservative: strict trade-through OR ask-side trade at/through my price; no queue-exhaust fills; latency 200ms; quotes reconstructed from aggTrades (pilot-validated: 89% exact, 92% within 1 tick)",
        "results": results,
    }
    out.write_text(json.dumps(report, indent=2))
    print(f"report -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
