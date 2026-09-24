#!/usr/bin/env python3
"""**사전등록** — 하이퍼리퀴드 고래 실제 청산가 뭉치는 가격을 끌어당기는가 / 지나면 연쇄가 나는가. (2026-09-24)

데이터: `live_hyperliquid_positions_collector_20260924.py` 의 hl_positions(2.5분 바퀴) + 바이낸스 ETHUSDT 1분봉.
등록일에는 몇 시간치뿐이라 **판정하지 않는다** -- MIN_DAYS 이상 쌓인 뒤 이 파일을 **고치지 않고** 돌린다.
결과를 보고 문턱·지평·거리를 바꾸면 사전등록이 아니게 된다(바꿀 거면 새 파일·새 날짜로).

H1 자석: 스냅샷 시점 현재가 DIST_LO~DIST_HI 안의, 한쪽의 가장 큰 뭉치(≥ MIN_ETH)에 H 안에 닿는 비율
    vs **같은 거리·같은 방향의 기준 닿음 확률**(전 기간 가격 경로에서 경험적으로, 거리 0.25% 칸별).
    지표 = 초과 닿음률(관측 − 기대). 일 블록 부트스트랩 95% CI 가 0 을 넘으면 «자석» 지지.
H2 연쇄: 닿은 사건만, 닿은 뒤 CONT_MIN 안에 같은 방향으로 더 간 최대 폭(bp)
    vs 같은 거리 기준 사건(뭉치 없는 스냅샷에서 같은 거리 가격에 닿은 경우)의 같은 값. 차이 CI.
독립성: 스냅샷은 **시간당 1개**(바퀴 간 거의 같은 포지션이라 2.5분 표본은 겹친다).
한계: 대상은 «48h 거래액 상위 300주소»라 오래 들고만 있는 고래는 빠진다. HL 은 ETH 테이커 흐름의 ~8%.

사용: python scripts/research_hl_whale_liq_magnet_20260924.py [--db 경로] [--force]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "live" / "hyperliquid_positions.duckdb"
# ── 사전등록 상수 (2026-09-24) ─────────────────────────────────────────────
MIN_DAYS = 14
MIN_ETH = 1000.0
DIST_LO, DIST_HI = 0.01, 0.05
BUCKET = 5.0                  # 뭉치 폭($) -- 대시보드 /api/hl-whale-liq 와 같다
HORIZONS_MIN = (60, 240)
CONT_MIN = 60
DIST_BIN = 0.0025
BOOT = 2000
SEED = 20260924


def klines(start_ms: int, end_ms: int) -> pd.DataFrame:
    out, s = [], start_ms
    while s < end_ms:
        d = json.load(urllib.request.urlopen(
            f"https://fapi.binance.com/fapi/v1/klines?symbol=ETHUSDT&interval=1m&startTime={s}&limit=1500", timeout=20))
        if not d:
            break
        out += d
        s = int(d[-1][0]) + 60000
        time.sleep(0.2)
    k = pd.DataFrame([(int(x[0]) // 1000, float(x[2]), float(x[3]), float(x[4])) for x in out],
                     columns=["t", "h", "l", "c"]).drop_duplicates("t").set_index("t")
    return k


def snapshots(db: Path) -> pd.DataFrame:
    con = duckdb.connect(str(db), read_only=True)
    cyc = con.execute("SELECT ts_ms FROM hl_cycles WHERE n_ok > 0 ORDER BY ts_ms").df()
    cyc["hour"] = cyc.ts_ms // 3_600_000
    first = cyc.groupby("hour").ts_ms.min()                     # 시간당 첫 완결 바퀴
    pos = con.execute("SELECT ts_ms, szi, liq_px FROM hl_positions WHERE coin='ETH' AND liq_px > 0").df()
    con.close()
    cyc_start = np.sort(cyc.ts_ms.values)
    pos["cyc"] = cyc_start[np.searchsorted(cyc_start, pos.ts_ms.values, side="right") - 1]
    return pos[pos.cyc.isin(first.values)]


def touch_time(k: pd.DataFrame, t0: int, level: float, up: bool, horizon_s: int) -> int | None:
    w = k.loc[t0 + 60: t0 + horizon_s]
    hit = w.index[(w.h >= level) if up else (w.l <= level)]
    return int(hit[0]) if len(hit) else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--force", action="store_true", help="MIN_DAYS 미만이어도 돌린다(판정 금지, 동작 점검용)")
    a = ap.parse_args()
    snap = snapshots(a.db)
    if snap.empty:
        print("스냅샷 없음"); return
    days = (snap.cyc.max() - snap.cyc.min()) / 86_400_000
    print(f"스냅샷 {snap.cyc.nunique()}개(시간당 1) · {days:.1f}일")
    if days < MIN_DAYS:
        print(f"⚠️ {MIN_DAYS}일 미만 -- 판정하지 않는다" + ("" if a.force else " (--force 로 동작만 점검)"))
        if not a.force:
            return
    k = klines(int(snap.cyc.min()) - 60_000, int(snap.cyc.max()) + max(HORIZONS_MIN) * 60_000 + 60_000)
    rows = []
    for cyc, g in snap.groupby("cyc"):
        t0 = int(cyc) // 60_000 * 60
        if t0 not in k.index:
            continue
        px = k.c.loc[t0]
        agg: dict[tuple[int, bool], float] = {}
        for szi, liq in zip(g.szi, g.liq_px):
            key = (int(round(liq / BUCKET)), szi > 0)
            agg[key] = agg.get(key, 0.0) + abs(szi)
        for long_side in (True, False):      # 롱 뭉치 = 아래(닿으면 하락), 숏 뭉치 = 위
            c = [(kk * BUCKET, v) for (kk, lg), v in agg.items() if lg == long_side and v >= MIN_ETH
                 and DIST_LO <= abs(kk * BUCKET / px - 1) <= DIST_HI and ((kk * BUCKET < px) == long_side)]
            if c:
                lvl, sz = max(c, key=lambda x: x[1])
                rows.append({"t0": t0, "px": px, "lvl": lvl, "sz": sz, "up": not long_side,
                             "dist": abs(lvl / px - 1), "day": t0 // 86400})
    ev = pd.DataFrame(rows)
    print(f"뭉치 사건 {len(ev)} (한쪽당 시간당 최대 1)")
    if ev.empty:
        return
    # 기준: 모든 시간당 시점 × 거리 칸 × 방향의 경험적 닿음 확률
    t0s = sorted({int(c) // 60_000 * 60 for c in snap.cyc if int(c) // 60_000 * 60 in k.index})
    rng = np.random.default_rng(SEED)
    for H in HORIZONS_MIN:
        base = {}
        for up in (True, False):
            for b in np.arange(DIST_LO, DIST_HI + 1e-9, DIST_BIN):
                d = b + DIST_BIN / 2
                hits = [touch_time(k, t, k.c.loc[t] * (1 + d if up else 1 - d), up, H * 60) is not None for t in t0s]
                base[(up, round(b, 4))] = np.mean(hits) if hits else np.nan
        ev["touch"] = [touch_time(k, r.t0, r.lvl, r.up, H * 60) is not None for r in ev.itertuples()]
        ev["exp"] = [base.get((r.up, round(np.floor(r.dist / DIST_BIN) * DIST_BIN, 4)), np.nan) for r in ev.itertuples()]
        e = ev.dropna(subset=["exp"])
        diff = (e.touch - e.exp)
        ds = e.day.unique()
        bs = [diff[e.day.isin(rng.choice(ds, len(ds)))].mean() for _ in range(BOOT)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"H1 자석 {H}분: 닿음 {e.touch.mean():.1%} vs 기대 {e.exp.mean():.1%} · 초과 {diff.mean():+.1%} [{lo:+.1%},{hi:+.1%}] · n={len(e)} · {len(ds)}일")
    # H2 연쇄: 닿은 뒤 CONT_MIN 안에 같은 방향 추가 이동(bp)
    def cont(t_hit: int, lvl: float, up: bool) -> float:
        w = k.loc[t_hit: t_hit + CONT_MIN * 60]
        return (w.h.max() / lvl - 1) * 1e4 if up else (1 - w.l.min() / lvl) * 1e4
    hits = [(r, touch_time(k, r.t0, r.lvl, r.up, max(HORIZONS_MIN) * 60)) for r in ev.itertuples()]
    xs = [cont(t, r.lvl, r.up) for r, t in hits if t]
    ctrl = []
    for t in t0s:
        for up in (True, False):
            d = rng.uniform(DIST_LO, DIST_HI); lvl = k.c.loc[t] * (1 + d if up else 1 - d)
            th = touch_time(k, t, lvl, up, max(HORIZONS_MIN) * 60)
            if th:
                ctrl.append(cont(th, lvl, up))
    if xs and ctrl:
        print(f"H2 연쇄: 뭉치 닿은 뒤 {CONT_MIN}분 추가 이동 중앙 {np.median(xs):.1f}bp (n={len(xs)}) vs 기준 {np.median(ctrl):.1f}bp (n={len(ctrl)})")


if __name__ == "__main__":
    main()
