#!/usr/bin/env python3
"""**«차트 보면 지지·저항에서 되돌아가는 것 같다»를 검정** (2026-09-11, 사용자).

앞 검정에서 저항 «근접»시 반등 76.0% 가 나왔다 -- 느낌과 일치한다. 그런데 같은 구성에서
플라시보 라인도 72.9% 반등했다. 원인 후보는 **정의의 비대칭**이다:

    근접 정의: 진입/관측 기준이 종가(레벨 **아래**)인데
               반등선 = 레벨-0.2% 는 가깝고, 돌파선 = 레벨+0.2% 는 멀다.
    -> 가까운 쪽이 더 자주 먼저 닿는다. 반등률이 높은 게 당연하다(산수).

그래서 **레벨을 기준으로 대칭**인 정의로 다시 잰다:

    레벨 R 은 봉 i-1 종가 시점에 확정된다(그때 이미 알던 값).
    봉 i 에서 고가가 R 을 **실제로 친다**(터치).
    봉 **i+1 부터** R*(1+t) [계속] 과 R*(1-t) [반전] 중 먼저 닿는 쪽을 센다.

이러면 두 목표가 레벨에서 같은 거리다. 반전이 50% 를 뚜렷이 넘으면 느낌이 옳고,
50% 근처면 느낌은 **기하 착시**다. 플라시보 라인을 같은 방식으로 함께 잰다.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT if (ROOT / "binance_data").exists() else Path(subprocess.run(
    ["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
    capture_output=True, text=True).stdout.strip()).parent
PANEL = DATA / "tmp/eth_liqmap_sr_panel_20260911/sr_panel_5m.parquet"
KL5 = DATA / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
H, SHIFT = 12, 20_000
TAUS = (0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.0075, 0.01)


def race(hi, lo, start, up, dn, span):
    """start 부터 span 봉: 1=위 먼저 · 0=아래 먼저 · -1=미해소."""
    out = np.full(len(start), -1, np.int8); n = len(hi)
    for k in range(len(start)):
        a = start[k]
        if a < 1 or a + span >= n:
            continue
        h, l = hi[a:a + span], lo[a:a + span]
        iu = np.flatnonzero(h >= up[k]); idn = np.flatnonzero(l <= dn[k])
        u = iu[0] if len(iu) else 1 << 30
        d = idn[0] if len(idn) else 1 << 30
        if u == d == 1 << 30:
            continue
        out[k] = int(u < d)
    return out


def boot_ci(vals, days, rng, n=2000):
    g = [vals[days == d] for d in np.unique(days)]
    b = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def main() -> int:
    d = pd.read_parquet(PANEL)
    kl = (pd.read_csv(KL5, usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"])
          .sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True))
    kl = kl[kl.timestamp.isin(set(d.timestamp))].reset_index(drop=True)
    hi, lo, cl = kl.high.to_numpy(float), kl.low.to_numpy(float), kl.close.to_numpy(float)
    ts = d.timestamp; n = len(d); idx = np.arange(n)
    rng = np.random.default_rng(20260911)
    roll = np.roll(idx, SHIFT)

    # ⭐무조건부 기준선. 이게 0.5 가 아니다 -- ETH 5분봉은 **아래로 더 빨리** 움직인다(음의 왜도).
    #   레벨 숫자를 이것과 비교하지 않으면 «되돌아간다»가 전부 착시로 읽힌다.
    base = {}
    for t in TAUS:
        y = race(hi, lo, idx + 1, cl * (1 + t), cl * (1 - t), H)
        r = y >= 0
        base[t] = float(y[r].mean())
        print(f"무조건부 기준선 ±{t*100:.2f}%: 아무 봉에서나 위 먼저 {base[t]:.4f} "
              f"(= 아래 먼저 {1-base[t]:.4f}) · n={r.sum():,}")

    for side, px_col, is_res in (("저항", "r1_px", True), ("지지", "s1_px", False)):
        lvl_all = d[px_col].to_numpy(float)
        dist = np.abs(lvl_all - cl) / cl
        keep = {}
        for lbl, dd in ((f"실제 {side}", dist), (f"플라시보({side} 거리)", dist[roll])):
            # 봉 i-1 에 확정된 레벨을 봉 i 가 실제로 치는가(터치). 판정은 봉 i+1 부터.
            L = np.full(n, np.nan)
            L[1:] = cl[:-1] * (1 + dd[:-1]) if is_res else cl[:-1] * (1 - dd[:-1])
            hit = np.zeros(n, bool)
            hit[1:] = (hi[1:] >= L[1:]) if is_res else (lo[1:] <= L[1:])
            ev = np.flatnonzero(hit & np.isfinite(L))
            ev = ev[(ev >= 1) & (ev + 1 + H < n)]
            print(f"\n[{lbl}] 터치 {len(ev):,}건 · 거리중앙 {np.nanmedian(dd[ev])*100:.3f}% "
                  f"(터치봉 종가가 레벨 {'위' if is_res else '아래'}로 마감한 비율 "
                  f"{(cl[ev] > L[ev]).mean() if is_res else (cl[ev] < L[ev]).mean():.3f})")
            days = pd.to_datetime(ts.to_numpy()[ev]).date
            for t in TAUS:
                y = race(hi, lo, ev + 1, L[ev] * (1 + t), L[ev] * (1 - t), H)
                r = y >= 0
                # 반전 = 저항이면 아래 먼저(0) · 지지면 위 먼저(1)
                rev = (1 - y[r]) if is_res else y[r]
                ci = boot_ci(rev.astype(float), np.asarray(days)[r], rng)
                # 기준선: 저항 반전 = 아래 먼저 · 지지 반전 = 위 먼저
                bl = (1 - base[t]) if is_res else base[t]
                print(f"   ±{t*100:.2f}%: 해소 {r.mean():.3f} · **반전 {rev.mean():.3f}** "
                      f"(일군집 95%CI [{ci[0]:.3f}, {ci[1]:.3f}]) · "
                      f"기준선 {bl:.3f} -> **초과 {(rev.mean()-bl)*100:+.1f}pp**")
                keep.setdefault(t, {})[lbl] = (dd[ev][r], rev)
        # ⭐터치 건수가 실제/플라시보로 다르다 -- 거리 분포가 안 맞는다는 뜻이다.
        #   거리 십분위 안에서만 비교하고 실제 쪽 건수로 가중해 합친다(교란 제거).
        print(f"  -- {side}: 거리 십분위 층화 후 실제-플라시보 반전율 격차 --")
        for t in TAUS:
            (da, ra), (db, rb) = keep[t][f"실제 {side}"], keep[t][f"플라시보({side} 거리)"]
            edge = np.nanpercentile(da, np.arange(0, 101, 10))
            ba, bb = np.digitize(da, edge[1:-1]), np.digitize(db, edge[1:-1])
            num = wsum = 0.0
            for g in range(10):
                ma, mb = ba == g, bb == g
                if ma.sum() >= 50 and mb.sum() >= 50:
                    num += ma.sum() * (ra[ma].mean() - rb[mb].mean()); wsum += ma.sum()
            print(f"     ±{t*100:.2f}%: {num/wsum*100:+.2f}pp (가중 n={int(wsum):,})" if wsum
                  else f"     ±{t*100:.2f}%: 층 부족")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
