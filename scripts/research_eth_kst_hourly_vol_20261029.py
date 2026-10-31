#!/usr/bin/env python3
"""ETH 한국 시간(KST) 30분 칸별 변동성 지도 (2026-10-29, 사용자 «시간대 별로 변동성이 큰 한국 시간대를 분석해줘»).

변동성 = 그 30분 칸 안 1분 로그수익 제곱합의 제곱근(실현변동성, bp). 칸마다 날짜 평균 → 하루 전체 칸 평균 대비 배수.
나눔: 연도 · 평일/주말(KST 요일) · 미국 서머타임 여부(미국장 개장이 22:30 KST 냐 23:30 KST 냐).
08-26 연구(eth_market_open_volatility_window)는 «개장 시각 기준 상대 분»이었다 -- 이번은 24시간 KST 절대 시각.
원천: $DATA_ROOT/binance_vision/klines1m/ETHUSDT-1m-*.parquet(선물 1분봉, 2022-01 ~ 2026-09-14 -- 9월 파일이 월 중간에 잘려 있다).
  DATA_ROOT=/home/kbj20/crypto-scalping/data python scripts/research_eth_kst_hourly_vol_20261029.py
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
OUT = ROOT / "docs/experiments/eth_kst_hourly_vol_20261029"


def load() -> pd.DataFrame:
    fs = sorted(glob.glob(str(DATA / "binance_vision/klines1m/ETHUSDT-1m-*.parquet")))
    assert fs, DATA
    d = pd.concat([pd.read_parquet(f, columns=["t", "c"]) for f in fs]).drop_duplicates("t").sort_values("t")
    d["ts"] = pd.to_datetime(d.t, unit="ms", utc=True)
    r = np.log(d.c).diff()
    r[d.t.diff() != 60_000] = np.nan                       # 빠진 분 뒤 수익은 버린다(여러 분 수익이 한 칸에 몰리지 않게)
    d["r2"] = r ** 2
    return d.dropna(subset=["r2"])


def slots(d: pd.DataFrame) -> pd.DataFrame:
    kst = d.ts.dt.tz_convert("Asia/Seoul")
    ny = d.ts.dt.tz_convert("America/New_York")
    d = d.assign(day=kst.dt.date, slot=kst.dt.hour * 2 + kst.dt.minute // 30, wkend=kst.dt.dayofweek >= 5,
                 dst=(ny.dt.tz_localize(None) - d.ts.dt.tz_localize(None)) == pd.Timedelta(hours=-4), year=kst.dt.year)
    g = d.groupby(["day", "slot"]).agg(rv=("r2", lambda x: np.sqrt(x.sum()) * 1e4), n=("r2", "size"),
                                        wkend=("wkend", "first"), dst=("dst", "first"), year=("year", "first"))
    return g[g.n >= 25].reset_index()                       # 30분 중 25분 이상 있는 칸만


def profile(g: pd.DataFrame) -> pd.Series:
    m = g.groupby("slot").rv.mean()
    return m / m.mean()


def label(s: int) -> str:
    return f"{s // 2:02d}:{30 * (s % 2):02d}"


def main() -> None:
    g = slots(load())
    OUT.mkdir(parents=True, exist_ok=True)
    cols = {"전체 2022-26": g, "2022-23": g[g.year <= 2023], "2025-26": g[g.year >= 2025],
            "2025-26 평일": g[(g.year >= 2025) & ~g.wkend], "2025-26 주말": g[(g.year >= 2025) & g.wkend],
            "2025-26 평일·서머타임": g[(g.year >= 2025) & ~g.wkend & g.dst], "2025-26 평일·표준시": g[(g.year >= 2025) & ~g.wkend & ~g.dst]}
    tab = pd.DataFrame({k: profile(v) for k, v in cols.items()})
    tab.index = [label(s) for s in tab.index]
    tab.insert(0, "2025-26 평균 bp", (g[g.year >= 2025].groupby("slot").rv.mean()).values)
    tab.to_csv(OUT / "kst_30min_profile.csv", float_format="%.3f")
    print(f"칸 수 {len(g):,} · 날 {g.day.nunique():,} · 2025-26 칸 평균 {g[g.year >= 2025].rv.mean():.1f}bp · "
          f"평일 서머 {g[(g.year >= 2025) & ~g.wkend & g.dst].day.nunique()}일 · 평일 표준 {g[(g.year >= 2025) & ~g.wkend & ~g.dst].day.nunique()}일")
    print(tab.round(2).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    fm.fontManager.addfont("/mnt/c/Windows/Fonts/malgun.ttf")
    plt.rcParams["font.family"] = fm.FontProperties(fname="/mnt/c/Windows/Fonts/malgun.ttf").get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(17, 7), dpi=130)
    x = np.arange(48)
    for k, col, lw, ls in (("2025-26 평일·서머타임", "#d1495b", 2.4, "-"), ("2025-26 평일·표준시", "#1f4e9c", 2.4, "-"),
                           ("2025-26 주말", "#2e9d5b", 2.0, "-"), ("2022-23", "#888", 1.4, "--")):
        ax.plot(x, tab[k].values, color=col, lw=lw, ls=ls, label=k)
    ax.axhline(1, color="#999", lw=.8)
    top = max(tab[["2025-26 평일·서머타임", "2025-26 평일·표준시"]].max()) * 1.08
    ax.set_ylim(0, top)
    for xs, txt in ((45, "미국장 개장\n22:30(서머)"), (47, "23:30\n(표준)"), (43, "미 지표\n21:30(서머)"), (18, "한국·일본장\n09:00"), (32, "유럽장\n16:00")):
        ax.axvline(xs, color="#bbb", lw=.8, ls=":")
        ax.text(xs, top * .98, txt, fontsize=8.5, ha="center", va="top", color="#555")
    ax.set_xticks(x[::2])
    ax.set_xticklabels([label(s) for s in x[::2]], fontsize=9)
    ax.set_xlabel("한국 시간(KST) 30분 칸 시작")
    ax.set_ylabel("하루 평균 대비 배수(실현변동성)")
    ax.set_title("ETH 한국 시간대별 변동성 — 30분 칸 실현변동성 ÷ 하루 평균 (1분봉, 2022-01 ~ 2026-09-14)", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "kst_30min_profile.png")


if __name__ == "__main__":
    main()
