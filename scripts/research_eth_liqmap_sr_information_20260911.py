#!/usr/bin/env python3
"""**S/R 이 돌파/되돌림 예측에 정보를 주는가** -- 경제성 아님, 정보량만 (2026-09-11).

사용자: *"경제성 돈이 되나 보려는 게 아니라, 지지와 저항은 돌파와 되돌림 예측에
주요한 정보를 주냐를 보고 싶어."*

선행 `research_eth_liqmap_sr_breakout_20260911.py` 의 «배리어 교체» 검정은 이 질문에
**불공정했다** -- 라벨이 «두 배리어 중 어느 쪽에 먼저 닿나»여서 거리비만으로 AUC .89~.93 이
나오고, 레벨이 기여할 여지가 남지 않았다. 여기서는 셋으로 갈라 다시 묻는다.

  1. 기하가 답을 **못 정하는 구간**(지지·저항 등거리)에서 레벨이 예측하는가
  2. 레벨에 **다가갔을 때** 돌파/반등을 예측하는가 (사용자 질문 그대로)
  3. 은퇴한 돌파/되돌림의 **±0.8×ATR 라벨**에 S/R 을 얹으면 오르는가

읽는 법: **ΔAUC(CTX → CTX+SR)** 하나만 본다. CTX 는 레벨을 전혀 모르는 가격/맥락 피쳐다.
라벨은 전부 종가 진입 · **다음 봉부터**(2026-09-10 체결가능 규약).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT if (ROOT / "binance_data").exists() else Path(subprocess.run(
    ["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
    capture_output=True, text=True).stdout.strip()).parent
# 패널 경로를 인자로 받는다 -- 레벨 출처(청산맵 / 거래량 프로파일)를 바꿔 끼우려고.
PANEL = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    DATA / "tmp/eth_liqmap_sr_panel_20260911/sr_panel_5m.parquet"
K5 = DATA / "binance_data/klines/{}USDT/{}USDT-5m-api.csv"
H, DELTA, TAU, K_ATR = 12, 0.0015, 0.002, 0.8
SEEDS = [20260911, 7, 131, 977, 20250401]
SHIFT = 20_000
WIN = {"VAL 25-09~12": ("2025-09-01", "2026-01-01"), "OOS 26-01~03": ("2026-01-01", "2026-04-01"),
       "최근 26-04~": ("2026-04-01", "2100-01-01")}
TRAIN_END = "2025-09-01"


def load(sym):
    return (pd.read_csv(str(K5).format(sym, sym), usecols=["timestamp", "high", "low", "close"],
                        parse_dates=["timestamp"])
            .sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True))


def touch(hi, lo, start, up, dn, span):
    """다음 봉부터 span 봉: 1=위 먼저 · 0=아래 먼저 · -1=미해소."""
    out = np.full(len(start), -1, np.int8); n = len(hi)
    for k in range(len(start)):
        a = start[k]
        if a < 1 or a + span >= n or not (up[k] > 0 and dn[k] > 0):
            continue
        h, l = hi[a:a + span], lo[a:a + span]
        iu = np.flatnonzero(h >= up[k]); idn = np.flatnonzero(l <= dn[k])
        u = iu[0] if len(iu) else 1 << 30
        d = idn[0] if len(idn) else 1 << 30
        if u == d == 1 << 30:
            continue
        out[k] = int(u < d)
    return out


def arm(X, y, ts, cols, tag, sub, rows):
    m0 = sub & (y >= 0)
    tr = (ts < TRAIN_END).to_numpy() & m0
    if tr.sum() < 500:
        return
    p = np.mean([HistGradientBoostingClassifier(random_state=s, max_iter=300)
                 .fit(X[tr][:, cols], y[tr]).predict_proba(X[:, cols])[:, 1] for s in SEEDS], axis=0)
    for name, (a, b) in WIN.items():
        m = ((ts >= a) & (ts < b)).to_numpy() & m0
        rows.append((tag, name, int(m.sum()), float(y[m].mean()),
                     roc_auc_score(y[m], p[m]) if m.sum() > 50 and len(set(y[m])) > 1 else np.nan))


def show(title, rows):
    print(f"\n=== {title} ===")
    print(f"{'팔':<22}{'창':<14}{'n':>8}{'기저':>8}{'AUC':>8}")
    print("-" * 62)
    for tag, win, n, base, auc in rows:
        print(f"{tag:<22}{win:<14}{n:>8,}{base:>8.3f}{auc:>8.4f}")


def main() -> int:
    d = pd.read_parquet(PANEL)
    e = load("ETH"); e = e[e.timestamp.isin(set(d.timestamp))].reset_index(drop=True)
    b = load("BTC").set_index("timestamp").reindex(d.timestamp).reset_index()
    hi, lo, cl = e.high.to_numpy(float), e.low.to_numpy(float), e.close.to_numpy(float)
    ts, idx, n = d.timestamp, np.arange(len(d)), len(d)
    S = pd.Series(cl); B = pd.Series(b.close.to_numpy(float))
    r5 = S.pct_change()
    atr = pd.Series(hi - lo).rolling(14).mean().to_numpy() / cl

    def pos(w):
        rl = S.rolling(w).min(); rh = S.rolling(w).max()
        return ((S - rl) / (rh - rl).replace(0, np.nan)).to_numpy()

    dR = (d.r1_px.to_numpy() - cl) / cl
    dS = (cl - d.s1_px.to_numpy()) / cl
    CTX = {"ret12": S.pct_change(12), "ret48": S.pct_change(48), "ret144": S.pct_change(144),
           "atr_pct": atr, "vol48": r5.rolling(48).std(), "pos48": pos(48), "pos144": pos(144),
           "btc12": B.pct_change(12), "btc48": B.pct_change(48),
           "div": S.pct_change(12).to_numpy() - B.pct_change(12).to_numpy(),
           "hour": ts.dt.hour, "wd": ts.dt.weekday}
    SR = {"d_res": dR, "d_sup": dS, "ratio": dR / (dR + dS),
          "w_res": d.r1_w, "w_sup": d.s1_w, "w_gap": d.r1_w.to_numpy() - d.s1_w.to_numpy(),
          "w_res2": d.r2_w, "w_sup2": d.s2_w,
          "res_span": (d.r2_px.to_numpy() - d.r1_px.to_numpy()) / cl,   # 저항 밀집도
          "sup_span": (d.s1_px.to_numpy() - d.s2_px.to_numpy()) / cl}
    F = pd.DataFrame({**CTX, **SR})
    X = np.nan_to_num(F.to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    C = [F.columns.get_loc(c) for c in CTX]
    R = [F.columns.get_loc(c) for c in SR]
    ALL = C + R

    # ── 1. 기하가 답을 못 정하는 구간 (지지·저항 등거리) ────────────────────────────
    y1 = touch(hi, lo, idx + 1, cl * (1 + dR), cl * (1 - dS), H)
    eq = np.abs(F["ratio"].to_numpy(float) - 0.5) <= 0.05
    # ⭐n 이 작아(창당 ~600) ΔAUC 의 노이즈 폭 자체를 모른다. S/R 컬럼만 순환이동한 플라시보를
    #   같은 라벨·같은 부분집합에 붙여, «레벨 모양 피쳐를 더하면 우연히 얼마나 오르는가»를 잰다.
    Xp = X.copy(); Xp[:, R] = X[np.roll(idx, SHIFT)][:, R]
    rows = []
    arm(X, y1, ts, R, "S/R 만", eq, rows)
    arm(X, y1, ts, C, "맥락만(레벨 모름)", eq, rows)
    arm(X, y1, ts, ALL, "맥락+S/R", eq, rows)
    arm(Xp, y1, ts, ALL, "맥락+플라시보S/R", eq, rows)
    show(f"1. 등거리 구간(|비율-0.5|<=0.05, 전체의 {eq.mean():.3f}) -- 저항 먼저 vs 지지 먼저", rows)

    # ── 2. 레벨 근접시 돌파 vs 반등 (양측) ─────────────────────────────────────────
    roll = np.roll(idx, SHIFT)
    for side, dist, sgn in (("저항", dR, +1), ("지지", dS, -1)):
        for lbl, dd in ((f"실제 {side}", dist), (f"플라시보({side} 거리)", dist[roll])):
            lv = cl * (1 + sgn * dd)
            near = (dd <= DELTA) & np.isfinite(lv)
            # 돌파 = 레벨 바깥 TAU 먼저 · 반등 = 레벨 안쪽 TAU 먼저
            # touch 는 «위 먼저»를 1 로 준다. 저항은 위로 뚫는 게 돌파, 지지는 아래로 뚫는 게
            # 돌파이므로 지지 쪽만 뒤집는다.
            y2 = touch(hi, lo, idx + 1, lv * (1 + TAU), lv * (1 - TAU), H)
            if sgn < 0:
                y2 = np.where(y2 >= 0, 1 - y2, -1).astype(np.int8)   # 지지는 아래로 뚫는 게 돌파
            rows = []
            arm(X, y2, ts, C, "맥락만(레벨 모름)", near, rows)
            arm(X, y2, ts, ALL, "맥락+S/R", near, rows)
            show(f"2-{lbl}: 근접 {near.sum():,}봉 -- 돌파(1) vs 반등(0)", rows)

    # ── 3. 은퇴한 돌파/되돌림의 ±0.8xATR 라벨에 S/R 을 얹으면 ────────────────────────
    y3 = touch(hi, lo, idx + 1, cl * (1 + K_ATR * atr), cl * (1 - K_ATR * atr), H)
    full = np.ones(n, bool)
    rows = []
    arm(X, y3, ts, C, "맥락만(레벨 모름)", full, rows)
    arm(X, y3, ts, R, "S/R 만", full, rows)
    arm(X, y3, ts, ALL, "맥락+S/R", full, rows)
    show(f"3. ±{K_ATR}xATR 배리어(은퇴 신호의 라벨) -- 위 먼저 vs 아래 먼저", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
