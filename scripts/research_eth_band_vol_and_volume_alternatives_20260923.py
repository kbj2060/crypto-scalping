#!/usr/bin/env python3
"""SMA144 밴드의 «ATR 대안» · 대시보드의 «거래대금 대안» (2026-09-23).

사용자: *"sma144 + atr 밴드를 업그레이드 · 거래대금보다 좋은 거래량 지표 · atr 보다 좋은 지표"*

## 채점 기준을 먼저 고정한다
「더 나은 변동성 추정치」는 추정오차로 정하지 않는다 — 대시보드에서 ATR 이 하는 일은 하나,
**추세 veto 의 히스테리시스 폭**이다(dashboard/server.py::trend_veto_rows). 그러니 같은 목적함수로 잰다:
    순추세 − 역추세 = mean_bp(측면 s 진입 | veto=s) − mean_bp(측면 s 진입 | veto=−s)
    TP +1.5% / SL −0.7%, 장중 고저 판정(라이브 규약), 최대 보유 288봉(24h).
서버 주석의 기준선(롱 +16.2bp · 숏 +18.3bp, 겹치지 않는 24h 창 1,719개)을 **양성 대조군**으로 먼저 재현한다.

## 폭 교란을 제거한다
추정치마다 스케일이 다르므로 K 를 그냥 1.0 으로 두면 「더 넓은 밴드」를 재는 것이 된다.
각 추정치의 K 를 **TRAIN(앞 30%)에서 중앙 밴드폭이 ATR144×1.0 과 같아지도록** 맞추고 나머지에서 평가한다.

## Part B — 거래대금 대안
거래대금의 대시보드 역할은 「전환 탐지」(거래대금 z288 q90 AND 체결속도 z288 q90)다.
그래서 같은 과제로 잰다: **앞 30분 실현 레인지가 상위 25% 인가**.
🔴 공짜로 맞히는 축 두 개를 통제한다 — ① 현재 봉 자신의 레인지 ② 시간대(거래량은 시간대 함수).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
KL = next(p for p in (HERE / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv",
                      Path("/home/kbj20/crypto-scalping/binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"))
          if p.exists())
OUT = HERE / "tmp/eth_band_vol_volume_alt_20260923"

N = 144                 # 밴드 창 = SMA 창 (5분봉 144 = 12시간)
TP, SL, MAXH = 0.015, 0.007, 288
BOOT, SEED = 600, 20260923
TRAIN_FRAC = 0.30


# ── 배리어: 추정치와 무관하므로 봉마다 한 번만 푼다 ────────────────────────────────
def barrier_bp(high, low, close, long: bool):
    """진입 close[t] · TP/SL 장중 고저 판정 · 같은 봉 동시터치는 SL 우선(보수적).
    미해결은 close[t+MAXH] 청산. 반환 bp, 꼬리 MAXH 개는 NaN."""
    n = len(close)
    tp_px = close * (1 + TP) if long else close * (1 - TP)
    sl_px = close * (1 - SL) if long else close * (1 + SL)
    out = np.full(n, np.nan)
    done = np.zeros(n, bool)
    idx = np.arange(n)
    for h in range(1, MAXH + 1):
        j = idx + h
        ok = (j < n) & ~done
        if not ok.any():
            break
        jj = j[ok]
        hi, lo = high[jj], low[jj]
        sl_now = (lo <= sl_px[ok]) if long else (hi >= sl_px[ok])
        tp_now = (hi >= tp_px[ok]) if long else (lo <= tp_px[ok])
        res = np.where(sl_now, -SL * 1e4, np.where(tp_now, TP * 1e4, np.nan))
        w = np.flatnonzero(ok)
        hit = np.isfinite(res)
        out[w[hit]] = res[hit]
        done[w[hit]] = True
    tail = ~done & (idx + MAXH < n)
    ex = close[idx[tail] + MAXH]
    out[tail] = ((ex / close[tail] - 1) if long else (1 - ex / close[tail])) * 1e4
    out[idx + MAXH >= n] = np.nan
    return out


# ── 변동성 추정치: 전부 «가격 대비 비율» 로 돌려준다 ─────────────────────────────
def estimators(df: pd.DataFrame) -> dict[str, np.ndarray]:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    O, H, L, C = (pd.Series(x) for x in (o, h, l, c))
    pc = C.shift()
    tr = pd.concat([H - L, (H - pc).abs(), (L - pc).abs()], axis=1).max(axis=1)
    r = np.log(C / pc)
    rm = lambda s: s.rolling(N).mean()

    hl2 = np.log(H / L) ** 2
    rs = np.log(H / O) * np.log(H / C) + np.log(L / O) * np.log(L / C)
    co2 = np.log(C / O) ** 2
    oc2 = np.log(O / pc) ** 2
    k_yz = 0.34 / (1.34 + (N + 1) / (N - 1))

    e = {
        "atr_wilder(현행)": tr.ewm(alpha=1 / N, adjust=False).mean() / C,
        "atr_sma":          rm(tr) / C,
        "parkinson":        np.sqrt(rm(hl2) / (4 * np.log(2))),
        "garman_klass":     np.sqrt((rm(0.5 * hl2) - (2 * np.log(2) - 1) * rm(co2)).clip(lower=0)),
        "rogers_satchell":  np.sqrt(rm(rs).clip(lower=0)),
        "yang_zhang":       np.sqrt((rm(oc2) - rm(pd.Series(np.log(O / pc))) ** 2
                                     + k_yz * (rm(co2) - rm(pd.Series(np.log(C / O))) ** 2)
                                     + (1 - k_yz) * rm(rs)).clip(lower=0)),
        "realized_cc":      r.rolling(N).std(),
        "ewma_rv":          np.sqrt(pd.Series(r ** 2).ewm(alpha=1 / N, adjust=False).mean()),
        # 점프 강건(Barndorff-Nielsen): 큰 한 방이 밴드를 부풀리지 않는다
        "bipower":          np.sqrt((np.pi / 2) * rm(pd.Series(np.abs(r) * np.abs(r).shift()))),
    }
    return {k: np.asarray(v, float) for k, v in e.items()}


DVOL = Path("/home/kbj20/crypto-scalping/data/derivatives/deribit_dvol/ETH_dvol_hourly.csv")


def project_estimators(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """🔴이 저장소가 **실제로 들고 있는** 변동성 지표들. A1 의 9종과 달리 «같은 창을 다르게
    읽는 것」이 아니라 **종류가 다르다** — 조건부 변동성 모델 · 레짐 상대 랭크 · 내재변동성.
    산식은 features/engineering.py · features/elite.py 의 배포본을 그대로 옮겼다."""
    c, h, l = df["close"], df["high"], df["low"]
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()            # _calc_atr(length=14)
    atr14p = atr14 / c
    r = np.log(c / pc).fillna(0.0)

    # GARCH(1,1) a=.10 b=.85 -- features/elite.py::_compute_garch 의 고정 파라미터
    e2 = (r ** 2).to_numpy()
    a_, b_ = 0.10, 0.85
    v0 = max(float(e2[0]), 1e-8)
    w_ = v0 * (1 - a_ - b_)
    s2 = np.empty(len(e2)); s2[0] = v0
    for t in range(1, len(e2)):
        s2[t] = w_ + a_ * e2[t - 1] + b_ * s2[t - 1]
    garch = pd.Series(np.sqrt(s2), index=c.index)

    # bb_width (20,2) -- engineering.py:272-278
    bm = c.rolling(20, min_periods=1).mean()
    bs = c.rolling(20, min_periods=1).std(ddof=0)
    bbw = (4 * bs) / (bm + 1e-8)

    # 레짐 상대 랭크 -- _rolling_pct_rank(288). rolling.rank 는 마지막 원소의 순위라 동치
    # (동점 처리만 average vs <= 로 다르다 -- 연속값이라 실질 차이 없음).
    rank = lambda x: x.rolling(288, min_periods=2).rank(pct=True)
    base = float(np.nanmedian(atr14p))

    rv12, rv288 = r.rolling(12).std(), r.rolling(288).std()

    out = {
        "garch_vol(1,1)":        garch,
        "bb_width(20,2)":        bbw,
        # 랭크/비율은 «폭» 단위가 아니다 -- 중앙 ATR 을 base 로 곱해 폭으로 만든다.
        # K 보정이 중앙값을 다시 맞추므로 base 선택은 결과에 영향이 없고, **동역학만** 남는다.
        "atr_pct_rank_288":      rank(atr14p) * base,
        "bb_width_pct_rank_288": rank(bbw) * base,
        "realized_vol_ratio":    (rv12 / rv288.replace(0, np.nan)) * base,
        # compression_score = 1 - max(두 랭크). 압축일수록 크다 -> 폭으로 쓰려면 뒤집는다.
        "1-compression_score":   np.maximum(rank(atr14p), rank(bbw)) * base,
    }
    if DVOL.exists():
        d = pd.read_csv(DVOL, parse_dates=["timestamp"]).set_index("timestamp")["close"]
        # 시각 T 에 마감된 시간봉은 T 이후에야 안다 -- reindex(ffill) 뒤 한 칸 민다.
        dv = d.reindex(df["timestamp"], method="ffill").to_numpy() / 100.0
        dv = pd.Series(dv, index=c.index).shift(1)
        out["DVOL(내재·미래지향)"] = dv
        # VRP 축: 내재 / 실현. 둘 다 연율화 불필요 -- 비율이라 상수배가 K 에 흡수된다.
        out["DVOL/실현 비율"] = (dv / (rv288 * np.sqrt(288 * 365)).replace(0, np.nan)) * base
    return {k: np.asarray(v, float) for k, v in out.items()}


def veto_side(dev: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """히스테리시스: 밴드 밖에서만 바뀌고 안이면 직전 유지. 서버 구현과 같은 규칙."""
    s = np.zeros(len(dev), np.int8)
    cur = 0
    for i in range(len(dev)):
        if not (np.isfinite(dev[i]) and np.isfinite(eps[i])):
            s[i] = 0
            continue
        if dev[i] > eps[i]:
            cur = 1
        elif dev[i] < -eps[i]:
            cur = -1
        s[i] = cur
    return s


def block_ci(vals, days, mask_a, mask_b, rng):
    """일 블록 부트스트랩. mask_a 평균 − mask_b 평균."""
    u = np.unique(days)
    by = {d: np.flatnonzero(days == d) for d in u}
    out = []
    for _ in range(BOOT):
        i = np.concatenate([by[d] for d in rng.choice(u, len(u), replace=True)])
        a, b = vals[i][mask_a[i]], vals[i][mask_b[i]]
        if len(a) > 20 and len(b) > 20:
            out.append(a.mean() - b.mean())
    if len(out) < BOOT // 3:
        return np.nan, np.nan
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


BARRIERS = {"TP1.5%/SL0.7%/24h": (0.015, 0.007, 288),
            "TP0.7%/SL0.7%/6h": (0.007, 0.007, 72)}
COST_BP = 1.36                     # USDC 왕복 (커밋 93c27f6c 와 같은 차감)


def score(side, lbp, sbp, days, ev, rng, ci=True):
    """R1 = mean_bp(측면 s | veto=s) - mean_bp(측면 s | veto=-s). 비용은 차분에서 상쇄된다."""
    out = {}
    for lbl, bp, s_ in (("롱", lbp, 1), ("숏", sbp, -1)):
        a, b = ev & (side == s_) & np.isfinite(bp), ev & (side == -s_) & np.isfinite(bp)
        v = np.nan_to_num(bp)
        out[lbl] = round(float(v[a].mean() - v[b].mean()), 2)
        out[lbl + "_CI"] = [round(x, 2) for x in block_ci(v, days, a, b, rng)] if ci else None
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(KL, parse_dates=["timestamp"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    c = df["close"].to_numpy(float)
    hi, lo = df["high"].to_numpy(float), df["low"].to_numpy(float)
    days = df["timestamp"].dt.floor("D").to_numpy()
    yr = df["timestamp"].dt.year.to_numpy()
    n = len(df)
    cut = int(n * TRAIN_FRAC)
    ndays_ev = len(np.unique(days[cut:]))
    print(f"[data] {n:,} bars  {df.timestamp.iloc[0]} .. {df.timestamp.iloc[-1]}  "
          f"days={len(np.unique(days)):,}  TRAIN<{df.timestamp.iloc[cut]}")

    global TP, SL, MAXH
    bars = {}
    for name, (tp, sl, mh) in BARRIERS.items():
        TP, SL, MAXH = tp, sl, mh
        print(f"[barrier] {name}")
        bars[name] = (barrier_bp(hi, lo, c, True) - COST_BP, barrier_bp(hi, lo, c, False) - COST_BP)

    sma144 = pd.Series(c).rolling(N).mean().to_numpy()
    dev144 = (c - sma144) / sma144
    est = estimators(df)
    ev = np.zeros(n, bool); ev[cut:] = True
    base_w = np.nanmedian(est["atr_wilder(현행)"][:cut])

    # ── A1. 추정치 9종 (폭을 TRAIN 중앙값으로 맞춰 «폭» 교란 제거) ────────────────
    print("\n[A1] 변동성 추정치 -- 밴드폭을 ATR144x1.0 에 맞춘 뒤 비교")
    a1 = []
    for name, v in est.items():
        k = base_w / np.nanmedian(v[:cut])
        side = veto_side(dev144, k * v)
        r = {"est": name, "K": round(float(k), 4),
             "밴드폭중앙%": round(float(np.nanmedian(k * v[cut:]) * 100), 4),
             "뒤집힘/일": round(float((np.diff(side[cut:]) != 0).sum()) / ndays_ev, 2)}
        for bname, (L, S) in bars.items():
            r[bname] = score(side, L, S, days, ev, rng)
        a1.append(r)
        b1, b2 = (r[k2] for k2 in BARRIERS)
        print(f"  {name:20s} 폭={r['밴드폭중앙%']:.3f}% 뒤집힘/일 {r['뒤집힘/일']:.2f}  "
              f"[1.5/24h] 롱{b1['롱']:+6.2f}{b1['롱_CI']} 숏{b1['숏']:+6.2f}{b1['숏_CI']}  "
              f"[0.7/6h] 롱{b2['롱']:+6.2f} 숏{b2['숏']:+6.2f}")

    # ── A2. 진짜 레버: 밴드폭 K x 창 N ─────────────────────────────────────────
    print("\n[A2] K x N 스윕 (atr_wilder) -- 추정치가 아니라 이 축이 레버인가")
    a2 = []
    for nn in (72, 144, 288, 576):
        sm = pd.Series(c).rolling(nn).mean().to_numpy()
        d = (c - sm) / sm
        tr = pd.concat([pd.Series(hi) - pd.Series(lo),
                        (pd.Series(hi) - pd.Series(c).shift()).abs(),
                        (pd.Series(lo) - pd.Series(c).shift()).abs()], axis=1).max(axis=1)
        a = (tr.ewm(alpha=1 / nn, adjust=False).mean() / pd.Series(c)).to_numpy()
        for k in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
            side = veto_side(d, k * a)
            r = {"N": nn, "K": k,
                 "밴드폭중앙%": round(float(np.nanmedian(k * a[cut:]) * 100), 4),
                 "뒤집힘/일": round(float((np.diff(side[cut:]) != 0).sum()) / ndays_ev, 2),
                 "순추세비중%": round(100 * float((side[cut:] != 0).mean()), 1)}
            ci = (nn == 144 and k in (0.0, 1.0, 3.0)) or (k == 1.0)
            for bname, (L, S) in bars.items():
                r[bname] = score(side, L, S, days, ev, rng, ci=ci)
            a2.append(r)
            b1, b2 = (r[k2] for k2 in BARRIERS)
            print(f"  N={nn:3d} K={k:.1f} 폭={r['밴드폭중앙%']:.3f}% 뒤집힘/일 {r['뒤집힘/일']:5.2f}  "
                  f"[1.5/24h] 롱{b1['롱']:+6.2f} 숏{b1['숏']:+6.2f}  "
                  f"[0.7/6h] 롱{b2['롱']:+6.2f} 숏{b2['숏']:+6.2f}"
                  + (f"  CI롱{b1['롱_CI']}" if ci else ""))

    # ── A4. 저장소가 실제로 가진 지표들 (A1 과 달리 «종류»가 다르다) ──────────────
    print("\n[A4] 저장소 지표 -- 창 겹침이 다르므로 «같은 창의 ATR» 과 짝지어 비교한다")
    a4 = []
    for name, v in project_estimators(df).items():
        # 🔴K 보정은 **그 지표가 살아 있는 구간의 앞 30%**로 한다. DVOL 은 2024-01 부터라
        # 전역 TRAIN(~2023-04)에 한 점도 없어 nanmedian 이 NaN -> 밴드가 영원히 안 뒤집혔다.
        fin = np.flatnonzero(np.isfinite(v))
        own_cut = fin[int(len(fin) * TRAIN_FRAC)] if len(fin) > 10 else cut
        k = base_w / np.nanmedian(v[fin[fin < own_cut]])
        side = veto_side(dev144, k * v)
        common = np.isfinite(v)
        common[:own_cut] = False                        # 자기 TRAIN 은 평가에서 뺀다
        pair = veto_side(dev144, 1.0 * est["atr_wilder(현행)"])   # 같은 구간의 ATR
        r = {"est": name, "K": round(float(k), 4),
             "겹치는일수": int(len(np.unique(days[common]))),
             "뒤집힘/일": round(float((np.diff(side[common]) != 0).sum())
                            / max(len(np.unique(days[common])), 1), 2)}
        for bname, (L, S) in bars.items():
            r[bname] = score(side, L, S, days, common, rng)
            r[bname + "_짝ATR"] = score(pair, L, S, days, common, rng, ci=False)
        a4.append(r)
        b = r[list(BARRIERS)[0]]; pb = r[list(BARRIERS)[0] + "_짝ATR"]
        b2 = r[list(BARRIERS)[1]]; pb2 = r[list(BARRIERS)[1] + "_짝ATR"]
        print(f"  {name:22s} 일수 {r['겹치는일수']:>5,} 뒤집힘/일 {r['뒤집힘/일']:5.2f}  "
              f"[1.5/24h] {b['롱']:+6.2f}/{b['숏']:+6.2f} (짝ATR {pb['롱']:+6.2f}/{pb['숏']:+6.2f})  "
              f"[0.7/6h] {b2['롱']:+6.2f}/{b2['숏']:+6.2f} (짝 {pb2['롱']:+6.2f}/{pb2['숏']:+6.2f})")

    # ── A3. 현행 설정의 연도별 부호 (커밋 93c27f6c 의 «5연도 양측 전부 양수» 확인) ──
    side = veto_side(dev144, 1.0 * est["atr_wilder(현행)"])
    a3 = {}
    for bname, (L, S) in bars.items():
        a3[bname] = {int(y): score(side, L, S, days, yr == y, rng, ci=False)
                     for y in np.unique(yr)}
    print("\n[A3] 현행(N=144,K=1.0) 연도별 R1 (전표본, TRAIN 포함)")
    for bname, per in a3.items():
        print(f"  {bname}: " + "  ".join(f"{y} {v['롱']:+5.1f}/{v['숏']:+5.1f}" for y, v in per.items()))

    (OUT / "part_a_band.json").write_text(json.dumps(
        {"estimators": a1, "sweep": a2, "project_estimators": a4, "per_year_current": a3,
         "note": "R1 = mean_bp(측면 s | veto=s) - mean_bp(측면 s | veto=-s), 일블록 부트스트랩"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def _selfcheck():
    # 배리어: TP · SL · 동시터치(SL 우선) · 꼬리 NaN
    c = np.array([100.0, 100, 100, 100, 100.0])
    h = np.array([100.0, 100, 101.6, 101.6, 100.0])
    l = np.array([100.0, 100, 100, 99.2, 100.0])
    global MAXH
    old, MAXH = MAXH, 2
    try:
        r = barrier_bp(h, l, c, long=True)
        assert abs(r[0] - 150.0) < 1e-6, r          # t=0: t+2 봉에서 TP
        assert abs(r[1] - 150.0) < 1e-6, r          # t=1: t+1 봉에서 TP
        assert abs(r[2] - (-70.0)) < 1e-6, r        # t=2: t+1 봉이 TP·SL 동시 -> SL 우선
        assert np.isnan(r[3]) and np.isnan(r[4]), r  # 꼬리 MAXH 개
    finally:
        MAXH = old
    # 히스테리시스: 밴드 안은 직전 유지
    s = veto_side(np.array([0.02, 0.0, -0.02, 0.0]), np.full(4, 0.01))
    assert list(s) == [1, 1, -1, -1], s


if __name__ == "__main__":
    _selfcheck()
    raise SystemExit(main())
