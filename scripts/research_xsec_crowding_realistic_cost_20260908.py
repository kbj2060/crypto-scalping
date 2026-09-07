#!/usr/bin/env python3
"""쏠림 페이드에 **종목별 실측 비용**을 물린다 (2026-09-08).

## 왜
지금까지는 비용을 5.5/12bp 상수로 놓았다. 그런데 부록 AH 감사에서 손익의 94% 가
**횡단면 꼬리**(그날 크게 움직인 종목)에서 나왔다. 꼬리 종목은 정확히 스프레드가 넓은 곳이다.
상수 비용 가정이 가장 안 믿기는 지점이므로 **종목별·일별 비용**으로 다시 판정한다.

⚠️1차판 결함 2건 수정: (1) 대입표현식 우선순위로 신호가 불리언이 돼 있었다
   (`mz_ := X > 0` 은 `mz_` 에 **불리언**을 넣는다) (2) **일봉** Corwin-Schultz 는 암호화폐에서
   스프레드가 아니라 변동성을 잰다(BTC 227bp 라는 말이 안 되는 값). 5분봉으로 교체.
   CS 수준값은 여전히 과대추정이므로 **상대 순서 진단**으로만 쓰고, 판정은 상수 비용 사다리로 한다.

## 비용 모형 (표준 수수료 · 할인 가정 금지)
왕복 단위명목당 = **테이커 수수료 10bp**(0.05% × 진입/청산 × 두 다리 / 2단위)
              + **평균 유효 스프레드**(두 다리 평균, Corwin-Schultz 일별 추정)
Corwin-Schultz(2012): 연속 두 봉의 고저 범위에서 유효 스프레드를 추정하는 표준 방법.
음수 추정치는 0 으로 절단(원논문 권고), 종목별 21일 중앙값으로 평활.

## 판정
표본외(2025-09-01~2026-07-31, 비겹침 1일) 순bp = 총수익 − 실측비용.
CI 하한 > 0 이면 통과. 상위 5% 날 제거 후에도 양수인지 병기.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "tmp/xsec_perp_screen_20260908"
KDIR = ROOT / "binance_data/klines"
OOS_A, OOS_B = "2025-09-01", "2026-07-31"
H = 288
K_GRID = (3, 5)
NU_GRID = (20, 30, 40)
FEE_RT = 10.0          # 테이커 왕복 · 단위명목당 (0.05% x 2체결 x 2다리 / 2단위)
LIQW, VOLW = 288, 576
BOOT = 4000
SEED = 20260908


def corwin_schultz(hi, lo):
    """일별 (T,N) 고가/저가 -> 유효 스프레드(비율). 음수는 0 절단."""
    with np.errstate(all="ignore"):
        b1 = np.log(hi / lo) ** 2
        beta = b1[:-1] + b1[1:]
        h2 = np.maximum(hi[:-1], hi[1:]); l2 = np.minimum(lo[:-1], lo[1:])
        gamma = np.log(h2 / l2) ** 2
        k = 3 - 2 * np.sqrt(2)
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        S = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    S = np.where(np.isfinite(S) & (S > 0), S, np.nan)
    out = np.full_like(hi, np.nan); out[1:] = S
    return out


def boot_ci(v, rng, B=BOOT):
    if len(v) < 10: return (np.nan, np.nan)
    return tuple(np.percentile(v[rng.integers(0, len(v), (B, len(v)))].mean(1), [2.5, 97.5]))


def main() -> int:
    rng = np.random.default_rng(SEED)
    z = np.load(DIR / "panel.npz", allow_pickle=True)
    ts = pd.to_datetime(z["ts"]); Om = z["O"]; Cm = z["C"]; Qm = z["Q"]; syms = list(z["syms"])
    # 일별 고저 (5분봉에서 집계)
    hi5 = {}; lo5 = {}
    for s in syms:
        f = KDIR / s / f"{s}-5m-api.csv"
        d = pd.read_csv(f, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
        d = d.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")
        d = d.reindex(ts)
        hi5[s] = d["high"].astype("float32"); lo5[s] = d["low"].astype("float32")
    Hd = pd.DataFrame(hi5); Ld = pd.DataFrame(lo5)
    print(f"5분봉 {Hd.shape} (일봉 CS 는 암호화폐에서 변동성만 잰다 -- 5분봉 사용)", flush=True)
    CS = corwin_schultz(Hd.to_numpy(), Ld.to_numpy())
    CSs = pd.DataFrame(CS, index=Hd.index, columns=Hd.columns).rolling(2016, min_periods=200).median()
    print("Corwin-Schultz 유효 스프레드(bp) -- 유동성 순위대별 중앙값:", flush=True)
    Qr = pd.DataFrame(Qm).rolling(LIQW, min_periods=LIQW // 2).median().to_numpy()
    liq = np.argsort(np.argsort(-np.nan_to_num(Qr, nan=-1.0), axis=1), axis=1)
    # 봉 격자로 되돌리기
    SPR = CSs.to_numpy() * 1e4
    for a, b in ((0, 10), (10, 20), (20, 30), (30, 40), (40, 60)):
        m = (liq >= a) & (liq < b)
        print(f"   순위 {a:>2}~{b:<2}: 중앙 {np.nanmedian(np.where(m, SPR, np.nan)):>6.1f}bp · "
              f"90분위 {np.nanpercentile(np.where(m, SPR, np.nan), 90):>6.1f}bp", flush=True)

    RAW = np.load(DIR / "metrics_panel.npz")["count_toptrader_long_short_ratio"]
    S = np.where(RAW > 0, np.log(np.maximum(RAW, 1e-9)), np.nan)
    lr = np.full_like(Cm, np.nan); lr[1:] = np.log(Cm[1:] / Cm[:-1])
    vol = pd.DataFrame(lr).rolling(VOLW, min_periods=VOLW // 2).std().to_numpy()
    fwd = np.full_like(Om, np.nan); fwd[:-(H + 1)] = Om[H + 1:] / Om[1:-H] - 1.0
    tid = np.arange(max(LIQW, VOLW) + 1, len(ts) - H - 2, H)
    tt = ts[tid]
    seg = np.where(tt < OOS_A, "IN", np.where(tt <= OOS_B + " 23:59:59", "OUT", "X"))

    print("\n" + "=" * 116)
    print(f"{'NU':>4}{'k':>3}{'가중':>7}{'구간':>5} {'n':>4} {'총bp':>8} {'실측비용':>9} "
          f"{'순bp [CI95]':>24} {'상5%제거 순':>11} {'상수12 순':>9}")
    print("=" * 116)
    npass = 0
    for NU in NU_GRID:
        el = (liq[tid] < NU) & np.isfinite(S[tid]) & np.isfinite(fwd[tid]) & np.isfinite(vol[tid]) \
             & np.isfinite(SPR[tid])
        sa = np.where(el, S[tid], np.nan); F = np.where(el, fwd[tid], np.nan)
        V = np.where(el, vol[tid], np.nan); P = np.where(el, SPR[tid], np.nan)
        nval = np.isfinite(sa).sum(1); order = np.argsort(np.where(np.isfinite(sa), sa, np.inf), 1)
        for k in K_GRID:
            gd = nval >= 2 * k + 2
            rr = np.flatnonzero(gd)
            if len(rr) < 50: continue
            lo_i = order[rr][:, :k]
            hi_i = order[rr][np.arange(len(rr))[:, None],
                             (nval[rr][:, None] - 1 - np.arange(k)[None, :])]
            fl = np.take_along_axis(F[rr], lo_i, 1); fh = np.take_along_axis(F[rr], hi_i, 1)
            sl = np.take_along_axis(P[rr], lo_i, 1); sh = np.take_along_axis(P[rr], hi_i, 1)
            wl = 1 / np.maximum(np.take_along_axis(V[rr], lo_i, 1), 1e-9)
            wh = 1 / np.maximum(np.take_along_axis(V[rr], hi_i, 1), 1e-9)
            cost_eq = FEE_RT + (np.nanmean(sl, 1) + np.nanmean(sh, 1)) / 2.0
            cost_vw = FEE_RT + ((sl * wl).sum(1) / wl.sum(1) + (sh * wh).sum(1) / wh.sum(1)) / 2.0
            arms = {"동일": ((fl.mean(1) - fh.mean(1)) / 2 * 1e4, cost_eq),
                    "1/vol": (((fl * wl).sum(1) / wl.sum(1) - (fh * wh).sum(1) / wh.sum(1)) / 2 * 1e4,
                              cost_vw)}
            sg = seg[rr]
            for an, (port, cost) in arms.items():
                for s_ in ("IN", "OUT"):
                    m = (sg == s_) & np.isfinite(port) & np.isfinite(cost)
                    if m.sum() < 25: continue
                    g = port[m]; c = cost[m]; net = g - c
                    lo, hi = boot_ci(net, rng)
                    t5 = net[g < np.percentile(g, 95)].mean()
                    print(f"{NU:>4}{k:>3}{an:>7}{s_:>5} {m.sum():>4} {g.mean():>+8.1f} "
                          f"{c.mean():>9.1f} {net.mean():>+8.1f}[{lo:>+6.1f},{hi:>+6.1f}] "
                          f"{t5:>+11.1f} {g.mean()-12:>+9.1f}", flush=True)
                    if s_ == "OUT" and lo > 0: npass += 1
    print("\n" + "=" * 116)
    print(f"⭐표본외에서 **실측비용 차감 후 CI 하한 > 0**: {npass}건")
    print(json.dumps({"pass": npass}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
