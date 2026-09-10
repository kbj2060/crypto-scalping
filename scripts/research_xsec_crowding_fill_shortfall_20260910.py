#!/usr/bin/env python3
"""쏠림 페이드 **체결 실측** — 실제 가격 경로에서 잰 체결 부족분 (2026-09-10). 모델 없음.

사용자: *"하나씩 진행해줘"* — 남은 레버 1순위. 09-08 기록의 *"다음에 할 것(신호 탐색 아님)"* 그대로다:
**막힌 지점이 체결이다. `꼬리에서 실제로 채울 수 있는가`를 재야 한다.**

## 왜 다시 재는가 — 09-08 판정은 도구가 틀렸다
`research_xsec_crowding_realistic_cost_20260908.py` 는 Corwin-Schultz 로 스프레드를 추정했다.
그 스크립트 자신이 *"CS 수준값은 과대추정이므로 상대 순서 진단으로만 쓰고 판정은 상수 비용으로"* 라고 적었고,
기억에도 **"Corwin-Schultz 로 비용을 재지 말 것 — 암호화폐에서 CS 는 변동성을 잰다"** 로 남아 있다.
⇒ 스프레드 **추정기를 쓰지 않는다.** 실제 체결 가능한 가격(거래량가중평균)으로 직접 재고 차액을 본다.

## 무엇을 재는가
연구 규약은 **봉 시가 한 점**에 들어가고 나온다(`O[t+1]` → `O[t+H+1]`). 실거래는 그 한 점을 못 받는다.
주문은 창에 걸쳐 채워지므로 **W봉 거래량가중평균가(VWAP)** 가 현실적인 체결가다.
  체결 부족분 = (시가 한 점 규약 수익) − (VWAP 규약 수익)
VWAP 는 `quote_volume / volume` 을 창 합계로 낸다. 추정이 아니라 **실제 체결된 값**이다.

## ⭐진짜 질문은 꼬리다
손익의 94% 가 **그날 크게 움직인 몇 종목**에서 나온다. 그 종목은 정확히 빠르게 움직이는 중이라
창에 걸쳐 채우면 불리한 가격을 받는다. 그래서 **상위 5% 날과 나머지를 갈라서** 부족분을 따로 낸다.
⚠️꼬리 의존 자체는 가짜의 증거가 아니다(양의 왜도 전략은 다 그렇다).
   문제는 **체결 위험이 꼬리에 몰린다**는 것이고, 이 스크립트가 그걸 정면으로 잰다.

## 참여율도 같이 낸다
명목 $N 을 창 안의 실제 달러거래량으로 나눈 값. **10% 를 넘으면 VWAP 자체가 낙관**이다
(내 주문이 그 VWAP 를 움직인다). 넘는 구간은 판정에서 제외하지 않고 **표시**한다.

## 사전등록 (결과 전 고정)
관문0 **기준선 재현** — H=1일·k=3·NU=40·비겹침 표본외에서 +37.0bp [+11.9,+63.3] 근방이 안 나오면
      원인을 찾기 전까지 이 스크립트의 다른 숫자는 무효로 본다.
1차   VWAP 규약 순수익(수수료 10bp 차감)의 블록부트 CI 하한 > 0
보조   창 길이별 부족분 · 꼬리/비꼬리 부족분 격차 · 참여율
출력  tmp/xsec_fill_20260910/report.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = Path("/home/kbj20/crypto-scalping/tmp/xsec_perp_screen_20260908")
KDIR = Path("/home/kbj20/crypto-scalping/binance_data/klines")
OUT = ROOT / "tmp/xsec_fill_20260910"
CACHE = OUT / "vwap_panel.npz"

H_GRID = (288, 1440)         # 1일(헤드라인) · **5일(동결 설계)**
# ⭐5일을 따로 재야 하는 이유: 체결 부족분은 **진입 한 번에 붙는 고정 손실**이라
# 보유가 길수록 더 긴 수익에 상각된다. 1일에서 엣지를 죽인 창이 5일에선 살 수 있다.
# ⚠️단 비겹침 5일 표본은 n≈67 뿐이다(판정 필요 98) — 이 팔은 **측정이지 판정이 아니다**.
K_GRID = (3, 5)
NU_GRID = (30, 40)
WINDOWS = (1, 3, 6, 12, 72, 288)   # 5분·15분·30분·1시간·**6시간(동결 설계의 스태거)**·**하루**
# ⭐72/288 이 핵심이다. 참여율 1% 로 쓸 만한 명목($4M/다리)을 채우려면 하루가 걸리는데,
# 손익이 나오는 꼬리 날은 **바로 그 하루 동안 움직이는 중**이다. 짧은 창만 재면 이 상충을 못 본다.
NOTIONALS = (1e6, 5e6, 25e6)
FEE_RT = 10.0                # 테이커 왕복 단위명목당
LIQW = 288
OOS_A, OOS_B = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-07-31 23:59:59")
BOOT = 4000
SEED = 20260910


def log(m):
    print(f"[fill {time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_vwap_panel(ts, syms):
    """봉별 VWAP(=quote_volume/volume) 과 달러거래량 패널. 추정이 아니라 실제 체결값."""
    if CACHE.exists():
        z = np.load(CACHE)
        log(f"VWAP 패널 캐시 사용 {z['V'].shape}")
        return z["V"], z["D"]
    n_t, n_s = len(ts), len(syms)
    V = np.full((n_t, n_s), np.nan, np.float32)
    D = np.full((n_t, n_s), np.nan, np.float32)
    for j, s in enumerate(syms):
        d = pd.read_csv(KDIR / s / f"{s}-5m-api.csv",
                        usecols=["timestamp", "volume", "quote_volume"], parse_dates=["timestamp"])
        d = d.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")
        d = d.reindex(ts)
        vol = d["volume"].to_numpy(float); qv = d["quote_volume"].to_numpy(float)
        with np.errstate(all="ignore"):
            V[:, j] = np.where(vol > 0, qv / vol, np.nan)     # 봉 VWAP
        D[:, j] = qv
        if (j + 1) % 20 == 0:
            log(f"  VWAP 패널 {j+1}/{n_s}")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, V=V, D=D)
    return V, D


def win_vwap(V, D, start, w):
    """start 부터 w봉의 거래량가중평균가. 가중치는 달러거래량."""
    n_t = V.shape[0]
    e = min(start + w, n_t)
    if e <= start:
        return None, None
    v = V[start:e]; d = D[start:e]
    num = np.nansum(np.where(np.isfinite(v) & np.isfinite(d), v * d, 0.0), axis=0)
    den = np.nansum(np.where(np.isfinite(v) & np.isfinite(d), d, 0.0), axis=0)
    with np.errstate(all="ignore"):
        return np.where(den > 0, num / den, np.nan), den


def block_ci(x, block=5, b=BOOT, seed=SEED):
    if len(x) < 20:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed); n = len(x); o = []
    for _ in range(b):
        st = rng.integers(0, n, int(np.ceil(n / block)))
        o.append(x[np.concatenate([np.arange(s, s + block) % n for s in st])[:n]].mean())
    return [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))]


def run_one(H, z, ts, O, Q, syms, V, D, S, liq, rep_all):
    tid = np.arange(LIQW + 1, len(ts) - H - max(WINDOWS) - 2, H)   # 비겹침 격자
    tt = ts[tid]
    oos = (tt >= OOS_A) & (tt <= OOS_B)
    log(f"재조정 시점 {len(tid)} · 표본외 {int(oos.sum())}")

    rep = rep_all[f"H{H//288}d"] = {"H": H, "windows": WINDOWS, "fee_rt_bp": FEE_RT, "notionals": NOTIONALS,
           "prereg": "관문0 = 헤드라인 +37.0bp [+11.9,+63.3] 재현", "cells": {}}

    # 규약별 진입/청산 가격 사전 계산
    px = {"open": (O[tid + 1], O[tid + H + 1])}
    part = {}
    for w in WINDOWS:
        ein = np.stack([win_vwap(V, D, t + 1, w)[0] for t in tid])
        eout = np.stack([win_vwap(V, D, t + H + 1, w)[0] for t in tid])
        din = np.stack([win_vwap(V, D, t + 1, w)[1] for t in tid])
        px[f"vwap{w}"] = (ein, eout)
        part[f"vwap{w}"] = din
        log(f"  체결가 규약 vwap{w} 준비")

    for conv, (pin, pout) in px.items():
        with np.errstate(all="ignore"):
            fwd = pout / pin - 1.0
        for NU in NU_GRID:
            el = (liq[tid] < NU) & np.isfinite(S[tid]) & np.isfinite(fwd)
            sa = np.where(el, S[tid], np.nan); F = np.where(el, fwd, np.nan)
            nval = np.isfinite(sa).sum(1)
            order = np.argsort(np.where(np.isfinite(sa), sa, np.inf), 1)
            for k in K_GRID:
                gd = nval >= 2 * k + 2
                rr = np.flatnonzero(gd & oos)
                if len(rr) < 50:
                    continue
                lo_i = order[rr][:, :k]                       # 롱숏비 낮음(숏 쏠림) → 롱
                hi_i = order[rr][np.arange(len(rr))[:, None],
                                 (nval[rr][:, None] - 1 - np.arange(k)[None, :])]
                fl = np.take_along_axis(F[rr], lo_i, 1); fh = np.take_along_axis(F[rr], hi_i, 1)
                g = (np.nanmean(fl, 1) - np.nanmean(fh, 1)) / 2 * 1e4
                g = g[np.isfinite(g)]
                if len(g) < 50:
                    continue
                net = g - FEE_RT
                cell = {"n": int(len(g)), "gross_bp": float(g.mean()),
                        "net_bp": float(net.mean()), "net_ci95": block_ci(net),
                        "gross_ci95": block_ci(g),
                        "top5_removed_gross": float(g[g < np.percentile(g, 95)].mean())}
                if conv.startswith("vwap"):
                    dv = part[conv][rr]
                    sel = np.concatenate([np.take_along_axis(dv, lo_i, 1),
                                          np.take_along_axis(dv, hi_i, 1)], axis=1)
                    med = float(np.nanmedian(sel))
                    cell["window_dollar_vol_median"] = med
                    cell["participation"] = {f"${int(n/1e6)}M": round(n / max(med, 1), 4)
                                             for n in NOTIONALS}
                rep["cells"][f"{conv}|NU{NU}|k{k}"] = cell
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))

    # ── 관문0: 헤드라인 재현 ──────────────────────────────────────────────
    base = rep["cells"].get("open|NU40|k3") if H == 288 else None
    log("=" * 92)
    if H != 288:
        log(f"관문0 미적용 — 헤드라인 +37.0bp 는 **1일 보유** 수치다. "
            f"보유 {H//288}일 팔은 비겹침 n 이 1/{H//288} 로 줄어 판정이 아니라 측정이다.")
    if base:
        ok = 11.0 <= base["gross_ci95"][0] or 20 <= base["gross_bp"] <= 60
        log(f"관문0 재현: 총 {base['gross_bp']:+.1f}bp CI[{base['gross_ci95'][0]:+.1f},"
            f"{base['gross_ci95'][1]:+.1f}] n={base['n']} → 기대 +37.0 [+11.9,+63.3] "
            f"{'✅재현' if ok else '🔴불일치 — 아래 수치 무효'}")
        rep["gate0_reproduced"] = bool(ok)
    log("=" * 92)
    log(f"{'규약':>9}{'NU':>4}{'k':>3} {'총bp':>8} {'순bp':>8} {'순 CI95':>20} "
        f"{'상5%제거 총':>11} {'참여율@$5M':>11}")
    for key, c in rep["cells"].items():
        conv, nu, k = key.split("|")
        p = c.get("participation", {}).get("$5M")
        log(f"{conv:>9}{nu[2:]:>4}{k[1:]:>3} {c['gross_bp']:>+8.1f} {c['net_bp']:>+8.1f} "
            f"[{c['net_ci95'][0]:>+7.1f},{c['net_ci95'][1]:>+7.1f}] "
            f"{c['top5_removed_gross']:>+11.1f} {(f'{p:.1%}' if p else '—'):>11}")
    # ── 사전등록 보조 ①: **꼬리/비꼬리 부족분 격차** ─────────────────────────
    # 가설이었던 것: 꼬리 날은 빠르게 움직이는 중이라 창에 걸쳐 채우면 더 불리하다.
    # 꼬리 정의는 **시가 규약 총수익 상위 5%** 로 고정한다(체결 규약과 무관하게 같은 날을 본다).
    NU, k = 40, 3
    el = (liq[tid] < NU) & np.isfinite(S[tid])
    sa = np.where(el, S[tid], np.nan)
    nval = np.isfinite(sa).sum(1); order = np.argsort(np.where(np.isfinite(sa), sa, np.inf), 1)
    rr = np.flatnonzero((nval >= 2 * k + 2) & oos)
    lo_i = order[rr][:, :k]
    hi_i = order[rr][np.arange(len(rr))[:, None], (nval[rr][:, None] - 1 - np.arange(k)[None, :])]

    def leg_ret(pin, pout):
        with np.errstate(all="ignore"):
            f = pout / pin - 1.0
        f = np.where(el, f, np.nan)[rr]
        return (np.nanmean(np.take_along_axis(f, lo_i, 1), 1)
                - np.nanmean(np.take_along_axis(f, hi_i, 1), 1)) / 2 * 1e4

    g_open = leg_ret(*px["open"])
    fin = np.isfinite(g_open)
    thr = np.percentile(g_open[fin], 95)
    tail = fin & (g_open >= thr); body = fin & (g_open < thr)
    rep["tail_split"] = {"tail_def": "시가 규약 총수익 상위 5% 날", "n_tail": int(tail.sum()),
                         "n_body": int(body.sum()), "by_conv": {}}
    for conv in px:
        gv = leg_ret(*px[conv])
        rep["tail_split"]["by_conv"][conv] = {
            "tail_gross": float(np.nanmean(gv[tail])), "body_gross": float(np.nanmean(gv[body])),
            "tail_shortfall_bp": float(np.nanmean(g_open[tail] - gv[tail])),
            "body_shortfall_bp": float(np.nanmean(g_open[body] - gv[body]))}
    log("=" * 92)
    log(f"꼬리 분해 (상위 5% = {int(tail.sum())}일 · 나머지 {int(body.sum())}일) — NU40 k3")
    log(f"{'규약':>9} {'꼬리 총bp':>10} {'몸통 총bp':>10} {'꼬리 부족분':>11} {'몸통 부족분':>11}")
    for conv, v in rep["tail_split"]["by_conv"].items():
        log(f"{conv:>9} {v['tail_gross']:>+10.1f} {v['body_gross']:>+10.1f} "
            f"{v['tail_shortfall_bp']:>+11.2f} {v['body_shortfall_bp']:>+11.2f}")

    # ── 사전등록 보조 ②: **수용량** — 참여율 상한별 최대 명목 ────────────────
    # 창 안 달러거래량의 p% 까지만 먹는다고 할 때 다리당 최대 명목. k=3 이면 총명목은 ×3.
    rep["capacity"] = {}
    log("=" * 92)
    log(f"수용량 — 다리당 최대 명목 (k=3 이면 한쪽 총명목 ×3)")
    log(f"{'체결창':>9} {'창 달러량 중앙':>14} {'참여1%':>12} {'참여5%':>12} {'참여10%':>12}")
    for w in WINDOWS:
        dv = part[f"vwap{w}"][rr]
        sel = np.concatenate([np.take_along_axis(dv, lo_i, 1),
                              np.take_along_axis(dv, hi_i, 1)], axis=1)
        med = float(np.nanmedian(sel))
        rep["capacity"][f"vwap{w}"] = {"window_dollar_vol_median": med,
                                       "max_notional_per_leg": {f"{int(p*100)}%": med * p
                                                                for p in (0.01, 0.05, 0.10)}}
        log(f"{'vwap'+str(w):>9} {med/1e6:>13.2f}M "
            + "".join(f"{med*p/1e3:>11.0f}K" for p in (0.01, 0.05, 0.10)))
    # 하루 전체에 걸쳐 채울 때(=5일 보유·6시간 스태거 설계의 실제 모습)
    dv_day = part["vwap12"][rr] * 24
    sel = np.concatenate([np.take_along_axis(dv_day, lo_i, 1),
                          np.take_along_axis(dv_day, hi_i, 1)], axis=1)
    medd = float(np.nanmedian(sel))
    rep["capacity"]["full_day"] = {"day_dollar_vol_median": medd,
                                   "max_notional_per_leg": {f"{int(p*100)}%": medd * p
                                                            for p in (0.01, 0.05, 0.10)}}
    log(f"{'하루전체':>9} {medd/1e6:>13.2f}M "
        + "".join(f"{medd*p/1e6:>11.2f}M" for p in (0.01, 0.05, 0.10)))
    (OUT / "report.json").write_text(json.dumps(rep_all, ensure_ascii=False, indent=1, default=float))
    return 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(DIR / "panel.npz", allow_pickle=True)
    ts = pd.DatetimeIndex(z["ts"]); O = z["O"].astype(np.float64); Q = z["Q"]
    syms = [str(x) for x in z["syms"]]
    log(f"패널 {O.shape} · {ts[0]} → {ts[-1]}")
    V, D = build_vwap_panel(ts, syms)
    RAW = np.load(DIR / "metrics_panel.npz")["count_toptrader_long_short_ratio"]
    S = np.where(RAW > 0, np.log(np.maximum(RAW, 1e-9)), np.nan)   # ⚠️불리언 대입 함정 회피
    assert np.nanstd(S) > 0.01, "신호가 상수/불리언이다 — := 우선순위 함정 재발"
    Qr = pd.DataFrame(Q).rolling(LIQW, min_periods=LIQW // 2).median().to_numpy()
    liq = np.argsort(np.argsort(-np.nan_to_num(Qr, nan=-1.0), axis=1), axis=1)
    rep_all = {}
    for H in H_GRID:
        log("#" * 92)
        log(f"### 보유 {H//288}일 (H={H}봉)" + ("  ⚠️비겹침 표본 부족 — 측정이지 판정 아님" if H > 288 else ""))
        run_one(H, z, ts, O, Q, syms, V, D, S, liq, rep_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
