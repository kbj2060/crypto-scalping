#!/usr/bin/env python3
"""보류 축 재개 — **변동성·비유동성이 되돌림을 강화하는가** (2026-09-11). 모델 없음.

사용자 2026-09-07: *"변동성 레인지는 나중에 내가 다시 연구할 수 있도록 기록해두고 다음에 알려줘."*
→ 2026-09-11 *"계속 끝까지 진행해줘"*.

## 기록된 재개 조건 3개 중 2개를 이제 충족한다
① **`atr_pct` 통제 후 증분을 봐라** — 이 축은 이미 아는 크기 축일 가능성이 크다
   (`atr_pct` 단독이 크기 라벨을 OOS AUC 0.708 로 맞히고 배포 사이징이 이미 그걸 쓴다).
   ⇒ 시험 변수를 **추세 실현변동성에 횡단면 잔차화**한 뒤에만 쓴다.
② **OOS 표본을 늘려라**(OOS n=107~340, AUC SE 0.055/0.031 이라 0.056 편차를 못 갈랐다).
   ⇒ [179종 × 2021-12~2026-08 확장 패널](xsec_crowding_extended_sample_decayed_20260910.md)로 **10배 이상**.
③ 타깃을 크기·손절폭으로 바꿔라 — 🔴**이건 폐기한다.** 2026-09-11 산술로 닫혔다:
   `순익=(2a−1)b−비용` 에서 a≈0.5 면 크기 타깃은 이익을 못 낸다
   ([[eth_volforecast_directional_cost_closed_20260911]]).

## 기록된 가설을 그대로 검정한다 (새로 만들지 않는다)
> *"변동성·비유동성↑ → 되돌림, 펀딩압력·OI·주문흐름 독성↑ → 지속"*
되돌림 쪽 단일피쳐 상위가 전부 변동성·유동성·스프레드였다(`parkinson_vol` .442/.446 ·
`bb_width` .452/.435 · `amihud_illiquidity_z` .470/.428 — 전부 0.5 **아래** = 되돌림 방향).

## 사전등록 (결과 전 고정)
설계  매 시각 종목을 시험변수 **3분위**로 나누고, 각 분위 안에서 **되돌림 포트폴리오**
      (직전 L시간 패자 롱 / 승자 숏, k=3)를 만든다.
방향  **되돌림 수익이 고변동성·고비유동성 분위에서 더 크다** — 사후 반전 금지.
1차   **단조성**: 상위분위 − 하위분위 되돌림 수익이 양수이고 일군집 부트 CI 가 0 배제.
      구간은 EXT/IN/OUT 각각. 풀링은 보조.
통제  시험변수를 매 시각 `rv288`(추세 24시간 실현변동성 = atr_pct 대응)에 **횡단면 회귀한 잔차**로 대체.
      ⇒ "이미 아는 크기 축" 을 제거한 뒤 남는 것만 본다.
비용  12bp 횡단면. 단조성 차이는 양다리가 상쇄되므로 **총수익 기준**으로 보고 순수익도 병기.
출력  tmp/xsec_volilliq_20260911/report.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "tmp/xsec_perp_screen_ext_20260910"
OUT = ROOT / "tmp/xsec_volilliq_20260911"
HORIZ = (12, 48, 288)                 # 1시간 · 4시간 · 24시간 (5분봉 단위)
K = 3
DV_MIN = 5e7
EXT_A, EXT_END = "2022-02-01", "2023-12-30"
OOS_A, OOS_B = "2025-09-01", "2026-07-31"
COST_BP = 12.0
BOOT = 4000
SEED = 20260911


def log(m):
    print(f"[vi {time.strftime('%H:%M:%S')}] {m}", flush=True)


def day_ci(vals, days, b=BOOT, seed=SEED):
    u = np.unique(days)
    if len(u) < 20:
        return [float("nan")] * 2, len(u)
    idx = {d: np.flatnonzero(days == d) for d in u}
    rng = np.random.default_rng(seed); o = []
    for _ in range(b):
        pick = rng.choice(u, len(u), replace=True)
        v = np.concatenate([vals[idx[d]] for d in pick])
        if len(v):
            o.append(v.mean())
    return [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))], len(u)


def xsec_resid(x, ctrl):
    """행(시각)마다 x 를 ctrl 에 회귀한 **잔차**. 둘 다 횡단면 순위로 바꿔 이상치에 둔감하게."""
    out = np.full_like(x, np.nan)
    for t in range(x.shape[0]):
        m = np.isfinite(x[t]) & np.isfinite(ctrl[t])
        if m.sum() < 12:
            continue
        a = pd.Series(x[t][m]).rank(pct=True).to_numpy()
        c = pd.Series(ctrl[t][m]).rank(pct=True).to_numpy()
        b = np.cov(c, a, ddof=1)[0, 1] / max(np.var(c, ddof=1), 1e-12)
        out[t, np.flatnonzero(m)] = a - b * c
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(DIR / "panel.npz", allow_pickle=True)
    ts = pd.DatetimeIndex(z["ts"]); C = z["C"].astype(np.float64); Q = z["Q"].astype(np.float64)
    O = z["O"].astype(np.float64)
    log(f"패널 {C.shape} · {ts[0]} → {ts[-1]}")

    r5 = np.full_like(C, np.nan); r5[1:] = np.log(C[1:] / C[:-1])
    RV = lambda w: pd.DataFrame(r5).rolling(w, min_periods=w // 2).std().to_numpy()
    rv288, rv48 = RV(288), RV(48)
    DV = pd.DataFrame(Q).rolling(288, min_periods=200).sum().to_numpy()
    with np.errstate(all="ignore"):
        amihud = pd.DataFrame(np.abs(r5) / np.maximum(Q, 1.0)).rolling(
            288, min_periods=200).mean().to_numpy()
        bbw = (pd.DataFrame(C).rolling(288, min_periods=200).std().to_numpy()
               / np.maximum(pd.DataFrame(C).rolling(288, min_periods=200).mean().to_numpy(), 1e-9))
        rvratio = rv48 / np.maximum(rv288, 1e-12)
    TESTS = {"amihud(비유동성)": amihud, "bb_width(레인지)": bbw, "rv비율(단기/장기)": rvratio}
    log("시험변수 3종 준비 · 통제 = rv288(추세 24시간 실현변동성 ≈ atr_pct)")

    rep = {"prereg": "되돌림 수익이 고변동성·고비유동성 분위에서 더 크다(사후반전 금지) · "
                     "시험변수는 rv288 에 횡단면 잔차화", "cells": {}}
    step = 288                                        # 비겹침 1일 격자
    for name, RAWX in TESTS.items():
        Xr = xsec_resid(RAWX, rv288)
        for H in HORIZ:
            fwd = np.full_like(O, np.nan); fwd[:-(H + 1)] = O[H + 1:] / O[1:-H] - 1.0
            past = np.full_like(C, np.nan); past[H:] = C[H:] / C[:-H] - 1.0
            tid = np.arange(600, len(ts) - H - 2, step)
            el = (DV[tid] >= DV_MIN) & np.isfinite(Xr[tid]) & np.isfinite(fwd[tid]) & np.isfinite(past[tid])
            xr = np.where(el, Xr[tid], np.nan); pa = np.where(el, past[tid], np.nan)
            fw = np.where(el, fwd[tid], np.nan)
            rows, tert = [], {}
            for q, lo_p, hi_p in (("저", 0.0, 1 / 3), ("중", 1 / 3, 2 / 3), ("고", 2 / 3, 1.0)):
                vals = []
                for i in range(len(tid)):
                    v = xr[i]; m = np.isfinite(v)
                    if m.sum() < 3 * (2 * K + 2):
                        vals.append(np.nan); continue
                    r = pd.Series(v[m]).rank(pct=True).to_numpy()
                    sel = np.flatnonzero(m)[(r > lo_p) & (r <= hi_p)]
                    p_ = pa[i][sel]; f_ = fw[i][sel]
                    ok = np.isfinite(p_) & np.isfinite(f_)
                    if ok.sum() < 2 * K + 2:
                        vals.append(np.nan); continue
                    sel, p_, f_ = sel[ok], p_[ok], f_[ok]
                    o = np.argsort(p_)
                    # 되돌림: 직전 패자 롱 − 직전 승자 숏
                    vals.append((f_[o[:K]].mean() - f_[o[-K:]].mean()) / 2 * 1e4)
                tert[q] = np.array(vals)
            hi, lo = tert["고"], tert["저"]
            diff = hi - lo
            days = pd.DatetimeIndex(ts[tid]).floor("D").astype("int64").to_numpy()
            for seg, m in (("EXT", (ts[tid] >= EXT_A) & (ts[tid] <= EXT_END)),
                           ("IN", (ts[tid] > EXT_END) & (ts[tid] < OOS_A)),
                           ("OUT", (ts[tid] >= OOS_A) & (ts[tid] <= OOS_B)),
                           ("ALL", (ts[tid] >= EXT_A) & (ts[tid] <= OOS_B))):
                mm = np.asarray(m) & np.isfinite(diff)
                if mm.sum() < 60:
                    continue
                ci, nd = day_ci(diff[mm], days[mm])
                cell = {"n": int(mm.sum()), "n_days": nd,
                        "hi_bp": float(np.nanmean(hi[mm])), "lo_bp": float(np.nanmean(lo[mm])),
                        "mid_bp": float(np.nanmean(tert["중"][mm])),
                        "diff_bp": float(np.nanmean(diff[mm])), "diff_ci95": ci,
                        "ci_excludes_zero": bool(np.isfinite(ci[0]) and ci[0] > 0),
                        "hi_net_bp": float(np.nanmean(hi[mm]) - COST_BP),
                        "monotone": bool(np.nanmean(tert["저"][mm]) <= np.nanmean(tert["중"][mm])
                                         <= np.nanmean(hi[mm]))}
                rep["cells"][f"{name}|H{H//12}h|{seg}"] = cell
    log("=" * 108)
    log(f"{'셀':>34} {'n':>5} {'저':>8} {'중':>8} {'고':>8} {'고−저':>8} {'일군집 CI95':>20} {'단조':>5}")
    for k, c in rep["cells"].items():
        log(f"{k:>34} {c['n']:>5} {c['lo_bp']:>+8.1f} {c['mid_bp']:>+8.1f} {c['hi_bp']:>+8.1f} "
            f"{c['diff_bp']:>+8.1f} [{c['diff_ci95'][0]:>+7.1f},{c['diff_ci95'][1]:>+7.1f}] "
            f"{'예' if c['monotone'] else '—':>5}" + ("  ⭐CI0배제" if c["ci_excludes_zero"] else ""))
    per = {s: [k for k, c in rep["cells"].items() if k.endswith("|" + s) and c["ci_excludes_zero"]]
           for s in ("EXT", "IN", "OUT", "ALL")}
    rep["summary"] = {s: len(v) for s, v in per.items()}
    log("=" * 108)
    log("⭐1차 기준(고−저 CI 0배제): " + " · ".join(f"{s} {len(v)}" for s, v in per.items()))
    trio = [k.rsplit("|", 1)[0] for k in per["OUT"]
            if k.rsplit("|", 1)[0] + "|IN" in per["IN"] and k.rsplit("|", 1)[0] + "|EXT" in per["EXT"]]
    log(f"⭐⭐**세 구간 전부 통과: {len(trio)}건** {trio}")
    rep["summary"]["three_window_pass"] = trio
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
