#!/usr/bin/env python3
"""횡단면 마지막 변형 -- **변동성 정규화 순위**와 **포지셔닝 수준(쏠림)** (2026-09-08).

## 두 가지 미탐색 변형
1. ⭐**변동성 정규화**: 지금까지의 되돌림 신호는 **원시 수익률**로 순위를 매겼다.
   그러면 순위 자체가 "변동성이 큰 종목"을 뽑는 쪽으로 오염된다. 표준적 처리는
   과거수익을 그 종목의 변동성으로 나눈 뒤 순위를 매기는 것이다(z 형태).
   1.24bp 가 과소추정이었는지 확인하는 유일하게 남은 방법론적 개선.
2. ⭐**수준(level) vs 변화(change)**: 09-08 메트릭 스크린은 전부 **변화량**이었다.
   쏠림 가설은 **수준**이다 -- 개미/상위트레이더 롱숏비가 지금 가장 높은 종목이 미달성과.
   펀딩 **수준**도 같은 계열(고펀딩 = 롱 쏠림).

신호 9종 × L4 × H4 × k2 × NU2. 진입 open[t+1] · 청산 open[t+1+H] · 겹침 재조정(step=H/2) ·
단위명목당 bp · 네 창 일군집 CI · 판정 = 네 창 CI 하한 > 비용(5.5/12bp).
우연 기대치를 반드시 병기한다(네 창 부호 일관 = 셀수/8).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "tmp/xsec_perp_screen_20260908"
FUND = ROOT / "tmp/xsec_funding_carry_20260908/funding.parquet"
SPLITS = {"TRAIN": ("2024-01-01", "2025-08-31"), "VAL": ("2025-09-01", "2025-12-31"),
          "OOS": ("2026-01-01", "2026-03-31"), "HOLDOUT_SPENT": ("2026-04-01", "2026-07-31")}
L_GRID = (6, 12, 48, 144)
H_GRID = (3, 6, 12, 48)
K_GRID = (3, 5)
NU_GRID = (20, 40)
VOLW = 576
LIQW = 288
BOOT = 3000
SEED = 20260908
COSTS = (5.5, 12.0)


def day_ci(v, day, rng, B=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8 or len(v) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def main() -> int:
    rng = np.random.default_rng(SEED)
    z = np.load(DIR / "panel.npz", allow_pickle=True)
    ts = pd.to_datetime(z["ts"]); Om = z["O"]; Cm = z["C"]; Qm = z["Q"]; syms = list(z["syms"])
    mz = np.load(DIR / "metrics_panel.npz", allow_pickle=True)
    OI = mz["sum_open_interest"]; OIV = mz["sum_open_interest_value"]
    TTc = mz["count_toptrader_long_short_ratio"]; TTp = mz["sum_toptrader_long_short_ratio"]
    RT = mz["count_long_short_ratio"]
    F = pd.read_parquet(FUND).reindex(columns=syms)
    Fw = F.reindex(ts, method="ffill").to_numpy(np.float32)      # 마지막 정산 요율(인과적)
    print(f"패널 {Om.shape[0]:,}봉 × {len(syms)}종목 · 펀딩 ffill 커버 "
          f"{np.isfinite(Fw).mean():.1%}", flush=True)
    lr = np.full_like(Cm, np.nan); lr[1:] = np.log(Cm[1:] / Cm[:-1])
    vol = pd.DataFrame(lr).rolling(VOLW, min_periods=VOLW // 2).std().to_numpy()
    Qr = pd.DataFrame(Qm).rolling(LIQW, min_periods=LIQW // 2).median().to_numpy()
    liq = np.argsort(np.argsort(-np.nan_to_num(Qr, nan=-1.0), axis=1), axis=1)
    day_all = ts.floor("D").to_numpy()
    win_of = np.full(len(ts), "", object)
    for w, (a, b) in SPLITS.items():
        win_of[(ts >= a) & (ts <= b + " 23:59:59")] = w
    turn = np.where(OIV > 0, Qm / np.maximum(OIV, 1e-9), np.nan)

    rows = []
    for L in L_GRID:
        past = np.full_like(Cm, np.nan); past[L:] = Cm[L:] / Cm[:-L] - 1.0
        with np.errstate(all="ignore"):
            pz = past / np.maximum(vol * np.sqrt(L), 1e-9)
            pz[~np.isfinite(pz)] = np.nan
            lg = lambda X: np.where(X > 0, np.log(np.maximum(X, 1e-9)), np.nan)
            d_rt = np.full_like(RT, np.nan); d_rt[L:] = lg(RT)[L:] - lg(RT)[:-L]
            d_ttc = np.full_like(TTc, np.nan); d_ttc[L:] = lg(TTc)[L:] - lg(TTc)[:-L]
        SIG = {"ret_raw": past, "ret_volnorm": pz,
               "retail_level": lg(RT), "retail_chg": d_rt,
               "tt_count_level": lg(TTc), "tt_count_chg": d_ttc,
               "tt_pos_level": lg(TTp), "funding_level": Fw, "turnover": lg(turn)}
        for H in H_GRID:
            fwd = np.full_like(Om, np.nan); fwd[:-(H + 1)] = Om[H + 1:] / Om[1:-H] - 1.0
            tid = np.arange(max(L, VOLW) + 1, len(ts) - H - 2, max(H // 2, 1))
            for sname, S in SIG.items():
                for NU in NU_GRID:
                    el = (liq[tid] < NU) & np.isfinite(S[tid]) & np.isfinite(fwd[tid])
                    sa = np.where(el, S[tid], np.nan); fw = np.where(el, fwd[tid], np.nan)
                    nval = np.isfinite(sa).sum(1)
                    order = np.argsort(np.where(np.isfinite(sa), sa, np.inf), axis=1)
                    for k in K_GRID:
                        gd = nval >= 2 * k + 2
                        if gd.sum() < 300: continue
                        rr = np.flatnonzero(gd)
                        lo_i = order[rr][:, :k]
                        hi_i = order[rr][np.arange(len(rr))[:, None],
                                         (nval[rr][:, None] - 1 - np.arange(k)[None, :])]
                        port = (np.take_along_axis(fw[rr], lo_i, 1).mean(1)
                                - np.take_along_axis(fw[rr], hi_i, 1).mean(1)) / 2.0 * 1e4
                        ww = win_of[tid][rr]; dd = day_all[tid][rr]
                        rec = dict(sig=sname, L=L, H=H, k=k, NU=NU)
                        ok = True
                        for w in SPLITS:
                            m = (ww == w) & np.isfinite(port)
                            if m.sum() < 100: ok = False; break
                            lo, hi = day_ci(port[m], dd[m], rng)
                            rec[f"{w}_g"] = float(port[m].mean()); rec[f"{w}_lo"] = lo
                            rec[f"{w}_hi"] = hi; rec[f"{w}_n"] = int(m.sum())
                        if ok: rows.append(rec)
        print(f"  L={L} 완료 (셀 {len(rows)})", flush=True)
    R = pd.DataFrame(rows); R.to_csv(DIR / "normalized_crowding.csv", index=False)
    print(f"\n셀 {len(R):,}\n", flush=True)
    print("=== 신호별 TRAIN 평균 총수익 (bp/재조정) ===")
    print(R.pivot_table(index="sig", columns="H", values="TRAIN_g").round(2).to_string())
    print("\n=== 신호별 네 창 CI 하한이 0 을 배제한 셀 수 (부호 무관) ===")
    R["ci0"] = [all(R.loc[i, f"{w}_lo"] > 0 for w in SPLITS)
                or all(R.loc[i, f"{w}_hi"] < 0 for w in SPLITS) for i in R.index]
    print(R.groupby("sig")["ci0"].agg(["sum", "count"]).to_string())
    for C in COSTS:
        a = sum(all(R.loc[i, f"{w}_lo"] > C for w in SPLITS) for i in R.index)
        b = sum(all(R.loc[i, f"{w}_hi"] < -C for w in SPLITS) for i in R.index)
        print(f"\n⭐비용 {C}bp 네 창 통과: 정방향 {a}/{len(R)} · 뒤집기 {b}/{len(R)}")
    sg = np.sign(R[[f"{w}_g" for w in SPLITS]])
    cons = sg.nunique(axis=1) == 1
    print(f"\n네 창 부호 일관 {int(cons.sum())}/{len(R)} (우연 기대 {len(R)/8:.0f})")
    Rc = R[cons].copy(); Rc["m"] = Rc[[f"{w}_g" for w in SPLITS]].abs().min(1)
    print("=== |네 창 최소| 상위 12 ===")
    print(Rc.sort_values("m", ascending=False).head(12)
          [["sig", "L", "H", "k", "NU", "TRAIN_g", "TRAIN_lo", "TRAIN_hi", "VAL_g", "OOS_g",
            "HOLDOUT_SPENT_g", "TRAIN_n"]].round(2).to_string(index=False))
    print("\n=== ret_raw vs ret_volnorm 직접 비교 (같은 L,H,k,NU) ===")
    cmpd = R[R.sig.isin(("ret_raw", "ret_volnorm"))].pivot_table(
        index=["L", "H", "k", "NU"], columns="sig", values="TRAIN_g")
    if len(cmpd):
        cmpd["차이"] = cmpd["ret_volnorm"] - cmpd["ret_raw"]
        print(cmpd.round(2).to_string())
        print(f"정규화가 더 큰 셀: {int((cmpd['차이']>0).sum())}/{len(cmpd)} · "
              f"평균 차이 {cmpd['차이'].mean():+.2f}bp")
    print(json.dumps({"cells": len(R)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
