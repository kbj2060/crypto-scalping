#!/usr/bin/env python3
"""횡단면 **포지셔닝·자금흐름 신호** 스크린 (2026-09-08).

## 왜
09-08 밤에 가격 기반 횡단면(되돌림·과잉반응·극단이탈·다일 모멘텀)과 펀딩 캐리를 전부 닫았다.
남은 미탐색 자원은 `binance_data/metrics/` — **60종 × 945일 × 5분봉**의 미결제약정·
상위트레이더 롱숏비·개미 롱숏비·테이커 매수매도비. 이 저장소에서 한 번도 횡단면으로 쓰인 적이 없다.

⭐근거: ETH 단일자산 경제축 스크린(09-04)에서 유일하게 새 축으로 살아남은 것이 **retail_shift**
(개미 롱숏비 급변의 반대) 였다 — 세 창 양수, 기존 규칙에 +3~7bp/일.
그 축을 **횡단면**으로 옮기면 (a) 베타가 설계상 제거되고 (b) 60종으로 검정력이 커진다.

## 신호 (전부 t 까지의 정보만)
- `d_oi`      : 미결제약정 L봉 변화율
- `d_retail`  : 개미 롱숏비(count_long_short_ratio) L봉 로그 변화  ← retail_shift 의 횡단면판
- `d_tt`      : 상위트레이더 포지션 롱숏비 L봉 로그 변화
- `taker`     : 테이커 매수/매도 거래량비 L봉 평균의 로그
- `oi_x_ret`  : d_oi × 과거수익 부호 (신규 진입 vs 청산 구분)
각 신호를 횡단면 순위 -> 하위 k 롱 / 상위 k 숏. 부호는 표에서 뒤집어 읽는다.
진입 `open[t+1]` · 청산 `open[t+1+H]` · 단위명목당 bp · 비용 5.5/12bp.
판정: 네 창(TRAIN/VAL/OOS/HOLDOUT) CI 하한 > 비용.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "tmp/xsec_perp_screen_20260908"
SPLITS = {"TRAIN": ("2024-01-01", "2025-08-31"), "VAL": ("2025-09-01", "2025-12-31"),
          "OOS": ("2026-01-01", "2026-03-31"), "HOLDOUT_SPENT": ("2026-04-01", "2026-07-31")}
L_GRID = (12, 48, 144, 288)
H_GRID = (12, 48, 144)
K_GRID = (3, 5)
NU_GRID = (20, 40)
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


def logdiff(X, L):
    Y = np.full_like(X, np.nan)
    with np.errstate(all="ignore"):
        Y[L:] = np.log(np.maximum(X[L:], 1e-12)) - np.log(np.maximum(X[:-L], 1e-12))
    Y[~np.isfinite(Y)] = np.nan
    return Y


def main() -> int:
    rng = np.random.default_rng(SEED)
    z = np.load(DIR / "panel.npz", allow_pickle=True)
    ts = pd.to_datetime(z["ts"]); Om = z["O"]; Cm = z["C"]; Qm = z["Q"]; syms = list(z["syms"])
    mz = np.load(DIR / "metrics_panel.npz", allow_pickle=True)
    OI = mz["sum_open_interest"]; TT = mz["sum_toptrader_long_short_ratio"]
    RT = mz["count_long_short_ratio"]; TK = mz["sum_taker_long_short_vol_ratio"]
    print(f"패널 {Om.shape[0]:,}봉 × {len(syms)}종목 · OI 커버 {np.isfinite(OI).mean():.1%}", flush=True)
    Qr = pd.DataFrame(Qm).rolling(LIQW, min_periods=LIQW // 2).median().to_numpy()
    liq = np.argsort(np.argsort(-np.nan_to_num(Qr, nan=-1.0), axis=1), axis=1)
    day_all = ts.floor("D").to_numpy()
    win_of = np.full(len(ts), "", object)
    for w, (a, b) in SPLITS.items():
        win_of[(ts >= a) & (ts <= b + " 23:59:59")] = w

    rows = []
    for L in L_GRID:
        pret = np.full_like(Cm, np.nan); pret[L:] = Cm[L:] / Cm[:-L] - 1.0
        d_oi = logdiff(OI, L); d_rt = logdiff(RT, L); d_tt = logdiff(TT, L)
        tk = pd.DataFrame(TK).rolling(L, min_periods=max(2, L // 2)).mean().to_numpy()
        with np.errstate(all="ignore"):
            tk = np.log(np.maximum(tk, 1e-12)); tk[~np.isfinite(tk)] = np.nan
        SIG = {"d_oi": d_oi, "d_retail": d_rt, "d_toptrader": d_tt, "taker": tk,
               "oi_x_ret": d_oi * np.sign(pret)}
        for H in H_GRID:
            fwd = np.full_like(Om, np.nan); fwd[:-(H + 1)] = Om[H + 1:] / Om[1:-H] - 1.0
            tid = np.arange(max(L, LIQW) + 1, len(ts) - H - 2, H)
            for sname, S in SIG.items():
                for NU in NU_GRID:
                    el = ((liq[tid] < NU) & np.isfinite(S[tid]) & np.isfinite(fwd[tid]))
                    sa = np.where(el, S[tid], np.nan); fw = np.where(el, fwd[tid], np.nan)
                    nval = np.isfinite(sa).sum(1)
                    order = np.argsort(np.where(np.isfinite(sa), sa, np.inf), axis=1)
                    for k in K_GRID:
                        gd = nval >= 2 * k + 2
                        if gd.sum() < 200: continue
                        rr = np.flatnonzero(gd)
                        lo_i = order[rr][:, :k]
                        hi_i = order[rr][np.arange(len(rr))[:, None],
                                         (nval[rr][:, None] - 1 - np.arange(k)[None, :])]
                        port = (np.take_along_axis(fw[rr], lo_i, 1).mean(1)
                                - np.take_along_axis(fw[rr], hi_i, 1).mean(1)) / 2.0 * 1e4
                        ww = win_of[tid][rr]; dd = day_all[tid][rr]
                        rec = dict(sig=sname, L=L, H=H, k=k, NU=NU, n=int(np.isfinite(port).sum()))
                        ok = True
                        for w in SPLITS:
                            m = (ww == w) & np.isfinite(port)
                            if m.sum() < 60: ok = False; break
                            lo, hi = day_ci(port[m], dd[m], rng)
                            rec[f"{w}_g"] = float(port[m].mean()); rec[f"{w}_lo"] = lo
                            rec[f"{w}_hi"] = hi; rec[f"{w}_n"] = int(m.sum())
                        if ok: rows.append(rec)
        print(f"  L={L} 완료 (셀 {len(rows)})", flush=True)
    R = pd.DataFrame(rows); R.to_csv(DIR / "metrics_signals.csv", index=False)
    print(f"\n셀 {len(R):,}\n", flush=True)
    for C in COSTS:
        a = R[[all(R.loc[i, f"{w}_lo"] > C for w in SPLITS) for i in R.index]]
        b = R[[all(R.loc[i, f"{w}_hi"] < -C for w in SPLITS) for i in R.index]]
        print(f"⭐비용 {C}bp -- 양(하위k 롱) 통과 {len(a)}/{len(R)} · 음(뒤집기) 통과 {len(b)}/{len(R)}")
        for nm, t in (("정방향", a), ("뒤집기", b)):
            if len(t):
                print(f"  [{nm}]")
                print(t[["sig", "L", "H", "k", "NU", "n"] + [f"{w}_g" for w in SPLITS]
                        + [f"{w}_lo" for w in SPLITS]].round(1).to_string(index=False))
    sg = np.sign(R[[f"{w}_g" for w in SPLITS]])
    cons = sg.nunique(axis=1) == 1
    Rc = R[cons].copy(); Rc["m"] = Rc[[f"{w}_g" for w in SPLITS]].abs().min(1)
    print(f"\n네 창 부호 일관 {int(cons.sum())}/{len(R)} (우연 기대 {len(R)/8:.0f})")
    print("=== |네 창 최소| 상위 15 ===")
    print(Rc.sort_values("m", ascending=False).head(15)
          [["sig", "L", "H", "k", "NU", "TRAIN_g", "TRAIN_lo", "TRAIN_hi",
            "VAL_g", "OOS_g", "HOLDOUT_SPENT_g", "TRAIN_n"]].round(2).to_string(index=False))
    print("\n=== 신호별 TRAIN 평균 총수익 ===")
    print(R.groupby(["sig", "H"])["TRAIN_g"].mean().unstack().round(2).to_string())
    print(json.dumps({"cells": len(R)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
