#!/usr/bin/env python3
"""**횡단면(cross-sectional) 5분봉 롱숏 스크린** -- 59개 바이낸스 무기한 (2026-09-08).

## 왜 새 축인가
이 저장소의 2년치 작업은 전부 **단일자산 시계열**(ETH 사건 발동 -> 방향 예측)이었고,
2026-09-08 확인에서 사건군 전체가 네 창 모두 순 −8~−14bp 로 닫혔다
(`research_eth_anchor_bothside_train_confirm_20260908.py`, TRAIN n=16,091 CI [−10.8,−6.2]).
총수익 ≈ 0 이므로 방향 정보 자체가 없다.

횡단면은 구조적으로 다르다:
- **롱숏 동수 = 시장 베타가 설계상 제거**된다. 09-08 에 발견한 "측면 거울상 잔존 베타" 함정
  ([[feedback_side_mirror_excess_is_residual_beta_not_signal_20260908]])이 원리적으로 생기지 않는다.
- 예측 대상이 "가격이 오르나"가 아니라 "**A 가 B 보다 더 오르나**" 로 바뀐다.
- 데이터는 이미 있다: `binance_data/klines/*/[SYM]-5m-api.csv` 59종 · 2024-01-01~2026-08-04.

## 설계 (모델 없음)
- 신호 = 과거 L 봉 수익률의 횡단면 순위. **하위 k 롱 · 상위 k 숏**(= 되돌림). 부호를 뒤집으면 추세추종.
- 진입 `open[t+1]`, 청산 `open[t+1+H]` (인과적, 겹치지 않게 H 봉마다 재조정).
- 총명목 2단위 -> **단위명목당 bp** = (롱평균 − 숏평균)/2 × 1e4. 비용도 단위명목당 왕복 `COST`.
- 유니버스 = 그 시점 직전 288봉 quote_volume 중앙값 상위 NU 종목(생존편향 주의: 상장폐지 종목 없음).
- 창 = 앵커 파이프라인과 동일 경계. **TRAIN 에서 고르고 VAL/OOS/HOLDOUT 으로 확인한다.**
- 귀무 = 같은 시점·같은 유니버스에서 롱숏을 **무작위 배정**(B회), 일군집 CI.
"""
from __future__ import annotations
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KDIR = ROOT / "binance_data/klines"
OUT = ROOT / "tmp/xsec_perp_screen_20260908"
SPLITS = {"TRAIN": ("2024-01-01", "2025-08-31"), "VAL": ("2025-09-01", "2025-12-31"),
          "OOS": ("2026-01-01", "2026-03-31"), "HOLDOUT_SPENT": ("2026-04-01", "2026-07-31")}
L_GRID = (3, 6, 12, 24, 48, 144, 288)
H_GRID = (3, 6, 12, 24, 48, 144)
K_GRID = (5, 10)
NU_GRID = (20, 40, 59)
COST = 7.8
LIQW = 288
BOOT = 2000
SEED = 20260908


PANEL = OUT / "panel.npz"


def load_panel():
    if PANEL.exists():
        z = np.load(PANEL, allow_pickle=True)
        ix = pd.to_datetime(z["ts"]); sy = list(z["syms"])
        return (pd.DataFrame(z["O"], index=ix, columns=sy),
                pd.DataFrame(z["C"], index=ix, columns=sy),
                pd.DataFrame(z["Q"], index=ix, columns=sy))
    files = sorted(glob.glob(str(KDIR / "*/*-5m-api.csv")))
    op, cl, qv = {}, {}, {}
    for f in files:
        sym = Path(f).parent.name
        d = pd.read_csv(f, usecols=["timestamp", "open", "close", "quote_volume"],
                        parse_dates=["timestamp"])
        d = d.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")
        op[sym] = d["open"].astype("float32"); cl[sym] = d["close"].astype("float32")
        qv[sym] = d["quote_volume"].astype("float32")
    O = pd.DataFrame(op).sort_index(); C = pd.DataFrame(cl).sort_index(); Q = pd.DataFrame(qv).sort_index()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PANEL, O=O.to_numpy(), C=C.to_numpy(), Q=Q.to_numpy(),
                        ts=O.index.to_numpy(), syms=np.array(list(O.columns)))
    return O, C, Q


def day_ci(v, day, rng, B=BOOT):
    """일군집 부트스트랩 -- 일별 합/개수만 재표집(수학적으로 동일, O(B x 일수))."""
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8 or len(v) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u))
    c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    o = s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0)
    return tuple(np.percentile(o, [2.5, 97.5]))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED)
    print("[1/3] 패널 로드 ...", flush=True)
    O, C, Q = load_panel()
    ts = O.index.to_numpy(); syms = list(O.columns)
    Om = O.to_numpy(); Cm = C.to_numpy(); Qm = Q.to_numpy()
    print(f"      {Om.shape[0]:,}봉 × {len(syms)}종목 · {ts[0]} ~ {ts[-1]}", flush=True)
    # 유동성 순위 (직전 288봉 중앙값, 인과적: t 까지만)
    print("[2/3] 유동성 순위 ...", flush=True)
    Qr = pd.DataFrame(Qm).rolling(LIQW, min_periods=LIQW // 2).median().to_numpy()
    liq_rank = np.argsort(np.argsort(-np.nan_to_num(Qr, nan=-1.0), axis=1), axis=1)  # 0=최대
    day_all = pd.Series(ts).dt.floor("D").to_numpy()
    win_of = np.full(len(ts), "", object)
    for w, (a, b) in SPLITS.items():
        m = (ts >= np.datetime64(a)) & (ts <= np.datetime64(b + "T23:59:59"))
        win_of[m] = w

    print("[3/3] 격자 ...", flush=True)
    rows = []
    for L in L_GRID:
        past = np.full_like(Cm, np.nan)
        past[L:] = Cm[L:] / Cm[:-L] - 1.0
        for H in H_GRID:
            fwd = np.full_like(Om, np.nan)
            fwd[:-(H + 1)] = Om[H + 1:] / Om[1:-H] - 1.0     # open[t+1] -> open[t+1+H]
            t0 = max(L, LIQW) + 1
            tidx = np.arange(t0, len(ts) - H - 2, H)          # 겹치지 않는 재조정
            for NU in NU_GRID:
                elig = (liq_rank[tidx] < NU) & np.isfinite(past[tidx]) & np.isfinite(fwd[tidx])
                pa = np.where(elig, past[tidx], np.nan)
                fw = np.where(elig, fwd[tidx], np.nan)
                nval = np.isfinite(pa).sum(1)
                order = np.argsort(np.where(np.isfinite(pa), pa, np.inf), axis=1)
                for k in K_GRID:
                    good = nval >= 2 * k + 2
                    if good.sum() < 200: continue
                    rr = np.arange(len(tidx))[good]
                    lo_i = order[good][:, :k]                                  # 하위 k = 최저 과거수익
                    hi_i = order[good][:, :][np.arange(len(rr))[:, None],
                                             (nval[good][:, None] - 1 - np.arange(k)[None, :])]
                    fl = np.take_along_axis(fw[good], lo_i, 1).mean(1)
                    fh = np.take_along_axis(fw[good], hi_i, 1).mean(1)
                    port = (fl - fh) / 2.0 * 1e4                                # 되돌림 부호
                    tt = tidx[good]
                    for w in SPLITS:
                        m = win_of[tt] == w
                        if m.sum() < 60: continue
                        v = port[m]; d = day_all[tt][m]
                        lo, hi = day_ci(v, d, rng)
                        rows.append(dict(L=L, H=H, k=k, NU=NU, win=w, n=int(m.sum()),
                                         gross=float(v.mean()), lo=lo, hi=hi,
                                         ndays=int(len(np.unique(d))),
                                         per_day=float(v.mean()) * (288 / H)))
    R = pd.DataFrame(rows); R.to_csv(OUT / "grid.csv", index=False)
    print(f"      셀 {len(R):,}\n", flush=True)

    P = R.pivot_table(index=["L", "H", "k", "NU"], columns="win", values=["gross", "lo", "hi"])
    tr = P[("gross", "TRAIN")]
    # TRAIN 에서 |총수익| 이 비용을 넘고 CI 가 0 을 배제하는 후보만 확인창으로 보낸다
    cand = P[(P[("lo", "TRAIN")] > COST) | (P[("hi", "TRAIN")] < -COST)]
    print(f"⭐TRAIN 에서 |총수익| CI 가 비용 {COST}bp 를 배제한 후보: {len(cand)}/{len(P)}")
    if len(cand):
        c = cand.copy(); c["absT"] = c[("gross", "TRAIN")].abs()
        c = c.sort_values("absT", ascending=False)
        print(f"\n{'L':>4}{'H':>5}{'k':>4}{'NU':>4} | {'TRAIN':>22} | {'VAL':>22} | {'OOS':>22} | {'HOLDOUT':>22}")
        conf = []
        for kk, r in c.head(25).iterrows():
            sgn = np.sign(r[("gross", "TRAIN")])
            line = f"{kk[0]:>4}{kk[1]:>5}{kk[2]:>4}{kk[3]:>4} | "
            allok = True
            for w in ("TRAIN", "VAL", "OOS", "HOLDOUT_SPENT"):
                g, lo, hi = r[("gross", w)], r[("lo", w)], r[("hi", w)]
                ok = (lo > COST) if sgn > 0 else (hi < -COST)
                allok &= bool(ok)
                line += f"{g:>+7.2f}[{lo:>+6.1f},{hi:>+6.1f}]{'o' if ok else 'x'} | "
            print(line, flush=True)
            if allok: conf.append(kk)
        print(f"\n⭐네 창 모두 비용 배제: {len(conf)}건  {conf}")
    else:
        b = P.reindex(tr.abs().sort_values(ascending=False).index).head(10)
        print("\nTRAIN 총수익 상위 10 (비용 미달, 참고):")
        print(b[[("gross", w) for w in ("TRAIN", "VAL", "OOS", "HOLDOUT_SPENT")]].round(2).to_string())
    print(json.dumps({"cells": len(R), "train_candidates": len(cand)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
