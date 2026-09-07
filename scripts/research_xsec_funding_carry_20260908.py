#!/usr/bin/env python3
"""**횡단면 펀딩 캐리** -- 60종 무기한, 8시간마다 극단 펀딩 롱숏 (2026-09-08).

## 왜
방향 축은 닫혔고(부록 AA), 횡단면 가격 되돌림은 실재하나 1~4bp 로 비용 미만(부록 AB/AC).
펀딩 캐리는 **가격 방향을 맞출 필요가 없는 수취 흐름**이고, 8시간 보유라 건당 비용이 크게 희석된다.
바이낸스 알트 펀딩은 극단에서 8시간당 ±5~30bp 로 비용(5.5~12bp)보다 한 자릿수 크다.

## 인과적 설계 (중요)
`calc_time` 에서 관측되는 `last_funding_rate` 는 **그 시점에 정산된** 요율이다.
- 결정: 시각 T 에 **정산 완료된** f(T) 로 순위를 매긴다.
- 진입: T 직후 5분봉 시가. 청산: T+8h 직후 5분봉 시가.
- 수취: 그 구간 끝에 정산되는 **f(T+8h)** — 결정 시점에 모르는 값이다(미래참조 아님).
⇒ 롱은 −f(T+8h), 숏은 +f(T+8h) 를 받는다. 캐리 = (−f_lo + f_hi)/2 (단위명목당).

총수익 = 가격 + 캐리. 비용 5.5(전메이커)/7.8(배포)/12(소형주) 왕복.
유니버스 = 직전 288 5분봉 quote_volume 중앙값 상위 NU.
판정: **네 창(TRAIN/VAL/OOS/HOLDOUT) 모두 일군집 CI 하한 > 비용.**
"""
from __future__ import annotations
import io, glob, json, os, zipfile
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAN = ROOT / "tmp/xsec_perp_screen_20260908/panel.npz"
FDIRS = (ROOT / "binance_data/funding_rate_other", ROOT / "binance_data/funding_rate")
OUT = ROOT / "tmp/xsec_funding_carry_20260908"
SPLITS = {"TRAIN": ("2024-01-01", "2025-08-31"), "VAL": ("2025-09-01", "2025-12-31"),
          "OOS": ("2026-01-01", "2026-03-31"), "HOLDOUT_SPENT": ("2026-04-01", "2026-07-31")}
K_GRID = (1, 2, 3, 5, 8)
NU_GRID = (20, 40, 60)
NPER = (1, 3, 9)          # 보유 펀딩주기 수 (8h / 24h / 72h)
LIQW = 288
BOOT = 3000
SEED = 20260908
COSTS = (5.5, 7.8, 12.0)


def load_funding(syms):
    out = {}
    for s in syms:
        fr = []
        for d in FDIRS:
            for f in sorted(glob.glob(str(d / f"{s}-fundingRate-*.zip"))):
                z = zipfile.ZipFile(f)
                for n in z.namelist():
                    if not n.endswith(".csv"): continue
                    fr.append(pd.read_csv(io.BytesIO(z.read(n))))
        if not fr: continue
        d_ = pd.concat(fr, ignore_index=True)
        d_ = d_[pd.to_numeric(d_["calc_time"], errors="coerce").notna()]
        d_["t"] = pd.to_datetime(d_["calc_time"].astype("int64"), unit="ms")
        d_ = d_.sort_values("t").drop_duplicates("t", keep="last")
        out[s] = d_.set_index("t")["last_funding_rate"].astype(float)
    return pd.DataFrame(out).sort_index()


def day_ci(v, day, rng, B=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8 or len(v) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED)
    z = np.load(PAN, allow_pickle=True)
    ts = pd.to_datetime(z["ts"]); Om = z["O"]; Qm = z["Q"]; syms = list(z["syms"])
    print(f"[1/3] 패널 {Om.shape[0]:,}봉 × {len(syms)}종목", flush=True)
    cache = OUT / "funding.parquet"
    if cache.exists():
        F = pd.read_parquet(cache)
    else:
        F = load_funding(syms); F.to_parquet(cache)
    F = F.reindex(columns=syms)
    print(f"      펀딩 {F.shape[0]:,}시점 × {F.notna().any().sum()}종목 · "
          f"{F.index[0]} ~ {F.index[-1]}", flush=True)

    # 펀딩 시각 -> 5분봉 인덱스 (그 시각 이후 첫 봉의 시가로 진입)
    pos = ts.searchsorted(F.index, side="left")
    val = (pos < len(ts) - 1)
    F = F[val]; pos = pos[val]
    Qr = pd.DataFrame(Qm).rolling(LIQW, min_periods=LIQW // 2).median().to_numpy()
    liq = np.argsort(np.argsort(-np.nan_to_num(Qr[pos], nan=-1.0), axis=1), axis=1)
    Fv = F.to_numpy()
    day_f = F.index.floor("D").to_numpy()
    win_of = np.full(len(F), "", object)
    for w, (a, b) in SPLITS.items():
        m = (F.index >= a) & (F.index <= b + " 23:59:59"); win_of[m] = w

    print("[2/3] 격자 ...", flush=True)
    rows = []
    for NP in NPER:
        # 가격: pos[i] 시가 -> pos[i+NP] 시가
        i0 = np.arange(len(F) - NP)
        p_in = Om[pos[i0]]; p_out = Om[pos[i0 + NP]]
        pret = p_out / p_in - 1.0
        # 캐리: 구간 끝 정산분들의 합 (i+1 .. i+NP)
        carry_l = np.zeros_like(pret); 
        for j in range(1, NP + 1):
            carry_l += np.nan_to_num(Fv[i0 + j], nan=0.0)
        fin = np.isfinite(pret) & np.isfinite(Fv[i0])
        for NU in NU_GRID:
            el = fin & (liq[i0] < NU)
            fr = np.where(el, Fv[i0], np.nan)
            nval = np.isfinite(fr).sum(1)
            frs = np.where(np.isfinite(fr), fr, np.inf)
            order = np.argsort(frs, axis=1)
            for k in K_GRID:
                gd = nval >= 2 * k + 2
                if gd.sum() < 200: continue
                rr = np.flatnonzero(gd)
                lo_i = order[rr][:, :k]                                   # 가장 음수 펀딩 = 롱
                hi_i = order[rr][np.arange(len(rr))[:, None],
                                 (nval[rr][:, None] - 1 - np.arange(k)[None, :])]
                pl = np.take_along_axis(np.where(el, pret, np.nan)[rr], lo_i, 1).mean(1)
                ph = np.take_along_axis(np.where(el, pret, np.nan)[rr], hi_i, 1).mean(1)
                cl = np.take_along_axis(np.where(el, carry_l, np.nan)[rr], lo_i, 1).mean(1)
                ch = np.take_along_axis(np.where(el, carry_l, np.nan)[rr], hi_i, 1).mean(1)
                price = (pl - ph) / 2.0 * 1e4
                carry = (-cl + ch) / 2.0 * 1e4
                tot = price + carry
                ww = win_of[i0][rr]; dd = day_f[i0][rr]
                good = np.isfinite(tot)
                rec = dict(NP=NP, NU=NU, k=k, n=int(good.sum()))
                ok = True
                for w in SPLITS:
                    m = good & (ww == w)
                    if m.sum() < 60: ok = False; break
                    lo, hi = day_ci(tot[m], dd[m], rng)
                    rec[f"{w}_g"] = float(tot[m].mean()); rec[f"{w}_lo"] = lo; rec[f"{w}_hi"] = hi
                    rec[f"{w}_carry"] = float(carry[m].mean()); rec[f"{w}_price"] = float(price[m].mean())
                    rec[f"{w}_n"] = int(m.sum())
                if ok: rows.append(rec)
    R = pd.DataFrame(rows); R.to_csv(OUT / "carry.csv", index=False)
    print(f"[3/3] 셀 {len(R):,}\n", flush=True)
    for C in COSTS:
        ok = R[[all(R.loc[i, f"{w}_lo"] > C for w in SPLITS) for i in R.index]]
        print(f"⭐비용 {C}bp: 네 창 모두 CI 하한 > 비용 {len(ok)}/{len(R)}")
        if len(ok):
            o = ok.copy(); o["day_net"] = (o["TRAIN_g"] - C) * (3.0 / o["NP"])
            print(o.sort_values("day_net", ascending=False).head(12)
                  [["NP", "NU", "k", "n"] + [f"{w}_g" for w in SPLITS] + ["day_net"]]
                  .round(2).to_string(index=False))
    print("\n=== 네 창 최소 총수익 상위 12 (캐리/가격 분해) ===")
    mn = R[[f"{w}_g" for w in SPLITS]].min(1)
    b = R.reindex(mn.sort_values(ascending=False).index).head(12)
    print(b[["NP", "NU", "k", "TRAIN_g", "TRAIN_lo", "TRAIN_carry", "TRAIN_price",
             "VAL_g", "OOS_g", "HOLDOUT_SPENT_g", "TRAIN_n"]].round(2).to_string(index=False))
    print(json.dumps({"cells": len(R)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
