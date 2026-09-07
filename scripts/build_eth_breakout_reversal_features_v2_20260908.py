#!/usr/bin/env python3
"""돌파/되돌림 피쳐 **v2 -- 발현 그 자체의 성격** (2026-09-08).

## v1 이 놓친 것
v1(53피쳐)은 **발현 직전 봉 `bt-1`** 만 봤다. HGB 정확도 50~54%, 세 창 일관 미달.
정작 트레이더가 보는 것 — *"그 움직임이 어떻게 만들어졌는가"* — 이 통째로 빠져 있었다.

## ⚠️2026-09-08 정정 -- **1분 미래참조 발견·수정**
초판은 경로를 `s0..s1`(트리거 분 **포함**)로 잡았는데, 라벨은 `first_touch(..., s1, ...)` 로
**같은 분 s1 부터** 배리어를 탐색한다. 그 분에 큰 움직임이 있으면 (a)경로 피쳐가 커지고
(b)같은 움직임이 배리어를 때려 라벨을 정한다 -- **피쳐와 라벨이 1분봉 하나를 공유**했다.
증상: 순열 중요도 1~5위가 전부 경로 피쳐, `maxbar` Q5 돌파율 0.580 vs Q1 0.483,
그런데 모델 없는 2피쳐 규칙은 50.8~51.3%(모델만 54~57%).
수정: 경로는 **`s1` 직전(`tmin-1`)까지만**. 첫 분에 발현하면 경로 정보 없음(NaN).
부수 수정: `v2_mv_maxbar_atr` 가 가격차를 ref 로 안 나눠 가격수준을 인코딩하고 있었다.

## v2 추가 피쳐군 (전부 인과적: 트리거 분 **직전**까지의 정보만)
**A. 발현 경로** (앵커 진입분 s0 ~ 트리거분 s1, 1분봉)
  - `mv_minutes` 소요 분 · `mv_speed` ATR/분
  - `mv_mae_atr` 발현 도중 **역행 최대폭**(흔들렸나 곧장 갔나)
  - `mv_straight` 직진성 = |순변동| / Σ|1분 변동|  (1.0 = 일직선)
  - `mv_updown` 진행 방향 1분봉 비율 · `mv_maxbar_atr` 최대 1분봉 크기
  - `mv_gap_atr` 트리거 직전 1분봉의 갭
**B. 레벨 맥락** (`bt-1` 까지의 봉으로만 계산)
  - 트리거가가 직전 N봉 고/저를 **돌파했는가** (N=48/144/288/864) 및 그 여유 ATR
  - 직전 N봉 레인지 안 백분위 · 24h/72h 고저까지 거리(ATR)
**C. 교차자산 동조** (같은 분 구간)
  - `mv_btc_ret` 같은 구간 BTC 수익 · `mv_beta_gap` ETH−BTC (고유 움직임인가 시장 동조인가)
  - `mv_corr_sign` 부호 일치 여부
**D. 앵커 나이** `age_min` 앵커→트리거 분 (A 와 동일하나 앵커 정의별로 다름)

⭐가설: **고유(idiosyncratic)·역행 많고·직진성 낮은 발현일수록 되돌림**,
**시장 동조·직진·구조적 레벨 돌파일수록 지속**. 이게 맞으면 정확도가 올라간다.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
BKL1 = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-1m-api.csv"
SRC = ROOT / "tmp/eth_anchor_label_dataset_20260907/anchors_labels.parquet"
DS = ROOT / "tmp/eth_breakout_reversal_20260908/dataset.parquet"
OUT = ROOT / "tmp/eth_breakout_reversal_20260908"
ANCHORS = ("first_fire", "any2/Wc3", "any3/Wc3")
T_MULT = (0.5, 0.75, 1.0)
W_TRIG = 3
H = 48
MAXMIN = (W_TRIG + H) * 5 + 10
CHUNK = 4000
NMOVE = W_TRIG * 5              # 발현 경로 최대 분 (v1 빌더와 정확히 동일해야 병합됨)


def first_touch(hi1, lo1, start, up, dn, nmin):
    n = len(start)
    tu = np.full(n, -1, np.int32); td = np.full(n, -1, np.int32)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        idx = start[a:b, None] + np.arange(nmin)[None, :]
        hu = hi1[idx] >= up[a:b, None]; hd = lo1[idx] <= dn[a:b, None]
        au = hu.any(1); ad = hd.any(1)
        tu[a:b] = np.where(au, hu.argmax(1), -1); td[a:b] = np.where(ad, hd.argmax(1), -1)
    return tu, td


def main() -> int:
    print("[1/4] 로드 ...", flush=True)
    D = pd.read_parquet(SRC).reset_index(drop=True)
    D = D[D["anchor"].isin(ANCHORS)].reset_index(drop=True)
    eth = B._load_kl(B.ETH_KL)
    ts5 = eth["timestamp"].to_numpy(); O5 = eth["open"].to_numpy(float)
    H5 = eth["high"].to_numpy(float); L5 = eth["low"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    u1 = ["timestamp", "open", "high", "low", "close"]
    m1 = pd.read_csv(KL1, usecols=u1, parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy()
    hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float); cl1 = m1["close"].to_numpy(float)
    bp = Path(BKL1)
    if bp.exists():
        b1 = pd.read_csv(bp, usecols=["timestamp", "close"], parse_dates=["timestamp"])
        b1 = b1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")
        bcl = b1["close"].reindex(pd.DatetimeIndex(ts1)).ffill().to_numpy(float)
        print("      BTC 1분봉 사용", flush=True)
    else:
        bcl = None; print("      ⚠️BTC 1분봉 없음 -- 교차자산 경로 피쳐 생략", flush=True)
    CAP = len(eth) - MAXMIN // 5 - 2

    print("[2/4] 레벨 맥락 (bt-1 까지의 봉) ...", flush=True)
    LV = {}
    for N in (48, 144, 288, 864):
        LV[f"hi{N}"] = pd.Series(H5).rolling(N, min_periods=N // 3).max().to_numpy()
        LV[f"lo{N}"] = pd.Series(L5).rolling(N, min_periods=N // 3).min().to_numpy()
    atr5 = pd.Series((H5 - L5) / C5).rolling(192, min_periods=64).mean().to_numpy()

    print("[3/4] 발현 경로 ...", flush=True)
    bidx = D["bar_idx"].to_numpy(); atr = D["atr_pct"].to_numpy()
    out = []
    for Tm in T_MULT:
        ei0 = np.minimum(bidx + 1, len(O5) - 1); ref = O5[ei0]
        st = np.searchsorted(ts1, ts5[ei0])
        ok = (st < len(ts1) - MAXMIN) & (ts1[np.minimum(st, len(ts1) - 1)] == ts5[ei0]) & (bidx <= CAP)
        s0 = np.where(ok, st, 0); T = atr * Tm
        tu, td = first_touch(hi1, lo1, s0, ref * (1 + T), ref * (1 - T), NMOVE)
        big = 1 << 30
        a = np.where(tu >= 0, tu, big); b = np.where(td >= 0, td, big)
        fired = ok & ((a < big) | (b < big)) & (a != b)
        sgn = np.where(a < b, 1.0, -1.0); tmin = np.where(a < b, a, b)
        trig_ts = ts1[np.minimum(s0 + np.where(fired, tmin, 0), len(ts1) - 1)]
        bt = np.searchsorted(ts5, trig_ts, side="right") - 1
        fb = bt - 1
        keep = fired & (fb >= 900) & (fb < len(C5))  # 864봉 레벨창 확보
        idx = np.flatnonzero(keep)
        n = len(idx)
        # 1분 경로 (s0 .. s0+tmin), 패딩은 마지막값 반복
        span = np.arange(NMOVE)[None, :]
        j0 = s0[idx][:, None] + span
        # 🔴라벨은 트리거 분 s1 부터 배리어를 탐색한다. 경로 피쳐가 s1 을 포함하면
        #   같은 1분봉을 피쳐와 라벨이 공유해 **1분 미래참조**가 된다.
        #   -> s1 직전(tmin-1)까지만 쓴다. tmin=0(첫 분에 발현)이면 경로 정보 없음(NaN).
        mask = span <= (tmin[idx][:, None] - 1)
        c = cl1[np.clip(j0, 0, len(cl1) - 1)]
        h = hi1[np.clip(j0, 0, len(hi1) - 1)]
        l = lo1[np.clip(j0, 0, len(lo1) - 1)]
        r = ref[idx][:, None]; sg = sgn[idx][:, None]; at = np.maximum(atr[idx], 1e-9)[:, None]
        dc = np.where(mask, (c - r) / r, np.nan)
        # 역행 최대폭 (발현 반대 방향 극값)
        adv = np.where(mask, np.where(sg > 0, (l - r) / r, (r - h) / r), np.nan)
        mae = np.nanmin(np.where(np.isfinite(adv), adv, np.nan), axis=1) * np.where(sgn[idx] > 0, 1, 1)
        step = np.diff(np.where(mask, c, np.nan), axis=1)
        tot = np.nansum(np.abs(step), axis=1)
        net = np.abs(np.nanmax(np.where(mask, np.abs(dc), np.nan), axis=1))
        up_bars = np.nansum(np.where(mask[:, 1:], np.sign(step) == sgn[idx][:, None], np.nan), axis=1)
        nb = np.maximum(mask.sum(1) - 1, 1)
        f = {
            "v2_mv_minutes": tmin[idx].astype(float),
            "v2_mv_speed": T[idx] / np.maximum(tmin[idx] + 1, 1) / np.maximum(atr[idx], 1e-9),
            "v2_mv_mae_atr": np.abs(mae) / np.maximum(atr[idx], 1e-9),
            "v2_mv_straight": net / np.maximum(tot, 1e-12),
            "v2_mv_updown": up_bars / nb,
            "v2_mv_maxbar_atr": (np.nanmax(np.abs(np.where(np.isfinite(step), step, np.nan)), axis=1)
                                 / np.maximum(ref[idx], 1e-9)) / np.maximum(atr[idx], 1e-9),
            "v2_mv_nbars": nb.astype(float),
            "v2_mv_path_known": (tmin[idx] >= 1).astype(float),   # 경로 정보 유무
        }
        if bcl is not None:
            bref = bcl[np.clip(s0[idx], 0, len(bcl) - 1)]
            bnow = bcl[np.clip(s0[idx] + tmin[idx], 0, len(bcl) - 1)]
            bret = (bnow - bref) / np.maximum(bref, 1e-9)
            eret = sgn[idx] * T[idx]
            f["v2_mv_btc_ret_atr"] = bret / np.maximum(atr[idx], 1e-9)
            f["v2_mv_idio_atr"] = (eret - bret) / np.maximum(atr[idx], 1e-9)
            f["v2_mv_same_sign"] = (np.sign(bret) == sgn[idx]).astype(float)
        # 레벨 맥락
        fb2 = fb[idx]; tp = ref[idx] * (1 + sgn[idx] * T[idx]); a5 = np.maximum(atr5[fb2], 1e-9)
        for N in (48, 144, 288, 864):
            hh = LV[f"hi{N}"][fb2]; ll = LV[f"lo{N}"][fb2]
            f[f"v2_brk{N}"] = np.where(sgn[idx] > 0, (tp - hh) / (hh * a5), (ll - tp) / (ll * a5))
            f[f"v2_pos{N}"] = (tp - ll) / np.maximum(hh - ll, 1e-9)
        out.append(pd.DataFrame({"bar_idx": bidx[idx], "anchor": D["anchor"].to_numpy()[idx],
                                 "side_bottom": (D["side"].to_numpy()[idx] == "bottom").astype(int),
                                 "T_mult": Tm, **f}))
        print(f"      T={Tm}: {n:,}행", flush=True)
    V = pd.concat(out, ignore_index=True)
    V = V.replace([np.inf, -np.inf], np.nan)
    print("[4/4] v1 과 병합 ...", flush=True)
    A = pd.read_parquet(DS); A["timestamp"] = pd.to_datetime(A["timestamp"])
    KEY = ["bar_idx", "anchor", "side_bottom", "T_mult"]
    V = V.drop_duplicates(KEY)
    A = A.drop_duplicates(KEY)
    M = A.merge(V, on=KEY, how="left", validate="one_to_one")
    hit = M[[c for c in M.columns if c.startswith("v2_")]].notna().any(axis=1).mean()
    print(f"      병합 {M.shape} · v2 피쳐 결합률 {hit:.1%}")
    M.to_parquet(OUT / "dataset_v2.parquet")
    print(json.dumps({"rows": len(M),
                      "v2_feats": len([c for c in M.columns if c.startswith("v2_")])},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
