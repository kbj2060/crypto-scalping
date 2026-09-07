#!/usr/bin/env python3
"""돌파/되돌림 **v3** -- 관찰창 · OI/자금흐름 · 완화 라벨 (2026-09-08).

사용자 지시 3건 동시 진행:
1. **더 긴 관찰창** -- 트리거 직후 판단하지 말고 OBS 분 더 보고 판단
2. **미결제약정·청산** -- OI 가 늘며 간 움직임인가, 줄며(청산·숏커버) 간 움직임인가
3. **완화 라벨** -- 배리어 P 를 0.5% -> 0.25/0.35% 로 낮춰 해소율↑·잡음↓

## 🔴누수 경계는 관찰창과 함께 움직인다 (부록 AM 교훈)
결정 시점 `s2 = s1 + OBS`. 라벨은 `s2` 부터 배리어 탐색.
⇒ **모든 피쳐는 `s2-1` 까지만.** 5분봉 피쳐는 `s2` 직전에 **완결된** 봉까지만.
   (초판이 `s1` 을 피쳐·라벨이 공유해 정확도가 4pp 부풀었다.)
기준가 = `cl1[s2-1]` (결정 시점에 알려진 마지막 종가). 배리어 = 기준가 × (1 ± P).
라벨 y=1 = **원래 발현 방향**으로 +P 먼저, y=0 = 반대로 먼저, 시간청산은 H봉 뒤 종가 부호.

## 관찰창이 주는 새 정보 (⭐)
OBS 동안 가격이 발현 방향으로 더 갔는가/되돌렸는가, OI 가 늘었는가 등.
"트리거만 보고 예측"에서 "**트리거 + 그 뒤 반응까지 보고 예측**"으로 바뀐다.

## OI/자금흐름 (⚠️한계 명시)
바이낸스 공개 청산 원장은 2024~2026 구간에 없다(`liq_magnet_collector.py` 는 2026-08 가동).
대신 5분봉 메트릭 패널의 **미결제약정**과 klines 의 **테이커 매수/매도 불균형**으로 구성한다:
  - `oi_d` 발현 구간 OI 변화율 · `oi_quad` OI↑/↓ × 가격방향 4분면(신규진입 vs 청산·숏커버)
  - `tk_imb` 테이커 불균형 z · `tk_extreme` 극단 여부 (청산 캐스케이드 대리)
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
XPAN = ROOT / "tmp/xsec_perp_screen_20260908/panel.npz"
XMET = ROOT / "tmp/xsec_perp_screen_20260908/metrics_panel.npz"
OUT = ROOT / "tmp/eth_breakout_reversal_20260908"
ANCHORS = ("first_fire", "any2/Wc3")
T_MULT = 0.75
W_TRIG = 3
OBS_GRID = (0, 5, 10, 15, 30)          # 관찰창 (분)
P_GRID = (0.0025, 0.0035, 0.0050)
H = 48
MAXMIN = (W_TRIG + H) * 5 + max(OBS_GRID) + 20
CHUNK = 4000
SIGNALS = ("sweep", "smt", "taker", "kal", "strz", "orth", "fib", "dem")


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


def zs(x, w):
    s = pd.Series(x)
    return ((s - s.rolling(w, min_periods=w // 3).mean())
            / s.rolling(w, min_periods=w // 3).std().replace(0, np.nan)).to_numpy()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/5] 로드 ...", flush=True)
    D = pd.read_parquet(SRC).reset_index(drop=True)
    D = D[D["anchor"].isin(ANCHORS)].reset_index(drop=True)
    eth = B._load_kl(B.ETH_KL); btc = B._load_kl(B.BTC_KL)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy()
    hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float); cl1 = m1["close"].to_numpy(float)
    b1 = pd.read_csv(BKL1, usecols=["timestamp", "close"], parse_dates=["timestamp"])
    b1 = b1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")
    bcl = b1["close"].reindex(pd.DatetimeIndex(ts1)).ffill().to_numpy(float)
    ts5 = eth["timestamp"].to_numpy()
    O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    H5 = eth["high"].to_numpy(float); L5 = eth["low"].to_numpy(float)
    V5 = eth["volume"].to_numpy(float); TB = eth["taker_buy_base"].to_numpy(float)
    bt5 = btc.set_index("timestamp")["close"].reindex(pd.DatetimeIndex(ts5)).ffill().to_numpy(float)
    CAP = len(eth) - MAXMIN // 5 - 2

    print("[2/5] 5분봉 피쳐 ...", flush=True)
    F = {}
    for k in (3, 6, 12, 48, 144):
        F[f"ret{k}"] = np.concatenate([np.full(k, np.nan), C5[k:] / C5[:-k] - 1.0])
        F[f"btc_ret{k}"] = np.concatenate([np.full(k, np.nan), bt5[k:] / bt5[:-k] - 1.0])
    F["atr_pct"] = pd.Series((H5 - L5) / C5).rolling(192, min_periods=64).mean().to_numpy()
    F["rng_z"] = zs((H5 - L5) / C5, 288); F["vol_z"] = zs(V5, 288)
    tbr = np.where(V5 > 0, TB / np.maximum(V5, 1e-12), np.nan)
    F["taker_buy_ratio"] = tbr; F["taker_z"] = zs(tbr, 288)
    lr = np.full(len(C5), np.nan); lr[1:] = np.log(C5[1:] / C5[:-1])
    F["rv48"] = pd.Series(lr).rolling(48, min_periods=16).std().to_numpy()
    F["rv288"] = pd.Series(lr).rolling(288, min_periods=96).std().to_numpy()
    F["rv_ratio"] = F["rv48"] / np.maximum(F["rv288"], 1e-12)
    for k in (3, 12, 48): F[f"eth_btc_sp{k}"] = F[f"ret{k}"] - F[f"btc_ret{k}"]
    LV = {}
    for N in (48, 144, 288, 864):
        LV[f"hi{N}"] = pd.Series(H5).rolling(N, min_periods=N // 3).max().to_numpy()
        LV[f"lo{N}"] = pd.Series(L5).rolling(N, min_periods=N // 3).min().to_numpy()
    atr5 = F["atr_pct"]

    print("[3/5] ⭐OI·자금흐름 ...", flush=True)
    z = np.load(XPAN, allow_pickle=True); mz = np.load(XMET, allow_pickle=True)
    xts = pd.DatetimeIndex(pd.to_datetime(z["ts"])); ei_s = list(z["syms"]).index("ETHUSDT")
    OI = mz["sum_open_interest"][:, ei_s].astype(float)
    pos = pd.Index(xts).get_indexer(pd.DatetimeIndex(ts5))
    oi5 = np.where(pos >= 0, OI[np.clip(pos, 0, len(OI) - 1)], np.nan)
    for k in (3, 6, 12, 48):
        F[f"oi_d{k}"] = np.concatenate([np.full(k, np.nan), oi5[k:] / np.maximum(oi5[:-k], 1e-9) - 1.0])
    F["oi_z"] = zs(oi5, 288)

    print("[4/5] 트리거·관찰창·라벨 ...", flush=True)
    bidx = D["bar_idx"].to_numpy(); atr = D["atr_pct"].to_numpy()
    ei0 = np.minimum(bidx + 1, len(O5) - 1); ref = O5[ei0]
    st = np.searchsorted(ts1, ts5[ei0])
    ok = (st < len(ts1) - MAXMIN) & (ts1[np.minimum(st, len(ts1) - 1)] == ts5[ei0]) & (bidx <= CAP)
    s0 = np.where(ok, st, 0); T = atr * T_MULT
    tu, td = first_touch(hi1, lo1, s0, ref * (1 + T), ref * (1 - T), W_TRIG * 5)
    big = 1 << 30
    a = np.where(tu >= 0, tu, big); b = np.where(td >= 0, td, big)
    fired = ok & ((a < big) | (b < big)) & (a != b)
    sgn = np.where(a < b, 1.0, -1.0); tmin = np.where(a < b, a, b)
    s1 = s0 + np.where(fired, tmin, 0)

    frames = []
    for OBS in OBS_GRID:
        s2 = s1 + OBS                                       # 결정 분
        dec_ts = ts1[np.clip(s2, 0, len(ts1) - 1)]
        bt = np.searchsorted(ts5, dec_ts, side="right") - 1  # 결정 분이 속한 5분봉
        fb = bt - 1                                          # ⭐완결된 마지막 봉
        keep = fired & (fb >= 900) & (s2 + H * 5 < len(ts1)) & (bt + H < len(C5))
        idx = np.flatnonzero(keep)
        ref2 = cl1[np.clip(s2[idx] - 1, 0, len(cl1) - 1)]     # ⭐s2 직전 종가
        # 라벨: s2 부터 ±P 첫터치
        lab = {}
        for Pv in P_GRID:
            t2u, t2d = first_touch(hi1, lo1, s2[idx], ref2 * (1 + Pv), ref2 * (1 - Pv), H * 5)
            x5 = np.minimum(bt[idx] + H, len(C5) - 1)
            clo = (C5[x5] - ref2) / ref2 * 1e4 * sgn[idx]
            uo = t2u >= 0; do_ = t2d >= 0
            au = np.where(uo, t2u, big); ad = np.where(do_, t2d, big)
            upf = uo & (au < ad); dnf = do_ & (ad < au)
            cont = np.where(sgn[idx] > 0, upf, dnf); rev = np.where(sgn[idx] > 0, dnf, upf)
            lab[f"y_p{int(Pv*10000)}"] = np.where(cont, 1, np.where(rev, 0, (clo > 0).astype(int)))
            lab[f"res_p{int(Pv*10000)}"] = (cont | rev).astype(int)
        # 경로 피쳐: s0 .. s2-1 (⭐결정 분 제외)
        span = np.arange(W_TRIG * 5 + max(OBS_GRID) + 1)[None, :]
        j0 = s0[idx][:, None] + span
        nlast = (s2[idx] - 1 - s0[idx])[:, None]
        mask = (span <= nlast) & (nlast >= 0)
        c = cl1[np.clip(j0, 0, len(cl1) - 1)]
        h = hi1[np.clip(j0, 0, len(hi1) - 1)]; l = lo1[np.clip(j0, 0, len(lo1) - 1)]
        r0 = ref[idx][:, None]; sg = sgn[idx][:, None]
        dc = np.where(mask, (c - r0) / r0, np.nan)
        adv = np.where(mask, np.where(sg > 0, (l - r0) / r0, (r0 - h) / r0), np.nan)
        step = np.diff(np.where(mask, c, np.nan), axis=1)
        tot = np.nansum(np.abs(step), axis=1)
        net = np.nanmax(np.where(mask, np.abs(dc), np.nan), axis=1)
        nb = np.maximum(mask.sum(1) - 1, 1)
        up_bars = np.nansum(np.where(mask[:, 1:], np.sign(step) == sg, np.nan), axis=1)
        at = np.maximum(atr[idx], 1e-9)
        f = {
            "mv_minutes": tmin[idx].astype(float), "obs_min": float(OBS),
            "mv_speed": T[idx] / np.maximum(tmin[idx] + 1, 1) / at,
            "mv_mae_atr": np.abs(np.nanmin(adv, axis=1)) / at,
            "mv_straight": net / np.maximum(tot, 1e-12),
            "mv_updown": up_bars / nb,
            "mv_maxbar_atr": (np.nanmax(np.abs(np.where(np.isfinite(step), step, np.nan)), axis=1)
                              / np.maximum(ref[idx], 1e-9)) / at,
            "mv_nbars": nb.astype(float),
        }
        # ⭐관찰창 반응 (OBS>0 일 때만 의미)
        prog = (ref2 - ref[idx]) / ref[idx] * sgn[idx]           # 결정시점까지 총 진행폭
        f["obs_progress_atr"] = (prog - T[idx]) / at             # 트리거 이후 추가 진행(음수=되돌림)
        f["obs_ret_atr"] = np.where(OBS > 0, (prog - T[idx]) / at, 0.0)
        bref = bcl[np.clip(s1[idx], 0, len(bcl) - 1)]
        bnow = bcl[np.clip(np.maximum(s2[idx] - 1, 0), 0, len(bcl) - 1)]
        f["mv_btc_ret_atr"] = (bnow - bref) / np.maximum(bref, 1e-9) / at
        f["mv_idio_atr"] = prog / at - f["mv_btc_ret_atr"]
        f["mv_same_sign"] = (np.sign(bnow - bref) == sgn[idx]).astype(float)
        # 레벨 맥락
        fb2 = fb[idx]; a5 = np.maximum(atr5[fb2], 1e-9)
        for N in (48, 144, 288, 864):
            hh = LV[f"hi{N}"][fb2]; ll = LV[f"lo{N}"][fb2]
            f[f"brk{N}"] = np.where(sgn[idx] > 0, (ref2 - hh) / (hh * a5), (ll - ref2) / (ll * a5))
            f[f"pos{N}"] = (ref2 - ll) / np.maximum(hh - ll, 1e-9)
        # ⭐OI 4분면 (완결봉 기준)
        oid = F["oi_d6"][fb2]
        f["oi_quad"] = np.sign(oid) * sgn[idx]          # +1 = OI↑&진행방향 (신규진입) / -1 = 청산·커버
        f["tk_imb"] = (F["taker_buy_ratio"][fb2] - 0.5) * sgn[idx] * 2
        f["tk_extreme"] = (np.abs(F["taker_z"][fb2]) > 2).astype(float)
        r = pd.DataFrame({"timestamp": pd.to_datetime(dec_ts[idx]),
                          "anchor": D["anchor"].to_numpy()[idx], "split": D["split"].to_numpy()[idx],
                          "bar_idx": bidx[idx], "OBS": OBS,
                          "side_bottom": (D["side"].to_numpy()[idx] == "bottom").astype(int),
                          "dir_up": (sgn[idx] > 0).astype(int), "T_atr": T[idx],
                          "atr_at_anchor": atr[idx], "n_signals": D["n_signals"].to_numpy()[idx],
                          **lab, **{f"g_{k}": v for k, v in f.items()}})
        for s in SIGNALS:
            r[f"sig_{s}"] = D["signals"].astype(str).str.contains(s).astype(int).to_numpy()[idx]
        for k, v in F.items(): r[f"f_{k}"] = v[fb2]
        r["f_hour"] = pd.to_datetime(ts5[fb2]).hour
        r["f_dow"] = pd.to_datetime(ts5[fb2]).dayofweek
        frames.append(r)
        rs = " ".join("P%d:%.3f" % (int(p * 1e4), r["res_p%d" % int(p * 1e4)].mean()) for p in P_GRID)
        ys = " ".join("%.4f" % r["y_p%d" % int(p * 1e4)].mean() for p in P_GRID)
        print("      OBS=%2d분: %,d건 · 해소율 %s · 돌파율 %s".replace("%,d", "{:,}").format(len(r))
              % (OBS, rs, ys) if False else
              f"      OBS={OBS:>2}분: {len(r):,}건 · 해소율 {rs} · 돌파율 {ys}", flush=True)
    A = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    A.to_parquet(OUT / "dataset_v3.parquet")
    print(f"\n[5/5] 저장 {A.shape}")
    print(json.dumps({"rows": len(A),
                      "feats": len([c for c in A.columns if c.startswith(("f_", "g_", "sig_"))])},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
