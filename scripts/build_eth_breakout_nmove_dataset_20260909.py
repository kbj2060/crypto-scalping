#!/usr/bin/env python3
"""돌파/되돌림 -- **발현 창(NMOVE)을 바꾼 사건 모집단** 재빌드 (2026-09-09).

사용자: *"사건 정의를 15분이 아니라 1시간으로 바꿔서 다시 진행해줘"*

## 왜 사건부터 다시 만드는가
앵커(증거신호 첫 발동)에서 **NMOVE 분 안에 ±T×ATR 최초 터치**가 발현이다. NMOVE 를 늘리면
    · 발현하는 앵커가 늘고(짧은 창에서 놓치던 느린 움직임이 들어온다)
    · 발현 시점 s1 이 뒤로 밀려 **트리거 봉·피쳐 봉·경로·레벨 맥락이 전부 바뀐다**
즉 라벨만이 아니라 **모집단 자체가 다른 실험**이다. 그래서 피쳐를 다시 만든다.

## 경계 계약 (CLAUDE.md 사건 라벨 경계)
라벨은 s1 **부터** 배리어를 탐색한다. 따라서
    · 경로 피쳐 : s0 .. s1-1        (트리거 분 제외)
    · 봉/레벨/메트릭/f154 : fb = bt-1  (트리거 봉 직전 완결봉)
NMOVE 를 바꿔도 이 관계는 그대로다. 빌드 후 감사 스크립트를 돌린다.

## 산출
`tmp/eth_breakout_nmove_20260909/dataset_nm{N}.parquet`
  · 기준 69피쳐(라이브 `live_eth_breakout_features_20260908.py` 와 같은 정의)
  · i_* 154피쳐(캐노니컬 프레임, fb 에서 샘플)
  · split(날짜 경계는 기존 데이터셋과 동일) · bar_idx/s1/bt/fb/tmin/dir_up
NMOVE=15 도 같은 코드로 만들어 **대조군**으로 쓴다(빌더 차이가 아니라 창 차이만 남긴다).
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402
import live_eth_breakout_features_20260908 as LF     # noqa: E402

KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
BKL1 = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-1m-api.csv"
ANCH = ROOT / "tmp/eth_anchor_label_dataset_20260907/anchors_labels.parquet"
XMET = ROOT / "tmp/xsec_perp_screen_20260908/metrics_panel.npz"
F154 = ROOT / "tmp/ilias_eth_154feature_dataset_extended_20260907/ilias_eth_154feature_2024_2026H1_combined.csv"
OUT = ROOT / "tmp/eth_breakout_nmove_20260909"
T_MULT, CHUNK = 0.75, 4000
SIGKEY = {"sweep": "sig_sweep", "smt": "sig_smt", "taker": "sig_taker", "kal": "sig_kal",
          "strz": "sig_strz", "orth": "sig_orth", "fib": "sig_fib", "dem": "sig_dem"}
# metrics_panel 컬럼 -> LF.METRICS 이름
MET_MAP = {"retail": "count_long_short_ratio", "ttc": "count_toptrader_long_short_ratio",
           "ttp": "sum_toptrader_long_short_ratio", "tkv": "sum_taker_long_short_vol_ratio"}
SPLITS = [("TRAIN", "2000-01-01", "2025-09-01"), ("VAL", "2025-09-01", "2026-01-01"),
          ("OOS", "2026-01-01", "2026-04-01"), ("HOLDOUT_SPENT", "2026-04-01", "2026-08-01")]


def first_touch(hi1, lo1, start, up, dn, nmin):
    n = len(start); tu = np.full(n, -1, np.int32); td = np.full(n, -1, np.int32)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n); ix = start[a:b, None] + np.arange(nmin)[None, :]
        hu = hi1[ix] >= up[a:b, None]; hd = lo1[ix] <= dn[a:b, None]
        tu[a:b] = np.where(hu.any(1), hu.argmax(1), -1)
        td[a:b] = np.where(hd.any(1), hd.argmax(1), -1)
    return tu, td


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmove", type=int, required=True, help="발현 창(분)")
    ap.add_argument("--path-lag", type=int, default=0,
                    help="경로 창을 이만큼 **더** 과거로 민다(트립와이어용). 0=계약대로 s1-1 까지")
    a = ap.parse_args()
    NM = a.nmove
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[1/6] 로드 (NMOVE={NM}분) ...", flush=True)
    D = pd.read_parquet(ANCH)
    D = D[D["anchor"] == "first_fire"].sort_values("timestamp").reset_index(drop=True)
    eth = B._load_kl(B.ETH_KL); btc = B._load_kl(B.BTC_KL)
    ts5 = eth["timestamp"].to_numpy()
    O5 = eth["open"].to_numpy(float); H5 = eth["high"].to_numpy(float)
    L5 = eth["low"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    V5 = eth["volume"].to_numpy(float)
    TB = (eth["taker_buy_base"] if "taker_buy_base" in eth else eth["taker_buy_volume"]).to_numpy(float)
    bt5 = pd.Series(btc["close"].to_numpy(float),
                    index=pd.DatetimeIndex(btc["timestamp"])).reindex(pd.DatetimeIndex(ts5)).ffill().to_numpy()
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy(); hi1 = m1["high"].to_numpy(float)
    lo1 = m1["low"].to_numpy(float); cl1 = m1["close"].to_numpy(float)
    b1 = pd.read_csv(BKL1, usecols=["timestamp", "close"], parse_dates=["timestamp"])
    bcl = (b1.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
             .set_index("timestamp")["close"].reindex(pd.DatetimeIndex(ts1)).ffill().to_numpy(float))

    print("[2/6] 봉·레벨·메트릭 피쳐 ...", flush=True)
    F, LV, atr5 = LF.bar_features(C5, H5, L5, V5, TB, bt5)
    z = np.load(XMET, allow_pickle=True)
    j = int(np.flatnonzero(z["syms"] == "ETHUSDT")[0])
    met = {k: z[v][:, j] for k, v in MET_MAP.items()}
    XS = LF.metric_features(met, z["ts"], ts5)

    print("[3/6] 발현 탐색 ...", flush=True)
    bidx = D["bar_idx"].to_numpy(); atr = D["atr_pct"].to_numpy(float)
    ei0 = np.minimum(bidx + 1, len(O5) - 1); ref = O5[ei0]
    st = np.searchsorted(ts1, ts5[ei0])
    ok = (st < len(ts1) - NM - 12 * 5 - 10) & (ts1[np.minimum(st, len(ts1) - 1)] == ts5[ei0])
    s0 = np.where(ok, st, 0); T = atr * T_MULT
    tu, td = first_touch(hi1, lo1, s0, ref * (1 + T), ref * (1 - T), NM)
    big = 1 << 30
    au = np.where(tu >= 0, tu, big); ad = np.where(td >= 0, td, big)
    fired = ok & ((au < big) | (ad < big)) & (au != ad)
    sgn = np.where(au < ad, 1.0, -1.0); tmin = np.where(au < ad, au, ad)
    s1 = s0 + np.where(fired, tmin, 0)
    bt = np.searchsorted(ts5, ts1[np.minimum(s1, len(ts1) - 1)], side="right") - 1
    fb = bt - 1
    keep = fired & (fb >= 900) & (bt < len(C5) - 1)
    idx = np.flatnonzero(keep); n = len(idx)
    print(f"      앵커 {len(D):,} · 발현 {fired.sum():,} ({fired.sum()/len(D):.1%}) · 사용 {n:,}", flush=True)

    print("[4/6] 발현 경로 (s0..s1-1, 트리거 분 제외) ...", flush=True)
    span = np.arange(NM)[None, :]
    j0 = s0[idx][:, None] + span
    # --path-lag: 경로 창 끝을 한 단위 더 밀어도 정확도가 유지되는가(경계 누수 판별).
    mask = span <= (tmin[idx][:, None] - 1 - a.path_lag)
    c = cl1[np.clip(j0, 0, len(cl1) - 1)]; h = hi1[np.clip(j0, 0, len(hi1) - 1)]
    l = lo1[np.clip(j0, 0, len(lo1) - 1)]
    r = ref[idx][:, None]; sg = sgn[idx][:, None]
    at = np.maximum(atr[idx], 1e-9)
    dc = np.where(mask, (c - r) / r, np.nan)
    adv = np.where(mask, np.where(sg > 0, (l - r) / r, (r - h) / r), np.nan)
    with np.errstate(all="ignore"):
        mae = np.nanmin(adv, axis=1)
        step = np.diff(np.where(mask, c, np.nan), axis=1)
        tot = np.nansum(np.abs(step), axis=1)
        net = np.nanmax(np.abs(dc), axis=1)
        up_bars = np.nansum(np.where(mask[:, 1:], np.sign(step) == sg, np.nan), axis=1)
        mx = np.nanmax(np.abs(step), axis=1)
    nb = np.maximum(mask.sum(1) - 1, 1)
    bref = bcl[np.clip(s0[idx], 0, len(bcl) - 1)]; bnow = bcl[np.clip(s1[idx], 0, len(bcl) - 1)]
    bret = (bnow - bref) / np.maximum(bref, 1e-9)
    P = {"v2_mv_minutes": tmin[idx].astype(float),
         "v2_mv_speed": T[idx] / np.maximum(tmin[idx] + 1, 1) / at,
         "v2_mv_mae_atr": np.abs(mae) / at,
         "v2_mv_straight": net / np.maximum(tot, 1e-12),
         "v2_mv_updown": up_bars / nb,
         "v2_mv_maxbar_atr": (mx / np.maximum(ref[idx], 1e-9)) / at,
         "v2_mv_nbars": nb.astype(float),
         "v2_mv_path_known": (tmin[idx] >= 1).astype(float),
         "v2_mv_btc_ret_atr": bret / at,
         "v2_mv_idio_atr": (sgn[idx] * T[idx] - bret) / at,
         "v2_mv_same_sign": (np.sign(bret) == sgn[idx]).astype(float)}

    print("[5/6] 조립 ...", flush=True)
    fb2 = fb[idx]; tp = ref[idx] * (1 + sgn[idx] * T[idx]); a5 = np.maximum(atr5[fb2], 1e-9)
    out = {f"f_{k}": arr[fb2] for k, arr in F.items()}
    tsf = pd.DatetimeIndex(ts5[fb2])
    out["f_hour"] = tsf.hour.to_numpy(float); out["f_dow"] = tsf.dayofweek.to_numpy(float)
    out["f_speed"] = T[idx] / np.maximum(tmin[idx] + 1, 1)
    out.update({f"x_{k}": arr[fb2] for k, arr in XS.items()})
    out.update(P)
    for N in LF.LEVELS:
        hh = LV[f"hi{N}"][fb2]; ll = LV[f"lo{N}"][fb2]
        out[f"v2_brk{N}"] = np.where(sgn[idx] > 0, (tp - hh) / (hh * a5), (ll - tp) / (ll * a5))
        out[f"v2_pos{N}"] = (tp - ll) / np.maximum(hh - ll, 1e-9)
    sig = D["signals"].to_numpy()[idx]
    for k, col in SIGKEY.items():
        out[col] = np.array([1.0 if k in str(s).split("+") else 0.0 for s in sig])
    out["dir_up"] = (sgn[idx] > 0).astype(float); out["trig_min"] = tmin[idx].astype(float)
    out["T_atr"] = T[idx]; out["atr_at_anchor"] = atr[idx]
    out["n_signals"] = D["n_signals"].to_numpy(float)[idx]
    out["side_bottom"] = (D["side"].to_numpy()[idx] == "bottom").astype(float)
    d = pd.DataFrame(out)
    d.insert(0, "timestamp", ts1[np.minimum(s1[idx], len(ts1) - 1)])
    d["bar_idx"] = bidx[idx]; d["s1"] = s1[idx]; d["bt"] = bt2 = bt[idx]; d["fb"] = fb2
    d["ref_px"] = ref[idx]; d["entry_px"] = tp
    ts = pd.to_datetime(d["timestamp"])
    d["split"] = "NONE"
    for nm, a0, a1 in SPLITS:
        d.loc[(ts >= pd.Timestamp(a0)) & (ts < pd.Timestamp(a1)), "split"] = nm

    print("[6/6] f154 결합 (fb 에서 샘플) ...", flush=True)
    Fx = pd.read_csv(F154, parse_dates=["timestamp"])
    Fx = Fx.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    fcols = [c for c in Fx.columns if c != "timestamp"]
    pos = pd.Index(Fx["timestamp"]).get_indexer(pd.DatetimeIndex(ts5[np.clip(fb2, 0, len(ts5) - 1)]))
    A = Fx[fcols].to_numpy(np.float32)
    Xf = np.where((pos >= 0)[:, None], A[np.clip(pos, 0, len(A) - 1)], np.nan)
    for jj, cc in enumerate(fcols): d[f"i_{cc}"] = Xf[:, jj]
    d = d.replace([np.inf, -np.inf], np.nan)
    tagl = "" if a.path_lag == 0 else f"_lag{a.path_lag}"
    p = OUT / f"dataset_nm{NM}{tagl}.parquet"
    d.to_parquet(p)
    meta = {"nmove": NM, "t_mult": T_MULT, "anchor": "first_fire", "n": int(len(d)),
            "fire_rate": float(fired.sum() / len(D)), "f154_cov": float((pos >= 0).mean()),
            "split_counts": d["split"].value_counts().to_dict(),
            "trig_min": {"median": float(np.median(tmin[idx])), "mean": float(tmin[idx].mean()),
                         "q90": float(np.percentile(tmin[idx], 90))}}
    meta["path_lag"] = a.path_lag
    json.dump(meta, open(OUT / f"meta_nm{NM}{tagl}.json", "w"), ensure_ascii=False, indent=1)
    print(f"저장 {p} · {len(d):,}행 · f154 결합 {(pos>=0).mean():.1%}")
    print(f"   발현까지 분: 중앙 {meta['trig_min']['median']:.0f} · 평균 {meta['trig_min']['mean']:.1f} "
          f"· 90% {meta['trig_min']['q90']:.0f}")
    print("   split:", meta["split_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
