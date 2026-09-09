#!/usr/bin/env python3
"""돌파/되돌림 섀도우 **라이브 피쳐 빌더** -- 학습 빌더와 1:1 대응 (2026-09-08).

⚠️CLAUDE.md `Position-Feature Train/Inference Parity Contract`:
   학습 빌더가 만든 피쳐와 라이브가 넣는 값은 이름·스케일·단위·소스가 정확히 일치해야 한다.
   이 파일이 그 계약의 라이브 쪽이고, 대응하는 학습 쪽은
     scripts/build_eth_breakout_reversal_dataset_20260908.py         (f_* · x_m_* · 사건)
     scripts/build_eth_breakout_reversal_features_v2_20260908.py     (v2_*)
   식을 고칠 때는 **양쪽을 같이** 고치고 반드시 `--parity` 를 다시 통과시킨다.

`--parity` 는 학습 데이터셋(dataset_v2.parquet)의 실제 사건을 무작위로 뽑아, 로컬 CSV 를
입력으로 이 모듈이 계산한 69피쳐를 저장값과 대조한다. 라이브 API 대신 같은 CSV 를 쓰므로
**식의 차이만** 검출된다(그게 목적이다 -- 데이터 신선도가 아니라 공식 일치를 본다).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

W_TRIG, NMOVE = 3, 15
LEVELS = (48, 144, 288, 864)
METRICS = ("retail", "ttc", "ttp", "tkv")


def zs(x, w):
    """학습 빌더 `zs()` 와 동일: rolling z, min_periods = w//3."""
    s = pd.Series(x)
    return ((s - s.rolling(w, min_periods=w // 3).mean())
            / s.rolling(w, min_periods=w // 3).std().replace(0, np.nan)).to_numpy()


def bar_features(C5, H5, L5, V5, TB, bt5):
    """f_* 중 봉 기반 21개 + 레벨맥락 + atr5. 전부 5분봉 배열 그대로 반환."""
    lr = np.full(len(C5), np.nan); lr[1:] = np.log(C5[1:] / C5[:-1])
    F = {}
    for k in (3, 6, 12, 48, 144):
        F[f"ret{k}"] = np.concatenate([np.full(k, np.nan), C5[k:] / C5[:-k] - 1.0])
        F[f"btc_ret{k}"] = np.concatenate([np.full(k, np.nan), bt5[k:] / bt5[:-k] - 1.0])
    F["atr_pct"] = pd.Series((H5 - L5) / C5).rolling(192, min_periods=64).mean().to_numpy()
    F["rng_z"] = zs((H5 - L5) / C5, 288)
    F["vol_z"] = zs(V5, 288)
    F["taker_buy_ratio"] = np.where(V5 > 0, TB / np.maximum(V5, 1e-12), np.nan)
    F["taker_z"] = zs(F["taker_buy_ratio"], 288)
    F["rv48"] = pd.Series(lr).rolling(48, min_periods=16).std().to_numpy()
    F["rv288"] = pd.Series(lr).rolling(288, min_periods=96).std().to_numpy()
    F["rv_ratio"] = F["rv48"] / np.maximum(F["rv288"], 1e-12)
    for k in (3, 12, 48):
        F[f"eth_btc_sp{k}"] = F[f"ret{k}"] - F[f"btc_ret{k}"]
    LV = {}
    for N in LEVELS:
        LV[f"hi{N}"] = pd.Series(H5).rolling(N, min_periods=N // 3).max().to_numpy()
        LV[f"lo{N}"] = pd.Series(L5).rolling(N, min_periods=N // 3).min().to_numpy()
    atr5 = pd.Series((H5 - L5) / C5).rolling(192, min_periods=64).mean().to_numpy()
    return F, LV, atr5


def metric_features(met: dict[str, np.ndarray], ts_met, ts5):
    """x_m_* 12개. met 값은 **비율 원값**이고 여기서 log 를 취한다(학습 빌더 `lg()` 와 동일).

    ⚠️학습 빌더는 메트릭 패널이 **float32** 로 저장돼 있어 `np.log` 도 float32 에서 돈다.
       라이브 API 값은 float64 라 그대로 두면 상대차 ~1e-4(절대 ~1e-7) 가 남는다. 트리 모델
       분할 해상도에 비하면 무의미하지만, 계약대로 **소스 정밀도까지 맞춘다**.
    """
    lg = lambda X: np.where(X > 0, np.log(np.maximum(X, 1e-9)), np.nan)
    out = {}
    pos = pd.Index(pd.DatetimeIndex(ts_met)).get_indexer(pd.DatetimeIndex(ts5))
    for nm in METRICS:
        v = lg(np.asarray(met[nm], dtype=np.float32)).astype(float)
        col = {f"m_{nm}_z": zs(v, 288)}
        for k in (12, 48):
            col[f"m_{nm}_d{k}"] = np.concatenate([np.full(k, np.nan), v[k:] - v[:-k]])
        for k, src in col.items():
            out[k] = np.where(pos >= 0, src[np.clip(pos, 0, len(src) - 1)], np.nan)
    return out


def path_features(hi1, lo1, cl1, bcl, s0, tmin, ref, sgn, atr, T, nmove=None):
    """v2_mv_* 11개. 🔴경로는 트리거 분 s1 을 **포함하지 않는다**(tmin-1 까지).

    같은 1분봉을 피쳐와 라벨이 공유하면 그 자체로 미래참조다 -- 그 봉의 큰 움직임이 피쳐를
    키우는 동시에 배리어를 때려 라벨을 정하기 때문이다(CLAUDE.md 사건 라벨 경계 계약).

    `nmove`: 발현 창(분). 아티팩트의 `emergence_window_min` 을 넘긴다 -- 창이 60분인데
    span 이 15면 tmin>=15 인 사건의 경로가 통째로 잘린다. 안 넘기면 옛 기본값(15).
    """
    span = np.arange(int(nmove or NMOVE))
    j0 = s0 + span
    mask = span <= (tmin - 1)
    c = cl1[np.clip(j0, 0, len(cl1) - 1)]
    h = hi1[np.clip(j0, 0, len(hi1) - 1)]
    l = lo1[np.clip(j0, 0, len(lo1) - 1)]
    at = max(atr, 1e-9)
    dc = np.where(mask, (c - ref) / ref, np.nan)
    adv = np.where(mask, np.where(sgn > 0, (l - ref) / ref, (ref - h) / ref), np.nan)
    with np.errstate(all="ignore"):
        mae = np.nanmin(np.where(np.isfinite(adv), adv, np.nan)) if np.isfinite(adv).any() else np.nan
        step = np.diff(np.where(mask, c, np.nan))
        tot = np.nansum(np.abs(step))
        net = np.abs(np.nanmax(np.where(mask, np.abs(dc), np.nan))) if np.isfinite(dc).any() else np.nan
        up_bars = np.nansum(np.where(mask[1:], np.sign(step) == sgn, np.nan))
        nb = max(int(mask.sum()) - 1, 1)
        mx = (np.nanmax(np.abs(np.where(np.isfinite(step), step, np.nan)))
              if np.isfinite(step).any() else np.nan)
    f = {
        "v2_mv_minutes": float(tmin),
        "v2_mv_speed": float(T / max(tmin + 1, 1) / at),
        "v2_mv_mae_atr": float(abs(mae) / at) if np.isfinite(mae) else np.nan,
        "v2_mv_straight": float(net / max(tot, 1e-12)) if np.isfinite(net) else np.nan,
        "v2_mv_updown": float(up_bars / nb),
        "v2_mv_maxbar_atr": float((mx / max(ref, 1e-9)) / at) if np.isfinite(mx) else np.nan,
        "v2_mv_nbars": float(nb),
        "v2_mv_path_known": float(tmin >= 1),
    }
    if bcl is not None:
        bref = bcl[np.clip(s0, 0, len(bcl) - 1)]
        bnow = bcl[np.clip(s0 + tmin, 0, len(bcl) - 1)]
        bret = (bnow - bref) / max(bref, 1e-9)
        eret = sgn * T
        f["v2_mv_btc_ret_atr"] = float(bret / at)
        f["v2_mv_idio_atr"] = float((eret - bret) / at)
        f["v2_mv_same_sign"] = float(np.sign(bret) == sgn)
    return f


def level_features(LV, atr5, fb, tp, sgn):
    """v2_brk* · v2_pos* 8개. 기준봉 fb = bt-1."""
    f = {}
    a5 = max(atr5[fb], 1e-9)
    for N in LEVELS:
        hh = LV[f"hi{N}"][fb]; ll = LV[f"lo{N}"][fb]
        f[f"v2_brk{N}"] = float((tp - hh) / (hh * a5)) if sgn > 0 else float((ll - tp) / (ll * a5))
        f[f"v2_pos{N}"] = float((tp - ll) / max(hh - ll, 1e-9))
    return f


def assemble(features_order, F, XS, ts5, fb, path, lev, ev):
    """69피쳐를 아티팩트가 기록한 **정확한 순서**로 벡터화한다."""
    v = {}
    for k, arr in F.items():
        v[f"f_{k}"] = float(arr[fb])
    v["f_hour"] = float(pd.Timestamp(ts5[fb]).hour)
    v["f_dow"] = float(pd.Timestamp(ts5[fb]).dayofweek)
    v["f_speed"] = float(ev["T_atr"] / max(ev["trig_min"] + 1, 1))
    for k, arr in XS.items():
        v[f"x_{k}"] = float(arr[fb])
    v.update(path); v.update(lev)
    for k in ("sig_sweep", "sig_smt", "sig_taker", "sig_kal",
              "sig_strz", "sig_orth", "sig_fib", "sig_dem"):
        v[k] = float(ev["signals"].get(k[4:], 0))
    for k in ("dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"):
        v[k] = float(ev[k])
    missing = [c for c in features_order if c not in v]
    if missing:
        raise KeyError(f"피쳐 누락 {len(missing)}개: {missing[:8]}")
    return np.array([v[c] for c in features_order], dtype=np.float32), v


# ------------------------------------------------------------------ 파리티 검사
def parity(n_check: int, seed: int) -> int:
    import build_eth_anchor_label_dataset_20260907 as B
    ART = ROOT / "data/live/breakout_reversal_shadow_artifact"
    meta = json.loads((ART / "meta.json").read_text())
    order = meta["features"]
    d = pd.read_parquet(ROOT / "tmp/eth_breakout_atr_state_20260908_s1/dataset_v2.parquet")
    d = d[(d.anchor == meta["anchor"]) & (d.T_mult == meta["t_mult"])].reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])

    eth = B._load_kl(B.ETH_KL); btc = B._load_kl(B.BTC_KL)
    ts5 = eth["timestamp"].to_numpy()
    O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    H5 = eth["high"].to_numpy(float); L5 = eth["low"].to_numpy(float)
    V5 = eth["volume"].to_numpy(float); TB = eth["taker_buy_base"].to_numpy(float)
    bt5 = btc.set_index("timestamp")["close"].reindex(pd.DatetimeIndex(ts5)).ffill().to_numpy(float)
    F, LV, atr5 = bar_features(C5, H5, L5, V5, TB, bt5)

    m1 = pd.read_csv(ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv",
                     usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy()
    hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float); cl1 = m1["close"].to_numpy(float)
    b1 = pd.read_csv(ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-1m-api.csv",
                     usecols=["timestamp", "close"], parse_dates=["timestamp"])
    b1 = b1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")
    bcl = b1["close"].reindex(pd.DatetimeIndex(ts1)).ffill().to_numpy(float)

    mz = np.load(ROOT / "tmp/xsec_perp_screen_20260908/metrics_panel.npz", allow_pickle=True)
    xts = pd.DatetimeIndex(pd.to_datetime(mz["ts"]))
    ei = list(np.load(ROOT / "tmp/xsec_perp_screen_20260908/panel.npz",
                      allow_pickle=True)["syms"]).index("ETHUSDT")
    met = {"retail": mz["count_long_short_ratio"][:, ei],
           "ttc": mz["count_toptrader_long_short_ratio"][:, ei],
           "ttp": mz["sum_toptrader_long_short_ratio"][:, ei],
           "tkv": mz["sum_taker_long_short_vol_ratio"][:, ei]}
    XS = metric_features(met, xts, ts5)

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(d), min(n_check, len(d)), replace=False)
    bad = {}; checked = 0
    for i in pick:
        r = d.iloc[i]
        bi = int(r["bar_idx"]); sgn = 1.0 if r["dir_up"] > 0 else -1.0
        atr = float(r["atr_at_anchor"]); T = float(r["T_atr"]); tmin = int(r["trig_min"])
        ref = O5[min(bi + 1, len(O5) - 1)]
        s0 = int(np.searchsorted(ts1, ts5[min(bi + 1, len(ts5) - 1)]))
        trig_ts = ts1[min(s0 + tmin, len(ts1) - 1)]
        bt = int(np.searchsorted(ts5, trig_ts, side="right") - 1)
        fb = bt - 1
        if fb < 900 or fb >= len(C5):
            continue
        pathf = path_features(hi1, lo1, cl1, bcl, s0, tmin, ref, sgn, atr, T)
        levf = level_features(LV, atr5, fb, ref * (1 + sgn * T), sgn)
        ev = {"T_atr": T, "trig_min": float(tmin), "dir_up": float(r["dir_up"]),
              "atr_at_anchor": atr, "n_signals": float(r["n_signals"]),
              "side_bottom": float(r["side_bottom"]),
              "signals": {s: int(r[f"sig_{s}"]) for s in
                          ("sweep", "smt", "taker", "kal", "strz", "orth", "fib", "dem")}}
        vec, vd = assemble(order, F, XS, ts5, fb, pathf, levf, ev)
        checked += 1
        for j, c in enumerate(order):
            a, b = float(vec[j]), float(r[c])
            if np.isnan(a) and np.isnan(b):
                continue
            # float32 로 저장된 값과 대조하므로 절대허용(atol)을 함께 둔다
            if not np.isfinite(a) or not np.isfinite(b) or not np.isclose(a, b, rtol=1e-4, atol=1e-6):
                bad.setdefault(c, []).append((abs(a - b), a, b))

    print(f"대조 {checked}건 × {len(order)}피쳐 = {checked*len(order):,}개 값")
    if not bad:
        print("✅파리티 통과 -- 전 피쳐 상대오차 < 1e-4")
        print(json.dumps({"parity": True, "checked": checked}, ensure_ascii=False))
        return 0
    print(f"❌불일치 {len(bad)}개 피쳐:")
    for c, v in sorted(bad.items(), key=lambda kv: -len(kv[1]))[:20]:
        d0, a, b = max(v)
        print(f"   {c:<22} {len(v):>4}/{checked}건 · 최대차 {d0:.6g} (라이브 {a:.6g} vs 학습 {b:.6g})")
    print(json.dumps({"parity": False, "bad": len(bad), "checked": checked}, ensure_ascii=False))
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260908)
    a = ap.parse_args()
    raise SystemExit(parity(a.n, a.seed) if a.parity else 0)
