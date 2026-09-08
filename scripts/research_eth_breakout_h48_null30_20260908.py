#!/usr/bin/env python3
"""H=48봉·±0.50% **최고 구성의 B=30 셔플 귀무** -- 병렬 조각 (2026-09-08).

TabICL B=5 결과는 세 창 모두 셔플 max 를 넘었지만 **p ≤ 0.167** 이 한계였고
VAL 은 셔플 max 대비 **0.0009 차이**였다. B=30 이면 p ≤ 0.032 로 내려간다.
⚠️셔플 평균이 0.50 이 아니다(B=5 실측 VAL 0.5163 / OOS 0.5003 / HOLDOUT 0.5132) --
   일군집 구조 때문이며 **셔플 분포가 진짜 기준선**이다.

TabICL 은 CPU 1fit≈24초(5피쳐)라 B=30 단일 프로세스면 5시간이다.
⇒ `--rep/--nrep` 로 조각내 병렬 실행한다. 결과는 `null30_<rep>.json` 에 쌓고 별도로 합친다.
사용: --rep 0 --nrep 8
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

MY = ROOT / "tmp/eth_breakout_atr_state_20260908_s1"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H, P = 48, 0.0050
EMB = pd.Timedelta(hours=4)
CHUNK = 4000
BTC = ["v2_mv_btc_ret_atr", "v2_mv_idio_atr", "v2_mv_same_sign", "dir_up", "T_atr"]


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--nrep", type=int, required=True)
    a = ap.parse_args()
    from tabicl import TabICLClassifier
    d = pd.read_parquet(MY / "dataset_v4.parquet").sort_values("timestamp").reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    eth = B._load_kl(B.ETH_KL)
    ts5 = eth["timestamp"].to_numpy(); O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy(); hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float)
    sgn = np.where(d["dir_up"].to_numpy() > 0, 1.0, -1.0)
    bi = d["bar_idx"].to_numpy(); T = d["T_atr"].to_numpy(float)
    entry = O5[np.minimum(bi + 1, len(O5) - 1)] * (1 + sgn * T)
    s1 = np.searchsorted(ts1, d["timestamp"].to_numpy())
    bt = np.searchsorted(ts5, ts1[np.clip(s1, 0, len(ts1) - 1)], side="right") - 1
    okm = (s1 > 0) & (s1 + H * 5 < len(ts1)) & (bt + H < len(C5))
    tu, td = first_touch(hi1, lo1, np.where(okm, s1, 0), entry * (1 + P), entry * (1 - P), H * 5)
    big = 1 << 30
    uo, do_ = tu >= 0, td >= 0
    au, ad = np.where(uo, tu, big), np.where(do_, td, big)
    cont = np.where(sgn > 0, uo & (au < ad), do_ & (ad < au))
    rev = np.where(sgn > 0, do_ & (ad < au), uo & (au < ad))
    x5 = np.minimum(bt + H, len(C5) - 1)
    clo = (C5[x5] - entry) / entry * 1e4 * sgn
    y = np.where(cont, 1, np.where(rev, 0, (clo > 0).astype(int)))
    ts = d["timestamp"]; sp = d["split"].to_numpy()
    months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    X = np.nan_to_num(d[BTC].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    out = []
    for b in range(a.rep, a.rep + a.nrep):
        t0 = time.time(); rng = np.random.default_rng(7000 + b)
        pred = np.full(len(y), np.nan)
        for i, mo in enumerate(uniq):
            if i < 5: continue
            te = (months == mo).to_numpy(); tr = (ts < ts[te].min() - EMB).to_numpy()
            if tr.sum() < 1500 or te.sum() < 30: continue
            itr = np.flatnonzero(tr); yy = y.copy(); yy[itr] = rng.permutation(yy[itr])
            c = TabICLClassifier(device="cpu", n_estimators=2, random_state=20260908, verbose=False)
            c.fit(X[itr], yy[itr]); pred[te] = c.predict_proba(X[te])[:, 1]
        r = {w: float(((pred[np.isfinite(pred) & (sp == w) & okm] > 0.5).astype(int)
                       == y[np.isfinite(pred) & (sp == w) & okm]).mean()) for w in WINS}
        out.append({"rep": b, **r})
        print(f"셔플{b} {time.time()-t0:.0f}초 " + " ".join(f"{w[:4]} {r[w]:.4f}" for w in WINS), flush=True)
        json.dump(out, open(MY / f"null30_h48_{a.rep}.json", "w"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
