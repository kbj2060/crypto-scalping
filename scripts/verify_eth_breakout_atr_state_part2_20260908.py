#!/usr/bin/env python3
"""ATR 상태 서술 견고성 감사 **파트2** -- 라벨 견고성 · 속도 통제 · 연도 안정성 (2026-09-08).

⚠️동시 세션이 같은 저장소에서 `tmp/eth_breakout_reversal_20260908/` 를 덮어쓰고 있어
   입력을 세션 전용 경로(`tmp/eth_breakout_atr_state_20260908_s1/`)로 분리했다.

파트1 통과: 상위5%일 제거해도 유지(0.444→0.450 등) · 동일 커버리지 무작위 대조군 네 창 p<0.05
(0.0020 / 0.0180 / 0.0040 / 0.0020).
파트2: (4) 배리어 P 를 바꿔도 같은 방향인가 (5) 이미 아는 '속도' 축의 다른 표현일 뿐인가
(6) 연·분기 안정성.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

MY = ROOT / "tmp/eth_breakout_atr_state_20260908_s1"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
ALLW = ("TRAIN", "VAL", "OOS", "HOLDOUT_SPENT")
H = 48
CHUNK = 4000
SEED = 20260908


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
    rng = np.random.default_rng(SEED)
    d = pd.read_parquet(MY / "dataset_v2.parquet")
    d = d[(d.anchor == "first_fire") & (d.T_mult == 1.0)].reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    eth = B._load_kl(B.ETH_KL)
    ts5 = eth["timestamp"].to_numpy(); O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy(); hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float)

    v = d["atr_at_anchor"].to_numpy(float); sp = d["split"].to_numpy()
    ts = d["timestamp"]; thr = np.nanpercentile(v[sp == "TRAIN"], 80); m0 = v >= thr
    sgn = np.where(d["dir_up"].to_numpy() > 0, 1.0, -1.0)
    bi = d["bar_idx"].to_numpy(); T = d["T_atr"].to_numpy(float)
    entry = O5[np.minimum(bi + 1, len(O5) - 1)] * (1 + sgn * T)
    s1 = np.searchsorted(ts1, d["timestamp"].to_numpy())
    okm = (s1 > 0) & (s1 < len(ts1) - H * 5 - 5)

    print("=" * 100)
    print("4) 라벨 견고성 -- 배리어 P 를 바꿔도 같은 방향인가 (ATR≥p80, 결정=트리거 시점)")
    print("=" * 100)
    big = 1 << 30
    for Pv in (0.0025, 0.0050, 0.0075, 0.0100):
        tu, td = first_touch(hi1, lo1, np.where(okm, s1, 0),
                             entry * (1 + Pv), entry * (1 - Pv), H * 5)
        uo = tu >= 0; do_ = td >= 0
        au = np.where(uo, tu, big); ad = np.where(do_, td, big)
        upf = uo & (au < ad); dnf = do_ & (ad < au)
        cont = np.where(sgn > 0, upf, dnf); rev = np.where(sgn > 0, dnf, upf)
        bt = np.searchsorted(ts5, ts1[np.clip(s1, 0, len(ts1) - 1)], side="right") - 1
        x5 = np.minimum(bt + H, len(C5) - 1)
        clo = (C5[x5] - entry) / entry * 1e4 * sgn
        yy = np.where(cont, 1, np.where(rev, 0, (clo > 0).astype(int)))
        line = f"   P=±{Pv*100:.2f}% 해소율 {(cont|rev)[okm].mean():.3f} | "
        for w in ALLW:
            m = m0 & okm & (sp == w)
            if m.sum() < 80: line += f"{w[:4]} -- | "; continue
            line += f"{w[:4]} 돌파 {yy[m].mean():.4f} 되돌림 {1-yy[m].mean():.4f} (n{m.sum():>4}) | "
        print(line, flush=True)

    print("\n" + "=" * 100)
    print("5) 속도 축과의 중복 -- 서로 통제하면 남는가 (P=0.5%, 원 라벨)")
    print("=" * 100)
    y = d["y"].to_numpy(int)
    spd = d["f_speed"].to_numpy(float) if "f_speed" in d.columns else T / np.maximum(d["trig_min"] + 1, 1)
    ok = np.isfinite(v) & np.isfinite(spd)
    aq = pd.qcut(pd.Series(v[ok]), 4, labels=False, duplicates="drop").to_numpy()
    sq = pd.qcut(pd.Series(spd[ok]), 4, labels=False, duplicates="drop").to_numpy()
    piv = pd.DataFrame({"a": aq, "s": sq, "y": y[ok]}).groupby(["a", "s"])["y"].mean().unstack()
    print("   행=ATR 4분위 · 열=속도 4분위 · 값=돌파율")
    print(piv.round(3).to_string())
    print(f"   ⇒ 속도 통제 후 ATR 효과(행4−행1) {piv.iloc[-1].mean()-piv.iloc[0].mean():+.4f} · "
          f"ATR 통제 후 속도 효과(열4−열1) {piv.iloc[:,-1].mean()-piv.iloc[:,0].mean():+.4f}")
    print(f"   상관 corr(ATR, 속도) = {np.corrcoef(v[ok], spd[ok])[0,1]:+.3f}")

    print("\n" + "=" * 100)
    print("6) 연·분기 안정성 (ATR≥p80)")
    print("=" * 100)
    g = pd.DataFrame({"y": y[m0], "yr": ts[m0].dt.year, "q": ts[m0].dt.to_period("Q").astype(str)})
    print("   연도: " + " · ".join(f"{k} {x['y'].mean():.3f}(n{len(x)})" for k, x in g.groupby("yr")))
    qq = g.groupby("q")["y"].agg(["mean", "count"])
    print(f"   분기 {len(qq)}개 중 돌파율<50% 인 분기 {int((qq['mean']<0.5).sum())}개")
    print("   " + " ".join(f"{i}:{r['mean']:.3f}(n{int(r['count'])})" for i, r in qq.iterrows()))
    print(json.dumps({"thr": float(thr)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
