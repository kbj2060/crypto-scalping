#!/usr/bin/env python3
"""양측 합산(베타 제거) 총수익 -- TRAIN 을 **선택 밖 확인창**으로 쓴다 (2026-09-08).

## 왜
같은 날 사건 스크린(342셀)에서 바닥측 초과가 OOS +15~23bp 로 컸지만, **천장측이 거의 정확히
반대 부호**(−7~−21)였다. corr(top, bottom) = −0.36, OOS 에서는 사실상 −1.
⇒ 그 초과는 앵커 정보가 아니라 **앵커 발동 시점의 잔존 베타**다(창 단위 귀무가 시점 군집을 못 맞춤).
**양측 합산이 베타 없는 통계**이고, 그 값은 VAL +2~9 · OOS −2~+5 · HOLDOUT −3~+3 이었다.

가장 좋아 보인 셀은 `any3/Wc1 · H=144 · ±2.0%` (VAL +9.4 / OOS +4.1 / HOLDOUT +2.5, 세 창 양수).
그러나 114셀을 훑었으므로 "세 창 양수"는 우연으로 14셀 나온다(0.5^3). 검정력이 필요하다.

## 이 스크립트
- **TRAIN 은 셀 선택에 쓰인 적이 없다**(모델을 학습하지 않는 순수 산술 규칙이므로 진짜 확인창).
- 네 창(TRAIN/VAL/OOS/HOLDOUT_SPENT) 전부에서 **양측 합산 순bp**(= 총 − 7.8)와 일군집 CI 를 낸다.
- 판정: 네 창 모두 순bp CI 하한 > 0 이면 승격 후보. 그 외는 기각.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
SRC = ROOT / "tmp/eth_anchor_label_dataset_20260907/anchors_labels.parquet"
OUT = ROOT / "tmp/eth_anchor_bothside_confirm_20260908"
ANCHORS = ("any3/Wc1", "any3/Wc3", "any4/Wc12", "any2/Wc3", "first_fire")
H_GRID = (48, 144, 288)
P_GRID = (1.5, 2.0, 3.0)
COST = 7.8
MAX_MIN = max(H_GRID) * 5
CHUNK = 3000
WINS = ("TRAIN", "VAL", "OOS", "HOLDOUT_SPENT")
BOOT = 3000
SEED = 20260908


def first_touch(hi1, lo1, start, up, dn):
    n = len(start)
    tu = np.full(n, -1, np.int32); td = np.full(n, -1, np.int32)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        idx = start[a:b, None] + np.arange(MAX_MIN)[None, :]
        hu = hi1[idx] >= up[a:b, None]; hd = lo1[idx] <= dn[a:b, None]
        au = hu.any(1); ad = hd.any(1)
        tu[a:b] = np.where(au, hu.argmax(1), -1); td[a:b] = np.where(ad, hd.argmax(1), -1)
    return tu, td


def pnl(t_up, t_dn, entry, ex, sgn, H, P):
    lim = H * 5; big = 1 << 30
    uo = (t_up >= 0) & (t_up < lim); do = (t_dn >= 0) & (t_dn < lim)
    tu = np.where(uo, t_up, big); td = np.where(do, t_dn, big)
    y = (ex - entry) / entry * 1e4 * sgn
    y = np.where(uo & (tu < td), sgn * P * 100.0, y)
    y = np.where(do & (td < tu), -sgn * P * 100.0, y)
    return np.where(uo & do & (tu == td), -P * 100.0, y)


def day_ci(v, day, rng, B=BOOT):
    u = np.unique(day)
    if len(u) < 8: return (np.nan, np.nan)
    idx = {x: np.flatnonzero(day == x) for x in u}
    o = [v[np.concatenate([idx[x] for x in rng.choice(u, len(u), True)])].mean() for _ in range(B)]
    return tuple(np.percentile(o, [2.5, 97.5]))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED)
    D = pd.read_parquet(SRC).reset_index(drop=True)
    eth = B._load_kl(B.ETH_KL)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy(); hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float)
    ts5 = eth["timestamp"].to_numpy(); O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    CAP = len(eth) - MAX_MIN // 5 - 2
    bi = D["bar_idx"].to_numpy()
    ei = np.minimum(bi + 1, len(O5) - 1); entry = O5[ei]
    st = np.searchsorted(ts1, ts5[ei])
    ok = (st < len(ts1) - MAX_MIN) & (ts1[np.minimum(st, len(ts1) - 1)] == ts5[ei]) & (bi <= CAP)
    s = np.where(ok, st, 0)
    print(f"[1/2] 터치 계산 (사건 {len(D):,}, 배리어 {len(P_GRID)}) ...", flush=True)
    TT = {P: first_touch(hi1, lo1, s, entry * (1 + P / 100), entry * (1 - P / 100)) for P in P_GRID}
    split = D["split"].to_numpy(); anc = D["anchor"].to_numpy()
    sgn = np.where(D["side"].to_numpy() == "bottom", 1.0, -1.0)
    day = pd.to_datetime(D["timestamp"]).dt.floor("D").to_numpy()

    print("[2/2] 채점 ...\n", flush=True)
    rows = []
    hdr = f"{'앵커':>11} {'H':>4} {'배리어':>7} | " + " | ".join(f"{w[:8]:>26}" for w in WINS)
    print(hdr); print("-" * len(hdr))
    for A_ in ANCHORS:
        for H in H_GRID:
            xi = np.minimum(ei + H, len(C5) - 1)
            for P in P_GRID:
                tu, td = TT[P]
                y_all = pnl(tu, td, entry, C5[xi], sgn, H, P)
                cells = {}
                line = f"{A_:>11} {H:>4} {'±'+str(P)+'%':>7} | "
                for w in WINS:
                    m = (anc == A_) & (split == w) & ok
                    if m.sum() < 40:
                        line += f"{'--':>26} | "; continue
                    net = y_all[m] - COST
                    lo, hi = day_ci(net, day[m], rng)
                    cells[w] = (float(net.mean()), lo, hi, int(m.sum()))
                    line += f"{net.mean():>+6.2f}[{lo:>+6.1f},{hi:>+6.1f}]n{m.sum():>4} | "
                print(line, flush=True)
                rows.append(dict(anchor=A_, H=H, P=P,
                                 **{f"{w}_{k}": cells.get(w, (np.nan,)*4)[i]
                                    for w in WINS for i, k in enumerate(("net", "lo", "hi", "n"))}))
    R = pd.DataFrame(rows); R.to_csv(OUT / "confirm.csv", index=False)
    four = R[[all(R.loc[i, f"{w}_lo"] > 0 for w in WINS) for i in R.index]]
    pos4 = R[[all(R.loc[i, f"{w}_net"] > 0 for w in WINS) for i in R.index]]
    print()
    print(f"⭐네 창 모두 순bp CI 하한 > 0: {len(four)}/{len(R)}")
    print(f"   (참고) 네 창 모두 점추정만 양수: {len(pos4)}/{len(R)} -- 우연 기대 {len(R)/16:.1f}건")
    if len(pos4):
        print(pos4[["anchor", "H", "P"] + [f"{w}_net" for w in WINS]].round(2).to_string(index=False))
    print(json.dumps({"cells": len(R), "four_window_ci": len(four), "four_window_sign": len(pos4)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
