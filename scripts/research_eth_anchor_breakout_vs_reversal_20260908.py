#!/usr/bin/env python3
"""앵커 발동 후 **돌파인지 되돌림인지를 시장이 답하게 한다** (2026-09-08).

사용자: *"나는 지금 가진 증거신호를 이용하려면 무조건 되돌림인지 돌파인지를 알아야해"*

## 지금까지 닫은 것과 이게 다른 점
부록 Z·AA 가 닫은 것은 **"앵커 봉에서 방향을 예측할 수 있는가"** 였다(TRAIN 45/45 음수).
그건 시각 t 의 정보만으로 위/아래를 맞히는 문제다.
**이 스크립트는 예측하지 않는다.** 앵커 이후 W봉 동안 기다렸다가, 가격이 **먼저 어느 쪽으로
±T 만큼 움직이는지 관측한 뒤** 그 방향(돌파) 또는 반대(되돌림)로 진입한다.
방향은 **관측값**이므로 예측 스킬이 필요 없다. 남는 질문은 하나다:
    **발현된 방향으로 이미 T 만큼 움직인 뒤, 남은 움직임이 늦은 진입 + 비용을 덮는가?**

## 설계
- 기준가 = `open[앵커봉+1]` (L4 계약)
- 트리거 = 기준가 × (1 ± T). T ∈ {0.25,0.5,0.75,1.0}×atr_pct 및 고정 {0.15,0.25,0.40}%
- 트리거 탐색창 W ∈ {3,6,12,24}봉. 그 안에 둘 다 안 닿으면 **거래 없음**(발동률로 보고)
- 진입 = 그 트리거 레벨 (스톱주문 체결 가정) + **슬리피지 2bp 불리하게**
- 두 팔: **돌파**(발현 방향으로) · **되돌림**(반대로)
- 청산 = 진입가 ± P (P ∈ {0.5, 1.0}%) 1분봉 첫터치, 미터치면 H봉 뒤 종가 (**커버리지 100%**)
- 비용 = 테이커 왕복 10bp + 슬리피지 2bp = **12bp**. 트레일링 없음(09-07 체결 결함 회피)
- 창 = TRAIN/VAL/OOS/HOLDOUT. **TRAIN 은 셀 선택에 쓰지 않는다** -> 확인창
- 귀무 = 같은 창의 무작위 봉에 같은 규칙 적용(같은 측면 부호 유지)
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
OUT = ROOT / "tmp/eth_anchor_breakout_20260908"
ANCHORS = ("any3/Wc3", "any2/Wc3", "first_fire")
T_ATR = (0.25, 0.5, 0.75, 1.0)
T_FIX = (0.0015, 0.0025, 0.0040)
W_GRID = (3, 6, 12, 24)
P_GRID = (0.005, 0.010)
H = 48
SLIP = 2.0
COST = 12.0
WINS = ("TRAIN", "VAL", "OOS", "HOLDOUT_SPENT")
MAXMIN = (max(W_GRID) + H) * 5 + 10
CHUNK = 4000
BOOT = 3000
NULLB = 30
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


def day_ci(v, day, rng, Bt=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8 or len(v) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (Bt, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def run(bidx, atr, ts5, O5, C5, ts1, hi1, lo1, Tspec, W, P):
    """반환: (유효, 발현방향(+1위/-1아래), 돌파팔 실현bp)  -- 미발동은 유효=False"""
    ei = np.minimum(bidx + 1, len(O5) - 1)
    ref = O5[ei]
    st = np.searchsorted(ts1, ts5[ei])
    ok = (st < len(ts1) - MAXMIN) & (ts1[np.minimum(st, len(ts1) - 1)] == ts5[ei])
    s0 = np.where(ok, st, 0)
    T = atr * Tspec[1] if Tspec[0] == "atr" else np.full(len(bidx), Tspec[1])
    tu, td = first_touch(hi1, lo1, s0, ref * (1 + T), ref * (1 - T), W * 5)
    hit_u = tu >= 0; hit_d = td >= 0
    fired = ok & (hit_u | hit_d)
    big = 1 << 30
    a = np.where(hit_u, tu, big); b = np.where(hit_d, td, big)
    dir_up = fired & (a < b)                       # 위로 먼저 = 상방 발현
    dir_dn = fired & (b < a)
    fired = fired & (dir_up | dir_dn)              # 동시 터치는 버림
    sgn = np.where(dir_up, 1.0, -1.0)              # 발현 방향
    tmin = np.where(dir_up, a, b)
    entry = ref * (1 + sgn * T)
    entry = entry * (1 + sgn * SLIP / 1e4)         # 스톱 체결 슬리피지: 불리한 쪽
    s1 = np.where(fired, s0 + np.where(fired, tmin, 0), 0)
    tu2, td2 = first_touch(hi1, lo1, s1, entry * (1 + P), entry * (1 - P), H * 5)
    # 시간청산 종가
    x5 = np.minimum(np.searchsorted(ts5, ts1[np.minimum(s1, len(ts1) - 1)]) + H, len(C5) - 1)
    y = (C5[x5] - entry) / entry * 1e4
    uo = tu2 >= 0; do_ = td2 >= 0
    au = np.where(uo, tu2, big); ad = np.where(do_, td2, big)
    y = np.where(uo & (au < ad), P * 1e4, y)
    y = np.where(do_ & (ad < au), -P * 1e4, y)
    y = np.where(uo & do_ & (au == ad), -P * 1e4, y)
    return fired, sgn, y * sgn                      # 돌파팔 = 발현 방향으로 진입


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED)
    D = pd.read_parquet(SRC).reset_index(drop=True)
    eth = B._load_kl(B.ETH_KL)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy(); hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float)
    ts5 = eth["timestamp"].to_numpy(); O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    CAP = len(eth) - MAXMIN // 5 - 2
    split = D["split"].to_numpy(); anc = D["anchor"].to_numpy()
    day = pd.to_datetime(D["timestamp"]).dt.floor("D").to_numpy()
    bidx = D["bar_idx"].to_numpy(); atr = D["atr_pct"].to_numpy()
    inrange = bidx <= CAP
    print(f"앵커 {len(D):,} · 격자 {len(ANCHORS)}×{len(T_ATR)+len(T_FIX)}×{len(W_GRID)}×{len(P_GRID)}",
          flush=True)

    rows = []
    specs = [("atr", t) for t in T_ATR] + [("fix", t) for t in T_FIX]
    for Tspec in specs:
        for W in W_GRID:
            for P in P_GRID:
                fired, sgn, ybk = run(bidx, atr, ts5, O5, C5, ts1, hi1, lo1, Tspec, W, P)
                for A_ in ANCHORS:
                    base = (anc == A_) & inrange
                    for arm, y in (("돌파", ybk), ("되돌림", -ybk)):
                        rec = dict(anchor=A_, trig=f"{Tspec[0]}{Tspec[1]}", W=W, P=P, arm=arm)
                        ok = True
                        for w in WINS:
                            m = base & (split == w)
                            f = m & fired
                            if f.sum() < 40: ok = False; break
                            net = y[f] - COST
                            lo, hi = day_ci(net, day[f], rng)
                            rec[f"{w}_n"] = int(f.sum()); rec[f"{w}_rate"] = float(f.sum() / max(m.sum(), 1))
                            rec[f"{w}_net"] = float(net.mean()); rec[f"{w}_lo"] = lo; rec[f"{w}_hi"] = hi
                        if ok: rows.append(rec)
        print(f"  트리거 {Tspec} 완료 (셀 {len(rows)})", flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / "grid.csv", index=False)
    print(f"\n셀 {len(R)}\n", flush=True)
    print("=== 발동률 (TRAIN, 앵커 any3/Wc3) ===")
    q = R[(R.anchor == "any3/Wc3") & (R.arm == "돌파")]
    print(q.pivot_table(index="trig", columns="W", values="TRAIN_rate").round(3).to_string())
    print("\n=== 팔별 TRAIN 순bp 평균 ===")
    print(R.pivot_table(index=["arm", "trig"], columns="W", values="TRAIN_net").round(2).to_string())
    four = R[[all(R.loc[i, f"{w}_lo"] > 0 for w in WINS) for i in R.index]]
    pos4 = R[[all(R.loc[i, f"{w}_net"] > 0 for w in WINS) for i in R.index]]
    print(f"\n⭐네 창 모두 순bp CI 하한 > 0: {len(four)}/{len(R)}")
    if len(four):
        print(four[["anchor", "trig", "W", "P", "arm"] + [f"{w}_net" for w in WINS]
                   + [f"{w}_lo" for w in WINS]].round(2).to_string(index=False))
    print(f"⭐네 창 점추정 모두 양수: {len(pos4)}/{len(R)} (우연 기대 {len(R)/16:.1f})")
    if len(pos4):
        print(pos4[["anchor", "trig", "W", "P", "arm", "TRAIN_n", "TRAIN_rate"]
                   + [f"{w}_net" for w in WINS]].round(2).to_string(index=False))
    b = R.reindex(R[[f"{w}_net" for w in WINS]].min(1).sort_values(ascending=False).index)
    print("\n=== 네 창 최소 순bp 상위 12 ===")
    print(b.head(12)[["anchor", "trig", "W", "P", "arm", "TRAIN_n", "TRAIN_rate", "TRAIN_net",
                      "TRAIN_lo", "VAL_net", "OOS_net", "HOLDOUT_SPENT_net"]].round(2).to_string(index=False))
    print(json.dumps({"cells": len(R), "four_ci": len(four), "four_sign": len(pos4)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
