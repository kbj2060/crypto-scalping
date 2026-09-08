#!/usr/bin/env python3
"""돌파/되돌림 진입의 **메이커 체결 가능성 실측** (2026-09-08).

사용자: *"메이커 체결 가능성 실측부터 진행해줘"*

## 왜 기존 실측(7.8bp)으로는 답이 안 되나
`maker_fill_shadow_worker.py` 가 잰 2.76bp/leg 는 **peg-maker** -- 최우선호가를 따라다니다
120초 안에 안 되면 테이커로 전환하는 정책이다. 이 전략이 쓰려는 건 다른 물건이다:
**가격이 어차피 닿을 레벨(트리거 레벨)에 지정가를 미리 걸어두는 것**. 추종도 타임아웃도 없다.
그래서 물어야 할 건 비용이 아니라 **"닿았는데 체결되는가"** 다.

## 측정
지정가는 시장이 내 가격을 **관통**해야 안전하게 체결된다(닿기만 하면 내 앞 큐가 남는다).
트리거 레벨을 L, 발현 방향을 sgn 이라 할 때
    상승 발현(sgn>0) -> L 에 매도 지정가.  관통 = high − L
    하락 발현(sgn<0) -> L 에 매수 지정가.  관통 = L − low
을 **트리거 분**과 **그 뒤 k분**에 대해 잰다. 틱(0.01 USDT)으로 환산해 보고한다.
  · 관통 ≥ 1틱  : 시장이 내 가격을 지나갔다 -- 큐 앞이 소진됐을 개연성이 높다
  · 관통 = 0    : 정확히 닿기만 했다 -- 큐 앞이 남았으면 미체결
보수적 기준은 **관통 ≥ 1틱**이다.

⚠️이건 상한이 아니라 **필요조건**이다. 관통했어도 내 앞 큐가 그보다 크면 미체결일 수 있다.
   큐 깊이는 1분봉으로 알 수 없다 -- 그래서 체결 **가능성의 상한**을 재는 것이고,
   실제 체결률은 이 값 이하다. 하한은 라이브 섀도우로만 나온다.
"""
from __future__ import annotations
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "8")
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

MY = ROOT / "tmp/eth_breakout_atr_state_20260908_s1"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
TICK = 0.01            # ETHUSDT 선물 호가단위
TM, H = 0.75, 12
HOLD_MIN = (1, 2, 3, 5, 10)   # 지정가를 몇 분간 유지하는가


def main() -> int:
    d = pd.read_parquet(MY / "dataset_v2.parquet")
    d = d[(d.anchor == "first_fire") & (d.T_mult == TM)].sort_values("timestamp").reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    eth = B._load_kl(B.ETH_KL)
    ts5 = eth["timestamp"].to_numpy(); O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy()
    hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float)

    sgn = np.where(d["dir_up"].to_numpy() > 0, 1.0, -1.0)
    bi = d["bar_idx"].to_numpy(); T = d["T_atr"].to_numpy(float)
    ref = O5[np.minimum(bi + 1, len(O5) - 1)]
    L = ref * (1 + sgn * T)                      # 트리거 레벨 = 진입가
    s1 = np.searchsorted(ts1, d["timestamp"].to_numpy())
    bt = np.searchsorted(ts5, ts1[np.clip(s1, 0, len(ts1) - 1)], side="right") - 1
    ok = (s1 > 0) & (s1 + H * 5 < len(ts1)) & (bt + H < len(C5)) & (bt >= 1)
    sp = d["split"].to_numpy()
    idx = np.flatnonzero(ok)
    print(f"트리거 {len(idx):,}건 · 레벨 중앙값 {np.median(L[idx]):.2f} · 1틱 = "
          f"{TICK/np.median(L[idx])*1e4:.3f}bp\n", flush=True)

    print("=" * 96)
    print(f"{'유지분':>7}{'관통≥1틱':>10}{'관통≥2틱':>10}{'관통≥5틱':>10}{'관통=0':>9}"
          f"{'관통 중앙':>11}{'관통 평균':>11}")
    print("=" * 96)
    rows = []
    for k in HOLD_MIN:
        span = np.arange(k)
        J = s1[idx][:, None] + span[None, :]
        HI = hi1[np.clip(J, 0, len(hi1) - 1)].max(axis=1)
        LO = lo1[np.clip(J, 0, len(lo1) - 1)].min(axis=1)
        pen = np.where(sgn[idx] > 0, HI - L[idx], L[idx] - LO)   # 관통 깊이(가격)
        pt = pen / TICK
        r = dict(hold=k, n=len(idx),
                 ge1=float((pt >= 1).mean()), ge2=float((pt >= 2).mean()),
                 ge5=float((pt >= 5).mean()), eq0=float((pt < 1).mean()),
                 med=float(np.median(pt)), mean=float(pt.mean()))
        rows.append(r)
        print(f"{k:>6}분{r['ge1']*100:>9.1f}%{r['ge2']*100:>9.1f}%{r['ge5']*100:>9.1f}%"
              f"{r['eq0']*100:>8.1f}%{r['med']:>10.1f}틱{r['mean']:>10.1f}틱")

    # 창별 안정성 (1분 유지 기준)
    span = np.arange(1)
    J = s1[idx][:, None] + span[None, :]
    HI = hi1[np.clip(J, 0, len(hi1) - 1)].max(axis=1)
    LO = lo1[np.clip(J, 0, len(lo1) - 1)].min(axis=1)
    pt1 = np.where(sgn[idx] > 0, HI - L[idx], L[idx] - LO) / TICK
    print("\n=== 창별 (트리거 분 안에서만) ===")
    for w in ("TRAIN", "VAL", "OOS", "HOLDOUT_SPENT"):
        m = sp[idx] == w
        if m.sum() < 50: continue
        print(f"  {w:>14}: n={m.sum():>6,} · 관통≥1틱 {(pt1[m] >= 1).mean()*100:>5.1f}% "
              f"· 중앙 {np.median(pt1[m]):>5.1f}틱")

    # 발현 방향별 -- 되돌림 팔(관통한 쪽에 반대로 선다)이 실제로 쓰는 값
    print("\n=== 발현 방향별 (트리거 분) ===")
    for nm, m in (("상승 발현(매도 지정가)", sgn[idx] > 0), ("하락 발현(매수 지정가)", sgn[idx] < 0)):
        print(f"  {nm}: n={m.sum():>6,} · 관통≥1틱 {(pt1[m] >= 1).mean()*100:>5.1f}% "
              f"· 중앙 {np.median(pt1[m]):>5.1f}틱")

    json.dump(rows, open(MY / "maker_fill_feasibility.json", "w"))
    print("\n⚠️이 값은 체결률의 **상한**이다 -- 관통해도 내 앞 큐가 더 크면 미체결이다.")
    print("   1분봉으로는 큐 깊이를 알 수 없다. 하한은 라이브 섀도우로만 나온다.")
    print(json.dumps({"done": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
