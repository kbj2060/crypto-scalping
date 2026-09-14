"""**L5 국소상대 워크포워드** — 부호가 돈다면 따라가면 된다 (2026-09-14, 사용자 *"살릴 순 없나"*).

## 왜 이걸 안 했나 (내 실수)
지금까지 방향 평가는 전부 **단일 분할**이었다: TRAIN(2024-03~2025-08) 한 번 학습하고 2022~23·
VAL·OOS·TEST 에 그대로 적용. 그런데 **2022~23 은 학습창보다 1~2년 «이전»** 이다 — 모델을 과거로
돌려 적용한 것이고, 관계가 시간에 따라 변하면 뒤로 적용하는 쪽이 더 크게 실패한다.
「2023 음수」를 불안정의 증거로 읽은 건 그래서 과했다.

그리고 반년 단위로 부호가 돈다면, **그 회전을 따라가도록 주기적으로 재학습**하는 게 답이다.
이 저장소의 Fresh-Forward 규칙이 요구하는 형태이기도 하다.

## 설계
`--every` 일마다 재학습하고, 학습은 **직전 `--lookback` 일**만 본다. 예측은 **항상 앞으로**.
🔴라벨 인과성: L5 는 중심 이동평균(창 W 앵커)을 빼므로 학습 시점 T 근처 라벨은 미래를 본다.
   그래서 학습 표본을 `T − (W/2 앵커 + 해결여유 30일)` 까지로 자른다. 안 자르면 재학습마다
   T 직전의 미래를 조금씩 훔친다.
비교군: 같은 워크포워드 틀에서 **원시 라벨(L0)** 도 함께 돌려 국소상대의 증분을 본다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_direction_barrier_label_20260914 as B  # noqa: E402
import research_eth_rl_gym_direction_ppo_20260914 as P  # noqa: E402
import rl_gym_direction_env_20260914 as G  # noqa: E402

OUT = ROOT / "data/research/eth_direction_barrier_label_20260914"
DAY = 288


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--win-anchors", type=int, default=3456, help="국소상대 창(앵커) — 0 이면 원시 라벨")
    ap.add_argument("--every", type=int, default=30, help="재학습 주기(일)")
    ap.add_argument("--lookback", type=int, default=365, help="학습에 쓰는 직전 일수")
    ap.add_argument("--stride", type=int, default=2, help="학습 앵커 간격(봉)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--start", default="2022-07-01", help="워크포워드 시작(그 전은 최초 학습분)")
    ap.add_argument("--tag", default="walkforward")
    a = ap.parse_args()
    B.FUNDING = False
    seeds = [int(x) for x in np.random.default_rng(20260914).integers(1, 1_000_000, size=a.seeds)]
    d, sm, win, S, cols, _ = P.prepare(G.DEFAULT_FAMILIES)
    c = d.close.to_numpy(float); hi = d.high.to_numpy(float); lo = d.low.to_numpy(float)
    ok = sm["ok"]; ts = d.timestamp.to_numpy()
    n = len(c)

    # 전 구간 라벨을 한 번만 만든다(평가는 매 봉, 학습은 stride 로 솎아 쓴다)
    print("전 구간 라벨 생성 …", flush=True)
    full = B.label_window(c, hi, lo, ok, 0, n - 1, 1, 0.03, 0.03)
    fidx, fy, fm, fb = full["idx"], full["y"], full["m"], full["bars"]
    pos = {int(i): k for k, i in enumerate(fidx)}
    print(f"  앵커 {len(fidx):,} · {str(ts[fidx[0]])[:10]} ~ {str(ts[fidx[-1]])[:10]}", flush=True)

    W = a.win_anchors
    i_start = int(np.searchsorted(ts, np.datetime64(a.start)))
    trades = []            # (진입봉, 방향기여bp, 순손익bp)
    free_at = -1
    model, next_fit = None, i_start
    for i in range(i_start, n - 1):
        if i >= next_fit:                       # ── 재학습 ─────────────────────
            # 🔴미래 차단: 중심창 절반 + 해결 여유 30일을 뺀 지점까지만 학습에 쓴다
            cut = i - (W * a.stride // 2 if W else 0) - 30 * DAY
            lo_tr = max(0, cut - a.lookback * DAY)
            m_tr = (fidx >= lo_tr) & (fidx < cut) & ((fidx - lo_tr) % a.stride == 0)
            if m_tr.sum() < 5000:
                next_fit = i + a.every * DAY; continue
            yy = fy[m_tr]
            if W:
                loc = pd.Series(yy).rolling(W, center=True, min_periods=W // 3).mean().to_numpy()
                yy = yy - np.where(np.isfinite(loc), loc, 0.0)
            ms = []
            for sd in seeds:
                mm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                   min_samples_leaf=200, l2_regularization=1.0,
                                                   random_state=sd)
                mm.fit(S[fidx[m_tr]], yy); ms.append(mm)
            model = ms; next_fit = i + a.every * DAY
            print(f"  재학습 {str(ts[i])[:10]} · 학습 {m_tr.sum():,}행 "
                  f"({str(ts[lo_tr])[:10]}~{str(ts[cut])[:10]})", flush=True)
        if model is None or i <= free_at or i not in pos:
            continue
        k = pos[i]
        pr = float(np.mean([mm.predict(S[i:i + 1])[0] for mm in model]))
        s = 1.0 if pr >= 0 else -1.0
        trades.append((i, s * fy[k], fm[k] + s * fy[k]))
        free_at = i + int(fb[k])

    tr = np.array([(t[1], t[2]) for t in trades])
    tidx = np.array([t[0] for t in trades])
    print(f"\n워크포워드 순차 거래 {len(tr):,}건 "
          f"({str(ts[tidx[0]])[:10]} ~ {str(ts[tidx[-1]])[:10]})")
    def line(name, mask):
        a_ = tr[mask]
        if len(a_) < 20:
            return
        se = a_[:, 0].std(ddof=1) / np.sqrt(len(a_))
        print(f"  {name:>10} n {len(a_):>5,} · 방향 {a_[:, 0].mean():>+7.2f}bp ± {se:>5.2f} · "
              f"t {a_[:, 0].mean() / se:>+5.2f} · 순손익 {a_[:, 1].mean():>+7.2f}bp")
    line("전체", np.ones(len(tr), bool))
    yrs = pd.to_datetime(ts[tidx])
    for y in sorted(set(yrs.year)):
        for h, sel in (("H1", yrs.month <= 6), ("H2", yrs.month > 6)):
            line(f"{y} {h}", (yrs.year == y) & sel)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(
        {"win_anchors": W, "every": a.every, "lookback": a.lookback, "n": len(tr),
         "dir_bp": float(tr[:, 0].mean()), "net_bp": float(tr[:, 1].mean()),
         "t": float(tr[:, 0].mean() / (tr[:, 0].std(ddof=1) / np.sqrt(len(tr)))),
         "trades": [[int(i), float(a_), float(b_)] for i, (a_, b_) in zip(tidx, tr)]},
        indent=1, ensure_ascii=False))
    print(f"저장: {OUT/(a.tag+'.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
