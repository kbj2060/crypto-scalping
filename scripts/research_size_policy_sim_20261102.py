#!/usr/bin/env python3
"""크기 정책 시뮬레이션 (2026-11-02): 실제 XRP 5분 경로 위에서 고정 배수 vs 3배 상한 vs 1%/SL 의 파산·낙폭·성장.
사전등록 docs/prereg/size_policy_sim_20261102.md. 파라미터 전부 고정.   python scripts/research_size_policy_sim_20261102.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sp = importlib.util.spec_from_file_location("rf", ROOT / "scripts" / "research_regime_f_20261102.py")
rf = importlib.util.module_from_spec(sp)
sp.loader.exec_module(rf)
OUT = ROOT / "tmp" / "size_policy_sim_20261102"
SL, TP = 0.005, 0.004                 # 사용자식 기하: 손절 0.5% · 익절 0.4% (최근 평균 이익 ~40bp)
POLICIES = {"7x": 7.0, "5x": 5.0, "3x": 3.0, "2x(=1%/SL)": 0.01 / SL}
EDGES = (0.0, 5.0, 10.0, 20.0, 40.0)  # 거래당 주입 엣지 bp (명목 대비)
DAYS, PER_DAY, NSIM, RUIN_DD, SEED = 30, 4, 20000, 0.30, 20261102
WINDOWS = {"학습구간 2024-25": ("2024-01-15", "2026-01-01"), "2026": ("2026-01-01", "2100-01-01")}


def trade_returns(coin="XRP"):
    """매 시 정각 롱·숏 진입의 거래 수익률(명목 대비, 비용 포함) 배열 R[day,hour,side]."""
    df = rf.load(coin)
    df = df.rename(columns={})
    i = np.arange(12, len(df) - rf.H - 3, 12)
    i = i[df.index[i].minute == 55]
    e = df.o.to_numpy()[i + 1]
    R = []
    for side in (1, -1):
        s = np.full(len(i), side)
        net, _ = rf.barrier(df, i, s, e * (1 + side * TP), e * (1 - side * SL))
        R.append(net / 1e4)
    R = np.stack(R, 1)                                    # (N,2) 비용 포함, 무효 거래는 0
    ts = df.index[i + 1]
    day = ts.normalize()
    out = pd.DataFrame({"day": day, "hour": ts.hour, "long": R[:, 0], "short": R[:, 1]})
    days = sorted(out.day.unique())
    arr = np.zeros((len(days), 24, 2))
    di = {d: k for k, d in enumerate(days)}
    arr[out.day.map(di).to_numpy(), out.hour.to_numpy(), 0] = out.long.to_numpy()
    arr[out.day.map(di).to_numpy(), out.hour.to_numpy(), 1] = out.short.to_numpy()
    return days, arr


def simulate(arr, edge_bp, mult, rng):
    """DAYS 연속 일 × PER_DAY 무작위 (시각,방향) 거래. 반환 (최종 자산, MDD) 각 NSIM."""
    D = arr.shape[0]
    start = rng.integers(0, D - DAYS, NSIM)
    eq = np.ones(NSIM); pk = np.ones(NSIM); mdd = np.zeros(NSIM)
    for d in range(DAYS):
        for _ in range(PER_DAY):
            h, s = rng.integers(0, 24, NSIM), rng.integers(0, 2, NSIM)
            r = arr[start + d, h, s] + edge_bp / 1e4
            eq = np.maximum(eq * (1 + mult * r), 0.0)
            pk = np.maximum(pk, eq)
            mdd = np.maximum(mdd, 1 - eq / pk)
    return eq, mdd


def block_ci(x, key, rng, n=2000):
    """시작일 블록 부트스트랩이 아닌 시뮬레이션 간 표준 부트스트랩(시뮬은 독립 추출)."""
    idx = rng.integers(0, len(x), (n, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


def main():
    rng = np.random.default_rng(SEED)
    days, arr = trade_returns()
    days = pd.DatetimeIndex(days)
    out = {"n_days": len(days), "gross_mean_bp": float(arr[arr != 0].mean() * 1e4), "nonzero_share": float((arr != 0).mean())}
    for w, (a, b) in WINDOWS.items():
        m = (days >= a) & (days < b)
        sub = arr[m]
        for edge in EDGES:
            res = {}
            for name, mult in POLICIES.items():
                eq, mdd = simulate(sub, edge, mult, rng)
                res[name] = dict(p_dd30=float((mdd >= RUIN_DD).mean()), p_ruin80=float((mdd >= 0.8).mean()), median=float(np.median(eq)), mean_log=float(np.mean(np.log(np.maximum(eq, 1e-6)))),
                                 p_loss=float((eq < 1).mean()), ci_dd30=block_ci((mdd >= RUIN_DD).astype(float), 0, rng).tolist())
            out[f"{w}|edge{edge:g}bp"] = res
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "result.json", "w"), indent=1, default=float)
    for k, v in out.items():
        if isinstance(v, dict):
            print(k, {p: (round(r["p_dd30"], 3), round(r["median"], 3), round(r["mean_log"], 3)) for p, r in v.items()})
        else:
            print(k, v)


if __name__ == "__main__":
    main()
