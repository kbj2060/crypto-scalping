#!/usr/bin/env python3
"""사용자가 **문장으로 서술한 규칙 그 자체**를 모델 없이 검정한다 (2026-09-08).

GBM 이 피쳐를 쓸 수 있는지와, *"CVD 급등인데 못 뚫으면 흡수 -> 되돌림"* 같은 **서술된 조건부 규칙**이
실제로 맞는지는 다른 질문이다. 트리 모델은 약한 주변효과를 상호작용으로 살려내지 못할 수 있고,
반대로 사람이 읽을 수 있는 규칙이 없으면 대시보드 표시로도 못 쓴다.

규칙마다 (1) 발동 커버리지 (2) 실제 돌파율 (3) 일군집 부트스트랩 CI 를 낸다.
⚠️X4 규칙: 커버리지를 항상 함께 보고한다. 결과로 걸러진 부분집합이 아님을 보이기 위해서다.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tmp/eth_breakout_reversal_20260908/dataset_v3.parquet"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
SEED = 20260908


def day_ci(v, day, rng, B=3000):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def main() -> int:
    rng = np.random.default_rng(SEED)
    A = pd.read_parquet(SRC); A["timestamp"] = pd.to_datetime(A["timestamp"])
    d = A[(A.anchor == "first_fire") & (A.T_mult == 0.75)].reset_index(drop=True)
    e = d[d.split.isin(WINS)].reset_index(drop=True)
    y = e["y"].to_numpy(); day = e["timestamp"].dt.floor("D").to_numpy()
    N = len(e)
    print(f"모집단: first_fire T=0.75 · 표본외 3창 · n={N:,} · 전체 돌파율 {y.mean():.4f}\n")

    def q(col, lo, hi):
        v = e[col].to_numpy(float)
        r = pd.Series(v).rank(pct=True).to_numpy()
        return np.isfinite(v) & (r >= lo) & (r < hi)

    RULES = [
        ("① 흡수: CVD 상위40% & 흡수계수 상위40%", "되돌림",
         lambda: q("v3c_cvd_r", .6, 1.01) & q("v3c_burn", .6, 1.01)),
        ("① 반대: CVD 하위40% & 흡수계수 하위40%", "돌파",
         lambda: q("v3c_cvd_r", 0, .4) & q("v3c_burn", 0, .4)),
        ("② OI 급감 하위20% (청산연료)", "되돌림", lambda: q("v3o_doi3", 0, .2)),
        ("② OI 급증 상위20% (신규자금)", "돌파", lambda: q("v3o_doi3", .8, 1.01)),
        ("② OI 급감 & 발현 빠름(속도 상위40%)", "되돌림",
         lambda: q("v3o_doi3", 0, .2) & q("f_speed", .6, 1.01)),
        ("③ 가짜벽: 앞벽 두껍(상위40%) & 취소중(상위40%)", "돌파",
         lambda: q("v3b_wall02", .6, 1.01) & q("v3b_ahead_drop", .6, 1.01)),
        ("③ 진짜벽: 앞벽 두껍(상위40%) & 취소없음(하위40%)", "되돌림",
         lambda: q("v3b_wall02", .6, 1.01) & q("v3b_ahead_drop", 0, .4)),
        ("③ 얇은 앞벽 하위20% (진공 돌파)", "돌파", lambda: q("v3b_wall02", 0, .2)),
        ("④ 3축 합의: 흡수↑ & OI↓ & 앞벽 두껍", "되돌림",
         lambda: q("v3c_burn", .6, 1.01) & q("v3o_doi3", 0, .4) & q("v3b_wall02", .6, 1.01)),
    ]
    out = []
    print(f"{'규칙':<44} {'예측':<5} {'커버리지':>9} {'실제돌파율':>10} {'95%CI':>20} {'정확도':>8} {'판정'}")
    print("-" * 118)
    for nm, call, fn in RULES:
        m = fn()
        n = int(m.sum())
        if n < 200:
            print(f"{nm:<44} {call:<5} {n/N:>8.1%}  표본부족({n})"); continue
        br = y[m].mean()
        lo, hi = day_ci(y[m].astype(float), day[m], rng)
        acc = br if call == "돌파" else 1 - br
        # 규칙이 주장하는 방향으로 CI 가 0.5 를 배제하는가
        ok = (lo > 0.5) if call == "돌파" else (hi < 0.5)
        print(f"{nm:<44} {call:<5} {n/N:>7.1%}({n:>5}) {br:>10.4f} [{lo:.4f},{hi:.4f}] {acc:>8.4f} "
              f"{'✅CI 배제' if ok else '❌CI 0.5 포함'}")
        out.append(dict(rule=nm, call=call, n=n, cov=n / N, br=float(br), lo=lo, hi=hi,
                        acc=float(acc), ci_excl=bool(ok)))
    print("\n" + "-" * 118)
    print(f"CI 가 0.5 를 배제한 규칙: {sum(o['ci_excl'] for o in out)} / {len(out)}")
    print(json.dumps({"rules": len(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
