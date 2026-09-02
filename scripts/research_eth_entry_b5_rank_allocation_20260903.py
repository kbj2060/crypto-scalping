#!/usr/bin/env python3
"""B5: 임계값 대신 **순위 기반 슬롯 배분** (2026-09-03).

B4 문제: τ 필터가 무작위 필터를 이기긴 하는데(3창 백분위 100%), VAL 최적 τ가 격자 끝(30bp)이고
거기서 n=32~67, 독립 블록 9일로 표본이 무너진다. 검정력이 없다.

접근 3 -- 자르지 말고 배분한다:
  임계값으로 주문을 버리는 대신, **대기 중인 주문 중 예측값 상위 M개만 살려두고 나머지는 취소**한다.
  포지션이 열려 있는 동안은 용량이 없으므로 대기 주문을 전부 취소한다(현실적·보수적).
  M을 낮추면 "가장 좋은 주문만 체결될 수 있게" 되고, **표본을 버리지 않으면서** 순위를 쓴다.

  대조군: 같은 M으로 **무작위 M개**를 살리는 정책. 순위가 값을 더하는지 분리한다.

이벤트 시뮬 (봉 단위 전진, 완전 인과):
  발주 = 트리거 봉, 만료 = 발주 + WAIT
  매 봉: 신규 추가 → 만료 제거 → 포지션 있으면 전부 취소 → 없으면 상위 M만 유지
        → 그 봉에 체결되는 것 중 예측 최고를 잡음
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from research_eth_entry_b1_cumulative_arms_20260903 import SEEDS, HP, stat  # noqa: E402
from research_eth_entry_direction_oracle_ceiling_20260903 import WAIT  # noqa: E402
from research_eth_regime_s12k3_label_train_20260902 import GBM3_MODEL_PATH, load_frame  # noqa: E402

OUT = ROOT / "tmp/eth_entry_b5_rank_20260903"
MS = [1, 2, 3, 5, 10, 20, 10**9]
B_RND = 60
RNG = np.random.default_rng(20260903)


def log(m): print(f"[b5] {m}", flush=True)


def run(orders, M, key):
    """orders: (place_i, expire_i, fill_i, exit_i, score, y) 정렬됨. key='score'면 상위M, 'rand'면 무작위M."""
    if not len(orders): return np.array([])
    ptr, active, pos_until, taken = 0, [], -1, []
    lo = orders[0][0]; hi = max(o[3] for o in orders) + 1
    for t in range(lo, hi):
        while ptr < len(orders) and orders[ptr][0] == t:
            active.append(orders[ptr]); ptr += 1
        if t <= pos_until:
            active = []                       # 포지션 중엔 용량 없음 → 전부 취소
            continue
        active = [o for o in active if o[1] >= t]
        if len(active) > M:
            if key == "score":
                active = sorted(active, key=lambda o: -o[4])[:M]
            else:
                idx = RNG.choice(len(active), M, replace=False)
                active = [active[i] for i in idx]
        cand = [o for o in active if o[2] == t]
        if cand:
            best = max(cand, key=lambda o: o[4]) if key == "score" else cand[RNG.integers(len(cand))]
            taken.append(best[5]); pos_until = best[3]
            active = [o for o in active if o is not best]
    return np.asarray(taken, float)


def main() -> int:
    d = pd.read_csv(ROOT / "tmp/eth_entry_b1_20260903/arm_rows.csv", parse_dates=["ts"])
    src = joblib.load(GBM3_MODEL_PATH); cols = src["feature_cols"]; med = src["feature_medians"]
    rf = load_frame()
    x = rf[["timestamp"] + cols].copy()
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med.get(c, 0.0))
    x = x.rename(columns={"timestamp": "ts"})
    dup = [c for c in cols if c in d.columns]
    x = x.rename(columns={c: c + "_r136" for c in dup})
    R136 = [(c + "_r136" if c in dup else c) for c in cols]
    d = d.merge(x, on="ts", how="left").dropna(subset=R136).reset_index(drop=True)
    cfg = json.loads((ROOT / "tmp/eth_causal_population_metalabel_20260902/config.json").read_text())
    base = [c for c in cfg["features"] if c != "is_bottom"]
    FEATS = base + ["arm", "sig_id", "sig_dir", "atr"] + R136
    X = d[FEATS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    tr = (d.split == "TRAIN").to_numpy()
    X = X.fillna(X[tr].median())
    per_seed = {s: HistGradientBoostingRegressor(random_state=s, **HP)
                .fit(X[tr], d.loc[tr, "y"]).predict(X) for s in SEEDS}
    d["pred"] = np.mean([per_seed[s] for s in SEEDS], axis=0)

    d["place_i"] = d.ts_i.astype(int)
    d["expire_i"] = d.place_i + WAIT
    d["fi"] = np.where(d.filled.astype(bool), d.fill_i.astype(float), np.inf)
    d["ei"] = np.where(d.filled.astype(bool), d.exit_i.astype(float), np.inf)
    log(f"행 {len(d):,} | 피쳐 {len(FEATS)} | WAIT={WAIT}")

    def orders_of(w, score):
        o = [(int(a), int(b), (int(f) if np.isfinite(f) else -1), (int(e) if np.isfinite(e) else -1),
              float(s), float(y))
             for a, b, f, e, s, y in zip(w.place_i, w.expire_i, w.fi, w.ei, score, w.y)]
        return sorted(o)

    log("\n=== M 스윕 (M=∞는 기존 선착순 = 기준선) ===")
    print(f"{'M':>5s} | " + " | ".join(f"{w:>26s}" for w in ("VAL", "OOS", "HOLDOUT")))
    rows = []
    for M in MS:
        cells, row = [], {"M": (M if M < 10**8 else -1)}
        for wn in ("VAL", "OOS", "HOLDOUT"):
            w = d[d.split == wn]
            v = run(orders_of(w, w.pred.to_numpy()), M, "score")
            nn, m, pf = stat(v)
            row[f"{wn}_bp"] = round(m, 2); row[f"{wn}_n"] = nn
            cells.append(f"{m:+7.2f}bp n={nn:4d} PF{pf:5.2f}")
        rows.append(row)
        print(f"{row['M']:5d} | " + " | ".join(f"{c:>26s}" for c in cells))
    r = pd.DataFrame(rows)
    fin = r[r.M > 0]
    bestM = int(fin.loc[fin.VAL_bp.idxmax(), "M"])
    br = fin.loc[fin.VAL_bp.idxmax()]
    log(f"\n⭐VAL 최적 M = {bestM} → OOS {br.OOS_bp:+.2f}bp (n{int(br.OOS_n)}) · "
        f"HOLDOUT {br.HOLDOUT_bp:+.2f}bp (n{int(br.HOLDOUT_n)})")

    log(f"\n=== 무작위 M 대조군 (같은 M={bestM}, B={B_RND}) ===")
    ctrl = {}
    for wn in ("VAL", "OOS", "HOLDOUT"):
        w = d[d.split == wn]
        real = float(r.loc[r.M == bestM, f"{wn}_bp"].iloc[0])
        rr = [stat(run(orders_of(w, w.pred.to_numpy()), bestM, "rand"))[1] for _ in range(B_RND)]
        rr = np.array(rr)
        ctrl[wn] = {"real": real, "rnd_mean": float(rr.mean()),
                    "lo": float(np.quantile(rr, .025)), "hi": float(np.quantile(rr, .975)),
                    "p": float((rr >= real).mean())}
        log(f"  {wn:8s} 실제 {real:+7.2f} vs 무작위M {rr.mean():+7.2f} "
            f"[{np.quantile(rr,.025):+.2f},{np.quantile(rr,.975):+.2f}] → p={float((rr>=real).mean()):.3f}")

    log("\n=== 시드별 (M 고정) ===")
    for wn in ("VAL", "OOS"):
        w = d[d.split == wn]; msk = (d.split == wn).to_numpy()
        base_inf = float(r.loc[r.M == -1, f"{wn}_bp"].iloc[0])
        vals = [stat(run(orders_of(w, per_seed[s][msk]), bestM, "score"))[1] for s in SEEDS]
        log(f"  {wn:5s} " + ", ".join(f"{v:+.2f}" for v in vals) +
            f"  → 선착순({base_inf:+.2f}) 초과 {sum(v>base_inf for v in vals)}/5")

    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "m_sweep.csv", index=False)
    json.dump({"best_M": bestM, "controls": ctrl, "WAIT": WAIT}, open(OUT / "b5_report.json", "w"), indent=2)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
