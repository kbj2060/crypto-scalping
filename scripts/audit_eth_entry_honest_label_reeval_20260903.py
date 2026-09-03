#!/usr/bin/env python3
"""전수조사 ③: **정직한 라벨 위에서 진입 모델을 다시 세운다** (2026-09-03).

L0(체결봉 전부 크레딧)가 미래참조로 확인됐으므로, 그 위에 세운 모든 결론이 무효다.
여기서는 1분해상 라벨(L3, 없으면 L1)로 **처음부터 다시** 묻는다:

  ① 필터 없이 양팔을 슬롯 제약으로 굴리면 얼마인가 (기계 자체의 엣지)
  ② 모델 필터가 그것을 이기는가
  ③ 모델 필터가 **같은 유지율의 무작위 필터**를 이기는가 (B=200)
  ④ 피쳐를 한 봉 밀어도(stale) 성과가 유지되는가 -- 유지되면 피쳐는 타이밍 정보를 안 담는 것이고,
     붕괴하면 타이밍이 핵심(또는 미래참조 잔재)이라는 뜻이다

⚠️임계값은 **TRAIN에서만** 유도한다(창별 매칭 금지 -- 라이브 불가).
⚠️모델은 HGB 5시드. TabPFN은 한 조합에 수십 분이라 여기서는 쓰지 않는다 -- 라벨이 바뀐 마당에
   학습기 비교는 그 다음 문제다.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from research_eth_entry_b1_cumulative_arms_20260903 import SEEDS, HP, stat  # noqa: E402
from research_eth_entry_b6_expand_20260903 import slotN  # noqa: E402

B6 = ROOT / "tmp/eth_entry_b6_expand_20260903"
L1D = ROOT / "tmp/eth_entry_intrabar_audit_20260903/labels.csv"
L3D = ROOT / "tmp/eth_entry_1m_resolved_20260903/labels_1m.csv"
OUT = ROOT / "tmp/eth_entry_honest_reeval_20260903"
NSLOT, B_RND = 4, 200
RNG = np.random.default_rng(20260903)


def log(m): print(f"[reeval] {m}", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    src = L3D if L3D.exists() else L1D
    LAB = pd.read_csv(src, parse_dates=["timestamp"])
    col = "y_L3" if "y_L3" in LAB.columns and LAB["y_L3"].notna().sum() > 1000 else "y_L1"
    log(f"라벨 출처 {src.name} · 사용 컬럼 **{col}** (유효 {int(LAB[col].notna().sum()):,})")

    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    sel = ((D.depth == 3.0) & (D.btf <= 6)).to_numpy()
    W = D[sel].reset_index(drop=True)
    W = W.merge(LAB[["timestamp", "signal", "arm", col]], on=["timestamp", "signal", "arm"],
                how="left")
    W = W[np.isfinite(W[col])].reset_index(drop=True)
    W["yh"] = W[col].to_numpy(float)
    log(f"후보 팔 {len(W):,} · " + " ".join(f"{k} {v:,}" for k, v in W.split.value_counts().items()))

    cfg = json.loads((ROOT / "tmp/eth_causal_population_metalabel_20260902/config.json").read_text())
    base = [c for c in cfg["features"] if c != "is_bottom"]
    excl = set(base + ["arm", "sig_id", "atr_pct", "depth", "y", "yh", col, "split",
                       "timestamp", "i", "side", "signal", "fi", "ei", "btf", "lim", "sd", "pred"])
    R = [c for c in W.columns if c.endswith("_r136")] + \
        [c for c in W.columns if c not in excl and not c.endswith("_r136")]
    R = list(dict.fromkeys([c for c in R if W[c].dtype.kind in "fiub"]))
    FE = list(dict.fromkeys(base + ["arm", "sig_id", "atr_pct", "depth"] + R))
    X = W[FE].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    tr = (W.split == "TRAIN").to_numpy()
    X = X.fillna(X[tr].median())
    y = W["yh"].to_numpy()
    Wm = {w: (W.split == w).to_numpy() for w in ("TRAIN", "VAL", "OOS", "HOLDOUT")}
    log(f"피쳐 {len(FE)}")

    def run(mask, keep=None):
        d = W[mask] if keep is None else W[mask & keep]
        t = slotN(d.assign(y=d.yh), NSLOT)
        return (float(np.mean(t) * 1e4) if len(t) else 0.0), int(len(t))

    print(f"\n{'':30s}" + "".join(f"{w:>12s}" for w in ("VAL", "OOS", "HOLDOUT")))
    # ① 무필터
    nf = {w: run(Wm[w]) for w in ("VAL", "OOS", "HOLDOUT")}
    print(f"{'① 무필터 (기계 자체)':30s}"
          + "".join(f"{nf[w][0]:+8.2f}({nf[w][1]:3d})" for w in ("VAL", "OOS", "HOLDOUT")))

    # ② 모델 필터 (TRAIN에서만 τ 유도 -- v1과 같은 유지율 0.2037 목표)
    p = np.mean([HistGradientBoostingRegressor(random_state=s, loss="squared_error", **HP)
                 .fit(X[tr], y[tr]).predict(X) for s in SEEDS], axis=0)
    tau = float(np.quantile(p[tr], 1 - 0.2037))
    keep = p > tau
    mf = {w: run(Wm[w], keep) for w in ("VAL", "OOS", "HOLDOUT")}
    print(f"{'② 모델 필터 (τ from TRAIN)':30s}"
          + "".join(f"{mf[w][0]:+8.2f}({mf[w][1]:3d})" for w in ("VAL", "OOS", "HOLDOUT")))

    # ④ 피쳐 한 봉 밀기 (stale)
    Xs = X.copy()
    shift_cols = [c for c in FE if c not in ("arm", "sig_id", "depth")]
    Xs[shift_cols] = X[shift_cols].shift(1).fillna(X[shift_cols].median())
    ps = np.mean([HistGradientBoostingRegressor(random_state=s, loss="squared_error", **HP)
                  .fit(Xs[tr], y[tr]).predict(Xs) for s in SEEDS], axis=0)
    ks = ps > float(np.quantile(ps[tr], 1 - 0.2037))
    sf = {w: run(Wm[w], ks) for w in ("VAL", "OOS", "HOLDOUT")}
    print(f"{'④ 피쳐 한 봉 밀기(stale)':30s}"
          + "".join(f"{sf[w][0]:+8.2f}({sf[w][1]:3d})" for w in ("VAL", "OOS", "HOLDOUT")))

    # ③ 무작위 필터 대조군
    print(f"\n=== ③ 무작위 필터 대조군 (B={B_RND}, 같은 유지율) ===")
    for w in ("VAL", "OOS", "HOLDOUT"):
        m = Wm[w]
        kf = float(keep[m].mean())
        rs = []
        for _ in range(B_RND):
            r = np.zeros(len(W), bool)
            idx = np.flatnonzero(m)
            r[RNG.choice(idx, size=max(int(round(kf * len(idx))), 1), replace=False)] = True
            rs.append(run(m, r)[0])
        rs = np.array(rs)
        pv = float((rs >= mf[w][0]).mean())
        print(f"  {w:8s} 실제 {mf[w][0]:+7.2f} · 무작위 평균 {rs.mean():+7.2f} "
              f"(유지율 {kf:.3f}) · **p={pv:.3f}** {'✅' if pv < 0.05 else '❌'}")

    json.dump({"label": col, "no_filter": nf, "model": mf, "stale": sf},
              open(OUT / "result.json", "w"), ensure_ascii=False, indent=2, default=str)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
