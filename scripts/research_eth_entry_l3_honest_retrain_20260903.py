#!/usr/bin/env python3
"""진입 필터 **L3(정직 라벨) 재학습** -- HGB vs TabPFN (2026-09-03).

배경: `docs/experiments/eth_entry_intrabar_fill_bar_credit_artifact_20260903.md`.
기존 라벨 L0는 **체결 봉의 유리한 폭(중앙 1.76 ATR)을 크레딧**해 미래참조였다. 1분봉으로
해상한 L3에서 전체 후보 팔 PF가 3.66 → **0.99**로 무너졌고, v1/v2 동결 수치와 대조군 7종,
아키텍처 스윕이 모두 그 라벨 위에 있었다.

여기서 다시 묻는다: **정직한 라벨에서 필터가 작동하는가.**

⚠️**구조는 재최적화하지 않는다**(depth 3.0 / wait 6 / slots 4 / SL3·ARM1·Trail0.1).
   그것들은 L0에서 골랐지만, 지금 다시 고르면 VAL/OOS/HOLDOUT을 또 태운다. 물을 것은 필터
   하나이고, 구조 재선택은 신선한 창을 확보한 뒤의 일이다.
⚠️**임계값은 TRAIN에서만** 유도한다. 유지율 스윕은 **보고용**이고 선택 근거로 쓰지 않는다
   (주 설정은 상속된 0.2037).
⚠️**HOLDOUT은 이미 여러 번 소진됐다.** 여기 숫자는 진단이지 표본외 성과가 아니다.
   그리고 1분봉이 2026-07-31까지라 HOLDOUT 후보가 2,585 → 2,212로 줄었다.
"""
from __future__ import annotations

import json
import os
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

from research_eth_entry_b1_cumulative_arms_20260903 import SEEDS, HP  # noqa: E402
from research_eth_entry_b6_expand_20260903 import slotN  # noqa: E402

B6 = ROOT / "tmp/eth_entry_b6_expand_20260903"
L3D = ROOT / "tmp/eth_entry_1m_resolved_20260903/labels_1m.csv"
OUT = ROOT / "tmp/eth_entry_l3_retrain_20260903"
NSLOT, KEEP0, SUB = 4, 0.2037, 18000
B_RND = 300
RNG = np.random.default_rng(20260903)
W3 = ("VAL", "OOS", "HOLDOUT")


def log(m): print(f"[l3] {m}", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    LAB = pd.read_csv(L3D, parse_dates=["timestamp"])
    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    W = D[((D.depth == 3.0) & (D.btf <= 6)).to_numpy()].reset_index(drop=True)
    W = W.merge(LAB[["timestamp", "signal", "arm", "y_L3"]], on=["timestamp", "signal", "arm"],
                how="left")
    W = W[np.isfinite(W.y_L3)].reset_index(drop=True)
    y = W["y_L3"].to_numpy(float)
    log(f"후보 팔 {len(W):,} · " + " ".join(f"{k} {v:,}" for k, v in W.split.value_counts().items()))

    cfg = json.loads((ROOT / "tmp/eth_causal_population_metalabel_20260902/config.json").read_text())
    base = [c for c in cfg["features"] if c != "is_bottom"]
    excl = set(base + ["arm", "sig_id", "atr_pct", "depth", "y", "y_L3", "split", "timestamp",
                       "i", "side", "signal", "fi", "ei", "btf", "lim", "sd", "pred"])
    R = [c for c in W.columns if c.endswith("_r136")] + \
        [c for c in W.columns if c not in excl and not c.endswith("_r136")]
    R = list(dict.fromkeys([c for c in R if W[c].dtype.kind in "fiub"]))
    FE = list(dict.fromkeys(base + ["arm", "sig_id", "atr_pct", "depth"] + R))
    X = W[FE].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    tr = (W.split == "TRAIN").to_numpy()
    X = X.fillna(X[tr].median())
    M = {w: (W.split == w).to_numpy() for w in W3}
    log(f"피쳐 {len(FE)} · TRAIN {int(tr.sum()):,}")

    def run(mask, keep=None):
        d = W[mask] if keep is None else W[mask & keep]
        t = slotN(d.assign(y=d.y_L3), NSLOT)
        return (float(np.mean(t) * 1e4) if len(t) else 0.0), int(len(t))

    nf = {w: run(M[w]) for w in W3}
    log("① 무필터 " + " ".join(f"{w} {nf[w][0]:+.2f}(n{nf[w][1]})" for w in W3))

    scores = {}
    # --- HGB 5시드 ---
    scores["HGB 회귀 5시드"] = np.mean(
        [HistGradientBoostingRegressor(random_state=s, loss="squared_error", **HP)
         .fit(X[tr], y[tr]).predict(X) for s in SEEDS], axis=0)
    log("HGB 완료")

    # --- TabPFN 분류 5멤버 ---
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TABPFN_TOKEN="):
                os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip('"')
    from tabpfn import TabPFNClassifier
    lab = (y > 0.0040).astype(int)
    itr = np.flatnonzero(tr)
    Xn = X.to_numpy(np.float32)
    ps = []
    for k, sd in enumerate(SEEDS):
        rs = np.random.default_rng(sd).choice(itr, size=min(SUB, len(itr)), replace=False)
        m = TabPFNClassifier(device="cuda", random_state=sd)
        m.fit(Xn[rs], lab[rs])
        ps.append(m.predict_proba(Xn)[:, 1])
        log(f"  TabPFN 멤버{k} 완료")
    scores["TabPFN 분류 5멤버"] = np.mean(ps, axis=0)

    print(f"\n{'':22s}" + "".join(f"{w:>14s}" for w in W3))
    print(f"{'① 무필터':22s}" + "".join(f"{nf[w][0]:+9.2f}(n{nf[w][1]:4d})" for w in W3))
    res = {}
    for nm, p in scores.items():
        thr = float(np.quantile(p[tr], 1 - KEEP0))
        keep = p > thr
        r = {w: run(M[w], keep) for w in W3}
        res[nm] = {"thr": thr, "perf": r, "keep": keep}
        print(f"{nm:22s}" + "".join(f"{r[w][0]:+9.2f}(n{r[w][1]:4d})" for w in W3))

    # --- 유지율 민감도 (보고용, 선택 아님) ---
    print(f"\n=== 유지율 민감도 (TRAIN에서 유도, 보고용) ===")
    print(f"{'모델':22s}{'유지율':>8s}" + "".join(f"{w:>12s}" for w in W3))
    for nm, p in scores.items():
        for kf in (0.05, 0.10, KEEP0, 0.30):
            thr = float(np.quantile(p[tr], 1 - kf))
            r = {w: run(M[w], p > thr) for w in W3}
            print(f"{nm:22s}{kf:8.3f}" + "".join(f"{r[w][0]:+8.2f}({r[w][1]:3d})" for w in W3))

    # --- 무작위 필터 대조군 ---
    print(f"\n=== 무작위 필터 대조군 (B={B_RND}, 같은 유지율) ===")
    for nm in res:
        keep = res[nm]["keep"]
        for w in W3:
            m = M[w]; kf = float(keep[m].mean()); idx = np.flatnonzero(m)
            rs = []
            for _ in range(B_RND):
                r0 = np.zeros(len(W), bool)
                r0[RNG.choice(idx, size=max(int(round(kf*len(idx))), 1), replace=False)] = True
                rs.append(run(m, r0)[0])
            rs = np.array(rs); act = res[nm]["perf"][w][0]
            pv = float((rs >= act).mean())
            print(f"  {nm:22s} {w:8s} 실제 {act:+7.2f} · 무작위 {rs.mean():+7.2f} "
                  f"· **p={pv:.3f}** {'✅' if pv < 0.05 else '❌'}")

    json.dump({"no_filter": nf,
               "models": {k: {"thr": v["thr"], "perf": v["perf"]} for k, v in res.items()}},
              open(OUT / "result.json", "w"), ensure_ascii=False, indent=2, default=str)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
