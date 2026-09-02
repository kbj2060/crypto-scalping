#!/usr/bin/env python3
"""B10: TabPFN 회귀 아키텍처 후보 (2026-09-03, 서버 GPU).

B9에서 로컬 4종(squared/absolute/quantile/winsorize)만 돌리고 TabPFN을 뺐다. 여기서 채운다.

쟁점: TRAIN 81,168행인데 TabPFN 컨텍스트 상한은 ~18k라 **78%를 버려야 한다.**
이 저장소 레짐 실측에서는 "전체 데이터 GBM(0.9108) > 컨텍스트 잘린 TabPFN 앙상블(0.8959)"이
나왔다 -- 데이터 손실이 학습기 우위의 4배였다. 여기서도 그런지 잰다.
손실을 줄이기 위해 **서로 다른 서브샘플 4개의 앙상블**로 돌린다(각 18k, 시드별 독립 추출).

⚠️예측 대상은 동결 설정 구간(depth 3.0 / btf<=6)만으로 제한한다 -- TabPFN 추론이
O(n_train x n_test)라 전 구간 예측은 비싸다.

⭐사전등록: B9와 동일 -- TabPFN이 현행 HGB squared를 **VAL·OOS 양 창에서** 못 이기면 현행 동결.
비교는 **동일 유지비율**로 한다(예측 스케일이 다르므로).
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

from research_eth_entry_b1_cumulative_arms_20260903 import SEEDS, HP, stat  # noqa: E402
from research_eth_entry_b6_expand_20260903 import slotN  # noqa: E402

B6 = ROOT / "tmp/eth_entry_b6_expand_20260903"
OUT = ROOT / "tmp/eth_entry_b10_tabpfn_20260903"
DEPTH, WAIT, TAU0, NSLOT = 3.0, 6, 0.0040, 4
SUBSAMPLE = 18000
N_MEMBERS = 4


def log(m): print(f"[b10] {m}", flush=True)


def main() -> int:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TABPFN_TOKEN="):
                os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip('"')
    try:
        from tabpfn import TabPFNRegressor
    except Exception as e:
        log(f"⛔ TabPFNRegressor import 실패: {type(e).__name__} {e}")
        log("   → TabPFN 회귀 미지원이면 이 축은 여기서 종료(분류 재구성은 별개 축)")
        return 1
    import inspect
    log(f"TabPFNRegressor 시그니처: {list(inspect.signature(TabPFNRegressor.__init__).parameters)[:12]}")

    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    cfg = json.loads((ROOT / "tmp/eth_causal_population_metalabel_20260902/config.json").read_text())
    base = [c for c in cfg["features"] if c != "is_bottom"]
    excl = set(base + ["arm", "sig_id", "atr_pct", "depth", "y", "split", "timestamp", "i",
                       "side", "signal", "fi", "ei", "btf", "lim", "sd", "pred"])
    R = [c for c in D.columns if c.endswith("_r136")] + \
        [c for c in D.columns if c not in excl and not c.endswith("_r136")]
    R = list(dict.fromkeys([c for c in R if D[c].dtype.kind in "fiub"]))
    FEATS = list(dict.fromkeys(base + ["arm", "sig_id", "atr_pct", "depth"] + R))
    X = D[FEATS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    tr = (D.split == "TRAIN").to_numpy()
    X = X.fillna(X[tr].median())
    y = D["y"].to_numpy()
    dsel = ((D.depth == DEPTH) & (D.btf <= WAIT)).to_numpy()
    log(f"행 {len(D):,} · TRAIN {int(tr.sum()):,} · 피쳐 {len(FEATS)} · 동결설정 구간 {int(dsel.sum()):,}")

    # ---- 기준선 HGB (전체 TRAIN) ----
    p0 = np.mean([HistGradientBoostingRegressor(random_state=s, loss="squared_error", **HP)
                  .fit(X[tr], y[tr]).predict(X) for s in SEEDS], axis=0)
    fracs, hgb = {}, {}
    for wn in ("VAL", "OOS", "HOLDOUT"):
        m = dsel & (D.split == wn).to_numpy()
        fracs[wn] = float((p0[m] > TAU0).mean())
        w = D[m]
        hgb[wn] = stat(slotN(w[p0[m] > TAU0], NSLOT))[1]
    log(f"HGB 기준선  " + " ".join(f"{k} {v:+.2f}bp(유지 {fracs[k]:.1%})" for k, v in hgb.items()))

    # ---- TabPFN 서브샘플 앙상블 ----
    itr = np.flatnonzero(tr)
    pred_rows = np.flatnonzero(dsel)
    Xp = X.iloc[pred_rows].to_numpy()
    log(f"TabPFN: TRAIN {len(itr):,} 중 {SUBSAMPLE:,}씩 {N_MEMBERS}개 추출 "
        f"(데이터 사용률 {SUBSAMPLE/len(itr):.0%}/멤버) · 예측대상 {len(pred_rows):,}행")
    parts = []
    for k in range(N_MEMBERS):
        rs = np.random.default_rng(SEEDS[k]).choice(itr, size=min(SUBSAMPLE, len(itr)), replace=False)
        reg = TabPFNRegressor(device="cuda", random_state=SEEDS[k])
        reg.fit(X.iloc[rs].to_numpy(), y[rs])
        parts.append(reg.predict(Xp))
        log(f"  멤버 {k+1}/{N_MEMBERS} 완료")
    pt_sub = np.mean(parts, axis=0)
    pt = np.full(len(D), np.nan); pt[pred_rows] = pt_sub

    log("\n=== 비교 (동일 유지비율) ===")
    print(f"{'구간':9s} | {'HGB(전체 TRAIN)':>20s} | {'TabPFN(18k×4 앙상블)':>22s}")
    tab = {}
    for wn in ("VAL", "OOS", "HOLDOUT"):
        m = dsel & (D.split == wn).to_numpy()
        w = D[m]; pv = pt[m]
        thr = np.quantile(pv, 1 - fracs[wn])
        v = slotN(w[pv > thr], NSLOT); nn, mm, _ = stat(v)
        tab[wn] = mm
        print(f"{wn:9s} | {hgb[wn]:+13.2f}bp      | {mm:+13.2f}bp n={nn:4d}")
    win = (tab["VAL"] > hgb["VAL"]) and (tab["OOS"] > hgb["OOS"])
    log(f"\n⭐사전등록 판정: TabPFN이 양 창에서 HGB를 "
        + ("**이김 → 후보 채택, 대조군 재검정 필요**" if win else "**못 이김 → HGB squared 동결 유지**"))
    corr = {wn: float(np.corrcoef(pt[dsel & (D.split == wn).to_numpy()],
                                  p0[dsel & (D.split == wn).to_numpy()])[0, 1])
            for wn in ("VAL", "OOS")}
    log(f"  참고: 두 모델 예측 상관 VAL {corr['VAL']:+.4f} / OOS {corr['OOS']:+.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"hgb": hgb, "tabpfn": tab, "keep_fracs": fracs, "tabpfn_wins": bool(win),
               "subsample": SUBSAMPLE, "members": N_MEMBERS,
               "data_used_per_member": SUBSAMPLE / len(itr), "pred_corr": corr},
              open(OUT / "b10_report.json", "w"), indent=2)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
