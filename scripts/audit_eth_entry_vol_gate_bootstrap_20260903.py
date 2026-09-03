#!/usr/bin/env python3
"""고변동성 게이트 -- **모델을 죽인 것과 같은 잣대로** 검정 (2026-09-03).

`research_eth_entry_simple_rules_l3_20260903.py`에서 `atr_pct >= p90`과 `parkinson_vol >= p90`이
3창 전부 크게 양수로 나왔다(VAL +13.86/+27.00 · OOS +24.16/+32.59 · HOLDOUT +64.13/+54.18,
무필터 +4.01/+6.02/−0.99). 파라미터 1개짜리 규칙이다.

⚠️그러나 161피쳐 모델도 **행 단위** 무작위 필터 대조군은 p=0.003으로 통과했다가 **일 단위
군집 부트스트랩**에서 CI 하한이 음수로 떨어져 무너졌다. 같은 잣대를 대지 않으면 같은 실수다.

여기서 재는 것:
  ① **일 단위 군집 부트스트랩** -- (게이트 − 무필터) 95% CI. 하한 > 0 이어야 통과
  ② **독립 일수** -- 42~45일이 모델을 죽인 벽이었다. 게이트는 유지율 10%라 더 적을 수 있다
  ③ **컷 민감도** {80, 85, 90, 95} -- p90이 절벽 위에 서 있는지
  ④ **두 승자의 상관** -- atr_pct와 parkinson_vol이 사실상 같은 것인지
  ⑤ **게이트 + 모델** 결합 -- 두 축이 독립적으로 기여하는지
"""
from __future__ import annotations

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

from research_eth_entry_b6_expand_20260903 import slotN  # noqa: E402

B6 = ROOT / "tmp/eth_entry_b6_expand_20260903"
L3D = ROOT / "tmp/eth_entry_1m_resolved_20260903/labels_1m_all.csv"
V3 = ROOT / "tmp/eth_entry_limit_fade_v3_l3arm1_20260903"
DEPTH, WAIT, NSLOT = 3.0, 6, 4
W3 = ("VAL", "OOS", "HOLDOUT")
RNG = np.random.default_rng(20260903)


def log(m): print(f"[volgate] {m}", flush=True)


def main() -> int:
    LAB = pd.read_csv(L3D, parse_dates=["timestamp"])
    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    A = D.merge(LAB[["timestamp", "signal", "arm", "depth", "btf", "y_L3"]],
                on=["timestamp", "signal", "arm", "depth", "btf"], how="left")
    A = A[np.isfinite(A.y_L3)].reset_index(drop=True)
    S = A[(A.depth == DEPTH) & (A.btf <= WAIT) & (A.arm == 1)].reset_index(drop=True)
    tr = (S.split == "TRAIN").to_numpy()
    M = {w: (S.split == w).to_numpy() for w in W3}
    S["day"] = pd.to_datetime(S.timestamp).dt.date

    def perf(mask):
        d = S[mask]
        if not len(d): return np.nan, 0
        t = slotN(d.assign(y=d.y_L3), NSLOT)
        return (float(np.mean(t) * 1e4) if len(t) else 0.0), int(len(t))

    # ④ 두 승자의 상관
    a1 = pd.to_numeric(S["atr_pct"], errors="coerce").to_numpy(float)
    p1 = pd.to_numeric(S["parkinson_vol"], errors="coerce").to_numpy(float)
    ok = np.isfinite(a1) & np.isfinite(p1)
    log(f"④ atr_pct vs parkinson_vol 상관 **{np.corrcoef(a1[ok], p1[ok])[0,1]:.4f}**")
    ga = a1 >= np.nanpercentile(a1[tr], 90)
    gp = p1 >= np.nanpercentile(p1[tr], 90)
    log(f"   두 게이트 겹침: {int((ga & gp).sum()):,} / 합집합 {int((ga | gp).sum()):,} "
        f"= **{(ga & gp).sum() / max((ga | gp).sum(), 1) * 100:.1f}%** -- 사실상 같은 축인가")

    # ③ 컷 민감도
    print(f"\n=== ③ 컷 민감도 ===")
    print(f"{'피쳐':>16s}{'컷':>5s}{'유지':>7s}" + "".join(f"{w:>15s}" for w in W3))
    for f, v in (("atr_pct", a1), ("parkinson_vol", p1)):
        for c in (80, 85, 90, 95):
            g = v >= np.nanpercentile(v[tr], c)
            r = {w: perf(M[w] & g) for w in W3}
            print(f"{f:>16s}{c:5d}{float(g[tr].mean()):7.2f}"
                  + "".join(f"{r[w][0]:+9.2f}(n{r[w][1]:3d})" for w in W3))

    # ①② 일 단위 군집 부트스트랩
    print(f"\n=== ①② 일 단위 군집 부트스트랩 (게이트 − 무필터, B=4000) ===")
    print(f"{'게이트':>16s}{'창':>9s}{'게이트일수':>10s}{'차이':>9s}{'95% CI':>22s}{'판정':>6s}")
    for f, g in (("atr_pct p90", ga), ("parkinson_vol p90", gp), ("합집합", ga | gp)):
        for w in W3:
            dg = S[M[w] & g]; dn = S[M[w]]
            if not len(dg): continue
            gk = pd.DataFrame({"d": dg.day.to_numpy(), "bp": dg.y_L3.to_numpy() * 1e4})
            nk = pd.DataFrame({"d": dn.day.to_numpy(), "bp": dn.y_L3.to_numpy() * 1e4})
            uk = gk.d.unique()
            diffs = []
            for _ in range(4000):
                s_ = RNG.choice(uk, size=len(uk), replace=True)
                x1 = gk[gk.d.isin(s_)].bp.mean(); x2 = nk[nk.d.isin(s_)].bp.mean()
                if np.isfinite(x1) and np.isfinite(x2): diffs.append(x1 - x2)
            dd = np.array(diffs)
            lo_, hi_ = np.percentile(dd, [2.5, 97.5])
            print(f"{f:>16s}{w:>9s}{len(uk):10d}{dd.mean():+9.2f}"
                  f"   [{lo_:+8.2f},{hi_:+8.2f}]{'  ✅' if lo_ > 0 else '  ❌'}")

    # ⑤ 게이트 + 모델
    if (V3 / "model.joblib").exists():
        import os
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("TABPFN_TOKEN="):
                    os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip('"')
        from tabpfn import TabPFNClassifier
        Q = joblib.load(V3 / "model.joblib")
        FE = Q["feature_cols"]
        X = S[FE].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X = X.fillna(pd.Series(Q["feature_medians"])).to_numpy(np.float32)
        loc = {int(v): i for i, v in enumerate(Q["context_index"])}
        ps = []
        for sd in Q["seeds"]:
            rs = np.random.default_rng(sd).choice(Q["context_index"], size=18000, replace=False)
            sel = np.array([loc[int(v)] for v in rs])
            m = TabPFNClassifier(device="cuda", random_state=sd)
            m.fit(Q["context_X"][sel], Q["context_y"][sel])
            ps.append(m.predict_proba(X)[:, 1])
            log(f"  ⑤ TabPFN 멤버 {len(ps)}")
        P = np.mean(ps, axis=0)
        mk = P > float(Q["policy"]["p_threshold"])
        print(f"\n=== ⑤ 게이트 + 모델 (두 축이 독립 기여하는가) ===")
        print(f"{'구성':>22s}" + "".join(f"{w:>15s}" for w in W3))
        for tag, mm in (("무필터", np.ones(len(S), bool)), ("모델만", mk),
                        ("게이트만(atr p90)", ga), ("게이트+모델", ga & mk)):
            r = {w: perf(M[w] & mm) for w in W3}
            print(f"{tag:>22s}" + "".join(f"{r[w][0]:+9.2f}(n{r[w][1]:3d})" for w in W3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
