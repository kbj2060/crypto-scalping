#!/usr/bin/env python3
"""OOF 재료 텐서 **무결성 감사** (2026-09-03).

사용자 질문: "OOF 텐서에 결함은 모두 고쳤나?"
알려진 수정: 레짐 2열 누락(조용히 건너뛰기) → 없으면 실패 + 열 수 검증 추가, 43열로 복구.
여기서는 **나머지가 있는지** 확인한다. 가정하지 않는다.

  ① 구본(51열)과의 열 차이 -- 무엇을 잃었나
  ② 행 수 차이 (구본 279,634 vs 신본 280,471)
  ③ fold별 커버리지 -- OOF는 fold마다 학습량이 다르다. fold1은 워밍업 4개월만 본다
  ④ 결측/퇴화 열 -- 상수이거나 전부 0인 열
  ⑤ 워밍업 구간이 정말 비어 있나
  ⑥ split별 proba 분포 -- fold 경계에서 튀는가
  ⑦ 인과성 -- 유지창 밖 활성값
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

NEW = ROOT / "data/materials/eth_evidence_signal_tensor_oof_20260903/eth_evidence_material_5m.parquet"
OLD = ROOT / "data/materials/eth_evidence_signal_tensor_20260902/eth_evidence_material_5m.parquet"
OOFD = ROOT / "tmp/eth_entry_oof_metalabel_20260903"
SRC = ROOT / "tmp/eth_causal_population_metalabel_20260902"
WARM = pd.Timestamp("2024-05-01")


def log(m): print(f"[tensor-audit] {m}", flush=True)


def main() -> int:
    N = pd.read_parquet(NEW)
    log(f"신본 {len(N):,}행 × {N.shape[1]}열 · {N.timestamp.min()} ~ {N.timestamp.max()}")

    # ① 열 차이
    if OLD.exists():
        O = pd.read_parquet(OLD)
        log(f"구본 {len(O):,}행 × {O.shape[1]}열 · {O.timestamp.min()} ~ {O.timestamp.max()}")
        lost = [c for c in O.columns if c not in N.columns]
        gained = [c for c in N.columns if c not in O.columns]
        print(f"\n=== ① 열 차이 ===")
        print(f"  잃은 열 {len(lost)}: {lost}")
        print(f"  얻은 열 {len(gained)}: {gained}")
        # ② 행 차이
        print(f"\n=== ② 행 차이 ===")
        so, sn = set(O.timestamp), set(N.timestamp)
        print(f"  신본에만 {len(sn-so):,}행 · 구본에만 {len(so-sn):,}행 · 공통 {len(sn&so):,}행")
        if sn - so:
            e = sorted(sn - so)
            print(f"  신본 전용 범위: {e[0]} ~ {e[-1]}")
    else:
        log("⚠️구본 없음 -- 열/행 비교 생략")

    cfg = json.loads((SRC / "config.json").read_text())["cfg"]

    # ③ fold별 커버리지
    print(f"\n=== ③ OOF fold별 커버리지 ===")
    print(f"{'신호':>26s}{'전체':>8s}{'fold1':>8s}{'fold2':>8s}{'fold3':>8s}{'fold4':>8s}{'최종':>8s}")
    for name in cfg:
        f = OOFD / f"{name}_oof.csv"
        if not f.exists(): continue
        d = pd.read_csv(f)
        src = d["oof_source"].fillna("").astype(str)
        cnt = {k: int(src.str.startswith(f"fold{k}").sum()) for k in (1, 2, 3, 4)}
        fin = int((~src.str.startswith("fold") & (src != "")).sum())
        print(f"{name:>26s}{len(d):8d}" + "".join(f"{cnt[k]:8d}" for k in (1, 2, 3, 4)) + f"{fin:8d}")

    # ④ 결측/퇴화 열
    print(f"\n=== ④ 결측·퇴화 열 ===")
    nan_cols = [c for c in N.columns if N[c].isna().any()]
    num = [c for c in N.columns if c != "timestamp"]
    const = [c for c in num if N[c].nunique(dropna=False) <= 1]
    allzero = [c for c in num if (pd.to_numeric(N[c], errors="coerce").fillna(0) == 0).all()]
    print(f"  결측 있는 열 {len(nan_cols)}: {nan_cols or '없음'}")
    print(f"  상수 열 {len(const)}: {const or '없음'}")
    print(f"  전부 0인 열 {len(allzero)}: {allzero or '없음'}")

    # ⑤ 워밍업
    print(f"\n=== ⑤ 워밍업(<{WARM.date()}) 구간 ===")
    w = N.timestamp < WARM
    sig_cols = [c for c in N.columns if any(c.startswith(s + "_") for s in cfg)]
    act = {c: float((pd.to_numeric(N.loc[w, c], errors="coerce").fillna(0) != 0).mean())
           for c in sig_cols if not c.endswith("_age")}
    bad = {k: v for k, v in act.items() if v > 0}
    age_cols = [c for c in sig_cols if c.endswith("_age")]
    age_ok = all(float((N.loc[w, c] == 1.0).mean()) == 1.0 for c in age_cols)
    print(f"  워밍업 {int(w.sum()):,}행 · 0이 아닌 신호열 {len(bad)}개 {'✅' if not bad else bad}")
    print(f"  age가 전부 1.0인가: {'✅' if age_ok else '⚠️아님'}")

    # ⑥ split별 proba 분포
    print(f"\n=== ⑥ split별 proba 분포 (fold 경계에서 튀는가) ===")
    ts = pd.DatetimeIndex(N.timestamp)
    sp = np.where(ts < pd.Timestamp("2025-09-01"), "TRAIN",
          np.where(ts < pd.Timestamp("2026-01-01"), "VAL",
          np.where(ts < pd.Timestamp("2026-04-01"), "OOS", "HOLDOUT")))
    print(f"{'신호':>26s}{'TRAIN':>10s}{'VAL':>10s}{'OOS':>10s}{'HOLDOUT':>10s}  (활성 평균 proba)")
    for name in cfg:
        c = f"{name}_proba"
        if c not in N.columns: continue
        v = pd.to_numeric(N[c], errors="coerce").to_numpy()
        row = []
        for s in ("TRAIN", "VAL", "OOS", "HOLDOUT"):
            m = (sp == s) & (v > 0)
            row.append(float(v[m].mean()) if m.any() else np.nan)
        print(f"{name:>26s}" + "".join(f"{x:10.4f}" for x in row))

    # ⑦ 인과성
    print(f"\n=== ⑦ 인과성 (유지창 밖 활성값) ===")
    bad_n = 0
    for name, cc in cfg.items():
        H = int(cc["horizon"])
        fc = N[f"{name}_fire"].to_numpy(); pc = N[f"{name}_proba"].to_numpy()
        act_i = np.flatnonzero(pc > 0)
        for i in act_i[:: max(1, len(act_i) // 3000)]:
            if not np.any(fc[max(0, i - H + 1):i + 1] != 0):
                bad_n += 1; break
    print(f"  위반 {bad_n}건 (0이어야 정상) {'✅' if bad_n == 0 else '❌'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
