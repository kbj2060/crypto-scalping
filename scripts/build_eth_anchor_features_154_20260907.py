#!/usr/bin/env python3
"""앵커에 **DC 엔지니어링 154피쳐** 조인 (2026-09-07).

사용자: *"150여개 피쳐가 있자나"*

`tmp/ilias_eth_154feature_dataset_20260821/ilias_eth_154feature_2024_2026H1_combined.csv`
(236,401행 x 155컬럼 · 2024-01-01 ~ 2026-03-31). 158 캐노니컬 -> 리던던시/VIF 정리 -> RIT 조합
-> 금융ML 문헌표준 추가로 만든 세트(`docs/model_contracts/eth_dc_engineered_feature_set_lineage_20260820.json`).

## 사용 전 관문 -- 룩어헤드 감사를 **이번에 처음 실행**했다
`scripts/audit_eth_154feature_lookahead_20260901.py` (2026-09-01 작성, 산출물이 없어 미실행 상태였음)
=> `data/research/eth_154feature_audit_20260901/report.json`
   leak_likely 0 · suspect 0 · suspect_future_peak 1 · exclude_model_output 3 · unusable 0 · **pass 150**
채택은 `pass` 150개만. 미래피크 1 · 모델출력 3(regime3_* 계열, 순환성)은 제외한다.
⚠️감사 한계(그 문서 원문): B/C 는 **총체적 누출 탐지기**이지 인과성 증명이 아니다.

## 이 피쳐가 199 자체제작본보다 나은 이유
① 커버리지가 **2024-01-01** 부터라 TRAIN 이 늘어난다(199본은 bookdepth 때문에 2024-04-20~)
② 리던던시·VIF 정리를 이미 거쳤다 ③ 감사 이력이 남는다
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
F154 = ROOT / "tmp/ilias_eth_154feature_dataset_20260821/ilias_eth_154feature_2024_2026H1_combined.csv"
AUDIT = ROOT / "data/research/eth_154feature_audit_20260901/report.json"
LAB = ROOT / "tmp/eth_anchor_training_set_20260907/train_set_P1_H48.parquet"
OUT = ROOT / "tmp/eth_anchor_features154_20260907"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = json.loads(AUDIT.read_text())
    feats = rep["features"]
    keep = [f["feature"] for f in feats if f.get("verdict") == "pass"]
    drop = {f["feature"]: f.get("verdict") for f in feats if f.get("verdict") != "pass"}
    print(f"[1/4] 감사 판정: pass {len(keep)} · 제외 {len(drop)} -> {drop}", flush=True)

    D = pd.read_parquet(LAB)
    D = D[D.anchor == "any3/Wc3"].reset_index(drop=True) if "anchor" in D else D
    print(f"[2/4] 앵커 {len(D):,}행 · {D.timestamp.min()} ~ {D.timestamp.max()}", flush=True)

    print("[3/4] 154피쳐 CSV 로드(필요 컬럼만) ...", flush=True)
    X = pd.read_csv(F154, usecols=["timestamp"] + keep, parse_dates=["timestamp"], low_memory=False)
    X = X.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    print(f"      {X.shape} · {X.timestamp.min()} ~ {X.timestamp.max()}", flush=True)

    M = D.merge(X, on="timestamp", how="inner")
    print(f"[4/4] 조인 {len(M):,}행 (앵커 대비 {len(M)/len(D)*100:.1f}%) · split {M.groupby('split').size().to_dict()}", flush=True)
    M = M.replace([np.inf, -np.inf], np.nan)

    sp = M["split"].to_numpy()
    checks = {"방향(y_bin)": M["y_bin"].to_numpy(), "혼재(y3==1)": (M["y3"].to_numpy() == 1).astype(float)}
    worst = []
    for lab, y in checks.items():
        mx, arg = 0.0, None
        for w in ("VAL", "OOS"):
            m = (sp == w) & np.isfinite(y)
            for c in keep:
                v = M[c].to_numpy(float)[m]
                k = np.isfinite(v)
                if k.sum() < 50 or len(np.unique(y[m][k])) < 2:
                    continue
                a = roc_auc_score(y[m][k], v[k]); a = max(a, 1 - a)
                if a > mx:
                    mx, arg = a, (w, c)
        worst.append({"label": lab, "max_auc": mx, "where": arg})
        print(f"      L3 누수검사 {lab:<14} 최대 {mx:.4f} ({arg[0]}, {arg[1]}) → {'⚠️FAIL' if mx >= 0.95 else 'OK'}", flush=True)

    nan = M[keep].isna().mean()
    print(f"      결측률 최대 {nan.max():.3f}({nan.idxmax()}) · 평균 {nan.mean():.3f} · 전부NaN {int((nan==1).sum())}개", flush=True)
    M.to_parquet(OUT / "features154.parquet", index=False)
    (OUT / "meta.json").write_text(json.dumps({
        "n_rows": len(M), "n_features": len(keep), "feature_cols": keep,
        "audit": {k: rep["counts"][k] for k in rep["counts"]}, "excluded": drop,
        "leak_check": worst}, indent=2, ensure_ascii=False, default=str))
    print(f"\n저장: {OUT} · {M.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
