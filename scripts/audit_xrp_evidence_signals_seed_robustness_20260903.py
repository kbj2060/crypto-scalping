#!/usr/bin/env python3
"""XRP 증거신호 **시드 견고성** -- 단일 시드로 평가된 3종을 다중 시드로 메운다.

## 왜

사용자 질문 "증거신호와 레짐분류기는 시드검증 통과했어?"에 답하려고 실태를 확인한 결과:

| 신호 | 시드 수 | 리포트 기록 |
|---|---|---|
| `demarker_extreme` | 4 (`[20260829, 141592, 271828, 577215]`) | ✅ per_seed AUC 있음 |
| `kalman_deviation_meanrev` | 4 (동일) | ✅ per_seed AUC 있음 |
| **`short_term_return_z`** | **1** (`SEED = 20260903`) | ⚠️단일 |
| **`taker_delta_z_climax`** | **1** | ⚠️단일 |
| **`orthogonal_combo`** | **1** | ⚠️단일 |

`research_xrp_evidence_signals_metalabel_tabpfn_20260903.py`가 `SEED` 하나만 쓴다.
⇒ 배포된 HOLDOUT AUC(str_z 0.6132 / taker 0.6091 / orthogonal 0.5599)는 **분산 정보가 없는
단일 시드 점추정**이다.

CLAUDE.md **Seed-Diversity Ensemble Promotion Gate**: N>=5개의 **진짜 다양한** 시드(고정 간격
증가 금지)에서 OOS 부호 일치 + **시드 리스트를 리포트에 기록**.

## 설계

  · 발동/라벨 빌드는 시드와 무관(결정론적) -> 신호당 1회만 만들고 **TabPFN만 시드별 재적합**
  · 시드 **8종, 랜덤 추출**(고정 간격 증가 금지 -- Sigma3-1h 전례)
  · **VAL + OOS만** 본다. ⚠️HOLDOUT은 이미 1회 소진됐고 시드별 재평가는 재노출이다.

## 판정 (실행 전 고정)

  각 신호가 **모든 시드에서 OOS AUC > 0.5**(부호 일치)여야 게이트의 실질 요건을 만족한다.
  시드 표준편차가 신호 간 AUC 차이와 같은 크기면, 그 차이는 시드 노이즈와 구분되지 않는다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_S = importlib.util.spec_from_file_location(
    "xrpmeta", ROOT / "scripts/research_xrp_evidence_signals_metalabel_tabpfn_20260903.py")
_m = importlib.util.module_from_spec(_S)
_S.loader.exec_module(_m)

OUT = ROOT / "data/research/xrp_evidence_signals_seed_robustness_20260903.json"

# ⭐랜덤 추출 8종 (고정 간격 증가 금지). 첫 값은 원본 단일 시드라 재현 대조가 된다.
SEEDS = [20260903, 811453, 30011, 947, 260317, 5387291, 68041, 1299709]
# ⚠️SPEC 키는 `taker_delta_z_climax`(z 포함)다 -- 자산/스크립트마다 축약명이 다르다
TARGETS = ["short_term_return_z", "taker_delta_z_climax", "orthogonal_combo"]
RECORDED = {"short_term_return_z": {"VAL": 0.6466, "OOS": 0.5753},
            "taker_delta_z_climax": {"VAL": 0.6142, "OOS": 0.5556},
            "orthogonal_combo": {"VAL": 0.5979, "OOS": 0.5847}}


def log(m): print(f"[sig-seed] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier

    df = pd.read_csv(_m.CAND_CSV) if hasattr(_m, "CAND_CSV") else None
    if df is None:
        df = pd.read_csv(ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903"
                         / "xrp_5m_evidence_signal_candidates_tier0.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    log(f"후보 CSV {len(df):,}행")
    log(f"시드 {len(SEEDS)}종 (랜덤 추출): {SEEDS}")
    log("⚠️VAL+OOS만 평가 -- HOLDOUT은 1회 소진됨, 시드별 재평가는 재노출이다")

    rep = {"asset": "XRPUSDT", "seeds": SEEDS,
           "seed_selection": "랜덤 추출(고정 간격 증가 아님)",
           "n_seeds": len(SEEDS), "holdout_touched": False,
           "recorded_single_seed": RECORDED, "signals": {}}

    for name in TARGETS:
        spec = _m.SPEC[name]
        mod = _m._mod(f"btc_{name}", _m.BTC_MODULE[name])
        feats = list(mod.FEATURE_COLUMNS)
        fn = getattr(mod, _m.PREP_FN[name], None)
        frame = fn(df.copy()) if fn is not None else df.copy()
        fires = _m.build(frame, name, spec, feats)
        sp = np.where(fires["timestamp"] < _m.TRAIN_END, "TRAIN",
             np.where(fires["timestamp"] < _m.VAL_END, "VAL",
             np.where(fires["timestamp"] < _m.OOS_END, "OOS", "HOLDOUT")))
        fires["split"] = sp
        tr = fires[fires.split == "TRAIN"]
        log("")
        log(f"=== {name}  ({spec['hit']} H={spec['h']} K={spec['k']}) ===")
        log(f"  TRAIN {len(tr):,} (hit {tr['hit'].mean():.4f}) | "
            f"VAL {int((fires.split=='VAL').sum())} / OOS {int((fires.split=='OOS').sum())}")
        per = {"VAL": [], "OOS": []}
        for sd in SEEDS:
            clf = TabPFNClassifier(device="cuda", random_state=sd, ignore_pretraining_limits=True)
            clf.fit(tr[feats], tr["hit"].to_numpy().astype(int))
            row = {"seed": sd}
            for s_ in ("VAL", "OOS"):
                g = fires[fires.split == s_]
                if len(g) < 30 or g["hit"].nunique() < 2:
                    row[s_] = None; continue
                p = np.concatenate([clf.predict_proba(g[feats].iloc[k:k+20000])[:, 1]
                                    for k in range(0, len(g), 20000)])
                a = float(roc_auc_score(g["hit"].astype(int), p))
                row[s_] = round(a, 4)
                per[s_].append(a)
            log(f"  seed={sd:<9} VAL {row['VAL']}  OOS {row['OOS']}")
        res = {"hit_type": spec["hit"], "horizon": spec["h"], "k": spec["k"],
               "n_train": int(len(tr)), "per_seed": []}
        for i, sd in enumerate(SEEDS):
            res["per_seed"].append({"seed": sd,
                                    "VAL": round(per["VAL"][i], 4) if i < len(per["VAL"]) else None,
                                    "OOS": round(per["OOS"][i], 4) if i < len(per["OOS"]) else None})
        for s_ in ("VAL", "OOS"):
            v = np.array(per[s_])
            res[s_] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                       "min": float(v.min()), "max": float(v.max()),
                       "all_above_half": bool((v > 0.5).all())}
            log(f"  {s_} 평균 {v.mean():.4f} ± {v.std(ddof=1):.4f}  "
                f"[{v.min():.4f}, {v.max():.4f}]  전부>0.5: {'✅' if (v > 0.5).all() else '❌'}")
        rec = RECORDED.get(name, {})
        for s_ in ("VAL", "OOS"):
            if rec.get(s_) is not None:
                d = abs(res[s_]["mean"] - rec[s_])
                log(f"  기록 단일시드 {s_} {rec[s_]:.4f}  vs  8시드 평균 {res[s_]['mean']:.4f}  (차 {d:+.4f})")
        rep["signals"][name] = res

    log("")
    log("=" * 72)
    log("판정 (사전 고정: 모든 시드에서 OOS AUC > 0.5)")
    log("=" * 72)
    allok = True
    for n, v in rep["signals"].items():
        ok = v["OOS"]["all_above_half"]
        allok &= ok
        log(f"  {n:<26} OOS {v['OOS']['mean']:.4f} ± {v['OOS']['std']:.4f}  "
            f"[{v['OOS']['min']:.4f}, {v['OOS']['max']:.4f}]  {'✅' if ok else '❌'}")
    log("")
    log(f"⇒ {'✅**부호 일관성 통과** (N=8, 랜덤 추출)' if allok else '⚠️**일부 시드에서 0.5 이하**'}")
    rep["all_oos_above_half"] = bool(allok)
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
