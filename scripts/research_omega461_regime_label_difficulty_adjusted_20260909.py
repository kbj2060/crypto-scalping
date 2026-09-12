"""레짐 분류기 비교 — **라벨 난이도 보정판**.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
선행: `docs/experiments/omega461_regime_headtohead_quality_20260909.md`

문제
----
wide24 HMM 과 cut2509 GBM 은 **정답(라벨)이 다르다**. 교차 표(각 모델 × 각 라벨)를 그대로 읽으면
"어느 라벨이 애초에 더 맞히기 쉬운가"와 "어느 모델이 더 좋은가"가 섞인다. 실제로 두 라벨은
클래스 비중부터 다르다(balancedish chop ~0.70 vs S12_K3 chop ~0.56).

보정 두 가지
-----------
**(1) 라벨 간 비교 가능한 스킬 점수.** 원점수 대신 우연 대비 초과분을 쓴다.
   · Cohen's κ            — 우연 일치 보정, 클래스 불균형에 강함
   · AMI                  — 조정 상호정보량, 라벨 순열/개수에 불변
   · bal_acc skill        — (BA − 1/3) / (1 − 1/3), 3-class 무작위 기준 정규화
   · log-loss skill       — 1 − LL / LL(클래스 사전확률 예측기), 그 라벨의 기저 엔트로피로 정규화

**(2) 라벨별 "천장" 측정 — 이게 핵심.** 같은 모델급(HistGradientBoosting, 동일 HP),
   같은 136피쳐, 같은 컷오프(≤2025-09-30), 같은 SEED 로 **balancedish 라벨에도** 적합한다.
   cut2509 GBM 의 S12_K3 점수는 이미 그 프로토콜의 S12_K3 천장이다. 두 천장을 나란히 놓으면
   "라벨 난이도 차이"가 분리되고, 각 분류기를 **자기 라벨 천장 대비 몇 %**로 읽을 수 있다.

   해석 규칙:
     · 두 천장이 비슷하면 → 라벨 난이도는 비슷하고, 원점수 차이는 모델 차이다.
     · balancedish 천장이 낮으면 → 그 라벨이 본질적으로 어렵고, HMM 의 낮은 원점수는
       불리한 라벨을 맡은 결과이지 모델이 나쁜 게 아니다.

준수: 신규 학습은 **balancedish 천장 측정용 HGB 1개뿐**이며, 컷오프(≤2025-09-30)를 지켜
평가창(VAL/OOS)은 학습에 들어가지 않는다. 라이브 파일 미변경.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (adjusted_mutual_info_score, balanced_accuracy_score,
                             cohen_kappa_score, log_loss)

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiment_regime3_current_hmm_wide24_20260529 import _current_labels3_thresholded  # noqa: E402
from research_eth_regime_s12k3_label_train_20260902 import GBM3_HP, SEED  # noqa: E402
from research_eth_regime_scalping_label_geometry_20260902 import _debounce, scaled_label  # noqa: E402
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402

WIDE24_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
CUT_DIR = ROOT / "data/ensemble/supervised/omega461_regimegbm_cut2509_20260909"
CUT_MODEL = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/regime_cut2509_model.joblib"
HMM_MODEL = WIDE24_DIR / "regime3_current_sensitive_hmm_wide24_2024.joblib"
CUT_PREFIX, W24_PREFIX = "regime3_s12k3_cut2509_", "regime3_current_sensitive_wide24_"
CLASSES = ("bull", "bear", "chop")
BASE_CSVS = {"2024": "training_features_2024.csv", "2025": "training_features_2025.csv",
             "2026_rebuilt": "training_features_2026_rebuilt.csv"}
CUTOFF_END = pd.Timestamp("2025-09-30 23:55:00")
SPLITS = {"validation": ("2025-10-01", "2025-12-31 23:55:00"),
          "oos": ("2026-01-01", "2026-02-28 23:55:00")}
OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909"


def load_all() -> pd.DataFrame:
    parts = []
    for tag, fn in BASE_CSVS.items():
        b = pd.read_csv(ROOT / "data/splits/year_oos" / fn, low_memory=False, parse_dates=["timestamp"])
        w = pd.read_csv(WIDE24_DIR / f"training_features_{tag}_regime3_current_sensitive_hmm_wide24.csv",
                        low_memory=False, parse_dates=["timestamp"])
        c = pd.read_csv(CUT_DIR / f"training_features_{tag}_{CUT_PREFIX}sidecar.csv",
                        low_memory=False, parse_dates=["timestamp"])
        parts.append(b.merge(w, on="timestamp", how="inner").merge(c, on="timestamp", how="inner"))
    df = (pd.concat(parts, ignore_index=True).sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last").reset_index(drop=True))
    return _with_raw_state12(df)


def probs(df: pd.DataFrame, prefix: str) -> np.ndarray:
    p = df[[f"{prefix}{c}_prob" for c in CLASSES]].to_numpy(np.float64)
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def skill(y: np.ndarray, p: np.ndarray) -> dict:
    """라벨 간 비교 가능한 스킬 점수들."""
    pred = p.argmax(1)
    ba = float(balanced_accuracy_score(y, pred))
    ll = float(log_loss(y, p, labels=[0, 1, 2]))
    prior = np.bincount(y, minlength=3) / len(y)          # 그 라벨의 클래스 사전확률
    ll_base = float(log_loss(y, np.tile(prior, (len(y), 1)), labels=[0, 1, 2]))
    return {"balanced_accuracy": round(ba, 4),
            "bal_acc_skill": round((ba - 1 / 3) / (1 - 1 / 3), 4),
            "cohen_kappa": round(float(cohen_kappa_score(y, pred, labels=[0, 1, 2])), 4),
            "adjusted_mutual_info": round(float(adjusted_mutual_info_score(y, pred)), 4),
            "log_loss": round(ll, 4), "log_loss_prior_baseline": round(ll_base, 4),
            "log_loss_skill": round(1.0 - ll / ll_base, 4)}


def main() -> int:
    df = load_all()
    ts = df["timestamp"]
    cut_pay = joblib.load(CUT_MODEL)
    cols, medians = cut_pay["feature_cols"], pd.Series(cut_pay["feature_medians"])
    t1 = float(cut_pay["label_spec"]["T1_er12"]); t2 = float(cut_pay["label_spec"]["T2_er24"])
    hmm_cfg = joblib.load(HMM_MODEL)["label_config"]

    y_bal = _current_labels3_thresholded(df, hmm_cfg)
    y_s12 = _debounce(scaled_label(df["close"], 12, t1, t2), 3)
    train_mask = (ts <= CUTOFF_END).to_numpy()

    X = (df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
           .replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0))

    # --- (2) balancedish 라벨의 천장: 같은 모델급/피쳐/컷오프/시드 ---
    print(f"[천장측정] HGB(동일 HP {GBM3_HP}, SEED {SEED}) 를 balancedish 라벨에 적합 "
          f"(컷오프 ≤{CUTOFF_END.date()}, {int(train_mask.sum()):,}봉)", flush=True)
    ceil_bal = HistGradientBoostingClassifier(random_state=SEED, **GBM3_HP)
    ceil_bal.fit(X[train_mask], y_bal[train_mask])
    p_ceil_bal = ceil_bal.predict_proba(X)

    models = {"wide24_HMM(전신)": probs(df, W24_PREFIX),
              "cut2509_GBM(후보)": probs(df, CUT_PREFIX),
              "HGB_ceiling(동일프로토콜)": p_ceil_bal}
    labels = {"balancedish": y_bal, "S12_K3": y_s12}

    report = {"protocol": {"hp": GBM3_HP, "seed": SEED, "cutoff_end": str(CUTOFF_END),
                           "n_features": len(cols), "train_bars": int(train_mask.sum())},
              "windows": {}}

    for split, (s, e) in SPLITS.items():
        m = ((ts >= s) & (ts <= e)).to_numpy()
        res = {"bars": int(m.sum()), "label_class_shares": {}, "scores": {}}
        print(f"\n{'='*94}\n[{split}] {s} ~ {e}  {int(m.sum()):,}봉", flush=True)
        for ln, y in labels.items():
            sh = {n: round(float((y[m] == i).mean()), 3) for i, n in enumerate(CLASSES)}
            res["label_class_shares"][ln] = sh
            print(f"  라벨 {ln:12s} 클래스비중 {sh}", flush=True)

        print(f"\n  {'모델':26s}{'라벨':12s}{'bal_acc':>9s}{'BAskill':>9s}{'kappa':>8s}"
              f"{'AMI':>8s}{'LLskill':>9s}", flush=True)
        for mn, p in models.items():
            for ln, y in labels.items():
                # 천장 모델은 자기 라벨(balancedish)에서만 의미가 있다
                if mn.startswith("HGB_ceiling") and ln != "balancedish":
                    continue
                sc = skill(y[m], p[m])
                res["scores"].setdefault(mn, {})[ln] = sc
                print(f"  {mn:26s}{ln:12s}{sc['balanced_accuracy']:9.4f}{sc['bal_acc_skill']:9.4f}"
                      f"{sc['cohen_kappa']:8.4f}{sc['adjusted_mutual_info']:8.4f}"
                      f"{sc['log_loss_skill']:9.4f}", flush=True)

        # --- 자기 라벨 천장 대비 ---
        ceil = {"balancedish": res["scores"]["HGB_ceiling(동일프로토콜)"]["balancedish"],
                "S12_K3": res["scores"]["cut2509_GBM(후보)"]["S12_K3"]}
        res["label_ceilings"] = ceil
        print(f"\n  [라벨 천장, 동일 프로토콜 HGB] balancedish bal_acc={ceil['balancedish']['balanced_accuracy']:.4f} "
              f"(κ={ceil['balancedish']['cohen_kappa']:.4f})  |  "
              f"S12_K3 bal_acc={ceil['S12_K3']['balanced_accuracy']:.4f} "
              f"(κ={ceil['S12_K3']['cohen_kappa']:.4f})", flush=True)
        rel = {}
        for mn, ln in (("wide24_HMM(전신)", "balancedish"), ("cut2509_GBM(후보)", "S12_K3")):
            sc = res["scores"][mn][ln]
            rel[mn] = {"own_label": ln,
                       "bal_acc_pct_of_ceiling": round(sc["balanced_accuracy"] / ceil[ln]["balanced_accuracy"] * 100, 1),
                       "kappa_pct_of_ceiling": round(sc["cohen_kappa"] / ceil[ln]["cohen_kappa"] * 100, 1)}
            print(f"  {mn:26s} 자기라벨({ln}) 천장 대비  "
                  f"bal_acc {rel[mn]['bal_acc_pct_of_ceiling']:5.1f}%   κ {rel[mn]['kappa_pct_of_ceiling']:5.1f}%", flush=True)
        res["vs_own_label_ceiling"] = rel
        report["windows"][split] = res

    (OUT / "regime_label_difficulty_adjusted.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n산출물: {OUT}/regime_label_difficulty_adjusted.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
