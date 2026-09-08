"""Phase 0 — 컷오프 재학습 레짐 GBM + 오메가용 6컬럼 사이드카 생성.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계약: docs/model_contracts/omega461_regimegbm_rebuild_contract.md
레지스트리: docs/model_contracts/omega461_regimegbm_rebuild_data_resources_20260909.md

왜 재학습하는가
--------------
배포된 대시보드 아티팩트 `tmp/eth_regime_s12k3_20260902/model.joblib` 의 train_range 는
**2024-01-01~2026-06-30** 으로, 이 라인의 validation(2025-10-01~12-31)과 OOS(2026-01-01~02-28)를
전부 포함한다. S12_K3 라벨 정의 자체는 인과적(efficiency_ratio/net/slope 전부 후방참조,
debounce 는 전방순차)이지만 **적합된 모델이 그 봉들의 정답을 이미 봤으므로**, 그 예측을 부모
TabM 의 피쳐로 넣으면 downstream 평가 전체가 오염된다. 사용자 승인(2026-09-09) 아래 해소안 (a)
컷오프 재학습으로 간다.

무엇을 고정하고 무엇만 바꾸는가
-----------------------------
모델 config(GBM3_HP), 136 feature_cols, feature_medians, 라벨 정의(S=12/K=3), SEED 전부
배포본과 동일하게 두고 **학습 컷오프만** 2026-06-30 → 2025-09-30 으로 당긴다. 라벨 임계값
T1/T2 도 같은 컷오프 마스크에서만 캘리브레이트한다(`s12k3_label` 이 train_mask 만 쓴다).

⚠️ 두 번째 누출 — TRAIN 창의 in-sample 낙관
------------------------------------------
컷오프만 당기면 부모의 TRAIN 창(2025-01~09)은 레짐 GBM 의 **in-sample** 이고 VAL/OOS 는
out-of-sample 이 된다. 부모가 학습 때는 지나치게 정확한 레짐 신호를 보고, 평가 때는 열화된
신호를 받는 train/inference 분포 불일치다. 그래서 이 스크립트는:
  · TRAIN 창 = **purged 시간블록 K-fold OOF** 예측
  · VAL/OOS+ = 컷오프 전체로 적합한 단일 모델 예측
을 쓰고, in-sample / OOF / VAL / OOS bal_acc 를 **전부 리포트에 남긴다**. 라벨이 후방참조라
fold 간 라벨 누출은 없고, 경계에는 라벨 lookback(2*S=24봉) 여유분으로 PURGE_BARS 만큼 퍼지를 준다.

6컬럼 유도식
-----------
`trading_bot_modules/odyssey_regime3_live.py::_append_current` (372-392행) 을 그대로 따른다 —
모델 비의존적이고 확률 3개만 필요하다:
    {p}{class}_prob = 정규화 확률
    {p}confidence   = proba.max(axis=1)
    {p}margin       = top1 - top2
    {p}entropy      = -sum(p*log p)/log(3)
클래스 순서는 wide24 payload 와 S12_K3 payload 둘 다 ['bull','bear','chop'] 로 일치함을 확인했다.

산출물은 **새 접두사**를 쓴다 — wide24 접두사를 재사용하면 라이브에서 도는 전신과 충돌한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from research_eth_regime_s12k3_label_train_20260902 import (  # noqa: E402
    GBM3_HP, GBM3_MODEL_PATH, SEED, s12k3_label,
)
from research_eth_regime_scalping_label_geometry_20260902 import TRAIN_CSVS  # noqa: E402
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402

# --- 이 라인의 split 계약 (contract.md 의 Dataset Split 과 반드시 일치) ---
SPAN_START = pd.Timestamp("2024-01-01T00:00:00")
CUTOFF_END = pd.Timestamp("2025-09-30T23:55:00")     # 레짐 GBM 학습 상한 == 이 라인의 train_end
VAL_START, VAL_END = pd.Timestamp("2025-10-01T00:00:00"), pd.Timestamp("2025-12-31T23:55:00")
OOS_START, OOS_END = pd.Timestamp("2026-01-01T00:00:00"), pd.Timestamp("2026-02-28T23:55:00")
SPAN_END = pd.Timestamp("2026-08-30T23:55:00")       # 사이드카는 프레임 끝까지 채운다

PREFIX = "regime3_s12k3_cut2509_"
CLASSES3 = ["bull", "bear", "chop"]
N_FOLDS = 5
PURGE_BARS = 48          # 라벨 lookback 2*S=24봉의 2배 여유
OUT_DIR = ROOT / "tmp/omega461_regimegbm_rebuild_20260909"
SIDECAR_DIR = ROOT / "data/ensemble/supervised/omega461_regimegbm_cut2509_20260909"


def load_span() -> pd.DataFrame:
    frames = [pd.read_csv(p, parse_dates=["timestamp"]) for p in TRAIN_CSVS]
    df = (pd.concat(frames, ignore_index=True).sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last").reset_index(drop=True))
    df = df[(df["timestamp"] >= SPAN_START) & (df["timestamp"] <= SPAN_END)].reset_index(drop=True)
    return _with_raw_state12(df)


def design_matrix(df: pd.DataFrame, cols: list[str], medians: dict) -> pd.DataFrame:
    x = df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
    return x.replace([np.inf, -np.inf], np.nan).fillna(pd.Series(medians)).fillna(0.0)


def fit_gbm(X, y, seed: int = SEED) -> HistGradientBoostingClassifier:
    m = HistGradientBoostingClassifier(random_state=seed, **GBM3_HP)
    m.fit(X, y)
    return m


def derive_six(proba: np.ndarray) -> dict[str, np.ndarray]:
    """odyssey_regime3_live.py::_append_current(372-392) 와 동일한 유도식."""
    p = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
    s = np.sort(p, axis=1)
    out = {f"{PREFIX}{n}_prob": p[:, i] for i, n in enumerate(CLASSES3)}
    out[f"{PREFIX}confidence"] = p.max(axis=1)
    out[f"{PREFIX}margin"] = s[:, -1] - s[:, -2]
    out[f"{PREFIX}entropy"] = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1) / np.log(3.0)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)

    src = joblib.load(GBM3_MODEL_PATH)
    cols, medians = src["feature_cols"], src["feature_medians"]
    print(f"[기준] GBM3 아티팩트에서 {len(cols)}피쳐/medians 승계, HP={GBM3_HP}, SEED={SEED}", flush=True)

    df = load_span()
    ts = df["timestamp"]
    print(f"[프레임] {len(df):,}봉  {ts.iloc[0]} ~ {ts.iloc[-1]}", flush=True)

    train_mask = (ts <= CUTOFF_END).to_numpy()
    val_mask = ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()
    oos_mask = ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()

    y, t1, t2 = s12k3_label(df, train_mask)
    shares = {n: round(float((y[train_mask] == i).mean()), 4) for i, n in enumerate(CLASSES3)}
    print(f"[라벨] S12_K3, 임계값 TRAIN 전용 캘리브레이션 T1={t1:.6f} T2={t2:.6f}", flush=True)
    print(f"       TRAIN {int(train_mask.sum()):,}봉 클래스비중 {shares}", flush=True)

    missing = [c for c in cols if c not in df.columns]
    X = design_matrix(df, cols, medians)
    print(f"[피쳐] median 대체 컬럼 {len(missing)}개 {missing[:6]}", flush=True)

    Xtr, ytr = X[train_mask], y[train_mask]

    # --- 컷오프 전체 적합 (VAL/OOS+ 예측용) ---
    full = fit_gbm(Xtr, ytr)
    proba = full.predict_proba(X)

    # --- TRAIN 창은 purged 시간블록 OOF 로 덮어쓴다 ---
    tr_idx = np.flatnonzero(train_mask)
    bounds = np.array_split(tr_idx, N_FOLDS)
    oof = np.zeros((len(tr_idx), 3), dtype=np.float64)
    pos = {v: i for i, v in enumerate(tr_idx)}
    for k, te in enumerate(bounds):
        lo, hi = te[0] - PURGE_BARS, te[-1] + PURGE_BARS
        fit_rows = tr_idx[(tr_idx < lo) | (tr_idx > hi)]
        m = fit_gbm(X.iloc[fit_rows], y[fit_rows], seed=SEED + k)
        oof[[pos[v] for v in te]] = m.predict_proba(X.iloc[te])
        print(f"   OOF fold {k+1}/{N_FOLDS}: fit {len(fit_rows):,}봉 -> 예측 {len(te):,}봉", flush=True)
    proba[tr_idx] = oof

    # --- 진단: in-sample / OOF / VAL / OOS ---
    ins_pred = full.predict(Xtr)
    diag = {
        "train_in_sample_bal_acc": round(float(balanced_accuracy_score(ytr, ins_pred)), 4),
        "train_oof_bal_acc": round(float(balanced_accuracy_score(ytr, oof.argmax(1))), 4),
        "validation_bal_acc": round(float(balanced_accuracy_score(y[val_mask], proba[val_mask].argmax(1))), 4),
        "oos_bal_acc": round(float(balanced_accuracy_score(y[oos_mask], proba[oos_mask].argmax(1))), 4),
    }
    print(f"\n[진단] in-sample {diag['train_in_sample_bal_acc']} | OOF {diag['train_oof_bal_acc']} "
          f"| VAL {diag['validation_bal_acc']} | OOS {diag['oos_bal_acc']}", flush=True)
    print(f"       in-sample 낙관폭 = {diag['train_in_sample_bal_acc']-diag['train_oof_bal_acc']:+.4f} "
          f"(OOF 를 쓰는 이유; 이 값이 크면 컷오프만으로는 부족하다는 뜻)", flush=True)

    six = derive_six(proba)
    pred = proba.argmax(1)
    flip = float((pred[1:] != pred[:-1]).mean())
    print(f"[출력] flip율 {flip:.4f}  클래스비중 "
          f"{ {n: round(float((pred==i).mean()),3) for i,n in enumerate(CLASSES3)} }", flush=True)

    out = pd.DataFrame({"timestamp": ts})
    for k, v in six.items():
        out[k] = v
    for name, src_csv in (("2024", TRAIN_CSVS[0]), ("2025", TRAIN_CSVS[1]), ("2026_rebuilt", TRAIN_CSVS[2])):
        stamps = pd.read_csv(src_csv, usecols=["timestamp"], parse_dates=["timestamp"])["timestamp"]
        part = out[out["timestamp"].isin(set(stamps))].reset_index(drop=True)
        p = SIDECAR_DIR / f"training_features_{name}_{PREFIX}sidecar.csv"
        part.to_csv(p, index=False)
        print(f"   사이드카 {p.name}: {len(part):,}행", flush=True)

    joblib.dump({
        "model_id": "omega461_regimegbm_cut2509_20260909", "classes": CLASSES3,
        "feature_cols": cols, "feature_medians": medians, "model": full, "config": GBM3_HP,
        "train_range": f"{SPAN_START}~{CUTOFF_END}", "seed": SEED,
        "label_spec": {"family": "S12_K3", "scale_bars": 12, "debounce_k": 3, "T1_er12": t1, "T2_er24": t2,
                       "thresholds_calibrated_on": "cutoff TRAIN only (<=2025-09-30)"},
        "prefix": PREFIX,
        "notes": "Omega4.6.1-RegimeGBM Rebuild Phase 0. 배포 대시보드 아티팩트와 config/피쳐/라벨정의/"
                 "시드 동일, 학습 컷오프만 2026-06-30 -> 2025-09-30. TRAIN 창 사이드카 값은 purged "
                 "시간블록 5-fold OOF, VAL/OOS+ 는 컷오프 전체 적합 모델. 대시보드 배포본과는 다른 "
                 "아티팩트이며 대시보드를 교체하지 않는다.",
    }, OUT_DIR / "regime_cut2509_model.joblib")

    report = {"cutoff": str(CUTOFF_END), "prefix": PREFIX, "n_bars": int(len(df)),
              "span": [str(ts.iloc[0]), str(ts.iloc[-1])],
              "train_bars": int(train_mask.sum()), "val_bars": int(val_mask.sum()),
              "oos_bars": int(oos_mask.sum()),
              "label_thresholds": {"T1_er12": t1, "T2_er24": t2},
              "train_class_shares": shares, "diagnostics": diag,
              "output_flip_rate": flip, "median_imputed_features": missing,
              "oof": {"n_folds": N_FOLDS, "purge_bars": PURGE_BARS,
                      "scheme": "contiguous time blocks over the cutoff TRAIN window"},
              "six_column_derivation": "odyssey_regime3_live.py::_append_current(372-392) 동일"}
    (OUT_DIR / "phase0_cutoff_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n산출물: {OUT_DIR}/phase0_cutoff_report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
