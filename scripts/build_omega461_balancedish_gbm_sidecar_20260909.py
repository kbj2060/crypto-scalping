"""balancedish 라벨 **유지**, HMM → GBM 모델급 교체 — 사이드카 생성 + 동일라벨 평가.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계약: docs/model_contracts/omega461_regimegbm_rebuild_contract.md
선행: `docs/experiments/omega461_regime_headtohead_quality_20260909.md` §5

왜 이 arm 인가
-------------
§5 의 라벨 난이도 보정에서 나온 결과: 같은 라벨(balancedish) 위에서 동일 프로토콜 HGB 가
전신 wide24 HMM 을 **+20.4pp(OOS bal_acc 0.9785 vs 0.7746)** 앞선다. 즉 "레짐을 GBM 으로
바꾸자"의 근거는 라벨 교체(S12_K3)가 아니라 **모델급 교체**일 수 있다.

이 arm 은 **라벨 교란을 완전히 제거한다** — 타깃은 전신과 똑같은 `balancedish_adx16_slope15_bb012`
이고 바뀌는 것은 추정기(12-state Gaussian HMM → HistGradientBoosting)뿐이다. 이 하위
프로젝트에서 가능한 가장 깨끗한 통제다.

프로토콜 (cut2509 arm 과 동일하게 고정)
--------------------------------------
· 학습 컷오프 ≤2025-09-30 (이 라인의 train_end) — VAL/OOS 는 학습에 들어가지 않는다
· 136 feature_cols / feature_medians / HP / SEED 는 GBM3 아티팩트에서 승계
· TRAIN 창은 **purged 시간블록 5-fold OOF**, VAL/OOS+ 는 컷오프 전체 적합 모델
  (balancedish 는 피쳐에서 κ≈0.96 으로 복원되므로 in-sample 은 거의 1.0 이 된다 —
   OOF 없이는 부모가 학습 때 비현실적으로 완벽한 레짐 신호를 보게 된다)
· 6컬럼 유도식은 `odyssey_regime3_live.py::_append_current`(372-392) 동일

⚠️ 해석 주의 — κ 0.96 의 양날
-----------------------------
balancedish 가 136피쳐에서 그렇게 잘 복원된다는 것은 **그 레짐 정의가 사실상 피쳐의 결정론적
함수**라는 뜻이다. 부모 TabM 의 102 base_cols 가 그 136피쳐와 크게 겹치므로, "더 정확한
balancedish 재현"이 부모에게 **새 정보를 주지 않을** 수 있다. 그래서 이 스크립트는 정확도만
보지 않고 **라벨 무관 잣대**(전방수익 판별력, 변동성 분리력)를 같이 낸다 — 전신 HMM 의 가치가
복원 가능한 규칙에서 *벗어나는* 부분(sticky=0.93 평활)에 있다면 그쪽에서 드러난다.

준수: 학습은 이 arm 의 레짐 GBM 하나뿐(컷오프 준수). 라이브 파일 미변경.
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
                             cohen_kappa_score, f1_score, log_loss)

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiment_regime3_current_hmm_wide24_20260529 import _current_labels3_thresholded  # noqa: E402
from research_eth_regime_s12k3_label_train_20260902 import GBM3_HP, GBM3_MODEL_PATH, SEED  # noqa: E402
from research_eth_regime_scalping_label_geometry_20260902 import TRAIN_CSVS  # noqa: E402
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402

WIDE24_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
HMM_MODEL = WIDE24_DIR / "regime3_current_sensitive_hmm_wide24_2024.joblib"
W24_PREFIX = "regime3_current_sensitive_wide24_"
PREFIX = "regime3_balgbm_cut2509_"
CLASSES = ["bull", "bear", "chop"]

SPAN_START = pd.Timestamp("2024-01-01T00:00:00")
CUTOFF_END = pd.Timestamp("2025-09-30T23:55:00")
SPAN_END = pd.Timestamp("2026-08-30T23:55:00")
SPLITS = {"validation": ("2025-10-01", "2025-12-31 23:55:00"),
          "oos": ("2026-01-01", "2026-02-28 23:55:00")}
N_FOLDS, PURGE_BARS = 5, 48
HORIZONS = (12, 48, 288)
OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909"
SIDECAR_DIR = ROOT / "data/ensemble/supervised/omega461_balgbm_cut2509_20260909"


def load_span() -> pd.DataFrame:
    frames = []
    for p in TRAIN_CSVS:
        b = pd.read_csv(p, low_memory=False, parse_dates=["timestamp"])
        tag = p.stem.replace("training_features_", "")
        w = pd.read_csv(WIDE24_DIR / f"training_features_{tag}_regime3_current_sensitive_hmm_wide24.csv",
                        low_memory=False, parse_dates=["timestamp"])
        frames.append(b.merge(w, on="timestamp", how="inner"))
    df = (pd.concat(frames, ignore_index=True).sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last").reset_index(drop=True))
    df = df[(df["timestamp"] >= SPAN_START) & (df["timestamp"] <= SPAN_END)].reset_index(drop=True)
    return _with_raw_state12(df)


def fit(X, y, seed=SEED):
    m = HistGradientBoostingClassifier(random_state=seed, **GBM3_HP)
    m.fit(X, y)
    return m


def derive_six(proba: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    p = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
    s = np.sort(p, axis=1)
    out = {f"{prefix}{n}_prob": p[:, i] for i, n in enumerate(CLASSES)}
    out[f"{prefix}confidence"] = p.max(axis=1)
    out[f"{prefix}margin"] = s[:, -1] - s[:, -2]
    out[f"{prefix}entropy"] = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1) / np.log(3.0)
    return out


def stability(pred):
    runs, c = [], 1
    for i in range(1, len(pred)):
        if pred[i] == pred[i - 1]:
            c += 1
        else:
            runs.append(c); c = 1
    runs.append(c)
    return {"flip_rate": round(float((pred[1:] != pred[:-1]).mean()), 4),
            "median_run_bars": float(np.median(runs))}


def forward_power(pred, close):
    out = {}
    for h in HORIZONS:
        fwd = np.full(len(close), np.nan)
        fwd[:-h] = (close[h:] - close[:-h]) / close[:-h]
        ok = np.isfinite(fwd)
        st = {}
        for i, n in enumerate(CLASSES):
            m = ok & (pred == i)
            st[n] = {"n": int(m.sum()),
                     "mean_bp": round(float(np.mean(fwd[m]) * 1e4), 2) if m.any() else None,
                     "std_bp": round(float(np.std(fwd[m]) * 1e4), 2) if m.any() else None}
        tstd = np.mean([st["bull"]["std_bp"], st["bear"]["std_bp"]]) if st["bull"]["std_bp"] and st["bear"]["std_bp"] else None
        out[f"h{h}"] = {"by_class": st,
                        "bull_minus_bear_bp": round(st["bull"]["mean_bp"] - st["bear"]["mean_bp"], 2)
                        if st["bull"]["mean_bp"] is not None and st["bear"]["mean_bp"] is not None else None,
                        "trend_over_chop_std_ratio": round(float(tstd / st["chop"]["std_bp"]), 4)
                        if tstd and st["chop"]["std_bp"] else None}
    return out


def scores(y, p):
    pred = p.argmax(1)
    prior = np.bincount(y, minlength=3) / len(y)
    return {"balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
            "macro_f1": round(float(f1_score(y, pred, average="macro", labels=[0, 1, 2])), 4),
            "cohen_kappa": round(float(cohen_kappa_score(y, pred, labels=[0, 1, 2])), 4),
            "adjusted_mutual_info": round(float(adjusted_mutual_info_score(y, pred)), 4),
            "log_loss": round(float(log_loss(y, p, labels=[0, 1, 2])), 4),
            "log_loss_skill": round(1.0 - float(log_loss(y, p, labels=[0, 1, 2]))
                                    / float(log_loss(y, np.tile(prior, (len(y), 1)), labels=[0, 1, 2])), 4)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    src = joblib.load(GBM3_MODEL_PATH)
    cols, medians = src["feature_cols"], src["feature_medians"]
    hmm_cfg = joblib.load(HMM_MODEL)["label_config"]

    df = load_span()
    ts = df["timestamp"]
    y = _current_labels3_thresholded(df, hmm_cfg)
    train_mask = (ts <= CUTOFF_END).to_numpy()
    X = (df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
           .replace([np.inf, -np.inf], np.nan).fillna(pd.Series(medians)).fillna(0.0))
    print(f"[프레임] {len(df):,}봉 {ts.iloc[0]} ~ {ts.iloc[-1]}", flush=True)
    print(f"[라벨] balancedish {hmm_cfg}", flush=True)
    print(f"[학습] 컷오프 ≤{CUTOFF_END.date()}  {int(train_mask.sum()):,}봉  "
          f"클래스비중 { {n: round(float((y[train_mask]==i).mean()),4) for i,n in enumerate(CLASSES)} }", flush=True)

    full = fit(X[train_mask], y[train_mask])
    proba = full.predict_proba(X)

    tr_idx = np.flatnonzero(train_mask)
    oof = np.zeros((len(tr_idx), 3))
    pos = {v: i for i, v in enumerate(tr_idx)}
    for k, te in enumerate(np.array_split(tr_idx, N_FOLDS)):
        lo, hi = te[0] - PURGE_BARS, te[-1] + PURGE_BARS
        fit_rows = tr_idx[(tr_idx < lo) | (tr_idx > hi)]
        oof[[pos[v] for v in te]] = fit(X.iloc[fit_rows], y[fit_rows], seed=SEED + k).predict_proba(X.iloc[te])
        print(f"   OOF fold {k+1}/{N_FOLDS}: fit {len(fit_rows):,} -> 예측 {len(te):,}", flush=True)
    proba[tr_idx] = oof

    diag = {"train_in_sample_bal_acc": round(float(balanced_accuracy_score(y[train_mask], full.predict(X[train_mask]))), 4),
            "train_oof_bal_acc": round(float(balanced_accuracy_score(y[train_mask], oof.argmax(1))), 4)}
    print(f"\n[진단] in-sample {diag['train_in_sample_bal_acc']} | OOF {diag['train_oof_bal_acc']} "
          f"| 낙관폭 {diag['train_in_sample_bal_acc']-diag['train_oof_bal_acc']:+.4f}", flush=True)

    six = derive_six(proba, PREFIX)
    out = pd.DataFrame({"timestamp": ts})
    for k, v in six.items():
        out[k] = v
    for p in TRAIN_CSVS:
        tag = p.stem.replace("training_features_", "")
        stamps = set(pd.read_csv(p, usecols=["timestamp"], parse_dates=["timestamp"])["timestamp"])
        part = out[out["timestamp"].isin(stamps)].reset_index(drop=True)
        part.to_csv(SIDECAR_DIR / f"training_features_{tag}_{PREFIX}sidecar.csv", index=False)
        print(f"   사이드카 {tag}: {len(part):,}행", flush=True)

    joblib.dump({"model_id": "omega461_balgbm_cut2509_20260909", "classes": CLASSES,
                 "feature_cols": cols, "feature_medians": medians, "model": full, "config": GBM3_HP,
                 "train_range": f"{SPAN_START}~{CUTOFF_END}", "seed": SEED, "prefix": PREFIX,
                 "label_spec": {"family": "balancedish_adx16_slope15_bb012", "config": hmm_cfg,
                                "source": "experiment_regime3_current_hmm_wide24_20260529._current_labels3_thresholded"},
                 "notes": "balancedish 라벨 유지, 추정기만 12-state Gaussian HMM -> HistGradientBoosting. "
                          "TRAIN 창은 purged 5-fold OOF, VAL/OOS+ 는 컷오프 전체 적합."},
                OUT / "regime_balgbm_cut2509_model.joblib")

    # --- 동일 라벨 위 정면 비교 ---
    report = {"diagnostics": diag, "windows": {}}
    for split, (s, e) in SPLITS.items():
        m = ((ts >= s) & (ts <= e)).to_numpy()
        d = df[m]
        close = pd.to_numeric(d["close"], errors="raise").to_numpy(np.float64)
        pw = d[[f"{W24_PREFIX}{c}_prob" for c in CLASSES]].to_numpy(np.float64)
        pw = pw / np.clip(pw.sum(axis=1, keepdims=True), 1e-12, None)
        models = {"wide24_HMM(전신)": pw, "balgbm_cut2509(후보)": proba[m]}
        print(f"\n{'='*88}\n[{split}] {s} ~ {e}  {int(m.sum()):,}봉  "
              f"라벨비중 { {n: round(float((y[m]==i).mean()),3) for i,n in enumerate(CLASSES)} }", flush=True)
        print(f"  {'모델':24s}{'bal_acc':>9s}{'macroF1':>9s}{'kappa':>8s}{'AMI':>8s}{'LLskill':>9s}", flush=True)
        res = {"bars": int(m.sum()), "scores": {}, "stability": {}, "forward_power": {}}
        for mn, p in models.items():
            sc = scores(y[m], p)
            res["scores"][mn] = sc
            print(f"  {mn:24s}{sc['balanced_accuracy']:9.4f}{sc['macro_f1']:9.4f}"
                  f"{sc['cohen_kappa']:8.4f}{sc['adjusted_mutual_info']:8.4f}{sc['log_loss_skill']:9.4f}", flush=True)
        print(f"\n  {'모델':24s}{'flip율':>9s}{'중앙지속':>9s}   bull−bear(bp)            추세/횡보 변동성비", flush=True)
        for mn, p in models.items():
            pred = p.argmax(1)
            st, fp = stability(pred), forward_power(pred, close)
            res["stability"][mn], res["forward_power"][mn] = st, fp
            sp = " ".join(f"h{h}={fp[f'h{h}']['bull_minus_bear_bp']:+7.1f}" for h in HORIZONS)
            vr = " ".join(f"h{h}={fp[f'h{h}']['trend_over_chop_std_ratio']:.3f}" for h in HORIZONS)
            print(f"  {mn:24s}{st['flip_rate']:9.4f}{st['median_run_bars']:9.1f}   {sp}   {vr}", flush=True)
        report["windows"][split] = res

    (OUT / "balgbm_same_label_comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n산출물: {OUT}/balgbm_same_label_comparison.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
