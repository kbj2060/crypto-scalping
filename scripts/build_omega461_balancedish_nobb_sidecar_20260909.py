"""balancedish **BB 덮어쓰기 제거** 변형(balnobb) 사이드카 생성 + 잣대 산출.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계기: 사용자 지시(2026-09-09) "BB 덮어쓰기 제거한 변형 만들어서 같은 잣대로 차트를 만들어서
비교해줘. 진입과 청산은 어차피 딥러닝 모델이 측정하고 우린 차트에 맞게 레짐만 잘 분류하면 돼."

무엇을 바꾸나 — 한 줄
--------------------
원본 `_current_labels3_thresholded` (experiment_regime3_current_hmm_wide24_20260529.py:130):

    labels = 2(chop)                                  # 기본
    labels[trending & (slope > +slope_min)] = 0       # bull
    labels[trending & (slope < -slope_min)] = 1       # bear
    labels[(adx < weak_adx_max) | (bb_width < tight_bb_max)] = 2   # ← 덮어쓰기

진단(`diagnose_omega461_balancedish_chop_override_20260909.py`)에서 확인된 사실:
· 추세 후보의 **42.2%(OOS 전체) / 52.9%(횡보 창)** 가 이 마지막 줄로 chop 이 된다
· 원인은 **BB 100%** — ADX 조건은 단 1건도 덮어쓰지 못한다. 논리적으로 당연하다:
  bull/bear 로 찍히려면 adx>=16 인데 덮어쓰기 조건은 adx<12 라 동시 성립 불가다.
  즉 `weak_adx_max` 는 **덮어쓰기로서는 죽은 조건**이고, 실질 덮어쓰기는 BB 하나다.
· 되돌려진 봉의 ADX 중앙값이 36~43 — 통상 "강한 추세"로 보는 25 를 크게 넘는다.

이 변형은 그 BB 절만 뺀다. ADX 절은 **남긴다**(이미 chop 인 봉에만 작용하므로 무해하고,
원본과의 차이를 BB 하나로 고정하기 위해).

프로토콜은 balgbm / s12k3 arm 과 **완전히 동일**하게 고정한다 — 동일 HGB HP, 동일 136
feature_cols/medians, 컷오프 ≤2025-09-30, SEED 7529, TRAIN 창 purged 5-fold OOF.
따라서 세 arm 의 차이는 여전히 **학습 타깃 라벨 하나뿐**이다.

⚠️ 예상되는 부작용: 덮어쓰기를 빼면 추세 비중이 30.1% → 52.1%(OOS 전체) 로 뛴다. 절반 이상이
추세가 되는 레짐이 라우팅에 쓸 만한지는 **차트로 판단**한다(사용자 기준: "차트에 맞게 레짐만
잘 분류하면 된다"). 안정성 지표도 같이 낸다 — 임계가 느슨해지면 경계 근처에서 더 자주
뒤집힐 수 있기 때문이다.

준수: 학습은 이 변형의 레짐 GBM 하나(컷오프 준수). 라이브 파일 미변경.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiment_regime3_current_hmm_wide24_20260529 import _adx, _num  # noqa: E402
from research_eth_regime_s12k3_label_train_20260902 import GBM3_HP, GBM3_MODEL_PATH, SEED  # noqa: E402
from research_eth_regime_scalping_label_geometry_20260902 import TRAIN_CSVS  # noqa: E402
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402

HMM_MODEL = (ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
                  / "regime3_current_sensitive_hmm_wide24_2024.joblib")
PREFIX = "regime3_balnobb_cut2509_"
CLASSES = ["bull", "bear", "chop"]
SPAN_START = pd.Timestamp("2024-01-01T00:00:00")
CUTOFF_END = pd.Timestamp("2025-09-30T23:55:00")
SPAN_END = pd.Timestamp("2026-08-30T23:55:00")
SPLITS = {"validation": ("2025-10-01", "2025-12-31 23:55:00"),
          "oos": ("2026-01-01", "2026-02-28 23:55:00")}
N_FOLDS, PURGE_BARS = 5, 48
OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909"
SIDECAR_DIR = ROOT / "data/ensemble/supervised/omega461_balnobb_cut2509_20260909"


def balancedish_nobb(frame: pd.DataFrame, cfg: dict) -> np.ndarray:
    """원본과 동일하되 덮어쓰기에서 `bb_width < tight_bb_max` 항만 제거."""
    close, high, low = _num(frame, "close"), _num(frame, "high"), _num(frame, "low")
    ema21 = close.ewm(span=21, adjust=False).mean()
    slope = ((ema21 - ema21.shift(5)) / (close * 5.0 + 1e-12)
             ).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    adx = _num(frame, "adx_14", np.nan)
    if adx.isna().all():
        adx = _adx(high, low, close)
    adx = adx.fillna(0.0).to_numpy()

    labels = np.full(len(frame), 2, dtype=np.int64)
    trending = adx >= float(cfg["trend_adx_min"])
    smin = float(cfg["slope_min"])
    labels[trending & (slope > smin)] = 0
    labels[trending & (slope < -smin)] = 1
    labels[adx < float(cfg["weak_adx_max"])] = 2     # BB 항 제거, ADX 항만 유지
    return labels


def load_span() -> pd.DataFrame:
    frames = [pd.read_csv(p, low_memory=False, parse_dates=["timestamp"]) for p in TRAIN_CSVS]
    df = (pd.concat(frames, ignore_index=True).sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last").reset_index(drop=True))
    df = df[(df["timestamp"] >= SPAN_START) & (df["timestamp"] <= SPAN_END)].reset_index(drop=True)
    return _with_raw_state12(df)


def fit(X, y, seed=SEED):
    m = HistGradientBoostingClassifier(random_state=seed, **GBM3_HP)
    m.fit(X, y)
    return m


def derive_six(proba: np.ndarray) -> dict[str, np.ndarray]:
    p = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
    s = np.sort(p, axis=1)
    out = {f"{PREFIX}{n}_prob": p[:, i] for i, n in enumerate(CLASSES)}
    out[f"{PREFIX}confidence"] = p.max(axis=1)
    out[f"{PREFIX}margin"] = s[:, -1] - s[:, -2]
    out[f"{PREFIX}entropy"] = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1) / np.log(3.0)
    return out


def stability(pred):
    runs, c = [], 1
    for i in range(1, len(pred)):
        if pred[i] == pred[i - 1]:
            c += 1
        else:
            runs.append(c); c = 1
    runs.append(c)
    r = np.array(runs)
    return {"transitions": int((pred[1:] != pred[:-1]).sum()),
            "flip_rate": round(float((pred[1:] != pred[:-1]).mean()), 4),
            "median_run_bars": float(np.median(r)), "mean_run_bars": round(float(r.mean()), 1),
            "share_le5_bars": round(float((r <= 5).mean()), 4)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    src = joblib.load(GBM3_MODEL_PATH)
    cols, medians = src["feature_cols"], src["feature_medians"]
    cfg = joblib.load(HMM_MODEL)["label_config"]

    df = load_span()
    ts = df["timestamp"]
    y = balancedish_nobb(df, cfg)
    train_mask = (ts <= CUTOFF_END).to_numpy()
    X = (df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
           .replace([np.inf, -np.inf], np.nan).fillna(pd.Series(medians)).fillna(0.0))
    print(f"[프레임] {len(df):,}봉  [라벨] balancedish_nobb (BB 덮어쓰기 제거)", flush=True)
    print(f"[학습] 컷오프 ≤{CUTOFF_END.date()} {int(train_mask.sum()):,}봉  클래스비중 "
          f"{ {n: round(float((y[train_mask]==i).mean()),4) for i,n in enumerate(CLASSES)} }", flush=True)

    full = fit(X[train_mask], y[train_mask])
    proba = full.predict_proba(X)
    tr_idx = np.flatnonzero(train_mask)
    oof = np.zeros((len(tr_idx), 3))
    pos = {v: i for i, v in enumerate(tr_idx)}
    for k, te in enumerate(np.array_split(tr_idx, N_FOLDS)):
        lo, hi = te[0] - PURGE_BARS, te[-1] + PURGE_BARS
        rows = tr_idx[(tr_idx < lo) | (tr_idx > hi)]
        oof[[pos[v] for v in te]] = fit(X.iloc[rows], y[rows], seed=SEED + k).predict_proba(X.iloc[te])
        print(f"   OOF fold {k+1}/{N_FOLDS}", flush=True)
    proba[tr_idx] = oof
    print(f"[진단] in-sample {balanced_accuracy_score(y[train_mask], full.predict(X[train_mask])):.4f} | "
          f"OOF {balanced_accuracy_score(y[train_mask], oof.argmax(1)):.4f}", flush=True)

    six = derive_six(proba)
    out = pd.DataFrame({"timestamp": ts})
    for k, v in six.items():
        out[k] = v
    for p in TRAIN_CSVS:
        tag = p.stem.replace("training_features_", "")
        stamps = set(pd.read_csv(p, usecols=["timestamp"], parse_dates=["timestamp"])["timestamp"])
        part = out[out["timestamp"].isin(stamps)].reset_index(drop=True)
        part.to_csv(SIDECAR_DIR / f"training_features_{tag}_{PREFIX}sidecar.csv", index=False)
        print(f"   사이드카 {tag}: {len(part):,}행", flush=True)

    joblib.dump({"model_id": "omega461_balnobb_cut2509_20260909", "classes": CLASSES,
                 "feature_cols": cols, "feature_medians": medians, "model": full, "config": GBM3_HP,
                 "train_range": f"{SPAN_START}~{CUTOFF_END}", "seed": SEED, "prefix": PREFIX,
                 "label_spec": {"family": "balancedish_adx16_slope15_NO_BB_OVERRIDE", "config": cfg,
                                "diff_vs_original": "덮어쓰기에서 (bb_width < tight_bb_max) 항 제거"},
                 "notes": "BB 덮어쓰기 제거 변형. 프로토콜은 balgbm/s12k3 arm 과 동일."},
                OUT / "regime_balnobb_cut2509_model.joblib")

    rep = {"label_spec": "balancedish_nobb", "windows": {}}
    for split, (s, e) in SPLITS.items():
        m = ((ts >= s) & (ts <= e)).to_numpy()
        pred = proba[m].argmax(1)
        st = stability(pred)
        sh = {n: round(float((pred == i).mean()), 3) for i, n in enumerate(CLASSES)}
        recon = round(float(cohen_kappa_score(y[m], pred, labels=[0, 1, 2])), 4)
        rep["windows"][split] = {"pred_shares": sh, "stability": st, "reconstruction_kappa": recon,
                                 "label_shares": {n: round(float((y[m] == i).mean()), 3)
                                                  for i, n in enumerate(CLASSES)}}
        print(f"\n[{split}] 예측비중 {sh}  재구성κ {recon}", flush=True)
        print(f"   전환 {st['transitions']}회  flip {st['flip_rate']}  중앙 {st['median_run_bars']:.0f}봉  "
              f"평균 {st['mean_run_bars']}봉  ≤5봉 {st['share_le5_bars']*100:.1f}%", flush=True)

    (OUT / "balnobb_build_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
    print(f"\n산출물: {OUT}/balnobb_build_report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
