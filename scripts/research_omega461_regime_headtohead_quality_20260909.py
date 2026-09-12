"""레짐 분류기 자체의 성능 정면 비교 — 전신 wide24 HMM vs 컷오프 재학습 S12_K3 GBM.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계약: docs/model_contracts/omega461_regimegbm_rebuild_contract.md

왜 별도 문서인가
---------------
Phase 1 은 **다운스트림 PnL** 을 쟀고 무신호로 끝났다(표본 29·13건). 이 스크립트는 그와 별개로
**레짐 태그 자체가 얼마나 좋은가**를 잰다. 두 모델의 자체 검증 bal_acc(0.7638 vs 0.8550)는
**서로 다른 라벨에 대한 값이라 직접 비교할 수 없다** — 그래서 여기서는 공통 잣대를 세 겹으로 쓴다.

공정성 확보
----------
평가창은 validation(2025-10~12)과 oos(2026-01~02)다. 이 두 창은
  · wide24 HMM  : 2024년만 학습 → OOS
  · cut2509 GBM : 2025-09-30 까지 학습 → OOS
**양쪽 모두 진짜 out-of-sample** 이다. (배포 대시보드 GBM 은 이 창을 학습에 포함하므로 제외한다.)

세 겹의 잣대
-----------
1. **양방향 교차 라벨** — 각 분류기를 **두 라벨 모두**에 대해 채점한다. 라벨은 각자 자기 모델에
   유리하므로 한쪽만 보면 안 된다. 두 방향을 다 보고해야 정직하다.
     · balancedish (HMM 의 타깃): `experiment_regime3_current_hmm_wide24_20260529._current_labels3_thresholded`
     · S12_K3 (GBM 의 타깃): 컷오프 TRAIN 에서 캘리브레이션된 임계값을 아티팩트에서 읽어 재사용
       (평가창에서 재캘리브레이션하지 않는다)
   지표: balanced accuracy, macro-F1, log-loss(확률 사용).
2. **라벨 무관 — 전방수익 판별력.** 좋은 레짐 태그라면 bull 은 이후 수익이 양, bear 는 음,
   chop 은 0 근처이면서 변동성이 낮아야 한다. 라벨 정의에 의존하지 않고 **오메가가 실제로
   이 태그를 쓰는 목적**(전문가 라우팅)에 가장 가까운 잣대다.
     · bull−bear 전방수익 격차(bp), chop 대비 추세 구간 변동성비
     · 호라이즌 12봉(1h) / 48봉(4h) / 288봉(24h)
3. **안정성** — flip율, 중앙 상태지속. 라우팅 신호가 얼마나 자주 바뀌는가.

준수: 신규 학습 없음(양쪽 다 저장된 예측/사이드카 재스코어링). 라이브 파일 미변경.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiment_regime3_current_hmm_wide24_20260529 import _current_labels3_thresholded  # noqa: E402
from research_eth_regime_scalping_label_geometry_20260902 import _debounce, scaled_label  # noqa: E402

WIDE24_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
CUT_DIR = ROOT / "data/ensemble/supervised/omega461_regimegbm_cut2509_20260909"
CUT_MODEL = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/regime_cut2509_model.joblib"
HMM_MODEL = WIDE24_DIR / "regime3_current_sensitive_hmm_wide24_2024.joblib"
CUT_PREFIX = "regime3_s12k3_cut2509_"
W24_PREFIX = "regime3_current_sensitive_wide24_"
CLASSES = ("bull", "bear", "chop")

BASE_CSVS = {"2024": "training_features_2024.csv", "2025": "training_features_2025.csv",
             "2026_rebuilt": "training_features_2026_rebuilt.csv"}
SPLITS = {"validation": ("2025-10-01", "2025-12-31 23:55:00"),
          "oos": ("2026-01-01", "2026-02-28 23:55:00")}
HORIZONS = (12, 48, 288)
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
    return (pd.concat(parts, ignore_index=True).sort_values("timestamp")
              .drop_duplicates("timestamp", keep="last").reset_index(drop=True))


def probs(df: pd.DataFrame, prefix: str) -> np.ndarray:
    p = df[[f"{prefix}{c}_prob" for c in CLASSES]].to_numpy(np.float64)
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def stability(pred: np.ndarray) -> dict:
    runs, c = [], 1
    for i in range(1, len(pred)):
        if pred[i] == pred[i - 1]:
            c += 1
        else:
            runs.append(c); c = 1
    runs.append(c)
    return {"flip_rate": float((pred[1:] != pred[:-1]).mean()),
            "median_run_bars": float(np.median(runs))}


def label_scores(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p.argmax(1)
    return {"balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
            "macro_f1": round(float(f1_score(y, pred, average="macro", labels=[0, 1, 2])), 4),
            "log_loss": round(float(log_loss(y, p, labels=[0, 1, 2])), 4)}


def forward_power(pred: np.ndarray, close: np.ndarray) -> dict:
    """라벨 무관 판별력: 태그별 전방수익 평균/변동성."""
    out = {}
    for h in HORIZONS:
        fwd = np.full(len(close), np.nan)
        fwd[:-h] = (close[h:] - close[:-h]) / close[:-h]
        ok = np.isfinite(fwd)
        stats = {}
        for i, n in enumerate(CLASSES):
            m = ok & (pred == i)
            stats[n] = {"n": int(m.sum()),
                        "mean_bp": round(float(np.mean(fwd[m]) * 1e4), 2) if m.any() else None,
                        "std_bp": round(float(np.std(fwd[m]) * 1e4), 2) if m.any() else None}
        bull, bear, chop = stats["bull"], stats["bear"], stats["chop"]
        spread = (bull["mean_bp"] - bear["mean_bp"]) if bull["mean_bp"] is not None and bear["mean_bp"] is not None else None
        trend_std = np.mean([s for s in (bull["std_bp"], bear["std_bp"]) if s is not None]) if bull["std_bp"] else None
        out[f"h{h}"] = {"by_class": stats, "bull_minus_bear_bp": round(spread, 2) if spread is not None else None,
                        "trend_over_chop_std_ratio": round(float(trend_std / chop["std_bp"]), 4)
                        if trend_std and chop["std_bp"] else None}
    return out


def main() -> int:
    df_all = load_all()
    cut_pay = joblib.load(CUT_MODEL)
    t1 = float(cut_pay["label_spec"]["T1_er12"]); t2 = float(cut_pay["label_spec"]["T2_er24"])
    hmm_cfg = joblib.load(HMM_MODEL)["label_config"]
    print(f"[라벨] balancedish cfg={hmm_cfg}", flush=True)
    print(f"[라벨] S12_K3 임계값(컷오프 TRAIN 캘리브레이션 재사용) T1={t1:.6f} T2={t2:.6f}", flush=True)

    # 라벨은 전체 구간에서 한 번에 만든다(롤링 lookback 이 창 경계에서 잘리지 않도록)
    y_bal_all = _current_labels3_thresholded(df_all, hmm_cfg)
    y_s12_all = _debounce(scaled_label(df_all["close"], 12, t1, t2), 3)

    report = {"windows": {}, "note": "두 창 모두 wide24 HMM(2024 학습)·cut2509 GBM(≤2025-09-30 학습) "
                                     "양쪽에 진짜 out-of-sample. 배포 대시보드 GBM은 이 창을 학습에 "
                                     "포함하므로 비교에서 제외."}
    for split, (s, e) in SPLITS.items():
        m = ((df_all["timestamp"] >= s) & (df_all["timestamp"] <= e)).to_numpy()
        d = df_all[m].reset_index(drop=True)
        close = pd.to_numeric(d["close"], errors="raise").to_numpy(np.float64)
        labels = {"balancedish_HMM타깃": y_bal_all[m], "S12K3_GBM타깃": y_s12_all[m]}
        models = {"wide24_HMM(전신)": probs(d, W24_PREFIX), "cut2509_GBM(후보)": probs(d, CUT_PREFIX)}

        print(f"\n{'='*74}\n[{split}] {s} ~ {e}  {len(d):,}봉", flush=True)
        for lname, y in labels.items():
            sh = {n: round(float((y == i).mean()), 3) for i, n in enumerate(CLASSES)}
            print(f"  라벨 {lname:18s} 클래스비중 {sh}", flush=True)

        res = {"bars": int(len(d)), "cross_label": {}, "forward_power": {}, "stability": {}}
        print(f"\n  {'모델':22s}{'라벨':20s}{'bal_acc':>9s}{'macroF1':>9s}{'logloss':>9s}", flush=True)
        for mname, p in models.items():
            for lname, y in labels.items():
                sc = label_scores(y, p)
                res["cross_label"].setdefault(mname, {})[lname] = sc
                print(f"  {mname:22s}{lname:20s}{sc['balanced_accuracy']:9.4f}"
                      f"{sc['macro_f1']:9.4f}{sc['log_loss']:9.4f}", flush=True)

        print(f"\n  {'모델':22s}{'flip율':>9s}{'중앙지속':>9s}   전방수익 bull−bear (bp)", flush=True)
        for mname, p in models.items():
            pred = p.argmax(1)
            st = stability(pred); fp = forward_power(pred, close)
            res["stability"][mname] = st
            res["forward_power"][mname] = fp
            spreads = "  ".join(f"h{h}={fp[f'h{h}']['bull_minus_bear_bp']:+8.1f}" for h in HORIZONS)
            print(f"  {mname:22s}{st['flip_rate']:9.4f}{st['median_run_bars']:9.1f}   {spreads}", flush=True)
        print(f"\n  {'모델':22s}추세/횡보 변동성비 (>1 이면 chop 이 실제로 더 잔잔)", flush=True)
        for mname in models:
            fp = res["forward_power"][mname]
            r = "  ".join(f"h{h}={fp[f'h{h}']['trend_over_chop_std_ratio']:.3f}" for h in HORIZONS)
            print(f"  {mname:22s}{r}", flush=True)
        report["windows"][split] = res

    (OUT / "regime_headtohead_quality.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n산출물: {OUT}/regime_headtohead_quality.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
