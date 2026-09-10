#!/usr/bin/env python3
"""대시보드 레짐 분류기를 **balnobb 라벨**로 교체 학습 — 2026-09-10, 사용자 지시
"balnobb으로 대시보드도 교체해줘".

`train_eth_regime_s12k3_20260902.py` 의 near-copy다. **바뀌는 것은 학습 타깃 라벨 하나뿐**:
S12_K3(효율비 er_12/er_24 + K=3 확인) → balancedish_nobb(ADX16 + slope15, **BB 덮어쓰기 제거**).
모델 HP(GBM3_HP)·136 feature_cols/medians·TRAIN 창·SEED·아티팩트 스키마 전부 동일하게 고정한다.

## 왜 오메가의 cut2509 아티팩트를 그대로 쓰지 않는가
balnobb 의 학습 컷오프 2025-09-30 은 **오메가 라인의 VAL(2025-10~12)/OOS(2026-01~02)를
보호하려고** 걸어둔 값이다. 대시보드는 그 제약이 없고 현행 s12k3 는 2026-06-30 까지 학습돼
있다 — cut2509 를 그대로 얹으면 11개월 낡은 모델로 후퇴한다. 그래서 **라벨만 가져오고
학습 창은 대시보드 것을 쓴다**(TRAIN_START~TRAIN_END).

## 반드시 같이 재는 것 — 표시 안정성
s12k3 가 채택된 근거 자체가 **리본 떨림을 절반으로 줄인 것**이다(예측 flip 0.0965 vs 0.1803,
"visibly flickery" 민원이 GBM2 프로젝트를 낳았다). balnobb 은 **debounce K=0** 이라 그 축을
되돌릴 수 있다(하위 프로젝트 계약 실측: OOS 전환 1,674회 · 상태의 48% 가 5봉 이내).
그래서 이 스크립트는 학습만 하지 않고 **배포본 s12k3 와 같은 창에서 flip/지속을 나란히** 낸다.
레짐 엔드포인트는 표시 전용이므로(트레이딩 경로 미소비, s12k3 트레이너 docstring 에서 확인)
판단 기준은 정확도가 아니라 **분류 충실도 + 표시 안정성**이다.

⚠️떨림이 과하면 `--debounce K` 로 예측 위에 K봉 확인을 건다. 라벨/모델 재학습이 필요 없는
   후처리이며(하위 프로젝트 계약 명시), 아티팩트 meta 에 K 를 기록한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from research_eth_regime_s12k3_label_train_20260902 import (  # noqa: E402
    GBM3_HP, GBM3_MODEL_PATH, SEED, load_frame,
)
from research_eth_regime_scalping_label_geometry_20260902 import TRAIN_END, TRAIN_START  # noqa: E402
from build_omega461_balancedish_nobb_sidecar_20260909 import HMM_MODEL, balancedish_nobb  # noqa: E402

MODEL_ID = "eth_regime_balnobb_20260910"
OUT_DIR = ROOT / f"tmp/{MODEL_ID}"
DEPLOYED = ROOT / "tmp/eth_regime_s12k3_20260902/model.joblib"
CLASSES3 = ["bull", "bear", "chop"]
EVAL_FROM = pd.Timestamp("2026-07-01")      # 두 모델 공통 미학습 구간(s12k3 TRAIN 끝 이후)


def _debounce(pred: np.ndarray, k: int) -> np.ndarray:
    """K봉 연속 확인. k<=1 이면 그대로 반환."""
    if k <= 1:
        return pred
    out = pred.copy()
    cur, run = pred[0], 1
    for i in range(1, len(pred)):
        if pred[i] == cur:
            run += 1
        else:
            run = 1
            cur = pred[i]
        out[i] = cur if run >= k else out[i - 1]
    return out


def _stability(pred: np.ndarray) -> dict:
    flip = float((pred[1:] != pred[:-1]).mean())
    runs, n = [], 1
    for i in range(1, len(pred)):
        if pred[i] == pred[i - 1]:
            n += 1
        else:
            runs.append(n); n = 1
    runs.append(n)
    r = np.array(runs)
    return {"flip_rate": round(flip, 4), "median_run_bars": float(np.median(r)),
            "n_transitions": int(len(r) - 1), "share_runs_le5": round(float((r <= 5).mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debounce", type=int, default=0, help="예측 위 K봉 확인(0/1=없음)")
    ap.add_argument("--write", action="store_true", help="아티팩트를 실제로 저장")
    a = ap.parse_args()

    src = joblib.load(GBM3_MODEL_PATH)
    feat_cols, medians = src["feature_cols"], src["feature_medians"]
    cfg = joblib.load(HMM_MODEL)["label_config"]

    df = load_frame()
    ts = df["timestamp"]
    tr = ((ts >= TRAIN_START) & (ts <= TRAIN_END)).to_numpy()
    y = balancedish_nobb(df, cfg)

    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    for c in feat_cols:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(medians.get(c, 0.0))

    shares = {n: round(float((y[tr] == i).mean()), 4) for i, n in enumerate(CLASSES3)}
    print(f"TRAIN {int(tr.sum()):,}봉 {TRAIN_START.date()}~{TRAIN_END.date()} | 라벨 배분 {shares}")
    print(f"라벨 cfg {cfg}", flush=True)

    model = HistGradientBoostingClassifier(random_state=SEED, **GBM3_HP).fit(x[tr], y[tr])
    assert list(model.classes_) == [0, 1, 2], f"예상 밖 클래스 순서 {model.classes_}"

    # ── 표시 안정성 정면 비교 (두 모델 공통 미학습 구간) ────────────────────────────
    ev = (ts >= EVAL_FROM).to_numpy()
    print(f"\n평가창 {ts[ev].min()} ~ {ts[ev].max()} ({int(ev.sum()):,}봉) — 두 모델 모두 미학습")
    dep = joblib.load(DEPLOYED)
    p_dep = dep["model"].predict(x.loc[ev, dep["feature_cols"]])
    p_new = model.predict(x.loc[ev, feat_cols])
    rows = [("배포본 s12k3 (K=3 라벨)", p_dep), ("balnobb K=0", p_new)]
    for k in (3, 6):
        rows.append((f"balnobb + 후처리 debounce K={k}", _debounce(p_new, k)))
    print(f"\n{'안':<30}{'flip':>8}{'중앙지속':>9}{'전환수':>8}{'≤5봉비율':>10}{'추세비중':>9}")
    stab = {}
    for name, pr in rows:
        s = _stability(pr); stab[name] = s
        trend_share = float((pr != 2).mean())
        print(f"{name:<30}{s['flip_rate']:>8.4f}{s['median_run_bars']:>9.1f}"
              f"{s['n_transitions']:>8d}{s['share_runs_le5']:>10.3f}{trend_share:>9.3f}")
    print(f"\n라벨 일치율(예측): {float((p_dep == p_new).mean()):.3f}")

    if not a.write:
        print("\n(--write 없이 실행 — 아티팩트 저장 안 함)")
        return

    pred_all = model.predict(x[feat_cols])
    k = max(a.debounce, 0)
    payload = {
        "model_id": MODEL_ID, "classes": CLASSES3,
        "feature_cols": feat_cols, "feature_medians": medians,
        "model": model, "config": GBM3_HP,
        "train_range": f"{TRAIN_START.isoformat()} ~ {TRAIN_END.isoformat()}",
        "serving_debounce_k": k,
        "label_spec": {"family": "balancedish_adx16_slope15_NO_BB_OVERRIDE (balnobb)",
                       "config": cfg,
                       "diff_vs_balancedish": "덮어쓰기에서 (bb_width < tight_bb_max) 항 제거",
                       "definition": ("ema21 slope5; adx_14; trending=adx>=trend_adx_min; "
                                      "bull=trending&slope>slope_min; bear=mirror; chop=rest; "
                                      "then chop override adx<weak_adx_max (BB 항 없음)"),
                       "debounce_k_in_label": 0},
        "display_stability": stab,
        "notes": ("2026-09-10 사용자 지시로 대시보드 레짐 라벨을 S12_K3 -> balnobb 로 교체. "
                  "balnobb 은 오메가4.6.1 레짐 척추로 채택된 라벨이나 그 아티팩트는 컷오프 "
                  "2025-09-30(오메가 VAL/OOS 보호용)이라 그대로 쓰면 대시보드가 11개월 낡는다 "
                  "— 라벨만 가져오고 학습 창은 대시보드 것(TRAIN_START~TRAIN_END)을 썼다. "
                  "모델 HP·136 피쳐·SEED·아티팩트 스키마는 s12k3 와 동일하므로 라이브는 "
                  "MODEL_PATH 한 줄만 바꾸면 된다. 레짐 엔드포인트는 표시 전용이다."),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, OUT_DIR / "model.joblib")
    (OUT_DIR / "train_report.json").write_text(json.dumps(
        {kk: vv for kk, vv in payload.items() if kk not in ("model", "feature_medians", "feature_cols")}
        | {"n_features": len(feat_cols), "train_class_shares": shares,
           "pred_shares_all": {n: round(float((pred_all == i).mean()), 4)
                               for i, n in enumerate(CLASSES3)}},
        indent=2, ensure_ascii=False))
    print(f"\n저장 {OUT_DIR / 'model.joblib'}  (serving_debounce_k={k})")


if __name__ == "__main__":
    main()
