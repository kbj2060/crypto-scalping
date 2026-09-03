#!/usr/bin/env python3
"""XRP 레짐 `S96_K9` 승격 확인 -- **미사용 창(2026-04-01~06-30) 단일 노출**.

## 왜

2026-09-03 격자 확장에서 `S96_K9`가 배포본 `S48_K6`을 네 축 전부 이겼다(Phase 2/3/3b/플리커).
그러나 그 평가는 2025-09~2026-03 구간에서 이뤄졌고, **원본 S48_K6 채택은 2026-07~08
구간에서 이뤄졌다**(그 창은 이미 1회 소진). 서로 다른 창이라 원본 비교를 그 자체의 잣대로
뒤집은 게 아니다.

⇒ **아직 한 번도 쓰이지 않은 2026-04-01~2026-06-30**에서 단 한 번 확인한다.

    학습:   2024-01-01 ~ 2026-03-31   (확인 창 직전까지)
    확인:   2026-04-01 ~ 2026-06-30   ⭐**이 스크립트가 이 창의 첫 사용이자 마지막 사용**
    미사용: 2026-07-01 이후            (소진된 원본 Phase3 창)

## 사전등록 판정 기준 (실행 전 고정, 사후 변경 금지)

  1. `S96_K9`의 bal_acc >= `S48_K6`의 bal_acc          (학습가능성 후퇴 없음)
  2. `S96_K9`의 pred_flip <= `S48_K6`의 pred_flip      (표시 안정성 후퇴 없음)
  3. `S96_K9`의 chop_recall >= 0.85                    (게이팅 세그먼트가 신호를 놓치지 않음)

셋 다 만족해야 교체한다. ⚠️이 창은 이번 실행으로 소진되며 재실행은 근거로 쓸 수 없다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sklearn.ensemble import HistGradientBoostingClassifier          # noqa: E402

_S = importlib.util.spec_from_file_location(
    "xrpp3", ROOT / "scripts/research_xrp_regime_s48k6_label_train_20260903.py")
_p = importlib.util.module_from_spec(_S)
_S.loader.exec_module(_p)

_C = importlib.util.spec_from_file_location(
    "xrpclean", ROOT / "scripts/research_xrp_regime_extended_label_phase3_clean_20260903.py")
_c = importlib.util.module_from_spec(_C)
_C.loader.exec_module(_c)

OUT = ROOT / "data/research/xrp_regime_s96k9_unspent_window_confirmation_20260903.json"

FIT_START = pd.Timestamp("2024-01-01T00:00:00")
FIT_END = pd.Timestamp("2026-03-31T23:55:00")
CONF_START = pd.Timestamp("2026-04-01T00:00:00")
CONF_END = pd.Timestamp("2026-06-30T23:55:00")
NEVER_READ_FROM = pd.Timestamp("2026-07-01T00:00:00")   # 소진된 원본 Phase3 창

CANDIDATE = (96, 9)
DEPLOYED = (48, 6)
CRITERIA = {"bal_acc_not_worse": True, "flip_not_worse": True, "chop_recall_min": 0.85}


def log(m): print(f"[confirm] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    _p._assert_xrp_canon()
    payload = joblib.load(_p.GBM3_MODEL_PATH)
    feat_cols, medians = payload["feature_cols"], payload["feature_medians"]
    df = _p.load_btc_frame(feat_cols)
    ts = df["timestamp"]

    fit = ((ts >= FIT_START) & (ts <= FIT_END)).to_numpy()
    conf = ((ts >= CONF_START) & (ts <= CONF_END)).to_numpy()
    never = (ts >= NEVER_READ_FROM).to_numpy()
    log(f"XRP canonical {len(df):,}행")
    log(f"학습 {int(fit.sum()):,}봉 ({FIT_START.date()}~{FIT_END.date()})")
    log(f"⭐확인 {int(conf.sum()):,}봉 ({CONF_START.date()}~{CONF_END.date()}) -- 이 창의 첫 사용")
    log(f"미사용 {int(never.sum()):,}봉 (2026-07-01+, 소진된 원본 Phase3 창) -- 읽지 않음")
    log("")
    log("사전등록 기준: (1) bal_acc 후퇴 없음  (2) pred_flip 후퇴 없음  (3) chop_recall >= 0.85")

    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    for c in feat_cols:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(medians.get(c, 0.0))
    close = df["close"]

    res = {}
    for tag, (s, k) in (("deployed", DEPLOYED), ("candidate", CANDIDATE)):
        y = _c.make_label(close, fit, s, k)
        m = HistGradientBoostingClassifier(random_state=_p.SEED, **_p.GBM3_HP).fit(
            x.loc[fit, feat_cols], y[fit])
        r = _p.evaluate(y[conf], m.predict(x.loc[conf, feat_cols]))
        r["label"] = f"S{s}_K{k}"
        r["train_shares"] = {c: float(np.mean(y[fit] == i)) for i, c in enumerate(_p.CLASSES3)}
        res[tag] = r
        log("")
        log(f"{tag:>9} {r['label']:<10} bal_acc={r['balanced_accuracy']:.4f} "
            f"chop_R={r['chop_recall']:.4f} chop_P={r['chop_precision']:.4f}")
        log(f"          bull_R={r['bull_recall']:.4f} bear_R={r['bear_recall']:.4f} "
            f"pred_flip={r['flip_rate']:.4f}")

    d, c = res["deployed"], res["candidate"]
    chk = {
        "bal_acc_not_worse": c["balanced_accuracy"] >= d["balanced_accuracy"],
        "flip_not_worse": c["flip_rate"] <= d["flip_rate"],
        "chop_recall_min": c["chop_recall"] >= CRITERIA["chop_recall_min"],
    }
    log("")
    log("=== 사전등록 기준 판정 ===")
    log(f"  (1) bal_acc  {c['balanced_accuracy']:.4f} >= {d['balanced_accuracy']:.4f}  "
        f"{'✅' if chk['bal_acc_not_worse'] else '❌'}  "
        f"(Δ {c['balanced_accuracy']-d['balanced_accuracy']:+.4f})")
    log(f"  (2) flip     {c['flip_rate']:.4f} <= {d['flip_rate']:.4f}  "
        f"{'✅' if chk['flip_not_worse'] else '❌'}  (Δ {c['flip_rate']-d['flip_rate']:+.4f})")
    log(f"  (3) chop_R   {c['chop_recall']:.4f} >= {CRITERIA['chop_recall_min']}  "
        f"{'✅' if chk['chop_recall_min'] else '❌'}")
    ok = all(chk.values())
    log("")
    log(f"⇒ {'✅**교체 승인** -- S48_K6 → S96_K9' if ok else '❌**교체 보류** -- 기준 미달'}")

    rep = {"window_spent_by_this_run": [str(CONF_START), str(CONF_END)],
           "fit": [str(FIT_START), str(FIT_END)],
           "never_read_from": str(NEVER_READ_FROM),
           "preregistered_criteria": CRITERIA, "checks": chk, "approved": bool(ok),
           "deployed": d, "candidate": c,
           "single_exposure": True, "rerun_is_not_evidence": True,
           "runtime_sec": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
