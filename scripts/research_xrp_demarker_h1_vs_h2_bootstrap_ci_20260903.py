#!/usr/bin/env python3
"""XRP demarker H=1 vs H=2 -- **표본 부트스트랩 CI**로 재판정 (시드분산은 잣대가 아니다).

## 왜 다시 하나 -- 앞 실행의 판정 기준이 약했다

`research_xrp_demarker_h1_vs_h2_tabpfn_matched_20260903.py`는 "시드분산 ±2σ 넘게 이기면
교체 검토"를 사전 기준으로 걸었고 H=1이 통과했다(VAL +0.0361 / OOS +0.0101).

⚠️**그러나 시드분산은 잘못된 분모다.** 그건 TabPFN의 모델 무작위성만 재고, **평가 표본이
유한하다는 사실**을 전혀 반영하지 않는다. 실제로:

    H=1 OOS: n=235, base rate 0.0936  ->  양성 **약 22건**
    H=2 OOS: n=235, base rate 0.1957  ->  양성 약 46건

양성 22건짜리 AUC의 표본 불확실성은 시드분산(0.0035)보다 **한 자릿수 이상 크다.**
⇒ 평가표본 부트스트랩으로 각 AUC의 95% CI를 구하고 **겹치는지**를 본다.

## ⚠️추가로, 이건 애초에 matched 비교가 아니다

두 라벨의 base rate가 **2배 넘게 다르다**(VAL 0.1309 vs 0.2584). 서로 다른 난이도의 문제이므로
AUC 대소를 그대로 "모델이 낫다"로 읽으면 안 된다
(`feedback_cross_model_auc_comparison_requires_matched_label_difficulty`).
그래서 AUC와 함께 **prevalence 무관 보조지표**도 같이 낸다:

  · AUC (부트스트랩 95% CI)
  · **AP / base rate** = PR-AUC를 무작위 기준선으로 나눈 값 (희소 라벨 비교에 적합)

## 판정 (재고정)

  H=1이 VAL·OOS 둘 다에서 **AUC CI가 H=2의 점추정을 배제**해야 교체를 검토한다.
  하나라도 겹치면 **현행 H=2 유지.**

⚠️HOLDOUT 미터치 (H=2의 0.6759로 이미 소진).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_S = importlib.util.spec_from_file_location(
    "xrpdem", ROOT / "scripts/research_xrp_demarker_extreme_metalabel_tabpfn_20260903.py")
_d = importlib.util.module_from_spec(_S)
_S.loader.exec_module(_d)

CAND_CSV = (ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903"
            / "xrp_5m_evidence_signal_candidates_tier0.csv")
EXPECTED_ROWS = 272_490
OUT = ROOT / "data/research/xrp_demarker_h1_vs_h2_bootstrap_ci_20260903.json"

K, HORIZONS, DEPLOYED_H = 1.5, [1, 2], 2
B_BOOT, BOOT_SEED = 4000, 20260903


def log(m): print(f"[boot] {m}", flush=True)


def fit_probas(tr, ev, cols, seed):
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(device="cuda", random_state=seed)
    clf.fit(tr[cols], tr["hit"].to_numpy().astype(int))
    return clf.predict_proba(ev[cols])[:, 1]


def boot_metrics(y, p, rng):
    """평가표본 부트스트랩. 양성/음성이 둘 다 있는 리샘플만 센다."""
    n = len(y)
    aucs, aps = [], []
    for _ in range(B_BOOT):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        pp = p[idx]
        aucs.append(roc_auc_score(yy, pp))
        aps.append(average_precision_score(yy, pp) / yy.mean())
    a = np.array(aucs); q = np.array(aps)
    return {"auc": float(roc_auc_score(y, p)),
            "auc_ci": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
            "ap_over_base": float(average_precision_score(y, p) / y.mean()),
            "ap_over_base_ci": [float(np.percentile(q, 2.5)), float(np.percentile(q, 97.5))],
            "n": int(n), "n_pos": int(y.sum()), "base_rate": float(y.mean()),
            "n_boot_used": int(len(a))}


def main() -> int:
    t0 = time.time()
    f = pd.read_csv(CAND_CSV)
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True).dt.tz_localize(None)
    f = _d.add_missing_features(f)
    if abs(len(f) - EXPECTED_ROWS) > 200:
        raise RuntimeError(f"행수 {len(f):,} != XRP 기대치 {EXPECTED_ROWS:,}")
    log(f"XRP 프레임 {len(f):,}행 | 부트스트랩 B={B_BOOT:,} | 시드 {_d.SEEDS[0]}")
    log(f"⚠️HOLDOUT 미터치")

    cols = [c for c in _d.FEATURE_COLUMNS]
    rng = np.random.default_rng(BOOT_SEED)
    rep = {"k": K, "gap": _d.CLUSTER_GAP, "B_boot": B_BOOT, "tabpfn_seed": _d.SEEDS[0],
           "holdout_touched": False, "deployed_horizon": DEPLOYED_H, "horizons": {}}

    for h in HORIZONS:
        fires, _st = _d.build_final_fires(f, h, K, _d.CLUSTER_GAP)
        ts = pd.to_datetime(fires["timestamp"])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_localize(None)
        fires = fires.assign(timestamp=ts)
        c = [x for x in cols if x in fires.columns]
        tr = fires.loc[fires["timestamp"] < _d.VAL_START].reset_index(drop=True)
        splits = {
            "val": fires.loc[(fires["timestamp"] >= _d.VAL_START)
                             & (fires["timestamp"] < _d.OOS_START)].reset_index(drop=True),
            "oos": fires.loc[(fires["timestamp"] >= _d.OOS_START)
                             & (fires["timestamp"] < _d.HOLDOUT_START)].reset_index(drop=True)}
        log("")
        log(f"=== HORIZON {h}{'  ⭐배포' if h == DEPLOYED_H else ''} ===")
        res = {}
        for tag, ev in splits.items():
            p = fit_probas(tr, ev, c, _d.SEEDS[0])
            y = ev["hit"].to_numpy().astype(int)
            m = boot_metrics(y, p, rng)
            res[tag] = m
            log(f"  {tag.upper():<4} n={m['n']:<4} 양성={m['n_pos']:<4} (base {m['base_rate']:.4f})")
            log(f"       AUC {m['auc']:.4f}  95%CI [{m['auc_ci'][0]:.4f}, {m['auc_ci'][1]:.4f}]"
                f"   폭 {m['auc_ci'][1]-m['auc_ci'][0]:.4f}")
            log(f"       AP/base {m['ap_over_base']:.3f}x  CI "
                f"[{m['ap_over_base_ci'][0]:.3f}, {m['ap_over_base_ci'][1]:.3f}]")
        rep["horizons"][str(h)] = res

    log("")
    log("=" * 70)
    log("판정 (재고정: H=1의 AUC CI가 H=2 점추정을 배제해야 교체 검토)")
    log("=" * 70)
    a, b = rep["horizons"]["1"], rep["horizons"][str(DEPLOYED_H)]
    v, allwin = {}, True
    log(f"{'분할':<5} {'H=1 AUC [CI]':>28} {'H=2 AUC':>9}  H=2가 H=1 CI 밖?")
    for tag in ("val", "oos"):
        lo, hi = a[tag]["auc_ci"]
        excl = b[tag]["auc"] < lo
        allwin &= excl
        v[tag] = {"h1_auc": a[tag]["auc"], "h1_ci": [lo, hi], "h2_auc": b[tag]["auc"],
                  "h2_below_h1_ci": bool(excl),
                  "h1_ap_over_base": a[tag]["ap_over_base"],
                  "h2_ap_over_base": b[tag]["ap_over_base"]}
        log(f"{tag.upper():<5} {a[tag]['auc']:.4f} [{lo:.4f}, {hi:.4f}] {b[tag]['auc']:>9.4f}  "
            f"{'✅배제' if excl else '❌겹침'}")
    log("")
    log(f"{'분할':<5} {'H=1 AP/base':>12} {'H=2 AP/base':>12}   (prevalence 무관 보조지표)")
    for tag in ("val", "oos"):
        log(f"{tag.upper():<5} {a[tag]['ap_over_base']:>12.3f} {b[tag]['ap_over_base']:>12.3f}")
    log("")
    log(f"⚠️base rate 불일치: H=1 VAL {a['val']['base_rate']:.4f}/OOS {a['oos']['base_rate']:.4f}"
        f"  vs  H=2 VAL {b['val']['base_rate']:.4f}/OOS {b['oos']['base_rate']:.4f}"
        f"  -- 서로 다른 난이도의 문제다")
    log("")
    log(f"⇒ {'⚠️**교체 검토 가능**' if allwin else '✅**현행 H=2 유지** -- 표본 불확실성 안에서 구분되지 않는다'}")
    rep["verdict"] = {"h1_beats_h2_on_both": bool(allwin), "by_split": v,
                      "note": "시드분산이 아니라 평가표본 부트스트랩 CI로 판정"}
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
