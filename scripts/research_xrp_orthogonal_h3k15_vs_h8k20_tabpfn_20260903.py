#!/usr/bin/env python3
"""XRP `orthogonal_combo` **H=3/K=1.5 vs 배포 H=8/K=2.0** TabPFN + 부트스트랩 CI 판정.

## 왜

2026-09-03 격자 감사 흐름:

  1. 자동 argmax는 8시드 **전부 `touch_giveback_sustained`** 를 고르고 셀은 6종으로 흩어진다.
     그 유형은 표본이 얇아(79/34) 원래 두께 감사에서 기각됐고, 사람이 `touch_mfe/8/2.0`
     (348/211)으로 갈아탄 것이 현재 배포본이다.
  2. **두께 제약(hits>=150) + 확장 격자**로 재선택하니 **`touch_mfe/3/1.5`가 5/8시드(62%)로
     최빈** 승자였다 -- lift **1.9178/1.8706**(배포본 ~1.56), hits 280/159.
     (단일 시드에서 먼저 보였던 `close_at_h/8/1.0`은 1/8로 **불안정**해 탈락.)

⚠️그러나 **격자 lift != 모델 품질**이다. 같은 날 demarker에서 격자는 H=1을 골랐지만 TabPFN
AUC는 부트스트랩 CI 안에서 H=2와 구분되지 않았다(포팅 프로토콜 §5-A).
⇒ 배포를 바꾸기 전에 **모델 단계 대조 + 평가표본 부트스트랩**으로 판정한다.

## 설계

`research_btc_orthogonal_combo_metalabel_tabpfn_20260901.py`(XRP가 재사용하는 모듈)의
`build_fires`를 그대로 쓰고 모듈 상수 `HORIZON`/`K`만 patch해 두 변형을 만든다.
두 변형 모두 **같은 코드·같은 시드**로 통과시킨다(대조군).

  · 평가표본 부트스트랩 B=4,000으로 AUC 95% CI
  · base rate가 다르므로 prevalence 무관 보조지표 `AP / base rate`도 함께

## 판정 (실행 전 고정)

  후보(H=3/K=1.5)의 **AUC CI가 배포본(H=8/K=2.0) 점추정을 VAL·OOS 둘 다에서 배제**해야
  교체를 검토한다. 하나라도 겹치면 **현행 유지**.

⚠️HOLDOUT 미터치 -- XRP orthogonal의 reserved holdout(0.5599)은 H=8/K=2.0으로 이미 소진.
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
    "orthomod", ROOT / "scripts/research_btc_orthogonal_combo_metalabel_tabpfn_20260901.py")
_o = importlib.util.module_from_spec(_S)
_S.loader.exec_module(_o)

CAND_CSV = (ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903"
            / "xrp_5m_evidence_signal_candidates_tier0.csv")
EXPECTED_ROWS = 272_490
OUT = ROOT / "data/research/xrp_orthogonal_h3k15_vs_h8k20_tabpfn_20260903.json"

VARIANTS = {"deployed_H8_K2.0": (8, 2.0), "candidate_H3_K1.5": (3, 1.5)}
DEPLOYED = "deployed_H8_K2.0"
B_BOOT, BOOT_SEED = 4000, 20260903
RECORDED_HOLDOUT_AUC = 0.5599      # 참고용, 재측정하지 않는다


def log(m): print(f"[ortho-tab] {m}", flush=True)


def boot(y, p, rng):
    n = len(y)
    aucs, aps = [], []
    for _ in range(B_BOOT):
        i = rng.integers(0, n, size=n)
        yy = y[i]
        if yy.min() == yy.max():
            continue
        aucs.append(roc_auc_score(yy, p[i]))
        aps.append(average_precision_score(yy, p[i]) / yy.mean())
    a, q = np.array(aucs), np.array(aps)
    return {"auc": float(roc_auc_score(y, p)),
            "auc_ci": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
            "ap_over_base": float(average_precision_score(y, p) / y.mean()),
            "ap_over_base_ci": [float(np.percentile(q, 2.5)), float(np.percentile(q, 97.5))],
            "n": int(n), "n_pos": int(y.sum()), "base_rate": float(y.mean())}


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier

    f = pd.read_csv(CAND_CSV)
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
    for fn in ("add_missing_features", "add_derived_features"):
        g = getattr(_o, fn, None)
        if g is not None:
            f = g(f)
    if abs(len(f) - EXPECTED_ROWS) > 200:
        raise RuntimeError(f"행수 {len(f):,} != XRP 기대치 {EXPECTED_ROWS:,} -- 다른 자산 데이터")
    log(f"XRP 프레임 {len(f):,}행 (자산 가드 통과) | 부트스트랩 B={B_BOOT:,}")
    log(f"⚠️HOLDOUT 미터치 (기록값 {RECORDED_HOLDOUT_AUC}, H=8/K=2.0으로 소진)")

    cols = [c for c in _o.FEATURE_COLUMNS]
    rng = np.random.default_rng(BOOT_SEED)
    rep = {"variants": {k: {"horizon": v[0], "k": v[1]} for k, v in VARIANTS.items()},
           "deployed": DEPLOYED, "B_boot": B_BOOT, "holdout_touched": False,
           "recorded_holdout_auc": RECORDED_HOLDOUT_AUC, "results": {}}

    save_h, save_k = _o.HORIZON, _o.K
    try:
        for tag, (h, k) in VARIANTS.items():
            _o.HORIZON, _o.K = h, k
            fires = _o.build_fires(f)
            # ⚠️tz 규약: orthogonal_combo 모듈은 **tz-aware**를 쓴다(frozen-contexts의
            # TZ_AWARE={"orthogonal_combo": True}). 분할 상수와 tz를 맞춘다 -- 벗기면 터진다.
            ts = pd.to_datetime(fires["timestamp"])
            want_aware = getattr(_o.VAL_START, "tzinfo", None) is not None
            if want_aware and ts.dt.tz is None:
                ts = ts.dt.tz_localize("UTC")
            elif not want_aware and ts.dt.tz is not None:
                ts = ts.dt.tz_localize(None)
            fires = fires.assign(timestamp=ts)
            c = [x for x in cols if x in fires.columns]
            tr = fires.loc[fires["timestamp"] < _o.VAL_START].reset_index(drop=True)
            sp = {"val": fires.loc[(fires["timestamp"] >= _o.VAL_START)
                                   & (fires["timestamp"] < _o.OOS_START)].reset_index(drop=True),
                  "oos": fires.loc[(fires["timestamp"] >= _o.OOS_START)
                                   & (fires["timestamp"] < _o.HOLDOUT_START)].reset_index(drop=True)}
            log("")
            log(f"=== {tag}  (H={h} K={k}){'  ⭐배포' if tag == DEPLOYED else ''} ===")
            log(f"  fires {len(fires):,} | TRAIN {len(tr):,} (hit {tr['hit'].mean():.4f})")
            res = {"n_fires": int(len(fires)), "n_train": int(len(tr)),
                   "train_base_rate": float(tr["hit"].mean())}
            for s, ev in sp.items():
                clf = TabPFNClassifier(device="cuda", random_state=_o.SEEDS[0]
                                       if hasattr(_o, "SEEDS") else 20260829)
                clf.fit(tr[c], tr["hit"].to_numpy().astype(int))
                p = clf.predict_proba(ev[c])[:, 1]
                y = ev["hit"].to_numpy().astype(int)
                m = boot(y, p, rng)
                res[s] = m
                log(f"  {s.upper():<4} n={m['n']:<4} 양성={m['n_pos']:<4} (base {m['base_rate']:.4f})")
                log(f"       AUC {m['auc']:.4f}  95%CI [{m['auc_ci'][0]:.4f}, {m['auc_ci'][1]:.4f}]"
                    f"  폭 {m['auc_ci'][1]-m['auc_ci'][0]:.4f}")
                log(f"       AP/base {m['ap_over_base']:.3f}x")
            rep["results"][tag] = res
    finally:
        _o.HORIZON, _o.K = save_h, save_k

    a = rep["results"]["candidate_H3_K1.5"]
    b = rep["results"][DEPLOYED]
    log("")
    log("=" * 72)
    log("판정 (후보 AUC CI가 배포본 점추정을 VAL·OOS 둘 다에서 배제해야 교체 검토)")
    log("=" * 72)
    v, allwin = {}, True
    for s in ("val", "oos"):
        lo, hi = a[s]["auc_ci"]
        excl = b[s]["auc"] < lo
        allwin &= excl
        v[s] = {"cand_auc": a[s]["auc"], "cand_ci": [lo, hi], "dep_auc": b[s]["auc"],
                "dep_below_cand_ci": bool(excl),
                "cand_ap": a[s]["ap_over_base"], "dep_ap": b[s]["ap_over_base"]}
        log(f"  {s.upper():<4} 후보 {a[s]['auc']:.4f} [{lo:.4f}, {hi:.4f}]   배포 {b[s]['auc']:.4f}   "
            f"{'✅배제' if excl else '❌겹침'}")
    log("")
    log(f"  AP/base  VAL 후보 {a['val']['ap_over_base']:.3f} vs 배포 {b['val']['ap_over_base']:.3f}"
        f"  |  OOS 후보 {a['oos']['ap_over_base']:.3f} vs 배포 {b['oos']['ap_over_base']:.3f}")
    log(f"  base rate 후보 VAL {a['val']['base_rate']:.4f}/OOS {a['oos']['base_rate']:.4f}"
        f"  vs 배포 VAL {b['val']['base_rate']:.4f}/OOS {b['oos']['base_rate']:.4f}")
    log("")
    log(f"⇒ {'⚠️**교체 검토 가능**' if allwin else '✅**현행 H=8/K=2.0 유지** -- 표본 불확실성 안에서 구분되지 않는다'}")
    rep["verdict"] = {"candidate_beats_deployed_on_both": bool(allwin), "by_split": v}
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
