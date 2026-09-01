#!/usr/bin/env python3
"""GBM 프록시로 내린 판정들을 **TabPFN 실측**으로 재확인 (2/2: T5~T6).

1편(`..._tabpfn_reverify_20260902.py`)이 T1~T4(모집단 불일치 / 8트리거 피쳐화 / 사건 샘플링 /
bp 하한)를 다뤘다. 여기서는 남은 두 건을 같은 방식으로 처리한다.

  T5 (15절) 154피쳐 확장 -- GBM: AUC +0.017이지만 미포착 사건 포착률은 제자리(순증분 0)
  T6 (11절) 재학습용 후보풀 3종(A/B/C) 3x3 행렬 -- GBM: A 유지 권고, C(ATR필터) 기각

## 방법 -- 1편과 동일

기존 스크립트의 **데이터 빌더·평가 절차를 그대로 쓰고 분류기만 교체**한다. 두 스크립트 다
호출빈도 일치·고정 평가타깃 같은 함정 방어가 이미 내장돼 있으므로 재구현하면 그 방어가 사라진다.

  - T6은 `HistGradientBoostingClassifier`가 **모듈 최상단** import라 모듈 속성만 바꾸면 된다.
  - T5는 그게 **main() 안의 지역 import**라 모듈 속성 교체가 안 먹는다. 지역 import는 호출
    시점에 `sklearn.ensemble`의 속성을 다시 읽으므로 **그 패키지 속성을 임시 교체**하고
    finally에서 되돌린다.

## ⚠️ 1편과 같은 구조적 주의

TabPFN은 컨텍스트 18,000행이 상한이라 학습셋이 그보다 크면 부분표집된다. T6의 C풀(7,796행)은
상한 아래라 그대로 쓰이지만 A/B풀은 잘린다 -- GBM에서 "C는 학습셋이 너무 줄어서 진다"고 본
판정이 TabPFN에서는 **표본크기 격차가 좁혀진 채로** 다시 매겨진다. 각 config의 context_used를
같이 볼 것.

⚠️T5는 원 실험 설계상 VAL/OOS를 모두 평가한다(OOS는 이미 노출된 구간, 재최적화 아님).
HOLDOUT 미터치. 라이브 코드 변경 없음.

Run on the server (GPU) via handoff:
  handoff.sh launch server <job> -- python scripts/research_eth_v_rebound_gbm_proxy_tabpfn_reverify2_20260902.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 1편의 TabPFN 어댑터를 그대로 재사용한다(컨텍스트 상한 처리·청크 예측 로직 복제 금지).
_rv = _load("reverify_t1t4", "scripts/research_eth_v_rebound_gbm_proxy_tabpfn_reverify_20260902.py")
TabPFNShim = _rv.TabPFNShim
SEEDS = _rv.SEEDS

OUT_JSON = ROOT / "data/research/eth_v_rebound_gbm_proxy_tabpfn_reverify_20260902/report_t5_t6.json"

GBM_REF = {
    "T5_154feature": {"note": "F0 Tier0 23 대비 F2(+통과피쳐)가 AUC +0.017이나 "
                              "미포착 포착률은 호출빈도 일치 시 순증분 0"},
    "T6_pool_variants": {"train_A": {"eval_A": 0.8093, "eval_B": 0.8014, "eval_C": 0.7696},
                         "train_B": {"eval_A": 0.8005, "eval_B": 0.8130, "eval_C": 0.7890},
                         "train_C": {"eval_A": 0.7907, "eval_B": 0.8033, "eval_C": 0.7884}},
}


def log(msg: str) -> None:
    print(f"[reverify2] {msg}", flush=True)


def save(report: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))


def main() -> int:
    t0 = time.time()
    import torch
    log(f"cuda: {torch.cuda.is_available()}")
    log(f"컨텍스트 상한 {_rv.CONTEXT_N:,} / 시드 {SEEDS}")

    report = {"signal": "v_rebound_gbm_proxy_tabpfn_reverify_part2", "asset": "ETHUSDT",
              "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
              "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
              "scope": {"model": "TabPFN (재확인 대상: HistGradientBoosting 프록시)",
                        "context_cap": _rv.CONTEXT_N, "seeds": SEEDS,
                        "holdout_touched": False, "live_code_changed": False,
                        "caveat": ("TabPFN은 컨텍스트 18,000행이 상한이라 큰 학습셋의 표본우위가 "
                                   "사라진다. T6의 C풀(7,796행)은 상한 아래라 그대로 쓰인다 -- "
                                   "'C는 학습셋이 작아서 졌다'는 GBM 판정이 재대결된다.")},
              "gbm_reference": GBM_REF, "results": {}}

    # =====================================================================
    # T6 (11절) 후보풀 3종 3x3 -- 모듈 최상단 import라 속성 교체로 충분
    # =====================================================================
    log("")
    log("=== T6 (11절) 재학습용 후보풀 A/B/C 3x3 행렬 ===")
    _pv = _load("poolvar_reverify",
                "scripts/research_eth_v_rebound_pool_variant_retrain_comparison_20260901.py")
    _pv.HistGradientBoostingClassifier = TabPFNShim
    _pv.SEEDS = SEEDS
    _pv.OUT_JSON = OUT_JSON.parent / "t6_pool_variants_tabpfn.json"
    try:
        _pv.main()
        report["results"]["T6_pool_variants"] = json.loads(_pv.OUT_JSON.read_text())
        log("  ✅T6 완료")
    except Exception as e:                            # noqa: BLE001
        log(f"  ⚠️T6 실패: {type(e).__name__}: {e}")
        report["results"]["T6_pool_variants"] = {"error": f"{type(e).__name__}: {e}"}
    save(report)

    # =====================================================================
    # T5 (15절) 154피쳐 -- main() 안 지역 import라 패키지 속성을 임시 교체
    # =====================================================================
    log("")
    log("=== T5 (15절) 154피쳐 확장 ===")
    _f154 = _load("f154_reverify",
                  "scripts/research_eth_v_rebound_154feature_uncovered_capture_20260901.py")
    _f154.SEEDS = SEEDS
    _f154.OUT_JSON = OUT_JSON.parent / "t5_154feature_tabpfn.json"
    import sklearn.ensemble as _skl
    _orig = _skl.HistGradientBoostingClassifier
    _skl.HistGradientBoostingClassifier = TabPFNShim
    try:
        _f154.main()
        report["results"]["T5_154feature"] = json.loads(_f154.OUT_JSON.read_text())
        log("  ✅T5 완료")
    except Exception as e:                            # noqa: BLE001
        log(f"  ⚠️T5 실패: {type(e).__name__}: {e}")
        report["results"]["T5_154feature"] = {"error": f"{type(e).__name__}: {e}"}
    finally:
        _skl.HistGradientBoostingClassifier = _orig

    report["runtime_sec"] = round(time.time() - t0, 1)
    save(report)
    log("")
    log(f"report saved -> {OUT_JSON}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
