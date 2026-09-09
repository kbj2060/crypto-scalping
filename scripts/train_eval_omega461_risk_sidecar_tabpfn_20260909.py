"""리스크 사이드카 회귀기를 HGB → TabPFN 으로 교체하는 래퍼.

세션 초반 교체 순위표(`docs/eth_omega4_6_1_accuracy_research_ideas_20260811.md` 계열 논의)의
**2번 항목**. 1번(quality_head → TabPFN 메타라벨)은 실행 후 겹침 보정에서 실패했다
(`docs/experiments/...` 참조: 명목 AUC 0.68 이었으나 블록 부트스트랩 CI 5셀 전부 0 포함,
비겹침 정밀도 −17.5pp).

왜 이 자리가 TabPFN 에 유리한가 — 실측 근거
------------------------------------------
사이드카 회귀기의 학습 표본은 **트레이드 단위 75건**이다(실측: `risk_label.rows=75`,
long 31 / short 44). `--side-split-model` 이므로 **한 모델이 31~44 표본 × 29피쳐**로 적합된다.
  · 현행 HGB 는 `max_iter=220, max_leaf_nodes=15` — 31 표본에 트리 220그루는 과적합 구간이다.
  · TabPFN 은 소표본 테이블(n<1000)을 위해 설계된 in-context learner다.
  · 1번 축을 무너뜨린 "컨텍스트 상한 18,000행에 학습셋이 잘린다"는 문제가 **여기엔 없다**
    (75 ≪ 18,000).
  · 겹침 표본 함정도 약하다 — 트레이드 단위라 애초에 겹치지 않는다. 대신 표본이 작다는
    다른 한계가 있고 그건 보고에 그대로 표시한다.
  · 사이징 축이라 이 저장소가 반복 실패한 "재분류" 교훈에 걸리지 않는다.

TabPFN 설계 특징을 실제로 쓴다 (드롭인 교체가 아니다)
-----------------------------------------------------
1. **`n_estimators` (기본 32)** — TabPFN 은 피쳐/데이터 순열 앙상블로 분산을 줄인다. n=31 에
   29피쳐면 순열 커버리지가 곧 성능이므로 이 값이 핵심 손잡이다(저장소 관례도 "추정기 수가 곧
   피쳐 커버리지"로 기록). 기본값(4 수준)으로 두면 TabPFN 을 절반만 쓰는 셈이다.
2. **예측 분포 사용** — `predict(output_type="quantiles"|"median"|"mean")`. TabPFNRegressor 는
   점추정이 아니라 **전체 예측 분포**를 준다. 사이드카는 예측 점수를 margin_fraction 으로
   매핑하는 **리스크 사이징** 모듈이므로, 평균 대신 **보수적 하위분위(q25/q10)** 를 쓰면
   하방 인식이 모델 자체에서 나온다 — 지금은 `selection_objective=log_risk` 의 사후 페널티로만
   구현돼 있는 것을 모델의 예측 분포로 직접 얻는 셈이다.
   ⚠️ 이건 의미가 달라지는 변경이므로 **별도 arm** 으로 돌린다(`--tabpfn-output`).
3. 서브샘플링 불필요 — 75행은 컨텍스트 상한(18,000)의 0.4% 다. 1번 축을 무너뜨린 표본 손실이
   여기엔 없다.

무엇을 바꾸나 — 회귀기 심볼 하나
-------------------------------
`_build_risk_model(kind, seed)` 가 단일 팩토리이므로 이것만 몽키패치한다. 나머지(피쳐 29개,
타깃 `trade net_per_notional`, side-split, 매핑 격자, 선택 제약, ATR 배리어)는 **전부 원본 그대로**.
저장소 규율(`feedback_gbm_proxy_fails_when_sample_size_is_the_driver`: "재구현하지 말고 분류기
심볼만 교체한다")을 따른다.

⚠️ sample_weight 처리 — 근사이며 명시한다
----------------------------------------
원본은 `model.fit(X, y, sample_weight=1.0 + clip(-mae*25, 0, 3))` (범위 1~4) 로 MAE 가 나쁜
트레이드를 가중한다. **TabPFNRegressor 는 sample_weight 를 받지 않는다.** 그래서 가중치를
**행 복제**로 근사한다(`round(w)` 회 반복) — 가중 학습의 표준 대체이고, n=31~44 라 복제해도
컨텍스트 상한 근처에도 못 간다. 이는 순수한 "회귀기만 교체"가 아니라 **가중치 근사가 섞인
교체**이므로, 결과 해석 시 이 사실을 함께 읽어야 한다. report 에 `sample_weight_handling`
필드로 기록한다.

사용법: 원본 트레이너의 모든 인자를 그대로 받고 `--model-kind tabpfn` 만 추가로 인식한다.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_eval_omega4_2_risk_sidecar_20260622 as sc  # noqa: E402

_ENV = ROOT / ".env"
if _ENV.exists():
    for _line in _ENV.read_text().splitlines():
        if _line.startswith("TABPFN_TOKEN="):
            os.environ["TABPFN_TOKEN"] = _line.split("=", 1)[1].strip().strip('"')

_orig_build = sc._build_risk_model


class TabPFNRiskRegressor:
    """TabPFNRegressor 를 sklearn 회귀기 인터페이스로 감싼다.

    `fit(X, y, sample_weight=None)` — sample_weight 는 행 복제로 근사한다(위 docstring 참조).
    `predict(X)` — float64 배열 반환.
    """

    def __init__(self, seed: int, device: str | None = None,
                 n_estimators: int = 32, output: str = "mean"):
        self.seed = int(seed)
        self.device = device
        self.n_estimators = int(n_estimators)
        self.output = str(output)          # mean | median | q25 | q10
        self.model = None
        self.n_fit_rows = None
        self.n_rows_after_weight = None

    def fit(self, X, y, sample_weight=None):
        from tabpfn import TabPFNRegressor
        import torch

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        self.n_fit_rows = int(len(Xa))
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float64)
            reps = np.maximum(1, np.rint(w)).astype(int)
            Xa = np.repeat(Xa, reps, axis=0)
            ya = np.repeat(ya, reps, axis=0)
        self.n_rows_after_weight = int(len(Xa))
        self.model = TabPFNRegressor(device=dev, random_state=self.seed,
                                     n_estimators=self.n_estimators)
        self.model.fit(Xa, ya)
        print(f"[tabpfn-sidecar] fit 완료 — 원표본 {self.n_fit_rows}행 → 가중복제 후 "
              f"{self.n_rows_after_weight}행, device={dev}, n_estimators={self.n_estimators}, "
              f"output={self.output}", flush=True)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("TabPFNRiskRegressor: fit before predict")
        Xa = np.asarray(X, dtype=np.float64)
        if self.output in ("mean", "median"):
            return np.asarray(self.model.predict(Xa, output_type=self.output), dtype=np.float64)
        if self.output in ("q25", "q10"):
            q = 0.25 if self.output == "q25" else 0.10
            out = self.model.predict(Xa, output_type="quantiles", quantiles=[q])
            arr = out[0] if isinstance(out, (list, tuple)) else out
            return np.asarray(arr, dtype=np.float64).reshape(-1)
        raise RuntimeError(f"unknown tabpfn output type: {self.output!r}")


_CFG = {"n_estimators": 32, "output": "mean"}


def _patched_build(kind: str, seed: int):
    """이 래퍼는 TabPFN 전용이므로 `kind` 와 무관하게 항상 TabPFN 을 돌려준다.

    ⚠️ 2026-09-09 버그: 원본 argparse 의 `choices` 를 통과시키려고 argv 의
    `--model-kind tabpfn` 을 `hgb` 로 바꿔 넣는데, 예전 구현은 여기서 `kind == "tabpfn"`
    을 검사했다. kind 는 이미 "hgb" 로 바뀐 뒤라 조건이 never-true 였고 **HGB 가 조용히
    학습됐다** — 스모크 결과가 HGB 판과 비트 단위로 동일해서 발견했다. kind 를 믿지 않는다."""
    print(f"[tabpfn-sidecar] _build_risk_model 호출(kind={kind!r}) → TabPFNRiskRegressor 반환",
          flush=True)
    return TabPFNRiskRegressor(seed, n_estimators=_CFG["n_estimators"], output=_CFG["output"])


def main() -> int:
    argv = list(sys.argv[1:])
    # 원본 argparse 의 choices 에 tabpfn 이 없으므로 파서를 통과시키기 위해 hgb 로 바꿔 넣고,
    # 팩토리 몽키패치가 실제 모델을 결정하게 한다. report 에 남는 model_kind 는 아래에서 덮어쓴다.
    # TabPFN 전용 인자를 먼저 떼어낸다(원본 argparse 는 모른다)
    for flag, key, cast in (("--tabpfn-n-estimators", "n_estimators", int),
                            ("--tabpfn-output", "output", str)):
        if flag in argv:
            j = argv.index(flag)
            _CFG[key] = cast(argv[j + 1])
            del argv[j: j + 2]

    use_tabpfn = False
    if "--model-kind" in argv:
        i = argv.index("--model-kind")
        if argv[i + 1] == "tabpfn":
            use_tabpfn = True
            argv[i + 1] = "hgb"
    if not use_tabpfn:
        raise SystemExit("이 래퍼는 --model-kind tabpfn 전용이다. 다른 kind 는 원본 스크립트를 직접 쓸 것")

    sc._build_risk_model = _patched_build
    print("[tabpfn-sidecar] _build_risk_model 몽키패치 — side-split 모델이 TabPFNRegressor 로 적합된다",
          flush=True)
    print(f"[tabpfn-sidecar] n_estimators={_CFG['n_estimators']} · output={_CFG['output']} "
          f"(mean=드롭인 / q25·q10=예측분포 하위분위 = 하방인식 사이징)", flush=True)
    print("[tabpfn-sidecar] sample_weight 는 행 복제(round(w))로 근사한다 — 순수 회귀기 교체가 아님",
          flush=True)
    sys.argv = [sys.argv[0], *argv]
    rc = sc.main()

    # report/pkl 의 model_kind 를 사실대로 고쳐 쓴다(hgb 로 남으면 나중에 오독된다)
    import json
    import pickle
    try:
        suffix = argv[argv.index("--out-suffix") + 1]
        cand = sorted((sc.ROOT / "tmp/causal_regen_20260516").glob(f"{sc.MODEL_ID}_{suffix}"))
        if not cand:
            raise RuntimeError(f"산출물 디렉토리를 찾지 못함: {sc.MODEL_ID}_{suffix}")
        for d in cand:
            p = d / "report.json"
            if p.exists():
                r = json.loads(p.read_text())
                r.setdefault("risk_model", {})["model_kind"] = "tabpfn"
                r["risk_model"]["sample_weight_handling"] = "row_replication_round_w (TabPFN has no sample_weight)"
                r["risk_model"]["tabpfn_n_estimators"] = _CFG["n_estimators"]
                r["risk_model"]["tabpfn_output_type"] = _CFG["output"]
                p.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
            q = d / "risk_sidecar.pkl"
            if q.exists():
                with open(q, "rb") as f:
                    pk = pickle.load(f)
                pk["model_kind"] = "tabpfn"
                pk["sample_weight_handling"] = "row_replication_round_w"
                pk["tabpfn_n_estimators"] = _CFG["n_estimators"]
                pk["tabpfn_output_type"] = _CFG["output"]
                with open(q, "wb") as f:
                    pickle.dump(pk, f)
        print(f"[tabpfn-sidecar] model_kind 를 'tabpfn' 으로 기록 완료 ({len(cand)}개 디렉토리)", flush=True)
    except Exception as e:  # 기록 실패는 학습 결과를 무효화하지 않지만 조용히 넘기지 않는다
        print(f"[tabpfn-sidecar] ⚠️ model_kind 기록 실패: {type(e).__name__}: {e}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
