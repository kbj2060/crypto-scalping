#!/usr/bin/env python3
"""GBM 프록시로 내린 판정들을 **TabPFN 실측**으로 재확인 (1/2: T1~T4).

## 왜

V자반등 라벨 재설계 과정의 판정 다수가 `HistGradientBoostingClassifier` 프록시로 내려졌다
(5시드가 필요해 TabPFN은 비용이 안 맞았다). 실제 서빙 모델은 TabPFN이므로 프록시 판정이
그대로 성립하는지 확인한다.

## ⚠️ 프록시→TabPFN 전환이 비교 자체를 바꾸는 지점 (이 재확인의 핵심)

GBM은 학습셋 **전체**를 쓴다. TabPFN은 in-context learner라 **컨텍스트 상한(18,000행)**이
있고, 그걸 넘는 학습셋은 부분표집된다 — 이게 라이브의 실제 동작이다. 따라서:

  - GBM에서 "표본이 커서 이겼다"로 해석됐던 판정은 TabPFN에서 **자동으로 공정한 재대결**이 된다.
    특히 T3(사건당 1봉 샘플링)은 GBM에서 S0가 182,969행 대 17,710행으로 이겼는데, TabPFN에서는
    양쪽 다 ~18k로 잘려 **표본크기 우위가 사라진다**. 문서 9-7이 "TabPFN 단계의 실제 질문"이라며
    미착수로 남긴 것이 정확히 이것이다.
  - 반대로 T1(모집단 불일치)은 A/B/B_sub 세 학습셋이 전부 ~18k로 수렴해 **표본크기 교란이
    완전히 제거된 순수 모집단 비교**가 된다.

즉 이건 "같은 실험 다시 돌리기"가 아니라 **다른 축이 통제된 재실험**이다. 수치가 달라지면
모델 탓인지 컨텍스트 상한 탓인지 구분해서 읽어야 한다 -- 그래서 각 config의 실제 사용
컨텍스트 행수(`context_used`)를 전부 기록한다.

## 방법

기존 스크립트의 **데이터 빌더와 평가 절차를 그대로 재사용**하고 분류기만 교체한다(재구현
금지 -- 라벨/피쳐/스플릿 로직을 다시 쓰면 그 차이가 결과에 섞인다).
  - T1(9-3), T3(9-7): 빌더가 함수라 직접 import해 루프를 새로 쓴다(평가셋 부분표집 통제 목적).
  - T2(9-4), T4(9-9): 빌더가 main() 안에 인라인이라 **분류기 심볼만 몽키패치**하고 그쪽
    main()을 그대로 호출한다. 결과 JSON 경로도 별도로 돌린다.

⚠️T2의 순열중요도는 `predict_proba`를 피쳐×반복 횟수만큼 부르는데 TabPFN에서는 비용이
과도해 **비활성화**한다(판정 근거는 AUC delta이므로 결론에 영향 없음 -- report에 명시).

⚠️전 구간 TRAIN+VAL만 사용(빌더가 VAL_END에서 자른다). OOS/HOLDOUT 미터치, 라이브 코드 변경 없음.

Run on the server (GPU) via handoff:
  handoff.sh launch server <job> -- python scripts/research_eth_v_rebound_gbm_proxy_tabpfn_reverify_20260902.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CONTEXT_N = 18000        # 라이브 연구 설정과 동일 (TabPFN 컨텍스트 상한)
EVAL_N = 15000           # 평가셋 부분표집 상한 (repo 선례: TabPFN 3시드 VAL 전체봉 15,000행)
EVAL_SEED = 20260902     # 평가 부분표집은 config 간 **동일**해야 하므로 모델 시드와 분리
SEEDS = [20260829, 141592, 271828]
PREDICT_CHUNK = 20000

OUT_JSON = ROOT / "data/research/eth_v_rebound_gbm_proxy_tabpfn_reverify_20260902/report_t1_t4.json"

# GBM 프록시가 보고한 값 (재확인 대상)
GBM_REF = {
    "T1": {"A_train_on_candidates": {"cand": 0.8090, "all": 0.5287},
           "B_train_on_all_bars": {"cand": 0.6933, "all": 0.6953},
           "B_sub_all_bars_subsampled": {"cand": 0.6438, "all": 0.6676}},
    "T2": {"F1_tier0_only": 0.6934, "F2_plus_triggers_raw": 0.6928,
           "F3_plus_triggers_dedup": 0.6933, "F4_plus_local_extreme_LOOKAHEAD": 0.7228},
    "T3": {"S0_all_bars": {"all": 0.6934, "thin": 0.6982},
           "S1_pos1perevent_negthin": {"all": 0.5960, "thin": 0.5962},
           "S1b_pos1perevent_negbaserate": {"all": 0.5931, "thin": 0.5906},
           "S1c_pos1perevent_RANDOMbar_negthin": {"all": 0.6552, "thin": 0.6582},
           "S2_uniform_thin_both": {"all": 0.6599, "thin": 0.6653}},
    "T4": {"floor_0": 0.6891, "floor_10": 0.6866, "floor_20": 0.6970,
           "floor_30": 0.7104, "floor_40": 0.7123},
}


def log(msg: str) -> None:
    print(f"[reverify] {msg}", flush=True)


class TabPFNShim:
    """`HistGradientBoostingClassifier` 자리에 그대로 끼워지는 TabPFN 어댑터.

    컨텍스트 상한을 넘는 학습셋은 무작위 부분표집한다 -- 라이브의 실제 동작이며, 이 재확인이
    측정하려는 바로 그 제약이다. 실제 사용 행수는 `context_used_`에 남긴다.
    """

    def __init__(self, random_state: int = 0, **_ignored):
        self.random_state = int(random_state)
        self._clf = None
        self.context_used_ = None
        self.context_capped_ = False

    def fit(self, X, y):
        from tabpfn import TabPFNClassifier
        X = pd.DataFrame(X)
        y = np.asarray(y)
        if len(X) > CONTEXT_N:
            rng = np.random.default_rng(self.random_state)
            idx = np.sort(rng.choice(len(X), size=CONTEXT_N, replace=False))
            X, y = X.iloc[idx], y[idx]
            self.context_capped_ = True
        self.context_used_ = int(len(X))
        self._clf = TabPFNClassifier(device="cuda", random_state=self.random_state,
                                     ignore_pretraining_limits=True)
        self._clf.fit(X, y)
        return self

    def predict_proba(self, X):
        X = pd.DataFrame(X)
        if len(X) <= PREDICT_CHUNK:
            return self._clf.predict_proba(X)
        return np.vstack([self._clf.predict_proba(X.iloc[i:i + PREDICT_CHUNK])
                          for i in range(0, len(X), PREDICT_CHUNK)])


def subsample(df: pd.DataFrame, n: int = EVAL_N, seed: int = EVAL_SEED) -> pd.DataFrame:
    """평가셋 부분표집. config 간 **같은 행**을 보도록 고정 시드를 쓴다."""
    if len(df) <= n:
        return df
    rng = np.random.default_rng(seed)
    return df.iloc[np.sort(rng.choice(len(df), size=n, replace=False))]


def save(report: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))


def delta_line(tag: str, tabpfn: float, gbm: float, std: float | None = None) -> str:
    d = tabpfn - gbm
    flag = "  " if abs(d) < 0.01 else ("↑" if d > 0 else "↓")
    s = f" ±{std:.4f}" if std is not None else ""
    return f"  {tag:38s} TabPFN {tabpfn:.4f}{s}   GBM {gbm:.4f}   delta {d:+.4f}{flag}"


def main() -> int:
    t0 = time.time()
    import torch
    from sklearn.metrics import roc_auc_score
    log(f"cuda: {torch.cuda.is_available()}")
    log(f"컨텍스트 상한 {CONTEXT_N:,} / 평가 부분표집 {EVAL_N:,} / 시드 {SEEDS}")

    report = {"signal": "v_rebound_gbm_proxy_tabpfn_reverify", "asset": "ETHUSDT",
              "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
              "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
              "scope": {"model": "TabPFN (재확인 대상: HistGradientBoosting 프록시)",
                        "context_cap": CONTEXT_N, "eval_subsample": EVAL_N, "seeds": SEEDS,
                        "splits": "TRAIN < 2025-09-01 / VAL 2025-09~12 (빌더가 VAL_END에서 절단)",
                        "oos_touched": False, "holdout_touched": False,
                        "live_code_changed": False,
                        "caveat": ("GBM은 학습셋 전체를 쓰고 TabPFN은 컨텍스트 18,000행이 상한이다. "
                                   "따라서 표본크기가 컸던 config의 우위는 TabPFN에서 구조적으로 "
                                   "사라진다 -- 수치 차이를 '모델 차이'로만 읽으면 안 된다. "
                                   "각 config의 context_used를 함께 볼 것.")},
              "gbm_reference": GBM_REF, "results": {}}

    # =====================================================================
    # T1 (9-3) 모집단 불일치 -- A(후보풀 학습)의 전체봉 붕괴가 TabPFN에서도 나오는가
    # =====================================================================
    log("")
    log("=== T1 (9-3) 학습 모집단 불일치 -- 후보풀 학습 vs 전체봉 학습 ===")
    _feas = _load("feas_reverify", "scripts/research_eth_v_rebound_every_bar_scoring_feasibility_20260901.py")
    FC = _feas.FEATURE_COLUMNS
    log("building every-bar long frame (T1/T3 공용) ...")
    long = _feas.build_long_frame()
    assert long["timestamp"].max() < _feas.VAL_END, "OOS/HOLDOUT 누출"
    labeled = long.loc[long["label"].notna()].dropna(subset=FC).reset_index(drop=True)
    tr = labeled.loc[labeled["split"] == "TRAIN"]
    va = labeled.loc[labeled["split"] == "VAL"]
    tr_cand = tr.loc[tr["is_candidate"]]
    va_cand = va.loc[va["is_candidate"]]
    log(f"  TRAIN all={len(tr):,} cand={len(tr_cand):,} | VAL all={len(va):,} cand={len(va_cand):,}")

    rng = np.random.default_rng(_feas.SEED)
    tr_sub = tr.iloc[np.sort(rng.choice(len(tr), size=len(tr_cand), replace=False))]
    t1_sets = {"A_train_on_candidates": tr_cand, "B_train_on_all_bars": tr,
               "B_sub_all_bars_subsampled": tr_sub}
    t1_evals = {"cand": subsample(va_cand), "all": subsample(va)}
    log(f"  평가셋(부분표집): 후보봉 {len(t1_evals['cand']):,}행 "
        f"(base {t1_evals['cand']['label'].mean():.4f}) / "
        f"전체봉 {len(t1_evals['all']):,}행 (base {t1_evals['all']['label'].mean():.4f})")

    t1 = {}
    for name, tset in t1_sets.items():
        per = {k: [] for k in t1_evals}
        ctx_used = None
        for sd in SEEDS:
            m = TabPFNShim(random_state=sd).fit(tset[FC], tset["label"].to_numpy())
            ctx_used = m.context_used_
            for ek, ed in t1_evals.items():
                per[ek].append(float(roc_auc_score(ed["label"].to_numpy(),
                                                   m.predict_proba(ed[FC])[:, 1])))
        t1[name] = {"train_n": int(len(tset)), "context_used": ctx_used,
                    **{f"auc_{k}": {"mean": round(float(np.mean(v)), 4),
                                    "std": round(float(np.std(v)), 4)} for k, v in per.items()}}
        log(f"  {name}  학습셋 {len(tset):,}행 -> 컨텍스트 {ctx_used:,}행")
        for ek in t1_evals:
            log(delta_line(f"    eval {ek}", np.mean(per[ek]), GBM_REF["T1"][name][ek],
                           float(np.std(per[ek]))))
    report["results"]["T1_population_mismatch"] = t1
    save(report)

    # =====================================================================
    # T3 (9-7) 사건당 1봉 샘플링 -- TabPFN에서는 S0의 표본크기 우위가 사라진다
    # =====================================================================
    log("")
    log("=== T3 (9-7) 사건 샘플링 -- ⭐TabPFN 컨텍스트 상한으로 S0의 표본우위가 제거된 재대결 ===")
    _es = _load("evsamp_reverify",
                "scripts/research_eth_v_rebound_every_bar_event_sampled_training_20260901.py")
    lab2 = _es.add_sampling_masks(labeled.copy(), _es.GAP)
    tr2 = lab2.loc[lab2["split"] == "TRAIN"]
    va2 = lab2.loc[lab2["split"] == "VAL"]
    t3_sets = {"S0_all_bars": tr2,
               "S1_pos1perevent_negthin": tr2.loc[tr2["s1_mask"]],
               "S1b_pos1perevent_negbaserate": tr2.loc[tr2["s1b_mask"]],
               "S1c_pos1perevent_RANDOMbar_negthin": tr2.loc[tr2["s1c_mask"]],
               "S2_uniform_thin_both": tr2.loc[tr2["is_uniform_thin"]]}
    t3_evals = {"all": subsample(va2), "thin": subsample(va2.loc[va2["is_uniform_thin"]])}
    log(f"  평가셋: 전체봉 {len(t3_evals['all']):,}행 / 균등솎기 {len(t3_evals['thin']):,}행")

    t3 = {}
    for name, tset in t3_sets.items():
        per = {k: [] for k in t3_evals}
        ctx_used = capped = None
        for sd in SEEDS:
            m = TabPFNShim(random_state=sd).fit(tset[FC], tset["label"].to_numpy())
            ctx_used, capped = m.context_used_, m.context_capped_
            for ek, ed in t3_evals.items():
                per[ek].append(float(roc_auc_score(ed["label"].to_numpy(),
                                                   m.predict_proba(ed[FC])[:, 1])))
        t3[name] = {"train_n": int(len(tset)), "context_used": ctx_used,
                    "context_capped": bool(capped),
                    "base_rate": round(float(tset["label"].mean()), 4),
                    **{f"auc_{k}": {"mean": round(float(np.mean(v)), 4),
                                    "std": round(float(np.std(v)), 4)} for k, v in per.items()}}
        log(f"  {name}  학습셋 {len(tset):,}행 -> 컨텍스트 {ctx_used:,}행"
            f"{'  ⚠️상한에 잘림' if capped else ''}  base {tset['label'].mean():.4f}")
        for ek in t3_evals:
            log(delta_line(f"    eval {ek}", np.mean(per[ek]), GBM_REF["T3"][name][ek],
                           float(np.std(per[ek]))))
    report["results"]["T3_event_sampling"] = t3
    save(report)
    del long, labeled, lab2, tr, va, tr_cand, va_cand, tr2, va2

    # =====================================================================
    # T2 (9-4) 8트리거 피쳐화 -- 분류기만 교체하고 원 스크립트 main()을 그대로 호출
    # =====================================================================
    log("")
    log("=== T2 (9-4) 8트리거 피쳐화 -- 원 스크립트 main() 몽키패치 실행 ===")
    _tf = _load("trigfeat_reverify",
                "scripts/research_eth_v_rebound_every_bar_trigger_features_20260901.py")
    _tf.HistGradientBoostingClassifier = TabPFNShim
    _tf.OUT_JSON = OUT_JSON.parent / "t2_trigger_features_tabpfn.json"
    _tf.shuffle_importance = lambda *a, **k: {}      # TabPFN에서는 비용 과도 -- 판정근거는 AUC delta
    log("  ⚠️순열중요도 비활성화(TabPFN 비용) -- AUC delta만으로 판정")
    try:
        _tf.main()
        t2 = json.loads(_tf.OUT_JSON.read_text()).get("results", {})
        log("  --- TabPFN vs GBM ---")
        for k, ref in GBM_REF["T2"].items():
            if k in t2 and t2[k].get("val_all_bars_auc") is not None:
                log(delta_line(k, t2[k]["val_all_bars_auc"], ref))
        report["results"]["T2_trigger_features"] = t2
    except Exception as e:                            # noqa: BLE001
        log(f"  ⚠️T2 실패: {type(e).__name__}: {e}")
        report["results"]["T2_trigger_features"] = {"error": f"{type(e).__name__}: {e}"}
    save(report)

    # =====================================================================
    # T4 (9-9) 절대 bp 하한 -- 동일 방식
    # =====================================================================
    log("")
    log("=== T4 (9-9) 절대 bp 하한 스윕 -- 원 스크립트 main() 몽키패치 실행 ===")
    _fl = _load("floor_reverify",
                "scripts/research_eth_v_rebound_absolute_bp_floor_sweep_20260901.py")
    _fl.HistGradientBoostingClassifier = TabPFNShim
    _fl.SEEDS = SEEDS
    _fl.OUT_JSON = OUT_JSON.parent / "t4_bp_floor_tabpfn.json"
    try:
        _fl.main()
        t4 = json.loads(_fl.OUT_JSON.read_text()).get("results", {})
        log("  --- TabPFN vs GBM (고정타겟 AUC) ---")
        for k, ref in GBM_REF["T4"].items():
            if k in t4:
                e = t4[k]["auc_on_FIXED_ref_target"]
                log(delta_line(k, e["mean"], ref, e.get("std")))
        report["results"]["T4_bp_floor"] = t4
    except Exception as e:                            # noqa: BLE001
        log(f"  ⚠️T4 실패: {type(e).__name__}: {e}")
        report["results"]["T4_bp_floor"] = {"error": f"{type(e).__name__}: {e}"}

    report["runtime_sec"] = round(time.time() - t0, 1)
    save(report)
    log("")
    log(f"report saved -> {OUT_JSON}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
