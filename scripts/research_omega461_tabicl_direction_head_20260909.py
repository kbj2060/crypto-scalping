"""TabICL 을 Omega 4.6.1 **direction head** 에 설계 특성대로 적용한다 (Stage 0 게이트).

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계약: docs/model_contracts/omega461_regimegbm_rebuild_contract.md
결정 근거: 2026-09-09 사용자 "tabicl과 tabpfn 의 설계 특징에 맞게 잘 적용해줘" / "병행해줘"

왜 direction head 인가 — TabICL 의 설계 사거리
---------------------------------------------
TabPFN 은 소표본 in-context 학습기다(이 저장소 관례상 컨텍스트 상한 18,000행). 반면 TabICL 은
**대형 테이블(~100k행)** 을 겨냥해 만들어졌다 — column-wise 임베딩으로 각 피쳐의 분포를 먼저
요약하고, 그 위에서 row-wise 어텐션을 돌린 뒤 데이터셋 전체를 컨텍스트로 삼는 2단 구조다.

Omega 4.6.1 안에서 이 사거리에 실제로 들어오는 헤드는 direction head 뿐이다:
  · direction head : 2025 학습 프레임 전체(수만~10만행) × 102 base col × 3클래스  ← TabICL 영역
  · risk sidecar   : 학습 표본 75건(롱31/숏44)                                    ← TabPFN 영역(트랙 ①)
사이드카에 TabICL 을 쓰는 건 설계 오용이고, direction head 에 TabPFN 을 쓰면 컨텍스트 상한 때문에
학습행의 80% 이상을 버려야 한다. 그래서 두 트랙을 나눈다.

설계에 맞춘 3가지 결정
---------------------
1) **표준화하지 않는다.** TabICL 의 column-wise 임베딩은 분포 인식형이고 내부 `norm_methods`
   (none/power 등)를 앙상블 축으로 쓴다. 밖에서 z-score 를 먹여 넣으면 그 축을 죽인다.
   그래서 `parent._base_input` 의 **원시 102열**을 그대로 넣는다(`_standardize_apply` 미사용).
2) **MoE 라우팅을 sample_weight 로 못 넘긴다.** 현행 TabM 은 레짐 확률을 행별 가중치로 써서
   전문가 3개를 소프트 분할한다. TabICL 은 sample_weight 를 받지 않으므로 두 팔로 나눈다:
     · `global` — 전문가 분할 없음. 레짐 6열이 이미 피쳐 안에 있고, in-context 학습기는
                  컨텍스트 전체를 조건으로 삼으므로 레짐별 거동을 문맥에서 학습한다.
                  **이쪽이 TabICL 설계에 맞는 형태다**(MoE 는 작은 MLP 의 표현력 한계를
                  우회하려던 장치인데, TabICL 엔 그 제약이 없다).
     · `moe`    — argmax(레짐)으로 컨텍스트를 하드 분할한 현행 구조의 충실한 이식.
3) **컨텍스트 사다리.** TabICL 의 주장 자체가 "컨텍스트가 클수록 좋다"이므로 그 주장을
   이 데이터에서 직접 잰다(4k→전량). 동시에 이 8GB 카드(라이브 스택과 공유, 여유 ~2.8GB)에서
   어느 사다리 칸까지 실제로 올라갈 수 있는지가 배포 가능성의 상한이다. OOM 은 실패로 기록만
   하고 다음 칸으로 넘어간다 — 조용히 축소하지 않는다.

사전 등록 킬 게이트 (VAL 에서만 판정, `validation_only`)
-------------------------------------------------------
K1  대조군 우위 : TabICL 최선 셀의 VAL balanced accuracy **중앙값**이 **자명 대조군 4종 전부**
                  (always_0/1/2 + stratified prior) 보다 높아야 한다.
                  ※ 이 저장소에서 라벨×모델 조합 7건을 닫은 게 정확히 이 대조군이다.
K2  현직 우위   : 같은 VAL 행에서 현직 TabM(Phase 2 balnobb 번들, 라우팅 적용) 대비
                  짝지은 델타의 **중앙값이 양수**여야 한다.
K2b Seed-Diversity: 그 델타의 **부호가 5시드 전부에서 같아야** 한다(저장소 Seed-Diversity
                  Ensemble Promotion Gate). 시드는 Phase 2/3 와 같은 집합이라 현직 번들과
                  시드별로 짝이 맞고, 질의 행은 시드 간 고정이라 델타가 같은 행 위에서 나온다.
K3  컨텍스트 단조: 컨텍스트를 4k→전량으로 늘릴 때 bal_acc 가 **증가**해야 한다
                  (증가하지 않으면 TabICL 을 쓸 이유 자체가 없다 — TabPFN 으로 충분).
K1~K3(및 K2b)을 모두 통과해야 Stage 1(경제성: `_prediction_output` → 결정 → PnL)로 간다.
정확도만 오르고 경제성이 0인 사례가 이 저장소에 반복해 있었으므로, K1~K3 통과는
**진행 허가일 뿐 채택 근거가 아니다.**
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_eval_omega4_3head_parent72_regimespine_balnobb_20260909 as spine  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as p72  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/tabicl_direction"
# Phase 2 스윕(run_omega461_regimespine_phase2_sweep_20260909.sh)이 실제로 쓴 라벨 계약.
DIR_LBL = (ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629"
           / "label_contracts/zigzag_action_labels_20260531")

# ⚠️ 2026-09-09 발견 — 라우팅은 balnobb 가 아니라 **wide24** 다.
# regimespine 래퍼는 `omega._load_omega_frames` / `omega._numeric_feature_cols` 두 개만 패치한다.
# MoE 라우팅은 `hard._route_id(frame)` 이 담당하는데 이 함수는 모듈 상수 `hard.ROUTE_COLS`
# (= wide24 확률 3열)를 직접 읽으므로 패치 대상 밖이었다. `_fit_expert_omega4` 의 전문가별
# 표본가중치(`parent._route_probs`)도 같은 상수를 쓴다.
# → Phase 2 의 balnobb arm 이 바꾼 것은 base_cols 102개 중 **입력 피쳐 6개뿐**이고,
#   전문가 배정·가중치는 양 arm 이 동일하게 wide24 였다.
# 2026-09-09 사용자 결정: "wide는 버리고 새로 만든 balnobb 로 그냥 진행해".
# → 라우팅 출처를 **balnobb 하나로 통일**한다. wide24 는 이 스크립트에서 완전히 빠진다.
#
# ⚠️ 현직 TabM 번들에 붙는 단서(수치 해석시 필요): 그 번들은 wide24 배정으로 학습됐으므로
#    balnobb 로 배정하면 학습 때와 다른 전문가에 행이 간다. 다만 전문가는 하드 분할이 아니라
#    레짐 확률을 **표본가중치**로 받아 전 행을 보고 학습하므로(`_fit_expert_omega4`),
#    "본 적 없는 입력"이 아니라 가중치가 어긋난 배정이다. 두 라벨 일치율이 69.3% 이므로
#    현직 값은 다소 보수적으로 읽어야 한다 — K2 는 그 전제 위의 판정이다.
ROUTE_COLS_SPINE = [f"{spine.NEW_PREFIX}{c}_prob" for c in ("bull", "bear", "chop")]
# 균형 컨텍스트의 천장은 **소수 클래스 크기**가 정한다. 클래스0 은 약 9,243행뿐이라
# 완전 균형 상한이 3x9,243 ≈ 27.7k 다. 그 위 칸(64k/전량)은 균형이 깨진 상태를 재는 것이고
# 1시드 실행에서 이미 재봤다(32k 0.5690 → 64k 0.5224 → 전량 0.5143, 균형 붕괴와 함께 하락).
# 5시드 실행에서 그 두 칸을 다시 도는 건 가장 비싼 칸(예측 561s/150s)에 GPU 를 태우는 낭비다.
LADDER = [4000, 8000, 16000, 32000]
QUERY_CAP = 6000                                    # 사다리 단계의 VAL 질의 표본
QUERY_SEED = 615372041                              # 질의 표본 선택 시드 — **시드 간 고정**
# Seed-Diversity Ensemble Promotion Gate: N>=5, 진짜 무작위(고정 간격 증가 금지).
# Phase 2/3 와 같은 시드 집합이라 현직 번들과 시드별로 짝이 맞는다.
SEEDS = [615372041, 208844917, 933105268, 471926350, 862017594]



def _binom_ge(k: int, n: int) -> float:
    """공정한 동전(p=0.5)에서 n 시드 중 k 개 이상이 양수일 확률.

    과반 규칙의 검정력을 그대로 드러내기 위해 찍는다 — 5시드에서 3/5 는 귀무에서도 50% 로
    나온다. **과반 통과는 증거가 아니다.** 정보는 Δ 중앙값과 그 일관성이 담는다.
    """
    from math import comb
    return sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)

def _bal_acc(y: np.ndarray, p: np.ndarray) -> float:
    accs = []
    for c in np.unique(y):
        m = y == c
        if m.sum():
            accs.append(float((p[m] == c).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def _macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    f1s = []
    for c in np.unique(y):
        tp = float(((p == c) & (y == c)).sum())
        fp = float(((p == c) & (y != c)).sum())
        fn = float(((p != c) & (y == c)).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def _score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"bal_acc": _bal_acc(y, p), "acc": float((p == y).mean()), "macro_f1": _macro_f1(y, p),
            "n": int(len(y))}


def _controls(y_ctx: np.ndarray, y_q: np.ndarray, rng: np.random.Generator) -> dict[str, dict]:
    """자명 대조군 — 이 저장소에서 조합 7건을 닫은 바로 그 대조군."""
    out = {}
    for c in (0, 1, 2):
        out[f"always_{c}"] = _score(y_q, np.full(len(y_q), c, dtype=np.int64))
    prior = np.bincount(y_ctx, minlength=3).astype(np.float64)
    prior /= prior.sum()
    out["stratified_prior"] = _score(y_q, rng.choice(3, size=len(y_q), p=prior).astype(np.int64))
    out["majority"] = out[f"always_{int(np.argmax(prior))}"]
    return out


def _select_ctx(y: np.ndarray, pool: np.ndarray, n_ctx: int, balanced: bool,
                seed: int = 0) -> tuple[np.ndarray, dict]:
    """컨텍스트 행 선택. (선택된 인덱스, 진단) 을 돌려준다.

    balanced=True 는 **클래스당 같은 수**를 뽑는다.
    왜 필요한가 — 판정 지표가 balanced accuracy 인데 현직 TabM 은
    `compute_sample_weight(class_weight="balanced")` 로 그 지표를 직접 최적화하도록 학습한다.
    TabICL 은 sample_weight 를 못 받으므로 **컨텍스트 구성**으로 같은 것을 인코딩해야 짝이 맞는다.
    컨텍스트 크기가 GPU 상한에 묶이므로 복제가 아니라 다수 클래스 하향표집이 맞다.

    ⚠️ 2026-09-09 버그와 그 수정 — **클래스마다 "최근 N개"를 뽑으면 안 된다.**
    클래스 0 은 전체의 11% 라 N 개를 채우려 훨씬 과거까지 거슬러 올라가고, 클래스 1(48%) 은
    최근 구간에서 끝난다. 그러면 세 클래스가 **서로 다른 시대**에서 뽑혀 "클래스 = 시대" 라는
    가짜 신호가 컨텍스트에 심긴다. 천천히 드리프트하는 피쳐 아무거나로 그 규칙을 학습하고,
    VAL 은 전부 최근이라 통째로 오작동한다 — 실측 정확도가 0.20 까지 떨어졌다(무작위 0.333 이하).

    수정: **창을 먼저 고정한다.** 최근에서부터 창을 넓혀 모든 클래스가 per 개씩 확보되는
    최소 창을 찾고, 그 창 **안에서** 다수 클래스를 무작위 하향표집한다. 세 클래스가 같은
    시간창을 공유하므로 시대 교란이 사라진다.
    """
    if not balanced:
        sel = pool[-min(int(n_ctx), len(pool)):]
        return sel, {"balanced": False, "window_rows": int(len(sel))}
    # ⚠️ 2026-09-10 수정 — 클래스 수를 3 으로 하드코딩하고 있었다. exit head 는 2 클래스라
    # per = n_ctx//3 이 되어 명목 컨텍스트의 2/3 만 쓰였다(두 팔 모두 동일하게 적용돼 비교
    # 자체는 편향되지 않았지만 칸 라벨이 실제 행 수와 어긋났다). 라벨 집합에서 직접 센다.
    classes = np.unique(y)                            # pool 이 아니라 **전체** y 기준(칸마다 흔들리지 않게)
    k = max(1, len(classes))
    per = max(1, int(n_ctx) // k)
    onehot = np.zeros((len(pool), k), dtype=np.int64)
    cls_pos = {int(c): i for i, c in enumerate(classes)}
    for i, v in enumerate(y[pool]):
        j = cls_pos.get(int(v))
        if j is not None:
            onehot[i, j] = 1
    cum = np.cumsum(onehot[::-1], axis=0)            # 최근 → 과거 누적
    ok = np.where((cum >= per).all(axis=1))[0]
    w = int(ok[0]) + 1 if len(ok) else len(pool)     # per 개씩 확보되는 최소 창
    win = pool[-w:]
    rng = np.random.default_rng(int(seed))
    parts = []
    for c in classes:
        idx = win[y[win] == c]
        parts.append(idx if len(idx) <= per else np.sort(rng.choice(idx, size=per, replace=False)))
    sel = np.sort(np.concatenate(parts)) if parts else pool[:0]
    # 진단: 클래스별 인덱스 범위. 범위가 크게 어긋나 있으면 '클래스=시대' 교란의 재발이다.
    spans = {int(c): [int(sel[y[sel] == c].min()), int(sel[y[sel] == c].max())]
             for c in classes if (y[sel] == c).any()}
    return sel, {"balanced": True, "window_rows": int(w), "per_class": int(per),
                 "class_index_span": spans,
                 "counts": {int(c): int((y[sel] == c).sum()) for c in classes}}


def _make_model(kind: str, n_estimators: int, seed: int, device: str):
    """모델 팩토리. 두 팔은 **같은 컨텍스트·같은 질의 행**을 받는다 — 모델만 다르다.

    tabpfn — 2026-09-09 사용자 지시로 추가. 이 저장소가 오래 들고 있던 "TabPFN 상한 18,000행"은
      모델 제약이 아니라 옛 스크립트의 관례 상수(v2 시절 10,000행 한계에서 파생)였다.
      설치된 tabpfn 8.5.0 의 기본 모델(v3)에서 실제 해석되는 상한을 직접 읽어 확인했다:
        MAX_NUMBER_OF_SAMPLES=1,000,000 · MAX_NUMBER_OF_FEATURES=2,000 · MAX_NUMBER_OF_CLASSES=160
      즉 78,568행 direction head 는 TabPFN 사거리 안이고, TabICL 로만 보낼 이유가 없었다.
      (v3 가중치는 비상업 라이선스라 라이브 배포는 불가 — 사용자 지시로 연구용 진행.)

    `balance_probabilities=False` 로 둔다 — 클래스 균형은 컨텍스트 구성(`_select_ctx`)에서
    이미 하고 있고, 여기서 또 보정하면 이중 교정이 되어 TabICL 팔과 짝이 깨진다.
    """
    if kind == "tabicl":
        from tabicl import TabICLClassifier
        return TabICLClassifier(n_estimators=int(n_estimators), random_state=int(seed),
                                device=device, offload_mode="auto", batch_size=2, verbose=False)
    if kind == "tabpfn":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(n_estimators=int(n_estimators), random_state=int(seed),
                                device=device, memory_saving_mode="auto",
                                balance_probabilities=False)
    raise ValueError(f"unknown model kind: {kind}")


def _gpu_mib() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 2 ** 20)
    except Exception:
        pass
    return float("nan")


def _fit_predict(x_ctx: pd.DataFrame, y_ctx: np.ndarray, x_q: pd.DataFrame, *,
                 n_estimators: int, seed: int, device: str,
                 kind: str = "tabicl") -> tuple[np.ndarray, dict]:
    import torch
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    clf = _make_model(kind, n_estimators, seed, device)
    t0 = time.time()
    clf.fit(x_ctx.to_numpy(np.float32), y_ctx)
    t_fit = time.time() - t0
    t0 = time.time()
    pred = clf.predict(x_q.to_numpy(np.float32)).astype(np.int64)
    t_pred = time.time() - t0
    return pred, {"fit_s": round(t_fit, 1), "predict_s": round(t_pred, 1),
                  "peak_gpu_mib": round(_gpu_mib(), 1), "context_rows": int(len(y_ctx)),
                  "model": kind}


def _incumbent(val_raw: pd.DataFrame, x_val: pd.DataFrame, base_cols: list[str], comp: str,
               seed: int, qidx: np.ndarray) -> dict | None:
    """Phase 2 balnobb 부모 번들의 direction head — 같은 VAL 행, 같은 라우팅."""
    import torch
    pdir = (ROOT / "tmp/causal_regen_20260516" /
            f"omega4_3head_parent72_loose_entry_quality_20260620_regimespine_balnobb_{comp}_s{seed}_20260909")
    bp = pdir / "true_3head_tabm_bundle.pt"
    if not bp.exists():
        return None
    bundle = torch.load(bp, map_location="cpu", weights_only=False)
    # `_base_input` 은 base_cols(102) 뒤에 POS_COLS(13) 를 붙여 115열을 돌려준다.
    # 계약 대조는 앞의 102열 목록으로 한다.
    if list(bundle["base_cols"]) != list(base_cols):
        raise RuntimeError(f"현직 번들 base_cols({len(bundle['base_cols'])}) 가 이 프레임"
                           f"({len(base_cols)}) 과 다르다 — 짝 비교 불가")
    dev = parent._device("cpu")
    preds = {e: parent._predict_payload(bundle["models"][e], x_val, device=dev) for e in hard.EXPERT_NAMES}
    route = np.argmax(val_raw[ROUTE_COLS_SPINE].to_numpy(np.float64), axis=1).astype(np.int64)
    direction = parent._routed(preds, route, "direction", 3)
    return {"bundle": str(bp), "pred": np.argmax(direction, axis=1).astype(np.int64)[qidx]}


def _run_ladder(*, arm: str, x_tr, y_tr, xq, yq, route_tr, route_q, n_ctx_list,
                balanced: bool, seed: int, n_estimators: int, device: str,
                kind: str = "tabicl") -> dict:
    out = {}
    for rung in n_ctx_list:
        n_ctx = len(y_tr) if rung == 0 else min(rung, len(y_tr))
        tag = "전량" if rung == 0 else f"{rung//1000}k"
        # 컨텍스트는 인과적으로 VAL 직전 구간(최근 행)에서 고른다 — 무작위 부분표집이
        # 아니라 실제 배포에서 쓸 수 있는 형태다.
        #   global: 전체 풀에서 곧바로 균형 선택.
        #   moe   : 최근 n_ctx 를 풀로 잡고 전문가로 나눈 **뒤** 전문가 안에서 균형을 맞춘다.
        pool = np.arange(len(y_tr))
        if arm == "global":
            sel, ctx_diag = _select_ctx(y_tr, pool, n_ctx, balanced, seed)
        else:
            sel, ctx_diag = pool[-n_ctx:], {"balanced": balanced, "per_expert": True}
        try:
            if arm == "global":
                pred, meta = _fit_predict(x_tr.iloc[sel].reset_index(drop=True), y_tr[sel], xq,
                                          n_estimators=n_estimators, seed=seed, device=device,
                                          kind=kind)
            else:
                pred = np.full(len(yq), -1, dtype=np.int64)
                meta = {"fit_s": 0.0, "predict_s": 0.0, "peak_gpu_mib": 0.0, "context_rows": 0,
                        "expert_rows": {}}
                for e in range(3):
                    # ⚠️ 2026-09-09 버그 수정 — 전문가별 예산은 n_ctx 가 아니라 **n_ctx/3** 이다.
                    # 전문가 하나가 가진 행 수 자체가 이미 약 n_ctx/3 이라, per=n_ctx//3 로 넘기면
                    # 클래스당 상한에 아무것도 안 걸려 **하향표집이 한 번도 발동하지 않는다**(=raw).
                    # 실측으로 moe 값이 raw 실행과 소수점 둘째 자리까지 같았고 context_rows 가
                    # 정확히 n_ctx 였다.
                    cm, _d = _select_ctx(y_tr, sel[route_tr[sel] == e], n_ctx // 3, balanced, seed + e)
                    ctx_diag[hard.EXPERT_NAMES[e]] = _d
                    qm = np.where(route_q == e)[0]
                    meta["expert_rows"][hard.EXPERT_NAMES[e]] = [int(len(cm)), int(len(qm))]
                    if len(qm) == 0:
                        continue
                    if len(cm) < 50 or len(np.unique(y_tr[cm])) < 2:
                        pred[qm] = int(np.bincount(y_tr[sel], minlength=3).argmax())
                        continue
                    pp, m = _fit_predict(x_tr.iloc[cm].reset_index(drop=True), y_tr[cm],
                                         xq.iloc[qm].reset_index(drop=True),
                                         n_estimators=n_estimators, seed=seed, device=device,
                                         kind=kind)
                    pred[qm] = pp
                    meta["fit_s"] += m["fit_s"]; meta["predict_s"] += m["predict_s"]
                    meta["context_rows"] += m["context_rows"]
                    meta["peak_gpu_mib"] = max(meta["peak_gpu_mib"], m["peak_gpu_mib"])
                if (pred < 0).any():
                    raise RuntimeError("moe: 예측이 채워지지 않은 질의 행이 있다")
            sc = _score(yq, pred)
            out[tag] = {**sc, **meta, "ctx_diag": ctx_diag}
            print(f"    ctx {tag:5s} ({meta['context_rows']:>7,}행)  bal_acc {sc['bal_acc']:.4f}  "
                  f"acc {sc['acc']:.4f}  macroF1 {sc['macro_f1']:.4f}  "
                  f"fit {meta['fit_s']:.0f}s pred {meta['predict_s']:.0f}s "
                  f"peakGPU {meta['peak_gpu_mib']:.0f}MiB", flush=True)
        except Exception as exc:  # OOM 포함 — 기록하고 계속(조용한 축소 금지)
            out[tag] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"    ctx {tag:5s}  ❌ {type(exc).__name__}: {str(exc)[:150]}", flush=True)
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", default="zig075", choices=["zig075", "h48qual"])
    ap.add_argument("--seeds", default=",".join(str(x) for x in SEEDS),
                    help="Seed-Diversity Gate: N>=5 진짜 무작위 시드(고정 간격 금지)")
    ap.add_argument("--query-seed", type=int, default=QUERY_SEED,
                    help="VAL 질의 표본 선택 시드. **시드 간 고정**이라 짝지은 비교가 같은 행 위에서 이뤄진다")
    ap.add_argument("--n-estimators", type=int, default=16)
    ap.add_argument("--balance-context", default="balanced", choices=["balanced", "raw"],
                    help="balanced: 클래스당 동수 하향표집(현직의 class_weight='balanced' 와 짝을 맞춤)")
    ap.add_argument("--model", default="tabicl", choices=["tabicl", "tabpfn"],
                    help="같은 컨텍스트·같은 질의 행에서 모델만 교체한 짝지은 비교")
    ap.add_argument("--ladder", default="", help="쉼표 구분 컨텍스트 크기(0=전량). 비우면 기본 사다리")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arms", default="global,moe")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    global LADDER
    if str(args.ladder).strip():
        LADDER = [int(x) for x in str(args.ladder).split(",") if x.strip()]
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    balanced = str(args.balance_context) == "balanced"

    base_cols = spine._install(args.component)
    # 라벨/품질 모드는 Phase 2 스윕과 같은 계약을 쓴다. direction head 게이트라 quality 는
    # 사용하지 않으므로 `same_as_direction`(zig075 arm 이 실제로 쓴 값)으로 고정한다.
    frames = p72._prepare_frames(
        disable_tp_sl=False, direction_label_dir=DIR_LBL,
        quality_mode="same_as_direction", quality_label_dir=None,
        quality_min_edge=0.0010, quality_max_mae=0.0100,
        quality_min_mfe_mae=1.20, quality_max_hold_bars=288)
    train_raw, val_raw = frames["train_raw"], frames["val_raw"]
    x_tr = parent._base_input(train_raw, base_cols)
    y_tr = train_raw["zigzag_action"].to_numpy(np.int64)
    x_va = parent._base_input(val_raw, base_cols)
    y_va = val_raw["zigzag_action"].to_numpy(np.int64)
    print(f"\n[데이터] TRAIN {len(y_tr):,}행 · VAL {len(y_va):,}행 · 피쳐 {x_tr.shape[1]}열", flush=True)
    print(f"  TRAIN 클래스 분포 {np.bincount(y_tr, minlength=3).tolist()}  "
          f"VAL {np.bincount(y_va, minlength=3).tolist()}", flush=True)

    # 질의 표본은 **시드 간 고정**이다. 모델 시드만 흔들어야 짝지은 델타가 같은 행 위에서 나온다.
    qrng = np.random.default_rng(int(args.query_seed))
    qidx = np.sort(qrng.choice(len(y_va), size=min(QUERY_CAP, len(y_va)), replace=False))
    xq, yq = x_va.iloc[qidx].reset_index(drop=True), y_va[qidx]
    route_tr = np.argmax(train_raw[ROUTE_COLS_SPINE].to_numpy(np.float64), axis=1)
    route_q = np.argmax(val_raw[ROUTE_COLS_SPINE].to_numpy(np.float64), axis=1)[qidx]
    print(f"  질의 표본 {len(yq):,}행 (질의 시드 {args.query_seed}, 시드 간 고정)", flush=True)

    rep: dict[str, Any] = {
        "component": args.component, "seeds": seeds, "query_seed": int(args.query_seed),
        "model": str(args.model),
        "n_estimators": int(args.n_estimators), "balance_context": str(args.balance_context),
        "n_train": int(len(y_tr)), "n_val": int(len(y_va)), "n_query": int(len(yq)),
        "ladder": LADDER,
        "routing_source": {
            "cols": ROUTE_COLS_SPINE, "applies_to": ["incumbent_tabm", "tabicl_moe"],
            "decision": "2026-09-09 사용자: wide24 폐기, balnobb 단일 라우팅",
            "caveat": "현직 번들은 wide24 배정으로 학습됐다(regimespine 래퍼가 hard.ROUTE_COLS 를 "
                      "패치하지 않아 Phase 2 양 arm 모두 wide24 라우팅). 전문가가 소프트 가중치로 "
                      "전 행을 보므로 치명적이진 않으나 현직 값은 보수적으로 읽어야 한다.",
            "tabicl_features": f"{spine.NEW_PREFIX}* (base_cols 102 중 레짐 6열)"}}

    rep["controls"] = _controls(y_tr, yq, np.random.default_rng(int(args.query_seed)))
    print(f"\n[대조군] (VAL 질의 {len(yq):,}행, 시드 간 고정)", flush=True)
    for k, v in rep["controls"].items():
        print(f"  {k:18s} bal_acc {v['bal_acc']:.4f}  acc {v['acc']:.4f}  macroF1 {v['macro_f1']:.4f}",
              flush=True)
    ctrl_best = max(v["bal_acc"] for k, v in rep["controls"].items() if k != "majority")

    rep["per_seed"] = {}
    for seed in seeds:
        print(f"\n{'='*96}\n[시드 {seed}]", flush=True)
        one: dict[str, Any] = {}
        inc = _incumbent(val_raw, x_va, base_cols, args.component, seed, qidx)
        if inc is None:
            print("  현직 번들 없음 — 이 시드는 K2 판정 제외", flush=True)
            one["incumbent"] = None
        else:
            one["incumbent"] = {"bundle": inc["bundle"], **_score(yq, inc["pred"])}
            i = one["incumbent"]
            print(f"  현직 TabM  bal_acc {i['bal_acc']:.4f}  acc {i['acc']:.4f}  "
                  f"macroF1 {i['macro_f1']:.4f}", flush=True)
        one["arms"] = {}
        for arm in arms:
            print(f"  [팔: {arm}]{'' if arm == 'global' else '  라우팅=balnobb'}", flush=True)
            one["arms"][arm] = _run_ladder(
                arm=arm, x_tr=x_tr, y_tr=y_tr, xq=xq, yq=yq, route_tr=route_tr, route_q=route_q,
                n_ctx_list=LADDER, balanced=balanced, seed=seed,
                n_estimators=int(args.n_estimators), device=str(args.device),
                kind=str(args.model))
        rep["per_seed"][str(seed)] = one
        (OUT / f"{args.model}_direction_{args.component}_5seed.json").write_text(
            json.dumps(rep, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    # ── 시드 집계 · 킬 게이트 ──
    print(f"\n{'='*96}\n[시드 집계] N={len(seeds)}  (VAL 전용 · validation_only)", flush=True)
    cells: dict[str, dict] = {}
    for arm in arms:
        for tag in [("전량" if r == 0 else f"{r//1000}k") for r in LADDER]:
            accs, deltas = [], []
            for seed in seeds:
                one = rep["per_seed"][str(seed)]
                d = one["arms"].get(arm, {}).get(tag)
                if not d or "bal_acc" not in d:
                    continue
                accs.append(d["bal_acc"])
                if one["incumbent"] is not None:
                    deltas.append(d["bal_acc"] - one["incumbent"]["bal_acc"])
            if not accs:
                continue
            cells[f"{arm}|{tag}"] = {
                "n_seeds": len(accs), "median_bal_acc": statistics.median(accs),
                "min_bal_acc": min(accs), "max_bal_acc": max(accs),
                "n_delta": len(deltas),
                "median_delta_vs_incumbent": statistics.median(deltas) if deltas else None,
                "seeds_better_than_incumbent": sum(1 for x in deltas if x > 0),
                "sign_agreement": (len(deltas) > 0 and
                                   (all(x > 0 for x in deltas) or all(x < 0 for x in deltas)))}
    inc_accs = [rep["per_seed"][str(s)]["incumbent"]["bal_acc"] for s in seeds
                if rep["per_seed"][str(s)]["incumbent"] is not None]
    rep["incumbent_summary"] = {"n": len(inc_accs),
                                "median_bal_acc": statistics.median(inc_accs) if inc_accs else None,
                                "min": min(inc_accs) if inc_accs else None,
                                "max": max(inc_accs) if inc_accs else None}
    print(f"  현직 TabM  중앙 {rep['incumbent_summary']['median_bal_acc']:.4f}  "
          f"[{rep['incumbent_summary']['min']:.4f}, {rep['incumbent_summary']['max']:.4f}]  "
          f"n={rep['incumbent_summary']['n']}", flush=True)
    print(f"  대조군 최고 {ctrl_best:.4f}\n", flush=True)
    print(f"  {'셀':16s} {'중앙bal':>8s} {'[최소,최대]':>18s} {'Δ중앙':>8s} {'이김':>6s} {'부호일치':>8s}",
          flush=True)
    for k, v in cells.items():
        dv = "n/a" if v["median_delta_vs_incumbent"] is None else f"{v['median_delta_vs_incumbent']:+.4f}"
        print(f"  {k:16s} {v['median_bal_acc']:8.4f} "
              f"[{v['min_bal_acc']:.4f},{v['max_bal_acc']:.4f}] {dv:>8s} "
              f"{v['seeds_better_than_incumbent']:>3d}/{v['n_delta']:<2d} "
              f"{'✅' if v['sign_agreement'] else '❌':>8s}", flush=True)
    rep["cells"] = cells

    print(f"\n{'='*96}\n[킬 게이트] 선택은 VAL 중앙값으로만 한다(validation_only)", flush=True)
    if not cells:
        rep["verdict"] = {"K1": None, "K2": None, "K3": None}
        print("  실행 가능한 셀 없음 — 판정 불가", flush=True)
    else:
        best_k, best = max(cells.items(), key=lambda kv: kv[1]["median_bal_acc"])
        k1 = best["median_bal_acc"] > ctrl_best
        k2 = (best["median_delta_vs_incumbent"] is not None
              and best["median_delta_vs_incumbent"] > 0)
        # 시드 규칙 — 2026-09-09 사용자 지시로 **과반(>n/2)** 으로 완화.
        # ⚠️ 이건 **페이즈 진행** 기준이지 승격 기준이 아니다. 저장소의 Seed-Diversity Ensemble
        #    Promotion Gate 는 그대로 N>=5 전부 일치를 요구한다. 이 결과를 승격 근거로 인용 금지.
        #    `seed_rule_null_p` 가 그 이유를 수치로 보여준다(3/5 는 귀무에서도 50%).
        n_d = best["n_delta"]
        k2_seed = n_d > 0 and best["seeds_better_than_incumbent"] * 2 > n_d
        k2_seed_strict = best["seeds_better_than_incumbent"] == n_d == len(seeds)
        seed_p = _binom_ge(best["seeds_better_than_incumbent"], n_d) if n_d else None
        mono = {}
        for arm in arms:
            xs = [cells[f"{arm}|{t}"]["median_bal_acc"]
                  for t in [("전량" if r == 0 else f"{r//1000}k") for r in LADDER]
                  if f"{arm}|{t}" in cells]
            mono[arm] = round(xs[-1] - xs[0], 4) if len(xs) >= 2 else None
        k3 = any(v is not None and v > 0 for v in mono.values())
        print(f"  최선 셀 : {best_k}  중앙 bal_acc {best['median_bal_acc']:.4f}", flush=True)
        print(f"  K1 대조군 우위 (>{ctrl_best:.4f})            : {'✅' if k1 else '❌'}", flush=True)
        print(f"  K2 현직 우위 (Δ중앙 "
              f"{'n/a' if best['median_delta_vs_incumbent'] is None else format(best['median_delta_vs_incumbent'], '+.4f')})"
              f"     : {'✅' if k2 else '❌'}", flush=True)
        print(f"  K2b 시드 과반 우위 ({best['seeds_better_than_incumbent']}/{n_d}) "
              f": {'✅' if k2_seed else '❌'}   "
              f"[귀무 p={seed_p:.3f} · 엄격(전부일치) {'✅' if k2_seed_strict else '❌'}]", flush=True)
        print(f"  K3 컨텍스트 단조 증가 {mono}   : {'✅' if k3 else '❌'}", flush=True)
        allp = bool(k1) and bool(k2) and bool(k2_seed) and bool(k3)
        rep["verdict"] = {"best_cell": best_k, "median_bal_acc": best["median_bal_acc"],
                          "control_best_bal_acc": ctrl_best,
                          "K1": bool(k1), "K2": bool(k2),
                          "K2b_seed_majority": bool(k2_seed),
                          "K2b_seed_all_agree_strict": bool(k2_seed_strict),
                          "seed_rule": "majority (user 2026-09-09) — phase progression only, "
                                       "NOT a promotion basis; repo gate still requires all-agree",
                          "seed_rule_null_p": seed_p,
                          "K3": bool(k3), "context_monotonicity": mono,
                          "stage1_allowed": allp}
        print(f"\n  → Stage 1(경제성) 진행 {'허가' if allp else '불가'}"
              f" — 통과해도 채택 근거는 아니다(정확도 개선이 경제 이득으로 이어지지 않은 사례가 "
              f"이 저장소에 반복돼 있다)", flush=True)

    pth = OUT / f"{args.model}_direction_{args.component}_5seed.json"
    pth.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n산출물: {pth}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
