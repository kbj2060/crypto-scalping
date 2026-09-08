# Phase 1 — 동결 부모 라우팅만 교체 A/B (2026-09-09)

상태: **완료. 판정 = 무신호(inconclusive).** VAL 은 후보가 크게 열세, OOS 는 크게 우세 —
표본이 12~29건이라 두 창을 가르는 것이 신호인지 잡음인지 구분할 수 없다.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계약: `docs/model_contracts/omega461_regimegbm_rebuild_contract.md`
스크립트: `scripts/research_omega461_regimegbm_phase1_routing_ab_20260909.py`
산출물: `tmp/omega461_regimegbm_rebuild_20260909/phase1_routing_ab/{report.json,*/*/ledger.csv}`

## 설계 — 정확히 한 줄만 바꾼다

부모 ThreeHeadTabM ×2컴포넌트 ×3전문가, 리스크 사이드카, 라우터, 배리어 **전부 동결**.
피쳐도 그대로다 — 부모의 102 `base_cols` 는 여전히 wide24 레짐 6컬럼을 포함한 채 들어간다.
바꾸는 것은 `_route_id(frame)` 가 argmax 하는 확률 3컬럼의 **출처** 하나뿐:

| arm | route |
|---|---|
| A (baseline) | `argmax(regime3_current_sensitive_wide24_{bull,bear,chop}_prob)` |
| B (candidate) | `argmax(regime3_s12k3_cut2509_{bull,bear,chop}_prob)` |

즉 "어느 전문가 서브넷이 이 봉에 답하는가"만 새 레짐이 정한다. `greedy_replay` /
`prepare_component` 는 `replay_omega4_6_1_greedy_router_20260706.py` 원본을 import 해 그대로 썼다.

## 결과

| 창 | 라우팅 일치 | arm | PnL | MDD | 거래 | WR | 컴포넌트 |
|---|---:|---|---:|---:|---:|---:|---|
| validation<br>2025-10-01~12-31<br>26,496봉 | 69.30% | A wide24 | **+36.82%** | −24.34% | 29 | 41.4% | zig075 22 / h48qual 7 |
| | | B cut2509 | **+10.97%** | −21.69% | 28 | 39.3% | zig075 20 / h48qual 8 |
| | | **Δ(B−A)** | **−25.86pp** | **+2.65pp** | −1 | −2.1pp | |
| oos<br>2026-01-01~02-28<br>16,992봉 | 68.15% | A wide24 | **+59.82%** | −12.88% | 13 | 53.8% | zig075 12 / h48qual 1 |
| | | B cut2509 | **+100.84%** | −14.38% | 12 | 66.7% | zig075 12 |
| | | **Δ(B−A)** | **+41.03pp** | **−1.50pp** | −1 | +12.9pp | |

전문가 라우팅 비중 (A → B):

| 창 | bull | bear | chop |
|---|---|---|---|
| validation | .256 → .219 | .248 → .206 | **.497 → .575** |
| oos | .239 → .206 | .260 → .212 | **.501 → .581** |

두 창 모두 새 레짐이 **chop 전문가로 약 8pp 더 보낸다**. 라우팅 일치율 69.30% / 68.15% 는
분류기 비교 문서의 69.3% 와 일치한다(`omega461_regime_classifier_comparison_20260909.md`).

## 판정 — 무신호

**부호가 두 창에서 정반대다.** VAL 은 −25.86pp, OOS 는 +41.03pp. MDD 도 방향이 반대다
(VAL 개선, OOS 악화). 그리고 **표본이 29건과 13건**이다 — OOS 의 +41pp 는 트레이드 한두 건이
만들 수 있는 크기다.

이 VAL/OOS 불일치 서명은 이 저장소가 이미 위험 신호로 기록해둔 것이다:
`omega4_6_1_upgrade_investigation_20260706.md` 의 여섯 후보 중 **셋이 같은 서명으로 기각**됐다.

계약의 후보 선택 규칙은 `candidate_selection_scope: validation_only` 다. **그 규칙만 적용하면
이 후보는 VAL 에서 25.86pp 열세이므로 선택되지 않는다.** OOS 우세는 선택 근거로 쓸 수 없다.

## 그럼에도 Phase 2 를 자동 기각하지 않는 이유 (계약에 사전 기재)

Phase 1 은 **의도적으로 불일치를 도입한 진단**이다. 세 전문가는 wide24 라우팅으로 학습됐는데
라우팅만 바꾸면 각 전문가가 학습 때와 다른 봉 분포를 받는다 — 특히 chop 전문가로 8pp 더
보내면서, 그 전문가가 학습 중 본 적 없는 종류의 봉을 처리하게 된다. Phase 2 의 재학습이
바로 이 불일치를 없애는 단계다.

따라서 이 결과는 **"후보가 나쁘다"의 근거가 아니라 "이 진단으로는 판별할 수 없다"의 근거**다.
Phase 2 진행 여부는 이 숫자가 아니라 사전 논거(재학습이 불일치를 제거한다)와 비용 판단으로
결정해야 한다.

## 이 실험이 답하지 않는 것

- 재학습 후에도 chop 편향이 유지되는지 — Phase 2 가 답한다.
- 12~29건 표본에서 나온 차이의 통계적 유의성 — 이 표본 크기로는 어떤 검정도 무력하다.
  전신 라인의 고질적 제약(6개월 25건)을 그대로 물려받았다.

## 준수

신규 학습 없음(부모·사이드카 전부 동결 재스코어링). 라이브 파일
(`trading_bot.py`, `omega4_6_1_live.py`, `runtime_config.py`, `dashboard/**`, `.env`) 미변경.
`greedy_replay`/`prepare_component` 재구현 없음 — 원본 import.
후보 선택은 validation 만 본다는 계약 규칙을 이 문서가 위반하지 않도록, OOS 수치는
**보고만 하고 선택 근거로 쓰지 않는다**.
