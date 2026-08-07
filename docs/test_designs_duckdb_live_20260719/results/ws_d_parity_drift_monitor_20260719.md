# WS-D 결과 — 패리티/드리프트 모니터 구축 + 자가검증 (2026-07-19 실행)

스크립트: [`scripts/research_ws_d_parity_drift_monitor_20260719.py`](../../../../scripts/research_ws_d_parity_drift_monitor_20260719.py)
원본 JSON: [`ws_d_parity_drift_monitor_20260719.json`](ws_d_parity_drift_monitor_20260719.json)

## D1 (패리티) — 이번 세션 범위 밖으로 명시

라이브 피처 재계산에는 전체 `features/elite.py` 파이프라인 재현이 필요해 이번 세션에서
구현하지 않음. 주입 테스트(A/B)와 실제 실행 모두 보류.

## D2 (드리프트) — 구현 + 자가검증 통과 + 실제 경보 1건 발견

**자가검증(합성 주입)**: `obi` 컬럼 전반부/후반부 분할 후, 후반부에 +1σ를 인위 주입.
- 무조작 상태 PSI = 0.0245 (OK 판정, 오탐 없음)
- 주입 후 PSI = 3.596 (ALERT 판정) → **검출 통과** (`detection_pass: true`)

**실제 실행 결과** (baseline: 최근 7일 이전 86,075행 vs recent: 최근 7일 9,901행):

| 컬럼 | PSI | 판정 |
|---|---|---|
| obi | 0.017 | OK |
| taker_buy_ratio | 0.007 | OK |
| shadow_toxicity_score | 0.006 | OK |
| shadow_queue_collapse | 0.016 | OK |
| shadow_absorption_score | 0.006 | OK |
| oi_delta_pct | 0.017 | OK |
| **funding_rate** | **1.585** | **ALERT** |

**실제 경보 1건 발견**: `funding_rate` 분포가 최근 7일 vs 이전 기간 사이에 큰 폭으로
이동(PSI 1.58, 기준 0.2의 8배). 펀딩비는 시장 상황에 따라 정상적으로 변하는 값이라
버그가 아닐 가능성이 높지만, **모니터가 실제로 작동한다는 증거**이자 최근 시장 레짐
변화의 정량적 신호. 수동 확인 권장 (WS-A의 월별 스프레드 재현성과 교차 검증 가능).

## D3 (섀도우 성과 이탈) — 컬럼 매칭 버그 발견/수정 + 중대한 발견

**버그**: 최초 실행 시 `pnl` 문자열만 찾아 5개 shadow DB 전부에서 컬럼 인식 실패
(`shadow_pnl` 테이블은 `pnl`이 아니라 `net_return` 컬럼 사용). 매칭 로직 수정 후 재실행.

**발견 — 5개 shadow 관측기 전부 `recent_7d_pnl_sum = 0.0`, 부트스트랩 CI도 계산 불가**
(과거 일별 데이터 부족). 원인 조사:

**최초 확인 시 오류 정정**: 처음엔 sol_shadow를 "BTC와 동일 구조(명시적 비활성)"로
추정했으나, `observer_metadata`를 직접 조회하니 **실제로는 다름** — sol_shadow는
`research_policy_enabled=True`, 가용률 79.2%로 **완전히 활성화된 상태**였다.
추정하지 말고 항상 실측할 것 (D3 자동 판별 로직 추가로 이 실수를 재발 방지, 아래 참고).

| 관측기 | model_id | `research_policy_enabled` | 가용률 | 상태 |
|---|---|---|---|---|
| btc_micro_scalp_shadow | `btc_micro_scalp_eth_v4_transfer_adapter_v1_20260718` | **False** | — (`state_q=[nan,nan,nan]`) | **BENIGN** — 명시적 비활성 워밍업, fresh_start 07-18 06:41 |
| sol_micro_scalp_shadow | `sol_micro_scalp_eth_v4_transfer_adapter_v1_20260718` | **True** | 79.2% | **UNRESOLVED** — 활성 상태인데 620개 결정 전부 무포지션 |
| eth_micro_scalp_v4_shadow | `eth_micro_scalp_source_stable_opportunity_moe_v4_20260718` | True | 높음 | **UNRESOLVED** — 876개 결정 전부 무포지션. 코드 추적(`scripts/run_eth_micro_scalp_v3_fresh_forward_observer_20260718.py:413-433`): `switch_agreement`은 `argmax(state_q)!=previous_idx`일 때만 계산되는 파생값이라 0/876은 "합의 실패"가 아니라 **정책의 argmax가 24시간 내내 단 한 번도 flat 이탈을 제안조차 안 했다**는 뜻. `improvement < policy.switch_margin_bp` 마진 게이트가 상류에 있음 — 보수적으로 정상 설정된 건지 잘못 캘리브레이션된 건지는 이 데이터만으로 판별 불가 |
| eth_micro_scalp_lifecycle_shadow | `eth_micro_scalp_dynamic_lifecycle_shadow_v1_20260718` | True | 높음 | **UNRESOLVED** — 동일 패턴(`parent_desired=0`) |
| sol_micro_scalp_entry_shadow | `sol_micro_scalp_entry_only_shadow_v1_20260718` | True | 높음 | **UNRESOLVED** — 동일 |

**D3 스크립트 개선 (실제 반영·재실행 완료)**: `observer_metadata.research_policy_enabled` +
`decisions.available` + `decisions.target_position` 논타겟 비율을 자동으로 조회해
"BENIGN(명시적 비활성/워밍업)" vs "UNRESOLVED(활성인데 무거래)"를 자동 분류하도록
`run_d3_shadow_deviation()`에 `zero_variance_explanation` 필드를 추가했다
(원본 JSON 참고). **4개 관측기가 UNRESOLVED로 분류됨 — 이는 버그 확정이 아니라
"이 데이터만으로는 정상 저주파 대기인지 멈춘 정책인지 구분 불가"라는 뜻**이며,
1주일 후 동일 잡을 재실행해 여전히 전량 무거래면 그때 실제 이상으로 격상해야 한다.

## D4 (수집 건강) — 즉시 조치가 필요한 발견 1건

각 DB 테이블의 최신행 나이/행수 실측 완료. **`microstructure.duckdb::decision_feature_frame`
(243컬럼, 상위 설계 문서가 패리티/드리프트의 핵심 아티팩트로 지목한 테이블)이
최신행 나이 **23,859분(≈16.6일)**, 마지막 기록이 2026-07-02 03:25.**

- 반면 같은 DB의 `decision_feature_frame_live_only_shadow_20260702`는 나이 14분으로
  정상 갱신 중 (299컬럼, 별도 스키마).
- 즉 **"메인" `decision_feature_frame`은 2026-07-02 이후 버려지고 shadow 변형이
  그 역할을 대신하고 있는데, 이름이 바뀌지 않아 겉보기엔 여전히 살아있는 것처럼 보임.**

  **근본 원인 확인 (코드 추적만, 데이터/설정 변경 없음)**:
  `trading_bot_modules/runtime_config.py:698-706`. 테이블명이
  `FINAL_GOVERNOR_OMEGA5_ENABLE` 플래그로 분기하도록 리팩토링되어 있다 —
  `True`면 `decision_feature_frame_{OMEGA5_MODEL_ID}`, `False`(기본값이며 `.env`에도
  오버라이드 없음을 확인)면 `decision_feature_frame_live_only_shadow_20260702`를 쓴다.
  **현재 코드 경로 어디서도 접미사 없는 `decision_feature_frame`이라는 이름은
  나오지 않는다** — 이 테이블은 이 분기 로직이 생기기 이전의 구버전 코드가 쓰던
  잔재다. 메모리에 기록된 "11일 무기록 사고"(2026-07-02 스키마 체크 하드 실패)는
  같은 자리에서 고쳐진 게 아니라 **새 테이블 체계로 갈아타는 리팩토링**이었던 것으로
  보인다. **즉 활성 버그가 아니라 코드 정리 잔재(orphan) — 필요한 조치는 "왜
  멈췄는지 규명"이 아니라 "삭제/보관 여부 결정"으로 다운그레이드.**
- 그 외 테이블(microstructure_1m류, orderbook_decision_snapshots류)은 전부
  최신행 나이 1~4분으로 정상.

## 자가검증(수락 기준) 결과 요약

| 수락 기준 | 결과 |
|---|---|
| D1 주입 테스트 A/B | 미실행 (범위 밖) |
| D2 주입 테스트(드리프트 검출) | **통과** |
| D2 무조작 오탐 0건 | **통과** (0.024 < 0.2) |
| D3 6개 shadow DB 리포트 생성 | **통과** (5개 실제 존재, 전부 리포트 생성 — 6번째 `eth_micro_scalp_lifecycle_shadow`도 포함) |
| D3 결측 감시 동작 | **통과** — `stale_alert` 정상 동작(전부 False). "0인데 왜 0인지" 자동 구분 로직도 세션 중 구현·재실행 완료 (`zero_variance_explanation` 필드, 위 표 참고) |
| D4 일간 잡 3일 연속 무오류 | 미검증 (1회성 실행만, 상시 운영 전환 필요) |
| 실행 시간/락 경합 | 통과 — 재시도 백오프로 라이브 봇과 충돌 없이 완료 (수 회 재시도 발생, 정상 동작) |

## 후속 조치 제안 (우선순위순)

1. **[P3, 원인 규명 완료·다운그레이드] `decision_feature_frame` 테이블 정리** —
   원인은 확인됨(코드 리팩토링으로 인한 orphan, 위 참고). 남은 조치는 규명이 아니라
   혼동 방지용 정리(삭제 또는 `_deprecated_20260702` 접미사로 이름 변경) 뿐, 긴급성 낮음.
2. **[P2, 완료]** D3 "0-변동성이 정상 워밍업인지 죽은 정책인지" 판별 로직 —
   세션 중 구현·재실행 완료. sol_shadow가 실제로는 BTC와 다른 상태(활성)임을
   이 자동화 덕분에 발견 (수작업 추정은 틀렸었음).
3. **[P3] `funding_rate` PSI 경보 수동 확인** — 버그인지 정상 레짐 변화인지 트레이더 확인 필요.
4. 5개 shadow 관측기는 **1주 후 재점검** — 그때도 전부 target_position=0이면 실제 이상.
