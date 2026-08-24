# ETH 공유 트렁크 레짐전문가 후보 — 데이터 및 리소스 관리 (2026-08-16)

이 문서는 `docs/model_contracts/eth_candidate_shared_trunk_regime_experts_contract_20260816.md`
(C3, `docs/experiments/eth_odyssey4_layer_and_parameter_improvement_proposal_20260816.md`의 가장
큰 항목)에서 새로 만지거나 검토한 모든 데이터/리소스를 모은다. 상위 제안 축(A1/B1/B2/C1/C2)의
리소스는 각 실험 문서에서 관리하며, 여기서는 C3 전용만 다룬다.

**새 리소스를 만질 때마다 그 턴에 행을 추가/갱신할 것** — 나중으로 미루지 않는다. 상태 값
컨벤션: `활성`, `인프라 확인됨-미착수`, `인프라 차단`, `검증 완료 — 긍정 결과`,
`검증 완료 — 부정 결과`.

## 코드 (스크립트·문서)

| 리소스 | 위치 | 용도 | 상태 |
|---|---|---|---|
| 공유 트렁크 후보 스크립트 | `scripts/research_eth_odyssey4_shared_trunk_regime_experts_20260816.py` | `SharedTrunkThreeHeadTabM`(트렁크 공유+3전문가 헤드) 구현, baseline(독립 트렁크 3모델) vs 후보 학습/평가, N≥5시드 루프 내장. `--feature-pipeline {light,true}`(기본값 `true`) 갱신 — 병행 세션이 복구한 진짜 라이브 102피처 파이프라인을 기본으로 사용하도록 변경 | 구현 완료 + 독립 리뷰 통과(버그 없음) + 로컬 sanity check 통과(143초, 크래시 없음, `light` 파이프라인으로 실행). 서버 N≥5시드 본실험은 GPU 대기로 미착수 — 재개 시 기본값이 `true`라 진짜 피처로 실행됨 |
| 계약 문서 | `docs/model_contracts/eth_candidate_shared_trunk_regime_experts_contract_20260816.md` | C3 결과/판정 | 진행 중 |
| 상위 제안 문서 | `docs/experiments/eth_odyssey4_layer_and_parameter_improvement_proposal_20260816.md` | C3 항목 원문(§C3), 설계 근거 | 완료(제안 단계) |
| 레이어 감사 문서 | `docs/experiments/eth_odyssey4_tabm_layer_design_review_20260816.md` | B1 저비용진단 동기(레짐전문가 유효표본수 격차) | 완료 |
| B1 진단 스크립트 | `scripts/diagnose_odyssey4_expert_effective_sample_size_20260816.py` | 3개 전문가 route_w.sum() vs len(route_w) 실측 — C3 동기 근거 수치 | 완료 |

## 재사용하는 기존 인프라 (참고용, 이 문서 소유 아님)

| 리소스 | 위치 | 용도 |
|---|---|---|
| 캐노니컬 3-head TabM 스크립트 (A1의 GCE 이식은 테스트 후 되돌려짐) | `scripts/train_eval_omega1_2_tabm_3head_20260603.py` | `ThreeHeadTabM.encode()` 구조 원본(plain CE), `gce_loss`(정의는 남아있으나 `_fit_expert_3head`는 더 이상 호출 안 함), `_routed`/`_prediction_output`/`_to_decisions`/`_metrics_with_shared_exit`/`_predict_loaded_exit`/`_load_payloads` 등 백테스트 파이프라인 그대로 재사용(미변경) |
| Regime3 라우팅 | `scripts/train_omega1_regime3_expert_direction_head_volpca_20260602.py` | `EXPERT_NAMES`/`ROUTE_COLS`/`_route_id`/`_route_probs` — 트렁크 공유 여부와 무관하게 라우팅 시맨틱 동일 유지 |
| `_prepare_frames_light()` 우회 헬퍼 | `scripts/research_eth_candidate_faithful_tabm_batchensemble_cheap_gate_20260816.py` | `_prepare_frames()`의 죽은 vsnlstm/chronos 체인을 우회, `feature_cols`=185개(프록시) — A1/B1/C1/C2/C3의 최초 실행 전부 공유 재사용 |
| 진짜 라이브 102피처 파이프라인 (병행 세션 산출물) | `scripts/eth_odyssey4_true_feature_pipeline_20260816.py`(`prepare_frames_true()`) | `_prepare_frames_light()`의 185피처 프록시를 대체하는 진짜 라이브 102 base(+13 pos)=115피처 계약 복구 — `omega._load_omega_frames()`는 정상 동작함을 확인하고 `hard._build_frame()`의 라벨 fetch만 우회, 결측 7컬럼은 수식 재현(5/7 완전 일치, funding_roc 2개는 연도 경계 콜드스타트 오차 문서화). C3 스크립트가 기본으로 임포트, C2 재확인 스크립트도 별도로 사용 | 병행 세션이 같은 날 생성, 이 세션에서 임포트/스모크테스트로 독립 검증(`x_train.shape==(78568,115)`, NaN 없음) |
| exit_head 독립 데이터셋 빌더 | `scripts/train_eval_omega1_2_tabm_exit_head_20260603.py`(`_build_exit_dataset_independent`) | exit 라벨 생성 — dev 메모리 제약으로 `max_samples=60000`(원래 무제한) 캡 적용, C1/C2/C3 공통 |

## 인프라

| 리소스 | 위치 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|
| 공유 GPU (서버) | `scripts/ops/handoff.sh` (`server` host) | N≥5시드 본실험 실행 | 이 세션 시작 시 `eth_nhits_moderntcn_direction_quality` 작업이 점유 중 — 확인 후 순서대로 진행 | 동시 실행 시 GPU 메모리 경합으로 두 작업 모두 오염 위험, 반드시 순차 실행 |
| dev 머신 (15GB RAM) | 로컬 | 로컬 sanity check(few-epoch) | 활성이지만 메모리 여유 낮음(동시 세션 5+개) | A1 검증 중 실제로 한 차례 OOM 발생(14GB 사용) — exit dataset `max_samples` 캡으로 완화(5GB 수준으로 감소 확인) |

## 산출물 디렉터리

| 리소스 | 위치 | 용도 | 상태 |
|---|---|---|---|
| 로컬 sanity check 출력 (light 185피처) | `tmp/causal_regen_20260516/eth_odyssey4_shared_trunk_regime_experts_20260816_sanity_check/report.json` | few-epoch(2) 실행 report.json — baseline 355,656 파라미터 vs shared 121,640 파라미터(34%), 크래시 없이 VAL+OOS 백테스트까지 완주 확인 | 완료 |
| 로컬 sanity check 출력 (true 115피처) | `tmp/causal_regen_20260516/eth_odyssey4_shared_trunk_regime_experts_20260816_sanity_check_true_features/report.json` | `--feature-pipeline true`로 재실행한 few-epoch(2) report.json — baseline 311,976 파라미터 vs shared 107,080 파라미터(34%, 동일 비율), 크래시 없이 VAL+OOS 백테스트까지 완주 확인. 서버 본실험(기본값 true) 착수 전 필수 확인 | 완료(2026-08-16) |
| 서버 N≥5시드 본실험 출력 | `tmp/causal_regen_20260516/eth_odyssey4_shared_trunk_regime_experts_20260816/`(서버 측, 기본 경로 그대로) | 최종 report.json, 모델 체크포인트 | 미실행(GPU 대기 — `eth_nhits_moderntcn_direction_quality`가 세션 종료 시점까지 계속 RUNNING) |

## 미검증 후보 / 보류

- 레짐 임베딩 조건부 헤드(대안 설계, 채택 안 함) — 계약 문서 §설계 참고, route_w 소프트가중치
  방식이 이미 검증된 라우팅 시맨틱과 더 잘 맞아서 채택하지 않음. 이 축이 부정 결과로 끝나면
  재검토 후보로 남겨둘 것.
