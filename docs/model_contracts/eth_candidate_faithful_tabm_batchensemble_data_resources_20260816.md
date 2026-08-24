# ETH 완전한 TabM(BatchEnsemble) 후보 — 데이터 및 리소스 관리 (2026-08-16)

이 문서는 이 후보(`docs/model_contracts/eth_candidate_faithful_tabm_batchensemble_contract_20260816.md`)에서
실제로 만지거나 검토한 모든 데이터 소스/리소스를 한 곳에 모은 목록이다.

**새 리소스를 만질 때마다 그 턴에 행을 추가/갱신할 것** — 나중으로 미루지 않는다. 상태 값 컨벤션: `활성`, `인프라 확인됨-미착수`, `인프라 차단`, `검증 완료 — 긍정 결과`, `검증 완료 — 부정 결과`.

## 이식 원본 (읽기 전용 참고)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| TabM 논문 원문 | arXiv:2410.24210 (HTML) | - | BatchEnsemble 정확한 수식(R/S/B), embedding ablation 수치 | 검증 완료 — 원문 직접 확인(WebFetch) | 이 계약의 "발견한 차이" 절이 이 원문 인용을 그대로 담고 있음 |
| 라이브 h48qual/zig075 TabM 구현 | `scripts/train_eval_omega1_2_tabm_3head_20260603.py`(`ThreeHeadTabM`, `_fit_expert_3head`, `_prepare_frames`) | - | 이식 원본이자 비교 기준선, 데이터 파이프라인 재사용 대상 | 활성 | 원본 미수정 — 신규 스크립트가 함수만 import해서 재사용 |

## 신규 산출물

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| Step A cheap_gate 스크립트 | `scripts/research_eth_candidate_faithful_tabm_batchensemble_cheap_gate_20260816.py` | h48qual 3개 regime expert, 단일시드 260816 | `ThreeHeadTabMFull`(R+S+B) vs 기존 `ThreeHeadTabM`(R only) 분류지표 비교 | 검증 완료 — 부정 결과(혼재) | `_prepare_frames_light()`가 죽은 LSTM/chronos 체인을 우회해 라벨을 직접 읽음 — feature_cols 185개(라이브 102개와 다름), 방향성 확인용이지 피처 완전 동일성 재현 아님 |
| Step A cheap_gate 리포트 | `tmp/causal_regen_20260516/eth_candidate_faithful_tabm_batchensemble_cheap_gate_20260816/report.json` | 위와 동일 | 3 expert × 2 arch의 val_loss/quality_val_loss/exit_val_loss/direction_balanced_accuracy 원본 수치 | 활성 | `docs/experiments/eth_candidate_faithful_tabm_batchensemble_20260816.md`에 요약 |

| N≥5 시드 재현 스크립트 | `scripts/research_eth_candidate_faithful_tabm_batchensemble_nseed_20260816.py` | h48qual 3개 regime expert, 시드 5개(`[211581, 262041, 393534, 646498, 707258]`, secrets.randbelow 추출) | Step A 단일시드 혼재 결과가 노이즈인지 재확인 | 검증 완료 — 부정 결과(일관된 악화) | 프레임/exit 데이터셋 준비를 5시드 전체가 1회만 공유(비용 절감), `research_eth_candidate_faithful_tabm_batchensemble_cheap_gate_20260816.py`의 `_fit_one`/`ThreeHeadTabMFull`을 import해서 재사용 |
| N≥5 시드 리포트 | `tmp/causal_regen_20260516/eth_candidate_faithful_tabm_batchensemble_nseed_20260816/report.json` | 위와 동일, 30회 학습(5시드×2아키텍처×3expert) | 시드별 원본 지표 + 집계(mean/std/부호일치 카운트) | 활성 | `docs/experiments/eth_candidate_faithful_tabm_batchensemble_20260816.md`에 요약 |

## 미검증 후보 / 보류

- **Step A(BatchEnsemble R+S+B 완성) cheap_gate**: **완료** — 3 expert × 4지표에서 부호 불일치(단일시드 노이즈 패턴), 계약서 cheap_gate 기준 미통과.
- **N≥5 시드 재현**: **완료, CLOSED** — 사용자 지시로 진행. 노이즈가 아니라 direction_balanced_accuracy에서 일관된 악화(chop 5/5, bull 4/5)로 판명. R+S+B 완성판(+6.5% 파라미터)이 약신호 데이터셋에서 과적합을 가속한 것으로 추정.
- **Step B(piecewise-linear embedding)**: 착수 안 함 — Step A/N≥5시드 모두 부정 결과라 전제 조건 불충족.
- **zig075 확장**: 착수 안 함 — h48qual 자체가 CLOSED.
