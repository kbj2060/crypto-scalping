# ETH 동시-2슬롯 포지션 후보 — 데이터 및 리소스 관리 (2026-08-16)

이 문서는 이 후보(`docs/model_contracts/eth_candidate_dual_slot_concurrent_positions_contract_20260816.md`)에서
실제로 만지거나 검토한 모든 데이터 소스/리소스를 한 곳에 모은 목록이다.

**새 리소스를 만질 때마다 그 턴에 행을 추가/갱신할 것** — 나중으로 미루지 않는다. 상태 값 컨벤션: `활성`, `인프라 확인됨-미착수`, `인프라 차단`, `검증 완료 — 긍정 결과`, `검증 완료 — 부정 결과`.

## 상속 자산 (그대로 재사용)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| Odyssey4 causal replay 하네스 | `scripts/research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.py` | 6창 | G0 기준값, 컴포넌트 준비 함수 재사용 | 활성 | 신규 replay 엔진의 베이스로 복사 예정(아직 미작성) |
| zig075 episode 라벨(재사용 대상) | `tmp/causal_regen_20260516/eth_candidate_conformal_veto_episode_labels_20260816/episode_labels_<window>_zig075.parquet` | 6창(단, cheap_gate는 VAL만 사용) | cheap_gate의 counterfactual 상한선 추정 원재료 | 활성 | conformal veto 후보가 만든 산출물 — 재시뮬레이션 없이 그대로 재사용, entry_signal_i+hold_bars로 h48qual 보유구간과 대조 |
| 우선순위 스왑/max-hold cheap_gate 결과 | `docs/experiments/eth_candidate_priority_swap_cheap_gate_20260816.md`, `docs/experiments/eth_candidate_h48qual_max_hold_cheap_gate_20260816.md` | VAL | 이 후보로 오게 된 근거(두 대안 기각 경위) | 활성 | 계약 "여기까지 오게 된 경위" 절에 요약 인용됨 |

## cheap_gate 산출물 (2026-08-16)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| cheap_gate 스크립트 | `scripts/research_eth_candidate_dual_slot_cheap_gate_20260816.py` | VAL, zig075 189건 중 h48qual에 막힌 185건 | 재사용 라벨 기반 상한선 추정 | 검증 완료 — 부정 결과(상한선 자체가 −149.91%) | G0 재현 값(no_gate 41.13%)까지 이 스크립트가 내부적으로 확인함 |
| cheap_gate 리포트 | `tmp/causal_regen_20260516/eth_candidate_dual_slot_cheap_gate_20260816/report.json` | - | 원 수치 근거 | 검증 완료 — 부정 결과 | `docs/experiments/eth_candidate_dual_slot_cheap_gate_20260816.md`에 표로 요약됨 |

## 미검증 후보 / 보류

- **동시-2슬롯 replay 엔진**: **미착수 — 착수 안 함**(cheap_gate 상한선이 음수라 계약 종결, 위 표 참고).
