# Odyssey4 — 데이터 및 리소스 관리 (2026-08-14)

이 문서는 Odyssey4 서브 프로젝트(`docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md`)에서 새로 만지는 리소스만 모은다. Odyssey(1)·Odyssey2·Odyssey3의 리소스(`docs/model_contracts/odyssey_eth_h48qual_data_resources_20260812.md`, `odyssey2_eth_live_injection_data_resources_20260813.md`, `odyssey3_eth_regime_guard_baseline_data_resources_20260814.md`)는 대부분 그대로 재사용한다.

**새 리소스를 만질 때마다 그 턴에 행을 추가/갱신할 것** — 나중으로 미루지 않는다.

## 코드 (스크립트 · 문서)

| 파일 | 용도 | 상태 |
|---|---|---|
| `docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md` | Odyssey4 베이스라인 계약 — Odyssey3 베이스라인 + zig075 SHORT 진입거부 결합, G0 참조값 확정 | 완료(부트스트랩) |
| `docs/model_contracts/odyssey4_eth_full_stack_architecture_20260814.md` | 전체 레이어 요약·다이어그램 — 피처~렛저 전체 파이프라인, Odyssey1~4 세대별 추가 계층 시각화 | 완료 |
| `scripts/research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.py` | Odyssey4의 유일한 신규 계층 구현 — Odyssey3 리소스로 이미 등록됨(실행 로그 #1), 이 문서는 재사용만 함 | 완료(CONFIRMED, Odyssey3 소유) |
| `scripts/train_eval_eth_omega461_zig075_learned_short_veto_tcn_20260814.py` | 실행 로그 #2 — 학습형 진입거부(TCN, 2024만 학습, 반사실 barrier 라벨, 5시드, 손익분기 임계값) 학습+평가 | 완료(부정 결과, REJECTED) |
| `docs/experiments/eth_omega461_zig075_learned_short_veto_tcn_20260814.md` | 실행 로그 #2 결과 문서 — OOS-Q1 반전 메커니즘(연도 밖 AUC 무작위 이하, 기저율 효과) 포함 | 완료(부정 결과) |
| `tmp/causal_regen_20260516/eth_omega461_zig075_learned_short_veto_tcn_20260814/` | 실행 로그 #2 산출물 — 모델 번들(5시드)·train_report·평가 report.json·렛저 | 완료 |
| `data/splits/year_oos/training_features_2024.csv` | 실행 로그 #2 학습 데이터(2024 전체, Omega4.6.1 계보 최초 사용 — 2025/2026 완전 홀드아웃 확보용) | 활성(서버에도 sync됨) |

## Odyssey3에서 그대로 재사용하는 핵심 인프라 (참고용, 이 문서 소유 아님)

| 리소스 | 위치 | 용도 |
|---|---|---|
| 다중구간 확인 게이트 | `scripts/eth_omega461_multiwindow_confirmation_gate_20260814.py` | VAL+OOS-Q1+OOS-Q2 단일터치 판정, Odyssey4도 그대로 사용 |
| 지속상승장 탐지기 (신규 자유변수 0개로 재사용) | `scripts/research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.py` (`build_detector`) | h48qual exit 가드(Odyssey3)와 zig075 진입거부(Odyssey4)가 공유하는 단일 탐지기 소스 |
| zig075 SHORT 진입거부 replay | `scripts/research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.py` | Odyssey4 베이스라인의 실제 구성요소 |
| 레짐인지형 가드 라이브 섀도우 | ~~`scripts/live_eth_regime_aware_exit_guard_shadow_20260814.py`~~ | **은퇴(2026-08-15)** — `eth-odyssey4-shadow`가 이 가드 로직을 byte-for-byte 포함하는 상위호환이라 중복 판단, 사용자가 직접 kill. 관찰은 아래 `eth-odyssey4-shadow`로 완전히 계승됨 |

## 신규 리소스 (섀도우 배포, 2026-08-14 구축 → 2026-08-15 cutover 완료)

| 파일 | 용도 | 상태 |
|---|---|---|
| `scripts/live_eth_odyssey4_zig075_entry_veto_shadow_20260814.py` | Odyssey4 통합 섀도우 — h48qual 레짐인지형 exit 가드(오디세이3, 무변경) + zig075 SHORT 지속상승장 진입거부(오디세이4 #1) 단일 프로세스. `live_eth_jmlam4_regime_swap_shadow_20260809.py`·`live_eth_exithead_asymmetric_shadow_20260813.py` 대체 | **완료** — 2026-08-15 사용자가 cutover 실행, 코딩 에이전트가 read-only SSH로 사후 검증(active+enabled, 08-14 23:40부터 무중단, 초기화 로그 임계값 계약과 일치) |
| `scripts/ops/systemd/eth-odyssey4-shadow.service` | 신규 systemd 유닛(기존 두 유닛과 동일 패턴) | **완료** — active+enabled 확인됨 |
| `scripts/ops/systemd/install_and_cutover_odyssey4_shadow_20260814.sh` | 1회 cutover 스크립트 — 신규 유닛 설치 + 기존 두 섀도우 stop/disable + 신규 섀도우 enable/start | **완료** — 사용자가 서버에서 직접 실행(2026-08-15), 검증 완료. `eth-jmlam4-shadow`/`eth-exithead-shadow` 유닛 둘 다 목록에서 완전히 사라짐 |

## 미검증 후보 / 보류

- h48qual SHORT로의 진입거부 확장 — 미검토, 낮은 우선순위(Odyssey4 계약 "다음 점검 대상 #3").
- `live_eth_regime_aware_exit_guard_shadow_20260814.py`(Odyssey3 자체 섀도우)는 이번 cutover 대상에서 제외 — 사용자가 명시적으로 요청한 두 섀도우(JM 레짐교체·exit_head 비대칭)만 제거. 오디세이4 섀도우 가동 후 h48qual 가드 관찰이 두 프로세스에서 중복되므로, Odyssey3 섀도우 자체 종료 여부는 별도 결정 필요.
