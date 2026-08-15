# Odyssey3 — 데이터 및 리소스 관리 (2026-08-14)

이 문서는 Odyssey3 서브 프로젝트(`docs/model_contracts/odyssey3_eth_regime_guard_baseline_contract_20260814.md`)에서 새로 만지는 리소스만 모은다. Odyssey(1)·Odyssey2의 리소스(`docs/model_contracts/odyssey_eth_h48qual_data_resources_20260812.md`, `odyssey2_eth_live_injection_data_resources_20260813.md`)는 대부분 그대로 재사용한다.

**새 리소스를 만질 때마다 그 턴에 행을 추가/갱신할 것** — 나중으로 미루지 않는다.

## 코드 (스크립트 · 문서)

| 파일 | 용도 | 상태 |
|---|---|---|
| `docs/model_contracts/odyssey3_eth_regime_guard_baseline_contract_20260814.md` | Odyssey3 베이스라인 계약 — Odyssey2 베이스라인 + 레짐인지형 가드(Odyssey2 #11) 결합, G0 참조값 확정 | 완료(부트스트랩) |
| `scripts/research_eth_omega461_zig075_sustained_uptrend_diagnosis_20260814.py` | 실행 로그 #1 — zig075 Q3 SHORT 약세 메커니즘 진단(렛저 재집계 + 원시 exit_head 확률 bar-by-bar 재계산·자체검증 + threshold 반사실 분석) | 완료(부정 결과) |
| `docs/experiments/eth_omega461_zig075_sustained_uptrend_guard_20260814.md` | 실행 로그 #1 결과 문서 — exit_head가 zig075 SHORT에서 분기 불문 구조적으로 거의 관여하지 않음을 확인, 유효한 post-entry 개입 없어 설계 불가로 종결 | 완료(부정 결과) |
| `tmp/causal_regen_20260516/eth_omega461_zig075_sustained_uptrend_diagnosis_20260814/report.json` | 실행 로그 #1 산출물 — 분기별(2025 Q1/Q2/Q3) zig075 SHORT 렛저 분해 + 53건 거래 각각의 bar-by-bar exit_head 확률 궤적 + 자체검증 결과 | 완료 |
| `scripts/research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.py` | 실행 로그 #2 — zig075 SHORT entry veto(베이스라인 탐지기 p90 재사용, 자유변수 0개): G0a/G0b 무결성 + 6창 fresh-forward replay + p75/p95 강건성 + 다중구간 판정 | 완료(CONFIRMED) |
| `docs/experiments/eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.md` | 실행 로그 #2 결과 문서 — VAL/OOS 무손상 + Q3 with_gate 부호 반전(−15.86%→+20.17%), 정직한 한계(참고 티어, forward 미검증) 포함 | 완료(CONFIRMED) |
| `tmp/causal_regen_20260516/eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814/report.json` | 실행 로그 #2 산출물 — 창별 베이스라인/후보 지표, veto 이벤트 전체, per-trade 렛저 diff, 강건성, 판정 | 완료 |
| `scripts/train_eth_omega461_exit_head_liveatr_relabel_walkforward_fold_20260814.py` | 실행 로그 #3 — h48qual exit_head liveATR 재라벨 원본 스크립트를 함수 단위로 재사용(미수정)하되 학습구간(`--train-start`/`--train-end`)만 파라미터화한 워크포워드 폴드 학습 스크립트 | 완료(NOT_ROBUST) |
| `scripts/eval_eth_omega461_exit_head_liveatr_relabel_walkforward_20260814.py` | 실행 로그 #3 — 4폴드(A~D) 컴포넌트·포트폴리오 평가 스크립트, G0 자체검증(폴드A 기존 공개 수치 재현) 포함 | 완료(NOT_ROBUST) |
| `docs/experiments/eth_omega461_exit_head_liveatr_relabel_walkforward_20260814.md` | 실행 로그 #3 결과 문서 — 4폴드 결과표, `TABM_2025`/`TABM_2026` 데이터 커버리지 한계 발견, 시드축(JM) 대비 해석 | 완료(NOT_ROBUST) |
| `tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_walkforward_fold_20260814_foldB/`, `..._foldC/`, `..._foldD/` | 실행 로그 #3 산출물 — 폴드별 h48qual/zig075 재학습 번들(`true_3head_tabm_bundle.pt`) + `report.json`(foldC는 서버에서 학습 후 `handoff.sh pull`로 회수, md5 일치 확인) | 완료 |
| `tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_walkforward_20260814/report.json` | 실행 로그 #3 산출물 — G0 자체검증 + 4폴드 컴포넌트/포트폴리오 전체 수치 + 판정 | 완료 |
| `docs/experiments/eth_omega461_exit_head_liveatr_relabel_walkforward_mechanism_diagnosis_20260815.md` | 실행 로그 #4 — 실행 로그 #3이 저장한 산출물(4폴드 report.json)만 재분석, 신규 학습·스크립트 없음: exit_head 발동 행태(발동률·승률·보유시간)가 4폴드 전부 동일함을 확인, 라벨 구성(`mfe_giveback_exit` 75.7~79.8%) 및 p95_trade_pnl 훼손폭이 승패를 가른 메커니즘 규명 | 완료 |

## Odyssey2에서 그대로 재사용하는 핵심 인프라 (참고용, 이 문서 소유 아님)

| 리소스 | 위치 | 용도 |
|---|---|---|
| 다중구간 확인 게이트 | `scripts/eth_omega461_multiwindow_confirmation_gate_20260814.py` | VAL+OOS-Q1+OOS-Q2 단일터치 판정, Odyssey3도 그대로 사용 |
| 레짐인지형 가드 로직 | `scripts/research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.py` | `greedy_replay_regime_aware_exit_guard`/탐지기(`dual_momentum`, threshold=0.802579) — Odyssey3 베이스라인의 실제 구성요소 |
| 레짐인지형 가드 라이브 섀도우 | `scripts/live_eth_regime_aware_exit_guard_shadow_20260814.py` | 서버 상시 실행 중, Odyssey3 베이스라인의 forward 관찰 소스 |

## 미검증 후보 / 보류

- (해소됨 — 실행 로그 #1) zig075에 동일 지속상승장 가드 적용은 `eth_omega461_zig075_sustained_uptrend_guard_20260814.md`에서 부정 결과로 종결. zig075의 exit_head가 SHORT 포지션에서 왜 구조적으로 확신도가 낮게 캘리브레이션됐는지(분기 불문) 자체는 여전히 미검증 — 향후 exit_head 재학습 등 다른 각도를 검토할 후보로 남김.
