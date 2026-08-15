# Odyssey4 — ETH zig075 진입거부 베이스라인 계약 문서 (2026-08-14)

## 상태

| 컴포넌트 | 상태 |
|---|---|
| **Odyssey4 베이스라인 확정** | `locked (연구 확정, 섀도우 미배포)` — Odyssey3 베이스라인(`asymmetric_tabm_liveatr` + h48qual 레짐인지형 exit 가드) **+ zig075 지속상승장 SHORT 진입거부**(실행 로그 #2, Odyssey3의 지속상승장 탐지기를 신규 자유변수 없이 entry-side veto로 재사용). 이 시점부터 Odyssey4는 이 상태를 새 비교 기준(reference)으로 삼는다. **주의**: 이 진입거부 계층은 아직 **어떤 프로세스에도 배포되지 않았다** — h48qual 레짐 가드처럼 서버 섀도우로 상시 관찰되고 있지 않고, 순수 fresh-forward replay 검증(연구 확정)만 완료된 상태다. 실거래 경로(`trading_bot.py`)는 물론, 서버 섀도우 프로세스도 미변경. "Odyssey4 베이스라인"은 연구 비교용 기준이지, 실거래·섀도우 배포 상태를 뜻하지 않는다. |

## 범위

- 목적: Odyssey3가 확정한 베이스라인(h48qual 레짐 가드) 위에, 사용자가 2026-08-14 세션에서 명시적으로 해제한 entry-side 개입을 공식 결합해 새 비교 기준으로 삼는다.
- **Odyssey1·Odyssey2의 실험 44건(entry-side 실패 29건 포함) 및 Odyssey3 실행 로그 #1(zig075 exit-side 개입 불가 진단) 전부 상속한다 — 재검증 불필요.**
- **유일한 신규 계층은 zig075 SHORT 진입거부뿐이다** — Odyssey3 베이스라인의 h48qual 레짐 가드·zig075 원본 로직·모든 모델 헤드·TP/SL·사이징·priority는 무변경.
- Odyssey(1)·Odyssey2·Odyssey3의 미해결 이슈(VAL 구간 신뢰성, exit_head 섀도우 승격기준 미정, `quality_threshold` 정렬버그 잔여 6개 스크립트, ATR TP/SL floor 버그 여부, 레짐 가드 forward 미검증)는 전부 그대로 상속된다.
- 라이브 파일(`trading_bot.py`/`trading_bot_modules/omega4_6_1_live.py`/`runtime_config.py`/`.env`) 미변경 원칙은 Odyssey(1)·Odyssey2·Odyssey3와 동일하게 유지. 서버 섀도우 프로세스(`live_eth_regime_aware_exit_guard_shadow_20260814.py`)도 이번 세션에서 미변경.
- 리소스 레지스트리: `docs/model_contracts/odyssey4_eth_entry_veto_baseline_data_resources_20260814.md`.
- **전체 레이어 요약·다이어그램**: `docs/model_contracts/odyssey4_eth_full_stack_architecture_20260814.md` — 피처부터 청산·렛저까지 전체 의사결정 파이프라인을 계층별로 정리하고 Odyssey1~4가 각각 어느 계층에 무엇을 추가했는지 시각화한 문서.
- 선행 계약: `docs/model_contracts/odyssey3_eth_regime_guard_baseline_contract_20260814.md`(Odyssey3), `docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md`(Odyssey2), `docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_contract.md`(Odyssey1 — 최상위 서사).

## G0 참조값 (Odyssey4 베이스라인, 실행 로그 #2 `report.json`에서 그대로 재사용 — 재계산 불필요)

| 창 | 티어 | Odyssey3 베이스라인 (no_gate / with_gate) | Odyssey4 (진입거부 p90 적용, no_gate / with_gate) | veto 발동 |
|---|---|---|---|---|
| 2025-Q1(참고) | context | 97.70%/−20.62%/28 · 44.98%/−20.62%/20 | 동일 | 0 bar |
| 2025-Q2(참고) | context | 106.45%/−13.23%/31 · 31.49%/−15.85%/19 | 65.83%/−14.17%/31 · 5.62%/−23.59%/19 | 10 bar |
| **2025-Q3(참고)** | context | −37.43%/−51.25%/27 · **−15.86%/−44.37%/21** | −10.63%/−29.66%/23 · **+20.17%/−19.72%/17** | 19 bar |
| VAL | val | 46.59%/−21.70%/35 · 77.31%/−21.76%/26 | 41.13%/−21.70%/35 · 77.31%/−21.76%/26(동일) | 12 bar |
| OOS-Q1 | oos_confirm | 93.27%/−15.48%/24 · 67.25%/−15.48%/19 | 동일(렛저 자체 동일) | 0 bar |
| OOS-Q2 | oos_confirm | −9.55%/−20.76%/13 · −12.69%/−20.76%/10 | 동일(렛저 자체 동일) | 0 bar |

판정: VAL 게이트 strict 통과, OOS-Q1+OOS-Q2 단일터치 strict/relaxed 모두 `CONFIRMED`. 근거·전체 과정: `docs/experiments/eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.md`.

향후 신규 후보는 이 표를 G0 기준으로 삼는다 — 진입거부 로직은
`scripts/research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.py`의
`greedy_replay_entry_veto`(가드 모듈의 `greedy_replay_regime_aware_exit_guard` 복사본에
`short_entry_veto_mask` 한 줄만 추가)로 재사용 가능.

## 신규 계층 정의 (Odyssey4가 추가한 유일한 것)

flat 상태 진입 루프에서 `component == zig075 && side == SHORT && 지속상승장 탐지기 ON(신호 bar)`이면
그 진입만 스킵한다. 탐지기는 Odyssey3 베이스라인이 이미 잠근 것을 그대로 재사용
(`dual_momentum>0`의 2016-bar rolling 비율, threshold=2025-Q1+Q2 전용 표본 p90=0.8025793650793651,
Q3/VAL/OOS 미참조) — **신규 자유변수 0개**. zig075 LONG·h48qual(레짐 가드 포함)·모든 모델
헤드·threshold·TP/SL·사이징·priority·exit-side는 전부 무변경.

## 다음 점검 대상

| # | 항목 | 근거 |
|---|---|---|
| 1 | ~~진입거부 섀도우 관찰 로깅 추가~~ — **구현·서버 배포 준비 완료(2026-08-14)**: `scripts/live_eth_odyssey4_zig075_entry_veto_shadow_20260814.py`(Odyssey3 h48qual 가드 + zig075 SHORT veto 통합, 서버에서 import/adapter/guard 생성까지 검증 완료) + `eth-odyssey4-shadow.service` + `install_and_cutover_odyssey4_shadow_20260814.sh` 전부 서버에 push됨. **cutover 자체(JM 레짐교체·exit_head 비대칭 두 섀도우 stop+disable, 오디세이4 섀도우 enable+start)는 root sudo가 필요해 코딩 에이전트가 실행 불가**(`deploy_watcher_sudoers`가 stop/disable/enable을 의도적으로 차단) — 사용자가 서버에서 `sudo bash scripts/ops/systemd/install_and_cutover_odyssey4_shadow_20260814.sh` 1회 실행 필요. | 사용자 지시(2026-08-14) |
| 2 | 레짐인지형 exit 가드(h48qual)·진입거부(zig075) 둘 다 forward 관찰 누적 — 섀도우가 실제 지속상승장을 한 번이라도 겪을 때까지는 두 계층 모두 진짜 검증 불가 | Odyssey2 #11·Odyssey4 실행 로그 #2 정직한 한계 |
| 3 | ~~h48qual SHORT에 동일 진입거부 확장~~ — **실행 완료(2026-08-15, 실행 로그 #4)**: CONFIRMED(약함/한계 있음), 참고 3분기 순효과 음수라 배포 권장 안 함 | 실행 로그 #4 |
| 4 | (낮은 우선순위, 상속) VAL 구간 신뢰성 근본원인, exit_head 승격기준 확정 | Odyssey1 미해결 이슈 12·13 |
| 5 | zig075 LONG 지속하락장 진입거부(실행 로그 #5, CONFIRMED)의 forward 관찰 누적 — 판정 근거가 OOS-Q2 거래 1건에 의존해 섀도우 배포 없이는 확신 상향 불가 | 실행 로그 #5 정직한 한계 |
| 6 | ~~zig075 LONG/하락장 손실의 bar-level 메커니즘 진단~~ — **실행 완료(2026-08-15, 실행 로그 #6)**: exit_head 0/33건 관여(SHORT 0/53건과 합쳐 방향 불문 0/86건), 방향/품질 확신도도 승패 미분리 — entry veto가 유일하게 유효했던 개입임을 사후 확인. 다만 실제 표본은 겹침 거래 4건뿐이라 판정 강도 자체는 못 올림 | 실행 로그 #6 |

## 실행 로그

Odyssey3까지의 실행 로그는 `docs/model_contracts/odyssey3_eth_regime_guard_baseline_contract_20260814.md`
참고. #1은 이 계약이 흡수한 규칙 veto(위 G0 표).

| # | 항목 | 결과 | 문서 |
|---|---|---|---|
| 1 | zig075 SHORT 진입거부 (규칙, 탐지기 재사용) | **CONFIRMED** — Odyssey4 베이스라인으로 흡수(위 G0 표) | `docs/experiments/eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.md` |
| 2 | **학습형 진입거부** (사용자 지시: RL/딥러닝) — TCN이 반사실 barrier 라벨(SL-first)로 거부 게이트를 학습, 2024만 학습·HP탐색 0·임계값은 손익분기 유도(p\*=0.6435)·5시드 | **부정 결과, REJECTED** — VAL strict 통과·시드 5/5 일치에도 OOS-Q1 반전(with_gate 67.25%→−12.70%, 승리 숏 4건 역선택). 메커니즘: 연도 밖 AUC 무작위 이하(2025-Q3 0.498, 2026 0.477), mask 발동 36~58%의 광역 필터로 퇴행 — 2025 내 개선은 판별이 아닌 기저율 효과. **규칙 veto가 유효한 해로 유지되며, "정보 부족" 가설이 모델 비개입 라벨·밀집 샘플 조건에서도 재확인됨(30번째 entry-side 학습 실패).** | `docs/experiments/eth_omega461_zig075_learned_short_veto_tcn_20260814.md` |
| 3 | **증거신호 사이징** (macro veto 위 계층, 별도 연구 라인의 외부 OHLCV+taker_buy 신호를 사이징 신호로 재사용 — entry 거부 아님) — TOP 증거신호 미확인 시 zig075 SHORT margin_fraction×0.5, 확인 시 그대로. v1(신호 8개)과 v2(순위안정성 문서 마스터랭킹 상위5개로 정제, 배율 불변) 둘 다 시도 | **부정 결과, REJECTED (양쪽 다)** — v1: VAL strict 탈락(77.31%→64.07%)으로 OOS 미개봉. v2(신호 목록만 정제, 배율은 그대로 0.5 유지): **더 악화**(VAL 77.31%→43.71%, 목표였던 2025-Q3조차 베이스라인 20.17%보다 낮은 17.68%로 반전). 원인: 신호를 좁힐수록 "OR 확인" 조건이 더 드물어져(확인율 2~13%) sized_down 거래가 늘어나는 구조적 문제 — 신호 품질을 아무리 높여도 이분법적 사이징 규칙 자체가 문제라 개선 안 됨. 배율 재조정은 사후선택이라 하지 않음. **오디세이4 macro veto는 이 결과와 무관하게 CONFIRMED 유지.** 이 축은 종결. | `docs/experiments/eth_omega461_zig075_short_evidence_sizing_20260814.md` |
| 4 | **h48qual SHORT 지속상승장 진입거부 확장** (다음 점검 대상 #3, zig075판과 동일 탐지기·임계값을 h48qual SHORT에 재사용, 자유변수 0개) | **CONFIRMED (약함/한계 있음)** — 사전등록 게이트는 형식상 통과하지만 판정 3창 중 VAL·OOS-Q2는 veto 발동 0건(무해성뿐), OOS-Q1은 거래 1건 교체(+1.35pp, 표본 1건). 참고 3분기 순효과가 오히려 음수(Q3 개선 +3.24pp < Q2 비용 −9.09pp, 순 −5.85pp) — zig075판(Q3 부호반전 +36pp급)과 대조적으로 존재 이유가 약함. **배포/섀도우 후보로 권장하지 않음.** 라이브 무변경. | `docs/experiments/eth_omega461_h48qual_short_entry_veto_sustained_uptrend_20260815.md` |
| 5 | **zig075 LONG 지속하락장 진입거부** (Odyssey4 베이스라인 위에 얹는 신규 후보, 상승장 탐지기의 거울상 — `dual_momentum<0` rolling 비율, 동일 레시피로 새로 계산한 p90=0.9712301587301587, 신규 자유변수 0개이나 산출 상수는 신규) | **CONFIRMED** — VAL·OOS-Q1은 veto 0건(무해성)이지만 **OOS-Q2는 실제 개입**: 37회 발동, 거래 1건 교체(2026-05-23 LONG 손절 −8.38% 제거 → 2026-05-24 SHORT 익절 +13.64%로 대체)로 `with_gate` PnL −12.69%→**+8.30%**(부호반전), MDD −20.76%→−13.72%(개선). 참고 2025-Q1도 큰 개선(+54pp). 다만 강건성이 zig075 SHORT판만큼 깨끗하지 않음(p75로 완화 시 Q3 참고창이 무변화→악화로 전환)과, 판정 근거가 사실상 판정 3창 중 1창·거래 1건에 의존한다는 한계가 있어 **섀도우 배포 최우선순위로 격상하기엔 이름**. LONG/하락장 손실의 bar-level 메커니즘 진단은 실행 로그 #6에서 사후 완료. 라이브 무변경. | `docs/experiments/eth_omega461_zig075_long_entry_veto_sustained_downtrend_20260815.md` |
| 6 | **zig075 LONG/하락장 손실 메커니즘 진단** (실행 로그 #1(SHORT/상승장)과 동형: exit_head 관여 여부 + bar-by-bar MFE/확률 재구성 + 반사실 threshold 스윕, 신규로 방향/품질 확신도 승패분리 검사 추가) | **`diagnosed`** — exit_head는 지속하락장 탐지기와 겹친 4건(전 6창 합산, 실제 체결까지 이어진 표본)에서 단 한 번도 관여하지 않음(0/33, LONG 전체 기준) — SHORT판 0/53건과 합쳐 zig075 exit_head는 방향 불문 0/86건으로 확정. 원칙 검증 범위(threshold≥0.80)의 반사실 exit-threshold는 전부 무반응, 그 아래(사후선택)로 내리면 손실거래 일부는 개선되지만 유일한 승리거래(2025-Q2)를 항상 해치는 SHORT판과 동일한 혼재 패턴. 방향/품질 확신도(dir_p_long/quality_for_action)도 1승3패를 구분 못함(승자값이 패자값 구간 한복판). **entry veto가 유일하게 유효했던 개입이라는 실행 로그 #5의 설계를 사후 지지**하지만, 표본 자체가 4건뿐이라 실행 로그 #5의 판정 강도는 이 진단으로 올라가지도 내려가지도 않음. | `docs/experiments/eth_omega461_zig075_long_downtrend_loss_mechanism_diagnosis_20260815.md` |

## 미해결 이슈

Odyssey(1)·Odyssey2·Odyssey3에서 상속(전부 유효):

- VAL 구간(2025-10~12) 자체의 신뢰성 문제 — Odyssey(1) 미해결 이슈 12.
- exit_head 섀도우 관찰기간·승격 판단기준 미정 — Odyssey(1) 미해결 이슈 13.
- `quality_threshold` 정렬버그, 동일 코드가 있는 미수정 6개 스크립트 — Odyssey(1) 미해결 이슈 14.
- ATR TP/SL floor가 버그인지 의도인지 — Odyssey(1) 미해결 이슈 15.
- h48qual 레짐 가드·zig075 진입거부 둘 다 **forward에서 진짜 지속 상승장을 한 번도 겪지 않았다**
  (OOS 데이터는 2026-07-12까지, 유일한 상승 구간은 12일로 표본 부족) — 두 계층 모두 "관찰 대기"
  지위.

## 승격 게이트

Odyssey(1)·Odyssey2·Odyssey3와 동일하게 적용:

- VAL 단독 승리는 승격 근거 아님 — 저비용 사전필터로만.
- 신규 post-entry/entry 후보는 VAL 자체 게이트 통과 후, 공식 OOS 확인을 OOS-Q1+OOS-Q2를 한 실행에서
  함께 여는 단일터치로 심사한다(`scripts/eth_omega461_multiwindow_confirmation_gate_20260814.py`,
  순차/반복 확인 금지).
- exit_head/entry 로직 자체를 바꾸는 실험에는 컴포넌트 가드레일(50% 상대악화·부호반전 금지) 적용.
- 재학습 모델은 N≥5개 진짜 다양한 시드 없이 신호/노이즈 판정하지 않는다(결정론적 룰 기반 개입은
  해당 없음 — 시드 축 자체가 존재하지 않음).
- 라이브 파일 무변경, 섀도우 배포 ≠ 승격. **섀도우 배포 자체도 아직 이뤄지지 않았음에 유의**
  (Odyssey3의 h48qual 가드와 달리 zig075 진입거부는 관찰 로깅조차 없는 순수 연구 결과).
