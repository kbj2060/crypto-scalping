# Odyssey3 — ETH 레짐인지형 가드 베이스라인 계약 문서 (2026-08-14)

## 상태

| 컴포넌트 | 상태 |
|---|---|
| **Odyssey3 베이스라인 확정** | `locked` — Odyssey2 베이스라인(`exit_head` 비대칭 재라벨, h48qual만 교체·zig075 원본 유지) **+ 레짐인지형 지속상승장 가드**(Odyssey2 #11, `dual_momentum` 기반 causal 탐지기로 지속 상승 레짐에서만 h48qual의 exit_head를 원본으로 되돌림). 이 시점부터 Odyssey3는 이 상태를 새 비교 기준(reference)으로 삼는다. **주의**: Odyssey2와 동일하게 섀도우(관찰 전용, 페이퍼)로만 검증됐다. 최초 관찰원이었던 `scripts/live_eth_regime_aware_exit_guard_shadow_20260814.py`(서버 상시 실행)는 **2026-08-15 Odyssey4 섀도우 cutover 후 중복으로 판단돼 사용자가 직접 종료** — 이 가드 로직은 그것을 byte-for-byte 포함하는 `eth-odyssey4-shadow`(`live_eth_odyssey4_zig075_entry_veto_shadow_20260814.py`)로 관찰이 계승됐다(Odyssey4 계약 실행 로그 #1 참고). 실제 `trading_bot.py` 라이브 의사결정 경로는 원본 그대로다 — "Odyssey3 베이스라인"은 연구 비교용 기준이지, 실거래 상태를 뜻하지 않는다. |

## 범위

- 목적: Odyssey2가 확정한 베이스라인 위에 레짐인지형 가드를 공식 결합해 새 비교 기준으로 삼고, 이 기준에서 추가로 무엇을 손볼 수 있는지 재점검한다.
- **Odyssey2의 실험 15건(레짐별 quality_threshold 3종·앙상블/오토인코더 사이징 2종·GBDT/TCN exit_head 전면교체 2종·대기압력/risk-controlled/SCRC exit-timing 조건부 3종·Conformal Kelly 사이징·zig075 exit_threshold·Gittins Index Deep RL) 전부 부정 결과 그대로 상속한다 — 재검증 불필요.** 근거: Odyssey3 베이스라인은 VAL·OOS-Q1·OOS-Q2 세 창 전부에서 Odyssey2 베이스라인과 수치까지 완전히 동일하다(가드가 이 세 창에서 실제 결정을 단 한 번도 바꾼 적이 없음을 #11이 직접 확인) — 즉 Odyssey2에서 내려진 모든 판정은 Odyssey3에서도 그대로 유효하다.
- **유일한 실질적 차이는 2025-Q3(지속 상승장, 참고용 context tier)뿐이다** — 이 창에서만 가드가 손상을 부분 완화한다(아래 G0 참조값 표).
- Odyssey(1)·Odyssey2의 미해결 이슈(VAL 구간 신뢰성, exit_head 섀도우 승격기준 미정, `quality_threshold` 정렬버그 잔여 6개 스크립트, ATR TP/SL floor 버그 여부)는 전부 그대로 상속된다.
- 라이브 파일(`trading_bot.py`/`trading_bot_modules/omega4_6_1_live.py`/`runtime_config.py`/`.env`) 미변경 원칙은 Odyssey(1)·Odyssey2와 동일하게 유지.
- 리소스 레지스트리: `docs/model_contracts/odyssey3_eth_regime_guard_baseline_data_resources_20260814.md`.
- 선행 계약: `docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md`(Odyssey2 — 15건 실험 전체 서사), `docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_contract.md`(Odyssey1 — 최상위 서사).

## G0 참조값 (Odyssey3 베이스라인, Odyssey2 #11 `report.json`에서 그대로 재사용 — 재계산 불필요)

| 창 | no_gate | with_gate | Odyssey2 대비 |
|---|---|---|---|
| VAL | 46.59%/-21.70%/35건 | 77.31%/-21.76%/26건 | 동일 |
| OOS-Q1 | 93.27%/-15.48%/24건 | 67.25%/-15.48%/19건 | 동일 |
| OOS-Q2 | -9.55%/-20.76%/13건 | -12.69%/-20.76%/10건 | 동일 |
| 2025-Q1(참고) | 97.70%/-20.62%/28건 | 44.98%/-20.62%/20건 | 동일 |
| 2025-Q2(참고) | 106.45%/-13.23%/31건 | 31.49%/-15.85%/19건 | 동일 |
| **2025-Q3(참고)** | **-37.43%/-51.25%/27건** | **-15.86%/-44.37%/21건** | **다름**(원 Odyssey2 -46.26%/-56.94%/38건·-18.87%/-43.49%/30건 대비 손상 완화) |

향후 신규 후보는 이 표를 G0 기준으로 삼는다 — VAL/OOS-Q1/OOS-Q2는 Odyssey2와 정확히 같은 숫자이므로, 기존 `eth_omega461_multiwindow_confirmation_gate_20260814.py`의 `load_all_windows`가 반환하는 h48qual 컴포넌트에 가드를 씌우는 것만 다르다(가드 로직은 `research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.py`의 `greedy_replay_regime_aware_exit_guard`/탐지기 재사용).

## 다음 점검 대상

Odyssey2에서 15건이 소진된 뒤 남는 것은, 이번 베이스라인 전환 자체가 새로 여는 각도들이다:

| # | 항목 | 근거 |
|---|---|---|
| ~~1~~ | ~~zig075에 동일 지속상승장 가드 적용~~ — **실행 로그 #1에서 종결(부정 결과)**: zig075의 exit_head는 분기 불문 구조적으로 거의 관여하지 않아 h48qual식 "모델 전환" 가드를 옮길 수 없고, 유효한 exit-side 대안도 없음 | `docs/experiments/eth_omega461_zig075_sustained_uptrend_guard_20260814.md` |
| 2 | 레짐인지형 가드의 forward 관찰 누적 — 섀도우가 실제 지속상승장을 한 번이라도 겪을 때까지는 진짜 검증 불가 | Odyssey2 #11 정직한 한계 |
| 3 | (낮은 우선순위, 상속) VAL 구간 신뢰성 근본원인, exit_head 승격기준 확정 | Odyssey1 미해결 이슈 12·13 |
| 4 | **entry-veto 섀도우 관찰 로깅 추가** — 서버 섀도우(`live_eth_regime_aware_exit_guard_shadow_20260814.py`)가 탐지기를 이미 bar마다 계산 중이므로 "이 bar에서 zig075 SHORT 진입이 떴다면 veto됐을 것" 관찰만 추가하면 됨. **서버 상시 실행 프로세스 수정이므로 배포는 별도 결정**(이 세션에서는 미실행) | 실행 로그 #2 후속 |
| 5 | (미검토, 낮은 우선순위) h48qual SHORT에 동일 veto 확장 — h48qual은 exit-side 가드가 이미 회전 가속을 완화 중이라 우선순위 낮음 | 실행 로그 #2 문서 "다음 단계" |

## 실행 로그

| # | 항목 | 결과 | 문서 |
|---|---|---|---|
| 4 | **h48qual exit_head 라이브 ATR 재라벨 — NOT_ROBUST 메커니즘 진단**(실행 로그 #3 후속, 사후 재분석) | 신규 학습·신규 replay 없이 실행 로그 #3이 이미 저장한 4폴드 산출물만 재분석. **원인**: 재라벨은 4개 폴드 전부에서 사실상 동일한 정책을 배운다(exit_head 발동 82~96%, 승률 -8.3~-30.7pp, 평균 보유시간 3.2~9.9배 압축 — 승리한 폴드A도 예외 아님). 라벨 구성도 4개 폴드 전부 거의 동일하다(양성률 18.6~19.9%, 그중 **75.7~79.8%가 `mfe_giveback_exit`** — MFE 0.6% 도달 후 65% 반납이면 발동하는 국소·후향적 되돌림 휴리스틱; 진짜 손절 신호인 `adverse_unreal_exit`는 17.6~22.5%뿐). 승패를 가른 유일한 지표는 p95_trade_pnl 훼손폭(승리 A -10.0% vs 패배 B/C/D -32~-78%) — **폴드A의 "승리"는 exit_head가 더 똑똑해서가 아니라, 그 확인구간(VAL)의 큰 승리거래들이 우연히 giveback 트리거 이후 upside가 적었을 뿐**이다. 학습량 부족 가설 기각(C/D는 A보다 1.6~2.0배 많은 데이터로 학습했지만 둘 다 패배). 섀도우 존속 여부에 새 액션을 요구하지 않음(같은 날 cutover로 은퇴 진행 중) — 다만 향후 이 레시피(라이브 ATR-adaptive 배리어+MFE giveback 재라벨)를 재시도한다면 구간·데이터량이 아니라 `mfe_giveback_exit` 트리거 자체(임계값 상향 또는 조건 제거)의 재설계가 먼저 필요함을 기록. | `docs/experiments/eth_omega461_exit_head_liveatr_relabel_walkforward_mechanism_diagnosis_20260815.md` |
| 3 | **h48qual exit_head 라이브 ATR 재라벨 — 워크포워드 재학습 강건성 검증**(Odyssey2 실행 로그 서사, 이미 섀도우 배포 중인 `live_eth_exithead_asymmetric_shadow_20260813.py`의 근거 레시피 대상) | **강건성 부재 확인**(`NOT_ROBUST`) — 원본 스크립트(`research_eth_omega461_exit_head_liveatr_relabel_20260813.py`)를 함수 단위로 재사용(미수정)하되 학습구간만 파라미터화해 3개 독립 창(2025-Q3/OOS-Q1/OOS-Q2, `load_all_windows()` 재사용)에서 처음부터 재학습. 컴포넌트·포트폴리오·no_gate·with_gate 전 지표에서 "재라벨이 원본 exit_head를 이긴다"는 패턴이 재현된 것은 미재학습 폴드(원 학습 실행 그 자체) 1건뿐 — 진짜 독립 재학습 3건은 전부 예외 없이 재라벨이 원본보다 나빴다(폴드D는 PnL 부호까지 반전). JM N=5-시드 사례(1~2/5 재현)보다도 명확한 실패(0/3). G0 자체검증으로 평가 파이프라인이 이미 공개된 폴드A 수치를 소수점까지 정확히 재현함을 확인해 신뢰성 확보. 실행 중 `TABM_2025`(2025-05-08부터)/`EVAL_CSV`+`TABM_2026`(2026-02-28까지) 커버리지 한계를 발견해 투명하게 보고(폴드 설계 자체는 결과 확인 후 변경하지 않음). 섀도우 배포 자체를 중단시키지는 않음(범위 밖) — forward 관찰 해석 시 이 강건성 부재를 함께 고려할 것을 남김. | `docs/experiments/eth_omega461_exit_head_liveatr_relabel_walkforward_20260814.md` |
| 2 | **zig075 지속상승장 SHORT entry veto** (사용자가 2026-08-14 세션에서 post-entry-only 제약을 명시적으로 해제 — Odyssey 계열 최초의 승인된 entry-side 개입) | **`CONFIRMED`** — 베이스라인 탐지기(p90=0.802579, Q1+Q2 캘리브레이션, Q3 미참조)를 그대로 zig075 SHORT 진입 veto로 재사용(신규 자유변수 0개). VAL 게이트 strict 통과(`with_gate` 완전 동일) + OOS-Q1/Q2 단일터치 strict CONFIRMED(두 창 모두 veto 발동 0건, 렛저 동일 = 무해성 증명). 2025-Q3 참고창: `with_gate` **−15.86%→+20.17% 부호 반전**, MDD −44.37%→−19.72%(7월 지속상승 구간 손절 숏 8건 제거, 슬롯은 h48qual이 승계). 비용: Q2 승리 숏 1건(+0.152) 제거. p75/p90/p95 전 백분위에서 Q3 개선 유지(p95는 Q1/Q2 비용 0으로 동일 Q3 효과 — 효과는 임계값 아티팩트 아님, 단 p90 유지 = 유일한 사전 고정값). **Q3는 참고 티어이므로 진짜 검증은 섀도우 forward 관찰 필요(h48qual 가드와 동일 지위). 베이스라인 정의는 아직 미변경 — 섀도우 관찰 후 결합 여부 결정.** | `docs/experiments/eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.md` |
| 1 | zig075에 동일 지속상승장 가드 적용 가능성 진단 | **부정 결과, 종결**(`diagnosed_no_valid_design`) — zig075 SHORT의 exit_head는 2025 Q1/Q2/Q3 어느 분기·어느 변형에서도 청산사유로 단 한 번도 등장하지 않는다(0/53건, 전부 stop_loss/take_profit) — Q3 국한 현상이 아니라 컴포넌트 전체의 구조적 특성. h48qual과 달리 zig075는 학습된 exit_head가 하나뿐이라 "모델 전환" 메커니즘을 옮길 수 없고, 유일하게 원칙적으로(Q3 배제) 캘리브레이션 가능한 exit_threshold 축은 이미 Odyssey2 #15가 VAL 전 구간(0.80~0.99)에서 강건한 개선 0개로 닫았으며 그 범위에선 Q3에서도 무반응이다. 그보다 낮은 threshold(0.60~0.75)는 Q3 결과를 본 뒤에야 알 수 있는 사후선택이라 탐지기 캘리브레이션과 같은 규율을 위반하고, 그렇게 봐도 Q3 손실거래 69%는 무반응인 채 Q1 승리거래 다수를 해친다. 2단계 개입은 설계하지 않음(entry-side 금지 + 유효한 exit-side 손잡이 없음). Odyssey3 베이스라인은 이 항목에서 변경 없음(h48qual만 가드 적용 유지, zig075는 원본 그대로 미해결). | `docs/experiments/eth_omega461_zig075_sustained_uptrend_guard_20260814.md` |

## 미해결 이슈

Odyssey(1)·Odyssey2에서 상속(전부 유효):

- VAL 구간(2025-10~12) 자체의 신뢰성 문제 — Odyssey(1) 미해결 이슈 12.
- exit_head 섀도우 관찰기간·승격 판단기준 미정 — Odyssey(1) 미해결 이슈 13(Odyssey2 #10이 "지속 상승장 최소 1회 관찰" 기준 후보 제안, 미확정).
- `quality_threshold` 정렬버그, 동일 코드가 있는 미수정 6개 스크립트 — Odyssey(1) 미해결 이슈 14.
- ATR TP/SL floor가 버그인지 의도인지 — Odyssey(1) 미해결 이슈 15.
- ~~zig075의 2025-Q3 숏 약세~~ — **실행 로그 #2에서 연구 수준 해결**(사용자가 post-entry-only 제약 해제 후 entry-veto CONFIRMED). 남은 것은 forward 검증뿐: Q3 개선은 참고 티어(in-sample OOF) 관찰이므로, 섀도우가 실제 지속상승장을 겪기 전까지는 h48qual 가드와 같은 "관찰 대기" 지위다. 베이스라인 결합 여부도 그때 결정.
- **신규(실행 로그 #3)**: h48qual exit_head 라이브 ATR 재라벨(현재 섀도우 배포 중)이 시간축 재학습에 강건하지 않음 — 진짜 독립 재학습 3/3 폴드가 원래 학습 실행의 "재라벨이 원본을 이긴다" 패턴을 재현하지 못했다(0/3). 섀도우 자체는 중단하지 않았으나, forward 관찰이 충분히 쌓일 때까지 이 섀도우의 VAL/OOS 근거를 "확정된 개선"이 아니라 "검증 필요한 관찰"로 취급할 것.

## 승격 게이트

Odyssey(1)·Odyssey2와 동일하게 적용:

- VAL 단독 승리는 승격 근거 아님 — 저비용 사전필터로만.
- 신규 post-entry 후보는 VAL 자체 게이트 통과 후, 공식 OOS 확인을 OOS-Q1+OOS-Q2를 한 실행에서 함께 여는 단일터치로 심사한다(`scripts/eth_omega461_multiwindow_confirmation_gate_20260814.py`, 순차/반복 확인 금지).
- exit_head 모델 자체를 바꾸는 실험에는 컴포넌트 가드레일(50% 상대악화·부호반전 금지) 적용.
- 재학습 모델은 N≥5개 진짜 다양한 시드 없이 신호/노이즈 판정하지 않는다(레포 Seed-Diversity Ensemble Promotion Gate).
- 라이브 파일 무변경, 섀도우 배포 ≠ 승격.
