# zig075 지속상승장 SHORT 진입 거부(entry veto) — Odyssey3 실행 로그 #2 (2026-08-14)

상태: **`CONFIRMED`** — VAL 게이트 strict 통과 + OOS-Q1/OOS-Q2 단일터치 strict/relaxed 동시
CONFIRMED. 판정 3창(VAL/OOS-Q1/OOS-Q2)의 `with_gate` 지표는 베이스라인과 **완전 동일**(OOS 두
창은 veto 발동 0건으로 렛저 자체가 동일)하고, 목표였던 2025-Q3 참고창은 `with_gate` 기준
**−15.86% → +20.17% 부호 반전**, MDD −44.37% → −19.72%. Odyssey 계열 최초의 승인된
entry-side 개입이자, 지속상승장 숏 손실 축에서 나온 최초의 깨끗한 긍정 결과.

## 배경 및 범위 변경 선언

Odyssey(1)·Odyssey2는 과제 지시상 **post-entry 개입만 허용**했고, 그 제약 아래 Odyssey3 실행
로그 #1(`docs/experiments/eth_omega461_zig075_sustained_uptrend_guard_20260814.md`)은 zig075의
2025-Q3 지속상승장 SHORT 약세를 `diagnosed_no_valid_design`으로 종결했다: exit_head는 세 분기
전부 청산 사유로 단 한 번도 등장하지 않고(0/53건), 원칙적으로 캘리브레이션 가능한
exit_threshold 범위(0.80~0.99)는 Q3에서 무반응이며, 그 아래는 사후선택이다.

**2026-08-14 세션에서 사용자가 이 제약을 명시적으로 해제했다**("이제 이 문제를 해결할 때가
왔어. 이 문제를 해결할 베이스라인 모델은 오디세이3 모델이야"). 이 실험은 그 지시에 따른
Odyssey 계열 최초의 승인된 entry-side 개입이다.

## 왜 이 설계인가 — 실패한 entry-side 29건과의 구분

Odyssey1/2가 기록한 entry-side 실패 29건은 전부 **모델 헤드에서 방향/품질 스킬을 만들거나
재선택하려는 시도**였다(재학습·재라벨·재게이팅·재캘리브레이션 — TabM/GBDT/오토인코더/TCN/CNN
× zigzag/trend-scanning/MFE 라벨, 전부 OOS에서 always-short에 패배). "direction_head에 방향
스킬이 없다"는 계약 수준의 확정 사실이고, 스킬 없는 소스의 부분집합 선택은 스킬을 만들지
못한다.

이 설계는 모델 헤드를 전혀 건드리지 않고, 모델 내부 신호로 부분집합을 고르지도 않는다(실행
로그 #1 진단: Q3 승자 3건과 패자 16건의 dir_p_short 0.718~0.825, quality_for_action
0.751~0.825로 내부 신호는 분리 불가; 레짐 라우팅 축도 Q1의 bull 라우팅 숏이 최대 승자라
불가). 대신 **이미 잠긴 외부 causal 레짐 신호로 숏 베타 노출 자체를 관리**한다:

- 탐지기: Odyssey3 베이스라인의 지속상승장 탐지기 그대로 — `dual_momentum>0`의 2016-bar(1주)
  rolling 비율, threshold = 2025-Q1+Q2 전용 표본의 p90 = 0.8025793650793651.
- **신규 자유변수 0개**: 공식·집계창·임계값 전부 상속. Q3/VAL/OOS는 캘리브레이션에 한 번도
  사용되지 않았다(`calibration_excludes_2025q3=true`). Odyssey1/2가 15번 지킨 "목표 구간을
  보지 않고 캘리브레이션" 규율이 그대로 유지된다.
- 라이브 실현 가능성 기증명: 동일 탐지기가 이미 서버 섀도우로 상시 실행 중이며 배치/라이브
  동치성 bar 단위 검증 완료(`verify_eth_regime_aware_exit_guard_shadow_detector_20260814.py`,
  `overall_pass=true`).

## 메커니즘 근거 (실행 로그 #1 진단 walk + 신호 bar 탐지기 오버레이)

| 분기 | 탐지기 ON 진입 (신호 bar 기준) | ON 거래 손익 합 | 나머지 손익 합 |
|---|---|---|---|
| 2025-Q1 | 0/10 | — | +0.653 |
| 2025-Q2 | 1/16 (승리거래 +0.152) | +0.152 | −0.006 |
| 2025-Q3 | **10/19** (손절 9 · 익절 1) | **−0.409** | −0.135 |

Q3 합집합 손실 −0.544의 75%가 탐지기 ON 진입 10건에 집중된다. Q3 손실은 진입타이밍 손실이다
(손절 거래 MFE 중앙값이 SL 거리의 41%, TP 비중 70%→44%→16% 급락) — 진입 자체를 제거하는 것이
남은 유일한 손잡이다. 단, 위 표는 거래 단위 나이브 계산이므로 실제 효과는 fresh-forward
포트폴리오 replay로만 판정했다(zig075 진입 거부 → 공유 슬롯이 h48qual·후속 신호에 풀림).

## 개입 정의

flat 상태 진입 루프에서 `component == zig075 && side == SHORT && 탐지기 ON(신호 bar)`이면 그
진입만 스킵. zig075 LONG·h48qual(레짐인지형 exit 가드 유지)·모든 모델 헤드·threshold·TP/SL·
사이징·priority·캡·exit-side 전부 무변경. 비교 베이스라인 = Odyssey3 베이스라인 전체
(asymmetric_tabm_liveatr + h48qual 레짐 가드 p90).

## 방법 (사전 등록)

신규 스크립트: `scripts/research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.py`

1. **detector_build**: 가드 모듈의 `build_detector()` 그대로 재사용, 재계산된 p90이 잠긴 값과
   1e-12 내 일치하지 않으면 즉시 중단 — **통과**(정확히 일치).
2. **G0a**: 가드 모듈 원본 replay로 Odyssey3 베이스라인(val+oos_q1) 재현 — **통과**.
3. **G0b**: 이 스크립트의 복사본 replay(veto 장치 포함, mask 미부착)가 6개 창 전부에서 계약
   G0 표와 일치(pnl/mdd ±0.05pp, trades 정확히) + veto_bars=0 — **통과**. 이 실행이 판정용
   베이스라인 렛저를 생성한다(후보와 같은 코드 경로, 차이는 veto mask 뿐인 ceteris paribus).
4. **후보 실행**: p90 veto, 6개 창 전부 단일 실행. 렛저 저장 + per-trade diff.
5. **강건성**(참고 전용): veto 임계값만 p75/p95(가드 실험이 사전 등록한 백분위)로 바꿔 2025
   분기 3개 재실행. h48qual exit 가드는 전 구간 p90 고정.
6. **판정**: `gate.summarize_multiwindow` strict(0pp)+relaxed(3pp) — VAL 게이트 통과 후
   OOS-Q1+OOS-Q2 단일터치(`with_gate` PnL·MDD 동시 비악화). 2025 분기는 참고 티어 유지 —
   Q3 개선이 이 실험의 존재 이유지만 판정에는 넣지 않았다.

## 결과

### 판정 — strict CONFIRMED

| 창 | 티어 | Odyssey3 베이스라인 no_gate / with_gate | entry-veto p90 no_gate / with_gate | veto 발동 |
|---|---|---|---|---|
| 2025-Q1 | 참고 | 97.70%/−20.62%/28 · 44.98%/−20.62%/20 | **동일** | 0 bar |
| 2025-Q2 | 참고 | 106.45%/−13.23%/31 · 31.49%/−15.85%/19 | 65.83%/−14.17%/31 · 5.62%/−23.59%/19 | 10 bar |
| **2025-Q3** | 참고 | −37.43%/−51.25%/27 · **−15.86%/−44.37%/21** | −10.63%/−29.66%/23 · **+20.17%/−19.72%/17** | 19 bar |
| VAL | 판정 | 46.59%/−21.70%/35 · 77.31%/−21.76%/26 | 41.13%/−21.70%/35 · **77.31%/−21.76%/26 (동일)** | 12 bar |
| OOS-Q1 | 판정 | 93.27%/−15.48%/24 · 67.25%/−15.48%/19 | **동일 (렛저 자체 동일)** | 0 bar |
| OOS-Q2 | 판정 | −9.55%/−20.76%/13 · −12.69%/−20.76%/10 | **동일 (렛저 자체 동일)** | 0 bar |

- VAL 게이트: strict **통과**(`with_gate` 완전 동일). OOS 단일터치: strict/relaxed 모두
  **CONFIRMED**.
- **Q3 효과**: `with_gate` −15.86% → **+20.17%**(부호 반전), MDD −44.37% → **−19.72%**;
  `no_gate` −37.43% → −10.63%, MDD −51.25% → −29.66%.

### 렛저 diff — 메커니즘 그대로

- **Q3 제거 8건**: 전부 zig075 SHORT stop_loss(합 −0.440), 7건이 2025-07-09~07-18 지속상승
  구간, 1건이 08-18 — 실행 로그 #1이 지목한 진입타이밍 손실 거래들과 정확히 일치. 풀린
  슬롯은 h48qual이 4건 이어받음(합 −0.094, LONG 익절 1건 포함).
- **Q2 비용 1건**: 06-16 zig075 SHORT 익절(+0.152) 제거, 슬롯이 zig075 LONG 손절(−0.074)로
  대체 — 이 창 악화의 전부. replay 전 오버레이에서 이미 공개된 비용(Q2 겹침 1/16)이며 단일
  거래 분산 수준. Q3 개선은 8건 제거로 메커니즘 방향과 일치하는 반면 Q2 비용은 1건 우연.
- **VAL**: 12-12 zig075 SHORT 익절(+0.138) 제거, 슬롯이 h48qual SHORT 익절(+0.095)로 대체 —
  `no_gate` −5.5pp, `with_gate`(판정 기준) 불변.

### 강건성 — Q3 효과는 임계값 아티팩트가 아님

| veto 임계값 | 2025-Q1 | 2025-Q2 | 2025-Q3 (with_gate) |
|---|---|---|---|
| p75 (0.5610) | 73.88%/27.51% (승자 1 제거) | 93.44%/23.20% (3 제거/5 추가) | **+39.32%/−19.72%** (12 제거) |
| **p90 (0.80258, 채택)** | **동일 (0 제거)** | 65.83%/5.62% (1 제거) | **+20.17%/−19.72%** (8 제거) |
| p95 (0.8773) | 동일 (0 제거) | **동일 (0 제거)** | **+20.17%/−19.72%** (8 제거, p90과 동일) |

세 백분위 전부에서 Q3 `with_gate`가 크게 양전환(+20~+39). p95에서는 Q1/Q2 비용이 완전히
사라지면서 Q3 효과가 p90과 동일하다 — 즉 효과는 p90이라는 특정 숫자가 아니라 탐지기 신호
자체에서 온다. **p90을 유지한다**: p95로 바꾸는 것이 백테스트상 우월해 보이지만, 그 판단
자체가 Q2 결과를 본 뒤의 사후선택이므로 하지 않는다(잠긴 베이스라인 값 p90이 유일하게
"결과를 보기 전에 고정된" 숫자다). p75는 비레짐 간섭이 커져 열등 — 임계값↓ = 간섭↑의 단조
구조로, p90은 안정 구간에 있다.

## 준수 확인

`fresh_forward_bar_by_bar=true`(replay는 i 증가 단방향 단일 pass, 탐지기는 순수 backward
rolling, veto는 신호 bar의 mask[i]만 읽음), `trade_ledgers_used_as_input=false`(실행 로그 #1의
진단 렛저는 동기 인용일 뿐 입력 아님 — 판정은 전부 신규 fresh-forward replay),
`saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false`.

`git diff` 확인(0줄): `trading_bot.py`, `trading_bot_modules/omega4_6_1_live.py`,
`trading_bot_modules/runtime_config.py`, 그리고 임포트만 한 기존 모듈 전부
(`research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.py`,
`eth_omega461_multiwindow_confirmation_gate_20260814.py`,
`replay_omega4_6_1_greedy_router_20260706.py` 등). 재학습 없음, GPU 불필요(DEVICE=cpu), conda
env `quant_ai`.

Seed-Diversity Ensemble Promotion Gate: 해당 없음(재학습 모델 없음, 시드 앙상블 주장 없음 —
개입은 결정론적 룰이라 시드 축이 존재하지 않음). Omega Artifact Integrity Promotion Gate:
해당 없음(기존 라이브 zig075/h48qual parent 예측 아티팩트 그대로 재사용, 신규 아티팩트 없음).

## 정직한 한계

1. **Q3는 참고 티어(in-sample OOF)다.** 판정 3창의 무손상은 확인됐지만, Q3 개선 자체는 훈련
   연도 분기에서 관찰된 것이다. h48qual 레짐 가드와 정확히 같은 지위 — **forward에서 진짜
   지속 상승장을 한 번도 겪지 않았다**(OOS 데이터는 2026-07-12까지, 유일한 상승 구간
   07-01~07-12 +18.72%는 12일로 표본 부족). 진짜 검증은 섀도우 forward 관찰뿐이다.
2. **판정 창 통과의 실질은 "무해성 증명"이다.** OOS 두 창은 veto 발동 0건(탐지기가 그 구간의
   zig075 SHORT 체결 진입과 안 겹침)이라 이 창들은 "이득 증명"이 아니라 "하락장/횡보장에서
   아무것도 건드리지 않음" 증명이다. 이는 설계 의도와 일치한다(보험은 화재가 없을 때 비용이
   0이어야 한다).
3. **Q2 비용은 실재한다**: 지속상승 신호가 뜬 구간에서도 가끔 숏이 이긴다(+0.152 1건 제거).
   레짐 베타 관리의 구조적 비용이며, p95 강건성 행이 보이듯 임계값을 높이면 사라지는 종류의
   경계 사례다.
4. zig075의 엣지 자체가 VAL 특이적(3.41×)이고 fresh 창에서 음수라는 기존 조사
   (`eth_val_oos_regime_mismatch_investigation_20260813.md`)는 이 실험과 별개로 유효하다 —
   이 veto는 zig075를 "살리는" 개입이 아니라 꼬리 손상을 줄이는 개입이다.

## 산출물

- 스크립트: `scripts/research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.py`
- report: `tmp/causal_regen_20260516/eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814/report.json`
  (창별 3중 비교, veto 이벤트 전체, per-trade 렛저 diff, 강건성, G0a/G0b 결과 포함)
- 렛저: 동 디렉토리 `portfolio_ledger_<창>_odyssey3_baseline.csv` /
  `portfolio_ledger_<창>_zig075_short_entry_veto_p90.csv`
- 실행 로그: `tmp/causal_regen_20260516/zig075_short_entry_veto_run_20260814.log`

## 다음 단계 / 미해결

- **섀도우 관찰 후보로 즉시 적격**: 탐지기는 이미 서버 섀도우
  (`live_eth_regime_aware_exit_guard_shadow_20260814.py`)가 bar마다 계산 중이므로, 같은
  프로세스에 "이 bar에서 zig075 SHORT 진입이 떴다면 veto됐을 것" 관찰 로깅을 추가하는 것이
  자연스러운 다음 단계다. **서버 상시 실행 프로세스 수정이므로 이 세션에서는 실행하지 않고
  후보로만 기록** — 배포는 별도 결정.
- 승격 기준은 h48qual 가드와 동일하게 미확정(Odyssey1 미해결 이슈 13) — "forward에서 지속
  상승장 최소 1회 관찰" 기준 후보가 이 개입에도 그대로 적용될 수 있다.
- h48qual SHORT에 같은 veto를 확장할지는 **미검토** — h48qual은 이미 exit-side 가드가 있고,
  Q3 손상의 주범(회전 가속)은 그 가드가 완화 중이라 우선순위 낮음. 검토한다면 동일한
  자유변수-0 원칙으로 가능.
