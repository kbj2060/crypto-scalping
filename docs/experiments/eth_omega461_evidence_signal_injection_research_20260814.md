# 증거 신호 → 라이브 Omega4.6.1 주입 전략 리서치 — 2026-08-14

상태: **리서치 완료 (설계만 — 구현·학습·백테스트 없음)**. Model Architect 페르소나 dispatch + 리드 세션 검증.

## 질문

두 개의 독립 구간(2025-09~2026-02, 2026-03~07)에서 순위가 재현된 ETH 5분봉 반전 증거 신호들
(`docs/experiments/eth_evidence_signal_ranking_stability_mar_jul_2026_20260814.md`)을 라이브
Omega4.6.1 ETH 모델(h48qual+zig075)에 어떻게 주입하는 것이 좋은가?

## 방법

Odyssey 서브프로젝트 관례대로 Model Architect 페르소나(단일 에이전트)에 설계 리서치를 위임하고,
리드 세션이 부하가 실리는 주장을 소스로 직접 재검증했다. 직접 검증한 항목:

- `docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md` 존재 + 실행 로그 #1~#16 실재
- `scripts/eth_omega461_multiwindow_confirmation_gate_20260814.py` 존재 + `load_all_windows`/`run_portfolio_variant`/`summarize_multiwindow` 3함수 실재
- `scripts/live_eth_regime_aware_exit_guard_shadow_20260814.py` 존재
- Odyssey2 #11의 핵심 수치(dual_momentum 게이트: Q3 no_gate 격차 82.4% 회복, 활성화율 Q3 43.0% vs 나머지 5개 창 5.4~11.6%) — 계약 문서 원문과 일치
- 대기압력(#7)·risk-controlled(#8) 소급 스트레스테스트 `REJECTED_SIGN_MISMATCH` 판정 — 원문 일치
- 증거 신호 공식의 인과성: `scripts/analyze_eth_broad_evidence_signal_sweep_20260814.py` 계열 전부 rolling/shift 기반, 입력은 OHLCV+`taker_buy_base`뿐
- 리플레이 하네스가 쓰는 패널 CSV(`data/splits/year_oos/training_features_2026_rebuilt.csv`, 142컬럼)에 `taker_buy_base`/`taker_buy_quote`/`net_taker_ratio`/`taker_acceleration` 실재 — **재학습 없이 오버레이로 백테스트 가능**

## 핵심 발견 0 — 가장 가까운 선행: Odyssey2가 이미 같은 라이브 모델에 post-entry 주입 16건을 소진했다

`odyssey2_eth_live_injection_contract_20260813.md`의 실행 로그 #1~#16: 레짐조건 threshold,
epistemic-MI 사이징, 오토인코더 latent 사이징, GBDT/TCN exit_head 교체, 대기압력 조건부 exit
threshold, conformal-Kelly 사이징, Gittins index exit 재정식화 등 — **승격 0건**, 상당수가 이
서브프로젝트의 "VAL 승리 → OOS 반전" 메타패턴을 재현했고, 유일한 미확정 생존은 #11/#13
레짐인지 exit 가드(섀도우 배포, forward 미확인).

이게 이번 질문에 중요한 이유 두 가지:

1. **Odyssey2의 16개 후보는 전부 내부 파생 신호였다**(모델 불일치, latent 압축, 같은 피쳐 패널의
   재스코어링, 레짐 감지기). OHLCV+taker 롤링 z-score 기반의 **외부 증거 신호 계열은 이 라이브
   모델에 한 번도 시도된 적 없는 진짜 신규 정보원**이다 — 이번 기회의 실체는 "새 주입 지점"이
   아니라 "새 신호원"이다.
2. 검증 도구가 이미 준비돼 있다: 다중구간 확인 게이트 모듈(6개 창: 2025 Q1~Q3 참고 + VAL 게이트 +
   OOS-Q1/OOS-Q2 단일터치)이 이 클래스 후보의 공식 심사 절차다. 과거 단일-OOS 확인 패턴은 이미
   폐기됐다.

## 신호 인벤토리 — 순위 안정성이 확인된 상위 5개

전부 인과적(rolling/shift, 확정 bar만 사용), 입력은 OHLCV+`taker_buy_base`.

| 신호 | 공식 요약 | 바닥 lift (원본/새구간) | 천장 lift |
|---|---|---|---|
| `orthogonal_combo` | 적응형 오실레이터 극단 AND `delta_z = z288(2*taker_buy_base - volume)` ≤ -2 동시 | 3.51 / **3.92** | 2.53 / **4.14** |
| `liquidity_sweep` | low가 직전 48bar 스윙로우 하향 돌파 후 종가는 위로 복귀(스탑헌트), 대칭 | 3.01 / **3.40** | 2.78 (원본, 천장 1위) |
| `volume_wick_climax` | 거래량 z≥2 AND 아래꼬리≥레인지 50% (Wyckoff) | 2.94 / 3.07 | 2.50 |
| `short_term_return_z` | 3bar(15분) 수익률의 rolling(288) z-score ≤ -2.5 / ≥ +2.5 | 2.90 / **3.38** | 2.72 |
| `taker_sell/buy_climax` | `delta_z` 단독 ≤ -2 / ≥ +2 | 2.75 / 3.12 | 2.29 |

공통 유보(원문 그대로 유지해야 함): precision 최고 43.9%(1h), 진양성도 피벗 전 평균 0.5~0.85%
추가 역행, lift 방법론은 회고적 피벗 거리 측정이지 백테스트가 아님 — **트레이딩 근거 주장은 반드시
fresh-forward 리플레이 하네스로 재도출**해야 한다. 순위 안정성(Spearman 0.976/0.924)이 검증한
것은 상대 순위지 절대 lift 크기가 아니다.

## 주입 지점 지도 (라이브 파이프라인 실행 순서 기준)

| 주입 지점 | 트레이드 모집단 변경? | 판정 |
|---|---|---|
| 진입 거부(반대증거 → 숏 진입 차단) | **예 — 진입측** | 고위험. quality-head relabel(진입 모집단 변경)의 OOS 반전 전례와 같은 클래스. direction_head 스킬 부재가 확정된 상태에서 거부 필터는 "스킬 없는 선택의 부분집합"만 바꾼다. |
| quality 게이트 보강(증거 피쳐 재학습) | **예 — 진입측** | 최저 사전확률. 게이트≈confidence 필터라는 확정 결론에 따라 어떤 피쳐를 넣어도 같은 실패 상속. |
| exit_head 입력 피쳐 추가(TabM 재학습) | 아니오 — post-entry | 생존 클래스이나 Odyssey2 #4/#5(GBDT/TCN)·#16(Gittins)이 "포트폴리오 개선처럼 보이나 컴포넌트 부호반전" 가드레일에 반복 걸린 위험지대. Tier 2. |
| **규칙 기반 exit 오버레이**(포지션 반대증거 → 조기청산/threshold 완화) | 아니오 — post-entry | **미시도 형태.** 가장 가까운 선행은 #11(레짐 레벨 감지기)·#7(내부 신호로 threshold 조건부 변경) — 같은 구조 클래스, 다른 신호원. Tier 1 최우선. |
| TP/SL 폭 조절 | (경제성 변경) | **닫힌 축** — SLTP 폭 학습·ATR 재보정 둘 다 부정 종결. 재제안 금지. |
| L4 사이징 사이드카 피쳐(GBM) | 아니오 — post-entry | 미시도(증거 신호로는). 단 #3(latent 사이징)이 13~28건 소표본 과적합으로 zig075 OOS 악화된 전례 유의. Tier 1. |
| 멀티슬롯 용량 게이팅 | **예 — 진입측** | **닫힌 축** — 상관노출 문제 미해결. 재제안 금지. |
| 섀도우봇 관찰 지표(행동 무변경 로깅) | 아니오 — 순수 관찰 | 비용·리스크 0. Tier 0. |

## 후보 랭킹

모든 백테스트 단계는 다중구간 확인 게이트 모듈로 심사(VAL 게이트 → OOS-Q1+OOS-Q2 단일터치,
2025 Q1~Q3는 참고 맥락), 기준선은 `baseline_both_original`과 `asymmetric_tabm_liveatr`(현
섀도우) **둘 다**와 대조한다.

### Tier 0 — 무학습·순수 진단 (순위와 무관하게 먼저)

- **A. 섀도우 관찰 로깅**: 5개 신호를 라이브 5분봉에서 bar-by-bar 계산해 기존 섀도우봇들의 실제
  진입/청산에 태깅만 한다(행동 무변경). 회고적 lift가 아니라 **섀도우봇의 실현 결과**와의 상관을
  먼저 확보. 비용 <1일, 리스크 0.
- **B. 냉동 예측 순위상관 진단**: 기존 per-bar 예측 CSV + 리플레이 렛저에 신호값을 조인해 "열린
  숏 보유 중 바닥증거 발화 → 이후 그 트레이드의 역행(MAE)/PnL"의 상관을 6개 창(VAL, OOS-Q1/Q2,
  2025 Q1~Q3) 전부에서 확인. 이 서브프로젝트의 표준 "0단계 진단" 패턴
  (`eth_h48qual_quality_for_action_rank_correlation_20260811.md`) 재사용. **킬 기준: 6개 창 중
  4개 이상 부호 일치 실패 → Tier 1 진행 안 함.** 비용 1~2일, 무학습.

### Tier 1 — post-entry·무재학습

- **C. 숏 포지션 반대증거 exit 오버레이 (Q3-2025 회전 가속 취약점 직격)**: h48qual 숏 보유 중
  `orthogonal_combo` 또는 `taker_sell_climax` 발화 시 (a) 즉시 청산 또는 (b) N-bar 동안 exit
  threshold 완화(#7과 같은 개입 형태). 근거: liveatr exit_head 재라벨은 회전 가속기라 Q3-2025
  지속 상승장에서 숏 8→18건 폭증·4.7배 악화 — 바닥증거는 정확히 그 레짐의 국소 반대증거다.
  **#11(dual_momentum 주단위 레짐 게이트)과의 관계를 명시**: 같은 문제를 겨냥한 선행이 이미
  섀도우 중이며, C는 신호원(오더플로우/꼬리 기하 vs 주단위 추세)과 입도(bar 레벨 국소 발화)가
  다른 보완 후보다 — 신규로 재발명하지 말고 #11 대비 증분 가치로 평가한다. Q3-2025 슬라이스에서
  4.7배 악화의 회복률을 #11의 82.4%와 직접 비교. 비용: dev-side, 무재학습(greedy_replay 사본
  패턴 재사용).
- **D. 사이징 사이드카 GBM 피쳐 추가**: 증거 z-score 5개를 side-split 사이징 GBM에 추가.
  post-entry 중 최소 폭발반경, 재학습은 GBM뿐(분 단위). 킬 기준: #3 전례처럼 zig075(진짜 엣지가
  있는 컴포넌트) OOS 악화 또는 VAL/OOS 불균형 과적합 시그니처.

### Tier 2 — post-entry·TabM 재학습

- **E. exit_head 입력 피쳐 확장**(25차원 + 증거 5개, liveatr 재라벨 레시피·모델 계열 불변,
  exit_head만 재학습): C/D가 신호 유효성을 보인 뒤에만. Odyssey2 #4/#5/#16이 걸린 컴포넌트-vs-
  포트폴리오 가드레일(컴포넌트 상대 50% 초과 악화 또는 부호반전 시 기각)을 필수 적용. Seed-
  Diversity Gate(N≥5 무작위 시드) 적용 대상. 비용: 서버 GPU(분 단위, 후보 이벤트 데이터셋 기존재).

## 안티골 (재제안 금지 목록)

- 증거 신호를 **진입측**(quality 재학습, 진입 거부, 멀티슬롯 게이트)으로 쓰지 않는다 — 7개 독립
  모델×라벨 조합의 OOS 전패 + quality relabel의 모집단 변경발 OOS 반전 전례.
- TP/SL 폭 조절 재제안 금지 — 닫힌 축.
- **펀딩 계열 증거 신호 추가 금지**: `funding_extreme`/`funding_flip`은 두 구간 모두 최하위
  (1.08~1.25, 천장 4h/8h는 1.0 미만)이고 펀딩 피쳐는 이미 라이브/FINAL12 패널에 있는데 효과가
  없다 — 약하고 중복인 정보원.
- 단일 OOS 통과를 "확인됨"으로 쓰지 않는다 — 다중구간 게이트 모듈이 공식 절차.
- lift 수치를 백테스트 증거처럼 인용하지 않는다 — 회고적 진단 전용.

## 사용자 결정 대기 (열린 질문)

1. 첫 실전 실험을 C(반대증거 exit 오버레이 — Q3 취약점 직격, exit 행동 직접 변경)로 할지 D(사이징
   피쳐 — 폭발반경 최소, Q3 문제와의 연결은 약함)로 할지. 단일터치 규율상 동시에 여러 개를 OOS에
   올리지 않는 게 좋다.
2. C를 기존 `eth_regime_aware_exit_guard_shadow`에 분기 추가로 붙일지(두 Q3 대응 메커니즘 동시
   관찰), 별도 섀도우로 분리할지(기여 분리).
3. 순위 안정성 2개 창으로 Tier 1에 바로 진입할지, 증거 신호 lift 순위 자체를 2025 Q1/Q2/Q3
   슬라이스에서 한 번 더 재확인(4+ 창 규율 충족)한 뒤 진행할지.

## 산출물

- 이 문서 (설계 리서치 — 스크립트/코드 산출 없음)
- 참고: `eth_evidence_signal_ranking_stability_mar_jul_2026_20260814.md`(신호 검증),
  `odyssey2_eth_live_injection_contract_20260813.md`(선행 16건 + 심사 절차),
  `eth_omega461_exit_head_liveatr_sustained_uptrend_vulnerability_20260814.md`(C의 표적 취약점),
  `eth_omega461_multiwindow_confirmation_gate_20260814.md`(검증 하네스)
