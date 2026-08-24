# ETH conformal 하방-LCB 거부게이트 후보 — episode 라벨 생성 및 인접-episode 상관 진단 (2026-08-16)

상태: **라벨 생성 완료, 필수 진단(계약 미해결 이슈 1)에서 심각한 문제 확인 — 다음 단계(HGB 학습)
전에 purge/embargo 설계가 필요하다는 결론.** OOS-Q1/OOS-Q2는 이 스크립트에서 로드조차 하지
않았다(학습 데이터 소스 자체가 아님).

## 배경

사용자가 (A) 회귀모델 착수를 선택한 뒤, 계약(`docs/model_contracts/
eth_candidate_conformal_downside_veto_contract_20260816.md`)의 "다음 단계" 2번을 실행했다:
h48qual/zig075 각각의 quality-gate 통과 episode(실현 트레이드가 아니라 신호 자체) 시작 bar마다,
그 컴포넌트 단독으로 진입했다면 어떻게 됐을지 causal하게 재시뮬레이션해서 `full`(net
가격변동률)과 `adverse`(최대 역행폭) 라벨을 만들었다. 스크립트: `scripts/
research_eth_candidate_conformal_veto_episode_labels_20260816.py`. 2025 Q1~Q3(학습) + VAL(캘리브레이션)
만 로드했다.

시뮬레이션은 실제 포트폴리오 replay와 동일한 로직(TP/SL/exit_head, h48qual은 Odyssey3
레짐가드 전환 포함)을 그 컴포넌트 하나만 독립적으로 적용한다. 정확성 확인: TP 적중 episode의
`full` ≈ take_profit floor(0.075) − 2×fee, SL 적중은 ≈ −stop_loss floor(0.040) − 2×fee로
정확히 나옴 — 알려진 ATR floor 포화값과 일치.

## 결과 1: 표본 수 (원 계약이 기대한 대로)

| 창 | h48qual episode | zig075 episode |
|---|---:|---:|
| 2025-Q1 | 326 | 881 |
| 2025-Q2 | 274 | 928 |
| 2025-Q3 | 528 | 923 |
| VAL | 254 | 789 |
| **학습 풀 합계(Q1~Q3)** | **1128** | **2732** |

원 계약이 예상한 대로 실현 트레이드(창당 10~35건)보다 수십 배 많다.

## 결과 2: 인접-episode 라벨 상관 — 심각, 계약 미해결 이슈 1이 실제 문제로 확인됨

| 창 | 컴포넌트 | n | lag-1 자기상관(full) | 근사 유효표본(n_eff) | capped 비율 |
|---|---|---:|---:|---:|---:|
| 2025-Q1 | h48qual | 326 | 0.757 | 45.2 | 1.5% |
| 2025-Q1 | zig075 | 881 | 0.783 | 107.0 | 10.0% |
| 2025-Q2 | h48qual | 274 | 0.545 | 80.7 | 5.5% |
| 2025-Q2 | zig075 | 928 | 0.765 | 123.3 | 14.7% |
| 2025-Q3 | h48qual | 528 | 0.847 | 43.7 | 9.7% |
| 2025-Q3 | zig075 | 923 | 0.772 | 118.8 | 16.5% |
| VAL | h48qual | 254 | 0.777 | 31.9 | 0.4% |
| VAL | zig075 | 789 | 0.727 | 124.7 | 18.1% |

`n_eff = n * (1-ρ)/(1+ρ)`(AR(1) 근사, 표준 유효표본크기 공식). **원 표본의 6~8배가 그대로
날아간다.** h48qual 학습 풀(Q1+Q2+Q3) 원표본 1128건이 유효표본 ~170건 수준으로,
zig075의 2732건이 ~350건 수준으로 줄어든다. h48qual 쪽은 특히 심각하다 — 원래 "실현 트레이드
26건보다 40배 많다"고 주장했던 근거(계약 본문 표)가, 유효표본 기준으로는 실현 트레이드의
6~7배 정도로 쪼그라든다.

원인은 자명하다: 인접한 episode는 종종 같은 추세/변동성 국면에서 발생해서(frac_within_12bar가
0.45~0.65 — episode의 절반가량이 직전 episode로부터 1시간 이내에 시작됨) 거의 같은 시장
상황을 두 번 세는 꼴이 된다. 이건 이 저장소가 다른 축에서 이미 반복적으로 마주친 문제와 정확히
같은 종류다(BTC tripbarrier entry axis의 "label spans overlap heavily: effective_n ~4,058 vs
43,798 nominal", `research_line_registry.json`) — **purge+embargo+uniqueness weighting 없이
naive quantile 캘리브레이션을 하면 잔차 분위수가 체계적으로 과신(overconfident)하게 된다.**
계약이 잔차분위수 LCB를 "validation-calibrated"라고 부르는 것 자체가, 이 상관을 방치하면
거짓 정밀도가 된다.

## 결과 3: capped episode (부차적 발견)

zig075는 창에 따라 10~18%의 episode가 2000-bar 안전 상한(≈7일)까지 아무 exit 조건도 못
만나 강제 종료된다 — 이 episode들의 `full` 라벨은 "자연스러운 청산 결과"가 아니라 "임의
시점에서 잘린 값"이라 다른 episode와 이질적이다. h48qual은 대체로 낮다(0.4~9.7%)지만 무시할
수준은 아니다.

## 라벨 자체의 건전성 (참고, 문제 아님)

- 양성 비율(`full>0`)은 39~71% 범위로 극단적 편향(전부 한쪽 클래스) 없음.
- `full_mean`은 대체로 작은 양수(0.007~0.03), 2025-Q2만 살짝 음수(−0.001~−0.002) — 알려진
  구간별 난이도 차이와 방향이 일치, 시뮬레이션이 이상 동작한다는 신호 없음.

## 결론 — 다음 단계 전에 반드시 필요한 것

**HGB 회귀 학습·잔차분위수 캘리브레이션을 지금 바로 하면 안 된다.** 필요한 것:

1. **Purge/embargo 설계**: 학습(Q1~Q3)과 캘리브레이션(VAL) 사이는 이미 시간순으로 분리돼 있어
   안전하지만, **같은 창 내부의 인접 episode끼리도** 최소 시간 간격 이하로는 학습셋에 중복
   포함시키지 않거나 가중치를 낮춰야 한다(purge). 정확한 embargo 폭은 lag-1 자기상관이
   실질적으로 0에 가까워지는 시차를 진단해서 정해야 한다(이 문서는 lag-1만 봤다 — lag-N
   자기상관 함수 전체를 봐야 embargo 폭을 정당화할 수 있음, 다음 단계에 포함).
2. **Uniqueness weighting**: purge만으로 표본을 버리면 유효표본이 더 줄어든다 — 이 저장소가
   다른 축에서 쓰는 관례(BTC tripbarrier 라인)처럼, 겹치는 episode에 낮은 가중치를 주는 방식이
   전량 제거보다 나을 수 있다.
3. **잔차분위수 캘리브레이션 자체도 purge된/uniqueness-weighted 잔차로 다시 계산**해야 한다 —
   지금처럼 원 잔차 분포에 그냥 분위수를 매기면 클러스터 안의 유사 잔차들이 분포 폭을 과소평가
   시켜 LCB가 실제보다 낙관적으로 계산된다.

## 아티팩트

- 스크립트: `scripts/research_eth_candidate_conformal_veto_episode_labels_20260816.py`
- 라벨 데이터(parquet, 창×컴포넌트별): `tmp/causal_regen_20260516/
  eth_candidate_conformal_veto_episode_labels_20260816/episode_labels_<window>_<component>.parquet`
- 리포트: `tmp/causal_regen_20260516/eth_candidate_conformal_veto_episode_labels_20260816/report.json`
