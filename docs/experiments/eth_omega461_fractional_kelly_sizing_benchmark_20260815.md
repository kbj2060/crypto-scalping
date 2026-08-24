# ETH Omega4.6.1 zig075 — Fractional Kelly 사이징 벤치마크(비-RL) (2026-08-15, Odyssey2 #24)

상태: **완료(그리드 2회 — v1 소규모 그리드, 사용자 요청으로 v2 확장 그리드 재확인).** 둘 다
결론은 같다: 컴포넌트 레벨(VAL, 사전등록 선정 기준 구간)에서 Kelly가 배포된 HGB 사이드카를 못
이긴다(PnL 열세) — 핵심 질문에 대한 답은 **아니오**. v2에서 격차는 좁혀졌다(-11.29%p→-6.36%p)
그러나 부호는 안 바뀌었다. 포트폴리오 레벨 6구간 단일터치 게이트는 v1·v2 둘 다 기계적으로
`CONFIRMED`가 찍혔으나, 이 프로젝트가 Gittins(#Odyssey4 실행로그)·GBDT·TCN exit_head 실험에서
반복 관찰한 "컴포넌트 경제성은 나빠지는데 공유슬롯 재순환 효과로 포트폴리오는 좋아 보인다"는
패턴과 정확히 일치해 **신뢰하지 않는다.**

## 배경과 목적

[[eth_odyssey4_rl_layer_integration_literature_research_20260815]] 3.3절 1단계의 권고를 실행한다:
"RL 사이징을 재시도하기 전에, direction/quality 확신도의 단조함수(fractional Kelly 등)로 이미
학습된 HGB 사이징 사이드카(`train_eval_omega4_2_risk_sidecar_20260622.py`)를 이길 수 있는지부터
가장 싸게 확인한다 — 이 비교는 논문에도, 이 프로젝트의 2026-06-23 RL 사이드카 실험
(`docs/model_contracts/omega4_4_rl_risk_sidecar_v1_full_20260623_contract.md`)에도 없었다."

## 방법

- **공식**: `f = p - (1-p)/b` (Kelly, 1956; 이진 승/패 베팅의 표준 fractional Kelly).
  `p = decision_quality_score`(parent가 선택한 action에 대한 quality 클래스 확률 — entry gate
  자체가 임계값을 대는 그 값), `b = decision_rr`(=take_profit/|stop_loss| 가격이동비, 이미
  `_risk_feature_frame`가 계산). 학습·`random_state` 없음 — 두 기존 컬럼의 닫힌형 함수, 시드
  분산 0.
- zig075만 개입, h48qual은 배포된 HGB 사이드카 그대로 동결(오늘 밤 다른 모든 Odyssey2 후보와
  같은 "테스트 대상 하나만 남기고 동결" 규율).
- `train_q50`/`train_iqr`(z-score 정규화 상수): 2025q1+q2+q3 active row 풀(n=6,428)의 Kelly
  스코어 분포에서 계산 — 배포된 pkl의 HGB 전용 상수는 재사용 불가(스코어 분포 자체가 다름).
- 마진 매핑(min_scale/max_scale/temp/floor/cap) VAL전용 소규모 그리드(108조합, long_scale/
  short_scale=1.0 고정) — 배포된 HGB 매핑을 만든 2,304조합 그리드보다 의도적으로 훨씬 작음("단순한
  비교"가 스스로 큰 그리드서치를 착취하면 비교 취지가 무너짐. 레버리지 매핑은 재탐색 없이 배포된
  zig075 pkl의 값 그대로 재사용 — 스코어 생성 방식만 유일한 변수로 격리.
- 2단계 그리드 평가(원 사이드카 스크립트의 `stage=grid_risk_mapping`→`selected_full_replay`
  구조를 그대로 차용 — margin/leverage가 exit_head 입력 피처라 발화 타이밍에 영향을 줄 수 있다는
  Conformal Kelly의 발견([[eth_omega461_conformal_kelly_sizing_scale_20260814]]) 때문에 그
  스크립트도 저렴한 근사(고정 렛저 재점수화)로 넓게 훑고 승자 1개만 정확한 풀 리플레이로
  재확인하는 구조를 쓴다): (1) 저렴 — G0에서 이미 만든 HGB 기준 렛저를 각 후보 margin/leverage로
  재점수화(`_ledger_metrics_with_margins`); (2) 정확 — 저렴 단계 승자만 진짜 bar-by-bar 풀
  리플레이(`replay_exit_variant`)로 재확인.
- 하네스: 오늘 밤 다른 모든 Odyssey2 사이징/exit 후보와 동일한 `eth_omega461_multiwindow_
  confirmation_gate_20260814`/`research_eth_omega461_exit_sweep_20260721` 인프라를 무수정
  재사용.

## 파이프라인 소스 불일치 발견 (이번 세션에 발견, 공개)

배포된 zig075 `report.json`의 `omega4_2_replayed_baseline`(OOS pnl=34.11%, 13건)은
`omega4._prepare_frames`(→`trade_candidates_2026_alpha6_...csv`, OOS 구간이 정확히
2026-01-01~02-28)라는 **오늘 밤 다른 모든 Odyssey2 후보가 쓰는 파이프라인과 다른** feature 소스로
계산됐다 — `eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md`가 이미 발견한
"alpha6/7-lineage vs extended/wide24" feature drift와 같은 종류의 불일치. 이 벤치마크는
`research_eth_omega461_exit_sweep_20260721.load_frame`(→`data/splits/year_oos/training_features_
*.csv`+wide24, oos_q1=2026-01-01~03-31) 파이프라인을 쓰므로, **report.json의 예전 숫자를
재현하려 하지 않고**, 같은(새) 파이프라인에서 배포된 HGB 사이드카를 처음부터 다시 리플레이해
"신선한" 기준선을 만들어 비교했다(G0 단계). 이래야 Kelly와 HGB가 진짜 같은 조건에서 비교된다.

## G0 — 자기정합성 확인

`_prep_zig075_score`(신규)가 만드는 `dec`(side/quality_score/notional_exposure/leverage/
take_profit/stop_loss)가 신뢰된 원본 `sweep.prep_component`의 `dec`와 val/oos_q1/oos_q2 **3개
구간 전부 완전 일치**(`dec_match=True`) — 새 코드가 공유하는 상류 파이프라인에 버그가 없음을
확인.

## 결과

### 컴포넌트 레벨(zig075 단독) — 핵심 질문에 대한 답

| 구간 | HGB(이 파이프라인, 신선) | Kelly(그리드 승자) | Kelly 우세? |
|---|---:|---:|---|
| 2025q1(맥락) | 43.89% / -14.16% / 24건 | 27.25% / -11.85% / 24건 | 아니오(PnL 열세) |
| 2025q2(맥락) | 60.83% / -13.38% / 27건 | 29.86% / -10.96% / 27건 | 아니오 |
| 2025q3(맥락) | -2.80% / -23.13% / 24건 | -9.09% / -26.59% / 24건 | 아니오 |
| **VAL(선정 기준)** | **40.31% / -13.07% / 29건** | **29.02% / -9.13% / 29건** | **아니오**(PnL 열세, MDD는 우세) |
| OOS-Q1 | 17.12% / -11.01% / 25건 | 17.49% / -11.89% / 25건 | 거의 동률 |
| OOS-Q2 | -5.85% / -11.55% / 13건 | -0.63% / -10.89% / 13건 | 예(PnL·MDD 둘 다 우세) |

**VAL(사전등록 선정 기준 구간)에서 Kelly는 HGB를 이기지 못한다** — PnL이 명확히 낮다(29.02% vs
40.31%, MDD는 더 낮음/우세). PnL·MDD 둘 다 비악화(Pareto 우세)라는 기준을 통과하지 못해
`beats_fresh_hgb_baseline_val_component_level=False`로 판정했다. 2025q1/q2/q3 맥락 구간에서도
HGB가 일관되게 PnL 우세다. OOS-Q2 한 구간만 Kelly가 명확히 낫다.

### 포트폴리오 레벨 — 기계적 게이트는 CONFIRMED, 그러나 신뢰하지 않는 이유

| 구간 | baseline(진짜 h48qual-HGB+zig075-HGB) with_gate | candidate(h48qual-HGB+zig075-Kelly) with_gate |
|---|---:|---:|
| VAL | 77.31% / -21.76% / 26건 | 57.15% / -21.43% / 26건 (**악화**) |
| OOS-Q1(게이트 대상) | 67.25% / -15.48% / 19건 | 82.06% / -13.84% / 18건 (개선) |
| OOS-Q2(게이트 대상) | -12.69% / -20.76% / 10건 | -8.65% / -18.68% / 10건 (개선) |

`summarize_multiwindow`의 단일터치 기준(OOS-Q1+OOS-Q2 둘 다 with_gate PnL 비악화, strict
mdd_slack=0/relaxed=3 둘 다)을 통과해 `final_verdict=CONFIRMED`가 찍힌다(포트폴리오 기준선은
공식 `asymmetric_tabm_liveatr` 레퍼런스와 교차검증 완료 — val/oos_q1 no_gate·with_gate 4개
지표 전부 일치). **그러나 이 판정을 승격 근거로 쓰지 않는다:**

1. 정확히 이 판정이 걸러내야 할 구간(VAL)에서 포트폴리오 자체도 악화됐다(77.31%→57.15%,
   -20.16%p). `summarize_multiwindow`는 설계상(모듈 자체 문서화) VAL을 pass/fail에 안 넣지만
   — 그건 "VAL 판정은 호출자 책임"이라는 위임일 뿐, VAL 포트폴리오 악화 자체가 사라지는 건
   아니다.
2. 더 근본적으로: zig075 컴포넌트 단독 경제성이 VAL에서 명확히 나빠졌는데(위 표, PnL
   40.31%→29.02%), 포트폴리오는 OOS 두 구간에서 개선된 것으로 보인다 — 이는 Gittins
   ([[eth_odyssey4_rl_layer_integration_literature_research_20260815]]가 인용한
   `docs/experiments/eth_omega461_gittins_index_exit_head_20260814.md`)·GBDT·TCN exit_head
   실험이 반복 관찰하고 이 프로젝트가 이미 "가드레일이 정확히 걸러내야 할 패턴"으로 명명한 바로
   그 모양이다.

### 그리드 확장 재확인 (v2, 사용자 요청 "그리드 넓혀서 다시 확인해줘")

v1 승자(`min_scale=1.0, max_scale=2.5, temp=1.7, floor=0.18, cap=0.45`)가 테스트한 5개 축
**전부**에서 그리드의 가장 공격적인 끝에 위치했다. 재시도 전에 먼저 확인한 사실: 이건 이 실험이
그리드를 임의로 좁게 잡아서가 아니었다 — 배포된 HGB 매핑을 만든 프로덕션 `live_exposure_grid`
자체도 정확히 같은 5개 지점에서 끝난다(`min_scale` 최저 1.0, `max_scale`/`temp`/`cap` 최고
2.50/1.70/0.45) — 즉 v1 그리드는 프로덕션이 실제로 탐색한 전체 타당 구간과 정확히 일치했고,
Kelly의 원시 스코어가 **그 구간 너머**를 원한다는 뜻이었다. 그래서 각 경계를 넘어 한 단계씩
확장했다(108조합 → 960조합, 그래도 프로덕션의 2,304조합보다 작게 유지). `cap`만 예외적으로
0.55에서 상한을 뒀다 — 고정 재사용 중인 레버리지 매핑의 `leverage_cap=3.0`과 곱해도
margin_fraction×leverage ≤ 1.65로 라이브 `NOTIONAL_CAP=1.8`(`omega4_6_1_runtime_contract.py`)
아래 안전하게 남는 값이다; 더 큰 cap은 라이브에서 `finalize_sizing()`이 어차피 잘라낼 notional로
"이기는" 비현실적 결과를 만들 위험이 있어 원천 배제했다.

| 구간 | HGB(신선) | Kelly v1(소규모 그리드) | **Kelly v2(확장 그리드)** |
|---|---:|---:|---:|
| 2025q1(맥락) | 43.89%/-14.16%/24 | 27.25%/-11.85%/24 | 26.74%/-12.60%/24 |
| 2025q2(맥락) | 60.83%/-13.38%/27 | 29.86%/-10.96%/27 | 32.03%/-12.27%/27 |
| 2025q3(맥락) | -2.80%/-23.13%/24 | -9.09%/-26.59%/24 | -8.79%/-29.20%/24 |
| **VAL(선정 기준)** | **40.31%/-13.07%/29** | 29.02%/-9.13%/29 | **33.95%/-9.53%/29** |
| OOS-Q1 | 17.12%/-11.01%/25 | 17.49%/-11.89%/25 | 16.94%/-14.38%/25 |
| OOS-Q2 | -5.85%/-11.55%/13 | -0.63%/-10.89%/13 | 0.47%/-11.90%/13 |

v2 승자: `min_scale=0.75, max_scale=3.5, temp=2.7, floor=0.26, cap=0.55`. **min_scale·floor는
이번엔 그리드 내부 지점을 골랐다**(각각 새 최저값 0.5/0.08이 아님) — 이 두 축은 사실상
수렴했다고 본다. **그러나 max_scale·temp·cap 3개는 v2 그리드에서도 다시 가장 공격적인 끝을
골랐다** — 즉 이 세 축은 여전히 미수렴, 원한다면 더 밀어붙일 여지가 남아있다(다만 cap은 위에서
설명한 안전 상한에 이미 도달). VAL 컴포넌트 격차는 좁혀졌다(-11.29%p → -6.36%p, `beats_fresh_
hgb_baseline_val_component_level`는 v1·v2 둘 다 `False`). 포트폴리오 6구간 게이트도 v2에서
재확인했다 — 여전히 기계적으로 `CONFIRMED`(strict·relaxed 둘 다), VAL 포트폴리오도 여전히
악화(77.31%→61.81%), 위 "신뢰하지 않는 이유" 2가지가 v2에도 그대로 적용된다.

**여기서 멈춘 이유(정직하게 명시)**: max_scale/temp를 계속 밀어붙이면 VAL 격차가 더 좁혀질
가능성이 있으나, sigmoid 계수를 점점 극단으로 밀어붙이는 탐색은 "이 특정 VAL 구간을 맞추기 위한
피팅"과 "정말로 강건한 규칙을 찾는 것"의 경계가 흐려지기 시작한다 — Kelly의 원래 장점("단순하고
연구자 자유도가 낮다")과 모순된다. cap은 이미 원칙에 근거한 안전 상한에 도달했다. 세 축을 더
밀어붙일지는 사용자 판단에 맡긴다.

## 정직한 결론

1. **핵심 질문("Kelly가 HGB를 이기는가")에 대한 답은 아니오다** — 적어도 컴포넌트 단독·VAL
   기준으로는. RL 레이어 조사가 제안한 "RL 전에 먼저 확인할 값싼 비교"는 이걸로 완료됐고,
   결과는 부정적이다.
2. 이는 RL 사이징에 대한 함의를 오히려 강화한다 — 단순한 닫힌형 규칙조차 기존 HGB 회귀를 못
   이긴다면, 같은 feature 천장에 부딪힐 RL이 이걸 넘어설 것이라 낙관할 근거는 약하다(다만
   반대 해석도 가능: HGB 자체가 이 12개 남짓 parent_outputs 피쳐로 이미 상당히 촘촘한 근사라는
   뜻일 수도 있다 — 확정 아님).
3. 포트폴리오 레벨의 `CONFIRMED`는 표면적 숫자일 뿐이다 — Omega Artifact Integrity Gate와
   Fresh-Forward 규정의 정신, 그리고 이 세션이 이미 여러 번 학습한 "포트폴리오는 좋아 보이는데
   컴포넌트가 나쁜" 패턴에 대한 경계를 그대로 적용해 **promotion 근거로 쓰지 않는다.**
4. **그리드 경계 캐비엇(v2로 재확인 완료)**: v1 승자가 5개 축 전부에서 그리드 경계에 위치했던
   문제를, 그리드를 거의 9배로 넓혀(108→960조합) 재확인했다 — **결론은 바뀌지 않았다**(VAL
   컴포넌트 PnL은 29.02%→33.95%로 좁혀졌지만 여전히 HGB의 40.31%에 못 미침). min_scale·floor는
   수렴했으나 max_scale·temp·cap 3개는 v2에서도 다시 경계를 골라 완전히 수렴했다고는 말할 수
   없다(cap은 라이브 NOTIONAL_CAP 기준 원칙적 안전 상한에 이미 도달). 상세: 위 "그리드 확장
   재확인(v2)" 절.
5. Kelly 스코어 자체는 학습·시드가 전혀 없어(닫힌형 공식) 이 비교는 seed-diversity 게이트가
   적용될 필요가 없다 — 이 축의 유일한 장점(재현성·제로 시드분산)은 실현됐으나, 경제적 성과
   자체가 HGB를 못 이겨 그 장점을 promotion으로 연결하지 못한다.

## 산출물

- `scripts/research_eth_omega461_fractional_kelly_sizing_benchmark_20260815.py`
- `tmp/causal_regen_20260516/eth_omega461_fractional_kelly_sizing_benchmark_20260815/report.json`

## 출처

- [[eth_odyssey4_rl_layer_integration_literature_research_20260815]] (이 실험의 동기)
- `docs/model_contracts/omega4_4_rl_risk_sidecar_v1_full_20260623_contract.md` (RL 사이징 사이드카 이력)
- [[eth_omega461_conformal_kelly_sizing_scale_20260814]] (같은 사이징 축의 직전 후보, 하네스·인과성 발견 재사용)
- `docs/experiments/eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md` (파이프라인 소스 불일치의 근거)
- `docs/experiments/eth_omega461_multiwindow_confirmation_gate_20260814.md` (6구간 단일터치 게이트 원 설계)
