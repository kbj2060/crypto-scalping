# ETH Omega4.6.1 zig075 — 변동성-스케일 Kelly 사이징 (비-RL, 2026-08-15, Odyssey2 #25)

상태: **완료.** 이 축도 컴포넌트 레벨(VAL)에서 HGB를 못 이긴다 — 오히려 plain Kelly(v2, 확장
그리드)보다도 낮다. 비-RL 사이징 후보 2전 2패로, 사이징 축에서 HGB를 이기는 닫힌형 규칙을 아직
못 찾았다.

## 배경과 진단

[[eth_omega461_fractional_kelly_sizing_benchmark_20260815]](plain Kelly)가 HGB를 못 이긴 이유를
먼저 진단했다: 배포된 zig075 `report.json`의 `atr_diag`가 `tp_p50==tp_p90==min_tp(0.075)`,
`sl_p50==sl_p90==min_sl(0.040)`을 train/validation/oos **전부**에서 보여준다 — 즉 90번째
백분위수까지도 TP/SL이 ATR floor에 고정돼, `decision_rr`(Kelly의 b항)이 최소 90% 이상의 행에서
사실상 상수(0.075/0.040=1.875)다. **plain Kelly는 사실상 `p`(quality_score) 하나만으로
판별하는 셈**이었던 반면, HGB는 클리핑되지 않은 원시 `atr_pct_runtime`을 12개 남짓
`parent_outputs` 피쳐 중 하나로 직접 쓴다. 이 손실된 정보를 명시적 역변동성 배수로 복원하면
격차가 좁혀지는지 테스트했다 — 문헌이 이미 확립한 "변동성-스케일 포지션 사이징"(Zhang, Zohren,
Roberts, arXiv:1911.10107, [[eth_odyssey4_rl_layer_integration_literature_research_20260815]]
S2가 인용)과 같은 원리다.

## 방법

- **공식**: `score = kelly_score * clip(atr_ref/atr_pct_runtime, 0.5, 2.0)`. `kelly_score`는
  plain Kelly와 완전히 동일(무수정 import). `atr_ref`는 같은 2025q1+q2+q3 active row 풀의
  중앙값(causal, 룩어헤드 없음). `vol_scale` 상하한(0.5, 2.0)은 **고정**(그리드서치 안 함) —
  "진단된 정보 하나를 최소한의 자유도로 복원"이라는 이 후보의 취지상 자체적으로 또 다른 다축
  그리드가 되면 안 된다는 원칙.
- 마진 매핑 그리드·레버리지 매핑·VAL전용 2단계 선택·6구간 컴포넌트+포트폴리오 확인 등 나머지
  전부는 plain Kelly 스크립트(v2, 확장 그리드 960조합)에서 **무수정 import**로 재사용 —
  스코어 생성 방식만 유일한 변수.
- G0: 이 스크립트의 `dec` 구성이 plain Kelly의 `dec`(이미 신뢰된 원본과 검증됨)와 val/oos_q1/
  oos_q2 3구간 전부 일치(전이적 검증) — 통과.

## 결과

### 컴포넌트 레벨(zig075 단독)

| 구간 | HGB(신선) | plain Kelly(v2) | **변동성-Kelly** |
|---|---:|---:|---:|
| 2025q1 | 43.89%/-14.16%/24 | 26.74%/-12.60%/24 | 31.92%/-10.21%/24 |
| 2025q2 | 60.83%/-13.38%/27 | 32.03%/-12.27%/27 | 22.80%/-10.85%/27 |
| 2025q3 | -2.80%/-23.13%/24 | -8.79%/-29.20%/24 | -12.52%/-31.70%/24 |
| **VAL(선정 기준)** | **40.31%/-13.07%/29** | 33.95%/-9.53%/29 | **29.10%/-8.58%/29** |
| OOS-Q1 | 17.12%/-11.01%/25 | 16.94%/-14.38%/25 | 11.90%/-13.74%/25 |
| OOS-Q2 | -5.85%/-11.55%/13 | 0.47%/-11.90%/13 | 1.30%/-11.93%/13 |

**VAL에서 변동성-Kelly(29.10%)는 HGB(40.31%)뿐 아니라 plain Kelly v2(33.95%)보다도 낮다** —
가설(진단된 정보를 복원하면 도움이 될 것)과 반대 방향. `beats_fresh_hgb_baseline_val_component_
level=False`. train_iqr이 plain Kelly의 0.073에서 0.309로 4배 넓어졌는데(vol_scale 배수의
직접적 효과), z-score 정규화가 이 스케일 차이를 흡수하도록 설계돼 있어(같은 상대 z-공간에서
그리드 탐색) 이게 그리드 불공정성 때문이라고 보지는 않는다 — 진짜 원인은 원시 ATR과 실현
거래손익의 관계가 "변동성 낮을 때 크게 베팅"이라는 단순 역비례로는 잘 안 잡힌다는 쪽에 가깝다고
판단한다(고변동성 구간이 오히려 큰 TP를 만드는 추세 동역학이 있을 수 있음 — 확정 아님).

### 포트폴리오 레벨

| 구간 | baseline with_gate | candidate with_gate |
|---|---:|---:|
| VAL | 77.31%/-21.76%/26 | 68.63%/-21.16%/26(악화) |
| OOS-Q1(게이트) | 67.25%/-15.48%/19 | 77.09%/-16.77%/18(PnL 개선, **MDD 악화 1.29%p**) |
| OOS-Q2(게이트) | -12.69%/-20.76%/10 | -10.48%/-19.86%/10(개선) |

`strict(mdd_slack=0)=REJECTED_SIGN_MISMATCH`(OOS-Q1 MDD 악화로 엄격기준 탈락) — plain Kelly는
strict·relaxed 둘 다 통과했던 것과 달리, 이번엔 **완화기준(mdd_slack=3pp)에서만** `CONFIRMED`.
plain Kelly보다 한 단계 더 약한 포트폴리오 신호다. 이유는 plain Kelly와 동일(컴포넌트 경제성
악화 vs 포트폴리오 개선 괴리) — **역시 승격 근거로 쓰지 않는다.**

## 정직한 결론

1. **비-RL 사이징 후보 2전 2패**: plain Kelly, 변동성-스케일 Kelly 둘 다 컴포넌트·VAL 기준
   HGB를 못 이겼다. 진단된 구체적 정보 손실(ATR 클리핑)을 명시적으로 복원해도 도움이 안
   됐다 — 오히려 plain Kelly보다 나빠졌다.
2. HGB가 이 12개 남짓 `parent_outputs` 피쳐 조합에서 상당히 촘촘한 근사라는 가설이 이제 2개의
   독립적 실패로 뒷받침된다(확정은 아니지만 근거가 쌓이고 있다).
3. `vol_scale` 상하한을 그리드서치했다면 결과가 달라졌을 수 있으나(테스트 안 함, 자유도를
   늘리지 않기 위한 의도적 선택), 그 경우 "단순한 비-RL 규칙" 취지 자체가 흐려진다.

## 산출물

- `scripts/research_eth_omega461_volatility_scaled_kelly_sizing_20260815.py`
- `tmp/causal_regen_20260516/eth_omega461_volatility_scaled_kelly_sizing_20260815/report.json`

## 출처

- [[eth_omega461_fractional_kelly_sizing_benchmark_20260815]] (plain Kelly, 하네스·진단의 출처)
- [[eth_odyssey4_rl_layer_integration_literature_research_20260815]] (변동성-스케일 사이징의 문헌 근거)
