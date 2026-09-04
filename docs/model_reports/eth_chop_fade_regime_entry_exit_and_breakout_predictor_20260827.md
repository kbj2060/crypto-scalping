# ETH chop-fade 전략 — 진입/손절 재설계(Stream 1) + 돌파예측 시도(Stream 2) — 2026-08-27

## Context

사용자의 실제 재량 전략: 증거신호(바닥→롱/천장→숏)로 chop을 페이드해서 먹다가, 돌파로 추세 전환 시
손절 필요. GBM2(추세/횡보 2-class, 같은 날 앞서 배포)를 진입게이트/손절트리거로 각각 다르게 쓰는
백테스트(Stream 1)와, 손절유발을 사전에 예측하는 신모델(Stream 2)을 시도했다.

## Stream 1 — 결과: 채택(효과는 작음)

- **진입 인내심(k_entry, GBM2 confirmed-chop 지속 요구 봉수) — 가설 반박**: 1/3/6/12봉 그리드에서
  더 오래 기다릴수록 총수익률이 일관되게 악화(orthogonal_combo -52.9%→-75.9%, short_term_return_z
  -69.1%→-79.0%). 해석: 오래 지속된 chop은 "안전"이 아니라 "곧 끝날 때가 됨"에 더 가까움.
  **k_entry=1(추가 인내 없음) 채택.**
- **손절(GBM2 raw trend_prob + 미실현손실 조건) — 1차 결과 불분명, 진단 후 개선**: theta_exit∈
  {0.5,0.6,0.7}, persist=1 첫 그리드는 총수익률에 뚜렷한 방향성 없음. 트레이드 단위 1:1 대조
  진단(1,348건 중 47건만 개입, 그중 36건은 자연SL 대비 평균 4.6봉 일찍 끊어 +0.29%p/건 이득, 11건은
  TP로 갔을 트레이드를 오발동으로 손실 전환 -0.84%p/건 손해 — 순효과는 양수지만 개입 빈도가 낮아
  전체 합산에서 노이즈에 묻힘)로 원인 규명. "지속시간(persist) 조건 추가로 오발동을 거른다"는
  가설을 2/3봉 지속조건으로 테스트했으나 **반박**(지속조건은 좋은 캐치까지 걸러내 효과를 오히려
  줄임 — persist 1→3로 갈수록 개선폭 축소, 두 신호 모두 일관). 대신 **임계값을 낮춘(theta=0.5)
  단일봉 트리거가 최선**: orthogonal_combo +1.52%p, short_term_return_z +3.47%p (theta=0.6 대비도
  개선).
- **최종 채택 설정**: k_entry=1, theta_exit=0.5, regime_persist_bars=1.
- **한계**: 개선폭(+1.5~3.5%p)은 기저 손실 규모(6윈도우 합산 -50~70%)에 비해 작음 — 전략을
  흑자로 전환하지 못함, "덜 나쁘게" 만드는 수준. beats_benchmark는 이 6윈도우의 강추세 구간
  buy&hold가 구조적으로 못 이기게 돼 있어(기존 cost-gate 결과와 동일 원인) 참고 지표로만 취급.

산출물: `scripts/backtest_eth_evidence_signal_regime_entry_exit_20260827.py`(1차 그리드),
`scripts/backtest_eth_evidence_signal_regime_stop_persistence_20260827.py`(persist 재검증),
`tmp/eth_evidence_signal_regime_entry_exit_20260827/report.json`,
`tmp/eth_evidence_signal_regime_stop_persistence_20260827/report.json`.

## Stream 2 — 결과: 기각(REJECTED, 신호 없음)

**목표**: chop-게이트된 증거신호 트리거 시점에 "이 트레이드가 48봉 안에 SL을 맞을지"를 사전
예측(HistGradientBoostingClassifier). 2026-05-30 `regime3_transition_h6_risk_prob`(AUC 0.676로
기각)과 달리 지평을 실제 보유기간(최대 48봉)에 맞추고, 라벨을 경제적 결과(SL 적중 여부)로,
피처에 증거신호 자체의 연속값(orthogonal_combo의 p_fast/p_slow/delta_z/funding_z,
short_term_return_z의 ret3_z)을 포함시켜 더 나은 결과를 기대했음.

**결과 — 두 신호 모두 실질적으로 우연 수준**:

| 신호 | 표본(TRAIN/내부홀드아웃/실OOS) | 내부 홀드아웃 bal_acc/AUC | 실OOS bal_acc/AUC |
|---|---|---|---|
| orthogonal_combo | 762/191/39 | 0.558 / **0.477** | 0.505 / 0.574 |
| short_term_return_z | 1516/380/112 | 0.510 / 0.516 | **0.437** / **0.429** |

실OOS(2026-07~08, 7주)는 표본이 39~112건으로 너무 작아 단독 신뢰 불가 — TRAIN 내부의 더 큰
시간분리 홀드아웃(191/380건)으로 교차검증한 결과, 두 신호 모두 0.50 근처에 흩어져 있고 방향도
일관되지 않음(bal_acc/AUC가 서로 반대 부호로 벗어나는 경우 존재) — **실제 예측력이 없다는 뜻**.
short_term_return_z의 실OOS 0.437/0.429(우연보다 낮음)는 표본이 작아 생긴 노이즈로 해석, 더 큰
홀드아웃의 ~0.50 근처 결과가 더 신뢰할 만한 신호.

**결론**: 05-30 시도(AUC 0.676)보다 오히려 약한 결과. 지평·라벨·피처를 실제 용도에 맞게 다르게
설계해도 예측력이 개선되지 않았다는 것은, 애초에 "이 시점에 이 트레이드가 SL을 맞을지"가 현재
가용 피처로는 안정적으로 학습 불가능한 문제라는 05-30 정책의 결론을 반박하기보다 재확인하는
쪽에 가깝다. **재시도 시 이 문서부터 확인 — 같은 프레이밍(경제적 SL-적중 라벨, GBM2 chop게이트,
증거신호 연속값 피처) 반복 금지.**

산출물: `scripts/train_eth_breakout_stopout_risk_20260827.py`,
`data/ensemble/reports/eth_breakout_stopout_risk_20260827_report.json`,
`tmp/eth_breakout_stopout_risk_20260827/model_*.joblib`(참고용으로 보존, 라이브 배포 안 함).

## 종합 결론

이 세션에서 시도한 두 갈래(규칙기반 재설계/예측모델) 모두 chop-fade 전략을 흑자로 만들지 못했다.
Stream 1의 소폭 손실축소(+1.5~3.5%p)만 실질적 성과 — 규칙 그대로(k_entry=1, theta_exit=0.5)
재량 손절 판단의 보조 참고선으로 쓸 수는 있으나, 자동화된 엣지로 승격할 근거는 아니다.
