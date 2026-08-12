# ETH h48qual — Long/Short/Cash 독립 3모델(one-vs-rest) 검증 + OOS·POST_OOS 격차 메커니즘 진단 (2026-08-12)

## 배경

사용자 제안("그럼 롱과 숏과 캐시를 따로 데이터를 취합해서 모델을 따로 만드는게 어때"): 공유
3-class softmax(CASH/LONG/SHORT) 대신 완전히 독립적인 이진분류기 3개(LONG-vs-rest,
SHORT-vs-rest, CASH-vs-rest)를 각자 학습해 확률을 비교(argmax)해서 최종 행동을 정하는 방식.
2026-08-11 진단(direction confidence calibration)에서 공유 3-class 모델의 클래스 간 확률
간섭이 관찰된 것이 동기.

## 방법

- 스크립트: `scripts/train_eval_eth_h48qual_onevsrest_specialist_20260812.py`(1차 실행 + HP
  탐색), `scripts/verify_eth_h48qual_onevsrest_reproduction_check_20260812.py`(재현성 검증,
  기존 winners.json 재사용 + 완전히 새 시드), `scripts/diagnose_eth_h48qual_onevsrest_hardregime_pilot_vs_softweight_always_short_20260812.py`(메커니즘 진단).
- zig075 소스 패널(145컬럼) + `zigzag_action` 라벨, `RAW_WIDE` 24피쳐(log_return,
  volatility_z, rsi, macd_hist, bb_width_z, wick_ratio, net_taker_ratio, cvd 계열, funding
  계열, btc_ret 계열, parkinson_vol, hurst_48, kalman_velocity, mtf_trend 계열).
- 헤드별(cash/long/short) 독립 Optuna 탐색(15 trial, TRAIN 내부 월별 확장윈도우 CV,
  embargo 48bar) → LightGBM 이진분류기, `class_weight_mode`(none/balanced) 튜닝 포함.
- 최종 N=5 진짜 무작위 시드(고정 간격 아님)로 각 헤드 독립 학습 → 3개 확률 argmax로 최종
  행동 결정 → omega 거래 시뮬레이션(`_metrics`)으로 always-short/always-long과 대조.
- **오늘의 표준 절차대로 POST_OOS 월별 분해를 첫 실행에 포함**(TCN 재현 실패 사례 이후 확립된
  습관).

## 결과 1 — 1차 실행 (seeds: 608345233, 877989137, 404117503, 267053963, 756316345)

| 구간 | 결과 |
|---|---|
| VAL | balanced_acc=0.504±0.002, PnL 혼재(1/5, 3/5, 1/5 승 vs short, cost1/2/3) |
| OOS | balanced_acc=0.501±0.001, **완패**(0/5, 0/5, 0/5) |
| POST_OOS | **5/5 전승**(cost1/2/3 전부), Wilcoxon p=0.0312(cost1/2/3) — 이 세션 최강 신호 |

POST_OOS 월별 분해(cost3): 3월 +3.17, 4월 -2.44, 5월 +3.37, 6월 -3.46, 7월 +4.43, 8월 -0.25.
월별복리 누적: model +4.64% vs always_short -1.77%.

## 결과 2 — 재현성 검증(완전히 새로운 N=5 시드, HP 재탐색 없이 winners.json 재사용)

seeds: 568534410, 813139642, 888095143, 96659357, 557791022.

| | 1차 실행 | 2차 실행(새 시드) |
|---|---|---|
| OOS | 0/5 완패(전 비용) | 0/5 완패(전 비용) — **동일** |
| POST_OOS cost1 | 5/5, p=0.03 | 4/5, p=0.09(약화) |
| POST_OOS cost2 | 5/5, p=0.03 | 5/5, p=0.03 — **재현** |
| POST_OOS cost3 | 5/5, p=0.03 | 5/5, p=0.03 — **재현** |
| POST_OOS 월별복리 | +4.64% vs -1.77% | +0.73% vs -1.32%(폭 6배 축소, 방향 동일) |

7월(+18.3% 가격)은 두 실행에서 model +4.43/always_short -8.16으로 **완전히 동일** — 강한
단방향 상승월에서는 시드와 무관하게 모델 행동이 수렴하는 것으로 보임. OOS는 두 독립 시드셋
모두 완패로 흔들림 없음.

**중간 결론**: TCN(재검증에서 완전 역전)이나 CNN(애초 유의하지 않음)과 달리, 이번 결과는
방향이 재현되지만 크기가 불안정한 중간 케이스 — 단정하지 않고 메커니즘 자체를 파고들기로
결정(사용자 지시 "파고들어가보자").

## 결과 3 — 메커니즘 진단: 왜 OOS는 완패, POST_OOS는 재현되는가

완전히 새로운 N=5 시드(224911649, 910729423, 661872615, 924875728, 41056711)로 재학습,
전체 구간 bar 단위 예측을 저장해 4가지 진단을 실행.

### (1) 피쳐 PSI drift — TRAIN 대비 OOS vs POST_OOS

평균 PSI: OOS=0.032, **POST_OOS=0.062**(오히려 2배 더 드리프트됨). 상위 드리프트 피쳐
(`cvd_288`, `parkinson_vol`, `eth_btc_ret_spread_12`)도 전부 POST_OOS 쪽이 더 큼. **"OOS가
낯선 분포라 실패한다"는 가설은 기각** — 분포이동은 오히려 모델이 이기는 POST_OOS 쪽이 더 크다.

### (2) 모델 예측 클래스 비율

OOS: CASH 4.7% / LONG 46.2% / SHORT 49.2%. POST_OOS: CASH 1.0% / LONG 47.7% / SHORT 51.3%.
**거의 동일** — 행동 패턴 자체가 구간별로 크게 달라지지 않는다. CASH 재현율이 두 구간 모두
극히 낮음(아래 (3) 참고) — 모델이 거의 항상 포지션을 잡는 경향.

### (3) OOS 클래스별 정밀도/재현율(정답 라벨 존재)

| 클래스 | precision | recall | support |
|---|---:|---:|---:|
| CASH | 0.474±0.010 | **0.185±0.005** | 2,020 |
| LONG | 0.620±0.002 | 0.647±0.003 | 7,484 |
| SHORT | 0.599±0.002 | 0.673±0.004 | 7,393 |

LONG/SHORT는 ~60% 근처로 무작위(33%)보다는 확실히 낫지만 결정적이지 않음. CASH 재현율
0.185 — 쉬어야 할 때도 거의 못 쉼(포지션을 거의 안 접는 경향은 OOS 실패의 한 요인일 수 있음).

### (4) 주간 반등(bounce) vs 하락주 풀링 — OOS·POST_OOS 둘 다 (핵심 발견)

| 구성 | n_bars | model | always_short | always_long | 승(short) | 승(long) |
|---|---:|---:|---:|---:|---:|---:|
| OOS / 반등주 | 4,801 | -16.56±0.02 | +12.67 | -16.05 | 0/5 | 0/5 |
| OOS / 하락주 | 12,096 | +4.23±6.85 | **+29.21** | -26.07 | 0/5 | 5/5 |
| POST_OOS / 반등주 | 14,688 | -9.25±4.79 | -5.22 | +1.11 | 0/5 | 0/5 |
| POST_OOS / 하락주 | 30,282 | **+22.06±5.90** | **-0.80** | -3.92 | 5/5 | 5/5 |

**반등주는 OOS·POST_OOS 둘 다 모델이 손해를 본다(0/5)** — 애초에 "휩소/반등 구간에서 모델이
유리하다"는 (TCN 때 제기했던) 가설을 여기서도 그대로 재현 시도했지만, 이번에도 **그 방향으로는
성립하지 않는다**. 대신 격차는 전적으로 **하락주 안에서** 발생하며, 결정적인 것은 `always_short`
기준선 자체의 성과다:

- **OOS 하락주**: `always_short`가 +29.21로 사실상 그 구간 최대치에 가까운 수익을 냄 —
  OOS 전체가 워낙 매끄러운 단조 하락(1월 -17.6%, 2월 -19.9%, 반전 없음)이라, 계속 숏을 들고
  있기만 해도 거의 이론적 최댓값을 챙긴다. 모델은 bar마다 갈아타면서(약 60% 정확도) 방향은
  대체로 맞히지만(`always_long` 대비 5/5 승), 매끄러운 추세에서는 불필요한 회전(수수료/슬리피지,
  일시적 오판)이 고정 숏 포지션 대비 손해만 될 뿐 우위가 없다.
- **POST_OOS 하락주**: `always_short`가 -0.80으로 사실상 이득이 없음 — 주 단위로는 순하락
  이지만 그 안에 상당한 되돌림(chop)이 섞여 있어, 고정 숏 포지션이 반복적으로 SL을 맞거나
  이익을 반납당하는 것으로 추정됨. 모델은 이 노이즈 안에서 국지적 방향(짧은 하락 구간엔 숏,
  짧은 반등 구간엔 롱/캐시)을 어느 정도 갈아타면서 `always_short`의 취약점을 피해가는 것으로
  보이며, 그 결과 절대 PnL도 크게 개선(+4.23 → +22.06)되고 상대 우위(대비 +29.21→-0.80의
  30pt 붕괴)도 동시에 발생한다.

즉 POST_OOS 신호는 "모델이 새로운 능력을 얻었다"기보다 **"`always_short`라는 비교 기준선이
매끄러운 하락에서만 강하고, 하락이 거칠어지면(chop) 급격히 약해진다"**는 기준선의 취약성이
크게 작용한 결과다. OOS는 이 데이터셋 전체에서도 이례적으로 매끄러운 2개월 구간이라 어떤
"완전 커밋 방향예측이 아닌" 모델도 `always_short`를 이기기 극히 어려운 조건이었을 가능성이
높다.

## 종합 해석

1. **PSI drift 가설 기각**: 분포이동은 OOS보다 POST_OOS가 더 크다 — "낯선 데이터라 실패"
   설명은 성립하지 않는다.
2. **행동 패턴 차이 가설 기각**: 클래스 선택 비율이 두 구간에서 거의 동일하다.
3. **"휩소/반등주가 유리하다" 메커니즘(TCN 때 제기, 이번에도 재시도) 재기각**: 반등주는 두
   구간 모두 모델이 손해를 본다. 이 가설은 이번 세션에서 두 번째로(TCN, 이제 one-vs-rest)
   직접 out-of-sample 테스트로 기각됐다.
4. **실제 메커니즘**: 격차는 하락주 내부, 그리고 그 하락주가 얼마나 "매끄러운가"(=`always_short`
   기준선이 얼마나 강한가)에 의해 거의 전부 설명된다. 모델의 절대 성과도 POST_OOS 하락주에서
   더 좋아지지만(+4.23→+22.06), 상대 우위의 대부분은 기준선 붕괴(+29.21→-0.80)에서 온다.

이는 POST_OOS 신호를 "일반화되는 진짜 방향 예측 엣지"로 단정하기 어렵게 만든다 — 오히려
**OOS가 이례적으로 어려운(매끄러운 단조추세) 테스트 구간**이었을 가능성과, **POST_OOS의
"승리"가 상당 부분 비교 기준선의 취약성에 기댄 것**이라는 두 가지를 같이 시사한다. LONG/SHORT
정밀도가 60% 안팎으로 완전히 무의미하지는 않지만(무작위 33% 대비), 이 정도 정확도로는 매끄러운
추세를 이기지 못하고 거친 추세에서만 근소한 우위를 내는 수준 — 실전에 쓸 만한 독립적 방향
엣지라고 보기엔 근거가 약하다.

## 결론

**one-vs-rest 분해는 공유 3-class 모델 대비 확실한 개선을 보여주지 못했다.** OOS는 완전
부정(두 독립 시드셋 모두 0/5), POST_OOS 양성 신호는 방향은 재현되지만(cost2/cost3, 두 시드셋
모두 5/5, p=0.03) 크기가 불안정(6배 차이)하고, 메커니즘 진단 결과 그 신호의 상당 부분이 진짜
방향 스킬이 아니라 비교 기준선(`always_short`)의 레짐별 취약성에서 나온다는 것이 밝혀졌다.
[[odyssey_eth_h48qual_subproject]]의 "5개 질적으로 다른 모델 유형이 전부 OOS에서 부정 수렴"
패턴에 6번째로 합류 — 다만 완전한 무신호(TCN 재검증, CNN)보다는 미묘한 케이스로 기록한다.

## 산출물

- `tmp/eth_h48qual_onevsrest_specialist_20260812/`(1차 실행) — `winners.json`,
  `pnl_comparison.csv`, `classification.csv`, `monthly_breakdown.csv`.
- `tmp/eth_h48qual_onevsrest_reproduction_check_20260812/`(재현성 검증) —
  `pnl_comparison.csv`, `monthly_breakdown.csv`.
- `tmp/eth_h48qual_onevsrest_regime_diagnosis_20260812/`(메커니즘 진단) —
  `feature_psi_drift.csv`, `class_distribution.csv`, `oos_precision_recall.csv`,
  `bounce_vs_down_weeks.csv`.

## 참고

실행 환경: `lightgbm`은 `base` conda 환경이 아니라 `/home/kbj20/anaconda3/envs/quant_ai`에만
설치돼 있음 — 이 계열 스크립트를 재실행할 때는 `quant_ai` 환경의 python3를 명시적으로 써야
한다.
