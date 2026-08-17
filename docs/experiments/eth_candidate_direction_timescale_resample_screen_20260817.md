# ETH 멀티-타임스케일 리샘플링 방향성 존재 스크린 (2026-08-17)

## 질문

"오디세이의 방향 예측 실패가 5분봉 해상도 탓인가 — 30분봉/1시간봉으로 리샘플링하면 방향성이 존재하는가."

같은 5m 원천 데이터를 15m/30m/1h/2h/4h 네이티브 봉으로 리샘플링하고, **모든 타임스케일에 동일한 방법론**을 적용해 다음-봉 방향 예측력의 존재 여부를 측정한다. 이것은 모델 승격 실험이 아니라 **존재 스크린(cheap gate)** 이다.

## 선행 클로저 — 이 실험이 서 있는 자리

| 선행 | 결과 | 이 실험과의 관계 |
|---|---|---|
| ETH 5m 방향 축 (probe 220후보 OOS AUC≤0.539, 40+ 라벨, 전 아키텍처 N≥5시드) | 전부 CLOSED | 5m 베이스라인 — 본 스크린의 대조군 |
| `btc_1h_native_swing_entry` (2026-08-07, LightGBM 5랜덤시드+purge/embargo+VAL-only 선택) | **CLOSED** — VAL 5/5 양수 → OOS 0/5 (중앙값 -13.67%) | 1h 리샘플링 가설의 BTC 직접 반증. 단 BTC≠ETH |
| ETH Sigma3-1h (HGB 방향, trend-scan 라벨, Tau1 두 번째 레그) | **CLOSED 2026-08-07** — N=8 랜덤시드 게이트에서 OOS 부호 반전 | ETH 1h의 1개 특정 라벨/모델 반증. "1h 전반의 방향성 부재" 증명은 아님 |
| `eth_weekly_tsmom_bias_cheap_gate_20260817` | CLOSED — mom1 벤치마크 백테스트 VAL/OOS 완전 반전 | 주간 리샘플링 반증 |
| 문헌 호라이즌 공백 (`docs/eth_direction_alpha_non_microstructure_research_20260817.md` §3.2) | 엄격한 증거는 ~30분(Eross et al. 2021, *Financial Review*, DOI:10.1111/fire.12290 — 유동성 공급 메커니즘)과 주간+(Liu–Tsyvinski 계보)에 양극화, **시간~일 중간 호라이즌은 문헌 공백** | 30m가 문헌상 유일하게 증거 있는 일중 스케일 — 본 스크린의 최우선 관찰 대상 |

미검정 공백: **ETH 30m 네이티브**, 그리고 "동일 방법론 전 타임스케일 나란히 비교"라는 측정 자체.

## 데이터·스플릿

- 원천: `data/rl_training_{2024,2025,2026}_unified.csv` 5m OHLCV(+volume, trades, taker_buy_base). 2024-01-01 ~ 2026-02-28.
- 리샘플링: 5m→{15m, 30m, 1h, 2h, 4h}, UTC 경계 정렬, OHLC=first/max/min/last, volume류=sum. 봉은 종료 시각에 확정된 것으로 취급(다음-봉 라벨만 사용, 미래 row 조인 없음).
- 스플릿(캐노니컬): TRAIN 2024-01-01~2025-08-31 / VAL 2025-09-01~2025-12-31 / OOS 2026-01-01~**2026-02-28** (원천 데이터 한계로 캐노니컬 03-31에서 잘림 — 명시).
- fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.

## 검정 설계 (실행 전 고정)

타깃: `y_ret = log(close_{t+1}/close_t)`, `y_sign = sign(y_ret)` (0 수익률 봉은 분류에서 제외).

**A. 모델프리 통계** (타임스케일×스플릿별)
- AC(1)~AC(12) of log returns.
- Lo–MacKinlay 분산비 VR(q), q∈{2,4,8}, 이분산-강건 z.
- 부호 지속률 P(sign_{t+1}=sign_t) vs 이항 귀무.

**B. 피쳐 IC 스크린** — 동일 레시피 전 스케일 적용: lag 수익률(1,2,3,6,12,24봉), RSI14, MACD hist, BB width, ATR%, 변동성 z, 거래량 z, taker buy 비율, 봉내 종가 위치. Spearman(feature_t, y_ret_{t+1}) 3-split. 귀무: 타깃 원형 시프트(circular shift) 200회 permutation z.
- 통과 기준: TRAIN |z|≥2 **AND** 3-split 부호 일치.

**C. GBM 프로브** — LightGBM binary(up/down), B의 피쳐 전부. 시드 5개 = `np.random.default_rng(20260817).integers(1, 999_999, 5)` (진짜 랜덤, CLAUDE.md 시드 정책). 스플릿 경계 purge/embargo 24봉. HP는 BTC 1h 게이트 값 준용(고정, 스윕 없음 — 스크린이므로).
- 지표: split별 AUC(시드 중앙값), 방향-only 정확도.

**D. 경제성 벤치마크 백테스트** — p_up>0.5+τ 롱 / <0.5−τ 숏, τ∈{0.00, 0.03, 0.05}는 VAL에서만 선택, 포지션 변경 시 왕복 10bp. 포지션은 close_t→close_{t+1}. 벤치마크는 **max(always_long, always_short)** (드리프트-as-스킬 방지). 봉당 |수익률| 중앙값(bp) vs 10bp 비용도 병기.

## 판정 규칙 (사전 고정)

타임스케일 T에 "방향성 존재" 판정 = 아래 **둘 다**:
1. (C) VAL·OOS **양쪽** 시드-중앙값 AUC>0.5, 부트스트랩(1000회) z≥2 최소 한쪽 + 다른 쪽 부호 일치.
2. (D) VAL에서 고른 τ로 OOS net PnL > max(always) OOS net PnL.

하나만 충족 → "부분 신호(추가 검증 필요)". 둘 다 미충족 → 해당 스케일 기각. 6개 스케일 동시 검정이므로 단일 스케일의 경계선 통과(z≈2)는 다중비교 할인 대상.

주의: 이 스크린을 통과해도 승격 근거가 아니다 — Omega Artifact Integrity/Fresh-Forward 게이트는 별도. OOS 2026-01~02 윈도우는 ETH 연구 라인들이 반복 관찰한 윈도우이므로, 통과 시 확증은 forward 데이터로만 가능.

## 결과 (2026-08-17 실행, 시드=[187151, 202281, 660298, 727738, 956431])

### 판정: 전 타임스케일 "방향성 존재(경제적)" 기각. 30m/1h 리샘플링은 해법이 아니다.

| 스케일 | GBM AUC VAL(중앙값/z) | GBM AUC OOS(중앙값/z) | IC pass 피쳐 수 | OOS net(선택 τ) vs max(always) | 사전등록 판정 |
|---|---|---|---|---|---|
| 5m | 0.525 / z=8.6 | 0.533 / z=7.7 | 10 | -19,208bp vs +4,481bp | 부분 신호(통계만) |
| 15m | **0.540** / z=7.8 | **0.536** / z=4.9 | 11 | -10,078bp vs +4,522bp | 부분 신호(통계만) |
| 30m | 0.528 / z=4.2 | 0.530 / z=3.0 | 12 | -7,429bp vs +4,539bp | 부분 신호(통계만) |
| 1h | 0.544 / z=4.3 | **0.494 / z=-0.4** | 7 | -11,900bp vs +4,606bp | **기각(OOS 부호 반전)** |
| 2h | 0.528 / z=2.1 | 0.507 / z=0.3 | 4 | -4,118bp vs +5,012bp | 기각 |
| 4h | 0.514 / z=0.7 | 0.479 / z=-0.6 | 0 | -6,084bp vs +5,188bp | 기각 |

### 판독 3줄

1. **통계적 예측력은 5m/15m/30m에 실재한다** (VAL·OOS 양쪽 AUC>0.5, z≥3). 그러나 정체는 새 정보가 아니라 **이미 알려진 단기 리버설**이다 — IC pass 피쳐 전원이 음의 부호(ret_1~ret_24, rsi14, taker_ratio 모두 음수). 15m OOS gross는 τ=0.05에서 +2,862bp/2,435플립 ≈ **플립당 ~1.2bp** — 정보시간 샘플링 클로저의 "gross ≤1.3bp vs 왕복 ~10bp 비용"과 정확히 같은 크기. 30m은 gross조차 ≈0.
2. **1h 이상에서는 통계 신호 자체가 죽는다.** 1h는 VAL AUC 0.544(6개 스케일 중 최고)가 OOS 0.494로 반전 — BTC 1h 네이티브 게이트(VAL 5/5 양수→OOS 0/5)·ETH Sigma3-1h(8시드 OOS 반전)와 동일 시그니처의 **세 번째 독립 재현**. 원인 후보는 샘플 수 붕괴(TRAIN 14.6k봉, OOS 1.4k봉)로 VAL 선택이 노이즈 적합이 되는 것.
3. **리샘플링은 신호/비용 비율을 개선하지 못한다.** 봉이 커지면 봉당 |수익률|(8.5→28bp)은 커지지만 AUC가 같이 죽어서, 6개 스케일 전부 OOS에서 max(always) 벤치마크에 완패. "5분봉이라서 방향을 못 찾는다"는 가설은 기각 — 방향 정보는 어느 캘린더 타임스케일에도 (비용을 이길 크기로는) 없다.

### 모델프리 통계 각주

AC(1)·분산비·부호지속 전부 |AC1|<0.07 수준으로, 유의해 보이는 항목(5m AC1 z=-3.5 등)도 스플릿 간 부호가 뒤집힌다(5m TRAIN/VAL 음수→OOS 양수). 봉 크기를 키워도 자기상관 구조가 생기지 않는다.

### 남는 것

- 15m/30m의 통계 신호는 **maker 체결(비용 <2bp) 경로**에서만 경제성 후보가 될 수 있다 — 이는 이미 세션 분할 클로저가 남긴 것과 같은 결론이며, 새 축이 아니다.
- 최초 스크린은 OHLCV 파생 피쳐 우주에 한정. 새 정보 원천(GEX, L2/OFI)은 별도 축에서 진행 중이며 이 결과와 독립.

## 부록: 전체 피쳐(150개)로 확장 재검정 (사용자 지적 반영, 2026-08-17)

**질문**: 위 결과는 15개 손수 제작한 OHLCV 파생 피쳐만 썼다 — unified 데이터셋에 있는 나머지 ~170개 컬럼(m7_* 메타모델 출력, ai_* 신호, cvp_*, regime_*, whale_*, ofi/ofti, hurst, funding_* 등)은 확인했는가?

**답**: 아니었다. 확장 스크립트 `scripts/research_eth_direction_timescale_resample_screen_fullfeat_20260817.py`로 재검정.

**피쳐 선정**: `data/rl_training_{2024,2025,2026}_unified.csv` 전체 수치형 컬럼(198개)에서 OHLCV·`target_*` 라벨 제외 → 187개 후보. TRAIN/VAL/OOS 각 구간 95% 미만 커버리지 컬럼 29개 제외(그중 `pred_mdjd`/`conf_mdjd`는 2025-12-31에 데이터가 끊기는 폐기된 모델 산출물로 확인, 2026-01-01부터만 존재하는 컬럼 27개는 별도 폐기 축의 신규 피쳐로 TRAIN 데이터가 없어 제외). 가격-트렌드 오염 체크(TRAIN Spearman(피쳐,종가) 절대값>0.5, 리포 정책)로 6개 추가 제외: `sum_open_interest_value`(0.553), `squeeze_power`(0.591), `m7_entry_long_price`/`m7_entry_short_price`/`m7_tp_price`/`m7_sl_price`(전부 0.999~1.0 — 절대가격 레벨이라 당연). **최종 150개 피쳐** 사용. 비-OHLCV 피쳐의 리샘플링은 마지막 관측값(해당 상위 봉 종료 시점에 실제로 알 수 있는 값)을 사용해 causal하게 처리.

**결과 — 결론 불변, 과적합 패턴 추가 발견**:

| 스케일 | IC pass | TRAIN AUC(중앙값) | VAL AUC(z) | OOS AUC(z) | OOS net(선택 τ) vs bench |
|---|---|---|---|---|---|
| 5m | 58/150 | 0.695 | 0.525(z=9.1) | 0.530(z=7.5) | -25,155bp vs +4,537bp |
| 15m | 50/150 | 0.813 | 0.536(z=7.7) | 0.527(z=3.5) | -13,082bp vs +4,522bp |
| 30m | 50/150 | 0.900 | 0.532(z=4.8) | 0.531(z=2.8) | -6,844bp vs +4,538bp |
| 1h | 35/150 | 0.981 | 0.521(z=2.1) | 0.520(z=1.3, **비유의**) | -5,792bp vs +4,606bp |
| 2h | 29/150 | 0.999 | 0.531(z=1.9) | 0.518(z=0.9, 비유의) | -1,654bp vs +5,012bp |
| 4h | 15/150 | **1.000** | 0.507(z=0.5) | 0.511(z=0.4, 비유의) | -827bp vs +5,188bp |

1. **경제성 판정은 불변**: 6개 스케일 전부 여전히 OOS net이 max(always) 벤치마크에 완패. 150개 피쳐로 늘려도 존재 스크린 통과 스케일은 0개.
2. **새로 드러난 문제 — TRAIN AUC가 표본/피쳐 비율에 정확히 비례해 치솟는다**: 5m(TRAIN 175k행/150피쳐) 0.695 → 4h(TRAIN 3.6k행/150피쳐) **1.000(완전 암기)**. VAL/OOS AUC는 이 TRAIN 상승과 무관하게 15피쳐 버전과 거의 동일한 수준(5m/15m/30m)이거나 더 낮다(1h~4h는 OOS z<1.4로 통계적으로 무작위와 구분 불가) — 피쳐를 135개 추가해도 일반화 가능한 정보는 늘지 않았고, 오히려 coarse 스케일에서 과적합 위험만 커졌다.
3. **IC 통과 피쳐 수가 스케일이 커질수록 단조 감소**(58→15) — AUC z 붕괴와 같은 패턴. "더 많은 피쳐 = 더 많은 신호"가 아니라 표본 수 붕괴가 지배적이다.
4. **해석 주의사항**: `close_btc`(BTC 절대가격)가 거의 모든 스케일에서 상위 중요도 피쳐로 반복 등장(TRAIN rho=0.26, 리포 오염 임계값 0.5 미만이라 필터되지 않음). 절대가격 레벨은 트리 분할이 "시대"(TRAIN/VAL/OOS 각기 다른 가격대)를 암묵적으로 학습할 위험이 있다 — 임계값 미달이라 배제하지 않았으나, VAL/OOS AUC가 여전히 무의미한 수준인 것을 보면 이 우려가 실제로 결과를 왜곡하진 않은 것으로 보인다.

**결론**: 100+개 전체 피쳐로 확장해도 원래 스크린의 판정(6/6 스케일 경제성 기각, 1h+ 통계신호 실종/비유의)은 그대로 유지된다. 산출물: 스크립트 `scripts/research_eth_direction_timescale_resample_screen_fullfeat_20260817.py`, 결과 `docs/experiments/eth_direction_timescale_resample_screen_fullfeat_20260817_results.json`.

## 실행 산출물

- 스크립트: `scripts/research_eth_direction_timescale_resample_screen_20260817.py`
- 결과 JSON: `docs/experiments/eth_direction_timescale_resample_screen_20260817_results.json`
