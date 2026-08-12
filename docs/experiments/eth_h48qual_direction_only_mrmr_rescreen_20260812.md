# ETH h48qual — Direction-only mRMR/knockoff 재스크리닝 (2026-08-12)

## 배경

신규 탐색 축 스카우팅([eth_h48qual_direction_skill_new_directions_scouting_20260812.md](eth_h48qual_direction_skill_new_directions_scouting_20260812.md))
(a)-1 후보 실행. FINAL12는 direction_head(`zigzag_action`)와 quality_head(`h48_conservative`)
각자 타겟에 대해 독립 mRMR을 돌리긴 했지만, 최종 12개 리스트는 두 relevance를 병합해서
만들어졌고 공통 윈도우도 2025 상반기 6개월뿐이었다 — **"direction 단독 기준 상위 N개" 리스트
자체가 만들어진 적이 없다**(`eth_h48qual_final12_feature_selection_20260811.md` 재확인).
이 문서는 quality 타겟을 완전히 배제하고 `zigzag_action` 3-class MI만으로 처음부터 다시
순위를 매긴다.

## 방법

- 스크립트: `scripts/rescreen_eth_h48qual_direction_only_mrmr_20260812.py`
- **피쳐 풀 소스 변경**: 기존 rescreen들이 쓰던 `fa_features.parquet`(2025-only, 세션 scratchpad
  전용, 이번 세션에 `tmp/eth_h48qual_fa_features_backup_20260812/`로 긴급 백업 완료 — 계약 문서가
  이미 경고한 소멸 위험을 해소)는 **이번엔 안 씀**. 대신 **커밋된**
  `data/splits/year_oos/eth_features_2024_2026_analysis.csv`(zig075 소스 패널, 145 raw 컬럼,
  2024-06-01~2026-08-04 커버)를 1차 풀로 사용 — fa_features.parquet의 2025-only보다 넓고 실제
  VAL/OOS 구간까지 포함한다. 이 패널엔 M7/AI teacher 컬럼이 아예 없어 Model Architect의
  2026-05-27 정책("direction-family M7/AI outputs는 active/candidate 입력에서 제외")을 별도
  필터링 없이 자동으로 지킨다.
- **TRAIN 윈도우**: 2024-06-01~2025-09-30(패널의 실제 커버리지에 맞춤) — canonical
  TRAIN(2024-01~2025-09, 183,936행)보다 5개월(2024 Jan~May) 짧은 부분집합이다. **VAL/OOS는
  canonical과 동일**(2025-10~12/2026-01~02, 이 패널이 실제로 커버).
- 이 세션 표준 오염 관례 그대로: `DENY_PREFIXES`/`DENY_TOKENS`(교사/미래/라벨 토큰),
  `PRICE_LIKE`(`sum_open_interest_value`), `REPLACE`(funding_pressure→diff1,
  whale_retail_ratio/count_long_short_ratio/sum_toptrader_long_short_ratio/funding_abs/
  long_squeeze_risk/squeeze_power/last_funding_rate→dt288) — `rescreen_eth_h48qual_quality_regression_pool201_20260811.py`의
  deny-list/REPLACE를 그대로 재사용(m7_vae_error는 이 패널에 없어 제외).
- **relevance**: `mutual_info_classif(feature, zigzag_action)`, TRAIN 전용, quality 타겟과
  섞지 않음.
- **mRMR**: TRAIN 상관행렬 기준 순차 그리디 선택(top25) → `|r|>0.5` 하드 중복제거.
- **오염도 체크**(표준 절차): 생존 후보 전부 `corr(close)` 확인, 배제 기준 0.561.
- **검증**: MI 랭킹만으로 끝내지 않고, 튜닝 없는 단일 LightGBM fit(early stopping만)으로
  FINAL12(이 패널에서 구할 수 있는 9개) 단독 vs FINAL12+신규후보의 VAL/OOS
  balanced_accuracy/macro_f1을 직접 대조.

## 결과 — direction-only relevance

MI 상위 20(TRAIN, n=140,226): `cvp_regime`(0.416) > `funding_roc_288`(0.216) >
`funding_pressure_diff1`(0.215) > `funding_roc_48`(0.164) > `vwap_dist_24`(0.151) >
**`mtf_trend_1h`(0.150)** > **`kalman_velocity`(0.149)** > **`rsi`(0.137)** > `ou_halflife`
(0.125) > **`mtf_trend_4h`(0.121)** > **`hma_slope`(0.114)** ... — 굵게 표시한 5개는 FINAL12에
없던 **고전적 기술적 방향 지표**(멀티타임프레임 추세, 칼만 속도, RSI, 이동평균 기울기)로,
quality 타겟과 섞인 기존 스크리닝에서는 상위권에 오르지 못했을 가능성을 보여준다.

mRMR top25(오염도 통과 25/25, 전부 `|corr(close)|<0.15`) → `|r|>0.5` 중복제거 후 **9개 생존**:
`cvp_regime`, `funding_pressure_diff1`, `mtf_trend_1h`, `funding_roc_48`, `ou_halflife`,
`funding_roc_12`, `breakout_strength`, `trades`, `sig_trend_health`. 위 5개 신규 후보 중
`kalman_velocity`/`rsi`/`mtf_trend_4h`/`hma_slope`는 `mtf_trend_1h`/`cvp_regime`과 높은 상관으로
중복제거 단계에서 탈락 — **`mtf_trend_1h`만 살아남음**.

**FINAL12 대비**: 겹침 5/12(`cvp_regime`, `funding_pressure_diff1`, `ou_halflife`,
`funding_roc_48`, `breakout_strength`) — 즉 FINAL12의 절반 가까이가 direction-only 기준으로도
여전히 살아남아 원래 선택이 완전히 잘못됐던 건 아님을 보여준다. FINAL12에만 있고 이번 랭킹에서
탈락한 7개(`m7_vae_error_dt288`, `realized_skewness`, `mta_funding`, `sig_whale_dt288`,
`sum_toptrader_long_short_ratio_dt288`, `vwap_dist_24`, `regime3_current_sensitive_wide24_chop_prob`)
중 3개(`m7_vae_error_dt288`, `sig_whale_dt288`, `regime3_current_sensitive_wide24_chop_prob`)는
애초에 이 패널에 없어 공정 비교가 안 됨(패널 세대 차이) — 나머지 4개는 direction-only 기준으로
진짜 순위가 밀린 것. **신규 후보 4개**(FINAL12엔 없음): `mtf_trend_1h`, `funding_roc_12`,
`trades`, `sig_trend_health`.

## 결과 — 가벼운 홀드아웃 비교

| 구성 | VAL balanced_acc | VAL macro_f1 | OOS balanced_acc | OOS macro_f1 |
|---|---:|---:|---:|---:|
| FINAL12(패널가용 9개) 단독 | 0.469 | 0.446 | 0.464 | 0.443 |
| FINAL12 + 신규후보 4개(13개) | **0.476** | **0.461** | **0.471** | **0.452** |

**VAL/OOS 둘 다, 두 지표 다 같은 방향으로 소폭 개선**(+0.7~1.5pp). 방향이 일관되다는 점은
고무적이지만, **단일 fit(튜닝·시드평균 없음)이라 이 프로젝트 표준([[tabm_hp_low_signal_pattern]]:
시드간 표준편차가 이런 작은 효과크기를 흔히 초과함)으로는 아직 신뢰할 수 없는 예비 신호일
뿐**이다. 참고로 이 비교의 "FINAL12"는 패널에 없는 3개 컬럼이 빠진 9개 버전이라, 진짜
FINAL12(12개)와의 직접 비교는 아니다(캐비어트로 유지).

## 결론

**예비적으로 긍정적 — 첫 세션에서 나온 유일한 "같은 방향, VAL과 OOS 둘 다" 개선 신호**지만
과잉해석 금지. 다음 단계로 값어치가 있는 것: (1) `mtf_trend_1h`(멀티타임프레임 1시간 추세)를
FINAL12에 추가한 **FINAL13** 후보로 N≥5 진짜 무작위 시드 GBDT/TabM 정식 검증(이 프로젝트
표준 always-short 대조 포함) — 이게 진짜 "새 신호"인지 "단일 fit 노이즈"인지 가르는 유일한
방법. (2) `funding_roc_12`/`trades`/`sig_trend_health`도 같은 절차로 개별/조합 확인.
(3) 이 결과가 N≥5 시드에서도 재현되지 않으면, 이 라인도 오라클 라벨/GBDT 백본 진단과 같은
"FINAL12(및 인접 확장) 자체의 정보량 부족" 결론에 합류하게 된다 — 아직 그렇게 단정하기엔
이르다.

## 부수 조치

`fa_features.parquet`(127M)·`fa_labels.npz`·`fa_meta.json`·관련 mRMR/knockoff 스크립트 3종을
`tmp/eth_h48qual_fa_features_backup_20260812/`로 백업 완료 — 계약 문서 데이터 리소스 등록부가
경고한 소멸 위험(세션 scratchpad 전용, git 추적 밖) 해소. 이 파일들은 여전히 M7/AI teacher
컬럼을 포함하고 있어(2025-only) 향후 "M7/AI risk/quality 계열만" 정책 하에 재활용 여지가
남아있다 — 이번 재스크리닝에는 정책상 direction-family가 배제돼야 해서 안 썼을 뿐.

## 산출물

`tmp/eth_h48qual_direction_only_mrmr_rescreen_20260812/` — `direction_only_mrmr_result.json`,
`holdout_comparison.json`.
