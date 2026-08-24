# Odyssey4 TabM 진짜 102-피처 라이브 계약 파이프라인 복구 (2026-08-16)

관련 문서(이 문서가 정정하는 대상): `docs/experiments/eth_odyssey4_gce_canonical_port_20260816.md`,
`docs/experiments/eth_odyssey4_purge_embargo_gap_20260816.md`,
`docs/experiments/eth_odyssey4_loss_weight_optuna_search_20260816.md`(확인함 — 셋 다
`_prepare_frames_light()`/`feature_cols`=185개 프록시를 명시). 프로젝트 메모리
`feedback_modern_dl_training_checklist.md`,
`eth_odyssey4_layer_improvement_proposal_20260816.md`도 같은 "라이브 102피처 파이프라인이 깨져있다"는
전제를 담고 있다 — 메모리 파일은 내가 직접 수정할 수 없으므로, 이 문서를 근거로 사용자가 갱신해야 한다.

## 배경 — 무엇이 과잉 진단이었나

이 세션(및 병행 세션)은 `train_eval_omega1_2_tabm_3head_20260603._prepare_frames()`가 dev/서버
양쪽에서 `FileNotFoundError`로 막힌다는 걸 확인하고, "라이브 102(+13pos)=115차원 파이프라인 자체가
깨졌다"고 결론짓고 `_prepare_frames_light()`(`_numeric_feature_cols`로 자동 탐지한 172~185개 컬럼
프록시)로 우회했다. **이 진단은 범위가 너무 넓었다.** 실제로 직접 추적한 결과:

1. `_prepare_frames()`가 기본 피처를 가져오는
   `train_eval_omega1_2_tabm_diffusion_risk_20260603._load_omega_frames()`는 **문제없이 로드된다**
   (`TRAIN_CSV`/`EVAL_CSV`, REGIME3_* 오버레이 CSV 3종 전부 존재, `quant_ai` 환경에서 직접 실행
   확인: `train.shape=(105064, 218)`, `eval_df.shape=(16897, 237)`).
2. 진짜 깨진 부분은 훨씬 좁다: `_prepare_frames()`가 `zigzag_action` 라벨 **한 컬럼만** 얻으려고
   별도로 `train_omega1_regime3_expert_direction_head_volpca_20260602._build_frame(year)`를
   호출하는데, 이 호출 체인(`volpca.ctx._build_frame` → `tsfm_chronos._build_frame(include_core=
   True)` → `base._exact_join(..., DIR3_VSNLSTM, ...)`)이 죽은 vsnlstm CSV
   (`data/ensemble/supervised/omega1_dir3_vsnlstm_full_20260531/training_features_*.csv`, 체크포인트만
   남고 CSV는 소실)에 의존한다. 라벨의 진짜 독립 출처는
   `train_omega1_direction_head_direction_only_20260602._add_labels(year)`이며, 이 함수는
   `tmp/causal_regen_20260516/zigzag_action_labels_20260531/zigzag_action_labels_<year>.csv`를 직접
   읽는다 — vsnlstm/chronos 체인과 완전히 무관하다. 이 우회는 이미
   `scripts/research_eth_candidate_faithful_tabm_batchensemble_cheap_gate_20260816.py`의
   `_prepare_frames_light()`가 같은 패턴(`label_base._add_labels(year)`)으로 확립해뒀던 것과 동일하다
   — 이번 작업은 그걸 재사용했을 뿐, 재도출하지 않았다.
3. `_numeric_feature_cols(train, eval_df)`는 두 프레임에 공통으로 있는 숫자 컬럼을 전부 자동
   탐지한다(오늘 기준 172개 — 이전 문서들의 "185개" 기록과 다른데, 공유 CSV가 동시 세션들에 의해
   계속 갱신되고 있어 시점에 따라 달라진 것으로 보인다). 이건 진짜 102개보다 **많다** — 다른 연구
   작업이 시간이 지나며 같은 CSV에 컬럼을 누적시켰기 때문이다. 라이브 배포 번들
   (`tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_
   quality_noctx_padded_e2_fulltrain_exit30k_20260630/true_3head_tabm_bundle.pt`의
   `base_cols`)에 정확히 기록된 순서 있는 102개 리스트와 비교한 결과: **eval_df(2026)에는 102개
   전부 존재**하지만, **train(2025)에는 7개가 빠져 있다**: `fibonacci_level`, `funding_roc_12`,
   `funding_roc_48`, `funding_z_score`, `short_squeeze_risk`, `hurst_288`, `regime_persistence`.
   이 7개는 vsnlstm/chronos처럼 영구 소실된 게 아니라, `features/engineering.py`/
   `features/high_order_state.py`의 파생 공식으로 **train에 이미 있는 컬럼만으로 복구 가능**했다.

## 7개 결측 컬럼 — 원본 공식 추적 및 필요 입력

| 컬럼 | 정의 위치 | 필요 raw 입력 | train에 이미 존재? |
|---|---|---|---|
| `fibonacci_level` | `QuantSignalFeatures._fibonacci`, `features/engineering.py:1000-1009` | `high`,`low`,`close` (288봉 스윙) | 예 |
| `funding_roc_12` | `FundingRateMomentum._calculate_roc(12)`, `:1045-1049` | `last_funding_rate` | 예 |
| `funding_roc_48` | `FundingRateMomentum._calculate_roc(48)`, `:1045-1049` | `last_funding_rate` | 예 |
| `funding_z_score` | `FundingRateMomentum._calculate_zscore(288)`, `:1051-1055` | `last_funding_rate` | 예 |
| `short_squeeze_risk` | `FundingRateMomentum._short_squeeze_score`+`._funding_extreme(-1.0)`, `:1057-1088` | `last_funding_rate`, `funding_roc_12`(위에서 계산), `oi_change_rate` | 예(`oi_change_rate`) |
| `hurst_288` | `HurstExponentFeatures._rolling_hurst_fast(288)`, `:1119-1134` | `close` | 예 |
| `regime_persistence` | `add_high_order_state_features`, `features/high_order_state.py:64-73` | `mtf_trend_1h`,`mtf_trend_4h`,`hma_slope`,`chop_index`,`breakout_strength` | 예(전부) |

7개 전부의 필요 입력이 이미 `train`에 있었다 — task brief가 제시한 두 방법 중 **(a) train 자체
컬럼으로 직접 재현**을 선택했다(원시 OHLCV/funding/OI 재처리, 즉 (b) `FeatureEngineer.process()`
전체 재실행은 불필요했다). (a)는 손대는 표면이 최소라 버그 위험이 가장 낮다는 게 이유다.
`short_squeeze_risk`는 `adaptive_squeeze=False`(ETH 라이브 기본값, `FundingRateMomentum.__init__`
docstring에 SOL 전용 옵션으로 명시)를 가정 — ETH 라이브 값과 정확히 같은 고정-분모(0.0002) 모드다.

구현: `scripts/eth_odyssey4_true_feature_pipeline_20260816.py`의
`compute_missing_train_columns()` — 위 표의 각 함수를 원본과 동일한 순서(roc_12 → short_squeeze_risk가
그걸 참조하므로)로, 함수 docstring에 원본 파일:라인 인용을 남겨 재현했다.

## 검증

### 1) 공식 충실도 — eval_df(2026)로 왕복 검증

eval_df(2026)에는 이 7개 컬럼의 **진짜 라이브 파이프라인 값**이 이미 있다. 같은 재현 함수를
eval_df 자체에 적용해서 그 진짜 값과 직접 diff했다(순수 공식 검증, train 복구와 독립적):

| 컬럼 | max abs diff | mismatch(>1e-6) | 해석 |
|---|---:|---:|---|
| `fibonacci_level` | 1.11e-16 | 0/16897 | 부동소수점 정밀도 수준 — 완전 일치 |
| `hurst_288` | 5.55e-17 | 0/16897 | 완전 일치 |
| `regime_persistence` | 2.22e-16 | 0/16897 | 완전 일치 |
| `short_squeeze_risk` | 3.67e-4 | 1/16897 | 사실상 완전 일치 |
| `funding_roc_12` | 0.647 | 11/16897 | 아래 참조 |
| `funding_roc_48` | 0.647 | 47/16897 | 아래 참조 |
| `funding_z_score` | 6.85 | 287/16897 | 아래 참조 |

funding_roc_12/48/funding_z_score의 불일치는 **공식 버그가 아니라 콜드스타트 경계 효과**임을
직접 추적으로 확인했다: 불일치는 전부 eval_df의 **맨 앞** 구간에만 몰려있다(마지막 불일치 행이
각각 11, 47, 286 — 정확히 window=12/48/288 근처). 원인: 진짜 라이브 파이프라인은 연속된 시계열
위에서 도는데, 연도별로 쪼갠 이 CSV(`EVAL_CSV`)는 2026-01-01 00:00부터 시작해서 그 이전(2025-12월)
이력이 없다 — `.shift(12)`/`.rolling(288)`가 그 few-row 구간에서 진짜 라이브 값이 참조했을 12월
데이터에 접근하지 못해 `NaN→fillna(0)`이 된다. 예: `2026-01-01 00:05` 행의 진짜 `funding_roc_12`는
0.647235인데, `funding_rate`가 `0.0001`(00:05~)로 `00:00`(0.000035)에서 막 바뀐 직후라 12봉 전
값이 CSV 경계 밖(NaN)이라 우리 재현은 0을 낸다. **공식 자체는 검증됐고, 유일한 한계는 연도
경계에서 최대 window(288봉=1일) 크기만큼의 워밍업 구간**이다.

### 2) train(2025) 분포 vs eval_df(2026) 실제값 분포 (스케일 sanity check)

| 컬럼 | train2025 mean/std | eval2026 mean/std | min/max 범위 |
|---|---|---|---|
| `fibonacci_level` | 0.525 / 0.266 | 0.486 / 0.254 | 둘 다 [0,1] |
| `funding_roc_12` | -0.00018 / 0.151 | -0.0057 / 0.226 | 둘 다 [-3.3, 1.8] 부근 |
| `funding_roc_48` | -0.00071 / 0.302 | -0.0228 / 0.452 | 둘 다 [-3.3, 1.8] 부근 |
| `funding_z_score` | -0.040 / 1.389 | -0.045 / 1.448 | 유사 |
| `short_squeeze_risk` | 0.014 / 0.040 | 0.063 / 0.115 | 둘 다 [0, ~0.8] |
| `hurst_288` | 0.526 / 0.040 | 0.521 / 0.037 | 거의 동일 |
| `regime_persistence` | 0.030 / 0.392 | -0.003 / 0.402 | 거의 동일 |

스케일/범위가 전 항목에서 합리적으로 일치한다(2025/2026 시장 레짐 차이로 인한 자연스러운 평균
이동은 있음 — 특히 short_squeeze_risk가 2026이 더 높은데, 이는 2026 구간에 펀딩비 극단치가 더
잦았다는 것으로 실제 시장 데이터와 일관됨). 큰 이탈이나 스케일 붕괴는 없음 — 공식이 잘못됐다는
신호가 아니다.

### 3) NaN/Inf, 라벨 무결성, purge/lookahead

- `train`/`val`/`oos` 세 split 전부 102-컬럼 매트릭스가 **NaN/Inf 0개**(`np.isfinite(...).all()
  == True`, `prepare_frames_true()`의 `_assert_clean_102`가 강제).
- `zigzag_action` 라벨: 세 split 전부 `unique={0,1,2}`, null 0개 — `label_base._add_labels`가
  자체적으로 `{0,1,2}` 외 값이 있으면 예외를 던지므로 이중 확인됨.
- 7개 복구 컬럼은 전부 `.shift(양수)`/`.rolling(window)` 형태의 순수 과거참조 연산이고, 음수
  shift나 미래 행 참조가 없다 — 룩어헤드 없음. `regime_persistence`의 `streak`도
  `groupby(cumsum).cumcount()`로 현재까지의 연속 구간 길이만 세는 과거참조 연산이다.
- **알려진 한계(경미)**: train(2025)도 eval_df와 같은 연도-경계 콜드스타트를 겪는다 — 2025-01-01
  기준 최대 288행(전체 105,064행의 0.27%)이 진짜 라이브 값(2024-12월 이력 포함)과 미세하게 다를
  수 있다. `data/eth_5m_1year.csv`(2023-12-31~2026-02-17 커버)로 워밍업을 채우는 것도 고려했지만,
  거기엔 `last_funding_rate`/`mtf_trend_1h` 등 파생 피처가 없어(raw OHLCV만) 전체
  `FeatureEngineer.process()` 재실행이 필요해지고, 이는 영향받는 행 수(0.27%) 대비 리스크/작업량이
  과하다고 판단해 보류했다. 필요해지면 별도로 재검토.

### 4) 엔드투엔드 통합 스모크 테스트

`prepare_frames_true()`가 만든 프레임을 기존 학습 파이프라인 함수(`base3head._base_input`,
`exit_head._build_exit_dataset_independent`, `base3head._exit_input_from_position_rows`)에 그대로
통과시켜 실제 drop-in 대체가 되는지 확인:

- `x_train.shape == (78568, 115)` — 102 base + 13 pos, 라이브 계약과 정확히 일치.
- 컬럼 순서: 앞 5개 `['open','high','low','close','volume']`, 뒤 5개
  `['pos_notional','pos_leverage','pos_exposure','pos_tp','pos_sl']` — 번들의 `base_cols`/`pos_cols`
  순서와 정확히 일치.
- `x_exit.shape == (306433, 115)`, 둘 다 전 항목 finite.
- `train_raw`(78,568) + `val_raw`(26,496) = `train_all`(105,064) — 라벨 정렬(`_align`)에서 행 손실
  없음(2025 라벨 커버리지 100%). `oos_raw`(16,897) = `eval_df` 원본 크기 그대로 — 2026도 손실 없음.

## 산출물

- 신규 파이프라인 모듈: `scripts/eth_odyssey4_true_feature_pipeline_20260816.py` —
  `true_base_cols()`(번들에서 순서 있는 102개 로드), `compute_missing_train_columns()`(7컬럼 복구),
  `prepare_frames_true()`(기존 `_prepare_frames()`/`_prepare_frames_light()`와 동일한 dict 반환
  형태의 drop-in 대체), `main()`(독립 실행 시 위 검증 전부 재현·출력).
- 기존 스크립트/라이브 배포 번들/`trading_bot_modules/`는 **미변경** — 순수 연구 인프라 신규
  파일 1개만 추가.

## fresh-forward 규칙 준수

`fresh_forward_bar_by_bar=n/a`(데이터 로딩/피처 복구 파이프라인 구축 및 검증 — 백테스트/학습을
수행하지 않음), `trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`.

## 이 문서가 정정하는 내용

- `eth_odyssey4_gce_canonical_port_20260816.md`, `eth_odyssey4_purge_embargo_gap_20260816.md`
  등 이 세션의 A1/B/C1/C2/C3 전부가 "라이브 102피처 파이프라인 자체가 깨져서 172~185개 프록시로
  우회했다"고 기록했다 — **파이프라인 자체는 깨지지 않았다.** 깨진 건 라벨 조회 경로 하나뿐이고,
  프록시가 컬럼 수가 다른(그리고 더 많은) 이유는 자동 탐지(`_numeric_feature_cols`)가 진짜 102개
  계약을 반영하지 않았기 때문이다. 그 실험들의 수치 자체는 (다른 축을 보는 데는) 여전히 유효하지만,
  "라이브 승격 근거로 쓰려면 진짜 102피처 복구가 필요하다"는 각 문서의 경고는 이제 **이 파이프라인으로
  해소 가능**하다 — 재실행 여부는 각 실험의 담당 세션 판단.
- 프로젝트 메모리 `feedback_modern_dl_training_checklist.md`, `eth_odyssey4_layer_improvement_
  proposal_20260816.md`의 "진짜 라이브 102피처 파이프라인은 dev/서버 양쪽에서 깨져 있다"는 기록도
  같은 이유로 갱신이 필요하다 — 사용자가 직접 처리 예정(이 세션은 메모리 파일을 편집할 권한이 없음).
  `eth_odyssey4_layer_improvement_proposal_20260816.md`는 이미 "Coordination note" 항목에서 이
  복구 작업이 진행 중임을 정확히 예견해서 기록해뒀다(`ae5e93a53a257ff71`에게 핸드오프 예정이라는
  내용까지) — 이 문서가 그 핸드오프의 결과물이다.
