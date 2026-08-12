# ETH h48qual — `direction_head` 방향 스킬 신규 탐색 축 스카우팅: 신규 피쳐 / 신규 라벨 / 신규 구간 (2026-08-12)

**문서 성격**: 순수 리서치/랭킹 문서. 학습·승격·라이브 코드 변경 없음. 아래 "그라운딩" 절의
가격 구간 계산 하나만 예외로, 기존 raw kline CSV를 직접 읽어 검증했다 — 나머지는 전부 기존
`docs/`/`scripts/` 산출물을 읽고 인용한 것이다.

## 배경

`docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_contract.md`의 2026-08-12
"🧭 사용자 결정"이 이 문서의 출발점이다: `quality_head` 대체 리서치가 9개 후보 전부 부정 결과로
닫혔고(`docs/experiments/eth_h48qual_quality_head_replacement_research_20260812.md`,
`..._candidates_1_2_3_and_formal_skill_test_20260812.md`), 결정적으로 게이트를 완전히 제거한
`direction_head` 원본조차 N=5 다양시드·4개 그룹(h48orig/h384 × VAL/OOS) 40칸 중 단 2칸만
always-short를 이겼다(Wilcoxon 단측 p≈1.0, 전 그룹)
(`docs/experiments/eth_h48qual_ungated_direction_h48orig_5seed_vs_always_short_20260812.md`).
메타라벨링/게이팅은 구조상 1차 신호의 부분집합만 고를 수 있으므로, `quality_head`를 아무리
잘 고쳐도 `direction_head` 자체에 없는 스킬을 만들 수 없다는 게 확정됐다.

사용자는 `quality_head` 투자를 전부 보류하고, "`direction_head`(또는 이 자산/타임프레임의 어떤
방향예측 방식이든)가 **다른 조건**에서 진짜 방향 스킬을 갖는가"를 서브 프로젝트 최상위 질문으로
승격했다. 세 갈래 후보가 제시됐고(a) 신규 피쳐, (b) 신규 방향 라벨, (c) 다른 구간 — 이 문서는
셋 다 아직 착수 전인 상태에서, 각 갈래의 후보를 검증비용 순으로 랭킹한다.

## 그라운딩 — 이미 닫힌 라인 확인 (제안 전 필수 점검)

이 서브 프로젝트 자체의 반복된 교훈(`feedback_defer_wide_blast_radius_cleanup` 류의 메모리와
같은 정신)은 "이미 시도된 것을 새 이름으로 재제안하지 말 것"이다. 아래를 전부 확인했다:

- **`docs/model_contracts/research_line_registry.json`의 `prior_lines` 전체(15개)** — 특히
  `eth_overnight_generic_feature_entry_filter_20260809`(모멘텀/OFI/변동성/Hawkes/Kaufman
  ER/cross-sectional relative strength 6개 피쳐군 + 44피쳐 kitchen sink, 3개 모델링 패러다임,
  ETH+BTC+SOL, 17/17 네거티브), `btc_tpfirst_three_way_label`(TP-first 3-way 라벨, 0/24),
  `btc_barrier_horizon_calibration`(BTC 배리어/호라이즌 스윕, 0/148), `btc_rho1_panel_direction`
  (cross-sectional 패널 방향, rank score 거의 상수), `btc_zigzag_as_entry_model_component`
  (zigzag 상태를 엔트리 모델에 결합하는 모든 방식, 전부 악화).
- **`docs/entry_exit_edge_root_cause_and_literature_review_20260809.md`** — 17개 아이디어
  falsification 게이트 밤샘 세션(Part 4) + 최신 문헌 리뷰(Part 2) + DL이 이 격차를 넘긴 사례
  3건 실증 테스트(Part 5: CUSUM 이벤트바, 60코인 cross-sectional 풀링, Chronos zero-shot — 전부
  실패).
- **`docs/eth_omega4_6_1_accuracy_research_ideas_20260811.md`** — 이 계약의 원 리서치 문서, 21개
  죽은 라인과 겹치는 것 확인 완료(§2), 4개 제안(3-1~3-4) 중 3-1/3-3은 계약에 반영, 3-2는 기존
  구현, 3-4는 이미 미해결 이슈로 열려 있고 후속 검증(레짐별 threshold)도 부정 결과로 닫힘.
- **`docs/model_contracts/odyssey_eth_h48qual_data_resources_20260812.md`** — 이 서브 프로젝트가
  이미 만진 모든 코드/데이터/모델 리소스의 정확한 경로. 아래 (a)/(c) 후보들의 인프라 가용성
  판단은 전부 이 문서와 직접 조인한 것.

아래 세 절 모두, 후보를 제안하기 전에 위 네 문서와 대조해 "이미 닫힌 것"과 "닫힌 것과
개념적으로 인접하지만 differentiator가 있는 것"을 구분했다.

**공통 검증 절차 (이 서브 프로젝트 표준, 아래 모든 후보에 적용)**: (1) 신규 raw-level 피쳐는
학습 전 `corr(price)`/`corr(시간순번)` 오염도부터 확인(`feedback_raw_feature_price_trend_contamination`
— `CapMVRVCur`/`whale_retail_ratio`에서 두 번 걸림); (2) 기준선은 반드시
`max(always_long, always_short)`, 0이 아님; (3) 시드 비교는 N≥5 진짜 다양 시드(Seed-Diversity
Ensemble Promotion Gate); (4) 저장 원장/과거 라벨을 승격 근거로 재사용하지 않음(Fresh-Forward
Rule).

---

## (a) 신규 피쳐 — `direction_head` 자체의 분류 정확도를 겨냥

### 그라운딩

FINAL12(현재 입력 12피쳐)의 실제 선택 방법을 다시 확인했다
(`docs/experiments/eth_h48qual_final12_feature_selection_20260811.md`): mRMR+knockoff을
`direction_head`(`zigzag_action`)와 `quality_head`(`h48_conservative`) 각각의 타겟에 대해
**독립적으로** 돌리긴 했지만, (i) 공통 윈도우가 **2025년 상반기(6개월)뿐**이고 전체 21개월
TRAIN이 아니며, (ii) 충돌쌍(`|r|>0.5`)을 각자 타겟 relevance로 재판정한 뒤 **하나의 12개 리스트로
병합**해서 `direction_head`가 실제로 받는 입력은 quality 타겟 기준으로 뽑힌 피쳐까지 섞여 있고,
"direction_head 단독 기준 상위 N개(캡 없음)" 리스트 자체는 만들어진 적이 없다(또는 만들어졌어도
중간 산출물로 버려짐 — 원본 mRMR/knockoff dedup 스크립트가 세션 scratchpad에만 있어 미커밋,
독립 재현 불가). 즉 과제가 지적한 대로 "FINAL12는 direction_head 전용으로 스크리닝된 적이
없다"는 게 문서로 확인된다.

새 데이터소스 축(8개 후보 — 마이크로구조/청산/Polymarket/펀딩스프레드+basis/Deribit/온체인/
hazard-competing-risk 라벨축/VPIN, `docs/experiments/eth_h48qual_quality_new_data_source_research_20260811.md`)은
전부 `quality_for_action` 또는 거래 결과와의 순위상관으로 스크리닝됐지 `direction_head`의
분류 정확도(예: holdout accuracy/F1/AUC on `zigzag_action`)로 스크리닝된 적이 없다 — 이 축도
과제가 요구한 구분과 일치한다.

### 후보 목록 (검증비용 순)

| 후보 | 메커니즘 | 닫힌 라인과 왜 다른가 | 검증 비용 | 솔직한 기대치 |
|---|---|---|---|---|
| **1. Direction-only mRMR/knockoff 재스크리닝** — `zigzag_action` 단독 타겟, 전체 21개월 TRAIN, 기존 201-피쳐 풀(h48qual 자체 145컬럼 연구패널 + REL11 회귀재스크리닝이 확장한 풀 병합) 재사용, K 캡 없이 mRMR 순위 전체 확인 | quality 타겟과 섞지 않고 순수 direction 분류 relevance(mutual_info_classif)만으로 재순위화. `scripts/rescreen_eth_h48qual_quality_regression_*_20260811.py`(커밋됨)를 최소 diff로 포크 — regression MI를 categorical MI로 교체 | FINAL12는 (i) H1-2025 6개월 창만, (ii) quality 타겟과 병합, (iii) 12개 캡 — 이 후보는 셋 다 제거. `eth_overnight` 44-kitchen-sink와도 다름: 그건 범용 기술/오더플로우 피쳐(모멘텀/OFI/변동성/Hawkes/KER)였고, 여기는 h48qual 자체 145~201컬럼 연구패널(OU half-life, m7 VAE 재구성오차, whale/toptrader 비율, `cvp_regime` 등)로 이미 다른 피쳐 계열 | **Tier 0** — 재학습 불필요, mRMR/MI 계산만. ⚠ 의존 파일 `fa_features.parquet`(127M)가 레포 밖 세션 스크래치패드(`/tmp/claude-1000/.../f6f0940b-.../scratchpad/`)에만 있음 — **2026-08-12 직접 확인, 아직 존재**하지만 git 추적 밖이라 세션 종료/재부팅 시 사라질 수 있음(데이터 리소스 문서가 이미 경고한 백업 위험과 동일). 착수한다면 먼저 레포 내 안전한 위치로 복사할 것 | 낮음~중간. Direction-only 재순위화가 FINAL12와 다른 상위 N을 낼 가능성은 있지만(선택 기준이 실제로 다르므로), N=5시드 확정 결과(ungated direction도 always-short 완패)가 이미 시사하듯 "관련성 랭킹 방법"보다 "이 피쳐 우주 자체의 정보량"이 병목일 가능성이 큼 — 그래도 FINAL12 선택이 direction 관점에서 최선이었는지 확인 안 된 마지막 빈틈이라 검증 가치는 있음 |
| **2. Fear & Greed Index 과거 백필** (alternative.me, 무료 공개 API, 2018년~ 일별 히스토리) | 시장 전반의 변동성/모멘텀/소셜/도미넌스를 합성한 외부 인덱스. TRAIN/VAL/OOS 전체(2024-01~2026-02)를 즉시 백필 가능 — 5분봉에 일별 값을 forward-fill해 `direction_head` 피쳐로 추가 | 이 서브 프로젝트가 이미 F4-C 수집기로 Fear&Greed를 수집 중이지만(`scripts/run_f4c_altdata_collector.py`, 2026-08-10부터 라이브 전방향 수집만), **의도적으로 과거 백필 없이 "3개월 뒤(2026-10) 사전등록 이벤트 스터디"로만 계획**됨(`docs/factor_execution_test_design_20260719.md` §F4-C). alternative.me는 무료 과거 백필이 실제로 존재하므로 이 계획을 바꾸지 않고도 지금 바로 TRAIN/VAL/OOS 전체에 조인 가능 — 다른 8개 신규 데이터소스 후보를 막은 "라이브 duckdb가 2026-05 이후만 커버" 벽이 이 후보엔 적용 안 됨. `eth_overnight` 44-피쳐 킷첸싱크에도 없던, 레포 최초의 외부 합성 지수 | **Tier 0** — API 호출 1회(무료, rate limit 관대), 재학습 전 반드시 `corr(price)`/`corr(시간순번)` 오염도부터 확인(표준 절차) | 낮음. (i) 일별 값이라 5분봉 288개에 동일값 반복 — bar 단위 정보량이 원천적으로 작음. (ii) F&G 자체가 realized-vol/모멘텀 성분을 포함하므로 이미 닫힌 Hawkes/변동성 피쳐군과 상관이 높을 위험(오염도 체크가 이걸 걸러낼 것). (iii) 문헌 자체가 감성데이터를 "업계 리포트 수준, 미검증"으로 평가(`entry_exit_edge_root_cause_and_literature_review_20260809.md` §2.9). 그래도 비용이 사실상 0이라 닫아두는 값어치는 있음 |
| 3. Cross-sectional 60-코인 패널을 ETH `direction_head`에 적용 (`data/panel/features`, `data/panel/tripbarrier` 재사용) | 이미 구축된 60개 USDT 무기한선물 패널(1,628만 행)을 ETH 피쳐에 시점별 z-score 조인 | BTC 대상으로는 이미 두 번 시도돼 실패(`btc_rho1_panel_direction`: rank score 거의 상수; Part 5-(B) 60코인 풀링 신경망: 3구간 중 2구간에서 ETH 전용 모델보다 못함, "크립토 알트코인은 대부분 BTC/ETH 베타에 종속돼 정보가 안 늘어남"). ETH 자체 `direction_head`(TabM+FINAL12)로는 미시도지만, 같은 메커니즘이 이미 두 번 부정된 상태 | Tier 1 — 패널 자체는 있지만 h48qual 파이프라인에 조인하는 코드는 새로 작성 필요 | 매우 낮음. 인접 증거 2건이 이미 "크립토 알트코인 풀링은 이질적 정보가 아니라 베타만 늘린다"는 동일 결론에 도달 — 재시도할 근거가 약함, 참고용으로만 기재 |
| (참고, 재제안 아님) 마이크로구조 toxicity/청산/Polymarket/펀딩스프레드+basis/Deribit/온체인 6개 | — | 전부 이미 검증됐거나(온체인 `CapMVRVCur` corr(price)=0.95~0.97로 오염 확정 기각, basis는 라벨변형간 부호 불일치로 기각) 인프라 벽으로 막힘(라이브 duckdb 2026-05~만 커버, VAL/OOS와 미중첩; Deribit/펀딩스프레드는 과거 조회 API 자체가 없음) | — | 이 문서가 다시 열지 않음 — `docs/model_contracts/odyssey_eth_h48qual_data_resources_20260812.md` "라이브 수집 duckdb"/"외부 다운로더" 두 표에 상태가 이미 정리돼 있음 |

---

## (b) 신규 방향 라벨 설계

### 그라운딩

현재 `zigzag_action` 파라미터(계약 문서 "헤드별 라벨" 표): ATR14 적응형 피벗 임계값
`max(1.0%, ATR*1.0)`, 8bar 이상 파동만 인정, 전환 지점 ±2bar 버퍼. 이 정확한 파라미터
조합의 **direction 타겟 자체**를 재스윕한 이력은 이 레포에 없다 — `eth_zigzag_swing_asymmetry_confidence_root_cause_20260811.md`/
`eth_zigzag_swing_shape_direction_asymmetry_check_20260811.md`는 **기존** 라벨의 스윙 형태를
분석했을 뿐 새 파라미터를 스윕하지 않았다.

**Trend-scanning 라벨(Lopez de Prado, max-\|t\|-value 선형적합)은 이미 ETH에 3중으로
시도됐다** — 재제안 전 반드시 확인해야 할 가장 중요한 선례:

1. **Sigma3** (`docs/model_contracts/sigma3_1h_trendscan_20260705_contract.md`) — ETH 1h,
   trend-scanning 라벨(윈도우 3~48h, threshold 2.5), HistGradientBoosting 5시드 앙상블.
   5-seed 검증에서 1/27 게이트만 통과했고, 사전등록 one-shot OOS(2026-03~06)에서
   **cost3 -3.88%로 실패**(cost1은 +7.34%로 근접했지만 3배 비용 스트레스는 못 넘김). TP
   히트율이 9%→40%로 크게 개선된 건 이 프로젝트 역사상 가장 유망했던 신호 중 하나였지만
   최종 게이트는 실패로 기록됨.
2. **Sigma6**(Sigma3 신호 + Regime3 `not_chop` 필터 + 트레일링) — 처음엔 OOS +45.9%/+16.6%로
   이 프로젝트에서 가장 강력한 fresh-window 결과처럼 보였으나, `FROZEN_BASELINE_REGISTRY.json`의
   주석에 따르면 **`tape_ensemble.parquet` 데이터가 조용히 재생성되면서 OOS가 -22.0%/-9.7%로
   반전**됐다 — 별도의 `compute_metrics()` 러너로 연결되지 않아 후속 확인이 안 된 상태로
   방치돼 있다.
3. **Sigma9**(BTC로 동일 파이프라인 이식) — BTC standalone 신호가 약함, combined book이
   ETH-Sigma6 단독보다 개선 안 됨(`research_negative_result_not_adopted`).
4. **BTC v1 trend-scan t2 full retrain**(`docs/model_reports/btc_v1_trendscan_t2_full_retrain_20260715.md`) —
   다른 자산(BTC)이지만 같은 라벨 메커니즘, TabM 3-head 아키텍처로 재학습, VAL +6.98%였으나
   확장 OOS -4.32%, risk sidecar OOS -19.05% — `promotion_pass=false`.

네 데이터 포인트 모두 방향이 다르지 않다(약함/실패/반전-후-미확인). 단, 넷 다 **1h 시간봉 +
GBM(HistGradientBoosting)** 조합이었지 h48qual의 **5분봉 + TabM + FINAL12** 조합으로는
한 번도 시도된 적이 없다 — 이게 유일한 differentiator다. `train_sigma3_1h_purged_walkforward_20260801.py`/
`_secondary_20260801.py`(2026-08-01 커밋, sigma3보다 최신)는 존재를 확인했으나 대응하는 결과
문서를 찾지 못했다 — 상태 불명, 착수 전 먼저 이 두 스크립트의 실행 여부/결과부터 확인할 것.

**TP-first/삼중장벽을 direction 타겟으로 쓰는 안**은 이미 닫힌 `btc_tpfirst_three_way_label`
(0/24, "P(TP)-P(SL)가 실제 품질과 무관")과 개념적으로 거의 동형이다 — 그리고 이 서브 프로젝트가
스스로 정립한 "필터형 vs 재추정형" 원칙(`eth_h48qual_quality_head_replacement_research_20260812.md`
"구조적 원칙" 절)이 이 재프레이밍을 "재추정형"(결과 확률 직접 예측)으로 분류해 경고 대상으로
못박아뒀다.

### 후보 목록 (검증비용 순)

| 후보 | 메커니즘 | 닫힌 라인과 왜 다른가 | 검증 비용 | 솔직한 기대치 |
|---|---|---|---|---|
| **1. `zigzag_action` 파라미터 재스윕** — ATR 배수(현재 1.0×)와 `min_wave_bars`(현재 8)를 각각 그리드로 흔들어 오라클(hindsight) 방향 스킬 천장부터 확인, 천장이 뚜렷할 때만 재학습 | h384/48bar 호라이즌 스윕은 이미 했지만(`quality_head`의 barrier horizon), **direction 타겟인 zigzag pivot 자체의 임계값**을 흔든 적은 없음. `h48qual`의 quality-horizon-sweep 방법론(`build_omega1_2_triple_barrier_labels_20260619.py`류)을 zigzag 빌더에 그대로 적용 가능 | Tier 0(오라클 천장 게이트만) → Tier 1(재학습, 유망한 설정에만) | 낮음~중간. 개념적으로 이미 닫힌 `btc_barrier_horizon_calibration`(BTC 배리어/호라이즌 스윕 0/148)과 인접 — 이 레포에서 임계값 스윕류의 성공률이 낮다는 사전확률을 적용해야 함. 다만 그건 quality 배리어였고 이건 direction 자체라 완전히 같은 실험은 아님 |
| 2. `min_wave_bars`만 단독 스윕(ATR배수 고정) | 후보 1의 저비용 부분집합 — 파라미터 1개만 흔들어 해석 쉬움 | 위와 동일 | Tier 0(오라클 천장만) | 낮음 — 후보 1보다도 좁은 탐색이라 기대치는 더 낮지만 비용도 더 낮음. 후보 1 전에 스팟체크로 먼저 돌릴 가치 |
| 3. Trend-scanning을 h48qual의 5분봉+TabM+FINAL12 조합에 재적용 | 시간봉/모델군만 바꿔 동일 라벨 재시도 | 유일한 차이가 timeframe+모델군뿐 — 이미 3중으로 약함/실패/반전 데이터 포인트가 있는 라벨 메커니즘. 재제안하되 **닫힌 것으로 취급**을 권고 | Tier 1(전체 재학습) | **매우 낮음** — 재제안 안 함이 기본값. 착수한다면 먼저 `train_sigma3_1h_purged_walkforward_*_20260801.py` 두 스크립트의 미확인 결과부터 복구해서 정말 새 근거가 없는지 확인 |
| 4. Triple-barrier/TP-first를 direction 타겟으로 재정의 | "다음 N봉 안에 LONG-TP/SHORT-TP 중 뭐가 먼저 닿는가"를 1차 방향 예측 자체로 사용 | `btc_tpfirst_three_way_label`(0/24)과 개념적으로 거의 동일한 재추정형 프레이밍 — 자산만 다름 | Tier 1 | 낮음 — 이 프로젝트 자체 이론(필터형/재추정형)이 실패를 예측하는 후보. 참고용으로만 기재, 우선순위 최하위 |
| 5. Hazard/competing-risks 방향 재구성 (pivot 형성을 continuous-time competing hazard로 모델링, Fine & Gray 1999) | zigzag pivot 스냅샷 분류 대신, "다음에 상승/하락 파동이 시작될 hazard rate"를 매 bar 추정 | 레포에 hazard 모델링 자체는 있지만(`research_eth_omega461_competing_risk_rescue_20260724.py` 등) 전부 **exit-side**(이미 연 포지션의 청산/구제) 또는 레짐전환용이지 **direction 진입 자체**를 겨냥한 적은 없음. `eth_h48qual_quality_new_data_source_research_20260811.md` 후보 7과 다름 — 그건 quality_head의 TP/SL barrier race를 hazard로 재구성하는 안(현재 quality 투자 보류로 근거 약화), 이건 zigzag pivot 형성 자체를 hazard로 재구성하는 안 | Tier 2(새 라벨 파이프라인+새 모델링 패러다임) | 낮음 — 더 풍부한 loss function이 이 피쳐 우주에 없는 정보를 만들어내진 않는다는 게 필터형/재추정형 원칙의 핵심 주장. 사변적 후보로만 기재 |

---

## (c) 다른 구간 재검증

### 그라운딩 — 실제로 뭐가 가능한지 직접 확인

계약 문서의 표준 구간(TRAIN 2024-01~2025-09, VAL 2025-10~12, OOS 2026-01~02)을 확인한 뒤,
**이 문서를 위해 canonical raw kline 파일**(`binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv`,
2023-12-31~2026-08-04 커버)에서 분기별 종가를 직접 뽑아 TRAIN 내부의 레짐 다양성을 확인했다:

| 구간 | 시작 종가 | 종료 종가 | 수익률 |
|---|---:|---:|---:|
| 2024 Q1 (01→04) | 2289.92 | 3647.41 | **+59.3%** |
| 2024 Q2 (04→07) | 3647.41 | 3435.34 | -5.8% |
| 2024 Q3 (07→10) | 3435.34 | 2602.19 | -24.3% |
| 2024 Q4 (10→익년01) | 2602.19 | 3339.41 | +28.3% |
| 2025 Q1 (01→04) | 3339.41 | 1823.76 | **-45.4%** |
| 2025 Q2 (04→07) | 1823.76 | 2486.15 | +36.3% |
| **2025 Q3 (07→10, TRAIN 마지막 분기)** | 2486.15 | 4149.09 | **+66.9%** |
| VAL (2025-10→2026-01) | 4149.09 | 2973.66 | -28.3% (계약 문서 "VAL -28%"와 정확히 일치) |
| OOS 근사(2026-01→03) | 2973.66 | 1966.39 | -33.9% (계약 문서 "OOS -36%"과 근접) |
| Post-OOS(2026-03→08, 미검증) | 1966.39 | 1864.28 | -5.2% (완만 — VAL/OOS의 급락과 성격이 다름) |

**핵심 발견**: TRAIN(2024-01~2025-09) 전체 누적은 오히려 **+81.3%**(2289.92→4149.09) —
계약 문서 h384 스윕 절이 언급한 "학습구간(-51%)"과 불일치한다(그 수치가 가리키는 정확한
기간·시리즈를 추적하지 못했다 — 다른 스윕의 학습 split일 수 있음, 확인 필요). 위 표는 이
문서를 위해 canonical 파일에서 직접 재계산했고 방법론을 명시했으므로 독립 재현 가능하다. 이게
맞다면 TRAIN은 순수 하락구간이 아니라 **강한 상승(2024 Q1 +59%)과 강한 하락(2025 Q1 -45%)이
섞이고 마지막 분기(2025 Q3)가 오히려 가장 강한 상승(+67%)** — 즉 VAL/OOS 직전 분기가 정반대
방향이었다는 뜻이다. 이건 "모델이 하락장에서 학습해서 하락에 편향됐다"는 단순 스토리와 안 맞고,
오히려 스윙 비대칭(숏이 롱보다 경로가 깔끔함, `eth_zigzag_swing_asymmetry_confidence_root_cause_20260811.md`)
같은 추세-방향과 무관한 원인이 여전히 유력하다는 뜻이라 — 후보 1(아래)의 가치를 높인다.

**데이터 가용성 벽 재확인**: `zigzag_action_labels_2026.csv`(direction 라벨의 1차 소스)는
**2026-02-28 16:00:00에서 끊긴다** — OOS 종료와 거의 동일 지점. 반면 raw kline은
2026-08-04까지 있다 — **가격 데이터는 있는데 라벨/피쳐가 없는** 5개월 구간(2026-03~07)이
존재한다는 뜻. 그런데 `tmp/causal_regen_20260516/extended_oos_20260702/` 디렉터리를 뒤져보니
**이미 다른(무관한) 프로젝트가 이 정확한 구간(2026-03-01~07-02)의 피쳐를 만들어뒀다** —
`training_features_2026_0301_0702_rebuilt_contract142_m7_live.csv` 등, `omega4_6_2`/`omega5`
계열 모델(h48qual과 다른 라인)의 fresh-forward 검증용으로 2026-07-02에 구축됨
(`docs/audits/omega5_short_momentum_v2_bar_forward_val_oos_20260702.md`,
`docs/audits/omega462_hf_policy_bar_forward_val_oos_20260702.md`). h48qual과 스키마가
호환되는지는 미확인이지만("contract142"라는 이름 자체가 h48qual의 102-피쳐 계약과 다른
피쳐-계약 버전임을 시사) — 이게 있다는 사실 자체가 "새 구간은 데이터부터 새로 모아야 한다"는
당초 가정을 뒤집는다. raw kline과 피쳐가 이미 있으므로, 이 구간에 대해 **재학습 없이 얼려진
h48qual 번들로 순수 추론만 하면**(TRAIN fullwindow 재생성에 이미 쓴 것과 같은 기법 —
`scripts/regenerate_eth_h48qual_fullwindow_train_predictions_20260812.py`) 진짜 미검증 구간의
예측을 뽑을 수 있을 가능성이 있다.

### 후보 목록 (검증비용 순)

| 후보 | 메커니즘 | 왜 이게 유효한가 / 닫힌 라인과 다른가 | 검증 비용 | 솔직한 기대치 |
|---|---|---|---|---|
| **1. TRAIN 내부 상승구간 홀드아웃** — 2025 Q2~Q3(2025-04~2025-10, +36%→+67% 상승 2연속 분기)를 이미 재구성된 전체구간 TRAIN 예측(`..._fullwindow_predictions_recheck_20260812/`, 183,936행, `scripts/regenerate_eth_h48qual_fullwindow_train_predictions_20260812.py` 산출물)에서 슬라이싱, always-long/always-short 대조 | 근거: 위 표. TRAIN이 실제로 레짐이 섞여 있다는 게 이 문서에서 처음 직접 정량화됨 — 계약 문서는 "다양한 레짐이 섞인 21개월"이라 정성적으로만 언급했었음(confidence 격차 진단 맥락) | **Tier 0** — 재학습·신규 데이터 불필요, 기존 CSV 재슬라이스 + PnL 계산만. **⚠ 캐비어트**: 모델이 이 정확한 데이터로 학습했으므로 인샘플 — "스킬이 있다"는 결론을 내릴 수 없고, 기껏해야 "하락구간 특정적 실패인지"에 대한 **약한 시사점**만 제공. 이 레포의 Fresh-Forward Rule상 이 결과 단독으로는 승격/모델선택 근거가 될 수 없음, 순수 스코프 판단용 | 낮음~중간이지만 **정보가치는 최고** — 비용이 사실상 0이라 다른 모든 (c) 후보 착수 전에 먼저 돌려서 "하락장 특정적" 가설이 최소한의 지지를 받는지부터 확인하는 게 합리적 |
| **2. Post-OOS 신규구간 확장(2026-03~07)** — 이미 구축된 `training_features_2026_0301_0702_rebuilt_contract142*.csv`가 h48qual 스키마(FINAL12 또는 102-base)와 호환되는지 먼저 컬럼 대조, 호환되면 얼려진 h48qual 번들(`true_3head_tabm_bundle.pt`)로 순수 추론(재학습 없음) | 진짜 out-of-sample — 모델이 한 번도 본 적 없는 기간이자 가격 성격도 다름(완만한 -5.2%, VAL/OOS의 급락과 다른 레짐). 라이브 duckdb 벽(2026-05~만 커버)이 이 후보엔 안 걸림 — raw kline·피쳐 둘 다 이미 3월부터 있음 | **Tier 1** — 스키마 호환성 확인이 선행 조건(공짜, 컬럼명 대조). 호환되면 순수 추론만이라 저렴(Tier 0에 가까움); 비호환이면 FINAL12의 기존 알려진 프로덕션 브릿지 갭(`vwap_dist_24`/`funding_roc_48`이 프로덕션 패널에 없음, 미해결 이슈 5)과 같은 문제가 재발할 수 있어 비용이 뜀 | 중간 — 이 문서의 가장 결정적인 (c) 후보. 다만 zigzag 라벨(정답)이 이 구간에 없으므로 "PnL만" 계산 가능하고 "분류 정확도" 비교는 못 함 — 그래도 always-short 대조 PnL은 계산 가능해 후보 9(구조적 한계 인정)의 핵심 질문에 직접 답함 |
| 3. Pre-2024 과거 구간(2022~2023 약세장/회복 사이클) | 확인된 raw kline이 2023-12-31부터라 이 구간 자체가 이 파일엔 없음 — 더 오래된 kline을 별도로 구해야 하고, 102-피쳐 연구패널(OU half-life, m7 VAE 등)도 이 구간에 대해 계산된 적 없음 | 진짜 다른 시장 사이클(다른 변동성 체제)이라는 점에서 후보 2보다 더 강한 differentiator | **Tier 2** — 신규 데이터 확보 + 전체 피쳐 파이프라인 재실행 필요, 이미 확인된 인프라 벽(과거 백필 가능한 원시 kline은 있어도 파생 피쳐 상당수가 이 구간에 없음)과 유사한 급 | 판정 불가 — 근시일 착수 비권장, 후보 1·2가 먼저 소진된 뒤 장기 옵션으로만 유지 |

---

## 전체 통합 우선순위

| 순위 | 후보 | 갈래 | 비용 | 이유 |
|---:|---|---|---|---|
| **1** | TRAIN 내부 상승구간 홀드아웃 | (c)-1 | Tier 0 | 사실상 무료, 최우선 실행. 결과가 "상승구간에서도 못 이김"이면 후보 2(더 비쌈)의 시급성이 낮아지고, "상승구간에선 다르다"는 힌트가 나오면 후보 2에 우선순위를 더 싣는 **의사결정 게이트** 역할 |
| **2** | Post-OOS 신규구간 확장(2026-03~07) | (c)-2 | Tier 1(호환성 확인은 Tier 0) | 진짜 OOS. 이미 구축된 남의 프로젝트 산출물을 재활용해 원래 예상보다 훨씬 싸게 가능해짐 — 후보 1 결과와 무관하게도 착수 가치가 있지만, 순서상 후보 1 다음이 합리적 |
| **3** | Direction-only mRMR/knockoff 재스크리닝 | (a)-1 | Tier 0(의존 파일 우선 백업 필요) | (c)와 독립적인 질문(피쳐 vs 구간)이라 병행 가능. `fa_features.parquet`가 레포 밖에만 있어 소멸 위험 — 착수 여부와 무관하게 **지금 바로 레포 내 안전한 위치로 백업**해두는 게 이 문서의 가장 시급한 부수 권고 |
| 4 | `zigzag_action` `min_wave_bars` 단독 스윕(오라클 천장만) | (b)-2 | Tier 0 | 가장 싼 (b) 후보, 오라클 천장이 안 보이면 후보 1(전체 재스윕)로 안 넘어가는 컷오프 역할 |
| 5 | Fear & Greed Index 백필 | (a)-2 | Tier 0 | 공짜지만 기대치 낮음, 오염도 체크부터 |
| 6 | `zigzag_action` ATR배수 재스윕(전체) | (b)-1 | Tier 0→1 | 후보 4가 신호를 보일 때만 |
| 7 이하 | 60코인 패널, trend-scanning 재적용, TP-first 재프레이밍, hazard 방향 재구성, pre-2024 구간 | (a)-3, (b)-3/4/5, (c)-3 | Tier 1~2 | 전부 인접 증거가 이미 부정적이거나 비용이 큼 — 위 6개가 전부 소진된 뒤에만 고려 |

### 지금 당장 뭘 먼저 돌릴 것인가

**TRAIN 내부 상승구간 홀드아웃(위 표 1순위)을 가장 먼저 돌린다.** 이유: (1) 비용이 사실상
0(기존 파일 재슬라이스), (2) 이 문서가 방금 직접 계산으로 확인한 사실(TRAIN이 순수 하락이
아니라 2024 Q1 +59%/2025 Q3 +67% 같은 강한 상승 분기를 포함) 덕분에 처음으로 실행 가능해짐 —
이전엔 "TRAIN도 하락"이라는(부정확했을 가능성이 있는) 가정 때문에 이 갈래 자체가 막혀
있었다고 볼 수 있음, (3) 결과가 이후 우선순위(특히 후보 2의 긴급도)를 결정하는 **게이트**
역할을 함 — 상승구간에서도 always-long/short 대비 스킬이 없다면 "이 자산/타임프레임/피쳐
조합엔 방향 스킬 자체가 없다"는 가설이 더 강해지고, 반대로 상승구간에서 다른 패턴이 보이면
더 비싼 후보 2(진짜 OOS 확장)에 자원을 배분할 근거가 생긴다. 인샘플이라는 근본적 한계는
분명히 남지만, 다음 스텝을 결정하는 정찰 목적으로는 이 비용 대비 정보가치가 이 문서의 모든
후보 중 가장 높다.

## 추가: (c)-1 실행 결과 (2026-08-12, 이 문서 작성 직후)

위 1순위 후보(TRAIN 내부 상승구간 홀드아웃)를 바로 실행했다.
`scripts/diagnose_eth_h48qual_train_uptrend_holdout_ungated_vs_always_long_short_20260812.py` —
2025-04-01~2025-10-01(2025 Q2+Q3, 복리로 +127.2%) 구간을 raw kline에서 직접 슬라이싱하고,
같은 구간의 fullwindow TRAIN 재생성 예측(재학습 없음)에 게이트 우회 기법을 적용.

| 구성 | pnl% | trades | wr |
|---|---:|---:|---:|
| gated(라이브 방식) | -11.01 | 50 | 32.0% |
| ungated(direction_head 원본) | -7.53 | 72 | 33.3% |
| always_short | -29.11 | 73 | 26.0% |
| **always_long** | **+36.51** | 72 | 47.2% |

`max(always_long, always_short)` = +36.51%. **ungated이 이걸 이기는가? 아니오** — 44%p 격차로
완패. 참고로 게이트가 살아있는 라이브 방식은 이 구간에서 아예 손실(-11.01%)이다 — 가격이
127% 오르는 동안 실전 방식 그대로면 돈을 잃었다는 뜻.

**해석**: 이 결과는 스카우팅 문서 본문이 명시한 "의사결정 게이트" 역할을 했다 — **상승구간에서도
지지 않을 거라는 기대가 깨졌다.** VAL/OOS의 실패가 "하락장이라 always-short이 유리해서"라는
단순 스토리가 아니라, 방향에 무관하게(상승이든 하락이든) 강한 추세 구간에서 naive 방향성
베팅조차 못 이긴다는 뜻 — 더 구조적인 문제로 보인다. 이건 후보 2(post-OOS 2026-03~07 확장,
Tier 1, 진짜 OOS라 더 비쌈)의 **긴급도를 낮춘다** — 다운/업 두 강한 추세 구간에서 이미 같은
결론에 도달했으므로, 세 번째(다른 방향의) 구간을 비싸게 확보해도 같은 결론이 나올 가능성이
높아졌다. 후보 2는 여전히 유효하지만("진짜 OOS"라는 값어치는 남아있음) 우선순위가 상대적으로
내려간다.

**한계, 반드시 유지**: 인샘플(모델이 이 정확한 데이터로 학습함) + 단일 인스턴스(N=5 시드 아님) —
"direction_head에 스킬이 없다"를 증명하지 않는다. 다만 "VAL/OOS 실패가 하락장 특정적"이라는
가설에 대한 반증으로는 충분히 유효하다(그 가설은 애초에 이 방향 — 하락장이 아니어도 진다 —
으로 반증되기 쉬운 형태였음).

## 결론

세 갈래 모두 "이미 닫힌 라인을 재발견"할 위험이 실재했다 — 특히 (b)의 trend-scanning은
문자 그대로 이미 ETH에서 3중으로 시도돼 약함/실패/반전-후-미확인 상태였고, (a)의 새 데이터소스
방향은 8개 후보가 이미 인프라 벽에 막혀 있었다. 이 문서가 실제로 여는 새 문은 (1) FINAL12가
direction 전용으로 스크리닝된 적이 없다는 구체적 확인(a-1), (2) F4-C가 의도적으로 안 쓴
Fear&Greed 무료 백필(a-2), (3) TRAIN이 실제로는 순수 하락구간이 아니라는 직접 재계산(c-1의
근거), (4) 남의 프로젝트가 이미 만들어둔 2026-03~07 피쳐 자산을 재활용하면 "새 구간" 검증이
당초 가정보다 훨씬 싸다는 발견(c-2)이다. 이 넷을 먼저 소진한 뒤에야 나머지(60코인 패널, zigzag
파라미터 재스윕, hazard 재구성 등 비용이 크거나 인접 증거가 이미 부정적인 후보)로 넘어가는
순서를 권한다.
