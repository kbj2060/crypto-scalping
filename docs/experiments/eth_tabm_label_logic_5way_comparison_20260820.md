# ETH TabM 라벨로직 5-way 비교 (154피쳐 + 단일모델 + N=3 시드)

## 배경

`eth_tabm_label_logic_retest_initiative_20260819` 서브프로젝트에서 라벨로직 후보
①DC/②CUSUM/③분포적회귀를 개별적으로 재검증해왔고, 별도로 154개 엔지니어링 피쳐셋(정리
112 + RIT조합 30 + financial-ML 12)을 구축했다. 사용자가 "이 154개 피쳐로 라벨 로직만
바꿔서 5개 테스트, 시드는 3개"를 요청 — zigzag(zig075 production)/h48qual/dc/cusum +
"이번에 테스트한 2개"(=②CUSUM, ③분포적회귀, 이 서브프로젝트 안에서 이미 다뤄온 두 후보)로
5개를 확정했다.

**N=3는 CLAUDE.md의 Seed-Diversity Ensemble Promotion Gate(N≥5)에 못 미치는 예비
스크리닝이다.** 이 문서의 어떤 결과도 승격/확정 근거가 아니며, "다음에 N≥5로 볼 가치가
있는가"를 가리는 것이 유일한 목적이다.

## 방법론

### 공통 스택 (5개 전부 동일)
- 피쳐: 154개 엔지니어링 셋(`eth_dc_engineered_features_canonicaldata_20260820.py`,
  VIF-clean 112 + RIT 조합 30 + financial-ML 12)
- 아키텍처: TabM **단일모델**(레짐당 별도 expert 없음, `_route_probs`→전체1.0 monkeypatch,
  bull만 실제학습 후 bear/chop에 복사 — `train_eth_ilias1_zig075_trial12_unified_single_model_20260819.py`
  선례와 동일 기법). 레짐피쳐(`regime3_current_sensitive_wide24_*`)는 그대로 입력에 남음.
- 데이터: canonical TRAIN=`data/splits/year_oos/training_features_2025.csv`
  (train<2025-10-01, validation>=2025-10-01), OOS=`training_features_2026_rebuilt.csv`
  (REGIME3_CURRENT_2026 커버리지로 사전필터링, 2026-01-01~06-30)
- 시드: 공유 3개 [133725056, 176495706, 796203462] (`random.SystemRandom().sample`,
  5개 라벨 전부 동일 시드값 사용 — "라벨로직만 바꾼다"는 통제 실험 취지)
- epochs=2 (이 서브프로젝트가 이미 epoch2→30 재학습으로 "조기수렴, 차이없음"을 classification
  head 대상으로 검증해둔 스크리닝 관례 그대로 적용 — regression head는 이 검증이 없다는 점은
  분포적회귀 섹션에 별도 명시)

### zigzag / h48qual / dc / cusum (classification, `eth_tabm_label_logic_5way_seed_variant_20260820.py`)
이산 zigzag_action{CASH,LONG,SHORT} 스키마를 공유하므로 `--direction-label-dir`/
`--quality-mode`/`--quality-label-dir`만 바꿔 동일 3-head classification 파이프라인에 꽂았다.

| 라벨 | direction-label-dir | quality-mode | 비고 |
|---|---|---|---|
| zigzag | `zigzag_action_labels_20260531` | same_as_direction | zig075 production과 동일 direction 소스 |
| h48qual | (zigzag와 동일) | quality_label_action + `sltp_h48_conservative_padded_to_zigzag_timestamps` | direction은 zigzag와 동일, quality gate만 h48 conservative |
| dc | `eth_directional_change_triple_barrier_labels_dense_cashfill_20260819` | same_as_direction | 이 서브프로젝트 후보① |
| cusum | `eth_cusum_triple_barrier_labels_dense_cashfill_20260820` | same_as_direction | 이 서브프로젝트 후보② |

`--exit-label-mode independent_entry_hold_offsets`로 4개 전부 통일했다(zig075/h48qual
라이브가 실제 쓰는 `entry_label_terminal_giveback`은 dc/cusum의 고립된 단일-bar dense-cashfill
이벤트에서 세그먼트 스캐너가 거의 전부 skip해 RuntimeError로 죽기 때문 — 이 제약은 DC 학습
착수 시 이미 확인됨). **이 선택 때문에 zigzag/h48qual 숫자는 실제 라이브 프로모션 설정과
다르다 — 이 문서의 결과를 라이브 프로모션 근거로 쓰지 않는다.**

### 분포적회귀 (regression, `eth_distributional_regression_tabm_dist_head_20260820.py`)
이산 direction/quality 라벨이 아니라 fixed-horizon(h48=48bar=4시간) 연속 forward log-return이
라벨이라(TP/SL/보유기간 개념 자체가 없음) classification head에 그대로 못 꽂힌다.
AskUserQuestion으로 "sign-proxy 근사" vs "진짜 regression head 구현" vs "4개만 진행"을
물었고, **사용자가 "진짜 regression head 구현"을 선택**했다.

- 트렁크: `ThreeHeadTabM.encode()`(k=8 BatchEnsemble, hidden=192, layers=3, CFG 기본
  하이퍼파라미터 그대로)를 바이트단위 복제, direction/quality/exit head 대신
  `dist_head`(2, mu+log_sigma) 하나만 부착 — "라벨로직만 바꾼다"는 취지를 아키텍처 레벨에서
  최대한 지킴
- 손실: Gaussian NLL(`0.5*log(2*pi)+log_sigma+0.5*((y-mu)/sigma)^2`), k-앙상블 성분은
  균등혼합 Gaussian의 정확한 total-variance 분해로 단일 (mu,sigma) 축약
- 트레이드 판정: z=mu/sigma를 TRAIN 분포 |z| 백분위수(50/60/70/80/90th)로 이산화, VAL PnL
  1위 임계값의 OOS를 확인(나머지 4개의 quality_threshold 선택절차와 동일 causal-safe 규칙)
- PnL은 **fixed-horizon 홀드**(barrier/TP-SL 없음, 왕복10bp 비용) — 나머지 4개의 barrier
  기반 TP/SL PnL과 절대수치 직접비교 불가. `cond_dir_acc`(부호일치율)만 개념적으로 비교 가능.

## 결과

### classification 4-way — VAL PnL 1위 임계값의 OOS (TRAIN에서만 후보를 고르는 이 세션
표준 causal-safe 절차)

| 라벨 | seed1 VAL→OOS | seed2 VAL→OOS | seed3 VAL→OOS | OOS 부호 |
|---|---|---|---|---|
| zigzag | 11.0→13.8 | -1.5→24.1 | -12.5→-5.1 | **2승1패** |
| h48qual | 21.7→17.1 | 13.7→6.1 | 11.3→1.1 | **3승0패** |
| dc | 19.4→50.8 | 23.9→-8.7 | 14.2→-11.6 | **1승2패** |
| cusum | 16.4→27.2 | 23.7→10.7 | 22.3→6.1 | **3승0패** |

(단위: PnL%, 초기자본 대비. 원본 수치는 `tmp/.../scratchpad/5way_comparison_summary.json`)

- h48qual/cusum: VAL 대비 OOS가 매 시드 축소되는 공통 패턴(h48qual 17.1→6.1→1.1, cusum
  27.2→10.7→6.1)이면서도 부호는 3/3 유지. h48qual은 시드당 거래수가 12~23건으로 적어
  cusum(56~65건)보다 표본 노이즈에 더 취약.
- zigzag/dc: OOS 부호 불일치 — 이 서브프로젝트가 반복 확인해온 "VAL은 맞고 OOS는 시드별
  반전"(h48qual/zig075/Sigma3-1h/DC N=5/CUSUM hp·aswa·bag 전부 동일 패턴) 재현.

### 분포적회귀 3-way

| seed | z임계값 | NLL | VAL pnl(bps) | OOS pnl(bps) | OOS cond_acc | OOS L/S |
|---|---|---|---|---|---|---|
| 133725056 | 1.017 | -1.725 | -37,474 | -34,200 | 0.536 | 2231/0 |
| 176495706 | 1.357 | -2.124 | -18,314 | -21,274 | 0.536 | 2/7930 |
| 796203462 | 0.729 | -1.480 | -58,089 | -19,334 | 0.507 | 2388/4 |

**OOS 3전0승, 전부 대폭 음수.** cond_dir_acc 0.507~0.545로 classification 4개의 조건부
방향정확도(이 서브프로젝트에서 반복 확인된 48~51% chance 대역)와 동급.

⚠️ **주목할 진단**: 3시드 전부 거의 한쪽 방향으로만 쏠렸다(seed1: LONG 100%, seed2: SHORT
99.97%, seed3: LONG 99.8%) — 그런데 그 쏠리는 **방향 자체가 시드마다 다르다**. 이는 모델이
조건부 신호가 아니라 초기화에 따라 우연히 결정되는 근사-상수 평균(mean-shift)을 학습했다는
뜻이며, classification 4개(대체로 균형잡힌 LONG/SHORT 혼합)보다도 구조를 덜 포착했다는
신호다. z-score 계산(sigma가 작을수록 작은 mu 차이도 크게 증폭)과 결합해 이런 붕괴가 나타난
것으로 보이며, 코드 로직(z 부호/임계값 분기) 자체는 직접 검증해 버그가 아님을 확인했다.

## 해석

1. **5개 라벨 전부 N=3에서 승격/확정 근거 없음** — CLAUDE.md N≥5 게이트 미달.
2. **h48qual/cusum이 이 5-way 비교에서 유일하게 3/3 OOS 양수** — zigzag/dc/distreg보다
   상대적으로 눈에 띄지만, N=3에서 3/3 우연 확률은 이항분포로 12.5%(양쪽 다 독립이라면
   12.5%×12.5%≈1.6%로 낮아지지만, 두 라벨의 direction 소스(h48qual은 zigzag와 동일
   direction!)가 사실상 강하게 상관되어 있어 "독립 확인 2건"으로 셀 수 없음 — h48qual의
   3/3은 zigzag의 direction 신호 위에 h48 quality 게이트를 얹은 것이라 cusum의 3/3과는
   성격이 다른 확인이다). 다음 단계로서만 볼 가치가 있다(N≥5, 완전 무교집합 신규시드).
3. **분포적회귀는 이 5개 중 가장 약함** — 진짜 regression head까지 구현했음에도 OOS 0/3,
   방향 자체가 시드마다 뒤집히는 불안정한 붕괴 패턴. 이 라벨 축은 추가 투자(전체 분포모수
   활용, Student-t 등 유연한 분포족, 다른 horizon)보다 종료 쪽에 무게가 실린다.
4. 154피쳐 엔지니어링 셋 자체가 5개 라벨 전부에서 여전히 "라벨을 바꿔도 근본 병목이 그대로
   드러난다"는 이 서브프로젝트의 누적 결론(40개 이상 라벨/기법/피쳐셋 조합 전부 chance 수렴)과
   다시 한 번 일치한다.

## 다음 단계 (사용자 결정 대기)

- h48qual/cusum을 N≥5(완전 무교집합 신규시드)로 재확인할지
- 분포적회귀 축을 여기서 종료할지
- 서브프로젝트 자체를 여기서 잠정종료할지(GEX 대기만 유일 생존 로드맵 후보로 남음)

## 한계

- epochs=2는 classification head에서만 검증된 스크리닝 관례이고 regression head는
  under-training 여부를 별도 검증하지 않았다 — distreg의 0/3 결과가 구조적 무신호 때문인지
  단순 미수렴 때문인지 이 문서만으로는 완전히 가르지 못한다.
- exit-label-mode를 4개 classification 라벨 전부 independent_entry_hold_offsets로
  통일해 zig075/h48qual 라이브 프로모션 설정과 다르다.
- 분포적회귀 PnL은 fixed-horizon 홀드로, 나머지 4개의 barrier 기반 TP/SL PnL과 방법론이
  달라 절대수치 비교 불가.

## 후속: zigzag/h48qual/cusum 구조 분석 (2026-08-21)

5-way 비교에서 상대적으로 나은 3개(zigzag 2/3, h48qual 3/3, cusum 3/3)를 더 깊게 보기 위해
차트(`chart_zigzag_h48qual_cusum_label_comparison_20260821.py`, TRAIN 2025-01-06~01-20)를
먼저 그렸고, 이어서 통계적 구조 분석(`eth_zigzag_h48qual_cusum_structural_similarity_20260821.py`,
공통커버리지 2024-01-01~2026-02-28, 227,186행)을 실행했다. h48qual은 5-way 학습에 실제 쓰인
h48_conservative quality label(`sltp_h48_conservative_padded_to_zigzag_timestamps`) 자체의
독립 tb_action을 그대로 사용(zigzag direction과 결합된 최종 게이트 결과가 아님).

### 방향전환 빈도 (사용자 관찰: "cusum이 롱/숏을 너무 자주 바꾸지 않냐" → 확인됨)

| 라벨 | flip_rate | 연속유지 median(bar) | 연속유지 mean(bar) |
|---|---|---|---|
| zigzag | 1.93% | 33bar (2.75h) | 53.0bar |
| h48qual | 5.80% | 11bar (55min) | 23.3bar |
| cusum | **42.17%** | **2bar (10min)** | 4.9bar |

cusum은 zigzag보다 약 22배, h48qual보다 약 7배 더 자주 방향을 바꾼다. 이는 zigzag가 확정
스윙(고점-저점) 전체를 하나의 방향으로 유지하도록 설계된 반면 cusum은 이벤트 단위(CUSUM 드리프트
임계값 돌파마다 독립 재평가)로 짧게짧게 재판정하기 때문 — 설계상 당연한 차이.

### 같은-bar 매칭 vs 순열귀무 — 예상외로 "우연 수준"

circular-shift 순열귀무(200회, 각 라벨 고유의 run-length 구조는 보존한 채 정렬만 무작위화)
대비 관측된 같은-bar 동시활성 빈도가 **3개 쌍 전부 empirical_p=0.99~1.00** — 즉 순열귀무보다
크지 않다(오히려 미세하게 낮음). 같은-bar 수준에서는 세 라벨이 "특별히 시간정렬돼있다"고
말할 근거가 없다. 단, 이건 cusum의 짧은 유지길이(2bar) 때문에 정확히-같은-bar 기준 자체가
지나치게 엄격한 것으로 보인다 — 아래 허용오차 확장 결과 참고.

### 허용오차(±5분/±15분) 확장 — 매칭률은 오르지만 cusum의 방향일치율은 낮은 채로 고정

| 쌍 | 같은-bar 매칭 | ±5min 매칭 | ±15min 매칭 | ±15min 방향일치 |
|---|---|---|---|---|
| zigzag-h48qual | 61.7%(기대치61.99%와 동일) | 75.9% | 86.3% | **81.7%** |
| zigzag-cusum | 35.0%(기대35.5%) | 77.7% | 97.8% | **66.1%** |
| h48qual-cusum | 36.0%(기대36.5%) | 78.6% | 97.9% | **79.1%** |

±15분까지 넓히면 cusum은 거의 항상(97.8~97.9%) 근처에 뭔가 활성 결정이 있다 -- 짧은 이벤트가
촘촘히 흩뿌려져 있기 때문. **그런데 그 근처 결정의 방향은 zigzag와 66.1%만 일치**(h48qual-zigzag
81.7%보다 뚜렷이 낮음, cusum-h48qual도 79.1%로 zigzag-h48qual보다 낮음) — cusum은 단순히
"같은 신호를 더 잘게 쪼갠 것"이 아니라 방향 판단 자체가 zigzag/h48qual과 1/3가량 다르다.

### zigzag 세그먼트 내부 포함관계

zigzag의 확정 스윙 세그먼트(n=2,050) 내부 bar 중 그 세그먼트와 같은 방향인 비율: h48qual
50.1%(사실상 반반), cusum 28.1%. 세그먼트당 h48qual/cusum 자신의 활성비율 상한(각 61.99%/
36.52%)을 감안해도 cusum이 zigzag 스윙에 "올라타는" 정도가 h48qual보다 뚜렷이 약하다.

### ATR%% 분포 (KS-test)

세 쌍 전부 p<1e-59로 유의하게 다름 -- cusum 활성bar의 ATR%% 중앙값(0.0025)이 zigzag(0.0022)/
h48qual(0.0023)보다 높다. cusum 임계값이 EWMA 변동성에 비례(메모리 기록, DC-vs-CUSUM 분석과
동일 메커니즘)해 상대적으로 더 변동성 높은 구간에서 발동하는 경향과 일치.

### 종합 해석

1. 시각적 "비슷함"의 원인은 같은-bar 시간정렬이 아니라, 셋 다 넓게 보면 ETH의 몇 안 되는
   대형 스윙 방향을 따라간다는 **약한 공통 트렌드 상관**과 **밀도 차이**(가격선을 촘촘히
   덮는 마커 자체가 시각적으로 "패턴이 비슷해 보이게" 만듦)에 가깝다.
2. h48qual은 zigzag와 구조적으로 가장 가깝다(±15분 방향일치 81.7%, 세그먼트포함 50.1%,
   flip_rate 3배 차이).
3. cusum은 zigzag/h48qual과 뚜렷이 다른 축이다 — 훨씬 짧은 커밋먼트(median 10분),
   방향일치율도 가장 낮음(66.1%), 더 높은 변동성 구간 선호. **그런데 5-way OOS 부호일관성은
   h48qual과 동률(3/3)로 가장 좋았다** — "구조가 다르다"가 "신호가 나쁘다"를 뜻하지 않는다는
   점은 열어둬야 한다. cusum의 상대적으로 나은 OOS 결과가 zigzag/h48qual과 다른, 더 국소적인
   (mean-reversion에 가까울 수 있는) 동역학을 포착했기 때문인지, 아니면 그저 N=3 노이즈인지는
   이 구조분석만으로 가르지 못한다 -- N≥5 재확인이 여전히 필요하다.

한계: h48qual 커버리지가 2026-02-28까지뿐(그 이후는 quality 미평가로 추정, 사용 전 확인
필요 -- 아직 근본원인 미조사)이라 이 구조분석은 2024-01~2026-02로 제한됐다. zigzag_segment_id
기반 세그먼트 방향은 세그먼트 내 최빈값(mode)으로 근사.

## 후속2: OOS 트레이드 원장 재구성 + 중요 정정 (2026-08-21)

`eth_zigzag_h48qual_cusum_oos_trade_ledger_20260821.py`. report.json은 집계통계만 저장하고
개별 트레이드는 저장하지 않아, `omega._metrics()`(exit_head 없는 하드 TP/SL/max_hold
백테스트, 저장된 학습모델 재로딩 불필요 -- oos_predictions_qXXX.csv만 있으면 됨)를 그대로
복제하되 매 트레이드를 기록하도록 계측. **재구성 집계(pnl/mdd/trades/wr)가 report.json과
소수점까지 정확히 일치함을 확인(cross-check 통과) -- 재구성 로직 신뢰 가능.** seed=133725056
(3개 라벨 공유 시드 중 첫번째, "베스트 시드 고르기" 아님)으로 통일.

### ⚠️ 정정: zigzag/h48qual과 dc/cusum의 "OOS"가 같은 기간이 아니었다

재구성 도중 `oos_predictions_*.csv` 실제 행수를 확인하다 발견: **zigzag/h48qual의 direction
소스(`zigzag_action_labels_20260531`)가 실제로는 2026-02-28에서 끊긴다**(파일명의
"20260531"과 무관, 빌드시점 원본데이터 자체 한계로 추정, 근본원인 미조사) — OOS가 실질
**16,897행(2026-01-01~02-28, 약 2개월)**뿐이었다. **dc/cusum(dense-cashfill 자체 라벨)은
canonical EVAL_CSV 전체 51,746행(2026-01-01~06-30, 약 6개월)을 그대로 커버.** 즉 5-way
비교 문서(위 "결과" 절)의 "OOS 부호" 비교는 **라벨마다 다른 길이의 기간**을 비교한 것이었다
— zigzag/h48qual은 2개월, dc/cusum은 6개월. h48qual/cusum의 "3/3 OOS 양수"를 동등하게
취급한 게 부정확했다. 이 사실은 5-way 학습 자체를 무효화하진 않는다(각 라벨이 실제로 가진
데이터 범위 안에서는 report.json 수치가 정확함, cross-check로 확인) — 다만 **라벨 간 비교의
공정성**에 문제가 있었다는 뜻이다.

### 트레이드 상세 (seed=133725056, VAL-베스트 threshold)

| 라벨 | 기간 | 트레이드 | 승률 | 평균승 | 평균패 | 청산사유 | 최종PnL |
|---|---|---|---|---|---|---|---|
| zigzag | 01-01~02-28 | 23 | 47.8% | +2.80% | -1.43% | SL11/TP11/forced1 | +13.76% |
| h48qual | 01-01~02-28 | 12 | 66.7% | +2.75% | -1.43% | SL4/TP8 | +17.10% |
| cusum(전체) | 01-01~06-30 | 59 | 47.5% | +2.64% | -1.54% | SL31/TP27/forced1 | +27.15% |
| **cusum(01-01~02-28만, 공정비교)** | 01-01~02-28 | 31 | 51.6% | +2.71% | -1.46% | SL15/TP16 | **+22.92%** |

**같은 2개월 구간으로 맞춰도 cusum(+22.92%)이 zigzag(+13.76%)/h48qual(+17.10%) 둘 다보다
높다** — 트레이드 수가 31건(zigzag의 1.3배, h48qual의 2.6배)으로 많아 표본이 더 두텁다는
점도 고려할 만하다. h48qual의 승률(66.7%)이 가장 높지만 표본이 12건뿐이라 신뢰구간이 넓다.
등가곡선 이미지: `tmp/research_20260821/chart_zigzag_h48qual_cusum_oos_equity_curves.png`
(zigzag/h48qual/cusum-Jan-Feb / cusum-전체 4개 곡선, 3개 다 01월 초반 -3~-4% 드로다운
이후 반등하는 유사한 형태를 보임 -- 앞선 구조분석에서 확인한 "약한 공통 트렌드 상관"과 정합).

### 다음 단계 갱신

zigzag/h48qual의 direction label(`zigzag_action_labels_20260531`)이 2026-02-28에서 끊기는
근본원인을 아직 조사하지 않았다 — N≥5 재확인을 하더라도 이 소스를 그대로 쓰면 zigzag/h48qual은
계속 2개월 OOS로 제한된다. 6개월 전체로 넓히려면 (a) 이 라벨소스를 06-30까지 재빌드하거나
(b) 2개월 제한을 인정하고 진행하거나 둘 중 하나를 결정해야 한다.
