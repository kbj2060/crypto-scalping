# BTC 칼만편차 평균회귀(kalman_deviation_meanrev) TabPFN 메타라벨 확인 (2026-09-01)

ETH 대시보드 증거신호(2026-08-31 신설, `compute_signals()` `SIGNAL_ORDER`)의 `kalman_deviation_meanrev`를
BTC로 이식. 그리드 스크리닝(CPU)과 TabPFN 학습(GPU)을 한 스크립트로 결합해 실행했다. ETH 자체
결과(H=12/GAP=12/K=2.5, VAL/OOS/HOLDOUT AUC **0.6569/0.6311/0.6284**)가 이미 확립되어 있어 별도
심층탐색 없이 "빠른 확인/보정" 수준으로 설계했으나, 진행 중 그리드 경계값 함정을 두 번 만나
각각 진단·해결했다(아래 상세).

## 요약

| 구간 | AUC (mean±std, 4시드) | n_train | n_eval |
|---|---|---:|---:|
| VAL | **0.7288 ± 0.0014** | 5,265 | 1,095 |
| OOS | **0.6242 ± 0.0006** | 5,265 | 783 |
| HOLDOUT (1회성) | **0.6709 ± 0.0015** | 5,265 | 1,261 |

**ETH 자체 결과(VAL 0.6569/OOS 0.6311/HOLDOUT 0.6284) 대비 VAL·HOLDOUT은 더 높고, OOS는 근소하게
낮다** — 세 구간 모두 ETH와 비교 가능한 수준이거나 그 이상. 최종 채택 (H,K,GAP) = **(10, 3.5, 6)**
— ETH 자체 설정(12, 2.5, 12)과는 GAP을 과제 지시대로 6으로 고정했고 HORIZON/K는 자체 그리드
스크리닝으로 확인했다(과정은 아래 참조). 원시 발동 바닥7,802건/천장8,278건 → 클러스터 중복제거(GAP=6)
후 바닥4,115건/천장4,315건, dropna 후 최종 8,404건(풀링 히트율 14.12%).

## 그리드 스크리닝 — 경계값 함정 두 번, 각각 진단·해결

과제가 지정한 원안은 "TRAIN 리프트 vs 무작위바 기준선"이었으나, 실행 중 두 가지 문제를 만나 둘 다
근본 원인을 확인한 뒤 방법을 조정했다. **둘 다 은폐하지 않고 스크립트 docstring·report.json에
그대로 기록**해 두었다(`grid_screen.selection_metric_note`, `grid_screen.boundary_check`).

### 문제1 — 자기 ATR 정규화로 인한 리프트 부호 역전

원안(각 바 자신의 `atr_pct`로 정규화한 히트 임계값, `research_btc_taker_delta_climax_gridscreen_
20260901.py`의 `random_baseline_hit` 패턴을 그대로 이식)으로 36칸(H×K) 전부를 계산하니 **전 칸이
리프트 <1.0x**(0.73x~0.95x)로 나오고 argmax가 K=1.5(그리드 경계)에 걸렸다. 진단: `kalman_dev_z`
극값 발동봉은 발동 그 순간 자신의 `atr_pct`가 이미 국소적으로 부풀어 있다(TRAIN 발동봉 평균
atr_pct 0.00299~0.00318 vs 미발동 풀 평균 0.00187, **~1.6~1.7배**) — 칼만편차와 롤링ATR이 같은
최근 변동성 급등에 함께 반응하기 때문. 발동봉 "자신의" 부풀려진 atr_pct로 히트 임계값을 정규화하면
발동봉에만 불리한 허들이 생겨, 실제로는 더 큰 원시 순방향 움직임(발동봉 평균 순방향 MFE가 풀 대비
바닥 53%/천장 56% 더 큼)을 가리는 것으로 확인됐다. **고정 임계값(TRAIN 풀의 중앙값 atr_pct 사용,
바닥/천장 양쪽에 동일 적용) 교차검증으로 확인**: ETH 자체 중심점(H=12,K=2.5)에서 리프트가
**바닥 1.5529x / 천장 1.5773x**로 뒤집힘 — ETH 자체 보고 리프트(바닥2.36x/천장2.16x)와 같은
자릿수. 해결책: 그리드 선택기준을 **GBM(HistGradientBoostingClassifier) VAL AUC**로 교체(원래
`atr_pct`가 무작위기준선 계산의 하드코딩 정규화 상수로 쓰였던 문제 자체가, GBM에서는 `atr_pct`가
그냥 24개 피쳐 중 하나로 들어가므로 자연히 해소됨) — 이는 ETH 자체가 이 신호에 실제로 썼던 방법
그대로다(`research_eth_kalman_demarker_gridscreen_20260831.py::screen_signal`). 자기정규화
리프트는 진단용으로 그리드 표에 계속 남겨뒀다(선택에는 미사용).

### 문제2 — GBM 선택도 그리드 하한(H=8)에 걸림, 확장검사로 해결

GBM AUC 기준 36칸 argmax는 **H=8, K=3.5(VAL AUC 0.7017)** — 또 그리드 경계(H=8)였다. 다만 H=10,
K=3.5(0.7013)가 사실상 동률이라 "진짜 하한쪽으로 계속 개선되는 추세"인지 "노이즈로 우연히 경계에
걸린 것"인지 판단이 필요했다. **1회성 확장검사**(H∈{4,5,6,7}, 동일 K그리드)를 실행:

| H(대표 최적K) | GBM VAL AUC |
|---:|---:|
| 4 (K=3.5) | 0.7388 |
| 8 (K=3.5, 원 그리드 하한) | 0.7017 |
| **10 (K=3.5, 최종 채택)** | **0.7013** |
| 12 (K=3.5) | 0.6655 |
| 16 (K=4.0) | 0.6429 |
| 20 (K=3.5) | 0.6465 |
| 24 (K=3.5) | 0.6301 |

확장 결과 H=4가 명목상 가장 높지만(0.7388) **그 자체가 새 경계(H=4)에 다시 걸렸고**, 인접 K값들이
매끄럽지 않고 요동친다(예: H=8에서 K=4.0→0.6519인데 K=4.5→0.7396로 급등 — 단일시드 무정규화 GBM의
표본노이즈로 판단, 진짜 신호라면 인접 K 사이 이런 급격한 비단조 변화가 나오기 어려움). **결정규칙**:
확장검사의 최고점이 (a) 자기 자신도 새 경계가 아니고 (b) 원 그리드의 최고 비경계칸보다 0.02 AUC
이상 확실히 앞서야만 채택 — 이번엔 (a)부터 불충족(H=4=확장그리드 하한)이라 **원 그리드의 최고
비경계칸인 H=10, K=3.5로 폴백**. 이 로직은 `finalize_horizon_choice()`에 재사용 가능한 함수로
구현해 report.json에 원시 argmax(`raw_argmax`)·확장검사 전체(`boundary_check`)·최종
채택(`chosen`)을 전부 남겼다.

## 최종 라벨 정의

`entry=close[i]`(발동봉 종가), `atr_pct=atr/close`. 바닥: `[i+1,i+10]` 구간 intrabar 고가가
`entry*(1+3.5*atr_pct)` 이상 터치하면 hit. 천장은 저가 기준 미러. 같은 방향 발동이 6봉 이내로
붙어있으면 `kalman_dev_z` 최극값 봉만 남기고 나머지는 병합(클러스터 중복제거, GAP=6 — 과제 지시로
고정, ETH 자체가 이 신호에 쓴 GAP=12는 재탐색하지 않음).

## 피쳐 (24개 = ETH 표준 23개 + `kalman_dev_z` 자신)

`research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py::FEATURE_COLUMNS` 23개를 그대로
가져오고(`is_bottom`/`delta_z`/`atr_pct`/`atr_percentile_864`/`hour_utc`/`weekday`/
`nyse_open_flag`/`p_fast`/`p_slow`/`ret3_z`/`vwap_dev_z`/`cvd_roll_roc_48`/`vol_z`/
`lower_wick_ratio`/`upper_wick_ratio`/`bb_pctb`/`adx14`/`pdi`/`ndi`/`bb_width_pctile`/`er_24`/
`realized_vol_ratio`/`rsi`), `kalman_dev_z` 자신을 24번째로 추가 — ETH 자체가 이 신호(및
demarker_extreme)에 쓴 확립된 컨벤션 그대로.

## 피쳐 중요도 (VAL, 순열중요도, 단일시드 baseline AUC 0.7279, 5회 반복)

| 순위 | 피쳐 | importance |
|---:|---|---:|
| 1 | `hour_utc` | **+0.06103** |
| 2 | `atr_pct` | +0.02838 |
| 3 | `bb_pctb` | +0.02108 |
| 4 | `atr_percentile_864` | +0.00890 |
| 5 | `adx14` | +0.00771 |
| 6 | `nyse_open_flag` | +0.00682 |
| 7 | `ndi` | +0.00653 |
| 8 | `er_24` | +0.00517 |
| 9 | `pdi` | +0.00481 |
| 10 | `rsi` | +0.00423 |

`hour_utc`가 다음 피쳐(atr_pct+bb_pctb 합산)보다도 큰 폭으로 1위 — 세션별 변동성 패턴이 이
프로젝트에서 이미 확인된 실제 효과(`eth_market_open_volatility_window_20260826` 등)와 궤를 같이
한다. 흥미롭게도 **트리거 자신인 `kalman_dev_z`는 24개 중 17위(+0.00130)로 하위권** — "트리거는
유효하게 발동하지만 세부 신뢰도는 다른 문맥 변수가 결정한다"는 이 프로젝트의 기존 패턴(dalton,
demarker)과 일치. `bb_pctb`가 상위권인 것도 demarker의 "트리거 자신보다 bb_pctb가 지배적" 발견과
유사한 방향이나, 이번 세션에서 별도 룩어헤드 재감사는 수행하지 않았다(캐비엇 참조).

## ETH 대비 정직한 비교

| | ETH (H=12/GAP=12/K=2.5) | BTC (H=10/GAP=6/K=3.5) |
|---|---:|---:|
| VAL AUC | 0.6569 | **0.7288** |
| OOS AUC | **0.6311** | 0.6242 |
| HOLDOUT AUC | 0.6284 | **0.6709** |

3구간 중 2구간(VAL, HOLDOUT)에서 BTC가 더 높고, OOS는 근소하게 낮다(0.6242 vs 0.6311, 차이
0.0069로 4시드 표준편차 0.0006~0.0015 대비 크지만 실질적으로는 작은 차이). **다만 HORIZON(12→10)과
GAP(12→6)이 모두 다르므로 완전한 apples-to-apples는 아니다** — GAP은 과제 지시로 애초에 고정,
HORIZON은 이번 그리드 스크리닝에서 독립적으로 확인된 값이라 우연히 ETH 중심(12)에 가깝지만 동일하지
않다. 그래도 ETH 중심점(H=12,K=2.5)에서의 고정임계값 리프트 교차검증(바닥1.55x/천장1.58x, ETH 자체
2.36x/2.16x와 같은 자릿수)까지 포함하면, **BTC에서도 이 신호가 ETH와 유사한 성질(원시 평균회귀
리프트 + 학습가능한 24피쳐 신호)을 가진다는 결론은 안정적**이라고 판단한다.

## 캐비엇

- **`balanced_accuracy=0.5000`, `accuracy=naive_majority_accuracy`가 모든 시드·모든 구간에서
  동일하게 나옴** — 버그가 아니라 양성비율(14.12%)이 낮아 기본 임계값 0.5에서 TabPFN이 전원
  음성(no-hit)으로만 예측하기 때문(순위지표인 AUC는 0.62~0.73으로 정상적으로 정보량을 담고 있음).
  `evaluate()` 자체가 ETH 원본에서 그대로 포팅된 함수라 이 프로젝트의 다른 불균형 라벨 리포트에서도
  같은 패턴이 나타날 수 있다.
- **단일시드 무정규화 GBM 그리드는 표본노이즈가 크다** — 위 "문제2"에서 실측(인접 K값 사이 0.05~0.09
  AUC대 요동). 최종 TabPFN 단계는 4시드 평균(표준편차 0.0006~0.0015)이라 훨씬 안정적이지만, (H,K)
  "선택" 자체는 노이즈가 큰 단일적합 GBM에 의존했다는 한계가 있다 — `finalize_horizon_choice()`의
  보수적 폴백 규칙으로 완화했으나 완전히 제거하지는 못한다.
- GAP(=6)은 과제 지시로 고정, 재탐색하지 않았다 — ETH 자체 최적(GAP=12)과 다르므로 GAP까지 포함한
  재탐색을 하면 결과가 바뀔 수 있다.
- 자기상관 레짐게이트(평균회귀 vs 모멘텀)는 ETH에서 이미 반대 방향으로 확인되어(모멘텀 레짐 발동이
  더 예측력이 높음, `docs/homer/README.md` 참조) 이번에도 적용하지 않았다 — BTC에서 별도 확인은
  하지 않았다.
- 경제성(cost-gate) 백테스트, 대시보드 배포는 이번 세션 범위 밖(요청되지 않음, 미실행).
- `fresh_forward_bar_by_bar=true`, `trade_ledgers_used_as_input=false`,
  `saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false` — report.json에
  명시. VAL/OOS/HOLDOUT은 학습에 전혀 쓰이지 않은 TRAIN-fit 모델로만 평가했고 HOLDOUT은 이번이
  유일한 노출.

## 다음 단계 (미실행)

- GAP 재탐색(현재 6으로 고정) — ETH 자체 최적(12)과 다른 이유가 진짜 자산차이인지 확인 필요.
- 경제성 게이트(트레일링스톱 SL/ARM/Trail 그리드), 대시보드 증거신호 칩 배포 여부 판단.
- `bb_pctb`/`hour_utc` 상위 지배에 대한 룩어헤드·오염 감사(demarker 사례처럼 필요할 수 있음).

## 파일 목록

- `scripts/research_btc_kalman_deviation_meanrev_metalabel_tabpfn_20260901.py` — 그리드 스크리닝
  (경계값 확장검사 포함) + TabPFN 학습 결합 스크립트, 서버(quant_ai, CUDA) 실행.
- `data/labels/btc_5m_evidence_signal_candidates_20260901/kalman_deviation_meanrev_tabpfn_report.json`
  — 전체 36칸 그리드, 경계 확장검사 전체, ETH중심 고정임계값 교차검증, TabPFN VAL/OOS/HOLDOUT
  시드별 결과, 순열중요도 24개 전체.
- `data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_kalman_deviation_meanrev_metalabel_features.csv`
  — 최종 채택 (H=10,K=3.5,GAP=6) 기준 발동봉별 24피쳐+hit 라벨.
- `data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv`
  — 입력 Tier0 데이터(별도 세션 기생성, `kalman_dev_z`는 미포함이라 이번 스크립트가 자체 계산).
