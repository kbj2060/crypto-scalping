# BTC 복합 오실레이터(orthogonal_combo) TabPFN 메타라벨 최종 확인 (2026-09-01)

호메로스 프로젝트 플래그십 신호 `orthogonal_combo`의 BTC 포팅, TabPFN 단계(라운드1 그리드스크린 →
라운드2 HIT_TYPE 그리드스크린 → 이번 TabPFN). 라벨점은 라운드2에서 이미 확정: **touch_mfe(순수
터치기반 MFE), H=8, K=2.0** — 라운드2의 "전역 1위"(touch_giveback_sustained, H=8/K=3.0)가 OOS
hit 2~5건으로 표본붕괴해 명시적으로 불신됐기 때문에, 그 대신 라운드2 자체 리더보드에서 표본이
가장 두터운(OOS hit 49 bottom/31 top) touch_mfe 지점을 그대로 썼다(변경 없음, 이번 라운드 과제
스펙에 고정값으로 명시됨).

## 실행 메모 — 1줄 버그 수정

최초 실행이 `TypeError: can't compare offset-naive and offset-aware datetimes`로 즉시 죽었다.
원인: `build_fires()`에서 `ts = frame["timestamp"].to_numpy()`가 tz-aware 컬럼을 tz-aware
`Timestamp` 객체의 object-dtype 배열로 반환하는데, 이를 `np.datetime64(START)`와 비교하면
`np.datetime64`가 조용히 tz 정보를 버리고(`UserWarning: no explicit representation of timezones
available for np.datetime64`) naive가 되어 tz-aware 값과 비교시 예외가 난다. 수정은 1줄, 로직
변경 없음: `ts[idx] >= np.datetime64(START)` → `ts[idx] >= START` (양쪽 다 tz-aware `Timestamp`로
직접 비교, 필터링 결과는 의도한 그대로). 로컬·서버 양쪽 스크립트 파일에 동일하게 반영, 재실행 후
정상 완주. 라벨 정의·피쳐·분할·시드 등 다른 어떤 것도 건드리지 않았다.

## 결과 — VAL/OOS/HOLDOUT AUC (TabPFN, 4시드)

| 구간 | AUC (mean±std) | n_train | n_eval | naive 다수클래스 정확도 | 모델 accuracy |
|---|---:|---:|---:|---:|---:|
| VAL | **0.6512 ± 0.0013** | 1,667 | 282 | 0.6135 | 0.6711 |
| OOS | **0.5891 ± 0.0027** | 1,667 | 224 | 0.6295 | 0.6496 |
| HOLDOUT (1회성) | **0.5933 ± 0.0017** | 1,667 | 338 | 0.6331 | 0.6058 |

전체 fire 2,511건(bottom 1,382/top 1,129, GAP=6 클러스터 dedup 후), hit-rate TRAIN 42.7%/VAL
38.7%/OOS 37.1%/HOLDOUT 36.7%. 시드분산은 세 구간 모두 매우 작다(±0.0013~0.0027) — 즉 아래
"불안정성" 논의는 TabPFN 자체의 시드 노이즈가 원인이 아니다.

⚠️ HOLDOUT에서 **모델 accuracy(0.6058)가 naive 다수클래스 정확도(0.6331)보다 낮다** —
threshold=0.5 기준 원시 정확도로는 "그냥 항상 miss라고 찍는" 기준선보다 못하다. AUC(0.5933)는
랭킹 능력만 보는 지표라 0.50(무작위)보다는 확실히 위지만, 실사용 임계값 기준 실익은 OOS/HOLDOUT
모두 얇다.

## 피쳐 중요도 (VAL, 순열중요도 5회반복, 단일시드 20260829, baseline AUC 0.6522)

| 순위 | 피쳐 | importance_mean | std |
|---:|---|---:|---:|
| 1 | `atr_percentile_864` | **+0.0586** | 0.0111 |
| 2 | `hour_utc` | +0.0250 | 0.0235 |
| 3 | `atr_pct` | +0.0120 | 0.0069 |
| 4 | `nyse_open_flag` | +0.0057 | 0.0051 |
| 5 | `bb_pctb` | +0.0055 | 0.0062 |

`atr_percentile_864`(변동성 백분위)가 압도적 1위 — 라운드1/2 내내 반복된 패턴과 일관(발동시점
변동성 낮음=유리). 6위 이하는 대부분 ±0.003~0.005 잡음대 안에 있어 사실상 무구분(`is_bottom`
-0.0025, `p_fast` -0.0031, `ndi` -0.0032 등 미세 음수 다수).

**주의할 이례점**: 세션타이밍 피쳐 `hour_utc`(2위)와 `nyse_open_flag`(4위)가 이 VAL 순열중요도
상위권에 있다. 이는 ETH `orthogonal_combo` 최종본의 ablation 결과와 **정반대 방향**이다 — ETH는
세션타이밍 3종(hour_utc/nyse_open_flag/weekday) 제거가 VAL만 손해고 OOS(+0.011)/HOLDOUT(+0.017)은
오히려 개선되는 "VAL 한정 과적합" 패턴이 확인되어 최종 20피쳐(세션타이밍 제외)로 확정됐다
(`docs/homer/README.md` orthogonal_combo 섹션). 이번 BTC 순열중요도는 VAL 단독 측정이라 이 두
피쳐가 BTC에서도 같은 방식으로 OOS/HOLDOUT을 갉아먹는지는 **검증되지 않았다** — ablation을
직접 돌리지 않았으므로 가설로만 남긴다(아래 다음 단계).

## 라운드2 raw-lift 진단 대비 — TabPFN이 확인하나, 복잡하게 만드나

라운드2 진단(`docs/experiments/btc_5m_orthogonal_combo_gridscreen_featureanalysis_20260901.md`):
touch_mfe를 포함한 4개 HIT_TYPE 전부에서 인접 그리드 셀 사이 OOS 리프트가 **0.556~1.148x로
뚜렷한 패턴 없이 출렁였다** — 가장 근거 있는 해석은 BTC OOS 구간의 작은 후보수(사이드당
76~131건)였지, HIT 정의 자체의 결함이 아니었다. 이번 라운드가 채택한 touch_mfe/H=8/K=2.0
지점 자신의 raw 리프트는 TRAIN 1.505 / VAL 1.318 / **OOS 1.148**(OOS hit 49 bottom/31 top) —
1.0을 넘긴 "생존" 지점이었지만, 인접 셀 민감도 진단(K=3.0 고정, H만 바꾸면 OOS가 0.556~1.000
사이를 움직임)이 이 1.148도 "그 지점만의 우연"일 수 있다는 의심을 남겼다.

TabPFN(24피쳐, 4시드, 전체모집단, exclude-middle 없음)로 같은 고정 지점을 다시 보면:

- **시드분산은 매우 작다**(±0.0013~0.0027) — 라운드2가 우려한 "출렁임"이 TabPFN 자체의
  실행별 노이즈에서 오는 건 아니다.
- **OOS(0.5891)와 HOLDOUT(0.5933)이 서로 거의 일치한다** — HOLDOUT은 라운드2 그리드서치가
  전혀 손대지 않은 완전히 새로운 구간인데도(`holdout_touched=false`였던 라운드2와 달리 이번이
  이 지점의 첫 HOLDOUT 노출), 독립적으로 재현된 OOS와 거의 같은 값에 도달했다. 순수 표본
  노이즈/그리드 셀 우연이었다면 HOLDOUT이 VAL 수준(0.65)이나 0.50 근처로 크게 벗어날 가능성이
  충분했는데 그러지 않았다.
- 대신 뚜렷한 **VAL(0.651) → OOS(0.589)/HOLDOUT(0.593) 하락**이 재현성 있게 나타난다 — "셀마다
  운이 다르다"보다는 "이 지점의 진짜 판별력이 VAL보다 OOS/HOLDOUT에서 일관되게 낮다"는 쪽에
  가까운 그림이다.

**정리하면 TabPFN은 라운드2 진단을 한쪽 방향으로는 복잡하게 만들고, 다른 방향으로는 확인한다.**
"그리드 셀마다 뚜렷한 패턴 없이 출렁인다"는 라운드2의 raw-lift 묘사와 달리, TabPFN 기준으로는
OOS와 HOLDOUT이 서로 잘 맞는 안정적이고 재현 가능한 패턴을 보인다 — 즉 이 지점의 OOS>1.0
리프트가 순전히 운이었다는 가설은 힘을 잃는다. 그러나 그 안정적인 패턴 자체가 "약한 신호"라는
결론은 그대로다(오히려 accuracy가 naive보다 낮아지는 HOLDOUT처럼 더 구체적으로 나쁘다) — raw-lift가
말한 "표본이 작아 신뢰하기 어렵다"는 경고와, TabPFN이 보여준 "표본과 무관하게 재현되는 얕은
edge"라는 결론은 서로 다른 메커니즘이지만 둘 다 **승격 기준에 한참 못 미친다**는 같은 결론으로
수렴한다.

## ETH `orthogonal_combo` TabPFN 대비

`docs/homer/README.md` 기준 ETH `orthogonal_combo` v2 최종 수치는 두 갈래다:

| 평가방식 | VAL | OOS | HOLDOUT | 비고 |
|---|---:|---:|---:|---|
| ETH kept-only 헤드라인(20피쳐, exclude-middle 64%만 평가) | 0.6844 | 0.7274 | 0.7245 | ⚠️심층검증으로 하향조정됨, "역대최고" 인용 금지 |
| **ETH 전체모집단 재평가**(exclude-middle 없이 전체) | **~0.665** | **~0.680** | **~0.667** | 이번 BTC run과 방법론상 동급 비교 대상 |
| **BTC (이번 run, 24피쳐, exclude-middle 없음)** | **0.6512** | **0.5891** | **0.5933** | — |

이번 BTC 스크립트는 애초에 exclude-middle을 쓰지 않는 순수 이진 라벨(binary hit/miss, 라운드2가
가진 clean touch_mfe 라벨에 ETH식 exclude-middle을 이식하지 말라는 이번 과제 스펙 지시)이므로,
공정한 비교 대상은 ETH의 kept-only 헤드라인(0.684/0.727/0.725, 64%만 평가한 더 쉬운 부분집합)이
아니라 **ETH의 전체모집단 재평가(약 0.665/0.680/0.667)**다. 그 기준으로도 BTC는 VAL/OOS/HOLDOUT
전 구간에서 ETH보다 낮고, 특히 OOS/HOLDOUT 격차가 크다(BTC 0.589/0.593 vs ETH 0.680/0.667, 약
0.07~0.09 낮음). 방향성도 다르다 — ETH는 OOS·HOLDOUT이 VAL과 비슷하거나 오히려 높은(edge가
out-of-sample에서 죽지 않는) 패턴인 반면, BTC는 VAL이 가장 높고 OOS/HOLDOUT이 뚜렷이 낮은
반대 패턴이다. 참고로 이 스크립트의 report.json에 미리 박혀 있던 `eth_orthogonal_combo_tabpfn_
reference`(VAL 0.723/OOS 0.7162/HOLDOUT 0.7076)는 실제로는 ablation 이전 "전체 23피쳐" 단계
수치(`docs/homer/README.md` ablation 표의 첫 행과 일치)이지 최종 v2 헤드라인이 아니다 — 이 문서의
비교는 과제 지시대로 `docs/homer/README.md` 원문의 정확한 최종 수치를 썼다.

## 결론

`status: exploratory_single_signal_below_promotion_bar` 그대로 유지. BTC `orthogonal_combo`는
touch_mfe/H=8/K=2.0 고정점에서 VAL은 쓸만하지만(0.651, naive 대비 +5.8pp accuracy) OOS·HOLDOUT은
얕고(AUC 0.59 내외) HOLDOUT 원시 accuracy는 naive보다도 낮다 — 경제성 게이트·배포 후보로 진행할
근거가 없다. 라운드2의 "raw-lift가 그리드 셀마다 출렁여 신뢰하기 어렵다"는 우려는, TabPFN으로 보면
"출렁임"이라기보다 "OOS·HOLDOUT에서 일관되게 얕은 edge"로 더 명확해졌을 뿐 해소되지 않았다.

**다음 단계(미실시, 제안만)**: (1) 세션타이밍 3피쳐(hour_utc/nyse_open_flag/weekday) ablation을
BTC에서도 직접 돌려 ETH와 같은 "VAL 한정 과적합"인지 확인 — 이번 순열중요도가 그 가능성을
시사하지만 검증되지 않음. (2) 경제성 백테스트는 이 AUC 수준에서는 착수 보류 권장.

## 산출물

- `scripts/research_btc_orthogonal_combo_metalabel_tabpfn_20260901.py` (서버 실행, tz 비교 버그
  1줄 수정 후 완주 — 로컬·서버 양쪽 반영)
- `data/labels/btc_5m_evidence_signal_candidates_20260901/orthogonal_combo_tabpfn_report.json`
  (로컬로 회수 완료)
- `data/labels/btc_5m_evidence_signal_candidates_20260901/orthogonal_combo_tabpfn_features.csv`
  (서버에만 존재, 2,511건 fire+피쳐+hit 원본)
- 라운드1: `scripts/research_btc_orthogonal_combo_gridscreen_20260901.py`
- 라운드2: `scripts/research_btc_orthogonal_combo_gridscreen_hittype_20260901.py`,
  `docs/experiments/btc_5m_orthogonal_combo_gridscreen_featureanalysis_20260901.md`

## Fresh-Forward 체크리스트

`fresh_forward_bar_by_bar=false`(라벨 분리력 TabPFN 분류 패스, 봉단위 순차 TP/SL 백테스트
아님), `trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`, `holdout_touched=true`(이 지점 최초 1회성 HOLDOUT 평가).
