# ETH Directional-Change/intrinsic-time triple-barrier 라벨 빌드 (2026-08-19)

**상태: 라벨 빌드 완료, 학습/평가 미실시**

## 배경

`docs/entry_exit_edge_external_labeling_literature_review_20260819.md` Part A 검토에서
TabM 3-head 모델의 direction_head 재테스트 후보 3개(directional-change 이벤트 샘플링,
CUSUM+TB, 분포적 회귀)를 선정했고, 사용자가 1번부터 착수를 요청했다.

Directional-Change/intrinsic-time 이벤트 샘플링(Razmi & Barak, arXiv:2501.06032)은
`core/event_label_engine.py`에 `event_method='directional_change'`로 이미 구현돼 있었지만,
저장소 전체에서 이 옵션으로 실제 라벨을 생성한 이력이 **단 한 번도 없었다** — 유일한 실행
이력은 모듈 자체의 `__main__` 스모크테스트(합성 데이터)뿐이었다. 이 세션은 이 엔진을 실제
ETH 5분봉 데이터에 처음으로 end-to-end 적용했다.

문헌조사 문서 자체는 이 후보에 "기대치는 낮게 잡으라"고 적었지만, 사용자는 이전 세션 이후
코드/데이터 환경이 달라졌다는 이유로 문서의 판정과 무관하게 재검증하기로 결정했다
(`eth_tabm_label_logic_retest_initiative_20260819` 메모리). 이 작업은 **라벨 생성까지만**이
범위다 — TabM 학습, 백테스트, promotion 판정은 후속 단계다.

## 설계 결정

사용자에게 직접 확인한 2가지 축:

- **배리어 컨벤션**: 엔진 자체의 `calibrate_barriers()` grid search로 자동튜닝(승인) —
  h48qual 로컬 배리어나 라이브 ATR 공식을 이식하는 대안은 `apply_triple_barrier()`를 우회해
  저수준 커널을 직접 호출해야 하는 추가 구현이 필요해 기각.
- **이벤트 밀도**: sparse(이벤트 발생 bar만, 승인) — forward-fill로 dense하게 채우는 대안은
  zigzag_action이 가졌던 것과 유사한 성격의 "낡은 라벨" 편향을 재도입할 위험이 있어 기각.

원시 bar는 기존 프로덕션 direction 라벨(zigzag_action, `build_zigzag_action_labels_v2_20260604.py`)과
동일한 `data/splits/year_oos/training_features_{2024,2025,2026_rebuilt}.csv`를 사용해, 이벤트
샘플링 로직만 다른 통제된 비교가 되게 했다.

## 구현

신규: [scripts/build_eth_directional_change_triple_barrier_labels_20260819.py](../../scripts/build_eth_directional_change_triple_barrier_labels_20260819.py)

- `core/event_label_engine.py`를 importlib 직접 로드로 사용(`core/__init__.py`의 python-binance
  의존성 때문에 이 dev 셸에서 일반 import가 깨짐 — `diagnose_eth_h48qual_dirhead_metalabel_via_event_label_engine_20260815.py`와
  동일 우회).
- 3개년 원시 bar를 concat(연도 경계 gap 없음 확인, 룩어헤드 없는 순방향 알고리즘이라 안전)한
  뒤 `directional_change_events(dc_theta=0.004)` → `calibrate_barriers()` → `generate_labels()`
  실행, 결과를 `event_time`의 연도로 재분할.
- 표준 3-way 라벨 `{-1,0,+1}` → `zigzag_action` 계약 `{0=CASH,1=LONG,2=SHORT}`로 매핑.
- `event_idx`/`t1_idx`(연도 내부 상대 위치 정수)는 출력에서 제외 — `zigzag_segment_id`가 과거
  연도별 재시작으로 병합 버그를 낸 전례와 같은 함정을 원천 차단.
- **파일명 수정 사항**: 최초 구현은 `dc_tb_action_labels_{year}.csv`로 출력했으나, 스모크
  테스트 중 `_read_labels()`(`train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py:70`)가
  `zigzag_action_labels_{year}.csv` 파일명을 하드코딩해서 찾는다는 걸 발견 — 별도 파일명을 쓰면
  `--direction-label-dir`로 이 디렉토리를 가리켜도 소비자가 파일을 못 찾는 문제가 있었다. 이미
  별도 `out_dir`로 격리돼 있어 파일명까지 다르게 할 필요가 없었으므로, 계약과 정확히 같은
  `zigzag_action_labels_{year}.csv`로 수정했다.

## 결과 (dc_theta=0.004, calibrate_barriers 자동튜닝)

자동튜닝 결과: `pt_mult=1.5, sl_mult=1.5, max_hold=24`(그리드 `pt/sl∈{1.0,1.5,2.0,3.0}`,
`max_hold∈{24,48,96}` 중 선택).

| 연도 | 이벤트 수 | LONG | SHORT | touch=pt | touch=sl | touch=timeout | 평균 보유 bar |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 7,665 | 48.83% | 51.17% | 48.51% | 50.87% | 0.63% | 3.82 |
| 2025 | 8,676 | 48.69% | 51.31% | 48.54% | 51.04% | 0.43% | 3.84 |
| 2026(~07-19) | 3,797 | 47.96% | 52.04% | 47.72% | 51.49% | 0.79% | 4.09 |

세 해 모두 LONG/SHORT가 48~52%로 균형적이고, `zigzag_action=0`(CASH)은 사실상 발생하지
않는다(0건) — DC 이벤트가 이미 "방향이 전환되고 있는 시점"만 골라내므로, 매 bar 상태를
서술하는 zigzag_action의 CASH 우세 분포와는 근본적으로 다른 형태다.

전체 `report.json`: `tmp/eth_directional_change_triple_barrier_labels_20260819/report.json`
(입력 경로, 그리드, per-year 상세, 알려진 한계 포함).

## 스모크 테스트

`train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py::_read_labels()`가 이
저장소에서 CLI로 라벨 디렉토리를 실제 교체할 수 있는 유일한 소비자임을 코드로 확인
(`train_eval_omega1_2_tabm_3head_20260603.py`는 `hard._build_frame(year)`으로 라벨을
하드코딩해 읽어 교체 불가능). 이 dev 셸엔 torch가 없어 `conda env` `quant_ai`
(torch 2.11.0+cu130)에서 실행:

```
[2025] _read_labels() 통과: n=8,676  dtype=int64  values=[1, 2]
[2026] _read_labels() 통과: n=3,797  dtype=int64  values=[1, 2]

[2025] feature bars=105,101  label bars=8,676  교집합비율=8.25%
[2026] feature bars=57,601  label bars=3,797  교집합비율=6.59%
```

스키마(`{timestamp,zigzag_action}`, int64, 값 ⊆ {0,1,2})는 무수정으로 통과한다. 다만
**sparse 출력이므로 하류 `_align()`(inner-join)과 결합하면 학습 표본이 전체 bar의 약
6.6~8.3%로 조용히 축소된다** — 크래시하지 않고 정상 동작처럼 보이는 게 함정이므로, 이
라벨로 실제 학습을 시도할 때 반드시 이 축소를 감안해야 한다.

## OOS 구간 시각 확인

Fresh-Forward OOS(2026-01-01~03-31) 구간에 대해 가격+라벨 오버레이 인터랙티브 차트를
만들어 사용자에게 전달함(이벤트 2,053건, LONG 979/SHORT 1,074). 코드 리뷰 외에 눈으로도
라벨이 실제 가격 변곡점 근처에 찍히는지 확인하는 용도 — 이 문서에는 정적 산출물을 남기지
않고 세션 내 Artifact로만 공유했다.

## ⚠️ 정정 — 30bp 비용기준은 근거없는 어림값이었음

아래 두 절(dc_theta 고빈도 실험, 이어지는 CUSUM 문서의 비용 비교)이 쓴 "30bp 미만이면
위험"이라는 기준은 저장소 어디서도 확인하지 않은 임의값이었다. 실제 코드 비용가정
(`train_eval_omega1_2_tabm_diffusion_risk_20260603.py:47-49`: FEE_RATE=5bp/side,
SLIP_RATE=2bp/side, MAKER_FEE_MULT=0.20)으로 재계산하면 실제 왕복비용은 **6bp(양쪽
메이커)~14bp(양쪽테이커)**이고, 이 기준으로는 DC의 OOS 평균 TP폭(39.4bp)이 최악의
경우 대비 2.8배 여유가 있다(<14bp 비율 2.9%, 위에서 말한 34.4%가 아님). **경제성
경고는 과장이었다.** 아래 dc_theta 스윕 실험 자체(고빈도가 경제성을 개선 못 한다는
결론)의 방향성은 유효하지만, "얼마나 심각한가"의 절대 수치는 30bp가 아니라 6~14bp
기준으로 다시 읽어야 한다. 상세: `eth_tabm_label_logic_retest_initiative_20260819`
메모리.

## dc_theta 고빈도 실험 (기각)

차트 리뷰 중 TP/SL 폭 중앙값(0.35%)의 34.4%가 30bp 미만이라는 비용-경제성 우려가 나온 뒤,
사용자가 "라벨을 더 많이 내보자"는 방향으로 `--dc-theta 0.002`(기존 0.004의 절반)를
`tmp/eth_directional_change_triple_barrier_labels_theta002_20260819/`에 실험했다.

| | theta=0.004(기존, 채택) | theta=0.002(실험) |
|---|---:|---:|
| OOS 이벤트 수 | 2,053 | 4,233 |
| 하루 평균 | 23.1건 | 47.6건 |
| 평균 간격 | 62.4분 | 30.3분 |
| TP/SL폭 중앙값 | 0.353% | 0.333% |
| TP폭 < 30bp 비율 | 34.4% | 40.6% |

이벤트 수는 2배 이상 늘었지만 **TP/SL폭 중앙값은 오히려 더 좁아지고 비용 미달 비율(<30bp)도
34.4%→40.6%로 악화** — theta를 낮춰 신호 빈도를 올리는 방향은 경제성을 개선하지 못한다.
사용자 지시로 이 실험은 기각하고, **canonical 출력은 theta=0.004(기존)로 유지**한다.
theta002 디렉토리는 삭제하지 않고 참고용으로만 남겨둔다.

## 스코프 경고

이 문서의 라벨 통계(균형 잡힌 분포, 낮은 timeout 비율 등)는 **라벨이 정상적으로 생성됐다는
것만 보여준다 — 방향 예측 edge가 있다는 근거가 아니다.** `docs/label_methodology_survey_20260815.md`가
기록한 대로 이 저장소의 40개 이상 선행 라벨 방법론이 전부 "학습 가능하나 방향 edge 없음"으로
수렴했다. edge 판정은 별도의 TabM 학습 + Fresh-Forward 평가 단계의 몫이며, 아직 그 단계는
시작하지 않았다.

## 다음 단계 (미착수)

1. 이 라벨로 TabM direction_head를 실제 학습(`quality-mode=same_as_direction` 최소 구성으로
   시작 가능 — 추가 진단 컬럼 없이도 동작).
2. Fresh-Forward VAL(2025-09~12)/OOS(2026-01~03) 평가, always-direction 벤치마크 대비 비교.
3. N≥5 시드로 재확인(CLAUDE.md Seed-Diversity Ensemble Promotion Gate) — 단일 시드 결과만으로
   판단하지 않는다.
4. 재테스트 후보 2/3(CUSUM+TB, 분포적 회귀)은 이 라인과 독립적으로 착수 예정
   (`eth_tabm_label_logic_retest_initiative_20260819` 메모리).

## 참고

- `docs/entry_exit_edge_external_labeling_literature_review_20260819.md` — 후보 선정 근거
- `docs/label_methodology_survey_20260815.md` — 40+ 선행 라벨 방법론 메타발견
- `core/event_label_engine.py` — 엔진 본체
- `scripts/build_zigzag_action_labels_v2_20260604.py` — 원시 bar 입력 관례를 그대로 따른 기존 direction 라벨 빌더
- 메모리: `eth_tabm_label_logic_retest_initiative_20260819`
