# BTC short_term_return_z 그리드 스크리닝 라운드2 — HIT_TYPE을 3번째 탐색축으로 (2026-09-01)

라운드1(`research_btc_short_term_return_z_gridscreen_20260901.py`)은 HORIZON×K만 그리드서치하고
hit 정의는 터치기반 MFE(touch_mfe) 하나로 고정했다. 사용자의 후속 질문 — "터치기반 MFE가 맞는
hit 정의라는 보장이 어디 있나, HORIZON·K뿐 아니라 **hit 정의 자체**도 신호별로 그리드서치해야
하는 거 아니냐" — 에 따라 이번 라운드는 **HIT_TYPE을 3번째 탐색축**으로 추가해 재실행했다.
특히 라운드1이 미해결로 남긴 **H=2(강함·불안정) vs H=6(약함·안정) 트레이드오프**가 hit 정의를
바꾸면 더 깔끔하게 풀리는지 확인하는 것이 핵심 목적이다.

## 라운드1(터치기반 단일방식) 대비 변경점

**결론부터: 트레이드오프는 절반만 풀렸다.**

- ❌ **단기(H=2~3) 자체는 여전히 불안정** — `touch_mae_capped`/`touch_giveback_sustained` 등
  "더 엄격한" hit 정의를 써도 H=2/H=3 최강 셀은 TRAIN/VAL/OOS 스프레드가 39~68%로 여전히
  요동친다(아래 "단기 불안정성은 hit 정의를 안 탄다" 표 참조). 사용자가 가설로 제시한
  "`touch_giveback_sustained`나 `touch_mae_capped`가 단기 호라이즌에서 자연스럽게 더 안정적일
  수 있다"는 **기각** — 오히려 `touch_giveback_sustained`는 H=2에서 4개 방식 중 **가장**
  불안정했다(flat 68.1%, touch_mfe의 59.3%보다도 나쁨).
- ✅ **단, 라운드1이 이미 찾은 "안정 구간"(H≈6) 안에서는 hit 정의가 실질적 차이를 만든다.**
  같은 H=6,K=1.75(라운드1의 권고 지점)에서 hit 정의만 바꿔보면:

  | HIT_TYPE (H=6,K=1.75 고정) | TRAIN 리프트 | VAL | OOS | TRAIN↔VAL↔OOS 스프레드 |
  |---|---:|---:|---:|---:|
  | `touch_mfe`(라운드1) | 1.559 | 1.504 | 1.545 | 3.67% |
  | `close_at_h` | 1.684 | 1.636 | 1.627 | 3.47% |
  | **`touch_mae_capped`** | **1.531** | **1.504** | **1.531** | **1.78%** |
  | `touch_giveback_sustained` | 2.073 | 1.682 | 2.539 | 50.9%(★같은 H인데 오히려 최악) |

  `touch_mae_capped`가 라운드1과 리프트 크기는 거의 동일(VAL 1.504로 완전 동일값)하면서
  스프레드를 **3.67%→1.78%로 절반 가까이 줄인다** — "먼저 target을 K*ATR 이상 찍었더라도,
  그 전에 반대방향으로 K_LOSS_MULT(=2.0)*ATR 이상 밀렸으면 무효" 조건이 TRAIN에서만 우연히
  살아남던 "선피해후회복" 케이스를 걸러내, TRAIN을 VAL/OOS 쪽으로 끌어내리는 방식으로
  작동한다(TRAIN이 1.559→1.531로 살짝 낮아지고 VAL/OOS는 거의 그대로 — "TRAIN 과최적화를
  깎아 VAL/OOS와 맞춘다"는 정확히 원하던 효과).
  K까지 재탐색하면(K=2.0) 스프레드가 **0.90%**까지 더 줄어든다 — 그리드 120칸 전체를
  통틀어 가장 평평한 셀. 같은 H=6에서 HGB 단일시드 VAL AUC도 `touch_mfe`(라운드1 자체
  재현측정 0.617) 대비 `touch_mae_capped`가 **0.656**로 소폭 상승 — 안정성뿐 아니라
  분류기 학습가능성 측면에서도 라벨이 "덜 시끄러워졌다"는 방향이 일치한다.

**요약**: hit 정의를 바꿔서 H=2를 "구제"할 수는 없었다(불안정성은 신호 자체의 표본희소성
문제이지 touch_mfe만의 결함이 아니었음이 이번에 확인됨). 하지만 라운드1이 이미 지목한 안정
구간(H≈6) **안에서** `touch_mae_capped`로 바꾸면 같은 리프트를 더 적은 TRAIN↔VAL↔OOS 흔들림으로
얻는다 — 이번 라운드의 최종 권고는 **HIT_TYPE=touch_mae_capped, H=6, K=2.0**(아래 상세).

## 데이터·분할·공통 방법론(라운드1과 동일)

`data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv`
(277,191행, BTCUSDT 5분봉). TRAIN `<2025-09-01` / VAL `2025-09-01~2026-01-01` /
OOS `2026-01-01~2026-04-01` / **HOLDOUT(`>=2026-04-01`) 이번에도 스크립트에서 즉시 truncate,
이후 어디에서도 참조 안 함**(`holdout_touched: false`). 클러스터 dedup(GAP=12봉, 같은 방향
연속발동을 `ret3_z` 최극값 봉 하나로 병합, causal)도 라운드1과 동일 — 단 `touch_giveback_
sustained`는 라벨 계산에 `2×HORIZON`봉의 forward window가 필요해서, 그 경우만 dedup을
`window=2H`(H가 아니라)로 재계산한다(같은 window 크기를 요구하는 조합끼리는 캐시 공유 —
예: H=3의 giveback(window=6)과 H=6의 나머지 3개 방식(window=6)은 **동일한 후보 풀**을 쓴다).
베이스라인(같은 기간·동일개수 무작위 비발동봉)도 window 크기별로 한 번씩만 뽑아 재사용.

## 4가지 HIT_TYPE 정의

바닥후보(`is_down`) 기준, 천장은 좌우대칭 미러. `entry=close[i]`, `atr=atr[i]`(가격단위, 라운드1과
동일 확인됨).

1. **`touch_mfe`**(라운드1 원본, 비교 기준선으로 그대로 유지): `high[i+1:i+H+1].max() >= entry+K·atr`
2. **`close_at_h`**(더 엄격 — 터치 후 되돌림엔 무효, H봉째 **종가**만 판정): `close[i+H] >= entry+K·atr`
3. **`touch_mae_capped`**(먼저 K·atr 찍었어도, 찍기 **전에** 반대방향으로 `K_LOSS_MULT(=2.0)·atr`
   이상 밀렸으면 무효 — `K_LOSS_MULT`는 K와 무관한 고정값, 이 프로젝트 fib_extension_exhaustion의
   MAE-cap 상수를 그대로 차용): touch_bar=최초로 `high>=entry+K·atr`인 봉; `MAE=entry-low[i+1:
   touch_bar+1].min()`; `MAE<=2.0·atr`이면 hit
4. **`touch_giveback_sustained`**(V_REBOUND식 지속성 체크 — FAST_WINDOW=H, FULL_WINDOW=2H,
   giveback상한=0.20, 둘 다 고정): `fast_mult=(close[i+1:i+H+1].max()-entry)/atr>=K` **AND**
   `giveback=(peak-end_price)/(peak-entry)<=0.20`(peak=`high[i+1:i+2H+1].max()`,
   end_price=`close[i+2H]`) — 즉 "빨리 K배 움직였고, 그 후 되돌림이 전체 상승폭의 20% 이하로
   유지됐는가"

## 선택 방법론(라운드1 대비 강화)

- **선택 점수 변경**: 라운드1은 바닥+천장 **합산** TRAIN 리프트로 선택했지만, 이번엔
  **`min(TRAIN 바닥리프트, TRAIN 천장리프트)`**(더 약한 쪽)로 선택 — 한쪽만 강해서 합산이
  부풀려지는 셀이 이기지 못하게 함.
- **게이트 강화**: TRAIN 기준 `n_candidate>=300` **AND** `n_hits>=30`을 바닥·천장 **각각
  독립적으로** 만족해야 함(라운드1은 합산 n>=300 하나였음).
- **안정성가드**: 델타법 상대표준오차(라운드1과 동일 수식, POOLED TRAIN hit/baseline 비율 기준)
  임계값을 라운드1의 경험적 10%에서 이번 과제 지정값 **15%**로 완화, **4개 HIT_TYPE 전부에
  동일하게 적용**(터치기반 방식뿐 아니라 `touch_mae_capped`/`touch_giveback_sustained`도 —
  이 둘은 추가 AND조건 때문에 같은 H,K에서 hit률이 `touch_mfe` 이하로 낮아지므로 오히려 더
  희귀사건 노이즈에 노출되기 쉬움).
- **평탄도 지표**(신규): `flatness_spread = (max-min)/min`(POOLED TRAIN/VAL/OOS 리프트 3개
  기준) — 라운드1이 "1.50~1.56 범위"라고 수작업으로 관찰했던 것을 이번엔 전 그리드에 대해
  프로그램적으로 계산.

## 그리드 스크리닝 — 전체 120칸 결과

4 HIT_TYPE × HORIZON[2,3,6,9,12,18] × K[1.0,1.5,1.75,2.0,2.5] = **120칸**. 전체 그리드는
`short_term_return_z_gridscreen_report.json`의 `grid`에 있음.

### 전역 "기계적 최강" vs "가장 안정" (게이트+안정성가드 통과 셀 중)

| | HIT_TYPE | H | K | TRAIN 리프트(min b/t) | VAL 리프트 | OOS 리프트 | 평탄도 |
|---|---|---:|---:|---:|---:|---:|---:|
| **전역 최강**(기계적) | `touch_giveback_sustained` | 2 | 1.0 | 2.763 | 1.778 | 2.917 | **68.1%**(불안정) |
| **전역 최안정** | `touch_mae_capped` | 6 | 2.0 | 1.412 | 1.480 | 1.494 | **0.90%** |

전역 최강 셀(`touch_giveback_sustained` H=2,K=1.0)은 TRAIN에서 바닥 3.18x/천장 2.76x로
그리드 전체 최고 리프트지만, VAL(1.78x)·OOS(2.92x) 사이 낙차가 커서 **`tradeoff_resolved=False`**
— 라운드1의 H=2 문제와 동일한 패턴(단기·소표본·희귀사건 노이즈)이 hit 정의를 바꿔도 재현됨을
확인.

### 단기 불안정성은 hit 정의를 안 탄다 (각 HIT_TYPE의 자체 "최강" 셀은 전부 H=2)

| HIT_TYPE | 최강 지점(H,K) | TRAIN(min b/t) | VAL | OOS | 평탄도 |
|---|---|---:|---:|---:|---:|
| `touch_mfe` | H=2,K=2.0 | 2.439 | 2.539 | 1.682 | 59.3% |
| `close_at_h` | H=2,K=1.75 | 2.268 | 2.500 | 1.800 | 38.9% |
| `touch_mae_capped` | H=2,K=2.0 | 2.368 | 2.667 | 1.682 | 58.6% |
| `touch_giveback_sustained` | H=2,K=1.0 | 2.763 | 1.778 | 2.917 | **68.1%(최악)** |

4개 HIT_TYPE 모두 자체 최강 지점이 H=2로 몰리고, 전부 평탄도 39~68%로 불안정 — 사용자가 제시한
가설("giveback/MAE-cap이 단기에 자연스럽게 더 안정적일 것")과 반대로, **가장 정교한(AND조건이
많은) `touch_giveback_sustained`가 오히려 단기에서 가장 불안정**했다. 단기 불안정성은 hit 정의의
결함이 아니라 이 신호 자체가 H=2~3에서 표본이 근본적으로 희소해지는 문제로 재확인됨.

### 각 HIT_TYPE의 "가장 안정된" 지점 (게이트+안정성가드 통과, flatness 최소)

| HIT_TYPE | 안정 지점(H,K) | TRAIN(min b/t) | VAL | OOS | 평탄도 |
|---|---|---:|---:|---:|---:|
| `touch_mfe`(라운드1 대응) | H=6,K=2.0 | 1.461 | 1.481 | 1.500 | 2.96% |
| `close_at_h` | H=18,K=1.75 | 1.295 | 1.366 | 1.375 | 1.21% |
| **`touch_mae_capped`(최종권고)** | **H=6,K=2.0** | **1.412** | **1.480** | **1.494** | **0.90%** |
| `touch_giveback_sustained` | H=12,K=1.5 | 1.494 | 1.576 | 1.533 | 7.09% |

`touch_mae_capped`가 리프트(1.41x)와 평탄도(0.90%) **둘 다** 4개 방식 중 최상위 조합 —
`close_at_h`가 평탄도는 근소하게 밀리지 않지만(1.21% vs 0.90%) 리프트가 더 낮고(1.29x) H가
18(1.5시간)까지 밀려 표본이 더 희석됨. `touch_giveback_sustained`는 리프트는 가장 높지만
(1.49x) 평탄도가 7.09%로 한 자릿수 뒤처짐 — window가 2H라 TRAIN/VAL/OOS 간 레짐차이에 더
노출되는 것으로 추정.

## 최종 권고: `touch_mae_capped`, HORIZON=6, K=2.0

| 구분 | 바닥 | 천장 | 합산(pooled) |
|---|---:|---:|---:|
| TRAIN n / hit수 | 1,172 / 384 | 1,212 / 370 | 2,384 / 754 |
| TRAIN 리프트 | 1.567 | 1.412 | 1.487 |
| VAL n / 리프트 | 218 / 1.522 | 234 / 1.446 | 452 / 1.480 |
| OOS n / 리프트 | 190 / 1.936 | 176 / 1.208 | 366 / 1.494 |

TRAIN relSE=4.96%(안정성가드 15% 통과), **선택점수(min bottom/top)=1.412**, 게이트(n>=300,
hits>=30 양쪽) 통과. 바닥·천장 모두 3구간 내내 1.2x 이상 유지(한쪽이 다른쪽을 가리는 라운드1
방식의 리스크 없음). OOS 바닥(1.936x)이 다른 구간보다 다소 높지만, 합산 기준 스프레드는
여전히 그리드 전체 최소(0.90%).

## 피쳐 분석 (`touch_mae_capped`, H=6, K=2.0, TRAIN n=2,372 사용가능 / VAL 452 / OOS 366)

### Point-biserial 상관 (hit vs 22개 Tier0 피쳐, TRAIN)

| 순위 | 피쳐 | 상관 |
|---:|---|---:|
| 1 | `vol_z` | **+0.1715** |
| 2 | `atr_percentile_864` | −0.1543 |
| 3 | `range_width_pct` | −0.1234 |
| 4 | `atr` | −0.1122 |
| 5 | `ndi` | +0.0594 |
| 6 | `upper_wick_ratio` | −0.0562 |

라운드1(H=1/2/3/6, `touch_mfe`)의 top-6와 **완전히 동일한 6개 피쳐, 부호도 전부 동일**(순위
5·6번 `ndi`/`upper_wick_ratio` 순서만 근소하게 뒤바뀜) — hit 정의를 바꿔도 "저변동성 구간의
극단 3봉수익률이 더 의미있는 신호"라는 해석은 그대로 재현된다.

### 순열중요도 (`HistGradientBoostingClassifier`, TRAIN학습→VAL평가, AUC=0.6560, 단일시드)

| 순위 | 피쳐 | importance |
|---:|---|---:|
| 1 | `vol_z` | **+0.02181** |
| 2 | `rsi` | +0.01548 |
| 3 | `bb_pctb` | +0.01506 |
| 4 | `atr_percentile_864` | +0.01245 |
| 5 | `cvd_roll_roc_48` | +0.01221 |
| 6 | `ndi` | +0.01131 |

라운드1(H=2, AUC=0.7399)의 top-6(`bb_pctb`/`ndi`/`vol_z`/`pdi`/`rsi`/`lower_wick_ratio`)와
피쳐 **집합 자체는 거의 겹치지만**(`vol_z`/`rsi`/`bb_pctb`/`ndi` 4개 공통) 순위는 뒤바뀜 —
`bb_pctb`가 라운드1에서는 압도적 1위였는데 이번엔 3위, `vol_z`가 3위→1위. 라운드1 문서도
"단일시드 순열중요도 순위는 호라이즌 바뀌면 흔들린다"고 이미 캐비엇을 달아뒀고, 이번 결과도
그 범위 안(같은 피쳐 풀, 순위만 재배열). AUC(0.656)는 H=6 자체 비교로는 라운드1이 재현체크
과정에서 관측한 `touch_mfe`,H=6의 0.617보다 소폭 높음 — MAE-cap이 라벨을 "덜 시끄럽게" 만든다는
평탄도 개선과 같은 방향.

## 이번 라운드에서 안 한 것 (라운드1과 동일 범위)

- TabPFN 학습 미실행(`HistGradientBoostingClassifier` 단일시드는 피쳐분석용 퀵체크일 뿐).
- 경제성(cost-gate) 미검증 — trailing-stop 등 실제 손익 시뮬레이션 없음.
- HOLDOUT(`>=2026-04-01`) 완전 미접근(`holdout_touched: false`, 코드로 강제).
- `touch_giveback_sustained`가 H=12에서 보인 "리프트는 높은데 평탄도만 처지는" 패턴의 원인
  (window=2H 자체 효과인지, giveback 상한 0.20이 이 신호엔 살짝 안 맞는 값인지)은 더 파고들지
  않음 — 이번 과제 범위는 4개 고정 정의 중 선택이었고, `touch_giveback_sustained`의 파라미터
  자체(FULL_WINDOW 배수, giveback 상한)를 별도로 스윕하는 건 범위 밖.

## 파일 목록

- `scripts/research_btc_short_term_return_z_gridscreen_hittype_20260901.py` — 이번 라운드
  스크립트(HIT_TYPE×HORIZON×K = 4×6×5 = 120칸, per-side 게이트+안정성가드+평탄도).
- `scripts/research_btc_short_term_return_z_gridscreen_20260901.py` — 라운드1 스크립트(보존,
  `touch_mfe` 단일방식, HORIZON×K 45칸).
- `data/labels/btc_5m_evidence_signal_candidates_20260901/short_term_return_z_gridscreen_report.json`
  — 이번 라운드 JSON으로 덮어씀: 120칸 전체 그리드(HIT_TYPE 포함) + `overall_strongest`/
  `overall_most_stable`/`family_leaderboard`(4개 HIT_TYPE별 strongest+most_stable) +
  `recommended` + `tradeoff_resolved`(bool) + 피쳐분석 전체.
- `data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv`
  — 입력 데이터(재사용, 이번에 수정 안 함).
