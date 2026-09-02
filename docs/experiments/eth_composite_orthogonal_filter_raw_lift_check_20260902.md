# 복합(AND-필터) 신호 사전점검 — REJECTED (풀링 아티팩트) (2026-09-02)

상태: 완료 (진단 전용). **판정: 복합도 안 된다. 풀링 창에서 7셀이 두 대조군을 통과했으나
VAL/OOS 분리시 0/7이 양쪽 동시 양수를 못 넘음.**

- 스크립트: `scripts/research_eth_composite_orthogonal_filter_raw_lift_check_20260902.py`
- 산출: `tmp/eth_composite_orthogonal_filter_raw_lift_check_20260902/{scorecard,val_oos_split}.csv`
- 발단: 2026-09-02 사용자 질문 "복합 오실레이터 신호처럼 복합으로 만들 순 없어?"
  (`orthogonal_combo` = 오실레이터 극단 AND 오더플로우 극단이 플래그십인 점에서 나온 합리적 질문)

## 범위 — 왜 필터 2종만인가

- **A-1 Lee-Mykland 제외**: 복합은 이미 그 실행 안에서 테스트됐다 —
  잔여분/교집합 분해의 `D_strz_confirmed_by_lm`이 곧 "STRZ AND LM"이고, 바닥 개선(3.06 vs 2.79)/
  천장 악화(2.39 vs 3.00)로 부호 불일치였다.
- **C-1 라운드넘버 제외**: 오프셋 플라시보에서 메커니즘 부재가 확인됐다. **없는 메커니즘은
  AND로 살릴 수 없다.**
- 남은 **CS/AR(겹침 22~43%)과 VPIN(36.3%)** 만 테스트. 둘 다 배포신호와 실제로 직교하며,
  직교성은 `orthogonal_combo`가 작동하는 바로 그 전제조건이다.

## 이 스크립트가 방어한 함정

어떤 필터든 AND하면 **정밀도는 기계적으로 오른다**(부분집합 선택 + n 축소). 이 저장소는 이 형태로
두 번 당했다(`orthogonal_combo` kept-only AUC 과대평가, `fib_extension_exhaustion` ARM=0.5
exit구조 아티팩트). 따라서 "복합 lift > 베이스 lift"는 아무것도 증명하지 못한다. 사전등록 대조군 2종:

1. **랜덤 부분표집 귀무분포(B=200)** — 베이스 발동에서 n_composite개를 무작위 추출해 lift 재계산.
   복합은 전체 베이스 lift가 아니라 **이 분포의 상위 꼬리**에 있어야 한다.
2. **임계값-매칭 베이스** — "더 적게, 더 좋게" 원하면 가장 싼 방법은 베이스 자기 임계값을 조이는
   것이다. `|ret3_z|`(또는 스윕 깊이) 상위 n개를 뽑아 비교. **베이스를 조이는 것보다 못한 필터는
   정보를 더한 게 아니라 그냥 희소성 손잡이일 뿐이다.**

## 1단계 결과 — 풀링 창에서는 7/60 셀 생존, 계열도 일관돼 보였다

| base | side | filter | H | n | base | **복합** | null%ile | 매칭 |
|---|---|---|---|---|---|---|---|---|
| short_term_return_z | top | vpin_p95 | 1h | 142 | 2.72 | **3.72** | 100.0 | 3.06 |
| liquidity_sweep | bottom | vpin_p95 | 1h | 135 | 3.01 | **4.14** | 100.0 | 4.08 |
| liquidity_sweep | top | vpin_p99 | 1h | 33 | 2.78 | **4.38** | 100.0 | 3.61 |
| short_term_return_z | bottom | vpin_p95 | 4h | 162 | 1.58 | 1.95 | 100.0 | 1.94 |
| liquidity_sweep | bottom | cs_p95 | 4h | 151 | 1.60 | 1.97 | 100.0 | 1.92 |
| short_term_return_z | bottom | vpin_p95 | 8h | 162 | 1.27 | 1.43 | 100.0 | 1.42 |
| short_term_return_z | top | vpin_p95 | 8h | 142 | 1.24 | 1.35 | 98.5 | 1.33 |

**7개 중 6개가 `vpin_p95`** — 2개 베이스 × 양 side × 여러 horizon에 걸친 **계열 일관 패턴**으로,
이 저장소가 "고립 생존자 과적합"과 구분해 실재로 취급해온 모양이었다. 액면대로면 오늘의 발견이다.

⚠️그러나 **60셀을 돌렸으니 5% 기준 우연 기대치가 ~3개**이고, 7개 중 5개는 매칭 대조군 대비
마진이 **0.01~0.06으로 사실상 동률**이었다. 그래서 바로 신뢰하지 않고 분리 검정으로 넘어갔다.

## ⭐2단계 — VAL/OOS 분리시 0/7 (결정타)

임계값-매칭 베이스 대비 증분(양수 = 필터가 더함):

| base | side | filter | H | **VAL** | **OOS** |
|---|---|---|---|---|---|
| short_term_return_z | top | vpin_p95 | 1h | **+1.39** | **−0.45** |
| liquidity_sweep | bottom | vpin_p95 | 1h | **+0.47** | **−1.82** |
| liquidity_sweep | top | vpin_p99 | 1h | **−2.16** | **+4.33** |
| short_term_return_z | bottom | vpin_p95 | 4h | −0.02 | −0.04 |
| liquidity_sweep | bottom | cs_p95 | 4h | −0.02 | +0.14 |
| short_term_return_z | bottom | vpin_p95 | 8h | −0.06 | +0.05 |
| short_term_return_z | top | vpin_p95 | 8h | −0.05 | +0.11 |

**양쪽 동시 양수: 0 / 7.** 전부 창 사이에서 부호가 뒤집힌다.

- 마진이 가장 컸던 셀들이 가장 나쁘다 — `liquidity_sweep top vpin_p99 1h`는 VAL −2.16 / OOS +4.33
  에 n=16/17로, 풀링 4.38x가 **OOS 17건이 전부 만들어낸 것**이었다.
- 최강 셀이던 `short_term_return_z top vpin_p95 1h`(3.72 vs 3.06)는 **OOS에서 매칭 베이스(4.16)가
  복합(3.72)을 이긴다** — 방향이 정확히 반대.
- 나머지는 ±0.02~0.14로 0 주변 노이즈.

이는 이 저장소가 반복 기록한 붕괴 패턴과 동형이다(하이브리드 zigzag 앵커 "VAL 개선/OOS 악화로
결론 불안정", 이중선형 상호작용 "pooled p=0.030 → 2026-only p=0.515").

## 판정

**REJECTED.** 복합(AND-필터)은 이 조합들에선 작동하지 않는다. 풀링 결과는 다중비교 + 창 풀링
아티팩트였다.

## ⭐교훈 두 가지

**1) 임계값-매칭 대조군이 진짜 기준이다.** 생존 7셀을 뺀 대부분의 셀에서 **매칭 베이스가 모든
필터를 이겼다** — 예: 1h `short_term_return_z` 바닥에서 매칭 3.70~4.31 vs 복합 2.66~3.31.
즉 **`short_term_return_z`/`liquidity_sweep`에서 "더 적게, 더 좋은" 발동을 원하면 답은
자기 임계값을 조이는 것이지 직교 필터를 붙이는 게 아니다.** 이건 바로 쓸 수 있는 결론이다.

**2) 랜덤-부분표집 귀무분포 + 임계값 매칭만으로는 부족하다.** 두 대조군을 다 통과하고 계열
일관성까지 보인 7셀이 **VAL/OOS 분리 하나에 전멸**했다. AND-필터류 복합 검정에는 세 번째 관문으로
**창 분리를 반드시** 넣어야 한다(풀링 창에서만 보면 오늘 "vpin_p95 필터 발견"으로 잘못 보고했을 것).

## klines 파생 축 — 최종 상태

| 시도 | 결과 |
|---|---|
| A-1 Lee-Mykland 단독 | REJECTED (중복, 겹침 78~96%) |
| A-3 Corwin-Schultz 단독 | 보류 (대조군 hl_range에 전패) |
| A-2 VPIN 단독 | REJECTED (대조군 volume에 전패) |
| C-1 라운드넘버 단독 | REJECTED (메커니즘 미재현, 플라시보 ≥ 진짜) |
| **복합 (AND-필터)** | **REJECTED (VAL/OOS 분리 0/7)** |

단독 4종 + 복합 1종 = **klines(OHLCV+taker) 파생 탐색 종결**. 남은 경로는 크기축·klines 밖
정보원뿐이다(옵션 스큐 — 아래 참고, raw L2 게이트 09-14/09-30, 온체인).
