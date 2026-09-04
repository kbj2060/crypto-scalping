# BTC V자급등락 6트리거 풀 내부 감사 — 배타적(exclusive) 발동 관점 (2026-09-01)

> ⚠️**정정 (같은 날)**: 아래 표의 `local_extreme` 수치(93.4%/89.4%, 5.07~5.45x)는 다른 5개
> 트리거와 다른 방식(인과적이 아닌 중심창 배치계산)으로 측정된 것으로 밝혀졌다 — 불공정 비교였다.
> 인과적으로 재정의하면 23.5%/19.4%(1.2~1.3x)로, 다른 세션의 ETH 라이브 27h 실측(23%)과 일치한다.
> "local_extreme이 union을 압도적으로 견인한다"는 아래 결론은 **철회** — 나머지 5개도 재측정해서
> 공정하게 비교해야 진짜 순위를 알 수 있다(미실행). 상세:
> [[v_rebound_local_extreme_lookahead_contamination_20260901]]. local_extreme을 제외한 나머지
> 5개 트리거(유동성스윕/체결쏠림/15분급변/오실레이터/확장소진)의 EXCLUSIVE 수치는 애초에
> 전부 인과적으로 계산됐으므로 이 정정의 영향을 받지 않는다.

## 배경

DeMarker/칼만편차를 8트리거 후보로 넣었을 때의 원인규명([[btc_v_rebound_feeder_gap_threshold_
screen_20260901]])에서 쓴 "순증후보 성공률" 방법론을, 이번엔 **새 후보가 아니라 이미 baseline에
있는 6개 트리거(liquidity_sweep/taker_delta_z_climax/short_term_return_z/orthogonal_combo/
fib_extension_exhaustion/local_extreme) 각각에 역으로 적용**했다 — "6개 안에도 V자급등락과
안 맞는 신호가 있는가"라는 질문에 답하기 위함. 방법론 전체 정리는
`docs/homer/v_rebound_feeder_signal_protocol.md`.

## 방법론

- 데이터: `data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv`
  (277,191행, 로컬 CPU 전용)
- `label_side()`(V자급등락 giveback 라벨, FAST_BARS=6/FULL_BARS=12/ATR_MULT=1.5/T_SUSTAIN=0.20)
  verbatim 재사용
- 각 트리거의 `bottom_{name}`/`top_{name}` 컬럼(Tier0 CSV에 이미 존재, 6개 전부)을 두 가지
  관점으로 집계:
  - **ALL**: 그 트리거가 발동한 모든 봉(다른 트리거와 동시발동 포함)
  - **EXCLUSIVE**: 6개 중 그 트리거 **하나만** 발동한 봉(동시발동 배제)
- **신규**: 무작위 봉 기준선 — 아무 트리거 조건 없이 전체 27만 봉에 `label_side()`만 적용한
  label==1 비율. 이전 실험들(DeMarker/칼만 스크리닝)은 "6트리거 union 전체 성공률" 대비 비율만
  봤는데, union 자체가 이미 local_extreme에 견인된 합성치라 왜곡 가능 — 이번에 처음 계산.

## 결과 — ALL(전체 발동) 관점

| 트리거 | side | raw_n | POOL(TRAIN+VAL) 성공률 | OOS 성공률 | HOLDOUT 성공률 |
|---|---|---|---|---|---|
| liquidity_sweep | 바닥 | 7,784 | 40.5%(n=5,990) | 41.4%(n=285) | 37.6%(n=482) |
| liquidity_sweep | 천장 | 7,534 | 36.6%(n=5,862) | 39.5%(n=276) | 37.7%(n=472) |
| taker_delta_z_climax | 바닥 | 6,844 | 28.6%(n=5,315) | 27.9%(n=315) | 27.5%(n=429) |
| taker_delta_z_climax | 천장 | 6,570 | 22.9%(n=5,098) | 25.1%(n=263) | 30.1%(n=465) |
| short_term_return_z | 바닥 | 4,353 | 27.5%(n=3,302) | 30.0%(n=220) | 28.3%(n=279) |
| short_term_return_z | 천장 | 4,299 | 20.5%(n=3,250) | 23.5%(n=196) | 21.2%(n=335) |
| orthogonal_combo | 바닥 | 2,237 | 23.2%(n=1,735) | 19.8%(n=121) | 20.7%(n=140) |
| orthogonal_combo | 천장 | 1,558 | 18.9%(n=1,243) | 23.9%(n=46) | 18.4%(n=114) |
| fib_extension_exhaustion | 바닥 | 928 | 48.4%(n=701) | 38.1%(n=42) | 43.4%(n=53) |
| fib_extension_exhaustion | 천장 | 1,009 | 41.1%(n=761) | 62.9%(n=35) | 44.6%(n=65) |
| local_extreme | 바닥 | 14,980 | 93.4%(n=11,286) | 93.1%(n=379) | 88.8%(n=624) |
| local_extreme | 천장 | 15,016 | 89.4%(n=11,375) | 88.9%(n=371) | 90.2%(n=644) |
| **무작위 봉 기준선** | 바닥/천장 | 전체 | **18.08% / 16.40%** | — | — |
| union(any_bottom/top_trigger) | 바닥/천장 | — | 46.5% / 42.2% | — | — |

## 결과 — EXCLUSIVE(배타적 발동, 다른 5개 미동시발동) 관점

| 트리거 | side | excl_n | POOL 성공률 | OOS 성공률 | HOLDOUT 성공률 | 무작위 대비 |
|---|---|---|---|---|---|---|
| liquidity_sweep | 바닥 | 3,995 | 17.0%(n=3,130) | 16.0%(n=156) | 14.9%(n=308) | **0.94x** |
| liquidity_sweep | 천장 | 3,928 | 13.4%(n=3,104) | 17.4%(n=167) | 13.3%(n=278) | **0.82x** |
| taker_delta_z_climax | 바닥 | 2,720 | 16.5%(n=2,155) | 14.4%(n=125) | 17.3%(n=185) | **0.91x** |
| taker_delta_z_climax | 천장 | 2,692 | 11.8%(n=2,135) | 11.1%(n=135) | 14.6%(n=185) | **0.72x** |
| short_term_return_z | 바닥 | 1,769 | 17.4%(n=1,351) | 17.0%(n=88) | 17.5%(n=137) | **0.96x** |
| short_term_return_z | 천장 | 1,850 | 11.6%(n=1,396) | 13.6%(n=103) | 12.7%(n=165) | **0.71x** |
| orthogonal_combo | 바닥 | 499 | 12.2%(n=413) | 7.1%(n=28) | 13.8%(n=29) | **0.67x** |
| orthogonal_combo | 천장 | **0** | 정의불가(발동 0건) | — | — | — |
| fib_extension_exhaustion | 바닥 | 100 | 28.9%(n=77, 표본작음) | 66.7%(n=3) | 33.3%(n=3) | 1.60x(불안정) |
| fib_extension_exhaustion | 천장 | 93 | 25.0%(n=64, 표본작음) | 50.0%(n=6) | 14.3%(n=7) | 1.52x(불안정) |
| local_extreme | 바닥 | 10,464 | 91.7%(n=7,849) | 90.9%(n=253) | 86.4%(n=442) | **5.07x** |
| local_extreme | 천장 | 10,546 | 86.5%(n=7,968) | 86.2%(n=253) | 88.2%(n=432) | **5.27x** |

## 결론

**ALL 관점**으로는 6개 신호 전부 무작위보다 확실히 낫다(1.15x~5.45x) — 각자 자기 고유 목적함수
(원래 그 신호의 연구 트랙)에서 이미 검증됐던 그대로다. **EXCLUSIVE 관점**으로 보면 그림이
완전히 달라진다:
- `local_extreme`(5.07~5.27x)이 압도적 — 롤링윈도우 극값이라는 정의 자체가 V자급등락의
  fast_move 조건과 구조적으로 강하게 맞물려 있어 거의 동어반복에 가깝다.
- `fib_extension_exhaustion`(1.52~1.60x)도 무작위보다 낫지만 배타적 발동 표본이 극히 작아
  (n=64~100) 불안정 — OOS/HOLDOUT 개별 수치(3~7건)는 사실상 참고만 가능.
- **`liquidity_sweep`/`taker_delta_z_climax`/`short_term_return_z`/`orthogonal_combo` 4개는
  배타적으로 발동했을 때 무작위 기준선과 사실상 같거나(0.9x대) 오히려 낮다(0.67~0.82x).**
  `orthogonal_combo` 천장은 6개 중 다른 무언가와 항상 동시발동하고 **단독 발동이 한 번도 없다.**

즉 6트리거 union의 headline 성공률(46.5%/42.2%)과 TabPFN 강한 성능(VAL 0.8351/OOS 0.8202/
HOLDOUT 0.8277)은 사실상 `local_extreme`(과 소표본이지만 `fib_extension_exhaustion`)이 견인하고
있고, 나머지 4개는 **다른 트리거와 같은 봉에서 동시발동할 때만** union에 기여하며 **혼자 발동했을
땐 V자급등락에 대해 거의 정보가 없다.**

이 4개 신호가 "가짜"라는 뜻은 아니다 — 각자의 원래 라벨/목적함수로는 검증된 신호다. 다만
**V자급등락이라는 이 특정 메타신호에 한해서는 기여가 거의 전적으로 local_extreme/fib와의
동시발동에서 나온다**는 뜻이다.

## 사람 판단이 필요한 후속 결정 (미실행)

1. 4개 약한 신호를 union에서 빼고 local_extreme+fib_extension_exhaustion 2트리거만으로 TabPFN
   재확인 — 후보수는 크게 줄지만 평균 품질은 오를 가능성.
2. 모델에 "어느 트리거가 발동시켰는가" 원-핫 피쳐를 추가해서 트리거 출처별 신뢰도를 명시적으로
   학습시킴 — union은 그대로 유지.
3. 현행 유지 — 이미 ETH급 성능을 내고 있어 "왜 잘 되는지"에 대한 설명일 뿐 반드시 고쳐야 할
   문제는 아니라고 볼 수도 있음.

## 산출물
- 로컬 스크래치 스크립트(저장소 미포함): `six_trigger_own_quality_check.py`
- 참조: `docs/homer/v_rebound_feeder_signal_protocol.md`의 "기존 6트리거 풀 내부 감사" 절
