# 조문 개정안 — 게이트를 «고정 절대 임계값»에서 «고정 분위 + 고정 창»으로 확장 (2026-09-18)

**상태: 제안. 승인 전에는 이 규약을 쓰는 모델을 «승격(실주문)»할 수 없다.**
섀도우(기록 전용)는 이 조문에 걸리지 않으므로 승인 없이 시작할 수 있다.

## 1. 무엇이 막혀 있나
`Omega Artifact Integrity Promotion Gate` 는 parent artifact 가
`train/validation/oos_predictions_qXXX.csv` 를 포함하고 `qXXX = round(quality_threshold*100)`
이기를 요구한다. 이 규약은 **임계값이 하나의 절대 확률값**이라고 전제한다.

Zeus v4 의 게이트는 절대값이 아니다:
```
점수 s_τ = D_τ[방향] − D_τ[cash]
임계값 t_τ = quantile( {s_u : u 는 직전 1,000 후보}, q=0.85 ),  shift(1) 로 s_τ 자신 제외
발화 ⇔ s_τ ≥ t_τ
```
`t_τ` 는 봉마다 다르므로 `qXXX` 로 적을 수 없다.

## 2. 왜 고정 분위가 필요한가 (고정 절대값이 실제로 해로웠던 증거)
- **h48qual 배포본**: `q=0.50` 이 그 모델의 **99분위 위**였다. 확률 최댓값이 0.485 라
  폴드당 171~200건만 쏘고 부호가 튀었다. 통과율을 맞추면 전 구간 우위였다.
  ⇒ 절대 임계값은 «모델의 확률 스케일이 바뀌면» 조용히 다른 정책이 된다.
- **Zeus v2→v3**: 단일 절대 q 는 확률분포가 시간에 따라 이동하면 **특정 창에 발화를 몰아준다**
  (2026-09-18 실측: 레짐 없는 판이 F3=잃는 창에 발화의 64.7%를 쏟아 +25.39 → +16.54).
  롤링 분위는 그 쏠림을 구조적으로 없앤다.

## 3. 개정 문안 (추가)
> `quality_threshold` 는 다음 둘 중 하나로 선언할 수 있다.
> **(a) 절대형** — 현행 그대로. 파일명 `..._qXXX.csv`, `qXXX = round(threshold*100)`.
> **(b) 분위형** — `{"kind": "rolling_quantile", "q": <0~1>, "window": <후보 수>,
> "score": <점수 함수 이름>, "shift": 1}`. 파일명은
> `..._rq<QQQ>w<WWWW>.csv` (`QQQ = round(q*1000)`, `WWWW = window`).
> 예: q=0.85·window=1000 ⇒ `validation_predictions_rq850w1000.csv`.
>
> 분위형을 쓰는 artifact 는 추가로 다음을 만족해야 한다.
> 1. **인과성**: 임계값 계산에 `shift(1)` 이 적용돼 그 후보 «자신»이 분위에 안 들어간다.
>    워밍업 구간의 대체 규칙(확장창 분위 등)을 report 에 명시한다.
> 2. **상태 영속**: 라이브/섀도우가 직전 `window` 개 점수를 디스크에 보존하고, 재시작 후
>    산출 임계값이 중단 전과 연속임을 확인하는 자체점검을 둔다.
> 3. **발화율 기록**: report 에 폴드별 «신호/일»과 «체결/일»을 둘 다 적는다
>    (절대형의 «통과율» 대응물).
> 4. 나머지 조항(exact-threshold parent prediction artifact, risk sidecar 기록,
>    저장 원장은 diagnostic 전용)은 **그대로 적용**한다 — 파일명 규약만 바뀐다.

## 4. 이 개정이 바꾸지 않는 것
- **Fresh-Forward** 규칙은 그대로다. 분위형은 shift(1) 롤링이라 bar-by-bar 인과성을 만족한다.
- **Seed-Diversity Ensemble Gate** 그대로.
- 저장 trade ledger 를 승격 근거로 쓰는 것은 여전히 금지.

## 5. 적용 대상
`docs/zeus/README.md §3` 의 Zeus Baseline v4. 승인 전까지 v4 는 **섀도우까지만** 진행한다.
