# ETH h48qual — direction_head 원본(ungated) N=5 시드 대 always-short/long 정식 검증 — 2026-08-12

## 배경 / 질문

사용자가 `quality_head`를 제거하고 `PiecewiseLinearEmbeddings` + `LinearEfficientEnsemble×3`
(`ThreeHeadTabMCorrected`) 백본 작업으로 바로 넘어가자고 제안했다. 백본 자체는 원래
"`quality_head` 게이트 재설계 전까지 보류"였으므로 게이트를 아예 없애는 것도 그 재설계의 한
형태로 볼 수 있어 보류 사유는 해소되지만, `direction_head`를 게이트 없이 단독으로 거래했을 때
always-long/short을 이기는지가 먼저 검증된 적이 없었다.

단일 라이브 번들(102피쳐, 실제 가중치, `diagnose_eth_h48qual_ungated_direction_vs_always_short_20260812.py`,
다른 세션이 작성해둔 스크립트를 이 세션이 실행) 결과: VAL 사실상 동률(+10.80% vs +10.86%), OOS
완패(-4.76% vs +13.53%). 같은 수치가 독립적으로 `docs/experiments/eth_h48qual_quality_head_replacement_research_20260812.md`
(candidate 9)에도 기록돼 있어 교차검증됐다. 그러나 단일 실행(1회)이라 이 레포의 N≥5 다양 시드
표준(Seed-Diversity Ensemble Promotion Gate, [[tabm_hp_low_signal_pattern]])에 못 미쳐 확정
취급할 수 없었다 — 그 문서 자신도 "단일 실행 기준... 확정으로 취급하면 안 된다"고 명시했다.
이 문서는 그 갭을 메운다.

## 방법

- **재학습 없음.** h48orig 5-seed 재현판(FINAL12 피쳐, 실제 h48qual 레시피 그대로 48bar 배리어
  라벨, `verify_eth_h48qual_always_short_baseline_h48orig_20260811.py`와 동일 시드
  `[260620, 481003, 26611, 903174, 155827]`/경로/`TRAIN_CSV`·`EVAL_CSV`)의 이미 저장된 예측
  CSV를 재사용.
- `_to_fixed_decisions`가 읽는 `{prefix}_final_action` 컬럼을 게이트 통과 여부와 무관하게
  `{prefix}_dir_action`(direction_head 원본 픽)으로 바꿔치기해 게이트를 완전히 우회 —
  `diagnose_eth_h48qual_ungated_direction_vs_always_short_20260812.py`의 기법을 5시드 루프에
  결합했다.
- `always_short`/`always_long`은 ungated 활성 bar set 전체를 강제 방향 전환해 구성(모델과 동일
  active set 대조 — 이 프로젝트 표준).
- `fee`/`slip` + `cost_mult=3.0`, `max_hold=0`/`cooldown=0` — 이 서브 프로젝트의 모든
  always-short 대조 스크립트와 동일 컨벤션.
- 스크립트: `scripts/diagnose_eth_h48qual_ungated_direction_h48orig_5seed_vs_always_short_20260812.py`
  (신규 작성, 위 두 기존 스크립트의 minimal-diff 결합).

## 결과

| split | ungated 평균±표준편차 | always_short 평균±표준편차 | 이긴 시드 | paired t | p |
|---|---:|---:|---:|---:|---:|
| VAL | -7.32% ± 11.28%p | +8.52% ± 1.03%p | **0/5** | -3.384 | **0.0277** |
| OOS | +3.58% ± 8.70%p | +22.89% ± 5.15%p | **0/5** | -7.468 | **0.0017** |

시드별 원본(`gated`/`ungated`/`always_short`/`always_long` PnL 전부)은 스크립트 출력 그대로:

```
  seed split  gated_pnl  gated_trades  ungated_pnl  ungated_trades  always_short_pnl  always_long_pnl
260620   VAL  -7.981600            45    -1.758348              42          8.251175       -14.621878
260620   OOS   8.024482            24    -6.556727              33         22.501344       -23.380161
481003   VAL -11.659950            25   -16.527329              41          8.598297       -13.677805
481003   OOS  11.725163            12    -1.240263              32         15.911322       -18.980324
 26611   VAL  -4.977475            42    -3.023854              43          9.197833       -14.106461
 26611   OOS  12.343992            26     7.318783              33         24.630665       -18.510581
903174   VAL -11.370748            39   -21.324127              51          6.929983       -13.429412
903174   OOS   9.213394            17    16.254768              27         30.082100       -20.975816
155827   VAL  -3.185496            30     6.030426              40          9.596572       -15.923436
155827   OOS -14.577219            16     2.108454              30         21.324098       -18.888015
```

## 해석

**5개 시드 전부, 양쪽 스플릿 전부에서 ungated `direction_head`가 always-short에 진다** —
통계적으로 유의(VAL p=0.028, OOS p=0.002). 단일 라이브 번들 실행이 시사했던 것(VAL 동률)보다
사실은 더 나쁘다 — 5시드 평균에서는 VAL도 확실한 패배(-7.32% vs +8.52%)로 나온다. 실제 라이브
번들(단일 인스턴스)이 이 5-시드 분포에서 상대적으로 나은 쪽에 속했을 가능성이 있다는 뜻이고,
이는 정확히 Seed-Diversity Gate가 경계하는 "단일 시드 결과를 신뢰하지 말라"는 패턴이다.

`quality_head` 게이트를 살렸을 때(`gated` 컬럼)와 비교해도 결론은 안 바뀐다 — gated 평균도 두
스플릿 다 always-short에 진다(이미 알려진 사실). 게이트 유무와 무관하게 `direction_head` 계열
전체가 이 VAL/OOS 구간에서 always-short 대비 검증된 방향 스킬을 보이지 못한다.

**`eth_h48qual_quality_head_replacement_research_20260812.md`의 candidate 9가 이제 단일 실행이
아니라 N=5 시드·통계적 유의성으로 확정된다**: `quality_head`를 어떻게 고치거나 무엇으로
대체하든(그 문서의 후보 1~8 — 메타라벨링·trust score·레짐별 threshold·conformal·evidential·
bandit 재구성) 구조적으로 `direction_head`가 이미 고른 것의 부분집합만 고를 수 있다.
`direction_head` 자체가 이 구간에서 always-short 대비 스킬이 없다는 게 이제 이 프로젝트의 표준
잣대로 확정된 이상, 후보 1~8에 대한 추가 엔지니어링 투자 근거는 약하다.

**사용자의 원래 제안(`quality_head` 제거 + 새 백본 진행)에 대한 직접 답: 근거 없음.** 새
임베딩/백본을 짓는다고 `direction_head`가 풀고 있는 문제(`zigzag_action` 예측) 자체가 바뀌지
않으므로, 모델 용량을 늘린다고 이 구간에 없는 신호가 생기지 않는다.

## 캐비어트

- h48orig 5-seed는 FINAL12(12개 피쳐) 재현판이지 실제 라이브 102피쳐 번들이 아니다 — 다만 이
  5-seed 세트야말로 이 프로젝트가 확보한, 실제 h48qual 레시피(48bar) 그대로의 유일한 다중시드
  자원이라 candidate 9를 검증할 수 있는 가장 적절한 도구다.
- 이번 결과가 "이 기간 direction_head에 영원히 신호가 없다"를 증명하진 않는다 — 증명하는 건
  "현재 확보한 FINAL12/102피쳐 조합으로 학습한 `zigzag_action` 분류기는, 이 VAL/OOS 하락장
  구간에서, always-short 대비 검증된 스킬이 없다"는 것. 다른 피쳐·라벨·구간이면 다를 수 있다.

## 다음

이 결과는 사용자가 제안한 경로(`quality_head` 제거 → 백본 진행)를 닫는다. 남는 선택지: (a)
candidate 9의 처방대로 `direction_head` 자체의 스킬 부재를 서브 프로젝트의 선결 질문으로
승격 — 새 피쳐·라벨 탐색이나 다른 구간 재검증, 또는 (b) 이미 진행 중이던 `direction_head` focal
loss 재학습(calibration 개선이지 새 신호 창출은 아니므로 이 결과가 그 자체를 무효화하진 않지만
기대치는 낮춰야 함)을 계속하는 것.
