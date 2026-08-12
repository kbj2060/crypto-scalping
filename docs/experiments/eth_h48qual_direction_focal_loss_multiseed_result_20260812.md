# ETH h48qual — direction_head focal loss(gamma=2.0) N=5 시드 결과 (2026-08-12)

## 배경

`quality_head` 대체 리서치(위 여러 문서)가 `direction_head` 자체의 스킬 부재를 N≥5 시드로
확정하기 전, 사용자가 먼저 지시한 트랙: `direction_head`의 클래스별 confidence 비대칭을
post-hoc 보정이 아니라 **학습 단계에서** 직접 겨냥하는 focal loss(Lin et al. 2017, Mukhoti et
al. 2020의 calibration 이점 문헌) 재학습. `--direction-focal-gamma` 옵트인 플래그를 공유
트레이너에 추가(gamma=0이면 기존과 완전 동일 — `git diff` 17줄 전부 추가만).

## 방법

- 학습 소스: 라이브 번들과 동일한 2024-2025 전체구간(2024-gap 버그 조사에서 재구성한
  `omega_clean_regime_only_24_25_inputs_20260629/`) — 트레이너 기본값(2025년만)을 그대로 쓰면
  라이브 번들과 비교 불가능한 모델이 된다는 걸 확인해 명시적으로 override.
- 시드: 이 세션 표준 5개(260620/481003/26611/903174/155827), `gamma=0`(기준선) ×
  `gamma=2.0`(focal, Lin et al. 표준값) = 10회 전체스케일 재학습(`--max-train-rows 0
  --epochs 20`, early stopping patience=8).
- 컴퓨트: dev CPU 3-way 병렬(회당 ~13분 단독, 병렬 경합 시 최대 ~40분) + 일부는 순차. 서버 GPU는
  이번 스윕엔 안 씀(dev로 완주 가능해서 — 별도 `quality_loss_weight=0` 파일럿에만 서버 사용).
- 평가: (a) `report.json`의 저장된 threshold별(0.40~0.80) VAL/OOS pnl/wr 시드평균, (b) 같은
  창(2024-2025 소스 기준 VAL/OOS)의 always-short 기준선, (c) TRAIN·OOS 클래스별(LONG/SHORT)
  reliability(confidence vs 실제정확도)·ECE·confidence 격차, 전부 5시드 평균.

## 결과

### PnL (q050 표준 threshold, 시드평균±표준편차)

| | gamma=0(기존 CE) | gamma=2.0(focal) | always_short(참고) |
|---|---:|---:|---:|
| VAL pnl | **+15.58±8.75** | +11.96±7.38 | +13.78 |
| VAL wr | 45.0% | 44.0% | 44.4% |
| OOS pnl | +3.51±7.33 | +1.06±5.25 | **+20.13** |
| OOS wr | 42.1% | 38.8% | 51.7% |

다른 threshold(0.40~0.60)에서도 gamma=0이 대체로 같거나 소폭 우세, gamma=2가 뚜렷하게 나은
칸은 없음(표준편차가 커서 통계적으로 유의한 차이라 보긴 어렵지만, 방향이 gamma=2에 유리한 쪽으로
나온 적이 없다). **둘 다 OOS에서 always_short(+20.13)에 크게 못 미친다** — 이건 이미
`eth_h48qual_quality_head_replacement_candidates_1_2_3_and_formal_skill_test_20260812.md`가
N≥5로 확정한 것과 일관된다.

### Calibration (TRAIN·OOS, 클래스별, 시드평균)

| 구간 | 클래스 | gamma=0 확신도/정확도/ECE | gamma=2.0 확신도/정확도/ECE |
|---|---|---|---|
| TRAIN | LONG | 0.684 / 0.784 / **ECE 0.100** | 0.559 / 0.766 / **ECE 0.207** |
| TRAIN | SHORT | 0.678 / 0.724 / **ECE 0.046** | 0.549 / 0.729 / **ECE 0.180** |
| OOS | LONG | 0.611 / 0.619 / **ECE 0.071** | 0.510 / 0.611 / **ECE 0.101** |
| OOS | SHORT | 0.630 / 0.609 / **ECE 0.076** | 0.517 / 0.610 / **ECE 0.094** |

confidence 격차(SHORT−LONG)는 TRAIN에서 gamma=0 −0.0056, gamma=2.0 −0.0100 — 격차 자체는 둘 다
이미 작다(이 재학습 전체가 명목상 근거로 삼았던 원 라이브 번들의 +0.0364~0.0485 격차와는 다른
값 — epoch 상한·early stopping 세팅이 원 라이브 번들과 다르기 때문, 상세는 "유의사항" 참고).

## 해석

**Focal loss가 calibration을 오히려 악화시켰다.** 정확도는 거의 그대로인데(TRAIN LONG 78.4%→76.6%,
거의 동일) 확신도만 전반적으로 크게 낮아져서(TRAIN LONG 0.684→0.559, SHORT 0.678→0.549)
ECE가 TRAIN에서 2~4배, OOS에서도 1.2~1.4배 악화됐다. 메커니즘: focal loss의 `(1-p_t)^gamma`
항이 이미 정답을 맞히는 "쉬운" 샘플의 loss 기여를 깎기 때문에, 학습이 그런 샘플의 확신도를
끝까지 밀어올릴 유인이 줄어든다 — 그 결과 정확도는 유지되면서 확신도만 전반적으로 낮게
수렴한다. 애초 가설(LONG의 과소신을 겨냥한 sharpening)과 정반대로, **LONG뿐 아니라 이미 상대적으로
잘 보정돼 있던 SHORT까지 똑같이 더 과소신으로 밀려났다** — 클래스별 비대칭을 줄이려던 목적에
비춰봐도 격차 자체는 별로 안 줄었고(오히려 TRAIN에서 살짝 더 벌어짐, −0.0056→−0.0100), 그냥
전체적으로 확신도만 깎였다.

## 결론

**Focal loss(gamma=2.0) 재학습, 부정 결과.** 목표했던 calibration 개선은커녕 정반대(ECE
악화)가 나왔고, PnL도 always-short는 물론 기존 표준 CE(gamma=0) 대비도 나은 방향이 아니다.
5시드 전부 같은 방향이라 시드 노이즈로 보기 어렵다. 다른 gamma 값(더 작은 값, 예:0.5~1.0)을
시도하면 다를 수 있으나, `quality_head` 대체 리서치가 이미 `direction_head` 자체의 스킬
부재를 N≥5 시드로 확정한 상태라 — calibration만 다루는 이 트랙의 실용적 가치 자체가 낮다는
점을 감안하면 추가 gamma 스윕에 우선순위를 둘 근거는 약하다.

## 유의사항

- 이 재학습(오늘, `--epochs 20` 상한+early stopping)의 확신도 격차(±0.005~0.010)는 원 라이브
  번들(같은 시드 260620 기준 +0.0364~0.0485, `eth_h48qual_direction_confidence_calibration_fullwindow_recheck_20260812.md`)보다
  훨씬 작다 — 원 번들의 정확한 epoch/조기종료 설정을 재현재 못 했을 가능성이 있다(원본 학습
  스크립트 실행 당시 인자가 report.json에 기록 안 돼 있어 정확한 재현이 원천적으로 어렵다는 게
  2024-gap 버그 조사에서 이미 확인됨). 이 문서의 결론(focal loss가 calibration을 악화시킨다)은
  gamma=0/2 두 조건이 **동일한 조건에서 학습됐다는 점**에 근거하므로 이 격차 자체와는 무관하게
  유효하다.
