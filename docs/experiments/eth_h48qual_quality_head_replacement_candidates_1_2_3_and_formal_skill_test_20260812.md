# ETH h48qual — quality_head 대체 후보 1·2·3 + 후보 9 정식 검증 결과 (2026-08-12)

## 배경

`docs/experiments/eth_h48qual_quality_head_replacement_research_20260812.md`(팀장 리서치)가
권장한 순서 — 재학습 불필요한 3개 테스트(후보 1·2·3)를 병렬로 돌려 스칼라 추출의 마지막 빈틈을
닫는 동시에, "direction_head 자체가 always-short 대비 진짜 스킬을 갖는가"(후보 9)를 이 세션
표준 잣대(N≥5 시드)로 정식 검증 — 를 그대로 실행했다. 전부 재학습 없음, 기존 h48orig(5시드)·
h384 v2(15시드) 재현판 저장 예측 재사용.

## 후보 1: dir_confidence / margin / entropy 직접 순위상관

`scripts/diagnose_eth_h48qual_dirconf_margin_entropy_rank_correlation_20260812.py`. 방법론은
0단계 진단(`quality_for_action` 버전)과 동일 — `dir_action` 기준 게이트 전 진입 시뮬레이션,
진입 시점 세 스칼라(`dir_confidence`=max prob, `margin`=1등−2등 확률차, `entropy`=분포 엔트로피)를
기록해 실현수익률과 spearman 상관.

| 변형 | 스플릿 | dir_confidence 풀링 rho | entropy 풀링 rho | margin 풀링 rho |
|---|---|---:|---:|---:|
| h48orig(5시드) | VAL | **+0.1148 (p=0.043)** | **−0.1345 (p=0.018)** | +0.0297 (p=0.60) |
| h48orig(5시드) | OOS | +0.0787 (p=0.28) | −0.0984 (p=0.18) | +0.0812 (p=0.26) |
| h384 v2(15시드) | VAL | −0.0187 (p=0.57) | +0.0221 (p=0.51) | +0.0041 (p=0.90) |
| h384 v2(15시드) | OOS | −0.0409 (p=0.34) | +0.0343 (p=0.42) | −0.0236 (p=0.58) |

**해석**: h48orig VAL에서 dir_confidence(양)·entropy(음, 같은 방향의 반대 부호)가 보정 전
p<0.05로 나오지만, h384(15시드)는 사실상 무상관(부호도 절반씩 갈림, 7/15·5/15 양수) — 이
세션이 확립한 라벨변형간 일관성 기준(후보 4/6이 걸렸던 것과 동일 패턴)을 통과하지 못한다.
다중비교(2변형×2스플릿×3스칼라=12검정) 보정 전 기준으로도 h48orig VAL 하나만 명목 유의라 약함.
**결론: 신뢰할 만한 신호 없음 — 부정.**

## 후보 2: Trust Score (Jiang, Kim, Guan, Gupta 2018)

`scripts/diagnose_eth_h48qual_trust_score_rank_correlation_20260812.py`. FINAL12 피쳐공간에서
TRAIN(zigzag_action 참라벨) 기준 클래스별 1-최근접이웃 거리로 `trust_score = d(비예측클래스)/d(예측클래스)`
계산(모델 확률 전혀 안 씀, StandardScaler+sklearn NearestNeighbors). TRAIN 표본: CASH 9,243 /
LONG 36,283 / SHORT 33,042.

| 변형 | 스플릿 | 풀링 rho | 양수 시드 비율 |
|---|---|---:|---|
| h48orig | VAL | +0.0203 (p=0.72) | 3/5 |
| h48orig | OOS | +0.0275 (p=0.71) | 3/5 |
| h384 v2 | VAL | −0.0611 (p=0.067, 방향도 반대) | 3/15 |
| h384 v2 | OOS | +0.0007 (p=0.99) | 9/15 |

**해석**: 4변형 전부 사실상 0에 가까운 상관, 시드 부호도 동전던지기 수준. **결론: 신뢰할 만한
신호 없음 — 부정.** (팀장 문서 자신의 "낮음~중간" 기대치가 낮은 쪽으로 확인됨.)

## 후보 3: 레짐별 quality threshold 재보정

`scripts/diagnose_eth_h48qual_regime_conditional_threshold_20260812.py`. `quality_for_action`은
threshold와 무관한 저장값이므로 재학습 없이 `final_action = dir_action if quality_for_action>=threshold
else CASH`를 직접 재구성 + `router_expert`(bull/bear/chop)로 마스킹, 0.40~0.80 9개 threshold ×
3레짐 스윕(시드 전체 풀링 평균).

| 변형 | 스플릿 | 풀링 최적 | bull 최적 | bear 최적 | chop 최적 |
|---|---|---|---|---|---|
| h48orig | VAL | 0.70 | 0.70(동일) | **0.40**(다름) | **0.65**(다름) |
| h48orig | OOS | 0.45 | **0.55**(다름) | **0.40**(다름) | 0.45(동일) |
| h384 v2 | VAL | 0.75 | **0.80**(다름) | 0.75(동일) | **0.70**(다름) |
| h384 v2 | OOS | 0.55 | **0.40**(다름) | **0.40**(다름) | **0.40**(다름) |

**해석 — 표면적으로는 "레짐마다 다르다"지만 신뢰할 수 없다**: 레짐별 "최적값"이 VAL과 OOS
사이에 거의 다 어긋난다(h48orig bear만 두 스플릿 다 0.40으로 일치, 나머지는 전부 불일치 — 특히
h384 bull은 VAL 0.80 vs OOS 0.40으로 정반대). 게다가 threshold가 높아질수록 레짐별 평균
거래수가 시드당 1건 미만까지 떨어지는 칸에서 "최적"이 뽑히는 경우가 많다(예: h48orig VAL chop
0.65는 시드당 평균 0.6건). 이건 이 프로젝트가 이미 겪은 TabM HP 저신호 패턴과 정확히 같은
그리드서치 노이즈 — VAL/OOS 안정성 검증을 거치니 "레짐 무관성을 시사"라는 팀장 문서의 낮은
기대치가 확인됐다. **결론: 신뢰할 만한 레짐별 차이 없음 — 부정.**

## 후보 9 정식 검증: direction_head 자체가 always-short를 이기는가 (N≥5 시드)

`scripts/diagnose_eth_h48qual_multiseed_ungated_direction_vs_always_short_20260812.py`.
2026-08-12 앞서 라이브 번들 1회 실행으로 본 결과(VAL 거의 동률, OOS 대패)를 h48orig(5시드)·
h384(15시드) 재현판 전체로 정식 재현.

| 변형 | 스플릿 | ungated pnl(평균±표준편차) | always_short pnl | ungated 승 시드 | Wilcoxon p(ungated>as) |
|---|---|---:|---:|---|---:|
| h384 v2 | VAL | −3.87±8.82 | +9.78±2.92 | 2/15 | 0.9999 |
| h384 v2 | OOS | −6.13±9.10 | +22.96±3.25 | 0/15 | 1.0000 |
| h48orig | VAL | −7.32±11.28 | +8.51±1.03 | 0/5 | 1.0000 |
| h48orig | OOS | +3.58±8.70 | +22.89±5.15 | 0/5 | 1.0000 |

**총 40칸(4그룹×시드) 중 ungated가 always_short를 이긴 칸: 2칸뿐(전부 h384 VAL).** 4그룹
전부에서 Wilcoxon 단측검정(ungated pnl > always_short pnl)이 압도적으로 기각된다(p≈1.0).
참고로 게이트 있는 현재 방식(`gated`)은 ungated보다는 대체로 낫지만(h384 VAL +8.61 vs −3.87)
그 자체도 always_short를 안정적으로 이기지는 못한다(h48orig VAL gated=−7.84 vs always_short=+8.51).

**결론: direction_head 자체가, 게이트 유무와 무관하게, 이 VAL/OOS 하락장 구간에서 always-short
대비 검증된 방향 스킬을 갖지 못한다 — 단일 실행이 아니라 N≥5 다양시드 기준으로 확정.**

## 종합 결론

네 갈래 전부 부정 결과다. 팀장 문서가 예견한 대로 — 가장 중요한 발견은 개별 후보의 순위가
아니라 후보 9의 정식 확정이다. 메타라벨링/셀렉티브 분류는 구조상 1차 신호(`direction_head`)가
이미 가진 것의 부분집합만 고를 수 있다. 그 1차 신호 자체가 이 구간에서 always-short 대비 스킬을
못 보이는 게 N≥5 시드로 확정됐으므로, `quality_head`-모양(후보 1~8류)의 어떤 정교한 재설계도
구조적으로 만들어낼 수 없는 것을 만들어낼 순 없다. 이 시점부터는 팀장 문서의 후보 9(구조적 한계
인정, `direction_head` 자체의 방향 엣지 존재를 서브프로젝트의 선결 질문으로 재정의)가 개별
가설이 아니라 **이 세션이 실제로 도달한 결론**으로 격상된다.
