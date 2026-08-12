# ETH h48qual — 레짐별 완전분리(hard filter) 학습 파일럿 (2026-08-12, 단일시드)

## 배경

사용자 제안: "레짐 별로 TabM을 따로 학습시키는건 어때?" 조사 결과 **h48orig(오늘 40칸 중 2칸으로
스킬 없음이 확정된 그 모델)가 이미 bull/bear/chop 3개를 완전히 별도 파일로 학습 중**이었음이
확인됨(`train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py`의 `_fit_expert_omega4`,
`for idx, expert in enumerate(hard.EXPERT_NAMES): _fit_expert_omega4(...)`), final15/jmlam4(JM
레짐)도 같은 부모 스크립트를 상속해 동일 구조. 다만 각 전문가는 **soft weight**(해당 레짐 확률로
가중, 전체 데이터를 다 봄 — `route_w = _route_probs(...)[:, expert_idx]`)로 학습되고 있어,
**hard filter**(그 레짐으로 argmax된 bar만, 나머지 가중치 0)는 안 해본 조합이었다.

## 방법

`--hard-regime-filter` 플래그를 `train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py`에
추가(기본값 False, 기존 동작 완전 보존 — `git diff` 13줄 추가 2줄 삭제만). True일 때
`route_w`/`exit_w`를 연속확률 대신 `argmax(route_probs) == expert_idx` 0/1 지시함수로 대체.

h48orig 레시피 그대로(FINAL12 피쳐, `quality_label_action`+원본 h48_conservative 48bar,
`--max-train-rows 30000 --epochs 10`), 시드 260620(soft-weight 버전에서 이미 결과가 있는 시드)
하나만 서버 GPU(`scripts/ops/handoff.sh`)로 파일럿 학습. 평가는
`diagnose_eth_h48qual_ungated_direction_h48orig_5seed_vs_always_short_20260812.py`와 동일
방법론(게이트 우회, always_short/long을 모델과 동일 active set에 강제).

## 결과

| | VAL gated | VAL always_short | OOS gated | OOS always_short |
|---|---:|---:|---:|---:|
| **soft-weight (기존 h48orig, seed 260620)** | -7.98% (45건) | +8.25% | +8.02% (24건) | +22.50% |
| **hard-filter (이 파일럿, seed 260620)** | **-17.21%** (37건, wr 24.3%) | +6.20% | **+1.72%** (22건, wr 36.4%) | +28.79% |

(ungated/always_long 등 전체 수치는 스크립트 출력 참고: VAL ungated -11.72%, OOS ungated -2.49%.)

## 해석

**두 스플릿 모두 hard-filter가 soft-weight보다 더 나쁘다** — VAL은 -7.98%→-17.21%(악화), OOS는
+8.02%→+1.72%(악화). always-short 대비 격차로 봐도 더 벌어진다(VAL: -16.2pp→-23.4pp, OOS:
-14.5pp→-27.1pp). 레짐마다 학습 데이터를 하드하게 쪼개는 게 그 레짐에 더 특화된 모델을 만드는 게
아니라, 전문가당 유효 표본만 줄여서(soft weight의 완만한 블렌딩이 사라짐) 더 노이즈한 모델을
만든 것으로 보인다 — 오늘 다른 파일럿(quality_loss_weight=0, `eth_h48qual_quality_loss_weight_zero_pilot_20260812.md`)에서
"정보를 더 제거하면 오히려 나빠진다"는 것과 같은 방향의 패턴.

**단일시드 파일럿이라 확정은 아니다** — 다만 두 스플릿 모두 같은 방향(악화)이고, 이미 소진된
soft-weight 버전(N=40칸)과 오늘의 다른 부정적 파일럿들과 정합적이므로, 우선순위를 높여
N≥5 재현에 투자할 근거는 약하다.

## 결론

**"레짐별로 TabM을 따로 학습"은 두 가지 버전(soft-weight: 이미 라이브 구조 자체, N=40칸으로
확정 부정 / hard-filter: 이 파일럿, 단일시드지만 두 스플릿 다 소프트보다 악화)이 전부 시도됐고
전부 도움이 안 된다.** 이 방향은 닫힌다. `direction_head` 방향 스킬 부재는 레짐 라우팅 방식(soft
든 hard든)의 문제가 아니라는 근거가 하나 더 추가됨 — 오늘 확정된 "레짐과 무관한 구조적 문제"
결론과 일치.
