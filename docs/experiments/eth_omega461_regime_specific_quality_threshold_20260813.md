# ETH Omega4.6.1 레짐별 quality_threshold 재보정 (2026-08-13, Odyssey2 #1)

## 배경

Odyssey(1) 미해결 이슈 4(리서치 문서 3-4번 항목) — 레짐별(bull/bear/chop) `quality_threshold`
재보정은 2026-08-11부터 한 번도 실제로 시도되지 않았다. 전역 threshold 스윕(0.40~0.80)은
0.55 이후 악화만 확인했지만, 레짐별로 나누면 다를 가능성이 남아있었다.

## 방법

`dir_action`/`quality_for_action`/`router_expert`는 threshold-무관 raw 컬럼으로 이미 저장돼
있어(`train_omega1_regime3_routed_expert_direction_quality_20260602._prediction_output`)
재학습이 전혀 필요 없다 — 기존 예측 CSV 하나를 로드해 `final_action`을 레짐별 threshold map으로
직접 재계산하고, 나머지 파이프라인(ATR TP/SL, 사이드카 사이징, exit_head 시뮬레이션)은
`research_eth_omega461_exit_sweep_20260721`/`replay_omega4_6_1_greedy_router_20260706`을
그대로 재사용했다.

**설계**: 3개 레짐 threshold를 한 번에 조인트 최적화하면 그리드가 11³=1331칸으로 폭발하고
다중비교 문제도 커지므로, (1) 레짐 하나씩 단변량 스윕(나머지 2개는 전역 baseline 고정,
그리드 {0.30~0.80, step 0.05}, VAL 컴포넌트 레벨 pnl 최대화), (2) 각 레짐의 VAL-최적값을
조합한 조인트 맵을 컴포넌트+포트폴리오 레벨로 재평가하는 2단계로 제한했다.

## 결과 — 혼재, 기준선 통과 실패

G0 자체검증 통과(baseline 재현 정확 일치). 조인트 맵: `h48qual={bull:0.30, bear:0.30,
chop:0.35}`, `zig075={bull:0.75(불변), bear:0.30, chop:0.35}`.

| | baseline | 조인트 레짐별 |
|---|---:|---:|
| h48qual 컴포넌트 PnL/MDD | +5.45% / -11.62% | **+45.93% / -11.00%**(둘 다 개선) |
| zig075 컴포넌트 PnL/MDD | +40.31% / -13.07% | +31.65% / -13.38%(둘 다 악화) |
| 포트폴리오 no_gate PnL/MDD | +36.82% / -24.34% | **+72.03%** / **-28.46%**(PnL 개선, MDD 악화) |
| 포트폴리오 with_gate PnL/MDD | +54.88% / -31.11% | 34.76% / **-25.02%**(PnL 악화, MDD 개선) |

**사전등록 기준(pnl·mdd 둘 다 비악화, no_gate·with_gate 둘 다)을 통과 못함** — no_gate는
PnL은 크게 개선하지만 MDD가 악화되고, with_gate는 정반대(MDD는 개선, PnL은 악화)라 어느
쪽으로도 깔끔한 승리가 아니다. **OOS는 열지 않았다.**

## 해석

h48qual은 임계값을 크게 낮춰서(0.50→0.30/0.30/0.35) 더 많은 트레이드를 들여보내는 쪽으로
갔고 zig075는 bear/chop만 낮췄다 — 결과적으로 게이트를 더 느슨하게 만든 효과다. Odyssey(1)의
핵심 결론(`direction_head`가 always-short 대비 검증된 방향 스킬 없음)을 감안하면, 게이트를
느슨하게 하는 건 "스킬 없는 direction_head의 더 넓은 부분집합을 통과시키는 것"이라 결과가
깔끔하지 않고 지표별로 엇갈리는 게 오히려 예상과 일치한다 — h48qual quality head relabel
(Odyssey1 결함감사)이 겪은 것과 같은 계열의 트레이드오프다.

## 미해결 / 다음 단계

- 단변량 스윕 후 조합하는 방식이라 레짐간 상호작용을 못 잡았을 가능성 — 진짜 조인트 최적화는
  다중비교 문제를 감수하고서라도 시도해볼 여지가 있으나, 이번엔 안 함(연구자 자유도 남용 우려).
- no_gate/with_gate 중 하나만 기준으로 삼는 절충안(예: with_gate만 기준)은 사전등록 안 된
  사후 기준 변경이라 시도하지 않음.
- 채택 가능한 변경 0건, 라이브 파일 미변경.

## 준수 확인

- `git diff` 기준 라이브 파일 무변경. 재학습 없음, VAL-then-단일OOS 규율(이번엔 VAL 탈락으로
  OOS 미실행). 스크립트:
  `scripts/research_eth_omega461_regime_specific_quality_threshold_20260813.py`. 산출물:
  `tmp/causal_regen_20260516/eth_omega461_regime_specific_quality_threshold_20260813/`.
