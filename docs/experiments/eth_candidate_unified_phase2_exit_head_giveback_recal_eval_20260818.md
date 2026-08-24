# eth_candidate Phase2 — exit_head giveback_min 재보정 평가 (마무리, 미완성 상태로 종결)

## 배경

[[eth_candidate_unified_single_component_redesign_20260817]] Phase2: quality라벨 B
(same_as_direction, threshold=0.80 확정)의 frozen parent(seed2559205075) 위에 exit_head만
`giveback_min` 0.65→0.25로 재보정해 재학습(`scripts/train_eth_candidate_unified_phase2_exit_head_giveback_recal_20260817.py`,
N=5 시드 목표). 오늘 세션에서 이 5시드 재실행을 시도했으나 WSL2 VM 반복 재시작으로 진행이
막혔고, 사용자가 "5개 다 하지 말고 지금까지 테스트한 걸로 마무리하자"→"지금까지 평가한 걸로
마무리, 다시 시작하지 말고"로 명확히 지시해 **추가 재실행 없이 이 시점에 존재하는 결과만으로
종결**한다.

## 가용 데이터의 한계 (먼저 명시)

- 목표 N=5 시드 중 **3개만 학습 완료**(548794457, 3646016929, 2988156591), 나머지 2개
  (858346535, 2584959503)는 미실행.
- 그 3개 전부 **오늘 발견한 pos_unrealized/pos_mfe/pos_mae 스케일 버그 수정(02:08:20) 이전
  (00:37~01:59)에 학습**돼, 여전히 버그 있는(0.45배 압축) 버전으로 만들어짐
  ([[eth_odyssey4_exit_head_liveatr_barrier_and_label_reaudit_20260818]]).
- 평가 중 seed2988156591의 번들 파일(`true_3head_tabm_bundle.pt`)이 누락돼(직전 배치 재실행이
  이 시드까지 도달하기 전에 중단된 정황) 평가 스크립트가 이 시드에서 실패, **실제 평가 완료는
  2개 시드**(548794457, 3646016929)뿐이다.
- 따라서 아래는 **N=2, 버그 수정 전 버전** 기준 — 이 리포의 N≥5 시드 규율
  ([[tabm_hp_low_signal_pattern]])을 충족 못 하는 예비 관찰이며, Phase2를 "성공" 또는
  "실패"로 확정할 근거가 아니다.

## 평가 방법

`scripts/eval_eth_candidate_unified_phase2_exit_head_20260818.py`. direction/quality는 frozen
parent(seed2559205075, threshold=0.80 — Phase1 확정값)에서 그대로 가져오고(Phase2 번들의
encoder/direction/quality는 parent와 동일한 state_dict이므로 수치상 동일), 각 Phase2 시드의
exit_head만 교체해 bar-by-bar 리플레이(TP/SL 우선, 그다음 exit_head prob≥0.95 — 라이브
EXIT_THRESHOLD와 동일). 사이징은 Phase1과 같은 고정값(notional=0.45, leverage=2.0, 이
컴포넌트엔 아직 학습된 risk sidecar가 없음) — G0/h48qual단독/zig075단독(전부 실제 risk
sidecar 사이징) 3개 기준선과 **직접 비교 불가**, 이 스크립트가 답하려는 건 "같은 quality-B
시스템에서 exit_head 유무가 차이를 만드는가" 하나뿐이다.

## 결과 — exit_head가 한 번도 발동하지 않았다

| 창 | pnl | mdd | trades | wr | exit_head 발동률 |
|---|---:|---:|---:|---:|---:|
| 2025Q1 | +5.70% | −10.94% | 23 | 39.1% | **0.0%** |
| 2025Q2 | +9.82% | −8.82% | 22 | 45.5% | **0.0%** |
| 2025Q3 | +16.13% | −7.74% | 17 | 52.9% | **0.0%** |
| VAL | +0.17% | −17.45% | 30 | 36.7% | **0.0%** |
| OOS-Q1 | +10.40% | −10.58% | 25 | 44.0% | **0.0%** |
| OOS-Q2 | −7.48% | −10.19% | 15 | 26.7% | **0.0%** |

두 시드(548794457, 3646016929) 결과가 6개 창 전부 소수점까지 완전히 동일했다 — 처음엔
스코어링 버그로 의심했으나, **Phase1의 exit_head 없는 버전(같은 parent seed, 같은 threshold,
같은 고정사이징)의 기존 결과와 대조한 결과 완전히 일치**했다(교차검증 완료, 신규 실행 없이
기존 CSV 조회만). 즉 버그가 아니라 실제 결과다: **이 2개 시드의 exit_head는 6개 창
전체에서 단 한 번도 exit_threshold(0.95)를 넘지 못했다** — TP/SL만으로 청산되는 시스템과
완전히 동일하게 행동한다.

## 해석 (미검증, 추가 조사 없이 종결)

h48qual의 기존 exit_head 문제는 "너무 늦게 발동"(MFE의 97.6% 반납 후)이었는데, 이 후보는
반대로 **"아예 발동 안 함"**이라는 다른 실패 양상을 보인다. 가능한 원인(순위 없이 나열,
검증 안 함):
1. `giveback_min=0.25`로 낮추면서 "발동해야 할" 양성 라벨의 성격 자체가 바뀌어 학습된 확률
   분포가 전반적으로 낮아졌을 가능성.
2. 아직 미수정인 pos_unrealized/pos_mfe/pos_mae 0.45배 압축 버그가 학습 신호를 왜곡했을
   가능성(이 2시드 전부 버그 있는 버전).
3. `exit_threshold=0.95`(라이브값 그대로 유지)가 이 새 라벨 레시피가 자연스럽게 내는
   확률값 대비 너무 높은 문턱일 가능성.

이 중 어느 것이 실제 원인인지는 이번 세션에서 추가로 파고들지 않는다 — 사용자 지시에 따라
여기서 종결.

## 결론 — Phase2는 "미완성 상태로 보류", 성공/실패 판정 아님

- N=2(목표 5)뿐이고 그마저 버그 수정 전 버전이라 **giveback_min=0.25 재보정이 도움이 되는지
  판단할 근거가 없다.**
- 확실히 관찰된 건 하나: 이 2개 시드에서 exit_head가 완전히 무력했다(발동률 0%) — 이건
  향후 재개 시 반드시 먼저 확인해야 할 신호다(발동률 자체가 0%인 채로는 threshold를 아무리
  튜닝해도 소용없을 수 있음 — h48qual/zig075 exit_head threshold 스윕에서 이미 확인된
  "발동 안 하면 threshold 조정도 의미 없다"는 패턴과 유사할 가능성).
- **다음 세션 재개 시**: (1) pos_unrealized/mfe/mae 버그 수정본으로 5시드 전체 재학습,
  (2) 이 평가 스크립트를 5시드 전체로 재실행, (3) exit_head 발동률이 여전히 0%면 giveback_min을
  더 낮추거나 exit_threshold를 낮추는 후속 스윕이 필요.

## 준수 확인

- fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
  saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
- live/섀도우 파일 무변경. 재현 스크립트: `scripts/eval_eth_candidate_unified_phase2_exit_head_20260818.py`.
  산출물: 평가가 3번째 시드에서 예외로 중단돼 CSV 저장 전에 끝났다 — 위 표는 stdout 로그에서
  직접 옮긴 것이며 `tmp/causal_regen_20260516/eth_candidate_unified_phase2_eval_20260818/`에는
  파일이 저장되지 않았다.
