# h48qual 숏 반대증거 exit 오버레이 — 소프트 변형 (Candidate C 후속, Odyssey2 #21) — 2026-08-15

상태: **VAL 게이트는 6/9 셀 통과, 그러나 유일한 단일터치 OOS(Q1+Q2)에서 결정적 기각.**

## 배경

`docs/experiments/eth_omega461_evidence_veto_exit_overlay_20260814.md`(Odyssey2 #18, "Candidate
C", 하드 변형)는 h48qual 숏 보유 중 `orthogonal_combo` 증거 신호 발화 시 즉시 강제청산했다가
VAL에서 결정적으로 기각됐다(with_gate PnL 77.31%→47.39%, -29.92%p). 그 문서 자신의 "다음
단계(미착수, 제안만)" 절이 명시적으로 제안한 대안을 이번에 시도했다: **즉시청산 대신, 발화 후
N bar 동안 h48qual exit_head의 확률 임계값(0.95, 정적)을 낮춰 "더 쉽게 나가되 여전히 exit_head
판단을 거치게" 하는 소프트 변형.** 시작 전 `docs/experiments/*20260815*.md`를 grep해 동일 시도가
없음을 확인했다.

## 메커니즘

`scripts/research_eth_omega461_evidence_veto_exit_overlay_20260814.py`(하드 변형)의 신호 구성
(`build_signal`)과 컴포넌트 준비(`prepare_evidence_veto_components`, `_prep_liveatr_only`)
함수를 **그대로 import해 재사용**(재구현 없음). 새로 작성한 것은
`scripts/research_eth_omega461_evidence_veto_exit_overlay_soft_variant_20260815.py`의
`greedy_replay_evidence_veto_soft_exit` 하나뿐이다 — 하드 변형의 강제청산 분기를 다음으로
교체:

- h48qual **숏 보유 중** bar i에 `orthogonal_combo`가 발화하면, N-bar 완화 카운트다운을
  시작(또는 이미 진행 중이면 N으로 리셋 — 누적 안 함).
- 카운트다운이 살아있는 동안(발화 bar 자신 포함) 그 bar의 exit_head 확률 체크는
  `comp["exit_threshold"]`(0.95) 대신 `relax_threshold`를 사용한다. exit_head 체크 자체는
  매 bar 정상적으로 수행되던 것과 동일 — 강제청산 분기가 완전히 사라졌다.
- TP/SL은 항상 우선. 포지션이 (TP/SL 또는 완화된 exit_head로) 종료되거나 새 포지션이
  열리면 카운트다운은 0으로 리셋. 롱 h48qual과 zig075(양방향) 완전 미변경.

## 사전등록 그리드 (실행 전 확정, 코드 상단에 기록)

- `RELAX_N_BARS_GRID = [3, 6, 12]` — 5분봉 기준 15/30/60분, 임의의 outcome-기반 선택이 아닌
  라운드-넘버 배수.
- `RELAX_THRESHOLD_GRID = [0.80, 0.85, 0.90]` — 새 임계값이 아니라
  `scripts/research_eth_omega461_exit_sweep_20260721.py`의 기존 `exit_grid = [0.999, 0.99, 0.97,
  0.95, 0.90, 0.85, 0.80, 0.70]`에 이미 있던 값 3개를 그대로 재사용(새 탐색 없음).
- 3×3 = 9 셀, **VAL만** 먼저 전부 실행.
- VAL 게이트 기준: `eth_omega461_multiwindow_confirmation_gate_20260814.summarize_multiwindow`가
  OOS-confirm 행에 적용하는 것과 동일한 row-level 로직(with_gate PnL ≥ baseline, with_gate
  MDD가 baseline 대비 `mdd_slack_pp` 이내)을 VAL 창에 적용, strict(mdd_slack_pp=0)와
  relaxed(mdd_slack_pp=3) 중 하나라도 통과하면 "clears_val".
- 동점 처리 규칙(실행 전 확정): with_gate PnL 최댓값 우선, 동률이면 `veto_fire_bars`가 가장
  적은 셀(더 단순한 개입) 우선.
- VAL을 통과한 셀이 하나라도 있으면 그 **단 하나만** OOS-Q1+OOS-Q2를 단일터치로 함께 연다(셀별
  재실행 없음). 9개 셀 전부 VAL 실패면 OOS를 열지 않고 여기서 기각.

## G0 자기검증

G0a(다중구간 게이트 모듈로 기준값 재현: val/oos_q1 4개 숫자 모두 일치), G0b(evidence_veto_mask
키가 아예 없는 컴포넌트로 소프트 리플레이를 돌려 순정 `greedy.greedy_replay`와 완전히 동일한
결과가 나오는지 — fire_bars=0, relax_active_bars=0 확인) 둘 다 통과.

## VAL 그리드 결과 (9개 셀 전부, no cherry-pick)

baseline(asymmetric_tabm_liveatr) with_gate = **77.31% / MDD -21.76% / 26 trades**.

| N | threshold | with_gate PnL | with_gate MDD | trades | fire_bars | relax_active_bars | clears_val |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 3 | 0.80 | 59.13% | -21.76% | 28 | 9 | 22 | **False** |
| 3 | 0.85 | 77.31% | -21.76% | 26 | 15 | 39 | True (no-op) |
| 3 | 0.90 | 77.31% | -21.76% | 26 | 15 | 39 | True (no-op) |
| 6 | 0.80 | 59.13% | -21.76% | 28 | 9 | 40 | **False** |
| 6 | 0.85 | 77.31% | -21.76% | 26 | 15 | 67 | True (no-op) |
| 6 | 0.90 | 77.31% | -21.76% | 26 | 15 | 67 | True (no-op) |
| 12 | 0.80 | 59.13% | -21.76% | 28 | 9 | 70 | **False** |
| 12 | 0.85 | 77.31% | -21.76% | 26 | 15 | 115 | True (no-op) |
| 12 | 0.90 | 77.31% | -21.76% | 26 | 15 | 115 | True (no-op) |

**정직하게 밝힐 부분**: threshold=0.85와 0.90은 N에 무관하게 baseline과 완전히 동일한 PnL/MDD/
trades를 낸다 — 즉 15번 발화했지만(`veto_fire_bars=15`), 완화된 임계값(0.85/0.90)이 실제
exit_head 확률을 한 번도 "0.95 미만·0.85(or 0.90) 이상" 구간에서 잡아내지 못해 개입 자체가
사실상 **완전한 no-op**이었다. VAL을 "통과"한 6개 셀은 진짜 개입 효과가 아니라 baseline과
바이트 단위로 동일한 결과다. 반대로 threshold=0.80은 진짜로 개입해(9회 발화 모두 조기청산 유발,
trades 26→28) VAL을 -18.18%p 악화시켜 결정적으로 실패한다 — 하드 변형과 같은 방향의 손상
패턴(발화당 손상이 큼)이 약한 형태로 재현된 것으로 해석된다.

동점 처리 규칙에 따라 선택된 VAL 승자: **N=3, threshold=0.85** (6개 동률 셀 중 그리드 순서상
가장 먼저 나온, 가장 작은 N — 사실상 어느 것을 골라도 VAL에서는 동일).

## OOS-Q1+OOS-Q2 단일터치 결과 (winner만, 재실행 없음)

| 창 | tier | liveatr with_gate | soft_veto(N=3,thr=0.85) with_gate | fire_bars |
|---|---|---:|---:|---:|
| 2025q1 | context | 44.98% | 56.65% (+11.67%p) | 18 |
| 2025q2 | context | 31.49% | 24.02% (-7.47%p) | 18 |
| 2025q3 | context | -18.87% | -25.81% (**-6.94%p, 악화**) | 56 |
| val | val | 77.31% | 77.31% (동일) | 15 |
| **oos_q1** | **oos_confirm** | 67.25% | **65.55% (-1.70%p)** | 2 |
| **oos_q2** | **oos_confirm** | -12.69% | -12.69% (동일, no-op) | 0 |

`summarize_multiwindow` 검증: strict(mdd_slack=0)·relaxed(mdd_slack=3) 둘 다
**REJECTED_SIGN_MISMATCH**. OOS-Q1에서 단 2회 발화가 with_gate PnL을 67.25%→65.55%로 깎아
`pnl_nonworse`를 직접 위반한다(MDD는 동일해 통과지만 PnL 단독으로 이미 탈락). OOS-Q2는 발화
0회로 완전 no-op.

**목표였던 Q3는 이번엔 오히려 악화됐다**(-18.87%→-25.81%, -6.94%p) — 하드 변형이 보였던
Q3 회복(no_gate 145%, with_gate 130% 복구)과 정반대 방향이다. N=3/thr=0.85가 VAL에서는
사실상 no-op이었던 것과 달리, Q3에서는 발화가 56회로 훨씬 잦아 완화된 임계값이 실제로
여러 번 개입했고, 그 개입이 (하드 변형의 "나쁜 재진입 차단" 가설과 달리) 손실을 늘리는
방향으로 작용했다.

## 결론

**기각.** VAL 게이트를 통과한 6개 셀은 전부 baseline과 동일한 결과를 내는 no-op이었을 뿐
진짜 개입이 아니었고, 유일하게 진짜로 개입한 threshold=0.80은 VAL에서 이미 결정적으로
패배했다. 사전등록 동점 규칙으로 뽑힌 VAL "승자"(N=3, thr=0.85)를 단일터치 OOS에 올렸을 때
OOS-Q1이 반전되어 기각된다 — 하드 변형이 "발화당 손상이 너무 크다"로 기각됐다면, 소프트
변형은 "완화 폭이 충분히 커야 실제로 개입하는데(0.80), 충분히 크면 하드 변형과 같은 종류의
손상이 재현되고, 개입하지 않을 만큼 완화 폭이 작으면(0.85/0.90) 애초에 아무 효과가 없다"는
딜레마로 기각된다. `orthogonal_combo` 기반 h48qual 숏 exit 개입은 하드(#18)·소프트(#21) 두
형태 모두 이 시점 기준 유효한 완화안을 찾지 못했다.

## 다음 단계 (미착수, 제안만)

- threshold 그리드를 0.80과 0.85 사이(예: 0.82/0.83)로 더 세분화하는 것은 VAL 26,209 bar
  중 실제 개입 사례가 9~15회에 불과한 표본 크기를 감안하면 오버피팅 위험이 커 권장하지
  않는다.
- `orthogonal_combo`를 exit 오버레이가 아니라 **엔트리 억제**(같은 신호가 숏 신규 진입 자체를
  막는 형태)로 쓰는 방향은 이번 세션 정책상("진입측 주입은 확정된 direction-스킬 부재 위에서
  금지") 재제안 금지 대상이라 이 경로는 막혀 있다.
- 증거 신호 exit 개입 계열은 이걸로 하드(#18)·소프트(#21) 2전 2패로 수렴 — 이 신호원으로
  h48qual exit을 건드리는 추가 변형은 새로운 메커니즘 가설 없이는 재시도 근거가 약하다.

## 준수 확인

신규 학습 없음. 신규 하이퍼파라미터 탐색 없음(`relax_threshold` 그리드는 기존 exit_threshold
스윕 값 재사용, `relax_n_bars` 그리드는 라운드 넘버). 라이브 파일(`trading_bot.py`,
`trading_bot_modules/omega4_6_1_live.py`, `runtime_config.py`, `.env`) 미변경(`git status`로
스크립트·문서 신규 생성만 확인). VAL을 OOS보다 먼저 판정하고 통과한 단 하나의 셀만 단일터치로
OOS를 연 것을 스크립트 레벨에서 강제(#18의 절차 결함 재발 방지). `fresh_forward_bar_by_bar=true`,
`trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`.

## 산출물

- `scripts/research_eth_omega461_evidence_veto_exit_overlay_soft_variant_20260815.py`
- `tmp/causal_regen_20260516/eth_omega461_evidence_veto_exit_overlay_soft_variant_20260815/report.json`
