# h48qual 숏 반대증거 exit 오버레이 (Candidate C, Odyssey2 #18) — 2026-08-14

상태: **VAL 단계에서 결정적 기각**. OOS 확인 창(OOS-Q1/Q2)은 통과했지만, 그 사실은 승격 근거가
아니다 — 이 서브프로젝트 규율상 VAL 패배는 그 자체로 강한 반대증거이며 OOS 통과가 이를 뒤집지
않는다.

## 배경

`docs/experiments/eth_omega461_evidence_signal_injection_research_20260814.md`(증거 신호 주입
전략 리서치)의 Tier 1 최우선 후보 C를 실전 구현·검증했다. 근거: exit_head liveATR 재라벨은
"회전 가속기"라, 2025-Q3(유일한 지속 상승장)에서 h48qual 숏 거래수가 8→18건(2.25배)으로
폭증해 포트폴리오 no_gate PnL이 -9.73%→-46.26%(4.7배 악화)로 무너진다
(`eth_omega461_exit_head_liveatr_sustained_uptrend_vulnerability_20260814.md`). 이미 섀도우 중인
Odyssey2 #11(dual_momentum 주단위 레짐 감지기, Q3 격차 82.4% 회복)과 같은 문제를 겨냥하되,
**다른 신호원(bar 레벨 오더플로우 소진 증거)·다른 입도**로 접근하는 보완 후보로 설계했다.

## 메커니즘

h48qual이 **숏 포지션을 보유 중**일 때, `orthogonal_combo`(적응형 오실레이터 %R/SlowK가 각자
rolling-864 바닥 10분위 AND 같은 bar의 순공격적매도 물량 z-score ≤ -2, 두 조건 동시)가 발화하면
그 bar에서 즉시 강제청산(`reason="evidence_veto"`)한다. 우선순위는 TP/SL 다음, exit_head와
같은 슬롯 — 하드 배리어는 그대로 존중한다. 롱 포지션과 zig075(양방향)는 완전히 미변경. 공식은
`scripts/analyze_eth_creative_reversal_evidence_signals_20260814.py`/
`scripts/backtest_eth_slowk_williamsr_persistence_confluence_20260814.py`에서 **그대로 재사용**
(새 임계값 탐색 없음 — 이미 두 독립 구간에서 검증된 정의 그대로). `taker_sell_climax` 단독이 아닌
`orthogonal_combo`만 트리거로 채택한 이유: `taker_sell_climax`(delta_z≤-2 단독)는
`orthogonal_combo`의 상위집합이라 둘을 OR로 묶으면 수학적으로 `taker_sell_climax` 단독과
동일해져, 강제청산이라는 무거운 액션에 더 약한 신호(precision 34.4%)가 조용히 섞여 들어간다 —
가장 강하고 희귀하고 보수적인 신호(precision 43.9%, 2025년 전체 활성화율 0.5~0.7%) 하나만 쓰는
게 원칙에 맞다.

## 방법

Odyssey2 #11의 검증된 템플릿(`scripts/research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.py`)을
그대로 재사용해 `scripts/research_eth_omega461_evidence_veto_exit_overlay_20260814.py` 작성.
G0a(다중구간 게이트 기준값 재현)·G0b(veto 강제비활성 항등성 검증) 둘 다 통과 확인 후 6개 창
전부(2025 Q1~Q3 참고, VAL, OOS-Q1/Q2 단일터치) 실행. `greedy_replay`의 렌네임 사본에 강제청산
분기만 추가(diff 최소화, #11과 동일 원칙). 새 학습·새 하이퍼파라미터 탐색 없음.

## 결과 — 신호 활성화율은 예상대로 희귀(0.5~0.7%), 그런데 발화당 손상이 매우 크다

| 창 | with_gate PnL (원본/liveatr/증거veto) | veto 발화 bar 수 |
|---|---|---:|
| 2025-Q1 | 28.54% / 44.98% / **26.15%** | 3 |
| 2025-Q2 | 39.99% / 31.49% / **18.21%** | 7 |
| 2025-Q3 | -9.73% / -18.87% / **-6.98%** | 11 |
| **VAL** | 54.88% / 77.31% / **47.39%** | 6 |
| OOS-Q1 | 44.48% / 67.25% / **69.61%** | 1 |
| OOS-Q2 | 9.85% / -12.69% / **-12.69%**(동일) | 0 |

**VAL이 결정적으로 패배한다**: liveatr 77.31% → 증거veto 47.39%(-29.92%p), no_gate도
46.59%→26.35%(-20.24%p) 악화. 다중구간 게이트 모듈 자체의 row-level 체크(`with_gate_pnl_
nonworse`)가 VAL에 대해 **False**를 직접 반환한다. veto는 겨우 6번만 발화했는데(전체 26,209
bar 중), 그 6번의 강제청산이 VAL 전체 PnL을 30%p 가까이 깎았다 — precision 43.9%인 신호를
"강제청산"이라는 무거운 액션에 쓰면, 그 43.9%가 맞는 경우에도 진짜 반전이 오기 전 추가 역행이
있고(스코어카드의 알려진 유보), 56%가 틀리는 경우엔 잘 가던 숏을 조기 청산한 뒤 슬롯이 풀려
재진입 → 손실 거래로 이어지는 이중 손상이 발생한다. Q1·Q2도 같은 패턴(liveatr 대비 각각
-18.83%p, -13.28%p) — **"숏이 실제로 잘 먹히던 구간"에서 이 메커니즘이 반복적으로 손해를
낸다.**

**목표했던 Q3는 오히려 원본을 능가한다**: no_gate 회복률 145%, with_gate 회복률 130% — 단순
"liveatr로 인한 악화분 복구"를 넘어 재라벨 이전 원본보다도 낫다. 이 결과 자체는 증거 신호가
Q3의 나쁜 숏 재진입을 실제로 걸러낸다는 메커니즘 가설을 지지한다.

**OOS-Q1/Q2는 통과**(각각 소폭 개선, 무변화)했지만, VAL이 이미 패배한 시점에서 이 통과는
승격 근거가 못 된다. **이 실행은 절차적 결함이 하나 있다**: 다중구간 게이트 모듈의 설계 원칙상
"VAL 게이트는 호출자가 OOS를 열기 전에 먼저 판정해야 한다"인데, 이 스크립트는 VAL 판정 없이
6개 창을 한 번에 다 돌렸다 — VAL이 이미 결정적으로 나쁘므로 최종 결론에는 영향이 없지만
(어차피 기각), 앞으로 이 패턴의 후보는 **VAL을 먼저 판정하고 통과할 때만 OOS를 여는 코드
구조**로 고쳐야 한다(#7/#8/#11이 지킨 절차를 이번엔 스크립트 레벨에서 안 지켰다는 뜻 — 정직하게
기록).

## 결론

**기각.** 하드 강제청산(즉시 exit) 메커니즘은 목표했던 Q3 취약점은 실제로 완화하지만
(145%/130% 회복), 숏이 실제로 유효한 VAL/Q1/Q2 구간에서 희귀한 발화당 손상이 너무 커서
순효과가 크게 마이너스다. `orthogonal_combo`의 precision(43.9%)이 하드 액션(포지션 강제종료)의
비용을 정당화하기엔 부족하다는 뜻으로 해석된다.

## 다음 단계 (미착수, 제안만)

- **소프트 변형**(연구 문서가 제시한 대안 (b)): 즉시청산 대신 발화 후 N-bar 동안 exit_threshold만
  완화 — Odyssey2 #7(대기압력) 개입 형태와 유사. 프리미엄 하드액션 대신 "더 쉽게 나가되 여전히
  exit_head 판단을 거치게" 하면 발화당 손상이 줄 가능성. 단, 이것도 VAL-우선 게이트로 판정해야
  한다.
- Odyssey2 #11(dual_momentum)과의 결합(둘 다 살아있는 후보로서 AND/OR 조합) — #11은 이미 VAL
  비악화를 통과했으므로, 결합 시 이번 후보의 VAL 손상이 어떻게 상호작용하는지 별도 확인 필요.
- 향후 이 클래스 후보는 스크립트 자체에 VAL 하드게이트를 코드화해 OOS를 조건부로만 열 것 —
  이번 실행의 절차적 결함을 표준 관행으로 고정.

## 준수 확인

신규 학습 없음. 신규 하이퍼파라미터 탐색 없음(기존 검증된 신호 정의 그대로). 라이브 파일
(`trading_bot.py`, `trading_bot_modules/omega4_6_1_live.py`, `runtime_config.py`, `.env`)
미변경(`git status`로 스크립트 신규 생성만 확인). `fresh_forward_bar_by_bar=true`,
`trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`.

## 산출물

- `scripts/research_eth_omega461_evidence_veto_exit_overlay_20260814.py`
- `tmp/causal_regen_20260516/eth_omega461_evidence_veto_exit_overlay_20260814/report.json`
