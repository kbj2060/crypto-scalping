# ETH h48qual 최대 보유시간 단축 후보 — 가벼운 확인 결과 (2026-08-16)

상태: **기각. 결과가 그럴듯해 보이지만 비단조(non-monotonic)하고, 이 저장소가 이미 21회+ 실패로
닫아둔 축("time exit" 재튜닝)과 정확히 겹친다. 원래 가설(zig075 기회 회복)도 검증되지 않았다.**
공식 계보 아님(성과 없음). OOS 미개봉.

## 배경

우선순위 스왑이 무효였던 뒤(`docs/experiments/eth_candidate_priority_swap_cheap_gate_20260816.md`),
남은 지렛대로 h48qual의 보유시간 자체를 줄여 슬롯 회전율을 높이는 안을 확인했다. h48qual VAL
보유시간 분포(실제 G0 렛저): p10=79, p25=138, **p50=188**, p75=405, p90=882 bar. exit_head는
h48qual VAL 14건 중 9건을 실제로 처리한다(zig075와 달리 죽지 않음) — max-hold cap은 그 나머지
긴 꼬리(388~1316 bar)만 노린다. 그리드 {150, 250, 400} bar. 스크립트: `scripts/
research_eth_candidate_h48qual_max_hold_cheap_gate_20260816.py`(`greedy_replay_entry_veto` 복사본에
h48qual 전용 강제청산 한 줄 추가, TP/SL 뒤·exit_head 앞).

## 결과

| max_hold cap | with_gate PnL | with_gate MDD | trades | h48qual 거래수(승률) | zig075 거래수 |
|---:|---:|---:|---:|---|---:|
| 기준(없음) | 77.31% | −21.76% | 26 | 14 (—) | 21(no_gate 기준) |
| 150 | 18.85% | −26.45% | 33 | 13 (30.8%) | 26 |
| **250** | **117.30%** | **−15.31%** | 32 | 21 (42.9%) | 20 |
| 400 | 42.02% | −20.65% | 28 | 14 (21.4%) | 24 |

## 판단 — 왜 250이 좋아 보여도 못 믿는가

1. **비단조**: 150→나쁨, 250→아주 좋음, 400→나쁨. 진짜 신호라면 인접 그리드값에서 어느 정도
   완만하게 변해야 하는데, 250 딱 한 점에서만 튀어 오른다 — 이 세션 내내 반복된 "VAL 단일 창
   튜닝 우연"의 전형적 모양이다.
2. **원래 가설이 검증되지 않았다**: 애초 동기는 "h48qual이 빨리 비켜주면 zig075가 기회를 더
   잡는다"였는데, 실제로 250에서 zig075 거래수는 21→20으로 오히려 줄었다. 대신 h48qual
   자신의 거래수가 14→21로 늘고 승률이 42.9%로 튀어서 포트폴리오가 좋아진 것 — **의도한
   메커니즘이 아니라 h48qual 자체의 재진입 사이클링 우연**으로 보인다.
3. **결정적으로, 이건 이미 닫힌 축이다.** `docs/model_contracts/research_line_registry.json`의
   `global_exit_constant_tuning` 항목 scope에 "TP/SL width, quality threshold, **time exit**,
   and second-slot tuning on an unchanged signal"이 명시돼 있고 reason은 "21+ exit rounds ...
   did not survive validation/OOS"다. max-hold cap은 정확히 "time exit" 재튜닝이다 — 이걸로
   벌써 두 번째(zig075@0.80과 함께)로 이 정확한 닫힌 축과 충돌하는 좋아 보이는 VAL 숫자를
   만났다.

## 결론

기각. 재시도하려면 registry의 retest_guidance("새 진입 신호 또는 독립적으로 재현된 실행
불일치")를 충족해야 하는데, 이 테스트는 둘 다 아니다 — 같은 h48qual/zig075 신호 위에서
exit 상수만 바꾼 것이다.

## 아티팩트

- 스크립트: `scripts/research_eth_candidate_h48qual_max_hold_cheap_gate_20260816.py`
- 리포트: `tmp/causal_regen_20260516/eth_candidate_h48qual_max_hold_cheap_gate_20260816/report.json`
