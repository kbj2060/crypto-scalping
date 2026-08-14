# 레짐별 quality_threshold(h48qual만 비대칭) 부활 시도 — 사이드분리 진단 (2026-08-14)

## 배경

사용자: "레짐별 quality_threshold(h48qual만 비대칭)은 버리기 너무 아까운 전략이야. 이 전략을
연구하고 살려서 쓸 수 없을까?" — `eth_omega461_regime_specific_quality_threshold_h48qual_only_asymmetric_20260814.md`
(Odyssey2 #1 후속)가 기각된 이유(with_gate PnL 1개 지표만 기준선 미달, no_gate는 PnL·MDD 둘 다
개선인 근접 기각)를 다시 파서 살릴 수 있는지 조사한다.

## 1. 렛저 레벨 진단 — with_gate PnL 부진의 정확한 원인

VAL 렛저를 baseline·후보 양쪽 재생성해 컴포넌트×사이드로 분해했다(재학습 없음, 기존 예측 재생).
with_gate(duration-gate) 활성 여부별로 h48qual 기여를 나누면:

| h48qual 사이드 | with_gate 활성(포함) | with_gate 비활성(제외) |
|---|---:|---:|
| SHORT(신규 18건) | 13건, 순수익 **+0.397** | 5건, 순수익 +0.191(제외돼 아까움) |
| LONG(신규 15건) | 14건, 순수익 **+0.005**(사실상 0) | 1건, -0.008 |

**LONG 쪽 14건이 with_gate에 그대로 포함되면서 순이익을 거의 안 내는 채로 포트폴리오를 희석**하고
있었다 — SHORT는 순수 이득인데 임계값을 사이드 구분 없이 낮추다 보니 LONG의 무가치한 노이즈까지
같이 딸려 들어온 것. 이 프로젝트가 반복 확인해온 "h48qual/direction_head는 LONG 방향 스킬이
없다"는 결론과 정확히 일치한다.

## 2. 사이드분리 수정 — SHORT는 유지, LONG만 재조정

SHORT는 원 실험이 찾은 레짐맵(bull=0.30, bear=0.30, chop=0.35) 그대로 두고, LONG 임계값만
전 그리드(0.30~0.80 + 완전차단 1.01)로 독립적으로 스윕(`research_eth_omega461_regime_threshold_
h48qual_side_aware_revival_20260814.py`). G0·G0b(원본 로직·사이드분리 로직 둘 다 baseline을
정확히 재현) 통과 확인 후:

| LONG 임계값 | with_gate PnL/MDD | 게이트(4지표 전부 비악화) |
|---|---:|---|
| 0.30 | +19.69%/-27.89% | 실패 |
| **0.35** | **+80.86%/-17.97%** | 통과(그러나 아래 참고) |
| 0.40 | -7.91%/-24.47% | 실패 |
| 0.45 | +3.87%/-16.49% | 실패 |
| 0.55~1.01 | +62.93%~+98.96% | **전 구간 통과, 단조 개선** |

**0.35는 양쪽(0.30, 0.40)이 모두 실패로 둘러싸인 고립된 스파이크**라 채택하지 않았다 — 이
세션이 반복 확인한 "MDD가 시드 간 소수점까지 동일 = 강건성이 아니라 공유 이벤트" 패턴과 같은
경고 신호(단일 지점 과최적화). 대신 **0.65부터 1.01까지 완전히 동일한 숫자로 안정된 고원**을
신뢰할 수 있는 후보로 골랐다 — h48qual LONG을 사실상 완전히 끄는 지점이며, 임의로 고른 수치가
아니라 이미 검증된 "LONG 스킬 없음" 결론을 그대로 적용한 것.

## 3. VAL 결과 — 오늘 밤 전체 최고

| | baseline no_gate | 후보 no_gate | baseline with_gate | 후보 with_gate |
|---|---:|---:|---:|---:|
| PnL | +36.82% | **+72.31%** | +54.88% | **+98.96%** |
| MDD | -24.34% | **-21.57%** | -31.11% | **-19.73%** |
| 거래수 | 29 | 28 | 22 | 21 |

4개 지표 전부, 넓은 마진으로 개선 — 오늘 밤 시도한 11개 후보 중 VAL 성과가 가장 좋았다.

## 4. OOS-Q1+OOS-Q2 단일터치 — 결정적 기각

다중구간 게이트(`eth_omega461_multiwindow_confirmation_gate_20260814.py`)의 표준 절차대로 VAL에서
고른 구성을 더 조정하지 않고 그대로 OOS-Q1+OOS-Q2를 한 번에 열었다:

| | baseline no_gate | 후보 no_gate | baseline with_gate | 후보 with_gate |
|---|---:|---:|---:|---:|
| OOS-Q1 | +49.32%/-16.20% | **+26.69%/-19.31%**(둘 다 악화) | +44.48%/-15.48% | **+18.27%/-19.80%**(둘 다 악화) |
| OOS-Q2 | +3.13%/-15.00% | **-13.63%/-19.42%**(흑자→적자) | +9.85%/-15.00% | **-0.49%/-19.42%**(흑자→적자) |

**두 창 다, 원기준·완화기준 둘 다 실패**(`REJECTED_BOTH_WINDOWS_FAIL`) — 이전 두 사례(대기압력·
risk-controlled)는 한 창만 반전됐는데, 이번은 양쪽 다 PnL·MDD 전부 악화되는 더 결정적인 반전이다.
2025 Q1/Q2/Q3 참고용 창은 오히려 대체로 개선(맥락 정보로만 취급, 판정 무관).

## 5. 왜 "원칙적인 수정"이 여전히 실패했는가

LONG을 끄는 결정 자체는 원칙적이었다(이미 검증된 결론의 재적용). 하지만 **SHORT의 레짐별 임계값
(0.30/0.30/0.35)은 애초에 원 실험(Odyssey2 #1)의 컴포넌트별 단변량 스윕이 VAL 컴포넌트 PnL을
직접 최대화해서 고른 값**이다 — 오늘 밤 사이드분리 수정은 그 위에 LONG 희석만 제거했을 뿐,
**SHORT 임계값 자체에 이미 배어 있던 VAL 선택편향은 전혀 건드리지 않았다.** "원칙적인 방향의
수정"이 파라미터 하나(LONG 처리)의 문제는 고쳤어도, 다른 파라미터(SHORT 레짐맵)가 이미
VAL에 맞춰 선택된 상태였다는 근본 문제는 그대로 남아있었던 것 — `eth_val_oos_regime_mismatch_
investigation_20260813.md`가 경고한 "사이징→threshold→신규후보까지 같은 저표본 VAL 창을 반복
재사용"의 정확한 사례다.

## 6. 그래도 남는 것 — LONG 스킬 부재의 독립 재확인

이번 시도의 채택 가능한 결과는 0건이지만, **완전히 다른 방법(렛저 사이드 분해)으로 "h48qual LONG
방향 스킬 없음"을 다시 한번 독립적으로 확인**했다는 점은 남는다 — Odyssey(1)이 GBDT/TabM/오토
인코더/TCN/CNN 등 7개 이상의 독립 조합으로, 오늘 밤은 이 렛저 분해로, 총 8번째 독립 확인이다.
이 결론 자체의 신뢰도가 더 강화됐다.

## 결론

**기각.** VAL에서 오늘 밤 최고 성과(4지표 전부 큰 폭 개선, 고립 스파이크가 아닌 넓은 안정 고원에서
선택)를 냈음에도 OOS-Q1+OOS-Q2 둘 다에서 결정적으로 반전됐다 — 원인은 사이드분리 수정 자체가
아니라, 그 위에 얹힌 SHORT 레짐맵이 이미 VAL-선택편향을 안고 있었기 때문으로 추정된다. 대기압력·
risk-controlled에 이어 세 번째로 "VAL 클린 통과 → OOS 반전"이며, 이번이 가장 결정적(양쪽 창 다
실패)이다. Odyssey2의 post-entry/entry-side 축 전체에서 지금까지 살아남은 유일한 개선은 여전히
Odyssey(1)의 exit_head 비대칭 재라벨(섀도우 배포, 미승격)뿐이다.

## 미해결 / 다음 단계

- SHORT 레짐맵 자체를 VAL이 아닌 다른(예: 2025 Q1~Q3 통합) 창에서 독립적으로 재선택하면 다를지는
  미검증 — 다만 표본이 더 작아질 위험(분기당 20건 미만)과 in-sample 문제가 겹쳐 신중해야 한다.
- 이 축(entry-side quality_threshold 재조정)은 이제 원 실험 + 이번 시도까지 2번 결정적으로
  실패했다 — 추가 변형 전에는 근본적으로 다른 접근(예: SHORT 임계값도 VAL 외 창에서 선택)이
  필요하다는 게 이 문서의 판단이다.

## 준수 확인

재학습 없음(기존 예측 재사용, 순수 threshold 재계산+리플레이). `trading_bot.py`/
`trading_bot_modules/omega4_6_1_live.py`/`trading_bot_modules/runtime_config.py`/`.env` 무수정
(`git diff` 0줄). zig075 완전 동결(전 레짐 0.75). `fresh_forward_bar_by_bar=true`,
`trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`. OOS는 정확히 1회만 열었다(첫 실행이 2025q1 컨텍스트 창의
프레임 정렬 버그로 컨텍스트 단계에서 크래시 — OOS-Q1/OOS-Q2 도달 전이라 실제 단일터치 소비 없이
버그만 수정 후 재실행, 재실행이 유일한 진짜 OOS 열람).

발견한 재사용 가능 버그: `research_eth_omega461_regime_specific_quality_threshold_20260813.portfolio_eval`의
호출 관례가 커버리지 갭이 있는 창(2025q1/q3)에서 `IndexError`를 낸다 — VAL/OOS/2025q2는 커버리지
100%라 지금까지 드러난 적 없던 잠재 결함. 이번 스크립트는 자체적으로 프레임을 두 컴포넌트 예측과
사전 교집합한 뒤 넘기는 방식으로 우회했다(원본 함수 자체는 무수정). 스크립트:
`scripts/research_eth_omega461_regime_threshold_h48qual_side_aware_revival_20260814.py`,
`_oos_20260814.py`. 산출물: `tmp/causal_regen_20260516/eth_omega461_regime_threshold_h48qual_side_aware_revival_20260814/{val_report,oos_report}.json`.
