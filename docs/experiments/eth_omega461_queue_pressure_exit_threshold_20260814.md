# ETH Omega4.6.1 대기압력(Queue Pressure) 조건부 exit_head 임계값 (2026-08-14, Odyssey2 #7)

상태: `tested_negative_closed` — **VAL 사전등록 게이트(포트폴리오 레벨, PnL·MDD 둘 다 비악화)를
3개 후보 중 threshold=0.80 하나만 통과했으나, 1회 한정 OOS 확인에서 반전**(포트폴리오 PnL
+93.27%→+59.08%, -34.19pp; MDD는 -15.48%→-15.48%로 사실상 동률). 규율대로 재튜닝 없이 결과만
보고하고 **채택 불가로 종결**한다. GBDT/TCN(Odyssey2 #4/#5)과 달리 exit_head 모델 자체는 전혀
바꾸지 않았고 대기압력 발생 빈도도 7%대(무조건적이지 않음)로 확인됐지만, 그럼에도 소표본
VAL→OOS 반전이 재현됐다.

## 배경

`docs/experiments/eth_omega461_post_entry_literature_scouting_20260814.md`(Odyssey2 #6, 문헌
스카우팅)가 1위로 랭킹한 후보를 구현한다. 그 문서의 핵심 재정식화: 오늘 밤 GBDT/TCN exit_head
실험(Odyssey2 #4/#5, `eth_omega461_gbdt_exit_head_20260813.md`/`eth_omega461_tcn_exit_head_
20260813.md`)이 공통으로 드러낸 "exit를 공격적으로 할수록 컴포넌트 단독 PnL은 악화하는데 공유
슬롯을 자주 비워 포트폴리오 지표는 개선된다"는 패턴은, OR 문헌의 **Gittins index retirement
formulation**(Dhankhar/Mishra/Bodas, arXiv:2405.01157)과 구조적으로 동일하다 — "지금 계속 쥐고
있을 가치 vs 놓아주고 은퇴했을 때 받는 고정가치"를 비교하는 문제이고, GBDT/TCN은 사실상 "무조건적
은퇴 성향"을 과하게 학습한 것에 가깝다.

이 실험은 그 논문의 개념(재학습 없이 근사한 "저비용 실전판")을 그대로 구현한다: **exit_head
모델 자체(TabM 라이브ATR 재라벨, Odyssey2 확정 베이스라인)는 전혀 바꾸지 않고**, h48qual의 고정
`EXIT_THRESHOLD=0.95`(`trading_bot_modules/omega4_6_1_live.py`의 라이브 상수)만 **"이 순간
zig075가 슬롯을 원하는가"**에 따라 조건부로 낮춘다. zig075 자신의 exit 로직·모델·threshold는
전혀 건드리지 않는다 — zig075는 오직 "대기압력 신호의 소스"로만 읽기 전용으로 쓰인다.

## 방법

### 대기압력(Queue Pressure)의 정확한 정의

h48qual이 공유 슬롯(포지션)을 보유 중인 매 bar에서:

- **대기압력 있음** ⟺ 그 bar에서 zig075의 `dir_action != CASH` **AND**
  `quality_for_action >= zig075의 quality_threshold(0.75)`.
- 이 값들은 재계산이 아니라 zig075의 기존 `*_predictions_qXXX.csv`가 이미 갖고 있는
  threshold-무관 원시 컬럼(`{prefix}dir_action`, `{prefix}quality_for_action`,
  `research_eth_omega461_regime_specific_quality_threshold_20260813.build_final_action`이 읽는
  것과 동일한 컬럼)에서 직접 읽는다 — 재학습·재추론 없음.
- 대기압력이 있으면 h48qual의 exit_head 확률 비교에 후보 threshold(0.80/0.85/0.90 그리드)를
  쓰고, 없으면 원래 0.95를 그대로 쓴다.

### 구현 — `greedy_replay` 무수정, 이름 바꾼 복사본만

`scripts/replay_omega4_6_1_greedy_router_20260706.py`의 `greedy_replay`는 **전혀 수정하지
않았다**. 새 스크립트 `scripts/research_eth_omega461_queue_pressure_exit_threshold_20260814.py`에
`greedy_replay_queue_pressure`라는 이름 바꾼 복사본을 만들어, 원본 149줄 중 **딱 한 블록**(exit_head
확률을 비교하는 `if prob >= comp["exit_threshold"]:` 줄)만 조건부 threshold 선택 로직으로
바꿨다(+ 진단 카운터 2개 추가). `PRIORITY`/`SCALE_MAP`/`LEVERAGE_CAP`/`NOTIONAL_CAP`은 로컬
상수로 재정의하지 않고 `greedy.PRIORITY` 등 원본 모듈 참조를 그대로 썼다 — Odyssey2 #5(TCN)의
`greedy_replay_windowed`가 같은 목적으로 쓴 패턴과 동일. `diff -u`로 실제 변경 범위를 직접
확인했다(아래 "준수 확인" 참고). GBDT(#4)가 모델 자체를 duck-typing으로 바꿔치기하고, TCN(#5)이
윈도우 슬라이싱을 위해 복사본을 만든 것과 달리, 이번엔 "청산 판정에 쓰는 threshold 값 선택"만
조건부로 바뀐다 — exit_head 모델(가중치)은 세 변형(원본/TabM 라이브ATR/이번 대기압력 정책) 모두
h48qual에 대해 동일한 TabM 라이브ATR 번들이다.

대기압력 마스크는 h48qual 컴포넌트 딕셔너리의 `queue_pressure_mask` 키에 bar 단위 불리언
배열로 얹었고, `active_comp`이 `"h48qual"`이고 이 키가 존재할 때만 조건부 threshold가 개입한다
— zig075가 슬롯을 잡고 있을 때는 이 블록 자체가 전혀 실행되지 않으므로 zig075는 항상 자기 자신의
`exit_threshold`(0.95)만 쓴다.

### G0 + G0b 자체검증

- **G0(과제 지정 범위, 포트폴리오 레벨)**: 새 하네스로 `baseline_both_original`
  (PnL+36.82%/MDD-24.34%/29건)과 `asymmetric_tabm_liveatr`(현재 확정 베이스라인,
  PnL+46.59%/MDD-21.70%/35건) 둘 다 기존 발표값과 정확히 일치함을 확인 후에만 진행. **PASS.**
- **G0b(이 세션이 추가한 자체 정합성 체크, 과제의 필수 G0 범위는 아님)**: `greedy_replay_queue_pressure`를
  `queue_pressure_threshold=0.95`(comp의 기존 `exit_threshold`와 동일한 축퇴 케이스)로 실행해도
  G0의 `asymmetric_tabm_liveatr` 참조값과 정확히 일치하는지 확인 — 복사본이 의도한 블록 밖에서
  원본과 100% 동일하게 동작함을 `diff` 텍스트 비교가 아니라 **실행 결과 수치 일치**로 증명한다.
  **PASS**(PnL 46.59%/MDD -21.70%/35건, 정확히 일치).
- **대기압력 마스크 교차검증**: 원시 컬럼(`dir_action`/`quality_for_action`) 기반 마스크와,
  이미 계산된 `zig075_dec["side"] != 0`(같은 파일의 이미 threshold-반영된 `final_action`에서
  유도) 기반 마스크를 비교 — 수학적으로 같은 값이어야 하는 두 유도 경로가 실제로 일치하는지
  확인하는 배관 점검. VAL 26,209 bar 전부·OOS 25,538 bar 전부에서 **불일치 0건**.

### VAL-then-단일OOS, 후보 그리드

baseline: TabM 라이브ATR exit_head, h48qual `EXIT_THRESHOLD=0.95` 고정, zig075 원본 —
포트폴리오 VAL PnL+46.59%/MDD-21.70%/35건. 후보: 대기압력 조건부 threshold ∈ {0.80, 0.85,
0.90}(대기압력 없을 때는 항상 0.95 유지). 게이트: **포트폴리오 레벨만**(컴포넌트 레벨 게이트
없음 — 대기압력은 zig075의 상태에 의존하는 개념이라 h48qual 단독(전액가상자본) 리플레이에는
아예 정의되지 않는다). VAL(2025-10-01~12-31)에서 게이트 통과 후보가 있으면(여러 개면 PnL 최고
하나만) OOS(2026-01-01~03-31)를 그 하나로만 딱 한 번.

## 대기압력 발생빈도 진단

과제가 요구한 핵심 진단: **h48qual이 포지션을 보유한 전체 bar 중 대기압력이 실제로 발생한
비율**.

| 측정 대상 | h48qual 보유 bar 수 | 대기압력 bar 수 | 빈도 |
|---|---:|---:|---:|
| 베이스라인 정책(threshold 항상 0.95, 즉 실제 확정 베이스라인의 VAL 보유 패턴) | 3,682 | 275 | **7.47%** |
| VAL 후보 threshold=0.80 (개입 반영된 보유 패턴) | 2,996 | 199 | 6.64% |
| VAL 후보 threshold=0.85 | 3,610 | 253 | 7.01% |
| VAL 후보 threshold=0.90 | 2,752 | 199 | 7.23% |
| OOS 후보 threshold=0.80(단일 확인) | 823 | 65 | 7.90% |

전 구간·전 후보에서 6.6%~7.9% 범위로 **일관되게 드물지도 무조건적이지도 않다** — GBDT/TCN이
"무조건 더 빨리" 청산해 exit_head 발동 비중이 82.5%→91.5%(GBDT)/100%(TCN)까지 치솟았던 것과
질적으로 다르다. 이 정책은 실제로 "필요할 때만"(전체 보유 시간의 약 1/14~1/13) 개입하는 조건부
정책이었음이 진단으로 확인된다 — 아래 결론에서 다루듯, 그럼에도 OOS에서 반전된 것은 GBDT/TCN과
다른 실패 메커니즘을 시사한다.

## 결과 — VAL 후보 스윕 (포트폴리오 레벨, 2025-10-01~12-31)

| threshold | PnL | ΔPnL vs baseline | MDD | ΔMDD vs baseline | 거래수 | 대기압력 빈도 | PnL 게이트 | MDD 게이트 | 종합 게이트 |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| baseline(항상 0.95) | +46.59% | — | -21.70% | — | 35 | (7.47%, 참고) | — | — | — |
| **0.80** | **+52.77%** | **+6.18pp** | **-21.70%** | **±0.00pp(동일)** | 38 | 6.64% | PASS | PASS | **PASS** |
| 0.85 | +57.27% | +10.68pp | -22.40% | -0.70pp(악화) | 38 | 7.01% | PASS | FAIL | FAIL |
| 0.90 | +43.78% | -2.80pp | -21.70% | ±0.00pp(동일) | 34 | 7.23% | FAIL | PASS | FAIL |

3개 후보 중 **threshold=0.80만 PnL·MDD 둘 다 비악화**로 게이트를 통과한다. 흥미롭게도 raw PnL은
0.85가 가장 높지만(+57.27%, baseline 대비 +10.68pp) MDD가 악화(-22.40%, -0.70pp)해 게이트에서
탈락한다 — threshold를 낮출수록 PnL이 단조 증가하는 관계가 아니다(0.80<0.85>0.90의 비단조
패턴). 0.80과 0.90은 공교롭게도 MDD가 baseline과 소수점 단위까지 사실상 동일한데, 이는 이
구간의 MDD를 결정짓는 특정 거래(아마도 zig075 쪽)가 두 threshold에서는 건드려지지 않고 0.85에서만
영향을 받았다는 뜻으로 해석된다 — 직접 추가로 파고들지는 않았다(이번 실험 범위 밖).

**VAL 승자: threshold=0.80**(패스한 유일한 후보이므로 "PnL 최고 하나만" 규칙이 적용될 다자간
비교 자체가 발생하지 않음).

## 결과 — OOS 단일 확인 (threshold=0.80, 2026-01-01~03-31)

⚠️ **반드시 함께 읽을 유보**: h48qual/zig075의 `quality_threshold`(0.50/0.75, baseline과 이
후보가 동일하게 공유)가 `docs/experiments/eth_omega461_oos_selection_bias_scope_and_resolution_
20260813.md`에서 이미 확인된 대로 2026-01-01~02-28 프레임에 OOS-pnl-1순위로 선택된 값이다 —
이 OOS 3개월 중 앞 2개월과 겹친다. baseline과 후보가 이 오염을 동일하게 공유하므로 **상대비교는
유효**하지만 절대 수치를 "깨끗한 미접촉 검증"으로 읽으면 안 된다(`eth_omega461_live_exit_head_
liveatr_relabel_20260813.md` "후속 3"과 동일한 유보).

| | OOS baseline | OOS 후보(0.80) | Δ |
|---|---:|---:|---:|
| PnL | +93.27%(발표값과 정확 일치, cross-check) | **+59.08%** | **-34.19pp(반전)** |
| MDD | -15.48% | -15.48% | 사실상 동일(1e-13% 수준 부동소수점 차이) |
| 거래수 | 24 | 27 | +3 |
| 슬롯 승자(source_component) | zig075:20, h48qual:4 | zig075:23, h48qual:4 | zig075 +3 |
| 대기압력 빈도 | (해당없음, baseline은 조건부 로직 자체가 없음) | 65/823 = 7.90% | — |

`oos_gate_pnl_nonworse=False`, `oos_gate_mdd_nonworse`도 부동소수점 상 근소하게 `False`로
찍히지만(-15.48477681964464 vs -15.48477681964463, 소수점 13번째 자리 차이로 사실상 동률) —
**MDD는 사실상 완전히 같고, 반전을 만든 건 전적으로 PnL**이다. OOS 3개월 동안 h48qual이 슬롯을
차지한 횟수는 baseline·후보 둘 다 4건으로 동일한데도(대기압력 조건부 threshold가 h48qual
자신의 진입 빈도를 바꾸지 않으므로 당연함), zig075가 얻는 거래 수는 20→23건으로 달라졌다 —
h48qual의 청산 타이밍이 미세하게 바뀌면서 공유 슬롯이 zig075에게 풀리는 시점 자체가 달라지는
"슬롯 재순환" 상호작용(GBDT/TCN 문서가 이미 여러 번 관찰한 것과 같은 메커니즘)이 이번에도
확인된다 — 다만 이번엔 그 재순환이 OOS에서 **불리한 방향**으로 작용했다.

**`oos_survives=False`** — VAL에서 확인된 개선이 OOS에서 재현되지 않았다.

## 해석 — 왜 VAL 승리가 OOS에서 반전됐는가 (추정, 이번 실험 범위에서 추가 검증하지 않음)

GBDT/TCN(Odyssey2 #4/#5)의 실패 메커니즘은 "exit_head가 무조건적으로 더 빨리 청산하도록
재학습되어 컴포넌트 단독 economics를 해쳤다"였다. 이번 실험은 그 메커니즘을 구조적으로
차단했다 — exit_head 모델 자체를 바꾸지 않았고, 대기압력 발생 빈도도 6.6~7.9%로 확인돼
"무조건적" 개입이 아니었다. 그럼에도 OOS에서 반전됐다는 사실은, 이번 실패가 GBDT/TCN과 **다른
계열**일 가능성을 시사한다:

- 대기압력 개입은 h48qual 보유 bar의 ~7%에서만 발동하므로, VAL(35→38건, +3건)·OOS(24→27건,
  +3건) 양쪽 다 **개입이 실제로 손댄 거래는 소수**다. 소수 거래의 결과가 VAL에서는 우연히
  좋은 방향, OOS에서는 우연히 나쁜 방향으로 갈렸을 가능성이 구조적 결함(GBDT/TCN처럼)보다
  개연성 있다 — 이 프로젝트가 반복 관찰해 온 "30~40건대 소표본 VAL 승리가 OOS로 일반화되지
  않는다"는 패턴(승격 게이트 문서의 "VAL 단독 승리는 승격 근거 아님" 원칙, 그리고 `docs/
  experiments/eth_omega461_post_entry_literature_scouting_20260814.md`가 인용한 Conformal
  Kelly 논문 자신의 "development window 성공이 진짜 OOS에서 부분적으로만 재현" 경고)과 같은
  계열로 보인다.
- 슬롯 재순환 상호작용(h48qual의 청산 타이밍 변화 → zig075가 얻는 거래 수 변화) 자체는
  GBDT/TCN에서도 확인된 메커니즘이지만, 그 방향(유리/불리)이 이번엔 VAL과 OOS에서 반대로
  나타났다 — 이 상호작용의 부호 자체가 안정적이지 않고 구간에 따라 달라질 수 있다는 뜻일 수
  있다(추정).

## 결론

**채택 불가.** VAL 사전등록 게이트(포트폴리오 PnL·MDD 둘 다 비악화)를 threshold=0.80만
통과했고, 오케스트레이터 지시대로 그 하나만 OOS를 1회 열었다. OOS에서 PnL이 +93.27%→+59.08%로
뚜렷이 반전(-34.19pp)해 `oos_survives=False`다. MDD는 사실상 동률이었으므로 반전의 원인은
전적으로 PnL이다.

Odyssey2 문헌 스카우팅(#6)의 1위 후보(대기압력 후처리 규칙)는 이것으로 **부정 결과 종결**한다.
다만 GBDT/TCN(#4/#5)과 달리 이 실패는 "exit_head를 무조건 더 공격적으로 만들어 컴포넌트
economics를 해쳤다"는 구조적 결함이 아니라(대기압력 자체가 드물지 않되 무조건적이지도 않은
~7%대 조건부 발동임을 진단으로 확인), 이 프로젝트가 여러 번 관찰한 "소표본 VAL 승리가 OOS로
일반화되지 않는다"는 더 일반적인 패턴에 가까워 보인다(추정, 확정 아님) — 이 구분이 향후 유사
post-entry 후처리 규칙(문헌 스카우팅 2위 Risk-Controlled Post-Processing 등)을 설계할 때 참고할
만하다.

## 미해결 / 다음 단계

- 대기압력 "발동 시 threshold를 얼마나 낮출지"만 스윕했다(0.80/0.85/0.90) — "대기압력을 어떻게
  정의할지" 자체(zig075의 quality_for_action 여유폭에 비례한 연속 스케일링, 또는 zig075의
  quality_threshold 자체를 스윕 축에 포함하는 등)는 미탐색. 이번 결과가 이미 부정이므로, 사전등록
  없는 사후 재설계는 시도하지 않았다.
- threshold=0.85가 VAL에서 가장 높은 raw PnL(+57.27%)을 보였지만 MDD 위반으로 게이트에서
  탈락했다 — MDD를 악화시킨 특정 거래가 무엇인지는 이번 실험에서 추가로 파고들지 않았다(범위 밖).
- "왜 VAL과 OOS에서 슬롯 재순환의 부호가 반대였는가"는 추정으로만 제시했고 직접 검증(예: 어느
  특정 거래가 원인인지 원장 대조)하지 않았다.
- 문헌 스카우팅 2위 후보(Risk-Controlled Post-Processing, Joshi/Wang/Hassani/Dobriban,
  arXiv:2605.06479)가 "위험이 높다고 형식적으로 보장되는 좁은 조건에서만 개입"하는 더 엄밀한
  버전으로 이 자리를 재시도할 근거가 남아있다 — 이 문서 범위 밖.
- 채택 가능한 변경 0건, 라이브 파일 미변경.

## 준수 확인

`fresh_forward_bar_by_bar=true`(`greedy_replay_queue_pressure`는 단일 순방향 causal 루프,
bar `i`에서 읽는 대기압력 마스크도 그 bar 자신의 이미 확정된 예측일 뿐 미래 행이 아님).
`trade_ledgers_used_as_input=false`(원장은 출력 전용). `saved_parent_exit_timestamps_used=false`.
`future_rows_used_for_entry=false`. direction_head/quality_head/quality_threshold/encoder 양쪽
컴포넌트 전부 동결(h48qual exit_head **모델**도 TabM 라이브ATR 그대로, threshold만 조건부로
바뀜). zig075는 모델·threshold·exit 로직 전부 무변경(대기압력 신호의 소스로만 읽기 전용 사용).

`git diff` 확인(세션 시작 전/후 모두 0줄): `scripts/replay_omega4_6_1_greedy_router_20260706.py`,
라이브 파일(`trading_bot.py`, `trading_bot_modules/omega4_6_1_live.py`,
`trading_bot_modules/runtime_config.py`, `.env`). `greedy_replay_queue_pressure`는
`greedy_replay`의 이름 바꾼 복사본임을 `diff -u`로 직접 확인 — 실질 변경은 시그니처/독스트링,
진단 카운터 2개, exit_head threshold 선택 블록 한 곳, `PRIORITY`/`SCALE_MAP`/`LEVERAGE_CAP`/
`NOTIONAL_CAP`을 로컬 상수 대신 `greedy.` 모듈 참조로 바꾼 것뿐이며 나머지 로직은 100% 동일.

G0(포트폴리오 레벨, 과제 지정 범위) PASS, G0b(이 세션이 추가한 자체 정합성 체크,
threshold=0.95 축퇴 케이스가 G0 레퍼런스와 정확히 일치하는지 실행으로 증명) PASS. 대기압력
마스크(원시 `dir_action`/`quality_for_action` 컬럼 유도) vs `zig075_dec['side']!=0`(이미
threshold 반영된 `final_action` 유도) 교차검증 — VAL 26,209 bar 전부·OOS 25,538 bar 전부에서
불일치 0건. OOS baseline(93.27%/-15.48%/24건)은 `eth_omega461_live_exit_head_liveatr_
relabel_20260813.md` "후속 3"의 발표값과 정확히 일치(cross-check). **OOS는 VAL 게이트를 통과한
threshold=0.80 하나에 대해서만, 1회 한정으로 열었다** — 결과와 무관하게 재튜닝 후 재실행 없음.

Seed-Diversity Ensemble Promotion Gate: 해당 없음(재학습 모델이 아닌 결정론적 post-processing
threshold 정책이고, 여러 시드를 평균/배깅한 앙상블 승격 주장도 없음). Omega Artifact Integrity
Promotion Gate: 해당 없음(신규 parent 예측 아티팩트를 만들거나 승격 주장하지 않음, 기존 TabM
라이브ATR parent 아티팩트를 그대로 재사용).

## 산출물

- 새 스크립트: `scripts/research_eth_omega461_queue_pressure_exit_threshold_20260814.py` — G0/G0b
  자체검증 + `greedy_replay_queue_pressure`(무수정 원본의 이름바꾼 복사본) + 대기압력 마스크
  구축/교차검증 + VAL 후보 스윕 + 게이트 판정 + OOS 단일 확인.
- report.json: `tmp/causal_regen_20260516/eth_omega461_queue_pressure_exit_threshold_20260814/
  report.json`(G0/G0b + 대기압력 진단 + VAL 3후보 + OOS 전부 포함).
- 거래 원장(diagnostic, 참고용): `tmp/causal_regen_20260516/
  eth_omega461_queue_pressure_exit_threshold_20260814/portfolio_ledger_val_g0b_degenerate_
  thr095.csv`, `portfolio_ledger_val_qp_thr{0.80,0.85,0.90}.csv`,
  `portfolio_ledger_oos_baseline_tabm_liveatr.csv`, `portfolio_ledger_oos_qp_thr0.80.csv`.
- 인용 문서: `docs/experiments/eth_omega461_post_entry_literature_scouting_20260814.md`(이
  실험의 이론적 근거, Odyssey2 #6), `docs/experiments/eth_omega461_gbdt_exit_head_20260813.md`/
  `eth_omega461_tcn_exit_head_20260813.md`(Odyssey2 #4/#5, 대비되는 실패 메커니즘),
  `docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md`(TabM 라이브ATR
  베이스라인 근거 + OOS baseline cross-check 출처), `docs/experiments/
  eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md`(OOS 유보 근거),
  `docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md`(서브 프로젝트 계약).
