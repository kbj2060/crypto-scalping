# ETH Omega4.6.1 zig075 exit_threshold 재보정 (2026-08-14, Odyssey2 우선순위 큐 #6)

상태: `tested_negative_closed` — **VAL 전체 그리드(8점)에서 로버스트하고 진짜(baseline과
동률이 아닌) 개선을 보인 후보가 0개**였다. 실질적 변화를 보인 유일한 지점(0.80)은 원기준
게이트를 통과했으나 이웃 그리드(0.85)가 실패해 고립 스파이크로 기각됐고, 나머지 통과 지점들은
전부 baseline과 바이트 단위로 동일한 디제너레이트 동률이었다. **OOS는 열지 않았다**
(`oos_opened=false`, `val_winner=None`).

## 배경

`docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md` 우선순위 큐 6번 항목:
"zig075 exit_head 개선 — 같은 live-ATR relabel 레시피는 이미 악화로 닫힘(Odyssey1). 다른 접근
(개별 재라벨 파라미터, 별도 exit_threshold 등) 미탐색." 오늘 밤 h48qual 쪽에서 진행된 exit_head
후처리 실험(GBDT/TCN 전면교체, 대기압력, risk-controlled, SCRC, 레짐인지형 가드)은 전부
`EXIT_THRESHOLD=0.95`(`research_eth_omega461_exit_sweep_20260721.BASELINE_EXIT_THRESHOLD`)를
h48qual·zig075 양쪽에 동일하게 고정한 채 h48qual의 exit_head **모델**만 바꿔봤다 — "zig075는
모델을 안 건드리고 그 threshold 숫자 자체만" 바꾸는 축은 이번이 처음이다. 재학습이 전혀
필요 없다: `replay_omega4_6_1_greedy_router_20260706.greedy_replay`는 이미 컴포넌트별
`exit_threshold`를 각자의 준비된 딕셔너리에서 읽는다(`if prob >= comp["exit_threshold"]:`) —
설정값만 바꾸면 되는 구조다.

## 가설 (양방향 다 열어둠, 방향 사전결정 없음)

오늘 밤 #10 조사(`eth_omega461_exit_head_liveatr_sustained_uptrend_vulnerability_20260814.md`)가
찾은 메커니즘: threshold를 낮추거나 liveATR로 재라벨하면 회전이 빨라진다. h48qual처럼 방향
스킬이 검증되지 않은 컴포넌트에서는 빠른 회전(나쁜 트레이드를 빨리 끊고 공유 슬롯을 비움)이
유리했다. zig075는 이 프로젝트에서 유일하게 검증된 방향별 엣지(숏 쪽 하락장 베타)를 가진
컴포넌트라 정반대일 수 있다:

- **(a) 회전을 늦추는 방향(threshold를 0.95보다 높임)**: 더 확신할 때만 나가므로 zig075의 좋은
  숏 트레이드를 끝까지 들고 갈 가능성.
- **(b) 회전을 빠르게 하는 방향(threshold를 0.95보다 낮춤)**: h48qual에서 통했던 것과 같은
  슬롯 회전 메커니즘이 zig075 쪽에서도 통할 가능성.

둘 다 같은 대칭 그리드로 테스트했고, 그리드 자체는 어느 방향도 편향하지 않는다.

## 방법

### 고정 vs 스윕

- **h48qual**: 오늘 밤 확정된 `asymmetric_tabm_liveatr` 섀도우 baseline 그대로 — liveATR
  재라벨 exit_head 번들(`research_eth_omega461_exit_head_portfolio_asymmetric_20260813.
  NEW_H48QUAL_BUNDLE`), `exit_threshold=0.95` 고정. 이 스크립트의 모든 실행에서 단 한 번도
  스윕되지 않음.
- **zig075**: direction_head/quality_head/encoder/exit_head **가중치**는 원본 그대로(라이브와
  동일, 재라벨·재학습 없음). **오직 `exit_threshold` 숫자만** {0.80, 0.85, 0.90, 0.92, 0.95
  (기준점 자기재현), 0.97, 0.98, 0.99} 그리드로 스윕 — 0.95 기준점 좌우로 좁은 간격(0.92/0.97/
  0.98/0.99)과 넓은 간격(0.80/0.85/0.90)을 모두 촘촘히 배치, 방향을 사전에 정하지 않았다.

### 재사용 원칙 (기존 함수/모듈 무수정)

`research_eth_omega461_exit_sweep_20260721.py`(`prep_component`/`replay_exit_variant`/
`run_grid`), `replay_omega4_6_1_greedy_router_20260706.py`(`greedy_replay`/`prepare_component`),
`research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py`(`_component_cfg`/
`_prepare_component_val` — 비대칭 baseline 빌더), `research_eth_omega461_live_sltp_mfe_width_
20260813.py`(`_duration_gated`), `research_eth_omega461_risk_controlled_exit_fallback_20260814.py`
(`_guardrail_ok`, 완화기준 가드레일 공식 그대로 재사용), `eth_omega461_multiwindow_confirmation_
gate_20260814.py`(`load_all_windows`/`run_portfolio_variant`/`summarize_multiwindow`/`_close`/
`ALL_WINDOWS`) 전부 **임포트 후 읽기만** — 전혀 수정하지 않았다. GBDT(#4)/TCN(#5)/대기압력(#7)/
risk-controlled(#8)와 달리 **이름 바꾼 복사본조차 필요 없었다** — `exit_threshold`가 이미
컴포넌트별 1급 설정값이라 개입 전체가 설정 딕셔너리 오버라이드 하나로 끝난다.

### 게이트 기준 (오늘 밤 #7/#8/#9/#14와 동일한 이중 기준 어휘)

이번 개입 대상은 h48qual이 아니라 zig075이므로, "컴포넌트"는 zig075 자신의 전액가상자본
단독 리플레이(`sweep.prep_component`+`sweep.replay_exit_variant`, `research_eth_omega461_
exit_sweep_20260721.py` 자신의 `main()` Experiment A와 동일한 조합)를 가리킨다.

- **원기준**: zig075-컴포넌트 NO_GATE PnL·MDD **AND** 포트폴리오 NO_GATE PnL·MDD 전부 baseline
  대비 비악화.
- **완화기준**(`docs/experiments/eth_omega461_relaxed_gate_rescoring_20260814.md`): 포트폴리오
  WITH_GATE PnL이 baseline보다 **엄격히 개선** **AND** WITH_GATE MDD가 3%p 이내 악화 **AND**
  컴포넌트 가드레일(`rc_mod._guardrail_ok` 무수정 재사용 — zig075-컴포넌트 PnL 부호반전
  또는 baseline 대비 50% 초과 상대악화 금지).

그리드 포인트는 **둘 중 하나만 통과해도** "passes_any"로 취급(오늘 밤 관례와 동일).

### 로버스트니스 판정 (그리드 이웃도 통과해야 인정)

과제 지시와 오늘 밤 #12(`research_eth_omega461_regime_threshold_h48qual_side_aware_revival_
20260814.py`) 선례를 그대로 적용: passes_any인 점이라도 **존재하는 모든 직접 이웃이 함께
통과**해야 "robust"로 인정한다. 편측만 실패해도 고립 스파이크로 기각, 헤드라인 숫자와 무관.

추가로 이번 스크립트에서만 필요했던 판정 하나를 더 넣었다: **baseline과 no_gate·with_gate
전부 바이트 단위로 동일한("디제너레이트 동률") 그리드 점은, 원기준 `>=` 비악화 공식상 자명하게
통과하더라도 "후보"로 세지 않는다.** 동률은 개선이 아니고, 단일터치 OOS 기회를 baseline을
안 건드린 것과 구별 불가능한 설정에 쓰는 건 낭비이기 때문이다. (아래 "발견 1" 참고 — 이
디제너레이트 동률이 실제로 그리드의 절반 이상을 차지했다.)

## G0 (필수, 포트폴리오 레벨)

zig075 `exit_threshold=0.95`(그리드 자체재현 기준점)가 오늘 밤 반복 검증된
`asymmetric_tabm_liveatr` baseline과 정확히 일치해야 한다 — VAL·OOS-Q1 둘 다 확인:

| 창 | 지표 | 실측 | 레퍼런스 | 일치 |
|---|---|---:|---:|---|
| VAL | no_gate | +46.59%/-21.70%/35건 | +46.59%/-21.70%/35건 | PASS |
| VAL | with_gate | +77.31%/-21.76%/26건 | +77.31%/-21.76%/26건 | PASS |
| OOS-Q1 | no_gate | +93.27%/-15.48%/24건 | +93.27%/-15.48%/24건 | PASS |
| OOS-Q1 | with_gate | +67.25%/-15.48%/19건 | +67.25%/-15.48%/19건 | PASS |

레퍼런스 출처: `tmp/causal_regen_20260516/eth_omega461_risk_controlled_exit_fallback_20260814/
report.json`(`eth_omega461_multiwindow_confirmation_gate_20260814.REFERENCE_VAL_OOSQ1_
ASYMMETRIC_TABM_LIVEATR`를 통해). **G0 PASS.**

부가 자체검증(과제의 필수 G0 범위는 아님): zig075 단독(전액가상자본) et=0.95 수치(PnL
+40.31%/MDD-13.07%/29건)가 `research_eth_omega461_exit_sweep_20260721.py` 자신이 2026-07-21에
이미 실행해 저장해 둔 `tmp/research_20260721/exit_threshold_sweep_VAL.csv`(zig075가 이
프로젝트에서 한 번도 손대진 적 없어 그 시점 수치가 여전히 유효)와 **정확히 일치**
(pnl/mdd/trades 전부) — 이번 스크립트의 컴포넌트-단독 조합이 독립적으로도 올바름을 재확인.

## 발견 1 — zig075는 보유 중 exit_head 확신이 이 구간에서 ~0.90을 거의 넘지 않는다

컴포넌트(zig075 단독) 그리드에서 exit_threshold가 0.90~0.999 구간 전부(0.90/0.92/0.95/0.97/
0.98/0.99) **완전히 동일한 수치**(PnL+40.31%/MDD-13.07%/29건)를 낸다 — zig075의 exit_head가
포지션을 보유한 bar에서 0.90 이상의 확률을 사실상 내지 않는다는 뜻이다. 포트폴리오 레벨에서도
동일한 패턴이 재현된다(0.90~0.999 전부 baseline과 no_gate·with_gate 완전 동일). **가설의
(a) 방향("회전을 늦추자")은 이 VAL 구간에서 구조적으로 시험 불가능하다** — 모델이 애초에
0.90을 넘는 확신을 표현하지 않으므로 바를 더 높여도 아무것도 달라지지 않는다. 이는 "시험했으나
효과 없음"이 아니라 "현재 모델 캘리브레이션상 그 방향에 여유가 없음"에 가깝다.

## 발견 2 — 유일한 실질 개선(0.80)은 이웃 실패로 고립 스파이크 판정

exit_threshold=0.80만 컴포넌트·포트폴리오 no_gate 양쪽에서 baseline과 실질적으로 다른(그리고
더 나은) 숫자를 낸다. 하지만 바로 옆 그리드점 0.85가 **컴포넌트 단독 경제성이 baseline보다
악화**(PnL+40.31%→+36.91%)해 원기준에서 탈락한다 — 흥미롭게도 0.85의 **포트폴리오** 수치는
baseline과 완전히 동일한 디제너레이트 동률이다(0.85가 만드는 추가 4건의 컴포넌트-단독 트레이드가
공유슬롯 포트폴리오 맥락에서는 zig075가 실제로 보유하는 bar 집합과 겹치지 않아 한 번도
발현되지 않는다). 즉 0.85는 "포트폴리오에 해 없음, 그러나 컴포넌트 단독으로는 손해"라는 이유로
실패하고, 그 실패가 0.80을 양쪽(왼쪽 이웃 없음, 오른쪽 이웃 0.85 실패)에서 고립시킨다.

## VAL 전체 그리드 결과

| exit_threshold | 컴포넌트(zig075 단독) PnL/MDD/거래수 | 포트폴리오 no_gate PnL/MDD/거래수 | 포트폴리오 with_gate PnL/MDD/거래수 | 원기준 | 완화기준 | 비고 |
|---|---:|---:|---:|---|---|---|
| 0.80 | +53.69%/-13.07%/39 | +56.25%/-21.70%/36 | +77.31%/-21.76%/26(baseline과 동률) | PASS | FAIL(with_gate 미개선) | **고립 스파이크로 기각**(이웃 0.85 실패) |
| 0.85 | +36.91%/-13.07%/33 | +46.59%/-21.70%/35(baseline과 동률) | +77.31%/-21.76%/26(동률) | **FAIL**(컴포넌트 PnL 악화) | FAIL | 포트폴리오는 디제너레이트 동률이나 컴포넌트 단독 악화로 탈락 — 0.80을 고립시키는 이웃 |
| 0.90 | +40.31%/-13.07%/29(동률) | +46.59%/-21.70%/35(동률) | +77.31%/-21.76%/26(동률) | PASS(자명) | FAIL(동률, 미개선) | 디제너레이트 동률 — 왼쪽 이웃(0.85) 실패로 "고립" 판정(아래 참고) |
| 0.92 | 동률 | 동률 | 동률 | PASS(자명) | FAIL | 디제너레이트 동률, 로버스트(양쪽 이웃 통과) |
| **0.95(기준점)** | +40.31%/-13.07%/29 | +46.59%/-21.70%/35 | +77.31%/-21.76%/26 | — | — | 자기 자신 |
| 0.97 | 동률 | 동률 | 동률 | PASS(자명) | FAIL | 디제너레이트 동률, 로버스트 |
| 0.98 | 동률 | 동률 | 동률 | PASS(자명) | FAIL | 디제너레이트 동률, 로버스트 |
| 0.99 | 동률 | 동률 | 동률 | PASS(자명) | FAIL | 디제너레이트 동률, 로버스트 |

**로버스트니스 판정**:
- passes_any(원기준 또는 완화기준 통과): {0.80, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99} — 0.85만
  탈락.
- 그리드 이웃까지 통과(robust): {0.92, 0.95, 0.97, 0.98, 0.99}.
- 고립 스파이크로 기각: **0.80**(실질 개선이지만 이웃 0.85 실패로 로버스트 아님, #12의
  LONG=0.35 기각과 동일 원칙), **0.90**(왼쪽 이웃 0.85가 실패라 기계적으로 "고립" 판정되지만,
  0.90 자신은 0.92~0.999와 완전히 동일한 안정 고원의 일부다 — 이건 스파이크가 아니라 비대칭
  그리드에서 이웃 규칙이 만든 경계 효과다. 실질적 결론에는 영향 없음: 0.90도 baseline과 동률이라
  어차피 "후보" 자격이 없다).
- robust 중 진짜 개선(baseline과 동률이 아닌) 후보: **0개** — {0.92, 0.95, 0.97, 0.98, 0.99}
  전부 디제너레이트 동률이라 후보에서 제외.

**VAL winner: 없음(`val_winner=None`)** — 로버스트하면서 동시에 baseline 대비 실질적으로
다른 지점이 그리드 전체에 하나도 없다.

## OOS

**열지 않았다.** 로버스트한 진짜 후보가 없어 과제의 "VAL 게이트 통과 후보가 있으면 OOS 단일터치"
조건 자체가 성립하지 않는다. 재튜닝이나 그리드 확장 없이 결과를 그대로 보고하고 종결한다
(사전등록 규율 — 결과를 본 뒤 그리드를 넓혀 재시도하지 않음).

## 해석

가설의 두 방향 모두 이 VAL 구간에서 실질적 지지를 얻지 못했다:

- **(a) 회전 둔화(threshold↑)**: zig075의 exit_head가 보유 bar에서 애초에 0.90을 넘는 확신을
  거의 내지 않아, 0.90 초과 전 구간이 구조적 무효화(no-op)다. "느리게 나가면 좋은 트레이드를
  더 들고 간다"는 메커니즘 자체가 이 모델·이 구간에서 발동할 여지가 없었다.
- **(b) 회전 가속(threshold↓)**: 0.80에서만 실질적이고 방향상 유리한 변화(컴포넌트·포트폴리오
  no_gate 둘 다 개선)가 나타났지만, with_gate는 baseline과 동률(개선 아님)이고 결정적으로
  이웃 그리드(0.85)가 컴포넌트 단독 경제성 악화로 실패해 그리드 전체가 "0.80 하나만 우연히
  좋아 보이는" 패턴과 구별되지 않는다 — 정확히 오늘 밤 #12가 정의한 "고립 스파이크는 헤드라인
  숫자와 무관하게 기각" 상황이다.

h48qual 쪽 GBDT/TCN/대기압력 실험들이 반복 관찰한 "슬롯 재순환"(한 컴포넌트의 청산 타이밍
변화가 다른 컴포넌트의 진입 기회를 바꾸는 상호작용) 메커니즘의 흔적도 이번엔 다른 형태로
나타났다 — 0.85가 컴포넌트 단독으로는 다른 궤적을 만들면서도 실제 공유슬롯 포트폴리오에서는
zig075가 보유하는 bar 집합 자체가 달라(h48qual 우선순위 때문에) 그 차이가 한 번도 발현되지
않는다는, 이전 실험들과는 다른 방향의 "포트폴리오가 컴포넌트 차이를 흡수해버리는" 사례다.

## 결론

**채택 불가.** 8점 VAL 그리드 전체를 스윕한 결과, 로버스트(그리드 이웃도 통과)하면서 동시에
baseline과 실질적으로 다른(디제너레이트 동률이 아닌) 후보가 하나도 없었다. 실질 개선을 보인
유일한 지점(0.80)은 사전등록된 이웃-로버스트니스 규칙에 따라 고립 스파이크로 기각했고, 나머지
"통과" 지점들은 baseline을 안 건드린 것과 바이트 단위로 동일했다. OOS는 열지 않았다.

Odyssey2 우선순위 큐 #6("zig075 exit_head 개선 — 별도 exit_threshold 미탐색") 항목은 이것으로
**부정 결과 종결**한다 — zig075의 exit_threshold를 단독으로 재보정하는 것만으로는(모델 재학습
없이) 이 VAL 구간에서 유의미한 개선을 찾지 못했다.

## 미해결 / 다음 단계

- 이번 실험은 zig075의 exit_threshold **하나의 숫자**만 스윕했다. "왜 0.85가 컴포넌트 단독으로
  악화되는 4건의 트레이드를 만드는지"는 원장 레벨로 추가 조사하지 않았다(범위 밖).
- 발견 1(모델이 0.90을 넘는 확신을 거의 안 낸다는 사실)은 zig075의 exit_head 확률 캘리브레이션
  자체에 대한 별도 질문을 시사한다 — 이번 실험 범위 밖.
- 0.80 근방(예: 0.78, 0.82)을 더 촘촘히 스윕하면 이웃-로버스트 조건을 만족하는 안정적인 저역
  고원이 존재하는지는 미확인 — 사전등록 그리드를 결과를 본 뒤 확장하는 것은 이번 실험의 규율
  위반이라 시도하지 않았다. 향후 별도 사전등록 실험으로 시도 가능.
- 채택 가능한 변경 0건, 라이브 파일 미변경.

## 준수 확인

`fresh_forward_bar_by_bar=true`(모든 리플레이는 `greedy.greedy_replay`/`sweep.replay_exit_
variant`의 단일 순방향 causal 루프, 이 스크립트가 새 bar-by-bar 로직을 추가하지 않음).
`trade_ledgers_used_as_input=false`(원장은 출력 전용). `saved_parent_exit_timestamps_used=false`.
`future_rows_used_for_entry=false`. h48qual은 direction_head/quality_head/encoder/exit_head
전부 동결(오늘 밤 확정 baseline 그대로), zig075도 direction_head/quality_head/encoder/exit_head
**가중치**는 동결 — exit_threshold 숫자만 스윕. h48qual은 이번 실험의 어떤 창에서도 건드리지
않았다.

`git diff` 확인(0줄): `trading_bot.py`, `trading_bot_modules/omega4_6_1_live.py`,
`trading_bot_modules/runtime_config.py`. `.env`는 gitignore 대상이라 세션 중 미접촉을 별도
확인. 기존 스크립트/모듈(`research_eth_omega461_exit_sweep_20260721.py`,
`replay_omega4_6_1_greedy_router_20260706.py`,
`research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py`,
`research_eth_omega461_live_sltp_mfe_width_20260813.py`,
`research_eth_omega461_risk_controlled_exit_fallback_20260814.py`,
`eth_omega461_multiwindow_confirmation_gate_20260814.py`) 전부 임포트만, 수정 없음(이름 바꾼
복사본조차 만들 필요가 없었음 — 위 "방법" 절 참고).

Seed-Diversity Ensemble Promotion Gate: 해당 없음(재학습 모델이 아닌 결정론적 threshold 재설정,
여러 시드 평균/배깅 앙상블 승격 주장 없음). Omega Artifact Integrity Promotion Gate: 해당
없음(신규 parent 예측 아티팩트 없음, 기존 라이브 h48qual/zig075 parent 아티팩트 그대로 재사용).

## 산출물

- 새 스크립트: `scripts/research_eth_omega461_zig075_exit_threshold_recalibration_20260814.py`
  — G0(포트폴리오 레벨, VAL+OOS-Q1) + zig075 컴포넌트 단독 그리드(`sweep.run_grid` 재사용, 역사적
  2026-07-21 파일과 교차검증) + 포트폴리오 8점 그리드(원기준/완화기준 이중 게이트) + 그리드
  이웃 로버스트니스 판정(고립 스파이크 명시적 기각) + 디제너레이트 동률 판정(후보 자격 제외) 전부
  단일 스크립트. OOS 단일터치는 로버스트한 진짜 후보가 없어 코드 경로상 실행되지 않음
  (`val_winner=None`이면 `return` — #9/#14와 동일한 "VAL 기각 시 OOS 미개방" 패턴).
- report.json: `tmp/causal_regen_20260516/eth_omega461_zig075_exit_threshold_recalibration_
  20260814/report.json`(G0 + 컴포넌트 그리드 + 역사적 교차검증 + VAL 8점 그리드 + 로버스트니스
  판정 + `val_winner=null` + `final_verdict="REJECTED_VAL_GATE"` 전부 포함).
- 거래 원장(diagnostic, 참고용): `tmp/causal_regen_20260516/
  eth_omega461_zig075_exit_threshold_recalibration_20260814/portfolio_ledger_val_g0_zig075_
  et095.csv`, `portfolio_ledger_oos_q1_g0_zig075_et095.csv`,
  `portfolio_ledger_val_val_zig075_et{0.80,0.85,0.90,0.92,0.95,0.97,0.98,0.99}.csv`.
- 인용 문서: `docs/experiments/eth_omega461_exit_head_liveatr_sustained_uptrend_vulnerability_
  20260814.md`(가설의 근거가 된 회전-가속 메커니즘, #10), `docs/experiments/
  eth_omega461_regime_threshold_h48qual_side_aware_revival_20260814.md`(그리드 이웃 로버스트니스
  판정 방법론의 선례, #12), `docs/experiments/eth_omega461_relaxed_gate_rescoring_20260814.md`
  (완화기준 정의), `docs/experiments/eth_omega461_multiwindow_confirmation_gate_20260814.md`
  (재사용한 다중구간 게이트 모듈), `docs/model_contracts/odyssey2_eth_live_injection_contract_
  20260813.md`(서브 프로젝트 계약, 우선순위 큐 #6).
