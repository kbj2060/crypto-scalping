# Yush 잔여 후보 #2 — 브레이크이븐 스톱 exit 오버레이 (2026-08-15, REJECTED)

## 배경

[[yush_orderflow_absorption_closed_20260815]]가 "부분적으로 취할 가치가 있는 것"으로 남겨둔 두 번째
후보: Yush의 리스크 규율 "수익 거래를 손실로 되돌리지 않기"(브레이크이븐 스톱) — 그 문서는 "Omega
TP/SL 프레임에서 직접 테스트 가능한 exit 오버레이 형태... 단 별도 실험이 필요하며 이 문서는 그것을
측정하지 않았다"고 명시했다. 이 문서가 그 실험이다.

## 방법

**메커니즘**: 포지션 MFE가 자기 자신의 take_profit 목표치의 `ACTIVATE_FRAC=0.5`(50%)에 도달하면
"무장"되고, 그 이후 가격이 진입가(0% 수익) 이하로 밀리면 즉시 청산. `ACTIVATE_FRAC=0.5`는 **사전
등록된 유일한 값**(스윕/튜닝 없음) — 모델 자신의 TP를 기준 스케일로 삼아 새 자유 상수를 만들지
않았다.

**신규 시뮬레이션 로직 없음** — `research_eth_omega461_exit_sweep_20260721.replay_exit_variant`가 이미
지원하는 PROPORTIONAL trailing-stop 훅(`trailing_activate_frac`+`trailing_retain_frac`)에서
`trailing_retain_frac=0.0`이 정확히 브레이크이븐 스톱과 같다(무장 후 "MFE의 0%까지 반납하면 청산" =
"진입가 이하로 밀리면 청산"). 컴포넌트 레벨은 이 조합을 그대로 호출. **포트폴리오 레벨 라우터**
(`replay_omega4_6_1_greedy_router_20260706.greedy_replay`)는 기존에 FIXED-DISTANCE 버전
(`trailing_trail_frac`)만 지원했다 — MFE가 거래마다 달라 고정폭으로는 "정확히 진입가"를 재현할 수
없으므로, 자매 함수와 동일한 패턴으로 `trailing_retain_frac`을 이 커밋에서 추가했다(기존 78개 호출부
전부 새 키워드 인자 기본값 None으로 영향 없음).

**중복 확인**: `research_eth_omega461_exit_ideas2_20260721.py`가 2026-07-21에 이미 유사한 trailing-stop
축을 스윕했으나 `retain_frac∈{0.3,0.4,0.5,0.6}`만 테스트, `0.0`(순수 브레이크이븐)은 시도된 적 없음
— 이 실험과 중복 아님.

**검증 강도**: VAL→단일 OOS가 아니라 `eth_omega461_multiwindow_confirmation_gate_20260814`의 전체
6창(2025 Q1/Q2/Q3 참고 + VAL + OOS-Q1/Q2 단일터치)을 사용했다 — Odyssey2가 "VAL은 6창 중 exit
개선 여지가 가장 적다"고 확인한 바 있어(exit-side 후보를 VAL 하나로 거르는 것은 이 저장소에서 특히
신뢰할 수 없음), exit 오버레이 후보에는 처음부터 강한 게이트를 적용했다. 컴포넌트(h48qual/zig075
단독) + 포트폴리오(둘 다 브레이크이븐 적용, 라우터 슬롯경쟁 포함) 둘 다 보고하되, 판정은 기존
Odyssey 후보들과 동일하게 포트폴리오 with_gate 기준.

## 결과

G0 자체검증(VAL 포트폴리오 no_gate 베이스라인이 알려진 기준값 재현): **PASS**(36.82%/-24.34%/29건).

포트폴리오 레벨(no_gate → with_gate, 베이스라인 → 브레이크이븐):

| 창 | 티어 | with_gate 베이스라인 | with_gate 브레이크이븐 | 승패 |
|---|---|---|---|---|
| 2025-Q1 | 참고 | 28.54%/-20.62%/19건 | 64.13%/-17.31%/25건 | ✅ 개선 |
| 2025-Q2 | 참고 | 39.99%/-10.82%/15건 | 9.05%/-23.24%/20건 | ❌ 악화 |
| 2025-Q3 | 참고 | -9.73%/-44.37%/19건 | -3.74%/-41.42%/21건 | ✅ 개선(단, 여전히 음수) |
| VAL | 게이트 상위 | 54.88%/-31.11%/22건 | 13.24%/-32.56%/27건 | ❌ 악화 |
| **OOS-Q1** | **판정** | **44.48%/-15.48%/20건** | **18.02%/-19.18%/19건** | ❌ **PnL -26.5pp, MDD 악화** |
| **OOS-Q2** | **판정** | **9.85%/-15.00%/10건** | **-46.47%/-47.77%/20건** | ❌ **부호반전+MDD 3배 이상 악화** |

**판정: `REJECTED_SIGN_MISMATCH`** — strict(0pp MDD 여유)와 relaxed(3pp 여유) 둘 다, OOS-Q1·OOS-Q2
단일터치 두 창 모두 실패. 참고 티어는 엇갈렸지만(Q1 개선/Q2 악화/Q3 개선), **실제 판정 대상인 두 OOS
창에서는 예외 없이 악화**됐고 특히 OOS-Q2는 재앙적이다.

메커니즘 관찰(컴포넌트 레벨 exit_reasons): 브레이크이븐 적용 시 거래수가 소폭 늘고(예: OOS-Q2
h48qual 12→15건, zig075 13→23건) `trailing_stop`이 새 청산사유로 등장하며 보유시간이 대체로
줄어든다. OOS-Q2 zig075는 승률이 30.8%→8.7%로 붕괴(23건 중 2승) — 다수 거래가 50% MFE 지점까지
갔다가 진입가로 되돌아온 뒤(비용 포함 소손실로 청산) 원래라면 도달했을 TP를 놓쳤다는 뜻이다.

## 결론

Yush의 리스크 규율("수익을 손실로 되돌리지 않기")은 직관적으로 방어적으로 들리지만, 이 모델·자산
조합에서는 **실제 승리거래의 상당수가 50% MFE 지점에서 진입가 근처까지 되돌아왔다가 다시 올라가는
비단조 경로를 거친다** — 브레이크이븐 스톱은 바로 그 되돌림에서 청산시켜 이후의 진짜 상승을 놓치게
한다. 이는 오늘 진단한 [[eth_omega461_exit_head_liveatr_relabel_walkforward_mechanism_diagnosis_20260815]]
(MFE giveback 기반 조기청산이 큰 승리거래를 깎아먹어 재현성이 없었던 것)와 같은 계열의 실패
메커니즘이고, `exit_ideas2`(2026-07-21)의 retain_frac 0.3~0.6 스윕이 전부 실패했던 것과도 방향이
같다 — **이 자산/모델에서는 "피크 대비 되돌림"을 트리거로 쓰는 exit 계열 전체가 구조적으로 불리하다**는
증거가 이제 세 번째 독립 사례로 쌓였다.

**채택하지 않음.** Yush 잔여 후보 2건 모두 종결 — [[yush_orderflow_absorption_closed_20260815]]의
"부분적으로 취할 가치가 있는 것" 항목은 이제 남은 게 없다.

## 산출물

- `scripts/research_eth_yush_breakeven_stop_exit_overlay_20260815.py`
- `scripts/replay_omega4_6_1_greedy_router_20260706.py`(`trailing_retain_frac` 파라미터 추가, 기존
  호출부 78개 전부 하위호환)
- `tmp/causal_regen_20260516/eth_yush_breakeven_stop_exit_overlay_20260815/report.json` + 창별 포트폴리오
  렛저 12개(6창×2변형)

## 준수 확인

`fresh_forward_bar_by_bar=true`(전 창 단일 순방향 bar-by-bar replay, `replay_exit_variant`/
`greedy_replay` 제어흐름 미수정 — 신규 파라미터는 기존 trailing-stop 분기의 대안 조건 하나 추가일
뿐 루프 구조는 그대로). `trade_ledgers_used_as_input=false`. `saved_parent_exit_timestamps_used=false`.
`future_rows_used_for_entry=false`. 신규 학습 없음(ACTIVATE_FRAC/RETAIN_FRAC은 TP/SL과 같은 런타임
실행 상수이지 학습 가중치가 아님 — 시드 다양성 축 해당 없음). `trading_bot.py`/
`trading_bot_modules/omega4_6_1_live.py`/`.env`/라이브 배포 번들 전부 미접촉.
