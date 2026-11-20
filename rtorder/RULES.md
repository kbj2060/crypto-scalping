# rtorder 규칙 추적표 (2026-11-20 · 첫 조각)

점검표 = `docs/rtcore_order_rules_inventory_20261120.md`(145개). 이 조각의 범위 = 계획 순수 함수(`src/plan.rs`) + 지정가 집행 상태 기계(`src/exec.rs`). 어디에도 연결하지 않았다.

상태: **옮김** = Rust 에 있고 시험이 지킨다 · **부분** = 규칙의 일부만 이 조각에 있다(나머지 몫과 조각을 적음) · **범위 아님** = 다음 조각 · **⚠시험 없음** = 옮긴 몫에 지켜 주는 시험이 없다.
시험 약어: `fx` = tests/plan_fixture.rs(파이썬 정답지 2,266건) · `sc` = tests/plan_selfcheck.rs(원본 자체점검 이전) · `inv` = tests/exec_invariants.rs · `mut` = tools/mutate.py(양성 대조).
다음 조각 이름: **접수**(라우트·confirm·게이트·잠금·칸) · **조립**(assemble_entry/exit_plan·계좌·풀·basis) · **감시**(bracket_tick·무장 파일·brackets 주문) · **대기 관리**(대기 칸·복구 파일·고아 정리) · **계좌·알림**(사용자 스트림) · **거래소 구현**(HTTP·서명·밴 기록).

## 1. 접수 문 G1–G18

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| G1 | 범위 아님 | 접수 |
| G2 | 범위 아님 | 접수 |
| G3 | 부분 | `plan::exec_enabled`(매번 환경 읽음 · 1/true/yes) · sc `exec_gate_reads_env_values` · 계획 `dry_run` 칸 fx · 403 응답은 접수 |
| G4 | 범위 아님 | 감시·계좌·알림 |
| G5 | 범위 아님 | 접수 |
| G6 | 범위 아님 | 접수 |
| G7 | 범위 아님 | 접수 |
| G8 | 범위 아님 | 접수 |
| G9 | 범위 아님 | 감시 |
| G10 | 범위 아님 | 접수 |
| G11 | 범위 아님 | 접수 |
| G12 | 범위 아님 | 접수 |
| G13 | 범위 아님 | 접수 |
| G14 | 범위 아님 | 접수 |
| G15 | 부분 | 계획 함수 안의 검사(측면·비율 (0,1]·호가)는 `build_entry_plan`/`build_exit_plan` 이 같은 사유 문자열로 거절 · fx(거절 553건) · 쿼리 파싱(레버리지·규칙·선·기다리기)은 접수 |
| G16 | 범위 아님 | 접수 |
| G17 | 부분 | 마진 방식·권장배수 신선도는 조립 · 규칙 계산 자체는 P17 |
| G18 | 범위 아님 | 접수 |

## 2. 계획 조립 P1–P30

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| P1 | 부분 | 미리보기·실주문이 같이 부를 함수 하나 = `build_entry_plan`·`build_exit_plan` (fx) · 조립 함수 자체는 조립 |
| P2 | 범위 아님 | 조립 |
| P3 | 범위 아님 | 조립(`existing_notional` 은 호출부가 넘긴다) |
| P4 | 범위 아님 | 조립 |
| P5 | 범위 아님 | 조립 |
| P6 | 부분 | `MARGIN_BUFFER`·가용 잔고 자르기 `build_entry_plan` · fx · mut «버퍼 1+2%» · 대기 몫 미리 빼기는 조립 |
| P7 | 범위 아님 | 조립 |
| P8 | 옮김 | `build_entry_plan`(상한까지 줄임·notes·capped·blocked) · fx · sc · mut «room <= 0» |
| P9 | 옮김 | `build_entry_plan`(비율은 상한 뒤) · fx · sc · mut «비율을 상한 전에» |
| P10 | 옮김 | `build_entry_plan`·`build_exit_plan` 비율 검사 · fx(거절) · sc |
| P11 | 옮김 | `floor_to`·`ceil_to`·진입/청산 가격 방향 · fx(경계 입력 10종) · mut «내림/올림 경계» «롱 진입가» |
| P12 | 부분 | 진입 최소 수량·명목 blocked `build_entry_plan` · fx · 지정가 진입 재검사는 조립 |
| P13 | 옮김 | positionSide 필수 · reduceOnly 없음(계획·`exec::LimitOrder` 에 칸 자체가 없다) · sc(reduceOnly 키 없음) |
| P14 | 부분 | `resolve_exit_position` · fx · sc · 새 계좌 읽기·감시 심볼 한정은 조립·감시 |
| P15 | 옮김 | `build_exit_plan`(전량이면 곱셈 안 함 · min) · fx · sc · mut «min → max» |
| P16 | 범위 아님 | 거래소 구현(`ensure_leverage`) |
| P17 | 부분 | `commit_rule_l`·`rec_rule_size` · fx(436·200건) · sc · mut «1/3 → 0.75» «0.9×» «동점» · 권장배수 읽기는 조립 |
| P18 | 옮김 | `rec_rule_size`(add_stop_frac) · `rec_rule_add_order` · fx · sc |
| P19 | 범위 아님 | 조립 |
| P20 | 부분 | `rec_rule_bracket`(−5% 가격 · prev 유지) · fx · sc · 무장 결정은 조립 |
| P21 | 범위 아님 | 조립 |
| P22 | 범위 아님 | 조립 |
| P23 | 범위 아님 | 조립 |
| P24 | 옮김 | `line_bracket` · fx(177) · sc · mut «롱 올림 → 내림» |
| P25 | 범위 아님 | 조립 |
| P26 | 범위 아님 | 조립 |
| P27 | 옮김 | `exit_deadline_sec`(None→120 · 넘침/0 나누기도 파이썬과 같은 예외) · `run_exit` 가 계획 마감 사용 · fx · sc |
| P28 | 범위 아님 | 조립 |
| P29 | 범위 아님 | 조립 |
| P30 | 범위 아님 | 조립 |

## 3. 지정가 집행 E1–E26

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| E1 | 부분 | `Exchange::place_limit_gtx` 하나뿐(GTX 만 나갈 수 있다 -- 구조) · ⚠시험 없음(구조로 보장) · 익절·dbad(GTC, M12)는 감시 |
| E2 | 부분 | 오류 문자열 규약(`Reply = Result<Order, String>`, "상태 [code] msg")을 형으로 고정 · 서명·밴 기록은 거래소 구현 |
| E3 | 옮김 | `is_post_only_reject` · inv11 · mut «코드 앞머리만» |
| E4 | 옮김 | `fill_maker` -5022 → 재호가 · 그 밖 rejected · inv06 · inv11 |
| E5 | 옮김 | `fill_maker` 호가 실패 → rejected · inv `e05_book_failure_after_post_only_reject_is_rejected` |
| E6 | 옮김 | ✅해결(2026-11-20, D2): `fill_maker`·`run_entry_wait`·`run_entry_limit` 이 공용 `resolve_unknown_post`(E7 처리) 하나를 쓴다 · inv `e06_…` · `d2_unknown_outcome_same_on_all_paths`(4경로×a~e+-1006+4xx 음성) · 퍼즈 «불명» 접수 4경로 · 음성 대조: 해결 전 exec.rs 로 d2·e06·fuzz 실패 |
| E7 | 옮김 | `resolve_unknown_post`: 네트워크·5xx·-1007·-1006 → 3회 조회(1초) → 찾으면 이어 감 → 끝내 모르면 표식 DELETE → error · inv `d2_…`(d·e) · mut «5xx 를 확정 거부로» |
| E8 | 옮김 | `ensure_closed`(재조회 상태 · 3회×1초) · 확인 못 하면 error · inv03·05·07 · fuzz · mut «취소 확인 실패해도 재호가» |
| E9 | 옮김 | 오류 때 마지막 체결량 · inv `e09_…` · mut «마지막 체결량 버림» |
| E10 | 부분 | 지정가 체결량 = 응답 `executedQty` · 장부 대조(check: 적힌 체결 = 거래소 체결) · 시장가 다리는 미정(M1~M9) |
| E11 | 옮김 | `TERMINAL` · inv04·partial 사례 · mut «PARTIALLY_FILLED 를 종료로» |
| E12 | 옮김 | 상태 조회 실패 재시도 · inv `e12_…` |
| E13 | 옮김 | `drifted`(주문 측면) · inv02·10(가격이 따라가는 방향) · mut «부등호 뒤집음» |
| E14 | 옮김 | `Exchange::book(symbol)` 에 주문 심볼을 넘긴다 · inv(가짜가 심볼 단언) |
| E15 | 옮김 | `REPEG_MAX`=40 · 컴파일 시점 단언 `REPEG_MAX×POLL_SEC ≥ FALLBACK_SEC` · inv11(상한 소진 → rejected) |
| E16 | 옮김 | 알림에 깨도 리페그 판단은 3초마다 · inv `e16_…` · mut «매번 리페그 판단» |
| E17 | 옮김 | `run_entry`·`run_exit`·PEG 가 `fill_maker` 하나 · ⚠시험 없음(구조 -- 원본도 Xself 소스 검사뿐) |
| E18 | 옮김 | `run_entry` 대기 취소 · inv `e18_…`(취소·부분 뒤 취소·확인 실패·음성 대조) · mut «취소 무시» |
| E19 | 부분 | 표식 `PEG_TAG`·`WAIT_TAG` 를 cid 에 · inv(wait·d2 가 표식 단언) · dbtp·dbsl·dbad·`cancel_bracket` 은 감시 |
| E20 | 옮김 | `run_entry_wait` · inv `wait_paths`(재호가 없음·-5022 재시도·거래소 취소·on_fill 한 번) · mut «on_fill 매번» «거래소 취소» |
| E21 | 옮김 | `run_entry_limit`(따라가지 않음 · -5022 는 수정 직후 3회만) · inv `limit_paths` |
| E22 | 옮김 | 수정은 체결 0 일 때만 · inv `limit_paths`(부분 뒤 수정) · mut «부분 뒤에도 수정» |
| E23 | 옮김 | 자동 재조정(20초·1bp·50회·체결 시작 뒤 멈춤) · inv `limit_paths` |
| E24 | 옮김 | 취소 확인 중 체결 몫 on_fill · inv `limit_paths`(on_fill 한 번) |
| E25 | 부분 | 닿은 뒤 따라가기 `run_entry`(maker_only → 잔량 버림) · inv `maker_only_and_market_entry_plans` · mut «maker_only 무시» · 닿음 감시 `run_entry_stop` 은 대기 관리 |
| E26 | 옮김 | `run_entry_limit` PEG(취소 확인 → action 비움 → 따라가기 → 잔량 버림) · inv `limit_paths`(PEG 두 사례) |

## 4. 대기 주문 W1–W12

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| W1 | 범위 아님 | 대기 관리 |
| W2 | 범위 아님 | 대기 관리 |
| W3 | 부분 | `LIMIT_MAX_SEC`(4시간) 만료 취소 · inv `limit_paths`(만료) · 기다리기 마감 잔량 시장가는 미정 M3 |
| W4 | 범위 아님 | 대기 관리 |
| W5 | 범위 아님 | 대기 관리 |
| W6 | 범위 아님 | 접수(수정 API) · 집행 쪽 «부분 뒤 수정 버림»은 E22 |
| W7 | 범위 아님 | 접수 |
| W8 | 범위 아님 | 접수 |
| W9 | 범위 아님 | 감시 |
| W10 | 범위 아님 | 대기 관리 |
| W11 | 범위 아님 | 대기 관리 |
| W12 | 옮김 | `exec::Shared`(집행 루프와 API 가 같은 칸) · inv(가짜가 시각표대로 action·amend 를 넣는다) |

## 5. 브래킷 B1–B33

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| B1 | 범위 아님 | 감시 |
| B2 | 옮김 | `bracket_key`·`bracket_rekey` · fx · ⚠옛 키·새 키 충돌 순서는 다를 수 있음(plan.rs ponytail) |
| B3 | 범위 아님 | 감시 |
| B4 | 범위 아님 | 감시 |
| B5 | 범위 아님 | 감시 |
| B6 | 범위 아님 | 감시(물타기 크기 계산 `rec_rule_add_order` 는 P18) |
| B7 | 부분 | `line_hit`(tp_resting 아님 → 서버가 닫음) · fx · sc · 닫기 실행은 감시 |
| B8 | 범위 아님 | 감시 |
| B9 | 옮김 | `bracket_action` clear(무장+5초 · ok · 0) · fx(armed_suite 280) · sc · mut «+5초 → +0» |
| B10 | 범위 아님 | 감시 |
| B11 | 옮김 | `bracket_sl_hit`(마감 5분봉 종가 · watch_from) · fx · sc |
| B12 | 옮김 | `bracket_action` 선 분기 · `line_hit`(손절 먼저) · `sl_legs_hit` · fx · sc · mut «닿음 >= → >» |
| B13 | 부분 | `sl_up_of`·`line_bracket(sl_watch)` · fx · sc · check_lines 는 접수 |
| B14 | 범위 아님 | 감시 |
| B15 | 범위 아님 | 감시 |
| B16 | 범위 아님 | 감시 |
| B17 | 범위 아님 | 감시 |
| B18 | 범위 아님 | 감시(발동 청산 = `run_exit` -- 마감 잔량은 미정 M9) |
| B19 | 옮김 | `bracket_merge`(OWNED_BY_SAVED · 쏜 틱의 tp_resting) · fx(92) · sc · mut «tp_resting 규칙 뒤집음» |
| B20 | 부분 | 판정 `sl_legs_hit`·`bracket_action`→SlLeg · fx · sc · 조각 청산 실행은 감시 |
| B21 | 부분 | 판정: sl_off 면 hold · fx · sc · 해제 API 는 접수·감시 |
| B22 | 범위 아님 | 감시 |
| B23 | 옮김 | `bracket_action` sl_off + fired "tp"/True · fx · sc |
| B24 | 범위 아님 | 감시(tp_off 판정은 `line_hit` 에 있음 · fx) |
| B25 | 범위 아님 | 접수·감시 |
| B26 | 범위 아님 | 접수·감시 |
| B27 | 옮김 | `split_tp_qty` · fx(194) · sc · mut «carry 버림» |
| B28 | 범위 아님 | 감시 |
| B29 | 범위 아님 | 감시 |
| B30 | 범위 아님 | 감시 |
| B31 | 범위 아님 | 감시 |
| B32 | 범위 아님 | 접수 |
| B33 | 범위 아님 | 계좌·알림 |

## 6. 청산 C1–C6

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| C1 | 범위 아님 | 조립 |
| C2 | 범위 아님 | 감시 |
| C3 | 범위 아님 | 접수(스위칭 -- 두 다리 다 시장가 = 미정 M11) |
| C4 | 범위 아님 | 접수 |
| C5 | 범위 아님 | 접수 |
| C6 | 범위 아님 | 접수 |

## 7. 재시작·종료·오류 R1–R9

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| R1 | 범위 아님 | 대기 관리(고아 정리 -- 표식은 E19) |
| R2 | 범위 아님 | 대기 관리 |
| R3 | 범위 아님 | 접수(`settle`) -- 집행 함수는 예외를 내지 않고 항상 종료 상태를 남긴다(inv ③ · fuzz 400건) |
| R4 | 부분 | 취소 확인 실패·지정가 접수 불명·PEG 호가 실패 → `Phase::Error` · inv03·05·07·d2 · API 키 없음은 거래소 구현 · 스위칭은 접수 |
| R5 | 범위 아님 | 접수 |
| R6 | 범위 아님 | 접수 |
| R7 | 부분 | `Exchange::nap` 계약(일찍 깸) · 실제 구현(취소가 안 삼켜짐)은 거래소 구현 · ⚠시험 없음(이 조각엔 실제 nap 이 없다) |
| R8 | 범위 아님 | 대기 관리 |
| R9 | 범위 아님 | 대기 관리 |

## 8. 계좌·체결 알림 A1–A11

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| A1 | 범위 아님 | 계좌·알림 |
| A2 | 범위 아님 | 계좌·알림 |
| A3 | 범위 아님 | 계좌·알림 |
| A4 | 범위 아님 | 계좌·알림 |
| A5 | 범위 아님 | 계좌·알림 |
| A6 | 범위 아님 | 계좌·알림 |
| A7 | 범위 아님 | 계좌·알림 |
| A8 | 범위 아님 | 계좌·알림 |
| A9 | 범위 아님 | 계좌·알림 |
| A10 | 부분 | 알림은 앞당기기만: `nap` 으로 일찍 깨도 3초 판단 그대로 · inv `e16_…` · 알림 수신은 계좌·알림 |
| A11 | 범위 아님 | 계좌·알림 |

## 집계

| 상태 | 개수 |
|---|---|
| 옮김 | 38 |
| 부분 | 22 |
| 범위 아님 | 85 |
| 미정(사용자 결정) | 0 -- 규칙 줄 자체는 없음. 시장가 분기 13곳은 아래 따로 |
| 옮긴 몫에 시험 없음(⚠) | 4 (E1 · E17 · R7 · B2 충돌 순서) |

## 시장가 분기 M1–M13 (모두 미정 -- 사용자 결정 대기 · 옮기지 않음)

| # | Rust 에서 닿는 자리 | 결과 |
|---|---|---|
| M1 | `exec::run_entry` 따라가기 마감 잔량 | `NeedsMarketDecision` · `market_branch="M1"` · inv02·10·e18 음성 대조 |
| M2 | `exec::run_entry` 계획 `market=true` | 같은 갈래 "M2" · inv `maker_only_and_market_entry_plans` |
| M3 | `exec::run_entry_wait` 마감·action=market | "M3" · inv `wait_paths` |
| M4 | `exec::run_entry_limit` action=market(취소 확인 뒤) | "M4" · inv `limit_paths` |
| M5 | `run_entry_stop` -- 이번 범위 아님 | -- |
| M6·M7 | `plan::build_exit_plan` 이 계획을 MARKET 으로 만든다(원본 값 그대로 -- 정답지 일치) → `exec::run_exit` 가 "M6/M7/M8" 로 멈춘다 | inv09 · fx |
| M8 | 조립(`market=True`) -- 집행에서는 M6/M7 과 같은 갈래 | inv09 |
| M9 | `exec::run_exit` 마감 잔량 | "M9" · inv02 |
| M10 | 비상 스탑(Algo STOP_MARKET) -- 감시 조각. 가격 계산 `rec_rule_bracket` 만 옮김 | -- |
| M11 | 스위칭 -- 접수 조각 | -- |
| M12 | 자동 물타기 GTC -- 감시 조각(크기 `rec_rule_add_order` 만 옮김) | -- |
| M13 | 대기 동작 접수 -- 접수 조각 | -- |

## 원본과 일부러 다르게 둔 곳

- 시장가 13곳 → 위 표(보내지 않고 멈춤).
- `run_entry`·`run_entry_wait`·`run_entry_limit`·`run_exit` 의 «API 키 없음 → error» · `ensure_leverage` 호출 → 거래소 구현·조립 조각.
- 파이썬 예외(0 나누기·inf→int·빈 목록 인덱스) → 같은 종류의 `PlanError`(정답지가 종류·문구까지 비교).
- ~~D2(«나갔는지 모르는 주문» 처리가 세 경로에서 다름)는 통일하지 않았다~~ → **해결(2026-11-20, 사용자 결정)**: 파이썬 원본을 먼저 고치고(`resolve_unknown_post` 공용 함수) Rust 도 같게 맞췄다 -- 원본과 다른 곳이 아니다. inv `d2_unknown_outcome_same_on_all_paths` 가 «세 경로가 같다»를 고정한다.
