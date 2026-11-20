# rtorder 규칙 추적표 (2026-11-20 · 첫·둘째·셋째 조각)

점검표 = `docs/rtcore_order_rules_inventory_20261120.md`(145개). 첫 조각 = 계획 순수 함수(`src/plan.rs`) + 지정가 집행 상태 기계(`src/exec.rs`). 둘째 조각 = 서명·요청 문자열(`src/sign.rs`) + 전송(`src/http.rs`, `Exchange` 구현) + 손절·익절·레버리지·고아 정리(`src/bracket.rs`) + 감시 진입(`exec::run_entry_stop`). 셋째 조각 = 손절·익절 감시와 주문 상태 칸(`src/watch.rs` -- orders.py 의 bracket_tick·watch_brackets·partial_exit_rearm_tp·arm_bracket·plan_from_current·keep_rearm·limit_recover·wait_cleanup·exit_slot·busy_or). 어디에도 연결하지 않았다.

상태: **옮김** = Rust 에 있고 시험이 지킨다 · **부분** = 규칙의 일부만 이 조각에 있다(나머지 몫과 조각을 적음) · **범위 아님** = 다음 조각 · **⚠시험 없음** = 옮긴 몫에 지켜 주는 시험이 없다.
시험 약어: `fx` = tests/plan_fixture.rs(파이썬 정답지 2,266건) · `sc` = tests/plan_selfcheck.rs(원본 자체점검 이전) · `inv` = tests/exec_invariants.rs · `rp` = tests/http_replay.rs(원본 파이썬 + 진짜 aiohttp 정답지 112시나리오 -- 요청 열 글자까지·결과·밴 기록·연결 수) · `mesh` = tests/http_mesh.rs(가짜 거래소를 HTTP 뒤에) · `mut` = tools/mutate.py(양성 대조) · `wf` = tests/watch_fixture.rs(원본 orders.py 감시 클로저 정답지 97사례·120틱·호출 419 -- 호출 열·저장본·상태 칸·로그·예외 · 양방향 왕복) · `wi` = tests/watch_invariants.rs(퍼즈 300×20틱 · 루프 · 재시작 · busy_or) · `wx` = tests/watch_exec.rs(감시 → exec::run_exit → 가짜 거래소 · 시장가 분기) · `rt` = tools/check_watch_roundtrip.py(Rust 가 쓴 파일을 파이썬 읽는 쪽으로 96/96). 감시 양성 대조 = mutate.py `watch` 27/27(wf 22 · wx 2 · wi 3).
다음 조각 이름: **접수**(라우트·confirm·게이트·잠금·칸) · **조립**(assemble_entry/exit_plan·계좌·풀·basis) · **감시**(bracket_tick·무장 파일·arm_bracket 등 orders.py 쪽) · **대기 관리**(대기 칸·복구 파일·재시작 훅) · **계좌·알림**(사용자 스트림).

## 1. 접수 문 G1–G18

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| G1 | 범위 아님 | 접수 |
| G2 | 범위 아님 | 접수 |
| G3 | 부분 | `plan::exec_enabled`(매번 환경 읽음 · 1/true/yes) · sc `exec_gate_reads_env_values` · 계획 `dry_run` 칸 fx · 403 응답은 접수 |
| G4 | 부분 | 감시 틱 정지 = `Watcher::bracket_tick`(스위치 꺼짐이면 계좌도 안 읽음 · 무장 파일 그대로) · wf `exec_switch_off`·`exec_switch_off_during_tick_still_fires`(틱 도중 꺼져도 그 틱은 끝까지 = 원본) · wi(퍼즈: 꺼짐이면 호출 0·저장본 불변) · mut «스위치 꺼져도 감시» · 고아 정리 시작 게이트(start_wait_cleanup)·사용자 스트림은 접수·계좌·알림 |
| G5 | 부분 | 칸 = `Watcher::entry` · `watch::busy`(working·submitting) · 409 응답은 접수 |
| G6 | 옮김 | `Watcher::exit_slot`(코인 칸을 만들어 둠 · 같은 코인 사용 중 None) · wf `two_coins_other_entry_busy_uses_side_slot`·`same_coin_entry_busy_skips`·`same_coin_side_slot_busy_skips` · wi(퍼즈: ETH 칸이 바쁘면 ETH 청산 0) · mut «같은 코인 칸이 바빠도» |
| G7 | 범위 아님 | 접수 |
| G8 | 부분 | `watch::busy_or`(tokio Mutex try_lock -- 잠겨 있으면 None = 409, 기다리지 않음) · wi `busy_or_refuses_while_locked` · 어느 엔드포인트가 잠금 안/밖인지는 접수 |
| G9 | 옮김 | `bracket_tick` 의 세 자리(선 발동 전 · 조립 뒤 · 분할 손절) -- 원본 그대로 조립(계좌 새로 읽기)은 잠금 중에도 한다(D9) · wf `lock_held_rule_D9`·`lock_held_lines_D9`·`lock_taken_during_plan`·`sl_leg_lock_held_skips` · wi(퍼즈: 잠금 중 청산 0 · 잠금으로 미룬 틱 813) · mut «선 발동 전 잠금 무시»(wf) «조립 뒤 잠금 무시»(wi) |
| G10 | 범위 아님 | 접수 |
| G11 | 범위 아님 | 접수 |
| G12 | 범위 아님 | 접수 |
| G13 | 범위 아님 | 접수 |
| G14 | 범위 아님 | 접수 |
| G15 | 부분 | 계획 함수 안의 검사(측면·비율 (0,1]·호가)는 `build_entry_plan`/`build_exit_plan` 이 같은 사유 문자열로 거절 · fx(거절 553건) · 쿼리 파싱(레버리지·규칙·선·기다리기)은 접수 |
| G16 | 범위 아님 | 접수 |
| G17 | 부분 | 마진 방식 읽기 `bracket::margin_type`(키 없음·조회 실패·빈 목록 → None · 섞이면 isolated) · rp(5) · 409/503 응답과 권장배수 신선도는 조립 · 규칙 계산 자체는 P17 |
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
| P14 | 부분 | `resolve_exit_position` · fx · sc · 감시 청산은 무장한 심볼만·새 계좌(`WatchIo::exit_plan(… only_symbol)` -- wf 호출 열이 only_symbol·fresh=True 를 대조) · 조립 자체는 조립 |
| P15 | 옮김 | `build_exit_plan`(전량이면 곱셈 안 함 · min) · fx · sc · mut «min → max» |
| P16 | 부분 | `bracket::ensure_leverage`·`current_leverage`(0·음수 = 안 보냄 · 같으면 안 보냄 · 거부·조회 실패를 결과에 · 응답에 값 없으면 목표값) · rp(8) · mut «같아도 보냄» · «실패해도 주문은 보낸다»·state 에 싣기는 호출부(접수 -- exec::run_entry 는 레버리지를 부르지 않는다) |
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
| P29 | 옮김 | `bracket::commission_bp`(live 1시간 · 실패 5분 · stale = 마지막 성공값 · fallback 2/5bp · 캐시는 `Binance` 하나에) · rp(«캐시·TTL» 5호출 · «처음부터 실패» · 키 없음) · mut «실패 캐시 1시간» «stale 대신 fallback» |
| P30 | 범위 아님 | 조립 |

## 3. 지정가 집행 E1–E26

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| E1 | 옮김 | 집행 = `Exchange::place_limit_gtx` 하나(GTX 만 -- 구조) · HTTP 구현이 `type=LIMIT&timeInForce=GTX` 로 보냄(rp 집행 8사례 URL 글자까지 · mesh 서버가 GTX 아닌 POST 면 panic) · 익절 GTX `place_bracket`·`place_tp_ladder`(rp) · 예외 dbad GTC = `place_add_order`(M12, 원본 그대로 · 사용자 결정 목록) |
| E2 | 옮김 | `http::Binance::signed`(예외 대신 `{"__error__"}` · 네트워크·시간초과·JSON 오류 = 숫자로 시작 안 함) · `sign::http_error`("<상태> [<code>] msg" · code 없음/null · msg 없음 = dict repr · dict 아님 = AttributeError) · 418/429 `sign::ban_note`(기록만, 주문·취소는 막지 않음) · aiohttp 의 멱등 메서드 한 번 재전송까지 · rp(signed 32사례 · 밴 기록 3) · mut «code 버림» «msg repr» «429 빠짐» «재전송 없음/POST 재전송» |
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
| E19 | 옮김 | `PEG_TAG`·`WAIT_TAG`(exec) · `bracket::{TP_TAG,SL_TAG,ADD_TAG,bracket_tag}` · `cancel_bracket` 은 표식+측면 글자로만 · 스탑은 그 심볼(칸 없으면 이 심볼) · only_sl/only_tp · 고아 정리는 dbwt·dbpg 만 · inv(wait·d2) · rp(cancel_bracket 7 · 고아 정리 2) · mut «only_tp 인데 물타기도» «다른 코인 스탑» «고아 정리가 dbtp 도» |
| E20 | 옮김 | `run_entry_wait` · inv `wait_paths`(재호가 없음·-5022 재시도·거래소 취소·on_fill 한 번) · mut «on_fill 매번» «거래소 취소» |
| E21 | 옮김 | `run_entry_limit`(따라가지 않음 · -5022 는 수정 직후 3회만) · inv `limit_paths` |
| E22 | 옮김 | 수정은 체결 0 일 때만 · inv `limit_paths`(부분 뒤 수정) · mut «부분 뒤에도 수정» |
| E23 | 옮김 | 자동 재조정(20초·1bp·50회·체결 시작 뒤 멈춤) · inv `limit_paths` |
| E24 | 옮김 | 취소 확인 중 체결 몫 on_fill · inv `limit_paths`(on_fill 한 번) |
| E25 | 옮김 | `exec::run_entry_stop`(닿기 전 주문 0 · 1초마다 touched(since) · 선 옮기면 since 를 지금으로 · 4시간 만료 · 닿으면 `run_entry(maker_only)` → 잔량 버림 · PEG = 지금 닿음 · TAKER = 미정 M5 · 닿은 뒤 호가 실패 = error(D11)) · inv `e25_stop_entry_paths`(9사례) · mut «since 그대로» «maker_only 빠짐» «M5 빠짐» · ⚠수정 API 가 plan 을 바꾸는 몫은 amend 칸 수량을 계획 사본에 싣는 것으로 갈음(exec.rs ponytail) |
| E26 | 옮김 | `run_entry_limit` PEG(취소 확인 → action 비움 → 따라가기 → 잔량 버림) · inv `limit_paths`(PEG 두 사례) |

## 4. 대기 주문 W1–W12

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| W1 | 옮김 | `Watcher::waits` · `wkey`·`wasset`·`wslots`(칸 번호 순)·`wait_slot`(setdefault) · wf(감시가 같은 쪽 대기를 거둘 때 · 재시작 기록 칸) · 칸을 여는 접수는 접수 |
| W2 | 부분 | 상수 `WAIT_SLOTS_MAX`=10 · 10개 넘으면 거절은 접수 |
| W3 | 부분 | `LIMIT_MAX_SEC`(4시간) 만료 취소 · inv `limit_paths`(만료) · `e25_stop_entry_paths`(감시 진입 만료) · 기다리기 마감 잔량 시장가는 미정 M3 |
| W4 | 부분 | `Watcher::limit_save`·`limit_path`(심볼·칸 _n · 임시 파일 → 교체 · 못 쓰면 로그만) · wf `limit_save_writes` · 왕복 wf `limit_file_round_trip` + rt(Rust 가 쓴 복구 파일을 파이썬 limit_recover 가 읽음) · «정상 종료에서만 지움»은 대기 관리(entry_then_bracket) |
| W5 | 옮김 | `Watcher::limit_recover`(6시간 · 계좌 못 읽음 유지 · 포지션 있고 무장 없으면 저장한 선으로 무장하고 placed 확인 · 고아 실패면 유지 · 예외 = «복구 실패» 유지) · wf `restart_cleanup_and_recover`·`restart_orphan_errors_keep_file`·`restart_account_fails_and_cleanup_raises`·`restart_already_armed_or_no_position` · mut «고아 실패여도 지움» «6시간 넘어도 복구» |
| W6 | 범위 아님 | 접수(수정 API) · 집행 쪽 «부분 뒤 수정 버림»은 E22 |
| W7 | 범위 아님 | 접수 |
| W8 | 범위 아님 | 접수 |
| W9 | 옮김 | `bracket_tick`(clear·fire 때 같은 심볼·같은 쪽 대기에 action=cancel · 다른 코인·반대쪽은 둠 · 분할 손절(sl_leg)엔 안 함 = 원본) · wf `position_gone_clears_and_cancels_same_side_wait` · mut «같은 쪽 대기 취소 빼기» |
| W10 | 부분 | 무장 함수 `keep_rearm`·`plan_from_current`·`arm_bracket` 은 옮김(wf) · 첫 체결 순간 부르는 고리(entry_then_bracket.first_fill)는 대기 관리 |
| W11 | 부분 | `plan_from_current`(지금 저장된 선 · fired·조각·exiting 이면 None) · wf `plan_from_current_variants`·`plan_from_current_fired_none` · entry_then_bracket 의 «mod·sl_off·tp_off·fired 면 덮지 않음» 분기는 대기 관리 |
| W12 | 옮김 | `exec::Shared`(집행 루프와 API 가 같은 칸) · inv(가짜가 시각표대로 action·amend 를 넣는다) |

## 5. 브래킷 B1–B33

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| B1 | 옮김 | `watch::bracket_load`(못 읽으면 {} · 옛 키 옮김)·`bracket_save`(임시 파일 → 교체) · 양방향 왕복: 파이썬이 쓴 원문 95개 → Rust 가 같은 값으로 읽음(wf) · Rust 가 다시 쓴 95개 + 대기 복구 파일 1 → 파이썬 bracket_load·limit_recover 가 형·repr 까지 같은 값으로 읽음(rt 96/96 · 까다로운 값 사례: 0.1+0.2·1e-07·1e+16·-0.0·2^53+1·5e-324·최대 실수·이모지·이스케이프·옛 키) · 🔴NaN·Infinity 는 파이썬만 읽는다(아래 «새로 본 것») |
| B2 | 옮김 | `bracket_key`·`bracket_rekey` · fx · wf `legacy_key_migrates_on_load`(옛 키 무장이 발동 · 저장 때 새 키로) · `legacy_and_new_key_collide`(같은 칸이면 파일 순서상 뒤의 것 = 파이썬 dict -- serde_json preserve_order 라 같다) |
| B3 | 부분 | `arm_bracket`(결과 칸 state["bracket"] 은 예외여도 채움 · 원본 칸 순서·값 그대로 무장 기록) · wf `arm_bracket_*` 4사례 · «체결이 있으면 모든 종료 경로»는 대기 관리(entry_then_bracket) |
| B4 | 옮김 | `bracket::place_bracket`(포지션 새로 읽어 그 측면 전량 · 옛 우리 주문 정리 → TP GTX + 비상 스탑 · strict · tp·스탑 없으면 placed=true(원본 all([]))) · rp(11사례: 정상·조회 실패·포지션 없음·익절 거부·익절 불명·strict·정리 실패 뒤 계속·숏 익절만·스탑만·둘 다 없음·키 없음) · mut «strict 무시» · 🔴불명 응답 조회 없음(원본 그대로, 남은 위험) |
| B5 | 옮김 | `bracket::post_stop` → `/fapi/v1/algoOrder` CONDITIONAL STOP_MARKET closePosition=true MARK_PRICE · rp(URL 글자까지) · mut «MARK_PRICE → CONTRACT_PRICE» · ⚠쓸지는 미정(시장가형 -- M10, 사용자 결정 목록) |
| B6 | 옮김 | `arm_rule_add`(스탑 placed 아니면 안 건다 · 원본 인자 순서·filters 는 검사 뒤에 읽음) · wf `arm_bracket_rule_first_with_add`·`arm_bracket_rule_stop_failed_no_add` · mut «스탑 실패여도 물타기» · 거는 함수 place_add_order 는 GTC(M12) |
| B7 | 옮김 | `line_hit`(tp_resting 아님 → 서버가 닫음) · fx · sc · 닫기 실행 `bracket_tick` · wf `lines_tp_touch_not_resting`·`lines_tp_touch_resting_holds`·`tp_leg_not_resting_touched_fires_full`·`tp_leg_resting_touched_holds` |
| B8 | 옮김 | `Watcher::watch_brackets(period)`(nap 먼저 → 틱 · 틱 오류는 «bracket watch: …» 로그만 · nap=false 면 멈춤 = 원본 CancelledError) · 주기 주입 가능(`WATCH_PERIOD_SEC`=2 기본) · wf `watch_loop_period_and_wake`(알림에 3초에 일찍 깸 · 계좌 예외 틱 뒤 계속)·`watch_loop_fires_on_second_tick` · wi `watch_loop_survives_every_failure_and_uses_injected_period` · mut «틱 실패에 멈춤» «주기 주입 무시» |
| B9 | 옮김 | `bracket_action` clear(무장+5초 · ok · 0) · fx(armed_suite 280) · sc · mut «+5초 → +0» |
| B10 | 옮김 | clear 정리 실패 → 무장 남김 + retry_after 30초 · wf `position_gone_clear_fails_retry_30s`·`filled_but_cleanup_fails_then_clear` · mut «정리 실패에도 무장 내림» |
| B11 | 옮김 | `bracket_sl_hit`(마감 5분봉 종가 · watch_from) · fx · sc · 입력 고르기(time+300 ≤ 지금인 봉 중 마지막 · 못 읽으면 None + 로그) = `bracket_tick` · wf `rule_*` 10사례(정확히 같은 값·한 틱 차이·무장 전 봉·꼬리만·sl_price 대체·미마감) · mut «미마감 봉도 씀» |
| B12 | 옮김 | `bracket_action` 선 분기 · `line_hit`(손절 먼저) · `sl_legs_hit` · fx · sc · 입력(무장·watch_from 뒤 1초 가격 범위) = `bracket_tick` · wf `lines_*` 14사례 · mut «닿음 >= → >»(fx) «무장 전 1초 가격도 씀»(wf) |
| B13 | 부분 | `sl_up_of`·`line_bracket(sl_watch)` · fx · sc · check_lines 는 접수 |
| B14 | 옮김 | `bracket_tick` 선 발동(exiting 먼저 → 익절 지정가 먼저 지움 → 실패면 10초 · 지웠으면 fired 를 곧바로 파일) · wf `lines_long_touch_exact`·`lines_tp_cancel_fails_waits_10s`(호출 열의 run_exit 에 그 순간 저장본을 함께 적어 «fired 가 청산 전에 파일에»를 대조) · wi 재시작(청산 도중 죽어도 fired 가 파일에) · mut «10초 → 30초» «fired 를 청산 전에 안 씀» |
| B15 | 옮김 | 발동 직전 armed_at·mod.at 다시 비교 · wf `line_moved_before_fire_skips` · mut «mod.at 비교 빼기» |
| B16 | 옮김 | 조립 뒤 sl_off 다시 봄(선 무장의 익절 fire 는 예외) · wf `disarm_during_plan_rule`·`disarm_before_fire_lines`·`sl_leg_disarmed_during_plan` · mut «조립 뒤 손절 해제 검사 빼기» |
| B17 | 옮김 | no_position 이면 정리하고 내림(실패면 남김) · 그 밖 오류·blocked 는 «닫힘» 아님 → 30초 · last_error = detail or error or blocked · wf `plan_no_position_clears`·`plan_no_position_clear_fails`·`plan_account_unavailable`·`plan_exception_502`·`plan_blocked`·`account_not_ok_sl_hit_plan_fails` · mut «조회 실패도 닫힘으로» |
| B18 | 옮김 | 발동 청산 = `WatchIo::run_exit` → 실제 구현 `watch::run_exit_into`(exec::run_exit) · 다 닫혔을 때만(filled_*) 정리 · 실패면 무장 유지 30초 · 예외 = settle → error · wf `exit_{rejected,error,taker_failed}_retry_after_30s`·`exit_raises_settle_error`·`exit_needs_market_python_taker` · wx(실제 집행: 메이커 체결 · 취소 확인 실패 error) · mut «재시도 시각 안 둠» · 🔴시장가 몫(M6·M7·M9)은 미정 → `market_wait`(wx · 아래) |
| B19 | 옮김 | `bracket_merge`(OWNED_BY_SAVED · 쏜 틱의 tp_resting) · fx(92) · sc · mut «tp_resting 규칙 뒤집음» |
| B20 | 옮김 | 판정 `sl_legs_hit`·`bracket_action`→SlLeg(fx · sc) · 실행 `Watcher::fire_leg`(쏜 조각 먼저 파일에 → 비율 청산 → 성공이면 쏜 조각 빼고 set_sl_legs → 익절 다시 걸기 · 실패면 30초·표시 유지) · wf `sl_legs_one_then_all`·`sl_leg_exit_fails_keeps_fired`·`sl_leg_lock_held_skips`·`sl_leg_disarmed_during_plan`·`sl_leg_with_tp_ladder_rearm` · mut «쏜 조각 표시 안 함» «비율 분모 무시» |
| B21 | 부분 | 거래소 쪽 = `cancel_bracket(only_sl)`·`clear_bracket` · rp · 판정 sl_off 는 `bracket_action`(fx) · 해제 API·«실패면 아무것도 안 바꿈»은 접수 |
| B22 | 부분 | 규칙 무장 다시 켜기의 거래소 쪽 = `bracket::place_stop`(포지션 확인 → 남은 우리 스탑 정리 → 못 지우면 안 걸음 → 스탑 · -2021 그대로 실패) · rp(6) · mut «옛 스탑 안 지움» · 선 무장 감시 재개·409 는 접수·감시 |
| B23 | 옮김 | `bracket_action` sl_off + fired "tp"/True · fx · sc · 실행 wf `lines_sl_off_tp_watch_fires_tp`(fired="tp")·`lines_fired_true_sl_off_holds` |
| B24 | 부분 | 끄기의 거래소 쪽 = `cancel_bracket(only_tp)`(tp_open 함께) · rp · `tp_watch_only`·켜기 검사는 접수·감시 |
| B25 | 범위 아님 | 접수·감시 |
| B26 | 범위 아님 | 접수·감시 |
| B27 | 옮김 | `split_tp_qty` · fx(194) · sc · mut «carry 버림» |
| B28 | 옮김 | `bracket::place_tp_ladder`(옛 익절 다 지운 뒤에만 · 사라진 조각은 직접 조회해 FILLED 일 때만 체결 · 필터는 조각이 없어도 읽음 · split_tp_qty) · rp(9사례: 정상·사라진 조각 체결/취소·전부 체결·먼지 합침·정리 실패·필터 예외·조회 실패·포지션 없음·키 없음) · mut «조회 없이 체결로» · `ladder_apply`(O) 는 접수 |
| B29 | 부분 | `watch::set_tp_legs`(다 빠지면 tp_resting False · 하나면 단일 칸으로)·`set_sl_legs`·`tp_watch_only` · wf `sl_leg_with_tp_ladder_rearm`·`sl_leg_tp_ladder_partial_cancel_watch_only` · mut «일부 취소 실패를 서버 감시로 안 넘김» · 사용자 스트림의 dbtp FILLED 반영(tp_leg_filled)은 계좌·알림 |
| B30 | 옮김 | `Watcher::partial_exit_rearm_tp`(조각 = place_tp_ladder · 단일 = place_bracket · 서버 감시·fired·tp_off·exiting 제외 · 그 사이 새 무장이면 안 덮음 · 조회 실패면 상태 그대로 · 예외는 tp_rearm 에) · wf `partial_exit_rearm_tp_single`·`…_lookup_fails_keeps_state`·`…_skips`·`…_rearmed_meanwhile`·`sl_leg_tp_ladder_raises`·`exit_then_rearm_partial` · mut «strict 로 몰래 고침(D7)» · D7 원본 그대로(strict=false) |
| B31 | 옮김 | `Watcher::keep_rearm`(무장 없으면 arm_bracket · 손절 조각 청산 중이면 건너뜀 · 아니면 익절 수량만) · wf `keep_rearm_paths`·`keep_rearm_sl_leg_fired_skips` · ponytail: 포지션별 asyncio.Lock 대신 `&mut self` 가 한 줄로 세운다 |
| B32 | 부분 | 청산 진행 표시 `Watcher::exiting`(감시가 첫 await 전에 넣고 끝나면 뺀다 · 예외로 새면 남는다 = 원본) · wf `exiting` 칸 대조 · 거절하는 API 는 접수 |
| B33 | 범위 아님 | 계좌·알림 |

## 6. 청산 C1–C6

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| C1 | 부분 | 감시 청산은 fresh 계좌로 조립을 부른다(wf 호출 열 fresh=True) · 조립·미리보기·제출은 조립·접수 |
| C2 | 부분 | `Watcher::exit_then_rearm`(run_exit → partial_exit_rearm_tp · settle 밖 = 원본) · wf `exit_then_rearm_partial` · 부분/전량 갈림은 접수 |
| C3 | 범위 아님 | 접수(스위칭 -- 두 다리 다 시장가 = 미정 M11) |
| C4 | 범위 아님 | 접수 |
| C5 | 범위 아님 | 접수 |
| C6 | 범위 아님 | 접수 |

## 7. 재시작·종료·오류 R1–R9

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| R1 | 옮김 | `bracket::cancel_wait_orphans`(dbwt·dbpg 만 · rp 2 · mut «dbtp 도») + 재시작 훅 `Watcher::wait_cleanup`(지운 주문은 그 코인 대기 칸 · 정리 실패는 eth 칸 · 예외면 errors 에 · restart_cancelled 기록) · wf `restart_*` 4사례 · wi `restart_mid_exit_keeps_arming_and_next_process_finishes` · 훅을 «스위치 켜진 곳에서만» 띄우는 것은 G4(접수) |
| R2 | 옮김 | `wait_cleanup` 이 정리 뒤 복구 파일(이름순)마다 `limit_recover` · 결과를 그 칸 recovered 에 · wf `restart_cleanup_and_recover`(6시간 넘은 SOL 파일 지움 + ETH 무장) |
| R3 | 부분 | 감시가 부르는 청산의 settle(예외 → phase=error «집행 오류 … — 거래소에서 직접 확인하세요» · done_at) = `Watcher::settle` · wf `exit_raises_settle_error` · 진입·대기·청산 버튼의 settle 은 접수 |
| R4 | 부분 | 취소 확인 실패·지정가 접수 불명·PEG/감시 진입 호가 실패 → `Phase::Error` · inv03·05·07·d2·e25 · mesh(HTTP 로도 같은 error) · API 키 없음 = `Binance::has_keys`(브래킷 함수들이 원본 문구로) -- 집행 함수(run_*)의 키 검사는 호출부 · 스위칭은 접수 |
| R5 | 범위 아님 | 접수 |
| R6 | 범위 아님 | 접수 |
| R7 | 옮김 | `Binance::nap` = tokio select(잠 · 알림) -- future 를 버리면 바로 끝나 «깨움·취소가 같은 틱이면 취소를 삼킴»(gh-86296)이 생길 자리가 없다 · mesh `nap_wakes_early_on_fill_alert` |
| R8 | 범위 아님 | 대기 관리 |
| R9 | 옮김 | 진행 상태(entry·side_exits·waits·exiting)는 `Watcher` 메모리 -- 재시작하면 사라지고 무장 파일이 감시를 잇는다 · wi `restart_mid_exit_keeps_arming_and_next_process_finishes`(청산 도중 죽음 → 새 프로세스: 고아 정리 → fired 래치로 익절 취소 없이 바로 청산 → 닫힘 → 무장 내림) · wf `lines_fired_latch_after_restart` |

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
| A10 | 부분 | 알림은 앞당기기만: `nap` 으로 일찍 깨도 3초 판단 그대로(inv `e16_…`) · 감시 루프도 알림에 일찍 깨되 주기는 그대로(wf `watch_loop_period_and_wake` · wi) · `Binance::wake`(Notify) · mesh `nap_wakes_early_on_fill_alert` · 알림 수신은 계좌·알림 |
| A11 | 범위 아님 | 계좌·알림 |

## 집계

| 상태 | 첫 조각 | 둘째 조각 뒤 | 셋째 조각 뒤 |
|---|---|---|---|
| 옮김 | 38 | 47 (+E1·E2·E19·E25·R7 부분→옮김 · +P29·B4·B5·B28 범위 아님→옮김) | **68** (+G6·G9·W1·W5·W9·B1·B6·B8·B10·B14·B15·B16·B17·B18·B30·B31·R2·R9 범위 아님→옮김 · +B7·B20·R1 부분→옮김) |
| 부분 | 22 | 21 (−5 위 · +P16·B22·B24·R1 범위 아님→부분 · B21 은 부분 그대로, 거래소 쪽 몫 추가) | **31** (+G4·G5·G8·W2·W4·W10·W11·B3·B29·B32·C1·C2·R3 범위 아님→부분 · −B7·B20·R1) |
| 범위 아님 | 85 | 77 | **46** |
| 미정(사용자 결정) | 0 | 0 -- 규칙 줄 자체는 없음. 시장가 분기 13곳은 아래 따로(B5 는 «옮김 · 쓸지 미정») | 0 -- 감시가 닿는 시장가 자리(M6·M7·M9)는 아래 표에 `market_wait` 로 |
| 옮긴 몫에 시험 없음(⚠) | 4 | 2 (E17 구조 · B2 충돌 순서) -- E1 은 mesh 단언, R7 은 mesh 시험으로 덮음 | 1 (E17 구조) -- B2 충돌 순서는 wf `legacy_and_new_key_collide` 로 덮음 |

남은 «범위 아님» 46줄(다음 조각 범위표): **접수 문** G1·G2·G7·G10·G11·G12·G13·G14·G16·G18 (10) · **계획 조립** P2·P3·P4·P5·P7·P19·P21·P22·P23·P25·P26·P28·P30 (13) · **대기 주문** W6·W7·W8 (3) · **브래킷(접수 API)** B25·B26·B33 (3) · **청산** C3·C4·C5·C6 (4) · **재시작·종료** R5·R6·R8 (3) · **계좌 스트림** A1–A9·A11 (10).
남은 «부분» 31줄 중 다음 조각 몫: 접수 G3·G4·G5·G8·G15·G17·W2·W6 쪽 · 조립 P1·P6·P12·P14·P16·P17·P20·C1 · 대기 관리 W3·W4·W10·W11·B3 · 접수 API B13·B21·B22·B24·B32·C2·R3·R4 · 계좌·알림 B29·A10 · 시장가 E10.

## 시장가 분기 M1–M13 (모두 미정 -- 사용자 결정 대기 · 옮기지 않음)

| # | Rust 에서 닿는 자리 | 결과 |
|---|---|---|
| M1 | `exec::run_entry` 따라가기 마감 잔량 | `NeedsMarketDecision` · `market_branch="M1"` · inv02·10·e18 음성 대조 |
| M2 | `exec::run_entry` 계획 `market=true` | 같은 갈래 "M2" · inv `maker_only_and_market_entry_plans` |
| M3 | `exec::run_entry_wait` 마감·action=market | "M3" · inv `wait_paths` |
| M4 | `exec::run_entry_limit` action=market(취소 확인 뒤) | "M4" · inv `limit_paths` |
| M5 | `exec::run_entry_stop` 닿기 전 action=market | `run_entry(market)` 뒤 `market_branch="M5"` · inv `e25_stop_entry_paths` |
| M6·M7 | `plan::build_exit_plan` 이 계획을 MARKET 으로 만든다(원본 값 그대로 -- 정답지 일치) → `exec::run_exit` 가 "M6/M7/M8" 로 멈춘다 · **손절 감시가 닿는 자리**: `watch::run_exit_into` → 칸 phase=needs_market_decision · 무장에 `market_wait` · 다시 안 쏨 | inv09 · fx · wx `stop_exit_high_vol_plan_is_market_m6_no_order_sent`·`stop_exit_dust_position_is_market_m7_no_order_sent`(거래소 주문 0) · mut «결정 대기를 체결로 봄» |
| M8 | 조립(`market=True`) -- 집행에서는 M6/M7 과 같은 갈래 | inv09 |
| M9 | `exec::run_exit` 마감 잔량 · **손절 감시가 닿는 자리**: 선·규칙·분할 손절 발동 청산의 따라가기 마감 뒤 잔량 → `market_wait`(branch M9 · 남은 수량) · 다음 틱부터 다시 안 쏨(결정 대기) | "M9" · inv02 · wx `stop_exit_deadline_remainder_is_market_branch_m9_and_waits` · mut «결정 대기 중 다시 쏨» |
| M10 | 비상 스탑 `bracket::post_stop`(place_bracket·place_stop 이 부름) -- **요청 모양만 옮겨 정답지와 맞춤**, 시장가형(발동 시 전량 시장가) | 쓸지 사용자 결정 · rp |
| M11 | 스위칭 -- 접수 조각 | -- |
| M12 | `bracket::place_add_order` GTC(원본 그대로 옮김 -- 이미 그 아래면 테이커 체결) · 부르는 쪽(arm_rule_add)은 감시 | 쓸지 사용자 결정 · rp · mut «GTC → GTX» |
| M13 | 대기 동작 접수 -- 접수 조각 | -- |

## 원본과 일부러 다르게 둔 곳

- 시장가 13곳 → 위 표(보내지 않고 멈춤).
- 손절 감시(셋째 조각): 발동 청산이 시장가 분기에 닿으면 파이썬은 시장가로 닫고 무장을 내린다 -- Rust 는 칸 `needs_market_decision` + 무장 `market_wait`(branch·qty·at·문구) + `last_error` 를 남기고 **그 무장은 다시 쏘지 않는다**(포지션이 사라지면 clear 는 그대로). 30초 뒤 다시 따라가게 두면 «조용히 지정가로 바꾸기»라서. 결정이 나면 이 칸을 지우는 쪽(접수)이 정한다. 파이썬은 이 칸을 모르고 무시한다(전환 순간 파일을 넘겨받으면 파이썬은 다시 쏜다).
- 감시 바깥(계좌·5분봉·1초 가격·청산 계획 조립·거래소 함수·잠금·스위치·시계)은 `WatchIo` 뒤 -- 실제 구현은 다음 조각(조립·계좌). `run_exit` 의 실제 몫만 `watch::run_exit_into` 로 있다(끝난 뒤 칸에 한 번 옮김 -- 도는 동안 화면은 "submitting").
- `keep_rearm` 의 포지션별 asyncio.Lock → `&mut self`(한 줄로 세움). 저장본 글자(구분자·실수 표기)는 다르고 뜻은 같다(B1 왕복). 원본이 dict 아닌 JSON 저장본에서 AttributeError 로 죽는 자리는 {}.
- `run_entry`·`run_entry_wait`·`run_entry_limit`·`run_exit` 의 «API 키 없음 → error» · `_clock_offset` · `ensure_leverage` 호출 → 접수 조각(함수 자체는 `http`·`bracket` 에 있다).
- 주문 응답이 dict 가 아니거나 executedQty 가 숫자가 아니면 원본은 예외(settle → error), Rust 는 숫자로 시작하지 않는 오류 문자열(접수 = «나갔는지 모름», 조회 = «재시도») -- `http::reply` ponytail.
- JSON 해석 오류 문구(JSONDecodeError 뒤)는 파이썬과 다르다(앞머리만 같다) · 밴 기록의 429 해제 시각은 실제 시계라 정답지와 폭만 비교.
- 원본은 키를 호출마다 환경에서 읽는다 -- `Binance::from_env` 는 만들 때 한 번.
- 파이썬 예외(0 나누기·inf→int·빈 목록 인덱스) → 같은 종류의 `PlanError`(정답지가 종류·문구까지 비교).
- ~~D2(«나갔는지 모르는 주문» 처리가 세 경로에서 다름)는 통일하지 않았다~~ → **해결(2026-11-20, 사용자 결정)**: 파이썬 원본을 먼저 고치고(`resolve_unknown_post` 공용 함수) Rust 도 같게 맞췄다 -- 원본과 다른 곳이 아니다. inv `d2_unknown_outcome_same_on_all_paths` 가 «세 경로가 같다»를 고정한다.

## 셋째 조각에서 원본에 새로 본 것(고치지 않음 · 시험이 원본 동작을 고정)

- 무장 한 칸이 망가지면(예: `symbol` 없음) 틱이 매번 KeyError 로 죽어 **다른 코인 손절 판정 전체가 멈춘다**(파일 순서상 앞이면) -- wf `corrupt_entry_blocks_all_coins`.
- 주문 잠금 중에도 규칙 무장 발동은 청산 계획 조립(새 계좌 조회 · 무게 ~50)을 매 2초 하고 버린다(잠금 검사가 조립 뒤) -- wf `lock_held_rule_D9`. 선 무장은 조립 전에 본다.
- 실행 스위치는 틱 시작에만 본다 -- 틱 도중 꺼져도 그 틱의 청산은 나간다 -- wf `exec_switch_off_during_tick_still_fires`.
- 저장본은 json 기본값이라 NaN·Infinity 도 쓰고 읽는다. serde_json 은 못 읽는다(Rust 는 {} = 감시 멈춤). 지금 쓰는 쪽이 유한값만 쓰는지는 이번 범위(arm_bracket·감시·복구)만 확인.
- 선 발동에서 익절 취소가 **예외**로 새면 `bracket_exiting` 에서 안 빠진다(그 키의 선 수정·익절 켜기가 재시작까지 거절) -- Rust 도 같게 둠.
- 헤지 롱·숏이 한 틱에 같이 닿으면 둘째 청산이 같은 위 칸을 `clear()` 해 첫 청산 결과가 화면에서 사라진다 -- wf `hedge_long_short_both_fire`.
