# rtorder 규칙 추적표 (2026-11-21 · 첫~일곱째 조각 + «sltp 만» 정리)

점검표 = `docs/rtcore_order_rules_inventory_20261120.md`(145개). 첫 조각 = 계획 순수 함수(`src/plan.rs`) + 지정가 집행 상태 기계(`src/exec.rs`). 둘째 조각 = 서명·요청 문자열(`src/sign.rs`) + 전송(`src/http.rs`, `Exchange` 구현) + 손절·익절·레버리지·고아 정리(`src/bracket.rs`) + 감시 진입(`exec::run_entry_stop`). 셋째 조각 = 손절·익절 감시와 주문 상태 칸(`src/watch.rs` -- orders.py 의 bracket_tick·watch_brackets·partial_exit_rearm_tp·arm_bracket·plan_from_current·keep_rearm·limit_recover·wait_cleanup·exit_slot·busy_or). 어디에도 연결하지 않았다.
넷째 조각 = 계좌 읽기·사용자 스트림·스트림 계좌(`src/account.rs`) + 계획 조립과 «받아 둔 값» 시세 원천(`src/assemble.rs`) + 접수 문(`src/intake.rs`, HTTP 와 뗀 핸들러 10개) + 감시의 계좌·조립·집행 자리 실제 구현(`src/wire.rs`). 역시 어디에도 연결하지 않았다.

상태: **옮김** = Rust 에 있고 시험이 지킨다 · **부분** = 규칙의 일부만 이 조각에 있다(나머지 몫과 조각을 적음) · **범위 아님** = 다음 조각 · **⚠시험 없음** = 옮긴 몫에 지켜 주는 시험이 없다.
시험 약어: `fx` = tests/plan_fixture.rs(파이썬 정답지 2,266건) · `sc` = tests/plan_selfcheck.rs(원본 자체점검 이전) · `inv` = tests/exec_invariants.rs · `rp` = tests/http_replay.rs(원본 파이썬 + 진짜 aiohttp 정답지 112시나리오 -- 요청 열 글자까지·결과·밴 기록·연결 수) · `mesh` = tests/http_mesh.rs(가짜 거래소를 HTTP 뒤에) · `mut` = tools/mutate.py(양성 대조) · `wf` = tests/watch_fixture.rs(원본 orders.py 감시 클로저 정답지 97사례·120틱·호출 419 -- 호출 열·저장본·상태 칸·로그·예외 · 양방향 왕복) · `wi` = tests/watch_invariants.rs(퍼즈 300×20틱 · 루프 · 재시작 · busy_or) · `wx` = tests/watch_exec.rs(감시 → exec::run_exit → 가짜 거래소 · 시장가 분기) · `rt` = tools/check_watch_roundtrip.py(Rust 가 쓴 파일을 파이썬 읽는 쪽으로 96/96). 감시 양성 대조 = mutate.py `watch` 27/27(wf 22 · wx 2 · wi 3).
여섯째 조각 시험: `re` = tests/runner_e2e.rs(접수 → 실행기 → 무장 → 감시 → 청산 · 실패 주입) · `us` = tests/userstream_ws.rs(127.0.0.1 HTTP·웹소켓) · `ah` = tests/account_http.rs(동시 계좌 조회 대 원본).
일곱째 조각 시험: `ff` = tests/flow_fixture.rs(원본 «전체 흐름» 정답지 tools/gen_flow_fixture.py -- 진짜 orders.entry_then_bracket·bracket_tick + 진짜 집행·브래킷 함수 + aiohttp 를 127.0.0.1 모의 거래소에 · 6시나리오 13걸음 · 요청 224(서명 URL 144 글자까지)·상태 칸·무장 파일·복구 파일·가짜 의존성 호출·로그·끝 시각).
넷째 조각 시험: `dk` = tests/desk_fixture.rs(원본 파이썬 정답지 tools/gen_desk_fixture.py -- 계좌 19 · 스트림 8 · 계획 64 · 접수 85(그중 시장가 미정 거절 6) · (가)=(나) 149사례 · 낡음 7 · 스트림 계좌 규칙 · 캐시 규칙) · `we` = tests/wire_e2e.rs(감시 → 계좌 → 조립 → 집행 5). 양성 대조 = mutate.py `desk` 30 + `wire` 2.
다음 조각 이름: **접수**(라우트·confirm·게이트·잠금·칸) · **조립**(assemble_entry/exit_plan·계좌·풀·basis) · **감시**(bracket_tick·무장 파일·arm_bracket 등 orders.py 쪽) · **대기 관리**(대기 칸·복구 파일·재시작 훅) · **계좌·알림**(사용자 스트림).

## 1. 접수 문 G1–G18

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| G1 | 부분 | `intake::POST_ONLY` 표(주문을 바꾸는 8개 핸들러) -- 라우트에 강제하는 것은 웹 서버 조각 |
| G2 | 옮김 | 주문을 바꾸는 핸들러 14개 전부 confirm=1(넷째 10 + 여섯째 익절 on/off·분할 조각·선 추가·선 수정) · `dk` *_no_confirm |
| G3 | 옮김 | `DeskIo::exec_enabled`(매 요청) · 403 = 핸들러 14개 · 미리보기 dry_run · `dk` |
| G4 | 옮김 | 감시 틱 정지 = `Watcher::bracket_tick`(스위치 꺼짐이면 계좌도 안 읽음 · 무장 파일 그대로) · wf `exec_switch_off`·`exec_switch_off_during_tick_still_fires`(틱 도중 꺼져도 그 틱은 끝까지 = 원본) · wi(퍼즈: 꺼짐이면 호출 0·저장본 불변) · mut «스위치 꺼져도 감시» · 시작 게이트(2026-11-21 계좌 알림 조각): `app::run` 이 원본 기동 훅처럼 시작 정리(wait_cleanup)는 스위치 · 사용자 스트림·갱신기는 스위치+키일 때만 띄움 · `ar` binary(스위치 없는 기동 = 정리·스트림 없음) · mut «실행 스위치 꺼져도 시작 정리» |
| G5 | 옮김 | `api_entry_submit`(진입 칸 하나 · 같은 코인 청산 칸도 봄) 409 · `dk` sub_entry_busy·sub_side_exit_busy · mut «같은 코인 청산 칸이 바빠도 진입» |
| G6 | 옮김 | `Watcher::exit_slot`(코인 칸을 만들어 둠 · 같은 코인 사용 중 None) · wf `two_coins_other_entry_busy_uses_side_slot`·`same_coin_entry_busy_skips`·`same_coin_side_slot_busy_skips` · wi(퍼즈: ETH 칸이 바쁘면 ETH 청산 0) · mut «같은 코인 칸이 바빠도» |
| G7 | 옮김 ⚠시험 없음 | `api_exit_submit`·`api_entry_submit`(대기) 조립 뒤 칸 다시 봄 -- 단일 실행 정답지라 «그 사이 바뀜»을 못 만든다(웹 서버에 붙일 때 동시 요청 시험) |
| G8 | 부분 | `watch::busy_or`(wi) + `intake::LOCKED` 표(잠금 안 4 · 밖 4 · 미리보기 2) -- 감싸는 것은 웹 서버 |
| G9 | 옮김 | `bracket_tick` 의 세 자리(선 발동 전 · 조립 뒤 · 분할 손절) -- 원본 그대로 조립(계좌 새로 읽기)은 잠금 중에도 한다(D9) · wf `lock_held_rule_D9`·`lock_held_lines_D9`·`lock_taken_during_plan`·`sl_leg_lock_held_skips` · wi(퍼즈: 잠금 중 청산 0 · 잠금으로 미룬 틱 813) · mut «선 발동 전 잠금 무시»(wf) «조립 뒤 잠금 무시»(wi) |
| G10 | 옮김 | `api_entry_submit` 추가 진입 대기 여럿(모두 같은 쪽 add_keep 지정가·감시 · 10개 미만) · `dk` sub_wait_pending·sub_limit_second_add_ok·sub_limit_second_not_add |
| G11 | 옮김 | dup(같은 종류·같은 선 1e-9) · `dk` sub_limit_dup_refused · mut «같은 선 중복 허용» |
| G12 | 옮김 | `Desk::running`(아직 도는 작업) 칸 재사용 안 함 · `dk` sub_limit_slot_task_running·sub_limit_all_slots_running · mut «도는 작업이 있는 칸 재사용» |
| G13 | 제거(사용자 결정 2026-11-21 sltp 만) | 스위칭 대기 409 -- 스위칭째 지움(원본 PR 61) · 옛 인자는 400 removed_option(`dk` xs_removed_*) |
| G14 | 옮김 | `intake::query_asset`(주문 코인 표만 · skhynix·mu 막힘) · `side_symbol`(손절 조작 심볼) · `dk` sub_bad_asset·dis_bad |
| G15 | 옮김 | `query_fraction`·`query_leverage`(범위 밖·글자 = None → 모델값 · inf = 원본처럼 500)·`query_rule`·`query_lines`·`query_limit`·`query_wid`·wait_min · `dk` 400 사례 14 · ponytail: «1_000» 밑줄 표기는 파이썬만 |
| G16 | 제거(사용자 결정 2026-11-21 sltp 만) | 스위칭 전량+반대 검사 -- 스위칭째 지움 · switch_to·sw_* = 400 removed_option |
| G17 | 제거(사용자 결정 2026-11-21 sltp 만) | 자동 규칙 코인·권장 배수 15분·교차 마진 검사 -- 자동 규칙째 지움 · `rule` 인자 = 400 removed_option · `Quotes::margin_type`·`bracket::margin_type`·`DeskIo::vol_mult` 삭제 |
| G18 | 범위 아님 | 웹 서버(server.py order_request_log) |

## 2. 계획 조립 P1–P30

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| P1 | 옮김 | `Desk::assemble_entry_plan`·`assemble_exit_plan` 하나를 미리보기·제출·수정이 같이 부른다 · 🔴2026-11-21 선 없는 진입 = 400 lines_required(조립 안 -- 미리보기도) · 지운 인자 = 400 removed_option(진입: rule·mode(limit 제외)·wait_min · 청산: switch_to·sw_*·limit 아닌 exec) · `dk` 계획 57 + 접수 139 · mut 3 |
| P2 | 옮김 | 계좌 못 읽음 503(진입·청산·전량) · 사이징 없음은 늘 있음(단순 규칙) · 조회 순서(계좌 → 호가·필터) 원본 gather 순서 · `dk` account_not_ok·exit_account_not_ok·xall_account_fail · 계좌 쪽 «포지션 조회 실패 = ok=false»(09-26) `account::fetch_account` · mut «포지션 조회 실패를 ok로» |
| P3 | 옮김 | `assemble::notional_sum`·`pool_positions`(멀티에셋·옛 응답은 전부) · `dk` soxl_short_usdt_pool·multi_assets_pool · mut «풀 가르기 없음» |
| P4 | 옮김 | 다른 심볼 늘리는 지정가(reduce_only 아님·같은 풀) · `dk` other_symbol_orders_and_waits_rule_shared · mut «reduce_only 무시» |
| P5 | 옮김 | 다른 코인 대기 = 걸린 주문 합과 큰 쪽 · 같은 코인 = 합 · skip_wait · `dk` 같은 사례·skip_wait_own_order_in_account · mut «큰 쪽 → 합» |
| P6 | 옮김 | 가용 잔고 버퍼(plan) + 안 걸린 대기 몫(dbwt·dbpg 표식만 · 자기 칸 되돌리기) `assemble_entry_plan` · `dk` |
| P7 | 옮김 | `assemble::pool_balance`(balances 없으면 balance · 풀 없으면 {}) · `dk` eth_long_no_balances_field·soxl · mut «풀 가르기 없음» |
| P8 | 옮김 | `build_entry_plan`(상한까지 줄임·notes·capped·blocked) · fx · sc · mut «room <= 0» |
| P9 | 옮김 | `build_entry_plan`(비율은 상한 뒤) · fx · sc · mut «비율을 상한 전에» |
| P10 | 옮김 | `build_entry_plan`·`build_exit_plan` 비율 검사 · fx(거절) · sc · 그 앞 접수 문의 removed_option·lines_required 거절(P1) |
| P11 | 옮김 | `floor_to`·`ceil_to`·진입/청산 가격 방향 · fx(경계 입력 10종) · mut «내림/올림 경계» «롱 진입가» |
| P12 | 옮김 | 지정가 진입 수량 재검사 → blocked · `dk` limit_tiny_blocked |
| P13 | 옮김 | positionSide 필수 · reduceOnly 없음(계획·`exec::LimitOrder` 에 칸 자체가 없다) · sc(reduceOnly 키 없음) |
| P14 | 옮김 | `assemble_exit_plan`(fresh · 실제 열린 심볼 · only_symbol · other_symbol_open) · `dk` exit_both_symbols_leftover·exit_only_symbol_* · 감시 wf |
| P15 | 옮김 | `build_exit_plan`(전량이면 곱셈 안 함 · min) · fx · sc · mut «min → max» |
| P16 | 옮김 | 집행 머리 `Exchange::prepare`(원본 run_entry·run_entry_limit 안: 키 확인 → 시계 → `bracket::ensure_leverage` · 결과 = 칸 leverage · 실패해도 주문) -- 일곱째 조각에서 `WatchIo::ensure_leverage` 를 지우고 옮김(지정가 대기가 레버리지를 안 맞추던 것을 `ff` 가 잡음) · rp · `ff` · 양성 대조 «prepare 끔» → ff 13건 불일치 |
| P17 | 제거(사용자 결정 2026-11-21 sltp 만) | 권장배수·margin_type·commit_rule_l·rec_rule_size -- 삭제(plan.rs · assemble.rs) · fx·sc 사례도 생성기에서 뺌 |
| P18 | 제거(사용자 결정 2026-11-21 sltp 만) | rec_rule_size·rec_rule_add_order -- 삭제 |
| P19 | 제거(사용자 결정 2026-11-21 sltp 만) | rule_shared 2배 공유 상한 -- 삭제(원본도 binding 에서 뺌) |
| P20 | 제거(사용자 결정 2026-11-21 sltp 만) | rec_rule_bracket −5% 규칙 손절 -- 삭제 · 진입 계획의 bracket 은 선(line_bracket) 또는 무장 그대로(add_lines)뿐 |
| P21 | 옮김 | `Desk::basis_for`(못 읽으면 1.0 · 0 이어도 1.0) · `dk` basis_ref_fails · mut «1.0 대신 다른 값» |
| P22 | 옮김 | 지정가 = 선×basis 롱 내림·숏 올림 · 호가 넘으면 400(감시 진입 예외) · 명목 유지 수량 · `dk` limit_long·limit_short·limit_marketable·stop_entry_above · mut «롱 내림 → 올림» |
| P23 | 옮김 | `runner::RunEx::reprice`(원본 limit_reprice: basis 1.0 이면 None · 호가 넘으면 None · 롱 내림·숏 올림) · `ff` «지정가 대기 자동 재조정»(20초 각본: basis 바뀜 → 취소 확인 → 새 가격 → 체결 → 무장 · 호가·필터 호출 순서까지) · 집행 쪽 E23 inv |
| P24 | 옮김 | `line_bracket` · fx(177) · sc · mut «롱 올림 → 내림» |
| P25 | 옮김 | 선 주문은 1초 체결가 원천 필수(400) · `dk` lines_no_feed |
| P26 | 옮김 | 같은 쪽 선 포지션 = 보낸 선 무시(add_keep · 무장 그대로 크기만) · 🔴선 없는 포지션(옛 규칙 무장·앱 진입)에 추가 진입 = 409 add_on_unlined_position(2026-11-21 PR 61 -- 옛 «선 버리고 −5% 규칙 손절로»·지정가 추가 409·규칙 크기 409 를 이것 하나로) · 청산 중 409 · `dk` add_on_*·limit_add_on_app_position_409 · mut «add_on_unlined 빠짐» |
| P27 | 옮김 | `exit_deadline_sec`(None→120 · 넘침/0 나누기도 파이썬과 같은 예외) · `run_exit` 가 계획 마감 사용 · fx · sc |
| P28 | 옮김 | 실주문 청산 조립(for_order) 펀딩·수수료 조회 없음 · `dk` exit_for_order_skips_display · mut «실주문 청산도 조회» |
| P29 | 옮김 | `bracket::commission_bp`(live 1시간 · 실패 5분 · stale = 마지막 성공값 · fallback 2/5bp · 캐시는 `Binance` 하나에) · rp(«캐시·TTL» 5호출 · «처음부터 실패» · 키 없음) · mut «실패 캐시 1시간» «stale 대신 fallback» |
| P30 | 옮김 | 스위치 상수(assemble.rs) + 위험모델 켜면 컴파일 단언 · 상한 무시 `override_x_from`(오타·음수 = 0) · `dk` override_x · ponytail: 위험모델 갈래(recommend_hold·prescribe)는 안 옮김 |

## 3. 지정가 집행 E1–E26

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| E1 | 옮김 | 집행 = `Exchange::place_limit_gtx` 하나(GTX 만 -- 구조) · HTTP 구현이 `type=LIMIT&timeInForce=GTX` 로 보냄(rp 집행 8사례 URL 글자까지 · mesh 서버가 GTX 아닌 POST 면 panic) · 익절 GTX `place_bracket`·`place_tp_ladder`(rp) · 예외 dbad GTC = `place_add_order`(M12, 원본 그대로 · 사용자 결정 목록) |
| E2 | 옮김 | `http::Binance::signed`(예외 대신 `{"__error__"}` · 네트워크·시간초과·JSON 오류 = 숫자로 시작 안 함) · `sign::http_error`("<상태> [<code>] msg" · code 없음/null · msg 없음 = dict repr · dict 아님 = AttributeError) · 418/429 `sign::ban_note`(기록만, 주문·취소는 막지 않음) · aiohttp 의 멱등 메서드 한 번 재전송까지 · rp(signed 32사례 · 밴 기록 3) · mut «code 버림» «msg repr» «429 빠짐» «재전송 없음/POST 재전송» |
| E3 | 옮김 | `is_post_only_reject` · inv11 · mut «코드 앞머리만» |
| E4 | 옮김 | `fill_maker` -5022 → 재호가 · 그 밖 rejected · inv06 · inv11 |
| E5 | 옮김 | `fill_maker` 호가 실패 → rejected · inv `e05_book_failure_after_post_only_reject_is_rejected` |
| E6 | 옮김 | ✅해결(2026-11-20, D2): `fill_maker`·`run_entry_limit` 이(기다리기 run_entry_wait 는 2026-11-21 제거) 공용 `resolve_unknown_post`(E7 처리) 하나를 쓴다 · inv `e06_…` · `d2_unknown_outcome_same_on_all_paths`(4경로×a~e+-1006+4xx 음성) · 퍼즈 «불명» 접수 4경로 · 음성 대조: 해결 전 exec.rs 로 d2·e06·fuzz 실패 |
| E7 | 옮김 | `resolve_unknown_post`: 네트워크·5xx·-1007·-1006 → 3회 조회(1초) → 찾으면 이어 감 → 끝내 모르면 표식 DELETE → error · inv `d2_…`(d·e) · mut «5xx 를 확정 거부로» |
| E8 | 옮김 | `ensure_closed`(재조회 상태 · 3회×1초) · 확인 못 하면 error · inv03·05·07 · fuzz · mut «취소 확인 실패해도 재호가» |
| E9 | 옮김 | 오류 때 마지막 체결량 · inv `e09_…` · mut «마지막 체결량 버림» |
| E10 | 옮김 | 지정가 체결량 = 응답 `executedQty` · 시장가 다리도 응답 체결량(0 = ACK → 보낸 수량, E10) · M9 만 원본 D3 그대로(filled = 전량) · rp 집행 13 · mut «ACK 를 0 으로» «D3 다르게» |
| E11 | 옮김 | `TERMINAL` · inv04·partial 사례 · mut «PARTIALLY_FILLED 를 종료로» |
| E12 | 옮김 | 상태 조회 실패 재시도 · inv `e12_…` |
| E13 | 옮김 | `drifted`(주문 측면) · inv02·10(가격이 따라가는 방향) · mut «부등호 뒤집음» |
| E14 | 옮김 | `Exchange::book(symbol)` 에 주문 심볼을 넘긴다 · inv(가짜가 심볼 단언) |
| E15 | 옮김 | `REPEG_MAX`=40 · 컴파일 시점 단언 `REPEG_MAX×POLL_SEC ≥ FALLBACK_SEC` · inv11(상한 소진 → rejected) |
| E16 | 옮김 | 알림에 깨도 리페그 판단은 3초마다 · inv `e16_…` · mut «매번 리페그 판단» |
| E17 | 옮김 | `run_entry`·`run_exit`·PEG 가 `fill_maker` 하나 · ⚠시험 없음(구조 -- 원본도 Xself 소스 검사뿐) |
| E18 | 옮김 | `run_entry` 대기 취소 · inv `e18_…`(취소·부분 뒤 취소·확인 실패·음성 대조) · `ff` «따라가기 진입 중 취소»(원본 entry-cancel 핸들러) · mut «취소 무시» |
| E19 | 옮김 | `PEG_TAG`·`WAIT_TAG`(exec) · `bracket::{TP_TAG,SL_TAG,ADD_TAG,bracket_tag}` · `cancel_bracket` 은 표식+측면 글자로만 · 스탑은 그 심볼(칸 없으면 이 심볼) · only_sl/only_tp · 고아 정리는 dbwt·dbpg 만 · inv(wait·d2) · rp(cancel_bracket 7 · 고아 정리 2) · mut «only_tp 인데 물타기도» «다른 코인 스탑» «고아 정리가 dbtp 도» |
| E20 | 제거(사용자 결정 2026-11-21 sltp 만) | run_entry_wait(기다리기) -- 삭제(exec.rs · runner · intake mode=wait · WAIT_CHOICES_MIN) · mode=wait·wait_min = 400 removed_option |
| E21 | 옮김 | `run_entry_limit`(따라가지 않음 · -5022 는 수정 직후 3회만) · inv `limit_paths` |
| E22 | 옮김 | 수정은 체결 0 일 때만 · inv `limit_paths`(부분 뒤 수정) · mut «부분 뒤에도 수정» |
| E23 | 옮김 | 자동 재조정(20초·1bp·50회·체결 시작 뒤 멈춤) · inv `limit_paths` |
| E24 | 옮김 | 취소 확인 중 체결 몫 on_fill · inv `limit_paths`(on_fill 한 번) |
| E25 | 옮김 | `exec::run_entry_stop`(닿기 전 주문 0 · 1초마다 touched(since) · 선 옮기면 since 를 지금으로 · 4시간 만료 · 닿으면 `run_entry(maker_only)` → 잔량 버림 · PEG = 지금 닿음 · TAKER = 미정 M5 · 닿은 뒤 호가 실패 = error(D11)) · inv `e25_stop_entry_paths`(9사례) · mut «since 그대로» «maker_only 빠짐» «M5 빠짐» · ⚠수정 API 가 plan 을 바꾸는 몫은 amend 칸 수량을 계획 사본에 싣는 것으로 갈음(exec.rs ponytail) |
| E26 | 옮김 | `run_entry_limit` PEG(취소 확인 → action 비움 → 따라가기 → 잔량 버림 · 칸 pegged = 전환 시각 · PEG 시작·끝 로그) · inv `limit_paths`(PEG 두 사례) · `ff` «대기 중 PEG» |

## 4. 대기 주문 W1–W12

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| W1 | 옮김 | `Watcher::waits` · `wkey`·`wasset`·`wslots`(칸 번호 순)·`wait_slot`(setdefault) · wf(감시가 같은 쪽 대기를 거둘 때 · 재시작 기록 칸) · 칸을 여는 접수는 접수 |
| W2 | 옮김 | `WAIT_SLOTS_MAX` 10 · `dk` sub_limit_all_slots_running |
| W3 | 옮김 | 4시간 만료(E21·E25 inv) · 기다리기(mode=wait · 마감 잔량 시장가 M3) 몫은 제거(사용자 결정 2026-11-21 sltp 만) |
| W4 | 옮김 | 접수·수정 때 `limit_save_at` · 정상 종료에서만 복구 파일 지움(`entry_then_bracket` -- error 면 남김) · wf·rt |
| W5 | 옮김 | `Watcher::limit_recover`(6시간 · 계좌 못 읽음 유지 · 포지션 있고 무장 없으면 저장한 선으로 무장하고 placed 확인 · 고아 실패면 유지 · 예외 = «복구 실패» 유지) · wf `restart_cleanup_and_recover`·`restart_orphan_errors_keep_file`·`restart_account_fails_and_cleanup_raises`·`restart_already_armed_or_no_position` · mut «고아 실패여도 지움» «6시간 넘어도 복구» |
| W6 | 옮김 | `api_limit_amend`(mode=limit·working·동작 없음·주문 있음/감시 닿기 전 · 일부 체결 뒤 거절 · 반영 중 거절 · 조립 뒤 다시 봄 · 선은 복구 파일에) · 도는 집행에 닿음(W12) · `dk` am_* · `ff` «지정가 대기 중 선 옮김» |
| W7 | 옮김 | «지금 시장가»(체결될 쪽 호가 = 반대편)·«PEG»(내 쪽) 둘 다 선 재검사 `api_wait_action` · `dk` wa_* · `ff` «대기 중 PEG»·«지금 시장가» |
| W8 | 옮김 | PEG 는 지정가 대기만(400) · `dk` wa_peg_on_wait_mode |
| W9 | 옮김 | `bracket_tick`(clear·fire 때 같은 심볼·같은 쪽 대기에 action=cancel · 다른 코인·반대쪽은 둠 · 분할 손절(sl_leg)엔 안 함 = 원본) · wf `position_gone_clears_and_cancels_same_side_wait` · mut «같은 쪽 대기 취소 빼기» |
| W10 | 옮김 | 첫 체결 순간 `RunEx::on_fill` = 추가 진입이면 keep_rearm · 아니면 plan_from_current 로 물타기 없이 무장 · 예외는 칸 bracket 에 · `re` |
| W11 | 옮김 | 대기 종료 뒤: 선 주문 = plan_from_current · 지정가 대기에서 mod·sl_off·tp_off·fired 면 덮지 않음(`entry_then_bracket`) · ⚠원본 전체 흐름 정답지 없음 |
| W12 | 옮김 | 🔴2026-11-21 다리: 도는 진입 작업의 집행 상태를 `Desk::live`(칸 이름 → `exec::Shared`)에 올리고 · 접수 핸들러(wait-action·limit-amend·entry-cancel)가 칸을 읽기 전 `pull_live`(집행 → 칸) · 쓴 뒤 `push_live`(칸 → 집행) · 집행은 원본과 같은 자리(3초 조회 뒤·1초 잠 뒤·따라가기 조회 뒤)에서 Shared 를 읽는다 · Desk = `runner::DeskCell`(tokio Mutex · 실행기는 집행이 기다리는 동안 안 잠금) · 칸 plan 은 원본처럼 같은 dict(선 옮김이 고친 선으로 무장·재조정·감시 선) · `ff` 작업 도중 화면 입력 6시나리오(원본 핸들러를 집행이 잠든 사이 실제 호출) · inv · mut 5 |

## 5. 브래킷 B1–B33

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| B1 | 옮김 | `watch::bracket_load`(못 읽으면 {} · 옛 키 옮김)·`bracket_save`(임시 파일 → 교체) · 양방향 왕복: 파이썬이 쓴 원문 95개 → Rust 가 같은 값으로 읽음(wf) · Rust 가 다시 쓴 95개 + 대기 복구 파일 1 → 파이썬 bracket_load·limit_recover 가 형·repr 까지 같은 값으로 읽음(rt 96/96 · 까다로운 값 사례: 0.1+0.2·1e-07·1e+16·-0.0·2^53+1·5e-324·최대 실수·이모지·이스케이프·옛 키) · 🔴NaN·Infinity 는 파이썬만 읽는다(아래 «새로 본 것») |
| B2 | 옮김 | `bracket_key`·`bracket_rekey` · fx · wf `legacy_key_migrates_on_load`(옛 키 무장이 발동 · 저장 때 새 키로) · `legacy_and_new_key_collide`(같은 칸이면 파일 순서상 뒤의 것 = 파이썬 dict -- serde_json preserve_order 라 같다) |
| B3 | 옮김 | 체결이 있으면(부분 포함) 모든 종료 경로에서 무장 · 결과 칸은 예외여도 채움 · `re` entry_* · mut «체결 뒤 무장 안 함» |
| B4 | 옮김 | `bracket::place_bracket`(포지션 새로 읽어 그 측면 전량 · 옛 우리 주문 정리(dbtp·dbad + 배포 전 dbsl) → TP GTX · strict) · 비상 스탑은 안 건다 -- 옛 규칙 무장의 backstop_price 가 오면 원본 NO_STOP 과 같은 «안 걸었음» 칸 · rp place_bracket 13 요청 열 통째로 같음 · mut «backstop 칸 빠짐» |
| B5 | 제거(M10) | −5% 비상 스탑 걸기(`post_stop`·`place_stop`) 없음 -- 2026-11-21 원본 PR 61 도 지웠다(이제 원본과 같음) · 손절 다시 켜기도 스탑 없이 |
| B6 | 제거(사용자 결정 2026-11-21 sltp 만) | 자동 물타기 `arm_rule_add`·`place_add_order`·무장 칸 add_order·auto -- 삭제 · 배포 전에 걸린 dbad 는 cancel_bracket 이 계속 지운다(`ff` «옛 규칙 무장 닫힘») |
| B7 | 옮김 | `line_hit`(tp_resting 아님 → 서버가 닫음) · fx · sc · 닫기 실행 `bracket_tick` · wf `lines_tp_touch_not_resting`·`lines_tp_touch_resting_holds`·`tp_leg_not_resting_touched_fires_full`·`tp_leg_resting_touched_holds` |
| B8 | 옮김 | `Watcher::watch_brackets(period)`(nap 먼저 → 틱 · 틱 오류는 «bracket watch: …» 로그만 · nap=false 면 멈춤 = 원본 CancelledError) · 주기 주입 가능(`WATCH_PERIOD_SEC`=2 기본) · wf `watch_loop_period_and_wake`(알림에 3초에 일찍 깸 · 계좌 예외 틱 뒤 계속)·`watch_loop_fires_on_second_tick` · wi `watch_loop_survives_every_failure_and_uses_injected_period` · mut «틱 실패에 멈춤» «주기 주입 무시» |
| B9 | 옮김 | `bracket_action` clear(무장+5초 · ok · 0) · fx(armed_suite 280) · sc · mut «+5초 → +0» |
| B10 | 옮김 | clear 정리 실패 → 무장 남김 + retry_after 30초 · wf `position_gone_clear_fails_retry_30s`·`filled_but_cleanup_fails_then_clear` · mut «정리 실패에도 무장 내림» |
| B11 | 옮김 | `bracket_sl_hit`(마감 5분봉 종가 · watch_from) · fx · sc · 입력 고르기(time+300 ≤ 지금인 봉 중 마지막 · 못 읽으면 None + 로그) = `bracket_tick` · wf `rule_*` 10사례(정확히 같은 값·한 틱 차이·무장 전 봉·꼬리만·sl_price 대체·미마감) · mut «미마감 봉도 씀» |
| B12 | 옮김 | `bracket_action` 선 분기 · `line_hit`(손절 먼저) · `sl_legs_hit` · fx · sc · 입력(무장·watch_from 뒤 1초 가격 범위) = `bracket_tick` · wf `lines_*` 14사례 · mut «닿음 >= → >»(fx) «무장 전 1초 가격도 씀»(wf) |
| B13 | 옮김 | `check_lines`(sl_watch · 화면이 보낸 slup) · 익절만 옮기면 손절 방향 그대로(`api_bracket_modify`) · `dk` mod_tp_only_keeps_sl_up·add_new_arm_sl |
| B14 | 옮김 | `bracket_tick` 선 발동(exiting 먼저 → 익절 지정가 먼저 지움 → 실패면 10초 · 지웠으면 fired 를 곧바로 파일) · wf `lines_long_touch_exact`·`lines_tp_cancel_fails_waits_10s`(호출 열의 run_exit 에 그 순간 저장본을 함께 적어 «fired 가 청산 전에 파일에»를 대조) · wi 재시작(청산 도중 죽어도 fired 가 파일에) · mut «10초 → 30초» «fired 를 청산 전에 안 씀» |
| B15 | 옮김 | 발동 직전 armed_at·mod.at 다시 비교 · wf `line_moved_before_fire_skips` · mut «mod.at 비교 빼기» |
| B16 | 옮김 | 조립 뒤 sl_off 다시 봄(선 무장의 익절 fire 는 예외) · wf `disarm_during_plan_rule`·`disarm_before_fire_lines`·`sl_leg_disarmed_during_plan` · mut «조립 뒤 손절 해제 검사 빼기» |
| B17 | 옮김 | no_position 이면 정리하고 내림(실패면 남김) · 그 밖 오류·blocked 는 «닫힘» 아님 → 30초 · last_error = detail or error or blocked · wf `plan_no_position_clears`·`plan_no_position_clear_fails`·`plan_account_unavailable`·`plan_exception_502`·`plan_blocked`·`account_not_ok_sl_hit_plan_fails` · mut «조회 실패도 닫힘으로» |
| B18 | 옮김 | 발동 청산 = `WatchIo::run_exit` → 실제 구현 `watch::run_exit_into`(exec::run_exit) · 다 닫혔을 때만(filled_*) 정리 · 실패면 무장 유지 30초 · 예외 = settle → error · wf `exit_{rejected,error,taker_failed}_retry_after_30s`·`exit_raises_settle_error`·`exit_needs_market_python_taker` · wx(실제 집행: 메이커 체결 · 취소 확인 실패 error) · mut «재시도 시각 안 둠» · 🔴시장가 몫(M6·M7·M9)은 미정 → `market_wait`(wx · 아래) |
| B19 | 옮김 | `bracket_merge`(OWNED_BY_SAVED · 쏜 틱의 tp_resting) · fx(92) · sc · mut «tp_resting 규칙 뒤집음» |
| B20 | 옮김 | 판정 `sl_legs_hit`·`bracket_action`→SlLeg(fx · sc) · 실행 `Watcher::fire_leg`(쏜 조각 먼저 파일에 → 비율 청산 → 성공이면 쏜 조각 빼고 set_sl_legs → 익절 다시 걸기 · 실패면 30초·표시 유지) · wf `sl_legs_one_then_all`·`sl_leg_exit_fails_keeps_fired`·`sl_leg_lock_held_skips`·`sl_leg_disarmed_during_plan`·`sl_leg_with_tp_ladder_rearm` · mut «쏜 조각 표시 안 함» «비율 분모 무시» |
| B21 | 옮김 | `api_bracket_disarm`(clear_bracket(only_sl) = 배포 전 우리 dbsl 만 지움 · 취소 실패면 502 아무것도 안 바꿈 · 그 사이 새 무장 409) · `dk` dis_* 6(원본과 같음 -- 일곱째 뒤 M10 «의도한 차이»에서 돌아옴) · mut «dbsl 취소 실패 무시» |
| B22 | 옮김 | `api_bracket_rearm` 선 무장 = 감시만 다시 · 규칙 무장 = 거래소 스탑 없이 sl_off 만 지움(sl_level·sl_price 둘 다 없으면 409 no_price) · 응답 backstop=null · `dk` re_* 7(원본 PR 61 과 같음) |
| B23 | 옮김 | `bracket_action` sl_off + fired "tp"/True · fx · sc · 실행 wf `lines_sl_off_tp_watch_fires_tp`(fired="tp")·`lines_fired_true_sl_off_holds` |
| B24 | 옮김 | `api_bracket_tp`(off = 우리 dbtp 만 · 일부 실패면 서버 감시로 · on = 5초 안 1초 체결가가 이미 너머면 409 · strict · 그 사이 무장 바뀌면 방금 건 것 지움) · `dk` tp_* 10 |
| B25 | 옮김 | `api_bracket_modify`(익절 레벨 그대로면 안 건드림 F3 · 옮기면 strict F2 · 정리 실패 = 서버 감시+경고 · 무장 바뀌면 방금 건 지정가 지움 S2 · 조각 tpi·sli · 지운 쪽 선 검사 안 함) · `dk` mod_* 15 |
| B26 | 옮김 | `api_bracket_add`(규칙 무장 409 · 그 선이 있으면 분할 조각(최대 4 · ±0.05% 같은 가격 거절) · 무장 없으면 새 선 무장(다른 쪽 never) · 익절 strict) · `dk` add_* 15 |
| B27 | 옮김 | `split_tp_qty` · fx(194) · sc · mut «carry 버림» |
| B28 | 옮김 | `bracket::place_tp_ladder`(옛 익절 다 지운 뒤에만 · 사라진 조각은 직접 조회해 FILLED 일 때만 체결 · 필터는 조각이 없어도 읽음 · split_tp_qty) · rp(9사례: 정상·사라진 조각 체결/취소·전부 체결·먼지 합침·정리 실패·필터 예외·조회 실패·포지션 없음·키 없음) · mut «조회 없이 체결로» · `ladder_apply`(O) 는 접수 |
| B29 | 옮김 | `set_tp_legs`·`set_sl_legs` · 조각 조작 `api_bracket_tp_legs`(`dk` tl_* 13) · 스트림 «우리 dbtp FILLED» → `userstream::Hub::tp_fills` 대기열 → `Watcher::drain_tp_fills` → `Watcher::tp_leg_filled`(원본 그대로 · 감시 루프가 틱 앞에서 비운다 -- 실행 파일 조각에서 부름) · wi `tp_leg_filled_from_stream_queue_drops_that_leg_only` |
| B30 | 옮김 | `Watcher::partial_exit_rearm_tp`(조각 = place_tp_ladder · 단일 = place_bracket · 서버 감시·fired·tp_off·exiting 제외 · 그 사이 새 무장이면 안 덮음 · 조회 실패면 상태 그대로 · 예외는 tp_rearm 에) · wf `partial_exit_rearm_tp_single`·`…_lookup_fails_keeps_state`·`…_skips`·`…_rearmed_meanwhile`·`sl_leg_tp_ladder_raises`·`exit_then_rearm_partial` · mut «strict 로 몰래 고침(D7)» · D7 원본 그대로(strict=false) |
| B31 | 옮김 | `Watcher::keep_rearm`(무장 없으면 arm_bracket · 손절 조각 청산 중이면 건너뜀 · 아니면 익절 수량만) · wf `keep_rearm_paths`·`keep_rearm_sl_leg_fired_skips` · ponytail: 포지션별 asyncio.Lock 대신 `&mut self` 가 한 줄로 세운다 |
| B32 | 옮김 | 청산 진행 중(exiting·fired·청산 칸 없음) 선 수정·익절 on/off·선 추가·조각 조작·추가 진입 거절 · `dk` *_fired |
| B33 | 범위 아님 | 응답 서버(계좌 응답의 `unprotected` -- with_bracket) |

## 6. 청산 C1–C6

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| C1 | 옮김 | 청산 제출·감시 = fresh(미리보기·전량 일괄은 2026-11-21 제거)(감시는 스트림 계좌 가능 -- 아래) · `dk` · `we` |
| C2 | 옮김 | 부분 청산 = `Task::Exit{rearm:true}`(exit_then_rearm) · 전량 = run_exit · `dk` xs_partial_ok·xs_full_ok · 실행 = `Watcher::exit_then_rearm`(wf) |
| C3 | 제거(사용자 결정 2026-11-21 sltp 만) | exit_then_switch -- 삭제(runner.rs · `Task::Switch`) |
| C4 | 제거(사용자 결정 2026-11-21 sltp 만) | 스위칭 위 칸만 -- 스위칭째 지움 |
| C5 | 제거(사용자 결정 2026-11-21 sltp 만) | 전량 일괄 청산 `api_exit_all_submit` -- 삭제(원본 라우트도 없음 · ⚠불확실 표시도 같이 사라짐) |
| C6 | 옮김 | `api_entry_cancel` 청산·스위칭·MARKET 은 취소 안 받음 · `dk` ec_* |

## 7. 재시작·종료·오류 R1–R9

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| R1 | 옮김 | `bracket::cancel_wait_orphans`(dbwt·dbpg 만 · rp 2 · mut «dbtp 도») + 재시작 훅 `Watcher::wait_cleanup`(지운 주문은 그 코인 대기 칸 · 정리 실패는 eth 칸 · 예외면 errors 에 · restart_cancelled 기록) · wf `restart_*` 4사례 · wi `restart_mid_exit_keeps_arming_and_next_process_finishes` · 훅을 «스위치 켜진 곳에서만» 띄우는 것은 G4(접수) |
| R2 | 옮김 | `wait_cleanup` 이 정리 뒤 복구 파일(이름순)마다 `limit_recover` · 결과를 그 칸 recovered 에 · wf `restart_cleanup_and_recover`(6시간 넘은 SOL 파일 지움 + ETH 무장) |
| R3 | 옮김 | `runner::settle`(진입·대기·청산 작업) + 감시 settle · `re` |
| R4 | 부분 | 취소 확인 실패·지정가 접수 불명·PEG/감시 진입 호가 실패 → `Phase::Error` · inv03·05·07·d2·e25 · mesh(HTTP 로도 같은 error) · API 키 없음 = `Binance::has_keys`(브래킷 함수들이 원본 문구로) -- 집행 함수(run_*)의 키 검사는 호출부 · 스위칭은 접수 |
| R5 | 범위 아님 | 웹 서버(종료 훅 · 세션 닫기 전 작업 취소) |
| R6 | 범위 아님 | 웹 서버(shutdown_timeout 2초) |
| R7 | 옮김 | `Binance::nap` = tokio select(잠 · 알림) -- future 를 버리면 바로 끝나 «깨움·취소가 같은 틱이면 취소를 삼킴»(gh-86296)이 생길 자리가 없다 · mesh `nap_wakes_early_on_fill_alert` |
| R8 | 옮김 | `entry_then_bracket`: phase=error 면 복구 파일을 남김(정상 종료에서만 지움) |
| R9 | 옮김 | 진행 상태(entry·side_exits·waits·exiting)는 `Watcher` 메모리 -- 재시작하면 사라지고 무장 파일이 감시를 잇는다 · wi `restart_mid_exit_keeps_arming_and_next_process_finishes`(청산 도중 죽음 → 새 프로세스: 고아 정리 → fired 래치로 익절 취소 없이 바로 청산 → 닫힘 → 무장 내림) · wf `lines_fired_latch_after_restart` |

## 8. 계좌·체결 알림 A1–A11

| # | 상태 | Rust 위치 · 시험 / 조각 |
|---|---|---|
| A1 | 옮김 | `AccountBook::produce` + `fetch_account`(동시 조회 `get_many` · 원본 gather 와 같은 요청 집합 · `ah` account_http 5) · 원장·income 은 화면 몫(안 옮김) |
| A2 | 옮김 | `AccountBook` last_good(30분 · 실패는 ok=false 그대로) · `dk` account_cache_rules_and_last_good |
| A3 | 옮김 | `account::fresh_needs_fetch`(1초) + `swr_decide`(10초 + 묵은 값) · `dk` 같은 시험 · 실제 캐시 자리 = `wire::Wired::account`(ponytail: 묵은 값 대신 기다려 새로) |
| A4 | 옮김 | `USER_STREAM_WS`(/private) · `dk` 스트림 사건 열의 connect 주소 |
| A5 | 옮김 | `Binance::listen_key`(POST·PUT·DELETE · 밴 중이면 안 부름 · 200 아니면 본문을 JSON 전에 밴 기록) · 닫기는 안 부른다 · `us` userstream_ws |
| A6 | 옮김 | `userstream::LiveStream`(tokio-tungstenite 0.26 · rustls ring 박아 둠 · heartbeat 120 ping · 수신 60초 시간초과 계속 · CLOSE·만료면 다시) · `us` · ponytail: pong 은 안 기다림 |
| A7 | 옮김 | 재연결 5초 ×2 최대 300 · 5분 넘게 붙었으면 5초 · 밴이면 남은+1초 · `dk` expired_reconnect_backoff·keepalive_* · mut «5분 → 50분» «+1 → +0» |
| A8 | 옮김 | 연결되자마자 dirty + wake · `dk` |
| A9 | 옮김 | `account::user_event_kind`(+ «o» 가 dict 아니면 원본처럼 예외 → 다시 연결) · `dk` connect_events_close·tp_leg_filled_raises_keeps_stream · mut «체결에 계좌 다시 읽기 빠짐» |
| A10 | 옮김 | 알림은 앞당기기만: `nap` 으로 일찍 깨도 3초 판단 그대로(inv `e16_…`) · 감시 루프도 알림에 일찍 깨되 주기는 그대로(wf `watch_loop_period_and_wake` · wi) · `Binance::wake`(Notify) · mesh `nap_wakes_early_on_fill_alert` · 알림 수신(2026-11-21): `app::run` 이 `LiveStream`(Hub.wake → 감시 nap 깸) + 계좌 갱신기를 함께 돌림 · 끊기면 감시·조회는 그대로(스트림 계좌만 못 씀 → REST) · `ar` stream_fill_* |
| A11 | 옮김 | `account::account_refresher`(0.3초 묶고 clear → 다시 읽기 → seq+1 → wake → 3초) · `dk` refresher_coalesce_and_gap · mut «0.3 → 0» · 실행 파일 자리 = `app::Refresher`(새 계좌 → 감시와 같은 캐시 `Wired.cache` · 스트림 계좌 대조 · `Hub.seq`) · `ar` stream_fill_*(연결+체결 0.3초 안 = 조회 1 · 3.6초 뒤에도 1) · mut «번호 안 올림» «clear 빠짐» «대조 빠짐» · 원본의 income_cache 비우기는 화면 몫(안 옮김) |

## 집계

| 상태 | 첫 조각 | 둘째 뒤 | 셋째 뒤 | 넷째 뒤 | 다섯째 뒤 | 여섯째 뒤 | 일곱째 뒤 | sltp 정리 뒤(→ 계좌 알림 조각 뒤) |
|---|---|---|---|---|---|---|---|---|
| 옮김 | 38 | 47 | 68 | **112** (+30 범위 아님→옮김 · +14 부분→옮김: G5·G15·G17·W2·P1·P6·P12·P14·P17·P20·C1·C2·B21·B22) -- 그중 ⚠시험 없음 G7 · ⚠불확실 C5 | **113** (+C4·E10 · B4·B22·B6 은 «M10 차이» 표시 · −B5 제거) | **135** (선 핸들러·실행기·전송·동시 조회) | **135** | **123** (−12 제거: G13·G16·G17·P17·P18·P19·P20·E20·C3·C4·C5·B6) → **125** (+G4·A10) |
| 부분 | 22 | 21 | 31 | **23** (+G1·G2·W7·A1·A5·A6 범위 아님→부분 · G3 은 부분 그대로) | **23** (+C3 · −E10) | **5** (G1·G4·G8·R4·A10) | **5** | **5** → 계좌 알림 조각 뒤 **3** (G1·G8·R4 -- 셋 다 웹 서버 접수 몫 · G4·A10 → 옮김) |
| 미정(사용자 결정) | 0 | 0 | 0 | **2** (C3·C4 = 스위칭 M11 -- 접수에서 «미정» 거절) | **0** (질문지 답 -- 시장가 유지 · M10 제거) · 제거 1(B5) | 0 · 제거 1 | 0 · 제거 1 | 0 · 제거 **13**(B5 + 위 12) |
| 범위 아님 | 85 | 77 | 46 | **8** (G18·R5·R6 웹 서버 · P23·R8 대기 작업 실행기 · B25·B26 안 함 · B33 응답 서버) | **8** | **4** (G18·R5·R6 웹 서버 · B33 응답 서버) | **4** | **4** |
| 옮긴 몫에 시험 없음(⚠) | 4 | 2 | 1 | 2 (E17 구조 · G7 동시 요청) | 2 + M4 원본 정답지 없음 | 3 (G7 · P23 원본 정답지 없음 · B29 대기열 소비자 없음) | **1** (G7 동시 요청) | **1** (G7 · C5 ⚠불확실은 제거로 사라짐) |

이 조각 뒤 남는 일(다음 조각 범위):
1. **웹 서버에 붙이기** -- `intake::POST_ONLY`·`LOCKED` 표대로 라우트·`busy_or` · 접수 기록(G18) · `server_ms` · NOCACHE · 종료 훅(R5·R6) · 상태 조회(`api_manual_entry_status`) · 동시 요청으로 G7(조립 뒤 칸 다시 보기) 시험.
2. **남은 접수 핸들러 4개** -- 익절 on/off(B24) · 분할 조각(tp_legs) · 선 추가(B26) · 선 수정(B25) · `check_lines`(B13)·`lines_quote` · 사용자 스트림의 `tp_leg_filled` 실제 구현(B29).
3. **작업 실행기** -- `Task` 를 받아 `entry_then_bracket`(W3·W4·W10·W11·B3 · ensure_leverage P16 · limit_reprice P23 · 복구 파일 R8) · 진입·대기 settle(R3) · 상태 칸(Obj) ↔ 집행 상태(ExecState) 다리(action·amend 전달 W12) · 끝나면 `Desk::task_done`.
4. **실시간 코어와 잇기** -- `assemble::Cached` 에 값 넣기(bookTicker 웹소켓 · 1분봉 → 변동성 · premiumIndex · 필터 · 수수료) · `WatchIo` 의 5분봉·1초 가격 · 사용자 스트림 사건 → `StreamAccount::apply` → `wire::Wired::stream` · 감시 «체결마다»(설계 §6).
5. **전송** -- listenKey HTTP(418/429 본문을 JSON 전에 기록 A5) · 사용자 스트림 웹소켓(전용 연결 A6 -- tokio-tungstenite 를 들일지 사용자 결정) · 계좌 조회 심볼별 동시 보내기(지금은 차례로).
6. **그림자 대조(M2 · 7일)** -- 같은 입력으로 Rust 계획 대 파이썬 계획 · 스트림 계좌 대 REST 어긋남 수(`StreamAccount::mismatches`, 분모 = 대조 횟수).
7. **응답 서버 몫** -- 계좌 응답의 `unprotected`(B33) · 원장·income·day_start·last_good 화면.

## 시장가 분기 M1–M13 (2026-11-21 사용자 질문지 답 -- docs/rtcore_design_20261120.md §0)

| # | 결정 | Rust 위치 · 시험 |
|---|---|---|
| M1 | 원본 동작 유지(진입 120초 마감 잔량 시장가) | `exec::run_entry` → `taker`(응답 체결량 · ACK = 보낸 수량) · rp 집행 5(정상·부분 뒤·ACK·거부·끊김 = taker_failed) · inv |
| M2 | 반만 유지(감시 진입 TAKER 가 쓰는 `run_entry` MARKET 갈래 -- 스위칭 반대 진입은 제거) | `run_entry` plan.market · rp «M2» · inv |
| M3 | 제거(사용자 결정 2026-11-21 sltp 만)(기다리기) | `run_entry_wait` 삭제 · rp·inv 사례도 뺌 |
| M4 | 유지(지정가 대기 «지금 시장가» -- 취소 확인 뒤 잔량) | `run_entry_limit` · inv `limit_paths` · ⚠원본 정답지 없음(가짜 서버로 대기 중 action 을 넣는 각본을 못 만들었다) |
| M5 | 유지(감시 진입 TAKER) | `run_entry_stop` → run_entry(MARKET) · rp «M5» · inv |
| M6·M7 | 유지(고변동성·먼지 청산 = 전량 시장가) | `run_exit` plan.market · rp 2 · wx · we · dk xs_high_vol_m6·xs_dust_m7(원본과 같음) |
| M8 | 제거(사용자 결정 2026-11-21 sltp 만)(«즉시» 청산) | 조립 market 인자 삭제 · exec=market = 400 removed_option(dk xs_removed_market) |
| M9 | 유지(따라가기 청산 마감 잔량 시장가 · D3 그대로: filled = 전량) | `run_exit` · rp 2 · wx · mesh · mut «D3 다르게» |
| M10 | **제거**(원본 PR 61 도 제거 -- 이제 같음) | B5 · 배포 전 dbsl 정리 경로는 양쪽에 남김 |
| M11 | 제거(사용자 결정 2026-11-21 sltp 만)(스위칭) | `Task::Switch`·`exit_then_switch` 삭제 |
| M12 | 제거(사용자 결정 2026-11-21 sltp 만)(자동 물타기 GTC) | B6 · 배포 전 dbad 정리 경로는 남김 |
| M13 | 유지(대기 «지금 시장가» 접수 -- 지정가 대기는 체결될 쪽 호가로 선 재검사) | `api_wait_action` · dk wa_market_m13(원본과 같음) |

시장가 다리의 뒤처리 = 원본 그대로: 오류(불명 응답 포함)면 `taker_failed`(조회·재시도 없음) · 성공이면 응답 체결량(0 = 보낸 수량) · `taker_order_id`. 감시가 부른 청산이 taker_failed 면 무장 유지 + 30초 뒤 재발동(그 전에 앞선 청산 주문 정리 -- 결함 수정 규칙).

## 원본과 일부러 다르게 둔 곳

- 시장가 13곳 → 위 표(보내지 않고 멈춤).
- 손절 감시(셋째 조각): 발동 청산이 시장가 분기에 닿으면 파이썬은 시장가로 닫고 무장을 내린다 -- Rust 는 칸 `needs_market_decision` + 무장 `market_wait`(branch·qty·at·문구) + `last_error` 를 남기고 **그 무장은 다시 쏘지 않는다**(포지션이 사라지면 clear 는 그대로). 30초 뒤 다시 따라가게 두면 «조용히 지정가로 바꾸기»라서. 결정이 나면 이 칸을 지우는 쪽(접수)이 정한다. 파이썬은 이 칸을 모르고 무시한다(전환 순간 파일을 넘겨받으면 파이썬은 다시 쏜다).
- 감시 바깥(계좌·5분봉·1초 가격·청산 계획 조립·거래소 함수·잠금·스위치·시계)은 `WatchIo` 뒤 -- 실제 구현은 다음 조각(조립·계좌). `run_exit` 의 실제 몫만 `watch::run_exit_into` 로 있다(끝난 뒤 칸에 한 번 옮김 -- 도는 동안 화면은 "submitting").
- `keep_rearm` 의 포지션별 asyncio.Lock → `&mut self`(한 줄로 세움). 저장본 글자(구분자·실수 표기)는 다르고 뜻은 같다(B1 왕복). 원본이 dict 아닌 JSON 저장본에서 AttributeError 로 죽는 자리는 {}.
- ~~`run_entry`·`run_exit` 등의 «API 키 없음 → error» · `_clock_offset` · `ensure_leverage` 호출 → 접수 조각~~ → 일곱째 조각에서 `Exchange::prepare`(exec 머리)로 옮겨 원본과 같다.
- 주문 응답이 dict 가 아니거나 executedQty 가 숫자가 아니면 원본은 예외(settle → error), Rust 는 숫자로 시작하지 않는 오류 문자열(접수 = «나갔는지 모름», 조회 = «재시도») -- `http::reply` ponytail.
- JSON 해석 오류 문구(JSONDecodeError 뒤)는 파이썬과 다르다(앞머리만 같다) · 밴 기록의 429 해제 시각은 실제 시계라 정답지와 폭만 비교.
- 원본은 키를 호출마다 환경에서 읽는다 -- `Binance::from_env` 는 만들 때 한 번.
- 파이썬 예외(0 나누기·inf→int·빈 목록 인덱스) → 같은 종류의 `PlanError`(정답지가 종류·문구까지 비교).
- ~~D2(«나갔는지 모르는 주문» 처리가 세 경로에서 다름)는 통일하지 않았다~~ → **해결(2026-11-20, 사용자 결정)**: 파이썬 원본을 먼저 고치고(`resolve_unknown_post` 공용 함수) Rust 도 같게 맞췄다 -- 원본과 다른 곳이 아니다. inv `d2_unknown_outcome_same_on_all_paths` 가 «세 경로가 같다»를 고정한다.

## 셋째 조각에서 원본에 새로 본 것(고치지 않음 · 시험이 원본 동작을 고정)

- ~~무장 한 칸이 망가지면 틱이 매번 KeyError 로 죽어 다른 코인 손절 판정 전체가 멈춘다~~ → ✅해결(파이썬 main 0ecc754a · Rust 다섯째 조각): 칸 단위로 예외를 잡고 건너뜀(지우지 않음 · dict 아닌 칸은 합치기에서 빠져 그대로) · (칸 키, 예외 종류)마다 로그·알림 한 번 -- wf corrupt_* 3 · mut «틱을 죽임» «매 틱 알림».
- 주문 잠금 중에도 규칙 무장 발동은 청산 계획 조립(새 계좌 조회 · 무게 ~50)을 매 2초 하고 버린다(잠금 검사가 조립 뒤) -- wf `lock_held_rule_D9`. 선 무장은 조립 전에 본다.
- 실행 스위치는 틱 시작에만 본다 -- 틱 도중 꺼져도 그 틱의 청산은 나간다 -- wf `exec_switch_off_during_tick_still_fires`.
- 저장본은 json 기본값이라 NaN·Infinity 도 쓰고 읽는다. serde_json 은 못 읽는다(Rust 는 {} = 감시 멈춤). 지금 쓰는 쪽이 유한값만 쓰는지는 이번 범위(arm_bracket·감시·복구)만 확인.
- 선 발동에서 익절 취소가 **예외**로 새면 `bracket_exiting` 에서 안 빠진다(그 키의 선 수정·익절 켜기가 재시작까지 거절) -- Rust 도 같게 둠.
- 헤지 롱·숏이 한 틱에 같이 닿으면 둘째 청산이 같은 위 칸을 `clear()` 해 첫 청산 결과가 화면에서 사라진다 -- wf `hedge_long_short_both_fire`.

## 넷째 조각에서 원본과 일부러 다르게 둔 곳

- **시장가 분기를 접수에서 «미정» 거절**(409 `market_undecided` + `branch`): M8 «즉시» 청산 · M11 스위칭 · M13 대기 «지금 시장가» · M6·M7(버튼 청산 조립이 MARKET 계획). 이유: 접수해 두고 집행이 `needs_market_decision` 에서 멈추면 화면은 «닫히는 중»인데 주문이 하나도 안 나가 포지션이 그대로다 -- 거절이면 «안 나갔다»가 바로 보인다. 칸·작업을 잡지 않는다. 감시(자동 손절)는 접수를 지나지 않으므로 셋째 조각대로 집행이 멈추고 무장에 `market_wait`. 따라가기 진입 마감 잔량(M1)·기다리기 마감(M3)은 접수하고 집행이 멈춘다(안 찬 진입 잔량은 포지션 위험이 아니다).
- 계좌 조회의 심볼별 userTrades·openOrders 를 동시에(gather)가 아니라 차례로 보낸다(요청 수·무게는 같다 · 정답지 요청 열이 정해진다) -- 실제 연결 때 동시로.
- `swr_cached` 의 «묵은 값 주고 뒤에서 갱신» 대신 `wire::Wired::account` 는 기다려 새로 읽는다(더 새 값 · 그만큼 늦다).
- 화면 몫(왕복 원장 파일 기록·income·day_start·추세 칸)은 계좌 스냅숏에 안 싣는다 -- 주문 경로가 안 읽는다.
- 위험모델 갈래(`recommend_hold`·`prescribe`·`entry_notional`·`exit_fraction_required`)는 안 옮겼다(스위치가 꺼져 닿지 않는다 · 켜면 컴파일 단언이 막는다).
- 질의 숫자 해석: 파이썬 `float()` 이 받는 «1_000» 같은 밑줄 표기는 Rust 가 못 읽는다(400) -- 화면은 보내지 않는다.

## «받아 둔 값» (사용자 결정 · 넷째 조각)

| 값 | 허용 나이 | 근거 | 낡으면 |
|---|---|---|---|
| 호가(주문 심볼·환산비 USDT 심볼) | 2초 | 원본은 발주마다 새로 묻는다 · 웹소켓 bookTicker 는 바뀔 때마다 온다(실서버 70초 45,883건) -- 2초 안 바뀜 = 흐름 멈춤 | (가)로 묻는다 |
| 1분 변동성 | 60초 | 마감 1분봉 30개로 잰다 -- 1분에 한 번 바뀜 · 청산 마감·M6 문턱에 들어간다 | (가) |
| 펀딩 | 300초 | 화면 «예상 비용» 표시 전용 · 정산 8시간 사이 천천히 | (가) |
| 수수료율 | 3600초 | 원본 FEE_TTL_SEC 그대로 · 표시 전용(P29) | (가) |
| 필터 | 86400초 | 원본은 프로세스 수명 동안 캐시 | (가) |
| 마진 방식 | 받아 두지 않음 | 자동 규칙 안전 검사(G17 · 10-09 격리 사고) | 늘 (가) |
| 포지션·미체결(청산 계획의 fresh 계좌) | 스트림 연결 중 · 같은 연결 · 마지막 REST 대조 60초 안 · 어긋남 없음 | 체결 알림마다 REST 로 다시 읽어 대조(0.3+3초) -- 60초 대조 없음 = 체결 없음 또는 재조회 막힘 | REST(그리고 대조) |

시험: (가)=(나) 149사례 계획·본문·칸·파일 완전히 같음 · 시세 조회 파이썬 496회 → (나) 5회(받아 두지 않은 심볼만) · 값마다 «한도 안 = 받아 둔 값을 씀 / 한도 넘음 = 버리고 (가)»(일부러 틀린 값을 넣어 쓰였는지 확인) · 스트림 계좌: 기준 없음·끊김·재연결·어긋남(알림 누락 = REST 는 0 인데 스트림은 0.1)·60초 넘음 → 전부 REST · 실패한 REST 는 기준을 안 바꿈 · 감시 청산이 스트림 계좌로 조립하면 REST 계좌 읽기 0(we).
조립에서 사라지는 거래소 왕복(파이썬 대 Rust (나)): 진입 = 공개 조회 4~5회(호가·환산비 호가 1~2·1분봉·펀딩 · 필터·수수료는 원래 캐시) → 0 · 청산 = 새 계좌 25요청(시계·잔고·포지션 + 심볼 11개마다 userTrades·openOrders) + 호가·1분봉(+펀딩·수수료) → 0(스트림 계좌를 쓸 수 있을 때) · 남는 것 = 마진 방식(자동 규칙만)과 주문 자체.

## 넷째 조각에서 원본에 새로 본 것(고치지 않음)

- `?lev=inf` → `int(float("inf"))` 의 OverflowError 를 아무도 안 잡아 aiohttp 500(query_leverage 는 TypeError·ValueError 만 잡는다) -- Rust 도 500.
- 레버리지를 안 보낸 진입(lev 없음)은 크기를 정책 천장 25배에서 고른 **거래소 설정 30배**로 잰다(`leverage_setting`) -- 그런데 `target_leverage` 는 None(처방 없음)이라 거래소 레버리지를 안 바꾼다. 거래소가 10배면 증거금이 모자라 거부될 수 있다(정답지 eth_short_half_model_lev: 50% 에 5.999 ETH = 1000×0.5×30÷2500). 화면은 늘 lev 를 보낸다(app.js MANUAL_LEV_AUTO).
- 감시 발동 청산의 조립(`bracket_tick` → assemble_exit_plan)은 `for_order` 를 안 넘겨 **화면용 펀딩·수수료까지 묻는다**(P28 은 버튼 청산만) -- 손절 청산이 그만큼 늦다(we 가 이 동작을 고정).
- 새 계좌 읽기 한 번 = HTTP 25요청(심볼 11개의 userTrades·openOrders 는 원장·화면 몫인데 청산 계획의 fresh 경로에 그대로 있다) -- 설계 §1 청산 조립 중앙 377ms 의 대부분으로 보인다(추정).
- 선 포지션 추가 진입(add_lines)은 «검사만 새 지금가로»라는 주석과 달리 선 검사를 아예 안 한다(조건식이 line_bracket 을 건너뛴다) -- 그래서 `add_lines_out_of_range` 400 은 닿지 않는 코드.
- 지정가 수정(`api_manual_limit_amend`)은 복구 파일을 **가격·수량을 고치기 전에** 쓴다 -- 파일의 price·quantity 는 수정 전 값, 선은 새 값.
- 위험모델이 꺼져 있는데 상한 칸의 `risk.reason` 은 `worker_no_risk`(워커가 위험을 안 냄)로 나온다 -- 표시 문구만.
- 사용자 스트림의 ORDER_TRADE_UPDATE 에서 `o` 가 dict 가 아니면 user_event_kind 가 AttributeError → 연결을 끊고 다시 붙는다(+ 계좌 다시 읽기) -- 메시지 하나로 재연결.

## 다섯째 조각 (2026-11-21)

- 시장가 분기 풀기(위 표) · M10 제거 · 파이썬 결함 수정 3건(main 0ecc754a)을 Rust 에 맞춤. 정답지 셋(http·watch·desk)은 결함 수정본 파이썬으로 다시 만들었다(`RTORDER_PY_ROOT` · 작업 폴더 파이썬은 안 고침).
- 결함 수정 규칙 반영: ① 익절 GTX·분할 익절 조각·물타기 GTC 의 불명 응답 = `bracket::resolve_resting`(exec::resolve_unknown_post · 3번 조회·1초 · 끝내 모르면 표식 DELETE «표식으로 취소를 보냈습니다» · 4xx 그대로) · ② 200 빈 본문 = `EmptyResponse: HTTP 200 본문 None`(http::signed 한 곳) · ③ 재발동 전 `bracket::cancel_exit_orders`(dbpg+측면 AND 닫는 쪽 · 읽기→지우기→다시 읽기) -- `Watcher::exit_retry_clear`(retry_after 있을 때만 · 잠금 중이거나 그 코인 청산 칸이 없으면 무장·retry_after 를 안 바꾸고 넘어감 · 실패면 30초·last_error) · ④ 망가진 칸 건너뛰기·알림 한 번(`WatchIo::alert` -- 보내기는 구현 쪽이 뒤에서). rp cancel_exit_orders 6·불명 응답 6 · wf 새 사례 10(refire_*·first_fire·corrupt_*).
- ⚠원본과 남은 차이: `armed_at` 이 숫자가 아니면 Rust 는 늘 그 칸을 건너뛴다(원본은 그 식에 닿는 갈래에서만 -- 손절 해제된 규칙 무장 + 포지션 있음이면 안 닿는다) · 예외가 난 칸의 «예외 전까지 고친 사본»은 남기되 changed 로 치지 않는다.

## 의도한 차이(사용자 결정 2026-11-21 «−5% 비상 스탑 Rust 에서 제거»)

원본 PR 61(사용자 «sltp 만», 2026-11-21)이 파이썬에서도 스탑 걸기를 지우고 배포 전 dbsl·dbad 정리 경로만 남겨서, 이 표의 줄은 전부 원본과 같아져 지웠다
(place_bracket backstop 칸 · cancel_bracket 스탑 목록·취소 · place_stop 없음 · 손절 해제 clear_bracket(only_sl) · 손절 다시 켜기 sl_off 만 · 자동 물타기 없음 · 규칙 손절 = 서버 5분봉 종가 감시만).
✅«파이썬 시절 dbsl 이 남으면 Rust 는 안 지운다»는 미결 항목은 닫혔다 -- Rust 도 포지션 닫힘(clear)·손절 해제·새 무장 정리 때 dbsl·dbad 를 지운다(`ff` «옛 규칙 무장 닫힘 → dbad·dbsl 정리»).
남은 의도한 차이: 없음.

## 여섯째 조각 (2026-11-21)

- 선 핸들러 4개(`intake.rs` api_bracket_tp · _tp_legs · _add · _modify + ladder_apply·lines_quote·check_lines·tp_legs_of·sl_legs_of) -- 원본 정답지 51사례(dk) 상태코드·본문·호출·무장 파일·로그 전부 일치 · (가)=(나) 도 같이.
- 작업 실행기 `runner.rs`(entry_then_bracket · exit_then_switch · limit_reprice · settle · ensure_leverage 자리 · 상태 칸↔집행 상태 다리 slot_to_exec/exec_to_slot · run_task) -- `re` 4(정상 흐름 · 익절 못 걸림 → 서버 감시로 닫힘 · 청산 오류 → 재발동 전 정리 · 따라가기 미체결 → M1 시장가 → 무장).
- 전송: `Binance::listen_key` + `userstream::LiveStream`(tokio-tungstenite 0.26.2 · rustls 0.23.46 ring -- rtcore 와 같은 판 · Cargo.lock 에 새 줄은 이 둘과 그 의존성) -- `us` 1(연결·사건·끊김·재연결·listenKey 만료·알림 누락 뒤 REST 대조 = 어긋남).
- 계좌 조회 동시화: `AccountRest::get_many`(Binance = spawn_blocking 동시 · 가짜 = 차례로) -- `ah` 5사례 «요청 집합(순서 무관)·스냅숏» == 원본(429 사례는 정답지 생성기가 밴 판정을 막아 두어 뺐다).
- M4 원본 정답지: http 생성기에 «n 번째 주문 조회 때 state action 넣기» 훅 · Rust 재생도 같은 자리(Virt::get_order) -- rp 2(바로 · 부분 체결 뒤) 일치.
- ⚠원본과 대조 못 한 것: 실행기 «전체 흐름» 정답지(원본 entry_then_bracket+bracket_tick 을 실제 X 함수로 127.0.0.1 에 돌리는 각본)는 만들지 못했다 -- 이번엔 Rust 끝에서 끝 시험(`re`)과 조각별 원본 대조(rp·wf·dk)로 갈음. → 일곱째 조각 `ff` 로 해소.

## 일곱째 조각 (2026-11-21)

- 원본 «전체 흐름» 정답지 `ff`: 진입 메이커 체결 → 익절 걸림 → 손절선 닿음 → 우리 익절 정리 → 청산 → 무장 해제 + 실패 주입(익절 못 검 → 서버 감시 · 청산 취소 확인 실패 → 재발동 전 정리 실패 30초 → 정리 성공 · 발동 전 익절 취소 실패 10초 · 따라가기 미체결 M1) + P23. 원본만 보내는 스탑 요청(openAlgoOrders -- M10 제거)은 원본 열에서 뺀다.
- `ff` 가 잡아 고친 Rust 결함 4: ① 집행 머리(키 확인·시계 요청·ensure_leverage)를 실행기가 빼먹음(지정가 대기는 레버리지 아예 안 맞춤) → `Exchange::prepare` + `exec::head` · `WatchIo::ensure_leverage` 삭제 ② done_at 이 시작 시각 → 끝난 시각(runner·wire 청산 경로 포함) ③ 상태 칸 `wait_until`·`amends`·`reprices` 빠짐 ④ 칸 `leverage` 를 집행 상태에서.
- B29 대기열 소비: `Watcher::tp_leg_filled`·`drain_tp_fills`(원본 tp_leg_filled).
- 남은 것: 그림자 실행(feature `live` 빌드 + 실행 때 그림자 모드 -- GET·listenKey 만 허용, 주문 경로 panic) · `bin/rtorder` 뼈대 · rtcore 값 어댑터(Cached·WatchIo 5분봉·1초가) · 웹 서버에 붙이기.

## «sltp 만» 정리 (2026-11-21 · 원본 PR 61 판 = `claude/stop-removal-20261121` @ 96a22610)

- 사용자 결정 «sltp(차트 선 주문 + 진입가 팝업 청산)만 쓴다 -- 플로팅 버튼 로직을 다 제거»를 Rust 에도. 지움: 자동 규칙 진입(권장배수·margin_type·commit_rule_l·rec_rule_size·rule_shared·rec_rule_bracket) · 자동 물타기(arm_rule_add·place_add_order·add_order·auto) · 기다리기(run_entry_wait·M3) · 스위칭(exit_then_switch·Task::Switch·M11·C3·C4) · «즉시» 청산(M8 · 조립 market 인자) · 청산 미리보기 · 전량 일괄 청산(C5).
- 새 거절 3종(원본과 같은 순서·같은 본문): 400 `removed_option`(진입 rule·mode≠limit·wait_min / 청산 switch_to·sw_*·limit 아닌 exec -- 이름순) · 400 `lines_required`(조립 안 · 미리보기도) · 409 `add_on_unlined_position`(선 없는 같은 쪽 포지션).
- 되살린 정리 경로: `cancel_bracket` 이 다시 openAlgoOrders 를 읽어 우리 dbsl 을 지운다(다른 심볼 스탑은 안 지움) · 손절 해제가 clear_bracket(only_sl) 을 부른다(실패 502) · place_bracket 은 backstop_price 가 오면 원본 NO_STOP 칸. dbad 는 원래대로 cancel_bracket 이 지운다.
- 정답지 전부 PR 61 판으로 다시: plan 1,359건 · http 117시나리오(요청 775 -- 스탑 요청을 빼던 대조 없앰, 통째로 같음) · watch 106 · desk 계획 57 + 접수 139 + 계좌 19 + 스트림 8 · flow 8시나리오(+옛 규칙 무장 닫힘 → dbad·dbsl 정리 · 감시 진입 닿음 → 따라가기 → 무장).
- `ff` 가 새로 잡아 고친 것: 감시 진입 칸 `triggered` 가 원본은 닿은 시각 문자열(None → iso) -- Rust 는 bool 이었다 → `Option<f64>` → iso.

## 그림자 실행 (2026-11-21 · C)

- **모드**: 기본 = 그림자. `shadow::mode_from` -- `RTORDER_MODE` 가 없거나 shadow = 그림자 · live 는 cargo feature `live` 빌드 + `RTORDER_LIVE_CONFIRM=실주문을-보낸다` 둘 다 있어야(하나만이면 시작 안 함 · 모르는 값도 시작 안 함). 허가증 `LiveProof` 는 모듈 밖에서 못 만든다.
- **전송 계층 막이**(`http::Binance::guard`): 그림자 모드면 GET·`/fapi/v1/listenKey` 가 아닌 요청(주문 POST·취소 DELETE·레버리지 POST·algo DELETE …)은 **보내기 전에 panic**. 루프백이 아닌 주소로 만든 값은 늘 그림자로 시작 · 끄는 길은 `arm_live(&LiveProof)`(feature `live` 에만 있음). 키를 읽는 `Binance::from_env` 도 feature `live` 에만 있다(기본 빌드 = 키 없음).
- **재생 도구**(`shadow::replay` · 실행기 `cargo run --example shadow_replay -- <기록.jsonl>`): 작업(chain) 하나의 기록 = start(계획·칸·무장 파일) · wire(왕복) · input(읽은 입력) · end(칸·무장 파일·복구 파일·로그). 127.0.0.1 가짜 서버가 기록된 응답을 차례로 돌려주고 진짜 `Binance`·`runner::entry_then_bracket`·`Watcher::bracket_tick`·`run_exit_into` 가 돈다. 실제 기록 비교는 timestamp·signature 를 빼고 표식 뒤 ms 를 `#` 로(비밀값은 재생에 넣지 않는다) · 가짜 시계 정답지는 글자 그대로.
- 시험: `ff`(전체 흐름 정답지 19걸음을 재생 도구로 글자 그대로 -- 요청 282 · 서명 URL 192 · 청산 접수 op=exit 2걸음 포함) · `sr` = tests/shadow_replay.rs(합성 실기록 차이 0 · 양성 대조 4 · 쓰기 5종 panic + 대조군 · 읽기·listenKey 는 보냄).
- 화면 입력(rec=input · what="handler" · args [핸들러 이름, 질의] · value {status, body} · t)을 그 시각에 넣는다: 집행이 잠든 자리(nap·sleep)에서 진짜 `Desk::api_*` 를 부르고 응답도 대조 -- W12 다리로 도는 집행에 닿는다. 합성 실기록 25작업 차이 0.

## 실행 파일 뼈대 · 허브 구독 (2026-11-21 · D · E)

- `src/app.rs` + `src/bin/rtorder.rs`: 설정(RTORDER_DIR 필수 -- 파이썬 서버 폴더(order_requests.jsonl 있음)는 거절 · RTORDER_BASE · RTORDER_WS · DASHBOARD_MANUAL_EXEC_SYMBOL · RTORDER_PERIOD · RTORDER_STREAM=1 · RTORDER_HUB · RTORDER_BAN_FILE · RTORDER_MAX_TICKS) → 모드 → 시작 복구(원본 순서: wait_cleanup = 남은 대기 주문 정리 → 지정가 대기 복구 파일로 무장) → 감시 루프(nap → 허브 값을 Cached 로 → drain_tp_fills → bracket_tick) + 사용자 스트림(켜면) + 허브 구독(켜면) → SIGTERM/SIGINT 로 멈춤(종료코드 0 · 설정·모드 오류 2).
  - 그림자: 손절 함수(정리·익절·분할 익절·고아 정리·재발동 정리)와 집행(GTX·시장가·취소·레버리지)은 보내지 않고 «안 보낸 쓰기» 목록에 적은 뒤 «못 함»을 돌려준다(원본 «키 없음»과 같은 모양 -- 무장은 남고 30초 뒤 다시). 그래도 새면 전송 막이가 panic.
  - ponytail: 원본은 정리를 작업으로 띄우고 바로 감시 -- 여기는 정리를 끝낸 뒤 감시 · 주문 잠금 없음(접수 문이 없다 -- R6). (알림(텔레그램)은 2026-11-21 계좌 알림 조각에서 보냄 -- 아래 절)
- `coreval::subscribe`(E): rtcore 허브에 붙어 `{"subscribe":[..]}` → 스냅숏(book bid_s·ask_s · quote vol_bpm·bar5m·kline_rx_us · supply seconds) · 밀기(book·kline·supply1s)를 `Feed` 에 · 모르는 칸·type 무시 · 글자 칸 없는 옛 book 은 안 받아 둠 · 끊기면 1·2·4…30초 뒤 다시. 감시 5분봉·1초가 = 화면 심볼(USDT) 몫 · 호가·변동성 = `sync_cached`(나이는 rx_us → 한도 넘으면 REST).
- 시험: `ar` = tests/app_run.rs(가짜 거래소로 기동 → 남은 dbwt 정리 · 복구 파일로 롱 무장 · 포지션 없는 숏 정리 → 멈춤 · 재시작 쓰기 0 · 그림자 쓰기 0·목록 3 · 진짜 실행 파일 SIGTERM 두 번 · 설정 오류 4) · `hf` = tests/hub_feed.rs(붙음 → 받아 둔 값 REST 0 → 끊김 → 3초 뒤 REST → 재연결 스냅숏 → 다시 받아 둔 값 · 변동성 None = REST).
- ✅~~실행기 ↔ 접수 다리 결함~~ → 2026-11-21 고침(W12 행): `Desk::live` + `pull_live`·`push_live` + `runner::DeskCell`. 같이 잡은 원본과 다른 곳 4: 선 옮김 뒤 옛 계획으로 무장하던 것(원본은 같은 dict) · 감시 선을 시작 때 한 번만 읽던 것 · 칸 `pegged` 가 참값(원본 = 시각) · PEG 시작·끝 로그 없음. ponytail: 집행 로그는 작업 끝에 모아 감시 로그로(원본은 그 순간 print -- 사이에 다른 로그가 끼면 순서가 갈린다).
- ⚠시험 없는 것: `drain_tp_fills` 를 틱 앞에서 부르는 자리(Hub 가 run 안에 있어 밖에서 넣을 수 없다 -- 함수 자체는 wi 시험) · ~~사용자 스트림을 켠 실행~~ → `ar` stream_fill_*(2026-11-21).

### 그림자 기록 -- 파이썬이 기록해 줘야 하는 칸 (파이썬은 안 고쳤다 · 사용자 결정 필요)

지금 기록으로는 재생할 수 없다: 접수 기록 G18 `data/live/order_requests.jsonl`(대시보드 서버 `order_request_log` 미들웨어) = 시각·경로·인자·상태·오류·걸린 ms·기기·접속 주소뿐(계획·칸·요청 열·응답 없음) · `order_timing` 줄(live_manual_peg_execute `_order_timing` 82행) = 첫 주문 POST 의 단계별 ms 뿐 · 무장 파일은 덮어쓰기라 변화 이력이 없다.
제안 = 새 파일 `data/live/order_shadow.jsonl`, 한 줄 = `{"rec": ..., "chain": ...}`. chain id 는 접수마다 하나 -- `order_mark`(X 76행)가 만드는 dict 에 `chain` 을 넣으면 `_ORDER_T` 맥락이 create_task 로 집행까지 따라간다 · 감시 틱은 `bracket_tick`(orders.py 1128행) 첫 줄에서 새 id 를 맥락에 넣는다. 🔴키·서명·listenKey 값은 적지 않는다.

| # | 어디(파일:행 · 함수) | 무엇을 | 칸 |
|---|---|---|---|
| 1 | orders.py:965 `entry_then_bracket` 첫 줄 | 작업 시작(진입) | rec=start · op="entry" · t=time.time() · mono=time.monotonic() · plan(사본) · limit · slot=state 사본 · file=bracket_load() 원본 · limits=복구 파일 이름들 |
| 2 | orders.py:2045 청산 접수 create_task 직전 | 작업 시작(청산) | rec=start · op="exit" · plan · rearm(frac<1) · slot · file |
| 3 | orders.py:1128 `bracket_tick` | 작업 시작(감시 틱) -- 요청이 나가거나 파일이 바뀐 틱만(2초마다 hold 는 안 적음) | rec=start · op="tick" · t · mono · slot(manual_entry_state·side_exit_states) · file |
| 4 | X:92 `signed` 의 session.request 앞뒤 | 서명 왕복 하나 | rec=wire · t(보내기 직전 time.time) · mono · method · target = `path?` + url 의 `&signature=` 앞까지(timestamp 포함) · status · body = 응답 **원문 글자**(json() 전 `await response.text()`) · error = 예외 문자열(끊김·시간초과) |
| 5 | X:430 `maker_price`(bookTicker) · 계좌 모듈 `_clock_offset`(34행 · /time) · `load_filters`(live_manual_peg_entry 60행 · exchangeInfo) | 서명 없는 왕복 | 4 와 같은 칸(load_filters 캐시 적중이면 왕복 없음 -- 대신 10 의 filters) |
| 6 | orders.py:1134 swr_cached("binance_account") | 읽은 입력 | rec=input · what="account" · value=받은 dict 전체(ok·generated_at·positions) · t |
| 7 | orders.py:1151 load_market_history | 읽은 입력 | what="bars" · args=[asset] · value=받은 목록(마감 봉만이어도 됨) |
| 8 | orders.py:1002 `stop_touched` · 1160 tick_one 의 sec_prices | 읽은 입력(부를 때마다) | what="sec" · args=[asset] · value=since 뒤 행 [[초, 가격]…](재생은 since 뒤만 쓴다) |
| 9 | orders.py:1195·1250 assemble_exit_plan | 읽은 입력 | what="exit_plan" · args=[pside, frac, fresh, asset, only_symbol] · value={"plan": …, "err": [본문, 상태] 또는 null} |
| 10 | orders.py:1005 `limit_reprice` 의 fetch_binance_json·load_filters · 853 `basis_for` 의 기준 호가 | 읽은 입력 | what="book" · args=[symbol] · value=호가 dict · what="filters" · value=필터 dict |
| 11 | X:49 `wake_fills` | 체결 알림으로 대기가 일찍 깬 시각(조회 횟수가 갈린다) | rec=input · what="wake" · t · mono |
| 12 | orders.py `api_manual_wait_action`·`api_manual_limit_amend`·`api_manual_entry_cancel`(도는 작업의 칸에 action·amend 를 쓰는 핸들러) | 화면이 작업 도중 넣은 값 | rec=input · what="handler" · chain(그 칸의 도는 작업) · t · args=[핸들러 이름, 질의 dict] · value={status, body}(재생이 같은 핸들러를 그 시각에 다시 부르고 응답도 대조) |
| 13 | orders.py `entry_then_bracket` 끝 · `settle` 뒤 · `bracket_tick` 끝 | 작업 끝 | rec=end · t · mono · slot(사본) · file(bracket_load) · limits · raised · logs(그 작업의 print 줄 -- 어렵다면 생략, 재생은 칸이 없으면 안 본다) |
| 14 | G18 `order_request_log` 의 row | 접수 기록과 그림자 기록 잇기 | `chain` 칸 하나 더 |

크기 어림: 따라가기 120초 ≈ 요청 80 · 무장 1회 ≈ 5 · 감시 발동 ≈ 10 -- 하루 접수 수십 건이면 수 MB. 재시작 복구(wait_cleanup·limit_recover)까지 재생하려면 produce_account 결과(what="produce_account")와 고아 정리 왕복(4)도 같은 chain 으로.

## 계좌 알림 → Wired · 텔레그램 (2026-11-21)

- **스트림 계좌를 청산 조립의 «새 계좌»로**(설계 §4-2): `Wired.stream = Arc<Hub>`(Hub.stream_account · Hub.us). 쓸 수 있으면(연결됨 · 같은 연결 번호 · 60초 안 REST 대조 · 마지막 대조 어긋남 없음) REST 왕복 없이 그것, 아니면 REST 로 읽고 대조(이유를 `fresh_source` 에). 나이 = 마지막 REST 대조 시각(`STREAM_TRUST_SEC` 60초) -- 넘으면 REST. 감시 틱 캐시(10초)와 갱신기가 읽을 때마다 대조한다.
  - 정답지: `gen_desk_fixture.py` stream_rest 6사례(익절 부분 체결 · 익절 다 체결로 닫힘 · 대기 주문 부분 체결 추가 · 새 심볼 숏 · 주문 사라짐 · 한 방향 BOTH) -- 원본 `fetch_account` 를 사건 전·후 거래소 상태로 각각 돌린 계좌와, 사건 전 계좌 위에 같은 알림 열을 반영한 스트림 계좌가 포지션(심볼·측면·수량·진입가·미실현)·미체결(심볼·방향·가격·남은 수량·reduce_only·표식) 일치 · 대조 어긋남 0(`dk` stream_account_matches_python_rest_after_same_events).
  - `StreamIo::state` -- 스트림 루프가 연결·끊김·알림 시각·오류를 바꿀 때마다 Hub.us 사본을 맞춘다(원본은 같은 dict 를 같이 본다).
  - **번호(원본 acct_seq)** = `Hub::acct_seq()`(체결 알림 뒤 계좌 다시 읽기 성공마다 +1 · 처음 0) -- rtserve /api/events 가 쓸 공개 함수. `app::run(cfg, mk, stop, hub)` 에 바깥에서 Hub 를 넘겨 같이 쥔다.
  - ponytail: 갱신기와 감시(Wired)는 계좌 장부(last_good 30분)를 따로 쥔다 -- 한쪽이 실패해 묵은 값으로 떨어질 때만 다르다(합치려면 Wired.book 을 Arc<Mutex>).
- **텔레그램**: 원본 주문 경로가 보내는 자리는 `bracket_tick_alert` 하나(감시가 건너뛴 칸 · 칸·예외 종류마다 한 번 · html.escape · 뒤에서 최대 8초). `telegram::Telegram`(TELEGRAM_BOT_TOKEN·TELEGRAM_CHAT_ID 둘 다 있어야 · 원본 json.dumps ensure_ascii 바이트 · HTTP 오류·연결 실패 = false) · `app::Ops::alert` 가 spawn_blocking 으로 보냄(결과는 원본처럼 안 봄) · **그림자 = 안 보내고 `TELEGRAM <문구>` 를 «안 보낸 쓰기» 목록에**. 토큰은 Debug 에 안 나온다.
  - 정답지: `gen_telegram_fixture.py` -- 원본 두 함수를 소스에서 꺼내(모듈 import 안 함 · ops_watchdog 는 import 때 점-env 를 읽는다) 가짜 urlopen 으로 6호출(같은 칸·같은 종류 중복 1 · `<>&'"` · 줄바꿈·탭·DEL·이모지) → 로그 줄 6 · 보낸 요청 5(경로·메서드·Content-Type·본문 바이트) 일치(`tg` = tests/telegram.rs).
- 양성 대조 ELEVEN 11/11(스트림 미체결 수량 · BOTH 측면 · 번호 · clear · 상태 사본 · 대조 · 시작 게이트 · html.escape · ensure_ascii · 그림자 전송 · HTTP 오류) · WIRE 첫 항목을 새 자리(wire.rs)로 옮김.
- 서버 그림자 가동에 필요한 것: `SHADOW.md`.

