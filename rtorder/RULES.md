# rtorder 규칙 추적표 (2026-11-20 · 첫·둘째 조각)

점검표 = `docs/rtcore_order_rules_inventory_20261120.md`(145개). 첫 조각 = 계획 순수 함수(`src/plan.rs`) + 지정가 집행 상태 기계(`src/exec.rs`). 둘째 조각 = 서명·요청 문자열(`src/sign.rs`) + 전송(`src/http.rs`, `Exchange` 구현) + 손절·익절·레버리지·고아 정리(`src/bracket.rs`) + 감시 진입(`exec::run_entry_stop`). 어디에도 연결하지 않았다.

상태: **옮김** = Rust 에 있고 시험이 지킨다 · **부분** = 규칙의 일부만 이 조각에 있다(나머지 몫과 조각을 적음) · **범위 아님** = 다음 조각 · **⚠시험 없음** = 옮긴 몫에 지켜 주는 시험이 없다.
시험 약어: `fx` = tests/plan_fixture.rs(파이썬 정답지 2,266건) · `sc` = tests/plan_selfcheck.rs(원본 자체점검 이전) · `inv` = tests/exec_invariants.rs · `rp` = tests/http_replay.rs(원본 파이썬 + 진짜 aiohttp 정답지 112시나리오 -- 요청 열 글자까지·결과·밴 기록·연결 수) · `mesh` = tests/http_mesh.rs(가짜 거래소를 HTTP 뒤에) · `mut` = tools/mutate.py(양성 대조).
다음 조각 이름: **접수**(라우트·confirm·게이트·잠금·칸) · **조립**(assemble_entry/exit_plan·계좌·풀·basis) · **감시**(bracket_tick·무장 파일·arm_bracket 등 orders.py 쪽) · **대기 관리**(대기 칸·복구 파일·재시작 훅) · **계좌·알림**(사용자 스트림).

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
| P14 | 부분 | `resolve_exit_position` · fx · sc · 새 계좌 읽기·감시 심볼 한정은 조립·감시 |
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
| W1 | 범위 아님 | 대기 관리 |
| W2 | 범위 아님 | 대기 관리 |
| W3 | 부분 | `LIMIT_MAX_SEC`(4시간) 만료 취소 · inv `limit_paths`(만료) · `e25_stop_entry_paths`(감시 진입 만료) · 기다리기 마감 잔량 시장가는 미정 M3 |
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
| B4 | 옮김 | `bracket::place_bracket`(포지션 새로 읽어 그 측면 전량 · 옛 우리 주문 정리 → TP GTX + 비상 스탑 · strict · tp·스탑 없으면 placed=true(원본 all([]))) · rp(11사례: 정상·조회 실패·포지션 없음·익절 거부·익절 불명·strict·정리 실패 뒤 계속·숏 익절만·스탑만·둘 다 없음·키 없음) · mut «strict 무시» · 🔴불명 응답 조회 없음(원본 그대로, 남은 위험) |
| B5 | 옮김 | `bracket::post_stop` → `/fapi/v1/algoOrder` CONDITIONAL STOP_MARKET closePosition=true MARK_PRICE · rp(URL 글자까지) · mut «MARK_PRICE → CONTRACT_PRICE» · ⚠쓸지는 미정(시장가형 -- M10, 사용자 결정 목록) |
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
| B21 | 부분 | 거래소 쪽 = `cancel_bracket(only_sl)`·`clear_bracket` · rp · 판정 sl_off 는 `bracket_action`(fx) · 해제 API·«실패면 아무것도 안 바꿈»은 접수 |
| B22 | 부분 | 규칙 무장 다시 켜기의 거래소 쪽 = `bracket::place_stop`(포지션 확인 → 남은 우리 스탑 정리 → 못 지우면 안 걸음 → 스탑 · -2021 그대로 실패) · rp(6) · mut «옛 스탑 안 지움» · 선 무장 감시 재개·409 는 접수·감시 |
| B23 | 옮김 | `bracket_action` sl_off + fired "tp"/True · fx · sc |
| B24 | 부분 | 끄기의 거래소 쪽 = `cancel_bracket(only_tp)`(tp_open 함께) · rp · `tp_watch_only`·켜기 검사는 접수·감시 |
| B25 | 범위 아님 | 접수·감시 |
| B26 | 범위 아님 | 접수·감시 |
| B27 | 옮김 | `split_tp_qty` · fx(194) · sc · mut «carry 버림» |
| B28 | 옮김 | `bracket::place_tp_ladder`(옛 익절 다 지운 뒤에만 · 사라진 조각은 직접 조회해 FILLED 일 때만 체결 · 필터는 조각이 없어도 읽음 · split_tp_qty) · rp(9사례: 정상·사라진 조각 체결/취소·전부 체결·먼지 합침·정리 실패·필터 예외·조회 실패·포지션 없음·키 없음) · mut «조회 없이 체결로» · `ladder_apply`(O) 는 접수 |
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
| R1 | 부분 | `bracket::cancel_wait_orphans`(dbwt·dbpg 만 · 목록 실패는 심볼별 오류로 · 지운 것의 체결·수량) · rp(2) · mut «dbtp 도» · 재시작 훅·`restart_cancelled` 기록은 대기 관리 |
| R2 | 범위 아님 | 대기 관리 |
| R3 | 범위 아님 | 접수(`settle`) -- 집행 함수는 예외를 내지 않고 항상 종료 상태를 남긴다(inv ③ · fuzz 400건) |
| R4 | 부분 | 취소 확인 실패·지정가 접수 불명·PEG/감시 진입 호가 실패 → `Phase::Error` · inv03·05·07·d2·e25 · mesh(HTTP 로도 같은 error) · API 키 없음 = `Binance::has_keys`(브래킷 함수들이 원본 문구로) -- 집행 함수(run_*)의 키 검사는 호출부 · 스위칭은 접수 |
| R5 | 범위 아님 | 접수 |
| R6 | 범위 아님 | 접수 |
| R7 | 옮김 | `Binance::nap` = tokio select(잠 · 알림) -- future 를 버리면 바로 끝나 «깨움·취소가 같은 틱이면 취소를 삼킴»(gh-86296)이 생길 자리가 없다 · mesh `nap_wakes_early_on_fill_alert` |
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
| A10 | 부분 | 알림은 앞당기기만: `nap` 으로 일찍 깨도 3초 판단 그대로 · inv `e16_…` · `Binance::wake`(Notify) · mesh `nap_wakes_early_on_fill_alert` · 알림 수신은 계좌·알림 |
| A11 | 범위 아님 | 계좌·알림 |

## 집계

| 상태 | 첫 조각 | 둘째 조각 뒤 |
|---|---|---|
| 옮김 | 38 | 47 (+E1·E2·E19·E25·R7 부분→옮김 · +P29·B4·B5·B28 범위 아님→옮김) |
| 부분 | 22 | 21 (−5 위 · +P16·B22·B24·R1 범위 아님→부분 · B21 은 부분 그대로, 거래소 쪽 몫 추가) |
| 범위 아님 | 85 | 77 |
| 미정(사용자 결정) | 0 | 0 -- 규칙 줄 자체는 없음. 시장가 분기 13곳은 아래 따로(B5 는 «옮김 · 쓸지 미정») |
| 옮긴 몫에 시험 없음(⚠) | 4 | 2 (E17 구조 · B2 충돌 순서) -- E1 은 mesh 단언, R7 은 mesh 시험으로 덮음 |

## 시장가 분기 M1–M13 (모두 미정 -- 사용자 결정 대기 · 옮기지 않음)

| # | Rust 에서 닿는 자리 | 결과 |
|---|---|---|
| M1 | `exec::run_entry` 따라가기 마감 잔량 | `NeedsMarketDecision` · `market_branch="M1"` · inv02·10·e18 음성 대조 |
| M2 | `exec::run_entry` 계획 `market=true` | 같은 갈래 "M2" · inv `maker_only_and_market_entry_plans` |
| M3 | `exec::run_entry_wait` 마감·action=market | "M3" · inv `wait_paths` |
| M4 | `exec::run_entry_limit` action=market(취소 확인 뒤) | "M4" · inv `limit_paths` |
| M5 | `exec::run_entry_stop` 닿기 전 action=market | `run_entry(market)` 뒤 `market_branch="M5"` · inv `e25_stop_entry_paths` |
| M6·M7 | `plan::build_exit_plan` 이 계획을 MARKET 으로 만든다(원본 값 그대로 -- 정답지 일치) → `exec::run_exit` 가 "M6/M7/M8" 로 멈춘다 | inv09 · fx |
| M8 | 조립(`market=True`) -- 집행에서는 M6/M7 과 같은 갈래 | inv09 |
| M9 | `exec::run_exit` 마감 잔량 | "M9" · inv02 |
| M10 | 비상 스탑 `bracket::post_stop`(place_bracket·place_stop 이 부름) -- **요청 모양만 옮겨 정답지와 맞춤**, 시장가형(발동 시 전량 시장가) | 쓸지 사용자 결정 · rp |
| M11 | 스위칭 -- 접수 조각 | -- |
| M12 | `bracket::place_add_order` GTC(원본 그대로 옮김 -- 이미 그 아래면 테이커 체결) · 부르는 쪽(arm_rule_add)은 감시 | 쓸지 사용자 결정 · rp · mut «GTC → GTX» |
| M13 | 대기 동작 접수 -- 접수 조각 | -- |

## 원본과 일부러 다르게 둔 곳

- 시장가 13곳 → 위 표(보내지 않고 멈춤).
- `run_entry`·`run_entry_wait`·`run_entry_limit`·`run_exit` 의 «API 키 없음 → error» · `_clock_offset` · `ensure_leverage` 호출 → 접수 조각(함수 자체는 `http`·`bracket` 에 있다).
- 주문 응답이 dict 가 아니거나 executedQty 가 숫자가 아니면 원본은 예외(settle → error), Rust 는 숫자로 시작하지 않는 오류 문자열(접수 = «나갔는지 모름», 조회 = «재시도») -- `http::reply` ponytail.
- JSON 해석 오류 문구(JSONDecodeError 뒤)는 파이썬과 다르다(앞머리만 같다) · 밴 기록의 429 해제 시각은 실제 시계라 정답지와 폭만 비교.
- 원본은 키를 호출마다 환경에서 읽는다 -- `Binance::from_env` 는 만들 때 한 번.
- 파이썬 예외(0 나누기·inf→int·빈 목록 인덱스) → 같은 종류의 `PlanError`(정답지가 종류·문구까지 비교).
- ~~D2(«나갔는지 모르는 주문» 처리가 세 경로에서 다름)는 통일하지 않았다~~ → **해결(2026-11-20, 사용자 결정)**: 파이썬 원본을 먼저 고치고(`resolve_unknown_post` 공용 함수) Rust 도 같게 맞췄다 -- 원본과 다른 곳이 아니다. inv `d2_unknown_outcome_same_on_all_paths` 가 «세 경로가 같다»를 고정한다.
