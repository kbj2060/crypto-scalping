# 실주문 경로 규칙 목록 — Rust(rtcore)로 옮기기 전 (2026-11-20)

배경: `docs/rtcore_design_20261120.md` §2·§4·§6. 이 문서는 **옮길 때 하나라도 빠뜨리면 오발주·손절 누락이 되는 규칙**을 코드·주석에서 그대로 뽑은 목록이다. 코드는 고치지 않았다. 시험은 실행하지 않았다(읽기만).

## 읽는 법

- **줄 번호는 2026-11-20 읽은 시점 기준**이다. 같은 워크트리에서 다른 에이전트가 `dashboard/orders.py`·`live/execution/…` 에 측정 코드를 넣는 중이라 몇 줄 밀릴 수 있다 — 반드시 **함수 이름**으로 찾는다.
- 파일 약어
  - **O** = `dashboard/orders.py` (계좌·접수 문·계획 조립·브래킷 감시·대기·사용자 스트림)
  - **X** = `live/execution/live_manual_peg_execute_20260912.py` (실제 전송·지정가 따라가기·취소 확인)
  - **P** = `scripts/live_manual_peg_entry_20260912.py` (무엇을 보낼지 정하는 순수 함수)
  - **S** = `dashboard/server.py` (라우트·캐시·세션·가격 원천 — Read 도구로 일부만 읽음)
- 시험 약어(모두 `test/` 아래, 이름은 `파일::함수`)

| 약어 | 파일 |
|---|---|
| inv | test_manual_peg_execute_invariants_20260913.py (`main()` 안 번호 시나리오 #1~#10) |
| smoke | test_manual_entry_preview_smoke_20260913.py |
| rule | test_rec_rule_entry_20261009.py |
| radd | test_rule_auto_add_20261014.py |
| lines | test_chart_lines_entry_20261028.py |
| disarm | test_bracket_disarm_20261015.py |
| prot | test_bracket_protection_20260926.py |
| wait | test_manual_entry_wait_20261010.py |
| cancel | test_manual_entry_cancel_20261011.py |
| limit | test_manual_limit_entry_20261031.py |
| limsrv | test_manual_limit_entry_server_20261031.py |
| stop | test_manual_stop_entry_20261109.py |
| many | test_add_entry_many_waits_20261118.py |
| switch | test_manual_switch_server_20261010.py |
| xmkt | test_exit_exec_switch_market_20261010.py |
| xall | test_manual_exit_all_20261022.py |
| prx | test_partial_exit_rearm_tp_20261104.py |
| us | test_user_stream_20261109.py |
| nsum | test_notional_sum_scope_20260919.py |
| pool | test_pool_balance_20261105.py |
| fee | test_commission_rate_display_20261010.py |
| olog | test_order_request_log_20261112.py |
| Pself | P 의 `_self_check` · `_self_check_rec_rule` (test/ 밖 자체점검, `python scripts/…` 로 돈다) |
| Xself | X 의 `_self_check` |

- «시험» 칸은 **시험 함수 이름·설명문으로 대응시킨 것**이다(inv 만 본문까지 읽었다). 본문이 그 규칙을 실제로 검사하는지는 inv 외에는 확인하지 않았다 — 그런 칸은 «(이름 기준)».
- 용어: 지정가 = LIMIT · GTX = post-only(메이커로만 걸리고, 지금 체결될 값이면 거래소가 -5022 로 거부) · 시장가 = MARKET(테이커) · 헤지 모드 = 같은 심볼에 롱·숏이 따로 열리는 계좌 설정 · basis = 주문 심볼(USDC) 가격 ÷ 차트 심볼(USDT) 가격 · 무장 = 서버가 손절·익절을 감시하는 상태(`manual_bracket_state.json` 한 칸) · 따라가기(peg·리페그) = 최우선 호가에 GTX 를 걸고 앞질리면 취소 확인 뒤 새 호가에 다시 거는 것.

---

## 1. 접수 문(게이트) — 18개

| # | 규칙 | 어기면 | 위치 | 근거·날짜 | 시험 |
|---|---|---|---|---|---|
| G1 | 주문을 바꾸는 요청은 전부 **POST** 로만 받는다(미리보기·상태만 GET). | 링크 미리보기·프리페치·주소창 재방문이 그대로 주문 | S:4963-4976(라우트 등록) · O:1765 `api_manual_entry_submit` docstring | 주석 «GET 이면 … 그대로 주문» | 없음 |
| G2 | 주문 요청은 **`confirm=1`** 이 없으면 400. 진입·청산·전량 청산·손절 해제/다시 켜기·익절 끄기/켜기·선 추가/수정·분할 조각·지정가 수정·대기 동작·진입 취소 **12개 전부**. | 프런트 버그가 곧 오발주 | O:1773 `api_manual_entry_submit` · 2149 `api_manual_exit_submit` · 2211 `api_manual_exit_all_submit` · 2268 `api_manual_bracket_disarm` · 2299 `…_rearm` · 2438 `…_tp_legs` · 2485 `…_tp` · 2570 `…_add` · 2667 `…_modify` · 2772 `api_manual_limit_amend` · 2833 `api_manual_wait_action` · 2878 `api_manual_entry_cancel` | 주석 «프런트만 믿으면 프런트 버그가 곧 오발주» | lines::test_confirm_required · disarm::test_guards · xall::test_guards |
| G3 | 실행 스위치 `DASHBOARD_MANUAL_EXEC_ENABLED`(1·true·yes)를 **매 요청마다 환경에서 다시 읽는다**. 꺼져 있으면 403. 미리보기는 꺼져 있어도 같은 계획을 돌려준다(`dry_run`). | 코드 기본값으로 실주문 / 꺼졌는데 주문 | P:54 `exec_enabled` · O:1796·2170·2216·2270·2301·2439·2487·2572·2669·2774·2836·2880 | 주석 «코드 기본값이 아니라 서버 .env 가 정답» | disarm::test_guards · xall::test_guards |
| G4 | 스위치가 꺼지면 **손절 감시 틱도 아무것도 안 한다**(무장 파일은 그대로). 재시작 고아 정리·사용자 스트림도 스위치가 켜진 곳에서만 시작. | 옮긴 뒤 감시가 꺼진 개발기에서 실주문 / 반대로 켜진 서버에서 감시가 안 돎 | O:1170 `bracket_tick` · 2919 `start_wait_cleanup` · 3031 `start_user_stream` | — | us::test_gate_off_starts_nothing (감시 틱 쪽 없음) |
| G5 | 진입·스위칭은 **계좌 전체에 칸 하나**(`manual_entry_state`). `working`·`submitting` 이면 새 진입은 409. | 두 진입이 겹쳐 상한 무력화 | O:638 · 643 `_busy` · 1800 `api_manual_entry_submit` | 주석 «겹치면 단건 상한의 뜻이 흐려진다» | wait::test_server_flow (이름 기준) |
| G6 | 청산은 **코인별 칸**: 위 칸이 비었으면 위 칸, 다른 코인이 쓰는 중이면 그 코인 칸, 같은 코인이 쓰는 중이면 409. | 한 코인 손절 청산(최대 2분) 동안 다른 코인 청산이 막힘 / 같은 코인 이중 청산 | O:641-653 `exit_slot` | 2026-10-20 사용자 «청산만 코인별 허용» | xall::test_all_coins_at_once_and_no_double_submit |
| G7 | 계획 조립(await) **뒤에 칸을 다시 확인**하고 바뀌었으면 409(손절 청산이 그 사이 칸을 잡았을 수 있다). | 같은 포지션 두 번 청산 | O:2191 `api_manual_exit_submit` · 2251 `api_manual_exit_all_submit` · 1833 `api_manual_entry_submit`(대기) | 주석 «두 번 내지 않는다» | xall::test_all_coins_at_once_and_no_double_submit |
| G8 | 주문 잠금 `manual_order_lock`: 잠겨 있으면 즉시 409(기다리지 않음). 진입 제출·청산 제출·전량 청산·지정가 수정·익절 on/off·분할 조각·선 추가·선 수정이 잠금 안. **대기 동작(wait-action)·진입 취소·손절 해제·손절 다시 켜기는 잠금 밖**. | 폰+데스크톱 동시 접수가 검사를 빠져나가 두 번 나감 | O:3035 `busy_or` · 655 · S:4963·4967·4970-4976(잠금) · S:4965·4966·4968·4969(잠금 밖) | 2026-10-16 «접수 검사~상태 점유 사이 await 동안 두 번째 접수가 … 두 번 나갔다» | 없음 |
| G9 | 감시 틱은 **주문 잠금이 잠겨 있으면 손절 발동을 건너뛰고 다음 틱에** 다시 본다. | 사용자 주문과 손절 청산이 겹쳐 이중 청산 | O:1218·1230·1259·1285 `bracket_tick` | 검토 B1 | 없음 |
| G10 | 같은 코인에 대기 진입이 있으면 새 진입 409. 예외: 걸린 대기가 **모두 같은 쪽 «추가 진입 대기(지정가·감시)»**이고 새 주문도 그렇고, 같은 종류·같은 선(USDT)이 아니며, 칸이 10개 미만일 때. | 대기 체결과 새 진입이 겹쳐 상한 초과 | O:1803-1813 · 1832-1841 `api_manual_entry_submit` | 2026-11-18 사용자 «추가 진입을 여러 개» · 검토 11-18 #2 | many::test_many_same_side_and_dup_refused · many::test_non_add_second_is_refused · limsrv::test_new_entry_blocked_while_limit_waits |
| G11 | 같은 종류·같은 선 대기가 이미 있으면 거절(응답이 끊겨 재시도·두 기기 중복). | 같은 지정가 두 벌 | O:1807 `dup` | 검토 11-18 #2 | many::test_many_same_side_and_dup_refused |
| G12 | 끝난 대기 칸이라도 그 칸의 집행 작업(체결 뒤 익절 다시 걸기)이 아직 돌면 새 대기에 주지 않는다. | 체결 뒤 처리 중인 칸 상태를 새 주문이 덮음 | O:1838-1842 | 검토 11-18 #2 | many::test_slot_not_reused_while_old_task_rearms |
| G13 | 대기가 있으면 **스위칭** 409(그냥 청산은 됨). | 반대 진입 뒤 옛 대기 체결 | O:2178 `api_manual_exit_submit` | — | 없음 |
| G14 | 주문 코인은 `MANUAL_EXEC_SYMBOLS` 표(eth·sol·xrp·btc·sndk·soxl)만. 시세만 보는 코인(skhynix·mu)은 주문 막힘. 손절/익절 조작 엔드포인트는 `symbol` 이 그 표의 값이어야 한다. | 지원 안 하는 심볼로 주문 | O:347-353 · 340 · 1933 `query_asset` · 2274·2305·2442·2491·2580·2673 | 2026-11-13 | smoke::test_bad_inputs_are_rejected_not_crashed (이름 기준) |
| G15 | 입력은 **조용히 고치지 않고 400**: side LONG/SHORT · 비율 0<pct≤100(없으면 100%) · 레버리지 1..거래소 최대(벗어나면 모델값) · 규칙 c/h(꺼져 있거나 다른 값이면 400) · 선 tp·sl 둘 다 양수·유한 · limit 은 선과 같이 · 기다리기 15/60분. | 절반 청산이 전량 / 직접 모드로 몰래 전환 | O:1938 `query_leverage` · 1952 `query_rule` · 1959 `query_lines` · 1970 `query_limit` · 1981 `query_fraction` · 1793 | 주석 «조용히 1.0 으로 떨어뜨리지 않는다» | smoke::test_bad_inputs_are_rejected_not_crashed · rule::test_bad_rule_is_rejected_not_ignored · limsrv::test_limit_needs_lines_and_valid_direction |
| G16 | 스위칭은 **전량(100%) 청산 + 반대 방향**만. | 일부 청산 뒤 반대 진입(양쪽 다리) | O:2163-2169 `api_manual_exit_submit` | — | switch::test_bad_switch_rejected |
| G17 | 자동 규칙 진입은 RULE_ASSETS(eth·xrp)만 · 그 코인 권장 배수가 없거나 15분 넘게 묵으면 503 · 마진 방식이 교차(cross)가 아니거나 못 읽으면 409. | 격리 마진에서 손절보다 청산이 먼저(10-09 실사고) | O:1558-1569 `assemble_entry_plan` · X:94 `margin_type` | 2026-10-09 «격리 20배는 청산 −4.6% 가 3σ 손절보다 안쪽» | rule::test_missing_or_stale_vol_blocks · rule::test_isolated_or_unknown_margin_blocks |
| G18 | 주문을 바꾸는 POST 는 전부 접수 기록(시각·경로·인자·결과·걸린 시간·기기·IP)을 한 줄씩 남긴다. | 사고 뒤 «누가 눌렀나»를 못 가림 | S:1312-1341 `order_request_log` | 2026-11-12 «02:45 SOXL 50% 청산이 누른 건지 가릴 수 없었다» | olog (파일 단위) |

## 2. 계획 조립 — 30개

| # | 규칙 | 어기면 | 위치 | 근거·날짜 | 시험 |
|---|---|---|---|---|---|
| P1 | **미리보기와 실주문은 같은 함수**(`assemble_entry_plan` → `build_entry_plan`, `assemble_exit_plan` → `build_exit_plan`)를 지난다. 제출은 프런트가 계산한 수량을 받지 않고 다시 조립한다. | «미리보기와 다른 주문» | O:1358 `assemble_entry_plan` · 1994 `assemble_exit_plan` · 1814 · 2181 | 주석 «두 곳에 복사해 두면 언젠가 한쪽만 고쳐져» | smoke::test_entry_preview_handler_completes · rule::test_submit_carries_rule_without_sending_orders |
| P2 | 계좌를 **못 읽으면 503**(진입·청산 모두). 못 읽음을 «포지션 없음»으로 돌리지 않는다. 사이징 없음 503 → 계좌 503 → 호가·필터 실패 순서. | 헐거운 상한 / 감시가 «정리»로 가서 익절·스탑 삭제 | O:1389-1396 · 2024-2027 · 2220 `api_manual_exit_all_submit` | 09-26 사고 · 주석 «못 읽음 = 모름» | prot::test_position_lookup_failure_is_account_failure |
| P3 | 상한은 **같은 증거금 풀(USDC/USDT)의 계좌 전체 명목**, 롱·숏 **절대값 합**(상쇄 가정 없음). 이 심볼만 센 값은 청산 거리 투영에만. | 다른 심볼 포지션만큼 상한이 헐거워짐 | O:496 `notional_sum` · 520 `pool_positions` · 1397-1406 | 2026-09-19(봇 ETHUSDT + 수동 ETHUSDC) · 11-05 | nsum · pool |
| P4 | 다른 심볼에 걸린 **늘리는 지정가**(reduce_only 아님·같은 풀)는 체결되면 같은 증거금을 먹으므로 여유에서 미리 뺀다. | 물타기 지정가가 다른 코인 진입 뒤에도 그 크기로 남음 | O:1407-1413 | 2026-10-20 감사 | rule::test_other_symbol_resting_order_reduces_room |
| P5 | 다른 대기 진입의 **남은 몫**도 뺀다 — 다른 코인 대기는 «걸린 주문 합»과 «대기 잔량 합» 중 큰 쪽, 같은 코인 다른 대기는 합. 수정 중인 자기 칸(skip_wait)은 뺀다. | 대기가 다 체결되면 상한 초과 | O:1414-1429 | 2026-11-10 · 11-18 검토 #3·#4 | many::test_room_counts_other_waits · limsrv::test_other_coin_limit_while_eth_waits_and_same_pool_reserve |
| P6 | 수량은 **가용 잔고 × (1−2%) × 주문 레버리지** 안으로 자른다(순자산 아님). 아직 거래소에 안 걸린 대기 몫(대기 표식 dbwt·dbpg 주문만 셈, dbtp 는 안 셈)의 증거금을 가용에서 미리 뺀다. | -2019 로 거부되고 SL/TP 도 안 걸림 | P:98-100 `MARGIN_BUFFER` · 164-171 `build_entry_plan` · O:1583-1611 | 2026-11-07 ETHUSDC 사고 · 검토 11-18 #1 | pool (이름 기준) |
| P7 | 순자산·가용은 **그 심볼이 증거금을 먹는 풀**의 값. | SOXLUSDT 에 USDC 순자산 → 크기 어긋남 | O:513 `pool_balance` · 1433 · 1604 | 2026-11-05 | pool |
| P8 | 상한을 넘으면 **막지 않고 남은 여유까지 줄인다**, 줄였다는 사실을 notes·`capped` 로 남긴다. 여유가 0 이하면 `blocked`. | 조용히 줄어든 주문 / 막혀야 할 주문이 나감 | P:142-163 · 184-190 | 2026-09-26 사용자 «넘치면 50%까지만» | Pself · rule::test_full_cap_blocks_add |
| P9 | 분할 비율은 **상한에 자른 뒤** 곱한다(권고에 곱하지 않음). | «33%»가 상한의 71% | P:172-181 | 2026-09-13 실측 | Pself |
| P10 | 비율이 (0,1] 밖이면 ValueError(진입·청산 둘 다). | 절반이 전량 / 0 이 «안 닫힘» | P:131 · 592 | — | Pself |
| P11 | 수량은 격자 **내림**. 진입가: 롱 = 최우선 매수호가 내림 · 숏 = 매도호가 **올림**. 청산은 거울(롱 청산 = 매도호가 올림 · 숏 청산 = 매수호가 내림). | 반올림으로 상한 초과 / 한 틱 공격적 → GTX 거부 | P:83 `_floor_to` · 90 `_ceil_to` · 134 · 597 | — | Pself |
| P12 | 최소 수량·최소 명목 미만이면 진입은 `blocked`. 지정가 진입은 가격비로 다시 잰 수량을 한 번 더 검사. | 거래소 거부 | P:184-190 · O:1689 | — | 없음(서버 경로) |
| P13 | 헤지 모드: **positionSide 필수**, **reduceOnly 를 보내지 않는다**(-1106). 과청산 방어는 **수량뿐**. | 롱/숏 상계 / 청산이 반대 신규 진입 | P:198 · 578-581 `build_exit_plan` · X:858 `run_exit` | — | Pself(«reduceOnly» 없음 검사) |
| P14 | 청산 수량은 **방금 읽은 포지션**(fresh, 캐시 우회)에서, 심볼은 **설정이 아니라 실제 열린 포지션**에서(진입 심볼 우선, 나머지는 `other_symbol_open` 으로 알림). 감시 청산은 **무장한 심볼만**. | 이미 닫힌 포지션을 다시 닫아 반대로 열림 / USDC 를 지키다 USDT 포지션을 닫음 | O:2008-2032 `assemble_exit_plan` · P:655 `resolve_exit_position` · O:1228·1280 `only_symbol` | 2026-09-19 · 11-12 검토 | Pself · lines::test_does_not_close_other_symbol |
| P15 | 부분 청산 수량은 포지션을 절대 넘지 않고, 전량이면 곱셈을 하지 않는다(부동소수 «거의 전량» 방지). | 과청산 / 먼지 잔량 | P:599-602 | — | Pself |
| P16 | 거래소 레버리지 = 게이지 값(있으면) 또는 모델 추천. 주문 전에 `ensure_leverage`: 이미 같으면 안 보냄, 실패해도 **주문은 보내고** 결과를 state 에 싣는다. | 천장이 순자산 30배로 남음(조용히) / 들어가야 할 때 못 들어감 | O:1707-1709 · X:147-167 `ensure_leverage` · 363-370 `run_entry` | 2026-09-13 | smoke::test_leverage_gauge_overrides_model · Xself |
| P17 | 자동 규칙 배수 L = min(12.4×권장배수, 청산 안전(max(3σ24, 5%)), 상한 2배 · «보유» 1.5배). 실효 L ≤ 0.9 × 거래소 레버리지. 첫 진입 = 이 방향 포지션 없음(반대 다리는 안 셈) → 상한의 1/3, 물타기 = 남은 여유. | 증거금 100% 로 거부 / 반대 다리 때문에 첫 진입이 98% | P:268-319 `commit_rule_l` · `rec_rule_size` · O:1570-1582 | 10-11·10-13·10-14·10-25 사용자 결정 | rule::test_first_entry_is_third_of_cap_and_stop_5pct · test_hold_mode_caps_at_1_5 · test_low_exchange_leverage_binds_rule_l · test_opposite_leg_does_not_turn_first_entry_into_add |
| P18 | 규칙 물타기: 이번 주문이 손절선까지 잃을 몫 ≤ L×5%×순자산 − 이미 든 몫의 손절 손실. 손절선 너머면 0. | 손절 1회 손실 한도 초과 | P:313-316 · O:1572-1578 | 2026-10-10 | rule::test_add_after_full_then_drop_keeps_loss_at_stop · Pself |
| P19 | 자동 규칙 손절이 무장된 포지션이 하나라도 있으면 **모든 수동 진입**도 계좌 명목 ≤ 순자산 × 2. | 다른 코인을 크게 넣어 «손절보다 청산이 멀다» 보장이 깨짐 | O:1485-1489 | 2026-10-20 감사 | rule::test_manual_entry_shares_rule_cap_when_rule_armed |
| P20 | 선 주문이 아닌 **모든 진입에 −5% 거래소 손절**(기준 = 첫 진입가, 같은 쪽 포지션이 있으면 평단). 이미 규칙 손절이 무장돼 있으면 **그 손절을 그대로**(물타기해도 안 물러남). 익절 없음. | 손절 없는 포지션 / 물타기로 손절선 후퇴 | P:347-364 `rec_rule_bracket` · O:1536-1556 | 2026-10-24 사용자 «손절은 −5% 하나로» | rule::test_every_entry_gets_minus5_stop · test_add_uses_remaining_room_and_keeps_armed_stop · test_add_without_rule_arming_anchors_on_average · smoke::test_bracket_is_minus5_stop_without_liquidation_levels |
| P21 | basis = 주문 심볼 중간가 ÷ USDT 중간가. 못 읽으면 1.0. | (손절가는 주문 심볼 가격이라 영향 작음) 5분봉 감시 레벨 수 bp 어긋남 | O:860-868 `basis_for` | — | 없음 |
| P22 | 지정가 진입가 = 진입선(USDT) × basis, **롱 내림 · 숏 올림**, 지금 호가를 넘으면(= 지금 체결) 400(감시 진입은 예외). 명목은 지금가 크기 그대로, 수량만 가격비로 다시 잼. | 지정가가 테이커로 / 숏을 올릴수록 명목 커짐 | O:1679-1690 | 2026-10-31 | limsrv::test_preview_limit_is_entry_line_and_keeps_notional · test_short_limit_rounds_up_and_marketable_is_rejected |
| P23 | 자동 재조정 목표가: basis 를 못 읽어 1.0 이면 None(안 바꿈), 호가를 넘을 값이면 None. | 낡은 basis 로 주문가 틀기 / 몰래 테이커 | O:1014-1028 `limit_reprice` | 검증 S1 | limsrv::test_reprice_callback_ignores_unread_basis_and_never_crosses |
| P24 | 선 브래킷 검사는 **방향만**(롱 익절 > 기준가, 손절 < 기준가 · 들어간 포지션 선 수정은 손절이 어느 쪽이든 받음). 거리 제한 없음. 익절 주문가 = 선 × basis 를 불리하지 않게(롱 올림 · 숏 내림). 거래소 스탑 없음(backstop None). | 반대쪽 익절 / 거리 제한이 남아 사용자가 놓은 곳에 못 검 | P:372-391 `line_bracket` | 11-11 최소 거리 삭제 · 11-14 «±15% 제한 자체를 제거» | lines::test_line_validation · lines::test_lines_hugging_price_go_out_as_dragged |
| P25 | 선 주문은 그 코인의 **1초 체결가 원천이 있어야** 받는다(없으면 400). | 손절 감시 불가 포지션 | O:1660 | — | 없음 |
| P26 | 같은 쪽 포지션이 있으면 **보낸 선은 쓰지 않는다**: 선 포지션이면 무장된 선·주문가 그대로 크기만(`add_keep`), 선 포지션이 아니면 선을 버리고 일반 추가(지정가 추가는 409). 선 포지션에 자동 규칙 크기는 409. 청산 진행 중이면 409. | 추가 진입이 기존 익절·손절을 덮음 | O:1539-1552 · 1645-1678 | 11-03 · 11-12 · 11-14 사용자 | lines::test_same_side_position_ignores_sent_lines · test_plain_add_on_lines_position_keeps_lines · test_preview_uses_armed_lines_not_the_sent_ones · test_rule_sizing_refused_on_lines_position · test_refused_when_fired_and_line_tool_on_plain_position_adds |
| P27 | 청산 마감 = 변동성 연동 `1500 / vol^1.472` 를 15~120초로 자름. 변동성을 모르면(None) **120초**(0 이 아님). 집행은 **계획의 마감**을 쓴다(모듈 상수 아님). | 빠른 장에서 불필요한 노출 / 변동성 연동 무력화 | P:561 `exit_deadline_sec` · 548 `realized_vol_bpm` · X:888 `run_exit` | 2026-09-13 섀도우 7,982legs | Pself · Xself |
| P28 | 실주문 청산 조립(`for_order=True`)은 화면용 펀딩·수수료 조회를 건너뛴다(수량·가격·시장가 판단은 같음). | 접수 지연(속도) | O:1998 · 2039-2040 | — | 없음 |
| P29 | 계정 수수료율은 **화면 «예상 비용» 표시 전용**. 크기·손절·판정은 표준 상수. | 프로모션 0% 를 경제성에 씀 | X:107-132 `commission_bp` | 2026-10-10 사용자 정책 | fee::test_commission_rate_cache_and_plan_cost |
| P30 | 사이징 스위치는 지금 값 그대로 옮긴다: 위험모델 끔(`SIZING_RISK_MODEL_ENABLED=False`) · 모델 상한만(`SIZING_CAP_MODEL_ONLY=True`) · 증거금 % 상한 끔(`SIZING_MARGIN_CAP_PCT=0`) · 규칙 켬 · 자동 물타기 켬 · 상한 무시 환경변수 `DASHBOARD_SIZING_CAP_EQUITY_X`(오타는 0=끔) · `effective_cap` 과 인라인 블록이 같은 규칙. | 미리보기와 실주문, 진입과 청산의 상한이 갈림 | O:444-493 · 816-843 `effective_cap` · 1441-1517 | 09-13~10-25 각 주석 | smoke::test_risk_model_off_eth_uses_simple_rule · test_model_only_cap_drops_ledger_and_equity · test_entry_and_exit_use_the_same_cap |

## 3. 지정가 집행 — 26개

| # | 규칙 | 어기면 | 위치 | 근거·날짜 | 시험 |
|---|---|---|---|---|---|
| E1 | 진입·청산·익절 지정가는 전부 **GTX**. (예외: 자동 물타기 dbad 는 GTC — §9 M9.) | 테이커 체결 | X:503 `fill_maker` · 597 `run_entry_wait` · 681 `run_entry_limit` · 296 `place_bracket` · 344 `place_tp_ladder` | — | Pself |
| E2 | 서명 요청은 예외를 던지지 않고 `{"__error__": "<HTTP상태> [<바이낸스 code>] msg"}` 를 돌려준다. **바이낸스 code 를 문자열에 남긴다**. 418/429 는 밴 기록만 하고 **주문·취소는 막지 않는다**. | «보냈는지 모르는» 채 흐름 끊김 / -5022 판정이 죽음(09-20) / 가짜 밴 기록으로 손절 청산 정지 | X:69-87 `signed` | 2026-09-20 실사고(리페그 0회) | Xself |
| E3 | -5022 판정은 **코드와 문구 둘 다** 본다(`"5022"` 또는 `"could not be executed as maker"`). | 급락장 첫 크로스에서 리페그 없이 포기 | X:468 `is_post_only_reject` | 2026-09-20 실사고 | Xself |
| E4 | 따라가기에서 -5022 = 실패가 아니라 «호가가 움직였다» → 새 호가로 다시(리페그 +1). 그 밖 오류는 **멈춤(rejected) — 몰래 테이커로 바꾸지 않는다**. | 오류에 시장가 전락 | X:512-523 `fill_maker` | — | inv #6 |
| E5 | 걸린 주문이 없을 때 호가 조회가 예외면 rejected 로 끝낸다(예외로 새지 않게). | phase 가 working 에 묶여 접수·손절 감시 정지 | X:516-519 · 604-608 | — | 없음 |
| E6 | 따라가기 접수가 네트워크·시간초과(오류 문자열이 숫자로 시작하지 않음)면 **나갔는지 모른다** → 표식(clientOrderId)으로 한 번 조회해 있으면 그 주문으로 이어 간다. | 살아 있는 GTX 가 손절 없이 체결 | X:506-511 `fill_maker` | 주석 «거부로 치면 살아 있는 GTX 가 손절 없이 체결» | 없음 (§어긋남 D2) |
| E7 | 지정가 대기 접수 오류가 네트워크·**5xx·-1007·-1006** 이면 상태 불명 → 표식으로 **최대 3번(1초 간격)** 조회, 끝내 모르면 **표식으로 DELETE 한 뒤** error 로 멈춘다. | 늦게 도착한 주문이 고아로 체결 | X:683-696 `run_entry_limit` | 검증 #1 · D3 | limit::test_unknown_submit_outcome_deletes_by_client_id_and_stops · test_5xx_unknown_status_is_treated_as_possibly_live_order |
| E8 | **재호가 전 취소 확인**: DELETE 응답이 아니라 **재조회 상태가 종료(FILLED·CANCELED·EXPIRED·REJECTED)** 인지 본다, 최대 3번(1초 간격). 확인 못 하면 **다시 걸지 않고** phase=error «거래소에서 직접 확인». | 2.0 ETH 포지션에 82 ETH 매도 지정가 41개 동시 생존 | X:419-436 `ensure_closed` · 550-557 · 626-631 · 740-745 | 2026-09-13 모의 거래소 검증 | inv #3 · #5 · #7 · cancel::test_c_unconfirmed_cancel_stops_with_error |
| E9 | 취소를 확인 못 해 멈출 때도 **마지막에 본 체결량(하한)** 을 돌려준다. | 부분 체결분에 손절이 안 걸림 | X:554 `fill_maker` | 주석 | 없음 |
| E10 | 체결량은 거래소 응답 `executedQty` 에서 읽는다(시장가도 «보통 전량»을 적지 않는다, 없을 때만 남은 수량). | 상태의 체결량이 틀림 | X:410-411 `run_entry` · 880 `run_exit` | 주석 | inv #9 |
| E11 | `PARTIALLY_FILLED`·`NEW` 는 종료 상태가 아니다. | 잔량을 안 넘김 | X:41 `TERMINAL` · 927 | — | Xself |
| E12 | 주문 상태 조회 실패는 **재시도**(주문은 살아 있다고 본다). | 살아 있는 주문을 버리고 재호가 | X:532-533 · 616-617 · 721-722 | — | 없음 |
| E13 | 앞지름 판정 기준은 **주문 측면(BUY/SELL)**: 매수는 최우선 매수호가가 내 위로, 매도는 매도호가가 내 아래로. | 진입 롱·청산 숏을 둘로 다룸 | X:462 `drifted` | 2026-09-15 | Xself |
| E14 | 호가는 **주문이 나가는 심볼**에서 읽는다(하드코딩 금지). | USDT 호가로 USDC 매수 → -5022 5/5 · 첫 주문 3초 만에 취소 | X:442-459 `maker_price` | 2026-09-19·20 | Xself |
| E15 | 리페그 상한 40회, `REPEG_MAX × POLL_SEC ≥ FALLBACK_SEC`. | 폭주 / 남은 시간 못 씀 | X:439 · 994 | — | Xself |
| E16 | 체결 알림으로 일찍 깨도 **상태만** 보고, 앞지름(리페그) 판단은 **3초마다**. | 알림 빈도로 리페그 상한이 일찍 바닥 → 잔량 시장가 | X:539-541 `fill_maker` | 2026-11-09 검토 | 없음 |
| E17 | 진입과 청산은 **같은 따라가기 루프 한 벌**(`fill_maker`). | 한쪽만 고쳐짐 | X:482-490 · 941-942 | 2026-09-15 | Xself |
| E18 | 진입 «대기 취소»(`action="cancel"`)는 다음 조회(≤3초)에 걸린 주문을 **취소 확인 뒤** 멈춘다. 잔량은 시장가로 안 넘기고, 든 몫은 그대로(손절은 호출부가 건다). | 취소했는데 시장가 | X:496-499 · 396-398 `run_entry` | 2026-10-11 | cancel::test_a_cancel_while_working · test_b_partial_fill_kept · test_negative_control_no_cancel_goes_taker · cancel::test_server_cancel_partial_then_bracket_path_once |
| E19 | 표식: `dbpg`+측면글자(따라가기) · `dbwt`+측면(기다리기·지정가 대기) · `dbtp`(익절) · `dbsl`(비상 스탑, Algo) · `dbad`(자동 물타기). 취소는 **표식으로만** 고르고 측면마다 다르다. 스탑 취소는 **그 심볼 것만**. | 롱 정리가 숏 주문을 지움 / 사용자가 앱에서 건 주문을 지움 / 다른 코인 스탑 삭제 | X:175-180 · 501 · 571-572 · 183-212 `cancel_bracket` | 09-24·09-25·10-20 | disarm::test_only_sl_skips_regular_orders · test_only_tp_cancels_only_take_profit · radd::test_cancel_bracket_deletes_only_ours · Xself |
| E20 | 기다리기(`run_entry_wait`): GTX 를 **한 번** 걸고 앞질려도 다시 걸지 않는다(처음 -5022 만 새 호가로 재시도). 첫 체결을 보면 한 번 `on_fill`(손절을 바로 건다). 거래소에서 사람이 취소하면 «취소»(시장가로 안 채움). | 60분 무방비 / 앱 취소를 시장가로 덮음 | X:576-650 | 2026-10-10 | wait::test_no_repeg_then_market_at_deadline · test_post_only_reject_retries_with_new_price · test_exchange_side_cancel_is_cancel · test_on_fill_once |
| E21 | 지정가 대기(`run_entry_limit`): 사용자 가격에 GTX, **따라가지도 시장가로 넘기지도 않는다**. -5022 는 «수정·재조정 직후»에만 연속 3번까지 새 목표가로 재시도, 아니면 거부로 끝냄. | 사용자 가격이 아닌 값에 체결 | X:657-713 | 2026-10-31 · 검증 D6 | limit::test_fixed_price_fill_no_repeg_no_market · test_post_only_reject_does_not_repeg · test_replace_rejected_after_amend_retries_instead_of_dying |
| E22 | 가격 수정(`amend`)은 **체결 0 일 때만**. 한 주라도 체결되면 수정은 버리고 남은 수량을 옛 가격에. 수정 수량 = 명목 유지 재계산값. | 한 포지션이 두 가격에 체결 · 선 기준 어긋남 | X:754-759 | 검증 D4 | limit::test_amend_cancels_then_replaces_with_new_price_and_qty · test_amend_after_partial_fill_keeps_old_price_and_rests_remainder |
| E23 | 자동 재조정: 20초마다, 목표가가 1bp 이상 벗어나면 같은 «수정» 경로, 최대 50회, **체결이 시작되면 멈춤**. | 큐 자리 상실 · 부분 체결 수량 어긋남 | X:654 · 730-738 | — | limit::test_auto_reprice_follows_basis_drift_then_stops_after_fill_starts · test_no_reprice_once_partially_filled |
| E24 | 취소 확인 중에 체결된 몫도 `on_fill`(손절)을 건다. | 그 몫 무방비 | X:751-753 | — | limit::test_on_fill_once_on_partial (이름 기준) |
| E25 | 감시 진입(`run_entry_stop`): 닿기 전엔 거래소 주문 없음. 1초마다 «since 뒤 1초 체결가가 선에 닿았나». 선을 옮기면 since 를 지금으로. 닿으면 최우선 호가 따라가기(`maker_only`) — **마감 뒤 잔량은 버림(시장가 없음)**. 4시간 넘으면 만료 취소. | 옛 선에 닿았던 가격으로 진입 / 시장가 전락 | X:796-825 · 399-401 · O:1008-1012 `stop_touched` | 2026-11-09 사용자 «지정가 체결은 절대 시장가로 바꾸지 않는다» | stop::test_touch_starts_maker_only_chase_at_best_bid · test_amend_resets_touch_window_and_cancel_places_nothing · test_chase_deadline_drops_remainder_never_market |
| E26 | 지정가 대기의 «PEG» 동작: 걸린 지정가를 취소 확인 → `action` 을 비우고(따라가는 중 ✕ 취소가 먹게) 최우선 호가 따라가기 120초 → 잔량 버림(시장가 없음). | 따라가는 중 취소 불가 / 시장가 전락 | X:773-789 | 2026-11-09 | stop::test_peg_chases_best_bid_and_drops_rest_never_market · test_peg_full_fill |

## 4. 대기 주문(기다리기·지정가 대기·감시 진입) — 12개

| # | 규칙 | 어기면 | 위치 | 근거·날짜 | 시험 |
|---|---|---|---|---|---|
| W1 | 대기는 진입 칸과 **따로**(코인별·칸별 `manual_wait_states`, 키 `<코인>` / `<코인>#n`). | 60분 동안 청산·손절 감시가 막힘 | O:658-677 | 2026-10-10 · 11-10 · 11-18 | wait::test_server_flow |
| W2 | 한 코인 대기 칸 최대 10(`WAIT_SLOTS_MAX`). | 조회 폭주(칸마다 3초 조회) | O:665 | ponytail 주석 | many::test_many_same_side_and_dup_refused |
| W3 | 최대 대기: 기다리기 15·60분 → 마감 뒤 잔량 시장가(§9 M3) · 지정가·감시 4시간 → **만료 취소(시장가 없음)**. | 다운 중 체결 노출 창 확대 | X:573 · 653 `LIMIT_MAX_SEC` · 791 | 검증 D2 | limit::test_expiry_cancels_without_market |
| W4 | 지정가 대기는 접수·수정 때 **계획을 파일**(`manual_limit_wait_<심볼>[_n].json`)에 남기고, **정상 종료(phase≠error)에서만** 지운다. 서버 종료로 취소되면 파일이 남는다. | 재시작 사이 체결분에 익절·손절이 없음 | O:888-900 `limit_save` · 1005-1006 · 2803 | 검증 D2 · #1 | limsrv::test_error_phase_keeps_recovery_file_and_final_arm_respects_modified_lines |
| W5 | 재시작 복구: 6시간 넘은 파일은 복구 안 하고 지움(알림) · 계좌 못 읽으면 파일 유지 · 포지션 있고 무장 없으면 **저장한 선으로 무장**하고 실제로 걸렸는지 확인 · 고아 정리에 실패했으면 파일 유지(다음 기동에 다시). | 며칠 묵은 계획으로 지금 포지션에 선 / 살아 있는 GTX 나중에 체결 | O:902-934 `limit_recover` | 검증 #1·#3 | limsrv::test_restart_arms_position_filled_while_server_was_down · test_restart_without_position_or_with_existing_bracket_places_nothing · test_restart_keeps_file_when_orphan_cancel_failed · test_restart_ignores_stale_file · test_restart_with_unreadable_account_keeps_file_and_says_so |
| W6 | 지정가 수정은 mode=limit·working·동작 없음·주문 있음(감시는 닿기 전) 일 때만. **일부 체결 뒤 거절**, 앞 수정이 반영 중이면 거절, 계획 조립 뒤 상태를 **다시** 확인. 수정한 선은 파일에도 남김. | 체결분과 다른 선 / 수정 겹침 | O:2767-2808 `api_manual_limit_amend` | 2026-10-31 | limsrv::test_submit_amend_cancel_flow · stop::test_amend_refused_once_triggered |
| W7 | 지정가 대기의 «지금 시장가»·«PEG»: 주문이 걸려 있고(감시는 닿기 전) 따라가는 중이 아닐 때만. 체결될 쪽 호가(USDT) 기준으로 익절·손절선을 **다시 검사**해 안 맞으면 409(추가 진입은 검사 없음). 시세 못 읽으면 거절. | 이미 선을 넘은 가격에 시장가 진입 | O:2845-2867 `api_manual_wait_action` | 2026-11-04·11-09 | limsrv::test_market_now_on_limit_wait_passes_action_to_loop · test_market_now_refused_when_lines_no_longer_fit_current_quote |
| W8 | «PEG» 는 지정가 대기에서만(기다리기에서는 cancel 아닌 동작이 시장가로 가므로). | 기다리기에 PEG = 시장가 | O:2843 | 2026-11-09 | stop::test_peg_refused_outside_limit_mode |
| W9 | 포지션이 손절·청산으로 닫히면 **같은 심볼·같은 쪽 대기 진입도 취소**(다른 코인 대기는 둠). | 나중에 체결돼 손절 없이 다시 열림 | O:1200-1204 `bracket_tick` | 2026-10-10 · 10-20 | 없음 |
| W10 | 대기 첫 체결 순간: 추가 진입이면 `keep_rearm`, 아니면 **지금 저장된 선**(`plan_from_current`)으로 무장. 예외는 결과 칸에 남기고 대기는 계속. | 60분 무방비 / 대기 중 바뀐 선 되돌림 | O:977-990 `entry_then_bracket` | 2026-10-20 · 11-13 | lines::test_limit_add_on_sl_off_rearms_tp_on_full_size · test_wait_plain_add_keeps_user_changes_made_while_waiting |
| W11 | 대기 종료 뒤 무장: 사용자가 대기 중 옮긴 선·손절 해제·청산 진행(fired)이 있으면 **옛 계획으로 덮지 않는다**. | 지운 익절 부활·옮긴 선 원위치·꺼진 손절이 다시 켜짐 | O:993-1004 `entry_then_bracket` · 936-948 `plan_from_current` | 검증 #2 · 2026-11-13 | lines(«2026-11-13 서브 에이전트 검증 결함 재현» 묶음) · limsrv::test_error_phase_keeps_recovery_file_and_final_arm_respects_modified_lines |
| W12 | 지정가 대기 상태 칸은 `order_id`·`client_order_id`·`pegged`·`triggered`·`amend`·`action` 을 집행 루프와 API 가 **같은 dict** 로 주고받는다(루프는 ≤3초마다 읽음). | 명령이 안 먹거나 두 번 먹음 | X:611·716·738·776-777·823 · O:2779-2806·2868 | — | stop::test_peg_action_on_limit_then_locked (이름 기준) |

## 5. 브래킷(손절·익절) — 33개

| # | 규칙 | 어기면 | 위치 | 근거·날짜 | 시험 |
|---|---|---|---|---|---|
| B1 | 무장 상태는 **파일**(`data/live/manual_bracket_state.json`)에 둔다, 쓰기는 임시 파일 → 교체. | 재시작하면 손절 실패 흔적이 사라짐 | O:870-886 `bracket_load` · `bracket_save` | 옛 3% 손절 사고(메모리에만 있었다) | 없음(간접) |
| B2 | 무장 키 = `심볼:측면`. 옛 키(LONG/SHORT)는 읽을 때 옮긴다. | ETH 롱과 SOL 롱이 서로 덮음 | P:506-519 `bracket_key` · `bracket_rekey` | 2026-09-26 | prot::test_state_key_is_symbol_and_side_and_legacy_keys_migrate |
| B3 | 체결이 있으면(부분 포함, **모든 종료 경로**) 무장한다. 결과 칸 `state["bracket"]` 은 예외가 나도 반드시 채운다. | 화면 폴링이 영원히 기다림 / 부분 체결분 무방비 | O:972-1004 `entry_then_bracket` · 1082-1106 `arm_bracket` | — | cancel::test_server_cancel_partial_then_bracket_path_once · test_server_cancel_zero_fill_no_bracket |
| B4 | `place_bracket`: 포지션을 **새로 읽어** 그 측면 전량으로 TP(GTX)·비상 스탑(Algo STOP_MARKET closePosition, 마크가 기준)을 건다. 옛 우리 주문(TP·물타기·스탑)을 먼저 지운다. 실패해도 진입을 되돌리지 않는다. `strict` 면 옛 주문을 다 못 지웠을 때 새로 안 건다. | 같은 쪽 전량 익절 두 벌 | X:267-304 | 2026-10-29 | 없음(호출부 시험은 가짜로 대체) |
| B5 | 조건부 주문(스탑)은 **`/fapi/v1/algoOrder` 로만**(`/fapi/v1/order` 의 STOP_MARKET 은 -4120 거부). | 손절이 한 번도 안 걸림(4일 이력 0건) | X:170-173 · 237-246 `_post_stop` | 2026-09-25 | Xself |
| B6 | 비상 스탑이 **안 걸렸으면 자동 물타기 지정가를 걸지 않는다**. | 무방비 포지션을 키움 | O:1108-1132 `arm_rule_add` | 2026-10-14 | radd::test_no_add_when_stop_failed · radd::test_fill_places_one_add_after_stop_and_arms_it |
| B7 | 익절 지정가가 못 걸렸으면(`tp_resting=False`) 서버가 1초 체결가 닿음으로 닫는다. | 익절 없음 | O:1103 · P:443-460 `line_hit` | — | lines::test_tp_touch_when_tp_order_not_resting |
| B8 | 감시 주기 2초, 체결 알림이 오면 일찍 깸. 한 틱의 예외는 삼키고 다음 틱(감시는 안 죽음). 취소(CancelledError)는 다시 던짐. | 감시 루프 사망 | O:1321-1329 `watch_brackets` | 2026-11-09 | 없음 |
| B9 | «포지션 사라짐(clear)» 은 **무장보다 5초 넘게 새로운** 계좌 조회 + 조회 성공 + 그 심볼·측면 수량 0 일 때만. | 진입 직후 캐시에 포지션이 안 보여 **살아 있는 포지션의 비상 스탑 삭제** | P:476-488 `bracket_action` | 주석 | Pself |
| B10 | clear 에서 우리 주문 정리가 실패하면 무장을 남기고 30초 뒤 다시. | 남은 물타기 지정가가 손절 없이 다시 열림 | O:1205-1215 | 주석 | 없음 |
| B11 | 규칙 무장(−5% 스탑)의 서버 손절 = **무장(또는 다시 켜기·수정) 뒤 마감된 USDT 5분봉 종가**가 레벨을 넘었나(꼬리 무시). 레벨 없으면 주문 심볼 가격 sl_price 로. | 꼬리 한 번에 청산 / 다시 켜기 직후 옛 종가로 바로 청산 | P:463-473 `bracket_sl_hit` · O:1183-1192 | 2026-09-27 사용자 · 10-29 검토 F1 | prot::test_sl_fires_on_closed_5m_bar_close_not_wick · lines::test_bar_close_backstop_respects_watch_from |
| B12 | 선 무장: **무장(watch_from) 뒤 1초 마지막 체결가 범위**(USDT 최저·최고)가 선에 닿음 + 체결 흐름이 끊겨도 5분봉 종가 이탈이면 닫음. 같은 틱에 둘 다면 **손절 먼저**. 한 번 쏘면(fired) 닫힐 때까지 계속 쏜다. | 틱 사이 꼬리 놓침 / 익절을 지운 뒤 닿음 창이 지나 조용히 멈춤 | P:489-500 · 424-434 `sl_legs_hit` · O:1193-1197 | 2026-10-28 | lines::test_touch_clears_tp_then_peg_exits · test_pre_arm_prices_and_no_touch_do_nothing · test_short_side_touch · test_prices_before_rearm_are_ignored |
| B13 | 손절 방향 `sl_up`(올라 닿으면 쏨): 기본 롱 False·숏 True. 사용자가 놓은 순간 화면이 보낸 방향(`slup`)이 있으면 그것. 익절만 옮기면 손절 방향은 그대로. | 롱 손절이 «올라 닿으면»으로 뒤집혀 무력화 | P:437-440 `sl_up_of` · O:2547-2562 `check_lines` · 2698-2699 | 2026-11-10 · 11-12 검토 | lines::test_sl_direction_kept_when_only_tp_moves · test_sl_direction_from_where_user_dropped_it |
| B14 | 선 무장 발동: **첫 await 전에** `bracket_exiting` 에 넣고 → 걸린 익절 지정가를 **먼저** 지운다(그 뒤 청산 수량을 읽음) → 못 지우면 10초 쉬고 다시 → 지웠으면 `fired` 를 **곧바로 파일에** 쓴다. | 수정 요청이 틈에 익절을 다시 검 / 익절+청산이 겹쳐 과청산 / 계좌 조회 2초 반복 | O:1255-1279 `bracket_tick` | 2026-10-28 검증 · 10-29 검토 B1 | lines::test_tp_cancel_failure_waits_and_does_not_exit |
| B15 | 발동 직전 다시 읽은 무장의 `armed_at`·`mod.at` 이 다르면 쏘지 않는다(다음 틱이 새 선으로). | 옛 선으로 청산 | O:1261-1263 | 2026-10-29 검토 S1 | lines::test_tick_does_not_revert_modified_lines (이름 기준) |
| B16 | 청산 계획 조립(await) **뒤** 손절 해제가 들어왔으면 쏘지 않는다(선 무장이 손절 없이 판단한 익절 fire 는 예외). | 해제했는데 청산 | O:1280-1283 | 주석 | 없음 |
| B17 | 발동 청산 계획이 `no_position` 이면 정리하고 내림 · 조회 실패·blocked 면 **«닫혔다»가 아니다** — 스탑을 두고 30초 뒤 다시(이유 `last_error`). | 조회 실패로 스탑 삭제 | O:1287-1297 | 주석 · 검토 P1 | 없음 |
| B18 | 발동 청산은 청산 버튼과 **같은 `run_exit`**(따라가기 → 마감 뒤 잔량 시장가 · §9 M7). **다 닫혔을 때만** 비상 스탑·남은 우리 주문을 지운다. 실패면 무장 유지 30초 뒤. | 청산 실패 뒤 무방비 | O:1298-1311 · 1166-1168 | 주석 | lines::test_exit_failure_keeps_fired_latch_and_retries_without_touch |
| B19 | 감시 결과는 다시 읽은 저장본에 **내가 본 armed_at 인 측면에만** 합친다. 손절 해제·익절 지움·선 수정 칸(`OWNED_BY_SAVED`)은 저장본이 주인(이번 틱이 쐈으면 tp_resting 만 틱 값). | 틱 사이 새 무장(물타기)을 지움 · 해제·다시 켜기·선 수정을 되돌림 | P:522-545 `bracket_merge` · O:1316-1319 | 2026-10-15 · 10-29 | Pself · lines::test_tick_does_not_revert_modified_lines |
| B20 | 분할 손절(`sl_leg`): 닿은 조각 무게 비율만큼 청산, **쏜 조각 표시를 먼저 파일에**, 성공하면 쏜 조각을 빼고 남은 수량으로 익절 조각을 다시 건다. 실패면 30초 뒤(쏜 표시 유지). 남은 조각이 다 닿거나 익절이 닿으면 전량. | 2분 청산 동안 같은 조각을 다시 쏨 | O:1217-1252 · P:494-499 | 2026-11-11 | lines::test_one_leg_partial_exit · test_all_legs_full_exit · test_failed_leg_exit_keeps_fired · test_fired_is_saved_before_exit_runs |
| B21 | 손절 해제(disarm): 거래소 우리 스탑(dbsl)만 지우고(물타기·TP 그대로), **취소 실패면 아무것도 안 바꾼다**. 그 사이 다시 무장됐으면 409. 이후 감시는 5분봉 손절도 끈다. | 한 겹만 풀린 상태 | O:2263-2293 · P:501-502 | 2026-10-14 사고(스탑만 지웠는데 5분봉 감시가 청산) | disarm::test_disarm_cancels_only_stop_and_marks_state · test_cancel_failure_changes_nothing |
| B22 | 손절 다시 켜기(rearm): 선 무장 = 감시만 다시(손절선이 없으면 409 · 익절 청산 진행 중 409 · `watch_from`=지금, 쏜 표시 지움). 규칙 무장 = 남은 우리 스탑을 먼저 지우고 포지션 확인 뒤 스탑 다시 → **못 걸면 아무것도 안 바꾼다**. 마크가가 이미 넘었으면 거래소 거부(-2021)를 그대로 실패로. | 감시만 켜진 반쪽 / 스탑 두 개 | O:2295-2343 · X:249-264 `place_stop` | 2026-10-15 | disarm::test_rearm · test_rearm_failure_changes_nothing · test_places_only_stop_after_clearing_old · lines::test_rearm_lines_watches_from_now_without_exchange_stop · test_rearm_refuses_never_armed_sl |
| B23 | 손절 해제 상태의 선 무장은 익절(서버 감시 몫)은 계속 쏜다(`fired="tp"`), 손절로 쏜 청산을 해제로 멈춘 것(`fired=True`)은 hold. | 손절을 지우면 익절까지 꺼짐 | P:490-493 | 2026-11-11 | Pself · disarm::test_lines_sl_only_removal |
| B24 | 익절 끄기/켜기: 선 무장만. 끄기 = 우리 dbtp 만 지움, **일부만 지워졌으면 익절선을 서버 감시로**(`tp_watch_only`). 켜기 = 지금가(5초 안 1초 체결가)가 이미 익절선 너머면 409, strict 로 다시. 그 사이 무장이 바뀌면 방금 건 지정가를 지운다(고아 방지). | 남은 조각 정체 불명 / 켜는 순간 닫힘 | O:2479-2536 · 2386-2399 | 2026-11-11·11-12 검토 | disarm::test_lines_tp_only_removal |
| B25 | 선 수정(modify): 선 무장만 · 청산 중(fired·exiting·칸 없음) 409 · 계좌에서 포지션 확인. 익절 **레벨(USDT)** 이 그대로면 지정가를 안 건드림(주문가 비교 금지). 옮기면 strict 로 다시, 옛 주문 정리 실패면 새 선은 서버 감시 + 경고. 키·포지션 조회 실패면 아무것도 안 바꿈. 무장이 바뀌었으면 방금 건 지정가를 지움. `watch_from`=지금. 지운 쪽 선은 검사·갱신하지 않음. | basis 1틱 변화가 익절 재발주 / 같은 쪽 익절 두 벌 / 옮기기 전 지나간 가격으로 청산 | O:2661-2765 `api_manual_bracket_modify` | 2026-10-29 · 검토 F2·F3·S2 · 11-04 | lines::test_move_tp_replaces_resting_order_and_watches_from_now · test_move_sl_only_touches_no_exchange_order · test_tp_cancel_failure_hands_tp_to_server_watch · test_early_place_failure_changes_nothing · test_sl_only_with_basis_drift_keeps_resting_tp · test_tp_place_failure_falls_back_to_server_watch · test_rejects · test_modify_ignores_removed_leg_even_if_out_of_range |
| B26 | 선 추가(add): 규칙 무장이면 409(손절 둘 겹침). 그 선이 이미 있으면 분할 조각 하나 더(최대 4, ±0.05% 같은 가격 거절). 무장 없는 포지션은 새 선 무장(다른 쪽은 «걸린 적 없음» 으로 꺼짐 — 다시 켜기 대상 아님). 익절은 strict 로 걸고 실패(키·포지션)면 아무것도 안 바꿈. | 손절 없는 포지션에 «켜짐» 표시 | O:2564-2659 | 2026-11-11 | lines::test_sl_on_position_without_bracket_makes_new_lines_arm · test_tp_on_position_without_bracket_places_only_tp · test_readd_removed_sl_at_new_price · test_refusals · test_add_second_tp_makes_two_equal_legs · test_add_second_sl_leg |
| B27 | 분할 익절 수량: 가까운 쪽부터 격자 내림, 최소 수량·명목 미달 조각은 먼 조각에 몰아줌, 합 ≤ 포지션. | 조각 합이 포지션 초과 / 거래소 거부 | P:394-421 `split_tp_qty` | 2026-11-11 | Pself |
| B28 | 분할 익절 다시 걸기(`place_tp_ladder`): 옛 우리 익절을 **다 지운 뒤에만** 건다. 걸려 있던 조각이 목록에 없으면 그 주문을 **직접 조회해 FILLED 일 때만** 체결로 침(거래소가 지운 것일 수 있다). | 같은 쪽 익절 겹침 / 지워진 조각이 영영 빠짐 | X:307-349 · O:2401-2422 `ladder_apply` | 2026-11-12 검토 | disarm::test_splits_by_weight_and_drops_filled · test_exchange_cancelled_leg_is_placed_again · test_cancel_failure_places_nothing · lines::test_partial_cancel_hands_legs_to_server_watch |
| B29 | 조각이 다 빠지면 `tp_resting=False`(포지션이 남았으면 마지막 익절선을 서버가 감시). 사용자 스트림의 «우리 dbtp FILLED» 는 그 조각을 뺀다. | 익절 주문도 서버 감시도 없음 | O:2351-2364 `set_tp_legs` · 2424-2432 `tp_leg_filled` · 2985-2989 | 2026-11-12 검토 | lines::test_weights_and_filled_leg_dropped (이름 기준) |
| B30 | 부분 청산 뒤 **남은 수량으로 익절 다시 걸기**: 익절이 거래소에 걸려 있던 무장만(서버 감시·fired·tp_off·exiting 제외). 규칙 무장은 스탑도 같이(손절 해제가 아니면). 그 사이 새 무장이면 덮지 않음. 지운 뒤 못 걸었으면 서버 감시로. | 익절 지정가가 포지션보다 커져 거래소가 줄이거나 지움 | O:1038-1076 `partial_exit_rearm_tp` · 2201 · 1246 | 2026-11-04 사용자 | prx::test_partial_exit_replaces_tp_for_remaining_qty · test_full_exit_or_no_resting_tp_leaves_orders_alone |
| B31 | 추가 진입 체결 = 기존 무장 **그대로**, 익절 수량만 합친 전량으로(`keep_rearm`, 포지션마다 잠금 한 줄). 무장이 그 사이 지워졌으면(새 포지션) 계획에 복사한 선으로 새로 무장. 손절 조각 청산 중이면 건너뜀. | 추가 진입이 SLTP 를 덮음 / 동시 체결로 익절 두 벌 / 무방비 새 포지션 | O:950-970 | 2026-11-14 사용자 · 검토 11-18 #5 · 검증 11-14(high) | lines::test_bracket_cleared_while_wait_add_pending_rearms · test_same_side_add_keeps_ladder · test_fill_rearms_same_lines_on_whole_position |
| B32 | 청산 진행 중(`bracket_exiting`·fired·청산 칸 없음)엔 선 수정·익절 on/off·선 추가·조각 조작·추가 진입을 거절. | 청산 중 익절 재발주 → 과청산 | O:657 · 2449 · 2497 · 2586 · 2682 · 1551 | 검토 B1 | lines::test_refuse_while_leg_fired · test_add_refusals |
| B33 | 계좌 응답에 «손절 무장이 없는 열린 포지션» 목록(`unprotected`)을 싣는다. | 무방비 포지션을 조용히 둠(10-20 XRP) | O:803-814 `with_bracket` | 2026-10-20 | rule::test_unarmed_position_gets_stop_on_add_and_account_lists_it |

## 6. 청산 — 6개

| # | 규칙 | 어기면 | 위치 | 근거·날짜 | 시험 |
|---|---|---|---|---|---|
| C1 | 청산 미리보기·제출·전량 청산·감시 청산은 **계좌를 새로 읽는다**(fresh). | 미리보기와 다른 수량 / 과청산 | O:2130-2137 · 2184 · 2219 · 1228·1280 | 2026-09-14 사용자 «청산 누르면 강제 조회부터» | 없음(직접) |
| C2 | 부분 청산(<100%)은 끝난 뒤 익절 다시 걸기(B30), 전량은 `run_exit` 만. | 익절 지정가가 남은 포지션보다 큼 | O:1078-1080 `exit_then_rearm` · 2201 | 2026-11-04 | prx |
| C3 | 스위칭: 전량 청산이 **다 닫혔을 때만**(phase filled_* 이고 체결 ≥ 수량×0.999) 반대 진입. 칸을 **await 없이** 진입 상태로 바꾼다. 게이트를 다시 보고 같은 `assemble_entry_plan(fresh)` 로 조립, 막히면 주문 없이 사유. 반대 진입도 손절 무장. 서버가 한 작업으로(브라우저가 꺼져도 끝까지). | 일부만 닫힌 채 반대 진입 / 사이에 다른 주문 / 화면 꺼지면 반대 진입 미도착(10-10) | O:1134-1164 `exit_then_switch` | 2026-10-10 10:51 사고 | switch::test_a_full_exit_then_one_opposite_entry · test_b_partial_exit_no_entry · test_c_blocked_entry_records_reason_no_order · test_d_client_gone_server_completes · test_direct_mode_switch_entry_gets_stop_even_with_sltp_off |
| C4 | 스위칭은 위 칸(`manual_entry_state`)만 쓴다(반대 진입이 따라오므로). | 진입 칸 규칙 우회 | O:2175 | — | 없음 |
| C5 | 전량 청산(submit-all): 방금 읽은 포지션마다 코인·방향 하나씩, 같은 코인·방향의 다른 심볼은 already_working, 주문 코인 표에 없는 심볼은 건드리지 않고 알림. | 한 버튼이 모르는 심볼을 닫음 | O:2205-2261 | 2026-10-22 | xall::test_all_coins_at_once_and_no_double_submit · test_guards · test_market_exec |
| C6 | 청산은 «대기 취소»를 받지 않는다(남은 포지션을 그대로 두는 것이라). | 청산 중단 → 포지션 방치 | O:2873-2889 `api_manual_entry_cancel` | — | cancel::test_server_exit_not_cancellable |

## 7. 재시작·종료·오류 — 9개

| # | 규칙 | 어기면 | 위치 | 근거·날짜 | 시험 |
|---|---|---|---|---|---|
| R1 | 재시작 직후 한 번: 거래소에 남은 **dbwt·dbpg 표식 주문**(기다리기·지정가 대기·따라가기)을 지운다(그 주문을 지켜보던 작업은 죽었다). dbtp·dbsl·dbad 는 지우지 않는다. 목록을 못 읽으면 그 사실을 남긴다. 결과는 대기 칸에 `restart_cancelled`. | 고아 GTX 가 손절 없이 체결 | X:828-848 `cancel_wait_orphans` · O:2891-2916 `wait_cleanup` | 2026-10-16 · 10-10 | wait::test_restart_cleans_only_tagged · test_restart_hook_records_cleanup |
| R2 | 그 다음 저장된 지정가 대기 파일로 복구(W5). | 재시작 사이 체결분 무방비 | O:2905-2915 | 검증 D2 | limsrv::test_restart_* |
| R3 | 집행 코루틴은 전부 `settle` 로 감싼다: 예외면 phase=error «거래소에서 직접 확인». | phase 가 working 에 묶여 모든 접수 409 · 서버 손절 감시도 안 쏨 | O:1030-1036 `settle` | 2026-10-16 감사 | 없음 |
| R4 | **phase=error 로 사람에게 넘기는 경우**: 취소 확인 실패(E8) · 지정가 접수 상태 불명(E7) · API 키 없음 · 집행 예외(R3) · 스위칭 반대 진입 막힘·오류. error 인 지정가 대기는 복구 파일을 지우지 않는다. | 모르는 상태에서 또 주문 | X:358 · 553-556 · 583 · 629 · 667 · 694 · 743 · 863 · O:1035 · 1152-1153 · 1163 · 1005 | — | inv #3 · limit::test_unknown_submit_outcome_deletes_by_client_id_and_stops |
| R5 | 종료 훅: 감시 태스크 취소 후 기다림. 주문·대기·청산·사용자 스트림 태스크는 `refresh_tasks` 에 넣어 **세션을 닫기 전에 취소**(주석상 `stop_http_session` 이 한다 — 본문은 못 읽음). | 닫힌 세션으로 주문 | O:1331-1339 · 1855 · 1848 · 2199 · 3032 · S:4988-4999 | — | 없음 |
| R6 | 종료 대기 2초(`shutdown_timeout`) — 끝나지 않는 SSE 때문에. | SIGTERM 행 → SIGKILL 로 on_cleanup 못 돎 | S:5023 | 주석 | 없음 |
| R7 | 기다림(nap)은 `asyncio.wait` 로(`wait_for` 는 3.11 에서 깨움과 취소가 같은 틱이면 취소를 삼킨다 gh-86296). | 종료 때 루프가 안 멈춤 | X:54-62 `nap` | 검토 | us::test_cancel_not_swallowed_when_woken_same_tick |
| R8 | 지정가 대기 태스크가 종료로 취소되면 복구 파일이 남아야 한다(정상 종료에서만 지움). | 재시작 뒤 복구 불가 | O:1005-1006 | 주석 | limsrv::test_error_phase_keeps_recovery_file… |
| R9 | 진행 중 상태(진입 칸·청산 칸·대기 칸·`bracket_exiting`)는 **메모리**다 — 재시작하면 사라지고, 따라가기 청산 중이던 dbpg 지정가는 R1 이 지운다(추정: 청산 도중 재시작이면 포지션은 그대로 남고 무장 파일이 다음 감시를 맡는다). | 옮길 때 «재시작 = 진행 중 주문 정리»라는 전제를 놓침 | O:638-660 · 657 | 추정 | 없음 |

## 8. 계좌·체결 알림(사용자 스트림) — 11개

| # | 규칙 | 어기면 | 위치 | 근거·날짜 | 시험 |
|---|---|---|---|---|---|
| A1 | 계좌 조회는 **한 곳**(`produce_account`)·한 캐시(10초 + 묵은 값 허용 `STALE_GRACE_SECONDS`). 수동 주문 심볼들도 체결 조회에 넣는다. | 조회가 두 벌로 갈라짐 / USDC 최근 거래 안 보임 | O:679-729 · 366 | 2026-09-19 · 10-02 | 없음 |
| A2 | 조회 실패(`ok=false`)는 **그대로** 둔다. 마지막 정상값(30분까지)은 화면용 `last_good` 로만 붙인다 — 감시·주문 경로는 못 씀. | 실패를 정상으로 읽어 상한·정리 판단 | O:722-728 | 2026-11-02 · 09-26 | 없음 |
| A3 | `?fresh=1` = 캐시 우회, 1초 안에 이미 새로 받았으면 그 값(연타에도 거래소는 한 번). | IP 한도 | O:785-795 | 2026-10-06 | 없음 |
| A4 | 사용자 스트림 주소는 `wss://fstream.binance.com/private/ws/<listenKey>`(`/private` 빠지면 알림이 안 옴). | 연결은 되는데 알림 0 | O:612 | 2026-11-09 공식 문서 | 없음 |
| A5 | listenKey: POST 로 받고 30분마다 PUT 연장, **지우지 않는다**(같은 계좌 다른 연결까지 끊김). 밴 중이면 부르지 않는다. 418/429 는 본문을 JSON 으로 읽기 전에 밴 기록. | 다른 연결 끊김 / 밴 해제 지연 | O:2925 · 2929-2938 `listen_key` · 2967-2972 | — | 없음 |
| A6 | 웹소켓은 **전용 세션**(전체 시간 제한 없음, 공용 세션은 10초 제한). heartbeat 120초, 수신 60초 시간초과는 계속, TEXT 아닌 메시지·`listenKeyExpired` 면 다시 연결. | 10초마다 끊김 | O:2940-2999 | coin_flow 주석 | us::test_connect_refreshes_account_and_trade_wakes |
| A7 | 재연결 대기 5초에서 두 배씩 최대 300초, 5분 넘게 붙어 있었으면(정상 24시간 끊김) 5초로 되돌림. | 붙자마자 끊김 반복에 계좌 조회 폭주 | O:3005-3009 | 검토 | 없음 |
| A8 | 연결되자마자 계좌를 한 번 읽고 기다리는 루프를 깨운다(끊긴 동안의 체결은 다시 안 옴). | 끊긴 사이 체결 누락 | O:2962-2963 | — | us::test_connect_refreshes_account_and_trade_wakes |
| A9 | 이벤트 분류: `ORDER_TRADE_UPDATE` 의 TRADE·CALCULATED = 체결(계좌 다시 읽기) · CANCELED·EXPIRED·AMENDMENT = 주문(루프만 깨움) · `ACCOUNT_UPDATE` = 체결 · 우리가 낸 NEW 등 = 무시. | 리페그마다 계좌 조회 | O:617-628 `user_event_kind` | — | us::test_kinds |
| A10 | **알림은 앞당기기만**: 끊겨도 주문 3초·감시 2초·계좌 10초 조회는 그대로 돈다. | 알림이 끊기면 체결·손절을 놓침 | O:2923-2925 · X:43-44 | 2026-11-09 | us::test_wake_cuts_nap_short · test_nap_times_out_without_wake |
| A11 | 체결 알림 → 0.3초 묶고 → 계좌 다시 읽기(수익 조회 캐시도 비움) → seq+1 → 루프 깨움 → 최소 3초 간격. | 연속 체결에 조회 폭주(IP 한도) | O:3011-3028 `account_refresher` · 614 | 2026-11-09 | us::test_connect_refreshes_account_and_trade_wakes |

## 9. 시장가(MARKET·테이커)가 쓰이는 모든 분기 — 13개

판단 없이 나열만 한다. «닿는 화면 동작» 은 `dashboard/live/app.js` 를 grep 으로만 확인했다.

| # | 어디서 | 조건 | 왜(주석 근거) | 닿는 화면 동작 |
|---|---|---|---|---|
| M1 | X:403 `run_entry` | 따라가기 진입이 마감 `FALLBACK_SEC`(120초)까지 다 안 찼고, 취소(`action=cancel`)도 `maker_only` 도 아님 → **남은 수량 MARKET** | 모듈 설명(X:7-19): 미체결 시장가 전락보다 리페그가 싸지만 마감엔 잔량 테이커(사용자 선택 b) | 진입 버튼(체결 방식 #snapExec «따라가기»=0, 기본) · **선(SLTP) 진입**도 이 경로(app.js 12735 «선 주문 = 따라가기(run_entry)만») · 자동 규칙 진입 |
| M2 | X:379-380 `run_entry` | 계획 `type=="MARKET"` → 지정가 없이 **전량 MARKET**(M1 의 시장가 다리로 바로) | 2026-10-10 «스위칭 반대 진입 = 즉시 체결» | 스위칭(M11) 의 반대 진입 · 감시 진입 대기 중 «TAKER»(M5) |
| M3 | X:642 `run_entry_wait` | 기다리기 15·60분 마감 잔량 **MARKET**, 또는 `action=="market"` | X:569-570 «미체결 포기는 안 된다 — 마감엔 반드시 시장가»(10-10 원장 재생) | 체결 방식 #snapExec 15·60(`mode=wait`) · 대기 칸 «지금 시장가»(#snapWaitMarket 두 번 클릭, app.js 13112-13120) |
| M4 | X:766 `run_entry_limit` | 지정가 대기 중 `action=="market"`: 걸린 지정가를 **취소 확인한 뒤** 남은 수량 MARKET(거래소에서 사람이 취소했으면 없음) | 2026-11-04 사용자 «대기 라벨에 taker 버튼» | 차트 대기 라벨 «TAKER»(두 번 클릭, app.js 870·597) · #snapWaitMarket |
| M5 | X:818-819 `run_entry_stop` | 감시 진입이 닿기 전 `action=="market"` → `run_entry(type=MARKET)` = **전량 MARKET** | X:799 «action market = 지금 시장가(TAKER 두 번)» | 감시 진입 대기 라벨 «TAKER» |
| M6 | P:606 `build_exit_plan` → X:871-883 `run_exit` | 1분봉 실현변동성 ≥ **30 bp/√분**(`EXIT_TAKER_VOL_BPM`) → 지정가 없이 **전량 MARKET** | P:44-49 «비용이 아니라 확실성 — 급락 중 «닫혔나» 확인하느라 초를 세지 않게, 대가 ~1.7bp» | 모든 청산(청산 버튼·전량 청산·**손절 발동 청산**·분할 손절·스위칭 청산) |
| M7 | P:613-619 `build_exit_plan` → X:871 | 청산 명목이 거래소 **최소 명목 미만**(먼지 포지션) → **MARKET** | 2026-09-13 감사 «막으면 화면에서 영영 못 닫는다» | 모든 청산(감시 청산 포함) |
| M8 | O:2050-2054 `assemble_exit_plan(market=True)` → X:871 | `exec=market` → **전량 MARKET** | 2026-10-10 «즉시»(청산 칩) | 청산 체결 방식 #snapExitExec=1(«즉시») 의 청산 제출·미리보기 · 전량 청산 `exec=market`(O:2244, **화면 호출부 못 찾음**) |
| M9 | X:906-914 `run_exit` | 따라가기 청산 마감(**계획의 15~120초**, 변동성 연동)까지 다 안 닫힘 → **남은 수량 MARKET**(체결량을 응답에서 안 읽고 전량으로 기록 — D3) | X:852-856 «청산 미체결은 포지션이 그대로 남는다» | 모든 지정가 청산 — **손절 발동 청산 포함**: 서버 감시 손절(규칙 5분봉 종가·선 1초 닿음·분할 손절·익절 서버 감시)은 `assemble_exit_plan(market=False)` → 보통 **GTX 따라가기 → 마감 뒤 잔량 시장가**, 변동성 ≥30 이거나 먼지면 처음부터 시장가(M6·M7) |
| M10 | X:237-246 `_post_stop` (place_bracket·place_stop·partial_exit_rearm_tp 가 부름) | 거래소 Algo **STOP_MARKET** `closePosition=true`, 발동 기준 **마크가** → 발동 시 거래소가 **시장가로 전량** 닫음 | X:170-173 · P:241-248(«시장가 손절 기대 추가 비용 17bp · 99분위 230bp») · O:1536 «모든 진입 −5% 거래소 스탑» | 선(SLTP)이 아닌 **모든 진입**(따라가기·기다리기·자동 규칙·스위칭 반대 진입)의 −5% 손절 · «손절 다시 켜기» 버튼(규칙 무장) · 부분 청산 뒤 스탑 다시 걸기 |
| M11 | O:2183-2185 `api_manual_exit_submit` · O:1156 `exit_then_switch` | 스위칭 = 청산 다리 `market=True`(M8) + 반대 진입 `type=MARKET`(M2) — **두 다리 다 시장가** | 2026-10-10 사용자 «떨어지는 장에서 매도 지정가가 따라가기만 했다» | 떠 있는 버튼 «스위칭»(app.js 13282 `switch: {to: …}` · `&switch_to=`) — 화면 청산 칩과 무관하게 서버가 강제 |
| M12 | X:223-234 `place_add_order` | 자동 물타기 지정가는 **GTC LIMIT**: 이미 그 가격 아래(롱)면 그 자리에서 **테이커 체결** | X:224 ««−0.75% 에 산다»의 뜻에 맞다» | 자동 규칙(rule=c/h) 첫 진입 체결 직후 자동(비상 스탑이 걸렸을 때만) |
| M13 | O:2845-2868 `api_manual_wait_action` → M3·M4·M5 | `action=market` 접수(지정가·감시는 지금 호가 기준 선 재검사 뒤) | — | 위 M3·M4·M5 의 버튼들 |

**손절 발동 때 집행 방식 정리**(가장 정확히): 
- 선(SLTP) 무장 손절·익절 서버 감시·분할 손절 → 서버가 판정 → `run_exit` → **지정가 GTX 따라가기(3초 리페그) → 마감(15~120초, 변동성 연동) 뒤 잔량 시장가**. 변동성 ≥30bp/√분이거나 먼지면 처음부터 시장가. 거래소 스탑 없음.
- 선이 아닌 진입(−5% 규칙 손절) → **거래소 STOP_MARKET(마크가 발동, 전량 시장가)** 이 1차, 서버의 «마감 5분봉 종가» 판정이 2차로 위와 같은 `run_exit`(따라가기 → 잔량 시장가).

## 10. 속도와 맞닿는 시간 상수

| 이름 | 값 | 무엇을 기다리나 | 위치 |
|---|---|---|---|
| `FALLBACK_SEC` | 120초 | 따라가기 진입 마감 → 잔량 시장가 · PEG 따라가기 마감 · 감시 진입 따라가기 마감 | X:39 (P:27 에 같은 이름·값이 **한 벌 더** — D1) |
| `POLL_SEC` | 3.0초 | 주문 상태 조회·리페그 판단 주기(알림으로 상태 조회만 앞당김) | X:40 · `fill_maker` 539 |
| `REPEG_MAX` | 40회 | 리페그 상한 | X:439 |
| `ensure_closed` | 3회 × 1초(`min(POLL_SEC,1.0)`) | 취소 뒤 종료 상태 확인 | X:419-436 |
| 지정가 접수 상태 불명 조회 | 3회 × 1초 | 표식으로 주문 찾기 | X:686-691 `run_entry_limit` |
| -5022 재시도(지정가 대기) | 연속 3회 · 1초 | 수정·재조정 직후 호가 넘어감 | X:699-709 |
| `LIMIT_MAX_SEC` | 4시간 | 지정가·감시 진입 최대 대기 | X:653 |
| `REPRICE_SEC` · `REPRICE_MIN_REL` · `REPRICE_MAX` | 20초 · 1bp · 50회 | 지정가 자동 재조정 | X:654 |
| 감시 진입 닿음 확인 | 1초 | `run_entry_stop` 루프 | X:815 |
| `WAIT_CHOICES_MIN` | 15·60분 | 기다리기 마감 | X:573 |
| `EXIT_DEADLINE_K`·`_EXP`·`_MIN`·`_MAX` | 1500 · 1.472 · 15초 · 120초 | 청산 마감(변동성 연동) | P:38-41 |
| `EXIT_VOL_WINDOW` | 1분봉 30개(+2 받아 미완결 1개 버림) | 청산 변동성 | P:42 · O:851-855 |
| `EXIT_TAKER_VOL_BPM` | 30 bp/√분 | 이상이면 청산 즉시 시장가 | P:50 |
| 감시 주기 `watch_brackets` | 2초(알림으로 앞당김) | 손절·익절 판정 | O:1323 |
| clear 판단 여유 | 무장 + 5초 | 계좌 조회가 무장보다 새로워야 | P:487 |
| `retry_after` | 30초(정리 실패·조회 실패·청산 실패) · 10초(선 발동 전 익절 취소 실패) | 감시 재시도 | O:1211 · 1233 · 1248 · 1294 · 1310 · 1268 |
| `BINANCE_ACCOUNT_CACHE_SECONDS` | 10초(+ `STALE_GRACE_SECONDS` 묵은 값 — 값 못 읽음) | 계좌 캐시 | O:366 · 780 · 1172 |
| `?fresh=1` 재사용 | 1초 | 연타 시 계좌 조회 한 번 | O:792 |
| `account_last_good` | 30분 | 화면용 마지막 정상 계좌 | O:727 |
| 권장 배수 신선도 | 15분 | 자동 규칙 진입 | O:1562 |
| 1초 체결가 신선도 | 5초 | 익절 켜기·선 검사 기준가 | O:2516 · 2544 |
| `LIMIT_REC_MAX_AGE` | 6시간 | 지정가 대기 복구 파일 유효 | O:902 |
| `FEE_TTL_SEC` · `FEE_RETRY_SEC` | 3600초 · 300초 | 수수료율 캐시(성공·실패) | X:112 |
| `load_filters` | 영구(프로세스 수명) | 틱·스텝·최소 수량 | P:51 · 60-62 |
| `USER_STREAM_KEEPALIVE_S` | 1800초 | listenKey 연장 | O:613 |
| `ACCOUNT_PUSH_GAP_S` | 3.0초 | 체결 알림 뒤 계좌 재조회 최소 간격 | O:614 · 3028 |
| 알림 묶음 | 0.3초 | 한 체결의 알림 여럿을 조회 한 번으로 | O:3014 |
| 웹소켓 heartbeat · 수신 시간초과 | 120초 · 60초 | 사용자 스트림 | O:2959 · 2974 |
| 재연결 대기 | 5초 → ×2 → 최대 300초 · 5분 붙어 있으면 5초 | 사용자 스트림 | O:2941 · 3006-3009 |
| `INCOME_REFRESH_SECONDS` | 60초 | 수익 조회(체결 알림이 0 으로 되돌림) | O:140 · 3019 |
| 변동성·펀딩 조회 시간초과 | 5초 | 계획 조립 | O:852 · 1871 |
| `fetch_binance_json` 기본 시간초과 | 10초 | 호가·klines 공개 조회 | S:1474 |
| 5분봉(감시용) 캐시 | 60초(`EVIDENCE_SIGNAL_CACHE_SECONDS`, 주석 기준) | 규칙 손절 판정 입력 | S:1661-1679 `load_market_history` |
| `shutdown_timeout` | 2초 | 종료 | S:5023 |
| `HOLD_FIXED_MIN` | 240분 | 보유 지평 폴백(위험모델 꺼져 표시용) | O:1881 |

---

## 시험이 지켜 주지 않는 규칙(옮길 때 가장 위험한 것) — 25줄 · 규칙 33개

| # | 규칙 요약 | 위치 |
|---|---|---|
| G1 | POST 전용 | S:4963-4976 |
| G4 | 스위치 꺼지면 감시 틱도 정지 | O:1170 `bracket_tick` |
| G8 | 주문 잠금(`busy_or`, 잠겨 있으면 즉시 409) · 잠금 밖 엔드포인트 4개 | O:3035 · S:4963-4976 |
| G9 | 감시 틱은 주문 잠금 중 손절 발동을 미룸 | O:1218·1259·1285 |
| G13 | 대기 있으면 스위칭 409 | O:2178 |
| P12 | 지정가 크기 최소 주문 재검사 | O:1689 |
| P21 | basis 못 읽으면 1.0 | O:860 `basis_for` |
| P25 | 1초 체결가 원천 없으면 선 주문 400 | O:1660 |
| P28 | 실주문 청산 조립은 펀딩·수수료 생략 | O:1998 |
| E5 | 걸린 주문 없을 때 호가 예외 → rejected | X:516-519 |
| E6 | 따라가기 접수 «나갔는지 모름» → 표식 조회 | X:506-511 `fill_maker` |
| E9 | 취소 확인 실패 때 마지막 체결량 반환 | X:554 |
| E12 | 상태 조회 실패는 재시도 | X:532 |
| E16 | 알림으로 깨도 리페그 판단은 3초마다 | X:539-541 |
| W9 | 포지션 닫히면 같은 쪽 대기 취소 | O:1200-1204 |
| B1 | 무장 파일 원자적 쓰기 | O:883-886 |
| B4 | `place_bracket` 자체(전량 TP·스탑·strict) — 호출부 시험은 전부 가짜로 대체 | X:267-304 |
| B8 | 감시 주기·예외 삼킴 | O:1321-1329 |
| B10 | clear 정리 실패 → 30초 뒤 | O:1205-1215 |
| B16 | 조립 뒤 손절 해제 재검사 | O:1280-1283 |
| B17 | 조회 실패는 «닫힘» 아님 → 스탑 유지 30초 | O:1287-1297 |
| C1·C4 | 청산 fresh 계좌 · 스위칭은 위 칸만 | O:2130·2184·2175 |
| R3·R5·R6·R9 | `settle` · 종료 훅 태스크 취소 · 2초 종료 · 메모리 상태 | O:1030 · S:4988-5023 |
| A1-A3·A4·A5·A7 | 계좌 단일 조회·last_good·fresh 재사용 · /private 주소 · listenKey 수명 · 재연결 백오프 | O:612-3009 |

추가로 **inv 의 리페그 시나리오는 실제로 리페그를 검사하지 못하는 것으로 보인다**(D5) — E4·E8·E15 의 «리페그 중» 부분은 사실상 시험 없음으로 봐야 한다(추정, 실행 안 함).

## 이미 어긋나 보이는 곳(고치지 않음)

| # | 무엇 | 근거 |
|---|---|---|
| D1 | `FALLBACK_SEC = 120.0` 이 두 파일에 따로 있다(X:39, P:27). 진입 집행은 X 값을, 계획의 `fallback_after_sec` 표시는 P 값을 쓴다 — 한쪽만 바꾸면 화면과 실제가 갈린다. | X:39 · P:27 · P:221 · X:384 |
| D2 | **«나갔는지 모르는 주문» 판정이 세 경로에서 다르다.** `fill_maker`(따라가기 진입·청산·PEG·감시 진입)는 오류 문자열이 숫자로 시작하지 않을 때만 표식 조회 1회 — **5xx·-1007·-1006 은 확정 거부로 보고 rejected 로 멈추며**, 못 찾아도 표식 DELETE 를 하지 않는다. `run_entry_limit` 은 5xx·-1007·-1006 도 «불명»으로 3회 조회 + 표식 DELETE(검증 #1·D3). `run_entry_wait` 은 **아예 없다** — 네트워크 오류도 바로 rejected(살아 있을 수 있는 dbwt GTX 를 두고 끝남; 재시작 고아 정리만 그걸 지운다). 주석 X:507 «거부로 치면 살아 있는 GTX 가 손절 없이 체결된다» 와 맞지 않는 경로가 둘. | X:506 · 684 · 599-603 |
| D3 | `run_exit` 마감 뒤 잔량 시장가는 체결량을 응답에서 읽지 않고 `filled=total` 로 적는다. `run_entry`(X:410 ««보통»을 상태에 적으면 안 된다»)·`run_exit` 의 MARKET 계획 경로(X:880)와 다르다. 스위칭의 «다 닫혔나»(O:1141, 0.999 기준)가 이 값을 본다. | X:913 vs 411 · 880 |
| D4 | 진입 정책 충돌: X:569-570 «미체결 포기는 안 된다 — 마감엔 반드시 시장가»(기다리기·따라가기) vs X:399 · 775 · 800 «지정가 체결은 절대 시장가로 바꾸지 않는다»(감시 진입·PEG·지정가 대기). 지금은 경로마다 다르게 구현돼 있다(M1·M3 시장가 / E21·E25·E26 시장가 없음). 모듈 설명(X:7-8 «남은 수량만 테이커»)도 옛 판. | X:7-8 · 569-570 · 399 |
| D5 | inv 시험이 낡아 보인다(추정 — 실행 안 함): 가짜 `maker_price(session, side)` 가 2인자인데 코드는 `maker_price(session, side, symbol)` 3인자로 부른다 → TypeError 가 `fill_maker` 의 `except Exception: continue`(X:544) 에 삼켜져 **드리프트 시나리오(#2·#3·#4·#7·#10)에서 리페그가 실제로 일어나지 않는다**. #10 은 `"stop" in st` 를 요구하는데 `run_entry` 는 옛 3% 손절(X:171 삭제) 뒤 `stop` 칸을 안 쓴다 → #10 은 실패할 것으로 보인다. | inv:80-88 · 215 vs X:517·543 |
| D6 | P 자체점검이 낡아 보인다(추정): P:1018 은 익절 3000(지금가 +20%)을, P:1069-1073 은 손절 5.2% 거리를 ValueError 로 기대하지만 2026-11-14 «±15% 제한 자체를 제거»(P:384) 로 거리 검사가 없다. | P:1018 · 1069-1073 vs 384 |
| D7 | `strict` 적용이 갈린다: 선 수정·익절 켜기·선 추가는 `place_bracket(strict=True)`(옛 주문 정리 실패면 새로 안 검) 인데, **`partial_exit_rearm_tp`(O:1065)와 `arm_bracket`(O:1091)은 strict 없음** — X:288 주석이 막으려던 «같은 쪽 전량 익절 둘»이 이 두 경로에선 가능. | O:1065 · 1091 vs 2522 · 2635 · 2738 |
| D8 | 감시 틱이 손절 발동 청산(`run_exit`, 최대 ~120초 + 시장가)을 **루프 안에서 await** 한다(O:1240 · 1302). 그동안 다른 코인·측면 판정이 멈춘다(추정: 두 코인이 동시에 손절에 닿으면 두 번째는 첫 청산이 끝날 때까지 대기). O:1284 주석(«다른 코인 주문이 도는 중이면 이 코인 칸으로 청산»)은 칸만 나눈 것이고 감시 루프는 직렬이다. | O:1240 · 1302 · 1284 |
| D9 | 감시 틱은 주문 잠금 중 손절을 미룬다(G9). 잠금은 계획 조립 동안 잡히는데 설계 문서 §1 실측 진입 계획 조립 최대 27,392ms — 그만큼 손절 발동이 늦을 수 있다(추정). | O:1259 · 1285 · design §1 |
| D10 | 주석의 계좌 캐시 나이 «30초»가 여러 곳에 남아 있는데 값은 10초(2026-10-02 변경) + 묵은 값 허용. | O:364-366 · 1414 · 2013 · P:482 |
| D11 | `run_entry_limit` PEG 분기·`run_entry_stop` 닿음 뒤의 `maker_price` 호출은 try 밖이다(X:778 · 824). 예외는 `settle` 이 error 로 받지만, 메시지가 «집행 오류»라 «주문 없음» 인지 구별이 안 된다(추정: 두 곳 다 그 시점에 걸린 주문은 없다). | X:778 · 824 |
| D12 | 손절 해제·다시 켜기·대기 동작(시장가 포함)·진입 취소는 주문 잠금 밖이다(G8). 다른 버튼들(익절 on/off·선 수정)의 docstring 은 «주문 잠금 busy_or» 를 방어로 적었는데 같은 무장을 바꾸는 손절 해제·다시 켜기는 잠금 없이 `armed_at` 비교로만 지킨다. | S:4965-4969 · O:2267 · 2298 |

## 못 읽은 범위

- **`dashboard/server.py`**(5,029줄) — Read 로 읽은 곳: 1-120 · 1100-1830 · 1900-2030 · 2400-2440 · 2860-3060 · 3100-3130 · 3300-3330 · 3700-3740 · 4380-5029. **안 읽은 곳**: 121-1099 · 1831-1899 · 2031-2399 · 2441-2859 · 3061-3099 · 3131-3299 · 3331-3699 · 3741-4379. 그래서 다음을 **확인하지 못했다**: `start_http_session`·`stop_http_session` 본문(주문 태스크 취소 순서·세션 시간 제한), `STALE_GRACE_SECONDS`·`SWR_SLOW_LOG_SECONDS`·`EVIDENCE_SIGNAL_CACHE_SECONDS`·`NOCACHE` 값, `publish_dashboard_events` 의 `keep_trip_ledger` 호출 주기(주석상 300초), SSE 의 `acct_seq` 사용. (이 파일은 지시대로 Bash 인자로 쓰지 않았고 개인정보 줄은 인용하지 않았다.)
- `dashboard/coin_flow.py` — 1초 체결가 원천(`footprint_state["sec"]`, 칸 6 = 마지막 체결가)의 보관 길이·갱신 방식.
- `scripts/live_binance_account_20260910.py` — `fetch_account`(포지션·`open_orders` 의 `cid`·`reduce_only` 필드 출처, `generated_at`), `_sign`, `_clock_offset`(시계 보정 −1021).
- `scripts/binance_ban_guard.py` — 밴 판정·기록.
- `scripts/live_eth_trade_plan_20260913.py` — `leverage_setting`·`plan_now`(모델 레버리지 `target_leverage` 출처)·`EXCHANGE_MAX_LEVERAGE`.
- `scripts/live_eth_risk_sizing_policy_20260913.py` — 지금 꺼진 위험모델 상한.
- `dashboard/live/app.js`(13,904줄) — grep 으로 주문 호출부 몇 곳만. 버튼 2단 확인(«한 번 더»)의 전체 목록, 전량 청산(`submit-all`) 호출부는 찾지 못했다.
- 시험 파일 — inv 만 본문 전체를 읽었다. 나머지는 함수 이름·설명문만 보고 규칙에 대응시켰다(본문이 그 규칙을 실제로 검사하는지 미확인). 시험은 하나도 실행하지 않았다.
- `dashboard/orders.py`·X·P 는 처음부터 끝까지 읽었다(읽는 동안 다른 에이전트가 앞의 두 파일을 수정 중이었다 — `git status` 에 M).
