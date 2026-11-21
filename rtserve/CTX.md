# 상황·시장 맥락 조사 표 (2026-11-22)

원본 줄은 전부 Read 로 본 실제 파일(dashboard/server.py · dashboard/flow_read.py, HEAD d1a40656). graft 줄 번호는 낡아서 안 썼다.
«Rust» 칸: ✅ 이미 있음(어디) · 🟡 이번 조각(src/ctx.rs) · ❌ 없음(다음 조각).

## 1. /api/situation · SSE `event: situation`

| 칸 | 원본 | 원천 | Rust |
|---|---|---|---|
| 본문 모양 | situation_payload 2734~2738 | — | 🟡 `ctx::situation_payload` (글자 일치 300건) |
| /api/situation | 2740~2741 `json_response(…, NOCACHE)` = json.dumps 기본 | — | 🟡 본문까지(라우팅은 lib/api 쪽 · 손 안 댐) |
| SSE 상황 | api_stream 3149~3183: `computed_at` 이 바뀔 때만 · `ensure_ascii=False, separators=(',',':')` · 0.25초 틱 | — | 🟡 `ctx::sse_situation` (한 건 글자 일치) · «바뀔 때만» 비교는 hub.rs 몫 |
| 계산 전 | _coin_ctx 1777 `{"now": {"ok": False, "reason": "계산 전"}, "computed_at": 0.0}` | — | 🟡 `ctx::initial_now` |
| now | compute_situation 2478~2480 `sit.classify(inp)` + computed_at | 코어 Situation(rtcore situation.rs · 체결로 만든 5분봉 -- 원본은 REST 캔들: 문서화된 차이) | ✅ 코어 classify · 🟡 `ctx::patch_now`(missing 빼고 evidence.veto·obi·persist 채움) |
| now.evidence.veto | classify 가 그대로 통과(방향 계산에 안 씀) | REST klines 720봉 12h 추세 | ❌ 원천 없음(veto 결정: 코어에 두지 않는다) -- 받는 쪽이 넣는 인자 |
| now.evidence.obi·persist | classify 통과 칸 | 호가 래스터 요약(`_heatmap_read(sym,300,3,True)` summary · swr 5초) 2463~2468 | ❌ 래스터 요약 읽기 없음 -- 인자 |
| collect_micro_ref 예외 | 1912~2022 안: `now = {"ok": False, "reason": repr(exc)[:160]}` | — | ❌ (hub 쪽 실패 처리) |
| computed_at | 2481 | 시계 | ✅ 코어가 채움 |
| streams.fo | fo_state 1801 {connected, since, last_event_ms, events, errors, last_error} · 갱신 1843~1869 | @forceOrder WS 상태 | ✅ 코어 Micro.streams.fo(2026-11-22 더함 · feed.rs·state.rs) → `ctx::streams_payload` · last_error 글자만 다름(대조에서 뺌) |
| streams.mp | mp_state 1774 {connected, since, last_ms, events, errors, last_error} + next_funding_ms(1894) | @markPrice@1s WS 상태 | ✅ 코어 Micro.streams.mp → `ctx::streams_payload`(next_funding_ms 키는 첫 마크 뒤) |
| read | 2491 `fr.read(evidence or {}, ctx)` · ETH 만(2489~2490) · 예외 2516~2517 `{"lines": [], "summary": "", "error": repr(exc)[:160]}` | ctx(§2) + evidence | 🟡 `ctx::read` = flow_read.read 250~427 · fuse 430~460 · outcome3 463~507 · gauges 197~247 전부 (글자 일치 4,000건) |
| vol_mult | 2451~2452 `fr.vol_size_mult(완결 candles_day, asset)` | candles_day = REST klines 300개 swr 60초(load_5m_day 2418~2438) | 🟡 `ctx::vol_size_mult` (일치 400건) · ❌ 5분봉 하루 원천(REST -- 코어 5분봉으로 대체할지 결정 필요) |
| vol_mult_by | 2738 코인마다 situation_state.vol_mult | 코인별 위와 같음 | 🟡 인자(코인별 값 목록) |

부수 효과: FUSED_LOG 장부 쓰기 2492~2515(봉마다 한 줄) = ✅ `ctx::FusedLog`(줄 글자 일치 600건 · _last_logged_bar 2408~2416 꼬리 14가지) -- `RTSERVE_READONLY` 가 "0" 이 아니면 안 쓴다(병행 중 기본). card30 예측 캐시.

## 2. 관계 읽기 입력 x = `_flow_read_ctx` (2288~2406) -- ✅ `ctx_x::flow_read_ctx`(재료를 인자로 · 202걸음 글자 일치) · 남은 것 = 재료 읽기

| x 칸 | 원본 줄 | 원천 | Rust 쪽 재료 |
|---|---|---|---|
| net60 · z60 | 2295~2315 | 풋프린트 봉 셀 v[0..5](바이낸스 · 크기 세 갈래) → fr_bar_net · 12봉 합 · 과거 ≥72 의 std | 코어 풋프린트 봉(✅ 메시지로 옴) -- 합·convolve 차례 옮길 것 |
| z60h · z30h | 2318~2324 | micro_state.flow_scales = micro_ref.flow_hour_scales(테이프 hot SQLite 14일 · 서버 메모리 계산 · collect_micro_ref 1942~1946 · 기준선 주기마다 · 실패면 있던 값) | ✅ `ctx_x::tape_flow_rows`+`flow_hour_scales_from_bars`(일치) · ❌ 주기·실패 유지는 hub 몫 |
| cvd30_z | 2325~2328 | 같은 격자 a[:,0] 6봉 합 | 코어 풋프린트 봉 |
| venues · net60_all · z60h_all · net30_all · z30h_all · cvd30_all_z · taker_all | 2332~2373 | OKX 봉 칸 · Bybit 봉(bybit_bars 759~774 · swr 20초/120 2470~2472) · flow_scales_all(flow_hour_scales_combos 1947~1952) · flow_all_hold | ✅ `ctx_x::bybit_bars` · `flow_hour_scales_combos`(일치) · hold 는 FlowState |
| okx30 | 2374~2376 | okx_bars 마지막 6봉(first_bar 뒤) | ✅ Micro.okx_bars |
| move60 · oi60 | 2377~2382 | inp.bars(5분봉 close · oi_delta 10개 이상) | ✅ 코어 situation 입력 봉 |
| move60_p75 | 2385~2388 | candles_day 300개(REST) np.percentile 75 | ✅ 계산 · ❌ candles_day 원천(REST) |
| range30_pct | 2390~2396 | candles_day 293개 30분 창 고저폭 분위 | ✅ 계산 · ❌ candles_day 원천(REST) |
| dir_p · reach_p | 2397~2398 `_card30_pred`(2266~2280 · HGB · ETH 모델 모든 코인) | rtmodel::card30 | ✅ rtmodel::card30(확인 필요: 같은 입력 봉 = 테이커 있는 완결 candles_day) |
| imb40 · imb40_pct | 2399~2403 | micro payload imb40 + imb40_ring(6시간 초별 · ≥600) | ❌ 깊은 호가(±0.4%) 불균형 -- 래스터/호가 |
| liq60s · agree · trigger_live · sr · act · vol_pct | 2404~2405 | micro payload(collect_micro_ref 1912~2022: liq60 · agree · trigger.live · sr(청산맵 레벨) · act · vol60_pct) | ✅ Micro.liq60 · vol60 · qi(코어) / ❌ agree·trigger(OFI 분위) · sr(청산맵 레벨) · vol60_pct(시간대 분위) |

### x 재료 -- 이번 조각에서 확인한 것 (Read 로 본 줄)
- **칸 합 차례**: 원본은 봉 dict 값을 «칸이 생긴 차례»로 더한다(fr_bar_net 2300~2301 · onet 2336~2337 · okx30 2376 은 여섯 봉 칸을 한 줄로). `flow_read_ctx` 는 칸을 받은 차례로 더해 원본과 비트가 같다. 코어는 칸을 가격 키 차례(BTreeMap)로 들어서 실전 입력은 마지막 비트만 갈릴 수 있다(Micro.okx_bars 와 같은 한계). 또 rtserve 가 받는 풋프린트 메시지는 칸이 round(x,3) 이라 그대로는 못 쓴다 → **코어가 반올림 전 칸(또는 원본 규칙으로 센 봉 순합)을 실어야 한다**(다음 조각 · rtcore).
- **veto(12h)**: compute_situation 2453 `load_market_history` → ETH 는 증거 캐시 프레임(load_market_history_from_evidence_cache 1665~1716 · `trend_veto_rows(src)` 1613~1646 = `df.tail(720)` 마감봉 · SMA144·ATR ewm(adjust=False, 창 첫 행부터) · 밴드 히스테리시스) 의 봉마다 veto · 상황 입력 봉에 붙는다(_situation_inputs 2191~2206 · 방금 닫힌 봉이 캐시에 없으면 직전 veto 이음). SOL·XRP 는 klines limit 1000 → kline_candles(1648~) → 같은 trend_veto_rows(tail 720). → 마감봉 ≥ 721개면 같은 값(ewm 이 창 첫 행에서 시작하므로 창 = 마지막 720개 마감봉이면 된다).
- **candles_day**(vol_mult · move60_p75 · range30_pct · card30 입력): load_5m_day 2418~2438 = klines **limit 300** · 받을 때 `시작+300 ≤ 지금`인 행만(형성 봉 뺌 → 보통 마감 299개) · swr 60초 · 방금 닫힌 봉이 없으면 2초 뒤 다시. → 720 한 번 받기로 둘 다 내려면 «마지막 300행 중 마감봉»으로 잘라야 move60_p75 의 개수(299)가 같다(마감봉 300개를 쓰면 분위 위치가 바뀐다). taker(r[9])도 실어야 card30 입력(_card30_bars 2282~2286 `"taker" in c`)이 된다.
- **flow_scales**: 파일 아님 -- 서버가 hot 테이프(binance_tape.sqlite 등)를 SQL 로 5분봉 합 → numpy MAD(micro_ref 186~259). 옮김 완료(위).
- **imb40·agree·trigger·sweep·sr·vol60_pct**: collect_micro_ref 1912~2022 -- 래스터 `read_window(sym, now_ms, 12, 1)`(scripts/live_orderflow_raster_collector_20260914.py) → micro_ref.raster_flow · ofi 분위(ofi_hist 600) · agree_state · mctx.sweep_depth · sr_context(load_liquidation_map) · baseline_from_tape(1분 거래량 분위). 🔴원본은 이 틱마다 REST bookTicker(MICRO_BOOK_URL)도 부른다(코어 호가로 대체 가능한지 판단 필요). 다음 조각.

## 붙이는 쪽(hub.rs) 사용법 -- `ctx_x::situation_tick` (2026-11-22)
1. lib.rs 에 `pub mod ctx; pub mod ctx_x; pub mod ctx_micro;` (ctx.rs 는 `crate::py` 를 부른다).
2. 코인마다 상태 하나: `FlowState`(flow_all_hold) · `rtcore::situation::FlowNets`(받은 값) · `ctx_micro::MicroRef`(미시 링·기준선) · ETH 는 `ctx::FusedLog::new(<FUSED_LOG_PATH>)`.
3. 코어 구독에서: Situation → 마지막 값 보관 · Micro → `ctx_x::apply_flow(&mut nets, mc.flow)`(snapshot 의 micro.flow 는 full) · `mc.streams` 보관 · 코어 호가(book)·vol60·liq60 보관.
4. 1초마다(원본 micro 틱): `w = ctx_micro::read_window(래스터 뿌리, "ethusdt", now_ms, 12)` → `micro.tick(now, 코어 호가, &w, vol60, &청산맵 레벨, &liq60)` 가 미시 payload 를 만든다.
5. 기준선: `micro.baseline_due(now)` 이면 스레드에서 `baseline_from_tape`·`tape_flow_rows→flow_hour_scales_from_bars`·`flow_hour_scales_combos` 를 읽고 `micro.baseline_set(now, 결과)` (flow_scales 는 «빈 결과면 있던 값», combos 는 칸마다 같은 규칙 -- 원본 1942~1954).
6. `XIn { now, nets, bybit(bybit_bars · 20초 캐시), flow_scales, flow_scales_all, bars(코어 상황 입력 봉의 close·oi_delta), day5(candles_day), card30, micro: &micro.payload, imb40_ring }`.
7. `let (payload, sse, x) = ctx_x::situation_tick(core_sit.as_ref(), veto, obi, persist, &streams, &xin, &mut flow_state, &vol_mult_by, Some((&mut fused_log, &card_p, &feat_bar)))` -- ETH 만.
8. `/api/situation` = `payload` 를 json_response 와 같이(기본 구분자 · NOCACHE) · SSE 는 매 틱 `sse` 를 그대로(원본은 computed_at 이 바뀔 때만 = 매 틱 바뀐다).
9. 다른 코인은 `flow_read_ctx(&xin, &mut st)` 만(시장 맥락 재료) · read 는 ETH 만(2489~2490).
10. 🔴`RTSERVE_READONLY` 가 "0" 이 아니면 장부를 안 쓴다(병행 중) · REST 원천(candles_day·veto·card30)은 `RTSERVE_REST` 있을 때만(다음 조각).

## 코어 원값 합(Micro.flow · 2026-11-22)
- rtcore situation.rs `flow_nets_tick`: 원본 2295~2303(fr_bar_net: 닫힌 봉 처음 볼 때·최근 두 봉만 다시 · 사라진 봉 지움) · 2332~2339(onet: 닫힌 봉 12개 이상일 때 격자 안 first_bar «뒤» · 사라진 봉 지움) · 2374~2376(okx30: 여섯 봉 칸 한 줄 합) · 2368~2369(테이커 비율 재료 = 마지막 닫힌 봉 Σ매수·Σ매도 **그 회차 값** -- 굳은 봉이면 fr_bar_net 과 다르다 → `bn_last` 로 따로).
- 칸 합 차례 = 원본 dict 차례: footprint.rs `Bar.order`(체결로 생긴 칸 차례) · `in_order`. **알려진 차이**: 복원(테이프·스냅숏)으로 생긴 칸은 키 차례로 뒤에 붙인다(원본도 재기동 뒤 자기 복원 차례) → 그 봉은 마지막 비트가 갈릴 수 있다.
- 화면 풋프린트 메시지 모양은 그대로(Micro 에 칸 `flow`·`streams` 만 더함 · 비면 직렬화에서 빠짐). app.js `rtMeasure`(4751~4755)는 `ex_ms > 0` 인 메시지만 지연에 넣고 대조는 supply 초만 본다 → Micro(ex_ms 0)는 세지 않는다.

## 미시 payload(ctx_micro.rs) · bookTicker 결정안
- 옮김: read_window(agg 1) · raster_flow(f32 합 = numpy float32 짝 합) · agree_state·qi_side_hyst · pct_rank · act_label · sr_context(동률이면 이름 글자 차례 = 파이썬 튜플 min) · sweep_depth · baseline_from_tape · 한 틱(1956~2008) · 기준선 주기(1926~1941). 268틱 글자 일치.
- **알려진 차이**: baseline_from_tape 의 초 합계는 원본에서 SQLite 글자 → pandas 읽기를 거친다(판마다 자릿수가 다르고 pandas precise_xstrtod 도 정확한 반올림이 아님) → 분 합 상대차 ≤ 4.3e-16(정답지). vol60_pct 는 60초 거래량이 분 합과 1ulp 안으로 붙을 때만 갈린다.
- 청산맵 레벨(load_liquidation_map)은 인자 -- REST klines 90일 계산이라 다른 조각.
- **bookTicker(원본 1955 · REST 1/초 · 가중치 1) → 코어 @bookTicker 최신값으로 대신 = REST 0 (권장)**. 정의는 같다(둘 다 최우선 호가·수량) -- 다른 것은 «언제 찍은 값인가»뿐:

| 칸 | 원본(REST) | 코어 호가로 | 차이 |
|---|---|---|---|
| mid · spread · mid_ring | REST 응답이 만들어진 순간(왕복 지연만큼 옛 값) | 마지막 WS 갱신(더 새것) | 같은 순간이면 0 · 평소 수 ms~수백 ms 의 시세 차 |
| qi → agree(QI 쪽) | 같은 REST 값 | 같은 WS 값 | 위와 같음 · 히스테리시스 경계에서 라벨이 한 틱 일찍/늦게 |
| sweep 의 top 보정 · depth25 | 래스터 초(1초 늦음)와 REST 최우선의 조합 | 래스터 초 + WS 최우선 | 상계 칸 판정만 -- 수량 몇 ETH 수준 |
| sr(sup_bp·res_bp·near) | REST mid 기준 | WS mid 기준 | bp 몇 소수점 · near 경계에서만 |
| 실패 동작 | REST 실패 = 그 틱 예외 → payload {available: false} | 코어 호가가 없으면(첫 수신 전) 틱을 건너뛰어 같은 결과 | 없음 |

  → 값이 «다른 정의»가 되는 칸은 없다. 대조 때 mid·spread·qi 파생 칸은 «시점 차이»로 빼고, 래스터·기준선·청산 파생 칸만 글자 대조한다.

## 3. /api/market-context (market_context_payload 2601~2708 · api_market_context 2717~2732) -- 전부 ❌(그다음 조각)

응답: 3초 안 못 끝나면 shield 로 뒤에서 마저 · 마지막 성공 120초 안이면 `{**last, "stale": True, "stale_age_s", "error"}` · 넘으면 `{"available": False, "error"}`.

| 칸 | 원본 줄 | 원천 · 캐시 |
|---|---|---|
| mid | 2637~2638 | micro payload mid · 없으면 마크 링 끝 |
| bands_usd | 2668 | size_bands(sym) 상수 |
| funding.bn · bn_pct180 · next_ms · bn_at_base · okx · okx_next_ms · hl_8h | 2669~2672 | 마크 링 deriv_from_ring(1800초) · fundingRate REST 540개 swr 3600/6h · mp_state · mc_collectors(okx_fund · hl_ctx) swr 15/120 |
| basis.bp · d30_bp · pct7d · hl_premium_bp | 2673~2674 | 마크 링 · mc_history.basis(swr 300/1800) · hl_ctx |
| lev | 2675 | mctx.lev_state(basis 분위, oi z1h) |
| oi(+okx·hl·bybit·all) | 2641~2653 · 2676 | mc_history.oi(5분 격자) 끝 칸을 oi_1s 라이브로 · mc_collectors okx_oi·hl_ctx·by_oi · oi_per 7일 합 |
| ls(global·top_pos·taker·_24h·age_s) | 2656~2661 · 2677~2678 | mc_collectors ls 행 · 칸마다 마지막 유효값 · 24h±2h |
| quad · move60 · oi60 · flow.* · btc | 2679~2686 | §2 의 x · evidence(cvd·btc_move_bp·btc_rel) |
| venues.okx_bp · hl_bp | 2691~2693 | okx_mark(마크↔마크 그 초) · hl 미드↔mid_ring 그 초 |
| book.spread · sweep · bps · bid25_pct · ask25_pct | 2694~2696 | micro payload spread·sweep · depth25_ring(6시간 ≥600) |
| hl_liq | 2699~2700 | mc_collectors hl_pos · HL 마크 |
| liq_profile | 2619 · 2701 | hot binance_ctx liquidations → mctx.liq_profile swr 60/600 |
| profile | 2629~2631 · 2702 | tape_levels(MICRO_TAPE_DB) swr 300/1800 · 실패 {"available": False, "error"} |
| iv_rank | 2625 · 2703 | Deribit DVOL REST 366일 swr 3600/6h · ETH 만 |
| burst | 2705~2706 | micro liq60 · ETH 만 |
| errors | 2707 | mc_collectors errors |

REST(fundingRate · Deribit DVOL · klines 300)는 밴 가드 규칙대로 가짜 서버 녹화본으로만 대조한다(바깥 접속 금지).

## 이번 조각 검증
- tools/gen_ctx_fixture.py(서버 판 venv · flow_read 직접 import) → tests/ctx_parity.rs 4개: read 4,000건(기본 dumps + SSE compact 둘 다) · vol_size_mult 400건 · payload/SSE 300건 · patch_now.
- 양성 대조: 그 자리 변이(되돌림 바이트 대조) -- 결과는 보고에.
