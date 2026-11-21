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

## 바깥 값이 크기를 정하는 할당 (2026-11-22 훑음 · ctx.rs·ctx_x.rs·ctx_micro.rs)
| 곳 | 크기를 정하는 것 | 한도 · 근거 |
|---|---|---|
| ctx_micro 래스터 창 | 파일 머리 n_bins · 행 머리 bin_lo | `crate::raster_read::read_window` 를 부르기만(사본 지움) -- 한도는 그 파일(rtserve 에이전트 몫) |
| ctx_micro `px_of`·`sweep_depth`·`raster_flow` 의 Vec | 창의 n_bins(위 한도 안) | 창 크기 × 12행 |
| ctx_micro `baseline_from_tape` 행 | SQL `GROUP BY ts_sec` · `ts_sec >= now − 7일` | 초마다 한 행 → 많아야 604,800행(약 10MB) -- 시각 범위가 상한 |
| ctx_micro 분 합 맵 | 위 행의 분 | ≤ 10,080 |
| ctx_micro 링 | ofi_hist·imb40_ring·depth25_ring | 600 · 21,600 · 21,600(원본 deque maxlen 1771~1773) · mid_ring 은 600초 넘은 것 지움(1959~1960) |
| ctx_x `tape_flow_rows` | SQL `GROUP BY 5분` · `[오늘0시 − days일, 오늘0시)` | ≤ days × 288 (14일 = 4,032) |
| ctx_x `flow_hour_scales_from_bars` 격자 | 행의 첫·끝 봉 | 위 시각 범위 안이라 ≤ days × 288 |
| ctx_x `bybit_bars` 봉 행 | SQL `GROUP BY 5분` · `[t0, t1)`(26시간) | ≤ 313 |
| ctx_x `bybit_bars` 끊김 행 | gaps 표 | `LIMIT 100000`(원본엔 없음 · ponytail 주석) |
| ctx_x `flow_read_ctx` 격자 | 코어 nets 의 첫·끝 닫힌 봉 | `MAX_GRID` 4,096봉 넘으면 합산 칸을 그 회차만 비움(원본엔 없음 · 코어 풋프린트 보관 288봉) |
| ctx `last_logged_bar` | 장부 파일 | 꼬리 16,384바이트만(원본 2412) |
| ctx_x `FlowNets`(받는 쪽) | 코어 Micro.flow 행 | 코어 풋프린트·OKX 봉 창 안 열쇠만 · `*_drop`·`full` 로 지움 |

기준선 주기: `MICRO_BASELINE_SECONDS = 3600`(server.py 972 · 정의) · 쓰이는 줄 = 1926 `now - micro_state["baseline_at"] >= (MICRO_BASELINE_SECONDS if micro_state["baseline"] else 60.0)` · 1939 `micro_state["baseline_at"] = now - MICRO_BASELINE_SECONDS + 60.0` (둘 다 Read 확인).

## C. REST 5분봉 재료(ctx_rest.rs · 2026-11-22) -- 전부 `RTSERVE_REST` 있을 때만 · 요청은 `ctx_rest::gate_fetch`(rtmodel::gate: 밴 파일 · 사용 가중치 1,200 넘으면 다음 분까지 쉼 · 간격 200ms)
| 재료 | 원본 줄 | 요청 | 캐시(swr) | 실패 때 원본 동작 | Rust |
|---|---|---|---|---|---|
| ETH·BTC 증거 프레임 → veto · 상황 캔들 tail 200 | load_chart_klines_frames 3185~3273 · 행 조립 1665~1716 | klines ETHUSDT·BTCUSDT 5m limit 1500(EVIDENCE_FETCH_LIMIT = scripts/live_evidence_signal_dashboard_20260823.py 144) | 60초 · max_stale 0(3269~3273 -- 늘 기다려 새로) | ETH 실패 = HTTPBadGateway evidence_signal_upstream_error → load_market_history 가 market_history_upstream_error(1679~1684) → compute_situation 예외 → now = {ok: false, reason: repr}(2020) · BTC 실패 = btc_df None → btc_candles None(2454~2457 · BTC 라벨만 빠짐) | `evidence_frame`(정렬·같은 시각 마지막·형성 봉 버림) → `candles_from_frame`(tail 200 + trend_veto tail 720) |
| SOL·XRP 상황 캔들 | load_market_history 1718~1736 · kline_candles 1648~1661 | klines 5m limit 1000 · timeout 3초 | 60초 · max_stale 600(STALE_GRACE 1040) | market_history_upstream_error → compute_situation 예외(위와 같음) | `kline_candles`(veto = 마감봉만 · 형성 봉 partial) |
| veto(12h 추세) | trend_veto_rows 1613~1646 · N 144(1597) · K 1.0(1611) | (위 프레임) | (위) | (위) | `trend_veto` = pandas rolling(144).mean()(roll_mean Kahan·같은 값·부호 고정) · ewm(alpha 1/144 → com 거쳐 다시 · adjust False) 비트 일치 |
| candles_day | load_5m_day 2418~2438 | klines 5m limit 300 · 시작+300 ≤ 지금인 행만 · taker r[9] | 60초 · max_stale 600 · 방금 닫힌 봉이 없고 2초 지났으면 식힘(2425~2426) | 예외면 로그만 · candles_day 는 옛 값 그대로(2446~2449) | `day5` · `day5_stale` |
| card30 | _card30_pred 2259~2280 · _card30_bars 2282~2286 · 모델 978~979(joblib) | 없음(candles_day 에서) | 봉마다 한 번(마지막 봉 시각이 같으면 그 값 · 코인마다) | 모델 없음/읽기 실패 = 한 번 알리고 {}(카드는 표로) | `Card30Cache` + rtmodel::card30::predict(card30.json = rtmodel/tools/export_card30.py 가 joblib 에서 뽑음 · 자리 결정 필요) |

검증: tools/gen_ctx_rest_fixture.py(옮겨 적은 trend_veto_rows·kline_candles·프레임 정리·행 조립·load_5m_day 거름 + 서버 판 pandas) → tests/ctx_rest_parity.rs: 24사례(ETH·BTC·SOL · 1500~144봉 · 형성 봉·중복·글자 자릿수) 캔들·veto·day5 · rolling/ewm 60급수 비트 일치.

### 상황 입력 봉의 고·저·종 -- 결정(2026-11-22): (a) 코어 그대로
원본은 REST 캔들(tail 200)의 high·low·close 를 쓰고(_situation_inputs 2189~2206), 코어 classify 는 체결로 센 5분봉을 쓴다. 바이낸스 klines 의 고·저·종은 같은 @trade 체결로 만들어진다.
**알려진 차이: 재연결로 체결을 놓친 봉(풋프린트 holes)에서만 원본 REST 캔들과 고·저·종이 다를 수 있다 -- 서버 병행 대조에서 holes 없는 봉은 같아야 한다.** REST 는 veto·candles_day 에만.

### card30 모델 -- 결정(2026-11-22)
`<저장소>/data/live/rtmodel/card30.json`(`ctx_rest::card30_path` · 내보내기 = rtmodel/tools/takeover.sh export-if-stale) · `ctx_rest::Card30File::get` 이 mtime 이 바뀌면 다시 읽는다 · 없거나 못 읽으면 원본 «모델 없음»(dir_p·reach_p = None → 카드는 표로). 원본과 다른 곳: 원본은 읽기 실패 뒤 다시 안 읽는다(card30["err"] 2255~2257).

## D. /api/market-context -- ctx_mc.rs (2026-11-22)
- 옮김: market_context_payload 2601~2708 전부 · api_market_context 묵은 값 갈래(2717~2732 → `mc_response`) · _mc_collectors(SQLite 여섯 + 롱숏비 duckdb) · _mc_history · _mc_liq_profile · tape_levels 931~969 · market_ctx 함수 11개(profile_nodes 의 np.exp·convolve "same" 세 갈래 포함) · LIQ_BURST_60S_USD 369.
- 검증: tools/gen_ctx_mc_fixture.py(market_ctx·micro_ref·situation import + 서버 함수 옮겨 적음 · 합성 파일 7개) → tests/ctx_mc_parity.rs: 본문 80사례(ETH·XRP · 마크 링 0~4000초 · depth25 599/700 · oi_1s 유무 · DVOL 150/366 · 펀딩 이력 0/10/540) · 수집기·이력·청산 프로파일·테이프 레벨 · 함수 360건 글자 일치. 양성 대조는 사본(scratchpad/mutproj mut_mc.py).
- **알려진 차이**:
  - `errors` 의 글자(원본 repr(exc)[:100] · Rust 오류 글자) -- 대조는 열쇠만.
  - 롱숏비 SQL: 원본 `ts >= now() - INTERVAL 25 HOUR`(ts = TIMESTAMPTZ · oi_lsratio_collector.py 78) -- 이 크레이트의 duckdb 는 ICU 가 없어 TIMESTAMPTZ 빼기를 못 한다 → `epoch(ts) >= 지금 − 90,000초`(같은 뜻 · 지금 = 벽시계).
  - tape_levels SQL 의 `floor()` = SQLite 수학 함수(파이썬 sqlite3 는 켜져 있음 · 이 크레이트의 묶음 SQLite 에는 없음) → 같은 값을 CAST 로(음수면 한 칸 아래).
- 붙이는 쪽 재료: 미시 payload(ctx_micro `MicroRef.payload`) · x(situation_tick 이 낸 것 -- 원본 situation_state["ctx"]) · evidence(코어 Situation) · 마크 링(코어 Mark 를 초 → (마크, 지수, 펀딩) · 7200초) · 미드 링(`MicroRef.mid_ring`) · depth25(`MicroRef.depth25_ring`) · oi_1s(hub oi_rows 의 링) · next_funding_ms(코어 Micro.streams.mp) · 캐시 `TTL` 표(2617~2629) · REST 몫 = `funding_hist_url`/`funding_hist`(바이낸스 · gate)·`dvol_url`/`dvol_year`(Deribit · gate.send(url, false) · ETH 만) -- 실패·RTSERVE_REST 없음 = 빈 목록(원본 2622~2627).

### 붙이는 쪽 사용법 -- `/api/market-context?asset=`
1. lib.rs 에 `pub mod ctx_mc;` · 코인 표에 없는 asset 은 404 `flow_off`(2719~2720).
2. 코인마다 상태: `mc_last_good: Map`(수집기 마지막 성공값) · `mc_last: Option<(V, f64)>`(마지막 성공 본문 · 시각).
3. 파일 몫은 swr 로(`ctx_mc::TTL` 의 ttl·max_stale 그대로): `mc_collectors(&paths, asset, okx_inst, symbol, &mut mc_last_good)` · `mc_history(&paths, now_ms, sym_lower, okx_inst, COIN, SYMBOL)` · `mc_liq_profile(&paths, SYMBOL, now, px_scale)` · `tape_levels(tape, sym_lower, 테이프 칸 폭, now)`(Err → `{"available": false, "error": …}`).
4. REST 몫(`RTSERVE_REST` 있을 때만 · gate): `funding_hist(gate_fetch(funding_hist_url(..)))` · ETH 만 `dvol_year(gate.send(dvol_url(now_ms), false))` -- 실패·꺼짐 = `[]`.
5. 메모리 몫으로 `McIn { asset, now, bands_usd(풋프린트 메시지의 retail_max_usd·whale_min_usd), micro: &MicroRef.payload, x(situation_tick 이 낸 x), ev(코어 evidence), mark_ring, mid_ring, depth25, oi_1s, next_funding_ms, px_scale(코인 풋프린트 칸 ÷ ETH 칸) }`.
6. `let r = market_context_payload(&i, &col, &hist, &prof, &fh, &dvy, &vprof)` -- 원본은 3초 안에 못 끝나면 실패 갈래(뒤에서 마저 돌려 캐시를 채움 · 2710~2715) → `mc_response(Ok(r) 또는 Err("TimeoutError: …"), &mut mc_last, now)` 를 json_response(NOCACHE)로.

## 3. 청산맵 레벨(load_liquidation_map 3461~3508) -- 표(옮기기 전)
| 항목 | 원본 | 값 |
|---|---|---|
| 캐시 | swr `liquidation_map:{asset}` 3505~3508 | ttl LIQUIDATION_MAP_CACHE_SECONDS(graft 213 = 60 -- 정의 줄 Read 미확인) · max_stale STALE_GRACE 600 |
| 부르는 곳 | collect_micro_ref 1991(매 틱 · 코인마다) · /api/liquidation-map | 틱은 1초지만 swr 60초라 실제 계산은 코인마다 분당 1번 |
| REST(회차마다) | liq_klines_df 3314~3337 ×2 | 1시간봉 LIQUIDATION_MAP_FETCH_LIMIT(쪽마다 1500 · 더 많으면 endTime 으로 앞쪽) + 5분봉 LIQUIDATION_MAP_5M_FETCH_LIMIT · 형성 봉 버림(keep_forming 거짓) |
| 그 밖 계산 | 3482~3501 | heatmap_history(compute_spliced_heatmap_history_5m) · tier_profile(+ liq_oi_usd = hot oi_1s·OKX·Bybit) -- 미시 payload(sr)는 **levels 만** 쓴다 |
| 미시가 읽는 칸 | 1991~1995 · micro_ref.sr_context 120~142 | warmed_up · support_levels·resistance_levels 의 price·weight_pct(가까운 순) |
| 레벨 계산 | scripts/live_liquidation_map_20260824.py compute_spliced_levels 303~362 · levels_from_bins 238~265 · _prepare_common 138~ · _bins_from_common 177~ · _own_bar_alive 107~ | 닫힌 1시간봉 tail(24) · 지지 = (고+저)/2 패스 · 저항 = 종가 패스 · 5분봉으로 자기 봉 안 쓸림 판정 |

재사용 판단: rtmodel::regime_band 에 같은 원본(_prepare_common·_bins_from_common·_own_bar_alive·levels_from_bins 한쪽)을 옮긴 것이 있다(regime_band.rs 116~170). 다만 ① 함수가 비공개(`prepare`·`bins`·`side`) ② 입력이 «5분봉 패널을 1시간으로 접은 것 + 이번 시간 부분봉»(_window_at) -- load_liquidation_map 은 REST 닫힌 1시간봉 tail(24) 그대로 ③ `side` 가 가격만 낸다(weight_pct·distance_pct 없음). → **rtmodel 의 그 셋을 pub 으로 열고 `side` 가 (가격, 무게%)를 내게 넓히는 것**이 가장 짧다(같은 원본을 둘로 옮기지 않게). rtmodel 은 내 쓰기 범위 밖이라 결정이 필요하다.

## 3-옛. /api/market-context (market_context_payload 2601~2708 · api_market_context 2717~2732) -- 전부 ❌(그다음 조각)

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
