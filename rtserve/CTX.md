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
| streams.fo | fo_state 1801 {connected, since, last_event_ms, events, errors, last_error} · 갱신 1843~1869 | @forceOrder WS 상태 | ❌ 코어 feed 상태를 이 꼴로 옮길 것(인자) |
| streams.mp | mp_state 1774 {connected, since, last_ms, events, errors, last_error} + next_funding_ms(1894) | @markPrice@1s WS 상태 | ❌ 위와 같음(인자) |
| read | 2491 `fr.read(evidence or {}, ctx)` · ETH 만(2489~2490) · 예외 2516~2517 `{"lines": [], "summary": "", "error": repr(exc)[:160]}` | ctx(§2) + evidence | 🟡 `ctx::read` = flow_read.read 250~427 · fuse 430~460 · outcome3 463~507 · gauges 197~247 전부 (글자 일치 4,000건) |
| vol_mult | 2451~2452 `fr.vol_size_mult(완결 candles_day, asset)` | candles_day = REST klines 300개 swr 60초(load_5m_day 2418~2438) | 🟡 `ctx::vol_size_mult` (일치 400건) · ❌ 5분봉 하루 원천(REST -- 코어 5분봉으로 대체할지 결정 필요) |
| vol_mult_by | 2738 코인마다 situation_state.vol_mult | 코인별 위와 같음 | 🟡 인자(코인별 값 목록) |

부수 효과(rtserve 가 하면 안 되는 것 · 파이썬이 도는 동안): FUSED_LOG 장부 쓰기 2492~2515(봉마다 한 줄) · card30 예측 캐시.

## 2. 관계 읽기 입력 x = `_flow_read_ctx` (2288~2406) -- 전부 ❌(다음 조각) · read 는 이미 x 를 인자로 받는다

| x 칸 | 원본 줄 | 원천 | Rust 쪽 재료 |
|---|---|---|---|
| net60 · z60 | 2295~2315 | 풋프린트 봉 셀 v[0..5](바이낸스 · 크기 세 갈래) → fr_bar_net · 12봉 합 · 과거 ≥72 의 std | 코어 풋프린트 봉(✅ 메시지로 옴) -- 합·convolve 차례 옮길 것 |
| z60h · z30h | 2318~2324 | micro_state.flow_scales(주말×UTC 시 14일 MAD · micro_ref) | ❌ 기준선 계산(테이프 14일) |
| cvd30_z | 2325~2328 | 같은 격자 a[:,0] 6봉 합 | 코어 풋프린트 봉 |
| venues · net60_all · z60h_all · net30_all · z30h_all · cvd30_all_z · taker_all | 2332~2373 | OKX 봉(onet = Micro.okx_bars ✅) · Bybit 봉(bybit_bars DB · swr 20초 2470~2472) · flow_scales_all · flow_all_hold | ❌ Bybit 봉 읽기 · 기준선 |
| okx30 | 2374~2376 | okx_bars 마지막 6봉(first_bar 뒤) | ✅ Micro.okx_bars |
| move60 · oi60 | 2377~2382 | inp.bars(5분봉 close · oi_delta 10개 이상) | ✅ 코어 situation 입력 봉 |
| move60_p75 | 2385~2388 | candles_day 300개(REST) np.percentile 75 | ❌ 5분봉 하루 |
| range30_pct | 2390~2396 | candles_day 293개 30분 창 고저폭 분위 | ❌ 5분봉 하루 |
| dir_p · reach_p | 2397~2398 `_card30_pred`(2266~2280 · HGB · ETH 모델 모든 코인) | rtmodel::card30 | ✅ rtmodel::card30(확인 필요: 같은 입력 봉 = 테이커 있는 완결 candles_day) |
| imb40 · imb40_pct | 2399~2403 | micro payload imb40 + imb40_ring(6시간 초별 · ≥600) | ❌ 깊은 호가(±0.4%) 불균형 -- 래스터/호가 |
| liq60s · agree · trigger_live · sr · act · vol_pct | 2404~2405 | micro payload(collect_micro_ref 1912~2022: liq60 · agree · trigger.live · sr(청산맵 레벨) · act · vol60_pct) | ✅ Micro.liq60 · vol60 · qi(코어) / ❌ agree·trigger(OFI 분위) · sr(청산맵 레벨) · vol60_pct(시간대 분위) |

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
