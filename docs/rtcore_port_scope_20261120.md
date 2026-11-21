# 러스트 이전 범위표 — «무엇이 실제로 쓰이고, 무엇이 안 쓰이는가» (2026-11-20)

읽기 전용 조사(코드·문서만 읽음 · 프로그램 실행·서버 접속·네트워크 호출 없음 · `.env` 안 읽음). 사용자 결정 «안 쓰는 것은 옮기지 않는다»를 적용하기 위한 한 장짜리 범위표다. 배경: `docs/rtcore_design_20261120.md` §11(R1~R7), `docs/rtcore_display_latency_map_20261120.md`, `docs/data_storage_map_20261107.md`.

## 0. 읽는 법

- **판정**: `쓰임` = 화면·다른 프로그램·전진 검정·감시 중 하나가 실제로 읽는다 / `안 쓰임` = 저장소 안에서 읽는 곳을 못 찾았고 그 부재를 서로 다른 방법으로 두 번 이상 확인했다 / `불확실` = 한쪽으로 못 박을 근거가 없다(질문은 §5).
- **«쓰임·전진»** = 전진 검정(사전등록 판정) 때문에 끊으면 안 되는 수집·기록. 판정일은 `dashboard/verdict_calendar.json` 의 항목 id 로 적었다. `docs/deploy_queue.md` 는 **없음**(`docs/deploy_queue_regime_band_20261102.md` 하나만 있고 지금 판정과 무관).
- **근거 표기**: `파일:줄` 은 직접 읽은 곳. «grep `문자열`» 은 그 문자열로 저장소(연구·docs·tmp·시험 제외, 시험은 따로 적음)를 검색해 0건/몇 건이었다는 뜻이다.
- **한계 하나(중요)**: `dashboard/server.py` 는 개인정보 보호 플러그인 때문에 grep 이 막혀 **Read 로 읽은 줄만** 근거로 썼다(§6 에 읽은 구간). 그래서 server.py 안의 «함수 단위 죽은 코드»는 전수 조사가 아니다.
- 휴대폰 앱 계약(`mobile/`)은 이 워크트리에 없다 → 모든 «화면 호출 0건»은 **앱 계약 미확인** 조건부다(§5 질문 1).
- 서버 crontab 은 저장소에 없다. 사용자가 알려 준 목록(2026-11-20 서버에서 읽음)을 믿고, 저장소 스크립트 머리말의 `cron` 설명으로 보충했다. 재확인 불가.

## 1. 표 1 — 서버 엔드포인트 (`dashboard/server.py:4954~5044` 등록 + `dashboard/orders.py` 핸들러)

등록 81개(경로 79개 · `acct-notes`·`acct-wrong` 이 GET/POST 한 쌍). 페이지·정적 6개는 맨 위에 따로. 화면 주기는 `dashboard/live/app.js` 의 `*_POLL_MS` 상수(app.js:16~1670, 3204~3326, 5605~5724, 12133, 12726) + 틱 허브(app.js:11553~11601, 500ms 틱 위에서 각 함수의 게이트)로 읽었다. 허브 양자화는 지연 지도 §1 참조.

**URL 이 조립되는 곳 확인**: `/api/manual-${kind}/preview|submit`(app.js:12536 · 13027, kind = entry|exit — `kind === "exit"` 분기 app.js:12531~12535) · `/api/manual-bracket/${act==="on"?"rearm":"disarm"}`(app.js:1462 · 12918) · `/api/manual-bracket/tp?act=`(app.js:1462) · `API_*_URL` 상수(app.js:1~9·83·127·184·350·353·1567·1574·1579·4721·5924) 전부 따라가 썼다. `fp_daily.js:44~46` 의 `PERIODS` 표가 일봉·4시간봉 URL 6개를 만든다.

### 1-A. 페이지·정적 (6)

| 경로 | 핸들러 | 화면 | 화면 밖 | 판정 | 근거 |
|---|---|---|---|---|---|
| `/` · `/dashboard/live` · `/dashboard/live/` · `/dashboard/live/index.html` | `index` · `dashboard_index`(버스터 치환) | 진입 페이지 | 터널·휴대폰 | 쓰임 | server.py:4954~4959 |
| `/dashboard/live/{sw.js,manifest.webmanifest}` | `pwa_asset`(no-cache) | `index.html:21`(manifest) · app.js:11728(`PUSH_SW_URL`) | 푸시 구독 | 쓰임 | server.py:4962 |
| `/dashboard/live/*`(정적) | `add_static` | index.html:31·641~645(styles.css·app.js·fp_daily.js·liq_profile.js·liq_tab.js·bubble_map.js) | — | 쓰임 | server.py:5050 · index.html:641~645 |

### 1-B. API (81)

| 경로 | 핸들러 | 화면 호출처 · 주기 | 화면 밖 호출처 | 판정 | 근거 |
|---|---|---|---|---|---|
| `GET /api/state` | `api_state` | **0건** (app.js 에 문자열 없음) | 시험만: `test/test_dashboard_server.py` · `test/test_dashboard_sse_state_view_20260920.py` · 앱 계약 미확인 | **안 쓰임** | grep `api/state` (dashboard/live·scripts·.sh·.py): scripts 는 `scripts/live_eth_chart_markers_20260909.py:108` 주석뿐. 화면은 상태를 `/api/events` 의 `state` 키로 받는다(app.js:2268 `payload?.state?.state`) |
| `GET /api/events`(SSE 1초) | `api_events` | app.js:2280 `new EventSource(API_EVENTS_URL)` · 연결 유지 | — | 쓰임 | 시세 8코인·`acct_seq`·상태 3블록. 지연 지도 §5-1 |
| `GET /api/stream`(SSE 0.25초) | `api_stream` | app.js:4741 `new EventSource(API_STREAM_URL + q)` | `rtcore/tools/shadow_supply.py`(그림자 대조 도구) | 쓰임 | 수급·상황 이벤트. app.js:4743~4753 |
| `GET /api/market-history` | `api_market_history` | app.js:2157 · 5분(`CANDLE_HISTORY_POLL_MS` app.js:38) + 15초 재시도 | 시험 `test_market_history_forming_20261011.py` 등 | 쓰임 | 5분봉 과거 |
| `GET /api/footprint` | `api_footprint` | app.js:6599 · 실효 0.5초(`FOOTPRINT_POLL_MS`=400 → 틱 양자화) | **`scripts/ops_watchdog.py:197`(`check_footprint_history`)** · `scripts/live_llm_brief_20261017.py:130` | 쓰임 | 닫힌 봉·보조 원천. 끄면 감시 경보 |
| `GET /api/footprint-week` | `api_footprint_week` | app.js:6644 · 필요할 때 한 번 / 6시간(`FP_WEEK_REFRESH_MS`) | 시험 `test_footprint_backfill_covers_window_20260923.py` | 쓰임 | 12시간 밖 1주일 |
| `GET /api/footprint-daily` | `api_footprint_daily` | fp_daily.js:44 `days` · 필요 시 + 30초~ 갱신 | `live_llm_brief_20261017.py:132` | 쓰임 | 일봉 풋프린트 |
| `GET /api/footprint-daily/cells` | `api_footprint_daily_cells` | fp_daily.js:44 `cells` | `live_llm_brief_20261017.py:134` | 쓰임 | 〃 |
| `GET /api/footprint-daily/xcvd` | `api_footprint_daily_xcvd` | fp_daily.js:631 | — | 쓰임 | 거래소별 CVD 레인 |
| `GET /api/footprint-daily/heat` | `api_footprint_daily_heat` | fp_daily.js:44 `heatUrl` | — | 쓰임 | 히트맵 |
| `GET /api/footprint-4h`(+`/cells` `/heat`) | `api_footprint_4h*` 3개 | fp_daily.js:46 | — | 쓰임 | 4시간봉 |
| `GET /api/supply-1s` | `api_supply_1s` | app.js:4617 · **스트림이 꺼졌을 때만**(app.js:4611 `if (liveStreamOn()) return`) 0.25~0.5초 | `rtcore/tools/shadow_supply.py` | 쓰임(폴백) | 평소엔 `/api/stream` 이 대신함. 폴백 보존 여부는 설계 결정 |
| `GET /api/oi-5m` | `api_oi_5m` | app.js:5338 · 15초(`OI_5M_POLL_MS` app.js:1565) | 시험 `test_oi5m_okx_coverage_guard_20260925.py` | 쓰임 | server.py:4366~4414 |
| `GET /api/breakout-detector` | `api_breakout_detector` | **0건** | — (시험 `test_breakout_*`만) | **안 쓰임** | server.py:4478~4483 · grep `api/breakout-detector`: server.py 한 곳뿐. app.js 에는 `live_eth_breakout_detector` 주석 언급 외 호출 없음 |
| `GET /api/gex` | `api_gex` | app.js:5259 · 60초(`GEX_POLL_MS`) | — | 쓰임 | 옵션 카드. server.py:4737 |
| `GET /api/opt-smile` | `opt_smile` 핸들러 | app.js:5265 | — | 쓰임 | 〃 |
| `GET /api/flow/heatmap` | `api_flow_heatmap` | app.js:5285~5287(`mode=rows`, 1초 게이트) · bubble_map.js:66(이미지 모드 `qty_i8` bubble_map.js:70~72) | — | 쓰임 | 두 모드 다 호출됨. server.py:4695~4735 |
| `GET /api/bubble-tape` | `api_bubble_tape` | bubble_map.js:49 · 1초 자체 타이머 | — | 쓰임 | 체결 버블 |
| `GET /api/chart-markers` | `api_chart_markers` | app.js:3164 · 60초 | — | **불확실** | 화면은 읽어 그리지만(app.js:10026·10885, `cm.available` 일 때만) 서버가 지금 `available:true` 를 주는지 코드로 확정 못 함(`scripts/live_eth_chart_markers_20260909.py:74~129` 가 `available:False` 분기 4개). §5 질문 3 |
| `GET /api/liquidation-5m-signal` | `api_liquidation_5m_signal` | app.js:3249 · 60초(`LIQUIDATION_5M_POLL_MS`) — **응답 `j` 를 버림** | — | **안 쓰임(값)** · 호출은 의존 | app.js:3249~3252 에서 `res.json()` 결과 `j` 를 이후 한 번도 안 쓴다. ⚠이 응답이 실패(`!res.ok`)하면 `throw` 로 아래 `/api/liquidation-5m-history` 호출(3257)까지 건너뛴다 → 옮길 때 «값 계산»은 빼도 되지만 **호출 성공 의존을 먼저 끊어야** 한다 |
| `GET /api/liquidation-5m-history` | `api_liquidation_5m_history` | app.js:3215(꼬리 2초) · 3257(전량 60초) · liq_tab.js:59 | `live_llm_brief_20261017.py:135` | 쓰임 | server.py:4416~4458 |
| `GET /api/liquidation-map` | `api_liquidation_map` | app.js:3275 · 30초 | `live_llm_brief_20261017.py:128` | 쓰임 | 청산맵 |
| `GET /api/liquidation-map/tiers` | `api_liquidation_tiers` | liq_profile.js:15 | `live_llm_brief_20261017.py:129` | 쓰임 | |
| `GET /api/sr-watch` | `api_sr_watch` | **0건** | **`scripts/live_sr_proximity_alert_20261104.py:31`**(10초 폴링, `supervisor_sr_proximity.sh`) | 쓰임(워커) | 화면이 아니라 알림 워커가 읽음. `sr_watch_view` server.py:246 |
| `GET /api/position-profile` | `api_position_profile` | app.js:1556 · 켤 때만 1초 · fp_daily.js:389 | — | 쓰임 | |
| `GET /api/cycle-path` | `api_cycle_path` | app.js:2904 · 계좌 카드 열 때 | 시험 `render_account_cycle_20261022.py` | 쓰임 | |
| `GET /api/macro-calendar` | `api_macro_calendar` | app.js:3315 · 6시간 | `live_llm_brief_20261017.py:136` | 쓰임 | |
| `GET /api/situation` | `api_situation` | app.js:5365 · 1초 **폴백만**(app.js:5361 `liveStreamOn()` 이면 건너뜀) | — | 쓰임(폴백) | 평소엔 `/api/stream` 의 `situation` 이벤트 |
| `GET /api/trend` | `api_trend` | app.js:5382 · 60초 | `live_llm_brief_20261017.py:136` | 쓰임 | |
| `GET /api/swing-lines` | `api_swing_lines` | app.js:5398 · 켰을 때만 60초 | — | 쓰임 | |
| `GET /api/dir-signals` | `api_dir_signals` | app.js:5915 · 10분 | — | 쓰임 | |
| `GET /api/regime-band` | `api_regime_band` | app.js:5613 · 5분 | — | 쓰임 | 전진 판정 fwd_band(2027-01-08) 표시 |
| `GET /api/risk-radar` | `api_risk_radar` | app.js:5713 · 30초 | — | 쓰임 | |
| `GET /api/sr-lean` | `api_sr_lean` | app.js:5627 · 2초 | — | 쓰임 | |
| `GET /api/liq-tab` | `api_liq_tab` | liq_tab.js:39(코인) · 49(`asset=all`) | — | 쓰임 | |
| `GET /api/vol-forecast` | `api_vol_forecast` | app.js:5573 · 5분 | — | 쓰임 | |
| `GET /api/dir4h` | `api_dir4h` | app.js:5496 · 2분 | — | 쓰임 | |
| `GET /api/dir-short` | `api_dir_short` | app.js:5517 · 1분 | — | 쓰임 | |
| `GET /api/pred-record` | `api_pred_record` | app.js:5732 · 1분 | — | 쓰임 | |
| `GET·POST /api/acct-notes` | `api_acct_notes`(`_save`) | app.js:2482(읽기) · 2515(저장) | — | 쓰임 | |
| `GET·POST /api/acct-wrong` | `api_acct_wrong`(`_save`) | app.js:2540 · 2546 · index.html:586 | — | 쓰임 | |
| `GET /api/box-hold` | `api_box_hold` | app.js:5547 · 1분 | `live_llm_brief_20261017.py:128` | 쓰임 | |
| `GET /api/open-rev` | `api_open_rev` | app.js:5529 · 2분 | 시험 `render_fc_combo_20261016.py` | 쓰임 | |
| `GET /api/iceberg` | `api_iceberg` | app.js:5421 · 30초 | — | 쓰임 | 워커 `supervisor_eth_iceberg.sh` |
| `GET /api/ob5m` | `api_ob5m` | app.js:6672 · 5초 | — | 쓰임 | |
| `GET /api/market-context` | `api_market_context` | app.js:6075 · 1초(`MARKET_CTX_POLL_MS`) | **`live_push_notifier_20260904.py:240`** · `live_llm_brief_20261017.py:128` | 쓰임 | 끄면 푸시 알림 입력이 빈다 |
| `GET /api/chip-score` | `api_chip_score` | app.js:12085 · 5분 | — | 쓰임 | |
| `GET /api/notify-center` | `api_notify_center` | app.js:12050 · 5분 | — | 쓰임 | 판정 달력·푸시 종류 |
| `GET /api/news` | `api_news` | app.js:3340 · 30초 | `live_llm_brief_20261017.py:136` | 쓰임 | |
| `GET /api/treasury-futures` | `api_treasury_futures` | app.js:3422(`?h=24`) · 3637 · 60초 | — | 쓰임 | |
| `GET /api/macro-trend` | `api_macro_trend` | app.js:3383 · 10분 | — | 쓰임 | |
| `GET /api/xyz100` | `api_xyz100` | app.js:3701 · 1초 | — | 쓰임 | |
| `GET /api/macro-today` | `api_macro_today` | app.js:3425 · 1분 | — | 쓰임 | |
| `GET /api/macro-spikes` | `api_macro_spikes` | app.js:3732 · 30초 | — | 쓰임 | |
| `GET /api/session-alerts` | `api_session_alerts` | app.js:3129 · 30초 | — | 쓰임 | server.py:4862 |
| `GET /api/push/config` | `api_push_config` | app.js:11893 | — | 쓰임 | server.py:4878 |
| `POST /api/push/subscribe` | `api_push_subscribe` | app.js:11905·11942 | — | 쓰임 | |
| `POST /api/push/unsubscribe` | `api_push_unsubscribe` | app.js:11870·11930 | — | 쓰임 | |
| `GET /api/push/devices` | `api_push_devices` | app.js:11901·11909 | — | 쓰임 | |
| `POST /api/push/test` | `api_push_test` | app.js:11975 | — | 쓰임 | |
| `GET /api/binance-account` | `order_api.api_binance_account` | app.js:2981 · 10초 + `acct_seq` 알림 시 즉시 | **`live_push_notifier_20260904.py:237`** · **`scripts/ops/morning_audit_20261012.py:160`** | 쓰임 | |
| `GET /api/manual-entry/preview` | `api_manual_entry_preview` | app.js:12536(`manual-${kind}`) · 사용자 조작 시 | — | 쓰임 | |
| `POST /api/manual-entry/submit` | `api_manual_entry_submit`(`busy_or`) | app.js:13027 | — | 쓰임 | 실주문 경로(R2) |
| `GET /api/manual-entry/status` | `api_manual_entry_status` | app.js:12969 · 13118 · 진행 중 1.5초 / 대기 3초 | — | 쓰임 | |
| `POST /api/manual-entry/wait-action` | `api_manual_wait_action` | app.js:632 · 13134 | — | 쓰임 | |
| `POST /api/manual-entry/cancel` | `api_manual_entry_cancel` | app.js:788 · 12960 | — | 쓰임 | |
| `POST /api/manual-entry/limit-amend` | `api_manual_limit_amend` | app.js:644 | — | 쓰임 | |
| `POST /api/manual-bracket/disarm` · `/rearm` | `api_manual_bracket_disarm`·`_rearm` | app.js:1462 · 12918 | — | 쓰임 | orders.py:2269·2301 |
| `POST /api/manual-bracket/tp` | `api_manual_bracket_tp` | app.js:1462(`tp?act=`) | — | 쓰임 | |
| `POST /api/manual-bracket/tp-legs` | `api_manual_bracket_tp_legs` | app.js:1437 | — | 쓰임 | |
| `POST /api/manual-bracket/add` | `api_manual_bracket_add` | app.js:993 | — | 쓰임 | |
| `POST /api/manual-bracket/modify` | `api_manual_bracket_modify` | app.js:1095 | — | 쓰임 | |
| `GET /api/manual-exit/preview` | `api_manual_exit_preview` | app.js:12536(`kind=exit`) | — | 쓰임 | |
| `POST /api/manual-exit/submit` | `api_manual_exit_submit` | app.js:13027(`kind=exit`) | — | 쓰임 | |
| `POST /api/manual-exit/submit-all` | `api_manual_exit_all_submit` | **0건**. 화면의 «전체 청산» 단추는 커밋 `57892f41`(포지션 칸 삭제)에서 사라졌다 | 시험 `test/test_manual_exit_all_20261022.py` · 앱 계약 미확인 | **불확실** | orders.py:2209~2268(약 60줄, 실주문). grep `submit-all`: app.js 0건. 휴대폰 앱이 부르는지 모름 → §5 질문 2 |
| `POST /api/client-timing` | `api_client_timing` | app.js:34 · 60초마다(값이 있을 때만) | 시험 `test_m0_timing_20261120.py` | 쓰임(측정용·임시) | server.py:5127~5145. 2026-11-20 측정용. 코어 이전 뒤 측정이 끝나면 뺄 수 있음 |

**표 1 요약**: 쓰임 76 · 안 쓰임 3(`state`·`breakout-detector`·`liquidation-5m-signal` 값) · 불확실 2(`chart-markers`·`manual-exit/submit-all`). 합 81. 쓰임 76 안에 «폴백만»(supply-1s·situation) 2개와 «화면 아닌 워커가 읽음»(sr-watch) 1개가 들어 있다.

## 2. 표 2 — 상주·예약 프로그램

### 2-A. 상주 (`scripts/ops/supervisor_*.sh` 25개 + 그 밖 상주)

전부 `scripts/ops/_supervise.sh` 로 죽으면 재기동. 감시자(`scripts/ops_watchdog.py`, systemd `ops-watchdog.service`)가 신선도를 점검하는 것은 «감시» 칸에 적었다 — **끄면 그 점검이 경보를 낸다**(해당 점검 줄을 같이 지워야 함).

| # | 프로그램(스크립트 줄수) | 출력물 | 읽는 곳 | 판정 | 근거 |
|---|---|---|---|---|---|
| 1 | `supervisor_binance_ctx` → `live_binance_ctx_collector_20261001.py`(224) | `data/hot/binance_ctx.sqlite`(OI 1초·마크 1초·청산) | 서버 server.py:62 `BINANCE_CTX_DB` · 청산 `live_liq_tab_20261107.py` · 전진 기록기 `live_liqmap_scorer_20261008.py` · 감시 `ops_watchdog.py:936~939` | 쓰임·전진(liqmap_fwd 2026-12-07) | grep `binance_ctx` |
| 2 | `supervisor_bookdepth` → `live_bookdepth_feat_20261028.py`(219) | `data/live/bookdepth/ethusdt.sqlite` | `live_eth_dir_short_20261028.py:43·49`(15분·1시간 방향 피쳐) | 쓰임·전진(fwd_dirshort 2027-01-08) | |
| 3 | `supervisor_breakout_detector` → `live_eth_breakout_detector_worker_20260911.py`(98)(+ 라이브러리 `live_eth_breakout_detector_20260911.py` 181) | `data/live/{eth,sol,xrp}_breakout_detector_state.json` | 소비자는 `/api/breakout-detector`(화면 0건)와 감시자 매니페스트 `scripts/ops/multicoin_collectors_20260926.py:129~130`(신선도 15/30분)뿐 | **안 쓰임** | grep `breakout_detector_state`: server.py · 시험 · 워커 · 매니페스트. 화면 호출 0(표 1) |
| 4 | `supervisor_bybit_liq` → `live_bybit_liquidation_collector_20261003.py`(121) | `data/hot/bybit_liq.sqlite` | 서버 청산 원 합산(server.py:4447~4450 `bybit_liq_events`) · `live_liqmap_scorer_20261008.py` | 쓰임·전진(liqmap_fwd) | |
| 5 | `supervisor_bybit_tape` → `live_bybit_trade_tape_collector_20261006.py`(318) | Bybit 테이프·OI·마크·펀딩 | 시장 맥락 OI 합산(`dashboard/market_ctx.py:140·147`) · `seal_to_lake.py` | 쓰임 | 테이프 자체의 소비자는 lake 봉인뿐 |
| 6 | `supervisor_depth_diff` → `live_depth_diff_collector_20260915.py`(173) | `orderflow/depthdiff/<코인>` 원시 | `live_ob5m`·`live_bookdepth_feat`·`live_eth_iceberg`(다른 워커) · 감시 `ops_watchdog.py:959` | 쓰임 | grep `depthdiff` |
| 7 | `supervisor_eth_iceberg` → `live_eth_iceberg_20261016.py`(291) | `data/live/iceberg/<코인>_state.json` | `/api/iceberg`(app.js:5421) · 감시 `ops_watchdog.py:964~965` | 쓰임 | |
| 8 | `supervisor_hyperliquid_book_ticker` → `live_hyperliquid_book_ticker_collector_20260923.py`(298) | ① 원시 BBO `orderflow/hyperliquid_bookticker/ETH` ② `HL_BT_CTX_ONLY=1` 변형 → `data/hot/hl_ctx.sqlite`(펀딩 등) | ② 시장 맥락 «HL 펀딩 8h»(`live_llm_brief_20261017.py:88` 가 `mc.funding.hl_8h` 사용 — 서버 쪽 읽는 줄은 못 확인) · ① 은 **감시 `ops_watchdog.py:961` · `raw_failover` · 매니페스트뿐, 읽는 코드 0** | **불확실** | 호가 전진 검정(ob_mid 등)은 10-31 사용자 결정으로 중단(`verdict_calendar.json`). 원시 BBO 를 연구로 쌓는지는 모름 → §5 질문 4 |
| 9 | `supervisor_hyperliquid_positions` → `live_hyperliquid_positions_collector_20260924.py`(396) | `data/hot/hl_positions*.sqlite` | `dashboard/position_profile.py` · 서버 HL 고래 청산 server.py:498~502·4451~4476 | 쓰임 | |
| 10 | `supervisor_hyperliquid_trades` → `live_hyperliquid_trade_collector_20260916.py`(209) | `orderflow/hyperliquid/<코인>` | 위 #9 가 입력으로 읽음(`live_hyperliquid_positions_collector_20260924.py:11·64`) · 감시 `ops_watchdog.py:960` | 쓰임(간접) | 고래 맞대결(whale_duel1)은 HL 이 아니라 **바이낸스+OKX 테이프**를 읽음(`docs/experiments/eth_whale_duel_fwd_prereg_20261007.md:68`) → 전진과 무관 |
| 11 | `supervisor_market_data_worker` → `scripts/duckdb_persist_worker.py`(142) + 루트 `microstructure_scanner.py`(884) · `tail_risk_interceptor.py`(543) · `oi_lsratio_collector.py`(301) | `dashboard_state.json`(session 3칸+updated_at) · `tail_risk.duckdb` + `liq_burst_state.json` · `microstructure.duckdb`(ETH·BTC·SOL) | ⓐ `dashboard_state.json` → 서버 `/api/events state` → 화면 `render()`(app.js:11521~11528 `topSession`) · 감시 `ops_watchdog.py:162` · ⓑ `liq_burst_state.json` → 서버(server.py:367) 청산 급증 배지(app.js:6110) · ⓒ `tail_risk.duckdb` → `compute_liquidation_5m_history`(server.py:4431~4436 폴백) · ⓓ `microstructure.duckdb` → **lake 봉인·감시뿐**(grep `microstructure_1m`: seal_to_lake · coin_config · live_trade_tape_collector 주석) | 쓰임(ⓐⓑ) / **불확실**(ⓒⓓ) | ⓒ 는 «ETH 외 코인에서 메모리 누적이 없을 때만» 타는 길이라 지금 타는지 모름. ⓓ 는 화면·라이브 읽기 0 → §5 질문 5·6 |
| 12 | `supervisor_news_judge` → `live_news_judge_20261005.py`(218) | `news_rss.sqlite` 의 judgments | `/api/news` 의 판정 칸 | **불확실**(정지 중) | 2026-10-27 부터 **의도적 정지**(메모리 `news_judge_paused`: v24 학습 끝나면 재개). 재개 의사 확인 필요 → §5 질문 7 |
| 13 | `supervisor_news_rss` → `live_news_rss_collector_20261005.py`(191) | `data/hot/news_rss.sqlite` | `/api/news`(app.js:3340) · server.py:559 `NEWS_DB` | 쓰임 | |
| 14 | `supervisor_ob5m` → `live_ob5m_20261026.py`(748) | `data/live/ob5m/<코인>.sqlite` | `/api/ob5m`(app.js:6672) | 쓰임 | |
| 15 | `supervisor_oi_lsratio_worker` → `duckdb_persist_worker.py`(OI·롱숏비 전용, ETH·BTC·SOL·XRP) | OI·롱숏비 duckdb | `seal_to_lake.py` · `live_evidence_signal_dashboard_20260823.py` 만 언급. 화면 «롱숏비»가 이 파일을 읽는지는 server.py 해당 구간을 못 읽어 미확인 | **불확실** | grep `oi_lsratio`. §5 질문 5 |
| 16 | `supervisor_okx_context` → `live_okx_context_collector_20260923.py`(419) | `okx_ctx.sqlite` | 서버 `OKX_CTX_DB_PATH`(server.py:434 import) · `live_liq_tab`·`live_liqmap_scorer` | 쓰임·전진(liqmap_fwd) | |
| 17 | `supervisor_okx_trade_tape` → `live_okx_trade_tape_collector_20260923.py`(482) | OKX 테이프 hot | 서버 `OKX_TAPE_DB_PATH`(server.py:434) · `seal_to_lake` → 고래 맞대결 | 쓰임·전진(whale_duel1 2027-11-26) | |
| 18 | `supervisor_ollaya` | 로컬 LLM 서버(`:11435`) | `live_news_judge_20261005.py:30` 만 | **불확실**(정지 중) | #12 와 한 쌍 |
| 19 | `supervisor_orderflow_raster` → `live_orderflow_raster_collector_20260914.py`(898) | `orderflow/raster/<코인>/*.f32` | 서버 `read_window`(server.py:4494·4641·4661·4686) → 히트맵·시장 맥락 · 감시 `ops_watchdog.py:958` | 쓰임 | |
| 20 | `supervisor_push_notifier` → `live_push_notifier_20260904.py`(337) | 웹푸시 발송 + `push_sent_log.jsonl` | 사용자 폰 · 서버 알림 센터(server.py:334·344 읽음) · 서버가 `PUSH_KINDS` 만 import(server.py:148) | 쓰임 | |
| 21 | `supervisor_signal_worker` → `live_signal_worker.py`(172) + `live_eth_sweep_v_rebound_signal_20260829.py`(610) | `data/live/eth_v_rebound_state.json` | **읽는 곳 0**(서버 엔드포인트는 삭제됨: `live_push_notifier_20260904.py:117` 주석 «지워져 404») | **안 쓰임** | grep `eth_v_rebound_state`: supervisor·워커·옛 문서뿐. 지금 서버에서 도는지는 crontab 미확인(돌아도 안 읽힘) |
| 22 | `supervisor_sr_lean` → `live_sr_lean_20261118.py`(344) | `sr_lean/ethusdt.jsonl`(사건·결과) · `ethusdt_feat.jsonl`(피쳐 87개) | `/api/sr-lean`(app.js:5627) · 전진 sr_ml_5m_forward(2026-12-17) | 쓰임·전진 | |
| 23 | `supervisor_sr_proximity` → `live_sr_proximity_alert_20261104.py`(108) | 텔레그램 «가까워짐» 알림 | 사용자 · 입력 `/api/sr-watch` | 쓰임 | |
| 24 | `supervisor_trade_tape` → `live_trade_tape_collector_20260916.py`(1,207) | `data/hot/binance_tape.sqlite`(+현물) | 서버(server.py:53 `HOT_TAPE_DB`) · `seal_to_lake` → whale_duel · 감시 `ops_watchdog.py:929~933` | 쓰임·전진(whale_duel1) | |
| 25 | `supervisor_xyz100` → `live_xyz100_collector_20261014.py`(185) | `data/hot/hl_xyz.sqlite` | `/api/xyz100`(app.js:3701) · 감시 `ops_watchdog.py:953` | 쓰임 | |

그 밖 상주:

| 프로그램 | 출력·읽는 곳 | 판정 | 근거 |
|---|---|---|---|
| 대시보드 서버(`dashboard/scripts/supervise_server.sh` → `server.py`) · 공개 터널 | 본체 | 쓰임 | 셸(R7) |
| `run_deribit_block_trade_collector_server.sh` → `live_deribit_block_trade_collector_20260928.py`(680) | `deribit_block_trades_state.json`·`deribit_gex_state.json`·`cboe`/옵션 duckdb | 서버 옵션 카드(server.py:386~391) · 감시 `ops_watchdog.py:220`(`check_gex_state`) | 쓰임 | |
| `ops-watchdog.service` → `scripts/ops_watchdog.py`(1,033줄 **파이썬**) | 텔레그램·`health_snapshot.json` | 쓰임 | ⚠R7 은 «감시자는 셸 그대로»라 했으나 이 파일과 `scripts/ops/*.py` 약 1,900줄은 **파이썬** — 범위에서 빠뜨리면 안 됨 |
| `prometheus-exporter.service` → `scripts/ops/prometheus_exporter.py`(118) | `:9101/metrics` | **불확실** | 읽는 쪽은 `docker/monitoring/prometheus/prometheus.yml`(저장소 안 정의뿐). 실제로 Grafana 를 보는지 모름 → §5 질문 8 |

### 2-B. 예약(cron) — 사용자가 읽어 준 목록 + 스크립트 머리말에서 찾은 것

| # | 항목(스크립트) | 주기 | 출력 | 읽는 곳 | 판정 | 근거 |
|---|---|---|---|---|---|---|
| 1 | `live_risk_radar_20261115.py`(317) | 매분 | `risk_radar_latest.json` · `risk_radar_forward.jsonl` | `dashboard/risk_radar.py` → `/api/risk-radar`(app.js:5713). `forward.jsonl` 은 읽는 곳 0 | 쓰임 (forward.jsonl 은 §5 질문 9) | 머리말 `:9~11` |
| 2 | `live_treasury_futures_collector_20261006.py`(93) | 매분(5일 덮어쓰기)+매시 | `treasury_futures.sqlite` | `/api/treasury-futures` · 감시 `ops_watchdog.py:942` · lake | 쓰임 | |
| 3 | `scripts/ops/raw_failover_20261107.py`(287) | 매분 | `raw_failover.json` | 감시 `check_raw_failover` | 쓰임 | |
| 4 | `live_cboe_option_chain_collector_20261101.py`(106) | */10 | `cboe_option_state.json` | `/api/gex`(server.py:387) · 전진 soxl_own_iv(2027-02-01) | 쓰임·전진 | |
| 5 | `live_macro_5m_collector_20261028.py`(123) | */5 | `macro_5m.sqlite` | Macro Trend·`/api/macro-today` · 감시 `ops_watchdog.py:947` | 쓰임 | |
| 6 | `live_macro_daily_collector_20261028.py`(181) | 매시 | `macro_daily.sqlite` | `dashboard/macro_trend.py` · 전진 macro_bias | 쓰임 | |
| 7 | `live_regime_band_cron_20261102.py`(66)+`_recorder`(91)+`live_regime_band_20261102.py`(71) | 매시 | `regime_band_forward*.jsonl` | `dashboard/regime_band.py` · `pred_record.py` | 쓰임·전진(fwd_band 2027-01-08) | |
| 8 | `live_liqmap_scorer_20261008.py`(145) | 매시 07분 | `liqmap_score.jsonl` · `liqmap_score_status.json` | 알림 센터 server.py:353~358 · 판정 | 쓰임·전진(liqmap_fwd 2026-12-07) | |
| 9 | `live_liq_tab_20261107.py`(440) | 매시 | `liq_tab_<코인>.json` | `/api/liq-tab`(liq_tab.js:39) | 쓰임 | 머리말 `:4` |
| 10 | `scripts/ops/data_quality_monitor_20261022.py`(390) | 매시 | 감시 보고 | 운영 | 쓰임 | |
| 11 | `live_eth_direction_forward_recorder_20261010.py`(95) | 매일 00:31 UTC | `eth_direction_forward.jsonl` | `dashboard/dir_signals.py` 언급 · `verdict_ledger_health` FRESH(12h) | 쓰임·전진(dir4h_p2 2027-03-14 · dir3asset 2027-04-08) | |
| 12 | `live_eth_maxpain_oos_ledger_20261007.py`(244) | `1-9 2,7`·`15,45 8` | `maxpain_oos_ledger.jsonl` | 판정 코드 `verdict_eth_maxpain_oos_20261118.py` — **판정이 11-18 에 «불통과·매매·전진 없음»으로 닫힘**(`verdict_calendar.json` maxpain_oos) | **안 쓰임** | ⚠`scripts/ops/verdict_ledger_health_20261107.py:19` 의 FRESH 표에 `maxpain_oos_ledger.jsonl: 30` 이 있어 기록기를 끄면 «기록 멈춤» 경보 → 그 줄을 같이 지운다 |
| 13 | `seal_to_lake.py`(299) | 매일 02:05 UTC | `data/lake/**` 일별 Parquet | 전진(whale_duel1·trend4_1 등) · 감시 `check_seal` | 쓰임·전진 | |
| 14 | `scripts/ops/backup_live_data.sh` | 04:00 | 백업 | 감시 `check_backup` | 쓰임(셸) | |
| 15 | `scripts/ops/morning_audit_20261012.py`(291) | 매일 | 아침 감사 | 사용자 · `check_host_disks` 소진 속도 입력 | 쓰임 | |
| 16 | `live_eth_dir4h_20261015.py --train`(464) | 월간 | `*_dir4h_model.joblib` | 서버 `/api/dir4h` · `verdict_ledger_health.check_models` | 쓰임 | |
| 17 | `live_vol4h_qx4_20261025.py`(335) | 월간 | `vol4h_qx4.joblib` | `/api/vol-forecast` · 전진 fwd_vol(2026-12-23) | 쓰임·전진 | |
| 18 | `live_eth_dir_short_20261028.py`(334) | 월간 | `*_dirshort*_model.joblib` | `/api/dir-short` · 전진 fwd_dirshort | 쓰임·전진 | |
| 19 | `fit_box_hold_platt90_20261026.py`(144) | 월 1일 | Platt90 계수 | `dashboard/box_hold.py` | 쓰임 | 머리말 `:4` |
| 20 | `live_soxl_weekend_forward_recorder_20261022.py`(155) | 매일 21:30 UTC | `soxl_weekend_forward*.jsonl` | 전진 soxl_wknd1(2027-04-20)·soxl_wknd2 | 쓰임·전진 | |
| 21 | `live_trend_core_forward_20261114.py`(170) | 매일 09:30·12:30 KST | `trend_core_forward.jsonl` | 읽는 코드 0 · 문서 `docs/operating_logic_20261114.md` 만. 판정 달력의 fwd_trend90/180 과의 연결은 코드에서 확인 못 함 | **불확실** | §5 질문 9 |
| 22 | `live_macro_bias_forward_20261028.py`(182) | 토·일 10:30 KST | `macro_bias_forward.jsonl` | 전진 fwd_macro_regime(2028-10-28) · `verdict_ledger_health` FRESH | 쓰임·전진 | |
| 23 | `run_f4c_altdata_collector.sh` → `.py`(157) | 매일 | `altdata.duckdb`(공포탐욕·펀딩 스프레드) | 감시 `ops_watchdog.py:921~922` 뿐. 모델 읽기는 은퇴한 봇 시대 문서 | **불확실** | §5 질문 10 |
| 24 | `live_llm_brief_20261017.py`(162) | cron 여부 불명(저장소에 등록 흔적 0) | 요약 JSON(`--out`) | 읽는 코드 0 · `docs/xrp_dashboard_gap_audit_20261031.md:44` «대시보드에 안 붙어 있음» | **불확실** | §5 질문 11 |
| 25 | `maker_fill_shadow_worker.py`(318) | 감독·cron 등록 흔적 0 | `maker_fill_shadow.duckdb` | `scripts/rl_1s_agent.py`(RL 축, 종결)와 문서뿐 | **안 쓰임** | grep `maker_fill_shadow`: 코드 2곳 + docs |
| 26 | `scripts/ops/multicoin_collectors_20260926.py`(514) | cron | `multicoin_collectors.json` | 감시 `ops_watchdog.py:439` | 쓰임 | 매니페스트에 `bo`(#3)가 들어 있음 |
| 27 | `scripts/ops/deploy_watcher.sh` | */10 | 배포 | — | 쓰임(셸) | |
| 28a | `gen_data_map_20261107.py`(매일 11:30) | | `data_map.md` | 사용자 | 쓰임 | |
| 28b | `backup_restore_check_20261107.py`(월 1) | | 복원 시험 | 감시 | 쓰임 | |
| 28c | `verdict_ledger_health_20261107.py` | | 판정 달력 건강 | 사용자 | 쓰임 | 표본: 달력의 `how` 스크립트 존재 · 모델 월 갱신 · 기록 신선도 |

**표 2 요약**: 상주 25개 = 쓰임 19 · 안 쓰임 2(#3 돌파·#21 신호 워커) · 불확실 4(#8·#12·#15·#18) / 예약 30행 = 쓰임 25 · 안 쓰임 2(#12 maxpain·#25 maker) · 불확실 3(#21·#23·#24).

## 3. 표 3 — 대시보드 파이썬 모듈 (`dashboard/*.py` 21개, 합 약 13,480줄)

호출자는 `import`/`from dashboard import` 를 grep(server.py 는 위 «한계» 때문에 Read 로 확인한 import 블록 server.py:134~177 사용).

| 모듈 | 줄 | 하는 일 | 누가 부르나 | 살아 있는 화면 칸·기능 | 판정 |
|---|---|---|---|---|---|
| `server.py` | 5,173 | 라우트 81개·SSE·스레드풀·미시 루프·모든 캐시 | 진입점 | 전부 | 쓰임 · **일부 죽음**: ① `api_state`(표 1) ② `api_breakout_detector`+`BREAKOUT_*` 상수+`load_breakout_detector`(server.py:378~382·4478~4483) ③ `api_liquidation_5m_signal` 계산 경로 ④ import 5개가 «상수 하나 때문에 통째»: `live_push_notifier`(`PUSH_KINDS`만, 337줄 import) · `ops_watchdog`(`send_telegram`만, 1,033줄) · `collect_deribit_option_gex`(`SPECS`만, 588줄) · `live_evidence_signal_dashboard`(`FETCH_LIMIT` 상수와 chart_markers 가 쓰는 `compute_signals`, 895줄). 이전 시 상수만 옮기면 줄어든다 |
| `orders.py` | 3,072 | 계좌 조회·수동 진입/청산·손절·익절 감시·사용자 스트림 | server `orders.register`(server.py:4944) | 주문 전부 | 쓰임 · `api_manual_exit_all_submit`(2209~2268, 60줄)은 표 1 불확실 |
| `coin_flow.py` | 1,045 | 코인별 흐름 엔진(풋프린트·1초 수급·OI·OKX·현물) | server(server.py:430) · orders | 풋프린트·수급·OI | 쓰임 |
| `footprint_daily.py` | 654 | 일봉·4시간봉 풋프린트 | server `fpd` | fp_daily.js | 쓰임 |
| `flow_read.py` | 603 | «데이터끼리 무슨 말을 하나» 문장 | server `fr` | «지금 읽기» | 쓰임 · 함수 `vol_size_mult` 는 모듈 밖·시험 어디서도 호출 못 찾음(server.py 호출은 확인 불가) |
| `macro_trend.py` | 434 | Macro Trend 카드 | server · 수집기 2개 | `/api/macro-trend` | 쓰임 |
| `market_ctx.py` | 414 | 시장 맥락 카드 계산 | server `mctx` · `llm_brief.py` | `/api/market-context` | 쓰임 |
| `micro_ref.py` | 331 | 미시 참고 계산(QI×OFI·분위·경보·폭발→OI) | server `mref` | 미시 루프 → 상황 카드·`/api/stream` | 쓰임 · 5개 표시 중 화면에 남은 칸은 미조사 |
| `box_hold.py` | 262 | 박스 유지 확률(HGB·Platt90·QX4I) | server | `/api/box-hold` | 쓰임 |
| `llm_brief.py` | 256 | LLM 해석 입력 요약 | **`scripts/live_llm_brief_20261017.py` 만**(+연구) · 서버는 import 안 함 | 없음(`xrp_dashboard_gap_audit`: «대시보드에 안 붙어 있음») | **불확실**(§5 질문 11) |
| `pred_record.py` | 253 | 적중 기록·4h 상대 강도 | server · `forward_display_verdicts_20261107.py` | ⑤ 예측 줄 아래 | 쓰임·전진 |
| `position_profile.py` | 204 | 포지션 프로파일 | server `pprof` | «포지션» 보기 | 쓰임 |
| `dir_signals.py` | 199 | 지평 사다리 방향 신호·4h 변동성·개장 되돌림 | server | `/api/dir-signals` 등 | 쓰임 |
| `situation.py` | 107 | 30분 카드 입력 근거(옛 시나리오 로직 걷어낸 뒤 남은 것) | server `sit` | 상황 카드 | 쓰임 |
| `acct_notes.py` | 85 | 계좌 카드 날짜별 메모 | server | 계좌 카드 | 쓰임 |
| `trend_rule.py` | 81 | 일 단위 추세 규칙 | server · orders | `/api/trend` | 쓰임 |
| `sr_lean.py` | 77 | `sr_lean` jsonl 꼬리 읽기 | server | `/api/sr-lean` | 쓰임 |
| `regime_band.py` | 65 | 횡보/추세 마지막 행 읽기 | server | `/api/regime-band` | 쓰임 |
| `worker_state.py` | 59 | 워커 상태 신선도 | server · orders | 여러 칸 | 쓰임 |
| `swing_lines.py` | 57 | 큰 스윙 선 | server | `/api/swing-lines` | 쓰임 |
| `risk_radar.py` | 45 | 위험 레이더 파일 읽기 | server | `/api/risk-radar` | 쓰임 |

**표 3 요약**: 쓰임 20 · 안 쓰임 0(모듈 통째) · 불확실 1(`llm_brief.py`). 함수·덩어리 단위로 죽은 것은 server.py 행에 적은 4가지 + `flow_read.vol_size_mult`(미확인). 함수 단위 자동 검사는 server.py 를 읽을 수 없어 **다른 20개 모듈에서 `모듈 밖·시험 밖 참조 0` 인 함수만** 뽑았고 결과는 `flow_read.vol_size_mult` 하나(그 외 `footprint_daily.*_payload` 4개는 server 가 호출할 것으로 보임).

### 3-1. 라이브 경로가 import 하는 `scripts/*` (범위에 들어가야 함 — 연구 스크립트가 아님)

server.py:53~177 · orders.py 가 가져오는 것: `live_trade_tape_collector`(1,207) · `data_store`(155) · `live_binance_ctx_collector`(224) · `live_evidence_signal_dashboard`(895) · `live_eth_chart_markers`(137) · `live_liquidation_5m_signal`(222) · `live_liquidation_map`(725) · `live_push_notifier`(337) · `ops_watchdog`(1,033) · `live_macro_calendar`(665) · `coin_config`(127) · `collect_deribit_option_gex`(588) · `binance_ban_guard`(105) · `push_webpush`(318) · `live_orderflow_raster_collector`(898) · `live_eth_dir4h`(464) · `live_eth_dir_short`(334) · `live_vol4h_qx4`(335) · `live_eth_sizing_vol_model`(358) · `live_manual_peg_entry`(1,076) · `live_binance_account`(530) · `live_eth_risk_sizing_policy`(177) · `live_eth_trade_plan`(827) · `live_eth_mae_quantile_model`(209) · `live/execution/live_manual_peg_execute_20260912.py`(1,067). 합 약 13,500줄.
⚠`live_eth_chart_markers` 는 연구 성격의 `build_eth_anchor_label_dataset_20260907.py`·`live_eth_extreme_detector_20260909.py`(304)를 import 한다(chart-markers 가 «불확실»인 이유와 같은 덩어리, §5 질문 3).
라이브 경로에서 import 하는 곳을 못 찾은(= 안 쓰임) 것: `live_liquidation_direction_signal_20260825.py`(197) — grep `liquidation_direction`: 코드 import 0, 주석 2곳 · `live_eth_vol_forecast_20260910.py`(168) — import 는 연구용 `exit_synth_features_20260910.py` 뿐.

## 4. 표 4 — 화면 코드 (`dashboard/live/*`: app.js 13,935 · fp_daily.js 1,200 · liq_tab.js 612 · bubble_map.js 372 · liq_profile.js 94 · sw.js 74 · index.html 648 · styles.css 4,133)

### 4-A. 실행될 수 없거나 값이 안 쓰이는 덩어리

방법: ① `function 이름` / `const 이름 = (…) =>` 정의마다 5개 파일+index.html 에서 같은 단어 출현 횟수 세기(1회 = 정의뿐) ② `el("id")`·`getElementById`·`querySelector("#id")` 144개가 index.html·JS 생성 코드에 존재하는지 ③ 주석·상수 스위치(`= false`) 찾기.

| 덩어리 | 위치(줄 범위·줄 수) | 근거 | 판정 |
|---|---|---|---|
| `ltLanePlace` | app.js:803~824 · 22줄 | 정의뿐(출현 1) | 안 쓰임 |
| `fmtTs` | app.js:2079~2089 · 11줄 | 정의뿐(출현 1) | 안 쓰임 |
| `balanceAssetNote` | app.js:2314~2323 · 10줄 | 정의뿐(출현 1) | 안 쓰임 |
| `BOX_ON_CHART = false` 가 끄는 차트 블록(«미래 칸 상자»·E2 «가격 경로»·옵션 부채꼴) | 상수 app.js:5539 · 쓰는 곳 app.js:10120~약 10303(**끝줄은 훑어 본 값, ±10줄 불확실**) · 약 184줄 | 주석 «셋 다 끔(되살리려면 true)» · `BOX_ON_CHART` 참조 4곳(10126·10130·10182) 모두 `&&` 로 막힘 | 안 쓰임 |
| `render()` 안 `const micro`·`tail`(app.js:11526) | 1줄 | 이후 사용처 못 찾음(`state.microstructure`·`state.tail_risk` 읽기 0) | 안 쓰임(소) |
| `refreshLiquidation5mSignal` 의 신호 호출(응답 `j` 버림) | app.js:3249~3252 | 표 1 | 안 쓰임(값) |
| 차트 증거신호 레인·이벤트 삼각형(`latestChartMarkers`) | app.js:10026~10100 근처 · 10885~10900 · `refreshChartMarkers` 3158~3172 | `cm.available` 이 서버 의존 | **불확실** |
| 주석 «`/api/position-sizing` 은 아직 살아 있다»(app.js:12124) | 주석 | 서버에 그 라우트 없음(표 1) — 주석만 낡음 | 코드 아님 |

보조 관찰: 폴백 전용 코드(스트림이 죽었을 때만): `refreshSupply1s`(app.js:4611~4617) · `refreshSituation`(5361~5365) — 옮길지는 «폴백을 남길 것인가» 결정.
외부 의존: `index.html:29·30`(Google Fonts · jsDelivr Pretendard) · 한글 글꼴. `app.js:1~9` 의 `API_*_URL` 은 전부 생존.

### 4-B. 데스크톱 / 휴대폰 분기 크기(대략)

- CSS: `@media` 블록 116개. 중괄호 깊이로 센 줄 수 — 휴대폰 쪽(`max-width`·`pointer: coarse`·`hover: none`·`orientation`) **약 646줄**, 넓은 화면(`min-width`) 약 57줄, 그 밖(모션 줄이기 등) 약 28줄 / 전체 4,133줄. (`pointer: coarse` 는 태블릿도 포함 — 정확 분리는 못 함.)
- JS: 휴대폰 분기 변수(`isMobileChartMode`·`matchMedia("(max-width: 720px)")`·`FPM`·`mob` 등)가 나오는 줄 약 52줄, `휴대폰|모바일|mobile|phone` 단어가 든 줄 151개 — 분기는 대부분 «한 줄짜리 조건»이라 실제로 분리되는 함수 덩어리는 작다(큰 덩어리: `renderCandleSvg` 8382~약 11520 안의 `FPW`/`FPM` 전체화면 분기 — 지난 11-20 휴대폰 가로 전체화면 작업).
- 따라서 «데스크톱만 옮기고 휴대폰을 뒤로 미루기»는 CSS 에서는 가능(약 16%)하지만 JS 에서는 분기가 흩어져 있어 이득이 작다.

## 5. 끝에

### 5-1. 옮기지 않아도 되는 것 (판정 «안 쓰임»만)

| 항목 | 줄 | 비고 |
|---|---|---|
| `/api/state` 핸들러 | (server.py 줄 미확인) | 시험 2개만 호출 → 시험도 같이 정리 |
| `/api/breakout-detector` + `BREAKOUT_*` + `load_breakout_detector` | (server.py 줄 미확인, 대략 수십) | |
| 돌파 워커 `live_eth_breakout_detector_worker` + `live_eth_breakout_detector` | 98 + 181 = **279** | 감시 매니페스트 줄도 같이 |
| 신호 워커 `live_signal_worker` + `live_eth_sweep_v_rebound_signal` | 172 + 610 = **782** | |
| `maker_fill_shadow_worker.py` | **318** | |
| `live_eth_maxpain_oos_ledger_20261007.py` | **244** | 판정 닫힘(11-18) · `verdict_ledger_health` FRESH 줄 동반 삭제 |
| `live_liquidation_direction_signal_20260825.py` | **197** | import 0 |
| `live_eth_vol_forecast_20260910.py` | **168** | 라이브 import 0 |
| `compute_liquidation_5m_signal`(`live_liquidation_5m_signal_20260825.py:144~222`) | **79** | 단 호출 성공 의존(표 1)을 먼저 끊을 것 |
| app.js 죽은 함수 3개 | 22+11+10 = **43** | |
| app.js `BOX_ON_CHART` 꺼진 블록 | 약 **184** | 끝줄 ±10 |
| **합계(셀 수 있는 것만)** | 파이썬 약 **2,067줄** + 자바스크립트 약 **227줄** = **약 2,294줄** | 전체(파이썬 실행 경로 약 33,300줄 · JS 약 16,300줄)의 약 6% · 1.4% |

(시험 파일·docs 는 줄 수에 안 넣었다. 셀 수 없는 server.py 쪽 핸들러·상수는 위 표에서 «줄 미확인».)

### 5-2. 불확실 — 사용자에게 물어야 확정되는 질문

1. **휴대폰 앱 계약**: 현재 앱(웹뷰)이 `/api/state` · `/api/breakout-detector` · `/api/liquidation-5m-signal` · `/api/manual-exit/submit-all` 같은 «화면 호출 0건» 경로를 직접 부르는 곳이 있나? (`mobile/` 이 이 워크트리에 없어 확인 불가. 없다면 표 1 의 «안 쓰임» 3개가 확정.)
2. **`/api/manual-exit/submit-all`(전체 청산, 실주문 60줄)**: 화면 단추는 이미 삭제됐는데 앱·손 명령으로 쓰나? 안 쓰면 옮기지 않는다.
3. **차트의 «증거신호 레인»과 ▲▼ 이벤트 삼각형**(`/api/chart-markers` + `live_eth_chart_markers`·`live_eth_extreme_detector`·`live_evidence_signal_dashboard`(895줄)·`build_eth_anchor_label_dataset`): 지금 ETH 차트에 실제로 보이나? 안 보이면 이 덩어리 약 1,500줄을 안 옮긴다.
4. **Hyperliquid 원시 호가(BBO) 수집기**(`live_hyperliquid_book_ticker_collector`, 298줄 · ETH 원시 파일): 호가 전진 검정은 중단됐고 읽는 코드가 0인데, 연구용 원시 보존으로 계속 쌓을 건가? (`HL_BT_CTX_ONLY` 변형은 맥락용이라 별개)
5. **`microstructure.duckdb`(ETH·BTC·SOL) 와 OI·롱숏비 duckdb**(루트 `microstructure_scanner.py` 884 + `oi_lsratio_collector.py` 301): 화면·라이브가 안 읽고 lake 봉인·감시만 읽는다. 연구 자료로 계속 쌓을 건가? 시장 맥락의 «롱숏비» 줄은 어디 값을 읽나(server.py 구간 미조사)?
6. **`tail_risk.duckdb` 폴백**(server.py:4431~4436): ETH 가 아닌 코인(주식 포함)의 «청산 원»을 지금 이 길로 받나? 아니면 `tail_risk_interceptor`(543) 의 DB 쓰기를 안 옮겨도 되나(청산 급증 배지용 `liq_burst_state.json` 은 별개로 필요)?
7. **뉴스 판정(`live_news_judge`) + 로컬 LLM(`supervisor_ollaya`)**: 2026-10-27 부터 의도적 정지. v24 학습 뒤 재개할 건가, 접을 건가? 접으면 218줄 + 셸을 안 옮긴다.
8. **Prometheus/Grafana(`prometheus_exporter.py` 118 + `docker/monitoring`)**: 실제로 보고 있나?
9. **전진 기록 jsonl 두 개**(`risk_radar_forward.jsonl` ← 매분 크론, `trend_core_forward.jsonl` ← `live_trend_core_forward` 170줄): 읽는 코드가 없고 판정 달력 항목과의 연결을 코드로 못 찾았다. 어느 사전등록 판정(예: fwd_trend90·fwd_trend180)에 쓰이나?
10. **`run_f4c_altdata_collector`(공포탐욕·펀딩 스프레드 duckdb)**: 감시만 보고 읽는 모델이 없다(봇 은퇴). 연구용 누적을 계속할 건가?
11. **`live_llm_brief_20261017.py`(162) + `dashboard/llm_brief.py`(256)**: 손으로 돌리는 도구인가, 완전히 접었나?
12. **폴백 경로 보존**(`/api/supply-1s`·`/api/situation` 폴링): 새 코어에서 SSE 가 죽을 때 폴링 폴백을 유지할 건가? (설계 결정)
13. **`/api/client-timing`**: M0 측정이 끝나면 접나? (측정 기간 결정)

### 5-3. 꺼도 되는 것 (지금 파이썬 쪽에서 돌고 있는데 안 쓰이는 것)과 끄면 영향받는 감시·알림

| 끄는 것 | 영향받는 감시·알림 | 같이 할 일 |
|---|---|---|
| 돌파 워커(`supervisor_breakout_detector`) | `scripts/ops/multicoin_collectors_20260926.py:129~130` 매니페스트가 `{eth,sol,xrp}_breakout_detector_state.json` 신선도를 15/30분 기준으로 본다 → **끄는 순간 경보** | 매니페스트의 `bo` 줄을 먼저 지움 |
| 신호 워커(`supervisor_signal_worker` · v_rebound) | 감시 항목은 못 찾음(`ops_watchdog.py` 에 `v_rebound` 없음). 지금 서버에서 돌고 있는지는 crontab 미확인 | 돌고 있으면 끄기 |
| `live_eth_maxpain_oos_ledger` cron | **`verdict_ledger_health_20261107.py:19` FRESH `maxpain_oos_ledger.jsonl: 30h`** → «기록 멈춤» | FRESH 항목 삭제 후 끄기 · 판정 달력의 `maxpain_oos`는 이미 닫힘 |
| `maker_fill_shadow_worker` | 감시 항목 못 찾음 | 돌고 있으면 끄기 |
| HL 원시 BBO(불확실 #4) | `ops_watchdog.py:961 eth_hl_bbo` 5/15분 · `raw_failover` · 매니페스트 | 끄려면 세 곳을 함께 |
| `/api/state` · `/api/breakout-detector` 라우트 | 시험 | 시험도 정리 |
| 뉴스 판정·ollaya(이미 정지) | — | 재개 여부 결정(질문 7) |

**끄면 안 되는 것(안 쓰임처럼 보여도)**: `/api/footprint`(감시자 `check_footprint_history` 가 호출 `ops_watchdog.py:197`) · `/api/binance-account`·`/api/market-context`(푸시 알림 데몬 입력) · `/api/sr-watch`(알림 워커) · `dashboard_state.json` 을 쓰는 `duckdb_persist_worker`(감시 `ops_watchdog.py:162` 가 2/5분 기준으로 읽고 화면 `render()` 가 읽음).

### 5-4. 이번 조사에서 새로 알게 된 것(범위에 영향)

- R7 이 «감시자·배포 워처·백업은 셸 그대로»라 했지만 `scripts/ops_watchdog.py`(1,033) · `scripts/ops/*.py`(raw_failover 287 · morning_audit 291 · data_quality_monitor 390 · multicoin 514 · backup_restore_check 179 · prometheus_exporter 118 · gen_data_map · verdict_ledger_health · remote_fresh_probe) 약 2,900줄은 **파이썬**이다. 이전 범위에 넣을지 결정이 필요하다.
- 서버가 «상수 하나» 때문에 큰 모듈을 통째로 import 한다(표 3 server.py 행 ④) — 이전 때 상수만 옮기면 약 2,800줄이 의존에서 빠진다(단 그 모듈들 자체는 다른 이유로 쓰이니 «안 옮김»은 아님).
- 화면 호출 0건이어도 **화면 밖 소비자**가 있는 라우트: `/api/sr-watch` · `/api/footprint`(감시) · `/api/binance-account`(푸시·아침 감사) · `/api/market-context`(푸시).
- `liquidation-5m-signal` 은 값이 버려지지만 실패하면 같은 `try` 안의 `5m-history` 호출까지 막는다(app.js:3249~3257).

## 6. 못 읽은 범위

- **`dashboard/server.py`**: grep 막힘 → Read 로 읽은 곳만: 1~615 · 4340~4680 · 4860~5173. **못 읽음**: 616~4339 · 4680~4860(대부분의 핸들러 본문·swr·미시 루프·SSE 본문 — 지연 지도가 읽은 구간은 그 문서를 인용). 표 1 의 «server.py 줄» 은 위 구간만 직접 확인.
- **`dashboard/orders.py`**(3,072): grep 만(2209·2269·2301 확인). 본문 미조사.
- **`app.js`**: grep + 선택 구간(1~10 · 3155~3200 · 3240~3275 · 5383~5400 · 10118~10127 · 10238~10262 · 10883~10893 · 11515~11615 · 12525~12540). 칸별 생존 렌더러 전수 검사 아님(죽은 함수는 «출현 1회»로만 검출).
- **`styles.css`**: `@media` 줄 수 센 것뿐. 안 쓰는 선택자 조사 안 함. **`index.html`**: id 존재 검사만(요소 단위 전수 안 봄).
- **함수 단위 죽은 코드**: server.py · orders.py 는 불가, 나머지 20개 모듈은 «모듈 밖·시험 밖 참조 0» 규칙으로만 검사. 큰 덩어리 눈대중 외 정밀 아님.
- **scripts**: 스크립트 머리말·grep 만. 각 수집기의 쓰기 주기·내부 로직 미조사. 서버 crontab 실물 미확인(사용자 구두 목록 + 머리말).
- **`docs/experiments/*` 의 사전등록 문서 본문**은 열지 않았다. 전진 검정 ↔ 기록기 연결은 `verdict_calendar.json` 항목 문구와 스크립트 머리말·grep 로만 판단(예: whale_duel 은 prereg 문서 68행 한 줄 확인).
- **`mobile/`**: 이 워크트리에 없음 → 앱 계약 미확인.
- 그림자·성능은 다루지 않았다(읽기 전용 범위표).
