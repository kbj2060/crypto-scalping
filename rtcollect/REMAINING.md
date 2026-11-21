# rtcollect 로 옮길 것 전수 표 (2026-11-21 · 범위 문서 docs/rtcore_port_scope_20261120.md 표 2 의 «쓰임»·«전진» 전부)

판정 «안 쓰임»(돌파 워커 · 신호 워커 · maxpain 기록기 · maker_fill_shadow)과 «불확실»(HL 호가 .bt · 뉴스 판정·ollaya · prometheus · trend_core_forward ·
f4c altdata · llm_brief)은 빼고 끝에 따로 적었다. 모델을 서빙하는 것은 «rtmodel» · 감시·알림·배포는 «범위 밖(R7·알림)».

### 상주 (scripts/ops/supervisor_*.sh)

| 프로그램 | 저장 형식 | 읽는 곳 | 전진 검정 | rtcollect | 막는 것 · 다음 |
|---|---|---|---|---|---|
| 바이낸스 체결 테이프 `live_trade_tape_collector` | SQLite hot | server 풋프린트·수급 · seal → 고래 맞대결 | whale_duel1 | **옮김** `binance_tape`(+REST 대조·복구) | — |
| OKX 체결 테이프 | SQLite hot | server · seal | whale_duel1 | **옮김** `okx_tape`(+REST) | — |
| Bybit 체결 테이프 + 맥락 | SQLite hot 둘 | 시장 맥락 OI · seal | — | **옮김** `bybit_tape`(+kline 대조) | — |
| Bybit 청산 | SQLite hot | server 청산 원 · liqmap_scorer | liqmap_fwd | **옮김** `bybit_liq` | — |
| OKX 맥락 | SQLite hot | server · coin_flow · liq_tab · liqmap_scorer | liqmap_fwd | **옮김** `okx_ctx` | — |
| 바이낸스 맥락 | SQLite hot | server · liq_tab · liqmap_scorer · dqm · 감시 | liqmap_fwd | **옮김** `binance_ctx`(OI REST 기본 꺼짐) | — |
| xyz100 | SQLite hot | /api/xyz100 · 감시 · seal | — | **옮김** `xyz` | — |
| HL 자산 맥락(book ticker CTX_ONLY) | SQLite hot | server 시장 맥락 · seal · dqm | — | **옮김** `hl_ctx` | — |
| HL 호가 `.bt`(book ticker 전체 · Pi) | 고정폭 `.bt` 시각 파일 → gz | Pi → 서버 전송 · 연구 | — | **옮김** `hl_bt`(rtcollect) | `.bt` 바이트 일치 · supervisor_rtcollect.sh 에 종류 추가 필요 |
| depth_diff | JSONL → 시각별 gz | ob5m · iceberg · bookdepth_feat | fwd_dirshort(간접) | **옮김** `depth`(스냅숏 REST 기본 꺼짐 · 전환 때 켬) | — |
| 호가 래스터 | 고정폭 `.f32`(+14일 뒤 parquet·duckdb 보관) | server read_window → 히트맵·시장 맥락 | — | **옮김** `raster`(.f32 쓰기만 · 스냅숏 REST 필수) | 보관(14일 뒤 parquet+duckdb 뷰) = **옮김** `rtduck raster_archive`(예약 · parquet 확장 켬 2026-11-21) -- 원본이 쓴 parquet 17개를 Rust 가 같은 행으로 · 서버 판 파이썬(duckdb 1.5.1)이 Rust 보관본 20개·뷰를 같은 값으로 읽음 · 지우기는 되읽은 행 수가 맞은 뒤에만(의도한 차이 #20) |
| ob5m 실시간 워커 `live_ob5m` | SQLite `data/live/ob5m/<코인>.sqlite` | /api/ob5m(화면) | — | **옮김** `ob5m`(기본 쓰기 자리 data/live/rtcollect/ob5m · 원본 자리는 RTCOLLECT_TAKEOVER=1) | 합성 14시간 depthdiff·테이프 → 봉·사건 벽·live 칸 7,417 비트 일치 · /api/ob5m 읽기 코드 교차 확인 |
| 호가 깊이 10열 `live_bookdepth_feat` | SQLite `data/live/bookdepth/ethusdt.sqlite` | live_eth_dir_short 피쳐(→ rtmodel) | fwd_dirshort | **옮김** `bookdepth` | 칸 2,110 비트 · 930 은 1e-12 안(최대 3.6e-15 = 원본 `arr @ prices` 가 BLAS ddot · 서버 OpenBLAS·개발기 MKL) · dir_short.book_row float32 161/161 같음 |
| 아이스버그 `live_eth_iceberg` | JSON 상태 + JSONL 사건 | /api/iceberg · 감시 | — | **옮김** `iceberg`(ETH·XRP · depthdiff·hot 테이프 읽기만) | 사건 글자·2초 state 3,582개 글자 일치 |
| HL 포지션 | SQLite hot `hl_positions*.sqlite` | server HL 고래 청산 · 시장 맥락 · seal | — | **옮김** `rtduck hl_positions`(HL info REST 전용 -- `RTDUCK_REST=1` 없으면 안 돈다) | 입력 = HL 체결 DuckDB 를 읽어서 rtduck 에 둠 |
| HL 체결 | DuckDB 일별 → 3일 뒤 gz | HL 포지션 입력 · 감시 | — | **옮김** `rtduck hl_trades` | — |
| market_data_worker `duckdb_persist_worker` | DuckDB 둘(microstructure 1분 ETH·BTC·SOL · tail_risk 1분 ETH) + JSON 둘(dashboard_state · liq_burst_state) | 화면 세션 칸·SSE 갱신 신호(dashboard_state) · 감시 check_dashboard·check_data_sources·신선도 4줄 · seal micro_1m·tail_risk_1m | — | **옮김** `rtduck market_data` = 미시구조(depth20·aggTrade 웹소켓 + OI·premiumIndex 10초 REST + aggTrades 대체 · 34칸) + 청산 요격기 + dashboard_state(장 달력 = pandas_market_calendars 를 뜬 표) · 미시구조 REST 기본 꺼짐 | 감시 check_data_sources 를 Rust 파일로 돌려 파이썬 파일과 같은 판정(OK) |
| oi_lsratio 워커 | DuckDB | **server 롱숏비(server.py:2536)** · seal | — | **옮김** `rtduck oi_lsratio`(REST 기본 꺼짐 · 밴 파일) | — |
| 뉴스 RSS | SQLite hot | /api/news | — | **옮김** `news`(quick-xml·regex · 조정자 결정 2026-11-21) | 원본 run() 90바퀴 정답지와 요청 68(머리글 포함)·기사 49행·polls 69행·스키마 일치 |
| deribit 블록체결·GEX | JSON 둘 + 옵션 DuckDB | server 옵션 카드 · 감시 | — | 안 옮김 | 옵션 DuckDB 몫(크레이트) · JSON 몫만 떼는 것은 결정 필요 |
| SR lean `live_sr_lean` | JSONL | /api/sr-lean | sr_ml_5m_forward | **rtmodel** | 모델 서빙 |
| 푸시 알림 · SR 근접 알림 | 웹푸시·텔레그램 | 폰 · 알림 센터 | — | 범위 밖(알림) | — |

### 예약(cron)

| 프로그램 | 주기 | 저장 형식 | 읽는 곳 | 전진 검정 | rtcollect | 막는 것 · 다음 |
|---|---|---|---|---|---|---|
| treasury_futures | 매분·매시 | SQLite | /api/treasury-futures · 감시 · lake | — | **옮김** `treasury_futures`(한 번 실행 · --range 1d/5d) | 🔴그림자에서 켜면 Yahoo 호출 두 배(429 위험) -- 기본 안 켬 |
| macro_5m | */5 | SQLite | Macro Trend · /api/macro-today · 감시 | — | **옮김** `macro_5m`(한 번 실행) | 〃 |
| macro_daily | 매시 | SQLite | macro_trend | macro_bias | 안 옮김 | pandas 계산 · 재무부 REST |
| cboe 옵션 체인 | */10 | 원본 gz + JSON | /api/gex | soxl_own_iv | **옮김** `cboe_chain` = 스냅숏·last_seqno + 요약기(`cboe_summary.rs` → cboe_option_state.json · 실현변동성 klines 는 `RTCOLLECT_REST=1` 일 때만) | state 글자 5개(두 종목·사흘·서머타임 끝나는 날 01:3x) · klines 요청 4 · rv 캐시 글자 일치 |
| liqmap_scorer | 매시 | JSONL·JSON | 알림 센터 · 판정 | liqmap_fwd | **옮김** `liqmap_scorer`(klines 는 `RTCOLLECT_REST=1` 일 때만) | 세 번 실행 새 줄 48 · 지도·실측 칸 글자 일치 · status 글자 · merge_asof 5초 경계·마크 구멍·NULL 마크·같은 ms |
| liq_tab | 매시 | JSON | /api/liq-tab | — | 안 옮김 | numpy·pandas 계산 + 바이낸스 klines REST |
| soxl_weekend_forward | 매일 | JSONL | 판정 | soxl_wknd1·2 | 안 옮김 | 바이낸스 REST + 규칙 계산 |
| macro_bias_forward | 주말 | JSONL | 판정 | fwd_macro_regime | 안 옮김 | pandas 계산 |
| risk_radar · regime_band · eth_direction_forward_recorder | 매분·매시·매일 | JSON·JSONL | 화면·판정 | regime: fwd_band · dir: dir4h_p2·dir3asset | **rtmodel** | 모델 예측 기록 |
| seal_to_lake | 매일 | Parquet | 전진 다수 · 감시 | 여럿 | **반쯤** `rtduck::seal`(DuckDB 원천 · 첫 표 binance/oi_lsratio_5m) | 원본도 DuckDB COPY 로 쓴다 · 🔴SQLite 원천은 sqlite 확장 없음 · hot 정리·seal_status 남음 · 실행 종류 아직 없음 |
| 월간 학습(dir4h·vol4h·dir_short·box_hold Platt) | 월간 | joblib | 서빙 | 여럿 | 파이썬 유지(학습) | — |
| raw_failover · data_quality_monitor · morning_audit · multicoin_collectors · gen_data_map · verdict_ledger_health · backup | 여러 | JSON·보고 | 감시·사용자 | — | 범위 밖(R7 운영) | — |

### pandas 계산 기록기 조사(2026-11-21 일곱째 조각 · liqmap_scorer 먼저 옮김)

| 기록기 | 계산 | 입력 · REST(원본 한 번 실행) | 기대는 모듈 | 전진 | 막는 것 · 다음 |
|---|---|---|---|---|---|
| liqmap_scorer(145줄) | 청산지도 4변형(numpy 원소별 · exp · 삽입 순서 dict) + merge_asof 실측 | 바이낸스 klines 1h×198 1회 · hot SQLite 셋(읽기만) | research_eth_liqmap_venue_leverage_validation(liq_map·realized) · live_liquidation_map(상수 셋) | liqmap_fwd | **옮김** `liqmap_scorer` -- 값 일치(아래 최종 표) |
| liq_tab(440줄) | 남은 매물 히트맵·first_touch·봉우리·최대 고통·탄력 + compute_tier_profile | 바이낸스 klines(코인당 시간 1회 · 첫 실행 몇 쪽) · hot SQLite(청산·OI 셋) · lake parquet(duckdb `read_parquet`) · 1h 봉 캐시 parquet | live_liquidation_map(compute_tier_profile) · coin_config | — | **옮김** `rtduck liq_tab` -- 두 번 실행 liq_tab_eth.json(175KB)·liq_tab_all.json 글자 일치 · 캐시 parquet 행·형 · 요청 URL(쪽 나눔·앞쪽 메움) |
| soxl_weekend_forward(155줄) | 주말 블록 가격·펀딩 합(표준 라이브러리만 -- statistics·zoneinfo) | 바이낸스 fapi 5분봉·펀딩(끝난 주말마다) | 없음 | soxl_wknd1·2 | **옮김** `rtcollect soxl_weekend`(`--record`·`--status` · 판정 `--look` 은 파이썬) -- 세 번 실행 새 줄 3·요청 32·중간 HTTP 500 때 남는 줄 일치(서머타임 끝나는 주말 · 펀딩 0건 = 정수 0 · lag) |
| macro_bias_forward(182줄) | 주 Z 점수(sample_week·feats_for·z_combo·z91 -- pandas rolling·rank) | FRED API(키) · Yahoo ^KS11 5년 · 바이낸스 현물 1d | research_macro_regime_bias(443줄 -- 연구 함수를 그대로 부른다) | fwd_macro_regime | **안 옮김(조정자 결정)** · 막는 것: ① 🔴FRED 키 자리 -- 원본은 `load_dotenv` 로 저장소 루트의 점-env 파일에서 `FRED_API_KEY` 를 읽는다(그 파일을 Rust 가 읽을지 · 환경변수만 받을지 · 다른 키 파일로 둘지 = 사용자 결정) ② 연구 모듈 443줄의 pandas rolling·rank·z 계산을 비트로 맞출 도구 ③ Yahoo 5년 일봉(그림자 ×2 · 429) |
| macro_daily(181줄) | 관측 append-only 병합 + 순유동성(netliq) | FRED(키) · 재무부 DTS · DefiLlama | dashboard/macro_trend(434줄 -- SOURCES 표·fetch_fred·fetch_stables·netliq 를 import) | macro_bias(입력) | **안 옮김(조정자 결정)** · 막는 것: ① 🔴FRED 키 자리 -- 원본은 `load_dotenv` 로 저장소 루트의 점-env 파일에서 `FRED_API_KEY` 를 읽는다(그 파일을 Rust 가 읽을지 · 환경변수만 받을지 · 다른 키 파일로 둘지 = 사용자 결정)(cron 이 그 파일을 안 읽어 스크립트가 스스로 읽는다) ② `dashboard/macro_trend` 의존 -- 계열 목록(SOURCES)·FRED 호출·netliq 식이 대시보드 모듈에 있어, 옮기면 rtserve(Macro Trend 카드)와 같은 표·식을 한 곳에 둘지 정해야 한다 ③ 계산 자체(관측이 바뀔 때만 행 추가)는 가볍다 |

## 최종 표 (2026-11-21 일곱째 조각 뒤)

«REST 배수» = 서버에서 그림자로 **켜면**(파이썬이 같은 일을 계속하는 동안) 같은 IP 에서 나가는 그 REST 가 몇 배가 되는가. ×1 = REST 없음(웹소켓·파일만).

| 묶음 | 프로그램 | 상태 | 크레이트 | 그림자에서 켜면 REST |
|---|---|---|---|---|
| 상주 | 바이낸스 체결 테이프 `binance_tape` | 옮김 | rtcollect | ×2 -- 기본 꺼짐(`RTCOLLECT_REST`) · 켜면 종목마다 klines 1/분+1/5분 + aggTrades 복구 |
| 상주 | OKX 체결 테이프 `okx_tape` | 옮김 | rtcollect | ×2 -- 기본 꺼짐 · 공백 없으면 종목당 약 1.2회/분(OKX 한도 20회/2초) |
| 상주 | Bybit 체결+맥락 `bybit_tape` | 옮김 | rtcollect | ×2 -- 기본 꺼짐 · 1회/분 |
| 상주 | Bybit 청산 `bybit_liq` | 옮김 | rtcollect | ×1 |
| 상주 | OKX 맥락 `okx_ctx` | 옮김 | rtcollect | ×2 -- 시작 때 ctVal 1회뿐 |
| 상주 | 바이낸스 맥락 `binance_ctx` | 옮김 | rtcollect | ×2 -- 🔴OI 약 540회/분 → 합쳐 1,080(IP 한도 45%) · 그림자 동안 끔 |
| 상주 | xyz100 `xyz` | 옮김 | rtcollect | ×2 -- 재접속 때만(4회/접속) |
| 상주 | HL 자산 맥락 `hl_ctx` · HL 호가 `.bt` `hl_bt` | 옮김 | rtcollect | ×1 |
| 상주 | depth_diff `depth` | 옮김 | rtcollect | ×2 -- 시간당 depth(가중치 20) 1회 · 전환 때 반드시 켬 |
| 상주 | 호가 래스터 `raster`(.f32) | 옮김 | rtcollect | ×2 -- 스냅숏 필수(합쳐 시간당 4회 안팎 · 그림자도 켬) |
| 상주 | ob5m · bookdepth · 아이스버그 | 옮김 | rtcollect | ×1(파일만 읽음) |
| 상주 | 롱숏비 `oi_lsratio` | 옮김 | rtduck | ×2 -- 20요청/5분 → 40 |
| 상주 | HL 체결 `hl_trades` | 옮김 | rtduck | ×1 |
| 상주 | HL 포지션 `hl_positions` | 옮김 | rtduck | ×2 -- 🔴가중치 600/분 → 1,200 = HL 한도 100%(전환 때만 켬) |
| 상주 | market_data_worker `market_data` | 옮김 | rtduck | ×2 -- 미시구조 36/분 → 72(기본 꺼짐) |
| 상주 | 뉴스 RSS `news` | 옮김 | rtcollect(quick-xml · regex) | ×2 -- 바깥 8곳(429 본 피드 있음) · 그림자 등록 후보에서 뺌 |
| 상주 | deribit 블록체결·GEX | 안 옮김(조사만 -- 아래) | — | (Deribit WS·REST · history API) |
| 상주 | SR lean | rtmodel | — | — |
| 상주 | 푸시·SR 근접 알림 | 범위 밖(알림) | — | — |
| 예약 | treasury_futures · macro_5m | 옮김 | rtcollect | ×2 -- 🔴Yahoo 429 위험 · 기본 안 켬 |
| 예약 | cboe 옵션 체인 + 요약기 | 옮김 | rtcollect | CBOE ×2(10분마다 종목 셋) · 바이낸스 klines ×2(종목마다 하루 1회 캐시 · 기본 꺼짐) |
| 예약 | liqmap_scorer | 옮김 | rtcollect | ×2 -- 바이낸스 klines 시간당 1회(limit 198 · 가중치 2) · 기본 꺼짐 |
| 예약 | 래스터 14일 보관(parquet+duckdb 뷰) | 옮김 `raster_archive` | rtduck | ×1(REST 없음) |
| 예약 | liq_tab | 옮김 `liq_tab` | rtduck | ×1(그림자 = REST 끄고 파이썬 1h 봉 캐시 복사) · 켜면 ×2(코인당 1회/시) |
| 예약 | soxl_weekend_forward | 옮김 `soxl_weekend`(`--record`·`--status`) | rtcollect | ×2 -- 새 주말마다 9요청 → 18/주 |
| 예약 | macro_bias_forward · macro_daily | 안 옮김(조정자 결정) | — | (FRED 키·Yahoo·재무부·DefiLlama) · 막는 것 = 위 조사 표 |
| 예약 | seal_to_lake | **반쯤** -- `rtduck::seal` DuckDB 원천(첫 표 binance/oi_lsratio_5m 일치) · SQLite 원천 13 스트림 남음(sqlite 확장) · hot 정리·상태 JSON 남음 | rtduck | ×1(REST 없음) |
| 예약 | risk_radar · regime_band · eth_direction_forward | rtmodel | — | — |
| 예약 | 월간 학습 | 파이썬 유지 | — | — |
| 예약 | raw_failover · dqm · morning_audit · multicoin · gen_data_map · verdict_ledger_health · backup | 범위 밖(R7 운영) | — | — |

**요약**: 상주 옮김 19(rtcollect 15 · rtduck 4) · 안 옮김 1(deribit -- 조사 표 아래) · rtmodel 1 · 범위 밖 1 /
예약 옮김 7(treasury · macro_5m · cboe+요약기 · liqmap_scorer · soxl_weekend · 래스터 보관 · liq_tab) · 안 옮김 2(macro_bias · macro_daily -- 조정자 결정) · 반쯤 1(seal_to_lake -- DuckDB 원천만) · rtmodel 3.
옮기지 않은 «불확실»: 뉴스 판정·ollaya · prometheus · trend_core_forward · f4c altdata · llm_brief / «안 쓰임»: 돌파 워커 · 신호 워커 · maxpain 기록기 · maker_fill_shadow.

---

## 형식이 다른 수집기 조사(넷째 조각)


**옮김(다섯째·여섯째 조각)**: 호가 래스터 .f32(`raster` -- 표준 라이브러리, 파일 바이트 일치) · depth_diff(`depth` -- flate2 · serde_json preserve_order 허용, 압축을 푼 내용 바이트 일치) · HL 자산 맥락 hl_ctx.sqlite(`hl_ctx`).
아래 표의 그 두 줄은 조사 기록으로 남긴다. **안 옮김**: raster 보관(parquet·duckdb) · HL 호가 .bt(불확실) · HL 체결·duckdb_persist_worker(duckdb 결정 전) · HL 포지션(입력이 duckdb) · 뉴스 RSS.

저장 형식이 SQLite 가 아니거나(또는 입력이 SQLite 가 아니어서) 지금 크레이트(rusqlite·serde_json·tokio·tokio-tungstenite·rustls·ureq)만으로는
«같은 파일»을 못 내는 것들. 줄 번호는 각 원본 파일. **형식을 SQLite 로 바꾸는 것은 «값이 바뀌는 변경»(읽는 쪽도 바뀐다) -- 사용자 결정.**

| 수집기 | 쓰는 것(형식·경로) | 읽는 곳 | 범위 문서 판정 | Rust 로 같은 파일을 내려면 | 일치를 보이는 법 | 대안 |
|---|---|---|---|---|---|---|
| `live_depth_diff_collector_20260915` | 시각별 JSONL `orderflow/depthdiff/<SYM>/<YYYY-MM-DDTHH>.jsonl` → 시각이 바뀌면 `.jsonl.gz`(compresslevel 6, 압축 성공 뒤 원본 삭제 :60-72) · 첫 줄 = REST 스냅숏 `{"_snapshot":…,"at_ms"}`(새 파일만 :100-105) · 그 뒤 WS 원문 프레임 · 순서 끊김 `{"_gap":true,…}`(:140-143) · flush 300행마다(:110) | `live_ob5m`(:57·96·114·436·458) · `live_eth_iceberg`(:36·193·235) · `live_bookdepth_feat`(:155·163 → **전진 fwd_dirshort**) · 감시 `ops_watchdog:959` · `raw_failover:36` | 쓰임(#6) -- 🔴문서에 빠진 것: bookdepth_feat 를 거쳐 **간접 전진 의존** | tungstenite·ureq + **flate2(새 크레이트)** · 스냅숏 줄 = `json.dumps(파싱한 응답)` 이라 키 순서 보존이 필요(serde_json `preserve_order` 기능) · 바이낸스 REST `depth limit=1000`(가중치 20, 시간당 1회) | gz 바이트는 못 맞춘다(헤더에 파일 이름·mtime) → **푼 내용 바이트 비교**(녹화 프레임 + 고정 시계) | SQLite 로 바꾸면 읽는 셋(꼬리 따라 읽기·gz 읽기)을 다 고쳐야 한다(하루 약 67만 행) |
| `live_orderflow_raster_collector_20260914` | `.f32` 고정폭 `orderflow/raster/<SYM>/<HH>.f32`: 헤더 32B `<4sHHfIqII`("FLWR", ver 1, 칸 240, 0.5, 0, 시각 시작 ms, 1000, 0) + 행 976B `<qif`+`f32[240]`(:91-94) · 매초 1행·매 행 flush(:177) · 14일 지난 것은 `raster_archive/…/<HH>.f32.parquet`(zstd) + `orderbook.duckdb` 뷰(:410-479, 실패 시 `.f32.gz`) | 서버 `read_window`·`vol_scores`·`approach_ratio`·`row_stats` import(히트맵·시장 맥락·`micro_ref`) · 감시 `ops_watchdog:958`·`725-771` | 쓰임(#19) · 전진 없음 | `.f32` = 표준 라이브러리(`to_le_bytes`)면 된다. 🔴바이트까지 같으려면 f32 더하기 순서 = 파이썬 dict 삽입 순서(갱신은 제자리 · pop 후 재삽입은 맨 뒤 · 스냅숏 병합 `{범위 밖 기존} | 스냅숏` :567-568, :644-651) -- 삽입 순서 맵 직접 구현 · 보관(parquet·zstd·duckdb)은 arrow/parquet/zstd/duckdb 크레이트(무겁다 -- duckdb 는 C++ 묶음 빌드) | 녹화 이벤트 + 가짜 시계 → `.f32` **바이트 비교** · parquet 은 pyarrow 로 되읽어 행 비교 | 보관 일괄 작업은 파이썬 cron 으로 떼어 둔다(시간당 1회 · 지연과 무관). 서버가 이 모듈을 import 하므로 형식을 바꾸면 `read_window` 도 옮겨야 한다 |
| `live_hyperliquid_book_ticker_collector_20260923` | ① `.bt` `orderflow/hyperliquid_bookticker/<COIN>/<HH>.bt`(헤더 `<4sHHqQII` "BTKR" · 행 `<qdfdf` 32B, 2000행·30초마다 flush, 시각이 바뀌면 gzip) ② `HL_BT_CTX_ONLY=1` 변형 → **SQLite** `data/hot/hl_ctx.sqlite`: `hl_asset_ctx(coin VARCHAR, recv_ms BIGINT, funding DOUBLE, open_interest DOUBLE, premium DOUBLE, oracle_px DOUBLE, mark_px DOUBLE, mid_px DOUBLE, day_ntl_vlm DOUBLE, day_base_vlm DOUBLE)` · `meta` · 인덱스 `(coin, recv_ms)`(:76-78·130-137) · 30초마다 한 트랜잭션 | ② 서버 시장 맥락(server.py:2505·2533·2552·2624) · `seal_to_lake:51·102` · `data_quality_monitor:49` / ① 감시·raw_failover·매니페스트뿐(읽는 코드 0) | **불확실**(#8, 질문 4 -- 호가 전진 검정 10-31 중단) | ① 표준 라이브러리 + flate2 · ② **지금 크레이트로 된다**(이번 조각과 같은 방식) | ① 재생 입력으로 바이트 비교 ② 표 덤프(recv_ms = 받은 시각이라 고정 시계) | ② 만 먼저 옮길 수 있다(질문 4 와 무관하게 «쓰임») -- 사용자가 ① 을 계속 쌓을지 정하면 함께 |
| `live_hyperliquid_trade_collector_20260916` | **DuckDB** `orderflow/hyperliquid/<COINS>/<YYYY-MM-DD>.duckdb`: `hl_trades(coin, ts_ms BIGINT, tid BIGINT, hash, side, px DOUBLE, sz DOUBLE, buyer, seller)` · `gaps`(:117-120) · 500행·20초마다(매번 연결 열고 닫음) · 3일 지난 파일 `.duckdb.gz`(:109-112) | 포지션 수집기의 대상 주소 뽑기(`positions:151-177`) · 감시 `ops_watchdog:960` | 쓰임(간접 · #10) | **duckdb 크레이트(bundled C++ -- 빌드가 무겁다)** + flate2 | 표 덤프 비교(파일 바이트는 못 맞춘다) | SQLite 로 바꾸면 포지션 수집기 `universe()` 한 곳만 고치면 된다(값이 바뀌는 변경 → 사용자 결정) |
| `duckdb_persist_worker.py`(+ 루트 `microstructure_scanner.py` 884 · `tail_risk_interceptor.py` 543 · `oi_lsratio_collector.py` 301) | DuckDB `microstructure.duckdb`(분마다 1행 × 3코인) · `tail_risk.duckdb` · `oi_lsratio.duckdb`(5분) + JSON `dashboard_state.json`(10초, 세션 3칸) · `liq_burst_state.json`(원자적 교체) | `dashboard_state` → 서버 `/api/events` → 화면 세션 · 감시 `ops_watchdog:162` · `liq_burst` → 청산 급증 배지(server.py:367) · `tail_risk` → 청산 5분 폴백(server.py:4431-4436, ETH 외) · `microstructure` → seal·감시뿐 · `oi_lsratio` → **서버 시장 맥락 롱숏비(server.py:2506·2520·2536 -- 이번에 확인)** · seal | ⓐⓑ 쓰임 · ⓒⓓ **불확실**(#11, 질문 5·6) · #15 oi_lsratio «불확실» → 🔴서버가 읽는 것이 확인돼 **«쓰임»으로 고칠 근거** | duckdb 크레이트 + 장 달력(pandas_market_calendars 의 JPX·LSE·NYSE 휴장·점심 :56-77 -- Rust 대응 없음, 휴장 표를 들여와야) + 미시구조 신호 884줄 재작성 | 표 덤프 · JSON 은 키 순서까지 비교 | JSON 둘(`dashboard_state`·`liq_burst_state`)만 Rust 로, DuckDB 쪽은 파이썬 유지 · 또는 SQLite 로 바꿔 서버·감시·seal 을 함께 고침 |
| `live_hyperliquid_positions_collector_20260924` | **SQLite** `data/hot/hl_positions.sqlite`(ETH) / `hl_positions_<코인들>.sqlite`(:61-63): `hl_positions`(14열) · `hl_universe(ts_ms, user, rank, notional_48h)`+ALTER `coin` · `hl_liquidations`(13열) · `hl_cycles` · 인덱스 넷(:72-87·261-264) · 바퀴 하나 = 트랜잭션 하나 | 서버 HL 고래 청산(server.py:475-509) · 시장 맥락(server.py:2535) · seal(:52-57·114-120) · 매니페스트 | 쓰임(#9) | rusqlite·ureq 로 된다 -- 🔴**입력이 HL 체결 수집기의 DuckDB**(read_only, :151-177)라 duckdb 읽기가 필요 | 녹화한 `/info` 응답 + 고정 시계 → 표 덤프 | 체결 수집기를 SQLite 로 바꾸면 duckdb 없이 옮길 수 있다 · HL REST `clearinghouseState` 0.4초 간격(분당 ≤150) |
| `live_news_rss_collector_20261005` | **SQLite** `data/hot/news_rss.sqlite`: `items(source, link, title, summary, categories, pub_ms, first_seen_ms, backfill, PRIMARY KEY(source, link))`(INSERT OR IGNORE) · `polls` · 인덱스 둘 · polls 7일만(:84-117) | 서버 `/api/news`(server.py:559·730) · 판정 워커(정지 중) | 쓰임(#13) · 전진 아님 | rusqlite·ureq 까지는 되지만 **XML 파싱·HTML 엔티티(`html.unescape`)·RFC822 날짜(`email.utils`)·정규식**(:48)에 허용 밖 크레이트가 필요 -- 직접 짜면 같은 결과를 보이기 어렵다 | 녹화한 HTTP 응답(피드 8개, ETag/Last-Modified) → items/polls 덤프 | 형식은 그대로 · **크레이트 허용 여부를 사용자에게** 물을 것(quick-xml/roxmltree · regex · 날짜) |

## 원본에서 본 이상한 점(고치지 않음 · 근거)

- depth_diff: 재접속 때 `prev_u=None`(:136) · 같은 시각 파일에 이어 쓸 때 스냅숏을 다시 안 넣는다(:100) → **재접속 사이 구멍이 `_gap` 으로 안 남는다**.
  시간 기준 flush 없음(:110 -- 조용한 코인은 mtime 이 멎는다) · 스냅숏 REST 가 동기 urllib(:78 -- 이벤트 루프 최대 10초 정지) ·
  재시작이 시각 경계를 넘으면 지난 시각 `.jsonl` 은 영영 압축 안 됨 · `.gz` 를 임시 이름 없이 바로 씀 · 감독 주석의 «.bt»는 틀림(`supervisor_depth_diff.sh:7`).
- raster: 크래시로 마지막 행이 반만 쓰였으면 채우기 계산(:171-173)이 행 경계를 벗어난 자리부터 채운다 · parquet 보관이 `_raster_loop` 안에서 동기(:700) · 스냅숏 직후 그 계기 이벤트를 버림(:609).
- HL 체결: 압축은 프로세스 안에서 날짜 전환을 볼 때만(:109 -- 자정을 넘긴 재시작이면 그날 파일은 압축 안 됨) · 압축이 쓰기 스레드 안에서 동기.
- HL 포지션: `seen_tids` 가 끝없이 커진다(:233) · 429 면 그 주소를 그 바퀴에서 건너뛴다(:300-303) · 매니페스트 `HL_POS_COINS=BTC,SOL,XRP` 면 파일 이름이
  `hl_positions_btc_sol_xrp.sqlite` 인데 서버는 `_sol_xrp`·`_btc` 를 읽는다(server.py:496-499) -- 서버에서 실제 조합 확인 필요.
- 미시구조: 분 버킷을 `datetime.now()`(지역 시각, naive)로 TIMESTAMPTZ 에 넣는다(:840·853) · 주석 «trading_bot.py 가 미시구조를 소유»(`duckdb_persist_worker.py:32-37`)는 봇 은퇴(10-12) 뒤 낡음 ·
  tail risk 는 ETH 만 쓰는데 ETH 외 폴백은 더 이상 갱신되지 않는 `tail_risk_1m_btc` 등을 읽는다(`coin_config.py:48·60`).
- 뉴스: 동기 루프라 한 피드 시간초과(20초)가 나머지를 민다(:126) · 저장 실패해도 polls 에 status 200·n 이 남는다(:127·146-150) · sec.gov 는 연락처 UA 를 요구하는데 브라우저 UA(:50).


## deribit 블록체결·GEX 조사 (2026-11-21 아홉째 조각 -- 옮기지 않음)

| 몫 | 원본 | 크기 | 하는 것 | 옮길 때 막는 것 |
|---|---|---|---|---|
| 체결 WS | live_deribit_block_trade_collector `run`·`row`·`write` | ~200줄 | `wss://www.deribit.com/ws/api/v2` 구독 `trades.option.{ETH,BTC,USDC}.100ms` · 하트비트 30초·90초 무음 = 끊김 · 20초마다 option_trades(trade_id PK)·gaps 에 · 재기동 틈은 REST 백필(24시간) | tokio-tungstenite·duckdb 로 된다(새 크레이트 없음) · 가짜 WS·REST 로 정답지 |
| 블록 묶음·구조 | `group_blocks`·`block_shape`·`_shape` | ~110줄 | 블록 다리 → 구조 13종(straddle…iron_condor) · BS 델타·베가 부호 | GEX 모듈의 `_parse_instrument`·`_bs_delta` 를 같이 옮겨야 한다(cboe_summary 의 BS 와 식이 비슷하나 별개) |
| 시간당 대조 | `reconcile`·`hourly_doi`·`hourly_flow` | ~140줄 | history.deribit.com 으로 3시간 겹쳐 빠진 체결·block_rfq_trades·block_future_legs 메움 · 시간당 OI 변화 | REST 셋(history·public 둘) |
| 체인·요약 10분 | collect_deribit_option_gex `poll_once`·`options_summary`·`write_state` | ~450줄(pandas) | ETH·BTC·SOL·XRP 체인 스냅숏 · GEX·감마 플립 · DVOL·실현 7일·만기별 ATM IV/RR/BF·max pain · OKX·Bybit 옵션 가격(다른 거래소) | pandas 계산 450줄을 값 일치로(cboe_summary 와 같은 종류의 일 -- 1~2 조각) · 다른 거래소 REST 둘 |
| 옛 파일 옮기기 | `migrate_once` | ~30줄 | 첫 기동 때 옛 duckdb 둘의 이력 복사 | 서버에서는 이미 끝남(옮길 필요 없음일 가능성 -- 확인) |
| 상태 JSON 둘 | deribit_block_trades_state.json · deribit_gex_state.json | — | 대시보드 «옵션» 카드의 원천 | 위 전부가 있어야 같은 글자 |

순서 제안: ① 체결 WS + 블록 묶음 + 상태 JSON(블록 몫) → ② 체인·요약(pandas 450줄) → ③ 시간당 대조. 🔴단일 writer: 그림자는 다른 duckdb 파일에(원본과 같은 파일이면 하나가 죽는다).
