# rtcollect·rtduck 이전 상태 (2026-11-21 열두째 조각 · 범위 문서 docs/rtcore_port_scope_20261120.md 표 2 의 «쓰임»·«전진» 전부)

**상태는 아래 두 표의 «상태» 칸 한 곳에만 적는다**(다른 절·SHADOW.md 는 «어떻게»만). 상태 값:
옮김(Rust 가 원본과 같은 파일·값을 낸다 -- 근거 칸) · 안 옮김(막는 것 = §3) · 파이썬 유지(학습) · rtmodel(모델 서빙 꾸러미 몫) · 범위 밖(감시·알림·배포 = R7).
범위 문서 판정 «안 쓰임»(돌파 워커 · 신호 워커 · maxpain 기록기 · maker_fill_shadow)·«불확실»(뉴스 판정·ollaya · prometheus · trend_core_forward · f4c altdata · llm_brief)은 표에 넣지 않았다.
«그림자에서 켜면 REST» = 서버에서 파이썬이 같은 일을 계속하는 동안 Rust 그림자도 그 REST 를 켜면 같은 IP 에서 몇 배가 되는가(×1 = REST 없음 · 웹소켓·파일만).
어떻게 띄우고 견주는지 = SHADOW.md(환경변수·의도한 차이 #1~#22·대조·전환).

## 1. 상주 (scripts/ops/supervisor_*.sh)

| 프로그램(원본) | 저장 | 읽는 곳 | 전진 | 상태 | 종류 | 그림자에서 켜면 REST | 일치 근거(합성 정답지 · cargo test) |
|---|---|---|---|---|---|---|---|
| 바이낸스 체결 테이프 `live_trade_tape_collector` | SQLite hot | 풋프린트·수급 · seal | whale_duel1 | 옮김 | rtcollect `binance_tape` | ×2 -- 기본 꺼짐 · 켜면 종목마다 klines 1/분+1/5분 + aggTrades 복구 | tape_parity · rest_parity |
| OKX 체결 테이프 | SQLite hot | server · seal | whale_duel1 | 옮김 | rtcollect `okx_tape` | ×2 -- 기본 꺼짐 · 공백 없으면 종목당 약 1.2회/분 | venue_parity · venue_rest_parity |
| Bybit 체결 테이프 + 맥락 | SQLite hot 둘 | 시장 맥락 OI · seal | — | 옮김 | rtcollect `bybit_tape` | ×2 -- 기본 꺼짐 · 1회/분 | venue_parity |
| Bybit 청산 | SQLite hot | 청산 원 · liqmap_scorer | liqmap_fwd | 옮김 | rtcollect `bybit_liq` | ×1 | venue_parity(의도한 차이 #1·#2) |
| OKX 맥락 | SQLite hot | server · coin_flow · liq_tab · liqmap_scorer | liqmap_fwd | 옮김 | rtcollect `okx_ctx` | ×2 -- 시작 때 ctVal 1회뿐 | ctx_parity |
| 바이낸스 맥락 | SQLite hot | server · liq_tab · liqmap_scorer · dqm · 감시 | liqmap_fwd | 옮김 | rtcollect `binance_ctx` | ×2 -- 🔴OI 약 540회/분 → 합쳐 1,080(IP 한도 45%) · 그림자 동안 끔 | ctx_parity |
| xyz100 | SQLite hot | /api/xyz100 · 감시 · seal | — | 옮김 | rtcollect `xyz` | ×2 -- 재접속 때만(4회/접속) | ctx_parity |
| HL 자산 맥락(book ticker CTX_ONLY) | SQLite hot | 시장 맥락 · seal · dqm | — | 옮김 | rtcollect `hl_ctx` | ×1 | ctx_parity |
| HL 호가 `.bt`(Pi) | 고정폭 `.bt` → gz | Pi → 서버 · 연구 | — | 옮김 | rtcollect `hl_bt` | ×1 | `.bt` 바이트 일치 |
| depth_diff | JSONL → 시각별 gz | ob5m · iceberg · bookdepth_feat | fwd_dirshort(간접) | 옮김 | rtcollect `depth` | ×2 -- 시간당 depth(가중치 20) 1회 · 전환 때 반드시 켬 | depth_parity(푼 내용 바이트) |
| 호가 래스터 | 고정폭 `.f32` | read_window → 히트맵·시장 맥락 | — | 옮김 | rtcollect `raster` | ×2 -- 스냅숏 필수(합쳐 시간당 4회 안팎 · 그림자도 켬) | raster_parity(파일 바이트) |
| ob5m 실시간 워커 `live_ob5m` | SQLite `data/live/ob5m/<코인>.sqlite` | /api/ob5m | — | 옮김 | rtcollect `ob5m` | ×1(파일만 읽음) | ob5m_parity 칸 7,417 비트 |
| 호가 깊이 10열 `live_bookdepth_feat` | SQLite `data/live/bookdepth/ethusdt.sqlite` | dir_short 피쳐(→ rtmodel) | fwd_dirshort | 옮김 | rtcollect `bookdepth` | ×1 | ob5m_parity 칸 2,110 비트 · 930 은 1e-12 안(BLAS ddot) |
| 아이스버그 `live_eth_iceberg` | JSON 상태 + JSONL 사건 | /api/iceberg · 감시 | — | 옮김 | rtcollect `iceberg`(ETH·XRP) | ×1(depthdiff·hot 테이프 읽기만) | iceberg_parity 2초 state 3,582 글자 |
| HL 체결 | DuckDB 일별 → 3일 뒤 gz | HL 포지션 입력 · 감시 | — | 옮김 | rtduck `hl_trades` | ×1 | hl_trades_parity · hl_trades_loop |
| HL 포지션 | SQLite hot `hl_positions*.sqlite` | HL 고래 청산 · 시장 맥락 · seal | — | 옮김 | rtduck `hl_positions` | ×2 -- 🔴가중치 600/분 → 1,200 = HL 한도 100%(전환 때만 켬 · REST 없이는 안 돈다) | hl_positions_parity |
| market_data_worker `duckdb_persist_worker` | DuckDB 둘 + JSON 둘(dashboard_state · liq_burst_state) | 화면 세션·SSE · 감시 · seal | — | 옮김 | rtduck `market_data` | ×2 -- 미시구조 36/분 → 72(기본 꺼짐) | market_parity · micro_parity · market_loop |
| oi_lsratio 워커 | DuckDB | server 롱숏비 · seal | — | 옮김 | rtduck `oi_lsratio` | ×2 -- 20요청/5분 → 40 | oi_parity |
| 뉴스 RSS | SQLite hot | /api/news | — | 옮김 | rtcollect `news`(quick-xml·regex) | ×2 -- 바깥 8곳(429 본 피드 있음) · 그림자 없이 전환 | news_parity 요청 68·기사 49·polls 69 |
| Deribit 체결·블록·GEX(`live_deribit_block_trade_collector` + `collect_deribit_option_gex`) | 옵션 DuckDB + JSON 둘 | 옵션 카드 · 감시 · seal | — | 옮김 | rtduck `deribit` | 체결 WS ×1(Deribit) · REST 켜면 ×2(백필 · 10분 체인 4코인 · OKX·Bybit 옵션 · 매시 history) -- 기본 꺼짐 | deribit_parity · deribit_gex_parity · deribit_recon_parity · deribit_loop |
| SR lean `live_sr_lean` | JSONL | /api/sr-lean | sr_ml_5m_forward | rtmodel | — | — | — |
| 푸시 알림 · SR 근접 알림 | 웹푸시·텔레그램 | 폰 · 알림 센터 | — | 범위 밖 | — | — | — |

## 2. 예약 (cron)

| 프로그램(원본) | 주기 | 저장 | 읽는 곳 | 전진 | 상태 | 종류 | 그림자에서 켜면 REST | 일치 근거 |
|---|---|---|---|---|---|---|---|---|
| treasury_futures | 매분·매시 | SQLite | /api/treasury-futures · 감시 · lake | — | 옮김 | rtcollect `treasury_futures` | ×2 -- 🔴Yahoo 429 위험 · 그림자 없이 전환 | cron_parity |
| macro_5m | */5 | SQLite | Macro Trend · /api/macro-today · 감시 | — | 옮김 | rtcollect `macro_5m` | ×2 -- 같은 이유 | cron_parity |
| cboe 옵션 체인 + 요약기 | */10 | 원본 gz + JSON | /api/gex | soxl_own_iv | 옮김 | rtcollect `cboe_chain` | CBOE ×2 · 바이낸스 klines ×2(종목마다 하루 1 · 기본 꺼짐) | cron_parity · cboe_summary_parity state 5 |
| liqmap_scorer | 매시 | JSONL·JSON | 알림 센터 · 판정 | liqmap_fwd | 옮김 | rtcollect `liqmap_scorer` | ×2 -- klines 시간당 1회(가중치 2) | liqmap_parity 새 줄 48 |
| soxl_weekend_forward(`--record`·`--status` -- 판정 `--look` 은 파이썬) | 매일 | JSONL | 판정 | soxl_wknd1·2 | 옮김 | rtcollect `soxl_weekend` | ×2 -- 새 주말마다 9요청 → 18/주 | soxl_weekend_parity |
| 래스터 14일 보관 | 매시 | parquet + duckdb 뷰 | 연구 | — | 옮김 | rtduck `raster_archive` | ×1 | raster_archive_parity(양방향 읽기) · 의도한 차이 #20 |
| liq_tab | 매시 | JSON + 1h 봉 parquet | /api/liq-tab | — | 옮김 | rtduck `liq_tab` | ×1(그림자 = REST 끄고 파이썬 1h 봉 캐시 복사) · 켜면 ×2 | liq_tab_parity(글자) |
| seal_to_lake | 매일 | Parquet lake | 전진 다수 · 감시 | 여럿 | 옮김 | rtduck `seal` | ×1 | seal_parity · seal_full_parity · seal_duck_parity · 스트림별 원본 dry-run 전부 건너뜀 · 서버 실자료 4 스트림 확인 |
| macro_daily | 매시 | SQLite | macro_trend | macro_bias | 안 옮김 | — | (FRED 키·재무부·DefiLlama) | — |
| macro_bias_forward | 주말 | JSONL | 판정 | fwd_macro_regime | 안 옮김 | — | (FRED 키·Yahoo·바이낸스 현물) | — |
| risk_radar · regime_band · eth_direction_forward_recorder | 매분·매시·매일 | JSON·JSONL | 화면·판정 | fwd_band · dir4h_p2·dir3asset | rtmodel | — | — | — |
| 월간 학습(dir4h·vol4h·dir_short·box_hold Platt) | 월간 | joblib | 서빙 | 여럿 | 파이썬 유지 | — | — | — |
| raw_failover · data_quality_monitor · morning_audit · multicoin_collectors · gen_data_map · verdict_ledger_health · backup | 여러 | JSON·보고 | 감시·사용자 | — | 범위 밖 | — | — | — |

**셈(위 두 표에서)**: 상주 = 옮김 20(rtcollect 15 · rtduck 5) · rtmodel 1 · 범위 밖 1 / 예약 = 옮김 8(rtcollect 5 · rtduck 3) · 안 옮김 2 · rtmodel 1줄(셋) · 파이썬 유지 1 · 범위 밖 1줄.

## 3. 안 옮김 둘 -- 막는 것(조정자 결정 대기)

- **macro_daily**(181줄 · 관측 append-only 병합 + 순유동성): ① 🔴FRED 키 자리 -- 원본은 `load_dotenv` 로 저장소 루트의 점-env 파일에서 `FRED_API_KEY` 를 읽는다(cron 이 그 파일을 안 읽어 스크립트가 스스로) --
  Rust 가 그 파일을 읽을지 · 환경변수만 받을지 · 다른 키 파일로 둘지 = 사용자 결정 ② 계열 목록(SOURCES)·FRED 호출·netliq 식이 `dashboard/macro_trend`(434줄)에 있어 rtserve(Macro Trend 카드)와 한 곳에 둘지 정해야 한다 ③ 계산 자체는 가볍다.
- **macro_bias_forward**(182줄 · 주 Z 점수): ① 같은 키 자리 ② 연구 모듈(research_macro_regime_bias 443줄)의 pandas rolling·rank·z 계산을 비트로 맞출 도구 ③ Yahoo ^KS11 5년 일봉(그림자 ×2 · 429).

옛 조사 표(형식이 다른 수집기 · pandas 기록기 · Deribit 몫 나누기)는 다 옮긴 뒤라 지웠다 -- git 기록(열한째 조각까지의 이 파일)에 남아 있다.

## 4. 원본에서 본 이상한 점(고치지 않음 · 근거)

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


## 5. 서버에서 옮겨 켤 때 순서 (2026-11-21 열셋째 조각 · 실행 = `rtcollect/tools/takeover.sh` · 사용자 «기다리지 말고 바로바로 반영 배포»)

**원칙**: 한 번에 하나 · 대기 기간은 두지 않는다(7일 병행·24시간 관찰 없음) -- 대신 «넘기기 직전 check 통과 → go → 60초 확인 → 다음 스트림 넘기기 전 check 한 번 더» ·
값 보존을 보인 뒤에만 옛 것을 끈다(줄이는 것은 기다리는 시간이지 대조가 아니다) · 위(원천)부터 아래(읽는 것)로.
쓰는 쪽은 경로를 안 바꾼다(읽는 쪽 server.py·감시·seal 은 그대로) -- Rust 가 원래 파일을 이어 쓴다(이어 써도 같은 행 · 각 조각에서 확인).

**한 스트림의 걸음**(그림자 → 대조 → 전환 → 되돌리기)
1. 그림자: Rust 를 그림자 경로(기본값)로 띄운다(상주 = `supervisor_rtcollect.sh`·`supervisor_rtduck.sh` · 예약 = SHADOW.md §15 줄). REST 는 원칙적으로 끈다(SHADOW.md 배수 표).
2. 대조 통과 기준: `takeover.sh check <종류> [<종목>]` -- 그림자가 떠 있던 창 전체에서 **설명 못 하는 차이 0** · 견준 것 > 0(N 일을 기다리지 않는다).
   설명되는 차이 = SHADOW.md §11 의도한 차이 #1~#22 · 재연결 공백(gaps 끼리 빼고) · REST 를 끈 그림자에서만 빈 몫. 도구: `shadow_tape.py`(테이프·맥락) · 표 덤프 비교(나머지) · seal 은 `seal(dry_run=True)` 전부 건너뜀.
3. 전환: (가) 그 Rust 그림자 감독 → 프로세스 끄기 (나) 파이썬 감독 bash → 파이썬 수집기 끄기 · crontab 줄 주석 (다) Rust 를 `RTCOLLECT_TAKEOVER=1`/`RTDUCK_TAKEOVER=1` + 원래 경로 변수로 띄움(REST 쓰는 종류는 `*_REST=1`) ·
   crontab 줄 바꿈 (라) 첫 쓰기·신선도(감시 `ops_watchdog`)·대시보드 카드 눈 확인 · 매니페스트 `multicoin_collectors.json` 의 runner 를 rtcollect/rtduck 로(SHADOW.md §9 «감시에 넣을 때»).
4. 되돌리기: Rust 감독 → Rust 끄기 · crontab 되돌림 · 파이썬 감독 다시 · 끈 사이는 gaps 로 남는다(바이낸스 테이프는 REST 복구가 메움). 파일 형식이 같아 파이썬이 Rust 가 쓴 파일을 그대로 이어 쓴다.
   🔴DuckDB 파일(oi_lsratio·hl_trades·market_data 둘·deribit·seal lake)은 단일 writer -- 반드시 한쪽을 끈 **뒤** 다른 쪽을 켠다.

**순서(위가 먼저)**

| 차례 | 종류 | 왜 이 차례 | 옮길 때 특이 |
|---|---|---|---|
| 1 | rtcollect `hl_ctx` · `xyz` · `hl_bt` | 화면만 읽음 · 전진 없음 | xyz 는 재접속 candleSnapshot REST(×1 로 돌아감) |
| 2 | rtcollect `depth` | ob5m·bookdepth·iceberg 의 원천 · 간접 전진(fwd_dirshort) | 스냅숏 REST 반드시 켬 |
| 3 | rtcollect `ob5m` · `bookdepth` · `iceberg` | depth 를 읽기만 | bookdepth = fwd_dirshort 피쳐 -- 대조 기준을 칸 비트로 |
| 4 | rtcollect `raster` → rtduck `raster_archive` | 화면 히트맵 · 보관은 래스터 뒤 | 보관을 옮기면 파이썬 래스터 수집기의 `_purge_old` 를 끄는 것과 같이(`OF_ARCHIVE=0` 대신 파이썬 쪽 정지) |
| 5 | rtduck `hl_trades` → `hl_positions` | 포지션이 체결 DuckDB 를 읽음 | hl_positions 는 HL REST 한도 100% 문제 -- 파이썬을 끈 뒤에만 `RTDUCK_REST=1` |
| 6 | rtduck `oi_lsratio` · `market_data` | 화면 롱숏비·세션·SSE · seal 원천 | 단일 writer · market_data 미시구조 REST 켬 |
| 7 | rtcollect `bybit_tape`·`okx_tape`·`binance_tape` | 🔴whale_duel1 전진(바이낸스+OKX 테이프) | 사용자 승인 뒤 · 바이낸스 `RTCOLLECT_REST=1`(복구) |
| 8 | rtcollect `bybit_liq`·`okx_ctx`·`binance_ctx` | 🔴liqmap_fwd 전진(청산·마크) | 사용자 승인 뒤 · binance_ctx OI REST 540/분 -- 파이썬을 끈 뒤에만 켬 |
| 9 | 예약 rtcollect `liqmap_scorer` · `soxl_weekend` · `cboe_chain`(+요약기) | 🔴전진 장부(liqmap_fwd·soxl_wknd·soxl_own_iv) | 장부 파일을 이어 씀 · 옮기는 날 두 쪽이 같은 줄을 썼는지 확인(이중 줄 금지 -- 파이썬 cron 을 먼저 주석) |
| 10 | 예약 rtduck `liq_tab` · rtcollect `treasury_futures`·`macro_5m` | 화면 | Yahoo 는 그림자 없이 바로 전환(×2 를 피하려고 -- 하루 대조는 옮긴 뒤 파이썬 단발 실행과) |
| 11 | rtcollect `news` | 화면 | 바깥 8곳 -- 그림자 없이 전환(같은 이유) |
| 12 | rtduck `deribit` | 옵션 카드 · seal 원천 | 단일 writer · `DERIBIT_BT_DIR=data/live RTDUCK_TAKEOVER=1 RTDUCK_REST=1` · 파이썬 GEX 매시 cron 이 꺼져 있는지 확인 |
| 13 | rtduck `seal` | 🔴lake = 전진 검정 다수의 원천 · hot 정리 | 맨 끝 · 원천이 모두 안정된 뒤 · `SEAL_LAKE=data/lake SEAL_STATUS=data/live/ops_watchdog/seal_status.json RTDUCK_TAKEOVER=1`(hot 정리가 켜진다) · 첫 날은 `--dry-run` 으로 «전부 건너뜀» 확인 뒤 |

남는 파이썬(옮기지 않음): macro_bias_forward · macro_daily(키 자리 결정 대기) · 월간 학습 · 감시 묶음(R7) · 모델 서빙(rtmodel 몫).

## 6. 넘겨도 되는가 (2026-11-21 열셋째 조각 · `takeover.sh` 종류별)

`bash rtcollect/tools/takeover.sh check|go|back <종류> [<종목>]` -- 걸음은 §5. 종목 = 도는 파이썬 수집기의 목록 **통째로**(인자 없으면 그 종류의 수집기 하나 · 인자가 목록 안의 한 종목이면 그 수집기 · 10-31 통합 뒤 테이프·맥락은 한 프로세스가 여러 종목).
check 는 그림자 감독 로그의 «SUPERVISOR starting» 시각마다 [그 앞 Rust 마지막 자료, 그 뒤 첫 자료] ±5초를 «그림자 꺼진 구간»으로 빼고(첫 기동 전은 전부) 그 수·길이·뺀 행을 찍는다 · 그림자 Rust 가 파이썬 목록 전부를 뜨는지도 본다(안 뜨는 종목이 있으면 불통과). go 는 check 를 다시 돌려 통과일 때만 바꾼다. 끊긴 초는 go 가 «파이썬 수집기 종료 → 출력 파일 mtime 첫 전진»으로 잰다.
«그림자» 칸 = `shadow_start.sh` 기본으로 뜨는가(예약 셋은 `shadow_start.sh cron` 줄을 조정자가 등록했을 때만). 뜨지 않은 종류는 check 가 «그림자 출력 없음»으로 불통과 → go 안 됨.

| 차례 | 종류 | 대조(check) | 끊김 예상 | 전진 검정 · 끊긴 구간이 남는 곳 | REST(넘긴 뒤) | 그림자 |
|---|---|---|---|---|---|---|
| 1 | `xyz` | shadow_tape 표 셋 · ctx 0.5% 기준 | 수 초 | — · 재접속 때 candleSnapshot 이 1분봉을 메움 | 켬(재접속 때만) | 뜸 |
| 1 | `hl_ctx` | shadow_tape hl_asset_ctx -- Rust 쪽 = hl_bt 그림자 hl_ctx.sqlite + hl_ctx 그림자 hl_ctx_only.sqlite 를 이어 하나로 | 수 초 | — · 그 초가 빔(gaps 없음) | 없음 | 뜸 · 서버에는 파이썬 전체 모드(hl_bt, 코인별)도 같은 hl_ctx.sqlite 에 쓴다 -- 원본도 프로세스 넷이 한 파일(WAL)이라 코인 하나씩(hl_bt <코인> · hl_ctx = CTX_ONLY 묶음) 넘겨도 된다(11-22) |
| 2 | `depth <SYM>` | shadow_tape --depth(첫·끝 시 빼고 프레임 글자) | 수 초 | fwd_dirshort(간접) · 시 파일의 `_gap` 줄 · ob5m·bookdepth·iceberg 는 끊긴 자리를 끊김으로 읽음 | 켬(스냅숏 · 시간당 1) | 뜸 |
| 3 | `ob5m <SYM>` · `bookdepth` · `iceberg <SYM>` | shadow_tape bars·marks / --events | 0(depthdiff 파일을 이어 읽는다 -- 다시 뜨면 따라잡음) | bookdepth = fwd_dirshort 피쳐 · 봉 값이 이어진다 | 없음 | 뜸 |
| 4 | `raster <SYM>` | shadow_tape --raster(§5 SHADOW 99%·0.95 기준) | 수 초 + 스냅숏 | — · 그 초 행이 무효 | 켬(스냅숏) | 뜸 · 🔴넘긴 종목은 파이썬 `_purge_old` 가 멈춘다 -- 전 종목을 넘긴 뒤 `raster_archive` 예약 줄을 TAKEOVER 로(스크립트 밖 · 손으로) |
| 5 | `hl_trades <코인>` | shadow_duck 날짜 파일마다 열쇠 (coin, tid) | 수 초 | — · gaps 표 | 없음 | 뜸 |
| 6 | `market_data` | shadow_duck 행 수(1% 안) | 수 초 | — · 미시구조 그 초 행 없음 · dashboard_state 몇 초 묵음 | 켬(OI·premium 10초) | 뜸(REST 끔 -- 행 수만 견줄 수 있다) |
| 6 | `oi_lsratio` | shadow_duck 열쇠 (ts, symbol) | 0(5분 봉을 REST 로 다시 받는다) | — | 켬 | 🔴안 뜸(REST ×2) -- 넘기려면 잠깐 `RTDUCK_SHADOW_REST=1` 로 그림자를 띄워 봉 몇 개를 쌓은 뒤 |
| 7 | `bybit_tape` · `okx_tape` · `binance_tape` · `binance_spot` | shadow_tape trade_tape_1s 등(gaps·backfill·verify 분 뺌) | 수 초 | 🔴whale_duel1(바이낸스+OKX) · gaps 표 `ws_reconnect` · 바이낸스·OKX 는 REST 복구가 메움(backfill_1m) | 켬 | 뜸 · 같은 파일에 쓰는 다른 파이썬이 있으면 거부 |
| 8 | `bybit_liq` · `okx_ctx` · `binance_ctx` | shadow_tape 표마다 | 수 초 | 🔴liqmap_fwd · 청산·마크 그 초 빔(바이낸스 맥락은 gaps 표 없음 -- 로그 시각으로) | okx_ctx 켬(ctVal) · binance_ctx 켬(OI 540/분 -- 파이썬이 꺼져 두 배 아님) · bybit_liq 없음 | 뜸(binance_ctx 의 oi_1s 는 그림자에서 비어 «한쪽이 비었다» -- 견주지 않음) |
| 9 | `liqmap_scorer` · `soxl_weekend` · `cboe_chain` | — | — | 🔴전진 장부(liqmap_fwd·soxl_wknd·soxl_own_iv) | — | 스크립트 밖 -- 장부 이중 줄 위험이라 손으로(§5 9) |
| 10 | `liq_tab`(예약) | shadow_tape --json(generated_ms 빼고 글자) | 0(다음 시각부터 Rust) | — | 켬(klines 코인당 1/시) | 예약 줄 등록 때만 |
| 10 | `treasury_futures` · `macro_5m` | — | — | — | — | 스크립트 밖 -- 그림자 없음(Yahoo ×2) |
| 11 | `news` | — | — | — | — | 스크립트 밖 -- 그림자 없음(바깥 8곳 ×2) |
| 12 | `deribit` | shadow_duck option_trades·블록(체인·요약 표는 그림자에서 비어 행 수만) | 수 초 + 20초 플러시 | — · 재기동 REST 백필이 메움(gaps.backfilled) | 켬(백필·10분 체인·매시 대조) | 뜸(체결 WS 만) |
| 13 | `seal`(예약) | shadow_duck --lake(그림자 lake 의 날짜 범위 · 파일마다 지문) | 0(다음 날 시각부터 Rust) | 🔴lake = 전진 검정 다수 · hot 정리가 켜진다(TAKEOVER) -- 첫 실행은 `--dry-run` 으로 «전부 건너뜀» 확인 권함 | 없음 | 예약 줄 등록 때만 |
| — | `hl_positions` | — | — | — | — | 안 다룸: hl_positions = 조회 순간마다 값이 달라 견줄 방법 없음(REST ×2) |

go 가 하는 일(상주): ① crontab 백업 `logs/supervisor/crontab_backup_<날짜시각>_takeover_<태그>.txt` ② 파이썬 @reboot 줄과 그림자 줄 앞에 `#takeover:<태그> ` · 끝에 Rust 줄(` #takeover:<태그>` 로 끝남) · 줄 수 +1·바뀐 줄 수 검증 뒤 적용 ·
multicoin 매니페스트의 그 항목을 `taken_over` 로(백업 `logs/supervisor/multicoin_backup_*` -- `resume` 이 파이썬을 다시 띄우지 않게) ③ 파이썬 `_supervise.sh` 부모 → 수집기 SIGTERM(10초·30초 안에 안 죽으면 멈춤) · 다시 떴는지 확인
④ 그림자 감독 → Rust ⑤ `*_TAKEOVER=1` + 진짜 경로 + (REST 종류) `*_REST=1` + 파이썬 environ 의 종목 값 그대로 ⑥ 60초: 프로세스 · mtime · 새 로그의 panic·crash-looping·exited 수.
back 은 이 태그가 바꾼 줄만 되돌린다(그 뒤 다른 종류를 넘겼어도 그 줄은 그대로) · 백업과 같은지 찍는다 · 파이썬은 주석 풀린 @reboot 줄의 명령으로 다시 띄운다 · 그림자는 다시 안 띄운다.
시험: `bash rtcollect/tools/takeover_selftest.sh`(가짜 crontab·가짜 프로세스 · 진짜 crontab 은 안 건드림).

### 6-1. check 가 무엇을 근거로 통과/불통과를 내나 (2026-11-22 열일곱째 조각)

공통: 그림자 감독 로그의 «SUPERVISOR starting» 마다 Rust 가 비어 있던 구간(가장 촘촘한 표의 가장 큰 빈틈 · 첫 기동 앞 전부)과
«끊김» 줄 수만큼의 «Rust 만 빈 곳»(연결 끊김) · «파이썬만 빈 곳»을 뺀다. 설명으로 치는 것과 그 까닭(둘 다 맞는 이유) 한 줄씩:
- 같은 줄 두 번(묶음 표): 한쪽이 같은 메시지를 두 번 받아 두 번 적음 -- 값 정보는 같다(okx_oi ZEC 1795273431545 파이썬 두 줄).
- Rust 재구독 현재값: 다시 붙으면 거래소가 «지금 값»을 한 번 더 보낸다 -- 값이 그 종목 직전 파이썬 줄과 같으면 새 정보가 아니다.
- depth 다시 붙은 자리: _gap 없이 pu 가 끊긴 곳 = 그 쪽 연결이 끊겼다 붙음(원본도 재접속 때 prev_u 를 비워 _gap 을 안 쓴다 · :136) -- 그 사이 다른 쪽만 받은 프레임.
- 1분봉: 뺄 구간에 걸친 봉(끊김이 걸친 분은 받은 체결 수가 갈린다).
- HL 재접속 빈틈(11-22): 두 구현 모두 끊기면 3초 쉬고 다시 붙는다 -- .bt 는 거래소 ms 로 «다른 쪽만 3초 넘게 받음», HL 맥락 표(hl_*)는 초 차 ≥3 을 끊김으로 본다(전에는 5초·10초 문턱이라 «끊김» 33·134줄이 0번 설명). 그 사이 받은 쪽 행은 거래소 값 그대로.
- deribit 강제청산 표시: WS 체결에는 liquidation 이 없고 매시 reconcile(REST)이 채운다(원본 :333-334·361) -- Rust 에 체인 표가 없거나 비면(그림자 REST 꺼짐) «그 칸만 파이썬 값 / Rust NULL» 은 설명.

| 종류 | 근거(통과 = 설명 못 함 0 · 견준 것 > 0) | 견주지 않는 것 |
|---|---|---|
| oi_lsratio | DuckDB 열쇠 (ts, symbol) 값 비트 · collected_at 뺌 | 🔴그림자는 REST 기본 꺼짐 → Rust 표가 비어 «한쪽이 비었다» → 견준 0 = 불통과. 넘기려면 그림자를 `RTDUCK_SHADOW_REST=1` 로 잠깐(봉 몇 개) |
| deribit | option_trades 열쇠 trade_id · block_rfq_trades rfq_id · block_future_legs trade_id 값 비트 · gaps ±5초 분 뺌 | 체인·요약·다른 거래소 OI 표 = 그림자 REST 꺼짐이라 Rust 가 비어 «한쪽이 비었다»(세지 않음) -- 그 몫은 정답지(deribit_gex·recon)뿐 |
| iceberg | 사건 JSONL 줄 글자 집합(Rust 첫 사건 t_end 부터) | state.json(2초 벽시계) |
| market_data | microstructure_1m·_btc·_sol · tail_risk_1m 행 수(같은 구간 1% 안) | 칸 값은 진단(같은 분 칸마다 같음 비율·상대차)만 -- 그림자 REST 꺼짐이라 poll 칸이 빈다. 파이썬에만 있는 표(decision_feature_frame 등 6)는 원본 몫이 아님(적기만) |
| ob5m · bookdepth | **재생 대조**(replay_book.py · 같은 h0·같은 시 파일·같은 «지금») · 결과 파일이 2시간 안이면 다시 안 돌림 | 실시간 그림자(사슬 시작·이력에 달림) |
| raster_archive | 넘긴 래스터 표지(#takeover:raster_…)가 있고 그림자 보관 로그 끝 200줄에 오류 0 | 값 대조 없음(그림자 보관은 그림자 래스터 파일을 보관한다) -- 정답지 raster_archive_parity 뿐 |
| hl_bt(새로) | .bt 시 파일 행 바이트 묶음(거래소 ts · 재기동·끊김 빼고) + hl_ctx.sqlite(두 Rust 파일 이음) | — |
| treasury_futures · macro_5m(새로 · 한 회차) | check 가 Rust 를 그 자리에서 한 번(REST 켬 · 그림자 경로) → 같은 (symbol, ts_sec) 봉 값 비트 · 끝 10분(자라는 봉) 뺌 · 바깥 요청을 찍음 | 파이썬이 그 앞에 쌓은 봉(범위 밖) |
| liqmap_scorer · soxl_weekend(새로 · 한 회차) | 같은 열쇠(origin · weekend) 줄을 받은·쓴 시각 칸 빼고 글자 · Rust 에만 있는데 파이썬 마지막 열쇠 뒤 = «아직 안 쓴 회차»(세지 않음) · 앞 = 설명 못 함 | go 는 회차 시각 앞뒤 3분이면 거부(장부 두 줄 금지) |
| cboe_chain(+요약 · 새로 · 한 회차) | 상태 JSON 열쇠 모양(2단)이 같은지 · 파이썬 체인·요약 두 줄을 주석, Rust 줄은 체인 줄마다(같은 종목 인자) | 값(받은 순간의 호가라 두 번 받으면 원래 다르다) -- 계산 근거는 정답지 cboe_summary_parity |
| news(새로 · 상주 · 한 회차) | check 가 Rust 뉴스를 NEWS_S(180)초만 그림자로 → Rust 가 받은 (source, link) 기사가 파이썬에 같은 글자로 있나(--subset) | 파이썬이 오래 모은 기사 |

