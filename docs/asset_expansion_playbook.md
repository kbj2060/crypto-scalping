# 새 코인·주식 추가 플레이북

2026-10-31에 ETH 전용이던 대시보드를 XRP·SOL·BTC까지 넓히면서 쓴 절차다(사용자 «다음 코인이나 주식을 추가할 때 바로 진행할 수 있게»). 수집기 쪽 절차는 `scripts/ops/multicoin_collectors_20260926.py`와 메모리 `collector_consolidation_20261031`에 있고, 이 문서는 그 뒤부터(대시보드·모델·운영)를 다룬다.

## 0. 원칙 세 가지

1. **화면·데이터는 ETH와 같은 구성으로, 예측 기능은 검정을 통과한 것만 켠다.** 같은 기능이라도 코인마다 맞는지가 다르다. 10-31 결과만 봐도 4h 방향은 SOL 통과·BTC 불통과, 박스 4h는 셋 다 불통과였다. 떨어진 기능은 그 코인에서 꺼 둔 채가 맞다.
2. **새 `eth`/`xrp` 삼항을 쓰지 않는다.** 코인별 차이는 이미 있는 표에 한 줄 넣는다(아래 §2 표). 표가 없으면 기존 표 패턴(`*_ASSETS` 집합, `{coin: …}` 사전)을 따른다. 10-31 리팩토링이 이 원칙으로 분기를 표로 바꿨다.
3. **양성 대조 먼저.** 연구·워커를 새 코인으로 돌리기 전에 같은 코드로 ETH(또는 이미 통과한 코인) 숫자를 1e-9 안에서 재현한다. 재현이 안 되면 새 코인 숫자를 보지 않는다.

## 1. 순서 (한 코인에 하루)

| 단계 | 할 일 | 걸린 시간(10-31 실측) |
|---|---|---|
| ① 수집 | `multicoin_collectors_20260926.py`의 `COINS`(+`DEEP`), OKX `CT_VALS`, 테이프 `BUCKETS`·`SIZE_BANDS_USD`, `scripts/coin_config.py` `COIN_CONFIG` → `plan` 사전점검 → 켬 | 1~2시간 |
| ② 원천 대기 | 대시보드 흐름(풋프린트·1초 수급)은 테이프가 쌓이면 바로. 연구는 vision 1분봉으로 즉시 가능 | — |
| ③ 대시보드 흐름 | §2-A 표에 한 줄씩 | 1시간 |
| ④ 재학습·검정 | §3 기능마다 사전등록 → 계산 → 판정. 에이전트 4개 병렬, 로컬 12코어·15GB 기준 두 차례로 | 2~6시간 |
| ⑤ 서빙 | 통과한 기능만 §2-B 표에 한 줄 + 모델 파일 서버 복사 + cron | 1~2시간 |
| ⑥ 배포 | §4 | 30분 |

## 2. 코인을 넣는 자리

### A. 화면·흐름 (검정 없이 ETH와 같은 구성)

| 무엇 | 자리 | 값 정하는 법 |
|---|---|---|
| 대시보드에 보일 코인 | `dashboard/server.py` `DASHBOARD_ASSETS` 기본값 · `dashboard/live/app.js` `SNAPSHOT_ASSET_KEYS`·`ASSET_CONFIG` · 🔴`dashboard/live/index.html` 탭 버튼(`data-asset`, 정적 HTML) | — |
| 풋프린트·1초 수급 | `dashboard/coin_flow.py` `FLOW_SPECS`(칸·자릿수) · app.js `FLOW_ASSETS` | 칸 = ETH와 같은 상대 폭(약 2bp)이면서 테이프 칸(`BUCKETS`)과 OKX 칸의 정수배. BTC $15·SOL 0.02·XRP 0.0003 |
| 고래·리테일 경계 | `scripts/live_trade_tape_collector_20260916.py` `SIZE_BANDS_USD` | 거래량 중 리테일·고래 비중이 ETH($1만/$10만)와 같아지는 금액. vision aggTrades 4일로 잰다(ETH로 양성 대조 → $10k/$100k 재현). BTC $13k/$160k. 바꾼 뒤 그 코인 테이프 수집기 재시작 |
| OI 1초·마크 | `scripts/live_binance_ctx_collector_20261001.py` `BN_CTX_OI_SYMBOLS`·`MARK_SYMBOLS` 기본값 | 코인당 바이낸스 가중치 +240/분(한도 2400). 넘치면 폴링 간격을 늘린다(ZEC·SNDK는 5초) |
| HL 청산 묶음 | `dashboard/server.py` `HL_LIQ_BY_ASSET` | 빠지면 시장 맥락 KeyError. 파일이 없으면 칸만 빈다 |
| 일봉·4h 풋프린트 | `dashboard/footprint_daily.py` `ASSETS` · `dashboard/live/fp_daily.js` 코인 설정 · 소급 빌드 `scripts/build_footprint_daily_asset.py`(`FP_SYM=… FP_MEM=2GB FP_THREADS=3`) | 칸 = ETH $1 ≈ 4bp와 같은 상대 폭(BTC $30). 소급본 `data/footprint_daily/<SYM>_*.parquet` 은 git 밖 → `handoff.sh push`. BTC 2020~ 아카이브 55GB를 한 달씩 받고 지웠다 |
| 호가벽·가짜벽(ob5m) | `scripts/live_ob5m_20261026.py` `COINS`(틱·벽 몫·KS·idx 상한) · `scripts/ops/supervisor_ob5m.sh` 코인 목록 | 연구 `COIN_CFG`(SOL .098·5/12/25, XRP .097·8/20/40, 그 밖 ETH 기본값). `--symbol X --parity <날>`로 연구 bars 대조 불일치 0 확인 후 서버에서 `bash scripts/ops/supervisor_ob5m.sh XUSDT` + crontab `@reboot` |
| 호가 히트맵·호가 유동성(±25/50/100bp) | 🔴ob5m 과 **별개 수집기**: `scripts/live_orderflow_raster_collector_20260914.py` (`OF_SYMBOL`·`OF_BIN_SIZE`) · `multicoin_collectors_20260926.py` `DEEP`/`RASTER_TRADFI` · `dashboard/server.py` `HEATMAP_SYMBOLS` | 빈 폭 = ETH 와 같은 상대 폭(`plan` 이 계산해 crontab 줄에 고정). 과거는 재구성 불가 — 켠 시점부터 쌓인다. **호가 데이터 «3종» 점검표**: ① depth-diff(호가벽 원천) ② ob5m(기울기·벽·표식) ③ 래스터(히트맵·유동성). 코인을 열 때 셋을 따로 확인한다 |
| 주문 | `dashboard/orders.py` `MANUAL_EXEC_SYMBOLS` · app.js `RULE_ASSETS`·`VOLM_BY` | 🔴돈 걸림 — 사용자 결정. 자동 규칙 상수는 §3 검정 뒤 |

### B. 예측 기능 (검정 통과 시에만)

| 기능 | 서빙 자리 | 연구 선례(코인 인자 있음) |
|---|---|---|
| 15분·1시간 방향 | `scripts/live_eth_dir_short_20261028.py` 모형 표·`ART65`·`--coin` · `/api/dir-short?asset=` · app.js `DIRSHORT_ASSETS`·`DIRSHORT_NOTE` · 씨앗 `data/live/<c>_dirshort_oos_seed.parquet` | `scripts/research_coin_dir_short_20261031.py` (팔 X=그 코인 학습, E=ETH 학습) |
| 4h 방향 | `scripts/live_eth_dir4h_20261015.py` `ARM`·`--coin` · `/api/dir4h?asset=` · app.js `DIR4H_ASSETS`·`DIR4H_NOTE` | `scripts/research_btc_sol_xrp_dir4h_20261031.py` (단독·ETH 적용·합동) |
| 4h 변동성(QX4)·부채꼴 | `dashboard/server.py` `VOL4H_ARTIFACTS` · `scripts/live_vol4h_qx4_20261025.py` `BETA_I`(옵션 IV 보정) · app.js `VOL4H_ASSETS`·부채꼴 σ 칸 표 | `scripts/research_btc_sol_vol4h_qx4_20261031.py` · 🔴QX4 아티팩트는 4코인(ETH·BTC·SOL·XRP)만 학습돼 있다 — 새 코인은 재학습이 먼저 |
| 박스 유지 30분·1h·보정 | `dashboard/box_hold.py` `HGB`(모델 파일, git 추적)·`PLATT90`(보정 통과 코인만) · app.js `BOX_ASSETS` · 보정 `scripts/fit_box_hold_platt90_20261026.py --asset` | `scripts/research_btc_sol_box_hold_20261031.py` · 4h `scripts/research_box4h_qx4_iso_coins_20261031.py` |
| 실현 감마 | app.js `RG_BY`(분위표·경계·배수·`tip`) + `OPT_TIPS.rg_<c>` (문구는 배수에서 `rgWord`가 고른다) | `scripts/research_btcsol_feature_retest_20261031.py` §3 |
| 옵션 스마일·닿음 등고선 | `dashboard/server.py` `api_opt_smile` 지원 목록 · Deribit 사양은 `scripts/collect_deribit_option_gex_20260815.py` `SPECS`(USDC 선형은 행사가 «1d25» 표기) | Deribit에 옵션이 있는 코인만 |
| 지금 추세·7일 예측 툴팁 | app.js `TREND_NOTE`·`CB7_HIT` | `scripts/research_xrp_trend_cb7d_20261031.py` |
| 자동 규칙 상수 | (서빙 미구현 — 상수가 코인 공통) `scripts/live_manual_peg_entry_20260912.py` `RULE_STOP_PCT`·`COMMIT_CAP_X`·`REC_ADD_DIST` | `scripts/research_xrp_auto_rule_constants_20261031.py --coin` (양측 95%) |
| 개장 되돌림·맞대결·아이스버그 | ETH 전용(+일부 XRP) — 재검정은 `research_btcsol_feature_retest_20261031.py` §1·§2 | |

## 3. 재학습·검정 하는 법

- **사전등록 스킬**(`.claude/skills/prereg`)을 그대로: 문서 → 계산 전 커밋 → 계산 → 결과 문서. 등급 2단(예측 단측 90% · 돈 걸린 것 양측 95%) + «통과 후보».
- **분할**: 학습 2024~25 · 검증 2026-01~05(팔 고르기) · 시험 2026-06~(1회). 같은 구간을 몇 번째 쓰는지 «왜» 절에 공개.
- **팔 세 가지를 사전에 고정**: 그 코인 단독 학습 · ETH 학습 모형 적용 · 다코인 합동. 10-31에 고른 팔이 코인마다 달랐다(15분 BTC=단독, 1시간 BTC=ETH 적용, 4h XRP=합동).
- **경계선 표시**: 검증 IC≈0에서 고른 팔이 시험만 통과하면(4h XRP) «경계선»으로 툴팁에 적고, 피어슨 IC·모델−롱도 같이 본다.
- **에이전트 병렬**: 공통 규칙 파일 하나를 만들어 모든 에이전트에 넘긴다(10-31 판: 사전등록·분할·데이터 경로·REST 금지·스레드 3·커밋 규칙·보고 형식). 각자 `isolation: worktree`, 끝나면 메인 브랜치에 `--no-ff` 병합.
- **데이터**: 1분봉 `data/binance_vision/klines1m/<SYM>-1m-YYYY-MM.parquet` + `klines1m_daily/`(열 t,h,l,c,v,tb). 없는 코인은 vision에서 받는다(USD-M 선물 zip → 같은 모양 변환, 기존 파일과 대조 0 확인). 🔴로컬 바이낸스 REST 대량 호출 금지(서버와 IP 공유).

## 4. 배포 체크리스트

1. 바뀐 경로만 명시해 커밋(`-a` 금지). `node --check dashboard/live/app.js` · `test/test_market_ctx_patterns_20260929.mjs` · 관련 파리티 시험.
2. `HANDOFF_HOSTS_FILE=/home/kbj20/crypto-scalping/scripts/ops/handoff.hosts.conf bash scripts/ops/check_deploy_drift.sh` 종료코드 0.
3. main에 ff 푸시 → CI `syntax-check` 성공 → 워처(10분)가 서버 반영 → 서버 `git rev-parse` 와 서빙 바이트(`curl …/app.js | grep`)로 확인. 대시보드 재시작 직후 1~2분은 응답이 없다(기동 중).
4. git 밖 파일(모델 joblib·풋프린트 parquet)은 메인 체크아웃에 복사한 뒤 `HANDOFF_RSYNC_Z=0 bash scripts/ops/handoff.sh push server <경로>`. 서버 같은 이름 파일이 있으면 먼저 md5 대조.
5. 수집기 재시작은 감독 프로세스는 두고 **자식 PID만** `kill -TERM`(`pkill -f`는 `_supervise.sh`까지 죽인다).
6. crontab: 백업(`crontab -l > data/live/backups/crontab_before_<무엇>_<날짜>.txt`) → 수정 → 백업과 diff로 바뀐 줄만 확인. 🔴같은 panel parquet을 쓰는 재학습 줄은 ETH와 **같은 잠금을 기다리게**(`flock /tmp/dir4h_train.lock`, `-n` 개별 잠금 금지).
7. 감시기 항목: 서버 `data/live/multicoin_collectors.json`(저장소 밖)의 `fresh`에 그 코인 테이블 신선도를 넣는다(백업 먼저).

## 5. 주식(TradFi) 메모

- 바이낸스 TradFi 무기한(예: SNDK)은 코인과 같은 경로로 수집된다. 다른 점: 장 마감 시간대 체결이 끊겨 신선도 기준을 넓혀야 하고, HL 포지션 대상이 아니며, 크기 구간 표본이 짧다(평일에 다시 잰다). 10-31 ZEC·SNDK 커밋 58e7399a·6f31751b 참고.
- 학습 데이터 기간이 짧으면 «학습 2024~25» 분할을 그대로 못 쓴다 — 사전등록 «구간» 절에 바꾼 분할과 이유를 적는다. 연속 거래가 아니므로 4h 변동성·박스처럼 시간 창을 쓰는 기능은 장중 봉만으로 다시 정의해야 할 수 있다.

- **대시보드 연결(11-01 SNDK, 사용자 «SNDK 선물 풋프린트만 있어도 돼»)**: §2-A 표에서 건드린 곳 = `scripts/coin_config.py` `COIN_CONFIG`(워커 없는 키는 «파일 없음 = 빈 응답») · `dashboard/live/index.html` **탭 버튼 `data-asset`**(🔴탭은 정적 HTML — 빠뜨리면 서버 목록에 있어도 안 보인다. 11-01 SNDK 가 실제로 이렇게 안 보였다) · `dashboard/server.py` `DASHBOARD_ASSETS` 기본값·`HL_LIQ_BY_ASSET`(없는 파일) · `dashboard/coin_flow.py` `FLOW_SPECS`(칸 $0.2 = 테이프 칸 0.05의 4배 · `spot=False`) · `app.js` `ASSET_CONFIG`·`FLOW_ASSETS`·`SNAPSHOT_ASSET_KEYS`. 예측 기능 Set(DIR4H·BOX·VOL4H 등)과 주문 표는 넣지 않았다 → 칸이 꺼지고 주문은 막힌다.
  - ~~🔴`MARKET_SYMBOLS`(orders.py)는 건드리지 않는다~~ — **2026-11-03 사용자 «둘 다 열어줘»로 주문을 열면서 `orders.MARKET_SYMBOLS`·`MANUAL_EXEC_SYMBOLS` 에 sndk·soxl 을 넣었다**(§5-2). 처음엔 계좌 조회가 심볼마다 `userTrades`·`openOrders` 를 더 불러 실계좌 호출이 늘기 때문에 `server.py` 사본에만 두었다 — 이제 그 호출 증가를 받아들인 것이다(IP 한도는 아직 못 쟀다).
  - 현물이 없는 종목은 `FlowSpec.spot_url=None` — `collect_spot_flow` 를 아예 띄우지 않는다(없으면 존재하지 않는 스트림에 재접속을 반복).
  - 미연결: HL(해당 없음). 일봉 풋프린트는 11-03 연결(SNDK 2026-04-07~·SOXL 2026-05-15~, `build_footprint_daily_asset.py` 의 `FP_SYM=SNDKUSDT|SOXLUSDT`, OI 는 패널 없이 metrics 아카이브만 · 칸 SNDK $0.5 · SOXL $0.03) · 4h 풋프린트는 수집기 이력(10-31·11-02~)만 · 옵션 칸은 CBOE(11-01~).

- **SOXL(11-02, 사용자 «soxlusdt 선물 대시보드에 추가»)**: SNDK 와 같은 자리에 한 줄씩(위 목록 + `live_okx_trade_tape` `BUCKETS`·`CT_VALS` + 테이프 `BUCKETS`·`SIZE_BANDS_USD` + `multicoin_collectors` `COINS`·`TRADFI` + ctx `OI_SLOW`·`MARK`). 칸 $0.02(테이프 0.01의 2배) · 크기 구간 잠정 $3.9k/$36k(aggTrades 6,000건·0.74h) -- 평일에 다시 잰다 · 깊은 호가 diff 는 SNDK 만(`DEPTH_TRADFI`; **SOXL 은 11-03 수동으로 시작** §5-2) · 옵션 카드는 11-02 연결(CBOE 체인 SOXL 2,872계약 수신 확인 · `write_state` 를 종목별 병합으로 고침 · cron 인자 `SNDK SOXL` · app.js `optData` 표).

- **SKHYNIX·MU(11-13, 사용자 «꾸준히 거래량 많은 주식 선물 2개»)**: 60일 평일 거래대금으로 골랐다(SKHYNIX 중앙 $11.6억·하위10% 5.3억 · MU 6.1억·3.6억, 서버 일봉 조회). SOXL 자리 그대로 + 🔴**`dashboard/orders.py` `MARKET_SYMBOLS`(시세 표)에도 넣어야 한다** — 서버 `DASHBOARD_ASSETS` 가 이 표에 없는 코인을 조용히 빼서 탭은 보이는데 ETH 봉이 그려졌다(553e99b0). 주문 표(`MANUAL_EXEC_SYMBOLS`)는 안 넣음 = 주문 막힘. 서버: 래스터 2 start(--only) · 묶음 3개 crontab 목록 sed + 감독자·자식 종료 후 같은 줄로 재기동 · 매니페스트 묶음 env 갱신 · ctx 자식만 재시작(백업 data/live/backups/*_20261113_skhynix_mu). 크기 구간은 짧은 표본을 ETH 같은 표본 값으로 나눠 환산한 잠정값. 미연결: 일봉 풋프린트 소급·호가벽 워커·청산 lake.
- **SKHYNIX·MU 2차(11-13 «전부 채우기», 13abfa10 등)**: 일봉 풋프린트 소급(MU 219일 2026-04-07~ · SKHYNIX 163일 06-02~ · 칸 $0.3·$0.4 · 일봉 대조 중앙오차 1e-13, 최대 10-27 하루 2~9% = 아카이브 쪽) → `handoff.sh push` md5 일치 · 🔴**빌더 기본 작업 폴더가 공용(`tmp/fp_daily_build` = ETH 조각)이라 새 종목이 ETH 조각을 «끝난 기간»으로 건너뛰어 ETH 를 내보냈다**(대조 assert 가 잡음) → 이제 종목별 폴더가 기본 · 고래·리테일 경계는 vision aggTrades 평일 4일로(MU $700/$5,400 · SKHYNIX $700/$4,300, ETH 09-22~25 대조 $10k/$100k 재현) · 계좌 체결·미체결 조회에서 시세 전용 종목 제외(`orders.MARKET_ONLY_ASSETS`) · ob5m 표·감독 허용·`DEPTH_TRADFI` 는 코드만 — 🔴서버 기동(깊은 호가 2·ob5m 2·crontab 4줄)과 테이프 묶음 재기동(경계 반영)은 권한 분류기가 막아 사용자 실행 대기: 서버 `/tmp/add_stocks_server2.sh`.
- **주식 옵션은 자동(11-13, 사용자 «주식류는 이런식으로»)**: 새 주식 코인은 `scripts/coin_config.py` 에 `"equity": True` 만 넣으면 CBOE 수집기(cron 인자와 무관하게 equity 코인 전부)가 체인을 모으고 요약기가 옵션 카드(오늘 체결 · 거래량·평소 대비 · 풋/콜 비중 · 감마 노출 주요 레벨)를 채운다. app.js `optData` 는 표 없이 서버 묶음에 그 코인이 있으면 보인다. 평소 대비는 스냅샷이 2거래일 넘게 쌓인 뒤부터.

### 5-1. 주식·ETF 옵션 (11-01 조사, 사용자 «주식 옵션 데이터를 다른 소스로»)

Binance 옵션은 BNB·BTC·DOGE·ETH·SOL·XRP·XAU·XAG 뿐이고 Deribit 은 ETH·BTC 뿐이라 **주식·ETF 옵션은 미국 거래소(OPRA)** 에서 받는다. ETH 의 Deribit 수집(체결 전체 + 체인 10분 + 요약)과 비교한 현황:

| 데이터 | 소스 | 상태 |
|---|---|---|
| 체인 스냅샷(호가·IV·그릭스·OI·거래량·마지막 체결) | CBOE 지연 시세 JSON `https://cdn-api.cboe.com/api/global/delayed_quotes/options/<SYM>.json` (키 없음, `cdn.cboe.com` 은 307) | **켬(11-01)** `scripts/live_cboe_option_chain_collector_20261101.py` SNDK · cron 10분 · 원본 gz 저장 ~0.7MB/회 |
| 옵션 체결 전체·블록 | OPRA 유료 공급자 | **필요(사용자 11-01)** · 공급자·약관 미정 — 아래 |

- **저장**: `data/live/options_equity/cboe/<SYM>/<UTC날짜>/<HHMMSS>.json.gz`. 응답의 `seqno` 가 직전과 같으면 저장하지 않는다(주말·휴장·장외, 달력 불필요). 종목 추가 = cron 줄 인자에 심볼.
- **용량**: SNDK 11,816계약·23만기·5.2MB → gz 0.7MB. 장중 6.5h × 6회 ≈ 27MB/일 ≈ 10GB/년/종목. 종목 N개면 N배.
- **🔴한계**: 공식 개발자 API 가 아니라 사이트 주소다. **자동 수집 약관 허용 여부 미확인**, 막히거나 바뀌면 조용히 멈춘다(cron 로그의 `실패` 줄만 남는다 -- 감시기 항목 없음). 시세는 지연이고 체결 한 건 한 건은 없다.
- **유료 공급자(2026-11-01 공식 페이지 조사, 가격은 가입 화면에서 확인 필요)**
  | 공급자 | 체결 한 건 한 건 | 가격·약관 확인된 것 |
  |---|---|---|
  | Databento OPRA.PILLAR | `trades` 스키마(체결 시각·거래소, **side 는 항상 N** — OPRA 가 안 준다) · `tbbo`(체결 시점 NBBO) · `mbp-1` | Standard 월 $199(개인·소규모), 실시간은 건수당 사용료 + 라이선스 통과 요금. **OPRA 별도 가격 페이지에서 못 찾음** |
  | Massive(구 Polygon) | `GET /v3/trades/{옵션티커}` 가격·수량·거래소·체결 조건 | 옵션 요금제·가격 페이지에서 못 찾음 |
  | Market Data(marketdata.app) | 체결 엔드포인트 언급 없음(호가·체인만) → **체결 전체 목적엔 부적합** | 무료 24h 지연 · Starter $30 15분 지연 · Trader $75 실시간 OPRA(비전문가) |
  OPRA 는 체결의 매수·매도 쪽(side)을 주지 않는다 — 공격적 매수/매도는 직전 호가와 비교해 추정해야 한다(틱 규칙).

## 6. 10-31 판정 기록 (다음 코인의 기대치)

| 기능 | BTC | SOL | XRP | 문서 |
|---|---|---|---|---|
| 15분·1시간 방향 | ✅ | ✅ | ✅ | docs/experiments/coin_dir_short_btc_sol_20261031.md · xrp_dir_short_20261031.md |
| 4h 방향 | ❌ | ✅ | ✅ 경계선 | btc_sol_xrp_dir4h_20261031.md |
| 4h 변동성 / 옵션 IV판 | ✅ / ✅ | ✅ / 데이터 부족 | ✅ / 데이터 부족 | btc_sol_vol4h_qx4_20261031.md |
| 박스 30분·1h / 1h 보정 / 4h | ✅ / ✅ / ❌ | ✅ / ❌ / ❌ | ✅ / ✅ / 후보 | btc_sol_box_hold_20261031.md · btc_sol_box4h_qx4_iso_20261031.md |
| 실현 감마 | ✅ | ✅ | ✅ | btcsol_feature_retest_20261031.md · xrp_realized_gamma_regime_20261031.md |
| 자동 규칙 상수 | 변경 권고(미적용) | 현행 | 현행 | btc_sol_auto_rule_constants_20261031.md · xrp_auto_rule_constants_20261031.md |
| 개장 되돌림 / 맞대결 | 후보 / ❌ | ❌ / ❌ | ❌ / ❌ | btcsol_feature_retest_20261031.md |

대략 «짧은 지평 방향·변동성·박스 30분/1h·실현 감마»는 코인이 바뀌어도 통과했고, «4h 방향·박스 4h·매매 신호»는 코인마다 갈렸다.

> **주식 옵션 카드(2026-11-01)**: Deribit 에 없는 종목은 CBOE 체인 요약(`scripts/cboe_option_summary_20261101.py`)을 `data/live/cboe_option_state.json` 으로 쓰고 `gex_payload` 가 `currencies.<SYM>` 으로 합친다. app.js `optData` 코인 표에 한 줄 추가. rv7·블록 거래는 없다.


### 5-2. 2026-11-03 주식(SNDK·SOXL) 격차 메우기 기록 (사용자 «이더리움과 같은 상태로» → «채울 수 있는 것 모두»)

한 일(커밋 순):
| 무엇 | 어디 | 비고 |
|---|---|---|
| **주문 열기** | `dashboard/orders.py` `MARKET_SYMBOLS`·`MANUAL_EXEC_SYMBOLS` 에 `sndk`→SNDKUSDT · `soxl`→SOXLUSDT, `server.py` 사본 제거 · 시험 `test/test_stock_exec_symbols_20261102.py` | 주식은 USDC 시장이 없어 **주문 심볼 = 시세 심볼**(basis 1). 거래소 규칙 조회(서버, 읽기): 두 심볼 step/tick/최소수량 0.01 · 최소주문금액 5. 🔴**실주문 시험은 한 번도 안 했다** — 장외 시간 체결·호가 거부는 미확인. 규칙(`RULE_ASSETS`)·ETH 위험 모델은 안 붙인다(증거금 50% 단순 규칙) |
| **옵션 카드 빈 칸** | `scripts/cboe_option_summary_20261101.py` · app.js `optFlowTodayHtml`·`optEqFlowSvg` · `OPT_TIPS.eqflow/eqflow2` | ① 실현 7일 = 바이낸스 선물 1시간봉(6시간 캐시 `data/live/cboe_rv7_cache.json`, **서버 cron 에서만** 바이낸스 호출) ② «블록 거래» 자리 = 오늘 체결 요약(콜·풋 계약·명목, 거래량 상위 계약) ③ «순매수 흐름» 자리 = 콜·풋 체결 누적 곡선(수집기가 10분마다 한 점 -- 배포 직후엔 점이 적다). 체결 한 건·블록·매수/매도는 CBOE 가 안 준다 |
| **일봉 풋프린트 소급** | `build_footprint_daily_asset.py`(SNDKUSDT·SOXLUSDT, `FP_FIRST` 상장월) · `footprint_daily.py` `ASSETS` · `fp_daily.js` | SNDK 208일(2026-04-07~) · SOXL 170일(05-15~). OI 패널이 없는 코인은 일별 metrics 아카이브만. 칸 SNDK $0.5 · SOXL $0.03 |
| **호가벽·가짜벽(ob5m)** | `scripts/live_ob5m_20261026.py` `COINS` · `supervisor_ob5m.sh` 허용 목록 | 틱 $0.01 · 벽 몫·KS 는 ETH 기본값 — **연구 대조(`--parity`) 없음 = 잠정**. 서버에서 SNDK·SOXL 워커 시작(11-03 02:00), SOXL 호가 diff 수집기도 시작(`DD_SYMBOL=soxlusdt supervisor_depth_diff.sh`) |
| **호가 히트맵·호가 유동성(래스터)** | `multicoin_collectors_20260926.py` `RASTER_TRADFI=(SNDK,SOXL)` · `server.py` `HEATMAP_SYMBOLS` | 11-03 19:38 서버 기동(`start --phase 2 --coins SNDK,SOXL`, install-cron 없이). 빈 폭 SNDK $0.25 · SOXL $0.02. 🔴**11-03 사용자 «SOXL 은 왜 호가 데이터를 수집하지 않지?»** — 처음 채울 때 depth-diff·ob5m 만 켜고 이 수집기를 빠뜨렸다(호가벽은 이미 나왔고 히트맵·유동성 카드만 빈 것이었다). 원인 진단 순서: 프로세스 → `/api/ob5m` 봉 수 → `HEATMAP_SYMBOLS` |

**해야 하는데 안 된 것(권한 분류기가 막음 → 사용자 수동)**: crontab `@reboot` — ① SOXL 호가 diff 수집기·SNDK/SOXL ob5m 워커 3줄은 11-03 사용자가 추가함 ② **래스터 SNDK·SOXL 2줄은 아직** (`multicoin_collectors_20260926.py plan --phase 2 --coins SNDK,SOXL` 이 출력하는 줄 그대로) — 재부팅하면 래스터 둘이 안 켜진다. 백업 `data/live/backups/crontab_before_stock_ob5m_20261103.txt`(85줄). 줄 형태는 SNDK 호가 diff 줄(`multicoin_bn_depthdiff_sndk_reboot`)과 ETH/XRP ob5m 줄(`sleep 75~85 && … supervisor_ob5m.sh <SYM>`)을 따른다.

**ETH 와 아직 다른 것(이유)**:
| 기능 | 상태 | 이유·다시 볼 때 |
|---|---|---|
| 평소 대비 z 점수(델타·고래·OI) | SOXL 비어 있음 · SNDK 일부(z60h 비어 있음) | 기준선을 쌓을 체결 이력이 11-02(SOXL)·10-31(SNDK)부터. 시간이 해결 |
| 일 분위(OI 24h·베이시스 7일) | SOXL 비어 있음 | 같음 |
| 4h 풋프린트 | 수집기 이력만 | 4h 는 소급 원천이 없다(테이프 기반) |
| 15분·1h·4h 방향 · 4h 변동성 · 박스 | **없음** | 상장 이력 SNDK 210일 · SOXL 172일 — 사전등록 분할(학습 2024~25)을 못 쓴다. 다시 볼 시점: **SNDK 2027-04 · SOXL 2027-05**(이력 ~1년) |
| 실현 감마 | «검증 안 됨» 표시 | 2021~24 기준 분포가 없다. 같은 시점에 다시 |
| 블록 거래·순매수 방향 | 못 본다 | CBOE 가 체결 한 건·방향을 안 준다 → OPRA 유료 공급자(§5-1) |
| HL 고래 청산가 · 현물 | 해당 없음 | 주식 선물은 HL·현물 대상이 아님 |

**교훈**
- 새 종목의 «일봉 소급»은 상장일부터 월 zip 이 있으면(SNDK 4월 17MB~SOXL 6월 237MB) 로컬에서 몇 분이면 끝난다 — 모델·검정과 달리 이력 길이가 걸림돌이 아니다.
- 서버 `main` 체크아웃에 다른 세션이 직접 커밋하면 감시기의 `--ff-only` 가 막힌다(11-03 3개 갈라짐). 서버 커밋을 별도 브랜치로 푸시해 origin/main 에 병합하면 서버가 그대로 빨리감기로 따라온다(서버를 되돌릴 필요 없음). 배포 후 `git rev-list --left-right --count HEAD...origin/main` 이 `0 0` 인지 본다.
- 바이낸스 URL 이 들어간 명령은 로컬 셸 가드가 막는다 — Edit 도구로 파일을 고치고, 호출은 서버에서(`handoff.sh launch server`).
