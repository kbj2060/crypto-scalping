# rtcollect 그림자 운영 (2026-11-21)

Rust 수집기(`rtcollect`)를 서버에서 **파이썬 수집기 옆에 나란히** 돌려, 같은 거래소 스트림으로 만든 행이 같은지 매일 대조한다.
파이썬 수집기·파이썬 파일은 건드리지 않는다. 전환(파이썬을 끄고 Rust 가 원래 파일을 이어받음)은 7일 대조를 통과한 뒤 사용자 승인으로.

## 1. 실행 파일 두기

- 개발기에서 빌드한다(서버에서 빌드하지 않는다 -- 라이브 서버 CPU): `cd rtcollect && cargo build --release -j 2` → `rtcollect/target/release/rtcollect`.
- 서버와 개발기가 같은 Ubuntu · glibc 2.39 · x86_64 라 그대로 돈다. SQLite 는 묶어서(bundled) 들어 있고, TLS 는 rustls(OpenSSL 없음).
  옮기기 전에 서버에서 `ldd --version | head -1` 이 2.39 인지 본다.
- 서버 위치: `<저장소>/bin/rtcollect`(감독 스크립트 기본값, 다른 곳이면 `RTCOLLECT_BIN`). 빌드 산출물이라 git 에 넣지 않는다.
  옮긴 뒤 `md5sum` 을 개발기 것과 맞춘다(배포 규칙: 덮어쓰기 전 대조).

## 2. 종류별 환경변수

| 종류(`RTCOLLECT_KIND`) | 심볼 | 쓰는 파일(기본) | 원본 파이썬 파일 |
|---|---|---|---|
| `binance_tape` | `TAPE_SYMBOL`(쉼표) · `TAPE_MARKET=futures` | `data/hot/rtcollect/binance_tape.sqlite` | `data/hot/binance_tape.sqlite` |
| `binance_tape` 현물 | `TAPE_SYMBOL` · `TAPE_MARKET=spot` | `data/hot/rtcollect/binance_spot_tape.sqlite` | `data/hot/binance_spot_tape.sqlite` |
| `okx_tape` | `OKX_TAPE_INST`(쉼표) | `data/hot/rtcollect/okx_tape.sqlite` | `data/hot/okx_tape.sqlite` |
| `bybit_tape` | `BYBIT_TAPE_SYMBOLS` | `data/hot/rtcollect/bybit_tape.sqlite` · `…/bybit_ctx.sqlite` | `data/hot/bybit_tape.sqlite` · `bybit_ctx.sqlite` |
| `bybit_liq` | `BYBIT_LIQ_SYMBOLS` | `data/hot/rtcollect/bybit_liq.sqlite` | `data/hot/bybit_liq.sqlite` |
| `okx_ctx` | `OKX_CTX_INST`(쉼표 -- 청산 전 종목 구독은 ETH 종목만) | `data/hot/rtcollect/okx_ctx.sqlite` | `data/hot/okx_ctx.sqlite` |
| `binance_ctx` | `BN_CTX_MARK_SYMBOLS` · `BN_CTX_OI_SYMBOLS` · `BN_CTX_OI_SLOW_SYMBOLS` | `data/hot/rtcollect/binance_ctx.sqlite` | `data/hot/binance_ctx.sqlite` |
| `xyz` | `HL_XYZ_COINS`(글자 그대로) | `data/hot/rtcollect/hl_xyz.sqlite` | `data/hot/hl_xyz.sqlite` |
| `hl_ctx` | `HL_BT_COIN`(쉼표 -- 원본은 코인마다 프로세스·`HL_BT_CTX_ONLY=1`) | `data/hot/rtcollect/hl_ctx.sqlite` | `data/hot/hl_ctx.sqlite` |
| `raster` | `OF_SYMBOL`(한 종목) · `OF_BIN_SIZE`·`OF_N_BINS`(원본과 같은 이름 · 기본 0.5·240 -- 종목마다 crontab 줄 값 그대로, `tools/shadow_start.sh` 가 읽는다) · 🔴`RTCOLLECT_REST=1` 필수(스냅숏 없이는 북을 못 세운다) | 폴더 `data/live/rtcollect/orderflow/raster/<SYM>/` | `data/live/orderflow/raster/<SYM>/`(폴더를 가리키면 종료코드 2) |
| `hl_bt` | `HL_BT_COIN`(한 코인 -- 원본과 같다) · `HL_BT_ROOT` | 폴더 `data/live/rtcollect/orderflow/hyperliquid_bookticker/<COIN>/<UTC 시>.bt` + 맥락 `data/hot/rtcollect/hl_ctx.sqlite` | `data/live/orderflow/hyperliquid_bookticker/`(폴더를 가리키면 종료코드 2) -- 원본 전체 모드는 Pi 에서 돈다 · hl_ctx 와 같이 뜨면 hl_ctx 는 `data/hot/rtcollect/hl_ctx_only.sqlite`(`HL_CTX_DB_PATH`) -- 대조는 두 파일을 합쳐 파이썬 hl_ctx.sqlite 와 |
| `iceberg` | `ICEBERG_SYMBOL`(ETHUSDT·XRPUSDT) · `DD_ROOT`·`TAPE_DB_PATH`(읽기만) | `data/live/rtcollect/iceberg/<coin>_{events.jsonl,state.json}`(`ICEBERG_OUT`) | `data/live/iceberg/`(가리키면 종료코드 2) -- 거래소 호출 없음 · 그림자 기본으로 띄운다 |
| `treasury_futures` · `macro_5m` · `cboe_chain` | 한 번 돌고 끝(cron) · `--range 1d|5d` · cboe 는 인자 종목 + SNDK·SOXL·MU | `data/hot/rtcollect/{treasury_futures,macro_5m}.sqlite`(`TREASURY_FUTURES_DB`·`MACRO_5M_DB`) · `data/live/rtcollect/options_equity/cboe`(`CBOE_OUT`) | 파이썬 파일·폴더면 종료코드 2 -- 🔴그림자 cron 은 등록하지 않는다: Yahoo 는 서버 국채선물(매분 4)·매크로(5분 16)와 같은 IP 라 그림자를 켜면 분당 Yahoo 호출이 ~7 → ~14(429 위험 · 원본 주석) · CBOE 는 10분 3 → 6 |
| `ob5m` · `bookdepth` | `OB_SYMBOL`(ETHUSDT) · `DD_ROOT`(depthdiff 폴더 · 읽기만) · ob5m 은 `TAPE_DB_PATH`(hot 테이프 · 읽기만) | `data/live/rtcollect/<종류>/<SYM>.sqlite` | `data/live/ob5m/`·`data/live/bookdepth/` 파일(`RTCOLLECT_TAKEOVER=1` 없으면 종료) -- 대조는 같은 시각의 닫힌 봉 행을 두 DB 에서 비교(bookdepth dm·dp 계열은 1e-12 · BLAS 내적 차) |
| `depth` | `DD_SYMBOL`(한 종목 -- 원본과 같다) | 폴더 `data/live/rtcollect/orderflow/depthdiff/<SYM>/` | `data/live/orderflow/depthdiff/<SYM>/`(폴더를 가리키면 종료코드 2) |

- 심볼 목록은 **지금 도는 파이썬과 같게**: `tr '\0' '\n' < /proc/<파이썬 pid>/environ | grep -E 'TAPE_SYMBOL|TAPE_MARKET|OKX_TAPE_INST|BYBIT_'`.
- 경로를 바꾸려면 `TAPE_DB_PATH` · `OKX_TAPE_DB_PATH` · `BYBIT_TAPE_DB` · `BYBIT_CTX_DB` · `BYBIT_LIQ_DB` · `OKX_CTX_DB_PATH` · `HL_XYZ_DB`(파이썬과 같은 이름)
  · `BN_CTX_DB`(바이낸스 맥락 -- 파이썬은 경로 변수가 없다).
- 바이낸스 맥락의 청산 종목 = `scripts/coin_config.py` COIN_CONFIG(실행 파일에 9개를 적어 두었다 -- 바뀌면 `binance_ctx.rs` 도) ∪ `BN_CTX_OI_SLOW_SYMBOLS`.
  🔴파이썬 파일(`data/hot/<이름>`)을 가리키면 실행 파일이 종료코드 2 로 거부한다(`RTCOLLECT_TAKEOVER=1` 일 때만 연다 -- 전환 전용).
- `RTCOLLECT_REST`: 비워 둔다(꺼짐). 아래 §6.
- 띄우기: 종류마다 `RTCOLLECT_KIND=<종류> <심볼 변수>=<목록> bash scripts/ops/supervisor_rtcollect.sh`(머리 주석의 crontab 줄 -- 등록은 사람).
  로그 `logs/supervisor/rtcollect_<KEY>_YYYYMMDD.log`. 같은 종류·겹치는 심볼이 이미 돌면 켜지지 않는다.

## 3. 디스크

같은 행을 쓰므로 **하루 증가량 = 지금 파이썬 hot 파일의 하루 증가량**과 같다(뷰·인덱스·PRAGMA 도 같다).

- 근거: 테이프 표(`WITHOUT ROWID`, 19칸)는 개발기 실측 **행당 약 100바이트**(31만 행 31.5MB, 합성 값). ETH 선물은 2026-09-16 실측
  하루 약 32.6만 행(0.1달러 칸) → **ETH 하나 약 33MB/일**. 다른 코인은 행 수가 더 적다(칸 폭이 가격의 0.4bp 로 같고 거래가 적다).
- 맥락 셋(개발기 실측 행당 크기, 표+인덱스, 합성 값): OKX 마크 67B(종목마다 초당 약 5행 = **종목당 약 29MB/일** -- 원본 주석의 «~10MB/일»보다 크다) ·
  OKX OI 84B(5~10초 간격 = 약 1MB/일) · 바이낸스 마크 73B(종목마다 초당 1행 = 6.3MB/일 × 9 = **약 57MB/일**) · 바이낸스 OI 48B(켰을 때만, 종목당 1MB/일 안팎) ·
  xyz ctx 74B(종목마다 초당 1행 = 6.4MB/일 × 4 = **약 26MB/일**, 주말·얇은 종목은 적다) · 청산·펀딩·1분봉은 작다.
- 서버에서 실제 행 수로 다시 재는 법(읽기 전용):
  `python3 -c "import sqlite3,time; c=sqlite3.connect('file:data/hot/binance_tape.sqlite?mode=ro',uri=True); print(c.execute('SELECT symbol,count(*) FROM trade_tape_1s WHERE ts_sec>=? GROUP BY 1',[int(time.time())-86400]).fetchall())"`
  → 행 수 × 100바이트. OKX·Bybit·현물 파일도 같은 식. `multicoin_collectors_20260926.py check --hours 24` 의 «MB/일» 칸도 같은 값이다.
- 파이썬 hot 파일은 봉인기(`seal_to_lake` prune_hot, 16일)가 줄이지만 **그림자 파일은 아무도 안 줄인다** -- 7일이면 위 값 × 7. 끝나면 지운다(§8).
- `data/hot/rtcollect/` 는 백업(`backup_live_data.sh` 의 `data/hot/*.sqlite`)·봉인·데이터 지도(`gen_data_map`)에 안 잡힌다(하위 폴더라).

## 4. 거래소 연결

파이썬과 **같은 수만큼 더** 붙는다: 바이낸스 선물 심볼마다 1 · 현물 심볼마다 1 · OKX 종목마다 1 · Bybit 테이프 1 · Bybit 청산 1 ·
OKX 맥락 종목마다 1 · 바이낸스 맥락 마크 종목마다 1 + 강제청산 1 · xyz 1(Hyperliquid).
지금 목록(ETH + `multicoin_collectors` COINS 9개)이면 바이낸스 선물 10 + 현물(목록 수) + OKX 10 + Bybit 2 + OKX 맥락(목록 수) + 바이낸스 맥락 10 + HL 1.
바이낸스 웹소켓 한도는 IP 당 «5분에 새 연결 300»이라 재시작 폭주가 아니면 여유가 있다. REST 는 기본 0회(§6).

## 5. 매일 대조

```
python3 rtcollect/tools/shadow_tape.py data/hot/binance_tape.sqlite      data/hot/rtcollect/binance_tape.sqlite      --from $(date -d '1 day ago' +%s) --to $(date +%s)
python3 rtcollect/tools/shadow_tape.py data/hot/binance_spot_tape.sqlite data/hot/rtcollect/binance_spot_tape.sqlite --from … --to …
python3 rtcollect/tools/shadow_tape.py data/hot/okx_tape.sqlite          data/hot/rtcollect/okx_tape.sqlite          --from … --to …
python3 rtcollect/tools/shadow_tape.py data/hot/bybit_tape.sqlite        data/hot/rtcollect/bybit_tape.sqlite        --from … --to …
python3 rtcollect/tools/shadow_tape.py data/hot/bybit_ctx.sqlite         data/hot/rtcollect/bybit_ctx.sqlite         --from … --to …
python3 rtcollect/tools/shadow_tape.py data/hot/bybit_liq.sqlite         data/hot/rtcollect/bybit_liq.sqlite         --from … --to …
python3 rtcollect/tools/shadow_tape.py data/hot/okx_ctx.sqlite           data/hot/rtcollect/okx_ctx.sqlite           --from … --to …
python3 rtcollect/tools/shadow_tape.py data/hot/binance_ctx.sqlite       data/hot/rtcollect/binance_ctx.sqlite       --from … --to …
python3 rtcollect/tools/shadow_tape.py data/hot/hl_xyz.sqlite            data/hot/rtcollect/hl_xyz.sqlite            --from … --to …
python3 rtcollect/tools/shadow_tape.py data/hot/hl_ctx.sqlite            data/hot/rtcollect/hl_ctx.sqlite            --from … --to …
python3 rtcollect/tools/shadow_tape.py --raster data/live/orderflow/raster/ETHUSDT data/live/rtcollect/orderflow/raster/ETHUSDT --hours 2026-11-22T00,2026-11-22T23
python3 rtcollect/tools/shadow_tape.py --depth data/live/orderflow/depthdiff/ETHUSDT data/live/rtcollect/orderflow/depthdiff/ETHUSDT --hours 2026-11-22T00,2026-11-22T23
```
depth 는 시 파일을 풀어 **거래소 프레임 줄을 글자 그대로** 맞춘다(스냅숏 줄은 두 쪽이 다른 순간에 REST 를 불러 원래 다르고 Rust 는 기본 null · `_gap` 줄은
연결 시각에 달렸다 -- 센 수만). 통과 = 날마다 «파이썬에만 0 · Rust에만 0», 단 두 쪽 로그의 끊김 시각 ±5초 안 프레임은 뺀다(제안).
래스터는 바이트 일치를 기대하지 않는다(두 프로세스가 스냅숏을 다른 순간에 받고 틱이 각자 벽시계다) -- «두 쪽 유효 초의 99% 이상 bin_lo·mid 같음 · 칸 같음 비율 중앙 0.95 이상»(제안).
값 일치는 개발기 정답지(원본과 같은 사건 열 → 파일 바이트 같음)가 보인다.
`«Rust에만 N(의도한 차이 후보 -- …)»` 는 §11 의 고친 결함으로 Rust 가 더 쓴 행이다 -- 불일치로 세지 않는다(그 수는 적는다).
표준 라이브러리만 · 두 파일 다 읽기 전용(`mode=ro`) · 한 시간 창씩 읽는다(큰 표를 통째로 안 올린다).
빼는 분: 두 쪽 gaps ±5초 · 두 쪽 backfill_1m ok 분 · 두 쪽 verify_1m 불일치 분 · 범위 끝 10초. 뺀 (종목, 분) 수를 같이 찍는다.

**통과 기준(7일 연속, 날마다)**: 표마다 «비교 N행 · 불일치 0 · 파이썬에만 0 · Rust에만 0», 그리고 «뺀 (종목,분)» 을 같이 적는다(분모).
뺀 분이 하루 1% 를 넘으면 그날은 통과로 세지 않는다(재연결이 잦다는 뜻 -- 원인을 먼저 본다).
청산(`bybit_liquidations`)은 사건이 드물어 비교 행이 0 인 날이 있다 -- 7일 합으로 0 이 아니어야 한다.
불일치가 나오면 `--show 20` 으로 행을 보고, 그 분이 한쪽의 재연결·REST 교체와 겹치는지 먼저 본다.

맥락 셋의 예외(원본 구조 때문 -- 아래 숫자 기준은 **제안**이다, 사용자 확인 전):
- **바이낸스 맥락**은 gaps 표가 없다(원본에 없다). 재연결 언저리의 «한쪽에만» 행은 두 쪽 로그의 `ws:`(끊김) 시각 ±5초 안이면 센다에서 뺀다.
  `oi_1s` 는 Rust 가 REST 를 꺼 두면 «한쪽이 비었다»가 정상이다. 마크·청산은 «불일치 0»이어야 한다(같은 (ts, symbol) 은 먼저 온 값).
- **xyz** `hl_xyz_ctx_1s` 의 recv_sec 는 받은 시각(벽시계)이라 비교에서 뺀다. 받은 초가 1칸 밀리면 같은 초로 합쳐지는 행 수가 갈린다 --
  «파이썬에만 + Rust에만»이 비교 행의 0.5% 아래면 통과로 센다. `hl_xyz_1m` 은 «불일치 0»(진행 중 봉은 범위 끝 10초가 뺀다).
  🔴단 끊김을 지난 분은 다를 수 있다: 두 쪽 다 끊길 때 마지막 2초 버퍼를 버리고(원본 그대로), **파이썬은 다시 붙을 때 REST 로 그 봉을 메우지만
  Rust 는 REST 가 꺼져 있어 못 메운다** -- 두 쪽 로그의 `끊김` 시각이 걸친 분은 뺀다.
- **OKX 맥락** gaps 는 채널 `all` 하나라 그 구간은 모든 종목에서 뺀다. 펀딩 `ts` 가 없는 메시지는 받은 시각이 들어간다(드물다).

## 6. REST 검증·복구(전부 기본 꺼짐)

`RTCOLLECT_REST=1` 이면 Rust 도 파이썬처럼 60초마다 공백·불일치 분을 aggTrades 로 다시 받아 1분봉과 1e-4 안에서 맞을 때만 분을 통째로 갈아 끼우고,
300초마다 1분봉과 대조한다(`verify_1m`). **그림자 동안은 꺼 둔다**: 파이썬이 같은 REST 를 이미 부르고 있어 켜면 바이낸스 호출(가중치)이 두 배가 되고,
봇·대시보드와 같은 IP 한도(2,400/분)를 나눈다.

| | 꺼짐(기본) | 켬 |
|---|---|---|
| 바이낸스 REST | 0 | 심볼마다 klines(1/분 + 1/5분) + aggTrades(20/회, 프로세스 전체 2초 간격 · IP 사용량 1500 이상이면 다음 분까지 양보) |
| `verify_1m`·`backfill_1m` | 표만 있고 빈다 | 파이썬과 같은 규칙으로 쌓인다 |
| 끊긴 구간 | gaps 에만 남고 영영 빈다 | 47시간 안이면 메워진다(3번 실패면 포기) |
| 418·429 | 해당 없음 | `data/live/binance_ban_until`(binance_ban_guard 공용 파일)에 해제 시각을 적고 그때까지 안 보낸다(원본 테이프 수집기에는 없던 것) |

전환 때 켠다(§9). 대조 도구는 두 쪽 표를 다 보고 REST 가 바꾼 분을 뺀다.

다른 종류의 REST(`RTCOLLECT_REST=1` 일 때만 -- 원본과 같은 규칙으로 가짜 서버 정답지 일치 확인: tests/ctx_parity.rs · tests/venue_rest_parity.rs):

| 종류 | 켜면 나가는 것 | 분당 호출(켰을 때 · 그림자면 파이썬과 합쳐 두 배) |
|---|---|---|
| `binance_ctx` OI | `/fapi/v1/openInterest` 종목마다 0.5초(4종목) · 5초(느린 5종목) | **약 540회/분**(가중치 1 = 540/분 -- IP 한도 2,400 의 22%, 파이썬과 합치면 1,080 = 45%). 🔴그림자 동안 켜지 말 것 |
| `okx_tape` | 시작 때 ctVal 1회 · 300초마다 history-candles 최대 6회 · 60초마다 공백 분 복구(분마다 1분봉 1 + 경계 1 + 쪽 100건마다 1) | 공백 없으면 종목당 약 1.2회/분 · 복구 중이면 프로세스 전체 0.15초 간격 상한(≤ 400회/분 -- OKX 한도 20회/2초 IP 단위를 파이썬과 나눈다) |
| `okx_ctx` | 시작 때 ctVal 1회 | 종목당 시작마다 1회 |
| `bybit_tape` | 300초마다 종목마다 v5 kline 1회(복구 없음 -- 원본에 없다) | 5종목 = 1회/분 |
| `xyz` | 접속할 때마다 종목마다 candleSnapshot POST(마지막 저장 봉부터, 없으면 72시간) | 재접속 때만(4회/접속) |
| `raster` | 시작 · 30분마다 · 드리프트 1% · 시퀀스 단절 때 `/fapi/v1/depth?limit=1000`(가중치 20 · 🔧밴 파일을 지킨다) | 평소 시간당 2회 안팎 -- **그림자도 켠다**(안 켜면 모든 행이 무효라 대조할 것이 없다. 합쳐도 시간당 4회 안팎) |
| `depth` | 새 시 파일마다 `/fapi/v1/depth?limit=1000`(가중치 20 · 🔧밴 파일을 지킨다) | 시간당 1회(= 분당 약 0.3 가중치). 끄면 첫 줄 `{"_snapshot":null}` -- 읽는 쪽(ob5m·bookdepth_feat)이 그 시 파일로 북을 못 세운다(«no_snapshot») → **전환 때 반드시 켠다** |

- 바이낸스 OI 의 418·429 는 공용 밴 파일(`RTCOLLECT_BAN_FILE`, 기본 `data/live/binance_ban_until`)에 적고 그때까지 안 보낸다(원본 binance_ban_guard 와 같다).
- 원본과 다른 한 곳: 원본은 1분봉 대조를 WS 수신 루프 안에서 기다려, 대조 응답이 이상하면(모양이 틀린 1분봉 행 · 바이낸스는 시간초과까지) 그 WS 연결이
  끊겨 재연결 공백이 적힌다. Rust 는 대조가 따로 스레드라 로그만 남는다 -- REST 를 켰을 때만 · 이상한 응답에서만 생기는 차이다.

## 7. 감시

- 감독(`supervisor_rtcollect.sh`)은 종료코드와 무관하게 15초 뒤 다시 띄운다. 정지 감시: 바이낸스·OKX·OKX 맥락 70(루프 120초·쓰기 600초 -- 원본과 같다),
  Bybit 테이프 3(체결 600초 없음), Bybit 청산·바이낸스 맥락·xyz 는 없음(원본과 같다). OKX 모르는 종목·OKX ctVal 갈라짐(REST 켬) = 1.
- 그림자 동안은 감시자(`ops_watchdog`·`multicoin_collectors` 매니페스트)에 **넣지 않는다**(경보가 섞인다). 로그만 매일 본다:
  `grep -c '연결 실패\|끊김\|정지 감지\|멈춤' logs/supervisor/rtcollect_*_$(date +%Y%m%d).log`.

## 8. 끝나면 지울 것(그림자를 접을 때)

1. crontab 의 `supervisor_rtcollect.sh` 줄 · 2. 감독 프로세스(먼저 감독 bash, 그다음 rtcollect -- 반대면 감독이 다시 띄운다)
3. `data/hot/rtcollect/`(그림자 파일 전부) · 4. `data/live/.supervisor_rtcollect_*.lock` · 5. `logs/supervisor/rtcollect_*`(남길 거면 보관)
`bin/rtcollect` 는 전환에 쓰니 남긴다.

## 9. 전환 순서(종류 하나씩 · 사용자 승인 뒤)

읽는 쪽(server.py `HOT_TAPE_DB`·`OKX_TAPE_DB_PATH`·`OKX_CTX_DB_PATH`·`BINANCE_CTX_DB`·`XYZ_DB` · `coin_flow` · `market_ctx` · `seal_to_lake` · 청산 탭 · liqmap 채점기 ·
`data_quality_monitor` · 감시자)은 **경로를 안 바꾼다** --
Rust 가 원래 파일을 이어서 쓴다(이전 조각에서 «파이썬이 쓰던 파일을 Rust 가 이어 써도 같은 행» 확인).
1. 그 종류의 그림자 감독·rtcollect 를 끈다(§8 의 2).
2. 파이썬 감독 bash → 파이썬 수집기 순으로 끄고, 그 crontab 줄을 주석으로.
3. `RTCOLLECT_TAKEOVER=1 <경로 변수>=data/hot/<원래 이름>`(바이낸스는 + `RTCOLLECT_REST=1`)로 `supervisor_rtcollect.sh` 를 띄우고 crontab 줄을 바꾼다.
   끈 순간부터 Rust 첫 체결까지는 gaps 에 `ws_reconnect` 로 남는다(바이낸스는 REST 가 메운다 · OKX·Bybit 는 원본도 REST 복구가 없거나 다른 길).
4. `multicoin_collectors.json` 매니페스트의 그 항목 `runner` 를 `rtcollect` 로, `match` 에 `RTCOLLECT_KIND` 를 더한다(아래 «감시에 넣을 때»).
5. 같은 날 `ops_watchdog` 신선도·대시보드 카드(풋프린트·청산·시장 맥락)를 눈으로 본다. 되돌리기 = 3 → 2 의 반대.

**감시에 넣을 때 볼 것**: 매니페스트의 `pids_of` 는 «명령줄의 어떤 인자가 runner 로 끝나는가 + /proc/<pid>/environ 이 match 와 글자까지 같은가»로 찾는다.
rtcollect 는 명령줄이 실행 파일 경로 하나라 runner=`rtcollect` 로 잡히고, 감독이 심볼 변수의 공백을 빼고 바이낸스는 소문자·나머지는 대문자로 넘긴다 --
match 의 목록 글자·순서가 감독에 준 것과 같아야 한다. 신선도(`fresh`)는 파일 경로가 같으니 그대로. `RSS_MB` 의 값(파이썬 155MB)은 다시 잰다.

## 10. 전진 검정 기록이 끊기지 않는 근거

- 그림자 동안 Rust 는 `data/hot/rtcollect/` 에만 쓴다. 실행 파일이 파이썬 파일 경로를 거부하고(`TAKEOVER` 없이는 종료코드 2), 감독 스크립트도 기본이 그 폴더다.
- 파이썬 수집기·감독·crontab·파이썬 파일은 그대로다. 고래 맞대결(whale_duel1 · 바이낸스+OKX 테이프)·liqmap 전진(Bybit 청산)이 읽는 파일은 파이썬 것 그대로.
- 대조 도구는 두 파일을 `mode=ro` 로만 연다.
- 전환 때 끊기는 것은 «파이썬을 끈 순간 ~ Rust 첫 체결»(재시작 한 번과 같은 수 초)이고 gaps 에 적힌다 -- 지금 감독이 파이썬을 다시 띄울 때와 같은 모양이다.

## 11. 의도한 차이 — 결함은 Rust 에서만 고쳤다 (사용자 결정 2026-11-21 질문지 7번)

파이썬 수집기는 그대로다. 아래 자리에서 Rust 는 원본과 **다르게** 동작하고, 그림자 대조 때 그 차이는 불일치가 아니라 «의도한 차이»로 적는다.
스위치는 없다(고친 동작 하나). 일치 시험의 정답지는 «원본 + 그 결함 고침» 파이썬으로 만든다(`gen_venue_fixture.FixedSrc` -- 원본 함수 글자에서 정해 둔 줄만
바꿔 다시 정의) · 고치지 않은 원본과 어디서 몇 행 다른지를 정답지 `intended` 에 적는다(아래 «정답지에서 본 차이» = [원본 행, 고친 행, 다른 행]).
고친 수집기가 원본이 끊던 자리에서 안 끊으면, 다음 접속 사건은 «서버가 닫음»으로 둔다(정답지·Rust replay 같게 -- `LENIENT`).

| # | 수집기 | 원본(파이썬) | Rust | 디스크에 나타나는 차이 | 정답지에서 본 차이 | 시험 |
|---|---|---|---|---|---|---|
| 1 | Bybit 청산 | 바인딩 못 하는 값(목록·사전 종목)이 든 행 하나가 버퍼에 남아 **재시작까지 모든 쓰기가 실패**(기록 멈춤) | 그 행만 버리고(로그) 계속 쓴다 | Rust 에만 그 뒤 청산 행이 있다 | bybit_liquidations 189 → 194 (5행) | venue_parity(끝 시나리오) |
| 2 | Bybit 청산 | ping·플러시·60초 침묵을 **벽시계**로 잰다(시계가 뒤로 뛰면 늦어진다) | 단조 시계로 잰다(recv_ms·공백 시각은 벽시계 그대로) | 시계가 뛴 날에만 ping·끊김 시각이 다르다(행 값은 같다) | — (가짜 시계 하나라 같다) | venue_loop(단조로 판정 · 개발기 실측 벽 19.28초 대 단조 20.02초) |
| 3 | Bybit 테이프 | tickers 의 못 읽는 숫자 칸(예: fundingRate "x")을 합쳐 두어 **그 종목의 모든 다음 tickers 가 예외 = 재연결 반복**(상태 오염), 그 메시지에서 OI 상태만 먼저 바뀌어 **OI 행이 빠진다** | 그 칸만 버리고(로그) 나머지 칸으로 행을 낸다 | Rust 에만 OI·마크 행이 더 있고 재연결(gaps)이 적다 | bybit_oi 17 → 18 · bybit_mark 62 → 63 · gaps 14 → 16(«서버가 닫음» 모형 포함) | venue_parity |
| 4 | OKX·바이낸스 테이프 | 1분봉 대조를 **WS 수신 루프 안에서** 기다려, 대조 응답이 이상하면(모양 틀린 행 · 바이낸스는 시간초과도) WS 가 끊기고 재연결 공백이 적힌다 | 대조는 따로 스레드 -- 로그만 | REST 를 켰을 때만 · 이상한 응답에서만 원본에 공백이 더 있다 | OKX 대조 «모양 틀린 행» = 원본 예외 · Rust 는 Err 로그 | venue_rest_parity(결과 "raised") |
| 5 | 테이프 복구(공용 backfill_loop) | 영구 거부 기록(`mark_backfill_failed`)이 실패하면 **복구 태스크가 조용히 죽는다**(수집은 계속 · 복구는 재시작까지 멈춤) | 로그 남기고 다음 분 · 다음 주기에 다시 | 그 경우 원본은 그 뒤 공백이 안 메워진다 | — (쓰기 실패 주입은 정답지에 없다) | 코드(rest.rs backfill_cycle) |
| 6 | 테이프 REST(바이낸스) | aiohttp 라 공용 밴 파일(binance_ban_guard)을 모른다 -- 차단 중에도 두드려 해제가 늦어진다(2026-09-26 사고) | 418·429 를 밴 파일에 적고 그때까지 안 보낸다(depth 스냅숏·바이낸스 맥락 OI 도) | 밴 중엔 Rust 요청 0 | rest_parity «오류»(밴 중 요청 0) | rest_parity |
| 7 | OKX 맥락 | 접속에 실패할 때마다 공백 시작을 다시 잡아 gaps 가 **마지막 실패 ~ 붙은 시각**만 적힌다 | **처음 끊긴 시각** ~ 붙은 시각 | gaps.from_ms 가 Rust 쪽이 이르다(대조는 두 쪽 gaps 를 합쳐 빼므로 행 비교엔 영향 없음) | gaps 1행 다름 | ctx_parity |
| 8 | 바이낸스 맥락 | 깨진 JSON 한 프레임에 **그 연결 전체를 끊는다** | 그 프레임만 건너뛴다 | 원본은 재연결 동안 마크·청산을 놓친다(Rust 에만 그 행) | (정답지에선 다음 접속이 바로 와 행 차이 0) | ctx_parity |
| 9 | 바이낸스 맥락 | OI 응답의 `time` 을 seen 에 **먼저** 넣고 값을 읽어, 값을 못 읽은 시각은 그 뒤 정상 응답도 **영구히 버린다** | 값을 읽은 뒤에 seen 에 넣는다 | REST 를 켰을 때 Rust 에만 그 시각 OI 행 | oi_1s 18 → 19 | ctx_parity |
| 10 | 바이낸스 맥락 | SIGTERM 이면 보류 행(최대 10초분)을 잃는다(asyncio 기본 처리 = finally 없이 끝) | 쓰고 끝낸다 | 재시작마다 원본이 최대 10초 비고 Rust 는 안 빈다 | — | 코드(main.rs) |
| 11 | 바이낸스 맥락 | 재연결 기록이 없다(gaps 표 없음) | `gaps(symbol, from_ms, to_ms, reason)` 표를 **더했다**(마크 = 소문자 종목 · 강제청산 = NULL) -- 기존 표·열은 그대로라 대시보드·dqm·liqmap 은 안 바뀐다 | Rust 파일에만 gaps 표 | gaps 0 → 20 | ctx_parity · venue_loop(끊김 셋 = gaps 셋) |
| 12 | xyz100 | 끊기면(예외·시한 셋·글자 아닌 메시지) 마지막 플러시 뒤 받은 행(최대 2초)을 **버린다** · 종료 때도 | 끊길 때·종료 때 쓴다 | Rust 에만 끊김 직전 행 | hl_xyz_1m 11 → 13 · hl_xyz_ctx_1s 73 → 124 | ctx_parity · venue_loop((9, 18) -- 원본이면 (6, 12)) |

| 13 | depth_diff | 시각이 바뀐 뒤 지난 시의 늦은 프레임이 오면 압축 중인 그 파일을 다시 열어 이어 쓰고, 압축 스레드가 끝나며 원본을 지워 **그 줄이 사라진다**(경쟁) -- 압축이 이미 끝났으면 새 `.jsonl`(스냅숏부터)을 만들어 읽는 쪽(`.jsonl` 먼저)이 그 시의 `.gz` 를 못 본다 · 늦은 프레임 하나에 **지금 시** 파일까지 압축해 곧 다시 연다 | 그 시 압축을 기다린 뒤 `.gz` 뒤에 gzip 멤버로 붙인다(파이썬 gzip.open 도 이어 읽는다) · 앞으로 넘어갈 때만 압축한다 | Rust 는 늦은 줄이 그 시 `.gz` 에 남고 지금 시 파일은 `.jsonl` 그대로 | 정답지엔 넣지 않았다(원본 결과가 돌릴 때마다 달라 정답지가 안 된다) | depth_parity `depth_late_frame_during_compress_is_kept`(2만 줄 압축 중 늦은 셋 → 2만+3 줄 · 지금 시 그대로) |
| 14 | 호가 래스터 | 스냅숏 REST 가 공용 밴 파일을 모른다 | 밴 중이면 안 보내고 418·429 를 밴 파일에 | 밴 중엔 Rust 는 동기 못 함(행 무효) | — | 코드(main.rs) |
| 15 | OKX 맥락 · 바이낸스 맥락(gaps) | 재연결 공백 «1.5초 미만은 안 적음» 을 벽시계로 쟀다 -- WSL 벽시계가 뒤로 뛰면(실측 −1.5~−2.4초) 3초 공백이 짧게 재여 gaps 행이 빠졌다 | 판단만 단조 시계 · 적는 from/to 는 벽시계 그대로 | 벽시계가 뛴 재연결에서 Rust 에만 gaps 행 | — | 코드(okx_ctx.rs·binance_ctx.rs·net.rs) |
| 16 | ob5m · bookdepth(북 재구성 `replay`) | 시 파일 스냅숏의 한쪽(또는 양쪽)이 비면 `np.flatnonzero(bid)[-1]` IndexError 로 프로세스가 죽는다(얇은 주식 선물 -- 2026-11-21 그림자 SNDKUSDT 에서 Rust 가 같은 자리에서 패닉) | 그 사슬을 «empty_book» 으로 끝내고 restart 규칙대로 다음 시 파일 스냅숏에서 다시 | 파이썬은 죽고(감독이 다시 띄움) Rust 는 다음 시부터 봉을 잇는다 | — | 코드(ob5m.rs `replay`) |
| 17 | 롱숏비 워커(rtduck oi_lsratio) | 바이낸스 IP 밴 파일을 모른다(밴 중에도 5분마다 20요청) | 밴 중이면 안 보내고(그 원천 실패 = 기존 값 유지) 418·429 는 밴 파일에 | 밴 동안 Rust 에만 그 폴이 없음 | — | 코드(rtduck/src/main.rs) |
| 18 | 아이스버그 | 시 파일 스냅숏 한쪽이 비면 `np.flatnonzero(...)[0]` IndexError 로 죽는다(ob5m #16 과 같은 자리) | 그 시를 «스냅샷 없음» 처럼 건너뛴다 | 파이썬은 죽고 Rust 는 다음 시부터 | — | 코드(iceberg.rs) |
| 19 | 예약 수집기 polls.status | 실패 글자 = 파이썬 예외 repr | HTTP 오류는 `<HTTPError 429: 'Too Many Requests'>` 그대로 · 그 밖(연결·JSON·모양)은 이름+글 근사 | 오류 행의 status 글자 | — | 코드(cron.rs) |
| 20 | 래스터 14일 보관(rtduck raster_archive) | parquet·gzip 쓰기가 성공하면 바로 원본 .f32 를 지운다 | 새로 쓴 보관본을 다시 읽어 행 수(0 아닌 칸·mid 행 / 푼 바이트)가 맞을 때만 지운다 · 안 맞으면 보관본을 지우고 원본 유지 | 정상일 땐 차이 없음 · 보관본이 깨진 경우에만 Rust 쪽에 원본이 남는다 | — (정답지에선 늘 맞다) | 코드(raster_archive.rs `archive_then_unlink`) |
| 21 | lake 봉인 hot 정리(rtduck seal) | 지울 날짜의 lake 파일이 **있는지만** 보고 hot 행을 지운다 | 지우기 전 그 (코인, 날짜) lake 파일을 다시 읽어 행 수 ≥ hot 행 수 일 때만 · 아니면 그 표는 안 지움 | 정상일 땐 차이 없음 | — | 코드(seal.rs prune_hot) |
| 22 | Deribit 체결(rtduck deribit) | SIGTERM 이면 버퍼(최대 20초분)를 버리고 죽는다 | 버퍼를 쓰고 0 | Rust 에만 끝 직전 체결 | — | deribit_loop |

`shadow_tape.py` 는 1·3·8·9·12 의 표에서 «Rust에만» 을 `(의도한 차이 후보 -- 이유)` 로 따로 적는다(«파이썬에만»·값 불일치는 그대로 실패).
원본 그대로 둔 것(고치지 않음 -- 범위 밖이거나 값이 원본 정의다): depth 의 재접속 사이 구멍이 `_gap` 으로 안 남는 것 · 재시작이 시 경계를 넘으면 지난 시
`.jsonl` 이 압축되지 않는 것 · 압축 중인 파일을 다시 열어 이어 쓴 줄이 사라질 수 있는 경쟁(보고 «이상한 점»).

## 12. rtduck — DuckDB 로 쓰는 수집기 (사용자 결정 2026-11-21 질문지 19: DuckDB 형식 그대로 · 따로 된 실행 파일)

실행 파일 `rtduck/target/release/rtduck`(DuckDB 1.5.1 묶음 + parquet 확장(2026-11-21 여덟째 조각 · 54.7 → 59.1MB · 재빌드 17.7분)) → 서버 `bin/rtduck` · 감독 `scripts/ops/supervisor_rtduck.sh <종류>`(🔴실행·등록 안 함).

| 종류(`RTDUCK_KIND`) | 목록 | 쓰는 파일(기본) | 원본 파이썬 파일 |
|---|---|---|---|
| `oi_lsratio` | `BOT_SYMBOLS` · `OI_POLL_INTERVAL_SEC` · `OI_FETCH_LIMIT` · `OI_ENABLE`(원본과 같다) · 🔴`RTDUCK_REST=1` 필수 | `data/live/rtduck/oi_lsratio.duckdb` | `data/live/oi_lsratio.duckdb`(가리키면 종료코드 2 · 전환 때만 `RTDUCK_TAKEOVER=1`) |
| `hl_trades` | `HL_COINS` · `HL_ROOT` | `data/live/rtduck/orderflow/hyperliquid/<코인_코인>/<날짜>.duckdb` | `data/live/orderflow/hyperliquid/…` |
| `hl_positions` | `HL_POS_COINS`(원본과 같다) · `HL_ROOT`(체결 DuckDB 읽기만 -- 그림자는 Rust hl_trades 의 data/live/rtduck/orderflow/hyperliquid) · 🔴`RTDUCK_REST=1` 필수 | `data/hot/rtduck/hl_positions[_코인들].sqlite` | `data/hot/hl_positions[_코인들].sqlite`(가리키면 종료코드 2) |
| `market_data` | `BOT_SYMBOLS`(미시구조 · 기본 ETHUSDT,BTCUSDT,SOLUSDT) · `TAIL_SYMBOLS`(요격기 · 서버 ETHUSDT) · `COLLECT_MICROSTRUCTURE`·`COLLECT_TAIL_RISK`·`WRITE_DASHBOARD_STATE`(원본과 같다) · 🔴미시구조 REST 는 `RTDUCK_REST=1` 일 때만 | 폴더 `data/live/rtduck/`(`RTDUCK_LIVE_DIR`): `microstructure.duckdb` · `tail_risk.duckdb` · `liq_burst_state.json` · `dashboard_state.json` | `data/live/` 의 같은 이름(가리키면 종료코드 2) |
| `raster_archive`(예약 · 한 번) | `OF_ROOT`(기본 `data/live/rtcollect/orderflow/raster` = Rust 래스터 그림자) · `OF_ARCHIVE_ROOT` · `OF_RETENTION_DAYS`(14) · `OF_ARCHIVE`(1) · `OF_SYMBOLS`(기본 OF_ROOT 아래 폴더 전부) | `data/live/rtduck/orderflow/raster_archive/<SYM>/<날짜>/<시각>.f32.parquet`·`…_mid.parquet`·`.gz` + `orderbook.duckdb`(뷰 book·book_mid) | `data/live/orderflow/raster(_archive)`(가리키면 종료코드 2) |
| `liq_tab`(예약 · 한 번) | `LIQ_TAB_LIVE`(기본 `data/live/rtduck`) · `LIQ_TAB_LAKE`(`data/lake` 읽기만) · `LIQ_TAB_HOT`(`data/hot` 읽기만) · `--coin` · 🔴klines 는 `RTDUCK_REST=1` 일 때만(꺼지면 원본 `LIQ_TAB_NO_REST` 처럼 저장된 1h 봉만) | `liq_tab_<코인>.json` · `liq_tab_k1h_<코인>.parquet` · `liq_tab_all.json` | `data/live/liq_tab_*`(LIQ_TAB_LIVE=data/live 면 종료코드 2) |

- 🔧밴 파일(`RTCOLLECT_BAN_FILE`, 기본 `data/live/binance_ban_until`)을 지킨다 -- 원본 롱숏비 워커는 모른다(밴 중에도 보냈다). 의도한 차이 #17.
- REST 부담 -- 그림자를 켜면 파이썬과 합쳐 **두 배**다(그래서 `tools/shadow_start.sh` 는 REST 종류를 기본으로 안 띄운다 · `RTDUCK_SHADOW_REST=1` 이면 띄움):
  - oi_lsratio: 5분마다 종목 × 원천 5 = 서버 4종목이면 20요청/5분 → 그림자 켜면 40요청/5분(바이낸스 /futures/data 가중치는 이 저장소에서 실측 안 함 · 밴 파일은 지킨다).
  - hl_positions: 원본 = 주소마다 clearinghouseState(가중치 2) 0.4초 간격 = 150회/분 = 가중치 300/분 · 서버 두 프로세스(ETH · 다코인)면 600/분(한도 1,200/분의 50%) →
    그림자 켜면 1,200/분 = **한도 100%**(429 가 나기 시작한다 -- 원본 주석의 «합계 50%» 전제가 깨진다). 켤 거면 파이썬 쪽을 끄는 전환 때만.
  - hl_trades: 웹소켓뿐(REST 없음) -- 그림자 기본으로 띄운다.
  - market_data: 요격기·dashboard_state 는 웹소켓뿐. 미시구조 REST = 종목마다 10초에 openInterest·premiumIndex 둘(+ 체결 웹소켓이 30초 넘게 조용하면 aggTrades limit 500 -- 시작 첫 바퀴에도 한 번) → 서버 3종목이면 **분당 36요청**,
    그림자에서 켜면 파이썬과 합쳐 **72/분**(접속 루프 시험의 실측 요청 열: OI → premiumIndex → aggTrades(첫 바퀴) → OI → premiumIndex …, 간격 = MS_POLL_INTERVAL_SEC). 그래서 `shadow_start.sh` 는 `RTDUCK_SHADOW_REST=1` 일 때만 켠다 --
    꺼 두면 웹소켓 칸은 쌓이고 poll_connected=false · data_stale=true(감시 check_data_sources 는 그 그림자 파일을 보지 않는다 -- 원본 파일만 본다).
- 매일 대조: 두 파일을 사본으로 떠서(쓰는 쪽이 열 때 다른 프로세스 열기가 막힌다 -- seal_to_lake 방식) 같은 ts 의 14칸·sources_ok 를 비교(collected_at 은 다르다).
- 🔴DuckDB 파일은 쓰는 프로세스가 하나다: 그림자와 파이썬이 **같은 파일**을 가리키면 하나가 «Conflicting lock» 으로 죽는다(위 기본 경로가 다른 이유).

- 장 달력(market_data 의 dashboard_state 세션 칸): `rtduck/src/sessions.json` = **서버 판 pandas_market_calendars 5.3.1.1** 의 JPX·LSE·NYSE schedule(2024~2035 · 표 `_source` 에 판이 적힌다 · 다시 뜨기 = `rtduck/tools/server_env.sh` venv 로 `tools/gen_sessions.py`). 개발기 5.4.0 판과 비교: JPX 2,931 · LSE 3,035 · NYSE 3,012 거래일이 **날짜·개장·마감·점심까지 하나도 다르지 않다**(2026-11-21). 범위 밖 날짜는 원본 예외 길(UTC 시각 규칙)이라 원본과 다를 수 있다.
- 🔴묶은 DuckDB(Rust)에는 ICU 가 없다: `TIMESTAMPTZ - INTERVAL` 이 Binder Error 라 부트스트랩(«최근 30분·60분·5분»)을 epoch µs 비교로 썼다(같은 순간 비교). 서버 SQL·seal_to_lake 는 파이썬 duckdb(ICU 있음)로 Rust 파일을 읽으니 상관없다.
- 분 저장(요격기·미시구조)은 원본이 지역 시각 `datetime.now()` 의 분·초로 정한다 -- 정답지는 TZ=Asia/Seoul(서버와 같다 · 일광절약 없음)로 만들어 Rust(UTC 분·초)와 같음을 고정했다. 오프셋이 분 단위인 시간대(실제 시간대는 모두 그렇다 · +05:45 도)면 «분이 바뀐 첫 15초» 판정과 저장 시각이 같다. 일광절약 전환 시각에는 지역 분이 같아도 순간이 한 시간 건너뛰어 원본 쪽이 한 분을 다르게 볼 수 있다(서버 KST 는 없음).

## 13. 계산 기록기 — CBOE 요약기 · 청산지도 점수 (2026-11-21 일곱째 조각)

| 종류 | 원본 | 쓰는 파일(기본 · 원본 경로는 `RTCOLLECT_TAKEOVER=1` 일 때만) | REST |
|---|---|---|---|
| `cboe_chain`(+요약기) | `collect_cboe_option_chain` 저장 직후 `cboe_option_summary_20261101.write_state` | `CBOE_STATE` = `data/live/rtcollect/cboe_option_state.json` · `CBOE_RV_CACHE` = `data/live/rtcollect/cboe_rv7_cache.json` | CBOE 체인(원본과 같다) + 🔴바이낸스 klines 1h×169(실현변동성 · 종목마다 · 캐시 하루) = `RTCOLLECT_REST=1` 일 때만 -- 꺼지면 `rv7: null` |
| `liqmap_scorer` | `live_liqmap_scorer_20261008`(매시 07분) | `LIQMAP_OUT` = `data/live/rtcollect/liqmap_score.jsonl` · `LIQMAP_STATUS` = `…/liqmap_score_status.json` · `LIQMAP_HOT`(읽기만, 기본 `data/hot`) | 🔴바이낸스 klines 1h×198 시간당 1회 -- `RTCOLLECT_REST=1` 이 아니면 «할 일 없음» 찍고 0 |

- 둘 다 밴 파일(`RTCOLLECT_BAN_FILE`)을 지키고 `RTCOLLECT_BINANCE_BASE` 로 주소를 바꾼다(시험 = 127.0.0.1 만 -- 아니면 패닉).
- 🔴liqmap 그림자는 빈 파일에서 시작하면 최근 168시간만 메운다(원본 규칙 그대로) -- 전진 판정 자료(`data/live/liqmap_score.jsonl`)는 파이썬 것이다. 그림자 대조 = 두 파일에서 같은 origin 줄 글자 비교.
- 값 맞춤의 전제(정답지와 다를 수 있는 곳 -- 결함 고침이 아니라 환경):
  - 쪽 합 `tot = sum(...)` 은 파이썬 3.11 규칙(차례로 더함) -- **서버 확인함: 파이썬 3.11.14 · sqlite 3.51.1(2026-11-21)**. 3.12+ 로 올리면 `sum()` 이 Neumaier 보정합이 된다 -- 정답지에서 보정합으로 바꿔도 `round(v/tot, 6)` 이 갈리는 칸 0(지도 칸 전부 중).
  - 원본 `sort_values("ts_ms")` 는 quicksort(안정 아님) -- 같은 ms 청산 여럿이 같은 칸에 들면 더하는 순서만 다를 수 있다(round 2 뒤 차이 0 -- 정답지에 같은 ms 세 거래소 8묶음).
  - `np.exp` = libm(정답지는 `NPY_DISABLE_CPU_FEATURES=X86_V4`). 서버 numpy 가 AVX512 exp 를 쓰면 1ulp 차이가 6자리 반올림을 넘을 수 있다(아주 드묾).
  - 청산 qty 가 NULL 이면 원본은 NaN 을 JSON `NaN` 으로 쓰고 Rust 는 `null`(serde) -- 수집기 셋 다 qty 를 늘 채운다.
- 요약기가 실패하면(사슬 모양 이상 등) 원본처럼 그 종목 «실패» · 종료코드 1 · 스냅숏 파일은 남는다.

- `soxl_weekend`(rtcollect · 예약 `--record`/`--status`): `SOXL_WEEKEND_OUT`(기본 `data/live/rtcollect/soxl_weekend_forward.jsonl` -- 파이썬 장부 `data/live/soxl_weekend_forward.jsonl` 은 TAKEOVER 때만) ·
  klines·fundingRate = `RTCOLLECT_REST=1` 일 때만. 판정 `--look`·`--insample` 은 파이썬 원본 그대로(사람이 돌리는 판정 도구 · looks 장부도 파이썬 것). UA 는 원본(urllib 기본)과 다르다(`soxl-weekend-forward`).

## 14. 래스터 보관 · 청산 탭 (2026-11-21 여덟째 조각 -- rtduck, parquet 확장)

- 래스터 보관 = 원본 `_purge_old` 와 같은 파일 이름(`<보관>/<SYM>/<날짜>/<원본 이름>.parquet` · `<원본 이름 줄기>_mid.parquet` · 실패 폴백 `.gz`) · 같은 열·형(ts_ms int64 · bin int32 · qty float · bin_size float / ts_ms · mid) · 같은 행(qty≠0 칸 · -0.0 은 0 · NaN 은 남김) · 같은 뷰 SQL.
  🔧**의도한 차이 #20**: 새로 쓴 보관본을 다시 읽어 행 수(parquet = 0 아닌 칸 수·mid 행 수 / gzip = 푼 바이트 = 원본)가 맞을 때만 원본을 지운다(원본은 쓰기만 보고 지운다) -- 안 맞으면 보관본을 지우고 원본을 남겨 다음 시간에 다시.
  이미 보관본이 있을 때는 원본처럼 원본만 지운다(그 보관본은 되읽지 않는다 -- 원본 동작 그대로). 처리 순서: 원본 = 디렉터리 순서(정해지지 않음) · Rust = 이름순(파일마다 따로라 결과는 같다).
  양방향 확인: Rust duckdb 가 원본(pyarrow)이 쓴 parquet 17개를 같은 행으로 읽음 · 서버 판 파이썬(duckdb 1.5.1 · pyarrow 26.0.1)이 Rust 가 쓴 보관본 20개·orderbook.duckdb 뷰를 같은 값으로 읽음.
- 청산 탭 = 원본 main() 과 liq_tab_<코인>.json · liq_tab_all.json **글자 일치**(정답지는 서버와 같은 numpy 2.2.5·pandas 2.2.3·OpenBLAS ZEN venv -- `rtmodel/tools/server_env.sh` + duckdb 1.5.1).
  값 맞춤에서 찾은 것: numpy 합은 8192칸 묶음마다 짝 합(그 묶음 합을 차례로) · `np.convolve(v, ones(11), "same")` = OpenBLAS ddot 꼬리라 창 안 차례 합(길이 < 11 이면 뒤집힌 차례 · 개발기 MKL numpy 는 다른 차례 -- 정답지를 개발기 numpy 로 뜨면 안 된다) ·
  `np.round(x, d)` = rint(x·10^d)/10^d(파이썬 round 와 다르다) · 서버에 bottleneck 이 없다(조정자 서버 확인 2026-11-21 -- 전제 맞음).
  1h 봉 캐시는 DuckDB 가 쓴 parquet(t int64 · high·low·close·qv double) -- 서버 판 pandas 가 같은 행·형으로 읽음(되돌리기 때 파이썬이 이어 쓸 수 있다).
  🔴같은 ms 청산이 여럿이면 원본 sort_values(quicksort)와 더하는 순서만 다를 수 있다(정답지엔 없음).

## 15. 예약 종류 그림자 등록 후보 (등록하지 않음 -- 사용자 결정)

`<저장소>` = 서버 저장소 폴더 · 실행 파일 = `bin/rtcollect`·`bin/rtduck`(개발기에서 빌드해 옮긴 것). 로그는 감독 로그와 같은 `logs/supervisor/`. 쓰는 곳은 전부 그림자 기본 경로(원본 파일을 가리키면 종료코드 2).

```
# rtcollect 예약(한 번 돌고 끝) -- 원본 줄과 같은 시각
7 * * * *    cd <저장소> && RTCOLLECT_KIND=liqmap_scorer RTCOLLECT_REST=1 bin/rtcollect >> logs/supervisor/rt_liqmap_scorer.log 2>&1
30 21 * * *  cd <저장소> && RTCOLLECT_KIND=soxl_weekend RTCOLLECT_REST=1 bin/rtcollect --record >> logs/supervisor/rt_soxl_weekend.log 2>&1
*/10 * * * * cd <저장소> && RTCOLLECT_KIND=cboe_chain RTCOLLECT_REST=1 bin/rtcollect <원본 줄과 같은 종목…> >> logs/supervisor/rt_cboe_chain.log 2>&1
# (treasury_futures · macro_5m 은 후보에서 뺀다 -- Yahoo 429 위험, 전환 때만)
# rtduck 예약
40 * * * *   cd <저장소> && RTDUCK_KIND=raster_archive bin/rtduck >> logs/supervisor/rt_raster_archive.log 2>&1
<원본 liq_tab 분 + 5> * * * * cd <저장소> && cp -p data/live/liq_tab_k1h_*.parquet data/live/rtduck/ && RTDUCK_KIND=liq_tab bin/rtduck >> logs/supervisor/rt_liq_tab.log 2>&1
```

| 종류 | 원본이 부르는 REST(한 번) | 그림자 줄대로 켜면 같은 IP 합계 | 메모 |
|---|---|---|---|
| liqmap_scorer | 바이낸스 klines 1h×198 1회/시(가중치 2) | ×2 = 2회/시(가중치 4) | REST 를 끄면 할 일이 없다(그래서 켠다 -- 분당 한도 2,400 에 비하면 무시할 양) |
| soxl_weekend | 새 주말마다 klines 8 + fundingRate 1 = 9요청/주 | ×2 = 18/주 | 끝난 주말이 없으면 0 |
| cboe_chain(+요약기) | CBOE 체인 종목마다 1/10분 · 바이낸스 klines 1h×169 종목마다 하루 1(캐시) | CBOE ×2 · 바이낸스 ×2(하루 종목 수만큼) | CBOE 한도는 이 저장소에서 실측 안 함 |
| raster_archive | 없음 | ×1 | 디스크만 · Rust 래스터 그림자 폴더를 보관 -- 그 그림자가 안 돌면 할 일 없음 |
| liq_tab | klines(코인당 1회/시 · 첫 실행만 몇 쪽) | **×1**(REST 끔) | 파이썬 1h 봉 캐시를 복사해 같은 봉으로 계산 → 두 쪽 liq_tab_<코인>.json 을 글자로 대조. 복사는 파이썬 줄이 끝난 뒤(분 + 5) |
| treasury_futures · macro_5m | Yahoo 매분·매시 / 5분 | ×2 | 후보에서 뺌(429) |

대조: liqmap = 같은 origin 줄 글자 · soxl = 같은 weekend 줄(recorded_at 빼고) · cboe = cboe_option_state.json 의 종목 칸(generated 시각 빼고) · liq_tab = generated_ms 빼고 글자 · 래스터 보관 = 같은 시각 파일을 두 쪽 duckdb 로 읽은 행.

## 16. 뉴스 RSS · lake 봉인 첫 표 · 간헐 실패 (2026-11-21 아홉째 조각)

- `news`(rtcollect · 상주): `NEWS_RSS_DB`(기본 `data/hot/rtcollect/news_rss.sqlite` -- 원본 `data/hot/news_rss.sqlite` 는 TAKEOVER 때만) · `NEWS_RSS_TICK_S`(5).
  피드 8개·주기·조건부 GET·429 4배 쉼·BWE 거르기·polls 7일 정리 = 원본 그대로. 🔴그림자로 켜면 바깥 사이트 8곳 호출이 **두 배**(tree 15초·financialjuice 30초에서 429 를 본 적 있다 -- 원본 주석) → 등록 후보에서 뺀다(전환 때만).
  감독 줄 후보: `@reboot sleep 90 && cd <저장소> && RTCOLLECT_KIND=news bash scripts/ops/supervisor_rtcollect.sh >> logs/supervisor/rtcollect_reboot.log 2>&1 &`(전환 때 NEWS_RSS_DB=data/hot/news_rss.sqlite RTCOLLECT_TAKEOVER=1).
  원본과 다를 수 있는 곳(정답지 밖): XML 인코딩은 UTF-8 만 · DOCTYPE 안 개체 선언 안 풂 · 잘못된 XML 의 오류 판정은 quick-xml 기준(expat 보다 너그러운 곳이 있을 수 있다 -- 그 경우 원본은 그 폴링 «실패», Rust 는 읽음) · int() 의 유니코드 숫자.
  HTML 개체 표는 `rtcollect/src/html.rs` 한 곳(rtserve 의 `src/html_entities.rs`·`text::unescape` 는 이쪽을 쓰도록 옮길 것 -- rtserve 에이전트 몫).
- `rtduck::seal`(봉인 · 실행 종류는 아직 없음): 원본 seal_to_lake 도 DuckDB `COPY … (format parquet, compression zstd)` 로 쓴다(pyarrow 아님). DuckDB 파일 원천만 됨 --
  binance/oi_lsratio_5m 으로 원본과 두 번(봉인 → 재봉인·거부·건너뜀) stats·파일·열·형·값 일치 · 서버 판 파이썬이 Rust lake 를 같은 값으로 읽고, 원본 seal() 을 그 위에 다시 돌리면 46파일 모두 «건너뜀»(지문 = hash(row) XOR 같음).
  🔴SQLite 원천(hot 테이프·맥락 13 스트림)은 `attach … (TYPE sqlite)` = sqlite 확장이 묶음에 없다 → 다음 조각(rusqlite 로 읽어 DuckDB 임시 표로 -- 형 대응을 sqlite_scanner 와 같게 해야 지문이 같다). 날짜 식은 ICU 없이 정수 ms 산수(같은 날짜).
- 간헐 실패(조정자 재실행 5회 중 1회): 부하(바쁜 고리 6개) 아래 rtduck 시험 9개 × 20회 → **hl_trades_loop 2/20**, 나머지 0/20.
  원인 = 시험용 가짜 시계(`RTDUCK_FAKE_START`)가 «벽시계 + 차이»라 WSL 벽시계가 1.5~2.4초 뒤로 뛰면 자정 넘기가 밀려, 다시 붙은 뒤 공백 줄이 전날 파일에 들어갔다(`gaps` 0줄).
  고침: 가짜 시계만 «시작 + 단조 경과»(실제 실행은 원본처럼 벽시계 그대로 -- 실행 파일 쪽 시험 전용 길) · 시험의 시간 창을 부하 여유로 넓힘.
  market_loop 는 20/20 통과했지만 같은 가짜 서버를 두 실행이 같이 써서, 밴 중 실행이 끝나기 직전의 요격기 재접속이 다음 실행 기록·각본 번호에 섞일 수 있는 경쟁이 있었다 → 실행마다 새 포트 서버 · 첫 실행은 «웹소켓 셋이 붙은 뒤»로 조건 대기.

## 17. lake 봉인 전부 · Deribit 첫 단계 (2026-11-21 열째 조각)

- `seal`(rtduck · 예약 매일 02:05 UTC · 한 번): SPECS 30 스트림 전부(글자 = 원본 SPECS · 시험이 대조) · hot 정리(HOT_PRUNE 23 줄 · 원본 순서) · seal_status.json.
  `SEAL_LAKE`(기본 `data/rtduck/lake` -- `data/lake` 는 TAKEOVER 때만) · `SEAL_STATUS`(기본 `data/live/rtduck/seal_status.json`) · `SEAL_ROOT` · `--dry-run` · `--only`.
  🔴hot 정리는 `RTDUCK_TAKEOVER=1` 일 때만 -- 그림자 lake 를 근거로 파이썬 수집기의 hot 행을 지우면 안 된다(그림자 = 봉인만).
  SQLite 원천은 rusqlite 로 읽어 임시 DuckDB(디스크, lake 아래 .snap_*)에 싣는다 -- 열 형 = sqlite_scanner 규칙(INT → BIGINT · TEXT/VARCHAR → VARCHAR · REAL/DOUBLE → DOUBLE).
  **스트림별 지문(원본 seal 을 Rust lake 위에서 dry-run → 전부 «건너뜀»)**: SQLite 13(binance tape·spot_tape·oi_1s·mark_1s·liquidations · okx tape·oi·mark·funding·liquidations · hl asset_ctx · bybit tape·oi·mark·funding · cme treasury · hl xyz_1m·xyz_ctx_1s · yahoo macro_5m)
  + 섞임 2(hl positions·liquidations = SQLite 셋 + DuckDB 하나) 모두 같음 · DuckDB 원천: oi_lsratio(아홉째) + micro_1m·tail_risk_1m·deribit 5(option_trades·chain·option_summary·block_rfq·block_future_legs -- BOOLEAN·INTEGER 열 포함 · 열한째) 모두 같음 · deribit/gex_summary 는 원천 표가 지금 수집기에 없다(missing_source).
  정답지의 sqlite 확장은 개발기 duckdb 1.5.5 의 것이다(서버 판 1.5.1 확장은 개발기에 없고, 설치 = 바깥 접속이라 하지 않았다) -- hash(row) 는 1.5.1 과 1.5.5 가 같다(oi_lsratio 46파일).
  ✅**서버 확인함(2026-11-21 · 읽기 전용 · 실자료)**: 서버 hot SQLite 에서 rtduck 으로 봉인한 lake 위에 서버 원본(duckdb 1.5.1 + 그 판 sqlite 확장) `S.seal(dry_run=True, only=…)` →
  bybit/funding 40파일·34,037행 · okx/funding 80파일·115,432행 · cme/treasury_fut_1m 28파일·30,333행 · bybit/oi 40파일·344,804행 넷 다 written 0 · resealed 0 · 전부 skipped_existing
  = 1.5.1 sqlite 확장의 형 대응이 1.5.5 와 같다(INT·TEXT·REAL). 다시 확인하는 법: `SEAL_ROOT=. SEAL_LAKE=/tmp/seal_check/lake RTDUCK_KIND=seal bin/rtduck --only <스트림>` 뒤
  `python -c "import scripts.seal_to_lake as S, pathlib; print(S.seal(lake=pathlib.Path('/tmp/seal_check/lake'), dry_run=True, only='<스트림>'))"` 가 written 0·resealed 0.
  등록 후보: `5 2 * * * cd <저장소> && RTDUCK_KIND=seal bin/rtduck >> logs/supervisor/rt_seal.log 2>&1`(REST 없음 · ×1 · 원본 02:05 와 겹치면 원천 사본이 둘 -- 03:05 로 미룰 것).
- `deribit`(rtduck · 상주): 체결 WS(구독·하트비트·test_request 답·90초 무음 = 끊김·20초마다 write) · 재기동 틈 REST 백필(`RTDUCK_REST=1` 일 때만 -- 그림자 기본 꺼짐) · gaps · 상태 JSON(블록 구조·헤지·흐름).
  `DERIBIT_BT_DIR`(기본 `data/live/rtduck` -- 원본 파일 `data/live/deribit_options.duckdb` 는 단일 writer 라 같이 열면 하나가 죽는다 · TAKEOVER 때만).
  ✅(열한째 조각) 10분 체인·GEX·옵션 요약 → option_chain_snapshot·option_summary·option_oi_other·deribit_gex_state.json(원본 poll_once 두 번과 요청 34·상태 글자·표 1,994행 일치)
  · 매시 history 대조 → 빠진 체결·강제청산 표시·block_future_legs·block_rfq_trades(원본 reconcile → write 와 요청·결과·표·상태 글자 일치 · ΔOI 46칸) -- 둘 다 `RTDUCK_REST=1` 일 때만(`RTDUCK_DERIBIT_CHAIN=0` 이면 끔).
  원본과 다른 곳: write_state 의 vov24·atm_oi_pct 는 SQL(ICU·json 확장) 대신 같은 행을 Rust 로 · recorded_at_utc 글자의 지역 오프셋 = libc localtime(서버 KST) · 체인 폴링 직전에 체결 버퍼를 비우지 않는다(지금 요약은 체결을 안 읽는다) ·
  헤지 다리 조회 순서 = 원본 GROUP BY 해시 순서(두 쪽 다 정해지지 않음 -- 결과 표는 같다) · 한 프로세스 DB 잠금(DB_LOCK) · 🔴연결마다 DuckDB 확장 자동 불러오기를 끈다(개발기 ~/.duckdb 의 ICU 가 TIMESTAMP→TIMESTAMPTZ 를 지역 시각으로 바꾸는 것을 막는다).
  남은 것: 옛 파일 옮기기(migrate_once -- 서버는 이미 옮겼을 것 · 확인 필요).
  hourly_doi 는 체인 표가 없으면 {}(원본과 같음) -- 체인 표가 있는 원본 DB 를 가리키면 원본은 ΔOI 를 내고 Rust 는 None.
  REST 배수: 켜면 재접속마다 목록 셋 × 쪽 수(×2) · WS 는 Deribit 이라 바이낸스 한도와 무관 · 등록 후보: `@reboot sleep 90 && cd <저장소> && RTDUCK_KIND=deribit bash scripts/ops/supervisor_rtduck.sh deribit …`(감독 스크립트의 종류 목록에 deribit 를 더해야 한다).
