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

`shadow_tape.py` 는 1·3·8·9·12 의 표에서 «Rust에만» 을 `(의도한 차이 후보 -- 이유)` 로 따로 적는다(«파이썬에만»·값 불일치는 그대로 실패).
원본 그대로 둔 것(고치지 않음 -- 범위 밖이거나 값이 원본 정의다): depth 의 재접속 사이 구멍이 `_gap` 으로 안 남는 것 · 재시작이 시 경계를 넘으면 지난 시
`.jsonl` 이 압축되지 않는 것 · 압축 중인 파일을 다시 열어 이어 쓴 줄이 사라질 수 있는 경쟁(보고 «이상한 점»).

## 12. rtduck — DuckDB 로 쓰는 수집기 (사용자 결정 2026-11-21 질문지 19: DuckDB 형식 그대로 · 따로 된 실행 파일)

실행 파일 `rtduck/target/release/rtduck`(DuckDB 1.5.1 묶음 · 51.6MB) → 서버 `bin/rtduck` · 감독 `scripts/ops/supervisor_rtduck.sh <종류>`(🔴실행·등록 안 함).

| 종류(`RTDUCK_KIND`) | 목록 | 쓰는 파일(기본) | 원본 파이썬 파일 |
|---|---|---|---|
| `oi_lsratio` | `BOT_SYMBOLS` · `OI_POLL_INTERVAL_SEC` · `OI_FETCH_LIMIT` · `OI_ENABLE`(원본과 같다) · 🔴`RTDUCK_REST=1` 필수 | `data/live/rtduck/oi_lsratio.duckdb` | `data/live/oi_lsratio.duckdb`(가리키면 종료코드 2 · 전환 때만 `RTDUCK_TAKEOVER=1`) |
| `hl_trades` | `HL_COINS` · `HL_ROOT` | `data/live/rtduck/orderflow/hyperliquid/<코인_코인>/<날짜>.duckdb` | `data/live/orderflow/hyperliquid/…` |
| `hl_positions` | `HL_POS_COINS`(원본과 같다) · `HL_ROOT`(체결 DuckDB 읽기만 -- 그림자는 Rust hl_trades 의 data/live/rtduck/orderflow/hyperliquid) · 🔴`RTDUCK_REST=1` 필수 | `data/hot/rtduck/hl_positions[_코인들].sqlite` | `data/hot/hl_positions[_코인들].sqlite`(가리키면 종료코드 2) |
| `market_data` | `BOT_SYMBOLS`(미시구조 · 기본 ETHUSDT,BTCUSDT,SOLUSDT) · `TAIL_SYMBOLS`(요격기 · 서버 ETHUSDT) · `COLLECT_MICROSTRUCTURE`·`COLLECT_TAIL_RISK`·`WRITE_DASHBOARD_STATE`(원본과 같다) · 🔴미시구조 REST 는 `RTDUCK_REST=1` 일 때만 | 폴더 `data/live/rtduck/`(`RTDUCK_LIVE_DIR`): `microstructure.duckdb` · `tail_risk.duckdb` · `liq_burst_state.json` · `dashboard_state.json` | `data/live/` 의 같은 이름(가리키면 종료코드 2) |

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

- 장 달력(market_data 의 dashboard_state 세션 칸): `rtduck/src/sessions.json` = 개발기 pandas_market_calendars 5.4.0 의 JPX·LSE·NYSE schedule(2024~2035). 🔴서버 판이 다르면 휴장·마감 규칙이 다를 수 있다 -- 서버에서 `tools/gen_sessions.py` 로 다시 떠서 비교. 범위 밖 날짜는 원본 예외 길(UTC 시각 규칙)이라 원본과 다를 수 있다.
- 🔴묶은 DuckDB(Rust)에는 ICU 가 없다: `TIMESTAMPTZ - INTERVAL` 이 Binder Error 라 부트스트랩(«최근 30분·60분·5분»)을 epoch µs 비교로 썼다(같은 순간 비교). 서버 SQL·seal_to_lake 는 파이썬 duckdb(ICU 있음)로 Rust 파일을 읽으니 상관없다.
- 분 저장(요격기·미시구조)은 원본이 지역 시각 `datetime.now()` 의 분·초로 정한다 -- 정답지는 TZ=Asia/Seoul(서버와 같다 · 일광절약 없음)로 만들어 Rust(UTC 분·초)와 같음을 고정했다. 오프셋이 분 단위인 시간대(실제 시간대는 모두 그렇다 · +05:45 도)면 «분이 바뀐 첫 15초» 판정과 저장 시각이 같다. 일광절약 전환 시각에는 지역 분이 같아도 순간이 한 시간 건너뛰어 원본 쪽이 한 분을 다르게 볼 수 있다(서버 KST 는 없음).
