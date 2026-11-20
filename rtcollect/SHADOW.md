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

- 심볼 목록은 **지금 도는 파이썬과 같게**: `tr '\0' '\n' < /proc/<파이썬 pid>/environ | grep -E 'TAPE_SYMBOL|TAPE_MARKET|OKX_TAPE_INST|BYBIT_'`.
- 경로를 바꾸려면 `TAPE_DB_PATH` · `OKX_TAPE_DB_PATH` · `BYBIT_TAPE_DB` · `BYBIT_CTX_DB` · `BYBIT_LIQ_DB`(파이썬과 같은 이름).
  🔴파이썬 파일(`data/hot/<이름>`)을 가리키면 실행 파일이 종료코드 2 로 거부한다(`RTCOLLECT_TAKEOVER=1` 일 때만 연다 -- 전환 전용).
- `RTCOLLECT_REST`: 비워 둔다(꺼짐). 아래 §6.
- 띄우기: 종류마다 `RTCOLLECT_KIND=<종류> <심볼 변수>=<목록> bash scripts/ops/supervisor_rtcollect.sh`(머리 주석의 crontab 줄 -- 등록은 사람).
  로그 `logs/supervisor/rtcollect_<KEY>_YYYYMMDD.log`. 같은 종류·겹치는 심볼이 이미 돌면 켜지지 않는다.

## 3. 디스크

같은 행을 쓰므로 **하루 증가량 = 지금 파이썬 hot 파일의 하루 증가량**과 같다(뷰·인덱스·PRAGMA 도 같다).

- 근거: 테이프 표(`WITHOUT ROWID`, 19칸)는 개발기 실측 **행당 약 100바이트**(31만 행 31.5MB, 합성 값). ETH 선물은 2026-09-16 실측
  하루 약 32.6만 행(0.1달러 칸) → **ETH 하나 약 33MB/일**. 다른 코인은 행 수가 더 적다(칸 폭이 가격의 0.4bp 로 같고 거래가 적다).
- 서버에서 실제 행 수로 다시 재는 법(읽기 전용):
  `python3 -c "import sqlite3,time; c=sqlite3.connect('file:data/hot/binance_tape.sqlite?mode=ro',uri=True); print(c.execute('SELECT symbol,count(*) FROM trade_tape_1s WHERE ts_sec>=? GROUP BY 1',[int(time.time())-86400]).fetchall())"`
  → 행 수 × 100바이트. OKX·Bybit·현물 파일도 같은 식. `multicoin_collectors_20260926.py check --hours 24` 의 «MB/일» 칸도 같은 값이다.
- 파이썬 hot 파일은 봉인기(`seal_to_lake` prune_hot, 16일)가 줄이지만 **그림자 파일은 아무도 안 줄인다** -- 7일이면 위 값 × 7. 끝나면 지운다(§8).
- `data/hot/rtcollect/` 는 백업(`backup_live_data.sh` 의 `data/hot/*.sqlite`)·봉인·데이터 지도(`gen_data_map`)에 안 잡힌다(하위 폴더라).

## 4. 거래소 연결

파이썬과 **같은 수만큼 더** 붙는다: 바이낸스 선물 심볼마다 1 · 현물 심볼마다 1 · OKX 종목마다 1 · Bybit 테이프 1 · Bybit 청산 1.
지금 목록(ETH + `multicoin_collectors` COINS 9개)이면 바이낸스 선물 10 + 현물(목록 수) + OKX 10 + Bybit 2.
바이낸스 웹소켓 한도는 IP 당 «5분에 새 연결 300»이라 재시작 폭주가 아니면 여유가 있다. REST 는 기본 0회(§6).

## 5. 매일 대조

```
python3 rtcollect/tools/shadow_tape.py data/hot/binance_tape.sqlite      data/hot/rtcollect/binance_tape.sqlite      --from $(date -d '1 day ago' +%s) --to $(date +%s)
python3 rtcollect/tools/shadow_tape.py data/hot/binance_spot_tape.sqlite data/hot/rtcollect/binance_spot_tape.sqlite --from … --to …
python3 rtcollect/tools/shadow_tape.py data/hot/okx_tape.sqlite          data/hot/rtcollect/okx_tape.sqlite          --from … --to …
python3 rtcollect/tools/shadow_tape.py data/hot/bybit_tape.sqlite        data/hot/rtcollect/bybit_tape.sqlite        --from … --to …
python3 rtcollect/tools/shadow_tape.py data/hot/bybit_ctx.sqlite         data/hot/rtcollect/bybit_ctx.sqlite         --from … --to …
python3 rtcollect/tools/shadow_tape.py data/hot/bybit_liq.sqlite         data/hot/rtcollect/bybit_liq.sqlite         --from … --to …
```
표준 라이브러리만 · 두 파일 다 읽기 전용(`mode=ro`) · 한 시간 창씩 읽는다(큰 표를 통째로 안 올린다).
빼는 분: 두 쪽 gaps ±5초 · 두 쪽 backfill_1m ok 분 · 두 쪽 verify_1m 불일치 분 · 범위 끝 10초. 뺀 (종목, 분) 수를 같이 찍는다.

**통과 기준(7일 연속, 날마다)**: 표마다 «비교 N행 · 불일치 0 · 파이썬에만 0 · Rust에만 0», 그리고 «뺀 (종목,분)» 을 같이 적는다(분모).
뺀 분이 하루 1% 를 넘으면 그날은 통과로 세지 않는다(재연결이 잦다는 뜻 -- 원인을 먼저 본다).
청산(`bybit_liquidations`)은 사건이 드물어 비교 행이 0 인 날이 있다 -- 7일 합으로 0 이 아니어야 한다.
불일치가 나오면 `--show 20` 으로 행을 보고, 그 분이 한쪽의 재연결·REST 교체와 겹치는지 먼저 본다.

## 6. REST 검증·복구(바이낸스만)

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

## 7. 감시

- 감독(`supervisor_rtcollect.sh`)은 종료코드와 무관하게 15초 뒤 다시 띄운다. 정지 감시: 바이낸스·OKX 70(루프 120초·쓰기 600초 -- 원본과 같다),
  Bybit 테이프 3(체결 600초 없음), Bybit 청산은 없음(원본과 같다).
- 그림자 동안은 감시자(`ops_watchdog`·`multicoin_collectors` 매니페스트)에 **넣지 않는다**(경보가 섞인다). 로그만 매일 본다:
  `grep -c '연결 실패\|끊김\|정지 감지\|멈춤' logs/supervisor/rtcollect_*_$(date +%Y%m%d).log`.

## 8. 끝나면 지울 것(그림자를 접을 때)

1. crontab 의 `supervisor_rtcollect.sh` 줄 · 2. 감독 프로세스(먼저 감독 bash, 그다음 rtcollect -- 반대면 감독이 다시 띄운다)
3. `data/hot/rtcollect/`(그림자 파일 전부) · 4. `data/live/.supervisor_rtcollect_*.lock` · 5. `logs/supervisor/rtcollect_*`(남길 거면 보관)
`bin/rtcollect` 는 전환에 쓰니 남긴다.

## 9. 전환 순서(종류 하나씩 · 사용자 승인 뒤)

읽는 쪽(server.py `HOT_TAPE_DB`·`OKX_TAPE_DB_PATH` · `market_ctx` · `seal_to_lake` · 청산 탭 · liqmap 채점기 · 감시자)은 **경로를 안 바꾼다** --
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
