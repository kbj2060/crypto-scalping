# rtcore 재시작 복원 — 조사와 설계 (2026-11-20 · 구현 안 함)

지금 rtcore 는 재시작하면 풋프린트 과거 봉·1초 수급 칸·청산 칸이 빈다(재시작 뒤 본 체결만 센다). 파이썬 대시보드가 재시작 때 무엇으로 되살리는지 정리하고, 코어가 같은 값을 되살릴 길을 고른다.

## 1. 파이썬이 되살리는 것 (dashboard/coin_flow.py · server.py, 2026-11-20 판)

| 무엇 | 언제 | 원천(경로·형식) | 함수 |
|---|---|---|---|
| 풋프린트 봉 칸(창 144봉) | 수집기 시작 직후 1회 | `data/live/footprint_<코인>.json` — 파이썬 자신이 30초마다(`ready` 일 때만)·종료 때 덮어쓴 스냅샷 | `footprint_load` coin_flow.py:419~440 · 쓰기 `footprint_save` 396~417 |
| 스냅샷 뒤 ~ 첫 체결 앞 공백(ETH) | 웹소켓 첫 체결 때 | 바이낸스 REST `fapi/v1/aggTrades`(가중치 20 · 예산 1,200회 · 2.5초 간격) | `footprint_backfill` 442~531 |
| 같은 공백(ETH 외) | 웹소켓 첫 체결 때 | 체결 테이프 SQLite `data/hot/binance_tape.sqlite` 표 `trade_tape_1s` | `footprint_tape_restore` 533~570 → `okx_tape_footprint` 129~203 |
| 1초 수급 바이낸스 선물 칸(11분) | 라이브 첫 초 t0 뒤 | 같은 테이프, `[t0−660, t0)` | `supply_1s_restore` 973~1018 → `tape_seconds` 204~216 |
| 1초 수급 OKX 칸 · OKX OI | 같음 | `data/hot/okx_tape.sqlite` `trade_tape_1s` · `data/hot/okx_ctx.sqlite` `okx_oi` | 같음 |
| 바이낸스 강제청산(11분 + 5분봉 24h) | 시작 1회 | `data/hot/binance_ctx.sqlite` `liquidations`(수집기 `live_binance_ctx_collector_20261001.py`) | server.py `liq_events_load` 1785~ |
| 봉 안 델타 `dhl`(링 밖 닫힌 봉) | 요청마다, 봉 끝 6분 뒤 | `binance_tape.sqlite` `trade_tape_1s` 초별 `sum(buy_qty) − sum(sell_qty)` 누적 | server.py `fp_dhl` 3952~3991 |
| 현물 1초 칸 | — | **되살리지 않는다**(테이프 없음, supply_1s_restore 주석) | — |

**스냅샷 json 형식**(`footprint_save`): `{"bar_seconds": 300, "bucket": <칸 폭>, "symbol": "ETHUSDT", "last_ms": <마지막 웹소켓 체결 T>, "cells_v": 3, "bars": {"<봉 시작 초>": {"<round(가격/칸 폭)>": [매수, 매도, 고래매수, 고래매도, 리테일매수, 리테일매도]}}}` — 마지막 144봉, 반올림 전 값. 읽을 때 `bar_seconds`·`bucket`·`symbol`·`cells_v` 가 다르면 통째로 버리고, 143봉보다 오래된 봉은 버린다. `last_ms` 가 공백 메우기의 시작점이다(`ready` 일 때만 저장 = «last_ms 까지 빈 곳 없음»).

**테이프 표 형식**(`live_trade_tape_collector_20260916.py` `_init_sqlite` 415~):
`trade_tape_1s(symbol TEXT, ts_sec INTEGER, price_bin INTEGER, buy_qty, sell_qty, buy_n, sell_n, buy_max, sell_max, retail_buy_qty, retail_sell_qty, whale_buy_qty, whale_sell_qty, whale_*_n, retail_*_n, order_*_n)`, PK `(symbol, ts_sec, price_bin)`, WAL. 초는 완결된 것만 5초마다 한 번 쓴다(`FLUSH_SECONDS`). 칸 폭은 `BUCKETS`(ETH 0.1 · BTC 1.0 · SOL 0.01 · XRP 0.0001 …) — 풋프린트 칸(ETH 0.5)의 약수. 고래·리테일은 라이브와 같은 되묶기(`TakerOrderAggregator`)·같은 경계(`size_bands`)로 센 주문 단위다. 공백은 `gaps` 표, 1분봉 대조는 `verify_1m`.

옮길 때 그대로 지켜야 할 규칙(파이썬 원문):
- 테이프 칸 → 풋프린트 칸 키: `int(round(round(price_bin × 테이프칸, 10) / 칸 폭))` (2026-10-20 SOL 동률 사고 — SQL 반올림 금지).
- 공백 경계 두 초는 뺀다: `lo = max(창 바닥, last_ms//1000 + 1)`, `hi = 첫 체결 ms//1000`(배타) — 끊기기 직전 초와 붙은 첫 초는 라이브가 일부 셌다. 두 초 일부 체결은 빠진다(5분봉의 1% 미만).
- 테이프가 `hi − 1` 까지 따라잡을 때까지 5초 간격 최대 12번 기다린다(5초마다 쓰므로).
- 테이프 복원은 칸에 **더한다**(`+=`), 스냅샷 복원은 봉을 **통째로** 쓴다.
- 1초 칸 복원은 라이브가 이미 가진 초를 **덮지 않는다**(`setdefault`). 가격 칸은 그 초 마지막 체결가가 아니라 칸 가중평균이다(테이프에 순서가 없음 — 오차 < 테이프 칸 1개).
- 🔴`okx_tape_footprint` 는 범위에 비례해 메모리를 쓴다 — 2026-11-20 개발기 WSL 정지(13.7GB). 범위를 창(144봉)으로 자르고 읽는다.

## 2. 코어가 같은 값을 되살리는 선택지

| | 읽는 것 | 값이 파이썬과 같은가 | 비용·위험 |
|---|---|---|---|
| (가) 파이썬 스냅샷 json 읽기(읽기 전용) | `data/live/footprint_<코인>.json` | 봉 칸은 **파이썬이 저장한 그대로**(최대 30초 낡음). 그 뒤 공백·1초 칸·dhl·청산은 못 채운다 | 새 의존성 없음(serde_json) · 0.8ms/파일. 🔴파이썬이 살아 있어야 한다 — §11 은 파이썬을 끄는 쪽이라 마지막엔 원천이 사라진다. 파이썬이 `ready` 가 아니면 30초 넘게 낡는다 |
| (나) 체결 테이프 SQLite 읽기(읽기 전용) | `binance_tape.sqlite`(봉·1초·dhl) · `okx_tape.sqlite`·`okx_ctx.sqlite`(OKX 칸·OI·청산) · `binance_ctx.sqlite`(청산) · `bybit_liq.sqlite` | 파이썬 ETH 외 코인 경로와 **같은 원천·같은 식**. ETH 는 파이썬이 REST 로 메우므로 공백 봉이 같은 값이라는 보장은 없다(테이프 = @trade 되묶기 · REST = aggTrades) — 라이브 규칙에는 테이프 쪽이 더 가깝다 | **새 의존성 `rusqlite`**(rtcollect 가 이미 0.37 bundled 를 쓴다 — 워크스페이스로 합치면 한 벌). 24h 창을 한 번에 읽지 말고 144봉·660초로 자른다. 수집기가 멈췄으면 `gaps` 만큼 빈다 |
| (다) 코어가 자기 스냅샷을 주기적으로 쓰기 | `data/live/rtcore_footprint_<코인>.json`(코어 전용) | 코어 자신의 값 그대로(봉 칸 + dhl + 1초 칸 660초). 쓴 뒤 ~ 재시작 사이 공백은 (나)로 메운다 | 쓰기 8코인 × 약 255KB / 30초(coin_flow.py:75 «144봉 스냅샷 ~255KB») · 원자적 쓰기(tmp → rename) · 파이썬과 독립 |
| (라) REST 백필 | 바이낸스 aggTrades | 파이썬 ETH 와 같은 원천 | **쓰지 않는다** — 서버 IP 한도를 봇과 공유(2026-09-26 418 밴 27분) |

## 3. 권고

**(다) + (나)** — 파이썬 ETH 외 코인 경로(스냅샷 → 테이프로 공백 메우기)와 같은 모양을 코어 안에서 만든다.
1. 코어가 `ready`(빈 곳 없음 = 시작 뒤 공백을 다 메움)일 때만 30초마다 자기 스냅샷을 쓴다(봉 칸 반올림 전 값 + `dhl` + 마지막 체결 `last_ms` + `bucket`·`cells_v`). 종료코드 70 으로 끝나기 직전에도 한 번.
2. 시작 때 스냅샷을 읽고(설정이 다르면 버림), `(last_ms, 첫 체결)` 공백과 1초 칸 660초를 테이프에서 메운다(위 «그대로 지켜야 할 규칙»). 청산은 `binance_ctx`·`okx_ctx`·`bybit_liq` 에서 660초 + 5분봉 창.
3. 메운 봉은 `dirty` 로 올려 다음 밀기가 싣는다(footprint.rs `restore` 자리 주석). 메우는 동안 `ready=false` — 화면은 «수집 중» 을 본다.
4. 시험: 합성 테이프 SQLite(정답지 생성기가 파이썬 `okx_tape_footprint`·`tape_seconds` 를 그대로 돌려 만든 값)와 일치 · 스냅샷 쓰기 → 읽기 왕복 비트 일치 · 그림자 대조기(`tools/shadow_footprint.py`)의 비교 시작 봉을 «복원 뒤 첫 봉» 으로 앞당겨 실서버 대조.

(가)는 테이프 의존성을 못 쓸 때만 임시로 — 파이썬을 끄는 순간 원천이 없어지고, 파이썬 값을 되읽어 대조하면 «파이썬과 파이썬» 을 비교하게 된다.

결정이 필요한 것: `rusqlite` 를 rtcore 의존성에 넣는 것(지금 조각은 의존성 추가 금지) · 코어 스냅샷 경로와 쓰기 주기.
