# rtorder 서버 그림자 가동에 필요한 것 (2026-11-21)

가동 여부·시각은 사용자 결정이다. 이 문서는 목록만 적는다. 그림자 = 쓰기(주문·취소·레버리지·익절·고아 정리·텔레그램)를 **보내지 않고** «안 보낸 쓰기» 목록(`그림자(안 보냄): …` 줄)에 적는다. 그래도 어딘가에서 쓰기가 새면 `http::Binance` 의 그림자 막이가 보내기 전에 panic 한다.

## 1. 빌드 두 가지

| 빌드 | 키 | 할 수 있는 것 |
|---|---|---|
| 기본 `cargo build --release` | 읽지 않음(코드가 없음) | 허브 구독·감시 틀만 돈다. 계좌를 못 읽으니(서명 요청이 «키 없음») 무장 판정이 안 된다. 사용자 스트림도 안 뜬다(키가 없으면 원본처럼 안 띄움). |
| `cargo build --release --features live` + `RTORDER_MODE` 비움(=shadow) | `BINANCE_API_KEY`·`BINANCE_SECRET_KEY` 를 **환경 변수**에서 읽음(점-env 는 안 읽는다) | 계좌·사용자 스트림까지 읽고, 쓰기는 그림자. **의미 있는 그림자는 이쪽이다.** 실전송은 `RTORDER_MODE=live` 와 `RTORDER_LIVE_CONFIRM=실주문을-보낸다` 가 둘 다 있어야 켜진다. 그림자에서는 둘 다 넣지 않는다. |

## 2. 환경 변수

| 이름 | 필수 | 뜻 · 기본 |
|---|---|---|
| `RTORDER_DIR` | 예 | Rust 전용 상태 폴더. 파이썬 폴더(`order_requests.jsonl` 이 있는 `data/live`)는 시작을 거절한다(무장 파일을 서로 덮지 않게). |
| `DASHBOARD_MANUAL_EXEC_ENABLED` | 예(`1`) | 원본 실행 스위치. 꺼져 있으면 원본 기동 훅처럼 시작 정리·사용자 스트림을 안 띄우고, 감시 틱도 아무것도 안 한다. |
| `RTORDER_LISTEN` | 아니오 | 루프백 HTTP 문(기본 `127.0.0.1:8791` · 루프백만 · 묶기 실패 = 시작 안 함 -- 7절). |
| `RTORDER_STREAM` | 권장(`1`) | 사용자 데이터 스트림 + 체결 뒤 계좌 갱신기. 스위치와 키가 있어야 뜬다. |
| `RTORDER_HUB` | 사실상 필수 | rtcore 허브 ws 주소. 감시 5분봉·1초 체결가는 허브에서만 받는다(REST 대체 없음). 허브가 없으면 손절 판정 자료가 비어 판정을 못 한다. 호가·1분 변동성도 허브 값을 먼저 쓴다(나이 한도를 넘으면 REST). |
| `RTORDER_BAN_FILE` | 권장 | 기본은 `RTORDER_DIR/binance_ban_until`. **파이썬과 같은 파일**(`data/live/binance_ban_until`, `scripts/binance_ban_guard.BAN_FILE`)을 가리키게 해야 같은 IP 의 418·429 차단을 서로 지킨다. 형식이 같다(초 단위 실수 한 줄). |
| `BINANCE_API_KEY`·`BINANCE_SECRET_KEY` | live 빌드에서 | 위 1절. |
| `DASHBOARD_MANUAL_EXEC_SYMBOL` | 아니오 | 기본 `ETHUSDC`. |
| `DASHBOARD_SIZING_CAP_EQUITY_X` | 아니오 | 원본과 같은 배수 상한 덮어쓰기. |
| `RTORDER_PERIOD` | 아니오 | 감시 주기 초(기본 2 = 원본). |
| `RTORDER_BASE`·`RTORDER_WS` | 아니오 | 기본 = 실거래소(`https://fapi.binance.com` · `wss://fstream.binance.com/private/ws/`). 시험에서만 바꾼다. |
| `TELEGRAM_BOT_TOKEN`·`TELEGRAM_CHAT_ID` | 아니오 | 그림자에서는 어차피 보내지 않는다(`TELEGRAM …` 줄로만 남음). 파이썬은 점-env 에서 읽지만 rtorder 는 환경 변수만 본다. |
| `RTORDER_TELEGRAM_BASE`·`RTORDER_MAX_TICKS`·`RTORDER_MODE`·`RTORDER_LIVE_CONFIRM` | 아니오 | 시험용 · 실전송용. 그림자에서는 넣지 않는다. |

## 3. 읽는 파일 · 쓰는 파일 (전부 `RTORDER_DIR` 안 · 밴 파일만 예외)

| 파일 | 읽기 | 쓰기 |
|---|---|---|
| `manual_bracket_state.json` | 매 틱 | 무장·정리·재시도 시각(임시 파일 → rename) |
| `manual_limit_wait_<심볼>.json`·`manual_limit_wait_<심볼>_<wid>.json` | 시작 정리(복구) | 복구가 끝나면 지움. 그림자에서는 걸지 못했으니 남긴다. |
| `order_requests.jsonl` | -- | 접수 기록(POST /api/manual-* 한 줄씩 · 원본과 같은 칸) |
| `.rtorder` | 시작 | 시작할 때 쓰는 표식(이 폴더는 rtorder 것) |
| 밴 파일(`RTORDER_BAN_FILE`) | 요청마다 | 418·429 를 받으면 |

**빈 폴더로 그림자를 켜면 무장이 하나도 없다.** 그림자에서는 문으로 들어온 진입이 거래소에 나가지 않아 체결·무장이 생기지 않는다. 파이썬과 같은 판정을 보려면 파이썬 `data/live/manual_bracket_state.json` 의 **사본**을 주기적으로 `RTORDER_DIR` 에 넣어야 한다. 어느 주기로 넣을지는 사용자 결정이다. 원본 파일을 rtorder 가 직접 쓰게 하면 안 된다.

## 4. 거래소로 나가는 읽기 요청 (파이썬 서버가 같은 계좌·같은 IP 로 같은 일을 하므로 그대로 두 배가 된다)

| 언제 | 요청 | 횟수 | 비고 |
|---|---|---|---|
| 감시 틱: 무장이 하나라도 있고 스위치가 켜져 있을 때 | 계좌 읽기 1회 = 25요청(`/fapi/v1/time` · `/fapi/v2/account` · `/fapi/v2/positionRisk` · 심볼 11개 × `/fapi/v1/userTrades`·`/fapi/v1/openOrders`) · 무게 약 50 | 10초 캐시라 **분당 6회 = 150요청 · 무게 약 300** | 무장이 없으면 0. 파이썬도 같은 조건에서 같은 양이다. |
| 체결 알림 뒤 계좌 갱신기 | 계좌 읽기 1회(위와 같음) | 연결 때 1회 + 체결마다(0.3초 묶음) · **최소 3초 간격 → 연속 체결이면 분당 최대 20회(500요청 · 무게 약 1,000)** | 파이썬 `account_refresher` 와 같은 규칙. 둘이 같은 알림을 받으니 체결마다 두 배가 된다. |
| 사용자 스트림 | `POST /fapi/v1/listenKey`(키만 · 서명 없음) | 연결마다 1회(보통 하루 1회 · 실패하면 5→10→…→300초 간격) · `PUT` 30분마다 | 같은 계좌의 listenKey 는 거래소가 하나로 돌려준다(파이썬과 공유됨). 닫기(DELETE)는 부르지 않는다(다른 쪽 연결이 끊긴다). |
| 손절선 닿음 → 청산 계획 조립 | 새 계좌(스트림 계좌를 쓸 수 있으면 0, 아니면 25요청) · 호가(`/fapi/v1/ticker/bookTicker`, 허브 값이 2초를 넘었을 때만) · 필터(`/fapi/v1/exchangeInfo`, 하루 캐시) · 1분봉(`/fapi/v1/klines`, 허브 변동성이 없을 때만) · 펀딩(`/fapi/v1/premiumIndex`) · 수수료(`/fapi/v1/commissionRate`, 서명 · 1시간 캐시) | 그림자에서는 청산을 «못 함»으로 받아 **30초마다 다시** 조립한다 → 닿아 있는 칸 하나당 분당 약 2회 | 파이썬은 실제로 청산해 포지션이 사라지므로 보통 한두 번에 끝난다. rtorder 는 다음 틱에 포지션이 없어진 것을 보고 정리로 넘어간다(정리도 그림자). |
| 시작 정리(`wait_cleanup`) | 없음 | 0 | 그림자에서는 남은 대기 주문 조회·취소를 보내지 않고 목록에만 적는다. |

허브(`RTORDER_HUB`)가 끊기면 호가·변동성이 REST 로 떨어진다. 청산 조립 때만 묻기 때문에 늘어나는 양은 작다. 감시 5분봉·1초 체결가는 REST 로 대신하지 않는다.

## 5. 키가 필요한 자리

- 서명 GET: 계좌 읽기 전부(`/fapi/v2/account`·`positionRisk`·`userTrades`·`openOrders`·`openAlgoOrders`) · `commissionRate`.
- 키만(서명 없음): `listenKey` POST·PUT.
- 쓰기(주문·취소·레버리지·algoOrder): 그림자에서는 보내지 않는다. 보내려 하면 panic.
- 키가 없는 빌드(기본)에서는 위 요청이 모두 원본의 «키 없음»과 같은 결과가 된다. 계좌는 `ok=false` 로 오고, 감시는 «못 읽음 = 모름»으로 무장을 그대로 둔다.

## 6. 배치 (2026-11-21 결정: rtorder 는 응답 서버와 **따로 둔 프로세스**)

근거는 두 가지다(조정자 결정). 설계 §4-3 이 주문 연결을 화면 응답과 같은 실행 줄에 두지 말라고 한다. 그리고 손절 감시가 화면 재배포나 응답 서버 장애와 같이 멈추면 안 된다. 127.0.0.1 한 겹의 비용(1ms 미만)은 거래소 왕복(약 41ms)보다 작다.

한 프로세스 안에서는 다음이 모두 **하나**를 같이 쓴다.
- `DeskCell` 하나: 감시 틱, 시작 정리, 접수 핸들러, 접수가 띄운 작업. 무장 칸과 대기 칸이 메모리에서도 하나다. 원본처럼 감시가 청산 중이면 같은 코인 접수는 409 를 받는다.
- 계좌 한 곳(`account::SharedAccount`): 감시 swr 10초, 접수의 새로 읽기, 체결 뒤 갱신기, 화면 `?fresh=1`(1초 안 받은 값은 재사용)이 같은 캐시와 같은 last_good 을 쓴다.
- `Front` 하나: 주문 잠금. 감시는 이 잠금이 잡혀 있으면 손절 발동을 다음 틱으로 미룬다(D9).

## 7. 루프백 HTTP 문 (`src/door.rs` · `RTORDER_LISTEN`)

- 주소는 `RTORDER_LISTEN` 이고 기본값은 `127.0.0.1:8791` 이다. 루프백이 아니면 설정 오류(종료코드 2)로 시작하지 않는다.
  - 묶기에 실패해도 시작하지 않는다(종료코드 2). 같은 주소로 둘째 rtorder 가 떠서 같은 계좌를 두 번 감시하는 일을 막는다.
  - 시작할 때 표준출력 첫 줄에 `rtorder 문 http://<주소>` 를 찍는다.
- 주문·계좌 주소 14개는 `route::Front::call` 그대로다. 본문 글자, 상태, Content-Type, Cache-Control, Allow 가 원본 aiohttp 와 바이트까지 같다.
- `GET /internal/acct` 는 `{"seq": <acct_seq>, "user_stream": {"connected", "last_event_at", "error"}}` 를 낸다. 응답 서버의 `/api/events` 가 1초마다 읽는다.
- 그 밖의 주소는 404 이고 본문은 `404: Not Found` 다.
- 접수 기록은 rtorder 가 `RTORDER_DIR/order_requests.jsonl` 에 직접 적는다. 원본 order_request_log 와 칸·순서가 같다.
  - at 과 ms 는 rtorder 가 잰다. at 은 KST 밀리초다.
  - ua 와 ip 는 응답 서버가 넘긴 머리에서 읽는다.
  - 폴더에는 `.rtorder` 표식을 둔다. 표식이 있으면 접수 기록이 있어도 파이썬 폴더로 보지 않는다.
- 접수가 내놓은 작업(진입 따라가기, 청산 따라가기)은 이 프로세스 안에서 돈다. 감시와 같은 `DeskCell` 위다.
- 연결과 작업은 감시와 같은 실행 줄에서 돌지만 서로 막지 않는다(FuturesUnordered). 클라이언트가 반쯤 보낸 요청을 쥐고 있거나, 끊거나, 쓰레기를 보내도 감시 틱은 그대로 돈다(`tests/door_bin.rs`).

### 응답 서버가 프록시할 때 넘겨야 하는 것

| 넘길 것 | 왜 |
|---|---|
| 메서드 · 경로 · **쿼리 원문**(풀지 말고 그대로) | 같은 키가 여럿이면 첫 값을 쓴다. `+` 와 `%XX` 는 rtorder 가 푼다 |
| `User-Agent` | 접수 기록 ua(200자) |
| `CF-Connecting-IP` | 접수 기록 ip 1순위(Cloudflare 를 거친 실제 접속 주소) |
| `X-Forwarded-For` | ip 2순위. 화면이 보낸 값이 없으면 **응답 서버가 자기 접속 주소(request.remote)를 넣어서** 넘긴다 -- 안 넣으면 기록 ip 가 늘 127.0.0.1 이 된다 |
| 응답의 상태 · `Content-Type` · `Cache-Control` · `Allow` · 본문 | 그대로 돌려준다. ETag 와 압축은 응답 서버 미들웨어 몫이다(원본도 바깥 미들웨어였다) |

- rtorder 문이 닫혀 있으면(죽음, 재시작 중) 응답 서버가 502 를 낸다. 손절 감시는 rtorder 안에 있어 응답 서버와 상관없이 돈다.
- 시간 제한: 요청은 칸을 5초까지 기다리고, 넘으면 409 already_working 이다(`route::DESK_WAIT_SEC`). 프록시 시간 제한은 이보다 길게(10초 이상) 잡는다.

### 한쪽이 죽으면 · 재시작 복구
- 손절은 거래소에 걸려 있지 않다. 비상 스탑은 2026-11-21 사용자 결정으로 없앴다. 손절은 감시 틱(2초)뿐이다. **rtorder 가 죽어 있는 동안에는 손절이 없다.** 거래소에 남는 것은 익절 지정가(dbtp)와 걸려 있던 진입 지정가(dbwt)뿐이다.
- 응답 서버가 죽으면 화면과 주문 접수만 멈춘다. 감시, 사용자 스트림, 도는 작업은 계속 돈다.
- rtorder 가 죽으면 무장 파일과 지정가 대기 복구 파일은 디스크에 남는다. 칸의 phase, 도는 작업, 주문 잠금은 사라진다(원본과 같다). supervisor 가 다시 띄우면 실행 스위치가 켜져 있을 때 이 순서로 돈다. ① 남은 dbwt·dbpg 를 지운다(지운 결과는 eth 대기 칸에 «재시작으로 취소»로 남는다). ② 복구 파일로 무장한다. ③ 감시를 시작한다. ④ 사용자 스트림이 붙자마자 계좌를 한 번 읽는다.
  - 청산 도중 죽었으면 무장이 남아 있다. 가격이 아직 손절선 너머면 다음 틱에 다시 쏜다.
  - 진입 따라가기 도중 죽었으면 걸려 있던 GTX 는 ①에서 지워진다. 이미 체결된 몫은 복구 파일 선으로 무장된다.
- 파이썬 서버의 실전송과 rtorder 실전송을 같은 계좌에서 동시에 켜면 손절이 두 번 나간다. 그림자일 때만 둘을 같이 돌릴 수 있다.

### 의도한 차이 (원본과 다르게 둔 것)
- 감시 틱 하나가 칸(DeskCell)을 쥔다. 손절 청산(따라가기 최대 약 2분)이 도는 틱이면 그동안 접수 요청(상태·계좌 조회 포함)이 5초 기다린 뒤 409 를 받는다. 원본은 await 사이에 다른 핸들러가 돌아 상태 조회가 바로 답했다.
  - ponytail: 넘어서려면 감시의 `run_exit` 를 칸 잠금 밖으로 빼는 재구성이 필요하다(감시 칸을 따로 잠그거나 청산 칸을 Arc 로).
- 요청 하나도 칸을 끝까지 쥔다. 조립 REST 동안 다른 주문 요청은 줄을 선다.

## 8. «그림자 기록» 제안서 -- 파이썬에 넣을 기록 코드의 모양 (고치지 않았다 · 서빙 코드라 사용자 결정)

파일은 새로 하나 쓴다: `data/live/order_shadow.jsonl`. 한 줄에 한 기록이고, 모양은 `{"rec": ..., "chain": ..., "t": time.time(), "mono": time.monotonic(), ...}` 이다. 쓰기는 한 곳에서만 한다.

```python
# live/execution/live_manual_peg_execute_20260912.py (모듈 위쪽 · 기존 _ORDER_T 옆)
SHADOW_LOG = Path(__file__).resolve().parents[2] / "data" / "live" / "order_shadow.jsonl"
_SHADOW_ON = os.getenv("ORDER_SHADOW_LOG") == "1"            # 기본 끔 -- 켜는 것은 사용자
_CHAIN: contextvars.ContextVar[str | None] = contextvars.ContextVar("chain", default=None)

def shadow_rec(rec: str, **kw) -> None:
    if not _SHADOW_ON:
        return
    try:                                                        # 기록 실패가 주문 흐름을 막으면 안 된다
        row = {"rec": rec, "chain": kw.pop("chain", None) or _CHAIN.get(), "t": time.time(), "mono": time.monotonic(), **kw}
        with SHADOW_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_redact(row), ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass

def new_chain(kind: str) -> str:
    c = f"{kind}-{time.time_ns():x}"
    _CHAIN.set(c)                                               # create_task 는 맥락을 복사한다 -- 집행 작업까지 같은 chain
    return c
```

| # | 자리 | 넣을 줄 |
|---|---|---|
| 1 | orders.py:965 `entry_then_bracket` 첫 줄 | `shadow_rec("start", op="entry", plan=copy.deepcopy(plan), limit=limit, slot=copy.deepcopy(state), file=bracket_load(), limits=sorted(p.name for p in LIVE_DIR.glob("manual_limit_wait*.json")))` |
| 2 | orders.py `api_manual_exit_submit` 의 create_task 바로 앞(2045 근처) | `shadow_rec("start", op="exit", plan=..., rearm=fraction < 1, slot=copy.deepcopy(st), file=bracket_load())` |
| 3 | orders.py:1128 `bracket_tick` 첫 줄 | `new_chain("tick")` -- start 줄은 틱 끝(13)에서 «요청이 나갔거나 파일이 바뀐 틱»일 때만 앞당겨 적는다(2초마다 쌓이지 않게: 틱 안 기록을 목록에 모았다가 조건이 맞을 때만 쓴다) |
| 4 | 실행 모듈:101 `signed` 의 `session.request` | 보내기 직전 `t_send = time.time()` · 응답에서 `raw = await response.read()`를 먼저 하고 그다음 기존 `payload = await response.json()`(본문은 캐시되어 동작이 같다) · `shadow_rec("wire", method=method, target=path + "?" + _strip_sig(params_qs), status=response.status, body=raw.decode("utf-8", "replace"), sent_at=t_send)` · 예외 갈래에서는 `error=f"{type(exc).__name__}: {exc}"` |
| 5 | `maker_price`(bookTicker) · 계좌 `_clock_offset`(/time) · `load_filters`(exchangeInfo) | 4 와 같은 칸(서명 없음) |
| 6 | orders.py:1134 swr_cached("binance_account") 뒤 | `shadow_rec("input", what="account", value=acct)` |
| 7 | orders.py:1151 load_market_history 뒤 | `shadow_rec("input", what="bars", args=[asset], value=bars)` |
| 8 | orders.py:1002 `stop_touched` · 1160 sec_prices 뒤 | `shadow_rec("input", what="sec", args=[asset], value=[r for r in rows if r[0] > since])` |
| 9 | orders.py:1195·1250 assemble_exit_plan 뒤 | `shadow_rec("input", what="exit_plan", args=[pside, frac, fresh, asset, only_symbol], value={"plan": plan, "err": err})` |
| 10 | `limit_reprice` 의 호가·필터 · `basis_for` 기준 호가 | `what="book"` / `what="filters"` |
| 11 | 실행 모듈:49 `wake_fills` | `shadow_rec("input", what="wake")` |
| 12 | `api_manual_wait_action`·`api_manual_limit_amend`·`api_manual_entry_cancel` 끝(응답 만들기 직전) | `shadow_rec("input", what="handler", chain=<그 칸의 chain>, args=[이름, dict(request.query)], value={"status": s, "body": body})` -- 칸 dict 에 `chain` 을 넣어 두면(1·2 에서) 핸들러가 꺼내 쓴다 |
| 13 | `entry_then_bracket` 끝(finally) · `settle` 뒤 · `bracket_tick` 끝 | `shadow_rec("end", slot=..., file=bracket_load(), limits=..., raised=repr(exc) or None)` |
| 14 | server.py:1371 `order_request_log` 의 row | `"chain": getattr(request, "_chain", None)` -- 접수 핸들러가 `request["chain"] = new_chain("api")` 로 남긴다 |

**가릴 칸 (`_redact`)**: 파일에 키·서명·listenKey 가 절대 남지 않게 한다.
- `target`: 서명 쿼리에서 `signature=` 와 그 뒤를 잘라 낸다(`_strip_sig`). `timestamp`·`recvWindow` 는 남긴다(재생이 시계 차이를 본다).
- 머리: `X-MBX-APIKEY` 는 적지 않는다(row 에 headers 칸 자체를 두지 않는다).
- listenKey: `/fapi/v1/listenKey` 응답 본문의 `listenKey` 값과 웹소켓 주소 끝은 `"<listenKey>"` 로 바꾼다.
- 텔레그램 토큰·채팅 번호: 주문 경로 기록에는 들어오지 않는다(보내기를 기록하지 않는다).
- 계좌 값(잔고·증거금·포지션 수량·진입가·미실현·원장 trades·income): **정확한 재생에는 진짜 값이 필요하다**(수량 반올림·상한·청산 수량이 이 값으로 정해진다). 그래서 값을 가리는 대신 두 벌로 나눈다.
  - 서버 원본 파일: chmod 600, git 밖(`data/live` 는 이미 무시 목록), 서버 밖으로 복사하지 않는다. 재생도 서버에서 돌린다.
  - 개발기로 가져올 사본: `tools/redact_shadow.py`(새로 만들 것)로 금액 칸 전부(`balance*`·`available*`·`margin*`·`wallet*`·`unrealized*`·`assets`·`trades`·`income*`)를 지우고, 서명 응답 본문은 상태 코드만 남긴다. 이 사본은 재생이 «구조(요청 순서·핸들러 응답)»만 맞춰 보고 금액이 드는 비교는 건너뛴다.
- `ua`·`ip`(14번)는 접수 기록에 이미 있다. 그림자 기록에는 넣지 않는다.

크기 어림(RULES.md 와 같다): 따라가기 120초 ≈ 요청 80 · 무장 1회 ≈ 5 · 감시 발동 ≈ 10 → 하루 접수 수십 건이면 수 MB.
