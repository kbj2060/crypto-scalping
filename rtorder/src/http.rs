//! 바이낸스 선물 REST 전송 -- 원본 `signed()`(live_manual_peg_execute_20260912.py) · `_clock_offset`(live_binance_account_20260910.py) ·
//! `maker_price` · `load_filters`(live_manual_peg_entry_20260912.py) 를 그대로(2026-11-20 둘째 조각). **어디에도 연결하지 않았다.**
//! 주소는 생성자 인자 -- 시험은 프로세스 안 가짜 서버(127.0.0.1)만 가리킨다. 이 크레이트에는 실제 주소로 붙는 실행 파일·예제가 없다.
//!
//! HTTP 클라이언트 = ureq 3(rustls·ring, 동기) 를 tokio 의 blocking 풀에서 부른다. 고른 까닭: 개발기 cargo 캐시에 이미 있어 새로 내려받을 것이 없다
//! (reqwest·hyper 는 없다). ponytail: 요청마다 blocking 풀 한 번 건너뜀(수십 µs) -- 거래소 왕복 ~41ms 앞에서 작다. 비동기 클라이언트가 필요해지면 reqwest 로.
//! 연결 재사용(keep-alive)은 ureq Agent 의 연결 풀(aiohttp 세션 커넥터와 같은 역할) · 시간초과는 요청 전체(aiohttp ClientTimeout(total) 과 같은 뜻).

use crate::exec::{Common, Exchange, LimitOrder, Order, OrderRef, Reply};
use crate::plan::Filters;
use crate::sign::{self, err_value, error_of, py_float, sign_query, str_or_empty, P};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const FAPI: &str = "https://fapi.binance.com";
/// 대시보드 공용 aiohttp 세션의 시간 제한(규칙 목록 A6 «공용 세션은 10초 제한»).
pub const TIMEOUT: Duration = Duration::from_secs(10);
/// aiohttp 는 멱등 메서드가 연결 뒤 끊기면 **한 번 다시 보낸다**(client.py IDEMPOTENT_METHODS · ClientOSError·ServerDisconnectedError).
/// 🔴POST(주문 접수)는 다시 보내지 않는다 -- 그 끊김은 «나갔는지 모름»으로 호출부(resolve_unknown_post)가 표식으로 찾는다.
const IDEMPOTENT: [&str; 6] = ["GET", "HEAD", "OPTIONS", "TRACE", "PUT", "DELETE"];

struct Resp {
    status: u16,
    ctype: String,
    retry_after: Option<String>,
    body: String,
}

enum Fail {
    Retryable(String), // 연결은 됐는데 끊김(aiohttp 가 멱등이면 한 번 더 보냄)
    Final(String),
}

fn once(agent: &ureq::Agent, method: &str, url: &str, key: Option<&str>) -> Result<Resp, Fail> {
    let call = |m: &str| -> Result<ureq::http::Response<ureq::Body>, ureq::Error> {
        macro_rules! go {
            ($b:expr, $send:ident) => {{
                let b = $b;
                let b = match key { Some(k) => b.header("X-MBX-APIKEY", k), None => b };
                b.$send()
            }};
        }
        match m {
            "GET" => go!(agent.get(url), call),
            "DELETE" => go!(agent.delete(url), call),
            "POST" => go!(agent.post(url), send_empty),
            "PUT" => go!(agent.put(url), send_empty),
            _ => unreachable!("메서드 {m}"),
        }
    };
    let classify = |e: ureq::Error| -> Fail {
        use std::io::ErrorKind as K;
        match e {
            ureq::Error::Timeout(_) => Fail::Final("TimeoutError: ".into()), // asyncio.TimeoutError 는 문구가 비어 있다
            ureq::Error::ConnectionFailed | ureq::Error::HostNotFound => Fail::Final(connect_err(url, "Connect call failed")),
            ureq::Error::Io(io) if io.kind() == K::ConnectionRefused => Fail::Final(connect_err(url, &io.to_string())),
            ureq::Error::Io(io) if io.kind() == K::ConnectionReset => Fail::Retryable("ClientOSError: [Errno 104] Connection reset by peer".into()),
            ureq::Error::Io(_) | ureq::Error::Protocol(_) => Fail::Retryable("ServerDisconnectedError: Server disconnected".into()),
            e => Fail::Final(format!("ClientError: {e}")),
        }
    };
    let mut r = call(method).map_err(classify)?;
    let h = |n: &str| r.headers().get(n).and_then(|v| v.to_str().ok()).map(str::to_string);
    let (status, ctype, retry_after) = (r.status().as_u16(), h("content-type").unwrap_or_default(), h("retry-after"));
    let body = r.body_mut().read_to_string().map_err(classify)?;
    Ok(Resp { status, ctype, retry_after, body })
}

fn connect_err(url: &str, why: &str) -> String {
    let hp = url.split("://").nth(1).and_then(|r| r.split('/').next()).unwrap_or("");
    format!("ClientConnectorError: Cannot connect to host {hp} ssl:default [{why}]")
}

/// aiohttp `response.json()`: 본문 형식이 application/json(·…+json) 이 아니면 ContentTypeError · 빈 본문 = None · 그 밖 json.loads.
fn json_of(r: &Resp, url: &str) -> Result<Value, String> {
    let ct = r.ctype.to_lowercase();
    let rest = ct.strip_prefix("application/").unwrap_or("\u{0}");
    let ok = rest.starts_with("json")
        || rest.find("+json").is_some_and(|i| i > 0 && rest[..i].chars().all(|c| c.is_alphanumeric() || "_.+-".contains(c)));
    if !ok {
        let msg = format!("Attempt to decode JSON with unexpected mimetype: {ct}");
        return Err(format!("ContentTypeError: {}, message={}, url={}", r.status, sign::py_repr(&Value::String(msg)), sign::py_repr(&Value::String(url.into()))));
    }
    let t = r.body.trim();
    if t.is_empty() {
        return Ok(Value::Null);
    }
    // ponytail: 문구는 파이썬 json 과 다르다(앞머리 «JSONDecodeError: » 만 같다 -- «나갔는지 모름» 판정엔 그것으로 충분)
    serde_json::from_str(t).map_err(|e| format!("JSONDecodeError: {e}"))
}

fn sys_time() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

pub struct Binance {
    base: String,
    key: String,
    secret: String,
    /// `_clock_offset` 이 준 값(ms). 파이썬처럼 집행 한 번에 한 번 맞춘다(`sync_clock`).
    pub offset: i64,
    agent: ureq::Agent,
    /// `time.time()`. 시험은 가짜 시계를 넣는다(표식·서명 시각이 정답지와 같게).
    pub clock: Arc<dyn Fn() -> f64 + Send + Sync>,
    t0: Instant,
    /// 체결 알림이 오면 `notify_waiters()` -- 기다리던 `nap` 이 일찍 깬다(원본 wake_fills · 알림은 앞당기기만, A10).
    pub wake: Arc<tokio::sync::Notify>,
    ban_file: PathBuf,
    filters: HashMap<String, Filters>,
    pub(crate) fees: HashMap<String, (f64, f64, f64, &'static str)>,
}

impl Binance {
    /// 키·비밀값·주소를 받는다. 밴 기록 파일(scripts/binance_ban_guard.py 의 BAN_FILE, 운영은 `data/live/binance_ban_until`)도 인자다.
    pub fn new(base: &str, key: &str, secret: &str, timeout: Duration, ban_file: PathBuf) -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            .http_status_as_error(false)
            .proxy(None) // aiohttp 기본(trust_env=False)과 같이 환경 프록시를 안 탄다
            .build()
            .into();
        Binance { base: base.trim_end_matches('/').into(), key: key.into(), secret: secret.into(), offset: 0, agent, clock: Arc::new(sys_time),
                  t0: Instant::now(), wake: Arc::new(tokio::sync::Notify::new()), ban_file, filters: HashMap::new(), fees: HashMap::new() }
    }

    /// 운영용: 키를 환경(`BINANCE_API_KEY`·`BINANCE_SECRET_KEY`)에서 읽는다. 시험은 쓰지 않는다(키는 `new` 로 주입).
    /// ponytail: 원본은 호출마다 환경을 다시 읽는다 -- 여기는 만들 때 한 번. 실행 중 키를 바꿀 일이 생기면 새로 만든다.
    pub fn from_env(base: &str, ban_file: PathBuf) -> Self {
        let g = |k: &str| std::env::var(k).unwrap_or_default();
        Self::new(base, &g("BINANCE_API_KEY"), &g("BINANCE_SECRET_KEY"), TIMEOUT, ban_file)
    }

    /// 원본 `if not (key and secret)`.
    pub fn has_keys(&self) -> bool {
        !self.key.is_empty() && !self.secret.is_empty()
    }

    pub fn time(&self) -> f64 {
        (self.clock)()
    }

    /// 밴 남은 초(ban_guard.ban_remaining -- 공용 파일이라 실제 시계). 계좌 조회 `_get` 이 보내기 전에 본다.
    pub fn ban_left(&self) -> f64 {
        sign::ban_remaining(&self.ban_file, sys_time())
    }

    async fn send(&self, method: &'static str, url: &str, signed: bool) -> Result<Resp, String> {
        let mut retry = IDEMPOTENT.contains(&method);
        loop {
            let (agent, u, k) = (self.agent.clone(), url.to_string(), signed.then(|| self.key.clone()));
            let r = tokio::task::spawn_blocking(move || once(&agent, method, &u, k.as_deref())).await.map_err(|e| format!("JoinError: {e}"))?;
            match r {
                Ok(x) => return Ok(x),
                Err(Fail::Retryable(_)) if retry => retry = false,
                Err(Fail::Retryable(e) | Fail::Final(e)) => return Err(e),
            }
        }
    }

    /// 원본 `signed()`: 서명 요청 하나. **예외 대신 값** -- 실패는 `{"__error__": ...}`(주문 흐름이 중간에 끊기면 «보냈는지 모르는» 상태가 남는다).
    /// 200 이 아니면: 418/429 는 밴 기록만(주문·취소는 막지 않는다) → `"<상태> [<code>] <msg>"`(sign::http_error).
    pub async fn signed(&mut self, method: &'static str, path: &str, params: &[(&str, P)]) -> Value {
        let ts = sign::ms(self.time()) + self.offset;
        let url = format!("{}{}?{}", self.base, path, sign_query(params, &self.secret, ts));
        let r = match self.send(method, &url, true).await {
            Ok(r) => r,
            Err(e) => return err_value(e), // 네트워크/TLS/시간초과 -- 숫자로 시작하지 않는다(= «나갔는지 모름»)
        };
        let payload = match json_of(&r, &url) {
            Ok(v) => v,
            Err(e) => return err_value(e), // 원본: JSON 을 못 읽으면 밴 기록 전에 예외 -- 418 이라도 기록 안 됨(원본 그대로)
        };
        if r.status != 200 {
            // ban_guard 는 프로세스 사이 공용 파일이라 **실제 시계**(time.time)로 적는다 -- 주입한 시계가 아니다
            sign::ban_note(&self.ban_file, r.status, &sign::py_str(&payload), r.retry_after.as_deref(), sys_time());
            return err_value(sign::http_error(r.status, &payload));
        }
        payload
    }

    /// 서명 없는 GET(aiohttp `session.get(url, params=...)` -- 상태 코드를 안 보고 JSON 을 읽는다). 실패 = 파이썬 예외 문자열.
    pub(crate) async fn public(&self, path: &str, query: &[(&str, P)]) -> Result<Value, String> {
        let url = if query.is_empty() { format!("{}{path}", self.base) } else { format!("{}{path}?{}", self.base, sign::urlencode(query)) };
        let r = self.send("GET", &url, false).await?;
        json_of(&r, &url)
    }

    /// 원본 `_clock_offset`: `/fapi/v1/time` 의 serverTime − 지금(ms). 무엇이든 실패하면 0.
    /// 🔴거래소는 timestamp 가 ~1초만 **앞서도** 거부한다(recvWindow 와 무관) -- WSL 시계가 1000ms 앞섰던 적 있다(2026-09-10).
    pub async fn clock_offset(&self) -> i64 {
        match self.public("/fapi/v1/time", &[]).await {
            Ok(v) => v.get("serverTime").and_then(|s| py_float(s).ok()).map_or(0, |s| s as i64 - sign::ms(self.time())),
            Err(_) => 0,
        }
    }

    pub async fn sync_clock(&mut self) {
        self.offset = self.clock_offset().await;
    }

    /// 원본 `maker_price` 의 호가 읽기: **주문이 나가는 그 심볼**의 최우선 (매수, 매도)(E14 · 2026-09-19·20 USDT 호가로 USDC 주문 → -5022 5/5).
    pub async fn book_ticker(&self, symbol: &str) -> Result<(f64, f64), String> {
        let b = self.public("/fapi/v1/ticker/bookTicker", &[("symbol", P::s(symbol))]).await?;
        let f = |k: &str| b.get(k).ok_or_else(|| format!("KeyError: '{k}'")).and_then(py_float);
        Ok((f("bidPrice")?, f("askPrice")?))
    }

    /// 원본 `load_filters`: 수량·가격 격자. 프로세스 수명 동안 캐시(심볼 규격은 안 변한다). 없는 심볼 = RuntimeError.
    pub async fn load_filters(&mut self, symbol: &str) -> Result<Filters, String> {
        if let Some(f) = self.filters.get(symbol) {
            return Ok(f.clone());
        }
        let info = self.public("/fapi/v1/exchangeInfo", &[]).await?;
        let none = Vec::new();
        for item in info.get("symbols").and_then(Value::as_array).unwrap_or(&none) {
            if item.get("symbol").and_then(Value::as_str) != Some(symbol) {
                continue;
            }
            let mut out = Filters { step: 0.001, tick: 0.01, min_qty: 0.0, min_notional: 0.0 };
            for f in item.get("filters").and_then(Value::as_array).unwrap_or(&none) {
                let g = |k: &str| f.get(k).ok_or_else(|| format!("KeyError: '{k}'")).and_then(py_float);
                match f.get("filterType").and_then(Value::as_str) {
                    Some("LOT_SIZE") => (out.step, out.min_qty) = (g("stepSize")?, g("minQty")?),
                    Some("PRICE_FILTER") => out.tick = g("tickSize")?,
                    Some("MIN_NOTIONAL" | "NOTIONAL") => {
                        let v = [f.get("notional"), f.get("minNotional")].into_iter().find(|v| sign::truthy(*v)).flatten();
                        out.min_notional = v.map_or(Ok(0.0), py_float)?;
                    }
                    _ => {}
                }
            }
            self.filters.insert(symbol.into(), out.clone());
            return Ok(out);
        }
        Err(format!("RuntimeError: symbol not in exchangeInfo: {symbol}"))
    }
}

/// 주문 응답 dict → `Reply`. ponytail: 원본은 dict 가 아니거나 executedQty 가 숫자가 아니면 예외로 집행이 죽는다(settle → error) --
/// 여기는 숫자로 시작하지 않는 오류 문자열로 돌려 상태 기계가 «나갔는지 모름»(접수)·«조회 실패 재시도»(조회)로 다룬다.
fn reply(r: Value) -> Reply {
    if let Some(e) = error_of(&r) {
        return Err(e.into());
    }
    if !r.is_object() {
        return Err(format!("TypeError: argument of type '{}' is not iterable", if r.is_null() { "NoneType" } else { "list" }));
    }
    Ok(Order { order_id: r.get("orderId").and_then(Value::as_i64), status: str_or_empty(&r, "status"), executed_qty: sign::executed_qty(&r)? })
}

/// 원본 집행 함수들의 요청 모양: 주문 번호로 묻고 지울 때는 `{**common, "orderId": oid}`(심볼·측면·포지션 측면까지),
/// 표식으로 묻고 지울 때는 `{"symbol", "origClientOrderId"}`(resolve_unknown_post).
fn ref_params(c: &Common<'_>, r: &OrderRef) -> Vec<(&'static str, P)> {
    match r {
        OrderRef::Id(id) => vec![("symbol", P::s(c.symbol)), ("side", P::s(c.side)), ("positionSide", P::s(c.position_side)), ("orderId", id.map_or(P::None, P::I))],
        OrderRef::Client(cid) => vec![("symbol", P::s(c.symbol)), ("origClientOrderId", P::s(cid))],
    }
}

impl Exchange for Binance {
    async fn place_limit_gtx(&mut self, o: &LimitOrder) -> Reply {
        let p = [("symbol", P::s(&o.symbol)), ("side", P::s(&o.side)), ("positionSide", P::s(&o.position_side)), ("type", P::s("LIMIT")),
                 ("timeInForce", P::s("GTX")), ("price", P::F(o.price)), ("quantity", P::F(o.quantity)), ("newClientOrderId", P::s(&o.client_order_id))];
        reply(self.signed("POST", "/fapi/v1/order", &p).await)
    }
    async fn get_order(&mut self, c: &Common<'_>, r: &OrderRef) -> Reply {
        reply(self.signed("GET", "/fapi/v1/order", &ref_params(c, r)).await)
    }
    async fn cancel_order(&mut self, c: &Common<'_>, r: &OrderRef) -> Reply {
        reply(self.signed("DELETE", "/fapi/v1/order", &ref_params(c, r)).await)
    }
    async fn book(&mut self, symbol: &str) -> Result<(f64, f64), String> {
        self.book_ticker(symbol).await
    }
    fn now(&self) -> f64 {
        self.t0.elapsed().as_secs_f64()
    }
    fn wall_ms(&self) -> i64 {
        sign::ms(self.time())
    }
    /// 원본 `nap`: asyncio.wait 로 기다리다 wake_fills 면 일찍 깬다(R7: wait_for 는 3.11 에서 깨움·취소가 같은 틱이면 취소를 삼켰다 gh-86296 --
    /// Rust future 는 버리면(drop) 바로 끝나므로 그 문제가 없다).
    async fn nap(&mut self, sec: f64) {
        let w = self.wake.clone();
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs_f64(sec)) => {}
            _ = w.notified() => {}
        }
    }
    async fn sleep(&mut self, sec: f64) {
        tokio::time::sleep(Duration::from_secs_f64(sec)).await
    }
}
