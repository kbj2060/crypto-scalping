//! 루프백 HTTP 문(2026-11-21 · 배치 결정 «rtorder 는 응답 서버와 따로 둔 프로세스»): 응답 서버(rtserve)가 주문·계좌 주소 14개를 여기로 넘긴다.
//! - 14개 = `route::Front::call` 그대로(글자·상태·머리) · `GET /internal/acct` = {seq, user_stream} · 그 밖 = 404(Front 가 낸다).
//! - 접수 기록(G18) = `RTORDER_DIR/order_requests.jsonl` 에 원본 order_request_log 와 같은 줄(at = KST 밀리초 · ms · ua 200자 · ip = CF-Connecting-IP → X-Forwarded-For → 접속 주소).
//! - 접수가 내놓은 작업(`Task`) = `runner::run_task` 를 이 문 안에서 돌린다(감시 틱과 같은 `DeskCell`).
//! 연결·작업은 같은 실행 줄의 FuturesUnordered(spawn 없음 -- current_thread). **127.0.0.1 이 아니면 bin 이 시작을 거절한다.**
use crate::account::SharedAccount;
use crate::app::{GuardEx, Ops};
use crate::assemble::{Cached, DeskIo};
use crate::http::Binance;
use crate::intake::Task;
use crate::pyfmt::{json_dumps, json_dumps_as};
use crate::route::Front;
use crate::runner::DeskCell;
use crate::userstream::Hub;
use crate::watch::{iso_utc, WatchIo};
use crate::wire::Wired;
use bytes::Bytes;
use futures_util::stream::{FuturesUnordered, StreamExt};
use http_body_util::Full;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use serde_json::{json, Value};
use std::future::Future;
use std::io::Write;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

/// 접수 핸들러가 닿는 바깥(실제 구현): 계좌 = `SharedAccount`(감시와 같은 캐시) · 손절·익절 함수 = `Ops`(그림자면 적기만).
pub struct DoorIo {
    pub ops: Ops,
    pub rest: Binance,
    pub acct: Arc<SharedAccount>,
    /// 사용자 스트림을 켰을 때만(스트림 계좌 대조).
    pub stream: Option<Arc<Hub>>,
}

impl DeskIo for DoorIo {
    fn now(&self) -> f64 {
        self.ops.bx.time()
    }
    fn exec_enabled(&self) -> bool {
        crate::plan::exec_enabled()
    }
    /// 원본 assemble_*_plan: fresh = produce_account(캐시에 안 넣음) · 아니면 swr 캐시.
    async fn account(&mut self, fresh: bool) -> Result<Value, String> {
        if fresh {
            self.acct.produce(&mut self.rest, self.stream.as_deref()).await
        } else {
            self.acct.cached(&mut self.rest, self.stream.as_deref()).await
        }
    }
    async fn account_screen(&mut self, fresh: bool) -> Result<Value, String> {
        if fresh {
            self.acct.refresh(&mut self.rest, self.stream.as_deref(), true).await
        } else {
            self.acct.cached(&mut self.rest, self.stream.as_deref()).await
        }
    }
    fn sec_prices(&mut self, asset: &str) -> Option<Vec<(f64, f64)>> {
        WatchIo::sec_prices(&mut self.ops, asset).ok().flatten()
    }
    async fn clear_bracket(&mut self, s: &str, p: &str, a: bool, b: bool) -> Result<Value, String> {
        WatchIo::clear_bracket(&mut self.ops, s, p, a, b).await
    }
    async fn place_bracket(&mut self, spec: &Value, s: &str, p: &str, strict: bool) -> Result<Value, String> {
        WatchIo::place_bracket(&mut self.ops, spec, s, p, strict).await
    }
    async fn place_tp_ladder(&mut self, legs: &Value, s: &str, p: &str) -> Result<Value, String> {
        WatchIo::place_tp_ladder(&mut self.ops, legs, s, p).await
    }
}

/// 문이 쥐는 것(프로세스에 하나).
pub struct Door<'a> {
    pub desk: Arc<DeskCell>,
    pub hub: Arc<Hub>,
    pub front: Arc<Front>,
    pub feed: Arc<std::sync::Mutex<crate::coreval::Feed>>,
    pub dir: PathBuf,
    /// 요청이 같이 쓰는 바깥(요청은 어차피 DeskCell 에서 줄 선다 -- 필터·수수료 캐시를 요청 사이에 남긴다).
    pub io: tokio::sync::Mutex<(DoorIo, Cached<Binance>)>,
    /// 작업 하나마다 새 바깥(감시와 같은 계좌·스트림 · 쓰기는 GuardEx/Ops 가 그림자를 지킨다).
    pub make_task_io: Box<dyn Fn() -> (Wired<Ops, Binance, Cached<Binance>, GuardEx>, GuardEx, Cached<Binance>) + 'a>,
}

type Resp = hyper::Response<Full<Bytes>>;

/// 쿼리 한 조각 풀기(yarl: '+' = 빈칸 · %XX = 바이트 · 깨진 %는 그대로 · UTF-8 아니면 대체 문자).
fn unquote(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < b.len() && b[i + 1].is_ascii_hexdigit() && b[i + 2].is_ascii_hexdigit() => {
                out.push(u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap(), 16).unwrap());
                i += 3;
                continue;
            }
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub fn parse_query(raw: &str) -> Vec<(String, String)> {
    raw.split('&').filter(|p| !p.is_empty()).map(|p| match p.split_once('=') {
        Some((k, v)) => (unquote(k), unquote(v)),
        None => (unquote(p), String::new()),
    }).collect()
}

/// 원본 datetime.fromtimestamp(t, KST).isoformat(timespec="milliseconds").
pub fn iso_kst_ms(t: f64) -> String {
    let k = t + 9.0 * 3600.0;
    let mut s = k.floor();
    let mut us = ((k - s) * 1e6).round_ties_even() as i64;
    if us >= 1_000_000 {
        (s, us) = (s + 1.0, us - 1_000_000);
    }
    format!("{}.{:03}+09:00", &iso_utc(s)[..19], us / 1000)
}

fn wall() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64()
}

fn header(h: &hyper::HeaderMap, k: &str) -> Option<String> {
    h.get(k).and_then(|v| v.to_str().ok()).map(String::from).filter(|s| !s.is_empty())
}

impl Door<'_> {
    async fn answer(&self, method: &str, path: &str, raw_q: &str, h: &hyper::HeaderMap, peer: std::net::SocketAddr) -> (Resp, Vec<Task>) {
        let t0 = wall();
        if path == "/internal/acct" && (method == "GET" || method == "HEAD") {
            let us = self.hub.us.lock().unwrap().clone();
            let body = json_dumps(&json!({"seq": self.hub.acct_seq(), "user_stream": {"connected": us.connected, "last_event_at": us.last_event_at, "error": us.error}}));
            return (resp(200, "application/json; charset=utf-8", &[("Cache-Control", "no-cache".into())], if method == "HEAD" { String::new() } else { body }), vec![]);
        }
        let pairs = parse_query(raw_q);
        let r = {
            let mut g = self.io.lock().await;
            let (io, q) = &mut *g;
            q.now = io.ops.bx.time();
            crate::coreval::sync_cached(&self.feed.lock().unwrap(), q);
            self.front.call(&self.desk, &self.hub, q, io, method, path, &pairs).await
        };
        if let Some(mut row) = r.log.clone() {
            // 원본 칸 순서: at · path · query · status · error · ms · ua · ip
            let ms = (1000.0 * (wall() - t0)).round_ties_even() as i64;
            let ua: String = header(h, "user-agent").unwrap_or_default().chars().take(200).collect();
            let ip = header(h, "cf-connecting-ip").or_else(|| header(h, "x-forwarded-for")).unwrap_or_else(|| peer.ip().to_string());
            let mut o = serde_json::Map::new();
            o.insert("at".into(), json!(iso_kst_ms(t0)));
            for k in ["path", "query", "status", "error"] {
                o.insert(k.into(), row[k].take());
            }
            o.insert("ms".into(), json!(ms));
            o.insert("ua".into(), json!(ua));
            o.insert("ip".into(), json!(ip));
            let line = json_dumps_as(&Value::Object(o), false) + "\n";
            if let Err(e) = std::fs::OpenOptions::new().create(true).append(true).open(self.dir.join("order_requests.jsonl")).and_then(|mut f| f.write_all(line.as_bytes())) {
                println!("order_request_log: 못 씀 {e}");
            }
        }
        let headers: Vec<(&str, String)> = r.headers.iter().map(|(k, v)| (*k, v.clone())).collect();
        (resp(r.status, r.content_type, &headers, r.body), r.task)
    }

    /// 받기 → 연결마다 http1 · 접수 작업은 같은 줄에서 끝까지. `stop` 이 오면 원본 aiohttp 종료 순서대로:
    /// ① 받기를 멈추고 ② 도는 접수 작업을 취소(원본 on_shutdown 의 refresh_tasks 취소 -- R5 · 대기 복구 파일은 남아 다음 기동이 이어 받는다)
    /// ③ 답을 만드는 중인 요청은 최대 2초 기다린다(원본 shutdown_timeout=2 -- R6) · 쉬는 keep-alive 연결은 기다리지 않는다.
    pub async fn serve(&self, l: tokio::net::TcpListener, stop: &crate::app::Stop) {
        let inflight = std::cell::Cell::new(0usize);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Task>();
        let mut conns: FuturesUnordered<Pin<Box<dyn Future<Output = ()> + '_>>> = FuturesUnordered::new();
        let mut tasks: FuturesUnordered<Pin<Box<dyn Future<Output = ()> + '_>>> = FuturesUnordered::new();
        loop {
            if stop.flag.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }
            tokio::select! {
                a = l.accept() => {
                    let Ok((s, peer)) = a else { continue };
                    let (tx, inflight) = (tx.clone(), &inflight);
                    let svc = service_fn(move |req: hyper::Request<hyper::body::Incoming>| {
                        let tx = tx.clone();
                        async move {
                            let _busy = Busy::new(inflight);
                            let (m, path, q) = (req.method().as_str().to_string(), req.uri().path().to_string(), req.uri().query().unwrap_or("").to_string());
                            let (resp, task) = self.answer(&m, &path, &q, req.headers(), peer).await;
                            for t in task {
                                let _ = tx.send(t);
                            }
                            Ok::<_, std::convert::Infallible>(resp)
                        }
                    });
                    conns.push(Box::pin(async move {
                        let _ = http1::Builder::new().serve_connection(TokioIo::new(s), svc).await;
                    }));
                }
                Some(t) = rx.recv() => {
                    let (mut wio, mut ex, mut q) = (self.make_task_io)();
                    tasks.push(Box::pin(async move { crate::runner::run_task(&self.desk, &mut ex, &mut wio, &mut q, t).await }));
                }
                Some(_) = conns.next(), if !conns.is_empty() => {}
                Some(_) = tasks.next(), if !tasks.is_empty() => {}
                _ = stop.notify.notified() => break,
            }
        }
        drop(l);
        tasks.clear(); // R5
        let until = std::time::Instant::now() + std::time::Duration::from_secs(SHUTDOWN_WAIT_SEC);
        while inflight.get() > 0 && std::time::Instant::now() < until {
            tokio::select! {
                Some(_) = conns.next(), if !conns.is_empty() => {}
                _ = tokio::time::sleep(std::time::Duration::from_millis(10)) => {}
            }
        }
    }
}

/// 원본 web.run_app(shutdown_timeout=2).
pub const SHUTDOWN_WAIT_SEC: u64 = 2;

/// 답을 만드는 중인 요청 수(끊겨 버려져도 줄어든다).
struct Busy<'a>(&'a std::cell::Cell<usize>);
impl<'a> Busy<'a> {
    fn new(c: &'a std::cell::Cell<usize>) -> Self {
        c.set(c.get() + 1);
        Busy(c)
    }
}
impl Drop for Busy<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() - 1);
    }
}

fn resp(status: u16, ctype: &str, headers: &[(&str, String)], body: String) -> Resp {
    let mut r = hyper::Response::new(Full::new(Bytes::from(body)));
    *r.status_mut() = hyper::StatusCode::from_u16(status).unwrap_or(hyper::StatusCode::INTERNAL_SERVER_ERROR);
    let h = r.headers_mut();
    h.insert("content-type", hyper::header::HeaderValue::from_str(ctype).unwrap());
    for (k, v) in headers {
        if let (Ok(k), Ok(v)) = (hyper::header::HeaderName::from_bytes(k.to_ascii_lowercase().as_bytes()), hyper::header::HeaderValue::from_str(v)) {
            h.insert(k, v);
        }
    }
    r
}

/// 시험·bin 용: 루프백 주소만(127.0.0.0/8 · ::1).
pub fn loopback_only(addr: &std::net::SocketAddr) -> Result<(), String> {
    if addr.ip().is_loopback() {
        Ok(())
    } else {
        Err(format!("RTORDER_LISTEN 은 루프백만 됩니다(받은 값 {addr}) -- 응답 서버가 같은 기계에서 넘긴다"))
    }
}
