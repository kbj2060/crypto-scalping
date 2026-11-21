//! rtserve -- 대시보드 응답 서버(파이썬 aiohttp, 8787)의 Rust 판(꾸러미 R6 · 첫 조각 2026-11-21).
//!
//! 원본 흐름을 그대로 본뜬다: 라우터 → 핸들러 → 미들웨어(안쪽 json_compress_etag → 바깥 static_asset_headers) → 응답 준비(압축 협상·길이).
//! 핸들러가 던진 HTTP 예외(302·400·403·404·405)는 미들웨어를 지나지 않고 aiohttp 기본 글자(`404: Not Found`)로 나간다 -- 원본과 같다.
//! 인증은 원본에도 없다(127.0.0.1 에만 묶고, 공개 경로는 Cloudflare 터널 + Access 가 막는다) -- 여기서도 루프백 아니면 시작을 거절한다.
//! 시험: tests/http_parity.rs(정답지 = tools/gen_http_fixture.py 가 진짜 aiohttp 객체로 뜬 바이트).

pub mod api;
pub mod api2;
pub mod api3;
pub mod ctx;
pub mod ctx_micro;
pub mod ctx_x;
pub mod fpd;
pub mod heatmap;
pub mod hub;
pub mod model;
pub mod order;
pub mod py;
pub mod raster_read;
pub mod swr;
pub mod text;

use std::convert::Infallible;
use std::fs;
use std::net::SocketAddr;
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, UNIX_EPOCH};

use bytes::Bytes;
use flate2::{Compress, Compression, FlushCompress};
use http_body_util::{combinators::BoxBody, BodyExt, Full, StreamBody};
use hyper::body::Frame;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use sha2::{Digest, Sha256};
use tokio::net::TcpListener;

/// 원본 server.py:1282 -- 내용 해시 버스터·no-cache·압축을 받는 화면 파일
pub const ASSET_FILES: [&str; 5] = ["app.js", "styles.css", "fp_daily.js", "liq_profile.js", "bubble_map.js"];
/// orders.py:338 MARKET_SYMBOLS 의 열쇠(DASHBOARD_ASSETS 가 이 안에서만 고른다 · client-timing 의 sym 허용 목록)
pub const MARKET_ASSETS: [&str; 9] = ["eth", "sol", "btc", "xrp", "hype", "sndk", "soxl", "skhynix", "mu"];
/// dashboard/coin_flow.py FLOW_SPECS 열쇠 -- 흐름 엔진(체결·풋프린트)이 있는 코인. 켜지는 것 = eth + 이 중 DASHBOARD_ASSETS(server.py:1721)
pub const FLOW_COINS: [&str; 8] = ["eth", "sol", "xrp", "btc", "sndk", "soxl", "skhynix", "mu"];
/// scripts/coin_config.py COIN_CONFIG 열쇠(2026-11-21) -- `?asset=` 신뢰 경계 허용 목록
pub const COIN_CONFIG: [&str; 9] = ["eth", "btc", "sol", "xrp", "hype", "sndk", "soxl", "skhynix", "mu"];

pub type Clock = Arc<dyn Fn() -> f64 + Send + Sync>;

#[derive(Clone)]
pub struct Config {
    /// 저장소 뿌리(원본 REPO_ROOT) -- data/live·data/hot·dashboard/live 를 여기서 찾는다
    pub root: PathBuf,
    /// 원본 DASHBOARD_ASSETS(꺼진 코인 탭을 index.html 에서 hidden 으로)
    pub assets: Vec<String>,
    pub now: Clock,
    /// env VAPID_PUBLIC_KEY 값 · VAPID_PRIVATE_KEY 가 비지 않았는지(값은 들고 있지 않는다 -- 발송은 파이썬 몫)
    pub vapid_public: String,
    pub vapid_private_set: bool,
    /// 로그 한 줄(원본 print · client_timing) -- 실행 파일은 표준 출력, 시험은 모아 본다
    pub log: Arc<dyn Fn(&str) + Send + Sync>,
    /// 파일을 고치는 핸들러(메모·오답·구독)의 읽기-고치기-쓰기를 한 줄로(원본은 이벤트 루프 하나라 저절로 한 줄이었다)
    pub write_lock: Arc<Mutex<()>>,
    /// client_timing 0.2초 상한의 마지막 시각(원본 _client_timing_at · time.monotonic)
    pub timing_at: Arc<Mutex<Option<std::time::Instant>>>,
    /// rtcore 허브에서 받아 쌓은 상태(허브가 없으면 빈 채로 -- 엔드포인트는 원본의 «아직 없음» 모양을 낸다)
    pub live: Arc<hub::Live>,
    /// /api/events 발행기
    pub events: Arc<hub::Events>,
    /// 원본 swr_cached(묵은 값 먼저·뒤에서 갱신) -- 키마다 원본과 같은 ttl·max_stale
    pub swr: Arc<swr::Swr>,
    /// rtorder 문(주문·계좌 14개를 넘길 곳 · 루프백만) -- None 이면 그 주소들은 502
    pub rtorder: Option<SocketAddr>,
    /// 넷째 조각 모델 핸들러(model.rs): 바이낸스 선물 REST 뿌리(원본 FAPI) · 모델 JSON 자리(None = data/live/rtmodel) · FRED 키(원본 env FRED_API_KEY) ·
    /// ext = 시험만(코인베이스·바이낸스 현물·FRED 주소의 뿌리를 가짜 서버로) · 모델·REST 캐시 상태
    pub rest: String,
    pub models: Option<PathBuf>,
    pub fred_key: Option<String>,
    pub ext: Option<String>,
    pub models_state: Arc<model::State>,
    /// 병행 가동(파이썬 8787 과 같은 data/live): 파일을 쓰는 POST 4개는 503 readonly_shadow · 접수 기록(order_requests.jsonl)도 안 적는다 -- 같은 파일을 둘이 쓰지 않게
    pub readonly: bool,
    /// 일봉·4시간 풋프린트(fpd.rs) 코인마다 하나 -- 처음 요청 때 만든다(원본 fp_daily 표)
    pub fpd: Arc<std::sync::OnceLock<std::collections::HashMap<String, fpd::Fpd>>>,
}

impl Config {
    pub fn new(root: PathBuf, assets: Vec<String>, now: Clock) -> Config {
        Config {
            root,
            assets,
            now,
            vapid_public: String::new(),
            vapid_private_set: false,
            log: Arc::new(|l| println!("{l}")),
            write_lock: Arc::new(Mutex::new(())),
            timing_at: Arc::new(Mutex::new(None)),
            live: Arc::new(hub::Live::default()),
            events: Arc::new(hub::Events { clients: Mutex::new(Vec::new()), latest: Mutex::new((serde_json::Value::Null, serde_json::json!({}))), acct_seq: Mutex::new(0) }),
            swr: Arc::new(swr::Swr::default()),
            rtorder: None,
            rest: String::new(), // 빈 값 = REST 꺼짐(모델 주소 · main 의 RTSERVE_REST 로 켠다)
            models: None,
            fred_key: None,
            ext: None,
            models_state: Arc::new(model::State::default()),
            readonly: false,
            fpd: Arc::default(),
        }
    }
    /// 켜진 흐름 엔진 코인(원본 `flows` 의 열쇠 · FLOW_SPECS 순서)
    pub fn flows(&self) -> Vec<&'static str> {
        FLOW_COINS.iter().copied().filter(|a| *a == "eth" || self.assets.iter().any(|x| x == a)).collect()
    }
    pub fn live_dir(&self) -> PathBuf {
        self.root.join("data/live")
    }
    pub fn dashboard_dir(&self) -> PathBuf {
        self.root.join("dashboard/live")
    }
}

/// 원본 server.py:424 -- `DASHBOARD_ASSETS` env(기본 "eth,btc,sol,xrp,sndk,soxl,skhynix,mu") 중 아는 코인만 · 비면 ["eth"]
pub fn parse_assets(env: Option<&str>) -> Vec<String> {
    let v: Vec<String> = env
        .unwrap_or("eth,btc,sol,xrp,sndk,soxl,skhynix,mu")
        .split(',')
        .map(|a| a.trim().to_lowercase())
        .filter(|a| MARKET_ASSETS.contains(&a.as_str()))
        .collect();
    if v.is_empty() { vec!["eth".into()] } else { v }
}

pub struct Req {
    pub method: String,
    /// 풀린 경로(쿼리 없음)
    pub path: String,
    pub query: Vec<(String, String)>,
    pub headers: Vec<(String, String)>, // 이름 소문자
    /// 쿼리 원문(풀지 않은 것 · rtorder 로 그대로 넘긴다)
    pub raw_query: String,
    pub body: Vec<u8>,
    /// 접속한 쪽 IP(원본 request.remote)
    pub peer: String,
}

impl Req {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
    /// aiohttp `request.query.get(k)`: 첫 값
    pub fn q(&self, k: &str) -> Option<&str> {
        self.query.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str())
    }
}

/// aiohttp 응답 하나(준비 전). `stream` = StreamResponse(FileResponse) -- JSON 미들웨어가 건드리지 않고, 압축하면 청크 전송.
#[derive(Debug, Clone, Default)]
pub struct Resp {
    pub status: u16,
    pub reason: Option<String>,
    pub headers: Vec<(String, String)>,
    pub body: Option<Vec<u8>>,
    pub stream: bool,
    pub compress: bool,
    pub close: bool,
    /// 핸들러가 «던진» 응답(HTTPException·500) -- 미들웨어를 안 지난다
    pub exc: bool,
    /// rtorder 가 받아 답한 응답(접수 기록은 rtorder 가 적었다 -- rtserve 는 안 적는다)
    pub upstream: bool,
}

impl Resp {
    pub fn get(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
    pub fn set(&mut self, name: &str, value: impl Into<String>) {
        let value = value.into();
        match self.headers.iter_mut().find(|(k, _)| k.eq_ignore_ascii_case(name)) {
            Some(h) => h.1 = value,
            None => self.headers.push((name.into(), value)),
        }
    }
    fn remove(&mut self, name: &str) {
        self.headers.retain(|(k, _)| !k.eq_ignore_ascii_case(name));
    }
    /// aiohttp `Response(body=…, content_type=ct, headers=NOCACHE)`
    pub fn bytes(ct: &str, body: Vec<u8>) -> Resp {
        Resp { status: 200, headers: vec![("Cache-Control".into(), "no-cache".into()), ("Content-Type".into(), ct.into())], body: Some(body), ..Default::default() }
    }
    /// aiohttp `web.json_response(payload, headers=NOCACHE)` -- `json.dumps` 기본 모양(", " · ": " · ensure_ascii)
    pub fn json(v: &serde_json::Value) -> Resp {
        Resp::bytes("application/json; charset=utf-8", rtcollect::py_dumps_default(v).into_bytes())
    }
    fn mime(&self) -> String {
        self.get("Content-Type").map(|c| c.split(';').next().unwrap_or("").trim().to_lowercase()).unwrap_or_default()
    }
}

/// aiohttp HTTPException 을 그대로 응답으로 낸 꼴(web_protocol: `resp = exc`) -- 본문 `"{status}: {reason}"` · text/plain
pub fn http_exc(status: u16, reason: Option<&str>, extra: &[(&str, &str)]) -> Resp {
    let reason = reason.map(str::to_string).unwrap_or_else(|| phrase(status).into());
    let mut headers = vec![("Content-Type".to_string(), "text/plain; charset=utf-8".to_string())];
    headers.extend(extra.iter().map(|(k, v)| (k.to_string(), v.to_string())));
    Resp { status, body: Some(format!("{status}: {reason}").into_bytes()), reason: Some(reason), headers, exc: true, ..Default::default() }
}

/// aiohttp web_protocol.handle_error(500) 의 글자(Accept 에 text/html 이 없을 때) · 연결 닫음
pub fn server_error() -> Resp {
    Resp {
        status: 500,
        headers: vec![("Content-Type".into(), "text/plain; charset=utf-8".into())],
        body: Some(b"500 Internal Server Error\n\nServer got itself in trouble".to_vec()),
        close: true,
        exc: true,
        ..Default::default()
    }
}

fn phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        302 => "Found",
        304 => "Not Modified",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Request Entity Too Large", // 파이썬 HTTPStatus 글자(hyper 는 «Payload Too Large»)
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "",
    }
}

// ───────────────────────── 라우터 ─────────────────────────

pub type Handler = fn(&Config, &Req) -> Resp;

/// 원본 등록부(server.py 4932~5025) 중 옮긴 것: (주소, 메서드, 핸들러). add_get 은 HEAD 도 받는다.
fn routes(p: &str) -> Vec<(&'static str, Handler)> {
    let mut v: Vec<(&'static str, Handler)> = match p {
        "/" => vec![("GET", |_, _| http_exc(302, None, &[("Location", "/dashboard/live/")]))],
        "/dashboard/live" | "/dashboard/live/" | "/dashboard/live/index.html" => vec![("GET", dashboard_index)],
        "/dashboard/live/sw.js" | "/dashboard/live/manifest.webmanifest" => vec![("GET", pwa_asset)],
        _ => Vec::new(),
    };
    v.extend(api::handler(p).map(|h| ("GET", h)));
    match p {
        "/api/supply-1s" => v.push(("GET", |c, r| hub::api_supply(c, &c.live, r))),
        "/api/footprint" => v.push(("GET", |c, r| hub::api_footprint(c, &c.live, r))),
        "/api/oi-5m" => v.push(("GET", hub::api_oi_5m)),
        "/api/liquidation-5m-history" => v.push(("GET", hub::api_liquidation_5m_history)),
        "/api/situation" => v.push(("GET", hub::api_situation)),
        _ => {}
    }
    v.extend(api2::routes(p));
    v.extend(api3::routes(p));
    v.extend(order::routes(p));
    v.extend(model::routes(p));
    v
}

fn route(cfg: &Config, req: &Req) -> Resp {
    let p = req.path.as_str();
    let m = if req.method == "HEAD" { "GET" } else { req.method.as_str() };
    let rs = routes(p);
    let static_rest = if rs.is_empty() { p.strip_prefix("/dashboard/live/") } else { None };
    if let Some(rest) = static_rest {
        return if m == "GET" { static_file(cfg, req, rest) } else { http_exc(405, None, &[("Allow", "GET,HEAD")]) };
    }
    if rs.is_empty() {
        return http_exc(404, None, &[]);
    }
    if let Some((_, h)) = rs.iter().find(|(x, _)| *x == m) {
        return h(cfg, req);
    }
    // aiohttp: Allow = 그 주소가 받는 메서드(GET 이면 HEAD 도) 이름 순
    let mut allow: Vec<&str> = rs.iter().flat_map(|(x, _)| if *x == "GET" { vec!["GET", "HEAD"] } else { vec![*x] }).collect();
    allow.sort();
    allow.dedup();
    http_exc(405, None, &[("Allow", &allow.join(","))])
}

/// 요청 하나 → 연결에 낼 응답(압축·길이까지 정한 것)
/// 병행 가동 때 막는 쓰기 주소(api2: 메모·오답·푸시 구독 -- 파이썬과 같은 파일을 rename 으로 덮는다)
pub const READONLY_BLOCK: [&str; 4] = ["/api/acct-notes", "/api/acct-wrong", "/api/push/subscribe", "/api/push/unsubscribe"];

pub fn handle(cfg: &Config, req: &Req) -> Resp {
    let t0 = (cfg.now)();
    if cfg.readonly && req.method == "POST" && READONLY_BLOCK.contains(&req.path.as_str()) {
        let mut r = Resp::json(&serde_json::json!({"ok": false, "error": "readonly_shadow"}));
        r.status = 503;
        return prepare(req, r);
    }
    let mut r = route(cfg, req);
    if !r.exc {
        // 미들웨어 둘(안쪽부터) -- 예외 응답(302·4xx·500)은 원본처럼 미들웨어를 안 지난다
        r = json_compress_etag(req, r);
        r = static_asset_headers(req, r);
    }
    if req.method == "POST" && req.path.starts_with("/api/manual-") && !r.upstream && !cfg.readonly {
        api2::order_request_log(cfg, req, &r, t0);
    }
    prepare(req, r)
}

/// 원본 json_compress_etag(server.py:1325)
fn json_compress_etag(req: &Req, mut r: Resp) -> Resp {
    if req.method != "GET" || r.stream || r.status != 200 || r.body.is_none() || r.mime() != "application/json" {
        return r;
    }
    if r.get("ETag").is_none() {
        let etag = format!("W/\"body-{}\"", &hex(&Sha256::digest(r.body.as_ref().unwrap()))[..16]);
        r.set("ETag", etag.clone());
        if r.get("Cache-Control").is_none() {
            r.set("Cache-Control", "no-cache");
        }
        if etag_matches(req, &etag) {
            let cc = r.get("Cache-Control").unwrap().to_string();
            return Resp { status: 304, headers: vec![("ETag".into(), etag), ("Cache-Control".into(), cc)], ..Default::default() };
        }
    }
    r.compress = true;
    r
}

/// 원본 etag_matches(server.py:1060): If-None-Match 를 쉼표로 갈라 앞뒤 공백을 벗긴 글자가 `*` 이거나 같으면
fn etag_matches(req: &Req, etag: &str) -> bool {
    req.header("if-none-match").unwrap_or("").split(',').any(|c| {
        let c = c.trim_matches(|ch: char| ch.is_whitespace());
        c == "*" || c == etag
    })
}

/// 원본 static_asset_headers(server.py:1287) -- 요청 경로의 마지막 이름으로 고른다
fn static_asset_headers(req: &Req, mut r: Resp) -> Resp {
    let name = req.path.trim_end_matches('/').rsplit('/').next().unwrap_or("");
    if ASSET_FILES.contains(&name) {
        r.set("Cache-Control", "no-cache");
        r.compress = true;
    } else if name.ends_with(".ttf") {
        r.set("Cache-Control", "public, max-age=31536000, immutable");
    } else if name.ends_with(".html") {
        r.set("Cache-Control", "no-cache");
    }
    r
}

/// aiohttp 응답 준비: 압축 협상(Accept-Encoding 에 «deflate» 글자가 있으면 deflate, 아니면 «gzip» · 부분 글자 일치) · 길이.
/// 304 는 본문·길이가 없다(압축을 켠 StreamResponse 는 304 에도 Content-Encoding 이 붙는다 -- 원본 그대로).
fn prepare(req: &Req, mut r: Resp) -> Resp {
    let not_modified = r.status == 304;
    if r.compress {
        let ae = req.header("accept-encoding").unwrap_or("").to_lowercase();
        let coding = if ae.contains("deflate") { Some("deflate") } else if ae.contains("gzip") { Some("gzip") } else { None };
        if let Some(c) = coding {
            r.set("Content-Encoding", c);
            r.remove("Content-Length");
            if !not_modified {
                r.body = Some(zlib(r.body.as_deref().unwrap_or(b""), c == "gzip"));
            }
        }
    }
    if not_modified {
        r.body = None;
        r.remove("Content-Length");
    } else if r.stream {
        // 길이를 모르는 StreamResponse(압축했거나 본문 없이 낸 404·403) = 청크(to_hyper) · Content-Type 기본값
        if r.get("Content-Type").is_none() {
            r.set("Content-Type", "application/octet-stream");
        }
    } else {
        let n = r.body.as_ref().map_or(0, Vec::len);
        r.set("Content-Length", n.to_string());
    }
    if r.close {
        r.set("Connection", "close");
    }
    r
}

fn prepare_exc(mut r: Resp) -> Resp {
    let n = r.body.as_ref().map_or(0, Vec::len);
    r.set("Content-Length", n.to_string());
    r
}

/// C zlib 수준 6(aiohttp ZLibCompressor 기본) -- deflate = zlib 머리(wbits 15) · gzip = zlib 의 gzip 머리(wbits 31)
pub fn zlib(data: &[u8], gzip: bool) -> Vec<u8> {
    let mut c = if gzip { Compress::new_gzip(Compression::new(6), 15) } else { Compress::new(Compression::new(6), true) };
    let mut out = Vec::with_capacity(data.len() / 2 + 64);
    loop {
        let before = c.total_in() as usize;
        out.reserve(out.len().max(64));
        let st = c.compress_vec(&data[before..], &mut out, FlushCompress::Finish).expect("zlib");
        if st == flate2::Status::StreamEnd {
            return out;
        }
    }
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

// ───────────────────────── 화면 문서·정적 파일 ─────────────────────────

/// 원본 _asset_buster(server.py:3593): ASSET_FILES 내용의 sha256 앞 12자(없는 파일은 이름 글자)
/// ponytail: 파일마다 (수정 시각 ns · 크기)가 같으면 지난 값 -- 병행 대조에서 매 요청 sha256 이 rtserve 를 파이썬보다 느리게 했다(10ms 대 3.6ms).
///   수정 시각·크기를 안 바꾸고 내용만 바꾸는 쓰기면 옛 버스터가 남는다(배포는 rename 이라 해당 없음).
fn asset_buster(dir: &Path) -> String {
    type Memo = std::sync::Mutex<Option<(PathBuf, Vec<Option<(u128, u64)>>, String)>>;
    static MEMO: std::sync::OnceLock<Memo> = std::sync::OnceLock::new();
    let sig: Vec<Option<(u128, u64)>> = ASSET_FILES.iter().map(|n| fs::metadata(dir.join(n)).ok().and_then(|m| {
        Some((m.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos(), m.len()))
    })).collect();
    let memo = MEMO.get_or_init(Default::default);
    if let Some((d, s, v)) = memo.lock().unwrap().as_ref() {
        if d == dir && *s == sig {
            return v.clone();
        }
    }
    let v = asset_buster_compute(dir);
    *memo.lock().unwrap() = Some((dir.to_path_buf(), sig, v.clone()));
    v
}

fn asset_buster_compute(dir: &Path) -> String {
    let mut h = Sha256::new();
    for name in ASSET_FILES {
        match fs::read(dir.join(name)) {
            Ok(b) => h.update(&b),
            Err(_) => h.update(name.as_bytes()),
        }
    }
    hex(&h.finalize())[..12].to_string()
}

/// `re.sub(r'(app\.js|…)\?v=[A-Za-z0-9._-]+', name?v=<v>)` 를 손으로(왼쪽부터 · 같은 자리에서는 목록 순서)
fn replace_busters(html: &str, v: &str) -> String {
    let b = html.as_bytes();
    let mut out = String::with_capacity(html.len());
    let mut i = 0;
    'outer: while i < b.len() {
        for name in ASSET_FILES {
            let pre = format!("{name}?v=");
            if b[i..].starts_with(pre.as_bytes()) {
                let s = i + pre.len();
                let n = b[s..].iter().take_while(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-')).count();
                if n > 0 {
                    out.push_str(&pre);
                    out.push_str(v);
                    i = s + n;
                    continue 'outer;
                }
            }
        }
        let ch = html[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// 원본 _hide_off_assets(server.py:3609): `<button[^>]*\bdata-asset="([a-z]+)"[^>]*>` 의 코인이 DASHBOARD_ASSETS 밖이면 ` hidden>`
fn hide_off_assets(html: &str, assets: &[String]) -> String {
    let mut out = String::with_capacity(html.len() + 64);
    let mut i = 0;
    while let Some(off) = html[i..].find("<button") {
        let start = i + off;
        let Some(gt) = html[start..].find('>').map(|g| start + g) else { break };
        let tag = &html[start..=gt];
        out.push_str(&html[i..start]);
        // 탐욕 [^>]* → 태그 안 «마지막» data-asset="[a-z]+" (앞 글자가 단어 글자가 아닐 것)
        let inner = &tag[7..tag.len() - 1];
        let mut asset = None;
        for (pos, _) in inner.match_indices("data-asset=\"") {
            let word_before = inner[..pos].chars().next_back().is_some_and(|c| c.is_alphanumeric() || c == '_');
            let rest = &inner[pos + 12..];
            let n = rest.bytes().take_while(u8::is_ascii_lowercase).count();
            if !word_before && n > 0 && rest.as_bytes().get(n) == Some(&b'"') {
                asset = Some(&rest[..n]);
            }
        }
        match asset {
            Some(a) if !(assets.iter().any(|x| x == a) || tag.contains(" hidden")) => {
                out.push_str(&tag[..tag.len() - 1]);
                out.push_str(" hidden>");
                i = gt + 1;
            }
            Some(_) => {
                out.push_str(tag);
                i = gt + 1;
            }
            None => {
                // 이 자리에서는 맞지 않는다 -- 정규식은 한 글자 뒤에서 다시 찾는다
                out.push('<');
                i = start + 1;
            }
        }
    }
    out.push_str(&html[i..]);
    out
}

/// 원본 dashboard_index(server.py:3625)
fn dashboard_index(cfg: &Config, _: &Req) -> Resp {
    let dir = cfg.dashboard_dir();
    let Ok(html) = fs::read_to_string(dir.join("index.html")) else { return server_error() };
    let html = hide_off_assets(&replace_busters(&html, &asset_buster(&dir)), &cfg.assets);
    Resp {
        status: 200,
        headers: vec![("Content-Type".into(), "text/html; charset=utf-8".into()), ("Cache-Control".into(), "no-cache".into())],
        body: Some(html.into_bytes()),
        compress: true,
        ..Default::default()
    }
}

/// 원본 pwa_asset(server.py:3635): FileResponse + no-cache
fn pwa_asset(cfg: &Config, req: &Req) -> Resp {
    let name = req.path.rsplit('/').next().unwrap();
    let mut r = file_response(req, &cfg.dashboard_dir().join(name));
    r.headers.insert(0, ("Cache-Control".into(), "no-cache".into()));
    r
}

/// 원본 add_static("/dashboard/live/", show_index=False) -- 🔴신뢰 경계: 폴더 밖으로 나가는 경로는 404(심볼릭 링크로 나가는 것도)
fn static_file(cfg: &Config, req: &Req, rel: &str) -> Resp {
    if rel.starts_with('/') || rel.contains('\0') {
        return http_exc(404, None, &[]);
    }
    let Ok(dir) = cfg.dashboard_dir().canonicalize() else { return http_exc(404, None, &[]) };
    let joined = dir.join(rel);
    let resolved = match joined.canonicalize() {
        Ok(p) => p,
        Err(_) => lexical(&joined), // 없는 파일: 글자로만 정리(원본 Path.resolve(strict=False))
    };
    if !resolved.starts_with(&dir) {
        return http_exc(404, None, &[]);
    }
    if resolved.is_dir() {
        return http_exc(403, None, &[]);
    }
    file_response(req, &resolved)
}

fn lexical(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            c => out.push(c),
        }
    }
    out
}

/// Python 3.11 mimetypes 기본표(aiohttp FileResponse 가 쓰는 MimeTypes()) 중 웹 파일 -- 없는 확장자는 octet-stream
fn guess_type(path: &Path) -> &'static str {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    match ext.as_str() {
        "js" | "mjs" => "application/javascript",
        "css" => "text/css",
        "html" | "htm" => "text/html",
        "json" => "application/json",
        "png" => "image/png",
        "svg" => "image/svg+xml",
        "ico" => "image/vnd.microsoft.icon",
        "webmanifest" => "application/manifest+json",
        "txt" => "text/plain",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "xml" => "text/xml",
        "wasm" => "application/wasm",
        "pdf" => "application/pdf",
        "csv" => "text/csv",
        "mp3" => "audio/mpeg",
        "mp4" => "video/mp4",
        _ => "application/octet-stream",
    }
}

/// aiohttp FileResponse: ETag `"{mtime_ns:x}-{size:x}"` · Last-Modified = ceil(st_mtime) · If-None-Match(약한 비교)·If-Modified-Since 면 304.
/// ponytail: Range(206)·If-Match/If-Unmodified-Since(412)·미리 압축한 .gz/.br 형제 파일은 안 옮겼다 -- 화면이 쓰지 않는다(필요하면 원본
///   web_fileresponse._make_response 순서대로 더한다).
fn file_response(req: &Req, path: &Path) -> Resp {
    let stream_status = |status: u16| Resp { status, stream: true, body: Some(Vec::new()), ..Default::default() };
    let meta = match fs::metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => return stream_status(403),
        Err(_) => return stream_status(404),
    };
    if !meta.is_file() {
        return stream_status(403);
    }
    let mtime_ns = meta.mtime() as i128 * 1_000_000_000 + meta.mtime_nsec() as i128;
    let etag = format!("{:x}-{:x}", mtime_ns, meta.size());
    let st_mtime = meta.mtime() as f64 + meta.mtime_nsec() as f64 * 1e-9; // CPython fill_time 과 같은 식
    let last_modified = httpdate::fmt_http_date(UNIX_EPOCH + Duration::from_secs(st_mtime.ceil() as u64));
    let inm = req.header("if-none-match");
    let not_modified = match inm {
        Some(v) => parse_etags(v).iter().any(|t| t == "*" || *t == etag),
        None => req
            .header("if-modified-since")
            .and_then(|v| httpdate::parse_http_date(v).ok())
            .is_some_and(|t| st_mtime <= t.duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())),
    };
    if not_modified {
        return Resp { status: 304, stream: true, headers: vec![("Etag".into(), format!("\"{etag}\"")), ("Last-Modified".into(), last_modified)], ..Default::default() };
    }
    let Ok(body) = fs::read(path) else { return stream_status(404) };
    Resp {
        status: 200,
        stream: true,
        headers: vec![
            ("Content-Type".into(), guess_type(path).into()),
            ("Etag".into(), format!("\"{etag}\"")),
            ("Last-Modified".into(), last_modified),
            ("Content-Length".into(), body.len().to_string()),
            ("Accept-Ranges".into(), "bytes".into()),
        ],
        body: Some(body),
        ..Default::default()
    }
}

/// aiohttp request.if_none_match: `*` 또는 따옴표 친 ETag 목록(W/ 약한 표시는 벗김) -- 값만
fn parse_etags(v: &str) -> Vec<String> {
    if v.trim() == "*" {
        return vec!["*".into()];
    }
    v.split(',')
        .filter_map(|t| {
            let t = t.trim();
            let t = t.strip_prefix("W/").unwrap_or(t);
            t.strip_prefix('"').and_then(|t| t.strip_suffix('"')).map(str::to_string)
        })
        .collect()
}

// ───────────────────────── HTTP(hyper) ─────────────────────────

fn decode(s: &str, plus: bool) -> String {
    let s = if plus { s.replace('+', " ") } else { s.to_string() };
    percent_encoding::percent_decode_str(&s).decode_utf8_lossy().into_owned()
}

pub fn parse_req<B>(r: &hyper::Request<B>, body: Vec<u8>, peer: String) -> Req {
    let uri = r.uri();
    let query = uri
        .query()
        .unwrap_or("")
        .split('&')
        .filter(|kv| !kv.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            (decode(k, true), decode(v, true))
        })
        .collect();
    Req {
        method: r.method().as_str().to_string(),
        path: decode(uri.path(), false),
        query,
        raw_query: uri.query().unwrap_or("").to_string(),
        headers: r.headers().iter().map(|(k, v)| (k.as_str().to_string(), String::from_utf8_lossy(v.as_bytes()).into_owned())).collect(),
        body,
        peer,
    }
}

type Body = BoxBody<Bytes, Infallible>;

pub fn to_hyper(r: Resp) -> hyper::Response<Body> {
    let chunked = r.stream && r.status != 304 && r.get("Content-Length").is_none();
    let data = Bytes::from(r.body.unwrap_or_default());
    let body: Body = if chunked {
        StreamBody::new(futures_util::stream::iter([Ok::<_, Infallible>(Frame::data(data))])).boxed()
    } else {
        Full::new(data).boxed()
    };
    let mut out = hyper::Response::new(body);
    *out.status_mut() = hyper::StatusCode::from_u16(r.status).unwrap();
    let canonical = out.status().canonical_reason().unwrap_or("");
    if let Some(reason) = r.reason.or_else(|| Some(phrase(r.status).to_string())).filter(|x| !x.is_empty() && x != canonical) {
        if let Ok(p) = hyper::ext::ReasonPhrase::try_from(reason.into_bytes()) {
            out.extensions_mut().insert(p);
        }
    }
    let h = out.headers_mut();
    for (k, v) in r.headers {
        if let (Ok(k), Ok(v)) = (hyper::header::HeaderName::from_bytes(k.to_lowercase().as_bytes()), hyper::header::HeaderValue::from_str(&v)) {
            h.append(k, v);
        }
    }
    out
}

/// 127.0.0.1 에 묶고 요청마다 블로킹 스레드에서 처리(핸들러는 파일·SQLite 를 읽는다)
pub async fn serve(listener: TcpListener, cfg: Config) -> std::io::Result<()> {
    let cfg = Arc::new(cfg);
    loop {
        let (stream, peer) = listener.accept().await?;
        let cfg = cfg.clone();
        tokio::spawn(async move {
            let svc = service_fn(move |req: hyper::Request<hyper::body::Incoming>| {
                let cfg = cfg.clone();
                async move {
                    // 본문 상한 = aiohttp client_max_size 1MB(넘으면 413) -- 원본은 읽을 때 재지만 이 서버의 POST 는 모두 본문을 읽는다
                    let (parts, body) = req.into_parts();
                    let limited = http_body_util::Limited::new(body, 1 << 20);
                    let body = match limited.collect().await {
                        Ok(b) => b.to_bytes().to_vec(),
                        Err(_) => {
                            let n = 1 << 20;
                            let r = http_exc(413, Some("Request Entity Too Large"), &[]);
                            let r = Resp { body: Some(format!("Maximum request body size {n} exceeded, actual body size {n}").into_bytes()), ..r };
                            return Ok::<_, Infallible>(to_hyper(prepare_exc(r)));
                        }
                    };
                    let head = hyper::Request::from_parts(parts, ());
                    let r = parse_req(&head, body, peer.ip().to_string());
                    // SSE 둘은 끝나지 않는 응답 -- 블로킹 스레드로 보내지 않는다
                    if r.method == "GET" && r.path == "/api/events" {
                        return Ok(cfg.events.connect(&cfg));
                    }
                    if r.method == "GET" && r.path == "/api/stream" {
                        return Ok(match hub::api_stream(cfg.clone(), cfg.live.clone(), &r) {
                            Ok(resp) => resp,
                            Err(e) => to_hyper(prepare(&r, e)),
                        });
                    }
                    let resp = tokio::task::spawn_blocking(move || handle(&cfg, &r)).await.unwrap_or_else(|_| server_error());
                    Ok::<_, Infallible>(to_hyper(resp))
                }
            });
            let _ = http1::Builder::new().serve_connection(TokioIo::new(stream), svc).await;
        });
    }
}

/// 루프백만 허용(원본도 127.0.0.1 · 공개는 터널 몫)
pub fn check_loopback(addr: &SocketAddr) -> Result<(), String> {
    if addr.ip().is_loopback() { Ok(()) } else { Err(format!("RTSERVE_ADDR={addr}: 루프백 주소만 쓴다(공개 경로는 Cloudflare 터널)")) }
}

