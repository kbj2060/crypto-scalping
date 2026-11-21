//! rtserve -- 대시보드 응답 서버(파이썬 aiohttp, 8787)의 Rust 판(꾸러미 R6 · 첫 조각 2026-11-21).
//!
//! 원본 흐름을 그대로 본뜬다: 라우터 → 핸들러 → 미들웨어(안쪽 json_compress_etag → 바깥 static_asset_headers) → 응답 준비(압축 협상·길이).
//! 핸들러가 던진 HTTP 예외(302·400·403·404·405)는 미들웨어를 지나지 않고 aiohttp 기본 글자(`404: Not Found`)로 나간다 -- 원본과 같다.
//! 인증은 원본에도 없다(127.0.0.1 에만 묶고, 공개 경로는 Cloudflare 터널 + Access 가 막는다) -- 여기서도 루프백 아니면 시작을 거절한다.
//! 시험: tests/http_parity.rs(정답지 = tools/gen_http_fixture.py 가 진짜 aiohttp 객체로 뜬 바이트).

pub mod api;
pub mod py;

use std::convert::Infallible;
use std::fs;
use std::net::SocketAddr;
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
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
/// orders.py:331 MARKET_SYMBOLS 의 열쇠(DASHBOARD_ASSETS 가 이 안에서만 고른다)
pub const MARKET_ASSETS: [&str; 5] = ["eth", "sol", "btc", "xrp", "hype"];
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
}

impl Config {
    pub fn live_dir(&self) -> PathBuf {
        self.root.join("data/live")
    }
    pub fn dashboard_dir(&self) -> PathBuf {
        self.root.join("dashboard/live")
    }
}

/// 원본 server.py:424 -- `DASHBOARD_ASSETS` env(기본 "eth,sol,xrp") 중 아는 코인만 · 비면 ["eth"]
pub fn parse_assets(env: Option<&str>) -> Vec<String> {
    let v: Vec<String> = env
        .unwrap_or("eth,sol,xrp")
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
        500 => "Internal Server Error",
        _ => "",
    }
}

// ───────────────────────── 라우터 ─────────────────────────

/// 원본 등록부(server.py 4932~5025) 중 이 조각이 옮긴 것. 모두 add_get(= GET·HEAD).
fn route(cfg: &Config, req: &Req) -> Resp {
    let p = req.path.as_str();
    let get = req.method == "GET" || req.method == "HEAD";
    let not_allowed = || http_exc(405, None, &[("Allow", "GET,HEAD")]);
    let handler: Option<fn(&Config, &Req) -> Resp> = match p {
        "/" => Some(|_, _| http_exc(302, None, &[("Location", "/dashboard/live/")])),
        "/dashboard/live" | "/dashboard/live/" | "/dashboard/live/index.html" => Some(dashboard_index),
        "/dashboard/live/sw.js" | "/dashboard/live/manifest.webmanifest" => Some(pwa_asset),
        _ => api::handler(p),
    };
    if let Some(h) = handler {
        return if get { h(cfg, req) } else { not_allowed() };
    }
    if let Some(rest) = p.strip_prefix("/dashboard/live/") {
        return if get { static_file(cfg, req, rest) } else { not_allowed() };
    }
    http_exc(404, None, &[])
}

/// 요청 하나 → 연결에 낼 응답(압축·길이까지 정한 것)
pub fn handle(cfg: &Config, req: &Req) -> Resp {
    let mut r = route(cfg, req);
    if !r.exc {
        // 미들웨어 둘(안쪽부터) -- 예외 응답(302·4xx·500)은 원본처럼 미들웨어를 안 지난다
        r = json_compress_etag(req, r);
        r = static_asset_headers(req, r);
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
fn asset_buster(dir: &Path) -> String {
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

pub fn parse_req<B>(r: &hyper::Request<B>) -> Req {
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
        headers: r.headers().iter().map(|(k, v)| (k.as_str().to_string(), String::from_utf8_lossy(v.as_bytes()).into_owned())).collect(),
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
    if let Some(reason) = r.reason.filter(|x| x != phrase(r.status)) {
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
        let (stream, _) = listener.accept().await?;
        let cfg = cfg.clone();
        tokio::spawn(async move {
            let svc = service_fn(move |req: hyper::Request<hyper::body::Incoming>| {
                let cfg = cfg.clone();
                async move {
                    let r = parse_req(&req);
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

