//! 뉴스 RSS 수집기 = scripts/live_news_rss_collector_20261005.py(상주 · 피드 8개 · 조건부 GET · SQLite hot `news_rss.sqlite`).
//!   XML = ElementTree 흉내(quick-xml NsReader): `iter("item")` = 이름공간 없는 item 전부(안에 든 것까지) · `findtext(tag)` = 바로 아래 첫 자식의
//!   `.text`(첫 자식 요소 앞 글자·CDATA 만 · 없으면 "" · 자식이 없으면 None) · 줄 끝 \r\n·\r → \n(expat 과 같다) · 모르는 개체 = 오류(그 폴링 실패).
//!   날짜 = email.utils.parsedate_to_datetime(_parsedate_tz) 손 이식 · 공백 = 파이썬 str.isspace(\x1c~\x1f 포함 -- Rust 공백과 다르다) · html.unescape = crate::html.
//!   ponytail: XML 선언의 인코딩은 UTF-8 만(원본 expat 은 latin-1·utf-16 도 읽는다) · DOCTYPE 안 개체 선언은 안 푼다(원본은 푼다) -- 지금 피드 8개는 해당 없음.
use crate::html::unescape;
use quick_xml::events::Event;
use quick_xml::name::ResolveResult;
use regex::Regex;
use rusqlite::Connection;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::LazyLock;

/// 이름: (url, 주기 초) -- 원본 FEEDS 순서.
pub const FEEDS: [(&str, &str, f64); 8] = [
    ("theblock", "https://www.theblock.co/rss.xml", 60.0),
    ("cointelegraph", "https://cointelegraph.com/rss", 60.0),
    ("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss", 60.0),
    ("tree", "https://news.treeofalpha.com/api/news?limit=50", 30.0),
    ("bwenews", "https://rss-public.bwe-ws.com/", 20.0),
    ("financialjuice", "https://www.financialjuice.com/feed.ashx?xy=rss", 60.0),
    ("fed", "https://www.federalreserve.gov/feeds/press_all.xml", 300.0),
    ("sec", "https://www.sec.gov/news/pressreleases.rss", 300.0),
];
pub const UA: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36";
const POLLS_KEEP_S: f64 = 7.0 * 86400.0;

/// 파이썬 str.isspace
pub fn py_space(c: char) -> bool {
    matches!(c, '\t'..='\r' | '\x1c'..='\x1f' | ' ' | '\u{85}' | '\u{a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}')
}

fn strip(s: &str) -> &str {
    s.trim_matches(py_space)
}

/// 파이썬 re `\s`(str 패턴) = str.isspace 와 같은 글자들
const SP: &str = r"[\t\n\x0B\x0C\r\x1C-\x1F \x{85}\x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}]";
static TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").unwrap());
static WS: LazyLock<Regex> = LazyLock::new(|| Regex::new(&format!("{SP}+")).unwrap());
/// 원본 BWE_DROP(re.IGNORECASE · match = 앞에서)
static BWE_DROP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(r"(?i)^{SP}*(?:[^:：]*(?:listing|上新)|(?:binance|okx|bybit|coinbase|upbit|bithumb|bitget|kucoin|gate|mexc|hyperliquid)\b[^:：]*)[:：]")).unwrap()
});

pub fn bwe_drop(title: &str) -> bool {
    BWE_DROP.is_match(title)
}

// ── RFC 822 날짜(email._parseaddr._parsedate_tz) ──────────────────────────────

const MONTHS: [&str; 24] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "january", "february", "march", "april", "may", "june",
                            "july", "august", "september", "october", "november", "december"];
const DAYS: [&str; 7] = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];

/// 파이썬 int(글자): 앞뒤 공백 · 부호 · 숫자 사이 밑줄 하나. ponytail: 유니코드 숫자(٣ 등)는 못 읽는다(원본은 읽는다).
fn py_int(s: &str) -> Option<i64> {
    let s = strip(s);
    let (neg, d) = match s.as_bytes().first()? {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };
    if d.is_empty() || d.starts_with('_') || d.ends_with('_') || d.contains("__") || !d.bytes().all(|b| b.is_ascii_digit() || b == b'_') {
        return None;
    }
    let v: i64 = d.replace('_', "").parse().ok()?;
    Some(if neg { -v } else { v })
}

fn tzname(s: &str) -> Option<i64> {
    Some(match s {
        "UT" | "UTC" | "GMT" | "Z" => 0,
        "AST" => -400,
        "ADT" => -300,
        "EST" => -500,
        "EDT" => -400,
        "CST" => -600,
        "CDT" => -500,
        "MST" => -700,
        "MDT" => -600,
        "PST" => -800,
        "PDT" => -700,
        _ => return None,
    })
}

/// `_parsedate_tz` → (y, m, d, H, M, S, tz 초 | None)
fn parsedate_tz(data: &str) -> Option<(i64, i64, i64, i64, i64, i64, Option<i64>)> {
    let mut d: Vec<String> = data.split(py_space).filter(|x| !x.is_empty()).map(str::to_string).collect();
    if d.is_empty() {
        return None;
    }
    if d[0].ends_with(',') || DAYS.contains(&d[0].to_lowercase().as_str()) {
        d.remove(0);
    } else if let Some(i) = d[0].rfind(',') {
        d[0] = d[0][i + 1..].to_string();
    }
    if d.len() == 3 {
        let st: Vec<String> = d[0].split('-').map(str::to_string).collect();
        if st.len() == 3 {
            d = st.into_iter().chain(d[1..].iter().cloned()).collect();
        }
    }
    if d.len() == 4 {
        let s = d[3].clone();
        let i = s.find('+').or_else(|| s.find('-'));
        match i {
            Some(i) if i > 0 => {
                d.truncate(3);
                d.push(s[..i].to_string());
                d.push(s[i..].to_string());
            }
            _ => d.push(String::new()),
        }
    }
    if d.len() < 5 {
        return None;
    }
    let (mut dd, mut mm, mut yy, mut tm, mut tz) = (d[0].clone(), d[1].clone(), d[2].clone(), d[3].clone(), d[4].clone());
    if dd.is_empty() || mm.is_empty() || yy.is_empty() {
        return None;
    }
    mm = mm.to_lowercase();
    if !MONTHS.contains(&mm.as_str()) {
        (dd, mm) = (mm, dd.to_lowercase());
        if !MONTHS.contains(&mm.as_str()) {
            return None;
        }
    }
    let mut mo = MONTHS.iter().position(|x| *x == mm).unwrap() as i64 + 1;
    if mo > 12 {
        mo -= 12;
    }
    if dd.ends_with(',') {
        dd.pop();
    }
    if yy.find(':').is_some_and(|i| i > 0) {
        (yy, tm) = (tm, yy);
    }
    if yy.ends_with(',') {
        yy.pop();
        if yy.is_empty() {
            return None;
        }
    }
    if !yy.chars().next().is_some_and(|c| c.is_numeric()) {
        (yy, tz) = (tz, yy);
    }
    if tm.ends_with(',') {
        tm.pop();
    }
    let parts: Vec<&str> = tm.split(':').collect();
    let (thh, tmm, tss) = match parts.len() {
        2 => (parts[0].to_string(), parts[1].to_string(), "0".to_string()),
        3 => (parts[0].to_string(), parts[1].to_string(), parts[2].to_string()),
        1 if tm.contains('.') => {
            let p: Vec<&str> = tm.split('.').collect();
            match p.len() {
                2 => (p[0].to_string(), p[1].to_string(), "0".to_string()),
                3 => (p[0].to_string(), p[1].to_string(), p[2].to_string()),
                _ => return None,
            }
        }
        _ => return None,
    };
    let (mut y, day, h, mi, s) = (py_int(&yy)?, py_int(&dd)?, py_int(&thh)?, py_int(&tmm)?, py_int(&tss)?);
    if y < 100 {
        y += if y > 68 { 1900 } else { 2000 };
    }
    let tzu = tz.to_uppercase();
    let mut off = tzname(&tzu).or_else(|| py_int(&tzu));
    if off == Some(0) && tzu.starts_with('-') {
        off = None;
    }
    let off = off.map(|o| if o == 0 { 0 } else { o.signum() * ((o.abs() / 100) * 3600 + (o.abs() % 100) * 60) });
    Some((y, mo, day, h, mi, s, off))
}

fn days_in(y: i64, m: i64) -> i64 {
    match m {
        2 if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// `pub_ms_of`: RFC 822 → ms · 깨지면 None · 시간대 없으면 UTC.
pub fn pub_ms_of(pub_: Option<&str>) -> Option<i64> {
    let s = pub_.filter(|s| !s.is_empty())?;
    let (y, m, d, h, mi, sec, tz) = parsedate_tz(s)?;
    // datetime(...) 범위 검사(ValueError → None) · timezone(timedelta) 는 ±24시간 미만만
    if !(1..=9999).contains(&y) || !(1..=12).contains(&m) || d < 1 || d > days_in(y, m) || !(0..24).contains(&h) || !(0..60).contains(&mi) || !(0..60).contains(&sec) {
        return None;
    }
    if tz.is_some_and(|t| t.abs() >= 86_400) {
        return None;
    }
    let days = crate::cboe_summary::days_from_civil(y, m, d);
    Some((days * 86_400 + h * 3600 + mi * 60 + sec - tz.unwrap_or(0)) * 1000)
}

// ── ElementTree 흉내 ─────────────────────────────────────────────────────────

struct El {
    /// None = 이름공간 붙은 요소(ET 에선 "{uri}이름" -- "item"·"link" 와 안 맞는다)
    name: Option<String>,
    /// 첫 자식 요소 앞 글자(ET .text) -- None = 없음
    text: Option<String>,
    kids: Vec<El>,
}

fn parse_tree_xml(xml: &[u8]) -> Result<El, String> {
    let mut r = quick_xml::reader::NsReader::from_reader(xml);
    r.config_mut().trim_text(false);
    let mut stack: Vec<El> = vec![];
    let mut root: Option<El> = None;
    let mut buf = vec![];
    let push_text = |stack: &mut Vec<El>, s: &str| {
        if let Some(top) = stack.last_mut() {
            if top.kids.is_empty() {
                top.text.get_or_insert_with(String::new).push_str(s);
            }
        }
    };
    loop {
        let (ns, ev) = r.read_resolved_event_into(&mut buf).map_err(|e| format!("ParseError: {e}"))?;
        match ev {
            Event::Start(ref e) | Event::Empty(ref e) => {
                let name = match ns {
                    ResolveResult::Unbound => Some(String::from_utf8_lossy(e.local_name().as_ref()).to_string()),
                    ResolveResult::Bound(_) => None,
                    ResolveResult::Unknown(p) => return Err(format!("ParseError: unbound prefix {}", String::from_utf8_lossy(&p))),
                };
                let el = El { name, text: None, kids: vec![] };
                let empty = matches!(ev, Event::Empty(_));
                if empty {
                    match stack.last_mut() {
                        Some(top) => top.kids.push(el),
                        None if root.is_none() => root = Some(el),
                        None => return Err("ParseError: junk after document element".into()),
                    }
                } else {
                    if stack.is_empty() && root.is_some() {
                        return Err("ParseError: junk after document element".into());
                    }
                    stack.push(el);
                }
            }
            Event::End(_) => {
                let el = stack.pop().ok_or("ParseError: mismatched tag")?;
                match stack.last_mut() {
                    Some(top) => top.kids.push(el),
                    None => root = Some(el),
                }
            }
            Event::Text(t) => {
                let s = t.xml10_content().map_err(|e| format!("ParseError: {e}"))?;
                if stack.is_empty() {
                    if !s.chars().all(|c| matches!(c, ' ' | '\t' | '\n' | '\r')) {
                        return Err("ParseError: syntax error".into());
                    }
                } else {
                    push_text(&mut stack, &s);
                }
            }
            Event::CData(t) => {
                let s = t.xml10_content().map_err(|e| format!("ParseError: {e}"))?;
                push_text(&mut stack, &s);
            }
            Event::GeneralRef(g) => {
                let ch = match g.resolve_char_ref().map_err(|e| format!("ParseError: {e}"))? {
                    Some(c) => c.to_string(),
                    None => match g.as_ref() {
                        b"lt" => "<".into(),
                        b"gt" => ">".into(),
                        b"amp" => "&".into(),
                        b"apos" => "'".into(),
                        b"quot" => "\"".into(),
                        o => return Err(format!("ParseError: undefined entity &{};", String::from_utf8_lossy(o))),
                    },
                };
                push_text(&mut stack, &ch);
            }
            Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }
    if !stack.is_empty() {
        return Err("ParseError: no element found".into());
    }
    root.ok_or_else(|| "ParseError: no element found".into())
}

fn findtext<'a>(e: &'a El, tag: &str) -> Option<&'a str> {
    e.kids.iter().find(|k| k.name.as_deref() == Some(tag)).map(|k| k.text.as_deref().unwrap_or(""))
}

fn iter_items<'a>(e: &'a El, out: &mut Vec<&'a El>) {
    if e.name.as_deref() == Some("item") {
        out.push(e);
    }
    for k in &e.kids {
        iter_items(k, out);
    }
}

/// (link, title, summary, categories, pub_ms)
pub type Item = (String, String, String, String, Option<i64>);

fn or_empty(a: Option<&str>) -> Option<&str> {
    a.filter(|s| !s.is_empty())
}

/// `parse(xml)`
pub fn parse(xml: &[u8]) -> Result<Vec<Item>, String> {
    let root = parse_tree_xml(xml)?;
    let mut items = vec![];
    iter_items(&root, &mut items);
    let mut out = vec![];
    for it in items {
        let raw = or_empty(findtext(it, "link")).or_else(|| or_empty(findtext(it, "guid"))).unwrap_or("");
        let link = strip(raw).split('?').next().unwrap_or("").trim_end_matches('/').to_string();
        if link.is_empty() {
            continue;
        }
        let pub_ms = pub_ms_of(findtext(it, "pubDate"));
        let desc = or_empty(findtext(it, "description")).unwrap_or("");
        let summary = strip(&WS.replace_all(&unescape(&TAG.replace_all(desc, " ")), " ")).to_string();
        let cats: Vec<&str> = it.kids.iter().filter(|k| k.name.as_deref() == Some("category")).filter_map(|k| k.text.as_deref().filter(|t| !t.is_empty())).map(strip).collect();
        let title = strip(&unescape(or_empty(findtext(it, "title")).unwrap_or(""))).to_string();
        out.push((link, title, summary, cats.join(","), pub_ms));
    }
    Ok(out)
}

fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        o => crate::py_str(Some(o), ""),
    }
}

/// `parse_tree(body)` -- Tree News JSON
pub fn parse_tree(body: &[u8]) -> Result<Vec<Item>, String> {
    let v: Value = serde_json::from_slice(body).map_err(|e| format!("JSONDecodeError: {e}"))?;
    let arr = v.as_array().ok_or("TypeError: not a list")?;
    arr.iter().map(|x| {
        let o = x.as_object().ok_or("AttributeError: get")?;
        let truthy = |k: &str| o.get(k).filter(|v| crate::py_truthy(v));
        let link = match truthy("url") {
            Some(u) => py_str(u),
            None => format!("tree:{}", py_str(o.get("_id").ok_or("KeyError: '_id'")?)),
        };
        let title = strip(&unescape(&truthy("title").map(py_str).unwrap_or_default())).to_string();
        let src = truthy("source").map(py_str).unwrap_or_default();
        let t = match truthy("time") {
            Some(Value::Number(n)) => Some(n.as_i64().unwrap_or_else(|| n.as_f64().unwrap().trunc() as i64)),
            Some(Value::String(s)) => Some(py_int(s).ok_or("ValueError: int")?),
            Some(Value::Bool(b)) => Some(i64::from(*b)),
            Some(_) => return Err("TypeError: int".to_string()),
            None => None,
        };
        Ok((link, title, String::new(), src, t))
    }).collect()
}

pub fn connect(path: &Path) -> rusqlite::Result<Connection> {
    if let Some(d) = path.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let c = Connection::open(path)?;
    c.busy_timeout(std::time::Duration::from_secs(30))?;
    c.query_row("PRAGMA journal_mode=WAL", [], |_| Ok(()))?;
    c.execute_batch("CREATE TABLE IF NOT EXISTS items(source TEXT, link TEXT, title TEXT, summary TEXT, categories TEXT,
        pub_ms INTEGER, first_seen_ms INTEGER, backfill INTEGER, PRIMARY KEY(source, link));
        CREATE TABLE IF NOT EXISTS polls(ts_ms INTEGER, source TEXT, status INTEGER, n_items INTEGER, ms INTEGER, age_s INTEGER);
        CREATE INDEX IF NOT EXISTS items_ts ON items(coalesce(pub_ms, first_seen_ms));
        CREATE INDEX IF NOT EXISTS polls_ts ON polls(ts_ms);")?;
    Ok(c)
}

/// `store` -- 새로 들어간 수(INSERT OR IGNORE · executemany = 한 트랜잭션 아님 · autocommit 줄마다).
pub fn store(c: &Connection, source: &str, items: &[Item], seen_ms: i64, backfill: bool) -> rusqlite::Result<usize> {
    let mut st = c.prepare_cached("INSERT OR IGNORE INTO items VALUES(?,?,?,?,?,?,?,?)")?;
    let mut n = 0;
    for it in items {
        n += st.execute(rusqlite::params![source, it.0, it.1, it.2, it.3, it.4, seen_ms, i64::from(backfill)])?;
    }
    Ok(n)
}

/// 응답 하나: (상태, 본문, 머리글(소문자 이름 → 값)). Err = 접속 실패(상태 0).
pub type Resp = Result<(u16, Vec<u8>, HashMap<String, String>), String>;

/// 상주 바퀴(`run`) -- clock = 벽시계 · sleep(초) 가 false 면 끝 · get(url, 머리글) · log = 원본 print 줄.
pub fn run(db: &Path, feeds: &[(&str, &str, f64)], clock: &mut dyn FnMut() -> f64, sleep: &mut dyn FnMut(f64) -> bool,
           get: &mut dyn FnMut(&str, &[(&str, String)]) -> Resp, tick_s: f64, log: &mut dyn FnMut(String)) -> rusqlite::Result<()> {
    let c = connect(db)?;
    let mut cond: HashMap<&str, Vec<(&str, String)>> = feeds.iter().map(|f| (f.0, vec![])).collect();
    let mut first: HashMap<&str, bool> = HashMap::new();
    for f in feeds {
        let n: i64 = c.query_row("SELECT count(*) FROM items WHERE source=?", [f.0], |r| r.get(0))?;
        first.insert(f.0, n == 0);
    }
    let mut due: HashMap<&str, f64> = feeds.iter().map(|f| (f.0, 0.0)).collect();
    log(format!("수집 {} → {}", feeds.iter().map(|f| format!("{}({}s)", f.0, f.2 as i64)).collect::<Vec<_>>().join(", "), db.display()));
    let mut pruned = 0.0;
    loop {
        if clock() - pruned > 3600.0 {
            if let Err(e) = c.execute("DELETE FROM polls WHERE ts_ms < ?", [((clock() - POLLS_KEEP_S) * 1000.0) as i64]) {
                log(format!("polls 정리 보류 {e}"));
            }
            pruned = clock();
        }
        for &(src, url, every) in feeds {
            let t0 = clock();
            if t0 < due[src] {
                continue;
            }
            due.insert(src, t0 + every);
            let (mut status, mut n, mut age): (i64, i64, Option<i64>) = (0, 0, None);
            let mut headers = vec![("User-Agent", UA.to_string())];
            headers.extend(cond[src].iter().cloned());
            match get(url, &headers) {
                Ok((st, body, h)) if (200..300).contains(&st) => {
                    status = st as i64;
                    age = h.get("age").filter(|a| !a.is_empty() && a.chars().all(|c| c.is_numeric())).and_then(|a| a.parse().ok());
                    let new_cond: Vec<(&str, String)> = [("If-None-Match", h.get("etag")), ("If-Modified-Since", h.get("last-modified"))]
                        .into_iter().filter_map(|(k, v)| v.filter(|v| !v.is_empty()).map(|v| (k, v.clone()))).collect();
                    let r = (|| -> Result<usize, String> {
                        let mut items = if src == "tree" { parse_tree(&body)? } else { parse(&body)? };
                        if src == "bwenews" {
                            items.retain(|it| !bwe_drop(&it.1));
                        }
                        n = items.len() as i64;
                        store(&c, src, &items, (clock() * 1000.0) as i64, first[src]).map_err(|e| format!("OperationalError({e})"))
                    })();
                    match r {
                        Ok(new) => {
                            cond.insert(src, new_cond); // 저장 뒤에만
                            if new > 0 && !first[src] {
                                log(format!("{src} 새 기사 {new}"));
                            }
                            first.insert(src, false);
                        }
                        Err(e) => log(format!("{src} 실패 {e}").chars().take(200).collect()),
                    }
                }
                Ok((st, _, _)) => {
                    status = st as i64; // 304 = 바뀐 것 없음 · 그 밖은 HTTPError
                    if st == 429 {
                        due.insert(src, t0 + 4.0 * every);
                    }
                    if st != 304 {
                        log(format!("{src} HTTP {st}"));
                    }
                }
                Err(e) => log(format!("{src} 실패 {e}").chars().take(200).collect()),
            }
            if let Err(e) = c.execute("INSERT INTO polls VALUES(?,?,?,?,?,?)", rusqlite::params![(t0 * 1000.0) as i64, src, status, n, ((clock() - t0) * 1000.0) as i64, age]) {
                log(format!("{src} polls 기록 실패 {e}"));
            }
        }
        if !sleep(tick_s) {
            return Ok(());
        }
    }
}
