//! Macro Trend 카드(?card=mt · app.js 3366~3620 renderMacroTrend · mtTodayCell · mtDayChart · mtOverlay · mtFeed) + 풋프린트 머리 «Macro» 요약(?card=mth · 3527 mtHeadHtml · 같은 자료).
//! 자료 = 원본과 같은 주소·주기: /api/macro-trend 10분(실패면 1분 뒤) · /api/macro-today 1분 · /api/treasury-futures?h=24 5분(1분 판 안에서) · /api/xyz100 1분(꼬리 since · 넓은 화면 밖 주기) ·
//!   /api/macro-calendar 6시간(실패면 1분 뒤 · 다음 발표·차트 발표선). 본문은 원본처럼 그 응답들이 올 때(와 줄을 누를 때)만 다시 짓는다 -- «실시간/n분 지연» 글은 그 시각 기준.
//! 자리 = 원본 styles.css(html.fit1 · 1920 에서 #mtBody 1838×917)를 헤드리스 크로미움에서 잰 값과 같게 손으로 놓은 것(compare/mt_js.mjs → mt_layout.py → 시험 js_parity).
//! 안 옮긴 것: 칸·줄 툴팁(title) · «?» 설명 팝업(단추 모양만) · 줄 위에 올림·초점 테 · 상세 칸 스크롤(넘치면 잘림) · 1500px 이하 한 단 · 휴대폰 판 · 머리 단추 누르면 카드로 가기.
//! ponytail: 줄바꿈 = UAX14 를 줄인 규칙(빈칸 뒤 · 한글/한자 옆 · 닫는 괄호·문장부호 앞과 여는 괄호 뒤는 안 끊음) -- 서버 문장에 다른 문자가 늘어 어긋나면 규칙을 더한다.
//! ponytail: 머리 h4·꼬리 줄·figcaption 은 한 줄로 둔다(원본은 넘치면 접힘) · 오류 키 차례 = 이름순(serde_json 맵 -- 원본은 받은 차례).
use crate::fp::{Anchor, Clip, Col, Prim};
use crate::srl::{js, truthy};
use serde_json::Value;

/// 글자 폭(글, 크기, 굵기, 고정폭 --font-mono 인가) -- 자간 빼고
pub type Measure<'a> = &'a dyn Fn(&str, f64, u16, bool) -> f64;

// ── 원본 상수 ──
const MT_FAM: [(&str, &[&str]); 6] = [("주식·위험", &["나스닥", "S&P500", "VIX", "STLFSI"]), ("신용", &["HY스프레드", "Baa스프레드"]), ("금리", &["10Y금리", "기대인플레", "기간프리미엄", "연방기금"]),
    ("달러·환율", &["달러지수", "USD/JPY", "EUR/USD", "원/달러", "USD/CNY"]), ("유동성", &["순유동성", "연준BS", "지준", "TGA", "RRP", "스테이블코인"]), ("원자재", &["브렌트"])];
const MT_CELLS: [(&str, &str); 4] = [("co", "BTC"), ("co", "ETH"), ("lead", "BTC"), ("lead", "ETH")];
fn today_key(n: &str) -> Option<&'static str> { match n { "나스닥" => Some("NQ"), "S&P500" => Some("SP"), "VIX" => Some("VIX"), "10Y금리" => Some("ZN"), _ => None } }
fn tname(k: &str) -> &'static str { match k { "NQ" => "나스닥100(XYZ100)", "SP" => "S&P500 무기한(xyz:SP500)", "VIX" => "VIX 지수(^VIX)", _ => "10년 국채선물(ZN) 가격" } }

// ── JS 수·글 ──
/// JS toFixed(d): 정확한 이진값의 십진 전개에서 반올림(동점 = 큰 쪽) · 음수 = "-" + (−x).toFixed
pub fn fixed(x: f64, d: usize) -> String {
    if x.is_nan() { return "NaN".into(); }
    if x < 0.0 { return format!("-{}", fixed(-x, d)); }
    let e = format!("{x:.120}");
    let (ip, fp) = e.split_at(e.find('.').unwrap());
    let fp = &fp[1..];
    let mut digits: Vec<u8> = ip.bytes().chain(fp.bytes().take(d)).map(|b| b - b'0').collect();
    if fp.as_bytes()[d] >= b'5' { // 다음 자리 ≥ 5 = 올림(정확한 전개라 동점도 올림 -- JS 와 같다)
        let mut i = digits.len();
        loop { if i == 0 { digits.insert(0, 1); break; } i -= 1; if digits[i] == 9 { digits[i] = 0; } else { digits[i] += 1; break; } }
    }
    let n = digits.len() - d;
    let s: String = digits.iter().map(|v| (b'0' + v) as char).collect();
    if d == 0 { s } else { format!("{}.{}", &s[..n], &s[n..]) }
}
fn num(v: &Value) -> Option<f64> { v.as_f64().filter(|x| x.is_finite()) }
fn f(v: &Value) -> f64 { v.as_f64().unwrap_or(f64::NAN) }
/// Math.round
fn jround(x: f64) -> f64 { (x + 0.5).floor() }
/// 3392 mtFmt
fn mt_fmt(v: &Value, unit: &str) -> String {
    let Some(v) = num(v) else { return "–".into() };
    let d = match unit { "bp" => 0, "$bn" => if v.abs() < 10.0 { 1 } else { 0 }, "%" => 1, _ => 2 };
    format!("{}{}{}", sign(v), fixed(v.abs(), d), if unit == "$bn" { "bn" } else { unit })
}
fn sign(v: f64) -> &'static str { if v > 0.0 { "+" } else if v < 0.0 { "−" } else { "" } }
/// 3397 mtR
fn mt_r(v: &Value) -> String { match num(v) { None => "–".into(), Some(v) => format!("{}{}", sign(v), fixed(v.abs(), 2)) } }
fn pm(v: f64) -> &'static str { if v >= 0.0 { "+" } else { "−" } }
/// `${x || 0}`
fn or0(v: &Value) -> String { if truthy(v) { js(v) } else { "0".into() } }

// ── 시각 ──
/// 브라우저 지역 시각 hh:mm(tz = getTimezoneOffset 분)
pub(crate) fn hm(t: f64, tz: i64) -> String { let m = ((t * 1000.0).floor() as i64 - tz * 60_000).div_euclid(60_000); format!("{:02}:{:02}", m.div_euclid(60).rem_euclid(24), m.rem_euclid(60)) }
fn days_civil(y: i64, m: i64, d: i64) -> i64 { let y = if m <= 2 { y - 1 } else { y }; let era = y.div_euclid(400); let yoe = y - era * 400; let doy = (153 * ((m + 9) % 12) + 2) / 5 + d - 1; era * 146097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719468 }
fn year_of(days: i64) -> i64 { let mut y = 1970 + days.div_euclid(366); while days_civil(y + 1, 1, 1) <= days { y += 1; } y }
/// 0 = 일요일
fn weekday(days: i64) -> i64 { (days + 4).rem_euclid(7) }
/// 미국 일광 절약 시간(3월 둘째 일요일 02:00 지역 표준 ~ 11월 첫 일요일 02:00 지역 일광) -- std = 표준시 UTC 차(시)
pub(crate) fn us_local(t: f64, std: i64) -> (i64, i64) {
    let s = t.floor() as i64;
    let y = year_of((s + std * 3600).div_euclid(86400));
    let sun = |m: i64, k: i64| { let d1 = days_civil(y, m, 1); d1 + (7 - weekday(d1)) % 7 + 7 * (k - 1) };
    let on = sun(3, 2) * 86400 + 2 * 3600 - std * 3600;
    let off = sun(11, 1) * 86400 + 2 * 3600 - (std + 1) * 3600;
    let l = s + (std + if s >= on && s < off { 1 } else { 0 }) * 3600;
    (weekday(l.div_euclid(86400)), l.rem_euclid(86400) / 60)
}
/// 3480 cmeOpen(시카고 일 17:00 ~ 금 16:00 · 매일 16~17시 정비)
fn cme_open(t: f64) -> bool { let (w, m) = us_local(t, -6); let h = m / 60; match w { 6 => false, 0 => h >= 17, 5 => h < 16, _ => h != 16 } }
/// 3449 vixOpen(뉴욕 평일 03:15~16:15)
fn vix_open(t: f64) -> bool { let (w, m) = us_local(t, -5); w != 0 && w != 6 && (195..975).contains(&m) }
/// Date.parse(ISO) / 1000 -- 영역 없으면 지역 시각(JS 와 같다)
pub(crate) fn parse_iso(s: &str, tz: i64) -> f64 {
    let b = s.as_bytes();
    let n = |a: usize, l: usize| s.get(a..a + l).and_then(|x| x.parse::<i64>().ok());
    let (Some(y), Some(mo), Some(d)) = (n(0, 4), n(5, 2), n(8, 2)) else { return f64::NAN };
    let (hh, mi) = (n(11, 2).unwrap_or(0), n(14, 2).unwrap_or(0));
    let mut i = 16;
    let mut sec = 0.0;
    if b.get(16) == Some(&b':') { let j = (17..b.len()).find(|&k| !(b[k].is_ascii_digit() || b[k] == b'.')).unwrap_or(b.len()); sec = s[17..j].parse().unwrap_or(0.0); i = j; }
    let off = match b.get(i) { Some(b'Z') => 0, Some(&c) if c == b'+' || c == b'-' => { let v = n(i + 1, 2).unwrap_or(0) * 60 + n(i + 4, 2).unwrap_or(0); if c == b'+' { v } else { -v } } _ => -tz };
    (days_civil(y, mo, d) * 86400 + hh * 3600 + mi * 60 - off * 60) as f64 + sec
}

// ── 입력 ──
/// 받아 둔 응답(원본 전역과 같은 뜻). now = Date.now()(ms · 본문은 마지막으로 다시 지은 시각) · tz = getTimezoneOffset
pub struct In<'a> {
    pub now: f64, pub tz: i64,
    pub trend: Option<&'a Value>, pub events: Option<&'a Value>,
    pub xyz: &'a [[f64; 2]], pub xyz_live: Option<&'a Value>, pub skew: f64,
    /// mtIntra = {at(초), p} · mtTreas = {at(ms), p}
    pub intra: Option<(f64, &'a Value)>, pub treas: Option<(f64, &'a Value)>,
    pub sel: Option<&'a str>,
}
impl In<'_> {
    fn ns(&self) -> f64 { self.now / 1000.0 }
    fn clock(&self) -> f64 { self.ns() + self.skew }
    /// 3720 xyzNow
    fn xyz_now(&self) -> Option<f64> { self.xyz_live.filter(|l| truthy(&l["px"]) && self.clock() - f(&l["recv_sec"]) < 30.0).map(|l| f(&l["px"])) }
    fn isym(&self, k: &str) -> Option<&Value> { self.intra.filter(|(_, p)| p["ok"] == true).map(|(_, p)| &p["symbols"][k]).filter(|v| !v.is_null()) }
    /// 3443 mtIntraAge
    fn intra_age(&self, k: &str) -> Option<f64> {
        let v = self.isym(k)?;
        let t = if k == "SP" { f(&v["live"]["recv_sec"]) } else { v["bars"].as_array().and_then(|b| b.last()).map_or(0.0, |b| f(&b[0])) };
        let (at, p) = self.intra?;
        (t != 0.0 && !t.is_nan()).then(|| f(&p["now_sec"]) - t + (self.ns() - at))
    }
    /// 3429 mtDay
    fn day(&self, k: &str) -> Vec<[f64; 2]> {
        let pts = |v: &Value| v.as_array().into_iter().flatten().map(|b| [f(&b[0]), f(&b[1])]).collect::<Vec<_>>();
        if k == "NQ" {
            let t = self.clock();
            let mut v: Vec<_> = self.xyz.iter().filter(|b| b[0] + 60.0 >= t - 86400.0 && b[0] + 60.0 <= t).map(|b| [b[0] + 60.0, b[1]]).collect();
            if let Some(l) = self.xyz_now() { v.push([t, l]); }
            return v;
        }
        if k == "SP" || k == "VIX" {
            let Some(v) = self.isym(k) else { return vec![] };
            let t = self.ns();
            let mut b: Vec<_> = pts(&v["bars"]).into_iter().filter(|b| b[0] >= t - 86400.0).collect();
            let lv = &v["live"];
            if truthy(lv) && self.intra_age(k).is_some_and(|a| a < 90.0) { b.push([t, f(&lv["px"])]); }
            return b;
        }
        let v = self.treas.filter(|(_, p)| p["ok"] == true).map(|(_, p)| &p["symbols"][k]);
        v.filter(|v| truthy(&v["bars"])).map(|v| pts(&v["bars"]).into_iter().filter(|b| b[0] >= self.ns() - 86400.0).collect()).unwrap_or_default()
    }
    /// 주요 발표(중요도 high · FOMC) [(초, 이름)]
    fn majors(&self) -> Vec<(f64, String)> {
        self.events.and_then(|e| e.as_array()).into_iter().flatten().filter(|e| e["importance"] == "high" || e["category"] == "fomc")
            .map(|e| (parse_iso(e["time_utc"].as_str().unwrap_or(""), self.tz), [&e["title_ko"], &e["title"]].iter().find(|v| truthy(v)).and_then(|v| v.as_str()).unwrap_or("지표").to_string())).collect()
    }
}
fn chg(b: &[[f64; 2]]) -> Option<f64> { (b.len() > 1).then(|| (b[b.len() - 1][1] / b[0][1] - 1.0) * 100.0) }
fn day_col(v: Option<f64>) -> Col { match v { Some(v) if v.abs() >= 0.02 => if v > 0.0 { Col::Good } else { Col::Bad }, _ => Col::Muted } }
fn day_col_k(k: &str, v: Option<f64>) -> Col { day_col(if k == "VIX" { v.map(|x| -x) } else { v }) }

/// 3485 mtFeed → (불 0 ok · 1 bad · 2 wait, 글, 툴팁)
fn feed(i: &In, r: &Value, n: &str) -> (u8, String, String) {
    let fd = &r["feed"];
    let s = |v: &Value| v.as_str().filter(|s| !s.is_empty()).map(str::to_string);
    let mut tips = vec![format!("{} · 마지막 값 {}{}", s(&fd["cadence"]).unwrap_or("공표 주기 모름".into()), s(&fd["last"]).unwrap_or("–".into()),
        if fd["age_d"].is_null() { String::new() } else { format!(" ({}일 전 · 한도 {}일)", js(&fd["age_d"]), js(&fd["max_d"])) })];
    let mut st = if fd["ok"] == false { 1 } else if truthy(&fd["ok"]) { 0 } else { 2 };
    let mut lab = s(&fd["short"]).unwrap_or("공표".into());
    if let Some(w) = s(&fd["why"]) { tips.push(format!("늦음: {w}")); }
    let wait = |st: &mut u8| if *st == 0 { *st = 2 };
    match today_key(n) {
        Some("NQ") => {
            let fresh = i.xyz_live.is_some_and(|l| i.clock() - f(&l["recv_sec"]) < 90.0);
            if i.xyz.is_empty() { lab = "받는 중".into(); wait(&mut st); } else { lab = if fresh { "실시간" } else { "끊김" }.into(); if !fresh { st = 1; } }
            tips.push(format!("장중: 나스닥100 실시간(XYZ100){}", if !i.xyz.is_empty() && !fresh { " -- 90초 넘게 새 값 없음" } else { "" }));
        }
        Some("ZN") => {
            let b = i.day("ZN");
            let age = b.last().map(|x| i.ns() - x[0]);
            let open = cme_open(i.ns());
            if !open { lab = "휴장".into(); } else if let Some(a) = age { lab = if a < 1800.0 { format!("{}분 지연", jround(a / 60.0).max(1.0)) } else { "늦음".into() }; if a >= 1800.0 { st = 1; } } else { lab = "받는 중".into(); wait(&mut st); }
            tips.push(format!("장중: 10년 국채선물(ZN) CME 무료 시세 -- {}", if open { age.map_or("받는 중".into(), |a| format!("마지막 봉 {}분 전(30분 넘으면 빨간불)", jround(a / 60.0))) } else { "CME 휴장(평일 16~17시 CT 정비 · 주말)".into() }));
        }
        Some("SP") => {
            let age = i.intra_age("SP");
            let fresh = age.is_some_and(|a| a < 90.0);
            if age.is_none() { lab = "받는 중".into(); wait(&mut st); } else { lab = if fresh { "실시간" } else { "끊김" }.into(); if !fresh { st = 1; } }
            tips.push(format!("장중: S&P500 무기한 실시간(xyz:SP500){} · 주말·미국 휴장 중 값은 코인 쪽 투기", if age.is_some() && !fresh { " -- 90초 넘게 새 값 없음" } else { "" }));
        }
        Some(_) => {
            let age = i.intra_age("VIX");
            let open = vix_open(i.ns());
            if !open { lab = "휴장".into(); } else if let Some(a) = age { lab = if a < 3600.0 { format!("{}분 지연", jround(a / 60.0).max(1.0)) } else { "늦음".into() }; if a >= 3600.0 { st = 1; } } else { lab = "받는 중".into(); wait(&mut st); }
            tips.push(format!("장중: VIX 지수(^VIX) 무료 시세 5분봉 -- {}", if open { age.map_or("받는 중".into(), |a| format!("마지막 봉 {}분 전(원천 약 15분 지연 + 5분 수집 · 60분 넘으면 빨간불)", jround(a / 60.0))) } else { "산출 안 하는 시간(미 동부 평일 03:15~16:15 · 주말)".into() }));
        }
        None => {}
    }
    (st, lab, format!("{}\n{}", tips.join("\n"), s(&r["realtime_note"]).unwrap_or_default()))
}

// ── 그리기 단위(시험이 원본 잰 값과 하나씩 맞댄다) ──
/// y = 글 상자 위(크로미움 글 조각 상자 = 바닥선 − 반올림 ascent) · k = 상세 칸 자르기 안
#[derive(Debug, Clone)]
pub enum It {
    T { x: f64, y: f64, s: String, size: f64, wt: u16, mono: bool, ls: f64, c: Col, a: f64, k: bool },
    B { x: f64, y: f64, w: f64, h: f64, c: Col, a: f64, rad: f64, k: bool },
    /// 안쪽 고리(테두리·inset box-shadow): 바깥 상자 + 두께
    O { x: f64, y: f64, w: f64, h: f64, lw: f64, c: Col, a: f64, rad: f64, k: bool },
    /// svg: 상자 · viewBox 크기 · meet(아니면 none = 늘임) · 불투명도 · 요소(viewBox 단위)
    S { x: f64, y: f64, w: f64, h: f64, vb: [f64; 2], meet: bool, a: f64, k: bool, els: Vec<El> },
}
#[derive(Debug, Clone, PartialEq)]
pub enum El {
    Line { p: [f64; 4], c: Col, a: f64, sw: f64, dash: Option<[f64; 2]>, ns: bool },
    Rect { r: [f64; 4], c: Col, a: f64 },
    Path { segs: Vec<Vec<[f64; 2]>>, c: Col, a: f64, sw: f64, dash: Option<[f64; 2]>, ns: bool },
}

/// 글꼴 세로: Pretendard 1950/−494(2048) · JetBrains Mono 1020/−300(1000) -- 크로미움은 둘 다 반올림해 더한다(줄 높이 normal = 합)
pub(crate) fn asc(z: f64, mono: bool) -> f64 { (z * if mono { 1.02 } else { 1950.0 / 2048.0 }).round() }
pub(crate) fn cont(z: f64, mono: bool) -> f64 { asc(z, mono) + (z * if mono { 0.3 } else { 494.0 / 2048.0 }).round() }
/// 줄 상자 위 → 글 상자 위(반 행간은 내림)
fn gt(line_top: f64, lh: f64, z: f64, mono: bool) -> f64 { line_top + ((lh - cont(z, mono)) / 2.0).floor() }

struct Cx<'a> { m: Measure<'a>, v: Vec<It>, k: bool }
#[derive(Clone, Copy)]
struct St { z: f64, wt: u16, mono: bool, ls: f64, c: Col }
const fn st(z: f64, wt: u16, mono: bool, ls: f64, c: Col) -> St { St { z, wt, mono, ls, c } }
impl Cx<'_> {
    fn w(&self, s: &str, t: St) -> f64 { (self.m)(s, t.z, t.wt, t.mono) + t.ls * s.chars().count() as f64 }
    /// 글 상자 위 y 그대로
    fn t(&mut self, x: f64, y: f64, s: impl Into<String>, t: St, a: f64) { self.v.push(It::T { x, y, s: s.into(), size: t.z, wt: t.wt, mono: t.mono, ls: t.ls, c: t.c, a, k: self.k }); }
    fn b(&mut self, x: f64, y: f64, w: f64, h: f64, c: Col, a: f64, rad: f64) { self.v.push(It::B { x, y, w, h, c, a, rad, k: self.k }); }
    fn o(&mut self, x: f64, y: f64, w: f64, h: f64, lw: f64, c: Col, a: f64, rad: f64) { self.v.push(It::O { x, y, w, h, lw, c, a, rad, k: self.k }); }
    fn s(&mut self, x: f64, y: f64, w: f64, h: f64, vb: [f64; 2], meet: bool, a: f64, els: Vec<El>) { self.v.push(It::S { x, y, w, h, vb, meet, a, k: self.k, els }); }
}

/// 3404 MTI 셰브런(viewBox 14 · 굵기 2 · 글자색)
fn ico(d: i8, c: Col) -> Vec<El> {
    let p = match d { 1 => vec![[3.0, 9.5], [7.0, 5.0], [11.0, 9.5]], -1 => vec![[3.0, 4.5], [7.0, 9.0], [11.0, 4.5]], _ => return vec![El::Line { p: [3.5, 7.0, 10.5, 7.0], c, a: 1.0, sw: 2.0, dash: None, ns: false }] };
    vec![El::Path { segs: vec![p], c, a: 1.0, sw: 2.0, dash: None, ns: false }]
}
fn dcol(d: i8) -> Col { match d { 1 => Col::Good, -1 => Col::Bad, _ => Col::Muted } }

/// 칸(.mt-tile): u·d·z + weak·off·obs · big = 표·상세(40×22 · 아이콘 svg 가 칸을 채움) · 아니면 범례(26×16 · 아이콘 11)
fn tile(cx: &mut Cx, x: f64, y: f64, dir: i8, mode: &str, big: bool) {
    let (w, h) = if big { (40.0, 22.0) } else { (26.0, 16.0) };
    let op = match mode { "weak" => 0.75, "obs" => 0.5, _ => 1.0 };
    match mode {
        "" => { let (c, a) = match dir { 1 => (Col::Good, 0.16), -1 => (Col::Bad, 0.16), _ => (Col::White, 0.06) }; cx.b(x, y, w, h, c, a, 6.0); }
        "off" => cx.o(x, y, w, h, 1.5, Col::Amber, 0.65, 6.0),
        "obs" => cx.o(x, y, w, h, 1.0, Col::SoftLine, op, 6.0),
        _ => cx.o(x, y, w, h, 1.0, Col::Line, op, 6.0),
    }
    if big { cx.s(x, y, w, h, [14.0, 14.0], true, op, ico(dir, dcol(dir))); } else { cx.s(x + 7.5, y + 2.5, 11.0, 11.0, [14.0, 14.0], true, op, ico(dir, dcol(dir))); }
}
/// 3576 tile(r, a) -- 없으면 «·»
fn tile_of(cx: &mut Cx, x: f64, y: f64, r: Option<&Value>, a: &str) {
    let c = r.map(|r| &r["live"]["assets"][a]).filter(|c| c.is_object());
    let (Some(r), Some(c)) = (r, c) else {
        let t = st(10.0, 700, true, 0.0, Col::Line);
        let w = cx.w("·", t);
        cx.t(x + (40.0 - w) / 2.0, gt(y + 6.0, 10.0, 10.0, true), "·", t, 1.0);
        return;
    };
    let _ = r;
    let b = f(&c["bias"]);
    let dir = if b > 0.0 { 1 } else if b < 0.0 { -1 } else { 0 };
    let mode = match c["state"].as_str().unwrap_or("") { "obs" => "obs", "off" => "off", "weak" | "stale" | "old" | "unknown" => "weak", _ => "" };
    tile(cx, x, y, dir, mode, true);
}

/// «?» 단추(.mt-q): 글 + 5 + 지름 14 원(테 --line) · 위 = 단추 위(안쪽 여백 2) · lh = 물려받은 줄 높이 · 반환 = 폭
fn qbtn(cx: &mut Cx, x: f64, y: f64, s: &str, t: St, lh: f64, qlh: f64) -> f64 {
    let tw = cx.w(s, t);
    cx.t(x, gt(y + 2.0 + (14.0 - lh) / 2.0, lh, t.z, false), s, t, 1.0);
    let cxx = x + tw + 5.0;
    cx.o(cxx, y + 2.0, 14.0, 14.0, 1.0, Col::Line, 1.0, 7.0);
    let q = st(9.5, t.wt, false, 0.0, Col::Muted);
    let qw = cx.w("?", q);
    cx.t(cxx + 1.0 + (12.0 - qw) / 2.0, gt(y + 3.0 + (12.0 - qlh) / 2.0, qlh, 9.5, false), "?", q, 1.0);
    tw + 5.0 + 14.0
}

// ── 줄바꿈 ──
fn cjk(c: char) -> bool { matches!(c as u32, 0x1100..=0x11FF | 0x3130..=0x318F | 0xAC00..=0xD7A3 | 0x4E00..=0x9FFF | 0x3040..=0x30FF) }
/// b 앞에서 끊어도 되나(a = 앞 글자)
pub(crate) fn brk(a: char, b: char) -> bool {
    if b == ' ' { return false; }
    if a == ' ' { return true; }
    if ")]}»,.:;!?%·…".contains(b) || "([{«".contains(a) { return false; }
    cjk(a) || cjk(b) || a == '—' || b == '—'
}
/// 글 덩어리들(글, 꼴) → 줄들 [(x, 글, 꼴)] -- 크로미움처럼 줄 끝 빈칸은 걸어 둔다(폭에 안 셈) · "\n" = <br>
fn wrap(cx: &Cx, runs: &[(String, St)], width: f64) -> Vec<Vec<(f64, String, St)>> {
    let mut lines = vec![];
    let ch: Vec<(char, usize)> = runs.iter().enumerate().flat_map(|(k, (s, _))| s.chars().map(move |c| (c, k))).collect();
    let seg_w = |a: usize, b: usize| -> f64 { // ch[a..b] 폭(덩어리마다 잰다)
        let mut w = 0.0;
        let mut i = a;
        while i < b { let k = ch[i].1; let j = (i..b).find(|&q| ch[q].1 != k).unwrap_or(b); let s: String = ch[i..j].iter().map(|q| q.0).collect(); w += cx.w(&s, runs[k].1); i = j; }
        w
    };
    let trim = |a: usize, mut b: usize| { while b > a && ch[b - 1].0 == ' ' { b -= 1; } b };
    let mut start = 0;
    while start < ch.len() {
        let hard = (start..ch.len()).find(|&q| ch[q].0 == '\n').unwrap_or(ch.len());
        let mut end = hard;
        if seg_w(start, trim(start, hard)) > width {
            let mut best = None;
            for q in start + 1..hard { if brk(ch[q - 1].0, ch[q].0) { if seg_w(start, trim(start, q)) <= width { best = Some(q); } else { break; } } }
            end = best.unwrap_or_else(|| (start + 1..hard).find(|&q| brk(ch[q - 1].0, ch[q].0)).unwrap_or(hard));
        }
        let e = trim(start, end);
        let mut line = vec![];
        let (mut i, mut x) = (start, 0.0);
        while i < e { let k = ch[i].1; let j = (i..e).find(|&q| ch[q].1 != k).unwrap_or(e); let s: String = ch[i..j].iter().map(|q| q.0).collect(); let w = cx.w(&s, runs[k].1); line.push((x, s, runs[k].1)); x += w; i = j; }
        lines.push(line);
        start = if end < ch.len() && ch[end].0 == '\n' { end + 1 } else { end };
        while start < ch.len() && ch[start].0 == ' ' && end != hard { start += 1; }
    }
    lines
}
/// 문단: 줄마다 줄 높이 lh · 반환 = 높이
fn para(cx: &mut Cx, x: f64, y: f64, w: f64, runs: &[(String, St)], lh: f64) -> f64 {
    let ls = wrap(cx, runs, w);
    for (n, l) in ls.iter().enumerate() { for (dx, s, t) in l { cx.t(x + dx, gt(y + n as f64 * lh, lh, t.z, t.mono), s.clone(), *t, 1.0); } }
    ls.len() as f64 * lh
}
/// 말줄임(text-overflow: ellipsis) -- 넘치면 들어가는 만큼 + «…»
fn ellipsis(cx: &Cx, s: &str, t: St, max: f64) -> String {
    if cx.w(s, t) <= max { return s.into(); }
    let cs: Vec<char> = s.chars().collect();
    let e = cx.w("…", t);
    let n = (0..cs.len()).rev().find(|&n| cx.w(&cs[..n].iter().collect::<String>(), t) + e <= max).unwrap_or(0);
    format!("{}…", cs[..n].iter().collect::<String>())
}

// ── 차트 ──
fn fx1(x: f64) -> f64 { fixed(x, 1).parse().unwrap_or(f64::NAN) }
/// 3457 mtDayChart(k, W, H, fut) 의 svg 요소 + 발표 이름표 [(글, 비율 pp)] -- 점 2개 미만이면 None
fn day_chart(i: &In, k: &str, w: f64, h: f64, fut: f64) -> Option<(Vec<El>, Vec<(String, f64)>, [f64; 2])> {
    let b = i.day(k);
    if b.len() < 2 { return None; }
    let now = if k == "NQ" { i.clock() } else { i.ns() };
    let (t0, t1) = (now - 86400.0, now + fut * 3600.0);
    let lo = b.iter().fold(f64::INFINITY, |a, x| a.min(x[1]));
    let hi = b.iter().fold(f64::NEG_INFINITY, |a, x| a.max(x[1]));
    let rg = if hi - lo != 0.0 { hi - lo } else { 1e-9 };
    let xx = |t: f64| (t - t0) / (t1 - t0) * w;
    let yy = |v: f64| 3.0 + (hi - v) / rg * (h - 6.0);
    let col = day_col_k(k, chg(&b));
    let now_x = fx1(xx(now));
    let last = b[b.len() - 1];
    let mut ev: Vec<(f64, String)> = if fut > 0.0 { i.majors().into_iter().filter(|e| e.0 > now && e.0 <= t1).collect() } else { vec![] };
    ev.sort_by(|p, q| p.0.total_cmp(&q.0));
    let mut labs: Vec<(f64, String)> = vec![];
    for e in &ev { match labs.last_mut() { Some(g) if e.0 - g.0 < 7200.0 => { if !g.1.contains(&e.1) { g.1 += &format!(" · {}", e.1); } } _ => labs.push(e.clone()) } }
    let lift = |a| El::Rect { r: [0.0; 4], c: Col::White, a };
    let mut v = vec![];
    if fut > 0.0 { if let El::Rect { c, a, .. } = lift(0.04) { v.push(El::Rect { r: [now_x, 0.0, fx1(w - now_x), h], c, a }); } }
    let y0 = fx1(yy(b[0][1]));
    v.push(El::Line { p: [0.0, y0, if fut > 0.0 { now_x } else { w }, y0], c: Col::Line, a: 1.0, sw: 1.0, dash: Some([2.0, 3.0]), ns: true });
    for e in &ev { let x = fx1(xx(e.0)); v.push(El::Line { p: [x, 0.0, x, h], c: Col::Amber, a: 0.75, sw: 1.0, dash: Some([2.0, 3.0]), ns: true }); }
    if fut > 0.0 && k != "NQ" && k != "SP" { v.push(El::Rect { r: [fx1(xx(last[0])), 0.0, fx1((xx(now) - xx(last[0])).max(0.0)), h], c: Col::White, a: 0.08 }); }
    v.push(El::Path { segs: vec![b.iter().map(|x| [fx1(xx(x[0])), fx1(yy(x[1]))]).collect()], c: col, a: 1.0, sw: 1.6, dash: None, ns: true });
    let labs = labs.into_iter().map(|g| (format!("{} {}", g.1, hm(g.0, i.tz)), (g.0 - t0) / (t1 - t0))).collect();
    Some((v, labs, [t0, now]))
}
/// 3542 mtOverlay(r, a) -- 겹친 선 svg 요소(viewBox 520×120) · 모자라면 None
fn overlay(r: &Value, a: &str) -> Option<Vec<El>> {
    let l = &r["live"];
    let arr = |v: &Value| v.as_array().map(|a| a.iter().map(num).collect::<Vec<_>>()).unwrap_or_default();
    let (spark, c) = (arr(&l["spark"]), arr(&l["coin13"][a]));
    if spark.iter().flatten().count() < 20 || c.iter().flatten().count() < 20 { return None; }
    let z = |xs: &[Option<f64>]| {
        let v: Vec<f64> = xs.iter().flatten().copied().collect();
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let s = (v.iter().map(|q| (q - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        let s = if s != 0.0 && !s.is_nan() { s } else { 1.0 };
        xs.iter().map(|x| x.map(|x| (x - m) / s)).collect::<Vec<_>>()
    };
    let (w, h) = (520.0, 120.0);
    let lead = r["kind"] == "lead";
    let (off, x1) = if lead { (14.0, 14.0) } else { (0.0, 0.0) };
    let pos = f(&r["assets"][a]["r_p3"]) >= 0.0;
    let ind: Vec<Option<f64>> = z(&spark).into_iter().map(|x| x.map(|x| if pos { x } else { -x })).collect();
    let cz = z(&c);
    let (n, m) = (ind.len() as f64, cz.len() as f64);
    let xx = |i: f64| ((i + 51.0) / (51.0 + x1)) * w;
    let yy = |v: f64| h / 2.0 - v.clamp(-2.6, 2.6) / 2.6 * (h / 2.0 - 5.0);
    let d = |xs: &[Option<f64>], from: f64, o: f64| { // 빈 주에서 끊는다
        let mut segs: Vec<Vec<[f64; 2]>> = vec![];
        for (k, v) in xs.iter().enumerate() {
            let Some(v) = v else { continue };
            let p = [fx1(xx(k as f64 + from + o)), fx1(yy(*v))];
            if k > 0 && xs[k - 1].is_some() { segs.last_mut().unwrap().push(p); } else { segs.push(vec![p]); }
        }
        segs
    };
    let now_x = fx1(xx(0.0));
    let mut v = vec![];
    if lead { v.push(El::Rect { r: [now_x, 0.0, fx1(w - now_x), h], c: Col::White, a: 0.06 }); }
    v.push(El::Line { p: [0.0, h / 2.0, w, h / 2.0], c: Col::SoftLine, a: 1.0, sw: 1.0, dash: None, ns: true });
    v.push(El::Line { p: [now_x, 0.0, now_x, h], c: Col::Line, a: 1.0, sw: 1.0, dash: Some([2.0, 3.0]), ns: true });
    v.push(El::Path { segs: d(&cz, 1.0 - m, 0.0), c: Col::Muted, a: 0.6, sw: 1.3, dash: None, ns: true });
    let nn = ind.len();
    v.push(El::Path { segs: d(if lead { &ind[..nn - 14] } else { &ind }, 1.0 - n, off), c: Col::Text, a: 1.0, sw: 2.0, dash: None, ns: true });
    if lead { v.push(El::Path { segs: d(&ind[nn - 15..], n - 15.0 + 1.0 - n, off), c: Col::Text, a: 1.0, sw: 2.0, dash: Some([4.0, 3.0]), ns: true }); }
    Some(v)
}

// ── 본문 ──
/// 한 장 결과: 그림 · 고른 지표(mtSel) · 줄 자리 [(위, 아래, 이름)](누르기) · 왼쪽 표 폭 · 상세 칸 상자
pub struct Out { pub items: Vec<It>, pub sel: Option<String>, pub rows: Vec<(f64, f64, String)>, pub left: f64, pub det: [f64; 4] }

const MUTED10: St = st(10.0, 700, false, 0.8, Col::Muted);

/// 3559 renderMacroTrend → #mtBody(폭 w · 높이 h) 안 그림
pub fn body(i: &In, w: f64, h: f64, m: Measure) -> Out {
    let mut cx = Cx { m, v: vec![], k: false };
    let p = i.trend;
    let mut out = Out { items: vec![], sel: i.sel.map(str::to_string), rows: vec![], left: w, det: [0.0; 4] };
    let Some(p) = p.filter(|p| p["ok"] == true) else {
        let s = match p { None => "불러오는 중…".into(), Some(p) => format!("매크로 데이터를 불러오지 못했다{}. 1분 뒤 다시 시도한다.", p["error"].as_str().filter(|e| !e.is_empty()).map(|e| format!(" — {e}")).unwrap_or_default()) };
        let t = st(12.5, 400, false, 0.0, Col::Muted);
        para(&mut cx, 0.0, 18.0, w, &[(s, t)], 15.0);
        out.items = cx.v;
        return out;
    };
    let rows: Vec<&Value> = p["rows"].as_array().map(|a| a.iter().collect()).unwrap_or_default();
    let by = |n: &str, k: &str| rows.iter().copied().find(|r| r["name"] == n && r["kind"] == k);
    let mut names: Vec<&str> = vec![];
    for r in &rows { let n = r["name"].as_str().unwrap_or(""); if !names.contains(&n) { names.push(n); } }
    let known: Vec<&str> = MT_FAM.iter().flat_map(|q| q.1.iter().copied()).collect();
    let mut fams: Vec<(&str, Vec<&str>)> = MT_FAM.iter().map(|(f, ns)| (*f, ns.iter().copied().filter(|n| names.contains(n)).collect())).collect();
    fams.push(("기타", names.iter().copied().filter(|n| !known.contains(n)).collect()));
    fams.retain(|q| !q.1.is_empty());
    let prim = |n: &str| by(n, "lead").or_else(|| by(n, "co")).unwrap();
    let state = |r: Option<&Value>, a: &str| r.map_or(Value::Null, |r| r["live"]["assets"][a]["state"].clone());
    let sel = match out.sel.as_deref().filter(|s| names.contains(s)) {
        Some(s) => s.to_string(),
        None => { // 처음엔 «셈» 칸 중 ETF 이후 위치가 가장 먼 지표
            let cnt: Vec<&str> = names.iter().copied().filter(|n| MT_CELLS.iter().any(|(k, a)| state(by(n, k), a) == "count")).collect();
            let mut c = if cnt.is_empty() { names.clone() } else { cnt };
            let az = |n: &str| num(&prim(n)["live"]["z"]).unwrap_or(0.0).abs();
            c.sort_by(|x, y| az(y).total_cmp(&az(x)));
            c.first().copied().unwrap_or_default().to_string()
        }
    };
    // 자리: 두 단(1.25fr · 1fr · 사이 28) · 꼬리 줄 13(위 14) · 위 여백 10
    let lw = (w - 28.0) * 1.25 / 2.25;
    let wrap_h = h - 10.0 - 14.0 - 13.0;
    // ── 왼쪽: 범례(아래) 먼저 -- 높이가 표 높이를 정한다 ──
    let key_t = st(11.0, 400, false, 0.0, Col::Muted);
    let txt = "테두리만 = 안 셈 · 칸에 올리거나 줄을 누르면 이유 · ";
    let qw = |cx: &Cx, s: &str| cx.w(s, key_t) + 5.0 + 14.0;
    let items: Vec<(f64, f64)> = {
        let tw = |s: &str| cx.w(s, key_t);
        vec![(26.0 + 4.0 + 26.0 + 4.0 + tw("셈"), 16.0), (30.0 + tw("중립 ±0.5σ"), 16.0), (30.0 + tw("약화·변동 없음·지연"), 16.0), (30.0 + tw("이탈"), 16.0), (30.0 + tw("관찰"), 16.0),
             (tw(txt) + 4.0 + qw(&cx, "세는 규칙") + 4.0 + qw(&cx, "감시"), 18.0)]
    };
    let mut lines: Vec<Vec<usize>> = vec![vec![]];
    let mut used = 0.0;
    for (k, it) in items.iter().enumerate() {
        let l = lines.last_mut().unwrap();
        if !l.is_empty() && used + 14.0 + it.0 > lw { lines.push(vec![k]); used = it.0; } else { if !l.is_empty() { used += 14.0; } used += it.0; l.push(k); }
    }
    let lhs: Vec<f64> = lines.iter().map(|l| l.iter().map(|&k| items[k].1).fold(0.0, f64::max)).collect();
    let key_h = lhs.iter().sum::<f64>() + 6.0 * (lines.len() as f64 - 1.0);
    let mx_h = wrap_h - 12.0 - key_h;
    let mut ky = 10.0 + mx_h + 12.0;
    for (l, lh) in lines.iter().zip(&lhs) {
        let mut x = 0.0;
        for &k in l {
            let top = ky + (lh - items[k].1) / 2.0;
            let lab = ["셈", "중립 ±0.5σ", "약화·변동 없음·지연", "이탈", "관찰"];
            match k {
                0 => { tile(&mut cx, x, top, 1, "", false); tile(&mut cx, x + 30.0, top, -1, "", false); cx.t(x + 60.0, gt(top + 1.5, 13.0, 11.0, false), lab[0], key_t, 1.0); }
                1..=4 => { let (d, md) = [(0, ""), (1, "weak"), (-1, "off"), (1, "obs")][k - 1]; tile(&mut cx, x, top, d, md, false); cx.t(x + 30.0, top + 1.5, lab[k], key_t, 1.0); }
                _ => {
                    cx.t(x, top + 2.5, txt, key_t, 1.0);
                    let mut bx = x + cx.w(txt, key_t) + 4.0;
                    bx += qbtn(&mut cx, bx, top, "세는 규칙", key_t, 13.0, 11.0) + 4.0;
                    qbtn(&mut cx, bx, top, "감시", key_t, 13.0, 11.0);
                }
            }
            x += items[k].0 + 14.0;
        }
        ky += lh + 6.0;
    }
    // ── 표(.mt-mx 격자) ──
    let c0 = (lw - 652.0).max(104.0);
    let cw = [c0, 66.0, 118.0, 84.0, 52.0, 52.0, 52.0, 52.0, 96.0];
    let mut cxs = [0.0; 9];
    for k in 1..9 { cxs[k] = cxs[k - 1] + cw[k - 1] + 10.0; }
    let nrows: usize = fams.iter().map(|q| q.1.len()).sum();
    let auto = 26.0 + 16.5 + 23.0 + 24.5 * fams.len() as f64;
    let fr = ((mx_h - auto) / nrows as f64).max(22.0);
    // 머리 1(26): 지표 · 13주 변화 · 오늘 24h · 위치(σ) · [지금 국면 ?] · [13주 뒤 편향 ?] · 수신
    let y1 = 10.0;
    let hy = gt(y1 + (26.0 - 16.5) / 2.0, 12.5, 10.0, false);
    let center = |cx: &Cx, s: &str, x: f64, wd: f64| x + (wd - cx.w(s, MUTED10)) / 2.0;
    cx.t(0.0, hy, "지표", MUTED10, 1.0);
    let x = cxs[1] + 66.0 - cx.w("13주 변화", MUTED10);
    cx.t(x, hy, "13주 변화", MUTED10, 1.0);
    let x = center(&cx, "오늘 24h", cxs[2], 118.0);
    cx.t(x, hy, "오늘 24h", MUTED10, 1.0);
    let x = center(&cx, "위치(σ)", cxs[3], 84.0);
    cx.t(x, hy, "위치(σ)", MUTED10, 1.0);
    for (g, s) in [(4, "지금 국면"), (6, "13주 뒤 편향")] {
        let gw = 114.0;
        let bw = cx.w(s, MUTED10) + 19.0;
        qbtn(&mut cx, cxs[g] + (gw - bw) / 2.0, y1, s, MUTED10, 12.5, 11.875);
        cx.b(cxs[g], y1 + 23.0, gw, 1.0, Col::SoftLine, 1.0, 0.0);
    }
    cx.t(cxs[8], hy, "수신", MUTED10, 1.0);
    // 머리 2(16.5): 코인 이름
    let y2 = y1 + 26.0;
    for (k, a) in [(4, "BTC"), (5, "ETH"), (6, "BTC"), (7, "ETH")] { let x = center(&cx, a, cxs[k], 52.0); cx.t(x, y2, a, MUTED10, 1.0); }
    // 머리 3(23): 합계 -- 아래 테 --line
    let y3 = y2 + 16.5;
    for k in 0..9 { cx.b(cxs[k], y3 + 22.0, cw[k], 1.0, Col::Line, 1.0, 0.0); }
    let s = &p["summary"];
    let tt = st(11.0, 700, true, 0.0, Col::Good);
    let sp = (cx.m)(" ", 11.0, 600, true);
    for (k, (kind, a)) in MT_CELLS.iter().enumerate() {
        let x = &s[*a];
        let (u, d) = (or0(&x[format!("{kind}_up")]), or0(&x[format!("{kind}_dn")]));
        let (wu, wd) = (cx.w(&u, tt), cx.w(&d, tt));
        let mut x0 = cxs[4 + k] + (52.0 - (12.0 + wu + sp + 12.0 + wd)) / 2.0;
        let base = y3 + 2.0 + 10.0;
        for (dir, n, wn) in [(1, u, wu), (-1, d, wd)] {
            cx.s(x0, base + 1.0 - 11.0, 11.0, 11.0, [14.0, 14.0], true, 1.0, ico(dir, dcol(dir)));
            cx.t(x0 + 12.0, base - 11.0, n, St { c: dcol(dir), ..tt }, 1.0);
            x0 += 12.0 + wn + sp;
        }
    }
    // 계열 줄(24.5) · 지표 줄(fr)
    let mut y = y3 + 23.0;
    let fam_t = st(10.5, 700, false, 1.05, Col::Muted);
    for (fname, ns) in &fams {
        cx.t(0.0, gt(y + 10.0, 10.5, 10.5, false), *fname, fam_t, 1.0);
        y += 24.5;
        for n in ns {
            let r = prim(n);
            let rh = fr;
            let (top, ch) = (y + 1.0, rh - 1.0);
            let mid = |hh: f64| top + (ch - hh) / 2.0;
            out.rows.push((y, y + rh, n.to_string()));
            if *n == sel { for k in 0..9 { cx.b(cxs[k], y, cw[k], rh, Col::White, 0.06, 0.0); } }
            for k in 0..9 { cx.b(cxs[k], y, cw[k], 1.0, Col::SoftLine, 1.0, 0.0); }
            let nt = st(12.5, 600, false, 0.0, Col::Text);
            cx.t(4.0, mid(15.0), *n, nt, 1.0);
            if truthy(&r["note"]) { let x = 4.0 + cx.w(n, nt); cx.t(x, mid(15.0) + 1.09375, "*", st(9.0, 600, false, 0.0, Col::Muted), 1.0); }
            let vt = st(11.5, 600, true, 0.0, Col::Text);
            let v = mt_fmt(&r["live"]["chg13"], r["unit"].as_str().unwrap_or(""));
            let x = cxs[1] + 66.0 - cx.w(&v, vt);
            cx.t(x, gt(mid(11.5), 11.5, 11.5, true), v, vt, 1.0);
            // 오늘 칸
            let na = st(11.0, 400, false, 0.0, Col::Muted);
            match today_key(n) {
                None => cx.t(cxs[2], mid(13.0), "—", na, 1.0),
                Some(k) => {
                    let b = i.day(k);
                    match chg(&b) {
                        None => cx.t(cxs[2], mid(13.0), "…", na, 1.0),
                        Some(v) => {
                            if let Some((els, _, _)) = day_chart(i, k, 64.0, 20.0, 0.0) { cx.s(cxs[2], mid(20.0), 64.0, 20.0, [64.0, 20.0], false, 1.0, els); }
                            cx.t(cxs[2] + 70.0, gt(mid(11.0), 11.0, 11.0, true), format!("{}{}%", pm(v), fixed(v.abs(), 2)), st(11.0, 700, true, 0.0, day_col_k(k, Some(v))), 1.0);
                        }
                    }
                }
            }
            // 위치 게이지
            let gy = mid(6.0);
            let gx = cxs[3];
            cx.b(gx, gy, 84.0, 6.0, Col::White, 0.05, 3.0);
            cx.o(gx, gy, 84.0, 6.0, 1.0, Col::SoftLine, 1.0, 3.0);
            cx.b(gx + 84.0 * 0.4167, gy, 84.0 * 0.1666, 6.0, Col::White, 0.08, 0.0);
            cx.b(gx + 42.0, gy - 2.0, 1.0, 10.0, Col::Line, 1.0, 0.0);
            if let Some(z) = num(&r["live"]["z"]) { let pct = fx1(50.0 + z.clamp(-3.0, 3.0) / 3.0 * 50.0); cx.b(gx + 84.0 * pct / 100.0 - 4.0, gy - 1.0, 8.0, 8.0, Col::Text, 1.0, 4.0); }
            for (k, (kind, a)) in MT_CELLS.iter().enumerate() { tile_of(&mut cx, cxs[4 + k] + 6.0, mid(22.0), by(n, kind).filter(|r| r["assets"][*a].is_object()), a); }
            // 수신 불
            let (fst, lab, _) = feed(i, r, n);
            let dy = mid(8.0);
            let (dc, da) = match fst { 0 => (Col::Good, 1.0), 1 => (Col::Bad, 1.0), _ => (Col::Muted, 0.5) };
            cx.b(cxs[8], dy, 8.0, 8.0, dc, da, 4.0);
            if fst == 1 { cx.o(cxs[8] - 3.0, dy - 3.0, 14.0, 14.0, 3.0, Col::Bad, 0.22, 7.0); }
            cx.t(cxs[8] + 14.0, mid(13.0), lab, st(10.5, 400, false, 0.0, Col::Muted), 1.0);
            y += rh;
        }
    }
    // ── 상세 칸 ──
    let rs: Vec<&Value> = [by(&sel, "co"), by(&sel, "lead")].into_iter().flatten().collect();
    let r0 = rs[rs.len() - 1];
    let l0 = &r0["live"];
    let dx0 = lw + 28.0;
    cx.b(dx0, 10.0, 1.0, wrap_h, Col::SoftLine, 1.0, 0.0);
    let dx = dx0 + 29.0;
    let dw = w - dx;
    out.det = [dx0 + 1.0, 10.0, w - dx0 - 1.0, wrap_h];
    cx.k = true;
    let mut y = 10.0;
    let ht = st(16.0, 700, false, 0.0, Col::Text);
    cx.t(dx, y, sel.clone(), ht, 1.0);
    let zs = num(&l0["z"]).map_or("–".into(), |z| format!("{}{}σ", pm(z), fixed(z.abs(), 1)));
    let x = dx + cx.w(&sel, ht) + 12.0;
    cx.t(x, y + 2.0, format!("{} · {zs}", mt_fmt(&l0["chg13"], r0["unit"].as_str().unwrap_or(""))), st(13.0, 600, true, 0.0, Col::Text), 1.0);
    y += 19.0 + 12.0;
    let meta = st(12.0, 400, false, 0.0, Col::Muted);
    let mut runs = vec![(format!("{} 주까지 13주 변화 · 수신 {}", l0["week"].as_str().filter(|s| !s.is_empty()).unwrap_or("–"), feed(i, r0, &sel).2.replace('\n', " · ")), meta)];
    if let Some(nt) = r0["note"].as_str().filter(|s| !s.is_empty()) { runs.push(("\n".into(), meta)); runs.push(("주의".into(), st(12.0, 600, false, 0.0, Col::Amber))); runs.push((format!(" {nt}"), meta)); }
    y += para(&mut cx, dx, y, dw, &runs, 18.6) + 12.0;
    // 왜 목록
    let cells: Vec<(&Value, &str)> = rs.iter().flat_map(|x| ["BTC", "ETH"].into_iter().filter(|a| x["assets"][*a].is_object()).map(move |a| (*x, a))).collect();
    let kt = st(11.0, 600, true, 0.0, Col::Muted);
    let wt = st(12.0, 400, false, 0.0, Col::Text);
    let kind_ko = |x: &Value| if x["kind"] == "co" { "지금" } else { "13주 뒤" };
    for (n, (x, a)) in cells.iter().enumerate() {
        if n > 0 { y += 6.0; }
        cx.t(dx, gt(y, 22.0, 11.0, true), format!("{a} {}", kind_ko(x)), kt, 1.0);
        tile_of(&mut cx, dx + 84.0, y, Some(x), a);
        let c = &x["live"]["assets"][*a];
        let why = c["why"].as_str().filter(|s| !s.is_empty()).map(str::to_string).unwrap_or_else(|| if truthy(&x["live"]) { "표본 부족(ETF 이후 13주 변화 30주 미만)".into() } else { "지금 값 없음(원천 실패)".into() });
        let hh = para(&mut cx, dx + 134.0, y + 3.0, dw - 134.0, &[(why, wt)], 17.4);
        y += (3.0 + hh).max(22.0);
    }
    y += 12.0;
    // 오늘 24시간 차트
    let fig = |cx: &mut Cx, y: f64, hh: f64| { cx.b(dx, y, dw, hh, Col::White, 0.05, 12.0); cx.o(dx, y, dw, hh, 1.0, Col::SoftLine, 1.0, 12.0); };
    let (ix, iw) = (dx + 13.0, dw - 26.0);
    let cap = st(12.0, 600, false, 0.0, Col::Text);
    let capv = st(11.0, 600, true, 0.0, Col::Muted);
    if let Some(tk) = today_key(&sel) {
        let b = i.day(tk);
        if let Some(v) = chg(&b) {
            let sub = match tk { "ZN" => format!(" · 금리{} · 약 10분 지연", if v < 0.0 { "↑" } else { "↓" }), "VIX" => format!(" · {} · 약 15분 지연", if v > 0.0 { "위험 회피↑" } else { "위험 선호↑" }), _ => String::new() };
            fig(&mut cx, y, 160.0);
            let yi = y + 11.0;
            cx.t(ix, yi, format!("{} 24시간 · 앞 12시간 주요 발표", tname(tk)), cap, 1.0);
            let sv = format!("오늘 {}{}%{sub}", pm(v), fixed(v.abs(), 2));
            let x = ix + iw - cx.w(&sv, capv);
            cx.t(x, yi - 2.0, sv, capv, 1.0);
            let (els, labs, [t0, now]) = day_chart(i, tk, 520.0, 110.0, 12.0).unwrap();
            let dc = yi + 14.0;
            cx.s(ix, dc, iw, 110.0, [520.0, 110.0], false, 1.0, els);
            let lt = st(9.5, 400, false, 0.0, Col::Amber);
            for (k, (s, pp)) in labs.iter().enumerate() {
                let top = dc + 2.0 + (k % 2) as f64 * 13.0;
                let maxw = iw * (if *pp > 0.85 { *pp } else { 1.0 - pp }) * 100.0 / 100.0;
                let s = ellipsis(&cx, s, lt, maxw - 6.0);
                let tw = cx.w(&s, lt);
                let x = if *pp > 0.85 { ix + iw * (1.0 - fixed((1.0 - pp) * 100.0, 2).parse::<f64>().unwrap() / 100.0) - 3.0 - tw } else { ix + iw * fixed(pp * 100.0, 2).parse::<f64>().unwrap() / 100.0 + 3.0 };
                cx.t(x, top, s, lt, 1.0);
            }
            let dt = st(9.5, 400, false, 0.0, Col::Muted);
            let ty = dc + 110.0 + 3.0;
            let p_now: f64 = fixed((now - t0) / (now + 12.0 * 3600.0 - t0) * 100.0, 2).parse().unwrap();
            cx.t(ix, ty, hm(t0, i.tz), dt, 1.0);
            let s = format!("지금 {}", hm(now, i.tz));
            let right: f64 = fixed(100.0 - p_now, 2).parse().unwrap();
            let x = ix + iw * (1.0 - right / 100.0) - cx.w(&s, dt);
            cx.t(x, ty, s, dt, 1.0);
            let x = ix + iw - cx.w("+12시간", dt);
            cx.t(x, ty, "+12시간", dt, 1.0);
            y += 160.0 + 12.0;
        }
    }
    // 관계 표 -- 열 폭 = 내용 최대 폭(+ 안쪽 여백 12)을 표 폭에 비례 배분(auto 표)
    let th = st(10.0, 700, false, 0.8, Col::Muted);
    let tdm = st(12.0, 400, true, 0.0, Col::Text);
    let tds = st(12.0, 400, false, 0.0, Col::Text);
    let heads = ["관계", "판정", "ETF 이후 r", "순열 p", "전반 / 후반", "2020–23", "최근 52주"];
    let body_rows: Vec<[String; 7]> = cells.iter().map(|(x, a)| {
        let v = &x["assets"][*a];
        let c = &x["live"]["assets"][*a];
        [format!("{a} {}", kind_ko(x)), v["verdict"].as_str().unwrap_or("undefined").to_string(), mt_r(&v["r_p3"]), num(&v["perm_p"]).map_or("–".into(), |q| fixed(q, 3)),
         format!("{} / {}", mt_r(&v["r_first_half"]), mt_r(&v["r_second_half"])), mt_r(&v["r_2020_23"]), mt_r(&c["roll52"])]
    }).collect();
    let mut mx = [0.0f64; 7];
    for k in 0..7 {
        mx[k] = cx.w(heads[k], th) + 12.0;
        for r in &body_rows { mx[k] = mx[k].max(cx.w(&r[k], if k == 1 { tds } else { tdm }) + 12.0); }
    }
    let tot: f64 = mx.iter().sum();
    let tw: Vec<f64> = mx.iter().map(|q| q * dw.max(tot) / tot).collect();
    let mut tx = [dx; 7];
    for k in 1..7 { tx[k] = tx[k - 1] + tw[k - 1]; }
    let place = |cx: &Cx, k: usize, s: &str, t: St| if k < 2 { tx[k] + 6.0 } else { tx[k] + tw[k] - 6.0 - cx.w(s, t) };
    for k in 0..7 { let x = place(&cx, k, heads[k], th); cx.t(x, gt(y, 10.0, 10.0, false), heads[k], th, 1.0); }
    for (n, r) in body_rows.iter().enumerate() {
        let rt = y + 16.5 + 27.0 * n as f64;
        for k in 0..7 {
            cx.b(tx[k], rt, tw[k], 1.0, Col::SoftLine, 1.0, 0.0);
            let t = if k == 1 { tds } else { tdm };
            let x = place(&cx, k, &r[k], t);
            cx.t(x, rt + 5.5 + if k == 1 { 1.0 } else { 0.0 }, r[k].clone(), t, 1.0);
        }
    }
    y += 16.0 + 27.0 * body_rows.len() as f64 + 12.0;
    // 겹친 그림(지금 · 13주 뒤)
    for x in &rs {
        let a = if x["assets"]["ETH"].is_object() { "ETH" } else { "BTC" };
        let Some(els) = overlay(x, a) else { continue };
        fig(&mut cx, y, 162.0);
        let yi = y + 11.0;
        cx.t(ix, yi, format!("{a} · {}", if x["kind"] == "lead" { "13주 뒤 — 지표를 14주 앞으로" } else { "지금 — 같은 13주" }), cap, 1.0);
        let sv = format!("최근 52주 r {}", mt_r(&x["live"]["assets"][a]["roll52"]));
        let xr = ix + iw - cx.w(&sv, capv);
        cx.t(xr, yi - 2.0, sv, capv, 1.0);
        cx.s(ix, yi + 14.0 + 6.0, iw, 120.0, [520.0, 120.0], false, 1.0, els);
        y += 162.0 + 12.0;
    }
    para(&mut cx, dx, y, dw, &[("굵은 선 = 지표 13주 변화(위 = 코인 강세 쪽) · 회색 = 코인 13주 수익 · 52주 안 표준화(모양만) · 표본내 연구(13주 독립 관측 ≈11), 신호 아님".into(), meta)], 18.6);
    // ── 꼬리 줄 ──
    cx.k = false;
    let week = rows.iter().filter_map(|r| r["live"]["week"].as_str().filter(|s| !s.is_empty())).max();
    let errs: Vec<&str> = p["errors"].as_object().map(|o| o.keys().map(String::as_str).collect()).unwrap_or_default();
    let late: Vec<&str> = names.iter().copied().filter(|n| feed(i, prim(n), n).0 == 1).collect();
    let ft = st(11.0, 400, false, 0.0, Col::Muted);
    let warn = st(11.0, 600, false, 0.0, Col::Amber);
    let mut runs = vec![(format!("기준 주 {} · FRED·DefiLlama·바이낸스 일·주 단위(1시간마다 갱신)", week.unwrap_or("–")), ft)];
    // 글 마디 = 원본 그대로(« · »는 앞 글 마디 끝에 붙는다)
    let sep = |runs: &mut Vec<(String, St)>, s: String| { match runs.last_mut() { Some(l) if l.1.c == Col::Muted => l.0 += " · ", _ => runs.push((" · ".into(), ft)) } runs.push((s, warn)); };
    if !errs.is_empty() { sep(&mut runs, format!("원천 실패 {}({})", errs.len(), errs.join(", "))); }
    if !late.is_empty() { sep(&mut runs, format!("수신 늦음 {}({})", late.len(), late.join(", "))); }
    let mut x = 0.0;
    let fy = 10.0 + wrap_h + 14.0;
    for (s, t) in runs { let ww = cx.w(&s, t); cx.t(x, fy, s, t, 1.0); x += ww; }
    out.items = cx.v;
    out.sel = Some(sel);
    out.left = lw;
    out
}

/// 3527 mtHeadHtml → (그림, 폭, 높이) -- 단추(.fr-mt) 왼쪽 위 기준
pub fn head(i: &In, m: Measure) -> (Vec<It>, f64, f64) {
    let mut cx = Cx { m, v: vec![], k: false };
    let Some(p) = i.trend.filter(|p| p["ok"] == true) else {
        let (a, b) = (st(12.0, 700, false, 0.0, Col::Text), st(10.5, 400, false, 0.0, Col::Muted));
        let s = if i.trend.is_some() { "불러오지 못함" } else { "불러오는 중…" };
        let (wa, wb) = (cx.w("Macro", a), cx.w(s, b));
        let gw = wa.max(wb);
        cx.t(10.0 + gw - wa, 4.0, "Macro", a, 1.0);
        cx.t(10.0 + gw - wb, gt(4.0 + 14.0 + 3.0, 10.5, 10.5, false), s, b, 1.0);
        return (cx.v, gw + 20.0, 35.5);
    };
    let s = &p["summary"];
    let lab = st(10.0, 700, false, 1.2, Col::Muted);
    let tx = st(11.0, 400, false, 0.0, Col::Text);
    let mu = st(11.0, 400, false, 0.0, Col::Muted);
    let bt = st(11.0, 700, true, 0.0, Col::Good);
    let sp = (cx.m)(" ", 11.0, 400, false);
    let n = |x: &Value, k: &str| or0(&x[k]);
    // 줄마다 [이름, 지금 칸, 13주 뒤 칸] -- 칸 = [(글, 꼴, 아이콘 방향)]
    type Run = (String, St, i8);
    let rows: Vec<(String, Vec<Run>, Vec<Run>)> = ["BTC", "ETH"].iter().map(|a| {
        let x = &s[*a];
        (a.to_string(), vec![("지금 ".into(), tx, 0), (n(x, "co_up"), bt, 1), (" ".into(), tx, 0), (n(x, "co_dn"), St { c: Col::Bad, ..bt }, -1)],
         vec![("13주 뒤 ".into(), tx, 0), (n(x, "lead_up"), bt, 1), (" ".into(), tx, 0), (n(x, "lead_dn"), St { c: Col::Bad, ..bt }, -1), (" ".into(), tx, 0),
              (format!("· 중립 {} · 안 셈 {}", n(x, "neutral"), n(x, "excluded")), mu, 0)])
    }).collect();
    let rw = |cx: &Cx, r: &[Run]| r.iter().map(|(s, t, d)| if s == " " { sp } else { cx.w(s, *t) + if *d != 0 { 12.0 } else { 0.0 } }).sum::<f64>();
    let c0 = rows.iter().map(|r| cx.w(&r.0, lab)).fold(0.0, f64::max);
    let c1 = rows.iter().map(|r| rw(&cx, &r.1)).fold(0.0, f64::max);
    let c2 = rows.iter().map(|r| rw(&cx, &r.2)).fold(0.0, f64::max);
    let xs = [10.0, 10.0 + c0 + 10.0, 10.0 + c0 + 10.0 + c1 + 10.0];
    for (k, (a, r1, r2)) in rows.iter().enumerate() {
        let top = 4.0 + 17.0 * k as f64;
        let base = top + 10.0;
        cx.t(xs[0] + c0 - cx.w(a, lab), top + 1.5 - 1.0, a.clone(), lab, 1.0);
        for (r, x0, cwid) in [(r1, xs[1], c1), (r2, xs[2], c2)] {
            let mut x = x0 + cwid - rw(&cx, r);
            for (s, t, d) in r {
                if s == " " { x += sp; continue; }
                if *d != 0 { cx.s(x, base + 1.0 - 11.0, 11.0, 11.0, [14.0, 14.0], true, 1.0, ico(*d, t.c)); x += 12.0; cx.t(x, base - 11.0, s.clone(), *t, 1.0); } else { cx.t(x, top, s.clone(), *t, 1.0); }
                x += cx.w(s, *t);
            }
        }
    }
    let mtg = c0 + 10.0 + c1 + 10.0 + c2;
    let mut wd = 10.0 + mtg + 10.0;
    let now = i.now;
    let mut nx: Vec<(f64, String)> = i.majors().into_iter().map(|(t, nm)| (t * 1000.0, nm)).filter(|e| e.0 > now).collect();
    nx.sort_by(|p, q| p.0.total_cmp(&q.0));
    if let Some((t, nm)) = nx.first() {
        let left = t - now;
        let sx = 10.0 + mtg + 14.0;
        cx.b(sx, 4.0, 1.0, 30.0, Col::SoftLine, 1.0, 0.0);
        let sm = st(10.5, 400, false, 0.0, Col::Muted);
        let ev = st(11.0, 700, true, 0.0, Col::Amber);
        let s = ellipsis(&cx, &format!("{nm} {}:{:02} 뒤", (left / 3600e3).floor(), ((left % 3600e3) / 60e3).floor()), ev, 280.0);
        let (ws, we) = (cx.w("다음 발표", sm), cx.w(&s, ev));
        let cw = ws.max(we);
        let x0 = sx + 1.0 + 14.0;
        cx.t(x0 + cw - ws, 5.0, "다음 발표", sm, 1.0);
        cx.t(x0 + cw - we, gt(5.0 + 13.0 + 4.0, 11.0, 11.0, true), s, ev, 1.0);
        wd += 14.0 + 1.0 + 14.0 + cw;
    }
    (cx.v, wd, 38.0)
}

// ── 그리기 ──
pub(crate) fn rr_pts(x: f64, y: f64, w: f64, h: f64, r: f64) -> Vec<[f64; 2]> {
    let r = r.min(w / 2.0).min(h / 2.0).max(0.0);
    let mut v = vec![];
    for (cx, cy, a0) in [(x + w - r, y + r, -90f64), (x + w - r, y + h - r, 0.0), (x + r, y + h - r, 90.0), (x + r, y + r, 180.0)] {
        for k in 0..=6 { let a = (a0 + 15.0 * k as f64).to_radians(); v.push([cx + r * a.cos(), cy + r * a.sin()]); }
    }
    v.push(v[0]);
    v
}
/// 그림 단위 → 그리기 목록(k = Clip::Plot -- paint 의 plot 상자 = Out::det) · 자간 글은 글자마다
pub fn prims(items: &[It], m: &dyn Fn(&str, f64, u16, bool) -> f64) -> Vec<Prim> {
    let mut v = vec![];
    let cl = |k: bool| if k { Clip::Plot } else { Clip::No };
    for it in items {
        match it {
            It::T { x, y, s, size, wt, mono, ls, c, a, k } => {
                let base = y + asc(*size, *mono);
                let one = |x: f64, s: String| if *mono { Prim::Mono { x, y: base, s, size: *size, wt: *wt, c: *c, a: *a, anchor: Anchor::Start, halo: false, clip: cl(*k) } } else { Prim::Text { x, y: base, s, size: *size, wt: *wt, c: *c, a: *a, anchor: Anchor::Start, halo: false, clip: cl(*k) } };
                if *ls == 0.0 { v.push(one(*x, s.clone())); continue; }
                let mut xx = *x;
                for ch in s.chars() { let t = ch.to_string(); let w = m(&t, *size, *wt, *mono); v.push(one(xx, t)); xx += w + ls; }
            }
            &It::B { x, y, w, h, c, a, rad, k } => v.push(Prim::Rect { x, y, w, h, c, a, rx: rad, clip: cl(k) }),
            &It::O { x, y, w, h, lw, c, a, rad, k } => {
                let (x, y, w, h) = (x + lw / 2.0, y + lw / 2.0, w - lw, h - lw);
                if rad == 0.0 { v.push(Prim::StrokeRect { x, y, w, h, c, a, lw, clip: cl(k) }); } else { v.push(Prim::Line { pts: rr_pts(x, y, w, h, rad - lw / 2.0), c, a, lw, dash: None, clip: cl(k) }); }
            }
            It::S { x, y, w, h, vb, meet, a, k, els } => {
                let (mut sx, mut sy) = (w / vb[0], h / vb[1]);
                let (mut ox, mut oy) = (0.0, 0.0);
                if *meet { let s = sx.min(sy); (sx, sy) = (s, s); ox = (w - vb[0] * s) / 2.0; oy = (h - vb[1] * s) / 2.0; }
                let pt = |p: [f64; 2]| [x + ox + p[0] * sx, y + oy + p[1] * sy];
                for e in els {
                    match e {
                        El::Rect { r, c, a: ea } => v.push(Prim::Rect { x: x + ox + r[0] * sx, y: y + oy + r[1] * sy, w: r[2] * sx, h: r[3] * sy, c: *c, a: ea * a, rx: 0.0, clip: cl(*k) }),
                        El::Line { p, c, a: ea, sw, dash, ns } => {
                            let lw = if *ns { *sw } else { sw * sx };
                            let pts = vec![pt([p[0], p[1]]), pt([p[2], p[3]])];
                            v.push(match dash { Some(d) => Prim::Dashed { pts, c: *c, a: ea * a, lw, on: d[0], off: d[1], clip: cl(*k) }, None => Prim::Line { pts, c: *c, a: ea * a, lw, dash: None, clip: cl(*k) } });
                        }
                        El::Path { segs, c, a: ea, sw, dash, ns } => {
                            let lw = if *ns { *sw } else { sw * sx };
                            for sg in segs {
                                let pts: Vec<_> = sg.iter().map(|q| pt(*q)).collect();
                                if pts.len() < 2 { continue; }
                                v.push(match dash { Some(d) => Prim::Dashed { pts, c: *c, a: ea * a, lw, on: d[0], off: d[1], clip: cl(*k) }, None => Prim::Line { pts, c: *c, a: ea * a, lw, dash: None, clip: cl(*k) } });
                            }
                        }
                    }
                }
            }
        }
    }
    v
}

// ── 받기 ──
use crate::feed::{Kind, Req};
/// 원본 전역(latestMacroTrend · mtTreas · mtIntra · xyzBars · xyzLive · xyzSkew · latestMacroEvents · mtSel)과 받기 시각
#[derive(Default)]
pub struct Store {
    /// ?card=mth = 머리 요약만
    pub head: bool,
    pub trend: Option<Value>, pub events: Option<Value>, pub treas: Option<(f64, Value)>, pub intra: Option<(f64, Value)>,
    pub xyz: Vec<[f64; 2]>, pub xyz_live: Option<Value>, pub skew: f64, pub sel: Option<String>,
    /// 본문을 마지막으로 다시 지은 시각(ms · 원본 renderMacroTrend 를 부른 때) -- 0 = 아직
    pub drawn: f64,
    trend_at: f64, cal_at: f64, today_at: f64, xyz_at: f64, xyz_busy: bool, treas_req: f64,
    /// 마지막 본문의 줄 자리(누르기)
    pub rows: Vec<(f64, f64, String)>, pub left: f64,
}
const MT_POLL_MS: f64 = 600_000.0;
const CAL_POLL_MS: f64 = 6.0 * 3600e3;
impl Store {
    /// app.js tick(숨으면 아무것도 안 함): refreshMacroCalendar · refreshMacroTrend · refreshMtToday · refreshXyz(넓은 화면 밖 = 1분)
    pub fn tick(&mut self, now: f64, hidden: bool, api: &str, out: &mut Vec<Req>) {
        if hidden { return; }
        let mut get = |kind, path: String| out.push(Req::Get { kind, url: format!("{api}{path}") });
        if now - self.cal_at >= CAL_POLL_MS { self.cal_at = now; get(Kind::MtCal, "/api/macro-calendar".into()); }
        if now - self.trend_at >= MT_POLL_MS { self.trend_at = now; get(Kind::MtTrend, "/api/macro-trend".into()); }
        if self.head { return; } // 머리 요약 = 달력 + macro-trend 만(renderMarketCtx 가 1초마다 그 값을 읽는다)
        if now - self.today_at >= 60_000.0 {
            self.today_at = now;
            if self.treas.as_ref().is_none_or(|t| now - t.0 > 5.0 * 60_000.0) { self.treas_req = now; get(Kind::MtTreas, "/api/treasury-futures?h=24".into()); }
            get(Kind::MtToday, "/api/macro-today".into());
        }
        if !self.xyz_busy && now - self.xyz_at >= 60_000.0 {
            self.xyz_at = now; self.xyz_busy = true;
            let since = self.xyz.last().map_or(0.0, |b| b[0] - 60.0);
            get(Kind::MtXyz, if since != 0.0 { format!("/api/xyz100?since={}", js(&Value::from(since))) } else { "/api/xyz100".into() });
        }
    }
    /// 응답(body = None 이면 실패 -- 접착은 HTTP 오류도 None) · now = Date.now()
    pub fn on_get(&mut self, kind: Kind, body: Option<&str>, now: f64) {
        let v = body.and_then(|b| serde_json::from_str::<Value>(b).ok());
        match kind {
            Kind::MtTrend => match v { Some(v) => { self.trend = Some(v); self.drawn = now; } None => self.trend_at = self.trend_at - MT_POLL_MS + 60_000.0 },
            Kind::MtCal => match v { Some(v) => self.events = Some(if v["events"].is_array() { v["events"].clone() } else { Value::Array(vec![]) }), None => self.cal_at = self.cal_at - CAL_POLL_MS + 60_000.0 },
            // ponytail: 국채·오늘을 같이 묻는다(원본은 국채를 기다린 뒤 오늘) -- 늦게 온 국채도 바로 다시 짓는다
            Kind::MtTreas => if let Some(v) = v { self.treas = Some((self.treas_req, v)); if self.trend.is_some() { self.drawn = now; } },
            Kind::MtToday => { if let Some(v) = v { self.intra = Some((now / 1000.0, v)); } if self.trend.is_some() { self.drawn = now; } }
            Kind::MtXyz => {
                self.xyz_busy = false;
                let Some(p) = v else { return };
                if let Some(t) = num(&p["now_sec"]) { self.skew = t - now / 1000.0; }
                if p["ok"] != true { self.xyz_live = None; return; }
                let bars: Vec<[f64; 2]> = p["bars"].as_array().into_iter().flatten().map(|b| [f(&b[0]), f(&b[1])]).collect();
                let first = self.xyz.is_empty();
                if truthy(&p["full"]) || first { self.xyz = bars; } else if let (Some(a), Some(b)) = (bars.first(), bars.last()) {
                    let (t0, cut) = (a[0], b[0] - 86400.0);
                    self.xyz.retain(|q| q[0] < t0 && q[0] >= cut);
                    self.xyz.extend(bars);
                }
                self.xyz_live = Some(p["live"].clone()).filter(|l| !l.is_null());
                if first && self.trend.is_some() { self.drawn = now; }
            }
            _ => {}
        }
    }
    pub fn input(&self, now: f64, tz: i64) -> In<'_> {
        In { now, tz, trend: self.trend.as_ref(), events: self.events.as_ref(), xyz: &self.xyz, xyz_live: self.xyz_live.as_ref(), skew: self.skew,
             intra: self.intra.as_ref().map(|(a, p)| (*a, p)), treas: self.treas.as_ref().map(|(a, p)| (*a, p)), sel: self.sel.as_deref() }
    }
    /// 줄 누르기(3621: 줄 어디든 -- 왼쪽 표 안) → 다시 짓기
    pub fn click(&mut self, x: f64, y: f64, now: f64) -> bool {
        if x < 0.0 || x > self.left { return false; }
        let Some(n) = self.rows.iter().find(|r| y >= r.0 && y < r.1).map(|r| r.2.clone()) else { return false };
        self.sel = Some(n);
        self.drawn = now;
        true
    }
}

#[cfg(test)]
pub(crate) mod t {
    use super::*;

    fn fx(name: &str) -> Value { serde_json::from_str(&std::fs::read_to_string(format!("{}/compare/fixture/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap() }
    fn nn(v: &Value) -> Option<&Value> { Some(v).filter(|v| !v.is_null()) }
    fn bars(v: &Value) -> Vec<[f64; 2]> { v.as_array().into_iter().flatten().map(|b| [f(&b[0]), f(&b[1])]).collect() }
    fn input<'a>(s: &'a Value, xyz: &'a [[f64; 2]]) -> In<'a> {
        In { now: f(&s["now"]) * 1000.0, tz: -540, trend: nn(&s["trend"]), events: nn(&s["events"]), xyz, xyz_live: s["xyz"].get("live").filter(|v| !v.is_null()), skew: 0.0,
             intra: nn(&s["intra"]).map(|v| (f(&v["at"]), &v["p"])), treas: nn(&s["treas"]).map(|v| (f(&v["at"]), &v["p"])), sel: s["sel"].as_str() }
    }
    pub(crate) fn wb(wt: f64, mono: bool) -> u16 { if mono { if wt >= 600.0 { 700 } else { 400 } } else if wt <= 500.0 { 400 } else if wt < 750.0 { 700 } else { 800 } }
    pub(crate) fn r4(v: &Value) -> [f64; 4] { let a = v.as_array().unwrap(); [f(&a[0]), f(&a[1]), f(&a[2]), f(&a[3])] }
    /// 색 = 그려지는 값(미리 곱한 8비트)
    pub(crate) fn col(c: Col, a: f64) -> [f64; 4] { let q = crate::draw::rgba(c, a); [q.r(), q.g(), q.b(), q.a()].map(|v| v as f64) }
    pub(crate) fn pmc(c: [f64; 4]) -> [f64; 4] { let q = eframe::egui::Color32::from_rgba_unmultiplied(c[0].round() as u8, c[1].round() as u8, c[2].round() as u8, (c[3] * 255.0).round() as u8); [q.r(), q.g(), q.b(), q.a()].map(|v| v as f64) }
    pub(crate) fn rad(s: &Value, r: [f64; 4]) -> f64 { let s = s.as_str().unwrap_or("0px"); if s.ends_with('%') { r[2].min(r[3]) / 2.0 } else { s.trim_end_matches("px").parse().unwrap_or(0.0) } }

    /// 비교 단위: 글(t: x · 위 · 글) · 칠(b) · 고리(o) · svg(s: 상자 · 요소 글로 적은 것)
    #[derive(Debug, Clone)]
    pub(crate) struct Q { k: char, r: [f64; 4], c: [f64; 4], s: String, z: f64, wt: u16, mono: bool, lw: f64, rad: f64, kk: bool }
    fn q(k: char, r: [f64; 4], c: [f64; 4]) -> Q { Q { k, r, c, s: String::new(), z: 0.0, wt: 0, mono: false, lw: 0.0, rad: 0.0, kk: false } }
    /// 요소 → 비교 글(색 8비트 · 수 1e-6 은 숫자 비교라 따로)
    fn el_key(e: &El) -> (String, Vec<f64>) {
        let cs = |c: Col, a: f64| format!("{:?}", col(c, a));
        match e {
            El::Line { p, c, a, sw, dash, ns } => (format!("line {} {sw} {dash:?} {ns}", cs(*c, *a)), p.to_vec()),
            El::Rect { r, c, a } => (format!("rect {}", cs(*c, *a)), r.to_vec()),
            El::Path { segs, c, a, sw, dash, ns } => (format!("path {} {sw} {dash:?} {ns} {:?}", cs(*c, *a), segs.iter().map(Vec::len).collect::<Vec<_>>()), segs.iter().flatten().flat_map(|p| p.to_vec()).collect()),
        }
    }
    /// 잰 svg 요소 → 같은 꼴
    fn el_exp(e: &Value) -> (String, Vec<f64>) {
        let at = &e["at"];
        let a = |k: &str| at[k].as_str().unwrap().parse::<f64>().unwrap();
        let dash = match e["dash"].as_str().unwrap() { "none" => None, d => { let v: Vec<f64> = d.split(',').map(|x| x.trim().trim_end_matches("px").parse().unwrap()).collect(); Some([v[0], v[1]]) } };
        let ns = e["ve"] == "non-scaling-stroke";
        let sc = |k: &str, o: &str| { let c = r4(&e[k]); let c = [c[0], c[1], c[2], c[3] * f(&e[o])]; format!("{:?}", pmc(c)) };
        let pts = |s: &str| -> Vec<Vec<[f64; 2]>> {
            let mut segs: Vec<Vec<[f64; 2]>> = vec![];
            let mut i = 0;
            let b = s.as_bytes();
            while i < b.len() {
                let c = b[i];
                let j = (i + 1..b.len()).find(|&q| b[q] == b'M' || b[q] == b'L').unwrap_or(b.len());
                let xy: Vec<f64> = s[i + 1..j].split(',').map(|x| x.trim().parse().unwrap()).collect();
                if c == b'M' { segs.push(vec![[xy[0], xy[1]]]); } else { segs.last_mut().unwrap().push([xy[0], xy[1]]); }
                i = j;
            }
            segs
        };
        match e["tag"].as_str().unwrap() {
            "line" => (format!("line {} {} {dash:?} {ns}", sc("stroke", "so"), f(&e["sw"])), vec![a("x1"), a("y1"), a("x2"), a("y2")]),
            "rect" => (format!("rect {}", sc("fill", "fo")), vec![a("x"), a("y"), a("width"), a("height")]),
            "polyline" => {
                let p: Vec<[f64; 2]> = at["points"].as_str().unwrap().split(' ').map(|xy| { let v: Vec<f64> = xy.split(',').map(|x| x.parse().unwrap()).collect(); [v[0], v[1]] }).collect();
                (format!("path {} {} {dash:?} {ns} {:?}", sc("stroke", "so"), f(&e["sw"]), vec![p.len()]), p.iter().flat_map(|q| q.to_vec()).collect())
            }
            "path" => { let sg = pts(at["d"].as_str().unwrap()); (format!("path {} {} {dash:?} {ns} {:?}", sc("stroke", "so"), f(&e["sw"]), sg.iter().map(Vec::len).collect::<Vec<_>>()), sg.iter().flatten().flat_map(|p| p.to_vec()).collect()) }
            t => panic!("{t}"),
        }
    }
    fn svg_key(els: &[(String, Vec<f64>)], vb: [f64; 2], meet: bool, a: f64) -> String { format!("{vb:?} {meet} {} {}", (a * 100.0).round(), els.iter().map(|e| e.0.clone()).collect::<Vec<_>>().join(" | ")) }

    pub(crate) fn expected(d: &Value) -> (Vec<Q>, Vec<Vec<(String, Vec<f64>)>>) {
        let mut out = vec![];
        let mut svgs = vec![];
        for t in d["texts"].as_array().unwrap() {
            let mono = t["mono"] == true;
            out.push(Q { k: 't', r: r4(&t["r"]), c: pmc(r4(&t["c"])), s: t["s"].as_str().unwrap().trim().into(), z: f(&t["size"]), wt: wb(f(&t["wt"]), mono), mono, lw: 0.0, rad: 0.0, kk: !t["k"].is_null() });
        }
        for b in d["boxes"].as_array().unwrap() { let r = r4(&b["r"]); out.push(Q { rad: rad(&b["rad"], r), kk: !b["k"].is_null(), ..q('b', r, pmc(r4(&b["c"]))) }); }
        for l in d["lines"].as_array().unwrap() {
            let r = r4(&l["r"]);
            if l["side"] == "all" { out.push(Q { lw: f(&l["w"]), rad: rad(&l["rad"], r), kk: !l["k"].is_null(), ..q('o', r, pmc(r4(&l["c"]))) }); } else { out.push(Q { kk: !l["k"].is_null(), ..q('b', r, pmc(r4(&l["c"]))) }); }
        }
        for g in d["rings"].as_array().unwrap() {
            let (r, s) = (r4(&g["r"]), f(&g["spread"]));
            let (r2, rd) = if g["inset"] == true { (r, rad(&g["rad"], r)) } else { ([r[0] - s, r[1] - s, r[2] + 2.0 * s, r[3] + 2.0 * s], rad(&g["rad"], r) + s) };
            out.push(Q { lw: s, rad: rd, kk: !g["k"].is_null(), ..q('o', r2, pmc(r4(&g["c"]))) });
        }
        for s in d["svgs"].as_array().unwrap() {
            let els: Vec<_> = s["els"].as_array().unwrap().iter().map(el_exp).collect();
            let vb = r4(&s["vb"]);
            out.push(Q { s: svg_key(&els, [vb[2], vb[3]], s["par"] != "none", f(&s["a"])), kk: !s["k"].is_null(), ..q('s', r4(&s["r"]), [0.0; 4]) });
            svgs.push(els);
        }
        (out, svgs)
    }
    pub(crate) fn actual(items: &[It], m: Measure) -> (Vec<Q>, Vec<Vec<(String, Vec<f64>)>>) {
        let mut out = vec![];
        let mut svgs = vec![];
        for it in items {
            out.push(match it {
                It::T { x, y, s, size, wt, mono, ls, c, a, k } => Q { k: 't', r: [*x, *y, m(s, *size, *wt, *mono) + ls * s.chars().count() as f64, cont(*size, *mono)], c: col(*c, *a), s: s.trim().into(), z: *size, wt: wb(*wt as f64, *mono), mono: *mono, lw: 0.0, rad: 0.0, kk: *k },
                &It::B { x, y, w, h, c, a, rad, k } => Q { rad, kk: k, ..q('b', [x, y, w, h], col(c, a)) },
                &It::O { x, y, w, h, lw, c, a, rad, k } => Q { lw, rad, kk: k, ..q('o', [x, y, w, h], col(c, a)) },
                It::S { x, y, w, h, vb, meet, a, k, els } => {
                    let e: Vec<_> = els.iter().map(el_key).collect();
                    let s = svg_key(&e, *vb, *meet, *a);
                    svgs.push(e);
                    Q { s, kk: *k, ..q('s', [*x, *y, *w, *h], [0.0; 4]) }
                }
            });
        }
        (out, svgs)
    }
    pub(crate) const TOL: f64 = 0.07;
    fn same(a: &Q, e: &Q) -> bool {
        let n = |x: f64, y: f64, t: f64| (x - y).abs() <= t;
        let cs = (0..4).all(|k| (a.c[k] - e.c[k]).abs() <= 1.0);
        cs && a.k == e.k && a.kk == e.kk && match a.k {
            't' => a.s == e.s && a.z == e.z && a.wt == e.wt && a.mono == e.mono && n(a.r[0], e.r[0], 0.6) && n(a.r[1], e.r[1], TOL),
            'o' => n(a.lw, e.lw, 1e-9) && n(a.rad, e.rad, TOL) && (0..4).all(|k| n(a.r[k], e.r[k], TOL)),
            'b' => n(a.rad, e.rad, TOL) && (0..4).all(|k| n(a.r[k], e.r[k], TOL)),
            _ => a.s == e.s && (0..4).all(|k| n(a.r[k], e.r[k], TOL)),
        }
    }
    /// 하나씩 짝짓기 → (짝 없는 rtui, 짝 없는 원본, 짝지은 svg 쌍 차례)
    pub(crate) fn pair(act: &[Q], exp: &[Q]) -> (Vec<Q>, Vec<Q>, Vec<(usize, usize)>) {
        let mut used = vec![false; act.len()];
        let (mut miss, mut ps) = (vec![], vec![]);
        let (mut si, mut sj) = (vec![], vec![]);
        let mut n = 0;
        for a in act { si.push(if a.k == 's' { n += 1; n - 1 } else { usize::MAX }); }
        n = 0;
        for e in exp { sj.push(if e.k == 's' { n += 1; n - 1 } else { usize::MAX }); }
        for (j, e) in exp.iter().enumerate() {
            match (0..act.len()).find(|&k| !used[k] && same(&act[k], e)) { Some(k) => { used[k] = true; if e.k == 's' { ps.push((si[k], sj[j])); } } None => miss.push(e.clone()) }
        }
        (act.iter().zip(&used).filter(|q| !q.1).map(|q| q.0.clone()).collect(), miss, ps)
    }
    /// 글자 폭 = 크로미움이 잰 폭(글 조각 · 자간 뺌) · 따로 잰 빈칸 · 없으면 같은 꼴 글자별 폭 합(줄바꿈 접두)
    pub(crate) fn lookup(cases: &[&Value], probe: &Value) -> impl Fn(&str, f64, u16, bool) -> f64 {
        let mut whole: Vec<(String, f64, u16, bool, f64)> = vec![];
        let mut chars: Vec<(char, f64, u16, bool, f64)> = vec![];
        for d in cases {
            for t in d["texts"].as_array().unwrap() {
                let (mono, z) = (t["mono"] == true, f(&t["size"]));
                let wt = wb(f(&t["wt"]), mono);
                let s = t["s"].as_str().unwrap();
                let ls = f(&t["ls"]);
                whole.push((s.to_string(), z, wt, mono, r4(&t["r"])[2] - ls * s.chars().count() as f64));
                for c in t["ch"].as_array().into_iter().flatten() { chars.push((c[0].as_str().unwrap().chars().next().unwrap(), z, wt, mono, f(&c[1]) - ls)); }
            }
        }
        for p in probe.as_array().unwrap() { let mono = p[3] == true; whole.push((p[0].as_str().unwrap().into(), f(&p[1]), wb(f(&p[2]), mono), mono, f(&p[4]))); }
        move |s: &str, z: f64, wt: u16, mono: bool| {
            let wt = wb(wt as f64, mono);
            let hit = |s: &str| whole.iter().find(|q| q.0 == s && q.1 == z && q.2 == wt && q.3 == mono).map(|q| q.4);
            hit(s).or_else(|| hit(s.trim_end())).or_else(|| hit(s.trim())).unwrap_or_else(|| s.chars().map(|c| chars.iter().find(|q| q.0 == c && q.1 == z && q.2 == wt && q.3 == mono).map(|q| q.4)
                .unwrap_or_else(|| panic!("잰 폭 없음: {s:?} ({c:?}) {z} {wt} {mono}"))).sum())
        }
    }
    pub(crate) fn show(v: &[Q]) -> String { v.iter().map(|q| format!("{} {:?} {:?} {:?} z{} w{}{} lw{} rad{}{}", q.k, q.s.chars().take(90).collect::<String>(), q.r.map(|x| (x * 1000.0).round() / 1000.0), q.c, q.z, q.wt, if q.mono { " M" } else { "" }, q.lw, q.rad, if q.kk { " K" } else { "" })).collect::<Vec<_>>().join("\n") }

    /// 한 장 맞대기(본문 · 머리) → (맞춘 단위 수, 맞춘 svg 수 개수)
    fn check(k: usize, s: &Value, d: &Value, sel: &Value, m: Measure) -> (usize, usize) {
        let xyz = bars(&s["xyz"]["bars"]);
        let i = input(s, &xyz);
        let o = body(&i, f(&s["w"]), f(&s["h"]), m);
        assert_eq!(o.sel.as_deref(), sel.as_str(), "입력 {k} 고른 지표");
        let (hv, hw, hh) = head(&i, m);
        let (mut n, mut nums) = (0, 0);
        for (nm, items, dd, wh) in [("본문", o.items, &d["body"], None), ("머리", hv, &d["head"], Some((hw, hh)))] {
            let (act, sa) = actual(&items, m);
            let (mut exp, se) = expected(dd);
            if wh.is_some() { for e in &mut exp { e.kk = false; } } // 머리: 말줄임 글(.fr-ev overflow hidden)은 자르기 상자가 아니다
            let (extra, miss, ps) = pair(&act, &exp);
            assert!(extra.is_empty() && miss.is_empty(), "입력 {k} {nm}\n남은 rtui {}\n{}\n못 찾은 원본 {}\n{}", extra.len(), show(&extra), miss.len(), show(&miss));
            for (a, e) in ps { // svg 요소 수 1e-6
                for (x, y) in sa[a].iter().zip(&se[e]) { assert!(x.1.len() == y.1.len() && x.1.iter().zip(&y.1).all(|(p, q)| (p - q).abs() < 1e-6), "입력 {k} {nm} svg {}\n{:?}\n{:?}", x.0, &x.1[..x.1.len().min(12)], &y.1[..y.1.len().min(12)]); nums += y.1.len(); }
            }
            if let Some((w, h)) = wh { assert!((w - f(&dd["w"])).abs() < TOL && (h - f(&dd["h"])).abs() < TOL, "입력 {k} 머리 크기 {w}×{h} vs {}×{}", dd["w"], dd["h"]); }
            n += exp.len();
        }
        (n, nums)
    }

    /// 지금 화면의 실제 코드(compare/mt_js.mjs)가 지은 HTML 을 원본 styles.css 로 헤드리스 크로미움에서 잰 것(compare/mt_layout.py) -- 글·색·자리·svg 가 전부 하나씩 짝지어져야 한다
    #[test]
    fn js_parity() {
        // MT_JS = 변이 시험용 다른 잰 값(compare/mt_layout.py MT_OUT) · MT_HTMLJS = 같은 변이의 고른 지표(mt_html.json)
        let inp = fx("mt_in.json");
        let js = std::env::var("MT_JS").map(|p| serde_json::from_str::<Value>(&std::fs::read_to_string(p).unwrap()).unwrap()).unwrap_or_else(|_| fx("mt_js.json"));
        let html = std::env::var("MT_HTMLJS").map(|p| serde_json::from_str::<Value>(&std::fs::read_to_string(p).unwrap()).unwrap()).unwrap_or_else(|_| fx("mt_html.json"));
        let cases = js["cases"].as_array().unwrap();
        let all: Vec<&Value> = cases.iter().flat_map(|c| [&c["body"], &c["head"]]).collect();
        let m = lookup(&all, &js["probe"]);
        let (mut n, mut nums) = (0, 0);
        for (k, ((s, d), h)) in inp.as_array().unwrap().iter().zip(cases).zip(html.as_array().unwrap()).enumerate() { let (a, b) = check(k, s, d, &h[2], &m); n += a; nums += b; }
        assert!(n > 4000 && nums > 25_000, "{n} {nums}");
        eprintln!("mt js_parity: 단위 {n} · svg 수 {nums}");
    }

    /// JS toFixed(정확한 이진값 · 동점 올림) · 미국 시간대(일광 절약 경계) · ISO 읽기
    #[test]
    fn js_helpers() {
        for (x, d, e) in [(1.005, 2, "1.00"), (0.125, 2, "0.13"), (2.5, 0, "3"), (-1.25, 1, "-1.3"), (-0.04, 1, "-0.0"), (99.95, 1, "100.0"), (12.25, 1, "12.3"), (0.0, 2, "0.00"), (1234.5678, 0, "1235"), (9.995, 2, "9.99")] {
            assert_eq!(fixed(x, d), e, "{x} {d}");
        }
        // 2026-03-08(일) 02:00 CST = 08:00 UTC 부터 CDT · 2026-11-01(일) 02:00 CDT = 07:00 UTC 부터 CST
        let t = |y: i64, mo: i64, d: i64, h: i64, mi: i64| (days_civil(y, mo, d) * 86400 + h * 3600 + mi * 60) as f64;
        assert_eq!(us_local(t(2026, 3, 8, 7, 59), -6), (0, 60 + 59));
        assert_eq!(us_local(t(2026, 3, 8, 8, 0), -6), (0, 3 * 60));
        assert_eq!(us_local(t(2026, 11, 1, 6, 59), -6), (0, 60 + 59));
        assert_eq!(us_local(t(2026, 11, 1, 7, 0), -6), (0, 60));
        assert!(!cme_open(t(2026, 11, 20, 22, 30)), "금 16:30 CT 닫힘");
        assert!(cme_open(t(2026, 11, 22, 23, 0)), "일 17:00 CT 열림");
        assert!(!cme_open(t(2026, 11, 18, 22, 30)), "평일 16시 정비");
        assert!(vix_open(t(2026, 11, 18, 8, 15)) && !vix_open(t(2026, 11, 18, 8, 14)), "03:15 ET");
        assert_eq!(parse_iso("2026-10-15T12:30:00+00:00", -540), t(2026, 10, 15, 12, 30));
        assert_eq!(parse_iso("2026-10-15T12:30:00.500Z", -540), t(2026, 10, 15, 12, 30) + 0.5);
        assert_eq!(parse_iso("2026-10-15T21:30:00+09:00", 0), t(2026, 10, 15, 12, 30));
        assert_eq!(parse_iso("2026-10-15T21:30", -540), t(2026, 10, 15, 12, 30), "영역 없으면 지역 시각");
        assert_eq!(hm(t(2026, 11, 18, 15, 5), -540), "00:05");
    }

    /// 받기: 원본 주기(달력 6시간 · macro-trend 10분 · 오늘 1분 · 국채 5분 · xyz 1분) · 실패 = 1분 뒤 다시(macro-trend · 달력) · xyz 꼬리 합치기 · 숨으면 안 묻는다 · 누르기
    #[test]
    fn poll() {
        let gets = |o: &mut Vec<Req>| o.drain(..).map(|r| match r { Req::Get { kind, url } => (kind, url), _ => panic!() }).collect::<Vec<_>>();
        let mut s = Store::default();
        let mut o = vec![];
        s.tick(1e9, true, "", &mut o);
        assert!(o.is_empty(), "숨으면 안 묻는다");
        s.tick(1e9, false, "", &mut o);
        assert_eq!(gets(&mut o).into_iter().map(|q| q.1).collect::<Vec<_>>(), ["/api/macro-calendar", "/api/macro-trend", "/api/treasury-futures?h=24", "/api/macro-today", "/api/xyz100"]);
        s.on_get(Kind::MtTrend, None, 1e9 + 100.0);
        s.on_get(Kind::MtCal, Some(r#"{"x":1}"#), 1e9);
        assert_eq!(s.events, Some(serde_json::json!([])), "events 없으면 빈 목록");
        s.on_get(Kind::MtTreas, Some(r#"{"ok":true,"symbols":{}}"#), 1e9 + 300.0);
        s.on_get(Kind::MtXyz, Some(r#"{"ok":true,"now_sec":1000003,"bars":[[999000,1],[999060,2]],"live":{"px":2,"recv_sec":1000001}}"#), 1e9);
        assert_eq!((s.xyz.len(), s.skew), (2, 3.0));
        s.tick(1e9 + 30_000.0, false, "", &mut o);
        assert!(gets(&mut o).is_empty(), "30초 안엔 없음");
        s.tick(1e9 + 60_000.0, false, "", &mut o);
        assert_eq!(gets(&mut o).into_iter().map(|q| q.1).collect::<Vec<_>>(), ["/api/macro-trend", "/api/macro-today", "/api/xyz100?since=999000"], "macro-trend 실패 = 1분 뒤 · 국채는 5분");
        s.on_get(Kind::MtXyz, Some(r#"{"ok":true,"now_sec":1000063,"bars":[[999060,2.5],[999120,3]],"live":null}"#), 1e9 + 60_100.0);
        assert_eq!(s.xyz, vec![[999000.0, 1.0], [999060.0, 2.5], [999120.0, 3.0]], "꼬리 = 겹친 봉부터 바꿈");
        assert!(s.xyz_live.is_none());
        s.on_get(Kind::MtTrend, Some(r#"{"ok":false}"#), 1e9 + 60_200.0);
        assert_eq!(s.drawn, 1e9 + 60_200.0, "받으면 다시 짓는다(ok 아님도)");
        s.tick(1e9 + 120_000.0, false, "", &mut o);
        let g = gets(&mut o).into_iter().map(|q| q.1).collect::<Vec<_>>();
        assert!(!g.contains(&"/api/macro-trend".to_string()) && g.contains(&"/api/macro-today".to_string()), "{g:?}");
        s.tick(1e9 + 360_000.0, false, "", &mut o);
        assert!(gets(&mut o).iter().any(|q| q.1 == "/api/treasury-futures?h=24"), "국채 5분 넘으면 다시");
        s.on_get(Kind::MtToday, Some(r#"{"ok":true}"#), 1e9 + 360_100.0);
        assert_eq!((s.intra.as_ref().unwrap().0, s.drawn), (1e6 + 360.1, 1e9 + 360_100.0));
        s.rows = vec![(100.0, 129.0, "VIX".into())];
        s.left = 900.0;
        assert!(!s.click(950.0, 110.0, 1.0) && !s.click(10.0, 130.0, 1.0));
        assert!(s.click(10.0, 110.0, 5.0) && s.sel.as_deref() == Some("VIX") && s.drawn == 5.0);
        let mut h = Store { head: true, ..Default::default() };
        h.tick(1e9, false, "", &mut o);
        assert_eq!(gets(&mut o).into_iter().map(|q| q.1).collect::<Vec<_>>(), ["/api/macro-calendar", "/api/macro-trend"], "머리 요약 = 달력 + macro-trend 만");
    }
}
