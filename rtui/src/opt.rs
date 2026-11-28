//! Option 카드(app.js renderOptions 5373 · ?card=opt): 머리(수집 시각) · 지금(예상 폭) · 실현 감마 · 행사가 사다리(+닿음 등고선) · 거래 · 심리(블록 거래) · 심리 · 옵션 만기 · 옵션 순매수 흐름.
//! 자료 = 원본이 부르는 같은 주소·주기: /api/gex 60초(5457 refreshGex · 숨으면 안 묻는다) → 성공하면 ETH·BTC·XRP 는 /api/opt-smile?asset= · 실현 감마 = /api/market-history 5분봉(마감봉만) ·
//!   가격 = 허브 «모든 종목»(latestLivePriceByAsset · 없으면 지수). 원본은 /api/gex 가 올 때만 다시 그린다 -- 여기도 그때의 시각·가격으로 그린다(Io::at·px).
//! 자리 = 원본 styles.css(이 작업 시점)를 헤드리스 크로미움 1920(html.fit1 · 카드 폭 1838)에 띄워 잰 값(compare/opt_layout.py → fixture/opt_js.json · 시험 js_parity).
//!   격자 = 4열 1 : .8 : .65 : 1.15(사이 28) · 줄 = 머리 55 · cov · 지금|사다리|거래 · 실현 감마|사다리|심리 · 만기|흐름 -- 사다리가 두 줄보다 크면 남는 몫을 두 줄에 반씩.
//! 안 옮긴 것: 주식(CBOE · o.src) 판 · 사다리 «내 선»(청산가·지정가 -- /api/binance-account) · 툴팁(title) · «?» 설명 팝업·범위 칩 누르기(모양만) · 좁은 화면(≤720)·세로 화면 배치 ·
//!   fitLayout 의 사다리 높이·그림 배율 되먹임(사다리 높이 = 입력 lad · 배율 = ck).
//! ponytail: ::before 이름표(«지금» 등) 글자 사이 .08em 은 안 벌린다(왼쪽 끝·높이만 같다) · 칩 줄·흐름 머리 범례는 한 줄로만(넘쳐 꺾이는 폭이면 고친다).
use crate::fp::{Anchor, Clip, Col, Prim};
use serde_json::Value;
use std::collections::HashMap;

/// 글자 폭(글, 크기, 굵기, 고정폭)
pub type Measure<'a> = &'a dyn Fn(&str, f64, u16, bool) -> f64;
const OPT: Col = Col::Rgb(0x8f, 0xd0, 0xc8); // --option
const GEX_POLL_MS: f64 = 60_000.0; // 1599

/// 이 카드의 받아 둔 자료(feed::Feed::opt) -- on = 이 카드만 띄운다 · gex = /api/gex · smile = 코인 → /api/opt-smile · at·px = /api/gex 가 온 시각·그때 허브 가격(원본은 그때 다시 그린다)
#[derive(Default)]
pub struct Io { pub on: bool, pub gex: Option<Value>, pub smile: HashMap<String, Value>, pub at: f64, pub px: Option<f64>, pub asked: f64 }
impl Io { pub fn due(&mut self, now: f64) -> bool { if now - self.asked < GEX_POLL_MS { return false; } self.asked = now; true } }

impl crate::feed::Feed {
    /// 5462 응답: 받으면 그 코인 스마일을 묻는다(ETH·BTC·XRP) · 실패 = 받아 둔 값에 error 만 싣는다(처음이면 그대로 없음)
    pub fn on_gex(&mut self, body: Option<&str>, now: f64) {
        match body.and_then(|b| serde_json::from_str::<Value>(b).ok()) {
            Some(v) => {
                self.opt.gex = Some(v);
                if let Some(cur) = cur(&self.asset, self.opt.gex.as_ref()).filter(|c| ["ETH", "BTC", "XRP"].contains(&c.as_str())) {
                    self.out.push(crate::feed::Req::Get { kind: crate::feed::Kind::Smile, url: format!("{}/api/opt-smile?asset={}", self.urls.api, cur.to_lowercase()) });
                }
            }
            None => match self.opt.gex.as_mut() { Some(g) => g["error"] = "fetch_failed".into(), None => return },
        }
        (self.opt.at, self.opt.px) = (now, self.prices.get(&self.asset).copied());
        self.dirty = true;
    }
    /// 5470 스마일: 받으면 넣고 · 실패 = 없앤다(등고선만 빠진다)
    pub fn on_smile(&mut self, body: Option<&str>) {
        let Some(c) = cur(&self.asset, self.opt.gex.as_ref()) else { return };
        match body.and_then(|b| serde_json::from_str::<Value>(b).ok()) { Some(v) => { self.opt.smile.insert(c, v); } None => { self.opt.smile.remove(&c); } }
        self.dirty = true;
    }
}

/// 받아 둔 응답 · now = 그린 시각(Date.now) · tz = getTimezoneOffset(분) · price = 허브 가격 · candles = (봉 시작 초, 종가) · lad = 사다리 높이(fitOptLadderH) · ck = 아래 그림 배율(fitOptChartK)
pub struct In<'a> {
    pub asset: &'a str, pub now: f64, pub tz: f64, pub price: Option<f64>, pub gex: Option<&'a Value>, pub smile: &'a HashMap<String, Value>,
    pub candles: &'a [(i64, f64)], pub lad: f64, pub ck: f64,
}
/// 그릴 것: html = 카드 왼쪽 위 기준 · svgs = (svg 왼쪽 위, svg 안 그림) · h = 카드 높이
pub struct Out { pub html: Vec<Prim>, pub svgs: Vec<(f64, f64, Vec<Prim>)>, pub h: f64 }

// ── 숫자 글(JS 와 같게) ──
fn f(v: &Value) -> f64 { v.as_f64().unwrap_or(f64::NAN) }
fn num(v: &Value) -> Option<f64> { v.as_f64() }
fn fx(x: f64, d: usize) -> String { crate::dirs::fixed(x, d) }
fn r1(x: f64) -> f64 { fx(x, 1).parse().unwrap() }
fn jr(x: f64) -> f64 { (x + 0.5).floor() } // Math.round
fn lu(x: f64) -> f64 { (x * 64.0).floor() / 64.0 } // 크로미움 LayoutUnit
/// 5030 optQ
fn q(v: f64) -> String { if v >= 20.0 { fx(v, 0) } else if v >= 10.0 { fx(v, 1) } else if v >= 1.0 { fx(v, 2) } else if v >= 0.1 { fx(v, 3) } else { fx(v, (1.0 - v.log10().floor()).max(0.0) as usize) } }
/// 5036 optUsd(+ 2095 fmtUsdCompact)
fn usd(v: f64) -> String {
    if v >= 1e9 { format!("${}B", fx(v / 1e9, 2)) } else if v >= 1e6 { format!("${}M", fx(v / 1e6, if v >= 1e8 { 0 } else if v >= 1e7 { 1 } else { 2 })) } else if v >= 1e3 { format!("${}k", fx(v / 1e3, 1)) } else { format!("${}", fx(v, 0)) }
}
/// 5031 optKst
fn kst(ms: f64, date: bool) -> String {
    let s = ((ms + 9.0 * 3600e3) / 1000.0).floor() as i64;
    let (z, t) = (s.div_euclid(86400) + 719468, s.rem_euclid(86400));
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let (d, m) = (doy - (153 * mp + 2) / 5 + 1, if mp < 10 { mp + 3 } else { mp - 9 });
    format!("{}{:02}:{:02}", if date { format!("{m:02}-{d:02} ") } else { String::new() }, t / 3600, t % 3600 / 60)
}
fn sgn1(v: f64) -> String { format!("{}{}", if v >= 0.0 { "+" } else { "−" }, fx(v.abs(), 1)) }
fn pm(v: Option<f64>) -> String { v.map_or("-".into(), |v| format!("±{}$", q(v))) }
fn pct(v: Option<f64>) -> String { v.map_or("-".into(), |v| format!("{}%", fx(v * 100.0, if v < 0.1 { 1 } else { 0 }))) }

/// 5021 optData 의 cur(대문자 · 옵션 묶음에 없으면 None)
fn cur(asset: &str, gex: Option<&Value>) -> Option<String> {
    let k = asset.to_uppercase();
    let cs = gex.map(|g| &g["currencies"]).filter(|c| c.is_object());
    (cs.is_none() || cs.unwrap().get(&k).is_some()).then_some(k)
}
fn iv(o: &Value) -> Option<f64> { num(&o["dvol"]).or(num(&o["iv30"])) }
fn sigma(o: &Value, px: f64, sec: f64) -> Option<f64> { iv(o).filter(|v| *v != 0.0).map(|v| px * (v / 100.0) * (sec / 31536000.0).sqrt()) }
fn front<'a>(o: &'a Value, now: f64) -> Option<&'a Value> { o["expiries"].as_array()?.iter().find(|e| f(&e["exp_ms"]) > now) }

/// 4992~5007 RG_REF · RG_BY(분위 표 · 경계 · 배수 · 설명 키)
const RG: [(&str, [f64; 13], [f64; 2], [&str; 3], &str); 4] = [
    ("ETH", [-0.2952, -0.2068, -0.1671, -0.1185, -0.0866, -0.0587, -0.0321, -0.0061, 0.022, 0.0558, 0.1017, 0.1419, 0.2252], [-0.0769, 0.0124], ["0.94", "0.99", "1.02"], "rg"),
    ("XRP", [-0.2948, -0.2159, -0.1785, -0.1309, -0.0977, -0.0703, -0.0445, -0.0189, 0.0100, 0.0437, 0.0925, 0.1359, 0.2321], [-0.0881, 0.0000], ["0.91", "0.98", "1.02"], "rg_xrp"),
    ("BTC", [-0.2927, -0.2152, -0.1742, -0.1287, -0.0960, -0.0698, -0.0455, -0.0198, 0.0091, 0.0441, 0.0906, 0.1313, 0.2096], [-0.0870, -0.0010], ["0.90", "1.01", "1.01"], "rg_btc"),
    ("SOL", [-0.2643, -0.1876, -0.1519, -0.1083, -0.0774, -0.0520, -0.0283, -0.0046, 0.0210, 0.0502, 0.0940, 0.1356, 0.2168], [-0.0687, 0.0121], ["0.89", "0.93", "0.96"], "rg_sol")];
const RG_Q: [f64; 13] = [1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 99.0];
/// 5009 optRealizedGamma → (값, 분위, 쪽 0 되돌림 · 1 보통 · 2 추세)
fn rgamma(c: &[(i64, f64)], now_s: f64, refq: &[f64; 13], cut: [f64; 2]) -> Option<(f64, f64, usize)> {
    let cl: Vec<f64> = c.iter().filter(|q| q.0 as f64 + 300.0 <= now_s && q.1 > 0.0).map(|q| q.1.ln()).collect();
    if cl.len() < 145 { return None; }
    let cl = &cl[cl.len() - 145..];
    let r: Vec<f64> = cl.windows(2).map(|w| w[1] - w[0]).collect();
    let m = r.iter().fold(0.0, |a, b| a + b) / r.len() as f64;
    let x: Vec<f64> = r.iter().map(|v| v - m).collect();
    let (mut num, mut den) = (0.0, 0.0);
    for (i, v) in x.iter().enumerate() { den += v * v; if i > 0 { num += v * x[i - 1]; } }
    if !(den > 0.0) { return None; }
    let v = num / den;
    let p = match refq.iter().position(|q| *q >= v) { None => 99.5, Some(0) => 0.5, Some(h) => RG_Q[h - 1] + (RG_Q[h] - RG_Q[h - 1]) * (v - refq[h - 1]) / (refq[h] - refq[h - 1]) };
    Some((v, p, if v <= cut[0] { 0 } else if v >= cut[1] { 2 } else { 1 }))
}
fn rg_word(x: &str) -> &'static str { let x: f64 = x.parse().unwrap(); if x <= 0.97 { "예상보다 약간 작은 쪽" } else if x >= 1.02 { "예상보다 약간 큰 쪽" } else { "예상과 비슷" } }

/// 5109 optNcdf
fn ncdf(z: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let d = 0.3989423 * (-z * z / 2.0).exp();
    let q = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    if z > 0.0 { 1.0 - q } else { q }
}
/// 5114 optTouchModel → (만기, 닿을 확률 함수, 등고선 가격 함수)
struct Touch<'a> { exp: f64, s: &'a Value, px: f64, t: f64 }
impl Touch<'_> {
    fn new<'a>(s: Option<&'a Value>, px: f64, now: f64) -> Option<Touch<'a>> {
        let s = s.filter(|s| s["ok"] == true)?;
        let t = (f(&s["exp_ms"]) - now) / 31536e6;
        (t > 0.0 && px > 0.0).then_some(Touch { exp: f(&s["exp_ms"]), s, px, t })
    }
    fn touch(&self, p: f64) -> f64 {
        let (s, xl) = (self.s, f(&self.s["xlim"]));
        let x = (p / f(&s["S"])).ln().min(xl).max(-xl);
        let c = |k: usize| f(&s["coef"][k]);
        let v = (c(0) * x * x + c(1) * x + c(2)).max(0.05);
        let up = ncdf(((self.px / p).ln() - 0.5 * v * v * self.t) / (v * self.t.sqrt()));
        (2.0 * up.min(1.0 - up)).min(1.0)
    }
    fn at(&self, t: f64, mut a: f64, mut b: f64) -> f64 {
        for _ in 0..50 { let c = (a + b) / 2.0; if (self.touch(a) - t) * (self.touch(c) - t) <= 0.0 { b = c; } else { a = c; } }
        (a + b) / 2.0
    }
    fn level(&self, t: f64, above: bool) -> f64 { if above { self.at(t, self.px * 1.0001, self.px * 1.25) } else { self.at(t, self.px * 0.75, self.px * 0.9999) } }
}

// ── 그림 붓 ──
fn tx(x: f64, y: f64, s: String, size: f64, wt: u16, c: Col, a: f64, anchor: Anchor) -> Prim { Prim::Text { x, y, s, size, wt, c, a, anchor, halo: false, clip: Clip::No } }
fn rc(x: f64, y: f64, w: f64, h: f64, c: Col, a: f64, rx: f64) -> Prim { Prim::Rect { x, y, w, h, c, a, rx, clip: Clip::No } }
fn ln(pts: Vec<[f64; 2]>, c: Col, a: f64, lw: f64, dash: Option<[f64; 2]>) -> Prim {
    match dash { Some([on, off]) => Prim::Dashed { pts, c, a, lw, on, off, clip: Clip::No }, None => Prim::Line { pts, c, a, lw, dash: None, clip: Clip::No } }
}
/// 둥근 테(테두리 1px 상자 = 바깥 x·y·폭·높이 · 반지름 r) -- 닫힌 선
fn round_box(x: f64, y: f64, w: f64, h: f64, r: f64, c: Col, a: f64) -> Prim {
    let (x, y, w, h) = (x + 0.5, y + 0.5, w - 1.0, h - 1.0);
    let r = (r - 0.5).min(w / 2.0).min(h / 2.0);
    let mut pts = vec![];
    for (cx, cy, a0) in [(x + w - r, y + r, -90.0f64), (x + w - r, y + h - r, 0.0), (x + r, y + h - r, 90.0), (x + r, y + r, 180.0)] {
        for k in 0..=6 { let t = (a0 + 15.0 * k as f64).to_radians(); pts.push([cx + r * t.cos(), cy + r * t.sin()]); }
    }
    pts.push(pts[0]);
    Prim::Line { pts, c, a, lw: 1.0, dash: None, clip: Clip::No }
}

/// HTML 글자 쌓기(가운데 y → 바닥선 · 폭을 돌려준다)
struct Html<'a> { v: Vec<Prim>, m: Measure<'a> }
impl Html<'_> {
    fn t(&mut self, x: f64, cy: f64, s: &str, size: f64, wt: u16, c: Col, a: f64) -> f64 { self.put(x, cy, s.into(), size, wt, c, a, false) }
    /// font-variant-numeric: tabular-nums
    fn tn(&mut self, x: f64, cy: f64, s: &str, size: f64, wt: u16, c: Col, a: f64) -> f64 { self.put(x, cy, crate::fp::tnum(s), size, wt, c, a, false) }
    fn put(&mut self, x: f64, cy: f64, s: String, size: f64, wt: u16, c: Col, a: f64, mono: bool) -> f64 {
        let w = (self.m)(&s, size, wt, mono);
        self.v.push(tx(x, cy + size * 0.36, s, size, wt, c, a, Anchor::Start));
        w
    }
    /// .opt-q «제목 ?» 단추(위 = 단추 상자 위) → 오른쪽 끝
    fn q(&mut self, x: f64, top: f64, title: &str) -> f64 {
        let tw = self.t(x, top + 9.297, title, 12.0, 700, OPT, 1.0);
        let (cx, cy) = (x + tw + 6.0 + 7.5, top + 9.5);
        self.v.push(Prim::Line { pts: (0..=24).map(|k| { let t = k as f64 * std::f64::consts::TAU / 24.0; [cx + 7.0 * t.cos(), cy + 7.0 * t.sin()] }).collect(), c: OPT, a: 0.55 * 0.75, lw: 1.0, dash: None, clip: Clip::No });
        self.v.push(tx(cx, cy + 3.6, "?".into(), 10.0, 700, OPT, 0.75, Anchor::Middle));
        x + tw + 21.0
    }
    fn line(&mut self, x: f64, y: f64, w: f64, c: Col) { self.v.push(rc(x, y, w, 1.0, c, 1.0, 0.0)); }
    /// 글 조각 이어 쓰기(한 줄) → 끝 x
    fn run(&mut self, mut x: f64, cy: f64, parts: &[(String, u16, Col)], size: f64) -> f64 { for (s, wt, c) in parts { x += self.t(x, cy, s, size, *wt, *c, 1.0); } x }
}

/// 키-값 칸 한 줄 · 설명 줄
enum Kv { Row(String, String, Col), Note(String, Col) }
fn kv(k: &str, v: String) -> Kv { Kv::Row(k.into(), v, Col::Text) }
const SEC_TOP: f64 = 9.0; // 윗선 1 + 안쪽 8
const SEC_BOT: f64 = 12.0;
const LABEL: f64 = 19.0; // ::before 13 + 아래 6
const H4: f64 = 20.703125 + 6.0;
const NOTE: f64 = 15.390625; // 11px × 1.4
const LEGS: f64 = 17.390625; // 12px × 1.45
/// 칸 최소 높이: flex = «지금»·심리·실현 감마(ETH) 13.5px 줄(+4) · 아니면 12.5px 블록(설명 위 여백 3 은 제목 아래 6 에 겹친다)
fn kv_min(items: &[Kv], label: bool, flex: bool) -> f64 {
    let mut h = SEC_TOP + if label { LABEL } else { 0.0 } + H4 + SEC_BOT;
    for (k, it) in items.iter().enumerate() { h += match it { Kv::Row(..) => if flex { 20.0 } else { 19.0 }, Kv::Note(..) => NOTE + if flex || k > 0 { 3.0 } else { 0.0 } }; }
    h
}
/// 키-값 칸 그리기(칸 = x·위·폭·높이 · flex 면 줄이 남는 높이를 나눠 갖는다 -- 최대 mx)
fn kv_draw(b: &mut Html, x: f64, top: f64, w: f64, h: f64, label: Option<&str>, title: &str, items: &[Kv], flex: bool, mx: f64) {
    b.line(x, top, w, Col::Line);
    let mut y = top + SEC_TOP;
    if let Some(l) = label { b.t(x, y + 6.5, l, 11.0, 800, Col::Muted, 1.0); y += LABEL; }
    b.q(x, y + 1.703, title);
    y += H4;
    let rows = items.iter().filter(|i| matches!(i, Kv::Row(..))).count() as f64;
    let notes: f64 = items.iter().filter(|i| matches!(i, Kv::Note(..))).count() as f64 * (3.0 + NOTE);
    let kh = if flex { ((top + h - SEC_BOT - y - notes) / rows).clamp(20.0, mx.max(20.0)) } else { 19.0 };
    let z = if flex { 13.5 } else { 12.5 };
    for (k, it) in items.iter().enumerate() {
        match it {
            Kv::Row(key, val, c) => {
                b.t(x, y + kh / 2.0, key, z, 400, Col::Muted, 1.0);
                let s = crate::fp::tnum(val);
                let vw = (b.m)(&s, z, 700, false);
                b.put(x + w - vw, y + kh / 2.0, s, z, 700, *c, 1.0, false);
                y += kh;
            }
            Kv::Note(s, c) => { if flex || k > 0 { y += 3.0; } b.t(x, y + 7.695, s, 11.0, 400, *c, 0.85); y += NOTE; }
        }
    }
}

/// 5167 optLadderChips(범위 «7일 안» 고정 -- 지금 화면 기본값) → 칩 줄 그림
fn chips(b: &mut Html, x: f64, y: f64, legend: Option<String>) {
    let mut bx = x;
    for (t, on) in [("가까운 만기", false), ("7일 안", true), ("전 만기", false)] {
        let bw = (b.m)(t, 11.5, 700, false) + 22.0;
        if on { b.v.push(rc(bx, y, bw, 26.0, OPT, 0.12, 13.0)); }
        b.v.push(if on { round_box(bx, y, bw, 26.0, 13.0, OPT, 0.6) } else { round_box(bx, y, bw, 26.0, 13.0, Col::SoftLine, 1.0) });
        b.t(bx + 11.0, y + 12.25, t, 11.5, 700, if on { OPT } else { Col::Muted }, 1.0);
        bx += bw + 6.0;
    }
    if let Some(l) = legend { b.run(bx + 4.0, y + 13.0, &[("선".into(), 700, OPT), (l, 400, Col::Muted)], 11.0); }
}

/// 5046 optLaneSvg
fn lane_svg(o: &Value, w: f64, now: f64, k: f64) -> (f64, Vec<Prim>) {
    let narrow = w < 480.0;
    let h = if narrow { 170.0 } else { jr(190.0 * k) };
    let (x0, x1) = if narrow { (10.0, w - 10.0) } else { (20.0, w - 20.0) };
    let base = if narrow { 104.0 } else { jr(118.0 * k) };
    let fs = if narrow { 10.0 } else { 11.0 };
    let xa = |hh: f64| x0 + (hh / 120.0) * (x1 - x0);
    let ex: Vec<&Value> = o["expiries"].as_array().map(|a| a.iter().filter(|e| f(&e["exp_ms"]) > now && (f(&e["exp_ms"]) - now) / 3.6e6 <= 120.0).collect()).unwrap_or_default();
    let oi = |e: &Value| f(&e["call_oi_usd"]) + f(&e["put_oi_usd"]);
    let mx = ex.iter().fold(1.0f64, |a, e| a.max(oi(e)));
    let mut v = vec![];
    for hh in (0..=120).step_by(24) { let xx = xa(hh as f64); v.push(ln(vec![[xx, 8.0], [xx, h - 18.0]], Col::Line, 0.5, 1.0, None)); }
    v.push(ln(vec![[x0, base], [x1, base]], Col::Line, 1.0, 1.0, None));
    v.push(ln(vec![[xa(0.0), 6.0], [xa(0.0), h - 16.0]], Col::Ink, 0.55, 1.0, Some([3.0, 3.0])));
    let pcs = |e: &Value| num(&e["pc"]).map_or("-".into(), |p| fx(p, 2));
    for e in &ex {
        let val = oi(e);
        let hh = 10.0 + (if narrow { 50.0 } else { 62.0 * k }) * (val / mx).sqrt();
        let cx = xa((f(&e["exp_ms"]) - now) / 3.6e6);
        let ly = base - if narrow { hh } else { hh.max(40.0) };
        v.push(rc(r1(cx - 7.0), r1(base - hh), 14.0, r1(hh), Col::Amber, 0.85, 3.0));
        let lab = usd(val);
        if cx + 12.0 + (fs + 1.0) * 0.62 * lab.len() as f64 > w { v.push(tx(r1(cx - 10.0), r1(ly + 11.0), lab, fs + 1.0, 800, Col::Amber, 1.0, Anchor::End)); }
        else { v.push(tx(r1(cx + 12.0), r1(ly + 11.0), lab, fs + 1.0, 800, Col::Amber, 1.0, Anchor::Start)); }
        if !narrow {
            v.push(tx(r1(cx + 12.0), r1(ly + 25.0), format!("pain {}", q(f(&e["pain"]))), 10.5, 400, Col::Muted, 1.0, Anchor::Start));
            v.push(tx(r1(cx + 12.0), r1(ly + 39.0), format!("P/C {}", pcs(e)), 10.5, 400, Col::Muted, 1.0, Anchor::Start));
        }
        let d = kst(f(&e["exp_ms"]), true);
        v.push(tx(r1(cx), base + 14.0, if narrow { d[..5].into() } else { d }, if narrow { 9.5 } else { 10.5 }, 600, Col::Muted, 1.0, Anchor::Middle));
    }
    if ex.len() >= 2 {
        let ivs: Vec<f64> = ex.iter().map(|e| f(&e["atm_iv"])).collect();
        let (lo, hi) = (ivs.iter().fold(f64::INFINITY, |a, b| a.min(*b)) - 1.0, ivs.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b)) + 1.0);
        let (yt, yb) = (base + 26.0, h - 24.0);
        let yiv = |x: f64| yb - ((x - lo) / (hi - lo)) * (yb - yt);
        v.push(tx(x0 + 2.0, yb + 2.0, "ATM IV".into(), 10.5, 700, OPT, 1.0, Anchor::Start));
        v.push(ln(ex.iter().map(|e| [r1(xa((f(&e["exp_ms"]) - now) / 3.6e6)), r1(yiv(f(&e["atm_iv"])))]).collect(), OPT, 1.0, 2.0, None));
        for e in &ex {
            let (cx, y) = (xa((f(&e["exp_ms"]) - now) / 3.6e6), yiv(f(&e["atm_iv"])));
            v.push(Prim::Circle { x: r1(cx), y: r1(y), r: 3.5, c: OPT, a: 1.0, clip: Clip::No });
            v.push(tx(r1(cx + 7.0), r1(y - 5.0), fx(f(&e["atm_iv"]), 0), 10.5, 700, OPT, 1.0, Anchor::Start));
        }
    }
    for hh in (0..=120).step_by(24) {
        let a = if hh == 0 { Anchor::Start } else if hh == 120 { Anchor::End } else { Anchor::Middle };
        v.push(tx(xa(hh as f64), h - 2.0, if hh == 0 { "지금".into() } else { format!("+{hh}h") }, 10.5, if hh == 0 { 700 } else { 500 }, if hh == 0 { Col::Ink } else { Col::Muted }, 1.0, a));
    }
    (h, v)
}

/// 5171 optLadderSvg(+ 5136 optTouchBack) -- 내 선(5150 optTouchFront)은 안 옮김
fn ladder_svg(o: &Value, w: f64, hh: f64, px: f64, tm: Option<&Touch>, now: f64) -> Vec<Prim> {
    let rows = o["strikes"]["week"].as_array().map(Vec::as_slice).unwrap_or(&[]);
    let (lo, hi, y0, y1) = (px * 0.94, px * 1.06, 34.0, hh - 8.0);
    let ya = |p: f64| y1 - ((p - lo) / (hi - lo)) * (y1 - y0);
    let vis: Vec<[f64; 3]> = rows.iter().map(|r| [f(&r[0]), f(&r[1]), f(&r[2])]).filter(|r| r[0] >= lo && r[0] <= hi).collect();
    let mid = jr((w + 46.0) / 2.0);
    let half = mid - 46.0;
    let mx = vis.iter().fold(1.0f64, |a, r| a.max(r[1].max(r[2])));
    let ks: Vec<f64> = vis.iter().map(|r| r[0]).collect();
    let step = if ks.len() > 1 { ks.windows(2).map(|q| q[1] - q[0]).fold(f64::INFINITY, f64::min) } else { 25.0 };
    let bh = 2f64.max(22f64.min((y1 - y0) * step / (hi - lo) - 1.0));
    let fsz = 8f64.max(10.5f64.min(bh - 1.0));
    let mut by_sum = vis.clone();
    by_sum.sort_by(|a, b| (b[1] + b[2]).total_cmp(&(a[1] + a[2])));
    let top: Vec<f64> = by_sum.iter().take(3).map(|r| r[0]).collect();
    let mut v = vec![tx(mid - 6.0, 12.0, "← 풋".into(), 10.0, 700, Col::Bad, 1.0, Anchor::End), tx(mid + 6.0, 12.0, "콜 →".into(), 10.0, 700, Col::Good, 1.0, Anchor::Start),
                     ln(vec![[mid, y0 - 4.0], [mid, y1]], Col::Line, 1.0, 1.0, None)];
    let mut chip_ys = vec![];
    if let Some(m) = tm {
        let lv: Vec<(f64, f64, bool, f64, f64)> = [(0.5, 0.95, false), (0.25, 0.6, true), (0.1, 0.38, true)].iter().map(|&(t, op, d)| (t, op, d, m.level(t, true), m.level(t, false))).collect();
        let (ta, tb) = (ya(lv[0].3.min(hi)), ya(lv[0].4.max(lo)));
        v.push(rc(30.0, r1(ta), w - 34.0, r1(tb - ta), OPT, 0.07, 0.0));
        for &(t, op, d, a, b) in &lv {
            for p in [a, b].into_iter().filter(|p| *p >= lo && *p <= hi) {
                let txt = format!("{}% {}", jr(t * 100.0), if p < 10.0 { fx(p, 3) } else { q(p) });
                let (tw, y) = (txt.len() as f64 * 6.2 + 12.0, ya(p));
                v.push(ln(vec![[30.0, r1(y)], [w - 4.0, r1(y)]], OPT, op, 1.1, d.then_some([6.0, 4.0])));
                v.push(rc(r1(w - 4.0 - tw), r1(y - 8.0), r1(tw), 15.0, Col::ChartBg, 0.9, 7.5));
                v.push(Prim::Mono { x: w - 9.0, y: r1(y + 3.5), s: txt, size: 10.0, wt: 800, c: OPT, a: op.max(0.7), anchor: Anchor::End, halo: false, clip: Clip::No });
            }
        }
        chip_ys = lv.iter().flat_map(|q| [q.3, q.4]).filter(|p| *p >= lo && *p <= hi).map(ya).collect();
    }
    for &[k, c, p] in &vis {
        let (y, pw, cw) = (ya(k), half * p / mx, half * c / mx);
        v.push(rc(r1(mid - pw), r1(y - bh / 2.0), r1(pw), r1(bh), Col::Bad, 0.75, 0.0));
        v.push(rc(mid, r1(y - bh / 2.0), r1(cw), r1(bh), Col::Good, 0.75, 0.0));
        if top.contains(&k) {
            let lab = usd(c + p);
            let callw = c >= p;
            let bw = if callw { cw } else { pw };
            let fit = bw >= lab.len() as f64 * fsz * 0.62 + 8.0;
            let mut x = mid + if callw { if fit { cw - 4.0 } else { cw + 4.0 } } else if fit { -pw + 4.0 } else { -pw - 4.0 };
            let mut anc = if callw == fit { Anchor::End } else { Anchor::Start };
            if fit && chip_ys.iter().any(|cy| (cy - y).abs() < 10.0) { x = mid + if callw { 6.0 } else { -6.0 }; anc = if callw { Anchor::Start } else { Anchor::End }; }
            v.push(tx(r1(x), r1(y + fsz * 0.36), lab, r1(fsz), 800, if fit { Col::ChartBg } else { Col::Text }, 1.0, anc));
        }
    }
    let every = step * 1f64.max(jr((hi - lo) / step / 8.0));
    for &k in ks.iter().filter(|k| (*k / every - jr(*k / every)).abs() < 1e-6) { v.push(tx(0.0, r1(ya(k) + 3.5), q(k), 9.5, 400, Col::Muted, 1.0, Anchor::Start)); }
    v.push(ln(vec![[30.0, r1(ya(px))], [w - 4.0, r1(ya(px))]], Col::Ink, 0.8, 1.0, Some([4.0, 3.0])));
    v.push(tx(w - 6.0, r1(ya(px) - 4.0), format!("지금 {}", q(px)), 10.0, 700, Col::Ink, 1.0, Anchor::End));
    if let Some(fr) = front(o, now).filter(|e| f(&e["pain"]) >= lo && f(&e["pain"]) <= hi) {
        let pn = f(&fr["pain"]);
        v.push(ln(vec![[30.0, r1(ya(pn))], [w - 4.0, r1(ya(pn))]], Col::Amber, 1.0, 1.0, Some([2.0, 4.0])));
        let share = num(&fr["oi_share"]).map_or(String::new(), |s| format!(" · 미결제 {}%", jr(s * 100.0)));
        v.push(tx(32.0, r1(ya(pn) + if pn < px { 12.0 } else { -4.0 }), format!("max pain {}{share}", q(pn)), 9.5, 700, Col::Amber, 1.0, Anchor::Start));
    }
    v
}

/// 5223 optNewTxt
fn new_txt(bs: &[Value]) -> String {
    let ok: Vec<&Value> = bs.iter().filter(|b| !b["doi"].is_null()).collect();
    let vv = ok.iter().fold(0.0, |a, b| a + f(&b["cb"]) + f(&b["cs"]) + f(&b["pb"]) + f(&b["ps"]));
    let doi = ok.iter().fold(0.0, |a, b| a + f(&b["doi"]));
    if !(vv > 0.0) { return String::new(); }
    let r = ((vv + doi) / (2.0 * vv)).clamp(0.0, 1.0);
    format!(" · 미결제 {}{} · 신규 약 {}%", if doi >= 0.0 { "+" } else { "−" }, fx(doi.abs(), 0), jr(r * 100.0))
}
/// 5229 optFlowSvg
fn flow_svg(fl: &[Value], w: f64, k: f64, cur: &str) -> (f64, Vec<Prim>) {
    let narrow = w < 700.0;
    let k = if narrow { 1.0 } else { k };
    let h = if narrow { 150.0 } else { jr(170.0 * k) };
    let (x0, x1) = if narrow { (34.0, w - 8.0) } else { (60.0, w - 20.0) };
    let (base, amp) = if narrow { (84.0, 44.0) } else { (jr(94.0 * k), jr(52.0 * k)) };
    let n = fl.len() as f64;
    let xa = |i: f64| x0 + (i / n) * (x1 - x0);
    let bw = ((x1 - x0) / n) * 0.36;
    let nets: Vec<[f64; 2]> = fl.iter().map(|b| [f(&b["cb"]) - f(&b["cs"]), f(&b["pb"]) - f(&b["ps"])]).collect();
    let mx = nets.iter().flatten().fold(1e-9f64, |a, b| a.max(b.abs()));
    let mut acc = 0.0;
    let cum: Vec<f64> = fl.iter().map(|b| { acc += f(&b["dlt"]); acc }).collect();
    let dm = cum.iter().fold(1e-9f64, |a, b| a.max(b.abs()));
    let mut v = vec![ln(vec![[x0, base], [x1, base]], Col::Line, 1.0, 1.0, None), tx(x0 - 6.0, base - amp + 8.0, "순매수".into(), 9.5, 400, Col::Muted, 1.0, Anchor::End),
                     tx(x0 - 6.0, base + amp - 2.0, "순매도".into(), 9.5, 400, Col::Muted, 1.0, Anchor::End)];
    for (i, [c, p]) in nets.iter().enumerate() {
        for (val, col, dx) in [(*c, Col::Good, 0.0), (*p, Col::Bad, bw + 1.0)] {
            let hh = amp * val.abs() / mx;
            if hh < 0.5 { continue; }
            v.push(rc(r1(xa(i as f64) + 3.0 + dx), r1(if val >= 0.0 { base - hh } else { base }), r1(bw), r1(hh), col, if i as f64 == n - 1.0 { 0.45 } else { 0.8 }, 0.0));
        }
        if i % (if narrow { 6 } else { 4 }) == 0 && xa(i as f64) + 25.0 <= w { v.push(tx(r1(xa(i as f64) + 3.0), h - 2.0, format!("{}시", &kst(f(&fl[i]["h"]), false)[..2]), 9.5, 400, Col::Muted, 1.0, Anchor::Start)); }
    }
    v.push(ln(cum.iter().enumerate().map(|(i, c)| [r1(xa(i as f64 + 1.0)), r1(base - (amp * c) / dm)]).collect(), OPT, 1.0, 2.0, None));
    let last = *cum.last().unwrap();
    v.push(tx(x1, r1(base - (amp * last) / dm - 6.0), format!("체결 순델타 {}{} {cur}", if last >= 0.0 { "+" } else { "" }, fx(last, 0)), if narrow { 10.0 } else { 11.0 }, 700, OPT, 1.0, Anchor::End));
    (h, v)
}

/// 5262 optBlkShape · 5267 optBlkLegs → 한 줄 조각들(글 조각 묶음 · 앞에서 꺾일 수 있나)
fn blk_legs(b: &Value) -> Vec<Vec<(String, u16, Col)>> {
    let mut out = vec![];
    if let Some(st) = b["structure"].as_str().filter(|s| !s.is_empty()) {
        const SHAPE: [(&str, &str); 17] = [("single", "단일"), ("straddle", "스트래들"), ("strangle", "스트랭글"), ("risk_reversal", "리스크 리버설"), ("synthetic", "합성 선물"), ("vertical", "수직 스프레드"),
            ("ratio_spread", "비율 스프레드"), ("calendar", "캘린더"), ("diagonal", "대각"), ("butterfly", "버터플라이"), ("ladder", "래더"), ("condor", "콘도르"), ("iron_condor", "아이언 콘도르"),
            ("iron_butterfly", "아이언 버터플라이"), ("box", "박스"), ("other", "복합"), ("future_only", "선물")];
        let d = match b["delta_side"].as_str() { Some("long") => " · 델타 롱", Some("short") => " · 델타 숏", Some("neutral") => " · 델타 중립", _ => "" };
        let vs = match b["vol_side"].as_str() { Some("long_vol") => " · 베가 롱", Some("short_vol") => " · 베가 숏", Some("neutral") => " · 베가 중립", _ => "" };
        let nm = SHAPE.iter().find(|q| q.0 == st).map_or(st, |q| q.1);
        out.push(vec![(format!("{nm}{}{d}{vs}", if crate::srl::truthy(&b["hedge"]) { " + 선물 헤지" } else { "" }), 700, Col::Text), (" ".into(), 400, Col::Text)]);
    }
    const MON: [&str; 12] = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];
    for (j, l) in b["legs"].as_array().into_iter().flatten().enumerate() {
        let name = l["instrument_name"].as_str().map_or_else(|| crate::srl::js(&l["instrument_name"]), String::from);
        // /-(\d+)([A-Z]{3})\d+-([\dd]+)-([CP])$/
        let p: Vec<&str> = name.split('-').collect();
        let m = (p.len() >= 4).then(|| (p[p.len() - 3], p[p.len() - 2], p[p.len() - 1])).filter(|(dt, k, cp)| {
            let dd = dt.bytes().take_while(u8::is_ascii_digit).count();
            let rest = &dt[dd..];
            dd > 0 && rest.len() > 3 && rest.as_bytes()[..3].iter().all(u8::is_ascii_uppercase) && rest[3..].bytes().all(|c| c.is_ascii_digit()) && !k.is_empty() && k.bytes().all(|c| c.is_ascii_digit() || c == b'd') && (*cp == "C" || *cp == "P")
        });
        let k = match m {
            Some((dt, k, cp)) => { let dd = dt.bytes().take_while(u8::is_ascii_digit).count(); let mo = &dt[dd..dd + 3];
                format!("{}-{:0>2} {}{cp}", MON.iter().position(|x| *x == mo).map_or(mo.to_string(), |i| format!("{:02}", i + 1)), &dt[..dd], k.replace('d', ".")) }
            None => name,
        };
        let buy = l["direction"] == "buy";
        if j > 0 { out.last_mut().unwrap().push((" · ".into(), 400, Col::Text)); }
        out.push(vec![((if buy { "매수" } else { "매도" }).into(), 700, if buy { Col::Good } else { Col::Bad }), (format!(" {k} ×{}", fx(f(&l["amount"]), 0)), 400, Col::Text)]);
    }
    out
}
/// 블록 한 줄 다리 글 배치(왼쪽 lx · 폭 lw) → (조각 자리들, 줄 수). 꺾기 = 다리 앞(« · » 는 앞 줄 끝에 남는다 · 끝 빈칸은 넘쳐도 된다)
fn legs_layout(items: &[Vec<(String, u16, Col)>], lw: f64, m: Measure) -> (Vec<(f64, usize, String, u16, Col)>, usize) {
    let w = |s: &str, wt: u16| m(&crate::fp::tnum(s), 12.0, wt, false);
    let (mut x, mut line, mut out) = (0.0, 0usize, vec![]);
    for (k, parts) in items.iter().enumerate() {
        // 다리 몸(마지막 « · »·« » 빼고) 이 안 들어가면 새 줄
        let body: f64 = parts.iter().filter(|p| p.0 != " · " && p.0 != " ").map(|p| w(&p.0, p.1)).sum();
        if k > 0 && x > 0.0 && x + body > lw + 0.01 { x = 0.0; line += 1; }
        for (s, wt, c) in parts { out.push((x, line, s.clone(), *wt, *c)); x += w(s, *wt); }
    }
    (out, line + 1)
}

/// 한 장(w = 카드 폭) -- 원본 renderOptions
pub fn card(i: &In, w: f64, m: Measure) -> Out {
    let mut b = Html { v: vec![], m };
    let cw = w - 34.0;
    let (x1, y0) = (17.0, 72.0);
    // 머리(.ops-health-head 55 · 아래 선) · 제목 «Option» 17px · 옆 한 줄 = 수집 시각(5103 optHead · 5096 optAsOf)
    b.v.push(rc(17.0, 71.0, cw, 1.0, Col::SoftLine, 1.0, 0.0));
    let tw = b.t(35.0, 43.5, "Option", 17.0, 400, Col::Text, 1.0);
    let hx = 35.0 + tw + 26.0;
    let gex = i.gex;
    let cur = cur(i.asset, gex);
    let g = cur.as_ref().and_then(|c| gex.filter(|g| crate::srl::truthy(&g["available"])).map(|g| &g["currencies"][c.as_str()]));
    let o = g.map(|g| &g["options"]).filter(|o| o.is_object());
    let err = gex.and_then(|g| g.get("error")).filter(|e| !e.is_null()).map(|e| e.as_str().map_or_else(|| crate::srl::js(e), String::from));
    let note_row = |b: &mut Html, s: &str, c: Col| { b.t(17.0, y0 + 3.0 + 7.695, s, 11.0, 400, c, 0.85); };
    let Some(o) = o else {
        b.tn(hx, 42.5, "Deribit · 참고, 신호 아님", 12.0, 400, Col::Muted, 1.0);
        let s = if cur.is_none() { "이 코인은 옵션을 수집하지 않습니다 (Deribit ETH·BTC·SOL·XRP · CBOE 미국 상장 주식만)".to_string() } else if let Some(e) = &err { format!("옵션 수집 지연 ({e})") } else { "옵션 불러오는 중…".into() };
        note_row(&mut b, &s, Col::Muted);
        return Out { html: b.v, svgs: vec![], h: y0 + 3.0 + NOTE + 10.0 + 17.0 };
    };
    let now = i.now;
    let src = o["src"].as_str().unwrap_or("Deribit");
    let rec = o["recorded_at_utc"].as_str().and_then(crate::data::iso_ms);
    match rec {
        Some(t) => {
            let age = (now - t) / 60e3;
            let hm = { let s = ((t - i.tz * 60e3) / 1000.0).floor() as i64; format!("{:02}:{:02}", s.rem_euclid(86400) / 3600, s.rem_euclid(3600) / 60) };
            if age > 25.0 {
                let x = hx + b.tn(hx, 42.5, &format!("{src} · 참고, 신호 아님 · "), 12.0, 400, Col::Muted, 1.0);
                b.tn(x, 42.5, &format!("{hm} 기준({} 전 값)", if age >= 120.0 { format!("{}시간", jr(age / 60.0)) } else { format!("{}분", jr(age)) }), 12.0, 400, Col::Amber, 1.0);
            } else { b.tn(hx, 42.5, &format!("{src} · 참고, 신호 아님 · {hm} 기준"), 12.0, 400, Col::Muted, 1.0); }
        }
        None => { b.tn(hx, 42.5, &format!("{src} · 참고, 신호 아님"), 12.0, 400, Col::Muted, 1.0); }
    }
    if o["src"].is_string() { // 주식(CBOE) 판은 안 옮김
        note_row(&mut b, "주식 옵션 판(CBOE)은 rtui 에 아직 없음 -- 지금 화면에서 본다", Col::Muted);
        return Out { html: b.v, svgs: vec![], h: y0 + 3.0 + NOTE + 10.0 + 17.0 };
    }
    let cur = cur.unwrap();
    let px = i.price.filter(|p| *p != 0.0).unwrap_or(f(&o["index"]));
    let fr = front(o, now);
    let hrs = fr.map(|e| (f(&e["exp_ms"]) - now) / 3.6e6);
    let ivv = iv(o);
    // 예상 폭 + 5360 optInfoParts(move)
    let mut mv = vec![kv("24시간 1σ", sigma(o, px, 86400.0).map_or("-".into(), |s| format!("{} ({}%)", pm(Some(s)), fx(ivv.unwrap() / 365f64.sqrt(), 1)))), kv("이번 5분 1σ", pm(sigma(o, px, 300.0)))];
    if o["dvol"].is_null() && !o["iv30"].is_null() { mv.push(Kv::Note(format!("DVOL 지수가 없는 코인 — 30일 ATM IV {}% 로 계산", fx(f(&o["iv30"]), 0)), Col::Muted)); }
    mv.push(kv(&hrs.map_or("다음 만기까지".into(), |h| format!("다음 만기까지({}h)", fx(h, 0))), pm(hrs.filter(|h| *h > 0.0).and_then(|h| sigma(o, px, h * 3600.0)))));
    let vrp = ivv.zip(num(&o["rv7"])).map(|(a, r)| a - r);
    mv.push(kv("IV(30일) − 실현(7일)", vrp.map_or("-".into(), |v| format!("{}{}pt", if v >= 0.0 { "+" } else { "" }, fx(v, 1)))));
    let (sf, cm) = (&o["surface"], &o["cm"]);
    let atm = |d: &str| num(&cm[d]["atm"]);
    if let Some(v) = g.and_then(|g| num(&g["opt_hist"]["vov24"])) { mv.push(kv("VoV · 24h", format!("내재 변동성 시간당 ±{}%", fx(v, 1)))); }
    if crate::srl::truthy(&sf["front"]) && crate::srl::truthy(&sf["front"]["tail"]) {
        let t = &sf["front"]["tail"];
        mv.push(kv(&format!("꼬리 확률 · 만기까지({}h)", fx(f(&sf["front"]["hours"]), 0)), format!("±2% 밖 {} · ±3% {} · ±5% {}", pct(num(&t["2"])), pct(num(&t["3"])), pct(num(&t["5"])))));
    }
    if crate::srl::truthy(&sf["7"]) && crate::srl::truthy(&sf["7"]["tail"]) { mv.push(kv(&format!("꼬리 확률 · {}일 만기", jr(f(&sf["7"]["hours"]) / 24.0)), format!("±5% 밖 {}", pct(num(&sf["7"]["tail"]["5"]))))); }
    // 실현 감마
    let rgc = RG.iter().find(|r| r.0 == cur);
    let rg = rgc.and_then(|r| rgamma(i.candles, now / 1000.0, &r.1, r.2));
    let (gtip, gitems) = match (rgc, rg) {
        (None, _) => ("rg", vec![Kv::Note("ETH·XRP·BTC·SOL 만 검증 — 이 코인은 표시 안 함".into(), Col::Muted)]),
        (Some(r), None) => (r.4, vec![Kv::Note("5분봉 12시간치 대기".into(), Col::Muted)]),
        (Some(r), Some((v, p, side))) => (r.4, vec![
            kv("12시간 자기상관", format!("{}{} · 2021~24 중 {}%", if v >= 0.0 { "+" } else { "−" }, fx(v.abs(), 3), if p < 50.0 { format!("하위 {}", jr(p).max(1.0)) } else { format!("상위 {}", jr(100.0 - p).max(1.0)) })),
            Kv::Row("국면".into(), ["되돌림 쪽 · 롱감마형", "보통", "추세 쪽 · 숏감마형"][side].into(), [Col::Good, Col::Text, Col::Amber][side]),
            kv("다음 4시간 변동성", format!("{} · {}배", rg_word(r.3[side]), r.3[side]))]),
    };
    let gflex = gtip == "rg"; // styles.css 1919·1930 은 [data-tip="rg"] 만 -- 다른 코인 칸은 격자 자리 없이 첫 빈 칸으로 가고 블록 배치(원본 그대로)
    // 심리 + optInfoParts(mood)
    let cmf = |k: &str| { let v7 = num(&cm["7"][k]); v7.map_or("-".into(), |v| format!("{}pt{}", sgn1(v), num(&cm["30"][k]).map_or(String::new(), |w| format!(" (30일 {})", sgn1(w))))) };
    let term = if ["7", "30", "60"].iter().all(|d| atm(d).is_none()) { "-".into() } else { ["7", "30", "60"].iter().map(|d| format!("{d}일 {}", atm(d).map_or("-".into(), |v| fx(v, 0)))).collect::<Vec<_>>().join(" → ") };
    let mut md = vec![kv("25Δ 리스크 리버설 · 7일", cmf("rr")), kv("버터플라이 · 7일", cmf("bf")), kv("기간 구조 · ATM IV", term)];
    if let (Some(a1), Some(a7)) = (atm("1"), atm("7")) {
        md.push(Kv::Row("기간 구조 기울기".into(), format!("1일−7일 {}pt{}", sgn1(a1 - a7), atm("30").map_or(String::new(), |a30| format!(" · 7일−30일 {}pt", sgn1(a7 - a30)))), if a1 > a7 { Col::Amber } else { Col::Text }));
    }
    // 블록 거래(5275 optBlkHtml · 데스크톱 = 늘 펼침)
    let bt = &gex.unwrap()["block_trades"];
    let blocks: Vec<&Value> = bt["by_coin"][cur.as_str()].as_array().map(|a| a.iter().collect()).unwrap_or_default();
    let mut rows: Vec<&Value> = blocks.clone();
    rows.sort_by(|a, b| f(&b["ts_ms"]).total_cmp(&f(&a["ts_ms"])));
    rows.truncate(5);
    let bwarn = bt["available"] == false;
    // ── 격자(styles.css 1914 · 2026-10-05 배치) ──
    let inner = cw - 84.0;
    let fr_ = inner / 3.6;
    let (e1, e2, e3) = (lu(fr_), lu(fr_ * 1.8), lu(fr_ * 2.45));
    let (c1, c2, c3, c4) = (e1, e2 - e1, e3 - e2, inner - e3);
    let (xa, xb, xc, xd) = (x1, x1 + c1 + 28.0, x1 + c1 + 28.0 + c2 + 28.0, x1 + c1 + 28.0 + c2 + 28.0 + c3 + 28.0);
    let (wl, wf) = (c1 + 28.0 + c2, c3 + 28.0 + c4);
    let wlad = c2 + 28.0 + c3;
    let svg_w = 640f64.min(jr(wlad));
    // 블록 한 줄 = [시각 auto][10][금액 64][10][다리 1fr]
    let blk_rows: Vec<_> = rows.iter().map(|r| {
        let t = kst(f(&r["ts_ms"]), false);
        let tw = m(&crate::fp::tnum(&t), 12.0, 400, false);
        let (lay, n) = legs_layout(&blk_legs(r), c4 - tw - 84.0, m);
        (t, usd(num(&r["notional_usd"]).unwrap_or(0.0)), tw, lay, n)
    }).collect();
    let fl = bt["flow_by_coin"][cur.as_str()].as_array().map(Vec::as_slice).unwrap_or(&[]);
    let (lh, lane_v) = lane_svg(o, wl, now, i.ck);
    let flow = (!fl.is_empty()).then(|| flow_svg(fl, wf, i.ck, &cur));
    let st_rows = o["strikes"]["week"].as_array().is_some_and(|a| !a.is_empty());
    let touch = Touch::new(i.smile.get(&cur), px, now);
    // 칸 최소 높이
    let move_min = kv_min(&mv, true, true);
    let gamma_min = kv_min(&gitems, false, gflex);
    let mood_min = kv_min(&md, false, true);
    let lad_min = SEC_TOP + LABEL + H4 + 26.0 + 6.0 + if st_rows { i.lad } else { NOTE } + SEC_BOT;
    let blk_min = SEC_TOP + LABEL + 19.0 + if bwarn { 3.0 + NOTE } else { 0.0 } + 4.0 + 33.0 + blk_rows.iter().map(|r| 13.0 + r.4 as f64 * LEGS).sum::<f64>() + SEC_BOT;
    let lane_min = SEC_TOP + 19.0 + 2.0 + lh + SEC_BOT;
    let flow_min = flow.as_ref().map_or(SEC_TOP + SEC_BOT, |q| SEC_TOP + 19.0 + 2.0 + q.0 + SEC_BOT);
    // 줄: 2 = cov(지연 설명) 또는 격자 자리 없는 실현 감마 · 3 · 4 · 5
    let g_in2 = !gflex && err.is_none();
    let r2 = if err.is_some() { 3.0 + NOTE + 10.0 } else if g_in2 { gamma_min } else { 0.0 };
    let mut r3 = move_min.max(blk_min);
    let mut r4 = mood_min.max(if g_in2 { 0.0 } else { gamma_min });
    let extra = lad_min - r3 - r4;
    if extra > 0.0 { let a = lu(extra / 2.0); r3 += a; r4 += extra - a; }
    let r5 = lane_min.max(flow_min);
    let (t3, t4, t5) = (y0 + r2, y0 + r2 + r3, y0 + r2 + r3 + r4);
    if let Some(e) = &err { note_row(&mut b, &format!("옵션 갱신 지연({e}) — 마지막으로 받은 값"), Col::Amber); }
    kv_draw(&mut b, xa, t3, c1, r3, Some("지금"), "예상 폭", &mv, true, 46.0);
    let (gt, gh) = if g_in2 { (y0, r2) } else { (t4, r4) };
    kv_draw(&mut b, xa, gt, c1, gh, None, "실현 감마", &gitems, gflex, 46.0);
    kv_draw(&mut b, xd, t4, c4, r4, None, "심리", &md, true, f64::INFINITY);
    // 사다리
    let mut svgs = vec![];
    b.line(xb, t3, wlad, Col::Line);
    b.t(xb, t3 + SEC_TOP + 6.5, "가격대", 11.0, 800, Col::Muted, 1.0);
    b.q(xb, t3 + SEC_TOP + LABEL + 1.703, "행사가 사다리");
    let cy = t3 + SEC_TOP + LABEL + H4;
    chips(&mut b, xb, cy, touch.as_ref().map(|t| format!(" {} 만기 전 닿을 확률 50·25·10% · 옵션가 역산 · 참고", kst(t.exp, true))));
    if st_rows { svgs.push((xb, cy + 32.0, ladder_svg(o, svg_w, i.lad, px, touch.as_ref(), now))); }
    else { b.t(xb, cy + 32.0 + 7.695, "행사가 데이터 없음(수집기 다음 주기에 채워진다)", 11.0, 400, Col::Muted, 0.85); }
    // 거래 · 심리(블록)
    b.line(xd, t3, c4, Col::Line);
    b.t(xd, t3 + SEC_TOP + 6.5, "거래 · 심리", 11.0, 800, Col::Muted, 1.0);
    let mut y = t3 + SEC_TOP + LABEL;
    b.q(xd, y, "블록 거래 · 지난 24시간");
    y += 19.0;
    if bwarn { y += 3.0; b.t(xd, y + 7.695, "블록 수집 지연 — 아래 목록은 마지막으로 받은 값", 11.0, 400, Col::Amber, 0.85); y += NOTE; }
    y += 4.0;
    b.v.push(round_box(xd, y, c4, 33.0, 10.0, Col::SoftLine, 1.0));
    if blocks.is_empty() { let x = xd + 11.0 + b.tn(xd + 11.0, y + 16.5, "블록 24h ", 12.5, 400, Col::Text, 1.0); b.tn(x, y + 16.5, "0건", 12.5, 700, OPT, 1.0); }
    else {
        let tot = num(&bt["sum_by_coin"][cur.as_str()]).unwrap_or_else(|| blocks.iter().fold(0.0, |a, q| a + num(&q["notional_usd"]).unwrap_or(0.0)));
        let mxb = Some(&bt["max_by_coin"][cur.as_str()]).filter(|v| crate::srl::truthy(v)).unwrap_or_else(|| blocks.iter().copied().reduce(|a, q| if num(&q["notional_usd"]).unwrap_or(0.0) > num(&a["notional_usd"]).unwrap_or(0.0) { q } else { a }).unwrap());
        let n24 = Some(&bt["n_by_coin"][cur.as_str()]).filter(|v| !v.is_null()).map_or(blocks.len().to_string(), crate::srl::js);
        let mut x = xd + 11.0;
        x += b.tn(x, y + 16.5, "블록 24h ", 12.5, 400, Col::Text, 1.0);
        x += b.tn(x, y + 16.5, &format!("{n24}건 · {}", usd(tot)), 12.5, 700, OPT, 1.0);
        b.tn(x, y + 16.5, &format!(" · 최대 {} {}", usd(num(&mxb["notional_usd"]).unwrap_or(0.0)), kst(f(&mxb["ts_ms"]), false)), 12.5, 400, Col::Text, 1.0);
    }
    y += 33.0;
    for (t, u, tw, lay, n) in &blk_rows {
        b.line(xd, y, c4, Col::SoftLine);
        let c0 = y + 7.0 + LEGS / 2.0;
        b.tn(xd, c0, t, 12.0, 400, Col::Muted, 1.0);
        b.tn(xd + tw + 10.0, c0, u, 12.0, 800, OPT, 1.0);
        for (x, l, s, wt, c) in lay { if s.trim().is_empty() { continue; } b.tn(xd + tw + 84.0 + x, c0 + *l as f64 * LEGS, s, 12.0, *wt, *c, 1.0); }
        y += 13.0 + *n as f64 * LEGS;
    }
    // 아래 줄: 만기 · 흐름
    let head = |b: &mut Html, x: f64, top: f64, title: &str, legend: &[(String, u16, Col)]| { let r = b.q(x, top, title); b.run(r + 12.0, top + 9.297, legend, 11.5); };
    b.line(xa, t5, wl, Col::Line);
    head(&mut b, xa, t5 + SEC_TOP, "옵션 만기 · 지금 ~ +120h", &[("막대".into(), 700, Col::Amber), (" 만기 규모(명목) · pain · P/C(미결제) · ".into(), 400, Col::Muted), ("선".into(), 700, OPT), (" 만기별 ATM IV · KST".into(), 400, Col::Muted)]);
    svgs.push((xa, t5 + 30.0, lane_v));
    b.line(xc, t5, wf, Col::Line);
    if let Some((_, fv)) = flow {
        let liq: f64 = fl.iter().map(|q| num(&q["liq"]).unwrap_or(0.0)).sum();
        let tail = format!(" 체결 순델타(24h) 누적 · 테이커 기준{}{}", new_txt(&fl[..fl.len() - 1]).replace(" · 미결제", " · 24h 미결제"), if fl.iter().any(|q| f(&q["liq"]) > 0.0) { format!(" · 강제청산 {}", fx(liq, 0)) } else { String::new() });
        head(&mut b, xc, t5 + SEC_TOP, "옵션 순매수 흐름 · 지난 24시간", &[("콜".into(), 700, Col::Good), (" · ".into(), 400, Col::Muted), ("풋".into(), 700, Col::Bad), (format!(" 매수−매도({cur}) · "), 400, Col::Muted), ("선".into(), 700, OPT), (tail, 400, Col::Muted)]);
        svgs.push((xc, t5 + 30.0, fv));
    }
    Out { html: b.v, svgs, h: t5 + r5 + 17.0 }
}

#[cfg(test)]
mod t {
    use super::*;
    use crate::fc::t::{It, lookup, pair};

    fn fx_(f: &str) -> Value { serde_json::from_str(&std::fs::read_to_string(format!("{}/compare/fixture/{f}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap() }
    /// 잰 값 한 장 → 비교 단위(fc::t::expected 꼴 -- 불투명(opacity)은 색 알파에 곱한다)
    fn expected(d: &Value) -> (Vec<It>, Vec<(String, f64, u16, bool, f64)>) {
        let mut d = d.clone();
        for k in ["texts", "lines"] { for t in d[k].as_array_mut().unwrap() { let op = t["op"].as_f64().unwrap_or(1.0); let a = t["c"][3].as_f64().unwrap(); t["c"][3] = (a * op).into(); } }
        crate::fc::t::expected(&d)
    }
    /// rtui 그림 → 비교 단위(fc::t::actual -- 닫힌 둥근 테(폭 ≠ 높이)는 테두리 상자 'o')
    fn actual(v: &[Prim], m: Measure) -> Vec<It> {
        let mut out = crate::fc::t::actual(&[(0.0, v.iter().filter(|p| !is_box(p)).map(clone).collect())], m);
        for p in v.iter().filter(|p| is_box(p)) {
            let Prim::Line { pts, c, a, .. } = p else { unreachable!() };
            let (x0, x1, y0, y1) = pts.iter().fold((f64::MAX, f64::MIN, f64::MAX, f64::MIN), |q, p| (q.0.min(p[0]), q.1.max(p[0]), q.2.min(p[1]), q.3.max(p[1])));
            let q = crate::draw::rgba(*c, *a);
            out.push(It { k: 'o', r: [x0 - 0.5, y0 - 0.5, x1 - x0 + 1.0, y1 - y0 + 1.0], c: [q.r(), q.g(), q.b(), q.a()].map(|v| v as f64), s: String::new(), size: 0.0, wt: 0, mono: false });
        }
        out
    }
    fn is_box(p: &Prim) -> bool { matches!(p, Prim::Line { pts, .. } if pts.len() > 26 && pts[0] == pts[pts.len() - 1]) }
    fn clone(p: &Prim) -> Prim {
        match p {
            Prim::Rect { x, y, w, h, c, a, rx, clip } => Prim::Rect { x: *x, y: *y, w: *w, h: *h, c: *c, a: *a, rx: *rx, clip: *clip },
            Prim::Line { pts, c, a, lw, dash, clip } => Prim::Line { pts: pts.clone(), c: *c, a: *a, lw: *lw, dash: *dash, clip: *clip },
            Prim::Text { x, y, s, size, wt, c, a, anchor, halo, clip } => Prim::Text { x: *x, y: *y, s: s.clone(), size: *size, wt: *wt, c: *c, a: *a, anchor: *anchor, halo: *halo, clip: *clip },
            _ => unreachable!(),
        }
    }
    /// svg 요소 하나 = (종류, 수들, 글, 색 rgb, 알파)
    type El = (String, Vec<f64>, String, [u8; 3], f64);
    fn var(s: &str) -> Col {
        match s { "var(--warn)" => Col::Amber, "var(--option)" => OPT, "var(--muted)" => Col::Muted, "var(--ink)" => Col::Ink, "var(--line)" => Col::Line, "var(--good)" => Col::Good,
                  "var(--bad)" => Col::Bad, "var(--chart-bg)" => Col::ChartBg, "var(--text)" => Col::Text, _ => panic!("색 {s}") }
    }
    fn rgb(c: Col) -> [u8; 3] { let q = crate::draw::rgba(c, 1.0); [q.r(), q.g(), q.b()] }
    fn svg_exp(g: &Value) -> Vec<El> {
        let mut out = vec![];
        for e in g["els"].as_array().unwrap() {
            let a = &e["a"];
            let n = |k: &str, d: f64| a[k].as_str().map_or(d, |s| s.parse().unwrap());
            let s = |k: &str| a[k].as_str().unwrap_or("");
            match e["tag"].as_str().unwrap() {
                "rect" if s("fill") == "transparent" => {}
                "rect" => out.push(("rect".into(), vec![n("x", 0.0), n("y", 0.0), n("width", 0.0), n("height", 0.0), n("rx", 0.0)], String::new(), rgb(var(s("fill"))), n("fill-opacity", 1.0))),
                "line" => out.push(("poly".into(), vec![n("x1", 0.0), n("y1", 0.0), n("x2", 0.0), n("y2", 0.0), n("stroke-width", 1.0), dash(s("stroke-dasharray"))[0], dash(s("stroke-dasharray"))[1]], String::new(), rgb(var(s("stroke"))), n("stroke-opacity", 1.0))),
                "path" => {
                    let mut v: Vec<f64> = s("d").replace(['M', 'L'], " ").split_whitespace().map(|x| x.parse().unwrap()).collect();
                    v.extend([n("stroke-width", 1.0), 0.0, 0.0]);
                    out.push(("poly".into(), v, String::new(), rgb(var(s("stroke"))), n("stroke-opacity", 1.0)));
                }
                "circle" => out.push(("circle".into(), vec![n("cx", 0.0), n("cy", 0.0), n("r", 0.0)], String::new(), rgb(var(s("fill"))), n("fill-opacity", 1.0))),
                "text" => {
                    let anc = match s("text-anchor") { "end" => 2.0, "middle" => 1.0, _ => 0.0 };
                    out.push(("text".into(), vec![n("x", 0.0), n("y", 0.0), n("font-size", 16.0), n("font-weight", 400.0), anc, if s("font-family").contains("mono") { 1.0 } else { 0.0 }],
                              e["s"].as_str().unwrap().into(), rgb(var(s("fill"))), n("fill-opacity", 1.0)));
                }
                t => panic!("svg 요소 {t}"),
            }
        }
        out
    }
    fn dash(s: &str) -> [f64; 2] { let v: Vec<f64> = s.split_whitespace().map(|x| x.parse().unwrap()).collect(); if v.len() == 2 { [v[0], v[1]] } else { [0.0, 0.0] } }
    fn svg_act(v: &[Prim]) -> Vec<El> {
        v.iter().map(|p| match p {
            &Prim::Rect { x, y, w, h, c, a, rx, .. } => ("rect".into(), vec![x, y, w, h, rx], String::new(), rgb(c), a),
            Prim::Line { pts, c, a, lw, .. } => ("poly".into(), pts.iter().flatten().copied().chain([*lw, 0.0, 0.0]).collect(), String::new(), rgb(*c), *a),
            Prim::Dashed { pts, c, a, lw, on, off, .. } => ("poly".into(), pts.iter().flatten().copied().chain([*lw, *on, *off]).collect(), String::new(), rgb(*c), *a),
            &Prim::Circle { x, y, r, c, a, .. } => ("circle".into(), vec![x, y, r], String::new(), rgb(c), a),
            Prim::Text { x, y, s, size, wt, c, a, anchor, .. } | Prim::Mono { x, y, s, size, wt, c, a, anchor, .. } => ("text".into(),
                vec![*x, *y, *size, *wt as f64, match anchor { Anchor::Start => 0.0, Anchor::Middle => 1.0, Anchor::End => 2.0 }, if matches!(p, Prim::Mono { .. }) { 1.0 } else { 0.0 }], s.clone(), rgb(*c), *a),
            _ => panic!("svg 에 안 쓰는 붓"),
        }).collect()
    }
    fn el_same(a: &El, e: &El) -> bool { a.0 == e.0 && a.2 == e.2 && a.3 == e.3 && (a.4 - e.4).abs() < 1e-6 && a.1.len() == e.1.len() && a.1.iter().zip(&e.1).all(|(x, y)| (x - y).abs() < 1e-6) }

    pub(crate) fn input<'a>(s: &'a Value, smile: &'a HashMap<String, Value>, c: &'a [(i64, f64)], d: &Value) -> In<'a> {
        In { asset: s["asset"].as_str().unwrap(), now: f(&s["now"]), tz: -540.0, price: s["price"].as_f64(), gex: Some(&s["gex"]).filter(|g| !g.is_null()), smile, candles: c, lad: f(&d["lad"]), ck: f(&d["ck"]) }
    }
    /// 한 장 맞대기 → (남은 rtui, 못 찾은 원본, svg 남은 rtui, svg 못 찾은 원본, 높이)
    pub(crate) fn check(s: &Value, d: &Value, top: &Value) -> (Vec<It>, Vec<It>, Vec<El>, Vec<El>, f64) {
        let (exp, widths) = expected(d);
        let lk = lookup(&widths);
        // « · » 는 줄 끝(끝 빈칸 빠진 폭)·줄 가운데 둘이 잰 표에 있다 -- 가운데 쪽(넓은 것) · « »(공백만 마디)는 표에 없다: 잰 자리 차 3
        let wmax = |s: &str, z: f64, wt: u16| widths.iter().filter(|q| q.0 == s && q.1 == z && q.2 == wt).map(|q| q.4).fold(f64::NAN, f64::max);
        let m = move |s: &str, z: f64, wt: u16, mono: bool| if s == " " { 3.0 } else if s == " · " && !mono { wmax(s, z, wt) } else { lk(s, z, wt, mono) };
        let smile: HashMap<String, Value> = s["smile"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let c: Vec<(i64, f64)> = s["candles"].as_array().unwrap().iter().map(|q| (q["time"].as_i64().unwrap(), f(&q["close"]))).collect();
        let o = card(&input(s, &smile, &c, top), f(&top["w"]), &m);
        let (a, b) = pair(&actual(&o.html, &m), &exp);
        let es = d["svgs"].as_array().unwrap();
        assert_eq!(o.svgs.len(), es.len(), "svg 수");
        let (mut sa, mut sb) = (vec![], vec![]);
        for ((x, y, v), g) in o.svgs.iter().zip(es) {
            assert!((x - f(&g["r"][0])).abs() < 0.06 && (y - f(&g["r"][1])).abs() < 0.06, "svg 자리 {x} {y} vs {}", g["r"]);
            let (act, mut exp) = (svg_act(v), svg_exp(g));
            for e in act { match exp.iter().position(|q| el_same(&e, q)) { Some(k) => { exp.remove(k); } None => sa.push(e) } }
            sb.extend(exp);
        }
        (a, b, sa, sb, o.h)
    }

    /// 지금 화면의 실제 함수(compare/opt_js.mjs 가 자른 것)를 원본 styles.css 로 헤드리스 크로미움에서 돌려 잰 것(compare/opt_layout.py) --
    /// HTML 글·상자·선은 하나씩 짝(fc 와 같은 허용) · svg 요소는 전부 1e-6 · 카드 높이 ±.02
    #[test]
    fn js_parity() {
        let (inp, js) = (fx_("opt_in.json"), std::env::var("OPT_JS").map(|p| serde_json::from_str::<Value>(&std::fs::read_to_string(p).unwrap()).unwrap()).unwrap_or_else(|_| fx_("opt_js.json")));
        let mut n = 0;
        for (k, (s, d)) in inp.as_array().unwrap().iter().zip(js["cases"].as_array().unwrap()).enumerate() {
            let (extra, miss, sa, sb, h) = check(s, d, &js);
            let ln = |v: &[It]| v.iter().map(|q| format!("{} {:?} {:?} {:?} {} {}{}", q.k, q.s, q.r.map(|x| (x * 1000.0).round() / 1000.0), q.c, q.size, q.wt, if q.mono { " M" } else { "" })).collect::<Vec<_>>().join("\n");
            let ls = |v: &[El]| v.iter().map(|q| format!("{q:?}")).collect::<Vec<_>>().join("\n");
            assert!(extra.is_empty() && miss.is_empty() && sa.is_empty() && sb.is_empty(), "입력 {k}\n남은 rtui\n{}\n못 찾은 원본\n{}\nsvg 남은 rtui\n{}\nsvg 못 찾은 원본\n{}", ln(&extra), ln(&miss), ls(&sa), ls(&sb));
            assert!((h - f(&d["h"])).abs() < 0.02, "입력 {k} 높이 {h} vs {}", d["h"]);
            n += ["texts", "boxes", "lines"].iter().map(|q| d[q].as_array().unwrap().len()).sum::<usize>() + d["svgs"].as_array().unwrap().iter().map(|g| g["els"].as_array().unwrap().len()).sum::<usize>();
        }
        assert!(n > 900, "{n}");
    }

    /// 60초 · 숨으면 안 묻고 소켓 닫음 · 허브는 «모든 종목» 가격만 · 5분봉 이력 · 받으면 그 코인 스마일(ETH·BTC·XRP 만) · 실패 = error 만 싣는다
    #[test]
    fn poll() {
        use crate::feed::{Feed, Kind, Req, Source, Urls};
        let gets = |f: &mut Feed| { let v: Vec<(Kind, String)> = f.out.drain(..).filter_map(|r| match r { Req::Get { kind, url } => Some((kind, url)), _ => None }).collect(); v };
        let mut f = Feed::new("eth", "ETHUSDT", Source::Push, Urls::default(), false);
        f.opt.on = true;
        f.tick(1e6, false, false);
        assert!(matches!(&f.out[..2], [Req::WsOpen(_), Req::WsSend(s)] if s == r#"{"subscribe":[]}"#), "{:?}", f.out);
        let g = gets(&mut f);
        assert_eq!(g[0], (Kind::Gex, "/api/gex".into()));
        assert!(g.iter().any(|q| q.0 == Kind::History), "{g:?}");
        f.tick(1e6 + 30e3, false, false);
        assert!(!gets(&mut f).iter().any(|q| q.0 == Kind::Gex));
        f.tick(1e6 + 60e3, true, false);
        assert_eq!(f.out.drain(..).collect::<Vec<_>>(), vec![Req::WsClose], "숨으면 안 묻는다");
        f.tick(1e6 + 61e3, false, false);
        assert!(gets(&mut f).iter().any(|q| q.0 == Kind::Gex));
        f.on_gex(None, 1.0);
        assert!(f.opt.gex.is_none(), "처음 실패 = 그대로 없음");
        f.on_gex(Some(r#"{"available":true,"currencies":{"ETH":{}}}"#), 2.0);
        assert_eq!(gets(&mut f), vec![(Kind::Smile, "/api/opt-smile?asset=eth".into())]);
        f.on_gex(Some("<html>"), 3.0);
        assert_eq!(f.opt.gex.as_ref().unwrap()["error"], "fetch_failed");
        f.on_smile(Some(r#"{"ok":true}"#));
        assert!(f.opt.smile.contains_key("ETH"));
        f.on_smile(None);
        assert!(f.opt.smile.is_empty());
        let mut s = Feed::new("sol", "SOLUSDT", Source::Push, Urls::default(), false);
        s.on_gex(Some(r#"{"available":true,"currencies":{"SOL":{}}}"#), 1.0);
        assert!(gets(&mut s).is_empty(), "SOL 은 스마일 없음");
    }
}
