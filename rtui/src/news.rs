//! News 카드 + ⑤ 다음 24시간(?card=news) -- app.js 3326~3840 뉴스 구역(NEWS_* · newsTiltHtml · newsAssetCounts · newsAgo · renderNews) +
//!   renderMarketCtx 의 ⑤ 몫(6446~6625 gCur·ef·us · macro24 · events · whenHtml) · mcGfx.timeline(6240) · mcUsSession(6374) · optFront · optQ.
//! 자료 = 원본과 같은 주소·주기: /api/news 30초 · /api/market-context?asset= 1초(흐름 코인만) · /api/gex 1분 · /api/macro-calendar 6시간(실패면 1분 뒤) · 숨으면 안 묻는다.
//! 자리 = 원본 styles.css(html.fit1 · 1920 에서 #newsBody 1838×917)를 헤드리스 크로미움에서 잰 값과 같게 손으로 놓은 것(compare/news_js.mjs → news_layout.py → 시험 js_parity).
//! 누르기: «관련만»·«코인+매크로» 칩(원본 newsView) · 기사 제목 = 새 탭으로 기사(원본 <a target=_blank> · http(s) 만) · 목록 위 휠 = 목록 스크롤(원본 .nw-list overflow-y).
//! 안 옮긴 것: 카드 머리(제목 · #newsMeta 건수 줄) · 툴팁(title) · 제목 밑줄·초점 테 · «?» 설명 팝업(단추 모양만) · 왼쪽 요약 스크롤(넘치면 잘림) · 좁은 화면(720 이하 · 목록 520 이하 두 줄 모양 · ⑤ 600 미만 목록) · 레일 단추.
//! ponytail: ⑤ 는 market-context 가 «available» 일 때 받은 값을 계속 쓰고 1초마다 지금 시각으로 다시 그린다(원본은 끊기면 마지막 그림에서 멈춤).
//! ponytail: 상태는 이 파일의 thread_local(웹어셈블리 = 한 스레드) -- 공용 Feed 구조체를 안 건드리려고. 카드가 늘어 공용 자리가 생기면 그리로.
use crate::fp::{Anchor, Clip, Col, Prim};
use crate::mt::{asc, brk, cont, fixed, hm, parse_iso, rr_pts, us_local, Measure};
use crate::srl::truthy;
use serde_json::Value;
use std::cell::RefCell;

/// --option(청록)
const OPT: Col = Col::Rgb(0x8f, 0xd0, 0xc8);
const NEWS_ASSETS: [&str; 6] = ["ETH", "BTC", "SOL", "XRP", "HYPE", "macro"];
fn src_short(s: &str) -> &str { match s { "theblock" => "Block", "cointelegraph" => "CT", "coindesk" => "CD", "tree" => "Tree", "bwenews" => "BWE", "financialjuice" => "FJ", "trumpstruth" => "Truth", "fed" => "Fed", "sec" => "SEC", _ => s } }
fn news_asset(a: &str) -> &str { match a { "macro" => "매크로", "other" => "기타", _ => a } }

// ── JS 수·글 ──
fn f(v: &Value) -> f64 { v.as_f64().unwrap_or(f64::NAN) }
/// Number(v) || 0
fn n0(v: &Value) -> f64 { v.as_f64().filter(|x| x.is_finite() && *x != 0.0).unwrap_or(0.0) }
/// v ?? 0
fn nz(v: &Value) -> f64 { v.as_f64().unwrap_or(0.0) }
fn s<'a>(v: &'a Value) -> &'a str { v.as_str().unwrap_or("") }
fn jround(x: f64) -> f64 { (x + 0.5).floor() }
fn fx(x: f64, d: usize) -> f64 { fixed(x, d).parse().unwrap_or(f64::NAN) }
/// String.prototype.slice(0, n) -- UTF-16 단위
fn slice16(s: &str, n: usize) -> String { String::from_utf16_lossy(&s.encode_utf16().take(n).collect::<Vec<_>>()) }
/// LayoutUnit(1/64 px) 내림·반올림 -- 크로미움 flex 남는 높이 나누기(반올림, 앞에서부터) · align-content: space-evenly(내림)
fn floor64(x: f64) -> f64 { (x * 64.0).floor() / 64.0 }
fn round64(x: f64) -> f64 { (x * 64.0).round() / 64.0 }
/// 3726 newsAgo
fn ago(ms: f64, now: f64) -> String {
    let m = jround((now - ms) / 60000.0).max(0.0) as i64;
    if m < 60 { format!("{m}분") } else { format!("{}시간{}", m / 60, if m % 60 != 0 { format!(" {}분", m % 60) } else { String::new() }) }
}
/// toLocaleString("en-US", 소수 0) -- 쉼표 · 반올림(0.5 는 0 에서 먼 쪽)
fn n_int(v: &Value) -> String {
    let Some(x) = v.as_f64().filter(|x| x.is_finite()) else { return "-".into() };
    let r = (x.abs() + 0.5).floor() as u64;
    let d = r.to_string();
    let mut o = String::new();
    for (k, c) in d.chars().enumerate() { if k > 0 && (d.len() - k) % 3 == 0 { o.push(','); } o.push(c); }
    if x < 0.0 && r > 0 { format!("-{o}") } else { o }
}
/// 5030 optQ
fn opt_q(v: &Value) -> String {
    let Some(x) = v.as_f64() else { return "-".into() };
    if x >= 20.0 { fixed(x, 0) } else if x >= 10.0 { fixed(x, 1) } else if x >= 1.0 { fixed(x, 2) } else if x >= 0.1 { fixed(x, 3) }
    else if x == 0.0 { "0.0".into() } else { fixed(x, (1 - x.abs().log10().floor() as i64).max(0) as usize) } // toPrecision(2)
}

// ── 그리기 단위(시험이 원본 잰 값과 하나씩 맞댄다) ──
/// k = 자르기 상자(overflow 조상들의 겹침) [x, y, 폭, 높이]
pub type K = Option<[f64; 4]>;
#[derive(Debug, Clone)]
pub enum It {
    /// y = 글 상자 위 · ls = 자간
    T { x: f64, y: f64, s: String, z: f64, wt: u16, ls: f64, c: Col, a: f64, k: K },
    B { x: f64, y: f64, w: f64, h: f64, c: Col, a: f64, rad: f64, k: K },
    /// 네 변 테(바깥 상자 · 두께)
    O { x: f64, y: f64, w: f64, h: f64, lw: f64, c: Col, a: f64, rad: f64, k: K },
    /// svg: 상자 · viewBox 폭·높이(상자에 고르게 늘임) · 요소(viewBox 단위)
    S { x: f64, y: f64, w: f64, h: f64, vb: [f64; 2], k: K, els: Vec<El> },
}
#[derive(Debug, Clone, PartialEq)]
pub enum El {
    /// 굵기 1
    Line { p: [f64; 4], c: Col, a: f64 },
    /// y = 바닥선
    Text { x: f64, y: f64, s: String, z: f64, wt: u16, anchor: Anchor, c: Col, a: f64 },
    /// ring = 주황 테 2(해킹·장애)
    Circle { x: f64, y: f64, r: f64, c: Col, a: f64, ring: bool },
}
fn and(a: K, b: [f64; 4]) -> K {
    let Some(a) = a else { return Some(b) };
    let (x0, y0) = (a[0].max(b[0]), a[1].max(b[1]));
    Some([x0, y0, (a[0] + a[2]).min(b[0] + b[2]) - x0, (a[1] + a[3]).min(b[1] + b[3]) - y0])
}
/// 줄 상자 위 → 글 상자 위(반 행간 내림 · 줄 높이 normal = 글 높이)
fn gt(top: f64, lh: f64, z: f64) -> f64 { top + ((lh - cont(z, false)) / 2.0).floor() }

struct Cx<'a> { m: Measure<'a>, v: Vec<It> }
impl Cx<'_> {
    fn w(&self, s: &str, z: f64, wt: u16, ls: f64) -> f64 { (self.m)(s, z, wt, false) + ls * s.chars().count() as f64 }
    #[allow(clippy::too_many_arguments)]
    fn t(&mut self, x: f64, y: f64, s: impl Into<String>, z: f64, wt: u16, ls: f64, c: Col, k: K) { self.v.push(It::T { x, y, s: s.into(), z, wt, ls, c, a: 1.0, k }); }
    #[allow(clippy::too_many_arguments)]
    fn b(&mut self, x: f64, y: f64, w: f64, h: f64, c: Col, a: f64, rad: f64, k: K) { self.v.push(It::B { x, y, w, h, c, a, rad, k }); }
    /// 말줄임: 넘치면(force = line-clamp 마지막 줄이라 늘) 들어가는 만큼(글 + «…» ≤ max) -- (보이는 글, 잘렸나)
    fn cut(&self, s: &str, z: f64, wt: u16, max: f64, force: bool) -> (String, bool) {
        if !force && self.w(s, z, wt, 0.0) <= max { return (s.into(), false); }
        let cs: Vec<char> = s.chars().collect();
        let e = self.w("…", z, wt, 0.0);
        let n = (0..cs.len()).rev().find(|&n| self.w(&cs[..n].iter().collect::<String>(), z, wt, 0.0) + e <= max).unwrap_or(0);
        (cs[..n].iter().collect(), true)
    }
    /// 줄바꿈(한 꼴 · 빈칸 뒤 · 한글 사이 -- mt::brk) → 줄들(줄 끝 빈칸 뺌)
    fn wrap(&self, s: &str, z: f64, wt: u16, width: f64) -> Vec<String> {
        let ch: Vec<char> = s.chars().collect();
        let w = |a: usize, b: usize| { let mut b = b; while b > a && ch[b - 1] == ' ' { b -= 1; } self.w(&ch[a..b].iter().collect::<String>(), z, wt, 0.0) };
        let (mut out, mut a) = (vec![], 0);
        while a < ch.len() {
            let mut end = ch.len();
            if w(a, end) > width {
                let mut best = None;
                for q in a + 1..ch.len() { if brk(ch[q - 1], ch[q]) { if w(a, q) <= width { best = Some(q); } else { break; } } }
                end = best.unwrap_or_else(|| (a + 1..ch.len()).find(|&q| brk(ch[q - 1], ch[q])).unwrap_or(ch.len()));
            }
            out.push(ch[a..end].iter().collect::<String>().trim_end().to_string());
            a = end;
            while a < ch.len() && ch[a] == ' ' { a += 1; }
        }
        out
    }
}

// ── 입력 ──
/// 받아 둔 응답(원본 전역 latestNews · latestMarketCtx · latestGex · latestMacroEvents · newsView) · now = Date.now()(ms) · tz = getTimezoneOffset
pub struct In<'a> {
    pub now: f64, pub tz: i64, pub asset: &'a str,
    pub news: Option<&'a Value>, pub mc: Option<&'a Value>, pub gex: Option<&'a Value>, pub events: Option<&'a Value>,
    pub rel_only: bool, pub coin_only: bool,
    /// 목록 스크롤(px)
    pub scroll: f64,
}
/// 한 장 결과: 그림 · 칩 자리(0 = 관련만 · 1 = 코인+매크로) · 기사 링크 자리 · 목록 상자 · 목록 내용 높이
#[derive(Default)]
pub struct Out { pub items: Vec<It>, pub chips: Vec<[f64; 4]>, pub links: Vec<([f64; 4], String)>, pub list: [f64; 4], pub list_h: f64 }

fn rel(it: &Value) -> bool { nz(&it["relevant"]) >= 0.5 }
fn srcs(it: &Value) -> Vec<String> {
    match it["sources"].as_array().filter(|a| !a.is_empty()) { Some(a) => a.iter().map(|x| s(x).to_string()).collect(), None => vec![s(&it["source"]).to_string()] }
}
fn sen_col(k: &str) -> Col { match k { "bullish" => Col::Good, "bearish" => Col::Bad, _ => Col::Muted } }

/// 다음 24시간 일정 [(ms, 이름, hi = 원본 e.hi: None = undefined)] -- 6446~6483
fn events(i: &In) -> Vec<(f64, String, Option<bool>)> {
    let Some(d) = i.mc else { return vec![] };
    let (a, now) = (i.asset, i.now);
    let mut v: Vec<(f64, String, Option<bool>)> = vec![];
    let fnext = &d["funding"]["next_ms"];
    if truthy(fnext) { v.push((f(fnext), "펀딩 정산".into(), None)); }
    let g = i.gex.filter(|g| truthy(&g["available"])).map(|g| &g["currencies"][a.to_uppercase()]).filter(|c| truthy(c));
    if let Some(o) = g.map(|c| &c["options"]).filter(|o| truthy(o)) {
        if let Some(e) = o["expiries"].as_array().into_iter().flatten().find(|e| f(&e["exp_ms"]) > now) {
            v.push((f(&e["exp_ms"]), format!("옵션 만기 · pain {}", if a == "eth" { n_int(&e["pain"]) } else { opt_q(&e["pain"]) }), None));
        }
    }
    if let Some((t, live)) = us_session(now) { v.push((t, if live { "미국장 마감" } else { "미국장 개장" }.into(), None)); }
    let day = |t: f64| ((t - i.tz as f64 * 60_000.0) / 86_400_000.0).floor();
    let d0 = day(now);
    for e in i.events.and_then(|e| e.as_array()).into_iter().flatten() {
        let t = parse_iso(s(&e["time_utc"]), i.tz) * 1000.0;
        let nm = [&e["title_ko"], &e["title"]].iter().find(|x| truthy(x)).map_or("지표".into(), |x| s(x).to_string());
        if t > now && (d0..d0 + 2.0).contains(&day(t)) { v.push((t, nm, Some(e["importance"] == "high"))); }
    }
    let mut acc: Vec<(f64, String, Option<bool>)> = vec![];
    for e in v {
        match acc.iter_mut().find(|q| (q.0 - e.0).abs() < 60e3) {
            Some(q) => { q.1 = format!("{} · {}", q.1, e.1); q.2 = if q.2 == Some(false) && e.2 == Some(false) { Some(false) } else if q.2 == Some(true) { Some(true) } else { e.2 }; }
            None => acc.push(e),
        }
    }
    acc
}
/// 6374 mcUsSession → (다음 시각, 장중인가)
fn us_session(now: f64) -> Option<(f64, bool)> {
    let d0 = (now / 86_400_000.0).floor();
    for k in -1..5 {
        for h in [13.0, 14.0] {
            let open = (d0 + k as f64) * 86_400_000.0 + h * 3_600_000.0 + 30.0 * 60_000.0;
            let (w, m) = us_local(open / 1000.0, -5);
            if !((1..=5).contains(&w) && m == 570) { continue; }
            let close = open + 6.5 * 3600e3;
            if now < close { let live = now >= open; return Some((if live { close } else { open }, live)); }
        }
    }
    None
}
/// 6240 mcGfx.timeline(events, w, 48) 요소(w ≥ 600)
fn timeline(i: &In, w: f64) -> Vec<El> {
    let now = i.now;
    let x2 = |hr: f64| 12.0 + (w - 24.0) * hr.min(24.0) / 24.0;
    let mut v = vec![El::Line { p: [12.0, 22.0, w - 12.0, 22.0], c: Col::Line, a: 1.0 }];
    for q in [0.0, 6.0, 12.0, 18.0, 24.0] {
        let anchor = if q == 0.0 { Anchor::Start } else if q == 24.0 { Anchor::End } else { Anchor::Middle };
        v.push(El::Text { x: x2(q), y: 46.0, s: if q == 0.0 { "지금".into() } else { format!("+{q}h") }, z: 10.0, wt: 400, anchor, c: Col::Muted, a: 1.0 });
    }
    let mut ev: Vec<_> = events(i).into_iter().filter(|e| e.0 > now).collect();
    ev.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut last = [-1e9, -1e9];
    for (k, e) in ev.iter().enumerate() {
        let hr = (e.0 - now) / 3.6e6;
        let (x, far, row) = (x2(hr), hr > 24.0, k % 2);
        let lab = format!("{} {}{}", e.1, hm(e.0 / 1000.0, i.tz), if far { " >" } else { "" });
        let c = if e.2 == Some(false) { Col::Muted } else { Col::Amber };
        let tw = lab.encode_utf16().count() as f64 * 10.5 * 0.62;
        let anchor = if x > w - 90.0 { Anchor::End } else if x < 90.0 { Anchor::Start } else { Anchor::Middle };
        let l0 = match anchor { Anchor::End => x - tw, Anchor::Start => x, Anchor::Middle => x - tw / 2.0 };
        v.push(El::Circle { x: fx(x, 1), y: 22.0, r: 5.0, c, a: if far { 0.45 } else { 1.0 }, ring: false });
        if l0 > last[row] + 8.0 { v.push(El::Text { x: fx(x, 1), y: if row == 1 { 38.0 } else { 12.0 }, s: lab, z: 10.5, wt: 400, anchor, c: Col::Text, a: 1.0 }); last[row] = l0 + tw; }
    }
    v
}

/// 3763 renderNews(+ ⑤ #mcWhen 이 .nw-main 끝) → #newsBody(폭 w · 높이 h) 안 그림 -- 응답이 ok 가 아니면 카드가 숨는다(빈 그림)
pub fn body(i: &In, w: f64, h: f64, m: Measure) -> Out {
    let mut out = Out::default();
    let Some(p) = i.news.filter(|p| p["ok"] == true) else { return out };
    let mut cx = Cx { m, v: vec![] };
    let now = if truthy(&p["now_ms"]) { f(&p["now_ms"]) } else { i.now };
    let coin = i.asset.to_uppercase();
    let empty = vec![];
    let items = p["items"].as_array().unwrap_or(&empty);
    let judged: Vec<&Value> = items.iter().filter(|it| truthy(&it["sentiment"])).collect();
    let wgt = |it: &Value| n0(&it["impact"]) * nz(&it["relevant"]);

    // 머리 칩(.nw-head: 오른쪽 붙임 · gap 6 · 칩 높이 26 · 글 11.5/1 굵기 600)
    let labs = ["관련만".to_string(), format!("{coin}+매크로")];
    let cw: Vec<f64> = labs.iter().map(|l| cx.w(l, 11.5, 700, 0.0) + 22.0).collect();
    let xs = [w - cw[1] - 6.0 - cw[0], w - cw[1]];
    for k in 0..2 {
        let on = if k == 0 { i.rel_only } else { i.coin_only };
        let (x, cwk) = (xs[k], cw[k]);
        if on { cx.b(x, 4.0, cwk, 26.0, OPT, 0.12, 999.0, None); }
        cx.v.push(It::O { x, y: 4.0, w: cwk, h: 26.0, lw: 1.0, c: if on { OPT } else { Col::SoftLine }, a: if on { 0.6 } else { 1.0 }, rad: 999.0, k: None });
        cx.t(x + 11.0, 4.0 + (26.0 - 11.5) / 2.0 + ((11.5 - cont(11.5, false)) / 2.0).floor(), labs[k].clone(), 11.5, 700, 0.0, if on { OPT } else { Col::Muted }, None);
        out.chips.push([x, 4.0, cwk, 26.0]);
    }

    // ── 왼쪽 요약(.nw-sum 380 · 세로 flex gap 10 · 남는 높이를 기울기·코인·고영향이 나눠 갖는다) ──
    let (sx, sw, top0) = (0.0, 380.0, 38.0);
    let ks: K = Some([sx, top0, sw, h - top0]);
    // 기울기 1·6·12시간
    let tilt = |hh: f64| {
        let xs: Vec<&&Value> = judged.iter().filter(|it| f(&it["ts_ms"]) >= now - hh * 3600e3 && rel(it)).collect();
        let mut sum = [0.0; 3];
        for it in &xs { if let Some(k) = ["bullish", "neutral", "bearish"].iter().position(|k| it["sentiment"] == *k) { let v = n0(&it["impact"]) * nz(&it["relevant"]); sum[k] += if v != 0.0 && !v.is_nan() { v } else { 0.01 }; } }
        let tot = sum[0] + sum[1] + sum[2];
        let pc = |k: usize| if tot != 0.0 { 100.0 * sum[k] / tot } else { 0.0 };
        let cnt = |k: &str| xs.iter().filter(|it| it["sentiment"] == k).count();
        (xs.len(), [pc(0), pc(1), pc(2)], [cnt("bullish"), cnt("neutral"), cnt("bearish")], tot != 0.0)
    };
    // 고영향 상위 6
    let mut top: Vec<&Value> = judged.iter().copied().filter(|it| rel(it)).collect();
    top.sort_by(|a, b| wgt(b).total_cmp(&wgt(a)));
    top.truncate(6);
    const LH: f64 = 16.796875; // line-height 1.4 × 12(LayoutUnit)
    let tops: Vec<(Vec<String>, f64)> = top.iter().map(|it| {
        let tt = slice16(&s(&it["title"]).split_whitespace().collect::<Vec<_>>().join(" "), 120);
        let ls = cx.wrap(tt.trim(), 12.0, 400, 308.0);
        let r1 = (ls.len().min(2) as f64 * LH).max(14.0);
        (ls, 1.0 + 5.0 + r1 + 1.0 + 13.0 + 5.0)
    }).collect();
    // 출처 줄(flex wrap · gap 4 10)
    let mut by: Vec<(String, usize)> = vec![];
    for it in items { for s0 in srcs(it) { match by.iter_mut().find(|q| q.0 == s0) { Some(q) => q.1 += 1, None => by.push((s0, 1)) } } }
    by.sort_by(|a, b| b.1.cmp(&a.1));
    let merged = items.iter().filter(|it| srcs(it).len() > 1).count();
    let mut spans: Vec<(String, String)> = by.iter().map(|(k, n)| (format!("{} ", src_short(k)), n.to_string())).collect();
    if merged > 0 { spans.push(("· 여러 출처 합침 ".into(), merged.to_string())); }
    let mut lines = 1;
    let mut place = vec![];
    { let mut x = 0.0; for (a, b) in &spans { let w0 = cx.w(a, 11.5, 400, 0.0); let ww = w0 + cx.w(b, 11.5, 700, 0.0); if x > 0.0 && x + ww > sw { x = 0.0; lines += 1; } place.push((x, lines - 1, w0)); x += ww + 10.0; } }
    let srcs_h = lines as f64 * 14.0 + (lines - 1) as f64 * 4.0;
    let nat = [15.0 + 97.0, 115.0, if top.is_empty() { 0.0 } else { 15.0 + tops.iter().map(|t| t.1).sum::<f64>() }];
    let ngrow = if top.is_empty() { 2 } else { 3 };
    let mut rem = (h - top0 - nat.iter().sum::<f64>() - srcs_h - ngrow as f64 * 10.0).max(0.0);
    let mut ex = [0.0; 3];
    for (k, e) in ex.iter_mut().enumerate().take(ngrow) { *e = round64(rem / (ngrow - k) as f64); rem -= *e; }
    let sech = |cx: &mut Cx, y: f64, s0: &str| cx.t(sx, y, s0, 11.0, 800, 0.88, Col::Muted, ks);
    // ① 기울기
    let mut y = top0;
    sech(&mut cx, y, "기울기 · 영향×관련성 가중");
    let slot = floor64(ex[0] / 4.0);
    for (j, hh) in [1.0, 6.0, 12.0].into_iter().enumerate() {
        let (n, pc, cn, any) = tilt(hh);
        let ty = y + 15.0 + slot + j as f64 * (27.0 + 8.0 + slot);
        cx.t(sx, ty, format!("{hh}시간 · {n}건"), 12.0, 800, 0.0, Col::Text, ks);
        let (a, b, c) = (format!("호재 {}", cn[0]), format!(" {}% · 중립 {} · {}% ", fixed(pc[0], 0), cn[1], fixed(pc[2], 0)), format!("악재 {}", cn[2]));
        let (wa, wb, wc) = (cx.w(&a, 12.0, 800, 0.0), cx.w(&b, 12.0, 400, 0.0), cx.w(&c, 12.0, 800, 0.0));
        let x0 = sx + sw - wa - wb - wc;
        cx.t(x0, ty, a, 12.0, 800, 0.0, Col::Good, ks);
        cx.t(x0 + wa, ty, b, 12.0, 400, 0.0, Col::Muted, ks);
        cx.t(x0 + wa + wb, ty, c, 12.0, 800, 0.0, Col::Bad, ks);
        let by0 = ty + 14.0 + 5.0;
        cx.b(sx, by0, sw, 8.0, Col::SoftLine, 1.0, 999.0, ks);
        if any {
            let kb = and(ks, [sx, by0, sw, 8.0]);
            let mut x = sx;
            for (k, (c, a)) in [(Col::Good, 1.0), (Col::Muted, 0.45), (Col::Bad, 1.0)].into_iter().enumerate() { let ww = sw * pc[k] / 100.0; cx.b(x, by0, ww, 8.0, c, a, 0.0, kb); x += ww; }
        }
    }
    y += nat[0] + ex[0] + 10.0;
    // ② 코인별 12시간(격자 auto 1fr auto auto · gap 3 10 · 가운데 맞춤)
    let rows: Vec<(&str, usize, usize, usize, f64)> = NEWS_ASSETS.iter().map(|&a| {
        let xs: Vec<&&Value> = judged.iter().filter(|it| it["asset"] == a && rel(it)).collect();
        let c = |k: &str| xs.iter().filter(|it| it["sentiment"] == k).count();
        (a, c("bullish"), c("bearish"), xs.len(), xs.iter().map(|it| n0(&it["impact"])).fold(0.0, f64::max))
    }).collect();
    let lsh = 0.42; // .h letter-spacing .04em × 10.5
    let ms: Vec<String> = rows.iter().map(|r| if r.3 > 0 { fixed(r.4, 1) } else { "–".into() }).collect();
    let c1 = rows.iter().map(|r| cx.w(news_asset(r.0), 12.0, 800, 0.0)).fold(cx.w("코인", 10.5, 400, lsh), f64::max);
    let c3 = rows.iter().map(|r| cx.w(&format!("{}건", r.3), 12.0, 400, 0.0)).fold(cx.w("12시간", 10.5, 400, lsh), f64::max);
    let c4 = ms.iter().map(|m| cx.w(m, 12.0, 700, 0.0)).fold(cx.w("최대 영향", 10.5, 700, lsh), f64::max);
    let c2 = sw - c1 - c3 - c4 - 30.0;
    let (x2, x3, x4) = (sx + c1 + 10.0, sx + c1 + c2 + 20.0, sx + sw - c4);
    let slot = floor64(ex[1] / 8.0);
    let hy = y + slot;
    cx.t(sx, hy, "코인", 10.5, 400, lsh, Col::Muted, ks);
    cx.t(x2, hy, "호재 · 악재", 10.5, 400, lsh, Col::Muted, ks);
    let r12 = cx.w("12시간", 10.5, 400, lsh);
    cx.t(x3 + c3 - r12, hy, "12시간", 10.5, 400, lsh, Col::Muted, ks);
    let rmx = cx.w("최대 영향", 10.5, 700, lsh);
    cx.t(x4 + c4 - rmx, hy, "최대 영향", 10.5, 700, lsh, Col::Muted, ks);
    for (j, (r, mm)) in rows.iter().zip(&ms).enumerate() {
        let ry = hy + 13.0 + (j + 1) as f64 * (3.0 + slot) + j as f64 * 14.0;
        cx.t(sx, ry, news_asset(r.0), 12.0, 800, 0.0, if r.0 == coin { OPT } else { Col::Text }, ks);
        cx.b(x2, ry + 4.0, c2, 6.0, Col::SoftLine, 1.0, 3.0, ks);
        if r.3 > 0 {
            let km = and(ks, [x2, ry + 4.0, c2, 6.0]);
            let n = r.3 as f64;
            cx.b(x2, ry + 4.0, c2 * r.1 as f64 / n, 6.0, Col::Good, 1.0, 0.0, km);
            cx.b(x2 + c2 * (n - r.2 as f64) / n, ry + 4.0, c2 * r.2 as f64 / n, 6.0, Col::Bad, 1.0, 0.0, km);
        }
        let cs = format!("{}건", r.3);
        let wc = cx.w(&cs, 12.0, 400, 0.0);
        cx.t(x3 + c3 - wc, ry, cs, 12.0, 400, 0.0, Col::Muted, ks);
        let wm = cx.w(mm, 12.0, 700, 0.0);
        cx.t(x4 + c4 - wm, ry, mm.clone(), 12.0, 700, 0.0, Col::Text, ks);
    }
    y += nat[1] + ex[1] + 10.0;
    // ③ 고영향 상위(격자 30 26 1fr · "s i t" ". . a" · gap 1 8 · 제목 두 줄에서 자름)
    if !top.is_empty() {
        sech(&mut cx, y, "고영향 상위 · 12시간");
        let slot = floor64(ex[2] / (top.len() + 1) as f64);
        let mut ry = y + 15.0 + slot;
        for (it, (ls, rh)) in top.iter().zip(&tops) {
            cx.b(sx, ry, sw, 1.0, Col::SoftLine, 1.0, 0.0, ks);
            let c = ry + 6.0;
            let st = s(&it["sentiment"]);
            let (sc, sl) = match st { "bullish" => (Col::Good, "호재"), "bearish" => (Col::Bad, "악재"), _ => (Col::Muted, "중립") };
            cx.t(sx, c, sl, 12.0, 800, 0.0, sc, ks);
            cx.t(sx + 38.0, c, fixed(n0(&it["impact"]), 1), 12.0, 800, 0.0, Col::Text, ks);
            let r1 = (ls.len().min(2) as f64 * LH).max(14.0);
            let kt = and(ks, [sx + 72.0, c, 308.0, r1]);
            for (n, l) in ls.iter().take(2).enumerate() {
                let ly = gt(c + n as f64 * LH, LH, 12.0);
                if n == 1 && ls.len() > 2 {
                    let (vis, _) = cx.cut(l, 12.0, 400, 308.0, true);
                    let vw = cx.w(&vis, 12.0, 400, 0.0);
                    cx.t(sx + 72.0, ly, vis, 12.0, 400, 0.0, Col::Text, kt);
                    cx.t(sx + 72.0 + vw, ly, "…", 12.0, 400, 0.0, Col::Text, kt);
                } else { cx.t(sx + 72.0, ly, l.clone(), 12.0, 400, 0.0, Col::Text, kt); }
            }
            cx.t(sx + 72.0, c + r1 + 1.0, format!("{} 전 · {}", ago(f(&it["ts_ms"]), now), news_asset(s(&it["asset"]))), 11.0, 400, 0.0, Col::Muted, ks);
            ry += rh + slot;
        }
        y += nat[2] + ex[2] + 10.0;
    }
    // ④ 출처
    for ((a, b), (x, ln, w0)) in spans.iter().zip(&place) {
        let ly = y + *ln as f64 * 18.0;
        cx.t(sx + x, ly, a.clone(), 11.5, 400, 0.0, Col::Muted, ks);
        cx.t(sx + x + w0, ly, b.clone(), 11.5, 700, 0.0, Col::Text, ks);
    }

    // ── 오른쪽(.nw-main): 12시간 축 svg · 목록 · ⑤ ──
    let (mx, mw) = (sw + 32.0, w - sw - 32.0);
    let lw = mw.round().max(320.0);
    let (lh_, l, r_, t_, b_) = (150.0, 30.0, 8.0, 10.0, 18.0);
    let t0 = now - (if truthy(&p["window_h"]) { f(&p["window_h"]) } else { 12.0 }) * 3600e3;
    let xx = |t: f64| l + (lw - l - r_) * (t - t0) / (now - t0);
    let yy = |v: f64| t_ + (lh_ - t_ - b_) * (1.0 - v / 3.0);
    let mut els = vec![];
    for v in [0.0, 1.0, 2.0, 3.0] {
        els.push(El::Line { p: [l, yy(v), lw - r_, yy(v)], c: Col::SoftLine, a: 1.0 });
        els.push(El::Text { x: l - 6.0, y: yy(v) + 3.0, s: format!("{v}"), z: 10.0, wt: 700, anchor: Anchor::End, c: Col::Muted, a: 1.0 });
    }
    for hh in [12.0, 9.0, 6.0, 3.0, 0.0] { els.push(El::Text { x: xx(now - hh * 3600e3), y: lh_ - 4.0, s: if hh != 0.0 { format!("-{hh}h") } else { "지금".into() }, z: 10.0, wt: 700, anchor: Anchor::Middle, c: Col::Muted, a: 1.0 }); }
    for it in judged.iter().filter(|it| f(&it["ts_ms"]) >= t0).rev() {
        let a = if rel(it) { if it["asset"] == coin.as_str() { 1.0 } else { 0.7 } } else { 0.25 };
        els.push(El::Circle { x: fx(xx(f(&it["ts_ms"])), 1), y: fx(yy(n0(&it["impact"])), 1), r: fx(2.2 + 3.6 * nz(&it["relevant"]), 1), c: sen_col(s(&it["sentiment"])), a, ring: nz(&it["incident"]) >= 0.8 });
    }
    let lane_h = lh_ * mw / lw;
    cx.v.push(It::S { x: mx, y: top0, w: mw, h: lane_h, vb: [lw, lh_], k: None, els });
    // ⑤ 가 있으면(지금 코인 응답이 available · ETH·XRP -- styles.css body.not-eth:not([data-coin=xrp]) #mcWhen 숨김) 목록 바닥이 그만큼 올라간다
    let when = i.mc.is_some_and(|d| truthy(&d["available"]) && d["asset"] == i.asset) && (i.asset == "eth" || i.asset == "xrp");
    let ly0 = top0 + lane_h + 6.0;
    let wy = h - 116.0;
    let lhh = if when { wy - 10.0 - ly0 } else { h - ly0 };
    let kl: K = Some([mx, ly0, mw, lhh]);
    out.list = [mx, ly0, mw, lhh];
    // 목록: 해킹·장애 고정(최대 5) + 최신부터 60건(관련만 · 이 코인+매크로 칩)
    let pins: Vec<usize> = items.iter().enumerate().filter(|(_, it)| truthy(&it["sentiment"]) && nz(&it["incident"]) >= 0.8).map(|q| q.0).take(5).collect();
    let mut list: Vec<usize> = pins.clone();
    list.extend(items.iter().enumerate().filter(|(k, it)| !pins.contains(k)
        && (!i.rel_only || if truthy(&it["sentiment"]) { rel(it) } else { !truthy(&it["model"]) })
        && (!i.coin_only || it["asset"] == coin.as_str() || it["asset"] == "macro")).map(|q| q.0).take(60));
    out.list_h = list.len() as f64 * 25.0;
    let a_x = mx + 284.0;
    let a_w = mw - 284.0;
    for (j, &k) in list.iter().enumerate() {
        let it = &items[k];
        let pin = pins.contains(&k);
        let jud = truthy(&it["sentiment"]);
        let dim = !pin && jud && !rel(it);
        let ry = ly0 + 25.0 * j as f64 - i.scroll;
        if pin { cx.b(mx - 6.0, ry, mw + 12.0, 25.0, Col::Amber, 0.09, 6.0, kl); }
        let after_pin = !pin && j > 0 && pins.contains(&list[j - 1]);
        let (bx, bw) = if pin { (mx - 6.0, mw + 12.0) } else { (mx, mw) };
        cx.b(bx, ry, bw, 1.0, if after_pin { Col::Line } else { Col::SoftLine }, 1.0, 0.0, kl);
        let c = ry + 6.0;
        let base = |z: f64| c + asc(12.0, false) - asc(z, false); // align-items: baseline
        cx.t(mx, c, hm(f(&it["ts_ms"]) / 1000.0, i.tz), 12.0, 400, 0.0, Col::Muted, kl);
        let ss = srcs(it);
        cx.t(mx + 48.0, base(11.0), ss.iter().map(|q| src_short(q)).collect::<Vec<_>>().join("·"), 11.0, 700, 0.22, if ss.len() > 1 { OPT } else { Col::Muted }, kl);
        let st = s(&it["sentiment"]);
        let (sl, sc, swt) = if jud { match st { "bullish" => ("호재", Col::Good, 800), "bearish" => ("악재", Col::Bad, 800), "neutral" => ("중립", Col::Muted, 700), o => (o, Col::Muted, 700) } } else { (if truthy(&it["model"]) { "본문 없음" } else { "대기" }, Col::Muted, 700) };
        let prob = jud.then(|| jround(100.0 * n0(&it[match st { "bullish" => "p_bull", "bearish" => "p_bear", "neutral" => "p_neu", _ => "" }])));
        cx.t(mx + 120.0, c, format!("{sl}{}", prob.map_or(String::new(), |q| format!(" {q}%"))), 12.0, swt, 0.0, sc, kl);
        let asn = news_asset(s(&it["asset"]));
        if !asn.is_empty() { cx.t(mx + 190.0, c, asn, 12.0, 700, 0.0, if it["asset"] == coin.as_str() { OPT } else { Col::Text }, kl); }
        let imp = n0(&it["impact"]);
        for q in 0..3 { cx.b(mx + 242.0 + 8.0 * q as f64, c + 2.0, 6.0, 10.0, if imp >= q as f64 + 0.5 { Col::Text } else { Col::SoftLine }, 1.0, 1.5, kl); }
        let ka = and(kl, [a_x, c, a_w, 14.0]);
        let mut x = a_x;
        if nz(&it["incident"]) >= 0.8 { cx.t(x, base(10.5), "해킹·장애", 10.5, 800, 0.0, Col::Amber, ka); x += cx.w("해킹·장애", 10.5, 800, 0.0) + 6.0; }
        let mut title = String::new();
        { let mut inside = false; for ch in s(&it["title"]).chars() { if ch == '<' && !inside { inside = true; } else if ch == '>' && inside { inside = false; title.push(' '); } else if !inside { title.push(ch); } } }
        let title = title.split_whitespace().collect::<Vec<_>>().join(" ");
        let tc = if dim { Col::Muted } else { Col::Text };
        let (vis, cut) = cx.cut(&title, 12.0, 400, a_x + a_w - x, false);
        let vw = cx.w(&vis, 12.0, 400, 0.0);
        cx.t(x, c, vis, 12.0, 400, 0.0, tc, ka);
        if cut { cx.t(x + vw, c, "…", 12.0, 400, 0.0, tc, ka); }
        let href = s(&it["link"]);
        if href.starts_with("http:") || href.starts_with("https:") { out.links.push(([a_x, c, a_w, 14.0], href.to_string())); }
    }
    if list.is_empty() { cx.t(mx, ly0 + ((16.5 - cont(11.0, false)) / 2.0).floor(), "조건에 맞는 기사가 없다.", 11.0, 400, 0.0, Col::Muted, kl); }
    // ⑤ 다음 24시간(#mcWhen 116 · 넘치면 잘림 · 윗변 --line · .mc-sec 윗변 --soft-line · 머리 «?» 단추 · 시간축 svg 폭 = 칸 − 8 을 칸 폭으로 늘임)
    if when {
        cx.b(mx, wy, mw, 1.0, Col::Line, 1.0, 0.0, None);
        let kw: K = Some([mx, wy, mw, 115.0]);
        cx.b(mx, wy + 1.0, mw, 1.0, Col::SoftLine, 1.0, 0.0, kw);
        let bt = wy + 1.0 + 1.0 + 10.0 + 1.703125; // h4 줄 상자 안 inline-flex 단추 위(크로미움 실측 -- h4 기본 글자 18.72px 의 바닥선 맞춤)
        let hs = "⑤ 다음 24시간";
        let tw = cx.w(hs, 12.0, 700, 0.24);
        cx.t(mx, bt + 2.0 + (15.0 - 14.4) / 2.0 + ((14.4 - cont(12.0, false)) / 2.0).floor(), hs, 12.0, 700, 0.24, Col::Text, kw);
        let qx = mx + tw + 6.0;
        cx.v.push(It::O { x: qx, y: bt + 2.0, w: 15.0, h: 15.0, lw: 1.0, c: Col::Line, a: 1.0, rad: 7.5, k: kw });
        let qw = cx.w("?", 10.0, 700, 0.0);
        cx.t(qx + 1.0 + (13.0 - qw) / 2.0, bt + 2.0 + 1.0 + (13.0 - cont(10.0, false)) / 2.0, "?", 10.0, 700, 0.0, Col::Muted, kw);
        let vw = (mw - 8.0).round().max(320.0);
        cx.v.push(It::S { x: mx, y: bt + 25.0, w: mw, h: 48.0 * mw / vw, vb: [vw, 48.0], k: kw, els: timeline(i, vw) });
    }
    out.items = cx.v;
    out
}

// ── 그리기 ──
/// 그림 단위 → [(자르기 상자, 그리기)] -- 자간 글은 글자마다 · svg 는 상자에 맞춰 늘임
pub fn prims(items: &[It], m: Measure) -> Vec<(K, Prim)> {
    let mut v = vec![];
    let cl = |k: &K| if k.is_some() { Clip::Plot } else { Clip::No };
    for it in items {
        match it {
            It::T { x, y, s, z, wt, ls, c, a, k } => {
                let base = y + asc(*z, false);
                let one = |x: f64, s: String| Prim::Text { x, y: base, s, size: *z, wt: *wt, c: *c, a: *a, anchor: Anchor::Start, halo: false, clip: cl(k) };
                if *ls == 0.0 { v.push((*k, one(*x, s.clone()))); continue; }
                let mut xx = *x;
                for ch in s.chars() { let t = ch.to_string(); let w = m(&t, *z, *wt, false); v.push((*k, one(xx, t))); xx += w + ls; }
            }
            &It::B { x, y, w, h, c, a, rad, k } => v.push((k, Prim::Rect { x, y, w, h, c, a, rx: rad, clip: cl(&k) })),
            &It::O { x, y, w, h, lw, c, a, rad, k } => {
                let (x, y, w, h) = (x + lw / 2.0, y + lw / 2.0, w - lw, h - lw);
                v.push((k, if rad == 0.0 { Prim::StrokeRect { x, y, w, h, c, a, lw, clip: cl(&k) } } else { Prim::Line { pts: rr_pts(x, y, w, h, rad - lw / 2.0), c, a, lw, dash: None, clip: cl(&k) } }));
            }
            It::S { x, y, w, h: _, vb, k, els } => {
                let sc = w / vb[0];
                let pt = |px: f64, py: f64| [x + px * sc, y + py * sc];
                for e in els {
                    match e {
                        El::Line { p, c, a } => v.push((*k, Prim::Line { pts: vec![pt(p[0], p[1]), pt(p[2], p[3])], c: *c, a: *a, lw: sc, dash: None, clip: cl(k) })),
                        El::Text { x: tx, y: ty, s, z, wt, anchor, c, a } => { let q = pt(*tx, *ty); v.push((*k, Prim::Text { x: q[0], y: q[1], s: s.clone(), size: z * sc, wt: *wt, c: *c, a: *a, anchor: *anchor, halo: false, clip: cl(k) })); }
                        El::Circle { x: cx, y: cy, r, c, a, ring } => {
                            let q = pt(*cx, *cy);
                            v.push((*k, Prim::Circle { x: q[0], y: q[1], r: r * sc, c: *c, a: *a, clip: cl(k) }));
                            if *ring { v.push((*k, Prim::Line { pts: (0..=24).map(|j| { let t = j as f64 / 24.0 * std::f64::consts::TAU; [q[0] + r * sc * t.cos(), q[1] + r * sc * t.sin()] }).collect(), c: Col::Amber, a: 1.0, lw: 2.0 * sc, dash: None, clip: cl(k) })); }
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
/// 원본 전역(latestNews · latestMarketCtx · latestGex · latestMacroEvents · newsView)과 받기 시각
pub struct Store {
    pub news: Option<Value>, pub mc: Option<Value>, pub gex: Option<Value>, pub events: Option<Value>,
    pub rel_only: bool, pub coin_only: bool, pub scroll: f64,
    news_at: f64, mc_at: f64, gex_at: f64, cal_at: f64,
    /// 마지막 그림의 누르기 자리
    pub chips: Vec<[f64; 4]>, pub links: Vec<([f64; 4], String)>, pub list: [f64; 4], pub list_h: f64,
}
impl Default for Store {
    fn default() -> Self { Self { news: None, mc: None, gex: None, events: None, rel_only: true, coin_only: false, scroll: 0.0, news_at: 0.0, mc_at: 0.0, gex_at: 0.0, cal_at: 0.0, chips: vec![], links: vec![], list: [0.0; 4], list_h: 0.0 } }
}
const NEWS_POLL_MS: f64 = 30_000.0;
const MC_POLL_MS: f64 = 1000.0;
const GEX_POLL_MS: f64 = 60_000.0;
const CAL_POLL_MS: f64 = 6.0 * 3600e3;
/// 56 FLOW_ASSETS(시장 맥락이 있는 코인)
const FLOW: [&str; 8] = ["eth", "btc", "sol", "xrp", "sndk", "soxl", "skhynix", "mu"];
impl Store {
    /// app.js tick(숨으면 아무것도 안 함): refreshMacroCalendar · refreshNews · refreshGex · refreshMarketCtx(흐름 코인만)
    pub fn tick(&mut self, now: f64, hidden: bool, api: &str, asset: &str, out: &mut Vec<Req>) {
        if hidden { return; }
        let mut get = |kind, path: String| out.push(Req::Get { kind, url: format!("{api}{path}") });
        if now - self.cal_at >= CAL_POLL_MS { self.cal_at = now; get(Kind::NewsCal, "/api/macro-calendar".into()); }
        if now - self.news_at >= NEWS_POLL_MS { self.news_at = now; get(Kind::News, "/api/news".into()); }
        if now - self.gex_at >= GEX_POLL_MS { self.gex_at = now; get(Kind::NewsGex, "/api/gex".into()); }
        if FLOW.contains(&asset) && now - self.mc_at >= MC_POLL_MS { self.mc_at = now; get(Kind::NewsMc, format!("/api/market-context?asset={asset}")); }
    }
    /// 응답(body = None 이면 실패 -- 접착은 HTTP 오류도 None) · 실패면 받아 둔 값 그대로(달력만 1분 뒤 다시)
    pub fn on_get(&mut self, kind: Kind, body: Option<&str>) {
        let v = body.and_then(|b| serde_json::from_str::<Value>(b).ok());
        match (kind, v) {
            (Kind::News, Some(v)) => self.news = Some(v),
            (Kind::NewsMc, Some(v)) => if truthy(&v["available"]) { self.mc = Some(v) },
            (Kind::NewsGex, Some(v)) => self.gex = Some(v),
            (Kind::NewsCal, Some(v)) => self.events = Some(if v["events"].is_array() { v["events"].clone() } else { Value::Array(vec![]) }),
            (Kind::NewsCal, None) => self.cal_at = self.cal_at - CAL_POLL_MS + 60_000.0,
            _ => {}
        }
    }
    pub fn input<'a>(&'a self, now: f64, tz: i64, asset: &'a str) -> In<'a> {
        In { now, tz, asset, news: self.news.as_ref(), mc: self.mc.as_ref(), gex: self.gex.as_ref(), events: self.events.as_ref(), rel_only: self.rel_only, coin_only: self.coin_only, scroll: self.scroll }
    }
    /// 누르기: 칩 = 보기 바꿈(true) · 제목 = 그 기사 주소
    pub fn click(&mut self, x: f64, y: f64) -> (bool, Option<String>) {
        let inside = |r: &[f64; 4]| x >= r[0] && x < r[0] + r[2] && y >= r[1] && y < r[1] + r[3];
        if let Some(k) = self.chips.iter().position(inside) { if k == 0 { self.rel_only = !self.rel_only; } else { self.coin_only = !self.coin_only; } return (true, None); }
        if !inside(&self.list) { return (false, None); }
        (false, self.links.iter().find(|l| inside(&l.0)).map(|l| l.1.clone()))
    }
    /// 목록 위 휠(dy = 아래로 굴리면 음수 -- egui) → 스크롤
    pub fn wheel(&mut self, x: f64, y: f64, dy: f64) -> bool {
        let r = self.list;
        if dy == 0.0 || !(x >= r[0] && x < r[0] + r[2] && y >= r[1] && y < r[1] + r[3]) { return false; }
        self.scroll = (self.scroll - dy).clamp(0.0, (self.list_h - r[3]).max(0.0));
        true
    }
}

thread_local! { static ST: RefCell<Option<Store>> = const { RefCell::new(None) }; }
/// ?card=news 로 켠다(web.rs start)
pub fn start() { ST.with(|s| *s.borrow_mut() = Some(Store::default())); }
pub fn on() -> bool { ST.with(|s| s.borrow().is_some()) }
/// Feed::tick 의 이 카드 몫 -- 켜져 있으면 true(다른 받기는 안 한다)
pub fn tick(now: f64, hidden: bool, api: &str, asset: &str, out: &mut Vec<Req>) -> bool {
    ST.with(|s| s.borrow_mut().as_mut().map(|st| st.tick(now, hidden, api, asset, out)).is_some())
}
/// web.rs on_get 의 이 카드 몫
pub fn on_get(kind: &str, body: Option<&str>) {
    let k = match kind { "News" => Kind::News, "NewsMc" => Kind::NewsMc, "NewsGex" => Kind::NewsGex, _ => Kind::NewsCal };
    ST.with(|s| if let Some(st) = s.borrow_mut().as_mut() { st.on_get(k, body); });
}
/// 한 장(Chart::ui 에서) -- 본문 = cfg.w_all×h_all(fill 이면 창 · 1920 화면 #newsBody 1838×917) · ⑤ 는 지금 시각으로 1초마다
pub fn ui(ch: &mut crate::Chart, ui: &mut eframe::egui::Ui, now: f64) {
    if ch.fill { let r = ui.ctx().content_rect(); (ch.cfg.w_all, ch.cfg.h_all) = ((r.width() - 2.0 * ch.origin.x) as f64, (r.height() - 2.0 * ch.origin.y) as f64); }
    let (w, h, o, tz) = (ch.cfg.w_all, ch.cfg.h_all, ch.origin, ch.cfg.tz_off_min);
    let p = ui.painter();
    if ch.bg.a() > 0 { p.rect_filled(p.clip_rect(), 0.0, ch.bg); }
    let measure = |s: &str, z: f64, wt: u16, _mono: bool| crate::draw::measure(p, s, z, wt);
    let (click, hover, dy) = ui.input(|i| (if i.pointer.primary_clicked() { i.pointer.interact_pos() } else { None }, i.pointer.hover_pos(), i.smooth_scroll_delta.y as f64));
    ST.with(|s| {
        let mut b = s.borrow_mut();
        let Some(st) = b.as_mut() else { return };
        if let Some(q) = hover { st.wheel((q.x - o.x) as f64, (q.y - o.y) as f64, dy); }
        if let Some(c) = click { if let (_, Some(url)) = st.click((c.x - o.x) as f64, (c.y - o.y) as f64) { ui.ctx().open_url(eframe::egui::OpenUrl::new_tab(url)); } }
        let out = body(&st.input(now, tz, &ch.feed.asset), w, h, &measure);
        if let Some(q) = hover { let (x, y) = ((q.x - o.x) as f64, (q.y - o.y) as f64); let on = |r: &[f64; 4]| x >= r[0] && x < r[0] + r[2] && y >= r[1] && y < r[1] + r[3];
            if out.chips.iter().any(on) || (on(&out.list) && out.links.iter().any(|l| on(&l.0))) { ui.ctx().set_cursor_icon(eframe::egui::CursorIcon::PointingHand); } }
        (st.chips, st.links, st.list, st.list_h) = (out.chips, out.links, out.list, out.list_h);
        st.scroll = st.scroll.clamp(0.0, (st.list_h - st.list[3]).max(0.0));
        let paint = |run: &[Prim], k: K| { let r = k.unwrap_or([0.0, 0.0, w, h]); crate::draw::paint(p, run, o, r, r, 1.0); };
        let (mut run, mut cur): (Vec<Prim>, K) = (vec![], None);
        for (k, pr) in prims(&out.items, &measure) { // 같은 자르기 상자끼리 한 번에
            if k != cur && !run.is_empty() { paint(&run, cur); run.clear(); }
            cur = k;
            run.push(pr);
        }
        paint(&run, cur);
    });
    ui.ctx().request_repaint_after(std::time::Duration::from_secs(1));
}

#[cfg(test)]
mod t {
    use super::*;
    use crate::mt::t::{col, lookup, pmc, r4, rad, wb};

    fn fx(name: &str) -> Value { serde_json::from_str(&std::fs::read_to_string(format!("{}/compare/fixture/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap() }
    fn nn(v: &Value) -> Option<&Value> { Some(v).filter(|v| !v.is_null()) }
    fn input(s: &Value) -> In<'_> {
        In { now: f(&s["now"]), tz: -540, asset: s["asset"].as_str().unwrap(), news: nn(&s["news"]), mc: nn(&s["mc"]), gex: nn(&s["gex"]), events: nn(&s["events"]),
             rel_only: s["view"]["relevant"] == true, coin_only: s["view"]["coin"] == true, scroll: 0.0 }
    }
    fn kk(v: &Value) -> K { v.as_array().map(|_| r4(v)) }
    /// 비교 단위: 글(t: x · 위 · 글) · 칠(b) · 테(o) · svg(s: 상자 · viewBox) -- kk = 자르기 상자
    #[derive(Debug, Clone)]
    struct Q { k: char, r: [f64; 4], c: [f64; 4], s: String, z: f64, wt: u16, lw: f64, rad: f64, kk: K }
    fn q(k: char, r: [f64; 4], c: [f64; 4], kk: K) -> Q { Q { k, r, c, s: String::new(), z: 0.0, wt: 0, lw: 0.0, rad: 0.0, kk } }
    fn anc(a: Anchor) -> &'static str { match a { Anchor::Start => "start", Anchor::Middle => "middle", Anchor::End => "end" } }
    fn el_key(e: &El) -> (String, Vec<f64>) {
        match e {
            El::Line { p, c, a } => (format!("line {:?}", col(*c, *a)), p.to_vec()),
            El::Text { x, y, s, z, wt, anchor, c, a } => (format!("text {s} {z} {} {} {:?}", wb(*wt as f64, false), anc(*anchor), col(*c, *a)), vec![*x, *y]),
            El::Circle { x, y, r, c, a, ring } => (format!("circle {:?}{}", col(*c, *a), if *ring { format!(" ring {:?} 2", col(Col::Amber, 1.0)) } else { String::new() }), vec![*x, *y, *r]),
        }
    }
    fn el_exp(e: &Value) -> (String, Vec<f64>) {
        let at = &e["at"];
        let a = |k: &str| at[k].as_str().unwrap().parse::<f64>().unwrap();
        let cc = |k: &str, o: &str| { let c = r4(&e[k]); format!("{:?}", pmc([c[0], c[1], c[2], c[3] * f(&e[o]) * f(&e["op"])])) };
        match e["tag"].as_str().unwrap() {
            "line" => (format!("line {}", cc("stroke", "so")), vec![a("x1"), a("y1"), a("x2"), a("y2")]),
            "text" => (format!("text {} {} {} {} {}", e["s"].as_str().unwrap(), f(&e["size"]), wb(f(&e["wt"]), false), e["anchor"].as_str().unwrap(), cc("fill", "fo")), vec![a("x"), a("y")]),
            "circle" => (format!("circle {}{}", cc("fill", "fo"), if e["stroke"].is_null() { String::new() } else { format!(" ring {} {}", cc("stroke", "so"), f(&e["sw"])) }), vec![a("cx"), a("cy"), a("r")]),
            t => panic!("{t}"),
        }
    }
    fn expected(d: &Value) -> (Vec<Q>, Vec<Vec<(String, Vec<f64>)>>) {
        let (mut out, mut svgs) = (vec![], vec![]);
        for t in d["texts"].as_array().unwrap() {
            out.push(Q { s: t["s"].as_str().unwrap().trim().into(), z: f(&t["size"]), wt: wb(f(&t["wt"]), false), ..q('t', r4(&t["r"]), pmc(r4(&t["c"])), kk(&t["k"])) });
        }
        for b in d["boxes"].as_array().unwrap() { let r = r4(&b["r"]); out.push(Q { rad: rad(&b["rad"], r), ..q('b', r, pmc(r4(&b["c"])), kk(&b["k"])) }); }
        for l in d["lines"].as_array().unwrap() {
            let r = r4(&l["r"]);
            out.push(if l["side"] == "all" { Q { lw: f(&l["w"]), rad: rad(&l["rad"], r), ..q('o', r, pmc(r4(&l["c"])), kk(&l["k"])) } } else { q('b', r, pmc(r4(&l["c"])), kk(&l["k"])) });
        }
        assert!(d["rings"].as_array().unwrap().is_empty());
        for s in d["svgs"].as_array().unwrap() {
            let vb = r4(&s["vb"]);
            out.push(Q { s: format!("{:?}", [vb[2], vb[3]]), ..q('s', r4(&s["r"]), [0.0; 4], kk(&s["k"])) });
            svgs.push(s["els"].as_array().unwrap().iter().map(el_exp).collect());
        }
        (out, svgs)
    }
    fn actual(items: &[It]) -> (Vec<Q>, Vec<Vec<(String, Vec<f64>)>>) {
        let (mut out, mut svgs) = (vec![], vec![]);
        for it in items {
            match it {
                It::T { x, y, s, z, wt, c, a, k, .. } => if !s.trim().is_empty() { out.push(Q { s: s.trim().into(), z: *z, wt: wb(*wt as f64, false), ..q('t', [*x, *y, 0.0, 0.0], col(*c, *a), *k) }) },
                &It::B { x, y, w, h, c, a, rad, k } => out.push(Q { rad, ..q('b', [x, y, w, h], col(c, a), k) }),
                &It::O { x, y, w, h, lw, c, a, rad, k } => out.push(Q { lw, rad, ..q('o', [x, y, w, h], col(c, a), k) }),
                It::S { x, y, w, h, vb, k, els } => { out.push(Q { s: format!("{vb:?}"), ..q('s', [*x, *y, *w, *h], [0.0; 4], *k) }); svgs.push(els.iter().map(el_key).collect()); }
            }
        }
        (out, svgs)
    }
    const TOL: f64 = 0.07;
    fn same(a: &Q, e: &Q) -> bool {
        let n = |x: f64, y: f64, t: f64| (x - y).abs() <= t;
        // 자르기 상자: 원본은 clientWidth·clientHeight(정수)로 잰다 -- ±1
        let kq = match (a.kk, e.kk) { (None, None) => true, (Some(p), Some(q)) => (0..4).all(|i| n(p[i], q[i], 1.0)), _ => false };
        kq && a.k == e.k && (0..4).all(|k| (a.c[k] - e.c[k]).abs() <= 1.0) && match a.k {
            't' => a.s == e.s && a.z == e.z && a.wt == e.wt && n(a.r[0], e.r[0], 0.6) && n(a.r[1], e.r[1], TOL),
            'o' => n(a.lw, e.lw, 1e-9) && n(a.rad, e.rad, TOL) && (0..4).all(|k| n(a.r[k], e.r[k], TOL)),
            'b' => n(a.rad, e.rad, TOL) && (0..4).all(|k| n(a.r[k], e.r[k], TOL)),
            _ => a.s == e.s && (0..4).all(|k| n(a.r[k], e.r[k], TOL)),
        }
    }
    fn show(v: &[Q]) -> String { v.iter().map(|q| format!("{} {:?} {:?} {:?} z{} w{} lw{} rad{} k{:?}", q.k, q.s.chars().take(60).collect::<String>(), q.r.map(|x| (x * 1000.0).round() / 1000.0), q.c, q.z, q.wt, q.lw, q.rad, q.kk)).collect::<Vec<_>>().join("\n") }

    /// 지금 화면의 실제 코드(compare/news_js.mjs)가 지은 HTML 을 원본 styles.css 로 헤드리스 크로미움에서 잰 것(compare/news_layout.py) -- 글·칠·테·자르기·svg 요소가 전부 하나씩 짝지어져야 한다
    #[test]
    fn js_parity() {
        // NEWS_JS = 변이 시험용 다른 잰 값(news_layout.py NEWS_OUT)
        let inp = fx("news_in.json");
        let js = std::env::var("NEWS_JS").map(|p| serde_json::from_str::<Value>(&std::fs::read_to_string(p).unwrap()).unwrap()).unwrap_or_else(|_| fx("news_js.json"));
        let cases = js["cases"].as_array().unwrap();
        // 글자 폭 표: 지어낸 «…»(폭 0) 은 뺀다
        let clean: Vec<Value> = cases.iter().filter(|c| !c.is_null()).map(|c| { let mut c = c.clone(); c["texts"] = Value::Array(c["texts"].as_array().unwrap().iter().filter(|t| t["synth"] != true).cloned().collect()); c }).collect();
        let refs: Vec<&Value> = clean.iter().collect();
        let m = lookup(&refs, &js["probe"]);
        let (mut n, mut nums) = (0, 0);
        for (k, (s, d)) in inp.as_array().unwrap().iter().zip(cases).enumerate() {
            let o = body(&input(s), f(&s["w"]), f(&s["h"]), &m);
            if d.is_null() { assert!(o.items.is_empty(), "입력 {k}: 카드가 숨어야 한다"); continue; }
            let (act, sa) = actual(&o.items);
            let (exp, se) = expected(d);
            let mut used = vec![false; act.len()];
            let mut miss = vec![];
            let (mut si, mut ps) = (0, vec![]);
            let idx: Vec<usize> = { let mut c = 0; act.iter().map(|a| if a.k == 's' { c += 1; c - 1 } else { usize::MAX }).collect() };
            for e in &exp {
                match (0..act.len()).find(|&j| !used[j] && same(&act[j], e)) { Some(j) => { used[j] = true; if e.k == 's' { ps.push((idx[j], si)); } } None => miss.push(e.clone()) }
                if e.k == 's' { si += 1; }
            }
            let extra: Vec<Q> = act.iter().zip(&used).filter(|q| !q.1).map(|q| q.0.clone()).collect();
            assert!(extra.is_empty() && miss.is_empty(), "입력 {k}\n남은 rtui {}\n{}\n못 찾은 원본 {}\n{}", extra.len(), show(&extra), miss.len(), show(&miss));
            for (a, e) in ps {
                assert_eq!(sa[a].len(), se[e].len(), "입력 {k} svg {e} 요소 수");
                for (x, y) in sa[a].iter().zip(&se[e]) {
                    assert!(x.0 == y.0 && x.1.len() == y.1.len() && x.1.iter().zip(&y.1).all(|(p, q)| (p - q).abs() < 1e-6), "입력 {k} svg {e}\n{x:?}\n{y:?}");
                    nums += y.1.len();
                }
            }
            n += exp.len();
        }
        assert!(n > 1900 && nums > 900, "{n} {nums}");
        eprintln!("news js_parity: 단위 {n} · svg 수 {nums}");
    }

    /// JS 수 글(optQ · toLocaleString · newsAgo) · 미국장(서머타임 경계) · 같은 분 일정 합치기 · 오늘·내일(지역 날짜)
    #[test]
    fn js_helpers() {
        assert_eq!([opt_q(&1.3.into()), opt_q(&12.25.into()), opt_q(&140.0.into()), opt_q(&0.25.into()), opt_q(&0.0123.into()), opt_q(&Value::Null)], ["1.30", "12.3", "140", "0.250", "0.012", "-"]);
        assert_eq!([n_int(&3200.0.into()), n_int(&1234567.5.into()), n_int(&999.4.into()), n_int(&Value::Null)], ["3,200", "1,234,568", "999", "-"]);
        assert_eq!([ago(0.0, 59.4 * 60e3), ago(0.0, 60.0 * 60e3), ago(0.0, 125.0 * 60e3), ago(10.0, 0.0)], ["59분", "1시간", "2시간 5분", "0분"]);
        let t = |s: &str| parse_iso(s, 0) * 1000.0;
        // 2026-11-18(수) 11:06 UTC = 뉴욕 06:06 EST → 개장 14:30 UTC · 3월 서머타임이면 13:30 UTC
        assert_eq!(us_session(t("2026-11-18T11:06Z")), Some((t("2026-11-18T14:30Z"), false)));
        assert_eq!(us_session(t("2026-11-18T15:00Z")), Some((t("2026-11-18T21:00Z"), true)));
        assert_eq!(us_session(t("2026-11-20T21:30Z")), Some((t("2026-11-23T14:30Z"), false)), "금 마감 뒤 = 월 개장");
        assert_eq!(us_session(t("2026-03-10T12:00Z")), Some((t("2026-03-10T13:30Z"), false)), "서머타임");
        let now = t("2026-11-18T11:06Z");
        let mc = serde_json::json!({ "asset": "eth", "available": true, "funding": { "next_ms": now + 3600e3 } });
        let ev = serde_json::json!([{ "time_utc": "2026-11-18T12:06:30+00:00", "importance": "low", "title_ko": "가" }, { "time_utc": "2026-11-19T14:59:00Z", "importance": "high", "title": "내일 끝" },
            { "time_utc": "2026-11-19T15:00:00Z", "importance": "high", "title_ko": "모레" }, { "time_utc": "2026-11-18T10:00:00Z", "importance": "high", "title_ko": "지남" }]);
        let i = In { now, tz: -540, asset: "eth", news: None, mc: Some(&mc), gex: None, events: Some(&ev), rel_only: true, coin_only: false, scroll: 0.0 };
        let e = events(&i);
        assert_eq!(e.iter().map(|q| (q.1.as_str(), q.2)).collect::<Vec<_>>(), [("펀딩 정산 · 가", Some(false)), ("미국장 개장", None), ("내일 끝", Some(true))], "같은 분 합침(undefined || false = false) · 서울 날짜로 내일까지");
    }

    /// 받기: 원본 주기(달력 6시간 · 뉴스 30초 · 옵션 1분 · 시장 맥락 1초 흐름 코인만) · 실패 = 받아 둔 값(달력만 1분 뒤) · 칩 · 링크 · 휠
    #[test]
    fn poll() {
        let gets = |o: &mut Vec<Req>| o.drain(..).map(|r| match r { Req::Get { url, .. } => url, _ => panic!() }).collect::<Vec<_>>();
        let mut s = Store::default();
        let mut o = vec![];
        s.tick(1e9, true, "", "eth", &mut o);
        assert!(o.is_empty(), "숨으면 안 묻는다");
        s.tick(1e9, false, "", "eth", &mut o);
        assert_eq!(gets(&mut o), ["/api/macro-calendar", "/api/news", "/api/gex", "/api/market-context?asset=eth"]);
        s.tick(1e9 + 1000.0, false, "", "eth", &mut o);
        assert_eq!(gets(&mut o), ["/api/market-context?asset=eth"]);
        s.on_get(Kind::NewsCal, None);
        s.on_get(Kind::NewsMc, Some(r#"{"asset":"eth","available":false}"#));
        assert!(s.mc.is_none(), "available 아니면 안 바꾼다");
        s.tick(1e9 + 60_000.0, false, "", "eth", &mut o);
        assert_eq!(gets(&mut o), ["/api/macro-calendar", "/api/news", "/api/gex", "/api/market-context?asset=eth"], "달력 실패 = 1분 뒤");
        s.tick(1e9 + 70_000.0, false, "", "hype", &mut o);
        assert!(gets(&mut o).is_empty(), "흐름 코인 아니면 시장 맥락 안 묻는다");
        s.on_get(Kind::News, Some(r#"{"ok":true}"#));
        s.on_get(Kind::News, None);
        assert!(s.news.is_some(), "실패면 받아 둔 값");
        (s.chips, s.list, s.list_h) = (vec![[0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 10.0, 10.0]], [0.0, 100.0, 50.0, 100.0], 400.0);
        s.links = vec![([0.0, 106.0, 50.0, 14.0], "https://x".into()), ([0.0, 231.0, 50.0, 14.0], "https://y".into())];
        assert_eq!(s.click(25.0, 5.0), (true, None));
        assert!(s.rel_only && s.coin_only);
        assert_eq!(s.click(5.0, 110.0), (false, Some("https://x".into())));
        assert_eq!(s.click(5.0, 235.0), (false, None), "목록 밖(스크롤로 가린 줄)은 안 연다");
        assert!(s.wheel(5.0, 150.0, -1000.0) && s.scroll == 300.0, "끝까지만");
        assert!(!s.wheel(5.0, 50.0, -10.0), "목록 밖 휠");
    }
}
