//! 왼쪽 위 «지표 방향» 판(?card=dirb) -- app.js renderDirBoard · dirReadNow · dirReadH · dirTri(FPB_W 268) 옮김.
//! 판 = 머리(«지표 방향» · 접기 단추 모양 · 위/아래 개수 막대) · 줄(rows: 보이는 표식마다 한 줄) · 바닥(cond: 큰 스윙 «넘으면 ▲ / 깨면 ▼») · «지금 읽기»(/api/situation read · ETH 만 · 3분 낡으면 없음).
//! rows·cond 는 원본에서 차트 그리기(SMA50·스윙 · renderObFvg · renderDirMarks -- 캔들로 계산)가 채운다 = 차트 카드 몫. 이 카드는 그 자리가 비어 «지금 읽기»만 산다(차트 표식을 옮기면 같은 Row·Cond 로 넘긴다).
//! 안 옮김: 툴팁(fpTipOn) · 접기 누르기(모양만 · 늘 펼침) · 휴대폰 목록(renderDirReadMob).
use crate::feed::{Kind, Req};
use crate::fp::{Anchor, Clip, Col, Prim};
use serde_json::Value;
use std::cell::RefCell;

pub type Measure<'a> = &'a dyn Fn(&str, f64, u16) -> f64;
pub const W: f64 = 268.0; // FPB_W
const RH: f64 = 23.0;
const PANEL: Col = Col::Rgb(46, 52, 62); // --panel-strong rgba(46, 52, 62, .92)

#[derive(Clone, Debug, serde::Deserialize)]
pub struct Row { pub d: i8, pub name: String, #[serde(default)] pub sub: String }
#[derive(Clone, Debug, serde::Deserialize)]
pub struct Cond { pub d: i8, pub tf: String, pub v: String, #[serde(default)] pub wait: bool }
/// read 한 줄 · g.v·g.on = 크기 값·켜짐
pub struct Line { pub topic: String, pub grade: String, pub dir: f64, pub text: String, pub gv: Option<f64>, pub gon: bool }
pub struct Read { pub hot: Vec<Line>, pub desc: Vec<Line> }

/// dirReadNow -- ETH 만 · 줄 없음·3분(180초) 넘게 안 갱신 = 없음 · 근거·약함 = 줄 · 나머지 = «설명»
pub fn read_now(s: Option<&Value>, asset: &str, now_ms: f64) -> Option<Read> {
    let s = s.filter(|_| asset == "eth")?;
    let lines = s["read"]["lines"].as_array().filter(|v| !v.is_empty())?;
    if now_ms / 1000.0 - s["computed_at"].as_f64().filter(|x| x.is_finite()).unwrap_or(0.0) > 180.0 { return None; }
    let st = |v: &Value| v.as_str().map_or_else(|| v.to_string(), str::to_owned);
    let (mut hot, mut desc) = (vec![], vec![]);
    for l in lines {
        let g = &l["g"];
        let x = Line { topic: st(&l["topic"]), grade: st(&l["grade"]), dir: l["dir"].as_f64().unwrap_or(0.0), text: st(&l["text"]), gv: g["v"].as_f64(), gon: g["on"].as_bool().unwrap_or(false) };
        if x.grade == "근거" || x.grade == "약함" { hot.push(x) } else { desc.push(x) }
    }
    Some(Read { hot, desc })
}
/// dirReadH
pub fn read_h(r: Option<&Read>) -> f64 { r.map_or(0.0, |r| 30.0 + r.hot.len() as f64 * RH + if r.desc.is_empty() { 0.0 } else { 22.0 } + 4.0) }
/// short(l): « — » 뒤 첫 조각(없으면 전체)의 «:» 앞
fn short(t: &str) -> &str { t.split(" — ").nth(1).filter(|s| !s.is_empty()).unwrap_or(t).split(':').next().unwrap_or("") }
/// fit(txt, font, max) -- 넘치면 한 글자씩 덜고 «…»
fn fit(txt: &str, z: f64, wt: u16, max: f64, m: Measure) -> String {
    let mut t: Vec<char> = txt.chars().collect();
    let cur = |t: &[char]| { let s: String = t.iter().collect(); if s == txt { s } else { s + "…" } };
    while t.len() > 1 && m(&cur(&t), z, wt) > max { t.pop(); }
    cur(&t)
}

/// 판 요소(원본 svg 요소 하나씩 · 문서 차례) -- R = rect(stroke = 테 var(--line)) · Tri = dirTri · Chev = 접기 단추 꺾쇠 · Ln = 가로 구분선 · T = text(y = 바닥선)
#[derive(Debug, Clone, PartialEq)]
pub enum E {
    R { x: f64, y: f64, w: f64, h: f64, rx: f64, c: Col, a: f64, stroke: bool },
    Tri { cx: f64, cy: f64, d: i8, c: Col, r: f64 },
    Chev([[f64; 2]; 3]),
    Ln { x1: f64, x2: f64, y: f64 },
    T { x: f64, y: f64, s: String, size: f64, wt: u16, c: Col, end: bool },
}
fn t(v: &mut Vec<E>, x: f64, y: f64, s: impl Into<String>, size: f64, wt: u16, c: Col, end: bool) { let s = s.into(); if !s.is_empty() { v.push(E::T { x, y, s, size, wt, c, end }); } }
fn stripe(v: &mut Vec<E>, l: f64, y: f64, i: usize) { if i % 2 == 0 { v.push(E::R { x: l + 6.0, y, w: W - 12.0, h: RH - 2.0, rx: 5.0, c: Col::Text, a: 0.035, stroke: false }); } }

/// renderDirBoard -- (l, t0) = 판 왼쪽 위(원본 G.ml + 6 · G.mt + B.top) · 반환 = (요소, 높이) · 그릴 것이 없으면 None
pub fn board(rows: &[Row], cond: &[Cond], r: Option<&Read>, fold: bool, l: f64, t0: f64, m: Measure) -> Option<(Vec<E>, f64)> {
    if rows.is_empty() && cond.is_empty() && r.is_none() { return None; }
    let up = rows.iter().filter(|q| q.d > 0).count();
    let dn = rows.iter().filter(|q| q.d < 0).count();
    let ups: Vec<&Cond> = cond.iter().filter(|c| c.d > 0).collect();
    let dns: Vec<&Cond> = cond.iter().filter(|c| c.d < 0).collect();
    let head = if rows.is_empty() { 30.0 } else { 44.0 };
    let foot = if fold { 0.0 } else { (if ups.is_empty() { 0.0 } else { 20.0 }) + if dns.is_empty() { 0.0 } else { 20.0 } };
    let hb = head + rows.len() as f64 * RH + if foot > 0.0 { foot + 10.0 } else { 4.0 };
    let h = if fold { head } else { hb + read_h(r) };
    let mut v = vec![E::R { x: l, y: t0, w: W, h, rx: 10.0, c: PANEL, a: 0.92 * 0.95, stroke: true }];
    t(&mut v, l + 12.0, t0 + 19.0, "지표 방향", 11.0, 700, Col::Muted, false);
    v.push(E::R { x: l + W - 30.0, y: t0 + 5.0, w: 24.0, h: 20.0, rx: 6.0, c: Col::Text, a: 0.06, stroke: false });
    let (cx, cy) = (l + W - 18.0, t0 + 15.0);
    v.push(E::Chev(if fold { [[cx - 4.0, cy - 2.0], [cx, cy + 2.0], [cx + 4.0, cy - 2.0]] } else { [[cx - 4.0, cy + 2.0], [cx, cy - 2.0], [cx + 4.0, cy + 2.0]] }));
    if !rows.is_empty() {
        let (bx, bw) = (l + 86.0, W - 98.0 - 30.0);
        let n = if up + dn == 0 { 1.0 } else { (up + dn) as f64 };
        let uw = (bw * up as f64 / n).round();
        let (u1, d1) = (if dn > 0 { 1.0 } else { 0.0 }, if up > 0 { 1.0 } else { 0.0 });
        if up > 0 { v.push(E::R { x: bx, y: t0 + 11.0, w: (uw - u1).max(2.0), h: 9.0, rx: 3.0, c: Col::Good, a: 1.0, stroke: false }); }
        if dn > 0 { v.push(E::R { x: bx + uw + d1, y: t0 + 11.0, w: (bw - uw - d1).max(2.0), h: 9.0, rx: 3.0, c: Col::Bad, a: 1.0, stroke: false }); }
        v.push(E::Tri { cx: bx + 5.0, cy: t0 + 32.0, d: 1, c: Col::Good, r: 4.0 });
        t(&mut v, bx + 13.0, t0 + 36.0, format!("위 {up}"), 11.0, 750, Col::Good, false);
        v.push(E::Tri { cx: bx + bw - 5.0, cy: t0 + 32.0, d: -1, c: Col::Bad, r: 4.0 });
        t(&mut v, bx + bw - 13.0, t0 + 36.0, format!("아래 {dn}"), 11.0, 750, Col::Bad, true);
    }
    if fold { return Some((v, h)); }
    for (i, q) in rows.iter().enumerate() {
        let (y, c) = (t0 + head + i as f64 * RH, if q.d > 0 { Col::Good } else { Col::Bad });
        stripe(&mut v, l, y, i);
        v.push(E::Tri { cx: l + 19.0, cy: y + (RH - 2.0) / 2.0, d: q.d, c, r: 5.0 });
        t(&mut v, l + 32.0, y + 15.0, &q.name, 12.0, 700, Col::Text, false);
        t(&mut v, l + W - 12.0, y + 15.0, &q.sub, 11.0, 550, Col::Muted, true);
    }
    let mut yc = t0 + head + rows.len() as f64 * RH + 4.0;
    if foot > 0.0 { v.push(E::Ln { x1: l + 10.0, x2: l + W - 10.0, y: yc }); }
    for (arr, d, word) in [(&ups, 1i8, "넘으면"), (&dns, -1, "깨면")] {
        if arr.is_empty() { continue; }
        let c = if d > 0 { Col::Good } else { Col::Bad };
        let ww = m(word, 11.0, 650);
        t(&mut v, l + 12.0, yc + 15.0, word, 11.0, 650, Col::Muted, false);
        v.push(E::Tri { cx: l + 12.0 + ww + 9.0, cy: yc + 11.0, d, c, r: 4.0 });
        let s = arr.iter().map(|q| format!("{} {}{}", q.tf, q.v, if q.wait { " 마감 대기" } else { "" })).collect::<Vec<_>>().join(" · ");
        t(&mut v, l + 12.0 + m(word, 11.0, 650) + 18.0, yc + 15.0, s, 11.0, 650, Col::Text, false);
        yc += 20.0;
    }
    let Some(r) = r else { return Some((v, h)) };
    // ── 지금 읽기 ──
    let ys = t0 + hb;
    let nu = r.hot.iter().filter(|q| q.dir > 0.0).count();
    let nd = r.hot.iter().filter(|q| q.dir < 0.0).count();
    if !rows.is_empty() || foot > 0.0 { v.push(E::Ln { x1: l + 10.0, x2: l + W - 10.0, y: ys }); }
    t(&mut v, l + 12.0, ys + 19.0, "지금 읽기", 11.0, 700, Col::Muted, false);
    let (cu, cd) = (if nu > 0 { Col::Good } else { Col::Muted }, if nd > 0 { Col::Bad } else { Col::Muted });
    v.push(E::Tri { cx: l + W - 64.0, cy: ys + 15.0, d: 1, c: cu, r: 4.0 });
    t(&mut v, l + W - 56.0, ys + 19.0, nu.to_string(), 11.0, 750, cu, false);
    v.push(E::Tri { cx: l + W - 32.0, cy: ys + 15.0, d: -1, c: cd, r: 4.0 });
    t(&mut v, l + W - 24.0, ys + 19.0, nd.to_string(), 11.0, 750, cd, false);
    for (i, q) in r.hot.iter().enumerate() {
        let y = ys + 28.0 + i as f64 * RH;
        let c = if q.dir > 0.0 { Col::Good } else if q.dir < 0.0 { Col::Bad } else if q.gv.is_some_and(|g| g >= 80.0) { Col::Amber } else { Col::Muted };
        stripe(&mut v, l, y, i);
        if q.dir != 0.0 { v.push(E::Tri { cx: l + 19.0, cy: y + (RH - 2.0) / 2.0, d: if q.dir > 0.0 { 1 } else { -1 }, c, r: 5.0 }); }
        else { v.push(E::R { x: l + 15.0, y: y + 6.5, w: 8.0, h: 8.0, rx: 2.0, c, a: 1.0, stroke: false }); }
        t(&mut v, l + 32.0, y + 15.0, &q.topic, 12.0, 700, Col::Text, false);
        let gw = m(&q.grade, 10.5, 700);
        let room = W - 12.0 - 32.0 - m(&q.topic, 12.0, 700) - gw - 22.0;
        t(&mut v, l + W - 12.0, y + 15.0, &q.grade, 10.5, 700, c, true);
        if room > 24.0 { t(&mut v, l + W - 12.0 - gw - 7.0, y + 15.0, fit(short(&q.text), 11.0, 550, room, m), 11.0, 550, Col::Muted, true); }
    }
    if !r.desc.is_empty() {
        let y = ys + 28.0 + r.hot.len() as f64 * RH;
        let on: Vec<&str> = r.desc.iter().filter(|q| q.gon).map(|q| q.topic.as_str()).collect();
        let lab = format!("설명 {}", r.desc.len());
        t(&mut v, l + 12.0, y + 14.0, &lab, 11.0, 650, Col::Muted, false);
        if !on.is_empty() { let lw = m(&lab, 11.0, 650); t(&mut v, l + 12.0 + lw + 8.0, y + 14.0, fit(&on.join(" · "), 11.0, 550, W - 24.0 - lw - 8.0, m), 11.0, 550, Col::Muted, false); }
    }
    Some((v, h))
}

/// 꼭짓점 셋(dirTri path 차례: 바닥 왼 · 바닥 오른 · 꼭지)
pub fn tri_pts(cx: f64, cy: f64, d: i8, r: f64) -> [[f64; 2]; 3] {
    if d > 0 { [[cx - r, cy + r * 0.8], [cx + r, cy + r * 0.8], [cx, cy - r * 0.8]] } else { [[cx - r, cy - r * 0.8], [cx + r, cy - r * 0.8], [cx, cy + r * 0.8]] }
}
/// 요소 → 그리기 목록(삼각형 = 바닥선까지 칠한 면 Area · 둥근 테 = 닫힌 꺾은선)
pub fn prims(v: &[E]) -> Vec<Prim> {
    let mut out = vec![];
    for e in v {
        match e {
            &E::R { x, y, w, h, rx, c, a, stroke } => {
                out.push(Prim::Rect { x, y, w, h, c, a, rx, clip: Clip::No });
                if stroke { out.push(Prim::Line { pts: crate::mt::rr_pts(x, y, w, h, rx), c: Col::Line, a: 1.0, lw: 1.0, dash: None, clip: Clip::No }); }
            }
            &E::Tri { cx, cy, d, c, r } => { let p = tri_pts(cx, cy, d, r); out.push(Prim::Area { pts: vec![p[0], p[2], p[1]], base: p[0][1], c, a: 1.0 }); }
            E::Chev(p) => out.push(Prim::Line { pts: p.to_vec(), c: Col::Muted, a: 1.0, lw: 1.8, dash: None, clip: Clip::No }),
            &E::Ln { x1, x2, y } => out.push(Prim::Line { pts: vec![[x1, y], [x2, y]], c: Col::Line, a: 1.0, lw: 1.0, dash: None, clip: Clip::No }),
            E::T { x, y, s, size, wt, c, end } => out.push(Prim::Text { x: *x, y: *y, s: s.clone(), size: *size, wt: *wt, c: *c, a: 1.0, anchor: if *end { Anchor::End } else { Anchor::Start }, halo: false, clip: Clip::No }),
        }
    }
    out
}

// ── 받기 ──
// ponytail: 상태를 Feed 칸 대신 스레드 지역에 둔다 -- Feed::new·Chart::new 가 한 줄 생성자라 공용 파일 기존 줄을 안 고치려고(웹어셈블리는 한 스레드). 공용 줄을 정리할 때 Feed 칸으로 옮긴다.
#[derive(Default)]
struct Store { sit: Option<Value>, at: f64 }
thread_local! { static ST: RefCell<Option<Store>> = const { RefCell::new(None) }; }
const POLL_MS: f64 = 1000.0; // SITUATION_POLL_MS
pub fn enable() { ST.with(|s| *s.borrow_mut() = Some(Store { sit: None, at: f64::NEG_INFINITY })); }
pub fn on() -> bool { ST.with(|s| s.borrow().is_some()) }
/// refreshSituation(1초 · 숨으면 안 묻는다) -- 이 판은 ETH 만 읽어 ETH 만 묻는다
pub fn tick(now: f64, hidden: bool, asset: &str, api: &str, out: &mut Vec<Req>) {
    ST.with(|s| if let Some(s) = s.borrow_mut().as_mut() {
        if !hidden && asset == "eth" && now - s.at >= POLL_MS { s.at = now; out.push(Req::Get { kind: Kind::Situation, url: format!("{api}/api/situation") }); }
    });
}
/// 실패 = 원본 {now: {ok: false}} -- read 가 없어 판이 안 보이는 것과 같다
pub fn on_get(body: Option<&str>) { ST.with(|s| if let Some(s) = s.borrow_mut().as_mut() { s.sit = body.and_then(|b| serde_json::from_str(b).ok()); }); }

impl crate::Chart {
    /// 판 한 장(왼쪽 위 = origin) -- 3분 낡음이 시간에 걸리니 1초마다 다시 그린다
    pub fn ui_dirb(&mut self, ui: &mut eframe::egui::Ui, now: f64) {
        let p = ui.painter();
        if self.bg.a() > 0 { p.rect_filled(p.clip_rect(), 0.0, self.bg); }
        let measure = |s: &str, z: f64, wt: u16| crate::draw::measure(p, s, z, wt);
        let r = ST.with(|s| s.borrow().as_ref().and_then(|s| read_now(s.sit.as_ref(), &self.feed.asset, now)));
        if let Some((v, h)) = board(&[], &[], r.as_ref(), false, 0.0, 0.0, &measure) { crate::draw::paint(p, &prims(&v), self.origin, [0.0, 0.0, W, h], [0.0, 0.0, W, h], 1.0); }
        ui.ctx().request_repaint_after(std::time::Duration::from_secs(1));
    }
}

#[cfg(test)]
mod t {
    use super::*;

    fn fx(f: &str) -> Value { serde_json::from_str(&std::fs::read_to_string(format!("{}/compare/fixture/{f}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap() }
    fn n(v: &Value) -> f64 { v.as_str().map_or(f64::NAN, |s| s.parse().unwrap()) }
    /// 비교 단위 = (종류, 수들, 글, rgb, 알파)
    type K = (&'static str, Vec<f64>, String, [u8; 3], f64);
    fn exp(e: &Value) -> Option<K> {
        let a = &e["at"];
        let c = |k: &str, o: &str| { let q = &e[k]; let op = a[o].as_str().map_or(1.0, |s| s.parse().unwrap()); ([0, 1, 2].map(|i| q[i].as_f64().unwrap() as u8), q[3].as_f64().unwrap() * op) };
        let nums = |d: &str| d.replace(['M', 'L', 'Z', 'l'], " ").split_whitespace().map(|x| x.parse().unwrap()).collect::<Vec<f64>>();
        Some(match e["tag"].as_str().unwrap() {
            "rect" => {
                let (rgb, al) = c("fill", "fill-opacity");
                if al == 0.0 { return None; } // 투명(마우스 받이 · 홀수 줄)
                ("rect", vec![n(&a["x"]), n(&a["y"]), n(&a["width"]), n(&a["height"]), a["rx"].as_str().map_or(0.0, |s| s.parse().unwrap()), if a["stroke"].is_null() { 0.0 } else { 1.0 }], String::new(), rgb, al)
            }
            "path" if a["fill"] == "none" => { // 꺾쇠: M x y l dx dy l dx dy
                let q = nums(a["d"].as_str().unwrap());
                let (rgb, al) = c("stroke", "stroke-opacity");
                ("chev", vec![q[0], q[1], q[0] + q[2], q[1] + q[3], q[0] + q[2] + q[4], q[1] + q[3] + q[5], n(&a["stroke-width"])], String::new(), rgb, al)
            }
            "path" => { let (rgb, al) = c("fill", "fill-opacity"); ("tri", nums(a["d"].as_str().unwrap()), String::new(), rgb, al) }
            "line" => { let (rgb, al) = c("stroke", "stroke-opacity"); assert_eq!(a["y1"], a["y2"]); ("ln", vec![n(&a["x1"]), n(&a["x2"]), n(&a["y1"])], String::new(), rgb, al) }
            "text" => {
                let s = e["s"].as_str().unwrap();
                if s.is_empty() { return None; }
                let (rgb, al) = c("fill", "fill-opacity");
                ("text", vec![n(&a["x"]), n(&a["y"]), n(&a["font-size"]), n(&a["font-weight"]), if a["text-anchor"] == "end" { 1.0 } else { 0.0 }], s.into(), rgb, al)
            }
            t => panic!("요소 {t}"),
        })
    }
    fn act(e: &E) -> K {
        let col = |c: Col, a: f64| { let [r, g, b, q] = crate::draw::rgba(c, 1.0).to_srgba_unmultiplied(); ([r, g, b], q as f64 / 255.0 * a) };
        let k = |k, v, s: &str, (rgb, al)| (k, v, s.to_string(), rgb, al);
        match e {
            &E::R { x, y, w, h, rx, c, a, stroke } => k("rect", vec![x, y, w, h, rx, if stroke { 1.0 } else { 0.0 }], "", col(c, a)),
            &E::Tri { cx, cy, d, c, r } => k("tri", tri_pts(cx, cy, d, r).concat(), "", col(c, 1.0)),
            E::Chev(p) => k("chev", p.concat().into_iter().chain([1.8]).collect(), "", col(Col::Muted, 1.0)),
            &E::Ln { x1, x2, y } => k("ln", vec![x1, x2, y], "", col(Col::Line, 1.0)),
            E::T { x, y, s, size, wt, c, end } => k("text", vec![*x, *y, *size, *wt as f64, if *end { 1.0 } else { 0.0 }], s, col(*c, 1.0)),
        }
    }
    /// 정답지 = 지금 화면의 실제 renderDirBoard 를 원본 styles.css 위 헤드리스 크로미움에서 돌린 svg 요소 전부(compare/dirb_js.mjs → dirb_layout.py) --
    /// 글 폭은 원본 measureTextW 가 잰 값을 그대로 쓴다 · 요소 차례·수·글·색(계산된 fill/stroke × 불투명) · 자리 1e-6
    #[test]
    fn js_parity() {
        let inp = fx("dirb_in.json");
        let js = std::env::var("DIRB_JS").map(|p| serde_json::from_str::<Value>(&std::fs::read_to_string(p).unwrap()).unwrap()).unwrap_or_else(|_| fx("dirb_js.json"));
        let mut total = 0;
        for (k, (s, d)) in inp.as_array().unwrap().iter().zip(js.as_array().unwrap()).enumerate() {
            let tab: Vec<(String, f64, u16, f64)> = d["measured"].as_array().unwrap().iter().map(|q| {
                let f = q[1].as_str().unwrap();
                let mut it = f.split_whitespace();
                let wt = it.next().unwrap().parse().unwrap();
                (q[0].as_str().unwrap().to_string(), it.next().unwrap().trim_end_matches("px").parse().unwrap(), wt, q[2].as_f64().unwrap())
            }).collect();
            let m = |s: &str, z: f64, wt: u16| tab.iter().find(|q| q.0 == s && q.1 == z && q.2 == wt).unwrap_or_else(|| panic!("입력 {k}: 원본이 안 잰 글 {s:?} {z} {wt}")).3;
            let rows: Vec<Row> = serde_json::from_value(s["rows"].clone()).unwrap();
            let cond: Vec<Cond> = serde_json::from_value(s["cond"].clone()).unwrap();
            let r = read_now(Some(&s["S"]).filter(|v| !v.is_null()), s["asset"].as_str().unwrap(), s["now"].as_f64().unwrap() * 1000.0);
            let (l, t0) = (s["G"]["ml"].as_f64().unwrap() + 6.0, s["G"]["mt"].as_f64().unwrap() + s["top"].as_f64().unwrap());
            let out = board(&rows, &cond, r.as_ref(), s["fold"].as_bool().unwrap(), l, t0, &m);
            let e: Vec<K> = d["els"].as_array().unwrap().iter().filter_map(exp).collect();
            let a: Vec<K> = out.as_ref().map_or(vec![], |o| o.0.iter().map(act).collect());
            assert_eq!(a.len(), e.len(), "입력 {k} 요소 수\nrtui {a:#?}\n원본 {e:#?}");
            for (i, (x, y)) in a.iter().zip(&e).enumerate() {
                let same = x.0 == y.0 && x.2 == y.2 && x.3.iter().zip(&y.3).all(|(p, q)| p.abs_diff(*q) <= 3) && (x.4 - y.4).abs() < 0.003 && x.1.len() == y.1.len() && x.1.iter().zip(&y.1).all(|(p, q)| (p - q).abs() < 1e-6);
                assert!(same, "입력 {k} 요소 {i}\nrtui {x:?}\n원본 {y:?}");
            }
            if let (Some((_, h)), Some(bg)) = (&out, e.first()) { assert_eq!(*h, bg.1[3], "입력 {k} 높이"); }
            total += e.len();
            // 그리기 목록도 만들어진다(삼각형 = 면 · 테 = 닫힌 선)
            if let Some((v, _)) = &out { assert!(prims(v).len() > v.len()); }
        }
        assert!(total > 150, "{total}");
    }

    /// 1초 · 숨으면·ETH 아니면 안 묻는다 · 실패 = 없음
    #[test]
    fn poll() {
        enable();
        let mut out = vec![];
        tick(1e6, false, "eth", "", &mut out);
        tick(1e6 + 500.0, false, "eth", "", &mut out);
        assert_eq!(out, vec![Req::Get { kind: Kind::Situation, url: "/api/situation".into() }]);
        tick(1e6 + 2000.0, true, "eth", "", &mut out);
        tick(1e6 + 2000.0, false, "btc", "", &mut out);
        assert_eq!(out.len(), 1);
        on_get(Some(r#"{"computed_at":1000,"read":{"lines":[{"topic":"a","grade":"근거","dir":1,"text":"x"}]}}"#));
        let r = ST.with(|s| read_now(s.borrow().as_ref().unwrap().sit.as_ref(), "eth", 1_100_000.0)).unwrap();
        assert_eq!((r.hot.len(), r.desc.len()), (1, 0));
        on_get(None);
        assert!(ST.with(|s| s.borrow().as_ref().unwrap().sit.is_none()));
        assert_eq!(short("a — b: c"), "b");
        assert_eq!(short("a — "), "a — ");
    }
}
