//! «청산» 탭 윗줄 왼쪽 «남은 매물 일봉별»(dashboard/live/liq_tab.js CH.remain 152~190 · heatUrl 128~144 · remView 147~151 · remainKpi 448~454 · 범례 lqRemLeg 481 ·
//!   끌기·휠·두 번 클릭 init 555~603 · 십자 가격선 el._xh 175~181 · 히트맵 색 = app.js densityColor(다크 칸 색 DENSITY_STOPS_DARK)).
//! 자료 = 같은 출처 /api/liq-tab(10분 · remain[기간]·days·price) + 허브 «모든 종목» 가격(liq.rs price 와 같은 대체).
//! 안 옮긴 것: 머리(제목 «남은 매물 일봉별» · «파생·서술» 꼬리표 · 기간 단추 7…1년 -- 기간은 ?win= 로 고정) · 호버 툴팁(el._hit) · 두 손가락 확대(터치 핀치) ·
//!   좁은 화면(1100px 이하) 지표 2열 · 시장 맥락 mid 가격 대체 · 라이트 테마 칸 색.
//! ponytail: 히트맵 = 칸마다 사각형(원본은 캔버스 그림 한 장을 2배 해상도로 굽고 끄는 동안 1배) -- 칸이 수만 개면 텍스처 한 장으로 굽는다.
use crate::fp::{Anchor, Clip, Col, Prim};
use crate::liq::{nice_step, pc, px, round};
use serde_json::Value;

/// 보기(S.rv): 왼쪽 날 번호(소수) · 보이는 날 수 · 가격 이동 · 가격 폭 배율
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rv { pub n: usize, pub i0: f64, pub span: f64, pub y_off: f64, pub yk: f64 }
/// 마지막 그림의 자리(el._g + 십자선에 쓰는 것)
#[derive(Clone, Copy, Debug)]
pub struct Geo { pub l: f64, pub xr: f64, pub tp: f64, pub ch: f64, pub cw: f64, pub k: f64, lo: f64, hi: f64, i0: f64, n: usize, h: f64 }
#[derive(Clone, Copy)]
struct Drag { x: f64, y: f64, i0: f64, y_off: f64, cw: f64, k: f64, moved: bool }
/// svg 상자 좌표 이벤트(브라우저 deltaX·Y 뜻: 양수 = 아래·오른쪽)
pub enum Ev { Down(f64, f64), Move(f64, f64), Up, Wheel { x: f64, y: f64, dx: f64, dy: f64, shift: bool }, Dbl }

#[derive(Default)]
pub struct Rem { pub win: u32, pub rv: Option<Rv>, drag: Option<Drag>, pub geo: Option<Geo> }

/// remView(n): 기간·코인이 바뀌면 처음 보기 · 범위를 조인다
fn rem_view(rv: &mut Option<Rv>, n: usize) -> Rv {
    if rv.is_none_or(|v| v.n != n) { *rv = Some(Rv { n, i0: 0.0, span: n as f64, y_off: 0.0, yk: 1.0 }); }
    let v = rv.as_mut().unwrap();
    let nf = n as f64;
    v.span = 5f64.max((nf + 4.0).min(v.span));
    v.i0 = (3.0 - v.span).max((nf - 3.0).min(v.i0));
    v.yk = 0.1f64.max(10f64.min(v.yk));
    *v
}

impl Rem {
    /// 끄는 중(4px 넘음 -- 손 모양 커서)
    pub fn panning(&self) -> bool { self.drag.is_some_and(|d| d.moved) }
    /// init 555~603: 끌기(4px 넘어야 시작) · 휠(왼쪽 가격축 = 가격 확대 · 가로·Shift = 이동 · 나머지 = 마우스 날 고정 확대) · 두 번 클릭 = 처음 보기. 자리는 지난 그림(el._g)
    pub fn event(&mut self, e: Ev) {
        let (Some(g), Some(v)) = (self.geo, self.rv.as_mut()) else { if matches!(e, Ev::Up) { self.drag = None; } return };
        match e {
            Ev::Down(x, y) => self.drag = Some(Drag { x, y, i0: v.i0, y_off: v.y_off, cw: g.cw, k: g.k, moved: false }),
            Ev::Move(x, y) => {
                let Some(d) = self.drag.as_mut() else { return };
                let (dx, dy) = (x - d.x, y - d.y);
                if !d.moved && dx.abs() < 4.0 && dy.abs() < 4.0 { return; }
                d.moved = true;
                v.i0 = d.i0 - dx / d.cw;
                v.y_off = d.y_off + dy * d.k; // 아래로 끌면 위쪽 가격(풋프린트와 같다)
            }
            Ev::Up => self.drag = None,
            Ev::Wheel { x, y, dx, dy, shift } => {
                if x < g.l { let f = (dy * 0.0015).exp(); v.y_off += (g.tp + g.ch / 2.0 - y) * g.k * (1.0 - f); v.yk *= f; }
                else if dx.abs() > dy.abs() || shift { v.i0 += (if dx != 0.0 { dx } else { dy }) / g.cw; }
                else {
                    let at = v.i0 + (x - g.l) / g.cw;
                    let span = 5f64.max((v.n as f64 + 4.0).min(v.span * (dy * 0.0015).exp()));
                    v.i0 = at - (at - v.i0) * span / v.span;
                    v.span = span;
                }
            }
            Ev::Dbl => self.rv = None,
        }
    }
}

/// app.js densityColor(t) -- 다크 DENSITY_STOPS_DARK
fn density(t: f64) -> Col {
    const S: [(f64, [f64; 3]); 4] = [(0.0, [34.0, 56.0, 84.0]), (0.35, [44.0, 82.0, 120.0]), (0.7, [62.0, 118.0, 166.0]), (1.0, [96.0, 156.0, 208.0])];
    let t = t.clamp(0.0, 1.0);
    for w in S.windows(2) {
        let ((t0, c0), (t1, c1)) = (w[0], w[1]);
        if t <= t1 {
            let f = (t - t0) / (t1 - t0);
            let m = |k: usize| round(c0[k] + (c1[k] - c0[k]) * f) as u8;
            return Col::Rgb(m(0), m(1), m(2));
        }
    }
    Col::Rgb(96, 156, 208)
}
/// .lq-cu / .lq-cd = color-mix(in srgb, good|bad 60%, --chart-bg)
const CU: Col = Col::Rgb(65, 125, 93);
const CD: Col = Col::Rgb(153, 99, 95);

/// CH.remain -- None = 빈 상태(«남은 매물 계산 전입니다») · (w, h) = svg 상자(.lq-d) · rv = 보기(여기서 조인다)
pub fn scene(tab: Option<&Value>, win: u32, p: f64, w: f64, h: f64, rv: &mut Option<Rv>) -> Option<(Vec<Prim>, Geo)> {
    let d = tab?;
    let rm = &d["remain"][win.to_string()];
    let cols = rm["cols"].as_array().filter(|c| !c.is_empty())?;
    let all = d["days"].as_array().map_or(&[][..], |v| &v[..]);
    let view = rm["view"].as_u64().unwrap_or(0) as usize;
    let days = &all[if view == 0 { 0 } else { all.len().saturating_sub(view) }..];
    let n = days.len();
    let f = |v: &Value| v.as_f64().unwrap_or(0.0); // null·없음 = 0(JS 곱셈)
    let (prof, sh) = (if w < 300.0 { 36.0 } else { 52.0 }, if h < 240.0 { 26.0 } else { 40.0 });
    let (l, tp, bt) = (44.0, 6.0, 15.0);
    let xr = w - (prof + 6.0);
    let ch = h - bt - sh - tp;
    let v = rem_view(rv, n);
    let cw = (xr - l) / v.span;
    let x = |i: f64| l + (i - v.i0 + 0.5) * cw;
    let (a, b) = (0i64.max(v.i0.floor() as i64), (n as i64 - 1).min((v.i0 + v.span).ceil() as i64 - 1));
    let vis = if b >= a { &days[a as usize..=b as usize] } else { days };
    let lo0 = vis.iter().map(|q| f(&q[3])).fold(f64::INFINITY, f64::min) * 0.985;
    let hi0 = vis.iter().map(|q| f(&q[2])).fold(f64::NEG_INFINITY, f64::max) * 1.015;
    let (md, hf) = ((lo0 + hi0) / 2.0 + v.y_off, (hi0 - lo0) / 2.0 * v.yk);
    let (lo, hi) = (md - hf, md + hf);
    let y = |q: f64| tp + ch * (1.0 - (q - lo) / (hi - lo));
    let g = Geo { l, xr, tp, ch, cw, k: (hi - lo) / ch, lo, hi, i0: v.i0, n, h };
    let line = |x1, y1, x2, y2, c, a, clip| Prim::Line { pts: vec![[x1, y1], [x2, y2]], c, a, lw: 1.0, dash: None, clip };
    let ax = |x, y, s: String, anchor, t: bool, halo, clip| Prim::Mono { x, y, s, size: 10.0, wt: if t { 700 } else { 500 }, c: if t { Col::Text } else { Col::Muted }, a: 1.0, anchor, halo, clip };
    let rect = |x, y, w, h, c, a, clip| Prim::Rect { x, y, w, h, c, a, rx: 0.0, clip };
    let mut s = vec![];
    if let Some(st) = nice_step(hi - lo, 5.0) {
        let mut q = (lo / st).ceil() * st;
        while q < hi { s.extend([line(l, y(q), xr, y(q), Col::SoftLine, 1.0, Clip::No), ax(l - 5.0, y(q) + 3.0, px(q), Anchor::End, false, false, Clip::No)]); q += st; }
    }
    // 히트맵(heatUrl): 칸 = 정수 px(왼 floor · 위 floor · 아래 ceil · 높이 최소 1) · 투명도 .85×t^2.5 · densityColor
    let x0 = l - v.i0 * cw;
    for (i, c) in cols[cols.len().saturating_sub(n)..].iter().enumerate() {
        let Some(c) = c.as_array().filter(|c| !c.is_empty()) else { continue };
        let (xa, xb) = ((x0 + i as f64 * cw).floor(), (x0 + (i + 1) as f64 * cw).floor());
        let c0 = f(&c[0]);
        for (k, q) in c.iter().enumerate().skip(1) {
            let t = f(q) / 1000.0;
            if !(t > 0.0) || xb < l || xa > xr { continue; }
            let cell = c0 + k as f64 - 1.0;
            let rl = |r: f64| rm["lo"].as_f64().unwrap_or(f64::NAN) + r * rm["row"].as_f64().unwrap_or(f64::NAN);
            let (top, bot) = (y(rl(cell + 1.0)).floor(), y(rl(cell)).ceil());
            s.push(rect(xa, top, xb - xa, (bot - top).max(1.0), density(t), 0.85 * t.powf(2.5), Clip::Plot));
        }
    }
    let (nf, wf) = (n as f64, win as f64);
    for i in a..=b {
        let (dd, xi) = (&days[i as usize], x(i as f64));
        let (o, c) = (f(&dd[1]), f(&dd[4]));
        s.push(line(xi, y(f(&dd[2])), xi, y(f(&dd[3])), Col::Muted, 1.0, Clip::Lane));
        s.push(rect(xi - cw * 0.3, y(o.max(c)), (cw * 0.6).max(1.0), (y(o) - y(c)).abs().max(1.0), if c >= o { CU } else { CD }, 1.0, Clip::Lane));
    }
    if wf < nf - 1.0 {
        let xs = x(nf - 1.0 - wf);
        s.push(Prim::Dashed { pts: vec![[xs, tp], [xs, tp + ch]], c: Col::Accent, a: 0.7, lw: 1.0, on: 2.0, off: 3.0, clip: Clip::Lane });
        s.push(ax(xs + 4.0, tp + 11.0, format!("오늘 지도 입력 시작 · {win}일 전"), Anchor::Start, false, true, Clip::Lane));
    }
    let (y0, hh, bw) = (tp + ch + 6.0 + (sh - 6.0) / 2.0, (sh - 6.0) / 2.0 - 1.0, (cw * 0.74).max(0.8));
    for i in a..=b {
        let dd = &days[i as usize];
        let op = if (i as f64) < nf - 1.0 - wf { 0.22 } else { 0.85 };
        let xi = x(i as f64) - bw / 2.0;
        s.push(rect(xi, y0 - hh * f(&dd[6]), bw, hh * f(&dd[6]), Col::Good, op, Clip::Lane));
        s.push(rect(xi, y0, bw, hh * f(&dd[5]), Col::Bad, op, Clip::Lane));
    }
    if p > lo && p < hi {
        s.push(Prim::Dashed { pts: vec![[l, y(p)], [xr, y(p)]], c: Col::Accent, a: 1.0, lw: 1.2, on: 4.0, off: 3.0, clip: Clip::No });
        s.push(ax(xr + 7.0, y(p) - 3.0, px(p), Anchor::Start, true, true, Clip::No));
    }
    let pr: Vec<&Value> = rm["prof"].as_array().into_iter().flatten().filter(|q| f(&q[0]) > lo && f(&q[0]) < hi).collect();
    let pm = pr.iter().map(|q| f(&q[2])).fold(1e-9, f64::max);
    let bh = (ch / ((hi - lo) / (p * 0.01)) - 1.0).max(1.5);
    for q in pr {
        let side = q[1].as_f64().is_some_and(|v| v != 0.0) || q[1] == true;
        s.push(rect(xr + 5.0, y(f(&q[0])) - bh / 2.0, (prof - 10.0) * f(&q[2]) / pm, bh, if side { Col::Good } else { Col::Bad }, 0.8, Clip::No));
    }
    s.push(line(l, y0, xr, y0, Col::Neutral, 0.38, Clip::No));
    s.push(ax(l - 5.0, y0 - hh / 2.0 + 3.0, "위".into(), Anchor::End, false, false, Clip::No));
    s.push(ax(l - 5.0, y0 + hh / 2.0 + 3.0, "아래".into(), Anchor::End, false, false, Clip::No));
    if b >= a {
        let nl = 4.min(b - a + 1);
        for q in 0..nl {
            let di = if nl < 2 { a } else { round(a as f64 + (b - a) as f64 * q as f64 / (nl - 1) as f64) as i64 };
            let lab: String = days[di as usize][0].as_str().unwrap_or("").chars().skip(5).collect();
            s.push(ax(l.max(xr.min(x(di as f64))), h - 3.0, lab, if q == 0 { Anchor::Start } else if q == nl - 1 { Anchor::End } else { Anchor::Middle }, false, false, Clip::No));
        }
    }
    Some((s, g))
}

/// 자르기 상자: (히트맵 = 그림 칸 · 묶음 = svg clipPath lqRemClip -- 아래 띠까지)
pub fn clips(g: &Geo) -> ([f64; 4], [f64; 4]) { ([g.l, g.tp, g.xr - g.l, g.ch], [g.l, g.tp, g.xr - g.l, g.h - 15.0 - g.tp]) }

/// el._xh: 마우스가 그림 칸 안이면 가로 가격선 + 그 날 세로선 + 왼쪽 가격 알약
pub fn xh(g: &Geo, x: f64, y: f64) -> Vec<Prim> {
    if !(x >= g.l && x <= g.xr && y >= g.tp && y <= g.tp + g.ch) { return vec![]; }
    let di = ((x - g.l) / g.cw + g.i0).floor().min(g.n as f64 - 1.0).max(0.0);
    let xi = g.l + (di - g.i0 + 0.5) * g.cw;
    let dl = |a: [f64; 2], b: [f64; 2]| Prim::Dashed { pts: vec![a, b], c: Col::Muted, a: 0.75, lw: 1.0, on: 2.0, off: 3.0, clip: Clip::No };
    vec![dl([g.l, y], [g.xr, y]), dl([xi, g.tp], [xi, g.h - 15.0]),
         Prim::Rect { x: 1.0, y: y - 8.0, w: g.l - 3.0, h: 16.0, c: Col::Accent, a: 1.0, rx: 3.0, clip: Clip::No },
         Prim::Mono { x: (g.l - 2.0) / 2.0, y: y + 3.5, s: px(g.lo + (g.hi - g.lo) * (1.0 - (y - g.tp) / g.ch)), size: 10.5, wt: 700, c: Col::OnFill, a: 1.0, anchor: Anchor::Middle, halo: false, clip: Clip::No }]
}

fn sgn(x: f64, d: usize) -> String { format!("{}{}", if x >= 0.0 { "+" } else { "−" }, crate::data::to_fixed(x.abs(), d)) }
fn pct(x: f64) -> String { if x.is_finite() { pc(x, 0) } else { "NaN%".into() } }

/// remainKpi: (제목, 값, 아래 글, 값 색)
pub fn kpi(tab: Option<&Value>, win: u32, p: f64) -> Vec<(String, String, String, Col)> {
    let Some(r) = tab.map(|t| &t["remain"][win.to_string()]).filter(|r| !r["ll"].is_null()) else { return vec![] };
    let f = |k: &str| r[k].as_f64().unwrap_or(f64::NAN);
    let mut o = vec![("아래(롱) 남음".into(), pct(f("ll") / f("lm")), "만든 양 대비".into(), Col::Bad), ("위(숏) 남음".into(), pct(f("ul") / f("um")), "만든 양 대비".into(), Col::Good)];
    for (k, lab) in [("lb", "남은 매물 아래 최대"), ("ub", "남은 매물 위 최대")] {
        let q = f(k);
        if q.is_finite() && q != 0.0 { o.push((lab.into(), px(q), format!("{}%", sgn((q / p - 1.0) * 100.0, 1)), Col::Text)); }
    }
    o.push(("±3% 안".into(), pct(f("near3")), format!("7일보다 오래됨 {}", pct(f("old7"))), Col::Text));
    o
}

/// 지표 줄 높이(.lq-kp: 위아래 1 + 11px 한 줄 13 + 18px/1.25 + 10.5px 한 줄 13 · Pretendard 줄 높이 normal = round(.952·크기)+round(.241·크기)) · 없으면 0
pub fn kpi_h(k: &[(String, String, String, Col)]) -> f64 { if k.is_empty() { 0.0 } else { 50.5 } }
/// .lq-rib(같은 폭 열 · 글은 칸 폭에서 … 로 자름) -- y = 0 이 줄 위
pub fn ribbon(k: &[(String, String, String, Col)], w: f64, measure: &dyn Fn(&str, f64, u16) -> f64) -> Vec<Prim> {
    let cw = w / k.len().max(1) as f64;
    let cut = |s: &str, z: f64| -> String {
        let m = cw - 8.0;
        if measure(s, z, 400) <= m { return s.into(); }
        let mut t: String = s.into();
        while !t.is_empty() && measure(&format!("{t}…"), z, 400) > m { t.pop(); }
        format!("{t}…")
    };
    let tx = |x, y, s: String, size, c| Prim::Text { x, y, s, size, wt: 400, c, a: 1.0, anchor: Anchor::Start, halo: false, clip: Clip::No };
    let mut v = vec![];
    for (i, (lab, val, sub, c)) in k.iter().enumerate() {
        let x = i as f64 * cw;
        v.push(tx(x, 1.0 + 10.0, cut(lab, 11.0), 11.0, Col::Muted));
        v.push(Prim::Mono { x, y: 14.0 + 17.73, s: val.clone(), size: 18.0, wt: 700, c: *c, a: 1.0, anchor: Anchor::Start, halo: false, clip: Clip::No });
        if !sub.is_empty() { v.push(tx(x, 36.5 + 10.0, cut(sub, 10.5), 10.5, Col::Muted)); }
    }
    v
}

/// lqRemLeg(.lq-leg: 10.5px --muted · 줄 1.4 · 칸 사이 가로 10 세로 2 · 넘치면 다음 줄) -- y0 = 범례 윗변 · 반환 (그림, 높이)
/// age_h = 계산 뒤 지난 시간(3시간 넘으면 «N시간 전 계산» --warn)
pub fn legend(win: u32, age_h: Option<f64>, w: f64, y0: f64, measure: &dyn Fn(&str, f64, u16) -> f64) -> (Vec<Prim>, f64) {
    let mut items = vec![(format!("칸 = 그날 마감 기준 지난 {win}일 입력 중 아직 안 닿은 청산가 · 진할수록 많음"), Col::Muted), ("오른쪽 = 오늘 남은 양".into(), Col::Muted), ("아래 띠 = 그 날 만든 양 중 지금까지 남은 비율".into(), Col::Muted)];
    if let Some(a) = age_h.filter(|a| *a > 3.0) { items.push((format!("{}시간 전 계산", crate::data::to_fixed(a, 0)), Col::Amber)); }
    let (lh, mut x, mut row, mut v) = (10.5 * 1.4, 0.0, 0.0, vec![]);
    for (t, c) in items {
        let tw = measure(&t, 10.5, 400);
        if x > 0.0 && x + tw > w { x = 0.0; row += 1.0; }
        v.push(Prim::Text { x, y: y0 + row * (lh + 2.0) + lh / 2.0 + 10.5 * 0.36, s: t, size: 10.5, wt: 400, c, a: 1.0, anchor: Anchor::Start, halo: false, clip: Clip::No });
        x += tw + 10.0;
    }
    (v, (row + 1.0) * lh + row * 2.0)
}

#[cfg(test)]
mod t {
    use super::*;
    use serde_json::json;

    fn fx(f: &str) -> Value { serde_json::from_str(&std::fs::read_to_string(format!("{}/compare/fixture/{f}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap() }
    fn cname(c: Col) -> Value {
        json!(match c { Col::SoftLine => "soft-line", Col::Neutral => "neutral", Col::Accent => "accent", Col::Muted => "muted", Col::Text => "text", Col::OnFill => "on-fill", Col::Bad => "bad", Col::Good => "good",
                        CU => "cu", CD => "cd", Col::Rgb(r, g, b) => return json!([r, g, b]), _ => "?" })
    }
    fn cl(c: &Clip) -> &'static str { match c { Clip::No => "no", Clip::Lane => "lane", Clip::Plot => "plot" } }
    fn an(a: &Anchor) -> &'static str { match a { Anchor::Start => "start", Anchor::Middle => "middle", Anchor::End => "end" } }
    /// 그리기 목록 → 정답지 모양(compare/rem_js.mjs parse 와 같은)
    fn norm(v: &[Prim]) -> Value {
        Value::Array(v.iter().map(|p| match p {
            Prim::Rect { x, y, w, h, c: c @ Col::Rgb(..), a, clip: Clip::Plot, .. } if !matches!(*c, CU | CD) => json!(["H", x, y, w, h, cname(*c), a, "plot"]),
            Prim::Line { pts, c, a, lw, clip, .. } => json!(["L", pts[0][0], pts[0][1], pts[1][0], pts[1][1], cname(*c), a, lw, null, cl(clip)]),
            Prim::Dashed { pts, c, a, lw, on, off, clip } => json!(["L", pts[0][0], pts[0][1], pts[1][0], pts[1][1], cname(*c), a, lw, [on, off], cl(clip)]),
            Prim::Mono { x, y, s, size, wt, c, anchor, halo, clip, .. } => json!(["T", x, y, s, size, wt, cname(*c), an(anchor), halo, cl(clip)]),
            Prim::Rect { x, y, w, h, c, a, rx, clip } => json!(["R", x, y, w, h, cname(*c), a, rx, cl(clip)]),
            _ => json!(["?"]),
        }).collect())
    }
    /// 숫자는 1e-6 안 · 글자·색·차례는 그대로
    fn same(a: &Value, b: &Value, at: &str) {
        match (a, b) {
            (Value::Number(x), Value::Number(y)) => assert!((x.as_f64().unwrap() - y.as_f64().unwrap()).abs() < 1e-6, "{at}: {x} 대 {y}"),
            (Value::Array(x), Value::Array(y)) => { assert_eq!(x.len(), y.len(), "{at}: 길이"); for (i, (p, q)) in x.iter().zip(y).enumerate() { same(p, q, &format!("{at}/{i}")); } }
            (Value::Object(x), Value::Object(y)) => { assert_eq!(x.len(), y.len(), "{at}: 열쇠"); for (k, p) in x { same(p, &y[k], &format!("{at}.{k}")); } }
            _ => assert_eq!(a, b, "{at}"),
        }
    }
    fn evs(c: &Value) -> Vec<Ev> {
        c["ev"].as_array().into_iter().flatten().map(|q| { let n = |i: usize| q[i].as_f64().unwrap_or(0.0); match q[0].as_str().unwrap() {
            "down" => Ev::Down(n(1), n(2)), "move" => Ev::Move(n(1), n(2)), "up" => Ev::Up, "dbl" => Ev::Dbl,
            _ => Ev::Wheel { x: n(1), y: n(2), dx: n(3), dy: n(4), shift: q[5] == true } } }).collect()
    }
    /// 정답지 한 입력을 Rust 로: 그림 → (이벤트 → 다시 그림)… → 마지막 그림 · 걸음마다 보기 · 십자선 · 지표
    fn run(c: &Value) -> Value {
        let (tab, (w, h), win) = (Some(&c["tab"]).filter(|t| t["ok"] == true), (c["w"].as_f64().unwrap(), c["h"].as_f64().unwrap()), c["win"].as_u64().unwrap() as u32);
        let p = crate::liq::price(c["price"].as_f64().unwrap(), tab);
        let mut r = Rem { win, ..Default::default() };
        let draw = |r: &mut Rem| { let o = scene(tab, win, p, w, h, &mut r.rv); r.geo = o.as_ref().map(|q| q.1); o };
        let mut out = draw(&mut r);
        let mut steps = vec![];
        for e in evs(c) { r.event(e); out = draw(&mut r); steps.push(r.rv.map_or(Value::Null, |v| json!([v.n, v.i0, v.span, v.y_off, v.yk]))); }
        let xs: Vec<Value> = c["xh"].as_array().into_iter().flatten().map(|q| norm(&xh(&out.as_ref().unwrap().1, q[0].as_f64().unwrap(), q[1].as_f64().unwrap()))).collect();
        let k: Vec<Value> = kpi(tab, win, p).into_iter().map(|(a, b, s, c)| json!([a, b, s, match c { Col::Bad => "lg", Col::Good => "sh", _ => "" }])).collect();
        match out {
            None => json!({"prims": null, "clip": null, "kpi": k, "steps": steps, "xh": xs}),
            Some((v, g)) => json!({"prims": norm(&v), "clip": clips(&g).1, "kpi": k, "steps": steps, "xh": xs}),
        }
    }

    /// 지금 화면의 실제 함수로 만든 정답지(compare/rem_js.mjs → fixture/rem_js.json · 같은 입력 rem_in.json)
    #[test]
    fn js_parity() {
        let (inp, js) = (fx("rem_in.json"), fx("rem_js.json"));
        let (mut cells, mut steps) = (0, 0);
        for (k, (c, j)) in inp.as_array().unwrap().iter().zip(js.as_array().unwrap()).enumerate() {
            let r = run(c);
            same(&r, j, &format!("입력 {k}"));
            cells += j["prims"].as_array().map_or(0, |v| v.iter().filter(|q| q[0] == "H").count());
            steps += j["steps"].as_array().unwrap().len();
        }
        assert!(inp.as_array().unwrap().len() >= 9 && cells > 10_000 && steps >= 20, "{cells} {steps}");
    }

    /// 양성 대조: 계산 갈래 하나씩 바꾸면 정답지와 어긋난다(시험이 실제로 무는지 -- 원본 쪽 변이는 커밋 메시지)
    #[test]
    fn controls() {
        let (inp, js) = (fx("rem_in.json"), fx("rem_js.json"));
        let bites = |j: &Value, r: Value| std::panic::catch_unwind(|| same(&r, j, "")).is_err();
        // ① 다른 기간(win)으로 그리면 시작선·띠 투명도가 달라진다
        let mut c = inp[1].clone(); c["win"] = json!(30);
        c["tab"]["remain"]["30"] = c["tab"]["remain"]["7"].clone();
        assert!(bites(&js[1]["prims"], run(&c)["prims"].clone()), "① 기간");
        // ② 이벤트를 하나 빼면(첫 휠) 보기 걸음이 달라진다
        let mut c = inp[7].clone(); c["ev"].as_array_mut().unwrap().remove(5);
        assert!(bites(&js[7]["steps"], run(&c)["steps"].clone()), "② 휠");
        // ③ 실시간 가격이 다르면 가격선·지표 % 가 달라진다
        let mut c = inp[0].clone(); c["price"] = json!(3100);
        assert!(bites(&js[0]["kpi"], run(&c)["kpi"].clone()), "③ 가격");
        // ④ 칸 색: densityColor 정지점 중간 값
        assert_eq!((density(0.0), density(0.35), density(0.5), density(1.0), density(2.0)), (Col::Rgb(34, 56, 84), Col::Rgb(44, 82, 120), Col::Rgb(52, 97, 140), Col::Rgb(96, 156, 208), Col::Rgb(96, 156, 208)));
        // ⑤ 같은 입력이면 맞는다
        same(&run(&inp[1]), &js[1], "⑤");
    }

    /// 보기 조이기 · 4px 안 끌기는 무시 · 두 번 클릭 = 처음 보기 · 범례 줄바꿈·늦은 계산 경고 · 지표 없는 입력
    #[test]
    fn view_and_legend() {
        let mut rv = Some(Rv { n: 10, i0: 99.0, span: 1.0, y_off: 0.0, yk: 50.0 });
        assert_eq!(rem_view(&mut rv, 10), Rv { n: 10, i0: 7.0, span: 5.0, y_off: 0.0, yk: 10.0 });
        assert_eq!(rem_view(&mut rv, 12).span, 12.0, "날 수가 바뀌면 처음 보기");
        let g = Geo { l: 44.0, xr: 700.0, tp: 6.0, ch: 300.0, cw: 10.0, k: 1.0, lo: 0.0, hi: 1.0, i0: 0.0, n: 12, h: 400.0 };
        let mut r = Rem { win: 60, rv, drag: None, geo: Some(g) };
        r.event(Ev::Down(100.0, 100.0)); r.event(Ev::Move(103.0, 97.0));
        assert_eq!(r.rv.unwrap().i0, 0.0, "4px 안");
        r.event(Ev::Move(80.0, 100.0)); r.event(Ev::Up); r.event(Ev::Move(0.0, 0.0));
        assert_eq!(r.rv.unwrap().i0, 2.0);
        r.event(Ev::Dbl);
        assert!(r.rv.is_none());
        let m = |s: &str, _: f64, _: u16| s.chars().count() as f64 * 6.0;
        let (v, h) = legend(60, Some(4.4), 400.0, 0.0, &m);
        assert!(h > 20.0 && matches!(&v[3], Prim::Text { s, c: Col::Amber, .. } if s == "4시간 전 계산"));
        assert_eq!(legend(7, Some(2.0), 4000.0, 0.0, &m).0.len(), 3);
        assert!(kpi(Some(&json!({"remain": {"60": {"cols": [[1, 5]]}}})), 60, 3000.0).is_empty());
    }
}
