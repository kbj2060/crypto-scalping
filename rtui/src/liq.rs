//! «청산» 탭 청산맵 한 장(지도 1·7·30·60·90·180·365일 -- dashboard/live/liq_tab.js CH.ladder 206~274 · mapOf 82~99 · cumAt 100 · ladRange 201~205 · 범례 lqLadLeg 484~486 ·
//!   지나간 칸 지우기 = liq_profile.js sweep 27~40 · liveTp 41~53 · livePrice 54~58).
//! 자료 = 같은 출처: 1일 /api/liquidation-map(30초 -- app.js LIQUIDATION_MAP_POLL_MS) · 7일 이상 /api/liquidation-map/tiers?days=(60초) · /api/liq-tab(10분 · 일봉 = 가로 범위 · 가격 대체값) ·
//!   허브 «모든 종목» 가격(latestLivePriceByAsset) · 지나간 칸 = 지도 뒤 5분봉 고저(/api/market-history).
//! 안 옮긴 것: 머리(제목 «청산맵» · «추정» 꼬리표 · 원천 버튼 1일…1년·실측 7일·HL 고래 -- 기간은 ?lad= 로 고정) · 실측 7일·HL 고래 원천 · 호버 툴팁(el._hit) ·
//!   «내 청산가» 선·범례(계좌 포지션) · 시장 맥락 mid 가격 대체(허브 가격 → liq-tab 가격만) · 좁은 화면 범례 줄바꿈.
//! ponytail: 지나간 범위는 그릴 때마다 지금가를 한 표본으로 넣는다(원본 = 5초 tick·풋프린트 기둥이 부를 때) -- 더 자주 지운다. 같게 하려면 표본을 5초 tick 에만.
use crate::data::{Candle, iso_ms, to_fixed};
use crate::fp::{Anchor, Clip, Col, Prim};
use serde_json::Value;
use std::collections::HashMap;

const LADDER_R: f64 = 6.0;
const FINE: usize = 1000;
const LEV: [Col; 4] = [Col::Lev10, Col::Lev25, Col::Lev50, Col::Amber];

/// 지나간 칸을 지운 지도(liveTp 결과)
pub struct Tp { lo: i64, bw: f64, share: bool, tiers: Vec<Vec<f64>> }

/// liveTp + sweep: m = 지도 응답(1일 = /api/liquidation-map · 7일+ = /tiers) · sw = 이 지도에서 가격이 지나간 [저, 고](새 지도가 오면 부르는 쪽이 None 으로) ·
/// live = 허브 가격(0 = 없음 → 지도 기준가) · cs = 5분봉(지도 뒤에 시작한 봉의 고저를 더한다)
pub fn live_tp(m: &Value, sw: &mut Option<[f64; 2]>, live: f64, cs: &[Candle]) -> Option<Tp> {
    let tp = &m["tier_profile"];
    if !tp.is_object() { return None; }
    let f = |v: &Value| v.as_f64().unwrap_or(f64::NAN);
    let cp = f(&tp["current_price"]);
    let p = if live > 0.0 { live } else { cp };
    let w = sw.get_or_insert([cp, cp]);
    let (mut lo, mut hi) = (w[0].min(p), w[1].max(p));
    let s = [&tp["known_until"], &m["generated_at"]].into_iter().filter_map(|v| v.as_str()).find(|s| !s.is_empty()).unwrap_or("");
    let t0 = iso_ms(s).map_or(f64::NAN, |ms| ms / 1000.0);
    if t0 > 0.0 { for c in cs.iter().rev().take_while(|c| c.time as f64 >= t0) { lo = lo.min(c.low); hi = hi.max(c.high); } }
    *w = [lo, hi];
    let (bw, lo0) = (f(&tp["bin_width"]), tp["lo"].as_i64().unwrap_or(0));
    let (k0, k1) = ((lo / bw).ceil(), (hi / bw).floor());
    let tiers = tp["tiers"].as_array().into_iter().flatten().map(|t| t["values"].as_array().into_iter().flatten().enumerate()
        .map(|(i, v)| { let k = (lo0 + i as i64) as f64; if k >= k0 && k <= k1 { 0.0 } else { f(v) } }).collect()).collect();
    Some(Tp { lo: lo0, bw, share: tp["unit"] == "share", tiers })
}

/// JS 객체 키 차례(정수 키: 0 이상은 오름차순 먼저 · 음수는 넣은 차례 뒤)
fn js_order<T>(v: Vec<(i64, T)>) -> Vec<(i64, T)> {
    let (mut pos, neg): (Vec<_>, Vec<_>) = v.into_iter().partition(|q| q.0 >= 0);
    pos.sort_by_key(|q| q.0);
    pos.extend(neg);
    pos
}
/// Math.round
pub(crate) fn round(x: f64) -> f64 { (x + 0.5).floor() }

/// mapOf: 0.25% 칸(rows · JS 키 차례) · 0.1% 누적(dn·up, ±100%)
pub struct Map { rows: Vec<(i64, [f64; 4])>, dn: Vec<f64>, up: Vec<f64>, share: bool }
pub fn map_of(tp: &Tp, p: f64) -> Option<Map> {
    if !(p > 0.0) { return None; }
    let (mut rows, mut at): (Vec<(i64, [f64; 4])>, HashMap<i64, usize>) = (vec![], HashMap::new());
    let (mut dn, mut up) = (vec![0.0; FINE + 1], vec![0.0; FINE + 1]);
    for (ti, t) in tp.tiers.iter().enumerate() {
        for (k, &v) in t.iter().enumerate() {
            if !(v > 0.0) { continue; }
            let d = ((tp.lo + k as i64) as f64 * tp.bw / p - 1.0) * 100.0;
            if d.abs() > FINE as f64 / 10.0 + 0.1 { continue; }
            let g = round(d / 0.25) as i64;
            if g != 0 && ti < 4 { let i = *at.entry(g).or_insert_with(|| { rows.push((g, [0.0; 4])); rows.len() - 1 }); rows[i].1[ti] += v; }
            let fi = FINE.min((d.abs() / 0.1).ceil() as usize);
            if d < 0.0 { dn[fi] += v } else { up[fi] += v }
        }
    }
    for i in 1..=FINE { dn[i] += dn[i - 1]; up[i] += up[i - 1]; }
    Some(Map { rows: js_order(rows), dn, up, share: tp.share })
}
fn cum_at(m: &Map, d: f64) -> f64 { let i = FINE.min(round(d.abs() / 0.1) as usize); if d < 0.0 { m.dn[i] } else { m.up[i] } }

pub(crate) fn usd(x: f64) -> String {
    if x.is_nan() { return "—".into(); }
    let (a, s) = (x.abs(), if x < 0.0 { "−" } else { "" });
    if a >= 1e9 { return format!("{s}${}B", to_fixed(a / 1e9, if a >= 1e10 { 0 } else { 1 })); }
    if a >= 1e6 { return format!("{s}${}M", to_fixed(a / 1e6, if a >= 1e8 { 0 } else { 1 })); }
    if a >= 1e3 { return format!("{s}${}K", to_fixed(a / 1e3, 0)); }
    format!("{s}${}", to_fixed(a, 0))
}
pub(crate) fn pc(x: f64, d: usize) -> String { format!("{}%", to_fixed(x * 100.0, d)) }
pub(crate) fn px(p: f64) -> String { crate::mc::n(p, if p >= 1000.0 { 0 } else if p >= 100.0 { 1 } else if p >= 1.0 { 2 } else { 4 }) }
pub(crate) fn nice_step(v: f64, n: f64) -> Option<f64> {
    let r = v / n;
    let m = 10f64.powf((if r == 0.0 || r.is_nan() { 1.0 } else { r }).log10().floor());
    [1.0, 2.0, 2.5, 5.0, 10.0].into_iter().map(|k| k * m).find(|&k| k >= r)
}
pub(crate) fn ticks(mx: f64, n: f64) -> Vec<f64> {
    let (mut o, Some(st)) = (vec![], nice_step(mx, n)) else { return vec![] };
    let mut v = st;
    while v <= mx * 1.001 { o.push(v); v += st; }
    o
}
/// toFixed(1) 을 숫자로(svg 속성 글자 그대로)
pub(crate) fn f1(x: f64) -> f64 { to_fixed(x, 1).parse().unwrap_or(x) }

/// ladRange: 가로 = 고른 기간 일봉 저가~고가(±1% 여유 · 최소 ±6% · 끝 ±100%) -- days = /api/liq-tab days([날, 시, 고, 저, 종, …])
fn lad_range(days: &[Value], nd: usize, p: f64) -> (f64, f64) {
    let (mut lo, mut hi, cap) = (p * (1.0 - LADDER_R / 100.0), p * (1.0 + LADDER_R / 100.0), FINE as f64 / 10.0);
    for d in &days[days.len().saturating_sub(nd)..] { lo = lo.min(d[3].as_f64().unwrap_or(f64::NAN)); hi = hi.max(d[2].as_f64().unwrap_or(f64::NAN)); }
    ((-cap).max((lo / p - 1.0) * 100.0 - 1.0), cap.min((hi / p - 1.0) * 100.0 + 1.0))
}

/// liq_tab.js price(): 허브 가격 → liq-tab 가격(시장 맥락 mid 는 안 옮김)
pub fn price(live: f64, tab: Option<&Value>) -> f64 { if live > 0.0 { live } else { tab.and_then(|t| t["price"].as_f64()).unwrap_or(0.0) } }

/// CH.ladder(지도 원천) -- None = 빈 상태(«지도를 받는 중입니다») · (w, h) = svg 상자(.lq-d)
pub fn scene(mp: Option<&Map>, tab: Option<&Value>, nd: usize, p: f64, w: f64, h: f64) -> Option<Vec<Prim>> {
    if !(p > 0.0) { return None; }
    let narrow = w < 420.0;
    let (l, r, tp, bt) = (if narrow { 44.0 } else { 62.0 }, if narrow { 44.0 } else { 62.0 }, 26.0, 22.0);
    let (ch, xr) = (h - tp - bt, w - r);
    let days = tab.and_then(|t| t["days"].as_array()).map_or(&[][..], |v| &v[..]);
    let (d_lo, d_hi) = lad_range(days, nd, p);
    let sp = d_hi - d_lo;
    let x = |d: f64| l + (xr - l) * (d - d_lo) / sp;
    let stp = [0.25, 0.5, 1.0, 2.0].into_iter().find(|q| sp / q <= (xr - l) / 3.0).unwrap_or(2.0);
    let bw = ((xr - l) * stp / sp - 1.0).max(1.0);
    let in_r = |d: f64| d >= d_lo && d <= d_hi;
    let g_of = |d: f64| { let r = round(d / 0.25); if r != 0.0 { r.signum() * (r.abs() * 0.25 / stp - 1e-9).ceil() } else { 0.0 } };
    let d_of = |g: f64| if g != 0.0 { g.signum() * ((g.abs() - 1.0) * stp + (stp + 0.25) / 2.0) } else { 0.0 };
    let mp = mp?;
    let fu = |v: f64| if mp.share { pc(v, if v >= 0.1 { 0 } else { 1 }) } else { usd(v) };
    // 묶음 G(쪽마다 stp 폭) -- 키 차례도 JS 객체 그대로
    let (mut bk, mut at): (Vec<(i64, (f64, [f64; 4], f64))>, HashMap<i64, usize>) = (vec![], HashMap::new());
    for (g, parts) in &mp.rows {
        let d = *g as f64 * 0.25;
        if !in_r(d) { continue; }
        let gg = g_of(d);
        let i = *at.entry(gg as i64).or_insert_with(|| { bk.push((gg as i64, (d_of(gg), [0.0; 4], 0.0))); bk.len() - 1 });
        let o = &mut bk[i].1;
        for j in 0..4 { o.1[j] += parts[j]; o.2 += parts[j]; }
    }
    let bars: Vec<(f64, [f64; 4], f64)> = js_order(bk).into_iter().map(|q| q.1).collect();
    let mx = bars.iter().map(|b| b.2).fold(1e-12, f64::max) * 1.08;
    let y = |v: f64| tp + ch * (1.0 - v / mx);
    let line = |x1, x2, y1, y2, c, a, lw| Prim::Line { pts: vec![[x1, y1], [x2, y2]], c, a, lw, dash: None, clip: Clip::No };
    let gl = |x1, x2, y1, y2| line(x1, x2, y1, y2, Col::SoftLine, 1.0, 1.0);
    let ax = |x, y, s: String, anchor, c, wt, halo| Prim::Mono { x, y, s, size: 10.0, wt, c, a: 1.0, anchor, halo, clip: Clip::No };
    let mut s = vec![];
    if !bars.is_empty() { for v in ticks(mx, 4.0) { s.extend([gl(l, xr, y(v), y(v)), ax(l - 6.0, y(v) + 3.0, fu(v), Anchor::End, Col::Muted, 500, false)]); } }
    let (p_lo, p_hi) = (p * (1.0 + d_lo / 100.0), p * (1.0 + d_hi / 100.0));
    if let Some(pst) = nice_step(p_hi - p_lo, if narrow { 6.0 } else { 9.0 }) {
        let mut pq = (p_lo / pst).ceil() * pst;
        while pq < p_hi {
            let xq = x((pq / p - 1.0) * 100.0);
            if (xq - x(0.0)).abs() > 36.0 { s.extend([gl(xq, xq, tp, tp + ch), ax(xq, h - 6.0, px(pq), Anchor::Middle, Col::Muted, 500, false)]); }
            pq += pst;
        }
    }
    s.push(line(l, xr, tp + ch, tp + ch, Col::Neutral, 0.38, 1.0));
    for b in &bars {
        let (bx, mut yy) = (x(b.0) - bw / 2.0, tp + ch);
        for (j, &v) in b.1.iter().enumerate() {
            if !(v > 0.0) { continue; }
            let hv = ch * v / mx;
            yy -= hv;
            s.push(Prim::Rect { x: f1(bx), y: f1(yy), w: f1(bw), h: f1(hv.max(0.0)), c: LEV[j], a: 1.0, rx: 0.0, clip: Clip::No });
        }
    }
    // 누적 = 지금가에서 그 가격까지 · 오른쪽 축
    let (cum_l, cum_s) = (cum_at(mp, d_lo), cum_at(mp, d_hi));
    let cmx = cum_l.max(cum_s).max(1e-12) * 1.08;
    let yc = |v: f64| tp + ch * (1.0 - v / cmx);
    let st = 0.1f64.max(sp / if narrow { 150.0 } else { 400.0 });
    for (z, c, end) in [(-1.0, Col::Bad, -d_lo), (1.0, Col::Good, d_hi)] {
        let mut pts = vec![];
        let mut q = 0.0;
        while q <= end + 1e-9 { pts.push([f1(x(z * q)), f1(yc(cum_at(mp, z * q)))]); q += st; }
        s.push(Prim::Area { pts: pts.clone(), base: tp + ch, c, a: 0.10 });
        s.push(Prim::Line { pts, c, a: 1.0, lw: 2.0, dash: None, clip: Clip::No });
    }
    for v in ticks(cmx, 4.0) { s.push(ax(xr + 6.0, yc(v) + 3.0, fu(v), Anchor::Start, Col::Muted, 500, false)); }
    let head = if narrow { format!("누적 롱 {} · 숏 {}", fu(cum_l), fu(cum_s)) } else {
        format!("누적 롱 청산({}까지) {} · 숏 청산({}까지) {}{}", px(p_lo), fu(cum_l), px(p_hi), fu(cum_s), if cum_l + cum_s > 0.0 { format!(" · 롱 몫 {}", pc(cum_l / (cum_l + cum_s), 0)) } else { String::new() })
    };
    s.push(ax(l + 6.0, 14.0, head, Anchor::Start, Col::Text, 600, true));
    if w >= 760.0 { s.push(ax(xr, 14.0, "막대 = 칸(왼쪽 축) · 선 = 누적(오른쪽 축)".into(), Anchor::End, Col::Muted, 500, true)); }
    let x0 = x(0.0);
    s.push(Prim::Dashed { pts: vec![[x0, tp], [x0, tp + ch]], c: Col::Accent, a: 1.0, lw: 1.2, on: 4.0, off: 3.0, clip: Clip::No });
    s.push(Prim::Rect { x: x0 - 30.0, y: h - bt + 2.0, w: 60.0, h: 16.0, c: Col::Accent, a: 1.0, rx: 3.0, clip: Clip::No });
    s.push(Prim::Mono { x: x0, y: h - bt + 13.5, s: px(p), size: 10.5, wt: 700, c: Col::OnFill, a: 1.0, anchor: Anchor::Middle, halo: false, clip: Clip::No });
    Some(s)
}

/// 빈 상태(.lq-empty: 여백 14/2 · 12px --muted) · 범례(.lq-leg: svg 아래 5 · 10.5px --muted · 칸 사이 10 · 색 칸 9×9 둥근 2 + 4) -- y0 = svg 바닥
/// ponytail: 글자 바닥선 = 줄 가운데 + 0.36×크기 근사(mc.rs 와 같은) · 범례는 한 줄(넓은 화면) -- 좁으면 줄바꿈을 더한다
pub fn empty() -> Vec<Prim> { empty_msg("지도를 받는 중입니다") }
pub fn empty_msg(s: &str) -> Vec<Prim> { vec![Prim::Text { x: 2.0, y: 14.0 + 7.2 + 12.0 * 0.36, s: s.into(), size: 12.0, wt: 400, c: Col::Muted, a: 1.0, anchor: Anchor::Start, halo: false, clip: Clip::No }] }
pub fn legend(y0: f64, measure: &dyn Fn(&str, f64, u16) -> f64) -> Vec<Prim> {
    let (cy, mut x, mut v) = (y0 + 5.0 + 10.5 * 1.4 / 2.0, 0.0, vec![]);
    for (c, t) in [(Some(Col::Lev10), "10배"), (Some(Col::Lev25), "20~25배"), (Some(Col::Lev50), "50배"), (Some(Col::Amber), "75~100배"), (None, "선 = 지금가부터 누적(빨강 롱 · 초록 숏)"), (None, "지도 눈금 = 세 거래소 OI×2 · 절대액 아님")] {
        if let Some(c) = c { v.push(Prim::Rect { x, y: cy + 10.5 * 0.36 - 8.0, w: 9.0, h: 9.0, c, a: 1.0, rx: 2.0, clip: Clip::No }); x += 13.0; } // 바닥선 −1 에 아래 변
        v.push(Prim::Text { x, y: cy + 10.5 * 0.36, s: t.into(), size: 10.5, wt: 400, c: Col::Muted, a: 1.0, anchor: Anchor::Start, halo: false, clip: Clip::No });
        x += measure(t, 10.5, 400) + 10.0;
    }
    v
}

#[cfg(test)]
mod t {
    use super::*;
    use serde_json::json;

    fn fx(f: &str) -> Value { serde_json::from_str(&std::fs::read_to_string(format!("{}/compare/fixture/{f}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap() }
    fn cname(c: Col) -> &'static str {
        match c { Col::SoftLine => "soft-line", Col::Neutral => "neutral", Col::Accent => "accent", Col::Muted => "muted", Col::Text => "text", Col::OnFill => "on-fill",
                  Col::Bad => "bad", Col::Good => "good", Col::Lev10 => "lev10", Col::Lev25 => "lev25", Col::Lev50 => "lev50", Col::Amber => "lev100", _ => "?" }
    }
    /// 그리기 목록 → 정답지 모양(compare/liq_js.mjs parse 와 같은)
    fn norm(v: &[Prim]) -> Value {
        Value::Array(v.iter().map(|p| match p {
            Prim::Line { pts, c, lw, .. } if *lw == 2.0 => json!(["P", pts, cname(*c), lw]), // 누적 곡선
            Prim::Line { pts, c, a, lw, .. } => json!(["L", pts[0][0], pts[0][1], pts[1][0], pts[1][1], cname(*c), a, lw, null]),
            Prim::Dashed { pts, c, a, lw, on, off, .. } => json!(["L", pts[0][0], pts[0][1], pts[1][0], pts[1][1], cname(*c), a, lw, [on, off]]),
            Prim::Mono { x, y, s, size, wt, c, anchor, halo, .. } => json!(["T", x, y, s, size, wt, cname(*c), match anchor { Anchor::Start => "start", Anchor::Middle => "middle", Anchor::End => "end" }, halo, true]),
            Prim::Rect { x, y, w, h, c, a, rx, .. } => json!(["R", x, y, w, h, cname(*c), a, rx]),
            Prim::Area { pts, base, c, a } => { let mut q = pts.clone(); q.push([f1(pts[pts.len() - 1][0]), *base]); q.push([pts[0][0], *base]); json!(["A", q, cname(*c), a]) }
            _ => json!(["?"]),
        }).collect())
    }
    /// 숫자는 1e-6 안 · 글자·색·차례는 그대로
    fn same(a: &Value, b: &Value, at: &str) {
        match (a, b) {
            (Value::Number(x), Value::Number(y)) => assert!((x.as_f64().unwrap() - y.as_f64().unwrap()).abs() < 1e-6, "{at}: {x} 대 {y}"),
            (Value::Array(x), Value::Array(y)) => { assert_eq!(x.len(), y.len(), "{at}: 길이 {a} 대 {b}"); for (i, (p, q)) in x.iter().zip(y).enumerate() { same(p, q, &format!("{at}/{i}")); } }
            _ => assert_eq!(a, b, "{at}"),
        }
    }
    fn candles(v: &Value) -> Vec<Candle> {
        v.as_array().into_iter().flatten().map(|c| Candle { time: c["time"].as_i64().unwrap(), open: 0.0, high: c["high"].as_f64().unwrap(), low: c["low"].as_f64().unwrap(), close: 0.0, partial: None, volume: 0.0, sma: None }).collect()
    }
    /// 정답지 한 입력을 Rust 로: 가격 표본 차례대로 지나간 범위를 쌓고 마지막 그림
    fn run(c: &Value) -> Option<Vec<Prim>> {
        let (tab, cs, mut sw, mut out) = (Some(&c["tab"]).filter(|t| t["ok"] == true), candles(&c["candles"]), None, None);
        for live in c["prices"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()) {
            let p = price(live, tab);
            let mp = live_tp(&c["map"], &mut sw, live, &cs).and_then(|t| map_of(&t, p));
            out = scene(mp.as_ref(), tab, c["days"].as_u64().unwrap() as usize, p, c["w"].as_f64().unwrap(), c["h"].as_f64().unwrap());
        }
        out
    }

    /// 지금 화면의 실제 함수로 만든 정답지(compare/liq_js.mjs → fixture/liq_js.json · 같은 입력 liq_in.json)
    #[test]
    fn js_parity() {
        let (inp, js) = (fx("liq_in.json"), fx("liq_js.json"));
        let (mut n, mut bars) = (0, 0);
        for (k, (c, j)) in inp.as_array().unwrap().iter().zip(js.as_array().unwrap()).enumerate() {
            match run(c) {
                None => assert!(j.is_null(), "입력 {k}: 빈 상태 아님"),
                Some(v) => { let r = norm(&v); same(&r, j, &format!("입력 {k}")); bars += v.iter().filter(|p| matches!(p, Prim::Rect { rx: 0.0, .. })).count(); }
            }
            n += 1;
        }
        assert!(n >= 12 && bars > 2000, "{n} {bars}");
    }

    /// 양성 대조: 입력 갈래를 하나씩 빼면 정답지와 어긋난다(시험이 실제로 무는지 -- 원본·Rust 계산 변이는 커밋 메시지)
    #[test]
    fn controls() {
        let (inp, js) = (fx("liq_in.json"), fx("liq_js.json"));
        let c1 = &inp[1];
        // ① 지나간 범위를 안 쌓으면(가격 표본 마지막 하나만) 칸이 남아 막대가 달라진다
        let (tab, p) = (Some(&c1["tab"]), 3001.0);
        let mp = live_tp(&c1["map"], &mut None, p, &[]).and_then(|t| map_of(&t, p));
        let v = scene(mp.as_ref(), tab, 7, p, 900.0, 520.0).unwrap();
        assert!(std::panic::catch_unwind(|| same(&norm(&v), &js[1], "")).is_err(), "① 지나간 칸");
        // ② 1일 지도: 지도 뒤 봉 고저를 빼면 달라진다
        let c2 = &inp[2];
        let mp = live_tp(&c2["map"], &mut None, 3005.0, &[]).and_then(|t| map_of(&t, 3005.0));
        let v = scene(mp.as_ref(), Some(&c2["tab"]), 1, 3005.0, 1000.0, 560.0).unwrap();
        assert!(std::panic::catch_unwind(|| same(&norm(&v), &js[2], "")).is_err(), "② 봉 고저");
        // ③ 일봉 범위(ladRange)를 안 보면(±6% 만) 가로가 달라진다
        let c7 = &inp[7];
        let mp = live_tp(&c7["map"], &mut None, 3000.0, &[]).and_then(|t| map_of(&t, 3000.0));
        let v = scene(mp.as_ref(), None, 30, 3000.0, 900.0, 520.0).unwrap();
        assert!(std::panic::catch_unwind(|| same(&norm(&v), &js[7], "")).is_err(), "③ 일봉 범위");
        // ④ 같은 입력이면 맞는다(위 셋이 «다른 입력» 때문만은 아님)
        same(&norm(&run(c7).unwrap()), &js[7], "④");
    }

    /// 자리: 좁은 판 = 여백 44 · 현재가 알약 60×16 · 머리는 짧은 글 · 정수 키 차례
    #[test]
    fn layout() {
        assert_eq!(js_order(vec![(-1, ()), (3, ()), (-5, ()), (0, ()), (1, ())]).iter().map(|q| q.0).collect::<Vec<_>>(), vec![0, 1, 3, -1, -5]);
        assert_eq!((usd(1.5e9), usd(2.5e10), usd(-1234.0), usd(999.5), px(3012.4), px(20.0), pc(0.3333, 1)), ("$1.5B".into(), "$25B".into(), "−$1K".into(), "$1000".into(), "3,012".into(), "20.00".into(), "33.3%".into()));
        let tp = Tp { lo: 2900, bw: 1.0, share: false, tiers: vec![(0..200).map(|i| if i % 3 == 0 { 1e5 } else { 0.0 }).collect(); 4] };
        let m = map_of(&tp, 3000.0).unwrap();
        let v = scene(Some(&m), None, 7, 3000.0, 400.0, 300.0).unwrap();
        assert!(v.iter().any(|p| matches!(p, Prim::Mono { s, wt: 600, .. } if s.starts_with("누적 롱 $"))));
        assert!(v.iter().any(|p| matches!(p, Prim::Rect { w, h, rx: 3.0, .. } if *w == 60.0 && *h == 16.0)));
        assert!(!v.iter().any(|p| matches!(p, Prim::Mono { s, .. } if s.starts_with("막대 ="))), "760 미만은 오른쪽 글 없음");
        assert!(scene(Some(&m), None, 7, 0.0, 900.0, 500.0).is_none() && scene(None, None, 7, 3000.0, 900.0, 500.0).is_none());
    }

    /// 1일 = 30초 /api/liquidation-map · 7일+ = 60초 /tiers · liq-tab 10분 · 숨으면 안 묻는다 · 새 지도 = 지나간 범위 처음부터 · 못 받으면 옛 값(1일 실패 = 지도 없음)
    #[test]
    fn poll() {
        use crate::feed::{Feed, Kind, Req, Source, Urls};
        let mut f = Feed::new("eth", "ETHUSDT", Source::Push, Urls::default(), false);
        (f.liq_only, f.liq_days) = (true, 7);
        for (t, hid) in [(1000.0, false), (30_000.0, false), (61_500.0, true), (62_000.0, false)] { f.tick(t, hid, false); }
        let urls: Vec<_> = f.out.iter().filter_map(|r| match r { Req::Get { kind: Kind::LiqTab | Kind::LiqMap, url } => Some(url.clone()), _ => None }).collect();
        assert_eq!(urls, vec!["/api/liq-tab?asset=eth", "/api/liquidation-map/tiers?asset=eth&days=7", "/api/liquidation-map/tiers?asset=eth&days=7"]);
        assert!(f.out.iter().any(|r| matches!(r, Req::WsSend(s) if s == r#"{"subscribe":[]}"#)), "허브 = «모든 종목» 가격만");
        f.on_liq_map(Some(r#"{"days":7,"tier_profile":{"bin_width":1,"lo":2990,"current_price":3000,"unit":"oi_usd","tiers":[{"values":[5,5]}]}}"#));
        f.liq_sw = Some([1.0, 2.0]);
        f.on_liq_map(Some(r#"{"days":7,"tier_profile":null}"#));
        f.on_liq_map(None);
        assert!(f.liq_map.is_some() && f.liq_sw.is_some(), "지도 없는 응답·실패 = 옛 지도");
        f.on_liq_map(Some(r#"{"days":7,"tier_profile":{"lo":1}}"#));
        assert!(f.liq_sw.is_none(), "새 지도 = 범위 처음부터");
        f.on_liq_tab(Some(r#"{"ok":true,"price":3001}"#));
        f.on_liq_tab(Some(r#"{"ok":false}"#));
        assert!(f.liq_tab.is_none(), "ok 아님 = 없음");
        let mut g = Feed::new("eth", "ETHUSDT", Source::Push, Urls::default(), false);
        (g.liq_only, g.liq_days) = (true, 1);
        g.tick(1000.0, false, false);
        assert!(g.out.iter().any(|r| matches!(r, Req::Get { kind: Kind::LiqMap, url } if url == "/api/liquidation-map?asset=eth")));
        g.on_liq_map(Some(r#"{"tier_profile":{"lo":1}}"#));
        g.on_liq_map(None);
        assert!(g.liq_map.is_none(), "1일 실패 = 지도 없음(원본 warmed_up false)");
    }
}
