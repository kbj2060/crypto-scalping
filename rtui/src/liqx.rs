//! «청산» 탭 아랫줄 세 칸(dashboard/live/liq_tab.js 아랫줄 .lq-bot): 오늘 누적(CH.today 109~126 · today 70~80 · toBar 64~68) · 최대 고통(CH.pain 183~209 · painOf 149~170 ·
//!   sweptRange 143~148 · reach·Phi 135~136 · painKpi 231~244 · 범례 lqPainLeg 304) · 코인별 최대 고통(CH.quad 215~230 · coinRows 210~214 · coinTable 245~251).
//! 자료 = 같은 출처: /api/liq-tab?asset=all(10분 fetchAll · 코인 차례 그대로) · /api/liquidation-5m-history?bars=288(30초 fetchBars) · /api/liq-tab(10분 · 띠 band·가격 대체) ·
//!   허브 «모든 종목» 가격(latestLivePriceByAsset) · 지나간 봉우리 = 계산 뒤 5분봉 고저(/api/market-history · 지금 코인만 -- 원본도 다른 코인은 지금가만).
//! 안 옮긴 것: 머리 꼬리표(실측·추정) · 주기 단추 7개(주기는 ?card=liqx1W 처럼 고정 · 기본 1일) · 머리 오른쪽 글(lqTodLab·lqCoLab) · 호버 툴팁(el._hit) ·
//!   좁은 화면(1100px 이하 · 1440px 이하 옆 칸 200) 배치 · 시장 맥락 mid 가격 대체 · 표 old 줄 title.
//! ponytail: 칸 배치(판 폭 = flex 0.9·2.1·1.7 · 표 = 글 폭 잰 값 · 줄 높이 근사)는 원본 CSS 를 손으로 옮긴 것 -- 그리기 요소(svg 안)만 정답지와 1e-6 대조.
use crate::data::{Candle, LiqBar, iso_ms, iso_of, to_fixed};
use crate::draw;
use crate::feed::Kind;
use crate::fp::{Anchor, Clip, Col, Prim};
use crate::liq::{f1, nice_step, pc, px, ticks, usd};
use crate::rem::{CD, CU};
use eframe::egui::{self, Color32, Painter, Pos2, pos2, vec2};
use serde_json::Value;

pub const PERS: [&str; 7] = ["12H", "24H", "48H", "3D", "1W", "2W", "1M"];
pub fn plab(k: &str) -> &'static str { match k { "12H" => "12시간", "24H" => "1일", "48H" => "2일", "3D" => "3일", "1W" => "1주", "2W" => "2주", _ => "1달" } }
const NEST_PCT: f64 = 0.0035;
const BOX: f64 = 0.005;
const STALE_MS: f64 = 7_200_000.0;
const ALL_MS: f64 = 600_000.0; // 9 DATA_POLL_MS
const BARS_MS: f64 = 30_000.0; // 9 BAR_POLL_MS

/// toBar 의 이 칸 몫(t = Date.parse/1000 · by·hl = 바이비트·HL 롱+숏)
pub struct Bar { t: f64, l: f64, s: f64, by: f64, hl: f64 }
fn to_bar(b: &LiqBar) -> Bar {
    let v = |x: &Option<crate::data::LiqVenue>| x.as_ref().map_or(0.0, |x| x.long_usd + x.short_usd);
    Bar { t: iso_ms(&b.ts).map_or(f64::NAN, |m| m / 1000.0), l: b.long_usd, s: b.short_usd, by: v(&b.bybit), hl: v(&b.hl) }
}

/// 이 카드만(feed.liqx): 고른 주기 · 코인별 묶음(S.all.coins -- 응답 차례) · 오늘 5분봉(S.bars) · 마지막으로 물은 시각
pub struct State { pub pp: &'static str, pub coins: Option<Vec<(String, Value)>>, pub bars: Vec<Bar>, all_at: f64, bars_at: f64 }
impl State {
    /// pp = 주기(빈 글자 = 1일 · 모르는 글자 = None)
    pub fn new(pp: &str) -> Option<Self> {
        let pp = PERS.into_iter().find(|k| *k == if pp.is_empty() { "24H" } else { pp })?;
        Some(Self { pp, coins: None, bars: vec![], all_at: f64::NEG_INFINITY, bars_at: f64::NEG_INFINITY })
    }
    /// tick(5초) 의 fetchAll(10분) · fetchBars(30초)
    pub fn due(&mut self, now: f64, asset: &str) -> Vec<(Kind, String)> {
        let mut o = vec![];
        if now - self.all_at >= ALL_MS { self.all_at = now; o.push((Kind::LiqAll, "/api/liq-tab?asset=all".into())); }
        if now - self.bars_at >= BARS_MS { self.bars_at = now; o.push((Kind::LiqBars, format!("/api/liquidation-5m-history?asset={asset}&bars=288"))); }
        o
    }
    /// fetchAll: ok 면 받고 · 실패·ok 아님 = 옛 값
    pub fn on_all(&mut self, body: Option<&str>) {
        #[derive(serde::Deserialize)]
        struct All { #[serde(default)] ok: Value, #[serde(default)] coins: Option<Coins> }
        let Some(a) = body.and_then(|b| serde_json::from_str::<All>(b).ok()) else { return };
        if a.ok == true { self.coins = Some(a.coins.map_or(vec![], |c| c.0)); }
    }
    /// fetchBars: bars 가 배열이면 받고 · 아니면 옛 값
    pub fn on_bars(&mut self, body: Option<&str>) {
        #[derive(serde::Deserialize)]
        struct H { bars: Option<Vec<LiqBar>> }
        if let Some(v) = body.and_then(|b| serde_json::from_str::<H>(b).ok()).and_then(|h| h.bars) { self.bars = v.iter().map(to_bar).collect(); }
    }
}
/// JSON 객체를 차례 그대로(serde_json 기본 Map 은 글자 차례로 줄 세운다 -- 원본 Object.keys = 넣은 차례)
struct Coins(Vec<(String, Value)>);
impl<'de> serde::Deserialize<'de> for Coins {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = Coins;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("object") }
            fn visit_map<A: serde::de::MapAccess<'de>>(self, mut m: A) -> Result<Coins, A::Error> {
                let mut v = vec![];
                while let Some(e) = m.next_entry::<String, Value>()? { v.push(e); }
                Ok(Coins(v))
            }
        }
        d.deserialize_map(V)
    }
}

fn f(v: &Value) -> f64 { v.as_f64().unwrap_or(f64::NAN) }
fn sgn(x: f64, d: usize) -> String { format!("{}{}", if x >= 0.0 { "+" } else { "−" }, to_fixed(x.abs(), d)) }
fn amt(share: bool, v: f64) -> String { if share { format!("{} 몫", pc(v, 1)) } else { usd(v) } }
fn phi(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let p = 0.3989423 * (-x * x / 2.0).exp() * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    if x > 0.0 { 1.0 - p } else { p }
}
fn reach(z: f64) -> f64 { 1f64.min(2.0 * (1.0 - phi(z))) }

/// 봉우리 하나(alt = 1등 금액의 90% 이상인 2등 [가격, 금액, 거리 %])
#[derive(Clone, Copy, Debug)]
pub struct Lev { pub px: f64, pub amt: f64, pub d: f64, pub z: f64, pub p: f64, pub alt: Option<[f64; 3]> }
#[derive(Clone, Copy, Debug)]
pub struct Per { pub dn: Option<Lev>, pub up: Option<Lev>, pub pull: Option<f64>, pub near: Option<&'static str>, pub pincer: bool }
/// painOf: 주기별(PERS 차례) · 중첩 [아래, 위] = (개수, 가격)
pub struct Pain { pub per: [Option<Per>; 7], pub nest: [(usize, f64); 2] }

/// sweptRange: 계산(k3 끝 + 1시간 · 없으면 계산 시각의 시) 뒤 5분봉 고저 + 지금가 → (저, 고)
fn swept(e: &Value, p: f64, cs: &[Candle]) -> (f64, f64) {
    let t0 = match e["k3"].as_array().filter(|k| !k.is_empty()) { Some(k) => f(&k[k.len() - 1][0]) + 3600.0, None => (f(&e["generated_ms"]) / 3_600_000.0).floor() * 3600.0 };
    let (mut lo, mut hi) = (p, p);
    for c in cs { if c.time as f64 >= t0 && c.high > 0.0 && c.low > 0.0 { hi = hi.max(c.high); lo = lo.min(c.low); } }
    (lo, hi)
}
/// painOf(E, P) -- cs = 그 코인 5분봉(지금 코인만 · 다른 코인 = 빈)
pub fn pain_of(e: Option<&Value>, p: f64, cs: &[Candle]) -> Option<Pain> {
    let e = e?;
    let sig = f(&e["sigma_d"]);
    if !e["per"].is_object() || !(p > 0.0) || !(sig > 0.0) { return None; }
    let (lo, hi) = swept(e, p, cs);
    let lev = |b: &Value| { let (x, a) = (f(&b[0]), f(&b[1])); let z = (x / p).ln().abs() / sig; Lev { px: x, amt: a, d: (x / p - 1.0) * 100.0, z, p: reach(z), alt: None } };
    let pick = |bins: &Value, below: bool| -> Option<Lev> {
        let mut a: Vec<&Value> = bins.as_array().into_iter().flatten().filter(|x| if below { f(&x[0]) < lo } else { f(&x[0]) > hi }).collect();
        a.sort_by(|x, y| f(&y[1]).partial_cmp(&f(&x[1])).unwrap_or(std::cmp::Ordering::Equal));
        let mut o = lev(a.first()?);
        if a.len() > 1 && f(&a[1][1]) >= 0.9 * f(&a[0][1]) { let q = lev(a[1]); o.alt = Some([q.px, q.amt, q.d]); }
        Some(o)
    };
    let mut per = [None; 7];
    for (i, k) in PERS.iter().enumerate() {
        let q = &e["per"][*k];
        if q.is_null() { continue; }
        let (dn, up) = (pick(&q["dn"], true), pick(&q["up"], false));
        let (gu, gd) = (up.map_or(0.0, |u| u.amt * u.p), dn.map_or(0.0, |d| d.amt * d.p));
        let both = dn.zip(up);
        per[i] = Some(Per { dn, up, pull: if gu + gd > 0.0 { Some(gu / (gu + gd)) } else { None },
                            near: both.map(|(d, u)| if (d.z - u.z).abs() < 0.1 { "비등" } else if d.z < u.z { "아래" } else { "위" }),
                            pincer: both.is_some_and(|(d, u)| d.p >= 0.5 && u.p >= 0.5) });
    }
    let nest = |up: bool| {
        let a: Vec<f64> = per.iter().flatten().filter_map(|q| if up { q.up } else { q.dn }).map(|l| l.px).collect();
        let (mut best, mut at) = (0, f64::NAN);
        for x in &a { let c = a.iter().filter(|y| (*y / x - 1.0).abs() <= NEST_PCT).count(); if c > best { best = c; at = *x; } }
        (best, at)
    };
    Some(Pain { nest: [nest(false), nest(true)], per })
}
fn pp_i(pp: &str) -> usize { PERS.iter().position(|k| *k == pp).unwrap_or(1) }

fn ax(x: f64, y: f64, s: String, anchor: Anchor, t: bool, halo: bool) -> Prim { Prim::Mono { x, y, s, size: 10.0, wt: if t { 700 } else { 500 }, c: if t { Col::Text } else { Col::Muted }, a: 1.0, anchor, halo, clip: Clip::No } }
fn line(x1: f64, y1: f64, x2: f64, y2: f64, c: Col, a: f64, lw: f64) -> Prim { Prim::Line { pts: vec![[x1, y1], [x2, y2]], c, a, lw, dash: None, clip: Clip::No } }
fn dashed(pts: Vec<[f64; 2]>, c: Col, a: f64, lw: f64, on: f64, off: f64) -> Prim { Prim::Dashed { pts, c, a, lw, on, off, clip: Clip::No } }
fn rect(x: f64, y: f64, w: f64, h: f64, c: Col, a: f64, rx: f64) -> Prim { Prim::Rect { x, y, w, h, c, a, rx, clip: Clip::No } }

/// CH.today 그림: s = 그리기 · band = 평소 25~75% 띠(위 p75 · 아래 p25 · 정해진 자리 at 에 칠한다 -- 볼록하지 않은 면이라 따로)
pub struct Today { pub s: Vec<Prim>, pub band: Option<(Vec<[f64; 2]>, Vec<[f64; 2]>)>, pub at: usize }
/// today() + CH.today -- None = 빈 상태(«오늘 기록이 아직 없습니다») · now = Date.now()
pub fn today(bars: &[Bar], tab: Option<&Value>, now: f64, w: f64, h: f64) -> Option<Today> {
    let d0 = (now / 86_400_000.0).floor() * 86400.0;
    let (mut cum, mut acc): (Vec<Option<f64>>, f64) = (vec![], 0.0);
    for b in bars {
        if b.t < d0 { continue; }
        acc += b.l + b.s - b.by - b.hl;
        let i = ((b.t - d0) / 300.0).floor();
        if !i.is_finite() { continue; }
        let i = i as usize;
        if cum.len() <= i { cum.resize(i + 1, None); }
        cum[i] = Some(acc);
    }
    for i in 1..cum.len() { if cum[i].is_none() { cum[i] = cum[i - 1]; } }
    let bd = tab.map(|t| &t["band"]).filter(|b| b.is_object());
    if cum.is_empty() && bd.is_none() { return None; }
    let arr = |k: &str| -> Vec<f64> { bd.and_then(|b| b[k].as_array()).into_iter().flatten().map(|v| v.as_f64().unwrap_or(0.0)).collect() };
    let (p25, p50, p75) = (arr("p25"), arr("p50"), arr("p75"));
    let (l, r, tp, bt) = (46.0, 8.0, 6.0, 15.0);
    let m0 = if bd.is_some() { p75.iter().copied().fold(f64::NEG_INFINITY, f64::max) } else { 0.0 };
    let mx = if cum.iter().any(|v| v.is_none()) { f64::NAN } else { cum.iter().flatten().copied().fold(m0.max(1.0), f64::max) } * 1.06; // Math.max(undefined) = NaN
    let x = |i: usize| l + (w - l - r) * (i + 1) as f64 / 288.0;
    let y = |v: f64| tp + (h - tp - bt) * (1.0 - v / mx);
    let mut s = vec![];
    for v in ticks(mx, 3.0) { s.extend([line(l, y(v), w - r, y(v), Col::SoftLine, 1.0, 1.0), ax(l - 5.0, y(v) + 3.0, usd(v), Anchor::End, false, false)]); }
    let at = s.len();
    let pts = |a: &[f64]| -> Vec<[f64; 2]> { a.iter().enumerate().map(|(i, v)| [f1(x(i)), f1(y(*v))]).collect() };
    let band = bd.map(|_| (pts(&p75), pts(&p25)));
    if bd.is_some() { s.push(dashed(pts(&p50), Col::Muted, 1.0, 1.2, 3.0, 3.0)); }
    if !cum.is_empty() {
        let raw: Vec<[f64; 2]> = cum.iter().enumerate().filter_map(|(i, v)| v.map(|v| [x(i), y(v)])).collect();
        let (e, lst) = (raw[raw.len() - 1], cum[cum.len() - 1].unwrap_or(f64::NAN));
        s.push(Prim::Line { pts: raw.iter().map(|q| [f1(q[0]), f1(q[1])]).collect(), c: Col::Accent, a: 1.0, lw: 1.8, dash: None, clip: Clip::No });
        s.push(Prim::Circle { x: e[0], y: e[1], r: 3.5, c: Col::Accent, a: 1.0, clip: Clip::No });
        s.push(if e[0] + 60.0 > w { ax(e[0] - 7.0, e[1] + 3.0, usd(lst), Anchor::End, true, true) } else { ax(e[0] + 7.0, e[1] + 3.0, usd(lst), Anchor::Start, true, true) });
    }
    let labs: &[(f64, &str)] = if w < 280.0 { &[(0.0, "09시"), (144.0, "21시"), (288.0, "09시")] } else { &[(0.0, "09시"), (72.0, "15시"), (144.0, "21시"), (216.0, "03시"), (288.0, "09시")] };
    for &(q, t) in labs { s.push(ax(l + (w - l - r) * q / 288.0, h - 3.0, t.into(), if q == 0.0 { Anchor::Start } else if q == 288.0 { Anchor::End } else { Anchor::Middle }, false, false)); }
    Some(Today { s, band, at })
}

/// CH.pain -- None = 빈 상태(«최대 고통 계산 전입니다») · e = 지금 코인 묶음 · p = 지금가 · cs = 지금 코인 5분봉
pub fn pain(e: Option<&Value>, pp: &str, p: f64, cs: &[Candle], w: f64, h: f64) -> Option<Vec<Prim>> {
    let r = pain_of(e, p, cs)?;
    let k: Vec<[f64; 5]> = e?["k3"].as_array().filter(|k| k.len() >= 2)?.iter().map(|b| [f(&b[0]), f(&b[1]), f(&b[2]), f(&b[3]), f(&b[4])]).collect();
    let sel = r.per[pp_i(pp)]?;
    let (nc, cwid, l, tp, bt) = (7.0, if w < 440.0 { 20.0 } else { 31.0 }, 46.0, 18.0, 15.0);
    let xr = w - nc * cwid - 4.0;
    let cw = (xr - l) / k.len() as f64;
    let lv: Vec<Lev> = r.per.iter().flatten().flat_map(|q| [q.dn, q.up]).flatten().collect();
    let mut axv: Vec<f64> = k.iter().map(|b| b[3]).chain(k.iter().map(|b| b[2])).chain([p]).collect();
    for z in [sel.dn, sel.up].into_iter().flatten() { axv.extend([z.px * (1.0 - BOX), z.px * (1.0 + BOX)]); }
    let (mut lo, mut hi) = (axv.iter().copied().fold(f64::INFINITY, f64::min), axv.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let pad = (hi - lo) * 0.05;
    (lo, hi) = (lo - pad, hi + pad);
    let y = |q: f64| tp + (h - tp - bt) * (1.0 - (lo.max(hi.min(q)) - lo) / (hi - lo));
    let in_ax = |q: f64| q >= lo && q <= hi;
    let ma = lv.iter().map(|x| x.amt).fold(1e-12, f64::max);
    let mut s = vec![];
    if let Some(st) = nice_step(hi - lo, 6.0) {
        let mut q = (lo / st).ceil() * st;
        while q < hi { s.extend([line(l, y(q), w, y(q), Col::SoftLine, 1.0, 1.0), ax(l - 5.0, y(q) + 3.0, px(q), Anchor::End, false, false)]); q += st; }
    }
    for (z, c) in [(sel.dn, Col::Bad), (sel.up, Col::Good)] { if let Some(z) = z { s.push(rect(l, y(z.px * (1.0 + BOX)), xr - l, 1f64.max(y(z.px * (1.0 - BOX)) - y(z.px * (1.0 + BOX))), c, 0.1, 0.0)); } }
    for (i, b) in k.iter().enumerate() {
        let x = l + (i as f64 + 0.5) * cw;
        s.push(line(x, y(b[2]), x, y(b[3]), Col::Muted, 1.0, 1.0));
        s.push(rect(f1(x - cw * 0.32), f1(y(b[1].max(b[4]))), f1((cw * 0.64).max(1.0)), f1((y(b[1]) - y(b[4])).abs().max(1.0)), if b[4] >= b[1] { CU } else { CD }, 1.0, 0.0));
        if (b[0] + 32400.0) % 86400.0 == 0.0 && cw * 24.0 >= 36.0 { s.push(ax(x, h - 3.0, iso_of((b[0] + 32400.0) as i64)[5..10].into(), Anchor::Middle, false, false)); }
    }
    for (z, c) in [(sel.dn, Col::Bad), (sel.up, Col::Good)] {
        if let Some(z) = z { s.extend([dashed(vec![[l, y(z.px)], [xr, y(z.px)]], c, 1.0, 1.2, 5.0, 3.0), ax(l + 4.0, y(z.px) - 4.0, format!("{} · {}%", px(z.px), sgn(z.d, 2)), Anchor::Start, true, true)]); }
    }
    for (j, kk) in PERS.iter().enumerate() {
        let (x0, on) = (xr + 4.0 + j as f64 * cwid, *kk == pp);
        if on { s.push(rect(x0, tp - 4.0, cwid - 2.0, h - tp - bt + 4.0, Col::White, 0.08, 3.0)); }
        s.push(ax(x0 + (cwid - 2.0) / 2.0, 10.0, plab(kk).replace("시간", "h"), Anchor::Middle, on, false));
        let Some(q) = r.per[j] else { continue };
        for (z, c) in [(q.dn, Col::Bad), (q.up, Col::Good)] {
            let Some(z) = z else { continue };
            let (ww, yy) = (4f64.max((cwid - 4.0) * (z.amt / ma).sqrt()), y(z.px));
            let pts = vec![[x0 + (cwid - 2.0 - ww) / 2.0, yy], [x0 + (cwid - 2.0 + ww) / 2.0, yy]];
            s.push(if in_ax(z.px) { Prim::Line { pts, c, a: 1.0, lw: 3.0, dash: None, clip: Clip::No } } else { dashed(pts, c, 0.5, 3.0, 2.0, 2.0) });
        }
    }
    s.push(dashed(vec![[l, y(p)], [w, y(p)]], Col::Accent, 1.0, 1.2, 4.0, 3.0));
    s.push(rect(0.0, y(p) - 8.0, l - 4.0, 16.0, Col::Accent, 1.0, 3.0));
    s.push(Prim::Mono { x: (l - 4.0) / 2.0, y: y(p) + 3.5, s: px(p), size: 10.5, wt: 700, c: Col::OnFill, a: 1.0, anchor: Anchor::Middle, halo: false, clip: Clip::No });
    Some(s)
}

/// coinRows 한 줄
pub struct Row { pub a: String, pub q: Per, pub old: bool }
/// coinRows(): 코인 차례 그대로 · 가격 = 지금 코인은 p · 다른 코인은 허브 가격(없으면 묶음 가격) · 5분봉은 지금 코인만
pub fn coin_rows(coins: Option<&[(String, Value)]>, asset: &str, pp: &str, p: f64, live: &dyn Fn(&str) -> f64, cs: &[Candle], now: f64) -> Vec<Row> {
    coins.into_iter().flatten().filter_map(|(a, e)| {
        let me = a == asset;
        let pa = if me { p } else { let q = live(a); if q > 0.0 { q } else { f(&e["price"]) } };
        let q = pain_of(Some(e), pa, if me { cs } else { &[] })?.per[pp_i(pp)]?;
        Some(Row { a: a.clone(), q, old: now - f(&e["generated_ms"]) > STALE_MS })
    }).collect()
}

/// CH.quad -- None = 빈 상태(«코인별 계산 전입니다»)
pub fn quad(rows: &[Row], asset: &str, w: f64, h: f64) -> Option<Vec<Prim>> {
    let rs: Vec<(&Row, Lev, Lev)> = rows.iter().filter_map(|r| Some((r, r.q.dn?, r.q.up?))).collect();
    if rs.len() < 2 { return None; }
    let (zp, l, r, tp, bt) = (0.6745, 30.0, 8.0, 8.0, 37.0);
    let mx = 1.5f64.max(rs.iter().map(|q| q.1.z.max(q.2.z)).fold(f64::NEG_INFINITY, f64::max) * 1.1);
    let x = |d: f64| l + (w - l - r) * d / mx;
    let y = |d: f64| tp + (h - tp - bt) * (1.0 - d / mx);
    let mut s = vec![rect(l, y(zp), x(zp) - l, y(0.0) - y(zp), Col::Amber, 0.14, 0.0)];
    for v in ticks(mx, 4.0) {
        let t = to_fixed(v, if v % 1.0 != 0.0 { 1 } else { 0 }) + "σ";
        s.extend([line(l, y(v), w - r, y(v), Col::SoftLine, 1.0, 1.0), line(x(v), tp, x(v), y(0.0), Col::SoftLine, 1.0, 1.0), ax(l - 4.0, y(v) + 3.0, t.clone(), Anchor::End, false, false), ax(x(v), y(0.0) + 11.0, t, Anchor::Middle, false, false)]);
    }
    s.extend([dashed(vec![[x(0.0), y(0.0)], [x(mx), y(mx)]], Col::Muted, 1.0, 1.2, 3.0, 3.0), line(l, y(0.0), w - r, y(0.0), Col::Neutral, 0.38, 1.0), line(l, tp, l, y(0.0), Col::Neutral, 0.38, 1.0)]);
    s.push(ax((l + w - r) / 2.0, h - 14.0, "가로 = 아래 롱까지 · 세로 = 위 숏까지(하루 σ)".into(), Anchor::Middle, false, false));
    s.push(ax((l + w - r) / 2.0, h - 3.0, "대각선 위 = 아래가 가까움".into(), Anchor::Middle, false, false));
    for (q, dn, up) in rs.iter().filter(|q| q.0.a != asset).chain(rs.iter().filter(|q| q.0.a == asset)) {
        let (cx, cy, m) = (x(dn.z), y(up.z), q.a == asset);
        s.push(Prim::Circle { x: cx, y: cy, r: if m { 4.5 } else { 3.5 }, c: if m { Col::Accent } else { Col::Neutral }, a: if q.old { 0.4 } else { 1.0 }, clip: Clip::No });
        s.push(ax(cx + if m { 0.0 } else { 6.0 }, cy + if m { -8.0 } else { 3.0 }, q.a.to_uppercase(), if m { Anchor::Middle } else { Anchor::Start }, m, true));
    }
    Some(s)
}

/// painKpi: 지표 넷(제목, 값, 아래 글, 값 색) · 가까운 쪽 · 괄호 글 · 인력 글 · 꼬리표(노랑?, 글)
pub struct Kpi { pub kp: Vec<(String, String, String, Col)>, pub near: String, pub nz: String, pub pull: String, pub tags: Vec<(bool, String)> }
pub fn kpi(e: Option<&Value>, pn: Option<&Pain>, pp: &str) -> Option<Kpi> {
    let (e, r) = (e?, pn?);
    let q = r.per[pp_i(pp)]?;
    let share = e["unit"] == "share";
    let side = |lab: &str, z: Option<Lev>, c: Col| match z { Some(z) => (lab.into(), px(z.px), format!("{}% · {}", sgn(z.d, 2), amt(share, z.amt)), c), None => (lab.into(), "—".into(), "남은 봉우리 없음".into(), Col::Text) };
    let pr = |lab: &str, z: Option<Lev>| match z { Some(z) => (lab.into(), pc(z.p, 0), format!("24시간 안 · {}σ", to_fixed(z.z, 1)), Col::Text), None => (lab.into(), "—".into(), String::new(), Col::Text) };
    let mut tags = vec![];
    if q.pincer { tags.push((true, "양방향 협공".to_string())); }
    for (i, z) in [q.dn, q.up].iter().enumerate() { if let Some(a) = z.and_then(|z| z.alt) { tags.push((false, format!("{} 비슷한 2등 {}", if i > 0 { "위" } else { "아래" }, px(a[0])))); } }
    for (i, z) in [q.dn, q.up].iter().enumerate() { if z.is_some_and(|z| z.d.abs() <= 1.0) { tags.push((false, format!("{} 1% 이내", if i > 0 { "위" } else { "아래" }))); } }
    for (k, lab) in [(1, "위"), (0, "아래")] { let (n, at) = r.nest[k]; if n >= 5 { tags.push((false, format!("{lab} 중첩 {n}/7 · {}", px(at)))); } }
    Some(Kpi {
        kp: vec![side("위 · 숏 최대 고통", q.up, Col::Good), pr("위 도달", q.up), side("아래 · 롱 최대 고통", q.dn, Col::Bad), pr("아래 도달", q.dn)],
        near: q.near.unwrap_or("—").into(),
        nz: q.dn.zip(q.up).map_or(String::new(), |(d, u)| format!(" (아래 {}σ · 위 {}σ)", to_fixed(d.z, 1), to_fixed(u.z, 1))),
        pull: q.pull.map_or("인력 —".into(), |u| format!("인력 위 {} : 아래 {}", pc(u, 0), pc(1.0 - u, 0))),
        tags,
    })
}

/// coinTable 한 줄: 종목 · on|old|"" · 협공 · 아래 [굵은, 작은] · 위 · 가까운 -- 빈 목록 = «코인별 계산 전입니다»
pub type TRow = (String, &'static str, bool, Option<[String; 2]>, Option<[String; 2]>, String);
pub fn table(rows: &[Row], asset: &str) -> Vec<TRow> {
    let cell = |z: Option<Lev>| z.map(|z| [format!("{}%", sgn(z.d, 1)), pc(z.p, 0)]);
    rows.iter().map(|r| (r.a.to_uppercase(), if r.a == asset { "on" } else if r.old { "old" } else { "" }, r.q.pincer, cell(r.q.dn), cell(r.q.up), r.q.near.unwrap_or("—").into())).collect()
}

/// lqPainLeg(글, 경고?) -- 지금 코인 묶음이 없으면 빈
pub fn legend(e: Option<&Value>, pp: &str, now: f64) -> Vec<(String, bool)> {
    let Some(e) = e else { return vec![] };
    let age = (now - f(&e["generated_ms"])) / 60000.0;
    let mut v = vec![(format!("띠 = 지도 {}(사다리와 같은 모델)의 ±0.5% 묶음 최대", plab(pp)), false), (format!("도달 = 하루 σ {} 무작위 보행", pc(f(&e["sigma_d"]), 1)), false),
                     ("오른쪽 7줄 = 주기별 · 굵기 = 금액".into(), false), (format!("{} · 서술용", if e["unit"] == "share" { "금액 = 주기 합 대비 몫" } else { "금액 = 세 거래소 OI×2 눈금" }), false)];
    if age > 90.0 { v.push((format!("{}시간 전 계산", to_fixed(age / 60.0, 0)), true)); }
    v
}

/// 띠 면(.lq-bd --neutral 15%): 이웃 두 점마다 사다리꼴 한 그물(위·아래가 같은 x 라 볼록 조각)
fn paint_band(pa: &Painter, top: &[[f64; 2]], bot: &[[f64; 2]], o: Pos2) {
    let c = draw::rgba(Col::Neutral, 0.15);
    let mut m = egui::Mesh::default();
    for i in 1..top.len().min(bot.len()) {
        let n = m.vertices.len() as u32;
        for q in [top[i - 1], top[i], bot[i], bot[i - 1]] { m.colored_vertex(o + vec2(q[0] as f32, q[1] as f32), c); }
        m.add_triangle(n, n + 1, n + 2); m.add_triangle(n, n + 2, n + 3);
    }
    pa.add(egui::Shape::mesh(m));
}

impl crate::Chart {
    /// «청산» 탭 아랫줄 한 장(feed.liqx) -- 상자 = cfg.w_all×h_all(fill 이면 창) · 판 셋(.lq-row gap 1 · 판 안 여백 9 11 8 · 머리 = 제목만)
    pub fn ui_liqx(&mut self, ui: &mut egui::Ui, now: f64) {
        if self.fill { let r = ui.ctx().content_rect(); (self.cfg.w_all, self.cfg.h_all) = ((r.width() - 2.0 * self.origin.x) as f64, (r.height() - 2.0 * self.origin.y) as f64); }
        let (w, h, f) = (self.cfg.w_all, self.cfg.h_all, &self.feed);
        let Some(x) = f.liqx.as_ref() else { return };
        let pa = ui.painter();
        if self.bg.a() > 0 { pa.rect_filled(pa.clip_rect(), 0.0, self.bg); }
        let measure = |s: &str, z: f64, wt: u16| draw::measure(pa, s, z, wt);
        let mmono = |s: &str, z: f64, wt: u16| pa.layout_no_wrap(s.to_owned(), draw::mono(z as f32, wt), Color32::WHITE).size().x as f64;
        let tab = f.liq_tab.as_ref();
        let p = crate::liq::price(f.prices.get(&f.asset).copied().unwrap_or(0.0), tab);
        let e = x.coins.as_deref().and_then(|c| c.iter().find(|q| q.0 == f.asset)).map(|q| &q.1);
        let live = |a: &str| f.prices.get(a).copied().unwrap_or(0.0);
        let rows = coin_rows(x.coins.as_deref(), &f.asset, x.pp, p, &live, &f.candles, now);
        let o = self.origin;
        let at = |dx: f64, dy: f64| o + vec2(dx as f32, dy as f32);
        let big = [0.0, 0.0, 1e5, 1e5];
        let put = |v: &[Prim], dx: f64, dy: f64| draw::paint(pa, v, at(dx, dy), big, big, 1.0);
        let tx = |x: f64, y: f64, s: String, size: f64, wt: u16, c: Col| Prim::Text { x, y, s, size, wt, c, a: 1.0, anchor: Anchor::Start, halo: false, clip: Clip::No };
        // 판 셋: flex 0.9 · 2.1 · 1.7(바탕 = 1px 선 색) · 머리 = 제목 12.5px 600 + 아래 5
        let ws = [0.9, 2.1, 1.7].map(|g| (w - 2.0) * g / 4.7);
        put(&[rect(0.0, 0.0, w, h, Col::SoftLine, 1.0, 0.0)], 0.0, 0.0);
        let mut x0 = 0.0;
        let mut boxes = [[0.0; 4]; 3];
        for (i, (pw, t)) in ws.iter().zip(["오늘 누적", "최대 고통", "코인별 최대 고통"]).enumerate() {
            put(&[rect(x0, 0.0, *pw, h, Col::ChartBg, 1.0, 0.0), tx(x0 + 11.0, 9.0 + 12.0, t.into(), 12.5, 600, Col::Text)], 0.0, 0.0);
            boxes[i] = [x0 + 11.0, 9.0 + 22.0 + 5.0, pw - 22.0, h - 9.0 - 8.0 - 27.0];
            x0 += pw + 1.0;
        }
        let empty = |b: [f64; 4], s: &str| put(&crate::liq::empty_msg(s), b[0], b[1]);
        // 오늘 누적
        let b = boxes[0];
        match today(&x.bars, tab, now, b[2], b[3]) {
            Some(t) => {
                put(&t.s[..t.at], b[0], b[1]);
                if let Some((top, bot)) = &t.band { paint_band(pa, top, bot, at(b[0], b[1])); }
                put(&t.s[t.at..], b[0], b[1]);
            }
            None => empty(b, "오늘 기록이 아직 없습니다"),
        }
        // 최대 고통: 왼쪽 236(지표 2×2 · 가까운 쪽 · 꼬리표 · 범례 -- 사이 6) + 14 + 사다리
        let b = boxes[1];
        let pn = pain_of(e, p, &f.candles);
        let (sw, mut sy) = (236.0, b[1]);
        if let Some(k) = kpi(e, pn.as_ref(), x.pp) {
            for row in k.kp.chunks(2) { put(&crate::rem::ribbon(row, sw, &measure), b[0], sy); sy += 50.5 + 4.0; }
            sy += 2.0;
            let pl1 = format!("가까운 쪽 {}{}", k.near, k.nz);
            let w1 = measure(&pl1, 11.5, 400);
            let (px2, py2) = if w1 + 14.0 + measure(&k.pull, 11.5, 400) <= sw { (w1 + 14.0, 0.0) } else { (0.0, 16.0) };
            put(&[tx(0.0, 11.0, pl1, 11.5, 400, Col::Muted), tx(px2, 11.0 + py2, k.pull.clone(), 11.5, 400, Col::Muted)], b[0], sy);
            sy += 14.0 + py2 + 6.0;
            let mut tx0 = 0.0;
            for (lit, t) in &k.tags {
                let tw = measure(t, 10.0, 600) + 12.0;
                if tx0 > 0.0 && tx0 + tw > sw { tx0 = 0.0; sy += 22.0; }
                let c = if *lit { Col::Amber } else { Col::Muted };
                put(&[Prim::StrokeRect { x: tx0 + 0.5, y: 0.5, w: tw - 1.0, h: 17.0, c: if *lit { Col::Amber } else { Col::Neutral }, a: if *lit { 0.45 } else { 0.26 }, lw: 1.0, clip: Clip::No }, tx(tx0 + 6.0, 12.6, t.clone(), 10.0, 600, c)], b[0], sy);
                tx0 += tw + 4.0;
            }
            if !k.tags.is_empty() { sy += 18.0 + 6.0; }
        }
        let (mut lx, lh) = (0.0, 10.5 * 1.4);
        for (t, warn) in legend(e, x.pp, now) {
            let tw = measure(&t, 10.5, 400);
            if lx > 0.0 && lx + tw > sw { lx = 0.0; sy += lh + 2.0; }
            put(&[tx(lx, lh / 2.0 + 10.5 * 0.36, t, 10.5, 400, if warn { Col::Amber } else { Col::Muted })], b[0], sy);
            lx += tw + 10.0;
        }
        let (pw, ph) = (b[2] - sw - 14.0, b[3]);
        match pain(e, x.pp, p, &f.candles, pw, ph) { Some(v) => put(&v, b[0] + sw + 14.0, b[1]), None => empty([b[0] + sw + 14.0, b[1], 0.0, 0.0], "최대 고통 계산 전입니다") }
        // 코인별: 사분면(남는 폭) + 12 + 표(글 폭)
        let b = boxes[2];
        let tr = table(&rows, &f.asset);
        let head = ["종목", "아래 롱 · 도달", "위 숏 · 도달", "가까운"];
        let cellw = |c: &Option<[String; 2]>| c.as_ref().map_or(measure("—", 11.5, 400), |c| mmono(&c[0], 12.0, 700) + measure(" ", 11.5, 400) + mmono(&c[1], 10.5, 500));
        let mut cw = head.map(|t| measure(t, 10.5, 500));
        for r in &tr {
            let nw = measure(&r.0, 11.5, if r.1 == "on" { 700 } else { 400 }) + if r.2 { measure(" ", 11.5, 400) + measure("협공", 10.0, 600) + 8.0 } else { 0.0 };
            for (i, v) in [nw, cellw(&r.3), cellw(&r.4), measure(&r.5, 11.5, 400)].into_iter().enumerate() { cw[i] = cw[i].max(v); }
        }
        let cw = cw.map(|v| v + 6.0);
        let tw = if tr.is_empty() { 0.0 } else { cw.iter().sum::<f64>().min(b[2]) };
        let qw = b[2] - if tr.is_empty() { 0.0 } else { tw + 12.0 };
        match quad(&rows, &f.asset, qw, b[3]) { Some(v) => put(&v, b[0], b[1]), None => empty(b, "코인별 계산 전입니다") }
        if tr.is_empty() { if rows.is_empty() && x.coins.is_some() { empty([b[0] + qw, b[1], 0.0, 0.0], "코인별 계산 전입니다"); } return; }
        let tx0 = b[0] + qw + 12.0;
        let xs: Vec<f64> = cw.iter().scan(0.0, |a, v| { let o = *a; *a += v; Some(o) }).collect();
        let mut v = vec![];
        for (i, t) in head.iter().enumerate() { v.push(tx(xs[i], 10.0, t.to_string(), 10.5, 500, Col::Muted)); }
        v.push(rect(0.0, 16.0, tw, 1.0, Col::SoftLine, 1.0, 0.0));
        let mut ry = 17.0;
        for r in &tr {
            let a = if r.1 == "old" { 0.5 } else { 1.0 };
            if r.1 == "on" { v.push(rect(0.0, ry, tw, 22.0, Col::White, 0.06, 0.0)); }
            let bl = ry + 3.0 + 12.5;
            v.push(Prim::Text { x: 0.0, y: bl, s: r.0.clone(), size: 11.5, wt: if r.1 == "on" { 700 } else { 400 }, c: Col::Text, a, anchor: Anchor::Start, halo: false, clip: Clip::No });
            if r.2 {
                let nx = measure(&r.0, 11.5, if r.1 == "on" { 700 } else { 400 }) + measure(" ", 11.5, 400);
                let pw = measure("협공", 10.0, 600) + 8.0;
                v.push(Prim::StrokeRect { x: nx + 0.5, y: ry + 3.5, w: pw - 1.0, h: 15.0, c: Col::Amber, a: 0.45 * a, lw: 1.0, clip: Clip::No });
                v.push(Prim::Text { x: nx + 4.0, y: bl - 1.0, s: "협공".into(), size: 10.0, wt: 600, c: Col::Amber, a, anchor: Anchor::Start, halo: false, clip: Clip::No });
            }
            for (i, (c, col)) in [(&r.3, Col::Bad), (&r.4, Col::Good)].into_iter().enumerate() {
                let cx = xs[i + 1];
                match c {
                    Some(c) => {
                        let b0 = mmono(&c[0], 12.0, 700) + measure(" ", 11.5, 400);
                        v.push(Prim::Mono { x: cx, y: bl, s: c[0].clone(), size: 12.0, wt: 700, c: col, a, anchor: Anchor::Start, halo: false, clip: Clip::No });
                        v.push(Prim::Mono { x: cx + b0, y: bl, s: c[1].clone(), size: 10.5, wt: 500, c: Col::Muted, a, anchor: Anchor::Start, halo: false, clip: Clip::No });
                    }
                    None => v.push(Prim::Text { x: cx, y: bl, s: "—".into(), size: 11.5, wt: 400, c: Col::Text, a, anchor: Anchor::Start, halo: false, clip: Clip::No }),
                }
            }
            v.push(Prim::Text { x: xs[3], y: bl, s: r.5.clone(), size: 11.5, wt: 400, c: Col::Text, a, anchor: Anchor::Start, halo: false, clip: Clip::No });
            v.push(rect(0.0, ry + 22.0, tw, 1.0, Col::SoftLine, 1.0, 0.0));
            ry += 23.0;
        }
        draw::paint(pa, &v, pos2(o.x + tx0 as f32, o.y + b[1] as f32), big, big, 1.0);
    }
}

#[cfg(test)]
mod t {
    use super::*;
    use serde_json::json;

    fn fx(f: &str) -> Value { serde_json::from_str(&std::fs::read_to_string(format!("{}/compare/fixture/{f}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap() }
    fn cname(c: Col) -> &'static str {
        match c { Col::SoftLine => "soft-line", Col::Neutral => "neutral", Col::Accent => "accent", Col::Muted => "muted", Col::Text => "text", Col::OnFill => "on-fill", Col::Bad => "bad", Col::Good => "good",
                  Col::Amber => "warn", Col::White => "white", CU => "cu", CD => "cd", _ => "?" }
    }
    fn an(a: &Anchor) -> &'static str { match a { Anchor::Start => "start", Anchor::Middle => "middle", Anchor::End => "end" } }
    /// 그리기 목록 → 정답지 모양(compare/liqx_js.mjs parse 와 같은) · 띠는 at 자리에 [위 + 아래 거꾸로]
    fn norm(v: &[Prim], band: Option<(&(Vec<[f64; 2]>, Vec<[f64; 2]>), usize)>) -> Value {
        let mut o: Vec<Value> = v.iter().map(|p| match p {
            Prim::Line { pts, c, lw, .. } if *lw == 1.8 => json!(["P", pts, cname(*c), lw, null]),
            Prim::Line { pts, c, a, lw, .. } => json!(["L", pts[0][0], pts[0][1], pts[1][0], pts[1][1], cname(*c), a, lw, null]),
            Prim::Dashed { pts, c, lw, on, off, .. } if pts.len() > 2 => json!(["P", pts, cname(*c), lw, [on, off]]),
            Prim::Dashed { pts, c, a, lw, on, off, .. } => json!(["L", pts[0][0], pts[0][1], pts[1][0], pts[1][1], cname(*c), a, lw, [on, off]]),
            Prim::Mono { x, y, s, size, wt, c, anchor, halo, .. } => json!(["T", x, y, s, size, wt, cname(*c), an(anchor), halo]),
            Prim::Rect { x, y, w, h, c, a, rx, .. } => json!(["R", x, y, w, h, cname(*c), a, rx]),
            Prim::Circle { x, y, r, c, a, .. } => json!(["C", x, y, r, cname(*c), a]),
            _ => json!(["?"]),
        }).collect();
        if let Some(((top, bot), at)) = band { let mut q = top.clone(); q.extend(bot.iter().rev()); o.insert(at, json!(["A", q, "neutral", 0.15])); }
        Value::Array(o)
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
    fn candles(v: &Value) -> Vec<Candle> {
        v.as_array().into_iter().flatten().map(|c| Candle { time: c["time"].as_i64().unwrap(), open: 0.0, high: c["high"].as_f64().unwrap(), low: c["low"].as_f64().unwrap(), close: 0.0, partial: None, volume: 0.0, sma: None }).collect()
    }
    /// 정답지 한 입력을 Rust 로(응답 글을 State 에 넣는 길 그대로)
    fn run(c: &Value) -> Value {
        let mut x = State::new(c["pp"].as_str().unwrap()).unwrap();
        x.on_all(c["allText"].as_str());
        x.on_bars(Some(&json!({"bars": c["bars"]}).to_string()));
        let (asset, now, tab) = (c["asset"].as_str().unwrap(), c["now"].as_f64().unwrap(), Some(&c["tab"]).filter(|t| t["ok"] == true));
        let lv = |a: &str| c["live"][a].as_f64().unwrap_or(0.0);
        let p = crate::liq::price(lv(asset), tab);
        let cs = candles(&c["candles"]);
        let e = x.coins.as_deref().and_then(|v| v.iter().find(|q| q.0 == asset)).map(|q| &q.1);
        let wh = |k: &str| (c[k][0].as_f64().unwrap(), c[k][1].as_f64().unwrap());
        let ((tw, th), (pw, ph), (qw, qh)) = (wh("tod"), wh("pain"), wh("quad"));
        let rows = coin_rows(x.coins.as_deref(), asset, x.pp, p, &lv, &cs, now);
        let td = today(&x.bars, tab, now, tw, th).map_or(Value::Null, |t| norm(&t.s, t.band.as_ref().map(|b| (b, t.at))));
        let pn = pain_of(e, p, &cs);
        let k = kpi(e, pn.as_ref(), x.pp).map_or(Value::Null, |k| json!({"kp": k.kp.iter().map(|q| json!([q.0, q.1, q.2, match q.3 { Col::Bad => "lg", Col::Good => "sh", _ => "" }])).collect::<Vec<_>>(),
                                                                       "pl": [k.near, k.nz, k.pull], "tags": k.tags}));
        let tb = table(&rows, asset);
        json!({
            "today": td,
            "pain": pain(e, x.pp, p, &cs, pw, ph).map_or(Value::Null, |v| norm(&v, None)),
            "quad": quad(&rows, asset, qw, qh).map_or(Value::Null, |v| norm(&v, None)),
            "kpi": k,
            "table": if tb.is_empty() { Value::Null } else { json!(tb) },
            "leg": legend(e, x.pp, now),
        })
    }

    /// 지금 화면의 실제 함수로 만든 정답지(compare/liqx_js.mjs → fixture/liqx_js.json · 같은 입력 liqx_in.json)
    #[test]
    fn js_parity() {
        let (inp, js) = (fx("liqx_in.json"), fx("liqx_js.json"));
        let mut n = [0; 3];
        for (i, (c, j)) in inp.as_array().unwrap().iter().zip(js.as_array().unwrap()).enumerate() {
            same(&run(c), j, &format!("입력 {i}"));
            for (q, k) in n.iter_mut().zip(["today", "pain", "quad"]) { *q += j[k].as_array().map_or(0, |v| v.len()); }
        }
        assert!(inp.as_array().unwrap().len() >= 8 && n[0] > 60 && n[1] > 1000 && n[2] > 100, "{n:?}");
    }

    /// 양성 대조: 칸마다 계산 갈래를 하나씩 바꾸면 정답지와 어긋난다(시험이 실제로 무는지 -- 원본 쪽 변이는 커밋 메시지)
    #[test]
    fn controls() {
        let (inp, js) = (fx("liqx_in.json"), fx("liqx_js.json"));
        let bites = |j: &Value, r: &Value| std::panic::catch_unwind(|| same(r, j, "")).is_err();
        let with = |i: usize, f: &dyn Fn(&mut Value)| { let mut c = inp[i].clone(); f(&mut c); run(&c) };
        // 오늘 ① 하루 경계: 지금을 하루 늦추면 어제 봉이 빠져 누적이 달라진다 ② 바이비트·HL 몫을 빼지 않으면(거래소 몫 지움) 선이 달라진다
        assert!(bites(&js[0]["today"], &with(0, &|c| c["now"] = json!(c["now"].as_f64().unwrap() + 86_400_000.0))["today"]), "오늘 ① 하루 경계");
        assert!(bites(&js[0]["today"], &with(0, &|c| for b in c["bars"].as_array_mut().unwrap() { b["bybit"] = Value::Null; b["hl"] = Value::Null; })["today"]), "오늘 ② 거래소 몫");
        assert!(bites(&js[0]["today"], &with(0, &|c| c["tab"]["band"] = Value::Null)["today"]), "오늘 ③ 띠");
        // 최대 고통 ① 지나간 봉우리: 5분봉을 빼면 고른 봉우리가 달라진다 ② 주기를 바꾸면 띠·굵은 칸이 달라진다 ③ 실시간 가격
        assert!(bites(&js[1]["pain"], &with(1, &|c| c["candles"] = json!([]))["pain"]), "고통 ① 지나간 봉우리");
        assert!(bites(&js[0]["pain"], &with(0, &|c| c["pp"] = json!("48H"))["pain"]), "고통 ② 주기");
        assert!(bites(&js[0]["kpi"], &with(0, &|c| c["live"]["eth"] = json!(3050))["kpi"]), "고통 ③ 가격");
        // 코인별 ① 다른 코인 가격(허브)을 안 쓰면 점이 달라진다 ② 계산 시각이 늦으면 흐림·old 가 달라진다 ③ 코인 차례
        assert!(bites(&js[0]["quad"], &with(0, &|c| c["live"]["btc"] = json!(0))["quad"]), "코인별 ① 허브 가격");
        assert!(bites(&js[5]["table"], &with(5, &|c| c["now"] = json!(c["all"]["coins"]["eth"]["generated_ms"].as_f64().unwrap() + 60_000.0))["table"]), "코인별 ② 오래됨");
        // ③ 코인 차례 = 응답 글 차례(serde_json 기본 Map 은 글자 차례) -- 지금 코인이 아닌 점·표 줄 차례가 그것을 따른다
        let mut x = State::new("24H").unwrap(); x.on_all(Some(r#"{"ok":true,"coins":{"xrp":{},"btc":{},"eth":{}}}"#));
        assert_eq!(x.coins.unwrap().iter().map(|q| q.0.as_str()).collect::<Vec<_>>(), ["xrp", "btc", "eth"], "코인별 ③ 응답 글 차례");
        // 같은 입력이면 맞는다
        same(&run(&inp[0]), &js[0], "같은 입력");
    }

    /// 묻기: 코인별 10분 · 5분봉 30초 · 숨으면 안 묻는다 · 실패·ok 아님 = 옛 값 · bars 가 배열이 아니면 옛 값 · 모르는 주기 = 카드 없음
    #[test]
    fn poll() {
        use crate::feed::{Feed, Req, Source, Urls};
        let mut f = Feed::new("sol", "SOLUSDT", Source::Push, Urls::default(), false);
        (f.rem_only, f.liqx) = (true, State::new(""));
        for (t, hid) in [(1000.0, false), (20_000.0, false), (31_000.0, false), (45_000.0, true), (61_500.0, false)] { f.tick(t, hid, false); }
        let urls: Vec<_> = f.out.iter().filter_map(|r| match r { Req::Get { url, .. } if !url.contains("market-history") => Some(url.as_str()), _ => None }).collect();
        assert_eq!(urls, ["/api/liq-tab?asset=sol", "/api/liq-tab?asset=all", "/api/liquidation-5m-history?asset=sol&bars=288", "/api/liquidation-5m-history?asset=sol&bars=288", "/api/liquidation-5m-history?asset=sol&bars=288"]);
        assert!(f.out.iter().any(|r| matches!(r, Req::WsSend(s) if s == r#"{"subscribe":[]}"#)), "허브 = «모든 종목» 가격만");
        let x = f.liqx.as_mut().unwrap();
        assert_eq!(x.pp, "24H");
        x.on_all(Some(r#"{"ok":true,"coins":{"sol":{"price":1}}}"#));
        x.on_all(Some(r#"{"ok":false,"reason":"not_built"}"#)); x.on_all(None); x.on_all(Some("<html>"));
        assert_eq!(x.coins.as_ref().unwrap().len(), 1, "ok 아님·실패 = 옛 값");
        x.on_bars(Some(r#"{"warmed_up":false,"bars":[{"ts":"2026-11-19T00:05:00+00:00","long_usd":5}]}"#));
        x.on_bars(Some(r#"{"error":"x"}"#)); x.on_bars(None);
        assert_eq!((x.bars.len(), x.bars[0].t, x.bars[0].l), (1, 1795046700.0, 5.0), "warmed_up 와 상관없이 받는다 · 배열 아님 = 옛 값");
        assert!(State::new("5D").is_none() && State::new("1M").is_some());
        assert!((reach(0.0) - 1.0).abs() < 1e-12 && (reach(1.0) - 0.3173).abs() < 1e-3, "24시간 도달 = 2(1−Φ(z))");
    }
}
