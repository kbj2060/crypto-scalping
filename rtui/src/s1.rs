//! 1초 수급 카드(지금 화면 «1초 수급» = app.js applySupply1s 4658 · mergedSupplySrc 293 · renderSupply1s 7128, 데스크톱 분기).
//! 자료 = rtcore 허브 /rt 의 snapshot.supply · supply1s(proto.rs Supply1s) 만. 줄 번호 = dashboard/live/app.js(이 작업 시점).
//! 안 옮긴 것: 청산 급증 배지(7349 · 시장 맥락 응답이 필요) · 휴대폰(compact) 분기 · 마우스 툴팁(<title>).
//! 🔴바이낸스 OI 는 허브에 없다(proto.rs 419) -- 지금 화면은 SSE 로 따로 받는다. 이 카드만 띄우면 OI 합은 비고 «OI —» 가 된다(같은 규칙: 293 oiSum 은 두 거래소가 다 있어야).
use crate::data::{fmt_qty, to_fixed};
use crate::fp::{Anchor, Clip, Col, Prim};
use serde_json::Value;
use std::collections::BTreeMap;

/// 191 SUPPLY_1S_SEGMENT -- 누적을 0 으로 되돌리는 벽시계 경계(5분봉과 같은 자리)
pub const SEG: i64 = 300;
/// 7198 SUPPLY_1S_GAP_SEC
const GAP: i64 = 5;
/// [리테일매수, 리테일매도, 고래매수, 고래매도, 총매수, 총매도, 가격]
pub type Row = [f64; 7];
/// [롱수량, 숏수량, 롱USD, 숏USD]
pub type Liq = [f64; 4];

#[derive(Default, Debug)]
pub struct State {
    pub bn: BTreeMap<i64, Row>, pub okx: BTreeMap<i64, Row>, pub spot: BTreeMap<i64, Row>,
    pub liq: BTreeMap<i64, Liq>, pub okx_liq: BTreeMap<i64, Liq>, pub bybit_liq: BTreeMap<i64, Liq>,
    pub oi: BTreeMap<i64, f64>, pub okx_oi: BTreeMap<i64, f64>,
    pub now: i64, pub okx_now: i64, pub spot_now: i64,
    pub okx_trade_age: Option<f64>, pub spot_trade_age: Option<f64>,
}

fn num(v: &Value) -> f64 { v.as_f64().unwrap_or(0.0) } // JS `c[i] || 0`
fn rows<const N: usize>(v: &Value, m: &mut BTreeMap<i64, [f64; N]>) {
    for r in v.as_array().into_iter().flatten() {
        let Some(a) = r.as_array() else { continue };
        let Some(s) = a.first().and_then(Value::as_f64) else { continue };
        let mut c = [0.0; N];
        for (i, x) in c.iter_mut().enumerate() { *x = a.get(i + 1).map_or(0.0, num); }
        m.insert(s as i64, c);
    }
}
fn pairs(v: &Value, m: &mut BTreeMap<i64, f64>) {
    for r in v.as_array().into_iter().flatten() {
        if let (Some(s), Some(x)) = (r[0].as_f64(), r[1].as_f64()) { m.insert(s as i64, x); }
    }
}
/// `Number(x) || old`
fn or_old(v: &Value, old: i64) -> i64 { match v.as_f64() { Some(x) if x != 0.0 => x as i64, _ => old } }

impl State {
    /// 4658 applySupply1s -- 허브 snapshot.supply 와 supply1s 메시지 본문이 같은 모양이다(같은 함수를 지난다)
    pub fn apply(&mut self, p: &Value) {
        rows(&p["seconds"], &mut self.bn);
        rows(&p["liq"], &mut self.liq);
        pairs(&p["oi"], &mut self.oi);
        rows(&p["okx"], &mut self.okx);
        rows(&p["okxLiq"], &mut self.okx_liq);
        rows(&p["bybitLiq"], &mut self.bybit_liq);
        pairs(&p["okxOi"], &mut self.okx_oi);
        rows(&p["spot"], &mut self.spot);
        // 4691 진행 중인 초 -- 칸만 채운다
        let part = &p["partial"];
        for (k, m) in [("bn", &mut self.bn), ("okx", &mut self.okx), ("spot", &mut self.spot)] {
            if part[k].is_array() { rows(&Value::Array(vec![part[k].clone()]), m); }
        }
        let age = |v: &Value| v.as_f64();
        (self.okx_now, self.okx_trade_age) = (or_old(&p["okxNow"], self.okx_now), age(&p["okxMeta"]["tradeAge"]));
        (self.spot_now, self.spot_trade_age) = (or_old(&p["spotNow"], self.spot_now), age(&p["spotMeta"]["tradeAge"]));
        self.now = or_old(&p["now"], self.now);
        // 4704 창 밖은 버린다(최악 now-300 · OI 갱신 3~7초라 +20 · 두 봉치)
        let keep = |now: i64| now - 2 * SEG - 20;
        let (f, fo, fs) = (keep(self.now), keep(self.okx_now), keep(self.spot_now));
        fn cut<T>(m: &mut BTreeMap<i64, T>, floor: i64) { *m = m.split_off(&floor); }
        cut(&mut self.bn, f); cut(&mut self.oi, f); cut(&mut self.liq, f); cut(&mut self.bybit_liq, f);
        cut(&mut self.okx, fo); cut(&mut self.okx_oi, fo); cut(&mut self.okx_liq, fo);
        cut(&mut self.spot, fs);
    }

    /// 293 mergedSupplySrc -- 본선 = 바이낸스 선물 + OKX(현물은 안 더한다) · 청산 = 셋 합 · OI = 두 거래소 계단 채움 합
    pub fn merged(&self) -> Src {
        let mut supply: BTreeMap<i64, Row> = BTreeMap::new();
        for m in [&self.bn, &self.okx] {
            for (s, c) in m {
                let t = supply.entry(*s).or_insert([0.0; 7]);
                for i in 0..6 { t[i] += c[i]; }
                if c[6] != 0.0 { t[6] = c[6]; }
            }
        }
        let mut liq: BTreeMap<i64, Liq> = BTreeMap::new();
        for m in [&self.liq, &self.okx_liq, &self.bybit_liq] {
            for (s, c) in m { let t = liq.entry(*s).or_insert([0.0; 4]); for i in 0..4 { t[i] += c[i]; } }
        }
        let now = self.now.max(self.okx_now);
        let ages = [("바이낸스", if self.now != 0 { Some((now - self.now) as f64) } else { None }), ("OKX", self.okx_trade_age)];
        let dead: Vec<&str> = ages.iter().filter(|(_, a)| a.is_none_or(|a| a > 10.0)).map(|(n, _)| *n).collect();
        let (mut sum, mut lb, mut lo) = (BTreeMap::new(), None, None);
        let mut keys: Vec<i64> = self.oi.keys().chain(self.okx_oi.keys()).copied().collect();
        keys.sort(); keys.dedup();
        for s in keys {
            if let Some(v) = self.oi.get(&s) { lb = Some(*v); }
            if let Some(v) = self.okx_oi.get(&s) { lo = Some(*v); }
            if let (Some(a), Some(b)) = (lb, lo) { sum.insert(s, a + b); }
        }
        let oi = if sum.is_empty() { self.oi.clone() } else { sum };
        let a = |x: Option<f64>| x.map_or("-".into(), |v| format!("{v}s"));
        let age = format!("체결 OKX {} · 현물 {}{}", a(self.okx_trade_age), a(self.spot_trade_age),
                          if dead.is_empty() { String::new() } else { format!(" · 합에서 빠짐: {}", dead.join("/")) });
        Src { supply, liq, oi, now, stale: !dead.is_empty(), age }
    }
}

pub struct Src { pub supply: BTreeMap<i64, Row>, pub liq: BTreeMap<i64, Liq>, pub oi: BTreeMap<i64, f64>, pub now: i64, pub stale: bool, pub age: String }

/// 선 모양 견본·선 = (색, 굵기, 불투명, 점선 [칠, 빔])
type Sw = (Col, f64, f64, Option<[f64; 2]>);
struct Pt { s: i64, v: f64 }

/// 7128 renderSupply1s(데스크톱) -- None = 이 봉의 초가 2개 미만(부르는 쪽: 지난 그림이 있으면 흐리게 남기고 없으면 wait_text)
/// tz_off_min = getTimezoneOffset() · measure = 글자 폭(글자, 크기, 굵기)
pub fn scene(st: &State, w: f64, h: f64, tz_off_min: i64, measure: &dyn Fn(&str, f64, u16) -> f64) -> Option<Vec<Prim>> {
    let sr = st.merged();
    let narrow = w < 760.0;
    let (mt, mb, ml, mr) = (6.0, 4.0, 100.0, 8.0);
    let cw = w - ml - mr;
    let (flow_top, flow_h) = (mt, h - mb - mt);
    let now = sr.now;
    let first = now.div_euclid(SEG) * SEG;
    let secs: Vec<i64> = sr.supply.range(first..=now).map(|(s, _)| *s).collect();
    if secs.len() < 2 { return None; }
    let mut v: Vec<Prim> = vec![];
    let x_at = |s: i64| ml + (s - first) as f64 / SEG as f64 * cw;
    let r1 = |x: f64| to_fixed(x, 1).parse::<f64>().unwrap_or(x); // toFixed(1)
    let text = |v: &mut Vec<Prim>, x: f64, y: f64, s: String, c: Col, anchor: Anchor, size: f64, wt: u16| v.push(Prim::Text { x, y, s, size, wt, c, a: 1.0, anchor, halo: false, clip: Clip::No });
    let seg = |l: [f64; 4], c: Col, a: f64, lw: f64, dash: Option<[f64; 2]>| match dash {
        Some([on, off]) => Prim::Dashed { pts: vec![[l[0], l[1]], [l[2], l[3]]], c, a, lw, on, off, clip: Clip::No },
        None => Prim::Line { pts: vec![[l[0], l[1]], [l[2], l[3]]], c, a, lw, dash: None, clip: Clip::No },
    };
    // 7203 pathOf -- 5초(또는 gap) 넘게 비거나 구간 경계면 끊는다(한 조각 = 그리기 하나 · 자료 선 = Clip::Plot 표식, 자르기 상자는 카드 전체)
    let path = |v: &mut Vec<Prim>, rows: &[Pt], y_of: &dyn Fn(f64) -> f64, gap: i64, sw: Sw| {
        let mut cur: Vec<[f64; 2]> = vec![];
        let mut flush = |cur: &mut Vec<[f64; 2]>| {
            if cur.len() >= 2 { let pts = std::mem::take(cur); v.push(match sw.3 { Some([on, off]) => Prim::Dashed { pts, c: sw.0, a: sw.2, lw: sw.1, on, off, clip: Clip::Plot }, None => Prim::Line { pts, c: sw.0, a: sw.2, lw: sw.1, dash: None, clip: Clip::Plot } }); }
            cur.clear();
        };
        let mut prev: Option<i64> = None;
        for r in rows {
            if prev.is_none_or(|p| r.s - p > gap || r.s.div_euclid(SEG) != p.div_euclid(SEG)) { flush(&mut cur); }
            cur.push([r1(x_at(r.s)), r1(y_of(r.v))]);
            prev = Some(r.s);
        }
        flush(&mut cur);
    };
    let qty = |x: f64| { let s = fmt_qty(x.abs()); if s.is_empty() { "0".into() } else { s } };
    let sgn = |x: f64| format!("{}{}", if x >= 0.0 { "+" } else { "−" }, qty(x));
    let cell = |s: i64| sr.supply.get(&s).copied().unwrap_or([0.0; 7]);
    let cum = |f: &dyn Fn(&Row) -> f64| { let mut a = 0.0; secs.iter().map(|s| { a += f(&cell(*s)); Pt { s: *s, v: a } }).collect::<Vec<_>>() };
    let cs_of = |rows: &[Pt]| rows.iter().fold(1e-9f64, |m, r| m.max(r.v.abs()));

    // 7255 OI: 경계 직전 마지막 관측(없으면 첫 관측)을 0 으로 둔 누적 증분
    let mut oi_base: Option<f64> = None;
    for (x, o) in sr.oi.range(..=now) { if *x <= first || oi_base.is_none() { oi_base = Some(*o); } }
    let oi_rows: Vec<Pt> = sr.oi.range(first + 1..=now).map(|(x, o)| Pt { s: *x, v: o - oi_base.unwrap_or(0.0) }).collect();
    let lane_h = flow_h / 2.0;
    // 7262 5초 넘게 빈 초 = «모름» 띠
    for p in secs.windows(2) {
        if p[1] - p[0] > GAP { v.push(Prim::Rect { x: x_at(p[0]), y: flow_top, w: (x_at(p[1]) - x_at(p[0])).max(1.0), h: flow_h, c: Col::Neutral, a: 0.07, rx: 0.0, clip: Clip::No }); }
    }
    const HEAD: f64 = 18.0;
    // 7277 laneFrame(데스크톱): 0선 · 구분선 · 왼쪽 칸 견본 + 이름(작게) + 값(14 · 800 · 부호 색)
    let lane_frame = |v: &mut Vec<Prim>, k: usize, items: &[(String, Sw)]| {
        let y0 = flow_top + k as f64 * lane_h;
        let zc = y0 + HEAD + (lane_h - HEAD) / 2.0;
        v.push(seg([ml, zc, ml + cw, zc], Col::Ink, 0.18, 1.0, None));
        if k > 0 { v.push(seg([4.0, y0, w - 4.0, y0], Col::Line, 1.0, 1.0, None)); }
        let fs = if narrow { 11.0 } else { 13.0 };
        let gap = fs + 9.0;
        let mut y = zc - items.len() as f64 * gap / 2.0 + fs - gap;
        for (txt, sw) in items {
            y += gap;
            v.push(seg([6.0, y - fs / 3.0, 18.0, y - fs / 3.0], sw.0, sw.2, sw.1.min(2.4), sw.3));
            let (name, val) = txt.split_once(' ').unwrap();
            text(v, 20.0, y, name.into(), Col::Muted, Anchor::Start, 10.5, 600);
            let c = if val.starts_with('+') { Col::Good } else if val.starts_with(['-', '−']) { Col::Bad } else { Col::Muted };
            text(v, ml - 2.0, y + 1.0, val.into(), c, Anchor::End, 14.0, 800);
        }
        (y0, zc, (lane_h - HEAD) / 2.0 - 4.0)
    };
    let scaled = |zc: f64, hh: f64, cs: f64| move |x: f64| zc - (x / cs) * hh * 0.9;

    // ── 줄 1: CVD · OI ──
    let cvd = cum(&|c| c[4] - c[5]);
    let acc = cvd.last().map_or(0.0, |r| r.v);
    let mut sa = 0.0;
    let spot: Vec<Pt> = st.spot.range(first..=now).map(|(x, c)| { sa += c[4] - c[5]; Pt { s: *x, v: sa } }).collect();
    let spot_on = spot.len() >= 2;
    let oi_end = oi_rows.last().map(|r| r.v);
    let mut items = vec![(format!("선물 {}", sgn(acc)), (Col::Ink, 1.8, 0.9, None))];
    if spot_on { items.push((format!("현물 {}", sgn(sa)), (Col::Spot, 1.6, 0.95, None))); }
    items.push((format!("OI {}", oi_end.map_or("—".into(), sgn)), (Col::Amber, 1.8, 0.95, None)));
    let (y0, zc, hh) = lane_frame(&mut v, 0, &items);
    { // 7369 머리 띠 시각(봉 시작 · 끝 · 지금) -- 배지가 없으니 시작 꼬리표는 왼쪽 칸 끝(ml)부터
        let hhmm = |t: i64| { let d = (t - tz_off_min * 60).rem_euclid(86400); format!("{:02}:{:02}", d / 3600, d / 60 % 60) };
        let (ty, t0) = (y0 + 14.0, format!("{} 봉 시작", hhmm(first)));
        let x0_end = ml + measure(&t0, 10.0, 500) + 8.0;
        text(&mut v, ml, ty, t0, Col::Muted, Anchor::Start, 10.0, 400);
        text(&mut v, ml + cw, ty, hhmm(first + SEG), Col::Muted, Anchor::End, 10.0, 400);
        if !narrow && ((now - first) as f64) < SEG as f64 * 0.82 && x_at(now) + 4.0 > x0_end {
            text(&mut v, x_at(now) + 4.0, ty, format!("지금 ({}초 경과)", now - first), Col::Muted, Anchor::Start, 10.0, 400);
        }
    }
    if cvd.len() >= 2 { path(&mut v, &cvd, &scaled(zc, hh, cs_of(&cvd)), GAP, (Col::Ink, 1.8, 0.9, None)); }
    if spot_on { path(&mut v, &spot, &scaled(zc, hh, cs_of(&spot)), 20, (Col::Spot, 1.6, 0.95, None)); }
    if oi_rows.len() >= 2 {
        let y_oi = scaled(zc, hh, cs_of(&oi_rows));
        path(&mut v, &oi_rows, &y_oi, 20, (Col::Amber, 1.8, 0.95, None));
        // 7384 oiAt: 그 초의 OI 누적선 값(20초 넘게 빈 사이는 가까운 쪽 · 아니면 선형)
        let oi_at = |sec: i64| {
            if sec <= oi_rows[0].s { return oi_rows[0].v; }
            for i in 1..oi_rows.len() {
                if oi_rows[i].s < sec { continue; }
                let (a, b) = (&oi_rows[i - 1], &oi_rows[i]);
                if b.s - a.s > 20 { return if b.s - sec <= sec - a.s { b.v } else { a.v }; }
                return a.v + (b.v - a.v) * (sec - a.s) as f64 / (b.s - a.s) as f64;
            }
            oi_rows[oi_rows.len() - 1].v
        };
        for x in &secs {
            let Some(lc) = sr.liq.get(x) else { continue };
            if !(lc[0] > 0.0 || lc[1] > 0.0) { continue; }
            let d = lc[1] - lc[0];
            v.push(Prim::Circle { x: r1(x_at(*x)), y: r1(y_oi(oi_at(*x))), r: r1(2.0 + (1.0 + d.abs()).log10() / 3001f64.log10() * 7.0),
                                  c: if d >= 0.0 { Col::Good } else { Col::Bad }, a: 0.92, clip: Clip::No });
        }
    }
    // ── 줄 2: 고래 · 중형 · 리테일 (눈금 공유) ── 7249 TIERS
    let tiers: [(&str, &dyn Fn(&Row) -> f64, f64, f64, Option<[f64; 2]>); 3] = [
        ("고래", &|c: &Row| c[2] - c[3], 2.4, 0.95, None),
        ("중형", &|c: &Row| (c[4] - c[2] - c[0]).max(0.0) - (c[5] - c[3] - c[1]).max(0.0), 1.7, 0.75, Some([6.0, 3.0])),
        ("리테일", &|c: &Row| c[0] - c[1], 1.5, 0.6, Some([2.0, 3.0])),
    ];
    let series: Vec<_> = tiers.iter().map(|(n, f, lw, op, dash)| (*n, cum(*f), (Col::Ink, *lw, *op, *dash))).collect();
    let items: Vec<(String, Sw)> = series.iter().map(|(n, r, sw)| (format!("{n} {}", sgn(r.last().map_or(0.0, |p| p.v))), *sw)).collect();
    let (_, zc, hh) = lane_frame(&mut v, 1, &items);
    let cs = series.iter().fold(1e-9f64, |m, (_, r, _)| m.max(cs_of(r)));
    for (_, r, sw) in &series { if r.len() >= 2 { path(&mut v, r, &scaled(zc, hh, cs), GAP, *sw); } }
    // 7426 «지금» 세로선 · 아직 안 온 시간 음영
    let nx = x_at(now);
    v.push(seg([nx, flow_top, nx, flow_top + flow_h], Col::Ink, 0.35, 1.0, Some([2.0, 3.0])));
    v.push(Prim::Rect { x: nx, y: flow_top, w: (ml + cw - nx).max(0.0), h: flow_h, c: Col::Lift, a: 0.05, rx: 0.0, clip: Clip::No });
    // 7440 원천이 죽었을 때만 빨간 경고
    if sr.stale { text(&mut v, ml + 4.0, flow_top + 12.0, format!("합산 · 바이낸스 선물 · OKX  ·  {}", sr.age), Col::Bad, Anchor::Start, 10.0, 400); }
    Some(v)
}

/// 7183 초가 2개 미만이고 지난 그림도 없을 때
pub fn wait_text(w: f64, h: f64) -> Vec<Prim> {
    vec![Prim::Text { x: w / 2.0, y: h / 2.0, s: "체결 테이프 집계 중...".into(), size: 16.0, wt: 400, c: Col::Muted, a: 1.0, anchor: Anchor::Middle, halo: false, clip: Clip::No }]
}

#[cfg(test)]
mod t {
    use super::*;
    use serde_json::json;

    fn texts(v: &[Prim]) -> Vec<String> { v.iter().filter_map(|p| if let Prim::Text { s, .. } = p { Some(s.clone()) } else { None }).collect() }
    fn lines(v: &[Prim]) -> Vec<(Vec<[f64; 2]>, Option<[f64; 2]>)> {
        v.iter().filter_map(|p| match p { Prim::Line { pts, clip: Clip::Plot, .. } => Some((pts.clone(), None)), Prim::Dashed { pts, on, off, clip: Clip::Plot, .. } => Some((pts.clone(), Some([*on, *off]))), _ => None }).collect()
    }

    /// 허브 스냅숏 supply 견본(봉 시작 600 · now 610): 바이낸스 600·601·602·609(602→609 = 7초 빔) + OKX 601 + 현물 · OKX OI · 청산
    fn snap() -> Value {
        json!({"now": 610, "retailMaxUsd": 1.0, "whaleMinUsd": 2.0,
            "seconds": [[299, 9, 9, 9, 9, 9, 9, 1.0], [600, 1, 0, 0, 0, 10, 2, 100.0], [601, 0, 1, 5, 0, 6, 1, 101.0], [602, 0, 0, 0, 3, 0, 4, 0]],
            "partial": {"bn": [609, 0, 0, 0, 0, 1, 0, 102.0], "okx": null, "spot": null},
            "liq": [[601, 2.0, 0.0, 200, 0]], "spot": [[600, 0, 0, 0, 0, 3, 1, 100.0], [605, 0, 0, 0, 0, 0, 1, 100.0]], "spotNow": 610,
            "spotMeta": {"connected": true, "tradeAge": 0.4, "errors": 0},
            "okx": [[601, 0, 0, 0, 0, 2, 0, 0]], "okxNow": 609, "okxMeta": {"connected": true, "tradeAge": 1.0, "oiAge": 2.0, "inst": "ETH-USDT-SWAP", "errors": 0},
            "okxLiq": [], "okxOi": [[590, 1000.0], [603, 1004.0]], "bybitLiq": [[601, 1.0, 0.0, 100, 0]]})
    }

    #[test]
    fn apply_and_merge() {
        let mut st = State::default();
        st.apply(&snap());
        // 창 밖(299 < 610-620 = -10 아님 → 남음 · 아래 now 를 밀면 빠진다) · partial 도 칸에 들어간다
        assert_eq!(st.bn.keys().copied().collect::<Vec<_>>(), vec![299, 600, 601, 602, 609]);
        let m = st.merged();
        assert_eq!(m.supply[&601], [0.0, 1.0, 5.0, 0.0, 8.0, 1.0, 101.0]); // 바이낸스 + OKX(가격은 0 이 아닌 쪽)
        assert_eq!(m.liq[&601], [3.0, 0.0, 300.0, 0.0]); // 바이낸스 + Bybit
        assert!(m.oi.is_empty(), "바이낸스 OI 가 없으면 합도 없다 -- 293 oiMain = oi1s(빈 것)");
        assert_eq!((m.now, m.stale), (610, false));
        assert_eq!(m.age, "체결 OKX 1s · 현물 0.4s");
        // 증분(supply1s): 같은 초는 덮어쓰고 now 가 밀리면 창 밖(< now-620)을 버린다 · okxMeta.tradeAge 없음 = 죽음
        st.apply(&json!({"now": 1000, "seconds": [[600, 2, 0, 0, 0, 20, 0, 0]], "partial": {}, "okxMeta": {"tradeAge": null}}));
        assert_eq!((st.bn.keys().next(), st.bn[&600][0], st.okx_now), (Some(&600), 2.0, 609));
        let m = st.merged();
        assert!(m.stale && m.age.ends_with("합에서 빠짐: OKX"), "{}", m.age);
        st.apply(&json!({"now": 1300}));
        assert!(st.bn.is_empty());
    }

    #[test]
    fn scene_values() {
        let mut st = State::default();
        st.apply(&snap());
        st.oi.insert(595, 50.0); st.oi.insert(605, 52.0); // 바이낸스 OI(지금 화면은 SSE 로 받는 칸) -- 합 = 계단 채움
        let ms = |s: &str, z: f64, _: u16| s.chars().count() as f64 * z * 0.5;
        let v = scene(&st, 400.0, 300.0, -540, &ms).unwrap();
        let t = texts(&v);
        // CVD = Σ(총매수−총매도) 600~609 = 8 + 7 − 4 + 1 = 12 · 현물 = 2 − 1 = 1 · OI = (52+1004) − (50+1000) = 6
        // 고래 = 5 − 3 = 2 · 리테일 = 1 − 1 = 0 · 중형 = Σ max(0,총매수−고래−리테일) − Σ max(0,총매도−…) = (9+3+0+1) − (2+0+1+0) = 10
        for s in ["선물", "+12.0", "현물", "+1.0", "OI", "+6.0", "고래", "+2.0", "중형", "+10.0", "리테일", "+0", "09:10 봉 시작", "09:15"] { assert!(t.contains(&s.to_string()), "{s} 없음: {t:?}"); }
        assert!(!t.iter().any(|s| s.starts_with("지금")), "좁은 칸(<760)에서는 «지금» 꼬리표 없음");
        // 선: 602 → 609 는 7초 빔 → CVD 가 두 조각(600~602 만 3점 · 609 한 점은 안 그림)
        let ls = lines(&v);
        let cvd = &ls[0].0;
        assert_eq!(cvd.len(), 3);
        assert_eq!(cvd[0][0], 100.0); // xAt(600) = ml
        assert_eq!(cvd[1][0], 101.0); // 100 + 1/300 × 292 = 100.973 → toFixed(1)
        // 줄 1: zc = 6 + 18 + (145−18)/2 = 87.5 · hh = 59.5 · cs = 15 → 600 의 y = 87.5 − 8/15 × 59.5 × .9 = 58.94
        assert_eq!(cvd[0][1], 58.9);
        // «모름» 띠 하나(602 → 609)
        assert_eq!(v.iter().filter(|p| matches!(p, Prim::Rect { c: Col::Neutral, .. })).count(), 1);
        // 청산 점(601): 롱 3 → v = −3 → 빨강 · r = 2 + log10(4)/log10(3001) × 7 = 3.2
        let c: Vec<_> = v.iter().filter_map(|p| if let Prim::Circle { r, c, .. } = p { Some((*r, *c)) } else { None }).collect();
        assert_eq!(c, vec![(3.2, Col::Bad)]);
        // 중형·리테일 점선 모양
        assert!(ls.iter().any(|l| l.1 == Some([6.0, 3.0])) && ls.iter().any(|l| l.1 == Some([2.0, 3.0])));
        // 초 하나뿐 → None(부르는 쪽이 흐리게/집계 중)
        let mut one = State::default();
        one.apply(&json!({"now": 900, "seconds": [[900, 0, 0, 0, 0, 1, 0, 1.0]]}));
        assert!(scene(&one, 400.0, 300.0, 0, &ms).is_none());
    }

    /// 지금 화면의 실제 함수로 그린 정답지(compare/s1_js.mjs → fixture/s1_js.json · 같은 입력 s1_in.json · 글자 폭 = 글자 수 × 크기 × .5)
    #[test]
    fn js_parity() {
        let fx = |f: &str| serde_json::from_str::<Value>(&std::fs::read_to_string(format!("{}/compare/fixture/{f}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap();
        let (inp, js) = (fx("s1_in.json"), fx("s1_js.json"));
        let mut st = State::default();
        for m in inp.as_array().unwrap() { st.apply(m); }
        let ms = |s: &str, z: f64, _: u16| s.chars().count() as f64 * z * 0.5;
        for w in [400.0, 900.0] {
            let v = scene(&st, w, 300.0, -540, &ms).unwrap();
            let j = &js[format!("w{w}")];
            let jt: Vec<String> = j["texts"].as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()).filter(|t| !t.is_empty()).collect();
            assert_eq!(texts(&v), jt, "글자 w={w}");
            let rp: Vec<(Vec<[f64; 2]>, Option<[f64; 2]>, f64)> = v.iter().filter_map(|p| match p {
                Prim::Line { pts, lw, clip: Clip::Plot, .. } => Some((pts.clone(), None, *lw)),
                Prim::Dashed { pts, lw, on, off, clip: Clip::Plot, .. } => Some((pts.clone(), Some([*on, *off]), *lw)),
                _ => None }).collect();
            let jp: Vec<(Vec<[f64; 2]>, Option<[f64; 2]>, f64)> = j["paths"].as_array().unwrap().iter().map(|p| (
                serde_json::from_value(p["pts"].clone()).unwrap(),
                p["dash"].as_str().map(|d| { let a: Vec<f64> = d.split(' ').map(|x| x.parse().unwrap()).collect(); [a[0], a[1]] }),
                p["lw"].as_f64().unwrap())).collect();
            assert_eq!(rp.len(), jp.len(), "선 조각 수 w={w}");
            for (a, b) in rp.iter().zip(&jp) { assert_eq!(a, b, "선 w={w}"); }
            let rc: Vec<[f64; 3]> = v.iter().filter_map(|p| if let Prim::Circle { x, y, r, .. } = p { Some([*x, *y, *r]) } else { None }).collect();
            assert_eq!(rc, serde_json::from_value::<Vec<[f64; 3]>>(j["circles"].clone()).unwrap(), "청산 점 w={w}");
            let rr: Vec<[f64; 4]> = v.iter().filter_map(|p| if let Prim::Rect { x, y, w, h, .. } = p { Some([*x, *y, *w, *h]) } else { None }).collect();
            let jr: Vec<[f64; 4]> = serde_json::from_value(j["rects"].clone()).unwrap();
            assert_eq!(rr.len(), jr.len());
            for (a, b) in rr.iter().zip(&jr) { assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9), "띠 {a:?} 대 {b:?}"); }
        }
    }
}
