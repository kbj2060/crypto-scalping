//! 차트 안 표식(fg 캔버스 · 가격 플롯 자르기): 스택 불균형 막대 · 미완 경매 꺾쇠(풋프린트 자료만) · 청산 동그라미(청산 5분봉 응답).
use crate::data::{Candle, lv};
use crate::fp::{Clip, Col, Geo, Prim};
use std::collections::BTreeMap;

/// 9769-9791 스택 불균형: 같은 봉에서 반대편의 3배 넘는 줄(FOOTPRINT_IMBALANCE_RATIO 150)이 같은 쪽으로 3줄 이상 이어지면 바깥에 굵은 막대(매수 = 봉 오른쪽 · 매도 = 왼쪽)
pub fn stack(g: &Geo, bar_rows: &[BTreeMap<i64, [f64; 2]>]) -> Vec<Prim> {
    let mut v = vec![];
    for (i, rows) in bar_rows.iter().enumerate() {
        let x = g.x_at(i);
        for (s, o, ex, col) in [(0, 1, x + g.bw + 3.0, Col::Good), (1, 0, x - 5.0, Col::Bad)] {
            let mut run: Vec<i64> = vec![];
            let mut flush = |run: &mut Vec<i64>| {
                if run.len() >= 3 {
                    let y_top = g.mt.max(g.y_at(g.row_lo(run[run.len() - 1] + 1)));
                    let y_bot = g.plot_bottom.min(g.y_at(g.row_lo(run[0])));
                    if y_bot - y_top > 2.0 { v.push(Prim::Rect { x: ex, y: y_top, w: 2.0, h: y_bot - y_top, c: col, a: 1.0, rx: 1.0, clip: Clip::Plot }); }
                }
                run.clear();
            };
            for (&k, c) in rows { // BTreeMap = 키 오름차순(9773 sort)
                let imb = c[s] > 0.0 && c[s] > c[o] * 3.0;
                if imb && !run.is_empty() && k != run[run.len() - 1] + 1 { flush(&mut run); }
                if imb { run.push(k); } else { flush(&mut run); }
            }
            flush(&mut run);
        }
    }
    v
}

/// 11332-11350 미완 경매: 닫힌 봉 고가·저가 칸에 매수·매도가 둘 다 체결됐는데 뒤 봉이 아직 안 간 가격 -- 고 = 봉 위 ︿ · 저 = 봉 아래 ﹀.
/// 지금가 0.1% 안쪽은 생략 · 같은 가격은 먼저 생긴 것 하나만. ponytail: DENSE(1주일 창)에선 끈다(JS 도 끈다 -- 여기선 창이 145봉 밑이라 늘 켜짐).
pub fn unfinished(g: &Geo, candles: &[Candle], by_time: &BTreeMap<i64, &Vec<Vec<f64>>>, bucket: f64) -> Vec<Prim> {
    let mut lines: Vec<(f64, f64, bool)> = vec![]; // x0, 가격, 고?
    if candles.len() < 2 { return vec![]; }
    let cp = if g.current != 0.0 && !g.current.is_nan() { g.current } else { candles[candles.len() - 1].close };
    let hq = bucket / 2.0;
    for (i, c) in candles.iter().enumerate() {
        if c.partial() || i == candles.len() - 1 { continue; }
        let r: Vec<&Vec<f64>> = by_time.get(&c.time).map_or(vec![], |lv_| lv_.iter().filter(|l| lv(l, 1) + lv(l, 2) > 0.0).collect());
        if r.is_empty() { continue; }
        let later = &candles[i + 1..];
        let open = |p: f64| !later.iter().any(|x| x.low <= p + hq && p - hq <= x.high) && (p / cp - 1.0).abs() >= 0.001;
        let top = r.iter().copied().reduce(|a, l| if l[0] > a[0] { l } else { a }).unwrap();
        let bot = r.iter().copied().reduce(|a, l| if l[0] < a[0] { l } else { a }).unwrap();
        for (l, hi) in [(top, true), (bot, false)] {
            let p = l[0];
            if lv(l, 1) > 0.0 && lv(l, 2) > 0.0 && open(p) && !lines.iter().any(|q| q.1 == p) { lines.push((g.x_at(i) + g.bw / 2.0, p, hi)); }
        }
    }
    lines.into_iter().map(|(x0, p, hi)| {
        let y = (g.y_at(p) + 0.5).floor() + 0.5; // Math.round(yAt(p)) + 0.5 (JS round = 반은 위로)
        let s = (g.bw * 0.35).min(5.0).max(3.5);
        let (yc, d) = if hi { (y - 7.0, 1.0) } else { (y + 7.0, -1.0) };
        Prim::Line { pts: vec![[x0 - s, yc + d * s * 0.6], [x0, yc - d * s * 0.6], [x0 + s, yc + d * s * 0.6]], c: Col::Text, a: 0.85, lw: 1.6, dash: None, clip: Clip::Plot }
    }).collect()
}

/// 11226-11289 청산 동그라미: 봉이 그린 가장 높은 가격(고가 · 서버 봉 칸 맨 위 가격) 위 14px · 반지름 √(금액/창 안 최대) · 숏 청산이 같거나 크면 초록 · 불투명 0.9
/// lane = 서버 봉 칸(fpBars = tfFpBars(fpBarsNow())) · by_ts = 봉 시각(초) → 청산 봉
/// DENSE(1주일 창) = 지금 화면은 색별 한 칠(cvUnion 8199 · 처음 본 색 순서) -- 같은 순서로 낸다. ponytail: 한 칠이라 겹친 원이 진해지지 않는 것은 못 맞춤(좌표는 같고 겹친 곳 픽셀만 다름) -- 필요하면 색별 메시로 묶는다
pub fn liq(g: &Geo, candles: &[Candle], lane: &std::collections::HashMap<i64, (&Vec<Vec<f64>>, Option<[f64; 3]>)>, by_ts: &std::collections::HashMap<i64, &crate::data::LiqBar>, dense: bool) -> Vec<Prim> {
    let usd = |c: &Candle| by_ts.get(&c.time).map(|b| (b.long_usd, b.short_usd));
    let peak = candles.iter().filter_map(usd).fold(0.0f64, |m, (l, s)| m.max(l + s));
    if !(peak > 0.0) { return vec![]; }
    let r_max = 13.0; // 데스크톱(mobileChart 거짓)
    let (r0, r1) = if g.slot < 4.0 { (1.5, r_max * 0.6) } else { (3.0, r_max) };
    let mut v = vec![];
    for (i, c) in candles.iter().enumerate() {
        let Some((lo, sh)) = usd(c) else { continue };
        if !(lo + sh > 0.0) { continue; }
        let r = r0 + ((lo + sh) / peak).sqrt() * (r1 - r0);
        let top_lv = lane.get(&c.time).map(|(lv, _)| lv.iter().map(|l| lv0(l)).fold(f64::NEG_INFINITY, f64::max)).filter(|t| t.is_finite());
        let top = c.high.max(top_lv.unwrap_or(f64::NEG_INFINITY));
        let cy = (g.mt + r + 1.0).max(g.y_at(top) - r - 14.0);
        v.push(Prim::Circle { x: g.x_at(i) + g.bw / 2.0, y: cy, r, c: if sh >= lo { Col::Good } else { Col::Bad }, a: 0.9, clip: Clip::Plot });
    }
    if dense {
        let col = |p: &Prim| if let Prim::Circle { c, .. } = p { *c } else { Col::Good };
        if let Some(first) = v.first().map(col) { v.sort_by_key(|p| col(p) != first); } // 안정 정렬 = 색마다 봉 순서 그대로
    }
    v
}
/// Number(l[0]) -- NaN 이면 최댓값 후보에서 빠진다(pz > top 이 거짓)
fn lv0(l: &[f64]) -> f64 { l.first().copied().filter(|x| !x.is_nan()).unwrap_or(f64::NEG_INFINITY) }
