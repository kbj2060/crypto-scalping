//! 풋프린트 아래 레인(플롯 안 바닥 · 데스크톱 2단 LANE_IN): 봉 색 줄(리본) · 봉 안 델타 캔들 · 거래대금 선 · 범례. app.js 11123-11221.
//! ponytail: 신규계약(OI) 응답이 없는 경로만(리본 = «매수/매도 우위» 흐린 칸) -- OI 를 받으면 11184-11185 의 꽉 찬 칸·테두리 칸과 QNAME 넷을 더한다.
//! DENSE(봉 칸 8px 밑): 리본·델타 몸통 = 색·농도별 한 칠(cvUnion 8199) -- 여기선 같은 순서의 사각형 목록(겹치는 1px 밑 델타 몸통은 egui 가 겹쳐 칠해 조금 진하다: 1주일 창만).
use crate::data::{Candle, lv};
use crate::fp::{Anchor, Clip, Col, Geo, Prim};
use std::collections::HashMap;

pub struct Row { pub turn: f64, pub delta: f64, pub dhl: Option<[f64; 3]> }

/// laneBarsOf 11134-11147 + supplyFlowOfBar 6852 (더하는 순서까지 같게 -- 델타 = 고래 + 중형 + 리테일)
/// by_t = fpBars 시각 → (칸, 봉 안 델타) -- 부르는 쪽이 같은 시각이면 먼저 온 것으로 만든다(11136)
pub fn rows(by_t: &HashMap<i64, (&Vec<Vec<f64>>, Option<[f64; 3]>)>, candles: &[Candle]) -> Vec<Option<Row>> {
    candles.iter().map(|c| by_t.get(&c.time).map(|&(levels, dhl)| {
        let (mut buy, mut sell, mut wb, mut ws, mut rb, mut rs, mut turn) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for l in levels { buy += lv(l, 1); sell += lv(l, 2); wb += lv(l, 3); ws += lv(l, 4); rb += lv(l, 5); rs += lv(l, 6); }
        for l in levels { turn += lv(l, 0) * (lv(l, 1) + lv(l, 2)); }
        let (whale, retail, mid) = (wb - ws, rb - rs, (buy - wb - rb) - (sell - ws - rs));
        Row { turn, delta: whale + mid + retail, dhl }
    })).collect()
}

/// cvUnion: 색·농도 키를 처음 넣은 순서대로, 키마다 넣은 순서의 사각형
fn union(v: Vec<(Col, f64, [f64; 4])>, l: Clip) -> Vec<Prim> {
    let mut keys: Vec<(Col, f64)> = vec![];
    for e in &v { if !keys.contains(&(e.0, e.1)) { keys.push((e.0, e.1)); } }
    keys.iter().flat_map(|k| v.iter().filter(move |e| (e.0, e.1) == *k).map(move |e| Prim::Rect { x: e.2[0], y: e.2[1], w: e.2[2], h: e.2[3], c: k.0, a: k.1, rx: 0.0, clip: l })).collect()
}

/// 11130 QNAME(OI 모름)
pub fn qname(d: f64) -> &'static str { if d >= 0.0 { "매수 우위" } else { "매도 우위" } }

/// (fg 캔버스 칠하기, svg 글자)
pub fn draw(g: &Geo, rows: &[Option<Row>], measure: &dyn Fn(&str, f64) -> f64) -> (Vec<Prim>, Vec<Prim>) {
    let (mut fg, mut svg) = (vec![], vec![]);
    if !rows.iter().any(Option::is_some) { return (fg, svg); } // 11160
    let l = Clip::Lane;
    let rect = |x, y, w, h, c, a, rx| Prim::Rect { x, y, w, h, c, a, rx, clip: l };
    // 11167-11168 레인 뒤를 차트 바탕색으로 막고 위에 경계선(cvHLine = 폭 1 사각형)
    fg.push(rect(g.ml, g.quad_y - 8.0, g.cw, g.plot_bottom - g.quad_y + 8.0, Col::ChartBg, 1.0, 0.0));
    fg.push(rect(g.ml, g.quad_y - 8.0 - 0.5, g.cw, 1.0, Col::Line, 1.0, 0.0));
    let (top, bot, rib, gap) = (g.quad_y, g.plot_bottom - 20.0, 22.0, 8.0); // 11170
    let chart_top = top + rib + gap;
    // 11176-11189 리본: OI 모름 = 불투명 0.12 · 둥근 3 · 칸 52px 이상이면 이름(svg 12px 700, 레인 자르기)
    let dense = g.slot < 8.0; // 8709 DENSE
    let mut un = vec![];
    let mut rib_col: Vec<(i64, usize)> = vec![]; // 11173 1px 밑: 픽셀 열마다 마지막 봉
    for (i, r) in rows.iter().enumerate() {
        let Some(r) = r else { continue };
        let col = if r.delta >= 0.0 { Col::Good } else { Col::Bad };
        if g.slot < 1.0 {
            let k = (g.x_at(i) + 0.5).floor() as i64;
            match rib_col.iter_mut().find(|e| e.0 == k) { Some(e) => e.1 = i, None => rib_col.push((k, i)) }
            continue;
        }
        if dense { un.push((col, 0.12, [g.x_at(i), top, g.slot.max(0.6), rib])); continue; }
        let (x, wd) = (g.x_at(i) + 1.0, (g.slot - 2.0).max(1.0));
        fg.push(rect(x, top, wd, rib, col, 0.12, 3.0));
        if g.slot >= 52.0 {
            svg.push(Prim::Text { x: x + wd / 2.0, y: top + rib / 2.0 + 4.5, s: qname(r.delta).into(), size: 12.0, c: col, a: 1.0, anchor: Anchor::Middle, halo: false, clip: l });
        }
    }
    for (k, i) in rib_col { let r = rows[i].as_ref().unwrap(); un.push((if r.delta >= 0.0 { Col::Good } else { Col::Bad }, 0.12, [k as f64, top, 1.0, rib])); }
    fg.extend(union(std::mem::take(&mut un), l));
    // 11194-11207 봉 안 델타 캔들: 몸통 = 0 → 마감 · 꼬리 = 봉 안 최고·최저 · √ 눈금
    let (lg, cx) = (12.0, |i: usize| g.x_at(i) + g.bw / 2.0);
    let lh = bot - chart_top - lg;
    let (y0, zero) = (chart_top + lg, chart_top + lg + lh / 2.0);
    let bwid = (g.slot * 0.62).min(g.slot - 2.0).max(1.0);
    let dh: Vec<Option<(f64, f64, f64, bool)>> = rows.iter().map(|r| r.as_ref().map(|r| match r.dhl {
        Some(d) => (d[0].min(0.0), d[1].max(0.0), d[2], true),
        None => (r.delta.min(0.0), r.delta.max(0.0), r.delta, false),
    })).collect();
    let d_max = dh.iter().flatten().fold(1e-9f64, |m, d| m.max((-d.0).max(d.1)));
    let yd = |v: f64| zero - js_sign(v) * (v.abs() / d_max).sqrt() * (lh / 2.0 - 1.0);
    fg.push(rect(g.ml, zero - 0.5, g.cw, 1.0, Col::Muted, 0.35, 0.0));
    for (i, d) in dh.iter().enumerate() {
        let Some((lo, hi, c, wick)) = *d else { continue };
        let (x, col) = (cx(i), if c >= 0.0 { Col::Good } else { Col::Bad });
        let (yb, hb, ww) = (zero.min(yd(c)), (yd(c) - zero).abs().max(1.0), (bwid * 0.2).min(3.0).max(1.5));
        if dense { un.push((col, 0.85, [x - bwid / 2.0, yb, bwid, hb])); continue; }
        if wick { fg.push(rect(x - ww / 2.0, yd(hi), ww, (yd(lo) - yd(hi)).max(1.0), col, 0.75, 1.0)); }
        fg.push(rect(x - bwid / 2.0, yb, bwid, hb, col, 0.85, 1.5));
    }
    fg.extend(union(un, l));
    // 11210-11214 거래대금 선(제 축 · 바닥 0) -- 값 없는 봉에서 끊는다 · 점 하나면 2px 사각형
    let u_max = rows.iter().flatten().fold(1e-9f64, |m, r| m.max(r.turn));
    let u_sc = (lh - 2.0) / u_max;
    let mut runs: Vec<Vec<[f64; 2]>> = vec![];
    let mut open = false;
    for (i, r) in rows.iter().enumerate() {
        match r {
            None => open = false,
            Some(r) => {
                if !open { runs.push(vec![]); open = true; }
                runs.last_mut().unwrap().push([cx(i), y0 + lh - 1.0 - r.turn * u_sc]);
            }
        }
    }
    for pts in runs {
        if pts.len() > 1 { fg.push(Prim::Line { pts, c: Col::Turnover, a: 0.9, lw: 1.4, dash: None, clip: l }); }
        else { fg.push(rect(pts[0][0] - 1.0, pts[0][1] - 1.0, 2.0, 2.0, Col::Turnover, 0.9, 0.0)); }
    }
    // 11215-11216 범례(svg 10px · 두 tspan -- 겹 빈칸은 하나로 접힌다)
    let (lx, ly, t1) = (g.ml + 4.0, chart_top + 9.0, "▮ 봉 안 델타 (몸통 = 마감 · 꼬리 = 봉 안 최고·최저 · √ 눈금)");
    svg.push(Prim::Text { x: lx, y: ly, s: t1.into(), size: 10.0, c: Col::Muted, a: 1.0, anchor: Anchor::Start, halo: false, clip: Clip::No });
    svg.push(Prim::Text { x: lx + measure(t1, 10.0), y: ly, s: " ─ 거래대금".into(), size: 10.0, c: Col::Turnover, a: 1.0, anchor: Anchor::Start, halo: false, clip: Clip::No });
    (fg, svg)
}

/// Math.sign(0) = 0 (Rust signum(0.0) = 1)
fn js_sign(v: f64) -> f64 { if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 } }
