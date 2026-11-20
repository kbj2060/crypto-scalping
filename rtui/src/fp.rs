//! 가공: 응답 → 그리기 목록(app.js renderCandleSvg 의 풋프린트 차트 부분, 데스크톱 넓은 화면 2단 경로 그대로).
//! 좌표는 svg viewBox 단위(= 데스크톱 dpr 1 의 css px, 상자 왼쪽 위 기준). 줄 번호 = dashboard/live/app.js(이 작업 시점).
//! 경로 고정(ponytail: 이 카드가 쓰는 한 경로만 -- 휴대폰·1단·전체화면(fp-wide)·끌기·세로 이동·30m/1h 묶기는 없다 — 그 카드를 옮길 때 같은 식으로 더한다):
//!   데스크톱(isMobileChartMode 거짓) · ETH 흐름 코인(flowOn) · 넓은 화면 2단(--fp-split 0.25) · 체결 버블 띠(FP_BUB) · 실시간(chartPanEnd null, chartPanRight 0, chartPanY 0) · 5m.
use crate::data::{Candle, Footprint, fmt_qty, to_fixed};
use std::collections::BTreeMap;

/// styles.css :root(다크) 의 색 이름
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Col { Good, Bad, Muted, Amber, Accent, Ink, Line, OnFill, ChartBg, Text, Turnover, SoftLine, HoverLine }

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Anchor { Start, Middle, End }

/// 자르기: Plot = 가격 플롯(app.js 8770 plotC) · Lane = 레인(8770 laneC: 레인 윗변 ~ 상자 바닥)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Clip { No, Plot, Lane }

/// 그리기 한 개 = app.js 의 캔버스 붓(cvRect·cvStrokeRect·cvStroke·cvText)·svg 요소 하나.
#[derive(Debug)]
pub enum Prim {
    /// rx = 둥근 모서리(캔버스 roundRect · svg rx -- 반지름은 min(rx, 폭/2, 높이/2))
    Rect { x: f64, y: f64, w: f64, h: f64, c: Col, a: f64, rx: f64, clip: Clip },
    StrokeRect { x: f64, y: f64, w: f64, h: f64, c: Col, a: f64, lw: f64, clip: Clip },
    Line { pts: Vec<[f64; 2]>, c: Col, a: f64, lw: f64, dash: Option<f64>, clip: Clip },
    /// y = 글자 바닥선(캔버스 fillText·svg <text> 와 같은 뜻) · halo = svg paint-order stroke 3px 차트 바탕색(기둥 머리 글자)
    Text { x: f64, y: f64, s: String, size: f64, c: Col, a: f64, anchor: Anchor, halo: bool, clip: Clip },
    Circle { x: f64, y: f64, r: f64, c: Col, a: f64, clip: Clip },
}

pub struct Cfg {
    /// svg viewBox 크기(app.js 8402-8403 wAll·hAll = 부모 폭 · svg 상자 높이) -- 배치(카드 밖)가 정한다
    pub w_all: f64,
    pub h_all: f64,
    /// styles.css 2262 #candleSvgSnapshot --fp-split
    pub split: f64,
    /// 창 봉 수(app.js 118 FP_BARS 36 · 버튼·휠이 바꾼다)
    pub window: f64,
    /// 브라우저 new Date().getTimezoneOffset()(분, 서울 −540)
    pub tz_off_min: i64,
}

/// 한 장의 기하(svg viewBox 단위) -- 레인·프로파일·표식·호버가 같은 값을 읽는다
pub struct Geo {
    pub w: f64, pub h: f64, pub ml: f64, pub mr: f64, pub mt: f64, pub mb: f64, pub cw: f64, pub ch: f64, pub plot_bottom: f64,
    pub quad_y: f64, pub lane_top: f64,
    pub y_min: f64, pub y_max: f64, pub y_span: f64, pub slot: f64, pub x0b: f64, pub bw: f64,
    pub row_size: f64, pub off: f64, pub row_px: f64, pub current: f64, pub draw_cells: bool,
    /// 프로파일 칸(8584-8588): 캔들 플롯 오른쪽 끝 · 체결 칸 폭 · 델타·TPO 칸 폭 · TPO 왼쪽 끝 · 델타 왼쪽 끝 · 체결 바닥선
    pub trade_x0: f64, pub trade_w: f64, pub dt_w: f64, pub c_tpo: f64, pub c_del: f64, pub t_anc: f64,
    /// 현재가 알약·호버 가격 배지 오른쪽 끝(10354 xRR)
    pub x_rr: f64,
}

impl Geo {
    pub fn x_at(&self, i: usize) -> f64 { self.ml + (i as f64 + self.x0b) * self.slot }
    pub fn y_at(&self, v: f64) -> f64 { self.mt + (self.y_max - v) * self.ch / self.y_span }
    pub fn row_of(&self, p: f64) -> i64 { ((p + self.off) / self.row_size).floor() as i64 }
    pub fn row_lo(&self, k: i64) -> f64 { k as f64 * self.row_size - self.off }
}

pub struct Scene {
    pub prims: Vec<Prim>,
    /// (이름, 시작, 끝) -- prims 안 요소 묶음(좌표 대조용)
    pub groups: Vec<(&'static str, usize, usize)>,
    pub g: Geo,
    /// 보이는 봉 · 봉별 화면 행(행 → [매수, 매도]) · 봉별 POC 행 -- 호버가 읽는다
    pub candles: Vec<Candle>,
    pub bar_rows: Vec<BTreeMap<i64, [f64; 2]>>,
    pub pocs: Vec<Option<i64>>,
    pub lane: Vec<Option<crate::lane::Row>>,
    pub dt: Option<crate::profile::Dt>,
}

impl Scene {
    fn add(&mut self, name: &'static str, mut v: Vec<Prim>) {
        let s = self.prims.len();
        self.prims.append(&mut v);
        self.groups.push((name, s, self.prims.len()));
    }
}

/// app.js 2114-2130 niceStep + axisTicks (더하기를 되풀이하는 것까지 같게 -- 부동소수 끝자리)
fn axis_ticks(min: f64, max: f64, target: f64) -> Vec<f64> {
    let rough = (max - min) / target.max(1.0);
    let mag = 10f64.powf((if rough == 0.0 { 1.0 } else { rough }).log10().floor());
    let norm = rough / mag;
    let step = mag * if norm <= 1.0 { 1.0 } else if norm <= 2.0 { 2.0 } else if norm <= 5.0 { 5.0 } else { 10.0 };
    let (start, end) = ((min / step).floor() * step, (max / step).ceil() * step);
    let mut out = vec![];
    let mut v = start;
    while v <= end + step * 0.5 { out.push(v); v += step; }
    out
}

/// 1970-01-01 부터 날 수 → (월, 일) (fmtDateTick 앞 «M/D » -- 하루 넘는 창)
fn month_day(days: i64) -> (i64, i64) {
    let z = days + 719_468;
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    (if mp < 10 { mp + 3 } else { mp - 9 }, doy - (153 * mp + 2) / 5 + 1)
}

/// measure(글자, 크기px) = 글자 폭(px). app.js 는 캔버스 measureText(textW)·svg getComputedTextLength 로 잰다.
pub fn prep(fp: &Footprint, full: &[Candle], cfg: &Cfg, measure: &dyn Fn(&str, f64) -> f64) -> Option<Scene> {
    // ── 창 자르기: renderSnapshotChart 8264-8268 (실시간: edge = len + chartPanRight(0)) ──
    let len = full.len();
    let n = cfg.window.min(len as f64);
    let edge = len as f64;
    let s0 = (edge - n + 1e-6).floor().max(0.0) as usize;
    let candles = &full[s0..len.min((edge - 1e-6).ceil() as usize)];
    let (nn, x0b) = (n, s0 as f64 - (edge - n)); // view.n · view.off
    if candles.is_empty() { return None; }
    // footprintForChart 6723-6744: 칸이 있는 봉만, 시각 → 칸
    let by_time: BTreeMap<i64, &Vec<Vec<f64>>> = fp.bars.iter().filter(|b| !b.levels.is_empty()).map(|b| (b.time, &b.levels)).collect();
    if by_time.is_empty() { return None; }
    // ponytail: 현재가 = 마지막 봉 종가(오프라인 하네스에서 지금 화면도 같다) — 실시간은 rtcore 밀기(체결가)를 받을 때 넣는다
    let current = candles[candles.len() - 1].close;

    // ── 기하: 8420-8603 데스크톱 넓은 화면 2단 ──
    let (w_all, h) = (cfg.w_all, cfg.h_all);
    let sub_w = ((w_all - 20.0) * cfg.split).round() + 35.0; // 8429 SPLIT_GAP 20 · FP_LEFT_SHIFT 35 (6563)
    let w = w_all - 20.0 - sub_w; // 8430
    let sub_total = 38.0 + 8.0; // 8451 SUB_LEGEND_H + SUB_GAP
    let ml = (36.0 - 26.0) + 26.0 - 35.0; // 8499 풋프린트 36 − CENTER_NUDGE + CENTER_NUDGE − FP_LEFT_SHIFT
    let mr = 34.0 - 26.0; // 8501
    let mt = 4.0 + sub_total + 6.0; // 8502 mtTop 4 + SUB_TOTAL + FP_TOP_ALIGN(6561)
    let mb = 150.0 + 2.0; // 8502 FP_BUB_BAND(6192) + 2
    let ps: f64 = if w >= 1000.0 { 0.75 } else { 0.76 }; // 8573
    let trade_w = (88.0 * ps).round(); // 8577
    let (liq_c, dt_w, dt_g) = ((72.0 * ps).round(), (64.0 * ps).round(), 6.0); // 8584
    let reg_w = liq_c + trade_w + 2.0 * dt_w + 0.0 + 4.0 * dt_g + 2.0; // 8585 (BOOK_W 0: 2단)
    let cw = w - ml - mr - reg_w; // 8586
    let quad_h = 108.0; // 8550
    let lane_in = quad_h + 24.0; // 8600
    let ch = h - mt - mb; // 8601 (LANE_IN 이면 레인은 플롯 안)
    let plot_bottom = mt + ch; // 8603
    let quad_y = plot_bottom - lane_in + 4.0; // 8616
    let lane_top = quad_y - 8.0; // 8768

    // ── 세로 범위: 8646-8689 ──
    let (mut min_p, mut max_p) = (current, current);
    for c in candles { min_p = min_p.min(c.low); max_p = max_p.max(c.high); }
    let q = fp.bucket;
    let (y_min, y_max) = {
        let pad = { let p = (max_p - min_p) * 0.15; if p == 0.0 { 1.0 } else { p } }; // CHART_Y_PAD_FOOTPRINT(163) · `|| 1`
        let (mut lo, mut hi) = (min_p - pad, max_p + pad);
        if ch > lane_in { lo -= (hi - lo) * lane_in / (ch - lane_in); } // 8675
        if q > 0.0 { lo = (lo / q).floor() * q; hi = (hi / q).ceil() * q; } // 8678
        (lo, hi)
    };
    let y_span = (y_max - y_min).max(1e-5);
    let slot = cw / nn; // 8700
    let x_at = |i: usize| ml + (i as f64 + x0b) * slot;
    let y_at = |v: f64| mt + (y_max - v) * ch / y_span;
    let bw = slot * 0.86; // 8704
    let dense = slot < 8.0; // 8709
    let mut bg: Vec<Prim> = vec![];
    let mut fg: Vec<Prim> = vec![];
    let mut svg: Vec<Prim> = vec![];

    // ── 격자: 8983-8984 (가격 눈금 글자는 2026-09-09 에 없앴다 -- 선만) ──
    for v in axis_ticks(y_min, y_max, 6.0) {
        let y = y_at(v);
        if y >= mt && y <= plot_bottom {
            bg.push(Prim::Rect { x: ml, y: y - 0.5, w: w - mr - ml, h: 1.0, c: Col::Line, a: 0.82, rx: 0.0, clip: Clip::No }); // cvHLine 8191
        }
    }
    // ── 현재가 선(칸 뒤 = 격자 바로 위): 8933-8937 · 10452-10462 ──
    let real_y = y_at(current); // 범위에 현재가가 들어 있어 늘 화면 안(8649)
    bg.push(Prim::Line { pts: vec![[ml, real_y], [w - mr, real_y]], c: Col::Accent, a: 0.85, lw: 1.0, dash: Some(4.0), clip: Clip::No });

    // ── 시간축: 8993-9034 (svg 층) ──
    {
        let last = candles.len() - 1;
        let dated = last > 0 && candles[last].time - candles[0].time >= 86_400;
        let min_tick_px = 52.0 * if dated { 1.7 } else { 1.0 };
        let bar_sec = if candles.len() > 1 { candles[1].time - candles[0].time } else { 300 };
        let tz = cfg.tz_off_min * 60;
        let tick_sec = [300, 600, 900, 1800, 3600, 7200, 10800, 14400, 21600, 43200, 86400, 172_800, 345_600].into_iter()
            .find(|&qq| qq % bar_sec == 0 && (qq / bar_sec) as f64 * slot >= cw / 7.0).unwrap_or(604_800);
        let mut idx: Vec<usize> = vec![];
        for (i, c) in candles.iter().enumerate() {
            let x = x_at(i) + bw / 2.0;
            if x >= ml && x <= ml + cw && (c.time - tz) % tick_sec == 0 { idx.push(i); }
        }
        if !idx.contains(&last) {
            match idx.last() {
                Some(&p) if x_at(last) - x_at(p) < min_tick_px => { *idx.last_mut().unwrap() = last; }
                _ => idx.push(last),
            }
        }
        for i in idx {
            let x = x_at(i) + bw / 2.0;
            svg.push(Prim::Line { pts: vec![[x, h - mb - 19.0], [x, h - mb - 16.0]], c: Col::Line, a: 1.0, lw: 1.0, dash: None, clip: Clip::No });
            let t = candles[i].time - tz;
            let mut s = format!("{:02}:{:02}", t.rem_euclid(86_400) / 3600, t.rem_euclid(3600) / 60);
            if dated { let (m, d) = month_day(t.div_euclid(86_400)); s = format!("{m}/{d} {s}"); }
            let tw = measure(&s, 13.0);
            svg.push(Prim::Text { x: x.max(2.0 + tw / 2.0), y: h - mb - 3.0, s, size: 13.0, c: Col::Muted, a: 1.0, anchor: Anchor::Middle, halo: false, clip: Clip::No });
        }
    }

    // ── 풋프린트 봉: 9040-9587 ──
    let atr = {
        let trs: Vec<f64> = candles.iter().enumerate().map(|(i, c)| if i == 0 { c.high - c.low } else {
            (c.high - c.low).max((c.high - candles[i - 1].close).abs()).max((c.low - candles[i - 1].close).abs()) }).collect();
        trs.iter().sum::<f64>() / (trs.len().max(1) as f64)
    };
    let min_row = ((y_span * 11.0 / ch) / q).ceil() * q; // 9048 FOOTPRINT_MIN_ROW_PX 11 (149)
    let row_size = q.max(min_row).max((0.2 * atr / q).round() * q); // 9049
    let row_px = row_size * ch / y_span;
    let off = q / 2.0; // 9056 행 경계 = 버킷 가장자리
    let row_of = |p: f64| ((p + off) / row_size).floor() as i64;
    let row_lo = |k: i64| k as f64 * row_size - off;
    // 9376-9386 서버 버킷을 화면 행으로 묶기. ponytail: BTreeMap = 오름차순 -- 서버 칸이 가격 오름차순이라 JS Map 넣은 순서와 같다
    let bar_rows: Vec<BTreeMap<i64, [f64; 2]>> = candles.iter().map(|c| {
        let mut rows = BTreeMap::new();
        if let Some(lv) = by_time.get(&c.time) {
            for l in lv.iter() {
                let cell = rows.entry(row_of(l[0])).or_insert([0.0, 0.0]);
                cell[0] += l[1]; cell[1] += l[2];
            }
        }
        rows
    }).collect();
    let (mut max_buy, mut max_sell) = (0.0f64, 0.0f64); // 9390
    for rows in &bar_rows { for c in rows.values() { max_buy = max_buy.max(c[0]); max_sell = max_sell.max(c[1]); } }
    const SHADES: [f64; 4] = [0.10, 0.22, 0.36, 0.54]; // 164 FOOTPRINT_SHADE_DARK
    let shade = |v: f64, max: f64| SHADES[(((if max > 0.0 { v / max } else { 0.0 }) * 4.0).floor() as usize).min(3)];
    let font_px = 9f64.min((row_px - 3.0).max(6.0)); // 9397
    let fpc_wc = 3f64.max(8f64.min(slot * 0.14)); // 9403
    let fp_thin = cfg.window > 12.0 + 1e-6 && 2f64.max((bw - fpc_wc - 2.0) / 2.0 - 0.5) < 5.0;
    let fpc = bw >= 14.0 || fp_thin; // 9405 "C"
    let mob = fp_thin; // 9410 (데스크톱)
    let fpc_w = if fpc { if fp_thin { bw } else { fpc_wc } } else { 0.0 }; // 9412
    let fpc_gap = if fpc { 2.0 } else { 0.0 };
    let half = 2f64.max((bw - fpc_w - fpc_gap) / 2.0 - 0.5); // 9413
    let show_qty = half >= 18.0; // 9416
    let draw_cells = bw >= 14.0 && !mob; // 9424
    let mx_cell = max_sell.max(max_buy); // 9468
    let draw_cell = |out: &mut Vec<Prim>, cx: f64, y_top: f64, v: f64, other: f64, col: Col, edge_x: f64| { // 9437-9442
        if half > 0.0 && row_px > 0.0 { out.push(Prim::Rect { x: cx, y: y_top, w: half, h: row_px, c: col, a: shade(v, mx_cell), rx: 0.0, clip: Clip::Plot }); }
        if v > 0.0 && row_px >= 7.0 && show_qty {
            out.push(Prim::Text { x: cx + half / 2.0, y: y_top + row_px / 2.0 + font_px * 0.36, s: fmt_qty(v), size: font_px, c: Col::Ink, a: 0.82, anchor: Anchor::Middle, halo: false, clip: Clip::Plot });
        }
        if v > 0.0 && v > other * 3.0 { // FOOTPRINT_IMBALANCE_RATIO(150)
            out.push(Prim::Rect { x: edge_x, y: y_top + 0.5, w: 2.0, h: (row_px - 1.0).max(1.0), c: col, a: 1.0, rx: 0.0, clip: Clip::Plot });
        }
    };
    struct Info { poc: Option<i64>, buy: f64, sell: f64, low: Option<i64> }
    let info = |rows: &BTreeMap<i64, [f64; 2]>| { // 9443 barInfo -- 같은 최대면 먼저 나온(낮은) 행
        let (mut poc, mut pv, mut buy, mut sell) = (None, 0.0, 0.0, 0.0);
        for (&k, c) in rows { buy += c[0]; sell += c[1]; if c[0] + c[1] > pv { pv = c[0] + c[1]; poc = Some(k); } }
        Info { poc, buy, sell, low: rows.keys().next().copied() }
    };
    let delta_font = if bw >= 34.0 { 11.0 } else { 9.0 }; // 9490
    let mut delta_boxes: Vec<(f64, f64, f64)> = vec![]; // x1, x2, y
    let mut deltas: Vec<Prim> = vec![];
    let mut bumped = 0usize;
    let mut poc_pts: Vec<[f64; 2]> = vec![];
    let mut pocs: Vec<Option<i64>> = vec![];
    for (i, rows) in bar_rows.iter().enumerate() {
        let (c, x, inf) = (&candles[i], x_at(i), info(rows));
        pocs.push(inf.poc);
        if let Some(k) = inf.poc { // 9531 pocPt
            let py = y_at(row_lo(k + 1));
            if !(py + row_px < mt || py > mt + ch) { poc_pts.push([x + bw / 2.0, py + row_px / 2.0]); }
        }
        // 9461-9484 봉 하나(칸 → 옆 캔들). ponytail: DENSE(봉 칸 8px 밑 = 145봉 넘는 창)·1px 밑 합치기는 안 옮겼다 — 휠로 넓히는 동작을 옮길 때
        if draw_cells {
            for (&key, cell) in rows {
                let y_top = y_at(row_lo(key + 1));
                if y_top + row_px < mt || y_top > mt + ch { continue; }
                let sx = x;
                let bx = sx + half + 1.0 + fpc_w + fpc_gap;
                let col = if cell[1] > cell[0] { Col::Bad } else if cell[0] > cell[1] { Col::Good } else { Col::Muted };
                draw_cell(&mut fg, sx, y_top, cell[1], cell[0], col, sx - 2.0);
                draw_cell(&mut fg, bx, y_top, cell[0], cell[1], col, bx + half + 2.0);
                if Some(key) == inf.poc { fg.push(Prim::StrokeRect { x: sx, y: y_top, w: bx + half - sx, h: row_px, c: Col::Amber, a: 0.85, lw: 1.0, clip: Clip::Plot }); }
            }
        }
        let color = if c.close >= c.open { Col::Good } else { Col::Bad };
        let (y_t, y_b) = (y_at(c.open.max(c.close)), y_at(c.open.min(c.close)));
        if fpc {
            let cx0 = if mob { x + (bw - fpc_w) / 2.0 } else { x + half + 1.0 + fpc_gap / 2.0 };
            let cxm = cx0 + fpc_w / 2.0;
            let (a, b) = (y_at(c.high), y_at(c.low));
            fg.push(Prim::Rect { x: cxm - 0.5, y: a.min(b), w: 1.0, h: (b - a).abs(), c: color, a: 1.0, rx: 0.0, clip: Clip::Plot }); // cvVLine 8190
            fg.push(Prim::Rect { x: cx0, y: y_t, w: fpc_w, h: (y_b - y_t).max(1.0), c: color, a: 1.0, rx: if fpc_w >= 5.0 { 1.0 } else { 0.0 }, clip: Clip::Plot });
        } else {
            let (a, b) = (y_at(c.high), y_at(c.low));
            fg.push(Prim::Rect { x: x + bw / 2.0 - 0.5, y: a.min(b), w: 1.0, h: (b - a).abs(), c: color, a: 0.5, rx: 0.0, clip: Clip::Plot });
            fg.push(Prim::StrokeRect { x, y: y_t, w: bw, h: (y_b - y_t).max(1.0), c: color, a: 0.6, lw: 1.0, clip: Clip::Plot });
        }
        // 9496-9528 델타 글자 자리(겹치면 한 줄씩 내림 -- 사슬)
        if inf.buy + inf.sell > 0.0 && !dense {
            let delta = inf.buy - inf.sell;
            let cells_bottom = inf.low.map_or(y_at(c.low), |k| y_at(row_lo(k)));
            let mut dy = (plot_bottom - 3.0).min(cells_bottom.max(y_at(c.low)) + 13.0);
            let label = format!("{}{}", if delta >= 0.0 { "+" } else { "−" }, fmt_qty(delta.abs()));
            let half_w = label.encode_utf16().count() as f64 * delta_font * 0.34 + 2.0; // JS .length = UTF-16 단위
            let (cxd, dy0) = (x + bw / 2.0, dy);
            for _ in 0..6 {
                let mut hit = false;
                for b in delta_boxes.iter().rev() {
                    if hit || b.1 + 26.0 < cxd - half_w { break; }
                    hit = (b.2 - dy).abs() < delta_font + 1.0 && cxd + half_w > b.0 && cxd - half_w < b.1;
                }
                if !hit || dy + delta_font + 2.0 > plot_bottom - 3.0 { break; }
                dy += delta_font + 2.0;
            }
            delta_boxes.push((cxd - half_w, cxd + half_w, dy));
            if dy != dy0 { bumped += 1; }
            deltas.push(Prim::Text { x: cxd, y: dy, s: label, size: delta_font, c: if delta >= 0.0 { Col::Good } else { Col::Bad }, a: 1.0, anchor: Anchor::Middle, halo: false, clip: Clip::Plot });
        }
    }
    // 9567 전부 아니면 전무(밀려난 글자가 30% 넘으면 안 그린다)
    if !deltas.is_empty() && bumped as f64 / deltas.len() as f64 <= 0.3 { fg.append(&mut deltas); }
    // 9578-9587 봉별 POC 선(한 봉 넘게 벌어지면 끊는다) · lineSt("poc") = 1.5 · 실선 · amber · 불투명 0.75 · 둥근 이음
    let mut seg: Vec<[f64; 2]> = vec![];
    let poc_line = |pts: Vec<[f64; 2]>| Prim::Line { pts, c: Col::Amber, a: 0.75, lw: 1.5, dash: None, clip: Clip::Plot };
    for (i, p) in poc_pts.iter().enumerate() {
        if i > 0 && p[0] - poc_pts[i - 1][0] > bw * 1.8 {
            if seg.len() > 1 { fg.push(poc_line(std::mem::take(&mut seg))); }
            seg.clear();
        }
        seg.push(*p);
    }
    if seg.len() > 1 { fg.push(poc_line(seg)); }

    let g = Geo { w, h, ml, mr, mt, mb, cw, ch, plot_bottom, quad_y, lane_top, y_min, y_max, y_span, slot, x0b, bw, row_size, off, row_px, current, draw_cells,
                  trade_x0: ml + cw, trade_w, dt_w, c_tpo: w - mr + 6.0 - dt_g - dt_w, c_del: w - mr + 6.0 - 2.0 * (dt_g + dt_w),
                  t_anc: w - mr + 6.0 - 3.0 * dt_g - 2.0 * dt_w, x_rr: w - mr - 2.0 }; // 8587-8588 (cBook = w − mr − BOOK_W(0) + 6) · 10354

    // ── 현재가 알약: 10476-10528 (데스크톱 풋프린트 = 프로파일 오른쪽 끝에 오른쪽 정렬 · 채운 알약 · 14px 800 · 고정폭 숫자) ──
    let mut pill = vec![];
    {
        let label_y = (mt + 9.0).max((plot_bottom - 9.0).min(real_y.max(mt + 9.0).min(plot_bottom - 9.0))); // declutterTagY 10351 + 10408
        let s = tnum(&to_fixed(current, 1)); // fmtNum(v, pxDp()) -- ETH dp 1 (45) · 10499 font-variant-numeric: tabular-nums
        let tw = measure(&s, 14.0);
        let bw0 = ((tw + 18.0) / 4.0).ceil() * 4.0; // pillW 4418
        pill.push(Prim::Rect { x: g.x_rr - bw0, y: label_y - 9.0, w: bw0, h: 18.0, c: Col::Ink, a: 1.0, rx: 9.0, clip: Clip::No });
        pill.push(Prim::Text { x: g.x_rr - 9.0, y: label_y + 4.5, s, size: 14.0, c: Col::OnFill, a: 1.0, anchor: Anchor::End, halo: false, clip: Clip::No });
    }

    // 층 순서(app.js 그대로): bg 캔버스 → fg 캔버스(칸·캔들·델타·POC 선 → 스택 9769 → 레인 11158 → 미완 경매 11332) → svg(시간축 → 프로파일 9299 → 레인 글자 → 알약 post)
    let candles = candles.to_vec();
    let lane_rows = crate::lane::rows(fp, &candles);
    let dt = crate::profile::build(&g, &candles, &by_time);
    let (lane_fg, lane_svg) = crate::lane::draw(&g, &lane_rows, measure);
    let mut sc = Scene { prims: vec![], groups: vec![], g, candles: vec![], bar_rows: vec![], pocs, lane: vec![], dt: None };
    sc.add("bg", bg);
    sc.add("fg", fg);
    sc.add("stack", if sc.g.draw_cells { crate::marks::stack(&sc.g, &bar_rows) } else { vec![] });
    sc.add("lane", lane_fg);
    sc.add("ua", crate::marks::unfinished(&sc.g, &candles, &by_time, q));
    sc.add("svg", svg);
    sc.add("profile", dt.as_ref().map_or(vec![], |d| crate::profile::draw(&sc.g, d)));
    sc.add("lane_text", lane_svg);
    sc.add("pill", pill);
    (sc.candles, sc.bar_rows, sc.lane, sc.dt) = (candles, bar_rows, lane_rows, dt);
    Some(sc)
}

/// font-variant-numeric: tabular-nums -- Pretendard 는 고정폭 숫자(tnum)를 U+E071~E07A 로도 싣는다(cmap 확인) · egui 는 오픈타입 기능을 못 켜서 글자를 바꾼다
pub fn tnum(s: &str) -> String {
    s.chars().map(|c| match c.to_digit(10) { Some(d) => char::from_u32(0xE071 + d).unwrap(), None => c }).collect()
}

#[cfg(test)]
mod t {
    use super::*;
    #[test]
    fn ticks_and_dates() {
        assert_eq!(axis_ticks(0.0, 10.0, 4.0), vec![0.0, 5.0, 10.0]);
        assert_eq!(month_day(0), (1, 1));
        assert_eq!(month_day(1_795_156_500 / 86_400), (11, 20)); // 2026-11-20 (UTC)
    }
}
