//! 창 상태와 마우스 조작(휠 확대·가로 휠·끌기·관성·더블클릭·5m/30m/1h) + 30m·1h 묶기. 줄 번호 = dashboard/live/app.js(이 작업 시점).
//! 지금 화면은 이 상태를 전역 변수(chartWindowBars·chartTfK·chartPanEnd·chartPanFrac·chartPanRight·chartPanY·chartYPerPx)로 든다 -- 여기선 View 하나.
//! 옮기지 않은 것: 미니맵(renderChartPan -- 넓은 화면 차트 위 줄은 숨김 7350) · 방향키(미니맵 초점) · Shift 측정 박스 · 손가락(휴대폰 분기).
use crate::data::Candle;
use std::collections::BTreeMap;

/// 118 FP_BARS(데스크톱) · 94 CHART_PAN_BARS · 115 CHART_WINDOW_MAX
pub const FP_BARS: f64 = 36.0;
pub const CHART_PAN_BARS: usize = 144;
pub fn window_max(k: i64) -> f64 { match k { 6 => 336.0, 12 => 168.0, _ => 2016.0 } }

pub struct Drag { x: f64, y: f64, lx: f64, py: f64, k: f64, slot: f64, moved: bool, v: Vec<(f64, f64)> }

/// 창 상태 -- 이름은 app.js 전역 그대로
pub struct View {
    pub bars: f64,           // chartWindowBars (소수)
    pub tf_k: i64,           // chartTfK 1·6·12
    pub pan_end: Option<i64>, // chartPanEnd (오른쪽 끝 봉 시각 · None = 실시간)
    pub pan_frac: f64,
    pub pan_right: f64,
    pub pan_y: f64,
    pub y_per_px: f64,       // chartYPerPx (마지막 그림이 채운다)
    pub fling: Option<(f64, f64)>, // (봉/ms, 지난 프레임 ms) -- chartFling
    pub drag: Option<Drag>,
}

impl Default for View {
    fn default() -> Self { Self { bars: FP_BARS, tf_k: 1, pan_end: None, pan_frac: 0.0, pan_right: 0.0, pan_y: 0.0, y_per_px: 0.0, fling: None, drag: None } }
}

/// 126 tfT: 5분봉 시각 → 그 봉이 든 tf 봉 시각
pub fn tft(k: i64, t: i64) -> i64 { if k > 1 { t - t % (300 * k) } else { t } }

impl View {
    pub fn want_week(&self) -> bool { self.tf_k > 1 || self.bars > FP_BARS || self.pan_end.is_some() } // 6639 fpWantWeek
    pub fn pan_right_max(&self) -> f64 { self.bars * 0.8 } // 100
    pub fn panned(&self) -> bool { self.pan_end.is_some() } // 8648

    /// 7272 chartPanEdge
    pub fn edge(&self, full: &[Candle]) -> f64 {
        let len = full.len() as f64;
        match self.pan_end {
            None => len + self.pan_right,
            Some(end) => {
                let i = full.iter().position(|c| c.time > end).map_or(len, |i| i as f64);
                len.min(self.bars).max(len.min(i + self.pan_frac))
            }
        }
    }
    /// 7278 chartPanTo -- 반환 = 바뀌었나(바뀌면 지금 화면은 다시 그리고 1주일을 부른다: 부르는 쪽이 want_week 로)
    pub fn pan_to(&mut self, full: &[Candle], edge: f64) -> bool {
        if full.is_empty() { return false; }
        let len = full.len() as f64;
        let e = len.min(self.bars).max((len + self.pan_right_max()).min(edge));
        let i = e.floor();
        let next = if e >= len - 1e-3 { None } else { Some(full[i as usize - 1].time) };
        let frac = if next.is_none() { 0.0 } else { e - i };
        let right = if next.is_none() { (e - len).max(0.0) } else { 0.0 };
        if next == self.pan_end && (frac - self.pan_frac).abs() < 1e-4 && (right - self.pan_right).abs() < 1e-4 { return false; }
        (self.pan_end, self.pan_frac, self.pan_right) = (next, frac, right);
        true
    }
    /// 7330 chartZoomAt(bars, at) -- at = 플롯 안 커서 비율
    pub fn zoom_at(&mut self, full: &[Candle], bars: f64, at: f64) {
        let len = full.len() as f64;
        if len == 0.0 { return; }
        let hi_b = window_max(self.tf_k).min(len.max(12.0));
        let mut bars = bars.min(hi_b).max(12.0);
        if bars < 12.0 + 1e-3 { bars = 12.0 } else if bars > hi_b - 1e-3 { bars = hi_b }
        if bars == self.bars { return; }
        let a = if self.pan_end.is_none() { 1.0 } else { at };
        let n0 = self.bars.min(len);
        let e0 = self.edge(full);
        let u = e0 - n0 + a * n0;
        self.bars = bars;
        let n1 = bars.min(len);
        self.pan_to(full, u - a * n1 + n1);
    }
    /// 7298 fpFitQty -- cw = 마지막 그림의 플롯 폭(chartGeo.cw)
    pub fn fit_qty(&mut self, cw: f64) {
        let half_at = |slot: f64| 2f64.max((slot * 0.86 - 3f64.max(8f64.min(slot * 0.14)) - 2.0) / 2.0 - 0.5); // 7297 fpHalfAt
        let mut n = (cw / 12.0).floor();
        while n > 12.0 && half_at(cw / n) < 18.0 { n -= 1.0; }
        self.fling = None; self.pan_end = None; self.pan_frac = 0.0; self.pan_y = 0.0;
        self.bars = n; self.pan_right = 2f64.min(self.pan_right_max());
    }
    /// 7304 chartPanReset
    pub fn reset(&mut self) { self.fling = None; self.pan_end = None; self.pan_frac = 0.0; self.pan_right = 0.0; self.pan_y = 0.0; }
    /// 1964 setChartWindow + 1985-1991 버튼(5m·30m·1h = FP_BARS · tfk 1·6·12): 반환 = 바뀌었나(바뀌면 풋프린트를 바로 묻는다)
    pub fn tf_button(&mut self, k: i64) -> bool {
        self.fling = None;
        let changed = !(self.bars == FP_BARS && self.tf_k == k);
        if changed { self.bars = FP_BARS; self.tf_k = k; }
        if self.pan_end.is_some() || self.pan_y != 0.0 || self.pan_right != 0.0 { self.reset(); }
        changed
    }

    /// 7935-7946 휠(세로 = 확대 · 가로 = 이동). dx·dy = 픽셀(줄·쪽 단위는 부르는 쪽이 40·800 곱) · px = 봉 한 칸 화면 px · at = 커서 비율
    pub fn wheel(&mut self, full: &[Candle], dx: f64, dy: f64, px: f64, at: f64) {
        self.fling = None;
        if dx.abs() > dy.abs() { if !full.is_empty() { let e = self.edge(full); self.pan_to(full, e + dx / px); } return; }
        self.zoom_at(full, self.bars * (dy.clamp(-400.0, 400.0) * 0.0025).exp(), at);
    }
    /// 7962-7970 누르기(가격 칸 안에서만 -- 부르는 쪽이 zoneY·zoneX 를 본다)
    pub fn press(&mut self, x: f64, y: f64, px: f64) {
        self.fling = None;
        self.drag = Some(Drag { x, y, lx: x, py: self.pan_y, k: self.y_per_px, slot: px, moved: false, v: vec![] });
    }
    /// 7993-8000 움직임(왼쪽 단추 눌린 채) -- 반환 = 그릴 것이 바뀌었나
    pub fn drag_to(&mut self, full: &[Candle], cx: f64, cy: f64, ts: f64, px: f64) -> bool {
        let Some(mut d) = self.drag.take() else { return false };
        if !d.moved && (cx - d.x).hypot(cy - d.y) < 5.0 { self.drag = Some(d); return false; }
        d.moved = true;
        // 7972-7981 dragTo
        if d.k != 0.0 { self.pan_y = d.py + (cy - d.y) * d.k; }
        d.slot = px;
        if !full.is_empty() { let e = self.edge(full); self.pan_to(full, e - (cx - d.lx) / d.slot); }
        d.lx = cx;
        d.v.push((ts, cx)); if d.v.len() > 6 { d.v.remove(0); }
        self.drag = Some(d);
        true
    }
    /// 8001-8006 놓기: 직전 ~100ms 속도로 관성(7982 flingFrom)
    pub fn release(&mut self, full: &[Candle], ts: f64) {
        let Some(d) = self.drag.take() else { return };
        if !d.moved || full.is_empty() { return; }
        let v: Vec<_> = d.v.iter().filter(|q| ts - q.0 < 100.0).collect();
        if v.len() < 2 || ts - v[v.len() - 1].0 > 40.0 { return; }
        let px_per_ms = (v[v.len() - 1].1 - v[0].1) / 8f64.max(v[v.len() - 1].0 - v[0].0);
        if px_per_ms.abs() > 0.25 { self.fling = Some((-px_per_ms / d.slot, ts)); }
    }
    /// 7316 chartFlingStart 의 한 프레임(t = 지금 ms) -- 반환 = 계속 도나
    pub fn fling_step(&mut self, full: &[Candle], t: f64) -> bool {
        let Some((mut v, t0)) = self.fling else { return false };
        let dt = 50f64.min(t - t0);
        v *= (-dt / 140.0).exp();
        let e = self.edge(full);
        if v.abs() * dt < 0.002 || !self.pan_to(full, e + v * dt) { self.fling = None; return false; }
        self.fling = Some((v, t));
        true
    }
}

/// 6755 tfCandles: 시가 = 첫 봉 · 고가 = 최대 · 저가 = 최소 · 거래량 = 합 · 나머지 = 마지막 봉(키가 없는 칸은 앞 값 -- Object.assign)
/// · 맨 앞 tf 봉이 덜 찼으면 뺀다. ponytail: 기억(tfCMemo)은 안 옮겼다 -- 같은 값을 매번 다시 묶는다(1주일 2천 봉 ≈ 수십 µs)
pub fn tf_candles(full: &[Candle], k: i64) -> Vec<Candle> {
    if k == 1 { return full.to_vec(); }
    let mut out: Vec<Candle> = vec![];
    let mut n0 = 0;
    for c in full {
        let t = tft(k, c.time);
        match out.last_mut() {
            Some(a) if a.time == t => {
                let (open, high, low, vol, partial) = (a.open, a.high.max(c.high), a.low.min(c.low), a.volume + c.volume, c.partial.or(a.partial));
                *a = Candle { time: t, open, high, low, volume: vol, partial, ..*c };
            }
            _ => out.push(Candle { time: t, ..*c }),
        }
        if out.len() == 1 { n0 += 1; }
    }
    if out.len() > 1 && n0 < k { out.remove(0); }
    out
}

/// 6791 tfLevels: 같은 가격 칸의 7열을 5분봉끼리 더한다 · 5분봉이 모자란 tf 봉은 뺀다(마지막 tf 봉만 예외) · 가격 오름차순.
pub fn tf_levels(by_time: &BTreeMap<i64, &Vec<Vec<f64>>>, bucket: f64, k: i64) -> BTreeMap<i64, Vec<Vec<f64>>> {
    let mut groups: BTreeMap<i64, Vec<&Vec<Vec<f64>>>> = BTreeMap::new();
    for (&t, lv) in by_time { groups.entry(tft(k, t)).or_default().push(lv); } // BTreeMap = 시각 오름차순(6794 sort)
    let last_t = groups.keys().next_back().copied();
    let mut out = BTreeMap::new();
    for (t, refs) in groups {
        if (refs.len() as i64) < k && Some(t) != last_t { continue; }
        let mut keys: Vec<i64> = vec![];
        let mut sum: std::collections::HashMap<i64, Vec<f64>> = Default::default();
        for lv in refs { for l in lv.iter() {
            let key = js_round(l[0] / bucket);
            let a = sum.entry(key).or_insert_with(|| { keys.push(key); vec![l[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] });
            for j in 1..7 { a[j] += crate::data::lv(l, j); }
        } }
        let mut lv: Vec<Vec<f64>> = keys.into_iter().map(|k| sum.remove(&k).unwrap()).collect();
        lv.sort_by(|a, b| a[0].total_cmp(&b[0])); // 안정 정렬(JS sort 도 안정)
        out.insert(t, lv);
    }
    out
}

/// JS Math.round(반은 +∞ 쪽)
pub fn js_round(x: f64) -> i64 { (x + 0.5).floor() as i64 }

#[cfg(test)]
mod t {
    use super::*;
    fn c(t: i64, o: f64, h: f64, l: f64, cl: f64) -> Candle { Candle { time: t, open: o, high: h, low: l, close: cl, partial: None, volume: 1.0, sma: None } }
    #[test]
    fn pan_zoom() {
        let full: Vec<Candle> = (0..144).map(|i| c(i * 300, 1.0, 2.0, 0.5, 1.5)).collect();
        let mut v = View::default();
        assert_eq!(v.edge(&full), 144.0);
        assert!(v.pan_to(&full, 100.5)); // 과거로 = 끝 봉 시각 + 소수
        assert_eq!((v.pan_end, v.pan_frac), (Some(99 * 300), 0.5));
        assert_eq!(v.edge(&full), 100.5);
        assert!(v.pan_to(&full, 1e9)); // 오른쪽 끝을 넘기면 실시간 + 빈 칸(최대 창의 80%)
        assert!(v.pan_end.is_none() && (v.pan_right - 36.0 * 0.8).abs() < 1e-9);
        v.reset();
        v.zoom_at(&full, 1000.0, 0.3); // 받은 만큼까지만 넓힌다
        assert_eq!(v.bars, 144.0);
        v.zoom_at(&full, 5.0, 0.3);
        assert_eq!(v.bars, 12.0);
        v.pan_to(&full, -50.0); // 맨 왼쪽 = 창 하나만큼
        assert_eq!(v.edge(&full), 12.0);
        v.fit_qty(1012.0); // 칸 숫자(half ≥ 18)가 들어가는 가장 넓은 창 = 봉 칸 54.2px 이상
        assert_eq!((v.bars, v.pan_right, v.pan_end), (18.0, 2.0, None));
    }
    #[test]
    fn tf() {
        // 30m: 앞 덜 찬 묶음(2봉) 뺌 · 시가 첫 봉 · 고저 극값 · 종가 마지막
        let full: Vec<Candle> = (4..20).map(|i| c(i * 300, i as f64, i as f64 + 1.0, i as f64 - 1.0, i as f64 + 0.5)).collect();
        let o = tf_candles(&full, 6);
        assert_eq!(o.len(), 3);
        assert_eq!((o[0].time, o[0].open, o[0].high, o[0].low, o[0].close, o[0].volume), (1800, 6.0, 12.0, 5.0, 11.5, 6.0));
        assert_eq!(o[2].time, 5400); // 마지막(형성) 묶음은 덜 차도 남는다
        let a = vec![vec![10.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0]];
        let b = vec![vec![10.0, 3.0, 4.0, 1.0, 0.0, 0.0, 0.0], vec![9.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]];
        let m: BTreeMap<i64, &Vec<Vec<f64>>> = [(0, &a), (300, &b), (1800, &a)].into_iter().collect();
        let t = tf_levels(&m, 0.5, 6);
        assert!(!t.contains_key(&0)); // 0 묶음은 5분봉 2개뿐 -- 모름
        assert_eq!(t[&1800], a);      // 마지막 묶음은 예외
    }
}
