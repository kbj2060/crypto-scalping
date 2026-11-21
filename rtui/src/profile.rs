//! 오른쪽 프로파일 기둥 중 풋프린트 응답만으로 그리는 둘: 체결 · 델타 칸 + TPO 칸(가치 영역 70%) + 기둥 머리 글자. app.js 9299-9351 (svg 층).
//! 같은 줄 = 같은 가격 행(fp.rs 행 격자). 청산 기둥(9186)은 청산맵 응답이 필요해 범위 밖.
use crate::data::{Candle, lv};
use crate::fp::{Anchor, Clip, Col, Geo, Prim};
use std::collections::{BTreeMap, HashMap};

/// 행 키 → [델타(매수−매도), TPO(그 가격에 체결이 있던 봉 수), 체결 총량, 고래, 리테일] -- 넣은 순서(JS Map)를 지킨다: 같은 최대값이면 먼저 넣은 행
pub struct Dt {
    pub keys: Vec<i64>,
    pub rows: HashMap<i64, [f64; 5]>,
    pub n_bars: usize,
    pub t_k: i64, pub tot_k: Option<i64>, pub va_lo: i64, pub va_hi: i64, pub d_max: f64, pub t_max: f64, pub tot: f64,
}

/// 9300-9321
pub fn build(g: &Geo, candles: &[Candle], by_time: &BTreeMap<i64, &Vec<Vec<f64>>>) -> Option<Dt> {
    let (mut keys, mut rows) = (vec![], HashMap::<i64, [f64; 5]>::new());
    for c in candles {
        let mut seen = vec![];
        for l in by_time.get(&c.time).map_or(&[][..], |v| v.as_slice()) {
            let k = g.row_of(l[0]);
            let a = rows.entry(k).or_insert_with(|| { keys.push(k); [0.0; 5] });
            a[0] += lv(l, 1) - lv(l, 2);
            a[2] += lv(l, 1) + lv(l, 2);
            a[3] += lv(l, 3) + lv(l, 4);
            a[4] += lv(l, 5) + lv(l, 6);
            if !seen.contains(&k) { seen.push(k); a[1] += 1.0; }
        }
    }
    let (mut d_max, mut t_max, mut t_k, mut t_sum, mut k_lo, mut k_hi, mut tot, mut tot_k) = (0f64, 0.0, None, 0.0, i64::MAX, i64::MIN, 0.0, None);
    for &k in &keys {
        let a = rows[&k];
        d_max = d_max.max(a[0].abs());
        if a[2] > tot { tot = a[2]; tot_k = Some(k); }
        t_sum += a[1];
        if a[1] > t_max { t_max = a[1]; t_k = Some(k); }
        k_lo = k_lo.min(k); k_hi = k_hi.max(k);
    }
    let t_k = t_k?; // 9313 tMax > 0
    // 9314-9319 가치 영역 = TPO 최다 행에서 위·아래 중 큰 쪽을 붙여 가며 전체 TPO 의 70% 까지
    let (mut up, mut dn, mut acc) = (t_k, t_k, t_max);
    let tpo = |k: i64| rows.get(&k).map_or(0.0, |a| a[1]);
    while acc < 0.7 * t_sum && (up < k_hi || dn > k_lo) {
        let u = if up < k_hi { tpo(up + 1) } else { -1.0 };
        let d = if dn > k_lo { tpo(dn - 1) } else { -1.0 };
        if u >= d { up += 1; acc += u.max(0.0); } else { dn -= 1; acc += d.max(0.0); }
    }
    Some(Dt { keys, rows, n_bars: candles.len(), t_k, tot_k, va_lo: dn, va_hi: up, d_max, t_max, tot })
}

/// 체결·델타 칸 왼쪽 끝 · 가운데(0선) · TPO 왼쪽 끝 (9320)
pub fn cols(g: &Geo) -> (f64, f64, f64) {
    let d_x0 = g.t_anc - g.trade_w;
    let dw = g.c_del + g.dt_w - d_x0;
    (d_x0, d_x0 + dw / 2.0, g.c_tpo)
}

/// 9322-9348
pub fn draw(g: &Geo, dt: &Dt) -> Vec<Prim> {
    let n = Clip::No;
    let (d_x0, d_mid, t_x0) = cols(g);
    let dw = (d_mid - d_x0) * 2.0;
    let (p_top, p_bot) = (g.mt, g.plot_bottom);
    let rect = |x, y, w, h, c, a| Prim::Rect { x, y, w, h, c, a, rx: 0.0, clip: n };
    let edge = |x, y, w, h| Prim::StrokeRect { x, y, w, h, c: Col::Ink, a: 0.9, lw: 1.2, clip: n };
    let mut v = vec![];
    let va_t = p_top.max(g.y_at(g.row_lo(dt.va_hi + 1)));
    let va_b = p_bot.min(g.y_at(g.row_lo(dt.va_lo)));
    if va_b > va_t { v.push(rect(d_x0, va_t, t_x0 + g.dt_w - d_x0, va_b - va_t, Col::Accent, 0.07)); }
    for &k in &dt.keys {
        let a = dt.rows[&k];
        let y_top = p_top.max(g.y_at(g.row_lo(k + 1)));
        let y_bot = p_bot.min(g.y_at(g.row_lo(k)));
        if y_bot - y_top < 1.0 { continue; }
        let hh = (y_bot - y_top - 1.0).max(1.0);
        let ow = if dt.tot > 0.0 { (dw / 2.0 - 1.0) * a[2] / dt.tot } else { 0.0 };
        let dw_ = if dt.d_max > 0.0 { (dw / 2.0 - 1.0) * a[0].abs() / dt.d_max } else { 0.0 };
        let tw = g.dt_w * a[1] / dt.t_max;
        // 9335 왼쪽 = 체결(총량, 주황) · 최다 체결 행은 진하게 + 테두리
        let is_tot = Some(k) == dt.tot_k;
        if ow > 0.5 {
            v.push(rect(d_mid - 1.0 - ow, y_top + 0.5, ow, hh, Col::Amber, if is_tot { 0.7 } else { 0.5 }));
            if is_tot { v.push(edge(d_mid - 1.0 - ow, y_top + 0.5, ow, hh)); }
        }
        // 9336 오른쪽 = 델타(|매수−매도|, 초록 + / 빨강 −)
        if dw_ > 0.5 { v.push(rect(d_mid + 1.0, y_top + 0.5, dw_, hh, if a[0] >= 0.0 { Col::Good } else { Col::Bad }, 0.85)); }
        // 9337-9340 TPO(가치 영역 안 진하게) · 최다 행 테두리 · 한 번만 지난 가격 = 점
        let in_va = k >= dt.va_lo && k <= dt.va_hi;
        v.push(rect(t_x0, y_top + 0.5, tw, hh, Col::Accent, if in_va { 0.55 } else { 0.28 }));
        if k == dt.t_k { v.push(edge(t_x0, y_top + 0.5, tw, hh)); }
        if a[1] == 1.0 { v.push(Prim::Circle { x: t_x0 + tw + 4.0, y: (y_top + y_bot) / 2.0, r: 1.8, c: Col::Accent, a: 1.0, clip: n }); }
    }
    // 9342-9343 델타 0 선 · 9344-9348 기둥 머리(9px 700 · 차트 바탕색 테두리 3px)
    v.push(Prim::Line { pts: vec![[d_mid, p_top], [d_mid, p_bot]], c: Col::SoftLine, a: 1.0, lw: 1.0, dash: None, clip: n });
    for (x, s) in [(g.trade_x0 + 4.0, "청산"), (d_x0, "체결 · 델타"), (t_x0, "TPO")] {
        v.push(Prim::Text { x, y: p_top - 4.0, s: s.into(), size: 9.0, wt: 700, c: Col::Muted, a: 1.0, anchor: Anchor::Start, halo: true, clip: n });
    }
    v
}
