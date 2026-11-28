//! 그리기: 그리기 목록 → egui 모양. 색 = styles.css :root(다크), 글꼴 = Pretendard(지금 화면의 --font-sans 첫 글꼴).
use crate::fp::{Anchor, Clip, Col, Prim, Scene};
use eframe::egui::{self, Color32, FontData, FontDefinitions, FontFamily, FontId, Painter, Pos2, Rect, Shape, Stroke, StrokeKind, pos2, vec2};
use std::sync::Arc;

/// styles.css 18-70 다크 토큰(--amber = --warn)
pub fn rgba(c: Col, a: f64) -> Color32 {
    let (r, g, b, a0) = match c {
        Col::Good => (0x5d, 0xbf, 0x83, 1.0),
        Col::Bad => (0xef, 0x93, 0x87, 1.0),
        Col::Muted => (0xa6, 0xac, 0xc1, 1.0),
        Col::Amber => (0xe9, 0x9c, 0x57, 1.0),
        Col::Accent => (0xdf, 0xe4, 0xee, 1.0),
        Col::Ink => (0xf5, 0xf6, 0xfb, 1.0),
        Col::Line => (203, 209, 227, 0.18),
        Col::OnFill => (0x12, 0x16, 0x1d, 1.0),
        Col::ChartBg => (0x17, 0x1b, 0x23, 1.0),
        Col::Text => (0xee, 0xf0, 0xf6, 1.0),
        Col::Turnover => (0x5a, 0xa9, 0xf5, 1.0),
        Col::SoftLine => (203, 209, 227, 0.095),
        Col::HoverLine => (203, 209, 227, 0.38),
        Col::Spot => (0xb8, 0x9c, 0xff, 1.0),
        Col::Neutral => (0xcb, 0xd1, 0xe3, 1.0),
        Col::Lift => (0x8b, 0x94, 0x9e, 1.0), // var(--lift-solid, #8b949e) -- styles.css 에 토큰이 없어 대체값

    };
    Color32::from_rgba_unmultiplied(r, g, b, (255.0 * a0 * a).round().clamp(0.0, 255.0) as u8)
}

/// 글꼴(지금 화면이 실제로 쓰는 것 · fonts/README.md): 본문 = Pretendard(--font-sans 첫 글꼴 · 400·700·800 세 굵기) ·
/// 툴팁 = JetBrains Mono(--font-mono · 400·700 -- 한글·없는 글자는 같은 굵기 Pretendard 로 떨어짐, 브라우저 글꼴 대체와 같은 순서).
/// 굵기마다 이름 붙은 패밀리 하나(egui 는 한 패밀리에 굵기 축이 없다): "w400"·"w700"·"w800" · "m400"·"m700". 뒤에 egui 기본 글꼴(▮ 같은 없는 글자).
/// 파일 = 공식 배포본을 화면 글자만 남겨 추린 것(fonts/subset.sh · *.sub.ttf -- ttf 는 저장소에 안 넣는다, 받는 주소·해시는 README).
pub fn fonts() -> FontDefinitions {
    let mut f = FontDefinitions::default();
    let base = f.families[&FontFamily::Proportional].clone();
    for (k, b) in [
        ("p400", &include_bytes!("../fonts/Pretendard-Regular.sub.ttf")[..]),
        ("p700", &include_bytes!("../fonts/Pretendard-Bold.sub.ttf")[..]),
        ("p800", &include_bytes!("../fonts/Pretendard-ExtraBold.sub.ttf")[..]),
        ("j400", &include_bytes!("../fonts/JetBrainsMono-Regular.sub.ttf")[..]),
        ("j700", &include_bytes!("../fonts/JetBrainsMono-Bold.sub.ttf")[..]),
    ] { f.font_data.insert(k.into(), Arc::new(FontData::from_static(b))); }
    let fam = |v: &[&str]| v.iter().map(|s| s.to_string()).chain(base.iter().cloned()).collect::<Vec<_>>();
    for (n, v) in [("w400", fam(&["p400"])), ("w700", fam(&["p700"])), ("w800", fam(&["p800"])), ("m400", fam(&["j400", "p400"])), ("m700", fam(&["j700", "p700"]))] {
        f.families.insert(FontFamily::Name(n.into()), v);
    }
    for fam in [FontFamily::Proportional, FontFamily::Monospace] { f.families.get_mut(&fam).unwrap().insert(0, "p700".into()); } // 이름 없이 부르는 곳(없어야 한다)은 옛 Bold
    f
}
/// 본문 글꼴(--font-sans) 굵기 → 패밀리 · 450 미만 = 400 · 750 미만 = 700 · 그 위 = 800(카드가 쓰는 굵기는 셋뿐)
pub fn fid(size: f32, wt: u16) -> FontId { FontId::new(size, FontFamily::Name(if wt < 450 { "w400" } else if wt < 750 { "w700" } else { "w800" }.into())) }
/// 툴팁 글꼴(--font-mono) · 700 이상 = Bold
pub fn mono(size: f32, wt: u16) -> FontId { FontId::new(size, FontFamily::Name(if wt >= 600 { "m700" } else { "m400" }.into())) }

/// 캔버스·svg 처럼 반 픽셀 좌표를 그대로(egui 기본은 사각형·직각 선을 픽셀에 붙인다) · 글꼴 힌팅 끔(크로미움 리눅스 글자 폭에 더 가깝다 -- compare/coords_36.md)
pub fn canvas_like(ctx: &egui::Context) {
    ctx.tessellation_options_mut(|o| { o.round_rects_to_pixels = false; o.round_line_segments_to_pixels = false; });
    ctx.all_styles_mut(|s| s.visuals.text_options.font_hinting = false);
}

pub fn measure(p: &Painter, s: &str, size: f64, wt: u16) -> f64 {
    p.layout_no_wrap(s.to_owned(), fid(size as f32, wt), Color32::WHITE).size().x as f64
}

/// 캔버스 roundRect(반지름 = min(rx, 폭/2, 높이/2)) -- egui 모서리 반지름은 정수(u8)라 1.5 같은 값은 볼록 다각형으로 직접(모서리당 6조각)
fn round_rect(p: &Painter, r: Rect, rx: f32, c: Color32) {
    let rx = rx.min(r.width() / 2.0).min(r.height() / 2.0);
    if rx.fract() == 0.0 { p.rect_filled(r, rx, c); return; }
    let mut v = Vec::with_capacity(28);
    for (cx, cy, a0) in [(r.max.x - rx, r.min.y + rx, -90f32), (r.max.x - rx, r.max.y - rx, 0.0), (r.min.x + rx, r.max.y - rx, 90.0), (r.min.x + rx, r.min.y + rx, 180.0)] {
        for k in 0..=6 { let a = (a0 + 15.0 * k as f32).to_radians(); v.push(pos2(cx + rx * a.cos(), cy + rx * a.sin())); }
    }
    p.add(Shape::convex_polygon(v, c, Stroke::NONE));
}

pub fn text(p: &Painter, at: Pos2, s: &str, size: f32, wt: u16, c: Color32, anchor: Anchor, halo: bool) {
    let g = p.layout_no_wrap(s.to_owned(), fid(size, wt), c);
    let base = g.rows.first().and_then(|r| r.row.glyphs.first().map(|gl| r.pos.y + gl.pos.y)).unwrap_or(0.0);
    let dx = match anchor { Anchor::Start => 0.0, Anchor::Middle => g.size().x / 2.0, Anchor::End => g.size().x };
    let at = at - vec2(dx, base);
    if halo { // svg paint-order: stroke 3px(바깥 1.5px) 차트 바탕색 -- 여덟 방향으로 1.5px 밀어 깐다
        let hg = p.layout_no_wrap(s.to_owned(), fid(size, wt), rgba(Col::ChartBg, 1.0));
        for k in 0..8 { let a = (45.0 * k as f32).to_radians(); p.galley(at + vec2(1.5 * a.cos(), 1.5 * a.sin()), hg.clone(), Color32::WHITE); }
    }
    p.galley(at, g, Color32::WHITE);
}

/// 툴팁(#chartTooltip): styles.css 741-776 · 3694 (배경 --pop-bg #1b212b 불투명 · 안쪽 여백 8/12 · 테두리 1px 흰 .1 · 둥근 6 · 그림자 0 4 12 검정 .2)
/// 자리 = app.js 11692 showTooltip(커서 +8/+15 · 오른쪽 넘치면 왼쪽으로 · 아래 넘치면 위로). 글꼴: --font-mono(JetBrains Mono · 한글은 Pretendard -- mono).
/// wrap = 델타·TPO 풀이(div white-space:normal · max-width 360): 접히면 폭이 360 그대로.
pub fn tip(p: &Painter, t: &mut crate::hover::Tip, vw: f32, vh: f32) {
    use crate::hover::Tc;
    use egui::text::{LayoutJob, TextFormat};
    let col = |c: Tc| match c { Tc::White => Color32::WHITE, Tc::Dim => Color32::from_rgba_unmultiplied(255, 255, 255, 184), Tc::C(c, a) => rgba(c, a as f64) };
    let (mut gs, mut w, mut h, mut wrapped) = (vec![], 0f32, 0f32, false);
    for l in &t.lines {
        let mut job = LayoutJob::default();
        if let Some(m) = t.wrap { job.wrap.max_width = m; }
        // 줄 높이 = 크로미움의 line-height:normal = round(ascent) + round(descent)(JetBrains Mono hhea 1020/−300 · 줄 간격 0) -- 11px 14 · 12px 16.
        // egui 는 (ascent − descent) 를 한 번에 픽셀로 올려 15·16 이 돼 줄마다 1px 씩 늘었다(툴팁 10줄 +10px · 2026-11-21)
        let lh = (1.02 * l.size).round() + (0.30 * l.size).round();
        let fmt = |c, wt| TextFormat { font_id: mono(l.size, wt), color: c, line_height: Some(lh), ..Default::default() };
        if l.runs.is_empty() { job.append(" ", 0.0, fmt(Color32::TRANSPARENT, 400)); } // <br> 만 있는 줄 = 글자 한 줄 높이
        for (s, c, wt) in &l.runs { job.append(s, 0.0, fmt(col(*c), *wt)); }
        let g = p.layout_job(job);
        wrapped |= g.rows.len() > 1;
        w = w.max(g.size().x);
        let gh = g.size().y;
        gs.push((h, g));
        h += gh + l.mb;
    }
    if wrapped { w = t.wrap.unwrap_or(w); }
    let (bw, bh) = (w + 26.0, h + 18.0);
    let (mx, my) = (t.at[0] as f32, t.at[1] as f32);
    let mut left = mx + 8.0;
    if left + bw > vw { left = mx - bw - 8.0; }
    left = left.max(4.0);
    let top = if my + 15.0 + bh > vh - 4.0 { 4f32.max(my - bh - 10.0) } else { my + 15.0 };
    let r = Rect::from_min_size(pos2(left, top), vec2(bw, bh));
    p.add(egui::epaint::Shadow { offset: [0, 4], blur: 12, spread: 0, color: Color32::from_black_alpha(51) }.as_shape(r, 6));
    p.rect(r, 6.0, Color32::from_rgb(0x1b, 0x21, 0x2b), Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 26)), StrokeKind::Inside);
    for (y, g) in gs { p.galley(pos2(left + 13.0, top + 9.0 + y), g, Color32::WHITE); }
    (t.x, t.y, t.w, t.h) = (left as f64, top as f64, bw as f64, bh as f64);
}

/// origin = svg 상자 왼쪽 위(화면 px) · 층 순서대로 나눠 부른다(lib.rs Chart::ui -- 호버 층이 가운데 끼인다)
pub fn paint_prims(p: &Painter, prims: &[Prim], sc: &Scene, origin: Pos2) {
    let g = &sc.g;
    paint(p, prims, origin, [g.ml, g.mt, g.cw, g.ch], [g.ml, g.lane_top, g.cw, g.h - g.lane_top], 1.0);
}

/// plot·lane = 자르기 상자 [x, y, 폭, 높이] · dim = 불투명 곱(1초 수급: 새 봉 첫 초에 지난 그림을 .35 로)
pub fn paint(p: &Painter, prims: &[Prim], origin: Pos2, plot: [f64; 4], lane: [f64; 4], dim: f64) {
    let o = origin.to_vec2();
    let rgba = |c: Col, a: f64| rgba(c, a * dim);
    let pt = |x: f64, y: f64| pos2(x as f32, y as f32) + o;
    let cut = |r: [f64; 4]| p.with_clip_rect(Rect::from_min_size(pt(r[0], r[1]), vec2(r[2] as f32, r[3] as f32)).intersect(p.clip_rect()));
    let (plot, lane) = (cut(plot), cut(lane));
    for pr in prims {
        let clip = match pr { Prim::Rect { clip, .. } | Prim::StrokeRect { clip, .. } | Prim::Line { clip, .. } | Prim::Text { clip, .. } | Prim::Circle { clip, .. } | Prim::Dashed { clip, .. } => *clip };
        let pa = match clip { Clip::No => p, Clip::Plot => &plot, Clip::Lane => &lane };
        match pr {
            &Prim::Rect { x, y, w, h, c, a, rx, .. } => {
                if w > 0.0 && h > 0.0 { round_rect(pa, Rect::from_min_size(pt(x, y), vec2(w as f32, h as f32)), rx as f32, rgba(c, a)); }
            }
            &Prim::StrokeRect { x, y, w, h, c, a, lw, .. } => {
                pa.rect_stroke(Rect::from_min_size(pt(x, y), vec2(w as f32, h as f32)), 0.0, Stroke::new(lw as f32, rgba(c, a)), StrokeKind::Middle);
            }
            &Prim::Circle { x, y, r, c, a, .. } => { pa.circle_filled(pt(x, y), r as f32, rgba(c, a)); }
            Prim::Line { pts, c, a, lw, dash, .. } => {
                let v: Vec<Pos2> = pts.iter().map(|q| pt(q[0], q[1])).collect();
                let s = Stroke::new(*lw as f32, rgba(*c, *a));
                match dash {
                    // 가로·세로 점선 = 맞대기 끝 사각형(캔버스 setLineDash·svg dasharray 와 같은 덮임 -- egui 선분은 끝을 깃털만큼 늘려 대시가 4→5~6px 로 보였다)
                    // ponytail: 축에 붙은 점선만(현재가 선 · 호버 십자선) — 기운 점선이 생기면 그때 사각형을 돌린다
                    Some(d) if v.len() == 2 && (v[0].y == v[1].y || v[0].x == v[1].x) => {
                        let (d, half, vert) = (*d as f32, *lw as f32 / 2.0, v[0].x == v[1].x);
                        let (a0, a1) = if vert { (v[0].y.min(v[1].y), v[0].y.max(v[1].y)) } else { (v[0].x, v[1].x) };
                        let mut t = a0;
                        while t < a1 {
                            let (q0, q1) = (t, (t + d).min(a1));
                            let r = if vert { Rect::from_min_max(pos2(v[0].x - half, q0), pos2(v[0].x + half, q1)) } else { Rect::from_min_max(pos2(q0, v[0].y - half), pos2(q1, v[0].y + half)) };
                            pa.rect_filled(r, 0.0, s.color);
                            t += 2.0 * d;
                        }
                    }
                    Some(d) => pa.extend(Shape::dashed_line(&v, s, *d as f32, *d as f32)),
                    None => { pa.add(Shape::line(v, s)); }
                }
            }
            Prim::Dashed { pts, c, a, lw, on, off, .. } => {
                let v: Vec<Pos2> = pts.iter().map(|q| pt(q[0], q[1])).collect();
                pa.extend(Shape::dashed_line(&v, Stroke::new(*lw as f32, rgba(*c, *a)), *on as f32, *off as f32));
            }
            Prim::Text { x, y, s, size, wt, c, a, anchor, halo, .. } => text(pa, pt(*x, *y), s, *size as f32, *wt, rgba(*c, *a), *anchor, *halo),
        }
    }
}
