//! 그리기: 그리기 목록 → egui 모양. 색 = styles.css :root(다크), 글꼴 = Pretendard(지금 화면의 --font-sans 첫 글꼴).
use crate::fp::{Anchor, Col, Prim, Scene};
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
    };
    Color32::from_rgba_unmultiplied(r, g, b, (255.0 * a0 * a).round().clamp(0.0, 255.0) as u8)
}

/// 글꼴: Pretendard 를 맨 앞에(egui 기본 글꼴은 Pretendard 에 없는 글자만). 굵기는 한 벌(Bold) -- 지금 화면의 400·800 도 Bold 로 그린다.
/// fonts/Pretendard-Bold.ttf (SIL OFL, fonts/Pretendard-LICENSE.txt) -- 저장소에는 라이선스만, ttf 는 orioncactus/pretendard v1.3.9 배포본.
pub fn fonts() -> FontDefinitions {
    let mut f = FontDefinitions::default();
    f.font_data.insert("pretendard".into(), Arc::new(FontData::from_static(include_bytes!("../fonts/Pretendard-Bold.ttf"))));
    for fam in [FontFamily::Proportional, FontFamily::Monospace] { f.families.get_mut(&fam).unwrap().insert(0, "pretendard".into()); }
    f
}

/// 캔버스·svg 처럼 반 픽셀 좌표를 그대로(egui 기본은 사각형·직각 선을 픽셀에 붙인다) · 글꼴 힌팅 끔(크로미움 리눅스 글자 폭에 더 가깝다 -- compare/coords_36.md)
pub fn canvas_like(ctx: &egui::Context) {
    ctx.tessellation_options_mut(|o| { o.round_rects_to_pixels = false; o.round_line_segments_to_pixels = false; });
    ctx.all_styles_mut(|s| s.visuals.text_options.font_hinting = false);
}

pub fn measure(p: &Painter, s: &str, size: f64) -> f64 {
    p.layout_no_wrap(s.to_owned(), FontId::proportional(size as f32), Color32::WHITE).size().x as f64
}

/// origin = svg 상자 왼쪽 위(화면 px)
pub fn paint(p: &Painter, sc: &Scene, origin: Pos2) {
    let o = origin.to_vec2();
    let pt = |x: f64, y: f64| pos2(x as f32, y as f32) + o;
    let plot = p.with_clip_rect(Rect::from_min_size(pt(sc.ml, sc.mt), vec2(sc.cw as f32, sc.ch as f32)).intersect(p.clip_rect()));
    for pr in &sc.prims {
        let clip = matches!(pr, Prim::Rect { clip: true, .. } | Prim::StrokeRect { clip: true, .. } | Prim::Line { clip: true, .. } | Prim::Text { clip: true, .. });
        let pa = if clip { &plot } else { p };
        match pr {
            &Prim::Rect { x, y, w, h, c, a, rx, .. } => {
                if w > 0.0 && h > 0.0 { pa.rect_filled(Rect::from_min_size(pt(x, y), vec2(w as f32, h as f32)), rx as f32, rgba(c, a)); }
            }
            &Prim::StrokeRect { x, y, w, h, c, a, lw, .. } => {
                pa.rect_stroke(Rect::from_min_size(pt(x, y), vec2(w as f32, h as f32)), 0.0, Stroke::new(lw as f32, rgba(c, a)), StrokeKind::Middle);
            }
            Prim::Line { pts, c, a, lw, dash, .. } => {
                let v: Vec<Pos2> = pts.iter().map(|q| pt(q[0], q[1])).collect();
                let s = Stroke::new(*lw as f32, rgba(*c, *a));
                match dash {
                    // 가로 점선 = 맞대기 끝 사각형(캔버스 setLineDash 와 같은 덮임 -- egui 선분은 끝을 깃털만큼 늘려 대시가 4→5~6px 로 보였다)
                    // ponytail: 가로 점선만(지금 쓰는 것은 현재가 선 하나) — 기운 점선이 생기면 그때 사각형을 돌린다
                    Some(d) if v.len() == 2 && v[0].y == v[1].y => {
                        let (d, half) = (*d as f32, *lw as f32 / 2.0);
                        let mut x = v[0].x;
                        while x < v[1].x {
                            pa.rect_filled(Rect::from_min_max(pos2(x, v[0].y - half), pos2((x + d).min(v[1].x), v[0].y + half)), 0.0, s.color);
                            x += 2.0 * d;
                        }
                    }
                    Some(d) => pa.extend(Shape::dashed_line(&v, s, *d as f32, *d as f32)),
                    None => { pa.add(Shape::line(v, s)); }
                }
            }
            Prim::Text { x, y, s, size, c, a, anchor, .. } => {
                let g = pa.layout_no_wrap(s.clone(), FontId::proportional(*size as f32), rgba(*c, *a));
                let base = g.rows.first().and_then(|r| r.row.glyphs.first().map(|gl| r.pos.y + gl.pos.y)).unwrap_or(0.0);
                let dx = match anchor { Anchor::Start => 0.0, Anchor::Middle => g.size().x / 2.0, Anchor::End => g.size().x };
                pa.galley(pt(*x, *y) - vec2(dx, base), g, Color32::WHITE);
            }
        }
    }
}
