//! 측정용: 데스크톱(1920×1080) 풋프린트 차트 한 장을 egui 로 그리는 CPU 시간.
//! 그리는 양은 app.js renderCandleSvg 한 번이 실제로 내는 그리기 명령 수(오프라인 헤드리스 크로미움에서 센 값)에 맞춘다:
//! 봉·칸·숫자·델타·POC·캔들·축·현재가는 의미대로 그리고, 나머지(레인·프로파일·오른쪽 1초 수급·표식)는 같은 개수의
//! 사각형·둥근 사각형·원·선(점 수)·글자(글자 수)로 채운다. 형성 봉 칸 값은 매 프레임 바뀐다.
//! 글꼴: fonts/Pretendard-Bold.ttf (SIL OFL, fonts/Pretendard-LICENSE.txt) -- 저장소에는 라이선스만 커밋, ttf 는
//! orioncactus/pretendard v1.3.9 배포본에서 받아 둔다.
#[cfg(feature = "kfont")]
use eframe::egui::FontData;
use eframe::egui::{self, Align2, Color32, Context, FontDefinitions, FontFamily, FontId, Pos2, Rect, Shape, Stroke, StrokeKind, pos2, vec2};
#[cfg(feature = "kfont")]
use std::sync::Arc;

/// 시나리오 = app.js 실측 명령 수(1920×1080 · dpr 1 · ETH 실응답 크기). 0: 기본 36봉(칸 숫자 없음), 1: 12봉(칸 숫자 있음)
pub struct Target {
    pub bars: usize,
    pub rows: usize,     // 그린 칸 행 수(봉 전체 합)
    pub half: f32,       // 칸 폭
    pub row_px: f32,     // 행 높이
    pub qty: bool,       // 칸 숫자
    pub rects: usize,    // fillRect + svg <rect>
    pub rrects: usize,   // roundRect
    pub circles: usize,  // arc + svg <circle>
    pub lines: usize,    // stroke + svg <line>/<polyline>/<path>
    pub pts: usize,      // 위 선들의 점 수
    pub texts: usize,    // fillText + svg <text>
    pub chars: usize,    // 글자 수
}

pub const TARGETS: [Target; 2] = [
    Target { bars: 36, rows: 228, half: 8.6, row_px: 13.3, qty: false, rects: 739 + 177, rrects: 116, circles: 28 + 2,
             lines: 87 + 34 + 12 + 34, pts: 379 + 68 + 464 + 1013, texts: 36 + 61, chars: 162 + 1332 },
    Target { bars: 12, rows: 69, half: 30.8, row_px: 25.2, qty: true, rects: 200 + 94, rrects: 55, circles: 8 + 7,
             lines: 47 + 35 + 12 + 24, pts: 145 + 70 + 152 + 957, texts: 150 + 67, chars: 536 + 1303 },
];

#[derive(Default, Debug, Clone, Copy)]
pub struct Cnt { pub rects: usize, pub rrects: usize, pub circles: usize, pub lines: usize, pub pts: usize, pub texts: usize, pub chars: usize }

struct P<'a> { p: &'a egui::Painter, c: Cnt }
impl P<'_> {
    fn rect(&mut self, r: Rect, col: Color32) { self.p.rect_filled(r, 0.0, col); self.c.rects += 1; }
    fn rrect(&mut self, r: Rect, col: Color32) { self.p.rect_filled(r, 2.0, col); self.c.rrects += 1; }
    fn circle(&mut self, c: Pos2, r: f32, col: Color32) { self.p.circle_filled(c, r, col); self.c.circles += 1; }
    fn line(&mut self, pts: Vec<Pos2>, s: Stroke) { self.c.lines += 1; self.c.pts += pts.len(); self.p.add(Shape::line(pts, s)); }
    fn text(&mut self, at: Pos2, a: Align2, s: &str, size: f32, col: Color32) {
        self.c.texts += 1; self.c.chars += s.chars().count();
        self.p.text(at, a, s, FontId::proportional(size), col);
    }
}

const GOOD: Color32 = Color32::from_rgb(0x5d, 0xbf, 0x83);
const BAD: Color32 = Color32::from_rgb(0xef, 0x93, 0x87);
const MUTED: Color32 = Color32::from_rgb(0xa6, 0xac, 0xc1);
const INK: Color32 = Color32::from_rgb(0xee, 0xf0, 0xf6);
const AMBER: Color32 = Color32::from_rgb(0xe9, 0x9c, 0x57);
const LINE: Color32 = Color32::from_rgba_premultiplied(30, 31, 34, 37);
const SHADES: [f32; 4] = [0.10, 0.22, 0.36, 0.54];

fn h(i: usize) -> f32 { // 결정적 의사난수 0..1
    let x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    ((x >> 40) as f32) / ((1u64 << 24) as f32)
}

pub fn fmt_qty(v: f32) -> String { // app.js fmtFootprintQty
    if v >= 1e6 { format!("{:.1}M", v / 1e6) } else if v >= 10000.0 { format!("{:.0}k", v / 1000.0) }
    else if v >= 1000.0 { format!("{:.1}k", v / 1000.0) } else if v >= 100.0 { format!("{v:.0}") } else { format!("{v:.1}") }
}

/// 한 장 그리기. f = 프레임 번호(형성 봉 칸 값이 매 프레임 바뀐다). churn = 모든 글자가 매 프레임 바뀌는 최악(글자 모양 캐시 무력화). 반환: 그린 명령 수.
pub fn paint(painter: &egui::Painter, t: &Target, f: usize, churn: bool) -> Cnt {
    let mut p = P { p: painter, c: Cnt::default() };
    p.p.rect_filled(Rect::from_min_size(pos2(0.0, 0.0), vec2(1920.0, 1080.0)), 0.0, Color32::from_rgb(0x16, 0x1a, 0x22));
    let plot = Rect::from_min_max(pos2(113.0, 183.0), pos2(1126.0, 770.0));
    let slot = plot.width() / t.bars as f32;
    let cw = (slot * 0.14).clamp(3.0, 8.0);
    let (y_max, row_size) = (1760.0_f32, 0.5 * (t.row_px / 4.0).round().max(1.0));
    let ypx = |price: f32| plot.min.y + (y_max - price) / row_size * t.row_px;

    // 격자 · 가격 눈금
    for k in 0..8 {
        let y = plot.min.y + k as f32 * plot.height() / 8.0;
        p.rect(Rect::from_min_size(pos2(plot.min.x, y), vec2(plot.width(), 1.0)), LINE);
        p.text(pos2(plot.max.x + 4.0, y), Align2::LEFT_CENTER, &format!("{:.1}", y_max - (y - plot.min.y) / t.row_px * row_size), 10.0, MUTED);
    }
    // 봉: 칸(매도|캔들|매수) · 숫자 · 불균형 · POC · 델타
    let mut poc = Vec::with_capacity(t.bars);
    let mut row0 = 18.0_f32;
    let per = t.rows as f32 / t.bars as f32;
    for i in 0..t.bars {
        let n = (per + (h(i) - 0.5) * 3.0).round().max(2.0) as usize;
        row0 += (h(i + 999) - 0.5) * 4.0;
        let x = plot.min.x + i as f32 * slot + (slot - (2.0 * t.half + cw + 3.0)) / 2.0;
        let live = i + 1 == t.bars;
        let (mut bt, mut st, mut pk, mut pv) = (0.0, 0.0, 0usize, 0.0);
        let cells: Vec<(f32, f32)> = (0..n).map(|k| {
            let bump = if live || churn { f as f32 * 0.37 + ((f + k) % 5) as f32 * 0.11 } else { 0.0 };
            (20.0 + h(i * 64 + k) * 3000.0 + bump, 20.0 + h(i * 64 + k + 7777) * 3000.0 + bump * 0.8)
        }).collect();
        for (k, &(b, s)) in cells.iter().enumerate() { bt += b; st += s; if b + s > pv { pv = b + s; pk = k; } }
        for (k, &(b, s)) in cells.iter().enumerate() {
            let y = ypx(y_max - (row0 + k as f32) * row_size);
            let col = if s > b { BAD } else { GOOD };
            let sx = x;
            let bx = x + t.half + 1.0 + cw + 2.0;
            for (cx, v, o, ex) in [(sx, s, b, sx - 2.0), (bx, b, s, bx + t.half + 2.0)] {
                p.rect(Rect::from_min_size(pos2(cx, y), vec2(t.half, t.row_px)), col.gamma_multiply(SHADES[((v / 3000.0 * 4.0) as usize).min(3)]));
                if t.qty { p.text(pos2(cx + t.half / 2.0, y + t.row_px / 2.0), Align2::CENTER_CENTER, &fmt_qty(v), 9.0, INK.gamma_multiply(0.82)); }
                if v > o * 3.0 { p.rect(Rect::from_min_size(pos2(ex, y + 0.5), vec2(2.0, t.row_px - 1.0)), col); }
            }
            if k == pk {
                p.c.lines += 1; p.c.pts += 5;
                p.p.rect_stroke(Rect::from_min_max(pos2(sx, y), pos2(bx + t.half, y + t.row_px)), 0.0, Stroke::new(1.0, AMBER.gamma_multiply(0.85)), StrokeKind::Inside);
                poc.push(pos2(x + t.half + cw / 2.0, y + t.row_px / 2.0));
            }
        }
        let top = ypx(y_max - row0 * row_size) - 4.0;
        let bot = ypx(y_max - (row0 + n as f32) * row_size) + 4.0;
        let up = bt >= st;
        let ccol = if up { GOOD } else { BAD };
        let cx = x + t.half + 2.0;
        p.rect(Rect::from_min_size(pos2(cx + cw / 2.0 - 0.5, top), vec2(1.0, bot - top)), ccol);
        p.rrect(Rect::from_min_max(pos2(cx, top + (bot - top) * 0.3), pos2(cx + cw, bot - (bot - top) * 0.3)), ccol);
        let d = bt - st;
        p.text(pos2(cx + cw / 2.0, bot + 13.0), Align2::CENTER_BOTTOM, &format!("{}{}", if d >= 0.0 { "+" } else { "−" }, fmt_qty(d.abs())), 11.0, ccol);
    }
    p.line(poc, Stroke::new(1.5, AMBER.gamma_multiply(0.75)));
    // 현재가 선(점선) + 꼬리표
    let yc = plot.center().y + (f % 7) as f32;
    let dashed = Shape::dashed_line(&[pos2(plot.min.x, yc), pos2(1370.0, yc)], Stroke::new(1.0, INK.gamma_multiply(0.55)), 3.0, 3.0);
    p.c.lines += 1; p.c.pts += 2;
    p.p.extend(dashed);
    p.rrect(Rect::from_min_size(pos2(1290.0, yc - 9.0), vec2(76.0, 18.0)), Color32::from_rgb(0x17, 0x1b, 0x23));
    p.text(pos2(1328.0, yc), Align2::CENTER_CENTER, &format!("{:.1}", 1741.8 + (f % 10) as f32 * 0.1), 13.0, INK);
    // 시간축
    for k in (0..t.bars).step_by((t.bars / 6).max(1)) {
        let m = (k * 5 + 40) % 60;
        p.text(pos2(plot.min.x + k as f32 * slot, 893.0), Align2::CENTER_CENTER, &format!("{}:{m:02}", 14 + (k * 5 + 40) / 60), 12.0, MUTED);
    }

    // ── 나머지 층을 같은 개수로 채운다(레인·프로파일·오른쪽 칸·표식·글자) ──────────────────
    let c0 = p.c;
    let lane = Rect::from_min_max(pos2(113.0, 775.0), pos2(1126.0, 880.0));
    let rest_r = t.rects.saturating_sub(c0.rects);
    for k in 0..rest_r { // 레인·밀도 칸·프로파일 막대
        let (gx, gy) = (k % 60, k / 60);
        let r = if gy < 6 { Rect::from_min_size(pos2(lane.min.x + gx as f32 * 16.8, lane.min.y + gy as f32 * 17.0), vec2(15.0, 15.0)) }
                else { Rect::from_min_size(pos2(1130.0 + (k % 3) as f32 * 80.0, 183.0 + ((k / 3) % 120) as f32 * 5.0), vec2(20.0 + h(k + f * (k % 4 == 0) as usize) * 50.0, 4.0)) };
        p.rect(r, if k % 2 == 0 { GOOD.gamma_multiply(0.4) } else { BAD.gamma_multiply(0.4) });
    }
    for k in 0..t.rrects.saturating_sub(p.c.rrects) { // 칩·꼬리표 배경
        p.rrect(Rect::from_min_size(pos2(113.0 + (k % 40) as f32 * 25.0, 900.0 + (k / 40) as f32 * 22.0), vec2(22.0, 18.0)), MUTED.gamma_multiply(0.3));
    }
    for k in 0..t.circles.saturating_sub(p.c.circles) { // 표식 원
        p.circle(pos2(150.0 + k as f32 * 30.0, 300.0 + h(k) * 300.0), 3.0 + h(k + 5) * 8.0, BAD.gamma_multiply(0.7));
    }
    let nl = t.lines.saturating_sub(p.c.lines).max(1);
    let np = t.pts.saturating_sub(p.c.pts);
    for k in 0..nl { // 오른쪽 1초 수급 선 · 프로파일 윤곽 · 0선
        let m = (np / nl + (k < np % nl) as usize).max(2);
        let y0 = 150.0 + (k % 12) as f32 * 22.0;
        let pts: Vec<Pos2> = (0..m).map(|j| pos2(1400.0 + j as f32 * 450.0 / m as f32, y0 + (h(k * 4096 + j + f * (k < 4) as usize) - 0.5) * 30.0)).collect();
        p.line(pts, Stroke::new(1.0, if k % 3 == 0 { AMBER } else { MUTED }));
    }
    let nt = t.texts.saturating_sub(p.c.texts).max(1);
    let nc = t.chars.saturating_sub(p.c.chars);
    const WORDS: [&str; 8] = ["청산", "체결 · 델타", "호가", "고래", "개미", "수급", "1초", "TPO"];
    for k in 0..nt { // 라벨(한글 섞음) -- 앞 10개는 매 프레임 값이 바뀐다(1초 수급 숫자)
        let want = nc / nt + (k < nc % nt) as usize;
        let mut s = format!("{} {:.1}", WORDS[k % 8], if k < 10 || churn { f as f32 * 0.7 + k as f32 } else { k as f32 * 1.3 });
        while s.chars().count() < want { s.push_str(WORDS[(k + s.len()) % 8]); s.push(' '); }
        let s: String = s.chars().take(want.max(1)).collect();
        p.text(pos2(1400.0 + (k % 3) as f32 * 150.0, 450.0 + (k / 3) as f32 * 13.0 % 600.0), Align2::LEFT_CENTER, &s, 11.0, MUTED);
    }
    p.c
}

pub fn fonts() -> FontDefinitions {
    #[allow(unused_mut)]
    let mut f = FontDefinitions::default();
    #[cfg(feature = "kfont")]
    {
        f.font_data.insert("pretendard".into(), Arc::new(FontData::from_static(include_bytes!("../fonts/Pretendard-Bold.ttf"))));
        for fam in [FontFamily::Proportional, FontFamily::Monospace] { f.families.get_mut(&fam).unwrap().push("pretendard".into()); }
    }
    f
}

/// 창 없이 egui 한 프레임: UI 구성(run_ui) + 테셀레이션. GPU 로 넘기는 일은 없다 -- 웹·네이티브가 같은 코드.
pub struct Bench { ctx: Context, t: usize, f: usize, churn: bool, pub cnt: Cnt }
impl Bench {
    pub fn new(scenario: usize, churn: bool) -> Self {
        let ctx = Context::default();
        ctx.set_fonts(fonts());
        Self { ctx, t: scenario, f: 0, churn, cnt: Cnt::default() }
    }
    /// (구성 ms, 테셀레이션 ms, 삼각형 정점 수)
    pub fn frame(&mut self, now: impl Fn() -> f64) -> (f64, f64, usize) {
        let raw = egui::RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1920.0, 1080.0))), ..Default::default() };
        let (t, f, ch) = (&TARGETS[self.t], self.f, self.churn);
        let mut cnt = Cnt::default();
        let t0 = now();
        let out = self.ctx.run_ui(raw, |ui| { cnt = paint(ui.painter(), t, f, ch); });
        let t1 = now();
        let prims = self.ctx.tessellate(out.shapes, out.pixels_per_point);
        let t2 = now();
        self.cnt = cnt;
        self.f += 1;
        let verts = prims.iter().map(|p| if let egui::epaint::Primitive::Mesh(m) = &p.primitive { m.vertices.len() } else { 0 }).sum();
        (t1 - t0, t2 - t1, verts)
    }
}

/// 창 있는 앱(네이티브 glow · 웹 WebGL): eframe 이 재는 frame.info().cpu_usage(앱 ui + 테셀레이션 + GL 제출)를 모은다.
pub struct App { pub t: usize, pub f: usize, pub cpu: Vec<f32>, pub limit: usize, pub on_done: Option<Box<dyn FnMut(&[f32])>> }
impl App {
    pub fn new(cc: &eframe::CreationContext, t: usize, limit: usize) -> Self {
        cc.egui_ctx.set_fonts(fonts());
        Self { t, f: 0, cpu: Vec::new(), limit, on_done: None }
    }
}
impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        if let Some(c) = frame.info().cpu_usage { if self.f > 30 { self.cpu.push(c * 1000.0); } }
        paint(ui.painter(), &TARGETS[self.t], self.f, false);
        self.f += 1;
        if self.cpu.len() == self.limit { if let Some(cb) = self.on_done.as_mut() { cb(&self.cpu); } }
        ui.ctx().request_repaint();
    }
}

pub fn stats(v: &[f64]) -> String {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let q = |a: f64| s[((a * s.len() as f64) as usize).min(s.len() - 1)];
    format!("n={} med={:.3} p99={:.3} max={:.3}", s.len(), q(0.5), q(0.99), s[s.len() - 1])
}

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = performance)]
        fn now() -> f64;
    }

    #[wasm_bindgen]
    pub struct WBench(Bench);
    #[wasm_bindgen]
    impl WBench {
        #[wasm_bindgen(constructor)]
        pub fn new(scenario: usize, churn: bool) -> WBench { WBench(Bench::new(scenario, churn)) }
        /// [구성 ms, 테셀레이션 ms, 정점 수, 사각형, 둥근, 원, 선, 점, 글자 항목, 글자 수]
        pub fn frame(&mut self) -> Vec<f64> {
            let (a, b, v) = self.0.frame(now);
            let c = self.0.cnt;
            vec![a, b, v as f64, c.rects as f64, c.rrects as f64, c.circles as f64, c.lines as f64, c.pts as f64, c.texts as f64, c.chars as f64]
        }
    }

    thread_local! { static CPU: std::cell::RefCell<Vec<f32>> = Default::default(); }
    #[wasm_bindgen]
    pub fn web_cpu() -> Vec<f32> { CPU.with(|c| c.borrow().clone()) }

    #[wasm_bindgen]
    pub async fn start(canvas_id: String, scenario: usize, frames: usize) -> Result<(), JsValue> {
        use eframe::wasm_bindgen::JsCast;
        let canvas = eframe::web_sys::window().unwrap().document().unwrap().get_element_by_id(&canvas_id).unwrap()
            .dyn_into::<eframe::web_sys::HtmlCanvasElement>()?;
        eframe::WebRunner::new().start(canvas, eframe::WebOptions::default(), Box::new(move |cc| {
            let mut a = App::new(cc, scenario, frames);
            a.on_done = Some(Box::new(|v| CPU.with(|c| *c.borrow_mut() = v.to_vec())));
            Ok(Box::new(a))
        })).await
    }
}
