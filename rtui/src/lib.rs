//! rtui -- 웹 대시보드 화면을 egui(네이티브 + 웹어셈블리)로. 첫 카드: 데스크톱 1920 풋프린트 차트 영역(지금 화면 그대로 옮김).
//! data = 응답 모양 · fp = 가공(app.js 식 그대로 → 그리기 목록) · draw = egui 그리기.
//! 마우스(호버·끌기·확대)는 아직 없다: 붙일 자리는 Chart::ui 가 받는 Ui(응답 영역)와 Scene 의 기하(xAt·yAt 의 재료)다.
pub mod data;
pub mod draw;
pub mod fp;

use eframe::egui::{self, Color32, Context, Pos2, Rect, vec2};

pub struct Chart {
    pub fp: data::Footprint,
    pub candles: Vec<data::Candle>,
    pub cfg: fp::Cfg,
    /// svg 상자 왼쪽 위(화면 px) -- 비교 캡처에서 지금 화면과 같은 자리
    pub origin: Pos2,
    /// 바탕(카드 면은 이 카드 밖 -- 투명이면 칠하지 않는다: 비교 캡처는 지금 화면의 바탕을 뒤에 깐다)
    pub bg: Color32,
}

impl Chart {
    pub fn from_json(fp_json: &str, mh_json: &str, cfg: fp::Cfg, origin: Pos2, bg: Color32) -> Result<Self, serde_json::Error> {
        let fp: data::Footprint = serde_json::from_str(fp_json)?;
        let mh: data::MarketHistory = serde_json::from_str(mh_json)?;
        Ok(Self { fp, candles: mh.candles, cfg, origin, bg })
    }
    /// 한 장: 가공(prep) + 그리기. 반환 = 그린 장면(좌표 대조용)
    pub fn ui(&self, ui: &mut egui::Ui) -> Option<fp::Scene> {
        let p = ui.painter();
        if self.bg.a() > 0 { p.rect_filled(p.clip_rect(), 0.0, self.bg); }
        let sc = fp::prep(&self.fp, &self.candles, &self.cfg, &|s, z| draw::measure(p, s, z))?;
        draw::paint(p, &sc, self.origin);
        Some(sc)
    }
    /// 형성 봉 칸에 체결이 들어온 것처럼 값을 바꾼다(측정용 -- rtui_proto/bench_js.py 의 자바스크립트 측정과 같은 식)
    pub fn nudge(&mut self, f: usize) {
        if let Some(b) = self.fp.bars.last_mut() {
            for (j, l) in b.levels.iter_mut().enumerate() {
                l[1] += 0.37 + ((f + j) % 5) as f64 * 0.11;
                l[2] += 0.29 + ((f * 3 + j) % 7) as f64 * 0.07;
            }
        }
    }
}

pub fn new_ctx() -> Context {
    let ctx = Context::default();
    ctx.set_fonts(draw::fonts());
    draw::canvas_like(&ctx);
    ctx
}

/// 창 없이 egui 한 프레임: UI 구성(가공 포함) + 테셀레이션 -- 웹·네이티브가 같은 코드. 반환 (구성 ms, 테셀레이션 ms)
pub fn bench_frame(ctx: &Context, ch: &Chart, now: &dyn Fn() -> f64) -> (f64, f64) {
    let raw = egui::RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1920.0, 1080.0))), ..Default::default() };
    let t0 = now();
    let out = ctx.run_ui(raw, |ui| { ch.ui(ui); });
    let t1 = now();
    let _ = ctx.tessellate(out.shapes, out.pixels_per_point);
    (t1 - t0, now() - t1)
}

pub fn stats(v: &[f64]) -> String {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let q = |a: f64| s[((a * s.len() as f64) as usize).min(s.len() - 1)];
    format!("n={} med={:.3} p99={:.3} max={:.3}", s.len(), q(0.5), q(0.99), s[s.len() - 1])
}

/// 좌표 대조용: 그린 목록을 JSON 으로(비교 하네스가 지금 화면의 캔버스 명령과 맞댄다)
pub fn scene_json(sc: &fp::Scene) -> String {
    use fp::Prim::*;
    let items: Vec<serde_json::Value> = sc.prims.iter().map(|p| match p {
        Rect { x, y, w, h, c, a, .. } => serde_json::json!({"k": "rect", "x": x, "y": y, "w": w, "h": h, "c": format!("{c:?}"), "a": a}),
        StrokeRect { x, y, w, h, c, .. } => serde_json::json!({"k": "srect", "x": x, "y": y, "w": w, "h": h, "c": format!("{c:?}")}),
        Line { pts, c, .. } => serde_json::json!({"k": "line", "pts": pts, "c": format!("{c:?}")}),
        Text { x, y, s, size, c, .. } => serde_json::json!({"k": "text", "x": x, "y": y, "s": s, "size": size, "c": format!("{c:?}")}),
    }).collect();
    serde_json::json!({"geo": {"w": sc.w, "mr": sc.mr, "ml": sc.ml, "cw": sc.cw, "mt": sc.mt, "ch": sc.ch, "pb": sc.plot_bottom, "lane_top": sc.lane_top,
        "ymin": sc.y_min, "ymax": sc.y_max, "slot": sc.slot, "bw": sc.bw, "row_px": sc.row_px, "cp": sc.current}, "prims": items}).to_string()
}

/// 창 있는 앱(네이티브 glow · 웹 WebGL)
pub struct App { pub chart: Chart, pub last: Option<String> }
impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if let Some(sc) = self.chart.ui(ui) { if self.last.is_none() { self.last = Some(scene_json(&sc)); } }
    }
}

pub fn default_cfg(window: f64, w_all: f64, h_all: f64, tz_off_min: i64) -> fp::Cfg {
    fp::Cfg { w_all, h_all, split: 0.25, window, tz_off_min }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use eframe::egui::pos2;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = performance)]
        fn now() -> f64;
    }

    fn chart(fp_json: &str, mh_json: &str, x: f32, y: f32, w: f64, h: f64, bars: f64, tz: i32, bg: u32) -> Result<Chart, JsValue> {
        // bg = 0xRRGGBB · 0xFFFFFFFF = 투명
        let bgc = if bg == u32::MAX { Color32::TRANSPARENT } else { Color32::from_rgb((bg >> 16) as u8, (bg >> 8) as u8, bg as u8) };
        Chart::from_json(fp_json, mh_json, default_cfg(bars, w, h, tz as i64), pos2(x, y), bgc).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    thread_local! { static LAST: std::cell::RefCell<Option<String>> = Default::default(); }
    /// 마지막으로 그린 장면(JSON) -- 첫 프레임 뒤에 생긴다
    #[wasm_bindgen]
    pub fn scene() -> Option<String> { LAST.with(|l| l.borrow().clone()) }

    struct WebApp(App);
    impl eframe::App for WebApp {
        fn clear_color(&self, _v: &egui::Visuals) -> [f32; 4] { self.0.chart.bg.to_normalized_gamma_f32() }
        fn ui(&mut self, ui: &mut egui::Ui, f: &mut eframe::Frame) {
            self.0.ui(ui, f);
            if let Some(s) = self.0.last.take() { LAST.with(|l| *l.borrow_mut() = Some(s)); }
        }
    }

    #[wasm_bindgen]
    #[allow(clippy::too_many_arguments)]
    pub async fn start(canvas_id: String, fp_json: String, mh_json: String, x: f32, y: f32, w: f64, h: f64, bars: f64, tz: i32, bg: u32) -> Result<(), JsValue> {
        use eframe::wasm_bindgen::JsCast;
        let ch = chart(&fp_json, &mh_json, x, y, w, h, bars, tz, bg)?;
        let canvas = eframe::web_sys::window().unwrap().document().unwrap().get_element_by_id(&canvas_id).unwrap()
            .dyn_into::<eframe::web_sys::HtmlCanvasElement>()?;
        eframe::WebRunner::new().start(canvas, eframe::WebOptions::default(), Box::new(move |cc| {
            cc.egui_ctx.set_fonts(draw::fonts());
            draw::canvas_like(&cc.egui_ctx);
            Ok(Box::new(WebApp(App { chart: ch, last: None })))
        })).await
    }

    /// 측정: 창 없이 n 프레임(앞 30 버림) -- [구성 ms, 테셀레이션 ms] × n 을 펼친 배열
    #[wasm_bindgen]
    #[allow(clippy::too_many_arguments)]
    pub fn bench(fp_json: String, mh_json: String, w: f64, h: f64, bars: f64, tz: i32, n: usize) -> Result<Vec<f64>, JsValue> {
        let mut ch = chart(&fp_json, &mh_json, 113.0, 126.0, w, h, bars, tz, 0x1e2229)?;
        let ctx = new_ctx();
        let mut out = vec![];
        for f in 0..n + 30 {
            ch.nudge(f);
            let (a, b) = bench_frame(&ctx, &ch, &now);
            if f >= 30 { out.push(a); out.push(b); }
        }
        Ok(out)
    }
}
