//! cargo run --release -- bench <봉 수> <프레임>   : 창 없이 가공 + UI 구성 + 테셀레이션(웹어셈블리와 같은 코드)
//! cargo run --release -- window <봉 수>          : glow 창(WSLg 는 GALLIUM_DRIVER=llvmpipe)
//! 자료 = compare/fixture(합성). 상자 = 1920×1080 데스크톱에서 지금 화면의 svg 자리.
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use rtui::*;
    let a: Vec<String> = std::env::args().collect();
    let bars: f64 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(36.0);
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/compare/fixture/");
    let rd = |f: &str| std::fs::read_to_string(format!("{dir}{f}")).expect("python -I compare/make_fixture.py 먼저");
    let bg = eframe::egui::Color32::from_rgb(0x1e, 0x22, 0x29);
    let mut ch = Chart::from_json(&rd("footprint.json"), &rd("market-history.json"), default_cfg(bars, 1751.0, 844.0, -540),
                                  eframe::egui::pos2(113.0, 126.0), bg);
    if a.get(1).map(String::as_str) == Some("bench") {
        let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(300);
        let ctx = new_ctx();
        let t0 = std::time::Instant::now();
        let now = move || t0.elapsed().as_secs_f64() * 1000.0;
        let first = bench_frame(&ctx, &mut ch, &now);
        let raw = eframe::egui::RawInput { screen_rect: Some(eframe::egui::Rect::from_min_size(eframe::egui::Pos2::ZERO, eframe::egui::vec2(1920.0, 1080.0))), ..Default::default() };
        let mut np = 0;
        let out = ctx.run_ui(raw, |ui| { np = ch.ui(ui, 0.0).map_or(0, |s| s.0.prims.len()); });
        let verts: usize = ctx.tessellate(out.shapes, 1.0).iter().map(|p| match &p.primitive { eframe::epaint::Primitive::Mesh(m) => m.vertices.len(), _ => 0 }).sum();
        println!("prims {np} · vertices {verts}");
        let (mut bu, mut te, mut to) = (vec![], vec![], vec![]);
        for f in 0..n + 29 {
            ch.nudge(f);
            let (x, y) = bench_frame(&ctx, &mut ch, &now);
            if f >= 29 { bu.push(x); te.push(y); to.push(x + y); }
        }
        println!("first frame build={:.2} tess={:.2} ms", first.0, first.1);
        println!("build  {}\ntess   {}\ntotal  {}", stats(&bu), stats(&te), stats(&to));
    } else {
        let opts = eframe::NativeOptions {
            viewport: eframe::egui::ViewportBuilder::default().with_inner_size([1920.0, 1080.0]),
            renderer: eframe::Renderer::Glow,
            ..Default::default()
        };
        eframe::run_native("rtui", opts, Box::new(move |cc| {
            cc.egui_ctx.set_fonts(rtui::draw::fonts());
            rtui::draw::canvas_like(&cc.egui_ctx);
            Ok(Box::new(App { chart: ch, last: None, now: || 0.0 }))
        })).unwrap();
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
