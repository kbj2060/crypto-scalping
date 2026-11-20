//! cargo run --release -- bench <0|1> <프레임> [churn] : 창 없이 UI 구성 + 테셀레이션 (웹어셈블리와 같은 코드)
//! cargo run --release -- window <0|1> <프레임>  : glow 창, eframe 의 cpu_usage (WSLg 는 GALLIUM_DRIVER=llvmpipe)
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use rtui_proto::*;
    let a: Vec<String> = std::env::args().collect();
    let (mode, sc, n) = (a[1].as_str(), a[2].parse::<usize>().unwrap(), a[3].parse::<usize>().unwrap());
    if mode == "bench" {
        let mut b = Bench::new(sc, a.get(4).is_some_and(|x| x == "churn"));
        let t0 = std::time::Instant::now();
        let now = move || t0.elapsed().as_secs_f64() * 1000.0;
        let first = b.frame(now);
        for _ in 0..29 { b.frame(now); }
        let (mut bu, mut te, mut to, mut vs) = (vec![], vec![], vec![], 0);
        for _ in 0..n { let (x, y, v) = b.frame(now); bu.push(x); te.push(y); to.push(x + y); vs = v; }
        println!("first frame build={:.2} tess={:.2} ms", first.0, first.1);
        println!("counts {:?} verts={vs}", b.cnt);
        println!("build  {}\ntess   {}\ntotal  {}", stats(&bu), stats(&te), stats(&to));
    } else {
        let opts = eframe::NativeOptions {
            viewport: eframe::egui::ViewportBuilder::default().with_inner_size([1920.0, 1080.0]),
            renderer: eframe::Renderer::Glow,
            ..Default::default()
        };
        eframe::run_native("rtui_proto", opts, Box::new(move |cc| {
            let mut app = App::new(cc, sc, n);
            app.on_done = Some(Box::new(|v| {
                println!("cpu_usage {}", stats(&v.iter().map(|x| *x as f64).collect::<Vec<_>>()));
                std::process::exit(0);
            }));
            Ok(Box::new(app))
        })).unwrap();
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
