//! 그림자 재생 실행기: 파이썬 서버가 남긴 그림자 기록(JSONL · 형식은 `rtorder::shadow` · 필요한 칸은 RULES.md «그림자 기록»)을 읽어
//! «같은 접수·같은 입력이면 Rust 가 낼 요청 열»을 127.0.0.1 가짜 서버(기록된 응답을 차례로 돌려준다) 위에서 돌리고 차이를 적는다.
//! 거래소·키·외부 네트워크를 쓰지 않는다. 비밀값은 받지 않는다(서명은 빼고 비교).
//! 기록 = 파이썬 `live/execution/order_shadow.py` 가 남기는 `data/live/order_shadow.jsonl`(2026-11-21 · 서버에서 그대로).
//! 실행: cargo run --example shadow_replay -- <기록.jsonl> [주문 심볼=ETHUSDC]   · 종료코드 0 = 차이 없음 · 1 = 차이 · 2 = 입력 오류

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("쓰는 법: shadow_replay <기록.jsonl> [ETHUSDC]");
        std::process::exit(2);
    };
    let symbol = args.get(2).cloned().unwrap_or_else(|| "ETHUSDC".into());
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(2)
    });
    let chs = rtorder::shadow::chains(&text).unwrap_or_else(|e| {
        eprintln!("기록 형식: {e}");
        std::process::exit(2)
    });
    let dir = std::env::temp_dir().join(format!("rtorder_shadow_replay_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let reports = rt.block_on(rtorder::shadow::replay(&chs, &symbol, &dir, false, None));
    let _ = std::fs::remove_dir_all(&dir);
    print!("{}", rtorder::shadow::report_text(&reports));
    let bad = reports.iter().filter(|r| !r.diffs.is_empty()).count();
    std::process::exit(if bad == 0 { 0 } else { 1 });
}
