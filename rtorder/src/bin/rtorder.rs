//! rtorder 실행 파일(2026-11-21 D 뼈대): 설정 → 모드 → 시작 복구 → 감시 루프(+사용자 스트림·rtcore 허브 구독) → SIGTERM/SIGINT 로 멈춤.
//! 🔴기본 = 그림자(쓰기를 보내지 않는다). 기본 빌드에는 키를 읽는 코드도 실전송도 없다 -- `cargo build --features live` + RTORDER_MODE=live + RTORDER_LIVE_CONFIRM 셋이 있어야 실전송.
//! 접수 HTTP(웹 서버)는 아직 없다(R6). 종료코드: 0 = 정상 멈춤 · 2 = 설정·모드 오류(시작 안 함).

use rtorder::app::{self, Stop};
use rtorder::http::Binance;
use rtorder::shadow::{self, Mode};

fn main() {
    let get = |k: &str| std::env::var(k).ok();
    let cfg = app::config_from(&get).unwrap_or_else(|e| {
        eprintln!("rtorder: 설정 오류 -- {e}");
        std::process::exit(2)
    });
    let mode = shadow::mode_from(&get).unwrap_or_else(|e| {
        eprintln!("rtorder: 모드 오류 -- {e}");
        std::process::exit(2)
    });
    std::fs::create_dir_all(&cfg.dir).unwrap_or_else(|e| {
        eprintln!("rtorder: 폴더 {}: {e}", cfg.dir.display());
        std::process::exit(2)
    });
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let stop = Stop::default();
    let (c2, s2) = (cfg.clone(), stop.clone());
    let mk = move || make(&c2, &mode);
    let s = rt.block_on(async move {
        let sig = s2.clone();
        tokio::spawn(async move {
            let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).unwrap();
            let mut int = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt()).unwrap();
            tokio::select! { _ = term.recv() => {}, _ = int.recv() => {} }
            sig.stop();
        });
        app::run(&cfg, &mk, s2).await
    });
    for l in &s.logs {
        println!("{l}");
    }
    for x in &s.would {
        println!("그림자(안 보냄): {x}");
    }
    println!("rtorder 종료 · 틱 {} · {} · 안 보낸 쓰기 {}", s.ticks, if s.shadow { "그림자" } else { "실전송" }, s.would.len());
}

/// 같은 설정의 전송 하나. 기본 빌드 = 키 없음(계좌도 못 읽는 순수 그림자) · feature live = 환경 키 · Live 모드면 그림자 해제.
fn make(cfg: &app::Config, mode: &Mode) -> Binance {
    #[cfg(feature = "live")]
    let mut b = Binance::from_env(&cfg.base, cfg.ban_file.clone());
    #[cfg(not(feature = "live"))]
    let mut b = Binance::new(&cfg.base, "", "", rtorder::http::TIMEOUT, cfg.ban_file.clone());
    match mode {
        Mode::Shadow => b.shadow_on(),
        #[cfg(feature = "live")]
        Mode::Live(p) => b.arm_live(p),
        #[cfg(not(feature = "live"))]
        Mode::Live(_) => b.shadow_on(), // 기본 빌드에서는 만들어지지 않는 값 -- 와도 그림자
    }
    b
}
