//! rtorder 실행 파일(2026-11-21 D 뼈대): 설정 → 모드 → 시작 복구 → 감시 루프(+사용자 스트림·rtcore 허브 구독) → SIGTERM/SIGINT 로 멈춤.
//! 🔴기본 = 그림자(쓰기를 보내지 않는다). 기본 빌드에는 키를 읽는 코드도 실전송도 없다 -- `cargo build --features live` + RTORDER_MODE=live + RTORDER_LIVE_CONFIRM 셋이 있어야 실전송.
//! 접수 = 루프백 HTTP 문(RTORDER_LISTEN · 기본 127.0.0.1:8791 · `door.rs`) -- 응답 서버가 주문·계좌 주소 14개를 넘긴다. 종료코드: 0 = 정상 멈춤 · 2 = 설정·모드 오류(시작 안 함).

use rtorder::app::{self, Stop};
use rtorder::http::Binance;
use rtorder::shadow::{self, Mode};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("replay") {
        std::process::exit(replay(&args[2..]));
    }
    let get = |k: &str| std::env::var(k).ok();
    let mut cfg = app::config_from(&get).unwrap_or_else(|e| {
        eprintln!("rtorder: 설정 오류 -- {e}");
        std::process::exit(2)
    });
    let mode = shadow::mode_from(&get).unwrap_or_else(|e| {
        eprintln!("rtorder: 모드 오류 -- {e}");
        std::process::exit(2)
    });
    {
        // 첫 줄 = 모드(감독 로그에서 바로 보이게)
        use std::io::Write;
        match &mode {
            Mode::Shadow => println!("모드: 그림자 · 쓰기 막힘"),
            Mode::Live(_) => println!("모드: 실전송"),
        }
        let _ = std::io::stdout().flush();
    }
    std::fs::create_dir_all(&cfg.dir).unwrap_or_else(|e| {
        eprintln!("rtorder: 폴더 {}: {e}", cfg.dir.display());
        std::process::exit(2)
    });
    cfg.echo = true;
    // 루프백 문: 묶기 실패(이미 다른 rtorder 가 이 주소를 쥠 등) = 시작 안 함 -- 같은 계좌를 감시하는 둘째 프로세스가 뜨지 않게
    let std_l = std::net::TcpListener::bind(cfg.listen).and_then(|l| l.set_nonblocking(true).map(|_| l)).unwrap_or_else(|e| {
        eprintln!("rtorder: 문 {} 을 열 수 없습니다 -- {e}", cfg.listen);
        std::process::exit(2)
    });
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let stop = Stop::default();
    let (c2, s2) = (cfg.clone(), stop.clone());
    let mk = move || make(&c2, &mode);
    let s = rt.block_on(async move {
        let sig = s2.clone();
        // 신호 받기를 먼저 건다 -- 문 주소 줄(= «떴다»)을 본 쪽이 곧바로 SIGTERM 을 보내도 기본 동작(즉사)으로 죽지 않게
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).unwrap();
        let mut int = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt()).unwrap();
        tokio::spawn(async move {
            tokio::select! { _ = term.recv() => {}, _ = int.recv() => {} }
            sig.stop();
        });
        {
            use std::io::Write;
            println!("rtorder 문 http://{}", std_l.local_addr().unwrap());
            let _ = std::io::stdout().flush();
        }
        let l = tokio::net::TcpListener::from_std(std_l).unwrap();
        app::run(&cfg, &mk, s2, Default::default(), Some(l)).await
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

/// `rtorder replay <order_shadow.jsonl> [--show-values] [--symbol ETHUSDC]` (2026-11-21 · 서버의 파이썬 그림자 실기록 재생).
/// 🔴바깥으로 아무것도 보내지 않는다: 127.0.0.1 가짜 서버(기록된 응답을 차례로 돌려준다)만 · 그 밖 주소로 가는 요청은 보내기 전에 panic(`http::LOOPBACK_ONLY`).
/// 입력은 읽기만(`<파일>.1` 이 있으면 먼저 -- 돌린 앞부분). 실기록에는 계좌 값이 있어 출력은 chain·op·요청 수·어긋난 위치(칸 이름)까지 -- 값은 `--show-values` 일 때만.
/// 종료코드 0 = 어긋남 없음 · 1 = 어긋남 · 2 = 입력 오류.
fn replay(args: &[String]) -> i32 {
    rtorder::http::LOOPBACK_ONLY.store(true, std::sync::atomic::Ordering::SeqCst);
    let show = args.iter().any(|a| a == "--show-values");
    let symbol = args.iter().position(|a| a == "--symbol").and_then(|i| args.get(i + 1)).cloned()
        .or_else(|| std::env::var("DASHBOARD_MANUAL_EXEC_SYMBOL").ok().filter(|s| !s.is_empty())).unwrap_or_else(|| "ETHUSDC".into());
    let Some(path) = args.iter().find(|a| !a.starts_with("--") && Some(*a) != args.iter().position(|x| x == "--symbol").and_then(|i| args.get(i + 1))) else {
        eprintln!("쓰는 법: rtorder replay <order_shadow.jsonl> [--show-values] [--symbol ETHUSDC]");
        return 2;
    };
    let mut texts = vec![];
    if let Ok(t) = std::fs::read_to_string(format!("{path}.1")) {
        texts.push(t);
    }
    match std::fs::read_to_string(path) {
        Ok(t) => texts.push(t),
        Err(e) => {
            eprintln!("{path}: {e}");
            return 2;
        }
    }
    let (chs, bad, api_only, wake) = rtorder::shadow::chains_lenient(&texts);
    let dir = std::env::temp_dir().join(format!("rtorder_replay_{}", std::process::id()));
    if std::fs::create_dir_all(&dir).is_err() {
        return 2;
    }
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let reports = rt.block_on(rtorder::shadow::replay(&chs, &symbol, &dir, false, None));
    let _ = std::fs::remove_dir_all(&dir);
    println!("rtorder replay · 파일 {}{} · 깨진 줄 {bad} · 접수만 chain {api_only} · 깨움 줄 {wake}", path, if texts.len() > 1 { " (+ .1)" } else { "" });
    print!("{}", rtorder::shadow::report_with(&reports, show));
    if reports.iter().any(|r| !r.truncated && !r.diffs.is_empty()) { 1 } else { 0 }
}
