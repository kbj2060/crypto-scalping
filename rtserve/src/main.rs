//! rtserve 실행 파일. 설정은 환경변수:
//!   RTSERVE_ROOT  저장소 뿌리(필수 · data/live·data/hot·dashboard/live 를 여기서 읽는다)
//!   RTSERVE_ADDR  묶을 주소(기본 127.0.0.1:8790 · 루프백만 · 8787 은 파이썬 서버 몫)
//!   DASHBOARD_ASSETS  원본과 같은 뜻(기본 eth,sol,xrp)

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[tokio::main]
async fn main() {
    let root = match std::env::var("RTSERVE_ROOT") {
        Ok(r) if !r.is_empty() => std::path::PathBuf::from(r),
        _ => {
            eprintln!("RTSERVE_ROOT 가 필요하다(저장소 뿌리)");
            std::process::exit(2);
        }
    };
    let addr: SocketAddr = match std::env::var("RTSERVE_ADDR").unwrap_or_else(|_| "127.0.0.1:8790".into()).parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("RTSERVE_ADDR: {e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = rtserve::check_loopback(&addr) {
        eprintln!("{e}");
        std::process::exit(2);
    }
    let cfg = rtserve::Config {
        root,
        assets: rtserve::parse_assets(std::env::var("DASHBOARD_ASSETS").ok().as_deref()),
        now: Arc::new(|| SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()),
    };
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap_or_else(|e| {
        eprintln!("bind {addr}: {e}");
        std::process::exit(1);
    });
    println!("rtserve http://{addr}/dashboard/live/", );
    tokio::select! {
        r = rtserve::serve(listener, cfg) => if let Err(e) = r { eprintln!("serve: {e}"); std::process::exit(1) },
        _ = tokio::signal::ctrl_c() => {}
    }
}
