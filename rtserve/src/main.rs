//! rtserve 실행 파일. 설정은 환경변수:
//!   RTSERVE_ROOT  저장소 뿌리(필수 · data/live·data/hot·dashboard/live 를 여기서 읽는다)
//!   RTSERVE_ADDR  묶을 주소(기본 127.0.0.1:8790 · 루프백만 · 8787 은 파이썬 서버 몫)
//!   DASHBOARD_ASSETS  원본과 같은 뜻(기본 eth,btc,sol,xrp,sndk,soxl,skhynix,mu)
//!   RTSERVE_RTORDER  rtorder 문(기본 127.0.0.1:8791 · 루프백만 · 주문·계좌 14개를 넘긴다)
//!   RTSERVE_HUB   rtcore 허브 주소(예 ws://127.0.0.1:8788/rt · 없으면 메모리 상태 엔드포인트는 빈 모양)
//!   VAPID_PUBLIC_KEY · VAPID_PRIVATE_KEY  알림 설정(/api/push/config · 원본과 같은 이름)
//!   RTSERVE_MODELS  모델 JSON 자리(기본 data/live/rtmodel · rtmodel/tools/export_*.py) · RTSERVE_REST 바이낸스 선물 REST 뿌리(기본 원본과 같은 곳) · FRED_API_KEY(원본과 같은 이름)

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
    let mut cfg = rtserve::Config::new(
        root,
        rtserve::parse_assets(std::env::var("DASHBOARD_ASSETS").ok().as_deref()),
        Arc::new(|| SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()),
    );
    // 알림 설정: 공개키 값과 «비밀키가 있는지»만(원본 /api/push/config 와 같은 뜻 · 비밀키 값은 들고 있지 않는다)
    cfg.vapid_public = std::env::var("VAPID_PUBLIC_KEY").unwrap_or_default();
    cfg.vapid_private_set = std::env::var("VAPID_PRIVATE_KEY").is_ok_and(|v| !v.is_empty());
    // rtorder 문(주문·계좌 14개) -- 기본 127.0.0.1:8791 · 루프백만
    match std::env::var("RTSERVE_RTORDER").unwrap_or_else(|_| "127.0.0.1:8791".into()).parse::<SocketAddr>() {
        Ok(a) if a.ip().is_loopback() => cfg.rtorder = Some(a),
        _ => {
            eprintln!("RTSERVE_RTORDER: 루프백 주소만");
            std::process::exit(2);
        }
    }
    // 모델 핸들러(model.rs): 모델 JSON 자리 · REST 뿌리(시험·그림자용 · 기본 원본 FAPI) · FRED 키(원본과 같은 이름 · 값은 들고만 있고 찍지 않는다)
    cfg.models = std::env::var("RTSERVE_MODELS").ok().map(std::path::PathBuf::from);
    if let Ok(r) = std::env::var("RTSERVE_REST") {
        cfg.rest = r;
    }
    cfg.fred_key = std::env::var("FRED_API_KEY").ok().filter(|k| !k.is_empty());
    let cfg = Arc::new(cfg);
    // rtcore 허브(예 ws://127.0.0.1:8788/rt) -- 없으면 메모리 상태 엔드포인트는 «아직 없음» 모양만 낸다
    if let Ok(url) = std::env::var("RTSERVE_HUB") {
        let syms: Vec<String> = cfg.flows().iter().map(|a| format!("{a}usdt")).collect();
        tokio::spawn(rtserve::hub::subscribe(url, syms, cfg.live.clone()));
    }
    tokio::spawn(cfg.events.clone().run(cfg.clone(), cfg.live.clone()));
    let cfg = (*cfg).clone();
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
