//! ⚠️기본 주소는 실제 바이낸스다. 개발기에서 실행하지 않는다(서버와 IP 한도 공유 -- 밴 위험). 서버 전용.
use rtcore::{take_latency, Cfg, BAD_JSON, LIVE_FEEDS, RECONNECTS, SKIPPED};
use std::sync::atomic::Ordering::Relaxed;
use std::time::Duration;

#[tokio::main]
async fn main() {
    let cfg = Cfg::from_env();
    let listen = std::env::var("RTCORE_LISTEN").unwrap_or_else(|_| "127.0.0.1:8788".into());
    let listener = tokio::net::TcpListener::bind(&listen).await.expect("bind");
    println!("rtcore 시작 -- {listen} · 심볼 {:?} · 침묵 {:?} · 문 {:?}", cfg.symbols, cfg.silence, cfg.hub);
    let (state, mut dead) = rtcore::start(&cfg, listener);
    let mut tick = tokio::time::interval(Duration::from_secs(60));
    tick.tick().await;
    use tokio::signal::unix::{signal, SignalKind};
    let (mut term, mut int) = (signal(SignalKind::terminate()).expect("SIGTERM"), signal(SignalKind::interrupt()).expect("SIGINT"));
    loop {
        tokio::select! {
            _ = tick.tick() => {}
            // 감시자(scripts/ops/supervisor_rtcore.sh)는 SIGTERM 으로 내린다 -- 스냅숏을 한 번 쓰고 끝낸다(메우는 중이면 쓰지 않는 조건은 그대로 · restore.rs).
            // 다음 판이 테이프로 메울 공백이 «내려가 있던 시간»으로 준다(coin_flow.py stop 1029~1031 과 같은 뜻).
            sig = async { tokio::select! { _ = term.recv() => "SIGTERM", _ = int.recv() => "SIGINT" } } => {
                let st = state.clone();
                let n = tokio::task::spawn_blocking(move || st.write_snapshots()).await.unwrap_or(0);
                println!("rtcore {sig} -- 스냅숏 {n}개 쓰고 끝낸다");
                std::process::exit(0);
            }
            why = &mut dead => {
                // 다시 띄우지 않고 끝낸다: 패닉은 대개 같은 자리에서 또 난다(이번 rustls 처럼) -- 감시자가 재기동·알림한다.
                println!("rtcore 종료: {} -- 감시자가 다시 띄운다", why.unwrap_or_else(|e| format!("감시 태스크 {e}")));
                // 다음 판이 메울 공백을 «재시작에 걸린 시간»으로 줄인다(coin_flow.py stop 1029~1031 과 같은 뜻 · 메우는 중이면 안 쓴다).
                println!("rtcore 종료 전 스냅숏 {}개", state.write_snapshots());
                std::process::exit(70);
            }
        }
        let lat = match take_latency() {
            Some((n, p50, p99, max)) => format!("n={n} 중앙={p50}us 상위1%={p99}us 최대={max}us"),
            None => "n=0".into(),
        };
        println!(
            "rtcore 60s 수신->송신 {lat} · 살아있는수신={} · 깨진JSON={} 건너뜀={} 재연결={} (누적)",
            LIVE_FEEDS.load(Relaxed),
            BAD_JSON.load(Relaxed),
            SKIPPED.load(Relaxed),
            RECONNECTS.load(Relaxed)
        );
    }
}
