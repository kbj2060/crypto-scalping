//! ⚠️기본 주소는 실제 바이낸스다. 개발기에서 실행하지 않는다(서버와 IP 한도 공유 -- 밴 위험). 서버 전용.
use rtcore::{take_latency, Cfg, BAD_JSON, LIVE_FEEDS, RECONNECTS, SKIPPED};
use std::sync::atomic::Ordering::Relaxed;
use std::time::Duration;

#[tokio::main]
async fn main() {
    let cfg = Cfg::from_env();
    let listen = std::env::var("RTCORE_LISTEN").unwrap_or_else(|_| "127.0.0.1:8788".into());
    let listener = tokio::net::TcpListener::bind(&listen).await.expect("bind");
    println!("rtcore 시작 -- {listen} · 심볼 {:?} · 침묵 {:?}", cfg.symbols, cfg.silence);
    let (_state, mut dead) = rtcore::start(&cfg, listener);
    let mut tick = tokio::time::interval(Duration::from_secs(60));
    tick.tick().await;
    loop {
        tokio::select! {
            _ = tick.tick() => {}
            why = &mut dead => {
                // 다시 띄우지 않고 끝낸다: 패닉은 대개 같은 자리에서 또 난다(이번 rustls 처럼) -- 감시자가 재기동·알림한다.
                println!("rtcore 종료: {} -- 감시자가 다시 띄운다", why.unwrap_or_else(|e| format!("감시 태스크 {e}")));
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
