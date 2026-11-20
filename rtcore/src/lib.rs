//! rtcore M1: 바이낸스 선물 웹소켓 → 메모리 상태 → 화면 웹소켓 밀기. 읽기 전용(주문·계좌·키 없음).
//! 설계: docs/rtcore_design_20261120.md
pub mod feed;
pub mod footprint;
pub mod hub;
pub mod proto;
pub mod state;
pub mod supply;

use std::sync::atomic::{AtomicI64, AtomicU64};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;

/// 심볼별 최근 체결 링 상한. ETH 평소 ~266건/초 기준 약 8초.
// ponytail: 고정 2048 -- 화면이 더 긴 체결 목록을 원하면 키운다(스냅숏 크기와 맞바꿈).
pub const RING_CAP: usize = 2048;
/// 구독자별 밀기 큐 상한. bookTicker 급증(수천 건/초) 1~3초를 버틴다. 넘치면 그 구독자만 끊는다(hub.rs).
pub const QUEUE_CAP: usize = 16384;

pub static BAD_JSON: AtomicU64 = AtomicU64::new(0);
pub static SKIPPED: AtomicU64 = AtomicU64::new(0);
pub static RECONNECTS: AtomicU64 = AtomicU64::new(0);
/// 지금 붙어 있는 수신 연결 수(60초 로그). 0 이면 수신이 하나도 없다.
pub static LIVE_FEEDS: AtomicI64 = AtomicI64::new(0);
static LATENCY_US: Mutex<Vec<u32>> = Mutex::new(Vec::new());

pub fn now_us() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_micros() as i64)
}

/// 수신→송신(µs). 벽시계가 뒤로 가면 0.
// ponytail: 60초분을 Vec 에 모아 정렬 -- 구독자×건수가 수백만/분이 되면 고정 버킷 히스토그램으로.
pub fn record_latency(us: i64) {
    LATENCY_US.lock().unwrap().push(us.clamp(0, u32::MAX as i64) as u32);
}

/// (건수, 중앙, 상위 1%, 최대) µs. 모은 것을 비운다.
pub fn take_latency() -> Option<(usize, u32, u32, u32)> {
    let mut v = std::mem::take(&mut *LATENCY_US.lock().unwrap());
    if v.is_empty() {
        return None;
    }
    v.sort_unstable();
    let n = v.len();
    Some((n, v[n / 2], v[(n * 99 / 100).min(n - 1)], v[n - 1]))
}

pub struct Cfg {
    pub symbols: Vec<String>,
    pub ws_base: String,
    pub market_ws_base: String,
    /// 나머지 원천(1초 수급). 빈 문자열 = 그 원천을 켜지 않는다(시험·부분 가동).
    pub force_url: String,
    pub spot_ws_base: String,
    pub okx_url: String,
    pub bybit_url: String,
    pub silence: Duration,
}

impl Cfg {
    pub fn from_env() -> Cfg {
        let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
        Cfg {
            symbols: env("RTCORE_SYMBOLS", "ethusdt")
                .split(',')
                .map(|s| s.trim().to_lowercase())
                .filter(|s| !s.is_empty())
                .collect(),
            ws_base: env("RTCORE_WS_BASE", "wss://fstream.binance.com/ws/"),
            market_ws_base: env("RTCORE_MARKET_WS_BASE", "wss://fstream.binance.com/market/ws/"),
            // 주소는 파이썬과 같다: server.py FORCE_ORDER_WS_URL · coin_flow.py _flow_spec spot_url · OKX_WS_URL · Bybit 청산 수집기 WS
            force_url: env("RTCORE_FORCE_WS_URL", "wss://fstream.binance.com/market/ws/!forceOrder@arr"),
            spot_ws_base: env("RTCORE_SPOT_WS_BASE", "wss://stream.binance.com:9443/ws/"),
            okx_url: env("RTCORE_OKX_WS_URL", "wss://ws.okx.com:8443/ws/v5/public"),
            bybit_url: env("RTCORE_BYBIT_WS_URL", "wss://stream.bybit.com/v5/public/linear"),
            silence: Duration::from_secs_f64(env("RTCORE_SILENCE_SECS", "30").parse().unwrap_or(30.0)),
        }
    }
}

/// 수신 연결들과 화면 서버를 띄운다(tokio 런타임 안에서 부른다).
/// 둘째 값은 수신 태스크 하나라도 끝나면(feed::run 은 영원히 돌므로 끝 = 패닉) 그 사유를 내는 핸들이다.
pub fn start(cfg: &Cfg, listener: TcpListener) -> (Arc<state::State>, tokio::task::JoinHandle<String>) {
    let state = state::State::new(&cfg.symbols, RING_CAP, QUEUE_CAP);
    let mut feeds = Vec::new();
    for s in &cfg.symbols {
        for url in [
            format!("{}{s}@trade", cfg.ws_base),
            format!("{}{s}@bookTicker", cfg.ws_base),
            format!("{}{s}@markPrice@1s", cfg.market_ws_base),
        ] {
            feeds.push((url.clone(), tokio::spawn(feed::run(url, state.clone(), cfg.silence))));
        }
    }
    for src in sources(cfg) {
        feeds.push((src.url.clone(), tokio::spawn(feed::run_src(src, state.clone(), cfg.silence))));
    }
    tokio::spawn(hub::serve(listener, state.clone()));
    let st = state.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_millis(state::FLUSH_MS));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay); // 밀렸으면 몰아서 여러 번 돌지 않는다
        loop {
            tick.tick().await;
            st.flush();
        }
    });
    (state, tokio::spawn(first_exit(feeds)))
}

/// 1초 수급의 나머지 원천 연결: 강제청산 1(전 종목) · 현물 종목마다 1 · OKX 1 · Bybit 1.
pub fn sources(cfg: &Cfg) -> Vec<feed::Src> {
    use feed::{Kind, Src};
    use tokio_tungstenite::tungstenite::Message;
    let frame = || Some(Message::Ping(Vec::new().into()));
    let mut out = vec![];
    if !cfg.force_url.is_empty() {
        out.push(Src { url: cfg.force_url.clone(), kind: Kind::Futures, sub: None, ping: frame(), syms: vec![] });
    }
    if !cfg.spot_ws_base.is_empty() {
        for s in cfg.symbols.iter().filter(|s| !supply::NO_SPOT.contains(&s.as_str())) {
            let url = format!("{}{s}@aggTrade", cfg.spot_ws_base);
            out.push(Src { url, kind: Kind::Spot, sub: None, ping: frame(), syms: vec![s.clone()] });
        }
    }
    let okx: Vec<&String> = cfg.symbols.iter().filter(|s| supply::okx_ct_val(&supply::okx_inst(s)).is_some()).collect();
    if !cfg.okx_url.is_empty() && !okx.is_empty() {
        let mut args: Vec<serde_json::Value> = okx
            .iter()
            .flat_map(|s| {
                let inst = supply::okx_inst(s);
                [serde_json::json!({"channel": "trades", "instId": inst}), serde_json::json!({"channel": "open-interest", "instId": inst})]
            })
            .collect();
        args.push(serde_json::json!({"channel": "liquidation-orders", "instType": "SWAP"}));
        let sub = serde_json::json!({"op": "subscribe", "args": args}).to_string();
        out.push(Src { url: cfg.okx_url.clone(), kind: Kind::Okx, sub: Some(sub), ping: Some(Message::text("ping")), syms: okx.into_iter().cloned().collect() });
    }
    let bybit: Vec<String> =
        cfg.symbols.iter().filter(|s| supply::BYBIT_SYMBOLS.contains(&s.as_str())).map(|s| format!("allLiquidation.{}", s.to_uppercase())).collect();
    if !cfg.bybit_url.is_empty() && !bybit.is_empty() {
        let sub = serde_json::json!({"op": "subscribe", "args": bybit}).to_string();
        out.push(Src { url: cfg.bybit_url.clone(), kind: Kind::Bybit, sub: Some(sub), ping: Some(Message::text(r#"{"op":"ping"}"#)), syms: vec![] });
    }
    out
}

/// 태스크들 중 처음 끝난 것의 사유. 2026-11-20 실서버: wss 패닉으로 수신 태스크가 전부 죽었는데
/// 프로세스는 살아 «n=0» 만 찍었다 -- 이 사유를 받은 main 이 0 아닌 코드로 끝낸다.
pub async fn first_exit(tasks: Vec<(String, tokio::task::JoinHandle<()>)>) -> String {
    let (names, hs): (Vec<_>, Vec<_>) = tasks.into_iter().unzip();
    let (r, i, _) = futures_util::future::select_all(hs).await;
    match r {
        Err(e) if e.is_panic() => format!("수신 태스크 패닉 {}", names[i]),
        _ => format!("수신 태스크 종료 {}", names[i]),
    }
}
