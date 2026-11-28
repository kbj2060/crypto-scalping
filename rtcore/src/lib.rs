//! rtcore M1: 바이낸스 선물 웹소켓 → 메모리 상태 → 화면 웹소켓 밀기. 읽기 전용(주문·계좌·키 없음).
//! 설계: docs/rtcore_design_20261120.md
pub mod bnoi;
pub mod feed;
pub mod footprint;
pub mod hub;
pub mod proto;
pub mod quote;
pub mod restore;
pub mod situation;
pub mod state;
pub mod supply;
pub mod tick;

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
    /// 2026-11-21 호가·1분봉만 받는 종목(주문 심볼 USDC -- rtorder Desk::exec_symbol 중 USDT 가 아닌 것). 체결·마크·풋프린트는 안 받는다.
    pub quote_symbols: Vec<String>,
    pub ws_base: String,
    pub market_ws_base: String,
    /// 나머지 원천(1초 수급). 빈 문자열 = 그 원천을 켜지 않는다(시험·부분 가동).
    pub force_url: String,
    pub spot_ws_base: String,
    pub okx_url: String,
    pub bybit_url: String,
    pub silence: Duration,
    /// 화면 문 단속(hub.rs 머리 주석).
    pub hub: hub::HubCfg,
    /// 재시작 복원(restore.rs): 코어 스냅숏 폴더(빈 문자열 = 복원 끔) · 체결 테이프 hot SQLite(읽기 전용).
    /// 상대 경로는 실행 위치 기준 -- 감시자(scripts/ops/supervisor_rtcore.sh)가 저장소 루트에서 띄운다.
    pub snap_dir: String,
    pub tape_db: String,
    /// 나머지 원천 복원(restore.rs «나머지 원천»): OKX 체결 테이프 · OKX 맥락(OI·청산) · 바이낸스 맥락(강제청산) · Bybit 청산.
    /// 기본 = 파이썬 수집기 파일. Rust 수집기 그림자로 갈아 끼울 때 data/hot/rtcollect/<같은 이름> 으로(rtcollect/SHADOW.md).
    pub okx_tape_db: String,
    pub okx_ctx_db: String,
    pub bn_ctx_db: String,
    pub bybit_liq_db: String,
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
            quote_symbols: env("RTCORE_QUOTE_SYMBOLS", "ethusdc,solusdc,xrpusdc,btcusdc")
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
            hub: hub::HubCfg::from_env(),
            snap_dir: env("RTCORE_SNAP_DIR", "data/live/rtcore"),
            // 파이썬 수집기 파일(live_trade_tape_collector HOT_TAPE_DB) -- Rust 수집기로 갈아 끼우면 그 파일로 바꾼다.
            tape_db: env("RTCORE_TAPE_DB", "data/hot/binance_tape.sqlite"),
            okx_tape_db: env("RTCORE_OKX_TAPE_DB", "data/hot/okx_tape.sqlite"),
            okx_ctx_db: env("RTCORE_OKX_CTX_DB", "data/hot/okx_ctx.sqlite"),
            bn_ctx_db: env("RTCORE_BN_CTX_DB", "data/hot/binance_ctx.sqlite"),
            bybit_liq_db: env("RTCORE_BYBIT_LIQ_DB", "data/hot/bybit_liq.sqlite"),
        }
    }
}

/// 수신 연결들과 화면 서버를 띄운다(tokio 런타임 안에서 부른다).
/// 둘째 값은 수신 태스크 하나라도 끝나면(feed::run 은 영원히 돌므로 끝 = 패닉) 그 사유를 내는 핸들이다.
/// 마지막 체결가(«모든 종목» Prices) 밀기 간격(ms). 원본 `/api/events` 는 1초 -- 화면 전광판이 1초마다 다시 그리므로 0.1초면 충분하다.
pub const PRICES_MS: u64 = 100;

pub fn start(cfg: &Cfg, listener: TcpListener) -> (Arc<state::State>, tokio::task::JoinHandle<String>) {
    let mut all = cfg.symbols.clone();
    all.extend(cfg.quote_symbols.iter().filter(|q| !cfg.symbols.contains(q)).cloned());
    let state = if cfg.snap_dir.is_empty() {
        state::State::new(&all, RING_CAP, QUEUE_CAP)
    } else {
        // 순서: 스냅숏을 읽은 뒤(수신 전) 수신을 붙인다. 테이프 메우기는 각 연결의 첫 체결 뒤에 스레드에서(restore.rs 머리 주석 2).
        let rc = restore::Cfg {
            snap_dir: cfg.snap_dir.clone().into(),
            tape: cfg.tape_db.clone().into(),
            tries: 12,
            wait: Duration::from_secs(5),
            okx_tape: cfg.okx_tape_db.clone().into(),
            okx_ctx: cfg.okx_ctx_db.clone().into(),
            bn_ctx: cfg.bn_ctx_db.clone().into(),
            bybit_liq: cfg.bybit_liq_db.clone().into(),
        };
        let (st, mut wake) = state::State::with_restore(&all, RING_CAP, QUEUE_CAP, rc, now_us());
        let s2 = st.clone();
        tokio::spawn(async move {
            while let Some((sym, w)) = wake.recv().await {
                let s3 = s2.clone();
                // 심볼·일감마다 따로(기다림 최대 60초가 서로 안 막게)
                tokio::task::spawn_blocking(move || match w {
                    restore::Wake::Bn => s3.run_restore(&sym),
                    restore::Wake::Okx => {
                        let line = s3.run_restore_okx(&sym);
                        // 풋프린트 봉 복원: 15초 뒤, 그다음 5분마다 최대 8번(원본 okx_footprint_restore · 이 스레드가 35분 산다)
                        for i in 0..restore::OKX_FP_ATTEMPTS {
                            std::thread::sleep(if i == 0 { restore::OKX_FP_FIRST } else { restore::OKX_FP_NEXT });
                            if s3.run_restore_okx_fp(&sym, now_us()).is_none_or(|(_, done)| done) {
                                break;
                            }
                        }
                        line
                    }
                    restore::Wake::OkxOi => s3.run_restore_okx_oi(&sym),
                });
            }
        });
        // 원본 fp_dhl 테이프 가지(요청마다 · 화면 폴링 400ms) -- 코어는 5초마다(봉 dhl 이 그만큼 늦게 뜰 수 있다)
        let s2 = st.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(5));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tick.tick().await;
                let s3 = s2.clone();
                let _ = tokio::task::spawn_blocking(move || s3.retry_dhl(now_us())).await;
            }
        });
        let s2 = st.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(restore::SNAP_EVERY);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            tick.tick().await;
            loop {
                tick.tick().await;
                let s3 = s2.clone();
                let _ = tokio::task::spawn_blocking(move || s3.write_snapshots()).await;
            }
        });
        st
    };
    let mut feeds = Vec::new();
    for s in &cfg.symbols {
        for url in [
            format!("{}{s}@trade", cfg.ws_base),
            format!("{}{s}@bookTicker", cfg.ws_base),
            format!("{}{s}@markPrice@1s", cfg.market_ws_base),
            // 2026-11-21 마감 1분봉(1분 변동성·5분봉 종가 -- quote.rs). kline 은 market 경로(markPrice 와 같은 곳 -- 바이낸스 경로 분리 규칙, 실서버 대조 전)
            format!("{}{s}@kline_1m", cfg.market_ws_base),
        ] {
            feeds.push((url.clone(), tokio::spawn(feed::run(url, state.clone(), cfg.silence))));
        }
    }
    // 주문 심볼(USDC): 호가·1분봉만
    for s in cfg.quote_symbols.iter().filter(|q| !cfg.symbols.contains(q)) {
        for url in [format!("{}{s}@bookTicker", cfg.ws_base), format!("{}{s}@kline_1m", cfg.market_ws_base)] {
            feeds.push((url.clone(), tokio::spawn(feed::run(url, state.clone(), cfg.silence))));
        }
    }
    let st = state.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_millis(PRICES_MS));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            st.flush_prices();
        }
    });
    for src in sources(cfg) {
        feeds.push((src.url.clone(), tokio::spawn(feed::run_src(src, state.clone(), cfg.silence))));
    }
    tokio::spawn(hub::serve(listener, state.clone(), cfg.hub.clone()));
    // 2026-11-28 바이낸스 OI 1초(bnoi.rs · hot 읽기 전용) -- 빈 경로 = 끔
    if !cfg.bn_ctx_db.is_empty() {
        tokio::spawn(bnoi::run(state.clone(), cfg.bn_ctx_db.clone().into(), cfg.symbols.clone()));
    }
    let st = state.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(1)); // 원본 micro-ref 1초 루프마다(SITUATION_EVERY_TICKS = 1)
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            st.situation_tick(now_us());
        }
    });
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
