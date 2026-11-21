//! rtduck: DuckDB 로 쓰는 수집기. RTDUCK_KIND = oi_lsratio | hl_trades | hl_positions.
//!   hl_positions HL_POS_COINS(기본 ETH -- 원본과 같다 · HL_POS_COIN 도) · HL_ROOT(체결 DuckDB 를 읽기만) · HL_POS_DB(기본 data/hot/rtduck/<원본과 같은 이름> --
//!               data/hot/<그 이름> 은 RTDUCK_TAKEOVER=1 일 때만) · 🔴HL info REST 전용이라 RTDUCK_REST=1 없으면 아무것도 안 한다
//!   hl_trades   HL_COINS(기본 ETH) · HL_ROOT(기본 data/live/rtduck/orderflow/hyperliquid -- 원본 폴더 data/live/orderflow/hyperliquid 는 RTDUCK_TAKEOVER=1 일 때만)
//!   oi_lsratio  BOT_SYMBOLS(기본 ETHUSDT,BTCUSDT,SOLUSDT -- 원본 기본값) · QUANT_OI_LSRATIO_DB_PATH(기본 data/live/rtduck/oi_lsratio.duckdb --
//!               원본 파일 data/live/oi_lsratio.duckdb 는 RTDUCK_TAKEOVER=1 일 때만) · OI_ENABLE · OI_POLL_INTERVAL_SEC · OI_FETCH_LIMIT(원본과 같다)
//!   🔴REST 는 RTDUCK_REST=1 일 때만 보낸다(기본 꺼짐 -- 꺼져 있으면 원천 다섯이 모두 실패한 것과 같아 아무것도 안 쓴다).
//!   🔧밴 파일(RTCOLLECT_BAN_FILE 기본 data/live/binance_ban_until)을 지킨다: 밴 중이면 안 보내고 418·429 는 기록(원본은 모른다).
//!   RTDUCK_ONCE=1 = 기다리지 않고 한 바퀴만 돌고 끝(시험·손 확인용). RTDUCK_OI_BASE = REST 주소(시험 = 127.0.0.1 가짜 서버).
use rtduck::oi;
use std::path::PathBuf;
use std::time::Duration;

fn env_s(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

fn die(code: i32, msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(code)
}

fn main() {
    match env_s("RTDUCK_KIND", "oi_lsratio").as_str() {
        "oi_lsratio" => run_oi(),
        "hl_trades" => run_hl_trades(),
        "hl_positions" => run_hl_positions(),
        k => die(2, &format!("모르는 종류 {k:?} -- oi_lsratio|hl_trades|hl_positions")),
    }
}

fn run_oi() {
    let syms: Vec<String> = env_s("BOT_SYMBOLS", "ETHUSDT,BTCUSDT,SOLUSDT").split(',').map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()).collect();
    let db = PathBuf::from(env_s("QUANT_OI_LSRATIO_DB_PATH", "data/live/rtduck/oi_lsratio.duckdb"));
    if db.ends_with("data/live/oi_lsratio.duckdb") && env_s("RTDUCK_TAKEOVER", "") != "1" {
        die(2, &format!("🔴{} 는 파이썬 워커의 파일이다 -- 그림자는 다른 경로에. 전환이면 RTDUCK_TAKEOVER=1", db.display()));
    }
    if !["1", "true", "yes", "on"].contains(&env_s("OI_ENABLE", "true").trim().to_lowercase().as_str()) {
        println!("OiLsRatioCollector disabled (OI_ENABLE=false)");
        loop {
            std::thread::sleep(Duration::from_secs(3600)); // 원본 워커도 꺼진 수집기만 두고 계속 산다
        }
    }
    let interval: f64 = env_s("OI_POLL_INTERVAL_SEC", "300").parse().unwrap_or_else(|_| die(2, "OI_POLL_INTERVAL_SEC 는 수"));
    let limit: i64 = env_s("OI_FETCH_LIMIT", "12").parse().unwrap_or_else(|_| die(2, "OI_FETCH_LIMIT 는 정수"));
    let base = env_s("RTDUCK_OI_BASE", oi::BASE);
    let rest = env_s("RTDUCK_REST", "") == "1";
    let once = env_s("RTDUCK_ONCE", "") == "1";
    let ban = PathBuf::from(env_s("RTCOLLECT_BAN_FILE", "data/live/binance_ban_until"));
    let agent = rtcollect::rest::agent(Duration::from_secs(15)); // urlopen(timeout=15)
    for s in &syms {
        match oi::init(&db, &oi::table(s)) {
            Ok(Some(t)) => println!("oi_lsratio({}) bootstrap: resuming after ts={t}µs", s.to_uppercase()),
            Ok(None) => println!("oi_lsratio({}) bootstrap: empty table, starting fresh", s.to_uppercase()),
            Err(e) => die(1, &format!("oi_lsratio({}) DB 준비 실패: {e}", s.to_uppercase())),
        }
    }
    println!("rtduck oi_lsratio {} · db {} · REST {}", syms.join(","), db.display(), if rest { "켬" } else { "🔴꺼짐(아무것도 안 쓴다)" });
    let mut get = |u: &str| -> Option<serde_json::Value> {
        if !rest {
            return None;
        }
        let now = rtcollect::rest::wall_s();
        if rtcollect::rest::ban_left(&ban, now) > 0.0 {
            eprintln!("binance IP ban -- 안 보냄: {u}");
            return None;
        }
        let mut r = match agent.get(u).header("User-Agent", oi::UA).call() {
            Ok(r) => r,
            Err(e) => {
                eprintln!("fetch failed {u}: {e}");
                return None;
            }
        };
        let st = r.status().as_u16();
        let ra = r.headers().get("retry-after").and_then(|v| v.to_str().ok()).map(str::to_string);
        let body = r.body_mut().read_to_string().unwrap_or_default();
        rtcollect::rest::ban_note(&ban, st, &body, ra.as_deref(), now);
        if st >= 400 {
            eprintln!("fetch failed {u}: HTTP {st}");
            return None;
        }
        serde_json::from_str(&body).map_err(|e| eprintln!("fetch failed {u}: {e}")).ok()
    };
    loop {
        if !once {
            // ponytail: 종목들을 한 스레드에서 차례로 돈다(원본은 종목마다 태스크 -- 같은 경계에 깨어 나란히). 요청 20개 남짓이라 몇 초 차이.
            std::thread::sleep(Duration::from_secs_f64(oi::sleep_s(rtcollect::rest::wall_s(), interval)));
        }
        for s in &syms {
            match oi::poll(s, limit, &base, &mut get) {
                Ok((rows, ok)) => {
                    let now_us = (rtcollect::rest::wall_s() * 1e6).round() as i64;
                    if let Err(e) = oi::upsert(&db, &oi::table(s), s, &rows, now_us) {
                        eprintln!("oi_lsratio({}) DB upsert error: {e}", s.to_uppercase());
                    }
                    println!("oi_lsratio({}) poll: sources_ok={ok}/5 upserted_rows={}", s.to_uppercase(), rows.len());
                }
                // 원본: 폴 루프 except → 15초 쉬고 다음 경계(그 폴은 버린다) -- 한 스레드라 다음 경계 잠이 그 몫을 한다
                Err(e) => eprintln!("oi_lsratio({}) poll loop error: {e}", s.to_uppercase()),
            }
        }
        if once {
            return;
        }
    }
}

/// HL 주소단위 체결(live_hyperliquid_trade_collector_20260916.py run()). SIGTERM = 남은 버퍼를 쓰고 끝(원본은 그냥 죽어 버퍼를 잃는다 -- 더 안전한 쪽).
fn run_hl_trades() {
    use std::cell::RefCell;
    use std::rc::Rc;
    let coins: Vec<String> = env_s("HL_COINS", "ETH").split(',').map(|c| c.trim().to_uppercase()).filter(|c| !c.is_empty()).collect();
    let root = PathBuf::from(env_s("HL_ROOT", "data/live/rtduck/orderflow/hyperliquid"));
    if root.ends_with("data/live/orderflow/hyperliquid") && env_s("RTDUCK_TAKEOVER", "") != "1" {
        die(2, &format!("🔴{} 는 파이썬 수집기의 폴더다 -- 그림자는 다른 경로에. 전환이면 RTDUCK_TAKEOVER=1", root.display()));
    }
    let url = env_s("RTDUCK_WS_URL", rtduck::hl_trades::WS);
    let h = Rc::new(RefCell::new(rtduck::hl_trades::HlTrades::new(&root, &coins, rtduck::hl_trades::wall())));
    println!("rtduck hl_trades {coins:?} → {}", h.borrow().db_path(rtduck::hl_trades::wall()).display());
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    tokio::task::LocalSet::new().block_on(&rt, async {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).unwrap();
        tokio::select! {
            _ = rtcollect::net::run_hl(h.clone(), url, || {}) => {}
            _ = term.recv() => {}
            _ = tokio::signal::ctrl_c() => {}
        }
    });
    let mut h = h.borrow_mut();
    let b = std::mem::take(&mut h.buf);
    if let Err(e) = h.write(&b, None, rtduck::hl_trades::wall()) {
        eprintln!("종료 때 {}행 쓰기 실패: {e}", b.len());
    }
    println!("종료 신호 -- 남은 {}행 쓰고 끝낸다", b.len());
}

/// 진짜 바깥: HL info POST(시한 15초 · aiohttp ClientTimeout(total=15)) · 벽시계 · 잠.
type Post = Box<dyn FnMut(&str) -> Result<(u16, serde_json::Value), String>>;

struct RealEnv {
    post: Post,
}

impl rtduck::hl_positions::Env for RealEnv {
    fn post(&mut self, body: &str) -> Result<(u16, serde_json::Value), String> {
        (self.post)(body)
    }
    fn now(&mut self) -> f64 {
        rtcollect::rest::wall_s()
    }
    fn sleep(&mut self, s: f64) {
        std::thread::sleep(Duration::from_secs_f64(s.max(0.0)));
    }
}

/// HL 고래 포지션(live_hyperliquid_positions_collector_20260924.py run()). 바퀴를 쉬지 않고 돈다(원본 그대로 -- 간격은 요청 사이 0.4초).
fn run_hl_positions() {
    let coins: Vec<String> = std::env::var("HL_POS_COINS").or_else(|_| std::env::var("HL_POS_COIN")).unwrap_or_else(|_| "ETH".into())
        .split(',').map(|c| c.trim().to_uppercase()).filter(|c| !c.is_empty()).collect();
    let name = rtduck::hl_positions::default_db(&coins);
    let db = PathBuf::from(env_s("HL_POS_DB", &format!("data/hot/rtduck/{name}")));
    if db.ends_with(format!("data/hot/{name}")) && env_s("RTDUCK_TAKEOVER", "") != "1" {
        die(2, &format!("🔴{} 는 파이썬 수집기의 파일이다 -- 그림자는 다른 경로에. 전환이면 RTDUCK_TAKEOVER=1", db.display()));
    }
    let trades = PathBuf::from(env_s("HL_ROOT", "data/live/orderflow/hyperliquid")); // 읽기만(파이썬 체결 수집기 파일도 그대로 읽어도 된다)
    if env_s("RTDUCK_REST", "") != "1" {
        println!("hl_positions {coins:?}: 🔴RTDUCK_REST=1 이 아니라 아무것도 안 한다(HL info 요청이 파이썬과 합쳐 두 배가 된다 -- SHADOW.md §12)");
        loop {
            std::thread::sleep(Duration::from_secs(3600));
        }
    }
    let mut c = rtduck::hl_positions::Collector::new(&db, &coins, &trades).unwrap_or_else(|e| die(1, &format!("DB 준비 실패: {e}")));
    let (agent, url) = (rtcollect::rest::agent(Duration::from_secs(15)), env_s("RTDUCK_HL_INFO_URL", rtduck::hl_positions::INFO_URL));
    let mut env = RealEnv { post: Box::new(move |body: &str| {
        let mut r = agent.post(&url).header("Content-Type", "application/json").send(body).map_err(|e| format!("ClientConnectorError: {e}"))?;
        let st = r.status().as_u16();
        let text = r.body_mut().read_to_string().map_err(|e| format!("ClientPayloadError: {e}"))?;
        if st >= 400 {
            return Ok((st, serde_json::Value::Null));
        }
        serde_json::from_str(&text).map(|v| (st, v)).map_err(|e| format!("ContentTypeError: {e}"))
    }) };
    println!("{} 포지션 수집 시작 (코인별 상위 {}주소 · {}초 간격 · db {} · 체결 {})", coins.join(","), c.universe_n, rtduck::hl_positions::REQ_GAP_S, db.display(), trades.display());
    loop {
        c.cycle(&mut env);
    }
}
