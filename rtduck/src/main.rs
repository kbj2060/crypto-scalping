//! rtduck: DuckDB 로 쓰는 수집기. RTDUCK_KIND = oi_lsratio | hl_trades | hl_positions | market_data.
//!   market_data duckdb_persist_worker.py 서버 형태의 화면·감시 몫: TAIL_SYMBOLS(기본 ETHUSDT -- 서버 감독과 같다) 청산 요격기 → QUANT_TAIL_DB_PATH(기본 data/live/rtduck/tail_risk.duckdb)
//!               + liq_burst_state.json · WRITE_DASHBOARD_STATE(기본 true) → dashboard_state.json (둘 다 RTDUCK_LIVE_DIR 기본 data/live/rtduck -- data/live 는 RTDUCK_TAKEOVER=1 일 때만)
//!               + 미시구조 COLLECT_MICROSTRUCTURE(기본 true) · BOT_SYMBOLS(기본 ETHUSDT,BTCUSDT,SOLUSDT) → QUANT_MICRO_DB_PATH(기본 data/live/rtduck/microstructure.duckdb)
//!               🔴미시구조의 REST(openInterest·premiumIndex 10초 · aggTrades 대체)는 RTDUCK_REST=1 일 때만 -- 꺼지면 poll_connected=false(data_stale)
//!   hl_positions HL_POS_COINS(기본 ETH -- 원본과 같다 · HL_POS_COIN 도) · HL_ROOT(체결 DuckDB 를 읽기만) · HL_POS_DB(기본 data/hot/rtduck/<원본과 같은 이름> --
//!               data/hot/<그 이름> 은 RTDUCK_TAKEOVER=1 일 때만) · 🔴HL info REST 전용이라 RTDUCK_REST=1 없으면 아무것도 안 한다
//!   hl_trades   HL_COINS(기본 ETH) · HL_ROOT(기본 data/live/rtduck/orderflow/hyperliquid -- 원본 폴더 data/live/orderflow/hyperliquid 는 RTDUCK_TAKEOVER=1 일 때만)
//!   oi_lsratio  BOT_SYMBOLS(기본 ETHUSDT,BTCUSDT,SOLUSDT -- 원본 기본값) · QUANT_OI_LSRATIO_DB_PATH(기본 data/live/rtduck/oi_lsratio.duckdb --
//!               원본 파일 data/live/oi_lsratio.duckdb 는 RTDUCK_TAKEOVER=1 일 때만) · OI_ENABLE · OI_POLL_INTERVAL_SEC · OI_FETCH_LIMIT(원본과 같다)
//!   🔴REST 는 RTDUCK_REST=1 일 때만 보낸다(기본 꺼짐 -- 꺼져 있으면 원천 다섯이 모두 실패한 것과 같아 아무것도 안 쓴다).
//!   🔧밴 파일(RTCOLLECT_BAN_FILE 기본 data/live/binance_ban_until)을 지킨다: 밴 중이면 안 보내고 418·429 는 기록(원본은 모른다).
//!   raster_archive 래스터 14일 보관(cron 매시 · 한 번 돌고 끝 -- 원본은 래스터 수집기 안에서 매시) · OF_ROOT(기본 data/live/rtcollect/orderflow/raster = Rust 래스터 그림자)
//!               · OF_ARCHIVE_ROOT(기본 data/live/rtduck/orderflow/raster_archive) · OF_RETENTION_DAYS(14) · OF_ARCHIVE(1 -- 0 이면 지우기만) · OF_SYMBOLS(쉼표 · 기본 OF_ROOT 아래 폴더 전부)
//!               🔴원본 폴더(data/live/orderflow/raster · …/raster_archive)는 RTDUCK_TAKEOVER=1 일 때만 -- 파이썬 래스터의 파일을 지운다
//!   liq_tab     풋프린트 «청산» 탭(cron 매시 · 한 번) · LIQ_TAB_LIVE(기본 data/live/rtduck -- data/live 는 RTDUCK_TAKEOVER=1 일 때만) · LIQ_TAB_LAKE(data/lake 읽기만) ·
//!               LIQ_TAB_HOT(data/hot 읽기만) · `--coin <코인>` · 🔴klines REST 는 RTDUCK_REST=1 일 때만(꺼지면 원본 LIQ_TAB_NO_REST 처럼 저장된 봉만)
//!   seal        lake 봉인(seal_to_lake -- cron 매일 02:05 UTC · 한 번) · SEAL_ROOT(원천 기준 폴더 · 기본 .) · SEAL_LAKE(기본 data/rtduck/lake -- data/lake 는 RTDUCK_TAKEOVER=1 일 때만) ·
//!               SEAL_STATUS(기본 data/live/rtduck/seal_status.json) · `--dry-run` · `--only venue/stream` · 🔴hot 정리(원천 행 지우기)는 RTDUCK_TAKEOVER=1 일 때만(그림자 lake 를 근거로 파이썬 hot 을 지우면 안 된다)
//!   deribit     Deribit 옵션 체결·블록(상주 · 첫 단계 -- 체인·요약·대조는 아직) · DERIBIT_BT_DIR(기본 data/live/rtduck -- data/live 는 RTDUCK_TAKEOVER=1 일 때만):
//!               deribit_options.duckdb(option_trades·gaps·…) · deribit_block_trades_state.json · 🔴재기동 틈 REST 백필은 RTDUCK_REST=1 일 때만 ·
//!               RTDUCK_DERIBIT_WS · RTDUCK_DERIBIT_REST(시험 = 127.0.0.1) · RTDUCK_DERIBIT_FLUSH_S(시험만 -- 원본 20) ·
//!               10분 체인·요약(deribit_gex_state.json)·매시 대조 = RTDUCK_REST=1 일 때 · RTDUCK_DERIBIT_CHAIN=0 이면 끔(RTCOLLECT_TEST=1 이면 127.0.0.1 밖 요청 거절)
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
        "market_data" => run_market(),
        "raster_archive" => run_raster_archive(),
        "liq_tab" => run_liq_tab(),
        "seal" => run_seal(),
        "deribit" => run_deribit(),
        k => die(2, &format!("모르는 종류 {k:?} -- oi_lsratio|hl_trades|hl_positions|market_data|raster_archive|liq_tab|seal|deribit")),
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

fn flag(k: &str, d: &str) -> bool {
    !["false", "0", ""].contains(&env_s(k, d).trim().to_lowercase().as_str()) // 원본 `not in ("false", "0", "")`
}

/// market_data_worker 의 청산 요격기 + dashboard_state(duckdb_persist_worker.py main()). SIGTERM = 끝(원본도 남길 보류분이 없다).
fn run_market() {
    use std::cell::RefCell;
    use std::rc::Rc;
    let live = PathBuf::from(env_s("RTDUCK_LIVE_DIR", "data/live/rtduck"));
    if live.ends_with("data/live") && env_s("RTDUCK_TAKEOVER", "") != "1" {
        die(2, &format!("🔴{} 는 파이썬 워커의 폴더다(dashboard_state.json · liq_burst_state.json) -- 전환 때만 RTDUCK_TAKEOVER=1", live.display()));
    }
    let db = PathBuf::from(env_s("QUANT_TAIL_DB_PATH", &live.join("tail_risk.duckdb").display().to_string()));
    if db.ends_with("data/live/tail_risk.duckdb") && env_s("RTDUCK_TAKEOVER", "") != "1" {
        die(2, &format!("🔴{} 는 파이썬 워커의 파일이다 -- 전환 때만 RTDUCK_TAKEOVER=1", db.display()));
    }
    let bot = env_s("BOT_SYMBOLS", "ETHUSDT,BTCUSDT,SOLUSDT");
    let micro: Vec<String> = if flag("COLLECT_MICROSTRUCTURE", "true") {
        bot.split(',').map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()).collect()
    } else {
        vec![]
    };
    let mdb = PathBuf::from(env_s("QUANT_MICRO_DB_PATH", &live.join("microstructure.duckdb").display().to_string()));
    if mdb.ends_with("data/live/microstructure.duckdb") && env_s("RTDUCK_TAKEOVER", "") != "1" {
        die(2, &format!("🔴{} 는 파이썬 워커의 파일이다 -- 전환 때만 RTDUCK_TAKEOVER=1", mdb.display()));
    }
    let rest = env_s("RTDUCK_REST", "") == "1";
    let ban = PathBuf::from(env_s("RTCOLLECT_BAN_FILE", "data/live/binance_ban_until"));
    let base = env_s("RTDUCK_BINANCE_REST", "https://fapi.binance.com");
    let (depth_url, trade_url) = (env_s("RTDUCK_DEPTH_WS_URL", rtduck::micro::DEPTH_WS), env_s("RTDUCK_TRADE_WS_URL", rtduck::micro::TRADE_WS));
    let tail: Vec<String> = if flag("COLLECT_TAIL_RISK", "true") {
        env_s("TAIL_SYMBOLS", &bot).split(',').map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()).collect()
    } else {
        vec![]
    };
    let dash = flag("WRITE_DASHBOARD_STATE", "true");
    let url = env_s("RTDUCK_FORCE_WS_URL", rtduck::market::FORCE_ORDER_WS);
    let wall: fn() -> f64 = rtcollect::rest::wall_s;
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    tokio::task::LocalSet::new().block_on(&rt, async {
        let mut tasks = tokio::task::JoinSet::new();
        // 원본 순서: 미시구조 → 요격기(시작 사이 1초 -- 새 DuckDB 파일에서 DDL 이 겹치지 않게)
        for s in &micro {
            let mut c = rtduck::micro::Scanner::new(s, &mdb);
            match c.init(wall()) {
                Ok(n) => println!("MS bootstrap({s}): rows={n}"),
                Err(e) => die(1, &format!("microstructure DB 준비 실패: {e}")),
            }
            let c = Rc::new(RefCell::new(c));
            let agent = rtcollect::rest::agent(Duration::from_secs(5)); // ClientTimeout(total=5)
            let (ban, base) = (ban.clone(), base.clone());
            // 🔧밴 파일을 지킨다(원본은 모른다) · REST 기본 꺼짐 -- 꺼지면 폴링이 예외 길(poll_connected=false · data_stale)
            let get: Box<dyn FnMut(&str) -> Result<serde_json::Value, String>> = Box::new(move |u: &str| {
                if !rest {
                    return Err("REST 꺼짐(RTDUCK_REST=1)".into());
                }
                let now = rtcollect::rest::wall_s();
                if rtcollect::rest::ban_left(&ban, now) > 0.0 {
                    return Err("binance IP ban -- 안 보냄".into());
                }
                let u = u.replacen("https://fapi.binance.com", &base, 1);
                let mut r = agent.get(&u).call().map_err(|e| e.to_string())?;
                let st = r.status().as_u16();
                let ra = r.headers().get("retry-after").and_then(|v| v.to_str().ok()).map(str::to_string);
                let body = r.body_mut().read_to_string().map_err(|e| e.to_string())?;
                rtcollect::rest::ban_note(&ban, st, &body, ra.as_deref(), now);
                serde_json::from_str(&body).map_err(|e| e.to_string()) // 원본 json(content_type=None) -- 상태 코드는 안 본다
            });
            tasks.spawn_local(rtduck::micro::run_streams(c.clone(), depth_url.replace("{symbol}", s), trade_url.replace("{symbol}", s), wall));
            tasks.spawn_local(rtduck::micro::run_poll(c.clone(), get, wall));
            tasks.spawn_local(rtduck::micro::run_scan(c, wall));
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        for s in &tail {
            let mut t = rtduck::market::TailRisk::new(s, &db, &live);
            match t.init(rtcollect::rest::wall_s()) {
                Ok(n) => println!("🛡️ TailRiskInterceptor bootstrap({s}): {n} rows loaded"),
                Err(e) => die(1, &format!("tail_risk DB 준비 실패: {e}")),
            }
            let t = Rc::new(RefCell::new(t));
            tasks.spawn_local(rtduck::market::run_ws(t.clone(), url.replace("{symbol}", s), wall));
            tasks.spawn_local(rtduck::market::run_ticks(t, wall));
            tokio::time::sleep(Duration::from_secs(1)).await; // 원본: 시작 사이 1초
        }
        if dash {
            let p = live.join("dashboard_state.json");
            tasks.spawn_local(async move {
                loop {
                    if let Err(e) = rtduck::market::write_atomic(&p, &rtduck::market::dashboard_state_text(wall())) {
                        eprintln!("dashboard_state write failed: {e}"); // 화면 갱신 신호일 뿐 -- 계속
                    }
                    tokio::time::sleep(Duration::from_secs(10)).await;
                }
            });
        }
        println!("rtduck market_data · 미시구조 {micro:?} → {} (REST {}) · 청산 {tail:?} → {} · dashboard_state {} · {}", mdb.display(),
                 if rest { "켬" } else { "🔴꺼짐" }, db.display(), if dash { "씀" } else { "안 씀" }, live.display());
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).unwrap();
        tokio::select! {
            _ = term.recv() => {}
            _ = tokio::signal::ctrl_c() => {}
            r = tasks.join_next() => die(70, &format!("작업 끝남: {r:?}")),
        }
    });
    println!("종료 신호 -- 끝낸다");
}

fn run_raster_archive() {
    let root = PathBuf::from(env_s("OF_ROOT", "data/live/rtcollect/orderflow/raster"));
    let archive = PathBuf::from(env_s("OF_ARCHIVE_ROOT", "data/live/rtduck/orderflow/raster_archive"));
    if (root.ends_with("data/live/orderflow/raster") || archive.ends_with("data/live/orderflow/raster_archive")) && env_s("RTDUCK_TAKEOVER", "") != "1" {
        die(2, "🔴data/live/orderflow/raster(_archive) 는 파이썬 래스터의 폴더다 -- 전환 때만 RTDUCK_TAKEOVER=1");
    }
    let days: i64 = env_s("OF_RETENTION_DAYS", "14").parse().unwrap_or_else(|_| die(2, "OF_RETENTION_DAYS 는 정수"));
    let do_archive = env_s("OF_ARCHIVE", "1") != "0";
    std::fs::create_dir_all(&archive).unwrap_or_else(|e| die(1, &format!("{}: {e}", archive.display())));
    let archive = archive.canonicalize().unwrap_or_else(|e| die(1, &format!("{}: {e}", archive.display()))); // 뷰 글롭은 절대 경로여야 다른 폴더에서 읽힌다(원본도 절대)
    let syms: Vec<String> = match std::env::var("OF_SYMBOLS") {
        Ok(v) => v.split(',').map(|s| s.trim().to_uppercase()).filter(|s| !s.is_empty()).collect(),
        Err(_) => {
            let mut v: Vec<String> = std::fs::read_dir(&root).map(|r| r.filter_map(|e| e.ok()).filter(|e| e.path().is_dir()).map(|e| e.file_name().to_string_lossy().to_string()).collect()).unwrap_or_default();
            v.sort();
            v
        }
    };
    let now = rtcollect::rest::wall_s();
    let mut view_done = false;
    for s in &syms {
        for l in rtduck::raster_archive::purge_old(&root, &archive, s, now, days, do_archive, &mut view_done) {
            println!("{s}: {l}");
        }
    }
}

fn run_liq_tab() {
    let live = PathBuf::from(env_s("LIQ_TAB_LIVE", "data/live/rtduck"));
    if live.ends_with("data/live") && env_s("RTDUCK_TAKEOVER", "") != "1" {
        die(2, "🔴data/live 의 liq_tab_* 는 파이썬 청산 탭의 파일이다 -- 전환 때만 RTDUCK_TAKEOVER=1");
    }
    std::fs::create_dir_all(&live).unwrap_or_else(|e| die(1, &format!("{}: {e}", live.display())));
    let (lake, hot) = (PathBuf::from(env_s("LIQ_TAB_LAKE", "data/lake")), PathBuf::from(env_s("LIQ_TAB_HOT", "data/hot")));
    let rest = env_s("RTDUCK_REST", "") == "1";
    let no_rest = !rest || !env_s("LIQ_TAB_NO_REST", "").is_empty();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let coin = args.iter().position(|a| a == "--coin").and_then(|i| args.get(i + 1)).cloned();
    let ban = PathBuf::from(env_s("RTCOLLECT_BAN_FILE", "data/live/binance_ban_until"));
    let base = rtcollect::cron::base("RTDUCK_BINANCE_BASE", rtcollect::cboe_summary::BINANCE); // 시험(RTCOLLECT_TEST=1)이면 127.0.0.1 만
    let agent = rtcollect::rest::agent(Duration::from_secs(20)); // urlopen(timeout=20)
    let mut get = |u: &str| -> Result<Vec<u8>, String> {
        let now = rtcollect::rest::wall_s();
        if rtcollect::rest::ban_left(&ban, now) > 0.0 {
            return Err("binance IP ban".into());
        }
        let u = u.replacen(rtcollect::cboe_summary::BINANCE, &base, 1);
        let mut r = agent.get(&u).header("User-Agent", "liq-tab").call().map_err(|e| format!("URLError({e})"))?;
        let st = r.status().as_u16();
        let body = r.body_mut().read_to_string().map_err(|e| e.to_string())?;
        rtcollect::rest::ban_note(&ban, st, &body, None, now);
        if st >= 400 { Err(format!("HTTPError({st})")) } else { Ok(body.into_bytes()) }
    };
    let cx = rtduck::liq_tab::Ctx { lake: &lake, live: &live, hot: &hot, no_rest };
    let now_ms = (rtcollect::rest::wall_s() * 1000.0) as i64;
    match rtduck::liq_tab::run(&cx, coin.as_deref(), now_ms, &mut get, &mut |s| std::thread::sleep(Duration::from_secs_f64(s))) {
        Ok(log) => log.iter().for_each(|l| println!("{l}")),
        Err(e) => die(1, &e),
    }
}

fn run_seal() {
    let lake = PathBuf::from(env_s("SEAL_LAKE", "data/rtduck/lake"));
    let takeover = env_s("RTDUCK_TAKEOVER", "") == "1";
    if lake.ends_with("data/lake") && !takeover {
        die(2, "🔴data/lake 는 파이썬 봉인의 lake 다(전진 검정 자료) -- 전환 때만 RTDUCK_TAKEOVER=1");
    }
    let root = PathBuf::from(env_s("SEAL_ROOT", "."));
    let status = PathBuf::from(env_s("SEAL_STATUS", "data/live/rtduck/seal_status.json"));
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dry = args.iter().any(|a| a == "--dry-run");
    let only = args.iter().position(|a| a == "--only").and_then(|i| args.get(i + 1)).cloned();
    let t0 = rtcollect::rest::wall_s();
    match rtduck::seal::main_once(&root, &lake, Some(&status), t0, dry, only.as_deref(), takeover) {
        Ok((st, log, code)) => {
            log.iter().for_each(|l| println!("{l}"));
            println!("[seal] {}lake={} {} {:.0}s", if dry { "DRY-RUN " } else { "" }, lake.display(), serde_json::Value::Object(st), rtcollect::rest::wall_s() - t0);
            std::process::exit(code)
        }
        Err(e) => die(1, &format!("[seal] {e}")),
    }
}

fn run_deribit() {
    let dir = PathBuf::from(env_s("DERIBIT_BT_DIR", "data/live/rtduck"));
    if dir.ends_with("data/live") && env_s("RTDUCK_TAKEOVER", "") != "1" {
        die(2, "🔴data/live 의 deribit_options.duckdb 는 파이썬 수집기의 파일이다(단일 writer -- 같이 열면 하나가 죽는다) -- 전환 때만 RTDUCK_TAKEOVER=1");
    }
    let (db, state) = (dir.join("deribit_options.duckdb"), dir.join("deribit_block_trades_state.json"));
    let ws_url = env_s("RTDUCK_DERIBIT_WS", rtduck::deribit::WS);
    let rest_base = env_s("RTDUCK_DERIBIT_REST", rtduck::deribit::REST);
    let rest = env_s("RTDUCK_REST", "") == "1";
    let flush_s: f64 = env_s("RTDUCK_DERIBIT_FLUSH_S", "20").parse().unwrap_or(rtduck::deribit::FLUSH_SEC);
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    rt.block_on(deribit_loop(db, state, ws_url, rest_base, rest, flush_s));
}

/// 원본 `run()`(WS 바퀴) -- 구독·하트비트 → 백필(REST 켬일 때) → 버퍼 20초마다 write · 끊기면 5초 뒤 다시(공백은 다음 접속의 write 에) ·
/// SIGTERM 이면 버퍼를 쓰고 0(원본은 안 쓰고 죽는다 -- 의도한 차이 #22).
async fn deribit_loop(db: PathBuf, state: PathBuf, ws_url: String, rest_base: String, rest: bool, flush_s: f64) {
    use futures_util::{SinkExt, StreamExt};
    use rtduck::deribit as d;
    use tokio_tungstenite::tungstenite::Message;
    let wall = rtcollect::rest::wall_s;
    let agent = rtcollect::rest::agent(Duration::from_secs(20)); // requests.get(timeout=20)
    let mut buf: Vec<Vec<duckdb::types::Value>> = vec![];
    let mut down: Option<(i64, String)> = None;
    let mut last = wall();
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).unwrap();
    let flush = |buf: &mut Vec<Vec<duckdb::types::Value>>, gaps: &[(i64, i64, String, Option<i64>)]| -> Result<(usize, usize), String> {
        let now = wall();
        let added = d::write(&db, &state, buf, gaps, now, &d::iso_utc(now))?;
        let n = buf.len();
        buf.clear();
        Ok((n, added))
    };
    let chans = d::channels();
    if rest && env_s("RTDUCK_DERIBIT_CHAIN", "1") != "0" {
        // 원본 chain_loop: 10분 체인·요약·deribit_gex_state.json + 매시 대조(history·헤지 다리·RFQ) -- 🔴REST 전용이라 RTDUCK_REST=1 일 때만.
        //   원본은 체인 폴링 직전에 체결 버퍼를 비운다(옛 딜러 감마 몫) -- 지금 요약은 체결을 안 읽어 Rust 는 비우지 않는다.
        let (db2, dir) = (db.clone(), db.parent().unwrap().to_path_buf());
        std::thread::spawn(move || {
            let agent = rtcollect::rest::agent(Duration::from_secs(90));
            let mut cache = rtduck::deribit_gex::Cache::default();
            let mut recon_at = 0.0;
            loop {
                let now = rtcollect::rest::wall_s();
                let now_us = (now * 1e6) as i64;
                let r = (|| -> Result<Vec<String>, String> {
                    let _g = d::DB_LOCK.lock().unwrap_or_else(|e| e.into_inner());
                    let con = d::connect(&db2)?;
                    rtduck::deribit_gex::poll_once(&con, &dir.join("deribit_gex_state.json"), now_us, now_us, d::local_offset_s(now as i64), &mut cache, &mut |u: &str, timeout: u64, ua: bool| {
                        if std::env::var("RTCOLLECT_TEST").as_deref() == Ok("1") && !u.starts_with("http://127.0.0.1:") {
                            return Err(format!("시험인데 바깥 주소 {u}"));
                        }
                        let mut rq = agent.get(u).config().timeout_global(Some(Duration::from_secs(timeout))).build();
                        if ua {
                            rq = rq.header("User-Agent", "Mozilla/5.0");
                        }
                        let mut r = rq.call().map_err(|e| format!("ConnectionError: {e}"))?;
                        let st = r.status().as_u16();
                        let mut b = vec![];
                        std::io::Read::read_to_end(&mut r.body_mut().as_reader(), &mut b).map_err(|e| e.to_string())?;
                        if st >= 400 { Err(format!("HTTPError: {st}")) } else { Ok(b) }
                    })
                })();
                match r {
                    Ok(log) => log.iter().for_each(|l| println!("[deribit_gex] {l}")),
                    Err(e) => println!("체인 폴링 실패: {e}"),
                }
                if rtcollect::rest::wall_s() - recon_at >= 3600.0 {
                    recon_at = rtcollect::rest::wall_s();
                    let mut log = vec![];
                    match d::reconcile(&db2, (recon_at * 1000.0) as i64, &mut |u: &str| {
                        if std::env::var("RTCOLLECT_TEST").as_deref() == Ok("1") && !u.starts_with("http://127.0.0.1:") {
                            return Err(format!("시험인데 바깥 주소 {u}"));
                        }
                        let mut r = agent.get(u).config().timeout_global(Some(Duration::from_secs(30))).build().call().map_err(|e| format!("ConnectionError: {e}"))?;
                        let st = r.status().as_u16();
                        let mut b = vec![];
                        std::io::Read::read_to_end(&mut r.body_mut().as_reader(), &mut b).map_err(|e| e.to_string())?;
                        if st >= 400 { Err(format!("HTTPError: {st}")) } else { Ok(b) }
                    }, &mut |l| log.push(l)) {
                        Ok(m) => println!("대조 {}", serde_json::Value::Object(m)),
                        Err(e) => println!("대조 실패: {e}"),
                    }
                    log.iter().for_each(|l| println!("{l}"));
                }
                std::thread::sleep(Duration::from_secs_f64(d::next_chain_wait(rtcollect::rest::wall_s())));
            }
        });
    }
    loop {
        let r: Result<(), String> = async {
            let (mut ws, _) = tokio_tungstenite::connect_async(ws_url.as_str()).await.map_err(|e| format!("OSError: {e}"))?;
            let sub = serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "public/subscribe", "params": {"channels": chans}});
            ws.send(Message::text(rtcollect::py_dumps_default(&sub))).await.map_err(|e| format!("ConnectionClosedError: {e}"))?;
            ws.send(Message::text(r#"{"jsonrpc": "2.0", "id": 2, "method": "public/set_heartbeat", "params": {"interval": 30}}"#)).await.map_err(|e| format!("ConnectionClosedError: {e}"))?;
            let now = (wall() * 1000.0) as i64;
            let seen = d::last_ts(&db)?;
            let since: Vec<i64> = seen.iter().map(|s| s.unwrap_or(0).max(now - d::BACKFILL_MAX_MS)).collect();
            let mut bf = if rest {
                d::backfill(&since, now, &mut |u: &str| {
                    let u = u.replacen(d::REST, &rest_base, 1);
                    let mut r = agent.get(&u).call().map_err(|e| format!("ConnectionError: {e}"))?;
                    let st = r.status().as_u16();
                    let mut b = vec![];
                    std::io::Read::read_to_end(&mut r.body_mut().as_reader(), &mut b).map_err(|e| e.to_string())?;
                    if st >= 400 { Err(format!("HTTPError: {st}")) } else { Ok(b) }
                })?
            } else {
                vec![] // 🔴REST 꺼짐 -- 그림자 기본(틈은 안 메운다)
            };
            let nbf = bf.len();
            let mut gaps: Vec<(i64, i64, String, Option<i64>)> = down.take().map(|(f, why)| (f, now, why, Some(nbf as i64))).into_iter().collect();
            for (i, api) in d::API_CURRENCIES.iter().enumerate() {
                if let Some(s) = seen[i].filter(|s| *s < since[i]) {
                    gaps.push((s, since[i], format!("unrecoverable {api}"), Some(0)));
                }
            }
            let (_, added) = flush(&mut bf, &gaps)?;
            println!("구독 {} · 백필 {nbf}건(새 {added}) · {}", chans.join("·"), db.display());
            loop {
                let m = tokio::select! {
                    _ = term.recv() => return Err("SIGTERM".into()),
                    m = tokio::time::timeout(Duration::from_secs_f64(d::IDLE_SEC), ws.next()) => m,
                };
                let txt = match m {
                    Err(_) => return Err("TimeoutError".into()),
                    Ok(None) => return Err("ConnectionClosedOK".into()),
                    Ok(Some(Err(e))) => return Err(format!("ConnectionClosedError: {e}")),
                    Ok(Some(Ok(Message::Text(t)))) => t.to_string(),
                    Ok(Some(Ok(Message::Close(_)))) => return Err("ConnectionClosedOK".into()),
                    Ok(Some(Ok(_))) => String::new(),
                };
                if !txt.is_empty() {
                    let v: serde_json::Value = serde_json::from_str(&txt).map_err(|e| format!("JSONDecodeError: {e}"))?;
                    if v.get("error").is_some() {
                        return Err(format!("RuntimeError: deribit error {}", v["error"]));
                    }
                    let p = &v["params"];
                    if p["type"] == "test_request" {
                        ws.send(Message::text(r#"{"jsonrpc": "2.0", "id": 3, "method": "public/test", "params": {}}"#)).await.map_err(|e| format!("ConnectionClosedError: {e}"))?;
                    } else if p["channel"].as_str().is_some_and(|c| chans.iter().any(|x| x == c)) {
                        for t in p["data"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
                            if t["instrument_name"].as_str().and_then(d::coin_of).is_some() {
                                buf.push(d::row(t)?);
                            }
                        }
                    }
                }
                if wall() - last >= flush_s {
                    match flush(&mut buf, &[]) {
                        Ok((n, added)) => println!("  +{n}행(새 {added})"),
                        Err(e) => println!("  기록 보류 {}행: {e}", buf.len()),
                    }
                    last = wall();
                }
            }
        }.await;
        let e = r.unwrap_err();
        if e == "SIGTERM" {
            if let Err(e) = flush(&mut buf, &[]) {
                eprintln!("  잔여 {}행 기록 실패: {e}", buf.len());
            }
            println!("종료 신호 -- 버퍼를 쓰고 끝낸다");
            std::process::exit(0);
        }
        println!("WS 끊김: {e} — 5초 후 재연결");
        let name = e.split(':').next().unwrap_or("Exception").to_string();
        down = down.or(Some(((wall() * 1000.0) as i64, name)));
        if let Err(e2) = flush(&mut buf, &[]) {
            println!("  잔여 {}행 기록 보류: {e2}", buf.len());
            let n = buf.len();
            if n > 200_000 {
                buf.drain(..n - 200_000);
            }
        }
        tokio::select! {
            _ = term.recv() => std::process::exit(0),
            _ = tokio::time::sleep(Duration::from_secs(5)) => {}
        }
    }
}
