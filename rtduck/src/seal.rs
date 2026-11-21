//! lake 봉인 = scripts/seal_to_lake.py(`seal()` · `prune_hot` · seal_status.json). 원본도 DuckDB 로 쓴다(`copy … (format parquet, compression zstd)`) -- 같은 판(1.5.1)·같은 SQL.
//!   원본과 다른 곳(묶은 DuckDB 에 확장이 없어서 · 값·지문은 같다 -- tests/seal_parity.rs):
//!   ① 날짜 = `date '1970-01-01' + floor(ts_ms / 86400000)`(원본 `cast(timezone('UTC', to_timestamp(ts_ms / 1000.0)) as date)` 는 ICU 확장) -- 정수 ms 면 같은 날짜.
//!   ② SQLite 원천: 원본 `attach … (TYPE sqlite)`(sqlite 확장) 대신 rusqlite 로 읽어 임시 DuckDB(디스크)에 싣는다 -- 열 형은 sqlite_scanner 규칙
//!      (선언에 INT → BIGINT · CHAR/CLOB/TEXT → VARCHAR · REAL/FLOA/DOUB → DOUBLE · 그 밖은 아직 거절). ponytail: 원천 표를 통째로 싣는다(hot 16일 -- 디스크 임시 파일).
//!   ③ COPY 의 (coin, day) 는 매개변수 대신 글자로 박는다.
//!   🔧의도한 차이 #21(prune_hot): 지우기 전에 그 (코인, 날짜)의 lake 파일을 다시 읽어 행 수 ≥ hot 행 수 인지 본다(원본은 파일이 있는지만) -- 아니면 그 표는 안 지운다.
use duckdb::Connection;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

pub const LAG_HOURS: i64 = 2;
pub const RESEAL_DAYS: i64 = 8;

/// (venue, stream, [(원천 상대 경로, 표, coin SQL)], ts_ms SQL)
pub type Spec<'a> = (&'a str, &'a str, Vec<(String, String, String)>, &'a str);

fn q(p: &Path) -> String {
    p.to_string_lossy().replace('\'', "''")
}

fn date(d: i64) -> String {
    let z = d + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let dd = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    format!("{:04}-{m:02}-{dd:02}", yoe + era * 400 + i64::from(m <= 2))
}

/// data_store.lake_path
pub fn lake_path(lake: &Path, venue: &str, stream: &str, coin: &str, day: &str) -> PathBuf {
    lake.join(venue).join(stream).join(format!("coin={coin}")).join(format!("date={day}")).join("part.parquet")
}

/// `snapshot`: 사본을 떠서 표마다 count 가 되는지(찢긴 사본이면 2초 뒤 다시 · 3번).
fn snapshot(src: &Path, tmp: &Path) -> Option<PathBuf> {
    let n = std::fs::read_dir(tmp).map(|r| r.count()).unwrap_or(0);
    let dst = tmp.join(format!("{n}_{}", src.file_name()?.to_string_lossy()));
    for _ in 0..3 {
        if std::fs::copy(src, &dst).is_err() {
            continue;
        }
        let ok = (|| -> duckdb::Result<()> {
            let c = Connection::open_with_flags(&dst, duckdb::Config::default().access_mode(duckdb::AccessMode::ReadOnly)?)?;
            let ts: Vec<String> = c.prepare("select table_name from duckdb_tables()")?.query_map([], |r| r.get(0))?.collect::<Result<_, _>>()?;
            for t in ts {
                c.query_row(&format!("select count(*) from \"{t}\""), [], |r| r.get::<_, i64>(0))?;
            }
            Ok(())
        })();
        if ok.is_ok() {
            return Some(dst);
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
    let _ = std::fs::remove_file(&dst);
    None
}

/// `seal(specs, root, lake, now)` → 원본과 같은 stats 사전 + 로그.
pub fn seal(specs: &[Spec], root: &Path, lake: &Path, now_s: f64, dry_run: bool, only: Option<&str>) -> Result<(Map<String, Value>, Vec<String>), String> {
    let last_day = (now_s as i64 - LAG_HOURS * 3600).div_euclid(86_400) - 1;
    let mut st: Map<String, Value> = ["written", "rows", "skipped_existing", "resealed", "failed", "missing_source"].iter().map(|k| (k.to_string(), json!(0))).collect();
    let inc = |st: &mut Map<String, Value>, k: &str, n: i64| {
        let v = st.get(k).and_then(Value::as_i64).unwrap_or(0) + n;
        st.insert(k.into(), json!(v));
    };
    let mut log = vec![];
    std::fs::create_dir_all(lake).map_err(|e| e.to_string())?;
    let tmp = lake.join(format!(".snap_{}_{}", std::process::id(), (now_s * 1e3) as i64));
    std::fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
    let r = (|| -> Result<(), String> {
        let con = Connection::open(tmp.join("stage.duckdb")).map_err(|e| e.to_string())?; // SQLite 원천을 실을 자리(디스크)
        con.execute_batch("set threads = 2; set memory_limit = '1GB'; SET autoinstall_known_extensions=false; SET autoload_known_extensions=false;").map_err(|e| e.to_string())?;
        let mut snaps: Vec<(PathBuf, Option<String>)> = vec![];
        let mut loaded: Vec<String> = vec![];
        for (venue, stream, sources, ts_sql) in specs {
            if only.is_some_and(|o| o != format!("{venue}/{stream}")) {
                continue;
            }
            let mut parts = vec![];
            for (rel, table, coin_sql) in sources {
                let src = root.join(rel);
                if !src.exists() {
                    inc(&mut st, "missing_source", 1);
                    continue;
                }
                if src.extension().is_some_and(|e| e == "sqlite") {
                    // 원본: 사본 없이 attach(read_only, TYPE sqlite) · 표가 없으면 missing_source
                    let k = match snaps.iter().position(|s| s.0 == src) {
                        Some(k) => k,
                        None => {
                            snaps.push((src.clone(), Some(format!("s{}", snaps.len()))));
                            snaps.len() - 1
                        }
                    };
                    let name = format!("{}__{table}", snaps[k].1.as_ref().unwrap());
                    if !loaded.contains(&name) {
                        if !load_sqlite(&con, &src, table, &name)? {
                            inc(&mut st, "missing_source", 1);
                            continue;
                        }
                        loaded.push(name.clone());
                    }
                    parts.push(format!("select *, ({coin_sql}) as __coin, (date '1970-01-01' + cast(floor(({ts_sql}) / 86400000.0) as integer)) as __day from main.\"{name}\""));
                    continue;
                }
                if !snaps.iter().any(|s| s.0 == src) {
                    let h = snapshot(&src, &tmp).map(|p| {
                        let alias = format!("s{}", snaps.len());
                        con.execute_batch(&format!("attach '{}' as {alias} (read_only)", q(&p))).map(|_| alias)
                    }).transpose().map_err(|e| e.to_string())?;
                    snaps.push((src.clone(), h));
                }
                let Some(h) = snaps.iter().find(|s| s.0 == src).unwrap().1.clone() else {
                    log.push(format!("[seal] 🔴 torn snapshot x3, skip: {rel}"));
                    inc(&mut st, "failed", 1);
                    continue;
                };
                let has: i64 = con.query_row("select count(*) from duckdb_tables() where database_name = ? and table_name = ?", [&h, table], |r| r.get(0)).map_err(|e| e.to_string())?;
                if has == 0 {
                    inc(&mut st, "missing_source", 1);
                    continue;
                }
                parts.push(format!("select *, ({coin_sql}) as __coin, (date '1970-01-01' + cast(floor(({ts_sql}) / 86400000.0) as integer)) as __day from {h}.\"{table}\""));
            }
            if parts.is_empty() {
                continue;
            }
            con.execute_batch(&format!("create or replace temp view v as {}", parts.join(" union all by name "))).map_err(|e| e.to_string())?;
            let cols: Vec<String> = con.prepare("describe select * exclude (__coin, __day) from v").and_then(|mut s| s.query_map([], |r| r.get(0))?.collect()).map_err(|e| e.to_string())?;
            let row = format!("hash(row({}))", cols.iter().map(|c| format!("\"{c}\"")).collect::<Vec<_>>().join(", "));
            let ld = date(last_day);
            let todo: Result<Vec<(String, i64, i64, u64)>, _> = con.prepare(&format!("select __coin, (__day - date '1970-01-01')::BIGINT, count(*), bit_xor({row}) from v \
                where __coin is not null and __day <= DATE '{ld}' group by all order by all"))
                .and_then(|mut s| s.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?.collect());
            let todo = match todo {
                Ok(t) => t,
                Err(e) => {
                    log.push(format!("[seal] 🔴 {venue}/{stream} read failed: {e}").chars().take(300).collect());
                    inc(&mut st, "failed", 1);
                    continue;
                }
            };
            for (coin, day, n, h) in todo {
                let ds = date(day);
                let out = lake_path(lake, venue, stream, &coin, &ds);
                if out.exists() {
                    if day <= last_day - RESEAL_DAYS {
                        inc(&mut st, "skipped_existing", 1);
                        continue;
                    }
                    let (on, oh): (i64, u64) = con.query_row(&format!("select count(*), bit_xor({row}) from read_parquet('{}', hive_partitioning = false)", q(&out)), [],
                        |r| Ok((r.get(0)?, r.get(1)?))).map_err(|e| e.to_string())?;
                    if (n, h) == (on, oh) {
                        inc(&mut st, "skipped_existing", 1);
                        continue;
                    }
                    if n < on {
                        log.push(format!("[seal] 🔴 재봉인 거부 {venue}/{stream} {coin} {ds}: 원천 {n} < lake {on}"));
                        inc(&mut st, "reseal_refused", 1);
                        continue;
                    }
                    inc(&mut st, "resealed", 1);
                }
                if dry_run {
                    inc(&mut st, "written", 1);
                    inc(&mut st, "rows", n);
                    continue;
                }
                std::fs::create_dir_all(out.parent().unwrap()).map_err(|e| e.to_string())?;
                let part = out.with_extension("parquet.part");
                con.execute_batch(&format!("copy (select * exclude (__coin, __day) from v where __coin = '{}' and __day = DATE '{ds}') to '{}' (format parquet, compression zstd)",
                    coin.replace('\'', "''"), q(&part))).map_err(|e| e.to_string())?;
                let got: i64 = con.query_row(&format!("select count(*) from read_parquet('{}')", q(&part)), [], |r| r.get(0)).map_err(|e| e.to_string())?;
                if got != n {
                    let _ = std::fs::remove_file(&part);
                    log.push(format!("[seal] 🔴 row mismatch {venue}/{stream} {coin} {ds}: {got} != {n}"));
                    inc(&mut st, "failed", 1);
                    continue;
                }
                std::fs::rename(&part, &out).map_err(|e| e.to_string())?;
                inc(&mut st, "written", 1);
                inc(&mut st, "rows", n);
            }
        }
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&tmp);
    r.map(|_| (st, log))
}

/// sqlite_scanner 의 선언 형 → DuckDB 형(관측 2026-11-21: duckdb 1.5.5 sqlite 확장 -- 서버 1.5.1 확장과 같은지는 서버 dry-run 으로 확인할 것).
fn sqlite_type(decl: &str) -> Result<&'static str, String> {
    let d = decl.to_lowercase();
    Ok(if d.contains("int") {
        "BIGINT"
    } else if d.contains("char") || d.contains("clob") || d.contains("text") {
        "VARCHAR"
    } else if d.contains("real") || d.contains("floa") || d.contains("doub") {
        "DOUBLE"
    } else {
        return Err(format!("선언 형 {decl:?} 는 아직(BLOB·NUMERIC·DATE 등 -- 봉인 표에는 없다)"));
    })
}

/// SQLite 표 하나를 DuckDB 표 `name` 으로 싣는다. false = 그 표가 없다.
fn load_sqlite(con: &Connection, src: &Path, table: &str, name: &str) -> Result<bool, String> {
    let s = rusqlite::Connection::open_with_flags(src, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).map_err(|e| e.to_string())?;
    let has: i64 = s.query_row("SELECT count(*) FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?", [table], |r| r.get(0)).map_err(|e| e.to_string())?;
    if has == 0 {
        return Ok(false);
    }
    let cols: Vec<(String, String)> = s.prepare(&format!("PRAGMA table_info(\"{table}\")")).and_then(|mut p| p.query_map([], |r| Ok((r.get(1)?, r.get(2)?)))?.collect()).map_err(|e| e.to_string())?;
    let types: Vec<&str> = cols.iter().map(|c| sqlite_type(&c.1)).collect::<Result<_, _>>()?;
    con.execute_batch(&format!("CREATE TABLE \"{name}\" ({})", cols.iter().zip(&types).map(|(c, t)| format!("\"{}\" {t}", c.0)).collect::<Vec<_>>().join(", "))).map_err(|e| e.to_string())?;
    let mut ap = con.appender(name).map_err(|e| e.to_string())?;
    let mut q = s.prepare(&format!("SELECT * FROM \"{table}\"")).map_err(|e| e.to_string())?;
    let mut rows = q.query([]).map_err(|e| e.to_string())?;
    while let Some(r) = rows.next().map_err(|e| e.to_string())? {
        let vals: Vec<duckdb::types::Value> = (0..cols.len()).map(|i| {
            use duckdb::types::Value as D;
            use rusqlite::types::ValueRef as S;
            Ok(match (r.get_ref(i).map_err(|e| e.to_string())?, types[i]) {
                (S::Null, _) => D::Null,
                (S::Integer(x), "BIGINT") => D::BigInt(x),
                (S::Integer(x), "DOUBLE") => D::Double(x as f64),
                (S::Real(x), "DOUBLE") => D::Double(x),
                (S::Text(t), "VARCHAR") => D::Text(String::from_utf8_lossy(t).to_string()),
                (v, t) => return Err(format!("{table}.{}: 저장 형 {v:?} 를 {t} 로(sqlite_scanner 는 형 불일치 오류)", cols[i].0)),
            })
        }).collect::<Result<_, String>>()?;
        ap.append_row(duckdb::appender_params_from_iter(vals)).map_err(|e| e.to_string())?;
    }
    ap.flush().map_err(|e| e.to_string())?;
    Ok(true)
}

/// hot 정리의 코인 바꾸기(원본 람다 다섯)
#[derive(Clone, Copy, Debug)]
pub enum CoinOf {
    /// `s.upper().removesuffix("USDT")`
    Bn,
    /// `s.split("-")[0]`
    Okx,
    /// `lambda s: "ALL"`
    All,
    /// `str`
    Same,
    /// `c.replace(":", "_")`
    Colon,
}

impl CoinOf {
    fn of(self, s: &str) -> String {
        match self {
            CoinOf::Bn => {
                let u = s.to_uppercase();
                u.strip_suffix("USDT").map(str::to_string).unwrap_or(u)
            }
            CoinOf::Okx => s.split('-').next().unwrap_or("").to_string(),
            CoinOf::All => "ALL".into(),
            CoinOf::Same => s.to_string(),
            CoinOf::Colon => s.replace(':', "_"),
        }
    }
}

/// (hot 상대 경로, 표, 초 SQL, 코인 열, venue, stream, 코인 바꾸기, 보존 일) -- 원본 HOT_PRUNE 순서 그대로.
pub fn hot_prune() -> Vec<(&'static str, String, &'static str, &'static str, &'static str, String, CoinOf, i64)> {
    use CoinOf::*;
    let mut v = vec![
        ("data/hot/binance_tape.sqlite", "trade_tape_1s".to_string(), "ts_sec", "symbol", "binance", "tape".to_string(), Bn, 16),
        ("data/hot/okx_tape.sqlite", "trade_tape_1s".into(), "ts_sec", "symbol", "okx", "tape".into(), Okx, 16),
    ];
    for t in ["oi", "mark", "funding"] {
        v.push(("data/hot/okx_ctx.sqlite", format!("okx_{t}"), "ts_ms / 1000", "inst", "okx", t.into(), Okx, 8));
    }
    v.push(("data/hot/okx_ctx.sqlite", "okx_liquidations".into(), "ts_ms / 1000", "inst_id", "okx", "liquidations".into(), All, 8));
    v.push(("data/hot/hl_ctx.sqlite", "hl_asset_ctx".into(), "recv_ms / 1000", "coin", "hl", "asset_ctx".into(), Same, 8));
    for f in ["data/hot/hl_positions.sqlite", "data/hot/hl_positions_sol_xrp.sqlite", "data/hot/hl_positions_btc.sqlite"] {
        v.push((f, "hl_positions".into(), "ts_ms / 1000", "coin", "hl", "positions".into(), Same, 8));
        v.push((f, "hl_liquidations".into(), "detected_ms / 1000", "coin", "hl", "liquidations".into(), Same, 8));
    }
    v.push(("data/hot/binance_spot_tape.sqlite", "trade_tape_1s".into(), "ts_sec", "symbol", "binance", "spot_tape".into(), Bn, 8));
    v.push(("data/hot/bybit_tape.sqlite", "trade_tape_1s".into(), "ts_sec", "symbol", "bybit", "tape".into(), Bn, 16));
    for t in ["oi", "mark", "funding"] {
        v.push(("data/hot/bybit_ctx.sqlite", format!("bybit_{t}"), "ts_ms / 1000", "inst", "bybit", t.into(), Bn, 8));
    }
    v.push(("data/hot/treasury_futures.sqlite", "treasury_fut_1m".into(), "ts_sec", "symbol", "cme", "treasury_fut_1m".into(), Same, 8));
    v.push(("data/hot/hl_xyz.sqlite", "hl_xyz_1m".into(), "ts_sec", "coin", "hl", "xyz_1m".into(), Colon, 8));
    v.push(("data/hot/hl_xyz.sqlite", "hl_xyz_ctx_1s".into(), "recv_sec", "coin", "hl", "xyz_ctx_1s".into(), Colon, 8));
    v.push(("data/hot/macro_5m.sqlite", "yahoo_5m".into(), "ts_sec", "symbol", "yahoo", "macro_5m".into(), Same, 8));
    for (t, st) in [("oi_1s", "oi_1s"), ("mark_price_1s", "mark_1s"), ("liquidations", "liquidations")] {
        v.push(("data/hot/binance_ctx.sqlite", t.into(), "ts_ms / 1000", "symbol", "binance", st.into(), Bn, 8));
    }
    v
}

/// 원본 SPECS 그대로(venue, stream, [(원천, 표, coin SQL)], ts_ms SQL) -- 시험이 파이썬 SPECS 와 글자로 대조한다.
pub fn specs() -> Vec<(&'static str, String, Vec<(String, String, String)>, &'static str)> {
    let (l, a) = ("data/live", "data/archive/live_retired_20261001");
    let sym = "upper(regexp_replace(symbol, '(?i)usdt$', ''))".to_string();
    let okx = "split_part(symbol, '-', 1)".to_string();
    let deribit = "split_part(split_part(instrument_name, '-', 1), '_', 1)".to_string();
    let sfx = |c: &str| if c == "eth" { String::new() } else { format!("_{c}") };
    let lit = |c: &str| format!("'{}'", c.to_uppercase());
    let one = |f: &str, t: &str, c: &str| vec![(f.to_string(), t.to_string(), c.to_string())];
    let bn_ctx = "data/hot/binance_ctx.sqlite";
    let okx_ctx = "data/hot/okx_ctx.sqlite";
    let dbo = format!("{l}/deribit_options.duckdb");
    let hl = |t: &str| ["data/hot/hl_positions.sqlite", "data/hot/hl_positions_sol_xrp.sqlite", "data/hot/hl_positions_btc.sqlite"].iter().map(|f| (f.to_string(), t.to_string(), "coin".to_string()))
        .chain(std::iter::once((format!("{l}/hyperliquid_positions_btc_sol_xrp_hype.from_pi.duckdb"), t.to_string(), "coin".to_string()))).collect::<Vec<_>>();
    let mut v = vec![
        ("binance", "tape".to_string(), one("data/hot/binance_tape.sqlite", "trade_tape_1s", &sym), "ts_sec * 1000"),
        ("binance", "spot_tape".into(), one("data/hot/binance_spot_tape.sqlite", "trade_tape_1s", &sym), "ts_sec * 1000"),
        ("binance", "oi_1s".into(), one(bn_ctx, "oi_1s", &sym), "ts_ms"),
        ("binance", "mark_1s".into(), one(bn_ctx, "mark_price_1s", &sym), "ts_ms"),
        ("binance", "liquidations".into(), one(bn_ctx, "liquidations", &sym), "ts_ms"),
        ("binance", "oi_lsratio_5m".into(), ["eth", "btc", "sol", "xrp"].iter().map(|c| (format!("{l}/oi_lsratio.duckdb"), format!("oi_lsratio_5m{}", sfx(c)), lit(c)))
            .chain(["xrp", "hype"].iter().map(|c| (format!("{a}/oi_lsratio_{c}.duckdb"), format!("oi_lsratio_5m_{c}"), lit(c)))).collect(), "epoch_ms(ts)"),
        ("binance", "micro_1m".into(), ["eth", "btc", "sol"].iter().map(|c| (format!("{l}/microstructure.duckdb"), format!("microstructure_1m{}", sfx(c)), lit(c)))
            .chain(["xrp", "hype"].iter().map(|c| (format!("{a}/microstructure_{c}.duckdb"), format!("microstructure_1m_{c}"), lit(c)))).collect(), "epoch_ms(ts)"),
        ("binance", "tail_risk_1m".into(), std::iter::once((format!("{l}/tail_risk.duckdb"), "tail_risk_1m".to_string(), lit("eth")))
            .chain(["btc", "sol"].iter().map(|c| (format!("{a}/tail_risk_btc_sol.duckdb"), format!("tail_risk_1m_{c}"), lit(c))))
            .chain(["xrp", "hype"].iter().map(|c| (format!("{a}/tail_risk_{c}.duckdb"), format!("tail_risk_1m_{c}"), lit(c)))).collect(), "epoch_ms(ts)"),
        ("okx", "tape".into(), one("data/hot/okx_tape.sqlite", "trade_tape_1s", &okx), "ts_sec * 1000"),
    ];
    for s in ["oi", "mark", "funding"] {
        v.push(("okx", s.into(), one(okx_ctx, &format!("okx_{s}"), "split_part(inst, '-', 1)"), "ts_ms"));
    }
    v.push(("okx", "liquidations".into(), one(okx_ctx, "okx_liquidations", "'ALL'"), "ts_ms"));
    v.push(("hl", "asset_ctx".into(), one("data/hot/hl_ctx.sqlite", "hl_asset_ctx", "coin"), "recv_ms"));
    v.push(("bybit", "tape".into(), one("data/hot/bybit_tape.sqlite", "trade_tape_1s", &sym), "ts_sec * 1000"));
    for s in ["oi", "mark", "funding"] {
        v.push(("bybit", s.into(), one("data/hot/bybit_ctx.sqlite", &format!("bybit_{s}"), &sym.replace("symbol", "inst")), "ts_ms"));
    }
    v.push(("cme", "treasury_fut_1m".into(), one("data/hot/treasury_futures.sqlite", "treasury_fut_1m", "symbol"), "ts_sec * 1000"));
    v.push(("hl", "xyz_1m".into(), one("data/hot/hl_xyz.sqlite", "hl_xyz_1m", "replace(coin, ':', '_')"), "ts_sec * 1000"));
    v.push(("hl", "xyz_ctx_1s".into(), one("data/hot/hl_xyz.sqlite", "hl_xyz_ctx_1s", "replace(coin, ':', '_')"), "recv_sec * 1000"));
    v.push(("yahoo", "macro_5m".into(), one("data/hot/macro_5m.sqlite", "yahoo_5m", "symbol"), "ts_sec * 1000"));
    v.push(("hl", "positions".into(), hl("hl_positions"), "ts_ms"));
    v.push(("hl", "liquidations".into(), hl("hl_liquidations"), "detected_ms"));
    v.push(("deribit", "option_trades".into(), one(&dbo, "option_trades", &deribit), "ts_ms"));
    v.push(("deribit", "chain".into(), one(&dbo, "option_chain_snapshot", "currency"), "epoch_ms(recorded_at_utc)"));
    v.push(("deribit", "gex_summary".into(), one(&dbo, "gex_summary", "currency"), "epoch_ms(recorded_at_utc)"));
    v.push(("deribit", "option_summary".into(), one(&dbo, "option_summary", "currency"), "epoch_ms(recorded_at_utc)"));
    v.push(("deribit", "block_rfq".into(), one(&dbo, "block_rfq_trades", "'ALL'"), "ts_ms"));
    v.push(("deribit", "block_future_legs".into(), one(&dbo, "block_future_legs", &deribit), "ts_ms"));
    v
}

/// `prune_hot` 한 표 → 지운 행 수(로그). 🔧#21: 지우기 전 lake 되읽기.
pub fn prune_hot(hot: &Path, table: &str, ts_sql: &str, coin_col: &str, venue: &str, stream: &str, coin_of: CoinOf, keep_days: i64, now_s: f64, lake: &Path, log: &mut Vec<String>) -> Result<i64, String> {
    if !hot.exists() {
        return Ok(0);
    }
    let cut = (now_s as i64).div_euclid(86_400) * 86_400 - keep_days * 86_400;
    let ro = rusqlite::Connection::open_with_flags(hot, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).map_err(|e| e.to_string())?;
    ro.busy_timeout(std::time::Duration::from_secs(30)).map_err(|e| e.to_string())?;
    let held: Vec<(String, i64, i64)> = ro.prepare(&format!("SELECT {coin_col}, CAST({ts_sql} AS INTEGER) / 86400, count(*) FROM {table} WHERE {ts_sql} < ? GROUP BY 1, 2"))
        .and_then(|mut s| s.query_map([cut], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.collect()).map_err(|e| e.to_string())?;
    drop(ro);
    let mut need: std::collections::BTreeMap<(String, String), i64> = Default::default();
    for (c, d, n) in held {
        *need.entry((coin_of.of(&c), date(d))).or_default() += n;
    }
    let unsealed: Vec<&(String, String)> = need.keys().filter(|(c, d)| !lake_path(lake, venue, stream, c, d).exists()).collect();
    if !unsealed.is_empty() {
        log.push(format!("[seal] 🔴 hot 정리 건너뜀 {venue}/{stream}: lake 에 없는 날짜 {:?}", &unsealed[..unsealed.len().min(5)]));
        return Ok(0);
    }
    let dc = Connection::open_in_memory().map_err(|e| e.to_string())?;
    for ((c, d), n) in &need {
        let got: i64 = dc.query_row(&format!("select count(*) from read_parquet('{}')", q(&lake_path(lake, venue, stream, c, d))), [], |r| r.get(0)).map_err(|e| e.to_string())?;
        if got < *n {
            log.push(format!("[seal] 🔴 hot 정리 건너뜀 {venue}/{stream}: {c} {d} lake {got}행 < hot {n}행(되읽기)"));
            return Ok(0);
        }
    }
    let mut con = rusqlite::Connection::open(hot).map_err(|e| e.to_string())?;
    con.busy_timeout(std::time::Duration::from_secs(30)).map_err(|e| e.to_string())?;
    let tx = con.transaction().map_err(|e| e.to_string())?;
    let n = tx.execute(&format!("DELETE FROM {table} WHERE {ts_sql} < ?"), [cut]).map_err(|e| e.to_string())? as i64;
    tx.commit().map_err(|e| e.to_string())?;
    let mut st = con.prepare("PRAGMA incremental_vacuum").map_err(|e| e.to_string())?;
    let mut r = st.query([]).map_err(|e| e.to_string())?;
    while r.next().map_err(|e| e.to_string())?.is_some() {}
    Ok(n)
}

/// 원본 `__main__`: seal → (실패 없고 dry-run·only 아니면) hot 정리 → seal_status.json. 반환 = (stats, 로그, 종료코드).
pub fn main_once(root: &Path, lake: &Path, status: Option<&Path>, now_s: f64, dry_run: bool, only: Option<&str>, prune: bool) -> Result<(Map<String, Value>, Vec<String>, i32), String> {
    let sp = specs();
    let sp: Vec<Spec> = sp.iter().map(|(v, s, src, ts)| (*v, s.as_str(), src.clone(), *ts)).collect();
    let (mut st, mut log) = seal(&sp, root, lake, now_s, dry_run, only)?;
    let failed = st["failed"].as_i64().unwrap_or(0);
    if !dry_run && only.is_none() && failed == 0 && prune {
        let mut hp = Map::new();
        for (f, t, ts, cc, v, s, c, k) in hot_prune() {
            hp.insert(format!("{v}/{s}"), json!(prune_hot(&root.join(f), &t, ts, cc, v, &s, c, k, now_s, lake, &mut log)?));
        }
        st.insert("hot_pruned".into(), Value::Object(hp));
    }
    if let (Some(p), false, None) = (status, dry_run, only) {
        let mut m = Map::new();
        m.insert("finished_at".into(), json!(now_s));
        m.extend(st.clone());
        std::fs::create_dir_all(p.parent().unwrap()).map_err(|e| e.to_string())?;
        let tmp = p.with_extension("tmp");
        std::fs::write(&tmp, rtcollect::py_dumps_default_utf8(&Value::Object(m))).and_then(|_| std::fs::rename(&tmp, p)).map_err(|e| e.to_string())?;
    }
    Ok((st, log, i32::from(failed > 0)))
}
