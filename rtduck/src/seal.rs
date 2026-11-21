//! lake 봉인 = scripts/seal_to_lake.py `seal()`(첫 조각 -- DuckDB 파일 원천만). 원본도 DuckDB 로 쓴다(`copy … (format parquet, compression zstd)`) -- 같은 판(1.5.1)·같은 SQL.
//!   원본과 다른 곳(묶은 DuckDB 에 확장이 없어서 · 값은 같다):
//!   ① 날짜 = `date '1970-01-01' + floor(ts_ms / 86400000)`(원본 `cast(timezone('UTC', to_timestamp(ts_ms / 1000.0)) as date)` 는 ICU 확장) -- 정수 ms 면 같은 날짜.
//!   ② COPY 의 (coin, day) 는 매개변수 대신 글자로 박는다.
//!   🔴SQLite 원천(hot 테이프·맥락 -- `attach … (TYPE sqlite)`)은 sqlite 확장이 없어 아직 못 한다 · jsonl 원천(`read_json_auto`)은 json 기능(재빌드) -- 지금 SPECS 에는 없다.
//!   hot 정리(`prune_hot`)·seal_status.json 도 아직(다음 조각).
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
        let con = Connection::open_in_memory().map_err(|e| e.to_string())?;
        con.execute_batch("set threads = 2; set memory_limit = '1GB'; SET autoinstall_known_extensions=false; SET autoload_known_extensions=false;").map_err(|e| e.to_string())?;
        let mut snaps: Vec<(PathBuf, Option<String>)> = vec![];
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
                if src.extension().is_some_and(|e| e != "duckdb") {
                    return Err(format!("🔴{rel}: DuckDB 가 아닌 원천은 아직(sqlite 확장 없음)"));
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
