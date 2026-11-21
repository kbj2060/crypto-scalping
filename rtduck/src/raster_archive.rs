//! 래스터 14일 보관 = live_orderflow_raster_collector_20260914 `_purge_old`·`_archive_then_unlink`·`_f32_to_parquet`·`_ensure_archive_db`(:410-533·704-713).
//!   만료(.f32 시각 < 지금 − OF_RETENTION_DAYS) → `<보관>/<SYM>/<날짜>/<이름>.parquet`(ts_ms int64 · bin int32 · qty float · bin_size float, qty≠0 칸만)
//!   + `<이름>_mid.parquet`(ts_ms · mid 전 행) · zstd → `orderbook.duckdb` 뷰 book·book_mid(글롭, 데이터 복사 없음) → 원본 지움.
//!   parquet 이 안 되면(헤더보다 짧다 · magic · 행 0) gzip(compresslevel 6) 으로. 이미 온전한 보관본이 있으면 원본만 지운다.
//!   🔧원본과 다른 곳(의도한 차이 #20): 새로 쓴 보관본을 **다시 읽어** 행 수(parquet: 0 아닌 칸 수·mid 행 수 / gzip: 푼 바이트 = 원본)가 맞을 때만 원본을 지운다
//!     -- 안 맞으면 쓴 보관본을 지우고 원본을 남긴다(다음 시간 재시도). 원본은 쓰기 성공만 보고 지운다.
//!   처리 순서: 원본은 글롭(디렉터리 순서 -- 정해지지 않음), Rust 는 이름순. 파일마다 따로라 결과는 같다.
use duckdb::{params, Connection};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"FLWR";
const HEADER: usize = 32;

/// `datetime.strptime(stem, "%Y-%m-%dT%H")`(UTC) → 초. 연 4자리 · 월·일·시 1~2자리 · 없는 날짜는 None(ValueError).
pub fn parse_stamp(stem: &str) -> Option<i64> {
    let (d, h) = stem.split_once('T')?;
    let mut it = d.splitn(3, '-');
    let (y, m, dd) = (it.next()?, it.next()?, it.next()?);
    let num = |s: &str, lo: usize, hi: usize| (s.len() >= lo && s.len() <= hi && s.bytes().all(|b| b.is_ascii_digit())).then(|| s.parse::<i64>().ok()).flatten();
    let (y, m, dd, h) = (num(y, 4, 4)?, num(m, 1, 2)?, num(dd, 1, 2)?, num(h, 1, 2)?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&dd) || h > 23 {
        return None;
    }
    let days = days_from_civil(y, m, dd);
    (civil(days) == (y, m, dd)).then_some(days * 86_400 + h * 3600)
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
}

fn civil(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

fn sql_str(p: &Path) -> String {
    p.to_string_lossy().replace('\'', "''")
}

/// `_f32_to_parquet`: Err = 원본 OSError 글(폴백 gzip). Ok = (0 아닌 칸 수, 행 수).
fn to_parquet(src: &Path, dst: &Path) -> Result<(usize, usize), String> {
    let raw = std::fs::read(src).map_err(|e| e.to_string())?;
    if raw.len() < HEADER {
        return Err("헤더보다 짧다".into());
    }
    if &raw[..4] != MAGIC {
        return Err("magic 불일치".into());
    }
    let nb = u16::from_le_bytes([raw[6], raw[7]]) as usize;
    let binsz = f32::from_le_bytes(raw[8..12].try_into().unwrap());
    let rb = 16 + 4 * nb;
    let n = (raw.len() - HEADER) / rb;
    if n == 0 {
        return Err("행이 없다".into());
    }
    let con = Connection::open_in_memory().map_err(|e| e.to_string())?;
    con.execute_batch("SET threads=1; SET autoinstall_known_extensions=false; SET autoload_known_extensions=false;
        CREATE TABLE b(ts_ms BIGINT, bin INTEGER, qty FLOAT, bin_size FLOAT); CREATE TABLE m(ts_ms BIGINT, mid FLOAT);").map_err(|e| e.to_string())?;
    let mut nz = 0;
    {
        let (mut ab, mut am) = (con.appender("b").map_err(|e| e.to_string())?, con.appender("m").map_err(|e| e.to_string())?);
        for r in 0..n {
            let o = HEADER + r * rb;
            let ts = i64::from_le_bytes(raw[o..o + 8].try_into().unwrap());
            let lo = i32::from_le_bytes(raw[o + 8..o + 12].try_into().unwrap());
            let mid = f32::from_le_bytes(raw[o + 12..o + 16].try_into().unwrap());
            for c in 0..nb {
                let q = f32::from_le_bytes(raw[o + 16 + 4 * c..o + 20 + 4 * c].try_into().unwrap());
                if q != 0.0 {
                    // np.nonzero: NaN 은 0 이 아니다 · -0.0 은 0
                    ab.append_row(params![ts, lo.wrapping_add(c as i32), q, binsz]).map_err(|e| e.to_string())?;
                    nz += 1;
                }
            }
            am.append_row(params![ts, mid]).map_err(|e| e.to_string())?;
        }
    }
    std::fs::create_dir_all(dst.parent().unwrap()).map_err(|e| e.to_string())?;
    let part = PathBuf::from(format!("{}.part", dst.display()));
    let stem = dst.file_stem().unwrap().to_string_lossy().to_string();
    let (mpart, mdst) = (dst.with_file_name(format!("{stem}_mid.parquet.part")), dst.with_file_name(format!("{stem}_mid.parquet")));
    con.execute_batch(&format!("COPY b TO '{}' (FORMAT PARQUET, COMPRESSION ZSTD)", sql_str(&part))).map_err(|e| e.to_string())?;
    std::fs::rename(&part, dst).map_err(|e| e.to_string())?;
    con.execute_batch(&format!("COPY m TO '{}' (FORMAT PARQUET, COMPRESSION ZSTD)", sql_str(&mpart))).map_err(|e| e.to_string())?;
    std::fs::rename(&mpart, &mdst).map_err(|e| e.to_string())?;
    Ok((nz, n))
}

fn count(p: &Path) -> Result<usize, String> {
    let con = Connection::open_in_memory().map_err(|e| e.to_string())?;
    con.query_row(&format!("SELECT count(*) FROM read_parquet('{}')", sql_str(p)), [], |r| r.get::<_, i64>(0)).map(|x| x as usize).map_err(|e| e.to_string())
}

/// `_ensure_archive_db` -- 글이 원본과 같은 뷰 둘.
pub fn ensure_view(archive: &Path) -> Result<(), String> {
    let con = Connection::open(archive.join("orderbook.duckdb")).map_err(|e| e.to_string())?;
    let g = sql_str(&archive.join("*").join("*").join("*T[0-9][0-9].f32.parquet"));
    let m = sql_str(&archive.join("*").join("*").join("*_mid.parquet"));
    let sym = r"regexp_extract(filename, '([^/\\]+)[/\\][^/\\]+[/\\][^/\\]+$', 1) AS symbol";
    con.execute_batch(&format!("CREATE OR REPLACE VIEW book AS SELECT {sym}, * EXCLUDE (filename) FROM read_parquet('{g}', filename = true, union_by_name = true);
        CREATE OR REPLACE VIEW book_mid AS SELECT {sym}, * EXCLUDE (filename) FROM read_parquet('{m}', filename = true);")).map_err(|e| e.to_string())
}

fn nonempty(p: &Path) -> bool {
    p.metadata().is_ok_and(|m| m.len() > 0)
}

/// `_archive_then_unlink(p, symbol)` -- 로그 한 줄.
pub fn archive_then_unlink(p: &Path, symbol: &str, archive: &Path, view_done: &mut bool) -> String {
    let name = p.file_name().unwrap().to_string_lossy().to_string();
    let day = p.file_stem().unwrap().to_string_lossy().split('T').next().unwrap().to_string();
    let dir = archive.join(symbol.to_uppercase()).join(&day);
    let (dst, gz) = (dir.join(format!("{name}.parquet")), dir.join(format!("{name}.gz")));
    if nonempty(&dst) || nonempty(&gz) {
        let _ = std::fs::remove_file(p);
        return format!("보관본 있음 {name} → 원본만 지움");
    }
    let before = p.metadata().map(|m| m.len()).unwrap_or(0);
    let (how, after, ok) = match to_parquet(p, &dst) {
        Ok((nz, n)) => {
            let mid = dst.with_file_name(format!("{}_mid.parquet", dst.file_stem().unwrap().to_string_lossy()));
            let ok = count(&dst) == Ok(nz) && count(&mid) == Ok(n);
            if !ok {
                let _ = std::fs::remove_file(&dst);
                let _ = std::fs::remove_file(&mid);
            } else if !*view_done {
                match ensure_view(archive) {
                    Ok(()) => *view_done = true,
                    Err(e) => eprintln!("보관 뷰 갱신 보류: {e} (parquet 은 정상)"),
                }
            }
            ("parquet", dst.metadata().map(|m| m.len()).unwrap_or(0), ok)
        }
        Err(e) => {
            eprintln!("parquet 변환 실패 {name}: {e} — gzip 으로 보관한다");
            let part = PathBuf::from(format!("{}.part", gz.display()));
            let r = std::fs::create_dir_all(&dir).and_then(|_| rtcollect::gzip_file(p, &part));
            let same = r.is_ok() && nonempty(&part) && rtcollect::gunzip_file(&part).ok() == std::fs::read(p).ok();
            if !same {
                let _ = std::fs::remove_file(&part);
                return format!("보관 실패 {name}: gzip {r:?} — 원본 유지(다음 시간 재시도)");
            }
            if let Err(e) = std::fs::rename(&part, &gz) {
                return format!("보관 실패 {name}: {e} — 원본 유지(다음 시간 재시도)");
            }
            ("gzip", gz.metadata().map(|m| m.len()).unwrap_or(0), true)
        }
    };
    if !ok {
        return format!("보관 실패 {name}: 되읽은 행 수가 다르다 — 보관본 지우고 원본 유지(다음 시간 재시도)");
    }
    if let Err(e) = std::fs::remove_file(p) {
        return format!("보관 실패 {name}: {e} — 원본 유지(다음 시간 재시도)");
    }
    format!("보관 {name} → {day}/{how} · {:.1}MB → {:.1}MB ({:.1}배)", before as f64 / 1e6, after as f64 / 1e6, before as f64 / after.max(1) as f64)
}

/// `_purge_old` 한 심볼: `<root>/<SYM>/*.f32` 중 시각 < now − retention 일 → 보관(또는 do_archive=false 면 그냥 지움). 반환 = 로그.
pub fn purge_old(root: &Path, archive: &Path, symbol: &str, now_s: f64, retention_days: i64, do_archive: bool, view_done: &mut bool) -> Vec<String> {
    let cutoff = now_s - (retention_days * 86_400) as f64;
    let Ok(rd) = std::fs::read_dir(root.join(symbol.to_uppercase())) else { return vec![] };
    let mut ps: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "f32") && !p.file_name().unwrap().to_string_lossy().starts_with('.')).collect();
    ps.sort();
    let mut log = vec![];
    for p in ps {
        let Some(stamp) = parse_stamp(&p.file_stem().unwrap().to_string_lossy()) else { continue };
        if (stamp as f64) < cutoff {
            if do_archive {
                log.push(archive_then_unlink(&p, symbol, archive, view_done));
            } else {
                let _ = std::fs::remove_file(&p);
            }
        }
    }
    log
}
