//! 첫 엔드포인트 묶음 -- 워커·수집기가 쓴 파일·SQLite 만 읽어 답하는 것(계산·REST·메모리 상태 없음).
//! 함수마다 원본 위치를 적는다(dashboard/server.py 2026-11-21 판 · 별도 모듈은 그 파일). 원본의 swr 캐시(1~60초)는 옮기지 않았다:
//! 읽기가 µs 라 캐시가 할 일이 없고, 캐시가 없으면 원본보다 «덜 묵은» 값이 나간다(의도한 차이 · 문서 §의도한 차이).

use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::types::Value as Sql;
use serde_json::{json, Map, Value};

use crate::py::{self, err, f, i, item, PyErr, R, V};
use crate::{http_exc, server_error, Config, Req, Resp, COIN_CONFIG};

pub fn handler(path: &str) -> Option<fn(&Config, &Req) -> Resp> {
    Some(match path {
        "/api/acct-notes" => acct_notes,
        "/api/acct-wrong" => acct_wrong,
        "/api/risk-radar" => risk_radar,
        "/api/sr-lean" => sr_lean,
        "/api/regime-band" => regime_band,
        "/api/liq-tab" => liq_tab,
        "/api/iceberg" => iceberg,
        "/api/ob5m" => ob5m,
        "/api/treasury-futures" => treasury_futures,
        "/api/xyz100" => xyz100,
        "/api/macro-today" => macro_today,
        "/api/macro-calendar" => macro_calendar,
        _ => return None,
    })
}

/// 원본 핸들러의 `try: … except Exception as exc: payload = {"ok": False, "reason": f"{type}: {exc}"}`
fn guarded(r: R<V>) -> Resp {
    Resp::json(&r.unwrap_or_else(|PyErr(m)| json!({"ok": false, "reason": m})))
}

/// 예외를 안 잡는 핸들러 -- 파이썬 예외 = 500
fn or500(r: R<V>) -> Resp {
    r.map_or_else(|_| server_error(), |v| Resp::json(&v))
}

fn read_text(p: &Path) -> Option<String> {
    fs::read(p).ok().and_then(|b| String::from_utf8(b).ok())
}

/// 파일 꼬리 n 바이트를 utf-8(replace)로 풀어 줄로 -- 잘린 첫 줄은 버린다(sr_lean·regime_band 같은 모양)
fn tail_lines(p: &Path, n: u64) -> Option<Vec<String>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut fh = fs::File::open(p).ok()?;
    let size = fh.seek(SeekFrom::End(0)).ok()?;
    fh.seek(SeekFrom::Start(size.saturating_sub(n))).ok()?;
    let mut b = Vec::new();
    fh.read_to_end(&mut b).ok()?;
    let s = String::from_utf8_lossy(&b);
    Some(py::splitlines(&s).into_iter().skip(usize::from(size > n)).map(str::to_string).collect())
}

// ───────────────────────── 계좌 메모 (dashboard/acct_notes.py) ─────────────────────────

/// GET /api/acct-notes (server.py:2866 · acct_notes.load: 사전이 아니거나 못 읽으면 {})
fn acct_notes(cfg: &Config, _: &Req) -> Resp {
    let d = read_text(&cfg.live_dir().join("acct_notes.json")).and_then(|s| py::loads(&s).ok()).filter(Value::is_object);
    Resp::json(&json!({"notes": d.unwrap_or_else(|| json!({}))}))
}

/// GET /api/acct-wrong (server.py:2878 · acct_notes.wrong_load: 목록이 아니거나 못 읽으면 [])
fn acct_wrong(cfg: &Config, _: &Req) -> Resp {
    let d = read_text(&cfg.live_dir().join("acct_wrong.json")).and_then(|s| py::loads(&s).ok()).filter(Value::is_array);
    Resp::json(&json!({"rows": d.unwrap_or_else(|| json!([]))}))
}

// ───────────────────────── 위험 레이더 (dashboard/risk_radar.py) ─────────────────────────

/// `calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))`
fn timegm_z(s: &str) -> R<i64> {
    let bad = || err("ValueError", format!("time data '{s}' does not match format '%Y-%m-%dT%H:%M:%SZ'"));
    let b = s.as_bytes();
    if b.len() != 20 || b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' || b[19] != b'Z' {
        return bad();
    }
    let n = |a: usize, z: usize| s[a..z].parse::<i64>().ok().filter(|_| s[a..z].bytes().all(|c| c.is_ascii_digit()));
    let (Some(y), Some(mo), Some(d), Some(h), Some(mi), Some(se)) = (n(0, 4), n(5, 7), n(8, 10), n(11, 13), n(14, 16), n(17, 19)) else { return bad() };
    if !(1..=12).contains(&mo) || d < 1 || d > days_in_month(y, mo) || h > 23 || mi > 59 || se > 61 {
        return bad();
    }
    Ok(days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 + se)
}

fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        2 if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// 1970-01-01 부터 날 수(그레고리력)
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * ((m + 9) % 12) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn is_params_name(nm: &str) -> bool {
    // r"risk_radar_params_\d{8}[a-z]?\.json"
    let Some(rest) = nm.strip_prefix("risk_radar_params_").and_then(|r| r.strip_suffix(".json")) else { return false };
    let b = rest.as_bytes();
    (b.len() == 8 || b.len() == 9 && b[8].is_ascii_lowercase()) && b[..8].iter().all(u8::is_ascii_digit)
}

fn risk_radar_payload(root: &Path, now: f64) -> R<V> {
    let latest = root.join("data/live/risk_radar_latest.json");
    if !latest.exists() {
        return Ok(json!({"ok": false, "reason": "기록 대기"}));
    }
    let d = py::loads(&read_text(&latest).ok_or_else(|| PyErr("UnicodeDecodeError: utf-8".into()))?)?;
    let nm = py::get(&d, "params")?.filter(|v| py::truthy(v)).map(py::to_str).unwrap_or_default();
    let mut params = root.join("dashboard/risk_radar_params_20261116.json");
    if is_params_name(&nm) && root.join("dashboard").join(&nm).exists() {
        params = root.join("dashboard").join(&nm);
    }
    let p = py::loads(&read_text(&params).ok_or_else(|| PyErr(format!("FileNotFoundError: [Errno 2] No such file or directory: '{}'", params.display())))?)?;
    let updated = item(&d, "updated")?;
    let Value::String(us) = updated else { return err("TypeError", format!("strptime() argument 1 must be str, not {}", py::type_name(updated))) };
    let age = now - timegm_z(us)? as f64;
    let mut coins = Map::new();
    let pc = item(&p, "coins")?;
    if let Some(c) = py::get(&d, "coins")? {
        let Value::Object(c) = c else { return err("AttributeError", format!("'{}' object has no attribute 'items'", py::type_name(c))) };
        for (k, o) in c {
            let Value::Object(o) = o else { return err("TypeError", format!("'{}' object is not a mapping", py::type_name(o))) };
            let pk = py::get(pc, k)?.cloned().unwrap_or_else(|| json!({}));
            let mut e = Map::new();
            e.insert("touch".into(), py::get(&pk, "touch")?.cloned().unwrap_or(Value::Null));
            for (a, b) in o {
                e.insert(a.clone(), b.clone());
            }
            coins.insert(k.to_lowercase(), Value::Object(e));
        }
    }
    Ok(json!({"ok": true, "updated": updated, "age_s": f(py::round_nd(age, 1)), "stale": age > 180.0,
              "touch_k": py::get(&p, "touch_k")?.cloned().unwrap_or(Value::Null), "coins": coins}))
}

/// GET /api/risk-radar (server.py:2799 → risk_radar.risk_radar_payload)
fn risk_radar(cfg: &Config, _: &Req) -> Resp {
    guarded(risk_radar_payload(&cfg.root, (cfg.now)()))
}

// ───────────────────────── 레벨 5분 (dashboard/sr_lean.py) ─────────────────────────

const SR_TAIL: u64 = 262_144;
const SR_SHOW_S: f64 = 48.0 * 3600.0;

fn sr_lean_payload(d: &Path, now: f64) -> R<V> {
    let Some(st) = read_text(&d.join("ethusdt_state.json")).and_then(|s| py::loads(&s).ok()) else {
        return Ok(json!({"ok": false, "reason": "워커 기록 없음"}));
    };
    // ev = {id: 사건} (dict: 같은 id 는 자리 그대로 값만 바뀜)
    let mut ev: Vec<(String, Map<String, V>)> = Vec::new();
    let mut late = 0i64;
    let log = d.join("ethusdt.jsonl");
    if log.exists() {
        for ln in tail_lines(&log, SR_TAIL).unwrap_or_default() {
            let mut line = || -> R<()> {
                let r = py::loads(&ln)?;
                let k = item(&r, "k")?;
                let lag_pos = || -> R<bool> { py::gt(py::get(&r, "lag")?.unwrap_or(&json!(0)), &json!(0)) };
                if k == "ev" && lag_pos()? {
                    late += 1;
                } else if k == "ev" {
                    let mut e = Map::new();
                    for f in ["a", "d", "side", "kind", "look", "L", "P", "grp"] {
                        e.insert(f.into(), item(&r, f)?.clone());
                    }
                    let id = item(&r, "id")?;
                    if matches!(id, Value::Object(_)) {
                        return err("TypeError", "unhashable type: 'dict'");
                    }
                    e.insert("y".into(), Value::Null);
                    e.insert("hit_s".into(), Value::Null);
                    let key = py::hkey(id);
                    match ev.iter_mut().find(|(k, _)| *k == key) {
                        Some(slot) => slot.1 = e,
                        None => ev.push((key, e)),
                    }
                } else if k == "res" {
                    let key = py::hkey(item(&r, "id")?);
                    if let Some(slot) = ev.iter_mut().find(|(k, _)| *k == key) {
                        let (y, h) = (item(&r, "y")?.clone(), item(&r, "hit_s")?.clone());
                        slot.1.insert("y".into(), y);
                        slot.1.insert("hit_s".into(), h);
                    }
                }
                Ok(())
            };
            match line() {
                Err(PyErr(m)) if m.starts_with("JSONDecodeError") || m.starts_with("KeyError") || m.starts_with("TypeError") => {}
                Err(e) => return Err(e),
                Ok(()) => {}
            }
        }
    }
    // sorted(ev.values(), key=(a, look)) -- 안정 정렬 · 형이 섞이면 TypeError
    let mut order: Vec<&Map<String, V>> = ev.iter().map(|(_, e)| e).collect();
    let mut bad = None;
    order.sort_by(|x, y| {
        let kx = json!([x["a"], x["look"]]);
        let ky = json!([y["a"], y["look"]]);
        match (py::lt(&kx, &ky), py::lt(&ky, &kx)) {
            (Ok(true), _) => std::cmp::Ordering::Less,
            (Ok(false), Ok(true)) => std::cmp::Ordering::Greater,
            (Ok(false), Ok(false)) => std::cmp::Ordering::Equal,
            (Err(e), _) | (_, Err(e)) => {
                bad = Some(e);
                std::cmp::Ordering::Equal
            }
        }
    });
    if let Some(e) = bad {
        return Err(e);
    }
    let mut seen = std::collections::HashSet::new();
    let mut shown: Vec<&Map<String, V>> = Vec::new();
    for e in order {
        if e["grp"] != "mid" && seen.insert(py::hkey(&json!([e["a"], e["side"], e["grp"]]))) {
            shown.push(e);
        }
    }
    let done: Vec<&&Map<String, V>> = shown.iter().filter(|e| !e["y"].is_null()).collect();
    let ok: Vec<i64> = done.iter().map(|e| i64::from(py::eq(&e["y"], &json!(1)) == (e["grp"] == "top"))).collect();
    let default_rate = json!({"top": [989, 1752], "bot": [1269, 2804]});
    let rate = py::get(&st, "rate")?.filter(|v| py::truthy(v)).unwrap_or(&default_rate);
    // 원본: round(100 * a / b) · round(100 * (1 - a / b))
    let pct = |k: &str, invert: bool| -> R<i64> {
        let r = item(rate, k)?;
        let at = |j: usize| match r {
            Value::Array(a) => a.get(j).map_or_else(|| err("IndexError", "list index out of range"), Ok),
            _ => err("TypeError", format!("'{}' object is not subscriptable", py::type_name(r))),
        };
        if invert { py::round0(100.0 * (1.0 - py::div(at(0)?, at(1)?)?)) } else { py::round0(py::div(&py::mul(&json!(100), at(0)?)?, at(1)?)?) }
    };
    let (top, bot) = (pct("top", false)?, pct("bot", true)?);
    let done_at = py::get(&st, "done")?.filter(|v| py::truthy(v)).cloned().unwrap_or(json!(0));
    let stale = py::gt(&py::sub(&f(now), &done_at)?, &json!(45))?;
    let cut = f(now - SR_SHOW_S);
    let mut ev_out = Vec::new();
    for e in &shown {
        if py::ge(&e["a"], &cut)? {
            ev_out.push(Value::Object((*e).clone()));
        }
    }
    let mut mid48 = 0;
    for (_, e) in &ev {
        if e["grp"] == "mid" && py::ge(&e["a"], &cut)? {
            mid48 += 1;
        }
    }
    let skip = done.len().saturating_sub(30);
    let mut recent = Vec::new();
    for (e, o) in done[skip..].iter().zip(&ok[ok.len().saturating_sub(30)..]) {
        recent.push(json!([py::mul(&e["d"], &json!(1000))?, o]));
    }
    Ok(json!({"ok": true, "now": f(now), "stale": stale, "warm_until": py::get(&st, "warm_until")?.cloned().unwrap_or(Value::Null),
              "pct": {"top": top, "bot": bot}, "ev": ev_out, "late": late, "mid48": mid48,
              "rec": {"n": ok.len(), "hit": ok.iter().sum::<i64>(), "recent": recent}}))
}

/// GET /api/sr-lean (server.py:2792 → sr_lean.sr_lean_payload)
fn sr_lean(cfg: &Config, _: &Req) -> Resp {
    guarded(sr_lean_payload(&cfg.live_dir().join("sr_lean"), (cfg.now)()))
}

// ───────────────────────── 횡보·추세 띠 (dashboard/regime_band.py) ─────────────────────────

/// `int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)` -- 저장소 크론이 쓰는 꼴(`YYYY-MM-DD[ T]HH:MM[:SS[.f]]`·날짜만)만.
/// ponytail: pandas 가 읽는 다른 꼴(`2026/11/16`·숫자 ns·시간대 붙은 글자)은 «깨진 줄»로 건너뛴다 -- 크론이 그 꼴을 안 쓴다.
fn pd_ts_ms(v: &V) -> Option<i64> {
    let Value::String(s) = v else { return None };
    let s = s.trim();
    let (date, time) = match s.find([' ', 'T']) {
        Some(k) => (&s[..k], &s[k + 1..]),
        None => (s, ""),
    };
    let dp: Vec<&str> = date.split('-').collect();
    if dp.len() != 3 || dp[0].len() != 4 || dp[1].len() != 2 || dp[2].len() != 2 {
        return None;
    }
    let num = |t: &str| t.bytes().all(|c| c.is_ascii_digit()).then(|| t.parse::<i64>().ok()).flatten();
    let (y, m, d) = (num(dp[0])?, num(dp[1])?, num(dp[2])?);
    if !(1..=12).contains(&m) || d < 1 || d > days_in_month(y, m) {
        return None;
    }
    let (mut hh, mut mm, mut ss, mut ns) = (0, 0, 0, 0i64);
    if !time.is_empty() {
        let (hms, frac) = time.split_once('.').unwrap_or((time, ""));
        let tp: Vec<&str> = hms.split(':').collect();
        if !(2..=3).contains(&tp.len()) || tp.iter().any(|t| t.len() != 2) {
            return None;
        }
        hh = num(tp[0])?;
        mm = num(tp[1])?;
        ss = if tp.len() == 3 { num(tp[2])? } else { 0 };
        if hh > 23 || mm > 59 || ss > 59 || (!frac.is_empty() && (frac.len() > 9 || num(frac).is_none())) {
            return None;
        }
        if !frac.is_empty() {
            ns = format!("{frac:0<9}").parse().ok()?;
        }
    }
    let secs = days_from_civil(y, m, d) * 86400 + hh * 3600 + mm * 60 + ss;
    // Timestamp.timestamp() = round(ns / 1e9, 6)
    let t = py::round_nd((secs as i128 * 1_000_000_000 + ns as i128) as f64 / 1e9, 6);
    Some((t * 1000.0).trunc() as i64)
}

fn regime_band_payload(log: &Path) -> R<V> {
    if !log.exists() {
        return Ok(json!({"ok": false, "reason": "기록 없음(전진 기록 크론 대기)"}));
    }
    let lines = tail_lines(log, 32_768).ok_or_else(|| PyErr("OSError: read".into()))?;
    let mut coins = Map::new();
    for ln in lines {
        let row = || -> R<(String, V)> {
            let r = py::loads(&ln)?;
            let coin = py::to_str(item(&r, "coin")?).to_lowercase();
            let ts = pd_ts_ms(item(&r, "ts")?).map_or_else(|| err("ValueError", "ts"), Ok)?;
            let mut o = Map::new();
            o.insert("decided_at_ms".into(), i(ts));
            o.insert("p_trend".into(), f(py::to_float(item(&r, "p_trend")?)?));
            o.insert("regime".into(), item(&r, "regime")?.clone());
            o.insert("confidence".into(), f(py::to_float(item(&r, "confidence")?)?));
            o.insert("support".into(), f(py::to_float(item(&r, "support")?)?));
            o.insert("resistance".into(), f(py::to_float(item(&r, "resistance")?)?));
            Ok((coin, Value::Object(o)))
        };
        if let Ok((c, o)) = row() {
            coins.insert(c, o);
        }
    }
    Ok(if coins.is_empty() { json!({"ok": false, "reason": "읽을 수 있는 기록 없음"}) } else { json!({"ok": true, "coins": coins}) })
}

/// GET /api/regime-band (server.py:2785 → regime_band.regime_band_payload)
fn regime_band(cfg: &Config, _: &Req) -> Resp {
    guarded(regime_band_payload(&cfg.live_dir().join("regime_band_forward.jsonl")))
}

// ───────────────────────── 청산 탭 (server.py:2808) ─────────────────────────

/// 원본 _query_coin_asset(server.py:4742): 소문자로 · COIN_CONFIG 밖이면 400 unsupported_asset
fn coin_asset(req: &Req) -> Result<String, Resp> {
    let a = req.q("asset").unwrap_or("eth").to_lowercase();
    if COIN_CONFIG.contains(&a.as_str()) { Ok(a) } else { Err(http_exc(400, Some("unsupported_asset"), &[])) }
}

/// GET /api/liq-tab -- 크론이 쓴 파일 바이트를 그대로(application/json · 압축)
fn liq_tab(cfg: &Config, req: &Req) -> Resp {
    let asset = if req.q("asset") == Some("all") { "all".to_string() } else {
        match coin_asset(req) {
            Ok(a) => a,
            Err(r) => return r,
        }
    };
    match fs::read(cfg.live_dir().join(format!("liq_tab_{asset}.json"))) {
        Ok(body) => {
            let mut r = Resp::bytes("application/json", body);
            r.compress = true;
            r
        }
        Err(_) => Resp::json(&json!({"ok": false, "reason": "not_built"})),
    }
}

// ───────────────────────── 아이스버그 (server.py:2958) ─────────────────────────

/// GET /api/iceberg -- 원본은 events.jsonl 의 깨진 줄·빈 줄(ValueError)·열쇠 빠짐을 안 잡아 500 이 난다(그대로 둔다)
fn iceberg(cfg: &Config, req: &Req) -> Resp {
    let asset = req.q("asset").unwrap_or("eth");
    if asset != "eth" && asset != "xrp" {
        return http_exc(400, Some("asset"), &[]);
    }
    let d = cfg.live_dir().join("iceberg");
    let now = (cfg.now)();
    let Some(st) = read_text(&d.join(format!("{asset}_state.json"))).and_then(|s| py::loads(&s).ok()) else {
        return Resp::json(&json!({"ok": false, "reason": "no_state"}));
    };
    or500((|| {
        let cut = f(now - 86400.0);
        let mut ev = Vec::new();
        let p = d.join(format!("{asset}_events.jsonl"));
        match fs::read(&p) {
            Ok(b) => {
                let s = String::from_utf8(b).map_err(|_| PyErr("UnicodeDecodeError".into()))?;
                for ln in py::splitlines(&s) {
                    let e = py::loads(ln)?;
                    if py::ge(item(&e, "t_end")?, &cut)? {
                        ev.push(e);
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return err("OSError", e.to_string()),
        }
        let age = py::sub(&f(now), item(&st, "updated_at")?)?;
        Ok(json!({"ok": true, "n": item(&st, "n_forward")?, "target": item(&st, "target")?, "age_s": py::round_v(&age, 1)?,
                  "valid": item(&st, "valid")?, "events": ev}))
    })())
}

// ───────────────────────── SQLite 읽기 (scripts/data_store.read_rows · .sqlite = mode=ro) ─────────────────────────

fn sql_v(v: Sql) -> V {
    match v {
        Sql::Null => Value::Null,
        Sql::Integer(x) => i(x),
        Sql::Real(x) => f(x),
        Sql::Text(s) => Value::String(s),
        Sql::Blob(b) => Value::String(String::from_utf8_lossy(&b).into_owned()), // ponytail: 파이썬은 bytes -- 이 표들에 BLOB 칸이 없다
    }
}

fn rows(db: &Path, sql: &str, params: &[Sql]) -> Result<Vec<Vec<V>>, rusqlite::Error> {
    let con = rusqlite::Connection::open_with_flags(db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX)?;
    con.busy_timeout(std::time::Duration::from_secs(10))?;
    let mut st = con.prepare(sql)?;
    let n = st.column_count();
    let out = st
        .query_map(rusqlite::params_from_iter(params.iter()), |r| (0..n).map(|j| r.get::<_, Sql>(j).map(sql_v)).collect::<Result<Vec<_>, _>>())?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(out)
}

/// 파이썬 `int(v)`(SQLite 값: 정수·실수)
fn int_v(v: &V) -> R<i64> {
    match v {
        Value::Number(n) => n.as_i64().map_or_else(|| Ok(n.as_f64().unwrap().trunc() as i64), Ok),
        Value::String(s) => s.trim().parse().or_else(|_| err("ValueError", format!("invalid literal for int() with base 10: '{s}'"))),
        _ => err("TypeError", format!("int() argument must be a string, a bytes-like object or a real number, not '{}'", py::type_name(v))),
    }
}

fn hot(cfg: &Config, name: &str) -> PathBuf {
    cfg.root.join("data/hot").join(name)
}

/// 원본 treasury_payload(server.py:596)
fn treasury_payload(db: &Path, hours: f64, show_h: f64) -> R<V> {
    let Ok(rs) = rows(db, "SELECT symbol, ts_sec, close, contract FROM treasury_fut_1m
                                      WHERE close IS NOT NULL AND ts_sec >= (SELECT max(ts_sec) FROM treasury_fut_1m) - ?
                                      ORDER BY symbol, ts_sec", &[Sql::Integer((hours * 3600.0) as i64)]) else {
        return Ok(json!({"ok": false, "symbols": {}}));
    };
    let mut out: Vec<(String, V, Vec<(i64, f64)>)> = Vec::new();
    for r in rs {
        let sym = py::to_str(&r[0]);
        let bar = (int_v(&r[1])?, py::to_float(&r[2])?);
        match out.iter_mut().find(|(s, _, _)| *s == sym) {
            Some(e) => e.2.push(bar),
            None => out.push((sym, r[3].clone(), vec![bar])),
        }
    }
    let mut syms = Map::new();
    for (sym, contract, b) in &out {
        let (first, last) = (b[0], b[b.len() - 1]);
        if first.1 == 0.0 {
            return err("ZeroDivisionError", "float division by zero");
        }
        let chg = (last.1 / first.1 - 1.0) * 100.0;
        let lo = last.0 as f64 - show_h * 3600.0;
        let bars: Vec<V> = b.iter().filter(|x| x.0 as f64 >= lo).map(|x| json!([x.0, f(x.1)])).collect();
        syms.insert(sym.clone(), json!({"contract": contract, "bars": bars, "chg12_pct": f(chg)}));
    }
    Ok(json!({"ok": !syms.is_empty(), "symbols": syms}))
}

/// GET /api/treasury-futures (server.py:2110) -- ?h=24 면 24시간 창·24시간 표시
fn treasury_futures(cfg: &Config, req: &Req) -> Resp {
    let db = hot(cfg, "treasury_futures.sqlite");
    or500(if req.q("h") == Some("24") { treasury_payload(&db, 24.0, 24.0) } else { treasury_payload(&db, 12.0, 1.0) })
}

const XYZ_COIN: &str = "xyz:XYZ100";

/// 원본 xyz100_payload(server.py:622)
fn xyz100_payload(db: &Path, since: i64, now: f64) -> R<V> {
    let now_sec = now.trunc() as i64;
    let q = || -> Result<(Vec<Vec<V>>, Vec<Vec<V>>), rusqlite::Error> {
        let b = rows(db, "SELECT ts_sec, close FROM hl_xyz_1m WHERE coin = ? AND close IS NOT NULL
                                      AND ts_sec >= max(?, (SELECT max(ts_sec) FROM hl_xyz_1m WHERE coin = ?) - ?) ORDER BY ts_sec",
            &[Sql::Text(XYZ_COIN.into()), Sql::Integer(since), Sql::Text(XYZ_COIN.into()), Sql::Integer(24 * 3600)])?;
        let l = rows(db, "SELECT recv_sec, mid, mark, oracle FROM hl_xyz_ctx_1s WHERE coin = ? ORDER BY recv_sec DESC LIMIT 1", &[Sql::Text(XYZ_COIN.into())])?;
        Ok((b, l))
    };
    let Ok((b, live)) = q() else { return Ok(json!({"ok": false, "bars": [], "live": null, "now_sec": now_sec})) };
    let lv = match live.first() {
        Some(r) => {
            let px = if r[1].is_null() { &r[2] } else { &r[1] };
            json!({"recv_sec": int_v(&r[0])?, "px": if px.is_null() { Value::Null } else { f(py::to_float(px)?) },
                   "oracle": if r[3].is_null() { Value::Null } else { f(py::to_float(&r[3])?) }})
        }
        None => Value::Null,
    };
    let bars = b.iter().map(|r| Ok(json!([int_v(&r[0])?, f(py::to_float(&r[1])?)]))).collect::<R<Vec<_>>>()?;
    Ok(json!({"ok": !bars.is_empty() || !lv.is_null(), "coin": XYZ_COIN, "full": since == 0, "bars": bars, "live": lv, "now_sec": now_sec}))
}

/// 파이썬 `int(s)`(공백·`+`·`_` 허용) → 실패면 None
fn py_int_str(s: &str) -> Option<i128> {
    let t = s.trim();
    let (neg, body) = match t.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    if body.is_empty() || body.starts_with('_') || body.ends_with('_') || body.contains("__") || !body.bytes().all(|c| c.is_ascii_digit() || c == b'_') {
        return None;
    }
    let digits: String = body.chars().filter(|c| *c != '_').collect();
    let v: i128 = digits.parse().unwrap_or(i128::MAX); // 아주 긴 수도 위에서 2**40 으로 잘린다
    Some(if neg { -v } else { v })
}

/// GET /api/xyz100 (server.py:2121) -- since = min(max(0, int(since or 0)), 2**40) · 못 읽으면 0
fn xyz100(cfg: &Config, req: &Req) -> Resp {
    let raw = req.q("since").filter(|s| !s.is_empty()).unwrap_or("0");
    let since = py_int_str(raw).unwrap_or(0).clamp(0, 1 << 40) as i64;
    or500(xyz100_payload(&hot(cfg, "hl_xyz.sqlite"), since, (cfg.now)()))
}

/// 원본 macro_today_payload(server.py:646)
fn macro_today_payload(xyz: &Path, m5: &Path, now: f64) -> R<V> {
    let now = now.trunc() as i64;
    let mut out = Map::new();
    if let Ok((b, lv)) = (|| -> Result<_, rusqlite::Error> {
        let b = rows(xyz, "SELECT ts_sec + 60, close FROM hl_xyz_1m WHERE coin = 'xyz:SP500' AND close IS NOT NULL AND ts_sec + 60 <= ?
                                       AND ts_sec >= (SELECT max(ts_sec) FROM hl_xyz_1m WHERE coin = 'xyz:SP500') - ? ORDER BY ts_sec",
            &[Sql::Integer(now), Sql::Integer(24 * 3600)])?;
        let lv = rows(xyz, "SELECT recv_sec, coalesce(mid, mark) FROM hl_xyz_ctx_1s WHERE coin = 'xyz:SP500' ORDER BY recv_sec DESC LIMIT 1", &[])?;
        Ok((b, lv))
    })() {
        if !b.is_empty() || !lv.is_empty() {
            let bars = b.iter().map(|r| Ok(json!([int_v(&r[0])?, f(py::to_float(&r[1])?)]))).collect::<R<Vec<_>>>()?;
            let live = match lv.first() {
                Some(r) if !r[1].is_null() => json!({"recv_sec": int_v(&r[0])?, "px": f(py::to_float(&r[1])?)}),
                _ => Value::Null,
            };
            out.insert("SP".into(), json!({"bars": bars, "live": live}));
        }
    }
    if let Ok(b) = rows(m5, "SELECT ts_sec + 300, close FROM yahoo_5m WHERE symbol = 'VIX' AND close IS NOT NULL
                                      AND ts_sec >= (SELECT max(ts_sec) FROM yahoo_5m WHERE symbol = 'VIX') - ? ORDER BY ts_sec", &[Sql::Integer(24 * 3600)]) {
        if !b.is_empty() {
            let bars = b.iter().map(|r| Ok(json!([int_v(&r[0])?, f(py::to_float(&r[1])?)]))).collect::<R<Vec<_>>>()?;
            out.insert("VIX".into(), json!({"bars": bars, "live": null}));
        }
    }
    Ok(json!({"ok": !out.is_empty(), "symbols": out, "now_sec": now}))
}

/// GET /api/macro-today (server.py:2117)
fn macro_today(cfg: &Config, _: &Req) -> Resp {
    or500(macro_today_payload(&hot(cfg, "hl_xyz.sqlite"), &hot(cfg, "macro_5m.sqlite"), (cfg.now)()))
}

// ───────────────────────── 5분 호가 요약 (server.py:2982) ─────────────────────────

fn ob5m_payload(db: &Path, now: f64) -> R<V> {
    let empty = json!({"bars": [], "walls": [], "marks": []});
    if !db.exists() {
        return Ok(empty);
    }
    let sqlerr = |e: rusqlite::Error| PyErr(format!("sqlite3.Error: {e}"));
    let cut = ((now - 7.0 * 86400.0) * 1000.0).trunc() as i64;
    let mut rs = rows(db, "SELECT t, warm, imb50, eat_n_a, wall_px_b, wall_usd_b, wall_px_a, wall_usd_a FROM bars WHERE t >= ? ORDER BY t", &[Sql::Integer(cut)]).map_err(sqlerr)?;
    let lv = rows(db, "SELECT upd, bar, marks FROM live WHERE k = 0 AND upd >= ?", &[Sql::Integer((now * 1000.0).trunc() as i64 - 30_000)]).unwrap_or_default();
    let text = |v: &V| -> R<V> {
        match v {
            Value::String(s) => py::loads(s),
            _ => err("TypeError", "the JSON object must be str, bytes or bytearray"),
        }
    };
    let (lb, lmarks) = match lv.first() {
        Some(r) => (text(&r[1])?, text(&r[2])?),
        None => (Value::Null, json!([])),
    };
    let mut live_t = Value::Null;
    if py::truthy(&lb) && (rs.is_empty() || py::gt(item(&lb, "t")?, &rs[rs.len() - 1][0])?) {
        let g = |k: &str| item(&lb, k).cloned();
        rs.push(vec![g("t")?, g("warm")?, g("imb50")?, Value::Null, g("wall_px_b")?, g("wall_usd_b")?, g("wall_px_a")?, g("wall_usd_a")?]);
        live_t = g("t")?;
    }
    let (mut bars, mut walls) = (Vec::new(), Vec::new());
    let mut run: Vec<(&str, Map<String, V>)> = Vec::new(); // dict: pop 뒤 다시 넣으면 끝으로
    for r in &rs {
        let (t, warm, imb, ev) = (&r[0], &r[1], &r[2], &r[3]);
        let mut b = Map::new();
        b.insert("t".into(), t.clone());
        b.insert("tilt_pct".into(), if imb.is_null() { Value::Null } else { py::round_v(&py::mul(imb, &json!(100))?, 2)? });
        b.insert("book_ok".into(), json!(!py::truthy(warm) && !imb.is_null()));
        b.insert("events_known".into(), json!(!ev.is_null()));
        if !live_t.is_null() && py::eq(t, &live_t) {
            b.insert("live".into(), json!(true));
        }
        bars.push(b);
        for (side, px, usd) in [("bid", &r[4], &r[5]), ("ask", &r[6], &r[7])] {
            let usd0 = if py::truthy(usd) { usd.clone() } else { f(0.0) };
            let pos = run.iter().position(|(s, _)| *s == side);
            if let Some(k) = pos {
                let w = &mut run[k].1;
                if !px.is_null() && py::eq(&w["px"], px) && py::eq(&w["t_end"], t) {
                    w.insert("t_end".into(), py::add(t, &json!(300_000))?);
                    let cur = w["usd_max"].clone();
                    if py::gt(&usd0, &cur)? {
                        w.insert("usd_max".into(), usd0);
                    }
                    continue;
                }
                walls.push(Value::Object(run.remove(k).1));
            }
            if !px.is_null() {
                let mut w = Map::new();
                w.insert("side".into(), json!(side));
                w.insert("px".into(), px.clone());
                w.insert("t_start".into(), t.clone());
                w.insert("t_end".into(), py::add(t, &json!(300_000))?);
                w.insert("usd_max".into(), usd0);
                run.push((side, w));
            }
        }
    }
    walls.extend(run.into_iter().map(|(_, w)| Value::Object(w)));
    let mut marks = Vec::new();
    for r in rows(db, "SELECT t, bar_t, side, kind, px, usd, dist_bp FROM marks WHERE bar_t >= ? ORDER BY t", &[Sql::Integer(cut)]).map_err(sqlerr)? {
        let usd = if py::truthy(&r[5]) { r[5].clone() } else { json!(0) };
        marks.push(json!({"t": r[0], "bar_t": r[1], "side": r[2], "kind": r[3], "px": r[4], "usd": py::round0_v(&usd)?, "dist_bp": py::round_v(&r[6], 2)?}));
    }
    let mut known: std::collections::HashSet<String> = bars.iter().filter(|b| b["events_known"] == true).map(|b| py::hkey(&b["t"])).collect();
    known.extend(marks.iter().map(|m| py::hkey(&m["bar_t"])));
    let Value::Array(lm) = &lmarks else { return err("TypeError", "not iterable") };
    for m in lm {
        if !known.contains(&py::hkey(item(m, "bar_t")?)) {
            marks.push(json!({"t": item(m, "t")?, "bar_t": item(m, "bar_t")?, "side": item(m, "side")?, "kind": item(m, "kind")?, "px": item(m, "px")?,
                              "usd": py::round0_v(item(m, "usd")?)?, "dist_bp": py::round_v(item(m, "dist_bp")?, 2)?, "live": true}));
        }
    }
    Ok(json!({"bars": bars, "walls": walls, "marks": marks}))
}

/// GET /api/ob5m -- COIN_CONFIG 밖 코인(소문자로 바꾸지 않는다)은 빈 목록 · 원본은 SQLite 오류를 안 잡는다(500)
fn ob5m(cfg: &Config, req: &Req) -> Resp {
    let asset = req.q("asset").unwrap_or("eth");
    if !COIN_CONFIG.contains(&asset) {
        return Resp::json(&json!({"bars": [], "walls": [], "marks": []}));
    }
    or500(ob5m_payload(&cfg.live_dir().join("ob5m").join(format!("{asset}usdt.sqlite")), (cfg.now)()))
}

// ───────────────────────── 거시 달력 (worker_state.worker_payload) ─────────────────────────

/// `datetime.fromisoformat(str(ts).replace(" ", "T"))`(파이썬 3.11) → UTC 마이크로초. 시간대 없으면 UTC.
/// ponytail: 3.11 이 받는 꼴 중 확장형(`YYYY-MM-DD[THH[:MM[:SS[.f]]]][Z|±HH[:MM]]`)만 -- 기본형(`20261118T1038`)·주 날짜(`2026-W47`)는 None.
fn iso_us(s: &str) -> Option<i64> {
    let s = s.replace(' ', "T");
    let (date, rest) = s.split_once('T').unwrap_or((&s, ""));
    let num = |t: &str| (!t.is_empty() && t.bytes().all(|c| c.is_ascii_digit())).then(|| t.parse::<i64>().ok()).flatten();
    let dp: Vec<&str> = date.split('-').collect();
    if dp.len() != 3 || dp[0].len() != 4 || dp[1].len() != 2 || dp[2].len() != 2 {
        return None;
    }
    let (y, m, d) = (num(dp[0])?, num(dp[1])?, num(dp[2])?);
    if !(1..=12).contains(&m) || d < 1 || d > days_in_month(y, m) {
        return None;
    }
    let (mut hh, mut mm, mut ss, mut us, mut off) = (0, 0, 0, 0, 0i64);
    if !rest.is_empty() {
        let k = rest.find(['Z', '+', '-']).unwrap_or(rest.len());
        let (t, tz) = rest.split_at(k);
        let (hms, frac) = t.split_once('.').map_or((t, None), |(a, b)| (a, Some(b)));
        let tp: Vec<&str> = hms.split(':').collect();
        if tp.is_empty() || tp.len() > 3 || tp.iter().any(|x| x.len() != 2) || (frac.is_some() && tp.len() != 3) {
            return None;
        }
        hh = num(tp[0])?;
        mm = if tp.len() > 1 { num(tp[1])? } else { 0 };
        ss = if tp.len() > 2 { num(tp[2])? } else { 0 };
        if let Some(fr) = frac {
            num(fr)?;
            us = format!("{:0<6}", &fr[..fr.len().min(6)]).parse().ok()?;
        }
        if hh > 23 || mm > 59 || ss > 59 {
            return None;
        }
        if tz == "Z" {
        } else if !tz.is_empty() {
            let sign = if tz.starts_with('-') { -1 } else { 1 };
            let z = &tz[1..];
            let (zh, zm) = match z.len() {
                2 => (num(z)?, 0),
                4 => (num(&z[..2])?, num(&z[2..])?),
                5 if &z[2..3] == ":" => (num(&z[..2])?, num(&z[3..])?),
                _ => return None,
            };
            if zh > 23 || zm > 59 {
                return None;
            }
            off = sign * (zh * 3600 + zm * 60);
        }
    }
    Some((days_from_civil(y, m, d) * 86400 + hh * 3600 + mm * 60 + ss - off) * 1_000_000 + us)
}

/// `datetime.fromtimestamp(now)` 의 마이크로초(짝수 반올림)
fn now_us(now: f64) -> i64 {
    let s = now.trunc();
    s as i64 * 1_000_000 + ((now - s) * 1e6).round_ties_even() as i64
}

/// 원본 worker_payload(path, 90.0, ts_field="generated_at", extra_missing={"events": []}) -- macro_calendar_payload(server.py:1150)
fn macro_calendar_payload(path: &Path, now: f64) -> R<V> {
    let base = [("available", json!(false)), ("error", json!("worker_state_missing")), ("tone", json!("neutral")), ("subText", json!("데이터 없음"))];
    let st = if path.exists() { py::loads(&read_text(path).ok_or_else(|| PyErr("UnicodeDecodeError".into()))?)? } else { Value::Null };
    if !py::truthy(&st) {
        let mut o: Map<String, V> = base.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
        o.insert("events".into(), json!([]));
        return Ok(Value::Object(o));
    }
    let ts = py::get(&st, "generated_at")?.cloned().unwrap_or(Value::Null);
    let age = iso_us(&py::to_str(&ts)).map(|t| py::round_nd((now_us(now) - t) as f64 / 1e6 / 60.0, 1));
    let stale = age.map_or(Value::Null, |a| f(py::round_nd(a, 1)));
    let Value::Object(mut o) = st else { unreachable!() };
    if age.is_some_and(|a| a > 90.0) {
        for (k, v) in base {
            o.insert(k.into(), v);
        }
        o.insert("error".into(), json!("worker_stale"));
        o.insert("stale_min".into(), stale);
        return Ok(Value::Object(o));
    }
    o.insert("stale_min".into(), stale);
    Ok(Value::Object(o))
}

/// GET /api/macro-calendar (server.py:4836 → load_macro_calendar → macro_calendar_payload) -- 원본은 예외를 안 잡는다(500)
fn macro_calendar(cfg: &Config, _: &Req) -> Resp {
    or500(macro_calendar_payload(&cfg.live_dir().join("macro_calendar_state.json"), (cfg.now)()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates() {
        assert_eq!(timegm_z("2026-11-15T00:00:00Z").unwrap(), 1_794_700_800);
        assert!(timegm_z("2026-11-15 00:00:00").is_err());
        assert_eq!(pd_ts_ms(&json!("2026-11-01 20:00:00")), Some(1_793_563_200_000));
        assert_eq!(iso_us("2026-11-01T20:00:00+09:00"), Some((1_793_563_200 - 9 * 3600) * 1_000_000));
        assert_eq!(iso_us("2026-11-01"), Some(1_793_491_200 * 1_000_000));
        assert_eq!(py_int_str(" +1_000 "), Some(1000));
        assert_eq!(py_int_str("1__0"), None);
    }
}
