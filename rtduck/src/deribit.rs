//! Deribit 옵션 체결·블록 = scripts/live_deribit_block_trade_collector_20260928.py 의 첫 단계(체결 WS · REST 백필 · option_trades/gaps · 블록 구조 · 상태 JSON).
//!   GEX 모듈(collect_deribit_option_gex_20260815)에서는 블록·흐름 그릭스가 쓰는 `_parse_instrument`·`_bs_delta` 만 값 보존으로 옮겼다(erf = libm -- 파이썬 math.erf 와 같다).
//!   🔴안 옮김(다음 단계): 10분 체인·GEX·옵션 요약(deribit_gex_state.json · option_chain_snapshot 등) · 시간당 history 대조(block_rfq_trades·block_future_legs 채우기) · 옛 파일 옮기기.
//!   그래서 hourly_doi 는 체인 표가 없으면 {}(원본도 표가 없으면 {}) -- 🔴체인 표가 있는 DB 를 가리키면 원본은 ΔOI 를 내고 Rust 는 None(아직).
use duckdb::{params, Connection};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::Path;

pub const WS: &str = "wss://www.deribit.com/ws/api/v2";
pub const REST: &str = "https://www.deribit.com/api/v2/public/get_last_trades_by_currency_and_time";
pub const API_CURRENCIES: [&str; 3] = ["ETH", "BTC", "USDC"];
pub const COINS: [&str; 4] = ["ETH", "BTC", "SOL", "XRP"];
const COIN_PREFIX: [(&str, &str); 4] = [("ETH-", "ETH"), ("BTC-", "BTC"), ("SOL_USDC-", "SOL"), ("XRP_USDC-", "XRP")];
pub const STATE_WINDOW_MS: i64 = 24 * 3600 * 1000;
pub const BACKFILL_MAX_MS: i64 = STATE_WINDOW_MS;
pub const FLUSH_SEC: f64 = 20.0;
pub const IDLE_SEC: f64 = 90.0;
pub const COLS: [&str; 18] = ["ts_ms", "trade_id", "trade_seq", "instrument_name", "direction", "amount", "price", "mark_price", "index_price", "iv", "tick_direction",
                              "liquidation", "is_block", "block_trade_id", "block_trade_leg_count", "block_rfq_id", "combo_id", "combo_trade_id"];

pub fn channels() -> Vec<String> {
    API_CURRENCIES.iter().map(|c| format!("trades.option.{c}.100ms")).collect()
}

pub fn coin_of(inst: &str) -> Option<&'static str> {
    COIN_PREFIX.iter().find(|(p, _)| inst.starts_with(p)).map(|x| x.1)
}

extern "C" {
    fn erf(x: f64) -> f64;
}

/// GEX `_parse_instrument`: ^[A-Z_]+-<d{1,2}MON yy>-<strike>-[CP]$ → (행사가, 만기 08:00 UTC 초, 콜?)
pub fn parse_instrument(name: &str) -> Option<(f64, f64, bool)> {
    let p: Vec<&str> = name.split('-').collect();
    if p.len() != 4 || p[0].is_empty() || !p[0].bytes().all(|b| b.is_ascii_uppercase() || b == b'_') {
        return None;
    }
    let (e, s, t) = (p[1], p[2], p[3]);
    // exp = \d{1,2}[A-Z]{3}\d{2}
    let nd = e.bytes().take_while(u8::is_ascii_digit).count();
    if !(1..=2).contains(&nd) || e.len() != nd + 5 || !e[nd..nd + 3].bytes().all(|b| b.is_ascii_uppercase()) || !e[nd + 3..].bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    // strike = \d+(?:[.d]\d+)?
    let sd = s.bytes().take_while(u8::is_ascii_digit).count();
    let ok = sd > 0 && (sd == s.len() || (matches!(s.as_bytes()[sd], b'.' | b'd') && s.len() > sd + 1 && s[sd + 1..].bytes().all(|b| b.is_ascii_digit())));
    if !ok || !(t == "C" || t == "P") {
        return None;
    }
    // strptime("%d%b%y") -- 없는 날짜·달 이름이면 ValueError(원본은 그대로 예외 -- 부르는 쪽이 잡는다) → None 으로(여기 부르는 곳은 모두 예외를 «구조 없음»으로 삼킨다)
    let day: i64 = e[..nd].parse().ok()?;
    let mon = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"].iter().position(|m| *m == &e[nd..nd + 3])? as i64 + 1;
    let yy: i64 = e[nd + 3..].parse().ok()?;
    let y = if yy < 69 { 2000 + yy } else { 1900 + yy };
    let dim = match mon {
        2 if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    if day < 1 || day > dim {
        return None;
    }
    let strike: f64 = s.replace('d', ".").parse().ok()?;
    Some((strike, (days_from_civil(y, mon, day) * 86_400 + 8 * 3600) as f64, t == "C"))
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
}

/// GEX `_bs_delta`
pub fn bs_delta(fwd: f64, strike: f64, iv_pct: f64, years: f64, call: bool) -> f64 {
    let sigma = iv_pct / 100.0;
    if fwd <= 0.0 || strike <= 0.0 || sigma <= 0.0 || years <= 0.0 {
        return 0.0;
    }
    let d1 = ((fwd / strike).ln() + 0.5 * sigma * sigma * years) / (sigma * years.sqrt());
    let nd1 = 0.5 * (1.0 + unsafe { erf(d1 / 2f64.sqrt()) });
    if call { nd1 } else { nd1 - 1.0 }
}

fn pyround(x: f64, n: usize) -> f64 {
    format!("{x:.n$}").parse().unwrap()
}

/// 파이썬 float(x)
fn fnum(v: &Value) -> Result<f64, String> {
    match v {
        Value::Number(n) => Ok(n.as_f64().unwrap()),
        Value::String(s) => s.trim().parse().map_err(|_| format!("ValueError: float({s:?})")),
        Value::Bool(b) => Ok(f64::from(u8::from(*b))),
        o => Err(format!("TypeError: float({o})")),
    }
}

/// `row(t)` -- DuckDB 에 넣을 값(파이썬 값 → DuckDB 형 변환은 원본 executemany 와 같게).
pub fn row(t: &Value) -> Result<Vec<duckdb::types::Value>, String> {
    use duckdb::types::Value as D;
    let g = |k: &str| t.get(k).filter(|v| !v.is_null());
    let need = |k: &str| t.get(k).ok_or_else(|| format!("KeyError: '{k}'"));
    let int = |v: Option<&Value>| -> Result<D, String> {
        Ok(match v {
            None => D::Null,
            Some(Value::Number(n)) => D::BigInt(n.as_i64().unwrap_or(n.as_f64().unwrap() as i64)),
            Some(Value::Bool(b)) => D::BigInt(i64::from(*b)),
            Some(o) => return Err(format!("형 {o} → BIGINT")),
        })
    };
    let dbl = |v: Option<&Value>| -> Result<D, String> { Ok(match v { None => D::Null, Some(x) => D::Double(fnum(x)?) }) };
    let txt = |v: Option<&Value>| -> D {
        match v {
            None => D::Null,
            Some(Value::String(s)) => D::Text(s.clone()),
            Some(o) => D::Text(o.to_string()),
        }
    };
    let s = |v: &Value| match v {
        Value::String(s) => s.clone(),
        Value::Number(n) if n.is_i64() || n.is_u64() => n.to_string(),
        Value::Number(n) => rtcollect::py_float_repr(n.as_f64().unwrap()),
        o => o.to_string(),
    };
    let ts = match need("timestamp")? {
        Value::Number(n) => n.as_i64().ok_or("ValueError: int")?,
        Value::String(x) => x.trim().parse().map_err(|_| "ValueError: int")?,
        _ => return Err("TypeError: int".into()),
    };
    let b = g("block_trade_id");
    Ok(vec![D::BigInt(ts), D::Text(s(need("trade_id")?)), int(g("trade_seq"))?, D::Text(need("instrument_name")?.as_str().ok_or("TypeError")?.to_string()),
            D::Text(need("direction")?.as_str().ok_or("TypeError")?.to_string()), D::Double(fnum(need("amount")?)?), D::Double(fnum(need("price")?)?),
            dbl(g("mark_price"))?, dbl(g("index_price"))?, dbl(g("iv"))?, int(g("tick_direction"))?, txt(g("liquidation")), D::Boolean(b.is_some()), txt(b),
            int(g("block_trade_leg_count"))?, g("block_rfq_id").map_or(D::Null, |v| D::Text(s(v))), txt(g("combo_id")), txt(g("combo_trade_id"))])
}

/// `_connect`(표 넷 -- 글자 그대로) · 잠겨 있으면 2초씩 15번.
pub fn connect(db: &Path) -> Result<Connection, String> {
    if let Some(d) = db.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let mut last = String::new();
    for i in 0..15 {
        match Connection::open(db) {
            Ok(con) => {
                con.execute_batch("CREATE TABLE IF NOT EXISTS option_trades (
        ts_ms BIGINT, trade_id VARCHAR PRIMARY KEY, trade_seq BIGINT, instrument_name VARCHAR,
        direction VARCHAR, amount DOUBLE, price DOUBLE, mark_price DOUBLE, index_price DOUBLE,
        iv DOUBLE, tick_direction INTEGER, liquidation VARCHAR, is_block BOOLEAN,
        block_trade_id VARCHAR, block_trade_leg_count INTEGER, block_rfq_id VARCHAR,
        combo_id VARCHAR, combo_trade_id VARCHAR);
    CREATE TABLE IF NOT EXISTS gaps (
        from_ms BIGINT, to_ms BIGINT, reason VARCHAR, backfilled INTEGER);
    CREATE TABLE IF NOT EXISTS block_rfq_trades (
        rfq_id VARCHAR PRIMARY KEY, ts_ms BIGINT, api VARCHAR, direction VARCHAR, amount DOUBLE, price DOUBLE,
        mark_price DOUBLE, combo_id VARCHAR, legs VARCHAR, hedge VARCHAR, index_prices VARCHAR);
    CREATE TABLE IF NOT EXISTS block_future_legs (
        trade_id VARCHAR PRIMARY KEY, ts_ms BIGINT, instrument_name VARCHAR, direction VARCHAR, amount DOUBLE,
        price DOUBLE, index_price DOUBLE, block_trade_id VARCHAR);").map_err(|e| e.to_string())?;
                return Ok(con);
            }
            Err(e) => {
                last = e.to_string();
                if !last.to_lowercase().contains("lock") || i == 14 {
                    return Err(last);
                }
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
        }
    }
    Err(last)
}

#[derive(Clone)]
struct Leg {
    ts_ms: i64,
    trade_id: String,
    instrument_name: String,
    direction: String,
    amount: f64,
    price: f64,
    mark_price: Option<f64>,
    index_price: Option<f64>,
    iv: Option<f64>,
    block_trade_id: Option<String>,
    leg_count: Option<i64>,
    block_rfq_id: Option<String>,
    combo_id: Option<String>,
}

struct Fut {
    instrument_name: String,
    direction: String,
    amount: f64,
    price: f64,
}

fn eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.max(b)
}

/// `_shape(o)` -- o = (만기, 'C'|'P', 행사가, 순수량) 정렬된 것
fn shape(o: &[(f64, char, f64, f64)]) -> &'static str {
    let n = o.len();
    let exps: std::collections::BTreeSet<u64> = o.iter().map(|x| x.0.to_bits()).collect();
    let kinds: std::collections::BTreeSet<char> = o.iter().map(|x| x.1).collect();
    let s: Vec<i32> = o.iter().map(|x| if x.3 > 0.0 { 1 } else { -1 }).collect();
    let q: Vec<f64> = o.iter().map(|x| x.3.abs()).collect();
    if n == 1 {
        return "single";
    }
    if n == 2 && exps.len() == 1 {
        if kinds.len() == 2 {
            let (c, p) = if o[0].1 == 'C' { (o[0], o[1]) } else { (o[1], o[0]) };
            if (c.3 > 0.0) == (p.3 > 0.0) {
                return if c.2 == p.2 { "straddle" } else { "strangle" };
            }
            return if c.2 == p.2 { "synthetic" } else { "risk_reversal" };
        }
        return if s[0] == s[1] { "other" } else if eq(q[0], q[1]) { "vertical" } else { "ratio_spread" };
    }
    if n == 2 {
        if kinds.len() == 1 && s[0] != s[1] && eq(q[0], q[1]) {
            return if o[0].2 == o[1].2 { "calendar" } else { "diagonal" };
        }
        return "other";
    }
    if exps.len() != 1 {
        return "other";
    }
    if n == 3 && kinds.len() == 1 {
        if s[0] == s[2] && s[2] != s[1] && eq(q[0], q[2]) && eq(q[1], q[0] + q[2]) {
            return "butterfly";
        }
        if eq(q[0], q[1]) && eq(q[1], q[2]) && ((s[0] != s[1] && s[1] == s[2]) || (s[0] == s[1] && s[1] != s[2])) {
            return "ladder";
        }
        return "other";
    }
    if n == 4 && q.iter().all(|x| eq(*x, q[0])) {
        if kinds.len() == 1 {
            return if s[0] == s[3] && s[3] != s[1] && s[1] == s[2] { "condor" } else { "other" };
        }
        let puts: Vec<&(f64, char, f64, f64)> = o.iter().filter(|x| x.1 == 'P').collect();
        let calls: Vec<&(f64, char, f64, f64)> = o.iter().filter(|x| x.1 == 'C').collect();
        if puts.len() != 2 {
            return "other";
        }
        let strikes: std::collections::BTreeSet<u64> = o.iter().map(|x| x.2.to_bits()).collect();
        if strikes.len() == 2 && puts[0].2 == calls[0].2 {
            let lo_c = calls[0].3 > 0.0;
            return if lo_c != (puts[0].3 > 0.0) && (calls[1].3 > 0.0) != lo_c { "box" } else { "other" };
        }
        if (puts[0].3 > 0.0) != (puts[1].3 > 0.0) && (calls[0].3 > 0.0) != (calls[1].3 > 0.0) && (puts[1].3 > 0.0) == (calls[0].3 > 0.0) {
            if puts[1].2 == calls[0].2 {
                return "iron_butterfly";
            }
            return if puts[1].2 < calls[0].2 { "iron_condor" } else { "other" };
        }
    }
    "other"
}

/// `block_shape(legs, ts_ms, fut)` -- Err = 원본 예외(그 블록은 구조 칸 없이)
fn block_shape(legs: &[Leg], ts_ms: i64, fut: Option<&Vec<Fut>>) -> Result<Map<String, Value>, String> {
    let mut net: Vec<(String, f64)> = vec![];
    for l in legs {
        let v = (if l.direction == "buy" { 1.0 } else { -1.0 }) * l.amount;
        match net.iter_mut().find(|x| x.0 == l.instrument_name) {
            Some(x) => x.1 += v,
            None => net.push((l.instrument_name.clone(), 0.0 + v)),
        }
    }
    let mut o: Vec<(f64, char, f64, f64)> = vec![];
    for (inst, qq) in &net {
        if let Some((k, exp, call)) = parse_instrument(inst) {
            if qq.abs() > 1e-12 {
                o.push((exp, if call { 'C' } else { 'P' }, k, *qq));
            }
        }
    }
    o.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)).then(a.2.partial_cmp(&b.2).unwrap()));
    let (mut nd, mut gd, mut nv, mut gv) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for l in legs {
        let sp = parse_instrument(&l.instrument_name);
        let (s, iv) = (l.index_price.unwrap_or(0.0), l.iv.unwrap_or(0.0));
        let Some((k, exp, call)) = sp else { continue };
        let yrs = (exp - ts_ms as f64 / 1000.0) / (365.0 * 86400.0);
        if s <= 0.0 || iv <= 0.0 || yrs <= 0.0 {
            continue;
        }
        let sig = iv / 100.0;
        let d1 = ((s / k).ln() + 0.5 * sig * sig * yrs) / (sig * yrs.sqrt());
        let de = bs_delta(s, k, iv, yrs, call);
        let ve = s * (-0.5 * d1 * d1).exp() / (2.0 * std::f64::consts::PI).sqrt() * yrs.sqrt() / 100.0;
        let qq = (if l.direction == "buy" { 1.0 } else { -1.0 }) * l.amount;
        (nd, gd, nv, gv) = (nd + qq * de, gd + (qq * de).abs(), nv + qq * ve, gv + (qq * ve).abs());
    }
    for f in fut.map(|v| v.as_slice()).unwrap_or(&[]) {
        let qq = (if f.direction == "buy" { 1.0 } else { -1.0 }) * f.amount;
        let d = if f.instrument_name.contains("_USDC") {
            qq
        } else {
            if f.price == 0.0 {
                return Err("ZeroDivisionError".into());
            }
            qq / f.price
        };
        (nd, gd) = (nd + d, gd + d.abs());
    }
    let side = |v: f64, g: f64, pos: &str, neg: &str| if g <= 0.0 { Value::Null } else if v.abs() < 0.1 * g { json!("neutral") } else { json!(if v > 0.0 { pos } else { neg }) };
    let mut m = Map::new();
    m.insert("structure".into(), json!(if o.is_empty() { "future_only" } else { shape(&o) }));
    m.insert("hedge".into(), json!(fut.is_some_and(|v| !v.is_empty())));
    m.insert("net_delta".into(), json!(pyround(nd, 4)));
    m.insert("net_vega_usd".into(), json!(pyround(nv, 2)));
    m.insert("delta_side".into(), side(nd, gd, "long", "short"));
    m.insert("vol_side".into(), side(nv, gv, "long_vol", "short_vol"));
    Ok(m)
}

fn opt(v: Option<f64>) -> Value {
    v.map_or(Value::Null, |x| json!(x))
}

/// `group_blocks(rows, fut)` -- 블록 단위(최신 먼저)
fn group_blocks(rows: &[Leg], fut: &BTreeMap<String, Vec<Fut>>) -> Vec<Value> {
    let mut rs: Vec<&Leg> = rows.iter().collect();
    rs.sort_by(|a, b| a.ts_ms.cmp(&b.ts_ms).then(a.trade_id.cmp(&b.trade_id)));
    let mut blocks: Vec<(String, Map<String, Value>, Vec<Leg>, f64, f64)> = vec![];
    for r in rs {
        let id = r.block_trade_id.clone().unwrap_or_default();
        let k = match blocks.iter().position(|b| b.0 == id) {
            Some(k) => k,
            None => {
                let mut m = Map::new();
                m.insert("block_trade_id".into(), r.block_trade_id.clone().map_or(Value::Null, Value::from));
                m.insert("ts_ms".into(), json!(r.ts_ms));
                m.insert("leg_count".into(), r.leg_count.map_or(Value::Null, Value::from));
                m.insert("legs_seen".into(), json!(0));
                m.insert("block_rfq_id".into(), r.block_rfq_id.clone().map_or(Value::Null, Value::from));
                m.insert("combo_id".into(), r.combo_id.clone().map_or(Value::Null, Value::from));
                m.insert("notional_usd".into(), json!(0.0));
                m.insert("premium_usd".into(), json!(0.0));
                m.insert("legs".into(), json!([]));
                blocks.push((id, m, vec![], 0.0, 0.0));
                blocks.len() - 1
            }
        };
        let b = &mut blocks[k];
        let idx = r.index_price.filter(|x| *x != 0.0).unwrap_or(0.0);
        let n = b.1["legs_seen"].as_i64().unwrap() + 1;
        b.1.insert("legs_seen".into(), json!(n));
        b.3 += r.amount * idx;
        b.4 += r.amount * r.price * (if r.instrument_name.contains("_USDC-") { 1.0 } else { idx });
        let mut l = Map::new();
        l.insert("instrument_name".into(), json!(r.instrument_name));
        l.insert("direction".into(), json!(r.direction));
        l.insert("amount".into(), json!(r.amount));
        l.insert("price".into(), json!(r.price));
        l.insert("iv".into(), opt(r.iv));
        l.insert("mark_price".into(), opt(r.mark_price));
        l.insert("index_price".into(), opt(r.index_price));
        l.insert("trade_id".into(), json!(r.trade_id));
        b.1["legs"].as_array_mut().unwrap().push(Value::Object(l));
        b.2.push(r.clone());
    }
    let mut out: Vec<(i64, Value)> = blocks.into_iter().map(|(id, mut m, legs, no, pr)| {
        m.insert("notional_usd".into(), json!(pyround(0.0 + no, 2)));
        m.insert("premium_usd".into(), json!(pyround(0.0 + pr, 2)));
        let ts = m["ts_ms"].as_i64().unwrap();
        if let Ok(sh) = block_shape(&legs, ts, fut.get(&id)) {
            m.extend(sh);
        }
        (ts, Value::Object(m))
    }).collect();
    out.sort_by(|a, b| b.0.cmp(&a.0)); // 안정 정렬 · 최신 먼저(파이썬 sorted(reverse=True) 도 같은 ts 끼리는 원래 순서)
    out.into_iter().map(|x| x.1).collect()
}

/// `hourly_flow(rows, doi)` -- 코인별 정시 버킷 25개
fn hourly_flow(rows: &[(i64, String, String, f64, Option<f64>, Option<f64>, Option<String>, bool)], now_s: f64) -> Value {
    let now_h = (now_s / 3600.0).floor() as i64 * 3_600_000;
    let hours: Vec<i64> = (0..=24).rev().map(|i| now_h - i * 3_600_000).collect();
    const K: [&str; 8] = ["cb", "cs", "pb", "ps", "dlt", "vg", "vgb", "liq"];
    let mut out: BTreeMap<&str, Vec<[f64; 8]>> = COINS.iter().map(|c| (*c, vec![[0.0; 8]; 25])).collect();
    for (ts, inst, dir, amt, iv, ix, liq, blk) in rows {
        let Some(c) = coin_of(inst) else { continue };
        let Some((k, exp, call)) = parse_instrument(inst) else { continue };
        let h = ts.div_euclid(3_600_000) * 3_600_000;
        let Some(bi) = hours.iter().position(|x| *x == h) else { continue };
        let b = &mut out.get_mut(c).unwrap()[bi];
        let buy = dir == "buy";
        b[if call { if buy { 0 } else { 1 } } else if buy { 2 } else { 3 }] += amt;
        if liq.as_deref().is_some_and(|s| !s.is_empty()) {
            b[7] += amt;
        }
        let yrs = (exp - *ts as f64 / 1000.0) / (365.0 * 86400.0);
        let d = bs_delta(ix.unwrap_or(0.0), k, iv.unwrap_or(0.0), yrs, call);
        b[4] += (if buy { 1.0 } else { -1.0 }) * amt * d;
        let (s, sig) = (ix.unwrap_or(0.0), iv.unwrap_or(0.0) / 100.0);
        if s > 0.0 && sig > 0.0 && yrs > 0.0 {
            let d1 = ((s / k).ln() + 0.5 * sig * sig * yrs) / (sig * yrs.sqrt());
            let vg = (if buy { 1.0 } else { -1.0 }) * amt * s * (-0.5 * d1 * d1).exp() / (2.0 * std::f64::consts::PI).sqrt() * yrs.sqrt() / 100.0;
            b[5] += vg;
            if *blk {
                b[6] += vg;
            }
        }
    }
    let mut m = Map::new();
    for c in COINS {
        let v: Vec<Value> = out[c].iter().zip(&hours).map(|(b, h)| {
            let mut o = Map::new();
            o.insert("h".into(), json!(h));
            for (i, k) in K.iter().enumerate() {
                o.insert((*k).into(), json!(pyround(b[i], 3)));
            }
            o.insert("doi".into(), Value::Null); // hourly_doi = {}(체인 표 없음 -- 머리 설명)
            Value::Object(o)
        }).collect();
        m.insert(c.into(), Value::Array(v));
    }
    Value::Object(m)
}

/// `_write(rows, gaps)` + 상태 JSON. 반환 = 새로 들어간 행 수.
pub fn write(db: &Path, state: &Path, rows: &[Vec<duckdb::types::Value>], gaps: &[(i64, i64, String, Option<i64>)], now_s: f64, iso_now: &str) -> Result<usize, String> {
    let mut con = connect(db)?;
    let cnt = |c: &Connection| c.query_row("SELECT count(*) FROM option_trades", [], |r| r.get::<_, i64>(0)).map_err(|e| e.to_string());
    let before = cnt(&con)?;
    {
        let tx = con.transaction().map_err(|e| e.to_string())?;
        {
            let mut st = tx.prepare(&format!("INSERT OR IGNORE INTO option_trades VALUES ({})", vec!["?"; COLS.len()].join(","))).map_err(|e| e.to_string())?;
            for r in rows {
                st.execute(duckdb::params_from_iter(r.iter())).map_err(|e| e.to_string())?;
            }
            for g in gaps {
                tx.execute("INSERT INTO gaps VALUES (?,?,?,?)", params![g.0, g.1, g.2, g.3]).map_err(|e| e.to_string())?;
            }
        }
        tx.commit().map_err(|e| e.to_string())?;
    }
    let added = (cnt(&con)? - before) as usize;
    let since = (now_s * 1000.0) as i64 - STATE_WINDOW_MS;
    let legs: Vec<Leg> = con.prepare(&format!("SELECT {} FROM option_trades WHERE is_block AND ts_ms >= ?", COLS.join(","))).and_then(|mut s| {
        s.query_map([since], |r| Ok(Leg { ts_ms: r.get(0)?, trade_id: r.get(1)?, instrument_name: r.get(3)?, direction: r.get(4)?, amount: r.get(5)?, price: r.get(6)?,
            mark_price: r.get(7)?, index_price: r.get(8)?, iv: r.get(9)?, block_trade_id: r.get(13)?, leg_count: r.get(14)?, block_rfq_id: r.get(15)?, combo_id: r.get(16)? }))?.collect()
    }).map_err(|e| e.to_string())?;
    let mut fut: BTreeMap<String, Vec<Fut>> = BTreeMap::new();
    if let Ok(v) = con.prepare("SELECT block_trade_id, instrument_name, direction, amount, price FROM block_future_legs WHERE ts_ms >= ?").and_then(|mut s| {
        s.query_map([since], |r| Ok((r.get::<_, Option<String>>(0)?, Fut { instrument_name: r.get(1)?, direction: r.get(2)?, amount: r.get(3)?, price: r.get(4)? })))?.collect::<Result<Vec<_>, _>>()
    }) {
        for (b, f) in v {
            fut.entry(b.unwrap_or_default()).or_default().push(f);
        }
    }
    let by_coin: Vec<(&str, Vec<Value>)> = COINS.iter().map(|c| (*c, group_blocks(&legs.iter().filter(|x| coin_of(&x.instrument_name) == Some(c)).cloned().collect::<Vec<_>>(), &fut))).collect();
    let fr0 = ((now_s / 3600.0).floor() as i64 - 24) * 3_600_000;
    let flow_rows: Vec<(i64, String, String, f64, Option<f64>, Option<f64>, Option<String>, bool)> = con.prepare(
        "SELECT ts_ms, instrument_name, direction, amount, iv, index_price, liquidation, is_block FROM option_trades WHERE ts_ms >= ?").and_then(|mut s| {
        s.query_map([fr0], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?, r.get::<_, Option<bool>>(7)?.unwrap_or(false))))?.collect()
    }).map_err(|e| e.to_string())?;
    drop(con);
    let flow = hourly_flow(&flow_rows, now_s);
    let mut bsum = Map::new();
    for (c, v) in &by_coin {
        // 원본: round(sum(b.get(...) or 0 ...), 3) -- 블록이 없거나 모두 구조 칸이 없으면 정수 0
        let sum = |k: &str, n: usize| -> Value {
            let (mut s, mut fl) = (0.0, false);
            for b in v {
                if let Some(x) = b.get(k).and_then(Value::as_f64).filter(|x| *x != 0.0) {
                    s += x; // 0.0·None 은 `or 0` = 정수 0(합이 정수로 남는다)
                    fl = true;
                }
            }
            if fl { json!(pyround(s, n)) } else { json!(0) }
        };
        bsum.insert((*c).into(), json!({"net_delta": sum("net_delta", 3), "net_vega_usd": sum("net_vega_usd", 1), "n": v.len()}));
    }
    let mut out = Map::new();
    out.insert("generated_at".into(), json!(iso_now));
    out.insert("coins".into(), json!(COINS));
    out.insert("block_sum_by_coin".into(), Value::Object(bsum));
    out.insert("source".into(), json!(format!("deribit {}", channels().join(" · "))));
    out.insert("window_hours".into(), json!(STATE_WINDOW_MS / 3_600_000));
    out.insert("n_blocks".into(), json!(by_coin[0].1.len()));
    out.insert("blocks".into(), Value::Array(by_coin[0].1.clone()));
    out.insert("blocks_by_coin".into(), Value::Object(by_coin.iter().map(|(c, v)| ((*c).to_string(), Value::Array(v.clone()))).collect()));
    out.insert("n_blocks_by_coin".into(), Value::Object(by_coin.iter().map(|(c, v)| ((*c).to_string(), json!(v.len()))).collect()));
    out.insert("flow_by_coin".into(), flow);
    let tmp = state.with_extension("json.tmp");
    std::fs::write(&tmp, rtcollect::py_dumps_default_utf8(&Value::Object(out))).and_then(|_| std::fs::rename(&tmp, state)).map_err(|e| e.to_string())?;
    Ok(added)
}

/// `last_ts()` -- 목록별 마지막 저장 시각
pub fn last_ts(db: &Path) -> Result<Vec<Option<i64>>, String> {
    let con = connect(db)?;
    API_CURRENCIES.iter().map(|api| {
        let pat = if *api == "USDC" { "%_USDC-%".to_string() } else { format!("{api}-%") };
        con.query_row("SELECT max(ts_ms) FROM option_trades WHERE instrument_name LIKE ?", [pat], |r| r.get(0)).map_err(|e| e.to_string())
    }).collect()
}

/// `backfill(since, until)` -- REST 쪽 넘김(get = URL → 본문 · Err = raise_for_status·접속 실패)
pub fn backfill(since: &[i64], until_ms: i64, get: &mut dyn FnMut(&str) -> Result<Vec<u8>, String>) -> Result<Vec<Vec<duckdb::types::Value>>, String> {
    let mut out = vec![];
    for (api, s0) in API_CURRENCIES.iter().zip(since) {
        let mut start = *s0;
        loop {
            let u = format!("{REST}?currency={api}&kind=option&start_timestamp={start}&end_timestamp={until_ms}&count=1000&sorting=asc&include_old=true");
            let v: Value = serde_json::from_slice(&get(&u)?).map_err(|e| format!("JSONDecodeError: {e}"))?;
            let res = v.get("result").ok_or("KeyError: 'result'")?;
            let trades = res.get("trades").and_then(Value::as_array).cloned().unwrap_or_default();
            for t in &trades {
                if coin_of(t["instrument_name"].as_str().ok_or("TypeError")?).is_some() {
                    out.push(row(t)?);
                }
            }
            let nxt = trades.iter().filter_map(|t| t["timestamp"].as_i64()).max().unwrap_or(start);
            if !res.get("has_more").is_some_and(rtcollect_truthy) || nxt <= start {
                break;
            }
            start = nxt;
        }
    }
    Ok(out)
}

fn rtcollect_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// 파이썬 datetime.now(timezone.utc).isoformat()
pub fn iso_utc(t: f64) -> String {
    let us = (t * 1e6).round() as i64;
    let (s, frac) = (us.div_euclid(1_000_000), us.rem_euclid(1_000_000));
    let d = s.div_euclid(86_400);
    let z = d + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let dd = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let sec = s.rem_euclid(86_400);
    let f = if frac != 0 { format!(".{frac:06}") } else { String::new() };
    format!("{:04}-{m:02}-{dd:02}T{:02}:{:02}:{:02}{f}+00:00", yoe + era * 400 + i64::from(m <= 2), sec / 3600, sec / 60 % 60, sec % 60)
}
