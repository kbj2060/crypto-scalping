//! Deribit ② = collect_deribit_option_gex_20260815 의 `poll_once`(체인 넷 · OKX·Bybit 옵션 · option_chain_snapshot · option_summary · deribit_gex_state.json).
//!   GEX 계산 도구(`_skew_interp`·`_surface_stats`·`_atm_spread_iv`·`_const_maturity`·사다리)는 rtcollect::cboe_summary 의 것을 쓴다(cboe 요약기가 같은 도구를 가져다 쓴다 --
//!   그 쪽 정답지로 이미 값 일치) · `_bs_gamma` 만 여기.
//!   원본과 다른 곳(묶은 DuckDB 에 ICU·json 확장이 없어서 · 값은 같다): write_state 의 vov24·atm_oi_pct 를 SQL(date_trunc·timezone·json_extract_string·now() - INTERVAL) 대신
//!   같은 행을 읽어 Rust 로 센다(arg_max = 값이 NULL 인 행은 건너뛴다 -- DuckDB arg_max 와 같다) · 지역 시각 글자(recorded_at_utc)는 부르는 쪽이 준 오프셋(서버 KST +09:00).
use rtcollect::cboe_summary::{atm_spread_iv, const_maturity, ladder, median, pyround_n, skew_interp, surface_stats, Row};
use rtcollect::ob5m::np_sum;
use duckdb::Connection;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

pub const PUBLIC: &str = "https://www.deribit.com/api/v2/public/";
pub const BASE_URL: &str = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency";
pub const CURRENCIES: [&str; 4] = ["ETH", "BTC", "SOL", "XRP"];
const OTHER: [&str; 2] = ["ETH", "BTC"];

/// (API 목록, 종목 접두어, 지수 이름, 역옵션)
fn spec(c: &str) -> (&'static str, &'static str, &'static str, bool) {
    match c {
        "ETH" => ("ETH", "ETH-", "eth_usd", true),
        "BTC" => ("BTC", "BTC-", "btc_usd", true),
        "SOL" => ("USDC", "SOL_USDC-", "sol_usdc", false),
        _ => ("USDC", "XRP_USDC-", "xrp_usdc", false),
    }
}

/// get(URL, timeout 초, Mozilla UA?) → 본문 · Err = 접속 실패·HTTP ≥ 400(원본 raise_for_status · OKX·Bybit 는 원본도 .json() 의 KeyError 로 같은 «실패»)
pub type Get<'a> = &'a mut dyn FnMut(&str, u64, bool) -> Result<Vec<u8>, String>;

fn url(base: &str, params: &[(&str, String)]) -> String {
    format!("{base}?{}", params.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("&"))
}

fn getj(get: Get, base: &str, params: &[(&str, String)], timeout: u64, ua: bool) -> Result<Value, String> {
    serde_json::from_slice(&get(&url(base, params), timeout, ua)?).map_err(|e| format!("JSONDecodeError: {e}"))
}

/// `_pub(method)` → result
fn pub_(get: Get, method: &str, timeout: u64, params: &[(&str, String)]) -> Result<Value, String> {
    getj(get, &format!("{PUBLIC}{method}"), params, timeout, false)?.get("result").cloned().ok_or_else(|| "KeyError: 'result'".into())
}

fn fz(v: &Value) -> f64 {
    // float(row.get(k) or 0.0)
    match v {
        Value::Number(n) => n.as_f64().unwrap(),
        Value::String(s) if !s.is_empty() => s.parse().unwrap_or(0.0),
        Value::Bool(true) => 1.0,
        _ => 0.0,
    }
}

/// `_bs_gamma`
fn bs_gamma(spot: f64, k: f64, iv_pct: f64, years: f64) -> f64 {
    let s = iv_pct / 100.0;
    if spot <= 0.0 || k <= 0.0 || s <= 0.0 || years <= 0.0 {
        return 0.0;
    }
    let d1 = ((spot / k).ln() + 0.5 * s * s * years) / (s * years.sqrt());
    let pdf = (-0.5 * d1 * d1).exp() / (2.0 * std::f64::consts::PI).sqrt();
    pdf / (spot * s * years.sqrt())
}

#[derive(Clone)]
pub struct C {
    pub inst: String,
    pub call: bool,
    pub strike: f64,
    pub exp_us: i64,
    pub days: f64,
    pub oi: f64,
    pub iv: f64,
    pub und: f64,
    pub mark: f64,
    pub vol: f64,
    pub gamma: f64,
    pub bid: f64,
    pub ask: f64,
}

impl C {
    fn row(&self) -> Row {
        Row { exp: self.exp_us / 1000, call: self.call, strike: self.strike, iv: self.iv, bid: self.bid, ask: self.ask, oi: self.oi, days: self.days }
    }
}

/// 프로세스 캐시 둘(원본 _LIST_CACHE · _RV_CACHE)
#[derive(Default)]
pub struct Cache {
    list: HashMap<String, (f64, Vec<Value>)>,
    rv: HashMap<String, (f64, f64)>,
}

/// `fetch_chain(currency)`
pub fn fetch_chain(cur: &str, now_us: i64, cache: &mut Cache, get: Get) -> Result<Vec<C>, String> {
    let (api, prefix, _, _) = spec(cur);
    let now = now_us as f64 / 1e6;
    let rows = match cache.list.get(api).filter(|h| now - h.0 < 60.0) {
        Some(h) => h.1.clone(),
        None => {
            let v = getj(get, BASE_URL, &[("currency", api.into()), ("kind", "option".into())], 20, false)?;
            let r = v.get("result").and_then(Value::as_array).cloned().unwrap_or_default();
            cache.list.insert(api.into(), (now, r.clone()));
            r
        }
    };
    let mut out = vec![];
    for r in rows.iter().filter(|r| r["instrument_name"].as_str().is_some_and(|s| s.starts_with(prefix))) {
        let inst = r["instrument_name"].as_str().unwrap();
        let Some((k, exp_s, call)) = crate::deribit::parse_instrument(inst) else { continue };
        let exp_us = exp_s as i64 * 1_000_000;
        let years = (exp_us - now_us) as f64 / 1e6 / (365.0 * 86400.0);
        if years <= 0.0 {
            continue;
        }
        let spot = fz(&r["underlying_price"]);
        let iv = fz(&r["mark_iv"]);
        out.push(C { inst: inst.into(), call, strike: k, exp_us, days: years * 365.0, oi: fz(&r["open_interest"]), iv, und: spot, mark: fz(&r["mark_price"]),
                     vol: fz(&r["volume"]), gamma: bs_gamma(spot, k, iv, years), bid: fz(&r["bid_price"]), ask: fz(&r["ask_price"]) });
    }
    Ok(out)
}

fn iso(us: i64, off_s: i64) -> String {
    let t = us + off_s * 1_000_000;
    let (s, frac) = (t.div_euclid(1_000_000), t.rem_euclid(1_000_000));
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
    let o = if off_s == 0 { "+00:00".to_string() } else { format!("{}{:02}:{:02}", if off_s < 0 { '-' } else { '+' }, off_s.abs() / 3600, off_s.abs() / 60 % 60) };
    format!("{:04}-{m:02}-{dd:02}T{:02}:{:02}:{:02}{f}{o}", yoe + era * 400 + i64::from(m <= 2), sec / 3600, sec / 60 % 60, sec % 60)
}

pub fn iso_utc(us: i64) -> String {
    iso(us, 0)
}

/// 첫 최솟값 자리(argsort()[:1] -- 같은 값이면 앞 것)
fn argmin_by(v: &[f64]) -> usize {
    let mut b = 0;
    for (i, x) in v.iter().enumerate() {
        if *x < v[b] {
            b = i;
        }
    }
    b
}

/// `options_summary(chain, currency)`
pub fn options_summary(chain: &[C], cur: &str, now_us: i64, cache: &mut Cache, get: Get, log: &mut dyn FnMut(String)) -> Result<Map<String, Value>, String> {
    let now = now_us as f64 / 1e6;
    let (_, _, index_name, inverse) = spec(cur);
    let mut out = Map::new();
    out.insert("recorded_at_utc".into(), json!(iso_utc(now_us)));
    out.insert("currency".into(), json!(cur));
    out.insert("dvol".into(), Value::Null);
    let r = pub_(get, "get_index_price", 30, &[("index_name", index_name.into())]).and_then(|v| v.get("index_price").and_then(Value::as_f64).ok_or_else(|| "KeyError: 'index_price'".into()));
    out.insert("index".into(), r.as_ref().map_or_else(|e| { log(format!("{cur}: index 실패 {e}")); Value::Null }, |x| json!(x)));
    if inverse {
        let params = [("currency", cur.to_string()), ("resolution", "60".into()), ("start_timestamp", (((now - 7200.0) * 1000.0) as i64).to_string()),
                      ("end_timestamp", ((now * 1000.0) as i64).to_string())];
        let r = pub_(get, "get_volatility_index_data", 30, &params).and_then(|v| v["data"].as_array().and_then(|a| a.last()).and_then(|x| x.get(4)).and_then(Value::as_f64).ok_or_else(|| "IndexError".to_string()));
        out.insert("dvol".into(), r.map_or_else(|e| { log(format!("{cur}: dvol 실패 {e}")); Value::Null }, |x| json!(x)));
    }
    let hit = cache.rv.get(cur).copied();
    let rv7: Option<f64> = if let Some(h) = hit.filter(|h| now - h.0 < 6.0 * 3600.0) {
        Some(h.1)
    } else {
        let inst = if inverse { format!("{cur}-PERPETUAL") } else { format!("{cur}_USDC-PERPETUAL") };
        let params = [("instrument_name", inst), ("resolution", "60".into()), ("start_timestamp", (((now - 7.0 * 86400.0) * 1000.0) as i64).to_string()),
                      ("end_timestamp", ((now * 1000.0) as i64).to_string())];
        match pub_(get, "get_tradingview_chart_data", 90, &params) {
            Ok(tv) => {
                let cl: Vec<f64> = tv["close"].as_array().map(|a| a.iter().filter_map(Value::as_f64).filter(|c| *c != 0.0).collect()).unwrap_or_default();
                let rets: Vec<f64> = cl.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
                let rv = (rets.len() > 24).then(|| {
                    let mut s = 0.0;
                    for x in &rets {
                        s += x * x;
                    }
                    (s / rets.len() as f64 * 24.0 * 365.0).sqrt() * 100.0
                });
                if let Some(v) = rv {
                    cache.rv.insert(cur.into(), (now, v));
                }
                rv
            }
            Err(e) => {
                let v = hit.filter(|h| now - h.0 < 24.0 * 3600.0).map(|h| h.1);
                log(format!("{cur}: rv7 실패 {e}{}", if v.is_some() { " (직전 값 사용)" } else { "" }));
                v
            }
        }
    };
    out.insert("rv7".into(), rv7.map_or(Value::Null, |x| json!(x)));
    let idx = out["index"].as_f64().filter(|x| *x != 0.0).unwrap_or_else(|| median(&mut chain.iter().map(|c| c.und).collect::<Vec<_>>()));
    // 만기별
    let mut by: BTreeMap<i64, Vec<&C>> = BTreeMap::new();
    for c in chain {
        by.entry(c.exp_us).or_default().push(c);
    }
    let mut exps: Vec<(Map<String, Value>, Vec<Row>, f64, i64, f64)> = vec![]; // (요약, 행, yrs, exp_us, fwd)
    for (exp, g) in &by {
        let yrs = g[0].days / 365.0;
        let fwd = g[0].und;
        let calls: Vec<&&C> = g.iter().filter(|c| c.call).collect();
        let puts: Vec<&&C> = g.iter().filter(|c| !c.call).collect();
        if calls.is_empty() || puts.is_empty() {
            continue;
        }
        let atm_k = g[argmin_by(&g.iter().map(|c| (c.strike - fwd).abs()).collect::<Vec<_>>())].strike;
        let at: Vec<f64> = g.iter().filter(|c| c.strike == atm_k).map(|c| c.iv).collect();
        let atm_iv = np_sum(&at) / at.len() as f64;
        let cd: Vec<f64> = calls.iter().map(|c| rtcollect::cboe_summary::bs_delta(fwd, c.strike, c.iv, yrs, true)).collect();
        let pd: Vec<f64> = puts.iter().map(|c| rtcollect::cboe_summary::bs_delta(fwd, c.strike, c.iv, yrs, false)).collect();
        let c25 = calls[argmin_by(&cd.iter().map(|d| (d - 0.25).abs()).collect::<Vec<_>>())].iv;
        let p25 = puts[argmin_by(&pd.iter().map(|d| (d + 0.25).abs()).collect::<Vec<_>>())].iv;
        let mut ks: Vec<f64> = g.iter().map(|c| c.strike).collect();
        ks.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ks.dedup();
        let pain_of = |p: f64| np_sum(&calls.iter().map(|c| (p - c.strike).max(0.0) * c.oi).collect::<Vec<_>>()) + np_sum(&puts.iter().map(|c| (c.strike - p).max(0.0) * c.oi).collect::<Vec<_>>());
        let mut pain = ks[0];
        let mut best = pain_of(pain);
        for &k in &ks[1..] {
            let v = pain_of(k);
            if v < best {
                (best, pain) = (v, k);
            }
        }
        let coi = np_sum(&calls.iter().map(|c| c.oi).collect::<Vec<_>>());
        let poi = np_sum(&puts.iter().map(|c| c.oi).collect::<Vec<_>>());
        let rows: Vec<Row> = g.iter().map(|c| c.row()).collect();
        let rr: Vec<&Row> = rows.iter().collect();
        let mut m = Map::new();
        m.insert("exp_ms".into(), json!(exp / 1000));
        m.insert("atm_iv".into(), json!(atm_iv));
        m.insert("rr25".into(), json!(c25 - p25));
        m.insert("bf25".into(), json!((c25 + p25) / 2.0 - atm_iv));
        m.insert("call_oi_usd".into(), json!(coi * idx));
        m.insert("put_oi_usd".into(), json!(poi * idx));
        m.insert("pc".into(), if coi != 0.0 { json!(poi / coi) } else { Value::Null });
        m.insert("pain".into(), json!(pain));
        m.extend(skew_interp(&rr, fwd, yrs));
        m.insert("fwd".into(), json!(fwd));
        m.insert("atm_oi_usd".into(), json!(np_sum(&g.iter().filter(|c| (c.strike / fwd - 1.0).abs() <= 0.02).map(|c| c.oi).collect::<Vec<_>>()) * idx));
        exps.push((m, rows, yrs, *exp, fwd));
    }
    let mut tot = 0.0;
    for e in &exps {
        tot += e.0["call_oi_usd"].as_f64().unwrap() + e.0["put_oi_usd"].as_f64().unwrap();
    }
    for e in exps.iter_mut() {
        let v = (e.0["call_oi_usd"].as_f64().unwrap() + e.0["put_oi_usd"].as_f64().unwrap()) / tot;
        e.0.insert("oi_share".into(), if tot != 0.0 { json!(v) } else { Value::Null });
    }
    let hours = |e: &(Map<String, Value>, Vec<Row>, f64, i64, f64)| e.0["hours"].as_f64().unwrap();
    let live: Vec<usize> = (0..exps.len()).filter(|i| hours(&exps[*i]) >= 3.0).collect();
    let near = |d: f64| -> Option<usize> {
        let mut b: Option<usize> = None;
        for &i in &live {
            if b.is_none_or(|j| (hours(&exps[i]) - d * 24.0).abs() < (hours(&exps[j]) - d * 24.0).abs()) {
                b = Some(i);
            }
        }
        b
    };
    let pick = [("front", live.first().copied()), ("7", near(7.0)), ("30", near(30.0))];
    let mut surf = Map::new();
    for (k, e) in pick {
        surf.insert(k.into(), e.map_or(Value::Null, |i| {
            let x = &exps[i];
            let rr: Vec<&Row> = x.1.iter().collect();
            let mut m = Map::new();
            m.insert("exp_ms".into(), json!(x.3 / 1000));
            m.insert("hours".into(), json!(pyround_n(hours(x), 2)));
            m.insert("fwd".into(), json!(x.4));
            m.extend(surface_stats(&rr, x.4, x.2));
            Value::Object(m)
        }));
    }
    out.insert("surface".into(), Value::Object(surf));
    out.insert("atm_spread_iv".into(), pick[1].1.map_or(Value::Null, |i| atm_spread_iv(&exps[i].1.iter().collect::<Vec<_>>(), exps[i].4, exps[i].2, !inverse)));
    out.insert("front_atm_oi_usd".into(), exps.first().map_or(Value::Null, |e| e.0["atm_oi_usd"].clone()));
    let maps: Vec<Map<String, Value>> = exps.iter().map(|e| e.0.clone()).collect();
    out.insert("expiries".into(), Value::Array(maps.iter().take(8).cloned().map(Value::Object).collect()));
    let mut cm = Map::new();
    for d in [1.0, 7.0, 30.0, 60.0] {
        cm.insert((d as i64).to_string(), const_maturity(&maps, d));
    }
    out.insert("cm".into(), Value::Object(cm));
    let mut n30: Option<usize> = None;
    let dist = |i: usize| ((exps[i].3 / 1000) as f64 / 1000.0 - now - 30.0 * 86400.0).abs();
    for i in 0..exps.len() {
        if n30.is_none_or(|j| dist(i) < dist(j)) {
            n30 = Some(i);
        }
    }
    out.insert("iv30".into(), n30.map_or(Value::Null, |i| exps[i].0["atm_iv"].clone()));
    let g_first = chain.iter().filter(|c| c.exp_us > now_us).map(|c| c.exp_us).min();
    let mut hedge = vec![];
    for e in exps.iter().filter(|e| (e.3 / 1000) as f64 / 1000.0 - now > 12.0 * 3600.0).take(2) {
        for r in chain.iter().filter(|c| c.exp_us == e.3) {
            let otm = (!r.call && idx * 0.9 <= r.strike && r.strike <= idx) || (r.call && idx <= r.strike && r.strike <= idx * 1.1);
            if otm && r.mark > 0.0 {
                hedge.push(json!({"exp_ms": e.3 / 1000, "k": r.strike, "type": if r.call { "C" } else { "P" }, "usd": r.mark * (if inverse { idx } else { 1.0 }), "iv": r.iv}));
            }
        }
    }
    out.insert("hedge".into(), Value::Array(hedge));
    let band: Vec<&C> = chain.iter().filter(|c| c.strike >= idx * 0.92 && c.strike <= idx * 1.08).collect();
    let lad = |f: &dyn Fn(&C) -> bool| ladder(&band.iter().filter(|c| f(c)).map(|c| {
        let u = c.oi * idx;
        (c.strike, if c.call { u } else { 0.0 }, if c.call { 0.0 } else { u })
    }).collect::<Vec<_>>());
    let mut st = Map::new();
    st.insert("front".into(), match g_first { Some(f) => Value::Array(lad(&|c| c.exp_us == f)), None => json!([]) });
    st.insert("week".into(), Value::Array(lad(&|c| c.days <= 7.0)));
    st.insert("all".into(), Value::Array(lad(&|c| c.days > 0.0)));
    st.insert("front_exp_ms".into(), g_first.map_or(Value::Null, |f| json!(f / 1000)));
    out.insert("strikes".into(), Value::Object(st));
    Ok(out)
}

pub fn ensure_tables(con: &Connection) -> duckdb::Result<()> {
    con.execute_batch("CREATE TABLE IF NOT EXISTS option_chain_snapshot (
            recorded_at_utc TIMESTAMPTZ, currency VARCHAR, instrument_name VARCHAR,
            option_type VARCHAR, strike DOUBLE, expiration_ts TIMESTAMPTZ, days_to_expiry DOUBLE,
            open_interest DOUBLE, mark_iv DOUBLE, underlying_price DOUBLE, mark_price DOUBLE,
            volume DOUBLE, gamma_bs DOUBLE
        );
        CREATE TABLE IF NOT EXISTS option_summary (
            recorded_at_utc TIMESTAMPTZ, currency VARCHAR, index_price DOUBLE, dvol DOUBLE, rv7 DOUBLE,
            front_atm_iv DOUBLE, front_rr25 DOUBLE, front_bf25 DOUBLE, gamma_flip DOUBLE, payload VARCHAR
        );
        CREATE TABLE IF NOT EXISTS option_oi_other (
            recorded_at_utc TIMESTAMPTZ, venue VARCHAR, currency VARCHAR, instrument_name VARCHAR, option_type VARCHAR,
            strike DOUBLE, expiration_ts TIMESTAMPTZ, open_interest DOUBLE, mark_iv DOUBLE, volume_24h DOUBLE
        )")
}

/// `parse_other(venue, name)` → (콜?, 행사가, 만기 µs)
fn parse_other(okx: bool, name: &str) -> Option<(bool, f64, i64)> {
    let p: Vec<&str> = name.split('-').collect();
    let num = |s: &str| {
        let d = s.bytes().take_while(u8::is_ascii_digit).count();
        (d > 0 && (d == s.len() || (s.as_bytes()[d] == b'.' && s.len() > d + 1 && s[d + 1..].bytes().all(|b| b.is_ascii_digit())))).then(|| s.parse::<f64>().ok()).flatten()
    };
    let up = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_uppercase());
    let (exp_days, k, cp) = if okx {
        if p.len() != 5 || !up(p[0]) || p[1] != "USD" || p[2].len() != 6 || !p[2].bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let (y, m, d) = (2000 + p[2][..2].parse::<i64>().ok()?, p[2][2..4].parse::<i64>().ok()?, p[2][4..].parse::<i64>().ok()?);
        let y = if y >= 2069 { y - 100 } else { y };
        ((y, m, d), num(p[3])?, p[4])
    } else {
        if !(p.len() == 4 || (p.len() == 5 && up(p[4]))) || !up(p[0]) {
            return None;
        }
        let e = p[1];
        let nd = e.bytes().take_while(u8::is_ascii_digit).count();
        if !(1..=2).contains(&nd) || e.len() != nd + 5 || !e[nd..nd + 3].bytes().all(|b| b.is_ascii_uppercase()) || !e[nd + 3..].bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let mon = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"].iter().position(|m| *m == &e[nd..nd + 3])? as i64 + 1;
        let yy: i64 = e[nd + 3..].parse().ok()?;
        ((if yy < 69 { 2000 + yy } else { 1900 + yy }, mon, e[..nd].parse().ok()?), num(p[2])?, p[3])
    };
    if cp != "C" && cp != "P" {
        return None;
    }
    let (y, m, d) = exp_days;
    if !(1..=12).contains(&m) || d < 1 || d > 31 {
        return None;
    }
    let days = crate::deribit::days_from_civil(y, m, d);
    Some((cp == "C", k, (days * 86_400 + 8 * 3600) * 1_000_000))
}

type Other = (String, String, String, bool, f64, i64, f64, Option<f64>, Option<f64>);

/// `fetch_other_venues` -- 실패한 거래소는 빈 채로
fn fetch_other(cur: &str, now_us: i64, get: Get, log: &mut dyn FnMut(String)) -> Vec<Other> {
    let mut out = vec![];
    let okx = (|| -> Result<Vec<Other>, String> {
        let oi = getj(get, "https://www.okx.com/api/v5/public/open-interest", &[("instType", "OPTION".into()), ("instFamily", format!("{cur}-USD"))], 20, true)?;
        let oi = oi.get("data").and_then(Value::as_array).cloned().ok_or("KeyError: 'data'")?;
        let iv = getj(get, "https://www.okx.com/api/v5/public/opt-summary", &[("instFamily", format!("{cur}-USD"))], 20, true)?;
        let mut ivm: HashMap<String, f64> = HashMap::new();
        for x in iv.get("data").and_then(Value::as_array).ok_or("KeyError: 'data'")? {
            if x.get("markVol").is_some_and(|v| truthy(v)) {
                ivm.insert(x["instId"].as_str().unwrap_or("").into(), fnum(&x["markVol"])? * 100.0);
            }
        }
        let mut v = vec![];
        for x in &oi {
            let id = x["instId"].as_str().ok_or("TypeError")?;
            let o = fnum(&x["oiCcy"])?;
            if let Some((call, k, exp)) = parse_other(true, id) {
                if exp > now_us && o > 0.0 {
                    v.push(("OKX".into(), cur.into(), id.into(), call, k, exp, o, ivm.get(id).copied(), None));
                }
            }
        }
        Ok(v)
    })();
    match okx {
        Ok(v) => out.extend(v),
        Err(e) => log(format!("{cur}: OKX 옵션 실패 {e}")),
    }
    let by = (|| -> Result<Vec<Other>, String> {
        let r = getj(get, "https://api.bybit.com/v5/market/tickers", &[("category", "option".into()), ("baseCoin", cur.into())], 20, true)?;
        let mut v = vec![];
        for x in r["result"]["list"].as_array().ok_or("KeyError: 'list'")? {
            let s = x["symbol"].as_str().ok_or("TypeError")?;
            let o = fnum(&x["openInterest"])?;
            if let Some((call, k, exp)) = parse_other(false, s) {
                if o > 0.0 {
                    let iv = x.get("markIv").filter(|v| truthy(v)).map(fnum).transpose()?.map(|v| v * 100.0);
                    let vol = x.get("volume24h").filter(|v| truthy(v)).map(fnum).transpose()?.unwrap_or(0.0);
                    v.push(("Bybit".into(), cur.into(), s.into(), call, k, exp, o, iv, Some(vol)));
                }
            }
        }
        Ok(v)
    })();
    match by {
        Ok(v) => out.extend(v),
        Err(e) => log(format!("{cur}: Bybit 옵션 실패 {e}")),
    }
    out
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn fnum(v: &Value) -> Result<f64, String> {
    match v {
        Value::Number(n) => Ok(n.as_f64().unwrap()),
        Value::String(s) => s.trim().parse().map_err(|_| format!("ValueError: float({s:?})")),
        o => Err(format!("TypeError: float({o})")),
    }
}

/// `poll_once(con)` -- 반환 = 로그
pub fn poll_once(con: &Connection, state: &Path, now_us: i64, db_now_us: i64, tz_off: i64, cache: &mut Cache, get: Get) -> Result<Vec<String>, String> {
    let mut log: Vec<String> = vec![];
    let mut lg = |s: String| log.push(s);
    ensure_tables(con).map_err(|e| e.to_string())?;
    for cur in OTHER {
        let rows = fetch_other(cur, now_us, get, &mut lg);
        if !rows.is_empty() {
            let tx = con.unchecked_transaction().map_err(|e| e.to_string())?;
            for r in &rows {
                tx.execute("INSERT INTO option_oi_other VALUES (make_timestamp(?)::TIMESTAMPTZ, ?, ?, ?, ?, ?, make_timestamp(?)::TIMESTAMPTZ, ?, ?, ?)",
                    duckdb::params![now_us, r.0, r.1, r.2, if r.3 { "call" } else { "put" }, r.4, r.5, r.6, r.7, r.8]).map_err(|e| e.to_string())?;
            }
            tx.commit().map_err(|e| e.to_string())?;
            lg(format!("{cur}: OKX·Bybit 옵션 {}종목", rows.len()));
        }
    }
    for cur in CURRENCIES {
        let chain = match fetch_chain(cur, now_us, cache, get) {
            Ok(c) => c,
            Err(e) => {
                lg(format!("{cur}: 체인 조회 실패 {e}"));
                continue;
            }
        };
        if chain.is_empty() {
            lg(format!("{cur}: empty response, skipping"));
            continue;
        }
        {
            let tx = con.unchecked_transaction().map_err(|e| e.to_string())?;
            for c in &chain {
                tx.execute("INSERT INTO option_chain_snapshot VALUES (make_timestamp(?)::TIMESTAMPTZ, ?, ?, ?, ?, make_timestamp(?)::TIMESTAMPTZ, ?, ?, ?, ?, ?, ?, ?)",
                    duckdb::params![now_us, cur, c.inst, if c.call { "call" } else { "put" }, c.strike, c.exp_us, c.days, c.oi, c.iv, c.und, c.mark, c.vol, c.gamma]).map_err(|e| e.to_string())?;
            }
            tx.commit().map_err(|e| e.to_string())?;
        }
        match options_summary(&chain, cur, now_us, cache, get, &mut lg) {
            Ok(opt) => {
                let f0 = opt["expiries"].as_array().and_then(|a| a.first()).cloned().unwrap_or(json!({}));
                let fv = |k: &str| f0.get(k).and_then(Value::as_f64);
                con.execute("INSERT INTO option_summary VALUES (make_timestamp(?)::TIMESTAMPTZ, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    duckdb::params![now_us, cur, opt["index"].as_f64(), opt["dvol"].as_f64(), opt["rv7"].as_f64(), fv("atm_iv"), fv("rr25"), fv("bf25"), Option::<f64>::None,
                                    rtcollect::py_dumps_default(&Value::Object(opt.clone()))]).map_err(|e| e.to_string())?;
                lg(format!("{cur}: index={} n={}", opt["index"], chain.len()));
            }
            Err(e) => lg(format!("{cur}: options_summary 실패 {e}")),
        }
    }
    write_state(con, state, db_now_us, now_us, tz_off)?;
    Ok(log)
}

fn jnum(v: Option<&Value>) -> Option<f64> {
    v.and_then(Value::as_f64)
}

/// `write_state(con)` -- 원본 SQL 의 vov24·atm_oi_pct 를 같은 행으로 Rust 에서(머리 설명).
pub fn write_state(con: &Connection, state: &Path, db_now_us: i64, gen_us: i64, tz_off: i64) -> Result<(), String> {
    let mut cur_m = Map::new();
    for cur in CURRENCIES {
        let row: Option<(i64, Option<f64>, Option<String>)> = con.query_row("SELECT epoch_us(recorded_at_utc), index_price, payload FROM option_summary WHERE currency = ? ORDER BY recorded_at_utc DESC LIMIT 1",
            [cur], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))).ok();
        let Some((ts, spot, payload)) = row else { continue };
        let mut oh = Map::new();
        oh.insert("vov24".into(), Value::Null);
        oh.insert("atm_oi_pct".into(), Value::Null);
        oh.insert("atm_oi_n".into(), json!(0));
        let rows: Vec<(i64, Option<f64>, Option<String>)> = con.prepare("SELECT epoch_us(recorded_at_utc), dvol, payload FROM option_summary WHERE currency = ? AND epoch_us(recorded_at_utc) >= ?")
            .and_then(|mut s| s.query_map(duckdb::params![cur, db_now_us - 30 * 86_400_000_000], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.collect()).map_err(|e| e.to_string())?;
        let pj = |p: &Option<String>| p.as_deref().and_then(|t| serde_json::from_str::<Value>(t).ok());
        // vov24: 지난 25시간 시간별 arg_max(coalesce(dvol, iv30), t)
        let mut hours: BTreeMap<i64, (i64, f64)> = BTreeMap::new();
        for (t, dvol, p) in rows.iter().filter(|r| r.0 >= db_now_us - 25 * 3_600_000_000) {
            let v = dvol.or_else(|| jnum(pj(p).as_ref().and_then(|j| j.get("iv30"))));
            if let Some(v) = v {
                let h = t.div_euclid(3_600_000_000);
                if hours.get(&h).is_none_or(|x| *t > x.0) {
                    hours.insert(h, (*t, v));
                }
            }
        }
        let v: Vec<f64> = hours.values().map(|x| x.1).filter(|x| *x != 0.0 && *x > 0.0).collect();
        if v.len() >= 12 {
            let d: Vec<f64> = v.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
            let mut s = 0.0;
            for x in &d {
                s += x;
            }
            let m = s / d.len() as f64;
            let mut q = 0.0;
            for x in &d {
                q += (x - m) * (x - m);
            }
            oh.insert("vov24".into(), json!((q / (d.len().saturating_sub(1).max(1)) as f64).sqrt() * 100.0));
        }
        // atm_oi_pct: 30일 안 UTC 07:00~07:09 의 front_atm_oi_usd
        let hist: Vec<f64> = rows.iter().filter(|r| {
            let s = r.0.div_euclid(1_000_000).rem_euclid(86_400);
            s / 3600 == 7 && s / 60 % 60 < 10
        }).filter_map(|r| jnum(pj(&r.2).as_ref().and_then(|j| j.get("front_atm_oi_usd")))).collect();
        let opts = pj(&payload).unwrap_or(Value::Null);
        if let Some(c) = jnum(opts.get("front_atm_oi_usd")) {
            if hist.len() >= 5 {
                oh.insert("atm_oi_pct".into(), json!(hist.iter().filter(|x| **x <= c).count() as f64 / hist.len() as f64));
                oh.insert("atm_oi_n".into(), json!(hist.len()));
            }
        }
        let mut m = Map::new();
        m.insert("opt_hist".into(), Value::Object(oh));
        m.insert("options".into(), opts);
        m.insert("recorded_at_utc".into(), json!(iso(ts, tz_off)));
        m.insert("spot_price".into(), spot.map_or(Value::Null, |x| json!(x)));
        cur_m.insert(cur.into(), Value::Object(m));
    }
    let out = json!({"generated_at": iso_utc(gen_us), "currencies": cur_m});
    let tmp = state.with_extension("json.tmp");
    std::fs::write(&tmp, rtcollect::py_dumps_default_utf8(&out)).and_then(|_| std::fs::rename(&tmp, state)).map_err(|e| e.to_string())
}
