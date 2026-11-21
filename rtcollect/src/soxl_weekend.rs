//! SOXLUSDT 주말 전진 기록기 = scripts/live_soxl_weekend_forward_recorder_20261022.py `--record`(cron 매일 21:30 UTC) · `--status`.
//!   끝난 주말(월 16:00 ET 봉 + 10분) 중 장부에 없는 것만 한 줄씩: SOXL·ETH 의 토 00·월 00·월 09:30·월 16:00 ET 직전 5분봉 종가(+lag) · SOXL 펀딩 합·횟수.
//!   🔴판정(`--look`)·표본 안(`--insample`)은 옮기지 않았다 -- 사람이 한 번 돌리는 판정 도구라 파이썬 그대로(판정 기록 looks 장부도 파이썬 것).
//!   시각 = 뉴욕 벽시계(zoneinfo fold=0 = cboe_summary::et_to_utc) · 펀딩 0건이면 원본 sum([]) = 정수 0.
use crate::cboe_summary::{civil, days_from_civil, et_to_utc, BINANCE};
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
use std::path::Path;

const FWD_FROM: (i64, i64, i64) = (2026, 10, 24);
const SYMS: [&str; 2] = ["SOXLUSDT", "ETHUSDT"];
const POINTS: [(&str, i64, i64, i64); 4] = [("sat00", 0, 0, 0), ("mon00", 2, 0, 0), ("mon0930", 2, 9, 30), ("mon1600", 2, 16, 0)];

/// `points(sat)` -- 토요일 일 번호 → (이름, UTC ms).
fn points(sat: i64) -> Vec<(&'static str, i64)> {
    POINTS.iter().map(|&(k, dd, h, m)| {
        let (y, mo, d) = civil(sat + dd);
        (k, et_to_utc(y, mo, d, h, m, 0) * 1000)
    }).collect()
}

fn iso_date(day: i64) -> String {
    let (y, m, d) = civil(day);
    format!("{y:04}-{m:02}-{d:02}")
}

/// `closed_weekends(FWD_FROM, now_ms)`
fn closed_weekends(now_ms: i64) -> Vec<i64> {
    let mut d = days_from_civil(FWD_FROM.0, FWD_FROM.1, FWD_FROM.2);
    let mut out = vec![];
    while points(d)[3].1 + 600_000 <= now_ms {
        out.push(d);
        d += 7;
    }
    out
}

fn read_rows(out: &Path) -> Result<Vec<Value>, String> {
    let Ok(t) = std::fs::read_to_string(out) else { return Ok(vec![]) };
    t.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).map_err(|e| format!("JSONDecodeError: {e}"))).collect()
}

type Get<'a> = &'a mut dyn FnMut(&str) -> Result<Vec<u8>, String>;

fn get(g: Get, path: &str) -> Result<Value, String> {
    serde_json::from_slice(&g(&format!("{BINANCE}/fapi/v1/{path}"))?).map_err(|e| format!("JSONDecodeError: {e}"))
}

fn fnum(v: &Value) -> Result<f64, String> {
    v.as_str().and_then(|s| s.parse().ok()).or_else(|| v.as_f64()).ok_or_else(|| format!("ValueError: float({v})"))
}

/// `record_weekend(sat)`
fn record_weekend(sat: i64, now_s: f64, g: Get) -> Result<Map<String, Value>, String> {
    let pt = points(sat);
    let mut row = Map::new();
    row.insert("weekend".into(), json!(iso_date(sat)));
    for (k, v) in &pt {
        row.insert(format!("{k}_utc_ms"), json!(v));
    }
    for sym in SYMS {
        for (k, t) in &pt {
            let ks = get(g, &format!("klines?symbol={sym}&interval=5m&endTime={}&limit=1", t - 300_000))?;
            let k1 = ks.as_array().and_then(|a| a.last()).ok_or("IndexError: list index out of range")?;
            let px = fnum(&k1[4])?;
            let lag = (t - 300_000 - k1[0].as_i64().ok_or("ValueError: int")?).div_euclid(60_000);
            row.insert(format!("{sym}_{k}"), json!(px));
            if lag != 0 {
                row.insert(format!("{sym}_{k}_lag_min"), json!(lag));
            }
        }
    }
    let f = get(g, &format!("fundingRate?symbol=SOXLUSDT&startTime={}&endTime={}&limit=100", pt[0].1 + 1, pt[1].1))?;
    let f = f.as_array().ok_or("TypeError: funding")?;
    let mut sum = json!(0); // 원본 sum([]) = 정수 0
    for x in f {
        let r = fnum(&x["fundingRate"])?;
        sum = json!(sum.as_f64().unwrap() + r);
    }
    row.insert("SOXLUSDT_funding_sum".into(), sum);
    row.insert("SOXLUSDT_funding_n".into(), json!(f.len()));
    let s = now_s.floor() as i64;
    row.insert("recorded_at_utc".into(), json!(format!("{}T{:02}:{:02}:{:02}Z", iso_date(s.div_euclid(86_400)), s.rem_euclid(86_400) / 3600, s.rem_euclid(3600) / 60, s.rem_euclid(60))));
    Ok(row)
}

/// `record()` -- 반환 = 기록한 주말들. 중간에 실패하면(원본 예외) 그 앞 줄까지 남고 Err.
pub fn record(out: &Path, now_s: f64, g: Get) -> Result<Vec<String>, String> {
    let done: BTreeSet<String> = read_rows(out)?.iter().map(|r| r["weekend"].as_str().map(str::to_string).ok_or_else(|| "KeyError: 'weekend'".to_string())).collect::<Result<_, _>>()?;
    if let Some(p) = out.parent() {
        std::fs::create_dir_all(p).map_err(|e| format!("OSError: {e}"))?;
    }
    let mut wrote = vec![];
    for sat in closed_weekends((now_s * 1000.0) as i64) {
        if done.contains(&iso_date(sat)) {
            continue;
        }
        let row = record_weekend(sat, now_s, g)?;
        use std::io::Write;
        std::fs::OpenOptions::new().create(true).append(true).open(out)
            .and_then(|mut f| f.write_all((crate::py_dumps_default_utf8(&Value::Object(row)) + "\n").as_bytes())).map_err(|e| format!("OSError: {e}"))?;
        wrote.push(iso_date(sat));
    }
    Ok(wrote)
}

/// `--status` 한 줄
pub fn status(out: &Path) -> Result<String, String> {
    let from = iso_date(days_from_civil(FWD_FROM.0, FWD_FROM.1, FWD_FROM.2));
    let rows: Vec<Value> = read_rows(out)?.into_iter().filter(|r| r["weekend"].as_str().is_some_and(|w| w >= from.as_str())).collect();
    let last = rows.last().and_then(|r| r["weekend"].as_str()).unwrap_or("-");
    Ok(format!("전진 {}주말 기록 · 마지막 {last} · 1차 점검 26 · 최종 52", rows.len()))
}
