//! 위험 레이더 cron 한 회차 -- scripts/live_risk_radar_20261115.py `main()`(+ `update_cache`·`update_dvol`)를 옮긴 것. 원본처럼 매분 cron 이 한 번 부르고 끝난다.
//! 코인(계수 파일 4코인 + 주식 세션 SOXL)마다: 1분봉 캐시 이어 받기(처음 31일 백필 · 닫힌 봉만) → (ETH·BTC) Deribit DVOL 시간봉(시간에 한 번 · 실패면 캐시로) →
//! σ̂·띠·앞 1시간·급변(rtmodel::risk_radar) → risk_radar_latest.json(원자적 교체) + risk_radar_forward.jsonl(줄 덧붙임). 그림자 = 다른 디렉터리(rtmodel/SHADOW.md).
//! 캐시는 원본 parquet 대신 이 실행 파일만의 작은 이진 파일(그림자 디렉터리 cache/) -- 값은 같은 행.
//!
//! 환경변수(기본값):
//!   RT_RADAR_OUT          data/live/rt_risk_radar      🔴data/live(= 파이썬 워커의 latest·forward·캐시 자리)면 종료코드 2(전환 때만 RT_RADAR_TAKEOVER=1)
//!   RT_RADAR_PARAMS       dashboard/risk_radar_params_20261116.json · RT_RADAR_PARAMS_SESS dashboard/risk_radar_params_soxl_20261117.json(없으면 SOXL 빠짐)
//!   RT_RADAR_FAPI         https://fapi.binance.com     · RT_RADAR_DERIBIT https://www.deribit.com
//!   RT_SR_BAN_FILE        data/live/binance_ban_until  (바이낸스만 -- 원본 binance_ban_guard 도 바이낸스 주소만 막는다)
//!   RT_RADAR_NOW_MS       (시험만) 고정 시각 · 백필 쪽 사이 0.5초 쉼을 건너뜀
//! 종료코드: 0 = 모든 코인 정상 · 1 = 한 코인이라도 오류(원본과 같음) · 2 = 설정 거부.
use rtmodel::ban::{ban_left, ban_note};
use rtmodel::risk_radar::{coin_out, iso, session_out, Bars, CoinP, Dvol, Params, J, STALE_DATA_S};
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

const MIN: i64 = 60_000;
const HOUR: i64 = 3_600_000;
const KEEP: usize = 31 * 1440 + 60; // «평소»(720 완결 시간 중 600 이상) + 7일 RV
const IV_KEEP_H: i64 = 900;

fn env(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.into())
}
fn die(code: i32, msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(code)
}
fn wall_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}

struct Http {
    agent: ureq::Agent,
    ban: PathBuf,
}

impl Http {
    /// requests.get(url, params, timeout=20).raise_for_status().json(). 바이낸스면 밴 파일을 먼저 본다.
    fn get(&self, url: &str, q: &[(&str, String)], binance: bool) -> Result<Value, String> {
        let now = wall_ms() as f64 / 1000.0;
        if binance && ban_left(&self.ban, now) > 0.0 {
            return Err(format!("BinanceBanned: {:.0}s left -- request not sent", ban_left(&self.ban, now)));
        }
        let qs: Vec<String> = q.iter().map(|(k, v)| format!("{k}={v}")).collect();
        let mut r = self.agent.get(&format!("{url}?{}", qs.join("&"))).call().map_err(|e| format!("ConnectionError: {e}"))?;
        let h = |n: &str| r.headers().get(n).and_then(|v| v.to_str().ok()).map(str::to_string);
        let (status, retry) = (r.status().as_u16(), h("retry-after"));
        let body = r.body_mut().read_to_string().map_err(|e| format!("ChunkedEncodingError: {e}"))?;
        if binance {
            ban_note(&self.ban, status, &body, retry.as_deref(), wall_ms() as f64 / 1000.0);
        }
        if status >= 400 {
            return Err(format!("HTTPError: {status}"));
        }
        serde_json::from_str(&body).map_err(|e| format!("JSONDecodeError: {e}"))
    }
}

// ── 캐시(이 실행 파일 전용 이진 형식: 행마다 i64 + f64 × k · 원자적 교체 · 깨지면 .bad 로 치우고 처음부터)
fn save_rows(p: &Path, rows: &[(i64, Vec<f64>)]) {
    let mut b = Vec::with_capacity(rows.len() * 40);
    for (t, v) in rows {
        b.extend(t.to_le_bytes());
        for x in v {
            b.extend(x.to_le_bytes());
        }
    }
    let tmp = p.with_extension("tmp");
    if let Some(d) = p.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    if let Err(e) = std::fs::write(&tmp, &b).and_then(|_| std::fs::rename(&tmp, p)) {
        eprintln!("캐시 저장 실패 {}: {e}", p.display());
    }
}
fn load_rows(p: &Path, k: usize) -> Vec<(i64, Vec<f64>)> {
    let Ok(b) = std::fs::read(p) else { return vec![] };
    let w = 8 * (k + 1);
    if b.len() % w != 0 {
        let _ = std::fs::rename(p, p.with_extension("bad")); // 깨진 캐시는 치워 두고 다시 백필
        return vec![];
    }
    b.chunks(w).map(|c| {
        let f = |i: usize| f64::from_le_bytes(c[8 * i..8 * i + 8].try_into().unwrap());
        (i64::from_le_bytes(c[..8].try_into().unwrap()), (1..=k).map(f).collect())
    }).collect()
}

fn num(v: &Value) -> Result<f64, String> {
    v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())).ok_or_else(|| format!("ValueError: float({v})"))
}

/// update_cache: 닫힌 1분봉(t·h·l·c·v)을 이어 붙인다. 처음엔 31일 백필(limit 1000 · 쪽 사이 0.5초) · 평소엔 몇 개(limit 99) · 쪽마다 저장.
fn update_cache(http: &Http, base: &str, coin: &str, now: i64, path: &Path, fixed: bool) -> Result<Vec<(i64, Vec<f64>)>, String> {
    let mut df = load_rows(path, 4);
    let mut start = match df.last() { Some(r) => r.0, None => (now / MIN - KEEP as i64) * MIN }; // 마지막 봉을 한 번 더 받아 덮는다
    if let Some(last) = df.last() {
        if df[df.len().saturating_sub(1441)..].iter().any(|r| r.1[3].is_nan()) {
            start = start.min(last.0 - 1500 * MIN); // 거래대금 없는 옛 캐시 → 1일+ 다시 받아 덮는다
        }
    }
    while start < now - MIN {
        let lim = if now - start > 99 * MIN { 1000 } else { 99 };
        let k = http.get(&format!("{base}/fapi/v1/klines"), &[("symbol", format!("{coin}USDT")), ("interval", "1m".into()), ("startTime", start.to_string()), ("limit", lim.to_string())], true)?;
        let mut rows = vec![];
        for x in k.as_array().ok_or("TypeError: klines")? {
            let close = x[6].as_i64().ok_or("TypeError")?;
            if close + 1000 < now {
                // 닫힌 봉만 -- 1초 여유(서버 시계가 조금 빨라도 형성 봉이 안 들어오게 · 검증 11-16)
                rows.push((x[0].as_i64().ok_or("TypeError")?, vec![num(&x[2])?, num(&x[3])?, num(&x[4])?, num(&x[5])?]));
            }
        }
        let Some(&(last_t, _)) = rows.last().map(|r| (r.0, ())) .as_ref() else { break };
        let mut m: BTreeMap<i64, Vec<f64>> = df.into_iter().collect();
        for (t, v) in rows {
            m.insert(t, v); // drop_duplicates("t", keep="last")
        }
        df = m.into_iter().collect();
        if df.len() > KEEP {
            df.drain(..df.len() - KEEP);
        }
        save_rows(path, &df);
        if last_t + MIN <= start {
            break;
        }
        start = last_t + MIN;
        if lim == 1000 && !fixed {
            std::thread::sleep(Duration::from_millis(500));
        }
    }
    Ok(df)
}

/// update_dvol: DVOL 시간봉 캐시(ts·close·got). 마지막 봉이 끝나면(또는 비었으면) 그 앞 3시간부터 다시 받아 덮는다 -- 시간에 한 번꼴.
fn update_dvol(http: &Http, base: &str, coin: &str, now: i64, path: &Path) -> Result<Vec<(i64, Vec<f64>)>, String> {
    let mut df = load_rows(path, 2); // [close, got]
    let mx = df.iter().map(|r| r.0).max();
    if mx.is_some_and(|m| now < m + HOUR) {
        return Ok(df);
    }
    let mut start = mx.map_or(now - IV_KEEP_H * HOUR, |m| m - 3 * HOUR);
    let mut rows = vec![];
    while start < now {
        let end = now.min(start + 500 * HOUR);
        let v = http.get(&format!("{base}/api/v2/public/get_volatility_index_data"),
                         &[("currency", coin.into()), ("start_timestamp", start.to_string()), ("end_timestamp", end.to_string()), ("resolution", "3600".into())], false)?;
        for x in v["result"]["data"].as_array().ok_or("KeyError: 'result'")? {
            rows.push((x[0].as_i64().ok_or("TypeError")?, num(&x[4])?));
        }
        start = end;
    }
    if !rows.is_empty() {
        let mut m: BTreeMap<i64, Vec<f64>> = df.into_iter().collect();
        for (t, c) in rows {
            m.insert(t, vec![c, now as f64]);
        }
        df = m.into_iter().filter(|r| r.0 >= now - IV_KEEP_H * HOUR).collect();
        save_rows(path, &df);
    }
    Ok(df)
}

fn bars(df: &[(i64, Vec<f64>)]) -> Bars {
    let col = |j: usize| df.iter().map(|r| r.1[j]).collect::<Vec<f64>>();
    Bars { t: df.iter().map(|r| r.0).collect(), h: col(0), l: col(1), c: col(2), v: col(3) }
}
fn dvol(df: &[(i64, Vec<f64>)]) -> Dvol {
    Dvol { ts: df.iter().map(|r| r.0).collect(), close: df.iter().map(|r| r.1[0]).collect(), got: df.iter().map(|r| r.1[1] as i64).collect() }
}

fn main() {
    let out = PathBuf::from(env("RT_RADAR_OUT", "data/live/rt_risk_radar"));
    let python = out.file_name() == Some("live".as_ref()) && out.parent().and_then(|d| d.file_name()) == Some("data".as_ref());
    if python && env("RT_RADAR_TAKEOVER", "") != "1" {
        die(2, &format!("🔴{} 는 파이썬 워커의 자리다(risk_radar_latest.json·forward.jsonl) -- 그림자는 다른 경로에(기본 data/live/rt_risk_radar). 전환이면 RT_RADAR_TAKEOVER=1", out.display()));
    }
    let pfile = PathBuf::from(env("RT_RADAR_PARAMS", "dashboard/risk_radar_params_20261116.json"));
    let ptxt = std::fs::read_to_string(&pfile).unwrap_or_else(|e| die(2, &format!("{}: {e}", pfile.display())));
    let pp: Params = serde_json::from_str(&ptxt).unwrap_or_else(|e| die(2, &format!("계수: {e}")));
    let order: Vec<String> = serde_json::from_str::<serde_json::Map<String, Value>>(&ptxt).ok()
        .and_then(|m| m.get("coins").and_then(Value::as_object).map(|c| c.keys().cloned().collect())).unwrap_or_default();
    let order: Vec<String> = { let raw: Value = serde_json::from_str(&ptxt).unwrap(); let mut v: Vec<(usize, String)> = order.into_iter().map(|c| (ptxt.find(&format!("\"{c}\":")).unwrap_or(usize::MAX), c)).collect(); v.sort(); let _ = raw; v.into_iter().map(|x| x.1).collect() }; // 파일 안 순서(원본 dict 순서)
    // 주식 세션 계수(없거나 깨져도 4코인은 그대로)
    let sess: Vec<(String, CoinP)> = std::fs::read_to_string(env("RT_RADAR_PARAMS_SESS", "dashboard/risk_radar_params_soxl_20261117.json")).ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok()).and_then(|v| v["coins"].as_object().cloned())
        .map(|m| m.into_iter().filter_map(|(k, v)| serde_json::from_value::<CoinP>(v).ok().map(|p| (k, p))).collect()).unwrap_or_default();
    let fixed: Option<i64> = std::env::var("RT_RADAR_NOW_MS").ok().map(|v| v.parse().unwrap_or_else(|_| die(2, "RT_RADAR_NOW_MS")));
    let now = fixed.unwrap_or_else(wall_ms);
    let (fapi, deribit) = (env("RT_RADAR_FAPI", "https://fapi.binance.com"), env("RT_RADAR_DERIBIT", "https://www.deribit.com"));
    let http = Http { agent: ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(20))).http_status_as_error(false).proxy(None).build().into(),
                      ban: PathBuf::from(env("RT_SR_BAN_FILE", "data/live/binance_ban_until")) };
    let cache = out.join("cache");
    let mut res: Vec<(&'static str, J)> = vec![];
    let mut all_ok = true;
    let coins: Vec<(String, &CoinP)> = order.iter().map(|c| (c.clone(), &pp.coins[c])).chain(sess.iter().map(|(c, p)| (c.clone(), p))).collect();
    for (coin, p) in coins {
        let r: Result<J, String> = (|| {
            let df = update_cache(&http, &fapi, &coin, now, &cache.join(format!("{coin}_1m.bin")), fixed.is_some())?;
            if df.is_empty() {
                return Err("IndexError: single positional indexer is out-of-bounds".into());
            }
            let b = bars(&df);
            if p.kind.as_deref() == Some("session") {
                return Ok(session_out(&b, p, now));
            }
            let dv = if p.beta60_iv.is_some() {
                let path = cache.join(format!("{coin}_dvol.bin"));
                match update_dvol(&http, &deribit, &coin, now, &path) {
                    Ok(d) => Some(d),
                    Err(e) => {
                        eprintln!("{coin} DVOL {e}"); // IV 실패는 1시간을 B2 식으로 떨어뜨릴 뿐 -- 캐시가 있으면 그것으로
                        Some(load_rows(&path, 2))
                    }
                }
            } else {
                None
            };
            let dv = dv.filter(|d| !d.is_empty()).map(|d| dvol(&d));
            Ok(coin_out(&b, p, &pp, dv.as_ref(), now))
        })();
        let o = r.unwrap_or_else(|e| J::O(vec![("error", J::S(e.chars().take(200).collect())), ("valid", J::B(false))]));
        if let J::O(kv) = &o {
            all_ok &= !kv.iter().any(|x| x.0 == "error");
        }
        res.push((Box::leak(coin.into_boxed_str()), o));
    }
    let name = pfile.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
    let (latest, lines) = rtmodel::risk_radar::assemble(now, &name, res);
    let _ = std::fs::create_dir_all(&out);
    let (lp, tmp) = (out.join("risk_radar_latest.json"), out.join("risk_radar_latest.tmp"));
    if let Err(e) = std::fs::write(&tmp, &latest).and_then(|_| std::fs::rename(&tmp, &lp)) {
        die(1, &format!("{}: {e}", lp.display()));
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(out.join("risk_radar_forward.jsonl")).unwrap_or_else(|e| die(1, &format!("forward: {e}")));
    for l in &lines {
        let _ = writeln!(f, "{l}");
    }
    println!("{} · 코인 {} · 지연 문턱 {STALE_DATA_S}초", iso(now), lines.len());
    std::process::exit(if all_ok { 0 } else { 1 });
}
