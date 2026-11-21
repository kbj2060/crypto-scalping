//! 셋째 조각(2026-11-21): 저장소 + 계산(요청 때 REST 없음) 묶음 중 옮긴 것 -- /api/session-alerts · /api/liquidation-5m-signal.
//! 원본 위치는 함수마다. 캐시는 swr.rs(원본 swr_cached 와 같은 ttl·max_stale).

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::api::{coin_asset, iso_parse, macro_calendar_payload, now_us, or500};
use crate::py::{self, err, f, item, PyErr, R, V};
use crate::{Config, Handler, Req, Resp};

pub fn routes(p: &str) -> Vec<(&'static str, Handler)> {
    match p {
        "/api/session-alerts" => vec![("GET", session_alerts)],
        "/api/liquidation-5m-signal" => vec![("GET", liquidation_5m_signal)],
        "/api/flow/heatmap" => vec![("GET", flow_heatmap)],
        "/api/footprint-daily" => vec![("GET", fp_daily)],
        "/api/footprint-daily/cells" => vec![("GET", fp_daily_cells)],
        "/api/footprint-4h" => vec![("GET", fp_4h)],
        "/api/footprint-4h/cells" => vec![("GET", fp_4h_cells)],
        _ => Vec::new(),
    }
}

/// 원본 load_macro_calendar(server.py:3491): swr_cached("macro_calendar", 6h, …, max_stale=600) -- /api/macro-calendar 와 같은 캐시
pub fn macro_calendar_cached(cfg: &Config) -> R<V> {
    let c = cfg.clone();
    cfg.swr.get("macro_calendar", 6.0 * 3600.0, 600.0, Arc::new(move || macro_calendar_payload(&c.live_dir().join("macro_calendar_state.json"), (c.now)())))
}

/// scripts/live_macro_calendar_20260826.py:633 compute_macro_event_alert(창 ±30분 · importance=high 만)
fn macro_event_alert(events: &V, now: f64) -> R<V> {
    let Value::Array(es) = events else { return err("TypeError", "events") };
    let mut active = Vec::new();
    for e in es {
        if py::get(e, "importance")? != Some(&json!("high")) {
            continue;
        }
        let t = match item(e, "time_utc") {
            Ok(Value::String(s)) => s.clone(),
            Ok(v) => return err("TypeError", format!("fromisoformat: argument must be str, not {}", py::type_name(v))),
            Err(_) => continue, // KeyError
        };
        let Some((us, aware)) = iso_parse(&t) else { continue }; // ValueError
        if !aware {
            return err("TypeError", "can't subtract offset-naive and offset-aware datetimes");
        }
        let minutes = (now_us(now) - us) as f64 / 1e6 / 60.0;
        if (-30.0..=30.0).contains(&minutes) {
            active.push(json!({"title_ko": item(e, "title_ko")?, "minutes_from_event": f(py::round_nd(minutes, 1))}));
        }
    }
    Ok(json!({"active": active}))
}

/// GET /api/session-alerts (server.py:4840) -- 달력(6시간 캐시) + 지금 시각으로 매번 계산
fn session_alerts(cfg: &Config, _: &crate::Req) -> Resp {
    or500(macro_calendar_cached(cfg).and_then(|cal| {
        let ev = py::get(&cal, "events")?.cloned().unwrap_or(json!([]));
        Ok(json!({"macro_event_alert": macro_event_alert(&ev, (cfg.now)())?}))
    }))
}

// ───────────────────────── 청산 게이지 (scripts/live_liquidation_5m_signal_20260825.py:144) ─────────────────────────

/// scripts/coin_config.py tail_risk_db_path · tail_risk_table
fn tail_risk(asset: &str) -> (&'static str, &'static str) {
    match asset {
        "btc" => ("tail_risk_btc_sol.duckdb", "tail_risk_1m_btc"),
        "sol" => ("tail_risk_btc_sol.duckdb", "tail_risk_1m_sol"),
        "eth" => ("tail_risk.duckdb", "tail_risk_1m"),
        "xrp" => ("tail_risk_xrp.duckdb", "tail_risk_1m_xrp"),
        "hype" => ("tail_risk_hype.duckdb", "tail_risk_1m_hype"),
        "sndk" => ("tail_risk_sndk.duckdb", "tail_risk_1m_sndk"),
        "soxl" => ("tail_risk_soxl.duckdb", "tail_risk_1m_soxl"),
        "skhynix" => ("tail_risk_skhynix.duckdb", "tail_risk_1m_skhynix"),
        _ => ("tail_risk_mu.duckdb", "tail_risk_1m_mu"),
    }
}

const BAR_MIN: i64 = 30;

/// pandas Timestamp(UTC).isoformat() / datetime.isoformat() -- 마이크로초가 0 이면 안 붙는다
fn iso_utc(us: i64) -> String {
    let secs = us.div_euclid(1_000_000);
    let frac = us.rem_euclid(1_000_000);
    let (y, m, d) = crate::api2::civil(secs.div_euclid(86400));
    let s = secs.rem_euclid(86400);
    let base = format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60);
    if frac == 0 { format!("{base}+00:00") } else { format!("{base}.{frac:06}+00:00") }
}

fn read_tail(db: &PathBuf, table: &str) -> Result<Vec<(i64, Option<f64>, Option<f64>)>, duckdb::Error> {
    let con = duckdb::Connection::open_with_flags(db, duckdb::Config::default().access_mode(duckdb::AccessMode::ReadOnly)?)?;
    let mut st = con.prepare(&format!("SELECT epoch_us(ts), long_usd_1m, short_usd_1m FROM {table} ORDER BY ts DESC LIMIT ?"))?;
    let rows = st.query_map([60i64], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

pub fn liq5m_payload(cfg: &Config, asset: &str) -> R<V> {
    let (file, table) = tail_risk(asset);
    let db = cfg.live_dir().join(file);
    let empty = |e: V| obj(vec![("warmed_up", json!(false)), ("error", e), ("long_usd_5m", Value::Null), ("short_usd_5m", Value::Null), ("bars_used", json!(0)),
        ("bar_start_utc", Value::Null), ("bar_elapsed_sec", Value::Null), ("latest_ts_utc", Value::Null)]);
    if !db.exists() {
        return Ok(empty(json!("db_missing")));
    }
    // 원본: 쓰는 쪽과 잠금이 겹치면 0·0.4·0.8·1.6초 뒤 다시
    let mut last = None;
    let mut rows = Vec::new();
    for delay in [0.0, 0.4, 0.8, 1.6] {
        if delay > 0.0 {
            std::thread::sleep(std::time::Duration::from_secs_f64(delay));
        }
        match read_tail(&db, table) {
            Ok(r) => {
                rows = r;
                last = None;
                break;
            }
            Err(e) => last = Some(e),
        }
    }
    if let Some(e) = last {
        // ponytail: 오류 글자는 duckdb-rs 문구(원본은 파이썬 duckdb 예외 글자) -- 화면은 warmed_up=false 만 본다
        return Ok(empty(json!(format!("db_read_error: {e}"))));
    }
    if rows.is_empty() {
        return Ok(empty(json!("no_rows")));
    }
    let now = now_us((cfg.now)());
    let secs = now.div_euclid(1_000_000);
    let bar_start = (secs - secs.rem_euclid(60 * BAR_MIN)) * 1_000_000; // 분을 30분 단위로 내림 · 초·마이크로초 0
    let in_bar: Vec<&(i64, Option<f64>, Option<f64>)> = rows.iter().filter(|r| r.0 >= bar_start).collect();
    let latest = iso_utc(rows[0].0);
    let elapsed = f((now - bar_start) as f64 / 1e6);
    if in_bar.is_empty() {
        return Ok(obj(vec![("warmed_up", json!(true)), ("error", Value::Null), ("long_usd_5m", f(0.0)), ("short_usd_5m", f(0.0)), ("bars_used", json!(0)),
            ("bar_start_utc", json!(iso_utc(bar_start))), ("bar_elapsed_sec", elapsed), ("latest_ts_utc", json!(latest))]));
    }
    let sum = |k: usize| -> f64 {
        let v: Vec<f64> = in_bar.iter().map(|r| if k == 1 { r.1 } else { r.2 }.unwrap_or(0.0)).collect();
        rtcollect::ob5m::np_sum(&v) // pandas Series.sum = numpy add.reduce(공유 함수)
    };
    Ok(obj(vec![("warmed_up", json!(true)), ("error", Value::Null), ("long_usd_5m", f(sum(1))), ("short_usd_5m", f(sum(2))), ("bars_used", json!(in_bar.len())),
        ("bar_start_utc", json!(iso_utc(bar_start))), ("bar_elapsed_sec", elapsed), ("latest_ts_utc", json!(latest))]))
}

fn obj(pairs: Vec<(&str, V)>) -> V {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect::<Map<String, V>>())
}

/// GET /api/liquidation-5m-signal (server.py:4751 · 캐시 LIQUIDATION_5M_SIGNAL_CACHE_SECONDS=1 · max_stale 0)
fn liquidation_5m_signal(cfg: &Config, req: &crate::Req) -> Resp {
    let a = match coin_asset(req) {
        Ok(a) => a,
        Err(r) => return r,
    };
    let c = cfg.clone();
    let key = format!("liquidation_5m:{a}");
    or500(cfg.swr.get(&key, 1.0, 0.0, Arc::new(move || liq5m_payload(&c, &a))).map_err(|PyErr(m)| PyErr(m)))
}

// ───────────────── 일봉·4시간 풋프린트 4(server.py 3714~3751 손잡이 · 3861~3912 엔드포인트 · 계산은 fpd.rs) ─────────────────

/// 원본 FPD 표 + fp_daily(코인마다 DailyFootprint 하나: lake = data_store.LAKE(DATA_LAKE 또는 data/lake) · hot 테이프 = data/hot/binance_tape.sqlite · 맥락 = binance_ctx.sqlite)
/// ponytail: data_store 의 «data/lake 가 없으면 백업 lake» 갈래는 안 옮겼다 -- 서버엔 data/lake 가 있다 · 필요하면 DATA_LAKE 로
fn fpd_of<'a>(cfg: &'a Config, a: &str) -> Option<&'a crate::fpd::Fpd> {
    cfg.fpd.get_or_init(|| {
        let lake = std::env::var("DATA_LAKE").map(PathBuf::from).unwrap_or_else(|_| cfg.root.join("data/lake"));
        crate::fpd::ASSETS.iter().filter_map(|x| {
            let src = crate::fpd::Src { lake: lake.clone(), hist: cfg.root.join("data/footprint_daily"), hot_tape: cfg.root.join("data/hot/binance_tape.sqlite"),
                                        hot_ctx: cfg.root.join("data/hot/binance_ctx.sqlite") };
            crate::fpd::Fpd::new(x.0, src).map(|f| (x.0.to_string(), f))
        }).collect()
    }).get(a)
}

/// _fpd_asset: ?asset= 그대로(소문자로 안 바꿈 · 없으면 eth) · 표 밖 = 400 asset
fn fpd_asset<'a>(cfg: &'a Config, req: &Req) -> Result<(String, &'a crate::fpd::Fpd), Resp> {
    let a = req.q("asset").unwrap_or("eth").to_string();
    match fpd_of(cfg, &a) {
        Some(f) => Ok((a, f)),
        None => Err(crate::http_exc(400, Some("asset"), &[])),
    }
}

/// _iso_day: 없음 = None · 있으면 'YYYY-MM-DD'(date.fromisoformat) 아니면 400 bad_date
fn iso_day_q(req: &Req, k: &str) -> Result<Option<String>, Resp> {
    req.q(k).map(|v| crate::fpd::iso_day(v).ok_or_else(|| crate::http_exc(400, Some("bad_date"), &[]))).transpose()
}

/// _iso_h4: 없음 = None · h4_key(_h4_sec(v)) == v 가 아니면 400 bad_key
fn iso_h4_q(req: &Req, k: &str) -> Result<Option<String>, Resp> {
    req.q(k).map(|v| if crate::fpd::iso_h4(v) { Ok(v.to_string()) } else { Err(crate::http_exc(400, Some("bad_key"), &[])) }).transpose()
}

/// 칸 계산 → swr(30 · 낡은 값 600) · 원본 to_thread 의 예외 = 500(ValueError 만 cells 에서 400 range)
fn fp_swr(cfg: &Config, key: String, f: impl Fn(&crate::fpd::Fpd, f64) -> Result<V, String> + Send + Sync + 'static, a: String) -> R<V> {
    let c = cfg.clone();
    cfg.swr.get(&key, 30.0, 600.0, Arc::new(move || f(fpd_of(&c, &a).unwrap(), (c.now)()).map_err(PyErr)))
}

macro_rules! tri {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(r) => return r,
        }
    };
}

fn fp_daily(cfg: &Config, req: &Req) -> Resp {
    let (a, _) = tri!(fpd_asset(cfg, req));
    let since = tri!(iso_day_q(req, "since"));
    match fp_swr(cfg, format!("fp_daily_days_{a}"), |f, now| f.days_payload(now), a) {
        Ok(full) => Resp::json(&crate::fpd::cols_since(&full, since.as_deref())),
        Err(_) => crate::server_error(),
    }
}

const CELL_KEYS: [&str; 3] = ["fp_daily_cells_", "fp_4h_cells_", "position_profile:"];

fn fp_daily_cells(cfg: &Config, req: &Req) -> Resp {
    let (a, f) = tri!(fpd_asset(cfg, req));
    let (frm, to) = (tri!(iso_day_q(req, "from")), tri!(iso_day_q(req, "to")));
    let (Some(frm), Some(to)) = (frm, to) else { return crate::http_exc(400, Some("from_to"), &[]) };
    let row = crate::fpd::parse_row(req.q("row"), f.sp.cell);
    cfg.swr.prune_oldest_half(800, &CELL_KEYS);
    let (f2, t2) = (frm.clone(), to.clone());
    match fp_swr(cfg, format!("fp_daily_cells_{a}_{frm}_{to}_{row}"), move |f, now| f.cells_payload(now, &f2, &t2, row), a) {
        Ok(v) => Resp::json(&v),
        Err(PyErr(m)) if m == "range" => crate::http_exc(400, Some("range"), &[]),
        Err(_) => crate::server_error(),
    }
}

fn fp_4h(cfg: &Config, req: &Req) -> Resp {
    let (a, _) = tri!(fpd_asset(cfg, req));
    let since = tri!(iso_h4_q(req, "since"));
    match fp_swr(cfg, format!("fp_4h_days_{a}"), |f, now| f.h4_payload(now), a) {
        Ok(full) => Resp::json(&crate::fpd::cols_since(&full, since.as_deref())),
        Err(_) => crate::server_error(),
    }
}

fn fp_4h_cells(cfg: &Config, req: &Req) -> Resp {
    let (a, f) = tri!(fpd_asset(cfg, req));
    let (frm, to) = (tri!(iso_h4_q(req, "from")), tri!(iso_h4_q(req, "to")));
    let (Some(frm), Some(to)) = (frm, to) else { return crate::http_exc(400, Some("from_to"), &[]) };
    let row = crate::fpd::parse_row(req.q("row"), f.sp.cell);
    let (f2, t2) = (frm.clone(), to.clone());
    match fp_swr(cfg, format!("fp_4h_cells_{a}_{frm}_{to}_{row}"), move |f, now| f.h4_cells_payload(now, &f2, &t2, row), a) {
        Ok(v) => Resp::json(&v),
        Err(_) => crate::server_error(),
    }
}

// ───────────────── /api/flow/heatmap(server.py api_flow_heatmap 4692~4732 · 계산 heatmap.rs · 읽기 raster_read.rs) ─────────────────

const HEATMAP_SYMBOLS: [&str; 8] = ["ethusdt", "btcusdt", "solusdt", "xrpusdt", "sndkusdt", "soxlusdt", "skhynixusdt", "muusdt"]; // server.py:396
const HEATMAP_MAX_COLS: i128 = 900; // server.py:397

/// 체결 대조 원천: 그 심볼의 흐름 엔진 풋프린트(허브 칸 · 가격 = round(키 × 칸 폭, dp) · 매수·매도 = round(x, 3) -- 원본은 반올림 전 칸 · 문서 §7-2 실측)
fn heatmap_fp(cfg: &Config, symbol: &str) -> Option<crate::heatmap::Fp> {
    if !cfg.flows().iter().any(|a| format!("{a}usdt") == symbol) {
        return None; // 원본 LookupError → fill_source None
    }
    let g = cfg.live.syms.lock().unwrap();
    Some(g.get(symbol).map_or_else(Vec::new, |s| s.fp_bars.iter().map(|(t, (cells, _))| (*t, cells.values().map(|l| (l.0, l.1, l.2)).collect())).collect()))
}

fn flow_heatmap(cfg: &Config, req: &Req) -> Resp {
    let symbol = req.q("symbol").filter(|v| !v.is_empty()).unwrap_or("ethusdt").to_lowercase();
    if !HEATMAP_SYMBOLS.contains(&symbol.as_str()) {
        return crate::http_exc(400, Some("unsupported_symbol"), &[]);
    }
    let (Some(cols), Some(agg)) = (crate::api::py_int_str(req.q("cols").unwrap_or("300")), crate::api::py_int_str(req.q("agg").unwrap_or("3"))) else {
        return crate::http_exc(400, Some("bad_cols_or_agg"), &[]);
    };
    let (cols, agg) = (cols.clamp(10, HEATMAP_MAX_COLS) as usize, agg.clamp(1, 60) as usize);
    let rows_only = req.q("mode") == Some("rows");
    let root = crate::raster_read::default_root(&cfg.root);
    let now_ms = || ((cfg.now)() * 1000.0) as i64;
    let (c, r2, s2) = (cfg.clone(), root.clone(), symbol.clone());
    let mut payload = match cfg.swr.get(&format!("heatmap_{symbol}_{cols}_{agg}_{}", rows_only as i32), 1.0, 0.0, Arc::new(move || {
        crate::heatmap::heatmap_read(&r2, &s2, ((c.now)() * 1000.0) as i64, cols, agg, rows_only, heatmap_fp(&c, &s2).as_ref()).map_err(PyErr)
    })) {
        Ok(v) => v,
        Err(_) => return crate::server_error(),
    };
    // 접근행동(4시간 · 120초 캐시) → rows 에 · 변동 신호 자(300초) · 지금 값(1초) → summary 에. 실패는 삼킨다(원본 try/except pass)
    if rows_only && payload["rows"].as_object().is_some_and(|m| !m.is_empty()) {
        let (r2, s2, t) = (root.clone(), symbol.clone(), now_ms());
        if let Ok(Value::Object(ap)) = cfg.swr.get(&format!("approach_{symbol}"), 120.0, 600.0, Arc::new(move || Ok(Value::Object(crate::heatmap::approach_read(&r2, &s2, t))))) {
            payload["rows"].as_object_mut().unwrap().extend(ap);
            let (r2, s2) = (root.clone(), symbol.clone());
            let rf = cfg.swr.get(&format!("volsig_ref_{symbol}"), 300.0, 600.0, Arc::new(move || {
                Ok(crate::heatmap::volsig_ref(&r2, &s2, t).map_or(json!({}), |(p, r, s)| json!({"past": p, "refill": r, "score": s})))
            }));
            if let (Ok(rf), true) = (rf, payload["summary"].as_object().is_some_and(|m| !m.is_empty())) {
                let (r2, s2) = (root.clone(), symbol.clone());
                if let Ok(Value::Object(vs)) = cfg.swr.get(&format!("volsig_{symbol}"), 1.0, 0.0, Arc::new(move || {
                    let fv = |k: &str| rf[k].as_array().map(|a| a.iter().filter_map(Value::as_f64).collect::<Vec<f64>>());
                    Ok(Value::Object(match (fv("past"), fv("refill"), fv("score")) {
                        (Some(p), Some(r), Some(s)) => crate::heatmap::volsig_read(&r2, &s2, t, &(p.iter().map(|x| *x as f32).collect(), r.iter().map(|x| *x as f32).collect(), s)),
                        _ => Map::new(), // 원본 _volsig_read: ref 가 {} 면 {}
                    }))
                })) {
                    payload["summary"].as_object_mut().unwrap().extend(vs);
                }
            }
        }
    }
    Resp::json(&payload)
}
