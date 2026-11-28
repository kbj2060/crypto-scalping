//! 넷째 조각(2026-11-22): 요청 때 계산하는 모델 핸들러 5개 -- 계산은 rtmodel::handlers(원본과 비트 대조는 rtmodel 시험), 여기는 원본 핸들러의 껍데기만:
//! 신뢰경계(코인 표) · swr(원본과 같은 ttl·max_stale) · 모델 다시 읽기(JSON mtime) · «모델 낡음»(rtmodel::model_stale → 오류, 원본과 다른 값을 안 냄) ·
//! 밴 가드(data/live/binance_ban_until · 보내지 않음) · 예외 → {"ok": false, "reason"}.
//! 원본: api_dir4h·dir4h_cached(server.py:2916~2945) · api_dir_short·dir_short_cached(2947~2975) · qx4_payload·api_vol_forecast(3049~3126) ·
//!       api_open_rev(3128~3140) · api_dir_signals(2796~2802) · fetch_binance_json(1372~1392) -- 전부 Read 로 읽은 줄.
//! 모델 JSON(rtmodel/tools/export_*.py)은 RTSERVE_MODELS(기본 data/live/rtmodel = rt_dirpred 전환 때 RT_DIR_MODELS 와 같은 곳):
//!   dir4h_<코인>.json · ds_<코인>.json + oos_<코인>.json · svm_<코인>.json · qx4.json. «낡음» 판정의 원본 joblib 자리 = data/live.
//! 적중 기록(원본 pred_record_safe)은 안 쓴다 -- rt_dirpred 가 5분마다 쓴다(같은 줄을 두 프로세스가 쓰지 않게 · 문서 §5).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime};

use rtmodel::dirrest::Rest;
use rtmodel::{dir4h, dirshort, qx4, signals::Svm};
use serde_json::{json, Value};

use crate::py::{PyErr, R, V};
use crate::{Config, Handler, Req, Resp};

pub fn routes(p: &str) -> Vec<(&'static str, Handler)> {
    let h: Handler = match p {
        "/api/dir4h" => api_dir4h,
        "/api/dir-short" => api_dir_short,
        "/api/vol-forecast" => api_vol_forecast,
        "/api/open-rev" => api_open_rev,
        "/api/dir-signals" => api_dir_signals,
        _ => return Vec::new(),
    };
    vec![("GET", h)]
}

type Oos = Vec<(&'static str, Vec<f64>)>;

/// 프로세스 안 상태(원본 dir4h_art·dirshort_art·vol4h_art·qx4_state 와 d4·ds 모듈의 _CACHE 둘)
#[derive(Default)]
pub struct State {
    r4: Mutex<Rest>,
    rs: Mutex<Rest>,
    a4: Mutex<HashMap<String, (Vec<SystemTime>, Arc<(dir4h::Art, V)>)>>,
    ads: Mutex<HashMap<String, (Vec<SystemTime>, Arc<(dirshort::Art, Oos, V)>)>>,
    svm: Mutex<HashMap<String, (Vec<SystemTime>, Arc<(Svm, V)>)>>,
    qx4: Mutex<Option<(Vec<SystemTime>, Arc<(qx4::Art, V)>)>>,
    qrows: Mutex<HashMap<String, Vec<Value>>>,
    /// QX4I: 코인별 Deribit ATM IV 시간 칸 기억(원본 qx4_state["iv"][coin])
    qiv: Mutex<HashMap<String, std::collections::BTreeMap<i64, Option<f64>>>>,
    /// 바이낸스 선물 REST 한 문(binance_get)의 공유 상태: 마지막으로 보낸 벽시계 · 가중치 문턱으로 안 보내는 끝 시각
    gate: Mutex<(f64, f64)>,
}

fn e(kind: &str, msg: impl AsRef<str>) -> PyErr {
    PyErr(format!("{kind}: {}", msg.as_ref()))
}

/// 원본 except → {"ok": False, "reason": f"{type}: {exc}"[:n]}
fn wrap(r: R<V>, cut: Option<usize>) -> Resp {
    Resp::json(&r.unwrap_or_else(|PyErr(m)| json!({"ok": false, "reason": cut.map_or(m.clone(), |n| m.chars().take(n).collect())})))
}

fn mtimes(ps: &[PathBuf]) -> Option<Vec<SystemTime>> {
    ps.iter().map(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok()).collect()
}

/// «낡음» 판정에 쓰는 칸만 남긴 작은 사본(model_stale 이 보는 source·sources·한 단계 안 source) -- 큰 모델 JSON 을 요청마다 다시 읽지 않게
fn sources(v: &Value) -> V {
    let mut m = serde_json::Map::new();
    for k in ["source", "sources"] {
        m.insert(k.into(), v[k].clone());
    }
    if let Some(o) = v.as_object() {
        for (k, x) in o {
            if x.get("source").is_some() {
                m.insert(format!("_{k}"), json!({"source": x["source"]}));
            }
        }
    }
    Value::Object(m)
}

/// mtime 이 바뀌었으면 다시 읽고(원본 `if da.get("mtime") != mt: … load`) · 읽은 뒤 «낡음»이면 오류
fn cached<T>(slot: &Mutex<HashMap<String, (Vec<SystemTime>, Arc<T>)>>, key: &str, files: &[PathBuf], src: &Path, load: impl FnOnce() -> Result<T, String>, srcs: impl Fn(&T) -> &V) -> R<Option<Arc<T>>> {
    let Some(mt) = mtimes(files) else { return Ok(None) };
    let mut g = slot.lock().unwrap();
    let hit = match g.get(key) {
        Some((m, a)) if *m == mt => a.clone(),
        _ => {
            let a = Arc::new(load().map_err(|m| e("ValueError", m))?);
            g.insert(key.into(), (mt, a.clone()));
            a
        }
    };
    drop(g);
    if let Some(why) = rtmodel::model_stale(&files[0], srcs(&hit), src) {
        return Err(e("ModelStale", why));
    }
    Ok(Some(hit))
}

fn models(cfg: &Config) -> PathBuf {
    cfg.models.clone().unwrap_or_else(|| cfg.live_dir().join("rtmodel"))
}

fn ban_file(cfg: &Config) -> PathBuf {
    cfg.live_dir().join("binance_ban_until")
}

fn agent(secs: u64) -> ureq::Agent {
    static A: OnceLock<Mutex<HashMap<u64, ureq::Agent>>> = OnceLock::new();
    let m = A.get_or_init(Default::default);
    m.lock().unwrap().entry(secs).or_insert_with(|| {
        ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(secs))).http_status_as_error(false).proxy(None).build().into()
    }).clone()
}

/// 받기 한 번 → (상태, Retry-After, 본문)
fn call(url: &str, secs: u64, ua: Option<&str>) -> Result<(u16, Option<String>, String), String> {
    let mut rq = agent(secs).get(url);
    if let Some(u) = ua {
        rq = rq.header("User-Agent", u);
    }
    let mut r = rq.call().map_err(|x| x.to_string())?;
    let retry = r.headers().get("retry-after").and_then(|v| v.to_str().ok()).map(str::to_string);
    let status = r.status().as_u16();
    let body = r.body_mut().read_to_string().map_err(|x| x.to_string())?;
    Ok((status, retry, body))
}

fn query(params: &[(&'static str, String)]) -> String {
    params.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("&")
}

/// 원본 d4._get(requests · 서버 안에선 binance_ban_guard 가 감쌈 · 20초): 밴 중이면 BinanceBanned · raise_for_status · json
fn rest_get(cfg: &Config, path: &str, params: &[(&'static str, String)]) -> Result<Value, String> {
    rest_on(cfg)?;
    let (status, body) = match binance_get(cfg, &format!("{}{path}?{}", cfg.rest, query(params))) {
        Ok(x) => x,
        Err(Refused::Ban(left)) => return Err(format!("BinanceBanned: binance IP ban: {left:.0}s left -- request not sent")),
        Err(Refused::Hold(left)) => return Err(format!("BinanceWeightHold: used weight over limit: {left:.0}s left -- request not sent")),
        Err(Refused::Conn(m)) => return Err(format!("ConnectionError: {m}")),
    };
    if status >= 400 {
        return Err(format!("HTTPError: {status}"));
    }
    serde_json::from_str(&body).map_err(|m| format!("JSONDecodeError: {m}"))
}

/// 원본 fetch_binance_json(10초): 밴 중 = 안 보냄(error_reason 있으면 HTTPServiceUnavailable binance_ip_ban · 없으면 None) · 200 아님 = 418·429 기록 뒤 HTTPBadGateway(reason) · None
/// 가중치 문턱인 분 = 밴과 같은 «안 보냄» 갈래(이유 글자만 binance_weight_hold)
pub(crate) fn fetch_binance(cfg: &Config, url: &str, reason: Option<&str>) -> Result<Option<Value>, String> {
    rest_on(cfg)?;
    let (status, body) = match binance_get(cfg, url) {
        Ok(x) => x,
        Err(Refused::Ban(_)) => return reason.map_or(Ok(None), |_| Err("HTTPServiceUnavailable: binance_ip_ban".into())),
        Err(Refused::Hold(_)) => return reason.map_or(Ok(None), |_| Err("HTTPServiceUnavailable: binance_weight_hold".into())),
        Err(Refused::Conn(m)) => return Err(format!("ClientError: {m}")),
    };
    if status != 200 {
        return reason.map_or(Ok(None), |r| Err(format!("HTTPBadGateway: {r}")));
    }
    serde_json::from_str(&body).map(Some).map_err(|m| format!("ContentTypeError: {m}"))
}

pub enum Refused {
    /// 밴 파일이 아직 밴 중(남은 초) -- 안 보냄
    Ban(f64),
    /// 앞 응답의 x-mbx-used-weight-1m 이 문턱을 넘은 분(남은 초) -- 안 보냄
    Hold(f64),
    /// 연결·본문 읽기 실패
    Conn(String),
}

fn env_num(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(d)
}

fn wall() -> f64 {
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

/// 🔴바이낸스 선물 REST 는 전부 여기 하나를 지난다(이 서버 IP 로 실주문이 나간다 · 한도 2,400/분 공유):
///   ① 밴 파일(data/live/binance_ban_until · cfg 시계)이 밴 중이면 안 보냄 · 418/429 는 그 파일에 적음 ② 앞 응답의 `x-mbx-used-weight-1m` 이
///   문턱(RT_REST_WEIGHT_LIMIT · 기본 1,200)을 넘었으면 그 분 끝(+0.5초)까지 안 보냄 ③ 요청 사이 최소 간격(RT_REST_GAP_MS · 기본 200ms).
///   보내기·418/429 기록·가중치 머리 읽기는 rtmodel::gate::Gate 그대로. ②는 Gate 처럼 «쉬며 기다리지» 않고 거절한다(화면 요청이 1분 붙잡히지 않게 --
///   원본의 못 받음 갈래로). 원본 fetch_binance_json·d4._get 에는 ②③이 없다(값은 그대로 · 느리게만).
/// ponytail: 한 문 = 잠금 하나로 바이낸스 요청을 한 줄로 세운다(원본은 연결 32개 동시) · Gate 는 시계 상자가 Send 가 아니라 스레드마다 하나
///   (공유 상태 = 위 잠금). rtmodel Gate 에 Send·«쉬지 말고 거절» 이 생기면 하나로.
pub fn binance_get(cfg: &Config, url: &str) -> Result<(u16, String), Refused> {
    use rtmodel::gate::{Fail, Gate};
    thread_local!(static G: std::cell::RefCell<Option<(PathBuf, Gate)>> = const { std::cell::RefCell::new(None) });
    let ban = ban_file(cfg);
    let left = rtmodel::ban::ban_left(&ban, (cfg.now)());
    if left > 0.0 {
        return Err(Refused::Ban(left));
    }
    let mut st = cfg.models_state.gate.lock().unwrap();
    let t = wall();
    if st.1 > t {
        return Err(Refused::Hold(st.1 - t));
    }
    let gap = env_num("RT_REST_GAP_MS", 200.0) / 1000.0;
    if t - st.0 < gap {
        std::thread::sleep(Duration::from_secs_f64(gap - (t - st.0)));
    }
    let r = G.with(|g| {
        let mut g = g.borrow_mut();
        if g.as_ref().is_none_or(|(p, _)| *p != ban) {
            *g = Some((ban.clone(), Gate::new(Duration::from_secs(20), ban.clone())));
        }
        let gate = &mut g.as_mut().unwrap().1;
        gate.weight = None;
        let r = gate.send(url, true);
        (r, gate.weight)
    });
    st.0 = wall();
    let limit = env_num("RT_REST_WEIGHT_LIMIT", 1200.0); // Gate 와 같은 이름·기본값
    if let Some(w) = r.1.filter(|w| *w > limit) {
        st.1 = (st.0 / 60.0).floor() * 60.0 + 60.5;
        (cfg.log)(&format!("[rest] 가중치 {w:.0} > 문턱 {limit:.0} -- 바이낸스 요청 {:.1}초 안 보냄", st.1 - st.0));
    }
    match r.0 {
        Ok(x) => Ok(x),
        Err(Fail::Banned(s)) => Err(Refused::Ban(s)),
        Err(Fail::Conn(m) | Fail::Body(m)) => Err(Refused::Conn(m)),
    }
}

/// REST 꺼짐(RTSERVE_REST 없음 = 병행 가동 기본 -- 파이썬과 같은 IP 로 같은 REST 를 겹쳐 부르지 않게): 보내지 않고 오류
fn rest_on(cfg: &Config) -> Result<(), String> {
    if cfg.rest.is_empty() { Err("RestOff: RTSERVE_REST unset -- request not sent".into()) } else { Ok(()) }
}

/// rtmodel 이 낸 본문 글자 → 값(swr 에 넣는다 · 직렬화는 rtserve 파이썬 서식 = 같은 글자)
fn parse(s: &str) -> R<V> {
    serde_json::from_str(s).map_err(|m| e("ValueError", m.to_string()))
}

fn now_ms_i(cfg: &Config) -> i64 {
    ((cfg.now)() * 1000.0) as i64
}

// ───────────────────────── /api/dir4h ─────────────────────────

const ARM: [&str; 3] = ["ETH", "SOL", "XRP"]; // d4.ARM

fn dir4h_produce(cfg: &Config, coin: &str) -> R<V> {
    let mj = models(cfg).join(format!("dir4h_{}.json", coin.to_lowercase()));
    let Some(art) = cached(&cfg.models_state.a4, coin, &[mj.clone()], &cfg.live_dir(), || {
        let v: Value = rtmodel::read_json(&mj)?;
        Ok((dir4h::Art::load(&mj)?, sources(&v)))
    }, |a| &a.1)? else { return Ok(json!({"ok": false, "reason": "artifact_missing"})) };
    let (now, wall) = (now_ms_i(cfg), (cfg.now)() * 1000.0);
    let mut get = |p: &str, q: &[(&'static str, String)]| rest_get(cfg, p, q);
    let mut r = cfg.models_state.r4.lock().unwrap();
    rtmodel::handlers::dir4h_body(&mut r, &mut get, now, wall, coin, &art.0).map_err(PyErr)
}

fn api_dir4h(cfg: &Config, req: &Req) -> Resp {
    let coin = req.q("asset").filter(|a| !a.is_empty()).unwrap_or("eth").to_uppercase();
    if !ARM.contains(&coin.as_str()) {
        return Resp::json(&json!({"ok": false, "reason": "unsupported_asset"}));
    }
    let c = cfg.clone();
    let k = coin.clone();
    wrap(cfg.swr.get(&format!("dir4h:{}", coin.to_lowercase()), 300.0, 1800.0, Arc::new(move || dir4h_produce(&c, &k))), Some(200))
}

// ───────────────────────── /api/dir-short ─────────────────────────

const DS: [&str; 4] = ["ETH", "BTC", "SOL", "XRP"]; // "ETH" + ds.TABLE

fn dir_short_produce(cfg: &Config, coin: &str) -> R<V> {
    let (d, lc) = (models(cfg), coin.to_lowercase());
    let (mj, oj) = (d.join(format!("ds_{lc}.json")), d.join(format!("oos_{lc}.json")));
    let Some(art) = cached(&cfg.models_state.ads, coin, &[mj.clone(), oj.clone()], &cfg.live_dir(), || {
        let (m, o): (Value, Value) = (rtmodel::read_json(&mj)?, rtmodel::read_json(&oj)?);
        let oos = dirshort::NAMES.iter().map(|(_, nm)| (*nm, o[*nm].as_array().map_or(vec![], |a| a.iter().map(|b| f64::from_bits(b.as_u64().unwrap_or(0))).collect()))).collect();
        Ok((dirshort::Art::from_json(&m)?, oos, sources(&m)))
    }, |a| &a.2)? else { return Ok(json!({"ok": false, "reason": "artifact_missing"})) };
    let (now, wall) = (now_ms_i(cfg), (cfg.now)() * 1000.0);
    let mut get = |p: &str, q: &[(&'static str, String)]| rest_get(cfg, p, q);
    let mut r = cfg.models_state.rs.lock().unwrap();
    if coin == "ETH" {
        let book = dirshort::book_sqlite(&cfg.live_dir().join("bookdepth/ethusdt.sqlite")).map_err(PyErr)?; // 원본 BOOK_DB · 회차마다 읽기 전용으로 연다
        rtmodel::handlers::dir_short_body(&mut r, &mut get, now, wall, coin, &art.0, &art.1, Some(&book))
    } else {
        rtmodel::handlers::dir_short_body(&mut r, &mut get, now, wall, coin, &art.0, &art.1, None)
    }.map_err(PyErr)
}

fn api_dir_short(cfg: &Config, req: &Req) -> Resp {
    let coin = req.q("asset").filter(|a| !a.is_empty()).unwrap_or("eth").to_uppercase();
    if !DS.contains(&coin.as_str()) {
        return Resp::json(&json!({"ok": false, "reason": "unsupported_asset"}));
    }
    let c = cfg.clone();
    let k = coin.clone();
    wrap(cfg.swr.get(&format!("dir_short:{}", coin.to_lowercase()), 300.0, 1800.0, Arc::new(move || dir_short_produce(&c, &k))), Some(200))
}

// ───────────────────────── /api/vol-forecast ─────────────────────────

const VOL: [&str; 4] = ["eth", "xrp", "btc", "sol"]; // VOL4H_ARTIFACTS

/// 원본 qx4_payload: 파일 없음·읽기 실패·이어 받기 실패·ok 아님 = None(HGB 로) · 🔴QX4I(Deribit 옵션 IV 칸)는 안 옮김(원본도 iv_now 가 실패하면 칸만 빠진다)
fn qx4_try(cfg: &Config, asset: &str) -> R<Option<V>> {
    let p = models(cfg).join("qx4.json");
    let mut slot = cfg.models_state.qx4.lock().unwrap();
    let Some(mt) = mtimes(std::slice::from_ref(&p)) else { return Ok(None) };
    if slot.as_ref().is_none_or(|(m, _)| *m != mt) {
        let Ok(v) = rtmodel::read_json::<Value>(&p) else { return Ok(None) };
        *slot = Some((mt, Arc::new((qx4::Art::from_json(&v), sources(&v)))));
    }
    let art = slot.as_ref().unwrap().1.clone();
    drop(slot);
    if let Some(why) = rtmodel::model_stale(&p, &art.1, &cfg.live_dir()) {
        return Err(e("ModelStale", why));
    }
    let base = format!("{}/fapi/v1/klines", cfg.rest);
    let mut fetch = |q: &[(&'static str, String)]| -> Vec<Value> {
        fetch_binance(cfg, &format!("{base}?{}", query(q)), None).ok().flatten().and_then(|v| v.as_array().cloned()).unwrap_or_default()
    };
    let mut rows = cfg.models_state.qrows.lock().unwrap();
    let mut ivs = cfg.models_state.qiv.lock().unwrap();
    let st = ivs.entry(asset.to_uppercase()).or_default();
    let mut get = |u: &str| ext_get(cfg, u).ok(); // 원본 iv_now 의 Deribit 받기 -- 실패 = 그 칸만 빠짐
    Ok(rtmodel::handlers::vol_forecast_qx4i(&mut fetch, &mut rows, &art.0, (cfg.now)() * 1000.0, asset, Some((st, &mut get))))
}

fn vol_produce(cfg: &Config, asset: &str) -> R<V> {
    match qx4_try(cfg, asset) {
        Ok(Some(v)) => return Ok(v),
        Ok(None) => {}
        Err(PyErr(m)) if m.starts_with("ModelStale") => return Err(PyErr(m)),
        Err(PyErr(m)) => (cfg.log)(&format!("vol4h qx4 error → HGB 대체: {m}")),
    }
    let p = models(cfg).join(format!("svm_{asset}.json"));
    let Some(art) = cached(&cfg.models_state.svm, asset, &[p.clone()], &cfg.live_dir(), || {
        let v: Value = rtmodel::read_json(&p)?;
        Ok((Svm::from_json(&v)?, sources(&v)))
    }, |a| &a.1)? else { return Ok(json!({"ok": false, "reason": "artifact_missing"})) };
    let sym = crate::hub::MARKET_SYMBOL[crate::MARKET_ASSETS.iter().position(|a| *a == asset).unwrap()];
    let mut fetch = |u: &str| fetch_binance(cfg, u, Some("vol_upstream_error")).map(|v| v.unwrap_or(Value::Null));
    parse(&rtmodel::handlers::vol_forecast_hgb(&mut fetch, &cfg.rest, sym, asset, &art.0, (cfg.now)() * 1000.0).map_err(PyErr)?)
}

fn api_vol_forecast(cfg: &Config, req: &Req) -> Resp {
    let asset = req.q("asset").filter(|a| !a.is_empty()).unwrap_or("eth").to_lowercase();
    if !VOL.contains(&asset.as_str()) {
        return Resp::json(&json!({"ok": false, "reason": "unsupported_asset"}));
    }
    let c = cfg.clone();
    let k = asset.clone();
    wrap(cfg.swr.get(&format!("vol4h:{asset}"), 300.0, 1800.0, Arc::new(move || vol_produce(&c, &k))), None)
}

// ───────────────────────── /api/open-rev · /api/dir-signals ─────────────────────────

fn api_open_rev(cfg: &Config, _: &Req) -> Resp {
    let c = cfg.clone();
    wrap(cfg.swr.get("open_rev", 120.0, 900.0, Arc::new(move || {
        let mut fetch = |u: &str| fetch_binance(&c, u, Some("open_rev_upstream_error")).map(|v| v.unwrap_or(Value::Null));
        parse(&rtmodel::handlers::open_rev_body(&mut fetch, &c.rest, (c.now)() * 1000.0).map_err(PyErr)?)
    })), Some(200))
}

/// 원본 dir_signals._get: urllib · User-Agent Mozilla/5.0 · 30초(바이낸스 현물도 requests 가 아니라 밴 가드 밖)
pub(crate) fn ext_get(cfg: &Config, url: &str) -> Result<Value, String> {
    rest_on(cfg)?;
    let u = match &cfg.ext {
        Some(b) => ["https://api.exchange.coinbase.com", "https://api.binance.com", "https://api.stlouisfed.org", "https://www.deribit.com"].iter()
            .find_map(|h| url.strip_prefix(h).map(|r| format!("{b}{r}"))).unwrap_or_else(|| url.into()),
        None => url.into(),
    };
    let (status, _, body) = call(&u, 30, Some("Mozilla/5.0")).map_err(|m| format!("URLError: {m}"))?;
    if status != 200 {
        return Err(format!("HTTPError: HTTP Error {status}"));
    }
    serde_json::from_str(&body).map_err(|m| format!("JSONDecodeError: {m}"))
}

fn api_dir_signals(cfg: &Config, _: &Req) -> Resp {
    let c = cfg.clone();
    wrap(cfg.swr.get("dir_signals", 3600.0, 86400.0, Arc::new(move || {
        let mut fetch = |u: &str| ext_get(&c, u);
        parse(&rtmodel::handlers::dir_signals_body(&mut fetch, c.fred_key.as_deref(), (c.now)()))
    })), None)
}
