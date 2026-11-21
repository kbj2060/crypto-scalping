//! 4h·15분/1시간 방향의 REST 입력 조립과 적중 기록 (2026-11-21 · R5).
//! 원본 scripts/live_eth_dir4h_20261015.py `_klines`·`_metric`·`_merge`·`_short_klines`·`rest_frames`·`daily_closes`(:286-368) 와
//! dashboard/pred_record.py `record_short`·`record_h4`·`_settled`·`_px`·`_append`(:59-99) -- 서버 `pred_record_loop`(server.py:2841)가 5분마다 부르는 길.
//! HTTP 는 모른다: 부르는 쪽이 `get(경로, [(이름, 값)])` 을 넘긴다(쿼리 순서 = 원본 requests params 딕셔너리 순서).
//! 🔴원본은 4h(`scripts.live_eth_dir4h_20261015`)와 15분·1시간(`live_eth_dir4h_20261015`, ds 가 따로 import)이 모듈이 달라 캐시도 따로다 --
//!   여기도 `Rest` 하나씩 따로 둔다(같은 REST 를 5분에 두 벌 부르는 것까지 원본 그대로 · 원본 주석 ponytail).
use crate::dir4h::{Frame, COINS};
use serde_json::Value;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

const NAN: f64 = f64::NAN;
const BAR: i64 = 300_000;
const DAY: i64 = 86_400_000;
/// REST 끝점 → (Frame.m 칸, 값 키, REST 스탬프 → vision 행 밀기(분)). 원본 METRIC_EP 순서.
pub const METRIC_EP: [(&str, usize, &str, i64); 5] = [
    ("openInterestHist", 0, "sumOpenInterest", 5),
    ("topLongShortPositionRatio", 2, "longShortRatio", 5),
    ("topLongShortAccountRatio", 1, "longShortRatio", 5),
    ("globalLongShortAccountRatio", 3, "longShortRatio", 5),
    ("takerlongshortRatio", 4, "buySellRatio", 0),
];

pub type Get<'a> = dyn FnMut(&str, &[(&'static str, String)]) -> Result<Value, String> + 'a;

fn num(v: &Value) -> Result<f64, String> {
    v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())).ok_or_else(|| format!("float({v})"))
}
fn int(v: &Value) -> Result<i64, String> {
    v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse().ok())).ok_or_else(|| format!("int({v})"))
}
fn rows(v: &Value) -> Result<&Vec<Value>, String> {
    v.as_array().ok_or_else(|| format!("목록이 아님: {v}"))
}

fn empty_frame() -> Frame {
    Frame { m: std::array::from_fn(|_| vec![]), ..Default::default() }
}

/// _klines: 5분봉 n 개를 endTime 으로 거슬러 받기(쪽당 ≤ 1000) · 형성 중 봉(close_ms ≥ now) 버림.
fn klines(get: &mut Get, sym: &str, n: usize, now_ms: i64) -> Result<Frame, String> {
    let mut all: Vec<Value> = vec![];
    let mut end: Option<i64> = None;
    while all.len() < n {
        let limit = 1000.min(n - all.len() + 1);
        let mut q = vec![("symbol", format!("{sym}USDT")), ("interval", "5m".into()), ("limit", limit.to_string())];
        if let Some(e) = end {
            q.push(("endTime", e.to_string()));
        }
        let got = get("/fapi/v1/klines", &q)?;
        let got = rows(&got)?.clone();
        if got.is_empty() {
            break;
        }
        end = Some(int(&got[0][0])? - 1);
        let short = got.len() < limit;
        all = got.into_iter().chain(all).collect();
        if short {
            break;
        }
    }
    let mut f = empty_frame();
    for r in &all {
        if int(&r[6])? < now_ms {
            f.ts.push(int(&r[0])?);
            f.high.push(num(&r[2])?);
            f.low.push(num(&r[3])?);
            f.close.push(num(&r[4])?);
            f.volume.push(num(&r[5])?);
            f.trades.push(num(&r[8])?);
            f.tbb.push(num(&r[9])?);
        }
    }
    let n = f.ts.len();
    f.m = std::array::from_fn(|_| vec![NAN; n]);
    Ok(f)
}

/// _metric: 5분 metrics 를 500개씩 거슬러 받기(시작 = 벽시계 − days · 원본도 time.time()). 반환 (vision 시각 ms, 값).
fn metric(get: &mut Get, sym: &str, ep: &str, key: &str, shift_min: i64, days: f64, wall_ms: f64) -> Result<Vec<(i64, f64)>, String> {
    let start = (wall_ms - days * 86_400_000.0) as i64; // int(time.time() * 1000 - days * 86_400_000)
    let mut all: Vec<Value> = vec![];
    let mut end: Option<i64> = None;
    loop {
        let mut q = vec![("symbol", format!("{sym}USDT")), ("period", "5m".into()), ("limit", "500".into())];
        if let Some(e) = end {
            q.push(("endTime", e.to_string()));
        }
        let got = get(&format!("/futures/data/{ep}"), &q)?;
        let got = rows(&got)?.clone();
        if got.is_empty() {
            break;
        }
        let t0 = int(&got[0]["timestamp"])?;
        end = Some(t0 - 1);
        let stop = got.len() < 500 || t0 <= start;
        all = got.into_iter().chain(all).collect();
        if stop {
            break;
        }
    }
    all.iter().map(|r| Ok((int(&r["timestamp"])? - shift_min * 60_000, num(&r[key])?))).collect()
}

/// _merge: 옛것 + 새것 → 같은 시각은 뒤 것 · 정렬 · 마지막 시각 − keep 일 이후만. 옛것이 없으면 새것 그대로(거르기만).
fn merge_k(old: Option<&Frame>, new: Frame, keep_days: i64) -> Frame {
    let f = match old {
        None => new,
        Some(o) => {
            let mut idx: HashMap<i64, (bool, usize)> = HashMap::new();
            for i in 0..o.ts.len() {
                idx.insert(o.ts[i], (false, i));
            }
            for i in 0..new.ts.len() {
                idx.insert(new.ts[i], (true, i));
            }
            let mut ts: Vec<i64> = idx.keys().copied().collect();
            ts.sort();
            let mut f = empty_frame();
            for t in ts {
                let (is_new, i) = idx[&t];
                let s = if is_new { &new } else { o };
                f.ts.push(t);
                f.high.push(s.high[i]);
                f.low.push(s.low[i]);
                f.close.push(s.close[i]);
                f.volume.push(s.volume[i]);
                f.trades.push(s.trades[i]);
                f.tbb.push(s.tbb[i]);
            }
            let n = f.ts.len();
            f.m = std::array::from_fn(|_| vec![NAN; n]);
            f
        }
    };
    let Some(&mx) = f.ts.iter().max() else { return f };
    let keep: Vec<usize> = (0..f.ts.len()).filter(|&i| f.ts[i] >= mx - keep_days * DAY).collect();
    let pick = |c: &Vec<f64>| keep.iter().map(|&i| c[i]).collect::<Vec<f64>>();
    Frame { ts: keep.iter().map(|&i| f.ts[i]).collect(), high: pick(&f.high), low: pick(&f.low), close: pick(&f.close), volume: pick(&f.volume), trades: pick(&f.trades),
            tbb: pick(&f.tbb), m: std::array::from_fn(|j| pick(&f.m[j])) }
}

fn merge_m(old: Option<&Vec<(i64, f64)>>, new: Vec<(i64, f64)>, keep_days: i64) -> Vec<(i64, f64)> {
    let mut v = match old {
        None => new,
        Some(o) => {
            let mut m: HashMap<i64, f64> = o.iter().copied().collect();
            m.extend(new);
            let mut v: Vec<(i64, f64)> = m.into_iter().collect();
            v.sort_by_key(|x| x.0);
            v
        }
    };
    if let Some(mx) = v.iter().map(|x| x.0).max() {
        v.retain(|x| x.0 >= mx - keep_days * DAY);
    }
    v
}

/// 한 모듈의 REST 캐시(원본 `_CACHE`·`_SHORT`).
#[derive(Default)]
pub struct Rest {
    pub k: HashMap<String, Frame>,
    m: HashMap<(String, &'static str), Vec<(i64, f64)>>,
    short: HashMap<(String, usize, i64), Frame>,
}

impl Rest {
    /// rest_frames(now_ms, others_n=340, coin): 첫 호출(또는 20시간 넘게 안 불림) = 대상 31일(5분봉 9,000 · metrics 30.5일), 다음부터 꼬리(300 · 0.5일).
    /// 나머지 코인 = 짧은 5분봉(5분봉 하나에 한 번만 받아 나눠 씀) · "BTC" 자리 코인만 OI 1.3일.
    pub fn frames(&mut self, get: &mut Get, now_ms: i64, wall_ms: f64, coin: &str) -> Result<HashMap<String, Frame>, String> {
        let kk = format!("{coin}_k");
        let first = self.k.get(&kk).is_none_or(|f| f.ts.iter().max().is_none_or(|&mx| now_ms - mx > 20 * 3_600_000));
        let new = klines(get, coin, if first { 9000 } else { 300 }, now_ms)?;
        let merged = merge_k(self.k.get(&kk), new, 32);
        self.k.insert(kk.clone(), merged);
        let mut tgt = self.k[&kk].clone();
        for (ep, col, key, shift) in METRIC_EP {
            let got = metric(get, coin, ep, key, shift, if first { 30.5 } else { 0.5 }, wall_ms)?;
            let ck = (coin.to_string(), ep);
            let mm = merge_m(self.m.get(&ck), got, 32);
            let at: HashMap<i64, f64> = mm.iter().copied().collect();
            tgt.m[col] = tgt.ts.iter().map(|t| at.get(t).copied().unwrap_or(NAN)).collect();
            self.m.insert(ck, mm);
        }
        let mut out = HashMap::new();
        out.insert(coin.to_string(), tgt);
        let oi_slot = if coin == "BTC" { "ETH" } else { "BTC" }; // OI 1h = frames_for 뒤 "BTC" 자리
        for sym in std::iter::once("ETH").chain(COINS).filter(|s| *s != coin) {
            let key = (sym.to_string(), 340usize, now_ms.div_euclid(BAR));
            if !self.short.contains_key(&key) {
                if self.short.len() > 200 {
                    self.short.clear();
                }
                let k = klines(get, sym, 340, now_ms)?;
                self.short.insert(key.clone(), k);
            }
            let mut k = self.short[&key].clone();
            if sym == oi_slot {
                let oi = metric(get, sym, "openInterestHist", "sumOpenInterest", 5, 340.0 / 288.0 + 0.1, wall_ms)?;
                let at: HashMap<i64, f64> = oi.into_iter().collect();
                k.m[0] = k.ts.iter().map(|t| at.get(t).copied().unwrap_or(NAN)).collect();
            }
            out.insert(sym.to_string(), k);
        }
        Ok(out)
    }
}

/// daily_closes: 1일봉 120개 중 닫힌 것 → (일 시작 ms, 종가).
pub fn daily_closes(get: &mut Get, now_ms: i64, coin: &str) -> Result<Vec<(i64, f64)>, String> {
    let got = get("/fapi/v1/klines", &[("symbol", format!("{coin}USDT")), ("interval", "1d".into()), ("limit", "120".into())])?;
    let mut out = vec![];
    for r in rows(&got)? {
        if int(&r[6])? < now_ms {
            out.push((int(&r[0])?, num(&r[4])?));
        }
    }
    Ok(out)
}

/// pred_record 한 파일의 «이미 쓴 결정 시각»(프로세스 안 · 처음엔 파일 꼬리 8MB 에서) + 덧붙이기.
#[derive(Default)]
pub struct Record {
    seen: HashMap<(std::path::PathBuf, &'static str), std::collections::HashSet<i64>>,
}

const TAIL_BYTES: u64 = 8_000_000;

impl Record {
    fn append(&mut self, path: &Path, kind: &'static str, rows: Vec<(i64, String)>) -> std::io::Result<usize> {
        let key = (path.to_path_buf(), kind);
        if !self.seen.contains_key(&key) {
            let mut s = std::collections::HashSet::new();
            if let Ok(b) = std::fs::read(path) {
                let start = b.len().saturating_sub(TAIL_BYTES as usize);
                let text = String::from_utf8_lossy(&b[start..]);
                for (j, ln) in text.lines().enumerate() {
                    if j == 0 && start > 0 {
                        continue; // 잘린 첫 줄
                    }
                    if let Ok(v) = serde_json::from_str::<Value>(ln) {
                        if v.get("k").and_then(Value::as_str) == Some(kind) {
                            if let Some(t) = v.get("t").and_then(Value::as_i64) {
                                s.insert(t);
                            }
                        }
                    }
                }
            }
            self.seen.insert(key.clone(), s);
        }
        let seen = self.seen.get_mut(&key).unwrap();
        let new: Vec<&(i64, String)> = rows.iter().filter(|r| !seen.contains(&r.0)).collect();
        if !new.is_empty() {
            if let Some(d) = path.parent() {
                std::fs::create_dir_all(d)?;
            }
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
            f.write_all(new.iter().map(|r| format!("{}\n", r.1)).collect::<String>().as_bytes())?;
            for r in &new {
                seen.insert(r.0);
            }
        }
        Ok(new.len())
    }

    /// _settled: 최신 봉에 빈 피쳐가 있으면 그 봉은 뺀다(다음 실행의 hist 에 다시 들어온다).
    fn settled(payload: &Value) -> Vec<Value> {
        let hist = payload["hist"].as_array().cloned().unwrap_or_default();
        if payload["nan_feats"].as_array().is_some_and(|a| !a.is_empty()) {
            hist.into_iter().filter(|h| h[0] != payload["decided_at_ms"]).collect()
        } else {
            hist
        }
    }

    fn px(closes: &HashMap<i64, f64>, t: i64) -> Option<f64> {
        closes.get(&(t - BAR)).copied().filter(|v| !v.is_nan())
    }

    /// record_short: hist [[결정 ms, 15분 분위, 1시간 분위], …] → {"k":"s","t","px","a","b"}.
    pub fn short(&mut self, dir: &Path, coin: &str, payload: &Value, closes: &HashMap<i64, f64>) -> std::io::Result<usize> {
        if payload["ok"] != Value::Bool(true) {
            return Ok(0);
        }
        let rows = Self::settled(payload).iter().filter_map(|h| {
            let t = h[0].as_i64()?;
            let px = Self::px(closes, t)?;
            Some((t, format!("{{\"k\":\"s\",\"t\":{t},\"px\":{},\"a\":{},\"b\":{}}}", crate::sr_lean::jf(px), jnum(&h[1]), jnum(&h[2]))))
        }).collect();
        self.append(&dir.join(format!("{}.jsonl", coin.to_uppercase())), "s", rows)
    }

    /// record_h4: hist [[결정 ms, 예측 bp], …] + 신호(문턱 lo·hi 밖).
    pub fn h4(&mut self, dir: &Path, coin: &str, payload: &Value, closes: &HashMap<i64, f64>) -> std::io::Result<usize> {
        if payload["ok"] != Value::Bool(true) {
            return Ok(0);
        }
        let (lo, hi) = (payload["lo"].as_f64().unwrap(), payload["hi"].as_f64().unwrap());
        let rows = Self::settled(payload).iter().filter_map(|h| {
            let t = h[0].as_i64()?;
            let px = Self::px(closes, t)?;
            let v = h[1].as_f64()?;
            let s = if v >= hi { 1 } else if v <= lo { -1 } else { 0 };
            Some((t, format!("{{\"k\":\"4\",\"t\":{t},\"px\":{},\"d\":{},\"s\":{s}}}", crate::sr_lean::jf(px), crate::sr_lean::jf(v))))
        }).collect();
        self.append(&dir.join(format!("{}.jsonl", coin.to_uppercase())), "4", rows)
    }
}

/// 상대 강도 9코인(dashboard/pred_record.py X_COINS · 기록 px 칸 순서).
pub const X_COINS: [&str; 9] = ["ETH", "BTC", "SOL", "XRP", "ADA", "AVAX", "BNB", "DOGE", "LINK"];

/// 서버 tick(server.py:2857): 9코인 `_klines(c, 60, now)` 의 (봉 시작 ms, 종가). 한 코인이라도 실패하면 Err -- 원본도 tick 통째로 건너뛴다(뒤 코인은 안 부름).
pub fn xsec_px(get: &mut Get, now_ms: i64) -> Result<Vec<Vec<(i64, f64)>>, String> {
    X_COINS.iter().map(|c| klines(get, c, 60, now_ms).map(|f| f.ts.into_iter().zip(f.close).collect())).collect()
}

impl Record {
    /// xsec_record: 9코인 모두 있는 봉만(dropna) · 49봉 이상 · 마지막과 49번째 앞이 정확히 240분 → 최신 스냅숏 글자
    /// (원본은 프로세스 안 _xs_latest · 여기는 부르는 쪽이 파일로) + 4h 격자(UTC 시 %4==3 · 55분 봉) 행을 xsec.jsonl 에 덧붙임. 반환 (새 줄 수, 스냅숏).
    pub fn xsec(&mut self, dir: &Path, px: &[Vec<(i64, f64)>]) -> std::io::Result<(usize, Option<String>)> {
        if px.len() < X_COINS.len() || px.iter().any(|s| s.is_empty()) {
            return Ok((0, None));
        }
        let maps: Vec<HashMap<i64, f64>> = px.iter().map(|s| s.iter().copied().collect()).collect();
        let mut ts: Vec<i64> = px[0].iter().map(|x| x.0).filter(|t| maps.iter().all(|m| m.get(t).is_some_and(|v| !v.is_nan()))).collect();
        ts.sort_unstable();
        ts.dedup();
        let n = ts.len();
        if n < 49 || ts[n - 1] - ts[n - 49] != 240 * 60_000 {
            return Ok((0, None)); // 결측 봉이 있으면 «49번째 앞»이 4h 앞이 아니다
        }
        let rel: Vec<f64> = maps.iter().map(|m| m[&ts[n - 1]] / m[&ts[n - 49]] - 1.0).collect();
        let mean = crate::box_hold::npsum(&rel) / rel.len() as f64; // Series.mean = numpy 짝 합 / n -- 서버에 bottleneck·numexpr 없음(2026-11-21 확인)
        let bp: Vec<String> = X_COINS.iter().zip(&rel).map(|(c, v)| format!("\"{c}\": {}", crate::sr_lean::jf(crate::box_hold::py_round((v - mean) * 1e4, 1)))).collect();
        let latest = format!("{{\"t\": {}, \"rel_bp\": {{{}}}}}", ts[n - 1] + BAR, bp.join(", "));
        let rows = ts.iter().filter(|&&t| t.rem_euclid(DAY) / 3_600_000 % 4 == 3 && t.rem_euclid(3_600_000) / 60_000 == 55).map(|&t| {
            let p: Vec<String> = X_COINS.iter().zip(&maps).map(|(c, m)| format!("\"{c}\":{}", crate::sr_lean::jf(m[&t]))).collect();
            (t + BAR, format!("{{\"k\":\"x\",\"t\":{},\"px\":{{{}}}}}", t + BAR, p.join(",")))
        }).collect();
        Ok((self.append(&dir.join("xsec.jsonl"), "x", rows)?, Some(latest)))
    }
}

/// JSON 숫자 → 파이썬 json.dumps 글자(정수면 정수, 실수면 repr).
fn jnum(v: &Value) -> String {
    match v.as_i64() {
        Some(i) if v.is_i64() => i.to_string(),
        _ => v.as_f64().map_or("null".into(), crate::sr_lean::jf),
    }
}

/// pred_record_safe 의 종가 표: 그 코인 5분봉 캐시(봉 시작 ms → 종가).
pub fn closes(r: &Rest, coin: &str) -> Option<HashMap<i64, f64>> {
    r.k.get(&format!("{coin}_k")).map(|f| f.ts.iter().copied().zip(f.close.iter().copied()).collect())
}
