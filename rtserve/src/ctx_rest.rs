//! 상황 카드의 REST 5분봉 재료(2026-11-22) -- veto(12h 추세) · candles_day(24h 분위·권장 배수·card30 입력) · 상황 입력 캔들(tail 200) · card30 예측.
//! 원본(Read 로 본 실제 줄): server.py load_chart_klines_frames 3185~3273(ETH·BTC 증거 프레임 · klines limit EVIDENCE_FETCH_LIMIT = 1500
//!   = scripts/live_evidence_signal_dashboard_20260823.py 144 · swr 60초 max_stale 0) · load_market_history_from_evidence_cache 1665~1716(tail 200 · veto) ·
//!   load_market_history 1718~1736(SOL·XRP = klines limit 1000 · kline_candles[-200:] · swr 60/600) · trend_veto_rows 1613~1646(TREND_VETO_N 144 · K 1.0 1597·1611) ·
//!   kline_candles 1648~1661 · load_5m_day 2418~2438(limit 300 · 시작+300 ≤ 지금 · swr 60/600 · 방금 닫힌 봉이 없으면 2초 뒤 식힘) ·
//!   _card30_bars 2282~2286 · _card30_pred 2259~2280 · CARD30_*_PATH 978~979 · 상수 190~194 · STALE_GRACE_SECONDS 1040.
//! 바깥 요청은 여기서 안 보낸다 -- `Fetch`(부르는 쪽이 rtmodel::gate 로 · `RTSERVE_REST` 있을 때만)를 받는다. 여기 함수는 «받은 klines → 값»뿐.
use crate::ctx_x::Candle;
use serde_json::Value as V;
use std::collections::BTreeMap;

pub const TREND_VETO_N: usize = 144;
pub const TREND_VETO_K: f64 = 1.0;
/// trend_veto_rows 기본 tail(TREND_VETO_N × 5 = 720 · 1613 · 1648).
pub const VETO_TAIL: usize = TREND_VETO_N * 5;
pub const EVIDENCE_FETCH_LIMIT: usize = 1500;
pub const CANDLES_TAIL: usize = 200;
pub const DAY5_LIMIT: usize = 300;
/// 캐시 수명(초): 증거 프레임·SOL/XRP 캔들·5분봉 하루 = EVIDENCE_SIGNAL_CACHE_SECONDS(192) · 묵은 값 허용 = 0(3272) · STALE_GRACE_SECONDS 600(1040).
pub const KLINES_TTL: f64 = 60.0;
pub const EVIDENCE_MAX_STALE: f64 = 0.0;
pub const STALE_GRACE: f64 = 600.0;

/// 부르는 쪽이 주는 GET(본문 JSON). 오류 글자는 원본 예외 모양(HTTPBadGateway: <reason> 등)으로.
pub type Fetch<'a> = &'a mut dyn FnMut(&str) -> Result<V, String>;

/// klines 한 행(원본 열: open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_base, …).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Kline {
    pub open_ms: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub close_ms: i64,
    pub taker: f64,
}

fn f(v: &V) -> Result<f64, String> {
    match v {
        V::String(s) => s.trim().parse().map_err(|_| format!("ValueError: could not convert string to float: {s:?}")),
        V::Number(n) => n.as_f64().ok_or("ValueError".into()),
        _ => Err("TypeError".into()),
    }
}
fn i(v: &V) -> Result<i64, String> {
    match v {
        V::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|x| x as i64)).ok_or("ValueError".into()),
        V::String(s) => s.trim().parse().map_err(|_| "ValueError".into()),
        _ => Err("TypeError".into()),
    }
}

/// 바이낸스 klines 본문 → 행. 모양이 틀리면 Err(원본은 그 회차 예외).
pub fn parse_klines(raw: &V) -> Result<Vec<Kline>, String> {
    let a = raw.as_array().ok_or("TypeError: klines")?;
    a.iter()
        .map(|r| {
            let r = r.as_array().filter(|r| r.len() >= 10).ok_or("ValueError: kline row")?;
            Ok(Kline { open_ms: i(&r[0])?, open: f(&r[1])?, high: f(&r[2])?, low: f(&r[3])?, close: f(&r[4])?, volume: f(&r[5])?, close_ms: i(&r[6])?, taker: f(&r[9])? })
        })
        .collect()
}

pub fn klines_url(base: &str, symbol: &str, limit: usize) -> String {
    format!("{base}/fapi/v1/klines?symbol={symbol}&interval=5m&limit={limit}")
}

/// pandas Series.rolling(w).mean()(min_periods = w): 더하기·빼기 따로 Kahan 보정 · 같은 값 연속이면 그 값 · 음수/양수만이면 부호 고정
/// (pandas/_libs/window/aggregations.pyx roll_mean · add_mean · remove_mean · calc_mean).
pub fn rolling_mean(v: &[f64], w: usize) -> Vec<f64> {
    let (mut nobs, mut neg, mut sum, mut c_add, mut c_rem, mut same, mut prev) = (0usize, 0usize, 0.0f64, 0.0f64, 0.0f64, 0usize, f64::NAN);
    let mut out = Vec::with_capacity(v.len());
    let add = |x: f64, nobs: &mut usize, sum: &mut f64, neg: &mut usize, c: &mut f64, same: &mut usize, prev: &mut f64| {
        if !x.is_nan() {
            *nobs += 1;
            let y = x - *c;
            let t = *sum + y;
            *c = t - *sum - y;
            *sum = t;
            if x.is_sign_negative() {
                *neg += 1;
            }
            if x == *prev { *same += 1 } else { *same = 1 }
            *prev = x;
        }
    };
    for k in 0..v.len() {
        let s = (k + 1).saturating_sub(w);
        if k == 0 {
            prev = v[0];
            same = 0;
            add(v[0], &mut nobs, &mut sum, &mut neg, &mut c_add, &mut same, &mut prev);
        } else {
            let s_prev = k.saturating_sub(w);
            for &x in &v[s_prev..s] {
                if !x.is_nan() {
                    nobs -= 1;
                    let y = -x - c_rem;
                    let t = sum + y;
                    c_rem = t - sum - y;
                    sum = t;
                    if x.is_sign_negative() {
                        neg -= 1;
                    }
                }
            }
            add(v[k], &mut nobs, &mut sum, &mut neg, &mut c_add, &mut same, &mut prev);
        }
        let r = if nobs >= w && nobs > 0 {
            let r = sum / nobs as f64;
            if same >= nobs { prev } else if neg == 0 && r < 0.0 { 0.0 } else if neg == nobs && r > 0.0 { 0.0 } else { r }
        } else {
            f64::NAN
        };
        out.push(r);
    }
    out
}

/// pandas Series.ewm(alpha=a, adjust=False).mean(): alpha 는 com 을 거쳐 다시 센다(1/(1 + (1−a)/a)) · 값이 같으면 그대로.
pub fn ewm_mean(v: &[f64], a: f64) -> Vec<f64> {
    let com = (1.0 - a) / a;
    let alpha = 1.0 / (1.0 + com);
    let (old_f, new_wt) = (1.0 - alpha, alpha);
    let mut out = Vec::with_capacity(v.len());
    if v.is_empty() {
        return out;
    }
    let mut weighted = v[0];
    let mut old_wt = 1.0;
    out.push(weighted);
    for &cur in &v[1..] {
        let obs = !cur.is_nan();
        if !weighted.is_nan() {
            old_wt *= old_f;
            if obs {
                if weighted != cur {
                    weighted = old_wt * weighted + new_wt * cur;
                    weighted /= old_wt + new_wt;
                }
                old_wt = 1.0;
            }
        } else if obs {
            weighted = cur;
        }
        out.push(weighted);
    }
    out
}

/// trend_veto_rows(1613~1646) 의 veto 만: 봉 시각(초) → +1 롱만 / −1 숏만 / 0 워밍업. `closed` = 시간순 마감봉.
// ponytail: sma·atr·drop(화면 /api/market-history 용)은 안 낸다 -- 상황 카드는 veto 만 읽는다(_situation_inputs 2197).
pub fn trend_veto(closed: &[Kline], tail: usize) -> BTreeMap<i64, i64> {
    let w = &closed[closed.len().saturating_sub(tail)..];
    let mut out = BTreeMap::new();
    if w.len() < TREND_VETO_N + 1 {
        return out;
    }
    let c: Vec<f64> = w.iter().map(|k| k.close).collect();
    let sma = rolling_mean(&c, TREND_VETO_N);
    let tr: Vec<f64> = (0..w.len())
        .map(|k| {
            let hl = w[k].high - w[k].low;
            if k == 0 { hl } else { hl.max((w[k].high - c[k - 1]).abs()).max((w[k].low - c[k - 1]).abs()) }
        })
        .collect();
    let atr = ewm_mean(&tr, 1.0 / TREND_VETO_N as f64);
    let mut cur = 0;
    for k in 0..w.len() {
        let dev = (c[k] - sma[k]) / sma[k];
        let eps = TREND_VETO_K * atr[k] / c[k];
        if !(dev.is_finite() && eps.is_finite()) {
            continue;
        }
        if dev > eps {
            cur = 1;
        } else if dev < -eps {
            cur = -1;
        }
        out.insert(w[k].open_ms.div_euclid(1000), cur);
    }
    out
}

/// 상황 입력 캔들 한 행(원본 candles 행의 _situation_inputs 가 읽는 칸 -- time·high·low·close·veto·partial).
#[derive(Clone, Debug, PartialEq)]
pub struct SitCandle {
    pub time: i64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub veto: Option<i64>,
    pub partial: bool,
}

/// 증거 프레임(3206~3214 · 3231~3243): 시각 정렬 · 같은 시각은 마지막 · 마지막 행이 아직 안 닫혔으면(close_time ≥ 지금) 버림.
pub fn evidence_frame(raw: &V, now_ms: i64) -> Result<Vec<Kline>, String> {
    let mut by: BTreeMap<i64, Kline> = BTreeMap::new();
    for k in parse_klines(raw)? {
        by.insert(k.open_ms, k); // drop_duplicates(keep="last") -- 🔴원본 sort_values 는 불안정 정렬이라 값이 다른 같은 시각 중복이면 어느 쪽이 남는지 정해지지 않는다(바이낸스는 안 보낸다)
    }
    let mut v: Vec<Kline> = by.into_values().collect();
    if v.last().is_some_and(|k| k.close_ms >= now_ms) {
        v.pop();
    }
    Ok(v)
}

/// load_market_history_from_evidence_cache(1691~1714): 프레임 tail(200) + trend_veto_rows(프레임 전체 → 그 안에서 tail 720).
pub fn candles_from_frame(frame: &[Kline]) -> Vec<SitCandle> {
    let veto = trend_veto(frame, VETO_TAIL);
    frame[frame.len().saturating_sub(CANDLES_TAIL)..]
        .iter()
        .map(|k| { let t = k.open_ms.div_euclid(1000); SitCandle { time: t, high: k.high, low: k.low, close: k.close, veto: veto.get(&t).copied(), partial: false } })
        .collect()
}

/// kline_candles(1648~1661)[-200:](SOL·XRP · load_market_history 1730): veto 는 마감봉(close_time < 지금)만으로 · 형성 봉은 partial.
pub fn kline_candles(raw: &V, now_ms: i64) -> Result<Vec<SitCandle>, String> {
    let rows = parse_klines(raw)?;
    let closed: Vec<Kline> = rows.iter().copied().filter(|k| k.close_ms < now_ms).collect();
    let veto = trend_veto(&closed, VETO_TAIL);
    let all: Vec<SitCandle> = rows
        .iter()
        .map(|k| { let t = k.open_ms.div_euclid(1000); SitCandle { time: t, high: k.high, low: k.low, close: k.close, veto: veto.get(&t).copied(), partial: k.close_ms >= now_ms } })
        .collect();
    Ok(all[all.len().saturating_sub(CANDLES_TAIL)..].to_vec())
}

/// load_5m_day produce(2434~2437): 시작 + 300 ≤ 지금(초)인 행만 · taker 칸 = r[9].
pub fn day5(raw: &V, now_s: f64) -> Result<Vec<Candle>, String> {
    Ok(parse_klines(raw)?
        .into_iter()
        .filter(|k| (k.open_ms.div_euclid(1000) + 300) as f64 <= now_s)
        .map(|k| Candle { time: k.open_ms.div_euclid(1000), high: k.high, low: k.low, close: k.close, volume: k.volume, taker: Some(k.taker) })
        .collect())
}

/// load_5m_day 의 식힘(2425~2426): 가진 값의 마지막 봉이 «방금 닫힌 봉»보다 앞이고 마지막 갱신이 2초 넘었으면 캐시를 식힌다(true).
pub fn day5_stale(have: &[Candle], bar_start: i64, age_s: f64) -> bool {
    have.last().is_some_and(|c| c.time < bar_start - 300) && age_s > 2.0
}

/// _card30_bars(2282~2286): candles_day(없으면 candles) 중 지금 봉 앞 · taker 있는 것 → rtmodel 봉.
pub fn card30_bars(day5: &[Candle], now: f64) -> Vec<rtmodel::card30::Bar> {
    let bar_start = (now as i64).div_euclid(300) * 300;
    day5.iter()
        .filter(|c| c.time < bar_start)
        .filter_map(|c| c.taker.map(|t| rtmodel::card30::Bar { time: c.time, high: c.high, low: c.low, close: c.close, volume: c.volume, taker: t }))
        .collect()
}

/// _card30_pred(2259~2280): 봉마다 한 번(마지막 봉 시각이 같으면 그 값) · 모델이 없거나 봉이 없으면 (None, None) · 피쳐가 없으면 빈 값을 그 봉에 기억.
#[derive(Default)]
pub struct Card30Cache {
    pub bar: Option<i64>,
    pub p: Option<(f64, f64)>,
}

impl Card30Cache {
    pub fn get(&mut self, m: Option<&rtmodel::card30::Card30>, bars: &[rtmodel::card30::Bar]) -> (Option<f64>, Option<f64>) {
        let (Some(m), Some(last)) = (m, bars.last()) else { return (None, None) };
        if self.bar != Some(last.time) {
            self.p = rtmodel::card30::predict(m, bars);
            self.bar = Some(last.time);
        }
        self.p.map_or((None, None), |(d, r)| (Some(d), Some(r)))
    }
}

/// 원본 fetch_binance_json 을 rtmodel::gate 로(밴 중이면 안 보냄 · 사용 가중치 문턱이면 다음 분까지 쉼): 200 아니면 `HTTPBadGateway: <reason>`.
pub fn gate_fetch(gate: &mut rtmodel::gate::Gate, url: &str, reason: &str) -> Result<V, String> {
    match gate.send(url, true) {
        Ok((200, body)) => serde_json::from_str(&body).map_err(|e| format!("ContentTypeError: {e}")),
        Ok(_) => Err(format!("HTTPBadGateway: {reason}")),
        Err(rtmodel::gate::Fail::Banned(_)) => Err("HTTPServiceUnavailable: binance_ip_ban".into()),
        Err(rtmodel::gate::Fail::Conn(m) | rtmodel::gate::Fail::Body(m)) => Err(format!("ClientError: {m}")),
    }
}

/// card30 모델 파일(결정 2026-11-22: `<저장소>/data/live/rtmodel/card30.json` -- 내보내기는 rtmodel/tools/takeover.sh export-if-stale 몫).
/// mtime 이 바뀌면 다시 읽는다(model.rs `cached` 와 같은 규칙). 없거나 못 읽으면 None = 원본 «모델 없음» 갈래(_card30_load 2255~2257 → `{}` → 카드는 표로).
/// 원본과 다른 곳: 원본은 읽기 실패를 한 번 알리고 다시 안 읽는다(card30["err"]) -- 여기는 파일이 바뀌면 다시 시도한다.
#[derive(Default)]
pub struct Card30File {
    mt: Option<std::time::SystemTime>,
    m: Option<std::sync::Arc<rtmodel::card30::Card30>>,
}

pub fn card30_path(repo: &std::path::Path) -> std::path::PathBuf {
    repo.join("data/live/rtmodel/card30.json")
}

impl Card30File {
    pub fn get(&mut self, path: &std::path::Path) -> Option<std::sync::Arc<rtmodel::card30::Card30>> {
        let mt = std::fs::metadata(path).and_then(|m| m.modified()).ok();
        if mt.is_none() {
            (self.mt, self.m) = (None, None);
            return None;
        }
        if mt != self.mt {
            self.mt = mt;
            self.m = std::fs::read_to_string(path)
                .map_err(|e| e.to_string())
                .and_then(|t| serde_json::from_str::<V>(&t).map_err(|e| e.to_string()))
                .and_then(|v| rtmodel::card30::Card30::from_json(&v))
                .map(std::sync::Arc::new)
                .map_err(|e| println!("card30 model: {e}"))
                .ok();
        }
        self.m.clone()
    }
}

// ── 청산맵 레벨(load_liquidation_map 3461~3508 의 레벨 몫 · 미시 payload 의 sr 재료) ──
/// LIQUIDATION_MAP_FETCH_LIMIT(211) · LIQUIDATION_MAP_5M_FETCH_LIMIT(271) · LIQUIDATION_MAP_CACHE_SECONDS(221) · LIQUIDATION_MAP_LOOKBACK_HOURS(196).
pub const LIQ_MAP_FETCH_LIMIT: usize = 48;
pub const LIQ_MAP_5M_FETCH_LIMIT: usize = 460;
pub const LIQ_MAP_TTL: f64 = 60.0;
pub const LIQ_MAP_LOOKBACK_H: usize = 24;

pub fn klines_url_iv(base: &str, symbol: &str, interval: &str, limit: usize) -> String {
    format!("{base}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}")
}

fn win(k: &[Kline]) -> rtmodel::regime_band::Win {
    rtmodel::regime_band::Win {
        t: k.iter().map(|x| x.open_ms.div_euclid(1000)).collect(),
        h: k.iter().map(|x| x.high).collect(),
        l: k.iter().map(|x| x.low).collect(),
        c: k.iter().map(|x| x.close).collect(),
        v: k.iter().map(|x| x.volume).collect(),
    }
}

/// compute_spliced_levels(scripts/live_liquidation_map_20260824.py 303~362)를 load_liquidation_map 입력으로(3473~3478):
/// liq_klines_df(1시간 · 48개 · 형성 봉 버림 3314~3337) 의 tail(24) · 기준가 = 마지막 닫힌 1시간봉 종가 · 5분봉(460개 · 형성 봉 버림) = 자기 봉 안 쓸림 판정.
/// 계산은 rtmodel::regime_band 의 prepare·bins·side_levels(같은 원본을 옮긴 것 -- 한 곳만). 반환 = 원본 반환 dict 에서 heatmap_bins 뺀 것.
// ponytail: heatmap_bins·heatmap_history·tier_profile 은 안 낸다(미시 payload 는 레벨만 읽는다 -- /api/liquidation-map 을 옮길 때 더한다).
pub fn liq_levels(raw_1h: &V, raw_5m: &V, now_ms: i64) -> Result<V, String> {
    use serde_json::json;
    let h = evidence_frame(raw_1h, now_ms)?;
    let m = evidence_frame(raw_5m, now_ms)?;
    let cp = h.last().map_or(0.0, |k| k.close);
    let w = &h[h.len().saturating_sub(LIQ_MAP_LOOKBACK_H)..];
    let empty = |reason: &str, n: usize| json!({"warmed_up": false, "error": reason, "current_price": null, "bars_used": n, "lookback_hours": 0.0, "support_levels": [], "resistance_levels": []});
    let (wv, mv) = (win(w), win(&m));
    let Some(cm) = rtmodel::regime_band::prepare(&wv, cp, &mv) else {
        let reason = if w.len() < 20 || !(cp > 0.0) { "insufficient_data" } else { "no_surviving_levels" };
        return Ok(empty(reason, w.len()));
    };
    let mid: Vec<f64> = (0..w.len()).map(|i| (wv.h[i] + wv.l[i]) / 2.0).collect();
    let (Some(bm), Some(bc)) = (rtmodel::regime_band::bins(&cm, &mid), rtmodel::regime_band::bins(&cm, &cm.close.clone())) else {
        return Ok(empty("no_surviving_levels", w.len()));
    };
    let lv = |l: Vec<(f64, f64, f64)>| serde_json::Value::Array(l.into_iter().map(|(p, wp, d)| json!({"price": p, "weight_pct": wp, "distance_pct": d})).collect());
    Ok(json!({"warmed_up": true, "error": null, "current_price": cp, "bars_used": w.len(),
              "lookback_hours": (wv.t[wv.t.len() - 1] - wv.t[0]) as f64 / 3600.0,
              "bin_width": crate::py::round_nd(cm.bw, 4),
              "support_levels": lv(rtmodel::regime_band::side_levels(&bm, cm.bw, cp, true)),
              "resistance_levels": lv(rtmodel::regime_band::side_levels(&bc, cm.bw, cp, false))}))
}
