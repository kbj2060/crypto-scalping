//! 미시 참고 payload(server.py collect_micro_ref 1912~2022 의 한 틱) -- 관계 읽기 x 의 imb40·agree·trigger·sr·act·vol_pct · 시장 맥락의 sweep·depth25 재료. 2026-11-22 옮김.
//! 원본(Read 로 본 실제 줄): collect_micro_ref 1912~2022 · micro_ref.py qi 39~41 · side_of 44~45 · raster_flow 48~82 · qi_side_hyst 85~94 · agree_state 97~103 ·
//! pct_rank 106~111 · act_label 114~117 · sr_context 120~142 · baseline_from_tape 152~175 · market_ctx.py sweep_depth 38~61 ·
//! 래스터 읽기는 crate::raster_read::read_window(원본 수집기 327~407 -- 그 파일이 옮긴 것을 부르기만).
//! 원본과 다르게 둔 것(결정 대기 · CTX.md «bookTicker»): 원본은 틱마다 REST bookTicker 를 부른다 -- 여기는 호가를 인자로 받는다(코어 @bookTicker 최신값을 넣으면 REST 0).
//! 청산맵 레벨(load_liquidation_map)도 인자(그 계산은 REST klines 90일 -- 다른 조각).
use serde_json::{json, Map, Value as V};
use std::collections::{BTreeMap, VecDeque};
use std::path::Path;

pub const BAND_SHALLOW_BP: f64 = 10.0;
pub const BAND_DEEP_BP: f64 = 40.0;
pub const QI_ENTER: f64 = 0.75;
pub const QI_EXIT: f64 = 0.45;
pub const TRIGGER_LIVE_S: i64 = 2;
pub const NEAR_BP: f64 = 15.0;
pub const SWEEP_BPS: [f64; 3] = [25.0, 50.0, 100.0];
/// 기준선 주기 = server.py 972 `MICRO_BASELINE_SECONDS = 3600`(쓰는 줄 1926·1939) · 기준선이 없을 때 다시 시도 60초(1926).
pub const BASELINE_S: f64 = 3600.0;
/// 래스터 창 = `crate::raster_read::read_window(root, sym, now_ms, 12, 1)`(원본 1962 · 같은 원본 327~407 을 한 곳에서만 옮긴다).
pub use crate::raster_read::Window;

/// np.sum(float32 1차원) -- numpy 짝 합을 f32 로(rtcollect::ob5m::np_sum 과 같은 갈래).
fn np_sum_f32(a: &[f32]) -> f32 {
    fn pw(a: &[f32]) -> f32 {
        let n = a.len();
        if n < 8 {
            return a.iter().fold(0.0f32, |s, v| s + v);
        }
        if n <= 128 {
            let mut r = [0f32; 8];
            r.copy_from_slice(&a[..8]);
            let m = n - n % 8;
            for i in (8..m).step_by(8) {
                for j in 0..8 {
                    r[j] += a[i + j];
                }
            }
            let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            for v in &a[m..] {
                res += v;
            }
            return res;
        }
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pw(&a[..n2]) + pw(&a[n2..])
    }
    if a.len() <= 8192 {
        return pw(a);
    }
    a.chunks(8192).fold(0.0f32, |s, c| s + pw(c))
}

/// micro_ref.qi(39~41).
pub fn qi(b: f64, a: f64) -> f64 {
    let s = b + a;
    if s > 0.0 { (b - a) / s } else { 0.0 }
}

/// micro_ref.raster_flow(48~82) 의 결과.
#[derive(Clone, Debug, PartialEq)]
pub struct Flow {
    pub mid: f64,
    pub imb10: f64,
    pub imb40: f64,
    pub bid40: f64,
    pub ask40: f64,
    pub ofi10: f64,
    pub ofi_secs: i64,
    /// 마지막 유효 초(collect_micro_ref 1970 이 같은 초로 sweep 을 잰다).
    pub last: usize,
}

fn px_of(w: &Window) -> Vec<f64> {
    (0..w.n_bins).map(|k| (w.bin_lo + k as i64) as f64 * w.bin_size).collect()
}

/// micro_ref.raster_flow(48~82). None = {"ok": False}.
pub fn raster_flow(w: &Window) -> Option<Flow> {
    let n = w.n_bins;
    let last = w.mid.iter().rposition(|m| m.is_finite())?;
    if n == 0 {
        return None;
    }
    let px = px_of(w);
    let m = w.mid[last] as f64;
    let band = |row: &[f32], bp: f64| -> (f64, f64) {
        let ww = m * bp / 1e4;
        let r: Vec<f32> = row.iter().zip(&px).filter(|(_, p)| **p >= m - ww && **p <= m + ww).map(|(q, _)| *q).collect();
        let pos: Vec<f32> = r.iter().copied().filter(|v| *v > 0.0).collect();
        let neg: Vec<f32> = r.iter().copied().filter(|v| *v < 0.0).collect();
        (np_sum_f32(&pos) as f64, (-np_sum_f32(&neg)) as f64)
    };
    let (b10, a10) = band(w.row(last), BAND_SHALLOW_BP);
    let (b40, a40) = band(w.row(last), BAND_DEEP_BP);
    let mut sums = vec![];
    let stop = last as i64 - 11;
    let mut i = last as i64;
    while i > stop.max(-1) {
        if !w.mid[i as usize].is_finite() {
            break;
        }
        sums.push(band(w.row(i as usize), BAND_DEEP_BP));
        i -= 1;
    }
    let mut ofi = 0.0;
    for k in 0..sums.len().saturating_sub(1) {
        let ((b1, a1), (b0, a0)) = (sums[k], sums[k + 1]);
        ofi += (b1 - b0) - (a1 - a0);
    }
    Some(Flow { mid: m, imb10: qi(b10, a10), imb40: qi(b40, a40), bid40: b40, ask40: a40, ofi10: ofi, ofi_secs: (sums.len() as i64 - 1).max(0), last })
}

/// micro_ref.side_of(44~45).
pub fn side_of(x: f64, thr: f64) -> &'static str {
    if x >= thr { "매수" } else if x <= -thr { "매도" } else { "중립" }
}

/// micro_ref.qi_side_hyst(85~94).
pub fn qi_side_hyst(q: f64, prev: &str) -> String {
    if !q.is_finite() {
        return prev.to_string();
    }
    match prev {
        "매수" => if q <= -QI_ENTER { "매도" } else if q >= QI_EXIT { "매수" } else { "중립" },
        "매도" => if q >= QI_ENTER { "매수" } else if q <= -QI_EXIT { "매도" } else { "중립" },
        _ => if q >= QI_ENTER { "매수" } else if q <= -QI_ENTER { "매도" } else { "중립" },
    }
    .to_string()
}

/// micro_ref.agree_state(97~103) → (동조 상태, 이번 QI 쪽).
pub fn agree_state(qi_val: f64, ofi10: f64, thr: f64, prev: &str) -> (String, String) {
    let q = qi_side_hyst(qi_val, prev);
    let o = side_of(ofi10, thr);
    if q == "중립" || o == "중립" {
        return ("중립".into(), q);
    }
    let s = if q == o && o == "매수" { "동조매수" } else if q == o && o == "매도" { "동조매도" } else { "갈림" };
    (s.into(), q)
}

/// micro_ref.pct_rank(106~111): 20개 미만이거나 x 가 유한하지 않으면 None · searchsorted(right) / 크기.
pub fn pct_rank(x: f64, sorted: &[f64]) -> Option<f64> {
    if sorted.len() < 20 || !x.is_finite() {
        return None;
    }
    Some(sorted.partition_point(|v| *v <= x) as f64 / sorted.len() as f64)
}

/// micro_ref.act_label(114~117).
pub fn act_label(p: Option<f64>) -> &'static str {
    match p {
        None => "기준없음",
        Some(p) if p >= 0.8 => "활발",
        Some(p) if p <= 0.2 => "조용",
        _ => "보통",
    }
}

fn price(lv: &V) -> f64 {
    match &lv["price"] {
        V::String(s) => s.trim().parse().unwrap_or(f64::NAN),
        v => v.as_f64().unwrap_or(f64::NAN),
    }
}

/// micro_ref.sr_context(120~142).
pub fn sr_context(levels: Option<&V>, mid: f64, imb40: Option<f64>) -> Map<String, V> {
    let mut out: Map<String, V> = ["sup", "res", "sup_bp", "res_bp"].iter().map(|k| (k.to_string(), V::Null)).collect();
    out.insert("near".into(), json!("없음"));
    out.insert("deep_wall".into(), V::Null);
    let Some(levels) = levels.filter(|l| l.as_object().is_some_and(|o| !o.is_empty())) else { return out };
    if !(mid > 0.0) {
        return out;
    }
    let list = |k: &str| levels.get(k).and_then(V::as_array).cloned().unwrap_or_default();
    let (mut sbp, mut rbp) = (None, None);
    if let Some(s) = list("support_levels").iter().find(|lv| price(lv) < mid) {
        let p = price(s);
        out["sup"] = json!(p);
        sbp = Some((mid - p) / mid * 1e4);
        out["sup_bp"] = json!(sbp);
    }
    if let Some(r) = list("resistance_levels").iter().find(|lv| price(lv) > mid) {
        let p = price(r);
        out["res"] = json!(p);
        rbp = Some((p - mid) / mid * 1e4);
        out["res_bp"] = json!(rbp);
    }
    // 파이썬 min((d, 이름, 부호)) -- d 가 같으면 이름 글자 차례
    let mut c: Vec<(f64, &str, f64)> = vec![];
    if let Some(b) = sbp {
        c.push((b.abs(), "지지근접", 1.0));
    }
    if let Some(b) = rbp {
        c.push((b.abs(), "저항근접", -1.0));
    }
    if let Some(&(d, name, sign)) = c.iter().min_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(b.1))) {
        if d <= NEAR_BP {
            out["near"] = json!(name);
            if let Some(i) = imb40 {
                out["deep_wall"] = json!(if i * sign > 0.0 { "레벨쪽" } else { "반대쪽" });
            }
        }
    }
    out
}

/// market_ctx.sweep_depth(38~61). `top` = (최우선 매수가, 수량, 최우선 매도가, 수량).
pub fn sweep_depth(row: &[f32], px: &[f64], mid: f64, top: Option<(f64, f64, f64, f64)>) -> V {
    let bs = if px.len() > 1 { px[1] - px[0] } else { 0.5 };
    let s_lo = (mid / bs).floor() * bs;
    let (mut bid, mut ask) = (vec![], vec![]);
    for bp in SWEEP_BPS {
        let w = mid * bp / 1e4;
        let pick = |up: bool| -> Vec<f32> {
            row.iter()
                .zip(px)
                .filter(|(q, p)| {
                    let ctr = **p + bs / 2.0;
                    if up { **p >= s_lo + bs && ctr <= mid + w && **q < 0.0 } else { **p + bs <= s_lo && ctr >= mid - w && **q > 0.0 }
                })
                .map(|(q, _)| *q)
                .collect()
        };
        let mut a = (-np_sum_f32(&pick(true))) as f64;
        let mut b = np_sum_f32(&pick(false)) as f64;
        if let Some((bpx, bq, apx, aq)) = top {
            b += if bpx >= s_lo { bq } else { 0.0 };
            a += if apx < s_lo + bs { aq } else { 0.0 };
        }
        ask.push(a);
        bid.push(b);
    }
    json!({"bid": bid, "ask": ask})
}

/// micro_ref.baseline_from_tape(152~175)의 minute_vol_sorted(시 → 그 시 1분 거래량 정렬). 분이 600개 미만이면 None · 읽기 실패는 Err.
/// 🔴알려진 차이: 원본은 초 합계를 SQLite 글자(group_concat)로 받아 pandas C 파서로 다시 읽는다 -- 값이 그 글자를 거친다.
///   글자 자릿수·끝자리가 SQLite 판마다 다르고(정답지 파이썬 sqlite3 3.52 = 짧은 왕복 또는 근사 17자리 · rusqlite 3.50.2 = 15자리),
///   pandas 읽기(precise_xstrtod)도 정확한 반올림이 아니다 → 원본 값이 원래 합에서 초마다 1ulp 안팎 비켜 있다(정답지 실측: 분 합 상대차 ≤ 1e-15).
///   여기는 원래 합을 그대로 쓴다 -- vol60_pct 가 갈리는 건 60초 거래량이 분 합과 1ulp 안으로 붙을 때뿐이다.
pub fn baseline_from_tape(db: &Path, symbol: &str, now_s: i64, days: i64) -> Result<Option<BTreeMap<i64, Vec<f64>>>, String> {
    let since = now_s - days * 86400;
    let err = |e: rusqlite::Error| format!("baseline_from_tape({}): {e}", db.display());
    let c = rusqlite::Connection::open_with_flags(db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX).map_err(err)?;
    // 원본 SQL 의 안쪽 질의 그대로(GROUP BY 1 = 초 차례) -- 값은 글자를 거치지 않고 그대로 받는다(머리 주석 «알려진 차이»)
    let mut q = c
        .prepare("SELECT ts_sec, sum(buy_qty + sell_qty) AS v FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ? GROUP BY 1")
        .map_err(err)?;
    let rows: Vec<(i64, Option<f64>)> = q.query_map(rusqlite::params![symbol, since], |r| Ok((r.get(0)?, r.get(1)?))).map_err(err)?.collect::<Result<_, _>>().map_err(err)?;
    if rows.is_empty() {
        return Ok(None);
    }
    let mut order = vec![];
    for (t, v) in rows {
        let v = v.ok_or("baseline_from_tape: v NULL")?;
        order.push((t - t.rem_euclid(60), v));
    }
    let mut mins: BTreeMap<i64, f64> = BTreeMap::new();
    for (m, v) in order {
        *mins.entry(m).or_insert(0.0) += v; // np.bincount(weights) = 줄 차례로 0 에서
    }
    if mins.len() < 600 {
        return Ok(None);
    }
    let mut out: BTreeMap<i64, Vec<f64>> = (0..24).map(|h| (h, vec![])).collect();
    for (m, v) in mins {
        out.get_mut(&(m.rem_euclid(86400) / 3600)).unwrap().push(v);
    }
    for v in out.values_mut() {
        v.sort_by(|a, b| a.total_cmp(b));
    }
    Ok(Some(out))
}

/// 코인 하나의 원본 micro_state(_coin_ctx 1765~1770) + 링(1771~1776).
pub struct MicroRef {
    pub baseline: Option<BTreeMap<i64, Vec<f64>>>,
    pub baseline_at: f64,
    pub qi_prev: String,
    pub trigger: (Option<String>, Option<i64>),
    pub ofi_hist: VecDeque<f64>,
    pub imb40_ring: VecDeque<f64>,
    pub depth25_ring: VecDeque<(f64, f64)>,
    pub mid_ring: BTreeMap<i64, f64>,
    pub tick: u64,
    pub payload: V,
}

impl Default for MicroRef {
    fn default() -> Self {
        MicroRef {
            baseline: None,
            baseline_at: 0.0,
            qi_prev: "중립".into(),
            trigger: (None, None),
            ofi_hist: VecDeque::new(),
            imb40_ring: VecDeque::new(),
            depth25_ring: VecDeque::new(),
            mid_ring: BTreeMap::new(),
            tick: 0,
            payload: json!({"available": false}),
        }
    }
}

fn push_cap<T>(d: &mut VecDeque<T>, v: T, cap: usize) {
    if d.len() == cap {
        d.pop_front();
    }
    d.push_back(v);
}

impl MicroRef {
    /// 1926: 기준선(과 flow_scales·flow_scales_all)을 다시 읽을 때인가.
    pub fn baseline_due(&self, now: f64) -> bool {
        now - self.baseline_at >= if self.baseline.is_some() { BASELINE_S } else { 60.0 }
    }

    /// 1927~1941: 읽은 결과를 넣는다(None = 실패·모자람 → 있던 기준선은 두고 60초 뒤 다시).
    pub fn baseline_set(&mut self, now: f64, got: Option<BTreeMap<i64, Vec<f64>>>) {
        self.baseline_at = now;
        match got {
            Some(b) => self.baseline = Some(b),
            None if self.baseline.is_some() => self.baseline_at = now - BASELINE_S + 60.0,
            None => {}
        }
    }

    /// 1956~2008 한 틱. `book` = (매수가, 수량, 매도가, 수량)(원본 REST bookTicker 자리) · `vol60` = 지난 60초 체결량(코어 Micro.vol60 = 1985~1987) ·
    /// `liq60` = {"long","short","n"}(코어 Micro.liq60 = 1996~1999) · `levels` = load_liquidation_map 결과.
    pub fn tick(&mut self, now: f64, book: (f64, f64, f64, f64), w: &Window, vol60: f64, levels: &V, liq60: &V) -> &V {
        let now_sec = now as i64;
        let (bid, bq, ask, aq) = book;
        let qi_val = qi(bq, aq);
        let mid = (bid + ask) / 2.0;
        self.mid_ring.insert(now_sec, mid);
        self.mid_ring.retain(|s, _| *s >= now_sec - 600);
        let flow = raster_flow(w);
        let mut sweep = V::Null;
        if let Some(f) = &flow {
            push_cap(&mut self.ofi_hist, f.ofi10.abs(), 600);
            push_cap(&mut self.imb40_ring, f.imb40, 6 * 3600);
            sweep = sweep_depth(w.row(f.last), &px_of(w), f.mid, Some(book));
            push_cap(&mut self.depth25_ring, (sweep["bid"][0].as_f64().unwrap(), sweep["ask"][0].as_f64().unwrap()), 6 * 3600);
        }
        let ofi_thr = (self.ofi_hist.len() >= 30).then(|| rtcollect::ob5m::np_median(&mut self.ofi_hist.iter().copied().collect::<Vec<_>>()));
        let agree = match (&flow, ofi_thr) {
            (Some(f), Some(t)) if t != 0.0 => {
                let (a, q) = agree_state(qi_val, f.ofi10, t, &self.qi_prev);
                self.qi_prev = q;
                a
            }
            _ => "기준없음".into(),
        };
        if agree == "동조매수" || agree == "동조매도" {
            self.trigger = (Some(agree.clone()), Some(now_sec));
        }
        let age = self.trigger.1.filter(|t| *t != 0).map(|t| now_sec - t);
        let trig = json!({"side": self.trigger.0, "ts": self.trigger.1, "age": age, "live": age.is_some_and(|a| a <= TRIGGER_LIVE_S)});
        let hour = now_sec.rem_euclid(86400) / 3600;
        let vol60_pct = self.baseline.as_ref().and_then(|b| pct_rank(vol60, &b[&hour]));
        let warm = levels.get("warmed_up").is_some_and(|v| v.as_bool().unwrap_or(false) || v.as_f64().is_some_and(|x| x != 0.0));
        let mut sr = sr_context(warm.then_some(levels), mid, flow.as_ref().map(|f| f.imb40));
        if warm {
            for (k, src) in [("sup_levels", "support_levels"), ("res_levels", "resistance_levels")] {
                let l: Vec<V> = levels.get(src).and_then(V::as_array).into_iter().flatten().map(|lv| json!({"price": lv["price"], "weight_pct": lv.get("weight_pct").cloned().unwrap_or(V::Null)})).collect();
                sr.insert(k.into(), V::Array(l));
            }
        }
        self.tick += 1;
        self.payload = json!({
            "available": true, "ts": now_sec, "mid": mid, "agree": agree, "trigger": trig,
            "imb40": flow.as_ref().map(|f| f.imb40), "vol60_pct": vol60_pct, "act": act_label(vol60_pct),
            "sr": V::Object(sr), "liq60": liq60, "spread": ask - bid, "sweep": sweep,
        });
        &self.payload
    }
}
