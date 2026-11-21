//! 풋프린트 «청산» 탭 = scripts/live_liq_tab_20261107.py `main()`(cron 매시) + live_liquidation_map_20260824.compute_tier_profile(5분봉 없이 -- liq_tab 이 그렇게 부른다).
//!   쓰는 것: `<live>/liq_tab_<코인>.json` · `<live>/liq_tab_k1h_<코인>.parquet`(1h 봉 캐시) · `<live>/liq_tab_all.json`(코인별 최대 고통 묶음).
//!   읽는 것: lake parquet(바이낸스·OKX 청산 -- DuckDB read_parquet) · hot SQLite(바이낸스 청산 · OI 셋) · 바이낸스 klines(1h · 1500개 쪽).
//!   값 맞춤(정답지 = 서버와 같은 numpy 2.2.5·pandas 2.2.3·OpenBLAS ZEN venv -- rtmodel/tools/server_env.sh):
//!     pandas groupby 합 = Kahan · Series.sum·np.sum = numpy 짝 합(np_sum) · np.round(x, d) = rint(x·10^d)/10^d · 파이썬 round = 정확 반올림 ·
//!     np.quantile·percentile = linear · median = 가운데 둘 (a+b)/2 · np.convolve(…, ones(11), "same") = 창 안 차례 합(OpenBLAS ddot 꼬리 -- 길이 < 11 이면 뒤집힌 차례) ·
//!     std(ddof=1) = 짝 합 평균·편차 제곱 짝 합 · log·exp = libm.
//!   🔴원본 sort_values 는 quicksort(안정 아님) -- 같은 ms 청산이 여럿이면 더하는 순서만 다를 수 있다(Rust = 안정 정렬). 정답지엔 같은 ms 가 없다.
use duckdb::Connection;
use rusqlite::OpenFlags;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const HR: i64 = 3_600_000;
const DAY: i64 = 86_400_000;
const KEEP_DAYS: i64 = 735;
const WINDOWS: [(i64, i64); 6] = [(7, 90), (30, 90), (60, 90), (90, 90), (180, 180), (365, 365)];
const ROW_PCT: f64 = 0.004;
const REAL_PCT: f64 = 0.0025;
const HEAT_CUT: f64 = 0.1;
const PAIN_PERIODS: [(&str, usize); 7] = [("12H", 12), ("24H", 24), ("48H", 48), ("3D", 72), ("1W", 168), ("2W", 336), ("1M", 720)];
const LEV: [i64; 6] = [10, 20, 25, 50, 75, 100];
const MMR: f64 = 0.005;
const TIER_GROUPS: [(&str, &[i64]); 4] = [("10", &[10]), ("20~25", &[20, 25]), ("50", &[50]), ("75~100", &[75, 100])];
/// scripts/coin_config.COIN_CONFIG 순서·binance_symbol(코인 목록 = lake 에 그 코인 청산 폴더가 있는 것).
pub const COINS: [(&str, &str); 9] = [("eth", "ETHUSDT"), ("btc", "BTCUSDT"), ("sol", "SOLUSDT"), ("xrp", "XRPUSDT"), ("hype", "HYPEUSDT"),
                                      ("sndk", "SNDKUSDT"), ("soxl", "SOXLUSDT"), ("skhynix", "SKHYNIXUSDT"), ("mu", "MUUSDT")];

/// np.sum / Series.sum: numpy 는 8192칸(버퍼) 묶음마다 짝 합을 내고 묶음 합을 차례로 더한다(2026-11-21 서버 판 venv 실측 -- 9천 칸 넘는 합에서 갈림).
/// 🔴rtcollect::ob5m::np_sum 은 묶음 없이 짝 합만 -- 8192칸 넘는 배열에선 numpy 와 다르다(그쪽 쓰임은 작은 배열뿐이라 그대로 둠).
fn np_sum(a: &[f64]) -> f64 {
    let mut s = 0.0;
    for c in a.chunks(8192) {
        s += rtcollect::ob5m::np_sum(c);
    }
    s
}

fn kahan(acc: &mut (f64, f64), v: f64) {
    let y = v - acc.1;
    let t = acc.0 + y;
    acc.1 = t - acc.0 - y;
    acc.0 = t;
}

fn pyround_n(x: f64, n: usize) -> f64 {
    format!("{x:.n$}").parse().unwrap()
}

fn np_round(x: f64, d: i32) -> f64 {
    let p = 10f64.powi(d);
    (x * p).round_ties_even() / p
}

/// np.quantile(a, q) linear(numpy _lerp)
fn quantile(a: &[f64], q: f64) -> f64 {
    let mut s = a.to_vec();
    s.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = s.len();
    let vi = (n - 1) as f64 * q;
    if vi >= (n - 1) as f64 {
        return s[n - 1];
    }
    let i = vi.floor();
    let g = vi - i;
    let (x, y) = (s[i as usize], s[i as usize + 1]);
    let d = y - x;
    if g >= 0.5 { y - d * (1.0 - g) } else { x + d * g }
}

fn median(a: &[f64]) -> f64 {
    let mut s = a.to_vec();
    s.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

#[derive(Clone, Default)]
pub struct K {
    pub t: Vec<i64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub qv: Vec<f64>,
}

fn sql_str(p: &Path) -> String {
    p.to_string_lossy().replace('\'', "''")
}

fn read_cache(p: &Path) -> Result<K, String> {
    let c = Connection::open_in_memory().map_err(|e| e.to_string())?;
    let mut st = c.prepare(&format!("SELECT t, high, low, close, qv FROM read_parquet('{}')", sql_str(p))).map_err(|e| e.to_string())?;
    let mut k = K::default();
    let mut rows = st.query([]).map_err(|e| e.to_string())?;
    while let Some(r) = rows.next().map_err(|e| e.to_string())? {
        k.t.push(r.get(0).map_err(|e| e.to_string())?);
        k.high.push(r.get(1).map_err(|e| e.to_string())?);
        k.low.push(r.get(2).map_err(|e| e.to_string())?);
        k.close.push(r.get(3).map_err(|e| e.to_string())?);
        k.qv.push(r.get(4).map_err(|e| e.to_string())?);
    }
    Ok(k)
}

fn write_cache(k: &K, p: &Path) -> Result<(), String> {
    let c = Connection::open_in_memory().map_err(|e| e.to_string())?;
    c.execute_batch("SET threads=1; CREATE TABLE k(t BIGINT, high DOUBLE, low DOUBLE, close DOUBLE, qv DOUBLE);").map_err(|e| e.to_string())?;
    {
        let mut a = c.appender("k").map_err(|e| e.to_string())?;
        for i in 0..k.t.len() {
            a.append_row(duckdb::params![k.t[i], k.high[i], k.low[i], k.close[i], k.qv[i]]).map_err(|e| e.to_string())?;
        }
    }
    let tmp = p.with_extension("tmp");
    c.execute_batch(&format!("COPY k TO '{}' (FORMAT PARQUET)", sql_str(&tmp))).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, p).map_err(|e| e.to_string())
}

/// `k1h_spans`
fn spans(old: Option<(i64, i64)>, now_ms: i64) -> Vec<(i64, i64)> {
    let need = now_ms - KEEP_DAYS * DAY;
    let mut s = vec![(old.map_or(need, |o| o.1 + HR), now_ms - HR)];
    if let Some((mn, _)) = old {
        if mn > need + DAY {
            s.insert(0, (need, mn));
        }
    }
    s
}

pub type Get<'a> = &'a mut dyn FnMut(&str) -> Result<Vec<u8>, String>;

/// `fetch_k1h` -- REST 실패(Err) = 저장분 + 그때까지 받은 것으로 계속 · JSON 이 깨지면 Err(원본 예외 = 그 코인 실패).
pub fn fetch_k1h(symbol: &str, cache: &Path, now_ms: i64, no_rest: bool, get: Get, sleep: &mut dyn FnMut(f64)) -> Result<K, String> {
    let old = if cache.exists() { read_cache(cache)? } else { K::default() };
    if no_rest {
        return Ok(old);
    }
    let mm = (!old.t.is_empty()).then(|| (*old.t.iter().min().unwrap(), *old.t.iter().max().unwrap()));
    let mut new: Vec<(i64, f64, f64, f64, f64)> = vec![];
    'outer: for (mut start, end) in spans(mm, now_ms) {
        while start < end {
            let u = format!("{}/fapi/v1/klines?symbol={symbol}&interval=1h&startTime={start}&limit=1500", rtcollect::cboe_summary::BINANCE);
            let body = match get(&u) {
                Ok(b) => b,
                Err(e) => {
                    println!("{symbol}: klines {e} -- 저장분으로 계속");
                    break 'outer;
                }
            };
            let raw: Vec<Vec<Value>> = serde_json::from_slice(&body).map_err(|e| format!("JSONDecodeError: {e}"))?;
            if raw.is_empty() {
                break;
            }
            let f = |v: &Value| v.as_str().and_then(|s| s.parse::<f64>().ok()).or_else(|| v.as_f64()).ok_or_else(|| "ValueError: float".to_string());
            for r in &raw {
                if r[6].as_i64().ok_or("ValueError: int")? < now_ms {
                    new.push((r[0].as_i64().ok_or("ValueError: int")?, f(&r[2])?, f(&r[3])?, f(&r[4])?, f(&r[7])?));
                }
            }
            if raw.len() < 1500 {
                break;
            }
            start = raw[raw.len() - 1][0].as_i64().ok_or("ValueError: int")? + HR;
            sleep(0.5);
        }
    }
    if new.is_empty() {
        return Ok(old);
    }
    let mut rows: Vec<(i64, f64, f64, f64, f64)> = (0..old.t.len()).map(|i| (old.t[i], old.high[i], old.low[i], old.close[i], old.qv[i])).chain(new).collect();
    let mut seen = BTreeSet::new();
    rows.retain(|r| seen.insert(r.0)); // drop_duplicates("t") = 처음 것
    rows.sort_by_key(|r| r.0);
    rows.retain(|r| r.0 >= now_ms - KEEP_DAYS * DAY);
    let k = K { t: rows.iter().map(|r| r.0).collect(), high: rows.iter().map(|r| r.1).collect(), low: rows.iter().map(|r| r.2).collect(),
                close: rows.iter().map(|r| r.3).collect(), qv: rows.iter().map(|r| r.4).collect() };
    write_cache(&k, cache)?;
    Ok(k)
}

/// `first_touch` -- 봉 i 뒤 처음 닿은 봉(없으면 -1). ponytail: 64칸 묶음 최솟값·최댓값으로 건너뛴다(원본은 numpy 전수).
fn first_touch(level: &[f64], long: bool, hi: &[f64], lo: &[f64]) -> Vec<i64> {
    const B: usize = 64;
    let n = hi.len();
    let nb = n.div_ceil(B);
    let bmin: Vec<f64> = (0..nb).map(|b| lo[b * B..((b + 1) * B).min(n)].iter().fold(f64::INFINITY, |a, &x| a.min(x))).collect();
    let bmax: Vec<f64> = (0..nb).map(|b| hi[b * B..((b + 1) * B).min(n)].iter().fold(f64::NEG_INFINITY, |a, &x| a.max(x))).collect();
    let hit = |j: usize, lv: f64| if long { lo[j] <= lv } else { hi[j] >= lv };
    level.iter().enumerate().map(|(i, &lv)| {
        let mut j = i + 1;
        while j < n {
            if j % B == 0 {
                let b = j / B;
                if !(if long { bmin[b] <= lv } else { bmax[b] >= lv }) {
                    j += B;
                    continue;
                }
            }
            if hit(j, lv) {
                return j as i64;
            }
            j += 1;
        }
        -1
    }).collect()
}

struct Pos {
    long: bool,
    liq: f64,
    w: f64,
    e: i64,
    s: i64,
}

/// `positions(k)` -- 레버리지 × (롱, 숏) 덩어리 순서.
fn positions(k: &K) -> Vec<Pos> {
    let n = k.t.len();
    let mid: Vec<f64> = (0..n).map(|i| (k.high[i] + k.low[i]) / 2.0).collect();
    let w: Vec<f64> = k.qv.iter().map(|q| q / LEV.len() as f64).collect();
    let day: Vec<i64> = k.t.iter().map(|t| t.div_euclid(DAY)).collect();
    let mut out = vec![];
    for &l in &LEV {
        let l = l as f64;
        for long in [true, false] {
            let liq: Vec<f64> = if long { mid.iter().map(|m| m * (1.0 - 1.0 / l + MMR)).collect() } else { k.close.iter().map(|c| c * (1.0 + 1.0 / l - MMR)).collect() };
            let sw = first_touch(&liq, long, &k.high, &k.low);
            for i in 0..n {
                out.push(Pos { long, liq: liq[i], w: w[i], e: day[i], s: if sw[i] >= 0 { day[(sw[i] as usize).min(n - 1)] } else { 1_000_000_000 } });
            }
        }
    }
    out
}

/// `heat_matrix` → (칸 × 표시일, 첫 칸 가격). 칸 없으면 (빈, 0.0).
fn heat_matrix(r: &[Pos], last_day: i64, view: i64, win: i64, row: f64) -> (Vec<Vec<f64>>, f64) {
    let d0 = last_day - view + 1;
    let sel: Vec<(i64, i64, i64, f64)> = r.iter().filter_map(|p| {
        let st = p.e.max(d0);
        let en = (p.e + win).min(p.s - 1).min(last_day);
        (st <= en).then(|| ((p.liq / row).floor() as i64, st, en, p.w))
    }).collect();
    if sel.is_empty() {
        return (vec![], 0.0);
    }
    let lo_b = sel.iter().map(|x| x.0).min().unwrap();
    let hi_b = sel.iter().map(|x| x.0).max().unwrap() + 1;
    let mut mat = vec![vec![0.0; view as usize + 1]; (hi_b - lo_b) as usize];
    for &(b, st, _, w) in &sel {
        mat[(b - lo_b) as usize][(st - d0) as usize] += w;
    }
    for &(b, _, en, w) in &sel {
        mat[(b - lo_b) as usize][(en - d0 + 1) as usize] -= w;
    }
    for rw in mat.iter_mut() {
        for j in 1..rw.len() {
            rw[j] += rw[j - 1];
        }
        rw.truncate(view as usize);
    }
    let mx = mat.iter().flatten().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    for x in mat.iter_mut().flatten() {
        if x.abs() < mx * 1e-9 {
            *x = 0.0;
        }
    }
    (mat, lo_b as f64 * row)
}

fn encode_heat(mat: &[Vec<f64>], lo: f64, row: f64, view: usize) -> Map<String, Value> {
    let pos: Vec<f64> = mat.iter().flatten().copied().filter(|&x| x > 0.0).collect();
    let mut o = Map::new();
    o.insert("lo".into(), json!(lo));
    o.insert("row".into(), json!(row));
    if pos.is_empty() {
        o.insert("cols".into(), json!([]));
        return o;
    }
    let q = quantile(&pos, 0.98);
    let t: Vec<Vec<f64>> = mat.iter().map(|rw| rw.iter().map(|&x| { let v = (x / q).min(1.0); if v < HEAT_CUT { 0.0 } else { v } }).collect()).collect();
    let cols: Vec<Value> = (0..view).map(|j| {
        let nz: Vec<usize> = (0..t.len()).filter(|&i| t[i][j] != 0.0).collect();
        match (nz.first(), nz.last()) {
            (Some(&a), Some(&b)) => {
                let mut c = vec![json!(a)];
                c.extend((a..=b).map(|i| json!((t[i][j] * 1000.0).round_ties_even() as i64)));
                Value::Array(c)
            }
            _ => json!([]),
        }
    }).collect();
    o.insert("cols".into(), Value::Array(cols));
    o
}

/// groupby(키)["w"].sum() -- Kahan · 키 오름차순.
fn group_sum<K2: Ord>(it: impl Iterator<Item = (K2, f64)>) -> BTreeMap<K2, f64> {
    let mut m: BTreeMap<K2, (f64, f64)> = BTreeMap::new();
    for (k, v) in it {
        kahan(m.entry(k).or_insert((0.0, 0.0)), v);
    }
    m.into_iter().map(|(k, v)| (k, v.0)).collect()
}

/// f64 열쇠(np.round 결과 -- 정수 값)를 순서 있는 열쇠로.
#[derive(PartialEq, PartialOrd, Clone, Copy)]
struct F(f64);
impl Eq for F {}
#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for F {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&o.0).unwrap()
    }
}

fn remain_kpi(r: &[Pos], last_day: i64, win: i64, price: f64, row: f64) -> Map<String, Value> {
    let z: Vec<&Pos> = r.iter().filter(|p| p.e >= last_day - win).collect();
    let a: Vec<&Pos> = z.iter().copied().filter(|p| p.s > last_day).collect();
    let mut o = Map::new();
    if a.is_empty() {
        return o;
    }
    let ws = |v: &[&Pos]| np_sum(&v.iter().map(|p| p.w).collect::<Vec<_>>());
    let (lo_, up): (Vec<&Pos>, Vec<&Pos>) = a.iter().partition(|p| p.long);
    let big = |x: &[&Pos]| -> Value {
        if x.is_empty() {
            return Value::Null;
        }
        let g = group_sum(x.iter().map(|p| (F((p.liq / row).round_ties_even()), p.w)));
        let mut best: Option<(F, f64)> = None;
        for (k, v) in g {
            if best.is_none_or(|b| v > b.1) {
                best = Some((k, v));
            }
        }
        json!(best.unwrap().0 .0 * row)
    };
    let tot = ws(&a);
    let step = price * 0.01;
    let prof = group_sum(a.iter().map(|p| ((F((p.liq / step).round_ties_even()), !p.long), p.w)));
    let zl: Vec<&Pos> = z.iter().copied().filter(|p| p.long).collect();
    let zs: Vec<&Pos> = z.iter().copied().filter(|p| !p.long).collect();
    let near: Vec<&Pos> = a.iter().copied().filter(|p| (p.liq / price - 1.0).abs() <= 0.03).collect();
    let old: Vec<&Pos> = a.iter().copied().filter(|p| p.e < last_day - 7).collect();
    o.insert("ll".into(), json!(ws(&lo_)));
    o.insert("ul".into(), json!(ws(&up)));
    o.insert("lm".into(), json!(ws(&zl)));
    o.insert("um".into(), json!(ws(&zs)));
    o.insert("lb".into(), big(&lo_));
    o.insert("ub".into(), big(&up));
    o.insert("near3".into(), json!(ws(&near) / tot));
    o.insert("old7".into(), json!(ws(&old) / tot));
    o.insert("prof".into(), Value::Array(prof.into_iter().map(|((b, short), w)| json!([pyround_n(b.0 * step, 8), i64::from(short), pyround_n(w / tot, 5)])).collect()));
    o
}

/// hot SQLite 한 값(`last`) -- 파일·표가 없으면 None · 0 도 None.
fn last(hot: &Path, db: &str, sql: &str, args: &[&dyn rusqlite::ToSql]) -> Option<f64> {
    let c = rusqlite::Connection::open_with_flags(hot.join(db), OpenFlags::SQLITE_OPEN_READ_ONLY).ok()?;
    let v: Option<f64> = c.query_row(sql, args, |r| r.get(0)).ok()?;
    v.filter(|x| *x != 0.0)
}

pub fn oi_usd(hot: &Path, symbol: &str, price: f64, now_ms: i64) -> Option<f64> {
    let bn = last(hot, "binance_ctx.sqlite", "select open_interest from oi_1s where symbol = ? order by ts_ms desc limit 1", &[&symbol.to_lowercase()])?;
    let since = now_ms - 600_000;
    let base = symbol.strip_suffix("USDT").unwrap_or(symbol);
    let ok = last(hot, "okx_ctx.sqlite", "select oi_base from okx_oi where inst = ? and ts_ms >= ? order by ts_ms desc limit 1", &[&format!("{}-USDT-SWAP", base.to_uppercase()), &since]);
    let by = last(hot, "bybit_ctx.sqlite", "select oi_base from bybit_oi where inst = ? and ts_ms >= ? order by ts_ms desc limit 1", &[&symbol.to_uppercase(), &since]);
    Some((bn + ok.unwrap_or(0.0) + by.unwrap_or(0.0)) * price)
}

/// numpy.convolve(v, ones(11), "same") -- OpenBLAS ddot 꼬리 = 창 안 차례 합(길이 < 11 이면 두 배열이 바뀌어 뒤집힌 차례 · 출력 11칸).
fn conv11(v: &[f64]) -> Vec<f64> {
    let m = v.len();
    if m >= 11 {
        (0..m).map(|j| {
            let mut s = 0.0;
            for x in &v[j.saturating_sub(5)..(j + 6).min(m)] {
                s += x;
            }
            s
        }).collect()
    } else {
        (0..11).map(|j| {
            let c = j + (m - 1) / 2;
            let mut s = 0.0;
            for i in (c.saturating_sub(10)..(c + 1).min(m)).rev() {
                s += v[i];
            }
            s
        }).collect()
    }
}

/// `peaks(v, px)`
fn peaks(v: &[f64], px: &[f64]) -> Value {
    let mut v = v.to_vec();
    let mut out = vec![];
    for _ in 0..3 {
        let cs = conv11(&v);
        let mut j = 0;
        for (i, &x) in cs.iter().enumerate() {
            if x > cs[j] {
                j = i;
            }
        }
        if !(cs[j] > 0.0) {
            break;
        }
        let (a, b) = (j.saturating_sub(5).min(v.len()), (j + 6).min(v.len()));
        let vp: Vec<f64> = (a..b).map(|i| v[i] * px[i]).collect();
        let c = np_sum(&vp) / np_sum(&v[a..b]);
        out.push(json!([pyround_n(c, 8), pyround_n(cs[j], 6)]));
        for i in 0..v.len() {
            if (px[i] / c - 1.0).abs() <= 0.01 {
                v[i] = 0.0;
            }
        }
    }
    Value::Array(out)
}

/// compute_tier_profile(w, price, oi) 중 pain 이 쓰는 것: (lo, round(bw, 6), 레버리지 구간 넷의 칸 값을 더한 것).
fn tier_profile(t: &[i64], high: &[f64], low: &[f64], close: &[f64], vol: &[f64], price: f64, oi: Option<f64>) -> Option<(i64, f64, Vec<f64>)> {
    let n = t.len();
    if n < 20 || !(price > 0.0) {
        return None;
    }
    let tl = t[n - 1];
    let w: Vec<f64> = (0..n).map(|i| {
        let age = ((tl - t[i]) * 1_000_000) as f64 / 1e9 / 3600.0;
        vol[i] * (-age / 240.0).exp() / LEV.len() as f64
    }).collect();
    let (mut fmin, mut fmax) = (vec![f64::INFINITY; n], vec![f64::NEG_INFINITY; n]);
    for i in (0..n - 1).rev() {
        fmin[i] = low[i + 1].min(fmin[i + 1]);
        fmax[i] = high[i + 1].max(fmax[i + 1]);
    }
    let bw = (price * 0.001).max(1e-9);
    let mid: Vec<f64> = (0..n).map(|i| (high[i] + low[i]) / 2.0).collect();
    let mut per: Vec<(Vec<i64>, Vec<f64>)> = vec![];
    for (_, levs) in TIER_GROUPS {
        let (mut ks, mut vs) = (vec![], vec![]);
        for &l in levs {
            let l = l as f64;
            for long in [true, false] {
                for i in 0..n {
                    let (entry, liq) = if long { (mid[i], mid[i] * (1.0 - 1.0 / l + MMR)) } else { (close[i], close[i] * (1.0 + 1.0 / l - MMR)) };
                    let alive = if long { fmin[i] > liq } else { fmax[i] < liq };
                    if alive && liq > 0.0 {
                        ks.push((liq / bw).round_ties_even() as i64);
                        vs.push(w[i] * 1.0 * entry);
                    }
                }
            }
        }
        per.push((ks, vs));
    }
    let allk: Vec<i64> = per.iter().flat_map(|p| p.0.iter().copied()).collect();
    if allk.is_empty() {
        return None;
    }
    let lo = *allk.iter().min().unwrap();
    let nn = (*allk.iter().max().unwrap() - lo + 1) as usize;
    let mut total = 0.0;
    for p in &per {
        total += np_sum(&p.1);
    }
    if !(total > 0.0) {
        return None;
    }
    let (scale, dec) = match oi {
        Some(o) if o > 0.0 => (2.0 * o / total, 2),
        _ => (1.0 / total, 8),
    };
    let mut sum = vec![0.0; nn];
    for (g, p) in per.iter().enumerate() {
        let mut arr = vec![0.0; nn];
        for (k, v) in p.0.iter().zip(&p.1) {
            arr[(k - lo) as usize] += v * scale;
        }
        for i in 0..nn {
            let x = np_round(arr[i], dec);
            sum[i] = if g == 0 { x } else { sum[i] + x };
        }
    }
    Some((lo, pyround_n(bw, 6), sum))
}

fn pain(k: &K, price: f64, oi: Option<f64>) -> Value {
    let n = k.t.len();
    if n < 170 {
        return Value::Null;
    }
    let vol: Vec<f64> = (0..n).map(|i| k.qv[i] / k.close[i]).collect();
    let mut per = Map::new();
    for (name, h) in PAIN_PERIODS {
        let m = h.max(24).min(n);
        let a = n - m;
        let mut v: Vec<f64> = vol[a..].to_vec();
        for x in v.iter_mut().take(m.saturating_sub(h)) {
            *x = 0.0;
        }
        let Some((lo, bw, vals)) = tier_profile(&k.t[a..], &k.high[a..], &k.low[a..], &k.close[a..], &v, price, oi) else { continue };
        let px: Vec<f64> = (0..vals.len()).map(|i| (lo + i as i64) as f64 * bw).collect();
        let dn: Vec<f64> = (0..vals.len()).map(|i| if px[i] < price { vals[i] } else { 0.0 }).collect();
        let up: Vec<f64> = (0..vals.len()).map(|i| if px[i] > price { vals[i] } else { 0.0 }).collect();
        per.insert(name.into(), json!({"dn": peaks(&dn, &px), "up": peaks(&up, &px)}));
    }
    let c = &k.close;
    let lg: Vec<f64> = c[n.saturating_sub(169)..].iter().map(|x| x.ln()).collect();
    let d: Vec<f64> = lg.windows(2).map(|w| w[1] - w[0]).collect();
    let mean = np_sum(&d) / d.len() as f64;
    let sq: Vec<f64> = d.iter().map(|x| (x - mean) * (x - mean)).collect();
    let sig = (np_sum(&sq) / (d.len() - 1) as f64).sqrt() * 24f64.sqrt();
    let k3: Vec<Value> = (n - 72..n).map(|i| json!([k.t[i].div_euclid(1000), c[i - 1], k.high[i], k.low[i], c[i]])).collect();
    json!({"price": price, "unit": if oi.is_some_and(|o| o != 0.0) { "oi_usd" } else { "share" }, "sigma_d": sig, "per": per, "k3": k3})
}

pub struct Ev {
    t: i64,
    side: Option<String>,
    usd: f64,
    px: Option<f64>,
    bn: bool,
}

fn parquets(d: &Path) -> bool {
    let Ok(rd) = std::fs::read_dir(d) else { return false };
    rd.flatten().any(|e| {
        let p = e.path();
        if p.is_dir() { parquets(&p) } else { p.extension().is_some_and(|x| x == "parquet") }
    })
}

/// `lake_read` → (t, side, usd, px) -- usd 는 None 이 있을 수 있다(dropna 전).
fn lake_read(lake: &Path, sub: &str, cols: &str, where_: &str) -> Result<Vec<(i64, Option<String>, Option<f64>, Option<f64>)>, String> {
    let d = lake.join(sub);
    if !parquets(&d) {
        return Ok(vec![]);
    }
    let c = Connection::open_in_memory().map_err(|e| e.to_string())?;
    let q = format!("select {cols} from read_parquet('{}/**/*.parquet', hive_partitioning=true, union_by_name=true) {where_}", sql_str(&d));
    let mut st = c.prepare(&q).map_err(|e| e.to_string())?;
    let rows = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))).map_err(|e| e.to_string())?;
    rows.collect::<Result<_, _>>().map_err(|e| e.to_string())
}

/// `events(coin, symbol)` -- t 순(안정 정렬).
pub fn events(lake: &Path, hot: &Path, coin: &str, symbol: &str) -> Result<Vec<Ev>, String> {
    let bn = lake_read(lake, &format!("binance/liquidations/coin={coin}"), "ts_ms t, side, usd, price px", "")?;
    let ok = lake_read(lake, "okx/liquidations/coin=ALL", "ts_ms t, pos_side side, bk_px*sz_base usd, bk_px px", &format!("where inst_id = '{coin}-USDT-SWAP'"))?;
    let since = bn.iter().map(|r| r.0).max().unwrap_or(0);
    let mut e: Vec<Ev> = vec![];
    let push = |e: &mut Vec<Ev>, rows: Vec<(i64, Option<String>, Option<f64>, Option<f64>)>, bn: bool| {
        e.extend(rows.into_iter().filter_map(|(t, side, usd, px)| usd.filter(|u| !u.is_nan()).map(|usd| Ev { t, side, usd, px, bn })));
    };
    push(&mut e, bn, true);
    push(&mut e, ok, false);
    let hot_rows = (|| -> Result<Vec<(i64, Option<String>, Option<f64>, Option<f64>)>, String> {
        let c = rusqlite::Connection::open_with_flags(hot.join("binance_ctx.sqlite"), OpenFlags::SQLITE_OPEN_READ_ONLY).map_err(|e| e.to_string())?;
        let mut s = c.prepare("select ts_ms t, side, usd, price px from liquidations where symbol = ? and ts_ms > ?").map_err(|e| e.to_string())?;
        let r = s.query_map(rusqlite::params![symbol, since], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))).map_err(|e| e.to_string())?;
        r.collect::<Result<_, _>>().map_err(|e| e.to_string())
    })();
    match hot_rows {
        Ok(rows) => push(&mut e, rows, true),
        Err(x) => println!("{}: hot 청산 {x} -- lake 만", coin.to_lowercase()),
    }
    e.sort_by_key(|x| x.t);
    Ok(e)
}

fn band(e: &[Ev], today: i64) -> Value {
    let x: Vec<&Ev> = e.iter().filter(|v| v.t.div_euclid(DAY) < today).collect();
    if x.is_empty() {
        return Value::Null;
    }
    let ok_days: BTreeSet<i64> = x.iter().filter(|v| !v.bn).map(|v| v.t.div_euclid(DAY)).collect();
    let all: BTreeSet<i64> = x.iter().map(|v| v.t.div_euclid(DAY)).collect();
    let ok_days = if ok_days.is_empty() { all.clone() } else { ok_days };
    let days: Vec<i64> = all.intersection(&ok_days).copied().collect();
    let days: Vec<i64> = days[days.len().saturating_sub(30)..].to_vec();
    if days.len() < 7 {
        return Value::Null;
    }
    let idx: BTreeMap<i64, usize> = days.iter().enumerate().map(|(i, d)| (*d, i)).collect();
    let mut m = vec![vec![0.0; 288]; days.len()];
    for v in &x {
        if let Some(&i) = idx.get(&v.t.div_euclid(DAY)) {
            m[i][(v.t.rem_euclid(DAY) / 300_000) as usize] += v.usd;
        }
    }
    for r in m.iter_mut() {
        for j in 1..288 {
            r[j] += r[j - 1];
        }
    }
    let mut o = Map::new();
    o.insert("n".into(), json!(days.len()));
    for (k, q) in [("p25", 0.25), ("p50", 0.5), ("p75", 0.75)] {
        o.insert(k.into(), Value::Array((0..288).map(|j| json!(quantile(&m.iter().map(|r| r[j]).collect::<Vec<_>>(), q).round_ties_even() as i64)).collect()));
    }
    Value::Object(o)
}

fn real_rows(e: &[Ev], price: f64, now_ms: i64) -> Value {
    let x: Vec<&Ev> = e.iter().filter(|v| v.bn && v.t >= now_ms - 7 * DAY && v.px.is_some_and(|p| p > 0.0)).collect();
    if x.is_empty() {
        return json!([]);
    }
    let step = price * REAL_PCT;
    let g = group_sum(x.iter().filter_map(|v| v.side.clone().map(|s| ((F((v.px.unwrap() / step).round_ties_even()), s), v.usd))));
    let mut rows: BTreeMap<F, (f64, f64)> = BTreeMap::new();
    for ((b, s), u) in g {
        let r = rows.entry(b).or_insert((0.0, 0.0));
        match s.as_str() {
            "long" => r.0 = u,
            "short" => r.1 = u,
            _ => {}
        }
    }
    Value::Array(rows.into_iter().map(|(b, (l, s))| json!([pyround_n(b.0 * step, 8), l.round_ties_even() as i64, s.round_ties_even() as i64])).collect())
}

fn elasticity(e: &[Ev], k: &K, now_ms: i64) -> Value {
    let x: Vec<&Ev> = e.iter().filter(|v| v.bn && v.t >= now_ms - 30 * DAY).collect();
    if x.is_empty() {
        return Value::Null;
    }
    let h = group_sum(x.iter().map(|v| (v.t.div_euclid(HR), v.usd)));
    let hmin = *h.keys().next().unwrap();
    let (mut up, mut dn) = (vec![], vec![]);
    for i in 0..k.t.len() {
        let hr = k.t[i].div_euclid(HR);
        if hr < hmin || i == 0 {
            continue; // 첫 봉 = pct_change NaN → dropna
        }
        let r = (k.close[i] / k.close[i - 1] - 1.0) * 100.0;
        let u = h.get(&hr).copied().unwrap_or(0.0);
        if r >= 0.5 {
            up.push(u / r);
        } else if r <= -0.5 {
            dn.push(u / -r);
        }
    }
    if up.len() < 10 || dn.len() < 10 {
        return Value::Null;
    }
    json!({"up": median(&up), "dn": median(&dn), "n_up": up.len(), "n_dn": dn.len()})
}

fn date(d: i64) -> String {
    let z = d + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let dd = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    format!("{:04}-{m:02}-{dd:02}", yoe + era * 400 + i64::from(m <= 2))
}

pub struct Ctx<'a> {
    pub lake: &'a Path,
    pub live: &'a Path,
    pub hot: &'a Path,
    pub no_rest: bool,
}

/// `build(coin, now_ms)` -- None = 봉 모자람(건너뜀).
pub fn build(cx: &Ctx, coin: &str, symbol: &str, now_ms: i64, get: Get, sleep: &mut dyn FnMut(f64)) -> Result<Option<Value>, String> {
    let k = fetch_k1h(symbol, &cx.live.join(format!("liq_tab_k1h_{coin}.parquet")), now_ms, cx.no_rest, get, sleep)?;
    let n = k.t.len();
    if n < 24 * 8 {
        println!("{coin}: 1시간봉 {n}개 -- 건너뜀");
        return Ok(None);
    }
    let price = k.close[n - 1];
    let row: f64 = format!("{:.1e}", price * ROW_PCT).parse().unwrap();
    let r = positions(&k);
    let last_day = k.t[n - 1].div_euclid(DAY);
    // daily: 일마다 첫 종가·최고·최저·끝 종가
    let mut daily: BTreeMap<i64, (f64, f64, f64, f64)> = BTreeMap::new();
    for i in 0..n {
        let d = k.t[i].div_euclid(DAY);
        let e = daily.entry(d).or_insert((k.close[i], f64::NEG_INFINITY, f64::INFINITY, k.close[i]));
        e.1 = e.1.max(k.high[i]);
        e.2 = e.2.min(k.low[i]);
        e.3 = k.close[i];
    }
    let made = group_sum(r.iter().map(|p| ((p.e, !p.long), p.w)));
    let left = group_sum(r.iter().filter(|p| p.s > last_day).map(|p| ((p.e, !p.long), p.w)));
    let days: Vec<Value> = daily.range(last_day - 364..).map(|(&d, &(o, h, l, c))| {
        let pct = |short: bool| {
            let m = made.get(&(d, short)).copied().unwrap_or(0.0);
            if m > 0.0 { pyround_n(left.get(&(d, short)).copied().unwrap_or(0.0) / m, 3) } else { 0.0 }
        };
        json!([date(d), o, h, l, c, pct(false), pct(true)])
    }).collect();
    let mut remain = Map::new();
    for (win, view) in WINDOWS {
        let (mat, lo) = heat_matrix(&r, last_day, view, win, row);
        let mut o = Map::new();
        o.insert("view".into(), json!(view));
        o.extend(encode_heat(&mat, lo, row, view as usize));
        o.extend(remain_kpi(&r, last_day, win, price, row));
        remain.insert(win.to_string(), Value::Object(o));
    }
    let e = events(cx.lake, cx.hot, &coin.to_uppercase(), symbol)?;
    let today = now_ms.div_euclid(DAY);
    let oi = oi_usd(cx.hot, symbol, price, now_ms);
    Ok(Some(json!({"ok": true, "v": 1, "coin": coin, "generated_ms": now_ms, "price": price, "kline_end_ms": k.t[n - 1] + HR, "days": days, "remain": remain,
                   "band": band(&e, today), "real": real_rows(&e, price, now_ms), "beta": elasticity(&e, &k, now_ms), "pain": pain(&k, price, oi)})))
}

fn write_atomic(p: &Path, s: &str) -> Result<(), String> {
    let tmp = p.with_extension("tmp");
    std::fs::write(&tmp, s).and_then(|_| std::fs::rename(&tmp, p)).map_err(|e| e.to_string())
}

/// `main()` -- coins = None 이면 lake 에 청산 폴더가 있는 코인(COIN_CONFIG 순서). 반환 = 로그.
pub fn run(cx: &Ctx, coin: Option<&str>, now_ms: i64, get: Get, sleep: &mut dyn FnMut(f64)) -> Result<Vec<String>, String> {
    let coins: Vec<(&str, &str)> = match coin {
        Some(c) => vec![*COINS.iter().find(|x| x.0 == c).ok_or_else(|| format!("KeyError: '{c}'"))?],
        None => COINS.iter().copied().filter(|(c, _)| cx.lake.join(format!("binance/liquidations/coin={}", c.to_uppercase())).exists()).collect(),
    };
    let mut log = vec![];
    let mut pains = Map::new();
    for (coin, sym) in coins {
        let out = match build(cx, coin, sym, now_ms, get, sleep) {
            Ok(Some(o)) => o,
            Ok(None) => continue,
            Err(e) => {
                log.push(format!("{coin}: 실패 {e}"));
                continue;
            }
        };
        if let Value::Object(p) = &out["pain"] {
            let mut m = Map::new();
            m.insert("generated_ms".into(), json!(now_ms));
            m.extend(p.clone());
            pains.insert(coin.into(), Value::Object(m));
        }
        let s = rtcollect::py_dumps_compact(&out);
        write_atomic(&cx.live.join(format!("liq_tab_{coin}.json")), &s)?;
        log.push(format!("{coin}: {}KB", s.len() / 1024));
    }
    if !pains.is_empty() {
        let p = cx.live.join("liq_tab_all.json");
        let mut old: Map<String, Value> = std::fs::read_to_string(&p).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .and_then(|v| v.get("coins").and_then(Value::as_object).cloned()).unwrap_or_default();
        for (k, v) in pains {
            old.insert(k, v);
        }
        write_atomic(&p, &rtcollect::py_dumps_compact(&json!({"ok": true, "coins": old})))?;
    }
    Ok(log)
}
