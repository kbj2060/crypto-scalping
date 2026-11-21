//! 청산지도 전진 점수 기록기 = scripts/live_liqmap_scorer_20261008.py(+ research_eth_liqmap_venue_leverage_validation_20261008.liq_map·realized).
//!   매시: 4시간 전 정시마다 «그때 만들 수 있던 지도»(A·B·D·F 넷)와 [o, o+4h) 실측 청산 → liqmap_score.jsonl 한 줄 · 요약 status.
//!   값 맞춤: numpy 원소별 연산 순서 그대로 · exp = libm(numpy 를 NPY_DISABLE_CPU_FEATURES=X86_V4 로 돌린 정답지) ·
//!   dict 누적 = 삽입 순서 · 쪽 합 = 파이썬 3.11 sum()(순서대로 더함 -- 3.12+ 는 Neumaier, 정답지 지도 칸 차이 0).
//!   merge_asof(backward, tolerance 5000ms 포함) · 🔴원본 sort_values 는 quicksort(안정 아님) -- 같은 ms 청산의 더하기 순서만 다를 수 있다(round 2 뒤 차이 0).
use rusqlite::{Connection, OpenFlags};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

pub const H: usize = 4;
pub const W: usize = 24;
const SPAN: i64 = 60;
const BACK_H: usize = 168;
pub const LIMIT: usize = W + H + BACK_H + 2;
const FWD_SINCE: &str = "2026-10-08T00:00:00+00:00";
const EVAL_BIN: f64 = 0.002;
const MMR: f64 = 0.005; // live_liquidation_map_20260824.MAINTENANCE_MARGIN_RATE
const HALFLIFE: f64 = 240.0; // RECENCY_HALFLIFE_HOURS
const T6: [i64; 6] = [10, 20, 25, 50, 75, 100];
const T14: [i64; 14] = [2, 3, 5, 8, 10, 12, 15, 20, 25, 33, 50, 75, 100, 125];
const HR: i64 = 3_600_000;

pub struct K {
    pub t: Vec<i64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
    qv: Vec<f64>,
    tbq: Vec<f64>,
}

/// `klines` 응답 → 끝난 봉만(`ct < time.time()*1000`).
pub fn parse_klines(body: &[u8], now: f64) -> Result<K, String> {
    let raw: Vec<Vec<Value>> = serde_json::from_slice(body).map_err(|e| format!("JSONDecodeError: {e}"))?;
    let mut k = K { t: vec![], high: vec![], low: vec![], close: vec![], volume: vec![], qv: vec![], tbq: vec![] };
    let f = |v: &Value| v.as_str().and_then(|s| s.parse::<f64>().ok()).or_else(|| v.as_f64()).ok_or("ValueError: float");
    for r in &raw {
        let i = |j: usize| r.get(j).and_then(Value::as_i64).ok_or("ValueError: int");
        let g = |j: usize| r.get(j).ok_or("IndexError").and_then(f);
        if (i(6)? as f64) >= now * 1000.0 {
            continue; // 진행 중인 봉
        }
        k.t.push(i(0)?);
        k.high.push(g(2)?);
        k.low.push(g(3)?);
        k.close.push(g(4)?);
        k.volume.push(g(5)?);
        k.qv.push(g(7)?);
        k.tbq.push(g(10)?);
    }
    Ok(k)
}

/// pandas Timestamp(ms, UTC).isoformat()
pub fn iso(t: i64) -> String {
    let (d, ms) = (t.div_euclid(86_400_000), t.rem_euclid(86_400_000));
    let (y, m, dd) = crate::cboe_summary::civil(d);
    let s = ms / 1000;
    let frac = if ms % 1000 != 0 { format!(".{:06}", ms % 1000 * 1000) } else { String::new() };
    format!("{y:04}-{m:02}-{dd:02}T{:02}:{:02}:{:02}{frac}+00:00", s / 3600, s / 60 % 60, s % 60)
}

type Key = (bool, i64); // (long, 상대 칸)

/// 삽입 순서 dict(`out.get(key, 0.0) + v`)
#[derive(Default)]
struct OMap {
    v: Vec<(Key, f64)>,
    ix: HashMap<Key, usize>,
}
impl OMap {
    fn add(&mut self, k: Key, x: f64) {
        match self.ix.get(&k) {
            Some(&i) => self.v[i].1 += x,
            None => {
                self.ix.insert(k, self.v.len());
                self.v.push((k, x));
            }
        }
    }
}

/// `V.liq_map(k, i, W, tiers, "splice", vol, halflife)` -- 봉 [i-W, i).
fn liq_map(k: &K, i: usize, tiers: &[i64], vl: &[f64], vs: &[f64], halflife: f64) -> (f64, OMap) {
    let r = i - W..i;
    let (hi, lo, cl) = (&k.high[r.clone()], &k.low[r.clone()], &k.close[r.clone()]);
    let n = W;
    let cp = cl[n - 1];
    let rec: Vec<f64> = (0..n).map(|j| (-((n - 1 - j) as f64) / halflife).exp()).collect();
    let tw = 1.0 / tiers.len() as f64;
    let (mut fmin, mut fmax) = (vec![f64::INFINITY; n], vec![f64::NEG_INFINITY; n]);
    for j in (0..n - 1).rev() {
        fmin[j] = lo[j + 1].min(fmin[j + 1]);
        fmax[j] = hi[j + 1].max(fmax[j + 1]);
    }
    let el: Vec<f64> = (0..n).map(|j| (hi[j] + lo[j]) / 2.0).collect();
    let mut out = OMap::default();
    for &l in tiers {
        let l = l as f64;
        for long in [true, false] {
            let (ent, vol) = if long { (&el[..], vl) } else { (cl, vs) };
            let fac = if long { 1.0 - 1.0 / l + MMR } else { 1.0 + 1.0 / l - MMR };
            for j in 0..n {
                let e = ent[j] * fac;
                let alive = if long { e < fmin[j] } else { e > fmax[j] };
                let off = ((e / cp - 1.0) / EVAL_BIN).floor() as i64;
                if alive && (if long { off < 0 } else { off >= 0 }) {
                    out.add((long, off), vol[i - W + j] * rec[j] * tw * 1.0 * ent[j]);
                }
            }
        }
    }
    (cp, out)
}

fn pyround_n(x: f64, n: usize) -> f64 {
    format!("{x:.n$}").parse().unwrap()
}

/// `_norm`: 쪽마다 ±SPAN 칸 합 1 · 1e-6 미만 버림 · 6자리.
fn norm(mp: &OMap) -> Value {
    let mut o = Map::new();
    for long in [true, false] {
        let mut tot = 0.0;
        for ((a, off), v) in &mp.v {
            if *a == long && off.abs() <= SPAN {
                tot += v;
            }
        }
        if tot == 0.0 {
            tot = 1.0;
        }
        let sorted: BTreeMap<i64, f64> = mp.v.iter().filter(|((a, off), _)| *a == long && off.abs() <= SPAN).map(|((_, off), v)| (*off, *v)).collect();
        let side: Map<String, Value> = sorted.into_iter().filter(|(_, v)| v / tot >= 1e-6).map(|(off, v)| (off.to_string(), json!(pyround_n(v / tot, 6)))).collect();
        o.insert(if long { "long" } else { "short" }.into(), Value::Object(side));
    }
    Value::Object(o)
}

pub struct Liq {
    ts: i64,
    long: bool,
    mark: f64,
    usd: f64,
}

/// `row_for(k, i, liq)`
fn row_for(k: &K, i: usize, liq: &[Liq]) -> Value {
    let dv: Vec<f64> = k.qv.iter().zip(&k.tbq).map(|(q, t)| q - t).collect();
    let mut maps = Map::new();
    let mut cp = 0.0;
    for (name, tiers, vl, vs, hl) in [("A", &T6[..], &k.volume, &k.volume, HALFLIFE), ("B", &T14[..], &k.volume, &k.volume, HALFLIFE),
                                      ("D", &T6[..], &k.tbq, &dv, HALFLIFE), ("F", &T6[..], &k.volume, &k.volume, 48.0)] {
        let (c, mp) = liq_map(k, i, tiers, vl, vs, hl);
        cp = c;
        maps.insert(name.into(), norm(&mp));
    }
    // realized
    let (t0, t1) = (k.t[i], k.t[i] + H as i64 * HR);
    let mn = k.low[i..i + H].iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let mx = k.high[i..i + H].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let long_lo = ((mn / cp - 1.0) / EVAL_BIN).floor() as i64;
    let short_hi = ((mx / cp - 1.0) / EVAL_BIN).floor() as i64;
    let mut r = OMap::default();
    let mut n_liq = 0;
    for x in liq.iter().filter(|x| x.ts >= t0 && x.ts < t1) {
        n_liq += 1;
        let off = ((x.mark / cp - 1.0) / EVAL_BIN).floor() as i64;
        if (x.long && off < 0) || (!x.long && off >= 0) {
            r.add((x.long, off), x.usd);
        }
    }
    let mut real = Map::new();
    for long in [true, false] {
        let s: BTreeMap<i64, f64> = r.v.iter().filter(|((a, _), _)| *a == long).map(|((_, o), u)| (*o, *u)).collect();
        real.insert(if long { "long" } else { "short" }.into(), Value::Object(s.into_iter().map(|(o, u)| (o.to_string(), json!(pyround_n(u, 2)))).collect()));
    }
    json!({"origin": iso(t0), "cp": cp, "H": H, "long_lo": long_lo, "short_hi": short_hi, "real": real, "n_liq": n_liq, "maps": maps})
}

/// `hot_liqs(t0, t1)`: 세 거래소 청산(쪽 모르면 버림) + 바이낸스 마크 1초 merge_asof(backward · 5000ms 포함 · 마크 없으면 버림).
pub fn hot_liqs(hot: &Path, t0: i64, t1: i64) -> Result<Vec<Liq>, String> {
    let q = |db: &str, sql: &str, a: (i64, i64)| -> Result<Vec<(i64, Option<String>, Option<f64>)>, String> {
        let c = Connection::open_with_flags(hot.join(db), OpenFlags::SQLITE_OPEN_READ_ONLY).map_err(|e| format!("OperationalError: {e}"))?;
        let mut s = c.prepare(sql).map_err(|e| format!("DatabaseError: {e}"))?;
        let rows = s.query_map(rusqlite::params![a.0, a.1], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))).map_err(|e| e.to_string())?;
        rows.collect::<Result<_, _>>().map_err(|e| e.to_string())
    };
    let mut d: Vec<(i64, bool, f64)> = vec![];
    for (db, sql) in [("binance_ctx.sqlite", "SELECT ts_ms, side, qty FROM liquidations WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?"),
                      ("okx_ctx.sqlite", "SELECT ts_ms, pos_side AS side, sz_base AS qty FROM okx_liquidations WHERE inst_id='ETH-USDT-SWAP' AND ts_ms>=? AND ts_ms<?"),
                      ("bybit_liq.sqlite", "SELECT ts_ms, side, qty FROM bybit_liquidations WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?")] {
        for (ts, side, qty) in q(db, sql, (t0, t1))? {
            let long = match side.as_deref() {
                Some("long" | "Buy") => true, // Bybit Buy = 롱 청산
                Some("short" | "Sell") => false,
                _ => continue,
            };
            d.push((ts, long, qty.unwrap_or(f64::NAN)));
        }
    }
    d.sort_by_key(|x| x.0); // ponytail: 안정 정렬 -- 원본 quicksort 와 같은 ms 끼리 순서만 다를 수 있다
    let m: Vec<(i64, Option<String>, Option<f64>)> = q("binance_ctx.sqlite", "SELECT ts_ms, NULL, mark FROM mark_price_1s WHERE symbol='ethusdt' AND ts_ms>=? AND ts_ms<?", (t0 - 5000, t1))?;
    let mut m: Vec<(i64, Option<f64>)> = m.into_iter().map(|(t, _, v)| (t, v)).collect();
    m.sort_by_key(|x| x.0);
    Ok(d.into_iter().filter_map(|(ts, long, qty)| {
        let j = m.partition_point(|x| x.0 <= ts);
        let (mt, mark) = *m.get(j.checked_sub(1)?)?;
        let mark = mark.filter(|_| ts - mt <= 5000)?;
        Some(Liq { ts, long, mark, usd: qty * mark })
    }).collect())
}

fn done_origins(out: &Path) -> Result<BTreeSet<String>, String> {
    let Ok(t) = std::fs::read_to_string(out) else { return Ok(BTreeSet::new()) };
    t.lines().filter(|l| !l.trim().is_empty()).map(|l| {
        let v: Value = serde_json::from_str(l).map_err(|e| format!("JSONDecodeError: {e}"))?;
        v["origin"].as_str().map(str::to_string).ok_or_else(|| "KeyError: 'origin'".to_string())
    }).collect()
}

fn write_status(out: &Path, status: &Path, now: f64) -> Result<(), String> {
    let o = done_origins(out)?;
    let fwd: Vec<&String> = o.iter().filter(|x| x.as_str() >= FWD_SINCE).collect();
    let days: BTreeSet<&str> = fwd.iter().map(|x| &x[..10]).collect();
    let up = iso(now.floor() as i64 * 1000).replace("+00:00", "Z");
    let v = json!({"n_fwd": fwd.len(), "days_fwd": days.len(), "last_origin": o.iter().next_back(), "updated": up});
    std::fs::write(status, crate::py_dumps_default(&v)).map_err(|e| format!("OSError: {e}"))
}

/// `main()` 한 번. klines = 가져오기(URL → 본문). 반환 = 로그 한 줄.
pub fn run(out: &Path, status: &Path, hot: &Path, now: f64, klines: &mut dyn FnMut(&str) -> Result<Vec<u8>, String>) -> Result<String, String> {
    let k = parse_klines(&klines(&format!("{}/fapi/v1/klines?symbol=ETHUSDT&interval=1h&limit={LIMIT}", crate::cboe_summary::BINANCE))?, now)?;
    let done = done_origins(out)?;
    let todo: Vec<usize> = (W..(k.t.len() + 1).saturating_sub(H)).filter(|&i| !done.contains(&iso(k.t[i]))).collect();
    let (Some(&a), Some(&b)) = (todo.first(), todo.last()) else {
        write_status(out, status, now)?;
        return Ok("새 기준 시각 없음".into());
    };
    let liq = hot_liqs(hot, k.t[a], k.t[b] + H as i64 * HR)?;
    if let Some(p) = out.parent() {
        std::fs::create_dir_all(p).map_err(|e| format!("OSError: {e}"))?;
    }
    let mut txt = String::new();
    for &i in &todo {
        txt += &crate::py_dumps_default_utf8(&row_for(&k, i, &liq));
        txt.push('\n');
    }
    use std::io::Write;
    std::fs::OpenOptions::new().create(true).append(true).open(out).and_then(|mut f| f.write_all(txt.as_bytes())).map_err(|e| format!("OSError: {e}"))?;
    write_status(out, status, now)?;
    Ok(format!("적음 {}줄 ({} ~ {}) · 청산 {}건", todo.len(), iso(k.t[a]), iso(k.t[b]), liq.len()))
}
