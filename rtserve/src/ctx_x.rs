//! 관계 읽기 입력 x(server.py `_flow_read_ctx` 2288~2406)와 그 재료 읽기 -- 2026-11-22 옮김.
//! 원본(Read 로 본 실제 줄): `_flow_read_ctx` 2288~2406 · `_card30_bars` 2282~2286 · `bybit_bars` 759~774 · `VENUE_SETS` 756 ·
//! micro_ref.py `is_weekend` 181~183 · `flow_hour_scales_from_bars` 186~217 · `sum_venue_bars` 220~227 · `flow_hour_scales_combos` 230~247 ·
//! `flow_hour_scales` 250~259(+ 그 뒤 return) · 갱신 주기 = collect_micro_ref 1926~1954(MICRO_BASELINE_SECONDS · 실패면 있던 값).
//! 원본의 «들고 있는 상태» 중 fr_bar_net · okx_bar_net_by(풋프린트 몫 · 반올림 전 봉 순합)는 **코어가 센다**(rtcore situation::flow_nets_tick · Micro.flow 로 옴 →
//! `apply_flow` 로 쌓는다) · flow_all_hold 는 `FlowState` 가 든다. 같은 걸음(1초)으로 부르면 같은 값이 나온다.
//! 일치 정답지: rtserve/tools/gen_ctx_x_fixture.py(원본 함수를 줄 번호와 함께 옮겨 적고 서버 판 numpy 로 부름) → tests/ctx_x_parity.rs.
use serde_json::{json, Map, Value as V};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

pub const BAR: i64 = 300;
/// server.py 756 -- 합산 조합(앞이 우선).
pub const VENUE_SETS: [&[&str]; 2] = [&["binance", "okx", "bybit"], &["binance", "okx"]];
const G3: [&str; 3] = ["whale", "mid", "retail"];

/// (주말 0/1, UTC 시) → {"mid","whale","retail"(,"mid30",…)} -- micro_ref.flow_hour_scales 반환.
pub type Scales = BTreeMap<(i64, i64), Map<String, V>>;

/// 원본이 make_app 안에 들고 있는 것: fr_bar_net(봉 → 델타·고래순·리테일순) · okx_bar_net_by[asset] · flow_all_hold[asset].
#[derive(Default, Clone, Debug)]
pub struct FlowState {
    pub hold: Option<Map<String, V>>,
}

/// Micro.flow 한 건을 받은 값에 얹는다(proto::FlowSums 받는 쪽 규칙).
pub fn apply_flow(n: &mut rtcore::situation::FlowNets, d: &rtcore::proto::FlowSums) {
    if d.full {
        n.bn.clear();
        n.okx.clear();
    }
    n.bn.extend(d.bn.iter().copied());
    n.okx.extend(d.okx.iter().copied());
    for b in &d.bn_drop {
        n.bn.remove(b);
    }
    for b in &d.okx_drop {
        n.okx.remove(b);
    }
    n.okx30 = d.okx30;
    n.bn_last = d.bn_last;
}

/// 완결 5분봉 하나(candles_day 또는 candles) -- `taker` 가 있는 것만 card30 입력(_card30_bars).
#[derive(Clone, Debug)]
pub struct Candle {
    pub time: i64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

pub struct XIn<'a> {
    pub now: f64,
    /// 코어가 센 풋프린트 몫(bn = 닫힌 봉 전부 · okx = onet · okx30).
    pub nets: &'a rtcore::situation::FlowNets,
    /// situation_state["bybit_bars"] (봉 → 매수·매도·고래순·리테일순).
    pub bybit: &'a BTreeMap<i64, [f64; 4]>,
    pub flow_scales: &'a Scales,
    /// 조합 이름("binance,okx,bybit") → 칸.
    pub flow_scales_all: &'a HashMap<String, Scales>,
    /// inp["bars"] 의 (close, oi_delta).
    pub bars: &'a [(Option<f64>, Option<f64>)],
    /// candles_day or candles (시간순).
    pub day5: &'a [Candle],
    /// _card30_pred 결과 (dir, reach).
    pub card30: (Option<f64>, Option<f64>),
    /// micro_state["payload"] (available 거짓이면 {} 로 본다).
    pub micro: &'a V,
    pub imb40_ring: &'a [f64],
}

/// micro_ref.is_weekend(181~183).
pub fn is_weekend(ts: i64) -> i64 {
    ((ts.div_euclid(86400) + 3).rem_euclid(7) >= 5) as i64
}

fn hour_utc(ts: i64) -> i64 {
    ts.rem_euclid(86400) / 3600
}

// ── numpy 꼴 계산(서버 판 numpy 2.2.5 · 정답지로 비트 대조) ──
/// np.convolve(v, np.ones(k), "valid"): 출력마다 길이 k 내적 -- BLAS ddot 은 16칸 미만이면 차례 합(0 에서).
fn roll(v: &[f64], k: usize) -> Vec<f64> {
    if v.len() < k {
        return vec![];
    }
    (0..=v.len() - k).map(|i| v[i..i + k].iter().fold(0.0, |s, x| s + x * 1.0)).collect()
}

/// np.std(ddof=0): 평균(짝 합 / n) → 차의 제곱 짝 합 / n → √.
fn np_std(a: &[f64]) -> f64 {
    let n = a.len() as f64;
    let m = rtcollect::ob5m::np_sum(a) / n;
    let sq: Vec<f64> = a.iter().map(|x| (x - m) * (x - m)).collect();
    (rtcollect::ob5m::np_sum(&sq) / n).sqrt()
}

/// np.percentile(a, q, method="linear"): 가상 위치 (n−1)·q/100 · _lerp(t ≥ 0.5 면 b − d·(1−t)).
fn np_percentile(a: &[f64], q: f64) -> f64 {
    let mut s = a.to_vec();
    s.sort_by(|x, y| x.total_cmp(y));
    let n = s.len();
    let vi = (n as f64 - 1.0) * (q / 100.0);
    let prev = vi.floor();
    let (i, j) = ((prev as usize).min(n - 1), (prev as usize + 1).min(n - 1));
    let t = vi - prev;
    let (lo, hi) = (s[i], s[j]);
    let d = hi - lo;
    if t >= 0.5 { hi - d * (1.0 - t) } else { lo + d * t }
}

fn finite_past(r: &[f64]) -> Vec<f64> {
    r[..r.len() - 1].iter().copied().filter(|x| x.is_finite()).collect()
}

fn num(m: &Map<String, V>, k: &str) -> Option<f64> {
    m.get(k).and_then(V::as_f64)
}

/// server.py _flow_read_ctx(2288~2406). 반환 = x dict(원본 차례).
pub fn flow_read_ctx(i: &XIn, st: &mut FlowState) -> Map<String, V> {
    let bar_start = (i.now as i64).div_euclid(BAR) * BAR;
    // done = 닫힌 풋프린트 봉 = 코어 fr_bar_net 의 열쇠(코어는 닫힌 봉을 전부 센다 · 풋프린트에서 사라진 봉은 지운다)
    let done: Vec<i64> = i.nets.bn.keys().copied().filter(|&b| b < bar_start).collect();
    let mut x = Map::new();
    let put = |x: &mut Map<String, V>, k: &str, v: V| {
        x.insert(k.to_string(), v);
    };
    let gmap = |vals: [f64; 3]| V::Object(G3.iter().zip(vals).map(|(g, v)| (g.to_string(), json!(v))).collect());
    if done.len() >= 12 {
        let (first, last) = (done[0], *done.last().unwrap());
        let grid: Vec<i64> = (first..=last).step_by(BAR as usize).collect();
        let nan3 = [f64::NAN; 3];
        let a: Vec<[f64; 3]> = grid.iter().map(|b| i.nets.bn.get(b).map_or(nan3, |v| [v[0], v[1], v[2]])).collect();
        let cols = |a: &[[f64; 3]]| -> [Vec<f64>; 3] {
            [a.iter().map(|r| r[1]).collect(), a.iter().map(|r| r[0] - r[1] - r[2]).collect(), a.iter().map(|r| r[2]).collect()]
        };
        let ser = cols(&a);
        let r60: Vec<Vec<f64>> = ser.iter().map(|v| roll(v, 12)).collect();
        let last_of = |r: &[Vec<f64>]| [r[0][r[0].len() - 1], r[1][r[1].len() - 1], r[2][r[2].len() - 1]];
        let hkey = (is_weekend(last), hour_utc(last));
        if r60.iter().all(|r| r[r.len() - 1].is_finite()) {
            let l60 = last_of(&r60);
            put(&mut x, "net60", gmap(l60));
            let past: Vec<Vec<f64>> = r60.iter().map(|r| finite_past(r)).collect();
            if past.iter().all(|p| p.len() >= 72 && np_std(p) > 0.0) {
                let sd: Vec<f64> = past.iter().map(|p| np_std(p)).collect();
                put(&mut x, "z60", gmap([l60[0] / sd[0], l60[1] / sd[1], l60[2] / sd[2]]));
            }
            if let Some(sc) = truthy_obj_map(i.flow_scales.get(&hkey)) {
                put(&mut x, "z60h", gmap(std::array::from_fn(|j| l60[j] / num(sc, G3[j]).unwrap())));
                let r30g: Vec<Vec<f64>> = ser.iter().map(|v| roll(v, 6)).collect();
                if G3.iter().all(|g| sc.contains_key(&format!("{g}30"))) && r30g.iter().all(|r| r[r.len() - 1].is_finite()) {
                    let l30 = last_of(&r30g);
                    put(&mut x, "z30h", gmap(std::array::from_fn(|j| l30[j] / num(sc, &format!("{}30", G3[j])).unwrap())));
                }
            }
        }
        let a0: Vec<f64> = a.iter().map(|r| r[0]).collect();
        let r30 = roll(&a0, 6);
        let p30 = finite_past(&r30);
        if r30[r30.len() - 1].is_finite() && p30.len() >= 72 && np_std(&p30) > 0.0 {
            put(&mut x, "cvd30_z", json!(r30[r30.len() - 1] / np_std(&p30)));
        }
        // 거래소 합산(2332~2373)
        let onet = &i.nets.okx;
        let ven_okx: Vec<[f64; 3]> = grid.iter().map(|b| onet.get(b).map_or(nan3, |o| [o[0], o[1], o[2]])).collect();
        let ven_bybit: Vec<[f64; 3]> = grid.iter().map(|b| i.bybit.get(b).map_or(nan3, |v| [v[0] - v[1], v[2], v[3]])).collect();
        let lb = last;
        let late = st.hold.as_ref().is_some_and(|h| !h.is_empty() && h["venues"].as_array().is_some_and(|v| v.iter().any(|s| s == "bybit")))
            && !i.bybit.contains_key(&lb)
            && i.now < (lb + 2 * BAR) as f64;
        if late {
            for (k, v) in st.hold.clone().unwrap() {
                x.insert(k, v);
            }
        }
        for vs in if late { &[][..] } else { &VENUE_SETS[..] } {
            let Some(sca) = truthy_obj_map(i.flow_scales_all.get(&vs.join(",")).and_then(|m| m.get(&hkey))) else { continue };
            // A = a + sum(ven[v] for v in vs[1:]) -- 파이썬 sum 은 0 에서
            let aa: Vec<[f64; 3]> = (0..grid.len())
                .map(|k| {
                    let mut add = [0.0; 3];
                    for v in &vs[1..] {
                        let src = if *v == "okx" { &ven_okx[k] } else { &ven_bybit[k] };
                        for j in 0..3 {
                            add[j] += src[j];
                        }
                    }
                    std::array::from_fn(|j| a[k][j] + add[j])
                })
                .collect();
            let sa = cols(&aa);
            let s60: Vec<Vec<f64>> = sa.iter().map(|v| roll(v, 12)).collect();
            if !s60.iter().all(|q| q[q.len() - 1].is_finite()) {
                continue;
            }
            let l60 = last_of(&s60);
            put(&mut x, "venues", json!(vs));
            put(&mut x, "net60_all", gmap(l60));
            put(&mut x, "z60h_all", gmap(std::array::from_fn(|j| l60[j] / num(sca, G3[j]).unwrap())));
            let s30: Vec<Vec<f64>> = sa.iter().map(|v| roll(v, 6)).collect();
            if G3.iter().all(|g| sca.contains_key(&format!("{g}30"))) && s30.iter().all(|q| q[q.len() - 1].is_finite()) {
                let l30 = last_of(&s30);
                put(&mut x, "net30_all", gmap(l30));
                put(&mut x, "z30h_all", gmap(std::array::from_fn(|j| l30[j] / num(sca, &format!("{}30", G3[j])).unwrap())));
            }
            let a0: Vec<f64> = aa.iter().map(|r| r[0]).collect();
            let r30a = roll(&a0, 6);
            let p30a = finite_past(&r30a);
            if r30a[r30a.len() - 1].is_finite() && p30a.len() >= 72 && np_std(&p30a) > 0.0 {
                put(&mut x, "cvd30_all_z", json!(r30a[r30a.len() - 1] / np_std(&p30a)));
            }
            // 코어가 같은 봉을 실었을 때만(시계가 어긋나 다른 봉이면 NaN → 비율 없음 -- 원본엔 없는 경우)
            let fb = i.nets.bn_last.filter(|l| l.0 == lb).map_or([f64::NAN; 2], |l| l.1);
            let has_by = vs.contains(&"bybit");
            let buy = fb[0] + onet[&lb][3] + if has_by { i.bybit[&lb][0] } else { 0.0 };
            let sell = fb[1] + onet[&lb][4] + if has_by { i.bybit[&lb][1] } else { 0.0 };
            if sell > 0.0 {
                put(&mut x, "taker_all", json!(buy / sell));
            }
            let keys = ["venues", "net60_all", "z60h_all", "net30_all", "z30h_all", "cvd30_all_z", "taker_all"];
            st.hold = Some(keys.iter().filter_map(|k| x.get(*k).map(|v| (k.to_string(), v.clone()))).collect());
            break;
        }
    }
    // okx30(2374~2376) -- 코어가 센 값
    if let Some(s) = i.nets.okx30 {
        put(&mut x, "okx30", json!(s));
    }
    // move60·oi60(2377~2382)
    let bs = i.bars;
    let truthy = |v: Option<f64>| v.is_some_and(|c| c != 0.0);
    if bs.len() >= 13 && truthy(bs[bs.len() - 1].0) && truthy(bs[bs.len() - 13].0) {
        put(&mut x, "move60", json!((bs[bs.len() - 1].0.unwrap() / bs[bs.len() - 13].0.unwrap() - 1.0) * 1e4));
        let ois: Vec<f64> = bs[bs.len() - 12..].iter().filter_map(|b| b.1).collect();
        if ois.len() >= 10 {
            put(&mut x, "oi60", json!(ois.iter().fold(0.0, |s, v| s + v)));
        }
    }
    // 24h 분위(2385~2396)
    let closed: Vec<&Candle> = i.day5.iter().filter(|c| c.time < bar_start).collect();
    let closes: Vec<f64> = closed[closed.len().saturating_sub(300)..].iter().map(|c| c.close).collect();
    if closes.len() >= 100 {
        let mv: Vec<f64> = (12..closes.len()).map(|k| ((closes[k] / closes[k - 12] - 1.0) * 1e4).abs()).collect();
        put(&mut x, "move60_p75", json!(np_percentile(&mv, 75.0)));
    }
    let cds = &closed[closed.len().saturating_sub(293)..];
    if cds.len() >= 66 {
        let rg: Vec<f64> = (5..cds.len())
            .map(|k| {
                let w = &cds[k - 5..=k];
                (w.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max) - w.iter().map(|c| c.low).fold(f64::INFINITY, f64::min)) / cds[k].close
            })
            .collect();
        let lastv = rg[rg.len() - 1];
        put(&mut x, "range30_pct", json!(rg.iter().filter(|v| **v <= lastv).count() as f64 / rg.len() as f64));
    }
    put(&mut x, "dir_p", json!(i.card30.0));
    put(&mut x, "reach_p", json!(i.card30.1));
    let empty = Map::new();
    let mp = i.micro.as_object().filter(|m| m.get("available").is_some_and(py_truthy)).unwrap_or(&empty);
    if let Some(imb) = mp.get("imb40").and_then(V::as_f64) {
        if i.imb40_ring.len() >= 600 {
            put(&mut x, "imb40", json!(imb));
            put(&mut x, "imb40_pct", json!(i.imb40_ring.iter().filter(|v| **v <= imb).count() as f64 / i.imb40_ring.len() as f64));
        }
    }
    let or_empty = |k: &str| mp.get(k).filter(|v| py_truthy(v)).cloned().unwrap_or(json!({}));
    put(&mut x, "liq60s", or_empty("liq60"));
    put(&mut x, "agree", mp.get("agree").cloned().unwrap_or(V::Null));
    put(&mut x, "trigger_live", json!(mp.get("trigger").filter(|v| py_truthy(v)).and_then(|t| t.get("live")).is_some_and(py_truthy)));
    put(&mut x, "sr", or_empty("sr"));
    put(&mut x, "act", mp.get("act").cloned().unwrap_or(V::Null));
    put(&mut x, "vol_pct", mp.get("vol60_pct").cloned().unwrap_or(V::Null));
    x
}

fn truthy_obj_map(m: Option<&Map<String, V>>) -> Option<&Map<String, V>> {
    m.filter(|o| !o.is_empty())
}

fn py_truthy(v: &V) -> bool {
    match v {
        V::Null => false,
        V::Bool(b) => *b,
        V::Number(n) => n.as_f64() != Some(0.0),
        V::String(s) => !s.is_empty(),
        V::Array(a) => !a.is_empty(),
        V::Object(o) => !o.is_empty(),
    }
}

// ── 재료 읽기 ──
fn ro(db: &Path) -> rusqlite::Result<rusqlite::Connection> {
    rusqlite::Connection::open_with_flags(db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX)
}

/// server.py bybit_bars(759~774): [t0, t1) Bybit 5분봉 → 봉: (매수, 매도, 고래 순, 리테일 순). 끊김이 걸친 봉은 뺀다. 없거나 실패면 빈 것.
pub fn bybit_bars(symbol: &str, t0: i64, t1: i64, db: &Path) -> BTreeMap<i64, [f64; 4]> {
    if !db.exists() {
        return BTreeMap::new();
    }
    let rd = || -> rusqlite::Result<(Vec<(i64, f64, f64, Option<f64>, Option<f64>)>, Vec<(i64, i64)>)> {
        let c = ro(db)?;
        let mut q = c.prepare(
            "SELECT ts_sec - ts_sec % 300, sum(buy_qty), sum(sell_qty), sum(whale_buy_qty - whale_sell_qty),
                    sum(retail_buy_qty - retail_sell_qty) FROM trade_tape_1s
             WHERE symbol = ? AND ts_sec >= ? AND ts_sec < ? GROUP BY 1",
        )?;
        let rows = q.query_map(rusqlite::params![symbol, t0, t1], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))?.collect::<Result<Vec<_>, _>>()?;
        let mut g = c.prepare("SELECT from_ms, to_ms FROM gaps WHERE symbol = ? AND from_ms < ? AND to_ms > ?")?;
        let gaps = g.query_map(rusqlite::params![symbol, t1 * 1000, t0 * 1000], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<Vec<_>, _>>()?;
        Ok((rows, gaps))
    };
    let Ok((rows, gaps)) = rd() else { return BTreeMap::new() };
    rows.into_iter()
        .filter(|(b, ..)| !gaps.iter().any(|&(f, t)| f < (b + 300) * 1000 && t > b * 1000))
        .map(|(b, bu, se, w, r)| (b, [bu, se, w.unwrap_or(0.0), r.unwrap_or(0.0)]))
        .collect()
}

/// micro_ref.flow_hour_scales_from_bars(186~217): (봉, 총 순, 고래, 리테일) → (주말, 시)별 60분·30분 롤링합의 1.4826·MAD.
pub fn flow_hour_scales_from_bars(rows: &[(i64, f64, f64, f64)]) -> Scales {
    let mut out = Scales::new();
    if rows.is_empty() {
        return out;
    }
    let t0 = rows[0].0;
    let n = ((rows[rows.len() - 1].0 - t0).div_euclid(300) + 1) as usize;
    let mut a = vec![[f64::NAN; 3]; n];
    for &(t, tot, wh, rt) in rows {
        a[(t - t0).div_euclid(300) as usize] = [tot - wh - rt, wh, rt];
    }
    for (win, suf) in [(12usize, ""), (6, "30")] {
        let r: Vec<Vec<f64>> = (0..3).map(|j| roll(&a.iter().map(|x| x[j]).collect::<Vec<_>>(), win)).collect();
        let len = r[0].len();
        for w in 0..2i64 {
            for h in 0..24i64 {
                if !suf.is_empty() && !out.contains_key(&(w, h)) {
                    continue;
                }
                let idx: Vec<usize> = (0..len)
                    .filter(|&k| {
                        let end = t0 + (k + win - 1) as i64 * 300;
                        is_weekend(end) == w && hour_utc(end) == h && (0..3).all(|j| r[j][k].is_finite())
                    })
                    .collect();
                if idx.len() < if w == 1 { 36 } else { 84 } {
                    continue;
                }
                let mad: Vec<f64> = (0..3)
                    .map(|j| {
                        let mut v: Vec<f64> = idx.iter().map(|&k| r[j][k]).collect();
                        let med = rtcollect::ob5m::np_median(&mut v.clone());
                        for x in v.iter_mut() {
                            *x = (*x - med).abs();
                        }
                        1.4826 * rtcollect::ob5m::np_median(&mut v)
                    })
                    .collect();
                if mad.iter().all(|m| *m > 1e-9) {
                    let e = out.entry((w, h)).or_default();
                    for (k, m) in ["mid", "whale", "retail"].iter().zip(&mad) {
                        e.insert(format!("{k}{suf}"), json!(m));
                    }
                }
            }
        }
    }
    out
}

/// micro_ref.sum_venue_bars(220~227): 모든 거래소에 있는 봉만 더한다(파이썬 sum = 0 에서 차례로).
pub fn sum_venue_bars(per: &[&[(i64, f64, f64, f64)]]) -> Vec<(i64, f64, f64, f64)> {
    if per.is_empty() {
        return vec![];
    }
    let maps: Vec<BTreeMap<i64, [f64; 3]>> = per.iter().map(|rows| rows.iter().map(|r| (r.0, [r.1, r.2, r.3])).collect()).collect();
    let mut common: BTreeSet<i64> = maps[0].keys().copied().collect();
    for m in &maps[1..] {
        common.retain(|b| m.contains_key(b));
    }
    common.into_iter().map(|b| {
        let s = |j: usize| maps.iter().fold(0.0, |a, m| a + m[&b][j]);
        (b, s(0), s(1), s(2))
    }).collect()
}

/// flow_hour_scales·_combos 의 SQL(250~259 · 236~238): 오늘 0시(UTC) 전 `days` 일 · since_ts 앞은 안 씀.
pub fn tape_flow_rows(db: &Path, symbol: &str, now_s: i64, days: i64, since_ts: i64) -> rusqlite::Result<Vec<(i64, f64, f64, f64)>> {
    let today = now_s.div_euclid(86400) * 86400;
    let c = ro(db)?;
    let mut q = c.prepare(
        "SELECT ts_sec - ts_sec % 300 AS b, sum(buy_qty - sell_qty), sum(whale_buy_qty - whale_sell_qty), \
         sum(retail_buy_qty - retail_sell_qty) FROM trade_tape_1s WHERE symbol = ? AND whale_buy_qty IS NOT NULL \
         AND ts_sec >= ? AND ts_sec < ? GROUP BY 1 ORDER BY 1",
    )?;
    let rows = q.query_map(rusqlite::params![symbol, (today - days * 86400).max(since_ts), today], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?;
    rows.collect()
}

/// micro_ref.flow_hour_scales_combos(230~247): 거래소마다 한 번 읽고 조합마다 기준선. 없는 파일·실패한 거래소는 빈 행(그 조합만 빈다).
pub fn flow_hour_scales_combos(srcs: &[(&str, &Path, &str)], now_s: i64, days: i64, since_ts: i64) -> HashMap<String, Scales> {
    let mut rows: HashMap<&str, Vec<(i64, f64, f64, f64)>> = HashMap::new();
    for &(k, p, s) in srcs {
        if VENUE_SETS.iter().any(|c| c.contains(&k)) {
            let r = if p.exists() { tape_flow_rows(p, s, now_s, days, since_ts) } else { Ok(vec![]) };
            rows.insert(k, r.unwrap_or_else(|e| {
                println!("flow_hour_scales_combos {}: {e}", p.display());
                vec![]
            }));
        }
    }
    VENUE_SETS
        .iter()
        .map(|c| {
            let per: Vec<&[(i64, f64, f64, f64)]> = c.iter().map(|k| rows.get(k).map_or(&[][..], |v| &v[..])).collect();
            (c.join(","), flow_hour_scales_from_bars(&sum_venue_bars(&per)))
        })
        .collect()
}

/// 붙이는 쪽(hub)이 원본 micro 틱마다(1초) **ETH 에 대해** 부르는 하나 -- server.py compute_situation 2478~2517 의 «받은 값 → 본문» 몫.
/// `core` = 코어가 마지막으로 민 Situation(직렬화 값 · 아직 없으면 None → 원본 «계산 전») · `veto`·`obi`·`persist` = 통과 칸(받는 쪽 원천) ·
/// `x_in` = x 재료(코어 Micro.flow 를 쌓은 nets · Bybit 봉 · 기준선 · 5분봉 하루 · card30 · 미시 payload · imb40 링) · `vol_mult_by` = 코인별 권장 배수.
/// 원본처럼 computed_at = `x_in.now`(코어는 값이 바뀔 때만 밀므로 시계는 부르는 쪽 것) · read 는 ETH 만(2489~2491).
/// 돌려줌: (/api/situation 본문, SSE 한 줄, 이 회차 x -- 시장 맥락이 같은 값을 읽는다 2488) · 장부는 `log` 가 있으면 한 줄(READONLY 면 안 씀).
#[allow(clippy::too_many_arguments)]
pub fn situation_tick(
    core: Option<&V>,
    veto: i64,
    obi: Option<f64>,
    persist: Option<f64>,
    streams: &rtcore::proto::Streams,
    x_in: &XIn,
    st: &mut FlowState,
    vol_mult_by: &[(String, V)],
    log: Option<(&mut crate::ctx::FusedLog, &V, &V)>,
) -> (V, String, Map<String, V>) {
    use crate::ctx;
    let now = x_in.now;
    let (now_v, computed) = match core {
        Some(c) => {
            let mut n = ctx::patch_now(c, veto, obi, persist);
            n["computed_at"] = json!(now);
            (n, json!(now))
        }
        None => (ctx::initial_now(), json!(0.0)),
    };
    let x = flow_read_ctx(x_in, st);
    let ev = now_v.get("evidence").filter(|e| e.as_object().is_some_and(|o| !o.is_empty())).cloned().unwrap_or(json!({}));
    let read = if core.is_some() { ctx::read(&ev, &V::Object(x.clone())) } else { V::Null };
    if let (Some((fl, card_p, feat_bar)), true) = (log, core.is_some()) {
        let mid = x_in.micro.as_object().filter(|m| m.get("available").is_some_and(py_truthy)).and_then(|m| m.get("mid")).cloned().unwrap_or(V::Null);
        fl.tick(now, &read, &mid, card_p, feat_bar);
    }
    let (fo, mp) = ctx::streams_payload(streams);
    let vm = vol_mult_by.iter().find(|(a, _)| a == "eth").map_or(V::Null, |(_, v)| v.clone());
    let payload = ctx::situation_payload(&now_v, &computed, &fo, &mp, &read, &vm, vol_mult_by);
    let sse = ctx::sse_situation(&payload);
    (payload, sse, x)
}
