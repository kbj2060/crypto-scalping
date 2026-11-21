//! 30분 상황 근거(2026-11-21 옮김) -- 파이썬 서버 `compute_situation` 의 «입력 접기 + classify» 몫. tokio 비의존.
//!
//! 원본: dashboard/server.py `_situation_inputs`(완결 5분봉 접기 · BTC 같은 창 · 마크 링) · `compute_situation` 의 prev_dir 잇기 ·
//!       dashboard/situation.py `classify`(슈미트 30분 방향·거부 봉·청산·펀딩/베이시스·BTC) · dashboard/micro_ref.py `deriv_from_ring`
//!       · server.py `collect_mark_price` 의 링(초 → (마크, 지수, 펀딩) · 10초마다 7,200초 밖 지움).
//! 일치 정답지: rtcore/tools/gen_situation_fixture.py(원본 `_situation_inputs` 를 server.py 에서 떠서 그대로 돌린다) → tests/situation_parity.rs.
//!
//! 입력을 어디서 받나(원본 → 코어 · 표 전체는 SITUATION.md):
//! - 봉 칸(델타·고저 대체값) = 풋프린트 봉(같은 원천 · footprint.rs) · 종가 대체 = 1초 링 마지막 체결가 → 없으면 봉 VWAP(원본 그대로).
//! - 캔들(고가·저가·종가): 원본은 REST 5분 klines(60초 캐시). 코어는 받은 `@trade` 로 봉마다 고가·저가·마지막 체결가를 센다
//!   (klines 가 같은 체결로 만든 값 -- 체결을 빠뜨리지 않았으면 같다). 복원(restore.rs)으로 테이프에서 메운 봉은 캔들이 없어
//!   원본의 «캔들 없음» 가지(풋프린트 칸·링·VWAP)로 간다. 원본은 캐시가 60초 늦어 방금 닫힌 봉이 그 가지로 가는데, 코어는 곧바로 캔들이 있다(일부러).
//! - BTC 같은 창: 원본은 REST BTC 캔들 · 코어는 btcusdt `@trade` 로 센 같은 캔들(btcusdt 를 받을 때만 · 아니면 원본의 «BTC 없음»).
//! - 청산: 원본은 봇 DB 5분 집계(compute_liquidation_5m_history) · 코어는 바이낸스 `@forceOrder` 5분 누적(supply.rs liq5m -- 같은 사건).
//! - 마크 링: `@markPrice@1s`(코어가 이미 받는 Mark 사건).
//! - 못 채움(`missing`): veto(12h 추세 · 5분봉 720개 워밍업 -- REST klines) · book(호가 래스터 파일) · oi_delta(classify 가 안 씀).
use crate::proto::{Evidence, Regime, Situation};
use std::collections::BTreeMap;

pub const WINDOW: usize = 6;
pub const TREND_ENTER: f64 = 0.45;
pub const TREND_EXIT: f64 = 0.25;
pub const REJECT_FRAC: f64 = 0.5;
pub const FUNDING_NEUTRAL: f64 = 0.0001;
pub const BASIS_PCT: f64 = 0.75;
pub const BAR_S: i64 = 300;
/// server.py MARK_PRICE_RING_S · 링 정리 간격 OI_1S_FLUSH_SECONDS.
pub const MARK_RING_S: f64 = 7200.0;
pub const MARK_TRIM_S: f64 = 10.0;
/// 코어가 못 채우는 classify 입력(원본은 채운다).
pub const MISSING: [&str; 2] = ["veto", "book"];

/// 완결 5분봉 하나(classify 입력).
#[derive(Clone, Debug, PartialEq)]
pub struct Bar {
    pub time: i64,
    pub high: Option<f64>,
    pub low: Option<f64>,
    pub close: Option<f64>,
    pub delta: f64,
    pub liq_long: f64,
    pub liq_short: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Deriv {
    pub funding: Option<f64>,
    pub basis_bp: Option<f64>,
    pub basis_d_bp: Option<f64>,
    pub basis_thr_bp: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Inputs {
    pub bars: Vec<Bar>,
    pub mid: Option<f64>,
    pub deriv: Deriv,
    /// (move_bp, range_bp)
    pub btc: Option<(f64, f64)>,
}

/// collect_mark_price 의 링: 초 → (마크, 지수, 펀딩). 정리는 원본처럼 «지난 정리 뒤 10초가 지났을 때» 벽시계 기준으로만.
pub struct MarkRing {
    pub ring: BTreeMap<i64, (f64, f64, f64)>,
    trimmed_at: f64,
}

impl MarkRing {
    /// `now_s` = 링을 만든 벽시계(원본 = 수신 작업 시작 시각).
    pub fn new(now_s: f64) -> MarkRing {
        MarkRing { ring: BTreeMap::new(), trimmed_at: now_s }
    }

    pub fn push(&mut self, ex_ms: i64, v: (f64, f64, f64), now_s: f64) {
        self.ring.insert(ex_ms.div_euclid(1000), v);
        if now_s - self.trimmed_at >= MARK_TRIM_S {
            self.ring.retain(|&s, _| s as f64 >= now_s - MARK_RING_S);
            self.trimmed_at = now_s;
        }
    }
}

/// micro_ref.deriv_from_ring(링, 지평, 분위, step_s=30, min_samples=60).
pub fn deriv_from_ring(ring: &BTreeMap<i64, (f64, f64, f64)>, horizon_s: i64, pct: f64) -> Deriv {
    let mut out = Deriv::default();
    let Some((&last, lv)) = ring.last_key_value() else { return out };
    let bas = |s: i64| {
        let r = ring[&s];
        (r.0 - r.1) / r.1 * 1e4
    };
    out.funding = Some(lv.2);
    out.basis_bp = Some(bas(last));
    if let Some((&prev, _)) = ring.range(..=last - horizon_s).next_back() {
        out.basis_d_bp = Some(bas(last) - bas(prev));
    }
    let mut ds: Vec<f64> = ring.keys().step_by(30).filter(|&&s| ring.contains_key(&(s - horizon_s))).map(|&s| (bas(s) - bas(s - horizon_s)).abs()).collect();
    if ds.len() >= 60 {
        ds.sort_by(f64::total_cmp);
        out.basis_thr_bp = Some(ds[(ds.len() - 1).min((ds.len() as f64 * pct) as usize)]);
    }
    out
}

/// server.py `_situation_inputs`. `fbars` = 풋프린트 봉(시각 오름차순) → 칸(가격 키 → [매수, 매도, …]) · `sec` = 1초 링(c[6] = 마지막 체결가)
/// · `candles`·`btc` = 봉 → [고가, 저가, 종가] · `liq` = 봉 → [롱USD, 숏USD] · `marks` = 마크 링 · `mid` = 최우선 호가 중간값.
#[allow(clippy::too_many_arguments)]
pub fn inputs<'a>(
    now_s: f64,
    bucket: f64,
    fbars: impl Iterator<Item = (i64, &'a BTreeMap<i64, [f64; 6]>)>,
    sec: &BTreeMap<i64, [f64; 7]>,
    candles: &BTreeMap<i64, [f64; 3]>,
    btc: Option<&BTreeMap<i64, [f64; 3]>>,
    liq: &BTreeMap<i64, [f64; 2]>,
    marks: &BTreeMap<i64, (f64, f64, f64)>,
    mid: Option<f64>,
) -> Inputs {
    let bar_start = (now_s as i64).div_euclid(BAR_S) * BAR_S;
    let mut want: Vec<(i64, &BTreeMap<i64, [f64; 6]>)> = fbars.filter(|(b, _)| *b < bar_start).collect();
    want.drain(..want.len().saturating_sub(2 * WINDOW + 1));
    let mut bars = Vec::with_capacity(want.len());
    for &(b, cells) in &want {
        // ponytail: 칸 합은 가격 키 순서(원본은 칸이 생긴 순서) -- 마지막 비트가 갈려도 cvd 는 정수로 반올림, 거부 봉 경계만 1ulp 안에서 갈린다.
        let buy: f64 = cells.values().map(|v| v[0]).sum();
        let sell: f64 = cells.values().map(|v| v[1]).sum();
        let c = candles.get(&b);
        let prices = || cells.keys().map(|&k| k as f64 * bucket);
        let mut close = c.map(|c| c[2]);
        if close.is_none() {
            close = match sec.range(b..b + BAR_S).filter(|(_, v)| v[6] != 0.0).next_back() {
                Some((_, v)) => Some(v[6]),
                None => {
                    let tot: f64 = cells.values().map(|v| v[0] + v[1]).sum();
                    (tot > 0.0).then(|| cells.iter().map(|(&k, v)| k as f64 * bucket * (v[0] + v[1])).sum::<f64>() / tot)
                }
            };
        }
        let l = liq.get(&b).copied().unwrap_or([0.0, 0.0]);
        bars.push(Bar {
            time: b,
            high: c.map(|c| c[0]).or_else(|| prices().reduce(f64::max)),
            low: c.map(|c| c[1]).or_else(|| prices().reduce(f64::min)),
            close,
            delta: buy - sell,
            liq_long: l[0],
            liq_short: l[1],
        });
    }
    let btc = btc.and_then(|bc| {
        if want.len() < WINDOW + 1 {
            return None;
        }
        let (c0, c1) = (bc.get(&want[want.len() - WINDOW - 1].0)?, bc.get(&want[want.len() - 1].0)?);
        let span: Vec<&[f64; 3]> = want[want.len() - WINDOW..].iter().filter_map(|(t, _)| bc.get(t)).collect();
        let hi = span.iter().map(|c| c[0]).reduce(f64::max)?;
        let lo = span.iter().map(|c| c[1]).reduce(f64::min)?;
        Some(((c1[2] - c0[2]) / c0[2] * 1e4, (hi - lo) / c1[2] * 1e4))
    });
    Inputs { bars, mid, deriv: deriv_from_ring(marks, WINDOW as i64 * BAR_S, BASIS_PCT), btc }
}

fn sign(x: f64) -> i64 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

/// 파이썬 round(x, n) -- supply.rs r3 와 같은 방식(정확한 십진 전개 · 동률 짝수).
fn rn(x: f64, n: usize) -> f64 {
    format!("{x:.n$}").parse().unwrap_or(x)
}

/// 파이썬 round(x) -> int.
fn ri(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// situation.classify. `prev_dir` = 직전 결과의 dir(없거나 ok=false 면 0 -- compute_situation 의 `.get("dir") or 0`).
/// `computed_at` 은 부르는 쪽이 채운다.
pub fn classify(inp: &Inputs, prev_dir: i64) -> Situation {
    let bars: Vec<&Bar> = inp.bars.iter().filter(|b| b.close.is_some_and(|c| c != 0.0) && b.high.is_some() && b.low.is_some()).collect();
    let miss = MISSING.iter().map(|s| s.to_string()).collect();
    if bars.len() < WINDOW + 1 {
        return Situation { ok: false, reason: Some(format!("완결 봉 {} < {}", bars.len(), WINDOW + 1)), dir: None, labels: vec![], evidence: None, regime: None, computed_at: 0.0, missing: miss };
    }
    let w = &bars[bars.len() - WINDOW..];
    let cl = |b: &Bar| b.close.unwrap();
    let last_c = cl(w[WINDOW - 1]);
    let base = cl(bars[bars.len() - WINDOW - 1]);
    let mid = inp.mid.filter(|&m| m != 0.0).unwrap_or(last_c);

    let hi = w.iter().map(|b| b.high.unwrap()).reduce(f64::max).unwrap();
    let lo = w.iter().map(|b| b.low.unwrap()).reduce(f64::min).unwrap();
    let rng_bp = (hi - lo) / mid * 1e4;
    let move_bp = (last_c - base) / base * 1e4;
    let ratio = if rng_bp > 0.0 { move_bp.abs() / rng_bp } else { 0.0 };
    let sgn = sign(move_bp);
    let d = if prev_dir == 0 {
        if ratio > TREND_ENTER { sgn } else { 0 }
    } else if sgn == prev_dir {
        if ratio < TREND_EXIT { 0 } else { prev_dir }
    } else if ratio > TREND_ENTER {
        sgn
    } else if ratio < TREND_EXIT {
        0
    } else {
        prev_dir
    };
    let along = d != 0 && sgn == d;
    let thr_next = if along { TREND_EXIT } else { TREND_ENTER };
    let margin = if along { ratio - thr_next } else { thr_next - ratio };
    let head = match d {
        1 => format!("상승 {move_bp:+.0}bp"),
        -1 => format!("하락 {move_bp:+.0}bp"),
        _ => format!("횡보 (±{:.0}bp 안)", thr_next * rng_bp),
    };

    let deltas: Vec<f64> = w.iter().map(|b| b.delta).collect();
    let maxd = deltas.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let last = deltas[WINDOW - 1];
    let dv = &inp.deriv;
    let crowd = match dv.funding {
        Some(fr) if (fr - FUNDING_NEUTRAL).abs() >= FUNDING_NEUTRAL => sign(fr - FUNDING_NEUTRAL),
        _ => 0,
    };
    let lead = match (dv.basis_d_bp, dv.basis_thr_bp) {
        (Some(bd), Some(bthr)) if d != 0 && bd.abs() >= bthr => {
            if sign(bd) == d { 1 } else { -1 }
        }
        _ => 0,
    };
    let d_btc = inp.btc.map(|(bm, brg)| if bm > 0.35 * brg { 1 } else if bm < -0.35 * brg { -1 } else { 0 });
    let ev = Evidence {
        mid,
        move_bp: rn(move_bp, 1),
        range_bp: rn(rng_bp, 1),
        dir: d,
        move_ratio: rn(ratio, 3),
        cvd: ri(deltas.iter().sum()),
        reject: d != 0 && sign(last) == -d && maxd > 0.0 && last.abs() >= REJECT_FRAC * maxd,
        liq_long: ri(w.iter().map(|b| b.liq_long).sum()),
        liq_short: ri(w.iter().map(|b| b.liq_short).sum()),
        veto: 0,
        obi: None,
        persist: None,
        funding: dv.funding,
        basis_bp: dv.basis_bp,
        basis_d_bp: dv.basis_d_bp,
        basis_thr_bp: dv.basis_thr_bp,
        crowd,
        lead,
        btc_move_bp: inp.btc.map(|b| rn(b.0, 1)),
        btc_rel: match d_btc {
            Some(x) if d != 0 => Some(if x == d { "동행" } else { "단독" }.to_string()),
            _ => None,
        },
    };
    Situation {
        ok: true,
        reason: None,
        dir: Some(d),
        labels: vec![head],
        evidence: Some(ev),
        regime: Some(Regime { ratio: rn(ratio, 3), thr: thr_next, margin: rn(margin, 3), enter: TREND_ENTER, exit: TREND_EXIT }),
        computed_at: 0.0,
        missing: miss,
    }
}

/// 미시 값(server.py collect_micro_ref payload 의 mid·spread·qi·vol60·liq60 · 2026-11-21). `book` = (매수호가, 수량, 매도호가, 수량) ·
/// `liq` = 바이낸스 강제청산 (거래소 ms, 롱 청산, USD) 받은 차례대로 · `okx_oi5m` = OKX OI 5분봉 [처음, 끝] · `from` = 실을 첫 봉(None = 안 실음).
pub fn micro(
    now_s: f64,
    book: Option<(f64, f64, f64, f64)>,
    sec: &BTreeMap<i64, [f64; 7]>,
    liq: impl Iterator<Item = (i64, bool, f64)>,
    okx_oi5m: &BTreeMap<i64, [f64; 2]>,
    from: Option<i64>,
) -> crate::proto::Micro {
    let now_sec = now_s as i64;
    // ponytail: 초 순서로 더한다(원본 dict 는 넣은 차례 -- 늦은 체결이 새 초를 만들면 끝에 온다) -- 마지막 비트만 갈린다. 같게 하려면 1초 링에 넣은 차례를 들어야 한다.
    // 원본 합은 0(정수)에서 시작한다 -- f64 Sum 은 -0.0 에서 시작해 빈 합이 -0.0 이 된다
    let vol60 = sec.range(now_sec - 60..now_sec).map(|(_, c)| c[4] + c[5]).fold(0.0, |a, x| a + x);
    let cut = now_s * 1000.0 - 60_000.0;
    let mut l = crate::proto::Liq60 { long: 0.0, short: 0.0, n: 0 };
    for (_, long, usd) in liq.filter(|e| e.0 as f64 >= cut) {
        if long { l.long += usd } else { l.short += usd }
        l.n += 1;
    }
    crate::proto::Micro {
        ts: now_sec,
        mid: book.map(|b| (b.0 + b.2) / 2.0),
        spread: book.map(|b| b.2 - b.0),
        qi: book.map(|b| {
            let s = b.1 + b.3;
            if s > 0.0 { (b.1 - b.3) / s } else { 0.0 }
        }),
        vol60,
        liq60: l,
        okx_oi5m: from.map_or_else(Vec::new, |f| okx_oi_rows(okx_oi5m, f)),
        okx_from: None,
        okx_bars: vec![],
        okx_full: false,
        streams: None,
        flow: None,
    }
}

/// server.py api_oi_5m 의 okx_d: 봉 Δ = 직전 봉 끝 → 이 봉 끝, 앞 봉이 없거나 300초 넘게 떨어졌으면 봉 안(끝 − 처음). `from` 부터.
pub fn okx_oi_rows(m: &BTreeMap<i64, [f64; 2]>, from: i64) -> Vec<(i64, f64, f64)> {
    let mut prev = m.range(..from).next_back().map(|(&b, v)| (b, v[1]));
    m.range(from..)
        .map(|(&ob, v)| {
            let d = match prev {
                Some((pb, pc)) if ob - pb <= BAR_S => v[1] - pc,
                _ => v[1] - v[0],
            };
            prev = Some((ob, v[1]));
            (ob, d, v[1])
        })
        .collect()
}

/// 관계 읽기 재료의 풋프린트 몫(server.py _flow_read_ctx 2295~2303 · 2332~2339 · 2374~2376 -- Read 로 본 줄). 반올림 전.
/// `bn` = 원본 fr_bar_net [Σ(v0−v1), Σ(v2−v3), Σ(v4−v5)] · `bn_last` = 마지막 닫힌 봉의 [Σv0, Σv1](테이커 비율 2368~2369 -- 원본은 그 회차 칸을 그 자리에서 더한다:
/// 굳은 봉이면 fr_bar_net 과 다른 때의 값이다) · `okx` = 원본 okx_bar_net_by[asset](onet 5칸) · `okx30` = 마지막 여섯 봉 OKX 칸 한 줄 합.
#[derive(Default, Clone, Debug, PartialEq)]
pub struct FlowNets {
    pub bn: BTreeMap<i64, [f64; 3]>,
    pub bn_last: Option<(i64, [f64; 2])>,
    pub okx: BTreeMap<i64, [f64; 5]>,
    pub okx30: Option<f64>,
}

fn sums(cells: &[[f64; 6]]) -> [f64; 5] {
    let s = |f: &dyn Fn(&[f64; 6]) -> f64| cells.iter().fold(0.0, |a, v| a + f(v)); // 파이썬 sum = 0 에서 차례로
    [s(&|v| v[0] - v[1]), s(&|v| v[2] - v[3]), s(&|v| v[4] - v[5]), s(&|v| v[0]), s(&|v| v[1])]
}

/// 한 회차(원본은 compute_situation 마다 = 1초). `fbars`·`okx_bars` = 봉 → 칸(원본 dict 차례 · footprint::Bar::in_order).
/// 원본 규칙: 닫힌 봉은 처음 볼 때와 «최근 두 봉»일 때만 센다 · 풋프린트에서 사라진 봉은 지운다 ·
/// OKX 는 닫힌 봉 12개 이상일 때 격자(첫 닫힌 봉 ~ 마지막 닫힌 봉) 안에서 first_bar «뒤» 봉만 · OKX 봉에서 사라진 봉은 지운다.
/// `cells` = 봉 하나의 칸을 원본 dict 차례로 꺼내는 함수(코어 = footprint::Bar::in_order · 시험 = 정답지 차례 그대로).
pub fn flow_nets_tick<B>(n: &mut FlowNets, now_s: f64, fbars: &BTreeMap<i64, B>, okx_first: i64, okx_bars: &BTreeMap<i64, B>, cells: impl Fn(&B) -> Vec<[f64; 6]>) {
    let bar_start = (now_s as i64).div_euclid(BAR_S) * BAR_S;
    let done: Vec<i64> = fbars.keys().copied().filter(|&b| b < bar_start).collect();
    for &b in &done {
        if !n.bn.contains_key(&b) || b >= bar_start - 2 * BAR_S {
            let v = sums(&cells(&fbars[&b]));
            n.bn.insert(b, [v[0], v[1], v[2]]);
        }
    }
    n.bn.retain(|b, _| fbars.contains_key(b));
    n.bn_last = done.last().map(|&b| {
        let v = sums(&cells(&fbars[&b]));
        (b, [v[3], v[4]])
    });
    if done.len() >= 12 {
        let (lo, hi) = (done[0], done[done.len() - 1]);
        for b in (lo..=hi).step_by(BAR_S as usize) {
            if okx_first != 0 && b > okx_first && okx_bars.contains_key(&b) && (!n.okx.contains_key(&b) || b >= bar_start - 2 * BAR_S) {
                n.okx.insert(b, sums(&cells(&okx_bars[&b])));
            }
        }
        n.okx.retain(|b, _| okx_bars.contains_key(b));
    }
    let last6 = &done[done.len().saturating_sub(6)..];
    n.okx30 = (okx_first != 0 && last6.len() == 6 && last6.iter().all(|b| *b > okx_first && okx_bars.contains_key(b)))
        .then(|| last6.iter().flat_map(|b| cells(&okx_bars[b])).fold(0.0, |s, v| s + (v[0] - v[1])));
}

/// 보낸 것(`sent`, None = 아직 안 보냄 → 통째로) 대비 바뀐 몫.
pub fn flow_delta(now: &FlowNets, sent: Option<&FlowNets>) -> crate::proto::FlowSums {
    fn diff<const N: usize>(a: &BTreeMap<i64, [f64; N]>, b: Option<&BTreeMap<i64, [f64; N]>>) -> (Vec<(i64, [f64; N])>, Vec<i64>) {
        let rows = a.iter().filter(|(k, v)| b.is_none_or(|b| b.get(k).is_none_or(|o| o.map(f64::to_bits) != v.map(f64::to_bits)))).map(|(k, v)| (*k, *v)).collect();
        let drop = b.map_or(vec![], |b| b.keys().filter(|k| !a.contains_key(k)).copied().collect());
        (rows, drop)
    }
    let (bn, bn_drop) = diff(&now.bn, sent.map(|s| &s.bn));
    let (okx, okx_drop) = diff(&now.okx, sent.map(|s| &s.okx));
    crate::proto::FlowSums { bn, bn_drop, bn_last: now.bn_last, okx, okx_drop, okx30: now.okx30, full: sent.is_none() }
}
