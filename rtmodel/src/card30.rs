//! 30분 카드 방향·닿음 확률 (2026-11-21 · R5). 원본: dashboard/flow_read.py `card30_features`(26피쳐) + dashboard/server.py `_card30_pred`
//! (봉마다 한 번 · ETH 모델을 모든 코인에 · 보정 = lib::card30). 서빙: compute_situation → 시장 맥락 ctx 의 dir_p·reach_p → 융합 카드·융합 장부.
//! 순서 규칙: .mean()·.sum() = numpy 짝 합 · std(ddof=1) = 짝 합 평균 → 제곱 짝 합 · 행 = float32.
use crate::box_hold::npsum;
use crate::{card30, Model};
use serde_json::Value;

/// 완결 5분봉 한 개(time = 봉 시작 초).
#[derive(Clone, Copy)]
pub struct Bar {
    pub time: i64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub taker: f64,
}

pub struct Head {
    cols: Vec<String>,
    k: f64,
    base: f64,
    models: Vec<Model>,
}
/// 방향(dir = 닿는다면 위 먼저) · 닿음(reach = 30분 안 한쪽에 닿음).
pub struct Card30 {
    dir: Head,
    reach: Head,
}

fn bit(v: &Value) -> Result<f64, String> {
    v.as_u64().map(f64::from_bits).ok_or_else(|| "비트".to_string())
}

impl Card30 {
    /// tools/export_card30.py 가 쓴 card30.json.
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let head = |h: &Value| -> Result<Head, String> {
            Ok(Head { cols: serde_json::from_value(h["cols"].clone()).map_err(|e| e.to_string())?, k: bit(&h["k"])?, base: bit(&h["base"])?,
                      models: serde_json::from_value(h["models"].clone()).map_err(|e| e.to_string())? })
        };
        Ok(Card30 { dir: head(&v["dir"])?, reach: head(&v["reach"])? })
    }
}

fn mean(x: &[f64]) -> f64 {
    npsum(x) / x.len() as f64
}

/// card30_features: 봉 293개 미만이거나 5분 간격이 끊기면 None. 반환 (이름, 값) 26개(원본 dict 순서).
pub fn features(bars: &[Bar]) -> Option<Vec<(String, f64)>> {
    if bars.len() < 293 {
        return None;
    }
    let b = &bars[bars.len() - 293..];
    if b.windows(2).any(|w| w[1].time - w[0].time != 300) {
        return None;
    }
    let col = |f: fn(&Bar) -> f64| b.iter().map(f).collect::<Vec<f64>>();
    let (h, lo, c, v, tb) = (col(|x| x.high), col(|x| x.low), col(|x| x.close), col(|x| x.volume), col(|x| x.taker));
    let n = c.len();
    let tail = |x: &[f64], k: usize| x[x.len() - k..].to_vec();
    let mut f: Vec<(String, f64)> = vec![];
    let lr: Vec<f64> = c.iter().map(|x| x.ln()).collect();
    for k in [1, 3, 6, 12, 24, 48, 144, 288] {
        f.push((format!("ret{k}"), (lr[n - 1] - lr[n - 1 - k]) * 1e4));
    }
    for w in [6, 24, 144, 288] {
        let hh = tail(&h, w).into_iter().fold(f64::NEG_INFINITY, f64::max);
        let ll = tail(&lo, w).into_iter().fold(f64::INFINITY, f64::min);
        f.push((format!("pos{w}"), (c[n - 1] - ll) / (hh - ll).max(1e-9)));
    }
    let rng: Vec<f64> = (0..n).map(|i| h[i] - lo[i]).collect();
    let atr = mean(&tail(&rng, 144));
    for w in [48, 144, 288] {
        f.push((format!("sma{w}"), (c[n - 1] - mean(&tail(&c, w))) / atr));
    }
    let imb: Vec<f64> = (0..n).map(|i| 2.0 * tb[i] / v[i].max(1e-9) - 1.0).collect();
    let iv: Vec<f64> = (0..n).map(|i| imb[i] * v[i]).collect();
    for k in [1, 6, 12, 48] {
        f.push((format!("imb{k}"), npsum(&tail(&iv, k)) / npsum(&tail(&v, k)).max(1e-9)));
    }
    let rg: Vec<f64> = (5..n).map(|i| {
        let hi = h[i - 5..=i].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let l = lo[i - 5..=i].iter().copied().fold(f64::INFINITY, f64::min);
        (hi - l) / c[i] * 1e4
    }).collect();
    let last = *rg.last().unwrap();
    f.push(("rg30".into(), last));
    let (lt, eq) = (rg.iter().filter(|x| **x < last).count(), rg.iter().filter(|x| **x == last).count());
    f.push(("rgq".into(), (lt as f64 + (eq + 1) as f64 / 2.0) / rg.len() as f64));
    let v288 = tail(&v, 288);
    let m = mean(&v288);
    let sd = (npsum(&v288.iter().map(|x| (x - m) * (x - m)).collect::<Vec<_>>()) / 287.0).sqrt();
    f.push(("volz".into(), (v[n - 1] - m) / sd));
    let (pc, hl) = (c[n - 2], (h[n - 1] - lo[n - 1]).max(1e-9));
    f.push(("last_body".into(), (c[n - 1] - pc) / hl));
    f.push(("wick".into(), ((h[n - 1] - c[n - 1].max(pc)) - (c[n - 1].min(pc) - lo[n - 1])) / hl));
    let t = b[n - 1].time;
    f.push(("hour".into(), (t / 3600 % 24) as f64));
    f.push(("dow".into(), ((t.div_euclid(86_400) + 3) % 7) as f64));
    Some(f)
}

/// _card30_pred: (dir, reach) 보정 확률 · 피쳐가 없으면 None(카드는 표로 물러선다).
pub fn predict(m: &Card30, bars: &[Bar]) -> Option<(f64, f64)> {
    let f = features(bars)?;
    let p = |hd: &Head| {
        let x: Vec<f64> = hd.cols.iter().map(|c| f.iter().find(|(k, _)| k == c).map_or(f64::NAN, |x| x.1)).collect();
        card30(&hd.models.iter().collect::<Vec<_>>(), &x, hd.k, hd.base)
    };
    Some((p(&m.dir), p(&m.reach)))
}
