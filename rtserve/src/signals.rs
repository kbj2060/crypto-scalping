//! 청산맵 차트 마커(2026-11-28): /api/chart-markers -- 바깥 요청은 model::binance_get 한 문(RTSERVE_REST 있을 때만).
//! 원본(Read 로 읽은 줄): server.py load_chart_markers 3213~3227(swr "chart_markers" 60초 · 묵음 600) · api_chart_markers 4710~4712 ·
//!   scripts/live_eth_chart_markers_20260909.py compute_chart_markers 64~129 · _onset_only 40~61(격자에 events 가 없어 늘 []) ·
//!   live_eth_extreme_detector_20260909.py _fetch 168~192 · build_eth_anchor_label_dataset_20260907.py SIGNALS·ABBR 59~63 ·
//!   live_evidence_signal_dashboard_20260823.py compute_signals 363~743 중 마커가 읽는 8종의 날 발동 열(funding_df=None)만 --
//!   _fill·_active·votes(표시층)·volume_wick·dalton 은 마커에 안 쓰여 안 옮김.
//! pandas 규칙: rolling mean = ctx_rest::rolling_mean · std = sr_lean::roll_std · max/min = sr_lean::roll_ext · rank(pct) = (작은 수 + (같은 수 + 1)/2) / 개수.
//! 시험: tests/routes7_parity.rs(정답지 = tools/gen_markers_fixture.py · 원본 모듈을 그대로 불러 requests 자리에 녹화 응답).

use std::sync::Arc;

use rtmodel::sr_lean::{roll_ext, roll_std};
use serde_json::{json, Value as V};

use crate::ctx_rest::rolling_mean;
use crate::{Config, Handler, Req, Resp};

const CHART_BARS: usize = 72;
const SIGNALS: [(&str, &str); 8] = [("taker_delta_z_climax", "taker"), ("short_term_return_z", "strz"), ("liquidity_sweep", "sweep"),
    ("orthogonal_combo", "orth"), ("smt_divergence", "smt"), ("fib_extension_exhaustion", "fib"), ("demarker_extreme", "dem"),
    ("kalman_deviation_meanrev", "kal")];
const NAN: f64 = f64::NAN;

pub fn routes(p: &str) -> Vec<(&'static str, Handler)> {
    match p {
        "/api/chart-markers" => vec![("GET", api_chart_markers)],
        _ => Vec::new(),
    }
}

#[derive(Clone, Copy)]
pub struct Bar {
    pub t: i64,
    pub o: f64,
    pub h: f64,
    pub l: f64,
    pub c: f64,
    pub v: f64,
    pub tb: f64,
}

fn num(v: &V) -> Result<f64, String> {
    match v {
        V::String(s) => s.trim().parse().map_err(|_| format!("ValueError: could not convert string to float: '{s}'")),
        V::Number(n) => n.as_f64().ok_or_else(|| "ValueError".into()),
        _ => Err("TypeError: float() argument must be a string or a real number".into()),
    }
}

fn int(v: &V) -> Result<i64, String> {
    v.as_i64().or_else(|| v.as_f64().map(|x| x as i64)).ok_or_else(|| "TypeError: int".into())
}

/// _fetch: 1500봉 두 쪽(두 번째 = endTime 첫 봉 − 1) · 못 받으면(연결·4xx/5xx·JSON 아님) None · 형성 봉 버림 · 같은 시각은 처음 것.
/// Err = 원본이 try 밖에서 내는 예외(모양 틀림 -- 글자는 원본과 다를 수 있다).
fn fetch(cfg: &Config, sym: &str) -> Result<Option<Vec<Bar>>, String> {
    if cfg.rest.is_empty() {
        return Ok(None); // RestOff: 안 보냄(원본에 없는 갈래 -- 못 받음과 같게)
    }
    let mut frames: Vec<Vec<V>> = Vec::new();
    let mut end: Option<i64> = None;
    for _ in 0..2 {
        let mut url = format!("{}/fapi/v1/klines?symbol={sym}&interval=5m&limit=1500", cfg.rest);
        if let Some(e) = end {
            url += &format!("&endTime={e}");
        }
        let d: V = match crate::model::binance_get(cfg, &url) {
            Ok((s, body)) if s < 400 => match serde_json::from_str(&body) {
                Ok(d) => d,
                Err(_) => return Ok(None),
            },
            _ => return Ok(None),
        };
        if !crate::py::truthy(&d) {
            break;
        }
        let rows = d.as_array().ok_or("KeyError: 0")?.clone();
        end = Some(int(rows[0].get(0).ok_or("KeyError: 0")?)? - 1);
        frames.push(rows);
    }
    if frames.is_empty() {
        return Ok(None);
    }
    let now_ms = ((cfg.now)() * 1000.0) as i64;
    let (mut out, mut seen): (Vec<Bar>, std::collections::HashSet<i64>) = Default::default();
    for r in frames.iter().rev().flatten() {
        let r = r.as_array().filter(|r| r.len() == 12).ok_or("ValueError: 12 columns passed")?;
        if int(&r[6])? >= now_ms {
            continue;
        }
        let t = int(&r[0])?;
        if !seen.insert(t) {
            continue;
        }
        out.push(Bar { t, o: num(&r[1])?, h: num(&r[2])?, l: num(&r[3])?, c: num(&r[4])?, v: num(&r[5])?, tb: num(&r[9])? });
    }
    out.sort_by_key(|b| b.t);
    Ok(Some(out))
}

fn shift(x: &[f64], k: usize) -> Vec<f64> {
    (0..x.len()).map(|i| if i < k { NAN } else { x[i - k] }).collect()
}

/// rolling(w, min_periods=w).rank(pct=True): 지금 값이 NaN 이거나 창 안 값 개수 < w 면 NaN · 같은 값은 평균 순위
pub fn rank_pct(x: &[f64], w: usize) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            let win = &x[(i + 1).saturating_sub(w)..=i];
            let n = win.iter().filter(|v| !v.is_nan()).count();
            if x[i].is_nan() || n < w {
                return NAN;
            }
            let (less, eq) = (win.iter().filter(|&&v| v < x[i]).count(), win.iter().filter(|&&v| v == x[i]).count());
            (less as f64 + (eq + 1) as f64 / 2.0) / n as f64
        })
        .collect()
}

/// (x − rolling mean) / rolling std(0 → NaN)
fn zscore(x: &[f64], w: usize) -> Vec<f64> {
    let (m, s) = (rolling_mean(x, w), roll_std(x, w, w));
    (0..x.len()).map(|i| if s[i] == 0.0 { NAN } else { (x[i] - m[i]) / s[i] }).collect()
}

type M2 = [[f64; 2]; 2];
fn mm(a: &M2, b: &M2) -> M2 {
    let mut r = [[0.0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    r
}

/// kalman_deviation_meanrev 의 수준 x[0] (F=[[1,1],[0,1]] · H=[1,0] · Q=1e-5·I · R=1e-3) -- 원본 numpy 행렬 곱을 성분 식으로
fn kalman_levels(c: &[f64]) -> Vec<f64> {
    let f: M2 = [[1.0, 1.0], [0.0, 1.0]];
    let ft: M2 = [[1.0, 0.0], [1.0, 1.0]];
    let (mut x, mut p): ([f64; 2], M2) = ([c[0], 0.0], [[1.0, 0.0], [0.0, 1.0]]);
    let mut out = Vec::with_capacity(c.len());
    for &z in c {
        x = [f[0][0] * x[0] + f[0][1] * x[1], f[1][0] * x[0] + f[1][1] * x[1]];
        p = mm(&mm(&f, &p), &ft);
        p = [[p[0][0] + 1e-5, p[0][1] + 0.0], [p[1][0] + 0.0, p[1][1] + 1e-5]];
        let s = (1.0 * p[0][0] + 0.0 * p[1][0]) * 1.0 + (1.0 * p[0][1] + 0.0 * p[1][1]) * 0.0 + 1e-3;
        let k = [(p[0][0] * 1.0 + p[0][1] * 0.0) / s, (p[1][0] * 1.0 + p[1][1] * 0.0) / s];
        let inn = z - (1.0 * x[0] + 0.0 * x[1]);
        x = [x[0] + k[0] * inn, x[1] + k[1] * inn];
        let ikh: M2 = [[1.0 - k[0] * 1.0, 0.0 - k[0] * 0.0], [0.0 - k[1] * 1.0, 1.0 - k[1] * 0.0]];
        p = mm(&ikh, &p);
        out.push(x[0]);
    }
    out
}

/// compute_signals 의 8종 날 발동(바닥, 꼭대기) -- 차례 = SIGNALS
pub fn fires(k: &[Bar], btc: &[Bar]) -> Vec<(Vec<bool>, Vec<bool>)> {
    let n = k.len();
    let col = |g: fn(&Bar) -> f64| k.iter().map(g).collect::<Vec<f64>>();
    let (h, l, c, v, tb) = (col(|b| b.h), col(|b| b.l), col(|b| b.c), col(|b| b.v), col(|b| b.tb));
    // 스토캐스틱 %K · 느린 %K · 864봉 순위
    let (hh, ll) = (roll_ext(&h, 14, 14, true), roll_ext(&l, 14, 14, false));
    let fast_k: Vec<f64> = (0..n).map(|i| {
        let r = hh[i] - ll[i];
        let r = if r == 0.0 { NAN } else { r };
        100.0 + -100.0 * (hh[i] - c[i]) / r
    }).collect();
    let slow_k = rolling_mean(&fast_k, 3);
    let (pf, ps) = (rank_pct(&fast_k, 864), rank_pct(&slow_k, 864));
    let delta: Vec<f64> = (0..n).map(|i| 2.0 * tb[i] - v[i]).collect();
    let dz = zscore(&delta, 288);
    let c3 = shift(&c, 3);
    let ret3: Vec<f64> = (0..n).map(|i| c[i] / c3[i] - 1.0).collect();
    let rz = zscore(&ret3, 288);
    let (slp, shp) = (shift(&roll_ext(&l, 48, 48, false), 1), shift(&roll_ext(&h, 48, 48, true), 1));
    // fib: 직전 48봉 창에서 저점·고점 위치(첫 최소·첫 최대)
    let (mut lpos, mut hpos) = (vec![-1i64; n], vec![-1i64; n]);
    if n > 48 {
        for i in 48..n {
            let j = i - 48;
            let (mut a, mut b) = (j, j);
            for q in j..i {
                if l[q] < l[a] { a = q }
                if h[q] > h[b] { b = q }
            }
            (lpos[i], hpos[i]) = (a as i64, b as i64);
        }
    }
    // smt: BTC 를 같은 시각에 붙임(없으면 NaN)
    let bm: std::collections::HashMap<i64, (f64, f64)> = btc.iter().map(|b| (b.t, (b.h, b.l))).collect();
    let (bh, bl): (Vec<f64>, Vec<f64>) = k.iter().map(|b| bm.get(&b.t).copied().unwrap_or((NAN, NAN))).unzip();
    let (bsl, bsh) = (shift(&roll_ext(&bl, 48, 48, false), 1), shift(&roll_ext(&bh, 48, 48, true), 1));
    // DeMarker(14)
    let clip0 = |x: f64| if x.is_nan() { 0.0 } else if x < 0.0 { 0.0 } else { x };
    let up: Vec<f64> = (0..n).map(|i| clip0(if i == 0 { NAN } else { h[i] - h[i - 1] })).collect();
    let dn: Vec<f64> = (0..n).map(|i| clip0(if i == 0 { NAN } else { l[i - 1] - l[i] })).collect();
    let (smax, smin) = (rolling_mean(&up, 14), rolling_mean(&dn, 14));
    let dem: Vec<f64> = (0..n).map(|i| { let d = smax[i] + smin[i]; if d == 0.0 { NAN } else { smax[i] / d } }).collect();
    let lv = kalman_levels(&c);
    let kd: Vec<f64> = (0..n).map(|i| (c[i] - lv[i]) / lv[i]).collect();
    let kz = zscore(&kd, 288);

    let both = |b: &dyn Fn(usize) -> bool, t: &dyn Fn(usize) -> bool| -> (Vec<bool>, Vec<bool>) { ((0..n).map(b).collect(), (0..n).map(t).collect()) };
    vec![
        both(&|i| dz[i] <= -2.0, &|i| dz[i] >= 2.0),
        both(&|i| rz[i] <= -2.5, &|i| rz[i] >= 2.5),
        both(&|i| l[i] < slp[i] && c[i] > slp[i], &|i| h[i] > shp[i] && c[i] < shp[i]),
        both(&|i| pf[i] <= 0.10 && ps[i] <= 0.10 && dz[i] <= -2.0, &|i| pf[i] >= 0.90 && ps[i] >= 0.90 && dz[i] >= 2.0),
        both(&|i| l[i] < slp[i] && bl[i] > bsl[i], &|i| h[i] > shp[i] && bh[i] < bsh[i]),
        both(&|i| {
            let r = shp[i] - slp[i];
            let r = if r == 0.0 { NAN } else { r };
            hpos[i] < lpos[i] && l[i] >= slp[i] - 0.618 * r && l[i] <= slp[i] - 0.272 * r
        }, &|i| {
            let r = shp[i] - slp[i];
            let r = if r == 0.0 { NAN } else { r };
            lpos[i] < hpos[i] && h[i] >= shp[i] + 0.272 * r && h[i] <= shp[i] + 0.618 * r
        }),
        both(&|i| dem[i] <= 0.10, &|i| dem[i] >= 0.90),
        both(&|i| kz[i] <= -2.0, &|i| kz[i] >= 2.0),
    ]
}

/// pd.Timestamp(naive).tz_localize("UTC").isoformat()
fn iso(ms: i64) -> String {
    let s = rtmodel::risk_radar::iso(ms);
    let s = &s[..s.len() - 1];
    if ms.rem_euclid(1000) == 0 { format!("{s}+00:00") } else { format!("{s}.{:03}000+00:00", ms.rem_euclid(1000)) }
}

/// compute_chart_markers("eth") -- 예외를 올리지 않는다
pub fn chart_markers(cfg: &Config) -> V {
    let fail = |e: &str| json!({"available": false, "asset": "eth", "error": e});
    let run = || -> Result<V, String> {
        let k = match fetch(cfg, "ETHUSDT")? {
            Some(k) if k.len() >= 1200 => k,
            _ => return Ok(fail("price_fetch_failed")),
        };
        let Some(btc) = fetch(cfg, "BTCUSDT")? else { return Ok(fail("btc_fetch_failed")) };
        let fs = fires(&k, &btc);
        let n = k.len();
        let lo = n.saturating_sub(CHART_BARS);
        let side = |top: bool| -> (Vec<V>, Vec<V>) {
            (lo..n).map(|i| {
                let on: Vec<&str> = SIGNALS.iter().zip(&fs).filter(|(_, f)| if top { f.1[i] } else { f.0[i] }).map(|(s, _)| s.1).collect();
                (json!(on.len()), json!(on.join(",")))
            }).unzip()
        };
        let ((eb, nb), (et, nt)) = (side(false), side(true));
        let times: Vec<String> = k[lo..].iter().map(|b| iso(b.t)).collect();
        Ok(json!({"spans": {}, "span_meta": {}, "spans_partial": [], "available": true, "asset": "eth", "bars": CHART_BARS,
                  "latest_ts_utc": times.last(), "times": times, "ev_bottom": eb, "ev_top": et, "ev_bottom_names": nb, "ev_top_names": nt,
                  "events": [], "n_signals_total": SIGNALS.len()}))
    };
    run().unwrap_or_else(|e| fail(&e))
}

fn api_chart_markers(cfg: &Config, req: &Req) -> Resp {
    let a = req.q("asset").unwrap_or("eth");
    let asset = if a.is_empty() { "eth".to_string() } else { a.to_lowercase() };
    if asset != "eth" {
        return Resp::json(&json!({"available": false, "unsupported": true, "asset": asset, "reason": "증거신호·극점 탐지기 격자를 ETH 에서만 만듭니다"}));
    }
    let c = cfg.clone();
    match cfg.swr.get("chart_markers", 60.0, 600.0, Arc::new(move || Ok(chart_markers(&c)))) {
        Ok(v) => Resp::json(&v),
        Err(_) => crate::server_error(),
    }
}
