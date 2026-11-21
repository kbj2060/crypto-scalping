//! 15분·1시간 방향 -- 피쳐(dir4h::build_features 와 같은 65열) → LightGBM 시드 앙상블(lgb) → 화면 payload (2026-11-21 · R5).
//! 원본 scripts/live_eth_dir_short_20261028.py `load_art`·`predict`·`oos_sorted`·`payload_from_features`(65열판 = BTC·SOL·XRP).
//! ETH 는 75열판(65 + 호가 깊이 워커 sqlite 10열 · warm·결측·빈 칸이면 그 봉 예측을 비움 -- `book_status`·`payload_book`). sqlite 읽기는 부르는 쪽(rtmodel 은 sqlite 의존성 없음).
//! `book_feats`·`dataset` 은 학습 몫(서빙은 워커가 이미 계산해 둔 sqlite 10열을 그대로 읽는다) -- 안 옮겼다.
use crate::box_hold::py_round;
use crate::dir4h::{Features, FEATS};
use crate::lgb::{ensemble, Booster};
use serde_json::{json, Value};

/// 지평(봉) → 이름. 원본 HZ·NAMES 순서(3, 12).
pub const NAMES: [(usize, &str); 2] = [(3, "15m"), (12, "1h")];

pub struct Art {
    pub coin: String,
    pub month: String,
    /// 지평 이름 → 학습 코인(원본 art["src"] · ETH 75열판 아티팩트엔 없음 = null)
    pub src: Option<Vec<(String, String)>>,
    /// 지평(3·12) → 시드[표적(부스터, 척도)]
    pub models: Vec<(usize, Vec<Vec<(Booster, f64)>>)>,
}

impl Art {
    /// tools/gen_dirshort_fixture.py 가 쓴 ds_<코인>.json.
    pub fn from_json(v: &Value) -> Result<Art, String> {
        let src = v["src"].is_object().then(|| NAMES.iter().filter_map(|(_, nm)| v["src"].get(*nm).and_then(Value::as_str).map(|s| (nm.to_string(), s.to_string()))).collect());
        let mut models = vec![];
        for (h, _) in NAMES {
            if let Some(m) = v["models"].get(h.to_string()) {
                models.push((h, crate::lgb::load_seeds(m)?));
            }
        }
        // coin 없음 = "ETH"(원본 art.get("coin", "ETH"))
        Ok(Art { coin: v["coin"].as_str().unwrap_or("ETH").into(), month: v["month"].as_str().ok_or("month")?.into(), src, models })
    }
}

/// np.quantile(q, p)(linear · 정렬된 q): numpy 2.4 `_QuantileMethods['linear']` 가상 색인 (n−1)·p 와 _lerp(γ ≥ 0.5 면 b − (b−a)(1−γ)).
/// 🔴처음엔 _compute_virtual_index(n·p + (1−p) − 1, 다른 방법들 몫)로 써서 1ulp 틀렸다 -- 설치본 numpy/lib/_function_base_impl.py 를 읽고 고침.
pub fn np_quantile(q: &[f64], p: f64) -> f64 {
    let n = q.len();
    let vi = (n - 1) as f64 * p;
    let (i, j, g) = if vi >= (n - 1) as f64 {
        (n - 1, n - 1, 0.0)
    } else if vi < 0.0 {
        (0, 0, 0.0)
    } else {
        let f = vi.floor();
        (f as usize, f as usize + 1, vi - f)
    };
    let (a, b) = (q[i], q[j]);
    let d = b - a;
    if g >= 0.5 { b - d * (1.0 - g) } else { a + d * g }
}

/// 호가 깊이 한 봉의 상태(원본 book_row): None = 행 없음 · warm 이 참 · 빈 칸 → 예측을 비운다.
pub fn book_status(row: Option<(f64, &[Option<f64>])>) -> (Option<Vec<f64>>, &'static str) {
    match row {
        None => (None, "missing"),
        Some((warm, _)) if warm != 0.0 => (None, "warm"),
        Some((_, v)) if v.iter().any(Option::is_none) => (None, "nan"),
        Some((_, v)) => (Some(v.iter().map(|x| x.unwrap() as f32 as f64).collect()), "ok"),
    }
}

/// payload_from_features(65열판): f 의 마지막 n_hist 봉(원본 live_payload 는 13). oos = 지평 이름별 정렬된 표본 밖 예측.
pub fn payload(art: &Art, f: &Features, n_hist: usize, oos: &[(&str, Vec<f64>)]) -> Value {
    payload_book(art, f, n_hist, oos, &|_| (Some(vec![]), "n/a"))
}

/// payload_from_features(ETH 75열판): book(봉 시작 ms) = 호가 깊이 워커 sqlite 한 봉(`book_status` 결과 -- 읽기는 부르는 쪽).
/// 온전한 봉만 예측·이력에 들어가고, 지금 봉이 온전하지 않으면 pred = null(원본과 같음).
pub fn payload_book(art: &Art, f: &Features, n_hist: usize, oos: &[(&str, Vec<f64>)], book: &dyn Fn(i64) -> (Option<Vec<f64>>, &'static str)) -> Value {
    let n = f.ts.len();
    let all: Vec<usize> = (n.saturating_sub(n_hist)..n).collect();
    let bk: Vec<(Option<Vec<f64>>, &str)> = all.iter().map(|&i| book(f.ts[i])).collect();
    let ok: Vec<usize> = (0..all.len()).filter(|&j| bk[j].0.is_some()).collect();
    let rows: Vec<usize> = ok.iter().map(|&j| all[j]).collect();
    let x = |j: usize| FEATS.iter().map(|c| f.cols[c][all[j]] as f32 as f64).chain(bk[j].0.clone().unwrap()).collect::<Vec<f64>>();
    let preds: Vec<Vec<f64>> = art.models.iter().map(|(_, seeds)| ok.iter().map(|&j| ensemble(seeds, &x(j))).collect()).collect();
    let q = |nm: &str| &oos.iter().find(|o| o.0 == nm).expect("표본 밖 분위").1;
    let pct = |nm: &str, v: f64| {
        let s = q(nm);
        py_round(100.0 * s.partition_point(|&a| a < v) as f64 / s.len() as f64, 1) // np.searchsorted(left)
    };
    let thr: Vec<(&str, f64, f64)> = oos.iter().map(|(nm, s)| (*nm, py_round(np_quantile(s, 0.2), 4), py_round(np_quantile(s, 0.8), 4))).collect();
    let last = n - 1;
    let nan: Vec<&str> = FEATS.iter().copied().filter(|c| !f.cols[c][last].is_finite()).collect();
    let hist: Vec<Value> = rows.iter().enumerate().map(|(j, &i)| {
        let mut r = vec![json!(f.ts[i] + 300_000)];
        for (k, (h, _)) in art.models.iter().enumerate() {
            let nm = NAMES.iter().find(|x| x.0 == *h).unwrap().1;
            r.push(json!(pct(nm, preds[k][j])));
        }
        Value::Array(r)
    }).collect();
    let mut pred = serde_json::Map::new();
    let now_ok = ok.last() == Some(&(all.len() - 1)); // 지금 봉의 호가 깊이가 온전할 때만(warm·결측 = 비움)
    for (k, (h, _)) in art.models.iter().enumerate().filter(|_| now_ok) {
        let nm = NAMES.iter().find(|x| x.0 == *h).unwrap().1;
        let v = preds[k][rows.len() - 1];
        let (_, lo, hi) = *thr.iter().find(|t| t.0 == nm).unwrap();
        pred.insert(nm.into(), json!({"score": py_round(v, 4), "pct": pct(nm, v), "signal": if v >= hi { 1 } else if v <= lo { -1 } else { 0 }}));
    }
    let src = art.src.as_ref().map_or(Value::Null, |s| Value::Object(s.iter().map(|(a, b)| (a.clone(), json!(b))).collect()));
    json!({"ok": true, "asset": art.coin.to_lowercase(), "src": src,
           "decided_at_ms": f.ts[last] + 300_000, "model_month": art.month, "book": bk.last().map_or("n/a", |b| b.1), "nan_feats": nan,
           "thr": thr.iter().map(|(nm, lo, hi)| (nm.to_string(), json!([lo, hi]))).collect::<serde_json::Map<_, _>>(),
           "pred": if now_ok { Value::Object(pred) } else { Value::Null }, "hist": hist})
}
