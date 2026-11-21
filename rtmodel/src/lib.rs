//! 저장된 scikit-learn 모델의 추론(2026-11-20 · 설계 문서 §11 R5 첫 조각). 학습은 파이썬 그대로 -- 여기는 예측만.
//!
//! 입력 파일 = `tools/export_hgb.py` 가 쓴 JSON(실수 = f64 비트). 따른 소스(scikit-learn 1.8.0):
//! - HGB: `_hist_gradient_boosting/gradient_boosting.py` `_raw_predict`(0 + 기준값, 트리 순서대로 +=) ·
//!   `_preprocess_X`(X 를 **float64** 로 · 유한값 검사 없음) · `_predictor.pyx` `_predict_one_from_raw_data`
//!   (NaN → missing_go_to_left · 아니면 `x <= num_threshold` 왼쪽) · `_loss/loss.py` HalfBinomialLoss → expit.
//! - ExtraTrees: `ensemble/_forest.py` predict_proba(X 를 **float32** 로 · NaN 은 허용 · 무한대는 예외 ·
//!   트리 확률을 0 부터 차례로 더해 트리 수로 나눔) · `tree/_tree.pyx` `_apply_dense`(f32 값 `<=` f64 임계값).
//! - SimpleImputer(median): 무한대 → 예외 · NaN → 통계값. StandardScaler: `X -= mean; X /= scale`.
//! - LogisticRegression(이진): `X @ coef + intercept` → expit(유한값 아니면 예외).
//! 파이썬이 예외를 내는 입력은 `None`.
use serde::{Deserialize, Deserializer};

pub mod ban;
pub mod box_hold;
pub mod dir4h;
pub mod dirrest;
pub mod dirshort;
pub mod lgb;
pub mod qx4;
pub mod regime_band;
pub mod risk_radar;
pub mod signals;
pub mod sr_lean;
pub mod sr_live;
use std::collections::HashMap;
use std::path::Path;

fn bits<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
    Ok(Vec::<u64>::deserialize(d)?.into_iter().map(f64::from_bits).collect())
}
fn bit<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    u64::deserialize(d).map(f64::from_bits)
}

pub const SKLEARN: &str = "1.8.0";

#[derive(Deserialize)]
pub struct HgbTree {
    feat: Vec<usize>,
    #[serde(deserialize_with = "bits")]
    thr: Vec<f64>,
    mgl: Vec<bool>,
    left: Vec<usize>,
    right: Vec<usize>,
    leaf: Vec<bool>,
    #[serde(deserialize_with = "bits")]
    value: Vec<f64>,
}

impl HgbTree {
    fn raw(&self, x: &[f64]) -> f64 {
        let mut i = 0;
        loop {
            if self.leaf[i] {
                return self.value[i];
            }
            let v = x[self.feat[i]];
            let go_left = if v.is_nan() { self.mgl[i] } else { v <= self.thr[i] };
            i = if go_left { self.left[i] } else { self.right[i] };
        }
    }
}

#[derive(Deserialize)]
pub struct EtTree {
    feat: Vec<i64>,
    #[serde(deserialize_with = "bits")]
    thr: Vec<f64>,
    left: Vec<i64>,
    right: Vec<i64>,
    mgl: Vec<bool>,
    #[serde(deserialize_with = "bits")]
    p1: Vec<f64>,
}

impl EtTree {
    fn p1(&self, x32: &[f64]) -> f64 {
        let mut i = 0;
        while self.left[i] != -1 {
            let v = x32[self.feat[i] as usize];
            let go_left = if v.is_nan() { self.mgl[i] } else { v <= self.thr[i] };
            i = (if go_left { self.left[i] } else { self.right[i] }) as usize;
        }
        self.p1[i]
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Step {
    Hgb {
        n_features: usize,
        #[serde(deserialize_with = "bit")]
        baseline: f64,
        trees: Vec<HgbTree>,
        /// 회귀(HistGradientBoostingRegressor · 제곱 오차 · 항등 연결): predict = raw 그대로 (2026-11-21 dir4h)
        #[serde(default)]
        regress: bool,
    },
    Imputer {
        #[serde(deserialize_with = "bits")]
        fill: Vec<f64>,
    },
    Scaler {
        #[serde(deserialize_with = "bits")]
        mean: Vec<f64>,
        #[serde(deserialize_with = "bits")]
        scale: Vec<f64>,
    },
    Et {
        trees: Vec<EtTree>,
    },
    Lr {
        #[serde(deserialize_with = "bits")]
        coef: Vec<f64>,
        #[serde(deserialize_with = "bit")]
        intercept: f64,
    },
}

/// 파이프라인(또는 단일 분류기) = 단계 목록. 마지막 단계가 분류기.
pub type Model = Vec<Step>;

pub fn expit(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// `predict_proba(x)[0, 1]` 한 줄(회귀 HGB 면 `predict(x)[0]`). x = 학습 열 순서 그대로.
pub fn predict_proba(model: &[Step], x: &[f64]) -> Option<f64> {
    let mut x = x.to_vec();
    for s in model {
        match s {
            Step::Hgb { n_features, baseline, trees, regress } => {
                assert_eq!(x.len(), *n_features);
                let mut raw = 0.0 + baseline;
                for t in trees {
                    raw += t.raw(&x);
                }
                return Some(if *regress { raw } else { expit(raw) });
            }
            Step::Imputer { fill } => {
                if x.iter().any(|v| v.is_infinite()) {
                    return None;
                }
                for (v, f) in x.iter_mut().zip(fill) {
                    if v.is_nan() {
                        *v = *f;
                    }
                }
            }
            Step::Scaler { mean, scale } => {
                if x.iter().any(|v| v.is_infinite()) {
                    return None;
                }
                for ((v, m), s) in x.iter_mut().zip(mean).zip(scale) {
                    *v = (*v - m) / s;
                }
            }
            Step::Et { trees } => {
                let x32: Vec<f64> = x.iter().map(|v| *v as f32 as f64).collect();
                if x32.iter().any(|v| v.is_infinite()) {
                    return None; // f32 로 내리며 넘친 값도 여기서 걸린다(sklearn 은 캐스트 뒤에 검사)
                }
                let mut sum = 0.0;
                for t in trees {
                    sum += t.p1(&x32);
                }
                return Some(sum / trees.len() as f64);
            }
            Step::Lr { coef, intercept } => {
                if !x.iter().all(|v| v.is_finite()) {
                    return None;
                }
                // ponytail: 차례 합 -- numpy 는 BLAS(합 순서 다름, 차이 ~1e-16). 1e-9 목표엔 충분 · 비트 일치가 필요하면 BLAS 순서를 따라야 함
                let mut z = 0.0;
                for (v, c) in x.iter().zip(coef) {
                    z += v * c;
                }
                return Some(expit(z + intercept));
            }
        }
    }
    panic!("모델 끝에 분류기가 없다")
}

#[derive(Deserialize)]
struct Header {
    sklearn: String,
}

pub fn read_json<T: for<'de> Deserialize<'de>>(p: &Path) -> Result<T, String> {
    let s = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    if let Ok(h) = serde_json::from_str::<Header>(&s) {
        if h.sklearn != SKLEARN {
            return Err(format!("{}: scikit-learn {} 로 내보낸 파일(이 추론기는 {SKLEARN} 소스 기준)", p.display(), h.sklearn));
        }
    }
    serde_json::from_str(&s).map_err(|e| format!("{}: {e}", p.display()))
}

// ── 박스 유지(dashboard/box_hold.py)
#[derive(Deserialize)]
pub struct BoxScale {
    pub deployable: bool,
    pub cols: Vec<String>,
    #[serde(deserialize_with = "bit")]
    pub train_base: f64,
    #[serde(deserialize_with = "bit")]
    pub q20: f64,
    #[serde(deserialize_with = "bit")]
    pub q80: f64,
    pub model: Model,
}

#[derive(Deserialize)]
pub struct BoxBundle {
    pub source: String,
    #[serde(deserialize_with = "bits")]
    pub prof: Vec<f64>,
    pub scales: HashMap<String, BoxScale>,
}

/// box_hold.logit: [1e-6, 1-1e-6] 로 자른 뒤 log(p/(1-p)).
pub fn logit_clip(p: f64) -> f64 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

/// box_hold.payload 1h Platt90: 1/(1+exp(-(a·logit(p)+b))).
pub fn platt90(p: f64, a: f64, b: f64) -> f64 {
    1.0 / (1.0 + (-(a * logit_clip(p) + b)).exp())
}

/// dashboard/server.py _card30_pred: 행을 float32 로 → 모델 평균 → k·logit(raw)+(1-k)·logit(base) (자르지 않음).
pub fn card30(models: &[&Model], x: &[f64], k: f64, base: f64) -> f64 {
    let x32: Vec<f64> = x.iter().map(|v| *v as f32 as f64).collect();
    let mut s = 0.0;
    for m in models {
        s += predict_proba(m, &x32).expect("HGB 는 예외가 없다");
    }
    let raw = s / models.len() as f64;
    let lg = |q: f64| (q / (1.0 - q)).ln();
    1.0 / (1.0 + (-(k * lg(raw) + (1.0 - k) * lg(base))).exp())
}

// ── 레벨 5분(scripts/live_sr_lean_20261118.py _score)
#[derive(Deserialize)]
pub struct SrLean {
    pub cols: Vec<String>,
    #[serde(deserialize_with = "bits")]
    pub thr: Vec<f64>,
    pub models: Vec<Model>,
    /// 2023 확인 검정의 «사건 쪽 먼저» 건수(화면 확률의 출처) -- 워커가 생존 표시 파일에 그대로 쓴다.
    pub rate: SrRate,
}

#[derive(Deserialize)]
pub struct SrRate {
    pub top: [i64; 2],
    pub bot: [i64; 2],
}

impl SrLean {
    /// (모델 평균 p, "top"|"bot"|"mid"). 파이썬이 예외를 내는 입력(무한대·f32 넘침)이면 None.
    pub fn predict(&self, x: &[f64]) -> Option<(f64, &'static str)> {
        let mut s = 0.0;
        for m in &self.models {
            s += predict_proba(m, x)?;
        }
        let p = s / self.models.len() as f64;
        Some((p, if p >= self.thr[1] { "top" } else if p <= self.thr[0] { "bot" } else { "mid" }))
    }
}

// ── QX4I 4h 등조(box_hold.with_qx4i · p_stay)
#[derive(Deserialize)]
pub struct Qx4iIso {
    #[serde(deserialize_with = "bits")]
    pub x: Vec<f64>,
    #[serde(deserialize_with = "bits")]
    pub y: Vec<f64>,
    #[serde(deserialize_with = "bit")]
    pub c_qx4i: f64,
    #[serde(deserialize_with = "bit")]
    pub c_qx4: f64,
}

pub fn p_stay(s: f64) -> f64 {
    if s < 0.02 {
        return 1.0;
    }
    let pi2 = std::f64::consts::PI * std::f64::consts::PI;
    // ponytail: 차례 합 -- np.sum 은 짝 합(pairwise). 항이 빠르게 0 으로 가 차이 ~1e-17
    let mut sum = 0.0;
    for k in 0..400 {
        let m = (2 * k + 1) as f64;
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        sum += sign / m * (-(m * m) * pi2 * s * s / 8.0).exp();
    }
    (4.0 / std::f64::consts::PI * sum).clamp(0.0, 1.0)
}

/// np.interp(x, xp, fp) 한 점(xp 오름차순).
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x > xp[n - 1] {
        return fp[n - 1];
    }
    if x < xp[0] {
        return fp[0];
    }
    if x == xp[n - 1] {
        return fp[n - 1];
    }
    let j = xp.partition_point(|v| *v <= x) - 1; // xp[j] <= x < xp[j+1]
    if x == xp[j] {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let r = slope * (x - xp[j]) + fp[j];
    if r.is_nan() {
        let r2 = slope * (x - xp[j + 1]) + fp[j + 1];
        if r2.is_nan() && fp[j] == fp[j + 1] {
            return fp[j];
        }
        return r2;
    }
    r
}

impl Qx4iIso {
    /// 4h 유지 확률(반올림 전). log_rv_iv 가 있으면 QX4I 눈금, 없으면 QX4 눈금.
    pub fn p(&self, log_rv: f64, log_rv_iv: Option<f64>, half_frac: f64) -> f64 {
        let lq = match log_rv_iv {
            Some(v) => v + self.c_qx4i,
            None => log_rv + self.c_qx4,
        };
        interp(p_stay(lq.exp().sqrt() / half_frac), &self.x, &self.y)
    }
}
