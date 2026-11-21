//! LightGBM 회귀 부스터 추론 -- `Booster.model_to_string()` 텍스트를 읽어 나무를 평가한다 (2026-11-21 · R5 · 15분·1시간 방향 dir_short).
//! 규칙은 설치된 lightgbm 4.7.0 의 predict 와 무작위·특수값·임계값 이웃 3만 행을 모델 80개에서 비트 대조해 정했다(tests/lgb.rs · tools/gen_lgb_fixture.py):
//! - 입력은 float32 를 double 로. |x| ≤ 1e-35(kZeroThreshold)는 0 으로(밀집 행을 희소 쌍으로 바꿀 때 «0» 으로 빠진다).
//! - 노드: decision_type 비트 1 = 기본 왼쪽 · 비트 2-3 = 결측 종류(0 없음 · 1 영 · 2 NaN). 결측 종류가 NaN 이 아니면 NaN 은 0 으로 본다.
//!   (영 · |x| ≤ 1e-35) 또는 (NaN · NaN) 이면 기본 방향 · 아니면 x ≤ 임계값 이 왼쪽. 범주형(비트 0)은 안 옮겼다(65열판 모델에 없음 -- 읽을 때 거부).
//! - 예측 = 0 에서 나무 차례로 더함(GBDT::PredictRaw) · 목적 regression·huber 는 변환 없음.
use serde_json::Value;

/// LightGBM kZeroThreshold = 1e-35f(float 상수) → double 1.0000000180025095e-35. 🔴f64 1e-35 로 쓰면 f32 −1e-35 입력(그 float 와 같은 값)이 0 으로 안 바뀐다(실측 13행).
const K_ZERO: f64 = 1e-35f32 as f64;

pub struct Tree {
    feat: Vec<usize>,
    thr: Vec<f64>,
    dt: Vec<u8>,
    left: Vec<i32>,
    right: Vec<i32>,
    leaf: Vec<f64>,
}

pub struct Booster {
    pub trees: Vec<Tree>,
    pub n_features: usize,
}

fn nums<T: std::str::FromStr>(blk: &str, key: &str) -> Result<Vec<T>, String> {
    let line = blk.lines().find(|l| l.starts_with(key) && l.as_bytes().get(key.len()) == Some(&b'=')).ok_or(format!("{key} 없음"))?;
    line[key.len() + 1..].split_whitespace().map(|s| s.parse::<T>().map_err(|_| format!("{key}: {s}"))).collect()
}

impl Booster {
    pub fn parse(txt: &str) -> Result<Booster, String> {
        let head = txt.split("Tree=").next().unwrap_or("");
        let obj = head.lines().find_map(|l| l.strip_prefix("objective=")).ok_or("objective 없음")?;
        let o = obj.split_whitespace().next().unwrap_or("");
        if !(o == "regression" || o == "huber") || obj.contains("sqrt") {
            return Err(format!("목적 {obj} -- 회귀(변환 없음)만 옮겼다"));
        }
        if head.lines().any(|l| l.starts_with("average_output")) || txt.contains("is_linear=1") {
            return Err("average_output·선형 나무는 안 옮겼다".into());
        }
        let n_features = nums::<usize>(head, "max_feature_idx")?[0] + 1;
        let mut trees = vec![];
        for blk in txt.split("\nTree=").skip(1) {
            let blk = blk.split("\nend of trees").next().unwrap();
            let nl = nums::<usize>(blk, "num_leaves")?[0];
            if nums::<usize>(blk, "num_cat")?[0] != 0 {
                return Err("범주형 분기는 안 옮겼다".into());
            }
            let leaf = nums::<f64>(blk, "leaf_value")?;
            if nl == 1 {
                trees.push(Tree { feat: vec![], thr: vec![], dt: vec![], left: vec![], right: vec![], leaf });
                continue;
            }
            let dt = nums::<u8>(blk, "decision_type")?;
            if dt.iter().any(|d| d & 1 == 1) {
                return Err("범주형 분기는 안 옮겼다".into());
            }
            trees.push(Tree { feat: nums(blk, "split_feature")?, thr: nums(blk, "threshold")?, dt, left: nums(blk, "left_child")?, right: nums(blk, "right_child")?, leaf });
        }
        Ok(Booster { trees, n_features })
    }

    /// 한 행 예측. x = float32 값(f64 로 담아 온 것) · 학습 열 순서.
    pub fn predict(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.n_features);
        let x: Vec<f64> = x.iter().map(|&v| if v.abs() <= K_ZERO { 0.0 } else { v }).collect();
        let mut s = 0.0;
        for t in &self.trees {
            s += t.eval(&x);
        }
        s
    }
}

impl Tree {
    fn eval(&self, x: &[f64]) -> f64 {
        if self.feat.is_empty() {
            return self.leaf[0];
        }
        let mut n = 0i32;
        while n >= 0 {
            let i = n as usize;
            let (d, mut v) = (self.dt[i], x[self.feat[i]]);
            let miss = (d >> 2) & 3;
            if v.is_nan() && miss != 2 {
                v = 0.0;
            }
            let default = (miss == 1 && (-K_ZERO..=K_ZERO).contains(&v)) || (miss == 2 && v.is_nan());
            let left = if default { d & 2 != 0 } else { v <= self.thr[i] };
            n = if left { self.left[i] } else { self.right[i] };
        }
        self.leaf[(!n) as usize]
    }
}

/// live_eth_dir_short_20261028.predict: 지평마다 {시드 5 평균(각 시드 = Σ 표적 예측 / 척도)}.
/// seeds[시드][표적] = (부스터, 척도). 원본 `np.mean([sum(m.predict(X) / s for m, s in ms) for ms in seeds], 0)`.
pub fn ensemble(seeds: &[Vec<(Booster, f64)>], x: &[f64]) -> f64 {
    let mut tot = 0.0;
    for (k, ms) in seeds.iter().enumerate() {
        let mut s = 0.0; // 파이썬 sum() = 0 에서 차례 합
        for (m, sc) in ms {
            s += m.predict(x) / sc;
        }
        tot = if k == 0 { s } else { tot + s };
    }
    tot / seeds.len() as f64
}

/// tools/gen_lgb_fixture.py 가 쓴 {"txt", "scale"(f64 비트)} 목록 → 부스터.
pub fn load_seeds(v: &Value) -> Result<Vec<Vec<(Booster, f64)>>, String> {
    v.as_array().ok_or("시드 목록 아님")?.iter().map(|ms| {
        ms.as_array().ok_or("표적 목록 아님")?.iter().map(|m| {
            Ok((Booster::parse(m["txt"].as_str().ok_or("txt")?)?, f64::from_bits(m["scale"].as_u64().ok_or("scale")?)))
        }).collect()
    }).collect()
}
