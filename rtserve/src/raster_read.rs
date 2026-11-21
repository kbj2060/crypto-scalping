//! 호가 래스터 읽기(2026-11-22 · 여섯째 조각) -- 원본 scripts/live_orderflow_raster_collector_20260914.py `read_window`(327~407)를 Read 로 확인해 옮김.
//! 파일 형식 = rtcollect/src/raster.rs 머리 주석(헤더 32B `<4sHHfIqII` · 행 16B `<qif` + f32[n_bins] · `<root>/<SYM>/<UTC YYYY-MM-DDTHH>.f32`).
//! 쓰는 곳: /api/flow/heatmap(api3.rs) · 상황 카드 재료(ctx*.rs -- 그쪽은 이 함수를 부르기만).
//! numpy 규칙: 무효 행(mid 가 NaN 이거나 ts ≤ 0 = 구멍) → mid NaN · 절대 가격축 정렬(행마다 bin_lo 가 다르다) · agg>1 = 열 묶음마다 |값| 최대(첫 최대 · 부호 유지 · NaN 이면 그 NaN) · mid = 묶음 안 유한값 최대.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub const N_BINS: usize = 240; // 원본 N_BINS(OF_N_BINS 기본)
pub const BIN_SIZE: f64 = 0.5; // 원본 BIN_SIZE(OF_BIN_SIZE 기본)
const HEADER: usize = 32;
const ROW_HEAD: usize = 16;

/// read_window 의 반환 dict 그대로. `qty` = 행(cols) × 칸(n_bins) 행 우선 f32.
#[derive(Clone, Debug, PartialEq)]
pub struct Window {
    pub symbol: String,
    pub t0_ms: i64,
    pub dt_s: i64,
    pub cols: usize,
    pub bin_size: f64,
    pub bin_lo: i64,
    pub n_bins: usize,
    pub qty: Vec<f32>,
    pub mid: Vec<f32>,
    pub valid_ratio: f64,
}

impl Window {
    /// 행 i 의 칸들
    pub fn row(&self, i: usize) -> &[f32] {
        &self.qty[i * self.n_bins..(i + 1) * self.n_bins]
    }
}

/// 원본 OF_ROOT 기본(`data/live/orderflow/raster`) -- 환경변수 OF_ROOT 가 이긴다(원본과 같은 이름)
pub fn default_root(repo: &Path) -> PathBuf {
    std::env::var("OF_ROOT").map(PathBuf::from).unwrap_or_else(|_| repo.join("data/live/orderflow/raster"))
}

/// _hour_path: <root>/<SYMBOL 대문자>/<YYYY-MM-DDTHH>.f32
pub fn hour_path(root: &Path, symbol: &str, hour_ms: i64) -> PathBuf {
    let s = hour_ms.div_euclid(1000);
    let (y, m, d) = crate::api2::civil(s.div_euclid(86400));
    root.join(symbol.to_uppercase()).join(format!("{y:04}-{m:02}-{d:02}T{:02}.f32", s.rem_euclid(86400) / 3600))
}

fn le_i64(b: &[u8]) -> i64 {
    i64::from_le_bytes(b[..8].try_into().unwrap())
}
fn le_f32(b: &[u8]) -> f32 {
    f32::from_le_bytes(b[..4].try_into().unwrap())
}

/// read_window(symbol, to_ms, cols, agg, root): [to_ms − cols·agg 초, to_ms] 를 절대 가격축으로
pub fn read_window(root: &Path, symbol: &str, to_ms: i64, cols: usize, agg: usize) -> Window {
    let total = cols * agg;
    let t_end = to_ms.div_euclid(1000) * 1000;
    let t0 = t_end - (total as i64 - 1) * 1000;
    let mut ts = vec![0i64; total];
    let mut lo = vec![0i64; total];
    let mut mid = vec![f32::NAN; total];
    let (mut n_bins, mut bin_size) = (N_BINS, BIN_SIZE);
    let mut raw: Option<Vec<f32>> = None;
    let mut hour = t0 - t0.rem_euclid(3_600_000);
    while hour <= t_end {
        if let Ok(mut fh) = std::fs::File::open(hour_path(root, symbol, hour)) {
            let mut head = [0u8; HEADER];
            if fh.read_exact(&mut head).is_ok() && &head[..4] == b"FLWR" && u16::from_le_bytes([head[4], head[5]]) == 1 {
                let hb = u16::from_le_bytes([head[6], head[7]]) as usize;
                if raw.is_none() {
                    (n_bins, bin_size) = (hb, le_f32(&head[8..]) as f64);
                    raw = Some(vec![0f32; total * n_bins]);
                }
                if hb == n_bins {
                    let rb = ROW_HEAD + 4 * n_bins;
                    let i0 = ((hour - t0).div_euclid(1000)).max(0) as usize;
                    let j0 = ((t0 - hour).div_euclid(1000)).max(0);
                    let j1 = 3600.min((t_end - hour).div_euclid(1000) + 1);
                    if j1 > j0 {
                        let mut buf = Vec::new();
                        if fh.seek(SeekFrom::Start((HEADER as i64 + j0 * rb as i64) as u64)).is_ok() {
                            let _ = (&mut fh).take(((j1 - j0) as usize * rb) as u64).read_to_end(&mut buf);
                        }
                        let r = raw.as_mut().unwrap();
                        for (k, row) in buf.chunks_exact(rb).enumerate() {
                            let i = i0 + k;
                            ts[i] = le_i64(row);
                            lo[i] = i32::from_le_bytes(row[8..12].try_into().unwrap()) as i64;
                            mid[i] = le_f32(&row[12..]);
                            for (b, c) in r[i * n_bins..(i + 1) * n_bins].iter_mut().zip(row[ROW_HEAD..].chunks_exact(4)) {
                                *b = le_f32(c);
                            }
                        }
                    }
                }
            }
        }
        hour += 3_600_000;
    }
    let raw = raw.unwrap_or_else(|| vec![0f32; total * n_bins]);
    let valid: Vec<bool> = (0..total).map(|i| mid[i].is_finite() && ts[i] > 0).collect();
    for i in 0..total {
        if !valid[i] {
            mid[i] = f32::NAN;
        }
    }
    let nv = valid.iter().filter(|v| **v).count();
    if nv == 0 {
        return Window { symbol: symbol.into(), t0_ms: t0, dt_s: agg as i64, cols, bin_size, bin_lo: 0, n_bins: 0, qty: vec![], mid: vec![f32::NAN; cols], valid_ratio: 0.0 };
    }
    let g_lo = (0..total).filter(|&i| valid[i]).map(|i| lo[i]).min().unwrap();
    let g_hi = (0..total).filter(|&i| valid[i]).map(|i| lo[i] + n_bins as i64).max().unwrap();
    let w = (g_hi - g_lo) as usize;
    let mut out = vec![0f32; total * w];
    for i in (0..total).filter(|&i| valid[i]) {
        let s = (lo[i] - g_lo) as usize;
        out[i * w + s..i * w + s + n_bins].copy_from_slice(&raw[i * n_bins..(i + 1) * n_bins]);
    }
    let (qty, mid) = if agg > 1 {
        let mut q = vec![0f32; cols * w];
        let mut m = vec![f32::NAN; cols];
        for c in 0..cols {
            for j in 0..w {
                // np.abs(o).argmax(axis=1): 첫 최대 · NaN 이 있으면 첫 NaN
                let mut pick = 0;
                let mut best = f32::NEG_INFINITY;
                for k in 0..agg {
                    let v = out[(c * agg + k) * w + j].abs();
                    if v.is_nan() {
                        pick = k;
                        break;
                    }
                    if v > best {
                        best = v;
                        pick = k;
                    }
                }
                q[c * w + j] = out[(c * agg + pick) * w + j];
            }
            let fin = (0..agg).map(|k| mid[c * agg + k]).filter(|x| x.is_finite());
            m[c] = fin.fold(None, |a: Option<f32>, x| Some(a.map_or(x, |a| a.max(x)))).unwrap_or(f32::NAN);
        }
        (q, m)
    } else {
        (out, mid)
    };
    Window { symbol: symbol.into(), t0_ms: t0, dt_s: agg as i64, cols, bin_size, bin_lo: g_lo, n_bins: w, qty, mid, valid_ratio: nv as f64 / total as f64 }
}
