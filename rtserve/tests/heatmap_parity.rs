//! 여섯째 조각: 호가 래스터 읽기(raster_read.rs)·히트맵 계산 ↔ 원본 함수(서버 판 numpy 2.2.5).
//! 정답지 = tools/gen_heatmap_fixture.py(커밋 안 하는 생성물 · tests/fixtures/gen/raster) · 배열은 f32 바이트 sha256 으로 비트 대조.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn gen() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gen/raster")
}

fn fx() -> Value {
    serde_json::from_str(&std::fs::read_to_string(gen().join("heatmap.json")).expect("서버 판 venv 로 tools/gen_heatmap_fixture.py 먼저")).unwrap()
}

pub fn sha_f32(a: &[f32]) -> String {
    let mut h = Sha256::new();
    for x in a {
        h.update(x.to_le_bytes());
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn read_window_matches_python() {
    let fx = fx();
    let mut bad = Vec::new();
    let cases = fx["read_window"].as_array().unwrap();
    for c in cases {
        let (to, cols, agg) = (c["to_ms"].as_i64().unwrap(), c["cols"].as_u64().unwrap() as usize, c["agg"].as_u64().unwrap() as usize);
        let w = rtserve::raster_read::read_window(&gen(), "ethusdt", to, cols, agg);
        let p = &c["window"];
        let got = serde_json::json!({"symbol": w.symbol, "t0_ms": w.t0_ms, "dt_s": w.dt_s, "cols": w.cols, "bin_size": w.bin_size, "bin_lo": w.bin_lo,
            "n_bins": w.n_bins, "qty": sha_f32(&w.qty), "mid": sha_f32(&w.mid), "valid_ratio": w.valid_ratio, "qty_shape": [if w.n_bins == 0 { cols } else { w.qty.len() / w.n_bins }, w.n_bins]});
        for k in ["symbol", "t0_ms", "dt_s", "cols", "bin_size", "bin_lo", "n_bins", "qty", "mid", "valid_ratio", "qty_shape"] {
            if got[k] != p[k] {
                bad.push(format!("{to}/{cols}/{agg} {k}: rust {} py {}", got[k], p[k]));
            }
        }
    }
    println!("read_window: 사례 {} · 다름 {}", cases.len(), bad.len());
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

fn dumps(v: &Value) -> String {
    rtserve::py::dumps(v, (", ", ": "), true, None)
}

fn diff_at(a: &str, b: &str) -> String {
    let i = a.bytes().zip(b.bytes()).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
    format!("글자 {i}: rust …{} / py …{}", &a[i.saturating_sub(60)..(i + 60).min(a.len())], &b[i.saturating_sub(60)..(i + 60).min(b.len())])
}

#[test]
fn heatmap_matches_python() {
    let fx = fx();
    let fp: rtserve::heatmap::Fp = fx["fp"].as_object().unwrap().iter().map(|(b, c)| {
        (b.parse().unwrap(), c.as_array().unwrap().iter().map(|r| (r[0].as_f64().unwrap(), r[1].as_f64().unwrap(), r[2].as_f64().unwrap())).collect())
    }).collect();
    let mut bad = Vec::new();
    for c in fx["heatmap"].as_array().unwrap() {
        let (to, cols, agg, ro) = (c["to_ms"].as_i64().unwrap(), c["cols"].as_u64().unwrap() as usize, c["agg"].as_u64().unwrap() as usize, c["rows_only"].as_bool().unwrap());
        let g = rtserve::heatmap::heatmap_read(&gen(), "ethusdt", to, cols, agg, ro, if c["fp"].as_bool().unwrap() { Some(&fp) } else { None });
        let mine = g.map(|v| dumps(&v)).unwrap_or_else(|e| format!("ERR {}", e.split(':').next().unwrap()));
        let want = c["body"].as_str().unwrap();
        if mine != want {
            bad.push(format!("heatmap {to}/{cols}/{agg}/{ro}: {}", diff_at(&mine, want)));
        }
    }
    for c in fx["approach"].as_array().unwrap() {
        let mine = dumps(&Value::Object(rtserve::heatmap::approach_read(&gen(), "ethusdt", c["to_ms"].as_i64().unwrap())));
        if mine != c["body"].as_str().unwrap() {
            bad.push(format!("approach {}: {}", c["to_ms"], diff_at(&mine, c["body"].as_str().unwrap())));
        }
    }
    for c in fx["volsig"].as_array().unwrap() {
        let to = c["to_ms"].as_i64().unwrap();
        let rf = rtserve::heatmap::volsig_ref(&gen(), "ethusdt", to);
        let n = rf.as_ref().map_or(0, |r| r.2.len());
        if n as u64 != c["ref_n"].as_u64().unwrap() {
            bad.push(format!("volsig ref {to}: n {n} vs {}", c["ref_n"]));
        }
        if let Some(r) = &rf {
            let sc: Vec<u8> = r.2.iter().flat_map(|x| x.to_le_bytes()).collect();
            let got = [sha_f32(&r.0), sha_f32(&r.1), Sha256::digest(&sc).iter().map(|b| format!("{b:02x}")).collect()];
            for (k, g) in ["past", "refill", "score"].iter().zip(got) {
                if c["ref"][k] != g.as_str() {
                    bad.push(format!("volsig ref {to} {k}"));
                }
            }
        }
        let read = rf.as_ref().map_or_else(serde_json::Map::new, |r| rtserve::heatmap::volsig_read(&gen(), "ethusdt", to, r));
        if dumps(&Value::Object(read.clone())) != c["read"].as_str().unwrap() {
            bad.push(format!("volsig read {to}: {} vs {}", dumps(&Value::Object(read)), c["read"]));
        }
    }
    println!("heatmap: 본문 {} · approach {} · volsig {} · 다름 {}", fx["heatmap"].as_array().unwrap().len(), fx["approach"].as_array().unwrap().len(), fx["volsig"].as_array().unwrap().len(), bad.len());
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// /api/flow/heatmap HTTP(api3.rs 조립 · 원본 4692~4732) + 400 갈래
#[path = "common/mod.rs"]
mod common;

#[test]
fn heatmap_route_over_http() {
    let fx = fx();
    let mut bad = Vec::new();
    let mut n = 0;
    for c in fx["api"].as_array().unwrap() {
        let to = c["to_ms"].as_f64().unwrap();
        let (addr, root) = common::start_with(&serde_json::Map::new(), 0, to / 1000.0, "eth,sol,xrp", &format!("hm_{n}"), |_| {});
        std::fs::create_dir_all(root.join("data/live/orderflow")).unwrap();
        std::os::unix::fs::symlink(gen(), root.join("data/live/orderflow/raster")).unwrap();
        let g = common::request(addr, "GET", &format!("/api/flow/heatmap{}", c["q"].as_str().unwrap()), &[]);
        let mine = String::from_utf8_lossy(&g.body).into_owned();
        if g.status != 200 || mine != c["body"].as_str().unwrap() {
            bad.push(format!("{}: {} {}", c["q"], g.status, diff_at(&mine, c["body"].as_str().unwrap())));
        }
        n += 1;
        if n == 1 {
            for (q, why) in [("?symbol=dogeusdt", "unsupported_symbol"), ("?cols=abc", "bad_cols_or_agg"), ("?agg=1.5", "bad_cols_or_agg")] {
                let g = common::request(addr, "GET", &format!("/api/flow/heatmap{q}"), &[]);
                n += 1;
                if g.status != 400 || g.body != format!("400: {why}").as_bytes() {
                    bad.push(format!("{q}: {} {}", g.status, String::from_utf8_lossy(&g.body)));
                }
            }
        }
        let _ = std::fs::remove_dir_all(root);
    }
    println!("heatmap HTTP: 요청 {n} · 다름 {}", bad.len());
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// 🔴깨진 래스터(행 bin_lo ±20억 · 머리 n_bins 65535 × 큰 창)로 할당하다 죽지 않는다 -- 원본 MemoryError 처럼 그 요청만 실패(heatmap = 500 · 읽기 = Err/빈 창)
#[test]
fn broken_raster_does_not_abort() {
    let dir = std::env::temp_dir().join(format!("rtserve_badraster_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let sym = dir.join("ETHUSDT");
    std::fs::create_dir_all(&sym).unwrap();
    let hour: i64 = 1_795_000_000_000 / 3_600_000 * 3_600_000;
    let head = |nb: u16| [&b"FLWR"[..], &1u16.to_le_bytes(), &nb.to_le_bytes(), &0.5f32.to_le_bytes(), &0u32.to_le_bytes(), &hour.to_le_bytes(), &1000u32.to_le_bytes(), &0u32.to_le_bytes()].concat();
    let mut f = head(240);
    for (k, lo) in [(0i64, 6000i32), (1, 2_000_000_000), (2, -2_000_000_000), (3, 6001)] {
        f.extend_from_slice(&(hour + k * 1000).to_le_bytes());
        f.extend_from_slice(&lo.to_le_bytes());
        f.extend_from_slice(&3000f32.to_le_bytes());
        f.extend(std::iter::repeat_n(1f32.to_le_bytes(), 240).flatten());
    }
    std::fs::write(rtserve::raster_read::hour_path(&dir, "ethusdt", hour), &f).unwrap();
    let to = hour + 10_000;
    let e = rtserve::raster_read::try_read_window(&dir, "ethusdt", to, 300, 1).unwrap_err();
    assert!(e.starts_with("MemoryError"), "{e}");
    assert_eq!(rtserve::raster_read::read_window(&dir, "ethusdt", to, 300, 1).n_bins, 0, "못 읽음 = 빈 창");
    assert!(rtserve::heatmap::heatmap_read(&dir, "ethusdt", to, 300, 1, false, None).is_err(), "히트맵 = 오류(→ 500)");
    assert!(rtserve::heatmap::approach_read(&dir, "ethusdt", to).is_empty());
    // 머리 n_bins 65535 × 54,000행 = 35억 칸
    std::fs::write(rtserve::raster_read::hour_path(&dir, "ethusdt", hour), head(65535)).unwrap();
    assert!(rtserve::raster_read::try_read_window(&dir, "ethusdt", to, 900, 60).unwrap_err().starts_with("MemoryError"));
    let _ = std::fs::remove_dir_all(dir);
}
