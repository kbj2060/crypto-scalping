//! «띠 안·이탈» 을 원본(scripts/live_regime_band_*_20261102.py · live_liquidation_map_20260824.py)과 대조. 정답지는 커밋하지 않는다(약 3MB):
//!     (tools/server_env.sh 의 환경) python rtmodel/tools/gen_regime_fixture.py
//!     cargo test -j 2 --release --test regime_band -- --nocapture
use rtmodel::regime_band::*;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;

fn fb(v: &Value) -> f64 {
    f64::from_bits(v.as_u64().unwrap())
}
fn fbs(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(fb).collect()
}
fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

struct Fake<'a> {
    ser: &'a Value,
    now: i64,
    vis: i64,
    reqs: Vec<(String, Option<i64>)>,
}
impl Fake<'_> {
    fn get(&mut self, path: &str, q: &[(&'static str, String)]) -> Result<Value, String> {
        assert_eq!(path, "/fapi/v1/klines");
        let p: HashMap<&str, &String> = q.iter().map(|(k, v)| (*k, v)).collect();
        assert_eq!((p["interval"].as_str(), p["limit"].as_str()), ("5m", "1500"));
        assert_eq!(q.iter().map(|x| x.0).collect::<Vec<_>>()[..3], ["symbol", "interval", "limit"]);
        let end = p.get("endTime").map(|v| v.parse::<i64>().unwrap());
        self.reqs.push((p["symbol"].clone(), end));
        let rows: Vec<&Value> = self.ser[&p["symbol"][..p["symbol"].len() - 4]].as_array().unwrap().iter().filter(|r| r[0].as_i64().unwrap() <= end.unwrap_or(self.now).min(self.vis)).collect();
        Ok(Value::Array(rows[rows.len().saturating_sub(1500)..].iter().map(|r| (*r).clone()).collect()))
    }
}

fn cut(p: Vec<Bar>) -> Vec<Bar> {
    let k = p.iter().rposition(|x| x.t.rem_euclid(3600) / 60 == 55).unwrap();
    p[..=k].to_vec()
}

fn check_level(g: &(f64, Vec<f64>, Vec<f64>), w: &Value, tag: &str, bad: &mut Vec<String>) -> usize {
    let (p, s, r) = (fb(&w["p"]), fbs(&w["sup"]), fbs(&w["res"]));
    let ok = same(g.0, p) && g.1.len() == s.len() && g.2.len() == r.len() && g.1.iter().zip(&s).chain(g.2.iter().zip(&r)).all(|(a, b)| same(*a, *b));
    if !ok {
        bad.push(format!("{tag} level: Rust {g:?} 대 파이썬 ({p}, {s:?}, {r:?})"));
    }
    1 + s.len() + r.len()
}

/// 원본 두 벌 칸([[번호, 무게 비트], …] | null)과 넣은 순서·무게 비트까지.
fn check_bins(df: &[Bar], t: i64, w: &Value, tag: &str, bad: &mut Vec<String>) -> usize {
    let (_, g) = spliced_bins(df, t);
    let pick = |k: usize| g.as_ref().map(|x| if k == 0 { x.0.clone() } else { x.1.clone() });
    let mut n = 0;
    for k in 0..2 {
        let want: Option<Vec<(i64, f64)>> = if w.is_null() { None } else { w[k].as_array().map(|a| a.iter().map(|e| (e[0].as_i64().unwrap(), fb(&e[1]))).collect()) };
        let got = pick(k).flatten();
        let ok = match (&got, &want) {
            (Some(a), Some(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.0 == y.0 && same(x.1, y.1)),
            (None, None) => true,
            _ => false,
        };
        if !ok {
            bad.push(format!("{tag} bins[{k}]: 길이 Rust {:?} 대 파이썬 {:?}", got.as_ref().map(Vec::len), want.as_ref().map(Vec::len)));
        }
        n += want.map_or(1, |b| b.len());
    }
    n
}

fn check_x(g: &[f64], w: &Value, cols: &[String], tag: &str, bad: &mut Vec<String>) -> usize {
    for ((a, b), c) in g.iter().zip(fbs(w)).zip(cols) {
        if !same(*a, b) {
            bad.push(format!("{tag} {c}: Rust {a:?} 대 파이썬 {b:?}"));
        }
    }
    g.len()
}

#[test]
fn regime_band_parity() {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let b = Bundle::from_json(&rtmodel::read_json::<Value>(&d.join("regime_band.json")).expect("python rtmodel/tools/gen_regime_fixture.py 먼저")).unwrap();
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(d.join("regime_fixture.json")).expect("정답지 없음 -- 머리말")).unwrap();
    let tmp = std::env::temp_dir().join(format!("rt_regime_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let (log, scored, summ) = (tmp.join("f.jsonl"), tmp.join("s.jsonl"), tmp.join("sum.json"));
    let (mut bad, mut n_lv, mut n_bin, mut n_x, mut n_p, mut n_file) = (vec![], 0, 0, 0, 0, 0);
    let mut last: Option<HashMap<String, Vec<Bar>>> = None;
    for run in fx["runs"].as_array().unwrap() {
        let (now, vis) = (run["now"].as_i64().unwrap(), run["vis"].as_i64().unwrap());
        let mut f = Fake { ser: &fx["series"], now, vis, reqs: vec![] };
        let got = cron(&mut |p, q| f.get(p, q), now, &b, &log, &scored, &summ);
        match (&got, run.get("out").and_then(Value::as_str)) {
            (Ok(g), Some(w)) => assert_eq!(g, w, "{now} 출력"),
            (Err(e), None) => assert!(e.starts_with(run["fail"].as_str().unwrap()), "{now} 실패 종류 {e}"),
            _ => panic!("{now}: Rust {got:?} 대 파이썬 {:?}", run.get("out").or(run.get("fail"))),
        }
        let wr: Vec<(String, Option<i64>)> = run["requests"].as_array().unwrap().iter().map(|r| (r[0].as_str().unwrap().to_string(), r[1].as_i64())).collect();
        assert_eq!(f.reqs, wr, "{now} 요청");
        for (k, p) in [("log", &log), ("scored", &scored), ("summary", &summ)] {
            assert_eq!(std::fs::read_to_string(p).ok().as_deref(), run["files"][k].as_str(), "{now} {k} 파일");
            n_file += 1;
        }
        let Some(coins) = run.get("coins") else { continue };
        let mut f = Fake { ser: &fx["series"], now, vis, reqs: vec![] };
        let panels: HashMap<String, Vec<Bar>> = COINS.iter().map(|c| (c.to_string(), cut(panel(&mut |p, q| f.get(p, q), &format!("{c}USDT"), now).unwrap()))).collect();
        for c in COINS {
            let w = &coins[c];
            assert_eq!(panels[c].len() as u64, w["n"].as_u64().unwrap());
            let o = predict_latest(&panels, c, &b).unwrap();
            n_lv += check_level(&o.level, &w["level"], &format!("{now} {c}"), &mut bad);
            n_bin += check_bins(&panels[c], o.ts, &w["bins"], &format!("{now} {c}"), &mut bad);
            n_x += check_x(&o.x, &w["x"], &b.features, &format!("{now} {c}"), &mut bad);
            n_p += 1;
            if !same(o.p, fb(&w["p"])) {
                bad.push(format!("{now} {c} p: Rust {:?} 대 파이썬 {:?}", o.p, fb(&w["p"])));
            }
        }
        last = Some(panels);
    }
    let panels = last.unwrap();
    for c in COINS {
        let g = &fx["grid"][c];
        let df = &panels[c];
        let bc = btc_on(df, &panels["BTC"]);
        for (k, i) in g["i"].as_array().unwrap().iter().enumerate() {
            let i = i.as_u64().unwrap() as usize;
            let lv = level_at(df, df[i].t + 300);
            n_lv += check_level(&lv, &g["level"][k], &format!("격자 {c} {i}"), &mut bad);
            if let Some(wb) = g["bins"].get(i.to_string()) {
                n_bin += check_bins(df, df[i].t + 300, wb, &format!("격자 {c} {i}"), &mut bad);
            }
            let (x, _, _) = features(df, &bc, i, &lv, &b.features);
            n_x += check_x(&x, &g["x"][k], &b.features, &format!("격자 {c} {i}"), &mut bad);
            let p = proba(&b, &x);
            n_p += 1;
            if !same(p, fb(&g["p"][k])) {
                bad.push(format!("격자 {c} {i} p: Rust {p:?} 대 파이썬 {:?}", fb(&g["p"][k])));
            }
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
    println!("레벨 칸 {n_lv} · 청산 칸(번호·무게) {n_bin} · 피쳐 칸 {n_x} · 확률 {n_p} · 파일 대조 {n_file} · 다름 {}", bad.len());
    assert!(bad.is_empty(), "{}", bad[..bad.len().min(30)].join("\n"));
}

/// rt_regime 실행 파일 두 회차(RT_REGIME_NOW_MS)를 127.0.0.1 가짜 HTTP 로: 출력 줄·세 파일 글자 == 정답지 · 파이썬 기록 자리 거부(종료코드 2) · 묵은 자료 FAIL(종료코드 1).
#[test]
fn regime_bin_over_fake_http() {
    use std::io::{Read, Write};
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(d.join("regime_fixture.json")).expect("정답지 없음")).unwrap();
    let dir = std::env::temp_dir().join(format!("rt_regime_bin_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(d.join("regime_band.json"), dir.join("regime_band.json")).unwrap();
    let now = std::sync::Arc::new(std::sync::Mutex::new(0i64));
    let (ser, nn) = (fx["series"].clone(), now.clone());
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for s in l.incoming() {
            let mut s = s.unwrap();
            let mut buf = vec![0u8; 8192];
            let mut n = 0;
            while !buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
                let k = s.read(&mut buf[n..]).unwrap();
                if k == 0 { break; }
                n += k;
            }
            let path = String::from_utf8_lossy(&buf[..n]).split(' ').nth(1).unwrap_or("").to_string();
            let q: Vec<(&'static str, String)> = path.split_once('?').unwrap().1.split('&').map(|kv| {
                let (k, v) = kv.split_once('=').unwrap();
                (Box::leak(k.to_string().into_boxed_str()) as &'static str, v.to_string())
            }).collect();
            let t = *nn.lock().unwrap();
            let body = Fake { ser: &ser, now: t, vis: t, reqs: vec![] }.get(path.split('?').next().unwrap(), &q).unwrap().to_string();
            let _ = s.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes());
        }
    });
    let out = dir.join("out");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::copy(d.join("regime_band.json"), out.join("regime_band.json")).unwrap();
    let run = |t: i64| {
        *now.lock().unwrap() = t;
        std::process::Command::new(env!("CARGO_BIN_EXE_rt_regime")).env("RT_REGIME_OUT", &out).env("RT_REGIME_REST", format!("http://127.0.0.1:{port}"))
            .env("RT_SR_BAN_FILE", dir.join("ban")).env("RT_REGIME_NOW_MS", t.to_string()).output().unwrap()
    };
    let runs = fx["runs"].as_array().unwrap();
    for r in &runs[..2] {
        let o = run(r["now"].as_i64().unwrap());
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        assert_eq!(String::from_utf8_lossy(&o.stdout).trim_end(), r["out"].as_str().unwrap());
        for (k, f) in [("log", "regime_band_forward.jsonl"), ("scored", "regime_band_forward_scored.jsonl"), ("summary", "regime_band_forward_summary.json")] {
            assert_eq!(std::fs::read_to_string(out.join(f)).ok().as_deref(), r["files"][k].as_str(), "{k}");
        }
    }
    let last = runs.last().unwrap();
    assert!(last.get("fail").is_some());
    let o = run(last["now"].as_i64().unwrap());
    assert_eq!(o.status.code(), Some(1), "묵은 자료 → FAIL");
    assert!(String::from_utf8_lossy(&o.stderr).starts_with("FAIL AssertionError"));
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_rt_regime")).env("RT_REGIME_OUT", dir.join("data/live")).output().unwrap();
    assert_eq!(o.status.code(), Some(2), "파이썬 기록 자리 거부");
    let _ = std::fs::remove_dir_all(&dir);
    println!("실행 파일 2회차 출력·파일 글자 같음 · FAIL 1 · 거부 2");
}
