//! 정답지 입력(또는 부하 입력)을 Rust 수집기에 먹여 SQLite 를 만든다. tools/gen_tape_fixture.py 가 부른다.
//!   cargo run --release --example replay -- <input.json> <out_dir> [--runs A:B] [--bench]
//! --bench: 한 판(첫 실행)만, 메시지 처리·쓰기 시간과 «수신 → 디스크» 지연(5초 플러시 정책 + 커밋 시간)을 잰다.
use rtcollect::binance::{replay, Collector};
use rtcollect::store::open_db;
use serde_json::Value;
use std::collections::VecDeque;
use std::path::Path;
use std::time::Instant;

fn cpu_s() -> f64 {
    // /proc/self/stat 의 utime+stime(14·15번째, 클록 틱 100Hz 가정)
    let s = std::fs::read_to_string("/proc/self/stat").unwrap();
    let f: Vec<&str> = s.rsplit(')').next().unwrap().split_whitespace().collect();
    (f[11].parse::<f64>().unwrap() + f[12].parse::<f64>().unwrap()) / 100.0
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let input: Value = serde_json::from_str(&std::fs::read_to_string(&a[1]).unwrap()).unwrap();
    let out = Path::new(&a[2]);
    std::fs::create_dir_all(out).unwrap();
    let mut runs = input["runs"].as_array().unwrap().clone();
    if let Some(i) = a.iter().position(|x| x == "--runs") {
        let (lo, hi) = a[i + 1].split_once(':').unwrap();
        let hi = if hi.is_empty() { runs.len() } else { hi.parse().unwrap() };
        runs = runs[lo.parse().unwrap()..hi].to_vec();
    }
    if !a.iter().any(|x| x == "--bench") {
        replay(&runs, out, |_, _| {}).unwrap();
        return;
    }
    let run = &runs[0];
    let ev = run["events"].as_array().unwrap();
    let con = open_db(&out.join("binance_tape.sqlite")).unwrap();
    let (c0, w0) = (cpu_s(), Instant::now());
    let mut c = Collector::new(con, "ethusdt", "futures", run["t0"].as_f64().unwrap()).unwrap();
    c.on_connect();
    let (mut proc_s, mut writes, mut lat) = (0.0, Vec::new(), Vec::new());
    let mut q: VecDeque<(f64, i64)> = VecDeque::new();
    let mut max_sec = 0i64;
    let mut n_msg = 0usize;
    for e in ev {
        if e[1] != "t" {
            let t = Instant::now();
            c.on_disconnect();
            writes.push(t.elapsed().as_secs_f64());
            continue;
        }
        let (now, text) = (e[0].as_f64().unwrap(), e[2].as_str().unwrap());
        n_msg += 1;
        // 지연 장부(시간 밖): 이 메시지의 초
        let sec = serde_json::from_str::<Value>(text).unwrap()["T"].as_i64().unwrap().div_euclid(1000);
        max_sec = max_sec.max(sec);
        q.push_back((now, sec));
        let t = Instant::now();
        let r = c.on_text(now, text).unwrap();
        let dt = t.elapsed().as_secs_f64();
        if r.is_some() {
            writes.push(dt);
            // 이 플러시에 실린 메시지 = 진행 중인 초(max_sec)보다 앞 초. 지연 = 플러시 시각 − 수신 + 커밋 벽시간
            q = q
                .into_iter()
                .filter(|&(r0, s)| {
                    if s < max_sec {
                        lat.push(now - r0 + dt);
                        false
                    } else {
                        true
                    }
                })
                .collect();
        } else {
            proc_s += dt;
        }
    }
    let (c1, w1) = (cpu_s(), w0.elapsed().as_secs_f64());
    let wsum: f64 = writes.iter().sum();
    let nw = writes.len();
    println!(
        "메시지 {n_msg} · 벽 {w1:.3}s · CPU {:.3}s · 메시지당 CPU {:.2}µs · 쓰기 {nw}회 중앙 {:.2}ms 최대 {:.2}ms · \
         쓰기 제외 메시지당 벽 {:.2}µs (on_text 만 {:.2}µs) · 수신→디스크 중앙 {:.3}s 상위1% {:.3}s 최대 {:.3}s (5초 플러시 정책 포함)",
        c1 - c0,
        (c1 - c0) / n_msg as f64 * 1e6,
        pct(&mut writes, 0.5) * 1e3,
        pct(&mut writes, 1.0) * 1e3,
        (w1 - wsum) / n_msg as f64 * 1e6,
        proc_s / n_msg as f64 * 1e6,
        pct(&mut lat, 0.5),
        pct(&mut lat, 0.99),
        pct(&mut lat, 1.0),
    );
}
