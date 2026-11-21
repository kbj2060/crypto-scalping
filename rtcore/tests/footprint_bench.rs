//! 풋프린트·1초 수급 밀기 속도와 메시지 양(개발기 수치 -- 실서버 아님). 거래소·소켓 없이 State 만 돈다.
//!   cargo test --release -j 2 --test footprint_bench -- --ignored --nocapture
//! 144봉(봉마다 체결 1,500건)을 먼저 채우고, 초당 3,000건을 실제 시간으로 넣으며 묶음 간격별로 잰다:
//! «체결 반영 → 그 체결이 실린 묶음의 JSON 직렬화 끝»(체결마다 · 허브 소켓 쓰기 전까지) · 메시지 수·바이트/초 · flush 한 번의 시간.
use rtcore::proto::Event;
use rtcore::state::State;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const RATE: u64 = 3000;
const SECS: u64 = 5;

struct Gen {
    tid: u64,
    px: f64,
    x: u64,
}

impl Gen {
    fn rnd(&mut self) -> f64 {
        self.x ^= self.x << 13;
        self.x ^= self.x >> 7;
        self.x ^= self.x << 17;
        (self.x >> 11) as f64 / (1u64 << 53) as f64
    }

    fn trade(&mut self, t_ms: i64) -> String {
        self.tid += 1;
        self.px = (self.px + (self.rnd() - 0.5) * 0.4).clamp(2990.0, 3010.0);
        let q = if self.rnd() < 0.9 { 0.001 + self.rnd() * 2.0 } else { 5.0 + self.rnd() * 40.0 };
        let m = self.rnd() < 0.5;
        format!(r#"{{"e":"trade","E":{t_ms},"T":{t_ms},"s":"ETHUSDT","t":{},"p":"{:.2}","q":"{:.3}","X":"MARKET","m":{m}}}"#, self.tid, self.px, q)
    }
}

fn pct(v: &mut [i64]) -> String {
    if v.is_empty() {
        return "-".into();
    }
    v.sort_unstable();
    let n = v.len();
    format!("중앙 {}µs · 상위1% {}µs · 최대 {}µs", v[n / 2], v[(n * 99 / 100).min(n - 1)], v[n - 1])
}

#[test]
#[ignore]
fn bench_flush_intervals() {
    let sym = "ethusdt".to_string();
    let mut g = Gen { tid: 1, px: 3000.0, x: 0x9E37_79B9_7F4A_7C15 };
    println!("묶음 간격 | 체결 반영→직렬화 끝(체결마다) | 풋프린트 메시지/초 · 바이트/초 | 수급 메시지/초 · 바이트/초 | flush 1번");
    for interval_ms in [0u64, 16, 50, 100] {
        let st = State::new(std::slice::from_ref(&sym), 16, 1 << 20);
        // 지난 144봉(12시간)을 먼저 -- 수신 시각도 그때로(24시간 창).
        let now_ms = rtcore::now_us() / 1000;
        let t0 = (now_ms / 1000 / 300 - 144) * 300 * 1000;
        for k in 0..144 * 1500i64 {
            let t = t0 + k * 200;
            st.apply(rtcore::feed::parse(&g.trade(t), t * 1000).unwrap());
        }
        st.flush();
        let snap = serde_json::to_string(&st.snapshot(&sym, 0).unwrap()).unwrap();
        let Event::Snapshot { footprint: Some(fp), .. } = st.snapshot(&sym, 0).unwrap().ev else { panic!() };
        let fp_bytes = serde_json::to_string(&fp).unwrap().len();
        let levels: usize = fp.bars.iter().map(|b| b.levels.len()).sum();
        if interval_ms == 0 {
            println!("스냅숏: 풋프린트 {}봉 · 칸 {levels} · {}KB(전체 스냅숏 {}KB)", fp.bars.len(), fp_bytes / 1024, snap.len() / 1024);
        }

        let mut rx = st.tx.subscribe();
        let done = Arc::new(AtomicBool::new(false));
        let applied: Arc<Mutex<Vec<(i64, i64)>>> = Arc::default(); // (수신 µs, 반영 끝 µs)
        let flushes: Arc<Mutex<Vec<(i64, i64)>>> = Arc::default(); // (flush 시작 µs, 직렬화 끝 µs)
        let (mut fp_n, mut fp_b, mut su_n, mut su_b, mut cost) = (0usize, 0usize, 0usize, 0usize, vec![]);
        let mut serialize = |rx: &mut tokio::sync::broadcast::Receiver<rtcore::proto::Stamped>| {
            while let Ok(m) = rx.try_recv() {
                let n = serde_json::to_string(&m).unwrap().len();
                match m.ev {
                    Event::Footprint(_) => (fp_n, fp_b) = (fp_n + 1, fp_b + n),
                    Event::Supply1s(_) => (su_n, su_b) = (su_n + 1, su_b + n),
                    _ => {}
                }
            }
        };
        let prod = {
            let (st, done, applied, flushes) = (st.clone(), done.clone(), applied.clone(), flushes.clone());
            let mut g = Gen { tid: g.tid + 1, px: g.px, x: g.x };
            std::thread::spawn(move || {
                let start = Instant::now();
                let mut sent = 0u64;
                let mut rx2 = st.tx.subscribe(); // 사건마다 모드: 여기서 바로 직렬화
                while start.elapsed() < Duration::from_secs(SECS) {
                    let due = start.elapsed().as_micros() as u64 * RATE / 1_000_000;
                    while sent < due {
                        let rx_us = rtcore::now_us();
                        st.apply(rtcore::feed::parse(&g.trade(rx_us / 1000), rx_us).unwrap());
                        let a = rtcore::now_us();
                        applied.lock().unwrap().push((rx_us, a));
                        if interval_ms == 0 {
                            st.flush();
                            while let Ok(m) = rx2.try_recv() {
                                let _ = serde_json::to_string(&m).unwrap();
                            }
                            flushes.lock().unwrap().push((a, rtcore::now_us()));
                        }
                        sent += 1;
                    }
                    std::thread::sleep(Duration::from_micros(200));
                }
                done.store(true, Relaxed);
            })
        };
        if interval_ms == 0 {
            prod.join().unwrap();
            serialize(&mut rx);
        } else {
            while !done.load(Relaxed) {
                std::thread::sleep(Duration::from_millis(interval_ms));
                let f0 = rtcore::now_us();
                st.flush();
                cost.push(rtcore::now_us() - f0);
                serialize(&mut rx);
                flushes.lock().unwrap().push((f0, rtcore::now_us()));
            }
            prod.join().unwrap();
        }
        // 체결마다: 반영이 끝난 뒤 시작한 첫 flush 의 직렬화 끝 − 수신
        let fl = flushes.lock().unwrap();
        let mut lat: Vec<i64> = applied
            .lock()
            .unwrap()
            .iter()
            .filter_map(|&(rx_us, a)| fl.iter().find(|f| f.0 >= a).map(|f| f.1 - rx_us))
            .collect();
        let n_tr = applied.lock().unwrap().len();
        let s = SECS as f64;
        println!(
            "{:>4}ms | {} (체결 {n_tr}) | {:.0}건 · {:.1}KB | {:.0}건 · {:.1}KB | {}",
            interval_ms,
            pct(&mut lat),
            fp_n as f64 / s,
            fp_b as f64 / s / 1024.0,
            su_n as f64 / s,
            su_b as f64 / s / 1024.0,
            if cost.is_empty() { "-".into() } else { pct(&mut cost) }
        );
    }
}
