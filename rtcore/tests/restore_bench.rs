//! 재시작 복원 시간·메모리(2026-11-21 · 무시 시험). 합성 24시간 테이프 = tools/gen_bench_tape.py(8코인 × 코인마다 약 32만 행).
//!   RTCORE_BENCH_TAPE=<테이프> /usr/bin/time -v cargo test --release --test restore_bench -- --ignored --nocapture
//! 찬 시작(스냅숏 없음 → 창 12시간을 테이프로) 8코인을 동시에 · 그다음 스냅숏을 쓰고 10초 공백 재시작(따뜻한 시작)을 잰다.
use rtcore::proto::{Event, Stamped};
use rtcore::restore::Cfg;
use rtcore::state::State;
use std::time::{Duration, Instant};

const END: i64 = 1_763_683_200; // gen_bench_tape.py END
const SYMS: [&str; 8] = ["ethusdt", "btcusdt", "solusdt", "xrpusdt", "sndkusdt", "soxlusdt", "skhynixusdt", "muusdt"];

fn first_trade(st: &State, sym: &str, ms: i64) {
    st.apply(Stamped { s: sym.into(), seq: 0, ex_ms: ms, rx_us: ms * 1000, tx_us: 0, ev: Event::Trade { id: 1, px: 100.0, qty: 1.0, sell: false } });
}

fn round(st: &std::sync::Arc<State>, syms: &[String]) -> Duration {
    let t = Instant::now();
    std::thread::scope(|s| {
        for sym in syms {
            let st = st.clone();
            s.spawn(move || {
                let t = Instant::now();
                let line = st.run_restore(sym).expect("일감");
                println!("  {sym}: {:.0}ms · {}", t.elapsed().as_secs_f64() * 1e3, line.chars().take(110).collect::<String>());
            });
        }
    });
    t.elapsed()
}

#[test]
#[ignore]
fn restore_bench() {
    let tape = std::env::var("RTCORE_BENCH_TAPE").expect("RTCORE_BENCH_TAPE=<gen_bench_tape.py 출력>");
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("restore_bench_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let syms: Vec<String> = SYMS.iter().map(|s| s.to_string()).collect();
    let cfg = || Cfg { snap_dir: dir.clone(), tape: tape.clone().into(), tries: 1, wait: Duration::ZERO, ..nowhere() };
    // 찬 시작
    let (st, _) = State::with_restore(&syms, 16, 1 << 16, cfg(), END * 1000 * 1000);
    for s in &syms {
        first_trade(&st, s, END * 1000 + 300);
    }
    let cold = round(&st, &syms);
    let t = Instant::now();
    let n = st.write_snapshots();
    let w = t.elapsed();
    let bytes: u64 = syms.iter().map(|s| std::fs::metadata(dir.join(format!("{s}.json"))).map_or(0, |m| m.len())).sum();
    println!("찬 시작 8코인 동시 {:.0}ms · 스냅숏 {n}개 쓰기 {:.0}ms · 합계 {}KB", cold.as_secs_f64() * 1e3, w.as_secs_f64() * 1e3, bytes / 1024);
    // 따뜻한 시작: 스냅숏(끝 초까지) + 10초 공백 -- 테이프는 끝 초까지라 공백 = 테이프에 없는 미래(구멍 없이 빈 초)
    let t = Instant::now();
    let (st2, _) = State::with_restore(&syms, 16, 1 << 16, cfg(), (END + 10) * 1000 * 1000);
    let load = t.elapsed();
    for s in &syms {
        first_trade(&st2, s, (END + 10) * 1000 + 300);
    }
    let warm = round(&st2, &syms);
    println!("따뜻한 시작: 스냅숏 8개 읽기 {:.0}ms · 메우기 8코인 동시 {:.0}ms", load.as_secs_f64() * 1e3, warm.as_secs_f64() * 1e3);
    let _ = std::fs::remove_dir_all(&dir);
}

/// 나머지 원천은 없는 파일(읽기 전용 열기가 실패해 그 원천만 빈다 -- 만들지 않는다).
fn nowhere() -> Cfg {
    let n = std::path::PathBuf::from("/rtcore_없는_경로/x.sqlite");
    Cfg { snap_dir: n.clone(), tape: n.clone(), tries: 1, wait: Duration::ZERO, okx_tape: n.clone(), okx_ctx: n.clone(), bn_ctx: n.clone(), bybit_liq: n }
}
