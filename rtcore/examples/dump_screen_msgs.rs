//! 화면 «rt 로 그리기» node 시험(test/test_rt_draw_20261121.mjs)의 입력: 진짜 코어(State)에 rtcore 정답지 입력을 넣어 허브가 내보내는 **실제 메시지**를 적는다.
//! 정답(옛 경로 = 원본 /api/supply-1s · /api/footprint 응답)은 같은 정답지에 이미 있다 -- 함께 싣는다.
//! 2026-11-22 바이낸스만 판: 원본 /api/footprint 는 바이낸스 선물 체결만이다(server.py api_footprint 4000~4005 «11-02 바이낸스만» · venues [binance-perp]).
//!   풋프린트 몫 = footprint_eth.json(원본 collect_footprint 를 돌린 정답지 · footprint_parity.rs 와 같은 재생) 끝의 since=0 전량.
//!   cargo run --offline -j 2 --example dump_screen_msgs -- <출력 경로>   # 기본 target/rt_screen_msgs.json (화면 쪽 test/fixtures 는 그쪽에서 바꾼다)
use rtcore::state::State;
use serde_json::{json, Value};

fn drain(rx: &mut tokio::sync::broadcast::Receiver<rtcore::proto::Stamped>, out: &mut Vec<Value>, keep: &[&str]) {
    while let Ok(m) = rx.try_recv() {
        let v = serde_json::to_value(&m).unwrap();
        if keep.iter().any(|k| v["type"] == *k) {
            out.push(v);
        }
    }
}

fn trade_text(r: &[Value], sym: &str, t0: i64) -> (String, i64) {
    let (kind, t_ms, tid, p, q, m, w) = (&r[1], r[2].as_i64().unwrap() + t0, &r[3], &r[4], &r[5], &r[6], r[7].as_i64().unwrap() + t0);
    if kind.as_i64() == Some(1) {
        return (json!({"e": "markPriceUpdate", "E": t_ms, "s": sym}).to_string(), w);
    }
    let x = if p.as_str() == Some("0") { "NA" } else { "MARKET" };
    (json!({"e": "trade", "E": w, "T": t_ms, "s": sym, "t": tid, "p": p, "q": q, "X": x, "m": m}).to_string(), w)
}

/// 정답지 행을 체결 글자로 넣는다(접속마다 reset_orders). `flush_every` 행마다 밀기 묶음.
fn replay(fx: &Value, t0: i64, flush: impl Fn(usize) -> bool, keep: &[&str]) -> (Value, Vec<Value>, f64) {
    let sym = fx["symbol"].as_str().unwrap().to_string();
    let st = State::new(std::slice::from_ref(&sym), 16, 1 << 20);
    let mut rx = st.tx.subscribe();
    let snap = serde_json::to_value(st.snapshot(&sym, 0).unwrap()).unwrap();
    let (mut conn, mut msgs, mut last_px) = (0, vec![], 0.0);
    for (i, r) in fx["rows"].as_array().unwrap().iter().enumerate() {
        let r = r.as_array().unwrap();
        let c = r[0].as_i64().unwrap();
        if c != conn {
            st.reset_orders(&sym);
            conn = c;
        }
        let (txt, w) = trade_text(r, &sym.to_uppercase(), t0);
        if let Some(m) = rtcore::feed::parse(&txt, w * 1000) {
            if let rtcore::proto::Event::Trade { px, .. } = m.ev {
                last_px = px;
            }
            st.apply(m);
        }
        if flush(i) {
            st.flush();
            st.flush_prices();
        }
        drain(&mut rx, &mut msgs, keep);
    }
    st.flush();
    st.flush_prices();
    drain(&mut rx, &mut msgs, keep);
    (snap, msgs, last_px)
}

fn main() {
    let sfx: Value = serde_json::from_str(include_str!("../tests/fixtures/supply1s_eth.json")).unwrap();
    let (s_snap, s_msgs, _) = replay(&sfx, 0, |_| true, &["supply1s"]);
    let ffx: Value = serde_json::from_str(include_str!("../tests/fixtures/footprint_eth.json")).unwrap();
    let t0 = ffx["t0"].as_i64().unwrap();
    let (f_snap, f_msgs, last_px) = replay(&ffx, t0, |i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 60 < 3, &["footprint", "prices"]);
    // 원본 since=0 전량(바이낸스만) -- [봉, 칸] · 이어받기 = 마지막 봉 − 300 이후
    let full: Vec<Value> = ffx["final"]["bars"].as_array().unwrap().iter().map(|b| json!([b["time"], b["levels"]])).collect();
    let since = full.last().map_or(0, |b| b[0].as_i64().unwrap()) - 300;
    let inc: Vec<Value> = full.iter().filter(|b| b[0].as_i64().unwrap() >= since).cloned().collect();
    let out = json!({
        "about": "rtcore 허브 실제 메시지(examples/dump_screen_msgs.rs · 2026-11-22 바이낸스만 판) + 같은 입력의 원본 응답(rtcore 정답지 supply1s_eth · footprint_eth)",
        "supply": {"symbol": sfx["symbol"], "snapshot": s_snap, "msgs": s_msgs, "want": sfx["final"]},
        "footprint": {"symbol": ffx["symbol"], "snapshot": f_snap,
                      "steps": [{"msgs": f_msgs, "full": full, "since": since, "inc": inc, "last_px": last_px}]},
    });
    let p = std::env::args().nth(1).unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/target/rt_screen_msgs.json").to_string());
    std::fs::write(&p, serde_json::to_string(&out).unwrap()).unwrap();
    println!("{p}: 수급 메시지 {} · 풋프린트 메시지 {} · 원본 전량 봉 {}", s_msgs.len(), f_msgs.len(), full.len());
}
