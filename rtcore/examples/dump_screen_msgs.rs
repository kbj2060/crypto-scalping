//! 화면 «rt 로 그리기» node 시험(test/test_rt_draw_20261121.mjs)의 입력: 진짜 코어(State)에 rtcore 정답지 입력을 넣어 허브가 내보내는 **실제 메시지**를 적는다.
//! 정답(옛 경로 = 원본 /api/supply-1s · /api/footprint 응답)은 같은 정답지에 이미 있다 -- 함께 싣는다.
//!   cargo run --offline -j 2 --example dump_screen_msgs     # rtcore 에서 → ../test/fixtures/rt_screen_msgs.json
use rtcore::proto::{Event, Stamped};
use rtcore::state::State;
use serde_json::{json, Value};

fn drain(rx: &mut tokio::sync::broadcast::Receiver<Stamped>, out: &mut Vec<Value>) {
    while let Ok(m) = rx.try_recv() {
        out.push(serde_json::to_value(&m).unwrap());
    }
}

fn supply() -> Value {
    // tests/supply1s_parity.rs 와 같은 재생(접속마다 reset_orders · 체결 글자 → feed::parse)
    let fx: Value = serde_json::from_str(include_str!("../tests/fixtures/supply1s_eth.json")).unwrap();
    let sym = fx["symbol"].as_str().unwrap().to_string();
    let st = State::new(std::slice::from_ref(&sym), 16, 1 << 20);
    let mut rx = st.tx.subscribe();
    let snap = serde_json::to_value(st.snapshot(&sym, 0).unwrap()).unwrap();
    let (mut conn, mut msgs) = (0, vec![]);
    for r in fx["rows"].as_array().unwrap() {
        let r = r.as_array().unwrap();
        let c = r[0].as_i64().unwrap();
        if c != conn {
            st.reset_orders(&sym);
            conn = c;
        }
        let (kind, t_ms, tid, p, q, m, w) = (&r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
        let txt = if kind.as_i64() == Some(1) {
            json!({"e": "markPriceUpdate", "E": t_ms, "s": sym.to_uppercase()}).to_string()
        } else {
            let x = if p.as_str() == Some("0") { "NA" } else { "MARKET" };
            json!({"e": "trade", "E": w, "T": t_ms, "s": sym.to_uppercase(), "t": tid, "p": p, "q": q, "X": x, "m": m}).to_string()
        };
        if let Some(m) = rtcore::feed::parse(&txt, w.as_i64().unwrap() * 1000) {
            st.apply(m);
        }
        st.flush();
        drain(&mut rx, &mut msgs);
    }
    msgs.retain(|m| m["type"] == "supply1s");
    json!({"symbol": sym, "snapshot": snap, "msgs": msgs, "want": fx["final"]})
}

fn footprint() -> Value {
    // tests/fp_okx_parity.rs 와 같은 재생 · 대조 시점마다 그때까지 받은 메시지와 원본 응답(전량·이어받기)
    let fx: Value = serde_json::from_str(include_str!("../tests/fixtures/fp_okx_sol.json")).unwrap();
    let sym = fx["symbol"].as_str().unwrap().to_string();
    let st = State::new(std::slice::from_ref(&sym), 16, 1 << 20);
    let mut rx = st.tx.subscribe();
    let snap = serde_json::to_value(st.snapshot(&sym, 0).unwrap()).unwrap();
    let evs = fx["events"].as_array().unwrap();
    let (mut ei, mut steps, mut last_px) = (0usize, vec![], 0.0);
    for cp in fx["checkpoints"].as_array().unwrap() {
        let upto = cp["after"].as_u64().unwrap() as usize;
        let mut msgs = vec![];
        while ei < upto {
            let e = &evs[ei];
            let ms = e[1].as_i64().unwrap();
            if e[0] == "bn" {
                last_px = e[3].as_f64().unwrap();
                let ev = Event::Trade { id: e[2].as_u64().unwrap(), px: last_px, qty: e[4].as_f64().unwrap(), sell: e[5].as_bool().unwrap() };
                st.apply(Stamped { s: sym.clone(), seq: 0, ex_ms: ms, rx_us: ms * 1000, tx_us: 0, ev });
            } else {
                for x in rtcore::feed::parse_okx(e[2].as_str().unwrap(), ms * 1000) {
                    st.apply(x);
                }
            }
            ei += 1;
            if ei % 5 == 0 {
                st.flush();
                st.flush_prices();
            }
        }
        st.flush();
        st.flush_prices();
        drain(&mut rx, &mut msgs);
        msgs.retain(|m| m["type"] == "footprint" || m["type"] == "prices");
        steps.push(json!({"msgs": msgs, "full": cp["full"], "since": cp["since"], "inc": cp["inc"], "last_px": last_px}));
    }
    json!({"symbol": sym, "snapshot": snap, "steps": steps})
}

fn main() {
    let out = json!({"about": "rtcore 허브 실제 메시지(examples/dump_screen_msgs.rs) + 같은 입력의 원본 응답(rtcore 정답지 supply1s_eth · fp_okx_sol)",
                     "supply": supply(), "footprint": footprint()});
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../test/fixtures/rt_screen_msgs.json");
    std::fs::write(p, serde_json::to_string(&out).unwrap()).unwrap();
    let s = &out["supply"]["msgs"];
    let f = out["footprint"]["steps"].as_array().unwrap();
    println!("{p}: 수급 메시지 {} · 풋프린트 대조 {} · 메시지 {}", s.as_array().unwrap().len(), f.len(), f.iter().map(|x| x["msgs"].as_array().unwrap().len()).sum::<usize>());
}
