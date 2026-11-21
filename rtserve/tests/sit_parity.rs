//! 상황 카드 통합(일곱째 조각): 코어가 미는 꼴 그대로(Live::apply 에 situation · micro(flow·streams·vol60·liq60) 메시지) → hub::situation_once →
//! HTTP `/api/situation` 본문 · `/api/stream` 의 `event: situation` 줄 == 원본(tools/gen_sit_fixture.py · 서버 판 venv · 같은 입력 · 빠진 재료는 원본 쪽도 «없음»).
//! 풋프린트 봉 칸 → Micro.flow 는 rtcore 의 flow_nets_tick·flow_delta(코어가 실제로 쓰는 함수)로 만든다. 양성 대조: 한 칸을 틀리게 넣으면 다르다.

mod common;

use rtcore::situation::{flow_delta, flow_nets_tick, FlowNets};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

type Cells = Vec<[f64; 6]>;

fn bars_of(v: &Value) -> BTreeMap<i64, Cells> {
    v.as_array().unwrap().iter().map(|b| (b[0].as_i64().unwrap(), b[1].as_array().unwrap().iter().map(|c| std::array::from_fn(|j| c[j].as_f64().unwrap())).collect())).collect()
}

fn msg(kind: &str, body: Value) -> String {
    let mut m = json!({"s": "ethusdt", "seq": 1, "ex_ms": 0, "rx_us": 0, "tx_us": 0, "type": kind});
    m.as_object_mut().unwrap().extend(body.as_object().unwrap().clone());
    m.to_string()
}

#[test]
fn situation_body_and_sse_match_python() {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gen/sit/sit.json");
    let steps: Value = serde_json::from_str(&std::fs::read_to_string(&p).expect("서버 판 venv 로 tools/gen_sit_fixture.py 먼저")).unwrap();
    let clock = Arc::new(AtomicU64::new(0));
    let held: Arc<Mutex<Option<rtserve::Config>>> = Default::default();
    let (c2, h2) = (clock.clone(), held.clone());
    let (addr, root) = common::start_with(&serde_json::Map::new(), 0, 0.0, "eth,sol,xrp", "sit", move |cfg| {
        cfg.now = Arc::new(move || f64::from_bits(c2.load(Ordering::SeqCst)));
        *h2.lock().unwrap() = Some(cfg.clone());
    });
    let cfg = held.lock().unwrap().clone().unwrap();
    // SSE 하나를 열어 둔다(supply 없음 = 상황만)
    let mut sse = std::net::TcpStream::connect(addr).unwrap();
    sse.write_all(b"GET /api/stream HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
    sse.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
    let mut sse_buf = Vec::new();
    let mut run = rtserve::hub::SitRun::default();
    let (mut core, mut sent): (FlowNets, Option<FlowNets>) = Default::default();
    let (mut n, mut bad) = (0, Vec::new());
    for (k, s) in steps.as_array().unwrap().iter().enumerate() {
        let now = s["now"].as_f64().unwrap();
        clock.store(now.to_bits(), Ordering::SeqCst);
        flow_nets_tick(&mut core, now, &bars_of(&s["fbars"]), s["okx_first_bar"].as_i64().unwrap(), &bars_of(&s["okx_bars"]), |c: &Cells| c.clone());
        let d = flow_delta(&core, sent.as_ref());
        sent = Some(core.clone());
        let mut core_sit = s["core"].clone();
        core_sit.as_object_mut().unwrap().insert("missing".into(), json!([])); // 코어는 받는 쪽이 채울 칸 이름을 싣는다(patch_now 가 뺀다)
        core_sit.as_object_mut().unwrap().insert("computed_at".into(), json!(now - 0.5)); // 코어 시계(받는 쪽이 자기 시계로 덮는다)
        let m = msg("situation", core_sit);
        if let Err(e) = serde_json::from_str::<rtcore::proto::Stamped>(&m) {
            panic!("#{k} 코어 Situation 꼴로 안 읽힘: {e}\n{}", &m[..m.len().min(600)]);
        }
        cfg.live.apply(&m);
        cfg.live.apply(&msg("micro", json!({"ts": now as i64, "mid": null, "spread": null, "qi": null, "vol60": 0.0, "liq60": {"long": 0.0, "short": 0.0, "n": 0},
                                             "streams": s["streams"], "flow": serde_json::to_value(&d).unwrap()})));
        rtserve::hub::situation_once(&cfg, &mut run);
        let g = common::request(addr, "GET", "/api/situation", &[]);
        let (mine, want) = (String::from_utf8_lossy(&g.body).into_owned(), s["body"].as_str().unwrap().to_string());
        n += 1;
        if mine != want {
            let i = mine.bytes().zip(want.bytes()).position(|(a, b)| a != b).unwrap_or(mine.len().min(want.len()));
            bad.push(format!("#{k} 본문 글자 {i}: rust …{} / py …{}", &mine[i.saturating_sub(60)..(i + 80).min(mine.len())], &want[i.saturating_sub(60)..(i + 80).min(want.len())]));
        }
        // SSE: computed_at 이 바뀌었으니 이 걸음의 줄이 온다
        let line = s["sse"].as_str().unwrap();
        let t0 = std::time::Instant::now();
        while !String::from_utf8_lossy(&sse_buf).contains(line) && t0.elapsed().as_secs() < 3 {
            let mut b = [0u8; 65536];
            if let Ok(m) = sse.read(&mut b) {
                sse_buf.extend_from_slice(&b[..m]);
            }
        }
        if !String::from_utf8_lossy(&sse_buf).contains(line) {
            bad.push(format!("#{k} SSE 줄이 안 옴(또는 다름)"));
        }
        sse_buf.clear();
    }
    // 양성 대조: 코어 Situation 의 evidence 한 칸(dir 또는 ok)을 바꾸면 본문이 달라야 한다
    let last = steps.as_array().unwrap().last().unwrap();
    let mut wrong = last["core"].clone();
    if wrong["ok"] == json!(true) {
        wrong["regime"]["margin"] = json!(wrong["regime"]["margin"].as_f64().unwrap() + 0.001); // 한 칸만 틀리게
    } else {
        wrong["reason"] = json!("x");
    }
    wrong.as_object_mut().unwrap().insert("missing".into(), json!([]));
    wrong.as_object_mut().unwrap().insert("computed_at".into(), json!(1.0));
    assert!(serde_json::from_str::<rtcore::proto::Stamped>(&msg("situation", wrong.clone())).is_ok(), "양성 대조 메시지가 읽혀야 한다");
    cfg.live.apply(&msg("situation", wrong));
    rtserve::hub::situation_once(&cfg, &mut run);
    let g = common::request(addr, "GET", "/api/situation", &[]);
    assert_ne!(String::from_utf8_lossy(&g.body), last["body"].as_str().unwrap(), "양성 대조: 틀린 코어 값이 같은 본문을 내면 시험이 아무것도 안 보는 것");
    println!("상황 카드 통합: 걸음 {n} · 본문·SSE 다름 {}", bad.len());
    for b in bad.iter().take(5) {
        println!("  {b}");
    }
    assert!(bad.is_empty());
    let _ = std::fs::remove_dir_all(root);
}
