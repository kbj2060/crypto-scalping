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

type Rest = Arc<Mutex<(String, Vec<String>)>>;
const DAY5_URL: &str = "/fapi/v1/klines?symbol=ETHUSDT&interval=5m&limit=300";

/// 127.0.0.1 가짜 REST: 5분봉 하루 URL 만 지금 본문 · 그 밖 404(청산맵 레벨 → 실패 갈래 = 레벨 없음) · 받은 경로를 적는다
fn fake_rest(st: Rest) -> String {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for s in l.incoming() {
            let Ok(mut s) = s else { continue };
            let (mut buf, mut n) = (vec![0u8; 65536], 0);
            while !buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
                match s.read(&mut buf[n..]) {
                    Ok(0) | Err(_) => break,
                    Ok(k) => n += k,
                }
            }
            let target = String::from_utf8_lossy(&buf[..n]).split(' ').nth(1).unwrap_or("").to_string();
            let mut g = st.lock().unwrap();
            g.1.push(target.clone());
            let (code, body) = if target == DAY5_URL { (200, g.0.clone()) } else { (404, "[]".to_string()) };
            drop(g);
            let _ = s.write_all(format!("HTTP/1.1 {code} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes());
        }
    });
    format!("http://127.0.0.1:{port}")
}

/// 둘째 판(2026-11-28): inp.bars(move60·oi60 = 풋프린트 봉 · candles_day 종가 · hot oi_1s) + candles_day(move60_p75·range30_pct·vol_mult) --
/// 가짜 REST 의 klines 300 → swr 60/600·식힘(원본 load_5m_day) → /api/situation 본문 · SSE 줄 · x 가 원본(tools/gen_sit_fixture.py main_bars)과 글자 일치 ·
/// 걸음마다 klines 를 보낸 횟수도 원본 swr 흉내와 같다. 양성 대조: 오답 종가 한 봉이면 x 가 달라야 한다.
#[test]
fn situation_bars_day5_match_python() {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gen/sit/sit_bars.json");
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(&p).expect("tools/gen_sit_fixture.py 먼저")).unwrap();
    let clock = Arc::new(AtomicU64::new(0));
    let rest: Rest = Default::default();
    let held: Arc<Mutex<Option<rtserve::Config>>> = Default::default();
    let (c2, h2, url) = (clock.clone(), held.clone(), fake_rest(rest.clone()));
    let mut files = serde_json::Map::new();
    files.insert("data/hot/binance_ctx.sqlite".into(), fx["oi_db"].clone());
    let (addr, root) = common::start_with(&files, 0, 0.0, "eth,sol,xrp", "sitbars", move |cfg| {
        let c3 = c2.clone();
        cfg.now = Arc::new(move || f64::from_bits(c3.load(Ordering::SeqCst)));
        let mut sw = rtserve::swr::Swr::default();
        let c4 = c2.clone();
        sw.mono = Arc::new(move || f64::from_bits(c4.load(Ordering::SeqCst))); // swr 나이도 같은 가짜 시계(원본 흉내와 같은 걸음)
        cfg.swr = Arc::new(sw);
        cfg.rest = url;
        *h2.lock().unwrap() = Some(cfg.clone());
    });
    let cfg = held.lock().unwrap().clone().unwrap();
    let mut run = rtserve::hub::SitRun { rest_at: Some(0.0), ..Default::default() };
    let (mut core, mut sent): (FlowNets, Option<FlowNets>) = Default::default();
    let mut bad = Vec::new();
    let steps = fx["steps"].as_array().unwrap();
    for (k, s) in steps.iter().enumerate() {
        let now = s["now"].as_f64().unwrap();
        clock.store(now.to_bits(), Ordering::SeqCst);
        flow_nets_tick(&mut core, now, &bars_of(&s["fbars"]), 0, &BTreeMap::new(), |c: &Cells| c.clone());
        let d = flow_delta(&core, sent.as_ref());
        sent = Some(core.clone());
        // 풋프린트 봉(통째로) -- 칸 = [가격, 매수, 매도, …]
        let px: BTreeMap<i64, Vec<f64>> = s["fp_prices"].as_array().unwrap().iter().map(|b| (b[0].as_i64().unwrap(), b[1].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect())).collect();
        let fbars: Vec<Value> = bars_of(&s["fbars"]).into_iter().map(|(t, cells)| {
            let lv: Vec<Value> = cells.iter().zip(&px[&t]).map(|(c, p)| json!([p, c[0], c[1], c[2], c[3], c[4], c[5]])).collect();
            json!({"time": t, "levels": lv})
        }).collect();
        cfg.live.apply(&msg("footprint", json!({"full": true, "bucket": 0.5, "barSeconds": 300, "barsExpected": 288, "ready": true, "retailMaxUsd": 10000.0, "whaleMinUsd": 100000.0, "bars": fbars})));
        let mut core_sit = s["core"].clone();
        core_sit.as_object_mut().unwrap().insert("missing".into(), json!([]));
        core_sit.as_object_mut().unwrap().insert("computed_at".into(), json!(now - 0.5));
        cfg.live.apply(&msg("situation", core_sit));
        cfg.live.apply(&msg("micro", json!({"ts": now as i64, "mid": null, "spread": null, "qi": null, "vol60": 0.0, "liq60": {"long": 0.0, "short": 0.0, "n": 0},
                                             "streams": s["streams"], "flow": serde_json::to_value(&d).unwrap()})));
        {
            let mut g = rest.lock().unwrap();
            g.0 = s["raw"].as_str().unwrap().to_string();
            g.1.clear();
        }
        rtserve::hub::situation_once(&cfg, &mut run);
        let n_day5 = rest.lock().unwrap().1.iter().filter(|t| *t == DAY5_URL).count();
        if n_day5 != s["fetched"].as_bool().unwrap() as usize {
            bad.push(format!("#{k} klines 보냄 {n_day5} / 원본 흉내 {}", s["fetched"]));
        }
        let x = rtserve::py::dumps(&Value::Object(cfg.live.sit.lock().unwrap().x.clone()), (", ", ": "), true, None);
        if x != s["x"].as_str().unwrap() {
            bad.push(format!("#{k} x\n  rust {x}\n  py   {}", s["x"].as_str().unwrap()));
        }
        let g = common::request(addr, "GET", "/api/situation", &[]);
        let (mine, want) = (String::from_utf8_lossy(&g.body).into_owned(), s["body"].as_str().unwrap().to_string());
        if mine != want {
            let i = mine.bytes().zip(want.bytes()).position(|(a, b)| a != b).unwrap_or(mine.len().min(want.len()));
            bad.push(format!("#{k} 본문 글자 {i}: rust …{} / py …{}", &mine[i.saturating_sub(60)..(i + 80).min(mine.len())], &want[i.saturating_sub(60)..(i + 80).min(want.len())]));
        }
        if cfg.live.sit.lock().unwrap().out.as_ref().map(|o| o.1.as_str()) != s["sse"].as_str() {
            bad.push(format!("#{k} SSE 줄 다름"));
        }
    }
    let has = |k: &str| steps.iter().filter(|s| s["has"][k] == json!(true)).count();
    println!("bars·day5: 걸음 {} · move60 {} · oi60 {} · p75 {} · range30 {} · vol_mult {} · VWAP {} · 다름 {}", steps.len(), has("move60"), has("oi60"), has("p75"), has("range30"), has("vm"), has("vwap"), bad.len());
    for b in bad.iter().take(5) {
        println!("  {b}");
    }
    assert!(has("move60") > 0 && has("oi60") > 0 && has("p75") > 0 && has("range30") > 0 && has("vm") > 0 && has("vwap") > 0, "사례가 갈래를 다 덮어야");
    assert!(bad.is_empty());
    let _ = std::fs::remove_dir_all(root);
}
