//! 허브 구독 엔드포인트 대조(2026-11-21). 가짜 허브 = 127.0.0.1 에 띄운 진짜 rtcore 허브(hub::serve)에 rtcore 시험 정답지의 합성 체결을
//! feed::parse → State::apply 로 넣는 것(rtcore/tests/supply1s_parity.rs·footprint_parity.rs 와 같은 입력·같은 묶음 간격). 거래소 접속 없음.
//! 원본 대조 근거는 rtcore 정답지의 `final`(= 파이썬 supply_1s_payload 의 바이낸스 선물 몫 · api_footprint 의 봉·칸) 을 그대로 쓴다:
//! rtserve 가 허브에서 받아 쌓은 값으로 낸 /api/supply-1s · /api/footprint 가 그 행과 같아야 한다. rtserve 는 처음 절반을 스냅숏으로,
//! 나머지를 밀기로 받는다(가운데서 붙는 화면).

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::request;
use rtcore::proto::Event;
use rtcore::state::State;
use serde_json::{json, Value};

fn text_supply(r: &[Value], sym: &str) -> String {
    let (kind, t_ms, tid, p, q, m, w) = (&r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
    if kind.as_i64() == Some(1) {
        return json!({"e": "markPriceUpdate", "E": t_ms, "s": sym}).to_string();
    }
    let x = if p.as_str() == Some("0") { "NA" } else { "MARKET" };
    json!({"e": "trade", "E": w, "T": t_ms, "s": sym, "t": tid, "p": p, "q": q, "X": x, "m": m}).to_string()
}

fn fx(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(format!("{}/../rtcore/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap()
}

/// rtcore 허브 하나(127.0.0.1:0)와 rtserve(같은 프로세스 · 허브 구독) -- (허브 State, rtserve 주소, rtserve Live)
fn rig(sym: &str, rt: &tokio::runtime::Runtime) -> (Arc<State>, std::net::SocketAddr, std::net::SocketAddr, Arc<rtserve::hub::Live>) {
    let st = State::new(&[sym.to_string()], 16, 1 << 16);
    let hub_addr = rt.block_on(async {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let a = l.local_addr().unwrap();
        let cfg = rtcore::hub::HubCfg { path: "/rt".into(), origins: vec![], max_conns: 4, max_subs: 8, ping: Duration::from_secs(60) };
        tokio::spawn(rtcore::hub::serve(l, st.clone(), cfg));
        a
    });
    let live: Arc<Mutex<Option<Arc<rtserve::hub::Live>>>> = Default::default();
    let lv = live.clone();
    let (addr, _root) = common::start_with(&Default::default(), 1_700_000_000_000_000_000, 1.7e9, "eth", &format!("hub_{sym}"), move |c| {
        *lv.lock().unwrap() = Some(c.live.clone());
    });
    let live = live.lock().unwrap().clone().unwrap();
    (st, hub_addr, addr, live)
}

fn connect(rt: &tokio::runtime::Runtime, hub: std::net::SocketAddr, sym: &str, live: &Arc<rtserve::hub::Live>) {
    rt.spawn(rtserve::hub::subscribe(format!("ws://{hub}/rt"), vec![sym.to_string()], live.clone()));
    for _ in 0..3000 { // 30초(다른 빌드가 돌 때 느리다)
        if live.syms.lock().unwrap().contains_key(sym) {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("허브 스냅숏을 못 받았다");
}

fn get_json(addr: std::net::SocketAddr, path: &str) -> Value {
    let g = request(addr, "GET", path, &[]);
    assert_eq!(g.status, 200, "{path}: {}", String::from_utf8_lossy(&g.body));
    serde_json::from_slice(&g.body).unwrap()
}

fn same_num(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same_num(p, q)),
        _ => a.as_f64().map(f64::to_bits) == b.as_f64().map(f64::to_bits) && a.is_i64() == b.is_i64(),
    }
}

#[test]
fn supply_1s_from_hub_matches_python_rows() {
    let fx = fx("supply1s_eth.json");
    let sym = fx["symbol"].as_str().unwrap().to_string();
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let (st, hub, addr, live) = rig(&sym, &rt);
    let rows = fx["rows"].as_array().unwrap();
    let half = rows.len() / 2;
    let mut conn = 0;
    for (i, r) in rows.iter().enumerate() {
        if i == half {
            st.flush();
            connect(&rt, hub, &sym, &live); // 가운데서 붙는다: 앞 절반 = 스냅숏 · 뒤 = 밀기
        }
        let r = r.as_array().unwrap();
        if r[0].as_i64().unwrap() != conn {
            st.reset_orders(&sym);
            conn = r[0].as_i64().unwrap();
        }
        if let Some(m) = rtcore::feed::parse(&text_supply(r, &sym.to_uppercase()), r[7].as_i64().unwrap() * 1000) {
            st.apply(m);
        }
        if (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 60 < 5 || i + 1 == rows.len() {
            st.flush();
        }
    }
    let fin = &fx["final"];
    let now = fin["now"].as_i64().unwrap();
    let mut got = Value::Null;
    for _ in 0..3000 {
        got = get_json(addr, "/api/supply-1s?asset=eth");
        if got["now"] == now && same_num(&got["partial"]["bn"], &fin["partial"]["bn"]) {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(got["now"], fin["now"], "now");
    assert!(same_num(&got["partial"]["bn"], &fin["partial"]["bn"]), "partial.bn {} != {}", got["partial"]["bn"], fin["partial"]["bn"]);
    let (g, w) = (got["seconds"].as_array().unwrap(), fin["seconds"].as_array().unwrap());
    assert_eq!(g.len(), w.len(), "seconds 행 수");
    let mut fields = 0;
    for (a, b) in g.iter().zip(w) {
        assert!(same_num(a, b), "행 {a} != {b}");
        fields += 8;
    }
    // 키 순서 = 원본(바이낸스 체결이 있을 때) · 바이낸스 OI 는 코어 범위 밖이라 빈 목록
    let keys: Vec<&str> = got.as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(keys, ["symbol", "now", "retailMaxUsd", "whaleMinUsd", "oi", "liq", "okx", "okxOi", "okxLiq", "okxNow", "bybitLiq", "okxMeta", "spot", "spotNow", "spotMeta", "partial", "seconds"]);
    assert_eq!(got["oi"], json!([]));
    // 증분: since = 마지막 행 → 새 행 없음 · since = 끝에서 10번째 → 그 뒤 9행
    let last = w.last().unwrap()[0].as_i64().unwrap();
    assert_eq!(get_json(addr, &format!("/api/supply-1s?since={last}"))["seconds"], json!([]));
    let tenth = w[w.len() - 10][0].as_i64().unwrap();
    assert_eq!(get_json(addr, &format!("/api/supply-1s?since={tenth}"))["seconds"].as_array().unwrap().len(), 9);
    // 꺼진 코인 = 404 flow_off (DASHBOARD_ASSETS=eth)
    let g404 = request(addr, "GET", "/api/supply-1s?asset=sol", &[]);
    assert_eq!((g404.status, String::from_utf8_lossy(&g404.body).into_owned()), (404, "404: flow_off".to_string()));
    eprintln!("supply_1s_from_hub: 행 {} · 필드 {fields} · 비트까지 같음", g.len());

    // SSE /api/stream?supply=1: 첫 프레임 = since=0 응답과 같은 본문 · 다음 프레임은 커서 뒤라 seconds 가 비었다
    let frames = sse_frames(addr, "/api/stream?supply=1&asset=eth", 2);
    let (ev, data) = frames[0].split_once('\n').unwrap();
    assert_eq!(ev, "event: supply");
    let first: Value = serde_json::from_str(data.strip_prefix("data: ").unwrap()).unwrap();
    assert_eq!(first["seconds"], got["seconds"]);
    let second: Value = serde_json::from_str(frames[1].split_once('\n').unwrap().1.strip_prefix("data: ").unwrap()).unwrap();
    assert_eq!(second["seconds"], json!([]), "커서가 안 옮겨졌다");
    let r404 = request(addr, "GET", "/api/stream?supply=1&asset=sol", &[]);
    assert_eq!(r404.status, 404);
}

/// 날 HTTP 로 SSE 를 열고 프레임 n 개(빈 줄로 끝나는 덩어리)를 읽는다 -- 청크 머리는 벗긴다
pub fn sse_frames(addr: std::net::SocketAddr, path: &str, n: usize) -> Vec<String> {
    use std::io::{Read, Write};
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    s.write_all(format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n").as_bytes()).unwrap();
    let mut buf = Vec::new();
    let mut chunk = [0u8; 65536];
    loop {
        let k = s.read(&mut chunk).unwrap();
        assert!(k > 0, "SSE 가 끊겼다");
        buf.extend_from_slice(&chunk[..k]);
        let txt = String::from_utf8_lossy(&buf).into_owned();
        let Some(h) = txt.find("\r\n\r\n") else { continue };
        let head = &txt[..h];
        assert!(head.contains("content-type: text/event-stream") && head.contains("cache-control: no-cache") && head.contains("x-accel-buffering: no"), "{head}");
        // 청크 풀기
        let mut body = String::new();
        let mut rest = &txt[h + 4..];
        while let Some(nl) = rest.find("\r\n") {
            let Ok(len) = usize::from_str_radix(&rest[..nl], 16) else { break };
            if rest.len() < nl + 2 + len + 2 {
                break;
            }
            body.push_str(&rest[nl + 2..nl + 2 + len]);
            rest = &rest[nl + 4 + len..];
        }
        let frames: Vec<String> = body.split("\n\n").filter(|x| !x.is_empty()).map(str::to_string).collect();
        if frames.len() >= n && body.ends_with("\n\n") {
            return frames[..n].to_vec();
        }
    }
}

#[test]
fn footprint_from_hub_matches_python_bars() {
    let fx = fx("footprint_eth.json");
    let sym = fx["symbol"].as_str().unwrap().to_string();
    let t0 = fx["t0"].as_i64().unwrap();
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let (st, hub, addr, live) = rig(&sym, &rt);
    let rows = fx["rows"].as_array().unwrap();
    let half = rows.len() / 2;
    let mut conn = 0;
    for (i, r) in rows.iter().enumerate() {
        if i == half {
            st.flush();
            connect(&rt, hub, &sym, &live);
        }
        let r = r.as_array().unwrap();
        if r[0].as_i64().unwrap() != conn {
            st.reset_orders(&sym);
            conn = r[0].as_i64().unwrap();
        }
        let (kind, t_ms, tid, p, q, m, w) = (&r[1], r[2].as_i64().unwrap() + t0, &r[3], &r[4], &r[5], &r[6], r[7].as_i64().unwrap() + t0);
        let txt = if kind.as_i64() == Some(1) {
            json!({"e": "markPriceUpdate", "E": t_ms, "s": sym.to_uppercase()}).to_string()
        } else {
            let x = if p.as_str() == Some("0") { "NA" } else { "MARKET" };
            json!({"e": "trade", "E": w, "T": t_ms, "s": sym.to_uppercase(), "t": tid, "p": p, "q": q, "X": x, "m": m}).to_string()
        };
        if let Some(m) = rtcore::feed::parse(&txt, w * 1000) {
            st.apply(m);
        }
        if (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 60 < 3 {
            st.flush();
        }
    }
    st.flush();
    let Event::Snapshot { footprint, .. } = st.snapshot(&sym, 0).unwrap().ev else { panic!() };
    let snap = footprint.unwrap();
    let fin = &fx["final"];
    let mut got = Value::Null;
    for _ in 0..3000 {
        got = get_json(addr, "/api/footprint?asset=eth&bars=144");
        // 밀기가 다 닿을 때까지(봉마다 칸이 코어 스냅숏과 같아질 때까지) -- 앞 판은 «마지막 봉 칸 수»만 봐서 부하가 있을 때 덜 닿은 채로 비교했다
        let caught = got["bars"].as_array().is_some_and(|b| {
            b.len() == snap.bars.len() && b.iter().zip(&snap.bars).all(|(g, s)| g["time"] == s.time && same_num(&g["levels"], &json!(s.levels)))
        });
        if caught {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(got["bucket"].as_f64(), fin["bucket"].as_f64());
    assert_eq!(got["barSeconds"], fin["barSeconds"]);
    assert_eq!(got["barsExpected"], fin["barsExpected"]);
    assert_eq!((got["full"].clone(), got["ready"].clone()), (json!(true), json!(snap.ready)));
    let (g, w) = (got["bars"].as_array().unwrap(), fin["bars"].as_array().unwrap());
    assert_eq!(g.len(), w.len(), "봉 수");
    let (mut levels, mut dhl) = (0, 0);
    for ((gb, wb), sb) in g.iter().zip(w).zip(&snap.bars) {
        assert_eq!(gb["time"], wb["time"]);
        assert!(same_num(&gb["levels"], &wb["levels"]), "봉 {} 칸", gb["time"]);
        levels += gb["levels"].as_array().unwrap().len();
        // dhl = 코어 판(파이썬과의 관계는 rtcore footprint_parity 가 가지별로 이미 대조) -- rtserve 는 받은 그대로 낸다
        match sb.dhl {
            Some(d) => {
                assert!(same_num(&gb["dhl"], &json!(d)), "봉 {} dhl", gb["time"]);
                dhl += 1;
            }
            None => assert!(gb.get("dhl").is_none()),
        }
        assert_eq!(gb["agg"], json!(false));
    }
    // 키 순서 = 원본 api_footprint
    let keys: Vec<&str> = got.as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(keys, ["symbol", "bucket", "barSeconds", "barsExpected", "ready", "full", "retailMaxUsd", "whaleMinUsd", "venues", "bars"]);
    // 창 12봉(기본) · since 증분(ready 일 때만)
    assert_eq!(get_json(addr, "/api/footprint")["bars"].as_array().unwrap().len(), 12.min(g.len()));
    let tail = g[g.len() - 2]["time"].as_i64().unwrap();
    let inc = get_json(addr, &format!("/api/footprint?bars=144&since={tail}"));
    assert_eq!(inc["full"], json!(!snap.ready));
    if snap.ready {
        assert_eq!(inc["bars"].as_array().unwrap().len(), 2);
    }
    eprintln!("footprint_from_hub: 봉 {} · 칸 {levels} · dhl {dhl} · 비트까지 같음", g.len());
}

#[test]
fn events_sse_first_frame_and_tick() {
    // 허브 없이: 상태 파일 + 가격 한 줄을 넣고 첫 프레임(접속 즉시)과 발행 프레임 모양을 본다
    let mut files = serde_json::Map::new();
    let st = json!({"session": {"name": "아시아"}, "microstructure": {"q": 1.5}, "tail_risk": null, "other": [1, 2]});
    files.insert("data/live/dashboard_state.json".into(), Value::String(b64e(st.to_string().as_bytes())));
    let live: Arc<Mutex<Option<Arc<rtserve::hub::Live>>>> = Default::default();
    let lv = live.clone();
    let (addr, _root) = common::start_with(&files, 1_700_000_000_000_000_000, 1_795_000_123.25, "eth,sol", "events", move |c| {
        *lv.lock().unwrap() = Some(c.live.clone());
    });
    let live = live.lock().unwrap().clone().unwrap();
    live.prices.lock().unwrap().insert("ETHUSDT".into(), (3001.5, 1));
    let frames = sse_frames(addr, "/api/events", 2);
    let first: Value = serde_json::from_str(frames[0].strip_prefix("data: ").unwrap()).unwrap();
    assert_eq!(first.as_object().unwrap().keys().collect::<Vec<_>>(), ["state", "tickers", "assets"]);
    assert_eq!(first["assets"], json!(["eth", "sol"]));
    let tick: Value = serde_json::from_str(frames[1].strip_prefix("data: ").unwrap()).unwrap();
    assert_eq!(tick.as_object().unwrap().keys().collect::<Vec<_>>(), ["state", "tickers", "assets", "acct_seq"]);
    // 상태가 바뀐 첫 주기(또는 접속 전에 이미 실렸으면 null) · 실리면 세 블록만
    if !tick["state"].is_null() {
        assert_eq!(tick["state"], json!({"state": {"session": {"name": "아시아"}, "microstructure": {"q": 1.5}, "tail_risk": null}}));
    } else {
        assert_eq!(first["state"], json!({"state": {"session": {"name": "아시아"}, "microstructure": {"q": 1.5}, "tail_risk": null}}));
    }
    assert_eq!(tick["tickers"], json!({"eth": {"symbol": "ETHUSDT", "price": 3001.5, "ts": "2026-11-18T11:08:43.250000+00:00"}}));
    // 비ASCII 는 날 글자(ensure_ascii=False)
    assert!(frames.iter().any(|f| f.contains("아시아")));
}

fn b64e(b: &[u8]) -> String {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for c in b.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for k in 0..4 {
            out.push(if k <= c.len() { T[(n >> (18 - 6 * k) & 63) as usize] as char } else { '=' });
        }
    }
    out
}


/// 2026-11-24: 틱 봉(/api/tick-bars)과 풋프린트 1분봉(/api/footprint?tf=1m)을 허브에서 받아 낸다. 가운데서 붙는다(앞 절반 = 스냅숏 · 뒤 = 밀기).
/// 1분봉 = 정답지 final_1m(server.py footprint_1m 옮겨 적은 것)과 비트까지 · 틱 봉 = 코어 스냅숏(rtcore tests/tick_bars.rs 가 계산을 따로 본다)의 반올림.
#[test]
fn tick_bars_and_footprint_1m_from_hub() {
    let fx = fx("footprint_eth.json");
    let sym = fx["symbol"].as_str().unwrap().to_string();
    let t0 = fx["t0"].as_i64().unwrap();
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let (st, hub, addr, live) = rig(&sym, &rt);
    let rows = fx["rows"].as_array().unwrap();
    let mut conn = 0;
    for (i, r) in rows.iter().enumerate() {
        if i == rows.len() / 2 {
            st.flush();
            connect(&rt, hub, &sym, &live);
        }
        let r = r.as_array().unwrap();
        if r[0].as_i64().unwrap() != conn {
            st.reset_orders(&sym);
            conn = r[0].as_i64().unwrap();
        }
        let (kind, t_ms, tid, p, q, m, w) = (&r[1], r[2].as_i64().unwrap() + t0, &r[3], &r[4], &r[5], &r[6], r[7].as_i64().unwrap() + t0);
        if kind.as_i64() == Some(1) {
            continue;
        }
        let x = if p.as_str() == Some("0") { "NA" } else { "MARKET" };
        let txt = json!({"e": "trade", "E": w, "T": t_ms, "s": sym.to_uppercase(), "t": tid, "p": p, "q": q, "X": x, "m": m}).to_string();
        if let Some(m) = rtcore::feed::parse(&txt, w * 1000) {
            st.apply(m);
        }
        if (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 60 < 3 {
            st.flush();
        }
    }
    st.flush();
    let Event::Snapshot { tick, .. } = st.snapshot(&sym, 0).unwrap().ev else { panic!() };
    let tick = tick.unwrap();
    let r2 = |x: f64| json!((x * 100.0).round_ties_even() / 100.0); // ETH 가격 2자리(값이 이미 0.01 격자라 같은 수)
    let r3 = |x: f64| json!(format!("{x:.3}").parse::<f64>().unwrap());
    let row = |b: &rtcore::proto::TickRow| {
        let cells: Vec<Value> = b.10.iter().map(|c| json!([r2(c.0), r3(c.1)])).collect();
        json!([b.0, b.1, b.2, r2(b.3), r2(b.4), r2(b.5), r2(b.6), r3(b.7), r3(b.8), b.9, cells])
    };
    let want = |since: &[i64]| -> Value {
        let mut m = serde_json::Map::new();
        for (i, l) in tick.lanes.iter().enumerate() {
            let lo = since.get(i).copied().unwrap_or(i64::MIN);
            m.insert(l.n.to_string(), json!({"k": l.k, "bars": l.bars.iter().filter(|b| b.0 >= lo).map(row).collect::<Vec<_>>(), "cur": l.cur.as_ref().map_or(Value::Null, row)}));
        }
        json!({"symbol": "ETHUSDT", "sizes": [100, 500, 1000, 2000, 5000], "ticks": m})
    };
    let mut got = Value::Null;
    for _ in 0..3000 {
        got = get_json(addr, "/api/tick-bars?asset=eth");
        if same_num(&got["ticks"]["100"]["cur"], &want(&[])["ticks"]["100"]["cur"]) && got["ticks"]["100"]["k"] == want(&[])["ticks"]["100"]["k"] {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let all = want(&[]);
    let keys: Vec<&str> = got.as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(keys, ["symbol", "sizes", "ticks"]);
    for n in ["100", "500", "1000", "2000", "5000"] {
        let (g, w) = (&got["ticks"][n], &all["ticks"][n]);
        assert_eq!(g["k"], w["k"], "{n} k");
        assert!(same_num(&g["bars"], &w["bars"]), "{n} bars");
        assert!(same_num(&g["cur"], &w["cur"]) || (g["cur"].is_null() && w["cur"].is_null()), "{n} cur");
    }
    let k100 = tick.lanes[0].k;
    assert!(k100 > 40 && tick.lanes[0].bars.len() == k100 as usize, "100틱 봉 {k100}");
    // since: 100틱 = 끝 3봉 · 500틱 = k(닫힌 봉 없음) · 못 읽는 칸·빠진 칸 = 보관분 전부
    let k = |i: usize| tick.lanes[i].k;
    let q = format!("/api/tick-bars?asset=eth&since={},{},x,{}", k(0) - 3, k(1), k(3) - 1);
    let inc = get_json(addr, &q);
    let w = want(&[k(0) - 3, k(1), i64::MIN, k(3) - 1]);
    for n in ["100", "500", "1000", "2000", "5000"] {
        assert!(same_num(&inc["ticks"][n]["bars"], &w["ticks"][n]["bars"]), "since {n}");
        assert!(same_num(&inc["ticks"][n]["cur"], &w["ticks"][n]["cur"]) || inc["ticks"][n]["cur"].is_null(), "since {n} cur 는 늘");
    }
    assert_eq!(inc["ticks"]["100"]["bars"].as_array().unwrap().len(), 3);
    assert_eq!(inc["ticks"]["500"]["bars"], json!([]));
    // 정수 칸은 정수로
    let b0 = &got["ticks"]["100"]["bars"][0];
    assert!(b0[0].is_i64() && b0[1].is_i64() && b0[2].is_i64() && b0[9].is_i64() && b0[9] == 100, "{b0}");
    // 11번째 칸(2026-11-25): [[가격, 수량], ..] · 가격 = ETH 칸 폭 0.5 격자 · 오름차순 · 고저 안 · 수량 3자리 · 칸 합 ≈ 봉 수량
    for n in ["100", "500", "1000", "2000", "5000"] {
        for b in got["ticks"][n]["bars"].as_array().unwrap().iter().chain(got["ticks"][n]["cur"].as_array().map(|_| &got["ticks"][n]["cur"])) {
            let b = b.as_array().unwrap();
            assert_eq!(b.len(), 11);
            let cells = b[10].as_array().unwrap();
            assert!(!cells.is_empty());
            let (hi, lo, vol) = (b[4].as_f64().unwrap(), b[5].as_f64().unwrap(), b[7].as_f64().unwrap());
            let mut prev = f64::MIN;
            let mut sum = 0.0;
            for c in cells {
                let (px, qty) = (c[0].as_f64().unwrap(), c[1].as_f64().unwrap());
                assert!(c.as_array().unwrap().len() == 2 && px > prev && (px * 2.0).fract() == 0.0 && px >= lo - 0.25 && px <= hi + 0.25, "{c}");
                assert_eq!(qty, format!("{qty:.3}").parse::<f64>().unwrap());
                (prev, sum) = (px, sum + qty);
            }
            assert!((sum - vol).abs() <= 0.0005 * (cells.len() + 1) as f64, "{n} 칸 합 {sum} vs {vol}");
        }
    }
    // 모르는 코인 = 풋프린트와 같은 404
    let (a, b) = (request(addr, "GET", "/api/tick-bars?asset=sol", &[]), request(addr, "GET", "/api/footprint?asset=sol", &[]));
    assert_eq!((a.status, &a.body, &a.headers), (b.status, &b.body, &b.headers));
    assert_eq!(a.status, 404);
    // 응답 머리 = 풋프린트와 같다(본문에 따라 달라지는 칸만 빼고)
    let hd = |g: &common::Got| g.headers.iter().filter(|(k, _)| !["date", "content-length", "etag"].contains(&k.to_ascii_lowercase().as_str())).cloned().collect::<Vec<_>>();
    assert_eq!(hd(&request(addr, "GET", "/api/tick-bars?asset=eth", &[])), hd(&request(addr, "GET", "/api/footprint?asset=eth", &[])));

    // 1분봉 응답 = 원본 footprint_1m 의 봉(비트까지)
    let f1 = &fx["final_1m"];
    let mut g1 = Value::Null;
    for _ in 0..3000 {
        g1 = get_json(addr, "/api/footprint?asset=eth&tf=1m");
        if same_num(&g1["bars"].as_array().map_or(json!([]), |b| json!(b.iter().map(|x| x["ohlc"].clone()).collect::<Vec<_>>())),
                    &json!(f1["bars"].as_array().unwrap().iter().map(|x| x["ohlc"].clone()).collect::<Vec<_>>())) {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let keys: Vec<&str> = g1.as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(keys, ["symbol", "bucket", "barSeconds", "barsExpected", "ready", "full", "retailMaxUsd", "whaleMinUsd", "venues", "bars"]);
    assert_eq!((g1["barSeconds"].clone(), g1["barsExpected"].clone(), g1["bucket"].as_f64()), (json!(60), json!(120), f1["bucket"].as_f64()));
    let (gb, wb) = (g1["bars"].as_array().unwrap(), f1["bars"].as_array().unwrap());
    assert_eq!(gb.len(), wb.len(), "1분봉 수");
    for (g, w) in gb.iter().zip(wb) {
        assert_eq!(g.as_object().unwrap().keys().collect::<Vec<_>>(), ["time", "ohlc", "levels"]);
        assert_eq!(g["time"], w["time"]);
        assert!(same_num(&g["ohlc"], &w["ohlc"]), "1분봉 {} ohlc", g["time"]);
        assert!(same_num(&g["levels"], &w["levels"]), "1분봉 {} 칸", g["time"]);
    }
    // since 증분: ready 일 때만(아니면 전량 -- 원본 규약)
    let last = wb.last().unwrap()["time"].as_i64().unwrap();
    let inc = get_json(addr, &format!("/api/footprint?asset=eth&tf=1m&since={last}"));
    if inc["ready"] == json!(true) {
        assert_eq!((inc["full"].clone(), inc["bars"].as_array().unwrap().len()), (json!(false), 1));
    } else {
        assert_eq!((inc["full"].clone(), inc["bars"].as_array().unwrap().len()), (json!(true), wb.len()));
    }
    eprintln!("tick_bars·1m: 100틱 k {k100} · 1분봉 {} 봉 · 비트까지 같음", gb.len());
}
