// 넷째 조각 정답지 대조(tools/gen_desk_fixture.py -- 원본 파이썬을 가짜 의존성으로 돌린 것):
// 계좌 스냅숏·나간 요청 열 · 사용자 스트림 사건 열·상태·로그 · 계획 dict 전체·나간 조회 열 · 접수 문 상태코드·본문·시작된 작업·상태 칸·무장 파일·로그.
// + «받아 둔 값»: 같은 값을 (나) Cached 에 넣으면 계획이 (가)와 완전히 같고 시세 조회가 사라진다 · 낡은 값은 (가)로 떨어진다.
#[path = "common/desk_io.rs"]
mod desk_io;

use desk_io::*;
use rtorder::account::{self, StreamAccount, UserStream};
use rtorder::assemble::{Cached, EntryArgs};
use rtorder::plan::Filters;
use serde_json::{json, Value};

/// M10(비상 스탑) 제거로 원본과 일부러 달라지는 접수 사례(손절 해제의 스탑 취소 · 손절 다시 켜기의 스탑 걸기).

fn fixture() -> Value {
    serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/desk_fixture.json")).unwrap()).unwrap()
}

#[test]
fn account_snapshot_and_request_order_match_python() {
    let fx = fixture();
    let mut n = 0;
    for c in fx["account"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let mut rest = FakeRest::new(&c["in"]);
        let syms: Vec<String> = c["in"]["symbols"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().into()).collect();
        let r = block_on(account::fetch_account(&mut rest, &syms, c["in"].get("quote").and_then(Value::as_str).unwrap_or("USDC"), account::DEFAULT_TRADE_LIMIT));
        match (&r, c["raised"].as_str()) {
            (Err(e), Some(py)) => assert_eq!(e, py, "{name}: 예외"),
            (Ok(v), None) => assert_same(v, &c["out"], &format!("{name}: 스냅숏")),
            _ => panic!("{name}: rust {r:?} python raised {:?}", c["raised"]),
        }
        assert_same(&Value::Array(rest.calls.clone()), &c["calls"], &format!("{name}: 요청 열"));
        n += 1;
    }
    assert_eq!(n, 19);
}

#[test]
fn user_stream_and_refresher_match_python() {
    let fx = fixture();
    for c in fx["stream"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let mut io = FakeStream::new(&c["in"]);
        let mut us = UserStream::default();
        if c["in"]["op"] == "stream" {
            block_on(account::user_stream_run(&mut io, &mut us));
        } else {
            block_on(account::account_refresher(&mut io, &mut us));
        }
        assert_same(&Value::Array(io.ev.clone()), &c["events"], &format!("{name}: 사건 열"));
        let py = &c["user_stream"];
        assert_eq!(us.connected, py["connected"].as_bool().unwrap(), "{name}");
        assert_eq!(us.seq, py["seq"].as_i64().unwrap(), "{name}");
        assert_eq!(us.last_event_at, py["last_event_at"].as_f64(), "{name}");
        assert_eq!(us.error.as_deref(), py["error"].as_str(), "{name}");
        assert_same(&json!(io.logs), &c["logs"], &format!("{name}: 로그"));
    }
}

fn run_case(c: &Value, cached: bool) -> (Value, Vec<Value>) {
    let inp = &c["in"];
    let (mut desk, _dir) = desk_from(inp);
    let mut io = FakeDeskIo::new(inp, desk.w.dir.clone());
    let mut fq = FakeQuotes::new(inp);
    let out = if cached {
        let mut q = Cached::new(fq);
        q.now = io.t;
        fill_cache(&mut q, inp);
        let o = dispatch(&mut desk, &mut q, &mut io, inp);
        fq = q.inner;
        o
    } else {
        dispatch(&mut desk, &mut fq, &mut io, inp)
    };
    let mut calls = io.calls.clone();
    calls.extend(fq.calls.clone());
    let mut o = out;
    o["slots"] = json!({"entry": desk.w.entry, "side": desk.w.side_exits, "waits": desk.w.waits});
    o["file"] = read_file(&desk.w.dir);
    o["limits"] = read_limits(&desk.w.dir);
    o["logs"] = json!(desk.w.logs);
    o["task_keys"] = json!(desk.running.iter().filter(|k| inp.get("tasks").and_then(|t| t.get(k.as_str())).is_none()).cloned().collect::<Vec<_>>());
    (o, calls)
}

fn fill_cache(q: &mut Cached<FakeQuotes>, inp: &Value) {
    let books = inp.get("books").cloned().unwrap_or_else(|| q.inner.books.clone());
    for (s, b) in books.as_object().unwrap() {
        q.book.insert(s.clone(), (b.clone(), q.now - 0.5));
    }
    let fil = q.inner.filters.clone();
    for (s, f) in fil.as_object().unwrap() {
        q.filters.insert(s.clone(), (serde_json::from_value::<Filters>(f.clone()).unwrap(), q.now - 3600.0));
    }
    for s in ["ETHUSDC", "ETHUSDT", "SOLUSDC", "SOLUSDT", "XRPUSDC", "SOXLUSDT", "BTCUSDC"] {
        let v = inp.get("vol").and_then(|m| m.get(s)).and_then(Value::as_f64);
        q.vol.insert(s.into(), (v, q.now - 10.0));
        let f = inp.get("funding").and_then(|m| m.get(s)).and_then(Value::as_f64);
        q.funding.insert(s.into(), (f, q.now - 100.0));
        let fee = inp.get("fees").and_then(|m| m.get(s)).cloned().unwrap_or(json!({"symbol": s, "maker_bp": 2.0, "taker_bp": 5.0, "source": "fallback"}));
        q.fees.insert(s.into(), (fee, q.now - 1000.0));
    }
}

#[test]
fn plans_and_intake_match_python() {
    let fx = fixture();
    let (mut n_plan, mut n_api) = (0, 0);
    for c in fx["desk"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let (o, calls) = run_case(c, false);
        for k in ["plan", "cap", "err", "status", "body", "started"] {
            if c.get(k).is_some() {
                assert_same(&o[k], &c[k], &format!("{name}: {k}"));
            }
        }
        for k in ["slots", "file", "limits"] {
            assert_same(&o[k], &c[k], &format!("{name}: {k}"));
        }
        if c.get("status").is_some() {
            assert_same(&o["logs"], &c["logs"], &format!("{name}: 로그"));
            assert_same(&o["task_keys"], &c["task_keys"], &format!("{name}: 작업 이름"));
            n_api += 1;
        } else {
            n_plan += 1;
        }
        assert_same(&sorted(calls), &sorted(c["calls"].as_array().unwrap().clone()), &format!("{name}: 나간 조회(순서 무시 -- 원본은 gather)"));
    }
    assert_eq!((n_plan, n_api), (57, 139), "사례 수(2026-11-21 PR 61 판 -- 지운 기능 사례는 빼고 새 거절 3종 · dbsl 정리 경로 포함 · 의도한 차이 0)");
}

#[test]
fn cached_quotes_give_identical_plans_without_quote_round_trips() {
    // (가)=(나): 같은 값을 받아 둔 원천에 넣으면 계획·본문이 완전히 같다 · 시세 조회(호가·필터·변동성·펀딩·수수료)는 0 · 마진 방식은 늘 (가)
    let fx = fixture();
    let (mut n, mut saved_py, mut saved_rs) = (0, 0usize, 0usize);
    for c in fx["desk"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let (a, ca) = run_case(c, false);
        let (b, cb) = run_case(c, true);
        for k in ["plan", "cap", "err", "status", "body", "started", "slots", "file"] {
            assert_same(&b[k], &a[k], &format!("{name}: (나) {k}"));
        }
        let quote = |v: &Vec<Value>| v.iter().filter(|x| matches!(x[0].as_str(), Some("book" | "filters" | "vol" | "funding" | "fees"))).count();
        saved_py += quote(&ca);
        saved_rs += quote(&cb);
        n += 1;
    }
    eprintln!("(가)=(나) {n}사례 · 시세 조회 파이썬 {saved_py}회 → Rust (나) {saved_rs}회(받아 두지 않은 심볼만)");
    assert!(saved_rs * 10 < saved_py, "받아 둔 값이 조회를 거의 다 없애야 한다");
}

#[test]
fn stale_or_missing_cached_values_fall_back_never_silently() {
    let fx = fixture();
    let c = fx["desk"].as_array().unwrap().iter().find(|c| c["name"] == "lines_long").unwrap();
    let inp = &c["in"];
    for (what, age, expect_miss) in [("book", account_age::BOOK_OK, false), ("book", account_age::BOOK_STALE, true), ("vol", 59.0, false), ("vol", 61.0, true),
                                     ("funding", 301.0, true), ("fees", 3601.0, true), ("filters", 86_401.0, true)] {
        let (mut desk, _d) = desk_from(inp);
        let mut io = FakeDeskIo::new(inp, desk.w.dir.clone());
        let mut q = Cached::new(FakeQuotes::new(inp));
        q.now = io.t;
        fill_cache(&mut q, inp);
        // 한 값만 낡게(그 값의 받은 시각을 now − age) + 낡은 쪽에는 일부러 틀린 값을 넣는다 -- 쓰였다면 계획이 달라진다
        match what {
            "book" => { q.book.insert("ETHUSDC".into(), (json!({"bidPrice": "2490.01", "askPrice": "2490.05"}), q.now - age)); }
            "vol" => { q.vol.insert("ETHUSDC".into(), (Some(99.0), q.now - age)); }
            "funding" => { q.funding.insert("ETHUSDC".into(), (Some(-77.0), q.now - age)); }
            "fees" => { q.fees.insert("ETHUSDC".into(), (json!({"symbol": "ETHUSDC", "maker_bp": 9.0, "taker_bp": 9.0, "source": "x"}), q.now - age)); }
            _ => { q.filters.insert("ETHUSDC".into(), (Filters { step: 1.0, tick: 1.0, min_qty: 1.0, min_notional: 0.0 }, q.now - age)); }
        }
        let a = EntryArgs { want_lev: Some(10), lines: Some((2600.0, 2400.0)), ..EntryArgs::new("LONG", "eth") };
        let r = block_on(desk.assemble_entry_plan(&mut q, &mut io, &a));
        let missed = q.misses.iter().any(|m| m.0 == what && m.1 == "ETHUSDC");
        assert_eq!(missed, expect_miss, "{what} 나이 {age}");
        if expect_miss {
            // 낡은 값은 버리고 (가)로 물어 파이썬과 같은 계획
            assert!(q.inner.calls.iter().any(|x| x[0] == what && x[1] == "ETHUSDC"), "{what}: (가)로 안 떨어졌다");
            assert_same(&r.unwrap().0, &c["plan"], &format!("{what} 낡음 → (가)"));
        } else if what == "book" {
            assert_eq!(r.unwrap().0["price"], json!(2490.01), "받아 둔 값(2초 안)을 써야 한다");
        }
    }
}

#[test]
fn stream_account_rules() {
    // 스트림 계좌는 «살아 있고 · 같은 연결 · 60초 안 대조 · 어긋남 없음» 일 때만. 아니면 REST(이유를 돌려준다).
    let base = json!({"ok": true, "positions": [{"symbol": "ETHUSDC", "side": "LONG", "qty": 0.4, "entry_price": 2400.0, "mark_price": 2450.0, "notional": 980.0}],
                      "open_orders": [{"symbol": "ETHUSDC", "side": "SELL", "price": 2600.0, "qty": 0.4, "reduce_only": false, "cid": "dbtpL1"}]});
    let mut us = UserStream { connected: true, epoch: 1, ..Default::default() };
    let mut sa = StreamAccount::default();
    assert_eq!(sa.usable(&us, 100.0).unwrap_err(), "no_base");
    assert!(sa.reconcile(&base, 100.0, 1));
    assert_eq!(sa.usable(&us, 100.0).unwrap()["positions"][0]["qty"], json!(0.4));
    // 체결 사건: 포지션 0.4 → 0.1 (부분 청산) · 익절 지정가 수량도 줄어듦
    sa.apply(&json!({"e": "ACCOUNT_UPDATE", "a": {"P": [{"s": "ETHUSDC", "ps": "LONG", "pa": "0.1", "ep": "2400.0", "up": "5.0"}]}}));
    sa.apply(&json!({"e": "ORDER_TRADE_UPDATE", "o": {"s": "ETHUSDC", "S": "SELL", "o": "LIMIT", "X": "PARTIALLY_FILLED", "c": "dbtpL1", "q": "0.4", "z": "0.3", "p": "2600"}}));
    let v = sa.usable(&us, 110.0).unwrap();
    assert_eq!(v["positions"][0]["qty"], json!(0.1));
    assert!((v["open_orders"][0]["qty"].as_f64().unwrap() - 0.1).abs() < 1e-12);
    // 60초 넘게 대조가 없으면 REST
    assert_eq!(sa.usable(&us, 161.0).unwrap_err(), "stale");
    // REST 가 같은 값을 말하면 깨끗한 대조
    let mut rest = base.clone();
    rest["positions"][0]["qty"] = json!(0.1);
    rest["open_orders"][0]["qty"] = json!(0.1);
    assert!(sa.reconcile(&rest, 120.0, 1));
    assert!(sa.usable(&us, 121.0).is_ok());
    // 알림 누락: 스트림은 0.1 이라 믿는데 REST 는 이미 0 (포지션 사라짐) → 어긋남 → 다음 깨끗한 대조까지 REST
    let mut gone = rest.clone();
    gone["positions"] = json!([]);
    assert!(!sa.reconcile(&gone, 130.0, 1));
    assert_eq!(sa.usable(&us, 131.0).unwrap_err(), "mismatch");
    assert_eq!(sa.mismatches, 1);
    assert!(sa.reconcile(&gone, 140.0, 1));
    assert!(sa.usable(&us, 141.0).is_ok());
    // 재연결 사이(끊긴 동안의 사건은 다시 안 온다) → 새 연결에서 대조 전까지 REST
    us.epoch = 2;
    assert_eq!(sa.usable(&us, 142.0).unwrap_err(), "reconnected");
    assert!(sa.reconcile(&gone, 143.0, 2)); // 다른 연결의 기준과는 비교하지 않는다
    assert!(sa.usable(&us, 144.0).is_ok());
    us.connected = false;
    assert_eq!(sa.usable(&us, 145.0).unwrap_err(), "stream_down");
    // 실패한 REST(ok=false)는 기준을 바꾸지 않는다(«못 읽음 = 모름»)
    us.connected = true;
    let before = sa.base.clone();
    sa.reconcile(&json!({"ok": false, "error": "418"}), 146.0, 2);
    assert_eq!(sa.base, before);
}

#[test]
fn stream_account_matches_python_rest_after_same_events() {
    // 원본 fetch_account(사건 전 계좌) 위에 같은 알림 열을 반영한 스트림 계좌 == 원본 fetch_account(사건 뒤 계좌)
    let fx = fixture();
    let cases = fx["stream_rest"].as_array().unwrap();
    assert_eq!(cases.len(), 6);
    let pos = |v: &Value| {
        let mut x: Vec<String> = v["positions"].as_array().unwrap().iter()
            .map(|p| format!("{}|{}|{:.8}|{:.8}|{:.8}", p["symbol"], p["side"], p["qty"].as_f64().unwrap(), p["entry_price"].as_f64().unwrap(), p["unrealized_pnl"].as_f64().unwrap())).collect();
        x.sort();
        x
    };
    let ords = |v: &Value| {
        let mut x: Vec<String> = v["open_orders"].as_array().unwrap().iter()
            .map(|o| format!("{}|{}|{:.8}|{:.8}|{}|{}", o["symbol"], o["side"], o["price"].as_f64().unwrap(), o["qty"].as_f64().unwrap(), o["reduce_only"], o["cid"])).collect();
        x.sort();
        x
    };
    for c in cases {
        let name = c["name"].as_str().unwrap();
        let us = UserStream { connected: true, epoch: 1, ..Default::default() };
        let mut sa = StreamAccount::default();
        assert!(sa.reconcile(&c["acct0"], 100.0, 1));
        for e in c["events"].as_array().unwrap() {
            sa.apply(e);
        }
        let v = sa.usable(&us, 101.0).unwrap();
        assert_eq!(pos(&v), pos(&c["acct1"]), "{name} 포지션");
        assert_eq!(ords(&v), ords(&c["acct1"]), "{name} 미체결");
        assert!(sa.reconcile(&c["acct1"], 102.0, 1), "{name} 대조는 깨끗해야 한다");
        assert_eq!(sa.mismatches, 0, "{name}");
    }
}

#[test]
fn stream_account_exit_plan_equals_rest_exit_plan_and_skips_account_read() {
    // 청산 조립이 스트림 계좌를 쓰면 REST 계좌와 같은 값일 때 같은 계획 -- 계좌 REST 왕복(심볼 13개 · 무게 ~50)이 사라진다
    let fx = fixture();
    let c = fx["desk"].as_array().unwrap().iter().find(|c| c["name"] == "exit_long_full").unwrap();
    let inp = &c["in"];
    let (mut desk, _d) = desk_from(inp);
    let mut io = FakeDeskIo::new(inp, desk.w.dir.clone());
    let mut sa = StreamAccount::default();
    let us = UserStream { connected: true, epoch: 7, ..Default::default() };
    sa.reconcile(&inp["account"], io.t - 5.0, 7);
    io.stream = Some(sa.usable(&us, io.t).unwrap());
    let mut q = FakeQuotes::new(inp);
    let plan = block_on(desk.assemble_exit_plan(&mut q, &mut io, "LONG", 1.0, true, "eth", false, None)).unwrap();
    assert_same(&plan, &c["plan"], "스트림 계좌 청산 계획");
    assert!(io.calls.iter().all(|x| x[0] != "account"), "REST 계좌를 안 읽어야 한다");
}

mod account_age {
    pub const BOOK_OK: f64 = 1.5;
    pub const BOOK_STALE: f64 = 2.5;
}

#[test]
fn account_cache_rules_and_last_good() {
    use rtorder::account::{fresh_needs_fetch, swr_decide, AccountBook, Swr};
    // swr_cached(server.py): ttl 안 = 그대로 · ttl+max_stale 안 = 묵은 값 + 뒤에서 갱신 · 그 밖·빈 캐시 = 기다려 새로
    assert_eq!(swr_decide(109.9, 100.0, true, 10.0, 60.0), Swr::Serve);
    assert_eq!(swr_decide(110.0, 100.0, true, 10.0, 60.0), Swr::ServeAndRefresh);
    assert_eq!(swr_decide(170.0, 100.0, true, 10.0, 60.0), Swr::Fetch);
    assert_eq!(swr_decide(100.0, 100.0, false, 10.0, 60.0), Swr::Fetch);
    // ?fresh=1: 1초 안에 받은 값이면 다시 안 묻는다(A3)
    assert!(!fresh_needs_fetch(100.9, 100.0) && fresh_needs_fetch(101.1, 100.0));
    // last_good(A2): 실패 응답은 ok=false 그대로 + 화면용 마지막 정상값(30분까지)만 붙는다
    let fx = fixture();
    let ok_in = &fx["account"].as_array().unwrap().iter().find(|c| c["name"] == "normal_usdc").unwrap()["in"];
    let bad_in = &fx["account"].as_array().unwrap().iter().find(|c| c["name"] == "risk_ban_418").unwrap()["in"];
    let mut book = AccountBook { eth_symbol: "ETHUSDC".into(), last_good: None };
    let mut r1 = FakeRest::new(ok_in);
    let good = block_on(book.produce(&mut r1)).unwrap();
    assert_eq!(good["ok"], json!(true));
    assert_eq!(good["exec_symbol"], json!("ETHUSDC"));
    let mut r2 = FakeRest::new(bad_in);
    r2.t = r1.t + 600.0;
    let bad = block_on(book.produce(&mut r2)).unwrap();
    assert_eq!(bad["ok"], json!(false), "실패는 실패 그대로(«못 읽음 = 모름»)");
    assert_eq!(bad["last_good"]["age_s"], json!(600));
    r2.t = r1.t + 1800.0;
    let bad2 = block_on(book.produce(&mut r2)).unwrap();
    assert!(bad2.get("last_good").is_none(), "30분 지나면 안 붙는다");
    assert_eq!(book.symbols().len(), 11, "시세 심볼(시세만 코인 뺌) + 주문 심볼, 중복 없이");
}
