//! 1초 수급 파이썬 일치 시험. 정답지 = rtcore/tools/gen_supply1s_fixture.py 가 dashboard/coin_flow.py 원본으로 만든 것.
//! 같은 합성 체결 열을 feed::parse → State::apply(실제 수신 경로)에 넣고 ① 중간·끝 1초 칸(반올림 전) ② 스냅숏 행
//! ③ 밀기만 받아 다시 지은 화면 쪽 칸 을 파이썬과 대조한다. 실수는 상대 오차 1e-9.
use rtcore::proto::{Event, SecRow};
use rtcore::state::State;
use serde_json::{json, Value};
use std::collections::BTreeMap;

const FX: &str = include_str!("fixtures/supply1s_eth.json");

fn close(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= 1e-9 * a.abs().max(b.abs())
}

fn row_vals(r: &SecRow) -> [f64; 7] {
    [r.1, r.2, r.3, r.4, r.5, r.6, r.7]
}

#[derive(Default)]
struct Tally {
    cells: usize,
    fields: usize,
    bit_equal: usize,
}

impl Tally {
    fn cmp(&mut self, what: &str, s: i64, got: &[f64], want: &[Value]) {
        assert_eq!(got.len(), want.len(), "{what} 초 {s} 칸 수");
        self.cells += 1;
        for (j, (g, w)) in got.iter().zip(want).enumerate() {
            let w = w.as_f64().unwrap();
            assert!(close(*g, w), "{what} 초 {s} 칸[{j}]: rust {g} != python {w}");
            self.fields += 1;
            self.bit_equal += (g.to_bits() == w.to_bits()) as usize;
        }
    }

    fn cells_vs(&mut self, what: &str, got: &[(i64, [f64; 7])], want: &Value) {
        let want = want.as_object().unwrap();
        let gk: Vec<i64> = got.iter().map(|c| c.0).collect();
        let wk: Vec<i64> = want.keys().map(|k| k.parse().unwrap()).collect();
        assert_eq!(gk, wk, "{what}: 초 목록이 다르다");
        for (s, v) in got {
            self.cmp(what, *s, v, want[&s.to_string()].as_array().unwrap());
        }
    }

    fn rows_vs(&mut self, what: &str, got: &[SecRow], want: &Value) {
        let want = want.as_array().unwrap();
        assert_eq!(got.len(), want.len(), "{what}: 행 수");
        for (g, w) in got.iter().zip(want) {
            let w = w.as_array().unwrap();
            assert_eq!(g.0, w[0].as_i64().unwrap(), "{what}: 초");
            self.cmp(what, g.0, &row_vals(g), &w[1..]);
        }
    }
}

fn text(r: &[Value], sym: &str) -> String {
    let (kind, t_ms, tid, p, q, m, w) = (&r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
    if kind.as_i64() == Some(1) {
        return json!({"e": "markPriceUpdate", "E": t_ms, "s": sym}).to_string(); // 파이썬도 «trade 아님»으로 건너뛴다
    }
    let x = if p.as_str() == Some("0") { "NA" } else { "MARKET" };
    json!({"e": "trade", "E": w, "T": t_ms, "s": sym, "t": tid, "p": p, "q": q, "X": x, "m": m}).to_string()
}

#[test]
fn supply1s_matches_python() {
    check(FX);
}

// 칸 폭이 작은 종목(0.0003 · 가격 4자리)과 가격이 큰 종목(칸 15 · 가격 0자리) -- ETH 칸 폭에선 값이 안 바뀌는 변형(응답 가격 반올림 등)을 잡는다.
#[test]
fn supply1s_matches_python_xrp() {
    check(&gen_fixture("supply1s_xrp.json"));
}

#[test]
fn supply1s_matches_python_btc() {
    check(&gen_fixture("supply1s_btc.json"));
}

/// 커밋 안 한 정답지(XRP·BTC -- 합쳐 약 1.9MB). 없으면 저장소 루트에서 `python3 rtcore/tools/gen_supply1s_fixture.py xrp`(btc 도)로 먼저 만든다.
fn gen_fixture(name: &str) -> String {
    let p = format!("{}/tests/fixtures/gen/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p}: {e} -- 생성기로 먼저 만든다(이 함수 주석)"))
}

fn check(fx: &str) {
    let fx: Value = serde_json::from_str(fx).unwrap();
    // 고래·리테일 경계 표와 보관 길이가 파이썬과 같은가(사본이 조용히 갈라지는 것을 막는다).
    for (s, b) in fx["bands"].as_object().unwrap() {
        let (r, w) = rtcore::supply::size_bands(s);
        assert_eq!((r, w), (b[0].as_f64().unwrap(), b[1].as_f64().unwrap()), "{s} 경계");
    }
    assert_eq!(fx["supply_1s_seconds"].as_i64().unwrap(), rtcore::supply::SUPPLY_1S_SECONDS);

    let sym = fx["symbol"].as_str().unwrap().to_string();
    let st = State::new(std::slice::from_ref(&sym), 16, 1 << 16);
    let mut rx = st.tx.subscribe();
    let rows = fx["rows"].as_array().unwrap();
    let cps = fx["checkpoints"].as_array().unwrap();
    let mut t = Tally::default();
    let (mut conn, mut ci, mut cps_done) = (0, 0, 0);
    let mut view: BTreeMap<i64, [f64; 7]> = BTreeMap::new(); // 밀기만 받은 화면 쪽(app.js applySupply1s 처럼 덮어쓴다)
    let mut pushes = 0;
    for (i, r) in rows.iter().enumerate() {
        let r = r.as_array().unwrap();
        let c = r[0].as_i64().unwrap();
        if c != conn {
            st.reset_orders(&sym); // feed::run 이 접속마다 부르는 것
            conn = c;
        }
        if let Some(m) = rtcore::feed::parse(&text(r, &sym.to_uppercase()), r[7].as_i64().unwrap() * 1000) {
            st.apply(m);
        }
        // 묶어 보내기(state.rs FLUSH_MS): 몇 사건마다 한 번 -- 흩어진 간격으로 불러 «묶어도 화면 값이 같다»를 본다. 마지막 줄 뒤엔 꼭.
        if (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 60 < 5 || i + 1 == rows.len() {
            st.flush();
        }
        while let Ok(m) = rx.try_recv() {
            if let Event::Supply1s(up) = m.ev {
                pushes += 1;
                for row in up.seconds.iter().chain(up.partial.bn.iter()) {
                    view.insert(row.0, row_vals(row));
                }
            }
        }
        if ci < cps.len() && cps[ci]["after"].as_u64().unwrap() as usize == i + 1 {
            t.cells_vs(&format!("중간 대조 {}", i + 1), &st.supply_cells(&sym).unwrap(), &cps[ci]["sec"]);
            ci += 1;
            cps_done += 1;
        }
    }
    assert_eq!(cps_done, cps.len(), "중간 대조를 다 못 했다");
    t.cells_vs("끝 1초 칸", &st.supply_cells(&sym).unwrap(), &fx["final_sec"]);

    let fin = &fx["final"];
    let Event::Snapshot { supply, .. } = st.snapshot(&sym, 0).unwrap().ev else { panic!() };
    assert_eq!(supply.now, fin["now"].as_i64().unwrap(), "now");
    t.rows_vs("스냅숏 seconds", &supply.seconds, &fin["seconds"]);
    t.rows_vs("스냅숏 partial", &[supply.partial.bn.unwrap()], &json!([fin["partial"]["bn"]]));
    assert_eq!((supply.retail_max_usd, supply.whale_min_usd), rtcore::supply::size_bands(&sym));

    // 화면 쪽: 밀기만으로 지은 칸 중 파이썬 창(now-660, now] 이 since=0 응답(seconds + partial)과 같아야 한다.
    let now = supply.now;
    let mut want: Vec<Value> = fin["seconds"].as_array().unwrap().clone();
    want.push(fin["partial"]["bn"].clone());
    let got: Vec<SecRow> = view
        .range(now - rtcore::supply::SUPPLY_1S_SECONDS + 1..=now)
        .map(|(&s, v)| (s, v[0], v[1], v[2], v[3], v[4], v[5], v[6]))
        .collect();
    t.rows_vs("밀기로 지은 화면", &got, &Value::Array(want));
    println!(
        "1초 수급 일치: 메시지 {} · 밀기 {pushes} · 중간 대조 {cps_done}회 · 비교한 초 칸 {} · 필드 {} (비트까지 같음 {})",
        rows.len(),
        t.cells,
        t.fields,
        t.bit_equal
    );
}
