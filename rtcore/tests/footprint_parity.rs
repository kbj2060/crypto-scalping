//! 풋프린트 파이썬 일치 시험. 정답지 = rtcore/tools/gen_footprint_fixture.py 가 dashboard/coin_flow.py 원본(collect_footprint →
//! footprint_add)과 server.py api_footprint·fp_dhl 옮겨 적은 것으로 만든 것.
//! 같은 합성 체결 열을 feed::parse → State::apply(실제 수신 경로)에 넣고, 흩어진 간격으로 flush(묶어 보내기)하며
//! ① 중간 대조: 봉 목록 · 그 사이 바뀐 봉의 칸(반올림 전) · 맨 끝 6봉 dhl ② 끝 칸 전체(288봉) ③ 스냅숏(= since=0, bars=144)
//! ④ 밀기만 받아 다시 지은 화면 쪽 봉 == 스냅숏(처음부터 받은 화면 · 중간에 스냅숏으로 붙은 화면 둘) 을 본다. 실수는 상대 1e-9.
//!
//! dhl 은 파이썬 가지별로 다르게 본다(footprint.rs 머리 주석 «파이썬과 다른 곳»):
//! - ring(링으로 잰 봉): 같아야 한다(없음도 같이).
//! - tape(파이썬이 테이프로 재는 닫힌 봉): 코어는 테이프가 없다 -- 링이 놓치기 직전 값을 얼린다. 값이 있으면 테이프 값과 같아야 하고,
//!   없으면(처음부터 덮인 적 없는 봉 · 얼린 뒤 늦은 체결) «몸통만»으로 센다.
//! - wait(파이썬이 아직 안 내는 6분): 코어가 먼저 내도 된다(센다).
//! - 파이썬 덮임 판정이 창 기준과 다른 순간(stale: 660초 넘게 늦은 체결이 링에 남은 동안)은 dhl 을 안 본다.
use rtcore::proto::{Event, FpBar, Footprint, Level};
use rtcore::state::State;
use serde_json::{json, Value};
use std::collections::BTreeMap;

const FX: &str = include_str!("fixtures/footprint_eth.json");

fn close(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= 1e-9 * a.abs().max(b.abs())
}

fn text(r: &[Value], sym: &str, t0: i64) -> String {
    let (kind, t_ms, tid, p, q, m, w) = (&r[1], r[2].as_i64().unwrap() + t0, &r[3], &r[4], &r[5], &r[6], r[7].as_i64().unwrap() + t0);
    if kind.as_i64() == Some(1) {
        return json!({"e": "markPriceUpdate", "E": t_ms, "s": sym}).to_string();
    }
    let x = if p.as_str() == Some("0") { "NA" } else { "MARKET" };
    json!({"e": "trade", "E": w, "T": t_ms, "s": sym, "t": tid, "p": p, "q": q, "X": x, "m": m}).to_string()
}

/// 화면 쪽: proto.rs Footprint 주석의 받는 규칙 그대로.
#[derive(Default)]
struct View {
    bars: BTreeMap<i64, (BTreeMap<u64, Level>, Option<[f64; 3]>)>,
    msgs: usize,
    bytes: usize,
}

impl View {
    fn put(&mut self, f: &Footprint) {
        self.msgs += 1;
        self.bytes += serde_json::to_string(f).unwrap().len();
        if f.full {
            self.bars.clear();
        }
        for b in &f.bars {
            let e = self.bars.entry(b.time).or_default();
            for l in &b.levels {
                e.0.insert(l.0.to_bits(), *l); // 가격 > 0 이라 비트 순서 = 값 순서
            }
            e.1 = b.dhl;
        }
        for t in &f.drop {
            self.bars.remove(t);
        }
        while self.bars.len() > f.bars_expected {
            self.bars.pop_first();
        }
    }

    fn same_as(&self, what: &str, snap: &Footprint) {
        let got: Vec<FpBar> = self.bars.iter().map(|(&t, (ls, d))| FpBar { time: t, levels: ls.values().copied().collect(), dhl: *d }).collect();
        assert_eq!(got.len(), snap.bars.len(), "{what}: 봉 수");
        for (g, w) in got.iter().zip(&snap.bars) {
            assert_eq!(g, w, "{what}: 봉 {}", w.time);
        }
    }
}

#[derive(Default)]
struct Tally {
    bars: usize,
    cells: usize,
    fields: usize,
    dhl_same: usize,
    dhl_body_only: usize,
    dhl_early: usize,
    stale_skipped: usize,
}

impl Tally {
    fn cells(&mut self, what: &str, b: i64, got: &[(i64, [f64; 6])], want: &Value) {
        let want = want.as_object().unwrap();
        let gk: Vec<i64> = got.iter().map(|c| c.0).collect();
        let mut wk: Vec<i64> = want.keys().map(|k| k.parse().unwrap()).collect();
        wk.sort_unstable();
        assert_eq!(gk, wk, "{what} 봉 {b}: 가격 칸 목록");
        self.bars += 1;
        for (k, v) in got {
            let w = want[&k.to_string()].as_array().unwrap();
            assert_eq!(w.len(), 6);
            for (j, (g, w)) in v.iter().zip(w).enumerate() {
                let w = w.as_f64().unwrap();
                assert!(close(*g, w), "{what} 봉 {b} 칸 {k}[{j}]: rust {g} != python {w}");
                self.fields += 1;
            }
            self.cells += 1;
        }
    }

    fn dhl(&mut self, what: &str, b: i64, kind: &str, py: &Value, rust: Option<[f64; 3]>) {
        let py: Option<[f64; 3]> = py.as_array().map(|a| [a[0].as_f64().unwrap(), a[1].as_f64().unwrap(), a[2].as_f64().unwrap()]);
        let eq = |a: [f64; 3], b: [f64; 3]| a == b; // round(x, 3) 끝 값이라 비트까지 같아야 한다
        match (kind, py, rust) {
            ("ring", None, None) => self.dhl_same += 1,
            ("ring" | "tape", Some(p), Some(r)) if eq(p, r) => self.dhl_same += 1,
            ("tape", _, None) => self.dhl_body_only += 1,
            ("wait", None, Some(_)) => self.dhl_early += 1,
            ("wait", None, None) => self.dhl_same += 1,
            _ => panic!("{what} 봉 {b} dhl({kind}): rust {rust:?} != python {py:?}"),
        }
    }
}

fn snap_of(st: &State, sym: &str) -> Footprint {
    let Event::Snapshot { footprint, .. } = st.snapshot(sym, 0).unwrap().ev else { panic!() };
    footprint.expect("칸 폭 표에 있는 종목은 풋프린트가 있다")
}

#[test]
fn footprint_matches_python() {
    check(FX);
}

// 칸 폭이 작은 종목(0.0003 · 가격 4자리)과 가격이 큰 종목(칸 15 · 가격 0자리) -- ETH 칸 폭에선 값이 안 바뀌는 변형(응답 가격 반올림 등)을 잡는다.
#[test]
fn footprint_matches_python_xrp() {
    check(&gen_fixture("footprint_xrp.json"));
}

#[test]
fn footprint_matches_python_btc() {
    check(&gen_fixture("footprint_btc.json"));
}

/// 커밋 안 한 정답지(XRP·BTC -- 합쳐 약 1.9MB). 없으면 저장소 루트에서 `python3 rtcore/tools/gen_footprint_fixture.py xrp`(btc 도)로 먼저 만든다.
fn gen_fixture(name: &str) -> String {
    let p = format!("{}/tests/fixtures/gen/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p}: {e} -- 생성기로 먼저 만든다(이 함수 주석)"))
}

fn check(fx: &str) {
    let fx: Value = serde_json::from_str(fx).unwrap();
    // 파이썬 표·상수의 사본이 갈라지지 않았는가.
    for (s, v) in fx["specs"].as_object().unwrap() {
        let want = (v[0].as_f64().unwrap(), v[1].as_u64().unwrap() as usize);
        assert_eq!(rtcore::footprint::spec(s), Some(want), "{s} 칸 폭·소수 자리");
    }
    assert_eq!(fx["keep_bars"].as_i64().unwrap(), rtcore::footprint::KEEP_BARS);
    assert_eq!(fx["window_bars"].as_u64().unwrap() as usize, rtcore::footprint::WINDOW_BARS);
    assert_eq!(fx["bar_seconds"].as_i64().unwrap(), rtcore::footprint::BAR_SECONDS);

    let sym = fx["symbol"].as_str().unwrap().to_string();
    let t0 = fx["t0"].as_i64().unwrap();
    let st = State::new(std::slice::from_ref(&sym), 16, 1 << 16);
    let mut rx = st.tx.subscribe();
    let rows = fx["rows"].as_array().unwrap();
    let cps = fx["checkpoints"].as_array().unwrap();
    let mut t = Tally::default();
    let (mut conn, mut ci) = (0, 0);
    let mut view = View::default(); // 처음부터 밀기만 받은 화면
    let mut late: Option<(u64, View)> = None; // 중간(5번째 대조)에 스냅숏으로 붙은 화면
    let mut drops = 0;
    let drain = |rx: &mut tokio::sync::broadcast::Receiver<_>, view: &mut View, late: &mut Option<(u64, View)>, drops: &mut usize| {
        while let Ok(m) = rx.try_recv() {
            let m: rtcore::proto::Stamped = m;
            if let Event::Footprint(f) = &m.ev {
                assert!(!f.full, "밀기는 늘 바뀐 몫");
                *drops += f.drop.len();
                view.put(f);
                if let Some((seq0, v)) = late {
                    if m.seq > *seq0 {
                        v.put(f);
                    }
                }
            }
        }
    };
    for (i, r) in rows.iter().enumerate() {
        let r = r.as_array().unwrap();
        let c = r[0].as_i64().unwrap();
        if c != conn {
            st.reset_orders(&sym);
            conn = c;
        }
        if let Some(m) = rtcore::feed::parse(&text(r, &sym.to_uppercase(), t0), (r[7].as_i64().unwrap() + t0) * 1000) {
            st.apply(m);
        }
        if (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 60 < 3 {
            st.flush();
        }
        drain(&mut rx, &mut view, &mut late, &mut drops);
        if ci < cps.len() && cps[ci]["after"].as_u64().unwrap() as usize == i + 1 {
            let cp = &cps[ci];
            let what = format!("중간 대조 {}", i + 1);
            let raw = st.footprint_cells(&sym).unwrap();
            let keys: Vec<i64> = raw.iter().map(|b| b.0).collect();
            let want: Vec<i64> = cp["bars"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
            assert_eq!(keys, want, "{what}: 봉 목록");
            for (b, cells) in cp["cells"].as_object().unwrap() {
                let b: i64 = b.parse().unwrap();
                let got = &raw.iter().find(|x| x.0 == b).unwrap().1;
                t.cells(&what, b, got, cells);
            }
            if ci == 5 {
                // 모으던 것을 아직 안 민 채 스냅숏으로 붙는다 -- 다음 묶음이 같은 값을 다시 덮어써도 같아야 한다.
                let Event::Snapshot { footprint, .. } = st.snapshot(&sym, 0).unwrap().ev else { panic!() };
                let mut v = View::default();
                v.put(&footprint.unwrap());
                late = Some((st.snapshot(&sym, 0).unwrap().seq, v));
            }
            let snap = snap_of(&st, &sym);
            if cp["stale"].as_bool().unwrap() {
                t.stale_skipped += 1;
            } else {
                for d in cp["dhl"].as_array().unwrap() {
                    let b = d[0].as_i64().unwrap();
                    let rust = snap.bars.iter().find(|x| x.time == b).expect("창 안 봉").dhl;
                    t.dhl(&what, b, d[1].as_str().unwrap(), &d[2], rust);
                }
            }
            st.flush();
            drain(&mut rx, &mut view, &mut late, &mut drops);
            view.same_as(&format!("{what} 밀기로 지은 화면"), &snap);
            ci += 1;
        }
    }
    assert_eq!(ci, cps.len(), "중간 대조를 다 못 했다");
    st.flush();
    drain(&mut rx, &mut view, &mut late, &mut drops);

    let raw = st.footprint_cells(&sym).unwrap();
    let fc = fx["final_cells"].as_object().unwrap();
    assert_eq!(raw.len(), fc.len(), "끝 봉 수");
    for (b, cells) in &raw {
        t.cells("끝 칸", *b, cells, &fc[&b.to_string()]);
    }

    let fin = &fx["final"];
    let snap = snap_of(&st, &sym);
    assert!(snap.full);
    assert_eq!(snap.bucket, fin["bucket"].as_f64().unwrap());
    assert_eq!(snap.bar_seconds, fin["barSeconds"].as_i64().unwrap());
    assert_eq!(snap.bars_expected as i64, fin["barsExpected"].as_i64().unwrap());
    assert_eq!((snap.retail_max_usd, snap.whale_min_usd), rtcore::supply::size_bands(&sym));
    let fb = fin["bars"].as_array().unwrap();
    assert_eq!(snap.bars.len(), fb.len(), "스냅숏 봉 수");
    let mut body_only = vec![];
    let (mut levels, mut tf) = (0, Tally::default());
    for (g, w) in snap.bars.iter().zip(fb) {
        assert_eq!(g.time, w["time"].as_i64().unwrap(), "스냅숏 봉 시각");
        let wl = w["levels"].as_array().unwrap();
        assert_eq!(g.levels.len(), wl.len(), "스냅숏 봉 {} 칸 수", g.time);
        for (l, wl) in g.levels.iter().zip(wl) {
            let gv = [l.0, l.1, l.2, l.3, l.4, l.5, l.6];
            for (j, (a, b)) in gv.iter().zip(wl.as_array().unwrap()).enumerate() {
                // 응답 값은 round(x, 3) 끝이라 비트까지 같아야 한다(입력 수량이 소수 3자리라 1e-9 로는 반올림 빠짐을 못 잡는다)
                assert_eq!(a.to_bits(), b.as_f64().unwrap().to_bits(), "스냅숏 봉 {} 칸 {:?}[{j}]: rust {a} != python {b}", g.time, l);
            }
            levels += 1;
        }
        let before = tf.dhl_body_only;
        tf.dhl("스냅숏", g.time, w["dhl"][0].as_str().unwrap(), &w["dhl"][1], g.dhl);
        if tf.dhl_body_only > before {
            body_only.push((g.time - fb[0]["time"].as_i64().unwrap()) / 300);
        }
    }
    println!("끝 스냅숏 dhl: 같음 {} · 몸통만 {} (창 첫 봉에서 몇 번째 봉: {body_only:?}) · 코어가 먼저 {}", tf.dhl_same, tf.dhl_body_only, tf.dhl_early);
    // 몸통만은 정답지가 일부러 만든 두 봉뿐: 늦은 체결이 새로 만든 빈 봉(450) · 얼린 뒤 늦은 체결이 온 봉(560). 늘면 규칙이 갈라진 것.
    assert_eq!(body_only, vec![30, 140], "끝 스냅숏 «몸통만» 봉");
    view.same_as("끝 밀기로 지은 화면(처음부터)", &snap);
    late.expect("중간 스냅숏").1.same_as("끝 밀기로 지은 화면(중간에 붙음)", &snap);
    assert!(drops > 0, "정답지가 «창 안 봉이 시각으로 지워짐»을 안 건드린다");
    assert!(t.dhl_same > 50 && t.stale_skipped > 0, "dhl 대조가 너무 적다");
    println!(
        "풋프린트 일치: 메시지 {} · 중간 대조 {} · 비교한 봉 {} · 칸 {} · 필드 {} · 스냅숏 봉 {} 칸 {levels} · \
         dhl 같음 {} · 몸통만(테이프 없음) {} · 코어가 먼저 냄 {} · stale 로 dhl 안 본 대조 {} · 밀기 {}건 {}B · drop {drops}",
        rows.len(),
        cps.len(),
        t.bars,
        t.cells,
        t.fields,
        snap.bars.len(),
        t.dhl_same,
        t.dhl_body_only,
        t.dhl_early,
        t.stale_skipped,
        view.msgs,
        view.bytes
    );
}
