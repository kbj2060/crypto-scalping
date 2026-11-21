//! OKX 풋프린트 봉 복원(restore.rs read_okx_fp · Supply::okx_fp_attempt) ↔ 원본 coin_flow.okx_footprint_restore 걸음별 일치.
//! 정답지: tools/gen_okx_fp_fixture.py(저장소 루트에서 먼저 돌린다 -- 커밋 안 함 · tests/fixtures/gen/okx_fp/).
//! 같은 DB 사본(okx_tape_start.sqlite)에서 시작한다. 걸음 k 의 시각까지 온 실시간 OKX 메시지를 넣고, 그 회차의 1분봉 대조 고침을 한 뒤 회차를 부른다.
//! 그다음 봉 칸(비트)·first_bar·OI 5분봉·청산 링·청산 원 OKX 몫을 원본 상태와 대조한다. Bybit 5분봉은 재시작 때 한 번.
use rtcore::restore::Cfg;
use rtcore::state::State;
use serde_json::Value;
use std::path::PathBuf;
use std::time::Duration;

fn gen_dir_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gen/okx_fp")
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

#[test]
fn okx_fp_restore_matches_python_step_by_step() {
    for v in ["a", "b"] {
        one(v);
    }
}

fn one(variant: &str) {
    let gen_dir = || gen_dir_root().join(variant);
    let p = gen_dir().join("okx_fp_sol.json");
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| {
        panic!("{}: {e} -- 저장소 루트에서 `python3 rtcore/tools/gen_okx_fp_fixture.py` 로 먼저 만든다(커밋 안 함)", p.display())
    }))
    .unwrap();
    let sym = fx["symbol"].as_str().unwrap().to_string();
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("okx_fp_{variant}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let tape = dir.join("okx_tape.sqlite");
    std::fs::copy(gen_dir().join("okx_tape_start.sqlite"), &tape).unwrap();
    let cfg = Cfg {
        snap_dir: dir.clone(),
        tape: dir.join("없는_테이프.sqlite"),
        tries: 1,
        wait: Duration::ZERO,
        okx_tape: tape.clone(),
        okx_ctx: gen_dir().join("okx_ctx.sqlite"),
        bn_ctx: dir.join("없는_맥락.sqlite"),
        bybit_liq: gen_dir().join("bybit_liq.sqlite"),
    };
    let r_us = (f(&fx["restart_s"]) * 1e6) as i64;
    let (st, _) = State::with_restore(std::slice::from_ref(&sym), 16, 1 << 16, cfg, r_us);

    // Bybit 5분봉(재시작 때 DB 에서 288봉 · 수집 시작이 걸친 봉 «뒤»만)
    let rtcore::proto::Event::Snapshot { liq5m, .. } = st.snapshot(&sym, r_us).unwrap().ev else { panic!() };
    let by = liq5m.unwrap().bybit.unwrap();
    let want = &fx["bybit5m"];
    assert_eq!(by.from, want["from"].as_i64(), "Bybit 수집 시작 봉");
    let wb: Vec<(i64, f64, f64, i64)> =
        want["bars"].as_object().unwrap().iter().map(|(b, v)| (b.parse().unwrap(), f(&v[0]), f(&v[1]), v[2].as_i64().unwrap())).collect();
    assert_eq!(by.bars, wb, "Bybit 5분봉");

    let live = fx["live"].as_array().unwrap();
    let (mut li, mut fields) = (0, 0usize);
    for (k, step) in fx["steps"].as_array().unwrap().iter().enumerate() {
        let now = f(&step["now"]);
        while li < live.len() && f(&live[li][0]) <= now {
            for x in rtcore::feed::parse_okx(live[li][1].as_str().unwrap(), (f(&live[li][0]) * 1e6) as i64) {
                st.apply(x);
            }
            li += 1;
        }
        if let Some(rows) = fx["fix"].get((k + 1).to_string()) {
            let con = rusqlite::Connection::open(&tape).unwrap();
            for r in rows.as_array().unwrap() {
                let args: Vec<rusqlite::types::Value> = r[1]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| match v {
                        Value::String(s) => rusqlite::types::Value::Text(s.clone()),
                        Value::Number(n) if n.is_i64() => rusqlite::types::Value::Integer(n.as_i64().unwrap()),
                        Value::Number(n) => rusqlite::types::Value::Real(n.as_f64().unwrap()),
                        _ => rusqlite::types::Value::Null,
                    })
                    .collect();
                con.execute(r[0].as_str().unwrap(), rusqlite::params_from_iter(args)).unwrap();
            }
        }
        let got = st.run_restore_okx_fp(&sym, (now * 1e6) as i64);
        assert!(got.is_some(), "걸음 {k}: 회차가 돌아야 한다");
        let (bars, first_bar, oi) = st.okx_fp(&sym).unwrap();
        assert_eq!(first_bar, step["first_bar"].as_i64().unwrap(), "걸음 {k} first_bar");
        let wbars = step["bars"].as_object().unwrap();
        assert_eq!(bars.len(), wbars.len(), "걸음 {k} 봉 수");
        for (b, cells) in &bars {
            let w = wbars[&b.to_string()].as_object().unwrap();
            assert_eq!(cells.len(), w.len(), "걸음 {k} 봉 {b} 칸 수");
            for (key, c) in cells {
                let wc = w[&key.to_string()].as_array().unwrap();
                for j in 0..6 {
                    assert_eq!(c[j].to_bits(), f(&wc[j]).to_bits(), "걸음 {k} 봉 {b} 칸 {key}[{j}]");
                    fields += 1;
                }
            }
        }
        let woi = step["oi5m"].as_object().unwrap();
        assert_eq!(oi.len(), woi.len(), "걸음 {k} OI 봉 수");
        for (b, v) in &oi {
            let w = &woi[&b.to_string()];
            assert_eq!((v[0].to_bits(), v[1].to_bits()), (f(&w[0]).to_bits(), f(&w[1]).to_bits()), "걸음 {k} OI 봉 {b}");
            fields += 2;
        }
        let ring = st.liq_events(&sym, rtcore::proto::Venue::Okx).unwrap();
        let wl = step["liq"].as_array().unwrap();
        assert_eq!(ring.len(), wl.len(), "걸음 {k} 청산 링 건수");
        for (g, w) in ring.iter().zip(wl) {
            assert_eq!((g.0, g.1, g.2.to_bits(), g.3.to_bits()), (w[0].as_i64().unwrap(), w[1].as_bool().unwrap(), f(&w[2]).to_bits(), f(&w[3]).to_bits()), "걸음 {k} 청산");
            fields += 4;
        }
        let rtcore::proto::Event::Snapshot { liq5m, .. } = st.snapshot(&sym, (now * 1e6) as i64).unwrap().ev else { panic!() };
        let ok = liq5m.unwrap().okx.unwrap();
        assert_eq!(ok.from, (first_bar != 0).then_some(first_bar), "걸음 {k} 청산 원 OKX 경계");
        let wo: Vec<(i64, f64, f64, i64)> =
            step["okx5m"].as_object().unwrap().iter().map(|(b, v)| (b.parse().unwrap(), f(&v[0]), f(&v[1]), v[2].as_i64().unwrap())).collect();
        assert_eq!(ok.bars, wo, "걸음 {k} 청산 원 OKX 몫");
        fields += wo.len() * 4;
    }
    assert!(st.run_restore_okx_fp(&sym, r_us).is_none(), "8회차 뒤에는 더 안 돈다");
    println!("OKX 풋프린트 복원 걸음별 일치(변형 {variant}): 걸음 {} · 필드 {fields} · Bybit 봉 {}", fx["steps"].as_array().unwrap().len(), wb.len());
    let _ = std::fs::remove_dir_all(&dir);
}
