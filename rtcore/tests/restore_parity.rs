//! 재시작 복원 일치 시험(2026-11-21 · src/restore.rs). 정답지 = tools/gen_restore_fixture.py(시험 전에 만든다 -- 커밋 안 함):
//! 합성 SOL `@trade` 열 · 그 열을 **원본 수집기**가 쓴 테이프 SQLite 세 벌(full · gaps · behind) · 원본 coin_flow 복원 함수가 되살린 봉·1초 칸·dhl.
//! (a) 왕복: 임의 시점 상태 → 스냅숏 파일 → 새 코어 → 상태가 비트까지 같다 · 그 뒤 같은 체결을 넣은 결과(스냅숏·밀기로 지은 화면)까지 같다.
//! (b) 테이프 메우기: 사례 11개(공백 짧음 · 봉 경계 · 11분 넘음 · 24시간 넘음 · 테이프 gaps · 스냅숏 없음 · 테이프 없음 · 둘 다 없음 ·
//!     판 번호 다름 · 깨짐 · 테이프가 덜 따라옴) -- 재시작 전 코어가 쓴 스냅숏 + 같은 테이프로 되살린 값 == 파이썬(실수 상대 1e-9 · dhl 은 round 3 끝 값 비트).
//! (c) 이어 붙이기: 계속 돈 코어 대 중간에 죽었다 되살린 코어 -- 다른 곳이 경계 봉 둘(+ 테이프가 덜 따라온 구간)뿐인지, 얼마나 다른지.
use rtcore::proto::{Event, Footprint, FpBar, Level, SecRow, Stamped, Supply1s};
use rtcore::restore::{self, Cfg};
use rtcore::state::State;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

const BS: i64 = 300;

struct Row {
    t: i64,
    tid: u64,
    p: String,
    q: String,
    m: bool,
    w: i64,
}

struct Fx {
    sym: String,
    rows: Vec<Row>,
    cases: Vec<Value>,
    tapes: BTreeMap<String, String>,
    raw: Value,
}

fn gen_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gen/restore")
}

fn fx() -> &'static Fx {
    static FX: OnceLock<Fx> = OnceLock::new();
    FX.get_or_init(|| {
        let p = gen_dir().join("restore_sol.json");
        let text = std::fs::read_to_string(&p)
            .unwrap_or_else(|e| panic!("{}: {e} -- 저장소 루트에서 `python3 rtcore/tools/gen_restore_fixture.py` 로 먼저 만든다(커밋 안 함)", p.display()));
        let v: Value = serde_json::from_str(&text).unwrap();
        let t0 = v["t0"].as_i64().unwrap();
        let rows = v["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| Row {
                t: r[0].as_i64().unwrap() + t0,
                tid: r[1].as_u64().unwrap(),
                p: r[2].as_str().unwrap().into(),
                q: r[3].as_str().unwrap().into(),
                m: r[4].as_bool().unwrap(),
                w: r[5].as_i64().unwrap() + t0,
            })
            .collect();
        let tapes = v["tapes"].as_object().unwrap().iter().map(|(k, x)| (k.clone(), x.as_str().unwrap().to_string())).collect();
        Fx { sym: v["symbol"].as_str().unwrap().into(), rows, cases: v["cases"].as_array().unwrap().clone(), tapes, raw: v }
    })
}

fn case(name: &str) -> &'static Value {
    fx().cases.iter().find(|c| c["name"] == name).unwrap()
}

fn close(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= 1e-9 * a.abs().max(b.abs())
}

/// 부를 때마다 새 임시 폴더(target/tmp 아래 -- 실제 data/ 는 안 건드린다). 시험이 나란히 돌아도 같은 사례가 폴더를 나누지 않게 번호를 붙인다.
fn tmp(name: &str) -> PathBuf {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("restore_{}_{n}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn feed(st: &State, rows: &[Row]) {
    let s = fx().sym.to_uppercase();
    for r in rows {
        let text = json!({"e": "trade", "E": r.w, "T": r.t, "s": s, "t": r.tid, "p": r.p, "q": r.q, "X": "MARKET", "m": r.m}).to_string();
        if let Some(m) = rtcore::feed::parse(&text, r.w * 1000) {
            st.apply(m);
        }
    }
}

fn cfg(snap_dir: &Path, tape: PathBuf) -> Cfg {
    Cfg { snap_dir: snap_dir.into(), tape, tries: 2, wait: Duration::ZERO }
}

/// 재시작 전 코어(메우기 없는 라이브와 같게: 테이프가 없어 첫 체결 메우기는 «구멍»만 남긴다) -- 파이썬 재시작 전 판과 같은 상태.
fn core_live(rows: &[Row], snap_dir: &Path) -> Arc<State> {
    let (a, _) = State::with_restore(std::slice::from_ref(&fx().sym), 16, 1 << 16, cfg(snap_dir, snap_dir.join("없는_테이프.sqlite")), rows[0].w * 1000);
    feed(&a, &rows[..1]);
    a.run_restore(&fx().sym).expect("첫 체결이 일감을 만든다");
    feed(&a, &rows[1..]);
    a
}

fn snap_of(st: &State) -> (u64, Footprint, Supply1s) {
    let m = st.snapshot(&fx().sym, 0).unwrap();
    let Event::Snapshot { footprint, supply, .. } = m.ev else { panic!() };
    (m.seq, footprint.unwrap(), supply)
}

/// 창 안 봉 중 구간들에 걸친 봉(코어 holes 기대값).
fn holey(fp: &Footprint, hs: &[(i64, i64)]) -> Vec<i64> {
    fp.bars.iter().map(|b| b.time).filter(|&b| hs.iter().any(|&(a, z)| b < z && b + BS > a)).collect()
}

/// 화면 쪽(proto.rs Footprint 받는 규칙 + holes 는 메시지마다 통째로) · 1초 칸(초 단위 덮어쓰기).
#[derive(Default)]
struct View {
    bars: BTreeMap<i64, (BTreeMap<u64, Level>, Option<[f64; 3]>)>,
    holes: Vec<i64>,
    secs: BTreeMap<i64, SecRow>,
}

impl View {
    fn put_fp(&mut self, f: &Footprint) {
        if f.full {
            self.bars.clear();
        }
        for b in &f.bars {
            let e = self.bars.entry(b.time).or_default();
            for l in &b.levels {
                e.0.insert(l.0.to_bits(), *l);
            }
            e.1 = b.dhl;
        }
        for t in &f.drop {
            self.bars.remove(t);
        }
        while self.bars.len() > f.bars_expected {
            self.bars.pop_first();
        }
        self.holes = f.holes.clone();
    }

    fn put_sup(&mut self, s: &Supply1s) {
        for r in s.seconds.iter().chain(s.partial.bn.iter()) {
            self.secs.insert(r.0, *r);
        }
    }

    fn drain(&mut self, rx: &mut tokio::sync::broadcast::Receiver<Stamped>, seq0: u64) {
        while let Ok(m) = rx.try_recv() {
            if m.seq <= seq0 {
                continue;
            }
            match &m.ev {
                Event::Footprint(f) => self.put_fp(f),
                Event::Supply1s(s) => self.put_sup(s),
                _ => {}
            }
        }
    }

    fn same_as(&self, what: &str, fp: &Footprint, sup: &Supply1s) {
        let got: Vec<FpBar> = self.bars.iter().map(|(&t, (ls, d))| FpBar { time: t, levels: ls.values().copied().collect(), dhl: *d }).collect();
        assert_eq!(got, fp.bars, "{what}: 밀기로 지은 봉 != 스냅숏");
        assert_eq!(self.holes, fp.holes, "{what}: 밀기로 받은 holes != 스냅숏");
        for r in sup.seconds.iter().chain(sup.partial.bn.iter()) {
            assert_eq!(self.secs.get(&r.0), Some(r), "{what}: 밀기로 받은 1초 칸 {} != 스냅숏", r.0);
        }
    }
}

#[derive(Default, Debug)]
struct Tally {
    bars: usize,
    cells: usize,
    fields: usize,
    secs: usize,
    sec_fields: usize,
    sec_snapshot_only: usize,
    dhl_same: usize,
    dhl_early: usize,
    dhl_snapshot: usize,
    /// 파이썬은 «없음», 코어는 스냅숏 값을 낸 봉(T0 부터 초 · 처음 본 것만).
    dhl_snapshot_at: BTreeSet<i64>,
    dhl_boundary: usize,
    holey: usize,
}

fn parse_cells(v: &Value) -> BTreeMap<i64, [f64; 6]> {
    v.as_object().unwrap().iter().map(|(k, c)| (k.parse().unwrap(), std::array::from_fn(|j| c[j].as_f64().unwrap()))).collect()
}

fn dhl_of(v: &Value) -> Option<[f64; 3]> {
    v.as_array().map(|a| [a[0].as_f64().unwrap(), a[1].as_f64().unwrap(), a[2].as_f64().unwrap()])
}

/// 코어 상태 대 파이썬 대조 한 번.
fn compare(what: &str, st: &State, cp: &Value, snap: Option<&restore::Snap>, boundary: Option<i64>, holes: &[(i64, i64)], t: &mut Tally) {
    let sym = &fx().sym;
    // 봉 칸(반올림 전)
    let raw = st.footprint_cells(sym).unwrap();
    let py = cp["bars"].as_object().unwrap();
    let pk: Vec<i64> = py.keys().map(|k| k.parse().unwrap()).collect::<BTreeSet<_>>().into_iter().collect();
    assert_eq!(raw.iter().map(|b| b.0).collect::<Vec<_>>(), pk, "{what}: 봉 목록");
    for (b, cells) in &raw {
        let want = parse_cells(&py[&b.to_string()]);
        assert_eq!(cells.iter().map(|c| c.0).collect::<Vec<_>>(), want.keys().copied().collect::<Vec<_>>(), "{what} 봉 {b}: 칸 키");
        for (k, c) in cells {
            for j in 0..6 {
                assert!(close(c[j], want[k][j]), "{what} 봉 {b} 칸 {k}[{j}]: rust {} != python {}", c[j], want[k][j]);
                t.fields += 1;
            }
            t.cells += 1;
        }
        t.bars += 1;
    }
    // 1초 칸
    let sec: BTreeMap<i64, [f64; 7]> = st.supply_cells(sym).unwrap().into_iter().collect();
    let ssec: BTreeMap<i64, [f64; 7]> = snap.map_or(BTreeMap::new(), |s| s.sec.iter().copied().collect());
    let py = cp["sec"].as_object().unwrap();
    for (s, v) in py {
        let s: i64 = s.parse().unwrap();
        let c = sec.get(&s).unwrap_or_else(|| panic!("{what}: 1초 칸 {s} 이 코어에 없다"));
        for j in 0..7 {
            let w = v[j].as_f64().unwrap();
            assert!(close(c[j], w), "{what} 1초 {s}[{j}]: rust {} != python {w}", c[j]);
            t.sec_fields += 1;
        }
        t.secs += 1;
    }
    for (s, c) in &sec {
        if !py.contains_key(&s.to_string()) {
            // 파이썬은 테이프가 없으면 1초 칸을 안 되살린다 -- 코어는 스냅숏 값(그대로)으로 메운다
            assert_eq!(ssec.get(s).map(|x| x.map(f64::to_bits)), Some(c.map(f64::to_bits)), "{what}: 코어에만 있는 1초 칸 {s} 이 스냅숏 값이 아니다");
            t.sec_snapshot_only += 1;
        }
    }
    // dhl(round 3 끝 값 -- 비트)
    let (_, fp, _) = snap_of(st);
    let sdhl: BTreeMap<i64, Option<[f64; 3]>> = snap.map_or(BTreeMap::new(), |s| s.bars.iter().map(|b| (b.0, b.2)).collect());
    let mut seen = 0;
    for d in cp["dhl"].as_array().unwrap() {
        let (b, kind, pv) = (d[0].as_i64().unwrap(), d[1].as_str().unwrap(), dhl_of(&d[2]));
        let cv = fp.bars.iter().find(|x| x.time == b).unwrap_or_else(|| panic!("{what}: 창 봉 {b} 이 코어 스냅숏에 없다")).dhl;
        seen += 1;
        match (pv, cv) {
            (p, c) if p == c => t.dhl_same += 1,
            (None, Some(_)) if kind == "wait" => t.dhl_early += 1,
            (None, Some(c)) if sdhl.get(&b) == Some(&Some(c)) => {
                t.dhl_snapshot += 1;
                t.dhl_snapshot_at.insert(b - fx().raw["t0"].as_i64().unwrap() / 1000);
            }
            _ if boundary == Some(b) => t.dhl_boundary += 1,
            _ => panic!("{what} 봉 {b} dhl({kind}): rust {cv:?} != python {pv:?}"),
        }
    }
    assert_eq!(seen, fp.bars.len(), "{what}: dhl 대조 봉 수");
    // 구멍(코어만 내는 표시)
    assert_eq!(fp.holes, holey(&fp, holes), "{what}: holes");
    t.holey += fp.holes.len();
}

fn ranges(v: &Value) -> Vec<(i64, i64)> {
    v.as_array().unwrap().iter().map(|h| (h[0].as_i64().unwrap(), h[1].as_i64().unwrap())).collect()
}

/// 사례 하나를 재시작 뒤 `upto` 행까지 돌린다(b · c 공용). 돌려주는 것 = 되살린 코어 · 재시작 전 코어 스냅숏.
fn restart(c: &Value, upto: usize, t: Option<&mut Tally>) -> (Arc<State>, Option<restore::Snap>) {
    let f = fx();
    let sym = &f.sym;
    let name = c["name"].as_str().unwrap();
    let dir = tmp(name);
    let snap_dir = dir.join("b");
    std::fs::create_dir_all(&snap_dir).unwrap();
    let mut snap = None;
    if let Some(s_idx) = c["S_idx"].as_u64() {
        let a = core_live(&f.rows[..s_idx as usize], &dir.join("a"));
        assert_eq!(a.write_snapshots(), 1, "{name}: 재시작 전 코어가 스냅숏을 써야 한다");
        let file = restore::snap_path(&dir.join("a"), sym);
        assert!(!file.with_extension("json.tmp").exists(), "임시 파일이 남았다");
        let bytes = std::fs::read(&file).unwrap();
        let s: restore::Snap = serde_json::from_slice(&bytes).unwrap();
        // 재시작 전 코어 스냅숏 == 파이썬 footprint_save(같은 라이브 입력 -- 봉 144 · last_ms)
        let py = &c["py_snapshot"];
        assert_eq!(s.last_ms, py["last_ms"].as_i64().unwrap(), "{name}: last_ms");
        let pb = py["bars"].as_object().unwrap();
        assert_eq!(s.bars.len(), pb.len(), "{name}: 스냅숏 봉 수");
        for (b, cells, _) in &s.bars {
            let want = parse_cells(&pb[&b.to_string()]);
            assert_eq!(cells.len(), want.len(), "{name}: 스냅숏 봉 {b} 칸 수");
            for (k, v) in cells {
                assert!((0..6).all(|j| close(v[j], want[k][j])), "{name}: 스냅숏 봉 {b} 칸 {k}");
            }
        }
        let out = match c["snap"].as_str().unwrap() {
            "ok" => bytes.clone(),
            "badver" => String::from_utf8(bytes.clone()).unwrap().replacen("\"v\":1,", "\"v\":999,", 1).into_bytes(),
            "corrupt" => bytes[..bytes.len() / 2].to_vec(),
            x => panic!("{x}"),
        };
        assert_ne!(out.len(), 0);
        std::fs::write(restore::snap_path(&snap_dir, sym), out).unwrap();
        snap = (c["snap"] == "ok").then_some(s);
    }
    let tape = match c["tape"].as_str().unwrap() {
        "none" => dir.join("없는_테이프.sqlite"),
        v => gen_dir().join(&f.tapes[v]),
    };
    let (b, _) = State::with_restore(std::slice::from_ref(sym), 16, 1 << 16, cfg(&snap_dir, tape), c["restart_ms"].as_i64().unwrap() * 1000);
    let mut rx = b.tx.subscribe();
    let (r, m, e) = (c["R_idx"].as_u64().unwrap() as usize, c["M"].as_u64().unwrap() as usize, c["E_idx"].as_u64().unwrap() as usize);
    feed(&b, &f.rows[r..r + m]);
    let (lo, hi) = (c["lo"].as_i64().unwrap(), c["hi"].as_i64().unwrap());
    // 메우는 중: 공백 [lo, hi) 에 걸친 봉이 holes 로 보인다
    let (seq0, fp0, sup0) = snap_of(&b);
    assert_eq!(fp0.holes, holey(&fp0, &[(lo, hi)]), "{name}: 메우는 중 holes");
    let mut view = View::default();
    view.put_fp(&fp0);
    view.put_sup(&sup0);
    b.flush();
    let line = b.run_restore(sym).expect("일감");
    assert!(line.starts_with(&format!("공백 [{lo}, {hi})")), "{name}: 경계가 파이썬(footprint_tape_restore)과 다르다: {line}");
    assert!(b.run_restore(sym).is_none(), "한 번 합치면 일감이 없다");
    b.flush();
    view.drain(&mut rx, seq0);
    let (_, fp1, sup1) = snap_of(&b);
    view.same_as(&format!("{name} 합친 뒤"), &fp1, &sup1);
    let boundary = Some((f.rows[r].t.div_euclid(1000)).div_euclid(BS) * BS);
    let holes = ranges(&c["holes"]);
    if let Some(t) = t {
        compare(&format!("{name} 합친 직후"), &b, &c["merge"], snap.as_ref(), None, &holes, t);
        feed(&b, &f.rows[r + m..e]);
        compare(&format!("{name} 재시작 + 420초"), &b, &c["end"], snap.as_ref(), boundary, &holes, t);
        feed(&b, &f.rows[e..upto]);
    } else {
        feed(&b, &f.rows[r + m..upto]);
    }
    (b, snap)
}

#[test]
fn restore_matches_python() {
    let f = fx();
    // 파이썬 표 사본이 갈라지지 않았는가
    for (s, v) in f.raw["tape_buckets"].as_object().unwrap() {
        assert_eq!(restore::tape_bucket(s), v.as_f64(), "{s} 테이프 칸 폭");
    }
    assert_eq!(f.raw["window_bars"].as_u64().unwrap() as usize, rtcore::footprint::WINDOW_BARS);
    assert_eq!(f.raw["supply_1s_seconds"].as_i64().unwrap(), rtcore::supply::SUPPLY_1S_SECONDS);
    let mut all = Tally::default();
    for c in &f.cases {
        let mut t = Tally::default();
        let e = c["E_idx"].as_u64().unwrap() as usize;
        restart(c, e, Some(&mut t));
        println!("{}: {t:?}", c["name"].as_str().unwrap());
        all.bars += t.bars;
        all.cells += t.cells;
        all.fields += t.fields;
        all.secs += t.secs;
        all.sec_fields += t.sec_fields;
        all.sec_snapshot_only += t.sec_snapshot_only;
        all.dhl_same += t.dhl_same;
        all.dhl_early += t.dhl_early;
        all.dhl_snapshot += t.dhl_snapshot;
        all.dhl_boundary += t.dhl_boundary;
        all.holey += t.holey;
    }
    println!("복원 일치 합계 사례 {} · {all:?}", f.cases.len());
    assert!(all.holey > 0 && all.sec_snapshot_only > 0 && all.dhl_snapshot > 0, "대조가 특수 갈래를 안 지났다");
}

/// (a) 왕복: 상태 → 스냅숏 파일 → 새 코어. 봉·1초 칸·dhl 이 비트까지 같고, 같은 다음 체결을 넣은 결과도 같다.
#[test]
fn snapshot_roundtrip() {
    let f = fx();
    let sym = &f.sym;
    let n = f.rows.len();
    let first_at = |sec: f64| f.rows.iter().position(|r| r.t >= f.rows[0].t + (sec * 1000.0) as i64).unwrap();
    // 다음 체결이 새 초에 오는 자리만(같은 초면 그 초는 «붙은 첫 초» -- 실시간 몫만 남는 경계 규칙이라 왕복 대상이 아니다)
    let new_sec = |i: usize| (i..n).find(|&j| f.rows[j].t.div_euclid(1000) > f.rows[j - 1].t.div_euclid(1000)).unwrap();
    let points = [
        ("스냅숏 자리 바로 뒤(주문 묶음이 열린 채)", new_sec(case("short")["S_idx"].as_u64().unwrap() as usize)),
        ("드문 체결 · 늦은 체결 뒤", new_sec(first_at(30_010.0))),
        ("촘촘한 체결 한가운데", new_sec(case("bar")["S_idx"].as_u64().unwrap() as usize - 500)),
        ("끝 가까이", new_sec(n - 4000)),
    ];
    let (mut bars, mut cells, mut secs) = (0, 0, 0);
    for (what, i) in points {
        let dir = tmp(&format!("rt{i}"));
        let a = core_live(&f.rows[..i], &dir);
        assert_eq!(a.write_snapshots(), 1);
        let file = restore::snap_path(&dir, sym);
        let bytes = std::fs::read(&file).unwrap();
        let now = f.rows[i].w * 1000;
        let s = restore::load(&file, sym, now / 1000).unwrap();
        assert_eq!(serde_json::to_vec(&s).unwrap(), bytes, "{what}: 읽어 다시 쓴 바이트가 다르다");
        let (b, _) = State::with_restore(std::slice::from_ref(sym), 16, 1 << 16, cfg(&dir, dir.join("없는_테이프.sqlite")), now);
        // 봉(창 144봉)이 비트까지 같다
        let ra = a.footprint_cells(sym).unwrap();
        let rb = b.footprint_cells(sym).unwrap();
        let ra = &ra[ra.len() - rb.len()..];
        assert_eq!(rb.len(), s.bars.len());
        for (x, y) in ra.iter().zip(&rb) {
            assert_eq!(x.0, y.0);
            assert_eq!(x.1.len(), y.1.len());
            for (p, q) in x.1.iter().zip(&y.1) {
                assert_eq!((p.0, p.1.map(f64::to_bits)), (q.0, q.1.map(f64::to_bits)), "{what}: 봉 {} 칸", x.0);
                cells += 1;
            }
            bars += 1;
        }
        // 재연결(새 연결 = 열린 주문 묶음을 버린다)로 같은 다음 체결을 둘 다에 -- 테이프가 없어 1초 칸은 스냅숏에서 온다
        a.reset_orders(sym);
        let mut rx = b.tx.subscribe();
        feed(&a, &f.rows[i..i + 1]);
        feed(&b, &f.rows[i..i + 1]);
        a.run_restore(sym).unwrap();
        b.run_restore(sym).unwrap();
        let sa = a.supply_cells(sym).unwrap();
        let sb = b.supply_cells(sym).unwrap();
        assert_eq!(sa.len(), sb.len(), "{what}: 1초 칸 수");
        for (x, y) in sa.iter().zip(&sb) {
            assert_eq!((x.0, x.1.map(f64::to_bits)), (y.0, y.1.map(f64::to_bits)), "{what}: 1초 칸");
            secs += 1;
        }
        let strip = |mut f: Footprint| {
            f.holes.clear(); // 구멍은 따로(a 는 처음부터 테이프 없이 산 판이라 제 구멍이 있다)
            f
        };
        let (seq0, fb0, sb0) = snap_of(&b);
        assert_eq!(strip(snap_of(&a).1), strip(fb0.clone()), "{what}: 되살린 직후 풋프린트 스냅숏(칸·dhl)");
        let mut view = View::default();
        view.put_fp(&fb0);
        view.put_sup(&sb0);
        for k in (i + 1..(i + 3000).min(n)).step_by(37) {
            feed(&a, &f.rows[k..(k + 37).min(n)]);
            feed(&b, &f.rows[k..(k + 37).min(n)]);
            b.flush();
            view.drain(&mut rx, seq0);
        }
        let (_, fa, ua) = snap_of(&a);
        let (_, fb, ub) = snap_of(&b);
        assert_eq!(strip(fa), strip(fb.clone()), "{what}: 그 뒤 3000건 -- 풋프린트");
        assert_eq!(ua.seconds, ub.seconds, "{what}: 그 뒤 3000건 -- 1초 칸");
        view.same_as(&format!("{what} 밀기"), &fb, &ub);
        println!("왕복 {what}(행 {i}): 봉 {} · 1초 {} · 그 뒤 봉 {}", rb.len(), sb.len(), fb.bars.len());
    }
    println!("왕복 합계: 봉 {bars} · 칸 {cells} · 1초 칸 {secs} (비트 일치)");
}

/// 쓰기 조건: 첫 체결 전·메우는 중엔 안 쓰고(파이썬 ready 와 같은 뜻), 합친 뒤에만 쓴다.
#[test]
fn snapshot_waits_for_merge() {
    let f = fx();
    let dir = tmp("wait");
    let (st, _) = State::with_restore(std::slice::from_ref(&f.sym), 16, 1 << 16, cfg(&dir, dir.join("없음.sqlite")), f.rows[0].w * 1000);
    assert_eq!(st.write_snapshots(), 0, "첫 체결 전");
    feed(&st, &f.rows[..50]);
    assert_eq!(st.write_snapshots(), 0, "메우는 중");
    st.run_restore(&f.sym).unwrap();
    assert_eq!(st.write_snapshots(), 1, "합친 뒤");
    st.reset_orders(&f.sym);
    feed(&st, &f.rows[50..60]);
    assert_eq!(st.write_snapshots(), 0, "재연결로 다시 메우는 중");
}

/// (c) 계속 돈 코어 대 중간에 죽었다 되살린 코어(같은 체결 열 · 재시작 + 780초 -- 1초 링이 전부 재시작 뒤 초가 된 때).
#[test]
fn restart_vs_continuous() {
    let f = fx();
    let sym = &f.sym;
    for name in ["short", "bar", "behind"] {
        let c = case(name);
        let r = c["R_idx"].as_u64().unwrap() as usize;
        let z = f.rows.iter().position(|x| x.t >= f.rows[r].t + 780_000).unwrap();
        let cont = core_live(&f.rows[..z], &tmp(&format!("cont_{name}")));
        let (rest, snap) = restart(c, z, None);
        let snap = snap.unwrap();
        let last_bar = snap.last_ms.div_euclid(1000).div_euclid(BS) * BS;
        let first_bar = f.rows[r].t.div_euclid(1000).div_euclid(BS) * BS;
        let holes = ranges(&c["holes"]);
        let mut allowed: BTreeSet<i64> = [last_bar, first_bar].into();
        for &(a, zz) in &holes {
            let mut b = a.div_euclid(BS) * BS;
            while b < zz {
                allowed.insert(b);
                b += BS;
            }
        }
        // 봉: 되살린 판의 창(144봉)
        let rc = cont.footprint_cells(sym).unwrap();
        let rr = rest.footprint_cells(sym).unwrap();
        let rc: BTreeMap<i64, BTreeMap<i64, [f64; 6]>> = rc.into_iter().map(|(b, c)| (b, c.into_iter().collect())).collect();
        let (mut same, mut diff) = (0, vec![]);
        for (b, cr) in &rr {
            let cc = &rc[b];
            let keys: BTreeSet<i64> = cc.keys().chain(cr.iter().map(|x| &x.0)).copied().collect();
            let crm: BTreeMap<i64, [f64; 6]> = cr.iter().copied().collect();
            let mut d = [0.0f64; 6];
            for k in keys {
                let (x, y) = (cc.get(&k).copied().unwrap_or([0.0; 6]), crm.get(&k).copied().unwrap_or([0.0; 6]));
                for j in 0..6 {
                    if !close(x[j], y[j]) {
                        d[j] += x[j] - y[j];
                    }
                }
            }
            if d == [0.0; 6] {
                same += 1;
            } else {
                let vol: f64 = cc.values().map(|c| c[0] + c[1]).sum();
                assert!(allowed.contains(b), "{name}: 경계 밖 봉 {b} 이 다르다 {d:?}");
                diff.push((b - first_bar, d.map(|x| (x * 1000.0).round() / 1000.0), (vol * 100.0).round() / 100.0));
            }
        }
        // 1초 칸: 재시작 + 780초면 링이 전부 재시작 뒤 초다 -- 비트까지 같다
        assert_eq!(cont.supply_cells(sym).unwrap(), rest.supply_cells(sym).unwrap(), "{name}: 1초 칸");
        // dhl
        let (_, fc, _) = snap_of(&cont);
        let (_, fr, _) = snap_of(&rest);
        let mut dd = vec![];
        for x in &fr.bars {
            let y = fc.bars.iter().find(|y| y.time == x.time).unwrap();
            if x.dhl != y.dhl {
                assert!(allowed.contains(&x.time), "{name}: 경계 밖 봉 {} dhl {:?} != {:?}", x.time, x.dhl, y.dhl);
                dd.push((x.time - first_bar, y.dhl, x.dhl));
            }
        }
        println!(
            "이어 붙이기 {name}: 창 {}봉 중 같음 {same} · 다름 {} [봉(첫 체결 봉 기준 초), 칸 합 차(계속 − 되살림: 매수·매도·고래매수·고래매도·리테일매수·리테일매도), 봉 총량] {diff:?} · dhl 다름 {dd:?} · 1초 칸 {} 같음",
            rr.len(),
            diff.len(),
            cont.supply_cells(sym).unwrap().len()
        );
    }
}
