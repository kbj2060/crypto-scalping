//! ETH 5분 호가 요약 실시간 워커(`scripts/live_ob5m_20261026.py`, 줄 번호는 그 파일) -- depthdiff 시 파일을 이어 읽어 북을 다시 세우고(`replay`)
//! 1초 줄(`sec_row`) → 5분 봉(`Bars`) → `data/live/ob5m/<코인>.sqlite`(bars · marks · live). /api/ob5m 이 읽는다. 호가 깊이 10열(`bookdepth`)도 이 북을 쓴다.
//! 숫자는 numpy 와 같은 순서로: `.sum()`·`np.mean` = 짝 합(`np_sum` -- rtmodel/tools/check_npsum.py 로 확정한 식) · `np.median` = 정렬 뒤 가운데(짝수면
//! 두 값 평균) · `np.log` = libm(서버 = AVX-512 없음 → numpy 가 libm 을 부른다 -- 개발기 정답지는 NPY_DISABLE_CPU_FEATURES=X86_V4 로 같은 쪽) ·
//! `argmax` = 첫 최대 · `round` = 짝수 쪽 반올림.
//! 🔴정의 확정 전 · 정보 표시용(예측 아님). 문턱은 아래 상수 한 곳(코인별 COINS). 주문·거래소 호출 없음.
use rusqlite::{params, Connection};
use serde_json::Value;
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::io::Read;
use std::path::{Path, PathBuf};

pub const BAR: i64 = 300_000;
pub const BIG_AGE_MS: i64 = 30_000; // 사건 벽은 30초 이상 서 있던 것만
pub const NEAR_BP: f64 = 5.0; // 현재가 5bp 안에서 체결 없이 사라짐 = 가짜벽 의심
pub const EAT_FILL: f64 = 0.8;
pub const PULL_FILL: f64 = 0.2;
pub const JUMP_WIN: i64 = 5_000;
pub const WARM_MS: i64 = 1_800_000; // 사슬 (재)시작 뒤 30분 = 깊은 칸이 덜 찼다 → 북 값 없음
pub const TAPE_GAP_S: i64 = 15;
pub const TAPE_PAD_S: i64 = 60;
pub const WALL_BACK_MS: i64 = 10_000; // 다른 칸이 10초 이하 잠깐 최대였다 돌아오면 같은 벽으로 이어 센다
pub const DROP_FRAC: f64 = 0.8;
pub const DROP_WIN: i64 = 1000;
pub const MIN_SHARE: f64 = 0.01;
pub const SETTLE_MS: i64 = 70_000; // 봉 끝 + 60초(테이프 빈 구간 ±60초) + 테이프 지연 10초
pub const LIVE_S: i64 = 5;
pub const BOOK_COLS: [&str; 18] = ["imb10", "imb50", "imb50_last", "dep10", "dep50", "dep50_min",
    "wall_frac_b", "wall_age_b", "sh_max_b", "wall_px_b", "wall_dist_b", "wall_usd_b",
    "wall_frac_a", "wall_age_a", "sh_max_a", "wall_px_a", "wall_dist_a", "wall_usd_a"];
pub const EV_COLS: [&str; 18] = ["eat_n_b", "eat_usd_b", "eat_jump_b", "eat_n_a", "eat_usd_a", "eat_jump_a",
    "fake_n_b", "fake_usd_b", "fake_jump_b", "fake_n_a", "fake_usd_a", "fake_jump_a",
    "pull_n_b", "pull_usd_b", "pull_jump_b", "pull_n_a", "pull_usd_a", "pull_jump_a"];

/// 코인별(:43-52): (틱 1/$, 벽 몫, 사건 추적 문턱 KS, idx 상한)
pub fn coin(sym: &str) -> Option<(f64, f64, [f64; 3], usize)> {
    Some(match sym {
        "ETHUSDT" | "SNDKUSDT" | "SOXLUSDT" | "SKHYNIXUSDT" | "MUUSDT" => (100.0, 0.05, [20.0, 50.0, 100.0], 1_000_000),
        "BTCUSDT" => (10.0, 0.05, [20.0, 50.0, 100.0], 3_000_000),
        "XRPUSDT" => (10_000.0, 0.097, [8.0, 20.0, 40.0], 1_000_000),
        "SOLUSDT" => (100.0, 0.098, [5.0, 12.0, 25.0], 1_000_000),
        _ => return None,
    })
}

/// np.sum(연속 f64 1차원): numpy 는 버퍼 8192칸 묶음마다 짝 합을 내고 그 묶음 합을 차례로 더한다(2026-11-21 서버 판 numpy 2.2.5 비트 대조 -- tests/np_sum_parity.rs).
/// 8192칸 이하는 짝 합 하나(전과 같다).
pub fn np_sum(a: &[f64]) -> f64 {
    if a.len() <= 8192 {
        return pairwise(a);
    }
    let mut s = 0.0;
    for c in a.chunks(8192) {
        s += pairwise(c);
    }
    s
}

/// numpy 짝 합(pairwise_sum): 8칸 미만 차례 합 · 128칸 이하 8갈래 펼침 · 넘으면 8의 배수로 반 나눔.
fn pairwise(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut s = 0.0;
        for v in a {
            s += v;
        }
        return s;
    }
    if n <= 128 {
        let mut r = [0.0; 8];
        r.copy_from_slice(&a[..8]);
        let m = n - n % 8;
        let mut i = 8;
        while i < m {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for v in &a[m..] {
            res += v;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise(&a[..n2]) + pairwise(&a[n2..])
}

pub fn np_mean(a: &[f64]) -> f64 {
    np_sum(a) / a.len() as f64
}

/// np.median: 정렬 뒤 가운데 · 짝수면 np.mean(두 값) = (0 + a + b) / 2.
pub fn np_median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

/// 파이썬 round(x) -> int(짝수 쪽).
fn pyround(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// numpy 슬라이스 [lo:hi] (끝은 길이로 자른다 · lo 는 부르는 쪽이 0 이상으로).
fn sl(a: &[f64], lo: i64, hi: i64) -> &[f64] {
    let n = a.len() as i64;
    let (lo, hi) = (lo.clamp(0, n), hi.clamp(0, n));
    if hi <= lo { &a[0..0] } else { &a[lo as usize..hi as usize] }
}

/// 사건(on_event dict): t · side(0 매수 · 1 매도) · px · qref · q · share · dist_bp · age20 (+ fill 은 Bars 가 채운다)
#[derive(Clone, Debug)]
pub struct Event {
    pub t: i64,
    pub side: u8,
    pub px: f64,
    pub qref: f64,
    pub q: f64,
    pub share: f64,
    pub dist_bp: f64,
    pub age20: i64,
    pub fill: f64,
}

type HistE = (i64, f64, [Option<i64>; 3]);

fn best(arr: &[f64], cur: i64, low_side: bool) -> i64 {
    let w = ((cur as f64 * 0.02) as i64).max(100);
    let lo = (cur - w).max(0);
    let s = sl(arr, lo, cur + w + 1);
    let hit = if low_side { s.iter().rposition(|x| *x != 0.0) } else { s.iter().position(|x| *x != 0.0) };
    if let Some(i) = hit {
        return lo + i as i64;
    }
    let g = if low_side { arr.iter().rposition(|x| *x != 0.0) } else { arr.iter().position(|x| *x != 0.0) };
    g.map_or(-1, |i| i as i64)
}

fn int(v: &Value) -> Option<i64> {
    crate::py_int(v).ok()
}

/// `replay`(:89-178): 첫 줄 = 스냅숏. 끝나면 상태 글자(no_snapshot · truncated · gap · sync_fail · empty_book · eof).
/// `on_sec(t, bid, ask, bb, ba)` = 매 초 첫 diff 적용 전 북 · `on_event(e)` = 벽이 1초 안에 80% 넘게 빠진 사건.
pub fn replay(lines: &mut dyn Iterator<Item = Vec<u8>>, tick: f64, ks: [f64; 3], idx_max: usize,
              on_sec: &mut dyn FnMut(i64, &[f64], &[f64], i64, i64), on_event: &mut dyn FnMut(Event)) -> &'static str {
    let mut bid = vec![0.0f64; idx_max];
    let mut ask = vec![0.0f64; idx_max];
    let Some(first) = lines.next() else { return "no_snapshot" };
    let Ok(first) = serde_json::from_slice::<Value>(&first) else { return "no_snapshot" };
    let snap = match first.get("_snapshot") {
        Some(s) if crate::py_truthy(s) => s.clone(),
        _ => return "no_snapshot",
    };
    for (arr, key) in [(&mut bid, "bids"), (&mut ask, "asks")] {
        for pq in snap[key].as_array().into_iter().flatten() {
            let i = pyround(crate::py_float(&pq[0]).unwrap() * tick);
            if (i as usize) < idx_max && i >= 0 {
                arr[i as usize] = crate::py_float(&pq[1]).unwrap();
            }
        }
    }
    let mut last_u = int(&snap["lastUpdateId"]).unwrap();
    let mut synced = false;
    // 🔧의도한 차이(SHADOW.md §11 #16): 스냅숏 한쪽이 비면 원본은 IndexError 로 죽는다(np.flatnonzero(bid)[-1] · 얇은 주식 선물에서 실제로
    // 2026-11-21 SNDKUSDT) -- Rust 는 그 사슬을 «empty_book» 으로 끝낸다(restart 규칙대로 다음 시 파일 스냅숏에서 다시).
    let (Some(b0), Some(a0)) = (bid.iter().rposition(|x| *x != 0.0), ask.iter().position(|x| *x != 0.0)) else { return "empty_book" };
    let (mut bb, mut ba) = (b0 as i64, a0 as i64);
    let (mut med, mut dep) = ([f64::NAN; 2], [f64::NAN; 2]);
    let mut tracked: HashMap<(u8, i64), Vec<HistE>> = HashMap::new();
    let mut cur_sec: Option<i64> = None;
    for line in lines {
        let Ok(m) = serde_json::from_slice::<Value>(&line) else { return "truncated" };
        if m.get("_gap").is_some_and(crate::py_truthy) {
            return "gap";
        }
        if m.get("e").and_then(Value::as_str) != Some("depthUpdate") {
            continue;
        }
        let (u, uu) = (int(&m["u"]).unwrap(), int(&m["U"]).unwrap());
        if !synced {
            if u < last_u {
                continue;
            }
            if !(uu <= last_u && last_u <= u) {
                return "sync_fail";
            }
            synced = true;
        } else if int(&m["pu"]) != Some(last_u) {
            return "gap";
        }
        last_u = u;
        let t = int(&m["T"]).unwrap();
        let sec = t.div_euclid(1000);
        if cur_sec.is_none_or(|c| sec > c) {
            cur_sec = Some(sec);
            let mid_i = (bb + ba) as f64 / 2.0;
            let w = (mid_i * 50e-4) as i64;
            for side in 0..2u8 {
                let (base, s) = if side == 0 { ((bb - w).max(0), sl(&bid, (bb - w).max(0), bb + 1)) } else { (ba, sl(&ask, ba, ba + w + 1)) };
                let mut nz: Vec<f64> = s.iter().copied().filter(|x| *x > 0.0).collect();
                let md = if nz.is_empty() { f64::NAN } else { np_median(&mut nz) };
                med[side as usize] = md;
                dep[side as usize] = np_sum(s);
                if side == 1 {
                    on_sec(t, &bid, &ask, bb, ba); // 매 초 첫 diff 적용 전 북
                }
                if md.is_finite() {
                    for (j, &q) in s.iter().enumerate() {
                        if q >= ks[0] * md {
                            tracked.entry((side, base + j as i64)).or_insert_with(|| vec![(t, q, ks.map(|k| (q >= k * md).then_some(t)))]);
                        }
                    }
                }
            }
            let far = (mid_i * 60e-4) as i64;
            tracked.retain(|k, h| {
                let last = h.last().unwrap();
                !((k.1 as f64 - mid_i).abs() > far as f64 || (last.0 < t - 1500 && last.2[0].is_none()))
            });
        }
        let (pre_bb, pre_ba) = (bb, ba);
        let mut touched: Vec<(u8, i64, f64)> = vec![];
        for (side, key) in [(0u8, "b"), (1u8, "a")] {
            let arr = if side == 0 { &mut bid } else { &mut ask };
            for pq in m[key].as_array().into_iter().flatten() {
                let i = pyround(crate::py_float(&pq[0]).unwrap() * tick);
                if (i as usize) < idx_max && i >= 0 {
                    let q = crate::py_float(&pq[1]).unwrap();
                    arr[i as usize] = q;
                    touched.push((side, i, q));
                }
            }
        }
        bb = best(&bid, bb, true);
        ba = best(&ask, ba, false);
        if bb < 0 || ba < 0 {
            return "empty_book";
        }
        let pre_mid = (pre_bb + pre_ba) as f64 / 2.0;
        for (side, i, q) in touched {
            let md = med[side as usize];
            if !md.is_finite() {
                continue;
            }
            let Some(h) = tracked.get_mut(&(side, i)) else {
                if q >= ks[0] * md && (i as f64 - pre_mid).abs() <= pre_mid * 50e-4 {
                    tracked.insert((side, i), vec![(t, q, ks.map(|k| (q >= k * md).then_some(t)))]);
                }
                continue;
            };
            let prev = h.last().unwrap().2;
            let mut s_new = [None; 3];
            for (j, k) in ks.iter().enumerate() {
                s_new[j] = if q >= k * md { Some(prev[j].unwrap_or(t)) } else { None };
            }
            let rf = h.iter().rev().find(|e| e.0 <= t - DROP_WIN).copied();
            if let Some(r) = rf {
                if let Some(r2) = r.2[0] {
                    if q <= (1.0 - DROP_FRAC) * r.1 && (r.0 - r2) >= 0 && (t - DROP_WIN) - r2 >= 1000 {
                        if r.1 >= MIN_SHARE * dep[side as usize] {
                            let dist = if side == 0 { pre_mid - i as f64 } else { i as f64 - pre_mid } / pre_mid * 1e4;
                            on_event(Event { t, side, px: i as f64 / tick, qref: r.1, q, share: r.1 / dep[side as usize], dist_bp: dist,
                                             age20: (t - DROP_WIN) - r2, fill: f64::NAN });
                        }
                        tracked.remove(&(side, i));
                        continue;
                    }
                }
            }
            h.push((t, q, s_new));
            while h.len() > 2 && h[1].0 <= t - DROP_WIN {
                h.remove(0);
            }
        }
    }
    "eof"
}

/// `sec_row`(:181-188) -- 도장 t−1: (t−1, bb, ba, ±10bp 매수·매도 합, ±50bp 합, 매수 최대 칸·수량, 매도 최대 칸·수량)
pub type SecRow = (i64, i64, i64, f64, f64, f64, f64, i64, f64, i64, f64);

fn argmax(a: &[f64]) -> usize {
    let mut k = 0;
    for (i, v) in a.iter().enumerate() {
        if *v > a[k] {
            k = i;
        }
    }
    k
}

pub fn sec_row(t: i64, bid: &[f64], ask: &[f64], bb: i64, ba: i64) -> SecRow {
    let m = (bb + ba) as f64 / 2.0;
    let (w50, w10) = ((m * 50e-4) as i64, (m * 10e-4) as i64);
    let (b50, a50) = (sl(bid, bb - w50, bb + 1), sl(ask, ba, ba + w50 + 1));
    let (jb, ja) = (argmax(b50), argmax(a50));
    (t - 1, bb, ba, np_sum(sl(bid, bb - w10, bb + 1)), np_sum(sl(ask, ba, ba + w10 + 1)), np_sum(b50), np_sum(a50),
     bb - w50 + jb as i64, b50[jb], ba + ja as i64, a50[ja])
}

/// `tape_bad`(:191-202): x 초가 테이프 빈 구간(연속 테이프 초 간격 > 15초) ±60초 안이거나 테이프 범위 밖이면 참.
pub fn tape_bad(sec: &[i64], x: &[i64]) -> Vec<bool> {
    let s: Vec<i64> = sec.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
    if s.is_empty() {
        return vec![true; x.len()];
    }
    let g: Vec<usize> = (0..s.len() - 1).filter(|&k| s[k + 1] - s[k] > TAPE_GAP_S).collect();
    let lo: Vec<i64> = g.iter().map(|&k| s[k] - TAPE_PAD_S).collect();
    let hi: Vec<i64> = g.iter().map(|&k| s[k + 1] + TAPE_PAD_S).collect();
    x.iter()
        .map(|&v| {
            let out = v < s[0] || v > s[s.len() - 1];
            if g.is_empty() {
                return out;
            }
            let j = lo.partition_point(|&l| l <= v) as i64 - 1;
            out || (j >= 0 && v <= hi[j.max(0) as usize])
        })
        .collect()
}

/// 봉·live 값(파이썬 dict 순서 그대로 -- json.dumps 와 같은 글자로 쓰려고).
#[derive(Clone, Debug, PartialEq)]
pub enum V {
    I(i64),
    F(f64),
    N,
}

pub type Rec = Vec<(String, V)>;

fn set(r: &mut Rec, k: &str, v: V) {
    match r.iter_mut().find(|x| x.0 == k) {
        Some(x) => x.1 = v,
        None => r.push((k.to_string(), v)),
    }
}

fn get(r: &Rec, k: &str) -> V {
    r.iter().find(|x| x.0 == k).map_or(V::N, |x| x.1.clone())
}

/// 파이썬 json.dumps(dict) 기본 모양(", " · ": ") -- 실수 = repr · NaN = NaN · None = null.
pub fn dumps(r: &Rec) -> String {
    let f = |v: &V| match v {
        V::I(i) => i.to_string(),
        V::F(x) => crate::py_float_repr(*x),
        V::N => "null".into(),
    };
    format!("{{{}}}", r.iter().map(|(k, v)| format!("\"{k}\": {}", f(v))).collect::<Vec<_>>().join(", "))
}

#[derive(Clone)]
struct SecR {
    mid: f64,
    warm: bool,
    imb10: f64,
    imb50: f64,
    dep10: f64,
    dep50: f64,
    s: [(f64, bool, f64, i64, f64, f64); 2], // (sh, on, age, i, dist_bp, usd)
}

pub struct Mark {
    pub t: i64,
    pub bar_t: i64,
    pub side: &'static str,
    pub kind: &'static str,
    pub px: f64,
    pub usd: f64,
    pub dist_bp: f64,
}

pub fn marks_json(m: &[Mark]) -> String {
    format!("[{}]", m.iter().map(|m| format!(r#"{{"t": {}, "bar_t": {}, "side": "{}", "kind": "{}", "px": {}, "usd": {}, "dist_bp": {}}}"#,
        m.t, m.bar_t, m.side, m.kind, crate::py_float_repr(m.px), crate::py_float_repr(m.usd), crate::py_float_repr(m.dist_bp))).collect::<Vec<_>>().join(", "))
}

/// 테이프 읽기: (lo_s, hi_s) → (행[(ts_sec, price_bin, buy, sell)], 공백[(from_s, to_s)])
pub type Tape<'a> = Box<dyn FnMut(i64, i64) -> (Vec<(i64, i64, f64, f64)>, Vec<(i64, i64)>) + 'a>;

/// `Bars`(:223-389): 1초 줄·사건 → 5분 봉(벽 나이는 계속 든 상태로).
pub struct Bars<'a> {
    tick: f64,
    wall_share: f64,
    bucket: f64,
    tape: Tape<'a>,
    pub out: Box<dyn FnMut(&Rec, Option<&[Mark]>) + 'a>,
    pub live: Option<Box<dyn FnMut(Option<&Rec>, &[Mark]) + 'a>>,
    live_lag: i64,
    age: [(HashMap<i64, i64>, HashMap<i64, i64>); 2],
    rows: Vec<(i64, Vec<SecR>)>, // 봉 → 줄(삽입 순서 = 파이썬 dict)
    evs: Vec<(i64, Vec<Event>)>,
    book: Vec<(i64, Rec)>,
    mids: VecDeque<(i64, f64)>,
    last_t: Option<i64>,
    c0: Option<i64>,
}

fn entry<T: Default>(v: &mut Vec<(i64, T)>, k: i64) -> &mut T {
    if let Some(i) = v.iter().position(|x| x.0 == k) {
        return &mut v[i].1;
    }
    v.push((k, T::default()));
    &mut v.last_mut().unwrap().1
}

fn take<T>(v: &mut Vec<(i64, T)>, k: i64) -> Option<T> {
    v.iter().position(|x| x.0 == k).map(|i| v.remove(i).1)
}

fn is_big(e: &Event, wall_share: f64) -> bool {
    e.share >= wall_share && e.age20 >= BIG_AGE_MS && e.dist_bp <= 50.0
}

fn kind_of(e: &Event, wall_share: f64) -> Option<&'static str> {
    if !is_big(e, wall_share) {
        return None;
    }
    let f = e.fill;
    Some(if f.is_nan() { "unk" } else if f >= EAT_FILL { "eat" } else if f < PULL_FILL && e.dist_bp <= NEAR_BP { "fake" } else if f < PULL_FILL { "pull" } else { "mixed" })
}

fn mark_of(e: &Event, b: i64, kind: &'static str) -> Mark {
    Mark { t: e.t, bar_t: b * BAR, side: if e.side == 0 { "bid" } else { "ask" }, kind, px: e.px, usd: (e.qref - e.q) * e.px, dist_bp: e.dist_bp }
}

impl<'a> Bars<'a> {
    pub fn new(sym: &str, tape: Tape<'a>, out: Box<dyn FnMut(&Rec, Option<&[Mark]>) + 'a>, live: Option<Box<dyn FnMut(Option<&Rec>, &[Mark]) + 'a>>) -> Self {
        let (tick, wall_share, _, _) = coin(sym).expect("코인");
        Self { tick, wall_share, bucket: crate::tape::bucket(&sym.to_lowercase()), tape, out, live, live_lag: 60_000,
               age: Default::default(), rows: vec![], evs: vec![], book: vec![], mids: VecDeque::new(), last_t: None, c0: None }
    }

    pub fn chain(&mut self) {
        self.c0 = None;
    }

    /// `sec`(:244-286). `now_ms` = 벽시계(진행 중 봉은 거래소 T 가 1분 안일 때만 -- 시작 재생 중엔 안 쓴다).
    pub fn sec(&mut self, row: SecRow, now_ms: f64) {
        let t = row.0;
        if self.last_t == Some(t) {
            return; // 같은 초 도장 중복 = 첫 것만
        }
        self.last_t = Some(t);
        if self.c0.is_none() {
            self.c0 = Some(t + 1);
        }
        let (_, bb, ba, b10, a10, b50, a50, wb_i, wb_q, wa_i, wa_q) = row;
        let mid = (bb + ba) as f64 / 2.0 / self.tick;
        let mut r = SecR { mid, warm: (t - self.c0.unwrap()) < WARM_MS, imb10: (b10 - a10) / (b10 + a10), imb50: (b50 - a50) / (b50 + a50),
                           dep10: (b10 + a10) * mid, dep50: (b50 + a50) * mid, s: [(0.0, false, 0.0, 0, 0.0, 0.0); 2] };
        for (s, (i, q, d)) in [(wb_i, wb_q, b50), (wa_i, wa_q, a50)].into_iter().enumerate() {
            let sh = q / d;
            let on = sh >= self.wall_share;
            let mut age = 0.0;
            if on {
                let (start, seen) = &mut self.age[s];
                if t - seen.get(&i).copied().unwrap_or(-10i64.pow(18)) > WALL_BACK_MS {
                    start.insert(i, t);
                }
                seen.insert(i, t);
                age = (t - start[&i]) as f64;
            }
            r.s[s] = (sh, on, age, i, (i as f64 / self.tick - mid).abs() / mid * 1e4, q * mid);
        }
        entry(&mut self.rows, t.div_euclid(BAR)).push(r);
        self.mids.push_back((t, mid));
        while self.mids.front().is_some_and(|m| m.0 < t - 1_200_000) {
            self.mids.pop_front();
        }
        for b in self.rows.iter().map(|x| x.0).filter(|b| (b + 1) * BAR <= t).collect::<Vec<_>>() {
            self.close_book(b);
        }
        for b in self.book.iter().map(|x| x.0).filter(|b| (b + 1) * BAR + SETTLE_MS <= t).collect::<Vec<_>>() {
            self.close(b);
        }
        if self.live.is_some() && t.div_euclid(1000) % LIVE_S == 0 && t as f64 >= now_ms - self.live_lag as f64 {
            self.emit_live(t);
        }
        if t.div_euclid(1000) % 600 == 0 {
            for (start, seen) in &mut self.age {
                for i in seen.iter().filter(|(_, v)| t - **v > WALL_BACK_MS).map(|x| *x.0).collect::<Vec<_>>() {
                    start.remove(&i);
                    seen.remove(&i);
                }
            }
            let dead: Vec<i64> = self.evs.iter().map(|x| x.0)
                .filter(|k| !self.rows.iter().any(|r| r.0 == *k) && !self.book.iter().any(|r| r.0 == *k) && (k + 1) * BAR + SETTLE_MS <= t).collect();
            self.evs.retain(|x| !dead.contains(&x.0));
        }
    }

    pub fn event(&mut self, e: Event) {
        entry(&mut self.evs, e.t.div_euclid(BAR)).push(e);
    }

    fn mid_at(&self, x: i64) -> f64 {
        let i = self.mids.partition_point(|m| m.0 <= x) as i64 - 1;
        if i >= 0 && x - self.mids[i as usize].0 <= 3000 { self.mids[i as usize].1 } else { f64::NAN }
    }

    fn close_book(&mut self, b: i64) {
        let rows = take(&mut self.rows, b).unwrap();
        let bk = self.book_of(b, &rows);
        (self.out)(&bk, None);
        self.book.push((b, bk));
    }

    fn emit_live(&mut self, t: i64) {
        let b = t.div_euclid(BAR);
        let bk = self.rows.iter().find(|r| r.0 == b).filter(|r| !r.1.is_empty()).map(|r| r.1.clone()).map(|rows| self.book_of(b, &rows));
        let mut keys: Vec<i64> = self.book.iter().map(|x| x.0).collect();
        keys.push(b);
        keys.sort_unstable();
        keys.dedup();
        let mut first: Vec<Event> = vec![];
        for k in keys {
            for e in self.evs.iter().filter(|x| x.0 == k).flat_map(|x| x.1.iter()) {
                if !first.iter().any(|f| f.t == e.t && f.side == e.side && f.px == e.px) {
                    first.push(e.clone());
                }
            }
        }
        let warm = |k: i64| -> bool {
            if let (Some(bk), true) = (&bk, k == b) {
                return get(bk, "warm") == V::I(1);
            }
            self.book.iter().find(|x| x.0 == k).map_or(true, |x| get(&x.1, "warm") == V::I(1))
        };
        let mut e: Vec<Event> = first.into_iter().filter(|e| is_big(e, self.wall_share) && !warm(e.t.div_euclid(BAR))).collect();
        let mut marks = vec![];
        if !e.is_empty() {
            let lo = e.iter().map(|x| x.t).min().unwrap().div_euclid(1000) - 180;
            let last = self.fills(&mut e, lo, t.div_euclid(1000) + 1, None);
            let ws = self.wall_share;
            let dead: Vec<i64> = e.iter().filter(|x| kind_of(x, ws) == Some("unk") && last.is_some_and(|l| x.t.div_euclid(1000) <= l))
                .map(|x| x.t.div_euclid(BAR)).collect();
            for x in &e {
                if let Some(k @ ("eat" | "fake")) = kind_of(x, ws) {
                    if !dead.contains(&x.t.div_euclid(BAR)) {
                        marks.push(mark_of(x, x.t.div_euclid(BAR), k));
                    }
                }
            }
        }
        if let Some(live) = &mut self.live {
            live(bk.as_ref(), &marks);
        }
    }

    /// `_fills`(:316-332). 반환 = 마지막 테이프 초.
    fn fills(&mut self, e: &mut [Event], lo_s: i64, hi_s: i64, end_s: Option<i64>) -> Option<i64> {
        let (rows, gaps) = (self.tape)(lo_s, hi_s);
        let mut sec = vec![lo_s];
        sec.extend(rows.iter().map(|r| r.0));
        sec.extend(end_s);
        let bad = tape_bad(&sec, &e.iter().map(|x| x.t.div_euclid(1000)).collect::<Vec<_>>());
        let mut cell: HashMap<(i64, i64, u8), f64> = HashMap::new();
        for &(ts, pb, bq, sq) in &rows {
            *cell.entry((ts, pb, 0)).or_insert(0.0) += sq;
            *cell.entry((ts, pb, 1)).or_insert(0.0) += bq; // 매수벽(0)을 먹는 건 테이커 매도
        }
        for (x, bd) in e.iter_mut().zip(bad) {
            let ts = x.t.div_euclid(1000);
            if bd || gaps.iter().any(|&(g0, g1)| g0 - TAPE_PAD_S <= ts && ts <= g1 + TAPE_PAD_S) {
                x.fill = f64::NAN;
                continue;
            }
            let pb = pyround(x.px / self.bucket);
            let mut tot = 0.0;
            for s in (x.t - 1000).div_euclid(1000)..=ts {
                tot += cell.get(&(s, pb, x.side)).copied().unwrap_or(0.0);
            }
            x.fill = (tot / (x.qref - x.q).max(1e-12)).min(1.0);
        }
        rows.iter().map(|r| r.0).max()
    }

    /// `_book`(:334-356)
    fn book_of(&self, b: i64, r: &[SecR]) -> Rec {
        let mids: Vec<f64> = r.iter().map(|x| x.mid).collect();
        let mx = |v: &[f64]| v.iter().copied().fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
        let mn = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, |a, b| if b < a { b } else { a });
        let col = |f: fn(&SecR) -> f64| r.iter().map(f).collect::<Vec<f64>>();
        let mut bk: Rec = vec![];
        set(&mut bk, "t", V::I(b * BAR));
        set(&mut bk, "n_sec", V::I(r.len() as i64));
        set(&mut bk, "warm", V::I(r.iter().any(|x| x.warm) as i64));
        set(&mut bk, "o", V::F(mids[0]));
        set(&mut bk, "h", V::F(mx(&mids)));
        set(&mut bk, "l", V::F(mn(&mids)));
        set(&mut bk, "c", V::F(*mids.last().unwrap()));
        set(&mut bk, "imb10", V::F(np_mean(&col(|x| x.imb10))));
        set(&mut bk, "imb50", V::F(np_mean(&col(|x| x.imb50))));
        set(&mut bk, "imb50_last", V::F(r.last().unwrap().imb50));
        set(&mut bk, "dep10", V::F(np_mean(&col(|x| x.dep10))));
        set(&mut bk, "dep50", V::F(np_mean(&col(|x| x.dep50))));
        set(&mut bk, "dep50_min", V::F(mn(&col(|x| x.dep50))));
        for (s, n) in [(0usize, "b"), (1usize, "a")] {
            let ages: Vec<f64> = r.iter().map(|x| x.s[s].2).collect();
            let on: Vec<f64> = r.iter().map(|x| x.s[s].1 as i64 as f64).collect();
            set(&mut bk, &format!("wall_frac_{n}"), V::F(np_mean(&on)));
            set(&mut bk, &format!("wall_age_{n}"), V::F(mx(&ages) / 1000.0));
            set(&mut bk, &format!("sh_max_{n}"), V::F(mx(&r.iter().map(|x| x.s[s].0).collect::<Vec<_>>())));
            let pick: Vec<f64> = r.iter().zip(&ages).map(|(x, a)| if x.s[s].1 { *a } else { -1.0 }).collect();
            let k = argmax(&pick); // 봉 안 가장 오래 선 벽(벽 아닌 초는 안 뽑힘 · 같으면 앞)
            let w = r[k].s[s];
            set(&mut bk, &format!("wall_px_{n}"), if w.1 { V::F(w.3 as f64 / self.tick) } else { V::N });
            set(&mut bk, &format!("wall_dist_{n}"), if w.1 { V::F(w.4) } else { V::N });
            set(&mut bk, &format!("wall_usd_{n}"), if w.1 { V::F(w.5) } else { V::N });
        }
        if get(&bk, "warm") == V::I(1) {
            for c in BOOK_COLS {
                set(&mut bk, c, V::N);
            }
        }
        let (o, h, l, c) = (mids[0], mx(&mids), mn(&mids), *mids.last().unwrap());
        set(&mut bk, "ret", V::F((c / o).ln() * 1e4));
        set(&mut bk, "rng", V::F((h / l).ln() * 1e4));
        bk
    }

    /// `close`(:358-389)
    fn close(&mut self, b: i64) {
        if self.rows.iter().any(|x| x.0 == b) {
            self.close_book(b);
        }
        let mut bk = take(&mut self.book, b).unwrap();
        let mut e: Vec<Event> = vec![];
        for x in take(&mut self.evs, b).unwrap_or_default() {
            if !e.iter().any(|f| f.t == x.t && f.side == x.side && f.px == x.px) {
                e.push(x); // 중복은 첫 것(연구 drop_duplicates)
            }
        }
        let (lo_s, hi_s) = ((b * BAR).div_euclid(1000), ((b + 1) * BAR).div_euclid(1000));
        if !e.is_empty() {
            self.fills(&mut e, lo_s - 180, hi_s + 80, Some(hi_s + 75));
        }
        let mut n_usd: Rec = EV_COLS.iter().filter(|k| !k.contains("_jump_")).map(|k| (k.to_string(), V::F(0.0))).collect();
        let mut jumps: Vec<(String, Vec<f64>)> = vec![];
        let (mut marks, mut known) = (vec![], get(&bk, "warm") != V::I(1));
        for x in &e {
            let Some(kind) = kind_of(x, self.wall_share) else { continue };
            known &= kind != "unk";
            if kind == "unk" || kind == "mixed" {
                continue;
            }
            let n = if x.side == 0 { "b" } else { "a" };
            let add = |r: &mut Rec, k: String, v: f64| {
                if let Some(V::F(a)) = r.iter().find(|z| z.0 == k).map(|z| z.1.clone()) {
                    set(r, &k, V::F(a + v));
                }
            };
            add(&mut n_usd, format!("{kind}_n_{n}"), 1.0);
            add(&mut n_usd, format!("{kind}_usd_{n}"), (x.qref - x.q) * x.px);
            let post = self.mid_at((x.t + JUMP_WIN).min((b + 1) * BAR - 1));
            let j = (post / self.mid_at(x.t - JUMP_WIN)).ln() * 1e4 * if x.side == 1 { 1.0 } else { -1.0 };
            let key = format!("{kind}_jump_{n}");
            match jumps.iter_mut().find(|z| z.0 == key) {
                Some(z) => z.1.push(j),
                None => jumps.push((key, vec![j])),
            }
            if kind == "eat" || kind == "fake" {
                marks.push(mark_of(x, b, kind));
            }
        }
        if known {
            for (k, v) in n_usd {
                set(&mut bk, &k, v);
            }
            for k in EV_COLS.iter().filter(|k| k.contains("_jump_")) {
                let j: Vec<f64> = jumps.iter().find(|z| z.0 == *k).map(|z| z.1.iter().copied().filter(|x| !x.is_nan()).collect()).unwrap_or_default();
                set(&mut bk, k, if j.is_empty() { V::N } else { V::F(j.iter().copied().fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a })) });
            }
        } else {
            for k in EV_COLS {
                set(&mut bk, k, V::N);
            }
        }
        (self.out)(&bk, Some(if known { &marks } else { &[] }));
    }
}

// ── 저장(:392-431) ──
pub fn db_open(path: &Path) -> rusqlite::Result<Connection> {
    if let Some(d) = path.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let con = Connection::open(path)?;
    con.busy_timeout(std::time::Duration::from_secs(30))?;
    con.query_row("PRAGMA journal_mode=WAL", [], |_| Ok(()))?;
    con.execute_batch("PRAGMA synchronous = NORMAL")?;
    let cols: Vec<String> = ["n_sec", "warm", "o", "h", "l", "c"].iter().map(|s| s.to_string())
        .chain(BOOK_COLS.iter().map(|s| s.to_string())).chain(["ret".to_string(), "rng".to_string()]).chain(EV_COLS.iter().map(|s| s.to_string()))
        .map(|c| format!("{c} REAL")).collect();
    con.execute(&format!("CREATE TABLE IF NOT EXISTS bars(t INTEGER PRIMARY KEY, {})", cols.join(", ")), [])?;
    con.execute("CREATE TABLE IF NOT EXISTS live(k INTEGER PRIMARY KEY, upd INTEGER, bar TEXT, marks TEXT)", [])?;
    con.execute("CREATE TABLE IF NOT EXISTS marks(t INTEGER, bar_t INTEGER, side TEXT, kind TEXT, px REAL, usd REAL, dist_bp REAL, PRIMARY KEY (t, side, px))", [])?;
    Ok(con)
}

fn sqlv(v: &V) -> rusqlite::types::Value {
    match v {
        V::I(i) => rusqlite::types::Value::Integer(*i),
        V::F(f) => rusqlite::types::Value::Real(*f),
        V::N => rusqlite::types::Value::Null,
    }
}

/// `db_write`(:405-424): 이미 있는 온전한(warm 아님) 봉은 덮지 않는다 · marks None = 봉 끝 북 값만(사건 열은 SETTLE_MS 뒤 같은 프로세스가).
pub fn db_write(con: &Connection, bk: &Rec, marks: Option<&[Mark]>, mine: &mut BTreeSet<i64>) -> rusqlite::Result<bool> {
    let t = match get(bk, "t") {
        V::I(t) => t,
        _ => unreachable!(),
    };
    let old: Option<(Option<f64>, Option<f64>)> = con.query_row("SELECT warm, eat_n_a FROM bars WHERE t = ?", [t], |r| Ok((r.get(0)?, r.get(1)?))).ok();
    if let Some((w, eat)) = old {
        if w == Some(0.0) && !mine.contains(&t) {
            if marks.is_none() || eat.is_some() || get(bk, "warm") == V::I(1) || get(bk, "eat_n_a") == V::N {
                return Ok(false);
            }
            let sets: Vec<String> = EV_COLS.iter().map(|k| format!("{k} = ?")).collect();
            let mut vals: Vec<rusqlite::types::Value> = EV_COLS.iter().map(|k| sqlv(&get(bk, k))).collect();
            vals.push(rusqlite::types::Value::Integer(t));
            con.execute(&format!("UPDATE bars SET {} WHERE t = ?", sets.join(", ")), rusqlite::params_from_iter(vals))?;
        } else {
            insert(con, bk)?;
        }
    } else {
        insert(con, bk)?;
    }
    if let Some(m) = marks {
        con.execute("DELETE FROM marks WHERE bar_t = ?", [t])?;
        for x in m {
            con.execute("INSERT OR REPLACE INTO marks VALUES (?, ?, ?, ?, ?, ?, ?)", params![x.t, x.bar_t, x.side, x.kind, x.px, x.usd, x.dist_bp])?;
        }
    }
    if marks.is_none() { mine.insert(t) } else { mine.remove(&t) };
    Ok(true)
}

fn insert(con: &Connection, bk: &Rec) -> rusqlite::Result<()> {
    let cols: Vec<&str> = bk.iter().map(|x| x.0.as_str()).collect();
    con.execute(&format!("INSERT OR REPLACE INTO bars({}) VALUES ({})", cols.join(", "), vec!["?"; cols.len()].join(", ")),
                rusqlite::params_from_iter(bk.iter().map(|x| sqlv(&x.1))))?;
    Ok(())
}

/// `db_live`(:427-431) -- 진행 중 봉 한 줄 갈아 끼우기(표시 전용). upd = 쓴 벽시계 ms.
pub fn db_live(con: &Connection, bk: Option<&Rec>, marks: &[Mark], upd_ms: i64) -> rusqlite::Result<()> {
    let bar = bk.map_or("null".to_string(), dumps);
    con.execute("INSERT OR REPLACE INTO live VALUES (0, ?, ?, ?)", params![upd_ms, bar, marks_json(marks)])?;
    Ok(())
}

// ── 시 파일 이어 읽기(`Source` :434-484) ──
pub fn hour_name(h: i64) -> String {
    crate::depth::hour_key(h * 3_600_000).unwrap()
}

/// 그 시의 파일(.jsonl 먼저 -- 수집기는 .gz 를 다 쓴 뒤에야 .jsonl 을 지운다)
pub fn hour_path(dd: &Path, h: i64) -> Option<PathBuf> {
    let n = hour_name(h);
    [dd.join(format!("{n}.jsonl")), dd.join(format!("{n}.jsonl.gz"))].into_iter().find(|p| p.exists())
}

/// 완성된 줄만 내준다. 다음 시 파일이 생기면 지금 파일을 EOF 까지 읽고 넘어간다 · 시 끝 2분 넘게 다음 파일이 없으면 수집기가 죽었다 -- 넘어간다.
/// `now` = 벽시계 초 · `sleep(초)` = false 면 멈춘다(시험 -- 원본은 영원히 기다린다). 원본 generator 처럼 줄을 내줄 때 n 을 세고,
/// 파일을 넘기는 것(hour += 1)은 그 파일의 남은 줄을 다 내준 **뒤**다(사슬이 그 사이에 끊기면 restart 가 옛 시 기준으로 센다).
pub struct Source<'a> {
    pub dd: PathBuf,
    pub hour: i64,
    pub n: u64,
    start: i64,
    rd: Option<Box<dyn Read>>,
    buf: Vec<u8>,
    pending: VecDeque<Option<Vec<u8>>>, // None = 여기서 파일을 넘긴다
    now: Box<dyn Fn() -> f64 + 'a>,
    sleep: Box<dyn Fn(f64) -> bool + 'a>,
    pub stopped: bool,
}

impl<'a> Source<'a> {
    pub fn new(dd: &Path, hour: i64, now: Box<dyn Fn() -> f64 + 'a>, sleep: Box<dyn Fn(f64) -> bool + 'a>) -> Self {
        Self { dd: dd.into(), hour, n: 0, start: hour, rd: None, buf: vec![], pending: VecDeque::new(), now, sleep, stopped: false }
    }

    /// 새 사슬(원본 lines() 를 다시 부름 = 새 generator: 쌓인 줄 버림 · 사슬 첫 파일 = 지금 시).
    pub fn begin(&mut self) {
        (self.buf, self.start) = (vec![], self.hour);
        self.pending.clear();
    }

    fn open(p: &Path) -> Box<dyn Read> {
        let f = std::fs::File::open(p).unwrap();
        if p.extension().is_some_and(|x| x == "gz") { Box::new(flate2::read::MultiGzDecoder::new(f)) } else { Box::new(f) }
    }

    /// 끊긴 뒤(:477-484): 사슬 첫 파일이 아닌 파일 첫머리(줄 ≤2)의 끊김이면 그 파일 스냅숏에서, 아니면 다음 시 파일에서.
    pub fn restart(&mut self, _st: &str) {
        self.rd = None;
        if !(self.n <= 2 && self.hour > self.start) {
            self.hour += 1;
        }
    }

    fn push_lines(&mut self, b: &[u8]) {
        for ln in b.split(|x| *x == b'\n').filter(|l| !l.is_empty()) {
            self.pending.push_back(Some(ln.to_vec()));
        }
    }
}

impl Iterator for Source<'_> {
    type Item = Vec<u8>;
    fn next(&mut self) -> Option<Vec<u8>> {
        loop {
            match self.pending.pop_front() {
                Some(Some(l)) => {
                    self.n += 1;
                    return Some(l);
                }
                Some(None) => {
                    (self.rd, self.hour) = (None, self.hour + 1);
                    continue;
                }
                None => {}
            }
            if self.stopped {
                return None;
            }
            if self.rd.is_none() {
                match hour_path(&self.dd, self.hour) {
                    None => {
                        if (self.now)() > ((self.hour + 1) * 3600 + 120) as f64 {
                            self.hour += 1;
                        }
                        self.stopped = !(self.sleep)(2.0);
                        continue;
                    }
                    Some(p) => (self.rd, self.n, self.buf) = (Some(Self::open(&p)), 0, vec![]),
                }
            }
            let mut chunk = vec![0u8; 1 << 20];
            let k = self.rd.as_mut().unwrap().read(&mut chunk).unwrap_or(0);
            if k > 0 {
                self.buf.extend_from_slice(&chunk[..k]);
                if let Some(cut) = self.buf.iter().rposition(|b| *b == b'\n') {
                    let rest = self.buf.split_off(cut + 1);
                    let done = std::mem::replace(&mut self.buf, rest);
                    self.push_lines(&done);
                }
                continue;
            }
            if hour_path(&self.dd, self.hour + 1).is_none() && (self.now)() <= ((self.hour + 1) * 3600 + 120) as f64 {
                self.stopped = !(self.sleep)(1.0); // 지금 시가 아직 자란다
                continue;
            }
            let mut tail = std::mem::take(&mut self.buf);
            let _ = self.rd.as_mut().unwrap().read_to_end(&mut tail);
            self.push_lines(&tail);
            self.pending.push_back(None);
        }
    }
}

/// hot 테이프 읽기(`_hot_tape` :495-500) -- 못 읽으면 그 봉 사건은 판정 불가(빈 테이프 · 죽지 않는다).
pub fn hot_tape(path: &Path, sym: &str, lo: i64, hi: i64) -> (Vec<(i64, i64, f64, f64)>, Vec<(i64, i64)>) {
    let r = (|| -> rusqlite::Result<_> {
        let con = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let mut st = con.prepare("SELECT ts_sec, price_bin, buy_qty, sell_qty FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ? AND ts_sec < ?")?;
        let rows = st.query_map(params![sym.to_lowercase(), lo, hi], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, Option<f64>>(2)?.unwrap_or(0.0), r.get::<_, Option<f64>>(3)?.unwrap_or(0.0)))
        })?.collect::<Result<Vec<_>, _>>()?;
        let mut st = con.prepare("SELECT from_ms / 1000, to_ms / 1000 FROM gaps WHERE symbol = ? AND to_ms >= ? AND from_ms < ?")?;
        let gaps = st.query_map(params![sym.to_lowercase(), lo * 1000, hi * 1000], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))?.collect::<Result<Vec<_>, _>>()?;
        Ok((rows, gaps))
    })();
    r.unwrap_or_else(|e| {
        eprintln!("테이프 읽기 실패 {e}");
        (vec![], vec![])
    })
}

/// `main()`(:503-529): 마지막 온전한 봉 1시간 앞 시 파일부터(오래 멈췄으면 최근 26시간) · 사슬이 끊기면 스냅숏에서 다시.
/// `now`·`sleep` = 벽시계·잠(시험은 가짜) · `seen(t)` = 1초 줄을 넘기기 직전(시험 시계용).
pub fn run(sym: &str, dd: &Path, tape: &Path, db: &Path, now: &dyn Fn() -> f64, sleep: &dyn Fn(f64) -> bool, seen: &dyn Fn(i64)) -> rusqlite::Result<()> {
    let (tick, _, ks, idx_max) = coin(sym).expect("코인");
    let con = db_open(db)?;
    let last: Option<f64> = con.query_row("SELECT max(t) FROM bars WHERE warm = 0", [], |r| r.get(0))?;
    let now_h = (now() as i64).div_euclid(3600);
    let h0 = (now_h - 26).max(last.map_or(now_h - 26, |l| (l as i64).div_euclid(3_600_000) - 1));
    eprintln!("시작 · 마지막 온전한 봉 {last:?} · {} 파일부터", hour_name(h0));
    let mine = std::cell::RefCell::new(BTreeSet::new());
    let out = |bk: &Rec, m: Option<&[Mark]>| {
        if let Err(e) = db_write(&con, bk, m, &mut mine.borrow_mut()) {
            eprintln!("봉 쓰기 실패 {e}");
        }
    };
    let live = |bk: Option<&Rec>, m: &[Mark]| {
        if let Err(e) = db_live(&con, bk, m, (now() * 1000.0) as i64) {
            eprintln!("live 쓰기 실패 {e}");
        }
    };
    let bars = std::cell::RefCell::new(Bars::new(sym, Box::new(|lo, hi| hot_tape(tape, sym, lo, hi)), Box::new(out), Some(Box::new(live))));
    let mut src = Source::new(dd, h0, Box::new(now), Box::new(sleep));
    loop {
        bars.borrow_mut().chain();
        src.begin();
        let st = {
            let mut on_sec = |t: i64, b: &[f64], a: &[f64], bb: i64, ba: i64| {
                seen(t);
                bars.borrow_mut().sec(sec_row(t, b, a, bb, ba), now() * 1000.0);
            };
            let mut on_ev = |e: Event| bars.borrow_mut().event(e);
            replay(&mut src, tick, ks, idx_max, &mut on_sec, &mut on_ev)
        };
        if src.stopped {
            return Ok(());
        }
        eprintln!("사슬 끝 {st} · {} {}줄 -- 스냅숏에서 다시", hour_name(src.hour), src.n);
        src.restart(st);
    }
}

/// 정답지 재생(시험·예제 공용 -- 원본 정답지 생성기와 같은 바깥): 테이프 SQLite 를 만들고, 시계 = 처음 한 번은 «첫 시 + 26시간»(시작 시 계산),
/// 뒤로는 «방금 넘긴 거래소 시각 + 10초» · 잠 = 멈춤(마지막 시 파일 끝). 결과 = out/{ob5m,bookdepth}.sqlite.
pub fn replay_fixture(fix: &Path, out: &Path) {
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(fix.join("ob5m_tape.json")).unwrap()).unwrap();
    let (h0, nh) = (meta["h0"].as_i64().unwrap(), meta["nh"].as_i64().unwrap());
    std::fs::create_dir_all(out).unwrap();
    let tape = out.join("binance_tape.sqlite");
    if !tape.exists() {
        let con = Connection::open(&tape).unwrap();
        con.execute_batch("CREATE TABLE trade_tape_1s(symbol TEXT, ts_sec INTEGER, price_bin INTEGER, buy_qty REAL, sell_qty REAL, buy_n INTEGER, sell_n INTEGER, \
            buy_max REAL, sell_max REAL, retail_buy_qty REAL, retail_sell_qty REAL, whale_buy_qty REAL, whale_sell_qty REAL, whale_buy_n INTEGER, \
            whale_sell_n INTEGER, retail_buy_n INTEGER, retail_sell_n INTEGER, order_buy_n INTEGER, order_sell_n INTEGER, PRIMARY KEY (symbol, ts_sec, price_bin)) WITHOUT ROWID;
            CREATE TABLE gaps(symbol TEXT, from_ms INTEGER, to_ms INTEGER, reason TEXT); BEGIN").unwrap();
        let rows: Vec<(i64, i64, f64, f64)> = meta["tape"].as_array().unwrap().iter()
            .map(|r| (r[0].as_i64().unwrap(), r[1].as_i64().unwrap(), r[2].as_f64().unwrap(), r[3].as_f64().unwrap())).collect();
        let have: BTreeSet<(i64, i64)> = rows.iter().map(|r| (r.0, r.1)).collect();
        for (a, b, c, d) in rows.iter().copied().chain((h0 * 3600..(h0 + nh) * 3600).filter(|s| !have.contains(&(*s, 25000))).map(|s| (s, 25000, 0.01, 0.0))) {
            con.execute("INSERT INTO trade_tape_1s(symbol, ts_sec, price_bin, buy_qty, sell_qty) VALUES ('ethusdt', ?, ?, ?, ?)", params![a, b, c, d]).unwrap();
        }
        con.execute("INSERT INTO gaps VALUES ('ethusdt', ?, ?, 'ws_reconnect')", params![(h0 + 9) * 3_600_000 + 100_000, (h0 + 9) * 3_600_000 + 130_000]).unwrap();
        con.execute_batch("COMMIT").unwrap();
    }
    let dd = fix.join("ob5m").join("ETHUSDT");
    for which in ["ob5m", "bookdepth"] {
        let (first, last_t) = (std::cell::Cell::new(true), std::cell::Cell::new(0i64));
        let now = || {
            if first.replace(false) {
                ((h0 + 26) * 3600) as f64 + 5.0
            } else {
                last_t.get() as f64 / 1000.0 + 10.0
            }
        };
        let sleep = |_s: f64| false;
        if which == "ob5m" {
            run("ETHUSDT", &dd, &tape, &out.join("ob5m.sqlite"), &now, &sleep, &|t| last_t.set(t)).unwrap();
        } else {
            crate::bookdepth::run("ETHUSDT", &dd, &out.join("bookdepth.sqlite"), &now, &sleep, &|t| last_t.set(t)).unwrap();
        }
    }
}
