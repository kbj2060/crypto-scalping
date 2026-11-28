//! 레벨 5분 -- 체결·1분봉 → 지지·저항 사건 → 피쳐 87 → 앙상블 → 워커 기록 줄 (2026-11-20 · R5 셋째 조각).
//! 원본 scripts/live_sr_lean_20261118.py `Live`(2026-11-18 배포 · A19) + 그것이 부르는 연구 함수
//! scripts/research_sr_ml_rl_20261118.py `candidates`·`one` 과 같은 값을 낸다. 표시 전용 -- 매매 근거 아님(진입 밴딧 비용 뒤 −7bp).
//! 사건 = 1h·4h·24h 굴림 고저의 접근(0.1σ 안)·돌파 · 결정 = 사건 초 + 5초 · 값 = 다음 300초 안 ±0.1σ1h 중 사건 쪽을 먼저 닿을 확률.
//!
//! 끝자리까지 맞추려고 따른 계산(numpy 2.4.3 · pandas 2.3.3 · 파이썬 3.11, 개발기 실측 대조):
//! - `rolling(1440, min_periods=1200).std()` = pandas roll_var(ddof 1): Welford + Kahan(더하기·빼기 보정 따로) + 같은 값 연속 수 → zsqrt.
//!   설치본에 .pyx 소스가 없어 기억한 식을 파이썬으로 옮겨 pandas 와 6만 칸 비트 대조(tools/check_rollvar.py).
//!   🔴창 시작이 프레임 첫 행에 묶여 있다(온라인 누적) -- 같은 3,000행 프레임(지금 분 − 2,999 ..= 지금 분)에서 처음부터 다시 돈다.
//! - `rolling(w, min_periods).max()/min()` = 창 안 NaN 뺀 최대/최소(값 개수 ≥ min_periods 일 때만).
//! - `.sum()`·`np.nansum` = numpy 짝 합(box_hold::npsum) · `np.cumsum` = 첫 값 그대로 + 차례 합(tools/check_npsum.py).
//! - `np.log`·`np.exp` = libm(서버 = AVX-512 없음 · 개발기는 NPY_DISABLE_CPU_FEATURES 로 같은 쪽).
//! - 파이썬 `sum()`(1분봉 거래량) = 3.11 차례 합. 🔴3.12 부터는 보정 합(Neumaier)이라 끝자리가 다를 수 있다 -- 서버 파이썬 버전 미확인.
//! - 기록 줄 = `json.dumps(dict(k=kind, **row), ensure_ascii=False)`: 실수 = repr(가장 짧은 왕복 십진) · NaN·Infinity 그대로.
use crate::box_hold::{npext, npsum, py_round};
use crate::SrLean;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

const KEEP_S: i64 = 3 * 3600; // 초 버퍼(피쳐는 3,610초 앞까지)
const KEEP_M: usize = 3000; // 분봉(24시간 고저 + σ 1,440분)
const HZ: i64 = 300; // 라벨 y_fp10_300: 300초 · ±0.10σ1h
const DB: f64 = 0.10;
const W: i64 = 5; // 결정 = 사건 초 + 5초
const PRE: usize = 3600;
const PAD: i64 = 10;
const WINS: [usize; 8] = [1, 3, 10, 30, 60, 300, 900, 3600];
const LOOKS: [i64; 3] = [60, 240, 1440];
const D_AP: f64 = 0.1;
const GAP: usize = 60; // 같은 (look, kind, side) 사건이 직전 60분에 없을 때만
const EPS: f64 = 1e-9;
const NAN: f64 = f64::NAN;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Kind {
    Apr, // 레벨의 0.1σ 안으로 들어옴
    Brk, // 레벨을 넘음
}
impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Apr => "apr",
            Kind::Brk => "brk",
        }
    }
}

/// REST 1분봉 한 줄에서 쓰는 칸.
#[derive(Clone, Copy)]
pub struct Kline {
    pub open_ms: i64,
    pub h: f64,
    pub l: f64,
    pub c: f64,
    pub v: f64,
    pub tb: f64,
}
impl Kline {
    /// `[open_ms, o, h, l, c, v, close_ms, qv, n, taker_buy_base, …]` (값은 문자열) -- 원본 `int(r[0])`·`float(r[2])` 꼴.
    pub fn from_rest(r: &[Value]) -> Option<Kline> {
        let f = |i: usize| r.get(i)?.as_str()?.parse::<f64>().ok();
        Some(Kline { open_ms: r.first()?.as_i64()?, h: f(2)?, l: f(3)?, c: f(4)?, v: f(5)?, tb: f(9)? })
    }
}

/// 체결 없던 초는 없음 · [ph, pl, pc, bq, sq, nt, q2].
type Sec = [f64; 7];
/// 분봉 [h, l, c, v, tb] (체결 없는 분 = 거래소 1분봉이면 h·l·c NaN·v 0, 내 초로 만들면 전부 NaN -- 원본 그대로).
type Bar = [f64; 5];

struct Pend {
    mi: i64,
    look: i64,
    kind: Kind,
    side: i64,
    a: i64,
    prev: i64,
    th: f64,
    l: f64,
    s1h: f64,
}

struct Wait {
    id: String,
    d: i64,
    side: i64,
    s1h: f64,
    p: f64,
}

struct Win {
    pc: Vec<f64>,
    ph: Vec<f64>,
    pl: Vec<f64>,
    bq: Vec<f64>,
    sq: Vec<f64>,
    nt: Vec<f64>,
    q2: Vec<f64>,
}

struct Cand {
    look: i64,
    kind: Kind,
    side: i64,
    th: f64,
    l: f64,
    s1h: f64,
}

/// 초 버퍼 + 분봉 + 사건 대기열. 네트워크를 모른다 -- 원본 `Live` 처럼 trade()·second()·klines() 만 받는다.
/// 기록 줄은 `log`(ethusdt.jsonl: ev·res·skip) · `feat`(ethusdt_feat.jsonl) 에 쌓인다 -- 부르는 쪽이 비워 가며 파일에 덧붙인다.
pub struct Live<'m> {
    art: &'m SrLean,
    sec: HashMap<i64, Sec>,
    /// 이 초 «다음»부터 빈틈 없음(시작·구멍 뒤 다시 잡는다). 원본 run() 은 첫 체결 때 `t0 = done = 체결 초`.
    pub t0: Option<i64>,
    /// 마지막으로 닫은 초.
    pub done: Option<i64>,
    m: BTreeMap<i64, Bar>,
    pub need_klines: bool,
    seen: HashSet<(i64, i64, Kind, i64)>,
    pend: Vec<Pend>,
    wait: Vec<Wait>,
    asecs: VecDeque<i64>,
    pub log: Vec<String>,
    pub feat: Vec<String>,
}

impl<'m> Live<'m> {
    pub fn new(art: &'m SrLean) -> Self {
        Live { art, sec: HashMap::new(), t0: None, done: None, m: BTreeMap::new(), need_klines: true, seen: HashSet::new(), pend: vec![], wait: vec![],
               asecs: VecDeque::new(), log: vec![], feat: vec![] }
    }

    /// 체결 → 1초 묶음(build_eth_aggtrades_1s_20261118.agg 와 같은 정의).
    pub fn trade(&mut self, t_ms: i64, p: f64, q: f64, first: i64, last: i64, buyer_maker: bool) {
        let r = self.sec.entry(t_ms.div_euclid(1000)).or_insert([p, p, p, 0.0, 0.0, 0.0, 0.0]);
        r[0] = if p > r[0] { p } else { r[0] }; // 파이썬 max(r0, p): 같으면 앞 값
        r[1] = if p < r[1] { p } else { r[1] };
        r[2] = p;
        r[if buyer_maker { 4 } else { 3 }] += q;
        r[5] += (last - first + 1) as f64;
        r[6] += q * q;
    }

    /// R.Tape.window 와 같은 꼴. 아직 안 온 초(> done)는 체결 없는 초로 채운다 -- 피쳐는 결정 초까지만 본다.
    fn window(&self, a: i64, b: i64) -> Option<Win> {
        if a <= self.t0? {
            return None; // a 직전 체결가가 있어야 한다(연구 Tape 와 같은 조건)
        }
        let n = (b - a + 1) as usize;
        let mut w = Win { pc: vec![NAN; n], ph: vec![NAN; n], pl: vec![NAN; n], bq: vec![0.0; n], sq: vec![0.0; n], nt: vec![0.0; n], q2: vec![0.0; n] };
        for s in a..=b.min(self.done.expect("done 없이 window")) {
            if let Some(r) = self.sec.get(&s) {
                let i = (s - a) as usize;
                (w.ph[i], w.pl[i], w.pc[i], w.bq[i], w.sq[i], w.nt[i], w.q2[i]) = (r[0], r[1], r[2], r[3], r[4], r[5], r[6]);
            }
        }
        if w.pc[0].is_nan() {
            w.pc[0] = (a - 3599..a).rev().find_map(|s| self.sec.get(&s).map(|r| r[2])).unwrap_or(NAN);
        }
        for i in 1..n {
            if w.pc[i].is_nan() {
                w.pc[i] = w.pc[i - 1];
            }
        }
        for i in 0..n {
            if w.ph[i].is_nan() {
                (w.ph[i], w.pl[i]) = (w.pc[i], w.pc[i]);
            }
        }
        Some(w)
    }

    /// 초 s 근처 체결을 잃었다 -- 그 뒤부터 다시 빈틈 없음으로 친다. 걸린 사건은 버린다(틀린 값을 내느니 안 낸다).
    pub fn hole(&mut self, s: i64) {
        (self.t0, self.need_klines) = (Some(s + 1), true);
        self.pend.clear();
        self.wait.clear();
    }

    /// REST 1분봉 -- 내 초 버퍼가 온전히 덮지 못한 분만 채운다. cur = 거래소 시각으로 지금 형성 중인 분(뺀다).
    pub fn klines(&mut self, rows: &[Kline], cur: i64) {
        let t0m = self.t0.expect("t0 없이 klines").div_euclid(60);
        for r in rows {
            let mi = r.open_ms.div_euclid(60000);
            if mi < cur && mi <= t0m {
                let dead = r.v <= 0.0; // 체결 없는 분은 연구 분봉에서 NaN
                self.m.insert(mi, if dead { [NAN, NAN, NAN, r.v, r.tb] } else { [r.h, r.l, r.c, r.v, r.tb] });
            }
        }
        self.trim_m();
        self.need_klines = false;
        for mi in t0m + 1..cur {
            let b = self.minute(mi, mi * 60 + 59); // klines 를 기다리는 동안 닫힌 분은 내 초로 만든다(초는 온전히 받았다)
            self.m.insert(mi, b);
        }
    }

    /// run() 의 «처음 받은 1분봉 끝 몇 분 다시 확인»(받은 지 2분 뒤 한 번 · REST 가 몇 초 늦어 직전 분이 덜 찬 값일 수 있다 -- 11-18 실측).
    pub fn recheck(&mut self, rows: &[Kline]) {
        let t0m = self.t0.expect("t0 없이 recheck").div_euclid(60);
        for r in rows {
            let mi = r.open_ms.div_euclid(60000);
            if mi <= t0m && self.m.contains_key(&mi) && r.v > 0.0 {
                self.m.insert(mi, [r.h, r.l, r.c, r.v, r.tb]);
            }
        }
    }

    fn trim_m(&mut self) {
        while self.m.len() > KEEP_M {
            self.m.pop_first();
        }
    }

    fn minute(&self, mi: i64, upto: i64) -> Bar {
        let rs: Vec<&Sec> = (mi * 60..=upto).filter_map(|s| self.sec.get(&s)).collect();
        if rs.is_empty() {
            return [NAN; 5];
        }
        let (mut h, mut l, mut v, mut tb) = (rs[0][0], rs[0][1], 0.0, 0.0);
        for r in &rs {
            h = if r[0] > h { r[0] } else { h };
            l = if r[1] < l { r[1] } else { l };
            v += r[3] + r[4]; // 파이썬 3.11 sum() = 차례 합
            tb += r[3];
        }
        [h, l, rs[rs.len() - 1][2], v, tb]
    }

    /// 닫힌 초마다.
    pub fn second(&mut self, s: i64) {
        self.done = Some(s);
        let mi = s.div_euclid(60);
        if s % 60 == 59 {
            self.sec.retain(|&x, _| x >= s - KEEP_S);
        }
        let Some(t0) = self.t0 else { return };
        if self.need_klines || self.m.is_empty() {
            return;
        }
        if mi * 60 > t0 {
            let b = self.minute(mi, s); // 이 분을 처음부터 온전히 받았을 때만 내 초로 분봉을 만든다(아니면 다음 klines 가 채운다)
            self.m.insert(mi, b);
        }
        if self.m.contains_key(&mi) {
            let base = mi - KEEP_M as i64 + 1;
            let fr: Vec<Bar> = (base..=mi).map(|k| self.m.get(&k).copied().unwrap_or([NAN; 5])).collect();
            if mi * 60 > t0 {
                for r in candidates(&fr) {
                    let key = (mi, r.look, r.kind, r.side);
                    if self.seen.contains(&key) {
                        continue;
                    }
                    let x = self.window(mi * 60, s).expect("mi*60 > t0");
                    let Some(k) = first_cross(&x.ph, &x.pl, r.side, r.kind, r.th) else { continue }; // RL.one 의 사건 초 규칙(체결 없는 초 = 직전 체결가)
                    let a = mi * 60 + k as i64;
                    self.seen.insert(key);
                    self.asecs.push_back(a);
                    let prev = self.asecs.iter().filter(|&&x| x >= a - 3600).count() as i64 - 1;
                    self.pend.push(Pend { mi, look: r.look, kind: r.kind, side: r.side, a, prev, th: r.th, l: r.l, s1h: r.s1h });
                }
            }
            if self.pend.iter().any(|e| s >= e.a + W) {
                let (due, keep): (Vec<Pend>, Vec<Pend>) = std::mem::take(&mut self.pend).into_iter().partition(|e| s >= e.a + W);
                self.pend = keep;
                // 분봉 맥락은 1,441분 앞까지 본다 -- 모자라면 NaN 피쳐로 채점하지 않는다(reindex 라 one() 의 검사는 안 걸린다)
                let short = fr[KEEP_M - 1443..KEEP_M - 1].iter().filter(|b| !b[2].is_nan()).count() < 1430;
                for e in &due {
                    self.score(e, if short { None } else { Some((&fr, base)) });
                }
            }
        }
        while self.asecs.front().is_some_and(|&x| x < s - 7200) {
            self.asecs.pop_front();
        }
        if s % 60 == 59 {
            self.seen.retain(|k| k.0 >= mi - 2);
            self.trim_m();
        }
        let (due, keep): (Vec<Wait>, Vec<Wait>) = std::mem::take(&mut self.wait).into_iter().partition(|e| s >= e.d + HZ);
        self.wait = keep;
        for e in &due {
            self.resolve(e);
        }
    }

    fn score(&mut self, e: &Pend, fr: Option<(&[Bar], i64)>) {
        let f = fr.and_then(|(fr, base)| self.one(fr, base, e));
        let Some(mut f) = f else {
            // 초 버퍼 1시간·분봉 1,441분이 아직 안 찼다
            self.log.push(format!("{{\"k\": \"skip\", \"a\": {}, \"look\": {}, \"kind\": \"{}\", \"side\": {}, \"why\": \"warm\"}}", e.a, e.look, e.kind.as_str(), e.side));
            return;
        };
        assert_eq!(f["asec"] as i64, e.a, "사건 초가 감지 때와 다르다(파이썬도 여기서 죽는다)");
        let hod = (e.a % 86400) as f64 / 86400.0;
        let tau = 2.0 * std::f64::consts::PI;
        f.insert("m_hsin".into(), (tau * hod).sin());
        f.insert("m_hcos".into(), (tau * hod).cos());
        f.insert("m_wkend".into(), ((e.a.div_euclid(86400) + 3).rem_euclid(7) >= 5) as u8 as f64); // 1970-01-01 = 목요일(3)
        f.insert("m_prev60".into(), e.prev as f64);
        let x: Vec<f64> = self.art.cols.iter().map(|c| f[c.as_str()]).collect();
        // 파이썬은 무한대·f32 넘침 입력에서 SimpleImputer 가 예외 → 워커가 죽고 감시 스크립트가 다시 띄운다
        let (p, grp) = self.art.predict(&x).expect("모델이 받지 않는 피쳐(파이썬도 여기서 죽는다)");
        let d = e.a + W;
        let pp = self.window(d, d).expect("d > t0").pc[0];
        let id = format!("{}_{}_{}_{}", e.a, e.look, e.kind.as_str(), e.side);
        self.log.push(format!(
            "{{\"k\": \"ev\", \"id\": \"{id}\", \"a\": {}, \"d\": {d}, \"lag\": {}, \"side\": {}, \"kind\": \"{}\", \"look\": {}, \"L\": {}, \"s1h\": {}, \"P\": {}, \"p\": {}, \"grp\": \"{grp}\"}}",
            e.a, self.done.unwrap() - d, e.side, e.kind.as_str(), e.look, jf(e.l), jf(e.s1h), jf(pp), jf(py_round(p, 6))
        ));
        let mut ln = format!("{{\"k\": \"feat\", \"id\": \"{id}\"");
        for (c, v) in self.art.cols.iter().zip(&x) {
            ln += &format!(", \"{c}\": {}", jf(*v));
        }
        self.feat.push(ln + "}");
        self.wait.push(Wait { id, d, side: e.side, s1h: e.s1h, p: pp });
    }

    fn resolve(&mut self, e: &Wait) {
        let Some(x) = self.window(e.d + 1, e.d + HZ) else { return }; // 라벨은 결정 다음 초부터(연구와 같은 식)
        let dd = DB * e.s1h;
        let (up, dn) = (e.p * dd.exp(), e.p * (-dd).exp());
        let n = x.ph.len();
        let iu = x.ph.iter().position(|&v| v >= up).unwrap_or(n);
        let idn = x.pl.iter().position(|&v| v <= dn).unwrap_or(n);
        let fp = if iu < idn { 1.0 } else if idn < iu { -1.0 } else if iu == n { NAN } else { 0.0 }; // R.first_pass
        let last = x.pc[n - 1];
        let lr = (last / e.p).ln();
        let sgn = if lr > 0.0 { 1.0 } else if lr < 0.0 { -1.0 } else { lr }; // np.sign: 0 → 0 · NaN → NaN
        let z = (if fp == 1.0 || fp == -1.0 { fp } else { sgn }) * e.side as f64;
        let y = if z > 0.0 { "1" } else if z < 0.0 { "0" } else { "null" };
        let hit = iu.min(idn); // 위·아래 어느 쪽이든 처음 닿은 초
        let hit_s = if hit < n { (hit + 1).to_string() } else { "null".into() };
        let r = py_round(e.side as f64 * lr * 1e4, 2);
        self.log.push(format!("{{\"k\": \"res\", \"id\": \"{}\", \"y\": {y}, \"hit_s\": {hit_s}, \"r\": {}}}", e.id, jf(r)));
    }

    /// RL.one -- 사건 하나의 피쳐(결정 초 d 까지의 초 · 사건 분 앞의 분봉). 원본의 라벨(y_·r_·q_)은 결정 뒤 자료라 피쳐가 아니어서 안 옮긴다.
    fn one(&self, fr: &[Bar], base: i64, e: &Pend) -> Option<HashMap<String, f64>> {
        let (mi, side, look, s1h) = (e.mi, e.side, e.look as usize, e.s1h);
        // 원본 창 끝 = mi*60+60+W+FWD+2 지만 d(≤ 분 끝 + 5초) 뒤는 피쳐가 안 본다 -- 같은 값. `mi - base - 1441 < 0` 검사는 3,000행 reindex 라 늘 거짓(원본 주석) -- 뺐다.
        let x = self.window(mi * 60 - PRE as i64 - PAD, mi * 60 + 60 + W)?;
        let o = PRE + PAD as usize;
        let a = o + first_cross(&x.ph[o..o + 60], &x.pl[o..o + 60], side, e.kind, e.th)?;
        let d = a + W as usize;
        let sf = side as f64;
        let (pc, ph, pl) = (&x.pc, &x.ph, &x.pl);
        let lp: Vec<f64> = pc[d - PRE..=d].iter().map(|v| v.ln()).collect(); // lp[k] = 초 d-PRE+k
        let p = pc[d];
        let vol: Vec<f64> = x.bq.iter().zip(&x.sq).map(|(b, s)| b + s).collect();
        let sv: Vec<f64> = x.bq.iter().zip(&x.sq).map(|(b, s)| sf * (b - s)).collect();
        let s1s = s1h / 60.0;
        let seg = |v: &[f64]| cum(&v[d - PRE + 1..=d]);
        let d1: Vec<f64> = lp.windows(2).map(|w| w[1] - w[0]).collect();
        let (cv, csv, cn, cq) = (seg(&vol), seg(&sv), seg(&x.nt), seg(&x.q2));
        let cr2 = cum(&d1.iter().map(|v| v * v).collect::<Vec<_>>());
        let cab = cum(&d1.iter().map(|v| v.abs()).collect::<Vec<_>>());
        let tail = |c: &[f64], w: usize| c[PRE] - c[PRE - w];
        let bv = tail(&cv, PRE) / PRE as f64 + EPS;
        let bn = tail(&cn, PRE) / PRE as f64 + EPS;
        let mut f: HashMap<String, f64> = HashMap::with_capacity(100);
        let mut put = |k: String, v: f64| {
            f.insert(k, v);
        };
        put("asec".into(), (mi * 60 - PRE as i64 - PAD + a as i64) as f64);
        for w in WINS {
            let wf = w as f64;
            let (v, n, net) = (tail(&cv, w), tail(&cn, w), lp[PRE] - lp[PRE - w]);
            if w < PRE {
                put(format!("m_vol{w}"), ((v / wf + EPS) / bv).ln());
                put(format!("m_n{w}"), ((n / wf + EPS) / bn).ln());
                put(format!("m_sz{w}"), ((v + EPS) / (n + EPS)).ln() - (bv / bn).ln());
            }
            put(format!("s_imb{w}"), tail(&csv, w) / (v + EPS));
            put(format!("s_ret{w}"), sf * net / (s1s * wf.sqrt()));
            put(format!("m_rv{w}"), (tail(&cr2, w).sqrt() / (s1s * wf.sqrt()) + 1e-3).ln());
            put(format!("m_conc{w}"), (tail(&cq, w) / (v * v + EPS) + 1e-6).ln());
            if w >= 10 {
                let (hi, lo) = (npext(&ph[d + 1 - w..=d], true), npext(&pl[d + 1 - w..=d], false));
                put(format!("s_rpos{w}"), (if side > 0 { p - lo } else { hi - p }) / (hi - lo + EPS));
                put(format!("s_eff{w}"), sf * net / (tail(&cab, w) + EPS));
            }
        }
        let ob = a..=d;
        let vo = npsum(&vol[ob.clone()]);
        put("s_imbo".into(), npsum(&sv[ob.clone()]) / (vo + EPS));
        put("m_volo".into(), ((vo / (W + 1) as f64 + EPS) / bv).ln());
        put("s_pos".into(), sf * (p / e.l).ln() / s1h);
        let ext = if side > 0 { npext(&ph[ob.clone()], true) } else { npext(&pl[ob], false) };
        put("s_ext".into(), sf * (ext / e.l).ln() / s1h);
        put("m_asec".into(), (a - o) as f64 / 60.0);
        let col = |j: usize| fr.iter().map(|b| b[j]).collect::<Vec<f64>>();
        let (hh, lo, cc, vv, tb) = (col(0), col(1), col(2), col(3), col(4));
        let i = (mi - base) as usize;
        let wd = i - look..i;
        let ext_w = if side > 0 { &hh[wd.clone()] } else { &lo[wd.clone()] };
        let j = nanarg(ext_w, side > 0).expect("np.nanargmax: 창이 전부 NaN(파이썬도 여기서 죽는다)");
        put("m_lage".into(), ((look - j) as f64).ln() / (look as f64).ln());
        put("m_near".into(), ext_w.iter().filter(|&&v| (v / e.l).ln().abs() < 0.1 * s1h).count() as f64 / look as f64);
        put("m_rng".into(), (nanext(&hh[wd.clone()], true) / nanext(&lo[wd], false)).ln() / (s1h * (look as f64 / 60.0).sqrt()));
        put("s_side".into(), sf);
        put("m_kind".into(), (e.kind == Kind::Brk) as u8 as f64);
        put("m_look".into(), (look as f64).ln());
        put("m_ls".into(), s1h.ln());
        for h in [15usize, 60, 240, 1440] {
            put(format!("s_mom{h}"), sf * (cc[i - 1] / cc[i - 1 - h]).ln() / (s1h * (h as f64 / 60.0).sqrt()));
        }
        for h in [60usize, 240] {
            let den = nansum(&vv[i - h..i]);
            let den = if 1e-9 > den { 1e-9 } else { den }; // 파이썬 max(den, 1e-9)
            put(format!("s_tb{h}"), sf * (2.0 * nansum(&tb[i - h..i]) / den - 1.0));
        }
        Some(f)
    }

    /// 생존 표시(ethusdt_state.json · run() 은 10초마다 임시 파일 → 바꿔치기). now = 벽시계 초(정수).
    pub fn state_json(&self, now: i64) -> String {
        // 1분봉을 못 받은 동안은 계속 «준비 중» · 준비 끝 = t0 + PRE + PAD + 60
        let warm = match self.t0 {
            Some(t0) if !self.need_klines => t0 + PRE as i64 + PAD + 60,
            _ => now + 3670,
        };
        let opt = |v: Option<i64>| v.map_or("null".into(), |x| x.to_string());
        let (t, b) = (self.art.rate.top, self.art.rate.bot);
        format!(
            "{{\"ts\": {now}, \"done\": {}, \"t0\": {}, \"warm_until\": {warm}, \"minutes\": {}, \"need_klines\": {}, \"thr\": [{}, {}], \"rate\": {{\"top\": [{}, {}], \"bot\": [{}, {}]}}}}",
            opt(self.done), opt(self.t0), self.m.len(), self.need_klines, jf(self.art.thr[0]), jf(self.art.thr[1]), t[0], t[1], b[0], b[1]
        )
    }
}

/// 사건 초: 고가(저가)가 문턱을 넘은(brk) · 닿은(apr) 첫 칸.
fn first_cross(ph: &[f64], pl: &[f64], side: i64, kind: Kind, th: f64) -> Option<usize> {
    if side > 0 {
        ph.iter().position(|&v| if kind == Kind::Brk { v > th } else { v >= th })
    } else {
        pl.iter().position(|&v| if kind == Kind::Brk { v < th } else { v <= th })
    }
}

/// [0, x0, x0+x1, …] -- `np.concatenate([[0.0], np.cumsum(x)])`.
fn cum(x: &[f64]) -> Vec<f64> {
    let mut c = Vec::with_capacity(x.len() + 1);
    c.push(0.0);
    let mut s = 0.0;
    for (k, v) in x.iter().enumerate() {
        s = if k == 0 { *v } else { s + v };
        c.push(s);
    }
    c
}

fn nansum(x: &[f64]) -> f64 {
    npsum(&x.iter().map(|v| if v.is_nan() { 0.0 } else { *v }).collect::<Vec<_>>())
}

/// np.nanmax / np.nanmin (전부 NaN → NaN).
fn nanext(x: &[f64], max: bool) -> f64 {
    nanarg(x, max).map_or(NAN, |j| x[j])
}

/// np.nanargmax / np.nanargmin: NaN 뺀 첫 최대(최소) 자리.
fn nanarg(x: &[f64], max: bool) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (k, &v) in x.iter().enumerate() {
        if !v.is_nan() && best.is_none_or(|b| if max { v > x[b] } else { v < x[b] }) {
            best = Some(k);
        }
    }
    best
}

/// pandas rolling(w, min_periods=minp).std()(ddof 1) 전체 열 -- tools/check_rollvar.py 가 비트 대조한 순서.
#[doc(hidden)]
pub fn roll_std(x: &[f64], w: usize, minp: usize) -> Vec<f64> {
    roll_var(x, w, minp).into_iter().map(|v| if v.is_nan() { NAN } else if v >= 0.0 { v.sqrt() } else { 0.0 }).collect() // zsqrt
}

/// pandas rolling(w, min_periods=minp).var()(ddof 1).
pub(crate) fn roll_var(x: &[f64], w: usize, minp: usize) -> Vec<f64> {
    let (mut nobs, mut mean, mut ssq, mut ca, mut cr) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut same, mut prev) = (0i64, x[0]);
    let mut out = vec![NAN; x.len()];
    for i in 0..x.len() {
        if i >= w {
            let v = x[i - w]; // 창에서 빠지는 값
            if !v.is_nan() {
                nobs -= 1.0;
                if nobs != 0.0 {
                    let pm = mean - cr;
                    let y = v - cr;
                    let t = y - mean;
                    cr = t + mean - y;
                    mean -= t / nobs;
                    ssq -= (v - pm) * (v - mean);
                } else {
                    (mean, ssq) = (0.0, 0.0);
                }
            }
        }
        let v = x[i];
        if !v.is_nan() {
            nobs += 1.0;
            same = if v == prev { same + 1 } else { 1 }; // GH#42064 같은 값 연속이면 0
            prev = v;
            let pm = mean - ca;
            let y = v - ca;
            let t = y - mean;
            ca = t + mean - y;
            mean = if nobs != 0.0 { mean + t / nobs } else { 0.0 };
            ssq += (v - pm) * (v - mean);
        }
        if nobs >= minp.max(1) as f64 && nobs > 1.0 {
            out[i] = if nobs == 1.0 || same as f64 >= nobs { 0.0 } else { ssq / (nobs - 1.0) };
        }
    }
    out
}

/// pandas rolling(w, min_periods=minp).max()/min() 전체 열: NaN 뺀 최대/최소, 값 개수 < minp 면 NaN.
#[doc(hidden)]
pub fn roll_ext(x: &[f64], w: usize, minp: usize, max: bool) -> Vec<f64> {
    let mut out = vec![NAN; x.len()];
    let mut dq: VecDeque<usize> = VecDeque::new();
    let mut cnt = 0usize;
    for i in 0..x.len() {
        if !x[i].is_nan() {
            cnt += 1;
            while dq.back().is_some_and(|&j| if max { x[j] <= x[i] } else { x[j] >= x[i] }) {
                dq.pop_back();
            }
            dq.push_back(i);
        }
        if i >= w && !x[i - w].is_nan() {
            cnt -= 1;
        }
        while dq.front().is_some_and(|&j| j + w <= i) {
            dq.pop_front();
        }
        if cnt >= minp && !dq.is_empty() {
            out[i] = x[dq[0]];
        }
    }
    out
}

/// RL.candidates 의 마지막 행(지금 분) 몫: kind brk = 고가(저가)가 직전 look 분 고점(저점) L 을 넘음 · apr = L 의 0.1σ 안으로 들어옴.
/// 각 (look, kind, side) 에서 직전 60분에 같은 사건 없음. 순서 = (look, kind, side) 오름차순(원본 sort_values).
fn candidates(fr: &[Bar]) -> Vec<Cand> {
    let n = fr.len();
    let lc: Vec<f64> = fr.iter().map(|b| b[2].ln()).collect();
    let dif: Vec<f64> = (0..n).map(|i| if i == 0 { NAN } else { lc[i] - lc[i - 1] }).collect();
    let sd = roll_std(&dif, 1440, 1200);
    let sq60 = 60f64.sqrt();
    let s1h = |r: usize| if r == 0 { NAN } else { sd[r - 1] * sq60 }; // .shift(1)
    let (h, l): (Vec<f64>, Vec<f64>) = fr.iter().map(|b| (b[0], b[1])).unzip();
    let last = n - 1;
    let s_now = s1h(last);
    let mut out = vec![];
    if s_now.is_nan() {
        return out;
    }
    for look in LOOKS {
        let minp = (look as f64 * 0.98) as usize; // int(look * .98)
        let (hr, lr) = (roll_ext(&h, look as usize, minp, true), roll_ext(&l, look as usize, minp, false));
        let hh = |r: usize| if r == 0 { NAN } else { hr[r - 1] };
        let ll = |r: usize| if r == 0 { NAN } else { lr[r - 1] };
        let up = |r: usize| hh(r) * (-D_AP * s1h(r)).exp();
        let dn = |r: usize| ll(r) * (D_AP * s1h(r)).exp();
        let mut set = |kind: Kind, side: i64, b: &dyn Fn(usize) -> bool, th: f64, lv: f64| {
            if b(last) && !(last - GAP..last).any(b) {
                out.push(Cand { look, kind, side, th, l: lv, s1h: s_now });
            }
        };
        set(Kind::Apr, -1, &|r| l[r] <= dn(r), dn(last), ll(last));
        set(Kind::Apr, 1, &|r| h[r] >= up(r), up(last), hh(last));
        set(Kind::Brk, -1, &|r| l[r] < ll(r), ll(last), ll(last));
        set(Kind::Brk, 1, &|r| h[r] > hh(r), hh(last), hh(last));
    }
    out
}

/// 파이썬 json.dumps 의 실수: repr(가장 짧은 왕복 십진 · 지수는 10^-5 이하·10^16 이상) · NaN · Infinity.
pub fn jf(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    let e = format!("{:e}", x.abs()); // "d.ddde±x" -- 가장 짧은 왕복 자릿수
    let (mant, exp) = e.split_once('e').unwrap();
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if digits == "0" {
        return format!("{sign}0.0");
    }
    let decpt = exp.parse::<i32>().unwrap() + 1;
    let nd = digits.len() as i32;
    let body = if decpt <= -4 || decpt > 16 {
        let rest = &digits[1..];
        let x = decpt - 1;
        format!("{}{}{}e{}{:02}", &digits[..1], if rest.is_empty() { "" } else { "." }, rest, if x < 0 { "-" } else { "+" }, x.abs())
    } else if decpt <= 0 {
        format!("0.{}{digits}", "0".repeat((-decpt) as usize))
    } else if decpt >= nd {
        format!("{digits}{}.0", "0".repeat((decpt - nd) as usize))
    } else {
        format!("{}.{}", &digits[..decpt as usize], &digits[decpt as usize..])
    };
    format!("{sign}{body}")
}
