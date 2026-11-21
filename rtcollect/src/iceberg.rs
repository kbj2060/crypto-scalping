//! 칸별 아이스버그 실시간 탐지기(`scripts/live_eth_iceberg_20261016.py`, 줄 번호는 그 파일) -- ETH·XRP.
//! depthdiff 시 파일을 이어 읽어 30초 창마다 창 시작 mid 칸 ±30칸 잔량(시작·최소·최대·끝)·최우선 극값 · 체결 = hot 테이프 price_bin 칸 합.
//! 사건 = 최우선 2칸 안 · 그 칸 테이커 체결 ≥ 직전 2880창 «창별 최대 칸 체결» 95분위 · 버팀 · 다시 채움 · 체결 ≥ 창 안 최대 표시 잔량 × 2.
//! 출력: `<sym>_events.jsonl`(한 줄 한 건 · 이미 있는 (t_end, side) 는 다시 안 씀) · `<sym>_state.json`(2초마다 -- /api/iceberg · 감시 신선도).
//! 북 재구성은 원본 `Book` 그대로(ob5m 의 replay 와 규칙이 다르다: 끊김에도 계속 읽고 창 valid=0 으로 표시한다).
use crate::ob5m::np_sum;
use rusqlite::{Connection, OpenFlags};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Read;
use std::path::{Path, PathBuf};

pub const K: i64 = 30;
pub const WIN: i64 = 30;
pub const NQ: usize = 2880;
pub const NMIN: usize = 1000;
pub const IDX_MAX: usize = 1_000_000;
pub const FWD0: i64 = 1_792_108_800;
pub const SETTLE: i64 = 5;
const NB: usize = (2 * K + 1) as usize;

/// 코인 → (눈금/$, 칸/$, 칸 가격 자릿수).
pub fn coin(sym: &str) -> Option<(i64, i64, usize)> {
    match sym {
        "ETHUSDT" => Some((100, 10, 2)),
        "XRPUSDT" => Some((10_000, 10_000, 4)),
        _ => None,
    }
}

/// 끝난 30초 창(:76-77).
#[derive(Clone, Debug)]
pub struct Win {
    pub w: i64,
    pub base: i64,
    pub valid: i64,
    pub bs: Vec<f64>,
    pub bmin: Vec<f64>,
    pub bmax: Vec<f64>,
    pub be: Vec<f64>,
    pub as_: Vec<f64>,
    pub amin: Vec<f64>,
    pub amax: Vec<f64>,
    pub ae: Vec<f64>,
    pub bb_min: i64,
    pub ba_max: i64,
    pub bb0: i64,
    pub ba0: i64,
}

fn pyround(x: f64) -> i64 {
    x.round_ties_even() as i64
}

pub struct Book {
    tick: i64,
    tpb: i64,
    bid: Vec<f64>,
    ask: Vec<f64>,
    last_u: i64,
    synced: bool,
    pub valid: i64,
    cur: Option<i64>,
    win: Option<Win>,
    bb: i64,
    ba: i64,
    pub done: Vec<Win>,
}

fn pf(v: &Value) -> Option<f64> {
    crate::py_float(v).ok()
}

impl Book {
    /// `Book.__init__`(:54-62). None = 스냅숏 한쪽이 비었다(원본은 IndexError 로 죽는다 -- 🔧그 시를 건너뛴다 · SHADOW.md §11 #18).
    pub fn new(snap: &Value, tick: i64, per: i64) -> Option<Self> {
        let (mut bid, mut ask) = (vec![0.0; IDX_MAX], vec![0.0; IDX_MAX]);
        for (arr, k) in [(&mut bid, "bids"), (&mut ask, "asks")] {
            for pq in snap[k].as_array().into_iter().flatten() {
                let i = pyround(pf(&pq[0])? * tick as f64);
                if (0..IDX_MAX as i64).contains(&i) {
                    arr[i as usize] = pf(&pq[1])?;
                }
            }
        }
        let bb = bid.iter().rposition(|x| *x != 0.0)? as i64;
        let ba = ask.iter().position(|x| *x != 0.0)? as i64;
        Some(Self { tick, tpb: tick / per, bid, ask, last_u: crate::py_int(&snap["lastUpdateId"]).ok()?, synced: false, valid: 1, cur: None, win: None, bb, ba, done: vec![] })
    }

    /// `buckets`(:45-48): base−K..base+K 칸 합(칸 하나 = 눈금 TPB 개 · numpy 짝 더하기).
    fn buckets(&self, arr: &[f64], base: i64) -> Vec<f64> {
        let lo = (base - K) * self.tpb - self.tpb / 2;
        (0..NB as i64).map(|r| {
            let a = (lo + r * self.tpb).max(0) as usize;
            np_sum(&arr[a.min(arr.len())..((a as i64 + self.tpb) as usize).min(arr.len())])
        }).collect()
    }

    fn best(&mut self) {
        let w = ((self.bb as f64 * 0.02) as i64).max(100);
        let (lo, hi) = ((self.bb - w).max(0) as usize, ((self.bb + w + 1) as usize).min(IDX_MAX));
        if let Some(i) = self.bid[lo..hi].iter().rposition(|x| *x != 0.0) {
            self.bb = (lo + i) as i64;
        }
        let (lo, hi) = ((self.ba - w).max(0) as usize, ((self.ba + w + 1) as usize).min(IDX_MAX)); // 🔴창 폭은 매수 쪽 w 그대로(원본)
        if let Some(i) = self.ask[lo..hi].iter().position(|x| *x != 0.0) {
            self.ba = (lo + i) as i64;
        }
    }

    fn tick(&mut self, sec: i64) {
        let w = sec.div_euclid(WIN);
        if self.win.as_ref().is_some_and(|x| x.w != w) {
            self.done.push(self.win.take().unwrap());
        }
        if self.win.is_none() {
            let base = pyround((self.bb + self.ba) as f64 / 2.0 / self.tpb as f64);
            let (b0, a0) = (self.buckets(&self.bid, base), self.buckets(&self.ask, base));
            self.win = Some(Win { w, base, valid: self.valid, bs: b0.clone(), bmin: b0.clone(), bmax: b0.clone(), be: b0, as_: a0.clone(), amin: a0.clone(),
                                  amax: a0.clone(), ae: a0, bb_min: self.bb, ba_max: self.ba, bb0: self.bb, ba0: self.ba });
        }
        let base = self.win.as_ref().unwrap().base;
        let (b1, a1) = (self.buckets(&self.bid, base), self.buckets(&self.ask, base));
        let (bb, ba, valid) = (self.bb, self.ba, self.valid);
        let x = self.win.as_mut().unwrap();
        for i in 0..NB {
            x.bmin[i] = x.bmin[i].min(b1[i]); // np.minimum(NaN 없음)
            x.bmax[i] = x.bmax[i].max(b1[i]);
            x.amin[i] = x.amin[i].min(a1[i]);
            x.amax[i] = x.amax[i].max(a1[i]);
        }
        (x.be, x.ae) = (b1, a1);
        x.bb_min = x.bb_min.min(bb);
        x.ba_max = x.ba_max.max(ba);
        x.valid = x.valid.min(valid);
    }

    /// `feed`(:85-113).
    pub fn feed(&mut self, line: &[u8]) {
        let Ok(m) = serde_json::from_slice::<Value>(line) else {
            self.valid = 0;
            return;
        };
        if m.get("e").and_then(Value::as_str) != Some("depthUpdate") {
            return;
        }
        let g = |k: &str| crate::py_int(&m[k]).unwrap_or(i64::MIN);
        let (u, uu) = (g("u"), g("U"));
        if !self.synced {
            if u < self.last_u {
                return;
            }
            if !(uu <= self.last_u && self.last_u <= u) {
                self.valid = 0;
            }
            self.synced = true;
        } else if m.get("pu").and_then(|v| crate::py_int(v).ok()) != Some(self.last_u) {
            self.valid = 0;
        }
        self.last_u = u;
        let sec = g("E").div_euclid(1000);
        if self.cur.is_none() {
            self.cur = Some(sec);
        }
        while sec > self.cur.unwrap() {
            let c = self.cur.unwrap();
            self.tick(c);
            self.cur = Some(c + 1);
        }
        for (k, bid) in [("b", true), ("a", false)] {
            for pq in m[k].as_array().into_iter().flatten() {
                let (Some(p), Some(q)) = (pf(&pq[0]), pf(&pq[1])) else { continue };
                let i = pyround(p * self.tick as f64);
                if (0..IDX_MAX as i64).contains(&i) {
                    if bid { self.bid[i as usize] = q } else { self.ask[i as usize] = q }
                }
            }
        }
        self.best();
    }

    /// `close`(:115-120).
    pub fn close(&mut self) {
        if let Some(c) = self.cur {
            self.tick(c);
        }
        if let Some(w) = self.win.take() {
            self.done.push(w);
        }
    }
}

/// np.quantile(a, q) 기본(linear · numpy _lerp).
pub fn np_quantile(a: &[f64], q: f64) -> f64 {
    let mut s = a.to_vec();
    s.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = s.len();
    let vi = (n - 1) as f64 * q;
    if vi >= (n - 1) as f64 {
        return s[n - 1];
    }
    let i = vi.floor();
    let g = vi - i;
    let (x, y) = (s[i as usize], s[i as usize + 1]);
    let d = y - x;
    if g >= 0.5 { y - d * (1.0 - g) } else { x + d * g }
}

/// 파이썬 round(x, n): 이진값 그대로 n 자리 짝수 반올림.
fn pyround_n(x: f64, n: usize) -> f64 {
    format!("{x:.n$}").parse().unwrap()
}

/// `window_events`(:123-145) -- 아이스버그만.
pub fn window_events(w: &Win, sb: &[f32], ss: &[f32], qs: f64, qb: f64, tick: i64, per: i64, dp: usize) -> Vec<Value> {
    if w.valid != 1 {
        return vec![];
    }
    let bin = 1.0 / per as f64;
    let bbk = (w.bb0 as f64 / tick as f64 * per as f64).round_ties_even() as i64 - w.base + K;
    let bak = (w.ba0 as f64 / tick as f64 * per as f64).round_ties_even() as i64 - w.base + K;
    let mut out = vec![];
    for side in [1i64, -1] {
        let k0 = if side == 1 { bbk } else { bak };
        let ks: Vec<i64> = (if side == 1 { k0 - 2..k0 + 1 } else { k0..k0 + 3 }).filter(|k| (0..=2 * K).contains(k)).collect();
        if ks.is_empty() {
            continue;
        }
        let ex = if side == 1 { ss } else { sb };
        let mut k = ks[0];
        for c in &ks {
            if ex[*c as usize] > ex[k as usize] {
                k = *c; // max(key=) = 처음 최대
            }
        }
        let vol = ex[k as usize] as f64;
        let thr = if side == 1 { qs } else { qb };
        if !(thr.is_finite() && thr > 0.0 && vol >= thr) {
            continue;
        }
        let px = (w.base + k - K) as f64 * bin;
        let f32v = |v: &Vec<f64>| v[k as usize] as f32 as f64;
        let (start, end, mx) = if side == 1 { (f32v(&w.bs), f32v(&w.be), f32v(&w.bmax)) } else { (f32v(&w.as_), f32v(&w.ae), f32v(&w.amax)) };
        let held = if side == 1 { w.bb_min as f64 / tick as f64 >= px - bin / 2.0 } else { w.ba_max as f64 / tick as f64 <= px + bin / 2.0 };
        if held && start > 0.0 && vol >= start && end >= 0.5 * start && vol >= 2.0 * mx {
            out.push(json!({"t_end": (w.w + 1) * WIN, "side": side, "px": pyround_n(px, dp), "vol": pyround_n(vol, 3), "max": pyround_n(mx, 3),
                            "start": pyround_n(start, 3), "end": pyround_n(end, 3)}));
        }
    }
    out
}

/// 탐지기 한 프로세스(`main()` :166-227)의 상태.
pub struct Iceberg {
    pub sym: String,
    pub dd: PathBuf,
    pub tape: PathBuf,
    pub out: PathBuf,
    tick: i64,
    per: i64,
    dp: usize,
    seen: HashSet<(i64, i64)>,
    pub n: i64,
    hs: VecDeque<f64>,
    hb: VecDeque<f64>,
    book: Option<Book>,
    fh: Option<std::fs::File>,
    path: Option<PathBuf>,
    buf: Vec<u8>,
    pend: Vec<Win>,
    last_w: Option<i64>,
}

fn hour_name(t: f64) -> String {
    format!("{}.jsonl", crate::depth::hour_key((t.floor() as i64) * 1000).unwrap_or_default())
}

impl Iceberg {
    pub fn new(sym: &str, dd: &Path, tape: &Path, out: &Path) -> std::io::Result<Self> {
        let (tick, per, dp) = coin(sym).ok_or_else(|| std::io::Error::other(format!("모르는 코인 {sym}")))?;
        std::fs::create_dir_all(out)?;
        let coin_l = sym[..sym.len() - 4].to_lowercase();
        let (mut seen, mut n) = (HashSet::new(), 0);
        if let Ok(t) = std::fs::read_to_string(out.join(format!("{coin_l}_events.jsonl"))) {
            for ln in t.lines() {
                let e: Value = serde_json::from_str(ln).map_err(std::io::Error::other)?;
                let te = e["t_end"].as_i64().unwrap_or(0);
                seen.insert((te, e["side"].as_i64().unwrap_or(0)));
                n += (te >= FWD0) as i64;
            }
        }
        Ok(Self { sym: sym.into(), dd: dd.into(), tape: tape.into(), out: out.into(), tick, per, dp, seen, n, hs: VecDeque::new(), hb: VecDeque::new(),
                  book: None, fh: None, path: None, buf: vec![], pend: vec![], last_w: None })
    }

    fn coin_l(&self) -> String {
        self.sym[..self.sym.len() - 4].to_lowercase()
    }

    /// `tape_cells`(:148-154): 창 → [(price_bin, 매수, 매도)].
    fn tape_cells(&self, lo: i64, hi: i64) -> HashMap<i64, Vec<(i64, f64, f64)>> {
        let mut got: HashMap<i64, Vec<(i64, f64, f64)>> = HashMap::new();
        let r = (|| -> rusqlite::Result<()> {
            let con = Connection::open_with_flags(&self.tape, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
            let mut st = con.prepare("SELECT ts_sec / ?, price_bin, sum(buy_qty), sum(sell_qty) FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ? AND ts_sec < ? GROUP BY 1, 2")?;
            let mut q = st.query(rusqlite::params![WIN, self.sym.to_lowercase(), lo, hi])?;
            while let Some(r) = q.next()? {
                let (w, b, bq, sq): (i64, i64, Option<f64>, Option<f64>) = (r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?);
                got.entry(w).or_default().push((b, bq.unwrap_or(0.0), sq.unwrap_or(0.0)));
            }
            Ok(())
        })();
        if let Err(e) = r {
            eprintln!("테이프 읽기 실패: {e}"); // 원본은 예외가 main 밖으로(감독이 다시 띄운다)
        }
        got
    }

    /// `warm`(:157-163).
    fn warm(&mut self, w0: i64) {
        let cells = self.tape_cells((w0 - NQ as i64) * WIN, w0 * WIN);
        for w in w0 - NQ as i64..w0 {
            let c = cells.get(&w).map(Vec::as_slice).unwrap_or(&[]);
            push(&mut self.hs, c.iter().map(|x| x.2).fold(None, |m: Option<f64>, x| Some(m.map_or(x, |m| m.max(x)))).unwrap_or(0.0));
            push(&mut self.hb, c.iter().map(|x| x.1).fold(None, |m: Option<f64>, x| Some(m.map_or(x, |m| m.max(x)))).unwrap_or(0.0));
        }
    }

    /// 한 바퀴(`main` while 몸통 :177-226). 반환 = 쓴 state 글자.
    pub fn step(&mut self, now: f64) -> String {
        let cur = self.dd.join(hour_name(now));
        if Some(&cur) != self.path.as_ref() && cur.exists() {
            if let Some(mut fh) = self.fh.take() {
                let mut rest = std::mem::take(&mut self.buf);
                let _ = fh.read_to_end(&mut rest);
                for ln in rest.split(|b| *b == b'\n') {
                    if !ln.is_empty() {
                        if let Some(b) = self.book.as_mut() {
                            b.feed(ln);
                        }
                    }
                }
            }
            if let Some(mut b) = self.book.take() {
                b.close();
                self.pend.append(&mut b.done);
            }
            self.fh = std::fs::File::open(&cur).ok();
            (self.path, self.buf) = (Some(cur.clone()), vec![]);
            println!("파일 {}", cur.file_name().unwrap().to_string_lossy());
        }
        if let Some(fh) = self.fh.as_mut() {
            let _ = fh.read_to_end(&mut self.buf);
            let cut = self.buf.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
            let lines: Vec<Vec<u8>> = self.buf[..cut].split(|b| *b == b'\n').map(<[u8]>::to_vec).collect();
            self.buf.drain(..cut);
            let lines = &lines[..lines.len().saturating_sub(1)]; // split 의 마지막 빈 조각(줄 끝) 뺌
            for ln in lines {
                if self.book.is_none() {
                    let snap = serde_json::from_slice::<Value>(ln).ok().and_then(|f| f.get("_snapshot").filter(|s| crate::py_truthy(s)).cloned());
                    match snap.and_then(|s| Book::new(&s, self.tick, self.per)) {
                        Some(b) => self.book = Some(b),
                        None => {
                            println!("{} 스냅샷 없음 -- 이 시 건너뜀", self.path.as_ref().unwrap().display());
                            self.fh = None;
                            break;
                        }
                    }
                } else {
                    self.book.as_mut().unwrap().feed(ln);
                }
            }
        }
        if let Some(b) = self.book.as_mut() {
            self.pend.append(&mut b.done);
        }
        let lw = self.last_w;
        self.pend.retain(|w| lw.is_none_or(|l| w.w > l));
        let nready = self.pend.iter().take_while(|w| ((w.w + 1) * WIN + SETTLE) as f64 <= now).count(); // 창 순서대로라 앞부분만 준비된다
        if nready > 0 {
            if self.last_w.is_none() {
                self.warm(self.pend[0].w);
            }
            let cells = self.tape_cells(self.pend[0].w * WIN, (self.pend[nready - 1].w + 1) * WIN);
            let coin_l = self.coin_l();
            for w in self.pend.drain(..nready).collect::<Vec<_>>() {
                let (mut sb, mut ss) = (vec![0f32; NB], vec![0f32; NB]);
                for (b, bq, sq) in cells.get(&w.w).map(Vec::as_slice).unwrap_or(&[]) {
                    let c = b - w.base + K;
                    if (0..=2 * K).contains(&c) {
                        sb[c as usize] += *bq as f32;
                        ss[c as usize] += *sq as f32;
                    }
                }
                let hs: Vec<f64> = self.hs.iter().copied().collect();
                let hb: Vec<f64> = self.hb.iter().copied().collect();
                let qs = if hs.len() >= NMIN { np_quantile(&hs, 0.95) } else { f64::NAN };
                let qb = if hb.len() >= NMIN { np_quantile(&hb, 0.95) } else { f64::NAN };
                for e in window_events(&w, &sb, &ss, qs, qb, self.tick, self.per, self.dp) {
                    let key = (e["t_end"].as_i64().unwrap(), e["side"].as_i64().unwrap());
                    if !self.seen.contains(&key) {
                        use std::io::Write;
                        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(self.out.join(format!("{coin_l}_events.jsonl"))).unwrap();
                        let _ = writeln!(f, "{}", crate::py_dumps_default(&e));
                        self.seen.insert(key);
                        self.n += (key.0 >= FWD0) as i64;
                        println!("아이스버그 {e}");
                    }
                }
                push(&mut self.hs, ss.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64);
                push(&mut self.hb, sb.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64);
                self.last_w = Some(w.w);
            }
        }
        let st = crate::py_dumps_default(&json!({
            "updated_at": now, "file": self.path.as_ref().map(|p| p.file_name().unwrap().to_string_lossy().to_string()),
            "valid": self.book.as_ref().map_or(0, |b| b.valid), "last_window_end": self.last_w.map(|l| (l + 1) * WIN), "n_forward": self.n,
            "target": if self.sym == "ETHUSDT" { Value::from(150) } else { Value::Null }, "fwd0": FWD0,
        }));
        let p = self.out.join(format!("{}_state.json", self.coin_l()));
        let tmp = p.with_extension("tmp");
        if std::fs::write(&tmp, &st).and_then(|_| std::fs::rename(&tmp, &p)).is_err() {
            eprintln!("state 쓰기 실패 {}", p.display());
        }
        st
    }
}

fn push(d: &mut VecDeque<f64>, v: f64) {
    if d.len() == NQ {
        d.pop_front();
    }
    d.push_back(v);
}

/// 정답지 입력 먹이기: 시계 t0 부터 2초씩 `step`(원본 time.sleep(2)) -- t_end 넘으면 끝. 반환 = 바퀴마다 state 글자.
pub fn replay(sym: &str, dd: &Path, tape: &Path, out: &Path, t0: f64, t_end: f64) -> Vec<String> {
    let mut ic = Iceberg::new(sym, dd, tape, out).unwrap();
    let (mut t, mut st) = (t0, vec![]);
    while t <= t_end {
        st.push(ic.step(t));
        t += 2.0;
    }
    st
}
