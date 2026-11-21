//! ETH 호가 깊이 10열 실시간 워커(`scripts/live_bookdepth_feat_20261028.py`, 줄 번호는 그 파일) -- 방향 예측 «book» 블록(fwd_dirshort 전진 검정의 입력)을
//! 서버 diff 전체북(`ob5m::replay` 그대로)에서 만든다 → `data/live/bookdepth/ethusdt.sqlite` bars.
//! 스냅숏 = 매 분 :01·:31 초(그 초에 diff 가 없으면 그 반분의 첫 초) · 봉 τ 값 = 봉 끝 1초 전까지 마지막 스냅숏 · d_age = 그 나이 + 1초.
//! 10열: dimb1·2·5 · dimbk_d1h(12봉 전과 차) · depth1_z(log(±1% 매수+매도) − 직전 288봉 평균, 최소 144) · bid_near · ask_near · d_age.
//! warm = 사슬 (재)시작 뒤 12시간 -- 먼 칸이 덜 찼다 → 피쳐 비움(원값·d_age 는 남김). 온전한 봉은 다시 쓰지 않는다.
//! 🔴`cum` 의 `arr[lo:hi] @ (가격)` 은 원본이 BLAS 내적(서버 OpenBLAS · 개발기 MKL)이라 끝자리가 기계·BLAS 마다 다르다 -- 여기는 차례 곱합
//!   (ponytail: OpenBLAS ddot 커널 순서를 흉내 내지 않았다 -- 서버와 끝자리까지 맞추려면 그 커널(Haswell AVX2 4갈래)을 옮겨야 한다).
//!   시험은 dm·dp 와 거기서 나온 열을 상대 1e-12 로 본다. np.log·np.mean 은 libm·짝 합(ob5m 과 같다).
use crate::ob5m::{np_mean, replay, Source};
use rusqlite::Connection;
use std::path::Path;

pub const KS: [f64; 3] = [1.0, 2.0, 5.0];
pub const BAR_S: i64 = 300;
pub const WARM_S: i64 = 12 * 3600; // 10-03 차가운 재생 -- ±5% 매도 비 1h .89 · 6~12h .976 · 12h+ .995
pub const HIST: i64 = 300;
pub const RAW: [&str; 6] = ["dm1", "dm2", "dm5", "dp1", "dp2", "dp5"];
pub const FEAT: [&str; 10] = ["dimb1", "dimb2", "dimb5", "dimb1_d1h", "dimb2_d1h", "dimb5_d1h", "depth1_z", "bid_near", "ask_near", "d_age"];

/// `cum`(:36-44): 중간가 기준 ±k% 안 누적 명목 $(인덱스 = 가격 × 틱).
pub fn cum(bid: &[f64], ask: &[f64], bb: i64, ba: i64, tick: f64) -> [f64; 6] {
    let mid = (bb + ba) as f64 / 2.0;
    let mut out = [0.0; 6];
    let spans: Vec<(&[f64], i64, i64)> = KS.iter().map(|k| (bid, (mid * (1.0 - k / 100.0)).ceil() as i64, bb + 1))
        .chain(KS.iter().map(|k| (ask, ba, (mid * (1.0 + k / 100.0)) as i64 + 1))).collect();
    for (o, (arr, lo, hi)) in out.iter_mut().zip(spans) {
        let hi = hi.min(arr.len() as i64); // IDX_MAX 넘는 칸은 replay 가 버린다 -- 길이 맞춤(리뷰 10-28)
        let mut s = 0.0;
        for i in lo.max(0)..hi {
            s += arr[i as usize] * (i as f64 / tick);
        }
        *o = s;
    }
    out
}

/// 한 봉(파이썬 dict 순서 그대로). None = NULL.
pub type Row = Vec<(&'static str, Option<f64>)>;

/// `Bars`(:47-84). hist = 최근 봉 원값(봉 시작 초 → RAW, 삽입 순서 = 파이썬 dict) -- 사슬 재시작에도 유지.
pub struct Bars<'a> {
    pub out: Box<dyn FnMut(i64, &Row) + 'a>,
    pub hist: Vec<(i64, [f64; 6])>,
    tick: f64,
    chain0: Option<i64>,
    snap: Option<(i64, [f64; 6])>,
    slot: Option<i64>,
    bar: Option<i64>,
}

impl<'a> Bars<'a> {
    pub fn new(out: Box<dyn FnMut(i64, &Row) + 'a>, tick: f64) -> Self {
        Self { out, hist: vec![], tick, chain0: None, snap: None, slot: None, bar: None }
    }

    /// 사슬 (재)시작. 끊긴 순간 진행 중 봉이 마지막 반분 스냅숏(:04:31 이후)까지 받았으면 완성 봉이므로 먼저 닫는다(리뷰 10-28).
    pub fn chain(&mut self) {
        if let (Some(b), Some(sn)) = (self.bar, self.snap) {
            if sn.0 >= b + BAR_S - 29 {
                self.close(b);
            }
        }
        (self.chain0, self.snap, self.slot, self.bar) = (None, None, None, None);
    }

    pub fn on_sec(&mut self, t: i64, bid: &[f64], ask: &[f64], bb: i64, ba: i64) {
        let s = t.div_euclid(1000);
        if self.chain0.is_none() {
            self.chain0 = Some(s);
        }
        let bar = s - s.rem_euclid(BAR_S);
        if let Some(b) = self.bar {
            if bar > b {
                self.close(b); // 다음 봉 첫 초 -- 스냅숏보다 먼저(이 초의 스냅숏은 다음 봉 것)
            }
        }
        self.bar = Some(bar);
        if s.rem_euclid(30) >= 1 && Some(s.div_euclid(30)) != self.slot {
            (self.slot, self.snap) = (Some(s.div_euclid(30)), Some((s, cum(bid, ask, bb, ba, self.tick))));
        }
    }

    fn close(&mut self, bar: i64) {
        let Some((s, v)) = self.snap else { return };
        let (cut, c0) = (bar + BAR_S - 1, self.chain0.unwrap());
        // warm 사슬 값은 DB 에서 불러온 온전한 원값을 덮지 않는다
        if cut - c0 >= WARM_S || !self.hist.iter().any(|h| h.0 == bar) {
            match self.hist.iter_mut().find(|h| h.0 == bar) {
                Some(h) => h.1 = v,
                None => self.hist.push((bar, v)),
            }
        }
        self.hist.retain(|h| h.0 > bar - HIST * BAR_S);
        let r = row(bar, &v, cut - s + 1, &self.hist, cut - c0);
        (self.out)(bar * 1000, &r);
    }
}

/// `row`(:87-98)
pub fn row(bar: i64, v: &[f64; 6], age: i64, hist: &[(i64, [f64; 6])], chain_age: i64) -> Row {
    let imb = |w: &[f64; 6]| -> [f64; 3] {
        let mut o = [f64::NAN; 3];
        for (i, x) in o.iter_mut().enumerate() {
            if w[i] + w[i + 3] > 0.0 {
                *x = (w[i] - w[i + 3]) / (w[i] + w[i + 3]);
            }
        }
        o
    };
    let now = imb(v);
    let ago = hist.iter().find(|h| h.0 == bar - 12 * BAR_S).map_or([f64::NAN; 3], |h| imb(&h.1));
    let lg: Vec<f64> = hist.iter().filter(|(k, w)| bar - 287 * BAR_S <= *k && *k <= bar && w[0] + w[3] > 0.0).map(|(_, w)| (w[0] + w[3]).ln()).collect();
    let z = if lg.len() >= 144 && v[0] + v[3] > 0.0 { (v[0] + v[3]).ln() - np_mean(&lg) } else { f64::NAN };
    let warm = chain_age < WARM_S;
    let mut r: Row = RAW.iter().zip(v).map(|(k, x)| (*k, Some(*x))).collect();
    let feat = [now[0], now[1], now[2], now[0] - ago[0], now[1] - ago[1], now[2] - ago[2], z,
                if v[2] > 0.0 { v[0] / v[2] } else { f64::NAN }, if v[5] > 0.0 { v[3] / v[5] } else { f64::NAN }, age as f64];
    for (k, x) in FEAT.iter().zip(feat) {
        // warm = 먼 칸 덜 참 → 피쳐 비움(원값·d_age 는 진단용으로 남김)
        r.push((k, if warm && *k != "d_age" { None } else { Some(x).filter(|x| x.is_finite()) }));
    }
    r.push(("chain_age_s", Some(chain_age as f64)));
    r.push(("warm", Some(warm as i64 as f64)));
    r
}

pub fn db_open(path: &Path) -> rusqlite::Result<Connection> {
    if let Some(d) = path.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let con = Connection::open(path)?;
    con.busy_timeout(std::time::Duration::from_secs(30))?;
    con.query_row("PRAGMA journal_mode=WAL", [], |_| Ok(()))?;
    con.execute_batch("PRAGMA synchronous = NORMAL")?;
    let cols: Vec<String> = RAW.iter().chain(FEAT.iter()).chain(["chain_age_s", "warm"].iter()).map(|c| format!("{c} REAL")).collect();
    con.execute(&format!("CREATE TABLE IF NOT EXISTS bars(t INTEGER PRIMARY KEY, {}, written_ms INTEGER)", cols.join(", ")), [])?;
    Ok(con)
}

/// `db_write`(:110-117): 온전한 봉은 다시 쓰지 않는다(라이브에 쓰인 값이 사후에 바뀌면 재현·감사가 깨진다).
pub fn db_write(con: &Connection, t: i64, r: &Row, written_ms: i64) -> rusqlite::Result<bool> {
    let old: Option<Option<f64>> = con.query_row("SELECT warm FROM bars WHERE t = ?", [t], |x| x.get(0)).ok();
    if old.is_some_and(|w| w == Some(0.0)) {
        return Ok(false);
    }
    let cols: Vec<&str> = std::iter::once("t").chain(r.iter().map(|x| x.0)).chain(std::iter::once("written_ms")).collect();
    let mut vals: Vec<rusqlite::types::Value> = vec![rusqlite::types::Value::Integer(t)];
    for (k, x) in r {
        // 파이썬은 warm 을 int 로 묶는다(REAL 칸 친화성으로 같은 값)
        vals.push(match (k, x) {
            (&"warm", Some(w)) => rusqlite::types::Value::Integer(*w as i64),
            (_, Some(f)) => rusqlite::types::Value::Real(*f),
            _ => rusqlite::types::Value::Null,
        });
    }
    vals.push(rusqlite::types::Value::Integer(written_ms));
    con.execute(&format!("INSERT OR REPLACE INTO bars({}) VALUES ({})", cols.join(", "), vec!["?"; cols.len()].join(", ")), rusqlite::params_from_iter(vals))?;
    Ok(true)
}

/// `db_hist`(:120-124): 재시작 때 최근 온전한 봉 원값을 hist 로(depth1_z 24h 창 · d1h 12봉 전). 순서 = 최근 먼저(원본 ORDER BY t DESC).
pub fn db_hist(con: &Connection, bars: &mut Bars) -> rusqlite::Result<()> {
    let mut st = con.prepare(&format!("SELECT t, {} FROM bars WHERE warm = 0 ORDER BY t DESC LIMIT {HIST}", RAW.join(", ")))?;
    let rows = st.query_map([], |r| {
        let mut v = [None; 6];
        for (j, x) in v.iter_mut().enumerate() {
            *x = r.get::<_, Option<f64>>(j + 1)?;
        }
        Ok((r.get::<_, i64>(0)?, v))
    })?;
    for x in rows {
        let (t, v) = x?;
        if v.iter().all(Option::is_some) {
            bars.hist.push((t.div_euclid(1000), v.map(Option::unwrap)));
        }
    }
    Ok(())
}

/// `main()`(:127-147): 먼 칸이 찬 사슬로 마지막 온전한 봉을 잇게 WARM_S + 1시간 앞부터(오래 멈췄으면 최근 26시간).
pub fn run(sym: &str, dd: &Path, db: &Path, now: &dyn Fn() -> f64, sleep: &dyn Fn(f64) -> bool, seen: &dyn Fn(i64)) -> rusqlite::Result<()> {
    let (tick, _, ks, idx_max) = crate::ob5m::coin(sym).expect("코인");
    let con = db_open(db)?;
    let last: Option<f64> = con.query_row("SELECT max(t) FROM bars WHERE warm = 0", [], |r| r.get(0))?;
    let now_h = (now() as i64).div_euclid(3600);
    let h0 = (now_h - 26).max(last.map_or(now_h - 26, |l| (l as i64).div_euclid(3_600_000) - 13));
    let out = |t: i64, r: &Row| {
        if let Err(e) = db_write(&con, t, r, (now() * 1000.0) as i64) {
            eprintln!("봉 쓰기 실패 {e}");
        }
    };
    let bars = std::cell::RefCell::new(Bars::new(Box::new(out), tick));
    db_hist(&con, &mut bars.borrow_mut())?;
    eprintln!("시작 · 마지막 온전한 봉 {last:?} · {} 파일부터 · hist {}봉", crate::ob5m::hour_name(h0), bars.borrow().hist.len());
    let mut src = Source::new(dd, h0, Box::new(now), Box::new(sleep));
    loop {
        bars.borrow_mut().chain();
        src.begin();
        let st = {
            let mut on_sec = |t: i64, b: &[f64], a: &[f64], bb: i64, ba: i64| {
                seen(t);
                bars.borrow_mut().on_sec(t, b, a, bb, ba);
            };
            replay(&mut src, tick, ks, idx_max, &mut on_sec, &mut |_| {})
        };
        if src.stopped {
            return Ok(());
        }
        eprintln!("사슬 끝 {st} · {} {}줄 -- 스냅숏에서 다시", crate::ob5m::hour_name(src.hour), src.n);
        src.restart(st);
    }
}
