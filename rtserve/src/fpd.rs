//! 일봉·4시간 풋프린트(2026-11-22 옮김) -- 원본 dashboard/footprint_daily.py(DailyFootprint · 654줄) + server.py 손잡이
//! (`_fpd_lake` 3714~3720 · `_fpd_rows` 3722~3725 · `_fpd_asset` 3730 · `_iso_day` 3737 · `_cols_since` 3746 · `_iso_h4` 3885 · 엔드포인트 3861~3912).
//! 원천: 소급본 `<hist>/<SYM>_{days,cells}.parquet` · lake `binance/{tape,oi_1s,liquidations}/coin=<COIN>/date=*/part.parquet`(DuckDB read_parquet ·
//! 원본 data_store.read 와 같은 SQL) · hot SQLite(trade_tape_1s · oi_1s · liquidations -- 원본 read_rows 와 같은 SQL).
//! 값 규칙(원본과 같게): np.sum = rtcollect::ob5m::np_sum(numpy 짝 합) · np.average = 곱의 np.sum ÷ 무게의 np.sum ·
//! pandas groupby sum = Kahan 보정 합(pandas 2.2 group_sum · 행 차례) · np.round(x, 3) = rint(x×1000)÷1000 · 파이썬 round = 정확한 십진 짝수 반올림.
//! 모듈 캐시 셋(원본 모듈 전역 · 코인마다 따로): `_recent`(닫힌 날 한 번) · `_today`(30초) · `_ctx_lake`(1시간) · `_h4`(닫힘 = 정착 2분 뒤 굳음 ·
//! 형성 = 30초마다 다시 · 빈칸 = «없음»으로 굳히고 10분마다 칸 끝 + 47시간 안 것만 다시 봄) · 소급본(파일 mtime 이 바뀌면 다시 읽음).
//! 일치 정답지: rtserve/tools/gen_fpd_fixture.py(원본 DailyFootprint 를 합성 원천·고정 시계로) → tests/fpd_parity.rs.
//! 라우팅(나중에 붙임 -- lib.rs·api*.rs 는 이 파일이 건드리지 않는다): 응답 = 아래 `*_payload` 의 값을 원본처럼 swr(30초 · 낡은 값 600초) 뒤 `cols_since`.
use rusqlite::{Connection, OpenFlags};
use serde_json::{json, Map, Value as V};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// (코인, 선물 심볼, 테이프 칸, 일봉 칸 CELL, 응답 가격 자릿수) -- footprint_daily.py ASSETS(30~34).
pub const ASSETS: &[(&str, &str, f64, f64, usize)] = &[
    ("eth", "ETHUSDT", 0.1, 1.0, 4),
    ("xrp", "XRPUSDT", 0.0001, 0.0005, 6),
    ("btc", "BTCUSDT", 1.0, 30.0, 2),
    ("sndk", "SNDKUSDT", 0.05, 0.5, 2),
    ("soxl", "SOXLUSDT", 0.01, 0.03, 2),
    ("skhynix", "SKHYNIXUSDT", 0.1, 0.4, 2),
    ("mu", "MUUSDT", 0.05, 0.3, 2),
];
pub const MAX_CELL_DAYS: usize = 400;
pub const TODAY_TTL: f64 = 30.0;
pub const CTX_DAYS: i64 = 45;
pub const CTX_LAKE_TTL: f64 = 3600.0;
pub const H4_SEC: i64 = 14400;
pub const H4_DAYS: i64 = 7;
pub const H4_SETTLE: i64 = 120;
pub const H4_BACKFILL: i64 = 47 * 3600;
pub const H4_GAP_TTL: f64 = 600.0;
const DAY: i64 = 86400;

// ── 날짜(UTC 일 번호 = 1970-01-01 부터 일 수) ──
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}
fn civil(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { yoe + era * 400 + 1 } else { yoe + era * 400 }, m, d)
}
fn ds(day: i64) -> String {
    let (y, m, d) = civil(day);
    format!("{y:04}-{m:02}-{d:02}")
}
fn day_of_sec(s: i64) -> i64 {
    s.div_euclid(DAY)
}
fn parse_day(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let (y, m, d): (i64, i64, i64) = (s[..4].parse().ok()?, s[5..7].parse().ok()?, s[8..].parse().ok()?);
    let dim = [31, if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    ((1..=12).contains(&m) && d >= 1 && d <= dim[(m - 1) as usize] && y >= 1).then(|| days_from_civil(y, m, d))
}
/// footprint_daily.h4_key: 칸 시작 초 → 'YYYY-MM-DDTHH'.
pub fn h4_key(ts: i64) -> String {
    let t = ts.div_euclid(H4_SEC) * H4_SEC;
    format!("{}T{:02}", ds(day_of_sec(t)), t.rem_euclid(DAY) / 3600)
}
/// footprint_daily._h4_sec(strptime 은 자리 수가 짧아도 받지만 그런 키는 h4_key 로 되돌리면 달라 _iso_h4 가 막는다 -- 여기는 정확한 꼴만).
pub fn h4_sec(key: &str) -> Option<i64> {
    let (d, h) = key.split_once('T')?;
    let h: i64 = (h.len() == 2).then(|| h.parse().ok()).flatten()?;
    (0..24).contains(&h).then_some(())?;
    Some(parse_day(d)? * DAY + h * 3600)
}

// ── 파이썬·numpy·pandas 수 규칙 ──
fn py_round(x: f64, nd: usize) -> f64 {
    rtserve_round(x, nd)
}
#[inline]
fn rtserve_round(x: f64, nd: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let r: f64 = format!("{x:.nd$}").parse().unwrap();
    if r == 0.0 { 0.0f64.copysign(x) } else { r }
}
/// np.round(x, 3)(실수 배열): x × 1000 → rint(짝수) → ÷ 1000.
fn np_round3(x: f64) -> f64 {
    (x * 1000.0).round_ties_even() / 1000.0
}
fn np_sum(a: &[f64]) -> f64 {
    rtcollect::ob5m::np_sum(a)
}
/// pandas groupby sum(Kahan · 행 차례) 한 칸의 누적.
#[derive(Clone, Copy, Default)]
struct Kahan {
    s: f64,
    c: f64,
}
impl Kahan {
    fn add(&mut self, v: f64) {
        let y = v - self.c;
        let t = self.s + y;
        self.c = t - self.s - y;
        self.s = t;
    }
}

/// 한 칸(일·4시간)의 요약. 4시간 칸은 맥락(oi·ll·ls)이 붙는다.
#[derive(Clone, Debug, PartialEq)]
pub struct Summ {
    pub open: f64,
    pub close: f64,
    pub high: f64,
    pub low: f64,
    pub buy_qty: f64,
    pub sell_qty: f64,
    pub buy_quote: f64,
    pub sell_quote: f64,
    pub oi: Option<f64>,
    pub ll: Option<f64>,
    pub ls: Option<f64>,
}
/// CELL 칸(칸 번호 오름차순, 매수, 매도).
pub type Cells = (Vec<i64>, Vec<f64>, Vec<f64>);
pub(crate) type Bucket = (Summ, Cells);

/// 테이프 행 묶음(초, 테이프 칸, 매수, 매도) -- 원본 DataFrame 한 묶음의 행 차례 그대로.
#[derive(Default, Clone)]
pub(crate) struct Tape {
    pub(crate) sec: Vec<i64>,
    pub(crate) pbin: Vec<i64>,
    pub(crate) buy: Vec<f64>,
    pub(crate) sell: Vec<f64>,
}
impl Tape {
    fn push(&mut self, s: i64, p: i64, b: f64, q: f64) {
        self.sec.push(s);
        self.pbin.push(p);
        self.buy.push(b);
        self.sell.push(q);
    }
    fn extend(&mut self, o: &Tape) {
        self.sec.extend(&o.sec);
        self.pbin.extend(&o.pbin);
        self.buy.extend(&o.buy);
        self.sell.extend(&o.sell);
    }
}

pub struct Spec {
    pub asset: String,
    pub symbol: String,
    pub coin: String,
    pub hot_sym: String,
    pub tape_bucket: f64,
    pub cell: f64,
    pub dp: usize,
    pub tape_per_cell: i64,
}
impl Spec {
    pub fn of(asset: &str) -> Option<Spec> {
        let &(a, sym, tb, cell, dp) = ASSETS.iter().find(|x| x.0 == asset)?;
        Some(Spec {
            asset: a.into(),
            symbol: sym.into(),
            coin: sym[..sym.len() - 4].into(),
            hot_sym: sym.to_lowercase(),
            tape_bucket: tb,
            cell,
            dp,
            tape_per_cell: (cell / tb).round_ties_even() as i64,
        })
    }
}

/// footprint_daily.day_from_tape: 한 묶음의 행 → (요약, CELL 칸). 시가·종가 = 첫·끝 초 VWAP · 고저 = 체결 있는 칸 극값.
pub(crate) fn day_from_tape(sp: &Spec, t: &Tape) -> Bucket {
    let n = t.sec.len();
    let mut o: Vec<usize> = (0..n).collect();
    o.sort_by_key(|&i| t.sec[i]); // stable(원본 argsort kind="stable")
    let sec: Vec<i64> = o.iter().map(|&i| t.sec[i]).collect();
    let px: Vec<f64> = o.iter().map(|&i| t.pbin[i] as f64 * sp.tape_bucket).collect();
    let buy: Vec<f64> = o.iter().map(|&i| t.buy[i]).collect();
    let sell: Vec<f64> = o.iter().map(|&i| t.sell[i]).collect();
    let q: Vec<f64> = (0..n).map(|i| t.buy[o[i]] + t.sell[o[i]]).collect();
    let vw = |want: i64| {
        let idx: Vec<usize> = (0..n).filter(|&i| sec[i] == want).collect();
        let qm: Vec<f64> = idx.iter().map(|&i| q[i]).collect();
        if np_sum(&qm) > 0.0 {
            let pq: Vec<f64> = idx.iter().map(|&i| px[i] * q[i]).collect();
            np_sum(&pq) / np_sum(&qm)
        } else {
            px[idx[0]]
        }
    };
    let has: Vec<f64> = (0..n).filter(|&i| q[i] > 0.0).map(|i| px[i]).collect();
    let bq: Vec<f64> = (0..n).map(|i| buy[i] * px[i]).collect();
    let sq: Vec<f64> = (0..n).map(|i| sell[i] * px[i]).collect();
    let summ = Summ {
        open: vw(sec[0]),
        close: vw(sec[n - 1]),
        high: has.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        low: has.iter().copied().fold(f64::INFINITY, f64::min),
        buy_qty: np_sum(&buy),
        sell_qty: np_sum(&sell),
        buy_quote: np_sum(&bq),
        sell_quote: np_sum(&sq),
        oi: None,
        ll: None,
        ls: None,
    };
    // groupby("b", sort=True).sum() -- Kahan · 행 차례(정렬된 행)
    let mut g: BTreeMap<i64, (Kahan, Kahan)> = BTreeMap::new();
    for i in 0..n {
        let b = t.pbin[o[i]].div_euclid(sp.tape_per_cell);
        let e = g.entry(b).or_default();
        e.0.add(buy[i]);
        e.1.add(sell[i]);
    }
    let cells = (g.keys().copied().collect(), g.values().map(|v| v.0.s).collect(), g.values().map(|v| v.1.s).collect());
    (summ, cells)
}

/// footprint_daily.rebin: CELL 칸 → row 칸. [첫 칸 하한($), [매수…], [매도…]] (빈 칸 0 채움 · np.add.at 차례 합 · np.round 3).
pub(crate) fn rebin(sp: &Spec, c: &Cells, row: i64) -> V {
    if c.0.is_empty() {
        return json!([0, [], []]);
    }
    let k: Vec<i64> = c.0.iter().map(|b| b.div_euclid(row)).collect();
    let lo = *k.iter().min().unwrap();
    let n = (k.iter().max().unwrap() - lo + 1) as usize;
    let (mut bb, mut ss) = (vec![0.0; n], vec![0.0; n]);
    for (i, kk) in k.iter().enumerate() {
        bb[(kk - lo) as usize] += c.1[i];
        ss[(kk - lo) as usize] += c.2[i];
    }
    json!([py_round((lo * row) as f64 * sp.cell, sp.dp), bb.iter().map(|&x| np_round3(x)).collect::<Vec<_>>(), ss.iter().map(|&x| np_round3(x)).collect::<Vec<_>>()])
}

pub const COLS: [&str; 14] = ["d", "o", "h", "l", "c", "bq", "sq", "bv", "sv", "oi", "ll", "ls", "src", "part"];

struct Row {
    s: Summ,
    src: &'static str,
}

/// footprint_daily.build_days: 일(또는 4시간) 요약 열 묶음. 소급본 날이 이긴다.
fn build_days(hist: Option<&[(String, Summ)]>, recent: &BTreeMap<String, Summ>, oi: &HashMap<String, f64>, liq: &HashMap<String, [f64; 2]>, liq_from: Option<i64>, today: &str) -> V {
    let mut rows: BTreeMap<String, Row> = BTreeMap::new();
    for (d, s) in hist.unwrap_or(&[]) {
        rows.insert(d.clone(), Row { s: s.clone(), src: "a" });
    }
    for (d, s) in recent {
        rows.entry(d.clone()).or_insert_with(|| Row { s: Summ { oi: None, ..s.clone() }, src: "t" });
    }
    let liq_day0 = liq_from.filter(|&v| v != 0).map(|v| ds(day_of_sec(v.div_euclid(1000))));
    let mut cols: Vec<Vec<V>> = vec![vec![]; COLS.len()];
    let r4 = |x: f64| V::from(py_round(x, 4));
    let opt = |x: Option<f64>| x.map_or(V::Null, r4);
    for (d, r) in rows.iter_mut() {
        if let Some(&v) = oi.get(d) {
            if r.s.oi.is_none() || d == today {
                r.s.oi = Some(v);
            }
        }
        let lq = liq_day0.as_ref().filter(|d0| d.as_str() >= d0.as_str()).map(|_| liq.get(d).copied().unwrap_or([0.0, 0.0]));
        let s = &r.s;
        let vals = [
            V::from(d.clone()), r4(s.open), r4(s.high), r4(s.low), r4(s.close), r4(s.buy_qty), r4(s.sell_qty), r4(s.buy_quote), r4(s.sell_quote),
            opt(s.oi), opt(lq.map(|x| x[0])), opt(lq.map(|x| x[1])), V::from(r.src),
        ];
        for (j, v) in vals.into_iter().enumerate() {
            cols[j].push(v);
        }
        cols[13].push(V::from(if d == today { 1 } else if liq_day0.as_deref() == Some(d.as_str()) { 2 } else { 0 }));
    }
    let mut m = Map::new();
    for (k, v) in COLS.iter().zip(cols) {
        m.insert(k.to_string(), V::Array(v));
    }
    V::Object(m)
}

/// 원천 읽기 자리. 원본의 read_lake / read_hot / read_ctx 주입과 같은 자리.
pub struct Src {
    pub lake: PathBuf,
    pub hist: PathBuf,
    pub hot_tape: PathBuf,
    pub hot_ctx: PathBuf,
}

#[derive(Default)]
struct St {
    hist: Option<((f64, f64), Vec<(String, Summ)>, HashMap<String, Cells>)>,
    recent_days: BTreeMap<String, Summ>,
    recent_cells: BTreeMap<String, Cells>,
    today: Option<(f64, String, Summ, Cells)>,
    ctx_lake: Option<(f64, (i64, i64), CtxLake)>,
    h4_closed: BTreeMap<String, Option<Bucket>>,
    h4_live: Vec<(String, Bucket)>,
    h4_at: f64,
    h4_gap_at: f64,
}
type CtxLake = (HashMap<String, f64>, BTreeMap<String, [f64; 2]>, Option<i64>);

pub type Res<T> = Result<T, String>;

/// 코인 하나(원본 for_asset 모듈 사본 + DailyFootprint 인스턴스). 캐시는 이 안에.
pub struct Fpd {
    pub sp: Spec,
    pub src: Src,
    st: Mutex<St>,
}

fn mtime(p: &Path) -> Option<f64> {
    let m = std::fs::metadata(p).ok()?.modified().ok()?;
    Some(m.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs_f64())
}

impl Fpd {
    pub fn new(asset: &str, src: Src) -> Option<Fpd> {
        Some(Fpd { sp: Spec::of(asset)?, src, st: Mutex::default() })
    }

    // ── 읽기 ──
    /// data_store.read 와 같은 SQL(호출마다 새 메모리 연결 -- 원본 duckdb.connect()) · 없거나 실패 = None(server.py _fpd_lake).
    fn lake_rows<T>(&self, stream: &str, start: Option<i64>, end: Option<i64>, cols: &str, row: impl Fn(&duckdb::Row) -> duckdb::Result<T>) -> Option<Vec<T>> {
        let con = duckdb::Connection::open_in_memory().ok()?;
        let pat = self.src.lake.join("binance").join(stream).join(format!("coin={}", self.sp.coin)).join("date=*").join("part.parquet");
        let mut w = vec![];
        if let Some(s) = start {
            w.push(format!("date >= DATE '{}'", ds(s)));
        }
        if let Some(e) = end {
            w.push(format!("date < DATE '{}'", ds(e)));
        }
        let q = format!(
            "select {cols} from read_parquet('{}', hive_partitioning = true, union_by_name = true){}",
            pat.to_string_lossy(),
            if w.is_empty() { String::new() } else { format!(" where {}", w.join(" and ")) }
        );
        let mut st = con.prepare(&q).ok()?;
        let rows = st.query_map([], |r| row(r)).ok()?;
        rows.collect::<Result<Vec<T>, _>>().ok()
    }
    /// read_rows(.sqlite) -- 파일이 없으면 빈 결과(«없음») · 못 읽으면 오류(원본 _fpd_rows: 굳지 않게 올린다).
    fn hot<T>(&self, db: &Path, sql: &str, params: &[&dyn rusqlite::ToSql], row: impl Fn(&rusqlite::Row) -> rusqlite::Result<T>) -> Res<Vec<T>> {
        if !db.exists() {
            return Ok(vec![]);
        }
        let con = Connection::open_with_flags(db, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI).map_err(|e| e.to_string())?;
        let mut st = con.prepare(sql).map_err(|e| e.to_string())?;
        let rows = st.query_map(params, row).map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<T>, _>>().map_err(|e| e.to_string())
    }

    /// load_hist: 소급본(없으면 None). 두 파일 mtime 이 바뀌면 다시 읽는다.
    fn load_hist(&self, st: &mut St) -> Res<()> {
        let fd = self.src.hist.join(format!("{}_days.parquet", self.sp.symbol));
        let fc = self.src.hist.join(format!("{}_cells.parquet", self.sp.symbol));
        let (Some(a), Some(b)) = (mtime(&fd), mtime(&fc)) else {
            st.hist = None;
            return Ok(());
        };
        if st.hist.as_ref().is_some_and(|h| h.0 == (a, b)) {
            return Ok(());
        }
        let con = duckdb::Connection::open_in_memory().map_err(|e| e.to_string())?;
        let q = format!(
            "select substr(cast(day as varchar), 1, 10), cast(open as double), cast(high as double), cast(low as double), cast(close as double), cast(buy_qty as double), cast(sell_qty as double), cast(buy_quote as double), cast(sell_quote as double), cast(oi as double) from read_parquet('{}')",
            fd.to_string_lossy()
        );
        let mut s = con.prepare(&q).map_err(|e| e.to_string())?;
        let days: Vec<(String, Summ)> = s
            .query_map([], |r| {
                let oi: Option<f64> = r.get(9)?;
                Ok((r.get(0)?, Summ { open: r.get(1)?, high: r.get(2)?, low: r.get(3)?, close: r.get(4)?, buy_qty: r.get(5)?, sell_qty: r.get(6)?, buy_quote: r.get(7)?, sell_quote: r.get(8)?, oi: oi.filter(|v| !v.is_nan()), ll: None, ls: None }))
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?;
        let q = format!("select substr(cast(day as varchar), 1, 10), cast(bin as bigint), cast(buy as double), cast(sell as double) from read_parquet('{}')", fc.to_string_lossy());
        let mut s = con.prepare(&q).map_err(|e| e.to_string())?;
        let mut cells: HashMap<String, Cells> = HashMap::new();
        for r in s.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, f64>(2)?, r.get::<_, f64>(3)?))).map_err(|e| e.to_string())? {
            let (d, b, x, y) = r.map_err(|e| e.to_string())?;
            let c = cells.entry(d).or_default();
            c.0.push(b);
            c.1.push(x);
            c.2.push(y);
        }
        st.hist = Some(((a, b), days, cells));
        Ok(())
    }
    fn hist_max(st: &St) -> Option<i64> {
        st.hist.as_ref().and_then(|h| h.1.iter().map(|x| x.0.as_str()).max().and_then(parse_day))
    }

    /// tape_rows_by_day: [start, today] 날짜별 테이프 행. lake 날은 lake, 없는 날은 hot(since 초부터). 반환 차례 = lake 날(오름) 뒤 hot 날(오름).
    fn tape_rows_by_day(&self, start: i64, today: i64, since: i64) -> Res<Vec<(String, Tape)>> {
        let mut out: Vec<(String, Tape)> = vec![];
        let lk = self.lake_rows("tape", Some(start), Some(today + 1), "ts_sec, price_bin, buy_qty, sell_qty", |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, f64>(2)?, r.get::<_, f64>(3)?)));
        let mut by: BTreeMap<String, Tape> = BTreeMap::new();
        for (s, p, b, q) in lk.unwrap_or_default() {
            by.entry(ds(day_of_sec(s))).or_default().push(s, p, b, q);
        }
        out.extend(by);
        let have: BTreeSet<String> = out.iter().map(|x| x.0.clone()).collect();
        let missing: Vec<String> = (start..=today).map(ds).filter(|d| !have.contains(d)).collect();
        if let Some(m0) = missing.first() {
            let t0 = since.max(parse_day(m0).unwrap() * DAY);
            let rows = self.hot(&self.src.hot_tape, "SELECT ts_sec, price_bin, buy_qty, sell_qty FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ?", &[&self.sp.hot_sym, &t0], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, f64>(2)?, r.get::<_, f64>(3)?))
            })?;
            let want: BTreeSet<&String> = missing.iter().collect();
            let mut hb: BTreeMap<String, Tape> = BTreeMap::new();
            for (s, p, b, q) in rows {
                let d = ds(day_of_sec(s));
                if want.contains(&d) {
                    hb.entry(d).or_default().push(s, p, b, q);
                }
            }
            out.extend(hb);
        }
        Ok(out)
    }

    /// _recent: 소급본 끝 다음 날 ~ 오늘의 (요약, 칸). 닫힌 날은 한 번 · 오늘은 TODAY_TTL.
    fn recent(&self, st: &mut St, now: f64) -> Res<(BTreeMap<String, Summ>, BTreeMap<String, Cells>)> {
        self.load_hist(st)?;
        let today = day_of_sec(now.floor() as i64);
        let start = Self::hist_max(st).map_or(today - 30, |d| d + 1);
        let closed: Vec<i64> = (start..today).filter(|&d| !st.recent_days.contains_key(&ds(d))).collect();
        let stale = !st.today.as_ref().is_some_and(|t| t.1 == ds(today) && now - t.0 < TODAY_TTL);
        if !closed.is_empty() || stale {
            let got = self.tape_rows_by_day(closed.first().copied().unwrap_or(today), today, 0)?;
            for (d, g) in got {
                let (s, c) = day_from_tape(&self.sp, &g);
                if d == ds(today) {
                    st.today = Some((now, d, s, c));
                } else if d < ds(today) {
                    st.recent_days.insert(d.clone(), s);
                    st.recent_cells.insert(d, c);
                }
            }
        }
        let (mut days, mut cells) = (st.recent_days.clone(), st.recent_cells.clone());
        if let Some((_, d, s, c)) = &st.today {
            if *d == ds(today) {
                days.insert(d.clone(), s.clone());
                cells.insert(d.clone(), c.clone());
            }
        }
        Ok((days, cells))
    }

    /// _ctx_from_lake: lake OI(그날 마지막 값) · 청산(일 합) · 청산 시작 ms. 1시간 캐시(시계 = now -- 원본은 time.time).
    fn ctx_from_lake(&self, st: &mut St, start: i64, today: i64, now: f64) -> CtxLake {
        if let Some((at, key, val)) = &st.ctx_lake {
            if *key == (start, today) && now - at < CTX_LAKE_TTL {
                return val.clone();
            }
        }
        let mut oi = HashMap::new();
        if let Some(mut o) = self.lake_rows("oi_1s", Some(start), Some(today + 1), "ts_ms, open_interest", |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?))) {
            o.sort_by_key(|x| x.0); // ponytail: 원본 sort_values 는 안정 정렬이 아니다 -- 같은 ms 두 행이면 갈릴 수 있다(1초 OI 는 ms 가 겹치지 않는다)
            for (t, v) in o {
                oi.insert(ds(day_of_sec(t.div_euclid(1000))), v); // groupby.last()
            }
        }
        let mut liq: BTreeMap<String, [f64; 2]> = BTreeMap::new();
        let lq = self.lake_rows("liquidations", None, None, "ts_ms, side, usd", |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Option<String>>(1)?, r.get::<_, Option<f64>>(2)?))).unwrap_or_default();
        let liq_from = lq.iter().map(|x| x.0).min();
        let mut g: BTreeMap<(String, String), Kahan> = BTreeMap::new();
        for (t, sd, usd) in &lq {
            if let Some(u) = usd {
                g.entry((ds(day_of_sec(t.div_euclid(1000))), sd.clone().unwrap_or_default())).or_default().add(*u);
            }
        }
        for ((d, sd), k) in g {
            liq.entry(d).or_insert([0.0, 0.0])[if sd == "long" { 0 } else { 1 }] += k.s;
        }
        let val = (oi, liq, liq_from);
        st.ctx_lake = Some((now, (start, today), val.clone()));
        val
    }

    /// oi_liq_by_day: lake 봉인일은 lake, 그 뒤는 hot.
    fn oi_liq_by_day(&self, st: &mut St, start: i64, today: i64, now: f64) -> Res<(HashMap<String, f64>, HashMap<String, [f64; 2]>, Option<i64>)> {
        let (loi, lliq, mut liq_from) = self.ctx_from_lake(st, start, today, now);
        let mut oi = loi.clone();
        let mut liq: HashMap<String, [f64; 2]> = lliq.iter().map(|(d, v)| (d.clone(), *v)).collect();
        let first_hot = loi.keys().filter_map(|d| parse_day(d)).map(|d| d + 1).fold(start, i64::max);
        let since = first_hot * DAY * 1000;
        for (t, v) in self.hot(&self.src.hot_ctx, "SELECT ts_ms, open_interest FROM oi_1s WHERE symbol = ? AND ts_ms >= ? ORDER BY ts_ms", &[&self.sp.hot_sym, &since], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?)))? {
            let d = ds(day_of_sec(t.div_euclid(1000)));
            if !loi.contains_key(&d) {
                oi.insert(d, v);
            }
        }
        for (t, sd, usd) in self.hot(&self.src.hot_ctx, "SELECT ts_ms, side, usd FROM liquidations WHERE symbol = ? AND ts_ms >= ?", &[&self.sp.symbol, &since], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Option<String>>(1)?, r.get::<_, Option<f64>>(2)?)))? {
            let d = ds(day_of_sec(t.div_euclid(1000)));
            if !lliq.contains_key(&d) {
                liq.entry(d).or_insert([0.0, 0.0])[if sd.as_deref() == Some("long") { 0 } else { 1 }] += usd.unwrap_or(0.0);
                liq_from = Some(liq_from.map_or(t, |x| x.min(t)));
            }
        }
        Ok((oi, liq, liq_from))
    }

    /// days_payload(now) -- /api/footprint-daily 본문(?since 전).
    pub fn days_payload(&self, now: f64) -> Res<V> {
        let mut st = self.st.lock().unwrap();
        self.load_hist(&mut st)?;
        let (recent, _) = self.recent(&mut st, now)?;
        let today = day_of_sec(now.floor() as i64);
        let mut start = today - CTX_DAYS;
        if let Some(h) = Self::hist_max(&st) {
            start = start.min(h + 1);
        }
        let (oi, liq, liq_from) = self.oi_liq_by_day(&mut st, start, today, now)?;
        let hist = st.hist.as_ref().map(|h| h.1.clone());
        let cols = build_days(hist.as_deref(), &recent, &oi, &liq, liq_from, &ds(today));
        let until = Self::hist_max(&st).map(ds);
        Ok(json!({"symbol": self.sp.symbol, "tz": "UTC", "cols": cols, "liq_from_ms": liq_from, "hist_until": until, "at": now}))
    }

    /// cells_payload(now, from, to, row) -- /api/footprint-daily/cells 본문. Err("range") = 원본 ValueError(→ 400 range).
    pub fn cells_payload(&self, now: f64, frm: &str, to: &str, row: i64) -> Res<V> {
        let mut st = self.st.lock().unwrap();
        self.load_hist(&mut st)?;
        let (_, recent_cells) = self.recent(&mut st, now)?;
        let (d0, d1) = (parse_day(frm).ok_or("range")?, parse_day(to).ok_or("range")?);
        if d1 < d0 || d1 - d0 > 4000 {
            return Err("range".into());
        }
        let hist_cells = st.hist.as_ref().map(|h| &h.2);
        let have: Vec<String> = (d0..=d1).map(ds).filter(|k| hist_cells.is_some_and(|h| h.contains_key(k)) || recent_cells.contains_key(k)).collect();
        if have.len() > MAX_CELL_DAYS {
            return Err("range".into());
        }
        let mut out = Map::new();
        for k in have {
            let c = hist_cells.and_then(|h| h.get(&k)).unwrap_or_else(|| &recent_cells[&k]);
            out.insert(k, rebin(&self.sp, c, row));
        }
        Ok(json!({"row": py_round(row as f64 * self.sp.cell, self.sp.dp), "cells": out}))
    }

    /// h4_ctx: [t0, now] 4시간 칸별 {oi(칸 마지막), ll·ls(롱·숏 청산 $)}. lake 봉인일 → 그 다음 날부터 hot.
    fn h4_ctx(&self, t0: i64, now: f64) -> Res<HashMap<String, (Option<f64>, Option<f64>, Option<f64>)>> {
        let (d0, d1) = (day_of_sec(t0), day_of_sec(now.floor() as i64) + 1);
        let mut out: HashMap<String, (Option<f64>, Option<f64>, Option<f64>)> = HashMap::new();
        let after_lake = |ms: Option<i64>, floor: i64| ms.map_or(floor, |m| floor.max((day_of_sec(m.div_euclid(1000)) + 1) * DAY * 1000));
        // OI
        let lk: Vec<(i64, f64)> = self.lake_rows("oi_1s", Some(d0), Some(d1), "ts_ms, open_interest", |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?))).unwrap_or_default().into_iter().filter(|x| x.0 >= t0 * 1000).collect();
        let hot = self.hot(&self.src.hot_ctx, "SELECT ts_ms, open_interest FROM oi_1s WHERE symbol = ? AND ts_ms >= ? ORDER BY ts_ms", &[&self.sp.hot_sym, &after_lake(lk.iter().map(|x| x.0).max(), t0 * 1000)], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?)))?;
        let mut oi: Vec<(i64, f64)> = lk.into_iter().chain(hot).collect();
        oi.sort_by_key(|x| x.0); // ponytail: 원본 sort_values 는 안정 정렬 아님(같은 ms 면 갈릴 수 있다)
        for (t, v) in oi {
            out.entry(h4_key(t.div_euclid(1000))).or_default().0 = Some(v);
        }
        // 청산
        let lq: Vec<(i64, Option<String>, Option<f64>)> = self
            .lake_rows("liquidations", Some(d0), Some(d1), "ts_ms, side, usd", |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Option<String>>(1)?, r.get::<_, Option<f64>>(2)?)))
            .unwrap_or_default()
            .into_iter()
            .filter(|x| x.0 >= t0 * 1000)
            .collect();
        let hq = self.hot(&self.src.hot_ctx, "SELECT ts_ms, side, usd FROM liquidations WHERE symbol = ? AND ts_ms >= ?", &[&self.sp.symbol, &after_lake(lq.iter().map(|x| x.0).max(), t0 * 1000)], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, Option<String>>(1)?, r.get::<_, Option<f64>>(2)?))
        })?;
        let mut g: BTreeMap<(i64, String), Kahan> = BTreeMap::new();
        for (t, sd, usd) in lq.into_iter().chain(hq) {
            if let Some(u) = usd {
                g.entry((t.div_euclid(1000).div_euclid(H4_SEC) * H4_SEC, sd.unwrap_or_default())).or_default().add(u);
            }
        }
        for ((k, sd), v) in g {
            let e = out.entry(h4_key(k)).or_default();
            if sd == "long" { e.1 = Some(v.s) } else { e.2 = Some(v.s) }
        }
        Ok(out)
    }

    /// _h4: 지난 H4_DAYS 일의 4시간 칸 + 형성 중 칸. 반환 (키 → 칸 · 차례 = 원본 dict 차례, 형성 중 키).
    fn h4(&self, now: f64) -> Res<(Vec<(String, Bucket)>, String)> {
        let mut st = self.st.lock().unwrap();
        let cur = (now.floor() as i64).div_euclid(H4_SEC) * H4_SEC;
        let first = cur - H4_DAYS * DAY;
        let want: Vec<String> = (first..cur).step_by(H4_SEC as usize).map(h4_key).collect();
        let sec = |k: &str| h4_sec(k).unwrap();
        let settled = |k: &str| now >= (sec(k) + H4_SEC + H4_SETTLE) as f64;
        if now - st.h4_gap_at >= H4_GAP_TTL {
            st.h4_closed.retain(|k, v| !(v.is_none() && now < (sec(k) + H4_SEC + H4_BACKFILL) as f64));
            st.h4_gap_at = now;
        }
        let live_keys: Vec<String> = st.h4_live.iter().map(|x| x.0.clone()).collect();
        let missing: Vec<String> = want.iter().filter(|k| !st.h4_closed.contains_key(*k) && !live_keys.contains(k)).cloned().collect();
        if !missing.is_empty() || now - st.h4_at >= TODAY_TTL {
            let t0 = first.max(missing.iter().chain(live_keys.iter()).map(|k| sec(k)).chain([cur]).min().unwrap());
            let frames = self.tape_rows_by_day(day_of_sec(t0), day_of_sec(now.floor() as i64), t0)?;
            // h4_split: 묶음들을 이어 붙인 차례로 4시간 칸별
            let mut all = Tape::default();
            for (_, t) in &frames {
                all.extend(t);
            }
            let mut by: BTreeMap<i64, Tape> = BTreeMap::new();
            for i in 0..all.sec.len() {
                by.entry(all.sec[i].div_euclid(H4_SEC) * H4_SEC).or_default().push(all.sec[i], all.pbin[i], all.buy[i], all.sell[i]);
            }
            let got: Vec<(String, Bucket)> = by.iter().map(|(&k, t)| (h4_key(k), day_from_tape(&self.sp, t))).collect();
            let ctx = self.h4_ctx(t0, now)?;
            let mut live = vec![];
            let got_keys: BTreeSet<String> = got.iter().map(|x| x.0.clone()).collect();
            for (k, mut v) in got {
                if sec(&k) < t0 {
                    continue;
                }
                if let Some(&(oi, ll, ls)) = ctx.get(&k) {
                    if oi.is_some() {
                        v.0.oi = oi;
                    }
                    if ll.is_some() {
                        v.0.ll = ll;
                    }
                    if ls.is_some() {
                        v.0.ls = ls;
                    }
                }
                if settled(&k) {
                    st.h4_closed.entry(k).or_insert(Some(v));
                } else {
                    live.push((k, v));
                }
            }
            for k in &missing {
                if !got_keys.contains(k) && settled(k) {
                    st.h4_closed.entry(k.clone()).or_insert(None);
                }
            }
            let floor = h4_key(first);
            st.h4_closed.retain(|k, _| k.as_str() >= floor.as_str());
            st.h4_live = live;
            st.h4_at = now;
        }
        let mut out: Vec<(String, Bucket)> = want.iter().filter_map(|k| st.h4_closed.get(k).cloned().flatten().map(|v| (k.clone(), v))).collect();
        for (k, v) in &st.h4_live {
            match out.iter_mut().find(|x| &x.0 == k) {
                Some(x) => x.1 = v.clone(),
                None => out.push((k.clone(), v.clone())),
            }
        }
        Ok((out, h4_key(cur)))
    }

    /// h4_payload(now) -- /api/footprint-4h 본문(?since 전).
    pub fn h4_payload(&self, now: f64) -> Res<V> {
        let (b, live_key) = self.h4(now)?;
        let s: BTreeMap<String, Summ> = b.iter().map(|(k, v)| (k.clone(), v.0.clone())).collect();
        let oi: HashMap<String, f64> = s.iter().filter_map(|(k, r)| r.oi.map(|v| (k.clone(), v))).collect();
        let liq: HashMap<String, [f64; 2]> = s.iter().map(|(k, r)| (k.clone(), [r.ll.unwrap_or(0.0), r.ls.unwrap_or(0.0)])).collect();
        let cols = build_days(None, &s, &oi, &liq, Some(1), &live_key);
        Ok(json!({"symbol": self.sp.symbol, "tz": "UTC", "period": "4h", "cols": cols, "at": now}))
    }

    /// h4_cells_payload(now, from, to, row) -- /api/footprint-4h/cells 본문(범위 검사 없음 -- 원본도 없다).
    pub fn h4_cells_payload(&self, now: f64, frm: &str, to: &str, row: i64) -> Res<V> {
        let (b, _) = self.h4(now)?;
        let mut out = Map::new();
        for (k, v) in b {
            if frm <= k.as_str() && k.as_str() <= to {
                out.insert(k, rebin(&self.sp, &v.1, row));
            }
        }
        Ok(json!({"row": py_round(row as f64 * self.sp.cell, self.sp.dp), "cells": out}))
    }
}

// ── server.py 손잡이 ──
/// parse_row(footprint_daily.py 552~558): 행 크기($) → CELL 칸 수 1~500. 숫자 아님·nan·inf = 5.
pub fn parse_row(v: Option<&str>, cell: f64) -> i64 {
    let Some(s) = v.filter(|s| !s.is_empty()) else { return 5 };
    let t = s.trim();
    // 파이썬 float(): 앞뒤 공백 · 숫자 사이 밑줄 · inf/nan/infinity(대소문자 무관)
    let ok_us = !t.starts_with('_') && !t.ends_with('_') && !t.contains("__") && !t.contains("_.") && !t.contains("._");
    let clean: String = if ok_us { t.replace('_', "") } else { t.to_string() };
    let Ok(x) = clean.parse::<f64>() else { return 5 };
    let r = x / cell;
    if !r.is_finite() {
        return 5;
    }
    (r.round_ties_even() as i64).clamp(1, 500)
}
/// _iso_day: 'YYYY-MM-DD' 만(원본 date.fromisoformat -- 3.11 은 'YYYYMMDD' 도 받는다: 그것도 받아 isoformat 으로 돌려준다).
pub fn iso_day(v: &str) -> Option<String> {
    if v.len() == 8 && v.bytes().all(|b| b.is_ascii_digit()) {
        return parse_day(&format!("{}-{}-{}", &v[..4], &v[4..6], &v[6..])).map(ds);
    }
    parse_day(v).map(ds)
}
/// _iso_h4: 'YYYY-MM-DDTHH'(HH = 00·04·…·20) -- h4_key(_h4_sec(v)) == v 인 것만.
pub fn iso_h4(v: &str) -> bool {
    h4_sec(v).is_some_and(|s| h4_key(s) == v)
}
/// _cols_since: ?since= 면 그 칸(날·4시간 키)부터만.
pub fn cols_since(full: &V, since: Option<&str>) -> V {
    let Some(since) = since.filter(|s| !s.is_empty()) else { return full.clone() };
    let cols = &full["cols"];
    let keep: Vec<usize> = cols["d"].as_array().map_or(vec![], |d| d.iter().enumerate().filter(|(_, x)| x.as_str().is_some_and(|x| x >= since)).map(|(i, _)| i).collect());
    let mut m = Map::new();
    for (k, v) in cols.as_object().into_iter().flatten() {
        m.insert(k.clone(), V::Array(keep.iter().map(|&i| v[i].clone()).collect()));
    }
    let mut out = full.as_object().cloned().unwrap_or_default();
    out.insert("cols".into(), V::Object(m));
    V::Object(out)
}
