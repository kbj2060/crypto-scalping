//! 재시작·재연결 복원(2026-11-21). 설계 = rtcore/RESTORE.md 권고 (다)+(나): 코어가 자기 스냅숏을 30초마다 쓰고,
//! 그 뒤 공백은 체결 테이프 hot SQLite(`trade_tape_1s`)에서 메운다. REST 는 쓰지 않는다(2026-09-26 418 밴 27분 -- IP 한도를 봇과 공유).
//! 범위: 바이낸스 선물 몫(풋프린트 봉 144 + 1초 칸 660초 + dhl) + 2026-11-21 둘째 조각(아래 «나머지 원천»).
//!
//! 나머지 원천(2026-11-21 · RESTORE.md §6 · 파이썬이 되살리는 것과 같은 원천·같은 식 -- 정답지 tools/gen_restore_venues_fixture.py):
//! - OKX 1초 칸: 첫 OKX 체결 초 t0 에서 [t0 − 660, t0) 를 OKX 테이프 `trade_tape_1s`(tape_seconds 와 같은 SQL · 칸 폭 = OKX 테이프 칸)로,
//!   실시간이 가진 초는 안 덮는다(supply_1s_restore fill · 테이프 끝이 t0 − 1 에 닿을 때까지 5초 × 12번).
//! - OKX 미결제약정: 첫 OI 초 t0 에서 [t0 − 660, t0) 를 `okx_ctx` `okx_oi`(시각 순 · 같은 초는 마지막 값 · 한 번만 읽는다 -- 파이썬도 «DB 는 기다려도 안 는다»).
//! - OKX 청산: 첫 OKX 체결 시각 t0_ms 앞 660초를 `okx_ctx` `okx_liquidations` 에서 링 **앞**에(남은 자리만 · 가장 새 것부터 -- okx_footprint_restore 958~964).
//!   🔴파이썬은 이 청산 복원을 OKX 풋프린트 봉 복원에 얹어 «OKX 테이프가 1분봉 대조로 온전한 봉»까지만 한다 -- 코어에는 OKX 풋프린트가 없어 그 조건 없이 660초를 읽는다(RESTORE.md §5).
//! - 바이낸스 강제청산: 시작 때 `binance_ctx` `liquidations` 최근 5,000건 중 지금 − 660초 이후(server.py liq_events_load 1785~1815 · usd 는 DB 값).
//! - Bybit 청산: 파이썬은 요청마다 수집기 DB 를 읽어 재시작 영향이 없다(server.py bybit_liq_events) -- 코어는 시작 때 창((지금 − 3 − 660, …])을 DB 에서 채우고,
//!   재시작 직후 실시간으로 다시 온 같은 사건은 건너뛴다(supply.rs liq).
//! 읽을 DB 는 파이썬 수집기 파일이 기본이고 환경변수로 Rust 수집기 파일(data/hot/rtcollect/…, 표 모양 같음)로 바꾼다(lib.rs Cfg).
//!
//! 원본(파이썬): dashboard/coin_flow.py footprint_save 396~417 · footprint_load 419~440 · footprint_tape_restore 533~570 ·
//!   okx_tape_footprint 129~203(봉 SQL·칸 키) · tape_seconds 204~216(1초 SQL) · supply_1s_restore 973~1018 ·
//!   dashboard/server.py fp_dhl 3952~3991(테이프 dhl SQL). 일치 정답지: tools/gen_restore_fixture.py → tests/restore_parity.rs.
//!
//! 순서(심볼마다):
//! 1. 시작(수신 전): 스냅숏 파일을 읽어 봉(통째로)·dhl·last_ms·holes 를 되살린다(State::with_restore · 파일 하나 수 ms).
//!    1초 칸은 «옆에 둔다»(Pending.snap_sec) -- 테이프가 있는 초는 테이프 값을 쓰려고(파이썬 supply_1s_restore 는 1초 칸을 테이프로만 되살린다).
//! 2. 수신을 먼저 붙인다. 그 연결의 첫 `@trade`(first_ms)가 오면 공백 [lo, hi) 를 정하고(begin_restore), 별도 스레드에서 테이프를
//!    읽어(read) 심볼 락 안에서 합친다(merge). 그사이 실시간 체결은 그대로 쌓는다 -- 합치기는 봉 칸 `+=`(순서 무관) · 1초 칸은 실시간이
//!    가진 초를 안 덮는다(setdefault) · dhl 은 링 밖 봉만 -- 라서 실시간과 섞여도 값이 같다(파이썬도 «받으면서 동시에 메운다» 533~538).
//!    복원을 먼저 끝내고 붙으면 «복원 끝 ~ 첫 체결» 이 다시 빈다(테이프는 5초마다 쓴다) -- 그래서 붙은 뒤의 첫 체결을 경계로 쓴다.
//! 3. 재연결도 같은 길이다(파이썬 footprint_tape_restore 는 재연결마다 돈다). 메우는 중에 또 끊기면 새 공백은 holes 로만 남긴다
//!    (파이썬은 «돌고 있는 백필이 있으면 새로 띄우지 않는다» 610~611 -- 그 공백을 조용히 잃는다).
//!
//! 경계(RESTORE.md «그대로 지켜야 할 규칙»):
//! - 봉: lo = max(창 바닥, last_ms 의 초 + 1), hi = first_ms 의 초(배타). 창 바닥 = first_ms 의 봉 − 143봉. last_ms 가 없으면 창 바닥부터.
//!   끊기기 직전 초(last_ms 의 초)·붙은 첫 초(first_ms 의 초)는 테이프로 안 더한다 -- 앞 초는 스냅숏이, 뒤 초는 실시간이 일부 셌다.
//!   그 두 초에서 «스냅숏 뒤 · 첫 체결 앞» 몫은 빠진다(5분봉의 1% 미만 · 파이썬과 같다 · 구멍으로 치지 않는다).
//! - 1초 칸: [t0 − 660, t0) 를 실시간 > 테이프(테이프 끝 초까지) > 코어 스냅숏 순으로. 실시간이 가진 초는 안 건드린다.
//!   t0 = 이 연결의 첫 체결의 초. 🔴파이썬은 `t0 = min(dst)`(supply_1s_restore 985) -- 붙은 뒤 늦게 온 체결이 더 이른 초를 만들면 t0 가 그 초로 당겨져
//!   (그 초, 첫 체결 초) 사이가 영영 안 채워진다(늦은 체결이 1초 칸에 구멍을 낸다). 코어는 첫 체결 초를 쓴다 -- RESTORE.md §5.
//!   테이프는 완결된 초만 5초마다 쓰므로 붙은 직후엔 first_s 앞 몇 초가 아직 없다 → 테이프 끝이 hi − 1 에 닿을 때까지 5초 간격 최대 12번 기다린다.
//!   끝내 못 닿으면 있는 데까지 메우고 [테이프 끝 + 1, hi) 를 holes 로.
//! - 칸 키 = int(round(round(price_bin × 테이프칸, 10) / 칸 폭)) -- SQL 반올림 금지(2026-10-20 SOL 동률 사고: SQLite 는 0 에서 먼 쪽, 라이브는 짝수 쪽).
//! - 테이프 봉 칸은 **더한다**(`+=`), 스냅숏 봉은 **통째로** 쓴다. 같은 초를 두 번 더하지 않는 것은 lo·hi 경계가 지킨다.
//! - 스냅숏은 «합치기가 한 번 끝났고 지금 메우는 중이 아닐 때»만 쓴다(파이썬 `ready` 일 때만 저장 390~392 와 같은 뜻 -- 2026-09-15 시험에서
//!   메우는 중인 봉을 저장했더니 다음 판이 «이미 있는 봉»으로 보고 건너뛰어 −83.7% 로 굳었다). last_ms 를 앞으로 밀면 그 앞 공백은 영영 안 메워진다.
use crate::footprint::{bar_of, dhl_of, dhl_ok, ring_floor, spec, Bar, BAR_SECONDS, KEEP_BARS, WINDOW_BARS};
use crate::proto::{SecRow, Supply1s};
use crate::supply::{r3, sec_row, Supply, SUPPLY_1S_SECONDS};
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// 스냅숏 판 번호(파이썬 `cells_v` 와 같은 뜻 -- 칸 모양이 바뀌면 올린다. 다르면 통째로 버리고 테이프로만 시작한다).
pub const SNAP_V: u32 = 1;
/// 스냅숏 주기. 파이썬과 같은 30초(coin_flow.py FOOTPRINT_SNAPSHOT_SECONDS). 이 주기는 «테이프가 없을 때 잃는 최대 시간»일 뿐이다 --
/// 테이프가 있으면 스냅숏 뒤 공백은 테이프가 메운다. 쓰기 = 8코인 × 수백 KB / 30초(초당 수십 KB) -- 더 줄일 까닭도 늘릴 까닭도 없다.
pub const SNAP_EVERY: Duration = Duration::from_secs(30);

/// 바이낸스 체결 테이프 칸 폭 -- scripts/live_trade_tape_collector_20260916.py BUCKETS 사본(풋프린트 칸 폭의 약수여야 칸이 안 찢긴다).
// ponytail: 파이썬 표 사본 -- 수집기 칸 폭을 바꾸면 tools/gen_restore_fixture.py 정답지의 tape_buckets 대조가 잡는다.
const TAPE_BUCKETS: &[(&str, f64)] = &[
    ("ethusdt", 0.1),
    ("btcusdt", 1.0),
    ("solusdt", 0.01),
    ("xrpusdt", 0.0001),
    ("hypeusdt", 0.001),
    ("zecusdt", 0.05),
    ("sndkusdt", 0.05),
    ("soxlusdt", 0.01),
    ("skhynixusdt", 0.1),
    ("muusdt", 0.05),
];

pub fn tape_bucket(symbol: &str) -> Option<f64> {
    TAPE_BUCKETS.iter().find(|b| b.0 == symbol).map(|b| b.1)
}

/// OKX 체결 테이프 칸 폭(키 = inst) -- scripts/live_okx_trade_tape_collector_20260923.py BUCKETS 사본.
// ponytail: 파이썬 표 사본 -- tools/gen_restore_venues_fixture.py 정답지의 okx_tape_buckets 대조가 갈라짐을 잡는다.
const OKX_TAPE_BUCKETS: &[(&str, f64)] = &[
    ("ETH-USDT-SWAP", 0.1),
    ("BTC-USDT-SWAP", 1.0),
    ("SOL-USDT-SWAP", 0.01),
    ("XRP-USDT-SWAP", 0.0001),
    ("HYPE-USDT-SWAP", 0.001),
    ("ZEC-USDT-SWAP", 0.05),
    ("SNDK-USDT-SWAP", 0.05),
    ("SOXL-USDT-SWAP", 0.01),
    ("SKHYNIX-USDT-SWAP", 0.1),
    ("MU-USDT-SWAP", 0.05),
];

pub fn okx_tape_bucket(inst: &str) -> Option<f64> {
    OKX_TAPE_BUCKETS.iter().find(|b| b.0 == inst).map(|b| b.1)
}

/// 복원 스레드에 알리는 일감 종류(state.rs apply → lib.rs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Wake {
    /// 바이낸스 선물 공백(봉·1초 칸·dhl) -- 연결마다.
    Bn,
    /// OKX 1초 칸 + OKX 청산 -- 시작 뒤 첫 OKX 체결 때 한 번.
    Okx,
    /// OKX 미결제약정 -- 시작 뒤 첫 OI 때 한 번.
    OkxOi,
}

pub struct Cfg {
    /// 코어 스냅숏 폴더(`<심볼>.json`). 파이썬의 data/live/footprint_<코인>.json 과 다른 파일 -- 파이썬 파일은 안 건드린다.
    pub snap_dir: PathBuf,
    /// 체결 테이프 hot SQLite. 읽기 전용으로만 연다.
    pub tape: PathBuf,
    /// 테이프가 hi − 1 까지 따라올 때까지 기다리는 횟수·간격(파이썬 12번 × 5초 -- 테이프는 5초마다 쓴다).
    pub tries: u32,
    pub wait: Duration,
    /// 나머지 원천(머리 주석 «나머지 원천»). 없는 파일이면 로그 한 줄을 남기고 그 원천만 빈 채로 간다.
    pub okx_tape: PathBuf,
    pub okx_ctx: PathBuf,
    pub bn_ctx: PathBuf,
    pub bybit_liq: PathBuf,
}

pub fn snap_path(dir: &Path, symbol: &str) -> PathBuf {
    dir.join(format!("{symbol}.json"))
}

/// 메울 공백 하나: 앞 연결(또는 스냅숏)의 마지막 체결 시각 · 이 연결의 첫 체결 시각.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Job {
    pub last_ms: i64,
    pub first_ms: i64,
}

impl Job {
    /// 봉을 테이프로 메울 초 [lo, hi)(footprint_tape_restore 539~542).
    pub fn range(&self) -> (i64, i64) {
        let floor = bar_of(self.first_ms) - (WINDOW_BARS as i64 - 1) * BAR_SECONDS;
        let lo = if self.last_ms > 0 { floor.max(self.last_ms.div_euclid(1000) + 1) } else { floor };
        (lo, self.first_ms.div_euclid(1000))
    }
}

/// Supply 안의 복원 상태.
#[derive(Default)]
pub struct Pending {
    /// 복원 켜짐(State::with_restore · 칸 폭 표에 있는 종목만).
    pub(crate) on: bool,
    /// 다음 `@trade` 가 이 연결의 첫 체결이다(연결마다 reset_orders 가 켠다).
    pub(crate) await_first: bool,
    pub(crate) job: Option<Job>,
    /// 스냅숏의 1초 칸 -- 합칠 때 테이프가 없는 초에만 쓴다.
    pub(crate) snap_sec: BTreeMap<i64, [f64; 7]>,
    /// 이번 판에서 합치기를 한 번 끝냈다(스냅숏 쓰기 조건).
    pub(crate) merged: bool,
    /// 다음 OKX 체결·OI 가 시작 뒤 첫 것이다(OKX 계약 크기를 아는 종목만 켠다). 파이썬처럼 시작 때 한 번만 -- 재연결엔 안 메운다.
    pub(crate) okx_await: bool,
    pub(crate) oi_await: bool,
    /// 메울 OKX 일감: 첫 체결 시각(ms) · 첫 OI 초.
    pub(crate) okx_job: Option<i64>,
    pub(crate) oi_job: Option<i64>,
    /// DB 에서 되살린 Bybit 청산의 가장 늦은 시각(ms) -- 이 앞 시각으로 다시 온 같은 사건은 건너뛴다(supply.rs liq).
    pub(crate) bybit_max_ms: i64,
}

/// 코어 스냅숏 파일. 실수는 반올림 전 값 그대로(serde_json float_roundtrip 으로 비트까지 되읽는다).
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct Snap {
    pub v: u32,
    pub symbol: String,
    pub bar_seconds: i64,
    pub bucket: f64,
    /// 스냅숏까지 받은 가장 늦은 `@trade` 시각(ms) -- 다음 공백 메우기의 시작.
    pub last_ms: i64,
    /// 창(가장 새 144봉) -- (봉 시작, [(칸 키, 6칸)], 그때 보이던 dhl).
    pub bars: Vec<(i64, Vec<(i64, [f64; 6])>, Option<[f64; 3]>)>,
    /// 1초 링(가장 새 초 − 660 부터) -- 7칸.
    pub sec: Vec<(i64, [f64; 7])>,
    /// 못 메운 초 구간(창 안에 걸친 것만) -- 다음 판도 그 봉을 «구멍 있음»으로 보인다.
    pub holes: Vec<(i64, i64)>,
}

/// 스냅숏 읽기(footprint_load 419~440): 판·종목·봉 길이·칸 폭이 다르면 버린다 · 143봉보다 오래된 봉은 버린다 ·
/// 창 안 봉이 하나도 없으면 스냅숏 전체를 버린다(파이썬은 last_ms 도 안 되살린다 -- 테이프로 창 전체를 메운다).
pub fn load(path: &Path, symbol: &str, now_ms: i64) -> Result<Snap, String> {
    let text = std::fs::read(path).map_err(|e| format!("스냅숏 없음({e})"))?;
    let mut s: Snap = serde_json::from_slice(&text).map_err(|e| format!("스냅숏 깨짐({e})"))?;
    let (bucket, _) = spec(symbol).ok_or("칸 폭 표에 없는 종목")?;
    if s.v != SNAP_V || s.symbol != symbol || s.bar_seconds != BAR_SECONDS || s.bucket != bucket {
        return Err(format!("스냅숏 판이 다르다(판 {} · {} · {}초 · 칸 {}) -- 버린다", s.v, s.symbol, s.bar_seconds, s.bucket));
    }
    let cutoff = bar_of(now_ms) - (WINDOW_BARS as i64 - 1) * BAR_SECONDS;
    s.bars.retain(|b| b.0 >= cutoff);
    if s.bars.is_empty() {
        return Err(format!("스냅숏 봉이 전부 창(143봉 전 {cutoff}) 밖 -- 버린다"));
    }
    Ok(s)
}

/// 원자적 쓰기: 임시 파일에 쓰고 디스크까지 밀어 넣은 뒤 이름을 바꾼다(중간에 죽어도 반쪽 파일이 안 남는다).
pub fn write(path: &Path, s: &Snap) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tmp = path.with_extension("json.tmp");
    let mut f = std::fs::File::create(&tmp)?;
    f.write_all(&serde_json::to_vec(s).map_err(std::io::Error::other)?)?;
    f.sync_all()?;
    std::fs::rename(&tmp, path)
}

/// 테이프에서 읽은 것(합치기 전).
#[derive(Debug, Default)]
pub struct Tape {
    /// 봉 → 칸 키 → [매수, 매도, 고래매수, 고래매도, 리테일매수, 리테일매도]([lo, hi) 초).
    pub bars: BTreeMap<i64, BTreeMap<i64, [f64; 6]>>,
    /// 초 → 1초 칸 7칸([first_s − 660, first_s)).
    pub secs: BTreeMap<i64, [f64; 7]>,
    /// 봉 → [최저, 최고, 마감](round 3 · 테이프가 봉을 다 든 봉만).
    pub dhl: BTreeMap<i64, [f64; 3]>,
    /// 테이프에 있는 이 종목의 가장 늦은 초(없으면 0).
    pub top: i64,
    /// 못 메운 초 구간: 수집기 끊김(gaps · 1분봉 대조가 맞은 분은 뺀다) · 테이프가 덜 따라옴.
    pub holes: Vec<(i64, i64)>,
}

// SQL 은 파이썬 원문 그대로(같은 SQLite 집계 = 같은 합 · 둘 다 3.43 이상이라 sum 이 보정 합산).
// coin_flow.py okx_tape_footprint 147~151
const SQL_BARS: &str = "SELECT ts_sec - ts_sec % ? AS bar, price_bin,
                  sum(buy_qty), sum(sell_qty),
                  sum(coalesce(whale_buy_qty, 0)), sum(coalesce(whale_sell_qty, 0)),
                  sum(coalesce(retail_buy_qty, 0)), sum(coalesce(retail_sell_qty, 0))
           FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ? AND ts_sec < ? GROUP BY 1, 2";
// coin_flow.py tape_seconds 209~214
const SQL_SECS: &str = "
        SELECT ts_sec, sum(coalesce(retail_buy_qty, 0)), sum(coalesce(retail_sell_qty, 0)),
               sum(coalesce(whale_buy_qty, 0)), sum(coalesce(whale_sell_qty, 0)), sum(buy_qty), sum(sell_qty),
               sum(price_bin * (buy_qty + sell_qty)) / nullif(sum(buy_qty + sell_qty), 0) * ?
        FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ? AND ts_sec < ? GROUP BY 1";
// server.py fp_dhl 3975~3978(봉당 한 줄 -- 1초 행 수만 개를 받지 않는다)
const SQL_DHL: &str = "WITH s AS (SELECT ts_sec - ts_sec % ? AS bar, ts_sec, sum(buy_qty) - sum(sell_qty) AS d FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ? AND ts_sec < ? GROUP BY ts_sec), c AS (SELECT bar, d, sum(d) OVER (PARTITION BY bar ORDER BY ts_sec) AS cum FROM s) SELECT bar, min(cum), max(cum), sum(d) FROM c GROUP BY bar";
const SQL_TOP: &str = "SELECT max(ts_sec) FROM trade_tape_1s WHERE symbol = ?";

/// 파이썬 round(x, 10) 뒤 / 칸 폭 → 짝수 반올림(coin_flow.py 163). `{:.10}` 은 정확값 기준·동률 짝수라 파이썬 round 와 같다(supply.rs r3).
fn tape_key(pb: i64, tape_bucket: f64, bucket: f64) -> i64 {
    let px: f64 = format!("{:.10}", pb as f64 * tape_bucket).parse().unwrap_or(f64::NAN);
    (px / bucket).round_ties_even() as i64
}

/// 파이썬 `float(x or 0.0)`.
fn f(r: &rusqlite::Row, i: usize) -> rusqlite::Result<f64> {
    Ok(r.get::<_, Option<f64>>(i)?.unwrap_or(0.0))
}

/// 테이프 읽기(읽기 전용 · 이 스레드를 `tries × wait` 까지 잡는다 -- 실행기 밖 스레드에서 부른다).
pub fn read(path: &Path, symbol: &str, job: Job, bucket: f64, tape_bucket: f64, tries: u32, wait: Duration) -> Result<Tape, String> {
    let (lo, hi) = job.range();
    let first_s = job.first_ms.div_euclid(1000);
    let floor = bar_of(job.first_ms) - (WINDOW_BARS as i64 - 1) * BAR_SECONDS;
    let err = |e: rusqlite::Error| format!("{}: {e}", path.display());
    // mode=ro(파이썬 data_store.read_rows 와 같다). 없는 파일은 만들지 않고 실패한다.
    let con = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX).map_err(err)?;
    con.busy_timeout(Duration::from_secs(10)).map_err(err)?;
    let mut t = Tape::default();
    for i in 0..tries.max(1) {
        t.top = con.query_row(SQL_TOP, [symbol], |r| r.get::<_, Option<i64>>(0)).map_err(err)?.unwrap_or(0);
        if t.top >= hi - 1 || i + 1 >= tries {
            break;
        }
        std::thread::sleep(wait);
    }
    // 봉 칸(okx_tape_footprint 158~166): 테이프 칸 → 풋프린트 칸 키는 파이썬에서(위 tape_key) · 같은 키로 모이는 테이프 칸은 SQL 행 순서로 더한다.
    let mut key: HashMap<i64, i64> = HashMap::new();
    let mut st = con.prepare(SQL_BARS).map_err(err)?;
    let mut rows = st.query(rusqlite::params![BAR_SECONDS, symbol, lo, hi]).map_err(err)?;
    while let Some(r) = rows.next().map_err(err)? {
        let (bar, pb): (i64, i64) = (r.get(0).map_err(err)?, r.get(1).map_err(err)?);
        let k = *key.entry(pb).or_insert_with(|| tape_key(pb, tape_bucket, bucket));
        let c = t.bars.entry(bar).or_default().entry(k).or_insert([0.0; 6]);
        for (j, x) in c.iter_mut().enumerate() {
            *x += f(r, 2 + j).map_err(err)?;
        }
    }
    drop(rows);
    // 1초 칸(tape_seconds): 가격은 그 초 마지막 체결가가 아니라 칸 가중평균(테이프에 순서가 없다 -- 오차 < 테이프 칸 1개).
    let mut st = con.prepare(SQL_SECS).map_err(err)?;
    let mut rows = st.query(rusqlite::params![tape_bucket, symbol, first_s - SUPPLY_1S_SECONDS, first_s]).map_err(err)?;
    while let Some(r) = rows.next().map_err(err)? {
        let mut c = [0.0; 7];
        for (j, x) in c.iter_mut().enumerate() {
            *x = f(r, 1 + j).map_err(err)?;
        }
        t.secs.insert(r.get(0).map_err(err)?, c);
    }
    drop(rows);
    // dhl(fp_dhl 테이프 가지 3983): 테이프가 봉을 다 든 봉만([창 바닥, min(테이프 끝 + 1, hi)) 의 온전한 봉).
    let end = (t.top + 1).min(hi);
    let end = end - end.rem_euclid(BAR_SECONDS);
    if floor < end {
        let mut st = con.prepare(SQL_DHL).map_err(err)?;
        let mut rows = st.query(rusqlite::params![BAR_SECONDS, symbol, floor, end]).map_err(err)?;
        while let Some(r) = rows.next().map_err(err)? {
            let (lo_, hi_, c) = (f(r, 1).map_err(err)?, f(r, 2).map_err(err)?, f(r, 3).map_err(err)?);
            // round(min(0.0, lo), 3) · round(max(0.0, hi), 3) -- 파이썬 min/max 는 앞의 것이 이긴다(-0.0 이 0.0 을 못 이긴다)
            t.dhl.insert(r.get(0).map_err(err)?, [r3(if lo_ < 0.0 { lo_ } else { 0.0 }), r3(if hi_ > 0.0 { hi_ } else { 0.0 }), r3(c)]);
        }
    }
    // 구멍: 수집기가 끊긴 구간(gaps) 중 1분봉 대조가 맞은 분(자동 복구 포함)은 뺀다(okx_tape_footprint ok_min 과 같은 뜻 · 파이썬 풋프린트 복원은 이 표를 안 본다).
    if lo < hi {
        let mut st = con.prepare("SELECT from_ms, to_ms FROM gaps WHERE symbol = ? AND to_ms > ? AND from_ms < ?").map_err(err)?;
        let gaps: Vec<(i64, i64)> = st.query_map(rusqlite::params![symbol, lo * 1000, hi * 1000], |r| Ok((r.get(0)?, r.get(1)?))).map_err(err)?.collect::<Result<_, _>>().map_err(err)?;
        let mut st = con.prepare("SELECT ts_min FROM verify_1m WHERE symbol = ? AND ts_min >= ? AND ts_min < ? AND abs(rel_err) <= 1e-4").map_err(err)?;
        let good: BTreeSet<i64> = st.query_map(rusqlite::params![symbol, lo - lo.rem_euclid(60), hi], |r| r.get(0)).map_err(err)?.collect::<Result<_, _>>().map_err(err)?;
        for (a, z) in gaps {
            // 끊기기 직전 초·붙은 첫 초까지 넣는다(그 두 초는 수집기가 일부만 셌다).
            let (s0, s1) = (a.div_euclid(1000), z.div_euclid(1000) + 1);
            let mut m = s0 - s0.rem_euclid(60);
            while m < s1 {
                let (x, y) = (m.max(s0).max(lo), (m + 60).min(s1).min(hi));
                if !good.contains(&m) && x < y {
                    t.holes.push((x, y));
                }
                m += 60;
            }
        }
        if t.top < hi - 1 {
            t.holes.push((lo.max(t.top + 1), hi));
        }
    }
    Ok(t)
}

impl Supply {
    /// 이 연결의 첫 `@trade` 앞(state.rs apply · 그 체결을 쌓기 전 -- last_ms 가 앞 연결의 값). 참 = 새 일감(스레드에 알린다).
    pub(crate) fn begin_restore(&mut self, first_ms: i64) -> bool {
        self.rs.await_first = false;
        let job = Job { last_ms: self.last_ms, first_ms };
        let (lo, hi) = job.range();
        if self.rs.job.is_some() {
            if lo < hi {
                println!("rtcore 복원: 메우는 중에 또 끊김 -- [{lo}, {hi}) {}초는 구멍으로 남긴다", hi - lo);
                self.fp.holes.push((lo, hi));
                self.fp.holes_dirty = true;
            }
            return false;
        }
        if lo < hi {
            self.fp.pending = Some((lo, hi));
            self.fp.holes_dirty = true;
        }
        self.rs.job = Some(job);
        true
    }

    /// 테이프 읽은 것을 합친다(심볼 락 안). 돌려주는 것 = 로그 한 줄 · 되살린 1초 칸 밀기.
    pub(crate) fn merge(&mut self, t: Result<Tape, String>, now_us: i64) -> (String, Option<Supply1s>) {
        let Some(job) = self.rs.job.take() else { return ("메울 일감 없음".into(), None) };
        self.rs.merged = true;
        let (lo, hi) = job.range();
        let first_s = job.first_ms.div_euclid(1000);
        let cut = bar_of(job.first_ms) - (KEEP_BARS - 1) * BAR_SECONDS;
        self.fp.pending = None;
        self.fp.holes_dirty = true;
        let snap_sec = std::mem::take(&mut self.rs.snap_sec);
        let top = t.as_ref().map_or(i64::MIN, |t| t.top);
        // ① 봉: 테이프 칸을 더한다(footprint_tape_restore 557~567 -- 스냅숏 봉은 시작 때 통째로 썼다).
        let mut touched = BTreeSet::new();
        let mut n_cells = 0;
        if let Ok(t) = &t {
            for (&b, cells) in t.bars.range(cut..) {
                let bar = self.fp.bars.entry(b).or_default();
                let dirty = self.fp.dirty.entry(b).or_default();
                for (&k, v) in cells {
                    let c = bar.cells.entry(k).or_insert([0.0; 6]);
                    for j in 0..6 {
                        c[j] += v[j];
                    }
                    dirty.insert(k);
                    n_cells += 1;
                }
                touched.insert(b);
            }
        }
        // ② 1초 칸: 실시간 > 테이프 > 스냅숏(setdefault -- supply_1s_restore 1000).
        let (mut n_tape, mut n_snap, mut restored) = (0, 0, vec![]);
        for s in first_s - SUPPLY_1S_SECONDS..first_s {
            if self.sec.contains_key(&s) {
                continue;
            }
            let v = match &t {
                Ok(t) if s <= top => t.secs.get(&s).inspect(|_| n_tape += 1),
                _ => snap_sec.get(&s).inspect(|_| n_snap += 1),
            };
            if let Some(v) = v {
                self.sec.insert(s, *v);
                restored.push(s);
            }
        }
        // ③ dhl: 링이 다 드는 봉은 밀기·스냅숏이 링으로 잰다. 그 밖 창 안 봉:
        //    테이프가 봉을 다 들면 테이프(fp_dhl 테이프 가지) · 아니면 링이 그 봉의 초를 다 들고 있으면(방금 [t0 − 660, t0) 를 채웠다) 링 ·
        //    둘 다 아니면 스냅숏 값(테이프가 안 건드린 봉만 -- 건드린 봉은 칸이 바뀌었으니 옛 값이 틀리다).
        //    테이프 값이 어긋남 검사에 걸리고 테이프가 안 건드린 봉이면 스냅숏 값(파이썬은 «없음» -- 테이프 수집기 공백 등).
        let floor = ring_floor(&self.sec).unwrap_or(i64::MAX);
        let win_lo = self.fp.window_lo().unwrap_or(i64::MIN);
        let w_floor = bar_of(job.first_ms) - (WINDOW_BARS as i64 - 1) * BAR_SECONDS;
        let end = top.saturating_add(1).min(hi);
        let mut n_dhl = 0;
        let keys: Vec<i64> = self.fp.bars.keys().copied().filter(|&b| win_lo <= b && b < floor).collect();
        for b in keys {
            let bar: &Bar = &self.fp.bars[&b];
            let mine = (!touched.contains(&b)).then_some(bar.dhl).flatten();
            let v = match &t {
                Ok(t) if b >= w_floor && b + BAR_SECONDS <= end => {
                    n_dhl += 1;
                    t.dhl.get(&b).and_then(|v| dhl_ok(*v, bar)).or(mine)
                }
                _ if b >= first_s - SUPPLY_1S_SECONDS => dhl_of(&self.sec, b, bar),
                _ => mine,
            };
            if v != bar.dhl {
                self.fp.bars.get_mut(&b).expect("방금 읽었다").dhl = v;
                self.fp.dirty.entry(b).or_default();
            }
        }
        // ④ 구멍
        let mut line = format!("공백 [{lo}, {hi}) {}초", (hi - lo).max(0));
        match &t {
            Ok(t) => {
                self.fp.holes.extend(&t.holes);
                line += &format!(
                    " · 테이프 봉 {}개 칸 {n_cells} · 1초 칸 테이프 {n_tape} 스냅숏 {n_snap} · dhl 테이프 {n_dhl}봉 · 테이프 {} · 구멍 {:?}",
                    touched.len(),
                    if t.top >= hi - 1 { "따라옴".to_string() } else { format!("끝이 {}초 모자람", hi - 1 - t.top) },
                    t.holes
                );
            }
            Err(e) => {
                if lo < hi {
                    self.fp.holes.push((lo, hi));
                }
                line += &format!(" · 🔴테이프 못 읽음({e}) -- 그 공백은 비어 있다(구멍) · 1초 칸 스냅숏 {n_snap}");
            }
        }
        self.fp.holes.retain(|h| h.1 > win_lo);
        // ⑤ 되살린 1초 칸을 밀기로(붙어 있던 화면도 받게 -- 파이썬은 새로 연 화면만 받는다 supply_1s_restore 977).
        let newest = self.sec.keys().next_back().copied().unwrap_or(0);
        let rows: Vec<SecRow> =
            restored.into_iter().filter(|&s| newest - SUPPLY_1S_SECONDS < s && s < newest).filter_map(|s| sec_row(&self.sec, s)).collect();
        let up = (!rows.is_empty()).then(|| {
            let mut m = self.base(now_us);
            m.seconds = rows;
            m
        });
        (line, up)
    }

    /// 스냅숏 파일에 쓸 것(락 안에서 떠서 락 밖에서 쓴다). 합치기를 한 번 끝냈고 메우는 중이 아닐 때만(머리 주석).
    pub(crate) fn to_snap(&self, symbol: &str) -> Option<Snap> {
        if !self.rs.merged || self.rs.job.is_some() || self.last_ms == 0 {
            return None;
        }
        let (bucket, _) = self.fp.spec?;
        let lo = self.fp.window_lo()?;
        let bars = self.fp.bars.range(lo..).map(|(&b, bar)| (b, bar.cells.iter().map(|(&k, c)| (k, *c)).collect(), self.fp.dhl_now(&self.sec, b, bar))).collect();
        let sec = ring_floor(&self.sec).map_or(vec![], |f| self.sec.range(f..).map(|(&s, c)| (s, *c)).collect());
        let holes = self.fp.holes.iter().filter(|h| h.1 > lo).copied().collect();
        Some(Snap { v: SNAP_V, symbol: symbol.to_string(), bar_seconds: BAR_SECONDS, bucket, last_ms: self.last_ms, bars, sec, holes })
    }

    /// 스냅숏 되살리기(시작 때 · 수신 전). 봉은 통째로, 1초 칸은 옆에 둔다(merge 가 테이프 없는 초에만 쓴다).
    pub(crate) fn load_snap(&mut self, s: Snap) {
        self.fp.bars = s.bars.into_iter().map(|(b, cells, dhl)| (b, Bar { cells: cells.into_iter().collect(), dhl })).collect();
        self.fp.holes = s.holes;
        self.last_ms = s.last_ms;
        self.rs.snap_sec = s.sec.into_iter().collect();
    }
}

// ── 나머지 원천(머리 주석) ─────────────────────────────────────────────────
fn open_ro(path: &Path) -> Result<Connection, String> {
    let con = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX).map_err(|e| format!("{}: {e}", path.display()))?;
    con.busy_timeout(Duration::from_secs(10)).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(con)
}

/// (거래소 시각 ms, 롱 청산, 수량, 세 번째 값) 행.
type LiqRowDb = (i64, bool, f64, f64);

fn liq_query(path: &Path, sql: &str, params: impl rusqlite::Params, long_is: &str) -> Result<Vec<LiqRowDb>, String> {
    let con = open_ro(path)?;
    let err = |e: rusqlite::Error| format!("{}: {e}", path.display());
    let mut st = con.prepare(sql).map_err(err)?;
    let rows = st
        .query_map(params, |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Option<String>>(1)?.as_deref() == Some(long_is), f(r, 2)?, f(r, 3)?)))
        .map_err(err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)?;
    Ok(rows)
}

/// 바이낸스 강제청산(server.py liq_events_load): 최근 5,000건(시각 내림차순 LIMIT) → 시각 순 → 지금 − 660초 이후만. (ts, 롱, 수량, usd).
pub fn read_bn_liq(path: &Path, symbol: &str, now_ms: f64) -> Result<Vec<LiqRowDb>, String> {
    let rows = liq_query(
        path,
        "SELECT ts_ms, side, qty, usd FROM liquidations WHERE symbol = ? ORDER BY ts_ms DESC LIMIT ?",
        rusqlite::params![symbol.to_uppercase(), crate::supply::LIQ_RING as i64],
        "long",
    )?;
    let cut = now_ms - (SUPPLY_1S_SECONDS * 1000) as f64;
    Ok(rows.into_iter().rev().filter(|r| r.0 as f64 >= cut).collect())
}

/// Bybit 청산: 코어가 보일 창((지금 − 3 − 660, …])만 시각 순으로(server.py _bybit_liq_refresh 의 SQL · side Buy = 롱 청산). (ts, 롱, 수량, 가격).
pub fn read_bybit_liq(path: &Path, symbol: &str, now_ms: i64) -> Result<Vec<LiqRowDb>, String> {
    let lo = now_ms.div_euclid(1000) - crate::supply::BYBIT_LAG_S - SUPPLY_1S_SECONDS;
    liq_query(
        path,
        // 지금보다 늦은 행은 없다(실서버) -- 시험 DB 는 미래 행을 들고 있어 막는다. 지금 − 3초 뒤 행도 읽어 둔다(창이 그쪽으로 자란다 · 다시 오면 supply.rs 가 건너뜀).
        "SELECT ts_ms, side, qty, price FROM bybit_liquidations WHERE symbol = ? AND ts_ms >= ? AND ts_ms <= ? ORDER BY ts_ms",
        rusqlite::params![symbol.to_uppercase(), (lo + 1) * 1000, now_ms],
        "Buy",
    )
}

/// OKX 청산 [lo_ms, hi_ms)(okx_tape_footprint 193~199 의 SQL · pos_side long = 롱 청산). (ts, 롱, 수량(기초자산), 파산가).
pub fn read_okx_liq(path: &Path, inst: &str, lo_ms: i64, hi_ms: i64) -> Result<Vec<LiqRowDb>, String> {
    liq_query(
        path,
        "SELECT ts_ms, pos_side, sz_base, bk_px FROM okx_liquidations WHERE inst_id = ? AND ts_ms >= ? AND ts_ms < ? AND sz_base > 0 ORDER BY ts_ms",
        rusqlite::params![inst, lo_ms, hi_ms],
        "long",
    )
}

/// OKX 미결제약정 [t0 − 660, t0) 초(supply_1s_restore oi_rows · 시각 순이라 같은 초는 마지막 값).
pub fn read_okx_oi(path: &Path, inst: &str, t0_s: i64) -> Result<BTreeMap<i64, f64>, String> {
    let con = open_ro(path)?;
    let err = |e: rusqlite::Error| format!("{}: {e}", path.display());
    let mut st = con.prepare("SELECT ts_ms, oi_base FROM okx_oi WHERE inst = ? AND ts_ms >= ? AND ts_ms < ? ORDER BY ts_ms").map_err(err)?;
    let mut out = BTreeMap::new();
    let mut rows = st.query(rusqlite::params![inst, (t0_s - SUPPLY_1S_SECONDS) * 1000, t0_s * 1000]).map_err(err)?;
    while let Some(r) = rows.next().map_err(err)? {
        out.insert(r.get::<_, i64>(0).map_err(err)?.div_euclid(1000), f(r, 1).map_err(err)?);
    }
    Ok(out)
}

/// OKX 1초 칸 [t0 − 660, t0)(tape_seconds · 테이프 끝이 t0 − 1 에 닿을 때까지 기다린다). (칸, 테이프 끝 초).
pub fn read_okx_secs(path: &Path, inst: &str, t0_s: i64, tape_bucket: f64, tries: u32, wait: Duration) -> Result<(BTreeMap<i64, [f64; 7]>, i64), String> {
    let con = open_ro(path)?;
    let err = |e: rusqlite::Error| format!("{}: {e}", path.display());
    let mut top = 0;
    for i in 0..tries.max(1) {
        top = con.query_row(SQL_TOP, [inst], |r| r.get::<_, Option<i64>>(0)).map_err(err)?.unwrap_or(0);
        if top >= t0_s - 1 || i + 1 >= tries {
            break;
        }
        std::thread::sleep(wait);
    }
    let mut out = BTreeMap::new();
    let mut st = con.prepare(SQL_SECS).map_err(err)?;
    let mut rows = st.query(rusqlite::params![tape_bucket, inst, t0_s - SUPPLY_1S_SECONDS, t0_s]).map_err(err)?;
    while let Some(r) = rows.next().map_err(err)? {
        let mut c = [0.0; 7];
        for (j, x) in c.iter_mut().enumerate() {
            *x = f(r, 1 + j).map_err(err)?;
        }
        out.insert(r.get(0).map_err(err)?, c);
    }
    Ok((out, top))
}

impl Supply {
    /// 시작 때(수신 전): 바이낸스 강제청산 · Bybit 청산을 DB 에서 링에 넣는다.
    pub(crate) fn load_liqs(&mut self, bn: Vec<LiqRowDb>, bybit: Vec<LiqRowDb>) {
        for (ts_ms, long, qty, usd) in bn {
            crate::supply::push_capped(&mut self.liq, crate::supply::LiqEv { ts_ms, long, qty, usd });
        }
        for (ts_ms, long, qty, px) in bybit {
            let i = self.bybit.partition_point(|e| e.ts_ms <= ts_ms);
            self.bybit.insert(i, crate::supply::LiqEv { ts_ms, long, qty, usd: qty * px });
            self.rs.bybit_max_ms = self.rs.bybit_max_ms.max(ts_ms);
        }
    }

    /// 시작 뒤 첫 OKX 체결 앞(state.rs apply). 참 = 새 일감.
    pub(crate) fn begin_okx(&mut self, ts_ms: i64) -> bool {
        self.rs.okx_await = false;
        self.rs.okx_job = Some(ts_ms);
        true
    }

    pub(crate) fn begin_okx_oi(&mut self, ts_ms: i64) -> bool {
        self.rs.oi_await = false;
        self.rs.oi_job = Some(ts_ms.div_euclid(1000));
        true
    }

    /// OKX 1초 칸 · OKX 청산 합치기(락 안). 돌려주는 것 = 로그 한 줄 · 밀기.
    pub(crate) fn merge_okx(&mut self, secs: Result<(BTreeMap<i64, [f64; 7]>, i64), String>, liq: Result<Vec<LiqRowDb>, String>, now_us: i64) -> (String, Option<Supply1s>) {
        let Some(t0_ms) = self.rs.okx_job.take() else { return ("메울 OKX 일감 없음".into(), None) };
        let t0 = t0_ms.div_euclid(1000);
        let mut restored = vec![];
        let line_secs = match secs {
            Ok((got, top)) => {
                // setdefault(supply_1s_restore 1000) -- 실시간이 가진 초는 그대로
                for (s, v) in got {
                    if let std::collections::btree_map::Entry::Vacant(e) = self.okx.sec.entry(s) {
                        e.insert(v);
                        restored.push(s);
                    }
                }
                format!("1초 칸 {} · 테이프 {}", restored.len(), if top >= t0 - 1 { "따라옴".to_string() } else { format!("끝이 {}초 모자람", t0 - 1 - top) })
            }
            Err(e) => format!("🔴1초 칸 테이프 못 읽음({e}) -- [{}, {t0}) 비어 있음", t0 - SUPPLY_1S_SECONDS),
        };
        let line_liq = match liq {
            Ok(ev) => {
                // 링 앞에 -- 남은 자리만, 가장 새 것부터(deque appendleft 가 차면 가장 최신이 밀려나는 것을 피한 원본 958~963)
                let room = crate::supply::LIQ_RING.saturating_sub(self.okx_liq.len());
                let n = ev.len().min(room);
                for &(ts_ms, long, qty, px) in ev[ev.len() - n..].iter().rev() {
                    self.okx_liq.push_front(crate::supply::LiqEv { ts_ms, long, qty, usd: qty * px });
                }
                format!("청산 {n}건")
            }
            Err(e) => format!("🔴청산 못 읽음({e})"),
        };
        let top = crate::supply::newest(&self.okx.sec);
        let mut m = self.base(now_us);
        m.okx = restored.into_iter().filter(|&s| top - SUPPLY_1S_SECONDS < s && s < top).filter_map(|s| sec_row(&self.okx.sec, s)).collect();
        m.okx_liq = crate::supply::liq_rows(&self.okx_liq, |s| top - SUPPLY_1S_SECONDS < s && s < top);
        let up = (!m.okx.is_empty() || !m.okx_liq.is_empty()).then_some(m);
        (format!("OKX t0 {t0}: {line_secs} · {line_liq}"), up)
    }

    /// OKX 미결제약정 합치기(락 안).
    pub(crate) fn merge_okx_oi(&mut self, got: Result<BTreeMap<i64, f64>, String>, now_us: i64) -> (String, Option<Supply1s>) {
        let Some(t0) = self.rs.oi_job.take() else { return ("메울 OKX OI 일감 없음".into(), None) };
        match got {
            Ok(got) => {
                let mut restored = vec![];
                for (s, v) in got {
                    if let std::collections::btree_map::Entry::Vacant(e) = self.okx_oi.entry(s) {
                        e.insert(v);
                        restored.push((s, v));
                    }
                }
                let top = self.okx_oi.keys().next_back().copied().unwrap_or(0);
                let n = restored.len();
                let mut m = self.base(now_us);
                m.okx_oi = restored.into_iter().filter(|r| r.0 > top - SUPPLY_1S_SECONDS).collect();
                let up = (!m.okx_oi.is_empty()).then_some(m);
                (format!("OKX OI t0 {t0}: {n}초"), up)
            }
            Err(e) => (format!("OKX OI t0 {t0}: 🔴못 읽음({e}) -- [{}, {t0}) 비어 있음", t0 - SUPPLY_1S_SECONDS), None),
        }
    }
}
