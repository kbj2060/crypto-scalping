//! 오더플로우 호가 래스터(`scripts/live_orderflow_raster_collector_20260914.py`, 줄 번호는 그 파일) -- 대시보드 히트맵의 유일한 원천.
//! 화면이 1초마다 `read_window` 로 읽는 고정폭 파일 `<root>/<SYM>/<UTC 시>.f32`(표준 라이브러리만):
//!   헤더 32B `<4sHHfIqII` = "FLWR", ver 1, n_bins(기본 240), bin_size(기본 0.5, f32), flags 0, 시 시작 ms, dt 1000, 0
//!   행 16 + 4·n_bins B `<qif` + f32[n_bins] = 초 ms, bin_lo, mid(f32), 잔량(부호 = 방향: + 매수호가 · − 매도호가) · 위치 = 32 + 초 인덱스 × 행 길이
//!   칸 크기·칸 수 = 원본과 같은 환경변수 `OF_BIN_SIZE`·`OF_N_BINS`(main.rs) -- 칸 번호는 파이썬 `p // bin`(`py_floordiv`, floor(p / bin) 과 다르다).
//!   mid = NaN 인 행 = 그 초의 북이 무효(재동기 중·수집 중단) -- 보간하지 않는다(없는 유동성을 그리는 것).
//! 🔴바이트까지 같으려면 잔량 더하기가 **f32 로, 파이썬 dict 삽입 순서대로**여야 한다(numpy 2 NEP 50: float32 칸 += 파이썬 float = q 를 f32 로
//!   바꾼 뒤 f32 덧셈) -- 북은 삽입 순서를 지키는 맵(`Book`): 갱신은 제자리 · 지우고 다시 넣으면 맨 뒤 · 스냅숏 병합 = «범위 밖 기존» 뒤에 «스냅숏».
//! 북 동기화(바이낸스 USD-M diff depth 규약)·재동기(드리프트 1% · 30분)·가지치기(±4%, 60틱마다)·무효 초는 원본 그대로.
//! 스냅숏 REST(`depth?limit=1000`, 가중치 20 -- 시작·30분·드리프트·단절 때, 평소 시간당 2회 안팎)는 main.rs `RTCOLLECT_REST=1` 일 때만 --
//!   끄면 북을 못 세워 **모든 행이 무효**다(그림자도 켜야 한다 -- SHADOW.md). 🔧밴 파일을 지킨다(원본은 모른다).
//! 보관(14일 지난 파일 → parquet·duckdb)과 지우기는 옮기지 않았다(파이썬 cron 몫 -- REMAINING.md).
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const WS: &str = "wss://fstream.binance.com/ws/{symbol}@depth@500ms";
pub const SNAPSHOT: &str = "https://fapi.binance.com/fapi/v1/depth?symbol={SYMBOL}&limit=1000";
pub const BIN_SIZE: f64 = 0.5; // 기본값(OF_BIN_SIZE)
pub const N_BINS: usize = 240; // 기본값(OF_N_BINS)
pub const PRUNE_PCT: f64 = 4.0; // 이 밖의 레벨은 버린다(래스터가 ±2.4% 라 여유 있게)
pub const RESYNC_DRIFT_PCT: f64 = 1.0; // 마지막 스냅숏 mid 대비 이만큼 밀리면 재동기(가지치기 구멍 방지)
pub const RESYNC_INTERVAL_SEC: f64 = 1800.0; // 주기적 재동기(누적 드리프트·버그 자가치유)
pub const STALE_SEC: f64 = 5.0; // 마지막 depth 이벤트가 이보다 오래되면 그 초는 무효
pub const RECV_TIMEOUT_S: f64 = 35.0; // ping 이 살아 있어도 스트림만 멎는 경우가 있었다 -- recv 를 묶는다
const HEADER: usize = 32;

/// 파이썬 float `a // b`(floatobject.c float_floor_div: fmod 로 나머지를 빼고 나눈 뒤 반올림 보정) -- `(a / b).floor()` 와는
/// 몫이 정수 바로 밑으로 반올림될 때 다르다(칸 0.0002·0.02·0.2 에서 0.0001·0.01 틱 가격의 약 3~28%).
pub fn py_floordiv(a: f64, b: f64) -> f64 {
    let m = a % b;
    let mut div = (a - m) / b;
    if m != 0.0 && (b < 0.0) != (m < 0.0) {
        div -= 1.0;
    }
    if div != 0.0 {
        let f = div.floor();
        if div - f > 0.5 { f + 1.0 } else { f }
    } else {
        (0.0f64).copysign(a / b)
    }
}

/// 삽입 순서를 지키는 float→float 맵(파이썬 dict). 지운 자리는 비워 두고 반쯤 비면 순서대로 다시 채운다.
#[derive(Default, Clone)]
pub struct Book {
    slots: Vec<Option<(f64, f64)>>,
    at: HashMap<u64, usize>,
}

fn k(p: f64) -> u64 {
    (p + 0.0).to_bits() // -0.0 == 0.0 (파이썬 dict 도 같은 열쇠)
}

impl Book {
    pub fn set(&mut self, p: f64, q: f64) {
        match self.at.get(&k(p)) {
            Some(&i) => self.slots[i] = Some((p, q)),
            None => {
                self.at.insert(k(p), self.slots.len());
                self.slots.push(Some((p, q)));
            }
        }
    }
    pub fn pop(&mut self, p: f64) {
        if let Some(i) = self.at.remove(&k(p)) {
            self.slots[i] = None;
            if self.at.len() * 2 < self.slots.len() && self.slots.len() > 64 {
                let v: Vec<(f64, f64)> = self.iter().collect();
                *self = Book::default();
                for (p, q) in v {
                    self.set(p, q);
                }
            }
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.slots.iter().flatten().copied()
    }
    pub fn is_empty(&self) -> bool {
        self.at.is_empty()
    }
    pub fn len(&self) -> usize {
        self.at.len()
    }
    fn retain(&self, f: impl Fn(f64) -> bool) -> Book {
        let mut b = Book::default();
        for (p, q) in self.iter().filter(|x| f(x.0)) {
            b.set(p, q);
        }
        b
    }
    fn max_key(&self) -> f64 {
        self.iter().map(|x| x.0).fold(f64::NEG_INFINITY, f64::max)
    }
    fn min_key(&self) -> f64 {
        self.iter().map(|x| x.0).fold(f64::INFINITY, f64::min)
    }
}

fn f(v: &Value) -> Result<f64, String> {
    crate::py_float(v).map_err(|e| e.0)
}

fn i(v: &Value) -> Result<i64, String> {
    crate::py_int(v).map_err(|e| e.0)
}

/// `seq_decision`(:109-121): apply | skip | resnapshot. 예외 = 그 연결을 접는다(원본 int(ev["u"]) 등).
pub fn seq_decision(synced: bool, last_u: i64, ev: &serde_json::Map<String, Value>) -> Result<&'static str, String> {
    let get = |k: &str| ev.get(k).ok_or_else(|| format!("KeyError: '{k}'"));
    let (u, uu) = (i(get("u")?)?, i(get("U")?)?);
    let pu = match ev.get("pu") {
        Some(v) => i(v)?,
        None => -1,
    };
    Ok(if !synced {
        if u < last_u {
            "skip"
        } else if uu > last_u {
            "resnapshot"
        } else {
            "apply"
        }
    } else if pu == last_u {
        "apply"
    } else {
        "resnapshot"
    })
}

/// `next_tick`(:124-135): 목표 초를 센다 -- 1초 안으로 늦었으면 지금 초를 바로 · 더 밀렸으면 현재 초로 따라잡는다(건너뛴 초는 writer 가 무효행으로).
pub fn next_tick(target: i64, now: f64) -> i64 {
    let (nxt, now_sec) = (target + 1, now as i64);
    if nxt >= now_sec { nxt } else { now_sec }
}

/// `_RasterWriter`(:138-183): 시각별 파일에 초당 한 행. 수집이 멈췄던 초는 무효행으로 메운다(오프셋 산식).
pub struct Writer {
    dir: PathBuf,
    bin: f64,
    n: usize,
    hour_ms: Option<i64>,
    fh: Option<File>,
}

fn row_bytes(n: usize, sec_ms: i64, bin_lo: i32, mid: f32, qty: Option<&[f32]>) -> Vec<u8> {
    let mut b = Vec::with_capacity(16 + 4 * n);
    b.extend_from_slice(&sec_ms.to_le_bytes());
    b.extend_from_slice(&bin_lo.to_le_bytes());
    b.extend_from_slice(&mid.to_le_bytes());
    match qty {
        Some(q) => q.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes())),
        None => b.resize(16 + 4 * n, 0),
    }
    b
}

impl Writer {
    pub fn new(root: &Path, symbol: &str, bin: f64, n: usize) -> Self {
        Self { dir: root.join(symbol.to_uppercase()), bin, n, hour_ms: None, fh: None }
    }

    fn open(&mut self, hour_ms: i64) -> std::io::Result<()> {
        self.fh = None;
        std::fs::create_dir_all(&self.dir)?;
        let path = self.dir.join(format!("{}.f32", crate::depth::hour_key(hour_ms).unwrap_or_default()));
        let fresh = std::fs::metadata(&path).map(|m| m.len() < HEADER as u64).unwrap_or(true);
        let mut fh = OpenOptions::new().read(true).write(true).create(true).truncate(fresh).open(&path)?;
        if fresh {
            let mut h = Vec::with_capacity(HEADER);
            h.extend_from_slice(b"FLWR");
            h.extend_from_slice(&1u16.to_le_bytes());
            h.extend_from_slice(&(self.n as u16).to_le_bytes());
            h.extend_from_slice(&(self.bin as f32).to_le_bytes()); // struct 'f' = f64 → f32 가까운 쪽(as f32 와 같다)
            h.extend_from_slice(&0u32.to_le_bytes());
            h.extend_from_slice(&hour_ms.to_le_bytes());
            h.extend_from_slice(&1000u32.to_le_bytes());
            h.extend_from_slice(&0u32.to_le_bytes());
            fh.write_all(&h)?;
        }
        (self.fh, self.hour_ms) = (Some(fh), Some(hour_ms));
        Ok(())
    }

    /// `write`(:163-177). qty None = 무효행(mid NaN · 잔량 0).
    pub fn write(&mut self, sec_ms: i64, bin_lo: i32, mid: f64, qty: Option<&[f32]>) -> std::io::Result<()> {
        let row = (16 + 4 * self.n) as u64;
        let hour_ms = sec_ms - sec_ms.rem_euclid(3_600_000);
        if Some(hour_ms) != self.hour_ms {
            self.open(hour_ms)?;
        }
        let target = HEADER as u64 + ((sec_ms - hour_ms) / 1000) as u64 * row;
        let fh = self.fh.as_mut().unwrap();
        let size = fh.seek(SeekFrom::End(0))?;
        if size < target {
            // 프로세스가 죽어 있던 초들 -- 무효행으로(보간 아님). 🔴원본 그대로: 끝 행이 반쪽이면 (target − size)//ROW 행만 덧붙여 그 뒤가 어긋난다.
            let pad = row_bytes(self.n, 0, 0, f32::NAN, None);
            for _ in 0..(target - size) / row {
                fh.write_all(&pad)?;
            }
        }
        fh.seek(SeekFrom::Start(target))?;
        fh.write_all(&row_bytes(self.n, sec_ms, bin_lo, mid as f32, qty))?;
        fh.flush()
    }
}

/// 수집기 상태(`OrderflowRasterCollector` :535-711).
pub struct Raster {
    pub bids: Book,
    pub asks: Book,
    pub synced: bool,
    pub last_u: i64,
    pub last_event_ts: f64,
    pub snap_mid: f64,
    pub snap_ts: f64,
    pub want_resync: bool,
    need_snapshot: bool,
    pub delay: f64,
    pub writer: Writer,
    tick: u64,
    pub rows_written: u64,
    pub rows_invalid: u64,
}

impl Raster {
    pub fn new(root: &Path, symbol: &str, bin: f64, n: usize) -> Self {
        Self { bids: Book::default(), asks: Book::default(), synced: false, last_u: 0, last_event_ts: 0.0, snap_mid: 0.0, snap_ts: 0.0,
               want_resync: false, need_snapshot: true, delay: 3.0, writer: Writer::new(root, symbol, bin, n), tick: 0, rows_written: 0, rows_invalid: 0 }
    }

    /// `_merge_snapshot`(:561-571): 스냅숏 범위 **안**만 교체 · 바깥 누적분은 남긴다(재시작 직후 ±0.43% 밖은 diff 가 채운다 -- 덮으면 스스로 지운다).
    pub fn merge_snapshot(&mut self, snap: &Value, wall: f64) -> Result<i64, String> {
        let side = |k: &str| -> Result<Book, String> {
            let mut b = Book::default();
            let rows: &[Value] = match snap.get(k) {
                Some(Value::Array(a)) => a,
                None => &[],
                Some(_) => return Err("TypeError".into()),
            };
            for r in rows {
                let (p, q) = (f(r.get(0).ok_or("IndexError")?)?, f(r.get(1).ok_or("IndexError")?)?);
                if q > 0.0 {
                    b.set(p, q);
                }
            }
            Ok(b)
        };
        let (bids, asks) = (side("bids")?, side("asks")?);
        if !bids.is_empty() && !asks.is_empty() {
            let (lo, hi) = (bids.min_key(), asks.max_key());
            let mut nb = self.bids.retain(|p| p < lo);
            bids.iter().for_each(|(p, q)| nb.set(p, q));
            let mut na = self.asks.retain(|p| p > hi);
            asks.iter().for_each(|(p, q)| na.set(p, q));
            (self.bids, self.asks) = (nb, na);
            self.snap_mid = (bids.max_key() + asks.min_key()) / 2.0;
        }
        self.snap_ts = wall;
        Ok(match snap.get("lastUpdateId") {
            Some(v) => i(v)?,
            None => 0,
        })
    }

    fn apply(&mut self, ev: &serde_json::Map<String, Value>) -> Result<(), String> {
        for (side, bid) in [("b", true), ("a", false)] {
            let rows: &[Value] = match ev.get(side) {
                Some(Value::Array(a)) => a,
                None => &[],
                Some(_) => return Err("TypeError".into()),
            };
            for r in rows {
                let (p, q) = (f(r.get(0).ok_or("IndexError")?)?, f(r.get(1).ok_or("IndexError")?)?);
                let book = if bid { &mut self.bids } else { &mut self.asks };
                if q == 0.0 { book.pop(p) } else { book.set(p, q) }
            }
        }
        Ok(())
    }

    fn prune(&mut self, mid: f64) {
        let (lo, hi) = (mid * (1.0 - PRUNE_PCT / 100.0), mid * (1.0 + PRUNE_PCT / 100.0));
        self.bids = self.bids.retain(|p| p >= lo);
        self.asks = self.asks.retain(|p| p <= hi);
    }

    /// 다음 depth 이벤트가 스냅숏을 부르는가(부르는 쪽이 REST 를 미리 받아 둔다 -- 원본은 await 하는 동안 틱이 계속 돈다).
    pub fn wants_snapshot(&self) -> bool {
        self.need_snapshot || self.want_resync
    }

    pub fn on_connect(&mut self) {
        (self.synced, self.need_snapshot) = (false, true);
    }

    /// WS 메시지 하나(:600-627). `snap` = 스냅숏을 받아야 할 때 부른다(Err = 그 연결을 접는다 -- 원본 raise_for_status·시간초과).
    /// Err = 연결을 접는다(대기 `delay` 만큼 쉬고 다시 · 1.5배씩 60초까지).
    pub fn on_text(&mut self, raw: &str, wall: f64, snap: &mut dyn FnMut() -> Result<Value, String>) -> Result<(), String> {
        let ev: Value = serde_json::from_str(raw).map_err(|e| format!("JSONDecodeError: {e}"))?;
        let Value::Object(ev) = ev else { return Err("AttributeError: .get".into()) };
        if ev.get("e").and_then(Value::as_str) != Some("depthUpdate") {
            return Ok(());
        }
        if self.need_snapshot || self.want_resync {
            // 스냅숏을 받는 동안 도착하는 이벤트는 WS 큐에 쌓인다(유실 아님) -- 이 이벤트는 버린다(원본 continue)
            let s = snap()?;
            self.last_u = self.merge_snapshot(&s, wall)?;
            (self.synced, self.need_snapshot, self.want_resync) = (false, false, false);
            return Ok(());
        }
        match seq_decision(self.synced, self.last_u, &ev)? {
            "skip" => return Ok(()),
            "resnapshot" => {
                eprintln!("시퀀스 단절(last_u={} U={:?} pu={:?}) -- 재동기", self.last_u, ev.get("U"), ev.get("pu"));
                if !self.synced {
                    // 방금 받은 스냅숏이 스트림보다 낡았다 -- 바로 다시 받지 말고 물러난다(IP 공유: 주문 경로)
                    return Err("RuntimeError: snapshot older than stream".into());
                }
                (self.synced, self.need_snapshot) = (false, true);
                return Ok(());
            }
            _ => {}
        }
        self.apply(&ev)?;
        self.last_u = i(&ev["u"])?;
        self.synced = true;
        self.delay = 3.0; // 동기된 뒤에만 물러남 초기화
        self.last_event_ts = wall;
        Ok(())
    }

    /// 연결 끝(예외) -- 반환 = 쉴 초(:629-632).
    pub fn on_error(&mut self) -> f64 {
        let d = self.delay;
        self.delay = (d * 1.5).min(60.0);
        d
    }

    /// `_row`(:633-652): 유효하면 (bin_lo, mid, 잔량).
    pub fn row(&self, wall: f64) -> Option<(i32, f64, Vec<f32>)> {
        if !self.synced || self.bids.is_empty() || self.asks.is_empty() || wall - self.last_event_ts > STALE_SEC {
            return None;
        }
        let (bb, ba) = (self.bids.max_key(), self.asks.min_key());
        if !(ba > bb && bb > 0.0) {
            return None;
        }
        let mid = (bb + ba) / 2.0;
        let (bin, n) = (self.writer.bin, self.writer.n);
        let bin_lo = py_floordiv(mid, bin) as i64 - (n / 2) as i64;
        let mut arr = vec![0f32; n];
        for (p, q) in self.bids.iter() {
            let j = py_floordiv(p, bin) as i64 - bin_lo;
            if (0..n as i64).contains(&j) {
                arr[j as usize] += q as f32; // numpy 2: float32 칸 += 파이썬 float = f32 덧셈
            }
        }
        for (p, q) in self.asks.iter() {
            let j = py_floordiv(p, bin) as i64 - bin_lo;
            if (0..n as i64).contains(&j) {
                arr[j as usize] -= q as f32; // 부호가 방향 -- 스프레드가 걸친 한 빈만 상계된다
            }
        }
        Some((bin_lo as i32, mid, arr))
    }

    /// 한 틱(`_raster_loop` 몸통 :665-683): 행 쓰기 · 재동기 판정 · 60틱마다 가지치기. `wall` = 그 틱이 깬 시각.
    pub fn tick(&mut self, target: i64, wall: f64) -> std::io::Result<()> {
        let sec_ms = target * 1000;
        match self.row(wall) {
            None => {
                self.writer.write(sec_ms, 0, f64::NAN, None)?;
                self.rows_invalid += 1;
            }
            Some((lo, mid, arr)) => {
                self.writer.write(sec_ms, lo, mid, Some(&arr[..]))?;
                self.rows_written += 1;
                if (mid - self.snap_mid).abs() / mid * 100.0 > RESYNC_DRIFT_PCT || wall - self.snap_ts > RESYNC_INTERVAL_SEC {
                    self.want_resync = true;
                }
                if self.tick % 60 == 0 {
                    self.prune(mid);
                }
            }
        }
        self.tick += 1;
        Ok(())
    }
}

/// 정답지 입력 먹이기. 사건 = [벽시계 초, 종류, 본문]: 'o' 접속 · 'f' 접속 실패 · 't' 메시지 · 'w' 35초 수신 시한 · 'c' 끊김.
/// 틱 = 시작 초 + 1 부터 매초 -- 사건 시각 t 앞(목표 ≤ t)의 틱을 먼저 돈다(정답지 생성기와 같은 규칙). 스냅숏 = `snaps`(본문 글자 · null = 실패).
pub fn replay(t0: f64, events: &[Value], symbol: &str, root: &Path, snaps: &[Value], bin: f64, n: usize) -> std::io::Result<Raster> {
    let mut r = Raster::new(root, symbol, bin, n);
    let mut target = t0 as i64 + 1;
    let mut k = 0;
    for e in events {
        let (t, kind, body) = (e[0].as_f64().unwrap(), e[1].as_str().unwrap(), e[2].as_str().unwrap_or(""));
        while (target as f64) <= t {
            r.tick(target, target as f64)?;
            target += 1;
        }
        let mut snap = || {
            let s = snaps.get(k).and_then(Value::as_str).map(|b| serde_json::from_str::<Value>(b).map_err(|e| e.to_string()));
            k += 1;
            s.unwrap_or_else(|| Err("ClientResponseError: 가짜 실패".into()))
        };
        match kind {
            "o" => r.on_connect(),
            "t" => {
                if r.on_text(body, t, &mut snap).is_err() {
                    r.on_error();
                }
            }
            _ => {
                r.on_error();
            }
        }
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    /// 파이썬 `a // b` 의 실제 값(python3 -c 로 확인 -- 1.5 // 0.0002 = 7499.0 · 150.0 // 0.02 = 7499.0 · 150.4 // 0.2 = 751.0 · 1.0 // 0.1 = 9.0).
    #[test]
    fn floordiv_matches_python() {
        for (a, b, want) in [(1.5, 0.0002, 7499.0), (150.0, 0.02, 7499.0), (150.4, 0.2, 751.0), (1.0, 0.1, 9.0), (2500.25, 0.5, 5000.0),
                             (-1.0, 0.3, -4.0), (0.0, 0.5, 0.0)] {
            assert_eq!(super::py_floordiv(a, b), want, "{a} // {b}");
        }
    }
}
