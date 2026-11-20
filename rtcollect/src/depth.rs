//! 주문단위 호가 변경(diff depth) 수집기(`scripts/live_depth_diff_collector_20260915.py`, 줄 번호는 그 파일) → 시각별 JSONL `<root>/<SYM>/<UTC 시>.jsonl`,
//! 시각이 바뀌면 지난 파일을 `.jsonl.gz`(compresslevel 6)로 압축하고 **압축이 성공한 뒤에만** 원본을 지운다.
//! 읽는 곳: live_ob5m(Source·replay) · live_bookdepth_feat(→ 전진 fwd_dirshort) · live_eth_iceberg. 통과 기준 = **압축을 푼 내용이 바이트까지 같다**.
//! · 줄 = 거래소 WS 원문 프레임 그대로(수집기가 해석하면 그 순간 원본이 사라진다) · 새 파일의 첫 줄 = REST 스냅숏 `{"_snapshot":…,"at_ms":…}`
//!   (diff 는 증분이라 그것만으론 절대 깊이를 못 만든다) · `pu` 가 직전 `u` 와 다르면 그 앞에 `{"_gap":true,"expected_pu":…,"got_pu":…}`.
//! 🔴스냅숏 REST(`/fapi/v1/depth?limit=1000`, 가중치 20, 시각마다 1회)는 기본 꺼짐(main.rs `RTCOLLECT_REST=1`) -- 끄면 첫 줄이
//!   `{"_snapshot":null,…}`(원본이 REST 실패 때 쓰는 것과 같다) = 읽는 쪽이 그 시 파일로 북을 못 세운다(«no_snapshot»). 전환 때는 켠다.
//! 줄 단위로 같게 쓰는 것이 핵심이라 압축 바이트(헤더의 파일 이름·시각)는 원본과 다르다 -- 읽는 쪽은 gzip 으로 풀어 읽는다.
use crate::{ms, py_dumps, py_eq, py_int, Reconnect};
use serde_json::{json, Value};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

pub const WS: &str = "wss://fstream.binance.com/ws/{symbol}@depth@100ms";
pub const REST: &str = "https://fapi.binance.com/fapi/v1/depth?symbol={SYM}&limit=1000";
/// 크래시 때 잃는 양을 300행(≈40초)으로 묶는다(:110).
pub const FLUSH_ROWS: usize = 300;

/// UTC 시 열쇠 `YYYY-MM-DDTHH`(datetime.fromtimestamp(ts/1000, utc).strftime). 범위 밖이면 None(원본 ValueError·OverflowError).
pub fn hour_key(ts_ms: i64) -> Option<String> {
    let secs = ts_ms.div_euclid(1000);
    let (days, h) = (secs.div_euclid(86_400), secs.rem_euclid(86_400) / 3600);
    // days → (y, m, d): Howard Hinnant civil_from_days
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    (1..=9999).contains(&y).then(|| format!("{y:04}-{m:02}-{d:02}T{h:02}"))
}

/// 끝난 시 파일 압축(`_gzip_done` :58-72) -- 원본은 압축 성공(크기 > 0) 뒤에만 지운다. 실패하면 원본 유지.
pub fn gzip_done(path: &Path) {
    let gz = PathBuf::from(format!("{}.gz", path.display()));
    let r = (|| -> std::io::Result<()> {
        let mut fi = File::open(path)?;
        let mut fo = flate2::write::GzEncoder::new(File::create(&gz)?, flate2::Compression::new(6));
        std::io::copy(&mut fi, &mut fo)?;
        fo.finish()?.sync_all()
    })();
    match r {
        Ok(()) if std::fs::metadata(&gz).map(|m| m.len() > 0).unwrap_or(false) => {
            let _ = std::fs::remove_file(path);
        }
        Ok(()) => {}
        Err(e) => eprintln!("압축 실패 {}: {e} -- 원본 유지", path.display()),
    }
}

/// `HourFile`(:75-123): 시각별 파일 하나. 시각이 바뀌면 닫고(압축) 새로 연다 · 재접속(close 뒤 같은 시)이면 이어 쓴다.
pub struct HourFile {
    dir: PathBuf,
    hour: Option<String>,
    fh: Option<BufWriter<File>>,
    rows: usize,
    /// 압축 스레드(시험은 끝에 기다린다).
    pub zips: Vec<std::thread::JoinHandle<()>>,
}

impl HourFile {
    pub fn new(root: &Path, symbol: &str) -> std::io::Result<Self> {
        let dir = root.join(symbol.to_uppercase());
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir, hour: None, fh: None, rows: 0, zips: vec![] })
    }

    /// `snap` = 새 파일일 때만 부른다 → 스냅숏(JSON 값, 실패면 None). `wall` = at_ms 의 벽시계.
    pub fn write(&mut self, ts_ms: i64, line: &str, wall: f64, snap: &mut dyn FnMut() -> Option<Value>) -> Result<(), Reconnect> {
        let key = hour_key(ts_ms).ok_or_else(|| Reconnect("ValueError: year out of range".into()))?;
        if self.hour.as_deref() != Some(key.as_str()) || self.fh.is_none() {
            if self.hour.as_deref() != Some(key.as_str()) {
                self.close(true);
            }
            let path = self.dir.join(format!("{key}.jsonl"));
            let new = !path.exists();
            let f = OpenOptions::new().create(true).append(true).open(&path).map_err(|e| Reconnect(format!("OSError: {e}")))?;
            let mut w = BufWriter::new(f);
            if new {
                let s = snap();
                let head = py_dumps(&json!({"_snapshot": s, "at_ms": ms(wall)}));
                writeln!(w, "{head}").map_err(|e| Reconnect(format!("OSError: {e}")))?;
            }
            (self.fh, self.hour, self.rows) = (Some(w), Some(key), 0);
        }
        let fh = self.fh.as_mut().unwrap();
        writeln!(fh, "{line}").map_err(|e| Reconnect(format!("OSError: {e}")))?;
        self.rows += 1;
        if self.rows % FLUSH_ROWS == 0 {
            let _ = fh.flush();
        }
        Ok(())
    }

    /// `close`(:115-123). compress = 시각이 바뀌었다 → 그 파일을 압축(스레드).
    pub fn close(&mut self, compress: bool) {
        if let Some(mut fh) = self.fh.take() {
            let _ = fh.flush();
        }
        if let (true, Some(h)) = (compress, &self.hour) {
            let p = self.dir.join(format!("{h}.jsonl"));
            if p.exists() {
                self.zips.push(std::thread::spawn(move || gzip_done(&p)));
            }
        }
    }

    pub fn join(&mut self) {
        for z in self.zips.drain(..) {
            let _ = z.join();
        }
    }
}

/// `run()`(:126-160)의 상태.
pub struct DepthCollector {
    pub out: HourFile,
    prev_u: Option<Value>,
    pub frames: u64,
    pub gaps: u64,
}

impl DepthCollector {
    pub fn new(root: &Path, symbol: &str) -> std::io::Result<Self> {
        Ok(Self { out: HourFile::new(root, symbol)?, prev_u: None, frames: 0, gaps: 0 })
    }

    /// 접속 직후(:136). 🔴원본 그대로: 재접속 사이 구멍은 `_gap` 으로 안 남는다(prev_u 를 비우므로) -- 읽는 쪽이 스냅숏 동기화에서 안다.
    pub fn on_connect(&mut self) {
        self.prev_u = None;
    }

    /// 프레임 하나(:137-146). Err = 그 연결을 접는다(깨진 JSON·사전 아님·시각을 못 읽음).
    pub fn on_text(&mut self, raw: &str, wall: f64, snap: &mut dyn FnMut() -> Option<Value>) -> Result<(), Reconnect> {
        let d: Value = serde_json::from_str(raw).map_err(|e| Reconnect(format!("JSONDecodeError: {e}")))?;
        let Value::Object(o) = &d else { return Err(Reconnect("AttributeError: .get".into())) };
        // int(d.get("T") or d.get("E") or time.time() * 1000)
        let ts = match o.get("T").filter(|v| crate::py_truthy(v)).or_else(|| o.get("E").filter(|v| crate::py_truthy(v))) {
            Some(v) => py_int(v)?,
            None => ms(wall),
        };
        let pu = o.get("pu").cloned().unwrap_or(Value::Null);
        if let Some(prev) = &self.prev_u {
            if !py_eq(&pu, prev) {
                self.gaps += 1; // 구멍은 숨기지 않고 센다 + 파일에 표시
                let g = py_dumps(&json!({"_gap": true, "expected_pu": prev, "got_pu": pu}));
                self.out.write(ts, &g, wall, snap)?;
            }
        }
        // prev_u = d.get("u") -- 없거나 null 이면 다음 프레임은 대조하지 않는다
        self.prev_u = o.get("u").filter(|v| !v.is_null()).cloned();
        self.out.write(ts, raw, wall, snap)?;
        self.frames += 1;
        Ok(())
    }

    /// 예외로 끊김(:157-160): 파일을 닫는다(압축 없음 -- 같은 시에 다시 붙으면 이어 쓴다).
    pub fn on_error(&mut self) {
        self.out.close(false);
    }
}

/// 정답지 입력 먹이기. 사건 = [벽시계 초, 종류, 본문]: 'o' 접속 · 'f' 접속 실패 · 't' 프레임 · 'c' 예외로 끊김 · 'e' 깨끗한 닫힘(원본 `async for` 가
/// 그냥 끝나 곧바로 다시 붙는다 -- 파일을 닫지 않는다). 스냅숏은 `snaps`(REST 본문 글자 · null = 실패)를 차례로. 끝나면 파일을 닫고 압축을 기다린다.
pub fn replay(events: &[Value], symbol: &str, root: &Path, snaps: &[Value]) -> std::io::Result<()> {
    let mut c = DepthCollector::new(root, symbol)?;
    let mut k = 0;
    // 스냅숏 = REST 본문 글자(null = 실패) -- 열쇠 순서를 받은 대로 읽어 다시 쓴다(원본 json.dumps(json.loads(본문)))
    let mut snap = || {
        let s = snaps.get(k).and_then(Value::as_str).and_then(|b| serde_json::from_str(b).ok());
        k += 1;
        s
    };
    for e in events {
        let (t, kind, body) = (e[0].as_f64().unwrap(), e[1].as_str().unwrap(), e[2].as_str().unwrap_or(""));
        match kind {
            "o" => c.on_connect(),
            "t" => {
                if c.on_text(body, t, &mut snap).is_err() {
                    c.on_error();
                }
            }
            "e" => {}
            _ => c.on_error(), // 'c' · 'f'
        }
    }
    c.out.close(false);
    c.out.join();
    Ok(())
}
