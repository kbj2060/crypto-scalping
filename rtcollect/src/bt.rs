//! 최우선 호가 시각 파일 `.bt`(`scripts/live_book_ticker_collector_20260914.py` `HourFile`, 줄 번호는 그 파일 -- HL 호가 수집기가 import 해 쓴다).
//!   `<root>/<SYM>/<UTC 시>.bt` = 헤더 32B `<4sHHqQII`("BTKR", ver 1, 행 32, 시 시작 ms, first_u, 0, 0) + 행 32B `<qdfdf`(ts_ms, 매수가, 매수량 f32, 매도가, 매도량 f32).
//!   시각이 바뀌면 지난 파일을 닫고 gzip(레벨 6 · 성공한 뒤에만 원본을 지운다). 재연결 뒤 같은 시각이면 헤더 없이 이어 쓴다.
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

pub const FLUSH_ROWS: u64 = 2000;
pub const FLUSH_S: f64 = 30.0;

pub struct HourFile {
    dir: PathBuf,
    hour: Option<String>,
    fh: Option<BufWriter<File>>,
    rows: u64,
    flushed: f64,
    /// 압축 스레드(depth.rs HourFile 과 같은 꼴 -- 수신 고리를 막지 않는다 · 원본도 스레드)
    zips: Vec<std::thread::JoinHandle<()>>,
}

impl HourFile {
    pub fn new(root: &Path, symbol: &str, now: f64) -> std::io::Result<Self> {
        let dir = root.join(symbol.to_uppercase());
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir, hour: None, fh: None, rows: 0, flushed: now, zips: vec![] })
    }

    /// `write`(:93-118). now = 단조 초(플러시 시각 -- 바이트에는 안 남는다).
    pub fn write(&mut self, ts_ms: i64, bid: f64, bq: f64, ask: f64, aq: f64, first_u: i64, now: f64) -> std::io::Result<()> {
        let key = crate::depth::hour_key(ts_ms).unwrap_or_default();
        if Some(&key) != self.hour.as_ref() || self.fh.is_none() {
            if Some(&key) != self.hour.as_ref() {
                self.close(true);
            }
            let path = self.dir.join(format!("{key}.bt"));
            let new = !path.exists();
            let mut fh = BufWriter::new(OpenOptions::new().create(true).append(true).open(&path)?);
            if new {
                let mut h = Vec::with_capacity(32);
                h.extend_from_slice(b"BTKR");
                h.extend_from_slice(&1u16.to_le_bytes());
                h.extend_from_slice(&32u16.to_le_bytes());
                h.extend_from_slice(&(ts_ms.div_euclid(3_600_000) * 3_600_000).to_le_bytes());
                h.extend_from_slice(&(first_u as u64).to_le_bytes());
                h.extend_from_slice(&[0u8; 8]);
                fh.write_all(&h)?;
            }
            (self.fh, self.hour, self.rows) = (Some(fh), Some(key), 0);
        }
        let fh = self.fh.as_mut().unwrap();
        let mut r = [0u8; 32];
        r[..8].copy_from_slice(&ts_ms.to_le_bytes());
        r[8..16].copy_from_slice(&bid.to_le_bytes());
        r[16..20].copy_from_slice(&(bq as f32).to_le_bytes()); // struct 'f' = 가까운 f32
        r[20..28].copy_from_slice(&ask.to_le_bytes());
        r[28..32].copy_from_slice(&(aq as f32).to_le_bytes());
        fh.write_all(&r)?;
        self.rows += 1;
        if self.rows % FLUSH_ROWS == 0 || now - self.flushed >= FLUSH_S {
            fh.flush()?;
            self.flushed = now;
        }
        Ok(())
    }

    /// `close`(:120-129): 시각(hour)은 남긴다. compress = 그 시 파일을 gzip(스레드 -- 원본과 같다).
    pub fn close(&mut self, compress: bool) {
        if let Some(mut f) = self.fh.take() {
            let _ = f.flush();
        }
        if let (true, Some(h)) = (compress, &self.hour) {
            let p = self.dir.join(format!("{h}.bt"));
            if p.exists() {
                self.zips.retain(|z| !z.is_finished());
                self.zips.push(std::thread::spawn(move || crate::depth::gzip_done(&p)));
            }
        }
    }

    /// 압축 스레드가 끝날 때까지(종료·시험).
    pub fn join(&mut self) {
        for z in self.zips.drain(..) {
            let _ = z.join();
        }
    }
}
