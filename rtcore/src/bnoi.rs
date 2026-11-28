//! 바이낸스 선물 미결제약정 1초 링 -- 원본 coin_flow.py collect_oi_1s(681~705) 와 같은 규칙:
//! 수집기가 쓰는 hot `binance_ctx.sqlite` 의 `oi_1s` 꼬리를 0.5초마다(마지막 ms − 30초부터 겹쳐) 읽어 초 → 마지막 값,
//! 읽은 행이 있을 때만 (지금 − 660초) 앞 초를 지운다. 거래소엔 묻지 않는다(읽기 전용 SQLite · 바깥 REST 0).
//! 밀기 = 링 꼭대기보다 새 초만(원본 화면 커서 sinceOi 와 같다 -- 이미 받은 초를 늦은 stamp 가 고쳐도 다시 안 보낸다) ·
//! 스냅숏 = supply_1s_payload(sinceOi=0) 의 oi_floor 규칙(꼭대기 − 660 초 뒤).
use crate::proto::{OiRow, Supply1s};
use crate::supply::{Supply, SUPPLY_1S_SECONDS};
use rusqlite::{Connection, OpenFlags};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

pub const EVERY: Duration = Duration::from_millis(500);
/// 원본: 읽기 실패면 5초 더 쉰다.
pub const BACKOFF: Duration = Duration::from_secs(5);
const OVERLAP_MS: i64 = 30_000;

/// `oi_1s` 의 `ts_ms > after_ms − 30초` 행(시각 순). `symbol` = 소문자(수집기가 그렇게 쓴다).
pub fn read(path: &Path, symbol: &str, after_ms: i64) -> Result<Vec<(i64, f64)>, String> {
    let err = |e: rusqlite::Error| format!("{}: {e}", path.display());
    let con = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX).map_err(err)?;
    let mut st = con.prepare_cached("SELECT ts_ms, open_interest FROM oi_1s WHERE symbol = ? AND ts_ms > ? ORDER BY ts_ms").map_err(err)?;
    let rows = st.query_map(rusqlite::params![symbol.to_lowercase(), after_ms - OVERLAP_MS], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?))).map_err(err)?;
    rows.collect::<Result<Vec<_>, _>>().map_err(err)
}

/// 링에서 (꼭대기 − 660, …] 이면서 `> since` 인 초.
pub fn window(ring: &BTreeMap<i64, f64>, since: i64) -> Vec<OiRow> {
    let top = ring.keys().next_back().copied().unwrap_or(0);
    ring.range(since.max(top - SUPPLY_1S_SECONDS) + 1..).map(|(&s, &v)| (s, v)).collect()
}

impl Supply {
    /// 읽은 행을 얹는다. 새 초가 없으면 None.
    pub fn bn_oi(&mut self, rows: &[(i64, f64)], now_us: i64) -> Option<Supply1s> {
        if rows.is_empty() {
            return None;
        }
        let top = self.bn_oi.keys().next_back().copied().unwrap_or(i64::MIN);
        for &(ts_ms, v) in rows {
            self.bn_oi.insert(ts_ms.div_euclid(1000), v);
        }
        let cut = now_us.div_euclid(1_000_000) - SUPPLY_1S_SECONDS;
        self.bn_oi = self.bn_oi.split_off(&cut);
        let mut m = self.base(now_us);
        m.oi = window(&self.bn_oi, top);
        (!m.oi.is_empty()).then_some(m)
    }
}

/// 0.5초마다 종목별 꼬리 읽기(블로킹 스레드). 커서 시작 = 지금 − 660초(원본 last_ms 초깃값).
pub async fn run(state: std::sync::Arc<crate::state::State>, path: std::path::PathBuf, symbols: Vec<String>) {
    let start = crate::now_us() / 1000 - SUPPLY_1S_SECONDS * 1000;
    let mut last: Vec<i64> = vec![start; symbols.len()];
    loop {
        let (st, p, syms, cur) = (state.clone(), path.clone(), symbols.clone(), last.clone());
        let (cur, ok) = tokio::task::spawn_blocking(move || {
            let mut cur = cur;
            let mut ok = true;
            for (s, c) in syms.iter().zip(cur.iter_mut()) {
                match read(&p, s, *c) {
                    Ok(rows) => {
                        if let Some(m) = rows.iter().map(|r| r.0).max() {
                            *c = (*c).max(m);
                        }
                        st.bn_oi(s, &rows, crate::now_us());
                    }
                    Err(e) => {
                        eprintln!("rtcore oi-1s hot 읽기 실패(다시 함) {s}: {e}");
                        ok = false;
                    }
                }
            }
            (cur, ok)
        })
        .await
        .unwrap_or((last.clone(), false));
        last = cur;
        tokio::time::sleep(if ok { EVERY } else { EVERY + BACKOFF }).await;
    }
}
