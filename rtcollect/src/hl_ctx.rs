//! 하이퍼리퀴드 자산 맥락(`scripts/live_hyperliquid_book_ticker_collector_20260923.py` 의 `HL_BT_CTX_ONLY=1` 변형, 줄 번호는 그 파일) → hl_ctx.sqlite.
//! 대시보드 시장 맥락의 «HL 펀딩·OI·프리미엄»(server.py `_mc_collectors` hl_ctx)·seal_to_lake·data_quality_monitor 가 읽는다.
//! 🔴`activeAssetCtx` 에는 거래소 시각이 없다 -- `recv_ms` = **우리 시계**(받은 시각)이고 meta 에 박는다(지연 분석이 거짓이 되지 않게).
//! ⭐sz·OI 는 코인 단위 그대로(OKX 처럼 계약 수가 아니다 -- ctVal 환산이 없다). OI 가 없는 메시지는 적지 않는다(이 표의 존재 이유).
//! 종류 hl_ctx = 맥락만(`HL_BT_CTX_ONLY=1` -- 서버 SOL·XRP) · hl_bt = 전체(bbo → 시각 파일 `.bt`(bt.rs) + 맥락 -- Pi 의 원본 형태, 2026-11-21).
use crate::bt::HourFile;
use crate::store::{in_tx, open_hot};
use crate::{py_float, py_str, Reconnect};
use rusqlite::{params, Connection};
use serde_json::{Map, Value};
use std::path::Path;
use std::rc::Rc;

pub const WS_URL: &str = "wss://api.hyperliquid.xyz/ws";
/// 이만큼 조용하면 {"method": "ping"} 을 보낸다(끊지 않는다 -- 재연결이 더 비싸다, :61).
pub const RECV_TIMEOUT_S: f64 = 30.0;
pub const FLUSH_S: f64 = 30.0;
pub const PENDING_CAP: usize = 50_000;
pub const PING: &str = r#"{"method": "ping"}"#;

// :65-68 · :112-115 -- 줄바꿈·들여쓰기까지 원본 글자
const DDL: [&str; 3] = [
    "CREATE TABLE IF NOT EXISTS hl_asset_ctx(\n    coin VARCHAR, recv_ms BIGINT, funding DOUBLE, open_interest DOUBLE, premium DOUBLE,\n    oracle_px DOUBLE, mark_px DOUBLE, mid_px DOUBLE, day_ntl_vlm DOUBLE, day_base_vlm DOUBLE)",
    "CREATE TABLE IF NOT EXISTS meta(key VARCHAR, value VARCHAR)",
    "CREATE INDEX IF NOT EXISTS hl_asset_ctx_i ON hl_asset_ctx(coin, recv_ms)",
];

pub fn open_db(path: &Path) -> rusqlite::Result<Rc<Connection>> {
    let con = open_hot(path, &DDL)?;
    for (k, v) in [("recv_ms", "🔴우리 시계다 -- activeAssetCtx 에는 거래소 시각이 없다"), ("sz_unit", "ETH (계약 수가 아니다)"),
                   ("l2book_note", "l2Book 은 중앙 5.4초 간격이라 받지 않는다")] {
        con.execute("DELETE FROM meta WHERE key = ?", [k])?;
        con.execute("INSERT INTO meta VALUES (?, ?)", params![k, v])?;
    }
    Ok(Rc::new(con))
}

/// (ts_ms, 매수가, 매수량, 매도가, 매도량) -- 🔴sz 는 코인 단위 그대로(ctVal 없음).
pub type Bbo = (i64, f64, f64, f64, f64);

/// `parse_bbo`(:85-98): 두 쪽이 다 있고(null 이면 버린다) 숫자로 읽히고 교차하지 않을 때만.
pub fn parse_bbo(data: &Map<String, Value>) -> Option<Bbo> {
    let Some(Value::Array(lv)) = data.get("bbo") else { return None };
    if lv.len() != 2 || !(crate::py_truthy(&lv[0]) && crate::py_truthy(&lv[1])) {
        return None;
    }
    let ts = crate::py_int(data.get("time")?).ok()?;
    let f = |i: usize, k: &str| lv[i].as_object().and_then(|o| o.get(k)).and_then(|v| py_float(v).ok());
    let (bp, bs, ap, az) = (f(0, "px")?, f(0, "sz")?, f(1, "px")?, f(1, "sz")?);
    (ts > 0 && bp > 0.0 && ap > bp && bs > 0.0 && az > 0.0).then_some((ts, bp, bs, ap, az))
}

pub type Row = (String, i64, Option<f64>, f64, Option<f64>, Option<f64>, Option<f64>, Option<f64>, Option<f64>, Option<f64>);

/// `parse_ctx`(:84-97). ctx 가 사전이 아니거나 OI 를 못 읽으면 None. 칸마다 float(ctx[k]) -- 없거나 못 읽으면 NULL(0 이 아니다).
pub fn parse_ctx(data: &Map<String, Value>, recv_ms: i64) -> Option<Row> {
    let Some(Value::Object(ctx)) = data.get("ctx") else { return None };
    let g = |k: &str| ctx.get(k).and_then(|v| py_float(v).ok());
    let oi = g("openInterest")?;
    Some((py_str(data.get("coin"), ""), recv_ms, g("funding"), oi, g("premium"), g("oraclePx"), g("markPx"), g("midPx"), g("dayNtlVlm"), g("dayBaseVlm")))
}

/// 한 코인의 `run()`(:120-168) 상태. 같은 파일의 코인들은 연결 하나(Rc)를 나눠 쓴다.
pub struct HlCtx {
    con: Rc<Connection>,
    pub coin: String,
    /// Some = 전체 모드(bbo 구독 · .bt 쓰기). last_ts·last_row = 역행·완전 중복 버리기(재연결을 넘어 이어진다).
    pub bt: Option<HourFile>,
    last_ts: i64,
    last_row: Option<Bbo>,
    pub n_bbo: u64,
    pub dropped: u64,
    buf: Vec<Row>,
    pending: Vec<Row>,
    flushed_at: f64,
}

impl HlCtx {
    pub fn new(con: Rc<Connection>, coin: &str, now: f64) -> Self {
        Self { con, coin: coin.to_uppercase(), bt: None, last_ts: -1, last_row: None, n_bbo: 0, dropped: 0, buf: vec![], pending: vec![], flushed_at: now }
    }

    pub fn subscribe(&self) -> Vec<String> {
        let kinds: &[&str] = if self.bt.is_some() { &["bbo", "activeAssetCtx"] } else { &["activeAssetCtx"] };
        kinds.iter().map(|k| format!(r#"{{"method": "subscribe", "subscription": {{"type": "{k}", "coin": {}}}}}"#, Value::from(self.coin.as_str()))).collect()
    }

    /// 글자 메시지(:143-166). Err = 연결을 접는다. 깨진 JSON 은 건너뛴다(플러시 판정도 안 한다).
    pub fn on_text(&mut self, now: f64, wall: f64, text: &str) -> Result<(), Reconnect> {
        let Ok(p) = serde_json::from_str::<Value>(text) else { return Ok(()) };
        let Value::Object(p) = p else { return Err(Reconnect("AttributeError: .get".into())) };
        let channel = p.get("channel").and_then(Value::as_str);
        if channel == Some("bbo") && self.bt.is_some() {
            let empty = Map::new();
            let data = match p.get("data").filter(|v| crate::py_truthy(v)) {
                None => &empty,
                Some(Value::Object(d)) => d,
                Some(_) => return Err(Reconnect("AttributeError: .get".into())),
            };
            // 버린 bbo 는 플러시 판정 없이 다음 메시지로(원본 continue, :195-201)
            let Some(b) = parse_bbo(data) else {
                self.dropped += 1;
                return Ok(());
            };
            if b.0 < self.last_ts || Some(b) == self.last_row {
                self.dropped += 1;
                return Ok(());
            }
            (self.last_ts, self.last_row) = (b.0, Some(b));
            self.bt.as_mut().unwrap().write(b.0, b.1, b.2, b.3, b.4, b.0, now).map_err(|e| Reconnect(format!("OSError: {e}")))?;
            self.n_bbo += 1;
        } else if channel == Some("activeAssetCtx") {
            // payload.get("data") or {} -- 참인 비사전이면 .get 에서 AttributeError
            let empty = Map::new();
            let data = match p.get("data").filter(|v| crate::py_truthy(v)) {
                None => &empty,
                Some(Value::Object(d)) => d,
                Some(_) => return Err(Reconnect("AttributeError: .get".into())),
            };
            if let Some(r) = parse_ctx(data, crate::ms(wall)) {
                self.buf.push(r);
            }
        }
        if now - self.flushed_at >= FLUSH_S {
            self.flushed_at = now;
            self.write();
        }
        Ok(())
    }

    /// 연결 끝(끊김·예외·접속 실패) -- 남은 행을 쓴다(:163-165).
    pub fn on_disconnect(&mut self) {
        if let Some(b) = self.bt.as_mut() {
            b.close(false); // 같은 시각이면 다음 연결이 헤더 없이 이어 쓴다
        }
        self.write();
    }

    /// `CtxStore.write`(:100-115): 보류 + 버퍼를 한 트랜잭션으로. 못 쓰면 보류(5만 행 넘으면 뒤 절반만 남긴다).
    pub fn write(&mut self) -> usize {
        self.pending.append(&mut self.buf);
        if self.pending.is_empty() {
            return 0;
        }
        let r = in_tx(&self.con, || {
            let mut st = self.con.prepare_cached("INSERT INTO hl_asset_ctx VALUES (?,?,?,?,?,?,?,?,?,?)")?;
            for x in &self.pending {
                st.execute(params![x.0, x.1, x.2, x.3, x.4, x.5, x.6, x.7, x.8, x.9])?;
            }
            Ok(())
        });
        match r {
            Ok(()) => {
                let n = self.pending.len();
                self.pending.clear();
                n
            }
            Err(e) => {
                if self.pending.len() > PENDING_CAP {
                    let h = self.pending.len() / 2;
                    self.pending.drain(..h);
                    eprintln!("[{}] 쓰기가 계속 막혀 절반 버림", self.coin);
                } else {
                    eprintln!("[{}] ctx 쓰기 보류 {}행: {e}", self.coin, self.pending.len());
                }
                0
            }
        }
    }
}

/// 정답지 입력 먹이기. 실행 = {coin, t0, events}, 사건 = [초, 종류, 본문]: 'o' 접속 · 'f' 접속 실패 · 't' 글자 · 'w' 30초 시한(ping) · 'c' 끝.
/// 반환 = 보낸 글자.
/// bt = Some(폴더) 면 전체 모드(hl_bt -- bbo 구독 · .bt). 실행 끝에 .bt 를 닫는다(원본은 실행이 끝나 파일이 닫힐 때 버퍼가 비워진다).
pub fn replay(runs: &[Value], path: &Path, bt: Option<&Path>) -> rusqlite::Result<Vec<String>> {
    let con = open_db(path)?;
    let mut sent = vec![];
    for run in runs {
        let mut c = HlCtx::new(con.clone(), run["coin"].as_str().unwrap(), run["t0"].as_f64().unwrap());
        if let Some(root) = bt {
            c.bt = Some(HourFile::new(root, &c.coin, run["t0"].as_f64().unwrap()).unwrap());
        }
        for e in run["events"].as_array().unwrap() {
            let (t, kind, body) = (e[0].as_f64().unwrap(), e[1].as_str().unwrap(), e[2].as_str().unwrap_or(""));
            match kind {
                "o" => sent.extend(c.subscribe()),
                "t" => {
                    if c.on_text(t, t, body).is_err() {
                        c.on_disconnect();
                    }
                }
                "w" => sent.push(PING.into()),
                _ => c.on_disconnect(),
            }
        }
        if let Some(b) = c.bt.as_mut() {
            b.close(false);
        }
    }
    Ok(sent)
}
