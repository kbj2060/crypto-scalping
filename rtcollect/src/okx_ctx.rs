//! OKX 맥락 수집기(`scripts/live_okx_context_collector_20260923.py`, 줄 번호는 그 파일) -- 미결제약정·마크·펀딩·청산을 공개 WS 하나로
//! → okx_ctx.sqlite. 대시보드 수급의 `okxOi`(okx_oi)·`okxLiq`(okx_liquidations)·시장 맥락 OKX 펀딩·마크가 여기서 온다.
//! 소켓은 `net::run_okx_ctx`, 여기는 메시지 → 행 · 보류 · 쓰기 · 공백. 시계 둘: `now` = 단조 초(플러시) · `wall` = 벽시계 초(공백·펀딩 ts 대체).
//! 🔴OI 는 WS `open-interest` 다(REST 폴링은 응답 노드마다 값이 달라 A->B->A 되돌림 60% -- 2026-09-23 정정, :8-15).
//! 🔴`oi`·`sz` 는 계약 수 -- OI 는 거래소 `oiCcy` 를 쓰고 `oi*ctVal` 과 어긋나면 로그로 떠든다(계속 쌓는다). 청산 `sz` 는 우리가 곱하고
//!   ctVal 모르는 종목은 sz_base = NULL(0 이 아니다: «없었다» ≠ «환산 못 한다»).
//! 🔴청산 `bkPx` 는 파산가격(바이낸스 forceOrder 의 체결가와 다르다) · `side` = 강제 주문 방향, `posSide` = 청산된 포지션.
//! 🔴청산은 instType=SWAP 전 종목이 한 구독에 온다 -- ETH 종목만 구독한다(종목마다 받으면 프로세스 수만큼 중복 저장, 2026-09-26).
//! 원본의 시작 때 REST ctVal 대조(assert_ct_val)는 `rest_venue::assert_ct_val`(기본 꺼짐).
use crate::store::{in_tx, open_hot};
use crate::{ms, py_f, py_f_int, py_repr, py_str, py_truthy, Reconnect};
use rusqlite::{params, Connection};
use serde_json::{Map, Value};
use std::path::Path;
use std::rc::Rc;

pub const LIQ_INST: &str = "ETH-USDT-SWAP";
pub const FLUSH_SECONDS: f64 = 10.0;
/// 마크는 5/s 지만 펀딩은 분 단위라 체결보다 여유(:74). 이 시간 아무 프레임도 없으면 재연결.
pub const RECV_TIMEOUT_S: f64 = 40.0;
pub const PENDING_CAP: usize = 200_000;

// 표(:76-90) -- 줄바꿈·들여쓰기까지 원본 글자(sqlite_master 에 그대로 남는다) + 인덱스(:177-181).
const DDL: [&str; 11] = [
    "CREATE TABLE IF NOT EXISTS okx_oi(\n         inst VARCHAR, ts_ms BIGINT, oi_contracts DOUBLE, oi_base DOUBLE, oi_usd DOUBLE)",
    "CREATE TABLE IF NOT EXISTS okx_mark(\n         inst VARCHAR, ts_ms BIGINT, mark_px DOUBLE)",
    "CREATE TABLE IF NOT EXISTS okx_funding(\n         inst VARCHAR, ts_ms BIGINT, funding_rate DOUBLE, funding_time BIGINT,\n         min_rate DOUBLE, max_rate DOUBLE, method VARCHAR)",
    "CREATE TABLE IF NOT EXISTS okx_liquidations(\n         inst_id VARCHAR, inst_family VARCHAR, ts_ms BIGINT, side VARCHAR, pos_side VARCHAR,\n         bk_px DOUBLE, sz_contracts DOUBLE, sz_base DOUBLE, bk_loss DOUBLE)",
    "CREATE TABLE IF NOT EXISTS gaps(\n         channel VARCHAR, from_ms BIGINT, to_ms BIGINT, reason VARCHAR)",
    "CREATE TABLE IF NOT EXISTS meta(key VARCHAR, value VARCHAR)",
    "CREATE INDEX IF NOT EXISTS okx_oi_i ON okx_oi(inst, ts_ms)",
    "CREATE INDEX IF NOT EXISTS okx_mark_i ON okx_mark(inst, ts_ms)",
    "CREATE INDEX IF NOT EXISTS okx_funding_i ON okx_funding(inst, ts_ms)",
    "CREATE INDEX IF NOT EXISTS okx_liquidations_i ON okx_liquidations(inst_id, ts_ms)",
    "CREATE INDEX IF NOT EXISTS gaps_i ON gaps(channel, to_ms)",
];

pub fn open_db(path: &Path) -> rusqlite::Result<Rc<Connection>> {
    open_hot(path, &DDL).map(Rc::new)
}

type Oi = (String, i64, f64, f64, Option<f64>);
type Mark = (String, i64, f64);
type Funding = (String, i64, f64, i64, Option<f64>, Option<f64>, String);
type Liq = (String, String, i64, String, String, Option<f64>, f64, Option<f64>, Option<f64>);

/// 표별 보류 행(원본 `pending` 사전). 못 쓰면 앞에 되돌려 붙인다.
#[derive(Default)]
pub struct Batch {
    pub oi: Vec<Oi>,
    pub mark: Vec<Mark>,
    pub funding: Vec<Funding>,
    pub liq: Vec<Liq>,
}

impl Batch {
    pub fn len(&self) -> usize {
        self.oi.len() + self.mark.len() + self.funding.len() + self.liq.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn extend(&mut self, b: Batch) {
        self.oi.extend(b.oi);
        self.mark.extend(b.mark);
        self.funding.extend(b.funding);
        self.liq.extend(b.liq);
    }
    fn prepend(&mut self, mut b: Batch) {
        b.extend(std::mem::take(self));
        *self = b;
    }
}

/// `for r in (payload.get("data") or [])` + `r.get(...)`: 거짓이면 없음 · 목록의 원소는 사전이어야(아니면 AttributeError = 재연결) ·
/// 참인 사전·글자는 열쇠·글자가 원소라 AttributeError · 숫자·True 는 TypeError -- 메시지 하나가 통째로 버려지고 재연결.
fn dicts<'a>(v: Option<&'a Value>) -> Result<Vec<&'a Map<String, Value>>, Reconnect> {
    match v {
        Some(Value::Array(a)) => a.iter().map(|x| x.as_object().ok_or_else(|| Reconnect("AttributeError: .get".into()))).collect(),
        Some(v) if py_truthy(v) => Err(Reconnect("AttributeError: .get".into())),
        _ => Ok(vec![]),
    }
}

/// `parse_message`(:101-144) -- WS 메시지 하나 → 행. 못 믿을 행은 조용히 버린다. `wall` = 펀딩 ts 가 없을 때 쓰는 벽시계(원본 time.time()).
pub fn parse_message(p: &Map<String, Value>, wall: f64) -> Result<Batch, Reconnect> {
    let mut out = Batch::default();
    let ch = match p.get("arg").filter(|v| py_truthy(v)) {
        None => None,
        Some(Value::Object(a)) => a.get("channel").and_then(Value::as_str),
        Some(_) => return Err(Reconnect("AttributeError: arg".into())),
    };
    let rows = || dicts(p.get("data"));
    match ch {
        Some("open-interest") => {
            for r in rows()? {
                let (inst, ts) = (py_str(r.get("instId"), ""), py_f_int(r.get("ts"))?);
                let (oi, ccy) = (py_f(r.get("oi")), py_f(r.get("oiCcy")));
                if let (false, true, Some(oi), Some(ccy)) = (inst.is_empty(), ts > 0, oi, ccy) {
                    out.oi.push((inst, ts, oi, ccy, py_f(r.get("oiUsd"))));
                }
            }
        }
        Some("mark-price") => {
            for r in rows()? {
                let (inst, ts, px) = (py_str(r.get("instId"), ""), py_f_int(r.get("ts"))?, py_f(r.get("markPx")));
                // `px and px > 0` -- NaN 은 참이지만 > 0 이 거짓
                if let (false, true, Some(px)) = (inst.is_empty(), ts > 0, px.filter(|x| *x > 0.0)) {
                    out.mark.push((inst, ts, px));
                }
            }
        }
        Some("funding-rate") => {
            for r in rows()? {
                let (inst, ts) = (py_str(r.get("instId"), ""), py_f_int(r.get("ts"))?);
                let Some(rate) = py_f(r.get("fundingRate")) else { continue };
                if inst.is_empty() {
                    continue;
                }
                // `ts or int(time.time() * 1000)` -- 음수 ts 는 그대로(원본에 ts > 0 검사가 없다)
                let ts = if ts != 0 { ts } else { ms(wall) };
                out.funding.push((inst, ts, rate, py_f_int(r.get("fundingTime"))?, py_f(r.get("minFundingRate")),
                                  py_f(r.get("maxFundingRate")), py_str(r.get("method"), "")));
            }
        }
        Some("liquidation-orders") => {
            for r in rows()? {
                let (inst, family) = (py_str(r.get("instId"), ""), py_str(r.get("instFamily"), ""));
                let ct = crate::okx::ct_val(&inst);
                for d in dicts(r.get("details"))? {
                    let (ts, sz) = (py_f_int(d.get("ts"))?, py_f(d.get("sz")));
                    let Some(sz) = sz.filter(|s| *s > 0.0) else { continue };
                    if inst.is_empty() || ts <= 0 {
                        continue;
                    }
                    out.liq.push((inst.clone(), family.clone(), ts, py_str(d.get("side"), ""), py_str(d.get("posSide"), ""),
                                  py_f(d.get("bkPx")), sz, ct.map(|c| sz * c), py_f(d.get("bkLoss"))));
                }
            }
        }
        _ => {}
    }
    Ok(out)
}

/// `oi * ctVal != oiCcy` 인 첫 행(:147-155) -- 계약 크기가 갈라진 순간. 로그만(계속 쌓는다).
pub fn oi_mismatch(rows: &[Oi]) -> Option<(String, f64, f64)> {
    rows.iter().find_map(|(inst, _, oi, ccy, _)| {
        let ct = crate::okx::ct_val(inst)?;
        (*ccy != 0.0 && (oi * ct - ccy).abs() > ccy.abs() * 1e-9).then(|| (inst.clone(), *ccy, oi * ct))
    })
}

fn write(con: &Connection, b: &Batch) -> rusqlite::Result<()> {
    in_tx(con, || {
        for r in &b.oi {
            con.prepare_cached("INSERT INTO okx_oi VALUES (?,?,?,?,?)")?.execute(params![r.0, r.1, r.2, r.3, r.4])?;
        }
        for r in &b.mark {
            con.prepare_cached("INSERT INTO okx_mark VALUES (?,?,?)")?.execute(params![r.0, r.1, r.2])?;
        }
        for r in &b.funding {
            con.prepare_cached("INSERT INTO okx_funding VALUES (?,?,?,?,?,?,?)")?.execute(params![r.0, r.1, r.2, r.3, r.4, r.5, r.6])?;
        }
        for r in &b.liq {
            con.prepare_cached("INSERT INTO okx_liquidations VALUES (?,?,?,?,?,?,?,?,?)")?
                .execute(params![r.0, r.1, r.2, r.3, r.4, r.5, r.6, r.7, r.8])?;
        }
        Ok(())
    })
}

/// 한 종목의 `collect()`·`_ws_loop()`(:252-325). 같은 파일의 종목들은 연결 하나(Rc)를 나눠 쓴다(원본 sqlite_shared).
pub struct CtxCollector {
    con: Rc<Connection>,
    pub inst: String,
    pub pending: Batch,
    flushed_at: f64,
    down_from: i64,
}

impl CtxCollector {
    /// None = ctVal 모르는 종목(원본 SystemExit). meta 는 시작마다 DELETE → INSERT(:261-267).
    pub fn new(con: Rc<Connection>, inst: &str, now: f64, wall: f64) -> rusqlite::Result<Option<Self>> {
        let Some(ct) = crate::okx::ct_val(inst) else { return Ok(None) };
        let scope = if inst == LIQ_INST {
            "instType=SWAP (전 종목) · sz_base 는 ctVal 아는 것만".to_string()
        } else {
            format!("없음 -- {LIQ_INST} 프로세스(okx_context.duckdb)가 전 종목을 받는다")
        };
        for (k, v) in [
            (format!("ct_val:{inst}"), py_repr(ct)),
            (format!("liquidation_scope:{inst}"), scope),
            ("bk_px_note".into(), "파산가격이다 -- 바이낸스 forceOrder 의 체결가와 다르다".into()),
            ("okx_oi_note".into(), "2026-09-23 05:30~23:59 KST 행은 REST 0.25초 폴링분이라 A->B->A 되돌림(응답 노드 불일치) 잡음이 ~60% 섞여 있다. 그 뒤는 WS open-interest(진짜 거래소 ts, 5~10초 간격)".into()),
        ] {
            con.execute("DELETE FROM meta WHERE key = ?", [&k])?;
            con.execute("INSERT INTO meta VALUES (?, ?)", params![k, v])?;
        }
        // down_from = 시작 시각(:273) · flushed_at = 수신 루프 시작(:282)
        Ok(Some(Self { con, inst: inst.into(), pending: Batch::default(), flushed_at: now, down_from: ms(wall) }))
    }

    /// 구독 글자(:268-271, json.dumps 기본 모양).
    pub fn subscribe(&self) -> String {
        let i = Value::from(self.inst.as_str());
        let mut a: Vec<String> =
            ["open-interest", "mark-price", "funding-rate"].iter().map(|c| format!(r#"{{"channel": "{c}", "instId": {i}}}"#)).collect();
        if self.inst == LIQ_INST {
            a.push(r#"{"channel": "liquidation-orders", "instType": "SWAP"}"#.into());
        }
        crate::py_subscribe(&a)
    }

    /// 구독을 보낸 직후(:288): 끊긴 구간(채널 "all"). 1.5초 미만은 안 적는다 · 실패해도 수집은 계속.
    pub fn on_connect(&mut self, wall: f64) {
        let to = ms(wall);
        if to - self.down_from < 1500 {
            return;
        }
        if let Err(e) = self.con.execute("INSERT INTO gaps VALUES (?,?,?,?)", params!["all", self.down_from, to, "reconnect"]) {
            eprintln!("[{}] gap 기록 실패(수집은 계속): {e}", self.inst);
        }
    }

    /// 글자 메시지 하나(:294-316). Err = 그 연결을 접는다(구독 거부 포함). 🔴플러시는 처리한 메시지 뒤에만(pong·깨진 JSON·이벤트는 건너뜀).
    pub fn on_text(&mut self, now: f64, wall: f64, text: &str) -> Result<(), Reconnect> {
        if text == "pong" {
            return Ok(());
        }
        let Ok(payload) = serde_json::from_str::<Value>(text) else { return Ok(()) };
        let Value::Object(p) = &payload else { return Err(Reconnect("AttributeError".into())) };
        let event = p.get("event");
        if event.and_then(Value::as_str) == Some("error") {
            eprintln!("[{}] 🔴구독 거부: {:?} (code {:?})", self.inst, p.get("msg"), p.get("code"));
            return Err(Reconnect("구독 거부".into()));
        }
        if event.is_some_and(py_truthy) {
            return Ok(());
        }
        let batch = parse_message(p, wall)?;
        if let Some((inst, ccy, got)) = oi_mismatch(&batch.oi) {
            eprintln!("[{}] 🔴OI 단위가 갈라졌다 {inst}: oiCcy {ccy} vs oi*ctVal {got} -- CT_VALS 를 확인할 것(계속 쌓기는 한다)", self.inst);
        }
        self.pending.extend(batch);
        if now - self.flushed_at >= FLUSH_SECONDS {
            self.flushed_at = now;
            self.flush();
        }
        Ok(())
    }

    /// 연결이 끝났다(시한·끊김·접속 실패·예외, :323-325): 공백 시작을 지금으로 · 보류분 쓰기.
    /// 🔴원본 그대로: 접속 실패마다 down_from 을 다시 잡아 다음 공백은 «마지막 실패 ~ 붙은 시각»만 적힌다(첫 끊김부터가 아니다).
    pub fn on_disconnect(&mut self, wall: f64) {
        self.down_from = ms(wall);
        self.flush();
    }

    /// `aflush`(:206-223). 반환 = 쓴 행 수.
    pub fn flush(&mut self) -> usize {
        let total = self.pending.len();
        if total == 0 {
            return 0;
        }
        let batch = std::mem::take(&mut self.pending);
        match write(&self.con, &batch) {
            Ok(()) => total,
            Err(e) => {
                self.pending.prepend(batch);
                if total > PENDING_CAP {
                    self.pending = Batch::default();
                    eprintln!("[{}] ⚠️쓰기가 계속 막혀 {total}행 버림 -- 그 구간은 소급 복원 불가다", self.inst);
                } else {
                    eprintln!("[{}] 쓰기 보류 {total}행, 다음 주기 재시도: {e}", self.inst);
                }
                0
            }
        }
    }
}

/// 정답지 입력 먹이기. 실행 = {inst, t0, events}, 사건 = [초(단조=벽), 종류, 본문]: 'o' 접속 · 't' 글자 · 'w' 40초 시한 · 'c' 글자 아닌 메시지 ·
/// 'f' 접속 실패. 종목(실행)은 차례로(원본도 종목마다 상태가 따로이고 파일·연결만 나눈다). 반환 = 실행마다 보낸 구독 글자.
pub fn replay(runs: &[Value], path: &Path) -> rusqlite::Result<Vec<String>> {
    let con = open_db(path)?;
    let mut subs = vec![];
    for run in runs {
        let t0 = run["t0"].as_f64().unwrap();
        let mut c = CtxCollector::new(con.clone(), run["inst"].as_str().unwrap(), t0, t0)?.expect("ctVal");
        subs.push(c.subscribe());
        for e in run["events"].as_array().unwrap() {
            let (t, kind, body) = (e[0].as_f64().unwrap(), e[1].as_str().unwrap(), e[2].as_str().unwrap_or(""));
            match kind {
                "o" => c.on_connect(t),
                "t" => {
                    if c.on_text(t, t, body).is_err() {
                        c.on_disconnect(t);
                    }
                }
                _ => c.on_disconnect(t),
            }
        }
    }
    Ok(subs)
}
