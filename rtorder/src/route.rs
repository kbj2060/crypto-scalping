//! 주문·계좌 주소 14개의 공개 진입점(2026-11-21) -- 응답 서버(rtserve)는 `Front::call` 을 그대로 부르고 답을 내보내기만 한다.
//! 원본 = server.py 라우트 등록(5004–5017) + 미들웨어 `order_request_log`(1352) + orders.py 핸들러 + `busy_or`. 여기 안에서 하는 것:
//! - 주소·메서드(G1): 표에 없는 주소 404 · 메서드가 다르면 405(aiohttp 글자 그대로 · Allow 머리) -- 2026-11-21 지운 manual-exit/preview·submit-all 도 404.
//! - 주문 잠금(G8): 표의 `busy` 주소는 잠금이 잡혀 있으면 409 already_working(원본 busy_or) · 잡으면 핸들러 끝까지 쥔다.
//! - confirm=1·실행 스위치(G2·G3)는 핸들러 몫(intake) 그대로.
//! - 본문 = 파이썬 json.dumps 글자(`pyfmt::json_dumps`) · 200 이면 Cache-Control: no-cache(원본 NOCACHE -- 200 응답에만 붙어 있다) · 진입·청산 접수 200 은 server_ms.
//! - 접수 기록(G18) 몫: `Reply::log` = POST /api/manual-* 의 {path, query(키마다 첫 값), status, error} -- 시각·ms·UA·IP 는 HTTP 를 쥔 쪽이 붙인다.
//! ponytail: 요청 하나가 `DeskCell` 을 처음부터 끝까지 쥔다(원본은 await 사이 다른 핸들러·감시가 같은 dict 를 본다) -- 조립(REST) 동안 다른 주문 요청은 기다린다.
use crate::assemble::{Desk, DeskIo, Quotes};
use crate::intake::{Query, Task};
use crate::plan::bracket_key;
use crate::pyfmt::json_dumps;
use crate::runner::DeskCell;
use crate::sign::{py_float, py_str, truthy};
use crate::userstream::Hub;
use crate::watch::bracket_load;
use serde_json::{json, Value};

/// (메서드, 주소, 이름, 주문 잠금 안) -- 원본 등록 순서.
pub const ROUTES: [(&str, &str, &str, bool); 14] = [
    ("GET", "/api/binance-account", "binance_account", false),
    ("GET", "/api/manual-entry/preview", "entry_preview", false),
    ("POST", "/api/manual-entry/submit", "entry_submit", true),
    ("GET", "/api/manual-entry/status", "entry_status", false),
    ("POST", "/api/manual-entry/wait-action", "wait_action", false),
    ("POST", "/api/manual-entry/cancel", "entry_cancel", false),
    ("POST", "/api/manual-entry/limit-amend", "limit_amend", true),
    ("POST", "/api/manual-bracket/disarm", "bracket_disarm", false),
    ("POST", "/api/manual-bracket/rearm", "bracket_rearm", false),
    ("POST", "/api/manual-bracket/tp", "bracket_tp", true),
    ("POST", "/api/manual-bracket/tp-legs", "bracket_tp_legs", true),
    ("POST", "/api/manual-bracket/add", "bracket_add", true),
    ("POST", "/api/manual-bracket/modify", "bracket_modify", true),
    ("POST", "/api/manual-exit/submit", "exit_submit", true),
];

#[derive(Debug, Clone, PartialEq)]
pub struct Reply {
    pub status: u16,
    pub content_type: &'static str,
    /// 본문 외 머리(Cache-Control · Allow).
    pub headers: Vec<(&'static str, String)>,
    pub body: String,
    /// 시작할 작업(`runner::run_task` 로 띄운다).
    pub task: Vec<Task>,
    /// 접수 기록 줄(POST /api/manual-* 만). 원본 칸 at·ms·ua·ip 는 부르는 쪽이 더한다.
    pub log: Option<Value>,
}

/// 칸(DeskCell)을 기다리는 최대 초 -- 넘으면 409 already_working(원본은 await 사이 다른 핸들러가 돌아 기다리지 않는다 · 의도한 차이).
pub const DESK_WAIT_SEC: f64 = 5.0;
const JSON: &str = "application/json; charset=utf-8";
const TEXT: &str = "text/plain; charset=utf-8";

fn json_reply(status: u16, body: &Value, task: Vec<Task>) -> Reply {
    if status == 500 && body.get("exception").is_some() {
        // 핸들러에서 샌 예외 -- aiohttp 기본 500 글자(본문에 예외를 싣지 않는다)
        return Reply { status, content_type: TEXT, headers: vec![], body: "500 Internal Server Error\n\nServer got itself in trouble".into(), task, log: None };
    }
    let headers = if status == 200 { vec![("Cache-Control", "no-cache".to_string())] } else { vec![] };
    Reply { status, content_type: JSON, headers, body: json_dumps(body), task, log: None }
}

/// 원본 order_request_log 의 error 칸: 본문이 2만 바이트 미만 JSON dict 이고 ok 가 거짓이면 error(없으면 "") · 아니면 null.
pub fn log_error(body: &str) -> Value {
    if body.len() >= 20000 {
        return Value::Null;
    }
    match serde_json::from_str::<Value>(body) {
        Ok(Value::Object(m)) if m.get("ok").is_some_and(|v| !truthy(Some(v))) => match m.get("error") {
            Some(e) if truthy(Some(e)) => e.clone(),
            _ => json!(""),
        },
        _ => Value::Null,
    }
}

/// 원본 with_bracket: 계좌에 무장 화면 칸 · 무방비 포지션 · 사용자 스트림 상태를 얹는다(받은 payload 는 그대로).
pub fn with_bracket(payload: &Value, armed_file: &serde_json::Map<String, Value>, hub: &Hub, now: f64) -> Value {
    let Some(p) = payload.as_object() else { return payload.clone() };
    const F: [&str; 15] = ["sl_price", "sl_name", "sl_off", "tp_off", "fired", "rule", "lines", "tp_level", "sl_level", "tp_resting", "sl_up", "tp_price", "basis", "tp_legs", "sl_legs"];
    let armed: serde_json::Map<String, Value> = armed_file.iter()
        .map(|(k, v)| (k.clone(), Value::Object(F.iter().map(|f| (f.to_string(), v.get(*f).cloned().unwrap_or(Value::Null))).collect()))).collect();
    let none = vec![];
    let unprotected: Vec<Value> = p.get("positions").and_then(Value::as_array).unwrap_or(&none).iter()
        .filter(|x| truthy(x.get("qty")) && x.get("qty").is_some_and(|q| py_float(q).unwrap_or(0.0) != 0.0))
        .map(|x| bracket_key(&py_str(x.get("symbol").unwrap_or(&Value::Null)), &py_str(x.get("side").unwrap_or(&Value::Null)).to_uppercase()))
        .filter(|k| !armed.get(k).is_some_and(|a| truthy(a.get("sl_price"))))
        .map(Value::String).collect();
    let us = hub.us.lock().unwrap().clone();
    let mut o = p.clone();
    o.insert("bracket_armed".into(), Value::Object(armed));
    o.insert("unprotected".into(), json!(unprotected));
    o.insert("user_stream".into(), json!({"connected": us.connected, "error": us.error, "seq": hub.acct_seq(),
        "last_event_age_s": us.last_event_at.filter(|t| *t != 0.0).map(|t| (now - t).round_ties_even() as i64)}));
    Value::Object(o)
}

/// 주문 잠금(원본 manual_order_lock) -- 응답 서버가 하나 만들어 모든 요청에 같이 넘긴다.
pub struct Front {
    pub order_lock: tokio::sync::Mutex<()>,
    /// 진입 칸 사본(칸을 쥘 때마다 갱신) -- 409 already_working 본문용. 잠금을 쥔 요청이 칸(DeskCell)을 쥐고 있으니 기다리지 않고 이것을 싣는다.
    pub entry_seen: std::sync::Mutex<Value>,
}

impl Default for Front {
    fn default() -> Self {
        Front { order_lock: Default::default(), entry_seen: std::sync::Mutex::new(json!({"phase": "idle"})) }
    }
}

impl Front {
    /// 요청 하나. `query` = 받은 순서 그대로(같은 키가 여럿이면 첫 값을 쓴다 -- aiohttp request.query.get).
    pub async fn call<Q: Quotes, I: DeskIo>(&self, desk: &DeskCell, hub: &Hub, qs: &mut Q, io: &mut I, method: &str, path: &str, query: &[(String, String)]) -> Reply {
        let t0 = std::time::Instant::now();
        let mut q = Query::new();
        let mut first = serde_json::Map::new();
        for (k, v) in query {
            if !q.contains_key(k) {
                q.insert(k.clone(), v.clone());
                first.insert(k.clone(), json!(v));
            }
        }
        let mut r = match ROUTES.iter().find(|r| r.1 == path) {
            None => Reply { status: 404, content_type: TEXT, headers: vec![], body: "404: Not Found".into(), task: vec![], log: None },
            Some(&(m, _, _, _)) if m != method && !(m == "GET" && method == "HEAD") => {
                let allow = if m == "GET" { "GET,HEAD" } else { m };
                Reply { status: 405, content_type: TEXT, headers: vec![("Allow", allow.to_string())], body: "405: Method Not Allowed".into(), task: vec![], log: None }
            }
            Some(&(_, _, name, locked)) => {
                let _g = if locked {
                    match self.order_lock.try_lock() {
                        Ok(g) => Some(g),
                        Err(_) => {
                            let e = match desk.try_lock() {
                                Ok(d) => Value::Object(d.w.entry.clone()),
                                Err(_) => self.entry_seen.lock().unwrap().clone(),
                            };
                            return self.finish(json_reply(409, &json!({"ok": false, "error": "already_working", "state": e}), vec![]), method, path, first);
                        }
                    }
                } else {
                    None
                };
                // 칸을 쥔 쪽(감시 틱의 손절 청산은 최대 ~2분)이 놓지 않으면 5초 뒤 409 -- 화면·응답 서버가 끝없이 매달리지 않게
                let mut d = match desk.try_lock() {
                    Ok(d) => d,
                    Err(_) => match tokio::time::timeout(std::time::Duration::from_secs_f64(DESK_WAIT_SEC), desk.lock()).await {
                        Ok(d) => d,
                        Err(_) => {
                            let e = self.entry_seen.lock().unwrap().clone();
                            return self.finish(json_reply(409, &json!({"ok": false, "error": "already_working", "state": e}), vec![]), method, path, first);
                        }
                    },
                };
                *self.entry_seen.lock().unwrap() = Value::Object(d.w.entry.clone());
                let (status, mut body, task) = handle(&mut d, hub, qs, io, name, &q).await;
                *self.entry_seen.lock().unwrap() = Value::Object(d.w.entry.clone());
                if status == 200 && matches!(name, "entry_submit" | "exit_submit") {
                    body["server_ms"] = json!(t0.elapsed().as_millis() as i64);
                }
                json_reply(status, &body, task)
            }
        };
        if method == "HEAD" {
            r.body.clear(); // aiohttp: GET 핸들러를 돌리고 본문만 버린다
        }
        self.finish(r, method, path, first)
    }

    fn finish(&self, mut r: Reply, method: &str, path: &str, query: serde_json::Map<String, Value>) -> Reply {
        if method == "POST" && path.starts_with("/api/manual-") {
            r.log = Some(json!({"path": path, "query": query, "status": r.status, "error": log_error(&r.body)}));
        }
        r
    }
}

async fn handle<Q: Quotes, I: DeskIo>(d: &mut Desk, hub: &Hub, qs: &mut Q, io: &mut I, name: &str, q: &Query) -> (u16, Value, Vec<Task>) {
    let r = match name {
        "binance_account" => {
            // 원본 ?fresh=1 = 새로 읽기(1초 안에 받은 값이면 그것) · 아니면 swr 캐시 -- `DeskIo::account_screen`
            return match io.account_screen(q.get("fresh").map(String::as_str) == Some("1")).await {
                Ok(p) => (200, with_bracket(&p, &bracket_load(&d.w.dir), hub, io.now()), vec![]),
                Err(e) => (500, json!({"error": "500 Internal Server Error", "exception": e}), vec![]),
            };
        }
        "entry_status" => return (200, entry_status(d, io, q), vec![]),
        "entry_preview" => d.api_entry_preview(qs, io, q).await,
        "entry_submit" => d.api_entry_submit(qs, io, q).await,
        "wait_action" => d.api_wait_action(qs, io, q).await,
        "entry_cancel" => d.api_entry_cancel(io, q),
        "limit_amend" => d.api_limit_amend(qs, io, q).await,
        "bracket_disarm" => d.api_bracket_disarm(io, q).await,
        "bracket_rearm" => d.api_bracket_rearm(io, q).await,
        "bracket_tp" => d.api_bracket_tp(io, q).await,
        "bracket_tp_legs" => d.api_bracket_tp_legs(io, q).await,
        "bracket_add" => d.api_bracket_add(qs, io, q).await,
        "bracket_modify" => d.api_bracket_modify(qs, io, q).await,
        "exit_submit" => d.api_exit_submit(qs, io, q).await,
        _ => unreachable!("ROUTES 표에 있는 이름만"),
    };
    (r.status, r.body, r.task)
}

/// 원본 api_manual_entry_status -- wait 칸(없으면 만든다)을 먼저 보고 나서 waits 를 싣는다(원본 dict 리터럴 순서 · setdefault).
fn entry_status<I: DeskIo>(d: &mut Desk, io: &I, q: &Query) -> Value {
    let asset = crate::intake::query_asset(d, q).unwrap_or_else(|| "eth".into());
    let wait = Value::Object(d.w.wait_slot(&asset, 1).clone());
    let wait_list: Vec<Value> = d.w.wslots(&asset).into_iter().filter_map(|k| {
        let v = d.w.waits[&k].as_object()?;
        (v.get("phase").and_then(Value::as_str) != Some("idle")).then(|| {
            let mut o = v.clone();
            o.insert("wid".into(), json!(k.split_once('#').map_or(1, |x| x.1.parse::<i64>().unwrap_or(1))));
            Value::Object(o)
        })
    }).collect();
    json!({"ok": true, "state": d.w.entry, "wait": wait, "waits": d.w.waits, "wait_list": wait_list, "side_exits": d.w.side_exits,
           "exec_enabled": io.exec_enabled(), "bracket_armed": bracket_load(&d.w.dir)})
}
