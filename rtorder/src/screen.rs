//! 계좌 카드 화면 몫(2026-11-23 · 11-22 전환 뒤 카드가 비었다) -- 원본 `dashboard/orders.py` 의
//! `record_account_trips`(잘림·겹침·항등식 가드) · `load_account_trip_rows`·`trip_habit` · `annotate_trip_trend`(trend_rule 5기간 부호만) ·
//! `account_day_start` · `income_fold`·`income_refresh`·`account_income`.
//! 파일 = 파이썬과 같은 폴더(RTORDER_SCREEN_DIR = data/live)의 같은 이름 -- 원장을 이어 쓴다. 표식이 있는 동안 파이썬은 이 파일들을 안 쓴다(쓰는 쪽은 늘 하나).
//! 🔴주문·감시 경로는 안 지난다 -- 화면 응답(`DoorIo::account_screen`)과 5분 원장 주기(app)만 부른다. 실패는 그 칸만 비고 계좌 응답은 그대로.
use crate::account::AccountRest;
use crate::pyfmt::round_n;
use crate::sign::{py_float, py_str, truthy, P};
use crate::watch::iso_utc;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

pub const TRIP_LEDGER: &str = "account_round_trips.jsonl";
pub const TRIP_HABITS: &str = "account_trip_habits.json";
pub const DAY_START: &str = "account_day_start.json";
pub const INCOME_STATE: &str = "account_income_daily.json";
pub const RECORD_SEC: f64 = 300.0; // 원본 ACCOUNT_TRIP_RECORD_SECONDS(브라우저가 닫혀 있어도 도는 주기)
pub const CARD_ROWS: usize = 400;
pub const CHECK_TOL_BP: f64 = 0.5;
pub const INCOME_REFRESH_SEC: f64 = 60.0;
pub const INCOME_BACKFILL_DAYS: i64 = 88;
pub const INCOME_WINDOW_MS: i64 = 7 * 86_400_000;
pub const INCOME_LIMIT: usize = 1000;
pub const INCOME_OVERLAP_MS: i64 = 600_000;
pub const INCOME_MAX_CALLS: usize = 8;
pub const TREND_TTL: f64 = 3600.0;
pub const TREND_MAX_STALE: f64 = 86_400.0;
const KST: f64 = 9.0 * 3600.0;
const LOOKBACKS: [usize; 5] = [7, 14, 28, 56, 90];
const MARKET_COINS: [&str; 7] = ["eth", "sol", "btc", "xrp", "hype", "sndk", "soxl"];
const TRANSFER_TYPES: [&str; 7] = ["TRANSFER", "INTERNAL_TRANSFER", "CROSS_COLLATERAL_TRANSFER", "COIN_SWAP_DEPOSIT", "COIN_SWAP_WITHDRAW", "AUTO_EXCHANGE",
                                   "STRATEGY_UMFUTURES_TRANSFER"];

type Span = (String, String, i64, i64);

fn kst_date(t: f64) -> String {
    iso_utc((t + KST).floor())[..10].to_string()
}
fn kst_midnight(t: f64) -> f64 {
    ((t + KST) / 86_400.0).floor() * 86_400.0 - KST
}
fn int_of(v: Option<&Value>) -> i64 {
    v.filter(|x| truthy(Some(x))).and_then(|x| py_float(x).ok()).map_or(0, |f| f as i64)
}

pub fn trip_key(t: &Value) -> String {
    let g = |k: &str| t.get(k).map_or("None".into(), py_str);
    format!("{}|{}|{}", g("symbol"), g("side"), g("entry_time"))
}
pub fn trip_span(t: &Value) -> Span {
    let entry = int_of(t.get("entry_time"));
    let exit = t.get("exit_time").filter(|x| truthy(Some(x))).map_or(entry, |x| int_of(Some(x)));
    (t.get("symbol").map_or("None".into(), py_str), t.get("side").map_or("None".into(), py_str), entry, exit)
}
fn read_lossy(p: &PathBuf) -> Option<String> {
    std::fs::read(p).ok().map(|b| String::from_utf8_lossy(&b).into_owned())
}
pub fn load_trip_keys(dir: &PathBuf) -> HashMap<String, Span> {
    let mut idx = HashMap::new();
    for line in read_lossy(&dir.join(TRIP_LEDGER)).unwrap_or_default().lines() {
        if let Ok(t) = serde_json::from_str::<Value>(line) {
            if t.is_object() {
                idx.insert(trip_key(&t), trip_span(&t));
            }
        }
    }
    idx
}
fn overlaps(t: &Value, seen: &HashMap<String, Span>) -> bool {
    let (s, side, lo, hi) = trip_span(t);
    seen.values().any(|(s2, side2, lo2, hi2)| *s2 == s && *side2 == side && lo <= *hi2 && *lo2 <= hi)
}

/// 원본 record_account_trips: 닫힌 왕복 중 처음 보는 것만 덧붙인다. (적은 건수, 로그 줄).
pub fn record_trips(dir: &PathBuf, payload: &Value, seen: &mut HashMap<String, Span>, now: f64) -> (usize, Vec<String>) {
    let mut logs = vec![];
    if !truthy(payload.get("ok")) {
        return (0, logs);
    }
    let none = vec![];
    let mut fresh = vec![];
    for t in payload.get("trades").and_then(Value::as_array).unwrap_or(&none) {
        if !truthy(t.get("closed")) || seen.contains_key(&trip_key(t)) {
            continue;
        }
        if truthy(t.get("truncated_open")) {
            logs.push(format!("account_round_trips: 잘린 체결 스트림이라 건너뜀 {}", trip_key(t)));
            continue;
        }
        if overlaps(t, seen) {
            logs.push(format!("account_round_trips: 겹침으로 건너뜀 {} (절단된 체결 스트림의 조각으로 보인다)", trip_key(t)));
            continue;
        }
        let check = t.get("pnl_check_bp").filter(|v| !v.is_null()).and_then(|v| py_float(v).ok());
        if !check.is_some_and(|c| c.abs() <= CHECK_TOL_BP) {
            logs.push(format!("account_round_trips: 항등식 불일치로 건너뜀 {} (check={}bp) -- 체결 스트림이 잘린 것으로 보인다", trip_key(t),
                              t.get("pnl_check_bp").map_or("None".into(), py_str)));
            continue;
        }
        fresh.push(t);
    }
    if fresh.is_empty() {
        return (0, logs);
    }
    let equity = payload.get("balance").and_then(|b| b.get("margin")).cloned().unwrap_or(Value::Null);
    let mut written = 0;
    match std::fs::OpenOptions::new().create(true).append(true).open(dir.join(TRIP_LEDGER)) {
        Ok(mut f) => {
            for t in &fresh {
                let mut row = t.as_object().cloned().unwrap_or_default();
                row.insert("recorded_at".into(), json!(iso_utc(now)));
                row.insert("equity_at_record".into(), equity.clone());
                if let Err(e) = f.write_all(format!("{}\n", Value::Object(row)).as_bytes()) {
                    logs.push(format!("account_round_trips append failed: {e}"));
                    break;
                }
                seen.insert(trip_key(t), trip_span(t)); // 적힌 줄만 -- 중간 실패에도 두 번 안 적힌다
                written += 1;
            }
        }
        Err(e) => logs.push(format!("account_round_trips append failed: {e}")),
    }
    (written, logs)
}

/// 원본 trip_habit: {adds, peak_x}.
fn trip_habit(t: &Value, backfill: Option<&Value>) -> (Value, Value) {
    let empty = json!({});
    let src = if t.get("peak_notional").is_some_and(|v| !v.is_null()) { t } else { backfill.unwrap_or(&empty) };
    let mut eq = src.get("equity_entry").filter(|v| !v.is_null()).and_then(|v| py_float(v).ok());
    if eq.is_none() {
        if let Some(r) = t.get("equity_at_record").filter(|v| !v.is_null()).and_then(|v| py_float(v).ok()) {
            eq = Some(r - t.get("net_pnl").filter(|v| truthy(Some(v))).and_then(|v| py_float(v).ok()).unwrap_or(0.0));
        }
    }
    let peak = src.get("peak_notional").filter(|v| truthy(Some(v))).and_then(|v| py_float(v).ok());
    let peak_x = match (peak, eq) {
        (Some(p), Some(e)) if e > 0.0 => json!(round_n(p / e, 2)),
        _ => Value::Null,
    };
    (src.get("adds").cloned().unwrap_or(Value::Null), peak_x)
}

/// 원본 load_account_trip_rows: 원장 꼬리 400줄 → 화면 칸.
pub fn load_trip_rows(dir: &PathBuf) -> Vec<Value> {
    let Some(text) = read_lossy(&dir.join(TRIP_LEDGER)) else { return vec![] };
    let habits: Value = read_lossy(&dir.join(TRIP_HABITS)).and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(json!({}));
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let mut rows = vec![];
    for line in &lines[lines.len().saturating_sub(CARD_ROWS)..] {
        let Ok(t) = serde_json::from_str::<Value>(line) else { continue };
        if !t.is_object() {
            continue;
        }
        let mut row = Map::new();
        for k in ["symbol", "side", "entry_time", "exit_time", "entry_price", "exit_price", "max_qty", "net_pnl"] {
            row.insert(k.into(), t.get(k).cloned().unwrap_or(Value::Null));
        }
        let (adds, peak_x) = trip_habit(&t, habits.get(trip_key(&t)));
        row.insert("adds".into(), adds);
        row.insert("peak_x".into(), peak_x);
        rows.push(Value::Object(row));
    }
    rows
}

fn coin_of(symbol: &Value) -> String {
    py_str(symbol).replace("USDC", "").replace("USDT", "").to_lowercase()
}

/// trend_rule.trend_payload 의 신호 부호(+1 · −1) -- 일봉이 모자라거나 0 이하 종가면 None. 5표라 0 은 없다.
pub fn trend_sign(c: &[f64]) -> Option<i64> {
    let n = c.len();
    if n < LOOKBACKS[4] + 2 || c.iter().any(|x| !(*x > 0.0)) {
        return None;
    }
    let s: i64 = LOOKBACKS.iter().map(|l| if c[n - 1] > c[n - 1 - l] { 1 } else { -1 }).sum();
    Some(if s > 0 { 1 } else { -1 })
}

/// 원본 annotate_trip_trend: 진입 전 완결 일봉만(ends[k-1] < 진입).
pub fn annotate_trend(rows: &mut [Value], daily: &HashMap<String, (Vec<f64>, Vec<i64>)>) {
    for r in rows.iter_mut() {
        let tr = r.get("symbol").map(coin_of).and_then(|c| daily.get(&c)).and_then(|(closes, ends)| {
            let entry = int_of(r.get("entry_time"));
            let k = ends.partition_point(|e| *e < entry);
            trend_sign(&closes[..k])
        });
        r["trend"] = tr.map_or(Value::Null, |s| json!(s));
    }
}

/// 원본 account_day_start: 그날(KST) 첫 조회의 순자산을 남기고 같은 날엔 그대로.
pub fn day_start(dir: &PathBuf, payload: &Value, now: f64) -> Option<Value> {
    let ok = truthy(payload.get("ok"));
    let pools = if payload.is_object() && !truthy(payload.get("multi_assets_margin")) {
        payload.get("balances").and_then(Value::as_object).cloned().unwrap_or_default()
    } else {
        Map::new()
    };
    let parts: Vec<Value> = if !pools.is_empty() { pools.values().cloned().collect() } else { vec![if ok { payload.get("balance").cloned()? } else { return None }] };
    let mut wallet = 0.0;
    let mut unreal = 0.0;
    for b in &parts {
        wallet += py_float(b.get("wallet")?).ok()?;
        unreal += b.get("unrealized").filter(|v| truthy(Some(v))).map_or(Ok(0.0), py_float).ok()?;
    }
    let day = kst_date(now);
    let path = dir.join(DAY_START);
    let cur: Value = read_lossy(&path).and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(json!({}));
    if cur.get("date").and_then(Value::as_str) == Some(day.as_str()) {
        return Some(cur);
    }
    let mut names: Vec<&String> = pools.keys().collect();
    names.sort();
    let cur = json!({"date": day, "equity": wallet + unreal, "wallet": wallet, "ts": now, "late_s": (now - kst_midnight(now)).round_ties_even() as i64, "pools": names});
    let tmp = path.with_extension("tmp");
    if std::fs::write(&tmp, cur.to_string()).and_then(|_| std::fs::rename(&tmp, &path)).is_err() {
        return None; // 원본: OSError 면 칸만 빠진다
    }
    Some(cur)
}

/// 원본 income_fold: income 줄 → days[KST 날짜] 합 · 같은 줄(tranId|incomeType|time)은 한 번만. 깨진 줄이 있으면 아무것도 안 바꾼다.
pub fn income_fold(state: &mut Value, rows: &[Value]) -> Result<(), String> {
    let mut parsed = vec![];
    for r in rows {
        let g = |k: &str| r.get(k).ok_or(format!("KeyError: '{k}'"));
        let t = py_float(g("time")?)? as i64;
        parsed.push((t, format!("{}|{}|{}", py_str(g("tranId")?), py_str(g("incomeType")?), py_str(g("time")?)), py_str(g("incomeType")?),
                     r.get("asset").filter(|v| truthy(Some(v))).map_or(String::new(), py_str), py_float(g("income")?)?));
    }
    if !state["recent"].is_object() {
        state["recent"] = json!({});
    }
    if !state["days"].is_object() {
        state["days"] = json!({});
    }
    for (t, key, typ, asset, amt) in parsed {
        if state["recent"].get(&key).is_some() {
            continue;
        }
        state["recent"][&key] = json!(t);
        let day = kst_date(t as f64 / 1000.0);
        if state["days"].get(&day).is_none() {
            state["days"][&day] = json!({"pnl": 0.0, "realized": 0.0, "funding": 0.0, "commission": 0.0, "other": 0.0, "transfer": 0.0});
        }
        let d = &mut state["days"][&day];
        let add = |v: &mut Value, a: f64| *v = json!(v.as_f64().unwrap_or(0.0) + a);
        if asset != "USDT" && asset != "USDC" {
            if !d["other_assets"].is_object() {
                d["other_assets"] = json!({});
            }
            add(&mut d["other_assets"][&asset], amt);
        } else if TRANSFER_TYPES.contains(&typ.as_str()) {
            add(&mut d["transfer"], amt);
        } else {
            let f = match typ.as_str() {
                "REALIZED_PNL" => "realized",
                "FUNDING_FEE" => "funding",
                "COMMISSION" => "commission",
                _ => "other",
            };
            add(&mut d[f], amt);
            add(&mut d["pnl"], amt);
        }
    }
    Ok(())
}

/// 원본 income_refresh: state 를 제자리에서 앞으로 민다(창마다 일관 -- 오류가 나도 거기까지 저장해도 된다). 꽉 찬 창은 반으로 줄여 다시.
pub async fn income_refresh<R: AccountRest>(state: &mut Value, rest: &mut R, offset: i64, now_ms: i64, max_calls: usize) -> Result<(), String> {
    if state.get("cursor").is_none() {
        let day0 = kst_midnight(now_ms as f64 / 1000.0) - INCOME_BACKFILL_DAYS as f64 * 86_400.0;
        *state = json!({"from": kst_date(day0), "cursor": (day0 * 1000.0) as i64, "days": {}, "recent": {}});
    }
    let cursor = |s: &Value| s["cursor"].as_i64().unwrap_or(0);
    let mut end = (cursor(state) + INCOME_WINDOW_MS).min(now_ms);
    let mut out = Ok(());
    for _ in 0..max_calls {
        let rows = rest.get("/fapi/v1/income", &[("startTime", P::I(cursor(state))), ("endTime", P::I(end)), ("limit", P::I(INCOME_LIMIT as i64))], offset).await;
        let Some(rows) = rows.as_array() else {
            out = Err(rows.get("__error__").map_or_else(|| py_str(&rows), py_str));
            break;
        };
        if rows.len() >= INCOME_LIMIT && end > cursor(state) {
            end = cursor(state) + (end - cursor(state)) / 2;
            continue;
        }
        if let Err(e) = income_fold(state, rows) {
            out = Err(e);
            break;
        }
        if end >= now_ms {
            state["cursor"] = json!(cursor(state).max(now_ms - INCOME_OVERLAP_MS));
            state["synced_at"] = json!(now_ms);
            break;
        }
        state["cursor"] = json!(end + 1);
        end = (end + 1 + INCOME_WINDOW_MS).min(now_ms);
    }
    let c = cursor(state);
    if let Some(r) = state["recent"].as_object_mut() {
        r.retain(|_, t| t.as_i64().unwrap_or(0) >= c);
    }
    out
}

/// «건너뜀» 줄은 처음 한 번만 남긴다(나머지 줄은 그대로).
fn quiet_repeats(said: &mut std::collections::HashSet<String>, logs: &mut Vec<String>) {
    logs.retain(|l| !l.contains("건너뜀") || said.insert(l.clone()));
}

/// 화면 몫 상태(프로세스에 하나 · `SharedAccount.screen`).
pub struct Screen {
    pub dir: PathBuf,
    seen: Option<HashMap<String, Span>>,
    income: Option<Value>,
    income_ts: f64,
    daily: HashMap<String, (f64, Vec<f64>, Vec<i64>)>,
    pub logs: Vec<String>,
    said: std::collections::HashSet<String>, // 이미 찍은 «건너뜀» 줄 -- 원장 주기마다 같은 줄이 다시 나와 로그 93%를 채웠다(11-23~28)
}

impl Screen {
    pub fn new(dir: PathBuf) -> Self {
        Screen { dir, seen: None, income: None, income_ts: 0.0, daily: HashMap::new(), logs: vec![], said: Default::default() }
    }

    /// 원본 account_income: {income_days, income_from, income_synced_at} · 10분 넘게 못 따라잡았으면 {}.
    async fn income<R: AccountRest>(&mut self, rest: &mut R) -> Map<String, Value> {
        let now = rest.now();
        if now - self.income_ts >= INCOME_REFRESH_SEC {
            self.income_ts = now;
            let path = self.dir.join(INCOME_STATE);
            let mut state = self.income.take().unwrap_or_else(|| read_lossy(&path).and_then(|s| serde_json::from_str(&s).ok()).filter(Value::is_object).unwrap_or(json!({})));
            let offset = rest.clock_offset().await;
            if let Err(e) = income_refresh(&mut state, rest, offset, (now * 1000.0) as i64, INCOME_MAX_CALLS).await {
                self.logs.push(format!("account_income: {e}"));
            }
            let tmp = path.with_extension("tmp");
            if let Err(e) = std::fs::write(&tmp, state.to_string()).and_then(|_| std::fs::rename(&tmp, &path)) {
                self.logs.push(format!("account_income save: {e}"));
            }
            self.income = Some(state);
        }
        let s = self.income.as_ref().cloned().unwrap_or(json!({}));
        let synced = s.get("synced_at").and_then(Value::as_f64).unwrap_or(0.0);
        let mut m = Map::new();
        if now * 1000.0 - synced <= 600_000.0 {
            m.insert("income_days".into(), s["days"].clone());
            m.insert("income_from".into(), s["from"].clone());
            m.insert("income_synced_at".into(), s["synced_at"].clone());
        }
        m
    }

    /// 완결 일봉 종가·종가 시각(봉 끝+1) 300개 · 1시간 캐시(실패면 하루까지 옛 값). ponytail: 원본 swr 은 묵은 값을 주고 뒤에서 갱신 -- 여기는 기다려 읽는다(코인당 시간에 한 번).
    async fn daily<R: AccountRest>(&mut self, rest: &mut R, coin: &str) -> Option<(Vec<f64>, Vec<i64>)> {
        let now = rest.now();
        let have = self.daily.get(coin).cloned();
        if have.as_ref().is_some_and(|h| now - h.0 < TREND_TTL) {
            return have.map(|h| (h.1, h.2));
        }
        let sym = format!("{}USDT", coin.to_uppercase());
        match rest.public("/fapi/v1/klines", &[("symbol", P::s(&sym)), ("interval", P::s("1d")), ("limit", P::I(300))]).await {
            Ok(Value::Array(raw)) => {
                let (mut c, mut e) = (vec![], vec![]);
                for r in &raw {
                    let (Some(end), Some(close)) = (r.get(6).and_then(|v| py_float(v).ok()), r.get(4).and_then(|v| py_float(v).ok())) else { continue };
                    if end < now * 1000.0 {
                        c.push(close);
                        e.push(end as i64 + 1);
                    }
                }
                self.daily.insert(coin.into(), (now, c.clone(), e.clone()));
                Some((c, e))
            }
            other => {
                self.logs.push(format!("trip trend {coin}: {}", other.map_or_else(|e| e, |v| py_str(&v))));
                have.filter(|h| now - h.0 < TREND_TTL + TREND_MAX_STALE).map(|h| (h.1, h.2))
            }
        }
    }

    /// 원본 produce_account 의 화면 몫(fetch 뒤): income(정상일 때) → 원장 기록 → ledger + trend → day_start.
    pub async fn enrich<R: AccountRest>(&mut self, p: &mut Value, rest: &mut R) {
        if !p.is_object() {
            return;
        }
        if truthy(p.get("ok")) {
            for (k, v) in self.income(rest).await {
                p[&k] = v;
            }
        }
        let seen = self.seen.get_or_insert_with(|| load_trip_keys(&self.dir));
        let (added, mut logs) = record_trips(&self.dir, p, seen, rest.now());
        if added > 0 {
            logs.push(format!("account_round_trips: +{added}건 (누적 {}건)", seen.len()));
        }
        // ponytail: 프로세스 수명 동안 쌓이는 집합(7일 창 왕복 키라 작다) -- 재시작하면 한 번씩 다시 찍힌다
        quiet_repeats(&mut self.said, &mut logs);
        self.logs.append(&mut logs);
        let mut rows = load_trip_rows(&self.dir);
        let mut daily = HashMap::new();
        let mut coins: Vec<String> = rows.iter().map(|r| r.get("symbol").map_or(String::new(), coin_of)).collect();
        coins.sort();
        coins.dedup();
        for c in coins.into_iter().filter(|c| MARKET_COINS.contains(&c.as_str())) {
            if let Some(d) = self.daily(rest, &c).await {
                daily.insert(c, d);
            }
        }
        annotate_trend(&mut rows, &daily);
        p["ledger"] = Value::Array(rows);
        if let Some(ds) = day_start(&self.dir, p, rest.now()) {
            p["day_start"] = ds;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kst_and_trend() {
        assert_eq!(kst_date(1_763_823_600.0), "2025-11-23"); // 2025-11-22 15:00 UTC = 11-23 00:00 KST
        assert_eq!(kst_date(1_763_823_599.9), "2025-11-22");
        assert_eq!(kst_midnight(1_763_830_000.0), 1_763_823_600.0);
        let up: Vec<f64> = (0..120).map(|i| 100.0 * 1.01f64.powi(i)).collect();
        assert_eq!(trend_sign(&up), Some(1));
        assert_eq!(trend_sign(&up.iter().rev().cloned().collect::<Vec<_>>()), Some(-1));
        assert_eq!(trend_sign(&up[..91]), None);
    }
}

#[cfg(test)]
#[test]
fn skip_lines_once() {
    let mut said = Default::default();
    let mk = || vec!["account_round_trips: 잘린 체결 스트림이라 건너뜀 A".to_string(), "account_round_trips: +1건 (누적 2건)".into()];
    let (mut a, mut b) = (mk(), mk());
    quiet_repeats(&mut said, &mut a);
    quiet_repeats(&mut said, &mut b);
    assert_eq!((a.len(), b), (2, vec!["account_round_trips: +1건 (누적 2건)".to_string()]));
}
