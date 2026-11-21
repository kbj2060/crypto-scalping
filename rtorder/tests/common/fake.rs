// 가짜 거래소(시계 포함) + 불변식 검사 -- exec_invariants.rs 와 http_mesh.rs 가 `include!` 로 같이 쓴다(2026-11-20 둘째 조각에서 떼어 냄, 내용 불변).
use rtorder::exec::*;
use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

fn block_on<F: Future>(f: F) -> F::Output {
    let mut f = std::pin::pin!(f);
    let mut cx = Context::from_waker(Waker::noop());
    match f.as_mut().poll(&mut cx) {
        Poll::Ready(v) => v,
        Poll::Pending => panic!("가짜 거래소는 기다리지 않는다"),
    }
}

const CALL_CAP: u32 = 20_000;

#[derive(Clone, Copy)]
enum Cmd {
    Action(Action),
    Amend(f64, Option<f64>),
    FillAfter(u32),
    Drift(bool),
}

#[derive(Debug, Clone)]
struct FOrder {
    qty: f64,
    filled: f64,
    status: String,
    cid: String,
    polls: u32,
    cancel_sent: bool,
}

#[derive(Default)]
struct Fake {
    t: f64,
    calls: u32,
    orders: BTreeMap<i64, FOrder>,
    next_id: i64,
    placed: Vec<LimitOrder>,
    fill_after: u32, // 0 = 체결 안 됨
    partial: Option<f64>,
    fail: Vec<&'static str>, // place · cancel · get_after_cancel · get · book
    ok_first: u32,           // 실패 종류마다 처음 몇 번은 성공 -- 재호가 **도는 중**에 터뜨린다
    counts: HashMap<&'static str, u32>,
    post_only_rejects: u32,
    place_err: Option<(String, bool)>, // (오류 문자열, 실제로는 접수됐나 = 유령)
    lookup_miss: u32,                  // 표식 조회 처음 n 번은 있어도 «없음»(거래소가 아직 처리 전, D2 (c))
    drift: bool,
    fill_on_cancel: bool,
    kill_at: Option<u32>,
    wake_every: Option<f64>,
    shared: Option<Shared>,
    script: Vec<(f64, Cmd)>,
    book_times: Vec<f64>,
    gets: u32,
    deletes: Vec<OrderRef>,
    lookups: u32,
    on_fills: u32,
    reprice_vals: Vec<f64>,
    reprice_calls: u32,
    get_fail_first: u32,              // 처음 n 번의 상태 조회 실패(E12)
    reject_at_placed: Option<usize>, // 지정가가 n 건 걸린 뒤 다음 접수 한 번을 -5022 로
    markets: Vec<(String, String, f64)>, // 시장가 다리 (측면, 포지션 측면, 수량) -- 즉시 전량 체결
    market_err: Option<String>,          // 시장가 접수 오류
    market_exec: Option<f64>,            // 응답 executedQty(None = 보낸 수량 · Some(0) = ACK)
}

impl Fake {
    fn tick(&mut self) {
        self.calls += 1;
        assert!(self.calls <= CALL_CAP, "가짜 호출 상한 {CALL_CAP} 초과 -- 루프가 안 끝난다");
        self.t += 0.001; // 요청 왕복(표식 ms 가 겹치지 않게)
        self.apply();
    }
    fn apply(&mut self) {
        let now = self.t;
        let due: Vec<Cmd> = self.script.iter().filter(|(at, _)| *at <= now).map(|x| x.1).collect();
        self.script.retain(|(at, _)| *at > now);
        for c in due {
            match c {
                Cmd::Action(a) => self.shared.as_ref().unwrap().lock().unwrap().action = Some(a),
                Cmd::Amend(p, q) => self.shared.as_ref().unwrap().lock().unwrap().amend = Some(Amend { price: p, qty: q, auto: false }),
                Cmd::FillAfter(n) => {
                    self.fill_after = n;
                    for o in self.orders.values_mut() { o.polls = 0; }
                }
                Cmd::Drift(d) => self.drift = d,
            }
        }
    }
    fn fails(&mut self, k: &'static str) -> bool {
        if !self.fail.contains(&k) {
            return false;
        }
        let n = self.counts.entry(k).or_default();
        *n += 1;
        *n > self.ok_first
    }
    fn live(&self) -> Vec<i64> {
        self.orders.iter().filter(|(_, o)| !TERMINAL.contains(&o.status.as_str())).map(|(i, _)| *i).collect()
    }
    /// 거래소 쪽 사실: 체결된 양 + 아직 체결될 수 있는 양(살아 있는 지정가 잔량).
    fn exposure(&self) -> (f64, f64) {
        let filled = self.orders.values().map(|o| o.filled).sum::<f64>() + self.markets.iter().map(|m| self.market_exec.unwrap_or(m.2)).sum::<f64>();
        let resting = self.orders.values().filter(|o| !TERMINAL.contains(&o.status.as_str())).map(|o| o.qty - o.filled).sum();
        (filled, resting)
    }
    fn reply(id: i64, o: &FOrder) -> Order {
        Order { order_id: Some(id), status: o.status.clone(), executed_qty: o.filled }
    }
    fn find(&self, r: &OrderRef) -> Option<i64> {
        match r {
            OrderRef::Id(id) => id.filter(|i| self.orders.contains_key(i)),
            OrderRef::Client(c) => self.orders.iter().find(|(_, o)| &o.cid == c).map(|(i, _)| *i),
        }
    }
}

impl Exchange for Fake {
    async fn place_market(&mut self, c: &Common<'_>, quantity: f64) -> Reply {
        self.tick();
        assert_eq!(c.symbol, "ETHUSDC", "주문 심볼이 계획과 다르다");
        if let Some(e) = &self.market_err {
            return Err(e.clone());
        }
        self.markets.push((c.side.into(), c.position_side.into(), quantity));
        self.next_id += 1;
        Ok(Order { order_id: Some(9_000_000 + self.next_id), status: "FILLED".into(), executed_qty: self.market_exec.unwrap_or(quantity) })
    }
    async fn place_limit_gtx(&mut self, o: &LimitOrder) -> Reply {
        self.tick();
        assert_eq!(o.symbol, "ETHUSDC", "주문 심볼이 계획과 다르다");
        if self.post_only_rejects > 0 {
            self.post_only_rejects -= 1;
            return Err("400 [-5022] Due to the order could not be executed as maker".into());
        }
        if self.reject_at_placed == Some(self.placed.len()) {
            self.reject_at_placed = None;
            return Err("400 [-5022] Due to the order could not be executed as maker".into());
        }
        if self.fails("place") {
            // 확정 거부(4xx). 2026-11-20 전엔 "500 boom" -- 5xx 는 «체결 상태 불명»이라(D2 해결) 확정 거부 사례로 쓸 수 없다
            return Err("400 [-2019] Margin is insufficient.".into());
        }
        let ghost = self.place_err.take();
        self.next_id += 1;
        if ghost.as_ref().is_none_or(|g| g.1) {
            self.orders.insert(self.next_id, FOrder { qty: o.quantity, filled: 0.0, status: "NEW".into(), cid: o.client_order_id.clone(), polls: 0, cancel_sent: false });
            self.placed.push(o.clone());
        }
        match ghost {
            Some((e, _)) => Err(e),
            None => Ok(Order { order_id: Some(self.next_id), status: "NEW".into(), executed_qty: 0.0 }),
        }
    }
    async fn get_order(&mut self, _c: &Common<'_>, r: &OrderRef) -> Reply {
        self.tick();
        if matches!(r, OrderRef::Client(_)) {
            self.lookups += 1;
            if self.fails("lookup") {
                return Err("TimeoutError: lookup".into());
            }
            if self.lookups <= self.lookup_miss {
                return Err("400 [-2013] Order does not exist.".into());
            }
        }
        let Some(id) = self.find(r) else { return Err("400 [-2013] Order does not exist.".into()) };
        if self.orders[&id].cancel_sent && self.fails("get_after_cancel") {
            return Err("500 read fail".into());
        }
        if self.get_fail_first > 0 {
            self.get_fail_first -= 1;
            return Err("TimeoutError: read".into());
        }
        if self.fails("get") {
            return Err("TimeoutError: read".into());
        }
        self.gets += 1;
        let (fa, partial, kill) = (self.fill_after, self.partial, self.kill_at);
        let o = self.orders.get_mut(&id).unwrap();
        o.polls += 1;
        if !TERMINAL.contains(&o.status.as_str()) && fa > 0 && o.polls >= fa {
            match partial {
                Some(p) if o.filled == 0.0 => (o.filled, o.status) = ((o.qty * p * 1e8).round() / 1e8, "PARTIALLY_FILLED".into()),
                Some(_) => {}
                None => (o.filled, o.status) = (o.qty, "FILLED".into()),
            }
        }
        if kill.is_some_and(|k| o.polls >= k) && !TERMINAL.contains(&o.status.as_str()) {
            o.status = "CANCELED".into();
        }
        Ok(Fake::reply(id, o))
    }
    async fn cancel_order(&mut self, _c: &Common<'_>, r: &OrderRef) -> Reply {
        self.tick();
        self.deletes.push(r.clone());
        if matches!(r, OrderRef::Client(_)) && self.fails("delete_cid") {
            return Err("TimeoutError: delete".into());
        }
        let id = self.find(r);
        if let Some(id) = id {
            let foc = self.fill_on_cancel;
            let o = self.orders.get_mut(&id).unwrap();
            o.cancel_sent = true;
            if foc && !TERMINAL.contains(&o.status.as_str()) {
                (o.filled, o.status) = (o.qty, "FILLED".into()); // 취소 직전 체결
                return Err("400 [-2011] Unknown order sent.".into());
            }
        }
        if self.fails("cancel") {
            return Err("500 cancel fail".into());
        }
        let Some(id) = id else { return Err("400 [-2011] Unknown order sent.".into()) };
        let o = self.orders.get_mut(&id).unwrap();
        if !TERMINAL.contains(&o.status.as_str()) {
            o.status = "CANCELED".into();
        }
        Ok(Fake::reply(id, o))
    }
    async fn book(&mut self, s: &str) -> Result<(f64, f64), String> {
        self.tick();
        assert_eq!(s, "ETHUSDC", "호가를 주문 심볼이 아닌 곳에서 읽는다(E14 · 2026-09-19·20)");
        self.book_times.push(self.t);
        if self.fails("book") {
            return Err("ClientError: book".into());
        }
        let (side, px) = self.placed.last().map(|o| (o.side.clone(), o.price)).unwrap_or(("BUY".into(), 2500.0));
        let px = (px * 100.0).round() / 100.0;
        Ok(match (self.drift, side.as_str()) {
            (true, "BUY") => (px + 0.01, px + 0.02), // 누가 내 위로 -- 매번 리페그 유발
            (true, _) => (px - 0.02, px - 0.01),
            (false, "BUY") => (px, px + 0.01),
            (false, _) => (px - 0.01, px),
        })
    }
    fn now(&self) -> f64 {
        self.t
    }
    fn wall_ms(&self) -> i64 {
        1_800_000_000_000 + (self.t * 1000.0) as i64
    }
    async fn nap(&mut self, sec: f64) {
        self.t += self.wake_every.map_or(sec, |w| w.min(sec));
        self.apply();
    }
    async fn sleep(&mut self, sec: f64) {
        self.t += sec;
        self.apply();
    }
    async fn on_fill(&mut self) {
        self.on_fills += 1;
    }
    async fn reprice(&mut self) -> Result<Option<f64>, String> {
        self.reprice_calls += 1;
        Ok(self.reprice_vals.get((self.reprice_calls - 1) as usize).or(self.reprice_vals.last()).copied())
    }
}

fn plan(side: &str, pside: &str, qty: f64) -> ExecPlan {
    ExecPlan { symbol: "ETHUSDC".into(), side: side.into(), position_side: pside.into(), market: false, price: 2500.0, quantity: qty,
               fallback_after_sec: 0.0, maker_only: false, vol_bpm: None, market_reason: None, target_leverage: 0 }
}
fn exit_plan() -> ExecPlan {
    plan("SELL", "LONG", 2.0)
}
fn entry_plan() -> ExecPlan {
    plan("BUY", "LONG", 2.0)
}

enum Kind {
    Entry,
    Exit,
    Wait(f64, bool),
    Limit(f64, bool, bool),
}

fn go(fx: &mut Fake, p: &ExecPlan, k: Kind) -> ExecState {
    let s: Shared = Arc::new(Mutex::new(ExecState::default()));
    fx.shared = Some(s.clone());
    block_on(async {
        match k {
            Kind::Entry => run_entry(fx, p, &s).await,
            Kind::Exit => run_exit(fx, p, &s).await,
            Kind::Wait(w, of) => run_entry_wait(fx, p, &s, w, of).await,
            Kind::Limit(m, of, rp) => run_entry_limit(fx, p, &s, m, of, rp).await,
        }
    });
    let out = s.lock().unwrap().clone();
    out
}

/// 불변식 셋 + 장부 대조. 모든 사례가 이것을 지난다.
fn check(name: &str, st: &ExecState, fx: &Fake, total: f64) {
    assert!(st.filled <= total + 1e-9, "{name}: 체결 {} > 계획 {total}", st.filled);
    let (real, resting) = fx.exposure();
    assert!(real + resting <= total + 1e-6, "{name}: 거래소 체결 {real} + 살아 있는 잔량 {resting} > 계획 {total}"); // ① 자기 채점이 아니라 거래소 쪽에서
    let live = fx.live();
    assert!(live.len() <= 1, "{name}: 살아남은 지정가 {live:?} -- 취소 실패 뒤에도 재호가했다"); // ②
    if !live.is_empty() {
        assert_eq!(st.phase, Phase::Error, "{name}: 주문이 살아 있는데 phase={:?} -- 사람에게 넘겨야 한다", st.phase);
    }
    assert!(st.done && !matches!(st.phase, Phase::Working | Phase::Idle), "{name}: 종료 안 됨 {:?}", st.phase); // ③
    if st.phase == Phase::Error {
        assert!(st.filled <= real + 1e-9, "{name}: 오류인데 적힌 체결 {} > 거래소 {real}", st.filled); // E9: 하한
    } else {
        assert!((st.filled - real).abs() < 1e-9, "{name}: 적힌 체결 {} != 거래소 체결 {real} -- 호출부가 손절을 잘못 건다", st.filled);
    }
    if st.phase == Phase::FilledTaker {
        assert!(live.is_empty() && !fx.markets.is_empty() && st.taker_order_id.is_some(), "{name}: 시장가 다리 전에 지정가를 다 지웠어야 {st:?}");
    }
}

