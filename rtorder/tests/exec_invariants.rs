//! 지정가 집행 불변식(파이썬 test_manual_peg_execute_invariants_20260913.py · test_manual_entry_cancel_20261011.py ·
//! test_manual_entry_wait_20261010.py · test_manual_limit_entry_20261031.py 의 사례 + 규칙 문서 «시험 없는 규칙» 중 집행 몫).
//! 고정하는 셋: ① 총 체결 ≤ 계획(거래소 쪽에서 다시 센다) ② 살아남은 지정가 ≤ 1 이고 그때는 phase=error ③ 어떤 경로로도 종료 상태.
//! 🔴메모: 2026-09-13 취소 확인 없이 재호가해 지정가 41개가 동시에 살았다.
//! 네트워크 0 · 시계는 가짜(3초·120초를 기다리지 않는다) · 가짜 호출은 상한 20,000번(넘으면 panic -- 끝없이 도는 재호가 방지).

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
        let filled = self.orders.values().map(|o| o.filled).sum();
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
    async fn get_order(&mut self, _s: &str, r: &OrderRef) -> Reply {
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
    async fn cancel_order(&mut self, _s: &str, r: &OrderRef) -> Reply {
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
               fallback_after_sec: 0.0, maker_only: false, vol_bpm: None, market_reason: None }
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
    if st.phase == Phase::NeedsMarketDecision {
        assert!(live.is_empty() && (st.market_qty - (total - st.filled)).abs() < 1e-9 && st.market_branch.is_some(), "{name}: 시장가 결정 대기 칸 {st:?}");
    }
}

// ── 파이썬 inv #1~#11 ───────────────────────────────────────────────────────

#[test]
fn inv01_normal_fill() {
    let mut fx = Fake { fill_after: 1, ..Default::default() };
    let st = go(&mut fx, &exit_plan(), Kind::Exit);
    check("정상", &st, &fx, 2.0);
    assert!(st.phase == Phase::FilledMaker && st.filled == 2.0 && fx.placed.len() == 1);
}

#[test]
fn inv02_drift_repegs_then_needs_market_decision() {
    let mut fx = Fake { drift: true, ..Default::default() };
    let st = go(&mut fx, &exit_plan(), Kind::Exit);
    check("리페그", &st, &fx, 2.0);
    // 재호가가 **실제로 돈다**(파이썬 시험이 가짜 시그니처 불일치로 재호가 0회인 채 통과하던 사고, 2026-11-20 f6aa1c14)
    // 마지막 앞지름은 리페그 수를 올린 뒤 마감에 걸려 새로 걸지 않는다(원본과 같다) -- 그래서 걸린 수 = 리페그 수 또는 +1
    assert!(st.repegs >= 30 && (fx.placed.len() as u32 == st.repegs || fx.placed.len() as u32 == st.repegs + 1), "repegs {} placed {}", st.repegs, fx.placed.len());
    assert_eq!((st.phase, st.market_branch), (Phase::NeedsMarketDecision, Some("M9"))); // 파이썬 = 잔량 시장가(filled_taker)
    assert!(fx.placed.windows(2).all(|w| w[1].price < w[0].price), "매도 재호가는 새 최우선 매도호가로 내려간다");
}

fn fail_cases(name: &str, kind: &'static str, phase: Phase, entry: bool) {
    for n in [0, 1, 3] {
        let label = format!("{name}(재호가 {n}회 뒤)");
        let mut fx = Fake { drift: true, fail: vec![kind], ok_first: n, ..Default::default() };
        let st = go(&mut fx, &if entry { entry_plan() } else { exit_plan() }, if entry { Kind::Entry } else { Kind::Exit });
        check(&label, &st, &fx, 2.0);
        assert_eq!(st.phase, phase, "{label}");
        assert_eq!(st.repegs, n, "{label}: 재호가 수 -- 0 이면 주입이 재호가 경로를 안 지났다");
        assert_eq!(fx.placed.len() as u32, n + (kind != "place") as u32, "{label}: 실패 뒤에 또 걸었다");
        assert_eq!(fx.live().len(), (kind == "cancel") as usize, "{label}");
        assert_eq!(st.market_branch, None, "{label}: 멈춰야 하는데 시장가 결정으로 갔다");
    }
}

#[test]
fn inv03_cancel_fail_during_repeg_stops_with_error() {
    fail_cases("취소실패+리페그", "cancel", Phase::Error, false);
}

#[test]
fn inv04_partial_fill_then_drift_rests_only_remainder() {
    let mut fx = Fake { fill_after: 1, partial: Some(0.5), drift: true, ..Default::default() };
    let st = go(&mut fx, &exit_plan(), Kind::Exit);
    check("부분체결+리페그", &st, &fx, 2.0);
    assert!(st.repegs >= 1 && fx.placed[1].quantity == 1.0 && fx.placed.iter().map(|o| o.quantity).sum::<f64>() <= 2.0 * (REPEG_MAX as f64 + 2.0));
}

#[test]
fn inv05_entry_cancel_fail() {
    fail_cases("진입/취소실패", "cancel", Phase::Error, true);
}

#[test]
fn inv06_place_fail_stops() {
    fail_cases("접수실패", "place", Phase::Rejected, false);
}

#[test]
fn inv07_reread_fail_after_cancel() {
    fail_cases("재조회실패", "get_after_cancel", Phase::Error, false);
}

#[test]
fn inv08_partial_exit_plan_never_exceeds() {
    let mut fx = Fake { fill_after: 1, ..Default::default() };
    let st = go(&mut fx, &ExecPlan { quantity: 1.0, ..exit_plan() }, Kind::Exit);
    check("부분청산", &st, &fx, 1.0);
    assert!(fx.placed.iter().map(|o| o.quantity).sum::<f64>() <= 1.0 + 1e-9);
}

#[test]
fn inv09_market_exit_plan_places_nothing() {
    let mut fx = Fake { fill_after: 1, ..Default::default() };
    let st = go(&mut fx, &ExecPlan { market: true, market_reason: Some("변동성".into()), ..exit_plan() }, Kind::Exit);
    check("시장가계획", &st, &fx, 2.0);
    assert!(fx.placed.is_empty() && st.market_branch == Some("M6/M7/M8") && st.market_qty == 2.0);
}

#[test]
fn inv10_entry_repegs_and_books_match_exchange() {
    let mut fx = Fake { drift: true, ..Default::default() };
    let st = go(&mut fx, &entry_plan(), Kind::Entry);
    check("진입/리페그", &st, &fx, 2.0);
    assert!(st.repegs >= 30 && st.market_branch == Some("M1"), "{st:?}");
    assert!(fx.placed.windows(2).all(|w| w[1].price > w[0].price), "매수 재호가는 위로 따라간다");
}

#[test]
fn inv11_post_only_reject_repegs_then_caps() {
    for (k, phase) in [(3, Phase::FilledMaker), (REPEG_MAX + 5, Phase::Rejected)] {
        let mut fx = Fake { fill_after: 1, post_only_rejects: k, ..Default::default() };
        let st = go(&mut fx, &exit_plan(), Kind::Exit);
        check("post-only", &st, &fx, 2.0);
        assert_eq!((st.phase, st.repegs, fx.placed.len()), (phase, k.min(REPEG_MAX), (phase == Phase::FilledMaker) as usize));
        assert!(fx.book_times.len() as u32 == k.min(REPEG_MAX), "-5022 마다 새 호가를 읽는다");
    }
}

// ── 규칙 문서 «시험 없는 규칙» 중 집행 몫 + 추가 실패 주입 ──────────────────────

#[test]
fn e06_unknown_outcome_lookup_adopts_ghost_order() {
    // 시간초과(숫자로 시작 안 함)인데 실제로는 접수됨 → 표식 조회로 그 주문을 이어 간다(두 번 걸지 않는다)
    let mut fx = Fake { fill_after: 2, place_err: Some(("TimeoutError: read".into(), true)), ..Default::default() };
    let st = go(&mut fx, &exit_plan(), Kind::Exit);
    check("유령 입양", &st, &fx, 2.0);
    assert!(st.phase == Phase::FilledMaker && fx.placed.len() == 1 && fx.lookups == 1);
    // 접수 안 됐으면 세 번 본 뒤 표식 DELETE → error(2026-11-20 D2 해결 전: 한 번 보고 DELETE 없이 rejected)
    let mut fx = Fake { fill_after: 1, place_err: Some(("TimeoutError: read".into(), false)), ..Default::default() };
    let st = go(&mut fx, &exit_plan(), Kind::Exit);
    check("유령 없음", &st, &fx, 2.0);
    assert!(st.phase == Phase::Error && fx.lookups == 3 && fx.deletes.len() == 1);
}

#[test]
fn d2_unknown_outcome_same_on_all_paths() {
    // D2 해결(2026-11-20): «나갔는지 모르는 주문»(5xx·-1007·-1006·네트워크) 처리가 세 경로(fill_maker = 청산·진입 · 기다리기 · 지정가 대기)에서 같다 --
    // 표식으로 3번 조회 → 찾으면 이어 감 → 끝내 모르면 표식 DELETE 뒤 error. 해결 전엔 fill_maker 는 네트워크 오류만 한 번 · 기다리기는 조회 없음(유령 생존 + rejected).
    let to = || "TimeoutError: read".to_string();
    // (이름, 설정, 기대 phase, 표식 조회 수, 표식 DELETE 수, 끝에 살아 있는 지정가 수)
    let cases: Vec<(&str, Fake, Phase, u32, usize, usize)> = vec![
        ("a 503·접수됨", Fake { place_err: Some(("503 Service Unavailable".into(), true)), ..Default::default() }, Phase::FilledMaker, 1, 0, 0),
        ("b -1007·접수됨", Fake { place_err: Some(("408 [-1007] Timeout waiting for response from backend server. Send status unknown; execution status unknown.".into(), true)), ..Default::default() }, Phase::FilledMaker, 1, 0, 0),
        ("b' -1006·접수됨", Fake { place_err: Some(("400 [-1006] An unexpected response was received from the message bus. Execution status unknown.".into(), true)), ..Default::default() }, Phase::FilledMaker, 1, 0, 0),
        ("c 시간초과·둘째 조회에 보임", Fake { place_err: Some((to(), true)), lookup_miss: 1, ..Default::default() }, Phase::FilledMaker, 2, 0, 0),
        ("d 접수 안 됨·3번 다 없음", Fake { place_err: Some((to(), false)), ..Default::default() }, Phase::Error, 3, 1, 0),
        ("e 조회·DELETE 전부 실패", Fake { place_err: Some((to(), true)), fail: vec!["lookup", "delete_cid"], ..Default::default() }, Phase::Error, 3, 1, 1),
        ("음성: 4xx 확정 거부", Fake { place_err: Some(("400 [-2019] Margin is insufficient.".into(), false)), ..Default::default() }, Phase::Rejected, 0, 0, 0),
    ];
    for (path, tag) in [("exit", PEG_TAG), ("entry", PEG_TAG), ("wait", WAIT_TAG), ("limit", WAIT_TAG)] {
        for (name, base, phase, lookups, dels, live_n) in &cases {
            let label = format!("불명 POST {path}/{name}");
            let mut fx = Fake { fill_after: 1, place_err: base.place_err.clone(), lookup_miss: base.lookup_miss, fail: base.fail.clone(), ..Default::default() };
            let (p, k) = match path {
                "exit" => (exit_plan(), Kind::Exit),
                "entry" => (entry_plan(), Kind::Entry),
                "wait" => (entry_plan(), Kind::Wait(900.0, false)),
                _ => (entry_plan(), Kind::Limit(60.0, false, false)),
            };
            let st = go(&mut fx, &p, k);
            check(&label, &st, &fx, 2.0);
            assert!(!(fx.live().len() > 0 && st.phase == Phase::Rejected), "{label}: 살아 있는 지정가인데 rejected");
            assert_eq!(st.phase, *phase, "{label}: {:?}", st.error);
            assert_eq!((fx.lookups, fx.deletes.len(), fx.live().len()), (*lookups, *dels, *live_n), "{label}: (표식 조회, DELETE, 살아 있는 지정가)");
            assert!(fx.deletes.iter().all(|d| matches!(d, OrderRef::Client(c) if c.starts_with(tag))), "{label}: DELETE 는 이 경로 표식으로 {:?}", fx.deletes);
            assert!(fx.placed.len() <= 1 && st.market_branch.is_none(), "{label}: 두 번 걸었거나 시장가 결정으로 갔다");
            if *phase == Phase::Error {
                assert!(st.error.as_deref().unwrap().contains("접수 여부를 확인하지 못했습니다"), "{label}");
            }
        }
    }
}

#[test]
fn fill_just_before_cancel_is_counted() {
    let mut fx = Fake { drift: true, fill_on_cancel: true, ..Default::default() };
    let st = go(&mut fx, &exit_plan(), Kind::Exit);
    check("취소 직전 체결", &st, &fx, 2.0);
    assert!(st.phase == Phase::FilledMaker && fx.placed.len() == 1 && st.filled == 2.0, "재조회 상태(FILLED)를 믿고 다시 걸지 않는다");
}

#[test]
fn partial_then_book_runs_away() {
    let mut fx = Fake { fill_after: 1, partial: Some(0.4), script: vec![(3.5, Cmd::Drift(true))], ..Default::default() };
    let st = go(&mut fx, &entry_plan(), Kind::Entry);
    check("부분 체결 뒤 달아남", &st, &fx, 2.0);
    assert!(st.repegs >= 1 && fx.placed[1].quantity == 1.2 && fx.placed[1].price > fx.placed[0].price);
}

#[test]
fn e16_early_wake_checks_book_only_every_poll_sec() {
    let mut fx = Fake { drift: false, wake_every: Some(0.5), ..Default::default() };
    let st = go(&mut fx, &exit_plan(), Kind::Exit);
    check("알림 일찍 깸", &st, &fx, 2.0);
    let gaps: Vec<f64> = fx.book_times.windows(2).map(|w| w[1] - w[0]).collect();
    assert!(!gaps.is_empty() && gaps.iter().all(|g| *g >= POLL_SEC - 1e-6), "리페그 판단 간격 {gaps:?}");
    assert!(fx.gets as usize > 4 * fx.book_times.len(), "상태 조회는 알림마다");
}

#[test]
fn e12_status_read_failure_is_retried() {
    // 처음 5번 조회 실패 → 주문은 살아 있다고 보고 재시도 → 체결
    let mut fx = Fake { fill_after: 1, get_fail_first: 5, ..Default::default() };
    let st = go(&mut fx, &exit_plan(), Kind::Exit);
    check("조회 실패 뒤 회복", &st, &fx, 2.0);
    assert!(st.phase == Phase::FilledMaker && fx.placed.len() == 1);
    // 끝까지 실패 → 버리고 재호가하지 않는다 → 취소 확인도 못 함 → error(사람에게)
    let mut fx = Fake { fill_after: 1, fail: vec!["get"], ..Default::default() };
    let st = go(&mut fx, &ExecPlan { fallback_after_sec: 30.0, ..exit_plan() }, Kind::Exit);
    check("조회 계속 실패", &st, &fx, 2.0);
    assert!(st.phase == Phase::Error && fx.placed.len() == 1);
}

#[test]
fn e05_book_failure_after_post_only_reject_is_rejected() {
    let mut fx = Fake { post_only_rejects: 1, fail: vec!["book"], ..Default::default() };
    let st = go(&mut fx, &exit_plan(), Kind::Exit);
    check("호가 실패", &st, &fx, 2.0);
    assert!(st.phase == Phase::Rejected && st.error.as_deref().unwrap().starts_with("호가 조회 실패"));
}

#[test]
fn e09_unconfirmed_cancel_reports_last_seen_fill() {
    let mut fx = Fake { fill_after: 1, partial: Some(0.5), drift: true, fail: vec!["cancel"], ..Default::default() };
    let st = go(&mut fx, &exit_plan(), Kind::Exit);
    check("취소 확인 실패 + 부분", &st, &fx, 2.0);
    assert!(st.phase == Phase::Error && st.filled == 1.0, "마지막에 본 체결량(하한)을 돌려준다: {}", st.filled);
}

// ── 진입 대기 취소(cancel 시험) · 감시 진입 · 시장가 계획 ───────────────────────

#[test]
fn e18_entry_cancel_paths() {
    let mut fx = Fake { script: vec![(5.0, Cmd::Action(Action::Cancel))], ..Default::default() };
    let st = go(&mut fx, &plan("BUY", "LONG", 1.0), Kind::Entry);
    check("취소", &st, &fx, 1.0);
    assert!(st.phase == Phase::Cancelled && fx.placed.len() == 1 && fx.deletes.len() == 1 && st.cancel_reason.as_deref() == Some("사용자 취소"));
    assert!(fx.t < 5.0 + POLL_SEC + 1.0, "취소는 다음 조회(≤3초)에 먹어야 한다 -- 마감까지 기다렸다: t={}", fx.t);
    let mut fx = Fake { fill_after: 1, partial: Some(0.3), script: vec![(5.0, Cmd::Action(Action::Cancel))], ..Default::default() };
    let st = go(&mut fx, &plan("BUY", "LONG", 1.0), Kind::Entry);
    check("부분 뒤 취소", &st, &fx, 1.0);
    assert!(st.phase == Phase::Cancelled && (st.filled - 0.3).abs() < 1e-12);
    let mut fx = Fake { fail: vec!["cancel"], script: vec![(5.0, Cmd::Action(Action::Cancel))], ..Default::default() };
    let st = go(&mut fx, &plan("BUY", "LONG", 1.0), Kind::Entry);
    check("취소 확인 실패", &st, &fx, 1.0);
    assert!(st.phase == Phase::Error && st.error.unwrap().contains("거래소에서 직접 확인") && fx.placed.len() == 1);
    // 음성 대조: 취소 없이 같은 상황이면 마감에 잔량 → (파이썬 시장가) 시장가 결정 대기
    let mut fx = Fake { fill_after: 1, partial: Some(0.3), ..Default::default() };
    let st = go(&mut fx, &plan("BUY", "LONG", 1.0), Kind::Entry);
    check("취소 없음", &st, &fx, 1.0);
    assert!(st.phase == Phase::NeedsMarketDecision && (st.market_qty - 0.7).abs() < 1e-12);
}

#[test]
fn maker_only_and_market_entry_plans() {
    let mut fx = Fake::default();
    let st = go(&mut fx, &ExecPlan { maker_only: true, ..entry_plan() }, Kind::Entry);
    check("감시 진입 따라가기 마감", &st, &fx, 2.0);
    assert!(st.phase == Phase::Cancelled && st.cancel_reason.unwrap().contains("시장가 없음"));
    let mut fx = Fake::default();
    let st = go(&mut fx, &ExecPlan { market: true, ..entry_plan() }, Kind::Entry);
    check("시장가 진입 계획", &st, &fx, 2.0);
    assert!(fx.placed.is_empty() && st.market_branch == Some("M2"));
}

// ── 기다리기(wait 시험) ──────────────────────────────────────────────────────

#[test]
fn wait_paths() {
    let mut fx = Fake { drift: true, ..Default::default() };
    let st = go(&mut fx, &entry_plan(), Kind::Wait(900.0, false));
    check("기다리기 마감", &st, &fx, 2.0);
    assert!(st.phase == Phase::NeedsMarketDecision && st.market_branch == Some("M3") && fx.placed.len() == 1 && fx.book_times.is_empty(), "앞질러도 다시 걸지 않는다");
    assert!(fx.placed[0].client_order_id.starts_with(WAIT_TAG));
    let mut fx = Fake { fill_after: 1, post_only_rejects: 2, ..Default::default() };
    let st = go(&mut fx, &entry_plan(), Kind::Wait(900.0, false));
    check("기다리기 -5022", &st, &fx, 2.0);
    assert_eq!((st.phase, st.repegs), (Phase::FilledMaker, 2));
    let mut fx = Fake { kill_at: Some(2), ..Default::default() };
    let st = go(&mut fx, &entry_plan(), Kind::Wait(900.0, false));
    check("거래소 취소", &st, &fx, 2.0);
    assert_eq!((st.phase, st.cancel_reason.as_deref()), (Phase::Cancelled, Some("거래소에서 CANCELED")));
    let mut fx = Fake { fill_after: 1, partial: Some(0.25), ..Default::default() };
    let st = go(&mut fx, &entry_plan(), Kind::Wait(60.0, true));
    check("첫 체결 on_fill", &st, &fx, 2.0);
    assert_eq!((fx.on_fills, st.phase), (1, Phase::NeedsMarketDecision));
    let mut fx = Fake { script: vec![(10.0, Cmd::Action(Action::Market))], ..Default::default() };
    let st = go(&mut fx, &entry_plan(), Kind::Wait(900.0, false));
    check("지금 시장가", &st, &fx, 2.0);
    assert_eq!((st.taker_reason, fx.deletes.len()), (Some("지금 시장가(사용자)"), 1));
}

// ── 지정가 대기(limit 시험) ──────────────────────────────────────────────────

#[test]
fn limit_paths() {
    let lp = || entry_plan();
    let mut fx = Fake { fill_after: 3, drift: true, ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(60.0, false, false));
    check("고정가", &st, &fx, 2.0);
    assert!(st.phase == Phase::FilledMaker && fx.placed.len() == 1 && fx.placed[0].price == 2500.0 && st.repegs == 0);
    let mut fx = Fake { script: vec![(5.0, Cmd::Action(Action::Cancel))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(60.0, false, false));
    check("취소", &st, &fx, 2.0);
    assert_eq!((st.phase, st.cancel_reason.as_deref(), fx.deletes.len()), (Phase::Cancelled, Some("사용자 취소"), 1));
    let mut fx = Fake { script: vec![(5.0, Cmd::Amend(2490.0, Some(2.008))), (12.0, Cmd::FillAfter(1))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(60.0, false, false));
    check("수정", &st, &fx, 2.008);
    assert_eq!(fx.placed.iter().map(|o| (o.price, o.quantity)).collect::<Vec<_>>(), vec![(2500.0, 2.0), (2490.0, 2.008)]);
    assert_eq!((st.amends, st.phase), (1, Phase::FilledMaker));
    // 일부 체결 뒤 수정 = 버림(검증 D4): 남은 수량만 옛 가격에
    let mut fx = Fake { fill_after: 1, partial: Some(0.4), script: vec![(5.0, Cmd::Amend(2490.0, Some(2.4))), (12.0, Cmd::Action(Action::Cancel))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(60.0, false, false));
    check("부분 뒤 수정", &st, &fx, 2.0);
    assert_eq!(fx.placed.iter().map(|o| (o.price, o.quantity)).collect::<Vec<_>>(), vec![(2500.0, 2.0), (2500.0, 1.2)]);
    let mut fx = Fake { post_only_rejects: 1, ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(60.0, false, false));
    check("-5022", &st, &fx, 2.0);
    assert!(st.phase == Phase::Rejected && st.error.unwrap().contains("메이커") && fx.placed.is_empty());
    let mut fx = Fake::default();
    let st = go(&mut fx, &lp(), Kind::Limit(10.0, false, false));
    check("만료", &st, &fx, 2.0);
    assert_eq!((st.phase, st.cancel_reason.as_deref()), (Phase::Cancelled, Some("대기 시간 만료")));
    let mut fx = Fake { kill_at: Some(2), ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(60.0, false, false));
    check("거래소 취소", &st, &fx, 2.0);
    assert_eq!(st.cancel_reason.as_deref(), Some("거래소에서 취소됨"));
    let mut fx = Fake { fill_after: 1, partial: Some(0.2), script: vec![(8.0, Cmd::Action(Action::Cancel))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(60.0, true, false));
    check("on_fill 한 번", &st, &fx, 2.0);
    assert_eq!((fx.on_fills, st.phase), (1, Phase::Cancelled));
    let mut fx = Fake { fill_after: 1, partial: Some(0.2), script: vec![(8.0, Cmd::Action(Action::Market))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(60.0, false, false));
    check("지금 시장가", &st, &fx, 2.0);
    assert!(st.market_branch == Some("M4") && (st.market_qty - 1.6).abs() < 1e-12 && fx.deletes.len() == 1, "걸린 지정가를 취소 확인한 뒤");
    let mut fx = Fake { kill_at: Some(2), script: vec![(4.0, Cmd::Action(Action::Market))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(60.0, false, false));
    check("사람이 취소한 뒤 시장가", &st, &fx, 2.0);
    assert_eq!(st.phase, Phase::Cancelled);
    // 자동 재조정: 한 번만(두 번째 확인은 1bp 미만) · 체결 시작 뒤엔 확인조차 안 한다
    let mut fx = Fake { reprice_vals: vec![2499.4], script: vec![(50.0, Cmd::FillAfter(1))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(120.0, false, true));
    check("재조정", &st, &fx, 2.0);
    assert_eq!(fx.placed.iter().map(|o| o.price).collect::<Vec<_>>(), vec![2500.0, 2499.4]);
    assert_eq!((st.reprices, st.amends, st.phase), (1, 0, Phase::FilledMaker));
    let mut fx = Fake { fill_after: 1, partial: Some(0.4), reprice_vals: vec![2400.0], script: vec![(60.0, Cmd::Action(Action::Cancel))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(120.0, false, true));
    check("부분 뒤 재조정 없음", &st, &fx, 2.0);
    assert_eq!((fx.reprice_calls, fx.placed.len()), (0, 1));
    // 수정 직후 -5022 → 죽지 않고 새 목표가로 다시(검증 D6)
    let mut fx = Fake { reprice_vals: vec![2491.0], reject_at_placed: Some(1), script: vec![(5.0, Cmd::Amend(2490.0, None)), (30.0, Cmd::FillAfter(1))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(120.0, false, true));
    check("수정 직후 -5022", &st, &fx, 2.0);
    assert_eq!((st.phase, fx.placed.iter().map(|o| o.price).collect::<Vec<_>>()), (Phase::FilledMaker, vec![2500.0, 2491.0]));
    // PEG 동작: 취소 확인 → 따라가기 → 잔량 버림(시장가 없음)
    let mut fx = Fake { script: vec![(5.0, Cmd::Action(Action::Peg)), (5.0, Cmd::Drift(true))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(600.0, false, false));
    check("PEG", &st, &fx, 2.0);
    assert!(st.phase == Phase::Cancelled && st.pegged && st.repegs >= 30 && st.market_branch.is_none() && st.cancel_reason.unwrap().contains("시장가 없음"));
    let mut fx = Fake { script: vec![(5.0, Cmd::Action(Action::Peg)), (6.0, Cmd::FillAfter(1))], ..Default::default() };
    let st = go(&mut fx, &lp(), Kind::Limit(600.0, false, false));
    check("PEG 체결", &st, &fx, 2.0);
    assert_eq!(st.phase, Phase::FilledMaker);
}

// ── 무작위 실패 주입(고정 시드) ──────────────────────────────────────────────
// D2 해결(2026-11-20) 뒤 «불명» 접수를 네 경로 모두에 넣는다(해결 전엔 fill_maker·기다리기가 유령을 남겨 지정가 대기에만 넣었다).

#[test]
fn fuzz_random_faults_keep_invariants() {
    let mut seed: u64 = 20261120;
    let mut rnd = |n: u64| { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) % n };
    let (mut runs, mut repegs) = (0, 0u64);
    for _ in 0..400 {
        let fail: Vec<&'static str> = ["cancel", "get_after_cancel", "get", "place", "book"].into_iter().filter(|_| rnd(5) == 0).collect();
        let mut fx = Fake {
            fill_after: [0, 1, 2, 5, 30][rnd(5) as usize] as u32, partial: [None, Some(0.3), Some(0.5)][rnd(3) as usize],
            fail, ok_first: rnd(4) as u32, post_only_rejects: rnd(4) as u32, drift: rnd(2) == 0, fill_on_cancel: rnd(6) == 0,
            kill_at: (rnd(8) == 0).then_some(3), wake_every: (rnd(4) == 0).then_some(0.7), ..Default::default()
        };
        let k = rnd(4);
        fx.place_err = (rnd(3) == 0).then(|| (["TimeoutError: x", "503 [-1007] Timeout"][rnd(2) as usize].into(), rnd(2) == 0));
        if rnd(3) == 0 {
            fx.script.push((rnd(60) as f64, Cmd::Action([Action::Cancel, Action::Market, Action::Peg][rnd(3) as usize])));
        }
        let p = if k == 1 { exit_plan() } else { entry_plan() };
        let st = go(&mut fx, &p, match k { 0 => Kind::Entry, 1 => Kind::Exit, 2 => Kind::Limit(300.0, true, false), _ => Kind::Wait(300.0, true) });
        check(&format!("무작위 #{runs}"), &st, &fx, 2.0);
        repegs += st.repegs as u64;
        runs += 1;
    }
    assert!(repegs > 100, "무작위 사례에서 재호가가 실제로 돌았다: {repegs}");
    println!("무작위 {runs}건 통과 · 재호가 합 {repegs}");
}
