//! 심볼별 최신 호가·마크·최근 체결 링.
//! 락: 심볼마다 std Mutex 하나. 잡는 동안 await 하지 않고 대입·링 push·broadcast send(동기) 만 한다 -- 수 µs.
//! 번호 매기기와 broadcast 를 같은 락 안에서 해서, 스냅숏(같은 락)과 밀기 사이에 빠짐·겹침이 없다.
//! 1초 수급·풋프린트는 사건마다 밀지 않고 FLUSH_MS 마다 모아 민다(flush). 스냅숏은 모으던 것까지 이미 반영한 상태라,
//! 스냅숏 뒤 첫 묶음이 같은 값을 한 번 더 덮어써도 결과가 같다.
use crate::proto::{Event, Situation, Stamped, StreamStat, Streams, Supply1s, Venue};
use crate::restore;
use crate::situation;
use crate::supply::Supply;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use tokio::sync::{broadcast, mpsc};

struct Sym {
    seq: u64,
    book: Option<Stamped>,
    mark: Option<Stamped>,
    trades: VecDeque<Stamped>,
    supply: Supply,
    /// 마지막 flush 뒤 모은 1초 수급(Supply1s::merge)과 그 묶음 첫 사건의 시각(ex_ms, rx_us).
    pending: Option<Supply1s>,
    first: Option<(i64, i64)>,
    /// 주문 경로가 쓰는 받아 둔 시세(quote.rs).
    quote: crate::quote::Quote,
    /// 30분 상황 근거(situation.rs): 체결로 센 5분봉 [고가, 저가, 종가] + 종가 체결 id · 마크 링 · 마지막 결과.
    ohlc: BTreeMap<i64, ([f64; 3], u64)>,
    marks: situation::MarkRing,
    sit: Option<Situation>,
    /// 마지막 미시 값(okx_oi5m 은 비움 -- 스냅숏이 전부를 다시 센다).
    micro: Option<crate::proto::Micro>,
    /// 마크(@markPrice@1s) 수신 상태(원본 mp_state).
    mp: StreamStat,
    /// 관계 읽기 재료(풋프린트 몫 · situation::flow_nets_tick) · 마지막으로 Micro 에 실어 보낸 것(None = 아직 → 통째로).
    nets: situation::FlowNets,
    nets_sent: Option<situation::FlowNets>,
}

/// 1초 수급·풋프린트 밀기 묶음 간격(ms). 화면은 한 프레임(60Hz ≈ 16.7ms)에 한 번 칠한다 -- 더 잘게 보내도 안 보이고
/// 메시지만 는다(ETH 평소 ~266건/초 → 사건마다면 초당 266건). 더 굵게 모으면 모은 만큼 늦다(최악 지연 = 이 간격).
/// 실측(개발기 릴리스): 묶음 하나 만들기 수 µs(tests/footprint_parity.rs bench) -- 간격을 줄이는 데 걸림돌은 메시지 수뿐이다.
pub const FLUSH_MS: u64 = 16;

pub struct State {
    syms: HashMap<String, Mutex<Sym>>,
    ring_cap: usize,
    pub tx: broadcast::Sender<Stamped>,
    /// 재시작 복원(restore.rs). None = 끔(시험·부분 가동).
    restore: Option<restore::Cfg>,
    /// 첫 체결이 와서 메울 공백이 생긴 심볼(과 일감 종류)을 복원 스레드에 알린다(lib.rs).
    wake: Option<mpsc::UnboundedSender<(String, restore::Wake)>>,
    /// «모든 종목» 메시지(`s = "*"` · Prices)의 번호. 보내기도 이 락 안에서 -- 번호와 보내는 순서가 같다.
    prices_seq: Mutex<u64>,
    /// 강제청산(전 종목 한 연결) 수신 상태(원본 fo_state -- 코인 공통 하나).
    fo: Mutex<StreamStat>,
}

impl State {
    pub fn new(symbols: &[String], ring_cap: usize, queue_cap: usize) -> Arc<Self> {
        Arc::new(Self::build(symbols, ring_cap, queue_cap))
    }

    fn build(symbols: &[String], ring_cap: usize, queue_cap: usize) -> State {
        let syms = symbols
            .iter()
            .map(|s| {
                let sym = Sym {
                    seq: 0,
                    book: None,
                    mark: None,
                    trades: VecDeque::new(),
                    supply: Supply::new(s),
                    pending: None,
                    first: None,
                    quote: Default::default(),
                    ohlc: BTreeMap::new(),
                    marks: situation::MarkRing::new(crate::now_us() as f64 / 1e6),
                    sit: None,
                    micro: None,
                    mp: StreamStat::default(),
                    nets: Default::default(),
                    nets_sent: None,
                };
                (s.clone(), Mutex::new(sym))
            })
            .collect();
        State { syms, ring_cap, tx: broadcast::channel(queue_cap).0, restore: None, wake: None, prices_seq: Mutex::new(0), fo: Mutex::new(StreamStat::default()) }
    }

    /// 복원을 켠 상태: 심볼마다 스냅숏 파일을 읽어 되살린다(수신 전 · `now_us` = 지금 벽시계 -- 143봉보다 오래된 봉은 버린다).
    /// 받는 쪽(둘째 값)은 «메울 공백이 생긴 심볼»을 받아 `run_restore` 를 부른다(lib.rs). 시험은 버리고 `run_restore` 를 직접 부른다.
    pub fn with_restore(symbols: &[String], ring_cap: usize, queue_cap: usize, cfg: restore::Cfg, now_us: i64) -> (Arc<Self>, mpsc::UnboundedReceiver<(String, restore::Wake)>) {
        let mut st = Self::build(symbols, ring_cap, queue_cap);
        let (wake, rx) = mpsc::unbounded_channel();
        for (s, cell) in &mut st.syms {
            let sup = &mut cell.get_mut().unwrap().supply;
            if sup.fp.spec.is_none() {
                continue; // 파이썬에 흐름 엔진이 없는 종목 -- 되살릴 것이 없다
            }
            sup.rs.on = true;
            sup.rs.await_first = true;
            let p = restore::snap_path(&cfg.snap_dir, s);
            match restore::load(&p, s, now_us.div_euclid(1000)) {
                Ok(snap) => {
                    println!("rtcore 복원 {s}: 스냅숏 {} -- 봉 {} · 1초 칸 {} · last_ms {} · 구멍 {:?}", p.display(), snap.bars.len(), snap.sec.len(), snap.last_ms, snap.holes);
                    sup.load_snap(snap);
                }
                Err(e) => println!("rtcore 복원 {s}: {} {e} -- 첫 체결 때 테이프로 창(144봉)을 메운다", p.display()),
            }
            // 나머지 원천(restore.rs 머리 주석): OKX 는 첫 체결·첫 OI 때, 청산 둘은 지금(수신 전 -- 파이썬 liq_events_load 도 수신 전 한 번).
            let okx = crate::supply::okx_ct_val(&sup.okx_inst).is_some();
            sup.rs.okx_await = okx;
            sup.rs.oi_await = okx;
            let bn = restore::read_bn_liq(&cfg.bn_ctx, s).unwrap_or_else(|e| {
                println!("rtcore 복원 {s}: 🔴강제청산 못 읽음({e}) -- 재시작 뒤 660초 청산 레인이 빈다");
                vec![]
            });
            let by = if crate::supply::BYBIT_SYMBOLS.contains(&s.as_str()) {
                restore::read_bybit_liq(&cfg.bybit_liq, s, now_us.div_euclid(1000)).unwrap_or_else(|e| {
                    println!("rtcore 복원 {s}: 🔴Bybit 청산 못 읽음({e}) -- 재시작 뒤 660초 Bybit 레인이 빈다");
                    vec![]
                })
            } else {
                vec![]
            };
            println!("rtcore 복원 {s}: 강제청산 {}건 · Bybit 청산 {}건(DB)", bn.len(), by.len());
            sup.load_liqs(bn, by, now_us as f64 / 1000.0);
            if crate::supply::BYBIT_SYMBOLS.contains(&s.as_str()) {
                match restore::read_bybit_liq_5m(&cfg.bybit_liq, s, now_us.div_euclid(1000)) {
                    Ok(got) => sup.load_bybit5m(got, now_us),
                    Err(e) => println!("rtcore 복원 {s}: 🔴Bybit 5분봉 청산 못 읽음({e}) -- 재시작 앞 봉은 Bybit «모름»"),
                }
            }
        }
        st.restore = Some(cfg);
        st.wake = Some(wake);
        (Arc::new(st), rx)
    }

    /// 메울 공백이 있으면 테이프를 읽어 합친다. 테이프 읽기(최대 tries × wait)는 락 밖, 합치기만 락 안 -- 수신을 막지 않는다.
    /// 실행기 밖 스레드에서 부른다(lib.rs spawn_blocking). 돌려주는 것 = 로그 한 줄(일감이 없으면 None).
    pub fn run_restore(&self, s: &str) -> Option<String> {
        let cfg = self.restore.as_ref()?;
        let cell = self.syms.get(s)?;
        let (job, bucket) = {
            let g = cell.lock().unwrap();
            (g.supply.rs.job?, g.supply.fp.spec?.0)
        };
        let t = match restore::tape_bucket(s) {
            Some(tb) => restore::read(&cfg.tape, s, job, bucket, tb, cfg.tries, cfg.wait),
            None => Err("테이프 칸 폭 표에 없는 종목".into()),
        };
        let mut g = cell.lock().unwrap();
        let now = crate::now_us();
        let (line, up) = g.supply.merge(t, now);
        Self::queue(&mut g, up, now);
        println!("rtcore 복원 {s}: {line}");
        Some(line)
    }

    fn wake(&self, s: &str, w: restore::Wake) {
        if let Some(tx) = &self.wake {
            let _ = tx.send((s.to_string(), w));
        }
    }

    /// OKX 1초 칸·청산을 메운다(run_restore 와 같은 방식 -- 읽기는 락 밖). 일감이 없으면 None.
    pub fn run_restore_okx(&self, s: &str) -> Option<String> {
        let cfg = self.restore.as_ref()?;
        let cell = self.syms.get(s)?;
        // 1초 칸 끝 초 = 첫 OKX 체결의 초(파이썬 t0 = min(okx_sec) 와는 늦은 체결이 있을 때만 다르다 -- restore.rs 머리 주석) · 청산도 그 시각
        let (t0_ms, inst) = {
            let g = cell.lock().unwrap();
            (g.supply.rs.okx_job?, g.supply.okx_inst.clone())
        };
        let t0 = t0_ms.div_euclid(1000);
        let secs = match restore::okx_tape_bucket(&inst) {
            Some(tb) => restore::read_okx_secs(&cfg.okx_tape, &inst, t0, tb, cfg.tries, cfg.wait),
            None => Err("OKX 테이프 칸 폭 표에 없는 종목".into()),
        };
        let now = crate::now_us();
        let mut g = cell.lock().unwrap();
        let (line, up) = g.supply.merge_okx(secs, now);
        Self::queue(&mut g, up, now);
        println!("rtcore 복원 {s}: {line}");
        Some(line)
    }

    /// OKX 풋프린트 봉 복원 회차 하나(restore.rs «OKX 풋프린트 봉 복원»). 읽기는 락 밖. `now_us` = 원본 time.time()(최근 15분 판정).
    /// None = 할 일 없음(복원 끔 · 첫 OKX 체결 전 · 이미 끝남). 돌려주는 둘째 값 = 끝났나(더 부르지 않는다).
    pub fn run_restore_okx_fp(&self, s: &str, now_us: i64) -> Option<(String, bool)> {
        let cfg = self.restore.as_ref()?;
        let cell = self.syms.get(s)?;
        let (t0, inst, bucket) = {
            let g = cell.lock().unwrap();
            if g.supply.okx_t0_ms == 0 || g.supply.okx_fr.done || g.supply.okx_fr.attempts >= restore::OKX_FP_ATTEMPTS {
                return None;
            }
            (g.supply.okx_t0_ms, g.supply.okx_inst.clone(), g.supply.okx_fp.spec?.0)
        };
        let lo = crate::footprint::bar_of(t0) - (crate::footprint::WINDOW_BARS as i64 - 1) * 300;
        let got = match restore::okx_tape_bucket(&inst) {
            Some(tb) => restore::read_okx_fp(&cfg.okx_tape, &cfg.okx_ctx, &inst, lo, t0, bucket, tb, now_us.div_euclid(1_000_000)),
            None => Err("OKX 테이프 칸 폭 표에 없는 종목".into()),
        };
        let mut g = cell.lock().unwrap();
        let line = g.supply.okx_fp_attempt(got);
        let done = g.supply.okx_fr.done || g.supply.okx_fr.attempts >= restore::OKX_FP_ATTEMPTS;
        println!("rtcore 복원 {s}: {line}");
        Some((line, done))
    }

    /// 시험용: OKX 풋프린트 봉(반올림 전) · first_bar · OKX OI 5분봉.
    pub fn okx_fp(&self, s: &str) -> Option<(Vec<(i64, Vec<(i64, [f64; 6])>)>, i64, Vec<(i64, [f64; 2])>)> {
        let g = self.syms.get(s)?.lock().unwrap();
        Some((g.supply.okx_fp.raw(), g.supply.okx_first_bar, g.supply.okx_oi5m.iter().map(|(&b, &v)| (b, v)).collect()))
    }

    /// OKX 미결제약정을 메운다.
    pub fn run_restore_okx_oi(&self, s: &str) -> Option<String> {
        let cfg = self.restore.as_ref()?;
        let cell = self.syms.get(s)?;
        let (t0, inst) = {
            let g = cell.lock().unwrap();
            (g.supply.rs.oi_job?, g.supply.okx_inst.clone())
        };
        let got = restore::read_okx_oi(&cfg.okx_ctx, &inst, t0);
        let now = crate::now_us();
        let mut g = cell.lock().unwrap();
        let (line, up) = g.supply.merge_okx_oi(got, now);
        Self::queue(&mut g, up, now);
        println!("rtcore 복원 {s}: {line}");
        Some(line)
    }

    /// 합친 몫을 다음 묶음 밀기에 싣는다.
    fn queue(g: &mut Sym, up: Option<Supply1s>, now: i64) {
        if let Some(up) = up {
            g.first.get_or_insert((0, now));
            match &mut g.pending {
                Some(p) => p.merge(up),
                None => g.pending = Some(up),
            }
        }
    }

    /// 시험용: 청산 링 -- (거래소 시각 ms, 롱 청산, 수량, USD).
    pub fn liq_events(&self, s: &str, venue: Venue) -> Option<Vec<(i64, bool, f64, f64)>> {
        Some(self.syms.get(s)?.lock().unwrap().supply.liq_events(venue))
    }

    /// 코어 스냅숏을 쓴다(lib.rs 가 SNAP_EVERY 마다 · 종료코드 70 직전 main.rs). 락 안에서는 값만 뜨고 직렬화·쓰기는 락 밖.
    /// 돌려주는 것 = 쓴 파일 수.
    pub fn write_snapshots(&self) -> usize {
        let Some(cfg) = &self.restore else { return 0 };
        let mut n = 0;
        for (s, cell) in &self.syms {
            let snap = cell.lock().unwrap().supply.to_snap(s);
            if let Some(snap) = snap {
                match restore::write(&restore::snap_path(&cfg.snap_dir, s), &snap) {
                    Ok(()) => n += 1,
                    Err(e) => println!("rtcore 스냅숏 쓰기 실패 {s}: {e}"),
                }
            }
        }
        n
    }

    /// 모르는 심볼은 버린다.
    pub fn apply(&self, mut m: Stamped) {
        let Some(cell) = self.syms.get(&m.s) else { return };
        let mut g = cell.lock().unwrap();
        g.seq += 1;
        m.seq = g.seq;
        let mut supply = None;
        match m.ev {
            Event::Trade { id, px, qty, sell } => {
                // 이 연결의 첫 체결 -- 그 앞 공백을 테이프로 메운다(쌓기 전에: last_ms 가 앞 연결의 값이어야 한다).
                if g.supply.rs.await_first && g.supply.begin_restore(m.ex_ms) {
                    self.wake(&m.s, restore::Wake::Bn);
                }
                if g.trades.len() == self.ring_cap {
                    g.trades.pop_front();
                }
                g.trades.push_back(m.clone());
                g.quote.last = Some((px, m.ex_ms, m.rx_us));
                g.quote.price_dirty = true;
                ohlc_add(&mut g.ohlc, px, id, m.ex_ms);
                // 24시간 봉 창은 «수신 시각» 기준(파이썬 time.time()) -- rx_us 가 그 벽시계다.
                supply = g.supply.trade(px, qty, m.ex_ms, sell, id as i64, m.rx_us);
            }
            Event::Liq { venue, long, qty, px } => {
                if venue == Venue::Bn {
                    // server.py 1853~1854: 대시보드 코인의 청산마다 events += 1 · last_event_ms = T
                    let mut fo = self.fo.lock().unwrap();
                    fo.events += 1;
                    fo.last_ms = Some(m.ex_ms);
                }
                supply = g.supply.liq(venue, long, qty, px, m.ex_ms, m.rx_us)
            }
            Event::Flow { venue, px, qty, sell } => {
                // 시작 뒤 첫 OKX 체결 -- 그 앞 660초(1초 칸·청산)를 테이프·맥락 DB 에서(쌓기 전에: t0 = 이 체결의 초)
                if venue == Venue::Okx && g.supply.rs.okx_await && g.supply.begin_okx(m.ex_ms) {
                    self.wake(&m.s, restore::Wake::Okx);
                }
                supply = g.supply.flow(venue, px, qty, sell, m.ex_ms, m.rx_us)
            }
            Event::Oi { oi, .. } => {
                if g.supply.rs.oi_await && g.supply.begin_okx_oi(m.ex_ms) {
                    self.wake(&m.s, restore::Wake::OkxOi);
                }
                supply = g.supply.oi(oi, m.ex_ms, m.rx_us)
            }
            Event::Book { bid, ask, ref bid_s, ref ask_s, .. } => {
                g.quote.book = Some(crate::quote::Book { bid_s: bid_s.clone(), ask_s: ask_s.clone(), bid, ask, ex_ms: m.ex_ms, rx_us: m.rx_us });
                g.book = Some(m.clone())
            }
            Event::Kline { open_ms, close, ref mut vol_bpm, ref mut bar5m, .. } => {
                (*vol_bpm, *bar5m) = g.quote.kline(open_ms, close, m.rx_us); // 실어 보내는 메시지에 이 봉까지의 변동성·5분봉 종가를 채운다
            }
            Event::Mark { mark, index, funding, next_funding_ms } => {
                // server.py 1893~1894: events += 1 · last_ms = E · next_funding_ms = int(T or 0) or None
                g.mp.events += 1;
                g.mp.last_ms = Some(m.ex_ms);
                g.mp.next_funding_ms = (next_funding_ms != 0).then_some(next_funding_ms);
                g.marks.push(m.ex_ms, (mark, index, funding), m.rx_us as f64 / 1e6);
                g.mark = Some(m.clone())
            }
            Event::Snapshot { .. } | Event::Supply1s(_) | Event::Footprint(_) | Event::Prices { .. } | Event::Liq5m(_) | Event::Situation(_) | Event::Micro(_) => return,
        }
        if let Some(up) = supply {
            g.first.get_or_insert((m.ex_ms, m.rx_us));
            match &mut g.pending {
                Some(p) => p.merge(up),
                None => g.pending = Some(up),
            }
        }
        let _ = self.tx.send(m); // 구독자 0명이면 Err -- 그대로 둔다
    }

    /// 모은 1초 수급과 풋프린트 바뀐 몫을 민다(lib.rs 가 FLUSH_MS 마다 부른다 · 시험은 직접 부른다).
    /// 묶음의 rx_us = 그 묶음 **첫** 사건의 수신 시각 -- 허브의 수신→송신 지연이 묶음 대기까지 센다.
    pub fn flush(&self) {
        for (s, cell) in &self.syms {
            let mut g = cell.lock().unwrap();
            let fp = g.supply.footprint_delta();
            let sup = g.pending.take();
            let l5 = g.supply.liq5m_delta();
            if fp.is_none() && sup.is_none() && l5.is_none() {
                continue;
            }
            let (ex_ms, rx_us) = g.first.take().unwrap_or_else(|| (0, crate::now_us()));
            for ev in [sup.map(Event::Supply1s), fp.map(Event::Footprint), l5.map(Event::Liq5m)].into_iter().flatten() {
                g.seq += 1;
                let _ = self.tx.send(Stamped { s: s.clone(), seq: g.seq, ex_ms, rx_us, tx_us: 0, ev });
            }
        }
    }

    /// 체결 스트림 재연결(feed): 열린 주문 묶음을 버린다(coin_flow.py 613 fp_orders.reset()).
    pub fn reset_orders(&self, s: &str) {
        if let Some(cell) = self.syms.get(s) {
            cell.lock().unwrap().supply.reset_orders();
        }
    }

    /// 강제청산·마크 연결 상태(원본 fo_state·mp_state 1843·1865·1869 · 1883). `sym` = None 이면 강제청산(전 종목 연결).
    /// `up` = Some(true) 붙음(connected · since = 지금) · Some(false) 끊김 · `err` = 오류 한 번(errors += 1 · last_error).
    pub fn stream_status(&self, sym: Option<&str>, up: Option<bool>, err: Option<&str>, now_s: f64) {
        let f = |st: &mut StreamStat| {
            match up {
                Some(true) => (st.connected, st.since) = (true, Some(now_s)),
                Some(false) => st.connected = false,
                None => {}
            }
            if let Some(e) = err {
                st.errors += 1;
                st.last_error = Some(e.chars().take(120).collect());
            }
        };
        match sym {
            None => f(&mut self.fo.lock().unwrap()),
            Some(s) => {
                if let Some(cell) = self.syms.get(s) {
                    f(&mut cell.lock().unwrap().mp)
                }
            }
        }
    }

    /// 현물·OKX 수신 연결 상태(화면 meta 의 connected·errors).
    pub fn feed_status(&self, s: &str, venue: Venue, up: bool) {
        if let Some(cell) = self.syms.get(s) {
            cell.lock().unwrap().supply.feed_status(venue, up);
        }
    }

    /// 시험용: 지금 1초 칸(반올림 전). 바이낸스 선물.
    pub fn supply_cells(&self, s: &str) -> Option<Vec<(i64, [f64; 7])>> {
        self.venue_cells(s, Venue::Bn)
    }

    /// 시험용: 원천별 1초 칸(반올림 전).
    pub fn venue_cells(&self, s: &str, venue: Venue) -> Option<Vec<(i64, [f64; 7])>> {
        Some(self.syms.get(s)?.lock().unwrap().supply.cells_of(venue).iter().map(|(&k, v)| (k, *v)).collect())
    }

    /// 시험용: 풋프린트 봉 → 칸(반올림 전).
    pub fn footprint_cells(&self, s: &str) -> Option<Vec<(i64, Vec<(i64, [f64; 6])>)>> {
        Some(self.syms.get(s)?.lock().unwrap().supply.footprint_raw())
    }

    /// 시험용: OKX OI 칸.
    pub fn okx_oi_cells(&self, s: &str) -> Option<Vec<(i64, f64)>> {
        Some(self.syms.get(s)?.lock().unwrap().supply.okx_oi_cells().iter().map(|(&k, &v)| (k, v)).collect())
    }

    pub fn snapshot(&self, s: &str, rx_us: i64) -> Option<Stamped> {
        let g = self.syms.get(s)?.lock().unwrap();
        let ex_ms = g.trades.back().map_or(0, |t| t.ex_ms);
        Some(Stamped {
            s: s.to_string(),
            seq: g.seq,
            ex_ms,
            rx_us,
            tx_us: 0,
            ev: Event::Snapshot {
                book: g.book.clone().map(Box::new),
                mark: g.mark.clone().map(Box::new),
                trades: g.trades.iter().cloned().collect(),
                supply: g.supply.snapshot(rx_us),
                footprint: g.supply.footprint(),
                quote: Some(g.quote.msg()),
                liq5m: g.supply.liq5m_all(),
                situation: g.sit.clone(),
                micro: g.micro.clone().map(|mut m| {
                    m.okx_oi5m = situation::okx_oi_rows(&g.supply.okx_oi5m, i64::MIN);
                    m.okx_bars = g.supply.okx_sums(None);
                    m.okx_full = m.okx_from.is_some();
                    m.flow = Some(situation::flow_delta(&g.nets, None));
                    m
                }),
            },
        })
    }

    /// 30분 상황 근거를 센다(lib.rs 가 1초마다 · 원본 SITUATION_EVERY_TICKS = 1). 흐름 엔진 종목만. 바뀌었으면 민다.
    /// BTC 같은 창은 btcusdt 를 받을 때만(그 종목의 체결 캔들을 먼저 떠 둔다 -- 락은 하나씩만 잡는다).
    pub fn situation_tick(&self, now_us: i64) {
        let now_s = now_us as f64 / 1e6;
        let btc: Option<BTreeMap<i64, [f64; 3]>> = self.syms.get("btcusdt").map(|c| c.lock().unwrap().ohlc.iter().map(|(&b, v)| (b, v.0)).collect());
        for (s, cell) in &self.syms {
            let mut g = cell.lock().unwrap();
            let Some((bucket, _)) = g.supply.fp.spec else { continue };
            // 미시 값(원본은 같은 1초 틱에서 payload 를 먼저 만들고 상황을 센다)
            let book = match &g.book {
                Some(Stamped { ev: Event::Book { bid, bid_qty, ask, ask_qty, .. }, .. }) => Some((*bid, *bid_qty, *ask, *ask_qty)),
                _ => None,
            };
            // 관계 읽기 재료(원본은 compute_situation 마다 _flow_read_ctx 가 센다 -- 미시 값과 상관없이 매 회차)
            let okx_first = g.supply.okx_first_bar;
            let mut nets = std::mem::take(&mut g.nets);
            situation::flow_nets_tick(&mut nets, now_s, &g.supply.fp.bars, okx_first, &g.supply.okx_fp.bars, |b| b.in_order());
            g.nets = nets;
            // 원본은 호가(REST bookTicker)를 받기 전에는 그 틱이 예외로 끝나 payload 가 {"available": false} 그대로다 -- 미시 값을 안 낸다.
            //   코어도 호가를 한 번 받은 뒤부터 민다(OKX OI 5분봉 바뀐 봉 표시는 그때까지 쌓아 둔다).
            if book.is_some() {
                let from = g.supply.okx_oi5m_dirty.first().copied();
                g.supply.okx_oi5m_dirty.clear();
                let liq = g.supply.liq.iter().map(|e| (e.ts_ms, e.long, e.usd));
                let mut mc = situation::micro(now_s, book, g.supply.cells_of(Venue::Bn), liq, &g.supply.okx_oi5m, from);
                mc.okx_from = (g.supply.okx_first_bar != 0).then_some(g.supply.okx_first_bar);
                let first = g.micro.is_none();
                let full = std::mem::take(&mut g.supply.okx_sum_full) || first;
                let dirty = std::mem::take(&mut g.supply.okx_sum_dirty);
                mc.okx_bars = g.supply.okx_sums(if full { None } else { Some(&dirty) });
                mc.okx_full = full && mc.okx_from.is_some();
                mc.streams = Some(Streams { fo: self.fo.lock().unwrap().clone(), mp: g.mp.clone() });
                mc.flow = Some(situation::flow_delta(&g.nets, g.nets_sent.as_ref()));
                let fresh = !mc.okx_oi5m.is_empty() || !mc.okx_bars.is_empty() || mc.okx_full || g.micro.as_ref().is_none_or(|o| crate::proto::Micro { ts: mc.ts, ..o.clone() } != mc);
                if fresh {
                    g.seq += 1;
                    let _ = self.tx.send(Stamped { s: s.clone(), seq: g.seq, ex_ms: 0, rx_us: now_us, tx_us: 0, ev: Event::Micro(mc.clone()) });
                }
                mc.okx_oi5m.clear();
                (mc.okx_bars, mc.okx_full) = (vec![], false);
                if fresh {
                    g.nets_sent = Some(g.nets.clone());
                }
                mc.flow = Some(situation::flow_delta(&g.nets, Some(&g.nets))); // 들고 있는 값 = «바뀐 것 없음»(다음 회차 비교용)
                g.micro = Some(mc);
            }
            let candles: BTreeMap<i64, [f64; 3]> = g.ohlc.iter().map(|(&b, v)| (b, v.0)).collect();
            let liq: BTreeMap<i64, [f64; 2]> = g.supply.liq5m.iter().map(|(&b, v)| (b, v.0)).collect();
            let mid = match &g.book {
                Some(Stamped { ev: Event::Book { bid, ask, .. }, .. }) => Some((bid + ask) / 2.0),
                _ => None,
            };
            let fbars = g.supply.fp.bars.iter().map(|(&b, bar)| (b, &bar.cells));
            let inp = situation::inputs(now_s, bucket, fbars, g.supply.cells_of(Venue::Bn), &candles, btc.as_ref(), &liq, &g.marks.ring, mid);
            let prev = g.sit.as_ref().and_then(|x| x.dir).unwrap_or(0);
            let mut res = situation::classify(&inp, prev);
            res.computed_at = now_s;
            let same = g.sit.as_ref().is_some_and(|o| Situation { computed_at: now_s, ..o.clone() } == res);
            g.sit = Some(res.clone());
            if same {
                continue;
            }
            g.seq += 1;
            let _ = self.tx.send(Stamped { s: s.clone(), seq: g.seq, ex_ms: 0, rx_us: now_us, tx_us: 0, ev: Event::Situation(res) });
        }
    }

    /// 시험용: 체결로 센 5분봉 [고가, 저가, 종가].
    pub fn ohlc(&self, s: &str) -> Option<Vec<(i64, [f64; 3])>> {
        Some(self.syms.get(s)?.lock().unwrap().ohlc.iter().map(|(&b, v)| (b, v.0)).collect())
    }

    /// 바뀐 종목의 마지막 체결가를 «모든 종목» 메시지(`s = "*"`)로 민다(lib.rs 가 PRICES_MS 마다). 바뀐 것이 없으면 안 보낸다.
    pub fn flush_prices(&self) {
        let mut rows = vec![];
        for (s, cell) in &self.syms {
            let mut g = cell.lock().unwrap();
            if std::mem::take(&mut g.quote.price_dirty) {
                if let Some((px, ex_ms, _)) = g.quote.last {
                    rows.push((s.clone(), px, ex_ms));
                }
            }
        }
        if !rows.is_empty() {
            self.send_prices(rows);
        }
    }

    fn send_prices(&self, mut rows: Vec<(String, f64, i64)>) -> u64 {
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        let mut seq = self.prices_seq.lock().unwrap();
        *seq += 1;
        let ex_ms = rows.iter().map(|r| r.2).max().unwrap_or(0);
        let _ = self.tx.send(Stamped { s: "*".into(), seq: *seq, ex_ms, rx_us: crate::now_us(), tx_us: 0, ev: Event::Prices { rows } });
        *seq
    }

    /// 붙은 구독자에게 처음 한 번: 모든 종목의 마지막 체결가(번호 = 지금 번호 -- 그 뒤 번호만 받으면 빠짐·겹침이 없다).
    pub fn prices_snapshot(&self) -> Stamped {
        let seq = self.prices_seq.lock().unwrap();
        let mut rows: Vec<(String, f64, i64)> =
            self.syms.iter().filter_map(|(s, c)| c.lock().unwrap().quote.last.map(|(px, ex_ms, _)| (s.clone(), px, ex_ms))).collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        Stamped { s: "*".into(), seq: *seq, ex_ms: 0, rx_us: crate::now_us(), tx_us: 0, ev: Event::Prices { rows } }
    }

    /// 1분봉 연결 재연결(feed): 빠진 분을 모르니 모은 종가를 버린다(quote.rs reset).
    pub fn kline_reset(&self, s: &str) {
        if let Some(cell) = self.syms.get(s) {
            cell.lock().unwrap().quote.reset();
        }
    }

    /// 읽기 API(같은 프로세스의 rtorder): REST bookTicker 꼴 dict + 받은 시각(µs). 받은 적 없으면 None.
    pub fn book_ticker(&self, s: &str) -> Option<(serde_json::Value, i64)> {
        self.syms.get(s)?.lock().unwrap().quote.book_ticker(s)
    }

    /// 읽기 API: 마감 1분봉 31개의 변동성(bp/√분) + 마지막 마감 1분봉을 받은 시각(µs). 31개가 안 모였으면 (None, 시각).
    pub fn vol_bpm(&self, s: &str) -> Option<(Option<f64>, i64)> {
        let g = self.syms.get(s)?.lock().unwrap();
        Some((g.quote.vol_bpm, g.quote.kline_rx_us))
    }

    /// 읽기 API: 받아 둔 시세 전부(스냅숏 칸과 같은 것).
    pub fn quote(&self, s: &str) -> Option<crate::proto::Quote> {
        Some(self.syms.get(s)?.lock().unwrap().quote.msg())
    }

    /// 읽기 API(손절 감시 · orders.py sec_prices): 1초 마지막 체결가 [(초, 가격)] -- 가격 0(그 초 체결 없음·크기 칸만 있는 초)은 뺀다
    /// (server.py 4951 `[(t, float(c[6])) for t, c in sec.items() if c[6]]`). 오래된 초부터.
    pub fn sec_prices(&self, s: &str) -> Option<Vec<(i64, f64)>> {
        let g = self.syms.get(s)?.lock().unwrap();
        Some(g.supply.cells_of(Venue::Bn).iter().filter(|(_, c)| c[6] != 0.0).map(|(&t, c)| (t, c[6])).collect())
    }
}

/// 체결 하나를 5분봉 [고가, 저가, 종가]에 더한다(klines 와 같은 값 -- 종가 = 체결 id 가 가장 큰 체결). 새 봉이 생길 때 15봉 앞을 지운다.
fn ohlc_add(ohlc: &mut BTreeMap<i64, ([f64; 3], u64)>, px: f64, id: u64, ex_ms: i64) {
    let b = crate::footprint::bar_of(ex_ms);
    match ohlc.get_mut(&b) {
        Some((v, last)) => {
            v[0] = v[0].max(px);
            v[1] = v[1].min(px);
            if id >= *last {
                (v[2], *last) = (px, id);
            }
        }
        None => {
            ohlc.insert(b, ([px, px, px], id));
            // 상황 근거는 완결 봉 13개만 본다(situation.rs WINDOW) -- 늦은 체결 몫으로 둘 더
            ohlc.retain(|&k, _| k > b - 15 * situation::BAR_S);
        }
    }
}
