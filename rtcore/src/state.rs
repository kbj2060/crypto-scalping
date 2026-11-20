//! 심볼별 최신 호가·마크·최근 체결 링.
//! 락: 심볼마다 std Mutex 하나. 잡는 동안 await 하지 않고 대입·링 push·broadcast send(동기) 만 한다 -- 수 µs.
//! 번호 매기기와 broadcast 를 같은 락 안에서 해서, 스냅숏(같은 락)과 밀기 사이에 빠짐·겹침이 없다.
//! 1초 수급·풋프린트는 사건마다 밀지 않고 FLUSH_MS 마다 모아 민다(flush). 스냅숏은 모으던 것까지 이미 반영한 상태라,
//! 스냅숏 뒤 첫 묶음이 같은 값을 한 번 더 덮어써도 결과가 같다.
use crate::proto::{Event, Stamped, Supply1s, Venue};
use crate::restore;
use crate::supply::Supply;
use std::collections::{HashMap, VecDeque};
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
}

impl State {
    pub fn new(symbols: &[String], ring_cap: usize, queue_cap: usize) -> Arc<Self> {
        Arc::new(Self::build(symbols, ring_cap, queue_cap))
    }

    fn build(symbols: &[String], ring_cap: usize, queue_cap: usize) -> State {
        let syms = symbols
            .iter()
            .map(|s| {
                let sym = Sym { seq: 0, book: None, mark: None, trades: VecDeque::new(), supply: Supply::new(s), pending: None, first: None };
                (s.clone(), Mutex::new(sym))
            })
            .collect();
        State { syms, ring_cap, tx: broadcast::channel(queue_cap).0, restore: None, wake: None }
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
            let bn = restore::read_bn_liq(&cfg.bn_ctx, s, now_us as f64 / 1000.0).unwrap_or_else(|e| {
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
            sup.load_liqs(bn, by);
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
        let liq = restore::read_okx_liq(&cfg.okx_ctx, &inst, t0_ms - crate::supply::SUPPLY_1S_SECONDS * 1000, t0_ms);
        let now = crate::now_us();
        let mut g = cell.lock().unwrap();
        let (line, up) = g.supply.merge_okx(secs, liq, now);
        Self::queue(&mut g, up, now);
        println!("rtcore 복원 {s}: {line}");
        Some(line)
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
                // 24시간 봉 창은 «수신 시각» 기준(파이썬 time.time()) -- rx_us 가 그 벽시계다.
                supply = g.supply.trade(px, qty, m.ex_ms, sell, id as i64, m.rx_us);
            }
            Event::Liq { venue, long, qty, px } => supply = g.supply.liq(venue, long, qty, px, m.ex_ms, m.rx_us),
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
            Event::Book { .. } => g.book = Some(m.clone()),
            Event::Mark { .. } => g.mark = Some(m.clone()),
            Event::Snapshot { .. } | Event::Supply1s(_) | Event::Footprint(_) => return,
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
            if fp.is_none() && sup.is_none() {
                continue;
            }
            let (ex_ms, rx_us) = g.first.take().unwrap_or_else(|| (0, crate::now_us()));
            for ev in [sup.map(Event::Supply1s), fp.map(Event::Footprint)].into_iter().flatten() {
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
            },
        })
    }
}
