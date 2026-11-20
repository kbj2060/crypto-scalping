//! 심볼별 최신 호가·마크·최근 체결 링.
//! 락: 심볼마다 std Mutex 하나. 잡는 동안 await 하지 않고 대입·링 push·broadcast send(동기) 만 한다 -- 수 µs.
//! 번호 매기기와 broadcast 를 같은 락 안에서 해서, 스냅숏(같은 락)과 밀기 사이에 빠짐·겹침이 없다.
//! 1초 수급·풋프린트는 사건마다 밀지 않고 FLUSH_MS 마다 모아 민다(flush). 스냅숏은 모으던 것까지 이미 반영한 상태라,
//! 스냅숏 뒤 첫 묶음이 같은 값을 한 번 더 덮어써도 결과가 같다.
use crate::proto::{Event, Stamped, Supply1s, Venue};
use crate::supply::Supply;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

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
}

impl State {
    pub fn new(symbols: &[String], ring_cap: usize, queue_cap: usize) -> Arc<Self> {
        let syms = symbols
            .iter()
            .map(|s| {
                let sym = Sym { seq: 0, book: None, mark: None, trades: VecDeque::new(), supply: Supply::new(s), pending: None, first: None };
                (s.clone(), Mutex::new(sym))
            })
            .collect();
        Arc::new(State { syms, ring_cap, tx: broadcast::channel(queue_cap).0 })
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
                if g.trades.len() == self.ring_cap {
                    g.trades.pop_front();
                }
                g.trades.push_back(m.clone());
                // 24시간 봉 창은 «수신 시각» 기준(파이썬 time.time()) -- rx_us 가 그 벽시계다.
                supply = g.supply.trade(px, qty, m.ex_ms, sell, id as i64, m.rx_us);
            }
            Event::Liq { venue, long, qty, px } => supply = g.supply.liq(venue, long, qty, px, m.ex_ms, m.rx_us),
            Event::Flow { venue, px, qty, sell } => supply = g.supply.flow(venue, px, qty, sell, m.ex_ms, m.rx_us),
            Event::Oi { oi, .. } => supply = g.supply.oi(oi, m.ex_ms, m.rx_us),
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
