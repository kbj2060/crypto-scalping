//! 심볼별 최신 호가·마크·최근 체결 링.
//! 락: 심볼마다 std Mutex 하나. 잡는 동안 await 하지 않고 대입·링 push·broadcast send(동기) 만 한다 -- 수 µs.
//! 번호 매기기와 broadcast 를 같은 락 안에서 해서, 스냅숏(같은 락)과 밀기 사이에 빠짐·겹침이 없다.
use crate::proto::{Event, Stamped};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

#[derive(Default)]
struct Sym {
    seq: u64,
    book: Option<Stamped>,
    mark: Option<Stamped>,
    trades: VecDeque<Stamped>,
}

pub struct State {
    syms: HashMap<String, Mutex<Sym>>,
    ring_cap: usize,
    pub tx: broadcast::Sender<Stamped>,
}

impl State {
    pub fn new(symbols: &[String], ring_cap: usize, queue_cap: usize) -> Arc<Self> {
        let syms = symbols.iter().map(|s| (s.clone(), Mutex::default())).collect();
        Arc::new(State { syms, ring_cap, tx: broadcast::channel(queue_cap).0 })
    }

    /// 모르는 심볼은 버린다.
    pub fn apply(&self, mut m: Stamped) {
        let Some(cell) = self.syms.get(&m.s) else { return };
        let mut g = cell.lock().unwrap();
        g.seq += 1;
        m.seq = g.seq;
        match m.ev {
            Event::Trade { .. } => {
                if g.trades.len() == self.ring_cap {
                    g.trades.pop_front();
                }
                g.trades.push_back(m.clone());
            }
            Event::Book { .. } => g.book = Some(m.clone()),
            Event::Mark { .. } => g.mark = Some(m.clone()),
            Event::Snapshot { .. } => return,
        }
        let _ = self.tx.send(m); // 구독자 0명이면 Err -- 그대로 둔다
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
            },
        })
    }
}
