//! 심볼별 최신 호가·마크·최근 체결 링.
//! 락: 심볼마다 std Mutex 하나. 잡는 동안 await 하지 않고 대입·링 push·broadcast send(동기) 만 한다 -- 수 µs.
//! 번호 매기기와 broadcast 를 같은 락 안에서 해서, 스냅숏(같은 락)과 밀기 사이에 빠짐·겹침이 없다.
use crate::proto::{Event, Stamped};
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
}

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
                let sym = Sym { seq: 0, book: None, mark: None, trades: VecDeque::new(), supply: Supply::new(s) };
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
                supply = g.supply.trade(px, qty, m.ex_ms, sell, id as i64, m.rx_us.div_euclid(1000));
            }
            Event::Book { .. } => g.book = Some(m.clone()),
            Event::Mark { .. } => g.mark = Some(m.clone()),
            Event::Snapshot { .. } | Event::Supply1s(_) => return,
        }
        let base = m.clone();
        let _ = self.tx.send(m); // 구독자 0명이면 Err -- 그대로 둔다
        if let Some(up) = supply {
            g.seq += 1;
            // ponytail: 체결마다 1건 민다(ETH 평소 ~266건/초) -- 화면이 버거우면 «같은 초 안 몇 ms 묶기»로(지연과 맞바꿈).
            let _ = self.tx.send(Stamped { seq: g.seq, ev: Event::Supply1s(up), ..base });
        }
    }

    /// 체결 스트림 재연결(feed): 열린 주문 묶음을 버린다(coin_flow.py 613 fp_orders.reset()).
    pub fn reset_orders(&self, s: &str) {
        if let Some(cell) = self.syms.get(s) {
            cell.lock().unwrap().supply.reset_orders();
        }
    }

    /// 시험용: 지금 1초 칸(반올림 전).
    pub fn supply_cells(&self, s: &str) -> Option<Vec<(i64, [f64; 7])>> {
        Some(self.syms.get(s)?.lock().unwrap().supply.cells().iter().map(|(&k, v)| (k, *v)).collect())
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
                supply: g.supply.snapshot(),
            },
        })
    }
}
