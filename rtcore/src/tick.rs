//! 틱 봉(2026-11-24) -- 바이낸스 선물 `@trade` 개별 체결 n 건마다 봉 하나(n = SIZES). 화면 dashboard/live/tick_charts.js · rtserve `/api/tick-bars`.
//! 흐름 엔진 종목(footprint::spec 이 있는 종목)만. 체결은 supply.rs `Supply::trade` 가 넘긴다(풋프린트와 같은 자리 · 같은 체결).
//!
//! 규칙:
//! - 봉 = proto::TickRow [번호, 첫 ms, 끝 ms, 시, 고, 저, 종, 수량 합, 델타(매수 − 매도), 체결 수]. n 번째 체결에서 닫히고 다음 체결부터 새 봉.
//! - 체결 id 가 지금까지 본 가장 큰 id 이하면(순서 밖·중복) 넣지 않는다 -- 개별 체결을 받은 순서대로만 센다.
//! - 크기마다 닫힌 봉 최근 TICK_KEEP 개 + 형성 봉 하나. `k` = 지금까지 닫힌 봉 총수(형성 봉 번호 = k).
//! - 스냅숏 = 전체(proto Snapshot.tick) · 밀기 = 묶음(state.rs FLUSH_MS)마다 «지난 밀기 뒤 닫힌 봉 + 형성 봉»(proto Event::TickBars).
//!   받는 규칙은 proto TickBars::merge.
//! - 재시작: 코어 스냅숏 파일(restore.rs Snap.tick)에 통째로 싣고 되살린다.
//!   ponytail: 꺼져 있던 사이 체결은 메우지 않는다 -- 테이프는 1초 칸 합이라 체결 순서·수(봉 경계·시가/종가)를 되살릴 수 없다.
//!   형성 봉은 그 공백을 품은 채 이어 센다. 메우려면 수집기가 개별 체결(또는 aggTrades 아닌 @trade id 범위)을 남겨야 한다.
use crate::proto::{TickBars, TickLane, TickRow};

pub const SIZES: [i64; 5] = [100, 500, 1000, 2000, 5000];

pub struct Ticks {
    pub(crate) on: bool,
    pub(crate) all: TickBars,
    /// 크기마다 지난 밀기 때의 k -- 이 번호 이상인 닫힌 봉을 다음 밀기에 싣는다.
    sent: Vec<i64>,
    pub(crate) last_id: Option<u64>,
    dirty: bool,
}

impl Ticks {
    pub fn new(on: bool) -> Ticks {
        let lanes = SIZES.iter().map(|&n| TickLane { n, k: 0, bars: vec![], cur: None }).collect();
        Ticks { on, all: TickBars { lanes }, sent: vec![0; SIZES.len()], last_id: None, dirty: false }
    }

    /// 개별 체결 하나(`sell` = 공격자 매도).
    pub fn add(&mut self, id: u64, ms: i64, px: f64, qty: f64, sell: bool) {
        if !self.on || self.last_id.is_some_and(|l| id <= l) {
            return;
        }
        self.last_id = Some(id);
        let d = if sell { -qty } else { qty };
        for l in &mut self.all.lanes {
            let c: &mut TickRow = l.cur.get_or_insert((l.k, ms, ms, px, px, px, px, 0.0, 0.0, 0));
            c.2 = ms;
            c.4 = c.4.max(px);
            c.5 = c.5.min(px);
            c.6 = px;
            c.7 += qty;
            c.8 += d;
            c.9 += 1;
            if c.9 == l.n {
                l.bars.push(l.cur.take().unwrap());
                l.k += 1;
                let over = l.bars.len().saturating_sub(crate::proto::TICK_KEEP);
                l.bars.drain(..over);
            }
        }
        self.dirty = true;
    }

    pub fn snapshot(&self) -> Option<TickBars> {
        self.on.then(|| self.all.clone())
    }

    /// 지난 밀기 뒤 바뀐 몫(체결이 없었으면 None).
    pub fn take_delta(&mut self) -> Option<TickBars> {
        if !std::mem::take(&mut self.dirty) {
            return None;
        }
        let lanes = self
            .all
            .lanes
            .iter()
            .zip(&mut self.sent)
            .map(|(l, s)| {
                let bars = l.bars.iter().filter(|b| b.0 >= *s).copied().collect();
                *s = l.k;
                TickLane { n: l.n, k: l.k, bars, cur: l.cur }
            })
            .collect();
        Some(TickBars { lanes })
    }

    /// 코어 스냅숏 파일에서(시작 때 · 수신 전). 크기 표가 다르면 버린다.
    pub(crate) fn load(&mut self, all: TickBars, last_id: Option<u64>) {
        if !self.on || all.lanes.iter().map(|l| l.n).ne(SIZES) {
            return;
        }
        self.sent = all.lanes.iter().map(|l| l.k).collect();
        (self.all, self.last_id) = (all, last_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 합성 체결 열: id 1.., 가격 = 100 + (i % 7), 수량 = 1 + i % 3, 짝수 번째 = 매도.
    fn feed(t: &mut Ticks, ids: std::ops::Range<u64>) {
        for i in ids {
            t.add(i, 1_000 + i as i64, 100.0 + (i % 7) as f64, 1.0 + (i % 3) as f64, i % 2 == 0);
        }
    }

    /// 손으로 센 봉(체결 i0..i0+n).
    fn expect(num: i64, i0: u64, n: u64) -> TickRow {
        let ids: Vec<u64> = (i0..i0 + n).collect();
        let px = |i: u64| 100.0 + (i % 7) as f64;
        let q = |i: u64| 1.0 + (i % 3) as f64;
        let hi = ids.iter().map(|&i| px(i)).fold(f64::MIN, f64::max);
        let lo = ids.iter().map(|&i| px(i)).fold(f64::MAX, f64::min);
        let vol: f64 = ids.iter().map(|&i| q(i)).sum();
        let d: f64 = ids.iter().map(|&i| if i % 2 == 0 { -q(i) } else { q(i) }).sum();
        (num, 1_000 + i0 as i64, 1_000 + (i0 + n - 1) as i64, px(i0), hi, lo, px(i0 + n - 1), vol, d, n as i64)
    }

    #[test]
    fn bars_close_on_nth_trade() {
        let mut t = Ticks::new(true);
        feed(&mut t, 1..100); // 99건 -- 100틱 봉은 아직 형성 중
        let l100 = &t.all.lanes[0];
        assert_eq!((l100.k, l100.bars.len(), l100.cur.unwrap().9), (0, 0, 99));
        feed(&mut t, 100..101); // 100번째에서 닫힘
        let l100 = &t.all.lanes[0];
        assert_eq!((l100.k, l100.cur), (1, None));
        assert_eq!(l100.bars[0], expect(0, 1, 100));
        feed(&mut t, 101..102); // 다음 체결부터 새 봉(번호 1)
        assert_eq!(t.all.lanes[0].cur.unwrap(), expect(1, 101, 1));
        // 크기별 k
        feed(&mut t, 102..12_001);
        let ks: Vec<i64> = t.all.lanes.iter().map(|l| l.k).collect();
        assert_eq!(ks, vec![120, 24, 12, 6, 2]);
        assert_eq!(t.all.lanes[4].bars[1], expect(1, 5_001, 5_000));
        assert_eq!(t.all.lanes[3].cur, None, "12000 = 2000 × 6 -- 막 닫혔다");
        assert_eq!(t.all.lanes[4].cur.unwrap(), expect(2, 10_001, 2_000));
    }

    #[test]
    fn keeps_last_240_and_skips_out_of_order() {
        let mut t = Ticks::new(true);
        feed(&mut t, 1..30_051); // 100틱 봉 300개 닫힘 + 50건 형성
        let l = &t.all.lanes[0];
        assert_eq!((l.k, l.bars.len()), (300, 240));
        assert_eq!(l.bars[0], expect(60, 6_001, 100)); // 닫힌 봉 i 의 번호 = k − 보관수 + i
        assert_eq!(l.bars[239].0, 299);
        assert_eq!(l.cur.unwrap(), expect(300, 30_001, 50));
        let before = t.all.clone();
        t.add(30_000, 9, 1.0, 9.0, false); // 순서 밖·중복 id
        t.add(30_050, 9, 1.0, 9.0, false);
        assert_eq!(t.all, before);
        // 꺼진 종목은 아무것도 안 든다
        let mut off = Ticks::new(false);
        feed(&mut off, 1..200);
        assert!(off.snapshot().is_none() && off.take_delta().is_none());
    }

    /// 밀기만 받아 지은 상태 == 코어 상태(처음부터 받은 쪽 · 가운데서 스냅숏으로 붙은 쪽).
    #[test]
    fn pushes_rebuild_snapshot() {
        let mut t = Ticks::new(true);
        let mut from_start = Ticks::new(true).snapshot().unwrap();
        let mut mid: Option<TickBars> = None;
        let mut i = 1u64;
        let mut step = 0u64;
        while i < 40_000 {
            step = (step * 37 + 11) % 997; // 묶음 크기 1~997 건(봉 여러 개가 한 묶음에 닫히는 경우 포함)
            feed(&mut t, i..i + step + 1);
            i += step + 1;
            if i > 20_000 && mid.is_none() {
                mid = t.snapshot(); // 가운데서 붙음 -- 다음 밀기는 스냅숏에 이미 든 봉을 또 싣는다(번호로 건너뛰어야 한다)
            }
            let d = t.take_delta().unwrap();
            from_start.merge(d.clone());
            if let Some(m) = &mut mid {
                m.merge(d);
            }
        }
        assert_eq!(from_start, t.all);
        assert_eq!(mid.unwrap(), t.all);
        assert!(t.take_delta().is_none(), "체결이 없으면 밀 것도 없다");
    }

    /// 코어 스냅숏 파일 왕복: 되살린 상태 == 쓴 상태 · 그 뒤 같은 체결을 넣으면 끊김 없이 이어진다 · 옛 파일(tick 없음)은 빈 채로.
    #[test]
    fn restore_roundtrip() {
        let mut a = crate::supply::Supply::new("ethusdt");
        let t0 = 1_763_700_000_000i64;
        let tr = |s: &mut crate::supply::Supply, i: u64| {
            s.trade(3000.0 + (i % 13) as f64 * 0.5, 0.25 + (i % 5) as f64, t0 + i as i64 * 7, i % 3 == 0, i as i64 + 10, (t0 + i as i64 * 7) * 1000);
        };
        for i in 1..7_777 {
            tr(&mut a, i);
        }
        a.rs.merged = true; // 스냅숏 쓰기 조건(restore.rs to_snap)
        let text = serde_json::to_vec(&a.to_snap("ethusdt").unwrap()).unwrap();
        let mut b = crate::supply::Supply::new("ethusdt");
        b.load_snap(serde_json::from_slice(&text).unwrap());
        assert_eq!((&b.tick.all, b.tick.last_id), (&a.tick.all, a.tick.last_id));
        tr(&mut b, 7_000); // 되살린 뒤 이미 센 id 는 다시 안 센다
        assert_eq!(b.tick.all, a.tick.all);
        for i in 7_777..12_000 {
            tr(&mut a, i);
            tr(&mut b, i);
        }
        assert_eq!(b.tick.all, a.tick.all);
        assert_eq!(b.tick.take_delta().unwrap().lanes.iter().map(|l| l.k).collect::<Vec<_>>(), vec![119, 23, 11, 5, 2]);
        // 옛 파일: tick 칸이 없다
        let mut v: serde_json::Value = serde_json::from_slice(&text).unwrap();
        v.as_object_mut().unwrap().retain(|k, _| !k.starts_with("tick"));
        let mut c = crate::supply::Supply::new("ethusdt");
        c.load_snap(serde_json::from_value(v).unwrap());
        assert_eq!((c.tick.all.lanes[0].k, c.tick.last_id), (0, None));
    }
}
