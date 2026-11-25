//! 틱 봉(2026-11-24) -- 바이낸스 선물 `@trade` 개별 체결 n 건마다 봉 하나(n = SIZES). 화면 dashboard/live/tick_charts.js · rtserve `/api/tick-bars`.
//! 흐름 엔진 종목(footprint::spec 이 있는 종목)만. 체결은 supply.rs `Supply::trade` 가 넘긴다(풋프린트와 같은 자리 · 같은 체결).
//!
//! 규칙:
//! - 봉 = proto::TickRow [번호, 첫 ms, 끝 ms, 시, 고, 저, 종, 수량 합, 델타(매수 − 매도), 체결 수, [[가격, 수량], ..]]. n 번째 체결에서 닫히고 다음 체결부터 새 봉.
//! - 가격 칸(2026-11-25 · 화면의 «보이는 구간 POC»): 칸 키 = round(가격 / 칸 폭)(풋프린트와 같은 규칙 · footprint.rs) · 가격 = rdp(키 × 칸 폭, 자릿수) ·
//!   매수·매도 구분 없이 체결 수량을 더한다(칸 합 = 봉 수량 합).
//! - 체결 id 가 지금까지 본 가장 큰 id 이하면(순서 밖·중복) 넣지 않는다 -- 개별 체결을 받은 순서대로만 센다.
//! - 크기마다 닫힌 봉 최근 TICK_KEEP 개 + 형성 봉 하나. `k` = 지금까지 닫힌 봉 총수(형성 봉 번호 = k).
//! - 스냅숏 = 전체(proto Snapshot.tick) · 밀기 = 묶음(state.rs FLUSH_MS)마다 «지난 밀기 뒤 닫힌 봉 + 형성 봉»(proto Event::TickBars).
//!   받는 규칙은 proto TickBars::merge.
//! - 재시작: 코어 스냅숏 파일(restore.rs Snap.tick)에 통째로 싣고 되살린다.
//!   ponytail: 꺼져 있던 사이 체결은 메우지 않는다 -- 테이프는 1초 칸 합이라 체결 순서·수(봉 경계·시가/종가)를 되살릴 수 없다.
//!   형성 봉은 그 공백을 품은 채 이어 센다. 메우려면 수집기가 개별 체결(또는 aggTrades 아닌 @trade id 범위)을 남겨야 한다.
use crate::proto::{BigRow, TickBars, TickLane, TickRow, BIG_KEEP};

pub const SIZES: [i64; 5] = [100, 500, 1000, 2000, 5000];

pub struct Ticks {
    /// 풋프린트 (칸 폭, 자릿수) -- None 이면 꺼짐.
    spec: Option<(f64, usize)>,
    pub(crate) all: TickBars,
    /// 크기마다 지난 밀기 때의 k -- 이 번호 이상인 닫힌 봉을 다음 밀기에 싣는다.
    sent: Vec<i64>,
    /// 다음 고래 주문 번호 · 지난 밀기 때의 번호.
    big_k: i64,
    big_sent: i64,
    pub(crate) last_id: Option<u64>,
    dirty: bool,
}

impl Ticks {
    pub fn new(spec: Option<(f64, usize)>) -> Ticks {
        let lanes = SIZES.iter().map(|&n| TickLane { n, k: 0, bars: vec![], cur: None }).collect();
        Ticks { spec, all: TickBars { lanes, big: vec![] }, sent: vec![0; SIZES.len()], big_k: 0, big_sent: 0, last_id: None, dirty: false }
    }

    /// 개별 체결 하나(`sell` = 공격자 매도).
    pub fn add(&mut self, id: u64, ms: i64, px: f64, qty: f64, sell: bool) {
        let Some((bucket, dp)) = self.spec else { return };
        if self.last_id.is_some_and(|l| id <= l) {
            return;
        }
        let cell = crate::footprint::rdp((px / bucket).round_ties_even() * bucket, dp);
        self.last_id = Some(id);
        let d = if sell { -qty } else { qty };
        for l in &mut self.all.lanes {
            let c: &mut TickRow = l.cur.get_or_insert_with(|| TickRow(l.k, ms, ms, px, px, px, px, 0.0, 0.0, 0, vec![]));
            c.2 = ms;
            c.4 = c.4.max(px);
            c.5 = c.5.min(px);
            c.6 = px;
            c.7 += qty;
            c.8 += d;
            c.9 += 1;
            match c.10.binary_search_by(|e| e.0.total_cmp(&cell)) {
                Ok(i) => c.10[i].1 += qty,
                Err(i) => c.10.insert(i, (cell, qty)),
            }
            if c.9 == l.n {
                l.bars.push(l.cur.take().unwrap());
                l.k += 1;
                let over = l.bars.len().saturating_sub(crate::proto::TICK_KEEP);
                l.bars.drain(..over);
            }
        }
        self.dirty = true;
    }

    /// 닫힌 고래 주문 하나(`ms` = 첫 체결). 경계 판정은 부르는 쪽(supply.rs -- 풋프린트 고래 칸과 같은 하한).
    pub fn big(&mut self, ms: i64, px: f64, qty: f64, sell: bool) {
        if self.spec.is_none() {
            return;
        }
        self.all.big.push(BigRow(self.big_k, ms, px, qty, sell));
        self.big_k += 1;
        let over = self.all.big.len().saturating_sub(BIG_KEEP);
        self.all.big.drain(..over);
        self.dirty = true;
    }

    pub fn snapshot(&self) -> Option<TickBars> {
        self.spec.map(|_| self.all.clone())
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
                let bars = l.bars.iter().filter(|b| b.0 >= *s).cloned().collect();
                *s = l.k;
                TickLane { n: l.n, k: l.k, bars, cur: l.cur.clone() }
            })
            .collect();
        let big = self.all.big.iter().filter(|b| b.0 >= self.big_sent).cloned().collect();
        self.big_sent = self.big_k;
        Some(TickBars { lanes, big })
    }

    /// 코어 스냅숏 파일에서(시작 때 · 수신 전). 크기 표가 다르면 버린다.
    pub(crate) fn load(&mut self, all: TickBars, last_id: Option<u64>) {
        if self.spec.is_none() || all.lanes.iter().map(|l| l.n).ne(SIZES) {
            return;
        }
        self.sent = all.lanes.iter().map(|l| l.k).collect();
        self.big_k = all.big.last().map_or(0, |b| b.0 + 1);
        self.big_sent = self.big_k;
        (self.all, self.last_id) = (all, last_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEC: Option<(f64, usize)> = Some((0.5, 2));

    #[test]
    fn big_orders_number_cap_push_restore() {
        let mut t = Ticks::new(SPEC);
        let mut got = t.snapshot().unwrap();
        for i in 0..(BIG_KEEP as i64 + 30) {
            t.big(1_000 + i, 100.0, 5.0, i % 2 == 0);
            if i % 7 == 0 {
                got.merge(t.take_delta().unwrap());
            }
        }
        got.merge(t.take_delta().unwrap());
        assert!(t.take_delta().is_none());
        let s = t.snapshot().unwrap();
        assert_eq!(got, s, "밀기만으로 지은 것 == 스냅숏");
        assert_eq!((s.big.len(), s.big[0].0, s.big.last().unwrap().0), (BIG_KEEP, 30, BIG_KEEP as i64 + 29));
        let mut r = Ticks::new(SPEC);
        r.load(s.clone(), None);
        r.big(9, 1.0, 2.0, true);
        assert_eq!(r.take_delta().unwrap().big, vec![BigRow(BIG_KEEP as i64 + 30, 9, 1.0, 2.0, true)], "복원 뒤 번호를 잇고 새 것만 민다");
        let old = serde_json::to_value(&s).unwrap();
        let lanes_only: TickBars = serde_json::from_value(serde_json::json!({"lanes": old["lanes"]})).unwrap();
        assert!(lanes_only.big.is_empty(), "옛 파일·메시지");
        let mut off = Ticks::new(None);
        off.big(1, 1.0, 1.0, false);
        assert!(off.snapshot().is_none() && off.take_delta().is_none());
    }

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
        let mut cells: Vec<(f64, f64)> = vec![];
        for &i in &ids {
            match cells.iter_mut().find(|c| c.0 == px(i)) {
                Some(c) => c.1 += q(i),
                None => cells.push((px(i), q(i))),
            }
        }
        cells.sort_by(|a, b| a.0.total_cmp(&b.0));
        TickRow(num, 1_000 + i0 as i64, 1_000 + (i0 + n - 1) as i64, px(i0), hi, lo, px(i0 + n - 1), vol, d, n as i64, cells)
    }

    #[test]
    fn bars_close_on_nth_trade() {
        let mut t = Ticks::new(SPEC);
        feed(&mut t, 1..100); // 99건 -- 100틱 봉은 아직 형성 중
        let l100 = &t.all.lanes[0];
        assert_eq!((l100.k, l100.bars.len(), l100.cur.as_ref().unwrap().9), (0, 0, 99));
        feed(&mut t, 100..101); // 100번째에서 닫힘
        let l100 = &t.all.lanes[0];
        assert_eq!((l100.k, &l100.cur), (1, &None));
        assert_eq!(l100.bars[0], expect(0, 1, 100));
        feed(&mut t, 101..102); // 다음 체결부터 새 봉(번호 1)
        assert_eq!(t.all.lanes[0].cur.clone().unwrap(), expect(1, 101, 1));
        // 크기별 k
        feed(&mut t, 102..12_001);
        let ks: Vec<i64> = t.all.lanes.iter().map(|l| l.k).collect();
        assert_eq!(ks, vec![120, 24, 12, 6, 2]);
        assert_eq!(t.all.lanes[4].bars[1], expect(1, 5_001, 5_000));
        assert_eq!(t.all.lanes[3].cur, None, "12000 = 2000 × 6 -- 막 닫혔다");
        assert_eq!(t.all.lanes[4].cur.clone().unwrap(), expect(2, 10_001, 2_000));
    }

    #[test]
    fn keeps_last_240_and_skips_out_of_order() {
        let mut t = Ticks::new(SPEC);
        feed(&mut t, 1..30_051); // 100틱 봉 300개 닫힘 + 50건 형성
        let l = &t.all.lanes[0];
        assert_eq!((l.k, l.bars.len()), (300, 240));
        assert_eq!(l.bars[0], expect(60, 6_001, 100)); // 닫힌 봉 i 의 번호 = k − 보관수 + i
        assert_eq!(l.bars[239].0, 299);
        assert_eq!(l.cur.clone().unwrap(), expect(300, 30_001, 50));
        let before = t.all.clone();
        t.add(30_000, 9, 1.0, 9.0, false); // 순서 밖·중복 id
        t.add(30_050, 9, 1.0, 9.0, false);
        assert_eq!(t.all, before);
        // 꺼진 종목은 아무것도 안 든다
        let mut off = Ticks::new(None);
        feed(&mut off, 1..200);
        assert!(off.snapshot().is_none() && off.take_delta().is_none());
    }

    /// 가격 칸: 칸 합 == 봉 수량(닫힌 봉 · 형성 봉 다) · 칸 경계 = 풋프린트 규칙(키 동률은 짝수 쪽) · 가격 오름차순 · 옛 10칸 행은 빈 목록.
    #[test]
    fn price_cells() {
        let mut t = Ticks::new(SPEC);
        // 칸 폭 0.5: 100.24 → 100.0 · 100.25(키 200.5 → 200) → 100.0 · 100.26 → 100.5 · 100.75(키 201.5 → 202) → 101.0 · 99.74 → 99.5
        for (i, px) in [100.24, 100.25, 100.26, 100.75, 99.74, 100.25].into_iter().enumerate() {
            t.add(i as u64 + 1, 1, px, (i + 1) as f64, i % 2 == 0);
        }
        let c = t.all.lanes[0].cur.clone().unwrap();
        assert_eq!(c.10, vec![(99.5, 5.0), (100.0, 9.0), (100.5, 3.0), (101.0, 4.0)]);
        // 칸 폭 0.0003·4자리(XRP): 가격 = rdp(키 × 칸 폭) 라 같은 칸은 같은 가격 글자
        let mut x = Ticks::new(Some((0.0003, 4)));
        x.add(1, 1, 2.10005, 1.0, false); // 키 7000.17 → 7000 → 2.1
        x.add(2, 1, 2.10044, 2.0, true); // 키 7001.47 → 7001 → 2.1003
        x.add(3, 1, 2.0999, 4.0, true); // 키 6999.67 → 7000 → 2.1
        assert_eq!(x.all.lanes[0].cur.clone().unwrap().10, vec![(2.1, 5.0), (2.1003, 2.0)]);
        // 긴 열: 칸 합 == 수량 합
        let mut y = Ticks::new(Some((0.5, 2)));
        let mut r = 3u64;
        for i in 1..23_456u64 {
            r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            y.add(i, i as i64, 3000.0 + ((r >> 40) % 900) as f64 * 0.01, ((r >> 20) % 1000) as f64 / 1000.0 + 0.001, r & 1 == 1);
        }
        let mut n = 0;
        for l in &y.all.lanes {
            for b in l.bars.iter().chain(&l.cur) {
                let sum: f64 = b.10.iter().map(|c| c.1).sum();
                assert!((sum - b.7).abs() < 1e-9, "{} {sum}", b.7);
                assert!(b.10.windows(2).all(|w| w[0].0 < w[1].0) && b.10.iter().all(|c| c.0 >= b.5 - 0.25 && c.0 <= b.4 + 0.25));
                n += 1;
            }
        }
        assert!(n > 240);
        // 옛 10칸 행
        let old: TickRow = serde_json::from_str("[3,1,2,1.5,2.0,1.0,1.5,9.0,-1.0,100]").unwrap();
        assert_eq!(old, TickRow(3, 1, 2, 1.5, 2.0, 1.0, 1.5, 9.0, -1.0, 100, vec![]));
        assert_eq!(serde_json::to_string(&c).unwrap().matches("[[99.5,5.0],[100.0,9.0],").count(), 1);
    }

    /// 밀기만 받아 지은 상태 == 코어 상태(처음부터 받은 쪽 · 가운데서 스냅숏으로 붙은 쪽).
    #[test]
    fn pushes_rebuild_snapshot() {
        let mut t = Ticks::new(SPEC);
        let mut from_start = Ticks::new(SPEC).snapshot().unwrap();
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
        let at_text = a.tick.all.clone();
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
        // 2026-11-24 판 파일: 봉 행이 10칸(가격 칸 없음) → 같은 봉 · 칸은 빈 목록
        let mut v: serde_json::Value = serde_json::from_slice(&text).unwrap();
        for l in v["tick"]["lanes"].as_array_mut().unwrap() {
            for r in l["bars"].as_array_mut().unwrap() {
                r.as_array_mut().unwrap().truncate(10);
            }
            if let Some(c) = l["cur"].as_array_mut() {
                c.truncate(10);
            }
        }
        let mut d = crate::supply::Supply::new("ethusdt");
        d.load_snap(serde_json::from_value(v).unwrap());
        let mut want = at_text;
        want.lanes.iter_mut().flat_map(|l| l.bars.iter_mut().chain(l.cur.as_mut())).for_each(|r| r.10.clear());
        assert_eq!(d.tick.all, want);
        assert!(want.lanes[0].bars[0].7 > 0.0);
        // 옛 파일: tick 칸이 없다
        let mut v: serde_json::Value = serde_json::from_slice(&text).unwrap();
        v.as_object_mut().unwrap().retain(|k, _| !k.starts_with("tick"));
        let mut c = crate::supply::Supply::new("ethusdt");
        c.load_snap(serde_json::from_value(v).unwrap());
        assert_eq!((c.tick.all.lanes[0].k, c.tick.last_id), (0, None));
    }
}
