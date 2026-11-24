//! 풋프린트(5분봉 × 가격 칸) -- 2026-11-20 파이썬에서 옮김. tokio 비의존. 바이낸스 선물 `@trade` 만(파이썬도 2026-11-02 부터 바이낸스만).
//!
//! 원본: dashboard/coin_flow.py `footprint_add`(338~394 의 봉 칸 몫) · 봉 보관(354~365) · FLOW_SPECS(269~279)
//!       · dashboard/server.py `api_footprint`(3993~4059 · since=0, bars=144) · `fp_dhl` 의 1초 링 가지(3952~3965) · `bar_delta_hl`(237~243).
//! 체결 하나를 한 번만 처리한다: supply.rs `Supply::add` 가 같은 호출에서 이 봉 칸과 1초 칸을 같이 쌓는다(되묶기 상태도 하나).
//! 일치 정답지: rtcore/tools/gen_footprint_fixture.py → rtcore/tests/footprint_parity.rs.
//!
//! 옮긴 규칙(괄호 = 파이썬 위치 · 사고/근거 날짜):
//! - 칸 = [매수, 매도, 고래매수, 고래매도, 리테일매수, 리테일매도](coin_flow.py 79~80). 1초 칸과 순서가 다르다.
//!   총량(0·1)은 개별 체결로, 고래·리테일(2~5)은 되묶은 테이커 주문으로(370~380 · 2026-09-19 «@trade 로 세면 고래가 사라진다»).
//!   중형($10k~$100k)은 칸이 없다 -- 화면이 매수 − 고래 − 리테일로 얻는다.
//! - 가격 칸 키 = round(가격 / 칸 폭)(파이썬 round = 동률 짝수 · 366). 응답 가격 = round(키 × 칸 폭, price_dp)(server.py 4054).
//! - 봉 창: 체결의 봉이 없고 «수신 시각의 봉 − 287봉» 보다 오래되면 통째로 버린다. 새 봉이 생길 때만 그보다 오래된 봉을 지운다(355~365).
//! - 주문 단위 칸은 주문의 **첫 체결** 시각의 봉에 들어간다(supply.rs 의 되묶기 규칙 그대로) -- 봉 경계 직전에 시작한 주문은 앞 봉.
//! - 스냅숏 = since=0 응답의 마지막 144봉(app.js CHART_PAN_BARS = 144 · FOOTPRINT_MAX_WINDOW_BARS). 봉 링은 288봉을 든다.
//! - `dhl`(봉 안 누적 델타 [최저, 최고, 마감] · 2026-11-10): 봉 안 1초 칸의 (총매수 − 총매도)를 시간순으로 누적(봉 시작 0 도 후보),
//!   마감값이 봉 칸 델타와 max(0.5, 1%) 넘게 어긋나면 내지 않는다(server.py 3953~3965 · 11-10 리뷰 «구멍 난 링»).
//!
//! 파이썬과 다른 곳(일부러):
//! - `dhl` 을 어디서 재나: 파이썬은 «링의 가장 오래된 초 ≤ 봉 시작»이면 링에서, 아니면 봉이 끝나고 6분 뒤 테이프 SQLite 에서
//!   다시 읽는다(server.py 3962~3990). 코어는 테이프를 안 읽는다(다음 조각). 대신 **링이 봉을 다 드는 동안**(봉 시작 ≥ 가장 새 초 − 660 --
//!   링은 그 위 초를 지우지 않는다) 링에서 재고, 링이 봉의 첫 초를 지우기 직전 값을 **얼려** 둔다(테이프와 같은 체결의 같은 합).
//!   얼린 뒤 그 봉에 체결이 더 오면 값을 모르니 지운다(= 몸통만). 체결이 촘촘하면 두 판정은 같은 순간이다. 다른 경우:
//!   ① 체결이 드물어(6분 넘게 빈 틈) 파이썬이 «안 덮임»으로 보는 봉 -- 파이썬은 6분 뒤 테이프 값을, 코어는 링 값을 곧바로 낸다(같은 값).
//!   ② 660초 넘게 늦게 온 체결은 다음 새 초까지 링에 남아(supply.rs 21~22) 그 사이 파이썬은 뒤 봉들을 구멍 난 링으로 재다 버린다
//!   (그 순간만 몸통만) -- 코어는 영향 없음. 시험은 ② 순간의 dhl 대조만 뺀다(tests/footprint_parity.rs).
//!   ③ 파이썬은 테이프 값을 한 번 읽으면 기억한다(fp_dhl_cache) -- 늦은 체결이 바꿔도 옛값. 코어는 지운다(몸통만).
//!   ④ 재시작 직후: 복원(restore.rs)이 링 밖 봉은 테이프로(파이썬과 같은 SQL), 링 안 봉은 테이프로 채운 링으로 잰다.
//! - `agg`(REST 백필로 메운 봉 표시)·`symbol`·`venues` 는 싣지 않는다 -- 코어는 REST 백필이 없어 agg 는 늘 거짓, 나머지 둘은 화면이 안 읽는다.
//! - `ready`: 파이썬은 «백필이 끝나 빈 구간이 없다». 코어는 «창 144봉이 다 찼다»로 둔다 -- 화면의
//!   «체결 테이프 수집 중 N/144봉» 이 코어가 실제로 가진 봉 수를 보이게. 정답지 비교 대상이 아니다.
//!   빈 구간은 따로 `holes`(복원이 못 메운 초 구간에 걸친 봉 · restore.rs)로 싣는다 -- 파이썬에는 없는 칸.
//!
//! 1분봉(2026-11-24 · 원본 2026-11-23 coin_flow.py footprint_add 의 bars1m·ohlc1m·m1_from 374~396 · 저장/복원 441~443·472~479 ·
//! server.py footprint_1m 3997~4016): 5분봉 칸과 **같은 호출·같은 칸 규칙**(5분봉 창 판정을 통과한 체결만 · 총량은 개별 체결 ·
//! 크기 구간은 되묶은 주문). 봉 = 체결 시각의 분. 새 분이 생길 때만 «그 분 − 119분» 앞을 지운다(보관 120분).
//! OHLC 는 개별 체결로만: 시가 = 가장 이른 체결 시각의 가격(같은 ms 면 먼저 온 것) · 종가 = 가장 늦은 시각(같은 ms 면 나중 것).
//! m1_from 은 복원 때만 정한다(restore.rs load_snap). ponytail: 재시작 공백의 분은 메우지 않는다(테이프 복원은 5분봉 칸만 --
//! 원본도 ETH 외 코인은 같다 · ETH 원본은 REST aggTrades 로 메운다). 공백에 걸친 분은 덜 센 채 보인다.
use crate::proto::{Fp1m, Fp1mBar, FpBar, Footprint, Level};
use crate::supply::{r3, SUPPLY_1S_SECONDS};
use std::collections::{BTreeMap, BTreeSet};

pub const BAR_SECONDS: i64 = 300;
pub const KEEP_BARS: i64 = 288;
/// 스냅숏·밀기 창(봉 수). server.py FOOTPRINT_MAX_WINDOW_BARS = app.js CHART_PAN_BARS.
pub const WINDOW_BARS: usize = 144;
/// 1분봉(coin_flow.py FOOTPRINT_1M_SECONDS · FOOTPRINT_1M_KEEP).
pub const M1_SECONDS: i64 = 60;
pub const M1_KEEP: i64 = 120;

/// 1분봉 하나: 칸(5분봉과 같은 6열) · OHLC([시, 고, 저, 종], 첫 ms, 끝 ms) -- 개별 체결이 하나도 없으면 없음.
#[derive(Default, Clone, Debug, PartialEq)]
pub(crate) struct Bar1 {
    pub(crate) cells: BTreeMap<i64, [f64; 6]>,
    pub(crate) ohlc: Option<([f64; 4], i64, i64)>,
}

/// 파이썬 round(x, dp)(supply.rs r3 주석과 같은 방식).
pub(crate) fn rdp(x: f64, dp: usize) -> f64 {
    format!("{:.*}", dp, x).parse().unwrap_or(0.0)
}

/// (심볼, 칸 폭, 응답 가격 소수 자리) -- coin_flow.py FLOW_SPECS 사본. 표에 없는 종목은 파이썬에 흐름 엔진이 없다 → 풋프린트 없음.
// ponytail: 파이썬 표 사본 -- 칸 폭을 바꾸면 tools/gen_footprint_fixture.py 를 다시 돌려 정답지 specs 대조가 잡는다.
const SPECS: &[(&str, f64, usize)] = &[
    ("ethusdt", 0.5, 2),
    ("solusdt", 0.02, 2),
    ("xrpusdt", 0.0003, 4),
    ("btcusdt", 15.0, 0),
    ("sndkusdt", 0.2, 2),
    ("soxlusdt", 0.02, 2),
    ("skhynixusdt", 0.2, 2),
    ("muusdt", 0.2, 2),
];

pub fn spec(symbol: &str) -> Option<(f64, usize)> {
    let s = symbol.to_lowercase();
    SPECS.iter().find(|x| x.0 == s).map(|x| (x.1, x.2))
}

pub fn bar_of(ms: i64) -> i64 {
    ms.div_euclid(1000).div_euclid(BAR_SECONDS) * BAR_SECONDS
}

#[derive(Default)]
pub(crate) struct Bar {
    pub(crate) cells: BTreeMap<i64, [f64; 6]>,
    pub(crate) dhl: Option<[f64; 3]>,
    /// 칸이 생긴 차례(체결로 생긴 칸만 -- 원본 dict 차례). 복원으로 생긴 칸은 없다 → `in_order` 가 키 차례로 뒤에 붙인다.
    pub(crate) order: Vec<i64>,
    /// 재시작 공백을 테이프로 메운 봉(ETH 만 -- proto FpBar.agg 주석).
    pub(crate) agg: bool,
}

impl Bar {
    /// 칸 값을 원본 dict 차례로(server.py _flow_read_ctx 의 `fbars[b].values()` 합 차례): 생긴 차례 → 나머지 키 차례.
    // ponytail: 복원(테이프·스냅숏)으로 생긴 칸은 키 차례 -- 원본도 재기동 뒤엔 자기 복원 차례라 그 봉은 마지막 비트가 갈릴 수 있다.
    pub(crate) fn in_order(&self) -> Vec<[f64; 6]> {
        let mut out: Vec<[f64; 6]> = self.order.iter().filter_map(|k| self.cells.get(k).copied()).collect();
        if out.len() < self.cells.len() {
            let seen: BTreeSet<i64> = self.order.iter().copied().collect();
            out.extend(self.cells.iter().filter(|(k, _)| !seen.contains(k)).map(|(_, c)| *c));
        }
        out
    }
}

pub struct Bars {
    pub(crate) spec: Option<(f64, usize)>,
    pub(crate) bars: BTreeMap<i64, Bar>,
    /// 마지막 밀기 뒤 바뀐 칸(봉 → 가격 키). 봉만 있고 키가 비면 dhl 만 바뀐 것.
    pub(crate) dirty: BTreeMap<i64, BTreeSet<i64>>,
    /// 창 안에 있다가 시각 정리로 지워진 봉(밀기 `drop`).
    dropped: Vec<i64>,
    /// 복원이 못 메운 초 구간 [시작, 끝) -- 걸친 봉이 메시지 `holes` 로 나간다(restore.rs). 스냅숏 파일에도 실린다.
    pub(crate) holes: Vec<(i64, i64)>,
    /// 지금 테이프에서 메우는 중인 구간(메우기가 끝나면 지우고, 못 메운 몫만 holes 로 남긴다).
    pub(crate) pending: Option<(i64, i64)>,
    /// holes·pending 이 바뀌었다 -- 다음 밀기가 봉이 안 바뀌었어도 나간다.
    pub(crate) holes_dirty: bool,
    /// 테이프 복원 봉에 agg 를 붙이나(원본이 REST aggTrades 로 메우는 종목 = ethusdt -- coin_flow.py 534 «ETH 외 코인» 은 테이프).
    pub(crate) agg_on_restore: bool,
    /// 테이프로 dhl 을 한 번 정한 봉(원본 fp_dhl_cache 3947 · 3983~3985 -- 값이 없어도 «정함»이면 다시 안 읽는다).
    pub(crate) dhl_tried: BTreeSet<i64>,
    /// 1분봉(머리 주석) · 이 초부터 «다 센» 봉 · 마지막 밀기 뒤 바뀐 칸 · 지운 봉 · m1_from 이 바뀌었다.
    pub(crate) m1: BTreeMap<i64, Bar1>,
    pub(crate) m1_from: i64,
    m1_dirty: BTreeMap<i64, BTreeSet<i64>>,
    m1_dropped: Vec<i64>,
}

/// 1초 링이 다 드는 가장 이른 초(가장 새 초 − 660). 링은 새 초 s 가 올 때 s − 660 밑만 지운다(supply.rs sec_cell).
pub fn ring_floor(sec: &BTreeMap<i64, [f64; 7]>) -> Option<i64> {
    sec.keys().next_back().map(|t| t - SUPPLY_1S_SECONDS)
}

/// server.py bar_delta_hl: 봉 시작(0)도 최저·최고 후보. min/max 는 파이썬처럼 «앞의 것이 이긴다»(-0.0 이 갈리지 않게).
fn delta_hl(deltas: impl Iterator<Item = f64>) -> [f64; 3] {
    let (mut c, mut lo, mut hi) = (0.0f64, 0.0f64, 0.0f64);
    for d in deltas {
        c += d;
        if c < lo {
            lo = c;
        }
        if c > hi {
            hi = c;
        }
    }
    [r3(lo), r3(hi), r3(c)]
}

/// fp_dhl 의 링 가지 한 봉(server.py 3958~3965). 마감 델타가 봉 칸 델타와 어긋나면 None.
// ponytail: 칸 델타 합은 키 순서로 더한다(파이썬은 칸이 생긴 순서) -- 허용폭(≥0.5) 경계에 1e-12 안으로 붙는 경우만 갈린다.
pub(crate) fn dhl_of(sec: &BTreeMap<i64, [f64; 7]>, b: i64, bar: &Bar) -> Option<[f64; 3]> {
    dhl_ok(delta_hl(sec.range(b..b + BAR_SECONDS).map(|(_, c)| c[4] - c[5])), bar)
}

/// server.py fp_dhl `ok`: 마감 델타가 봉 칸 델타와 max(0.5, 1%) 안이어야 낸다(링·테이프 어느 쪽에서 쟀든 같은 검사).
pub(crate) fn dhl_ok(v: [f64; 3], bar: &Bar) -> Option<[f64; 3]> {
    let cd: f64 = bar.cells.values().map(|c| c[0] - c[1]).sum();
    ((v[2] - cd).abs() <= f64::max(0.5, 0.01 * cd.abs())).then_some(v)
}

impl Bars {
    pub fn new(symbol: &str) -> Bars {
        Bars { spec: spec(symbol), bars: BTreeMap::new(), dirty: BTreeMap::new(), dropped: vec![], holes: vec![], pending: None, holes_dirty: false,
               agg_on_restore: symbol.eq_ignore_ascii_case("ethusdt"), dhl_tried: BTreeSet::new(),
               m1: BTreeMap::new(), m1_from: 0, m1_dirty: BTreeMap::new(), m1_dropped: vec![] }
    }

    /// 체결의 봉을 들고 있어도 되나(coin_flow.py 355~365). 새 봉이면 만들고 창 밖 봉을 지운다.
    pub fn keep(&mut self, bar: i64, now_ms: i64) -> bool {
        if self.bars.contains_key(&bar) {
            return true;
        }
        let cutoff = bar_of(now_ms) - (KEEP_BARS - 1) * BAR_SECONDS;
        if bar < cutoff {
            return false; // 창을 벗어난 봉 -- 넣어봐야 바로 지워진다
        }
        let win_lo = self.window_lo();
        self.bars.insert(bar, Bar::default());
        let kept = self.bars.split_off(&cutoff); // 새 봉이 생길 때만 정리(체결마다 돌 일이 아니다)
        for (&b, _) in &self.bars {
            self.dirty.remove(&b);
            if win_lo.is_some_and(|lo| b >= lo) {
                self.dropped.push(b); // 봉이 드문 종목(주식 선물 주말 등)은 시각으로 지운 봉이 화면 창 안에 있을 수 있다
            }
        }
        self.bars = kept;
        true
    }

    /// footprint_add 의 봉 칸 몫(coin_flow.py 366~380). `keep` 이 참일 때만 부른다.
    #[allow(clippy::too_many_arguments)]
    pub fn add(&mut self, bar: i64, px: f64, qty: f64, sell: bool, order: bool, whale_min: f64, retail_max: f64) {
        let Some((bucket, _)) = self.spec else { return };
        let k = (px / bucket).round_ties_even() as i64;
        let b = self.bars.get_mut(&bar).expect("keep 이 만들었다");
        if !b.cells.contains_key(&k) {
            b.order.push(k);
        }
        let cell = b.cells.entry(k).or_insert([0.0; 6]);
        let side = sell as usize;
        if !order {
            cell[side] += qty;
        } else {
            let notional = px * qty;
            if notional >= whale_min {
                cell[2 + side] += qty;
            } else if notional < retail_max {
                cell[4 + side] += qty;
            }
        }
        self.dirty.entry(bar).or_default().insert(k);
    }

    /// footprint_add 의 1분봉 몫(coin_flow.py 374~408). `add` 와 같은 인자로 바로 뒤에 부른다(5분봉 창을 통과한 체결만).
    #[allow(clippy::too_many_arguments)]
    pub fn add_1m(&mut self, ts_ms: i64, px: f64, qty: f64, sell: bool, order: bool, whale_min: f64, retail_max: f64) {
        let Some((bucket, _)) = self.spec else { return };
        let k = (px / bucket).round_ties_even() as i64;
        let m = ts_ms.div_euclid(1000).div_euclid(M1_SECONDS) * M1_SECONDS;
        if !self.m1.contains_key(&m) {
            self.m1.insert(m, Bar1::default());
            let kept = self.m1.split_off(&(m - (M1_KEEP - 1) * M1_SECONDS)); // 새 분이 생길 때만 정리
            for b in std::mem::replace(&mut self.m1, kept).into_keys() {
                self.m1_dirty.remove(&b);
                self.m1_dropped.push(b);
            }
        }
        let b = self.m1.get_mut(&m).expect("방금 넣었다");
        let cell = b.cells.entry(k).or_insert([0.0; 6]);
        let side = sell as usize;
        if !order {
            cell[side] += qty;
            match &mut b.ohlc {
                None => b.ohlc = Some(([px; 4], ts_ms, ts_ms)),
                Some((o, first, last)) => {
                    o[1] = o[1].max(px);
                    o[2] = o[2].min(px);
                    if ts_ms < *first {
                        (o[0], *first) = (px, ts_ms);
                    }
                    if ts_ms >= *last {
                        (o[3], *last) = (px, ts_ms);
                    }
                }
            }
        } else {
            let notional = px * qty;
            if notional >= whale_min {
                cell[2 + side] += qty;
            } else if notional < retail_max {
                cell[4 + side] += qty;
            }
        }
        self.m1_dirty.entry(m).or_default().insert(k);
    }

    /// 1분봉 메시지(스냅숏 = 전부 · 밀기 = 바뀐 봉의 바뀐 칸 + 그 봉 OHLC). 표에 없는 종목은 None.
    fn msg_1m(&self, full: bool, dirty: &BTreeMap<i64, BTreeSet<i64>>, drop: Vec<i64>) -> Option<Fp1m> {
        let (_, dp) = self.spec?;
        let bar = |t: i64, b: &Bar1, ks: Option<&BTreeSet<i64>>| Fp1mBar { time: t, ohlc: b.ohlc.map(|o| o.0.map(|x| rdp(x, dp))), levels: self.rows(&b.cells, ks) };
        let bars = if full {
            self.m1.iter().map(|(&t, b)| bar(t, b, None)).collect()
        } else {
            dirty.iter().filter_map(|(t, ks)| self.m1.get(t).map(|b| bar(*t, b, Some(ks)))).collect()
        };
        Some(Fp1m { full, m1_from: self.m1_from, bars, drop })
    }

    pub fn snapshot_1m(&self) -> Option<Fp1m> {
        self.msg_1m(true, &BTreeMap::new(), vec![])
    }

    pub fn take_delta_1m(&mut self) -> Option<Fp1m> {
        if self.m1_dirty.is_empty() && self.m1_dropped.is_empty() {
            return None;
        }
        let dirty = std::mem::take(&mut self.m1_dirty);
        let drop = std::mem::take(&mut self.m1_dropped);
        self.msg_1m(false, &dirty, drop)
    }

    /// 1초 링이 [lo, hi) 에서 시작하는 봉들의 첫 초를 곧 지운다 -- 지우기 직전 링으로 dhl 을 얼린다(머리 주석).
    pub fn freeze(&mut self, sec: &BTreeMap<i64, [f64; 7]>, lo: i64, hi: i64) {
        for (&b, bar) in self.bars.range_mut(lo..hi) {
            let v = dhl_of(sec, b, bar);
            if v != bar.dhl {
                bar.dhl = v;
                self.dirty.entry(b).or_default();
            }
        }
    }

    /// 링 밖 봉에 체결이 더 왔다 -- 얼린 값은 이제 틀리다(테이프 없이는 다시 못 잰다).
    pub fn forget_dhl(&mut self, b: i64) {
        if let Some(bar) = self.bars.get_mut(&b) {
            bar.dhl = None;
        }
    }

    /// 화면 창(가장 새 144봉)의 첫 봉. 봉이 144 보다 적으면 가장 오래된 봉.
    pub(crate) fn window_lo(&self) -> Option<i64> {
        self.bars.keys().rev().take(WINDOW_BARS).last().copied()
    }

    fn levels(&self, bar: &Bar, keys: Option<&BTreeSet<i64>>) -> Vec<Level> {
        self.rows(&bar.cells, keys)
    }

    fn rows(&self, cells: &BTreeMap<i64, [f64; 6]>, keys: Option<&BTreeSet<i64>>) -> Vec<Level> {
        let (bucket, dp) = self.spec.expect("spec 있을 때만 부른다");
        let row = |k: i64, c: &[f64; 6]| {
            let px = rdp(k as f64 * bucket, dp); // 파이썬 round(k * 칸 폭, dp) -- supply.rs r3 주석
            (px, r3(c[0]), r3(c[1]), r3(c[2]), r3(c[3]), r3(c[4]), r3(c[5]))
        };
        match keys {
            None => cells.iter().map(|(&k, c)| row(k, c)).collect(),
            Some(ks) => ks.iter().filter_map(|k| cells.get(k).map(|c| row(*k, c))).collect(),
        }
    }

    /// 지금 보이는 dhl: 링이 봉을 다 들면 링에서 재고, 아니면 얼린 값(스냅숏·코어 스냅숏 파일이 같은 값을 쓴다).
    pub(crate) fn dhl_now(&self, sec: &BTreeMap<i64, [f64; 7]>, b: i64, bar: &Bar) -> Option<[f64; 3]> {
        if ring_floor(sec).is_some_and(|f| f <= b) { dhl_of(sec, b, bar) } else { bar.dhl }
    }

    /// 창 안 봉 중 holes·pending 에 걸친 봉(봉 [b, b+300) 와 구간 [a, z) 가 겹침).
    fn holey(&self) -> Vec<i64> {
        let hs: Vec<(i64, i64)> = self.holes.iter().copied().chain(self.pending).collect();
        if hs.is_empty() {
            return vec![];
        }
        let lo = self.window_lo().unwrap_or(i64::MIN);
        self.bars.range(lo..).map(|(&b, _)| b).filter(|&b| hs.iter().any(|&(a, z)| b < z && b + BAR_SECONDS > a)).collect()
    }

    fn head(&self, full: bool, retail_max: f64, whale_min: f64, bars: Vec<FpBar>) -> Footprint {
        let n = self.bars.len().min(WINDOW_BARS);
        Footprint {
            holes: self.holey(),
            full,
            bucket: self.spec.map_or(0.0, |s| s.0),
            bar_seconds: BAR_SECONDS,
            bars_expected: WINDOW_BARS,
            ready: n == WINDOW_BARS,
            retail_max_usd: retail_max,
            whale_min_usd: whale_min,
            bars,
            drop: vec![],
        }
    }

    /// 구독 직후 1회 = 파이썬 `/api/footprint?bars=144&since=0` 의 봉·칸. 표에 없는 종목은 None.
    pub fn snapshot(&self, sec: &BTreeMap<i64, [f64; 7]>, retail_max: f64, whale_min: f64) -> Option<Footprint> {
        self.spec?;
        let lo = self.window_lo().unwrap_or(i64::MIN);
        let bars = self.bars.range(lo..).map(|(&b, bar)| FpBar { time: b, levels: self.levels(bar, None), dhl: self.dhl_now(sec, b, bar), agg: bar.agg }).collect();
        Some(self.head(true, retail_max, whale_min, bars))
    }

    /// 마지막 밀기 뒤 바뀐 것: 바뀐 봉의 바뀐 칸 + 그 봉의 dhl(실린 봉은 dhl 을 통째로 갈아 끼운다 -- 없으면 없음) + 지운 봉.
    /// 받은 쪽은 칸을 가격으로 덮어쓰고, 봉 수가 144 를 넘으면 오래된 쪽을 버린다(app.js refreshFootprint 와 같은 규칙).
    pub fn take_delta(&mut self, sec: &BTreeMap<i64, [f64; 7]>, retail_max: f64, whale_min: f64) -> Option<Footprint> {
        self.spec?;
        // 링이 다 드는 봉(많아야 3봉)은 매번 다시 잰다 -- 스냅숏과 같은 식이라 받은 쪽이 스냅숏과 같아진다.
        if let Some(f) = ring_floor(sec) {
            for (&b, bar) in self.bars.range_mut(f..) {
                let v = dhl_of(sec, b, bar);
                if v != bar.dhl {
                    bar.dhl = v;
                    self.dirty.entry(b).or_default();
                }
            }
        }
        if self.dirty.is_empty() && self.dropped.is_empty() && !self.holes_dirty {
            return None;
        }
        self.holes_dirty = false;
        let lo = self.window_lo().unwrap_or(i64::MIN);
        let dirty = std::mem::take(&mut self.dirty);
        let bars = dirty
            .range(lo..)
            .filter_map(|(b, ks)| self.bars.get(b).map(|bar| FpBar { time: *b, levels: self.levels(bar, Some(ks)), dhl: bar.dhl, agg: bar.agg }))
            .collect();
        let mut m = self.head(false, retail_max, whale_min, bars);
        m.drop = std::mem::take(&mut self.dropped);
        Some(m)
    }

    /// 원본 fp_dhl 의 테이프 가지(server.py 3957~3967): 창 안 봉 중 1초 링이 안 덮고(링 바닥 앞) · dhl 이 없고 · 아직 안 정했고 ·
    /// 닫힌 지 6분(360초) 지난 봉 → 테이프에서 읽을 봉. 원본은 /api/footprint 요청마다 · 코어는 주기 일감(lib.rs)이 부른다.
    pub fn dhl_need(&self, sec: &BTreeMap<i64, [f64; 7]>, now_s: f64) -> Vec<i64> {
        if self.spec.is_none() {
            return vec![];
        }
        let floor = ring_floor(sec).unwrap_or(i64::MAX);
        let lo = self.window_lo().unwrap_or(i64::MIN);
        self.bars
            .range(lo..)
            .filter(|(&b, bar)| b < floor && bar.dhl.is_none() && !self.dhl_tried.contains(&b) && (b + BAR_SECONDS) as f64 <= now_s - 360.0)
            .map(|(&b, _)| b)
            .collect()
    }

    /// 읽은 테이프 dhl 을 얹는다(원본 3983~3987): 읽기 성공이면 «그 봉 값이 있거나 닫힌 지 15분 지남» 일 때 정함으로 적고 ·
    /// 값이 칸 델타와 맞으면(dhl_ok) 그 봉 dhl 로. 읽기 실패(None)면 아무것도 안 적는다(다음 회차에 다시).
    pub fn dhl_apply(&mut self, need: &[i64], got: Option<&BTreeMap<i64, [f64; 3]>>, now_s: f64) -> usize {
        let Some(got) = got else { return 0 };
        let mut n = 0;
        for &b in need {
            if got.contains_key(&b) || (b + BAR_SECONDS) as f64 <= now_s - 900.0 {
                self.dhl_tried.insert(b);
            }
            let Some(bar) = self.bars.get_mut(&b) else { continue };
            if bar.dhl.is_none() {
                if let Some(v) = got.get(&b).and_then(|v| dhl_ok(*v, bar)) {
                    bar.dhl = Some(v);
                    self.dirty.entry(b).or_default();
                    n += 1;
                }
            }
        }
        let keep: BTreeSet<i64> = self.bars.keys().copied().collect();
        self.dhl_tried.retain(|b| keep.contains(b));
        n
    }

    /// 시험용: 봉 → 칸(반올림 전).
    pub fn raw(&self) -> Vec<(i64, Vec<(i64, [f64; 6])>)> {
        self.bars.iter().map(|(&b, bar)| (b, bar.cells.iter().map(|(&k, c)| (k, *c)).collect())).collect()
    }
}

#[cfg(test)]
mod dhl_retry {
    use super::*;

    /// 원본 fp_dhl 테이프 가지의 고를 봉·정함 규칙(server.py 3957~3987).
    #[test]
    fn need_and_apply_follow_python() {
        let mut fp = Bars::new("solusdt");
        let now = 10_000.0;
        for b in [6000, 6300, 9000, 9300, 9600] {
            fp.bars.insert(b, Bar::default());
        }
        fp.bars.get_mut(&6000).unwrap().cells.insert(1, [5.0, 2.0, 0.0, 0.0, 0.0, 0.0]); // 칸 델타 3
        fp.bars.get_mut(&6300).unwrap().dhl = Some([0.0, 1.0, 1.0]); // 이미 있음 → 안 고른다
        // 1초 링이 9300 부터 덮는다 → 9300·9600 은 링 몫 · 9000 봉(끝 9300)은 지금 10000 에서 닫힌 지 6분 넘음
        let mut sec = BTreeMap::new();
        sec.insert(9300 + 660, [0.0; 7]);
        assert_eq!(fp.dhl_need(&sec, now), vec![6000, 9000]);
        assert_eq!(fp.dhl_need(&sec, 9500.0), vec![6000], "9000 봉은 닫힌 지 6분 전");
        // 읽기 실패 = 아무것도 안 적는다
        assert_eq!(fp.dhl_apply(&[6000, 9000], None, now), 0);
        assert_eq!(fp.dhl_need(&sec, now), vec![6000, 9000]);
        // 6000 = 값 있음(맞음) · 9000 = 값 없음 · 닫힌 지 15분 안 → 정함 아님(다음에 다시)
        let got: BTreeMap<i64, [f64; 3]> = [(6000, [-1.0, 4.0, 3.0])].into_iter().collect();
        assert_eq!(fp.dhl_apply(&[6000, 9000], Some(&got), now), 1);
        assert_eq!(fp.bars[&6000].dhl, Some([-1.0, 4.0, 3.0]));
        assert_eq!(fp.dhl_need(&sec, now), vec![9000]);
        // 15분 지나면 값이 없어도 정함
        fp.dhl_apply(&[9000], Some(&BTreeMap::new()), 9300.0 + 900.0);
        assert!(fp.dhl_need(&sec, 9300.0 + 900.0).is_empty());
        // 칸 델타와 안 맞는 값은 안 얹는다(원본 ok)
        let mut fp2 = Bars::new("solusdt");
        fp2.bars.insert(6000, Bar::default());
        fp2.bars.get_mut(&6000).unwrap().cells.insert(1, [5.0, 2.0, 0.0, 0.0, 0.0, 0.0]);
        let bad: BTreeMap<i64, [f64; 3]> = [(6000, [-1.0, 9.0, 9.0])].into_iter().collect();
        assert_eq!(fp2.dhl_apply(&[6000], Some(&bad), now), 0);
        assert!(fp2.bars[&6000].dhl.is_none());
    }
}

#[cfg(test)]
mod m1_restore {
    use crate::supply::Supply;

    /// 1분봉 저장·복원(coin_flow.py footprint_save 441~443 · footprint_load 472~479): 파일 왕복 == 쓴 상태 · 지금 − 119분 앞은 버림 ·
    /// 옛 파일(1분봉 칸 없음)은 m1_from = last_ms 뒤 첫 분.
    #[test]
    fn roundtrip() {
        let mut a = Supply::new("ethusdt");
        let t0 = 1_763_700_000_000i64; // 분 경계
        for i in 0..20_000i64 {
            let ms = t0 + i * 997; // 약 5.5시간
            a.trade(3000.0 + (i % 17) as f64 * 0.25, 0.5 + (i % 7) as f64, ms, i % 3 == 0, i + 1, ms * 1000);
        }
        a.rs.merged = true;
        let snap = a.to_snap("ethusdt").unwrap();
        assert_eq!(snap.m1.as_ref().unwrap().bars.len(), 120);
        let dir = std::env::temp_dir().join(format!("rtcore_m1_{}", std::process::id()));
        let p = dir.join("fp.json");
        crate::restore::write(&p, &snap).unwrap();
        let now = a.last_ms + 30 * 60_000; // 30분 뒤 재시작 -- 앞 30분어치(30봉)는 보관 밖
        let mut b = Supply::new("ethusdt");
        b.load_snap(crate::restore::load(&p, "ethusdt", now).unwrap());
        let cut = now.div_euclid(60_000) * 60 - 119 * 60;
        let want: Vec<_> = a.fp.m1.iter().filter(|(t, _)| **t >= cut).map(|(t, v)| (*t, v.clone())).collect();
        assert_eq!(b.fp.m1.iter().map(|(t, v)| (*t, v.clone())).collect::<Vec<_>>(), want);
        assert_eq!((b.fp.m1.len(), b.fp.m1_from), (90, 0));
        // 옛 파일
        let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
        v.as_object_mut().unwrap().remove("m1");
        let mut c = Supply::new("ethusdt");
        c.load_snap(serde_json::from_value(v).unwrap());
        assert!(c.fp.m1.is_empty());
        assert_eq!(c.fp.m1_from, (a.last_ms + 59_999) / 60_000 * 60);
        assert!(c.fp.m1_from * 1000 > a.last_ms && c.fp.m1_from % 60 == 0);
        let _ = std::fs::remove_dir_all(dir);
    }
}
