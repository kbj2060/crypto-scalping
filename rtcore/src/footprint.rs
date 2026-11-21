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
use crate::proto::{FpBar, Footprint, Level};
use crate::supply::{r3, SUPPLY_1S_SECONDS};
use std::collections::{BTreeMap, BTreeSet};

pub const BAR_SECONDS: i64 = 300;
pub const KEEP_BARS: i64 = 288;
/// 스냅숏·밀기 창(봉 수). server.py FOOTPRINT_MAX_WINDOW_BARS = app.js CHART_PAN_BARS.
pub const WINDOW_BARS: usize = 144;

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
        Bars { spec: spec(symbol), bars: BTreeMap::new(), dirty: BTreeMap::new(), dropped: vec![], holes: vec![], pending: None, holes_dirty: false }
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
        let (bucket, dp) = self.spec.expect("spec 있을 때만 부른다");
        let row = |k: i64, c: &[f64; 6]| {
            let px = format!("{:.*}", dp, k as f64 * bucket).parse().unwrap_or(0.0); // 파이썬 round(k * 칸 폭, dp) -- supply.rs r3 주석
            (px, r3(c[0]), r3(c[1]), r3(c[2]), r3(c[3]), r3(c[4]), r3(c[5]))
        };
        match keys {
            None => bar.cells.iter().map(|(&k, c)| row(k, c)).collect(),
            Some(ks) => ks.iter().filter_map(|k| bar.cells.get(k).map(|c| row(*k, c))).collect(),
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
        let bars = self.bars.range(lo..).map(|(&b, bar)| FpBar { time: b, levels: self.levels(bar, None), dhl: self.dhl_now(sec, b, bar) }).collect();
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
            .filter_map(|(b, ks)| self.bars.get(b).map(|bar| FpBar { time: *b, levels: self.levels(bar, Some(ks)), dhl: bar.dhl }))
            .collect();
        let mut m = self.head(false, retail_max, whale_min, bars);
        m.drop = std::mem::take(&mut self.dropped);
        Some(m)
    }

    /// 시험용: 봉 → 칸(반올림 전).
    pub fn raw(&self) -> Vec<(i64, Vec<(i64, [f64; 6])>)> {
        self.bars.iter().map(|(&b, bar)| (b, bar.cells.iter().map(|(&k, c)| (k, *c)).collect())).collect()
    }
}
