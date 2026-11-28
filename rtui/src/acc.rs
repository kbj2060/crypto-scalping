//! Account 카드(?card=acc) -- app.js 계좌 구역: renderSnapshotAccountCore(2734) · acctPools(2359) · acctWealthHtml(2375) · ACCT_*·acct* 글(2459~2477) · acctTabApply(2582) ·
//!   acctTodaySvg(2598) · acctCalendar(2621) · acctCurveSvg(2649) · acctStatGroups·acctStatsHtml(2695) · cycleSvg(2855) · renderCycleCard(2903).
//! 자료 = 원본과 같은 주소·주기: /api/binance-account 10초(숨으면 안 묻는다 · ok 아니면 서버 last_good 또는 30분 안 마지막 정상값으로 그린다 -- 2721) ·
//!   /api/cycle-path?symbol=&start= (보이는 순환만 · 60초 · 2924).
//! 자리 = 원본 styles.css(html.fit1 · 1920 화면 #acctCard 1872 폭 · 머리 아래 934)를 헤드리스 크로미움에서 잰 값과 같게 손으로 놓은 것(compare/acc_layout.py → 시험 js_parity).
//! 누르기(보기만 -- 원본과 같은 상태): 곡선 기간 칩(오늘·1주·1개월·전체) · 아래 탭(순환·성과 통계·달력·복기) · 순환 탭(열린 포지션이 둘 이상) · 달력 달 넘김 ‹ › · 날 고르기.
//! 안 옮긴 것: 카드 머리(제목 · 순자산 줄 · 요약 알약 · «거래소 실계좌» · 강제 조회 단추) · 메모 칸(/api/acct-notes)·오답노트(/api/acct-wrong) = 입력이라 ⛔(오답노트 탭은 모양만 · 누르지 않는다 · 메모 자리는 비움) ·
//!   툴팁(title) · 호버 테 · 초점 테 · 말줄임 아닌 줄바꿈(풀 이름·경고 줄이 넘치면 한 줄로 넘침) · 좁은 화면(900 이하) · 기간 칩 기억(localStorage).
//! ponytail: 상태는 이 파일의 thread_local(웹어셈블리 = 한 스레드 · news.rs 와 같은 꼴) -- 공용 Feed 를 안 건드리려고.
//! ponytail: svg 칠 경로(곡선 밑 면 · 낙폭 면)는 «곡선 → 바닥선 → 처음» 꼴만 안다(fp::Prim::Area) -- 원본 그림이 다른 꼴을 쓰면 그때 다각형 칠을 더한다.
use crate::fp::{tnum, Anchor, Clip, Col, Prim};
use crate::mt::{asc, cont, fixed, rr_pts, Measure};
use crate::srl::truthy;
use serde_json::Value;
use std::cell::RefCell;

// ── JS 수·글 ──
fn f(v: &Value) -> f64 { match v { Value::Number(n) => n.as_f64().unwrap_or(f64::NAN), Value::String(s) => s.trim().parse().unwrap_or(f64::NAN), Value::Bool(b) => *b as u8 as f64, Value::Null => 0.0, _ => f64::NAN } }
/// Number(v) || 0
fn n0(v: &Value) -> f64 { let x = f(v); if x.is_nan() { 0.0 } else { x } }
fn s(v: &Value) -> String { match v { Value::String(s) => s.clone(), Value::Null => String::new(), v => v.to_string() } }
fn jround(x: f64) -> f64 { (x + 0.5).floor() }
fn fx(x: f64, d: usize) -> f64 { fixed(x, d).parse().unwrap_or(f64::NAN) }
/// toLocaleString("en-US", { minimumFractionDigits: lo, maximumFractionDigits: hi }) -- 쉼표 · 0.5 는 0 에서 먼 쪽
fn loc(x: f64, lo: usize, hi: usize) -> String {
    if !x.is_finite() { return if x.is_nan() { "NaN".into() } else if x > 0.0 { "∞".into() } else { "-∞".into() }; }
    let t = fixed(x.abs(), hi);
    let (ip, fp) = t.split_once('.').map_or((t.as_str(), ""), |(a, b)| (a, b));
    let mut fp = fp.to_string();
    while fp.len() > lo && fp.ends_with('0') { fp.pop(); }
    let mut o = String::new();
    for (k, c) in ip.chars().enumerate() { if k > 0 && (ip.len() - k) % 3 == 0 { o.push(','); } o.push(c); }
    if !fp.is_empty() { o = format!("{o}.{fp}"); }
    if x < 0.0 && (ip != "0" || fp.chars().any(|c| c != '0')) { format!("-{o}") } else { o }
}
/// JS Number → 글(템플릿 `${v}`)
fn jn(x: f64) -> String { if x == 0.0 { "0".into() } else { format!("{x}") } }
const DAY: f64 = 86_400_000.0;
const KST: f64 = 9.0 * 3_600_000.0;
fn civil(days: i64) -> (i64, i64, i64) {
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + if m <= 2 { 1 } else { 0 }, m, d)
}
fn days_civil(y: i64, m: i64, d: i64) -> i64 { let y = if m <= 2 { y - 1 } else { y }; let era = y.div_euclid(400); let yoe = y - era * 400; let doy = (153 * ((m + 9) % 12) + 2) / 5 + d - 1; era * 146097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719468 }
fn kday(ms: f64) -> i64 { ((ms + KST) / DAY).floor() as i64 }
/// acctKey(KST 날짜 «YYYY-MM-DD»)
fn key(ms: f64) -> String { let (y, m, d) = civil(kday(ms)); format!("{y:04}-{m:02}-{d:02}") }
fn ymd(k: &str) -> (i64, i64, i64) { (k[0..4].parse().unwrap_or(1970), k.get(5..7).and_then(|x| x.parse().ok()).unwrap_or(1), k.get(8..10).and_then(|x| x.parse().ok()).unwrap_or(1)) }
/// acctDay0(키) = 그날 KST 0시(ms)
fn day0(k: &str) -> f64 { let (y, m, d) = ymd(k); days_civil(y, m, d) as f64 * DAY - KST }
fn hm(ms: f64) -> String { let m = (((ms + KST).floor() / 60000.0).floor() as i64).rem_euclid(1440); format!("{:02}:{:02}", m / 60, m % 60) }
fn md(ms: f64) -> String { let (_, m, d) = civil(kday(ms)); format!("{m}/{d}") }
fn su(v: f64) -> String { format!("{}${}", if v >= 0.0 { "+" } else { "−" }, loc(v.abs(), 0, if v.abs() < 100.0 { 2 } else { 0 })) }
fn suc(v: f64) -> String { let a = v.abs(); format!("{}${}", if v >= 0.0 { "+" } else { "−" }, if a >= 1000.0 { format!("{}k", fixed(a / 1000.0, 1)) } else { fixed(a, 0) }) }
#[derive(Clone, Copy, PartialEq)]
enum Tone { Good, Bad, Neutral }
fn tone(v: f64) -> Tone { if v > 0.0 { Tone::Good } else if v < 0.0 { Tone::Bad } else { Tone::Neutral } }
/// 색 칸(.good·.bad 는 칠하고 .neutral 은 물려받음)
fn tc(t: Tone, inherit: Col) -> Col { match t { Tone::Good => Col::Good, Tone::Bad => Col::Bad, Tone::Neutral => inherit } }
fn pct(p: Option<f64>) -> String {
    match p.filter(|p| p.is_finite()) { None => "—".into(), Some(p) => { let a = (p * 100.0).abs(); format!("{}{}%", if p >= 0.0 { "+" } else { "−" }, if a < 10.0 { fixed(a, 1) } else { jn(jround(a)) }) } }
}
fn dur(ms: f64) -> String { let m = jround(ms / 60000.0); if m < 60.0 { format!("{}분", jn(m)) } else if m < 1440.0 { format!("{}시간 {:02}분", (m / 60.0).floor(), m.rem_euclid(60.0) as i64) } else { format!("{}일", fixed(m / 1440.0, 1)) } }
fn coin(sym: &str) -> String { sym.strip_suffix("USDT").or_else(|| sym.strip_suffix("USDC")).unwrap_or(sym).to_string() }
fn px(v: f64) -> String { let d = if v.abs() < 10.0 { 4 } else { 2 }; loc(v, d, d) }
fn usd(v: f64) -> String { if !v.is_finite() { return "-".into(); } format!("{}${}", if v >= 0.0 { "" } else { "-" }, loc(v.abs(), 0, if v.abs() < 10.0 { 4 } else { 2 })) }
fn cq(v: f64) -> String { loc(fx(v, 4), 0, 4) }
fn quote(sym: &str) -> &'static str { if sym.ends_with("USDC") { "USDC" } else { "USDT" } }

// ── 그리기 단위(시험이 원본 잰 값과 하나씩 맞댄다) ──
/// k = 자르기 상자(overflow 조상들의 겹침) [x, y, 폭, 높이]
pub type K = Option<[f64; 4]>;
#[derive(Debug, Clone)]
pub enum It {
    /// y = 글 상자 위 · tnum = font-variant-numeric: tabular-nums(Pretendard 숫자만)
    T { x: f64, y: f64, s: String, z: f64, wt: u16, mono: bool, tnum: bool, ls: f64, c: Col, a: f64, k: K },
    B { x: f64, y: f64, w: f64, h: f64, c: Col, a: f64, rad: f64, k: K },
    /// 네 변 테·윤곽·안쪽 고리(바깥 상자 · 두께)
    O { x: f64, y: f64, w: f64, h: f64, lw: f64, c: Col, a: f64, rad: f64, k: K },
    /// svg: 상자 · viewBox 폭·높이(meet -- 가운데 맞춰 고르게 줄임) · 요소(viewBox 단위)
    S { x: f64, y: f64, w: f64, h: f64, vb: [f64; 2], k: K, els: Vec<El> },
}
type Paint = Option<(Col, f64)>;
/// d 명령(M·H·V·L·Z + 수)
pub type D = Vec<(char, Vec<f64>)>;
#[derive(Debug, Clone, PartialEq)]
pub enum El {
    Line { p: [f64; 4], c: Col, a: f64, sw: f64, dash: Vec<f64> },
    Rect { r: [f64; 4], rx: f64, fill: Paint, stroke: Paint, sw: f64, dash: Vec<f64> },
    Path { d: D, fill: Paint, stroke: Paint, sw: f64, dash: Vec<f64> },
    Circle { p: [f64; 3], fill: Paint, stroke: Paint, sw: f64 },
    /// halo = paint-order stroke(차트 바탕 3px) · ts = 뒤에 붙는 tspan(글 · 굵기 · 색)
    Text { x: f64, y: f64, s: String, z: f64, wt: u16, mono: bool, anchor: Anchor, c: Col, a: f64, halo: bool, ts: Option<(String, u16, Col, f64)> },
}
fn tx(x: f64, y: f64, s: impl Into<String>, z: f64, c: Col, anchor: Anchor) -> El { El::Text { x, y, s: s.into(), z, wt: 400, mono: false, anchor, c, a: 1.0, halo: false, ts: None } }
fn line(p: [f64; 4], c: Col, a: f64) -> El { El::Line { p, c, a, sw: 1.0, dash: vec![] } }

#[derive(Clone, Copy)]
struct St { z: f64, wt: u16, mono: bool, tnum: bool, c: Col, a: f64, ls: f64 }
const fn st(z: f64, wt: u16, mono: bool, tnum: bool, c: Col) -> St { St { z, wt, mono, tnum, c, a: 1.0, ls: 0.0 } }
impl St { fn c(self, c: Col) -> St { St { c, ..self } } fn a(self, a: f64) -> St { St { a, ..self } } fn w(self, wt: u16) -> St { St { wt, ..self } } /** letter-spacing(em) */ fn ls(self, em: f64) -> St { St { ls: em * self.z, ..self } } }
struct Cx<'a> { m: Measure<'a>, v: Vec<It>, k: K }
impl Cx<'_> {
    fn w(&self, s: &str, t: St) -> f64 { (self.m)(&if t.tnum && !t.mono { tnum(s) } else { s.to_string() }, t.z, t.wt, t.mono) + t.ls * s.chars().count() as f64 }
    /// 글 상자 위 y · 폭을 돌려준다
    fn t(&mut self, x: f64, y: f64, s: impl Into<String>, t: St) -> f64 { let s = s.into(); let w = self.w(&s, t); self.v.push(It::T { x, y, s, z: t.z, wt: t.wt, mono: t.mono, tnum: t.tnum && !t.mono, ls: t.ls, c: t.c, a: t.a, k: self.k }); w }
    #[allow(clippy::too_many_arguments)]
    fn b(&mut self, x: f64, y: f64, w: f64, h: f64, c: Col, a: f64, rad: f64) { if w <= 0.0 || h <= 0.0 { return; } self.v.push(It::B { x, y, w, h, c, a, rad, k: self.k }); }
    #[allow(clippy::too_many_arguments)]
    fn o(&mut self, x: f64, y: f64, w: f64, h: f64, lw: f64, c: Col, a: f64, rad: f64) { self.v.push(It::O { x, y, w, h, lw, c, a, rad, k: self.k }); }
}
fn and(a: K, b: [f64; 4]) -> K {
    let Some(a) = a else { return Some(b) };
    let (x0, y0) = (a[0].max(b[0]), a[1].max(b[1]));
    Some([x0, y0, (a[0] + a[2]).min(b[0] + b[2]) - x0, (a[1] + a[3]).min(b[1] + b[3]) - y0])
}
/// 줄 상자 위 → 글 상자 위(반 행간 내림)
fn gt(top: f64, lh: f64, z: f64, mono: bool) -> f64 { top + ((lh - cont(z, mono)) / 2.0).floor() }
/// LayoutUnit(1/64 px) 내림
fn fl(x: f64) -> f64 { (x * 64.0).floor() / 64.0 }
/// 격자 fr 칸 시작 = 정확한 누적을 1/64 내림(크로미움 실측) -- (시작들, 끝)
fn tracks(x0: f64, total: f64, fr: &[f64], gap: f64) -> Vec<(f64, f64)> {
    let sum: f64 = fr.iter().sum();
    let unit = (total - gap * (fr.len() - 1) as f64) / sum;
    let mut acc = 0.0;
    let starts: Vec<f64> = fr.iter().map(|q| { let s0 = fl(x0 + acc); acc += q * unit + gap; s0 }).collect();
    starts.iter().enumerate().map(|(i, &s0)| (s0, if i + 1 < starts.len() { starts[i + 1] - gap - s0 } else { fl(fr[i] * unit) })).collect()
}

// ── 계좌 자료(원본 계산 그대로) ──
struct Pool { name: String, i: usize, wallet: f64, unreal: f64, equity: f64, avail: f64, used_pct: f64, n: usize, pos: String, share: f64 }
struct Pools { pools: Vec<Pool>, total: f64, wallet: f64 }
fn live_pos(a: &Value) -> Vec<&Value> { a["positions"].as_array().into_iter().flatten().filter(|p| n0(&p["qty"]).abs() > 0.0).collect() }
/// 2359 acctPools
fn pools(a: &Value) -> Pools {
    let raw: Vec<(String, &Value)> = if truthy(&a["multi_assets_margin"]) { vec![("통합".into(), &a["balance"])] } else {
        ["USDC", "USDT"].iter().filter_map(|q| {
            let b = &a["balances"][*q];
            let b = if truthy(b) { Some(b) } else { a["assets"].as_array().into_iter().flatten().find(|x| x["asset"] == *q) };
            b.filter(|b| truthy(b)).map(|b| (q.to_string(), b))
        }).collect()
    };
    let live = live_pos(a);
    let mut v: Vec<Pool> = raw.into_iter().enumerate().map(|(i, (name, b))| {
        let wallet = n0(&b["wallet"]);
        let unreal = n0(&b["unrealized"]);
        let equity = { let m = n0(&b["margin"]); if m != 0.0 { m } else { wallet + unreal } };
        let avail = n0(&b["available"]);
        let im = &b["initial_margin"];
        let used = if !im.is_null() && f(im).is_finite() { f(im) } else { (equity - avail).max(0.0) };
        let ps: Vec<&&Value> = live.iter().filter(|p| name == "통합" || s(&p["symbol"]).ends_with(&name)).collect();
        let pos = ps.iter().map(|p| format!("{} {}", coin(&s(&p["symbol"])), if p["side"] == "LONG" { "롱" } else { "숏" })).collect::<Vec<_>>().join(" · ");
        Pool { name, i, wallet, unreal, equity, avail, used_pct: if equity > 0.0 { used / equity * 100.0 } else { 0.0 }, n: ps.len(), pos, share: 0.0 }
    }).filter(|p| p.equity > 0.0 || p.n > 0).collect();
    let total: f64 = v.iter().map(|p| p.equity).sum();
    for p in &mut v { p.share = if total > 0.0 { p.equity / total * 100.0 } else { 0.0 }; }
    Pools { total, wallet: v.iter().map(|p| p.wallet).sum(), pools: v }
}
struct Trip<'a> { v: &'a Value, sym: String, exit: f64, entry: f64, pnl: f64 }
fn trip(v: &Value) -> Trip<'_> { Trip { v, sym: s(&v["symbol"]), exit: f(&v["exit_time"]), entry: f(&v["entry_time"]), pnl: n0(&v["net_pnl"]) } }
/// 2695 acctStatGroups 한 묶음
struct Grp { q: String, n: usize, wn: usize, ln: usize, tot: f64, aw: f64, al: f64, mdd: f64, win: f64 }
fn stat_groups(a: &[&Trip]) -> Vec<Grp> {
    let mk = |q: String, g: Vec<&&Trip>| {
        let w: Vec<f64> = g.iter().map(|t| t.pnl).filter(|v| *v > 0.0).collect();
        let l: Vec<f64> = g.iter().map(|t| t.pnl).filter(|v| *v <= 0.0).collect();
        let (gw, gl): (f64, f64) = (w.iter().sum(), l.iter().sum());
        let (mut c, mut pk, mut mdd) = (0.0f64, 0.0f64, 0.0f64);
        for t in &g { c += t.pnl; pk = pk.max(c); mdd = mdd.min(c - pk); }
        Grp { q, n: g.len(), wn: w.len(), ln: l.len(), tot: g.iter().map(|t| t.pnl).sum(), aw: if w.is_empty() { 0.0 } else { gw / w.len() as f64 }, al: if l.is_empty() { 0.0 } else { gl / l.len() as f64 }, mdd, win: jround(w.len() as f64 / g.len() as f64 * 100.0) }
    };
    let by: Vec<(&str, Vec<&&Trip>)> = ["USDC", "USDT"].iter().map(|q| (*q, a.iter().filter(|t| quote(&t.sym) == *q).collect::<Vec<_>>())).filter(|x| !x.1.is_empty()).collect();
    if by.len() > 1 { let mut v = vec![mk("합산".into(), a.iter().collect())]; v.extend(by.into_iter().map(|(q, g)| mk(q.into(), g))); v }
    else { vec![mk(by.first().map_or("합산", |x| x.0).into(), a.iter().collect())] }
}

// ── 입력 ──
/// 받아 둔 응답 · 보기 상태(원본 전역 acctSpanIdx · acctCalBack · acctSelDay · acctTab · cycleTab) · now = Date.now()(ms)
pub struct In<'a> {
    pub now: f64,
    /// latestBinanceAccount(ok 인 것 · 실패면 낡은 값) -- None = «불러오지 못함»
    pub acct: Option<&'a Value>,
    pub err: &'a str,
    pub span: usize, pub back: i64, pub sel: Option<&'a str>, pub tab: &'a str, pub cytab: &'a str,
    /// /api/cycle-path rows («심볼|쪽|첫 체결ms» → [[ms, 종가]..])
    pub paths: &'a [(String, Value)],
}
/// 누르기 자리
#[derive(Debug, Clone, PartialEq)]
pub enum Hit { Span(usize), Month(i64), Day(String), Tab(&'static str), Cy(String) }
#[derive(Default)]
pub struct Out { pub items: Vec<It>, pub hits: Vec<([f64; 4], Hit)>, /** 받아야 할 순환 가격선(키 · 심볼 · 시작ms) */ pub want: Vec<(String, String, f64)> }

const SPANS: [(&str, f64); 4] = [("오늘", 0.0), ("1주", 7.0), ("1개월", 30.0), ("전체", -1.0)];
const CHIP: St = st(11.5, 700, true, false, Col::Muted);

/// 칩 하나(.chipset .chip · 24 높이 · 안쪽 9 · 테 1) → 폭
fn chip(cx: &mut Cx, x: f64, y: f64, parts: &[&str], on: bool) -> f64 {
    let t = CHIP.c(if on { Col::Amber } else { Col::Muted });
    let tw: f64 = parts.iter().map(|p| cx.w(p, t)).sum();
    let w = tw + 20.0;
    cx.b(x, y, w, 24.0, if on { Col::Amber } else { Col::White }, if on { 0.13 } else { 0.05 }, 7.0);
    if on { cx.o(x, y, w, 24.0, 1.0, Col::Amber, 0.45, 7.0); }
    let mut xx = x + 10.0;
    for p in parts { if !p.trim().is_empty() { cx.t(xx, y + 4.25, *p, t); } xx += cx.w(p, t); }
    w
}

/// renderSnapshotAccountCore → 머리 아래 본문(폭 w · 높이 h) 그림
pub fn body(i: &In, w: f64, h: f64, m: Measure) -> Out {
    let mut out = Out::default();
    let mut cx = Cx { m, v: vec![], k: None };
    // 틀(styles.css html.fit1): .acct-mid(안쪽 12 18 0 · 최소 300 · flex .75) · .acct-bot(바깥 12 18 18 · flex 1.5) · 카드 아래 안쪽 16 + 테 1
    let mid_h = ((h - 17.0 - 30.0) * 0.75 / 2.25).max(300.0);
    let (mx, mw) = (17.0, w - 34.0);
    let inner = mw - 36.0 - 12.0;
    let cw = fl(inner * 1.9 / 2.9);
    let (curve_x, curve_y, curve_w, curve_h) = (mx + 18.0 + 14.0, 24.0, cw - 28.0, mid_h - 12.0 - 24.0);
    let (rx, rw) = (mx + 18.0 + cw + 12.0 + 14.0, inner - cw - 28.0);
    let (bx, by) = (mx + 18.0 + 14.0, mid_h + 12.0 + 12.0);
    let bw = mw - 36.0 - 28.0;
    let bot_h = h - 17.0 - 18.0 - mid_h - 12.0;
    let (body_y, body_h) = (by + 24.0 + 10.0, bot_h - 24.0 - 24.0 - 10.0);
    let kb: K = Some([bx, body_y, bw, body_h]);

    let Some(a) = i.acct else {
        // 2738 «계좌를 불러오지 못했습니다» · 아래 탭은 처음 그대로(순환)
        cx.t(rx, curve_y + 16.0, format!("계좌를 불러오지 못했습니다.{}", if i.err.is_empty() { String::new() } else { format!(" ({})", i.err) }), st(16.0, 400, false, false, Col::Muted));
        tabs(&mut cx, &mut out, bx, by, i.tab_or("cy"), None);
        out.items = cx.v;
        return out;
    };
    let now = i.now;
    let positions = live_pos(a);
    let wallet = n0(&a["balance"]["wallet"]);
    let unreal: f64 = positions.iter().map(|p| n0(&p["unrealized_pnl"])).sum();
    let equity = wallet + unreal;
    let pw = pools(a);
    let wallet_d = if pw.pools.is_empty() { wallet } else { pw.wallet };
    let equity_d = if pw.pools.is_empty() { equity } else { pw.total };
    let today = key(now);
    let t0 = day0(&today);
    let mut tr: Vec<Trip> = a["ledger"].as_array().into_iter().flatten().map(trip).filter(|t| t.exit > 0.0).collect();
    tr.sort_by(|x, y| x.exit.total_cmp(&y.exit));
    let tt: Vec<&Trip> = tr.iter().filter(|t| t.exit >= t0).collect();
    let real: f64 = tt.iter().map(|t| t.pnl).sum();
    let inc = (truthy(&a["income_days"]) && truthy(&a["income_from"])).then(|| a["income_days"].as_object()).flatten();
    let inc_from = s(&a["income_from"]);
    let inc_on = |k: &str| inc.is_some() && k >= inc_from.as_str();
    let inc_sum = |k: &str| inc.into_iter().flatten().map(|(d, o)| if d.as_str() >= k { n0(&o["pnl"]) + n0(&o["transfer"]) } else { 0.0 }).sum::<f64>();
    let inc_cut = if inc.is_some() { day0(&inc_from) } else { f64::INFINITY };
    let wallet_at = |t: f64, t1: Option<f64>| -> f64 {
        let base = if inc_on(&key(t)) { wallet_d - inc_sum(&key(t)) } else {
            wallet_d - if inc.is_some() { inc_sum(&inc_from) } else { 0.0 } - tr.iter().map(|x| if x.exit >= t && x.exit < inc_cut { x.pnl } else { 0.0 }).sum::<f64>()
        };
        base + t1.map_or(0.0, |t1| { let (k0, k1) = (key(t), key(t1)); inc.into_iter().flatten().map(|(d, o)| if d.as_str() >= k0.as_str() && d.as_str() < k1.as_str() { n0(&o["transfer"]).max(0.0) } else { 0.0 }).sum() })
    };
    let ds = &a["day_start"];
    let ds_ok = truthy(ds) && s(&ds["date"]) == today && f(&ds["late_s"]) <= 900.0 && (pw.pools.len() < 2 || ds["pools"].as_array().map_or(0, |x| x.len()) >= pw.pools.len());
    let dsv = ds_ok.then(|| f(&ds["equity"]));
    let unreal0 = dsv.map(|d| d - f(&ds["wallet"]));
    let inc_t = if inc_on(&today) { Some(inc.and_then(|m| m.get(&today)).cloned().unwrap_or(Value::Null)) } else { None };
    let real_t = inc_t.as_ref().map_or(real, |o| n0(&o["pnl"]));
    let start = dsv.unwrap_or_else(|| if inc_t.is_some() { wallet_at(t0, None) } else { wallet_d - real });
    let net = if inc_t.is_some() { real_t + unreal - unreal0.unwrap_or(0.0) } else if let Some(d) = dsv { equity_d - d } else { real + unreal };
    let today_pct = (start > 0.0).then(|| net / start);

    // ── 곡선 칸(기간 칩 + svg) ──
    let mut x = curve_x;
    for (k, (lab, _)) in SPANS.iter().enumerate() { let cwk = chip(&mut cx, x, curve_y, &[lab], k == i.span); out.hits.push(([x, curve_y, cwk, 24.0], Hit::Span(k))); x += cwk + 4.0; }
    let span_days = SPANS[i.span].1;
    let from = if span_days == 0.0 { t0 } else if span_days > 0.0 { now - span_days * DAY } else { 0.0 };
    let sa: Vec<&Trip> = tr.iter().filter(|t| t.exit >= from).collect();
    let (vw, vh) = (curve_w.round().max(300.0), (curve_h.round() - 40.0).max(160.0));
    let py = curve_y + 24.0 + 10.0;
    let has_pos = !positions.is_empty();
    if span_days == 0.0 { cx.v.push(It::S { x: curve_x, y: py, w: curve_w, h: fl(curve_w * vh / vw), vb: [vw, vh], k: None, els: today_svg(&tt, unreal, has_pos, t0, now, vw, vh) }); }
    else if !sa.is_empty() { cx.v.push(It::S { x: curve_x, y: py, w: curve_w, h: fl(curve_w * vh / vw), vb: [vw, vh], k: None, els: curve_svg(&sa, unreal, has_pos, now, vw, vh, &pw) }); }
    else { empty(&mut cx, curve_x, py, curve_w, "이 기간에 닫힌 왕복이 없습니다."); }

    // ── 오른쪽 기둥(.aw) ──
    let sg = (!sa.is_empty()).then(|| stat_groups(&sa).remove(0));
    if !pw.pools.is_empty() { wealth(&mut cx, a, &pw, rx, curve_y, rw, net, today_pct, sg.as_ref(), SPANS[i.span].0); }

    // ── 아래 탭 ──
    // 2909 열린 순환: (심볼, 쪽)마다 잘리지 않은 것 중 진입이 가장 늦은 하나 · 포지션이 있는 것만
    let mut pick: Vec<(String, &Value)> = vec![];
    for t in a["trades"].as_array().into_iter().flatten() {
        if truthy(&t["closed"]) || t["legs"].as_array().is_none_or(|l| l.is_empty()) { continue; }
        let k = format!("{}|{}", s(&t["symbol"]), s(&t["side"]));
        match pick.iter_mut().find(|q| q.0 == k) {
            None => pick.push((k, t)),
            Some(q) => { let (ct, nt) = (truthy(&q.1["truncated_open"]), truthy(&t["truncated_open"])); if (ct && !nt) || (ct == nt && f(&t["entry_time"]) > f(&q.1["entry_time"])) { q.1 = t; } }
        }
    }
    let open: Vec<(String, &Value, &Value)> = pick.into_iter().filter_map(|(k, t)| positions.iter().find(|p| p["symbol"] == t["symbol"] && p["side"] == t["side"]).map(|p| (k, t, *p))).collect();
    let tab = if ["cy", "st", "cal"].contains(&i.tab) { i.tab } else if has_pos { "cy" } else { "st" };
    tabs(&mut cx, &mut out, bx, by, tab, Some(open.len()));
    cx.k = kb;
    match tab {
        "cy" => cycle(&mut cx, &mut out, i, &open, equity, bx, body_y, bw, body_h),
        "st" => if sa.is_empty() { empty(&mut cx, bx, body_y, bw, "이 기간에 닫힌 왕복이 없습니다."); } else { stats(&mut cx, &sa, bx, body_y, bw, body_h); },
        _ => {
            // 달력 | 고른 날
            let cols = tracks(bx, bw, &[1.0, 1.0], 20.0);
            let sel = i.sel.filter(|k| *k < today.as_str());
            calendar(&mut cx, &mut out, i, a, &tr, &today, sel, &wallet_at, inc, &inc_from, inc_cut, cols[0].0, body_y, cols[0].1, body_h);
            day_box(&mut cx, &tr, &tt, &today, sel, inc, &inc_on, &wallet_at, cols[1].0, body_y, cols[1].1, body_h);
        }
    }
    out.items = cx.v;
    out
}
impl In<'_> { fn tab_or<'b>(&'b self, d: &'b str) -> &'b str { if ["cy", "st", "cal"].contains(&self.tab) { self.tab } else { d } } }
/// .acct-empty(안쪽 14 · 둥근 12 · 유리 바탕 · 점선 테 · 12px 흐림)
fn empty(cx: &mut Cx, x: f64, y: f64, w: f64, msg: &str) {
    let y = y + 12.0;
    cx.b(x, y, w, 44.0, Col::White, 0.05, 12.0);
    cx.o(x, y, w, 44.0, 1.0, Col::SoftLine, 1.0, 12.0);
    cx.t(x + 15.0, y + 15.0, msg, st(12.0, 400, false, false, Col::Muted));
}
/// 아래 탭 줄(.acct-tabs) -- n = 열린 순환 수(없으면 «불러오지 못함» 처음 모양)
fn tabs(cx: &mut Cx, out: &mut Out, x0: f64, y: f64, on: &str, n: Option<usize>) {
    let ns = n.filter(|n| *n > 0).map(|n| n.to_string());
    let mut x = x0;
    for (k, lab) in [("cy", "진행 중 순환"), ("st", "성과 통계"), ("cal", "달력·복기"), ("wr", "오답노트")] {
        let parts: Vec<&str> = if k == "cy" { match &ns { Some(n) => vec![lab, " ", n.as_str()], None => vec![lab] } } else { vec![lab] };
        let w = chip(cx, x, y, &parts, k == on);
        if k != "wr" { out.hits.push(([x, y, w, 24.0], Hit::Tab(k))); }
        x += w + 4.0;
    }
}

/// 2375 acctWealthHtml(.aw 안쪽 0 · 칸 사이 10)
#[allow(clippy::too_many_arguments)]
fn wealth(cx: &mut Cx, a: &Value, w: &Pools, x: f64, y0: f64, wd: f64, net: f64, tp: Option<f64>, sg: Option<&Grp>, span_lab: &str) {
    let tn = |z: f64, wt: u16, c: Col| st(z, wt, false, true, c);
    let p1 = |v: f64| format!("{}%", fixed(v, if v >= 10.0 || v == 0.0 { 0 } else { 1 }));
    // ① 순자산 / 오늘(줄 19 = 본문 16px 줄 · 큰 숫자 줄 32)
    cx.t(x, y0 + 4.0, format!("순자산{} / 오늘", if w.pools.len() > 1 { " · 두 풀 합" } else { "" }), tn(11.5, 400, Col::Muted));
    let hy = y0 + 19.0;
    let bwid = cx.t(x, hy - 5.0, usd(w.total), st(30.0, 800, true, false, Col::Ink));
    let nc = tc(tone(net), Col::Text);
    let sx = x + bwid + 12.0;
    let s1 = format!("{} ", su(net));
    let w1 = cx.t(sx, hy + 6.0, s1, st(20.0, 800, true, false, nc));
    cx.t(sx + w1 + 4.0, hy + 12.0, pct(tp), st(14.0, 800, true, false, nc));
    let mut y = y0 + 51.0 + 10.0;
    // ② 풀 비중 막대(두 풀일 때 · 높이 20 · flex 기준 0 + 안쪽 20)
    if w.pools.len() > 1 {
        let gs: Vec<f64> = w.pools.iter().map(|p| p.share.max(4.0)).collect();
        let free = wd - 3.0 * (w.pools.len() - 1) as f64 - 20.0 * w.pools.len() as f64;
        let tot: f64 = gs.iter().sum();
        let mut xx = x;
        for (p, g) in w.pools.iter().zip(&gs) {
            let pw = if p.i + 1 == w.pools.len() { wd - (xx - x) } else { (( 20.0 + free * g / tot) * 64.0).round() / 64.0 };
            let (bg, ba, fg) = if p.i == 0 { (Col::Accent, 1.0, Col::OnFill) } else { (Col::Accent, 0.38, Col::Ink) };
            cx.b(xx, y, pw, 20.0, bg, ba, 8.0);
            let k = cx.k;
            cx.k = and(k, [xx, y, pw.round(), 20.0]);
            cx.t(xx + 10.0, y + 3.0, format!("{} {}", p.name, p1(p.share)), st(12.0, 700, false, true, fg));
            cx.k = k;
            xx += pw + 3.0;
        }
        y += 20.0 + 10.0;
    }
    // ③ 풀 줄(.aw-p: auto | 1fr | auto · 위 테 · 위 안쪽 8)
    let unprot: Vec<String> = a["unprotected"].as_array().into_iter().flatten().map(s).collect();
    let ch = cx.w("0", tn(12.0, 400, Col::Text));
    for p in &w.pools {
        cx.b(x, y, wd, 1.0, Col::SoftLine, 1.0, 0.0);
        let r0 = y + 9.0;
        // 이름 칸(nowrap · 넘치면 말줄임 · 최대 22ch)
        let stops: Vec<String> = live_pos(a).iter().filter(|q| s(&q["symbol"]).ends_with(&p.name) && unprot.contains(&format!("{}:{}", s(&q["symbol"]), s(&q["side"]).to_uppercase()))).map(|q| coin(&s(&q["symbol"]))).collect();
        let parts: Vec<(String, St)> = {
            let mut v = vec![(p.name.clone(), tn(12.5, 700, Col::Text)), (" ".into(), tn(12.0, 400, Col::Text)), (if p.n > 0 { p.pos.clone() } else { "포지션 없음".into() }, tn(12.0, 400, Col::Muted))];
            if !stops.is_empty() { v.push((" ".into(), tn(12.0, 400, Col::Text))); v.push((format!("🔴{} 손절 없음", stops.join("·")), tn(12.0, 400, Col::Bad))); }
            v
        };
        let nat: f64 = parts.iter().map(|(t, sty)| cx.w(t, *sty)).sum();
        let c1 = nat.min(22.0 * ch);
        let pv = usd(p.equity);
        let pvs = st(15.0, 700, true, false, Col::Text);
        let c3 = cx.w(&pv, pvs);
        let c2 = wd - c1 - c3 - 24.0;
        let k = cx.k;
        cx.k = and(k, [x, r0 + 2.5, c1.round(), 15.0]);
        let mut xx = x;
        for (t, sty) in &parts { if !t.trim().is_empty() { cx.t(xx, r0 + 2.5 + if sty.z == 12.5 { 0.0 } else { 1.0 }, t.clone(), *sty); } xx += cx.w(t, *sty); }
        cx.k = k;
        let mxx = x + c1 + 12.0;
        cx.b(mxx, r0 + 7.5, c2, 5.0, Col::SoftLine, 1.0, 3.0);
        let k = cx.k;
        cx.k = and(k, [mxx, r0 + 7.5, c2.round(), 5.0]);
        cx.b(mxx, r0 + 7.5, fl(c2 * p.used_pct.min(100.0) / 100.0), 5.0, Col::Amber, 1.0, 3.0);
        cx.k = k;
        cx.t(x + wd - c3, r0, pv, pvs);
        // 아래 줄(.aw-sub: 양 끝)
        let sy = r0 + 25.0;
        let m = tn(11.5, 400, Col::Muted);
        let mut xx = x + cx.t(x, sy, "증거금 ", m);
        xx += cx.t(xx, sy, p1(p.used_pct), m.c(if p.used_pct >= 75.0 { Col::Amber } else { Col::Muted }));
        cx.t(xx, sy, format!(" · 가용 {}", usd(p.avail)), m);
        let (l1, l2) = (format!("지갑 {} · 미실현 ", usd(p.wallet)), su(p.unreal));
        let (w1, w2) = (cx.w(&l1, m), cx.w(&l2, m));
        cx.t(x + wd - w1 - w2, sy, l1, m);
        cx.t(x + wd - w2, sy, l2, m.c(tc(tone(p.unreal), Col::Muted)));
        y += 48.0 + 10.0;
    }
    // ④ 승률 · 최대 낙폭(2열 · 위 테)
    cx.b(x, y, wd, 1.0, Col::SoftLine, 1.0, 0.0);
    let cols = tracks(x, wd, &[1.0, 1.0], 12.0);
    let m = tn(11.5, 400, Col::Muted);
    let big = st(19.0, 800, true, false, Col::Text);
    let yy = y + 9.0;
    let rows: [(String, String, Col, String); 2] = [
        (format!("승률 ({span_lab})"), sg.map_or("—".into(), |g| format!("{}%", jn(g.win))), Col::Text,
         sg.map_or("닫힌 왕복 없음".into(), |g| format!("{}건 · 손익비 {}", g.n, if g.al != 0.0 { fixed(g.aw / g.al.abs(), 2) } else { "—".into() }))),
        ("최대 낙폭".into(), sg.map_or("—".into(), |g| suc(g.mdd)), if sg.is_some_and(|g| g.mdd < 0.0) { Col::Bad } else { Col::Text }, format!("{span_lab} 누적 실현 기준")),
    ];
    for ((cx0, _), (l1, v, c, l2)) in cols.iter().zip(rows) {
        cx.t(*cx0, yy, l1, m);
        cx.t(*cx0, gt(yy + 16.0, 20.9, 19.0, true), v, big.c(c));
        cx.t(*cx0, yy + 16.0 + 20.890625 + 2.0, l2, m);
    }
}

/// 2598 acctTodaySvg
fn today_svg(tt: &[&Trip], unreal: f64, has_pos: bool, t0: f64, now: f64, w: f64, h: f64) -> Vec<El> {
    let (x0, x1, y0, y1) = (6.0, w - 6.0, 8.0, h - 16.0);
    let xx = |t: f64| x0 + ((t - t0) / DAY).clamp(0.0, 1.0) * (x1 - x0);
    let mut c = 0.0;
    let cum: Vec<(f64, f64)> = tt.iter().map(|t| { let a = c; c += t.pnl; (a, c) }).collect();
    let mut vals = vec![0.0];
    for q in &cum { vals.push(q.0); vals.push(q.1); }
    vals.push(c + unreal);
    let (lo, hi) = (vals.iter().copied().fold(f64::INFINITY, f64::min), vals.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let yy = |v: f64| y1 - (v - lo) / (hi - lo).max(1e-9) * (y1 - y0);
    let mut v = vec![El::Rect { r: [xx(now), y0, (x1 - xx(now)).max(0.0), y1 - y0], rx: 0.0, fill: Some((Col::Ink, 0.03)), stroke: None, sw: 1.0, dash: vec![] }, line([x0, yy(0.0), x1, yy(0.0)], Col::Ink, 0.22)];
    for hr in [0.0, 6.0, 12.0, 18.0, 24.0] { v.push(tx(x0 + hr / 24.0 * (x1 - x0), h - 3.0, format!("{}시", jn(hr)), 10.0, Col::Muted, if hr == 0.0 { Anchor::Start } else if hr == 24.0 { Anchor::End } else { Anchor::Middle })); }
    let mut d: D = vec![('M', vec![x0, yy(0.0)])];
    for (t, (a, b)) in tt.iter().zip(&cum) {
        let x = xx(t.exit);
        d.push(('H', vec![fx(x, 1)])); d.push(('V', vec![fx(yy(*b), 1)]));
        v.push(El::Rect { r: [fx(x - 4.0, 1), fx(yy(*a).min(yy(*b)), 1), 8.0, fx((yy(*a) - yy(*b)).abs().max(1.5), 1)], rx: 1.5, fill: Some((if b >= a { Col::Good } else { Col::Bad }, 1.0)), stroke: None, sw: 1.0, dash: vec![] });
    }
    d.push(('H', vec![fx(xx(now), 1)]));
    v.push(El::Path { d, fill: None, stroke: Some((Col::Ink, 0.55)), sw: 1.4, dash: vec![] });
    if has_pos {
        let col = if unreal >= 0.0 { Col::Good } else { Col::Bad };
        v.push(El::Rect { r: [fx(xx(now) - 4.0, 1), fx(yy(c).min(yy(c + unreal)), 1), 8.0, fx((yy(c + unreal) - yy(c)).abs().max(1.5), 1)], rx: 1.5, fill: Some((col, 0.3)), stroke: Some((col, 1.0)), sw: 1.0, dash: vec![2.0, 2.0] });
    }
    v.push(El::Line { p: [xx(now), y0, xx(now), y1], c: Col::Ink, a: 0.4, sw: 1.0, dash: vec![2.0, 3.0] });
    v
}

/// 2649 acctCurveSvg
fn curve_svg(a: &[&Trip], unreal: f64, has_pos: bool, now: f64, w: f64, h: f64, pw: &Pools) -> Vec<El> {
    let split = a.iter().any(|t| quote(&t.sym) == "USDC") && a.iter().any(|t| quote(&t.sym) == "USDT");
    let pool_un = |q: &str| pw.pools.iter().find(|p| p.name == q).map_or(0.0, |p| p.unreal);
    let (ml, mr, top) = (54.0, if split { 84.0 } else { 10.0 }, 10.0);
    let mid = jround(h * 0.66);
    let bot = h - 18.0;
    let ts = a[0].entry;
    let xx = |t: f64| ml + (t - ts) / (now - ts).max(1.0) * (w - ml - mr);
    let (mut c, mut peak, mut cu, mut cv) = (0.0f64, 0.0f64, 0.0, 0.0);
    struct P { t: f64, c: f64, dd: f64, u: f64, v: f64 }
    let pts: Vec<P> = a.iter().map(|t| { c += t.pnl; if quote(&t.sym) == "USDC" { cu += t.pnl } else { cv += t.pnl } peak = peak.max(c); P { t: t.exit, c, dd: c - peak, u: cu, v: cv } }).collect();
    let end_c = c + unreal;
    let end_p = [cu + pool_un("USDC"), cv + pool_un("USDT")];
    let mut los = vec![0.0, end_c];
    los.extend(pts.iter().map(|p| p.c));
    let mut his = los.clone();
    if split { los.extend(pts.iter().map(|p| p.u.min(p.v))); los.extend(end_p); his.extend(pts.iter().map(|p| p.u.max(p.v))); his.extend(end_p); }
    let lo = los.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = his.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let yy = |v: f64| mid - (v - lo) / (hi - lo).max(1e-9) * (mid - top);
    let dd_min = pts.iter().map(|p| p.dd).fold(-1e-9, f64::min);
    let yd = |v: f64| mid + 16.0 + (v / dd_min) * (bot - mid - 16.0);
    let mut v = vec![];
    for q in [lo, 0.0, hi] {
        v.push(line([ml, yy(q), w - mr, yy(q)], Col::Ink, if q == 0.0 { 0.22 } else { 0.07 }));
        v.push(tx(ml - 6.0, yy(q) + 4.0, suc(q), 10.0, Col::Muted, Anchor::End));
    }
    let (mut d, mut dd): (D, D) = (vec![('M', vec![ml, yy(0.0)])], vec![('M', vec![ml, yd(0.0)])]);
    for p in &pts { d.push(('H', vec![fx(xx(p.t), 1)])); d.push(('V', vec![fx(yy(p.c), 1)])); dd.push(('H', vec![fx(xx(p.t), 1)])); dd.push(('V', vec![fx(yd(p.dd), 1)])); }
    d.push(('H', vec![fx(xx(now), 1)])); dd.push(('H', vec![fx(xx(now), 1)]));
    let closed = |d: &D, base: f64| { let mut q = d.clone(); q.push(('V', vec![base])); q.push(('H', vec![ml])); q.push(('Z', vec![])); q };
    v.push(El::Path { d: closed(&d, yy(0.0)), fill: Some((Col::Ink, 0.05)), stroke: None, sw: 1.0, dash: vec![] });
    v.push(El::Path { d: d.clone(), fill: None, stroke: Some((Col::Ink, 0.85)), sw: 1.6, dash: vec![] });
    if split {
        let mut ends: Vec<(&str, f64, f64, f64)> = vec![];
        for (k, (q, dash, op)) in [("USDC", vec![], 0.7), ("USDT", vec![5.0, 3.0], 0.55)].into_iter().enumerate() {
            let mut dq: D = vec![('M', vec![ml, yy(0.0)])];
            for p in &pts { dq.push(('H', vec![fx(xx(p.t), 1)])); dq.push(('V', vec![fx(yy(if k == 0 { p.u } else { p.v }), 1)])); }
            dq.push(('H', vec![fx(xx(now), 1)])); dq.push(('V', vec![fx(yy(end_p[k]), 1)]));
            v.push(El::Path { d: dq, fill: None, stroke: Some((Col::Ink, op)), sw: 1.2, dash });
            ends.push((q, yy(end_p[k]), end_p[k], 0.0));
        }
        ends.push(("합산", yy(end_c), end_c, 0.0));
        ends.sort_by(|x, y| x.1.total_cmp(&y.1));
        for k in 0..ends.len() { ends[k].3 = if k > 0 && ends[k].1 < ends[k - 1].3 + 12.0 { ends[k - 1].3 + 12.0 } else { ends[k].1 }; }
        for e in &ends { let sum = e.0 == "합산"; v.push(El::Text { x: w - mr + 5.0, y: e.3 + 3.5, s: format!("{} {}", e.0, suc(e.2)), z: 10.5, wt: if sum { 700 } else { 500 }, mono: false, anchor: Anchor::Start, c: if sum { Col::Ink } else { Col::Muted }, a: 1.0, halo: false, ts: None }); }
    }
    if has_pos { v.push(El::Line { p: [xx(now), yy(c), xx(now), yy(end_c)], c: if unreal >= 0.0 { Col::Good } else { Col::Bad }, a: 1.0, sw: 3.0, dash: vec![3.0, 2.0] }); }
    v.push(tx(ml, mid + 12.0, "낙폭(직전 최고점 대비)", 10.0, Col::Muted, Anchor::Start));
    v.push(El::Path { d: closed(&dd, yd(0.0)), fill: Some((Col::Bad, 0.2)), stroke: None, sw: 1.0, dash: vec![] });
    v.push(El::Path { d: dd, fill: None, stroke: Some((Col::Bad, 0.7)), sw: 1.0, dash: vec![] });
    v.push(tx(ml - 6.0, yd(dd_min) + 4.0, suc(dd_min), 10.0, Col::Bad, Anchor::End));
    // 최악 3건(손실만) -- 이름표끼리 90px 안이면 둘째부터 동그라미만
    let mut order: Vec<usize> = (0..a.len()).collect();
    order.sort_by(|&x, &y| a[x].pnl.total_cmp(&a[y].pnl));
    let mut lab_x: Vec<f64> = vec![];
    for &k in order.iter().take(3).filter(|&&k| a[k].pnl < 0.0) {
        let (x, cy) = (xx(a[k].exit), yy(pts[k].c));
        let rxx = x > w - 110.0;
        let room = lab_x.iter().all(|q| (q - x).abs() > 90.0);
        if room { lab_x.push(x); }
        v.push(El::Circle { p: [x, cy, 5.0], fill: None, stroke: Some((Col::Bad, 1.0)), sw: 1.6 });
        if room { v.push(El::Text { x: x + if rxx { -8.0 } else { 8.0 }, y: cy + if rxx { -8.0 } else { 4.0 }, s: format!("{} {}", md(a[k].exit), suc(a[k].pnl)), z: 10.5, wt: 400, mono: false, anchor: if rxx { Anchor::End } else { Anchor::Start }, c: Col::Bad, a: 1.0, halo: false, ts: None }); }
    }
    for p in &pts { v.push(El::Rect { r: [fx(xx(p.t) - 3.0, 1), top, 6.0, mid - top], rx: 0.0, fill: None, stroke: None, sw: 1.0, dash: vec![] }); }
    let tk = if now - ts > 20.0 * DAY { 7.0 * DAY } else { DAY };
    let mut t = day0(&key(ts)) + tk;
    while t < now { v.push(tx(xx(t), h - 3.0, md(t), 10.0, Col::Muted, Anchor::Middle)); t += tk; }
    v
}

/// 2706 acctStatsHtml(fit: 카드 = 칸 높이 · 줄 넷이 남는 높이를 고르게)
fn stats(cx: &mut Cx, a: &[&Trip], x0: f64, y0: f64, w: f64, h: f64) {
    let g = stat_groups(a);
    let n = g.len() as f64;
    let cw = (w - 14.0 * (n - 1.0)) / n;
    let mx = g.iter().map(|x| x.tot.abs()).fold(1.0, f64::max);
    let mxa = g.iter().map(|x| x.aw.max(x.al.abs())).fold(1.0, f64::max);
    let mxd = g.iter().map(|x| x.mdd.abs()).fold(1.0, f64::max);
    let half = |v: f64, m: f64| fx(if m > 0.0 { (v.abs() / m * 100.0).min(100.0) / 2.0 } else { 0.0 }, 1);
    let rh = (h - 1.0 - 10.0 - 2.0 - 15.0 - 36.0) / 4.0;
    for (k, x) in g.iter().enumerate() {
        let cx0 = x0 + k as f64 * (cw + 14.0);
        cx.b(cx0, y0, cw, 1.0, Col::SoftLine, 1.0, 0.0);
        let hy = y0 + 11.0;
        let qw = cx.t(cx0, hy, x.q.clone(), st(13.0, 700, false, false, Col::Text));
        cx.t(cx0 + qw + 8.0, hy + 1.0, format!("{}건", x.n), st(12.0, 400, false, false, Col::Muted));
        let bar_x = cx0 + 74.0 + 8.0;
        let bar_w = cw - 74.0 - 54.0 - 16.0;
        for r in 0..4 {
            let ry = hy + 15.0 + 9.0 + r as f64 * (rh + 9.0);
            let lab = ["누적 실현", "승 · 패", "건당 이익 · 손실", "최대 낙폭"][r];
            cx.t(cx0, ry + (rh - 14.0) / 2.0, lab, st(12.0, 400, false, false, Col::Muted));
            let by = ry + (rh - 12.0) / 2.0;
            cx.b(bar_x, by, bar_w, 12.0, Col::SoftLine, 1.0, 3.0);
            let k0 = cx.k;
            cx.k = and(k0, [bar_x, by, bar_w.round(), 12.0]);
            if r < 3 && r != 1 { cx.b(bar_x + bar_w * 0.5, by, 1.0, 12.0, Col::Line, 1.0, 0.0); }
            let seg = |cx: &mut Cx, left: bool, p: f64, c: Col| { let ww = fl(bar_w * p / 100.0); let xx = if left { bar_x + bar_w * 0.5 } else { bar_x + bar_w * 0.5 - ww }; cx.b(xx, by, ww, 12.0, c, 1.0, 0.0); };
            let (val, vc) = match r {
                0 => { seg(cx, x.tot >= 0.0, half(x.tot, mx), if x.tot >= 0.0 { Col::Good } else { Col::Bad }); (suc(x.tot), tc(tone(x.tot), Col::Text)) }
                1 => {
                    let (gw, gl) = (if x.wn > 0 { x.wn as f64 } else { 0.001 }, if x.ln > 0 { x.ln as f64 } else { 0.001 });
                    let ww = fl(bar_w * gw / (gw + gl));
                    cx.b(bar_x, by, ww, 12.0, Col::Good, 1.0, 0.0);
                    cx.b(bar_x + ww, by, bar_w - ww, 12.0, Col::Bad, 1.0, 0.0);
                    (format!("{}%", jn(x.win)), Col::Text)
                }
                2 => { seg(cx, true, half(x.aw, mxa), Col::Good); seg(cx, false, half(x.al, mxa), Col::Bad); (if x.al != 0.0 { fixed(x.aw / x.al.abs(), 2) } else { "—".into() }, Col::Text) }
                _ => { cx.b(bar_x, by, fl(bar_w * fx((x.mdd.abs() / mxd * 100.0).clamp(0.0, 100.0), 1) / 100.0), 12.0, Col::Bad, 1.0, 0.0); (suc(x.mdd), Col::Bad) }
            };
            cx.k = k0;
            let vs = st(12.5, 700, true, false, vc);
            let vw = cx.w(&val, vs);
            cx.t(cx0 + cw - vw, gt(ry + (rh - 12.5) / 2.0, 12.5, 12.5, true), val, vs);
        }
    }
}

/// 2621 acctCalendar + 달력 머리(2822) · 아래 알림 줄
#[allow(clippy::too_many_arguments)]
fn calendar(cx: &mut Cx, out: &mut Out, i: &In, a: &Value, tr: &[Trip], today: &str, sel: Option<&str>, wallet_at: &dyn Fn(f64, Option<f64>) -> f64,
            inc: Option<&serde_json::Map<String, Value>>, inc_from: &str, inc_cut: f64, x0: f64, y0: f64, w: f64, h: f64) {
    struct Dd<'a> { pnl: f64, n: usize, inc: Option<&'a Value> }
    let mut days: Vec<(String, Dd)> = vec![];
    for t in tr { let k = key(t.exit); match days.iter_mut().find(|q| q.0 == k) { Some(q) => { q.1.pnl += t.pnl; q.1.n += 1; } None => days.push((k, Dd { pnl: t.pnl, n: 1, inc: None })) } }
    if let Some(m) = inc {
        for (k, o) in days.iter_mut() { if k.as_str() >= inc_from { o.inc = Some(m.get(k.as_str()).unwrap_or(&Value::Null)); o.pnl = o.inc.map_or(0.0, |v| n0(&v["pnl"])); } }
        for (k, o) in m { if k.as_str() >= inc_from && !days.iter().any(|q| &q.0 == k) && n0(&o["pnl"]).abs() >= 0.005 { days.push((k.clone(), Dd { pnl: n0(&o["pnl"]), n: 0, inc: Some(o) })); } }
    }
    let get = |k: &str| days.iter().find(|q| q.0 == k).map(|q| &q.1);
    let (ty, tm, _) = ymd(today);
    let mi = ty * 12 + tm - 1 - i.back;
    let (yy, mm) = (mi.div_euclid(12), mi.rem_euclid(12) + 1);
    let ym = format!("{yy:04}-{mm:02}");
    let mut oldest = vec![tr.first().map_or(ym.clone(), |t| key(t.exit)[..7].to_string())];
    if inc.is_some() { oldest.push(inc_from.get(..7).unwrap_or("").to_string()); }
    let oldest = oldest.into_iter().min().unwrap_or_default();
    let pct_of = |pnl: f64, t: f64, t1: f64| { let wv = wallet_at(t, Some(t1)); (wv > 0.0).then(|| pnl / wv) };
    let max_abs = days.iter().map(|q| q.1.pnl.abs()).fold(1e-9, f64::max);
    let first = day0(&format!("{ym}-01"));
    let dim = (days_civil(if mm == 12 { yy + 1 } else { yy }, if mm == 12 { 1 } else { mm + 1 }, 1) - days_civil(yy, mm, 1)) as usize;
    let lead = ((kday(first) + 4).rem_euclid(7) + 6) % 7;
    // 셀 값 먼저(이익일·손실일·달 합)
    let (mut mo, mut wd, mut ld) = (0.0, 0, 0);
    for d in 1..=dim { if let Some(o) = get(&format!("{ym}-{d:02}")) { mo += o.pnl; if o.n > 0 || o.inc.is_some_and(|v| truthy(&v["realized"])) { if o.pnl > 0.0 { wd += 1 } else { ld += 1 } } } }
    let mo_pct = pct_of(mo, first, first + dim as f64 * DAY);
    // 머리(.acct-cal-head: 가운데 맞춤 · 사이 8) -- ‹ › = 기본 단추(바탕 #efefef · 2px 돋움 테 · 13.33px)
    let ub = |cx: &mut Cx, x: f64, s0: &str, dis: bool| -> f64 {
        let t = st(13.3333, 400, false, false, Col::Rgb(0, 0, 0));
        let bw = cx.w(s0, t) + 16.0;
        let al = if dis { 0.3 * 0.35 } else { 1.0 };
        cx.b(x, y0, bw, 21.0, Col::Rgb(239, 239, 239), al, 0.0);
        cx.o(x, y0, bw, 21.0, 2.0, if dis { Col::Rgb(118, 118, 118) } else { Col::Rgb(0, 0, 0) }, if dis { 0.3 * 0.35 } else { 1.0 }, 0.0);
        cx.t(x + 8.0, y0 + 3.0, s0, t.c(if dis { Col::Rgb(16, 16, 16) } else { Col::Rgb(0, 0, 0) }).a(if dis { 0.3 * 0.35 } else { 1.0 }));
        bw
    };
    let dis_l = ym <= oldest;
    let w1 = ub(cx, x0, "‹", dis_l);
    if !dis_l { out.hits.push(([x0, y0, w1, 21.0], Hit::Month(1))); }
    let ty_s = format!("{}년 {}월", yy, mm);
    let tw = cx.t(x0 + w1 + 8.0, y0 + 3.0, ty_s, st(13.0, 700, false, false, Col::Text));
    let x2 = x0 + w1 + 8.0 + tw + 8.0;
    let w2 = ub(cx, x2, "›", i.back == 0);
    if i.back != 0 { out.hits.push(([x2, y0, w2, 21.0], Hit::Month(-1))); }
    let m12 = st(12.0, 400, false, false, Col::Muted);
    let (s1, s2, s3) = (format!("이익일 {wd} · 손실일 {ld} · "), su(mo), pct(mo_pct));
    let mc = tc(tone(mo), Col::Muted);
    let (a1, a2, a3) = (cx.w(&s1, m12), cx.w(&s2, m12.w(700)), cx.w(&s3, m12.w(700)));
    let sp = cx.w(" ", m12);
    let mut xx = x0 + w - a1 - a2 - sp - a3;
    xx += cx.t(xx, y0 + 3.5, s1, m12);
    xx += cx.t(xx, y0 + 3.5, s2, m12.w(700).c(mc)) + sp;
    cx.t(xx, y0 + 3.5, s3, m12.w(700).c(mc));
    // 아래 알림 줄(.acct-foot: 안쪽 0 20 16 · 11px · 경고 = 위 6 · 11.5px 줄 1.5 · 주황)
    let fail: Vec<String> = a["trades_failed"].as_array().into_iter().flatten().map(s).collect();
    let cut: Vec<String> = a["trades_truncated"].as_array().into_iter().flatten().map(s).collect();
    let warn = [if fail.is_empty() { String::new() } else { format!("체결 조회 실패: {} — 이 심볼의 진행 중 순환·원장이 잠시 빠졌습니다(다음 조회에 복구)", fail.join(", ")) },
                if cut.is_empty() { String::new() } else { format!("체결 이력 잘림: {} — 오래된 왕복·진행 중 합계가 창 안 몫만일 수 있습니다", cut.join(", ")) }].iter().filter(|x| !x.is_empty()).cloned().collect::<Vec<_>>().join(" · ");
    let foot = if inc.is_some() && ym.as_str() <= inc_from.get(..7).unwrap_or("") {
        Some(format!("칸 = 그날 지갑에 들어온 실현+펀딩+수수료 · {}부터(바이낸스 기록은 최근 3개월){}", md(inc_cut), if tr.first().is_some_and(|t| t.exit < inc_cut) { " · 그 앞은 닫힌 거래 왕복 합" } else { "" }))
    } else if !tr.is_empty() && ym <= oldest { Some(format!("원장은 {}부터입니다(서버 꼬리 최근 400왕복).", md(tr[0].exit))) } else { None };
    // (위 바깥 · 줄 높이 · 글) -- 격자 줄 사이 6 · 아래 안쪽 16
    let mut feet: Vec<(f64, f64, String, St)> = vec![];
    if let Some(t) = foot { feet.push((0.0, 13.0, t, st(11.0, 400, false, false, Col::Muted))); }
    if !warn.is_empty() { feet.push((6.0, 17.25, warn, st(11.5, 400, false, false, Col::Amber))); }
    let fh: f64 = feet.iter().map(|q| 6.0 + q.0 + q.1 + 16.0).sum();
    let mut fy = y0 + h - fh;
    for (mt, lh, t, sty) in feet { fy += 6.0 + mt; cx.t(x0 + 20.0, gt(fy, lh, sty.z, false), t, sty); fy += lh + 16.0; }
    // 달력 격자(7 1fr + 1.1fr · 사이 3 · 요일 줄 auto · 주 줄들이 남는 높이를 고르게)
    let gy = y0 + 21.0 + 6.0;
    let gh = h - 21.0 - 6.0 - fh;
    let cols = tracks(x0, w, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1], 3.0);
    let dow = st(10.5, 400, false, false, Col::Muted);
    for (k, d) in ["월", "화", "수", "목", "금", "토", "일", "주"].iter().enumerate() { let dw = cx.w(d, dow); cx.t(cols[k].0 + (cols[k].1 - dw) / 2.0, gy, *d, dow); }
    let nrow = (lead as usize + dim).div_ceil(7);
    let rows = tracks(gy + 13.0 + 3.0, gh - 13.0 - 3.0, &vec![1.0; nrow], 3.0);
    let vb = st(10.5, 700, true, false, Col::Text).ls(-0.02);
    let ve = st(9.5, 700, true, false, Col::Text).ls(-0.03);
    let cell_v = |cx: &mut Cx, right: f64, top: f64, rh: f64, b: &str, e: &str, c: Col, bw: u16| {
        let vt = top + fl((rh - 22.0) / 2.0);
        let bs = vb.c(c).w(bw);
        let w1 = cx.w(b, bs);
        cx.t(right - w1, gt(vt, 10.5, 10.5, true), b, bs);
        let w2 = cx.w(e, ve.c(c));
        cx.t(right - w2, gt(vt + 12.5, 9.5, 9.5, true), e, ve.c(c).a(0.78));
    };
    let (mut wk, mut col, mut wk0) = (0.0, lead as usize, first);
    let mut row = 0;
    for d in 1..=dim {
        let k = format!("{ym}-{d:02}");
        let o = get(&k);
        if let Some(o) = o { wk += o.pnl; }
        if col == 0 { wk0 = day0(&k); }
        let (cx0, cwid) = cols[col];
        let (ry, rh) = rows[row];
        let is_sel = sel == Some(k.as_str());
        match o {
            Some(o) => { let al = jround((0.1 + 0.6 * (o.pnl.abs() / max_abs).sqrt()) * 100.0) / 100.0; cx.b(cx0, ry, cwid, rh, if o.pnl >= 0.0 { Col::Good } else { Col::Bad }, al, 6.0); }
            None => cx.b(cx0, ry, cwid, rh, Col::Ink, 0.03, 6.0),
        }
        if k.as_str() == today { cx.o(cx0 - 1.0, ry - 1.0, cwid + 2.0, rh + 2.0, 1.0, Col::Accent, 1.0, 6.0); }
        if is_sel { cx.o(cx0, ry, cwid, rh, 2.0, Col::Amber, 1.0, 6.0); }
        if k.as_str() <= today { out.hits.push(([cx0, ry, cwid, rh], Hit::Day(k.clone()))); }
        let kc = cx.k;
        cx.k = and(kc, [cx0, ry, cwid.round(), rh.round()]);
        let ds = st(10.0, if is_sel { 700 } else { 400 }, false, false, if is_sel { Col::Amber } else { Col::Muted });
        let iy = if o.is_some() { ry + 3.0 + fl((rh - 6.0 - 22.0) / 2.0) } else { ry + 3.0 + fl((rh - 6.0 - 12.0) / 2.0) };
        cx.t(cx0 + 6.0, iy, d.to_string(), ds);
        if let Some(o) = o {
            let dp = pct_of(o.pnl, day0(&k), day0(&k) + DAY);
            let b = if o.pnl.abs() < 9.95 { format!("{}${}", if o.pnl >= 0.0 { "+" } else { "−" }, fixed(o.pnl.abs(), 1)) } else { suc(o.pnl) };
            cell_v(cx, cx0 + cwid - 6.0, ry + 3.0, rh - 6.0, &b, &pct(dp), Col::Text, 700);
        }
        cx.k = kc;
        col += 1;
        if col == 7 || d == dim {
            let (wx, ww) = cols[7];
            cx.b(wx, ry, 1.0, rh, Col::SoftLine, 1.0, 0.0);
            if wk != 0.0 { cell_v(cx, wx + ww - 4.0, ry, rh, &suc(wk), &pct(pct_of(wk, wk0, day0(&k) + DAY)), tc(tone(wk), Col::Text), 900); }
            wk = 0.0; col = 0; row += 1;
        }
    }
}

/// 고른 날 상세(#snapAcctDay: 합 줄 + 거래 목록) -- 아래 메모 칸(⛔ 입력)은 자리만 비운다
#[allow(clippy::too_many_arguments)]
fn day_box(cx: &mut Cx, tr: &[Trip], tt: &[&Trip], today: &str, sel: Option<&str>, inc: Option<&serde_json::Map<String, Value>>, inc_on: &dyn Fn(&str) -> bool,
           wallet_at: &dyn Fn(f64, Option<f64>) -> f64, x0: f64, y0: f64, w: f64, h: f64) {
    const NOTE_H: f64 = 106.0;
    let bh = h - NOTE_H - 8.0;
    let kc = cx.k;
    cx.k = and(kc, [x0, y0, w, bh]);
    let dk = sel.unwrap_or(today);
    let s0 = day0(dk);
    let vt: Vec<&Trip> = if sel.is_some() { tr.iter().filter(|t| t.exit >= s0 && t.exit < s0 + DAY).collect() } else { tt.to_vec() };
    let vreal: f64 = vt.iter().map(|t| t.pnl).sum();
    let inc_v = if inc_on(dk) { Some(inc.and_then(|m| m.get(dk)).unwrap_or(&Value::Null)) } else { None };
    let vfig = inc_v.map_or(vreal, |o| n0(&o["pnl"]));
    let wins = vt.iter().filter(|t| t.pnl > 0.0).count();
    let wbase = wallet_at(s0, Some(s0 + DAY));
    let fc = tc(tone(vfig), Col::Text);
    let w1 = cx.t(x0, y0 - 3.0, su(vfig), st(18.0, 700, true, false, fc));
    let w2 = cx.t(x0 + w1 + 10.0, y0 - 2.0, pct((wbase > 0.0).then(|| vfig / wbase)), st(17.0, 700, true, false, tc(tone(vfig), Col::Muted)).ls(-0.01));
    let (_, m, d) = ymd(dk);
    let wday = ["일", "월", "화", "수", "목", "금", "토"][(kday(s0) + 4).rem_euclid(7) as usize];
    cx.t(x0 + w1 + 10.0 + w2 + 10.0, y0 + 3.0, format!("{m:02}/{d:02}({wday}){} · 닫힌 거래 {}건{}{}", if sel.is_some() { "" } else { " · 오늘" }, vt.len(),
        if vt.is_empty() { String::new() } else { format!(" (승 {} 패 {})", wins, vt.len() - wins) }, if inc_v.is_some() { " · 펀딩·수수료 포함" } else { "" }), st(12.5, 400, false, false, Col::Muted));
    // 목록(.acct-day-ol: 줄 23 · 사이 2 · 104 | .8fr | 1.2fr | 1fr | auto · 안쪽 4 6)
    let ly = y0 + 18.0 + 8.0;
    cx.k = and(kc, [x0, ly, w, bh - 26.0]);
    if vt.is_empty() {
        cx.b(x0, ly, w, 23.0, Col::Ink, 0.03, 6.0);
        cx.t(x0 + 6.0, ly + 4.0, "닫힌 거래 없음", st(12.5, 400, false, false, Col::Muted));
    }
    let bs = |c: Col| st(12.5, 700, true, false, c);
    let mono = st(11.5, 400, true, false, Col::Muted);
    let sans = st(12.5, 400, false, false, Col::Text);
    let mut ry = ly;
    for (j, t) in vt.iter().enumerate() {
        // 줄마다 제 격자(104 | .8fr | 1.2fr | 1fr | auto = 이 줄 손익 폭) · 바닥선 맞춤(위 안쪽 4 + 12) · 시각이 104 를 넘으면 빈칸에서 두 줄(줄 13.8)
        let v = t.v;
        let s1 = su(t.pnl);
        let bw = cx.w(&s1, bs(Col::Text));
        let fr = (w - 12.0 - 40.0 - 104.0 - bw) / 3.0;
        let xs = [x0 + 6.0, x0 + 6.0 + 114.0, fl(x0 + 6.0 + 114.0 + 0.8 * fr + 10.0), fl(x0 + 6.0 + 114.0 + 2.0 * fr + 20.0)];
        let tm = format!("{}{}→{}", if key(t.entry) == key(t.exit) { String::new() } else { format!("{} ", md(t.entry)) }, hm(t.entry), hm(t.exit));
        let two = cx.w(&tm, mono) > 104.0 && tm.contains(' ');
        let lh = if two { 8.0 + 1.0 + fl(13.8) * 2.0 } else { 23.0 };
        if j % 2 == 0 { cx.b(x0, ry, w, lh, Col::Ink, 0.03, 6.0); }
        let top = ry + 4.0;
        match tm.split_once(' ').filter(|_| two) { Some((l1, l2)) => { cx.t(xs[0], top, l1, mono); cx.t(xs[0], top + fl(13.8), l2, mono); } None => { cx.t(xs[0], top, tm, mono); } }
        let xx = xs[1] + cx.t(xs[1], top, format!("{} {} ", coin(&t.sym), if v["side"] == "LONG" { "롱" } else { "숏" }), sans);
        cx.t(xx, top, jn(fx(n0(&v["max_qty"]), 4)), sans.w(700));
        cx.t(xs[2], top, format!("{} → {}", px(f(&v["entry_price"])), px(f(&v["exit_price"]))), mono.c(Col::Text));
        let adds = n0(&v["adds"]);
        let pk = f(&v["peak_x"]);
        let hs = format!("{}{}{}", dur(t.exit - t.entry), if adds != 0.0 { format!(" · 추가 {}회", jn(adds)) } else { String::new() }, if pk > 6.0 { " · " } else { "" });
        let hw = cx.t(xs[3], top, hs, sans.c(Col::Muted));
        if pk > 6.0 { cx.t(xs[3] + hw, top, format!("{}배", jn(pk)), sans.c(Col::Amber)); }
        cx.t(x0 + w - 6.0 - bw, top - 1.0, s1, bs(tc(tone(t.pnl), Col::Text)));
        ry += lh + 2.0;
    }
    cx.k = kc;
}

/// 2903 renderCycleCard(한 번에 하나 · 열린 순환이 둘 이상이면 탭) + 2855 cycleSvg
#[allow(clippy::too_many_arguments)]
fn cycle(cx: &mut Cx, out: &mut Out, i: &In, open: &[(String, &Value, &Value)], equity: f64, x0: f64, y0: f64, w: f64, h: f64) {
    let now = i.now;
    let Some(shown) = open.iter().find(|o| o.0 == i.cytab).or(open.first()) else {
        empty(cx, x0, y0, w, "진행 중인 순환이 없습니다 — 포지션을 열면 여기에 뜹니다.");
        return;
    };
    let mut y = y0;
    if open.len() > 1 {
        let mut x = x0;
        for o in open { let lab = format!("{} {}", coin(&s(&o.1["symbol"])), if o.1["side"] == "LONG" { "롱" } else { "숏" }); let cw = chip(cx, x, y, &[&lab], o.0 == shown.0); out.hits.push(([x, y, cw, 24.0], Hit::Cy(o.0.clone()))); x += cw + 4.0; }
        y += 24.0 + 12.0;
    }
    let (t, pos) = (shown.1, shown.2);
    let legs: Vec<[f64; 7]> = t["legs"].as_array().into_iter().flatten().map(|g| { let mut q = [0.0; 7]; for (k, v) in q.iter_mut().enumerate() { *v = n0(&g[k]); } q }).collect();
    let pkey = format!("{}|{}|{}", s(&t["symbol"]), s(&t["side"]), jn(f(&t["entry_time"])));
    out.want.push((pkey.clone(), s(&t["symbol"]), legs[0][0] - 20.0 * 60000.0));
    let rows: Vec<[f64; 2]> = i.paths.iter().find(|p| p.0 == pkey).map(|p| p.1.as_array().into_iter().flatten().map(|r| [n0(&r[0]), n0(&r[1])]).collect()).unwrap_or_default();
    let long = t["side"] == "LONG";
    let cn = coin(&s(&t["symbol"]));
    let dir = if long { 1.0 } else { -1.0 };
    let real: f64 = legs.iter().map(|g| g[5]).sum();
    let fee: f64 = legs.iter().map(|g| g[6]).sum();
    let unreal = n0(&pos["unrealized_pnl"]);
    let total = real + unreal - fee;
    let fpx = legs[0][1];
    let nin = legs.iter().filter(|g| g[3] != 0.0).count();
    let cut = truthy(&t["truncated_open"]);
    let hmn = |ms: f64| format!("{}{}", if key(ms) == key(now) { String::new() } else { format!("{} ", md(ms)) }, hm(ms));
    // 머리(가로 · 바닥선 맞춤 · 사이 10)
    let base = y + 14.0;
    cx.b(x0, base - 12.0, 48.0, 17.0, Col::Good, 1.0, 999.0);
    let lw = cx.w("진행 중", st(11.0, 800, false, false, Col::OnFill));
    cx.t(x0 + 8.0, base - 10.0, "진행 중", st(11.0, 800, false, false, Col::OnFill));
    let _ = lw;
    let tw = cx.t(x0 + 58.0, base - 14.0, format!("{cn} {} 순환", if long { "롱" } else { "숏" }), st(15.0, 800, false, false, Col::Text));
    cx.t(x0 + 58.0 + tw + 10.0, base - 11.0, format!("{} 시작 · {} · {} {} · {} {}", hmn(legs[0][0]), dur(now - legs[0][0]), if long { "매수" } else { "매도" }, nin, if long { "매도" } else { "매수" }, legs.len() - nin), st(12.0, 400, false, false, Col::Muted));
    y += 19.0 + 12.0;
    // 타일 여섯(6열 · 사이 8 · 안쪽 8 10 · 칸 68)
    let (mut qq, mut aa, mut peak) = (0.0f64, 0.0f64, 0.0f64);
    for g in &legs { if g[3] != 0.0 { aa = (aa * qq + g[1] * g[2]) / (qq + g[2]); qq += g[2]; } else { qq -= g[2]; } peak = peak.max(qq * aa); }
    let pct2 = |v: f64| format!("{}{}%", if v >= 0.0 { "+" } else { "−" }, fixed((v * 100.0).abs(), 2));
    let mark = n0(&pos["mark_price"]);
    let hold0 = peak;
    let hold = peak / fpx * (mark - fpx) * dir;
    let cyc = total - hold;
    let (tot_r, hold_r) = (total / peak, hold / hold0);
    let tiles: [(&str, String, Option<String>, Option<Tone>, String); 6] = [
        ("지금", format!("{} {cn}", cq(n0(&pos["qty"]))), None, None, format!("평단 {}", px(n0(&pos["entry_price"])))),
        ("미실현", su(unreal), None, Some(tone(unreal)), format!("현재가 {}", px(mark))),
        ("실현(부분 정리)", su(real), None, Some(tone(real)), format!("수수료 ${}", fixed(fee, 2))),
        ("합계", su(total), (!cut).then(|| pct2(tot_r)), Some(tone(total)), format!("{}{}", if cut { "체결 이력이 잘려 창 안 몫만 · ".to_string() } else { format!("최대 원금 {} 대비 · ", suc(peak).chars().skip(1).collect::<String>()) }, if equity > 0.0 { format!("순자산 {}", pct(Some(total / equity))) } else { String::new() })),
        ("그냥 보유였다면", if cut { "—".into() } else { su(hold) }, (!cut).then(|| pct2(hold_r)), (!cut).then(|| tone(hold)), if cut { "체결 이력이 잘려 첫 체결 모름".into() } else { format!("최대 원금 {} · 첫 진입가 @{}", suc(hold0).chars().skip(1).collect::<String>(), px(fpx)) }),
        ("순환 효과", if cut { "—".into() } else { su(cyc) }, (!cut).then(|| { let d = (tot_r - hold_r) * 100.0; format!("{}{}%p", if d >= 0.0 { "+" } else { "−" }, fixed(d.abs(), 2)) }), (!cut).then(|| tone(cyc)), "합계 − 그냥 보유 · 수익률 차".into()),
    ];
    let tcol = tracks(x0, w, &[1.0; 6], 8.0);
    for ((tx0, tw0), (lab, val, small, tn, sub)) in tcol.iter().zip(tiles) {
        cx.b(*tx0, y, *tw0, 68.0, Col::White, 0.05, 10.0);
        let m11 = st(11.0, 400, false, false, Col::Muted);
        cx.t(tx0 + 10.0, y + 8.0, lab, m11);
        let c = tn.map_or(Col::Text, |t| tc(t, Col::Text));
        let bs = st(15.0, 800, true, false, c);
        let has = small.is_some();
        let vw = cx.t(tx0 + 10.0, y + 24.0, if has { format!("{val} ") } else { val }, bs);
        if let Some(sm) = small { cx.t(tx0 + 10.0 + vw + 2.0, y + 27.0, sm, st(12.0, 700, true, false, c).a(0.85)); }
        cx.t(tx0 + 10.0, y + 47.0, sub, m11);
    }
    y += 68.0 + 12.0;
    // 그림(높이 = 칸 높이 − 나머지 · 2962 cyRest)
    let rest = (y - y0) + 10.0 + 24.0;
    let ch = (h - rest).max(60.0);
    cx.b(x0, y, w, ch, Col::ChartBg, 1.0, 10.0);
    cx.v.push(It::S { x: x0, y, w, h: ch, vb: [w, ch], k: cx.k, els: cycle_svg(&legs, long, mark, n0(&pos["entry_price"]), &rows, now, w, ch) });
    y += ch + 10.0;
    // 체결 칩(최신부터 · 한 줄만 보임 -- 넘치면 숨김)
    let kc = cx.k;
    cx.k = and(kc, [x0, y, w, 24.0]);
    let mut x = x0;
    let t12 = st(12.0, 700, true, false, Col::Text);
    for g in legs.iter().rev() {
        let b = if g[3] != 0.0 { long } else { !long };
        let tm = hmn(g[0]);
        let (s1, s2, s3) = (format!("{} ", if b { "매수" } else { "매도" }), px(g[1]), format!(" ×{}{}{}", cq(g[2]), if g[4] != 0.0 { "" } else { " · 테이커" }, if g[5] != 0.0 { " " } else { "" }));
        let em = (g[5] != 0.0).then(|| su(g[5]));
        let ws = [cx.w(&tm, t12), cx.w(&s1, t12), cx.w(&s2, t12.w(800)), cx.w(&s3, t12), em.as_ref().map_or(0.0, |e| 4.0 + cx.w(e, t12))];
        let cw = 8.0 + 12.0 + ws[0] + 6.0 + ws[1] + ws[2] + ws[3] + ws[4] + 8.0;
        if x > x0 && x + cw > x0 + w { break; }
        cx.b(x, y, cw, 24.0, Col::White, 0.05, 7.0);
        cx.b(x + 8.0, y + 9.0, 6.0, 6.0, if b { Col::Good } else { Col::Bad }, 1.0, 3.0);
        let mut xx = x + 20.0;
        cx.t(xx, y + 4.0, tm, t12.c(Col::Muted)); xx += ws[0] + 6.0;
        cx.t(xx, y + 4.0, s1, t12); xx += ws[1];
        cx.t(xx, y + 4.0, s2, t12.w(800)); xx += ws[2];
        cx.t(xx, y + 4.0, s3, t12); xx += ws[3];
        if let Some(e) = em { cx.t(xx + 4.0, y + 4.0, e, t12.c(tc(tone(g[5]), Col::Text))); }
        x += cw + 6.0;
    }
    cx.k = kc;
}
#[allow(clippy::too_many_arguments)]
fn cycle_svg(legs: &[[f64; 7]], long: bool, mark: f64, entry: f64, rows: &[[f64; 2]], now: f64, w: f64, h: f64) -> Vec<El> {
    let (l, r, t, b) = (8.0, 104.0, 10.0, 22.0);
    let (mut q, mut avg) = (0.0f64, 0.0f64);
    let steps: Vec<(f64, f64)> = legs.iter().map(|g| { if g[3] != 0.0 { avg = (avg * q + g[1] * g[2]) / (q + g[2]); q += g[2]; } else { q -= g[2]; } (g[0], avg) }).collect();
    let t0 = legs[0][0] - 20.0 * 60000.0;
    let mut path: Vec<[f64; 2]> = rows.iter().filter(|r| r[0] >= t0).copied().collect();
    if path.is_empty() { path.extend(legs.iter().map(|g| [g[0], g[1]])); }
    path.push([now, mark]);
    let ps: Vec<f64> = path.iter().map(|r| r[1]).chain(legs.iter().map(|g| g[1])).collect();
    let (mut lo, mut hi) = (ps.iter().copied().fold(f64::INFINITY, f64::min), ps.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let pad = (hi - lo).max(hi * 1e-4) * 0.22;
    lo -= pad; hi += pad;
    let x0 = path[0][0].min(legs[0][0]);
    let xx = |ms: f64| l + (ms - x0) / (now - x0).max(1.0) * (w - l - r);
    let yy = |p: f64| t + (hi - p) / (hi - lo) * (h - t - b);
    let mut d: D = vec![];
    for (ms, a) in &steps { if d.is_empty() { d.push(('M', vec![fx(xx(*ms), 1), fx(yy(*a), 1)])); } else { d.push(('H', vec![fx(xx(*ms), 1)])); d.push(('V', vec![fx(yy(*a), 1)])); } }
    d.push(('H', vec![fx(xx(now), 1)]));
    let first = legs[0][1];
    let mut v = vec![El::Path { d, fill: None, stroke: Some((Col::Accent, 1.0)), sw: 1.3, dash: vec![3.0, 3.0] },
                     El::Line { p: [l, fx(yy(first), 1), w - r, fx(yy(first), 1)], c: Col::Amber, a: 1.0, sw: 1.0, dash: vec![5.0, 4.0] }];
    let mut pd: D = vec![];
    for (k, p) in path.iter().enumerate() { pd.push((if k == 0 { 'M' } else { 'L' }, vec![fx(xx(p[0]), 1), fx(yy(p[1]), 1)])); }
    v.push(El::Path { d: pd, fill: None, stroke: Some((Col::Neutral, 1.0)), sw: 1.5, dash: vec![] });
    let buy = |inc: bool| if inc { long } else { !long };
    if legs.len() <= 30 {
        let mut placed: Vec<[f64; 2]> = vec![];
        let narrow = w < 600.0;
        let gap = if narrow { 62.0 } else { 104.0 };
        for g in legs {
            let x = xx(g[0]).max(l + 46.0).min(w - r - 46.0);
            let up = !buy(g[3] != 0.0);
            let mut ty = yy(g[1]) + if up { -10.0 } else { 18.0 };
            let mut moved = false;
            while placed.iter().any(|p| (ty - p[1]).abs() < 13.0 && (x - p[0]).abs() < gap) { ty += if up { -14.0 } else { 14.0 }; moved = true; }
            if narrow && (moved || ty < t + 10.0 || ty > h - b || legs.iter().any(|q| (x - xx(q[0])).abs() < gap / 2.0 + 6.0 && (ty - 4.0 - yy(q[1])).abs() < 13.0)) { continue; }
            placed.push([x, ty]);
            v.push(El::Text { x: fx(x, 1), y: fx(ty, 1), s: px(g[1]), z: 10.5, wt: 800, mono: true, anchor: Anchor::Middle, c: if up { Col::Bad } else { Col::Good }, a: 1.0, halo: true, ts: (!narrow).then(|| (format!(" ×{}", cq(g[2])), 600, Col::Muted, 1.0)) });
        }
    }
    for g in legs {
        let (x, y) = (xx(g[0]), yy(g[1]));
        let bb = buy(g[3] != 0.0);
        let c = if bb { Col::Good } else { Col::Bad };
        let d: D = if bb { vec![('M', vec![x - 5.0, y + 4.0]), ('L', vec![x + 5.0, y + 4.0]), ('L', vec![x, y - 5.0]), ('Z', vec![])] } else { vec![('M', vec![x - 5.0, y - 4.0]), ('L', vec![x + 5.0, y - 4.0]), ('L', vec![x, y + 5.0]), ('Z', vec![])] };
        v.push(El::Path { d, fill: Some(if g[4] != 0.0 { (Col::ChartBg, 1.0) } else { (c, 1.0) }), stroke: Some((c, 1.0)), sw: 1.6, dash: vec![] });
    }
    v.push(El::Circle { p: [fx(xx(now), 1), fx(yy(mark), 1), 3.5], fill: Some((Col::Ink, 1.0)), stroke: None, sw: 1.0 });
    let mut labs = [(yy(mark), format!("지금 {}", px(mark)), Col::Ink), (yy(entry), format!("평단 {}", px(entry)), Col::Accent), (yy(first), format!("첫 {}", px(first)), Col::Amber)];
    labs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut prev = -1e9;
    for (y, s1, c) in labs { let y = y.max(prev + 12.0); prev = y; v.push(El::Text { x: w - r + 6.0, y: fx(y + 4.0, 1), s: s1, z: 10.0, wt: 600, mono: true, anchor: Anchor::Start, c, a: 1.0, halo: false, ts: None }); }
    v.push(El::Text { x: l, y: h - 6.0, s: hm(x0), z: 10.0, wt: 600, mono: true, anchor: Anchor::Start, c: Col::Muted, a: 1.0, halo: false, ts: None });
    v.push(El::Text { x: w - r, y: h - 6.0, s: format!("{} KST", hm(now)), z: 10.0, wt: 600, mono: true, anchor: Anchor::End, c: Col::Muted, a: 1.0, halo: false, ts: None });
    v
}

// ── 그리기 ──
fn pts_of(d: &D) -> Vec<Vec<[f64; 2]>> {
    let mut segs: Vec<Vec<[f64; 2]>> = vec![];
    let mut cur = [0.0, 0.0];
    for (c, n) in d {
        match c {
            'M' => { cur = [n[0], n[1]]; segs.push(vec![cur]); }
            'L' => { cur = [n[0], n[1]]; if let Some(s) = segs.last_mut() { s.push(cur); } }
            'H' => { cur[0] = n[0]; if let Some(s) = segs.last_mut() { s.push(cur); } }
            'V' => { cur[1] = n[0]; if let Some(s) = segs.last_mut() { s.push(cur); } }
            _ => { if let Some(s) = segs.last_mut() { let f0 = s[0]; s.push(f0); } }
        }
    }
    segs
}
/// 그림 단위 → [(자르기 상자, 그리기)] · svg 는 viewBox 를 상자에 meet 로
pub fn prims(items: &[It], m: Measure) -> Vec<(K, Prim)> {
    let mut v = vec![];
    let cl = |k: &K| if k.is_some() { Clip::Plot } else { Clip::No };
    for it in items {
        match it {
            It::T { x, y, s, z, wt, mono, tnum: tn, ls, c, a, k } => {
                let base = y + asc(*z, *mono);
                let s = if *tn { tnum(s) } else { s.clone() };
                let one = |x: f64, s: String| if *mono { Prim::Mono { x, y: base, s, size: *z, wt: *wt, c: *c, a: *a, anchor: Anchor::Start, halo: false, clip: cl(k) } } else { Prim::Text { x, y: base, s, size: *z, wt: *wt, c: *c, a: *a, anchor: Anchor::Start, halo: false, clip: cl(k) } };
                if *ls == 0.0 { v.push((*k, one(*x, s))); continue; }
                let mut xx = *x; // 자간 = 글자마다
                for ch in s.chars() { let t = ch.to_string(); let w = m(&t, *z, *wt, *mono); v.push((*k, one(xx, t))); xx += w + ls; }
            }
            &It::B { x, y, w, h, c, a, rad, k } => v.push((k, Prim::Rect { x, y, w, h, c, a, rx: rad, clip: cl(&k) })),
            &It::O { x, y, w, h, lw, c, a, rad, k } => {
                let (x, y, w, h) = (x + lw / 2.0, y + lw / 2.0, w - lw, h - lw);
                v.push((k, if rad == 0.0 { Prim::StrokeRect { x, y, w, h, c, a, lw, clip: cl(&k) } } else { Prim::Line { pts: rr_pts(x, y, w, h, (rad - lw / 2.0).min(w / 2.0).min(h / 2.0)), c, a, lw, dash: None, clip: cl(&k) } }));
            }
            It::S { x, y, w, h, vb, k, els } => {
                let sc = (w / vb[0]).min(h / vb[1]);
                let (ox, oy) = (x + (w - vb[0] * sc) / 2.0, y + (h - vb[1] * sc) / 2.0);
                let pt = |p: [f64; 2]| [ox + p[0] * sc, oy + p[1] * sc];
                let stroke = |pts: Vec<[f64; 2]>, c: Col, a: f64, sw: f64, dash: &[f64]| match dash { [on, off, ..] => Prim::Dashed { pts, c, a, lw: sw * sc, on: on * sc, off: off * sc, clip: cl(k) }, _ => Prim::Line { pts, c, a, lw: sw * sc, dash: None, clip: cl(k) } };
                for e in els {
                    match e {
                        El::Line { p, c, a, sw, dash } => v.push((*k, stroke(vec![pt([p[0], p[1]]), pt([p[2], p[3]])], *c, *a, *sw, dash))),
                        El::Rect { r, rx, fill, stroke: sk, sw, dash } => {
                            let q = pt([r[0], r[1]]);
                            if let Some((c, a)) = fill { v.push((*k, Prim::Rect { x: q[0], y: q[1], w: r[2] * sc, h: r[3] * sc, c: *c, a: *a, rx: rx * sc, clip: cl(k) })); }
                            if let Some((c, a)) = sk { let (x1, y1, x2, y2) = (r[0], r[1], r[0] + r[2], r[1] + r[3]); v.push((*k, stroke(vec![pt([x1, y1]), pt([x2, y1]), pt([x2, y2]), pt([x1, y2]), pt([x1, y1])], *c, *a, *sw, dash))); }
                        }
                        El::Path { d, fill, stroke: sk, sw, dash } => {
                            for sg in pts_of(d) {
                                if let Some((c, a)) = fill {
                                    if sg.len() > 3 { // «곡선 → 바닥선 → 처음» 꼴(곡선 밑 면) = 바닥선까지 사다리꼴 띠
                                        let base = sg[sg.len() - 3][1];
                                        v.push((*k, Prim::Area { pts: sg[..sg.len() - 3].iter().map(|q| pt(*q)).collect(), base: oy + base * sc, c: *c, a: *a }));
                                    } else if sg.len() == 4 { // 체결 삼각형
                                        let (p0, p1, p2) = (pt(sg[0]), pt(sg[1]), pt(sg[2])); // 밑변 두 점 사이로 꼭짓점 = 밑변까지 띠 둘
                                        v.push((*k, Prim::Area { pts: vec![p0, p2, p1], base: p0[1], c: *c, a: *a }));
                                    }
                                }
                                if let Some((c, a)) = sk { v.push((*k, stroke(sg.iter().map(|q| pt(*q)).collect(), *c, *a, *sw, dash))); }
                            }
                        }
                        El::Circle { p, fill, stroke: sk, sw } => {
                            let q = pt([p[0], p[1]]);
                            if let Some((c, a)) = fill { v.push((*k, Prim::Circle { x: q[0], y: q[1], r: p[2] * sc, c: *c, a: *a, clip: cl(k) })); }
                            if let Some((c, a)) = sk { v.push((*k, Prim::Line { pts: (0..=24).map(|j| { let t = j as f64 / 24.0 * std::f64::consts::TAU; [q[0] + p[2] * sc * t.cos(), q[1] + p[2] * sc * t.sin()] }).collect(), c: *c, a: *a, lw: sw * sc, dash: None, clip: cl(k) })); }
                        }
                        El::Text { x: tx0, y: ty, s, z, wt, mono, anchor, c, a, halo, ts } => {
                            let q = pt([*tx0, *ty]);
                            let w1 = m(s, *z, *wt, *mono) * sc;
                            let w2 = ts.as_ref().map_or(0.0, |t| m(&t.0, *z, t.1, *mono) * sc);
                            let x1 = match anchor { Anchor::Start => q[0], Anchor::Middle => q[0] - (w1 + w2) / 2.0, Anchor::End => q[0] - w1 - w2 };
                            let one = |x: f64, s: String, wt: u16, c: Col, a: f64| if *mono { Prim::Mono { x, y: q[1], s, size: z * sc, wt, c, a, anchor: Anchor::Start, halo: *halo, clip: cl(k) } } else { Prim::Text { x, y: q[1], s, size: z * sc, wt, c, a, anchor: Anchor::Start, halo: *halo, clip: cl(k) } };
                            v.push((*k, one(x1, s.clone(), *wt, *c, *a)));
                            if let Some((s2, w2t, c2, a2)) = ts { v.push((*k, one(x1 + w1, s2.clone(), *w2t, *c2, *a2))); }
                        }
                    }
                }
            }
        }
    }
    v
}

// ── 받기 ──
use crate::feed::{Kind, Req};
/// 원본 전역(latestBinanceAccount · lastGoodAccount · staleAccount · lastAccountError · cyclePaths · acctSpanIdx · acctCalBack · acctSelDay · acctTab · cycleTab)
pub struct Store {
    pub acct: Option<Value>, pub good: Option<(f64, Value)>, pub stale: Option<Value>, pub err: String,
    pub paths: Vec<(String, Value)>, path_at: Vec<(String, f64)>, pending: std::collections::VecDeque<String>,
    pub span: usize, pub back: i64, pub sel: Option<String>, pub tab: String, pub cytab: String,
    acct_at: f64,
    pub hits: Vec<([f64; 4], Hit)>,
}
impl Default for Store {
    fn default() -> Self { Self { acct: None, good: None, stale: None, err: String::new(), paths: vec![], path_at: vec![], pending: Default::default(), span: 2, back: 0, sel: None, tab: String::new(), cytab: String::new(), acct_at: f64::NEG_INFINITY, hits: vec![] } }
}
/// 1649 BINANCE_ACCOUNT_POLL_MS
const ACCT_POLL_MS: f64 = 10_000.0;
impl Store {
    /// 2991 refreshBinanceAccount(10초 · 숨으면 안 묻는다)
    pub fn tick(&mut self, now: f64, hidden: bool, api: &str, out: &mut Vec<Req>) {
        if hidden || now - self.acct_at < ACCT_POLL_MS { return; }
        self.acct_at = now;
        out.push(Req::Get { kind: Kind::AccAcct, url: format!("{api}/api/binance-account") });
    }
    /// 2970 renderBinanceAccount -- ok 면 그것 · 아니면 서버 last_good(나이만큼 옛날) 또는 30분 안 마지막 정상값으로 그린다(낡은 값 · 오류 글은 보관)
    pub fn on_acct(&mut self, body: Option<&str>, now: f64) {
        let v = body.and_then(|b| serde_json::from_str::<Value>(b).ok());
        if let Some(v) = v.as_ref().filter(|v| v["ok"] == true) { self.acct = Some(v.clone()); self.good = Some((now, v.clone())); self.stale = None; self.err.clear(); return; }
        self.acct = None;
        let p = v.unwrap_or(Value::Null);
        self.err = [&p["hint"], &p["error"]].iter().find(|x| truthy(x)).map(|x| s(x)).unwrap_or_default();
        let lg = &p["last_good"];
        self.stale = if truthy(lg) && lg["ok"] == true { Some(lg.clone()) } else { self.good.as_ref().filter(|g| now - g.0 < 1_800_000.0).map(|g| g.1.clone()) };
    }
    /// /api/cycle-path(보이는 순환 · 60초) -- want = 이번 그림이 보인 순환
    pub fn want(&mut self, want: &[(String, String, f64)], now: f64, api: &str, out: &mut Vec<Req>) {
        for (k, sym, start) in want {
            let at = self.path_at.iter_mut().find(|q| &q.0 == k);
            if at.as_ref().is_some_and(|q| now - q.1 <= 60_000.0) { continue; }
            match at { Some(q) => q.1 = now, None => self.path_at.push((k.clone(), now)) }
            self.pending.push_back(k.clone());
            out.push(Req::Get { kind: Kind::AccPath, url: format!("{api}/api/cycle-path?symbol={sym}&start={}", jn(*start)) });
        }
    }
    /// ponytail: 응답은 물은 차례대로 온다고 본다(순환마다 60초에 한 번이라 겹칠 일이 드묾) -- 어긋나면 키를 응답에 실어야 한다
    pub fn on_path(&mut self, body: Option<&str>) {
        let Some(k) = self.pending.pop_front() else { return };
        let Some(v) = body.and_then(|b| serde_json::from_str::<Value>(b).ok()).filter(|v| v["ok"] == true) else { return };
        match self.paths.iter_mut().find(|q| q.0 == k) { Some(q) => q.1 = v["rows"].clone(), None => self.paths.push((k, v["rows"].clone())) }
    }
    pub fn input(&self, now: f64) -> In<'_> {
        In { now, acct: self.acct.as_ref().or(self.stale.as_ref()), err: &self.err, span: self.span, back: self.back, sel: self.sel.as_deref(), tab: &self.tab, cytab: &self.cytab, paths: &self.paths }
    }
    /// 누르기(2486 · 2591 · 2899) → 보기가 바뀌었나
    pub fn click(&mut self, x: f64, y: f64, now: f64) -> bool {
        let Some(h) = self.hits.iter().find(|(r, _)| x >= r[0] && x < r[0] + r[2] && y >= r[1] && y < r[1] + r[3]).map(|q| q.1.clone()) else { return false };
        match h {
            Hit::Span(k) => self.span = k,
            Hit::Month(d) => self.back = (self.back + d).max(0),
            Hit::Day(k) => self.sel = if self.sel.as_deref() == Some(k.as_str()) || k == key(now) { None } else { Some(k) },
            Hit::Tab(k) => self.tab = k.into(),
            Hit::Cy(k) => self.cytab = k,
        }
        true
    }
}

thread_local! { static ST: RefCell<Option<Store>> = const { RefCell::new(None) }; }
/// ?card=acc 로 켠다(web.rs start)
pub fn start() { ST.with(|s| *s.borrow_mut() = Some(Store::default())); }
pub fn on() -> bool { ST.with(|s| s.borrow().is_some()) }
/// Feed::tick 의 이 카드 몫 -- 켜져 있으면 true(다른 받기는 안 한다)
pub fn tick(now: f64, hidden: bool, api: &str, out: &mut Vec<Req>) -> bool {
    ST.with(|s| s.borrow_mut().as_mut().map(|st| st.tick(now, hidden, api, out)).is_some())
}
/// web.rs on_get 의 이 카드 몫
pub fn on_get(kind: &str, body: Option<&str>, now: f64) {
    ST.with(|s| if let Some(st) = s.borrow_mut().as_mut() { if kind == "AccPath" { st.on_path(body) } else { st.on_acct(body, now) } });
}
/// 한 장(Chart::ui 에서) -- 본문 = cfg.w_all×h_all(fill 이면 창 · 1920 화면 = 1872×934)
pub fn ui(ch: &mut crate::Chart, ui: &mut eframe::egui::Ui, now: f64) {
    if ch.fill { let r = ui.ctx().content_rect(); (ch.cfg.w_all, ch.cfg.h_all) = ((r.width() - 2.0 * ch.origin.x) as f64, (r.height() - 2.0 * ch.origin.y) as f64); }
    let (w, h, o) = (ch.cfg.w_all, ch.cfg.h_all, ch.origin);
    let p = ui.painter();
    if ch.bg.a() > 0 { p.rect_filled(p.clip_rect(), 0.0, ch.bg); }
    let measure = |s: &str, z: f64, wt: u16, mono: bool| if mono { p.layout_no_wrap(s.to_owned(), crate::draw::mono(z as f32, wt), eframe::egui::Color32::WHITE).size().x as f64 } else { crate::draw::measure(p, s, z, wt) };
    let click = ui.input(|i| if i.pointer.primary_clicked() { i.pointer.interact_pos() } else { None });
    let api = ch.feed.urls.api.clone();
    ST.with(|s| {
        let mut b = s.borrow_mut();
        let Some(st) = b.as_mut() else { return };
        if let Some(c) = click { st.click((c.x - o.x) as f64, (c.y - o.y) as f64, now); }
        let out = body(&st.input(now), w, h, &measure);
        st.hits = out.hits;
        st.want(&out.want, now, &api, &mut ch.feed.out);
        let paint = |run: &[Prim], k: K| { let r = k.unwrap_or([0.0, 0.0, w, h]); crate::draw::paint(p, run, o, r, r, 1.0); };
        let (mut run, mut cur): (Vec<Prim>, K) = (vec![], None);
        for (k, pr) in prims(&out.items, &measure) {
            if k != cur && !run.is_empty() { paint(&run, cur); run.clear(); }
            cur = k;
            run.push(pr);
        }
        paint(&run, cur);
    });
    ui.ctx().request_repaint_after(std::time::Duration::from_secs(1));
}

#[cfg(test)]
mod t {
    use super::*;
    use crate::mt::t::{col, lookup, pmc, r4, rad, wb};

    fn fx_(name: &str) -> Value { serde_json::from_str(&std::fs::read_to_string(format!("{}/compare/fixture/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap() }
    fn kk(v: &Value) -> K { v.as_array().map(|_| r4(v)) }
    /// 비교 단위: 글(t) · 칠(b) · 테(o) · svg(s: 상자 · viewBox)
    #[derive(Debug, Clone)]
    struct Q { k: char, r: [f64; 4], c: [f64; 4], s: String, z: f64, wt: u16, mono: bool, lw: f64, rad: f64, kk: K }
    fn q(k: char, r: [f64; 4], c: [f64; 4], kk: K) -> Q { Q { k, r, c, s: String::new(), z: 0.0, wt: 0, mono: false, lw: 0.0, rad: 0.0, kk } }
    fn pc(p: &Paint) -> [f64; 4] { p.map_or([0.0; 4], |(c, a)| col(c, a)) }
    fn anc(a: Anchor) -> &'static str { match a { Anchor::Start => "start", Anchor::Middle => "middle", Anchor::End => "end" } }
    fn dnums(d: &D) -> (String, Vec<f64>) { (d.iter().map(|c| c.0).collect(), d.iter().flat_map(|c| c.1.clone()).collect()) }
    /// d 글 → (명령 글자들, 수들)
    fn dparse(s: &str) -> (String, Vec<f64>) {
        let (mut cmds, mut nums, mut cur) = (String::new(), vec![], String::new());
        let flush = |cur: &mut String, nums: &mut Vec<f64>| { if !cur.is_empty() { nums.push(cur.parse().unwrap()); cur.clear(); } };
        let b: Vec<char> = s.chars().collect();
        for (i, &c) in b.iter().enumerate() {
            if c.is_ascii_alphabetic() && c != 'e' { flush(&mut cur, &mut nums); cmds.push(c); }
            else if c == ' ' || c == ',' { flush(&mut cur, &mut nums); }
            else if c == '-' && !cur.is_empty() && !(i > 0 && b[i - 1] == 'e') { flush(&mut cur, &mut nums); cur.push(c); }
            else { cur.push(c); }
        }
        flush(&mut cur, &mut nums);
        (cmds, nums)
    }
    fn dash_of(v: &[f64]) -> String { format!("{v:?}") }
    fn el_key(e: &El) -> (String, Vec<f64>) {
        match e {
            El::Line { p, c, a, sw, dash } => (format!("line {:?} {sw} {}", col(*c, *a), dash_of(dash)), p.to_vec()),
            El::Rect { r, rx, fill, stroke, sw, dash } => (format!("rect {:?} {:?}{}", pc(fill), pc(stroke), if stroke.is_some() { format!(" {sw} {}", dash_of(dash)) } else { String::new() }), vec![r[0], r[1], r[2], r[3], *rx]),
            El::Path { d, fill, stroke, sw, dash } => { let (c, n) = dnums(d); (format!("path {c} {:?} {:?}{}", pc(fill), pc(stroke), if stroke.is_some() { format!(" {sw} {}", dash_of(dash)) } else { String::new() }), n) }
            El::Circle { p, fill, stroke, sw } => (format!("circle {:?} {:?}{}", pc(fill), pc(stroke), if stroke.is_some() { format!(" {sw}") } else { String::new() }), p.to_vec()),
            El::Text { x, y, s, z, wt, mono, anchor, c, a, halo, ts } => (format!("text {s} {z} {} {mono} {} {:?} {halo} {:?}", wb(*wt as f64, *mono), anc(*anchor), col(*c, *a), ts.as_ref().map(|t| (t.0.clone(), wb(t.1 as f64, *mono), col(t.2, t.3)))), vec![*x, *y]),
        }
    }
    fn el_exp(e: &Value) -> (String, Vec<f64>) {
        let at = &e["at"];
        let a = |k: &str| at[k].as_str().map_or(0.0, |v| v.parse::<f64>().unwrap());
        let cc = |k: &str, o: &str| { if e[k].is_null() { return [0.0; 4]; } let c = r4(&e[k]); let v = pmc([c[0], c[1], c[2], c[3] * f(&e[o]) * f(&e["op"])]); if v[3] == 0.0 { [0.0; 4] } else { v } };
        let dash: Vec<f64> = match e["dash"].as_str().unwrap_or("none") { "none" => vec![], d => d.split(',').map(|x| x.trim().trim_end_matches("px").parse().unwrap()).collect() };
        let sk = |stroke: [f64; 4]| if stroke == [0.0; 4] { String::new() } else { format!(" {} {}", f(&e["sw"]), dash_of(&dash)) };
        match e["tag"].as_str().unwrap() {
            "line" => (format!("line {:?} {} {}", cc("stroke", "so"), f(&e["sw"]), dash_of(&dash)), vec![a("x1"), a("y1"), a("x2"), a("y2")]),
            "rect" => { let s0 = cc("stroke", "so"); (format!("rect {:?} {:?}{}", cc("fill", "fo"), s0, sk(s0)), vec![a("x"), a("y"), a("width"), a("height"), a("rx")]) }
            "path" => { let s0 = cc("stroke", "so"); let (c, n) = dparse(at["d"].as_str().unwrap()); (format!("path {c} {:?} {:?}{}", cc("fill", "fo"), s0, sk(s0)), n) }
            "polyline" => { let s0 = cc("stroke", "so"); let (_, n) = dparse(at["points"].as_str().unwrap()); let c: String = (0..n.len() / 2).map(|k| if k == 0 { 'M' } else { 'L' }).collect(); (format!("path {c} {:?} {:?}{}", cc("fill", "fo"), s0, sk(s0)), n) }
            "circle" => { let s0 = cc("stroke", "so"); (format!("circle {:?} {:?}{}", cc("fill", "fo"), s0, if s0 == [0.0; 4] { String::new() } else { format!(" {}", f(&e["sw"])) }), vec![a("cx"), a("cy"), a("r")]) }
            "text" => {
                let mono = e["mono"] == true;
                let ts = e["ts"].as_array().and_then(|v| v.first()).map(|t| { let c = r4(&t["fill"]); (t["s"].as_str().unwrap().to_string(), wb(f(&t["wt"]), mono), pmc([c[0], c[1], c[2], c[3] * f(&t["fo"])])) });
                (format!("text {} {} {} {mono} {} {:?} {} {:?}", e["s"].as_str().unwrap(), f(&e["size"]), wb(f(&e["wt"]), mono), e["anchor"].as_str().unwrap(), cc("fill", "fo"), e["po"].as_str().unwrap_or("").starts_with("stroke"), ts), vec![a("x"), a("y")])
            }
            t => panic!("{t}"),
        }
    }
    fn expected(d: &Value) -> (Vec<Q>, Vec<Vec<(String, Vec<f64>)>>) {
        let (mut out, mut svgs) = (vec![], vec![]);
        for t in d["texts"].as_array().unwrap() { let mono = t["mono"] == true; out.push(Q { s: t["s"].as_str().unwrap().trim().into(), z: f(&t["size"]), wt: wb(f(&t["wt"]), mono), mono, ..q('t', r4(&t["r"]), pmc(r4(&t["c"])), kk(&t["k"])) }); }
        for b in d["boxes"].as_array().unwrap() { let r = r4(&b["r"]); out.push(Q { rad: rad(&b["rad"], r), ..q('b', r, pmc(r4(&b["c"])), kk(&b["k"])) }); }
        for l in d["lines"].as_array().unwrap() {
            let r = r4(&l["r"]);
            out.push(if l["side"] == "all" { Q { lw: f(&l["w"]), rad: if l["style"] == "outline" { 6.0 } else { rad(&l["rad"], r) }, ..q('o', r, pmc(r4(&l["c"])), kk(&l["k"])) } } else { q('b', r, pmc(r4(&l["c"])), kk(&l["k"])) });
        }
        for g in d["rings"].as_array().unwrap() { assert!(g["inset"] == true); let r = r4(&g["r"]); out.push(Q { lw: f(&g["spread"]), rad: rad(&g["rad"], r), ..q('o', r, pmc(r4(&g["c"])), kk(&g["k"])) }); }
        for s0 in d["svgs"].as_array().unwrap() {
            let vb = r4(&s0["vb"]);
            out.push(Q { s: format!("{:?}", [vb[2], vb[3]]), ..q('s', r4(&s0["r"]), [0.0; 4], kk(&s0["k"])) });
            svgs.push(s0["els"].as_array().unwrap().iter().map(el_exp).collect());
        }
        (out, svgs)
    }
    fn actual(items: &[It]) -> (Vec<Q>, Vec<Vec<(String, Vec<f64>)>>) {
        let (mut out, mut svgs) = (vec![], vec![]);
        for it in items {
            match it {
                It::T { x, y, s, z, wt, mono, c, a, k, .. } => if !s.trim().is_empty() { out.push(Q { s: s.trim().into(), z: *z, wt: wb(*wt as f64, *mono), mono: *mono, ..q('t', [*x, *y, 0.0, 0.0], col(*c, *a), *k) }) },
                &It::B { x, y, w, h, c, a, rad, k } => out.push(Q { rad, ..q('b', [x, y, w, h], col(c, a), k) }),
                &It::O { x, y, w, h, lw, c, a, rad, k } => out.push(Q { lw, rad, ..q('o', [x, y, w, h], col(c, a), k) }),
                It::S { x, y, w, h, vb, k, els } => { out.push(Q { s: format!("{vb:?}"), ..q('s', [*x, *y, *w, *h], [0.0; 4], *k) }); svgs.push(els.iter().map(el_key).collect()); }
            }
        }
        (out, svgs)
    }
    const TOL: f64 = 0.07;
    fn same(a: &Q, e: &Q) -> bool {
        let n = |x: f64, y: f64, t: f64| (x - y).abs() <= t;
        let kq = match (a.kk, e.kk) { (None, None) => true, (Some(p), Some(q)) => (0..4).all(|i| n(p[i], q[i], 1.0)), _ => false };
        kq && a.k == e.k && (0..4).all(|k| (a.c[k] - e.c[k]).abs() <= 1.0) && match a.k {
            't' => a.s == e.s && a.z == e.z && a.wt == e.wt && a.mono == e.mono && n(a.r[0], e.r[0], 0.6) && n(a.r[1], e.r[1], TOL),
            'o' => n(a.lw, e.lw, 1e-9) && n(a.rad, e.rad, TOL) && (0..4).all(|k| n(a.r[k], e.r[k], TOL)),
            'b' => n(a.rad, e.rad, TOL) && (0..4).all(|k| n(a.r[k], e.r[k], TOL)),
            _ => a.s == e.s && (0..4).all(|k| n(a.r[k], e.r[k], TOL)),
        }
    }
    fn show(v: &[Q]) -> String { v.iter().map(|q| format!("{} {:?} {:?} {:?} z{} w{}{} lw{} rad{} k{:?}", q.k, q.s.chars().take(60).collect::<String>(), q.r.map(|x| (x * 1000.0).round() / 1000.0), q.c, q.z, q.wt, if q.mono { " M" } else { "" }, q.lw, q.rad, q.kk)).collect::<Vec<_>>().join("\n") }
    fn input<'a>(s0: &'a Value, paths: &'a [(String, Value)]) -> In<'a> {
        In { now: f(&s0["now"]), acct: Some(&s0["acct"]).filter(|v| !v.is_null()), err: s0["err"].as_str().unwrap_or(""), span: s0["span"].as_u64().unwrap() as usize, back: s0["back"].as_i64().unwrap(),
             sel: s0["sel"].as_str(), tab: s0["tab"].as_str().unwrap(), cytab: s0["cytab"].as_str().unwrap(), paths }
    }
    fn untnum(s0: &str) -> String { s0.chars().map(|c| match c as u32 { u @ 0xE071..=0xE07A => char::from_digit(u - 0xE071, 10).unwrap(), _ => c }).collect() }

    /// 지금 화면의 실제 코드를 원본 styles.css 로 헤드리스 크로미움에서 돌려 잰 것(compare/acc_layout.py) -- 글·칠·테·자르기·svg 요소가 전부 하나씩 짝지어져야 한다
    #[test]
    fn js_parity() {
        let inp = fx_("acc_in.json");
        let js = std::env::var("ACC_JS").map(|p| serde_json::from_str::<Value>(&std::fs::read_to_string(p).unwrap()).unwrap()).unwrap_or_else(|_| fx_("acc_js.json"));
        let cases = js["cases"].as_array().unwrap();
        let clean: Vec<Value> = cases.iter().map(|c| { let mut c = c.clone(); c["texts"] = Value::Array(c["texts"].as_array().unwrap().iter().filter(|t| t["synth"] != true).cloned().collect()); c }).collect();
        let refs: Vec<&Value> = clean.iter().collect();
        let lk = lookup(&refs, &js["probe"]);
        let m = move |s0: &str, z: f64, wt: u16, mono: bool| lk(&untnum(s0), z, wt, mono);
        let (mut n, mut nums) = (0, 0);
        let only: Option<usize> = std::env::var("ACC_CASE").ok().and_then(|v| v.parse().ok());
        let mut fails = vec![];
        for (k, (s0, d)) in inp.as_array().unwrap().iter().zip(cases).enumerate() {
            if only.is_some_and(|o| o != k) { continue; }
            let paths: Vec<(String, Value)> = s0["paths"].as_object().unwrap().iter().map(|(a, b)| (a.clone(), b.clone())).collect();
            let o = body(&input(s0, &paths), f(&d["w"]), f(&d["h"]), &m);
            let (act, sa) = actual(&o.items);
            let (exp, se) = expected(d);
            let mut used = vec![false; act.len()];
            let (mut miss, mut ps, mut si) = (vec![], vec![], 0);
            let idx: Vec<usize> = { let mut c = 0; act.iter().map(|a| if a.k == 's' { c += 1; c - 1 } else { usize::MAX }).collect() };
            for e in &exp {
                match (0..act.len()).find(|&j| !used[j] && same(&act[j], e)) { Some(j) => { used[j] = true; if e.k == 's' { ps.push((idx[j], si)); } } None => miss.push(e.clone()) }
                if e.k == 's' { si += 1; }
            }
            let extra: Vec<Q> = act.iter().zip(&used).filter(|q| !q.1).map(|q| q.0.clone()).collect();
            if !(extra.is_empty() && miss.is_empty()) { fails.push(format!("입력 {k}\n남은 rtui {}\n{}\n못 찾은 원본 {}\n{}", extra.len(), show(&extra), miss.len(), show(&miss))); continue; }
            for (a, e) in ps {
                assert_eq!(sa[a].len(), se[e].len(), "입력 {k} svg {e} 요소 수\n{:?}\n{:?}", sa[a].iter().map(|x| &x.0).collect::<Vec<_>>(), se[e].iter().map(|x| &x.0).collect::<Vec<_>>());
                for (x, y) in sa[a].iter().zip(&se[e]) {
                    assert!(x.0 == y.0 && x.1.len() == y.1.len() && x.1.iter().zip(&y.1).all(|(p, q)| (p - q).abs() < 1e-6), "입력 {k} svg {e}\n{x:?}\n{y:?}");
                    nums += y.1.len();
                }
            }
            n += exp.len();
        }
        assert!(fails.is_empty(), "{}", fails.join("\n\n"));
        if only.is_none() { assert!(n > 1500 && nums > 6000, "{n} {nums}"); }
        eprintln!("acc js_parity: 단위 {n} · svg 수 {nums}");
    }

    /// JS 수 글(acctSu · acctSuc · acctPct · acctDur · acctPx · fmtUsd · cycleQ · toLocaleString) · KST 날짜
    #[test]
    fn js_helpers() {
        assert_eq!([su(24.316), su(-0.004), su(1234.5), su(-99.999)], ["+$24.32", "−$0", "+$1,235", "−$100"]);
        assert_eq!([suc(-607.4), suc(4999.0), suc(0.0), suc(-1050.0)], ["−$607", "+$5.0k", "+$0", "−$1.1k"]);
        assert_eq!([pct(Some(0.0104)), pct(Some(-0.123)), pct(None), pct(Some(f64::NAN)), pct(Some(-0.0))], ["+1.0%", "−12%", "—", "—", "+0.0%"]);
        assert_eq!([dur(59.4 * 60e3), dur(61.0 * 60e3), dur(1440.0 * 60e3), dur(1500.0 * 60e3)], ["59분", "1시간 01분", "1.0일", "1.0일"]);
        assert_eq!([px(1.3007), px(3150.2), px(91800.0)], ["1.3007", "3,150.20", "91,800.00"]);
        assert_eq!([usd(2532.4), usd(-3.21), usd(9.87654)], ["$2,532.4", "-$3.21", "$9.8765"]);
        assert_eq!([cq(0.85), cq(12.0), cq(0.123456)], ["0.85", "12", "0.1235"]);
        assert_eq!([key(1795000000000.0), hm(1795000000000.0), md(1795000000000.0)], ["2026-11-18", "20:06", "11/18"]);
        assert_eq!(day0("2026-11-18"), 1794927600000.0);
        assert_eq!(key(1794927600000.0 - 1.0), "2026-11-17", "KST 0시 경계");
    }

    /// 받기·누르기: 10초 · 숨으면 안 묻는다 · 실패 = 서버 last_good 또는 30분 안 마지막 정상값 · 순환 가격선 60초 · 날 다시 누르면 오늘
    #[test]
    fn poll_click() {
        let gets = |o: &mut Vec<Req>| o.drain(..).map(|r| match r { Req::Get { kind, url } => (kind, url), _ => panic!() }).collect::<Vec<_>>();
        let mut s0 = Store::default();
        let mut o = vec![];
        s0.tick(1e9, true, "", &mut o);
        assert!(o.is_empty(), "숨으면 안 묻는다");
        s0.tick(1e9, false, "", &mut o);
        s0.tick(1e9 + 9_000.0, false, "", &mut o);
        assert_eq!(gets(&mut o), [(Kind::AccAcct, "/api/binance-account".to_string())]);
        s0.on_acct(Some(r#"{"ok":true,"ledger":[]}"#), 1e9);
        s0.on_acct(Some(r#"{"ok":false,"error":"HTTP 502"}"#), 1e9 + 60_000.0);
        assert!(s0.acct.is_none() && s0.stale.is_some() && s0.err == "HTTP 502", "실패 = 마지막 정상값으로 그림");
        assert!(s0.input(0.0).acct.is_some());
        s0.on_acct(None, 1e9 + 1_900_000.0);
        assert!(s0.input(0.0).acct.is_none(), "30분 넘으면 «불러오지 못함»");
        s0.on_acct(Some(r#"{"ok":false,"last_good":{"ok":true,"age_s":5}}"#), 1e9);
        assert!(s0.input(0.0).acct.is_some(), "서버 last_good");
        let w = vec![("ETHUSDC|LONG|5".to_string(), "ETHUSDC".to_string(), 1.0)];
        s0.want(&w, 1e9, "", &mut o);
        s0.want(&w, 1e9 + 30_000.0, "", &mut o);
        assert_eq!(gets(&mut o), [(Kind::AccPath, "/api/cycle-path?symbol=ETHUSDC&start=1".to_string())], "60초에 한 번");
        s0.on_path(Some(r#"{"ok":true,"rows":[[1,2]]}"#));
        assert_eq!(s0.paths, [("ETHUSDC|LONG|5".to_string(), serde_json::json!([[1, 2]]))]);
        s0.hits = vec![([0.0, 0.0, 10.0, 10.0], Hit::Day("2026-11-12".into())), ([20.0, 0.0, 10.0, 10.0], Hit::Month(1)), ([40.0, 0.0, 10.0, 10.0], Hit::Tab("cal"))];
        let now = 1795000000000.0;
        assert!(s0.click(5.0, 5.0, now) && s0.sel.as_deref() == Some("2026-11-12"));
        assert!(s0.click(5.0, 5.0, now) && s0.sel.is_none(), "같은 날 다시 = 오늘");
        assert!(s0.click(25.0, 5.0, now) && s0.back == 1);
        assert!(s0.click(45.0, 5.0, now) && s0.tab == "cal");
        assert!(!s0.click(100.0, 5.0, now));
    }
}
