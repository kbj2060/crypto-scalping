//! 풋프린트 머리 작은 칸(?card=head) -- 1920×1080 넓은 화면(html.fit1 · 카드 레일 보임 · ≥1700) 맨 위 줄과 풋프린트 카드 머리의 네 덩이:
//!   세션(buildSessionHtml → #topSession · 맨 위 왼쪽) · 코인 이름(renderSnapshotAssetTabs #fpCoinName · 맨 위 가운데 · 표시만) ·
//!   발표 배지·급변 칩(renderMacroEventAlert · macroSpikeChips → .session-alert-center · 코인 탭 왼쪽) · 머리 포지션 요약(renderFpPos · posMarginShare → #fpPos · 카드 머리 정중앙 · 표시만).
//! 자리 = 페이지 좌표 그대로(카드 상자 0,0 = 페이지 왼쪽 위 · 1920×136) -- 덩이마다 원본 styles.css 를 헤드리스 크로미움에서 잰 값(compare/head_js.mjs → head_layout.py → 시험 js_parity).
//! 자료: /api/state 5초(세션 -- 원본은 SSE /api/events 밀기 ~2.5초 · 접착에 SSE 가 없다) · /api/session-alerts 30초(실패 = 배지 숨김) · /api/macro-spikes 30초(실패 = 받아 둔 값) ·
//!   /api/binance-account 10초 + 허브 «모든 종목» 가격(feed acct_only 몫 그대로).
//! 안 옮긴 것: 툴팁(title) · 코인 탭 누르기(코인 이름은 글만) · 세션 받기 전 «-» · 좁은 화면 배치 · 휴대폰 «다른 포지션» 칩.
//! ponytail: 경보 칸 폭 = 코인 탭(고정 칩 4개 · 기본 고정 eth·sol·xrp·sndk)과 나눠 줄어드는 flex 를 상수로 -- 지금 코인이 고정 밖이면 칩이 하나 늘어 실제보다 넓게 잡는다(탭을 옮기면 그 폭을 넘긴다).
//! ponytail: 상태는 이 파일의 thread_local(웹어셈블리 = 한 스레드) -- 공용 Feed 구조체를 안 건드리려고.
use crate::fp::Col;
use crate::mt::{cont, fixed, It, Measure};
use serde_json::Value;
use std::cell::RefCell;
use std::collections::HashMap;

pub const W: f64 = 1920.0;
pub const H: f64 = 136.0;
/// 페이지 자리(1920×1080 · 잰 값): 맨 위 줄 왼쪽·폭·가운데 높이 · 카드 머리 가운데
const TOP_L: f64 = 96.0;
const TOP_W: f64 = 1800.0;
const TOP_CY: f64 = 32.0;
const HEAD_CX: f64 = 996.0;
const HEAD_CY: f64 = 107.0;
/// .snap-head 에서 경보 칸·코인 탭 말고 고정 폭: 간격 20×4 + «전체» 단추 67.03125 + 다크·알림 66×2 − «전체» 왼쪽 여백 12
const SNAP_FIXED: f64 = 267.03125;
const TABS_W: f64 = 230.0;
const TABS_MIN: f64 = 110.0;

fn t(v: &mut Vec<It>, x: f64, y: f64, s: impl Into<String>, size: f64, wt: u16, mono: bool, ls: f64, c: Col, a: f64) { v.push(It::T { x, y, s: s.into(), size, wt, mono, ls, c, a, k: false }); }
fn b(v: &mut Vec<It>, x: f64, y: f64, w: f64, h: f64, c: Col, a: f64, rad: f64) { v.push(It::B { x, y, w, h, c, a, rad, k: false }); }
fn tw(m: Measure, s: &str, z: f64, wt: u16, mono: bool, ls: f64) -> f64 { m(s, z, wt, mono) + ls * s.chars().count() as f64 }
/// 줄 상자 위 → 글 상자 위(반 행간 내림)
fn gt(top: f64, lh: f64, z: f64, mono: bool) -> f64 { top + ((lh - cont(z, mono)) / 2.0).floor() }
fn mv(v: Vec<It>, dx: f64, dy: f64) -> impl Iterator<Item = It> {
    v.into_iter().map(move |it| match it {
        It::T { x, y, s, size, wt, mono, ls, c, a, k } => It::T { x: x + dx, y: y + dy, s, size, wt, mono, ls, c, a, k },
        It::B { x, y, w, h, c, a, rad, k } => It::B { x: x + dx, y: y + dy, w, h, c, a, rad, k },
        o => o,
    })
}

// ── JS 수 ──
/// Number(v) -- 없음·null = 0(쓰임이 전부 `|| 0`·`> 0` 뒤)
fn num(v: &Value) -> f64 {
    match v { Value::Number(n) => n.as_f64().unwrap_or(f64::NAN), Value::String(s) => { let s = s.trim(); if s.is_empty() { 0.0 } else { s.parse().unwrap_or(f64::NAN) } } Value::Bool(b) => *b as u8 as f64, Value::Null => 0.0, _ => f64::NAN }
}
fn or0(x: f64) -> f64 { if x.is_nan() { 0.0 } else { x } }
/// toLocaleString("en-US", 소수 d 고정)
fn loc(x: f64, d: usize) -> String {
    let s = fixed(x.abs(), d);
    let (ip, fp) = s.split_once('.').map_or((s.as_str(), ""), |(a, b)| (a, b));
    let mut o = String::new();
    for (k, c) in ip.chars().enumerate() { if k > 0 && (ip.len() - k) % 3 == 0 { o.push(','); } o.push(c); }
    format!("{}{o}{}", if x < 0.0 { "-" } else { "" }, if fp.is_empty() { String::new() } else { format!(".{fp}") })
}
/// 443 lineFmt(lineDig: 100 이상 2자리 · 아래 4자리)
fn line_fmt(v: f64) -> String { loc(v, if v >= 100.0 { 2 } else { 4 }) }
fn sg(v: f64) -> &'static str { if v >= 0.0 { "+" } else { "−" } }

// ── 세션 ──
/// 2132 buildSessionHtml -- 아시아·유럽·미국 Number(x || 0) >= 0.5
pub fn sess_on(s: &Value) -> [bool; 3] { ["session_asia", "session_europe", "session_us"].map(|k| or0(num(&s[k])) >= 0.5) }
/// (그림, 폭, 높이) -- .top-session flex 간격 8 · 항목 = 점 10 + 6 + 글(mono 700 11) · 구분 «|»(16px · 좌우 여백 8)
pub fn session(on: [bool; 3], m: Measure) -> (Vec<It>, f64, f64) {
    let ih = cont(11.0, true);
    let h = ih.max(cont(16.0, false));
    let (iy, mut x, mut v) = ((h - ih) / 2.0, 0.0, vec![]);
    for (k, (nm, on)) in ["아시아", "유럽", "미국"].into_iter().zip(on).enumerate() {
        if k > 0 {
            x += 8.0 + 8.0; // 간격 + 구분 왼쪽 여백
            t(&mut v, x, gt(0.0, h, 16.0, false), "|", 16.0, 400, false, 0.0, Col::Line, 1.0);
            x += tw(m, "|", 16.0, 400, false, 0.0) + 8.0 + 8.0;
        }
        if on { b(&mut v, x, iy + (ih - 10.0) / 2.0, 10.0, 10.0, Col::Good, 1.0, 5.0) } else { b(&mut v, x, iy + (ih - 10.0) / 2.0, 10.0, 10.0, Col::White, 0.22, 5.0) }
        t(&mut v, x + 16.0, iy, nm, 11.0, 700, true, 0.0, if on { Col::Text } else { Col::Muted }, 1.0);
        x += 16.0 + tw(m, nm, 11.0, 700, true, 0.0);
    }
    (v, x, h)
}

// ── 코인 이름 ──
/// 1831 #fpCoinSym = 코인 대문자 · .fp-coin-name 800 30px/1 · 자간 .06em
pub fn coin(asset: &str, m: Measure) -> (Vec<It>, f64, f64) {
    let s = asset.to_uppercase();
    let w = tw(m, &s, 30.0, 800, false, 1.8);
    let mut v = vec![];
    t(&mut v, 0.0, gt(0.0, 30.0, 30.0, false), s, 30.0, 800, false, 1.8, Col::Ink, 1.0);
    (v, w, 30.0)
}

// ── 발표 배지 · 급변 칩 ──
/// 3145 renderMacroEventAlert(macro_event_alert) -- active 가 비어 있지 않으면 보인다
pub fn alert_on(p: &Value) -> bool { p["active"].as_array().is_some_and(|a| !a.is_empty()) }
const SPIKE_SHOW_MIN: f64 = 40.0;
fn spike_nm(k: &str) -> &'static str { match k { "ZF" => "5년", "ZN" => "10년", _ => "나스닥" } }
fn jround(x: f64) -> f64 { (x + 0.5).floor() }
/// 3740 macroSpikeChips -- (색, 글) 최근 것부터 · now = 서버 시각(s)
pub fn spike_chips(spk: Option<&Value>, now: f64) -> Vec<(Col, String)> {
    let Some(s) = spk.filter(|s| s["ok"] == true).map(|s| &s["symbols"]) else { return vec![] };
    let last = |k: &str| s[k]["events"].as_array().and_then(|e| e.iter().filter(|x| num(&x[0]) + 300.0 >= now - SPIKE_SHOW_MIN * 60.0).last()).map(|e| [num(&e[0]), num(&e[1]), num(&e[2])]);
    let ago = |e: &[f64; 3]| jround((now - e[0] - 300.0) / 60.0).max(0.0);
    let cl = |r: f64| if r < 0.0 { Col::Bad } else { Col::Good };
    let mut out: Vec<(f64, Col, String)> = vec![];
    if let Some(nq) = last("NQ") { out.push((ago(&nq), cl(nq[2]), format!("나스닥 {}{}% · {}분", if nq[2] >= 0.0 { "+" } else { "−" }, fixed(nq[2].abs(), 2), ago(&nq)))); }
    let rs: Vec<(&str, [f64; 3])> = ["ZF", "ZN"].into_iter().filter_map(|k| last(k).map(|e| (k, e))).collect();
    let sign = |x: f64| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { x };
    let same = rs.len() == 2 && sign(rs[0].1[2]) == sign(rs[1].1[2]);
    let groups: Vec<&[(&str, [f64; 3])]> = if same { vec![&rs[..]] } else { rs.chunks(1).collect() };
    for g in groups {
        let e = g[0].1;
        let a = g.iter().map(|q| ago(&q.1)).fold(f64::INFINITY, f64::min);
        let names = g.iter().map(|q| spike_nm(q.0).replace('년', "")).collect::<Vec<_>>().join("·");
        out.push((a, cl(e[2]), format!("{names}년 금리{} · {a}분", if e[2] < 0.0 { "↑" } else { "↓" })));
    }
    out.sort_by(|p, q| p.0.total_cmp(&q.0));
    out.into_iter().map(|q| (q.1, q.2)).collect()
}
/// 경보 칸 폭 계산 재료(.snap-head flex): l0 = 칸 왼쪽(줄어들 때) · s = 칸 + 코인 탭이 나눠 쓰는 폭 · tb·tmin = 탭 기준·최소 폭
#[derive(Clone, Copy, Debug)]
pub struct Room { pub l0: f64, pub s: f64, pub tb: f64, pub tmin: f64 }
/// 맨 위 그리드(1fr auto 1fr · 간격 20): 가운데 칸 = 코인 이름 폭 cw
pub fn room(cw: f64) -> Room {
    let col = (TOP_W - 40.0 - cw) / 2.0;
    Room { l0: TOP_L + col + 20.0 + cw + 20.0, s: col - SNAP_FIXED, tb: TABS_W, tmin: TABS_MIN }
}
/// .session-alert-center(flex · 간격 8 · 줄바꿈 · 높이 32 넘침 숨김) -- 배지(경보 주황 · 맥박 = pulse 불투명) + 칩 → (그림, 칸 왼쪽, 폭, 높이)
pub fn alerts(badge: bool, chips: &[(Col, String)], r: Room, pulse: f64, m: Measure) -> (Vec<It>, f64, f64, f64) {
    let mut all: Vec<(Col, &str, f64)> = vec![];
    if badge { all.push((Col::Amber, "⚠️ 발표 변동성 주의", pulse)); }
    all.extend(chips.iter().map(|c| (c.0, c.1.as_str(), 1.0)));
    if all.is_empty() { return (vec![], r.l0 + r.s - r.tb, 0.0, 0.0); }
    let ws: Vec<f64> = all.iter().map(|c| tw(m, c.1, 11.0, 700, false, 0.0) + 26.0).collect();
    let ma = ws.iter().sum::<f64>() + 8.0 * (ws.len() - 1) as f64;
    // flex 줄임: 넘친 만큼 «안쪽» 기준 폭 비례로 칸·탭이 나눈다(탭 안쪽 = 바깥 − 안쪽 여백 3×2 · 탭은 최소에서 멈춤)
    let (w, x0) = if ma + r.tb <= r.s { (ma, r.l0 + r.s - r.tb - ma) } else {
        let (o, ti) = (ma + r.tb - r.s, r.tb - 6.0);
        let tt = r.tb - o * ti / (ma + ti);
        (if tt < r.tmin { r.s - r.tmin } else { ma - o * ma / (ma + ti) }, r.l0)
    };
    let (mut v, mut x) = (vec![], 0.0);
    for (k, (c, s, a)) in all.iter().enumerate() {
        if k > 0 && x + ws[k] > w + 1e-9 { break; } // 둘째 줄 = 숨음
        b(&mut v, x, 0.0, ws[k], 32.0, *c, *a, 999.0);
        t(&mut v, x + 13.0, gt(10.5, 11.0, 11.0, false), *s, 11.0, 700, false, 0.0, Col::OnFill, *a);
        x += ws[k] + 8.0;
    }
    (v, x0, w, 32.0)
}

// ── 머리 포지션 요약 ──
/// 970 armedBasis
fn armed_basis(b: &Value) -> f64 {
    let f = |k| num(&b[k]);
    if f("sl_price") > 0.0 && f("sl_level") > 0.0 { f("sl_price") / f("sl_level") } else if f("tp_price") > 0.0 && f("tp_level") > 0.0 { f("tp_price") / f("tp_level") } else { or0(f("basis")) }
}
fn up(p: &Value) -> String { p["side"].as_str().map_or_else(|| p["side"].to_string(), str::to_string).to_uppercase() }
/// 2423 entryUsdt
fn entry_usdt(a: &Value, pos: &Value, px: f64) -> f64 {
    let en = num(&pos["entry_price"]);
    let sym = pos["symbol"].as_str().unwrap_or("");
    if !(en > 0.0) || !sym.ends_with("USDC") { return en; }
    let b = &a["bracket_armed"][format!("{sym}:{}", up(pos))];
    let (mk, bb) = (num(&pos["mark_price"]), if b.is_object() { armed_basis(b) } else { 0.0 });
    en / if bb > 0.0 { bb } else if mk > 0.0 && px > 0.0 { mk / px } else { 1.0 }
}
/// 한 줄씩 글(값만 -- 자리는 pos 가)
#[derive(Debug, PartialEq)]
pub struct Pos { pub short: bool, pub pnl: String, pub pct: String, pub entry: String, pub margin: Option<String>, pub good: bool }
/// 2439 renderFpPos(snapshotAccountPosition · execSymbolFor · posMarginShare) -- None = 숨김
pub fn fp_pos(a: Option<&Value>, prices: &HashMap<String, f64>, asset: &str) -> Option<Pos> {
    let a = a?;
    let market = crate::asset_cfg(asset).map_or_else(|| format!("{}USDT", asset.to_uppercase()), |c| c.0.to_string());
    let ex = &a["exec_symbols"][asset];
    let exec = if ex.as_str().is_some_and(|s| !s.is_empty()) { ex.as_str().unwrap().to_uppercase() } else if asset == "eth" { a["exec_symbol"].as_str().unwrap_or("").to_uppercase() } else { String::new() };
    let base = market.trim_end_matches("USDT");
    let order = if !exec.is_empty() && exec != market && exec.starts_with(base) { vec![exec, market] } else { vec![market] };
    let ps = a["positions"].as_array()?;
    let pos = order.iter().find_map(|s| ps.iter().find(|p| p["symbol"].as_str() == Some(s.as_str())))?;
    let cur = prices.get(asset).copied().unwrap_or(0.0);
    let qty = num(&pos["qty"]).abs();
    let en = entry_usdt(a, pos, cur);
    if !(qty > 0.0 && en > 0.0 && cur > 0.0) { return None; }
    let sgn = if up(pos) == "SHORT" { -1.0 } else { 1.0 };
    let (pct, pnl) = ((cur - en) / en * 100.0 * sgn, (cur - en) * qty * sgn);
    // posMarginShare
    let q: String = { let s: Vec<char> = pos["symbol"].as_str().unwrap_or("").chars().collect(); s[s.len().saturating_sub(4)..].iter().collect() };
    let bal = if a["multi_assets_margin"].as_bool().unwrap_or(false) || a["multi_assets_margin"].as_f64().is_some_and(|x| x != 0.0) { a["balance"].clone() } else {
        Some(&a["balances"][q.as_str()]).filter(|v| v.is_object()).cloned().or_else(|| a["assets"].as_array().and_then(|v| v.iter().find(|x| x["asset"].as_str() == Some(q.as_str())).cloned())).unwrap_or(Value::Null)
    };
    let eq = { let m = num(&bal["margin"]); if m != 0.0 && !m.is_nan() { m } else { or0(num(&bal["wallet"])) + or0(num(&bal["unrealized"])) } };
    let lev = num(&pos["leverage"]);
    let im = if lev > 0.0 { or0(num(&pos["notional"])).abs() / lev } else { 0.0 };
    let mp = (eq > 0.0 && im > 0.0).then(|| im / eq * 100.0);
    Some(Pos { short: sgn < 0.0, pnl: format!("{}${}", sg(pnl), line_fmt(pnl.abs())), pct: format!("{}{}%", sg(pct), fixed(pct.abs(), 2)), entry: format!("진입 {}", line_fmt(en)),
               margin: mp.map(|p| format!("{}%", fixed(p, if p >= 10.0 { 0 } else { 1 }))), good: pnl >= 0.0 })
}
/// .fp-pos(세로 flex 간격 4 · 줄 = 가로 flex 바닥선 맞춤 간격 8 -- `.fp-pos > div` 가 .fp-pnl-row·.fp-sub 의 10 을 이긴다) → (그림, 폭, 높이)
pub fn pos(p: &Pos, m: Measure) -> (Vec<It>, f64, f64) {
    let c = if p.good { Col::Good } else { Col::Bad };
    let cc = if p.short { Col::Bad } else { Col::Good };
    let lab = if p.short { "SHORT" } else { "LONG" };
    let mut v = vec![];
    // 1줄: 칩(10.5 800 · 자간 .08em · 안쪽 3·6 · 가운데) + 손익(mono 800 26/1)
    let cw = tw(m, lab, 10.5, 800, false, 0.84) + 12.0;
    let cy = (26.0 - 16.5) / 2.0;
    b(&mut v, 0.0, cy, cw, 16.5, cc, 0.16, 4.0);
    t(&mut v, 6.0, gt(cy + 3.0, 10.5, 10.5, false), lab, 10.5, 800, false, 0.84, cc, 1.0);
    t(&mut v, cw + 8.0, gt(0.0, 26.0, 26.0, true), &p.pnl, 26.0, 800, true, 0.0, c, 1.0);
    let w1 = cw + 8.0 + tw(m, &p.pnl, 26.0, 800, true, 0.0);
    // 2줄: 수익률(14 700 / 1.2) · 진입(12 500 / 1.2) · 증거금(12 500 + 굵게 700) -- 바닥선 = 가장 높은 글의 위 반 행간 + 올림
    let (y2, lb, ls) = (26.0 + 4.0, 14.0 * 1.2, 12.0 * 1.2);
    let ob = gt(0.0, lb, 14.0, false) + crate::mt::asc(14.0, false);
    let os = gt(0.0, ls, 12.0, false) + crate::mt::asc(12.0, false);
    let base = ob.max(os);
    t(&mut v, 0.0, gt(y2 + base - ob, lb, 14.0, false), &p.pct, 14.0, 700, false, 0.0, c, 1.0);
    let mut x = tw(m, &p.pct, 14.0, 700, false, 0.0) + 8.0;
    let sy = gt(y2 + base - os, ls, 12.0, false);
    t(&mut v, x, sy, &p.entry, 12.0, 500, false, 0.0, Col::Muted, 1.0);
    x += tw(m, &p.entry, 12.0, 500, false, 0.0);
    if let Some(mg) = &p.margin {
        x += 8.0;
        t(&mut v, x, sy, "증거금 ", 12.0, 500, false, 0.0, Col::Muted, 1.0);
        x += tw(m, "증거금 ", 12.0, 500, false, 0.0);
        t(&mut v, x, sy, mg, 12.0, 700, false, 0.0, Col::Ink, 1.0);
        x += tw(m, mg, 12.0, 700, false, 0.0);
    }
    let h2 = (base - ob + lb).max(base - os + ls);
    (v, w1.max(x), y2 + h2)
}

// ── 한 장 ──
/// 받아 둔 응답(원본 전역 latestMainState.session · session-alerts · macroSpk·macroSpkSkew · latestBinanceAccount · latestLivePriceByAsset · activeSnapshotAsset) · now = ms
pub struct In<'a> { pub asset: &'a str, pub now: f64, pub sess: Option<[bool; 3]>, pub alert: bool, pub spk: Option<&'a Value>, pub skew: f64, pub acct: Option<&'a Value>, pub prices: &'a HashMap<String, f64> }
/// 덩이 넷을 페이지 자리에 -- (세션 · 코인 · 경보 · 포지션) 각 (그림, [x, y, 폭, 높이])
pub type Block = (Vec<It>, [f64; 4]);
pub fn blocks(i: &In, rm: Option<Room>, pulse: f64, m: Measure) -> [Option<Block>; 4] {
    let s = i.sess.map(|on| { let (v, w, h) = session(on, m); (v, [TOP_L, TOP_CY - h / 2.0, w, h]) });
    let (cv, cw, ch) = coin(i.asset, m);
    let c = (cv, [TOP_L + (TOP_W - 40.0 - cw) / 2.0 + 20.0, TOP_CY - ch / 2.0, cw, ch]);
    let chips = spike_chips(i.spk, i.now / 1000.0 + i.skew);
    let (av, ax, aw, ah) = alerts(i.alert, &chips, rm.unwrap_or_else(|| room(cw)), pulse, m);
    let a = (aw > 0.0).then(|| (av, [ax, TOP_CY - ah / 2.0, aw, ah]));
    let p = fp_pos(i.acct, i.prices, i.asset).map(|p| { let (v, w, h) = pos(&p, m); (v, [HEAD_CX - w / 2.0, HEAD_CY - h / 2.0, w, h]) });
    [s, Some(c), a, p]
}
pub fn card(i: &In, pulse: f64, m: Measure) -> Vec<It> {
    blocks(i, None, pulse, m).into_iter().flatten().flat_map(|(v, r)| mv(v, r[0], r[1])).collect()
}
/// statusPulse 1.4초(불투명 1 → .55 → 1 · ease-in-out ≈ smoothstep)
fn pulse(t_ms: f64) -> f64 { let p = (t_ms / 700.0).rem_euclid(2.0); let u = if p < 1.0 { p } else { 2.0 - p }; 1.0 - 0.45 * u * u * (3.0 - 2.0 * u) }

// ── 받기 ──
use crate::feed::{Kind, Req};
#[derive(Default)]
pub struct Store { pub sess: Option<[bool; 3]>, pub alert: bool, pub spk: Option<Value>, pub skew: f64, at: [f64; 3] }
const STATE_MS: f64 = 5000.0;
const ALERT_MS: f64 = 30_000.0; // 1682 SESSION_ALERTS_POLL_MS
const SPIKE_MS: f64 = 30_000.0; // refreshMacroSpikes
impl Store {
    pub fn tick(&mut self, now: f64, hidden: bool, api: &str, out: &mut Vec<Req>) {
        if hidden { return; }
        for (k, (kind, gap, path)) in [(Kind::HeadState, STATE_MS, "/api/state"), (Kind::HeadAlert, ALERT_MS, "/api/session-alerts"), (Kind::HeadSpk, SPIKE_MS, "/api/macro-spikes")].into_iter().enumerate() {
            if now - self.at[k] >= gap { self.at[k] = now; out.push(Req::Get { kind, url: format!("{api}{path}") }); }
        }
    }
    /// 실패: 세션·급변 = 받아 둔 값 · 경보 = 숨김(refreshSessionAlerts catch)
    pub fn on_get(&mut self, kind: Kind, body: Option<&str>, now: f64) {
        let v = body.and_then(|b| serde_json::from_str::<Value>(b).ok());
        match (kind, v) {
            (Kind::HeadState, Some(v)) => if v["state"]["session"].is_object() { self.sess = Some(sess_on(&v["state"]["session"])) },
            (Kind::HeadAlert, v) => self.alert = v.is_some_and(|v| alert_on(&v["macro_event_alert"])),
            (Kind::HeadSpk, Some(v)) => { if let Some(t) = v["now_sec"].as_f64() { self.skew = t - now / 1000.0; } self.spk = Some(v); }
            _ => {}
        }
    }
}
thread_local! { static ST: RefCell<Option<Store>> = const { RefCell::new(None) }; }
pub fn start() { ST.with(|s| *s.borrow_mut() = Some(Store { at: [f64::NEG_INFINITY; 3], ..Default::default() })); }
pub fn on() -> bool { ST.with(|s| s.borrow().is_some()) }
pub fn tick(now: f64, hidden: bool, api: &str, out: &mut Vec<Req>) { ST.with(|s| if let Some(st) = s.borrow_mut().as_mut() { st.tick(now, hidden, api, out); }); }
pub fn on_get(kind: &str, body: Option<&str>, now: f64) {
    let k = match kind { "HeadState" => Kind::HeadState, "HeadAlert" => Kind::HeadAlert, _ => Kind::HeadSpk };
    ST.with(|s| if let Some(st) = s.borrow_mut().as_mut() { st.on_get(k, body, now); });
}
/// 한 장(Chart::ui 에서) -- 상자 = 1920×136 페이지 자리 · 맥박이 있으면 자주, 아니면 1초마다(급변 «N분» 이 흐른다)
pub fn ui(ch: &mut crate::Chart, ui: &mut eframe::egui::Ui, now: f64) {
    let p = ui.painter();
    if ch.bg.a() > 0 { p.rect_filled(p.clip_rect(), 0.0, ch.bg); }
    let measure = |s: &str, z: f64, wt: u16, mono: bool| if mono { p.layout_no_wrap(s.to_owned(), crate::draw::mono(z as f32, wt), eframe::egui::Color32::WHITE).size().x as f64 } else { crate::draw::measure(p, s, z, wt) };
    let busy = ST.with(|s| {
        let b = s.borrow();
        let Some(st) = b.as_ref() else { return false };
        let i = In { asset: &ch.feed.asset, now, sess: st.sess, alert: st.alert, spk: st.spk.as_ref(), skew: st.skew, acct: ch.feed.acct.as_ref(), prices: &ch.feed.prices };
        let mut v = crate::mt::prims(&card(&i, pulse(now), &measure), &measure);
        for q in &mut v { if let crate::fp::Prim::Text { s, .. } = q { if s.contains('\u{fe0f}') { *s = s.replace('\u{fe0f}', ""); } } } // ⚠️ 의 글꼴 고르개(egui 는 네모로 그린다)
        crate::draw::paint(p, &v, ch.origin, [0.0, 0.0, W, H], [0.0, 0.0, W, H], 1.0);
        st.alert
    });
    ui.ctx().request_repaint_after(std::time::Duration::from_millis(if busy { 50 } else { 1000 }));
}

#[cfg(test)]
mod t {
    use super::*;
    use crate::mt::t::{actual, expected, lookup, pair, show, TOL};

    fn fx(f: &str) -> Value { serde_json::from_str(&std::fs::read_to_string(format!("{}/compare/fixture/{f}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap() }
    fn nn(v: &Value) -> Option<&Value> { Some(v).filter(|v| !v.is_null()) }
    fn r4(v: &Value) -> [f64; 4] { crate::mt::t::r4(v) }

    /// 지금 화면의 실제 함수(compare/head_js.mjs)가 지은 HTML 을 진짜 index.html + 원본 styles.css 위에서 원본 renderSnapshotAssetTabs·placeTopLeft 로 놓고 잰 것(compare/head_layout.py) --
    /// 덩이마다 글·칠이 하나씩 짝(글 x ±0.6 · 나머지 ±0.07) · 덩이 크기 · 페이지 자리(경보 칸 = 잰 flex 재료 · 고정 칩 코인은 상수 room() 도 같아야)
    #[test]
    fn js_parity() {
        let inp = fx("head_in.json");
        let js = std::env::var("HEAD_JS").map(|p| serde_json::from_str::<Value>(&std::fs::read_to_string(p).unwrap()).unwrap()).unwrap_or_else(|_| fx("head_js.json"));
        let cases = js["cases"].as_array().unwrap();
        let all: Vec<&Value> = cases.iter().flat_map(|c| [&c["sess"], &c["coin"], &c["alert"], &c["pos"], &c["alert"]["hid"]]).collect(); // hid = 둘째 줄로 넘어 숨은 칩의 글(폭만)
        let m = lookup(&all, &js["probe"]);
        let mut n = 0;
        for (k, (s, d)) in inp.as_array().unwrap().iter().zip(cases).enumerate() {
            let px: HashMap<String, f64> = serde_json::from_value(s["px"].clone()).unwrap();
            let spk = nn(&s["spk"]);
            let skew = spk.and_then(|v| v["now_sec"].as_f64()).map_or(0.0, |t| t - s["now"].as_f64().unwrap());
            let i = In { asset: s["asset"].as_str().unwrap(), now: s["now"].as_f64().unwrap() * 1000.0, sess: Some(sess_on(&s["sess"])), alert: alert_on(&s["alert"]), spk, skew, acct: nn(&s["acct"]), prices: &px };
            let rr = &d["room"];
            let rm = Room { l0: rr["l0"].as_f64().unwrap(), s: rr["s"].as_f64().unwrap(), tb: rr["tb"].as_f64().unwrap(), tmin: rr["tmin"].as_f64().unwrap() };
            let bl = blocks(&i, Some(rm), 1.0, &m);
            if ["eth", "xrp", "sol"].contains(&i.asset) { // 고정 칩 코인: 상수 재료 = 잰 재료
                let q = room(bl[1].as_ref().unwrap().1[2]);
                assert!((q.l0 - rm.l0).abs() < TOL && (q.s - rm.s).abs() < TOL && q.tb == rm.tb && q.tmin == rm.tmin, "입력 {k} room {q:?} vs {rm:?}");
            }
            for (j, nm) in ["sess", "coin", "alert", "pos"].into_iter().enumerate() {
                let dd = &d[nm];
                let Some((items, r)) = &bl[j] else { assert!(dd["texts"].as_array().unwrap().is_empty() && dd["w"] == 0.0, "입력 {k} {nm}: 원본은 보인다"); continue };
                let (act, _) = actual(items, &m);
                let (exp, _) = expected(dd);
                let (extra, miss, _) = pair(&act, &exp);
                assert!(extra.is_empty() && miss.is_empty(), "입력 {k} {nm}\n남은 rtui {}\n{}\n못 찾은 원본 {}\n{}", extra.len(), show(&extra), miss.len(), show(&miss));
                let pg = r4(&d["page"][nm]);
                assert!((0..4).all(|q| (r[q] - pg[q]).abs() < TOL), "입력 {k} {nm} 자리 {r:?} vs {pg:?}");
                n += exp.len();
            }
        }
        assert!(n > 90, "{n}");
        eprintln!("head js_parity: 단위 {n}");
    }

    /// JS 수 글 · 급변 칩 경계(40분 · 0분 · 같은 쪽 한 칩) · 포지션 고르기(주문 심볼 먼저) · 받기 주기 · 실패
    #[test]
    fn helpers_poll() {
        assert_eq!([line_fmt(1234.5), line_fmt(2.48731), line_fmt(99.99996), line_fmt(0.0)], ["1,234.50", "2.4873", "100.0000", "0.0000"]);
        let now = 1_795_000_000.0;
        let spk = serde_json::json!({ "ok": true, "symbols": { "ZF": { "events": [[now - 300.0 - 2400.0, 4.0, -0.1]] }, "ZN": { "events": [[now - 300.0 - 2401.0, 4.0, -0.1]] } } });
        assert_eq!(spike_chips(Some(&spk), now), [(Col::Bad, "5년 금리↑ · 40분".to_string())], "40분 = 보임 · 1초 넘으면 빠짐");
        let px: HashMap<String, f64> = [("eth".to_string(), 3000.0)].into();
        let a = serde_json::json!({ "positions": [{ "symbol": "ETHUSDT", "side": "LONG", "qty": 1, "entry_price": 2900 }, { "symbol": "ETHUSDC", "side": "SHORT", "qty": 1, "entry_price": 3100 }], "exec_symbol": "ETHUSDC" });
        assert!(fp_pos(Some(&a), &px, "eth").unwrap().short, "주문 심볼(USDC) 먼저");
        assert_eq!(fp_pos(Some(&a), &px, "btc"), None);
        let gets = |o: &mut Vec<Req>| o.drain(..).map(|r| match r { Req::Get { url, .. } => url, _ => panic!() }).collect::<Vec<_>>();
        let mut s = Store { at: [f64::NEG_INFINITY; 3], ..Default::default() };
        let mut o = vec![];
        s.tick(1e9, true, "", &mut o);
        assert!(o.is_empty(), "숨으면 안 묻는다");
        s.tick(1e9, false, "", &mut o);
        assert_eq!(gets(&mut o), ["/api/state", "/api/session-alerts", "/api/macro-spikes"]);
        s.tick(1e9 + 5000.0, false, "", &mut o);
        assert_eq!(gets(&mut o), ["/api/state"]);
        s.on_get(Kind::HeadState, Some(r#"{"state":{"session":{"session_asia":1,"session_us":"0.6"}}}"#), 0.0);
        s.on_get(Kind::HeadState, None, 0.0);
        assert_eq!(s.sess, Some([true, false, true]), "실패 = 받아 둔 값");
        s.on_get(Kind::HeadAlert, Some(r#"{"macro_event_alert":{"active":[{"title_ko":"x","minutes_from_event":1}]}}"#), 0.0);
        assert!(s.alert);
        s.on_get(Kind::HeadAlert, None, 0.0);
        assert!(!s.alert, "경보 실패 = 숨김");
        s.on_get(Kind::HeadSpk, Some(r#"{"ok":true,"symbols":{},"now_sec":1000}"#), 990_000.0);
        assert_eq!(s.skew, 10.0);
    }
}
