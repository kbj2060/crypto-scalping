//! 계좌 칩 = 넓은 화면 맨 위 보유 종목 손익 전광판(#pnlTicker · app.js renderPnlTicker 3905 · tickerPnl 3887 · entryUsdt 2423 · armedBasis 970 · acctCoin 2475).
//! 자료 = 같은 출처 /api/binance-account(10초 · app.js 1649 BINANCE_ACCOUNT_POLL_MS · ok 가 아니면 latestBinanceAccount = null → 숨김) +
//!   코인별 마지막 체결가 = rtcore 허브 «모든 종목» Prices(app.js 4916 msg.type === "prices" → latestLivePriceByAsset).
//! 자리 = styles.css 2636~2648: 상자 높이 42 · 칸 폭 132(안쪽 여백 좌우 16) · 칸 사이 왼쪽 줄 1px --line · 이름 600 11px/1.3 sans --muted · 손익 700 18px/1.15 mono ·
//!   상자 폭 = min(틈, 칸 셋) · 칸은 상자 가운데(넘치면 왼쪽부터 = safe center) · 손익 숫자는 칸 안에서 6초에 한 벌씩 왼쪽으로 흐른다(두 벌 −50%).
//! 안 옮긴 것: 칸 누르면 그 코인 탭으로(이 페이지는 코인 탭이 없다) · 올리면 흐름 멈춤 · 휴대폰 «다른 포지션» 칩(posChipsSet) · 크롬 탭 제목 ·
//!   상자 자리 계산(세션~코인 이름 틈 -- 이 페이지는 틈 = 창 폭) · 체결 알림 acct_seq 즉시 조회(SSE /api/events -- 접착에 SSE 가 없다, 10초 조회만).
//! ponytail: 이름 굵기 600 = 700 글꼴 · letter-spacing .02em 없음 · 흐름 양끝 10px 흐려짐(mask) 없음 -- 글꼴·마스크를 더하면 된다.
use crate::data::to_fixed;
use crate::fp::Col;
use serde_json::Value;
use std::collections::HashMap;

/// JS Number(v) -- null·없음 = 0(이 카드의 쓰임에선 undefined 의 NaN 과 결과가 같다: 전부 `|| 0`·`> 0` 뒤에 쓴다)
fn num(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
        Value::String(s) => { let s = s.trim(); if s.is_empty() { 0.0 } else { s.parse().unwrap_or(f64::NAN) } }
        Value::Bool(b) => *b as u8 as f64,
        Value::Null => 0.0,
        _ => f64::NAN,
    }
}
/// `Number(x) || d`
fn or(x: f64, d: f64) -> f64 { if x == 0.0 || x.is_nan() { d } else { x } }
/// 2475 acctCoin
pub fn coin(s: &str) -> &str { s.strip_suffix("USDT").or_else(|| s.strip_suffix("USDC")).unwrap_or(s) }
fn up(p: &Value) -> String { p["side"].as_str().map_or_else(|| p["side"].to_string(), str::to_string).to_uppercase() }

/// 970 armedBasis
fn armed_basis(b: &Value, dflt: f64) -> f64 {
    let f = |k| num(&b[k]);
    if f("sl_price") > 0.0 && f("sl_level") > 0.0 { f("sl_price") / f("sl_level") } else if f("tp_price") > 0.0 && f("tp_level") > 0.0 { f("tp_price") / f("tp_level") } else { or(f("basis"), dflt) }
}
/// 2423 entryUsdt -- 주문 심볼(USDC) 진입가를 차트(USDT) 가격으로
fn entry_usdt(a: &Value, pos: &Value, chart_px: f64) -> f64 {
    let en = num(&pos["entry_price"]);
    let sym = pos["symbol"].as_str().unwrap_or("");
    if !(en > 0.0) || !sym.ends_with("USDC") { return en; }
    let b = &a["bracket_armed"][format!("{sym}:{}", up(pos))];
    let (mk, bb) = (num(&pos["mark_price"]), if b.is_null() { 0.0 } else { armed_basis(b, 0.0) });
    en / if bb > 0.0 { bb } else if mk > 0.0 && chart_px > 0.0 { mk / chart_px } else { 1.0 }
}
/// 3887 tickerPnl -- 그 코인 1초 가격이 있으면 (지금 − 진입) × 수량 · 없으면 거래소 미실현
fn pnl(a: &Value, p: &Value, prices: &HashMap<String, f64>) -> f64 {
    let cur = prices.get(&coin(p["symbol"].as_str().unwrap_or("")).to_lowercase()).copied().unwrap_or(0.0);
    let qty = or(num(&p["qty"]), 0.0).abs();
    let en = if cur > 0.0 { entry_usdt(a, p, cur) } else { 0.0 };
    if en > 0.0 { (cur - en) * qty * if up(p) == "SHORT" { -1.0 } else { 1.0 } } else { or(num(&p["unrealized_pnl"]), 0.0) }
}

pub struct Chip { pub label: String, pub val: String, pub c: Col }

/// 3905 renderPnlTicker 의 칸 글자·색 -- 빈 목록 = 숨김. acct = /api/binance-account 응답(ok 인 것만 -- feed 가 거른다)
pub fn chips(acct: Option<&Value>, prices: &HashMap<String, f64>) -> Vec<Chip> {
    let Some(a) = acct else { return vec![] };
    a["positions"].as_array().into_iter().flatten().filter(|p| or(num(&p["qty"]), 0.0).abs() > 0.0).map(|p| {
        let u = pnl(a, p, prices);
        let s = p["symbol"].as_str().unwrap_or("");
        let abs = if u.is_finite() { to_fixed(u.abs(), 2) } else { "Infinity".into() };
        Chip { label: format!("{} {}", coin(s), if p["side"] == "LONG" { "롱" } else { "숏" }), val: format!("{}${abs}", if u >= 0.0 { "+" } else { "−" }), c: if u >= 0.0 { Col::Good } else { Col::Bad } }
    }).collect()
}

/// 2636 높이 · 2638 칸 폭 · 3914 칸 셋
pub const H: f64 = 42.0;
const CW: f64 = 132.0;
const PAD: f64 = 16.0;

/// 한 장 -- (x0, y0) = 상자 왼쪽 위 · w = 쓸 수 있는 폭 · t = 시각(ms, 흐름) · 반환 = 흐르는 중(다시 그려야 한다)
pub fn paint(p: &eframe::egui::Painter, cs: &[Chip], x0: f32, y0: f32, w: f64, t: f64) -> bool {
    use crate::draw::{fid, mono, rgba};
    use eframe::egui::{Align2, Rect, Stroke, pos2, vec2};
    if cs.is_empty() { return false; }
    let bw = w.min(3.0 * CW);
    let bx = x0 as f64 + (w - bw) / 2.0;
    let total = cs.len() as f64 * CW;
    let sx = bx + if total <= bw { (bw - total) / 2.0 } else { 0.0 };
    let pb = p.with_clip_rect(Rect::from_min_size(pos2(bx as f32, y0), vec2(bw as f32, H as f32)).intersect(p.clip_rect()));
    // grid align-content center: 이름 줄 14.3 + 손익 줄 20.7 = 35 → 위 3.5
    let (ly, vy) = (y0 as f64 + 3.5 + 14.3 / 2.0, y0 as f64 + 3.5 + 14.3 + 20.7 / 2.0);
    for (i, c) in cs.iter().enumerate() {
        let x = sx + i as f64 * CW;
        if i > 0 { pb.line_segment([pos2(x as f32 + 0.5, y0), pos2(x as f32 + 0.5, y0 + H as f32)], Stroke::new(1.0, rgba(Col::Line, 1.0))); }
        pb.text(pos2((x + PAD) as f32, ly as f32), Align2::LEFT_CENTER, &c.label, fid(11.0, 600), rgba(Col::Muted, 1.0));
        // .mq(칸 안쪽 100) 안에서 [값 + 32] 두 벌이 6초에 한 벌 폭만큼 왼쪽으로
        let g = p.layout_no_wrap(c.val.clone(), mono(18.0, 700), rgba(c.c, 1.0));
        let run = g.size().x as f64 + 32.0;
        let off = -(t.rem_euclid(6000.0) / 6000.0) * run;
        let pm = pb.with_clip_rect(Rect::from_min_size(pos2((x + PAD) as f32, y0), vec2((CW - 2.0 * PAD) as f32, H as f32)).intersect(pb.clip_rect()));
        for k in 0..2 { pm.galley(pos2((x + PAD + off + k as f64 * run) as f32, (vy - g.size().y as f64 / 2.0) as f32), g.clone(), rgba(c.c, 1.0)); }
    }
    true
}

#[cfg(test)]
mod t {
    use super::*;

    /// 지금 화면의 실제 함수로 만든 정답지(compare/acct_js.mjs → fixture/acct_js.json · 같은 입력 acct_in.json)
    #[test]
    fn js_parity() {
        let fx = |f: &str| serde_json::from_str::<Value>(&std::fs::read_to_string(format!("{}/compare/fixture/{f}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap();
        let (inp, js) = (fx("acct_in.json"), fx("acct_js.json"));
        let mut n = 0;
        for (i, j) in inp.as_array().unwrap().iter().zip(js.as_array().unwrap()) {
            let a = Some(&i[0]).filter(|a| a["ok"] == true); // 2972 latestBinanceAccount = payload.ok ? payload : null
            let px: HashMap<String, f64> = serde_json::from_value(i[1].clone()).unwrap();
            let cs = chips(a, &px);
            let jc = j["chips"].as_array().unwrap();
            assert_eq!(cs.is_empty(), j["hidden"] == true);
            assert_eq!(cs.len(), jc.len());
            for (c, q) in cs.iter().zip(jc) {
                let cls = match c.c { Col::Good => "good", Col::Bad => "bad", _ => "" };
                assert_eq!((c.label.as_str(), c.val.as_str(), cls), (q["label"].as_str().unwrap(), q["val"].as_str().unwrap(), q["cls"].as_str().unwrap()));
                n += 1;
            }
        }
        assert!(n >= 10, "{n}");
    }

    /// 10초마다 계좌 · 허브는 구독 없이 붙어 «모든 종목» 가격만 · Prices 번호가 비면 다시 붙는다(심볼 번호와 따로)
    #[test]
    fn feed_acct() {
        use crate::feed::{Feed, Kind, Req, Source, Urls};
        let mut f = Feed::new("eth", "ETHUSDT", Source::Push, Urls::default(), false);
        f.acct_only = true;
        for t in [20_000.0, 25_000.0, 30_000.0] { f.tick(t, false, false); }
        let gets: Vec<_> = f.out.iter().filter_map(|r| if let Req::Get { kind: Kind::Acct, url } = r { Some(url.as_str()) } else { None }).collect();
        assert_eq!(gets, ["/api/binance-account"; 2]);
        assert!(f.out.contains(&Req::WsOpen("/rt".into())) && f.out.contains(&Req::WsSend(r#"{"subscribe":[]}"#.into())));
        assert!(!f.out.iter().any(|r| matches!(r, Req::Get { kind, .. } if *kind != Kind::Acct)));
        f.out.clear();
        f.on_ws(r#"{"s":"*","seq":5,"ex_ms":0,"rx_us":0,"tx_us":1,"type":"prices","rows":[["ethusdt",3000.5,1],["dogeusdt",1.0,1]]}"#, 0.0);
        f.on_ws(r#"{"s":"*","seq":6,"ex_ms":0,"rx_us":0,"tx_us":1,"type":"prices","rows":[["solusdt",150.0,1]]}"#, 0.0);
        assert_eq!((f.prices.get("eth"), f.prices.get("sol"), f.prices.len()), (Some(&3000.5), Some(&150.0), 2));
        assert!(f.out.is_empty());
        f.on_ws(r#"{"s":"*","seq":8,"ex_ms":0,"rx_us":0,"tx_us":1,"type":"prices","rows":[]}"#, 0.0);
        assert_eq!(f.out, vec![Req::WsClose]);
        f.on_acct(Some(r#"{"ok":false}"#));
        assert!(f.acct.is_none());
        f.on_acct(Some(r#"{"ok":true,"positions":[{"symbol":"ETHUSDT","side":"LONG","qty":1,"entry_price":3000}]}"#));
        assert_eq!(chips(f.acct.as_ref(), &f.prices)[0].val, "+$0.50");
    }
}
