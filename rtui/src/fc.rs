//! ⑤ 예측 줄 전체(app.js fcRowHtml 5995 · ?card=dirs): 머리 «종합» 칩(ETH) · 지금 추세(+7일 추세) · 15분 · 레벨 5분(ETH) · 1시간 · 4h 예측 · 횡보·추세 · 15분 폭 · 앞 1시간 · 4h 상대 강도.
//! 15분·1시간 = dirs.rs · 레벨 5분 = srl.rs · 칸 그리기 = srl.rs scene(게이지·적중 막대). 여기는 나머지 칸 값과 그 칸만의 자리.
//! 자료 = 원본이 부르는 같은 주소·주기: /api/trend?asset= 60초(실패 = ok 아님으로 덮음) · /api/dir4h?asset= 120초(ETH·SOL·XRP) · /api/open-rev 120초(ETH) · /api/regime-band 300초 ·
//!   /api/risk-radar 30초 · /api/pred-record 60초 · /api/dir-short 60초 · /api/sr-lean 2초(ETH) · 허브 «모든 종목» 가격(15분 폭 $) -- 그 밖은 받아 둔 값 그대로.
//! 자리 = 원본 styles.css(이 작업 시점)를 헤드리스 크로미움 1920(넓은 화면 #mcBody.mc-cmp · 폭 404)에 띄워 잰 값(compare/fc_layout.py → fixture/fc_js.json · 시험 js_parity).
//! 안 옮긴 것: 칸 툴팁(title) · «?» 설명 팝업(단추 모양만 그림) · 좁은 화면 판.
//! ponytail: «급변» 칩 테두리 = 네모(둥근 7px 은 채움만) · 7일 추세 칩 칸 폭 = 판정 글 폭(브라우저는 칩 글 폭과 큰 쪽) -- 칩 글이 더 넓어지면 고친다.
use crate::dirs;
use crate::fp::{Anchor, Clip, Col, Prim};
use crate::srl::{self, Cell, Right, truthy};
use serde_json::Value;

/// 5687 DIR4H_ASSETS
pub const DIR4H: [&str; 3] = ["eth", "sol", "xrp"];
/// 글자 폭(글, 크기, 굵기, 고정폭 --font-mono 인가)
pub type Measure<'a> = &'a dyn Fn(&str, f64, u16, bool) -> f64;

/// 받아 둔 응답(None = 아직) · now = Date.now() · now_s = 레벨 5분 서버 시계(srLeanNowS) · price = 허브 가격(latestLivePriceByAsset)
pub struct In<'a> {
    pub asset: &'a str, pub now: f64, pub now_s: f64, pub price: Option<f64>,
    pub trend: Option<&'a Value>, pub dir4h: Option<&'a Value>, pub open_rev: Option<&'a Value>, pub dirs: Option<&'a Value>, pub pred: Option<&'a Value>,
    pub rb: Option<&'a Value>, pub rr: Option<&'a Value>, pub srl: Option<&'a Value>,
}

fn ok(v: Option<&Value>) -> Option<&Value> { v.filter(|v| v["ok"] == true) }
fn f(v: &Value) -> f64 { v.as_f64().unwrap_or(f64::NAN) }
fn sgn_col(s: f64) -> Col { if s > 0.0 { Col::Good } else if s < 0.0 { Col::Bad } else { Col::Muted } }
fn word(s: f64) -> &'static str { if s > 0.0 { "▲ 상승" } else if s < 0.0 { "▼ 하락" } else { "보류" } }
fn text(x: f64, y: f64, s: String, size: f64, wt: u16, c: Col, mono: bool) -> Prim {
    if mono { Prim::Mono { x, y, s, size, wt, c, a: 1.0, anchor: Anchor::Start, halo: false, clip: Clip::No } } else { Prim::Text { x, y, s, size, wt, c, a: 1.0, anchor: Anchor::Start, halo: false, clip: Clip::No } }
}
fn rect(x: f64, y: f64, w: f64, h: f64, c: Col, a: f64, rx: f64) -> Prim { Prim::Rect { x, y, w, h, c, a, rx, clip: Clip::No } }
/// 글자 가운데 y → 바닥선(mc.rs·srl.rs 와 같은 근사)
fn by(center: f64, size: f64) -> f64 { center + size * 0.36 }
/// 칸 가로 자리(srl::scene 과 같다): (게이지 칸 x, 적중 칸 x, 칸 폭)
fn cols(w: f64) -> (f64, f64, f64) { let cw = (w - 8.0 - 56.0 - 4.0) / 2.0; (62.0, 62.0 + cw + 2.0, cw) }

/// 6036 종합 세 부품 [4h · 추세 · 09시](None = 자료 없음 «-») → (합, 같은 쪽 수, 부품)
fn parts(i: &In) -> (f64, usize, [(&'static str, Option<f64>); 3]) {
    let g4 = dir4h_now(i);
    let old4 = g4.is_some_and(|g| i.now - f(&g["decided_at_ms"]) > 15.0 * 60e3);
    let t = ok(i.trend);
    let orv = ok(i.open_rev).filter(|_| i.asset == "eth");
    let on = orv.is_some_and(|o| i.now >= f(&o["entry_ms"]) && i.now < f(&o["exit_due_ms"]));
    let p = [("4h", g4.filter(|_| !old4).map(|g| f(&g["signal"]))),
             ("추세", t.map(|t| { let s = f(&t["signal"]); if s.abs() >= 0.5 { s.signum() } else { 0.0 } })),
             ("09시", orv.map(|o| if on { f(&o["open_pos"]) } else { 0.0 }))];
    let net: f64 = p.iter().map(|q| q.1.unwrap_or(0.0)).sum();
    let agree = p.iter().filter(|q| q.1.is_some_and(|s| s != 0.0 && s.signum() == net.signum())).count();
    (net, agree, p)
}
fn dir4h_now<'a>(i: &In<'a>) -> Option<&'a Value> { ok(i.dir4h).filter(|g| g["asset"].as_str().unwrap_or("eth") == i.asset) }

/// 머리 줄(h4 · 위 여백 6 · 높이 19): ETH = «종합» 칩(가운데 쪽 -- auto 여백 둘이 남는 폭을 나눔) · 오른쪽 끝 «?» 단추(15 · 테 --line)
fn head(i: &In, w: f64, m: Measure) -> Vec<Prim> {
    let cy = 6.0 + 9.5;
    let bx = w - 15.0;
    let mut v = vec![Prim::Line { pts: (0..=24).map(|k| { let q = k as f64 * std::f64::consts::TAU / 24.0; [bx + 7.5 + 7.0 * q.cos(), cy + 7.0 * q.sin()] }).collect(), c: Col::Line, a: 1.0, lw: 1.0, dash: None, clip: Clip::No },
                     Prim::Text { x: bx + 7.5, y: by(cy, 10.0), s: "?".into(), size: 10.0, wt: 700, c: Col::Muted, a: 1.0, anchor: Anchor::Middle, halo: false, clip: Clip::No }];
    if i.asset != "eth" { return v; }
    let (net, agree, p) = parts(i);
    let fs = sgn_col(net);
    let b = if net > 0.0 { "▲ 롱 쪽" } else if net < 0.0 { "▼ 숏 쪽" } else { "중립" };
    let sm = (net != 0.0).then(|| format!("{agree}/3"));
    let chips: Vec<(String, Option<f64>)> = p.iter().map(|&(n, s)| (format!("{n} {}", match s { None => "-", Some(s) if s > 0.0 => "▲", Some(s) if s < 0.0 => "▼", _ => "·" }), s)).collect();
    let (w0, w1) = (m("종합 ", 10.5, 700, false), m(b, 12.0, 800, false));
    let w2 = sm.as_ref().map(|s| m(s, 10.5, 700, true));
    let cw: Vec<f64> = chips.iter().map(|c| m(&c.0, 10.0, 700, true) + 10.0).collect();
    let total = w0 + 5.0 + w1 + w2.map_or(0.0, |x| 5.0 + x + 2.0) + cw.iter().map(|c| 5.0 + c).sum::<f64>();
    let mut x = (w - total - 15.0 - 8.0) / 2.0;
    v.push(text(x, by(cy, 10.5), "종합 ".into(), 10.5, 700, Col::Muted, false));
    x += w0 + 5.0;
    v.push(text(x, by(cy, 12.0), b.into(), 12.0, 800, fs, false));
    x += w1 + 5.0;
    if let (Some(s), Some(ww)) = (sm, w2) { v.push(text(x, by(cy, 10.5), s, 10.5, 700, fs, true)); x += ww + 2.0 + 5.0; }
    for ((s, q), ww) in chips.into_iter().zip(cw) {
        let (c, a) = match q { Some(q) if q > 0.0 => (Col::Good, 0.16), Some(q) if q < 0.0 => (Col::Bad, 0.16), _ => (Col::Muted, 0.12) };
        v.push(rect(x, cy - 8.0, ww, 16.0, c, a, 3.0));
        v.push(text(x + 5.0, by(cy, 10.0), s, 10.0, 700, if a == 0.12 { Col::Muted } else { c }, true));
        x += ww + 5.0;
    }
    v
}

/// 칩 줄(.fc-tl: 칸 폭 tw · 사이 2 · 높이 16 · 글 가운데) -- (글, 색)
fn chips(x0: f64, y: f64, tw: f64, items: &[(String, Col)], m: Measure) -> Vec<Prim> {
    let n = items.len() as f64;
    let cw = (tw - 2.0 * (n - 1.0)) / n;
    items.iter().enumerate().flat_map(|(k, (s, c))| {
        let x = x0 + k as f64 * (cw + 2.0);
        [rect(x, y, cw, 16.0, *c, 0.16, 3.0), text(x + (cw - m(s, 10.0, 700, true)) / 2.0, by(y + 8.0, 10.0), s.clone(), 10.0, 700, *c, true)]
    }).collect()
}

/// 5771 fcTrendCell(+ 5782 f1Html 오른쪽) -- 높이 · 그림. 자료 없으면 «-»(13px 가운데)
fn trend(i: &In, w: f64, m: Measure) -> (f64, Vec<Prim>) {
    let (x2, x3, cw) = cols(w);
    let Some(t) = ok(i.trend) else {
        let h = 25.188;
        return (h, vec![text(4.0, by(h / 2.0, 11.0), "지금 추세".into(), 11.0, 700, Col::Muted, false), text(x2 + (cw - m("-", 13.0, 700, false)) / 2.0, by(h / 2.0, 13.0), "-".into(), 13.0, 700, Col::Muted, false)]);
    };
    let votes = t["votes"].as_array().map(Vec::as_slice).unwrap_or(&[]);
    let ups = f(&t["ups"]);
    let up = f(&t["signal"]) > 0.0;
    let f1 = t.get("f1").filter(|x| x["votes"].is_array());
    let h = if f1.is_some() { 46.594 } else { 45.0 };
    let top = 6.0 + (h - 12.0 - 33.0) / 2.0; // .fc-t(판정 줄 13 + 4 + 칩 16)가 칸 안 가운데
    let (main, cnt) = (format!("{} ", if up { "▲ 상승" } else { "▼ 하락" }), format!("{}:{}", srl::js(&Value::from(ups)), srl::js(&Value::from(votes.len() as f64 - ups))));
    let (wm, wc) = (m(&main, 13.0, 700, false), m(&cnt, 11.0, 700, false));
    let x = x2 + 6.0 + (cw - 12.0 - wm - 2.0 - wc) / 2.0;
    let c = if up { Col::Good } else { Col::Bad };
    let mut v = vec![text(4.0, by(h / 2.0, 11.0), "지금 추세".into(), 11.0, 700, Col::Muted, false),
                     text(x, top + 11.3, main, 13.0, 700, c, false), text(x + wm + 2.0, top + 11.3, cnt, 11.0, 700, Col::Muted, false)];
    let tw = (cw - 12.0).min(150.0);
    let items: Vec<_> = votes.iter().map(|q| (srl::js(&q["L"]), if truthy(&q["up"]) { Col::Good } else { Col::Bad })).collect();
    v.extend(chips(x2 + 6.0 + (cw - 12.0 - tw) / 2.0, top + 17.0, tw, &items, m));
    if let Some(f1) = f1 { // .f1 = [이름 auto][8][판정 · 표 수] / [칩 max 132] · 오른쪽 칸에서 가장 높아 칸 위 6 부터
        let s = f(&f1["signal"]);
        let (g, n) = (if s > 0.0 { "▲ 상승" } else if s < 0.0 { "▼ 하락" } else { "중립" }, format!("{}:{} · 참고용", srl::js(&f1["ups"]), srl::js(&f1["downs"])));
        let (wh, wg, wn) = (m("7일 추세", 11.0, 700, false), m(g, 13.0, 700, false), m(&n, 10.5, 400, false));
        let c2 = x3 + wh + 8.0;
        v.extend([text(x3, by(6.0 + 7.2, 11.0), "7일 추세".into(), 11.0, 700, Col::Muted, false), text(c2, by(6.0 + 6.5, 13.0), g.into(), 13.0, 700, sgn_col(s), false),
                  text(c2 + wg + 6.0, by(6.0 + 7.5, 10.5), n, 10.5, 400, Col::Muted, false)]);
        let items: Vec<_> = f1["votes"].as_array().unwrap().iter().map(|q| (srl::js(&q["L"]), sgn_col(f(&q["s"])))).collect();
        v.extend(chips(c2, 6.0 + 14.594 + 4.0, (wg + 6.0 + wn).min(132.0), &items, m));
    }
    (h, v)
}

/// 4h 예측 칸(fcRowHtml 6013~6031 · 6095)
fn d4(i: &In) -> Cell {
    let g = dir4h_now(i);
    let old = g.is_some_and(|g| i.now - f(&g["decided_at_ms"]) > 15.0 * 60e3);
    let right = if DIR4H.contains(&i.asset) { dirs::hit(i.pred, i.asset, "4h") } else { Right::None };
    let Some(g) = g else { return Cell { hz: "4h 예측".into(), c: Col::Muted, main: "-".into(), gauge: None, zone: [20.0, 80.0], risk: false, ago: None, right } };
    let (lo, hi, bp) = (f(&g["lo"]), f(&g["hi"]), f(&g["pred_bp"]));
    let r = lo.abs().max(hi.abs()) * 1.6;
    let x = |b: f64| dirs::x1f(50.0 + b / r * 50.0);
    let cut = f(&g["decided_at_ms"]) - 3600e3 + 150e3;
    let ago = g["hist"].as_array().and_then(|h| h.iter().filter(|q| f(&q[0]) <= cut).last()).map(|q| x(f(&q[1])));
    let s = f(&g["signal"]);
    Cell { hz: "4h 예측".into(), c: if old { Col::Muted } else { sgn_col(s) }, main: word(s).into(), gauge: Some(x(bp)), zone: [x(lo), x(hi)], risk: false, ago, right }
}

/// 횡보·추세 칸(6076~6089 rbCell)
fn band(i: &In) -> Cell {
    let rb = ok(i.rb).map(|r| &r["coins"][i.asset]).filter(|r| r.is_object());
    let right = dirs::hit(i.pred, i.asset, "band");
    let (c, main) = match rb {
        None => (Col::Muted, "-"),
        Some(rb) => {
            let (pt, conf) = (f(&rb["p_trend"]), f(&rb["confidence"]));
            if i.now - f(&rb["decided_at_ms"]) > 90.0 * 60e3 { (Col::Muted, "-") } else if conf < 0.2 { (Col::Muted, "애매") } else if pt >= 0.5 { (Col::Amber, "추세") } else { (Col::Muted, "횡보") }
        }
    };
    Cell { hz: "횡보·추세".into(), c, main: main.into(), gauge: None, zone: [20.0, 80.0], risk: false, ago: None, right }
}

/// 오른쪽 .rr-r 세로 줄(오른쪽 붙임 · 사이 3 · 칸 안 가운데): (글, 크기, 색, 급변 칩인가) · ch = 칸 안 높이
fn rr_col(w: f64, ch: f64, items: &[(String, f64, Col, bool)], m: Measure) -> Vec<Prim> {
    let hs: Vec<f64> = items.iter().map(|q| if q.3 { 19.0 } else { q.1 }).collect();
    let mut y = 6.0 + (ch - hs.iter().sum::<f64>() - 3.0 * (items.len() as f64 - 1.0)) / 2.0;
    let r = w - 4.0;
    let mut v = vec![];
    for ((s, z, c, al), h) in items.iter().zip(hs) {
        if *al { // .rr-al 11px 700 sans · 안쪽 3/7 · 테 1
            let tw = m(s, 11.0, 700, false);
            v.extend([rect(r - tw - 16.0, y, tw + 16.0, 19.0, Col::Amber, 0.13, 7.0), Prim::StrokeRect { x: r - tw - 15.5, y: y + 0.5, w: tw + 15.0, h: 18.0, c: Col::Amber, a: 0.45, lw: 1.0, clip: Clip::No },
                      text(r - tw - 8.0, by(y + 9.5, 11.0), s.clone(), 11.0, 700, *c, false)]);
        } else {
            v.push(text(r - m(s, *z, 700, true), by(y + h / 2.0 - 0.25, *z), s.clone(), *z, 700, *c, true));
        }
        y += h + 3.0;
    }
    v
}

/// 5963 rrCells → (높이, 그림) 둘(15분 폭 · 앞 1시간)
fn risk(i: &In, w: f64, m: Measure) -> [(f64, Vec<Prim>); 2] {
    let r = ok(i.rr).filter(|r| r["stale"] != true);
    let o = r.map(|r| &r["coins"][i.asset]).filter(|o| truthy(&o["valid"]) && !truthy(&o["error"]));
    let plain = |hz: &str, main: String, c: Col| { let c = Cell { hz: hz.into(), c, main, gauge: None, zone: [0.0, 100.0], risk: true, ago: None, right: Right::None }; (srl::height(&c), c) };
    let ms = |a: &str, z: f64, wt: u16| m(a, z, wt, false);
    let sc = |(h, c): (f64, Cell)| (h, srl::scene(&c, w, &ms));
    let Some(o) = o else { return [sc(plain("15분 폭", "-".into(), Col::Muted)), sc(plain("앞 1시간", "-".into(), Col::Muted))] };
    let f2 = |x: f64| dirs::fixed(x, 2);
    let px = i.price.filter(|p| *p != 0.0).unwrap_or(f(&o["price"]));
    let usd = |pct: f64| { let x = px * pct / 100.0; format!("${}", if x >= 100.0 { dirs::fixed(x, 0) } else if x >= 1.0 { dirs::fixed(x, 1) } else { dirs::fixed(x, (2.0 - x.log10().floor()) as usize) }) };
    let al = truthy(&o["alert"]["active"]);
    let (b80, b95) = (f(&o["band80_pct"]), f(&o["band95_pct"]));
    let q = b80 / f(&o["sigma15_pct"]);
    // 15분 폭: 판정(15px 가운데 · 줄 위) + .rr-sub(고정폭 11 · 왼쪽) · 오른쪽 = 급변 칩 또는 $ + 7일 기준
    let (x2, _, cw) = cols(w);
    let h = 43.297;
    let main = format!("±{}%", f2(b80));
    let mut v = vec![text(4.0, by(h / 2.0, 11.0), "15분 폭".into(), 11.0, 700, Col::Muted, false),
                     text(x2 + (cw - m(&main, 15.0, 700, false)) / 2.0, by(13.0, 15.0), main.clone(), 15.0, 700, if al { Col::Amber } else { Col::Muted }, false),
                     text(x2 + 6.0, by(30.0, 11.0), format!("95% ±{}%", f2(b95)), 11.0, 700, Col::Muted, true)];
    let mut items = vec![if al { (format!("급변 {}분 전 · 넓게", srl::js(&o["alert"]["minutes_ago"])), 11.0, Col::Amber, true) } else { (usd(b80), 11.5, Col::Text, false) }];
    if !o["baseline7d_sigma15_pct"].is_null() { items.push((format!("7일 기준 ±{}%", f2(f(&o["baseline7d_sigma15_pct"]) * q)), 11.0, Col::Muted, false)); }
    v.extend(rr_col(w, h - 12.0, &items, m));
    let c15 = (h, v);
    // 앞 1시간: 주식 = 세션 이름 + σ 한 줄 · 코인 = 조용/보통/격함 + 게이지(0.4~2.5배 로그 · 1 가운데) + 평소의 N배 · σ
    if !o["session"].is_null() && truthy(&o["session"]) {
        let (hh, c) = plain("앞 1시간", o["session"].as_str().unwrap_or("").into(), Col::Muted);
        let mut v = srl::scene(&c, w, &ms);
        v.extend(rr_col(w, hh - 12.0, &[(format!("1시간 σ {}%", f2(f(&o["sigma60_pct"]))), 11.5, Col::Text, false)], m));
        return [c15, (hh, v)];
    }
    let st = o["state60"].as_str().unwrap_or("-");
    let ratio = o["ratio60"].as_f64().filter(|r| *r != 0.0);
    let xx = |x: f64| dirs::x1f(50.0 + x.ln() / 2.5f64.ln() * 50.0);
    let c = Cell { hz: "앞 1시간".into(), c: if st == "격함" { Col::Amber } else { Col::Muted }, main: st.into(), gauge: ratio.map(xx), zone: [xx(0.7), xx(1.5)], risk: true, ago: None, right: Right::None };
    let hh = srl::height(&c);
    let mut v = srl::scene(&c, w, &ms);
    if let Some(r) = ratio { v.extend(rr_col(w, hh - 12.0, &[(format!("평소의 {}배", f2(r)), 11.5, Col::Text, false), (format!("1시간 σ {}%", f2(f(&o["sigma60_pct"]))), 11.0, Col::Muted, false)], m)); }
    [c15, (hh, v)]
}

/// 5948 predXsecHtml «4h 상대 강도»(위 4 · 머리 6/13/4 · 막대 줄 76 · 아래 4 + 글 14.85) → (높이, 그림) · 없으면 None
fn xsec(i: &In, w: f64, m: Measure) -> Option<(f64, Vec<Prim>)> {
    let pr = ok(i.pred)?;
    let mut xs: Vec<(String, f64)> = pr["xsec"]["latest"]["rel_bp"].as_object()?.iter().map(|(k, v)| (k.clone(), f(v))).collect();
    if xs.is_empty() { return None; }
    xs.sort_by(|a, b| b.1.total_cmp(&a.1)); // ponytail: 같은 값 차례 = 이름순(serde_json 맵은 키 정렬 · 원본은 받은 차례) -- 같은 bp 가 흔해지면 preserve_order
    let sc = xs.iter().fold(30.0f64, |a, q| a.max(q.1.abs()));
    let (hy, ry) = (4.0 + 6.0, 4.0 + 6.0 + 13.0 + 4.0);
    let hw = m("4h 상대 강도", 10.5, 700, false);
    let mut v = vec![text(0.0, by(hy + 6.5, 10.5), "4h 상대 강도".into(), 10.5, 700, Col::Muted, false), rect(hw + 8.0, hy + 6.0, w - hw - 8.0, 1.0, Col::Line, 1.0, 0.0)];
    let cw = w / xs.len() as f64;
    for (k, (name, val)) in xs.iter().enumerate() {
        let x = k as f64 * cw;
        let me = name.to_lowercase() == i.asset;
        v.push(rect(x, ry + 26.0, cw, 1.0, Col::Line, 1.0, 0.0));
        let h: f64 = dirs::fixed((val.abs() / sc * 24.0).max(1.0), 0).parse().unwrap();
        let (c, y) = if *val >= 0.0 { (Col::Good, ry + 26.0 - h) } else { (Col::Bad, ry + 26.0) };
        let c = if me { let q = crate::draw::rgba(c, 1.0); let b = |u: u8| (u as f64 * 1.45).round().min(255.0) as u8; Col::Rgb(b(q.r()), b(q.g()), b(q.b())) } else { c }; // .xs-c.me i filter: brightness(1.45)
        v.push(rect(x + cw * 0.22, y, cw * 0.56, h, c, 0.55, 2.0));
        let (tc, wn, wv) = if me { (Col::Text, 800, 800) } else { (Col::Muted, 700, 400) };
        let vs = crate::fp::tnum(&format!("{}{}", if *val >= 0.0 { "+" } else { "−" }, dirs::fixed(val.abs(), 0)));
        for (s, wt, cy) in [(name.clone(), wn, ry + 76.0 - 18.0), (vs, wv, ry + 76.0 - 6.0)] { v.push(text(x + (cw - m(&s, 11.0, wt, false)) / 2.0, by(cy, 11.0), s, 11.0, wt, tc, false)); }
    }
    let ic = if truthy(&pr["xsec"]["n"]) { let c = f(&pr["xsec"]["ic"]); format!("기록 전진 IC {}{}({}회)", if c >= 0.0 { "+" } else { "−" }, dirs::fixed(c.abs(), 2), srl::js(&pr["xsec"]["n"])) } else { "전진 기록 쌓는 중".into() };
    let ny = ry + 76.0 + 4.0;
    v.push(text(0.0, by(ny + 6.5, 11.0), format!("높은 쪽 → 이후 약세 · {ic}"), 11.0, 400, Col::Muted, false));
    Some((ny + 14.85, v))
}

/// 한 장: (y, 그림) 묶음들 · 칸 사이 점선(.fc-c + .fc-c) · 전체 높이. w = 칸 폭(.mc-fc)
pub fn section(i: &In, w: f64, m: Measure) -> (Vec<(f64, Vec<Prim>)>, f64) {
    let ms = |a: &str, z: f64, wt: u16| m(a, z, wt, false);
    let sc = |c: Cell| (srl::height(&c), srl::scene(&c, w, &ms));
    let mut rows = vec![trend(i, w, m), sc(dirs::cell(i.dirs, i.pred, i.asset, i.now, "15m"))];
    if i.asset == "eth" { rows.push(sc(srl::cell(i.srl, i.now_s))); }
    rows.extend([sc(dirs::cell(i.dirs, i.pred, i.asset, i.now, "1h")), sc(d4(i)), sc(band(i))]);
    rows.extend(risk(i, w, m));
    let mut out = vec![(0.0, head(i, w, m))];
    let mut y = 29.0; // 위 여백 6 + 머리 19 + 4
    for (k, (h, v)) in rows.into_iter().enumerate() {
        if k > 0 { out.push((0.0, vec![Prim::Dashed { pts: vec![[0.0, y + 0.5], [w, y + 0.5]], c: Col::SoftLine, a: 1.0, lw: 1.0, on: 3.0, off: 3.0, clip: Clip::No }])); y += 1.0; }
        out.push((y, v));
        y += h;
    }
    if let Some((h, v)) = xsec(i, w, m) { out.push((y, v)); y += h; }
    (out, y)
}

#[cfg(test)]
pub(crate) mod t {
    use super::*;

    fn fx(f: &str) -> Value { serde_json::from_str(&std::fs::read_to_string(format!("{}/compare/fixture/{f}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap() }
    fn nn(v: &Value) -> Option<&Value> { Some(v).filter(|v| !v.is_null()) }
    pub(crate) fn input(s: &Value) -> In<'_> {
        In { asset: s["asset"].as_str().unwrap(), now: f(&s["now"]), now_s: f(&s["now"]) / 1000.0, price: s["price"].as_f64(), trend: nn(&s["trend"]), dir4h: nn(&s["dir4h"]), open_rev: nn(&s["openRev"]),
             dirs: nn(&s["dirShort"]), pred: nn(&s["pred"]), rb: nn(&s["rb"]), rr: nn(&s["rr"]), srl: nn(&s["srl"]) }
    }
    fn untnum(s: &str) -> String { s.chars().map(|c| match c as u32 { u @ 0xE071..=0xE07A => char::from_digit(u - 0xE071, 10).unwrap(), _ => c }).collect() }
    /// 굵기 갈래(--font-sans 500 = 400 글꼴 · mono 600 이상 = Bold)
    fn wb(wt: f64, mono: bool) -> u16 { if mono { if wt >= 600.0 { 700 } else { 400 } } else if wt <= 500.0 { 400 } else if wt < 750.0 { 700 } else { 800 } }
    /// 비교 단위: 글(t) · 칠한 상자(b) · 점선(d: x, y 가운데, 폭) · 동그라미 테(r: 가운데) · 네모 테(o)
    #[derive(Debug, Clone)]
    pub(crate) struct It { pub k: char, pub r: [f64; 4], pub c: [f64; 4], pub s: String, pub size: f64, pub wt: u16, pub mono: bool }
    /// 색 = 그려지는 값(egui Color32 -- 미리 곱한 8비트)
    fn col(c: Col, a: f64) -> [f64; 4] { let q = crate::draw::rgba(c, a); [q.r(), q.g(), q.b(), q.a()].map(|v| v as f64) }
    fn pm(c: [f64; 4]) -> [f64; 4] { let q = eframe::egui::Color32::from_rgba_unmultiplied(c[0].round() as u8, c[1].round() as u8, c[2].round() as u8, (c[3] * 255.0).round() as u8); [q.r(), q.g(), q.b(), q.a()].map(|v| v as f64) }
    fn bare(k: char, r: [f64; 4], c: [f64; 4]) -> It { It { k, r, c, s: String::new(), size: 0.0, wt: 0, mono: false } }
    /// 지금 화면 잰 값(fixture/fc_js.json 한 장) → 비교 단위 · 글자 폭 표. .xs-c.me 막대는 filter: brightness(1.45)(잰 배경색엔 안 들어감)
    pub(crate) fn expected(d: &Value) -> (Vec<It>, Vec<(String, f64, u16, bool, f64)>) {
        let r4 = |v: &Value| -> [f64; 4] { let a = v.as_array().unwrap(); [f(&a[0]), f(&a[1]), f(&a[2]), f(&a[3])] };
        let (mut out, mut widths) = (vec![], vec![]);
        for t in d["texts"].as_array().unwrap() {
            let (mono, wt) = (t["mono"] == true, wb(f(&t["wt"]), t["mono"] == true));
            out.push(It { k: 't', r: r4(&t["r"]), c: pm(r4(&t["c"])), s: t["s"].as_str().unwrap().trim().into(), size: f(&t["size"]), wt, mono });
            widths.push((t["s"].as_str().unwrap().to_string(), f(&t["size"]), wt, mono, r4(&t["r"])[2]));
        }
        for b in d["boxes"].as_array().unwrap() {
            let mut c = r4(&b["c"]);
            if b["f"].as_str() == Some("brightness(1.45)") { for k in 0..3 { c[k] = (c[k] * 1.45).round().min(255.0); } }
            out.push(bare('b', r4(&b["r"]), pm(c)));
        }
        for l in d["lines"].as_array().unwrap() {
            let (r, c) = (r4(&l["r"]), pm(r4(&l["c"])));
            out.push(match (l["style"].as_str().unwrap(), l["all"] == true, l["rad"] == "50%") {
                ("dashed", _, _) => bare('d', [r[0], r[1] + 0.5, r[2], 0.0], c),
                (_, true, true) => bare('r', [r[0] + r[2] / 2.0, r[1] + r[3] / 2.0, 0.0, 0.0], c),
                (_, true, false) => bare('o', r, c),
                _ => bare('b', [r[0], r[1], r[2], 1.0], c),
            });
        }
        (out, widths)
    }
    /// rtui 그림 → 비교 단위(글 = 왼쪽 x · 가운데 y = 바닥선 − 0.36·크기)
    pub(crate) fn actual(groups: &[(f64, Vec<Prim>)], m: Measure) -> Vec<It> {
        let mut out = vec![];
        for (dy, v) in groups {
            for p in v {
                out.push(match p {
                    Prim::Text { x, y, s, size, wt, c, a, anchor, .. } | Prim::Mono { x, y, s, size, wt, c, a, anchor, .. } => {
                        let mono = matches!(p, Prim::Mono { .. });
                        let tw = m(s, *size, *wt, mono);
                        let x = match anchor { Anchor::Start => *x, Anchor::Middle => x - tw / 2.0, Anchor::End => x - tw };
                        It { k: 't', r: [x, y + dy - size * 0.36, tw, 0.0], c: col(*c, *a), s: untnum(s).trim().into(), size: *size, wt: wb(*wt as f64, mono), mono }
                    }
                    &Prim::Rect { x, y, w, h, c, a, .. } => bare('b', [x, y + dy, w, h], col(c, a)),
                    &Prim::Circle { x, y, r, c, a, .. } => bare('b', [x - r, y + dy - r, 2.0 * r, 2.0 * r], col(c, a)),
                    Prim::Dashed { pts, c, a, .. } => bare('d', [pts[0][0], pts[0][1] + dy, pts[1][0] - pts[0][0], 0.0], col(*c, *a)),
                    Prim::Line { pts, c, a, .. } => {
                        let (x0, x1) = pts.iter().fold((f64::MAX, f64::MIN), |q, p| (q.0.min(p[0]), q.1.max(p[0])));
                        let (y0, y1) = pts.iter().fold((f64::MAX, f64::MIN), |q, p| (q.0.min(p[1]), q.1.max(p[1])));
                        bare('r', [(x0 + x1) / 2.0, (y0 + y1) / 2.0 + dy, 0.0, 0.0], col(*c, *a))
                    }
                    &Prim::StrokeRect { x, y, w, h, c, a, lw, .. } => bare('o', [x - lw / 2.0, y + dy - lw / 2.0, w + lw, h + lw], col(c, a)),
                    Prim::Area { .. } => panic!("면은 안 쓴다"),
                });
            }
        }
        out
    }
    /// 같은 것: 종류·글·크기·굵기·색(그려지는 8비트 ±1) · 자리(글 x ±.6 · 가운데 y ±1.2 · 상자·선 ±.06)
    pub(crate) fn same(a: &It, e: &It) -> bool {
        let cs = (0..4).all(|k| (a.c[k] - e.c[k]).abs() <= 1.0);
        let n = |x: f64, y: f64, t: f64| (x - y).abs() <= t;
        cs && a.k == e.k && match a.k {
            't' => a.s == e.s && a.size == e.size && a.wt == e.wt && a.mono == e.mono && n(a.r[0], e.r[0], 0.6) && n(a.r[1], e.r[1] + e.r[3] / 2.0, 1.2),
            'r' => n(a.r[0], e.r[0], 0.06) && n(a.r[1], e.r[1], 0.06),
            'd' => n(a.r[0], e.r[0], 0.06) && n(a.r[1], e.r[1], 0.06) && n(a.r[2], e.r[2], 0.06),
            _ => (0..4).all(|k| n(a.r[k], e.r[k], 0.06)),
        }
    }
    /// 하나씩 짝짓기 -- (짝 없는 rtui, 짝 없는 원본)
    pub(crate) fn pair(act: &[It], exp: &[It]) -> (Vec<It>, Vec<It>) {
        let mut used = vec![false; act.len()];
        let mut miss = vec![];
        for e in exp { match (0..act.len()).find(|&k| !used[k] && same(&act[k], e)) { Some(k) => used[k] = true, None => miss.push(e.clone()) } }
        (act.iter().zip(&used).filter(|q| !q.1).map(|q| q.0.clone()).collect(), miss)
    }
    /// 글자 폭 = 크로미움이 잰 폭(글 마디 상자 -- 앞뒤 빈칸 포함)
    pub(crate) fn lookup(widths: &[(String, f64, u16, bool, f64)]) -> impl Fn(&str, f64, u16, bool) -> f64 + '_ {
        move |s: &str, z: f64, wt: u16, mono: bool| {
            let (s, wt) = (untnum(s), wb(wt as f64, mono));
            let hit = |eq: &dyn Fn(&str) -> bool| widths.iter().find(|q| eq(&q.0) && q.1 == z && q.2 == wt && q.3 == mono).map(|q| q.4);
            hit(&|r| r == s).or_else(|| hit(&|r| r.trim() == s.trim())).unwrap_or_else(|| panic!("잰 폭 없음: {s:?} {z} {wt} {mono}"))
        }
    }
    /// 한 장 맞대기 → (짝 없는 rtui, 짝 없는 원본, rtui 높이)
    pub(crate) fn check(s: &Value, d: &Value, w: f64) -> (Vec<It>, Vec<It>, f64) {
        let (exp, widths) = expected(d);
        let m = lookup(&widths);
        let (g, h) = section(&input(s), w, &m);
        let (a, b) = pair(&actual(&g, &m), &exp);
        (a, b, h)
    }

    /// 지금 화면의 실제 함수(compare/fc_js.mjs)가 지은 HTML 을 원본 styles.css 로 헤드리스 크로미움에서 잰 것(compare/fc_layout.py) -- 글·색·자리가 전부 하나씩 짝지어지고 높이가 같아야 한다
    #[test]
    fn js_parity() {
        // FC_JS = 변이 시험용 다른 잰 값(compare/fc_layout.py FC_OUT)
        let (inp, js) = (fx("fc_in.json"), std::env::var("FC_JS").map(|p| serde_json::from_str::<Value>(&std::fs::read_to_string(p).unwrap()).unwrap()).unwrap_or_else(|_| fx("fc_js.json")));
        let w = f(&js["w"]);
        let mut n = 0;
        for (k, (s, d)) in inp.as_array().unwrap().iter().zip(js["cases"].as_array().unwrap()).enumerate() {
            let (extra, miss, h) = check(s, d, w);
            let ln = |v: &[It]| v.iter().map(|q| format!("{} {:?} {:?} {:?} {} {}{}", q.k, q.s, q.r.map(|x| (x * 1000.0).round() / 1000.0), q.c, q.size, q.wt, if q.mono { " M" } else { "" })).collect::<Vec<_>>().join("\n");
            assert!(extra.is_empty() && miss.is_empty(), "입력 {k}\n남은 rtui\n{}\n못 찾은 원본\n{}", ln(&extra), ln(&miss));
            assert!((h - f(&d["h"])).abs() < 0.02, "입력 {k} 높이 {h} vs {}", d["h"]);
            n += d["texts"].as_array().unwrap().len() + d["boxes"].as_array().unwrap().len() + d["lines"].as_array().unwrap().len();
        }
        assert!(n > 600, "{n}");
    }

    /// 원본 주기(1분·2분·2분·5분·30초·2초) · 코인 갈래(dir4h ETH·SOL·XRP · open-rev·sr-lean ETH) · 숨으면 안 묻고 소켓 닫음 · 허브는 «모든 종목» 가격만(구독 없음)
    /// 응답: 지금 추세만 실패가 ok 아님으로 덮고 나머지는 받아 둔 값 그대로
    #[test]
    fn poll() {
        use crate::feed::{Feed, Kind, Req, Source, Urls};
        let gets = |f: &mut Feed| { let v: Vec<(Kind, String)> = f.out.drain(..).filter_map(|r| match r { Req::Get { kind, url } => Some((kind, url)), _ => None }).collect(); v };
        let mut f = Feed::new("eth", "ETHUSDT", Source::Push, Urls::default(), false);
        f.dirs_only = true;
        f.tick(1e6, false, false);
        assert!(matches!(&f.out[..2], [Req::WsOpen(_), Req::WsSend(s)] if s == r#"{"subscribe":[]}"#), "{:?}", f.out);
        let kinds = |v: Vec<(Kind, String)>| v.into_iter().map(|q| q.0).collect::<Vec<_>>();
        assert_eq!(kinds(gets(&mut f)), vec![Kind::Trend, Kind::Dir4h, Kind::OpenRev, Kind::Band, Kind::Radar, Kind::Srl, Kind::DirShort, Kind::PredRec]);
        f.tick(1e6 + 2000.0, false, false);
        assert_eq!(kinds(gets(&mut f)), vec![Kind::Srl]);
        f.tick(1e6 + 30_000.0, false, false);
        assert_eq!(kinds(gets(&mut f)), vec![Kind::Radar, Kind::Srl]);
        f.tick(1e6 + 60_000.0, false, false);
        assert_eq!(kinds(gets(&mut f)), vec![Kind::Trend, Kind::Radar, Kind::Srl, Kind::DirShort, Kind::PredRec]);
        f.tick(1e6 + 120_000.0, true, false);
        assert_eq!(f.out.drain(..).collect::<Vec<_>>(), vec![Req::WsClose], "숨으면 안 묻는다");
        f.tick(1e6 + 120_500.0, false, false);
        assert_eq!(kinds(gets(&mut f)), vec![Kind::Trend, Kind::Dir4h, Kind::OpenRev, Kind::Radar, Kind::Srl, Kind::DirShort, Kind::PredRec]);
        f.tick(1e6 + 300_000.0, false, false);
        assert!(gets(&mut f).contains(&(Kind::Band, "/api/regime-band".into())));
        for (a, sym, want) in [("sol", "SOLUSDT", vec!["/api/trend?asset=sol", "/api/dir4h?asset=sol", "/api/regime-band", "/api/risk-radar", "/api/dir-short?asset=sol", "/api/pred-record"]),
                               ("btc", "BTCUSDT", vec!["/api/trend?asset=btc", "/api/regime-band", "/api/risk-radar", "/api/dir-short?asset=btc", "/api/pred-record"]),
                               ("soxl", "SOXLUSDT", vec!["/api/trend?asset=soxl", "/api/regime-band", "/api/risk-radar", "/api/pred-record"])] {
            let mut g = Feed::new(a, sym, Source::Push, Urls::default(), false);
            g.dirs_only = true;
            g.tick(1e6, false, false);
            assert_eq!(gets(&mut g).into_iter().map(|q| q.1).collect::<Vec<_>>(), want, "{a}");
        }
        f.on_fc(Kind::Trend, Some(r#"{"ok":true,"signal":1}"#));
        f.on_fc(Kind::Trend, None);
        assert_eq!(f.fc[0], Some(serde_json::json!({ "ok": false })));
        f.on_fc(Kind::Radar, Some(r#"{"ok":true}"#));
        f.on_fc(Kind::Radar, None);
        f.on_fc(Kind::Radar, Some("<html>"));
        assert_eq!(f.fc[4], Some(serde_json::json!({ "ok": true })));
        f.on_fc(Kind::Trend, Some("<html>"));
        assert_eq!(f.fc[0], Some(serde_json::json!({ "ok": false })), "못 읽음도 실패");
    }
}
