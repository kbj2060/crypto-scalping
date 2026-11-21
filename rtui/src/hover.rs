//! 마우스 반응 첫 걸음: 십자선 · 커서 가격 배지 · 봉/칸/델타·TPO 툴팁. app.js 10680-10903(svg.onmousemove) + showTooltip 11692.
//! 끌기·휠 확대·시간 단위 바꾸기는 다음 조각 -- 붙일 자리: Chart::ui 의 포인터 입력(lib.rs) · fp::Cfg.window.
//! 범위 밖 갈래(다른 응답이 필요): 포지션 기둥(posInfo) · 청산 기둥(liqInfo) · 체결 기둥(tradeInfo, 휴대폰) · 호가 띠(bookInfo) · 네 숫자(statsGeo) · 차트 표식 줄(trigLines).
use crate::data::{fmt_qty, to_fixed};
use crate::fp::{Anchor, Clip, Col, Prim, Scene, tnum};
use crate::lane::qname;
use crate::profile;

/// 툴팁 글자색(.chart-tooltip 11px #fff · b 12px rgba(255,255,255,.72) · span = --muted · 줄마다 style 이 덮는다)
#[derive(Clone, Copy, Debug)]
pub enum Tc { White, Dim, C(Col, f32) }

/// 한 줄 = 글자 조각들(글자 · 색 · 굵기 400|700 -- <b>·font-weight:700 조각) · 크기 · 아래 여백(b 블록 margin-bottom 4px) · 빈 줄 = <br> 만 있는 줄
/// 글꼴 = --font-mono(JetBrains Mono · 한글은 Pretendard 로 떨어짐 -- draw::mono)
pub struct TipLine { pub runs: Vec<(String, Tc, u16)>, pub size: f32, pub mb: f32 }

/// 툴팁: 줄 · 줄바꿈 폭(white-space:normal + max-width 360 -- 델타·TPO 풀이) · 마우스(화면 px) · 그린 상자(draw::tip 이 채운다)
pub struct Tip { pub lines: Vec<TipLine>, pub wrap: Option<f32>, pub at: [f64; 2], pub x: f64, pub y: f64, pub w: f64, pub h: f64 }

pub struct Hover { pub prims: Vec<Prim>, pub tip: Option<Tip> }

fn line(runs: Vec<(String, Tc, u16)>) -> TipLine { TipLine { runs, size: 11.0, mb: 0.0 } }
fn b(s: String, c: Tc) -> TipLine { TipLine { runs: vec![(s, c, 700)], size: 12.0, mb: 4.0 } } // <b> = bold
fn w(s: impl Into<String>) -> (String, Tc, u16) { (s.into(), Tc::White, 400) }
fn round(v: f64) -> f64 { (v + 0.5).floor() } // JS Math.round
fn sgn_q(v: f64) -> String { format!("{}{}", if v >= 0.0 { "+" } else { "−" }, fmt_qty(v.abs())) } // 11132 sgnQ
fn usd(v: f64) -> String { // 2064 fmtUsdCompact
    if v >= 1e6 { format!("${}M", to_fixed(v / 1e6, 2)) } else if v >= 1e3 { format!("${}k", to_fixed(v / 1e3, 1)) } else { format!("${}", to_fixed(v, 0)) }
}

/// (mx, my) = svg 좌표 · at = 마우스 화면 px(툴팁 자리) · measure = 글자 폭
pub fn hover(sc: &Scene, mx: f64, my: f64, at: [f64; 2], tz_off_min: i64, measure: &dyn Fn(&str, f64, u16) -> f64) -> Hover {
    let g = &sc.g;
    let (mut prims, mut tip) = (vec![], None);
    let n = Clip::No;
    let dash = |pts: Vec<[f64; 2]>| Prim::Line { pts, c: Col::HoverLine, a: 1.0, lw: 1.0, dash: Some(4.0), clip: n };
    let price_at = g.y_max - (my - g.mt) * g.y_span / g.ch; // 10758 · 10782
    // 10755-10770 가로 십자선 + 커서 가격 배지(현재가 알약과 같은 오른쪽 끝 · 폭 = 글자 + 18 · 고정폭 숫자)
    if my >= g.mt && my <= g.plot_bottom {
        prims.push(dash(vec![[g.ml, my], [g.w - g.mr, my]]));
        let by = g.mt.max((g.plot_bottom - 18.0).min(my - 9.0));
        let s = tnum(&to_fixed(price_at, sc.dp));
        let bw = measure(&s, 12.0, 700) + 18.0;
        prims.push(Prim::Rect { x: g.x_rr - bw, y: by, w: bw, h: 18.0, c: Col::Accent, a: 1.0, rx: 9.0, clip: n });
        prims.push(Prim::Text { x: g.x_rr - 9.0, y: by + 13.0, s, size: 12.0, wt: 700, c: Col::OnFill, a: 1.0, anchor: Anchor::End, halo: false, clip: n });
    }
    let (d_x0, _, t_x0) = profile::cols(g);
    if let Some(dt) = sc.dt.as_ref().filter(|_| mx > d_x0 && mx < t_x0 + g.dt_w) {
        // 10827-10840 델타·TPO 칸 위: 그 가격 행 풀이
        let k = ((price_at + g.off) / g.row_size).floor() as i64;
        if let Some(a) = dt.rows.get(&k) {
            let fq = |q: f64| if q.abs() >= 1000.0 { to_fixed(q / 1000.0, 1) + "k" } else if q.abs() >= 10.0 { to_fixed(q, 0) } else { to_fixed(q, 1) };
            let dp = 1.max(format!("{}", g.off).split('.').nth(1).map_or(0, str::len));
            let pct = |v: f64| format!("{}%", round(100.0 * v / a[2].max(1e-9)));
            let up = a[0] >= 0.0;
            let mut l4 = vec![(format!("TPO {}/{}봉", a[1], dt.n_bars), Tc::C(Col::Muted, 1.0), 700), w(" — 이 가격에 체결이 있던 봉 수")];
            if k == dt.t_k { l4.push(w(" · 최다(POC)")); }
            l4.push(w(if k >= dt.va_lo && k <= dt.va_hi { " · 가치 영역(70%) 안" } else { " · 가치 영역 밖" }));
            if a[1] == 1.0 { l4.push(w(" · 한 번만 지난 가격")); }
            tip = Some(vec![
                line(vec![w(format!("{}–{}", to_fixed(k as f64 * g.row_size - g.off, dp), to_fixed((k + 1) as f64 * g.row_size - g.off, dp)))]),
                line(vec![(format!("델타 {}{} {}", if up { "+" } else { "−" }, fq(a[0].abs()), sc.unit), Tc::C(if up { Col::Good } else { Col::Bad }, 1.0), 700),
                          w(format!(" (오른쪽 막대 · {} 우위 · 테이커 기준)", if up { "매수" } else { "매도" }))]),
                line(vec![(format!("체결 {} {}", fq(a[2]), sc.unit), Tc::C(Col::Amber, 1.0), 700),
                          w(format!(" (왼쪽 막대) · 고래 {} · 리테일 {}{}", pct(a[3]), pct(a[4]), if Some(k) == dt.tot_k { " · 최다 체결" } else { "" }))]),
                line(l4),
                line(vec![(format!("보이는 봉 {}개 합 · 30분 문자표 TPO 가 아니라 5분봉 수 · 참고, 신호 아님", dt.n_bars), Tc::C(Col::Muted, 0.7), 400)]),
            ]).map(|lines| Tip { lines, wrap: Some(360.0), at, x: 0.0, y: 0.0, w: 0.0, h: 0.0 });
        }
    } else if !(mx < g.ml || mx > g.w - g.mr || mx > g.ml + g.cw) { // 10870 기둥 칸 빈 곳 = 봉 툴팁 아님
        // 10872-10902 봉 툴팁: 세로 십자선 + 시각·OHLC + 커서 밑 칸 + 레인 값
        let last = sc.candles.len() - 1;
        let idx = (((mx - g.ml) / g.slot - g.x0b).floor().max(0.0) as usize).min(last);
        let c = &sc.candles[idx];
        let tx = g.x_at(idx) + g.bw / 2.0;
        prims.push(dash(vec![[tx, g.mt], [tx, g.h - g.mb]]));
        let t = c.time - tz_off_min * 60;
        let mut lines = vec![b(format!("{:02}:{:02}", t.rem_euclid(86_400) / 3600, t.rem_euclid(3600) / 60), Tc::Dim), line(vec![])];
        for (nm, v) in [("시가", c.open), ("고가", c.high), ("저가", c.low), ("종가", c.close)] { lines.push(line(vec![w(format!("{nm}: {}", to_fixed(v, sc.dp.max(2))))])); }
        // 9560-9565 cellTip -- 칸을 그리는 폭에서만
        if my >= g.mt && my <= g.plot_bottom && g.draw_cells {
            let k = g.row_of(price_at);
            if let Some(&[bq, sq]) = sc.bar_rows[idx].get(&k) {
                let imb = |v: f64, o: f64| if v > 0.0 && v > o * 3.0 { format!(" · 불균형 {}", if o > 0.0 { to_fixed(v / o, 1) + "배" } else { "일방".into() }) } else { String::new() };
                lines.push(line(vec![w(format!("칸 {} · 매수 {}{} · 매도 {}{}{}", to_fixed(g.row_lo(k) + g.row_size / 2.0, sc.dp), to_fixed(bq, 1), imb(bq, sq), to_fixed(sq, 1), imb(sq, bq),
                                                if Some(k) == sc.pocs[idx] { " · POC(최다 체결)" } else { "" }))]));
            }
        }
        // 11277-11282 liqTip: 청산 금액(롱/숏) · 진행 중 · 거래소별(liqVenueText 276)
        if let Some(lb) = sc.liq.get(idx).and_then(|x| x.as_ref()).filter(|b| b.long_usd + b.short_usd > 0.0) {
            lines.push(line(vec![w(format!("청산 {} (롱 {} / 숏 {}){}", usd(lb.long_usd + lb.short_usd), usd(lb.long_usd), usd(lb.short_usd), if lb.partial { " · 진행 중" } else { "" }))]));
            for s in liq_venue_text(lb) { lines.push(line(vec![w(s)])); }
        }
        // 11148-11155 laneTip (신규계약 = 모름: OI 응답 범위 밖)
        match &sc.lane.get(idx).and_then(|r| r.as_ref()) {
            None => { lines.push(line(vec![])); lines.push(line(vec![w("풋프린트 없음")])); }
            Some(r) => {
                let col = if r.delta >= 0.0 { Col::Good } else { Col::Bad };
                lines.push(line(vec![]));
                lines.push(b(qname(r.delta).into(), Tc::C(col, 1.0)));
                lines.push(line(vec![w(format!("· 델타 {} · 신규계약 모름 {}", sgn_q(r.delta), sc.unit))]));
                lines.push(line(vec![(format!("거래대금 {}", usd(r.turn)), Tc::C(Col::Turnover, 1.0), 400)]));
                if let Some(d) = r.dhl { lines.push(line(vec![w(format!("봉 안 델타 최고 {} · 최저 {} · 마감 {} {}", sgn_q(d[1]), sgn_q(d[0]), sgn_q(d[2]), sc.unit))])); }
            }
        }
        tip = Some(Tip { lines, wrap: None, at, x: 0.0, y: 0.0, w: 0.0, h: 0.0 });
    }
    Hover { prims, tip }
}

/// 276 liqVenueText 의 줄들(앞 «\n» 마다 한 줄): 금액이 있는 거래소(바이낸스 = 합 − 나머지) · 끝에 «합산: 본 거래소»
fn liq_venue_text(b: &crate::data::LiqBar) -> Vec<String> {
    use crate::data::LiqVenue;
    let u = |x: &LiqVenue| x.long_usd + x.short_usd;
    let parts = [("OKX", &b.okx_detail), ("Bybit", &b.bybit), ("HL 고래", &b.hl)];
    let sum = |f: fn(&LiqVenue) -> f64| parts.iter().map(|(_, x)| x.as_ref().map_or(0.0, f)).sum::<f64>();
    let bn = LiqVenue { long_usd: b.long_usd - sum(|x| x.long_usd), short_usd: b.short_usd - sum(|x| x.short_usd), n: (b.events - sum(|x| x.n)).max(0.0) };
    let js_num = |n: f64| if n.fract() == 0.0 && n.abs() < 1e15 { format!("{}", n as i64) } else { format!("{n}") }; // ponytail: JS Number→글자는 정수만 같게(건수는 정수)
    let mut out: Vec<String> = std::iter::once(("바이낸스", Some(&bn))).chain(parts.iter().map(|(n, x)| (*n, x.as_ref())))
        .filter_map(|(nm, x)| x.filter(|x| u(x) >= 0.5).map(|x| format!("{nm} {} {}건(롱 {}/숏 {})", usd(u(x)), js_num(x.n), usd(x.long_usd.max(0.0)), usd(x.short_usd.max(0.0)))))
        .collect();
    let seen: Vec<&str> = [Some("바이낸스"), b.okx.then_some("OKX"), b.bybit.as_ref().map(|_| "Bybit"), b.hl.as_ref().map(|_| "HL 고래(추적 300지갑)")].into_iter().flatten().collect();
    out.push(format!("합산: {}", seen.join(" · ")));
    out
}
