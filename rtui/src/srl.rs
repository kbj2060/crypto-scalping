//! «레벨 5분» 칸(⑤ 예측 줄의 ETH 전용 지지·저항 칸 · app.js srLeanCell 5835 · srLeanRead 5829 · fcRowHtml 안 cell 6004 · ETH 만 6094).
//! 자료 = 같은 출처 /api/sr-lean(2초 · app.js refreshSrLean 5812 · ETH 만 · 실패·못 읽음 = 받아 둔 값 그대로 · ok 일 때 서버 시계 차 srLeanSkew).
//! 자리 = styles.css(이 작업 시점) 2842~2857 · 2869~2873 · 3232~3235: 칸 = [이름 56][12][게이지 1fr][12][적중 기록 1fr] · 안쪽 여백 6/4 ·
//!   게이지 = 판정 글자 자리 20 + 막대 4 + 아래 6(좌우 6+4 안) · 판정 글자 15px 700 이 점 위(bottom 10 · 점 따라 x% 만큼 왼쪽으로) · 게이지 없으면 가운데 ·
//!   오른쪽 = 적중 막대(높이 18 · 아래 8 점선 · 막대 폭 2~8 · 사이 2 · 오른쪽 붙임) + [적중률 800][ 분모 500] 11px.
//! 안 옮긴 것: 칸 툴팁(title -- 설명·남은 초·48시간 건수) · 차트 위 표식(renderSrLean -- 풋프린트 카드 몫) · 넓은 화면 아닌 판(지지·저항 게이지 행 srRowsHtml -- 휴대폰·세로).
//! ponytail: 굵기 500 = 400 글꼴 · 게이지 양끝 띠의 한쪽만 둥근 모서리 = 네 모서리 둥근(2px) · 점선 3/3 -- 글꼴·모서리별 반지름을 더하면 된다.
use crate::fp::{Anchor, Clip, Col, Prim};
use serde_json::Value;

/// 5810 SR_LEAN_HZ -- 판정 다음 초부터 300초(+ 초가 닫히는 여유)
pub const HZ: f64 = 305.0;

/// 5829 srLeanRead 의 보이는 몫: (이어짐, 가격이 먼저 갈 쪽이 위, 진행 중)
fn read(e: &Value, now_s: f64) -> (bool, bool, bool) {
    let cont = e["grp"] == "top";
    let side = e["side"].as_f64().unwrap_or(f64::NAN);
    let up = (if cont { side } else { -side }) > 0.0;
    let live = e["y"].is_null() && now_s - e["d"].as_f64().unwrap_or(f64::NAN) < HZ;
    (cont, up, live)
}

pub enum Right { Wait, Rec { bars: Vec<bool>, pct: String, pc: Option<Col>, frac: String } }
/// c = 칸 색(--fc) · gauge = 점 자리 %(None = 게이지 없음 · 글자 가운데)
pub struct Cell { pub c: Col, pub main: String, pub gauge: Option<f64>, pub right: Right }

/// JS 숫자 → 글자(정수면 소수점 없이)
fn js(v: &Value) -> String { match v.as_f64() { Some(x) if x.fract() == 0.0 && x.abs() < 1e15 => format!("{}", x as i64), Some(x) => format!("{x}"), None => "undefined".into() } }

/// 5835 srLeanCell -- r = 받아 둔 응답(None = 아직) · now_s = 서버 시계(srLeanNowS)
pub fn cell(r: Option<&Value>, now_s: f64) -> Cell {
    let ok = r.is_some_and(|r| r["ok"] == true);
    let rec = r.filter(|_| ok).map(|r| &r["rec"]).filter(|x| x["n"].as_f64().is_some_and(|n| n != 0.0));
    let right = match rec {
        None => Right::Wait,
        Some(x) => {
            let (hit, n) = (x["hit"].as_f64().unwrap_or(f64::NAN), x["n"].as_f64().unwrap());
            let q = hit / n;
            Right::Rec { bars: x["recent"].as_array().into_iter().flatten().map(|b| match &b[1] { Value::Bool(v) => *v, v => v.as_f64().is_some_and(|f| f != 0.0) }).collect(),
                         pct: js(&Value::from((100.0 * hit / n + 0.5).floor())), pc: if q >= 0.55 { Some(Col::Good) } else if q <= 0.45 { Some(Col::Bad) } else { None },
                         frac: format!("{}/{}", js(&x["hit"]), js(&x["n"])) }
        }
    };
    let Some(r) = r.filter(|_| ok) else { return Cell { c: Col::Muted, main: "-".into(), gauge: None, right } };
    let warm = r["warm_until"].as_f64().is_some_and(|w| now_s < w);
    let stale = r["stale"].as_bool() == Some(true);
    let cur = if stale || warm { None } else { r["ev"].as_array().into_iter().flatten().rev().map(|e| read(e, now_s)).find(|q| q.2) };
    let Some((cont, up, _)) = cur else {
        return Cell { c: Col::Muted, main: (if stale { "멈춤" } else if warm { "준비 중" } else { "대기" }).into(), gauge: Some(50.0), right };
    };
    let pct = js(&r["pct"][if cont { "top" } else { "bot" }]);
    Cell { c: if up { Col::Good } else { Col::Bad }, main: format!("{} {} {pct}%", if up { "▲" } else { "▼" }, if cont { "이어짐" } else { "되돌림" }), gauge: Some(if up { 95.0 } else { 5.0 }), right }
}

/// 칸 높이 -- 게이지 있으면 30(20 + 4 + 6) · 없으면 적중 줄 18 이 정한다 · + 위아래 6
pub fn height(c: &Cell) -> f64 { if c.gauge.is_some() { 42.0 } else { 30.0 } }

/// 한 장(w = 칸 폭) -- 자리는 머리 주석의 CSS 계산
pub fn scene(c: &Cell, w: f64, measure: &dyn Fn(&str, f64, u16) -> f64) -> Vec<Prim> {
    let t = |x, y, s: String, size, wt, c, anchor| Prim::Text { x, y, s, size, wt, c, a: 1.0, anchor, halo: false, clip: Clip::No };
    let rect = |x, y, w, h, c, a, rx| Prim::Rect { x, y, w, h, c, a, rx, clip: Clip::No };
    let cy = height(c) / 2.0;
    let base = |size: f64| cy + size * 0.36; // 가운데 줄 글자 바닥선(mc.rs 와 같은 근사)
    let cw = (w - 8.0 - 56.0 - 24.0) / 2.0;
    let (x2, x3) = (4.0 + 56.0 + 12.0, 4.0 + 56.0 + 12.0 + cw + 12.0);
    let mut v = vec![t(4.0, base(11.0), "레벨 5분".into(), 11.0, 700, Col::Muted, Anchor::Start)];
    match c.gauge {
        Some(x) => {
            let (bx, bw) = (x2 + 6.0 + 4.0, cw - 12.0 - 8.0);
            v.extend([rect(bx, 26.0, bw, 4.0, Col::Muted, 0.18, 2.0), rect(bx, 26.0, bw * 0.1, 4.0, Col::Bad, 0.45, 2.0), rect(bx + bw * 0.9, 26.0, bw * 0.1, 4.0, Col::Good, 0.45, 2.0),
                      Prim::Circle { x: bx + bw * x / 100.0, y: 28.0, r: 4.0, c: Col::Text, a: 1.0, clip: Clip::No }]);
            let lw = measure(&c.main, 15.0, 700);
            v.push(t(bx + (bw - lw) * x / 100.0, 5.0 + 7.5 + 15.0 * 0.36, c.main.clone(), 15.0, 700, c.c, Anchor::Start)); // 상자 5~20
        }
        None => v.push(t(x2 + cw / 2.0, base(15.0), c.main.clone(), 15.0, 700, c.c, Anchor::Middle)),
    }
    match &c.right {
        Right::Wait => v.push(t(x3, base(11.0), "쌓는 중".into(), 11.0, 400, Col::Muted, Anchor::Start)),
        Right::Rec { bars, pct, pc, frac } => {
            let (p1, p2) = (crate::fp::tnum(&format!("{pct}%")), crate::fp::tnum(&format!(" {frac}")));
            let (w1, w2) = (measure(&p1, 11.0, 800), measure(&p2, 11.0, 400));
            let s = (cw - 8.0 - w1 - w2).max(0.0);
            let n = bars.len() as f64;
            let bw = ((s - 2.0 * (n - 1.0)) / n).clamp(2.0, 8.0);
            v.push(Prim::Dashed { pts: vec![[x3, cy + 0.5], [x3 + s, cy + 0.5]], c: Col::Line, a: 1.0, lw: 1.0, on: 3.0, off: 3.0, clip: Clip::No });
            for (i, &u) in bars.iter().enumerate() {
                let r = x3 + s - (n - 1.0 - i as f64) * (bw + 2.0); // 오른쪽 끝 · 넘치면 왼쪽이 잘린다(overflow hidden)
                let l = (r - bw).max(x3);
                let h = if u { 16.0 } else { 6.0 };
                if r > l { v.push(rect(l, cy + 9.0 - h, r - l, h, if u { Col::Good } else { Col::Bad }, 0.7, 1.0)); }
            }
            v.push(t(x3 + s + 8.0, base(11.0), p1, 11.0, 800, pc.unwrap_or(Col::Text), Anchor::Start));
            v.push(t(x3 + s + 8.0 + w1, base(11.0), p2, 11.0, 400, Col::Muted, Anchor::Start));
        }
    }
    v
}

#[cfg(test)]
mod t {
    use super::*;

    /// 지금 화면의 실제 함수로 만든 정답지(compare/srl_js.mjs → fixture/srl_js.json · 같은 입력 srl_in.json)
    #[test]
    fn js_parity() {
        let fx = |f: &str| serde_json::from_str::<Value>(&std::fs::read_to_string(format!("{}/compare/fixture/{f}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap();
        let (inp, js) = (fx("srl_in.json"), fx("srl_js.json"));
        let mut n = 0;
        for (k, (i, j)) in inp.as_array().unwrap().iter().zip(js.as_array().unwrap()).enumerate() {
            let c = cell(Some(&i[0]).filter(|r| !r.is_null()), i[1].as_f64().unwrap());
            let cls = |c: Col| match c { Col::Good => "up", Col::Bad => "dn", _ => "flat" };
            assert_eq!((j["hz"].as_str().unwrap(), cls(c.c), c.main.as_str(), c.gauge), ("레벨 5분", j["c"].as_str().unwrap(), j["main"].as_str().unwrap(), j["gauge"].as_f64()), "입력 {k}");
            // 게이지 양끝 띠는 늘 10%
            if c.gauge.is_some() { assert_eq!(j["zones"], serde_json::json!([["dn", "width", 10], ["up", "left", 90]]), "입력 {k}"); }
            match &c.right {
                Right::Wait => assert_eq!((j["bars"].is_null(), j["note"].as_str()), (true, Some("쌓는 중")), "입력 {k}"),
                Right::Rec { bars, pct, pc, frac } => {
                    let jb: Vec<(bool, f64)> = serde_json::from_value(j["bars"].clone()).unwrap();
                    assert_eq!(bars.iter().map(|&u| (u, if u { 16.0 } else { 6.0 })).collect::<Vec<_>>(), jb, "입력 {k}");
                    let pcls = match pc { Some(Col::Good) => "up", Some(Col::Bad) => "dn", _ => "" };
                    assert_eq!((format!("{pct}%"), pcls, frac.as_str()), (j["pct"].as_str().unwrap().to_string(), j["pcls"].as_str().unwrap(), j["frac"].as_str().unwrap()), "입력 {k}");
                }
            }
            n += 1;
        }
        assert!(n >= 12, "{n}");
    }

    /// 자리: 판정 글자 = 점 위(x% 만큼 왼쪽으로) · 막대는 오른쪽 붙임 · 넘치면 왼쪽이 잘린다
    #[test]
    fn layout() {
        let m = |s: &str, z: f64, _: u16| s.chars().count() as f64 * z * 0.5;
        let c = Cell { c: Col::Good, main: "▲ 이어짐 56%".into(), gauge: Some(95.0), right: Right::Rec { bars: vec![true; 40], pct: "50".into(), pc: None, frac: "20/40".into() } };
        let v = scene(&c, 404.0, &m);
        // cw = (404 − 88)/2 = 158 · 막대 x = 72 + 10 = 82 · 폭 138 · 점 = 82 + 131.1
        assert!(v.iter().any(|p| matches!(p, Prim::Circle { x, y: 28.0, .. } if (*x - (82.0 + 138.0 * 0.95)).abs() < 1e-9)));
        let lw = 9.0 * 7.5;
        assert!(v.iter().any(|p| matches!(p, Prim::Text { x, s, c: Col::Good, .. } if s == "▲ 이어짐 56%" && (*x - (82.0 + (138.0 - lw) * 0.95)).abs() < 1e-9)));
        let x3 = 72.0 + 158.0 + 12.0;
        let bars: Vec<(f64, f64)> = v.iter().filter_map(|p| if let Prim::Rect { x, w, c: Col::Good, a, .. } = p { (*a == 0.7).then_some((*x, *w)) } else { None }).collect();
        assert!(bars.iter().all(|b| b.0 >= x3 && b.1 >= 0.0) && bars.len() < 40 && (bars[0].0 - x3).abs() < 1e-9, "{bars:?}");
        let g = cell(None, 0.0);
        assert_eq!((height(&g), g.main.as_str()), (30.0, "-"));
        assert!(matches!(&scene(&g, 404.0, &m)[1], Prim::Text { anchor: Anchor::Middle, x, .. } if *x == 72.0 + 79.0));
    }

    /// ETH 만 2초마다 · 숨으면 안 묻는다 · 실패·못 읽음 = 받아 둔 값 그대로 · ok 면 서버 시계 차
    #[test]
    fn poll() {
        use crate::feed::{Feed, Kind, Req, Source, Urls};
        let mut f = Feed::new("eth", "ETHUSDT", Source::Poll, Urls::default(), false);
        f.srl_only = true;
        for (t, hid) in [(2000.0, false), (3000.0, false), (4100.0, true), (4200.0, false), (5000.0, false), (6300.0, false)] { f.tick(t, hid, false); }
        let urls: Vec<_> = f.out.iter().map(|r| match r { Req::Get { kind: Kind::Srl, url } => url.clone(), _ => panic!("다른 것을 묻지 않는다") }).collect();
        assert_eq!(urls, vec!["/api/sr-lean"; 3]);
        f.on_srl(Some(r#"{"ok":true,"now":1000.5,"stale":false,"pct":{"top":56,"bot":55},"ev":[{"a":990,"d":990,"side":1,"grp":"top","y":null}]}"#), 999_000.0);
        assert_eq!(f.srl_now(999_000.0 + 2000.0), 1002.5);
        assert_eq!(cell(f.srl.as_ref(), f.srl_now(999_000.0)).main, "▲ 이어짐 56%");
        f.on_srl(None, 0.0);
        f.on_srl(Some("<html>"), 0.0);
        assert!(f.srl.is_some());
        let mut s = Feed::new("sol", "SOLUSDT", Source::Poll, Urls::default(), false);
        s.srl_only = true;
        s.tick(5000.0, false, false);
        assert!(s.out.is_empty(), "ETH 만");
    }
}
