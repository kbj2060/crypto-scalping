//! «레벨 5분» 칸(⑤ 예측 줄의 ETH 전용 지지·저항 칸 · app.js srLeanCell 5835 · srLeanRead 5829 · fcRowHtml 안 cell 6004 · ETH 만 6094).
//! 자료 = 같은 출처 /api/sr-lean(2초 · app.js refreshSrLean 5812 · ETH 만 · 실패·못 읽음 = 받아 둔 값 그대로 · ok 일 때 서버 시계 차 srLeanSkew).
//! 자리 = styles.css(이 작업 시점) 2832 · 2842~2857 · 2869~2873 · 3232~3235: 칸 = [이름 56][2][게이지 1fr][2][적중 기록 1fr] · 안쪽 여백 6/4 ·
//!   (11-29 고침: 넓은 화면 #mcBody.mc-cmp .fc-c gap 2 · .fc-t > .d4g 가로 여백 0 -- 전엔 12·4 로 잡아 게이지·막대가 4~10px 어긋났다 · 헤드리스 크로미움 실측 = 시험 layout)
//!   게이지 = 판정 글자 자리 20 + 막대 4 + 아래 6(좌우 6 안) · 판정 글자 15px 700 이 점 위(bottom 10 · 점 따라 x% 만큼 왼쪽으로) · 게이지 없으면 가운데 ·
//!   오른쪽 = 적중 막대(높이 18 · 아래 8 점선 · 막대 폭 2~8 · 사이 2 · 오른쪽 붙임) + [적중률 800][ 분모 500] 11px.
//! 안 옮긴 것: 칸 툴팁(title -- 설명·남은 초·48시간 건수) · 차트 위 표식(renderSrLean -- 풋프린트 카드 몫) · 넓은 화면 아닌 판(지지·저항 게이지 행 srRowsHtml -- 휴대폰·세로).
//! ponytail: 굵기 500 = 400 글꼴 · 게이지 양끝 띠의 한쪽만 둥근 모서리 = 네 모서리 둥근(2px) · 점선 3/3 -- 글꼴·모서리별 반지름을 더하면 된다.
use crate::fp::{Anchor, Clip, Col, Prim};
use serde_json::Value;

/// 칸 이름
const HZ_NAME: &str = "레벨 5분";
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

/// 오른쪽 칸(.fz-r): None = 칸 없음 · Note = 글만(.pr-n) · Rec = 막대(맞음 Some(true) 초록·Some(false) 빨강·None 회색, 높이 px) + 적중률(글·색·분모) 또는 None = «쌓는 중»
pub enum Right { None, Note(String), Rec { bars: Vec<(Option<bool>, f64)>, pct: Option<(String, Option<Col>, String)> } }
/// ⑤ 예측 줄 한 칸(fcRowHtml 안 cell) -- hz = 이름 · c = 칸 색(--fc) · gauge = 점 자리 %(None = 게이지 없음 · 글자 가운데) · zone = 양끝 띠 [왼 띠 폭 %, 오른 띠 시작 %] · ago = 빈 점 자리 %
/// risk = 띠 색이 위험 레이더(.z.rq 회색 34% · .z.rl 주황 45%) -- 아니면 .z.dn 빨강 · .z.up 초록 45%
pub struct Cell { pub hz: String, pub c: Col, pub main: String, pub gauge: Option<f64>, pub zone: [f64; 2], pub risk: bool, pub ago: Option<f64>, pub right: Right }

/// JS 숫자 → 글자(정수면 소수점 없이)
pub(crate) fn js(v: &Value) -> String { match v.as_f64() { Some(x) if x.fract() == 0.0 && x.abs() < 1e15 => format!("{}", x as i64), Some(x) => format!("{x}"), None => "undefined".into() } }

/// 5835 srLeanCell -- r = 받아 둔 응답(None = 아직) · now_s = 서버 시계(srLeanNowS)
pub fn cell(r: Option<&Value>, now_s: f64) -> Cell {
    let ok = r.is_some_and(|r| r["ok"] == true);
    let rec = r.filter(|_| ok).map(|r| &r["rec"]).filter(|x| x["n"].as_f64().is_some_and(|n| n != 0.0));
    let right = match rec {
        None => Right::Note("쌓는 중".into()),
        Some(x) => {
            let (hit, n) = (x["hit"].as_f64().unwrap_or(f64::NAN), x["n"].as_f64().unwrap());
            let q = hit / n;
            Right::Rec { bars: x["recent"].as_array().into_iter().flatten().map(|b| { let u = truthy(&b[1]); (Some(u), if u { 16.0 } else { 6.0 }) }).collect(),
                         pct: Some((js(&Value::from((100.0 * hit / n + 0.5).floor())), if q >= 0.55 { Some(Col::Good) } else if q <= 0.45 { Some(Col::Bad) } else { None },
                         format!("{}/{}", js(&x["hit"]), js(&x["n"])))) }
        }
    };
    let Some(r) = r.filter(|_| ok) else { return Cell { hz: HZ_NAME.into(), c: Col::Muted, main: "-".into(), gauge: None, zone: [10.0, 90.0], risk: false, ago: None, right } };
    let warm = r["warm_until"].as_f64().is_some_and(|w| now_s < w);
    let stale = r["stale"].as_bool() == Some(true);
    let cur = if stale || warm { None } else { r["ev"].as_array().into_iter().flatten().rev().map(|e| read(e, now_s)).find(|q| q.2) };
    let Some((cont, up, _)) = cur else {
        return Cell { hz: HZ_NAME.into(), c: Col::Muted, main: (if stale { "멈춤" } else if warm { "준비 중" } else { "대기" }).into(), gauge: Some(50.0), zone: [10.0, 90.0], risk: false, ago: None, right };
    };
    let pct = js(&r["pct"][if cont { "top" } else { "bot" }]);
    Cell { hz: HZ_NAME.into(), c: if up { Col::Good } else { Col::Bad }, main: format!("{} {} {pct}%", if up { "▲" } else { "▼" }, if cont { "이어짐" } else { "되돌림" }), gauge: Some(if up { 95.0 } else { 5.0 }), zone: [10.0, 90.0], risk: false, ago: None, right }
}

/// 칸 높이 -- 게이지 있으면 30(20 + 4 + 6) · 없으면 적중 막대 줄 18 · 막대 없으면(글만 · 칸 없음) 판정 글자 15 · + 위아래 6
/// (11-29 고침: 글만(«쌓는 중») 줄도 30 으로 잡았다 -- 원본은 27 · fc::t::js_parity 실측)
pub fn height(c: &Cell) -> f64 { if c.gauge.is_some() { 42.0 } else if matches!(c.right, Right::Rec { .. }) { 30.0 } else { 27.0 } }

/// JS 참·거짓(0·NaN·null·빈 글 = 거짓)
pub(crate) fn truthy(v: &Value) -> bool {
    match v { Value::Null => false, Value::Bool(b) => *b, Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0), Value::String(s) => !s.is_empty(), _ => true }
}

/// 한 장(w = 칸 폭) -- 자리 = 넓은 화면 오른쪽 칸(#mcBody.mc-cmp) 안 칸을 헤드리스 크로미움에서 잰 값(styles.css 2832 gap 2 · 2848 .fc-t > .d4g 가로 여백 0):
///   [4][이름 56][2][게이지 칸 cw][2][적중 칸 cw][4] · 게이지 = 칸 안쪽 여백 6 안 · 점 8 · 빈 점 7(테 1px -- 1.5px 가 1 로 그려진다)
pub fn scene(c: &Cell, w: f64, measure: &dyn Fn(&str, f64, u16) -> f64) -> Vec<Prim> {
    let t = |x, y, s: String, size, wt, c, anchor| Prim::Text { x, y, s, size, wt, c, a: 1.0, anchor, halo: false, clip: Clip::No };
    let rect = |x, y, w, h, c, a, rx| Prim::Rect { x, y, w, h, c, a, rx, clip: Clip::No };
    let cy = height(c) / 2.0;
    let base = |size: f64| cy + size * 0.36; // 가운데 줄 글자 바닥선(mc.rs 와 같은 근사)
    let cw = (w - 8.0 - 56.0 - 4.0) / 2.0;
    let (x2, x3) = (4.0 + 56.0 + 2.0, 4.0 + 56.0 + 2.0 + cw + 2.0);
    let mut v = vec![t(4.0, base(11.0), c.hz.clone(), 11.0, 700, Col::Muted, Anchor::Start)];
    match c.gauge {
        Some(x) => {
            let (bx, bw, [z0, z1]) = (x2 + 6.0, cw - 12.0, c.zone.map(|z| z / 100.0));
            let ((cl, al), (cr, ar)) = if c.risk { ((Col::Muted, 0.34), (Col::Amber, 0.45)) } else { ((Col::Bad, 0.45), (Col::Good, 0.45)) };
            v.extend([rect(bx, 26.0, bw, 4.0, Col::Muted, 0.18, 2.0), rect(bx, 26.0, bw * z0, 4.0, cl, al, 2.0), rect(bx + bw * z1, 26.0, bw * (1.0 - z1), 4.0, cr, ar, 2.0)]);
            if let Some(a) = c.ago { // ponytail: 빈 점 = 꺾은선 24조각 -- 테 그리기가 생기면 그것으로
                let (ax, r) = (bx + bw * a / 100.0, 3.0);
                v.push(Prim::Line { pts: (0..=24).map(|i| { let q = i as f64 * std::f64::consts::TAU / 24.0; [ax + r * q.cos(), 28.0 + r * q.sin()] }).collect(), c: Col::Muted, a: 1.0, lw: 1.0, dash: None, clip: Clip::No });
            }
            v.push(Prim::Circle { x: bx + bw * x / 100.0, y: 28.0, r: 4.0, c: Col::Text, a: 1.0, clip: Clip::No });
            let lw = measure(&c.main, 15.0, 700);
            v.push(t(bx + (bw - lw) * x / 100.0, 5.0 + 7.5 + 15.0 * 0.36, c.main.clone(), 15.0, 700, c.c, Anchor::Start)); // 상자 5~20
        }
        None => v.push(t(x2 + cw / 2.0, base(15.0), c.main.clone(), 15.0, 700, c.c, Anchor::Middle)),
    }
    match &c.right {
        Right::None => {}
        Right::Note(s) => v.push(t(x3, base(11.0), s.clone(), 11.0, 400, Col::Muted, Anchor::Start)),
        Right::Rec { bars, pct } => {
            let (p1, p2) = match pct { Some((p, _, f)) => (crate::fp::tnum(&format!("{p}%")), crate::fp::tnum(&format!(" {f}"))), None => (String::new(), "쌓는 중".into()) };
            let (w1, w2) = (if p1.is_empty() { 0.0 } else { measure(&p1, 11.0, 800) }, measure(&p2, 11.0, 400));
            let s = (cw - 8.0 - w1 - w2).max(0.0);
            let n = bars.len() as f64;
            let bw = ((s - 2.0 * (n - 1.0)) / n).clamp(2.0, 8.0);
            v.push(Prim::Dashed { pts: vec![[x3, cy + 0.5], [x3 + s, cy + 0.5]], c: Col::Line, a: 1.0, lw: 1.0, on: 3.0, off: 3.0, clip: Clip::No });
            for (i, &(u, h)) in bars.iter().enumerate() {
                let r = x3 + s - (n - 1.0 - i as f64) * (bw + 2.0); // 오른쪽 끝 · 넘치면 왼쪽이 잘린다(overflow hidden)
                let l = (r - bw).max(x3);
                let (c, a) = match u { Some(true) => (Col::Good, 0.7), Some(false) => (Col::Bad, 0.7), None => (Col::Muted, 0.45) };
                if r > l && h > 0.0 { v.push(rect(l, cy + 9.0 - h, r - l, h, c, a, 1.0)); }
            }
            if let Some((_, pc, _)) = pct { v.push(t(x3 + s + 8.0, base(11.0), p1, 11.0, 800, pc.unwrap_or(Col::Text), Anchor::Start)); }
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
                Right::Note(t) => assert_eq!((j["bars"].is_null(), j["note"].as_str()), (true, Some(t.as_str())), "입력 {k}"),
                Right::Rec { bars, pct: Some((pct, pc, frac)) } => {
                    let jb: Vec<(bool, f64)> = serde_json::from_value(j["bars"].clone()).unwrap();
                    assert_eq!(bars.iter().map(|&(u, h)| (u.unwrap(), h)).collect::<Vec<_>>(), jb, "입력 {k}");
                    let pcls = match pc { Some(Col::Good) => "up", Some(Col::Bad) => "dn", _ => "" };
                    assert_eq!((format!("{pct}%"), pcls, frac.as_str()), (j["pct"].as_str().unwrap().to_string(), j["pcls"].as_str().unwrap(), j["frac"].as_str().unwrap()), "입력 {k}");
                }
                _ => panic!("입력 {k}"),
            }
            n += 1;
        }
        assert!(n >= 12, "{n}");
    }

    /// 자리 = 지금 화면 styles.css 를 헤드리스 크로미움에 띄워 잰 값(폭 404 · #mcBody.mc-cmp 안 · 게이지 85.5% · 빈 점 40% · 막대 2개 · «▲ 상승» 34.77px · «56%»+« 9/16» 54.55px):
    /// .d4g x 68 폭 156 · 점 가운데 201.38 · 빈 점 가운데 130.39 · .gl x 171.65 · .pr-sp i x 319.45(폭 8 높이 16 y 14) · .pr-n x 345.45 · 둘째 줄(게이지·오른쪽 없음) y 42 높이 28 · 글 가운데 x 68 + 156/2
    #[test]
    fn layout() {
        let m = |_: &str, z: f64, wt: u16| if z == 15.0 { 34.77 } else if wt == 800 { 30.0 } else { 24.55 };
        let c = Cell { hz: "15분 예측".into(), c: Col::Good, main: "▲ 상승".into(), gauge: Some(85.5), zone: [20.0, 80.0], risk: false, ago: Some(40.0),
                       right: Right::Rec { bars: vec![(Some(true), 16.0), (Some(false), 6.0)], pct: Some(("56".into(), Some(Col::Good), "9/16".into())) } };
        let v = scene(&c, 404.0, &m);
        let near = |a: f64, b: f64| (a - b).abs() < 0.02; // 크로미움 배치 단위 1/64px
        assert!(v.iter().any(|p| matches!(p, Prim::Rect { x, w, y: 26.0, c: Col::Muted, .. } if near(*x, 68.0) && near(*w, 156.0))));
        assert!(v.iter().any(|p| matches!(p, Prim::Rect { x, w, c: Col::Bad, a, .. } if *a == 0.45 && near(*x, 68.0) && near(*w, 31.2))));
        assert!(v.iter().any(|p| matches!(p, Prim::Circle { x, y: 28.0, .. } if near(*x, 201.38))));
        let ring: Vec<_> = v.iter().filter_map(|p| if let Prim::Line { pts, .. } = p { Some(pts) } else { None }).collect();
        assert!(ring.len() == 1 && near(ring[0].iter().map(|q| q[0]).fold(f64::MIN, f64::max) - 3.0, 130.39));
        assert!(v.iter().any(|p| matches!(p, Prim::Text { x, s, c: Col::Good, .. } if s == "▲ 상승" && near(*x, 171.65))));
        assert!(v.iter().any(|p| matches!(p, Prim::Rect { x, w, y, h, c: Col::Good, a, .. } if *a == 0.7 && near(*x, 319.45) && *w == 8.0 && *y == 14.0 && *h == 16.0)));
        assert!(v.iter().any(|p| matches!(p, Prim::Text { x, wt: 800, .. } if near(*x, 345.45))));
        let g = Cell { hz: "1시간 예측".into(), c: Col::Muted, main: "-".into(), gauge: None, zone: [20.0, 80.0], risk: false, ago: None, right: Right::None };
        assert!(matches!(&scene(&g, 404.0, &m)[1], Prim::Text { anchor: Anchor::Middle, x, .. } if near(*x, 68.0 + 78.0)));
        // 넘치면 왼쪽이 잘린다
        let c = Cell { hz: "레벨 5분".into(), c: Col::Good, main: "x".into(), gauge: None, zone: [10.0, 90.0], risk: false, ago: None, right: Right::Rec { bars: vec![(Some(true), 16.0); 40], pct: None } };
        let bars: Vec<(f64, f64)> = scene(&c, 404.0, &m).iter().filter_map(|p| if let Prim::Rect { x, w, c: Col::Good, .. } = p { Some((*x, *w)) } else { None }).collect();
        assert!(bars.iter().all(|b| b.0 >= 232.0 - 1e-9) && bars.len() < 40 && bars[0].0 < 232.0 + 4.0, "{bars:?}");
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
