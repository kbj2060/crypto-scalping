//! 시장 맥락 ①② 게이지 묶음(지금 화면 오른쪽 칸 1초 수급 아래 맨 위 = #mcBody.mc-cmp 의 .mc-strip-wall · app.js renderMarketCtx 6420 의 넓은 화면 분기 6571~6605 ·
//! 그림 = mcGfx.pct·sig·seg4 6185~6216). 자료 = 같은 출처 /api/market-context?asset= (1초 · app.js refreshMarketCtx 6259 · rtserve ctx_mc.rs 가 응답).
//! 자리 = styles.css(이 작업 시점): 3252~3266 .mc-strip-wall(높이 128 · 위 2 · 아래 8 · 아래 줄 1 · 두 칸 사이 12 · 네 줄이 높이를 똑같이) ·
//!   3264~3266 한 줄 = [이름 44][6][게이지 gv][6][값 오른쪽 맞춤] · 4분면(seg4 2×2) = 줄 상자에 viewBox 를 맞춰 줄인다(meet).
//!   ①② 는 bare 타일(6582 · 상태 칩·«?» 없음 -- 6587·6594 의 wallTile).
//! 안 옮긴 것: 줄 툴팁(title) · ⑤ 예측 줄(fcRowHtml -- 다른 응답) · 휴대폰·좁은 화면 판(① 상태 칩 · ④ 벽) · 머리 발표 띠(#fpRates).
//! ponytail: 글자 굵기 500(이름·4분면 꺼진 칸) = 400 글꼴(Pretendard 500 을 안 묶었다 -- fonts/subset.sh 에 Medium 을 더하면 된다).
use crate::data::to_fixed;
use crate::fp::{Anchor, Clip, Col, Prim};
use serde_json::Value;

/// 6433 n(v, dp) = toLocaleString("en-US", 소수 dp 자리 고정) -- 0 이상만(sg 가 부호를 뗀다) · 반올림은 toFixed 와 같은 «반은 0 에서 먼 쪽»
pub fn n(v: f64, dp: usize) -> String {
    let s = to_fixed(v, dp);
    let (i, f) = s.split_once('.').map_or((s.as_str(), None), |(a, b)| (a, Some(b)));
    let mut g = String::new();
    for (k, c) in i.chars().enumerate() { if k > 0 && (i.len() - k) % 3 == 0 { g.push(','); } g.push(c); }
    f.map_or(g.clone(), |f| format!("{g}.{f}"))
}
/// 6434 sg -- 없으면 "-" · 부호 = "+"/"−"(U+2212)
fn sg(v: Option<f64>, dp: usize, u: &str) -> String { v.map_or("-".into(), |v| format!("{}{}{u}", if v >= 0.0 { "+" } else { "−" }, n(v.abs(), dp))) }
/// 6437 tone -- None = 색 없음(글자 기본색)
fn tone(v: Option<f64>, thr: f64) -> Option<Col> { v.filter(|v| v.abs() >= thr).map(|v| if v > 0.0 { Col::Good } else { Col::Bad }) }
/// JS Math.round(x) = floor(x + .5)
fn round(x: f64) -> f64 { (x + 0.5).floor() }
fn r1(x: f64) -> f64 { to_fixed(x, 1).parse().unwrap_or(x) }

fn rect(x: f64, y: f64, w: f64, h: f64, c: Col, a: f64, rx: f64) -> Prim { Prim::Rect { x, y, w, h, c, a, rx, clip: Clip::No } }
fn vline(x: f64, y0: f64, y1: f64, c: Col, dash: Option<f64>) -> Prim { Prim::Line { pts: vec![[x, y0], [x, y1]], c, a: 1.0, lw: 1.0, dash, clip: Clip::No } }

/// 6186 분위 0~1 · 양끝 10% 주황 띠 · 값 없으면 점 없음(svg 좌표 w×16)
pub fn pct(p: Option<f64>, w: f64) -> Vec<Prim> {
    let x = 4.0 + (w - 8.0) * p.unwrap_or(0.0).clamp(0.0, 1.0);
    let mut v = vec![rect(4.0, 6.0, w - 8.0, 4.0, Col::White, 0.12, 2.0), rect(4.0, 6.0, (w - 8.0) * 0.1, 4.0, Col::Amber, 0.4, 2.0),
                     rect(4.0 + (w - 8.0) * 0.9, 6.0, (w - 8.0) * 0.1, 4.0, Col::Amber, 0.4, 2.0), vline(w / 2.0, 3.0, 13.0, Col::Line, None)];
    if p.is_some() { v.push(Prim::Circle { x: r1(x), y: 8.0, r: 5.0, c: Col::Text, a: 1.0, clip: Clip::No }); }
    v
}
/// 6192 σ −3~+3 · 0 가운데 · ±2σ 점선
pub fn sig(zv: Option<f64>, w: f64) -> Vec<Prim> {
    let (c, s) = (w / 2.0, (w / 2.0 - 4.0) / 3.0);
    let v = zv.unwrap_or(0.0).clamp(-3.0, 3.0);
    let x0 = if v >= 0.0 { c } else { c + v * s };
    let fill = match zv { Some(z) if z.abs() >= 0.5 => if z > 0.0 { Col::Good } else { Col::Bad }, _ => Col::Muted };
    let mut o = vec![rect(4.0, 7.0, w - 8.0, 2.0, Col::White, 0.1, 0.0), vline(c - 2.0 * s, 3.0, 13.0, Col::Line, Some(2.0)), vline(c + 2.0 * s, 3.0, 13.0, Col::Line, Some(2.0)),
                     vline(c, 2.0, 14.0, Col::Muted, None)];
    if zv.is_some() { o.push(rect(r1(x0), 4.0, r1((v.abs() * s).max(1.5)), 8.0, fill, 1.0, 2.0)); }
    o
}
/// 6206 seg4(two = true · 2×2): 신규 롱 · 숏커버 / 신규 숏 · 롱정리 -- 지금 칸만 불(w×32)
pub fn seg4(key: Option<&str>, w: f64) -> Vec<Prim> {
    let cw = (w - 2.0) / 2.0;
    let mut o = vec![];
    for (i, (k, t)) in [("up_up", "신규 롱"), ("up_dn", "숏커버"), ("dn_up", "신규 숏"), ("dn_dn", "롱정리")].into_iter().enumerate() {
        let (on, col) = (key == Some(k), if k.starts_with("up") { Col::Good } else { Col::Bad });
        let (y0, x) = ((i / 2) as f64 * 16.0, (i % 2) as f64 * (cw + 2.0));
        let dx = r1(x + 9.0);
        o.push(if on { rect(r1(x), 1.0 + y0, r1(cw), 14.0, col, 0.16, 3.0) } else { rect(r1(x), 1.0 + y0, r1(cw), 14.0, Col::White, 0.06, 3.0) });
        if on { o.push(Prim::StrokeRect { x: r1(x), y: 1.0 + y0, w: r1(cw), h: 14.0, c: col, a: 0.55, lw: 1.0, clip: Clip::No }); }
        o.push(Prim::Circle { x: dx, y: 8.0 + y0, r: 3.5, c: col, a: if on { 1.0 } else { 0.22 }, clip: Clip::No });
        if on { o.push(Prim::Circle { x: dx, y: 8.0 + y0, r: 6.0, c: col, a: 0.25, clip: Clip::No }); }
        o.push(Prim::Text { x: r1(x + 17.0), y: 11.5 + y0, s: t.into(), size: 11.0, wt: if on { 700 } else { 400 }, c: if on { Col::Text } else { Col::Muted }, a: if on { 1.0 } else { 0.7 },
                            anchor: Anchor::Start, halo: false, clip: Clip::No });
    }
    o
}

/// 줄 하나: 이름 · 값(None = 4분면 -- 값 칸 없이 타일 폭) · 값 색 · 게이지(svg 좌표) · 게이지 폭·높이
pub struct Row { pub name: &'static str, pub val: Option<String>, pub tone: Option<Col>, pub g: Vec<Prim>, pub gw: f64, pub gh: f64 }

fn f(v: &Value) -> Option<f64> { v.as_f64() }
/// `a ?? b`
fn or(a: &Value, b: &Value) -> Option<f64> { f(a).or_else(|| f(b)) }

/// 6571~6605 넓은 화면 ①②(bare) -- Err = 6427 «불러오는 중» / 지연 글. bw = #mcBody 폭(clientWidth = G.cmpW)
pub fn tiles(d: Option<&Value>, asset: &str, bw: f64) -> Result<[Vec<Row>; 2], String> {
    let d = d.filter(|d| d["asset"].as_str() == Some(asset)); // 다른 코인 응답은 «불러오는 중»
    let Some(d) = d.filter(|d| d["available"].as_bool() == Some(true)) else {
        return Err(match d.and_then(|d| d.get("error")).filter(|e| !e.is_null()) {
            Some(e) => format!("시장 맥락 지연 ({})", e.as_str().map_or(e.to_string(), str::to_string)),
            None => "불러오는 중…".into(),
        });
    };
    let (fu, b, oi, fl) = (&d["funding"], &d["basis"], &d["oi"], &d["flow"]);
    let z = [&fl["z30h_all"], &fl["z30h"]].into_iter().find(|v| v.is_object()).unwrap_or(&Value::Null); // 6443 fl.z30h_all || fl.z30h || {}
    let oia = (oi["all"].is_object() && !oi["all"]["z30m"].is_null()).then(|| &oi["all"]); // 6470
    let oz = f(&oia.unwrap_or(oi)["z30m"]);
    let gv = ((bw - 12.0) / 2.0).floor() - 96.0; // 6603 gwWall
    let gv = gv.max(40.0);
    let at_base = fu["bn_at_base"].as_bool() == Some(true);
    let pp = round(f(&fu["bn_pct180"]).unwrap_or(0.0) * 100.0);
    let row = |name, val: String, tone, g| Row { name, val: Some(val), tone, g, gw: gv, gh: 16.0 };
    let cvd = or(&fl["cvd30_all_z"], &fl["cvd30_z"]);
    let lev = vec![
        row("펀딩", if at_base { "기본".into() } else { format!("{pp}%") }, None, pct(if at_base { None } else { f(&fu["bn_pct180"]) }, gv)),
        row("베이시스", sg(f(&b["bp"]), 1, "bp"), None, pct(f(&b["pct7d"]), gv)),
        row("델타", sg(cvd, 1, "σ"), tone(cvd, 0.5), sig(cvd, gv)),
        row("OI", sg(oz, 1, "σ"), tone(oz, 1.5), sig(oz, gv)),
    ];
    let mut flow: Vec<Row> = [("고래", "whale"), ("중형", "mid"), ("리테일", "retail")].into_iter()
        .map(|(nm, k)| { let v = f(&z[k]); row(nm, sg(v, 1, "σ"), tone(v, 0.5), sig(v, gv)) }).collect();
    flow.push(Row { name: "", val: None, tone: None, g: seg4(d["quad"]["key"].as_str(), gv + 96.0), gw: gv + 96.0, gh: 32.0 });
    Ok([lev, flow])
}

/// 3263 .mc-strip-wall 높이
pub const H: f64 = 128.0;

/// 한 장(w = #mcBody 폭 · 높이 H) -- 줄 자리는 머리 주석의 CSS 계산
pub fn scene(d: Option<&Value>, asset: &str, w: f64) -> Vec<Prim> {
    let t = |x, y, s: String, size, wt, c, anchor| Prim::Text { x, y, s, size, wt, c, a: 1.0, anchor, halo: false, clip: Clip::No };
    let tiles = match tiles(d, asset, w) {
        Ok(t) => t,
        Err(s) => return vec![t(0.0, 16.0, s, 12.0, 400, Col::Muted, Anchor::Start)], // .mc-note
    };
    let (top, inner) = (2.0, H - 2.0 - 8.0 - 1.0); // border-box: 위 2 · 아래 8 · 아래 줄 1
    let col_w = (w - 12.0) / 2.0;
    let rh = inner / 4.0;
    let mut v = vec![rect(0.0, H - 1.0, w, 1.0, Col::SoftLine, 1.0, 0.0)];
    for (k, rows) in tiles.into_iter().enumerate() {
        let x0 = k as f64 * (col_w + 12.0);
        for (i, r) in rows.into_iter().enumerate() {
            let y0 = top + i as f64 * rh;
            let cy = y0 + rh / 2.0;
            // 4분면 = 줄 상자(col_w × rh)에 viewBox meet · 나머지 = 게이지(16 높이)를 가운데에
            let (s, gx, gy) = if r.val.is_none() { let s = (col_w / r.gw).min(rh / r.gh); (s, x0 + (col_w - r.gw * s) / 2.0, y0 + (rh - r.gh * s) / 2.0) } else { (1.0, x0 + 50.0, cy - 8.0) };
            v.extend(r.g.into_iter().map(|p| place(p, gx, gy, s)));
            let Some(val) = r.val else { continue };
            // 글자 바닥선 ≈ 가운데 + 0.36 em(line-height 1 상자의 가운데 정렬)
            v.push(t(x0, cy + 11.0 * 0.36, r.name.into(), 11.0, 400, Col::Muted, Anchor::Start));
            v.push(t(x0 + col_w, cy + 11.5 * 0.36, crate::fp::tnum(&val), 11.5, 700, r.tone.unwrap_or(Col::Text), Anchor::End));
        }
    }
    v
}

/// svg 좌표 → 카드 좌표(옮기고 s 배)
fn place(p: Prim, ox: f64, oy: f64, s: f64) -> Prim {
    let q = |x: f64, y: f64| [ox + x * s, oy + y * s];
    match p {
        Prim::Rect { x, y, w, h, c, a, rx, clip } => { let [x, y] = q(x, y); Prim::Rect { x, y, w: w * s, h: h * s, c, a, rx: rx * s, clip } }
        Prim::StrokeRect { x, y, w, h, c, a, lw, clip } => { let [x, y] = q(x, y); Prim::StrokeRect { x, y, w: w * s, h: h * s, c, a, lw, clip } }
        Prim::Line { pts, c, a, lw, dash, clip } => Prim::Line { pts: pts.iter().map(|p| q(p[0], p[1])).collect(), c, a, lw, dash: dash.map(|d| d * s), clip },
        Prim::Dashed { pts, c, a, lw, on, off, clip } => Prim::Dashed { pts: pts.iter().map(|p| q(p[0], p[1])).collect(), c, a, lw, on: on * s, off: off * s, clip },
        Prim::Circle { x, y, r, c, a, clip } => { let [x, y] = q(x, y); Prim::Circle { x, y, r: r * s, c, a, clip } }
        Prim::Text { x, y, s: st, size, wt, c, a, anchor, halo, clip } => { let [x, y] = q(x, y); Prim::Text { x, y, s: st, size: size * s, wt, c, a, anchor, halo, clip } }
        p @ (Prim::Area { .. } | Prim::Mono { .. }) => p, // 시장 맥락은 안 쓴다
    }
}

#[cfg(test)]
mod t {
    use super::*;

    /// svg 칠 글자 → (색, 불투명) -- mcGfx 가 쓰는 것만
    fn paint(fill: &str, op: Option<&str>) -> (Col, f64) {
        let c = match fill {
            "rgb(var(--lift) / .12)" => return (Col::White, 0.12), "rgb(var(--lift) / .1)" => return (Col::White, 0.1), "rgb(var(--lift) / .06)" => return (Col::White, 0.06),
            "var(--warn)" => Col::Amber, "var(--line)" => Col::Line, "var(--text)" => Col::Text, "var(--muted)" => Col::Muted, "var(--good)" => Col::Good, "var(--bad)" => Col::Bad,
            x => panic!("모르는 칠 {x}"),
        };
        (c, op.map_or(1.0, |o| o.parse().unwrap()))
    }
    /// 그리기 하나 → 비교 꼴 [종류, 숫자..] + 색
    fn key(p: &Prim) -> (String, Vec<f64>, Col, f64) {
        match p {
            Prim::Rect { x, y, w, h, c, a, rx, .. } => ("rect".into(), vec![*x, *y, *w, *h, *rx], *c, *a),
            Prim::StrokeRect { x, y, w, h, c, a, .. } => ("stroke".into(), vec![*x, *y, *w, *h], *c, *a),
            Prim::Line { pts, c, a, dash, .. } => ("line".into(), vec![pts[0][0], pts[0][1], pts[1][0], pts[1][1], dash.unwrap_or(0.0)], *c, *a),
            Prim::Circle { x, y, r, c, a, .. } => ("circle".into(), vec![*x, *y, *r], *c, *a),
            Prim::Text { x, y, s, size, wt, c, a, .. } => (format!("text {s} {wt}"), vec![*x, *y, *size], *c, *a),
            Prim::Dashed { .. } | Prim::Area { .. } | Prim::Mono { .. } => unreachable!(),
        }
    }
    fn js_key(s: &Value) -> Vec<(String, Vec<f64>, Col, f64)> {
        let a = &s["a"];
        let g = |k: &str| a[k].as_str().map(|x| x.parse::<f64>().unwrap()).unwrap_or(0.0);
        let st = |k: &str| a[k].as_str();
        match s["tag"].as_str().unwrap() {
            "rect" => {
                let (c, al) = paint(st("fill").unwrap(), st("opacity").or(st("fill-opacity")));
                let mut v = vec![("rect".into(), vec![g("x"), g("y"), g("width"), g("height"), g("rx")], c, al)];
                if let Some(sc) = st("stroke") { let (c, al) = paint(sc, st("stroke-opacity")); v.push(("stroke".into(), vec![g("x"), g("y"), g("width"), g("height")], c, al)); }
                v
            }
            "line" => { let (c, al) = paint(st("stroke").unwrap(), None); vec![("line".into(), vec![g("x1"), g("y1"), g("x2"), g("y2"), st("stroke-dasharray").map_or(0.0, |_| 2.0)], c, al)] }
            "circle" => { let (c, al) = paint(st("fill").unwrap(), st("fill-opacity")); vec![("circle".into(), vec![g("cx"), g("cy"), g("r")], c, al)] }
            "text" => {
                let (c, al) = paint(st("fill").unwrap(), st("fill-opacity"));
                let wt = if g("font-weight") >= 600.0 { 700 } else { 400 }; // 500 = 400 글꼴(머리 주석)
                vec![(format!("text {} {wt}", s["text"].as_str().unwrap()), vec![g("x"), g("y"), g("font-size")], c, al)]
            }
            x => panic!("{x}"),
        }
    }

    /// 지금 화면의 실제 함수로 만든 정답지(compare/mc_js.mjs → fixture/mc_js.json · 같은 입력 mc_in.json)
    #[test]
    fn js_parity() {
        let fx = |f: &str| serde_json::from_str::<Value>(&std::fs::read_to_string(format!("{}/compare/fixture/{f}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap();
        let (inp, js) = (fx("mc_in.json"), fx("mc_js.json"));
        let mut shapes = 0;
        for w in [404.0, 300.0] {
            for (d, j) in inp.as_array().unwrap().iter().zip(js[format!("w{w}")].as_array().unwrap()) {
                let ts = tiles(Some(d), "eth", w).unwrap();
                for (rows, jt) in ts.iter().zip(j["tiles"].as_array().unwrap()) {
                    let jr = jt["rows"].as_array().unwrap();
                    assert_eq!(rows.len(), jr.len());
                    for (r, q) in rows.iter().zip(jr) {
                        assert_eq!(r.name, q["name"].as_str().unwrap());
                        assert_eq!(r.val.as_deref(), q["val"].as_str(), "값 {} w={w}", r.name);
                        let cls = match r.tone { Some(Col::Good) => "mc-good", Some(Col::Bad) => "mc-bad", _ => "" };
                        if r.val.is_some() { assert_eq!(cls, q["cls"].as_str().unwrap(), "색 {}", r.name); }
                        assert_eq!((r.gw, r.gh), (q["w"].as_f64().unwrap(), q["h"].as_f64().unwrap()), "게이지 크기 {}", r.name);
                        let jk: Vec<_> = q["shapes"].as_array().unwrap().iter().flat_map(js_key).collect();
                        let rk: Vec<_> = r.g.iter().map(key).collect();
                        assert_eq!(rk.len(), jk.len(), "도형 수 {} w={w}", r.name);
                        for (a, b) in rk.iter().zip(&jk) {
                            assert!(a.0 == b.0 && a.2 == b.2 && (a.3 - b.3).abs() < 1e-9 && a.1.iter().zip(&b.1).all(|(x, y)| (x - y).abs() < 1e-9), "{} w={w}: {a:?} 대 {b:?}", r.name);
                        }
                        shapes += rk.len();
                    }
                }
            }
        }
        assert!(shapes > 200, "{shapes}");
        let wait: Vec<String> = js["wait"].as_array().unwrap().iter().map(|x| x["note"].as_str().unwrap().into()).collect();
        assert_eq!(tiles(None, "eth", 404.0).err().unwrap(), wait[0]);
        assert_eq!(tiles(Some(&serde_json::json!({"asset": "eth", "available": false, "error": "HTTP 502"})), "eth", 404.0).err().unwrap(), wait[1]);
    }

    #[test]
    fn layout_and_fmt() {
        assert_eq!((n(1234.5, 1), n(0.25, 1), n(7.0, 0), sg(Some(-0.04), 1, "bp")), ("1,234.5".into(), "0.3".into(), "7".into(), "−0.0bp".into()));
        let d = serde_json::json!({"asset": "eth", "available": true, "quad": {"key": "up_up"}, "flow": {"z30h": {"whale": 1.0}}});
        let v = scene(Some(&d), "eth", 404.0);
        // 줄 높이 = (128 − 11) / 4 = 29.25 · 둘째 칸 왼쪽 = (404 − 12)/2 + 12 = 208 · 고래 값 오른쪽 끝 = 404
        let tx: Vec<(f64, f64, String)> = v.iter().filter_map(|p| if let Prim::Text { x, y, s, .. } = p { Some((*x, *y, s.clone())) } else { None }).collect();
        assert!(tx.contains(&(208.0, 2.0 + 29.25 / 2.0 + 3.96, "고래".into())), "{tx:?}");
        assert!(tx.iter().any(|t| t.0 == 404.0 && t.2 == crate::fp::tnum("+1.0σ")));
        // 4분면: s = min(196/196, 29.25/32) → 높이가 정한다 · 켜진 칸 «신규 롱» 글자 굵게
        assert!(v.iter().any(|p| matches!(p, Prim::Text { s, wt: 700, size, .. } if s == "신규 롱" && (*size - 11.0 * 29.25 / 32.0).abs() < 1e-9)));
        assert_eq!(scene(Some(&serde_json::json!({"asset": "sol", "available": true})), "eth", 404.0).len(), 1, "다른 코인 응답 = 불러오는 중");
    }

    /// 1초마다 한 번 · 숨으면 안 묻는다 · 실패 = 불러오는 중
    #[test]
    fn poll() {
        use crate::feed::{Feed, Kind, Req, Source, Urls};
        let mut f = Feed::new("eth", "ETHUSDT", Source::Poll, Urls::default(), false);
        f.mc_only = true;
        for (t, hid) in [(1000.0, false), (1500.0, false), (2100.0, true), (2200.0, false), (2700.0, false)] { f.tick(t, hid, false); }
        let urls: Vec<_> = f.out.iter().map(|r| match r { Req::Get { kind: Kind::Mc, url } => url.clone(), _ => panic!("허브·풋프린트를 묻지 않는다") }).collect();
        assert_eq!(urls, vec!["/api/market-context?asset=eth"; 2]);
        f.on_mc(None);
        // 실패 응답엔 asset 이 없다 → 6426 «다른 코인 응답»과 같이 «불러오는 중…»(지금 화면도 같다 -- 지연 글은 서버가 asset 을 실은 실패만)
        assert!(matches!(&scene(f.mc.as_ref(), "eth", 404.0)[0], Prim::Text { s, .. } if s == "불러오는 중…"));
    }
}
