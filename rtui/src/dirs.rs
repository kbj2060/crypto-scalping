//! ⑤ 예측 줄 «15분 예측»·«1시간 예측» 두 줄(app.js fcRowHtml 안 sCell 6054 · prHit 5928 · dirShortNow 5714) -- 칸 그리기는 srl.rs(같은 cell).
//! 자료 = 같은 출처 /api/dir-short?asset=(1분 · ETH·BTC·SOL·XRP 만 · 5715 refreshDirShort) + /api/pred-record(1분 · 5917) · 실패·못 읽음 = 받아 둔 값 그대로.
//! 안 옮긴 것: 칸 툴팁(title -- 점수·문턱·분위·적중 설명) · 사이의 «레벨 5분»(ETH -- ?card=srl) · ⑤ 줄 나머지 칸.
use crate::srl::{Cell, Right, js, truthy};
use crate::fp::Col;
use serde_json::Value;

/// 5708 DIRSHORT_ASSETS
pub const ASSETS: [&str; 4] = ["eth", "btc", "sol", "xrp"];

/// JS toFixed(1)(정확한 동점은 큰 쪽 -- Rust 는 짝수 쪽) · 0~100 자르기 · null = 0
fn x1(v: &Value) -> f64 {
    let x = v.as_f64().unwrap_or(0.0).clamp(0.0, 100.0);
    let e = format!("{x:.80}"); // 정확한 십진 전개 -- 동점 = 소수 둘째 자리 5 뒤가 전부 0
    let f = &e[e.find('.').unwrap() + 1..];
    if f[1..].starts_with('5') && f[2..].bytes().all(|b| b == b'0') { (x * 10.0).ceil() / 10.0 } else { format!("{x:.1}").parse().unwrap() }
}

/// 5928 prHit(nm) 의 보이는 몫(pr = /api/pred-record 응답)
fn hit(pr: Option<&Value>, asset: &str, nm: &str) -> Right {
    let Some(pr) = pr.filter(|p| p["ok"] == true) else { return Right::None };
    let r = &pr["coins"][asset][nm];
    if r.is_null() || !(truthy(&r["n"]) || !r["hit"].is_null()) { return Right::Note("쌓는 중".into()); }
    let f = |v: &Value| v.as_f64().unwrap_or(f64::NAN);
    let recent = r["recent"].as_array().filter(|a| !a.is_empty());
    let bars = match recent {
        Some(a) => a.iter().map(|b| { let u = truthy(&b[1]); (Some(u), if u { 16.0 } else { 6.0 }) }).collect(),
        None => r["daily"].as_array().into_iter().flatten().map(|d| {
            let (n, h) = (f(&d[1]), f(&d[2]));
            let px = (16.0 * h / n + 0.5).floor().max(3.0); // NaN = height:NaNpx(무시) → 0
            (if h / n > 0.5 { Some(true) } else if h / n < 0.5 { Some(false) } else { None }, if px.is_nan() { 0.0 } else { px })
        }).collect(),
    };
    let pct = truthy(&r["n"]).then(|| {
        let q = f(&r["hit"]) / f(&r["n"]);
        (js(&Value::from((100.0 * q + 0.5).floor())), if q >= 0.55 { Some(Col::Good) } else if q <= 0.45 { Some(Col::Bad) } else { None }, format!("{}/{}", js(&r["hit"]), js(&r["n"])))
    });
    Right::Rec { bars, pct }
}

/// 6054 sCell -- ds = /api/dir-short 응답 · pr = /api/pred-record 응답 · nm = "15m" | "1h" · now = Date.now()
pub fn cell(ds: Option<&Value>, pr: Option<&Value>, asset: &str, now: f64, nm: &str) -> Cell {
    let (ko, back, col) = if nm == "15m" { ("15분", 3, 1) } else { ("1시간", 12, 2) };
    let gs = ds.filter(|d| d["ok"] == true && d["asset"].as_str().unwrap_or("eth") == asset);
    let old = gs.is_some_and(|g| now - g["decided_at_ms"].as_f64().unwrap_or(f64::NAN) > 15.0 * 60e3);
    let p = gs.filter(|_| !old).map(|g| &g["pred"][nm]).filter(|p| p.is_object());
    let hist = gs.and_then(|g| g["hist"].as_array()).map(Vec::as_slice).unwrap_or(&[]);
    let ago = hist.len().checked_sub(1 + back).map(|i| x1(&hist[i][col]));
    let s = p.map_or(0.0, |p| p["signal"].as_f64().unwrap_or(0.0));
    let (c, main) = match p { None => (Col::Muted, "-"), Some(_) if s > 0.0 => (Col::Good, "▲ 상승"), Some(_) if s < 0.0 => (Col::Bad, "▼ 하락"), Some(_) => (Col::Muted, "보류") };
    Cell { hz: format!("{ko} 예측"), c, main: main.into(), gauge: p.map(|p| x1(&p["pct"])), zone: 20.0, ago: p.and(ago), right: hit(pr, asset, nm) }
}

#[cfg(test)]
mod t {
    use super::*;

    /// 지금 화면의 실제 함수로 만든 정답지(compare/dirs_js.mjs → fixture/dirs_js.json · 같은 입력 dirs_in.json)
    #[test]
    fn js_parity() {
        let fx = |f: &str| serde_json::from_str::<Value>(&std::fs::read_to_string(format!("{}/compare/fixture/{f}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap();
        let (inp, out) = (fx("dirs_in.json"), fx("dirs_js.json"));
        fn nn(v: &Value) -> Option<&Value> { Some(v).filter(|v| !v.is_null()) }
        let mut n = 0;
        for (k, (i, jj)) in inp.as_array().unwrap().iter().zip(out.as_array().unwrap()).enumerate() {
            for (nm, j) in ["15m", "1h"].iter().zip(jj.as_array().unwrap()) {
                let c = cell(nn(&i[0]), nn(&i[1]), i[2].as_str().unwrap(), i[3].as_f64().unwrap(), nm);
                let cls = |c: Col| match c { Col::Good => "up", Col::Bad => "dn", _ => "flat" };
                let num = |v: &Value| v.as_str().map(|s| s.parse::<f64>().unwrap());
                assert_eq!((c.hz.as_str(), cls(c.c), c.main.as_str(), c.gauge, c.ago), (j["hz"].as_str().unwrap(), j["c"].as_str().unwrap(), j["main"].as_str().unwrap(), num(&j["gauge"]), num(&j["ago"])), "입력 {k} {nm}");
                // 1e-6: 점 자리 글자(toFixed(1))를 숫자로 -- 같은 글자면 같은 수
                if let (Some(a), Some(b)) = (c.gauge, num(&j["gauge"])) { assert!((a - b).abs() < 1e-6 && format!("{a:.1}") == j["gauge"].as_str().unwrap(), "입력 {k} {nm}"); }
                if c.gauge.is_some() { assert_eq!(j["zones"], serde_json::json!([["dn", "width", 20], ["up", "left", 80]]), "입력 {k} {nm}"); }
                let r = &j["right"];
                match &c.right {
                    Right::None => assert!(r.is_null(), "입력 {k} {nm}"),
                    Right::Note(t) => assert_eq!((r["bars"].is_null(), r["note"].as_str()), (true, Some(t.as_str())), "입력 {k} {nm}"),
                    Right::Rec { bars, pct } => {
                        let jb: Vec<(String, String)> = serde_json::from_value(r["bars"].clone()).unwrap();
                        let mine: Vec<(String, String)> = bars.iter().map(|&(u, h)| ((match u { Some(true) => "u", Some(false) => "d", None => "" }).into(), js(&Value::from(h)))).collect();
                        assert_eq!(mine, jb, "입력 {k} {nm}");
                        match pct {
                            Some((p, pc, f)) => {
                                let pcls = match pc { Some(Col::Good) => "up", Some(Col::Bad) => "dn", _ => "" };
                                assert_eq!((format!("{p}%"), pcls, f.as_str()), (r["pct"].as_str().unwrap().to_string(), r["pcls"].as_str().unwrap(), r["frac"].as_str().unwrap()), "입력 {k} {nm}");
                            }
                            None => assert_eq!((r["pct"].is_null(), r["note"].as_str()), (true, Some("쌓는 중")), "입력 {k} {nm}"),
                        }
                    }
                }
                n += 1;
            }
        }
        assert!(n >= 22, "{n}");
    }

    /// toFixed(1) 동점·자르기
    #[test]
    fn fixed1() {
        for (v, e) in [(12.25, 12.3), (87.75, 87.8), (0.05, 0.1), (99.95, 100.0), (104.0, 100.0), (-3.0, 0.0), (64.44, 64.4), (12.35, 12.3), (0.25, 0.3), (0.35, 0.3)] { assert_eq!(x1(&Value::from(v)), e, "{v}"); }
        assert_eq!(x1(&Value::Null), 0.0);
    }

    /// ETH·BTC·SOL·XRP 만 1분마다 dir-short · pred-record 는 코인과 상관없이 1분 · 숨으면 안 묻는다 · 실패·못 읽음 = 받아 둔 값 그대로
    #[test]
    fn poll() {
        use crate::feed::{Feed, Kind, Req, Source, Urls};
        let mut f = Feed::new("sol", "SOLUSDT", Source::Poll, Urls::default(), false);
        f.dirs_only = true;
        for (t, hid) in [(60_000.0, false), (61_000.0, false), (125_000.0, true), (126_000.0, false), (150_000.0, false)] { f.tick(t, hid, false); }
        let got: Vec<_> = f.out.iter().map(|r| match r { Req::Get { kind, url } => (*kind, url.as_str()), _ => panic!("다른 것을 묻지 않는다") }).collect();
        assert_eq!(got, vec![(Kind::DirShort, "/api/dir-short?asset=sol"), (Kind::PredRec, "/api/pred-record"), (Kind::DirShort, "/api/dir-short?asset=sol"), (Kind::PredRec, "/api/pred-record")]);
        f.on_dirs(Some(r#"{"ok":true,"asset":"sol","decided_at_ms":1000,"pred":{"15m":{"pct":90,"signal":1}},"hist":[]}"#));
        f.on_pred(Some(r#"{"ok":true,"coins":{"sol":{"15m":{"n":2,"hit":1,"recent":[[0,1],[0,0]]}}}}"#));
        f.on_dirs(None);
        f.on_dirs(Some("<html>"));
        f.on_pred(None);
        let c = cell(f.dirs.as_ref(), f.pred.as_ref(), "sol", 2000.0, "15m");
        assert_eq!((c.main.as_str(), c.gauge), ("▲ 상승", Some(90.0)));
        assert!(matches!(c.right, Right::Rec { ref bars, pct: Some(_) } if bars.len() == 2));
        let mut h = Feed::new("hype", "HYPEUSDT", Source::Poll, Urls::default(), false);
        h.dirs_only = true;
        h.tick(60_000.0, false, false);
        assert!(matches!(&h.out[..], [Req::Get { kind: Kind::PredRec, .. }]), "dir-short 는 네 코인만");
    }
}
