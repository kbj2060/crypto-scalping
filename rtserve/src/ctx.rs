//! 상황 카드(2026-11-22 옮김 첫 조각) -- /api/situation 본문 · SSE /api/stream 의 `event: situation` 본문.
//! 원본(Read 로 본 실제 줄): server.py `situation_payload` 2734~2738 · `api_situation` 2740~2741 · `api_stream` 3149~3183(상황 = computed_at 이 바뀔 때 ·
//! `json.dumps(…, ensure_ascii=False, separators=(',', ':'))`) · `compute_situation` 2421~2498(now = classify 결과 + computed_at · read = fr.read · vol_mult)
//! · dashboard/flow_read.py 전부(`read` 250~427 · `fuse` 430~460 · `outcome3` 463~507 · `gauges` 197~247 · `vol_size_mult` 123~136 · 상수·문구 15~168).
//! 입력: 코어 `Situation`(rtcore situation.rs -- classify 그대로 · veto·obi·persist 는 «통과 칸»이라 받는 쪽이 채워 넣는다: `patch_now`) ·
//! 관계 읽기 입력 `x`(server.py `_flow_read_ctx` 가 만드는 dict -- 다음 조각) · 5분봉 하루(REST · vol_size_mult) · 수신 상태(fo·mp).
//! 일치 정답지: rtserve/tools/gen_ctx_fixture.py(원본 flow_read 를 import 해서 부름) → tests/ctx_parity.rs.
use serde_json::{json, Map, Value as V};

pub const Z_SIDE: f64 = 0.5;
pub const CVD_Z_BIG: f64 = 1.0;
pub const FLAT_FRAC: f64 = 0.25;
pub const DEEP_HI: f64 = 0.8;
pub const DEEP_LO: f64 = 0.2;
pub const NEAR_BP: f64 = 15.0;
pub const LIQ_MIN_USD: f64 = 50_000.0;
pub const FUSE_MIN_VOTES: i64 = 2;
pub const GATE_PCT: f64 = 2.0 / 3.0;
pub const SYM_K: f64 = 0.5;
pub const DIR_COIN_PP: f64 = 5.0;
pub const FUSE_DIR_P: f64 = 0.567;
/// (레짐, a, g) → (n, 위 또는 이동 %, 아래 또는 반대 %, 미도달 %) -- flow_read.CARD_CELLS(36~58).
const CARD_CELLS: &[(&str, i64, i64, i64, f64, f64, f64)] = &[
    ("trend", -1, 0, 5654, 43.3, 46.1, 10.6),
    ("trend", -1, 1, 7150, 37.1, 38.2, 24.6),
    ("trend", -1, 2, 9788, 25.8, 26.7, 47.5),
    ("trend", 0, 0, 32646, 44.2, 45.3, 10.4),
    ("trend", 0, 1, 32024, 37.2, 35.9, 26.9),
    ("trend", 0, 2, 33742, 27.9, 23.9, 48.2),
    ("trend", 1, 0, 26093, 45.2, 45.9, 8.9),
    ("trend", 1, 1, 29338, 39.4, 37.2, 23.4),
    ("trend", 1, 2, 35864, 30.1, 24.5, 45.4),
    ("trend", 2, 0, 1135, 49.0, 45.1, 5.9),
    ("trend", 2, 1, 3321, 42.5, 41.0, 16.5),
    ("trend", 2, 2, 11637, 33.4, 25.6, 41.0),
    ("range", -1, 0, 2354, 45.4, 47.0, 7.6),
    ("range", -1, 1, 3892, 41.5, 41.4, 17.2),
    ("range", -1, 2, 6178, 27.1, 32.9, 40.0),
    ("range", 0, 0, 60540, 45.8, 46.0, 8.3),
    ("range", 0, 1, 40873, 39.2, 38.6, 22.2),
    ("range", 0, 2, 24574, 28.1, 30.9, 41.0),
    ("range", 1, 0, 2650, 47.9, 44.1, 8.1),
    ("range", 1, 1, 4269, 41.1, 40.1, 18.8),
    ("range", 1, 2, 6695, 27.9, 31.1, 41.0),
];
const REACH_EVID: &str = "닿을 확률 = HGB(5분봉 26피쳐) · TEST 2025~ AUC .78(옛 표 .72) · 예측 구간별 실측 ±2pp";
const DIR_EVID: &str = "방향 = HGB(5분봉 26피쳐, 1분봉 선착 라벨) · TEST 2025~ AUC .525 · 50:50 에서 5pp 이상 벌어지는 건 결정의 ~7%, 그때 방향 적중 ~56% · 나머지는 동전 — 30분 방향은 대부분 예측이 안 된다 · 융합 발동 중엔 모델과 발동 실측(57%)의 평균";
const FUSE_EVID: &str = "ETH 3.7년(서버와 같은 24h 정의): 건당 TRAIN +4.1bp[+0.2,+8.2] · TEST +6.6bp[+2.1,+11.4] · 4/4년 · 롱숏 둘 다 + · 회전 귀무 p<0.001 · 적중 48%(자주 조금 지고 가끔 크게 번다, 중앙 −2bp) · BTC 재현 안 됨 · 하루 ~1.7번 · 메이커 진입이어야 남는다";
/// 방향 → [(표 이름 접두, 부호, 문구)] -- flow_read.PUSH(138~143).
const PUSH_UP: [(&str, i64, &str); 4] = [("고래", 1, "고래 매수 · 리테일/중형 매도"), ("가격↔OI", 1, "1시간 하락 + OI 감소(롱 이탈 끝물)"), ("추세정렬", 1, "12h 상승 추세 안 30분 상승"), ("거부 봉", 1, "하락 중 매수 거부 봉")];
const PUSH_DN: [(&str, i64, &str); 4] = [("고래", -1, "고래 매도 · 리테일/중형 매수"), ("가격↔OI", -1, "1시간 하락 + OI 증가(새 숏)"), ("추세정렬", -1, "12h 하락 추세 안 30분 하락"), ("거부 봉", -1, "상승 중 매도 거부 봉")];

fn evid(k: &str) -> &'static str {
    match k {
        "price_flow" => "3.7년 실측: 체결↔가격 역행은 봉의 1/4로 흔하고, 뒤따르는 가격 차이 +0.2bp — 방향 신호 아님",
        "whale_retail" => "3.7년 실측: 갈리면 60분 고래 쪽 +6bp[+3,+10]·4/4년·BTC +3.5 — 메이커로만 비용을 넘는다 · 전진 검정 중(1차 2027-11 · 최종 2029-01)",
        "whale_mid" => "3.7년 실측: 고래↔중형 갈리면 60분 고래 쪽 +7.6bp[+3,+12]",
        "same_side" => "3.7년 실측: 크기별 단독 추종은 셋 다 되돌림(리테일 최악) — 쏠림을 따라가지 말 것",
        "oi_dn_up" => "4.7년 1시간: 하락+OI↑ 다음 1시간 −2.3bp(0/5년 반등) — 숏 조기청산 금지",
        "oi_dn_dn" => "4.7년 1시간: 하락+OI↓ 다음 1시간 +1.2bp(OI↑ 대비 +3.5·5/5년) — 숏 거두기 순 +0.5bp로 얇다",
        "oi_up" => "상승 쪽 사분면은 1시간 틀에서 안 쟀다(30분 카드 감사에선 스퀴즈·신규유입 효과 0)",
        "oi_small" => "1초~5분 OI 는 체결의 4초 뒤 그림자 — 큰 1시간 이동에서만 뜻이 있다",
        "liq_result" => "청산은 가격을 따라온다(분→다음 분 −0.34, 반대 +0.01) · 버스트 뒤 5~60분 평평 — 꼭지 신호 아님",
        "liq_flush" => "69일: 추세 반대편 청산 뒤 추세 방향 60분 +13.7bp[−2.4,+28] — CI 0 포함",
        "deep_wall" => "5분 홀드아웃 통과: 깊은 매수벽−매도벽 +7~8bp(t 2.2) — 상승장 두 창·점유 40%라 «상태»에 가깝다",
        "qi" => "최우선 호가+호가 증감 동조는 1~15초만 앞선다(+0.4bp/15초) — 수수료를 못 넘는다",
        "wall_sr" => "벽은 지지·저항이 아니다(닿은 뒤 반등 0.509)",
        "sr_near" => "레벨 ±15bp 는 청산이 날 자리(청산 확률 0.52 vs 0.28) — 뚫림/버팀 방향은 없다",
        "sr_wall" => "깊은 벽이 레벨 쪽이면 버팀 후보(n 82/156, 1.5~2.5SE)",
        "basis" => "선물·현물 주도 구분의 예측력은 안 쟀다",
        "funding" => "펀딩 쏠림 반대 매매: 4.7년 극단 펀딩 0/10·30분 감사 부호 반대",
        "btc" => "BTC 동행/단독: 30분 감사 효과 0(±0.2pp)",
        "veto_align" => "30분 지속률 정렬 52.8% vs 역행 50.5%(+2.2pp·6/6년) — 작다",
        "act" => "활동은 움직임의 크기를 말한다(방향 아님) — 변동성 확대 OOS AUC 0.82",
        _ => unreachable!("{k}"),
    }
}

// ── 파이썬 값·서식 ──
fn n(v: &V, k: &str) -> Option<f64> {
    v.get(k).and_then(V::as_f64)
}
fn truthy(x: Option<f64>) -> bool {
    x.is_some_and(|v| v != 0.0)
}
/// flow_read._s: `0 if not x else (1 if x > 0 else -1)` -- NaN 은 참이라 −1.
fn s(x: Option<f64>) -> i64 {
    match x {
        None => 0,
        Some(v) if v == 0.0 => 0,
        Some(v) if v > 0.0 => 1,
        _ => -1,
    }
}
/// 파이썬 format(x, '.Nf') / '+.Nf' (정확한 이진 값의 짝수 반올림 -- Rust 와 같다).
fn fx(x: f64, nd: usize, plus: bool) -> String {
    if plus { format!("{x:+.nd$}") } else { format!("{x:.nd$}") }
}
/// 천 단위 쉼표(',' 옵션) -- 정수 부분에만.
fn commas(t: String) -> String {
    let (sign, rest) = match t.chars().next() {
        Some(c @ ('+' | '-')) => (c.to_string(), &t[1..]),
        _ => (String::new(), &t[..]),
    };
    let (int, frac) = rest.split_once('.').map_or((rest, None), |(a, b)| (a, Some(b)));
    let mut out = String::new();
    for (i, ch) in int.chars().enumerate() {
        if i > 0 && (int.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    format!("{sign}{out}{}", frac.map_or(String::new(), |f| format!(".{f}")))
}
fn fc(x: f64, nd: usize, plus: bool) -> String {
    commas(fx(x, nd, plus))
}
fn py_round(x: f64, nd: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let r: f64 = format!("{x:.nd$}").parse().unwrap();
    if r == 0.0 { 0.0f64.copysign(x) } else { r }
}
fn side(x: f64, buy: &'static str, sell: &'static str) -> &'static str {
    if x > 0.0 { buy } else { sell }
}
fn updn(x: f64) -> &'static str {
    if x > 0.0 { "상승" } else { "하락" }
}
/// flow_read._k: 게이지 칸 짧은 수(+36k · +2.1k · +420 · −1.2M).
fn k(v: f64) -> String {
    let a = v.abs();
    let sg = if v > 0.0 { "+" } else if v < 0.0 { "−" } else { "" };
    if a >= 1e6 {
        format!("{sg}{}M", fx(a / 1e6, 1, false))
    } else if a >= 1e4 {
        format!("{sg}{}k", fx(a / 1e3, 0, false))
    } else if a >= 1e3 {
        format!("{sg}{}k", fx(a / 1e3, 1, false))
    } else {
        format!("{sg}{}", fx(a, 0, false))
    }
}
fn cl(v: f64) -> f64 {
    // 파이썬 max(-100.0, min(100.0, v)): 범위 안이면 v 그대로
    if v > 100.0 { 100.0 } else if v < -100.0 { -100.0 } else { v }
}
fn tilt(a: f64, b: f64) -> Option<f64> {
    (a != 0.0 || b != 0.0).then(|| (a - b) / (a.abs() + b.abs()) * 100.0)
}
fn obj(pairs: Vec<(&str, V)>) -> V {
    let mut m = Map::new();
    for (k, v) in pairs {
        m.insert(k.to_string(), v);
    }
    V::Object(m)
}
fn gv(kind: &str, v: f64, txt: String, on: bool) -> V {
    obj(vec![("kind", json!(kind)), ("v", json!(v)), ("txt", json!(txt)), ("on", json!(on))])
}

/// flow_read.gauges(197~247): 관계마다 게이지 하나. 차례 = 원본 dict 차례.
fn gauges(ev: &V, x: &V) -> Vec<(String, V)> {
    let mut g: Vec<(String, V)> = vec![];
    let (mv, cvd, cz) = (n(ev, "move_bp"), n(ev, "cvd"), n(x, "cvd30_z"));
    if let Some(c) = cvd {
        let v = match cz {
            Some(z) => cl(z * 33.0),
            None => if c > 0.0 { 50.0 } else if c < 0.0 { -50.0 } else { 0.0 },
        };
        g.push(("가격↔체결".into(), gv("d", v, k(c), c != 0.0)));
    }
    let net = x.get("net60").filter(|v| v.as_object().is_some_and(|o| !o.is_empty()));
    let z = x.get("z60").filter(|v| v.as_object().is_some_and(|o| !o.is_empty()));
    if let Some(net) = net {
        let v = match z {
            Some(z) => Some(cl(n(z, "whale").unwrap() * 50.0)),
            None => tilt(n(net, "whale").unwrap(), n(net, "retail").unwrap()),
        };
        g.push(("고래".into(), gv("d", v.filter(|v| *v != 0.0).unwrap_or(0.0), k(n(net, "whale").unwrap()), v.is_some())));
    }
    let (m60, oi60, p75) = (n(x, "move60"), n(x, "oi60"), n(x, "move60_p75"));
    if let (Some(m60), Some(oi60)) = (m60, oi60) {
        let big = truthy(p75) && m60 < 0.0 && m60.abs() >= p75.unwrap() && oi60 != 0.0;
        let v = if big { (if oi60 > 0.0 { -1.0 } else { 1.0 }) * f64::min(100.0, 50.0 * m60.abs() / p75.unwrap()) } else { 0.0 };
        g.push(("가격↔OI".into(), gv("d", v, format!("{}bp", fx(m60, 0, true)), big)));
    }
    if ev.get("liq_long").is_some() {
        let (ll, ls) = (n(ev, "liq_long").unwrap_or(0.0), n(ev, "liq_short").unwrap_or(0.0));
        let v = tilt(ls, ll);
        g.push(("청산".into(), gv("d", v.filter(|v| *v != 0.0).unwrap_or(0.0), format!("${}", k(ls - ll).trim_start_matches('+')), ll + ls >= LIQ_MIN_USD)));
    }
    let (ip, imb, obi) = (n(x, "imb40_pct"), n(x, "imb40"), n(ev, "obi"));
    if let (Some(ip), Some(imb)) = (ip, imb) {
        g.push(("호가↔체결".into(), gv("d", cl((ip - 0.5) * 200.0), fx(imb, 2, true), ip >= DEEP_HI || ip <= DEEP_LO)));
    } else if let Some(obi) = obi {
        g.push(("호가↔체결".into(), gv("d", cl(obi * 100.0), fx(obi, 2, true), false)));
    }
    let sr = x.get("sr").cloned().unwrap_or(V::Null);
    let (sb, rb) = (n(&sr, "sup_bp"), n(&sr, "res_bp"));
    let near = sr.get("near").and_then(V::as_str).filter(|s| !s.is_empty()).unwrap_or("없음").to_string();
    if sb.is_some() || rb.is_some() {
        let hold = near != "없음" && sr.get("deep_wall").and_then(V::as_str) == Some("레벨쪽");
        let dist = [sb, rb].into_iter().flatten().fold(f64::INFINITY, f64::min);
        let v = if hold { if near == "지지근접" { 70.0 } else { -70.0 } } else { 0.0 };
        g.push(("지지/저항".into(), gv("d", v, format!("{} {}bp", if Some(dist) == sb { "지지" } else { "저항" }, fx(dist, 0, false)), hold)));
    }
    let (bd, thr) = (n(ev, "basis_d_bp"), n(ev, "basis_thr_bp"));
    if n(ev, "basis_bp").is_some() {
        let v = match (bd, thr) {
            (Some(bd), Some(t)) if t != 0.0 => cl(bd / t * 100.0),
            _ => 0.0,
        };
        g.push(("선물↔현물".into(), gv("d", v, bd.map_or("—".into(), |b| fx(b, 1, true)), truthy(n(ev, "lead")))));
    }
    if let Some(bm) = n(ev, "btc_move_bp") {
        let v = if truthy(mv) { cl(bm / mv.unwrap().abs() * 100.0) } else { 0.0 };
        g.push(("ETH↔BTC".into(), gv("d", v, format!("{}bp", fx(bm, 0, true)), !ev.get("btc_rel").is_none_or(V::is_null))));
    }
    let (veto, d30) = (n(ev, "veto").unwrap_or(0.0), n(ev, "dir").unwrap_or(0.0));
    if veto != 0.0 && d30 != 0.0 {
        g.push(("추세↔이동".into(), gv("d", 50.0 * veto + 50.0 * d30, format!("{}{}", if veto > 0.0 { "↑" } else { "↓" }, if d30 > 0.0 { "↑" } else { "↓" }), veto == d30)));
    }
    if let Some(vp) = n(x, "vol_pct") {
        g.push(("활동".into(), gv("m", 100.0 * vp, fx(100.0 * vp, 0, false), vp >= 0.8 || vp <= 0.2)));
    }
    g
}

/// flow_read.outcome3(463~507).
fn outcome3(ev: &V, gp: Option<f64>, votes: &[(String, i64)], score: i64, pdir: Option<f64>, preach: Option<f64>) -> V {
    let (Some(d), Some(gp)) = (ev.get("dir").and_then(V::as_i64), gp) else { return V::Null };
    let reg = if d != 0 { "trend" } else { "range" };
    let a = (-1).max((if d != 0 { 2 } else { 1 }).min(if d != 0 { score * d } else { score }));
    let g = if gp < 1.0 / 3.0 { 0 } else if gp < 2.0 / 3.0 { 1 } else { 2 };
    let &(_, _, _, cn, mut p1, mut p2, mut pn) = CARD_CELLS.iter().find(|c| c.0 == reg && c.1 == a && c.2 == g).unwrap();
    let up = d == 0 || d > 0;
    let (mut p_up, mut p_dn) = if up { (p1, p2) } else { (p2, p1) };
    let reach_model = preach.is_some_and(|p| 0.0 < p && p < 1.0);
    if reach_model {
        pn = py_round(100.0 * (1.0 - preach.unwrap()), 1);
        let t = f64::max(p1 + p2, 1e-9);
        (p1, p2) = (p1 / t * (100.0 - pn), p2 / t * (100.0 - pn));
        (p_up, p_dn) = if up { (py_round(p1, 1), py_round(p2, 1)) } else { (py_round(p2, 1), py_round(p1, 1)) };
    }
    let reach = py_round(100.0 - pn, 1);
    let sd = if score.abs() >= FUSE_MIN_VOTES && gp >= GATE_PCT { score.signum() } else { 0 };
    let mut pdir = pdir;
    let blend = sd != 0 && pdir.is_some_and(|p| 0.0 < p && p < 1.0);
    if blend {
        let pd = pdir.unwrap();
        let ps = (if sd > 0 { pd } else { 1.0 - pd }) + FUSE_DIR_P;
        pdir = Some(if sd > 0 { ps / 2.0 } else { 1.0 - ps / 2.0 });
    }
    let model_dir = pdir.is_some_and(|p| 0.0 < p && p < 1.0);
    if model_dir {
        p_up = py_round(reach * pdir.unwrap(), 1);
        p_dn = py_round(reach - p_up, 1);
    }
    let share = 100.0 * p_up / f64::max(p_up + p_dn, 1e-9);
    let (mid, rg) = (n(ev, "mid"), n(ev, "range_bp"));
    let (mut hi, mut lo) = (None, None);
    if truthy(mid) && truthy(rg) {
        let (m, r) = (mid.unwrap(), rg.unwrap());
        hi = Some(m * (1.0 + SYM_K * r / 1e4));
        lo = Some(m * (1.0 - SYM_K * r / 1e4));
    }
    let on: Vec<(String, i64)> = votes.iter().map(|(n_, v)| (if n_.starts_with("고래") { "고래".to_string() } else { n_.clone() }, *v)).collect();
    let push = |p: &[(&str, i64, &str); 4]| V::Array(p.iter().map(|(key, sg, t)| obj(vec![("t", json!(t)), ("on", json!(on.contains(&(key.to_string(), *sg))))])).collect());
    let wide = obj(vec![("t", json!(format!("30분 폭 하루 {}분위 — 넓을수록 목표가 멀다", fx(100.0 * gp, 0, false)))), ("on", json!(g == 2))]);
    let dist = |sg: f64| if truthy(rg) { json!(sg * SYM_K * rg.unwrap()) } else { V::Null };
    let cols = json!([
        obj(vec![("key", json!("up")), ("p", json!(p_up)), ("target", json!(hi)), ("dist_bp", dist(1.0)), ("push", push(&PUSH_UP))]),
        obj(vec![("key", json!("dn")), ("p", json!(p_dn)), ("target", json!(lo)), ("dist_bp", dist(-1.0)), ("push", push(&PUSH_DN))]),
        obj(vec![("key", json!("none")), ("p", json!(pn)), ("band", if truthy(hi) { json!([lo, hi]) } else { V::Null }), ("push", json!([wide]))]),
    ]);
    let note = format!("{} · 다음 30분 안 ±0.5×30분 폭 중 하나에 닿는가", if reach_model { REACH_EVID.to_string() } else { format!("닿을 확률 = 3.7년 실측 · 같은 상태 {}번", commas(cn.to_string())) });
    obj(vec![
        ("reg", json!(reg)), ("dir", json!(d)), ("a", json!(a)), ("g", json!(g)), ("n", json!(cn)), ("cols", cols), ("reach", json!(reach)),
        ("up_share", json!(py_round(share, 1))), ("reach_src", json!(if reach_model { "model" } else { "table" })),
        ("dir_src", json!(if model_dir { if blend { "model+fuse" } else { "model" } } else { "table" })),
        ("coin", json!((share - 50.0).abs() < DIR_COIN_PP)), ("dir_note", json!(DIR_EVID)), ("note", json!(note)),
    ])
}

/// flow_read.fuse(430~460).
fn fuse(ev: &V, x: &V) -> V {
    let mut votes: Vec<(String, i64)> = vec![];
    if let Some(z) = x.get("z60").filter(|v| v.as_object().is_some_and(|o| !o.is_empty())) {
        let zw = n(z, "whale").unwrap();
        for (other, name) in [("retail", "고래↔리테일"), ("mid", "고래↔중형")] {
            let zo = n(z, other).unwrap();
            if zw.abs() >= Z_SIDE && zo.abs() >= Z_SIDE && s(Some(zw)) != s(Some(zo)) {
                votes.push((name.into(), s(Some(zw))));
                break;
            }
        }
    }
    let (m60, oi60, p75) = (n(x, "move60"), n(x, "oi60"), n(x, "move60_p75"));
    if let (Some(m), true, Some(p)) = (m60, truthy(oi60), p75) {
        if m < 0.0 && m.abs() >= p {
            votes.push(("가격↔OI".into(), if oi60.unwrap() > 0.0 { -1 } else { 1 }));
        }
    }
    let (veto, d30) = (ev.get("veto").and_then(V::as_i64).unwrap_or(0), ev.get("dir").and_then(V::as_i64).unwrap_or(0));
    if veto != 0 && d30 != 0 && veto == d30 {
        votes.push(("추세정렬".into(), d30));
    }
    if ev.get("reject").and_then(V::as_bool).unwrap_or(false) && d30 != 0 {
        votes.push(("거부 봉".into(), -d30));
    }
    let score: i64 = votes.iter().map(|v| v.1).sum();
    let gp = n(x, "range30_pct");
    let gate = gp.is_some_and(|g| g >= GATE_PCT);
    let sd = if score.abs() >= FUSE_MIN_VOTES && gate { score.signum() } else { 0 };
    let tag = if votes.is_empty() { "켜진 표 없음".to_string() } else { votes.iter().map(|(n_, v)| format!("{n_}{}", if *v > 0 { "↑" } else { "↓" })).collect::<Vec<_>>().join(" · ") };
    let gtxt = match gp {
        None => "관문 ?".to_string(),
        Some(g) => format!("30분 폭 하루 {}분위{}", fx(100.0 * g, 0, false), if gate { "(통과)" } else { "(미달)" }),
    };
    let text = if sd != 0 {
        format!("{} · 다음 30분 — {tag} · {gtxt}", if sd > 0 { "롱" } else { "숏" })
    } else {
        let need = if score.abs() < FUSE_MIN_VOTES { "표가 한쪽으로 2개 필요" } else { "크기 관문 미달" };
        format!("대기 — {tag} (합 {:+}) · {gtxt} · {need}", score)
    };
    let out = outcome3(ev, gp, &votes, score, n(x, "dir_p"), n(x, "reach_p"));
    obj(vec![
        ("side", json!(sd)), ("votes", V::Array(votes.iter().map(|(n_, v)| json!([n_, v])).collect())), ("score", json!(score)), ("gate", json!(gate)),
        ("gate_pct", json!(gp)), ("text", json!(text)), ("note", json!(FUSE_EVID)), ("outcome", out),
    ])
}

/// flow_read.read(250~427): 관계 줄 · 요약 · 융합. `ev` = classify evidence · `x` = _flow_read_ctx.
pub fn read(ev: &V, x: &V) -> V {
    let mut l: Vec<(String, String, String, String, i64)> = vec![];
    let mut add = |topic: &str, text: String, note: String, grade: &str, d: i64| {
        l.push((topic.into(), text, note, grade.into(), if grade == "근거" || grade == "약함" { d } else { 0 }));
    };
    // ① 가격 ↔ 체결
    let (mv, rg, cvd) = (n(ev, "move_bp"), n(ev, "range_bp"), n(ev, "cvd"));
    let (okx, cz) = (n(x, "okx30"), n(x, "cvd30_z"));
    if let (Some(mv), Some(cvd)) = (mv, cvd) {
        let obs = format!("30분 {}bp · 바이낸스 체결 {} ETH{}", fx(mv, 0, true), fc(cvd, 0, true), okx.map_or(String::new(), |o| format!(" · OKX {}", fc(o, 0, true))));
        let flat = rg.is_some_and(|r| mv.abs() < FLAT_FRAC * r);
        let loud = cz.is_none_or(|z| z.abs() >= CVD_Z_BIG);
        let t = if flat && loud && cvd != 0.0 {
            format!("{obs} — 시장가가 {}로 몰렸는데 가격은 제자리: 반대편 지정가가 받아내는 중(흡수)", side(cvd, "매수", "매도"))
        } else if s(Some(cvd)) == s(Some(mv)) || cvd == 0.0 {
            format!("{obs} — 시장가가 가격을 {}", if mv > 0.0 { "올렸다" } else { "내렸다" })
        } else if okx.is_some() && s(okx) == s(Some(mv)) {
            format!("{obs} — 바이낸스 시장가는 {}인데 가격은 {}: OKX 가 {}로 끌었다", side(cvd, "매수", "매도"), updn(mv), side(okx.unwrap(), "매수", "매도"))
        } else {
            format!("{obs} — 시장가는 {}인데 가격은 {}: 지정가 {}가 받아냈다(흡수)", side(cvd, "매수", "매도"), updn(mv), side(mv, "매수", "매도"))
        };
        add("가격↔체결", t, evid("price_flow").into(), "설명", 0);
    }
    // ② 고래 ↔ 중형 ↔ 리테일
    let net = x.get("net60").filter(|v| v.as_object().is_some_and(|o| !o.is_empty()));
    let z = x.get("z60").filter(|v| v.as_object().is_some_and(|o| !o.is_empty()));
    if let Some(net) = net {
        let obs = format!("60분 순매수 고래 {} · 중형 {} · 리테일 {} ETH", fc(n(net, "whale").unwrap(), 0, true), fc(n(net, "mid").unwrap(), 0, true), fc(n(net, "retail").unwrap(), 0, true));
        match z {
            None => add("고래↔리테일", obs + " — 기준 분포 수집 중(서버 기동 뒤 6시간)", evid("same_side").into(), "설명", 0),
            Some(z) => {
                let (zw, zm, zr) = (n(z, "whale").unwrap(), n(z, "mid").unwrap(), n(z, "retail").unwrap());
                let big = |v: f64| v.abs() >= Z_SIDE;
                let m60 = n(x, "move60");
                let against = m60.is_some() && s(Some(zw)) == -s(m60);
                if big(zw) && big(zr) && s(Some(zw)) != s(Some(zr)) {
                    add("고래↔리테일", format!("{obs} — 고래 {} · 리테일 {}로 갈렸다: 고래 쪽{}", side(zw, "매수", "매도"), side(zr, "매수", "매도"), if against { " (고래가 가격과 반대로 섰다 — 이때 더 강했다 +10bp)" } else { "" }), evid("whale_retail").into(), "약함", s(Some(zw)));
                } else if big(zw) && big(zm) && s(Some(zw)) != s(Some(zm)) {
                    add("고래↔중형", format!("{obs} — 고래 {} · 중형 {}로 갈렸다: 고래 쪽", side(zw, "매수", "매도"), side(zm, "매수", "매도")), evid("whale_mid").into(), "약함", s(Some(zw)));
                } else if big(zw) && big(zm) && big(zr) && s(Some(zw)) == s(Some(zm)) && s(Some(zm)) == s(Some(zr)) {
                    add("고래↔리테일", format!("{obs} — 셋 다 {}: 한쪽 쏠림", side(zw, "매수", "매도")), evid("same_side").into(), "설명", 0);
                } else {
                    add("고래↔리테일", format!("{obs} — 크기별로 뚜렷한 갈림·쏠림 없음"), evid("same_side").into(), "설명", 0);
                }
            }
        }
    }
    // ③ 가격 ↔ OI
    let (m60, oi60, p75) = (n(x, "move60"), n(x, "oi60"), n(x, "move60_p75"));
    if let (Some(m60), Some(oi60)) = (m60, oi60) {
        let obs = format!("1시간 {}bp · OI {} ETH", fx(m60, 0, true), fc(oi60, 0, true));
        if p75.is_none_or(|p| m60.abs() < p) {
            add("가격↔OI", format!("{obs} — 이동이 작다(하루 상위 25% 밖): OI 방향만으로는 말할 게 없다"), evid("oi_small").into(), "설명", 0);
        } else if m60 < 0.0 && oi60 > 0.0 {
            add("가격↔OI", format!("{obs} — 하락 + OI 증가: 새 숏이 밀고 있다"), evid("oi_dn_up").into(), "근거", -1);
        } else if m60 < 0.0 {
            add("가격↔OI", format!("{obs} — 하락 + OI 감소: 롱이 손절·청산으로 빠진다"), evid("oi_dn_dn").into(), "근거", 1);
        } else if oi60 > 0.0 {
            add("가격↔OI", format!("{obs} — 상승 + OI 증가: 새 롱 유입"), evid("oi_up").into(), "미측정", 0);
        } else {
            add("가격↔OI", format!("{obs} — 상승 + OI 감소: 숏 커버(스퀴즈)로 오른다"), evid("oi_up").into(), "미측정", 0);
        }
    }
    // ④ 청산
    let (ll, ls) = (n(ev, "liq_long").unwrap_or(0.0), n(ev, "liq_short").unwrap_or(0.0));
    let s60 = x.get("liq60s").cloned().unwrap_or(V::Null);
    let (s6l, s6s) = (n(&s60, "long"), n(&s60, "short"));
    let tail = if s6l.unwrap_or(0.0) + s6s.unwrap_or(0.0) > 0.0 { format!(" · 최근 60초 롱 ${} / 숏 ${}", fc(s6l.unwrap_or(0.0), 0, false), fc(s6s.unwrap_or(0.0), 0, false)) } else { String::new() };
    if ev.get("liq_long").is_none() {
    } else if ll + ls < LIQ_MIN_USD {
        add("청산", format!("30분 청산 롱 ${} · 숏 ${}{tail} — 거의 없다", fc(ll, 0, false), fc(ls, 0, false)), evid("liq_result").into(), "설명", 0);
    } else {
        let dom = if ls > ll { 1 } else { -1 };
        let veto = ev.get("veto").and_then(V::as_i64).unwrap_or(0);
        let who = if dom > 0 { "숏" } else { "롱" };
        let obs = format!("30분 청산 롱 ${} · 숏 ${}{tail}", fc(ll, 0, false), fc(ls, 0, false));
        if veto != 0 && dom == -veto {
            add("청산↔추세", format!("{obs} — 12시간 추세({})의 반대편({who})이 털렸다: 눌림 청산", updn(veto as f64)), evid("liq_flush").into(), "약함", veto);
        } else if mv.is_some() && s(mv) == dom {
            add("청산↔가격", format!("{obs} — 이동이 {who}을 청산시켰다(결과)"), evid("liq_result").into(), "설명", 0);
        } else {
            add("청산↔가격", format!("{obs} — {who} 청산이 우세"), evid("liq_result").into(), "설명", 0);
        }
    }
    // ⑤ 호가 ↔ 체결
    let (ip, imb) = (n(x, "imb40_pct"), n(x, "imb40"));
    let (agree, live) = (x.get("agree").and_then(V::as_str), x.get("trigger_live").is_some_and(|v| v.as_bool().unwrap_or(false)));
    if let (Some(ip), Some(imb), true) = (ip, imb, ip.is_some_and(|p| p >= DEEP_HI || p <= DEEP_LO)) {
        let d = if ip >= DEEP_HI { 1 } else { -1 };
        let rank = if d > 0 { 100.0 * (1.0 - ip) } else { 100.0 * ip };
        let mut t = format!("깊은 호가(±0.4%) 불균형 {}(6시간 {} {}%) — {} 우위", fx(imb, 2, true), if d > 0 { "상위" } else { "하위" }, fx(rank, 0, false), side(d as f64, "매수벽", "매도벽"));
        if truthy(cvd) && s(cvd) == -d {
            t += &format!(", 시장가({})를 받아낼 쪽이 두껍다", side(cvd.unwrap(), "매수", "매도"));
        }
        add("호가↔체결", t, evid("deep_wall").into(), "약함", d);
    } else if matches!(agree, Some("동조매수") | Some("동조매도")) && live {
        add("호가↔체결", format!("최우선 호가 기울기와 호가 증감이 {} 쪽으로 동조(방금)", agree.unwrap().chars().skip(2).collect::<String>()), evid("qi").into(), "설명", 0);
    } else {
        let (obi, pers) = (n(ev, "obi"), n(ev, "persist"));
        let mut t = "호가 불균형 자료 없음".to_string();
        if let Some(o) = obi {
            t = format!("호가 불균형 {}{} — 깊은 벽 쏠림 없음", fx(o, 2, true), pers.map_or(String::new(), |p| format!(" · 벽 지속 {}%", fx(p * 100.0, 0, false))));
        }
        add("호가↔체결", t, evid("wall_sr").into(), "설명", 0);
    }
    // ⑥ 청산맵 지지/저항
    let sr = x.get("sr").cloned().unwrap_or(V::Null);
    let (sb, rb) = (n(&sr, "sup_bp"), n(&sr, "res_bp"));
    if sb.is_some() || rb.is_some() {
        let mut parts = vec![];
        if let Some(rb) = rb {
            parts.push(format!("저항 {}(+{}bp)", fc(n(&sr, "res").unwrap(), 1, false), fx(rb, 0, false)));
        }
        if let Some(sb) = sb {
            parts.push(format!("지지 {}(−{}bp)", fc(n(&sr, "sup").unwrap(), 1, false), fx(sb, 0, false)));
        }
        let near = sr.get("near").and_then(V::as_str).filter(|s| !s.is_empty()).unwrap_or("없음").to_string();
        if near != "없음" {
            let mut t = format!("{} — {}에 붙어 있다: 청산이 날 자리", parts.join(" · "), near.chars().take(2).collect::<String>());
            match sr.get("deep_wall").and_then(V::as_str).filter(|s| !s.is_empty()) {
                Some(dw) => {
                    t += &format!(", 깊은 벽은 {dw}");
                    let hold = dw == "레벨쪽";
                    add("지지/저항", t, format!("{} · {}", evid("sr_near"), evid("sr_wall")), if hold { "약함" } else { "설명" }, if hold { if near == "지지근접" { 1 } else { -1 } } else { 0 });
                }
                None => add("지지/저항", t, evid("sr_near").into(), "설명", 0),
            }
        } else {
            add("지지/저항", format!("{} — 둘 다 ±{}bp 밖", parts.join(" · "), fx(NEAR_BP, 0, false)), evid("sr_near").into(), "설명", 0);
        }
    }
    // ⑦ 선물 ↔ 현물 · 펀딩
    let (bas, bd, lead, fr, crowd) = (n(ev, "basis_bp"), n(ev, "basis_d_bp"), ev.get("lead").and_then(V::as_i64), n(ev, "funding"), ev.get("crowd").and_then(V::as_i64));
    if let Some(bas) = bas {
        let mut t = format!("베이시스 {}bp{}", fx(bas, 1, true), bd.map_or(String::new(), |b| format!("(30분 {})", fx(b, 1, true))));
        if bd.is_none() {
            t += " — 30분 변화 수집 중";
        } else if !truthy(n(ev, "dir")) {
            t += " — 30분 추세가 없어 주도 판정 보류";
        } else {
            t += match lead.unwrap_or(0) {
                1 => " — 선물이 이동을 끌었다",
                -1 => " — 현물이 이동을 끌었다",
                _ => " — 선물·현물 한쪽이 끈 흔적 없음",
            };
        }
        if let Some(f) = fr {
            t += &format!(" · 펀딩 {}%{}", fx(f * 100.0, 4, true), match crowd.unwrap_or(0) { 1 => "(롱 쏠림)", -1 => "(숏 쏠림)", _ => "" });
        }
        add("선물↔현물", t, format!("{} · {}", evid("basis"), evid("funding")), if lead.unwrap_or(0) != 0 { "미측정" } else { "설명" }, 0);
    }
    // ⑧ ETH ↔ BTC
    if let Some(bm) = n(ev, "btc_move_bp") {
        let rel = ev.get("btc_rel").and_then(V::as_str).unwrap_or("");
        add("ETH↔BTC", format!("BTC 30분 {}bp — {}", fx(bm, 0, true), match rel { "동행" => "같이 움직인다(시장 전체)", "단독" => "ETH 혼자 움직인다", _ => "ETH 가 추세가 아니라 비교 보류" }), evid("btc").into(), "설명", 0);
    }
    // ⑨ 추세 veto ↔ 30분 이동
    let (veto, d30) = (ev.get("veto").and_then(V::as_i64).unwrap_or(0), ev.get("dir").and_then(V::as_i64).unwrap_or(0));
    if veto != 0 && d30 != 0 {
        let al = veto == d30;
        add("추세↔이동", format!("12시간 추세 {} 안의 30분 {} — {}", updn(veto as f64), updn(d30 as f64), if al { "정렬: 더 갈 쪽에 조금 기운다" } else { "추세 안의 짧은 역행: 지속은 동전" }), evid("veto_align").into(), if al { "약함" } else { "설명" }, if al { d30 } else { 0 });
    }
    // ⑩ 활동
    let (act, vp) = (x.get("act").and_then(V::as_str).filter(|s| !s.is_empty()), n(x, "vol_pct"));
    if let Some(vp) = vp {
        let edge = vp >= 0.8 || vp <= 0.2;
        add("활동", format!("60초 거래량 같은 시간대 {}분위{} — {}", fx(100.0 * vp, 0, false), act.map_or(String::new(), |a| format!("({a})")), if vp >= 0.8 { "움직임이 커질 자리" } else if vp <= 0.2 { "움직임이 작을 자리" } else { "움직임 크기 평소" }), evid("act").into(), if edge { "근거" } else { "설명" }, 0);
    }

    let fused = fuse(ev, x);
    let g = gauges(ev, x);
    let mut lines = vec![];
    for (topic, text, note, grade, d) in &l {
        let key = if topic.starts_with("고래") { "고래" } else if topic.starts_with("청산") { "청산" } else { topic.as_str() };
        let mut gg = g.iter().find(|x| x.0 == key).map(|x| x.1.clone()).unwrap_or(V::Null);
        if !gg.is_null() && *d != 0 {
            let v = gg["v"].as_f64().unwrap();
            gg["v"] = json!(*d as f64 * f64::max(v.abs(), 30.0));
            gg["on"] = json!(true);
        }
        lines.push(obj(vec![("topic", json!(topic)), ("text", json!(text)), ("note", json!(note)), ("grade", json!(grade)), ("dir", json!(d)), ("g", gg)]));
    }
    let dirs: Vec<&(String, String, String, String, i64)> = l.iter().filter(|x| x.4 != 0).collect();
    let (up, dn) = (dirs.iter().filter(|x| x.4 > 0).count(), dirs.iter().filter(|x| x.4 < 0).count());
    let summary = if dirs.is_empty() {
        "방향을 말할 근거가 있는 관계가 지금은 없다 — 아래는 상태 설명".to_string()
    } else {
        let names = dirs.iter().map(|x| format!("{}({})", x.0, if x.4 > 0 { "↑" } else { "↓" })).collect::<Vec<_>>().join(" · ");
        format!("방향 근거 {}개: {names}{}", dirs.len(), if up > 0 && dn > 0 { " — 서로 엇갈린다" } else { "" })
    };
    obj(vec![("lines", V::Array(lines)), ("summary", json!(summary)), ("up", json!(up)), ("down", json!(dn)), ("fused", fused)])
}

/// flow_read.vol_size_mult(123~136): 완결 5분봉(시간순 · time·close) 마지막 289개 → 권장 배수. 모자라거나 끊기면 None.
pub fn vol_size_mult(bars: &[(i64, f64)], asset: &str) -> V {
    let (med, rf) = match asset {
        "eth" => (157.3941696083474, 0.7657638410664205),
        "xrp" => (167.3260352009301, 0.7586429414343359),
        _ => return V::Null,
    };
    if bars.len() < 289 {
        return V::Null;
    }
    let b = &bars[bars.len() - 289..];
    if b.windows(2).any(|w| w[1].0 - w[0].0 != 300) {
        return V::Null;
    }
    let lc: Vec<f64> = b.iter().map(|x| x.1.ln()).collect();
    let rr: Vec<f64> = lc.windows(2).map(|w| (w[1] - w[0]) * 1e4).map(|r| r * r).collect();
    let sig = (rtcollect::ob5m::np_sum(&rr) / 6.0).sqrt();
    let (lo, hi) = (0.25, 3.0);
    json!({"sigma_bp": sig, "mult": (sig / med).clamp(lo, hi) / rf, "bar": b[b.len() - 1].0, "ref_bp": med * rf, "lo": lo / rf, "hi": hi / rf})
}

/// 코어 Situation(직렬화 값) → 원본 situation_state["now"] 모양: `missing` 칸을 빼고, classify 의 «통과 칸»(evidence.veto·obi·persist)을
/// 받는 쪽이 가진 값으로 채운다(원본 situation.classify 65·69 -- 방향·문턱 계산에는 안 쓰인다). 코어가 ok=false 면 evidence 가 없어 그대로.
pub fn patch_now(core: &V, veto: i64, obi: Option<f64>, persist: Option<f64>) -> V {
    let mut m = core.as_object().cloned().unwrap_or_default();
    m.remove("missing");
    if let Some(ev) = m.get_mut("evidence").and_then(V::as_object_mut) {
        ev.insert("veto".into(), json!(veto));
        ev.insert("obi".into(), json!(obi));
        ev.insert("persist".into(), json!(persist));
    }
    V::Object(m)
}

/// 계산 전 상태(server.py _coin_ctx 1777): now = {ok: false, reason: "계산 전"} · computed_at = 0.0.
pub fn initial_now() -> V {
    json!({"ok": false, "reason": "계산 전"})
}

/// server.py situation_payload(2734~2738). `read` = 관계 읽기 결과(또는 원본 예외 모양 {lines:[], summary:"", error}) · 없으면 None.
pub fn situation_payload(now: &V, computed_at: &V, fo: &V, mp: &V, read: &V, vol_mult: &V, vol_mult_by: &[(String, V)]) -> V {
    let mut by = Map::new();
    for (a, v) in vol_mult_by {
        by.insert(a.clone(), v.clone());
    }
    json!({"now": now, "computed_at": computed_at, "streams": {"fo": fo, "mp": mp}, "read": read, "vol_mult": vol_mult, "vol_mult_by": V::Object(by)})
}

/// SSE `event: situation` 한 건(api_stream 3175~3179): `json.dumps(…, ensure_ascii=False, separators=(',', ':'))`.
pub fn sse_situation(payload: &V) -> String {
    format!("event: situation\ndata: {}\n\n", rtserve::py::dumps(payload, (",", ":"), false, None))
}

/// 코어 수신 상태 → 원본 fo_state(1801)·mp_state(1774) dict 모양(키 차례 그대로). mp 의 next_funding_ms 는 첫 마크 뒤에야 키가 생긴다(1893~1894).
/// last_error 글자는 Rust 오류(원본 repr(exc)[:120] 과 같을 수 없다 -- 대조에서 뺄 칸).
pub fn streams_payload(s: &rtcore::proto::Streams) -> (V, V) {
    let (f, m) = (&s.fo, &s.mp);
    let fo = json!({"connected": f.connected, "since": f.since, "last_event_ms": f.last_ms, "events": f.events, "errors": f.errors, "last_error": f.last_error});
    let mut mp = json!({"connected": m.connected, "since": m.since, "last_ms": m.last_ms, "events": m.events, "errors": m.errors, "last_error": m.last_error});
    if m.events > 0 {
        mp["next_funding_ms"] = json!(m.next_funding_ms);
    }
    (fo, mp)
}

/// 융합 카드 장부 한 줄(server.py compute_situation 2494~2513 · ETH 만). `read` = 이 회차 관계 읽기 · `mid` = inp["mid"](미시 payload mid) ·
/// `card_p` = card30["p"](dir·reach) · `feat_bar` = card30["bar"]. outcome 이 없으면(None 또는 빈 dict) None.
pub fn fused_log_line(now: f64, read: &V, mid: &V, card_p: &V, feat_bar: &V) -> Option<String> {
    let fu = read.get("fused").filter(|v| v.as_object().is_some_and(|o| !o.is_empty())).cloned().unwrap_or(json!({}));
    let o3 = fu.get("outcome").filter(|v| v.as_object().is_some_and(|o| !o.is_empty()))?;
    let bar_now = (now as i64).div_euclid(300) * 300;
    let cols = o3["cols"].as_array()?;
    let mut p = Map::new();
    for c in cols {
        p.insert(c["key"].as_str()?.to_string(), c["p"].clone());
    }
    let g = |k: &str| fu.get(k).cloned().unwrap_or(V::Null);
    let src = o3.get("dir_src").and_then(V::as_str).unwrap_or("");
    let line = json!({"ts": now as i64, "bar": bar_now, "side": g("side"), "mid": mid, "votes": g("votes"), "score": g("score"), "gate_pct": g("gate_pct"),
        "cell": [o3["reg"], o3["a"], o3["g"]], "p": V::Object(p),
        "dir_p": if src.starts_with("model") { card_p.get("dir").cloned().unwrap_or(V::Null) } else { V::Null },
        "reach_p": if o3.get("reach_src").and_then(V::as_str) == Some("model") { card_p.get("reach").cloned().unwrap_or(V::Null) } else { V::Null },
        "feat_bar": feat_bar, "dir_src": o3.get("dir_src").cloned().unwrap_or(V::Null), "up_share": o3.get("up_share").cloned().unwrap_or(V::Null),
        "up": cols[0].get("target").cloned().unwrap_or(V::Null), "dn": cols[1].get("target").cloned().unwrap_or(V::Null)});
    Some(rtserve::py::dumps(&line, (", ", ": "), false, None) + "\n")
}

/// server.py _last_logged_bar(2408~2416): 꼬리 16KB 의 마지막 줄 bar. 못 읽으면 None.
pub fn last_logged_bar(path: &std::path::Path) -> Option<i64> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).ok()?;
    let size = f.seek(SeekFrom::End(0)).ok()?;
    f.seek(SeekFrom::Start(size.saturating_sub(16384))).ok()?;
    let mut b = vec![];
    f.read_to_end(&mut b).ok()?;
    let t = String::from_utf8_lossy(&b).replace('\u{FFFD}', ""); // decode("utf-8", "ignore")
    let last = t.trim().lines().last()?;
    let v: V = serde_json::from_str(last).ok()?;
    match &v["bar"] {
        V::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)),
        V::String(s) => s.trim().parse().ok(),
        V::Bool(b) => Some(*b as i64),
        _ => None,
    }
}

/// 장부 쓰기(server.py 2497~2515): 봉마다 한 줄 · 재기동 뒤 첫 회차는 파일 꼬리에서 «마지막으로 쓴 봉»을 되살린다.
/// 🔴`RTSERVE_READONLY` 가 "0" 이 아니면(병행 중 기본) 쓰지 않는다 -- 파이썬이 같은 파일에 쓰는 동안 두 번 쓰지 않게. 전환 때 0 으로.
pub struct FusedLog {
    pub path: std::path::PathBuf,
    /// None = 아직 안 되살림(원본 "fused_bar" not in situation_state).
    bar: Option<Option<i64>>,
}

impl FusedLog {
    pub fn new(path: std::path::PathBuf) -> Self {
        FusedLog { path, bar: None }
    }

    /// 쓴 줄(안 썼으면 None). 쓰기 실패는 원본처럼 알리고 넘어간다(fused log: …).
    pub fn tick(&mut self, now: f64, read: &V, mid: &V, card_p: &V, feat_bar: &V) -> Option<String> {
        if std::env::var("RTSERVE_READONLY").as_deref() != Ok("0") {
            return None;
        }
        let line = fused_log_line(now, read, mid, card_p, feat_bar)?;
        let bar_now = (now as i64).div_euclid(300) * 300;
        let last = *self.bar.get_or_insert_with(|| last_logged_bar(&self.path));
        if last == Some(bar_now) {
            return None;
        }
        self.bar = Some(Some(bar_now));
        use std::io::Write;
        let r = std::fs::OpenOptions::new().append(true).create(true).open(&self.path).and_then(|mut f| f.write_all(line.as_bytes()));
        if let Err(e) = r {
            println!("fused log: {e}");
        }
        Some(line)
    }
}
