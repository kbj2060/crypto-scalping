//! 웹어셈블리 입구. 입출력(fetch·WebSocket·타이머·탭 숨김)은 index.html 의 얇은 접착이 한다:
//! Rust 가 할 일(feed::Req)을 `window.rtui_io(JSON)` 로 넘기고, 접착은 결과를 on_get · on_ws · on_ws_close 로 돌려준다. 타이머 = 접착의 500ms tick(app.js POLL_MS 16).
use super::*;
use eframe::egui::pos2;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance)]
    fn now() -> f64;
    #[wasm_bindgen(js_namespace = Date, js_name = now)]
    fn date_now() -> f64;
    #[wasm_bindgen(js_namespace = window)]
    fn rtui_io(reqs: &str);
}

thread_local! {
    static CH: RefCell<Option<Rc<RefCell<Chart>>>> = Default::default();
    static CTX: RefCell<Option<Context>> = Default::default();
    /// 마지막으로 그린 장면 -- JSON 은 하네스가 scene() 을 부를 때만 만든다(매 프레임 만들면 프레임당 3~4ms: rtui/bench 측정)
    static LAST: RefCell<Option<(fp::Scene, Option<hover::Hover>)>> = Default::default();
    /// 측정(compare/live_bench.py): ui() 끝 시각(performance.now) · ui() 시간(ms)
    static LAT: RefCell<(Vec<f64>, Vec<f64>)> = Default::default();
}

fn with<R>(f: impl FnOnce(&mut Chart) -> R) -> Option<R> {
    let ch = CH.with(|c| c.borrow().clone())?;
    let r = f(&mut ch.borrow_mut());
    flush(&mut ch.borrow_mut());
    Some(r)
}
/// 쌓인 할 일을 접착에 넘기고, 그릴 것이 바뀌었으면 다시 그리기를 부른다(바뀐 게 없으면 egui 는 프레임을 안 돈다)
fn flush(ch: &mut Chart) {
    if !ch.feed.out.is_empty() { rtui_io(&serde_json::to_string(&std::mem::take(&mut ch.feed.out)).unwrap()); }
    if ch.feed.take_dirty() { CTX.with(|c| if let Some(c) = c.borrow().as_ref() { c.request_repaint(); }); }
}

/// 마지막으로 그린 장면(JSON) -- 비교 하네스
#[wasm_bindgen]
pub fn scene() -> Option<String> { LAST.with(|l| l.borrow().as_ref().map(|(sc, hv)| scene_json(sc, hv.as_ref()))) }
/// 창 상태(JSON) -- 비교 하네스
#[wasm_bindgen]
pub fn state() -> Option<String> { with(|c| c.state_json().to_string()) }
#[wasm_bindgen]
pub fn tick(hidden: bool) { with(|c| { let w = c.view.want_week(); c.feed.tick(date_now(), hidden, w) }); }
/// kind = "Footprint" | "History" | "Week" | "LiqSig" | "Liq" | "LiqTail" · body = 응답 글자(실패면 null)
#[wasm_bindgen]
pub fn on_get(kind: &str, body: Option<String>) {
    with(|c| match kind {
        "Footprint" => c.feed.on_footprint(body.as_deref()),
        "History" => c.feed.on_history(body.as_deref(), date_now()),
        "Week" => c.feed.on_week(body.as_deref(), date_now()),
        "LiqSig" => c.feed.on_liq_sig(body.as_deref()),
        "Liq" => c.feed.on_liq(body.as_deref()),
        "LiqTail" => c.feed.on_liq_tail(body.as_deref()),
        "Mc" => c.feed.on_mc(body.as_deref()),
        "Acct" => c.feed.on_acct(body.as_deref()),
        "Srl" => c.feed.on_srl(body.as_deref(), date_now()),
        "DirShort" => c.feed.on_dirs(body.as_deref()),
        "PredRec" => c.feed.on_pred(body.as_deref()),
        "Trend" => c.feed.on_fc(feed::Kind::Trend, body.as_deref()),
        "Dir4h" => c.feed.on_fc(feed::Kind::Dir4h, body.as_deref()),
        "OpenRev" => c.feed.on_fc(feed::Kind::OpenRev, body.as_deref()),
        "Band" => c.feed.on_fc(feed::Kind::Band, body.as_deref()),
        "Radar" => c.feed.on_fc(feed::Kind::Radar, body.as_deref()),
        "LiqTab" => c.feed.on_liq_tab(body.as_deref()),
        "LiqMap" => c.feed.on_liq_map(body.as_deref()),
        "LiqAll" => if let Some(x) = c.feed.liqx.as_mut() { x.on_all(body.as_deref()) },
        "LiqBars" => if let Some(x) = c.feed.liqx.as_mut() { x.on_bars(body.as_deref()) },
        _ => {}
    });
}
#[wasm_bindgen]
pub fn on_ws(text: &str) {
    with(|c| c.feed.on_ws(text, date_now()));
}
#[wasm_bindgen]
pub fn on_ws_close() { with(|c| c.feed.on_ws_close()); }
/// rtcore 밀기 자리(아직 연결 없음 -- rtcore/src/proto.rs Footprint JSON)
#[wasm_bindgen]
pub fn on_push(json: &str) -> Result<(), JsValue> {
    let p: feed::Push = serde_json::from_str(json).map_err(|e| JsValue::from_str(&e.to_string()))?;
    with(|c| c.feed.apply_push(p));
    Ok(())
}
/// 하네스: 창 봉 수를 바로(지금 화면 쪽은 chartWindowBars 를 바로 쓴다 -- compare/capture.py)
#[wasm_bindgen]
pub fn set_bars(n: f64) { with(|c| { c.view.bars = n; c.feed.dirty = true; }); }
/// 5m·30m·1h 버튼 = 1·6·12
#[wasm_bindgen]
pub fn tf(k: i32) { with(|c| c.tf(k as i64, date_now())); }
/// 측정: [ui() 끝 시각..] NaN [ui() 시간 ms..] 를 비우며 돌려준다
#[wasm_bindgen]
pub fn take_lat() -> Vec<f64> { LAT.with(|l| { let mut l = l.borrow_mut(); let mut v = std::mem::take(&mut l.0); v.push(f64::NAN); v.extend(std::mem::take(&mut l.1)); v }) }

struct WebApp(Rc<RefCell<Chart>>);
impl eframe::App for WebApp {
    fn clear_color(&self, _v: &egui::Visuals) -> [f32; 4] { self.0.borrow().bg.to_normalized_gamma_f32() }
    fn ui(&mut self, ui: &mut egui::Ui, _f: &mut eframe::Frame) {
        let t0 = now();
        let mut ch = self.0.borrow_mut();
        let r = ch.ui(ui, date_now());
        flush(&mut ch);
        if r.is_some() { LAST.with(|l| *l.borrow_mut() = r); }
        let t1 = now();
        LAT.with(|l| { let mut l = l.borrow_mut(); l.0.push(t1); l.1.push(t1 - t0); });
    }
}

/// asset = ?asset= · (x, y) = 상자 왼쪽 위 · w·h = viewBox(0 이면 창에 맞춤) · tz = getTimezoneOffset · bg = 0xRRGGBB(0xFFFFFFFF 투명)
/// week = 지표 스위치 기본값이 1주일을 늘 부르나(feed::Feed::week_always) · api·ws·hub = 주소(feed::Urls -- 빈 글자면 기본) · push = 풋프린트·체결을 rtcore 허브에서(feed::Source::Push)
/// card = "s1" 이면 1초 수급 카드만(s1.rs · push 여야 자료가 온다) · "mc" 면 시장 맥락 ①② 만(mc.rs · 같은 출처 /api/market-context) · "acct" 면 계좌 칩만(acct.rs) · "srl" 이면 «레벨 5분» 칸만(srl.rs · 같은 출처 /api/sr-lean · ETH 만) · "dirs" 면 ⑤ 예측 줄 전체(fc.rs · 같은 출처 /api/trend · dir-short · sr-lean · dir4h · open-rev · regime-band · risk-radar · pred-record + 허브 가격 · push 로 연다) · "liq7"(liqN · N = 지도 기간 1·7·30·60·90·180·365) 이면 «청산» 탭 청산맵만(liq.rs · push 로 연다) · "rem60"(remN · N = 입력 기간 7·30·60·90·180·365) 이면 «청산» 탭 «남은 매물 일봉별»만(rem.rs · push 로 연다)
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub async fn start(canvas_id: String, asset: String, x: f32, y: f32, w: f64, h: f64, tz: i32, bg: u32, week: bool, api: String, ws: String, push: bool, hub: String, card: String) -> Result<(), JsValue> {
    use eframe::wasm_bindgen::JsCast;
    let (sym, dp, unit) = asset_cfg(&asset).ok_or_else(|| JsValue::from_str("흐름 코인이 아니다(app.js FLOW_ASSETS)"))?;
    let mut urls = feed::Urls::default();
    if !api.is_empty() { urls.api = api; }
    if !ws.is_empty() { urls.trade_ws = ws; }
    if !hub.is_empty() { urls.hub = hub; }
    let bgc = if bg == u32::MAX { Color32::TRANSPARENT } else { Color32::from_rgb((bg >> 16) as u8, (bg >> 8) as u8, bg as u8) };
    let mut cfg = default_cfg(view::FP_BARS, w, h, tz as i64);
    (cfg.dp, cfg.unit) = (dp, unit.into());
    let mut ch = Chart::new(feed::Feed::new(&asset, sym, if push { feed::Source::Push } else { feed::Source::Poll }, urls, week), cfg, pos2(x, y), bgc);
    ch.fill = w == 0.0;
    ch.feed.s1_only = card == "s1"; // 1초 수급 카드만(index.html ?card=s1 -- 허브 밀기로 연다)
    ch.feed.mc_only = card == "mc";
    ch.feed.srl_only = card == "srl";
    ch.feed.dirs_only = card == "dirs";
    if let Some(d) = card.strip_prefix("liq").and_then(|d| d.parse::<u32>().ok()).filter(|d| [1, 7, 30, 60, 90, 180, 365].contains(d)) { (ch.feed.liq_only, ch.feed.liq_days) = (true, d); }
    if let Some(d) = card.strip_prefix("rem").and_then(|d| d.parse::<u32>().ok()).filter(|d| [7, 30, 60, 90, 180, 365].contains(d)) { (ch.feed.rem_only, ch.rem.win) = (true, d); }
    if let Some(x) = card.strip_prefix("liqx").and_then(crate::liqx::State::new) { (ch.feed.liqx, ch.feed.rem_only, ch.feed.source) = (Some(x), true, feed::Source::Push); } // «청산» 탭 아랫줄(?card=liqx · liqx1W 처럼 주기) -- 허브 가격이라 늘 밀기
    ch.feed.acct_only = card == "acct"; // 계좌 칩(맨 위 전광판)만 -- 같은 출처 /api/binance-account + 허브 «모든 종목» 가격(push 로 연다)
    let ch = Rc::new(RefCell::new(ch));
    CH.with(|c| *c.borrow_mut() = Some(ch.clone()));
    let canvas = eframe::web_sys::window().unwrap().document().unwrap().get_element_by_id(&canvas_id).unwrap()
        .dyn_into::<eframe::web_sys::HtmlCanvasElement>()?;
    eframe::WebRunner::new().start(canvas, eframe::WebOptions::default(), Box::new(move |cc| {
        cc.egui_ctx.set_fonts(draw::fonts());
        draw::canvas_like(&cc.egui_ctx);
        CTX.with(|c| *c.borrow_mut() = Some(cc.egui_ctx.clone()));
        Ok(Box::new(WebApp(ch)))
    })).await
}

/// 측정: 창 없이 n 프레임(앞 30 버림) -- [구성 ms, 테셀레이션 ms] × n 을 펼친 배열
#[wasm_bindgen]
pub fn bench(fp_json: String, mh_json: String, w: f64, h: f64, bars: f64, tz: i32, n: usize) -> Vec<f64> {
    let mut ch = Chart::from_json(&fp_json, &mh_json, default_cfg(bars, w, h, tz as i64), pos2(113.0, 126.0), Color32::from_rgb(0x1e, 0x22, 0x29));
    let ctx = new_ctx();
    let mut out = vec![];
    for f in 0..n + 30 {
        ch.nudge(f);
        let (a, b) = bench_frame(&ctx, &mut ch, &now);
        if f >= 30 { out.push(a); out.push(b); }
    }
    out
}
