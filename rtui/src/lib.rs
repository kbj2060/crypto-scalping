//! rtui -- 웹 대시보드 화면을 egui(네이티브 + 웹어셈블리)로. 첫 카드: 데스크톱 1920 풋프린트 카드(지금 화면 그대로 옮김).
//! data = 응답 모양 · feed = 자료가 들어오는 자리(묻기/밀기 · @trade 형성 봉 · 현재가) · view = 창 상태·마우스 조작·30m/1h 묶기
//! · fp = 차트 가공(기하·칸·캔들·델타·POC·축·알약) · lane = 아래 레인 · profile = 체결·델타·TPO 기둥 · marks = 스택·미완 경매
//! · hover = 십자선·가격 배지·툴팁 · draw = egui 그리기.
pub mod acct;
pub mod data;
pub mod draw;
pub mod feed;
pub mod fp;
pub mod hover;
pub mod lane;
pub mod liq;
pub mod marks;
pub mod profile;
pub mod mc;
pub mod s1;
pub mod srl;
pub mod view;

use eframe::egui::{self, Color32, Context, Event, MouseWheelUnit, PointerButton, Pos2, Rect, vec2};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};

/// 마지막 그림의 자리(app.js chartGeo -- 끌기·휠이 화면 px → 봉으로 바꾼다)
#[derive(Clone, Copy, Default)]
pub struct LastGeo { pub ml: f64, pub cw: f64, pub n: f64, pub mt: f64, pub pb: f64, pub side_x: f64 }

pub struct Chart {
    pub feed: feed::Feed,
    pub view: view::View,
    pub cfg: fp::Cfg,
    /// svg 상자 왼쪽 위(화면 px) -- 비교 캡처에서 지금 화면과 같은 자리
    pub origin: Pos2,
    /// 바탕(투명이면 칠하지 않는다: 비교 캡처는 지금 화면의 바탕을 뒤에 깐다)
    pub bg: Color32,
    /// 창 크기에 맞춰 cfg.w_all·h_all 을 매 프레임 다시 잡는다(혼자 뜨는 페이지)
    pub fill: bool,
    pub geo: Option<LastGeo>,
    /// 4417 pillWMax: 글자 수 → 본 가장 넓은 알약 폭
    pill_max: RefCell<HashMap<usize, f64>>,
    /// 1초 수급 카드가 마지막으로 그린 것(새 봉 첫 1~2초에 흐리게 남긴다 -- app.js 7180)
    s1_last: Option<Vec<fp::Prim>>,
}

impl Chart {
    pub fn new(feed: feed::Feed, cfg: fp::Cfg, origin: Pos2, bg: Color32) -> Self {
        Self { feed, view: view::View::default(), cfg, origin, bg, fill: false, geo: None, pill_max: Default::default(), s1_last: None }
    }

    /// 1초 수급 카드 한 장(feed.s1_only) -- 상자 = cfg.w_all×h_all(fill 이면 창)
    pub fn ui_s1(&mut self, ui: &mut egui::Ui) {
        if self.fill { let r = ui.ctx().content_rect(); (self.cfg.w_all, self.cfg.h_all) = ((r.width() - 2.0 * self.origin.x) as f64, (r.height() - 2.0 * self.origin.y) as f64); }
        let (w, h) = (self.cfg.w_all, self.cfg.h_all);
        let p = ui.painter();
        if self.bg.a() > 0 { p.rect_filled(p.clip_rect(), 0.0, self.bg); }
        let measure = |s: &str, z: f64, wt: u16| draw::measure(p, s, z, wt);
        let all = [0.0, 0.0, w, h];
        match s1::scene(&self.feed.s1, w, h, self.cfg.tz_off_min, &measure) {
            Some(v) => { draw::paint(p, &v, self.origin, all, all, 1.0); self.s1_last = Some(v); }
            None => match &self.s1_last { Some(v) => draw::paint(p, v, self.origin, all, all, 0.35), None => draw::paint(p, &s1::wait_text(w, h), self.origin, all, all, 1.0) },
        }
    }
    /// 시장 맥락 ①② 한 장(feed.mc_only) -- 폭 = cfg.w_all(fill 이면 창) · 높이 = mc::H
    pub fn ui_mc(&mut self, ui: &mut egui::Ui) {
        if self.fill { self.cfg.w_all = (ui.ctx().content_rect().width() - 2.0 * self.origin.x) as f64; }
        let p = ui.painter();
        if self.bg.a() > 0 { p.rect_filled(p.clip_rect(), 0.0, self.bg); }
        let all = [0.0, 0.0, self.cfg.w_all, mc::H];
        draw::paint(p, &mc::scene(self.feed.mc.as_ref(), &self.feed.asset, self.cfg.w_all), self.origin, all, all, 1.0);
    }
    /// 계좌 칩 한 장(feed.acct_only) -- 폭 = cfg.w_all(fill 이면 창) · 높이 = acct::H · 숫자가 흐르는 동안 계속 다시 그린다
    pub fn ui_acct(&mut self, ui: &mut egui::Ui, now: f64) {
        if self.fill { self.cfg.w_all = (ui.ctx().content_rect().width() - 2.0 * self.origin.x) as f64; }
        let p = ui.painter();
        if self.bg.a() > 0 { p.rect_filled(p.clip_rect(), 0.0, self.bg); }
        let cs = acct::chips(self.feed.acct.as_ref(), &self.feed.prices);
        if acct::paint(p, &cs, self.origin.x, self.origin.y, self.cfg.w_all, now) { ui.ctx().request_repaint(); }
    }
    /// «레벨 5분» 칸 한 장(feed.srl_only · ETH 만 -- 6094) -- 폭 = cfg.w_all(fill 이면 창) · 남은 시간이 판정을 바꾸니 1초마다 다시 그린다
    pub fn ui_srl(&mut self, ui: &mut egui::Ui, now: f64) {
        if self.fill { self.cfg.w_all = (ui.ctx().content_rect().width() - 2.0 * self.origin.x) as f64; }
        let p = ui.painter();
        if self.bg.a() > 0 { p.rect_filled(p.clip_rect(), 0.0, self.bg); }
        if self.feed.asset != "eth" { return; }
        let c = srl::cell(self.feed.srl.as_ref(), self.feed.srl_now(now));
        let measure = |s: &str, z: f64, wt: u16| draw::measure(p, s, z, wt);
        let all = [0.0, 0.0, self.cfg.w_all, srl::height(&c)];
        draw::paint(p, &srl::scene(&c, self.cfg.w_all, &measure), self.origin, all, all, 1.0);
        ui.ctx().request_repaint_after(std::time::Duration::from_secs(1));
    }
    /// «청산» 탭 청산맵 한 장(feed.liq_only) -- svg 상자 = cfg.w_all×h_all(fill 이면 창 − 범례 줄) · 범례는 그 아래
    pub fn ui_liq(&mut self, ui: &mut egui::Ui) {
        if self.fill { let r = ui.ctx().content_rect(); (self.cfg.w_all, self.cfg.h_all) = ((r.width() - 2.0 * self.origin.x) as f64, (r.height() - 2.0 * self.origin.y - 20.0) as f64); }
        let (w, h, f) = (self.cfg.w_all, self.cfg.h_all, &mut self.feed);
        let live = f.prices.get(&f.asset).copied().unwrap_or(0.0);
        let p = liq::price(live, f.liq_tab.as_ref());
        let mp = f.liq_map.as_ref().and_then(|m| liq::live_tp(m, &mut f.liq_sw, live, &f.candles)).and_then(|t| liq::map_of(&t, p));
        let pa = ui.painter();
        if self.bg.a() > 0 { pa.rect_filled(pa.clip_rect(), 0.0, self.bg); }
        let measure = |s: &str, z: f64, wt: u16| draw::measure(pa, s, z, wt);
        let mut v = liq::scene(mp.as_ref(), f.liq_tab.as_ref(), f.liq_days as usize, p, w, h).unwrap_or_else(liq::empty);
        v.extend(liq::legend(h, &measure));
        let all = [0.0, 0.0, w, h + 20.0];
        draw::paint(pa, &v, self.origin, all, all, 1.0);
    }
    /// 합성 자료로 바로(측정·네이티브 창): 풋프린트 응답 + 캔들 이력을 받은 것처럼
    pub fn from_json(fp_json: &str, mh_json: &str, cfg: fp::Cfg, origin: Pos2, bg: Color32) -> Self {
        let mut f = feed::Feed::new("eth", "ETHUSDT", feed::Source::Poll, feed::Urls::default(), false);
        f.on_history(Some(mh_json), 0.0);
        f.on_footprint(Some(fp_json));
        let mut ch = Self::new(f, cfg, origin, bg);
        ch.view.bars = ch.cfg.window;
        ch
    }

    /// 5m·30m·1h 버튼(1985) -- now = Date.now()
    pub fn tf(&mut self, k: i64, now: f64) {
        if self.view.tf_button(k) { self.feed.kick_footprint(now, self.view.want_week()); }
        self.feed.dirty = true; // 창이 바뀌었다 -- 다시 그린다(자료가 그대로여도)
    }

    /// 마우스 → 창 상태(app.js 7935-8016 · 지난 그림의 자리로 잰다). t = 이 프레임 시각(ms)
    fn input(&mut self, ui: &egui::Ui, t: f64) {
        let Some(g) = self.geo else { return };
        let o = self.origin;
        let sv = |p: Pos2| ((p.x - o.x) as f64, (p.y - o.y) as f64); // 화면 px → svg 좌표(데스크톱 배율 1)
        let (wa, ha) = (self.cfg.w_all, self.cfg.h_all);
        let inbox = |p: Pos2| { let (x, y) = sv(p); x >= 0.0 && y >= 0.0 && x <= wa && y <= ha }; // svg 상자 안에서만 이벤트를 받는다
        let plot = |p: Pos2| { let (x, y) = sv(p); inbox(p) && y >= g.mt && y <= g.pb && x <= g.side_x }; // zoneY = "plot" · zoneX ≠ "side"
        let px = g.cw / g.n; // chartSlotPx
        let at = |p: Pos2| ((sv(p).0 - g.ml) / g.cw).clamp(0.0, 1.0);
        let (events, hover, down, dbl) = ui.input(|i| (i.events.clone(), i.pointer.hover_pos(), i.pointer.primary_down(), i.pointer.button_double_clicked(PointerButton::Primary)));
        let full = view::tf_candles(&self.feed.chart_full(), self.view.tf_k);
        for ev in events {
            match ev {
                Event::MouseWheel { unit, delta, modifiers, .. } => {
                    let Some(p) = hover else { continue };
                    if modifiers.ctrl || !inbox(p) || sv(p).0 > g.side_x { continue; } // 7936-7937 (오른쪽 수급 칸은 풋프린트 줌을 안 건드린다)
                    let k = match unit { MouseWheelUnit::Line => 40.0, MouseWheelUnit::Page => 800.0, _ => 1.0 };
                    self.view.wheel(&full, -delta.x as f64 * k, -delta.y as f64 * k, px, at(p));
                }
                Event::PointerButton { pos, button, pressed: true, modifiers, .. } => {
                    self.view.fling = None; // 7958 chartFlingStop(어느 단추든)
                    if button == PointerButton::Primary && !modifiers.shift && plot(pos) { self.view.press(pos.x as f64, pos.y as f64, px); } // Shift = 측정 박스(안 옮김)
                }
                Event::PointerMoved(pos) if self.view.drag.is_some() => {
                    if !down { self.view.drag = None; continue; } // 7993 !(ev.buttons & 1)
                    self.view.drag_to(&full, pos.x as f64, pos.y as f64, t, px);
                }
                Event::PointerButton { button: PointerButton::Primary, pressed: false, .. } => self.view.release(&full, t),
                _ => {}
            }
        }
        // 8012 더블클릭 = 칸 숫자가 들어가는 가장 넓은 창 · 실시간 + 빈 칸 2
        if dbl && hover.is_some_and(plot) { self.view.fit_qty(g.cw); }
    }

    /// 한 장: 입력 → 가공(prep) + 그리기 + 호버. now = Date.now() · 반환 = 그린 장면·호버(좌표 대조용)
    pub fn ui(&mut self, ui: &mut egui::Ui, now: f64) -> Option<(fp::Scene, Option<hover::Hover>)> {
        if self.feed.s1_only { self.ui_s1(ui); return None; }
        if self.feed.mc_only { self.ui_mc(ui); return None; }
        if self.feed.acct_only { self.ui_acct(ui, now); return None; }
        if self.feed.srl_only { self.ui_srl(ui, now); return None; }
        if self.feed.liq_only { self.ui_liq(ui); return None; }
        let t = ui.input(|i| i.time) * 1000.0;
        if self.fill { // 혼자 뜨는 페이지: 창 = 화면 -- 가격 플롯 왼쪽 열(w)이 화면 폭을 채우는 viewBox 폭(8429-8430 을 거꾸로)
            let r = ui.ctx().content_rect();
            let target = r.width() as f64 - 2.0 * self.origin.x as f64;
            let mut wa = target.ceil();
            while wa - 20.0 - (((wa - 20.0) * 0.25).round() + 35.0) < target { wa += 1.0; }
            (self.cfg.w_all, self.cfg.h_all) = (wa, (r.height() - 2.0 * self.origin.y) as f64);
        }
        self.input(ui, t);
        if self.view.fling.is_some() {
            let full = view::tf_candles(&self.feed.chart_full(), self.view.tf_k);
            if self.view.fling_step(&full, t) { ui.ctx().request_repaint(); }
        }
        let want = self.view.want_week();
        self.feed.maybe_week(now, want); // 7287·7339·7302 옮기기·넓히기·더블클릭이 1주일을 부른다
        self.cfg.window = self.view.bars;

        // ── renderSnapshotChart 8245-8258 의 입력 ──
        let k = self.view.tf_k;
        let full = view::tf_candles(&self.feed.chart_full(), k);
        if full.is_empty() { return None; }
        let (bucket, merged) = self.feed.for_chart(want)?; // ponytail: 풋프린트가 없으면(웜업·첫 응답 전) 그리지 않는다 -- 지금 화면은 채운 캔들로 그린다
        let bars = self.feed.bars_now(want);
        let mut by5: BTreeMap<i64, &Vec<Vec<f64>>> = bars.iter().filter(|b| !b.levels.is_empty()).map(|b| (b.time, &b.levels)).collect();
        if let Some((tm, lv)) = &merged { by5.insert(*tm, lv); }
        let tf_cells = if k > 1 { Some(view::tf_levels(&by5, bucket, k)) } else { None };
        let by_time = match &tf_cells { Some(m) => m.iter().map(|(t, v)| (*t, v)).collect(), None => by5 };
        // 8528 레인 = tfFpBars(fpBarsNow()) -- 서버 봉 그대로(형성 봉도 서버 값 · 30m/1h 면 dhl 없음)
        let tf_lane = if k > 1 { Some(view::tf_levels(&bars.iter().map(|b| (b.time, &b.levels)).collect(), bucket, k)) } else { None };
        let mut lane: HashMap<i64, (&Vec<Vec<f64>>, Option<[f64; 3]>)> = HashMap::new();
        match &tf_lane {
            Some(m) => for (t, v) in m { lane.entry(*t).or_insert((v, None)); },
            None => for b in &bars { lane.entry(b.time).or_insert((&b.levels, b.dhl.as_ref().filter(|d| d.len() >= 3).map(|d| [d[0], d[1], d[2]]))); },
        }
        let win = fp::Win { edge: self.view.edge(&full), panned: self.view.panned(), pan_y: self.view.pan_y };
        let last_close = { let n = full.len().min((win.edge - 1e-6).ceil().max(1.0) as usize); full[n - 1].close };
        let or_close = |v: f64| if v > 0.0 { v } else { last_close };
        let pill_w = |tw: f64, len: usize| { let w = ((tw + 18.0) / 4.0).ceil() * 4.0; let mut m = self.pill_max.borrow_mut(); let e = m.entry(len).or_insert(w); *e = e.max(w); *e };
        let liq = view::tf_liq(&self.feed.liq_full(), k); // 8284 tfLiq(weekJoin("liq", latestLiquidation5mHist))
        let inp = fp::Input { full: &full, bucket, by_time, lane, current: or_close(self.feed.price), shown: or_close(self.feed.shown), liq: &liq, pill_w: &pill_w };

        let p = ui.painter();
        if self.bg.a() > 0 { p.rect_filled(p.clip_rect(), 0.0, self.bg); }
        let measure = |s: &str, z: f64, wt: u16| draw::measure(p, s, z, wt);
        let sc = fp::prep(&inp, &win, &self.cfg, &measure)?;
        let g = &sc.g;
        self.view.y_per_px = g.y_span / g.ch.max(1.0); // 8690 chartYPerPx
        self.geo = Some(LastGeo { ml: g.ml, cw: g.cw, n: self.cfg.window.min(full.len() as f64), mt: g.mt, pb: g.plot_bottom, side_x: g.w - g.mr }); // 8589 fpProfileSpan[1]
        // svg.onmousemove(10727) -- 끄는 중에도 호버는 그대로(지금 화면도 다시 그린 뒤 마지막 커서로 재생 8299)
        let mut hv = ui.input(|i| i.pointer.hover_pos()).and_then(|pos| {
            let (mx, my) = ((pos.x - self.origin.x) as f64, (pos.y - self.origin.y) as f64);
            if mx < 0.0 || my < 0.0 || mx > self.cfg.w_all || my > self.cfg.h_all { return None; }
            Some(hover::hover(&sc, mx, my, [pos.x as f64, pos.y as f64], self.cfg.tz_off_min, &measure))
        });
        // 층: 호버 층(10683 hoverGroup)은 프로파일 기둥 뒤 · 레인 글자(11158)와 알약(post 10522) 앞 · 툴팁은 HTML 이라 맨 위
        let k = sc.groups.iter().find(|g| g.0 == "lane_text").map_or(sc.prims.len(), |g| g.1);
        draw::paint_prims(p, &sc.prims[..k], &sc, self.origin);
        if let Some(h) = &hv { draw::paint_prims(p, &h.prims, &sc, self.origin); }
        draw::paint_prims(p, &sc.prims[k..], &sc, self.origin);
        if let Some(t) = hv.as_mut().and_then(|h| h.tip.as_mut()) { let sr = ui.ctx().content_rect(); draw::tip(p, t, sr.width(), sr.height()); }
        Some((sc, hv))
    }
    /// 형성 봉 칸에 체결이 들어온 것처럼 값을 바꾼다(측정용 -- rtui_proto/bench_js.py 의 자바스크립트 측정과 같은 식)
    pub fn nudge(&mut self, f: usize) {
        if let Some((_, b)) = self.feed.fp_bars.iter_mut().next_back() {
            for (j, l) in b.levels.iter_mut().enumerate() {
                l[1] += 0.37 + ((f + j) % 5) as f64 * 0.11;
                l[2] += 0.29 + ((f * 3 + j) % 7) as f64 * 0.07;
            }
        }
    }
    /// 창 상태(비교 하네스가 지금 화면의 전역 변수와 맞댄다)
    pub fn state_json(&self) -> serde_json::Value {
        let v = &self.view;
        serde_json::json!({"bars": v.bars, "tfk": v.tf_k, "end": v.pan_end, "frac": v.pan_frac, "right": v.pan_right, "py": v.pan_y, "ypp": v.y_per_px})
    }
}

pub fn new_ctx() -> Context {
    let ctx = Context::default();
    ctx.set_fonts(draw::fonts());
    draw::canvas_like(&ctx);
    ctx
}

/// 창 없이 egui 한 프레임: UI 구성(가공 포함) + 테셀레이션 -- 웹·네이티브가 같은 코드. 반환 (구성 ms, 테셀레이션 ms)
pub fn bench_frame(ctx: &Context, ch: &mut Chart, now: &dyn Fn() -> f64) -> (f64, f64) {
    let raw = egui::RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1920.0, 1080.0))), ..Default::default() };
    let t0 = now();
    let out = ctx.run_ui(raw, |ui| { ch.ui(ui, 0.0); });
    let t1 = now();
    let _ = ctx.tessellate(out.shapes, out.pixels_per_point);
    (t1 - t0, now() - t1)
}

pub fn stats(v: &[f64]) -> String {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let q = |a: f64| s[((a * s.len() as f64) as usize).min(s.len() - 1)];
    format!("n={} med={:.3} p99={:.3} max={:.3}", s.len(), q(0.5), q(0.99), s[s.len() - 1])
}

/// 좌표 대조용: 그린 목록을 JSON 으로(비교 하네스가 지금 화면의 캔버스 명령·svg 요소와 맞댄다) · g = 묶음 이름(fp.rs Scene::add)
pub fn prims_json(prims: &[fp::Prim], groups: &[(&str, usize, usize)]) -> Vec<serde_json::Value> {
    use fp::Prim::*;
    let grp = |i: usize| groups.iter().find(|g| g.1 <= i && i < g.2).map_or("", |g| g.0);
    prims.iter().enumerate().map(|(i, p)| { let g = grp(i); match p {
        Rect { x, y, w, h, c, a, rx, clip } => serde_json::json!({"g": g, "k": "rect", "x": x, "y": y, "w": w, "h": h, "c": format!("{c:?}"), "a": a, "rx": rx, "clip": format!("{clip:?}")}),
        StrokeRect { x, y, w, h, c, a, lw, .. } => serde_json::json!({"g": g, "k": "srect", "x": x, "y": y, "w": w, "h": h, "c": format!("{c:?}"), "a": a, "lw": lw}),
        Line { pts, c, a, lw, dash, .. } => serde_json::json!({"g": g, "k": "line", "pts": pts, "c": format!("{c:?}"), "a": a, "lw": lw, "dash": dash}),
        Text { x, y, s, size, wt, c, a, anchor, halo, .. } => serde_json::json!({"g": g, "k": "text", "x": x, "y": y, "s": s, "size": size, "wt": wt, "c": format!("{c:?}"), "a": a, "anchor": format!("{anchor:?}"), "halo": halo}),
        Circle { x, y, r, c, a, .. } => serde_json::json!({"g": g, "k": "circle", "x": x, "y": y, "r": r, "c": format!("{c:?}"), "a": a}),
        Dashed { pts, c, a, lw, on, off, .. } => serde_json::json!({"g": g, "k": "line", "pts": pts, "c": format!("{c:?}"), "a": a, "lw": lw, "dash": [on, off]}),
        Area { pts, base, c, a } => serde_json::json!({"g": g, "k": "area", "pts": pts, "base": base, "c": format!("{c:?}"), "a": a}),
        Mono { x, y, s, size, wt, c, a, anchor, halo } => serde_json::json!({"g": g, "k": "text", "mono": true, "x": x, "y": y, "s": s, "size": size, "wt": wt, "c": format!("{c:?}"), "a": a, "anchor": format!("{anchor:?}"), "halo": halo}),
    }}).collect()
}

pub fn scene_json(sc: &fp::Scene, hv: Option<&hover::Hover>) -> String {
    let g = &sc.g;
    serde_json::json!({"geo": {"w": g.w, "mr": g.mr, "ml": g.ml, "cw": g.cw, "mt": g.mt, "ch": g.ch, "pb": g.plot_bottom, "lane_top": g.lane_top, "quad_y": g.quad_y,
        "ymin": g.y_min, "ymax": g.y_max, "slot": g.slot, "bw": g.bw, "row_px": g.row_px, "cp": g.current, "off": g.x0b},
        "t0": sc.candles.first().map(|c| c.time), "t1": sc.candles.last().map(|c| c.time), "nc": sc.candles.len(),
        "prims": prims_json(&sc.prims, &sc.groups),
        "hover": hv.map(|h| serde_json::json!({"prims": prims_json(&h.prims, &[]), "tip": h.tip.as_ref().map(|t| serde_json::json!({
            "x": t.x, "y": t.y, "w": t.w, "h": t.h, "lines": t.lines.iter().map(|l| l.runs.iter().map(|r| r.0.clone()).collect::<String>()).collect::<Vec<_>>()}))}))}).to_string()
}

/// 창 있는 앱(네이티브 glow · 웹 WebGL)
/// last = 마지막으로 그린 장면 JSON(비교 하네스가 읽는다 -- 장면이 바뀐 프레임만 만든다)
pub struct App { pub chart: Chart, pub last: Option<String>, pub now: fn() -> f64 }
impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if let Some((sc, hv)) = self.chart.ui(ui, (self.now)()) { self.last = Some(scene_json(&sc, hv.as_ref())); }
    }
}

pub fn default_cfg(window: f64, w_all: f64, h_all: f64, tz_off_min: i64) -> fp::Cfg {
    fp::Cfg { w_all, h_all, split: 0.25, window, tz_off_min, dp: 1, unit: "ETH".into() }
}

/// app.js 44-53 ASSET_CONFIG(흐름 코인 FLOW_ASSETS 55 만) -- (심볼, dp, 이름)
pub fn asset_cfg(a: &str) -> Option<(&'static str, usize, &'static str)> {
    Some(match a {
        "eth" => ("ETHUSDT", 1, "ETH"), "sol" => ("SOLUSDT", 2, "SOL"), "btc" => ("BTCUSDT", 0, "BTC"), "xrp" => ("XRPUSDT", 4, "XRP"),
        "sndk" => ("SNDKUSDT", 2, "SNDK"), "soxl" => ("SOXLUSDT", 2, "SOXL"), "skhynix" => ("SKHYNIXUSDT", 2, "SKHYNIX"), "mu" => ("MUUSDT", 2, "MU"),
        _ => return None,
    })
}

#[cfg(target_arch = "wasm32")]
mod web;
