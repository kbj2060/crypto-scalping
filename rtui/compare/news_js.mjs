// News 카드 + ⑤ 다음 24시간 정답지: 지금 화면의 실제 코드(app.js 뉴스 구역 = NEWS_* · newsRel · newsTiltHtml · newsAssetCounts · newsAgo · renderNews
// + renderMarketCtx 의 ⑤ 몫 = n · gCur/ef/us · macro24 · events · sec · qSec · whenHtml 줄 + mcGfx.timeline · mcUsSession · optFront · optQ · fmtHourMinute · isTodayOrTomorrowLocal)를
// 가짜 전역에서 돌린 HTML → fixture/news_html.json. 그 HTML 을 compare/news_layout.py 가 원본 styles.css 로 헤드리스 크로미움에 띄워 잰다.
// 기사·숫자는 전부 지어낸 값. 실행: TZ=Asia/Seoul node rtui/compare/news_js.mjs   (바깥 접속 없음) · APP_JS=<app.js 사본> OUT=<html json> 로 변이 시험
import fs from "node:fs";
const here = new URL(".", import.meta.url).pathname;
const src = fs.readFileSync(process.env.APP_JS || here + "../../dashboard/live/app.js", "utf8");
const fn = (name) => { const i = src.indexOf(`\nfunction ${name}(`); if (i < 0) throw name; return src.slice(i, src.indexOf("\n}\n", i) + 3); };
const line = (head, from = 0) => { const i = src.indexOf("\n" + head, from); if (i < 0) throw head; return src.slice(i, src.indexOf("\n", i + 1)); };
const span = (a, b) => { const i = src.indexOf(a), j = src.indexOf(b, i); if (i < 0 || j < 0) throw a; return src.slice(i, j); };
if (process.env.TZ !== "Asia/Seoul") throw new Error("TZ=Asia/Seoul 로 돌린다(시각 = 브라우저 지역 시각 · rtui 는 tz_off_min -540)");
const MC = src.indexOf("\nfunction renderMarketCtx(");
const tl = span("\n  timeline(events, w = 900", "\n  },\n};");
const news = [fn("escapeHtml"), line("const helpAttrs = "), fn("fmtHourMinute"), fn("isTodayOrTomorrowLocal"), fn("mcUsSession"), line("const optQ = "), line("const optFront = "),
  span("\nconst NEWS_POLL_MS", "\nasync function refreshNews"), span("\n// 2026-10-06 뉴스 카드 «기울기»", "\n\n// ── Macro Trend"), fn("newsAgo"), fn("renderNews"), fn("nwWhenFit"),
  `const mcGfx = {${tl}\n  },\n};`].join("\n");
// renderMarketCtx 안 ⑤ 몫(지역 변수 이름 그대로 -- f = d.funding 은 원본 한 줄 안이라 여기서 같은 뜻으로)
const when = ["const A = activeSnapshotAsset, eth = A === \"eth\";", line("  const n = (v, dp = 0)"), line("  const sec = (key, title, inner)", MC), "const f = d.funding || {};",
  span("\n  const gCur = ", "\n  // 2026-09-30 사용자 «주요 경제 일정"), span("\n  const macro24 = ", "\n  const sw = bk.sweep"), "const G = mcGfx;", line("  const qSec = "),
  span("\n  const events = [", "\n  let htmlB = ["), line("  const whenBox = ", MC), line("  const whenW = ", MC), line("  const whenHtml = ", MC)].join("\n");

let seed = 11;
const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const r2 = (v) => Math.round(v * 100) / 100;
const NA = 1795000000e3;   // 수 11-18 20:06:40 KST(미 동부 06:06 -- 미국장 개장 전)
const NB = 1795038400e3;   // 목 11-19 06:46:40 KST(미 동부 16:46 전날 -- 장 끝난 뒤)
const SRC = ["theblock", "cointelegraph", "coindesk", "tree", "bwenews", "financialjuice", "trumpstruth", "fed", "sec"];
const AS = ["ETH", "BTC", "SOL", "XRP", "HYPE", "macro", "other", "ETH", "BTC", "macro"];
const SEN = ["bullish", "bearish", "neutral", "bullish", null, "bearish", "neutral", "bullish"];
const WORDS = ["이더리움", "현물 ETF", "자금 유입", "연준 의장", "발언", "SEC", "승인", "지연", "해킹", "거래소", "출금 중단", "고래 지갑", "대량 이동", "금리", "인하 기대", "Ethereum", "staking", "upgrade", "Binance", "liquidations", "spike", "&", "<b>굵게</b>", "\"인용\"", "'작은'"];
const mk = (now, k, o = {}) => {
  const sen = o.sen !== undefined ? o.sen : SEN[k % SEN.length];
  const srcs = k % 7 === 3 ? [SRC[k % 9], SRC[(k + 4) % 9]] : k % 11 === 5 ? [SRC[k % 9], "x<y>", SRC[(k + 2) % 9]] : null;
  const p = [r2(rnd()), r2(rnd()), r2(rnd())];
  const it = { ts_ms: now - Math.round((o.ago ?? (k * 9.7 + rnd() * 5)) * 60e3), title: o.title ?? WORDS.filter((_, i) => (i + k) % (3 + (k % 4)) === 0).join(" "),
    link: k % 9 === 4 ? "javascript:alert(1)" : `https://news.example/${k}?a=1&b=2`, source: srcs ? srcs[0] : (k % 13 === 6 ? "newsrc" : SRC[k % 9]), asset: o.asset ?? AS[k % AS.length],
    sentiment: sen, p_bull: p[0], p_bear: p[1], p_neu: p[2], impact: o.impact ?? r2(rnd() * 3), relevant: o.rel !== undefined ? o.rel : (k % 6 === 2 ? 0.3 : r2(0.5 + rnd() / 2)),
    incident: o.inc ?? (k % 17 === 8 ? 0.9 : r2(rnd() * 0.5)), model: sen ? "jevk5:4b" : (k % 2 ? "jevk5:4b" : null) };
  if (srcs) it.sources = srcs;
  if (o.norel) delete it.relevant;
  return it;
};
const full = (now, n = 70) => {
  const items = Array.from({ length: n }, (_, k) => mk(now, k));
  items[0].title = "  여러   빈칸과\n줄바꿈 <i>태그</i>가 섞인   제목 ";
  items[2].incident = 0.95; items[2].relevant = 0.9; items[2].sentiment = "bearish";
  items[3].sentiment = "mystery"; items[3].relevant = 0.8;
  items[5].impact = 2.5;   // 막대 셋째 칸 경계(≥ 2.5)
  items[6].impact = 1.49;
  Object.assign(items[1], { impact: 2.95, relevant: 1, sentiment: "bullish", asset: "ETH" });   // 긴 제목이 고영향 상위 1등(두 줄 자름)
  items.forEach((it, k) => { if (k % 5 === 1) it.title = `[긴 제목 ${k}] ` + Array.from({ length: k % 3 ? 14 : 60 }, (_, i) => WORDS[(k + i * 5) % WORDS.length]).join(" "); });
  items[1].title += " — 세 줄 넘게 이어지는 고영향 제목은 두 줄에서 자른다 " + WORDS.join(" ");
  items.sort((a, b) => b.ts_ms - a.ts_ms);
  return { ok: true, now_ms: now, window_h: 12, items };
};
const small = (now) => ({ ok: true, now_ms: now, window_h: 12, items: [mk(now, 4, { sen: null }), mk(now, 6, { rel: 0.2 }), mk(now, 7, { sen: null, ago: 30 })].map((it, i) => (i === 0 ? { ...it, model: null } : it)) });
const ev = (t, nm, imp = "high", cat = "") => ({ time_utc: new Date(t).toISOString().replace(".000Z", "+00:00"), importance: imp, category: cat, title_ko: nm, title: "x" });
const evs = (now) => [ev(now + 2 * 3600e3, "미국 CPI"), ev(now + 2 * 3600e3 + 20e3, "근원 CPI", "medium"), ev(now + 0.4 * 3600e3, "주택 착공", "low"), ev(now + 3.4 * 3600e3, "FOMC 의사록", "medium", "fomc"),
  ev(now + 3.6 * 3600e3, "실업수당 청구"), ev(now + 3.7 * 3600e3, "필라델피아 연은"), ev(now + 10 * 3600e3, "PMI 속보치"), ev(now + 17.9 * 3600e3, "GDP 잠정치"), ev(now + 26.5 * 3600e3, "내일 늦은 발표"),
  ev(now + 31 * 3600e3, "모레 발표"), ev(now - 3600e3, "지난 발표"), { time_utc: new Date(now + 5 * 3600e3).toISOString(), importance: "high", title: "Unemployment" }];
const mc = (now, o = {}) => ({ asset: o.asset || "eth", available: o.off ? false : true, funding: o.nofund ? {} : { next_ms: now + 1.9 * 3600e3 } });
const gex = (now, cur = "ETH", pain = 3200) => ({ available: true, currencies: { [cur]: { options: { expiries: [{ exp_ms: now - 3600e3, pain: 1 }, { exp_ms: now + 36 * 3600e3, pain }, { exp_ms: now + 200 * 3600e3, pain: 9 }] } } } });
const W = 1838, H = 917;   // #newsBody 1920 한 화면(html.fit1) -- news_layout.py 가 실제 카드 구조로 재서 확인한다
const inputs = [
  { now: NA, asset: "eth", w: W, h: H, view: { relevant: true, coin: false }, news: null, mc: null, gex: null, events: null },
  { now: NA, asset: "eth", w: W, h: H, view: { relevant: true, coin: false }, news: { ok: false }, mc: mc(NA), gex: gex(NA), events: evs(NA) },
  { now: NA, asset: "eth", w: W, h: H, view: { relevant: true, coin: false }, news: full(NA), mc: mc(NA), gex: gex(NA), events: evs(NA) },
  { now: NA, asset: "eth", w: W, h: H, view: { relevant: false, coin: true }, news: full(NA), mc: mc(NA, { nofund: true }), gex: null, events: [] },
  { now: NB, asset: "xrp", w: W, h: H, view: { relevant: true, coin: true }, news: full(NB, 30), mc: mc(NB, { asset: "xrp" }), gex: gex(NB, "XRP", 1.3), events: evs(NB) },
  { now: NA, asset: "eth", w: W, h: H, view: { relevant: true, coin: false }, news: small(NA), mc: mc(NA, { off: true }), gex: gex(NA), events: evs(NA) },
  { now: NA, asset: "sol", w: W, h: H, view: { relevant: true, coin: true }, news: full(NA, 40), mc: mc(NA, { asset: "sol" }), gex: gex(NA, "SOL", 140), events: evs(NA) },
  { now: NA, asset: "eth", w: W, h: H, view: { relevant: false, coin: false }, news: full(NA, 12), mc: mc(NA), gex: gex(NA), events: evs(NA).slice(0, 2) },
];
fs.writeFileSync(here + "fixture/news_in.json", JSON.stringify(inputs) + "\n");
const MAINW = 1426;   // .nw-main 폭(1838 − 380 − 32) -- news_layout.py 가 잰 값과 맞대 본다
const run = (s) => {
  const now = s.now;
  class D extends Date { constructor(...a) { super(...(a.length ? a : [now])); } static now() { return now; } }
  const H = {};
  const main = { clientWidth: MAINW, appendChild() {}, classList: { contains: () => true }, style: { setProperty() {} } };
  const body = { contains: () => false, querySelector: (q) => (q === ".nw-main" ? main : null) };
  const whenBox = { clientWidth: MAINW, parentElement: main, offsetHeight: 0 };
  const card = { hidden: false, clientWidth: 1872, appendChild() {} };
  const env = { Date: D, document: { querySelector: () => null, activeElement: null }, helpKey: null, activeSnapshotAsset: s.asset, latestNews: s.news,
    el: (id) => ({ newsCard: card, newsBody: body, mcWhen: whenBox })[id] || null, setH: (id, h) => { H[id] = h; }, getComputedStyle: () => ({ marginTop: "0" }), __S: s, __H: H };
  const px = new Proxy(env, { has: (t, k) => k !== Symbol.unscopables, get: (t, k) => (k in t ? t[k] : globalThis[k]), set: (t, k, v) => { t[k] = v; return true; } });
  return new Function("px", `with (px) { ${news}\nlatestNews = __S.news; Object.assign(newsView, __S.view);\nrenderNews();\n`
    + `let whenHtml = "";\nconst d = __S.mc, latestGex = __S.gex, latestMacroEvents = __S.events;\nif (d && d.available && ["eth", "btc", "sol", "xrp"].includes(activeSnapshotAsset)) { ${when}\n setH("mcWhen", whenHtml); }\n`
    + `return [el("newsCard").hidden, __H.newsBody || "", __H.mcWhen || ""]; }`)(px);
};
const res = inputs.map(run);
fs.writeFileSync(process.env.OUT || here + "fixture/news_html.json", JSON.stringify(res) + "\n");
for (const [hid, b, w] of res) console.log(`숨김 ${hid} · 본문 ${b.length}자 · ⑤ ${w.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").slice(0, 110)}`);
