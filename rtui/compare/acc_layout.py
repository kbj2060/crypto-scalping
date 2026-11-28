"""Account 카드 정답지(?card=acc): 지금 화면의 실제 코드(app.js 계좌 구역 = acctPools · acctWealthHtml · ACCT_* · acctTabApply · acctTodaySvg · acctCalendar · acctCurveSvg ·
acctStatGroups · acctStatsHtml · renderSnapshotAccount(Core) · cycleSvg · renderCycleCard)를 가짜 전역에서 그대로 돌린다 -- 진짜 #acctCard 틀(index.html 그대로) · 원본 styles.css ·
헤드리스 크로미움 1920×1080(html.fit1). 순환 칸은 원본처럼 그린 뒤 잰 나머지(cyRest)로 한 번 더 그린다(같은 DOM).
계좌 값은 전부 지어낸 값(아래 LCG) -- 실제 /api/binance-account 응답은 받지도 쓰지도 않는다. 입력 → fixture/acc_in.json · 잰 값 → fixture/acc_js.json.
기준 = 카드 머리(.ops-health-head) 아래 왼쪽 · 머리·메모 칸(.acct-note)·오답노트 칸은 안 잰다(옮기지 않음).
잰 것: 글(글자별 폭 · 자르기 상자) · 칠 · 테 · 윤곽(outline) · 고리(inset box-shadow) · ::before 점(.cy-fills) · ::after 가운데 선(.as-div) · svg 자식 전부(속성 · 계산된 색 · tspan).
네트워크 없음. 실행: unshare -rn bash -c 'ip link set lo up; python3 -I rtui/compare/acc_layout.py <worktree>' · APP_JS·STYLES_CSS·ACC_OUT = 변이 시험"""
import datetime, json, os, pathlib, sys
from playwright.sync_api import sync_playwright

WT = pathlib.Path(sys.argv[1]).resolve()
FX, FONTS = WT / "rtui/compare/fixture", WT / "rtui/fonts"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from harness import FONT_CSS, MONO_CSS, LAUNCH

src = pathlib.Path(os.environ.get("APP_JS", WT / "dashboard/live/app.js")).read_text()
def fn(name):
    i = src.index(f"\nfunction {name}(")
    return src[i:src.index("\n}\n", i) + 3]
def line(head):
    i = src.index("\n" + head)
    return src[i:src.index("\n", i + 1)]
def span(a, b):
    i = src.index(a)
    return src[i:src.index(b, i)]
BODY = "\n".join([line("const setH = "), fn("escapeHtml"), fn("fmtUsd"), fn("acctPools"), fn("acctWealthHtml"),
    span("\nconst ACCT_SPANS", "\nfunction acctTripTip("), fn("acctTripTip"), line("let acctTab = "), fn("acctTabApply"), fn("acctTodaySvg"),
    span("\nconst acctIncTip", "\nfunction acctCalendar("), fn("acctCalendar"), fn("acctCurveSvg"), fn("acctStatGroups"), fn("acctStatsHtml"),
    fn("renderSnapshotAccount"), fn("renderSnapshotAccountCore"), span("\nconst cyclePaths = {}", "\ndocument.addEventListener("), line("let cyRest = "), fn("renderCycleCard")])
html = pathlib.Path(WT / "dashboard/live/index.html").read_text()
i = html.index('<section id="acctCard"')
CARD = html[i:html.index("</section>", i) + 10].replace('class="panel ops-health-panel acct-panel"', 'class="panel ops-health-panel acct-panel" style="width:1872px"')

# ── 지어낸 입력 ──
seed = 7
def rnd():
    global seed
    seed = (seed * 1103515245 + 12345) % 2147483648
    return seed / 2147483648
r2 = lambda v: round(v * 100) / 100
NOW = 1795000000000          # 수 11-18 20:06:40 KST
D, H, M = 86400000, 3600000, 60000
T0 = 1794927600000            # 11-18 00:00 KST
SYMS = [("ETHUSDC", 3150.0, 0.4), ("SOLUSDT", 141.0, 9.0), ("XRPUSDC", 1.3007, 800.0), ("BTCUSDT", 91800.0, 0.02)]
def ledger(n, days, syms=SYMS):
    out = []
    for k in range(n):
        sym, px, q = syms[k % len(syms)]
        ex = NOW - int(rnd() * days * D) if k % 9 else T0 + int((k % 4 + 1) * 2.3 * H)
        en = ex - int((0.05 + rnd() * 30) * H) if k % 5 else ex - int(rnd() * 50 * M)
        pnl = r2((rnd() - 0.42) * 70) if k % 13 else r2(-120 - rnd() * 150)
        if k % 17 == 3: pnl = r2(180 + rnd() * 90)
        if k % 23 == 7: pnl = r2(rnd() * 5 - 2.5)
        if k % 29 == 11: pnl = 0.0
        out.append({"symbol": sym, "side": "LONG" if k % 3 else "SHORT", "entry_time": en, "exit_time": ex if k % 31 != 30 else 0, "entry_price": r2(px * (1 + (rnd() - 0.5) * 0.05)) if px > 10 else round(px * (1 + (rnd() - 0.5) * 0.05), 4),
                    "exit_price": r2(px * (1 + (rnd() - 0.5) * 0.05)) if px > 10 else round(px * (1 + (rnd() - 0.5) * 0.05), 4), "max_qty": round(q * (0.5 + rnd() * 2), 5), "net_pnl": pnl,
                    "adds": k % 4, "peak_x": [2, 5, 7, 9][k % 4], "trend": [1, -1, None][k % 3]})
    return out
def legs(t0, px, n, long, maker_every=2):
    v, p = [], px
    for j in range(n):
        p = r2(p * (1 + (rnd() - 0.5) * 0.012)) if px > 10 else round(p * (1 + (rnd() - 0.5) * 0.012), 4)
        inc = 1 if j < 2 or j % 3 != 2 else 0
        v.append([t0 + j * int(19 * M + rnd() * 25 * M), p, round(0.1 + rnd() * 0.5, 4) if px > 1000 else round(1 + rnd() * 6, 3), inc, 1 if j % maker_every == 0 else 0, 0 if inc else r2((rnd() - 0.4) * 30), r2(0.05 + rnd() * 0.6)])
    return v
def path(start, px, step=5):
    rows, p, t = [], px, start
    while t < NOW:
        p = r2(p * (1 + (rnd() - 0.5) * 0.004)) if px > 10 else round(p * (1 + (rnd() - 0.5) * 0.004), 4)
        rows.append([t, p]); t += step * M
    return rows
def income(days):
    out = {}
    for k in range(days):
        d = datetime.datetime.fromtimestamp((T0 - k * D + 9 * H) / 1000, datetime.timezone.utc).strftime("%Y-%m-%d")
        if k % 6 == 4: continue
        out[d] = {"pnl": r2((rnd() - 0.45) * 90), "transfer": 500.0 if k == 12 else (-200.0 if k == 30 else 0), "realized": r2((rnd() - 0.45) * 90) if k % 5 else 0,
                  "funding": r2((rnd() - 0.5) * 2), "commission": r2(-rnd() * 3), "other": 0.0, **({"other_assets": {"BNB": 0.0012}} if k == 3 else {})}
    return out

L1 = ledger(70, 48)
eth_legs = legs(NOW - 7 * H - 13 * M, 3150.0, 6, True)
sol_legs = legs(NOW - 26 * H, 141.3, 4, False, 3)
many = legs(NOW - 30 * H, 3150.0, 34, True)
pos_eth = {"symbol": "ETHUSDC", "side": "LONG", "qty": 0.85, "entry_price": 3150.2, "mark_price": 3177.7, "unrealized_pnl": 23.4, "notional": 2701.0, "leverage": 5}
pos_sol = {"symbol": "SOLUSDT", "side": "SHORT", "qty": 12.0, "entry_price": 141.3, "mark_price": 142.275, "unrealized_pnl": -11.7, "notional": 1707.3, "leverage": 3}
two = {"ok": True, "balance": {"wallet": 1880.5, "unrealized": 23.4, "margin": 1903.9, "available": 1200.3},
       "balances": {"USDC": {"wallet": 1880.5, "unrealized": 23.4, "margin": 1903.9, "available": 1200.3, "initial_margin": 703.6},
                    "USDT": {"wallet": 640.2, "unrealized": -11.7, "margin": 628.5, "available": 400.0, "initial_margin": 228.5}},
       "positions": [pos_eth, pos_sol, {"symbol": "XRPUSDC", "side": "LONG", "qty": 0, "unrealized_pnl": 0, "mark_price": 1.3, "entry_price": 0, "notional": 0}],
       "trades": [{"symbol": "ETHUSDC", "side": "LONG", "closed": False, "entry_time": eth_legs[0][0], "legs": eth_legs},
                  {"symbol": "ETHUSDC", "side": "LONG", "closed": False, "entry_time": eth_legs[0][0] + 9 * M, "truncated_open": True, "legs": eth_legs[2:]},
                  {"symbol": "SOLUSDT", "side": "SHORT", "closed": False, "entry_time": sol_legs[0][0], "legs": sol_legs},
                  {"symbol": "BTCUSDT", "side": "LONG", "closed": True, "entry_time": NOW - 3 * D, "legs": legs(NOW - 3 * D, 91800.0, 2, True)}],
       "ledger": L1, "unprotected": ["SOLUSDT:SHORT"], "income_days": income(40), "income_from": "2026-10-10",
       "day_start": {"date": "2026-11-18", "late_s": 130, "equity": 2510.2, "wallet": 2500.1, "pools": [{}, {}]}, "trades_truncated": ["XRPUSDC"]}
paths = {f"ETHUSDC|LONG|{eth_legs[0][0]}": path(eth_legs[0][0] - 20 * M, 3150.0), f"SOLUSDT|SHORT|{sol_legs[0][0]}": path(sol_legs[0][0] - 20 * M, 141.3, 15)}
uni = {"ok": True, "multi_assets_margin": True, "balance": {"wallet": 3000.0, "unrealized": 0, "margin": 3000.0, "available": 3000.0}, "positions": [], "trades": [],
       "ledger": ledger(14, 6, [("ETHUSDT", 3150.0, 0.4), ("SOLUSDT", 141.0, 9.0)])}
cut_legs = legs(NOW - 50 * H, 1.3007, 3, True)
one = {"ok": True, "balance": {"wallet": 902.0, "unrealized": -3.21, "margin": 898.79, "available": 650.0}, "balances": {"USDC": {"wallet": 902.0, "unrealized": -3.21, "margin": 898.79, "available": 650.0}},
       "positions": [{"symbol": "XRPUSDC", "side": "LONG", "qty": 420.0, "entry_price": 1.3102, "mark_price": 1.30255, "unrealized_pnl": -3.21, "notional": 547.1, "leverage": 10}],
       "trades": [{"symbol": "XRPUSDC", "side": "LONG", "closed": False, "entry_time": cut_legs[0][0], "truncated_open": True, "legs": cut_legs}], "ledger": [], "trades_failed": ["ETHUSDC"], "unprotected": ["XRPUSDC:LONG"]}
big = dict(two, trades=[{"symbol": "ETHUSDC", "side": "LONG", "closed": False, "entry_time": many[0][0], "legs": many}], positions=[pos_eth], unprotected=[], trades_truncated=[], trades_failed=["BTCUSDT", "SOLUSDT"])
paths_big = {f"ETHUSDC|LONG|{many[0][0]}": path(many[0][0] - 20 * M, 3150.0, 10)}
def case(acct, tab="", span=2, back=0, sel=None, cytab="", p=None, err=""):
    return {"now": NOW, "acct": acct, "tab": tab, "span": span, "back": back, "sel": sel, "cytab": cytab, "paths": p or {}, "err": err}
inputs = [
    case(two, "cy", 2, p=paths),
    case(two, "st", 1, p=paths),
    case(two, "cal", 3, sel="2026-11-12", p=paths),
    case(two, "cy", 0, cytab="SOLUSDT|SHORT", p=paths),
    case(uni, "", 0),
    case(None, err="HTTP 502"),
    case(one, "cal", 3),
    case(one, "", 2),
    case(big, "cy", 3, p=paths_big),
    case(two, "cal", 2, back=1, p=paths),
    case(uni, "cal", 1, back=2),
    case(uni, "st", 3),
    case(uni, "cy", 0),
    case(one, "st", 1),
    case(two, "st", 2, p=paths),
]
(FX / "acc_in.json").write_text(json.dumps(inputs, ensure_ascii=False) + "\n")

RUN = r"""([body, s]) => {
  const now = s.now;
  class D extends Date { constructor(...a) { super(...(a.length ? a : [now])); } static now() { return now; } }
  const env = { Date: D, el: (id) => document.getElementById(id), setT: (id, t) => { const e = document.getElementById(id); if (e) e.textContent = t; },
    syncOrderCoinGate() {}, snapshotAccountPosition: () => null, acctNoteSync() {}, acctNotesLoad() {}, acctWrongLoad() {}, fetch: () => new Promise(() => {}),
    latestBinanceAccount: s.acct, staleAccount: null, lastAccountError: s.err, __S: s };
  const px = new Proxy(env, { has: (t, k) => k !== Symbol.unscopables, get: (t, k) => (k in t ? t[k] : globalThis[k]), set: (t, k, v) => { t[k] = v; return true; } });
  new Function("px", `with (px) { ${body}
    acctSpanIdx = __S.span; acctCalBack = __S.back; acctSelDay = __S.sel; acctTab = __S.tab; cycleTab = __S.cytab; acctNotes = {};
    for (const [k, rows] of Object.entries(__S.paths)) cyclePaths[k] = { at: now, rows };
    renderSnapshotAccount(); }`)(px);
}"""

DUMP = r"""() => {
  const root = document.getElementById("acctCard"), head = root.querySelector(".ops-health-head"), cr = root.getBoundingClientRect();
  const o = { left: cr.left, top: head.getBoundingClientRect().bottom, width: cr.width, height: cr.bottom - head.getBoundingClientRect().bottom };
  const q = (v) => Math.round(v * 1000) / 1000;
  const R = (r) => [r.left - o.left, r.top - o.top, r.width, r.height].map(q);
  const col = (s) => {
    if (s === "none" || !s) return null;
    let m = /^rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?\)$/.exec(s);
    if (m) return [+m[1], +m[2], +m[3], m[4] == null ? 1 : +m[4]];
    m = /^color\(srgb ([\d.e-]+) ([\d.e-]+) ([\d.e-]+)(?: \/ ([\d.]+))?\)$/.exec(s);
    if (m) return [m[1] * 255, m[2] * 255, m[3] * 255, m[4] == null ? 1 : +m[4]].map((v, i) => (i < 3 ? Math.round(v * 100) / 100 : v));
    throw s;
  };
  const skip = (e) => !!e.closest(".ops-health-head, .acct-note, #acctNoteWrong");
  const op = (e) => { let a = 1; for (let p = e; p && p !== root.parentElement; p = p.parentElement) a *= +getComputedStyle(p).opacity; return a; };
  const ca = (c, a) => (c ? [c[0], c[1], c[2], q(c[3] * a)] : null);
  const clipOf = (e) => {
    let c = null;
    for (let p = e.parentElement; p && p !== root; p = p.parentElement) {
      const cs = getComputedStyle(p);
      if (cs.overflowX !== "visible" || cs.overflowY !== "visible") { const r = p.getBoundingClientRect(); const v = [r.left + p.clientLeft, r.top + p.clientTop, r.left + p.clientLeft + p.clientWidth, r.top + p.clientTop + p.clientHeight];
        c = c ? [Math.max(c[0], v[0]), Math.max(c[1], v[1]), Math.min(c[2], v[2]), Math.min(c[3], v[3])] : v; }
    }
    return c && [c[0] - o.left, c[1] - o.top, c[2] - c[0], c[3] - c[1]].map(q);
  };
  const texts = [], boxes = [], lines = [], rings = [], svgs = [];
  const out = (r, k) => !!k && (r[0] >= k[0] + k[2] || r[1] >= k[1] + k[3] || r[0] + r[2] <= k[0] || r[1] + r[3] <= k[1]);   // 자르기 상자 밖(넘쳐 숨은 체결 칩 둘째 줄 등) = 안 보임
  const tw = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let n; (n = tw.nextNode());) {
    if (!n.textContent.trim()) continue;
    const e = n.parentElement;
    if (e.closest("svg") || skip(e)) continue;
    const cs = getComputedStyle(e), rg = document.createRange();
    rg.selectNodeContents(n);
    let rs = [...rg.getClientRects()].filter((r) => r.width > 0);
    if (!rs.length) continue;
    const base = { size: parseFloat(cs.fontSize), wt: +cs.fontWeight, mono: cs.fontFamily.startsWith('"JetBrains'), tnum: cs.fontVariantNumeric.includes("tabular"),
      ls: cs.letterSpacing === "normal" ? 0 : parseFloat(cs.letterSpacing), c: ca(col(cs.color), op(e)), cls: typeof e.className === "string" ? e.className : "", k: clipOf(e.firstChild || e) };
    const t = n.textContent, ch = [];
    for (let i = 0; i < t.length; i++) { const r1 = document.createRange(); r1.setStart(n, i); r1.setEnd(n, i + 1); const b = [...r1.getClientRects()].filter((r) => r.width > 0 || t[i] === " ")[0]; ch.push(b ? [t[i], b.left, b.top, b.width] : [t[i], null, null, 0]); }
    const ell = e.scrollWidth > e.clientWidth && cs.textOverflow === "ellipsis";
    if (ell) {   // 말줄임 한 줄: 보이는 조각(좁은 상자)만 + 끝 «…»
      const keep = [];
      for (const r of rs) { const j = keep.findIndex((x) => Math.abs(x.top - r.top) < 1); if (j < 0) keep.push(r); else { if (r.width < keep[j].width) keep[j] = r; keep[j].cut = true; } }
      rs = keep.map((r) => Object.assign({ left: r.left, top: r.top, right: r.right, width: r.width, height: r.height }, { cut: !!r.cut }));
    }
    rs.forEach((r) => {
      const mine = ch.filter((c) => c[2] != null && Math.abs(c[2] - r.top) < 1 && c[1] >= r.left - 0.5 && (r.cut ? c[1] + c[3] <= r.right + 0.5 : c[1] < r.right + 0.5));
      const item = { ...base, s: mine.map((c) => c[0]).join(""), r: R(r) };
      if (out(item.r, base.k)) return;
      item.ch = (r.cut ? ch.filter((c) => c[2] != null) : mine).map((c) => [c[0], q(c[3])]);
      texts.push(item);
      if (r.cut) texts.push({ ...base, s: "…", r: R({ left: r.right, top: r.top, width: 0, height: r.height }), synth: true });
    });
  }
  for (const e of root.querySelectorAll("*")) {
    if (skip(e) || (e.closest("svg") && e.tagName !== "svg")) continue;
    const cs = getComputedStyle(e), r = e.getBoundingClientRect();
    if (!r.width && !r.height) continue;
    if (cs.display === "contents" || cs.visibility === "hidden") continue;
    const a = op(e), k = clipOf(e), cls = typeof e.className === "string" ? e.className : (e.className.baseVal || "");
    const bg = col(cs.backgroundColor);
    if (bg && bg[3] > 0) boxes.push({ r: R(r), c: ca(bg, a), rad: cs.borderTopLeftRadius, cls, k });
    const sides = ["Top", "Right", "Bottom", "Left"].map((s) => [s, parseFloat(cs[`border${s}Width`]), cs[`border${s}Style`], cs[`border${s}Color`]]).filter((x) => x[1] > 0 && x[2] !== "none");
    if (sides.length === 4 && sides.every((x) => x[3] === sides[0][3] && x[1] === sides[0][1])) { const c = col(sides[0][3]); if (c[3] > 0) lines.push({ side: "all", r: R(r), w: sides[0][1], c: ca(c, a), rad: cs.borderTopLeftRadius, style: sides[0][2], cls, k }); }
    else for (const [s, w, st, c] of sides) {
      const rr = s === "Top" ? { left: r.left, top: r.top, width: r.width, height: w } : s === "Bottom" ? { left: r.left, top: r.bottom - w, width: r.width, height: w }
        : s === "Left" ? { left: r.left, top: r.top, width: w, height: r.height } : { left: r.right - w, top: r.top, width: w, height: r.height };
      if (col(c)[3] > 0) lines.push({ side: s, r: R(rr), w, c: ca(col(c), a), style: st, cls, k });
    }
    if (cs.outlineStyle !== "none" && parseFloat(cs.outlineWidth) > 0) { const w = parseFloat(cs.outlineWidth), f = parseFloat(cs.outlineOffset) + w;
      lines.push({ side: "all", r: R({ left: r.left - f, top: r.top - f, width: r.width + 2 * f, height: r.height + 2 * f }), w, c: ca(col(cs.outlineColor), a), rad: cs.borderTopLeftRadius, style: "outline", cls, k }); }
    if (cs.boxShadow !== "none") for (const m of cs.boxShadow.matchAll(/((?:rgba?|color)\([^)]*\))\s+(-?[\d.]+)px\s+(-?[\d.]+)px\s+([\d.]+)px\s+(-?[\d.]+)px(\s+inset)?/g))
      rings.push({ r: R(r), c: ca(col(m[1]), a), dx: +m[2], dy: +m[3], blur: +m[4], spread: +m[5], inset: !!m[6], rad: cs.borderTopLeftRadius, cls, k });
    for (const ps of ["::before", "::after"]) {   // .cy-fills span::before 점 · .as-div::after 가운데 선
      const p = getComputedStyle(e, ps);
      if (p.content === "none" || p.content === "normal" || p.display === "none") continue;
      const pc = col(p.backgroundColor);
      if (!pc || !(pc[3] > 0)) continue;
      let pr;
      if (p.position === "absolute") { const lft = parseFloat(p.left), top = parseFloat(p.top), w = parseFloat(p.width), h = r.height - parseFloat(p.top) - parseFloat(p.bottom);
        pr = { left: r.left + e.clientLeft + (p.left.endsWith("%") ? e.clientWidth * lft / 100 : lft), top: r.top + e.clientTop + top, width: w, height: e.clientHeight - top - parseFloat(p.bottom) }; }
      else {   // inline-block 점: 첫 글자 줄 앞 -- 자리는 span 안쪽 왼쪽 + 그 줄 바닥선 위 vertical-align
        const rg = document.createRange(); rg.selectNodeContents(e); const fr = [...rg.getClientRects()][0];
        pr = { left: r.left + parseFloat(cs.paddingLeft) + e.clientLeft, top: null, width: parseFloat(p.width), height: parseFloat(p.height) };
        // 점 자리 = 잰다(같은 모양 span 을 잠깐 끼운다)
        const d = document.createElement("i"); d.style.cssText = `display:inline-block;width:${p.width};height:${p.height};margin-right:${p.marginRight};vertical-align:${p.verticalAlign};font:inherit`;
        e.classList.add("__nb"); e.insertBefore(d, e.firstChild); const dr = d.getBoundingClientRect(); pr.left = dr.left; pr.top = dr.top; d.remove(); e.classList.remove("__nb");
      }
      boxes.push({ r: R(pr), c: ca(pc, a), rad: p.borderTopLeftRadius, cls: cls + ps, k: p.position === "absolute" ? clipOf(e.firstChild || e) && (() => { const c = clipOf(e); const er = e.getBoundingClientRect(); const v = [er.left - o.left, er.top - o.top, er.width, er.height]; if (!c) return v.map(q); const x0 = Math.max(c[0], v[0]), y0 = Math.max(c[1], v[1]); return [x0, y0, Math.min(c[0] + c[2], v[0] + v[2]) - x0, Math.min(c[1] + c[3], v[1] + v[3]) - y0].map(q); })() : k });
    }
    if (e.tagName === "svg") {
      const vb = e.viewBox.baseVal, els = [];
      for (const c of e.children) {
        if (c.tagName === "title") continue;
        const s = getComputedStyle(c), at = {};
        for (const x of c.attributes) at[x.name] = x.value;
        const it = { tag: c.tagName, at, stroke: col(s.stroke), fill: col(s.fill), so: +s.strokeOpacity, fo: +s.fillOpacity, op: +s.opacity, sw: parseFloat(s.strokeWidth), dash: s.strokeDasharray };
        if (c.tagName === "text") {
          Object.assign(it, { s: [...c.childNodes].filter((n) => n.nodeType === 3).map((n) => n.textContent).join(""), size: parseFloat(s.fontSize), wt: +s.fontWeight, anchor: s.textAnchor, mono: s.fontFamily.startsWith('"JetBrains'), po: s.paintOrder });
          it.ts = [...c.querySelectorAll("tspan")].map((t) => { const u = getComputedStyle(t); return { s: t.textContent, wt: +u.fontWeight, fill: col(u.fill), fo: +u.fillOpacity }; });
        }
        els.push(it);
      }
      svgs.push({ r: R(r), vb: [vb.x, vb.y, vb.width, vb.height], cls, a, k, els });
    }
  }
  const vis = (x) => !out(x.r, x.k);
  return { w: q(o.width), h: q(o.height), texts, boxes: boxes.filter(vis), lines: lines.filter(vis), rings: rings.filter(vis), svgs: svgs.filter(vis) };
}"""

css = pathlib.Path(os.environ.get("STYLES_CSS", WT / "dashboard/live/styles.css")).read_text()
out = []
with sync_playwright() as p:
    b = p.chromium.launch(args=LAUNCH)
    try:
        pg = b.new_page(viewport={"width": 1920, "height": 1080}, timezone_id="Asia/Seoul")
        def route(r):
            n = r.request.url.rsplit("/", 1)[-1]
            if "fx.invalid/font/" in r.request.url: return r.fulfill(body=(FONTS / n).read_bytes(), content_type="font/ttf")
            r.abort()
        pg.route("**/*", route)
        for k, s in enumerate(inputs):
            pg.set_content(f'<!doctype html><html class="fit1" style="--apph:64px"><head><style>{FONT_CSS}{MONO_CSS}</style><style>{css}</style>'
                           '<style>.__nb::before{content:none!important} *,*::before,*::after{animation:none!important;transition:none!important}</style></head>'
                           f'<body>{CARD}</body></html>')
            pg.evaluate("() => Promise.all([...document.fonts].map((f) => f.load()))")
            pg.wait_for_function("document.fonts.check('700 15px \"Pretendard Variable\"') && document.fonts.check('700 11px \"JetBrains Mono\"')")
            pg.evaluate(RUN, [BODY, s])
            pg.evaluate("document.fonts.ready")
            if os.environ.get("ACC_DBG"):
                print(k, pg.evaluate("""(qs) => qs.map((q) => [q, [...document.querySelectorAll(q)].map((e) => { const r = e.getBoundingClientRect(), o = document.querySelector('#acctCard .ops-health-head').getBoundingClientRect(), c = document.querySelector('#acctCard').getBoundingClientRect();
                    return [r.left - c.left, r.top - o.bottom, r.width, r.height].map((v) => Math.round(v * 64) / 64); })])""", os.environ["ACC_DBG"].split("|")))
            if os.environ.get("ACC_SHOT"): pg.locator("#acctCard").screenshot(path=f'{os.environ["ACC_SHOT"]}/acc_{k}.png')
            out.append(pg.evaluate(DUMP))
        probe = pg.evaluate("""(ps) => ps.map(([s, z, w, mono]) => { const e = document.createElement("span");
            e.style.cssText = `font: ${w} ${z}px ${mono ? '"JetBrains Mono"' : '"Pretendard Variable"'}; white-space: pre; position: absolute`; e.textContent = s; document.body.append(e);
            const r = e.getBoundingClientRect().width; e.remove(); return [s, z, w, mono, Math.round(r * 1000) / 1000]; })""",
            [[" ", z, w, m] for z in (10, 10.5, 11, 11.5, 12, 12.5, 13, 14, 15, 19, 20, 30) for w in (400, 600, 700, 800) for m in (False, True)] + [["…", 12, 400, False], ["…", 12, 700, False], ["…", 12.5, 700, False]])
    finally:
        b.close()
pathlib.Path(os.environ.get("ACC_OUT", FX / "acc_js.json")).write_text(json.dumps({"cases": out, "probe": probe}, ensure_ascii=False) + "\n")
for k, o in enumerate(out):
    print(k, f'{o["w"]}x{o["h"]} 글 {len(o["texts"])} 상자 {len(o["boxes"])} 선 {len(o["lines"])} 고리 {len(o["rings"])} svg {[len(s["els"]) for s in o["svgs"]]}')
