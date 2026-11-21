#!/usr/bin/env python3
"""플로팅 주문 버튼 제거(2026-11-21) 뒤 남은 주문 화면을 실제 브라우저로 -- 서버·거래소 **없이**(render_entry_cancel_20261011 패턴).

    python3 -u test/render_ord_stat_20261121.py [OUT_DIR]

🔴주문 0: 페이지 출처가 가짜 호스트(http://dash.local) · 모든 요청을 조건식으로 가로채 가짜 응답 · 바깥 호스트 abort · WebSocket 가짜 ·
  누르기 전 가짜 POST 로 가로채기 선검사(아니면 즉시 중단).
검사(1920·390):
  ① 차트 위 상태 줄(#ordStat): 성공(체결 · TP/손절 걸림)은 10초 뒤 저절로 접힌다 · 실패(«🔴TP/SL 을 못 걸었습니다»)는 빨강으로 남는다 ·
     짧은 안내(ordSay) 성공은 접히고 실패는 남는다 · ✕ 가 다 닫는다 · 비어 있으면 상자 자체가 안 보인다.
  ② «손절 없음» 판정(posNoSl): 무장 손절 있음 = 아님 · 꺼짐(sl_off) = 손절 없음 · 무장 없음 = 손절 없음.
  ③ 선 도구 배수(lineLev)가 발주 쿼리에 실린다(&lev=25 · 자동이면 &lev=10) · 청산 발주엔 lev·exec 가 없다(늘 지정가 따라가기) ·
     규칙(rule)·스위칭(switch_to)·기다리기(mode=wait) 인자는 어디에도 없다.
  JS 오류 0 · 바깥 호스트 요청은 전부 abort.
"""
import json, pathlib, sys
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

WT = pathlib.Path(__file__).resolve().parents[1]
DASH = WT / "dashboard/live"
ORIGIN = "http://dash.local"
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/render_ord_stat_20261121"); OUT.mkdir(parents=True, exist_ok=True)
CT = {".js": "application/javascript", ".css": "text/css", ".html": "text/html", ".json": "application/json",
      ".webmanifest": "application/manifest+json", ".svg": "image/svg+xml", ".png": "image/png"}
OK_ST = {"phase": "filled_maker", "kind": "entry", "side": "LONG", "asset": "eth", "quantity": 1.0, "filled": 1.0,
         "bracket": {"tp": {"placed": True, "price": 2600}, "backstop": {"placed": True, "price": 2400}}}
BAD_ST = {"phase": "filled_maker", "kind": "entry", "side": "LONG", "asset": "eth", "quantity": 1.0, "filled": 1.0,
          "bracket": {"reason": "하네스 — 거래소 응답 없음"}}
S = {"state": {"phase": "idle"}}
log, fails, errs = [], [], []


def handle(route):
    req = route.request; u = req.url; pr = urlparse(u)
    log.append((req.method, u))
    if not u.startswith(ORIGIN):
        return route.abort()
    js = lambda obj, st=200: route.fulfill(status=st, body=json.dumps(obj), content_type="application/json")   # noqa: E731
    if pr.path.startswith("/api/manual-"):
        if req.method != "GET":
            if pr.path.endswith("/submit"):
                return js({"ok": True, "state": {"phase": "submitting", "side": "LONG", "kind": "exit" if "exit" in pr.path else "entry"}})
            return js({"ok": False, "error": "harness"}, 400)
        if pr.path == "/api/manual-entry/status":
            return js({"ok": True, "state": S["state"], "wait_list": []})
        return route.fulfill(status=503, body="{}")
    if pr.path.startswith("/api/"):
        return route.fulfill(status=503, body="{}", content_type="application/json")
    name = "index.html" if pr.path in ("/dashboard/live/", "/dashboard/live") else pr.path.split("/dashboard/live/", 1)[-1]
    f = DASH / name
    return route.fulfill(body=f.read_bytes(), content_type=CT.get(f.suffix, "application/octet-stream")) if f.is_file() else route.fulfill(status=404, body="")


posts = lambda n0, frag: [u for m, u in log[n0:] if m != "GET" and frag in u]   # noqa: E731
BOX = """() => { const o = document.getElementById('ordStat'), r = document.getElementById('snapEntryResult'), s = document.getElementById('ordSay');
  return { shown: getComputedStyle(o).display !== 'none', res: !r.hidden, resText: r.textContent, resBad: !!r.querySelector('.bad'),
           say: !s.hidden, sayBad: s.classList.contains('bad'), inView: o.getBoundingClientRect().right <= innerWidth + 1 }; }"""
# 플랫 계좌(배수가 기존 포지션에 묶이지 않게) -- 발주 쿼리 검사용
FLAT = "() => { latestBinanceAccount = { exec_symbol: 'ETHUSDC', exec_symbols: { eth: 'ETHUSDC' }, positions: [], bracket_armed: {} }; }"

with sync_playwright() as p:
    b = p.chromium.launch()
    for w, h in ((1920, 1080), (390, 844)):
        ctx = b.new_context(viewport={"width": w, "height": h}, device_scale_factor=1, has_touch=w < 500)
        pg = ctx.new_page(); pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.route(lambda u: True, handle); pg.route_web_socket(lambda u: True, lambda ws: None)
        pg.add_init_script("try { localStorage.clear(); } catch (e) {}")
        pg.goto(f"{ORIGIN}/dashboard/live/", wait_until="load", timeout=60000); pg.wait_for_timeout(2000)
        n0 = len(log)
        pg.evaluate("() => fetch('/api/manual-entry/submit?side=LONG&confirm=1&probe=1', {method:'POST'})"); pg.wait_for_timeout(300)
        if not posts(n0, "probe=1"):
            print("🔴가로채기 실패 -- 중단"); sys.exit(2)
        s0 = pg.evaluate(BOX)
        if s0["shown"]: fails.append(f"[{w}] 아무 일도 없는데 상태 줄이 보인다 {s0}")

        # ③ 배수 → 발주 쿼리 · 청산 쿼리
        pg.evaluate(FLAT)
        n1 = len(log)
        S["state"] = OK_ST
        pg.evaluate("() => { lineLevAuto = false; lineLev = 25; manualFireNow('LONG', 'entry', { pct: 10, tp: '2600', sl: '2400', fresh: true }); }")
        pg.wait_for_timeout(1500)
        q = posts(n1, "/api/manual-entry/submit")
        if len(q) != 1 or "&lev=25" not in q[0] or "&pct=10" not in q[0] or "&tp=2600&sl=2400" not in q[0]:
            fails.append(f"[{w}] 선 진입 쿼리에 배수 25·비율·선이 안 실렸다 {q}")
        # ① 성공 = 보였다가 10초 뒤 접힘
        s1 = pg.evaluate(BOX); print(w, "성공 직후", s1)
        if not (s1["shown"] and s1["res"] and "걸림" in s1["resText"] and not s1["resBad"] and s1["inView"]):
            fails.append(f"[{w}] 성공 결과가 상태 줄에 안 보인다 {s1}")
        pg.locator("#ordStat").screenshot(path=str(OUT / f"ok_{w}.png"))
        pg.wait_for_timeout(10600)
        s2 = pg.evaluate(BOX); print(w, "성공 10초 뒤", s2["shown"], s2["res"])
        if s2["shown"] or s2["res"]: fails.append(f"[{w}] 성공 결과가 10초 뒤에도 남았다 {s2}")
        n2 = len(log)
        pg.evaluate("() => { lineLevAuto = true; manualFireNow('LONG', 'entry', { pct: 5, tp: '2600', sl: '2400', fresh: true }); }")
        pg.wait_for_timeout(1500)
        q2 = posts(n2, "/api/manual-entry/submit")
        if len(q2) != 1 or "&lev=10" not in q2[0]: fails.append(f"[{w}] 자동 배수 10 이 안 실렸다 {q2}")
        pg.wait_for_timeout(10600)
        # ① 실패 = 빨강으로 남음
        S["state"] = BAD_ST
        n3 = len(log)
        pg.evaluate("() => manualFireNow('LONG', 'exit', { pct: 50, src: 'ep-mouse' })")
        pg.wait_for_timeout(1500)
        q3 = posts(n3, "/api/manual-exit/submit")
        if len(q3) != 1 or "&pct=50" not in q3[0] or "&lev=" in q3[0] or "exec=" in q3[0]:
            fails.append(f"[{w}] 청산 쿼리 {q3} -- pct 50 · lev·exec 없음이어야 한다")
        pg.wait_for_timeout(10600)
        s3 = pg.evaluate(BOX); print(w, "실패 10초 뒤", s3["shown"], s3["resBad"], s3["resText"][:50])
        if not (s3["shown"] and s3["res"] and s3["resBad"] and "TP/SL 을 못 걸었습니다" in s3["resText"]):
            fails.append(f"[{w}] 실패가 10초 뒤 남아 있지 않거나 빨강이 아니다 {s3}")
        pg.locator("#ordStat").screenshot(path=str(OUT / f"bad_{w}.png"))
        # 짧은 안내: 성공은 접히고 실패는 남는다
        pg.evaluate("() => ordSay('익절선 걸림 2,600.00')"); pg.wait_for_timeout(10600)
        if pg.evaluate(BOX)["say"]: fails.append(f"[{w}] 성공 안내가 10초 뒤에도 남았다")
        pg.evaluate("() => ordSay('손절 해제 못 함: 하네스', 'bad')"); pg.wait_for_timeout(10600)
        s4 = pg.evaluate(BOX)
        if not (s4["say"] and s4["sayBad"]): fails.append(f"[{w}] 실패 안내가 사라졌다 {s4}")
        # ✕ = 전부 닫음
        pg.click("#ordStat .ord-x"); pg.wait_for_timeout(200)
        s5 = pg.evaluate(BOX)
        if s5["shown"] or s5["res"] or s5["say"]: fails.append(f"[{w}] ✕ 뒤에도 남았다 {s5}")
        # ② 손절 없음 판정
        got = pg.evaluate("""() => { const pos = { symbol: 'ETHUSDC', side: 'LONG', qty: 1 }, k = 'ETHUSDC:LONG', r = [];
          for (const b of [{ sl_price: 2400, lines: true }, { sl_price: 2400, sl_off: { at: 1 } }, null]) {
            latestBinanceAccount = { positions: [pos], bracket_armed: b ? { [k]: b } : {} }; r.push(posNoSl(pos)); }
          return r; }""")
        if got != [False, True, True]: fails.append(f"[{w}] 손절 없음 판정 {got} != [false, true, true]")
        ctx.close()
    b.close()

bad_args = [u for m, u in log if m != "GET" and any(x in u for x in ("rule=", "switch_to", "mode=wait", "wait_min"))]
if bad_args: fails.append(f"지운 기능의 인자가 나갔다 {bad_args}")
outside = [u for _, u in log if not u.startswith(ORIGIN)]
print("바깥 호스트 요청(전부 abort)", len(outside), "· JS 오류", errs[:3])
if errs: fails.append(f"JS 오류 {errs[:3]}")
print("FAIL" if fails else "PASS", *fails, sep="\n")
sys.exit(1 if fails else 0)
