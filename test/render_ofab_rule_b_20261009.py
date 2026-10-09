#!/usr/bin/env python3
"""떠다니는 주문 버튼 B안(2026-10-09, 사용자 선택) -- 권고 규칙 «자동» 띠·버튼 두 줄·꾹 진입에 rule 이 실리는지 실제 브라우저로.

    python3 -u test/render_ofab_rule_b_20261009.py [OUT_DIR]     # 터널 127.0.0.1:18787 필요

🔴주문 0: /api/manual- 전부 조건식(predicate) 가로채기 · 누르기 전 가짜 제출로 가로채기 선검사(아니면 즉시 중단) · 전후 실서버 마지막 주문 번호 대조.
🔴실계좌에 포지션이 있으면 30초 재조회가 «포지션 없음» 상태를 되살린다 -- 타이머를 멈추고 시작한다.
검사(1920·390): 직접 = «5%» · 띠 «자동» → 카드 #snapRule·localStorage 동기화 · 버튼 «첫 $ · 손절 가격» 두 줄 · 진입 줄 rule-on ·
  (1920) 마우스 1.5초 꾹 → 미리보기·제출(가짜)에 rule=c · 띠 «직접» → «5%» 복귀.
"""
import json, pathlib, sys, urllib.request
from urllib.parse import parse_qs, urlparse
from playwright.sync_api import sync_playwright
WT = pathlib.Path(__file__).resolve().parents[1]
DASH = WT / "dashboard/live"; URL = "http://127.0.0.1:18787/dashboard/live/"
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/render_ofab_rule_b"); OUT.mkdir(parents=True, exist_ok=True)
PREVIEW = json.loads((WT / "test/fixtures/manual_entry_preview_20261005.json").read_text("utf-8"))
files = {n: (DASH / n).read_text("utf-8") for n in ("app.js", "styles.css", "index.html", "fp_daily.js", "liq_profile.js")}
def real_last():
    with urllib.request.urlopen("http://127.0.0.1:18787/api/manual-entry/status", timeout=20) as r:
        s = json.load(r).get("state") or {}
    return s.get("order_id"), s.get("started_at")
log, fails = [], []
def manual(route):
    u = route.request.url; log.append((route.request.method, u))
    if "/submit" in u or "/execute" in u or "/cancel" in u or route.request.method != "GET":
        route.fulfill(body=json.dumps({"ok": True, "state": {"phase": "submitting"}, "plan": PREVIEW.get("plan")}), content_type="application/json"); return
    if "/status" in u:
        route.fulfill(body=json.dumps({"ok": True, "state": {"phase": "idle"}}), content_type="application/json"); return
    q = {k: v[0] for k, v in parse_qs(urlparse(u).query).items()}
    body = json.loads(json.dumps(PREVIEW)); body["exec_enabled"] = True
    if q.get("rule"):
        body["plan"]["rule"] = {"l": 6.2, "binding": "노출 맞춤", "cap_notional": 3000.0, "first": True, "room": 3000.0, "target_notional": 2250.0,
                                "sl_price": 2300.0, "sl_name": "첫 진입가 −7.5%(3σ)"}
    route.fulfill(body=json.dumps(body), content_type="application/json")
EMPTY = """() => {
  for (let i = 1; i < 99999; i++) window.clearInterval(i);   // 실계좌 재조회가 비운 포지션을 되살리지 않게(테스트 인공물 방지)
  latestBinanceAccount = Object.assign({}, latestBinanceAccount || {}, { positions: [] });
  manualExitSyncButtons(); renderSnapshotAccount(); ofabSync(); renderOfab();
}"""
before = real_last()
with sync_playwright() as p:
    b = p.chromium.launch()
    for w, h in ((1920, 1080), (390, 844)):
        pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
        pg.route(lambda u: "/api/manual-" in u, manual)
        for name, ct in (("app.js", "application/javascript"), ("fp_daily.js", "application/javascript"), ("liq_profile.js", "application/javascript"), ("styles.css", "text/css")):
            pg.route(f"**/{name}*", (lambda body, ct: lambda r: r.fulfill(body=body, content_type=ct))(files[name], ct))
        pg.route(URL, lambda r: r.fulfill(body=files["index.html"], content_type="text/html"))
        pg.add_init_script("try { localStorage.setItem('entryRule', '0'); } catch (e) {}")
        pg.goto(URL, wait_until="load", timeout=60000); pg.wait_for_timeout(7000)
        # 선검사: 가짜 제출이 가로채지는가(아니면 즉시 중단)
        n0 = len(log)
        pg.evaluate("() => fetch('/api/manual-entry/submit?side=LONG&confirm=1&pct=1&probe=1', {method:'POST'})"); pg.wait_for_timeout(500)
        if not any("probe=1" in u for _, u in log[n0:]):
            print("🔴가로채기 실패 -- 중단"); sys.exit(2)
        pg.evaluate(EMPTY); pg.wait_for_timeout(300)
        st0 = pg.evaluate("() => ({ strip: !document.getElementById('ofabRule').hidden, l: document.getElementById('ofabLongFrac').textContent, two: document.getElementById('ofabLong').classList.contains('two') })")
        if not st0["strip"] or st0["l"] != "5%" or st0["two"]: fails.append(f"[{w}] 직접 상태 {st0}")
        pg.screenshot(path=str(OUT / f"b_direct_{w}.png"), clip=pg.evaluate("() => { const r = document.getElementById('ofabFlat').getBoundingClientRect(); return {x: Math.max(0, r.x - 60), y: Math.max(0, r.y - 60), width: Math.min(innerWidth - Math.max(0, r.x - 60), r.width + 120), height: r.height + 76}; }"))
        pg.click("#ofabRule [data-v='1']"); pg.wait_for_timeout(500)
        st1 = pg.evaluate("""() => ({ card: document.getElementById('snapRule').value, store: localStorage.getItem('entryRule'),
            l: document.getElementById('ofabLongFrac').textContent, s: document.getElementById('ofabShortFrac').textContent,
            two: document.getElementById('ofabLong').classList.contains('two'),
            pressed: [...document.querySelectorAll('#ofabRule button')].map(b => b.getAttribute('aria-pressed')).join(','),
            ruleOn: document.getElementById('snapRuleBox').closest('.entry-line').classList.contains('rule-on') })""")
        print(w, "자동", st1)
        if st1["card"] != "1" or st1["store"] != "1" or not st1["l"].startswith("첫 $") or "손절" not in st1["s"] or not st1["two"] or st1["pressed"] != "false,true" or not st1["ruleOn"]:
            fails.append(f"[{w}] 자동 상태 {st1}")
        pg.screenshot(path=str(OUT / f"b_auto_{w}.png"), clip=pg.evaluate("() => { const r = document.getElementById('ofabFlat').getBoundingClientRect(); return {x: Math.max(0, r.x - 60), y: Math.max(0, r.y - 60), width: Math.min(innerWidth - Math.max(0, r.x - 60), r.width + 120), height: r.height + 76}; }"))
        # 펼친 패널: 자동이면 진입비율 게이지·SL/TP 숨김 + «크기» 칩 보임(10-09 사용자 «진입비율 게이지가 왔다갔다 된다»)
        VIS = """() => { const v = (e) => !!e && getComputedStyle(e).display !== 'none' && e.getBoundingClientRect().height > 0;
                  return { frac: v(document.getElementById('snapEntryFrac')), sltp: v(document.querySelector('.sltp-toggle')),
                           rule: v(document.querySelector('#snapRuleBox .chipset')), inPanel: !!document.querySelector('#ofabPanel #snapRuleBox'),
                           vm: v(document.getElementById('snapVolMult')) }; }"""
        pg.click("#ofabMore"); pg.wait_for_timeout(700)
        pv1 = pg.evaluate(VIS)
        print(w, "패널(자동)", pv1)
        if pv1 != {"frac": False, "sltp": False, "rule": True, "inPanel": True, "vm": False}: fails.append(f"[{w}] 자동 패널 {pv1}")
        pg.locator("#ofabPanel").screenshot(path=str(OUT / f"b_panel_auto_{w}.png"))
        pg.click("#ofabPanel #snapRuleBox .chip[data-v='0']"); pg.wait_for_timeout(500)
        pv2 = pg.evaluate(VIS)
        if not (pv2["frac"] and pv2["sltp"] and pv2["rule"]): fails.append(f"[{w}] 직접 패널 {pv2}")
        pg.click("#ofabPanel #snapRuleBox .chip[data-v='1']"); pg.wait_for_timeout(400)
        pg.click("#ofabMore"); pg.wait_for_timeout(500)
        if w > 500:   # 마우스 1.5초 꾹 = 미리보기 → 제출(가짜) 에 rule=c
            n1 = len(log)
            bb = pg.evaluate("() => document.getElementById('ofabLong').getBoundingClientRect().toJSON()")
            pg.mouse.move(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2); pg.mouse.down(); pg.wait_for_timeout(1900); pg.mouse.up(); pg.wait_for_timeout(1500)
            new = [u for _, u in log[n1:]]
            pv = [u for u in new if "/manual-entry/preview" in u and "side=LONG" in u]; sb = [u for u in new if "/manual-entry/submit" in u]
            print(w, "꾹 → 미리보기", len(pv), "제출(가짜)", len(sb), sb[:1])
            if not sb or "rule=c" not in sb[0] or not any("rule=c" in u for u in pv): fails.append(f"[{w}] 꾹 진입에 rule 없음 {new}")
        pg.click("#ofabRule [data-v='0']"); pg.wait_for_timeout(300)
        if pg.evaluate("() => document.getElementById('ofabLongFrac').textContent") != "5%": fails.append(f"[{w}] 직접으로 못 돌아옴")
        pg.close()
    b.close()
after = real_last()
if before != after: fails.append(f"🔴실서버 마지막 주문이 바뀜 {before} → {after}")
print("실서버 마지막 주문 전후", before == after, "· 가로챈 요청", len(log))
print("FAIL" if fails else "PASS", *fails, sep="\n")
sys.exit(1 if fails else 0)
