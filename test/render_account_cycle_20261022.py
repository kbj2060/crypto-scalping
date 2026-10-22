#!/usr/bin/env python3
"""10-22 계좌 카드 «진행 중 순환»(시안 A)을 실제 브라우저로 -- 서버·거래소 **없이**(render_fee_rate_20261010 패턴).

    python3 -u test/render_account_cycle_20261022.py [OUT_DIR]

픽스처 = 실체결 10-20 23:10 ETHUSDC 롱 7체결(열린 왕복, round_trips 로 접은 legs) + 그 구간 1분 종가 · 스냅샷 10-21 02:40 KST.
시각은 «지금»에 맞춰 통째로 민다. 🔴주문 0: 출처 가짜 호스트 · 모든 요청 가로채기 · 바깥 호스트 abort · 주문 API 는 부르지 않는다.
검사(1920·390): 타일 6칸 값(합계·그냥 보유·순환 효과 $·% = 파이썬 재계산) · 체결가 글자·칩 = 체결 수 · 칩마다 정확한 체결가 ·
  /api/cycle-path 가 체결 심볼로 불림 · 포지션 없으면 칸이 빈다 · 페이지 오류 0.
"""
import json, pathlib, sys, time
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

WT = pathlib.Path(__file__).resolve().parents[1]
DASH = WT / "dashboard/live"
ORIGIN = "http://dash.local"
URL = f"{ORIGIN}/dashboard/live/"
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/render_account_cycle_20261022"); OUT.mkdir(parents=True, exist_ok=True)
FX = json.loads((WT / "test/fixtures/account_cycle_20261022.json").read_text("utf-8"))
CT = {".js": "application/javascript", ".css": "text/css", ".html": "text/html", ".json": "application/json",
      ".webmanifest": "application/manifest+json", ".svg": "image/svg+xml", ".png": "image/png"}
FONT = pathlib.Path("/mnt/c/Windows/Fonts/malgun.ttf")
SHIFT = int(time.time() * 1000) - FX["snap"]
trip = json.loads(json.dumps(FX["trip"]))
trip["entry_time"] += SHIFT
for g in trip["legs"]:
    g[0] += SHIFT
PATH = [[t + SHIFT, c] for t, c in FX["path"]]
P = PATH[-1][1]
q = avg = 0.0
for _, p, qty, inc, *_ in trip["legs"]:
    if inc:
        avg = (avg * q + p * qty) / (q + qty); q += qty
    else:
        q -= qty
POS = {"symbol": "ETHUSDC", "side": "LONG", "qty": round(q, 3), "entry_price": round(avg, 2), "mark_price": P, "leverage": 10.0,
       "notional": q * P, "unrealized_pnl": q * (P - avg), "liquidation_price": 1500.0}
real = sum(g[5] for g in trip["legs"]); fee = sum(g[6] for g in trip["legs"])
total = real + POS["unrealized_pnl"] - fee
hold = trip["legs"][0][2] * (P - trip["legs"][0][1]); cyc = total - hold
qq = aa = peak = 0.0
for _, p_, qty, inc, *_ in trip["legs"]:            # 합계 % 분모 = 순환 중 최대 원가(수량×평단) · 그냥 보유 % 분모 = 첫 체결 원금
    if inc:
        aa = (aa * qq + p_ * qty) / (qq + qty); qq += qty
    else:
        qq -= qty
    peak = max(peak, qq * aa)
totR, holdR = total / peak, hold / (trip["legs"][0][2] * trip["legs"][0][1])
pc = lambda r: f"{'+' if r >= 0 else '−'}{abs(r * 100):.2f}%"   # noqa: E731
su = lambda v: f"{'+' if v >= 0 else '−'}${abs(v):,.2f}".rstrip("0").rstrip(".")   # noqa: E731 -- app.js acctSu(|v|<100 · 소수 최대 2자리, 끝 0 없음)
S = {"positions": [POS]}
log, fails, errors = [], [], []


def account():
    return {"ok": True, "generated_at": "2026-10-21T00:00:00+00:00", "exec_symbol": "ETHUSDC", "exec_symbols": {"eth": "ETHUSDC"},
            "balance": {"margin": 1278.77, "wallet": 1278.77 - POS["unrealized_pnl"], "available": 900.0, "initial_margin": 100.0},
            "positions": S["positions"], "leverage_by_symbol": {"ETHUSDC": 10.0}, "trades": [trip] if S["positions"] else [], "ledger": []}


def handle(route):
    req = route.request; u = req.url; path = urlparse(u).path
    log.append((req.method, u))
    if not u.startswith(ORIGIN):
        return route.abort()
    js = lambda obj: route.fulfill(body=json.dumps(obj), content_type="application/json")   # noqa: E731
    if path == "/api/binance-account":
        return js(account())
    if path == "/api/cycle-path":
        return js({"ok": True, "interval": "1m", "rows": PATH})
    if path == "/__font/kr.ttf" and FONT.is_file():
        return route.fulfill(body=FONT.read_bytes(), content_type="font/ttf")
    if path.startswith("/api/"):      # 주문 API 포함 전부 503 -- 이 하네스는 주문 버튼을 누르지 않는다
        return route.fulfill(status=503, body="{}", content_type="application/json")
    name = "index.html" if path in ("/dashboard/live/", "/dashboard/live") else path.split("/dashboard/live/", 1)[-1]
    f = DASH / name
    if f.is_file():
        return route.fulfill(body=f.read_bytes(), content_type=CT.get(f.suffix, "application/octet-stream"))
    return route.fulfill(status=404, body="")


with sync_playwright() as p:
    b = p.chromium.launch()
    for w, h in ((1920, 1080), (390, 844)):
        S["positions"] = [POS]
        ctx = b.new_context(viewport={"width": w, "height": h}, device_scale_factor=2, has_touch=w < 500)
        pg = ctx.new_page()
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.route(lambda u: True, handle)
        pg.route_web_socket(lambda u: True, lambda ws: None)
        pg.add_init_script("try { localStorage.clear(); } catch (e) {}"
                           "document.addEventListener('DOMContentLoaded', () => { const st = document.createElement('style');"
                           "st.textContent = ['Noto Sans KR', 'Pretendard Variable', 'JetBrains Mono', 'Space Grotesk'].map((f) => `@font-face { font-family: '${f}';"
                           " src: url(/__font/kr.ttf); unicode-range: U+1100-11FF, U+3130-318F, U+AC00-D7AF; }`).join(' '); document.head.appendChild(st); });")
        pg.goto(URL, wait_until="load", timeout=60000); pg.wait_for_timeout(2500)
        pg.evaluate(f"() => renderBinanceAccount({json.dumps(account())})"); pg.wait_for_timeout(800)
        box = pg.locator("#snapAcctCycle")
        tiles = box.locator(".cy-tiles b").all_inner_texts()
        chips = box.locator(".cy-fills span").all_inner_texts()
        labels = box.locator("text.cy-fx").count()
        pts = pg.evaluate("() => (document.querySelector('#snapAcctCycle polyline')?.getAttribute('points') || '').split(' ').length")
        print(w, tiles, len(chips), labels, pts)
        for i, v in {3: f"{su(total)} {pc(totR)}", 4: f"{su(hold)} {pc(holdR)}", 5: f"{su(cyc)} {pc(totR - holdR)}p"}.items():
            if i >= len(tiles) or tiles[i] != v: fails.append(f"[{w}] 타일 {i} {tiles[i] if i < len(tiles) else None} != {v}")
        n = len(trip["legs"])
        if len(chips) != n or labels != n: fails.append(f"[{w}] 칩 {len(chips)} · 글자 {labels} != {n}")
        for (_, px, qty, *_), c in zip(trip["legs"], chips):
            if f"{px:,.2f}" not in c or f"×{qty:g}" not in c: fails.append(f"[{w}] 칩 {c!r} 에 {px:,.2f} ×{qty:g} 없음")
        if pts < len(PATH): fails.append(f"[{w}] 가격선 점 {pts} < {len(PATH)}(경로 안 받음)")
        if not any("/api/cycle-path?symbol=ETHUSDC" in u for _, u in log): fails.append(f"[{w}] cycle-path 요청 없음")
        pg.evaluate("() => document.getElementById('snapAcctCycle').scrollIntoView()"); pg.wait_for_timeout(200)
        box.screenshot(path=str(OUT / f"cycle_{w}.png"))
        pg.locator("#acctCard").screenshot(path=str(OUT / f"acct_card_{w}.png"))
        S["positions"] = []
        pg.evaluate(f"() => renderBinanceAccount({json.dumps(account())})"); pg.wait_for_timeout(300)
        if pg.evaluate("() => document.getElementById('snapAcctCycle').innerHTML.trim()"): fails.append(f"[{w}] 청산 뒤에도 칸이 남음")
        ctx.close()
    b.close()

if errors: fails.append(f"페이지 오류 {errors[:3]}")
print(f"P {P} q {q:.3f} avg {avg:.2f} total {total:.2f} hold {hold:.2f} cyc {cyc:.2f} · 가로챈 요청 {len(log)}")
print("FAIL" if fails else "PASS", *fails, sep="\n")
sys.exit(1 if fails else 0)
