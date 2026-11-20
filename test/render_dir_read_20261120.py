#!/usr/bin/env python3
"""«지금 읽기»(지표 방향 판 아래, 2026-11-20 시안 C)를 실제 화면에 그려 확인한다.

    python3 -u test/render_dir_read_20261120.py --harness DIR [--shot OUT.png]

DIR = 디자인 하네스 폴더(트레이딩룸 아티팩트의 desk.html · shim.js · fixtures.json · *.js · styles.css — 서버 없이 실서비스 응답 스냅숏으로 돈다).
그 위에 로컬 dashboard/live/app.js 를 덮는다. 네트워크·주문은 shim 이 막는다.
검사: 실데이터 읽기 칸이 뜸 · 가짜 4줄(▼▼▲ + 크기 네모)에서 개수·등급·말줄임 · 3분 낡으면 숨김 · 접으면 숨김 · ETH 아니면 없음 · 휴대폰엔 판 없음 · JS 오류 0.
"""
import argparse, functools, http.server, pathlib, threading

APP = pathlib.Path(__file__).resolve().parents[1] / "dashboard" / "live" / "app.js"
SYN = """() => {
  window.fetch = () => new Promise(() => {}); if (window.liveStream) liveStream.close();   // 가짜 줄을 폴링이 덮지 않게
  const mk = (topic, text, grade, dir, v) => ({ topic, text, note: "근거", grade, dir, g: { kind: "d", v, on: true } });
  const rest = latestSituation.read.lines.filter((l) => l.grade === "설명" && !["고래↔리테일", "가격↔OI", "호가↔체결", "활동"].includes(l.topic));
  latestSituation.read.lines = [mk("고래↔리테일", "관찰 — 고래 매도 · 리테일 매수로 갈렸다: 고래 쪽", "약함", -1, -60), mk("가격↔OI", "관찰 — 하락 + OI 증가: 새 숏", "근거", -1, -100),
    mk("호가↔체결", "관찰 — 매수벽 우위, 시장가(매도)를 받아낼 쪽이 두껍다", "약함", 1, 90), mk("활동", "관찰 — 움직임이 커질 자리", "근거", 0, 93)].concat(rest);
  latestSituation.computed_at = Date.now() / 1000; renderSnapshotChart();
  const T = () => [...document.querySelectorAll("#dirBoard text")].map((t) => t.textContent), syn = T();
  const sq = document.querySelectorAll('#dirBoard rect[rx="2"]').length, nrest = rest.length;
  latestSituation.computed_at = Date.now() / 1000 - 400; renderSnapshotChart(); const stale = T();
  latestSituation.computed_at = Date.now() / 1000; dirFold = true; renderSnapshotChart(); const fold = T(); dirFold = false;
  const a = activeSnapshotAsset; activeSnapshotAsset = "sol"; const other = dirReadNow(); activeSnapshotAsset = a;
  return { syn, sq, nrest, stale, fold, other };
}"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--harness", required=True); ap.add_argument("--shot")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    H = type("H", (http.server.SimpleHTTPRequestHandler,), {"log_message": lambda *x: None})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(H, directory=a.harness))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/desk.html"
    with sync_playwright() as p:
        b = p.chromium.launch()
        for w, h in ((1920, 1080), (390, 844)):
            pg = b.new_page(viewport={"width": w, "height": h}); errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.route("**/app.js*", lambda r: r.fulfill(path=str(APP), content_type="text/javascript"))
            pg.goto(url, wait_until="load"); pg.wait_for_timeout(9000)
            texts = pg.evaluate('() => [...document.querySelectorAll("#dirBoard text")].map((t) => t.textContent)')
            if w == 390:
                assert not texts, texts                       # 휴대폰엔 판이 없다
                m = pg.evaluate('() => { const b = document.getElementById("dirReadM"); return b && !b.hidden ? b.innerText : ""; }')
                assert "지금 읽기" in m and "설명 " in m, m      # 2026-11-20 대신 차트 아래 목록(renderDirReadMob)
            else:
                assert "지금 읽기" in texts and any(t.startswith("설명 ") for t in texts), texts
                if a.shot: pg.screenshot(path=a.shot)
                r = pg.evaluate(SYN); s = r["syn"]; i = s.index("지금 읽기")
                assert s[i + 1:i + 3] == ["1", "2"], s            # ▲1 ▼2
                assert [s[s.index(n) + 1] for n in ("고래↔리테일", "가격↔OI", "호가↔체결", "활동")] == ["약함", "근거", "약함", "근거"], s
                assert r["sq"] == 1 and f"설명 {r['nrest']}" in s, (r["sq"], s)   # 크기 줄 네모 하나
                assert any(t.endswith("…") for t in s), s        # 긴 풀이는 말줄임
                assert "지금 읽기" not in r["stale"] and "지금 읽기" not in r["fold"] and r["other"] is None, r
            assert not errs, errs
            pg.close()
        b.close()
    print("render_dir_read OK")


if __name__ == "__main__":
    main()
