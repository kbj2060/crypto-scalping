"""개발기 자체 확인: serve.py 를 띄우고 크로미움으로 페이지 하나씩 끝까지 돌려 결과(window.BENCH)와 «페이지가 부른 주소 목록»을 찍고 서버를 끈다.
python3 -I rtui/bench/headless.py [--gpu] [--rates 100,300,800] [--repeat 3] [--secs 8] [--port 8791] <페이지...>
  페이지 = rtui | js | 'rtui?aa=0' · 'rtui?src=push'(가짜 허브) 처럼 주소 인자를 붙여도 된다
  --gpu    = 헤드리스 대신 WSLg 창(x11)으로 띄워 Mesa d3d12 → 윈도 그래픽 카드로 그린다(화면에 창이 잠깐 뜬다 -- gpu_probe.py)
  --repeat = 주소마다 N 번(순서: 반복 → 부하 → 페이지 -- 같은 주소가 몰려 같은 흔들림을 타지 않게) · 끝에 중앙값 표(마크다운)와 results/table_<시각>.json
  매 실행 전·후 loadavg(1분)를 적는다 -- 다른 에이전트의 컴파일과 겹친 실행을 가려 보라고(표의 «부하 평균» 칸 = 그 주소 실행들의 최댓값)
바깥 요청 = 127.0.0.1:<포트> 가 아닌 모든 요청·웹소켓(브라우저는 이름 풀이를 전부 막아 실제로 나가지는 않는다)"""
import argparse, json, os, pathlib, statistics, subprocess, sys, time
from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("pages", nargs="+")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--rates", default="300")
ap.add_argument("--repeat", type=int, default=1)
ap.add_argument("--secs", type=float, default=8)
ap.add_argument("--port", type=int, default=8791)
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"
(HERE / "results").mkdir(exist_ok=True)
log = open(HERE / "results/serve.log", "a")
srv = subprocess.Popen(["bash", "-c", f"ulimit -v 4000000; exec timeout 7200 {sys.executable} -I {HERE / 'serve.py'} --port {a.port}"], stdout=log, stderr=subprocess.STDOUT)
KEYS = [("도착→표시(rAF 기준) 중앙", "표시 중앙"), ("도착→표시 상위 1%", "표시 상위1%"), ("도착→그리기 끝 중앙", "그리기 끝 중앙"), ("프레임 간격 중앙", "간격 중앙"),
        ("프레임 간격 상위 1%", "간격 상위1%"), ("프레임당 앱 rAF 콜백 중앙", "콜백 중앙"), ("메인 스레드 바쁨 %(부하)", "바쁨 %"), ("긴 작업 수", "긴 작업"),
        ("유휴 바쁨 %", "유휴 바쁨 %"), ("첫 화면 ms", "첫 화면")]
rows = {}
try:
    time.sleep(1.5)
    env, args = dict(os.environ), ["--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1", "--disable-background-networking"]
    if a.gpu:
        env |= {"LD_LIBRARY_PATH": "/usr/lib/wsl/lib", "GALLIUM_DRIVER": "d3d12"}
        args += ["--ignore-gpu-blocklist", "--enable-gpu", "--ozone-platform=x11", "--use-gl=angle", "--use-angle=gl"]
    with sync_playwright() as p:
        for rep in range(a.repeat):
            for rate in [int(x) for x in a.rates.split(",")]:
                for page in a.pages:
                    nm, _, extra = page.partition("?")
                    url = (f"{BASE}/rtui/index.html?asset=eth&" if nm == "rtui" else f"{BASE}/dashboard/live/?") + f"bench={a.secs}&rate={rate}" + (f"&{extra}" if extra else "")
                    la0 = os.getloadavg()[0]
                    br = p.chromium.launch(headless=not a.gpu, args=args, env=env)
                    try:
                        pg = br.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1, service_workers="block", locale="ko-KR", timezone_id="Asia/Seoul").new_page()
                        urls = []
                        pg.on("request", lambda r: urls.append(r.url))
                        pg.on("websocket", lambda w: urls.append(w.url))
                        pg.on("pageerror", lambda e: print("pageerror", str(e)[:200]))
                        pg.goto(url)
                        pg.wait_for_function("window.BENCH && (window.BENCH.error || window.BENCH.saved)", timeout=(a.secs + 60) * 1000, polling=200)
                        r = pg.evaluate("window.BENCH")
                    finally:
                        br.close()
                    la1 = os.getloadavg()[0]
                    ext = sorted({u for u in urls if not u.startswith((BASE, f"ws://127.0.0.1:{a.port}")) and not u.startswith("data:")})
                    s = r.get("summary", r)
                    print(f"== {page} {rate}/초 #{rep + 1} {'(그래픽 카드 · WSLg)' if a.gpu else '(소프트웨어 렌더 · 헤드리스)'} · loadavg {la0:.2f}→{la1:.2f}")
                    print(json.dumps(s, ensure_ascii=False))
                    print("저장", r.get("saved"), "· 요청", len(urls), "건 · 바깥", ext or "0건", "· 페이지 안 기록", json.dumps(r.get("external"), ensure_ascii=False), flush=True)
                    rows.setdefault((page, rate), []).append({**s, "loadavg": max(la0, la1), "external_seen": len(ext)})
finally:
    srv.terminate()
    try: srv.wait(5)
    except subprocess.TimeoutExpired: srv.kill()
    log.close()

med = lambda v: round(statistics.median(v), 2) if v else None
print(f"\n| 부하 | 페이지 | 렌더러 | n | " + " | ".join(k[1] for k in KEYS) + " | 부하 평균(최대) | 바깥 |")
print("|" + "---|" * (len(KEYS) + 6))
table = []
for (page, rate), rs in sorted(rows.items(), key=lambda x: (x[0][1], x[0][0])):
    ok = [r for r in rs if "렌더러" in r]
    m = {k[1]: med([r[k[0]] for r in ok if isinstance(r.get(k[0]), (int, float))]) for k in KEYS}
    rend = ok[0]["렌더러"] if ok else "실패"
    gpu = "AMD" if "AMD" in rend else ("SwiftShader" if "SwiftShader" in rend else rend[:20])
    la = max(r["loadavg"] for r in rs)
    table.append({"page": page, "rate": rate, "renderer": rend, "n": len(ok), **m, "loadavg_max": la})
    print(f"| {rate} | {page} | {gpu} | {len(ok)} | " + " | ".join(str(m[k[1]]) for k in KEYS) + f" | {la:.2f} | {sum(r['external_seen'] for r in rs)} |")
f = HERE / f"results/table_{time.strftime('%Y%m%d-%H%M%S')}.json"
f.write_text(json.dumps({"gpu": a.gpu, "secs": a.secs, "repeat": a.repeat, "rows": table}, ensure_ascii=False, indent=1))
print("표", f)
