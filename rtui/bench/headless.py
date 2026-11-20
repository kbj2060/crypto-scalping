"""개발기 자체 확인: serve.py 를 띄우고 크로미움으로 페이지 하나씩 끝까지 돌려 결과(window.BENCH)와 «페이지가 부른 주소 목록»을 찍고 서버를 끈다.
python3 -I rtui/bench/headless.py [--gpu] [--rate 300] [--secs 8] [--port 8791] <페이지...>
  페이지 = rtui | js | 'rtui?aa=0' 처럼 주소 인자를 붙여도 된다
  --gpu  = 헤드리스 대신 WSLg 창(x11)으로 띄워 Mesa d3d12 → 윈도 그래픽 카드로 그린다(화면에 창이 잠깐 뜬다 -- gpu_probe.py)
바깥 요청 = 127.0.0.1:<포트> 가 아닌 모든 요청·웹소켓(브라우저는 이름 풀이를 전부 막아 실제로 나가지는 않는다)"""
import argparse, json, os, pathlib, subprocess, sys, time
from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("pages", nargs="+")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--rate", type=int, default=300)
ap.add_argument("--secs", type=float, default=8)
ap.add_argument("--port", type=int, default=8791)
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"
srv = subprocess.Popen(["bash", "-c", f"ulimit -v 4000000; exec timeout 900 {sys.executable} -I {HERE / 'serve.py'} --port {a.port}"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
try:
    print(srv.stdout.readline().strip(), flush=True)
    env, args = dict(os.environ), ["--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1", "--disable-background-networking"]
    if a.gpu:
        env |= {"LD_LIBRARY_PATH": "/usr/lib/wsl/lib", "GALLIUM_DRIVER": "d3d12"}
        args += ["--ignore-gpu-blocklist", "--enable-gpu", "--ozone-platform=x11", "--use-gl=angle", "--use-angle=gl"]
    with sync_playwright() as p:
        for page in a.pages:
            nm, _, extra = page.partition("?")
            url = (f"{BASE}/rtui/index.html?asset=eth&" if nm == "rtui" else f"{BASE}/dashboard/live/?") + f"bench={a.secs}&rate={a.rate}" + (f"&{extra}" if extra else "")
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
            ext = sorted({u for u in urls if not u.startswith((BASE, f"ws://127.0.0.1:{a.port}")) and not u.startswith("data:")})
            print(f"== {page} {'(그래픽 카드 · WSLg)' if a.gpu else '(헤드리스)'}")
            print(json.dumps(r.get("summary", r), ensure_ascii=False, indent=0))
            print("저장", r.get("saved"), "· 요청", len(urls), "건 · 바깥", ext or "0건", "· 페이지 안 기록", json.dumps(r.get("external"), ensure_ascii=False), flush=True)
finally:
    srv.terminate()
    try: srv.wait(5)
    except subprocess.TimeoutExpired: srv.kill()
    time.sleep(0.2)
    print("[서버]", srv.stdout.read().strip().replace("\n", " | ")[-600:])
