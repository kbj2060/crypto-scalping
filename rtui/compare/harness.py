"""비교 하네스 공용: 오프라인 헤드리스 크로미움(모든 요청을 가로채 디스크에서 준다 · 거래소·서버 접속 없음) · 같은 시계 · 가짜 WebSocket/EventSource · 캔버스 그리기 기록.
capture.py · ops.py · feed.py 가 같이 쓴다."""
import pathlib
from urllib.parse import urlparse

T_FX = 1795156790   # 하네스 시계(합성 마지막 봉 1795156500 안)
TZ = "Asia/Seoul"
LAUNCH = ["--host-resolver-rules=MAP * ~NOTFOUND", "--proxy-server=127.0.0.1:9", "--disable-background-networking"]
# 지금 화면이 웹 글꼴(jsdelivr Pretendard Variable)을 받는 자리에 같은 글꼴의 Bold 한 벌만 준다(오프라인 -- 굵기 400·800 도 Bold 로 그려진다)
FONT_CSS = "@font-face{font-family:'Pretendard Variable';src:url(http://fx.invalid/font/p.ttf) format('truetype');font-weight:45 920;}"
CT = {".js": "application/javascript", ".css": "text/css", ".html": "text/html", ".png": "image/png", ".wasm": "application/wasm", ".json": "application/json"}

INIT = r"""(() => {
  const OFF = %d * 1000 - Date.now(); const RD = Date;
  class D extends RD { constructor(...a) { if (a.length) super(...a); else super(RD.now() + OFF); } static now() { return RD.now() + OFF; } }
  window.Date = D;
  window.WebSocket = class { constructor(u) { this.url = u; this.readyState = 3; (window.__wsAll = window.__wsAll || []).push(this); } send() {} close() {} addEventListener() {} };   // 열린 것처럼 남는 죽은 소켓(바이낸스에 안 붙는다) -- ops·feed 가 onmessage 를 직접 부른다
  window.EventSource = class { constructor(u) { this.url = u; this.readyState = 1; } addEventListener() {} removeEventListener() {} close() { this.readyState = 2; } };
  // 겹침 선(VWAP·SMA·스윙·OB 등)은 이번 범위 밖 -- 사용자 스위치(카드 «지표»)로 끈다. POC 선만 켠다.
  try { localStorage.setItem('mcLine', JSON.stringify({ poc: true, vwap: false, veto: false, sma50: false, sw1h: false, sw30m: false, sw30mOn: 1, ob: false, fvg: false, orb: false, cci: false, rsi2: false, rdiv: false, srl: false, bb: false, wvwap: false })); } catch (e) {}
  // 캔버스 그리기 기록(좌표 대조용) -- window.__rec 이 켜져 있을 때만. 경로 명령(moveTo·lineTo·roundRect·rect·arc)은 다음 fill·stroke 에 붙인다.
  window.__rec = null; const P = CanvasRenderingContext2D.prototype;
  const tag = (ctx) => ctx.canvas.classList.contains('fp-cv') ? (ctx.canvas === ctx.canvas.parentNode.querySelector('.fp-cv') ? 'bg' : 'fg') : null;
  const r3 = (v) => typeof v === 'number' ? Math.round(v * 1000) / 1000 : (typeof v === 'string' ? v : null);
  let path = [];
  // cvUnion(8199)은 Path2D 에 사각형을 모아 ctx.fill(p) 한 번 -- 그 사각형도 기록한다
  const PR = Path2D.prototype.rect; Path2D.prototype.rect = function (...a) { (this.__r = this.__r || []).push(['rect', ...a.map(r3)]); return PR.apply(this, a); };
  // 청산 동그라미 1주일 창(cvUnion 색별 한 칠 = moveTo + arc) -- 원도 기록한다(다섯째 조각)
  const PA = Path2D.prototype.arc; Path2D.prototype.arc = function (...a) { (this.__r = this.__r || []).push(['arc', ...a.map(r3)]); return PA.apply(this, a); };
  const ob = P.beginPath; P.beginPath = function () { if (window.__rec && tag(this)) path = []; return ob.apply(this, arguments); };
  for (const k of ['moveTo', 'lineTo', 'roundRect', 'rect', 'arc', 'closePath']) { const o = P[k]; P[k] = function (...a) { if (window.__rec && tag(this)) path.push([k, ...a.map(r3)]); return o.apply(this, a); }; }
  for (const k of ['fillRect', 'fillText', 'strokeRect', 'stroke', 'fill']) { const o = P[k]; P[k] = function (...a) {
    const L = window.__rec && tag(this);
    if (L) { const r = { k, L, a: a.map(r3), s: this.fillStyle, st: this.strokeStyle, al: this.globalAlpha, f: this.font, lw: this.lineWidth, ta: this.textAlign };
             if (k === 'stroke' || k === 'fill') { r.p = a[0] instanceof Path2D ? (a[0].__r || []).slice() : path.slice(); r.dash = this.getLineDash(); r.join = this.lineJoin; }
             window.__rec.push(r); }
    return o.apply(this, a); }; }
})();""" % T_FX


def make_route(wt, api, font=None):
    """api = {경로: 파일} -- /dashboard/live/* · /rtui/* 는 작업 디렉터리에서, 그 밖은 404(바깥 주소는 abort)"""
    wt = pathlib.Path(wt)
    LIVE, FONT = wt / "dashboard/live", font or wt / "rtui/fonts/Pretendard-Bold.ttf"
    def route(r):
        u = urlparse(r.request.url)
        if r.request.method != "GET":
            return r.abort()
        if u.hostname == "cdn.jsdelivr.net" and "pretendard" in u.path:
            return r.fulfill(body=FONT_CSS, content_type="text/css", headers={"Access-Control-Allow-Origin": "*"})
        if u.hostname == "fonts.googleapis.com":
            return r.fulfill(body="", content_type="text/css")
        if u.hostname != "fx.invalid":
            return r.abort()
        if u.path == "/font/p.ttf":
            return r.fulfill(body=FONT.read_bytes(), content_type="font/ttf")
        if u.path.startswith("/dashboard/live/"):
            f = LIVE / (u.path[len("/dashboard/live/"):] or "index.html")
            if f.is_file() and f.name != "sw.js":
                return r.fulfill(body=f.read_bytes(), content_type=CT.get(f.suffix, "application/octet-stream"))
            return r.fulfill(status=404, body="")
        if u.path.startswith("/rtui/"):
            f = wt / "rtui" / u.path[len("/rtui/"):]
            if f.is_file():
                return r.fulfill(body=f.read_bytes(), content_type=CT.get(f.suffix, "application/octet-stream"))
            return r.fulfill(status=404, body="")
        if u.path in api:
            return r.fulfill(body=pathlib.Path(api[u.path]).read_bytes(), content_type="application/json")
        return r.fulfill(status=404, body="{}", content_type="application/json")


    return route
