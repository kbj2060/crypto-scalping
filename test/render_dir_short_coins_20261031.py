#!/usr/bin/env python3
"""⑤ «15분 예측»·«1시간 예측» 칸 BTC·SOL·XRP(2026-10-31) 실화면 확인 -- render_xrp_20261031.open_page 하네스(서버 터널 127.0.0.1:18787 + 로컬 app.js).

/api/dir-short 는 로컬 아티팩트(data/live/{eth,btc,xrp}_dirshort65_model.joblib)로 vision 패널 마지막 13봉 응답을 만들어 route 로 준다(시각만 지금으로).
검사(1920·390 × btc·sol·xrp·eth): 두 칸 글자 = 응답 방향 · 아래 줄 = 응답 분위 · 툴팁 = 코인별 모델 줄·«분위만»(점수 숫자 없음) ·
요청 asset = 그 코인 · ETH 응답이 다른 코인 화면에 섞이지 않음(ETH 가짜 = 분위 99) · 가로 스크롤 없음 · JS 오류 0.
    python3 -u test/render_dir_short_coins_20261031.py [--shot DIR]
"""
import json, pathlib, re, sys, time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts")); sys.path.insert(0, str(HERE))
import pandas as pd  # noqa: E402
import live_eth_dir4h_20261015 as d4  # noqa: E402
import live_eth_dir_short_20261028 as ds  # noqa: E402
from render_xrp_20261031 import json_route, open_page  # noqa: E402
from test_dir_short_coins_parity_20261031 import PANEL  # noqa: E402

PROBE = """() => { const c = (hz) => [...document.querySelectorAll('.fc-c')].find((e) => (e.querySelector('.hz') || {}).textContent === hz);
  const g = (e) => e ? { main: e.querySelector('b').textContent, sub: e.querySelector('.sub').textContent, tip: e.title, gauge: !!e.querySelector('.d4g') } : null;
  return { m15: g(c('15분 예측')), h1: g(c('1시간 예측')), hscroll: document.documentElement.scrollWidth > window.innerWidth + 1 }; }"""
NOTE = {"btc": "15분 = BTC 학습 · 1시간 = ETH 로 학습한 모형", "sol": "15분·1시간 = ETH 로 학습한 모형", "xrp": "15분 = XRP 학습 · 1시간 = ETH 로 학습한 모형"}


def payloads():
    frames = {s: (lambda f: f[f.timestamp >= f.timestamp.max() - pd.Timedelta(days=130)])(pd.read_parquet(PANEL / f"{s}USDT.parquet")) for s in ["ETH"] + d4.COINS}
    out, now = {}, int(time.time() * 1000) // 300_000 * 300_000
    for coin in ds.TABLE:
        f = d4.frames_for(frames, coin); dc = f["ETH"].set_index("timestamp").close.groupby(lambda x: x.normalize()).last()   # 일봉 = 서빙 daily_closes 자리
        p = ds.payload_from_features(ds.load_art(coin), d4.build_features(f, dc).tail(13))
        sh = now - p["decided_at_ms"]
        p["decided_at_ms"] = now; p["hist"] = [[h[0] + sh] + h[1:] for h in p["hist"]]
        out[coin.lower()] = p
    return out


def sub_of(q):
    s = f"하위 {round(q['pct'])}%" if q["pct"] < 50 else f"상위 {round(100 - q['pct'])}%"
    return s if q["signal"] else s + (f" · 롱까지 {round(80 - q['pct'])}" if q["pct"] >= 50 else f" · 숏까지 {round(q['pct'] - 20)}")


if __name__ == "__main__":
    from playwright.sync_api import sync_playwright
    shot = pathlib.Path(sys.argv[sys.argv.index("--shot") + 1]) if "--shot" in sys.argv else None
    P, asked, fails, errs, blocked = payloads(), [], [], [], []
    print({c: {nm: (q["pct"], q["signal"]) for nm, q in p["pred"].items()} for c, p in P.items()}, flush=True)
    fake = {**P["sol"], "asset": "eth", "pred": {nm: {"score": 9.0, "pct": 99.0, "signal": 1} for nm in ("15m", "1h")}}   # 섞이면 «상위 1%»
    body = lambda u: (asked.append(u), P.get(u.split("asset=")[1][:3], fake) if "asset=" in u else fake)[1]   # noqa: E731
    word = {1: "▲ 상승", -1: "▼ 하락", 0: "보류"}
    with sync_playwright() as pw:
        br = pw.chromium.launch()
        for coin in ("btc", "sol", "xrp", "eth"):
            for w, h in ((1920, 1080), (390, 844)):
                pg = open_page(br, w, h, coin, errs, blocked, [("**/api/dir-short*", json_route(body))])
                pg.evaluate("() => { dirShortLastFetchAt = 0; return refreshDirShort(); }"); pg.wait_for_timeout(1500)
                pg.evaluate("() => renderMarketCtx()"); pg.wait_for_timeout(500)
                r, tag = pg.evaluate(PROBE), f"{coin}·{w}"
                print(tag, json.dumps({k: ({kk: (vv[:80] if isinstance(vv, str) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v)
                                       for k, v in r.items()}, ensure_ascii=False), flush=True)
                ok = bool(r["m15"] and r["h1"])
                for nm, k in (("15m", "m15"), ("1h", "h1")):
                    c = r[k] or {}
                    if coin != "eth":
                        q = P[coin]["pred"][nm]
                        ok = ok and c.get("main") == word[q["signal"]] and c.get("sub") == sub_of(q) and c.get("gauge")
                        ok = ok and NOTE[coin] in c.get("tip", "") and not re.search(r"점수 [+-]?\d+\.\d", c.get("tip", ""))
                    else:
                        ok = ok and c.get("sub") == "상위 1%"   # ETH 화면은 ETH 응답을 그대로 쓴다(양성 대조)
                if not ok or r["hscroll"]:
                    fails.append(f"[{tag}] {r}")
                if shot:
                    pg.screenshot(path=str(shot / f"dirshort_{coin}_{w}.png"), full_page=(w == 390))
                pg.close()
        br.close()
    fails += [f"JS 오류 {e}" for e in errs] + [f"막힌 요청 {u}" for u in blocked] + [f"asset 없는 요청 {u}" for u in asked if "asset=" not in u]
    print("\n".join(fails) or "render dir-short coins PASS")
    sys.exit(1 if fails else 0)
