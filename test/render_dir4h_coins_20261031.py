#!/usr/bin/env python3
"""⑤ «4h 예측» 칸 SOL·XRP(2026-10-31) 실화면 확인 -- render_xrp_20261031.open_page 하네스(서버 터널 127.0.0.1:18787 + 로컬 app.js).

/api/dir4h 는 로컬 아티팩트(data/live/{sol,xrp}_dir4h_model.joblib)로 vision 패널 마지막 봉 응답을 만들어 route 로 준다(시각만 지금으로).
검사(1920·390 × eth·sol·xrp): 4h 칸 글자 = 응답 방향 · 요청 asset = 그 코인 · SOL 툴팁 «ETH 학습 모형» · XRP 툴팁 «검증 26-01~05 IC ≈ 0» ·
ETH 응답이 SOL 화면에 섞이지 않음 · 가로 스크롤 없음 · JS 오류 0.
    python3 -u test/render_dir4h_coins_20261031.py [--shot DIR]
"""
import json, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts")); sys.path.insert(0, str(HERE))
import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import live_eth_dir4h_20261015 as d4  # noqa: E402
from render_xrp_20261031 import json_route, open_page  # noqa: E402
from test_dir4h_coins_parity_20261031 import PANEL  # noqa: E402

PROBE = """() => { const c = [...document.querySelectorAll('.fc-c')].find((e) => (e.querySelector('.hz') || {}).textContent === '4h 예측');
  return { main: c ? c.querySelector('b').textContent : null, sub: c ? c.querySelector('.sub').textContent : null, tip: c ? c.title : null,
           gauge: !!(c && c.querySelector('.d4g')), hscroll: document.documentElement.scrollWidth > window.innerWidth + 1 }; }"""


def payloads():
    frames = {s: (lambda f: f[f.timestamp >= f.timestamp.max() - pd.Timedelta(days=130)])(pd.read_parquet(PANEL / f"{s}USDT.parquet")) for s in ["ETH"] + d4.COINS}
    out, now = {}, int(time.time() * 1000) // 300_000 * 300_000
    for coin in ("SOL", "XRP"):
        p = d4.payload_from_features(joblib.load(d4.ARTIFACTS[coin]), d4.build_features(d4.frames_for(frames, coin)))
        sh = now - p["decided_at_ms"]
        p["decided_at_ms"] = now; p["hist"] = [[t + sh, v] for t, v in p["hist"]]
        out[coin.lower()] = p
    return out


if __name__ == "__main__":
    from playwright.sync_api import sync_playwright
    shot = pathlib.Path(sys.argv[sys.argv.index("--shot") + 1]) if "--shot" in sys.argv else None
    P, asked, fails, errs, blocked = payloads(), [], [], [], []
    print({c: (p["pred_bp"], p["lo"], p["hi"], p["signal"]) for c, p in P.items()}, flush=True)
    eth_fake = {**P["sol"], "asset": "eth", "pred_bp": 999.0}   # 섞이면 «+999» 가 보인다
    body = lambda u: (asked.append(u), P.get(u.split("asset=")[1][:3], eth_fake) if "asset=" in u else eth_fake)[1]   # noqa: E731
    word = {1: "▲ 상승", -1: "▼ 하락", 0: "보류"}
    with sync_playwright() as pw:
        br = pw.chromium.launch()
        for coin in ("sol", "xrp", "eth"):
            for w, h in ((1920, 1080), (390, 844)):
                pg = open_page(br, w, h, coin, errs, blocked, [("**/api/dir4h*", json_route(body))])
                pg.evaluate("() => { dir4hLastFetchAt = 0; return refreshDir4h(); }"); pg.wait_for_timeout(1500)
                pg.evaluate("() => renderMarketCtx()"); pg.wait_for_timeout(500)
                r, tag = pg.evaluate(PROBE), f"{coin}·{w}"
                print(tag, json.dumps({k: (v[:90] if isinstance(v, str) else v) for k, v in r.items()}, ensure_ascii=False), flush=True)
                if coin != "eth":
                    p = P[coin]
                    ok = r["main"] == word[p["signal"]] and r["gauge"] and "999" not in (r["sub"] or "") and f"{abs(p['pred_bp']):.0f}bp" in (r["sub"] or "")
                    ok = ok and ("ETH 학습 모형" in r["tip"] if coin == "sol" else "검증 26-01~05 IC ≈ 0" in r["tip"])
                else:
                    ok = "+999" in (r["sub"] or "")   # ETH 화면은 ETH 응답을 그대로 쓴다(양성 대조)
                if not ok or r["hscroll"]:
                    fails.append(f"[{tag}] {r}")
                if shot:
                    pg.screenshot(path=str(shot / f"dir4h_{coin}_{w}.png"), full_page=(w == 390))
                pg.close()
        br.close()
    bad_req = [u for u in asked if "asset=" not in u]
    fails += [f"JS 오류 {e}" for e in errs] + [f"막힌 요청 {u}" for u in blocked] + [f"asset 없는 요청 {u}" for u in bad_req]
    print("\n".join(fails) or "render dir4h coins PASS")
    sys.exit(1 if fails else 0)
