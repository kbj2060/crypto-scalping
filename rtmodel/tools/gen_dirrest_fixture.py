"""4h·15분/1시간 방향 «REST 요청 열 → 캐시 이어 붙이기 → payload → 적중 기록(pred_record)» 정답지 (2026-11-21 · tests/dirrest.rs).

서버 pred_record_loop(dashboard/server.py:2841-2863)가 5분마다 하는 일을 원본 함수로 흉내 낸다: 코인마다 dir_short → dir4h 의
`live_payload`(= rest_frames + daily_closes + build_features + payload_from_features) → pred_record.record_short / record_h4.
REST 는 원본 `_get` 을 가짜로 갈아 끼운다(합성 거래소 · 요청과 응답을 그대로 적음) · 시계(time.time)도 가짜. 루프 자체·서버·main 은 띄우지 않는다.
🔴서버는 4h 를 `scripts.live_eth_dir4h_20261015`, 15분·1시간은 ds 안의 `live_eth_dir4h_20261015` 로 불러 캐시가 둘이다 -- 여기도 4h 용 모듈을 따로 읽는다.
모델·분위 씨앗 대역은 gen_dirshort_fixture.py 와 같다(ETH 판 없음 · 분위 씨앗은 ETH 것).

    python rtmodel/tools/gen_dirrest_fixture.py      # → tests/fixtures/dirrest_fixture.json (커밋 안 함)
"""
import importlib.util
import json
import sys
import tempfile
import types
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "dashboard")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_eth_dir_short_20261028 as ds  # noqa: E402
import pred_record as PR  # noqa: E402
import gen_dir4h_fixture as G  # noqa: E402

LIVE = Path("/home/kbj20/crypto-scalping/data/live")
OUT = ROOT / "rtmodel" / "tests" / "fixtures"
BAR, DAY = 300_000, 86_400_000
EPS = {"openInterestHist": "sumOpenInterest", "topLongShortPositionRatio": "longShortRatio", "topLongShortAccountRatio": "longShortRatio",
       "globalLongShortAccountRatio": "longShortRatio", "takerlongshortRatio": "buySellRatio"}


class Exchange:
    """합성 거래소: 심볼마다 5분봉(형성 중 봉 포함) · metrics · 1일봉. 값은 시각의 결정적 함수(같은 요청 = 같은 응답)."""
    def __init__(self, t0, t1, late=()):
        self.t0, self.t1, self.cache, self.late = t0, t1, {}, set(late)   # late = metrics 가 한 봉 늦게 올라오는 시각들(봉 마감 직후 REST 가 아직 안 올림 → nan_feats)

    def series(self, sym):
        if sym not in self.cache:
            rng = np.random.default_rng(int.from_bytes(sym.encode().ljust(8, b"_")[:8], "little"))
            n = (self.t1 - self.t0) // BAR + 1
            px = 10 ** rng.uniform(-0.5, 4.5) * np.exp(np.cumsum(rng.normal(0, 1.5e-3, n)))
            self.cache[sym] = dict(px=px, v=np.exp(rng.normal(8, 1, n)), tb=rng.uniform(.3, .7, n), tr=np.exp(rng.normal(6, .5, n)),
                                   m={ep: np.exp(np.cumsum(rng.normal(0, 2e-3, n))) * (1e6 if ep == "openInterestHist" else 1.0) for ep in EPS})
        return self.cache[sym]

    def get(self, now_ms, path, params):
        sym = params["symbol"][:-4]; S = self.series(sym); lim = int(params["limit"]); end = int(params.get("endTime", now_ms))
        if path == "/fapi/v1/klines" and params["interval"] == "5m":
            last = min(end, now_ms) // BAR * BAR; first = max(self.t0, last - (lim - 1) * BAR)
            out = []
            for t in range(first, last + 1, BAR):
                i = (t - self.t0) // BAR; c = S["px"][i]; o = S["px"][max(i - 1, 0)]
                partial = t + BAR - 1 >= now_ms                        # 형성 중 봉: 지금까지만(값을 조금 다르게)
                c = c * (1.0003 if partial else 1.0)
                out.append([t, f"{o:.6g}", f"{max(o, c) * 1.001:.6g}", f"{min(o, c) * 0.999:.6g}", f"{c:.6g}", f"{S['v'][i]:.3f}", t + BAR - 1, "0",
                            int(S["tr"][i]), f"{S['v'][i] * S['tb'][i]:.3f}", "0", "0"])
            return out
        if path == "/fapi/v1/klines" and params["interval"] == "1d":
            d1 = now_ms // DAY * DAY
            return [[d, "1", "1", "1", f"{S['px'][(d - self.t0) // BAR] if d >= self.t0 else S['px'][0]:.6g}", "1", d + DAY - 1, "0", 1, "1", "0", "0"]
                    for d in range(d1 - (lim - 1) * DAY, d1 + 1, DAY)]
        ep = path.rsplit("/", 1)[1]
        last = min(end, now_ms - BAR * (2 if now_ms in self.late else 1)) // BAR * BAR; first = max(self.t0, last - (lim - 1) * BAR)
        return [{"symbol": params["symbol"], EPS[ep]: f"{S['m'][ep][(t - self.t0) // BAR]:.6f}", "timestamp": t} for t in range(first, last + 1, BAR)]


def module(name):
    sp = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "live_eth_dir4h_20261015.py")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def main():
    warnings.filterwarnings("ignore")
    d4h = module("d4_server")                                          # 서버가 4h 에 쓰는 모듈(캐시 따로)
    ds.ART65 = {s: LIVE / f"{s.lower()}_dirshort65_model.joblib" for s in ds.ART65}
    S = pd.read_parquet(LIVE / "eth_dirshort_oos_seed.parquet"); Q = {nm: np.sort(S[f"p{nm}"].to_numpy(float)) for nm in ds.NAMES.values()}
    arts4 = {c: joblib.load(LIVE / f"{c.lower()}_dir4h_model.joblib") for c in ("SOL", "XRP")}
    artss = {}
    for c in ("BTC", "SOL", "XRP"):
        ds._OOS[c] = Q; artss[c] = ds.load_art(c)
    T0 = int(pd.Timestamp("2026-11-20 13:35:20", tz="UTC").timestamp()) * 1000
    ex = Exchange(T0 - 45 * DAY, T0 + 3 * DAY, late=(T0 + 600_000,))
    rec_dir = Path(tempfile.mkdtemp()); PR._seen.clear()
    steps = []
    # 5분 루프(+20초) · 셋째는 metrics 한 봉 늦음(빈 피쳐 → 그 봉 기록 안 함) · 19.5시간 뒤 = 아직 꼬리 · 그 21시간 뒤 = «20시간 넘게 안 불림» → 처음부터
    for now in (T0, T0 + 300_000, T0 + 600_000, T0 + 600_000 + 19 * 3600_000 + 1_800_000, T0 + 600_000 + 40 * 3600_000 + 1_800_000):
        for coin in ("BTC", "SOL", "XRP"):
            for kind in ("dir_short", "dir4h"):
                if kind == "dir4h" and coin not in arts4:
                    continue
                mod = ds.d4 if kind == "dir_short" else d4h
                reqs = []

                def fake_get(path, params, now=now, reqs=reqs):
                    r = ex.get(now, path, params); reqs.append([path, [[k, str(v)] for k, v in params.items()], r]); return r
                clock = types.SimpleNamespace(time=lambda now=now: now / 1000)
                mod._get, mod.time, ds.time = fake_get, clock, clock
                pay = ds.live_payload(artss[coin]) if kind == "dir_short" else d4h.live_payload(arts4[coin])
                k = mod._CACHE[f"{coin}_k"].set_index("timestamp").close            # pred_record_safe 와 같은 종가
                (PR.record_short if kind == "dir_short" else PR.record_h4)(coin, pay, k, rec_dir)
                steps.append({"kind": kind, "coin": coin, "now_ms": now, "requests": reqs, "payload": G.enc(pay)})
                print(kind, coin, now, "REST", len(reqs), "· pred", pay.get("pred") or pay.get("pred_bp"), flush=True)
    recs = {p.name: p.read_text() for p in sorted(rec_dir.glob("*.jsonl"))}
    fx = {"oos": {nm: G.bits(q) for nm, q in Q.items()}, "steps": steps, "records": recs}
    p = OUT / "dirrest_fixture.json"; p.write_text(json.dumps(fx)); print(p.name, f"{p.stat().st_size:,} 바이트 · 단계 {len(steps)} · 기록 {[(k, v.count(chr(10))) for k, v in recs.items()]}")


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
