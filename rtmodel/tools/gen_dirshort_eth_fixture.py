"""ETH 15분·1시간(75열판) 서빙 길 끝에서 끝까지 정답지 (2026-11-22 · tests/dirrest.rs eth_dir_short_handler).
원본 그대로: ds.live_payload(art) = d4.rest_frames(가짜 _get · 합성 거래소 = gen_dirrest_fixture.Exchange) + build_features + payload_from_features
(+ 호가 깊이 BOOK_DB = 임시 sqlite · warm·빈 칸·없는 봉 포함). 모델 = 서버 사본 tmp/rtmodel_models_20261121/eth_dirshort_model.joblib(md5 같음 · data/live 로 옮기지 않음).
표본 밖 분위 씨앗 = 개발기 eth_dirshort_oos_seed.parquet(서버와 md5 같음 3885488c). 회차 둘: 처음(31일치) → 5분 뒤(꼬리).
    (tools/server_env.sh 의 환경) python rtmodel/tools/gen_dirshort_eth_fixture.py      # → tests/fixtures/dirshort_eth_fixture.json · ds_eth.json(gen_dirshort_fixture 와 같은 꼴)
"""
import json
import sqlite3
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
import gen_dir4h_fixture as G  # noqa: E402
from gen_dirrest_fixture import Exchange, BAR, DAY  # noqa: E402

LIVE = Path("/home/kbj20/crypto-scalping/data/live")
OUT = ROOT / "rtmodel" / "tests" / "fixtures"


def main():
    warnings.filterwarnings("ignore")
    ds.ARTIFACT = Path("/home/kbj20/crypto-scalping/tmp/rtmodel_models_20261121/eth_dirshort_model.joblib")
    art = joblib.load(ds.ARTIFACT)
    S = pd.read_parquet(LIVE / "eth_dirshort_oos_seed.parquet"); Q = {nm: np.sort(S[f"p{nm}"].to_numpy(float)) for nm in ds.NAMES.values()}
    ds._OOS["ETH"] = Q
    T0 = int(pd.Timestamp("2026-11-20 13:35:20", tz="UTC").timestamp()) * 1000
    ex = Exchange(T0 - 45 * DAY, T0 + 3 * DAY)
    db = Path(tempfile.mkdtemp()) / "ethusdt.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE bars (t INTEGER PRIMARY KEY, warm INTEGER, " + ", ".join(f'"{c}" REAL' for c in ds.BOOKF) + ")")
    rng, book = np.random.default_rng(75), {}
    for k, t in enumerate(range(T0 // BAR * BAR - 40 * BAR, T0 // BAR * BAR + 3 * BAR, BAR)):
        if k == 20:
            continue                                                               # 없는 봉
        v = [float(x) for x in rng.normal(0, 1, len(ds.BOOKF))]
        if k == 25:
            v[2] = None                                                            # 빈 칸
        warm = int(k == 30)
        con.execute(f"INSERT INTO bars VALUES ({', '.join('?' * (2 + len(v)))})", [t, warm, *v]); book[str(t)] = [warm, *v]
    con.commit(); con.close()
    ds.BOOK_DB = db
    steps = []
    for now in (T0, T0 + 300_000):
        reqs = []

        def fake_get(path, params, now=now, reqs=reqs):
            r = ex.get(now, path, params); reqs.append([path, [[k, str(v)] for k, v in params.items()], r]); return r
        clock = types.SimpleNamespace(time=lambda now=now: now / 1000)
        ds.d4._get, ds.d4.time, ds.time = fake_get, clock, clock
        pay = ds.live_payload(art)
        steps.append({"now_ms": now, "requests": reqs, "payload": G.enc(pay)})
        print("ETH dir_short", now, "REST", len(reqs), "book", pay["book"], "pred", pay["pred"], "hist", len(pay["hist"]), flush=True)
    mj = {"coin": "ETH", "month": art["month"], "src": art.get("src"), "sources": [ds.ARTIFACT.name],
          "models": {str(h): [[{"txt": m.booster_.model_to_string(), "scale": G.bits([s])[0]} for m, s in ms] for ms in seeds] for h, seeds in art["models"].items()}}
    (OUT / "ds_eth.json").write_text(json.dumps(mj))
    p = OUT / "dirshort_eth_fixture.json"
    p.write_text(json.dumps({"oos": {nm: G.bits(q) for nm, q in Q.items()}, "book": book, "steps": steps}))
    print(p.name, f"{p.stat().st_size:,}")


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
