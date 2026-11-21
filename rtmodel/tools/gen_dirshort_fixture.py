"""15분·1시간 방향(scripts/live_eth_dir_short_20261028.py) «REST 꼴 5분봉 → 피쳐 65(4h 와 같은 build_features) → LightGBM 시드 앙상블 → 화면 payload»
정답지 -- 65열판 코인(SOL·BTC·XRP · 호가 깊이 안 씀) 끝에서 끝까지 (2026-11-21 · tests/dirshort.rs).

원본 함수 그대로: load_art(코인 → 지평마다 TABLE 의 학습 코인 65열판 모형) · d4.build_features · payload_from_features(최근 13봉).
안전(소스 확인): 원본은 import 때 sqlite3·numpy·pandas 와 4h 모듈만 -- live_payload·rest_frames(REST)는 부르지 않는다. 65열판은 sqlite 를 안 연다.
🔴대역 두 가지(개발기에 없음): ① 모델 경로 ART65 를 메인 체크아웃 data/live/ 로(읽기만) ② 표본 밖 분위 씨앗은 ETH 것(eth_dirshort_oos_seed.parquet)만
   개발기에 있어 SOL·BTC·XRP 모두 그것을 쓴다(ds._OOS 캐시에 미리 넣음) -- 분위·문턱 «식» 대조이지 그 코인의 실제 문턱이 아니다.
ETH 75열판(호가 깊이 10열 · sqlite)은 서버 모델 사본(tmp/rtmodel_models_20261121 · md5 같음)과 임시 sqlite 로 4벌(2026-11-21 추가).

    python rtmodel/tools/gen_dirshort_fixture.py      # → tests/fixtures/dirshort_fixture.json · ds_{sol,btc,xrp}.json (커밋 안 함)
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_eth_dir_short_20261028 as ds  # noqa: E402
import gen_dir4h_fixture as G  # noqa: E402  -- 같은 REST 꼴 합성기

LIVE = Path("/home/kbj20/crypto-scalping/data/live")
OUT = ROOT / "rtmodel" / "tests" / "fixtures"


def enc(o):
    return G.enc(o)


def main():
    warnings.filterwarnings("ignore")
    ds.ART65 = {s: LIVE / f"{s.lower()}_dirshort65_model.joblib" for s in ds.ART65}
    S = pd.read_parquet(LIVE / "eth_dirshort_oos_seed.parquet")
    Q = {nm: np.sort(S[f"p{nm}"].to_numpy(float)) for nm in ds.NAMES.values()}
    fx = {"oos": {nm: G.bits(q) for nm, q in Q.items()}, "sets": []}
    rng = np.random.default_rng(20261122)
    for coin in ("SOL", "BTC", "XRP"):
        ds._OOS[coin] = Q
        art = ds.load_art(coin)
        mj = {"coin": coin, "month": art["month"], "src": art["src"],
              "models": {str(h): [[{"txt": m.booster_.model_to_string(), "scale": G.bits([s])[0]} for m, s in ms] for ms in seeds] for h, seeds in art["models"].items()}}
        (OUT / f"ds_{coin.lower()}.json").write_text(json.dumps(mj))
        for name, end in (("normal", "2026-11-20 13:35"), ("gaps", "2026-11-21 00:20"), ("zero_vol", "2026-11-22 07:40")):
            frames, dc = G.case(rng, "SOL" if coin == "SOL" else "XRP", name, end)   # 합성기는 대상 이름 SOL·XRP 만 안다 -- BTC 는 XRP 꼴 대상을 BTC 자리로
            if coin == "BTC":
                frames["XRP"], frames["BTC"] = frames["BTC"], frames["XRP"]
            F = ds.d4.build_features(ds.d4.frames_for(frames, coin), dc).tail(13)
            pay = ds.payload_from_features(art, F)
            raw = ds.predict(art, F[ds.d4.FEATS].to_numpy(np.float32))       # 반올림 전 예측(payload 는 4자리로 반올림해 끝자리를 가린다)
            fx["sets"].append({"coin": coin, "name": name, "frames": {s: {"ts": (f.timestamp.astype("int64") // 10**6).tolist(), **{c: G.bits(f[c]) for c in G.COLS + ds.d4.MCOLS}} for s, f in frames.items()},
                               "dc": [[int(t.value // 10**6) for t in dc.index], G.bits(dc.to_numpy())], "payload": enc(pay), "raw": {ds.NAMES[h]: G.bits(v) for h, v in raw.items()}})
            print(coin, name, "pred", pay["pred"], "thr", pay["thr"], flush=True)
    # ── ETH 75열판(서버 모델 사본 · 호가 깊이 sqlite 10열 -- 임시 sqlite 로 warm·빈 칸·없는 봉·지금 봉 warm·전부 없음)
    import sqlite3, tempfile, joblib
    ds.ARTIFACT = Path("/home/kbj20/crypto-scalping/tmp/rtmodel_models_20261121/eth_dirshort_model.joblib")   # 서버와 md5 같음 · data/live 로 옮기지 않는다
    art = joblib.load(ds.ARTIFACT)
    ds._OOS["ETH"] = Q
    mj = {"coin": "ETH", "month": art["month"], "src": art.get("src"),
          "models": {str(h): [[{"txt": m.booster_.model_to_string(), "scale": G.bits([s])[0]} for m, s in ms] for ms in seeds] for h, seeds in art["models"].items()}}
    (OUT / "ds_eth.json").write_text(json.dumps(mj))
    for name, end, kind in (("normal", "2026-11-20 13:35", "ok"), ("gaps", "2026-11-21 00:20", "holes"), ("zero_vol", "2026-11-22 07:40", "now_warm"), ("normal", "2026-11-23 01:05", "none")):
        frames, dc = G.case(rng, "SOL", name, end)
        frames["ETH"], frames["SOL"] = frames["SOL"], frames["ETH"]                 # 합성기 대상(9,000봉·metrics)을 ETH 자리로
        F = ds.d4.build_features(ds.d4.frames_for(frames, "ETH"), dc).tail(13)
        bars = [int(pd.Timestamp(x).value // 10**6) for x in F.ts]
        db = Path(tempfile.mkdtemp()) / "ethusdt.sqlite"
        con = sqlite3.connect(db)
        con.execute(f"CREATE TABLE bars (t INTEGER PRIMARY KEY, warm INTEGER, {', '.join(f'{c!r} REAL' for c in ds.BOOKF)})".replace("'", '"'))
        book = {}
        for j, b in enumerate(bars):
            if kind == "none" or (kind == "holes" and j == 4):
                continue                                                           # 없는 봉
            v = [float(x) for x in rng.normal(0, 1, len(ds.BOOKF))]
            warm = int((kind == "holes" and j == 7) or (kind == "now_warm" and j == len(bars) - 1))
            if kind == "holes" and j == 9:
                v[3] = None                                                        # 빈 칸
            con.execute(f"INSERT INTO bars VALUES ({', '.join('?' * (2 + len(v)))})", [b, warm, *v])
            book[str(b)] = [warm, *v]
        con.commit(); con.close()
        ds.BOOK_DB = db
        pay = ds.payload_from_features(art, F)
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = [ds.book_row(con, b) for b in bars]; con.close()
        ok = [i for i, (bk, _) in enumerate(rows) if bk is not None]
        raw = ds.predict(art, np.hstack([F.iloc[ok][ds.d4.FEATS].to_numpy(np.float32), np.vstack([rows[i][0] for i in ok])])) if ok else {}
        fx["sets"].append({"ok_rows": ok, "coin": "ETH", "name": f"{name}/{kind}", "book": book, "frames": {s: {"ts": (f.timestamp.astype("int64") // 10**6).tolist(), **{c: G.bits(f[c]) for c in G.COLS + ds.d4.MCOLS}} for s, f in frames.items()},
                           "dc": [[int(t.value // 10**6) for t in dc.index], G.bits(dc.to_numpy())], "payload": enc(pay), "raw": {ds.NAMES[h]: G.bits(v) for h, v in raw.items()}})
        print("ETH", name, kind, "pred", pay["pred"], "book", pay["book"], "hist", len(pay["hist"]), flush=True)
    qr = np.random.default_rng(5)                                    # np.quantile 식 자체(반올림 전) -- 길이·분위 여러 개
    fx["quant"] = [[G.bits(np.sort(x)), G.bits([q]), G.bits([np.quantile(x, q)])] for x in (qr.normal(0, 3, int(n)) for n in qr.integers(2, 500, 300)) for q in (.2, .8, .5, .1, .9)]
    p = OUT / "dirshort_fixture.json"; p.write_text(json.dumps(fx)); print(p.name, f"{p.stat().st_size:,} 바이트 · 벌 {len(fx['sets'])}")


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
