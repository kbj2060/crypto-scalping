"""15분·1시간 방향(scripts/live_eth_dir_short_20261028.py) «REST 꼴 5분봉 → 피쳐 65(4h 와 같은 build_features) → LightGBM 시드 앙상블 → 화면 payload»
정답지 -- 65열판 코인(SOL·BTC·XRP · 호가 깊이 안 씀) 끝에서 끝까지 (2026-11-21 · tests/dirshort.rs).

원본 함수 그대로: load_art(코인 → 지평마다 TABLE 의 학습 코인 65열판 모형) · d4.build_features · payload_from_features(최근 13봉).
안전(소스 확인): 원본은 import 때 sqlite3·numpy·pandas 와 4h 모듈만 -- live_payload·rest_frames(REST)는 부르지 않는다. 65열판은 sqlite 를 안 연다.
🔴대역 두 가지(개발기에 없음): ① 모델 경로 ART65 를 메인 체크아웃 data/live/ 로(읽기만) ② 표본 밖 분위 씨앗은 ETH 것(eth_dirshort_oos_seed.parquet)만
   개발기에 있어 SOL·BTC·XRP 모두 그것을 쓴다(ds._OOS 캐시에 미리 넣음) -- 분위·문턱 «식» 대조이지 그 코인의 실제 문턱이 아니다.
ETH 75열판(호가 깊이 10열 · sqlite)은 모델이 개발기에 없어 못 한다.

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
    qr = np.random.default_rng(5)                                    # np.quantile 식 자체(반올림 전) -- 길이·분위 여러 개
    fx["quant"] = [[G.bits(np.sort(x)), G.bits([q]), G.bits([np.quantile(x, q)])] for x in (qr.normal(0, 3, int(n)) for n in qr.integers(2, 500, 300)) for q in (.2, .8, .5, .1, .9)]
    p = OUT / "dirshort_fixture.json"; p.write_text(json.dumps(fx)); print(p.name, f"{p.stat().st_size:,} 바이트 · 벌 {len(fx['sets'])}")


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
