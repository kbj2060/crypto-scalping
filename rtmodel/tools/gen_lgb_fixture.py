"""15분·1시간 방향(scripts/live_eth_dir_short_20261028.py) LightGBM 추론 정답지 (2026-11-21 · tests/lgb.rs).

부스터는 `model_to_string()` 텍스트로 내보낸다(Rust 가 그 텍스트를 읽어 나무를 평가). 모델 = 개발기에 있는 65열판(메인 체크아웃 data/live/
{eth,btc,xrp}_dirshort65_model.joblib -- 읽기만 · ETH 75열판은 개발기에 없다). 원본 모듈은 import 때 numpy·pandas·sqlite3 와 4h 모듈만 부른다
(REST·sqlite 접속은 함수 안) -- 여기서는 `predict` 만 부른다.
입력 행 = float32(원본 F[FEATS].to_numpy(np.float32)): 피쳐 범위 안 무작위 · NaN·0·−0·±inf·±1e-36·±1e-35·±2e-35·f32 최대 ·
무작위 나무 노드의 임계값 그 자리와 f32 이웃(그 행이 실제로 지나는 노드). 결측·0 처리 규칙은 이 비트 대조로 정한다(기억에 기대지 않는다).

    python rtmodel/tools/gen_lgb_fixture.py         # → tests/fixtures/lgb_{eth,btc,xrp}.json (커밋 안 함)
"""
import json
import re
import sys
from pathlib import Path

import joblib
import lightgbm
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_dir_short_20261028 as ds  # noqa: E402

LIVE = Path("/home/kbj20/crypto-scalping/data/live")
OUT = ROOT / "rtmodel" / "tests" / "fixtures"
N = 10_000
SPECIAL = np.array([np.nan, 0.0, -0.0, np.inf, -np.inf, 1e-36, -1e-36, 1e-35, -1e-35, 2e-35, -2e-35, 3.4028235e38, -3.4028235e38, 1.0, -1.0], np.float32)


def b64(a):
    return np.ascontiguousarray(np.asarray(a, np.float64)).view(np.uint64).tolist()


def nodes(txt):
    """[(feature, threshold)] -- 텍스트에서 바로(Rust 와 같은 원천)."""
    out = []
    for blk in txt.split("Tree=")[1:]:
        f = re.search(r"^split_feature=(.*)$", blk, re.M); t = re.search(r"^threshold=(.*)$", blk, re.M)
        if f:
            out += list(zip(map(int, f.group(1).split()), map(float, t.group(1).split())))
    return out


def main():
    assert lightgbm.__version__ == "4.7.0", lightgbm.__version__
    rng = np.random.default_rng(20261121)
    for coin in ("eth", "btc", "xrp"):
        art = joblib.load(LIVE / f"{coin}_dirshort65_model.joblib")
        txts = {h: [[m.booster_.model_to_string() for m, _ in ms] for ms in seeds] for h, seeds in art["models"].items()}
        info = re.search(r"^feature_infos=(.*)$", txts[min(txts)][0][0], re.M).group(1).split()
        lo = np.array([float(x.strip("[]").split(":")[0]) if x != "none" else -1.0 for x in info]); hi = np.array([float(x.strip("[]").split(":")[1]) if x != "none" else 1.0 for x in info])
        nf = len(lo); span = np.maximum(hi - lo, 1e-6)
        X = (lo - .1 * span + rng.random((N, nf)) * 1.2 * span).astype(np.float32)
        k = N // 4                                                   # 4분의 1: 칸마다 10% 확률로 특수값
        msk = rng.random((k, nf)) < 0.1; X[:k][msk] = rng.choice(SPECIAL, msk.sum())
        T = [nd for h in txts for seeds in txts[h] for t in seeds for nd in nodes(t)]
        for i in range(k, N // 2):                                    # 다음 4분의 1: 임계값 자리·f32 이웃 셋씩
            for _ in range(3):
                f, thr = T[rng.integers(len(T))]
                v = np.float32(thr); X[i, f] = rng.choice([v, np.nextafter(v, np.float32(np.inf)), np.nextafter(v, np.float32(-np.inf))])
        each = {h: [[b64(m.predict(X)) for m, _ in ms] for ms in seeds] for h, seeds in art["models"].items()}
        P = ds.predict(art, X)                                       # 시드 5 평균(각 시드 = Σ 표적 예측/척도)
        d = {"coin": coin, "lightgbm": lightgbm.__version__, "source": f"{coin}_dirshort65_model.joblib", "month": art["month"], "feats": art["feats"],
             "models": {str(h): [[{"txt": t, "scale": b64([s])[0]} for t, (_, s) in zip(tt, ms)] for tt, ms in zip(txts[h], art["models"][h])] for h in txts},
             "rows": b64(X.astype(np.float64).ravel()), "nf": nf, "each": {str(h): v for h, v in each.items()}, "pred": {str(h): b64(v) for h, v in P.items()}}
        p = OUT / f"lgb_{coin}.json"; p.write_text(json.dumps(d))
        print(p.name, f"{p.stat().st_size:,} 바이트 · 지평 {sorted(txts)} · 모델 {sum(len(s) * len(s[0]) for s in txts.values())} · 행 {N}", flush=True)


if __name__ == "__main__":
    main()
