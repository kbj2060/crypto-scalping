"""30분 카드 방향·닿음(card30) 정답지 (2026-11-21 · tests/card30.rs). 서빙 = dashboard/server.py compute_situation → _card30_pred(봉마다 한 번 · ETH 모델을 모든 코인에)
→ 시장 맥락 ctx 의 dir_p·reach_p → fr.read(융합 카드)·융합 장부(flow_read_fused.jsonl). 피쳐 = dashboard/flow_read.card30_features(원본 그대로 부름).
_card30_pred 의 보정 식 6줄은 server.py 안 닫힌 함수라 부를 수 없어 아래 pred() 에 글자 그대로 옮겼다(server.py:2251-2259).
모델 = 서버 tmp/card30_direction_20260926 사본(코디네이터가 가져옴 · md5 같음) -- 메인 체크아웃 tmp/rtmodel_models_20261121.
    (tools/server_env.sh 의 환경) python rtmodel/tools/gen_card30_fixture.py      # → tests/fixtures/card30_fixture.json · card30.json
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from dashboard import flow_read as fr  # noqa: E402
from export_hgb import SKLEARN_OK, b1  # noqa: E402
from export_card30 import export  # noqa: E402

M = Path("/home/kbj20/crypto-scalping/tmp/rtmodel_models_20261121")


def pred(m, f):
    lg = lambda q: np.log(q / (1 - q))   # noqa: E731
    p = {}
    for key, M_ in m.items():
        row = np.array([[f[c] for c in M_["cols"]]], dtype=np.float32)
        raw = float(np.mean([mm.predict_proba(row)[0, 1] for mm in M_["models"]]))
        p[key] = float(1 / (1 + np.exp(-(M_["k"] * lg(raw) + (1 - M_["k"]) * lg(M_["base"])))))
    return p


def main():
    if sklearn.__version__ != SKLEARN_OK:
        raise SystemExit("server_env.sh 의 환경으로")
    fx = ROOT / "rtmodel/tests/fixtures"
    m = {"dir": joblib.load(M / "model.joblib"), "reach": joblib.load(M / "model_reach.joblib")}
    export(M, fx / "card30.json")
    rng = np.random.default_rng(30)
    n = 2400
    t0 = 1_795_000_000 // 300 * 300
    c = 3000 * np.exp(np.cumsum(rng.normal(0, 2e-3, n)))
    h, lo = c * (1 + np.abs(rng.normal(0, 1e-3, n))), c * (1 - np.abs(rng.normal(0, 1e-3, n)))
    v = np.exp(rng.normal(6, 1, n)); tb = v * rng.uniform(0.2, 0.8, n)
    v[1500:1510] = 0.0; tb[1500:1510] = 0.0                                        # 거래 0(1e-9 바닥)
    h[1700:1720] = lo[1700:1720] = c[1700:1720] = c[1699]                          # 고=저(폭 0 → 1e-9 바닥) · 같은 값
    bars = [{"time": t0 + 300 * i, "high": float(f"{h[i]:.2f}"), "low": float(f"{lo[i]:.2f}"), "close": float(f"{c[i]:.2f}"), "volume": float(f"{v[i]:.3f}"),
             "taker": float(f"{tb[i]:.3f}")} for i in range(n)]
    for i in range(n):
        bars[i]["high"] = max(bars[i]["high"], bars[i]["close"]); bars[i]["low"] = min(bars[i]["low"], bars[i]["close"])
    cases = []
    for end in list(range(250, n, 7)) + [n]:
        w = bars[max(0, end - 300):end]
        f = fr.card30_features(w)
        cases.append({"end": end, "f": None if f is None else {k: b1(x) for k, x in f.items()}, "p": None if f is None else {k: b1(x) for k, x in pred(m, f).items()}})
    gap = bars[:1200] + bars[1201:1300]                                            # 봉 하나 빠짐 → None
    cases.append({"gap": True, "f": None if fr.card30_features(gap) is None else "?"})
    (fx / "card30_fixture.json").write_text(json.dumps({"bars": bars, "cases": cases}))
    print(len(cases), sum(x["f"] is None for x in cases), "None")


if __name__ == "__main__":
    main()
