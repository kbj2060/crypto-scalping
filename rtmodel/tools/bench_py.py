"""파이썬 한 줄 predict_proba 시간(examples/bench.rs 와 같은 행). 스레드 1개 · 기본 둘 다.  python rtmodel/tools/bench_py.py [box]"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
FX = json.loads((ROOT / "rtmodel/tests/fixtures/pred_fixture.json").read_text())
N = 300   # 개발기 CPU 를 오래 잡지 않게


def rows(v):
    return np.array(v, dtype=np.uint64).view(np.float64)


def stat(name, f, X):
    for i in range(20):
        f(X[i])
    ns = []
    for i in range(N):
        t = time.perf_counter_ns(); f(X[i % len(X)]); ns.append(time.perf_counter_ns() - t)
    ns = np.array(ns) / 1000
    print(f"{name:<44} n={N} 중앙 {np.median(ns):,.0f}µs · 상위1% {np.quantile(ns, .99):,.0f}µs · 최대 {ns.max():,.0f}µs")


def main():
    b = joblib.load(ROOT / "dashboard/box_hold_hgb_20261016.joblib")["scales"]["1h"]
    case = next(c for c in FX["hgb_cases"] if c["file"] == "box_eth.json" and c["scale"] == "1h")
    X = rows(case["rows"])
    m, cols = b["model"], b["cols"]
    a = joblib.load(ROOT / "dashboard/sr_lean_model_20261118.joblib")
    for mm in a["models"]:   # live_sr_lean.load_art 그대로
        for _, st in getattr(mm, "steps", [(None, mm)]):
            if hasattr(st, "n_jobs"):
                st.n_jobs = 1
    XS = rows(FX["sr_lean"]["rows"]); XS = XS[np.isfinite(XS).all(1)]
    for lim in ("기본", 1):
        ctx = threadpool_limits(1) if lim == 1 else threadpool_limits(None)
        with ctx:
            stat(f"[{lim}] 박스 ETH 1h · DataFrame(box_hold 꼴)", lambda r: m.predict_proba(pd.DataFrame([r], columns=cols))[0, 1], X)
            stat(f"[{lim}] 레벨 5분 앙상블 · DataFrame(live 꼴)", lambda r: float(np.mean([mm.predict_proba(pd.DataFrame([r], columns=a["cols"]))[:, 1] for mm in a["models"]])), XS)


def box():
    """박스 유지 봉 열 → payload(server.py produce 와 같은 호출: payload → ETH 는 with_qx4i). examples/bench.rs 와 같은 봉 열."""
    import sys
    sys.path.insert(0, str(ROOT / "dashboard"))
    import box_hold as BH
    bx = json.loads((ROOT / "rtmodel/tests/fixtures/box_fixture.json").read_text())
    for asset in ("eth", "sol"):
        s = next(x for x in bx["sets"] if x["asset"] == asset and x["name"] == "normal")
        B = np.array([r[1:] for r in s["bars"]], np.uint64).view(np.float64)
        k = pd.DataFrame({"timestamp": pd.to_datetime([r[0] for r in s["bars"]], unit="ms", utc=True), "high": B[:, 0], "low": B[:, 1],
                          "close": B[:, 2], "volume": B[:, 3], "taker_buy_base": B[:, 4]})
        vol = {"ok": True, "asof_ms": int(s["bars"][-1][0]) + 60_000, "log_rv": -10.0, "log_rv_iv": -10.0}
        f = (lambda _: BH.with_qx4i(BH.payload(k, [], "eth"), vol)) if asset == "eth" else (lambda _: BH.payload(k, [], asset))
        for lim in ("기본", 1):
            with (threadpool_limits(1) if lim == 1 else threadpool_limits(None)):
                stat(f"[{lim}] 박스 {asset} 봉 열 → payload", f, [0] * 20)


if __name__ == "__main__":
    if sys.argv[1:] == ["box"]:
        box()
        raise SystemExit
    main()
