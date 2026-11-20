"""파이썬 한 줄 predict_proba 시간(examples/bench.rs 와 같은 행). 스레드 1개 · 기본 둘 다.  python rtmodel/tools/bench_py.py [box|sr]"""
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


def sr():
    """레벨 5분: 정답지 한 벌(normal0)의 연산을 원본 Live 로 다시 흘려 second() 한 번 시간(사건 채점이 든 초 · 안 든 초 따로). examples/bench.rs 와 같은 벌."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import warnings
    import live_sr_lean_20261118 as LS
    warnings.filterwarnings("ignore")
    fx = json.loads((ROOT / "rtmodel/tests/fixtures/sr_fixture.json").read_text())
    st = next(x for x in fx["sets"] if x["name"] == "normal0")
    for lim in ("기본", 1):
        with (threadpool_limits(1) if lim == 1 else threadpool_limits(None)):
            art = LS.load_art() if lim == 1 else __import__("joblib").load(LS.MODEL)    # 기본 = n_jobs 를 그대로(ET 8) · 1 = load_art(워커 그대로)
            n = [0]; L = LS.Live(art, lambda k, r: n.__setitem__(0, n[0] + (k == "ev")))
            plain, scored = [], []
            for o in st["ops"]:
                if o[0] == "trade":
                    L.trade(*o[1:])
                elif o[0] == "init":
                    L.t0 = L.done = o[1]
                elif o[0] == "klines":
                    L.klines(o[1], o[2])
                elif o[0] == "recheck":
                    for r in o[1]:
                        mi = int(r[0]) // 60000; ref = [float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[9])]
                        if mi <= L.t0 // 60 and mi in L.M.index and ref[3] > 0:
                            L.M.loc[mi] = ref
                elif o[0] == "hole":
                    L.hole(o[1])
                elif o[0] == "close":
                    for s_ in range(L.done + 1, o[1] + 1):
                        k0 = n[0]; t = time.perf_counter_ns(); L.second(s_); dt = (time.perf_counter_ns() - t) / 1000
                        (scored if n[0] > k0 else plain).append(dt)
            for nm, v in (("second() 사건 채점 없는 초", plain), ("second() 사건 채점이 든 초", scored)):
                v = np.array(v); print(f"[{lim}] 레벨 5분 {nm:<26} n={len(v)} 중앙 {np.median(v):,.0f}µs · 상위1% {np.quantile(v, .99):,.0f}µs · 최대 {v.max():,.0f}µs")


if __name__ == "__main__":
    if sys.argv[1:] == ["sr"]:
        sr()
        raise SystemExit
    if sys.argv[1:] == ["box"]:
        box()
        raise SystemExit
    main()
