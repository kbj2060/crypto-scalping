"""파이썬 4시간 방향(build_features + payload_from_features)·15분/1시간 LightGBM 앙상블 한 줄 시간 (2026-11-21). examples/bench.rs 와 같은 입력.
    python rtmodel/tools/bench_py_dir.py"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_dir4h_20261015 as d4  # noqa: E402
import live_eth_dir_short_20261028 as ds  # noqa: E402

LIVE = Path("/home/kbj20/crypto-scalping/data/live")
FX = ROOT / "rtmodel/tests/fixtures"


def stat(name, f, n):
    f()
    ns = []
    for _ in range(n):
        t = time.perf_counter_ns(); f(); ns.append(time.perf_counter_ns() - t)
    ns = np.array(ns) / 1e6
    print(f"{name:<46} n={n} 중앙 {np.median(ns):,.2f}ms · 최대 {ns.max():,.2f}ms")


def main():
    import warnings; warnings.filterwarnings("ignore")
    fx = json.loads((FX / "dir4h_fixture.json").read_text())
    s = next(x for x in fx["sets"] if x["coin"] == "SOL" and x["name"] == "normal")
    frames = {k: pd.DataFrame({"timestamp": pd.to_datetime(v["ts"], unit="ms"), **{c: np.array(v[c], np.uint64).view(np.float64) for c in v if c != "ts"}}) for k, v in s["frames"].items()}
    dc = pd.Series(np.array(s["dc"][1], np.uint64).view(np.float64), index=pd.to_datetime(s["dc"][0], unit="ms"))
    art = joblib.load(LIVE / "sol_dir4h_model.joblib")
    lg = joblib.load(LIVE / "eth_dirshort65_model.joblib")
    X = np.random.default_rng(0).normal(0, 1, (1, 65)).astype(np.float32)
    for lim in ("기본", 1):
        with (threadpool_limits(1) if lim == 1 else threadpool_limits(None)):
            stat(f"[{lim}] 4h build_features + payload (9,000봉 · 20코인)", lambda: d4.payload_from_features(art, d4.build_features(d4.frames_for(frames, "SOL"), dc)), 5)
            stat(f"[{lim}] 15분·1시간 LightGBM 앙상블 한 줄 (모델 40)", lambda: ds.predict(lg, X), 50)


if __name__ == "__main__":
    main()
