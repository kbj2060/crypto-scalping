#!/usr/bin/env python3
"""«소진형 대량 델타» 봉 뒤 4시간 **지속** — 사전등록 검정 (설계 동결 163a0688, 2026-09-30 20:38 KST).

설계서: docs/experiments/eth_exhaustion_delta_4h_continuation_prereg_20260930.md — 정의·판정 그대로, 바꾸지 않는다.
사건·대조군·라벨·중복 제거·부트스트랩은 되돌림 검정 스크립트의 함수를 **그대로** 쓴다(c = −r). 데이터 로더만 새로 쓴다.
  A = ETH 2020-02-01 ~ 2021-11-30 (vision 월 파일, 2020-01 워밍업) — Q1~Q4
  B = BTC 2022-01-01 ~ 2026-09-15 (로컬 api + vision 2023-12·2026-08·2026-09 일별, 2021-12 워밍업) — Q5
사용: --selfcheck / (인자 없음) 1회 실행
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_exhaustion_delta_4h_prereg_20260930 as R   # noqa: E402  같은 정의·같은 코드

MAIN = Path("/home/kbj20/crypto-scalping")
OUT = ROOT / "docs/experiments/eth_exhaustion_delta_4h_continuation_prereg_20260930_result.json"
A_RANGE = ("2020-02-01", "2021-11-30 23:55:00")
B_RANGE = ("2022-01-01", "2026-09-15 23:55:00")
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def read_vision(path: str) -> pd.DataFrame:
    """vision klines CSV -- 옛 파일은 머리줄이 없고 새 파일은 있다. 둘 다 받는다."""
    first = open(path).readline()
    x = pd.read_csv(path, header=0 if first.startswith("open_time") else None, names=None if first.startswith("open_time") else COLS)
    return pd.DataFrame({"timestamp": pd.to_datetime(x.open_time.astype("int64"), unit="ms"), "open": x.open, "high": x.high,
                         "low": x.low, "close": x.close, "volume": x.volume, "taker_buy_base": x.taker_buy_volume})


def finish(d: pd.DataFrame, lo: str, hi: str) -> pd.DataFrame:
    d = d.drop_duplicates("timestamp", keep="first").sort_values("timestamp")
    d = d[(d.timestamp >= lo) & (d.timestamp <= hi)]
    full = pd.date_range(d.timestamp.min(), d.timestamp.max(), freq="5min")
    d = d.set_index("timestamp").reindex(full); d.index.name = "timestamp"
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["delta"] = 2 * d.taker_buy_base - d.volume
    return d


def load_a() -> pd.DataFrame:
    fs = sorted(glob.glob(str(MAIN / "binance_data/klines/ETHUSDT/vision5m/ETHUSDT-5m-202[01]-*.csv")))
    return finish(pd.concat([read_vision(f) for f in fs]), "2020-01-01", A_RANGE[1])


def load_b() -> pd.DataFrame:
    a = pd.read_csv(MAIN / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv",
                    usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"], parse_dates=["timestamp"])
    v = [read_vision(f) for f in sorted(glob.glob(str(MAIN / "binance_data/klines/BTCUSDT/vision5m/BTCUSDT-5m-*.csv")))]
    return finish(pd.concat([a] + v), "2021-12-01", B_RANGE[1])


def cont(d: pd.DataFrame, rng: tuple, mult=R.MULT, h=R.H):
    ev, ct = R.events_a(d, mult); eb, cb = R.events_b(d, mult)
    ka, kb, ca, cbb = R.dedup(ev), R.dedup(eb), R.dedup(ct), R.dedup(cb)
    la, lb = R.labels(d, ka, h), R.labels(d, kb, h)
    ok = bool(np.array_equal(ka, kb) and np.array_equal(ca, cbb) and np.allclose(la.r, lb.r, atol=1e-9, rtol=0))
    sel = lambda x: x[(x.t >= rng[0]) & (x.t <= rng[1])].assign(r=lambda z: -z.r).reset_index(drop=True)   # r 칸 = c(지속)
    return sel(la), sel(R.labels(d, ca, h)), ok, (len(ka), len(kb), len(ca), len(cbb))


def desc(E: pd.DataFrame) -> dict:
    c = E.r.to_numpy()
    lo, hi = np.quantile(c, [0.05, 0.95]); tr = c[(c >= lo) & (c <= hi)]
    return {"n": len(c), "mean": float(c.mean()), "P_cont": float((c > 0).mean()), "median": float(np.median(c)),
            "trim5_mean": float(tr.mean()), "q90_cont": float(np.quantile(c, 0.9)), "q10_rev": float(-np.quantile(c, 0.1))}


def run() -> dict:
    res = {"frozen_commit": "163a0688"}
    dA = load_a(); EA, CA, okA, nA = cont(dA, A_RANGE)
    dB = load_b(); EB, CB, okB, nB = cont(dB, B_RANGE)
    print(f"독립 재구성 A {okA} {nA} · B {okB} {nB}", flush=True)
    if not (okA and okB):
        raise SystemExit("🔴 재구성 불일치 -- 결과를 쓰지 않는다")
    res["A"] = {"n": len(EA), "days": int(EA.day.nunique()), "n_control": len(CA), "mean_c": float(EA.r.mean()), "ci": R.boot(EA, lambda x: x.r.mean()),
                "by_year": {int(k): [float(v["mean"]), int(v["count"])] for k, v in EA.groupby(EA.t.dt.year).r.agg(["mean", "count"]).iterrows()},
                "control_mean_c": float(CA.r.mean()), "diff": float(EA.r.mean() - CA.r.mean()), "diff_ci": R.boot_diff(EA, CA),
                "buy": [float(EA[EA.side == "buy"].r.mean()), int((EA.side == "buy").sum())],
                "sell": [float(EA[EA.side == "sell"].r.mean()), int((EA.side == "sell").sum())]}
    res["B"] = {"n": len(EB), "days": int(EB.day.nunique()), "mean_c": float(EB.r.mean()), "ci": R.boot(EB, lambda x: x.r.mean())}
    a, b = res["A"], res["B"]
    res["pass"] = {"Q1": a["mean_c"] > 0 and a["ci"][0] > 0,
                   "Q2": all(v[0] > 0 for k, v in a["by_year"].items() if k in (2020, 2021)) and {2020, 2021} <= set(a["by_year"]),
                   "Q3": a["diff"] > 0 and a["diff_ci"][0] > 0,
                   "Q4": a["buy"][0] > 0 and a["sell"][0] > 0,
                   "Q5": b["mean_c"] > 0 and b["ci"][0] > 0}
    rep = {"A_desc": desc(EA), "B_desc": desc(EB),
           "B_control_mean_c": float(CB.r.mean()), "B_diff": float(EB.r.mean() - CB.r.mean()), "B_diff_ci": R.boot_diff(EB, CB),
           "B_buy": [float(EB[EB.side == "buy"].r.mean()), int((EB.side == "buy").sum())],
           "B_sell": [float(EB[EB.side == "sell"].r.mean()), int((EB.side == "sell").sum())]}
    for m in (30.0, 40.0):
        rep[f"A_mult{int(m)}"] = [float(cont(dA, A_RANGE, m)[0].r.mean()), len(cont(dA, A_RANGE, m)[0])]
    for h in (24, 96):
        rep[f"A_h{h * 5 // 60}h"] = float(cont(dA, A_RANGE, h=h)[0].r.mean())
    res["report_only"] = rep
    return res


def _selfcheck() -> None:
    R._selfcheck()
    import tempfile, os
    rows = "1577836800000,129.12,129.5,128.7,128.8,100,1577837099999,1,5,60,1,0\n"
    with tempfile.TemporaryDirectory() as t:
        p1, p2 = os.path.join(t, "a.csv"), os.path.join(t, "b.csv")
        open(p1, "w").write(rows); open(p2, "w").write(",".join(COLS) + "\n" + rows)
        for p in (p1, p2):
            x = read_vision(p)
            assert len(x) == 1 and x.taker_buy_base.iloc[0] == 60 and str(x.timestamp.iloc[0]) == "2020-01-01 00:00:00", x
    E = pd.DataFrame({"r": [-10.0, 5.0], "t": pd.to_datetime(["2020-03-01", "2020-03-02"]), "side": ["buy", "sell"], "day": pd.to_datetime(["2020-03-01", "2020-03-02"])})
    assert desc(E.assign(r=-E.r))["mean"] == 2.5          # c = −r 부호
    print("continuation selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selfcheck", action="store_true"); a = ap.parse_args()
    if a.selfcheck:
        _selfcheck(); sys.exit(0)
    r = run(); OUT.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str)); print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
