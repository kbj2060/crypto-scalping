"""4시간 방향(scripts/live_eth_dir4h_20261015.py) «REST 꼴 5분봉·metrics → build_features(65열) → HGB 회귀 → 화면 payload» 정답지 (2026-11-21 · tests/dir4h.rs).

정답은 원본 모듈의 build_features·frames_for·predict·payload_from_features 를 그대로 부른다. 안전(소스 확인): 그 모듈은 import 때 json·numpy·pandas 만
부르고, REST(requests)는 `_get` 안에서만 import 한다 -- 여기서는 rest_frames·daily_closes·live_payload 를 부르지 않는다. 입력은 rest_frames 가
돌려주는 꼴(대상 코인 = 5분봉 9,000개 + metrics 5열 · 나머지 19코인 = 5분봉 340개 · "BTC" 자리 코인만 OI)을 합성해 만든다.
모델은 개발기에 있는 SOL·XRP 판(메인 체크아웃 data/live/ -- 읽기만 · ETH 판은 개발기에 없다).

    python rtmodel/tools/export_dir4h.py && python rtmodel/tools/gen_dir4h_fixture.py      # → tests/fixtures/dir4h_fixture.json (커밋 안 함)
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_dir4h_20261015 as d4  # noqa: E402

LIVE = Path("/home/kbj20/crypto-scalping/data/live")
OUT = ROOT / "rtmodel" / "tests" / "fixtures"
COLS = ["high", "low", "close", "volume", "trades", "taker_buy_base"]


def bits(a):
    return np.ascontiguousarray(np.asarray(a, np.float64)).view(np.uint64).tolist()


def klines(rng, n, end, px, sig):
    ts = pd.date_range(end=end, periods=n, freq="5min")
    c = px * np.exp(np.cumsum(rng.normal(0, sig, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, sig / 2, n))); l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, sig / 2, n)))
    v = np.round(np.exp(rng.normal(8, 1, n)), 3); tr = np.round(v / np.exp(rng.normal(0, .3, n)) * 3); tb = np.round(v * rng.uniform(.3, .7, n), 3)
    return pd.DataFrame({"timestamp": ts, "high": np.round(h, 4), "low": np.round(l, 4), "close": np.round(c, 4), "volume": v, "trades": tr.astype(float), "taker_buy_base": tb})


def metrics(rng, ts, oi_only=False):
    n = len(ts)
    d = pd.DataFrame({"timestamp": ts, "sum_open_interest": np.round(1e6 * np.exp(np.cumsum(rng.normal(0, 2e-3, n))), 3)})
    if not oi_only:
        for c in ("count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"):
            d[c] = np.round(np.exp(rng.normal(0, .05, n) + np.cumsum(rng.normal(0, 1e-3, n))), 4)
    return d


def case(rng, coin, name, end):
    """rest_frames 꼴 frames(코인 이름 그대로) + 일봉. name 에 따라 구멍·0 거래량·NaN metrics·겹친 시각·빠진 코인·짧은 패널."""
    n = {"short": 8000, "exact": 8064}.get(name, 9000)
    tgt = klines(rng, n, end, {"SOL": 150.0, "XRP": 1.3}[coin], 1.5e-3)
    m = metrics(rng, tgt.timestamp)
    if name == "nan_metrics":
        m.loc[m.index[-700:-500], "sum_open_interest"] = np.nan; m.loc[m.index[-60:-30], "count_long_short_ratio"] = np.nan
    if name == "zero_vol":
        k = rng.choice(n, 40, replace=False); tgt.loc[k, ["volume", "taker_buy_base", "trades"]] = 0.0
        tgt.loc[tgt.index[-14:-2], ["volume", "taker_buy_base", "trades"]] = 0.0       # 최근 1시간 거래 0 → log(0) = −inf
    if name == "gaps":
        tgt = tgt.drop(index=rng.choice(np.arange(100, n - 5), 60, replace=False)).reset_index(drop=True)
    if name == "dup_ts":
        tgt = pd.concat([tgt, tgt.iloc[[-50, -20]].assign(close=lambda d: d.close * 1.001)]).reset_index(drop=True)   # 같은 시각 두 줄 -- grid 는 뒤 것
    for c in d4.MCOLS[1:]:
        if c not in m:
            m[c] = np.nan
    frames = {coin: tgt.merge(m, on="timestamp", how="left")}
    for sym in [s for s in ["ETH"] + d4.COINS if s != coin]:
        if name == "missing_coin" and sym in ("ADA", "TRX"):
            continue
        k = klines(rng, 340, end, 10 ** rng.uniform(-1, 4), 2e-3)
        if name == "gaps" and sym == "BTC":
            k = k.drop(index=[100, 101, 102, 250]).reset_index(drop=True)
        if sym == "BTC":
            k = k.merge(metrics(rng, k.timestamp, oi_only=True), on="timestamp", how="left")
        frames[sym] = k
    for f in frames.values():
        for c in d4.MCOLS:
            if c not in f:
                f[c] = np.nan
    days = pd.date_range(end=pd.Timestamp(end).normalize() - pd.Timedelta(days=1), periods=120, freq="D")
    dc = pd.Series(np.round({"SOL": 150.0, "XRP": 1.3}[coin] * np.exp(np.cumsum(rng.normal(0, .03, 120))), 4), index=days)
    return frames, dc


def enc(o):
    if isinstance(o, dict):
        return {k: enc(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [enc(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return {"b": int(np.float64(o).view(np.uint64))}
    return o


def main():
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(20261121)
    fx = {"sets": []}
    for coin in ("SOL", "XRP"):
        art = joblib.load(LIVE / f"{coin.lower()}_dir4h_model.joblib")
        names = ["normal", "gaps", "zero_vol", "nan_metrics", "dup_ts", "missing_coin", "short", "exact"]
        ends = ["2026-11-20 13:35", "2026-11-20 23:55", "2026-11-21 00:00", "2026-11-21 00:20", "2026-11-21 00:25", "2026-11-22 07:40", "2026-11-23 16:00", "2026-11-24 03:10"]
        for name, end in zip(names, ends):
            frames, dc = case(rng, coin, name, end)
            enc_frames = {s: {"ts": (f.timestamp.astype("int64") // 10**6).tolist(), **{c: bits(f[c]) for c in COLS + d4.MCOLS}} for s, f in frames.items()}
            try:
                F = d4.build_features(d4.frames_for(frames, coin), dc)
            except ValueError as e:     # 대상 패널 < 8,064봉: lag(x, 8064) 가 길이가 안 맞아 예외(서버 응답은 ok False) -- Rust 도 Err
                fx["sets"].append({"coin": coin, "name": name, "frames": enc_frames, "dc": [[int(t.value // 10**6) for t in dc.index], bits(dc.to_numpy())], "err": str(e)})
                print(coin, name, "예외", e, flush=True); continue
            pay = d4.payload_from_features(art, F)
            tail = F.tail(60)
            fx["sets"].append({"coin": coin, "name": name,
                               "frames": enc_frames,
                               "dc": [[int(t.value // 10**6) for t in dc.index], bits(dc.to_numpy())],
                               "ts": (tail.ts.astype("int64") // 10**6).tolist(), "feats": {c: bits(tail[c]) for c in d4.FEATS},
                               "pred": bits(d4.predict(art, tail)), "payload": enc(pay)})
            print(coin, name, len(F), "행 · NaN 피쳐", pay["nan_feats"], "· 예측", pay["pred_bp"], flush=True)
    p = OUT / "dir4h_fixture.json"; p.write_text(json.dumps(fx)); print(p.name, f"{p.stat().st_size:,} 바이트 · 벌 {len(fx['sets'])}")


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
