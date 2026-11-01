#!/usr/bin/env python3
"""횡보/추세 분별 전진 기록기·채점기 (2026-11-02). 매 시 정각 4코인 예측을 JSONL 로 쌓고(record), 4시간이 지난 행을 기록해 둔 지지·저항으로 채점한다(score).
라이브 서빙과 같은 predict_latest 를 쓴다. 이 스크립트는 서버에 등록하지 않았다(배포는 사용자 결정) -- 등록 방법은 docs/deploy_queue_regime_band_20261102.md.
  record: python scripts/live_regime_band_recorder_20261102.py record <패널 디렉터리>   (코인별 {COIN}USDT_5m.parquet)
  score : python scripts/live_regime_band_recorder_20261102.py score  <패널 디렉터리>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_regime_band_20261102 as S  # noqa: E402

LOG = S.ROOT / "data" / "live" / "regime_band_forward.jsonl"
COINS = ("BTC", "ETH", "SOL", "XRP")
BINS = (0, 0.2, 0.4, 0.6, 0.8, 1.01)


def record(panels: dict, log: Path = LOG) -> list[dict]:
    log.parent.mkdir(parents=True, exist_ok=True)
    seen = {(r["ts"], r["coin"]) for r in map(json.loads, log.open()) } if log.exists() else set()
    out = []
    for coin in COINS:
        o = S.predict_latest(panels, coin)
        row = dict(ts=str(o["ts"]), coin=coin, p_trend=o["p_trend"], regime=o["regime"], confidence=o["confidence"], support=o["support"], resistance=o["resistance"])
        if (row["ts"], coin) not in seen:
            out.append(row)
    with log.open("a") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def score(panels: dict, log: Path = LOG) -> tuple[pd.DataFrame, dict]:
    rows = [json.loads(l) for l in log.open()] if log.exists() else []
    res = []
    for r in rows:
        df = panels[r["coin"]]
        t = pd.Timestamp(r["ts"])
        w = df.c.loc[t: t + pd.Timedelta(minutes=5 * 47)]                      # 결정 시각 이후 48봉 종가(= 학습 라벨 yA48 과 같은 창)
        if len(w) < 48:
            continue                                                          # 아직 4시간 안 지남
        y = int(((w > r["resistance"]) | (w < r["support"])).any())
        res.append(dict(r, y=y, correct=int((r["p_trend"] >= 0.5) == bool(y))))
    d = pd.DataFrame(res)
    if d.empty:
        return d, {"n_scored": 0, "n_pending": len(rows)}
    d["bin"] = pd.cut(d.confidence, BINS, right=False)
    summ = dict(n_scored=len(d), n_pending=len(rows) - len(d), acc=float(d.correct.mean()), trend_rate=float(d.y.mean()),
                by_conf={str(k): dict(n=int(len(g)), acc=float(g.correct.mean())) for k, g in d.groupby("bin", observed=True)},
                by_coin={c: float(g.correct.mean()) for c, g in d.groupby("coin")})
    return d, summ


if __name__ == "__main__":
    cmd, dirp = sys.argv[1], Path(sys.argv[2])
    panels = {c: pd.read_parquet(dirp / f"{c}USDT_5m.parquet", columns=["o", "h", "l", "c", "qv", "tbqv", "n"]) for c in COINS}
    if cmd == "record":
        print(json.dumps(record(panels), ensure_ascii=False))
    else:
        print(json.dumps(score(panels)[1], ensure_ascii=False, indent=1))
