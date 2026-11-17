#!/usr/bin/env python3
"""ETHUSDT 선물 체결(aggTrades, Binance Vision 일 파일) → 1초 묶음 (2026-11-18). 지지·저항 «매매 강도» 연구용 원자료.

  python scripts/build_eth_aggtrades_1s_20261118.py 2024-01-01 2026-11-16 [--workers 6]     # 없는 날만 받아 가공(zip 은 저장하지 않는다)
  python scripts/build_eth_aggtrades_1s_20261118.py --selftest

산출: tmp/sr_intensity_20261118/agg1s/ETHUSDT-YYYY-MM-DD.parquet — 체결이 있던 초만.
  s 초(UTC epoch) · po·ph·pl·pc 그 초의 첫·최고·최저·마지막 체결가 · bq·sq 테이커 매수·매도 수량 · nb·ns 묶음 체결 수 · nt 원 체결 수 · mq 최대 한 건 수량 · q2 수량 제곱합(큰 체결 쏠림).
is_buyer_maker = True 면 매수자가 메이커 = 테이커 매도. Vision 정적 파일만 쓴다(REST 아님).
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/sr_intensity_20261118/agg1s"
URL = "https://data.binance.vision/data/futures/um/daily/aggTrades/ETHUSDT/ETHUSDT-aggTrades-{}.zip"


def agg(d: pd.DataFrame) -> pd.DataFrame:
    """체결 표(price·qty·first·last·t ms·ibm) → 1초 묶음. 시간순 정렬을 가정하지 않는다."""
    d = d.sort_values(["t", "first"], kind="stable")
    s = (d.t.to_numpy() // 1000).astype(np.int64); p, q = d.price.to_numpy(float), d.qty.to_numpy(float); sell = d.ibm.to_numpy(bool)
    g = pd.DataFrame({"s": s, "p": p, "bq": np.where(sell, 0.0, q), "sq": np.where(sell, q, 0.0), "nb": (~sell).astype(np.int32), "ns": sell.astype(np.int32),
                      "nt": (d["last"].to_numpy() - d["first"].to_numpy() + 1).astype(np.int64), "q": q, "q2": q * q}).groupby("s", sort=True)
    o = g.agg(po=("p", "first"), ph=("p", "max"), pl=("p", "min"), pc=("p", "last"), bq=("bq", "sum"), sq=("sq", "sum"), nb=("nb", "sum"), ns=("ns", "sum"), nt=("nt", "sum"), mq=("q", "max"), q2=("q2", "sum"))
    return o.reset_index()


def one(day: str) -> str:
    p = OUT / f"ETHUSDT-{day}.parquet"
    if p.exists():
        return f"{day} 있음"
    for attempt in range(3):
        try:
            raw = urllib.request.urlopen(URL.format(day), timeout=120).read(); break
        except urllib.error.HTTPError as e:
            return f"{day} 없음({e.code})"
        except Exception as e:                                        # 일시 오류는 다시
            if attempt == 2:
                return f"{day} 실패 {type(e).__name__}"
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        d = pd.read_csv(z.open(z.namelist()[0]), header=None, low_memory=False)
    if not str(d.iloc[0, 0]).isdigit():
        d = d.iloc[1:]
    d = d.iloc[:, :7]; d.columns = ["id", "price", "qty", "first", "last", "t", "ibm"]
    d = d.assign(price=d.price.astype(float), qty=d.qty.astype(float), first=d["first"].astype("int64"), last=d["last"].astype("int64"), t=d.t.astype("int64"),
                 ibm=d.ibm.astype(str).str.lower().eq("true"))
    o = agg(d); tmp = p.with_suffix(".tmp"); o.to_parquet(tmp); tmp.replace(p)
    return f"{day} {len(d)}건 → {len(o)}초"


def selftest() -> None:
    d = pd.DataFrame({"price": [100.0, 101.0, 99.0, 100.5, 100.2], "qty": [1.0, 2.0, 3.0, 4.0, 5.0], "first": [1, 2, 4, 5, 9], "last": [1, 3, 4, 8, 9],
                      "t": [1000, 1500, 1999, 2000, 3999], "ibm": [False, True, False, True, False]})
    o = agg(d.sample(frac=1, random_state=0)).set_index("s")                 # 섞여 들어와도 같은 결과
    a = o.loc[1]; assert (a.po, a.ph, a.pl, a.pc) == (100.0, 101.0, 99.0, 99.0) and a.bq == 4.0 and a.sq == 2.0 and a.nb == 2 and a.ns == 1 and a.nt == 4 and a.mq == 3.0 and a.q2 == 14.0
    assert o.loc[2].sq == 4.0 and o.loc[2].nt == 4 and o.loc[3].bq == 5.0 and len(o) == 3
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("start", nargs="?"); ap.add_argument("end", nargs="?"); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    OUT.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(a.start, a.end, inclusive="left").strftime("%Y-%m-%d").tolist()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(one, days, chunksize=4)):
            if i % 50 == 0 or "실패" in r or "없음" in r:
                print(i, r, flush=True)
