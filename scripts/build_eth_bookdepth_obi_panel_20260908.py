#!/usr/bin/env python3
"""bookDepth -> **1분 OBI 패널** 캐시 (2026-09-08).

`binance_data/bookDepth/ETHUSDT/*.zip` 869일 · 30초 스냅샷 · percentage ∈ {±0.2…±5}
각 스냅샷에서 매수측(percentage<0)/매도측(>0) notional 을 합쳐 OBI 를 만들고 1분으로 축약한다.
  obi_all = (bid - ask) / (bid + ask)                 전 레벨
  obi_tight = ±1% 이내만 · obi_wide = ±5% 까지
  depth_tot = 총 notional (호가 두께)
⚠️스푸핑 판별은 이 데이터로 원리적으로 불가(취소 이벤트가 없다) -- OBI 수준·변화만 쓴다.
"""
from __future__ import annotations
import glob, io, json, zipfile
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "binance_data/bookDepth/ETHUSDT"
OUT = ROOT / "tmp/eth_breakout_atr_state_20260908_s1/bookdepth_obi_1m.parquet"


def main() -> int:
    files = sorted(glob.glob(str(SRC / "*.zip")))
    print(f"파일 {len(files)}개", flush=True)
    parts = []
    for i, f in enumerate(files):
        try:
            z = zipfile.ZipFile(f)
            n = [x for x in z.namelist() if x.endswith(".csv")][0]
            d = pd.read_csv(io.BytesIO(z.read(n)))
        except Exception:
            continue
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
        d = d.dropna(subset=["timestamp"])
        p = d["percentage"].astype(float); nt = d["notional"].astype(float)
        bid = p < 0
        g = d.assign(_b=np.where(bid, nt, 0.0), _a=np.where(~bid, nt, 0.0),
                     _bt=np.where(bid & (p >= -1), nt, 0.0), _at=np.where((~bid) & (p <= 1), nt, 0.0))
        s = g.groupby("timestamp")[["_b", "_a", "_bt", "_at"]].sum()
        s["obi_all"] = (s["_b"] - s["_a"]) / np.maximum(s["_b"] + s["_a"], 1e-9)
        s["obi_tight"] = (s["_bt"] - s["_at"]) / np.maximum(s["_bt"] + s["_at"], 1e-9)
        s["depth_tot"] = s["_b"] + s["_a"]
        s = s[["obi_all", "obi_tight", "depth_tot"]].resample("1min").mean()
        parts.append(s)
        if (i + 1) % 100 == 0: print(f"   {i+1}/{len(files)}", flush=True)
    P = pd.concat(parts).sort_index()
    P = P[~P.index.duplicated(keep="last")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    P.to_parquet(OUT)
    print(f"저장 {P.shape} · {P.index[0]} ~ {P.index[-1]} · 커버 {P['obi_all'].notna().mean():.1%}")
    print(json.dumps({"rows": len(P)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
