#!/usr/bin/env python3
"""ETH bookDepth 30초 패널 빌더 (2026-09-08).

`binance_data/bookDepth/ETHUSDT/*.zip` (869일, 2024-04-20~2026-09-05) 을
30초 해상도 넓은 표로 편다. 밴드는 **누적** depth(±0.2/1/2/3/4/5%).

## ⚠️데이터 결함 (이 세션에서 발견, 반드시 필터할 것)
한쪽 밴드 전체가 상수로 굳거나 근접 밴드가 붕괴하는 구간이 있다:
  - 2025-10-11~10-15, 2025-10-30~12-01  (BID 붕괴, 38일 -- **VAL 창의 31%**)
  - 2026-09-04~                          (ASK 근접밴드 붕괴, 라이브 영향)
스냅샷 단위로 `bd_ok` 를 계산해 둔다(한쪽 1% 깊이가 반대쪽의 2% 미만이면 불량).
"""
from __future__ import annotations
import zipfile, glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = sorted(glob.glob(str(ROOT / "binance_data/bookDepth/ETHUSDT/*.zip")))
OUT = ROOT / "data/research/eth_bookdepth_30s_20260908.parquet"
BANDS = [-5.0, -4.0, -3.0, -2.0, -1.0, -0.2, 0.2, 1.0, 2.0, 3.0, 4.0, 5.0]


def main() -> int:
    frames = []
    for i, f in enumerate(SRC):
        try:
            z = zipfile.ZipFile(f)
            d = pd.read_csv(z.open(z.namelist()[0]))
        except Exception as e:
            print(f"  skip {os.path.basename(f)}: {e}", flush=True); continue
        d["percentage"] = d["percentage"].astype(float)
        p = d.pivot_table(index="timestamp", columns="percentage", values="depth", aggfunc="last")
        for b in BANDS:
            if b not in p.columns:
                p[b] = np.nan
        p = p[BANDS]
        p.columns = [f"d{str(b).replace('-','m').replace('.','p')}" for b in BANDS]
        p.index = pd.to_datetime(p.index)
        frames.append(p.astype(np.float32))
        if i % 100 == 0:
            print(f"  [{i}/{len(SRC)}] {os.path.basename(f)}", flush=True)
    P = pd.concat(frames).sort_index()
    P = P[~P.index.duplicated(keep="last")]
    b1 = P["dm1p0"].to_numpy(float); a1 = P["d1p0"].to_numpy(float)
    tot = b1 + a1
    P["bd_ok"] = ((np.isfinite(tot)) & (tot > 0)
                  & (a1 >= 0.02 * b1) & (b1 >= 0.02 * a1)).astype(np.int8)
    P.index.name = "ts"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    P.reset_index().to_parquet(OUT, index=False)
    print(f"[done] {P.shape} {P.index.min()} ~ {P.index.max()} bd_ok={P.bd_ok.mean():.4f} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
