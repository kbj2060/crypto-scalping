#!/usr/bin/env python3
"""**5분봉마다의 청산맵 지지/저항 패널** (2026-09-11, 사용자 요청).

사용자: "돌파와 되돌림에서 중요한 건 저항과 지지를 돌파하는가 돌아오는가."
그 질문을 재려면 봉마다 «그때의 지지·저항이 어디였나»가 있어야 한다.

레벨은 **라이브 함수를 그대로 호출**한다(`compute_spliced_levels`, 2026-08-26 배포판:
지지=mid 패스 · 저항=close 패스). 식을 두 벌 두지 않는다.

## 인과성
- 1시간 룩백창은 **그 5분봉이 시작하기 전에 이미 닫힌** 1시간봉만 쓴다(`T + 1h <= t_i`).
  라이브보다 최대 1시간 보수적이지만 부분봉이 절대 안 섞인다.
- `current_price` 는 **그 5분봉의 종가**다 -- 결정 시점에 실제로 아는 값이고 진입 가능한 값이다
  (2026-09-10 A/B 라벨 감사의 «기준가는 체결 가능해야» 규칙).
- 라이브 함수 안의 future_min_low/future_max_high 는 **룩백창 내부**의 이미-발동 필터라
  창이 결정 시점에서 끝나는 한 미래참조가 아니다(원본 주석 확인).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT if (ROOT / "binance_data").exists() else Path(subprocess.run(
    ["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
    capture_output=True, text=True).stdout.strip()).parent
sys.path.insert(0, str(ROOT / "scripts"))
from live_liquidation_map_20260824 import LOOKBACK_HOURS, compute_spliced_levels  # noqa: E402

KL5 = DATA / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT = DATA / "tmp/eth_liqmap_sr_panel_20260911"
KEEP = 3          # 측면당 최근접 3개까지 기록 -- 4번째부터는 5% 밖이 대부분이라 쓸 일이 없다


def main() -> int:
    kl = (pd.read_csv(KL5, usecols=["timestamp", "high", "low", "close", "volume"],
                      parse_dates=["timestamp"])
          .sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True))
    h = (kl.set_index("timestamp").resample("1h")
         .agg({"high": "max", "low": "min", "close": "last", "volume": "sum"})
         .dropna().reset_index())
    # 1시간봉 h[k] 는 [ts_k, ts_k+1h) 를 덮는다 -> ts_k + 1h <= t_i 인 것만 쓴다.
    h_end = h["timestamp"].to_numpy() + np.timedelta64(1, "h")
    t5 = kl["timestamp"].to_numpy()
    last = np.searchsorted(h_end, t5, side="right") - 1        # 쓸 수 있는 마지막 1시간봉 인덱스

    cl = kl["close"].to_numpy(float)
    n = len(kl)
    cols = {f"{s}{j}_{f}": np.full(n, np.nan)
            for s in ("s", "r") for j in range(1, KEEP + 1) for f in ("px", "w")}
    n_sup = np.zeros(n, np.int8); n_res = np.zeros(n, np.int8)

    t0 = time.time()
    lo = int(np.searchsorted(last, LOOKBACK_HOURS))            # 창이 다 차는 첫 봉
    for i in range(lo, n):
        k = last[i]
        if k < LOOKBACK_HOURS:
            continue
        lv = compute_spliced_levels(h.iloc[k + 1 - LOOKBACK_HOURS:k + 1], cl[i])
        sup, res = lv["support_levels"], lv["resistance_levels"]
        n_sup[i] = len(sup); n_res[i] = len(res)
        for j in range(min(KEEP, len(sup))):
            cols[f"s{j+1}_px"][i] = sup[j]["price"]; cols[f"s{j+1}_w"][i] = sup[j]["weight_pct"]
        for j in range(min(KEEP, len(res))):
            cols[f"r{j+1}_px"][i] = res[j]["price"]; cols[f"r{j+1}_w"][i] = res[j]["weight_pct"]
        if (i - lo) % 50000 == 0:
            done = i - lo + 1
            print(f"  {done:,}/{n - lo:,}  {time.time() - t0:.0f}s", flush=True)

    out = pd.DataFrame({"timestamp": kl["timestamp"], "close": cl,
                        "n_sup": n_sup, "n_res": n_res, **cols}).iloc[lo:].reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "sr_panel_5m.parquet")
    got = out["s1_px"].notna()
    print(f"\n{len(out):,}행 · {out.timestamp.min()} ~ {out.timestamp.max()}")
    print(f"지지 있음 {got.mean():.3f} · 저항 있음 {out['r1_px'].notna().mean():.3f} · "
          f"양쪽 다 {(got & out['r1_px'].notna()).mean():.3f}")
    print(f"저장: {OUT / 'sr_panel_5m.parquet'}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
