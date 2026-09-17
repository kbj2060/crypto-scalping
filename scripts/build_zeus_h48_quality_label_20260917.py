#!/usr/bin/env python3
"""Zeus — **h48_conservative 품질 라벨**을 2022~2026 으로 생성 (2026-09-17, dev/CPU)

h48qual 부모의 품질 타깃(`quality_mode=quality_label_action`)을 Zeus 프레임 전 구간에 만든다.
배리어 시뮬은 서버에서 받은 캐시(`h48_retune/tb_h48cons_sim.parquet`, fee=0 으로 계산)를 쓰고,
여기서는 **비용만 사후 적용**한다 -- 식은 라벨 빌더와 동일:

    q = ret − fee − 0.20·max(−mae, 0) − 0.003·(reason == "sl")
    action = LONG if q_L>0 and q_L>=q_S · SHORT if q_S>0 · else CASH

⭐`fee_cost = 42bp` 는 «우리가 내는 비용»이 아니라 **「이 정도는 벌어야 후보로 친다」는
선별 문턱**으로 일한다(README §2.6). 실측 비용(1.02bp)으로 «정정»하면 CASH 가 40%→20% 로
줄며 품질 머리가 걸러낼 게 없어진다. 그래서 **배포값 42bp 를 주 라벨로** 둔다.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "crypto-scalping/tmp/omega461_longwindow_20260917"
SIM = DATA / "h48_retune/tb_h48cons_sim.parquet"
OUT = DATA / "zeus_h48_quality_labels_20260917"
MAE_PEN, SL_PEN = 0.20, 0.003
FEES = {"deployed_42bp": 0.0042, "usdc3x_3.06bp": 0.000306, "zero": 0.0}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(SIM)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    frame = pd.read_parquet(DATA / "features_with_regime_2022_2026_realfunding.parquet",
                            columns=["timestamp"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    print(f"배리어 시뮬 {len(d):,}행 · 프레임 {len(frame):,}행", flush=True)

    rows = []
    for name, fee in FEES.items():
        lq = (d.tb_long_ret - fee - MAE_PEN * np.maximum(-d.tb_long_mae, 0.0)
              - SL_PEN * (d.tb_long_reason == "sl"))
        sq = (d.tb_short_ret - fee - MAE_PEN * np.maximum(-d.tb_short_mae, 0.0)
              - SL_PEN * (d.tb_short_reason == "sl"))
        a = np.where((lq > 0) & (lq >= sq), 1, np.where(sq > 0, 2, 0)).astype(np.int64)
        out = pd.DataFrame({"timestamp": d.timestamp, "tb_action": a,
                            "tb_quality": np.maximum(lq, sq)})
        j = frame.merge(out, on="timestamp", how="left")
        miss = int(j.tb_action.isna().sum())
        j["tb_action"] = j.tb_action.fillna(0).astype(np.int64)   # 시뮬 없는 끝단은 CASH
        sh = np.bincount(j.tb_action.to_numpy(), minlength=3) / len(j)
        path = OUT / f"h48cons_{name}.parquet"
        j.to_parquet(path, index=False)
        rows.append({"fee": name, "fee_val": fee, "cash": float(sh[0]), "long": float(sh[1]),
                     "short": float(sh[2]), "active": float(1 - sh[0]), "unmatched": miss})
        print(f"{name:<16} CASH/LONG/SHORT = {sh.round(4)} · 활성률 {(1-sh[0])*100:.1f}% "
              f"· 프레임 미매칭 {miss:,} · 저장 {path.name}", flush=True)
    (OUT / "meta.json").write_text(json.dumps(
        {"source": str(SIM), "mae_pen": MAE_PEN, "sl_pen": SL_PEN, "variants": rows},
        indent=2, default=float))
    print(f"저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
