#!/usr/bin/env python3
"""OOS 첫 주(2026-01-01~01-08)의 regime3 라우터 확률 + 가격을 뽑아서 사용자 규칙(최대확률
>=0.5면 그 클래스 확정, 아니면 chop)을 적용한 레짐 시계열을 CSV로 저장 -- 차트용 원자료."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

START = "2026-01-01"
END = "2026-01-08"
OUT_PATH = ROOT / "tmp/causal_regen_20260516/eth_regime3_week_chart_20260819.csv"


def main() -> int:
    wd = gate.WINDOW_DEFS["oos_q1"]
    frame = sweep.load_frame(wd["start"], wd["end"], base_csv=wd["base_csv"], wide24_csv=wd["wide24_csv"])
    frame, _ = gate._drop_route_nan(frame)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    week = frame[(frame["timestamp"] >= START) & (frame["timestamp"] < END)].reset_index(drop=True)
    print(f"rows in week: {len(week)}", flush=True)

    probs = week[hard.ROUTE_COLS].to_numpy(dtype=np.float64)
    argmax_id = np.argmax(probs, axis=1)
    max_prob = probs.max(axis=1)
    names = np.array(hard.EXPERT_NAMES, dtype=object)
    regime = np.where(max_prob >= 0.5, names[argmax_id], "chop")

    out = pd.DataFrame({
        "timestamp": week["timestamp"],
        "close": pd.to_numeric(week["close"], errors="raise"),
        "bull_prob": probs[:, 0],
        "bear_prob": probs[:, 1],
        "chop_prob": probs[:, 2],
        "max_prob": max_prob,
        "argmax_regime": names[argmax_id],
        "regime_confirmed_0p5": regime,
    })
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"saved {len(out)} rows to {OUT_PATH}", flush=True)
    print("regime_confirmed_0p5 counts:", out["regime_confirmed_0p5"].value_counts().to_dict(), flush=True)
    print("argmax_regime counts (no 0.5 rule):", out["argmax_regime"].value_counts().to_dict(), flush=True)
    print("=== STUDY DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
