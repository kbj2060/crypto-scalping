#!/usr/bin/env python3
"""BTC 증거신호: **비용이 장벽인가, 엣지가 없는가**를 가른다 (진단 전용).

경제성게이트가 672셀 전부 음수로 끝났다(최선 VAL -3.85 / OOS -3.43bp). 두 해석이 가능하다:

  (a) 총이익(gross)은 있는데 표준비용 10bp가 다 먹는다 -> 실행개선(메이커)으로 살릴 여지
  (b) 총이익 자체가 없다 -> 신호가 죽었다. 실행을 아무리 개선해도 안 된다

같은 그리드를 **비용 0**으로 다시 돌려 총이익을 직접 잰다.

⚠️**이 숫자는 승격 근거가 아니다.** 이 저장소 규칙상 엣지는 표준 수수료를 넘어야 하고
(수수료 우대 가정 금지), 여기 gross는 "왜 실패했는가"를 설명하는 진단값일 뿐이다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_S = importlib.util.spec_from_file_location(
    "btcgate", ROOT / "scripts/gate_btc_evidence_signals_trailing_economics_20260902.py")
_g = importlib.util.module_from_spec(_S)
_S.loader.exec_module(_g)

OUT = ROOT / "data/research/btc_evidence_signals_costgate_20260902/gross_vs_cost.json"


def log(m): print(f"[btc-gross] {m}", flush=True)


def main() -> int:
    net = json.loads((ROOT / "data/research/btc_evidence_signals_costgate_20260902/report.json"
                      ).read_text())["signals"]
    kl = pd.read_csv(_g.KLINES)
    kl["timestamp"] = pd.to_datetime(kl["timestamp"], utc=True).dt.tz_localize(None)
    kl = kl.sort_values("timestamp").reset_index(drop=True)

    _g.ROUNDTRIP_COST_RATE = 0.0                       # ⚠️진단 전용
    log("비용 0으로 재실행 -- 총이익(gross) 직접 측정")
    log(f"{'신호':<26}{'gross VAL':>11}{'gross OOS':>11}{'net VAL':>10}{'net OOS':>10}{'비용':>8}")
    rep = {}
    for name, rel, builder, prep, kind in _g.SIGNALS:
        if name not in net or "cells" not in net[name]:
            continue
        fires, _ = _g.build_fires(name, rel, builder, prep, kind)
        fires["timestamp"] = pd.to_datetime(fires["timestamp"])
        if fires["timestamp"].dt.tz is not None:
            fires["timestamp"] = fires["timestamp"].dt.tz_localize(None)
        fires = fires.loc[fires["timestamp"] < _g.HOLDOUT_START].reset_index(drop=True)
        cells, _ns = _g.run_grid(kl, fires, _g.HORIZON[name])
        gb = max(cells, key=lambda c: min(c["val_fwd_bp"], c["oos_fwd_bp"]))
        nb = max(net[name]["cells"], key=lambda c: min(c["val_fwd_bp"], c["oos_fwd_bp"]))
        drag = gb["val_fwd_bp"] - nb["val_fwd_bp"]
        log(f"{name:<26}{gb['val_fwd_bp']:>11.2f}{gb['oos_fwd_bp']:>11.2f}"
            f"{nb['val_fwd_bp']:>10.2f}{nb['oos_fwd_bp']:>10.2f}{drag:>8.2f}")
        npass = sum(1 for c in cells if c["val_fwd_bp"] > 0 and c["oos_fwd_bp"] > 0)
        rep[name] = {"gross_best": {k: gb[k] for k in ("sl", "arm", "trail", "val_fwd_bp",
                                                       "oos_fwd_bp", "val_flip_bp", "oos_flip_bp")},
                     "gross_n_passing_96": npass,
                     "net_best_val_bp": nb["val_fwd_bp"], "net_best_oos_bp": nb["oos_fwd_bp"]}
        log(f"{'':<26}└ 비용0에서 동시양수 {npass}/96")
    log("")
    gp = sum(v["gross_n_passing_96"] for v in rep.values())
    log(f"=== 판정: 비용0에서도 동시양수 {gp}/672 ===")
    log("  ⇒ " + ("(a) 비용이 장벽 -- 총이익은 존재. 실행개선 여지 있음"
                  if gp > 100 else
                  "(b) ⭐**총이익 자체가 없다** -- 비용 문제가 아니다. 신호가 이 자산에서 죽었다"))
    OUT.write_text(json.dumps({"note": "진단 전용, 승격 근거 아님",
                               "gross_total_passing_672": gp, "signals": rep},
                              ensure_ascii=False, indent=2))
    log(f"report -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
