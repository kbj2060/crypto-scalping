#!/usr/bin/env python3
"""ETH demarker/kalman **새 셀에서 K 재스윕** -- 사용자 지시로 셀 교체 진행 (2026-09-03).

## 지시

  demarker_extreme          H=8/GAP=12  ->  **H=6/GAP=24**
  kalman_deviation_meanrev  H=12/GAP=12 ->  **H=16/GAP=24**

## 왜 K를 다시 스윕하는가

현행 K(demarker 0.70, kalman 2.5)는 **옛 H/GAP에서** 최적화된 값이다
(`research_eth_kalman_demarker_ksweep_20260831.py`, SIGNAL_CONFIG에 옛 셀이 박혀 있다).
H를 바꾸면 목표 도달에 주어지는 시간이 바뀌므로 최적 K도 함께 움직인다 --
K를 그대로 두고 H만 갈아끼우면 두 축이 어긋난 셀이 된다.

## K 격자

원본 `K_GRID = [1.0 .. 6.0]`은 **하단 경계에서 승자가 나왔던 전례**가 있다
(README §5.6: 확장하니 진짜 정점이 K=0.70이었다). ⇒ 하단을 0.4까지 열어 둔다.

    K_GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]

⚠️선택 규칙은 원본과 동일한 `min(VAL, OOS)` AUC를 쓴다 -- 규칙을 바꾸면 현행 K와 비교가
성립하지 않는다. 이 규칙에 OOS가 들어 있다는 한계는 그대로 안고 간다(원본 설계).
⚠️HOLDOUT(>=2026-04-01) 미터치.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = ROOT / "data/research/eth_kalman_demarker_ksweep_new_cells_20260903.json"
SRC = "research_eth_kalman_demarker_ksweep_20260831.py"

NEW_CELLS = {"demarker_extreme": {"horizon": 6, "gap": 24},
             "kalman_deviation_meanrev": {"horizon": 16, "gap": 24}}
OLD_CELLS = {"demarker_extreme": {"horizon": 8, "gap": 12, "K": 0.70},
             "kalman_deviation_meanrev": {"horizon": 12, "gap": 12, "K": 2.5}}
K_GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]


def log(m): print(f"[eth-ksweep-new] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    sp = importlib.util.spec_from_file_location("m_ks", ROOT / "scripts" / SRC)
    mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)

    log(f"원본 셀      {OLD_CELLS}")
    log(f"새 셀        {NEW_CELLS}")
    log(f"원본 K_GRID  {mod.K_GRID}")
    log(f"확장 K_GRID  {K_GRID}  (하단 0.4까지 -- README §5.6 전례)")

    mod.SIGNAL_CONFIG = {k: dict(v) for k, v in NEW_CELLS.items()}
    mod.K_GRID = list(K_GRID)
    rc = mod.main()
    if rc != 0:
        log("⚠️ksweep main() 실패"); return 1

    res = pd.read_csv(ROOT / "tmp/eth_kalman_demarker_gridscreen_20260831/ksweep_results.csv")
    dst = ROOT / "tmp/eth_kalman_demarker_ksweep_new_cells_20260903"
    dst.mkdir(parents=True, exist_ok=True)
    res.to_csv(dst / "ksweep_results.csv", index=False)

    rep = {"asset": "ETHUSDT", "new_cells": NEW_CELLS, "old_cells": OLD_CELLS,
           "k_grid": K_GRID, "selection_rule": "min(VAL,OOS) AUC (원본 규칙 그대로)",
           "holdout_touched": False, "signals": {}}

    log(""); log("=" * 92)
    log("K 재스윕 결과 (새 셀)")
    log("=" * 92)
    for name, cell in NEW_CELLS.items():
        sub = res[res.signal == name].sort_values("min_val_oos", ascending=False).reset_index(drop=True)
        if sub.empty:
            log(f"  {name}: 결과 없음"); continue
        best = sub.iloc[0]
        old_k = OLD_CELLS[name]["K"]
        oldrow = sub[abs(sub.K - old_k) < 1e-9]
        log("")
        log(f"  {name}  H={cell['horizon']}/GAP={cell['gap']}")
        log(f"    {'K':>6}{'VAL':>9}{'OOS':>9}{'min':>9}{'히트율':>9}{'n_train':>9}")
        for _, r in sub.sort_values("K").iterrows():
            mk = ""
            if abs(r.K - best.K) < 1e-9:
                mk = "  ⭐새 최적"
            if abs(r.K - old_k) < 1e-9:
                mk += "  (현행 K)"
            log(f"    {r.K:>6.2f}{r.val_auc:>9.4f}{r.oos_auc:>9.4f}{r.min_val_oos:>9.4f}"
                f"{r.hit_rate:>9.3f}{int(r.n_train):>9}{mk}")
        edge = best.K in (K_GRID[0], K_GRID[-1])
        log(f"    ⇒ 새 최적 K={best.K}  min(VAL,OOS)={best.min_val_oos:.4f}  "
            f"{'⚠️격자 경계' if edge else '✅내부값'}")
        if not oldrow.empty:
            o = oldrow.iloc[0]
            log(f"      현행 K={old_k} 를 새 셀에 그대로 쓰면 min={o.min_val_oos:.4f} "
                f"(Δ {best.min_val_oos - o.min_val_oos:+.4f}) -- K 재스윕이 필요한 이유")
        rep["signals"][name] = {
            "cell": cell, "best_K": float(best.K),
            "best": {"val_auc": float(best.val_auc), "oos_auc": float(best.oos_auc),
                     "min_val_oos": float(best.min_val_oos), "hit_rate": float(best.hit_rate),
                     "n_train": int(best.n_train)},
            "old_K_at_new_cell": (None if oldrow.empty else
                                  {"K": old_k, "min_val_oos": float(oldrow.iloc[0].min_val_oos),
                                   "val_auc": float(oldrow.iloc[0].val_auc),
                                   "oos_auc": float(oldrow.iloc[0].oos_auc)}),
            "k_at_grid_edge": bool(edge),
            "full_sweep": sub[["K", "val_auc", "oos_auc", "min_val_oos", "hit_rate",
                               "n_train", "n_val", "n_oos"]].to_dict("records")}

    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(""); log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
