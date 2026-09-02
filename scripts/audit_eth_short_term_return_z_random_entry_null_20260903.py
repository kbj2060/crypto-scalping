#!/usr/bin/env python3
"""ETH `short_term_return_z` 경제성게이트 통과의 **적대적 검증**.

게이트에서 96/96 동시양수 · 방향뒤집기 96/96 · ARM>=1.0 진짜 72셀이 나왔다.
이 정도로 완전한 결과는 그 자체가 경고이므로, BTC/XRP와 동일한 사전등록 3종 검정을 건다.

  A. ⭐무작위 진입 귀무분포(단일 셀, B=200) -- 실제가 95백분위 위여야 타이밍에 정보가 있다.
  B. 무작위 진입 귀무분포(격자 전체, B=20) -- 통과 셀 개수 자체가 우연인지.
  C. 측면별 갭 -- 롱만/숏만 따로, 양쪽 다 정방향>뒤집기여야 한다.

무작위 진입 풀은 **klines에서 직접 ATR(14봉)을 계산**해 만든다 -- 그 봉의 실제 ATR을 써야
"같은 변동성 조건에서 타이밍만 무작위"라는 귀무가 성립한다.

⚠️HOLDOUT 미터치. VAL+OOS만 본다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_S = importlib.util.spec_from_file_location(
    "ethstrzgate", ROOT / "scripts/gate_eth_short_term_return_z_trailing_economics_20260903.py")
_g = importlib.util.module_from_spec(_S)
_S.loader.exec_module(_g)

REPORT = ROOT / "data/research/eth_short_term_return_z_costgate_20260903/report.json"
OUT = ROOT / "data/research/eth_short_term_return_z_costgate_20260903/random_entry_null.json"
B_CELL, B_GRID, SEED = 200, 20, 20260903


def log(m): print(f"[eth-null] {m}", flush=True)


def one_cell(kl, fires, H, cell):
    save = (_g.SL_GRID, _g.ARM_GRID, _g.TRAIL_GRID)
    _g.SL_GRID, _g.ARM_GRID, _g.TRAIL_GRID = [cell[0]], [cell[1]], [cell[2]]
    try:
        cells, ns = _g.run_grid(kl, fires, H)
    finally:
        _g.SL_GRID, _g.ARM_GRID, _g.TRAIL_GRID = save
    return cells[0]["val_fwd_bp"], cells[0]["oos_fwd_bp"], ns


def atr_pool(kl):
    """klines에서 14봉 ATR을 계산해 무작위 진입 후보 풀을 만든다."""
    h, l, c = kl["high"].to_numpy(), kl["low"].to_numpy(), kl["close"].to_numpy()
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    pool = pd.DataFrame({"timestamp": kl["timestamp"], "atr_pct": atr / c})
    return pool.loc[pool["atr_pct"].notna()
                    & (pool["timestamp"] < _g.HOLDOUT_START)].reset_index(drop=True)


def random_fires(pool, n_long, n_short, rng):
    sub = pool.iloc[np.sort(rng.choice(len(pool), size=n_long + n_short, replace=False))]
    sub = sub[["timestamp", "atr_pct"]].reset_index(drop=True)
    side = np.array(["bottom"] * n_long + ["top"] * n_short)
    rng.shuffle(side)
    sub["side"] = side
    return sub[["timestamp", "side", "atr_pct"]]


def main() -> int:
    t0 = time.time()
    rep_in = json.loads(REPORT.read_text())
    best = rep_in["best_arm_ge_1"]
    if best is None:
        log("통과 셀 없음 -- 검정 불필요")
        return 0
    cell = (best["sl"], best["arm"], best["trail"])
    H = rep_in["horizon_bars"]

    kl = pd.read_csv(_g.KLINES)
    kl["timestamp"] = pd.to_datetime(kl["timestamp"], utc=True).dt.tz_localize(None)
    kl = kl.sort_values("timestamp").reset_index(drop=True)
    pool = atr_pool(kl)
    log(f"무작위 진입 풀 {len(pool):,}봉 (HOLDOUT 이전, ATR14 유효)")

    fires = pd.read_csv(_g.FIRES_CSV, usecols=["timestamp", "side", "atr_pct"])
    fires["timestamp"] = pd.to_datetime(fires["timestamp"], utc=True).dt.tz_localize(None)
    fires = fires.loc[fires["timestamp"] < _g.HOLDOUT_START].reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    log("")
    log(f"=== short_term_return_z  셀 SL={cell[0]} ARM={cell[1]} Trail={cell[2]}  H={H} ===")
    rv, ro, _ = one_cell(kl, fires, H, cell)
    nl = int((fires["side"].astype(str) == "bottom").sum())
    ns_ = len(fires) - nl
    log(f"  실제: VAL {rv:+.2f}  OOS {ro:+.2f}bp   (롱 {nl} / 숏 {ns_})")

    # ---- A ----
    nv, no = [], []
    for _ in range(B_CELL):
        a, b, _n = one_cell(kl, random_fires(pool, nl, ns_, rng), H, cell)
        nv.append(a); no.append(b)
    nv, no = np.array(nv), np.array(no)
    pv = float((nv < rv).mean() * 100); po = float((no < ro).mean() * 100)
    log(f"  A) 무작위진입 귀무 VAL 평균 {nv.mean():+.2f} → 실제 백분위 **{pv:.1f}%** "
        f"{'✅' if pv >= 95 else '❌'}")
    log(f"     무작위진입 귀무 OOS 평균 {no.mean():+.2f} → 실제 백분위 **{po:.1f}%** "
        f"{'✅' if po >= 95 else '❌'}")

    # ---- B ----
    cnts = []
    for _ in range(B_GRID):
        cs, _n = _g.run_grid(kl, random_fires(pool, nl, ns_, rng), H)
        cnts.append(sum(1 for c in cs
                        if c["val_fwd_bp"] > 0 and c["oos_fwd_bp"] > 0
                        and c["val_fwd_bp"] > c["val_flip_bp"]
                        and c["oos_fwd_bp"] > c["oos_flip_bp"] and c["arm"] >= 1.0))
    cnts = np.array(cnts)
    obs = rep_in["n_genuine_arm_ge_1"]
    pc = float((cnts < obs).mean() * 100)
    log(f"  B) 통과셀 {obs} vs 무작위 평균 {cnts.mean():.1f} (최대 {cnts.max()}) "
        f"백분위 **{pc:.1f}%** {'✅' if pc >= 95 else '❌'}")

    # ---- C ----
    sides = {}
    for lab, m in (("롱", fires["side"].astype(str) == "bottom"),
                   ("숏", fires["side"].astype(str) == "top")):
        sub = fires.loc[m].reset_index(drop=True)
        fv, fo, _n = one_cell(kl, sub, H, cell)
        xv, xo, _n = one_cell(kl, sub.assign(
            side=np.where(sub["side"] == "bottom", "top", "bottom")), H, cell)
        sides[lab] = {"n": len(sub), "val_fwd": fv, "val_flip": xv, "oos_fwd": fo,
                      "oos_flip": xo, "val_gap": fv - xv, "oos_gap": fo - xo}
        log(f"  C) {lab} n={len(sub):>5}  VAL 정 {fv:+7.2f} 뒤 {xv:+7.2f} 갭 {fv-xv:+7.2f} | "
            f"OOS 정 {fo:+7.2f} 뒤 {xo:+7.2f} 갭 {fo-xo:+7.2f} "
            f"{'✅' if (fv-xv) > 0 and (fo-xo) > 0 else '❌'}")

    both = all(s["val_gap"] > 0 and s["oos_gap"] > 0 for s in sides.values())
    ok = pv >= 95 and po >= 95 and pc >= 95 and both
    log(f"  ⇒ short_term_return_z: {'✅**진짜**' if ok else '❌검증 미통과'}")

    out = {"asset": "ETHUSDT", "signal": "short_term_return_z", "cell": list(cell), "H": H,
           "B_cell": B_CELL, "B_grid": B_GRID, "seed": SEED, "holdout_touched": False,
           "real": {"val_bp": rv, "oos_bp": ro},
           "A_random_entry": {"val_null_mean": float(nv.mean()), "val_pctile": pv,
                              "oos_null_mean": float(no.mean()), "oos_pctile": po},
           "B_pass_count": {"observed": obs, "null_mean": float(cnts.mean()),
                            "null_max": int(cnts.max()), "pctile": pc},
           "C_side": sides, "passed": bool(ok), "runtime_sec": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({out['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
