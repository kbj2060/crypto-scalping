#!/usr/bin/env python3
"""XRP 증거신호 경제성게이트 통과의 **적대적 검증** -- 진입 타이밍이 무작위보다 나은가.

## 왜 필요한가

XRP 게이트에서 **5종 전부** 통과했다(96/96, 95/96, 84/96, 18/96, 15/96). 이렇게 완전한
결과는 그 자체가 경고다 -- 2026-09-02 BTC에서 "0/672 전패"가 인덱스 버그였듯,
반대 방향의 완전함도 계측 아티팩트일 수 있다.

**트레일링스톱은 진입이 무작위여도 높은 승률을 만든다**는 게 이 저장소의 기록된 사실이다
(ETH orthogonal_combo: 승률 91~96%인데 무작위 진입도 83~85%). 그러므로 "통과했다"만으로는
부족하고 **무작위 진입 대비 우위**를 봐야 한다.

## 검정 3종 (`audit_btc_evidence_signals_costgate_random_entry_null_20260902.py` 그대로)

  A. ⭐**무작위 진입 귀무분포(단일 셀, B=200)** -- 실제 통과 셀(ARM>=1.0 최선)을 고정하고,
     같은 개수/같은 롱숏 비율의 진입 시점을 **무작위로** 뽑아 같은 브래킷을 돌린다.
     실제 성과가 95번째 백분위 위여야 "타이밍에 정보가 있다".
  B. **무작위 진입 귀무분포(격자 전체, B=20)** -- 통과 셀 개수 자체가 우연인지.
  C. **측면별 갭** -- 롱만/숏만 따로. 양쪽 다 정방향>뒤집기여야 한다.

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
    "xrpgate", ROOT / "scripts/gate_xrp_evidence_signals_trailing_economics_20260903.py")
_g = importlib.util.module_from_spec(_S)
_S.loader.exec_module(_g)

REPORT = ROOT / "data/research/xrp_evidence_signals_costgate_20260903/report.json"
OUT = ROOT / "data/research/xrp_evidence_signals_costgate_20260903/random_entry_null.json"
B_CELL, B_GRID, SEED = 200, 20, 20260903


def log(m): print(f"[xrp-null] {m}", flush=True)


def one_cell(kl, fires, H, cell, flip=False):
    _g.ROUNDTRIP_COST_RATE = 0.001
    sl, arm, tr = cell
    save = (_g.SL_GRID, _g.ARM_GRID, _g.TRAIL_GRID)
    _g.SL_GRID, _g.ARM_GRID, _g.TRAIL_GRID = [sl], [arm], [tr]
    try:
        cells, ns = _g.run_grid(kl, fires, H)
    finally:
        _g.SL_GRID, _g.ARM_GRID, _g.TRAIL_GRID = save
    c = cells[0]
    k = "flip" if flip else "fwd"
    return c[f"val_{k}_bp"], c[f"oos_{k}_bp"], ns


def random_fires(frame, n_long, n_short, rng):
    """같은 개수/같은 롱숏 비율의 **무작위 진입 시점**을 만든다(ATR은 그 봉의 실제 ATR).

    ⚠️XRP 모듈마다 `atr`/`atr_pct` 중 무엇을 만드는지 다르다 -- 있는 쪽을 쓴다."""
    if "atr_pct" in frame.columns:
        f = frame.loc[frame["atr_pct"].notna()].reset_index(drop=True)
        sub = f.iloc[np.sort(rng.choice(len(f), size=n_long + n_short, replace=False))]
        sub = sub[["timestamp", "atr_pct"]].reset_index(drop=True)
    else:
        f = frame.loc[frame["atr"].notna() & frame["close"].notna()].reset_index(drop=True)
        sub = f.iloc[np.sort(rng.choice(len(f), size=n_long + n_short, replace=False))]
        sub = sub[["timestamp", "atr", "close"]].reset_index(drop=True)
        sub["atr_pct"] = sub["atr"] / sub["close"]
    side = np.array(["bottom"] * n_long + ["top"] * n_short)
    rng.shuffle(side)
    sub["side"] = side
    return sub[["timestamp", "side", "atr_pct"]]


def main() -> int:
    t0 = time.time()
    rep_in = json.loads(REPORT.read_text())["signals"]
    kl = pd.read_csv(_g.KLINES)
    kl["timestamp"] = pd.to_datetime(kl["timestamp"], utc=True).dt.tz_localize(None)
    kl = kl.sort_values("timestamp").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    out = {"asset": "XRPUSDT", "B_cell": B_CELL, "B_grid": B_GRID, "seed": SEED,
           "holdout_touched": False, "signals": {}}

    for name, rel, builder, prep, kind in _g.SIGNALS:
        v = rep_in.get(name, {})
        g1 = v.get("genuine_arm_ge_1") or []
        if not g1:
            log(f"{name}: 통과 셀 없음 -- 건너뜀")
            continue
        best = max(g1, key=lambda c: min(c["val_fwd_bp"], c["oos_fwd_bp"]))
        cell = (best["sl"], best["arm"], best["trail"])
        H = v["horizon_bars"]
        log("")
        log(f"=== {name}  셀 SL={cell[0]} ARM={cell[1]} Trail={cell[2]}  H={H} ===")

        fires, frame = _g.build_fires(name, rel, builder, prep, kind)
        fires["timestamp"] = pd.to_datetime(fires["timestamp"])
        if fires["timestamp"].dt.tz is not None:
            fires["timestamp"] = fires["timestamp"].dt.tz_localize(None)
        fires = fires.loc[fires["timestamp"] < _g.HOLDOUT_START].reset_index(drop=True)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        if frame["timestamp"].dt.tz is not None:
            frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
        frame = frame.loc[frame["timestamp"] < _g.HOLDOUT_START].reset_index(drop=True)

        rv, ro, _ = one_cell(kl, fires, H, cell)
        nl = int((fires["side"].astype(str) == "bottom").sum())
        ns_ = len(fires) - nl
        log(f"  실제: VAL {rv:+.2f}  OOS {ro:+.2f}bp   (롱 {nl} / 숏 {ns_})")

        # ---- A) 무작위 진입 귀무 (단일 셀) ----
        nv, no = [], []
        for _ in range(B_CELL):
            rf = random_fires(frame, nl, ns_, rng)
            a, b, _n = one_cell(kl, rf, H, cell)
            nv.append(a); no.append(b)
        nv, no = np.array(nv), np.array(no)
        pv = float((nv < rv).mean() * 100); po = float((no < ro).mean() * 100)
        log(f"  A) 무작위진입 귀무 VAL 평균 {nv.mean():+.2f} → 실제 백분위 **{pv:.1f}%** "
            f"{'✅' if pv >= 95 else '❌'}")
        log(f"     무작위진입 귀무 OOS 평균 {no.mean():+.2f} → 실제 백분위 **{po:.1f}%** "
            f"{'✅' if po >= 95 else '❌'}")

        # ---- B) 통과 셀 개수 귀무 (격자 전체) ----
        cnts = []
        for _ in range(B_GRID):
            rf = random_fires(frame, nl, ns_, rng)
            _g.ROUNDTRIP_COST_RATE = 0.001
            cs, _n = _g.run_grid(kl, rf, H)
            cnts.append(sum(1 for c in cs
                            if c["val_fwd_bp"] > 0 and c["oos_fwd_bp"] > 0
                            and c["val_fwd_bp"] > c["val_flip_bp"]
                            and c["oos_fwd_bp"] > c["oos_flip_bp"] and c["arm"] >= 1.0))
        cnts = np.array(cnts)
        obs = v["n_genuine_arm_ge_1"]
        pc = float((cnts < obs).mean() * 100)
        log(f"  B) 통과셀 {obs} vs 무작위 평균 {cnts.mean():.1f} (최대 {cnts.max()}) "
            f"백분위 **{pc:.1f}%** {'✅' if pc >= 95 else '❌'}")

        # ---- C) 측면별 갭 ----
        sides = {}
        for lab, m in (("롱", fires["side"].astype(str) == "bottom"),
                       ("숏", fires["side"].astype(str) == "top")):
            sub = fires.loc[m].reset_index(drop=True)
            if len(sub) < 30:
                log(f"  C) {lab}: 표본 부족 {len(sub)}"); continue
            fv, fo, _n = one_cell(kl, sub, H, cell)
            xv, xo, _n = one_cell(kl, sub.assign(
                side=np.where(sub["side"] == "bottom", "top", "bottom")), H, cell)
            sides[lab] = {"n": len(sub), "val_fwd": fv, "val_flip": xv,
                          "oos_fwd": fo, "oos_flip": xo,
                          "val_gap": fv - xv, "oos_gap": fo - xo}
            log(f"  C) {lab} n={len(sub):>5}  VAL 정 {fv:+7.2f} 뒤 {xv:+7.2f} 갭 {fv-xv:+7.2f} | "
                f"OOS 정 {fo:+7.2f} 뒤 {xo:+7.2f} 갭 {fo-xo:+7.2f} "
                f"{'✅' if (fv-xv) > 0 and (fo-xo) > 0 else '❌'}")

        both = all(s["val_gap"] > 0 and s["oos_gap"] > 0 for s in sides.values()) and len(sides) == 2
        ok = pv >= 95 and po >= 95 and pc >= 95 and both
        log(f"  ⇒ {name}: {'✅**진짜**' if ok else '❌검증 미통과'}")
        out["signals"][name] = {
            "cell": list(cell), "H": H, "real": {"val_bp": rv, "oos_bp": ro},
            "A_random_entry": {"val_null_mean": float(nv.mean()), "val_pctile": pv,
                               "oos_null_mean": float(no.mean()), "oos_pctile": po},
            "B_pass_count": {"observed": obs, "null_mean": float(cnts.mean()),
                             "null_max": int(cnts.max()), "pctile": pc},
            "C_side": sides, "passed": bool(ok)}

    log("")
    log("=== 종합 ===")
    for k, v2 in out["signals"].items():
        a = v2["A_random_entry"]
        log(f"  {k:<26} A(VAL/OOS) {a['val_pctile']:5.1f}/{a['oos_pctile']:5.1f}%  "
            f"B {v2['B_pass_count']['pctile']:5.1f}%  {'✅' if v2['passed'] else '❌'}")
    ok = [k for k, v2 in out["signals"].items() if v2["passed"]]
    log(f"  ⇒ 검증 통과: {ok if ok else '없음'}")
    out["passed_signals"] = ok
    out["runtime_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({out['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
