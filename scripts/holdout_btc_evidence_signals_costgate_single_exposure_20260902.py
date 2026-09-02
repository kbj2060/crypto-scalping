#!/usr/bin/env python3
"""BTC 증거신호 경제성게이트 **HOLDOUT 단일 노출** (2026-04-01 ~).

⚠️⚠️**이 스크립트는 한 번만 돌린다.** 결과를 보고 셀을 바꿔 재실행하면 HOLDOUT이 오염된다.
그래서 셀과 통과 기준을 **코드에 하드코딩**해 사전등록한다.

## 사전등록 -- 셀 (VAL/OOS에서 이미 확정, HOLDOUT을 보고 고르지 않는다)

    1군 (주 판정 대상) -- 롱/숏 x VAL/OOS 4개 측면-구간 전부 정방향 양수였던 신호
      demarker_extreme          H= 8  SL=2.0  ARM=1.0  Trail=0.1
      short_term_return_z       H= 6  SL=3.0  ARM=1.5  Trail=0.1
      kalman_deviation_meanrev  H=10  SL=4.0  ARM=1.5  Trail=0.1

    2군 (참고 보고만, 판정에 쓰지 않음) -- 갭은 양수였으나 VAL 숏이 음수였던 신호
      orthogonal_combo          H= 8  SL=4.0  ARM=1.0  Trail=0.1
      taker_delta_climax        H= 6  SL=4.0  ARM=1.0  Trail=0.1

## 사전등록 -- 통과 기준 (전부 만족해야 통과)

    (1) HOLDOUT 정방향 평균 > 0
    (2) HOLDOUT 정방향 > 방향뒤집기
    (3) 무작위 진입 귀무분포(B=200) 대비 백분위 >= 95%
    (4) 롱/숏 각각의 (정방향 - 뒤집기) 갭 > 0

## 배경

VAL/OOS에서 이 셀들은 다음을 보였다(`btc_evidence_signal_economics_gate_20260902.md`):

    demarker  VAL +6.46 / OOS +8.91bp    str_z  VAL +4.94 / OOS +8.98bp
    kalman    VAL +2.41 / OOS +6.17bp

그리고 무작위 진입 귀무분포 대비 백분위 100%, 측면별 갭 전부 양수를 통과했다.
⚠️최초 "0/672 전패" 결론은 108바 인덱스 오프셋 버그였고 수정됐다(같은 문서 §0).
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
    "btcgate", ROOT / "scripts/gate_btc_evidence_signals_trailing_economics_20260902.py")
_g = importlib.util.module_from_spec(_S)
_S.loader.exec_module(_g)

_A = importlib.util.spec_from_file_location(
    "randnull", ROOT / "scripts/audit_btc_evidence_signals_costgate_random_entry_null_20260902.py")
_a = importlib.util.module_from_spec(_A)
_A.loader.exec_module(_a)

OUT = ROOT / "data/research/btc_evidence_signals_costgate_20260902/holdout_single_exposure.json"
B_NULL, SEED = 200, 20260902
HOLDOUT_START = pd.Timestamp("2026-04-01")

# ⭐사전등록 셀 -- 절대 수정 금지 (VAL/OOS 확정값)
PREREG = {
    "demarker_extreme":         {"tier": 1, "H": 8,  "cell": (2.0, 1.0, 0.1), "val": 6.46, "oos": 8.91},
    "short_term_return_z":      {"tier": 1, "H": 6,  "cell": (3.0, 1.5, 0.1), "val": 4.94, "oos": 8.98},
    "kalman_deviation_meanrev": {"tier": 1, "H": 10, "cell": (4.0, 1.5, 0.1), "val": 2.41, "oos": 6.17},
    "orthogonal_combo":         {"tier": 2, "H": 8,  "cell": (4.0, 1.0, 0.1), "val": 1.15, "oos": 10.95},
    "taker_delta_climax":       {"tier": 2, "H": 6,  "cell": (4.0, 1.0, 0.1), "val": 0.64, "oos": 2.90},
}


def log(m): print(f"[holdout] {m}", flush=True)


def one_cell_window(kl, fires, H, cell, start, end, flip=False):
    """지정 구간만 평가. `_g.run_grid`의 VAL/OOS 창을 HOLDOUT으로 갈아끼운다."""
    sv, so, sh = _g.VAL_START, _g.OOS_START, _g.HOLDOUT_START
    sg = (_g.SL_GRID, _g.ARM_GRID, _g.TRAIL_GRID)
    _g.VAL_START, _g.OOS_START, _g.HOLDOUT_START = start, end, end   # val창 = [start, end)
    _g.SL_GRID, _g.ARM_GRID, _g.TRAIL_GRID = [cell[0]], [cell[1]], [cell[2]]
    _g.ROUNDTRIP_COST_RATE = 0.001
    try:
        cells, ns = _g.run_grid(kl, fires, H)
    finally:
        _g.VAL_START, _g.OOS_START, _g.HOLDOUT_START = sv, so, sh
        _g.SL_GRID, _g.ARM_GRID, _g.TRAIL_GRID = sg
    c = cells[0]
    k = "flip" if flip else "fwd"
    return c[f"val_{k}_bp"], c["val_n"], c["val_wr"], ns["val"]


def main() -> int:
    t0 = time.time()
    if OUT.exists():
        log("⚠️⚠️이미 HOLDOUT 결과 파일이 있다 -- 단일 노출 원칙상 재실행 금지.")
        log(f"   기존: {OUT}")
        return 1
    log("=" * 78)
    log("⭐사전등록 (HOLDOUT을 보기 전에 확정한 값)")
    for n, p in PREREG.items():
        log(f"   [{p['tier']}군] {n:<26} H={p['H']:>2} SL={p['cell'][0]} ARM={p['cell'][1]} "
            f"Trail={p['cell'][2]}   (VAL {p['val']:+.2f} / OOS {p['oos']:+.2f})")
    log("   통과기준: (1) 정방향>0 (2) 정방향>뒤집기 (3) 무작위진입 백분위>=95% (4) 롱·숏 갭>0")
    log("=" * 78)

    kl = pd.read_csv(_g.KLINES)
    kl["timestamp"] = pd.to_datetime(kl["timestamp"], utc=True).dt.tz_localize(None)
    kl = kl.sort_values("timestamp").reset_index(drop=True)
    hend = kl["timestamp"].max() + pd.Timedelta(minutes=5)
    log(f"HOLDOUT 구간: {HOLDOUT_START} ~ {kl['timestamp'].max()}")
    rng = np.random.default_rng(SEED)
    rep = {"preregistered": {k: {**v, "cell": list(v["cell"])} for k, v in PREREG.items()},
           "holdout_start": str(HOLDOUT_START), "holdout_end": str(kl["timestamp"].max()),
           "B_null": B_NULL, "seed": SEED, "signals": {}}

    for name, rel, builder, prep, kind in _g.SIGNALS:
        if name not in PREREG:
            continue
        pr = PREREG[name]
        H, cell = pr["H"], pr["cell"]
        log("")
        log(f"=== [{pr['tier']}군] {name}  SL={cell[0]} ARM={cell[1]} Trail={cell[2]} H={H} ===")
        fires, frame = _g.build_fires(name, rel, builder, prep, kind)
        for d in (fires, frame):
            d["timestamp"] = pd.to_datetime(d["timestamp"])
            if d["timestamp"].dt.tz is not None:
                d["timestamp"] = d["timestamp"].dt.tz_localize(None)
        fh = fires.loc[fires["timestamp"] >= HOLDOUT_START].reset_index(drop=True)
        frh = frame.loc[frame["timestamp"] >= HOLDOUT_START].reset_index(drop=True)
        if len(fh) < 30:
            log(f"  ⚠️HOLDOUT fires {len(fh)}건 -- 표본 부족")
            rep["signals"][name] = {"error": f"too few fires: {len(fh)}"}
            continue

        fv, fn, fw, ncand = one_cell_window(kl, fh, H, cell, HOLDOUT_START, hend)
        xv, _n, _w, _c = one_cell_window(kl, fh, H, cell, HOLDOUT_START, hend, flip=True)
        log(f"  fires {len(fh):,}  후보 {ncand}  체결 {fn}")
        log(f"  ⭐HOLDOUT 정방향 **{fv:+.2f}bp**  뒤집기 {xv:+.2f}bp  갭 {fv-xv:+.2f}  승률 {fw*100:.1f}%")

        nl = int((fh["side"].astype(str) == "bottom").sum()); nsh = len(fh) - nl
        nulls = []
        for _ in range(B_NULL):
            rf = _a.random_fires(frh, nl, nsh, rng)
            a, _b, _c2, _d = one_cell_window(kl, rf, H, cell, HOLDOUT_START, hend)
            nulls.append(a)
        nulls = np.array(nulls)
        pct = float((nulls < fv).mean() * 100)
        log(f"  무작위진입 귀무 평균 {nulls.mean():+.2f}bp → 백분위 **{pct:.1f}%** "
            f"{'✅' if pct >= 95 else '❌'}")

        sides = {}
        for lab, m in (("롱", fh["side"].astype(str) == "bottom"),
                       ("숏", fh["side"].astype(str) == "top")):
            sub = fh.loc[m].reset_index(drop=True)
            if len(sub) < 20:
                log(f"  {lab}: 표본 부족 {len(sub)}"); continue
            a, an, _w2, _c3 = one_cell_window(kl, sub, H, cell, HOLDOUT_START, hend)
            b, _bn, _w3, _c4 = one_cell_window(kl, sub.assign(
                side=np.where(sub["side"] == "bottom", "top", "bottom")), H, cell,
                HOLDOUT_START, hend, flip=False)
            sides[lab] = {"n": int(an), "fwd": a, "flip": b, "gap": a - b}
            log(f"  {lab} n={an:>5}  정 {a:+7.2f}  뒤 {b:+7.2f}  갭 {a-b:+7.2f} "
                f"{'✅' if a - b > 0 else '❌'}")

        c1, c2 = fv > 0, fv > xv
        c3 = pct >= 95
        c4 = len(sides) == 2 and all(s["gap"] > 0 for s in sides.values())
        ok = c1 and c2 and c3 and c4
        log(f"  ⇒ (1){'✅' if c1 else '❌'} (2){'✅' if c2 else '❌'} "
            f"(3){'✅' if c3 else '❌'} (4){'✅' if c4 else '❌'}  "
            f"⇒ {'✅**HOLDOUT 생존**' if ok else '❌미통과'}")
        rep["signals"][name] = {
            "tier": pr["tier"], "cell": list(cell), "H": H,
            "n_fires": int(len(fh)), "n_candidates": ncand, "n_trades": int(fn),
            "fwd_bp": fv, "flip_bp": xv, "gap_bp": fv - xv, "win_rate": fw,
            "null_mean": float(nulls.mean()), "null_pctile": pct,
            "sides": sides, "criteria": {"c1": c1, "c2": c2, "c3": c3, "c4": c4},
            "passed": bool(ok), "val_bp": pr["val"], "oos_bp": pr["oos"]}

    log("")
    log("=" * 78)
    log("=== HOLDOUT 종합 (단일 노출, 재실행 금지) ===")
    log(f"{'':4}{'신호':<26}{'VAL':>8}{'OOS':>8}{'HOLDOUT':>10}{'뒤집기':>9}{'귀무%':>8}  판정")
    for k, v in rep["signals"].items():
        if "error" in v:
            log(f"    {k:<26} ⚠️{v['error']}"); continue
        log(f"{v['tier']}군 {k:<26}{v['val_bp']:>8.2f}{v['oos_bp']:>8.2f}"
            f"{v['fwd_bp']:>10.2f}{v['flip_bp']:>9.2f}{v['null_pctile']:>8.1f}"
            f"  {'✅생존' if v['passed'] else '❌미통과'}")
    t1 = [k for k, v in rep["signals"].items() if v.get("passed") and v.get("tier") == 1]
    log("")
    log(f"  ⇒ ⭐**1군 생존: {t1 if t1 else '없음'}**")
    rep["tier1_survivors"] = t1
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    log("⚠️이 파일이 존재하면 재실행이 차단된다(단일 노출).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
