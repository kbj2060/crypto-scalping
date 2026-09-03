#!/usr/bin/env python3
"""XRP 인과적 증거신호 2구성 **정식 승격 관문** -- 적대적 검증 + HOLDOUT 단일노출.

## 대상 (⭐사전등록 -- VAL/OOS만 보고 확정, 이 파일 작성 시점에 고정)

2026-09-03 인과적(앵커 없는) 경로에서 순환이동 플라시보 귀무까지 통과한 2구성:

| 구성 | 진짜셀 | 귀무평균 | 백분위 |
|---|---|---|---|
| `str_z_chopgate` | 3 | **0.03** | **100.0%** |
| `consensus_K3_GAP3_H12` | 5 | 0.95 | 97.5% |

**셀은 VAL/OOS의 min(VAL,OOS) argmax로 고정한다:**

    str_z_chopgate         SL=1.5 ARM=2.0 Trail=0.1  (VAL +1.75 / OOS +1.57)
    consensus_K3_GAP3_H12  SL=1.5 ARM=2.0 Trail=0.3  (VAL +1.26 / OOS +2.34)

⚠️**이 셀들은 사후 변경 금지.** HOLDOUT을 본 뒤 다른 셀로 갈아타면 그 자체가 홀드아웃 오염이다.

## 관문 (실행 전 고정)

**Phase A -- 적대적 검증 (VAL/OOS만, HOLDOUT 미터치)**
  A1. **무작위진입 귀무 B=200**: 같은 개수/같은 롱숏 비율의 진입 시점을 무작위로 뽑아
      같은 셀·같은 브래킷을 돌린다. 실제가 VAL·OOS 둘 다 **95백분위 이상**이어야 한다.
      (플라시보 순환이동과 다른 질문이다 -- 저건 "트리거 위치", 이건 "타이밍 자체".)
  A2. **측면별 갭**: 롱만/숏만 따로. 양쪽 다 VAL·OOS에서 정방향 > 뒤집기여야 한다.
      (BTC 1h에서 숏만 맞고 롱이 −73bp였던 전례.)

**Phase B -- HOLDOUT 단일노출** (Phase A 통과 구성만)
  B1. HOLDOUT 평균 bp > 0
  B2. 정방향 > 뒤집기
  B3. **트레이드별 부트스트랩 95%CI 하한 > 0** (오늘 BTC 감사의 교훈 -- 평균만으로는 부족.
      BTC "생존 3종" 중 CI가 0을 제외한 건 하나뿐이었다.)

셋 다 만족해야 승격 후보다.

## ⚠️창 규율

XRP 증거신호 HOLDOUT(2026-04-01~)은 **분류 AUC로는 소진**됐지만 **경제성 게이트로는 한 번도
노출된 적이 없다**(오늘 게이트·지연확정·인과 실험 전부 `holdout_touched: False`).
⇒ 경제성 축에서 이번이 **첫 노출이자 마지막 노출**이다. 재실행은 근거로 쓸 수 없다.

⚠️2구성을 동시에 노출한다. **사전등록된 2-arm 검정**이므로 낚시가 아니지만,
다중성이 있으므로 한쪽만 통과하면 그 사실을 명시한다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402

_S = importlib.util.spec_from_file_location(
    "surv", ROOT / "scripts/research_xrp_survivors_detail_and_placebo_20260903.py")
_s = importlib.util.module_from_spec(_S)
_S.loader.exec_module(_s)
_f, _t = _s._f, _s._t

OUT = ROOT / "data/research/xrp_causal_evidence_formal_gate_20260903.json"

MARGIN_FRACTION, LEVERAGE, COST = 0.30, 3.0, 0.001      # ⭐10bp 불변
VAL_START, OOS_START, HOLDOUT_START = _t.VAL_START, _t.OOS_START, _t.HOLDOUT_START
B_NULL, B_BOOT, SEED = 200, 10_000, 20260903

# ⭐⭐사전등록 -- 절대 수정 금지
PREREG = {
    "str_z_chopgate": {
        "kind": "gated", "params": {"signal": "short_term_return_z"},
        "cell": (1.5, 2.0, 0.1), "val": 1.75, "oos": 1.57,
        "placebo_pctile": 100.0},
    "consensus_K3_GAP3_H12": {
        "kind": "consensus", "params": {"K": 3, "gap": 3, "H": 12},
        "cell": (1.5, 2.0, 0.3), "val": 1.26, "oos": 2.34,
        "placebo_pctile": 97.5},
}
CRITERIA = {"A1_random_entry_pctile": 95.0, "A2_side_gap_both": True,
            "B1_holdout_bp_positive": True, "B2_beats_flip": True,
            "B3_bootstrap_ci_excludes_zero": True}


def log(m): print(f"[gate] {m}", flush=True)


def one_cell(kl, dec, is_long, atr, H, cell, window, flip=False, ledger=False):
    """지정 창에서 사전등록 셀 1개만 평가."""
    sl, arm, tr = cell
    ts = kl["timestamp"]
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    lo, hi = window
    el = set(np.flatnonzero(purged_decision_mask(ts, start=lo, end=hi, horizon_bars=H)).tolist())
    m = np.array([d in el for d in dec])
    if not m.any():
        return (None, None) if ledger else None
    sgn = -1.0 if flip else 1.0
    r = simulate_single_position(
        timestamps=ts, open_px=o, high=h, low=l, close=c,
        decision_indices=dec[m], scores=(np.where(is_long, 1.0, -1.0) * sgn)[m],
        tp_moves=np.full(int(m.sum()), 999.0), sl_moves=(sl * atr)[m],
        upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=H,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=COST,
        arm_moves=(arm * atr)[m], trail_moves=(tr * atr)[m])
    led = r.ledger
    st = {"n": int(len(led)),
          "mean_bp": float(led["trade_return"].mean() * 1e4) if len(led) else float("nan"),
          "win_rate": float((led["price_move"] > 0).mean()) if len(led) else float("nan")}
    return (st, led) if ledger else st


def main() -> int:
    t0 = time.time()
    frame, kl, atr, cm = _s.build_ctx()
    rng = np.random.default_rng(SEED)
    W = {"VAL": (VAL_START, OOS_START), "OOS": (OOS_START, HOLDOUT_START),
         "HOLDOUT": (HOLDOUT_START, frame["timestamp"].max() + pd.Timedelta(minutes=5))}
    log(f"프레임 {len(frame):,}봉 | HOLDOUT {int((frame['timestamp'] >= HOLDOUT_START).sum()):,}봉")
    log(f"사전등록 셀: " + " / ".join(f"{k} {v['cell']}" for k, v in PREREG.items()))
    log("⚠️경제성 축 HOLDOUT 첫 노출 -- 재실행은 근거로 쓸 수 없다")

    rep = {"preregistered": PREREG, "criteria": CRITERIA, "B_null": B_NULL, "B_boot": B_BOOT,
           "seed": SEED, "cost_bp": 10.0, "arms": {}}

    for label, pr in PREREG.items():
        cb, ct, H = _s.triggers(frame, pr["kind"], pr["params"])
        gate = (pr["kind"] == "gated")
        sel = (cb | ct)
        if gate:
            sel = sel & np.nan_to_num(cm, nan=False)
        idx = np.flatnonzero(sel)
        idx = idx[np.isfinite(atr[idx]) & (atr[idx] > 0) & (idx < len(kl) - 1)]
        is_long = cb[idx]
        cell = pr["cell"]
        log(""); log("=" * 76); log(f"{label}  셀 SL={cell[0]} ARM={cell[1]} Trail={cell[2]} H={H}")
        log("=" * 76)

        arm_out = {"cell": list(cell), "H": H, "n_fires": int(len(idx))}

        # ---------- Phase A1: 무작위진입 귀무 ----------
        pool = np.flatnonzero(np.isfinite(atr) & (atr > 0))
        pool = pool[pool < len(kl) - 1]
        nl, ns_ = int(is_long.sum()), int((~is_long).sum())
        real = {w: one_cell(kl, idx, is_long, atr[idx], H, cell, W[w]) for w in ("VAL", "OOS")}
        log(f"  실제 VAL {real['VAL']['mean_bp']:+.2f}bp(n={real['VAL']['n']}) "
            f"OOS {real['OOS']['mean_bp']:+.2f}bp(n={real['OOS']['n']})  (롱 {nl}/숏 {ns_})")
        null = {"VAL": [], "OOS": []}
        for b in range(B_NULL):
            pick = np.sort(rng.choice(pool, size=nl + ns_, replace=False))
            sd = np.zeros(len(pick), dtype=bool); sd[:nl] = True; rng.shuffle(sd)
            for w in ("VAL", "OOS"):
                st = one_cell(kl, pick, sd, atr[pick], H, cell, W[w])
                if st and np.isfinite(st["mean_bp"]):
                    null[w].append(st["mean_bp"])
            if (b + 1) % 50 == 0:
                log(f"   ...귀무 {b+1}/{B_NULL}")
        a1 = {}
        for w in ("VAL", "OOS"):
            arr = np.array(null[w])
            pct = float((arr < real[w]["mean_bp"]).mean() * 100)
            a1[w] = {"null_mean": float(arr.mean()), "pctile": pct,
                     "passed": bool(pct >= CRITERIA["A1_random_entry_pctile"])}
            log(f"  A1 {w}: 무작위 평균 {arr.mean():+.2f}bp → 실제 백분위 **{pct:.1f}%** "
                f"{'✅' if a1[w]['passed'] else '❌'}")
        a1_ok = all(v["passed"] for v in a1.values())

        # ---------- Phase A2: 측면별 갭 ----------
        a2, a2_ok = {}, True
        for lab, mask in (("롱", is_long), ("숏", ~is_long)):
            sub, sl_ = idx[mask], is_long[mask]
            if len(sub) < 30:
                log(f"  A2 {lab}: 표본 부족({len(sub)})"); a2_ok = False; continue
            d = {}
            for w in ("VAL", "OOS"):
                f_ = one_cell(kl, sub, sl_, atr[sub], H, cell, W[w])
                x_ = one_cell(kl, sub, sl_, atr[sub], H, cell, W[w], flip=True)
                gap = (f_["mean_bp"] - x_["mean_bp"]) if (f_ and x_) else float("nan")
                d[w] = {"fwd": f_["mean_bp"] if f_ else None,
                        "flip": x_["mean_bp"] if x_ else None, "gap": gap,
                        "n": f_["n"] if f_ else 0}
            ok = all(np.isfinite(d[w]["gap"]) and d[w]["gap"] > 0 for w in ("VAL", "OOS"))
            a2_ok &= ok
            a2[lab] = {**d, "passed": bool(ok)}
            log(f"  A2 {lab} n={d['VAL']['n']}/{d['OOS']['n']}  "
                f"VAL 정{d['VAL']['fwd']:+.2f} 뒤{d['VAL']['flip']:+.2f} 갭{d['VAL']['gap']:+.2f} | "
                f"OOS 정{d['OOS']['fwd']:+.2f} 뒤{d['OOS']['flip']:+.2f} 갭{d['OOS']['gap']:+.2f} "
                f"{'✅' if ok else '❌'}")

        phase_a = bool(a1_ok and a2_ok)
        log(f"  ⇒ Phase A: {'✅통과' if phase_a else '❌미통과 -- HOLDOUT 열지 않는다'}")
        arm_out.update({"real_val_oos": real, "A1": a1, "A2": a2, "phase_a_passed": phase_a})

        # ---------- Phase B: HOLDOUT 단일노출 ----------
        if not phase_a:
            arm_out["phase_b"] = {"skipped": "Phase A 미통과"}
            rep["arms"][label] = arm_out
            continue
        log("")
        log("  ⭐Phase B -- HOLDOUT 단일노출 (사전등록 셀 고정)")
        hf, led = one_cell(kl, idx, is_long, atr[idx], H, cell, W["HOLDOUT"], ledger=True)
        hx = one_cell(kl, idx, is_long, atr[idx], H, cell, W["HOLDOUT"], flip=True)
        if hf is None or led is None or not len(led):
            log("   ⚠️HOLDOUT 트레이드 0건"); arm_out["phase_b"] = {"error": "no trades"}
            rep["arms"][label] = arm_out; continue
        v = led["trade_return"].to_numpy() * 1e4
        bs = rng.integers(0, len(v), size=(B_BOOT, len(v)))
        boot = v[bs].mean(axis=1)
        lo95, hi95 = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        b1 = hf["mean_bp"] > 0
        b2 = hf["mean_bp"] > hx["mean_bp"]
        b3 = lo95 > 0
        log(f"   HOLDOUT {hf['mean_bp']:+.2f}bp (뒤 {hx['mean_bp']:+.2f})  n={hf['n']}  "
            f"승률 {hf['win_rate']*100:.1f}%")
        log(f"   부트스트랩 95%CI [{lo95:+.2f}, {hi95:+.2f}]bp")
        log(f"   B1 평균>0 {'✅' if b1 else '❌'} | B2 뒤집기우위 {'✅' if b2 else '❌'} | "
            f"B3 CI하한>0 {'✅' if b3 else '❌'}")
        ok = bool(b1 and b2 and b3)
        log(f"   ⇒ {'⭐**승격 후보**' if ok else '❌승격 미달'}")
        arm_out["phase_b"] = {"fwd": hf, "flip": hx, "ci95": [lo95, hi95],
                              "B1": bool(b1), "B2": bool(b2), "B3": bool(b3), "passed": ok}
        arm_out["promoted"] = ok
        rep["arms"][label] = arm_out

    log(""); log("=" * 78)
    log("종합 -- 정식 관문")
    log("=" * 78)
    n_ok = 0
    for label, v in rep["arms"].items():
        pa = v.get("phase_a_passed")
        pb = (v.get("phase_b") or {}).get("passed")
        n_ok += bool(pb)
        hb = (v.get("phase_b") or {}).get("fwd", {}).get("mean_bp")
        log(f"  {label:<24} PhaseA {'✅' if pa else '❌'}  "
            f"PhaseB {'✅' if pb else ('❌' if pa else '미실행')}"
            f"{('  HOLDOUT ' + format(hb, '+.2f') + 'bp') if hb is not None else ''}")
    log("")
    log(f"⇒ 승격 후보: **{n_ok}건**")
    log("⚠️2-arm 사전등록 검정이었다 -- 한쪽만 통과했다면 다중성을 감안해 읽을 것")
    rep["n_promoted"] = n_ok
    rep["holdout_exposed"] = True
    rep["holdout_note"] = "경제성 축 첫 노출이자 마지막 노출 -- 재실행은 근거로 쓸 수 없다"
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
