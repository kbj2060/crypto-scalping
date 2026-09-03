#!/usr/bin/env python3
"""XRP 생존 구성의 **실제 셀 수치** + **순환이동 플라시보 귀무**.

## 왜

2026-09-03 레짐 게이팅/다중신호 합의에서 XRP만 6건 통과했다(BTC 0건):

    str_z + chop게이팅          진짜 3셀
    합의 K>=3 GAP3  H12         진짜 5셀
    합의 K>=3 GAP3  H24         진짜 2셀
    합의 K>=3 GAP6  H12         진짜 2셀
    합의 K>=3 GAP6  H24         진짜 3셀
    합의 K>=3 GAP12 H24         진짜 1셀

⚠️앞 스크립트의 "격자최선" 로그는 **믿을 수 없다** -- `max(cells, key=min(val,oos))`가
거래 0건 셀의 NaN 때문에 비교가 깨졌다(NaN 비교는 전부 False라 첫 원소가 max로 남는다).
`n_genuine`은 명시적 `>` 비교라 정확하지만 **수치는 다시 뽑아야 한다.**

⚠️그리고 96셀 격자에서 1~5셀 통과는 **그 자체로는 아무것도 뜻하지 않는다**. 같은 날 지정가
플라시보에서 무작위 트리거도 48셀 중 평균 1.7~5.8셀을 통과시켰다.
⇒ **순환이동 플라시보 귀무**를 걸어 통과 셀 수가 우연보다 나은지 본다.

## 설계

  · 생존 구성만 재실행해 **진짜 셀의 실제 VAL/OOS/뒤집기/n**을 낸다.
  · 같은 구성에서 트리거를 **원형 이동**(개수·군집구조 보존, 가격 정렬만 파괴)해 B회 돌리고
    통과 셀 수의 귀무분포를 만든다.
  · 판정: 실제 통과 셀 수가 귀무 **95백분위 이상**.

⚠️기준 불변(10bp, 96셀, 뒤집기, ARM>=1.0, 두께 100). HOLDOUT 미터치.
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

_F = importlib.util.spec_from_file_location(
    "final2", ROOT / "scripts/research_xrp_btc_regime_gate_and_consensus_20260903.py")
_f = importlib.util.module_from_spec(_F)
_F.loader.exec_module(_f)
_t = _f._t

from live_evidence_signal_dashboard_20260823 import compute_signals   # noqa: E402

OUT = ROOT / "data/research/xrp_survivors_detail_placebo_20260903.json"
B_NULL, SEED = 40, 20260903
ASSET = "XRP"

# (라벨, 종류, 파라미터)
CONFIGS = [
    ("str_z_chopgate", "gated", {"signal": "short_term_return_z"}),
    ("consensus_K3_GAP3_H12", "consensus", {"K": 3, "gap": 3, "H": 12}),
    ("consensus_K3_GAP3_H24", "consensus", {"K": 3, "gap": 3, "H": 24}),
    ("consensus_K3_GAP6_H12", "consensus", {"K": 3, "gap": 6, "H": 12}),
    ("consensus_K3_GAP6_H24", "consensus", {"K": 3, "gap": 6, "H": 24}),
    ("consensus_K3_GAP12_H24", "consensus", {"K": 3, "gap": 12, "H": 24}),
]


def log(m): print(f"[surv] {m}", flush=True)


def build_ctx():
    cfg = _t.ASSETS[ASSET]
    raw = pd.read_csv(cfg["klines"], parse_dates=["timestamp"])
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    partner = pd.read_csv(cfg["partner"], usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    funding = _t.load_funding(cfg["funding"])
    for d in (raw, partner):
        d["timestamp"] = d["timestamp"].astype("datetime64[ns]")
    frame = compute_signals(raw, btc_df=partner, funding_df=funding)
    frame["timestamp"] = frame["timestamp"].astype("datetime64[ns]")
    kl = frame[["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)
    atr = frame["atr_pct"].to_numpy(float)
    cm, _tag = _f.chop_mask(ASSET, frame["timestamp"])
    return frame, kl, atr, cm


def triggers(frame, kind, p):
    """구성별 (bottom, top) 인과적 첫발동 배열과 H를 만든다."""
    if kind == "gated":
        spec = _t.SIGNALS[p["signal"]]
        col = frame[spec["col"]].to_numpy(float); thr = spec["thr"][0]
        b, t_ = ((col <= thr), (col >= 1.0 - thr)) if spec["kind"] == "bounded" \
            else ((col <= -thr), (col >= thr))
        cb = _t.first_of_cluster(np.nan_to_num(b, nan=False), spec["gap"])
        ct = _t.first_of_cluster(np.nan_to_num(t_, nan=False), spec["gap"])
        return cb, ct, spec["H"][ASSET]
    rb, rt = [], []
    for sname, spec in _t.SIGNALS.items():
        col = frame[spec["col"]].to_numpy(float); thr = spec["thr"][0]
        b, t_ = ((col <= thr), (col >= 1.0 - thr)) if spec["kind"] == "bounded" \
            else ((col <= -thr), (col >= thr))
        rb.append(np.nan_to_num(b, nan=False)); rt.append(np.nan_to_num(t_, nan=False))
    nb, nt = np.sum(rb, axis=0), np.sum(rt, axis=0)
    cb = _t.first_of_cluster(nb >= p["K"], p["gap"])
    ct = _t.first_of_cluster(nt >= p["K"], p["gap"])
    return cb, ct, p["H"]


def evaluate(kl, atr, cb, ct, H, cm=None, gate=False):
    sel = (cb | ct)
    if gate:
        sel = sel & np.nan_to_num(cm, nan=False)
    idx = np.flatnonzero(sel)
    idx = idx[np.isfinite(atr[idx]) & (atr[idx] > 0) & (idx < len(kl) - 1)]
    if len(idx) < 50:
        return None, None, len(idx)
    cells, ns = _f.run_grid(kl, idx, cb[idx], atr[idx], H)
    return cells, ns, len(idx)


def main() -> int:
    t0 = time.time()
    frame, kl, atr, cm = build_ctx()
    rng = np.random.default_rng(SEED)
    nbars = len(frame)
    rep = {"asset": ASSET, "B_null": B_NULL, "seed": SEED, "holdout_touched": False,
           "criteria_unchanged": True, "configs": {}}

    for label, kind, p in CONFIGS:
        cb, ct, H = triggers(frame, kind, p)
        gate = (kind == "gated")
        cells, ns, nfire = evaluate(kl, atr, cb, ct, H, cm, gate)
        if cells is None:
            log(f"{label}: 발동 부족({nfire})"); continue
        gen = _f.genuine(cells)
        thin = ns["val"] < _f.MIN_CANDIDATES or ns["oos"] < _f.MIN_CANDIDATES
        log("")
        log("=" * 72)
        log(f"{label}   발동 {nfire:,}  후보 V{ns['val']}/O{ns['oos']}{'(얇음)' if thin else ''}  "
            f"진짜 {len(gen)}셀")
        log("=" * 72)
        for c in sorted(gen, key=lambda x: -min(x["val_fwd_bp"], x["oos_fwd_bp"])):
            log(f"   SL={c['sl']:<4} ARM={c['arm']:<4} Trail={c['trail']:<4}  "
                f"VAL {c['val_fwd_bp']:+7.2f}(뒤{c['val_flip_bp']:+7.2f}) n={c['val_n']:<5} | "
                f"OOS {c['oos_fwd_bp']:+7.2f}(뒤{c['oos_flip_bp']:+7.2f}) n={c['oos_n']}")

        # ---- 순환이동 플라시보 귀무 ----
        null = []
        for b in range(B_NULL):
            sh = int(rng.integers(nbars // 20, nbars - nbars // 20))
            nb_, nt_ = np.roll(cb, sh), np.roll(ct, sh)
            cs, _n, nf = evaluate(kl, atr, nb_, nt_, H, cm, gate)
            null.append(len(_f.genuine(cs)) if cs is not None else 0)
            if (b + 1) % 10 == 0:
                a = np.array(null)
                log(f"   ...플라시보 {b+1}/{B_NULL}  평균 {a.mean():.2f} 최대 {a.max()}")
        a = np.array(null)
        pct = float((a < len(gen)).mean() * 100)
        ok = pct >= 95.0 and not thin
        log(f"   귀무: 평균 {a.mean():.2f}  95분위 {np.percentile(a, 95):.1f}  최대 {a.max()}")
        log(f"   ⇒ 실제 {len(gen)}셀 백분위 **{pct:.1f}%**  "
            f"{'✅통과' if ok else ('❌얇음' if thin else '❌다중검정 산물')}")
        rep["configs"][label] = {
            "kind": kind, "params": p, "H": H, "n_fires": nfire, "n_candidates": ns,
            "thin": bool(thin), "n_genuine": len(gen),
            "genuine_cells": sorted(gen, key=lambda x: -min(x["val_fwd_bp"], x["oos_fwd_bp"])),
            "null_mean": float(a.mean()), "null_p95": float(np.percentile(a, 95)),
            "null_max": int(a.max()), "pctile": pct, "passed": bool(ok)}

    log(""); log("=" * 76)
    log("종합 -- 플라시보 귀무까지 통과하는 구성")
    log("=" * 76)
    log(f"{'구성':<26}{'진짜':>6}{'귀무평균':>10}{'귀무95%':>9}{'백분위':>9}  판정")
    n_ok = 0
    for label, v in rep["configs"].items():
        n_ok += v["passed"]
        log(f"{label:<26}{v['n_genuine']:>6}{v['null_mean']:>10.2f}{v['null_p95']:>9.1f}"
            f"{v['pctile']:>8.1f}%  {'✅' if v['passed'] else '❌'}")
    log("")
    log(f"⇒ 플라시보까지 통과: **{n_ok}건**")
    rep["n_passed"] = n_ok
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
