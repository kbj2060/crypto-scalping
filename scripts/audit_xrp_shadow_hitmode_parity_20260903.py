#!/usr/bin/env python3
"""XRP 섀도우 러너의 `resolve()`가 **연구 구현과 같은 답**을 내는지 대조 (체크리스트 8번).

BTC에서 `HIT_SPEC` 모드 2건이 틀려 라이브 hit률이 2.6배 과대평가된 사고가 있었다.
정적 검토로는 못 잡았고, **연구 스크립트 원본 함수와 직접 대조**해서야 확정됐다.
XRP는 그 검증을 배포 **전에** 한다.

무작위 400 지점 x 양측 = 신호당 800건을, 각 신호의 확정 HIT_TYPE에 해당하는
`research_btc_short_term_return_z_gridscreen_hittype_20260901.py`의 원본 구현과 비교한다.
(그 파일이 4개 HIT_TYPE 구현의 정본이고, XRP 그리드스크린도 이 함수들로 셀을 골랐다.)

불일치 0이 아니면 배포하지 않는다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for q in (ROOT, ROOT / "scripts"):
    if str(q) not in sys.path:
        sys.path.insert(0, str(q))


def _m(n, rel):
    sp = importlib.util.spec_from_file_location(n, ROOT / rel)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


R = _m("xrpshadow", "scripts/live_xrp_evidence_signal_shadow_runner_20260903.py")
G = _m("hitfns", "scripts/research_btc_short_term_return_z_gridscreen_hittype_20260901.py")
R.log = lambda *a, **k: None
OUT = ROOT / "data/research/xrp_shadow_hitmode_parity_20260903.json"
N, SEED = 400, 20260903


def main() -> int:
    rng = np.random.default_rng(SEED)
    n = 1200
    cl = 3.0 + np.cumsum(rng.normal(0, 0.004, n))          # XRP 가격대
    hi = cl + np.abs(rng.normal(0.003, 0.0018, n))
    lo = cl - np.abs(rng.normal(0.003, 0.0018, n))
    tr = np.maximum(hi[1:] - lo[1:],
                    np.maximum(np.abs(hi[1:] - cl[:-1]), np.abs(lo[1:] - cl[:-1])))
    atr = np.concatenate([[np.nan], pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()])
    bars = pd.DataFrame({"timestamp": pd.date_range("2026-09-03", periods=n, freq="5min", tz="UTC"),
                         "open": cl, "high": hi, "low": lo, "close": cl})

    def runner(sig, i, side):
        st = {"pending": [{"signal": sig, "side": side, "bar_utc": str(bars.timestamp[i]),
                           "proba": 0.5, "entry": float(cl[i]), "atr": float(atr[i]),
                           "recorded_utc": "x"}], "ledger": [], "expired": []}
        R.resolve(st, bars)
        return st["ledger"][0]["hit"] if st["ledger"] else None

    def ref(mode, i, side, H, k):
        idx = np.array([i])
        if mode == "touch":
            return int(G.hit_touch_mfe(hi, lo, cl, atr, idx, H, k, side)[0])
        if mode == "touch_mae_capped":
            return int(G.hit_touch_mae_capped(hi, lo, cl, atr, idx, H, k, side)[0])
        if mode == "touch_giveback_sustained":
            return int(G.hit_touch_giveback_sustained(hi, lo, cl, atr, idx, H, k, side)[0])
        if mode == "close_at_h":
            return int(G.hit_close_at_h(cl, atr, idx, H, k, side)[0])
        raise ValueError(mode)

    idx = rng.choice(np.arange(60, n - 120), size=N, replace=False)
    rep, allok = {}, True
    print(f"[xrp-parity] 무작위 {N}지점 x 양측 = 신호당 {N*2}건", flush=True)
    for sig, spec in R.HIT_SPEC.items():
        bad = nn = 0
        for i in idx:
            for side in ("bottom", "top"):
                got = runner(sig, int(i), side)
                if got is None:
                    continue
                nn += 1
                if got != ref(spec["mode"], int(i), side, spec["horizon"], spec["k"]):
                    bad += 1
        ok = bad == 0
        allok &= ok
        print(f"  {'✅' if ok else '❌'} {sig:<26} mode={spec['mode']:<26} n={nn:>4} 불일치 {bad}", flush=True)
        rep[sig] = {"mode": spec["mode"], "horizon": spec["horizon"], "k": spec["k"],
                    "resolve_bars": R._resolve_bars(spec), "n": nn, "mismatch": bad, "ok": ok}
    print(f"[xrp-parity] ⇒ {'✅ 전부 일치 -- 배포 가능' if allok else '❌ 불일치 -- 배포 금지'}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"seed": SEED, "n_points": N, "all_ok": allok, "signals": rep},
                              ensure_ascii=False, indent=2))
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
