#!/usr/bin/env python3
"""3시드 앙상블의 **VAL 상위 5% 임계값** 재보정 (5시드 0.8158의 3시드 대응값).

## 왜 필요한가

`diag_eth_vreb_econ_3seed_vs_5seed_impact_20260902.py` 결과: 3시드는 확률 분포가 더 넓어
**같은 임계값 0.8158에서 호출이 41% 늘어난다**(2,619 -> 3,697건, 유지율 78.5%).
즉 시드만 줄이면 규격서가 검증한 것과 **다른 전략**을 섀도우가 돌리게 된다.

규격서의 선정 **규칙**은 "VAL 상위 5%"이고 0.8158은 5시드 앙상블에서 그 규칙을 구현한 값이다.
따라서 규칙을 보존하려면 **3시드의 VAL 상위 5%**를 새로 잡아야 한다.

⚠️OOS로 임계값을 잡으면 OOS 적합이다. **VAL에서만** 잡는다(원래와 같은 창).
⭐**정합성 확인**: 같은 방법으로 5시드 VAL 상위 5%를 계산해 0.8158이 재현되는지 먼저 본다.
   재현되면 이 방법이 원래 보정과 일치한다는 뜻이다.
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

_P = importlib.util.spec_from_file_location(
    "pf", ROOT / "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_pf = importlib.util.module_from_spec(_P); _P.loader.exec_module(_pf)
_L = importlib.util.spec_from_file_location(
    "sig", ROOT / "scripts/live_eth_v_rebound_econ_autotrade_signal_20260902.py")
_sig = importlib.util.module_from_spec(_L); _L.loader.exec_module(_sig)

FEATURES, N3 = _sig.FEATURES, _sig.ENSEMBLE_SEEDS
SPEC_THR = 0.8158
OUT = ROOT / "data/research/eth_vreb_econ_3seed_impact_20260902/val_threshold.json"


def log(m): print(f"[thr] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    ctx = pd.read_csv(_sig.CTX_CSV)
    seeds = sorted(int(s) for s in ctx["seed"].unique())
    keep3 = seeds[:N3]
    log(f"시드 전체 {seeds} / 라이브 3시드 {keep3}")

    _s1 = _pf._s1
    _s1.VAL_END = _pf.OOS_END
    sig, feat, eth = _s1.build_sig()
    sb = _s1.label_param(sig, True, ambig="drop", anchor="wick",
                         atr_mult=1.50, t_sustain=0.20, full_bars=12)
    st = _s1.label_param(sig, False, ambig="drop", anchor="wick",
                         atr_mult=1.50, t_sustain=0.20, full_bars=12)
    ev = _s1.long_frame_for(sig, feat, sb, st)
    ev["timestamp"] = pd.to_datetime(ev["timestamp"])
    if ev["timestamp"].dt.tz is not None:
        ev["timestamp"] = ev["timestamp"].dt.tz_localize(None)
    ev = ev.dropna(subset=FEATURES)
    val = ev.loc[(ev["timestamp"] >= pd.Timestamp("2025-09-01"))
                 & (ev["timestamp"] < pd.Timestamp("2026-01-01"))].reset_index(drop=True)
    log(f"VAL 후보 {len(val):,}행 (2025-09-01 ~ 2025-12-31)")

    probs = {}
    for sd, g in ctx.groupby("seed"):
        clf = TabPFNClassifier(device="cuda", random_state=int(sd), ignore_pretraining_limits=True)
        clf.fit(g[FEATURES], g["label"].to_numpy())
        probs[int(sd)] = np.concatenate(
            [clf.predict_proba(val[FEATURES].iloc[k:k+20000])[:, 1]
             for k in range(0, len(val), 20000)])
        log(f"  seed {sd} 채점 완료")

    p5 = np.vstack([probs[s] for s in seeds]).mean(axis=0)
    p3 = np.vstack([probs[s] for s in keep3]).mean(axis=0)
    q5, q3 = float(np.quantile(p5, 0.95)), float(np.quantile(p3, 0.95))
    log("")
    log(f"⭐정합성 확인: 5시드 VAL 상위5% = **{q5:.4f}**  (규격서 {SPEC_THR})  "
        f"차이 {q5-SPEC_THR:+.4f}  {'✅재현' if abs(q5-SPEC_THR) < 0.005 else '❌불일치'}")
    log(f"⭐3시드 VAL 상위5% = **{q3:.4f}**  (5시드 대비 {q3-q5:+.4f})")
    n5 = int((p5 >= SPEC_THR).sum()); n3s = int((p3 >= SPEC_THR).sum()); n3n = int((p3 >= q3).sum())
    log(f"   VAL 호출: 5시드@{SPEC_THR} {n5:,}  |  3시드@{SPEC_THR} {n3s:,} ({n3s/n5*100-100:+.0f}%)  "
        f"|  3시드@{q3:.4f} {n3n:,} ({n3n/n5*100-100:+.0f}%)")
    log(f"   ⇒ 임계값을 {q3:.4f}로 옮기면 호출 빈도가 5시드 규격과 다시 맞는다")
    rep = {"seeds_all": seeds, "seeds_live3": keep3, "n_val": int(len(val)),
           "spec_threshold_5seed": SPEC_THR, "recomputed_q95_5seed": q5,
           "sanity_reproduced": bool(abs(q5 - SPEC_THR) < 0.005),
           "recommended_threshold_3seed": q3,
           "val_calls": {"5seed_at_spec": n5, "3seed_at_spec": n3s, "3seed_at_new": n3n},
           "runtime_sec": round(time.time()-t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
