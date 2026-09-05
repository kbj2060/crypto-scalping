#!/usr/bin/env python3
"""발동 후 **진입 지연**별 페이드 vs 지속 — "지속 창이 닫히면 페이드가 되는가" (2026-09-05).

사용자 질문: *"지속구간이면 천장 발동이 롱이고 되돌림 구간이면 천장 발동이 숏이야?"*

대시보드의 `되돌림 대기`는 **지속 창(12봉)이 닫혔다**는 상태 표시일 뿐, 규칙에 페이드 팔은 없다
(러너 `continuation_state`: `phase = "continuation" if bars_since <= 12 else "fade_watch"`).
그 상태에서 화면이 보여주는 건 **F0 경제모델이 그 방향을 불렀는가**이고, F0 자체가 5.23에서
"직전 움직임의 지속을 사는 단기 모멘텀 모델"로 규정됐다. 그래도 "12봉 뒤 페이드"는 직접 측정된 적이
없으므로 여기서 잰다.

규격(5.23 상속, 자유도 없음): 8종 raw 첫발동(GAP12) 합집합 → 진입 `open[i+k]` (k = 지연 봉수),
ATR은 **결정 봉 i+k−1**의 값(인과), sim_exit(5.0/1.5/0.1) 200봉, −10bp, 동시 5 슬롯.
k ∈ {1, 3, 6, 12, 13, 18, 24, 36, 48} × 방향 {페이드, 지속} × 창 {TRAIN, VAL, OOS}.
k=1이 배포 규칙(다음 봉 시가)이고, k≥13이 화면의 "되돌림 대기" 구간이다.
HOLDOUT 미접촉.
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
sys.path.insert(0, str(ROOT / "scripts"))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


C1 = _load("c1_delay", "scripts/research_eth_composite_direction_trend_pullback_20260905.py")
OUT = ROOT / "data/research/eth_fire_entry_delay_fade_vs_cont_20260905"
DELAYS = (1, 3, 6, 12, 13, 18, 24, 36, 48)
WINDOWS = ("TRAIN", "VAL", "OOS")


def log(m): print(f"[delay] {m}", flush=True)


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    B = C1.build()
    pos, sd, split, ts, bidx = B["pos"], B["sd"], B["split"], B["ts"], B["bidx"]
    o, h, l, c = B["o"], B["h"], B["l"], B["c"]
    n = len(h)
    prev = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    atr_all = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    par = float(np.nanmax(np.abs(atr_all[bidx] - B["atr"])))
    log(f"ATR 파리티 |Δ|max {par:.3e} (프레임 atr와 동일해야 함)")
    assert par < 1e-8, "ATR 재계산 불일치 — 중단"
    fade_sign = np.where(sd == 1, 1.0, -1.0); cont_sign = -fade_sign

    rep = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "delays": list(DELAYS), "cell": C1.CELL,
           "cost_bp": C1.COST, "max_concurrent": C1.CAP, "holdout_touched": False, "atr_parity_max_abs": par, "rows": {}}
    for k in DELAYS:
        st = bidx + k                                     # 진입 봉 (시가)
        ok = (st + C1.FWD < n) & np.isfinite(atr_all[st - 1])
        ix = st[:, None] + np.arange(C1.FWD)
        ix = np.where(ok[:, None], ix, 0)
        H, L, C = h[ix], l[ix], c[ix]
        ent = o[np.where(ok, st, 0)]; av = atr_all[np.where(ok, st - 1, 14)]   # ATR = 결정 봉(진입 직전 완결봉)
        rec = {"n_usable": int(ok.sum())}
        for nm, sgn in (("fade", fade_sign), ("cont", cont_sign)):
            ret, ex = C1.sim_exit(ent, av, sgn, H, L, C, *C1.CELL)
            p = ret * 1e4 - C1.COST
            d = {}
            for w in WINDOWS:
                m = ok & (split == w)
                if m.sum() < 100:
                    continue
                r = C1.pf(C1.cand_of(ts[m], pos[m] + k, pos[m] + k + ex[m], p[m]))
                if r is None:
                    continue
                d[w] = {x: r["stats"][x] for x in ("n", "exp_bp", "win_rate", "day_ci95", "per_day", "daily_sharpe_ann")}
            rec[nm] = d
        rep["rows"][f"k{k}"] = rec
        log(f"  k={k:>2}봉 지연: " + " | ".join(
            f"{w} 페이드 {rec['fade'][w]['exp_bp']:>6}{str(rec['fade'][w]['day_ci95']):>16} · 지속 {rec['cont'][w]['exp_bp']:>6}{str(rec['cont'][w]['day_ci95']):>16}"
            for w in WINDOWS if w in rec["fade"]))
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"완료 {time.time()-t0:.0f}s → {OUT/'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
