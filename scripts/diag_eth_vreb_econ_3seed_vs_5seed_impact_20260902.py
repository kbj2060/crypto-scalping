#!/usr/bin/env python3
"""ETH V자반등 경제라벨 라이브 스코어러: **3시드 vs 5시드** 영향 측정 (사용자 결정 후속).

## 왜 재는가

GPU 절감을 위해 앙상블을 5시드 -> 3시드로 줄였다(`live_eth_v_rebound_econ_autotrade_signal_20260902.py`).
그런데 규격서의 **임계값 0.8158은 5시드 앙상블 확률로 보정된 값**이고, 시드 앙상블이
VAL 분위 선정 분산을 제거한 것(std 5.34 -> 0.021)이 이 후보의 통과 근거였다.
시드를 줄이면 확률 분포가 이동하므로 **같은 임계값이 같은 것을 뜻하지 않을 수 있다.**

## 무엇을 재나

동결 컨텍스트의 5개 시드 중 **정렬 후 앞 3개**(라이브가 실제로 쓰는 선택)로 앙상블한 확률을
5시드 앙상블과 비교한다. 평가 대상은 OOS 구간(2026-01~03) 후보 전체.

  · 확률 상관 / 평균절대차 / 분위별 이동
  · 임계값 0.8158에서 **호출 집합이 얼마나 달라지는가**(교집합/추가/누락)
  · 3시드 기준으로 상위 5%를 다시 잡으면 임계값이 얼마가 되는가(참고용, 변경 권고는 아님)

⚠️HOLDOUT 미터치. 진단 전용 -- 이 결과로 임계값을 바꾸면 그 자체가 재보정이므로 별도 결정이다.
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
_pf = importlib.util.module_from_spec(_P)
_P.loader.exec_module(_pf)

_L = importlib.util.spec_from_file_location(
    "sig", ROOT / "scripts/live_eth_v_rebound_econ_autotrade_signal_20260902.py")
_sig = importlib.util.module_from_spec(_L)
_L.loader.exec_module(_sig)

CTX = _sig.CTX_CSV
FEATURES = _sig.FEATURES
THR = _sig.PROBA_THRESHOLD
N3 = _sig.ENSEMBLE_SEEDS
OUT = ROOT / "data/research/eth_vreb_econ_3seed_impact_20260902/report.json"


def log(m): print(f"[3v5] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    ctx = pd.read_csv(CTX)
    seeds = sorted(ctx["seed"].unique())
    keep3 = seeds[:N3]
    log(f"동결 컨텍스트 시드 {seeds}  → 라이브가 쓰는 3시드 {keep3}")

    # 평가 대상: OOS 후보. `research_eth_v_rebound_ensemble_portfolio_sim_20260902.py::main()`의
    # 프레임 빌드 경로를 그대로 따른다(재구현하면 패리티가 조용히 깨진다).
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
    ev = ev.loc[(ev["timestamp"] >= pd.Timestamp("2026-01-01"))
                & (ev["timestamp"] < pd.Timestamp("2026-04-01"))].reset_index(drop=True)
    log(f"평가 대상 OOS 후보 {len(ev):,}행")

    probs = {}
    for sd, g in ctx.groupby("seed"):
        clf = TabPFNClassifier(device="cuda", random_state=int(sd), ignore_pretraining_limits=True)
        clf.fit(g[FEATURES], g["label"].to_numpy())
        probs[sd] = np.concatenate([clf.predict_proba(ev[FEATURES].iloc[k:k + 20000])[:, 1]
                                    for k in range(0, len(ev), 20000)])
        log(f"  seed {sd} 채점 완료")

    p5 = np.vstack([probs[s] for s in seeds]).mean(axis=0)
    p3 = np.vstack([probs[s] for s in keep3]).mean(axis=0)

    corr = float(np.corrcoef(p5, p3)[0, 1])
    mad = float(np.abs(p5 - p3).mean())
    log("")
    log(f"⭐확률 상관 {corr:.5f}   평균절대차 {mad:.5f}   최대차 {np.abs(p5-p3).max():.4f}")

    c5, c3 = p5 >= THR, p3 >= THR
    inter = int((c5 & c3).sum())
    log(f"⭐임계값 {THR} 에서:")
    log(f"   5시드 호출 {c5.sum():,}건   3시드 호출 {c3.sum():,}건")
    log(f"   교집합 {inter:,}  (5시드 기준 유지율 **{inter/max(c5.sum(),1)*100:.1f}%**)")
    log(f"   3시드에서 새로 생김 {int((~c5 & c3).sum()):,}   사라짐 {int((c5 & ~c3).sum()):,}")

    q95_5, q95_3 = float(np.quantile(p5, 0.95)), float(np.quantile(p3, 0.95))
    log(f"⭐상위 5% 분위: 5시드 {q95_5:.4f}  3시드 {q95_3:.4f}  (차이 {q95_3-q95_5:+.4f})")
    log("   ⇒ 3시드로 상위 5%를 다시 잡으려면 임계값이 이만큼 이동한다(참고용)")

    ss = np.array([float(np.std([probs[s][i] for s in seeds])) for i in range(0, len(ev), 97)])
    log(f"시드간 확률 std(표본): 평균 {ss.mean():.4f}  95분위 {np.quantile(ss,0.95):.4f}")

    rep = {"seeds_all": [int(s) for s in seeds], "seeds_live3": [int(s) for s in keep3],
           "n_eval_oos": int(len(ev)), "threshold": THR,
           "corr": corr, "mean_abs_diff": mad, "max_abs_diff": float(np.abs(p5-p3).max()),
           "calls_5seed": int(c5.sum()), "calls_3seed": int(c3.sum()),
           "intersection": inter, "retention_pct": round(inter/max(int(c5.sum()),1)*100, 1),
           "added": int((~c5 & c3).sum()), "dropped": int((c5 & ~c3).sum()),
           "q95_5seed": q95_5, "q95_3seed": q95_3,
           "seed_std_mean": float(ss.mean()), "runtime_sec": round(time.time()-t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
