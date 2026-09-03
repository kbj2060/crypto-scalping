#!/usr/bin/env python3
"""XRP·BTC 지정가 진입의 **순환이동 플라시보 귀무** -- 통과 셀 수가 우연보다 나은가.

## 왜

2026-09-03 지정가 이식 v2에서 8종이 통과했다(즉시 taker 대조군은 전부 0셀).
**그러나 격자가 신호당 48셀이고 신호가 11종이라 528셀을 훑었다.** 다중검정이다.

ETH가 5.18절에서 이 결과를 인정받을 때 건 스트레스 3종 중 하나가 이것이다:

> ③**순환이동 플라시보** 통과개수 귀무 -- 양창 21 vs 5.6(**p=0.000**), 3창 9 vs 2.0(**p=0.025**)

## 설계

트리거 배열을 **원형으로 이동**(circular shift)시킨다. 이동은
  · 트리거 **개수**를 보존하고
  · 트리거의 **군집 구조**(연속 발동 패턴)도 보존하며
  · 가격과의 **정렬만 깨뜨린다**
⇒ "이 트리거가 가리키는 시점에 정보가 있는가"만 정확히 제거하는 플라시보다.

같은 격자를 B회 돌려 통과 셀 수의 귀무분포를 만들고, 실제 통과 수의 백분위를 낸다.

## 판정 (실행 전 고정)

  실제 통과 셀 수가 귀무분포의 **95백분위 이상**이어야 "격자를 훑어서 얻은 게 아니다".
  미달이면 그 신호는 다중검정 산물로 본다.

⚠️VAL+OOS만(판정 기준과 동일). HOLDOUT 미터치.
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

_R = importlib.util.spec_from_file_location(
    "restport", ROOT / "scripts/research_xrp_btc_resting_limit_entry_20260903.py")
_r = importlib.util.module_from_spec(_R)
_R.loader.exec_module(_r)

_E = _r._e                                   # ETH 상태기계 (run/stats)
from live_evidence_signal_dashboard_20260823 import compute_signals   # noqa: E402

SRC = ROOT / "data/research/xrp_btc_resting_limit_entry_v2_20260903.json"
OUT = ROOT / "data/research/xrp_btc_resting_limit_placebo_null_20260903.json"

B_NULL, SEED = 40, 20260903
MIN_PASS_TO_TEST = 1        # 통과 셀이 1개 이상인 신호만 검정


def log(m): print(f"[placebo] {m}", flush=True)


def count_passes(tb, tt, atr, o, h, l, c, W, sp):
    """v2와 **동일한** 통과 기준으로 depth>0 셀 수를 센다(VAL·OOS만)."""
    n = 0
    for depth in _r.DEPTHS:
        if depth == 0.0:
            continue
        for wait in _r.WAITS:
            for cname, cost in _r.COSTS.items():
                ok = True
                for wn in ("VAL", "OOS"):
                    lo, hi = W[wn]
                    f_ = _E.stats(_E.run(tb, tt, atr, o, h, l, c, lo, hi, depth=depth,
                                         wait=wait, cost=cost, flip=False, **sp))
                    x_ = _E.stats(_E.run(tb, tt, atr, o, h, l, c, lo, hi, depth=depth,
                                         wait=wait, cost=cost, flip=True, **sp))
                    if not (f_["mean_bp"] > 0 and f_["mean_bp"] > x_["mean_bp"]):
                        ok = False
                        break
                n += ok
    return n


def main() -> int:
    t0 = time.time()
    src = json.loads(SRC.read_text())
    rng = np.random.default_rng(SEED)
    rep = {"B_null": B_NULL, "seed": SEED, "holdout_touched": False,
           "placebo": "트리거 배열 원형이동(개수·군집구조 보존, 가격 정렬만 파괴)",
           "criterion": "실제 통과셀 수가 귀무분포 95백분위 이상", "assets": {}}

    for asset, cfg in _r.ASSETS.items():
        res_src = src["assets"].get(asset, {})
        targets = {k: v for k, v in res_src.items() if v.get("n_pass_limit", 0) >= MIN_PASS_TO_TEST}
        if not targets:
            continue
        log("")
        log("#" * 74)
        log(f"{asset}  -- 검정 대상 {list(targets)}")
        log("#" * 74)

        raw = pd.read_csv(cfg["klines"], parse_dates=["timestamp"])
        raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        partner = pd.read_csv(cfg["partner"], usecols=["timestamp", "high", "low"],
                              parse_dates=["timestamp"])
        funding = _r.load_funding(cfg["funding"])
        for d in (raw, partner):
            d["timestamp"] = d["timestamp"].astype("datetime64[ns]")
        frame = compute_signals(raw, btc_df=partner, funding_df=funding)
        frame["timestamp"] = frame["timestamp"].astype("datetime64[ns]")
        ts = frame["timestamp"]
        o, h, l, c = (frame[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        atr = frame["atr_pct"].to_numpy(float)
        idx = pd.DatetimeIndex(ts)
        W = {"VAL": (int(idx.searchsorted(_r.VAL_START)), int(idx.searchsorted(_r.OOS_START))),
             "OOS": (int(idx.searchsorted(_r.OOS_START)), int(idx.searchsorted(_r.HOLDOUT_START)))}
        nbars = len(frame)

        out = {}
        for name, v in targets.items():
            col = _r.COL_ALIAS.get(name, name)
            tb = frame[f"bottom_{col}"].fillna(False).to_numpy(bool)
            tt = frame[f"top_{col}"].fillna(False).to_numpy(bool)
            sp = v["cell"]
            obs = v["n_pass_limit"]
            log("")
            log(f"=== {name}  실제 통과 {obs}셀  (B={B_NULL} 순환이동) ===")
            null = []
            for b in range(B_NULL):
                sh = int(rng.integers(nbars // 20, nbars - nbars // 20))
                nb, nt = np.roll(tb, sh), np.roll(tt, sh)
                null.append(count_passes(nb, nt, atr, o, h, l, c, W, sp))
                if (b + 1) % 10 == 0:
                    a = np.array(null)
                    log(f"   ...{b+1}/{B_NULL}  귀무 평균 {a.mean():.1f} 최대 {a.max()}")
            a = np.array(null)
            pct = float((a < obs).mean() * 100)
            ok = pct >= 95.0
            log(f"   귀무: 평균 {a.mean():.2f}  중앙값 {np.median(a):.1f}  최대 {a.max()}  "
                f"95분위 {np.percentile(a, 95):.1f}")
            log(f"   ⇒ 실제 {obs}셀 백분위 **{pct:.1f}%**  {'✅통과' if ok else '❌다중검정 산물'}")
            out[name] = {"observed": obs, "null_mean": float(a.mean()),
                         "null_median": float(np.median(a)), "null_max": int(a.max()),
                         "null_p95": float(np.percentile(a, 95)),
                         "pctile": pct, "passed": bool(ok), "null": a.tolist()}
        rep["assets"][asset] = out

    log("")
    log("=" * 76)
    log("종합 -- 통과 셀 수가 순환이동 플라시보보다 나은가")
    log("=" * 76)
    log(f"{'자산':<5}{'신호':<26}{'실제':>6}{'귀무평균':>9}{'귀무95%':>9}{'백분위':>9}  판정")
    n_ok = 0
    for asset, out in rep["assets"].items():
        for name, v in out.items():
            n_ok += v["passed"]
            log(f"{asset:<5}{name:<26}{v['observed']:>6}{v['null_mean']:>9.2f}"
                f"{v['null_p95']:>9.1f}{v['pctile']:>8.1f}%  {'✅' if v['passed'] else '❌'}")
    log("")
    log(f"⇒ 플라시보 귀무까지 통과: **{n_ok}종**")
    rep["n_passed"] = n_ok
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
