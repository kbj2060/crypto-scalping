#!/usr/bin/env python3
"""지정가 진입 통과조합의 스트레스 검정 -- 경계·우연·트리거 가치 (2026-09-02).

`research_eth_resting_limit_entry_20260902.py`에서 str_z depth=3.0이 3창 전부 방향뒤집기까지
통과했다. 완전 인과 규칙으로는 처음이라 값어치가 있으나, 경고등 셋을 먼저 꺼야 한다:

  S1 그리드 경계 -- depth 3.0이 격자의 끝이다(README 5.6). 3.0~6.0으로 넓혀 봉우리가 안에
     있는지 본다. 끝에서만 통과하면 그건 경계 아티팩트다.
  S2 무작위 진입 대조군 -- "3 ATR 아래 지정가를 걸고 기다린다"가 **트리거와 무관하게** 이미
     수익이면 신호는 아무것도 더하지 않는다. 같은 발동률의 무작위 봉으로 같은 기계를 돌린다.
  S3 순환이동 플라시보 -- 트리거 계열만 원형이동해 가격과의 정렬을 깬다. 클러스터 구조는
     보존되므로 "타이밍이 실제로 의미 있는가"만 검정된다. 통과 개수의 귀무분포를 만든다.

S2/S3 중 하나라도 통과 수준이 실제와 비슷하면 이 발견은 무효다.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research_eth_regime_gated_costgate_ensemble_20260902 import (  # noqa: E402
    HOLDOUT_START, KLINES_PATH, OOS_START, VAL_START)
from research_eth_resting_limit_entry_20260902 import SPEC, run, stats  # noqa: E402

OUT_DIR = ROOT / "tmp/eth_resting_limit_stress_20260902"
B_NULL = 40
RNG = np.random.default_rng(20260902)


def log(m): print(f"[stress] {m}", flush=True)


def eval_cfg(tb, tt, arrs, W, sp, depth, wait, cost):
    atr, o, h, l, c = arrs
    rec = {}
    ok3 = ok2 = True
    for wn, (a, b) in W.items():
        r = stats(run(tb, tt, atr, o, h, l, c, a, b, depth=depth, wait=wait, cost=cost, flip=False, **sp))
        f = stats(run(tb, tt, atr, o, h, l, c, a, b, depth=depth, wait=wait, cost=cost, flip=True, **sp))
        rec[f"{wn}_mean"] = r["mean_bp"]; rec[f"{wn}_pf"] = r["pf"]; rec[f"{wn}_n"] = r["n"]
        p = r["total_bp"] > max(f["total_bp"], 0)
        ok3 &= p
        if wn in ("VAL", "OOS"):
            ok2 &= p
    rec["flip3창"] = "O" if ok3 else "X"; rec["flip양창"] = "O" if ok2 else "X"
    return rec


def main() -> int:
    from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker
    from research_eth_kalman_demarker_gridscreen_20260831 import load_klines
    from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame

    src = load_klines(); ind = build_indicator_frame(src)
    kl = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    m = kl.merge(ind[["timestamp", "atr_pct", "ret3_z"]], on="timestamp", how="left")
    m = m.merge(pd.DataFrame({"timestamp": src["timestamp"],
                              "dem": compute_demarker(src["high"], src["low"]).to_numpy()}),
                on="timestamp", how="left")
    ts = m["timestamp"]
    arrs = (m["atr_pct"].to_numpy(float),) + tuple(m[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    r3 = m["ret3_z"].to_numpy(float); dm = m["dem"].to_numpy(float)
    TB = {"short_term_return_z": np.nan_to_num(r3, nan=0.0) <= -2.5,
          "demarker_extreme": np.nan_to_num(dm, nan=0.5) <= 0.10}
    TT = {"short_term_return_z": np.nan_to_num(r3, nan=99.0) >= 2.5,
          "demarker_extreme": np.nan_to_num(dm, nan=0.5) >= 0.90}
    idx = pd.DatetimeIndex(ts)
    W = {wn: (int(idx.searchsorted(lo)), int(idx.searchsorted(hi)))
         for wn, lo, hi in (("VAL", VAL_START, OOS_START), ("OOS", OOS_START, HOLDOUT_START),
                            ("HOLDOUT", HOLDOUT_START, ts.max()))}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- S1: 그리드 확장 ----------
    log("=== S1: depth 3.0~6.0 확장 (10bp 보수 가정) ===")
    rows = []
    for name in TB:
        for depth in (2.5, 3.0, 3.5, 4.0, 5.0, 6.0):
            for wait in (3, 6, 12):
                rec = {"signal": name, "depth": depth, "wait": wait}
                rec.update(eval_cfg(TB[name], TT[name], arrs, W, SPEC[name], depth, wait, 0.0010))
                rows.append(rec)
    s1 = pd.DataFrame(rows)
    s1.to_csv(OUT_DIR / "s1_grid_extension.csv", index=False)
    pd.set_option("display.width", 240)
    print(s1.to_string(index=False))
    n_pass2 = int((s1["flip양창"] == "O").sum()); n_pass3 = int((s1["flip3창"] == "O").sum())
    log(f"실제 통과: 양창 {n_pass2}/{len(s1)}, 3창 {n_pass3}/{len(s1)}")

    # ---------- S2: 무작위 진입 대조군 ----------
    log(f"\n=== S2: 무작위 봉 진입 대조군 (같은 발동률, B={B_NULL//2}) ===")
    s2 = []
    for name in ("short_term_return_z",):
        rate_b = float(TB[name].mean()); rate_t = float(TT[name].mean())
        for depth, wait in ((3.0, 3), (3.0, 6)):
            got = []
            for _ in range(B_NULL // 2):
                rb = RNG.random(len(r3)) < rate_b
                rt = RNG.random(len(r3)) < rate_t
                rec = eval_cfg(rb, rt, arrs, W, SPEC[name], depth, wait, 0.0010)
                got.append((rec["VAL_mean"], rec["OOS_mean"], rec["HOLDOUT_mean"],
                            rec["flip3창"] == "O"))
            g = pd.DataFrame(got, columns=["VAL", "OOS", "HOLD", "pass3"])
            real = s1[(s1.signal == name) & (s1.depth == depth) & (s1.wait == wait)].iloc[0]
            log(f"  depth{depth}/wait{wait}: 무작위 평균 VAL {g.VAL.mean():+.2f} OOS {g.OOS.mean():+.2f} "
                f"HOLD {g.HOLD.mean():+.2f} | 3창통과율 {g.pass3.mean():.0%}")
            log(f"    실제      VAL {real.VAL_mean:+.2f} OOS {real.OOS_mean:+.2f} HOLD {real.HOLDOUT_mean:+.2f}"
                f" | 백분위 VAL {(g.VAL < real.VAL_mean).mean():.0%} OOS {(g.OOS < real.OOS_mean).mean():.0%} "
                f"HOLD {(g.HOLD < real.HOLDOUT_mean).mean():.0%}")
            s2.append({"signal": name, "depth": depth, "wait": wait,
                       "null_VAL": round(float(g.VAL.mean()), 2), "null_OOS": round(float(g.OOS.mean()), 2),
                       "null_HOLD": round(float(g.HOLD.mean()), 2),
                       "null_pass3_rate": round(float(g.pass3.mean()), 3),
                       "real_VAL": real.VAL_mean, "real_OOS": real.OOS_mean, "real_HOLD": real.HOLDOUT_mean,
                       "pct_VAL": round(float((g.VAL < real.VAL_mean).mean()), 3),
                       "pct_OOS": round(float((g.OOS < real.OOS_mean).mean()), 3),
                       "pct_HOLD": round(float((g.HOLD < real.HOLDOUT_mean).mean()), 3)})
    pd.DataFrame(s2).to_csv(OUT_DIR / "s2_random_entry_null.csv", index=False)

    # ---------- S3: 순환이동 플라시보 (통과 개수 귀무분포) ----------
    log(f"\n=== S3: 트리거 순환이동 플라시보, 통과 개수 귀무분포 (B={B_NULL}) ===")
    counts2, counts3 = [], []
    n = len(r3)
    for _ in range(B_NULL):
        sh = int(RNG.integers(2000, n - 2000))
        c2 = c3 = 0
        for name in TB:
            tb = np.roll(TB[name], sh); tt = np.roll(TT[name], sh)
            for depth in (2.5, 3.0, 3.5, 4.0, 5.0, 6.0):
                for wait in (3, 6, 12):
                    r = eval_cfg(tb, tt, arrs, W, SPEC[name], depth, wait, 0.0010)
                    c2 += r["flip양창"] == "O"; c3 += r["flip3창"] == "O"
        counts2.append(c2); counts3.append(c3)
    c2a, c3a = np.array(counts2), np.array(counts3)
    log(f"  귀무 양창 통과수 평균 {c2a.mean():.1f} (95% 상한 {np.quantile(c2a,0.95):.0f}) vs 실제 {n_pass2}"
        f"  -> p={float((c2a>=n_pass2).mean()):.3f}")
    log(f"  귀무 3창 통과수 평균 {c3a.mean():.1f} (95% 상한 {np.quantile(c3a,0.95):.0f}) vs 실제 {n_pass3}"
        f"  -> p={float((c3a>=n_pass3).mean()):.3f}")
    json.dump({"real_pass2": n_pass2, "real_pass3": n_pass3,
               "null_pass2": counts2, "null_pass3": counts3,
               "p_pass2": float((c2a >= n_pass2).mean()), "p_pass3": float((c3a >= n_pass3).mean())},
              open(OUT_DIR / "s3_circular_shift_null.json", "w"), indent=2)
    log(f"\n산출: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
