#!/usr/bin/env python3
"""ETH 7종 게이트 앙상블의 DSR / PBO -- 미이행 승격관문 이행 (2026-09-02).

이 저장소는 [[eth_live_stack_never_passed_dsr_pbo_20260819]] 이후 선택편향 통계를 정식으로 다시
돌린 적이 없다. `research_eth_all7_gated_ensemble_clean_regime_20260902.py`가 28개 arm
(top1~7 x {plain,ethchop,btcchop,bothchop})을 탐색했으므로, 그 승자를 그냥 쓰면 "28번 중 최고"의
선택편향이 들어간다. 그걸 정량화한다.

  - Deflated Sharpe Ratio: 28개 trial의 Sharpe 분산을 알고 있을 때, 승자의 Sharpe가 우연히
    최고가 됐을 확률을 깎아낸 값. DSR p > 0.95 면 선택편향을 감안해도 유의하다고 본다.
  - PBO (CSCV): 기간을 조합적으로 반씩 갈라 "IS 최고 arm이 OOS에서 중앙값 아래로 떨어지는" 빈도.
    PBO < 0.5가 최소선, 통상 < 0.2를 원한다.

수익 계열은 VAL+OOS 전 구간을 일 단위로 버킷팅해 만든다(무거래일은 0). 28개 arm 전부 같은
그리드를 쓰므로 CSCV 행렬이 정렬된다.
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

from core.selection_stats import deflated_sharpe_ratio, pbo_cscv, sharpe  # noqa: E402
from research_eth_regime_gated_costgate_ensemble_20260902 import (  # noqa: E402
    HOLDOUT_START, KLINES_PATH, OOS_START, SIGNALS as SIG5, VAL_START)
from research_eth_evidence_signal_ensemble_pnl_20260902 import (  # noqa: E402
    per_fire_outcomes, sequential_portfolio)
from research_eth_all7_gated_ensemble_clean_regime_20260902 import (  # noqa: E402
    BUILT, ETH_CLEAN, BTC_CLEAN, FIRE_CACHE, VAL_SEL)

OUT_DIR = ROOT / "tmp/eth_all7_dsr_pbo_20260902"


def log(m: str) -> None:
    print(f"[dsr_pbo] {m}", flush=True)


def main() -> int:
    st = pd.read_csv(VAL_SEL).set_index("signal")
    cfg7 = {}
    for n in SIG5:
        sl, arm, tr = st.loc[n, "val_only"].replace("SL", "").replace("ARM", "").replace("Tr", "").split("/")
        cfg7[n] = {"sl": float(sl), "arm": float(arm), "trail": float(tr),
                   "horizon": SIG5[n]["horizon"], "src": ROOT / SIG5[n]["fires"]}
    for n, c in BUILT.items():
        cfg7[n] = {"sl": c["sl"], "arm": c["arm"], "trail": c["trail"],
                   "horizon": c["horizon"], "src": FIRE_CACHE / f"{n}.csv"}

    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts = klines["timestamp"]
    o, h, l, c = (klines[k].to_numpy() for k in ("open", "high", "low", "close"))
    pos_of = {t: i for i, t in enumerate(pd.DatetimeIndex(ts))}

    er = pd.read_parquet(ETH_CLEAN); br = pd.read_parquet(BTC_CLEAN)
    eth_chop = set(er.loc[er["regime"] == 2, "timestamp"])
    btc_chop = set(br.loc[br["regime"] == 2, "timestamp"])

    tabs = {}
    for name, b in cfg7.items():
        f = pd.read_csv(b["src"], parse_dates=["timestamp"])
        f = f.loc[f["timestamp"] < HOLDOUT_START]
        f = f[f["timestamp"].isin(pos_of)].copy()
        f["pos"] = [pos_of[t] for t in f["timestamp"]]
        f = f.sort_values("pos").reset_index(drop=True)
        dec = f["pos"].to_numpy(np.int64)
        sc = np.where(f["side"].to_numpy() == "bottom", 1.0, -1.0)
        atr = f["atr_pct"].to_numpy(float)
        ok = np.isfinite(atr) & (atr > 0)
        t = per_fire_outcomes(ts, o, h, l, c, dec[ok], sc[ok], atr[ok], b["horizon"],
                              b["sl"], b["arm"], b["trail"])
        t["signal"] = name
        t["decision_pos"] = [pos_of[x] for x in t["decision_ts"]]
        t["eth_chop"] = t["decision_ts"].isin(eth_chop)
        t["btc_chop"] = t["decision_ts"].isin(btc_chop)
        tabs[name] = t

    order = json.loads((ROOT / "tmp/eth_all7_gated_ensemble_clean_20260902/contract.json")
                       .read_text())["priority_order"]
    prio = {n: i for i, n in enumerate(order)}
    log(f"우선순위: {order}")

    allc = pd.concat([tabs[n] for n in order], ignore_index=True)
    allc["prio"] = allc["signal"].map(prio)
    w = allc[(allc.decision_ts >= VAL_START) & (allc.decision_ts < HOLDOUT_START)]

    # ---- 28 arm의 일 단위 수익 행렬 ----
    days = pd.date_range(VAL_START, HOLDOUT_START - pd.Timedelta(days=1), freq="D")
    cols, names = [], []
    for k in range(1, len(order) + 1):
        base = w[w["signal"].isin(order[:k])]
        gates = {"plain": np.ones(len(base), bool),
                 "ethchop": base["eth_chop"].to_numpy(),
                 "btcchop": base["btc_chop"].to_numpy(),
                 "bothchop": (base["eth_chop"] & base["btc_chop"]).to_numpy()}
        for gn, g in gates.items():
            led = sequential_portfolio(base[g], prio)
            if led.empty:
                continue
            s = (led.assign(d=led["decision_ts"].dt.floor("D"))
                    .groupby("d")["trade_return"].sum().reindex(days, fill_value=0.0))
            cols.append(s.to_numpy()); names.append(f"top{k}_{gn}")
    mat = np.column_stack(cols)
    log(f"수익 행렬 {mat.shape[0]}일 x {mat.shape[1]} arm")

    sh = np.array([sharpe(mat[:, i]) for i in range(mat.shape[1])])
    best = int(np.argmax(sh))
    log(f"\nSharpe 최고 arm = {names[best]} (일 Sharpe {sh[best]:.4f}); "
        f"28 arm Sharpe 평균 {sh.mean():.4f} 표준편차 {sh.std(ddof=1):.4f}")

    res = {"arms": names, "daily_sharpe": {n: round(float(v), 5) for n, v in zip(names, sh)}}

    # ---- DSR: 승자 + 주요 후보들 ----
    log("\n=== Deflated Sharpe Ratio (28 trial 선택편향 반영) ===")
    dsr_rows = []
    focus = [names[best]] + [n for n in ("top1_plain", "top1_ethchop", "top1_bothchop",
                                         "top2_ethchop", "top3_ethchop", "top3_plain")
                             if n in names and n != names[best]]
    for nm in focus:
        i = names.index(nm)
        d = deflated_sharpe_ratio(mat[:, i], sh)
        dsr_rows.append({"arm": nm, "sharpe": round(float(sh[i]), 4),
                         **{k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                            for k, v in d.items()}})
    ddf = pd.DataFrame(dsr_rows)
    print(ddf.to_string(index=False))
    res["dsr"] = dsr_rows

    # ---- PBO ----
    log("\n=== PBO (CSCV) ===")
    for ns in (10, 8, 6):
        try:
            p = pbo_cscv(mat, n_splits=ns)
            print(f"  n_splits={ns}: " + ", ".join(
                f"{k}={round(float(v),4) if isinstance(v,(int,float)) else v}" for k, v in p.items()))
            res[f"pbo_splits{ns}"] = {k: (float(v) if isinstance(v, (int, float)) else v)
                                      for k, v in p.items()}
        except Exception as e:
            print(f"  n_splits={ns}: 실패 {type(e).__name__} {str(e)[:80]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mat, index=days, columns=names).to_csv(OUT_DIR / "daily_returns_matrix.csv")
    (OUT_DIR / "dsr_pbo.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    log(f"\n산출: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
