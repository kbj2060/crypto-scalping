#!/usr/bin/env python3
"""게이트 앙상블 재평가 -- 오염 제거된 레짐 예측으로 (2026-09-02).

`research_eth_final_regime_gated_ensemble_20260902.py`와 **완전히 동일한 절차**를 돌리되, 레짐
예측만 clean-cutoff 재훈련본(TRAIN <= 2025-08-31, `train_eth_btc_regime_clean_cutoff_20260902.py`)
으로 갈아끼운다. 원본은 두 레짐 모델의 TRAIN이 2024-01-01~2026-06-30이라 평가창 VAL(2025-09~12)/
OOS(2026-01~03)를 통째로 포함했고, 그래서 게이트가 예측이 아니라 적합이었다.

바뀌는 것은 오직 chop 마스크의 출처다. 신호 발동집합/설정(VAL-only 동결)/청산/비용/방향뒤집기/
순차 포트폴리오 로직은 전부 동일하므로, 원본과의 차이는 순수하게 **게이트 오염분**이다.

Fresh-Forward 계약 (CLAUDE.md) 선언:
  fresh_forward_bar_by_bar      = true   -- simulate_single_position이 봉 단위로 전진하며
                                            decision i -> entry i+1 open -> intrabar TP/SL 판정
  trade_ledgers_used_as_input   = false  -- 입력은 신호 발동 CSV(발동시각/side/atr)뿐,
                                            저장된 체결원장을 되읽지 않는다
  saved_parent_exit_timestamps_used = false
  future_rows_used_for_entry    = false  -- 게이트도 그 봉까지의 피처로 만든 예측만 사용
  regime_gate_in_sample         = false  -- 이 스크립트의 핵심 수정점
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
    HOLDOUT_START, KLINES_PATH, OOS_START, SIGNALS, VAL_START)
from research_eth_evidence_signal_ensemble_pnl_20260902 import (  # noqa: E402
    per_fire_outcomes, sequential_portfolio, summarize)

ETH_CLEAN = ROOT / "tmp/eth_regime_s12k3_clean_20260902/predictions.parquet"
BTC_CLEAN = ROOT / "tmp/btc_regime_s24k3_clean_20260902/predictions.parquet"
VAL_SEL = ROOT / "tmp/eth_ensemble_val_selected_oos_eval_20260902/config_stability.csv"
OUT_DIR = ROOT / "tmp/eth_gated_ensemble_clean_regime_20260902"
TOPK = ["short_term_return_z", "liquidity_sweep"]
CONTRACT = {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
            "regime_gate_in_sample": False}


def log(m: str) -> None:
    print(f"[clean_gate] {m}", flush=True)


def main() -> int:
    st = pd.read_csv(VAL_SEL).set_index("signal")
    cfgs = {}
    for n in TOPK:
        sl, arm, tr = st.loc[n, "val_only"].replace("SL", "").replace("ARM", "").replace("Tr", "").split("/")
        cfgs[n] = {"sl": float(sl), "arm": float(arm), "trail": float(tr)}
    log(f"동결 설정(VAL-only): {cfgs}")

    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts = klines["timestamp"]
    o, h, l, c = (klines[k].to_numpy() for k in ("open", "high", "low", "close"))
    idx = pd.DatetimeIndex(ts)

    er = pd.read_parquet(ETH_CLEAN); br = pd.read_parquet(BTC_CLEAN)
    eth_chop = set(er.loc[er["regime"] == 2, "timestamp"])
    btc_chop = set(br.loc[br["regime"] == 2, "timestamp"])
    log(f"clean chop 봉수 -- ETH {len(eth_chop):,} / BTC {len(btc_chop):,}")

    tabs = {}
    for name in TOPK:
        cfg, b = SIGNALS[name], cfgs[name]
        f = pd.read_csv(ROOT / cfg["fires"], parse_dates=["timestamp"])
        f = f.loc[f["timestamp"] < HOLDOUT_START].sort_values("pos").reset_index(drop=True)
        dec = f["pos"].to_numpy(np.int64)
        sc = np.where(f["side"].to_numpy() == "bottom", 1.0, -1.0)
        atr = f["atr_pct"].to_numpy(float)
        for sgn, lb in ((1.0, "real"), (-1.0, "flip")):
            t = per_fire_outcomes(ts, o, h, l, c, dec, sc * sgn, atr, cfg["horizon"],
                                  b["sl"], b["arm"], b["trail"])
            t["signal"] = name
            t["decision_pos"] = [int(idx.get_loc(x)) for x in t["decision_ts"]]
            t["eth_chop"] = t["decision_ts"].isin(eth_chop)
            t["btc_chop"] = t["decision_ts"].isin(btc_chop)
            tabs[(name, lb)] = t
        r = tabs[(name, "real")]
        log(f"{name}: {len(r)}건 | eth_chop {r.eth_chop.mean():.3f} btc_chop {r.btc_chop.mean():.3f}")

    prio = {n: i for i, n in enumerate(TOPK)}
    rows = []
    for lb in ("real", "flip"):
        allc = pd.concat([tabs[(n, lb)] for n in TOPK], ignore_index=True)
        allc["prio"] = allc["signal"].map(prio)
        for wn, (lo, hi) in (("VAL", (VAL_START, OOS_START)), ("OOS", (OOS_START, HOLDOUT_START))):
            w = allc[(allc.decision_ts >= lo) & (allc.decision_ts < hi)]
            for k in (1, 2):
                base = w[w["signal"].isin(TOPK[:k])]
                gates = {"plain": np.ones(len(base), bool),
                         "ethchop": base["eth_chop"].to_numpy(),
                         "btcchop": base["btc_chop"].to_numpy(),
                         "bothchop": (base["eth_chop"] & base["btc_chop"]).to_numpy()}
                for gname, g in gates.items():
                    s = summarize(sequential_portfolio(base[g], prio), f"top{k}_{gname}")
                    s.update({"window": wn, "kind": lb}); rows.append(s)

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "clean_gated_ensemble.csv", index=False)
    pd.set_option("display.width", 240)

    for wn in ("VAL", "OOS"):
        log(f"\n=== {wn} (real, clean regime) ===")
        t = df[(df.kind == "real") & (df.window == wn)].set_index("arm")
        print(t[["n", "total_bp", "mean_bp", "pf", "max_dd"]].round(3).to_string())

    # ---- n-독립 Δ표: 각 게이트 arm vs 같은 k의 plain ----
    log("\n=== n-독립 Δ (게이트 - plain), clean regime ===")
    piv = df[df.kind == "real"].pivot_table(index="arm", columns="window",
                                            values=["mean_bp", "pf", "n"])
    d_rows = []
    for k in (1, 2):
        b = f"top{k}_plain"
        for g in ("ethchop", "btcchop", "bothchop"):
            a = f"top{k}_{g}"
            d_rows.append({
                "arm": a,
                "dmean_VAL": piv.loc[a, ("mean_bp", "VAL")] - piv.loc[b, ("mean_bp", "VAL")],
                "dmean_OOS": piv.loc[a, ("mean_bp", "OOS")] - piv.loc[b, ("mean_bp", "OOS")],
                "dpf_VAL": piv.loc[a, ("pf", "VAL")] - piv.loc[b, ("pf", "VAL")],
                "dpf_OOS": piv.loc[a, ("pf", "OOS")] - piv.loc[b, ("pf", "OOS")],
                "n_VAL": int(piv.loc[a, ("n", "VAL")]), "n_OOS": int(piv.loc[a, ("n", "OOS")]),
            })
    dd = pd.DataFrame(d_rows)
    dd["양창"] = np.where((dd.dmean_VAL > 0) & (dd.dmean_OOS > 0), "O", "X")
    print(dd.round(3).to_string(index=False))
    dd.to_csv(OUT_DIR / "clean_gate_deltas.csv", index=False)

    # ---- 방향뒤집기 대조 ----
    log("\n=== 방향뒤집기 (real - flip, total_bp) ===")
    fp = df.pivot_table(index="arm", columns=["window", "kind"], values="total_bp")
    gap = pd.DataFrame({"VAL_gap": fp[("VAL", "real")] - fp[("VAL", "flip")],
                        "OOS_gap": fp[("OOS", "real")] - fp[("OOS", "flip")],
                        "VAL_real": fp[("VAL", "real")], "OOS_real": fp[("OOS", "real")]})
    gap["통과"] = np.where((gap.VAL_gap > 0) & (gap.OOS_gap > 0) &
                          (gap.VAL_real > 0) & (gap.OOS_real > 0), "O", "X")
    print(gap.round(1).to_string())
    gap.to_csv(OUT_DIR / "clean_flip_control.csv")

    (OUT_DIR / "contract.json").write_text(json.dumps(CONTRACT, indent=2))
    log(f"\n산출: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
