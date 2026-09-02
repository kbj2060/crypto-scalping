#!/usr/bin/env python3
"""ETH 7종 게이트 앙상블의 홀드아웃 진단 (2026-09-02).

⚠️**이것은 진단이지 승격 근거가 아니다.** 이유를 정확히 적는다:

  - 이 7개 신호 각각은 2026-08-30~31에 **자기 경제성게이트를 위해 홀드아웃을 이미 1회 소비**했다
    (str_z +3.70bp, demarker +11.53bp, kalman +5.80bp, orthogonal_combo +3.78bp,
     smt +3.24bp, liquidity_sweep +1.97bp, taker는 부호 뒤집힘). 즉 이 구간 데이터는
    이미 사람 눈을 거쳤고, 단일노출 원칙상 재사용은 승격 근거가 못 된다.
  - 반면 **이 평가에서 새로운 것들**은 그 구간을 본 적이 없다: ①clean-cutoff 레짐 게이트
    (TRAIN <= 2025-08-31이라 홀드아웃이 학습창 밖), ②1슬롯 순차 포트폴리오 구성,
    ③top-k 조합, ④28 arm 중 선택.

따라서 이 숫자는 "이미 본 신호들을, 처음 보는 방식으로 묶었을 때" 얼마나 버티는지를 보여준다.
특히 VAL/OOS 대비 축소율이 관심사다 -- str_z 단독은 홀드아웃에서 VAL/OOS의 약 30%로 줄었다.
진짜 승격 근거는 전진 섀도우로 새로 벌어야 한다.
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
    HOLDOUT_START, KLINES_PATH, OOS_START, SIGNALS as SIG5, VAL_START)
from research_eth_evidence_signal_ensemble_pnl_20260902 import (  # noqa: E402
    per_fire_outcomes, sequential_portfolio, summarize)
from research_eth_all7_gated_ensemble_clean_regime_20260902 import (  # noqa: E402
    BUILT, ETH_CLEAN, BTC_CLEAN, FIRE_CACHE, VAL_SEL)

OUT_DIR = ROOT / "tmp/eth_all7_holdout_diagnostic_20260902"
DISCLOSURE = {
    "is_promotion_evidence": False, "reason": "6 signals each already spent a single holdout exposure",
    "fresh_components": ["clean-cutoff regime gate (TRAIN<=2025-08-31)",
                         "1-slot sequential portfolio", "top-k union", "28-arm selection"],
    "regime_gate_in_sample": False, "trade_ledgers_used_as_input": False,
    "fresh_forward_bar_by_bar": True,
}


def log(m: str) -> None:
    print(f"[holdout_diag] {m}", flush=True)


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
    hold_end = ts.max()

    er = pd.read_parquet(ETH_CLEAN); br = pd.read_parquet(BTC_CLEAN)
    eth_chop = set(er.loc[er["regime"] == 2, "timestamp"])
    btc_chop = set(br.loc[br["regime"] == 2, "timestamp"])

    tabs = {}
    for name, b in cfg7.items():
        f = pd.read_csv(b["src"], parse_dates=["timestamp"])
        f = f[f["timestamp"].isin(pos_of)].copy()          # 홀드아웃 포함 전 구간
        f["pos"] = [pos_of[t] for t in f["timestamp"]]
        f = f.sort_values("pos").reset_index(drop=True)
        dec = f["pos"].to_numpy(np.int64)
        sc = np.where(f["side"].to_numpy() == "bottom", 1.0, -1.0)
        atr = f["atr_pct"].to_numpy(float)
        ok = np.isfinite(atr) & (atr > 0)
        for sgn, lb in ((1.0, "real"), (-1.0, "flip")):
            t = per_fire_outcomes(ts, o, h, l, c, dec[ok], sc[ok] * sgn, atr[ok], b["horizon"],
                                  b["sl"], b["arm"], b["trail"])
            t["signal"] = name
            t["decision_pos"] = [pos_of[x] for x in t["decision_ts"]]
            t["eth_chop"] = t["decision_ts"].isin(eth_chop)
            t["btc_chop"] = t["decision_ts"].isin(btc_chop)
            tabs[(name, lb)] = t

    order = json.loads((ROOT / "tmp/eth_all7_gated_ensemble_clean_20260902/contract.json")
                       .read_text())["priority_order"]
    prio = {n: i for i, n in enumerate(order)}
    log(f"우선순위(VAL 기준, 동결): {order}")
    log(f"홀드아웃 {HOLDOUT_START.date()} ~ {hold_end.date()} ({(hold_end-HOLDOUT_START).days}일)")

    rows = []
    for lb in ("real", "flip"):
        allc = pd.concat([tabs[(n, lb)] for n in order], ignore_index=True)
        allc["prio"] = allc["signal"].map(prio)
        for wn, (lo, hi) in (("VAL", (VAL_START, OOS_START)), ("OOS", (OOS_START, HOLDOUT_START)),
                             ("HOLDOUT", (HOLDOUT_START, hold_end + pd.Timedelta(minutes=5)))):
            w = allc[(allc.decision_ts >= lo) & (allc.decision_ts < hi)]
            for k in range(1, len(order) + 1):
                base = w[w["signal"].isin(order[:k])]
                gates = {"plain": np.ones(len(base), bool),
                         "ethchop": base["eth_chop"].to_numpy(),
                         "btcchop": base["btc_chop"].to_numpy(),
                         "bothchop": (base["eth_chop"] & base["btc_chop"]).to_numpy()}
                for gn, g in gates.items():
                    s = summarize(sequential_portfolio(base[g], prio), f"top{k}_{gn}")
                    s.update({"window": wn, "kind": lb}); rows.append(s)
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "holdout_diagnostic.csv", index=False)

    r = df[df.kind == "real"]
    piv = r.pivot_table(index="arm", columns="window", values=["mean_bp", "pf", "n"])
    fl = df.pivot_table(index="arm", columns=["window", "kind"], values="total_bp")
    tab = pd.DataFrame({
        "VAL_mean": piv[("mean_bp", "VAL")], "OOS_mean": piv[("mean_bp", "OOS")],
        "HOLD_mean": piv[("mean_bp", "HOLDOUT")], "HOLD_pf": piv[("pf", "HOLDOUT")],
        "HOLD_n": piv[("n", "HOLDOUT")].astype(int),
        "축소율": (piv[("mean_bp", "HOLDOUT")] /
                 ((piv[("mean_bp", "VAL")] + piv[("mean_bp", "OOS")]) / 2)).round(3),
        "HOLD_flip통과": np.where((fl[("HOLDOUT", "real")] > fl[("HOLDOUT", "flip")]) &
                                (fl[("HOLDOUT", "real")] > 0), "O", "X"),
    })
    log("\n=== 홀드아웃 진단 (⚠️승격 근거 아님) ===")
    print(tab.round(2).to_string())
    tab.to_csv(OUT_DIR / "holdout_summary.csv")

    surv = tab[(tab.HOLD_mean > 0) & (tab["HOLD_flip통과"] == "O")]
    log(f"\n홀드아웃 양수 + 방향뒤집기 통과: {len(surv)}/{len(tab)} arm")
    log(f"그중 축소율 상위 5: \n{surv.sort_values('축소율', ascending=False)[['HOLD_mean','HOLD_pf','HOLD_n','축소율']].head(5).round(2).to_string()}")
    (OUT_DIR / "disclosure.json").write_text(json.dumps(DISCLOSURE, indent=2, ensure_ascii=False))
    log(f"\n산출: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
