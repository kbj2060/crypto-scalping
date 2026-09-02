#!/usr/bin/env python3
"""ETH 전 신호(7종) x 레짐게이트 x top-k 앙상블 -- 오염 제거 레짐으로 (2026-09-02).

기존 앙상블 연구(`research_eth_regime_gated_costgate_ensemble_20260902.py`)는 SIGNALS에 5종만
담았다. 그런데 홀드아웃 경제성을 실제로 통과한 ETH 신호는 6종이고, **그중 최고 2개
(demarker_extreme +11.53bp, kalman_deviation_meanrev +5.80bp)가 앙상블 연구에서 통째로 빠져
있었다.** 둘은 후보풀 트랙(2026-08-31)에서 나왔고 각각 96/96 그리드 전체 통과라 이 저장소에서
선택편향 증거가 가장 강한 신호들이다. 이 스크립트가 그 공백을 메운다.

7종 = 기존 5종(taker/str_z/liquidity_sweep/orthogonal_combo/smt_divergence)
      + demarker_extreme + kalman_deviation_meanrev

레짐 게이트는 clean-cutoff 재훈련본(TRAIN <= 2025-08-31)만 쓴다 -- 평가창이 학습창 밖이라
게이트가 진짜 out-of-sample 예측이다.

설정 출처(모두 동결, 이 스크립트에서 새로 고르지 않음):
  기존 5종 -- tmp/eth_ensemble_val_selected_oos_eval_20260902/config_stability.csv 의 val_only
  demarker/kalman -- backtest_eth_kalman_demarker_trailing_holdout_exposure_20260831.py 의 확정값
                     (둘 다 96/96 그리드 전체 통과라 셀 선택 자유도가 사실상 없다)

Fresh-Forward 계약: fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false, regime_gate_in_sample=false
HOLDOUT(>=2026-04-01)은 이 스크립트에서 건드리지 않는다.
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

ETH_CLEAN = ROOT / "tmp/eth_regime_s12k3_clean_20260902/predictions.parquet"
BTC_CLEAN = ROOT / "tmp/btc_regime_s24k3_clean_20260902/predictions.parquet"
VAL_SEL = ROOT / "tmp/eth_ensemble_val_selected_oos_eval_20260902/config_stability.csv"
OUT_DIR = ROOT / "tmp/eth_all7_gated_ensemble_clean_20260902"
FIRE_CACHE = OUT_DIR / "built_fires"

BUILT = {
    "demarker_extreme": {"horizon": 8, "gap": 12, "K": 0.70, "sl": 2.0, "arm": 1.5, "trail": 0.1},
    "kalman_deviation_meanrev": {"horizon": 12, "gap": 12, "K": 2.5, "sl": 4.0, "arm": 1.5, "trail": 0.1},
}
CONTRACT = {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
            "regime_gate_in_sample": False, "holdout_touched": False}


def log(m: str) -> None:
    print(f"[all7] {m}", flush=True)


def build_missing_fires() -> None:
    """demarker/kalman 발동집합을 후보풀 트랙과 동일한 절차로 재생성해 CSV로 남긴다."""
    FIRE_CACHE.mkdir(parents=True, exist_ok=True)
    if all((FIRE_CACHE / f"{n}.csv").exists() for n in BUILT):
        log("built fires 캐시 사용")
        return
    from research_eth_candidate_pool_raw_lift_check_20260831 import (
        kalman_level_and_velocity, rolling_zscore)
    from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker
    from research_eth_kalman_demarker_gridscreen_20260831 import build_fires, load_klines
    from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (
        FEATURE_COLUMNS, build_indicator_frame)

    kl = load_klines()
    ind = build_indicator_frame(kl)
    log(f"klines {len(kl):,} / indicator frame 완료")

    dem = compute_demarker(kl["high"], kl["low"])
    i1 = ind.copy(); i1["dem"] = dem.to_numpy()
    c = BUILT["demarker_extreme"]
    f1 = build_fires(kl, i1, dem >= 0.90, dem <= 0.10, dem.fillna(0.5).to_numpy(),
                     FEATURE_COLUMNS + ["dem"], c["horizon"], c["gap"], c["K"])
    f1.to_csv(FIRE_CACHE / "demarker_extreme.csv", index=False)
    log(f"demarker_extreme fires {len(f1):,}")

    lev, _ = kalman_level_and_velocity(kl["close"].to_numpy())
    kd = pd.Series((kl["close"].to_numpy() - lev) / lev, index=kl.index)
    kz = rolling_zscore(kd)
    i2 = ind.copy(); i2["kalman_dev_z"] = kz.to_numpy()
    c = BUILT["kalman_deviation_meanrev"]
    f2 = build_fires(kl, i2, kz >= 2.0, kz <= -2.0, kz.fillna(0.0).to_numpy(),
                     FEATURE_COLUMNS + ["kalman_dev_z"], c["horizon"], c["gap"], c["K"])
    f2.to_csv(FIRE_CACHE / "kalman_deviation_meanrev.csv", index=False)
    log(f"kalman_deviation_meanrev fires {len(f2):,}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_missing_fires()

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
    idx = pd.DatetimeIndex(ts)
    pos_of = {t: i for i, t in enumerate(idx)}

    er = pd.read_parquet(ETH_CLEAN); br = pd.read_parquet(BTC_CLEAN)
    eth_chop = set(er.loc[er["regime"] == 2, "timestamp"])
    btc_chop = set(br.loc[br["regime"] == 2, "timestamp"])

    tabs = {}
    for name, b in cfg7.items():
        f = pd.read_csv(b["src"], parse_dates=["timestamp"])
        f = f.loc[f["timestamp"] < HOLDOUT_START]
        f = f[f["timestamp"].isin(pos_of)].copy()
        f["pos"] = [pos_of[t] for t in f["timestamp"]]        # 공통 klines 기준으로 재매핑
        f = f.sort_values("pos").reset_index(drop=True)
        dec = f["pos"].to_numpy(np.int64)
        sc = np.where(f["side"].to_numpy() == "bottom", 1.0, -1.0)
        atr = f["atr_pct"].to_numpy(float)
        ok = np.isfinite(atr) & (atr > 0)
        dec, sc, atr = dec[ok], sc[ok], atr[ok]
        for sgn, lb in ((1.0, "real"), (-1.0, "flip")):
            t = per_fire_outcomes(ts, o, h, l, c, dec, sc * sgn, atr, b["horizon"],
                                  b["sl"], b["arm"], b["trail"])
            t["signal"] = name
            t["decision_pos"] = [pos_of[x] for x in t["decision_ts"]]
            t["eth_chop"] = t["decision_ts"].isin(eth_chop)
            t["btc_chop"] = t["decision_ts"].isin(btc_chop)
            tabs[(name, lb)] = t
        log(f"{name}: {len(tabs[(name,'real')])}건 (fires {len(f)}, atr유효 {int(ok.sum())})")

    # ---- 단독 성적 -> VAL bp 순으로 우선순위 결정 ----
    solo = []
    for name in cfg7:
        for wn, (lo, hi) in (("VAL", (VAL_START, OOS_START)), ("OOS", (OOS_START, HOLDOUT_START))):
            for lb in ("real", "flip"):
                t = tabs[(name, lb)]
                w = t[(t.decision_ts >= lo) & (t.decision_ts < hi)].copy()
                w["prio"] = 0
                s = summarize(sequential_portfolio(w, {name: 0}), name)
                s.update({"window": wn, "kind": lb}); solo.append(s)
    sdf = pd.DataFrame(solo)
    sdf.to_csv(OUT_DIR / "solo_7signals.csv", index=False)
    log("\n=== 7종 단독 (순차 1슬롯, real) ===")
    sp = sdf[sdf.kind == "real"].pivot_table(index="arm", columns="window",
                                             values=["n", "mean_bp", "pf", "total_bp"])
    fg = sdf.pivot_table(index="arm", columns=["window", "kind"], values="total_bp")
    sp[("flip_ok", "")] = np.where((fg[("VAL", "real")] > fg[("VAL", "flip")]) &
                                   (fg[("OOS", "real")] > fg[("OOS", "flip")]), "O", "X")
    print(sp.round(2).to_string())

    order = sdf[(sdf.kind == "real") & (sdf.window == "VAL")].sort_values(
        "mean_bp", ascending=False)["arm"].tolist()
    log(f"\nVAL bp 우선순위: {order}")
    prio = {n: i for i, n in enumerate(order)}

    # ---- top-k x gate ----
    rows = []
    for lb in ("real", "flip"):
        allc = pd.concat([tabs[(n, lb)] for n in order], ignore_index=True)
        allc["prio"] = allc["signal"].map(prio)
        for wn, (lo, hi) in (("VAL", (VAL_START, OOS_START)), ("OOS", (OOS_START, HOLDOUT_START))):
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
    df.to_csv(OUT_DIR / "all7_topk_gated.csv", index=False)

    log("\n=== top-k x gate (real) -- mean_bp / PF / n ===")
    for wn in ("VAL", "OOS"):
        t = df[(df.kind == "real") & (df.window == wn)].set_index("arm")
        print(f"\n--- {wn} ---")
        print(t[["n", "total_bp", "mean_bp", "pf", "max_dd"]].round(3).to_string())

    fp = df.pivot_table(index="arm", columns=["window", "kind"], values="total_bp")
    gap = pd.DataFrame({"VAL_gap": fp[("VAL", "real")] - fp[("VAL", "flip")],
                        "OOS_gap": fp[("OOS", "real")] - fp[("OOS", "flip")],
                        "VAL_real": fp[("VAL", "real")], "OOS_real": fp[("OOS", "real")]})
    gap["통과"] = np.where((gap.VAL_gap > 0) & (gap.OOS_gap > 0) &
                          (gap.VAL_real > 0) & (gap.OOS_real > 0), "O", "X")
    gap.to_csv(OUT_DIR / "all7_flip_control.csv")
    log(f"\n방향뒤집기 통과 arm: {int((gap['통과']=='O').sum())}/{len(gap)}")
    print(gap[gap["통과"] == "X"].round(1).to_string() if (gap["통과"] == "X").any() else "  (전부 통과)")

    (OUT_DIR / "contract.json").write_text(json.dumps(
        {**CONTRACT, "priority_order": order, "configs": {k: {kk: vv for kk, vv in v.items()
                                                              if kk != "src"} for k, v in cfg7.items()}},
        indent=2, ensure_ascii=False))
    log(f"\n산출: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
