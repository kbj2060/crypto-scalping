#!/usr/bin/env python3
"""clean-cutoff 레짐 게이트의 시드 견고성 -- CLAUDE.md Seed-Diversity 계약 이행 (2026-09-02).

계약 요구: 학습 시드를 바꿔 승격을 주장하려면 N>=5개의 **진짜 무작위** 시드(고정 간격 증분 금지)
에서 OOS 부호가 일치해야 하고, 시드 리스트를 리포트에 남겨야 한다.

여기서 시드가 들어가는 유일한 지점은 레짐 분류기(HistGradientBoostingClassifier)의 random_state다.
라벨 임계값은 백분위 보정이라 결정적이고, 신호 발동집합/청산/비용/순차 포트폴리오도 결정적이다.
따라서 per-fire outcome은 한 번만 계산하고 **게이트 마스크만** 시드별로 다시 만든다.

시드는 random.sample로 뽑았다(2026-09-02): [76010, 130820, 194636, 331076, 703883]
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

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from research_eth_regime_gated_costgate_ensemble_20260902 import (  # noqa: E402
    HOLDOUT_START, KLINES_PATH, OOS_START, SIGNALS, VAL_START)
from research_eth_evidence_signal_ensemble_pnl_20260902 import (  # noqa: E402
    per_fire_outcomes, sequential_portfolio, summarize)
from train_eth_btc_regime_clean_cutoff_20260902 import CLEAN_TRAIN_END  # noqa: E402

SEEDS = [76010, 130820, 194636, 331076, 703883]
VAL_SEL = ROOT / "tmp/eth_ensemble_val_selected_oos_eval_20260902/config_stability.csv"
OUT_DIR = ROOT / "tmp/eth_clean_regime_gate_seed_robustness_20260902"
TOPK = ["short_term_return_z", "liquidity_sweep"]


def log(m: str) -> None:
    print(f"[seed_rob] {m}", flush=True)


def _prep(kind: str):
    """(df, X, train_mask, y) -- 시드와 무관한 부분을 한 번만 만든다."""
    if kind == "eth":
        from research_eth_regime_s12k3_label_train_20260902 import (
            GBM3_HP, GBM3_MODEL_PATH, load_frame, s12k3_label)
        from research_eth_regime_scalping_label_geometry_20260902 import TRAIN_START as TS
        src = joblib.load(GBM3_MODEL_PATH); df = load_frame(); lab = s12k3_label
    else:
        from research_btc_regime_s24k3_label_train_20260902 import (
            GBM3_HP, GBM3_MODEL_PATH, TRAIN_START as TS, load_btc_frame, s24k3_label)
        src = joblib.load(GBM3_MODEL_PATH)
        df = load_btc_frame(src["feature_cols"]); lab = s24k3_label
    cols, med = src["feature_cols"], src["feature_medians"]
    tr = ((df["timestamp"] >= TS) & (df["timestamp"] <= CLEAN_TRAIN_END)).to_numpy()
    y, _, _ = lab(df, tr)
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    for c in cols:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(med.get(c, 0.0))
    return df["timestamp"], x, tr, y, GBM3_HP


def main() -> int:
    st = pd.read_csv(VAL_SEL).set_index("signal")
    cfgs = {n: dict(zip(("sl", "arm", "trail"),
                        map(float, st.loc[n, "val_only"].replace("SL", "").replace("ARM", "")
                            .replace("Tr", "").split("/")))) for n in TOPK}

    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts = klines["timestamp"]
    o, h, l, c = (klines[k].to_numpy() for k in ("open", "high", "low", "close"))
    idx = pd.DatetimeIndex(ts)

    # --- 시드 무관: per-fire outcome 1회 계산 ---
    tabs = {}
    for name in TOPK:
        cfg, b = SIGNALS[name], cfgs[name]
        f = pd.read_csv(ROOT / cfg["fires"], parse_dates=["timestamp"])
        f = f.loc[f["timestamp"] < HOLDOUT_START].sort_values("pos").reset_index(drop=True)
        dec = f["pos"].to_numpy(np.int64)
        sc = np.where(f["side"].to_numpy() == "bottom", 1.0, -1.0)
        atr = f["atr_pct"].to_numpy(float)
        t = per_fire_outcomes(ts, o, h, l, c, dec, sc, atr, cfg["horizon"],
                              b["sl"], b["arm"], b["trail"])
        t["signal"] = name
        t["decision_pos"] = [int(idx.get_loc(x)) for x in t["decision_ts"]]
        tabs[name] = t
    log(f"per-fire outcomes 고정: {{{', '.join(f'{k}:{len(v)}' for k, v in tabs.items())}}}")

    eth_prep, btc_prep = _prep("eth"), _prep("btc")
    prio = {n: i for i, n in enumerate(TOPK)}
    rows = []

    for seed in SEEDS:
        masks = {}
        for tag, (pts, x, tr, y, hp) in (("eth", eth_prep), ("btc", btc_prep)):
            m = HistGradientBoostingClassifier(random_state=seed, **hp).fit(x[tr], y[tr])
            pred = m.predict(x)
            masks[tag] = set(pts[pred == 2])
            log(f"seed {seed} {tag}: 학습창밖 chop {float((pred[~tr] == 2).mean()):.3f} "
                f"라벨일치 {float((pred[~tr] == y[~tr]).mean()):.3f}")
        allc = pd.concat([tabs[n] for n in TOPK], ignore_index=True)
        allc["prio"] = allc["signal"].map(prio)
        allc["eth_chop"] = allc["decision_ts"].isin(masks["eth"])
        allc["btc_chop"] = allc["decision_ts"].isin(masks["btc"])
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
                    s.update({"window": wn, "seed": seed}); rows.append(s)

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "seed_arms.csv", index=False)

    log("\n=== Δmean_bp (게이트 - plain), 시드별 ===")
    out = []
    for k in (1, 2):
        for g in ("ethchop", "btcchop", "bothchop"):
            r = {"arm": f"top{k}_{g}"}
            for wn in ("VAL", "OOS"):
                ds = []
                for sd in SEEDS:
                    a = df[(df.seed == sd) & (df.window == wn) & (df.arm == f"top{k}_{g}")]["mean_bp"].iloc[0]
                    b = df[(df.seed == sd) & (df.window == wn) & (df.arm == f"top{k}_plain")]["mean_bp"].iloc[0]
                    ds.append(a - b)
                r[f"{wn}_deltas"] = ", ".join(f"{v:+.2f}" for v in ds)
                r[f"{wn}_pos"] = f"{sum(v > 0 for v in ds)}/5"
            r["판정"] = "PASS" if r["VAL_pos"] == "5/5" and r["OOS_pos"] == "5/5" else "FAIL"
            out.append(r)
    res = pd.DataFrame(out)
    print(res.to_string(index=False))
    res.to_csv(OUT_DIR / "seed_deltas.csv", index=False)
    (OUT_DIR / "seeds.json").write_text(json.dumps(
        {"seeds": SEEDS, "drawn_by": "random.sample(range(1000,999999),5)",
         "fixed_increment": False, "clean_train_end": str(CLEAN_TRAIN_END)}, indent=2))
    log(f"\n산출: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
