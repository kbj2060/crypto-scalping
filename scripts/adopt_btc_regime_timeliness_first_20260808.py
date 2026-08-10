"""Single OOS read + adoption of the timeliness-first regime detector (2026-08-08).

Contract: docs/experiments/btc_regime_timeliness_first_20260808.json (written before this ran).
Candidate: the frozen zigzag-only detector plus a causal turn-suspect routing overlay at boost
0.5 -- an overlay, so the frozen stability-first model stays intact underneath.

Order of operations enforced here:
  1. rebuild the frozen nowcaster and REGRESSION-CHECK it against its recorded VAL 70.1
  2. rebuild the overlay and regression-check its recorded VAL (70.1 total / 49.8 Q1 / 5-bar run)
  3. ONE OOS read, then evaluate the five pre-registered gates
  4. adopt (write a second frozen-config JSON) only if all five pass
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from audit_btc_regime_classifier_lag_20260808 import (  # noqa: E402
    by_quintile, detection_lag, dir_of, lag_profile, wave_position,
)
from build_btc_regime_classifier_zoo_20260808 import REGIME_COLORS, C_BULL, C_BEAR, C_CHOP, INK  # noqa: E402
from chart_btc_jm_regime_verification_20260808 import causal_zigzag  # noqa: E402
from ensemble_btc_regime_classifier_theta005_20260808 import logit, sigmoid  # noqa: E402
from refine_btc_regime_classifier_theta005_20260808 import (  # noqa: E402
    PANEL_PATH, PURGE, SCORE_SCALES, jump_decode_proba, summarize, to_named,
)
from test_statistical_jump_model_regimes_20260808 import contiguous_runs, zigzag_oracle  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)

CONTRACT = ROOT / "docs/experiments/btc_regime_timeliness_first_20260808.json"
FROZEN_STABLE = ROOT / "data/research/btc_regime_theta005_frozen_config_20260808.json"
FROZEN_TIMELY = ROOT / "data/research/btc_regime_theta005_timeliness_frozen_config_20260808.json"
OUT_PARQUET = ROOT / "data/research/btc_regime_theta005_timeliness_20260808.parquet"
OUT_DIR = ROOT / "tmp/regime_timeliness_20260808"
THETA, FAST_THETA, BOOST = 0.005, 0.001, 0.5
VAL_EXPECTED = {"total": 70.1, "q1": 49.8, "run": 5.0}

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT.read_text())
    stable = json.loads(FROZEN_STABLE.read_text())
    thetas = stable["pipeline"]["1_features"]["thresholds"]
    seeds = stable["pipeline"]["2_nowcaster"]["seed_bag"]
    lam = float(stable["pipeline"]["4_decode"].split("lambda=")[1])

    panel = pd.read_csv(PANEL_PATH, low_memory=False, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts = panel["timestamp"]
    close = panel["close"].to_numpy(dtype=np.float64)
    oracles = {t: zigzag_oracle(close, threshold=t)[0] for t in SCORE_SCALES}
    o_dir, pivots = zigzag_oracle(close, threshold=THETA)
    pos = wave_position(o_dir, pivots, len(close))

    train_mask = (ts <= TRAIN_END).to_numpy()
    tr_idx = np.flatnonzero(train_mask)[:-PURGE]
    tr_idx = tr_idx[o_dir[tr_idx] != 0]
    y = (o_dir[tr_idx] == 1).astype(int)
    v_idx = np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy())
    o_idx = np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy())

    xm = np.column_stack([causal_zigzag(close, threshold=t) for t in thetas]).astype(np.float32)
    fast = causal_zigzag(close, threshold=FAST_THETA).astype(np.int8)
    slow = causal_zigzag(close, threshold=THETA).astype(np.int8)
    suspect = (fast != slow) & (fast != 0) & (slow != 0)

    ps = []
    for s in seeds:
        clf = lgb.LGBMClassifier(objective="binary", n_estimators=700, learning_rate=0.05,
                                 num_leaves=63, min_child_samples=200, feature_fraction=0.8,
                                 bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
                                 random_state=s, n_jobs=-1, verbosity=-1)
        clf.fit(xm[tr_idx], y)
        ps.append(clf.predict_proba(xm)[:, 1])
    p_frozen = np.mean(ps, axis=0)
    st_frozen = to_named(jump_decode_proba(p_frozen, lam))
    st_timely = to_named(jump_decode_proba(
        sigmoid(logit(p_frozen) + BOOST * np.where(suspect, fast.astype(float), 0.0)), lam))

    def measure(st, idx):
        d = dir_of(st)
        s = summarize(st, oracles, idx)
        lp = lag_profile(d, o_dir, idx)
        return {"total": s["agree"]["0.005"], "coverage_pct": s["coverage_pct"],
                "median_run_bars": s["median_run_bars"], "n_flips": s["n_flips"],
                **by_quintile(d, o_dir, pos, idx),
                "peak_lag_bars": lp["peak_lag_bars"], "peak_agreement_pct": lp["peak_agreement_pct"],
                "detection_lag_median": detection_lag(d, o_dir, pivots, idx[0], idx[-1])["median_bars"]}

    val_frozen, val_timely = measure(st_frozen, v_idx), measure(st_timely, v_idx)
    reg = {"frozen_val_total": val_frozen["total"], "expected": stable["measured"]["val_2025Q4_agreement_pct"],
           "overlay_val": {k: val_timely[k] for k in ("total", "Q1", "median_run_bars")},
           "overlay_expected": VAL_EXPECTED}
    reg_ok = (abs(val_frozen["total"] - stable["measured"]["val_2025Q4_agreement_pct"]) <= 0.2
              and abs(val_timely["total"] - VAL_EXPECTED["total"]) <= 0.2
              and abs(val_timely["Q1"] - VAL_EXPECTED["q1"]) <= 0.5)
    reg["regression_ok"] = bool(reg_ok)
    print(json.dumps({"regression": reg}, indent=2), flush=True)
    if not reg_ok:
        (OUT_DIR / "adoption.json").write_text(json.dumps({"regression": reg, "aborted": True}, indent=2))
        print("ABORT: could not reproduce the recorded VAL numbers; no OOS read taken")
        return 1

    # ---- the single OOS read
    oos_frozen, oos_timely = measure(st_frozen, o_idx), measure(st_timely, o_idx)
    g = contract["adoption_gates_single_oos_read"]
    gates = {
        "gate_1_no_accuracy_regression": {"bar": 67.0, "value": oos_timely["total"],
                                          "pass": bool(oos_timely["total"] >= 67.0)},
        "gate_2_timeliness_gain_persists": {"bar": 38.8, "value": oos_timely["Q1"],
                                            "pass": bool((oos_timely["Q1"] or 0) >= 38.8)},
        "gate_3_run_floor": {"bar": 4, "value": oos_timely["median_run_bars"],
                             "pass": bool(oos_timely["median_run_bars"] >= 4)},
        "gate_4_not_a_phase_artifact": {"bar": ">=0", "value": oos_timely["peak_lag_bars"],
                                        "pass": bool((oos_timely["peak_lag_bars"] or 0) >= 0)},
        "gate_5_detection_lag": {"bar": oos_frozen["detection_lag_median"],
                                 "value": oos_timely["detection_lag_median"],
                                 "pass": bool(oos_timely["detection_lag_median"] <= oos_frozen["detection_lag_median"])},
    }
    adopt = all(v["pass"] for v in gates.values())
    out = {"contract": str(CONTRACT.relative_to(ROOT)), "regression": reg,
           "val": {"frozen": val_frozen, "timeliness": val_timely},
           "oos_single_read": {"frozen": oos_frozen, "timeliness": oos_timely},
           "gates": gates, "adopt": adopt}
    (OUT_DIR / "adoption.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("=== OOS single read", flush=True)
    for nm, m in (("frozen(stability)", oos_frozen), ("timeliness overlay", oos_timely)):
        print(f"  {nm:20} total {m['total']:6}  Q1 {m['Q1']:6}  run {m['median_run_bars']:5}  "
              f"detlag {m['detection_lag_median']:4}  peak@{m['peak_lag_bars']}", flush=True)
    print(json.dumps({"gates": gates, "ADOPT": adopt}, indent=2), flush=True)

    pd.DataFrame({"timestamp": ts, "close": close, "oracle005": o_dir,
                  "stability_first": st_frozen, "timeliness_first": st_timely,
                  "turn_suspect": suspect.astype(np.int8)}).to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET}", flush=True)

    if adopt:
        frozen_timely = {
            "name": "btc_regime_theta005_timeliness_first_boost05",
            "status": "FROZEN",
            "role": "timeliness-first — early turn detection. The stability-first detector "
                    "(btc_regime_theta005_zigzagonly_S2fine5_lam05) is unchanged and remains the "
                    "default for labelling/charting where flicker is costly. Downstream code must "
                    "name which detector it reads.",
            "contract": str(CONTRACT.relative_to(ROOT)),
            "pipeline": {
                "1_base": stable["name"] + " (unchanged; this is an overlay, not a retrain)",
                "2_turn_suspect": f"causal zigzag(theta={FAST_THETA}) != causal zigzag(theta={THETA}), both nonzero",
                "3_overlay": f"logit(p) += {BOOST} * czz({FAST_THETA}) on turn-suspect bars",
                "4_decode": f"jump-penalized causal online DP, lambda={lam}",
            },
            "measured": {"val": val_timely, "oos_single_read": oos_timely,
                         "stability_first_for_comparison": {"val": val_frozen, "oos": oos_frozen}},
            "gates": gates,
            "caveats": stable["caveats"] + [
                "shorter runs than the stability-first detector by design; use it to answer 'has the "
                "wave turned yet', not to paint stable regime blocks",
            ],
            "artifacts": {"states": str(OUT_PARQUET.relative_to(ROOT)),
                          "scripts": ["scripts/adopt_btc_regime_timeliness_first_20260808.py"]},
        }
        FROZEN_TIMELY.write_text(json.dumps(frozen_timely, indent=2, ensure_ascii=False))
        print(f"wrote {FROZEN_TIMELY}", flush=True)

    idx = np.flatnonzero((ts >= ts.iloc[-1] - pd.Timedelta(days=args.days)).to_numpy())
    h_ts = ts.to_numpy()[idx]
    onamed = np.where(o_dir == 1, 2, np.where(o_dir == -1, 0, 1)).astype(np.int8)
    sus_named = np.where(suspect, 0, 1).astype(np.int8)
    strips = [(onamed, "오라클 0.5% (사후)  "), (st_timely, "적시성 우선 (신규)  "),
              (st_frozen, "안정성 우선 (기존)  "), (sus_named, "turn-suspect 구간  ")]
    fig, axes = plt.subplots(1 + len(strips), 1, figsize=(15, 8.4), sharex=True,
                             gridspec_kw={"height_ratios": [10] + [0.72] * len(strips), "hspace": 0.08})
    ax = axes[0]
    for s, e, stt in contiguous_runs(st_timely[idx]):
        seg = slice(s, min(e + 2, len(idx)))
        ax.plot(h_ts[seg], close[idx][seg], color=REGIME_COLORS[stt], linewidth=1.4)
    verdict = "채택" if adopt else "미채택"
    ax.set_title(f"적시성 우선 감지기 [{verdict}] — OOS 전체 {oos_timely['total']}% / "
                 f"파동초반 {oos_timely['Q1']}% / 중앙런 {int(oos_timely['median_run_bars'])}bar   "
                 f"(안정성 우선: {oos_frozen['total']}% / {oos_frozen['Q1']}% / "
                 f"{int(oos_frozen['median_run_bars'])}bar) — 최근 {args.days}일",
                 loc="left", fontsize=12, color=INK)
    ax.grid(axis="y", color="#000000", alpha=0.07, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in
                       (("bull", C_BULL), ("chop", C_CHOP), ("bear", C_BEAR))],
              loc="upper left", frameon=False, fontsize=9, ncol=3)
    for sax, (arr, lb) in zip(axes[1:], strips):
        for s, e, stt in contiguous_runs(arr[idx]):
            sax.axvspan(h_ts[s], h_ts[min(e + 1, len(idx) - 1)], color=REGIME_COLORS[stt], linewidth=0)
        sax.set_yticks([])
        sax.set_ylabel(lb, rotation=0, ha="right", va="center", fontsize=9, color=INK)
        for side in ("top", "right", "left", "bottom"):
            sax.spines[side].set_visible(False)
    outp = OUT_DIR / "week_timeliness_first.png"
    fig.savefig(outp, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
