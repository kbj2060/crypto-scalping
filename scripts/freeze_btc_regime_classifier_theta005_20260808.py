"""Freeze the theta=0.5% BTC regime classifier and render its acceptance chart (2026-08-08).

Final config after the selection -> refinement -> ensemble -> seed-bagging rounds.  Writes a
frozen-config JSON that downstream code should read instead of hard-coding parameters, and a
last-7-day chart of the frozen classifier against the oracle and the round-by-round lineage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from build_btc_regime_classifier_zoo_20260808 import REGIME_COLORS, C_BULL, C_BEAR, C_CHOP, INK  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import contiguous_runs  # noqa: E402

SEEDBAG = ROOT / "data/research/btc_regime_theta005_seedbag_20260808.parquet"
ENSEMBLE = ROOT / "data/research/btc_regime_theta005_ensemble_20260808.parquet"
THETA005 = ROOT / "data/research/btc_regime_theta005_20260808.parquet"
OUT_DIR = ROOT / "tmp/regime_theta005_20260808"
FROZEN_PATH = ROOT / "data/research/btc_regime_theta005_frozen_config_20260808.json"

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

    bag = pd.read_parquet(SEEDBAG)
    ens = pd.read_parquet(ENSEMBLE)
    base = pd.read_parquet(THETA005)
    seedbag_report = json.loads((OUT_DIR / "seedbag.json").read_text())

    frozen = {
        "name": "btc_regime_theta005_seedbag8_w65_lam05",
        "frozen_at": "2026-08-08",
        "reference_definition": {"oracle": "retrospective zigzag wave direction", "theta": 0.005,
                                 "note": "oracle is scoring-only and uses future bars; never an input"},
        "pipeline": {
            "1_features": "btc_features_2024_2026_regimeline.csv panel (RAW_LEVEL_COLS excluded) "
                          "+ causal zigzag states at theta {0.002,0.0035,0.005,0.008,0.012}",
            "2_nowcaster": {"model": "LGBMClassifier binary", "n_estimators": 700, "learning_rate": 0.05,
                            "num_leaves": 63, "min_child_samples": 200, "feature_fraction": 0.8,
                            "bagging_fraction": 0.8, "bagging_freq": 1, "reg_lambda": 1.0,
                            "seed_bag": seedbag_report["seeds"], "aggregation": "mean of probabilities",
                            "train": "<= 2025-08-31 minus a 576-bar label purge, rows with oracle dir != 0"},
            "3_vote_probability": "empirical P(bull | sum of the 5 causal zigzag directions), "
                                  "estimated on the same purged TRAIN rows, >=200-row bins else prior, "
                                  "clipped to [0.02, 0.98]",
            "4_blend": "logit(p) = 0.65*logit(p_nowcaster_bag) + 0.35*logit(p_vote)",
            "5_decode": "jump-penalized CAUSAL online DP, lambda=0.5: "
                        "V_t(s) = -log p_t(s) + min(V_{t-1}(s), min_s' V_{t-1}(s') + lambda)",
        },
        "measured": {"val_2025Q4_agreement_pct": seedbag_report["primary_test"]["seedbag"]["val"],
                     "oos_2026Q1_agreement_pct": seedbag_report["primary_test"]["seedbag"]["oos"],
                     "coverage_pct": seedbag_report["primary_test"]["seedbag"]["coverage_pct"],
                     "median_run_bars_val": seedbag_report["primary_test"]["seedbag"]["median_run_bars"],
                     "per_seed_spread": seedbag_report["seed_spread"]},
        "lineage": [{"round": "selection", "config": "czz05", "val": 61.3, "oos": 60.8},
                    {"round": "refinement", "config": "lgbm_jm_lam1", "val": 66.0, "oos": 63.5},
                    {"round": "ensemble", "config": "ens_w65_lam0.5", "val": 67.2, "oos": 65.3},
                    {"round": "seed-bagging", "config": "seedbag8_w65_lam0.5", "val": 67.6, "oos": 65.6}],
        "caveats": [
            "seed-bagging gain (+0.4 VAL / +0.3 OOS) is within the per-seed spread (std 0.25-0.27); "
            "its real value is removing seed-selection risk, not accuracy",
            "median run 8 bars (VAL) / 7 (OOS) ~ 35-40 min: this is a LABELLING and VISUALIZATION "
            "detector; using it directly as a trading gate would churn (~2,300 flips/quarter) and "
            "needs its own contract",
            "the theta=0.005 wave amplitude (median 0.97%) is below the 2x0.5% confirmation cost, "
            "so wave-following at this scale is not a standalone trading edge",
        ],
        "artifacts": {"states": str(SEEDBAG.relative_to(ROOT)),
                      "ensemble_round": str(ENSEMBLE.relative_to(ROOT)),
                      "selection_round": str(THETA005.relative_to(ROOT)),
                      "scripts": ["scripts/select_btc_regime_classifier_theta005_20260808.py",
                                  "scripts/refine_btc_regime_classifier_theta005_20260808.py",
                                  "scripts/ensemble_btc_regime_classifier_theta005_20260808.py",
                                  "scripts/seedbag_btc_regime_classifier_theta005_20260808.py"]},
    }
    FROZEN_PATH.write_text(json.dumps(frozen, indent=2, ensure_ascii=False))
    print(f"wrote {FROZEN_PATH}", flush=True)

    ts = pd.to_datetime(bag["timestamp"])
    close = bag["close"].to_numpy(dtype=np.float64)
    idx = np.flatnonzero((ts >= ts.iloc[-1] - pd.Timedelta(days=args.days)).to_numpy())
    h_ts = ts.to_numpy()[idx]
    final = bag["seedbag_primary"].to_numpy().astype(np.int8)
    onamed = np.where(bag["oracle005"].to_numpy() == 1, 2,
                      np.where(bag["oracle005"].to_numpy() == -1, 0, 1)).astype(np.int8)
    lineage = [(onamed, "오라클 0.5% (사후)  "),
               (final, "최종 seedbag8 w65 λ0.5  "),
               (ens["ens_w65_lam0.5"].to_numpy().astype(np.int8), "3라운드 ens_w65_lam0.5  "),
               (ens["lgbm_jm_lam1"].to_numpy().astype(np.int8), "2라운드 lgbm_jm_lam1  "),
               (base["czz05"].to_numpy().astype(np.int8), "1라운드 czz05  ")]

    fig, axes = plt.subplots(1 + len(lineage), 1, figsize=(15, 9), sharex=True,
                             gridspec_kw={"height_ratios": [10] + [0.72] * len(lineage), "hspace": 0.08})
    ax = axes[0]
    for s, e, stt in contiguous_runs(final[idx]):
        seg = slice(s, min(e + 2, len(idx)))
        ax.plot(h_ts[seg], close[idx][seg], color=REGIME_COLORS[stt], linewidth=1.4)
    m = frozen["measured"]
    ax.set_title(f"θ=0.5% 최종 고정 분류기 — VAL {m['val_2025Q4_agreement_pct']}% / "
                 f"OOS {m['oos_2026Q1_agreement_pct']}%  커버리지 {m['coverage_pct']}%  "
                 f"중앙런 {int(m['median_run_bars_val'])}bar   (출발점 czz05 61.3/60.8) — 최근 {args.days}일",
                 loc="left", fontsize=12.5, color=INK)
    ax.grid(axis="y", color="#000000", alpha=0.07, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in
                       (("bull", C_BULL), ("chop", C_CHOP), ("bear", C_BEAR))],
              loc="upper left", frameon=False, fontsize=9, ncol=3)
    for sax, (arr, lb) in zip(axes[1:], lineage):
        for s, e, stt in contiguous_runs(arr[idx]):
            sax.axvspan(h_ts[s], h_ts[min(e + 1, len(idx) - 1)], color=REGIME_COLORS[stt], linewidth=0)
        sax.set_yticks([])
        sax.set_ylabel(lb, rotation=0, ha="right", va="center", fontsize=9, color=INK)
        for side in ("top", "right", "left", "bottom"):
            sax.spines[side].set_visible(False)
    out = OUT_DIR / "week_frozen_theta005.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
