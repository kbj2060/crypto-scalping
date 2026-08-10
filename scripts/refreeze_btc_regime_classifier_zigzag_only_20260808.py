"""Lag-audit the re-selected zigzag-only classifier, then re-freeze it (2026-08-08).

The previous freeze was revoked because its accuracy came from a feature block that turned out to
be dead weight.  Before freezing the replacement, it gets the same scrutiny the user applied to
its predecessor: is the improvement real information, or just a faster mechanical lag?  The new
model reads only causal zigzag states at {0.1, 0.2, 0.35, 0.5, 0.8}%, and the two finest of those
flip very quickly, so "it is just a fast zigzag" is the live alternative hypothesis.

Tests re-run on the new states (same definitions as audit_btc_regime_classifier_lag_20260808.py):
detection lag after each oracle pivot, agreement by wave quintile, agreement on bars where czz05
is wrong, and the share of bars where it simply equals czz05.  Reference rows: czz05 (the free
mechanical detector at the target threshold), czz01 (the FASTEST input it is given -- if the model
were merely echoing its fastest input, czz01 would explain it), and the revoked panel config.
Writes the frozen config JSON and a last-7-day chart.
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
from audit_btc_regime_classifier_lag_20260808 import (  # noqa: E402
    agree, by_quintile, detection_lag, dir_of, lag_profile, wave_position,
)
from build_btc_regime_classifier_zoo_20260808 import REGIME_COLORS, C_BULL, C_BEAR, C_CHOP, INK  # noqa: E402
from chart_btc_jm_regime_verification_20260808 import causal_zigzag  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import contiguous_runs, zigzag_oracle  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    VAL_START, VAL_END, OOS_START, OOS_END,
)

ZZ_PATH = ROOT / "data/research/btc_regime_theta005_zigzagonly_20260808.parquet"
SEEDBAG_PATH = ROOT / "data/research/btc_regime_theta005_seedbag_20260808.parquet"
BASE_PATH = ROOT / "data/research/btc_regime_theta005_20260808.parquet"
OUT_DIR = ROOT / "tmp/regime_theta005_20260808"
FROZEN_PATH = ROOT / "data/research/btc_regime_theta005_frozen_config_20260808.json"
THETA = 0.005

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    sel = json.loads((OUT_DIR / "reselect.json").read_text())
    assert sel["adopt"], "re-selection did not clear the OOS bar"

    zz = pd.read_parquet(ZZ_PATH)
    ts = pd.to_datetime(zz["timestamp"])
    close = zz["close"].to_numpy(dtype=np.float64)
    o_dir, pivots = zigzag_oracle(close, threshold=THETA)
    new = zz["zigzagonly_final"].to_numpy().astype(np.int8)
    d_new = dir_of(new)
    d_czz05 = causal_zigzag(close, threshold=0.005).astype(np.int8)
    d_czz01 = causal_zigzag(close, threshold=0.001).astype(np.int8)
    old = pd.read_parquet(SEEDBAG_PATH)["seedbag_primary"].to_numpy().astype(np.int8)
    d_old = dir_of(old)

    v_idx = np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy())
    o_idx = np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy())
    pos = wave_position(o_dir, pivots, len(close))

    audit = {}
    for wname, idx in (("val_2025Q4", v_idx), ("oos_2026Q1", o_idx)):
        wrong = idx[(d_czz05[idx] != 0) & (o_dir[idx] != 0) & (d_czz05[idx] != o_dir[idx])]
        blk = {}
        for nm, d in (("zigzag_only_new", d_new), ("panel_config_revoked", d_old),
                      ("czz05", d_czz05), ("czz01_fastest_input", d_czz01)):
            lp = lag_profile(d, o_dir, idx)
            blk[nm] = {"agreement_pct": agree(d, o_dir, idx),
                       "peak_lag_bars": lp["peak_lag_bars"], "peak_agreement_pct": lp["peak_agreement_pct"],
                       "detection_lag_median": detection_lag(d, o_dir, pivots, idx[0], idx[-1])["median_bars"],
                       "by_wave_quintile": by_quintile(d, o_dir, pos, idx),
                       "where_czz05_wrong_pct": agree(d, o_dir, wrong),
                       "equals_czz05_pct": round(float(np.mean(d[idx] == d_czz05[idx])) * 100, 1)}
        audit[wname] = blk
        print(f"=== {wname}", flush=True)
        for nm, v in blk.items():
            print(f"  {nm:22} agree {v['agreement_pct']:6}  peak@{v['peak_lag_bars']:>4} {v['peak_agreement_pct']:6}"
                  f"  detlag {v['detection_lag_median']:5}  Q1 {v['by_wave_quintile']['Q1']:6}"
                  f"  czzWrong {v['where_czz05_wrong_pct']:6}  =czz05 {v['equals_czz05_pct']:6}", flush=True)
    (OUT_DIR / "lag_audit_zigzagonly.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    f = sel["final"]
    frozen = {
        "name": "btc_regime_theta005_zigzagonly_S2fine5_lam05",
        "status": "FROZEN",
        "frozen_at": "2026-08-08",
        "supersedes": "btc_regime_theta005_seedbag8_w65_lam05 (revoked: its 130 panel features were "
                      "worth -1.2pp OOS; see lag_audit.json)",
        "reference_definition": {"oracle": "retrospective zigzag wave direction", "theta": THETA,
                                 "note": "oracle is scoring-only and uses future bars; never an input"},
        "pipeline": {
            "1_features": {"kind": "causal zigzag direction states ONLY (no panel features)",
                           "thresholds": f["thresholds"], "mode": f["mode"], "n_features": len(f["thresholds"])},
            "2_nowcaster": {"model": "LGBMClassifier binary", "n_estimators": 700, "learning_rate": 0.05,
                            "num_leaves": 63, "min_child_samples": 200, "feature_fraction": 0.8,
                            "bagging_fraction": 0.8, "bagging_freq": 1, "reg_lambda": 1.0,
                            "seed_bag": sel["seeds"], "aggregation": "mean of probabilities",
                            "train": "<= 2025-08-31 minus a 576-bar label purge, rows with oracle dir != 0"},
            "3_blend": f"w={f['w']} (1.0 = the vote blend was dropped; it did not help this feature set)",
            "4_decode": f"jump-penalized CAUSAL online DP, lambda={f['lambda']}",
        },
        "selection": {"procedure": sel["procedure"], "n_val_cells_scored": sel["n_val_cells_scored"],
                      "stage1_winner": sel["stage1_winner"], "stage2_winner": sel["stage2_winner"],
                      "oos_reads": 1, "adopt_bar_oos": sel["adopt_bar_oos"]},
        "measured": {"val_2025Q4_agreement_pct": sel["measured"]["val_2025Q4"]["agree"]["0.005"],
                     "oos_2026Q1_agreement_pct": sel["oos_agreement_pct"],
                     "coverage_pct": sel["measured"]["val_2025Q4"]["coverage_pct"],
                     "median_run_bars_val": sel["measured"]["val_2025Q4"]["median_run_bars"]},
        "lag_audit": audit,
        "lineage": [{"config": "czz05 (no fitting)", "val": 61.3, "oos": 60.8},
                    {"config": "lgbm_jm_lam1 (panel)", "val": 66.0, "oos": 63.5},
                    {"config": "ens_w65_lam0.5 (panel+vote)", "val": 67.2, "oos": 65.3},
                    {"config": "seedbag8 (panel+vote) [REVOKED]", "val": 67.6, "oos": 65.6},
                    {"config": "zigzag-only S2_fine5 lam0.5", "val": sel["measured"]["val_2025Q4"]["agree"]["0.005"],
                     "oos": sel["oos_agreement_pct"]}],
        "caveats": [
            "median run 8 bars (~40 min): labelling/visualization detector, NOT a trading gate; "
            "gate promotion needs its own contract because of turnover",
            "theta=0.005 wave amplitude (median 0.97%) is below the 2x0.5% confirmation cost, so "
            "wave-following at this scale is not a standalone trading edge",
            "the model reads only zigzag geometry -- it carries no market-state information, so do "
            "not expect it to transfer to questions the zigzags do not describe",
        ],
        "artifacts": {"states": str(ZZ_PATH.relative_to(ROOT)),
                      "scripts": ["scripts/reselect_btc_regime_classifier_zigzag_only_20260808.py",
                                  "scripts/refreeze_btc_regime_classifier_zigzag_only_20260808.py",
                                  "scripts/audit_btc_regime_classifier_lag_20260808.py"]},
    }
    FROZEN_PATH.write_text(json.dumps(frozen, indent=2, ensure_ascii=False))
    print(f"wrote {FROZEN_PATH}", flush=True)

    idx = np.flatnonzero((ts >= ts.iloc[-1] - pd.Timedelta(days=args.days)).to_numpy())
    h_ts = ts.to_numpy()[idx]
    onamed = np.where(o_dir == 1, 2, np.where(o_dir == -1, 0, 1)).astype(np.int8)
    base = pd.read_parquet(BASE_PATH)["czz05"].to_numpy().astype(np.int8)
    strips = [(onamed, "오라클 0.5% (사후)  "), (new, "신규 지그재그-only  "),
              (old, "구 패널 구성 (취소)  "), (base, "czz05 (기계적)  ")]
    fig, axes = plt.subplots(1 + len(strips), 1, figsize=(15, 8.4), sharex=True,
                             gridspec_kw={"height_ratios": [10] + [0.72] * len(strips), "hspace": 0.08})
    ax = axes[0]
    for s, e, stt in contiguous_runs(new[idx]):
        seg = slice(s, min(e + 2, len(idx)))
        ax.plot(h_ts[seg], close[idx][seg], color=REGIME_COLORS[stt], linewidth=1.4)
    m = frozen["measured"]
    ax.set_title(f"θ=0.5% 재고정 — 지그재그 5개만 사용  VAL {m['val_2025Q4_agreement_pct']}% / "
                 f"OOS {m['oos_2026Q1_agreement_pct']}%  커버리지 {m['coverage_pct']}%  "
                 f"중앙런 {int(m['median_run_bars_val'])}bar   (취소된 구성 67.6/65.6, czz05 61.3/60.8) "
                 f"— 최근 {args.days}일", loc="left", fontsize=12, color=INK)
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
    out = OUT_DIR / "week_refrozen_zigzagonly.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
