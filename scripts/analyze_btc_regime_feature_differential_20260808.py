"""Per-regime feature analysis, redone around the question that was never actually asked (2026-08-08).

Every earlier pass (Stage R on the D2, JM/czz and timeliness gates) ranked features by |AUC-0.5|
WITHIN each regime and checked whether that sign survived train->VAL.  That answers "which features
carry signal inside a regime", which is not the same as "which features behave DIFFERENTLY across
regimes" -- a feature that is top-20 in both bull and bear with the same sign is not
regime-specific at all, yet it dominates those lists.  This pass measures the differential
directly and asks whether it persists.

Four analyses, over four gates chosen to span the run-length scale:
  d2_rule       trailing 288-bar return +-4% (the originally closed line's gate)
  jm_lam32      Statistical Jump Model k3 (median run ~130 bars; the best sign stability seen)
  czz4          causal 4% directional change (median run ~630 bars)
  stability05   the frozen theta=0.5% detector (median run ~8 bars -- included for contrast; it is
                far faster than the 288-bar label horizon and is expected to show no differential)

  1  IDENTITY     what a regime IS in feature space: standardized mean per regime on TRAIN, ranked
                  by |bull mean - bear mean|.  No labels involved.
  2  WITHIN       per-regime AUC against the triple-barrier action label (the old Stage R view),
                  kept so the two views can be compared side by side.
  3  DIFFERENTIAL delta = AUC_bull - AUC_bear per feature, ranked by |delta| on TRAIN, then the
                  sign of delta re-measured on VAL and on OOS.  A real regime-specific effect must
                  keep its sign; noise will not.  Random-baseline: the same persistence statistic
                  computed for 20 random feature subsets, so "how often would this happen anyway"
                  is on the page.
  4  CARRIERS     whether the positioning / funding / CVD family (this project's repeatedly
                  reconfirmed signal carriers) also carries the DIFFERENTIAL, or whether the
                  differential lives somewhere else entirely.

Scope: the regime-conditioned ENTRY axis is closed (three independent gates, two gate forms).
This is descriptive analysis of a closed axis -- OOS is read here as a persistence measurement,
not as a selection criterion, and nothing here is adopted or promoted.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_eval_btc_regime_conditioned_entry_20260808 import auc_binary, load_all  # noqa: E402

OUT_DIR = ROOT / "tmp/regime_feature_analysis_20260808"
ZOO_PATH = ROOT / "data/research/btc_regime_classifier_zoo_20260808.parquet"
TIMELY_PATH = ROOT / "data/research/btc_regime_theta005_timeliness_20260808.parquet"
TOP_N = 25
N_RANDOM = 20
SEED = 903174
CARRIER_PAT = ("toptrader", "long_short", "whale", "funding", "cvd", "oi_", "open_interest",
               "crowding", "positioning", "taker")
INK, C_BULL, C_BEAR, C_NEU, C_OK = "#1F2430", "#2563EB", "#D9542B", "#9AA0A6", "#0E7C66"

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def regime_auc(x, action, idx, regime, r):
    sub = idx[regime[idx] == r]
    a = action[sub]
    nz = a != 0
    out = np.full(x.shape[1], np.nan)
    if nz.sum() < 200:
        return out
    yv = (a[nz] == 1).astype(int)
    for f in range(x.shape[1]):
        out[f] = auc_binary(x[sub, f][nz].astype(np.float64), yv)
    return out


def is_carrier(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in CARRIER_PAT)


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel, ts, x, feat_cols, action, tp_moves, sl_moves, d2, train_mask, val_mask, oos_mask = load_all()
    tr_idx, v_idx, o_idx = (np.flatnonzero(m) for m in (train_mask, val_mask, oos_mask))

    zoo = pd.read_parquet(ZOO_PATH)
    timely = pd.read_parquet(TIMELY_PATH)
    assert len(zoo) == len(panel) and len(timely) == len(panel), "state frames misaligned with panel"
    gates = {"d2_rule": d2.astype(np.int8),
             "jm_lam32": zoo["jm"].to_numpy().astype(np.int8),
             "czz4": zoo["czz4"].to_numpy().astype(np.int8),
             "stability05": timely["stability_first"].to_numpy().astype(np.int8)}

    rng = np.random.default_rng(SEED)
    report: dict = {"n_features": len(feat_cols), "top_n": TOP_N,
                    "scope": "descriptive analysis of a CLOSED axis; OOS is a persistence "
                             "measurement, not a selection criterion"}

    for gname, regime in gates.items():
        occ = {n: round(float((regime[tr_idx] == r).mean()), 3)
               for r, n in ((0, "bear"), (1, "chop"), (2, "bull"))}
        runs = np.diff(np.flatnonzero(np.diff(regime) != 0))
        blk = {"occupancy_train": occ,
               "median_run_bars": float(np.median(runs)) if len(runs) else None}

        # 1 IDENTITY -- no labels, just what the regime looks like in feature space
        mu = x[tr_idx].mean(axis=0)
        sd = np.where(x[tr_idx].std(axis=0) > 0, x[tr_idx].std(axis=0), 1.0)
        z = (x - mu) / sd
        m_bull = np.nanmean(z[tr_idx[regime[tr_idx] == 2]], axis=0)
        m_bear = np.nanmean(z[tr_idx[regime[tr_idx] == 0]], axis=0)
        sep = np.nan_to_num(m_bull - m_bear)
        ident = np.argsort(-np.abs(sep))[:TOP_N]
        blk["1_identity_top"] = [{"feature": feat_cols[i], "bull_minus_bear_sd": round(float(sep[i]), 3)}
                                 for i in ident[:12]]

        # 2 WITHIN -- the old Stage R view, kept for comparison
        auc_tr = {r: regime_auc(x, action, tr_idx, regime, r) for r in (0, 2)}
        auc_v = {r: regime_auc(x, action, v_idx, regime, r) for r in (0, 2)}
        auc_o = {r: regime_auc(x, action, o_idx, regime, r) for r in (0, 2)}
        within = {}
        for r, nm in ((0, "bear"), (2, "bull")):
            dev = np.abs(np.nan_to_num(auc_tr[r], nan=0.5) - 0.5)
            top = np.argsort(-dev)[:TOP_N]
            s_tr = np.sign(auc_tr[r][top] - 0.5)
            keep_v = float(np.mean(s_tr == np.sign(np.nan_to_num(auc_v[r][top], nan=0.5) - 0.5)))
            keep_o = float(np.mean(s_tr == np.sign(np.nan_to_num(auc_o[r][top], nan=0.5) - 0.5)))
            within[nm] = {"sign_kept_val": round(keep_v, 3), "sign_kept_oos": round(keep_o, 3),
                          "top5": [feat_cols[i] for i in top[:5]]}
        blk["2_within_regime"] = within

        # 3 DIFFERENTIAL -- the question this pass exists to answer
        d_tr = np.nan_to_num(auc_tr[2], nan=0.5) - np.nan_to_num(auc_tr[0], nan=0.5)
        d_v = np.nan_to_num(auc_v[2], nan=0.5) - np.nan_to_num(auc_v[0], nan=0.5)
        d_o = np.nan_to_num(auc_o[2], nan=0.5) - np.nan_to_num(auc_o[0], nan=0.5)
        top_d = np.argsort(-np.abs(d_tr))[:TOP_N]
        keep_v = float(np.mean(np.sign(d_tr[top_d]) == np.sign(d_v[top_d])))
        keep_o = float(np.mean(np.sign(d_tr[top_d]) == np.sign(d_o[top_d])))
        keep_both = float(np.mean((np.sign(d_tr[top_d]) == np.sign(d_v[top_d]))
                                  & (np.sign(d_tr[top_d]) == np.sign(d_o[top_d]))))
        rand_v, rand_o = [], []
        for _ in range(N_RANDOM):
            sel = rng.choice(len(feat_cols), size=TOP_N, replace=False)
            rand_v.append(float(np.mean(np.sign(d_tr[sel]) == np.sign(d_v[sel]))))
            rand_o.append(float(np.mean(np.sign(d_tr[sel]) == np.sign(d_o[sel]))))
        blk["3_differential"] = {
            "max_abs_delta_auc_train": round(float(np.abs(d_tr).max()), 4),
            "median_abs_delta_top": round(float(np.median(np.abs(d_tr[top_d]))), 4),
            "sign_kept_val": round(keep_v, 3), "sign_kept_oos": round(keep_o, 3),
            "sign_kept_both": round(keep_both, 3),
            "random_subset_baseline": {"val_mean": round(float(np.mean(rand_v)), 3),
                                       "oos_mean": round(float(np.mean(rand_o)), 3)},
            "top10": [{"feature": feat_cols[i], "delta_train": round(float(d_tr[i]), 4),
                       "delta_val": round(float(d_v[i]), 4), "delta_oos": round(float(d_o[i]), 4),
                       "sign_kept": bool(np.sign(d_tr[i]) == np.sign(d_v[i]) == np.sign(d_o[i])),
                       "carrier": is_carrier(feat_cols[i])} for i in top_d[:10]],
        }

        # 4 CARRIERS
        carrier_mask = np.array([is_carrier(c) for c in feat_cols])
        blk["4_carriers"] = {
            "n_carrier_features": int(carrier_mask.sum()),
            "carrier_share_of_top_within_bull": round(float(np.mean(
                carrier_mask[np.argsort(-np.abs(np.nan_to_num(auc_tr[2], nan=0.5) - 0.5))[:TOP_N]])), 3),
            "carrier_share_of_top_differential": round(float(np.mean(carrier_mask[top_d])), 3),
            "carrier_share_of_all_features": round(float(carrier_mask.mean()), 3),
            "mean_abs_delta_carriers": round(float(np.abs(d_tr[carrier_mask]).mean()), 4),
            "mean_abs_delta_others": round(float(np.abs(d_tr[~carrier_mask]).mean()), 4),
        }
        report[gname] = blk
        print(json.dumps({gname: {"occupancy": occ, "median_run": blk["median_run_bars"],
                                  "within_bull_val": within["bull"]["sign_kept_val"],
                                  "diff_sign_kept_val": blk["3_differential"]["sign_kept_val"],
                                  "diff_sign_kept_oos": blk["3_differential"]["sign_kept_oos"],
                                  "diff_random_baseline_oos": blk["3_differential"]["random_subset_baseline"]["oos_mean"],
                                  "carrier_share_diff": blk["4_carriers"]["carrier_share_of_top_differential"]}},
                         ensure_ascii=False), flush=True)

    (OUT_DIR / "regime_feature_differential.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # ---------------- chart
    gnames = list(gates)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    ax = axes[0]
    xs = np.arange(len(gnames))
    kv = [report[g]["3_differential"]["sign_kept_val"] * 100 for g in gnames]
    ko = [report[g]["3_differential"]["sign_kept_oos"] * 100 for g in gnames]
    rb = [report[g]["3_differential"]["random_subset_baseline"]["oos_mean"] * 100 for g in gnames]
    ax.bar(xs - 0.22, kv, width=0.42, color=C_OK, label="VAL 부호 유지")
    ax.bar(xs + 0.22, ko, width=0.42, color=C_BULL, label="OOS 부호 유지")
    ax.plot(xs, rb, marker="_", markersize=26, linestyle="none", color=C_BEAR, label="무작위 기준(OOS)")
    ax.axhline(50, color=C_NEU, linewidth=1.0, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(gnames, fontsize=9, rotation=15)
    ax.set_ylabel("상위 25개 차분 피처의 부호 유지율 %", fontsize=9)
    ax.set_title("① 레짐 차분(bull−bear)은 지속되는가\n50% = 동전던지기", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    g = "jm_lam32"
    top = report[g]["3_differential"]["top10"]
    names = [t["feature"][:26] for t in top][::-1]
    dtr = [t["delta_train"] for t in top][::-1]
    dv = [t["delta_val"] for t in top][::-1]
    do = [t["delta_oos"] for t in top][::-1]
    ys = np.arange(len(names))
    ax.barh(ys + 0.26, dtr, height=0.24, color=C_NEU, label="train")
    ax.barh(ys, dv, height=0.24, color=C_OK, label="VAL")
    ax.barh(ys - 0.26, do, height=0.24, color=C_BULL, label="OOS")
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("ΔAUC = bull − bear", fontsize=9)
    ax.set_title(f"② 차분 상위 10개 ({g})\n부호가 뒤집히면 레짐 효과가 아니다", loc="left",
                 fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    share_all = [report[g]["4_carriers"]["carrier_share_of_all_features"] * 100 for g in gnames]
    share_within = [report[g]["4_carriers"]["carrier_share_of_top_within_bull"] * 100 for g in gnames]
    share_diff = [report[g]["4_carriers"]["carrier_share_of_top_differential"] * 100 for g in gnames]
    ax.bar(xs - 0.26, share_all, width=0.25, color=C_NEU, label="전체 피처 중")
    ax.bar(xs, share_within, width=0.25, color=C_OK, label="레짐 내 상위 중")
    ax.bar(xs + 0.26, share_diff, width=0.25, color=C_BULL, label="차분 상위 중")
    ax.set_xticks(xs)
    ax.set_xticklabels(gnames, fontsize=9, rotation=15)
    ax.set_ylabel("포지셔닝/펀딩/CVD 계열 비중 %", fontsize=9)
    ax.set_title("③ 캐리어 계열이 차분도 담당하는가", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8)

    for a in axes:
        a.grid(axis="x" if a is axes[1] else "y", color="#000000", alpha=0.08, linewidth=0.8)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.suptitle("레짐별 피처 분석 재수행 — 레짐 '내부' 신호가 아니라 레짐 '간 차이'를 본다",
                 fontsize=13, y=1.02)
    out = OUT_DIR / "regime_feature_differential.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
