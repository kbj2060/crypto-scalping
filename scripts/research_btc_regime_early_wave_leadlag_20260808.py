"""Follow-up: is the early-weighted model a LEADING turn indicator? (2026-08-08)

The position-weighted study produced a result that does not look like a regime classifier at all:
  weight scheme       total   Q1     Q5     median run
  uniform (frozen)    70.1    34.0   90.2   8
  exp(-pos/0.30)      52.1    78.3   26.6   10
  exp(-pos/0.15)      42.2    84.1   12.4   19
The tau=0.15 model is 84% right at the START of a wave and 12% right at the END -- i.e. late in a
wave it is already calling the opposite direction, with LONGER runs (19 bars) than the frozen
model, so it is not oscillating.  Two explanations fit that shape: (i) it leads -- it flips before
the pivot, anticipating the turn; (ii) it lags so badly it is still showing the previous wave.
Only (i) is consistent with a HIGH Q1, but the lag profile settles it directly: a leading detector
peaks at NEGATIVE k when agreement is measured against the oracle shifted by k bars.

Also charts the two frontiers the research round produced: best zigzag threshold by wave position,
and the total-vs-early agreement exchange curve.  Research only -- nothing is selected or adopted.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from audit_btc_regime_classifier_lag_20260808 import agree, dir_of, lag_profile, wave_position  # noqa: E402
from chart_btc_jm_regime_verification_20260808 import causal_zigzag  # noqa: E402
from ensemble_btc_regime_classifier_theta005_20260808 import logit, sigmoid  # noqa: E402
from refine_btc_regime_classifier_theta005_20260808 import (  # noqa: E402
    PANEL_PATH, PURGE, jump_decode_proba, to_named,
)
from test_statistical_jump_model_regimes_20260808 import zigzag_oracle  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    TRAIN_END, VAL_START, VAL_END,
)

OUT_DIR = ROOT / "tmp/regime_early_wave_20260808"
FROZEN_PATH = ROOT / "data/research/btc_regime_theta005_frozen_config_20260808.json"
THETA = 0.005
INK, C_A, C_B, C_C, C_D = "#1F2430", "#2563EB", "#D9542B", "#0E7C66", "#7C3AED"

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(FROZEN_PATH.read_text())
    thetas = frozen["pipeline"]["1_features"]["thresholds"]
    seeds = frozen["pipeline"]["2_nowcaster"]["seed_bag"]
    lam = float(frozen["pipeline"]["4_decode"].split("lambda=")[1])
    research = json.loads((OUT_DIR / "early_wave_research.json").read_text())

    panel = pd.read_csv(PANEL_PATH, low_memory=False, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts = panel["timestamp"]
    close = panel["close"].to_numpy(dtype=np.float64)
    o_dir, pivots = zigzag_oracle(close, threshold=THETA)
    pos = wave_position(o_dir, pivots, len(close))
    train_mask = (ts <= TRAIN_END).to_numpy()
    tr_idx = np.flatnonzero(train_mask)[:-PURGE]
    tr_idx = tr_idx[o_dir[tr_idx] != 0]
    y = (o_dir[tr_idx] == 1).astype(int)
    tr_pos = np.nan_to_num(pos[tr_idx], nan=0.5)
    v_idx = np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy())
    xm = np.column_stack([causal_zigzag(close, threshold=t) for t in thetas]).astype(np.float32)
    fast = causal_zigzag(close, threshold=0.001).astype(np.int8)
    slow = causal_zigzag(close, threshold=0.005).astype(np.int8)
    suspect = (fast != slow) & (fast != 0) & (slow != 0)

    def fit(w):
        ps = []
        for s in seeds:
            clf = lgb.LGBMClassifier(objective="binary", n_estimators=700, learning_rate=0.05,
                                     num_leaves=63, min_child_samples=200, feature_fraction=0.8,
                                     bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
                                     random_state=s, n_jobs=-1, verbosity=-1)
            clf.fit(xm[tr_idx], y, sample_weight=w)
            ps.append(clf.predict_proba(xm)[:, 1])
        return np.mean(ps, axis=0)

    variants = {}
    p_uni = fit(None)
    variants["uniform (고정 모델)"] = to_named(jump_decode_proba(p_uni, lam))
    print("fitted uniform", flush=True)
    for tau, label in ((0.30, "exp τ0.30"), (0.15, "exp τ0.15")):
        p = fit(np.exp(-tr_pos / tau))
        variants[label] = to_named(jump_decode_proba(p, lam))
        print(f"fitted {label}", flush=True)
    z = logit(p_uni) + 0.5 * np.where(suspect, fast.astype(float), 0.0)
    variants["turn-suspect boost0.5"] = to_named(jump_decode_proba(sigmoid(z), lam))

    prof = {}
    for name, st in variants.items():
        d = dir_of(st)
        lp = lag_profile(d, o_dir, v_idx)
        prof[name] = {"peak_lag_bars": lp["peak_lag_bars"], "peak_agreement_pct": lp["peak_agreement_pct"],
                      "at_lag0_pct": lp["at_lag0_pct"], "profile": lp["profile"],
                      "leads": bool(lp["peak_lag_bars"] is not None and lp["peak_lag_bars"] < 0)}
        print(json.dumps({name: {k: prof[name][k] for k in ("peak_lag_bars", "peak_agreement_pct",
                                                            "at_lag0_pct", "leads")}}), flush=True)
    (OUT_DIR / "leadlag.json").write_text(json.dumps(prof, indent=2, ensure_ascii=False))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    ax = axes[0]
    for (name, pr), c in zip(prof.items(), (C_A, C_C, C_D, C_B)):
        ks = sorted(int(k) for k in pr["profile"])
        ax.plot(ks, [pr["profile"][str(k)] if str(k) in pr["profile"] else pr["profile"][k] for k in ks],
                color=c, linewidth=1.7, label=name)
    ax.axvline(0, color="#9AA0A6", linewidth=1.0, linestyle="--")
    ax.set_title("① 선행/지연 판별 (VAL)\n피크가 음수 k면 선행 지표", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("k (바) — 음수 = 오라클을 미래로 당겼을 때", fontsize=9)
    ax.set_ylabel("일치율 %", fontsize=9)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    fro = research["A_frontier"]["by_threshold"]
    qs = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    for k, c in zip(["czz_0.0005", "czz_0.001", "czz_0.002", "czz_0.005", "czz_0.012"],
                    ("#7C3AED", C_A, C_C, C_B, "#B45309")):
        ax.plot(range(5), [fro[k][q] for q in qs], marker="o", markersize=5, color=c,
                linewidth=1.6, label=k.replace("czz_", "θ=") )
    ax.set_xticks(range(5))
    ax.set_xticklabels(["파동\n초반", "Q2", "Q3", "Q4", "파동\n후반"], fontsize=9)
    ax.set_title("② 파동 위치별 최적 임계값이 다르다\n초반=가장 미세, 후반=목표 스케일", loc="left",
                 fontsize=11, color=INK)
    ax.set_ylabel("일치율 %", fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[2]
    ex = research["D_exchange_rate"]
    ko = {"B_uniform": "가중 없음(고정)", "B_linear": "선형 가중", "B_exp_tau015": "exp τ0.15",
          "B_exp_tau030": "exp τ0.30", "B_early_only_q1q2": "초반만", "C_route_boost0.5": "라우팅 0.5",
          "C_route_boost1": "라우팅 1.0", "C_route_boost2": "라우팅 2.0"}
    for k, v in ex.items():
        c = C_C if k.startswith("C_") else C_A
        ax.scatter(v["Q1"], v["total"], s=48, color=c, zorder=3)
        ax.annotate(ko.get(k, k), (v["Q1"], v["total"]), fontsize=7.5, color=INK,
                    xytext=(4, 4), textcoords="offset points")
    ax.scatter([research["A_frontier"]["oracle_routed_bound"]["Q1"]],
               [research["A_frontier"]["oracle_routed_bound"]["total"]],
               marker="*", s=220, color="#B45309", zorder=4)
    ax.annotate("사후 라우팅 상한", (research["A_frontier"]["oracle_routed_bound"]["Q1"],
                                   research["A_frontier"]["oracle_routed_bound"]["total"]),
                fontsize=8, color="#B45309", xytext=(6, -10), textcoords="offset points")
    ax.set_xlabel("파동 초반(Q1) 일치율 %", fontsize=9)
    ax.set_ylabel("전체 일치율 %", fontsize=9)
    ax.set_title("③ 교환율 — 초반을 사려면 전체를 판다\n(파랑=가중 재학습, 주황=인과 라우팅)",
                 loc="left", fontsize=11, color=INK)

    for a in axes:
        a.grid(color="#000000", alpha=0.08, linewidth=0.8)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.suptitle("파동 초반 정보 심층 연구 — 선행성 판별 · 위치별 최적 스케일 · 교환율", fontsize=13, y=1.02)
    out = OUT_DIR / "early_wave_research.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
