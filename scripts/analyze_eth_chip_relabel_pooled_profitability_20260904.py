"""반전 칩 8종 풀 — 'L2 재학습 확률로 발동을 고르면 수익성이 좋아지나' (2026-09-04, 사용자 질문)

입력: data/research/eth_chip_relabel_econ_dir_20260904/robustness_<sig>.parquet (VAL/OOS 발동별 p_new(L2 5시드 평균)·p_dep(배포 3시드)·net_bp(10bp 차감)·diff·endr·day)
선택 = 신호 안에서 확률 상위 30%(칩별 순위 → 풀). 비교 = 전체 발동 / 배포 확률 상위 30% / L2 확률 상위 30%.
불확실성 = 일 군집 부트스트랩(B=2000, 신호 공통 날짜 = 같은 날 상관까지 흡수) + 신호별 층화 무작위 30% 귀무(B=1000).
연구/개발 점수(사전 저장 라벨, 프레시포워드 아님) — 승격 근거 아님.
"""
from pathlib import Path
import json, sys
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "data/research/eth_chip_relabel_econ_dir_20260904"
REV = ["taker_delta_z_climax", "short_term_return_z", "liquidity_sweep", "orthogonal_combo", "smt_divergence", "fib_extension_exhaustion", "demarker_extreme", "kalman_deviation_meanrev"]
TOP = 0.3; B_CI = 2000; B_NULL = 1000; rng = np.random.default_rng(20260904)


def top_flag(g, col):
    k = max(int(len(g) * TOP), 10); f = np.zeros(len(g), bool); f[np.argsort(-g[col].to_numpy())[:k]] = True; return f


def cluster_ci(x, days, fn):
    u = np.unique(days); idx = {d: np.flatnonzero(days == d) for d in u}; v = []
    for _ in range(B_CI):
        ii = np.concatenate([idx[d] for d in rng.choice(u, len(u), replace=True)]); r = fn(ii)
        if r is not None: v.append(r)
    return [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)]


def main():
    fr = []
    for s in REV:
        p = D / f"robustness_{s}.parquet"
        if not p.exists(): print("missing", s); continue
        d = pd.read_parquet(p); d["signal"] = s; fr.append(d)
    X = pd.concat(fr, ignore_index=True); X["timestamp"] = pd.to_datetime(X["timestamp"]); out = {"signals": sorted(X["signal"].unique().tolist()), "windows": {}}
    for w in ("VAL", "OOS"):
        Y = X[X["split"] == w].reset_index(drop=True); Y["top_new"] = False; Y["top_dep"] = False; Y["q_new"] = 0
        for s, g in Y.groupby("signal"):
            Y.loc[g.index, "top_new"] = top_flag(g, "p_new"); Y.loc[g.index, "top_dep"] = top_flag(g, "p_dep")
            Y.loc[g.index, "q_new"] = pd.qcut(g["p_new"].rank(method="first"), 5, labels=False).to_numpy()
        net, diff, endr, days = Y["net_bp"].to_numpy(), Y["diff"].to_numpy(), Y["endr"].to_numpy(), Y["day"].to_numpy(); tn, td = Y["top_new"].to_numpy(), Y["top_dep"].to_numpy()
        ndays = len(np.unique(days))
        o = {"n_fires": int(len(Y)), "n_days": int(ndays), "fires_per_day": round(len(Y) / ndays, 1), "picks_per_day_top30": round(tn.sum() / ndays, 1),
             "net_all": round(float(net.mean()), 2), "net_top30_dep": round(float(net[td].mean()), 2), "net_top30_new": round(float(net[tn].mean()), 2), "net_bot30_new": round(float(net[Y["q_new"] <= 0].mean()), 2),
             "diff_all": round(float(diff.mean()), 2), "diff_top30_dep": round(float(diff[td].mean()), 2), "diff_top30_new": round(float(diff[tn].mean()), 2),
             "dir_all": round(float((endr > 0).mean()), 3), "dir_top30_dep": round(float((endr[td] > 0).mean()), 3), "dir_top30_new": round(float((endr[tn] > 0).mean()), 3),
             "win_rate_top30_new": round(float((net[tn] > 0).mean()), 3), "win_rate_all": round(float((net > 0).mean()), 3),
             "overlap_top30_new_vs_dep": round(float((tn & td).sum() / tn.sum()), 3),
             "quintile_net_new": [round(float(net[Y["q_new"] == q].mean()), 2) for q in range(5)],
             "ci_net_top30_new": cluster_ci(net, days, lambda ii: float(net[ii][tn[ii]].mean()) if tn[ii].any() else None),
             "ci_gain_new_vs_all": cluster_ci(net, days, lambda ii: float(net[ii][tn[ii]].mean() - net[ii].mean()) if tn[ii].any() else None),
             "ci_new_vs_dep": cluster_ci(net, days, lambda ii: float(net[ii][tn[ii]].mean() - net[ii][td[ii]].mean()) if tn[ii].any() and td[ii].any() else None),
             "ci_diff_gain_new_vs_all": cluster_ci(diff, days, lambda ii: float(diff[ii][tn[ii]].mean() - diff[ii].mean()) if tn[ii].any() else None)}
        # 층화 무작위 귀무: 신호별 30%를 무작위로 골랐을 때 풀 이득 분포
        sig = Y["signal"].to_numpy(); groups = {s: np.flatnonzero(sig == s) for s in np.unique(sig)}; g_net, g_diff = [], []
        for _ in range(B_NULL):
            pick = np.concatenate([rng.choice(ii, max(int(len(ii) * TOP), 10), replace=False) for ii in groups.values()])
            g_net.append(net[pick].mean() - net.mean()); g_diff.append(diff[pick].mean() - diff.mean())
        g_net, g_diff = np.array(g_net), np.array(g_diff)
        o["gain_net_new"] = round(o["net_top30_new"] - o["net_all"], 2); o["gain_net_dep"] = round(o["net_top30_dep"] - o["net_all"], 2)
        o["null_pct_gain_net_new"] = round(float((g_net < o["gain_net_new"]).mean() * 100), 1); o["null_pct_gain_net_dep"] = round(float((g_net < o["gain_net_dep"]).mean() * 100), 1)
        o["null_pct_gain_diff_new"] = round(float((g_diff < (o["diff_top30_new"] - o["diff_all"])).mean() * 100), 1); o["null_sd_gain_net"] = round(float(g_net.std()), 2)
        # 신호별 부호 집계
        per = []
        for s, g in Y.groupby("signal"):
            a, b, c = g["net_bp"].mean(), g.loc[g["top_dep"], "net_bp"].mean(), g.loc[g["top_new"], "net_bp"].mean(); per.append((s, round(a, 2), round(b, 2), round(c, 2)))
        o["per_signal_net_all_dep_new"] = per; o["n_signals_new_gt_all"] = int(sum(1 for _, a, _, c in per if c > a)); o["n_signals_new_gt_dep"] = int(sum(1 for _, _, b, c in per if c > b))
        out["windows"][w] = o
        print(f"== {w}: fires {o['n_fires']} days {o['n_days']} ({o['fires_per_day']}/day, top30 picks {o['picks_per_day_top30']}/day)")
        print(f"  net/pick  all {o['net_all']:+.2f} | dep top30 {o['net_top30_dep']:+.2f} | NEW top30 {o['net_top30_new']:+.2f} (CI {o['ci_net_top30_new']}) | NEW bot20 {o['net_bot30_new']:+.2f} | quintiles {o['quintile_net_new']}")
        print(f"  gain NEW vs all {o['gain_net_new']:+.2f} CI {o['ci_gain_new_vs_all']} null pct {o['null_pct_gain_net_new']} | dep vs all {o['gain_net_dep']:+.2f} null pct {o['null_pct_gain_net_dep']} | NEW vs dep {o['net_top30_new']-o['net_top30_dep']:+.2f} CI {o['ci_new_vs_dep']}")
        print(f"  diff  all {o['diff_all']:+.2f} | dep {o['diff_top30_dep']:+.2f} | NEW {o['diff_top30_new']:+.2f} (gain CI {o['ci_diff_gain_new_vs_all']}, null pct {o['null_pct_gain_diff_new']}) | dir all/dep/NEW {o['dir_all']}/{o['dir_top30_dep']}/{o['dir_top30_new']} | win all/NEW {o['win_rate_all']}/{o['win_rate_top30_new']} | overlap {o['overlap_top30_new_vs_dep']}")
        print(f"  signals NEW>all {o['n_signals_new_gt_all']}/8, NEW>dep {o['n_signals_new_gt_dep']}/8")
    (D / "pooled_profitability.json").write_text(json.dumps(out, ensure_ascii=False, indent=1)); print("POOLED_DONE")


if __name__ == "__main__":
    main()
