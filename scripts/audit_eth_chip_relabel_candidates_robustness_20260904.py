"""교체 후보 칩(L2 경제방향 컨텍스트) 견고성 감사 — 2026-09-04

relabel(research_eth_chip_relabel_econ_dir_contexts_20260904)이 REPLACE_CANDIDATE로 판정한 칩에 대해,
diff-이득(top30 − 전체)이 (a) 규모(ATR) 기계효과가 아닌지 → ATR 정규화 이득, (b) 무작위 부분표집 귀무(B=500)보다 큰지,
(c) 일 군집 부트스트랩 CI(top30 − 나머지), (d) 5시드 각각에서 재현되는지 확인한다. 배포 컨텍스트(D)도 같은 잣대로 병기.
사전 판정: REPLACE = 두 창 ATR정규화 이득 > 0 ∧ OOS 귀무 백분위 ≥ 95 ∧ VAL 귀무 백분위 ≥ 90 ∧ 두 창 모두 5시드 중 ≥4 ATR정규화 이득 > 0.
HOLDOUT 미접촉. 산출: data/research/eth_chip_relabel_econ_dir_20260904/robustness_<sig>.{json,parquet}
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
FRAMES = ROOT / "tmp/eth_chip_accuracy_upgrade_20260904/frames"; F0FRAME = ROOT / "tmp/homer_entry_v2_20260904/frame.parquet"
KL = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"; OUTR = ROOT / "data/research/eth_chip_relabel_econ_dir_20260904"
CAND = [a for a in sys.argv[1:]] or ["short_term_return_z", "liquidity_sweep"]
SEEDS = [20260829, 141592, 271828, 577215, 20260904]; TOP = 0.3; B_NULL = 500; B_CI = 1000
from research_eth_chip_relabel_econ_dir_contexts_20260904 import live_mask


def log(m): print(f"[robust] {m}", flush=True)


def gains(p, diff, diff_atr, endr):
    k = max(int(len(p) * TOP), 10); top = np.zeros(len(p), bool); top[np.argsort(-p)[:k]] = True
    return {"gain_diff": float(diff[top].mean() - diff.mean()), "gain_diff_atr": float(diff_atr[top].mean() - diff_atr.mean()),
            "gain_dir": float((endr[top] > 0).mean() - (endr > 0).mean()), "top_minus_rest_diff": float(diff[top].mean() - diff[~top].mean())}, top


def null_pct(obs, diff, diff_atr, rng):
    k = max(int(len(diff) * TOP), 10); g1, g2 = [], []
    for _ in range(B_NULL):
        ii = rng.choice(len(diff), k, replace=False); g1.append(diff[ii].mean() - diff.mean()); g2.append(diff_atr[ii].mean() - diff_atr.mean())
    g1, g2 = np.array(g1), np.array(g2)
    return {"pct_gain_diff": float((g1 < obs["gain_diff"]).mean() * 100), "pct_gain_diff_atr": float((g2 < obs["gain_diff_atr"]).mean() * 100),
            "null_sd_gain_diff": float(g1.std()), "null_sd_gain_diff_atr": float(g2.std())}


def cluster_ci(top, diff, days, rng):
    u = np.unique(days); idx = {d: np.flatnonzero(days == d) for d in u}; vals = []
    for _ in range(B_CI):
        pick = rng.choice(u, len(u), replace=True); ii = np.concatenate([idx[d] for d in pick]); t = top[ii]
        if t.sum() == 0 or (~t).sum() == 0: continue
        vals.append(diff[ii][t].mean() - diff[ii][~t].mean())
    return {"ci_lo": float(np.percentile(vals, 2.5)), "ci_hi": float(np.percentile(vals, 97.5)), "n_days": int(len(u))}


def main():
    from tabpfn import TabPFNClassifier
    from live_evidence_signal_metalabel_20260829 import METALABEL_SIGNALS, FEATURE_COLUMNS
    t0 = time.time(); rng = np.random.default_rng(20260904)
    F0 = pd.read_parquet(F0FRAME, columns=["timestamp", "is_downside", "net_bp", "net_bp_flip", "atr", "entry"]); F0["timestamp"] = pd.to_datetime(F0["timestamp"]); econ = F0.set_index(["timestamp", "is_downside"])
    kl = pd.read_csv(KL, parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True); c = kl["close"].to_numpy(float); kidx = pd.Series(np.arange(len(kl)), index=kl["timestamp"])
    for sig in CAND:
        cfg = METALABEL_SIGNALS[sig]; H = int(cfg["horizon_bars"]); cols = list(cfg.get("feature_columns", FEATURE_COLUMNS))
        d = pd.read_parquet(FRAMES / f"{sig}.parquet"); d["timestamp"] = pd.to_datetime(d["timestamp"])
        d = d[live_mask(d["pos"].to_numpy(), d["is_bottom"].to_numpy(), H)].copy(); d["is_downside"] = d["is_bottom"].astype(int)
        e = econ.reindex(pd.MultiIndex.from_arrays([d["timestamp"].to_numpy(), d["is_downside"].to_numpy()], names=["timestamp", "is_downside"]))
        d["net_bp"] = e["net_bp"].to_numpy(); d["net_bp_flip"] = e["net_bp_flip"].to_numpy(); atr, ent = e["atr"].to_numpy(float), e["entry"].to_numpy(float)
        d["atr_bp"] = np.where(atr < 1.0, atr * 1e4, atr / ent * 1e4)                       # F0 atr 단위 자동 판별(비율 or 가격)
        d = d.dropna(subset=["net_bp", "atr_bp"] + cols).reset_index(drop=True)
        d["y_econ"] = (d["net_bp"] > d["net_bp_flip"]).astype(int); ki = d["timestamp"].map(kidx).to_numpy(); ok = np.isfinite(ki); d = d[ok].reset_index(drop=True); ki = ki[ok].astype(int)
        sgn = np.where(d["is_downside"] == 1, 1.0, -1.0); d["endr"] = sgn * (c[np.minimum(ki + H, len(c) - 1)] - c[ki]) / c[ki]
        d["diff"] = d["net_bp"] - d["net_bp_flip"]; d["diff_atr"] = d["diff"] / d["atr_bp"]; d["day"] = d["timestamp"].dt.strftime("%Y-%m-%d")
        tr = (d["split"] == "TRAIN").to_numpy(); S = {w: (d["split"] == w).to_numpy() for w in ("VAL", "OOS")}
        log(f"{sig} n TRAIN {tr.sum()} VAL {S['VAL'].sum()} OOS {S['OOS'].sum()} atr_bp median {np.median(d['atr_bp']):.1f}")
        ctx = pd.read_csv(cfg["train_context"]); ycol = "label" if "label" in ctx.columns else "hit"
        pD = {w: [] for w in S}; pN = {w: [] for w in S}
        for sd in SEEDS[:3]:
            clf = TabPFNClassifier(device="cuda", random_state=int(sd)).fit(ctx[cols], ctx[ycol].to_numpy().astype(int))
            for w, m in S.items(): pD[w].append(clf.predict_proba(d.loc[m, cols])[:, 1])
        for sd in SEEDS:
            clf = TabPFNClassifier(device="cuda", random_state=int(sd)).fit(d.loc[tr, cols], d.loc[tr, "y_econ"].to_numpy())
            for w, m in S.items(): pN[w].append(clf.predict_proba(d.loc[m, cols])[:, 1])
        R = {"signal": sig, "H": H, "n": {"TRAIN": int(tr.sum()), **{w: int(m.sum()) for w, m in S.items()}}, "atr_bp_median": float(np.median(d["atr_bp"])), "arms": {}}
        for arm, P in (("D_touch_deployed", pD), ("NEW_L2_live", pN)):
            R["arms"][arm] = {}
            for w, m in S.items():
                diff, diff_atr, endr, days = d.loc[m, "diff"].to_numpy(), d.loc[m, "diff_atr"].to_numpy(), d.loc[m, "endr"].to_numpy(), d.loc[m, "day"].to_numpy()
                p = np.mean(P[w], axis=0); g, top = gains(p, diff, diff_atr, endr)
                o = {**{k: round(v, 4) for k, v in g.items()}, **{k: round(v, 3) for k, v in null_pct(g, diff, diff_atr, rng).items()}, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in cluster_ci(top, diff, days, rng).items()}}
                o["per_seed_gain_diff_atr"] = [round(gains(q, diff, diff_atr, endr)[0]["gain_diff_atr"], 4) for q in P[w]]
                o["per_seed_gain_dir"] = [round(gains(q, diff, diff_atr, endr)[0]["gain_dir"], 4) for q in P[w]]
                dec = pd.qcut(pd.Series(p).rank(method="first"), 5, labels=False); o["quintile_mean_diff"] = [round(float(diff[dec == q].mean()), 2) for q in range(5)]
                R["arms"][arm][w] = o
                if arm == "NEW_L2_live": d.loc[m, "p_new"] = p
                else: d.loc[m, "p_dep"] = p
        N = R["arms"]["NEW_L2_live"]
        ok = all(N[w]["gain_diff_atr"] > 0 for w in S) and N["OOS"]["pct_gain_diff"] >= 95 and N["OOS"]["pct_gain_diff_atr"] >= 95 and N["VAL"]["pct_gain_diff"] >= 90 and N["VAL"]["pct_gain_diff_atr"] >= 90 \
            and all(sum(1 for g in N[w]["per_seed_gain_diff_atr"] if g > 0) >= 4 for w in S)
        R["verdict"] = "REPLACE" if ok else "KEEP"
        (OUTR / f"robustness_{sig}.json").write_text(json.dumps(R, ensure_ascii=False, indent=1))
        d.loc[~tr, ["timestamp", "split", "is_downside", "p_new", "p_dep", "net_bp", "net_bp_flip", "diff", "diff_atr", "endr", "day"]].to_parquet(OUTR / f"robustness_{sig}.parquet", index=False)
        for w in S:
            log(f"{sig:>22s} {w} NEW gain diff {N[w]['gain_diff']:.2f} atr {N[w]['gain_diff_atr']:.3f} dir {N[w]['gain_dir']:+.3f} | null pct {N[w]['pct_gain_diff']:.0f}/{N[w]['pct_gain_diff_atr']:.0f} | top-rest CI [{N[w]['ci_lo']:.1f},{N[w]['ci_hi']:.1f}] days {N[w]['n_days']} | seeds atr {N[w]['per_seed_gain_diff_atr']} | quintiles {N[w]['quintile_mean_diff']}")
            D = R["arms"]["D_touch_deployed"][w]; log(f"{'':>22s} {w} DEP gain diff {D['gain_diff']:.2f} atr {D['gain_diff_atr']:.3f} dir {D['gain_dir']:+.3f} | null pct {D['pct_gain_diff']:.0f}/{D['pct_gain_diff_atr']:.0f} | quintiles {D['quintile_mean_diff']}")
        log(f"{sig} => {R['verdict']} ({time.time()-t0:.0f}s)")
    print("ROBUST_DONE", flush=True)


if __name__ == "__main__":
    main()
