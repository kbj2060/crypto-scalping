#!/usr/bin/env python3
"""칩 라벨 교체 검증 -- 라이브 결정 모집단에서 **경제 방향 라벨(L2: net_bp > net_bp_flip)** 컨텍스트 vs 현재 배포 컨텍스트(터치 라벨) (2026-09-04, 서버 GPU).

라벨 대안 연구(eth_chip_label_alternatives_20260904)에서 반전 8종은 L2(두 측면 경제 방향)가, 추세 5종은 L5(순유리폭)가 공통 잣대(상위30% 선별 이득)에서
K×ATR 터치를 이겼다. 여기서는 **배포 가능한 형태**로 재검한다:
  · 피쳐 = 라이브 칩 스키마(taker FEATURE_COLUMNS 23 + demarker/kalman 고유 1개)  ← chipacc 프레임(raw 발동 전부)
  · 모집단 = 라이브 결정 봉(같은 측면 raw 발동이 직전 horizon 안에 없음), TRAIN(<2025-09-01) 컨텍스트
  · 라벨 = 반전: L2 (F0 프레임 두 측면 경제라벨, 페이드 측면 행) / 추세: L5 (MFE−MAE ≥ K'·ATR, H=12, K' TRAIN 50/50)
  · 비교 = D(현재 배포 컨텍스트, 터치 라벨) vs NEW: VAL/OOS 상위30% 선별의 (net, net−flip) 이득, 방향 정확도, 자기라벨 AUC(참고), 5시드
  · 판정(사전): NEW의 diff-이득이 VAL·OOS 모두 D보다 크고 OOS top30 diff > 0 → 교체 후보
동결 컨텍스트: data/labels/eth_5m_evidence_chip_econdir_20260904/<sig>_train_context_L2_live_20260904.csv (추세: eth_5m_trend_signals_v1_20260904/<sig>_train_context_L5_20260904.csv)
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
FRAMES = ROOT / "tmp/eth_chip_accuracy_upgrade_20260904/frames"; F0FRAME = ROOT / "tmp/homer_entry_v2_20260904/frame.parquet"; POPCFG = ROOT / "tmp/eth_causal_population_metalabel_20260902/config.json"
TRG = ROOT / "data/research/eth_trend_signals_v1_screen_20260904"; KL = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUTR = ROOT / "data/research/eth_chip_relabel_econ_dir_20260904"; CTX_R = ROOT / "data/labels/eth_5m_evidence_chip_econdir_20260904"; CTX_T = ROOT / "data/labels/eth_5m_trend_signals_v1_20260904"
REV = ["taker_delta_z_climax", "short_term_return_z", "liquidity_sweep", "orthogonal_combo", "smt_divergence", "fib_extension_exhaustion", "demarker_extreme", "kalman_deviation_meanrev"]
TREND = ["regime_pullback_resume", "oi_confirmed_breakout", "spot_led_move", "btc_leadlag", "liquidity_vacuum"]; SEEDS = [20260829, 141592, 271828, 577215, 20260904]; TOP = 0.3


def log(m): print(f"[relabel] {m}", flush=True)


def live_mask(pos, isb, H):
    m = np.zeros(len(pos), bool)
    for sd in (0, 1):
        ii = np.flatnonzero(isb == sd); p = pos[ii]; last = -10**9
        for j, x in enumerate(p):
            if x - last >= H: m[ii[j]] = True
            last = x
    return m


def metrics(y, p, net, flip, endr):
    from sklearn.metrics import roc_auc_score
    k = max(int(len(y) * TOP), 10); top = np.argsort(-p)[:k]; bot = np.argsort(p)[:k]
    o = {"n": int(len(y)), "auc_own": round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) > 1 else None,
         "net_all": round(float(net.mean()), 2), "net_top30": round(float(net[top].mean()), 2), "net_bot30": round(float(net[bot].mean()), 2),
         "diff_all": round(float((net - flip).mean()), 2), "diff_top30": round(float((net - flip)[top].mean()), 2), "diff_bot30": round(float((net - flip)[bot].mean()), 2)}
    o["gain_net"] = round(o["net_top30"] - o["net_all"], 2); o["gain_diff"] = round(o["diff_top30"] - o["diff_all"], 2)
    if endr is not None:
        o["dir_all"] = round(float((endr > 0).mean()), 3); o["dir_top30"] = round(float((endr[top] > 0).mean()), 3)
    return o


def main():
    from tabpfn import TabPFNClassifier
    from live_evidence_signal_metalabel_20260829 import METALABEL_SIGNALS, FEATURE_COLUMNS
    t0 = time.time(); OUTR.mkdir(parents=True, exist_ok=True); CTX_R.mkdir(parents=True, exist_ok=True); CTX_T.mkdir(parents=True, exist_ok=True)
    F0 = pd.read_parquet(F0FRAME, columns=["timestamp", "is_downside", "net_bp", "net_bp_flip"]); F0["timestamp"] = pd.to_datetime(F0["timestamp"]); econ = F0.set_index(["timestamp", "is_downside"])
    kl = pd.read_csv(KL, parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True); c = kl["close"].to_numpy(float); kidx = pd.Series(np.arange(len(kl)), index=kl["timestamp"])
    report = {"holdout_touched": False, "signals": {}}
    for sig in REV:
        cfg = METALABEL_SIGNALS[sig]; H = int(cfg["horizon_bars"]); cols = list(cfg.get("feature_columns", FEATURE_COLUMNS))
        d = pd.read_parquet(FRAMES / f"{sig}.parquet"); d["timestamp"] = pd.to_datetime(d["timestamp"])
        d = d[live_mask(d["pos"].to_numpy(), d["is_bottom"].to_numpy(), H)].copy()          # 라이브 결정 봉
        d["is_downside"] = d["is_bottom"].astype(int)                                          # 페이드 측면 = 칩 방향
        e = econ.reindex(pd.MultiIndex.from_arrays([d["timestamp"].to_numpy(), d["is_downside"].to_numpy()], names=["timestamp", "is_downside"]))
        d["net_bp"] = e["net_bp"].to_numpy(); d["net_bp_flip"] = e["net_bp_flip"].to_numpy(); d = d.dropna(subset=["net_bp"] + cols).reset_index(drop=True)
        d["y_econ"] = (d["net_bp"] > d["net_bp_flip"]).astype(int); ki = d["timestamp"].map(kidx).to_numpy(); ok = np.isfinite(ki); d = d[ok].reset_index(drop=True); ki = ki[ok].astype(int)
        sgn = np.where(d["is_downside"] == 1, 1.0, -1.0); endr = sgn * (c[np.minimum(ki + H, len(c) - 1)] - c[ki]) / c[ki]
        tr = (d["split"] == "TRAIN").to_numpy(); S = {w: (d["split"] == w).to_numpy() for w in ("VAL", "OOS")}
        R = {"H": H, "n": {"TRAIN": int(tr.sum()), **{w: int(m.sum()) for w, m in S.items()}}, "arms": {}}
        # D: 배포 컨텍스트 (터치 라벨) → 같은 잣대
        ctx = pd.read_csv(cfg["train_context"]); ycol = "label" if "label" in ctx.columns else "hit"; per = {w: [] for w in S}
        for sd in SEEDS[:3]:
            clf = TabPFNClassifier(device="cuda", random_state=int(sd)).fit(ctx[cols], ctx[ycol].to_numpy().astype(int))
            for w, m in S.items(): per[w].append(clf.predict_proba(d.loc[m, cols])[:, 1])
        R["arms"]["D_touch_deployed"] = {w: metrics(d.loc[m, "y_econ"].to_numpy(), np.mean(per[w], axis=0), d.loc[m, "net_bp"].to_numpy(), d.loc[m, "net_bp_flip"].to_numpy(), endr[m]) for w, m in S.items()}
        # NEW: L2 컨텍스트 (라이브 결정 봉 TRAIN)
        per = {w: [] for w in S}; aucs = {w: [] for w in S}
        for sd in SEEDS:
            clf = TabPFNClassifier(device="cuda", random_state=int(sd)).fit(d.loc[tr, cols], d.loc[tr, "y_econ"].to_numpy())
            for w, m in S.items():
                p = clf.predict_proba(d.loc[m, cols])[:, 1]; per[w].append(p)
        R["arms"]["NEW_L2_live"] = {w: {**metrics(d.loc[m, "y_econ"].to_numpy(), np.mean(per[w], axis=0), d.loc[m, "net_bp"].to_numpy(), d.loc[m, "net_bp_flip"].to_numpy(), endr[m]),
                                        "gain_diff_per_seed": [round(float(metrics(d.loc[m, "y_econ"].to_numpy(), q, d.loc[m, "net_bp"].to_numpy(), d.loc[m, "net_bp_flip"].to_numpy(), None)["gain_diff"]), 2) for q in per[w]]} for w, m in S.items()}
        Dm, Nm = R["arms"]["D_touch_deployed"], R["arms"]["NEW_L2_live"]
        ok_rule = all(Nm[w]["gain_diff"] > Dm[w]["gain_diff"] for w in S) and Nm["OOS"]["diff_top30"] > 0 and all(sum(1 for g in Nm[w]["gain_diff_per_seed"] if g > Dm[w]["gain_diff"]) >= 4 for w in S)
        R["verdict"] = "REPLACE_CANDIDATE" if ok_rule else "KEEP"
        d.loc[tr, ["pos", "timestamp", "side", "is_bottom", "net_bp", "net_bp_flip", "y_econ"] + [x for x in cols if x != "is_bottom"]].rename(columns={"y_econ": "hit"}).to_csv(CTX_R / f"{sig}_train_context_L2_live_20260904.csv", index=False)
        report["signals"][sig] = R
        log(f"{sig:>26s} n {R['n']} | D gain diff {Dm['VAL']['gain_diff']}/{Dm['OOS']['gain_diff']} net {Dm['VAL']['gain_net']}/{Dm['OOS']['gain_net']} | NEW gain diff {Nm['VAL']['gain_diff']}/{Nm['OOS']['gain_diff']} net {Nm['VAL']['gain_net']}/{Nm['OOS']['gain_net']} top30diff OOS {Nm['OOS']['diff_top30']} auc {Nm['VAL']['auc_own']}/{Nm['OOS']['auc_own']} => {R['verdict']} ({time.time()-t0:.0f}s)")
    # 추세 5종: L5 (MFE−MAE ≥ K'·ATR, H=12) -- 피쳐 = chipacc와 같은 build_indicator_frame 스키마가 필요 → 호메로스 파이프라인의 컨텍스트(터치 라벨)와 같은 행에 L5 라벨을 붙인다
    from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame
    ind = build_indicator_frame(kl); h, l = kl["high"].to_numpy(float), kl["low"].to_numpy(float); atrp = ind["atr_pct"].to_numpy(float); n = len(kl); feat = [x for x in FEATURE_COLUMNS if x != "is_bottom"]
    K_GRID = np.round(np.arange(0.1, 6.01, 0.05), 2); H = 12
    for sig in TREND:
        t = pd.read_parquet(TRG / f"triggers_{sig}.parquet"); t["timestamp"] = pd.to_datetime(t["timestamp"]); ki = t["timestamp"].map(kidx).to_numpy(); ok = np.isfinite(ki); t = t[ok].reset_index(drop=True); ki = ki[ok].astype(int); keep = ki < n - H - 1; t = t[keep].reset_index(drop=True); ki = ki[keep]
        sgn = np.where(t["is_downside"] == 1, 1.0, -1.0); ent = c[ki]; a = atrp[ki]
        fav = np.array([h[i + 1:i + H + 1].max() if s_ > 0 else l[i + 1:i + H + 1].min() for i, s_ in zip(ki, sgn)]); adv = np.array([l[i + 1:i + H + 1].min() if s_ > 0 else h[i + 1:i + H + 1].max() for i, s_ in zip(ki, sgn)])
        netexc = (sgn * (fav - ent) / ent - (-sgn * (adv - ent) / ent)) / a; endr = sgn * (c[ki + H] - ent) / ent
        X = ind.iloc[ki][feat].reset_index(drop=True); X["is_bottom"] = t["is_downside"].astype(int).to_numpy(); X["timestamp"] = t["timestamp"].to_numpy(); X["pos"] = ki
        e = econ.reindex(pd.MultiIndex.from_arrays([t["timestamp"].to_numpy(), t["is_downside"].astype(int).to_numpy()], names=["timestamp", "is_downside"])); X["net_bp"] = e["net_bp"].to_numpy(); X["net_bp_flip"] = e["net_bp_flip"].to_numpy(); X["netexc"] = netexc; X["endr"] = endr
        X["split"] = np.where(X["timestamp"] < pd.Timestamp("2025-09-01"), "TRAIN", np.where(X["timestamp"] < pd.Timestamp("2026-01-01"), "VAL", "OOS")); X = X.dropna(subset=feat + ["net_bp"]).reset_index(drop=True)
        tr = (X["split"] == "TRAIN").to_numpy(); S = {w: (X["split"] == w).to_numpy() for w in ("VAL", "OOS")}
        rates = np.array([(X.loc[tr, "netexc"] >= k).mean() for k in K_GRID]); K5 = float(K_GRID[int(np.argmin(np.abs(rates - 0.5)))]); X["hit"] = (X["netexc"] >= K5).astype(int)
        cols = feat + ["is_bottom"]; per = {w: [] for w in S}
        for sd in SEEDS:
            clf = TabPFNClassifier(device="cuda", random_state=int(sd), ignore_pretraining_limits=True).fit(X.loc[tr, cols], X.loc[tr, "hit"].to_numpy())
            for w, m in S.items(): per[w].append(clf.predict_proba(X.loc[m, cols])[:, 1])
        R = {"H": H, "K5": K5, "n": {"TRAIN": int(tr.sum()), **{w: int(m.sum()) for w, m in S.items()}}, "arms": {"NEW_L5": {}}}
        for w, m in S.items():
            R["arms"]["NEW_L5"][w] = {**metrics(X.loc[m, "hit"].to_numpy(), np.mean(per[w], axis=0), X.loc[m, "net_bp"].to_numpy(), X.loc[m, "net_bp_flip"].to_numpy(), X.loc[m, "endr"].to_numpy()),
                                      "auc_per_seed": [round(float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(X.loc[m, "hit"], q)), 4) for q in per[w]]}
        N = R["arms"]["NEW_L5"]; ok_rule = all(N[w]["auc_own"] is not None and N[w]["auc_own"] >= 0.55 for w in S) and all(N[w]["diff_top30"] > N[w]["diff_all"] for w in S) and N["OOS"]["net_top30"] > 0
        R["verdict"] = "CHIP_CANDIDATE" if ok_rule else "NOT_YET"
        X.loc[tr, ["pos", "timestamp", "is_bottom", "hit", "netexc", "net_bp", "net_bp_flip"] + feat].to_csv(CTX_T / f"{sig}_train_context_L5_20260904.csv", index=False)
        report["signals"][sig] = R
        log(f"{sig:>26s} n {R['n']} K5 {K5} | L5 auc {N['VAL']['auc_own']}/{N['OOS']['auc_own']} gain diff {N['VAL']['gain_diff']}/{N['OOS']['gain_diff']} net top30 {N['VAL']['net_top30']}/{N['OOS']['net_top30']} (all {N['VAL']['net_all']}/{N['OOS']['net_all']}) => {R['verdict']} ({time.time()-t0:.0f}s)")
    (OUTR / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str)); print("RELABEL_DONE")


if __name__ == "__main__":
    main()
