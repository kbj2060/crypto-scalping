#!/usr/bin/env python3
"""칩 라벨 기준 심층 검증 -- K×ATR 터치 라벨 vs 대안 7종, **공통 잣대**로 비교 (2026-09-04, 서버 GPU).

사용자: "K×ATR 라벨 격자 기준에 회의적이다. 더 좋은 아이디어를 테스트해줘."

## 설계 (결과 보기 전 고정)
모집단  반전 8종 raw 첫발동(GAP12, 방향=페이드=칩 방향) + 추세 5종 첫발동(방향=지속). 피쳐 = F0 프레임 Tier0 23(V자반등 스키마, is_downside 포함).
라벨 8종 (방향 d 기준, 발동봉 종가 entry, 창 H = 반전은 배포 horizon_bars, 추세는 12봉):
  L0 touch        MFE ≥ K·ATR (K = TRAIN 50/50)                            ← 현행 칩 기준
  L1 touch_mae    MFE ≥ K·ATR ∧ MAE < 2K·ATR (5.7절)
  L5 net_exc      (MFE − MAE) ≥ K'·ATR (순유리폭, K' = TRAIN 50/50)           ← 창의: 양방향 변동성 상쇄, 방향 성분만
  L6 end_sign     H봉 뒤 종가 수익률 부호 (시간청산 방향)
  L7 tbar_sym     대칭 트리플배리어 ±1.0·ATR 먼저 닿는 쪽(타임아웃 제외 학습)     ← López de Prado
  L9 touch_floor  MFE ≥ max(K·ATR, 20bp) (5.9절 ATR 하한)
  L3 econ_pos     F0 트레일링(5.0/1.5/0.1, 200봉) − 10bp > 0 (경제 결과)          ← V자반등 경제라벨 발상
  L2 econ_dir     net_bp(방향) > net_bp(반대) (두 측면 경제)
학습  TabPFN 3시드, TRAIN(<2025-09-01) → VAL/OOS.
잣대(라벨 무관, 공통)  모델 확률 상위 30% 선별 발동의 (a) 방향 순손익 net_bp(F0 트레일링, 10bp), (b) 두 측면 차이 net−flip, (c) H봉 만기 방향 정확도,
       각각 전체 평균 대비 이득(top30 − all). 하위 30%도 같이(단조성). AUC는 자기 라벨 기준이라 참고치.
판정  라벨별 신호 평균 선별 이득(OOS·VAL)과 승수(몇 신호에서 L0를 이겼나). HOLDOUT 미접촉.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
KL = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"; FRAME = ROOT / "tmp/homer_entry_v2_20260904/frame.parquet"; CARD = ROOT / "tmp/homer_entry_v2_20260904/model_card.json"
OOFD = ROOT / "tmp/eth_entry_oof_metalabel_20260903"; TRG = ROOT / "data/research/eth_trend_signals_v1_screen_20260904"; POPCFG = ROOT / "tmp/eth_causal_population_metalabel_20260902/config.json"
OUT = ROOT / "data/research/eth_chip_label_alternatives_20260904"
REV = ["taker_delta_z_climax", "short_term_return_z", "liquidity_sweep", "orthogonal_combo", "smt_divergence", "fib_extension_exhaustion", "demarker_extreme", "kalman_deviation_meanrev"]
TREND = ["regime_pullback_resume", "oi_confirmed_breakout", "spot_led_move", "btc_leadlag", "liquidity_vacuum"]
SEEDS = [20260829, 141592, 271828]; GAP = 12; TOP = 0.30; K_GRID = np.round(np.arange(0.1, 6.01, 0.05), 2); FLOOR_BP = 20.0
LABELS = ["L0_touch", "L1_touch_mae", "L5_net_exc", "L6_end_sign", "L7_tbar_sym", "L9_touch_floor", "L3_econ_pos", "L2_econ_dir"]


def log(m): print(f"[labels] {m}", flush=True)


def first_fire(pos_arr, gap):
    keep = []; last = -10**9
    for p in pos_arr:
        if p - last > gap:
            keep.append(p)
        last = p
    return np.array(keep, int)


def main():
    from tabpfn import TabPFNClassifier
    from sklearn.metrics import roc_auc_score
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    kl = pd.read_csv(KL, parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    if kl["timestamp"].dt.tz is not None:
        kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    h, l, c = (kl[x].to_numpy(float) for x in ("high", "low", "close")); kidx = pd.Series(np.arange(len(kl)), index=kl["timestamp"]); n = len(kl)
    prev = np.r_[np.nan, c[:-1]]; tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev))); atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy(); atr_pct = atr / c
    F0 = json.loads(CARD.read_text())["arms"]["F0"]
    D = pd.read_parquet(FRAME, columns=list(dict.fromkeys(["timestamp", "is_downside", "split", "net_bp", "net_bp_flip"] + F0))); D["timestamp"] = pd.to_datetime(D["timestamp"]); key = D.set_index(["timestamp", "is_downside"])
    HZ = {k: int(v["horizon"]) for k, v in json.loads(POPCFG.read_text())["cfg"].items()}
    fires = {}
    for s in REV:
        d = pd.read_csv(OOFD / f"{s}_oof.csv", usecols=["pos", "timestamp", "side"]).drop_duplicates(["pos", "side"]).sort_values("pos"); rows = []
        for side, isd in (("bottom", 1), ("top", 0)):
            ps = first_fire(d.loc[d.side == side, "pos"].to_numpy(), GAP); rows.append(pd.DataFrame({"timestamp": pd.to_datetime(d.set_index("pos").loc[ps, "timestamp"].to_numpy()), "is_downside": isd}))
        fires[s] = (pd.concat(rows), HZ[s])
    for s in TREND:
        p = TRG / f"triggers_{s}.parquet"
        if p.exists():
            t = pd.read_parquet(p); t["timestamp"] = pd.to_datetime(t["timestamp"]); fires[s] = (t[["timestamp", "is_downside"]], 12)
    report = {"labels": LABELS, "top_frac": TOP, "seeds": SEEDS, "holdout_touched": False, "signals": {}}
    for s, (fr, H) in fires.items():
        X = fr.merge(D, on=["timestamp", "is_downside"], how="inner").dropna(subset=F0).reset_index(drop=True)
        X["ki"] = X["timestamp"].map(kidx); X = X[np.isfinite(X["ki"])].copy(); X["ki"] = X["ki"].astype(int); X = X[X["ki"] < n - H - 1].reset_index(drop=True)
        ki = X["ki"].to_numpy(); sgn = np.where(X["is_downside"].to_numpy() == 1, 1.0, -1.0); ent = c[ki]; a = atr_pct[ki]
        fav = np.array([h[i + 1:i + H + 1].max() if sg > 0 else l[i + 1:i + H + 1].min() for i, sg in zip(ki, sgn)]); adv = np.array([l[i + 1:i + H + 1].min() if sg > 0 else h[i + 1:i + H + 1].max() for i, sg in zip(ki, sgn)])
        mfe = sgn * (fav - ent) / ent; mae = -sgn * (adv - ent) / ent; endr = sgn * (c[ki + H] - ent) / ent
        # 대칭 트리플배리어 ±1.0 ATR 먼저 닿는 쪽
        tb = np.full(len(ki), -1)
        for j, (i, sg) in enumerate(zip(ki, sgn)):
            up_b = ent[j] * (1 + a[j]); dn_b = ent[j] * (1 - a[j]); hit_up = np.flatnonzero(h[i + 1:i + H + 1] >= up_b); hit_dn = np.flatnonzero(l[i + 1:i + H + 1] <= dn_b)
            fu = hit_up[0] if len(hit_up) else 10**9; fd = hit_dn[0] if len(hit_dn) else 10**9
            if fu == fd == 10**9: continue
            first_up = fu < fd; tb[j] = 1 if (first_up == (sg > 0)) else 0            # 방향 쪽 배리어가 먼저면 1
        tsX = X["timestamp"]; trm = (X["split"] == "TRAIN").to_numpy(); S = {w: (X["split"] == w).to_numpy() for w in ("VAL", "OOS")}
        pk = mfe / a; mm = mae / a
        def cal(vals, cond=None):
            rates = np.array([((vals >= k) if cond is None else ((vals >= k) & cond(k))).mean() for k in K_GRID]); j = int(np.argmin(np.abs(rates - 0.5))); return float(K_GRID[j])
        K0 = cal(pk[trm]); K1 = cal(pk[trm], lambda k: mm[trm] < 2 * k); K5 = cal((pk - mm)[trm])
        Y = {"L0_touch": (pk >= K0).astype(int), "L1_touch_mae": ((pk >= K1) & (mm < 2 * K1)).astype(int), "L5_net_exc": ((pk - mm) >= K5).astype(int), "L6_end_sign": (endr > 0).astype(int),
             "L7_tbar_sym": tb, "L9_touch_floor": (mfe >= np.maximum(K0 * a, FLOOR_BP / 1e4)).astype(int), "L3_econ_pos": (X["net_bp"].to_numpy() > 0).astype(int), "L2_econ_dir": (X["net_bp"].to_numpy() > X["net_bp_flip"].to_numpy()).astype(int)}
        net = X["net_bp"].to_numpy(); flip = X["net_bp_flip"].to_numpy(); dir_ok = (endr > 0).astype(int)
        R = {"H": H, "n": {"TRAIN": int(trm.sum()), **{w: int(m.sum()) for w, m in S.items()}}, "K": {"L0": K0, "L1": K1, "L5": K5}, "labels": {}}
        for lab in LABELS:
            y = Y[lab]; valid = y >= 0; trv = trm & valid
            if y[trv].mean() in (0, 1) or trv.sum() < 100:
                R["labels"][lab] = {"skipped": "degenerate"}; continue
            per = {w: [] for w in S}
            for sd in SEEDS:
                clf = TabPFNClassifier(device="cuda", random_state=int(sd), ignore_pretraining_limits=True).fit(X.loc[trv, F0], y[trv])
                for w, m in S.items():
                    per[w].append(clf.predict_proba(X.loc[m, F0])[:, 1])
            res = {"train_rate": round(float(y[trv].mean()), 3)}
            for w, m in S.items():
                p = np.mean(per[w], axis=0); k = max(int(m.sum() * TOP), 10); idx = np.flatnonzero(m); top = idx[np.argsort(-p)[:k]]; bot = idx[np.argsort(p)[:k]]
                yy = y[m]; vm = yy >= 0
                res[w] = {"auc_own": round(float(roc_auc_score(yy[vm], p[vm])), 4) if len(np.unique(yy[vm])) > 1 else None,
                          "all": {"net": round(float(net[m].mean()), 2), "diff": round(float((net[m] - flip[m]).mean()), 2), "dir": round(float(dir_ok[m].mean()), 3)},
                          "top30": {"net": round(float(net[top].mean()), 2), "diff": round(float((net[top] - flip[top]).mean()), 2), "dir": round(float(dir_ok[top].mean()), 3)},
                          "bot30": {"net": round(float(net[bot].mean()), 2), "diff": round(float((net[bot] - flip[bot]).mean()), 2), "dir": round(float(dir_ok[bot].mean()), 3)}}
                res[w]["gain_net"] = round(res[w]["top30"]["net"] - res[w]["all"]["net"], 2); res[w]["gain_dir"] = round(res[w]["top30"]["dir"] - res[w]["all"]["dir"], 3); res[w]["mono"] = bool(res[w]["top30"]["net"] > res[w]["bot30"]["net"])
            R["labels"][lab] = res
            log(f"{s:>26s} {lab:>14s} tr_rate {res['train_rate']:.2f} | " + " | ".join(f"{w} auc {res[w]['auc_own']} top30 net {res[w]['top30']['net']:+6.1f} (all {res[w]['all']['net']:+5.1f}) dir {res[w]['top30']['dir']:.3f} (all {res[w]['all']['dir']:.3f})" for w in S) + f"  ({time.time()-t0:.0f}s)")
        report["signals"][s] = R
        (OUT / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    # 라벨별 요약: 신호 평균 선별 이득, L0 대비 승수
    summ = {}
    for lab in LABELS:
        for w in ("VAL", "OOS"):
            g = [R["labels"][lab][w]["gain_net"] for R in report["signals"].values() if "skipped" not in R["labels"].get(lab, {"skipped": 1})]
            gd = [R["labels"][lab][w]["gain_dir"] for R in report["signals"].values() if "skipped" not in R["labels"].get(lab, {"skipped": 1})]
            wins = [1 for R in report["signals"].values() if "skipped" not in R["labels"].get(lab, {"skipped": 1}) and "skipped" not in R["labels"].get("L0_touch", {"skipped": 1}) and R["labels"][lab][w]["gain_net"] > R["labels"]["L0_touch"][w]["gain_net"]]
            summ[f"{lab}/{w}"] = {"mean_gain_net_bp": round(float(np.mean(g)), 2) if g else None, "mean_gain_dir": round(float(np.mean(gd)), 3) if gd else None, "n_signals": len(g), "beats_L0": len(wins)}
    report["summary"] = summ; (OUT / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    print(f"\n{'label/window':>22s} {'mean gain net bp':>16s} {'mean gain dir':>13s} {'n':>3s} {'beats L0':>8s}")
    for k, v in summ.items():
        print(f"{k:>22s} {v['mean_gain_net_bp']!s:>16s} {v['mean_gain_dir']!s:>13s} {v['n_signals']:>3d} {v['beats_L0']:>8d}")
    print("LABELS_DONE")


if __name__ == "__main__":
    main()
