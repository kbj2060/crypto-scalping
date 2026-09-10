#!/usr/bin/env python3
"""종합 청산 모델 성능 스윕 (2026-09-10) -- 사용자 목표 "AUC 60%". 정당한 레버를 전부 시험하고 누수 대조군을 같이 돈다.

레버: (1) HGB 용량 (2) 추가 입력 OI 4종 (3) 라벨 지평 fwd12 고정 · 상한 48봉 (4) 리스크 라벨(잔여 구간 역행 ≥1ATR)
      (5) 상태 조건부 부분집합(|u|≥1ATR) (6) 모집단: 증거신호 발동 봉 진입만
누수 대조군: 피쳐를 한 봉 **뒤**(lag, 정직해야 소폭 하락)와 한 봉 **앞**(lead, 미래참조 -- 크게 뛰어야 정상)으로 밀어
  각 라벨의 "누수 시 기대 AUC" 를 재고, 단변량 최대 AUC 도 같이 낸다(CLAUDE.md: 모델−단변량 격차 5pp 이상이면 누수 의심).
선택은 VAL, 보고는 OOS/HOLDOUT. 스크리닝은 1시드, 최종 후보만 5시드.
출력 tmp/eth_exit_synth_20260910/sweep.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import exit_synth_features_20260910 as EF  # noqa: E402
import build_eth_exit_synth_dataset_20260910 as BD  # noqa: E402
from research_eth_exit_synth_model_20260910 import SPLITS, SEEDS, HGB, cluster_boot_delta_auc  # noqa: E402

D = ROOT / "tmp/eth_exit_synth_20260910"
HGB_BIG = dict(max_iter=800, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=500, l2_regularization=2.0,
               early_stopping=False)


def log(m: str) -> None:
    print(f"[sweep {time.strftime('%H:%M:%S')}] {m}", flush=True)


def windows(ts: pd.Series) -> dict[str, np.ndarray]:
    return {k: ((ts >= a) & (ts <= b)).to_numpy() for k, (a, b) in SPLITS.items()}


def fit_eval(X: np.ndarray, y: np.ndarray, win: dict, seed: int, hgb: dict = HGB) -> dict[str, float]:
    m = HistGradientBoostingClassifier(random_state=seed, **hgb).fit(X[win["TRAIN"]], y[win["TRAIN"]])
    return {w: float(roc_auc_score(y[win[w]], m.predict_proba(X[win[w]])[:, 1])) for w in ("VAL", "OOS", "HOLDOUT")}


def univariate_max(X: np.ndarray, y: np.ndarray, cols: list[str], mask: np.ndarray) -> tuple[str, float]:
    best = ("", 0.5)
    yy = y[mask]
    for j, c in enumerate(cols):
        x = X[mask, j]
        ok = np.isfinite(x)
        if ok.sum() < 1000 or yy[ok].min() == yy[ok].max():
            continue
        a = roc_auc_score(yy[ok], x[ok]); a = max(a, 1 - a)
        if a > best[1]:
            best = (c, float(a))
    return best


def oi_features(eth: pd.DataFrame) -> pd.DataFrame:
    m = pd.read_csv(ROOT / "data/TOTAL_ETHUSDT_metrics_2024_2026.csv", parse_dates=["create_time"])
    s = pd.Series(np.log(m["sum_open_interest_value"].to_numpy(float)), index=m["create_time"] - pd.Timedelta(minutes=5))
    v = s.reindex(pd.DatetimeIndex(eth["timestamp"])).ffill(limit=6)
    out = pd.DataFrame({"oi_d12": v.diff(12), "oi_d48": v.diff(48), "oi_d288": v.diff(288),
                        "oi_z288": (v - v.rolling(288, min_periods=96).mean()) / v.rolling(288, min_periods=96).std()})
    return out.reset_index(drop=True)


def risk_label(ck: pd.DataFrame, eth: pd.DataFrame, cap: int, k_atr: float = 1.0) -> np.ndarray:
    """잔여 구간(t+1..t0+cap)의 역행이 ATR k배 이상인가 -- 방향이 아니라 리스크를 묻는 라벨."""
    H = eth["high"].to_numpy(float); L = eth["low"].to_numpy(float); C = eth["close"].to_numpy(float)
    t = ck["t"].to_numpy(); t0 = ck["t0"].to_numpy(); s = ck["pos_side"].to_numpy()
    ct = C[t]; worst = np.zeros(len(ck))
    end = t0 + cap
    for j in range(1, cap):
        idx = np.minimum(t + j, len(C) - 1)
        adv = np.where(s > 0, (L[idx] - ct) / ct, (ct - H[idx]) / ct)
        worst = np.where(t + j <= end, np.minimum(worst, adv), worst)
    atr = ck["pos_u"].to_numpy() / np.where(ck["pos_u_atr"].to_numpy() != 0, ck["pos_u_atr"].to_numpy(), np.nan)
    atr = np.where(np.isfinite(atr), atr, np.nanmedian(atr))
    return (worst <= -k_atr * np.abs(atr)).astype(np.int8)


def main() -> int:
    ck = pd.read_parquet(D / "checkpoints.parquet"); bars = pd.read_parquet(D / "bars.parquet")
    man = json.loads((D / "manifest.json").read_text()); fcols = man["feature_cols"]; pcols = EF.POS_COLS
    eth = BD.load_klines("eth", "ETHUSDT")
    assert len(eth) == len(bars)
    bars = pd.concat([bars, oi_features(eth)], axis=1)
    ocols = ["oi_d12", "oi_d48", "oi_d288", "oi_z288"]
    t = ck["t"].to_numpy(); ts = pd.to_datetime(ck["ts"]); win = windows(ts)
    y = ck["y_exit"].to_numpy()
    Xpos = ck[pcols].to_numpy(np.float32)
    Xd = bars[fcols].to_numpy(np.float32)
    XB = np.hstack([Xpos, Xd[t]])
    res: dict[str, dict] = {}
    s0 = SEEDS[0]

    def run(name, X, yy, w=win, hgb=HGB, cols=None):
        r = fit_eval(X, yy, w, s0, hgb)
        uv = univariate_max(X, yy, cols or (pcols + fcols), w["OOS"]) if cols is not False else None
        res[name] = {"auc": r, "univariate_max_oos": uv, "n_train": int(w["TRAIN"].sum()), "pos_rate": float(yy[w["TRAIN"]].mean())}
        log(f"{name:28s} VAL {r['VAL']:.4f} OOS {r['OOS']:.4f} HO {r['HOLDOUT']:.4f}" + (f" · 단변량 최대 {uv[0]} {uv[1]:.4f}" if uv else ""))

    run("base_B", XB, y)
    # 누수 대조군: 대시보드 피쳐를 한 봉 뒤/앞으로
    run("ctrl_lag1_B", np.hstack([Xpos, Xd[np.maximum(t - 1, 0)]]), y, cols=False)
    run("ctrl_lead1_B(leak)", np.hstack([Xpos, Xd[np.minimum(t + 1, len(Xd) - 1)]]), y, cols=False)
    run("lever1_hgb_big", XB, y, hgb=HGB_BIG, cols=False)
    XBo = np.hstack([XB, bars[ocols].to_numpy(np.float32)[t]])
    run("lever2_plus_oi", XBo, y, cols=pcols + fcols + ocols)
    run("lever3_label_fwd12", XB, (ck["fwd12"].to_numpy() < 0).astype(np.int8))
    yr = risk_label(ck, eth, man["cap"], 1.0)
    run("lever4_risk_mae1atr", XB, yr)
    run("lever4_risk_mae1atr_ctrl_lead1", np.hstack([Xpos, Xd[np.minimum(t + 1, len(Xd) - 1)]]), yr, cols=False)
    sub = np.abs(ck["pos_u_atr"].to_numpy()) >= 1.0
    wsub = {k: v & sub for k, v in win.items()}
    run("lever5_subset_|u|>=1atr", XB, y, w=wsub, cols=False)
    # 모집단: 증거신호 발동 봉(어느 측면이든 12봉 안) 진입만
    fired_recent = (bars["ev_bottom_n12"].to_numpy() + bars["ev_top_n12"].to_numpy()) > 0
    ent = fired_recent[ck["t0"].to_numpy()]
    went = {k: v & ent for k, v in win.items()}
    run("lever6_pop_evidence_entry", XB, y, w=went, cols=False)
    # 상한 48봉 재구축
    BD.CAP, BD.CK = 48, list(range(1, 48, 4))
    ck48 = BD.build_checkpoints(bars, eth); ck48 = ck48[np.isfinite(ck48["pos_u"])].reset_index(drop=True)
    t48 = ck48["t"].to_numpy(); w48 = windows(pd.to_datetime(ck48["ts"]))
    X48 = np.hstack([ck48[pcols].to_numpy(np.float32), Xd[t48]])
    run("lever3_label_cap48", X48, ck48["y_exit"].to_numpy(), w=w48, cols=False)
    run("lever4_risk_cap48_mae1atr", X48, risk_label(ck48, eth, 48, 1.0), w=w48, cols=False)
    # 최종: 방향형 라벨 중 VAL 최고 팔 5시드 + 군집 CI (base 대비)
    dir_arms = {k: v for k, v in res.items() if k.startswith(("base", "lever1", "lever2", "lever3_label_fwd12"))}
    best = max(dir_arms, key=lambda k: dir_arms[k]["auc"]["VAL"])
    Xbest = {"base_B": XB, "lever1_hgb_big": XB, "lever2_plus_oi": XBo, "lever3_label_fwd12": XB}[best]
    ybest = (ck["fwd12"].to_numpy() < 0).astype(np.int8) if best == "lever3_label_fwd12" else y
    hb = HGB_BIG if best == "lever1_hgb_big" else HGB
    P = {"best": np.zeros(len(y)), "base": np.zeros(len(y))}
    seeds_auc = {"best": [], "base": []}
    for s in SEEDS:
        for nm, X_, y_, h_ in (("best", Xbest, ybest, hb), ("base", XB, ybest, HGB)):
            m = HistGradientBoostingClassifier(random_state=s, **h_).fit(X_[win["TRAIN"]], y_[win["TRAIN"]])
            P[nm][~win["TRAIN"]] += m.predict_proba(X_[~win["TRAIN"]])[:, 1] / len(SEEDS)
            seeds_auc[nm].append({w: float(roc_auc_score(y_[win[w]], m.predict_proba(X_[win[w]])[:, 1])) for w in ("VAL", "OOS")})
    final = {"best_arm": best, "ensemble_auc": {nm: {w: float(roc_auc_score(ybest[win[w]], P[nm][win[w]])) for w in ("VAL", "OOS", "HOLDOUT")} for nm in P},
             "seeds": seeds_auc}
    for w in ("VAL", "OOS"):
        lo, hi, mean = cluster_boot_delta_auc(ybest[win[w]], P["base"][win[w]], P["best"][win[w]], ck["pid"].to_numpy()[win[w]], n_boot=400)
        final[f"delta_vs_base_{w}"] = {"ci95": [lo, hi], "mean": mean}
    res["final"] = final
    log(f"final {best}: " + json.dumps(final["ensemble_auc"]) + " · Δ " + json.dumps({k: v for k, v in final.items() if k.startswith('delta')}))
    (D / "sweep.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
