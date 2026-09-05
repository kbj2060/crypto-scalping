#!/usr/bin/env python3
"""칩 라벨 뒤집기 — **실제 재학습**으로 확인 (2026-09-05).

사용자: *"증거신호의 라벨 0을 지속 신호의 라벨 1로 두고 학습하면 지속 증거신호가 되는 거 아니야?"* → *"재학습을 실시한 거야?"*

앞선 진단(`diagnose_eth_chip_prob_flipped_as_continuation_selector_20260905.py`)은 **기존 OOF 확률**을 쓰고
"라벨을 뒤집어 학습해도 같은 분류기"라는 항등식에 기댔다. 그 전제를 **실제 학습기로 검증**하고,
항등식이 안 통하는 더 강한 형태(**지속 수익성 자체를 라벨로 새 모델 학습**)까지 돌린다.

## 세 타깃 (같은 피쳐·같은 학습기·같은 분할)
  T1  y = hit          칩 라벨 재현 (기준)
  T2  y = 1 − hit      ⭐**라벨 뒤집기** — p_T2 ≈ 1 − p_T1 이면 "같은 분류기"가 실증된다
  T3  y = (지속 net_bp > 0)   ⭐**지속 수익성 자체를 라벨로** — 항등식이 안 통하는 진짜 새 모델
  (T4 회귀는 축 14가 이미 함 — 상위10%가 8종 중 5종을 악화. 여기서는 분류만)

## 규격
  모집단  8종 raw 첫발동(GAP12) 합집합, (봉,측면) 중복 제거 — 5.23/축14와 동일
  피쳐    Tier0 22종 + is_downside + 신호 원핫 8 (신호별 캘리브레이션 차이를 모델이 흡수하게 함)
  학습기  HistGradientBoosting, 5시드. **TRAIN에서만 적합**, VAL/OOS는 표본외 1회 평가
  평가    (a) 항등식: corr(p_T1, 1−p_T2) · max|Δ| · AUC 동일성
          (b) AUC: 각 모델이 "지속이 이익인가"를 얼마나 가르는가
          (c) 매매 팔: 상위/하위 30% 선별 · 순위 사이징 → cont_all 대비 일별 짝비교 CI (§5.27 표준)
  누수가드 VAL AUC ≥ 0.99면 즉시 중단
판정: VAL·OOS 두 창 모두 짝비교 CI 하한 > 0. HOLDOUT 미접촉.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


C1 = _load("c1_rt", "scripts/research_eth_composite_direction_trend_pullback_20260905.py")
OOFD = ROOT / "tmp/eth_entry_oof_metalabel_20260903"
OUT = ROOT / "data/research/eth_chip_label_flip_retrain_20260905"
SEEDS = [20260905, 771103, 480219, 913057, 264488]
FEATS = ["sweep_penetration_atr", "atr_percentile_864", "range_width_pct", "hour_utc", "weekday",
         "p_fast", "p_slow", "vwap_dev_z", "cvd_roll_roc_48", "vol_z", "lower_wick_ratio", "upper_wick_ratio",
         "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile", "ret3_z", "rsi", "delta_z",
         "flow_aligned_delta_z", "atr_pct"]
WINDOWS = ("TRAIN", "VAL", "OOS")


def log(m): print(f"[retrain] {m}", flush=True)


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    B = C1.build()
    pos, sd, split, ts = B["pos"], B["sd"], B["split"], B["ts"]
    cont_bp, cont_ex, D = B["cont_bp"], B["cont_ex"], B["D"]
    Fp = B["Fp"].reset_index(drop=True)
    key = D.set_index(["pos", "is_downside"]).reindex(pd.MultiIndex.from_arrays([pos, sd], names=["pos", "is_downside"]))
    X = np.hstack([key[FEATS].to_numpy(float), sd.reshape(-1, 1).astype(float)])   # is_downside는 인덱스 레벨이라 따로 붙인다
    oh = pd.get_dummies(pd.Categorical(Fp["signal"], categories=C1.SIGNALS)).to_numpy(float)   # 신호 원핫
    X = np.hstack([X, oh])
    rows = []
    for s in C1.SIGNALS:
        d = pd.read_csv(OOFD / f"{s}_oof.csv", usecols=["pos", "side", "hit"])
        d["is_downside"] = (d["side"] == "bottom").astype(int); d["signal"] = s
        rows.append(d[["pos", "is_downside", "signal", "hit"]])
    O = pd.concat(rows).drop_duplicates(["pos", "is_downside", "signal"])
    hit = pd.DataFrame({"pos": pos, "is_downside": sd, "signal": Fp["signal"].to_numpy()}).merge(
        O, on=["pos", "is_downside", "signal"], how="left")["hit"].to_numpy(float)
    y_cont = (cont_bp > 0).astype(int)
    tr = split == "TRAIN"
    ok = np.isfinite(hit) & np.isfinite(X).all(1)
    log(f"n={len(pos):,} 유효={int(ok.sum()):,} · 피쳐 {X.shape[1]} · TRAIN {int((tr&ok).sum()):,}")

    P = {}
    for tag, y in (("T1_hit", hit), ("T2_flip", 1 - hit), ("T3_cont_profit", y_cont.astype(float))):
        ps = []
        for sdd in SEEDS:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=4,
                                               l2_regularization=1.0, random_state=sdd)
            m.fit(X[tr & ok], y[tr & ok])
            ps.append(m.predict_proba(X)[:, 1])
        P[tag] = np.mean(ps, axis=0)
        au = {w: round(float(roc_auc_score(y[(split == w) & ok], P[tag][(split == w) & ok])), 4) for w in WINDOWS}
        log(f"  {tag}: 자기 라벨 AUC {au}")
        if au["VAL"] >= 0.99:
            log("⛔ 누수 가드 발동 (VAL AUC ≥ 0.99)"); return 1

    rep = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "seeds": SEEDS, "n_features": int(X.shape[1]),
           "holdout_touched": False, "n": int(len(pos))}
    # (a) 항등식
    d = P["T1_hit"] - (1.0 - P["T2_flip"])
    rep["flip_identity"] = {"corr_p1_vs_1minus_p2": round(float(np.corrcoef(P["T1_hit"][ok], (1 - P["T2_flip"])[ok])[0, 1]), 6),
                            "max_abs_diff": round(float(np.abs(d[ok]).max()), 6), "mean_abs_diff": round(float(np.abs(d[ok]).mean()), 6),
                            "auc_hit_T1": {w: round(float(roc_auc_score(hit[(split == w) & ok], P["T1_hit"][(split == w) & ok])), 4) for w in WINDOWS},
                            "auc_hit_1minusT2": {w: round(float(roc_auc_score(hit[(split == w) & ok], (1 - P["T2_flip"])[(split == w) & ok])), 4) for w in WINDOWS}}
    log(f"항등식: corr {rep['flip_identity']['corr_p1_vs_1minus_p2']} · max|Δ| {rep['flip_identity']['max_abs_diff']} · 평균|Δ| {rep['flip_identity']['mean_abs_diff']}")
    log(f"  AUC(hit): T1 {rep['flip_identity']['auc_hit_T1']} vs 1−T2 {rep['flip_identity']['auc_hit_1minusT2']}")
    # (b) 지속 수익성 판별력
    rep["auc_for_continuation_profit"] = {}
    for tag, sign in (("T1_hit", -1.0), ("T2_flip", +1.0), ("T3_cont_profit", +1.0)):
        rep["auc_for_continuation_profit"][tag] = {
            w: round(float(roc_auc_score(y_cont[(split == w) & ok], sign * P[tag][(split == w) & ok])), 4) for w in WINDOWS}
    log(f"지속이익 AUC: {json.dumps(rep['auc_for_continuation_profit'], ensure_ascii=False)}")
    # (c) 매매 팔
    base = {w: C1.pf(C1.cand_of(ts[split == w], pos[split == w] + 1, pos[split == w] + 1 + cont_ex[split == w], cont_bp[split == w])) for w in WINDOWS}
    rep["baseline_cont_all"] = {w: base[w]["stats"] for w in WINDOWS}
    rep["arms"] = {}
    for tag, score in (("T3_cont_profit", P["T3_cont_profit"]), ("1minus_T1_hit", 1.0 - P["T1_hit"]), ("T2_flip", P["T2_flip"])):
        for q in (0.30, 0.50):
            nm = f"{tag}_top{int(q*100)}%"
            rec = {}
            for w in WINDOWS:
                m = (split == w) & ok
                thr = np.quantile(score[m], 1 - q); mm = m & (score >= thr)
                if mm.sum() < 100:
                    continue
                r = C1.pf(C1.cand_of(ts[mm], pos[mm] + 1, pos[mm] + 1 + cont_ex[mm], cont_bp[mm]))
                rec[w] = {"n": r["stats"]["n"], "exp_bp": r["stats"]["exp_bp"], "day_ci95": r["stats"]["day_ci95"],
                          "vs_cont_all": C1.day_paired(r["pnl"], r["ts"], base[w]["pnl"], base[w]["ts"])}
            rep["arms"][nm] = rec
        # 순위 사이징 (건수 유지)
        nm = f"{tag}_size_by_rank"
        rec = {}
        for w in WINDOWS:
            m = (split == w) & ok
            rk = pd.Series(score[m]).rank(pct=True).to_numpy()
            wt = 1.0 + 0.5 * (2 * rk - 1); wt = wt / wt.mean()
            r = C1.pf(C1.cand_of(ts[m], pos[m] + 1, pos[m] + 1 + cont_ex[m], cont_bp[m] * wt))
            rec[w] = {"n": r["stats"]["n"], "exp_bp": r["stats"]["exp_bp"], "daily_sharpe_ann": r["stats"]["daily_sharpe_ann"],
                      "vs_cont_all": C1.day_paired(r["pnl"], r["ts"], base[w]["pnl"], base[w]["ts"])}
        rep["arms"][nm] = rec
    P_ = {"rule": "VAL·OOS 두 창 모두 vs_cont_all CI 하한 > 0", "passes": []}
    for nm, rec in rep["arms"].items():
        v, o = rec.get("VAL", {}).get("vs_cont_all"), rec.get("OOS", {}).get("vs_cont_all")
        if v and o and v["ci95"][0] > 0 and o["ci95"][0] > 0:
            P_["passes"].append(nm)
    P_["n_pass"] = len(P_["passes"]); rep["verdict"] = P_
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"완료 {time.time()-t0:.0f}s → {OUT/'report.json'}")
    for nm, rec in rep["arms"].items():
        if "VAL" in rec:
            print(f"  {nm:34s} " + " | ".join(
                f"{w} n={rec[w]['n']:>5} exp={rec[w]['exp_bp']:>6} Δ={rec[w]['vs_cont_all']['diff_bp_day']:>7}{str(rec[w]['vs_cont_all']['ci95']):>18}" for w in WINDOWS if w in rec))
    log(f"판정: 통과 {P_['n_pass']}개 {P_['passes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
