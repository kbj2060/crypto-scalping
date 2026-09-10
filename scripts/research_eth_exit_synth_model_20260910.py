#!/usr/bin/env python3
"""종합 청산 모델 학습·평가·동결 (2026-09-10) -- 사용자 지시 "대시보드 실시간 데이터·지표·증거신호·이벤트를 종합".

입력  tmp/eth_exit_synth_20260910/{bars,checkpoints}.parquet (build_eth_exit_synth_dataset_20260910.py)
팔    A = 포지션 상태만(POS_COLS)  ·  B = A + 대시보드 종합 피쳐 전부
      (A 가 "시계+기하" 기준선이다 -- 09-04 Phase 0 에서 학습 청산모델 AUC 0.60 의 실체가 pos_bars_left 였다)
분할  TRAIN <2025-09-01 · VAL 2025-09~12 · OOS 2026-01~03 · HOLDOUT 2026-04~ (CLAUDE.md 표준, HOLDOUT 은 보고만)
사전등록 킬게이트 (전부 통과해야 gate_pass=true):
  G1 ΔAUC(B−A) 포지션 군집 부트 95% CI 하한 > 0, VAL 과 OOS 둘 다
  G2 무작위 5시드 ΔAUC 5/5 양수(VAL·OOS)
  G3 정책값: B 의 "p≥τ 면 지금 청산"(τ 는 VAL 에서 선택) 평균 bp 가 VAL·OOS 에서 (a) 상한까지 보유 (b) 같은
     비율 무작위 청산 둘 다를 이김
결과와 무관하게 아티팩트는 동결하고 카드는 수치를 그대로 표시한다(사용자 결정). gate_pass 가 화면에 나간다.
출력  tmp/eth_exit_synth_20260910/report.json · data/live/eth_exit_synth_artifact/{model.joblib,meta.json}
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import exit_synth_features_20260910 as EF  # noqa: E402

D = ROOT / "tmp/eth_exit_synth_20260910"
ART = ROOT / "data/live/eth_exit_synth_artifact"
SPLITS = {"TRAIN": ("2024-01-01", "2025-08-31 23:59"), "VAL": ("2025-09-01", "2025-12-31 23:59"),
          "OOS": ("2026-01-01", "2026-03-31 23:59"), "HOLDOUT": ("2026-04-01", "2030-01-01")}
SEEDS = [int(s) for s in np.random.default_rng(20260910).integers(1, 10**6, 5)]   # 랜덤 추출(고정 간격 아님)
HGB = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
           early_stopping=False)
EXIT_RATE_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
N_BOOT = 1000


def log(m: str) -> None:
    print(f"[exit-model {time.strftime('%H:%M:%S')}] {m}", flush=True)


def cluster_boot_delta_auc(y, pa, pb, pid, n_boot=N_BOOT, seed=0):
    """포지션(pid) 단위 리샘플 -- 체크포인트 단위면 CI 가 가짜로 좁아진다(09-03 설계 수정2)."""
    rng = np.random.default_rng(seed)
    order = np.argsort(pid, kind="stable"); y, pa, pb, pid = y[order], pa[order], pb[order], pid[order]
    uniq, start = np.unique(pid, return_index=True)
    end = np.append(start[1:], len(pid))
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([np.arange(start[i], end[i]) for i in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        out.append(roc_auc_score(yy, pb[idx]) - roc_auc_score(yy, pa[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float(np.mean(out))


def policy_value(df: pd.DataFrame, p: np.ndarray, tau: float, rng=None) -> float:
    """포지션별 첫 p≥τ 체크포인트에서 청산, 없으면 상한 보유. 평균 bp. rng 가 있으면 같은 비율 무작위 청산."""
    d = df[["pid", "pos_hold", "pos_u", "fwd_cap"]].copy()
    d["p"] = p
    d = d.sort_values(["pid", "pos_hold"])
    if rng is not None:
        rate = float((p >= tau).mean())
        d["hit"] = rng.random(len(d)) < rate
    else:
        d["hit"] = d["p"] >= tau
    first = d[d["hit"]].groupby("pid").first()
    last = d.groupby("pid").last()
    pnl = last["pos_u"] + last["fwd_cap"]                 # 상한 보유 (근사: 2차항 무시)
    pnl.loc[first.index] = first["pos_u"]
    return float(pnl.mean() * 1e4)


def main() -> int:
    ck = pd.read_parquet(D / "checkpoints.parquet"); bars = pd.read_parquet(D / "bars.parquet")
    man = json.loads((D / "manifest.json").read_text())
    fcols = man["feature_cols"]; pcols = EF.POS_COLS
    X_dash = bars[fcols].to_numpy(np.float32)[ck["t"].to_numpy()]
    df = pd.concat([ck.reset_index(drop=True), pd.DataFrame(X_dash, columns=fcols)], axis=1)
    del X_dash
    ts = pd.to_datetime(df["ts"])
    win = {k: ((ts >= a) & (ts <= b)).to_numpy() for k, (a, b) in SPLITS.items()}
    log(f"rows {len(df)} · " + " · ".join(f"{k} {int(v.sum())}" for k, v in win.items()))
    y = df["y_exit"].to_numpy(); pid = df["pid"].to_numpy()
    arms = {"A": pcols, "B": pcols + fcols}
    P = {a: {s: np.zeros(len(df), np.float32) for s in SEEDS} for a in arms}
    auc = {a: {w: [] for w in SPLITS if w != "TRAIN"} for a in arms}
    tr = win["TRAIN"]
    for a, cols in arms.items():
        Xa = df[cols].to_numpy(np.float32)
        for s in SEEDS:
            t = time.time()
            m = HistGradientBoostingClassifier(random_state=s, **HGB).fit(Xa[tr], y[tr])
            P[a][s][~tr] = m.predict_proba(Xa[~tr])[:, 1]
            for w in auc[a]:
                auc[a][w].append(float(roc_auc_score(y[win[w]], P[a][s][win[w]])))
            log(f"arm {a} seed {s}: " + " ".join(f"{w} {auc[a][w][-1]:.4f}" for w in auc[a]) + f" · {time.time() - t:.0f}s")
            if a == "B" and s == SEEDS[0]:
                mB, XB = m, Xa
        del Xa
    ens = {a: np.mean([P[a][s] for s in SEEDS], axis=0) for a in arms}
    rep = {"seeds": SEEDS, "hgb": HGB, "splits": SPLITS, "n_rows": int(len(df)), "n_positions": int(df.pid.nunique()),
           "auc": {a: {w: {"seed_mean": float(np.mean(v)), "seeds": v, "ensemble": float(roc_auc_score(y[win[w]], ens[a][win[w]]))}
                       for w, v in auc[a].items()} for a in arms},
           "clock_only_auc": {w: float(roc_auc_score(y[win[w]], -df["pos_left"].to_numpy()[win[w]])) for w in auc["A"]},
           "delta_auc_B_minus_A": {}, "seed_sign_agree": {}, "policy": {}}
    for w in ("VAL", "OOS", "HOLDOUT"):
        lo, hi, mean = cluster_boot_delta_auc(y[win[w]], ens["A"][win[w]], ens["B"][win[w]], pid[win[w]])
        rep["delta_auc_B_minus_A"][w] = {"ci95": [lo, hi], "mean": mean}
        d = np.array(auc["B"][w]) - np.array(auc["A"][w])
        rep["seed_sign_agree"][w] = {"n_pos": int((d > 0).sum()), "n": len(d), "deltas": d.tolist()}
        log(f"ΔAUC {w}: mean {mean:+.4f} CI [{lo:+.4f}, {hi:+.4f}] · seeds {(d > 0).sum()}/{len(d)}")
    # --- 정책: τ 는 VAL 에서 청산 비율 격자 중 평균 bp 최대 (B 앙상블) ---
    rng = np.random.default_rng(1)
    pv = ens["B"][win["VAL"]]
    grid = []
    for r in EXIT_RATE_GRID:
        tau = float(np.quantile(pv, 1 - r))
        grid.append((policy_value(df[win["VAL"]], pv, tau), tau, r))
    best_val, tau, rate = max(grid)
    rep["policy"]["tau"] = tau; rep["policy"]["exit_rate_val"] = rate
    rep["policy"]["val_grid"] = [{"rate": r, "tau": t_, "bp": b} for b, t_, r in grid]
    for w in ("VAL", "OOS", "HOLDOUT"):
        dw = df[win[w]]
        rep["policy"][w] = {
            "B_model": policy_value(dw, ens["B"][win[w]], tau),
            "A_model": policy_value(dw, ens["A"][win[w]], float(np.quantile(ens["A"][win["VAL"]], 1 - rate))),
            "hold_to_cap": policy_value(dw, np.zeros(int(win[w].sum())), 1.0),
            "random_same_rate": float(np.mean([policy_value(dw, ens["B"][win[w]], tau, rng) for _ in range(20)])),
            "exit_rate": float((ens["B"][win[w]] >= tau).mean()),
        }
        log(f"policy {w}: " + " · ".join(f"{k} {v:+.2f}" if isinstance(v, float) and k != "exit_rate" else f"{k} {v:.3f}"
                                         for k, v in rep["policy"][w].items()))
    # --- 순열 중요도 (B, 첫 시드, OOS 서브샘플) ---
    oi = np.flatnonzero(win["OOS"]); sub = np.random.default_rng(2).choice(oi, min(60000, len(oi)), replace=False)
    imp = permutation_importance(mB, XB[sub], y[sub], scoring="roc_auc", n_repeats=3, random_state=0, n_jobs=6)
    order = np.argsort(-imp.importances_mean)[:20]
    rep["perm_importance_top20"] = [{"feature": arms["B"][i], "auc_drop": float(imp.importances_mean[i])} for i in order]
    # --- 사전등록 게이트 ---
    g1 = all(rep["delta_auc_B_minus_A"][w]["ci95"][0] > 0 for w in ("VAL", "OOS"))
    g2 = all(rep["seed_sign_agree"][w]["n_pos"] == len(SEEDS) for w in ("VAL", "OOS"))
    g3 = all(rep["policy"][w]["B_model"] > max(rep["policy"][w]["hold_to_cap"], rep["policy"][w]["random_same_rate"])
             for w in ("VAL", "OOS"))
    rep["gate"] = {"G1_delta_auc_ci": g1, "G2_seed_sign": g2, "G3_policy_beats_hold_and_random": g3, "gate_pass": g1 and g2 and g3}
    rep["created_utc"] = datetime.now(timezone.utc).isoformat()
    (D / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"gate: {rep['gate']}")
    # --- 동결: B 5시드 (라이브는 평균) ---
    ART.mkdir(parents=True, exist_ok=True)
    XB_all = df[arms["B"]].to_numpy(np.float32)
    models = [HistGradientBoostingClassifier(random_state=s, **HGB).fit(XB_all[tr], y[tr]) for s in SEEDS]
    joblib.dump(models, ART / "model.joblib")
    meta = {"rule_id": "eth_exit_synth_hgb_cap24_20260910", "created_utc": rep["created_utc"], "cap_bars": man["cap"],
            "feature_cols": arms["B"], "pos_cols": pcols, "dash_cols": fcols, "seeds": SEEDS,
            "tau": tau, "tau_partial": float(np.quantile(pv, 1 - min(1.0, 2 * rate))), "exit_rate_val": rate,
            "auc": {a: {w: rep["auc"][a][w]["ensemble"] for w in rep["auc"][a]} for a in arms},
            "delta_auc_B_minus_A": rep["delta_auc_B_minus_A"], "policy": {w: rep["policy"][w] for w in ("VAL", "OOS", "HOLDOUT")},
            "gate": rep["gate"], "train_span": [str(ts[tr].min()), str(ts[tr].max())], "excluded_inputs": man["excluded"],
            "label": "y_exit = side·(close[t0+CAP]−close[t]) < 0, entries every 4th bar both sides, checkpoints odd k<24"}
    (ART / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))
    log(f"frozen → {ART}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
