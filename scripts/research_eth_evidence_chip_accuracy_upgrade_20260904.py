#!/usr/bin/env python3
"""증거신호 칩 **정확도 업그레이드** -- 인과 발동 모집단 재학습 + 레짐·청산맵 피쳐, 배포 컨텍스트와 대조 (2026-09-04).

사용자 정정: "대시보드 정보 칩의 정확도와 경제성을 끌어올리는 업그레이드". 이 스크립트는 정확도 축이다.

## 왜 이 순서인가
배포된 8종 칩의 TabPFN 컨텍스트는 **클러스터 앵커 봉**으로 학습됐는데 라이브는 **raw 단일봉 발동**에서 호출한다
(live_evidence_signal_metalabel_20260829 docstring). 인과 모집단(raw 발동, 09-02 prep) 실측은 배포 표기와 다르다
(예: demarker VAL 0.5255 vs 배포 0.7527). 즉 첫 번째 정확도 업그레이드는 **라이브가 실제로 보는 모집단으로 다시 학습해
확률을 정직하게 만드는 것**이고, 두 번째가 레짐(S12_K3 OOF)·청산맵 거리(원시 %, ATR 교란 통제) 피쳐다.

## 설계 (결과 보기 전 고정)
  모집단  `tmp/eth_causal_population_metalabel_20260902/<sig>_causal_fires.csv` (raw 발동, 신호별 K/H 라벨 `hit`, Tier0 23)
  분할    TRAIN <2025-09-01 / VAL <2026-01-01 / OOS <2026-04-01. **HOLDOUT 미접촉.**
  팔      D 배포 컨텍스트(앵커 학습, 그대로 적합) → 인과 VAL/OOS 채점 (기준선, "지금 칩이 보여주는 확률의 품질")
          F0 Tier0(인과 재학습) · F1 +레짐 one-hot(ETH S12_K3 확장 OOF) · F2 +청산맵(지지/저항 최근접 거리 원시 %·가중치·동측 거리)
          · F3 둘 다. 청산맵 거리는 ATR 단위도 같이 넣되 원시 %가 주 피쳐(ATR 교란 실측 후 결정).
  학습기  TabPFN(서버 GPU), TRAIN 인과 발동 전부(≤18k) 컨텍스트, **5시드**(random_state). 로컬은 HGB 프록시.
  지표    VAL/OOS AUC(5시드 평균±sd), Brier, 상위 30% 적중률·리프트, 캘리브레이션 기울기(logit(hit)~logit(p)).
  판정    칩 교체 후보 = 팔 X가 D 대비 VAL AUC +0.01 이상 **그리고** OOS AUC 비열화 **그리고** 5시드 중 4시드 이상 OOS 개선.
          (신호 8 × 팔 4 = 32 시행이므로 두 창 동시 요구가 다중성 방어다.) 경제성 주장은 여기서 하지 않는다.
  부수    레짐별·청산맵 삼분위별 조건부 적중률(VAL+OOS, 기술통계).

사용:  --stage build  (CPU) 프레임 생성 -> tmp/eth_chip_accuracy_upgrade_20260904/frames/<sig>.parquet
       --stage eval --learner tabpfn|hgb  -> report_<learner>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

POP = ROOT / "tmp/eth_causal_population_metalabel_20260902"
REG = ROOT / "tmp/eth_entry_oof_regime_20260903/regime_oof_eth.parquet"
LIQ = ROOT / "tmp/eth_fire_cont_liqmap_20260904/hourly_levels.parquet"
OUT = ROOT / "tmp/eth_chip_accuracy_upgrade_20260904"
SIGNALS = ["taker_delta_z_climax", "short_term_return_z", "liquidity_sweep", "orthogonal_combo",
           "smt_divergence", "fib_extension_exhaustion", "demarker_extreme", "kalman_deviation_meanrev"]
SEEDS = [20260829, 141592, 271828, 577215, 20260904]
TOP_FRAC = 0.30
REG_COLS = ["reg_bull", "reg_bear", "reg_chop"]
LIQ_COLS = ["liq_d_same_pct", "liq_d_opp_pct", "liq_w_same", "liq_w_opp", "liq_d_same_atr"]


def log(m): print(f"[chipacc] {m}", flush=True)


def tier0_cols(d, signal=None):
    """Tier0 23(is_bottom 제외 22) + 신호 고유 24번째 피쳐(demarker: dem, kalman: kalman_dev_z) -- 라이브 METALABEL_SIGNALS와 동일."""
    skip = {"pos", "timestamp", "side", "move_atr_mult", "hit", "split", "is_bottom", "dem", "kalman_dev_z", "sup_dist_pct", "sup_w", "res_dist_pct", "res_w", "regime_code"}
    cols = [c for c in d.columns if c not in skip and not c.startswith(("reg_", "liq_", "regime_"))]
    if signal == "demarker_extreme":
        cols = cols + ["dem"]
    if signal == "kalman_deviation_meanrev":
        cols = cols + ["kalman_dev_z"]
    return cols


# ----------------------------------------------------------------------------- build
def stage_build():
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "frames").mkdir(exist_ok=True)
    reg = pd.read_parquet(REG); reg["timestamp"] = pd.to_datetime(reg["timestamp"])
    # demarker/kalman 칩의 24번째 피쳐(dem / kalman_dev_z)는 인과 프레임에 없다 -- 라이브와 같은 함수로 klines에서 계산해 timestamp로 붙인다.
    from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker
    from research_eth_candidate_pool_raw_lift_check_20260831 import kalman_level_and_velocity, rolling_zscore
    kl = pd.read_csv(ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv", usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    extra = pd.DataFrame({"timestamp": kl["timestamp"], "dem": compute_demarker(kl["high"], kl["low"]).to_numpy()})
    levels, _ = kalman_level_and_velocity(kl["close"].to_numpy()); extra["kalman_dev_z"] = rolling_zscore(pd.Series((kl["close"].to_numpy() - levels) / levels)).to_numpy()
    liq = pd.read_parquet(LIQ).sort_values("timestamp"); liq["timestamp"] = pd.to_datetime(liq["timestamp"])
    summary = {}
    for s in SIGNALS:
        d = pd.read_csv(POP / f"{s}_causal_fires.csv", parse_dates=["timestamp"])
        d = d.loc[d["split"] != "HOLDOUT"].copy()                       # HOLDOUT 행 자체를 버린다
        d["timestamp"] = d["timestamp"].dt.tz_localize(None) if d["timestamp"].dt.tz is not None else d["timestamp"]
        d = d.merge(reg.rename(columns={"regime_oof": "regime_code", "oof_source": "regime_oof_src"}), on="timestamp", how="left")
        d = d.merge(extra, on="timestamp", how="left")
        code = d["regime_code"].fillna(-1).astype(int)
        d["reg_bull"] = (code == 0).astype(np.int8); d["reg_bear"] = (code == 1).astype(np.int8); d["reg_chop"] = (code == 2).astype(np.int8)
        d = d.sort_values("timestamp")
        d = pd.merge_asof(d, liq[["timestamp", "sup_dist_pct", "sup_w", "res_dist_pct", "res_w"]], on="timestamp", direction="backward")
        sup = d["sup_dist_pct"].abs().fillna(5.0); res = d["res_dist_pct"].abs().fillna(5.0)
        bottom = d["is_bottom"].to_numpy() == 1
        d["liq_d_same_pct"] = np.where(bottom, sup, res); d["liq_d_opp_pct"] = np.where(bottom, res, sup)
        d["liq_w_same"] = np.where(bottom, d["sup_w"].fillna(0), d["res_w"].fillna(0)); d["liq_w_opp"] = np.where(bottom, d["res_w"].fillna(0), d["sup_w"].fillna(0))
        d["liq_d_same_atr"] = d["liq_d_same_pct"] / 100.0 / d["atr_pct"].replace(0, np.nan)
        d = d.sort_values("pos").reset_index(drop=True)
        n_bad = int((d.loc[d["split"] == "TRAIN", "regime_oof_src"].fillna("").astype(str).str.startswith("final")).sum())
        d.to_parquet(OUT / "frames" / f"{s}.parquet", index=False)
        summary[s] = {"n": int(len(d)), "splits": d["split"].value_counts().to_dict(), "hit_rate": round(float(d["hit"].mean()), 4),
                      "regime_final_in_train": n_bad, "regime_missing": int((code == -1).sum()), "liq_missing": int(d["sup_dist_pct"].isna().sum())}
        log(f"{s:26s} n {len(d):6,} {summary[s]['splits']} hit {summary[s]['hit_rate']:.3f} regime -1 {summary[s]['regime_missing']} final-in-TRAIN {n_bad}")
        assert n_bad == 0, "레짐 OOF 'final' 출처가 TRAIN에 섞임"
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


# ----------------------------------------------------------------------------- eval helpers
def metrics(y, p):
    from sklearn.metrics import roc_auc_score, brier_score_loss
    y = np.asarray(y, int); p = np.asarray(p, float); out = {"n": int(len(y)), "base": round(float(y.mean()), 4)}
    if len(np.unique(y)) < 2 or len(y) < 30:
        return {**out, "auc": None}
    out["auc"] = round(float(roc_auc_score(y, p)), 4); out["brier"] = round(float(brier_score_loss(y, p)), 4)
    k = max(int(len(y) * TOP_FRAC), 10); top = np.argsort(-p)[:k]
    out["top30_hit"] = round(float(y[top].mean()), 4); out["top30_lift"] = round(float(y[top].mean() / max(y.mean(), 1e-9)), 3)
    # 캘리브레이션 기울기: logit(hit) ~ a + b*logit(p)  (b≈1이면 정직, <1이면 과신)
    pc = np.clip(p, 1e-4, 1 - 1e-4); z = np.log(pc / (1 - pc))
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6).fit(z.reshape(-1, 1), y); out["calib_slope"] = round(float(lr.coef_[0][0]), 3); out["calib_intercept"] = round(float(lr.intercept_[0]), 3)
    except Exception:  # noqa: BLE001
        out["calib_slope"] = None
    return out


def fit_predict(learner, Xtr, ytr, Xs: dict, seed):
    if learner == "tabpfn":
        from tabpfn import TabPFNClassifier
        clf = TabPFNClassifier(device="cuda", random_state=int(seed), ignore_pretraining_limits=True).fit(Xtr, ytr)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, random_state=int(seed)).fit(Xtr, ytr)
    return {w: clf.predict_proba(X)[:, 1] for w, X in Xs.items()}


def deployed_context(s):
    """배포 칩의 동결 컨텍스트 (live_evidence_signal_metalabel_20260829.METALABEL_SIGNALS) -> (X, y, feature_cols)."""
    from live_evidence_signal_metalabel_20260829 import METALABEL_SIGNALS, FEATURE_COLUMNS
    cfg = METALABEL_SIGNALS[s]; path = Path(cfg["train_context"])
    if not path.exists():
        return None
    ctx = pd.read_csv(path); cols = list(cfg.get("feature_columns", FEATURE_COLUMNS))
    ycol = "label" if "label" in ctx.columns else "hit"
    return ctx, cols, ycol


def stage_eval(learner):
    t0 = time.time(); rng = np.random.default_rng(20260904)
    report = {"learner": learner, "seeds": SEEDS, "top_frac": TOP_FRAC, "holdout_touched": False, "signals": {}}
    for s in SIGNALS:
        d = pd.read_parquet(OUT / "frames" / f"{s}.parquet")
        T0 = tier0_cols(d, s) + ["is_bottom"]
        arms = {"F0": T0, "F1": T0 + REG_COLS, "F2": T0 + LIQ_COLS, "F3": T0 + REG_COLS + LIQ_COLS}
        tr = d.loc[d["split"] == "TRAIN"]; S = {w: d.loc[d["split"] == w] for w in ("VAL", "OOS")}
        R = {"n": {w: int(len(S[w])) for w in S}, "n_train": int(len(tr)), "arms": {}}
        # D: 배포 컨텍스트 (앵커 학습) -> 인과 VAL/OOS 채점. 배포 칩의 피쳐 열(dem/kalman_dev_z)이 인과 프레임에 없으면 건너뜀.
        dep = deployed_context(s)
        if dep is not None:
            ctx, cols, ycol = dep
            if all(c in d.columns for c in cols):
                per = {w: [] for w in S}
                for sd in SEEDS[:3] if learner == "tabpfn" else SEEDS[:1]:
                    P = fit_predict(learner, ctx[cols].to_numpy(float), ctx[ycol].to_numpy(int), {w: S[w][cols].to_numpy(float) for w in S}, sd)
                    for w in S:
                        per[w].append(P[w])
                R["arms"]["D_deployed_ctx"] = {w: metrics(S[w]["hit"], np.mean(per[w], axis=0)) for w in S}
                R["arms"]["D_deployed_ctx"]["ctx_rows"] = int(len(ctx)); R["arms"]["D_deployed_ctx"]["missing_cols"] = []
            else:
                R["arms"]["D_deployed_ctx"] = {"skipped": "feature cols not in causal frame", "missing_cols": [c for c in cols if c not in d.columns]}
        for arm, cols in arms.items():
            Xtr = tr[cols].to_numpy(float); ytr = tr["hit"].to_numpy(int)
            m = np.isfinite(Xtr).all(axis=1); Xtr, ytr = Xtr[m], ytr[m]
            per = {w: [] for w in S}; per_seed_auc = {w: [] for w in S}
            for sd in SEEDS:
                Xs = {}
                for w in S:
                    X = S[w][cols].to_numpy(float); X = np.where(np.isfinite(X), X, np.nan_to_num(X, nan=0.0)); Xs[w] = X
                P = fit_predict(learner, Xtr, ytr, Xs, sd)
                for w in S:
                    per[w].append(P[w]); mm = metrics(S[w]["hit"], P[w]); per_seed_auc[w].append(mm.get("auc"))
            R["arms"][arm] = {w: {**metrics(S[w]["hit"], np.mean(per[w], axis=0)), "auc_seed_mean": round(float(np.nanmean([a for a in per_seed_auc[w] if a is not None])), 4),
                                  "auc_seed_sd": round(float(np.nanstd([a for a in per_seed_auc[w] if a is not None])), 4), "auc_per_seed": per_seed_auc[w]} for w in S}
            if arm == "F0":
                p_all = {w: np.mean(per[w], axis=0) for w in S}
        # 조건부 적중률 (VAL+OOS): 레짐 / 청산맵 동측 거리 삼분위(TRAIN 컷, 원시 %)
        vo = pd.concat([S["VAL"], S["OOS"]]); vo = vo.assign(p_f0=np.r_[p_all["VAL"], p_all["OOS"]])
        code = np.select([vo["reg_bull"] == 1, vo["reg_bear"] == 1, vo["reg_chop"] == 1], ["bull", "bear", "chop"], "none")
        cond = {"regime": {}, "liq_same_tertile_pct": {}}
        for r_ in ("bull", "bear", "chop"):
            m = code == r_
            if m.sum() >= 30:
                cond["regime"][r_] = {"n": int(m.sum()), "hit": round(float(vo.loc[m, "hit"].mean()), 4), "hit_top30_f0": round(float(vo.loc[m].nlargest(max(int(m.sum() * TOP_FRAC), 10), "p_f0")["hit"].mean()), 4)}
        _, e = pd.qcut(tr["liq_d_same_pct"], 3, retbins=True, duplicates="drop"); e = np.r_[-np.inf, e[1:-1], np.inf]
        tert = pd.cut(vo["liq_d_same_pct"], bins=e, labels=["near", "mid", "far"][: len(e) - 1]).astype(str)
        for t_ in ("near", "mid", "far"):
            m = (tert == t_).to_numpy()
            if m.sum() >= 30:
                cond["liq_same_tertile_pct"][t_] = {"n": int(m.sum()), "hit": round(float(vo.loc[m, "hit"].mean()), 4)}
        cond["liq_tertile_edges_pct"] = [round(float(x), 3) for x in e[1:-1]]
        R["conditional_hit_val_oos"] = cond
        # 판정
        D = R["arms"].get("D_deployed_ctx", {}); dv, do = (D.get("VAL") or {}).get("auc"), (D.get("OOS") or {}).get("auc")
        R["candidates"] = {}
        for arm in arms:
            a = R["arms"][arm]; av, ao = a["VAL"]["auc"], a["OOS"]["auc"]
            per_seed_gain = sum(1 for x in a["OOS"]["auc_per_seed"] if x is not None and do is not None and x >= do)
            ok = (dv is not None and av is not None and av >= dv + 0.01 and ao is not None and do is not None and ao >= do and per_seed_gain >= 4)
            R["candidates"][arm] = {"pass": bool(ok), "d_val_auc": round(av - dv, 4) if (av is not None and dv is not None) else None,
                                    "d_oos_auc": round(ao - do, 4) if (ao is not None and do is not None) else None, "seeds_oos_ge_deployed": per_seed_gain}
        report["signals"][s] = R
        log(f"{s:26s} D {dv}/{do} | " + " ".join(f"{a} {R['arms'][a]['VAL']['auc']}/{R['arms'][a]['OOS']['auc']}" for a in arms) + f"  ({time.time()-t0:.0f}s)")
    (OUT / f"report_{learner}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"\n{'signal':>26s} {'nV/nO':>11s} {'D val/oos':>13s} {'F0':>13s} {'F1 +reg':>13s} {'F2 +liq':>13s} {'F3 both':>13s} {'best pass':>10s}")
    for s, R in report["signals"].items():
        D = R["arms"].get("D_deployed_ctx", {}); dtxt = f"{(D.get('VAL') or {}).get('auc')}/{(D.get('OOS') or {}).get('auc')}"
        arms_txt = " ".join(f"{R['arms'][a]['VAL']['auc']:.3f}/{R['arms'][a]['OOS']['auc']:.3f}".rjust(13) for a in ("F0", "F1", "F2", "F3"))
        passed = [a for a, c in R["candidates"].items() if c["pass"]]
        print(f"{s:>26s} {R['n']['VAL']:>5d}/{R['n']['OOS']:<5d} {dtxt:>13s} {arms_txt} {','.join(passed) or '-':>10s}")
    print("\n[conditional hit VAL+OOS]")
    for s, R in report["signals"].items():
        print(f"  {s:>26s} regime {R['conditional_hit_val_oos']['regime']} | liq {R['conditional_hit_val_oos']['liq_same_tertile_pct']}")
    log(f"eval 완료 -> {OUT/f'report_{learner}.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["build", "eval"], required=True); ap.add_argument("--learner", choices=["tabpfn", "hgb"], default="hgb")
    a = ap.parse_args(); stage_build() if a.stage == "build" else stage_eval(a.learner)
