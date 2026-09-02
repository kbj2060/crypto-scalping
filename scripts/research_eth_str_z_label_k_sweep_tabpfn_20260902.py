#!/usr/bin/env python3
"""라벨 K를 올려 재학습한 증거신호는 어떤가 -- short_term_return_z, K 스윕.
2026-09-02 사용자: "학습 데이터부터 이미 k를 늘린 라벨을 학습한 증거신호가 궁금한거야".

바꾸는 것은 **라벨 하나뿐**이다. 현행 라벨은
    hit = (12봉 내 intrabar MFE_pct) >= ATR_HIT_MULT * atr_pct,  ATR_HIT_MULT = 1.75
이고, 1.75는 "발동 집합에서 ~50/50 분할이 되도록" 캘리브레이션된 값이다 -- 즉 **K는 이 저장소에서
언제나 학습 균형/AUC 기준으로만 정해졌고 PnL 기준으로 정해진 적이 없다.** 그 공백을 메운다.

피처(Tier0 23개)와 발동 집합은 건드리지 않는다. MFE는 K와 무관하므로 저장된 fires에 대해
MFE를 한 번만 재계산하고 K별로 `hit`만 다시 만들어 TabPFN을 재학습한다 -- 피처 재빌드 불필요.

⚠️ HOLDOUT(>=2026-04-01)은 **평가하지 않는다**. 이 신호는 단일노출 홀드아웃을 이미 소진했다.
   VAL/OOS만 보고, 승격 주장이 아니라 "K를 키우면 학습된 신호가 어떻게 달라지는가"의 관측이다.

⚠️ K를 올리면 양성 비율이 떨어진다(실측 MFE 분포: x1 76.8% / x2 45.0% / x3 22.2% / x4 11.3%).
   AUC는 클래스 불균형에 비교적 강하지만 **AUC끼리의 비교는 라벨이 다르면 문제 난이도가 달라져
   무의미할 수 있다**(이 저장소 규칙: 서로 다른 라벨의 AUC 직접비교로 판단 금지). 그래서 판정은
   AUC가 아니라 **같은 잣대의 경제성**으로 한다 -- 각 K의 모델 점수 상위 분위로 진입을 골라,
   동일한 트레일링스톱 + 동일 비용(10bp) + 방향뒤집기 대조로 OOS PnL을 비교한다.
"""
from __future__ import annotations

import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
# ⚠️2026-09-02: core의 트레일링스톱 구현(_resolve_trade_trailing / arm_moves·trail_moves)은 이
# 저장소에서 **아직 커밋되지 않은 로컬 수정**이다(HEAD의 core에는 arm_moves가 0개). 그래서 서버에는
# 그 API가 없고, 경제성 평가를 서버에서 돌릴 수 없다. 파이프라인을 둘로 나눈다:
#   --stage train : 서버(GPU/TabPFN) -- 학습 + 시드평균 확률을 CSV로 저장
#   --stage econ  : dev(로컬)        -- 그 CSV를 읽어 동일 트레일링 + 방향뒤집기로 경제성 평가
import argparse  # noqa: E402
from core.causal_futures_backtest import purged_decision_mask  # noqa: E402
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS, SEEDS, evaluate,
)


def run_panel(train: pd.DataFrame, eval_df: pd.DataFrame, cols: list[str], tag: str) -> dict:
    """run_tabpfn_panel과 동일한 4-시드 절차. 원본은 집계 지표만 반환하는데 여기서는 경제성
    평가에 모델 점수가 필요하므로 시드 평균 확률도 함께 돌려준다(로직 변경 없음)."""
    from tabpfn import TabPFNClassifier
    rows, probas = [], []
    y = eval_df["hit"].to_numpy().astype(int)
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[cols], train["hit"].to_numpy().astype(int))
        pr = clf.predict_proba(eval_df[cols])[:, 1]
        probas.append(pr)
        r = evaluate(pr, y); r["seed"] = seed; rows.append(r)
        log(f"  [{tag}] seed={seed}: auc={r['auc']:.4f} bal_acc={r['balanced_accuracy']:.4f}")
    tb = pd.DataFrame(rows)
    return {"auc_mean": round(float(tb["auc"].mean()), 4),
            "auc_std": round(float(tb["auc"].std(ddof=1)), 4),
            "balanced_accuracy_mean": round(float(tb["balanced_accuracy"].mean()), 4),
            "proba_mean": np.mean(probas, axis=0)}

FIRES = ROOT / "data/labels/eth_5m_short_term_return_z_metalabel_20260829/eth_5m_short_term_return_z_metalabel_features.csv"
KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
HORIZON = 12
K_GRID = [1.75, 2.5, 3.0, 4.0]          # 1.75 = 현행 배포 라벨
VAL_START, OOS_START, HOLDOUT_START = (pd.Timestamp(x) for x in ("2025-09-01", "2026-01-01", "2026-04-01"))
SEL_QUANTILES = [0.0, 0.3, 0.5, 0.7]     # 모델 점수 상위 (1-q) 만 진입
TRAIL = (3.0, 1.0, 0.05)                  # 현행 str_z 최적 청산 -- K 비교 내내 고정
MARGIN, LEV, COST = 0.30, 3.0, 0.001
OUT = ROOT / "tmp/eth_str_z_label_k_sweep_20260902"


def log(m): print(f"[k_sweep] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["train", "econ"], required=True)
    args = ap.parse_args()
    if args.stage == "econ":
        return stage_econ()
    f = pd.read_csv(FIRES, parse_dates=["timestamp"])
    kl = pd.read_csv(KLINES, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    hi, lo, cl = kl["high"].to_numpy(), kl["low"].to_numpy(), kl["close"].to_numpy()
    o = kl["open"].to_numpy(); ts = kl["timestamp"]

    pos = f["pos"].to_numpy(); is_bot = (f["side"] == "bottom").to_numpy()
    ok = pos + HORIZON < len(cl)
    f = f.loc[ok].reset_index(drop=True); pos, is_bot = pos[ok], is_bot[ok]
    mfe = np.empty(len(f))
    for i, (p_, b_) in enumerate(zip(pos, is_bot)):
        w = slice(p_ + 1, p_ + 1 + HORIZON); e = cl[p_]
        mfe[i] = (hi[w].max() / e - 1.0) if b_ else (1.0 - lo[w].min() / e)
    f["mfe_pct"] = mfe
    log(f"fires {len(f)}  (MFE 재계산 완료)")

    t = f["timestamp"]
    m_tr = (t < VAL_START).to_numpy()
    m_va = ((t >= VAL_START) & (t < OOS_START)).to_numpy()
    m_oo = ((t >= OOS_START) & (t < HOLDOUT_START)).to_numpy()
    log(f"TRAIN {m_tr.sum()} / VAL {m_va.sum()} / OOS {m_oo.sum()}  (HOLDOUT 미평가)")

    ev = purged_decision_mask(ts, start=VAL_START, end=OOS_START, horizon_bars=HORIZON)
    eo = purged_decision_mask(ts, start=OOS_START, end=HOLDOUT_START, horizon_bars=HORIZON)
    vset, oset = set(np.flatnonzero(ev).tolist()), set(np.flatnonzero(eo).tolist())

    def econ(sub: pd.DataFrame, which: str):
        dec = sub["pos"].to_numpy(np.int64)
        sc = np.where(sub["side"].to_numpy() == "bottom", 1.0, -1.0)
        atr = sub["atr_pct"].to_numpy(float)
        keep = np.array([d in (vset if which == "VAL" else oset) for d in dec])
        if keep.sum() < 30: return None
        sl, arm, tr = TRAIL
        out = {}
        for sgn, lb in ((1.0, "real"), (-1.0, "flip")):
            r = simulate_single_position(
                timestamps=ts, open_px=o, high=hi, low=lo, close=cl,
                decision_indices=dec[keep], scores=(sc * sgn)[keep],
                tp_moves=np.full(int(keep.sum()), 999.0), sl_moves=(sl * atr)[keep],
                upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=HORIZON,
                margin_fraction=MARGIN, leverage=LEV, roundtrip_cost_rate=COST,
                arm_moves=(arm * atr)[keep], trail_moves=(tr * atr)[keep]).ledger
            v = r["trade_return"].to_numpy()
            w_, l_ = v[v > 0].sum(), -v[v < 0].sum()
            out[lb] = {"n": len(v), "mean_bp": float(v.mean() * 1e4), "total_bp": float(v.sum() * 1e4),
                       "pf": float(w_ / l_) if l_ > 0 else float("inf")}
        return out

    results = {}
    for K in K_GRID:
        f["hit"] = (f["mfe_pct"] >= K * f["atr_pct"]).astype(int)
        rate = float(f.loc[m_tr, "hit"].mean())
        log(f"\n=== K={K} (TRAIN 양성률 {rate:.3f}) ===")
        if rate < 0.05 or rate > 0.95:
            log("  클래스 극단 불균형 -- 스킵"); continue
        train = f.loc[m_tr].reset_index(drop=True)
        val, oos = f.loc[m_va].reset_index(drop=True), f.loc[m_oo].reset_index(drop=True)
        rv = run_panel(train, val, FEATURE_COLUMNS, "VAL")
        ro = run_panel(train, oos, FEATURE_COLUMNS, "OOS")
        log(f"  AUC  VAL {rv['auc_mean']:.4f}±{rv['auc_std']:.4f}   OOS {ro['auc_mean']:.4f}±{ro['auc_std']:.4f}"
            f"   (⚠️K가 다르면 문제 난이도가 달라 AUC 직접비교는 무의미)")
        ent = {"train_hit_rate": rate, "val_auc": rv["auc_mean"], "oos_auc": ro["auc_mean"],
               "val_bal_acc": rv["balanced_accuracy_mean"], "oos_bal_acc": ro["balanced_accuracy_mean"]}
        for label, panel, mask in (("VAL", rv, m_va), ("OOS", ro, m_oo)):
            sub = f.loc[mask, ["pos", "timestamp", "side", "atr_pct"]].reset_index(drop=True).copy()
            sub["score"] = np.asarray(panel["proba_mean"], dtype=float)
            sub["K"], sub["window"] = K, label
            OUT.mkdir(parents=True, exist_ok=True)
            sub.to_csv(OUT / f"scores_K{K}_{label}.csv", index=False)
        results[str(K)] = ent

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "k_sweep_train.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float))
    log(f"\nWrote {OUT}/k_sweep.json")
    return 0


def stage_econ() -> int:
    """dev 전용: 서버가 저장한 시드평균 확률 CSV를 읽어 동일 청산·비용으로 경제성 비교."""
    from core.causal_futures_backtest import simulate_single_position
    kl = pd.read_csv(KLINES, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts = kl["timestamp"]; o = kl["open"].to_numpy()
    hi, lo, cl = kl["high"].to_numpy(), kl["low"].to_numpy(), kl["close"].to_numpy()
    ev = purged_decision_mask(ts, start=VAL_START, end=OOS_START, horizon_bars=HORIZON)
    eo = purged_decision_mask(ts, start=OOS_START, end=HOLDOUT_START, horizon_bars=HORIZON)
    sets = {"VAL": set(np.flatnonzero(ev).tolist()), "OOS": set(np.flatnonzero(eo).tolist())}
    sl, arm, tr = TRAIL
    log(f"{'K':>5s} {'win':>4s} {'상위':>5s} {'n':>5s} {'mean_bp':>8s} {'PF':>6s} {'total_bp':>10s} {'flip_gap':>10s}")
    for K in K_GRID:
        for win in ("VAL", "OOS"):
            fp = OUT / f"scores_K{K}_{win}.csv"
            if not fp.exists(): continue
            sub = pd.read_csv(fp, parse_dates=["timestamp"])
            for q in SEL_QUANTILES:
                thr = np.quantile(sub["score"], q) if q > 0 else -np.inf
                d = sub.loc[sub["score"] >= thr]
                dec = d["pos"].to_numpy(np.int64)
                sc0 = np.where(d["side"].to_numpy() == "bottom", 1.0, -1.0)
                atr = d["atr_pct"].to_numpy(float)
                keep = np.array([x in sets[win] for x in dec])
                if keep.sum() < 30: continue
                res = {}
                for sgn, lb in ((1.0, "real"), (-1.0, "flip")):
                    L = simulate_single_position(
                        timestamps=ts, open_px=o, high=hi, low=lo, close=cl,
                        decision_indices=dec[keep], scores=(sc0 * sgn)[keep],
                        tp_moves=np.full(int(keep.sum()), 999.0), sl_moves=(sl * atr)[keep],
                        upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=HORIZON,
                        margin_fraction=MARGIN, leverage=LEV, roundtrip_cost_rate=COST,
                        arm_moves=(arm * atr)[keep], trail_moves=(tr * atr)[keep]).ledger
                    v = L["trade_return"].to_numpy()
                    w_, l_ = v[v > 0].sum(), -v[v < 0].sum()
                    res[lb] = (len(v), v.mean() * 1e4, v.sum() * 1e4, (w_ / l_) if l_ > 0 else float("inf"))
                log(f"{K:5.2f} {win:>4s} {(1-q)*100:4.0f}% {res['real'][0]:5d} {res['real'][1]:+8.2f} "
                    f"{res['real'][3]:6.2f} {res['real'][2]:+10.1f} {res['real'][2]-res['flip'][2]:+10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
