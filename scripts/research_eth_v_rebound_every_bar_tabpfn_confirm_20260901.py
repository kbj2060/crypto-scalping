#!/usr/bin/env python3
"""매 봉 스코어링 재설계 -- TabPFN 확인 (GBM 프록시 결과 재현 여부 + 컨텍스트 샘플링 결정).

## 이 스크립트가 답하는 두 질문

1. **TabPFN이 GBM 프록시 결과를 재현하는가?** 전체 봉 학습 시 GBM은 VAL 전체봉 AUC 0.6953을
   냈고, 이 값이 held_up 층화 추정치(0.66~0.69)와 독립 수렴했다. 이 저장소는 GBM과 TabPFN이
   갈린 전례가 있으므로(V_REBOUND 자신이 TabPFN에서 GBM을 +0.020/+0.014로 이겼고, 돌파지속은
   "GBM 약함이 TabPFN 사망을 증명 안 함") 실제 모델로 확인해야 한다. **재현 안 되면 매 봉
   설계 전체가 무효다 -- 이게 관문이다.**

2. **컨텍스트를 어떻게 뽑을 것인가?** 전체 봉 TRAIN은 182,969행으로 TabPFN 컨텍스트에 다 못
   들어간다(17,969행이 이미 라이브 사이클 6.6초, 전체는 60초 캐시를 넘길 것). 크기 x 샘플링
   방식을 함께 재서 실현 가능한 최선을 찾는다.

   ⚠️2026-09-01 GBM 실측 교훈: **event-first 샘플링은 쓰지 말 것**(사건당 첫 봉만 뽑으면
   가장 약하고 비대표적인 양성만 남아 -0.097). 무작위/균등이 낫다. 여기서는
   - `random`: 라벨비율 보존 없이 단순 무작위(자연 비율이 그대로 따라옴)
   - `stratified`: 전체 TRAIN의 자연 라벨비율을 정확히 맞춰 층화추출
   두 가지를 비교한다.

## 비교 기준선

같은 평가셋(VAL 전체봉)에서:
  - GBM 전체봉 학습(182,969행): AUC 0.6953 <- 재현 대상
  - GBM 후보풀 학습을 전체봉에 적용: 0.5287 (붕괴, kept-only 착시)

⚠️ VAL만 사용. OOS/HOLDOUT 미터치. 라이브 코드 변경 없음.

Run on the server (GPU) via handoff:
  handoff.sh launch server <job> -- python scripts/research_eth_v_rebound_every_bar_tabpfn_confirm_20260901.py
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
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FEAS = ROOT / "scripts/research_eth_v_rebound_every_bar_scoring_feasibility_20260901.py"
_spec = importlib.util.spec_from_file_location("everybar_feas_tabpfn", FEAS)
_feas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_feas)

OUT_JSON = ROOT / "data/research/eth_v_rebound_every_bar_tabpfn_20260901/report.json"

FEATURE_COLUMNS = _feas.FEATURE_COLUMNS
SEEDS = [20260829, 141592, 271828]
CONTEXT_SIZES = [6000, 12000, 18000]
SAMPLINGS = ["random", "stratified"]
EVAL_CAP = 15000  # VAL 전체봉 37k를 다 예측하면 느려서 상한을 둔다(무작위 부분추출, 시드 고정)

GBM_REFERENCE = {"all_bars_train_on_all_bars": 0.6953, "all_bars_train_on_candidates": 0.5287}


def log(msg: str) -> None:
    print(f"[everybar_tabpfn] {msg}", flush=True)


def sample_context(tr: pd.DataFrame, n: int, how: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if how == "random":
        idx = rng.choice(len(tr), size=min(n, len(tr)), replace=False)
        return tr.iloc[np.sort(idx)]
    # stratified: 전체 TRAIN의 자연 라벨비율을 정확히 재현
    y = tr["label"].to_numpy()
    pos_idx, neg_idx = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = int(round(n * (len(pos_idx) / len(y))))
    n_pos = max(1, min(n_pos, len(pos_idx)))
    n_neg = min(n - n_pos, len(neg_idx))
    take = np.concatenate([rng.choice(pos_idx, n_pos, replace=False),
                            rng.choice(neg_idx, n_neg, replace=False)])
    return tr.iloc[np.sort(take)]


def main() -> int:
    t0 = time.time()
    from sklearn.metrics import roc_auc_score
    from tabpfn import TabPFNClassifier
    import torch
    log(f"cuda: {torch.cuda.is_available()}")

    log("building all-bar long frame (features + V0 labels)...")
    long = _feas.build_long_frame()
    long = long.loc[long["label"].notna()].dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    tr = long.loc[long["split"] == "TRAIN"].reset_index(drop=True)
    va = long.loc[long["split"] == "VAL"].reset_index(drop=True)
    log(f"TRAIN {len(tr)} (base={tr['label'].mean():.4f})  VAL {len(va)} (base={va['label'].mean():.4f})")

    rng = np.random.default_rng(20260901)
    if len(va) > EVAL_CAP:
        keep = np.sort(rng.choice(len(va), EVAL_CAP, replace=False))
        va_eval = va.iloc[keep]
        log(f"VAL 평가 부분추출: {len(va)} -> {len(va_eval)} (base={va_eval['label'].mean():.4f})")
    else:
        va_eval = va
    Xva = va_eval[FEATURE_COLUMNS].to_numpy(dtype=float)
    yva = va_eval["label"].to_numpy()
    one_row = Xva[:1]

    results = {}
    for how in SAMPLINGS:
        for size in CONTEXT_SIZES:
            key = f"{how}_{size}"
            aucs, cyc, ratios = [], [], []
            for sd in SEEDS:
                ctx = sample_context(tr, size, how, sd)
                ratios.append(float(ctx["label"].mean()))
                Xtr, ytr = ctx[FEATURE_COLUMNS].to_numpy(dtype=float), ctx["label"].to_numpy()
                clf = TabPFNClassifier(device="cuda", random_state=sd, ignore_pretraining_limits=True)
                t = time.time(); clf.fit(Xtr, ytr); t_fit = time.time() - t
                t = time.time(); clf.predict_proba(one_row); t_pred1 = time.time() - t
                aucs.append(roc_auc_score(yva, clf.predict_proba(Xva)[:, 1]))
                cyc.append(t_fit + t_pred1)
            a = np.array(aucs)
            results[key] = {
                "sampling": how, "context_size": size,
                "auc_mean": round(float(a.mean()), 4), "auc_std": round(float(a.std()), 4),
                "label_ratio_mean": round(float(np.mean(ratios)), 4),
                "live_cycle_sec_mean": round(float(np.mean(cyc)), 2),
                "delta_vs_gbm_all_bars": round(float(a.mean()) - GBM_REFERENCE["all_bars_train_on_all_bars"], 4),
            }
            r = results[key]
            log(f"  {key:18s} AUC {r['auc_mean']:.4f}±{r['auc_std']:.4f}  "
                f"(GBM 0.6953 대비 {r['delta_vs_gbm_all_bars']:+.4f})  "
                f"라이브사이클 {r['live_cycle_sec_mean']:.2f}s  라벨비율 {r['label_ratio_mean']:.4f}")

    best = max(results.values(), key=lambda r: r["auc_mean"])
    log("")
    log(f"=== 최고: {best['sampling']}_{best['context_size']} AUC {best['auc_mean']:.4f} "
        f"(GBM 프록시 0.6953 대비 {best['delta_vs_gbm_all_bars']:+.4f}) ===")
    log(f"    후보풀 학습을 전체봉에 적용했을 때(0.5287) 대비: {best['auc_mean'] - GBM_REFERENCE['all_bars_train_on_candidates']:+.4f}")

    report = {
        "signal": "v_rebound_every_bar_tabpfn_confirm", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope": {"model": "TabPFN (배포판과 동일 계열)", "population": "ALL bars (every-bar scoring)",
                  "holdout_touched": False, "oos_touched": False, "live_code_changed": False,
                  "eval_subsample_cap": EVAL_CAP,
                  "purpose": "GBM 프록시(전체봉 0.6953)를 TabPFN이 재현하는지 + 컨텍스트 샘플링 결정"},
        "gbm_reference": GBM_REFERENCE, "seeds": SEEDS,
        "context_sizes": CONTEXT_SIZES, "samplings": SAMPLINGS,
        "train_n": int(len(tr)), "val_n": int(len(va)), "val_eval_n": int(len(va_eval)),
        "results": results, "best": best,
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
