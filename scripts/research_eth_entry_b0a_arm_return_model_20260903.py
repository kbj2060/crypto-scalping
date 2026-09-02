#!/usr/bin/env python3
"""B0-a: 팔별 수익 예측 모델 -- 진입 방향 모델의 첫 관문 (2026-09-03).

문제 형태 (2026-09-03 사용자와 확정):
  각 트리거는 팔 두 개를 낳는다 -- 신호방향 지정가(아래 매수/위 매도)와 역방향 지정가.
  둘 다 "3 ATR excursion 페이드"이고, **둘 다 그 자체로 수익**이다
  (신호방향 +14~23bp, 역방향 +7~13bp, 전 창).
  그래서 기준선은 "신호방향만"이 아니라 ⭐**양쪽 다 걸기**다 -- 그게 모델 없이
  오라클A(방향 완벽선택)를 이긴다(VAL +33.54 vs +26.69).
  트리거당 총 체결이 1.01~1.02건이라 주문은 둘이어도 포지션은 사실상 하나다.

  모델의 일: **음수가 예상되는 팔을 빼는 것.** 오라클B 여지는 양쪽 대비 +22~39%.

설계
----
  관측 단위 : (트리거, 팔) 쌍. 미체결 팔도 행으로 둔다(발주가 행동이고 결과는 0).
  타깃      : 그 팔의 순수익 (미체결=0). 즉 E[return | 이 팔을 건다].
  피쳐      : **트리거 봉까지의 규칙 피쳐만** -- Tier0 23 + 팔 ID + 신호 ID + side.
              ⚠️메타라벨 `_pct`는 넣지 않는다(필터 TRAIN과 100% 겹쳐 누수).
              누수 없는 상태에서 **경로/시장 피쳐만으로 되는가**를 먼저 본다.
  모델      : HistGradientBoostingRegressor (n≈21k라 TabPFN 컨텍스트 상한 초과)
  결정      : 예측수익 > τ 이면 그 팔을 건다. τ는 **VAL에서만** 고른다.
  기준선    : 양쪽 다 걸기 (모델 없음)

사전등록 판정
------------
  1. 모델 정책이 **양쪽 다 걸기**를 VAL·OOS 양 창에서 이길 것
  2. ⭐**역선택 4분할** -- 모델이 유지한 팔의 실제 평균수익 > 뺀 팔의 실제 평균수익.
     (오늘 클라이맥스 예측기와 버스트 필터가 둘 다 여기서 뒤집혀 있었다)
  3. 무작위 5시드 부호 일치
  4. anti-stable -- TRAIN 상위 피쳐 중요도가 VAL에서 유지되는가
HOLDOUT은 진단 표시만.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

SRC = ROOT / "tmp/eth_causal_population_metalabel_20260902"
ARMS = ROOT / "tmp/eth_entry_direction_oracle_v2_20260903/both_arms.csv"
OUT_DIR = ROOT / "tmp/eth_entry_b0a_20260903"
SEEDS = [76010, 130820, 194636, 331076, 703883]
HP = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=60,
          l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
          n_iter_no_change=25)
TAUS = [-np.inf, -0.0005, 0.0, 0.0002, 0.0005, 0.0010]


def log(m): print(f"[b0a] {m}", flush=True)


def stat(v):
    v = np.asarray(v, float)
    if len(v) == 0: return (0, 0.0, 0.0)
    w, l = v[v > 0].sum(), -v[v < 0].sum()
    return (len(v), float(v.mean() * 1e4), float(w / l) if l > 0 else float("inf"))


def main() -> int:
    cfg = json.loads((SRC / "config.json").read_text())["cfg"]
    base = [c for c in json.loads((SRC / "config.json").read_text())["features"] if c != "is_bottom"]

    a = pd.read_csv(ARMS, parse_dates=["ts"])
    # (트리거, 팔) 쌍으로 melt
    rows = []
    for armname, retc, fillc in (("sig", "p_sig", "sig_filled"), ("flip", "p_flip", "flip_filled")):
        t = a[["signal", "ts", "sig_dir", "atr", "split"]].copy()
        t["arm"] = 1 if armname == "sig" else 0        # 1=신호방향, 0=역방향
        t["y"] = a[retc].to_numpy()
        t["filled"] = a[fillc].to_numpy().astype(int)
        rows.append(t)
    d = pd.concat(rows, ignore_index=True)

    # 트리거 봉 Tier0 피쳐 조인
    feats = []
    for name in cfg:
        f = SRC / f"{name}_causal_fires.csv"
        if not f.exists(): continue
        x = pd.read_csv(f, parse_dates=["timestamp"])
        x = x[["timestamp"] + base].copy()
        x["signal"] = name
        feats.append(x)
    F = pd.concat(feats, ignore_index=True).rename(columns={"timestamp": "ts"})
    d = d.merge(F, on=["signal", "ts"], how="left")
    d["sig_id"] = pd.Categorical(d["signal"]).codes
    d = d.sort_values("ts").reset_index(drop=True)
    FEATS = base + ["arm", "sig_id", "sig_dir", "atr"]
    d[FEATS] = d[FEATS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    tr = (d.split == "TRAIN").to_numpy()
    d[FEATS] = d[FEATS].fillna(d.loc[tr, FEATS].median())
    log(f"(트리거,팔) {len(d):,}행 | " + " ".join(f"{k} {int(v):,}" for k, v in d.split.value_counts().items()))
    log(f"  체결률 {float(d.filled.mean()):.1%} | 피쳐 {len(FEATS)}개 (메타라벨 _pct 미포함)")

    # ---- 기준선 ----
    log("\n=== 기준선: 양쪽 다 걸기 (모델 없음) ===")
    basen = {}
    for wn in ("VAL", "OOS", "HOLDOUT"):
        w = d[d.split == wn]
        n, m, pf = stat(w.y)
        basen[wn] = m
        log(f"  {wn:8s} n={n:6,} 팔당 {m:+6.2f}bp PF{pf:5.2f} · 트리거당 {m*2:+6.2f}bp")

    # ---- 학습 (5시드) ----
    preds = {}
    for sd in SEEDS:
        mdl = HistGradientBoostingRegressor(random_state=sd, **HP).fit(d.loc[tr, FEATS], d.loc[tr, "y"])
        preds[sd] = mdl.predict(d[FEATS])
    d["pred"] = np.mean([preds[s] for s in SEEDS], axis=0)
    for wn in ("VAL", "OOS"):
        w = d[d.split == wn]
        log(f"\n예측-실제 상관 {wn}: {float(np.corrcoef(w.pred, w.y)[0,1]):+.4f}")

    # ---- 임계값 스윕 (τ는 VAL에서만 고름) ----
    log("\n=== τ 스윕 (팔당 bp / 유지비율) ===")
    print(f"{'τ(bp)':>8s} | " + " | ".join(f"{w:>26s}" for w in ("VAL", "OOS", "HOLDOUT")))
    res = []
    for tau in TAUS:
        cells, row = [], {"tau_bp": (tau * 1e4 if np.isfinite(tau) else -999)}
        for wn in ("VAL", "OOS", "HOLDOUT"):
            w = d[d.split == wn]
            keep = w.pred > tau
            kept = w.y.to_numpy()[keep.to_numpy()]
            # 정책 수익 = 유지한 팔만 (뺀 팔은 거래 없음 = 0), 팔 전체로 평균
            pol = np.where(keep.to_numpy(), w.y.to_numpy(), 0.0)
            n, m, pf = stat(pol)
            row[f"{wn}_bp"] = round(m, 2); row[f"{wn}_keep"] = round(float(keep.mean()), 3)
            cells.append(f"{m:+6.2f}bp 유지{float(keep.mean()):5.1%} PF{pf:5.2f}")
        res.append(row)
        print(f"{row['tau_bp']:8.1f} | " + " | ".join(f"{x:>26s}" for x in cells))
    log(f"  (기준선 = τ=-inf 행: VAL {basen['VAL']:+.2f} / OOS {basen['OOS']:+.2f} / HOLDOUT {basen['HOLDOUT']:+.2f})")

    # ---- ⭐역선택 4분할 (VAL 최적 τ로) ----
    rdf = pd.DataFrame(res)
    fin = rdf[rdf.tau_bp > -999]
    best = fin.loc[fin.VAL_bp.idxmax()]
    tau = best.tau_bp / 1e4
    log(f"\n=== ⭐역선택 4분할 (VAL 최적 τ = {best.tau_bp:+.1f}bp) ===")
    print(f"{'구간':9s} {'그룹':14s} {'n':>7s} {'실제 평균bp':>12s}")
    for wn in ("VAL", "OOS", "HOLDOUT"):
        w = d[d.split == wn]
        k = (w.pred > tau).to_numpy(); y = w.y.to_numpy()
        for lab, mask in (("유지(모델선택)", k), ("제외(모델버림)", ~k)):
            if mask.sum() == 0: continue
            print(f"{wn:9s} {lab:14s} {int(mask.sum()):7,} {y[mask].mean()*1e4:12.2f}")
    log("  → 유지 > 제외 여야 정상. 뒤집히면 역선택(오늘 두 번 걸린 패턴)")

    # ---- 5시드 부호 일치 ----
    log("\n=== 시드별 정책 우위 (VAL 최적 τ, 기준선 대비) ===")
    for wn in ("VAL", "OOS"):
        w = d[d.split == wn]; ds = []
        for sd in SEEDS:
            p = preds[sd][d.split.to_numpy() == wn]
            pol = np.where(p > tau, w.y.to_numpy(), 0.0)
            ds.append(pol.mean() * 1e4 - basen[wn])
        log(f"  {wn:5s} " + ", ".join(f"{x:+.2f}" for x in ds) +
            f"  → 양수 {sum(x>0 for x in ds)}/5")

    # ---- anti-stable ----
    log("\n=== anti-stable 진단 (순열중요도 TRAIN vs VAL) ===")
    m0 = HistGradientBoostingRegressor(random_state=SEEDS[0], **HP).fit(d.loc[tr, FEATS], d.loc[tr, "y"])
    sub_tr = d[tr].sample(min(4000, int(tr.sum())), random_state=0)
    va = d.split == "VAL"
    sub_va = d[va].sample(min(4000, int(va.sum())), random_state=0)
    it = permutation_importance(m0, sub_tr[FEATS], sub_tr.y, n_repeats=3, random_state=0)
    iv = permutation_importance(m0, sub_va[FEATS], sub_va.y, n_repeats=3, random_state=0)
    corr = float(np.corrcoef(it.importances_mean, iv.importances_mean)[0, 1])
    top = np.argsort(-it.importances_mean)[:20]
    keep_sign = int((iv.importances_mean[top] > 0).sum())
    log(f"  TRAIN-VAL 중요도 상관 {corr:+.4f}  (음수면 anti-stable 재현 = 중단 사유)")
    log(f"  TRAIN 상위 20 중 VAL에서도 양의 중요도: {keep_sign}/20")
    log(f"  TRAIN 상위 8: {[FEATS[i] for i in top[:8]]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT_DIR / "arm_rows_scored.csv", index=False)
    rdf.to_csv(OUT_DIR / "tau_sweep.csv", index=False)
    json.dump({"baseline_bp": basen, "best_tau_bp": float(best.tau_bp),
               "perm_corr_train_val": corr, "top20_sign_kept": keep_sign,
               "features": FEATS, "seeds": SEEDS, "metalabel_pct_used": False},
              open(OUT_DIR / "b0a_report.json", "w"), indent=2)
    log(f"\n산출: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
