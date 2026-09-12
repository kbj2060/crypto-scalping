#!/usr/bin/env python3
"""크기 축 재검정 — **변동성 지속성인가, 진짜 예측인가** (2026-09-07).

사용자: *"MASHT도 크기를 잘 맞추지 않았어? 이전에 얘기한 크기와 V자 급등락이 뭐가 다르지?"*

## 질문
이번 세션에서 살아남은 유일한 축이 "크기"였다(앵커 양성대조 0.77, MASHT 0.76).
그런데 그 라벨은 `range_pct > TRAIN 중앙값` -- **원시 퍼센트**다. ATR 로 나누지 않았다.
그러면 "지금 변동성이 높으면 앞으로 4시간도 폭이 넓다"는 **변동성 지속성**을 맞히는 것일 수 있고,
실제로 `atr_pct` 한 줄이 0.7576/0.7899/0.7430 을 낸다.

반면 V자 S 라벨은 `move >= 1.5*atr` 로 **ATR 정규화**돼 있다 -- "현재 변동성 **대비**" 큰가를 묻는다.
분모에 atr 이 있으니 변동성이 높을수록 오히려 달성이 어렵다. 다른 질문이다.

## 검정 (같은 앵커 모집단, 라벨만 정규화 여부로 갈라)
  S_raw   `range_pct > TRAIN 중앙값`              -- 지금까지 쓴 크기 라벨
  S_norm  `range_pct/atr_pct > TRAIN 중앙값`      -- ⭐ATR 정규화. 변동성 수준을 나눠 없앤다
각각에 대해
  (a) `atr_pct` 단독 (학습 없음)  (b) 전체 피쳐 + TabPFN
을 재고, V자 S 에도 같은 (a) 를 적용해 대조한다.

## 읽는 법
- S_norm 이 (a),(b) 모두 0.5 근처면 → **크기 축은 변동성 지속성이었다.** 세션의 유일한 생존 결과가
  "ATR 을 다시 말한 것"이 되고, 배포 사이징이 이미 atr_pct 를 쓰므로 새로운 것은 없다.
- S_norm 이 (b) 에서만 살아나면 → 변동성 수준 너머의 진짜 예측이 있다는 뜻이다.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tmp/eth_anchor_features154_20260907"
WT = ROOT / "tmp/eth_anchor_window_tensor_20260907"
VRD = ROOT / "tmp/eth_v_rebound_decomp_20260907"
OUT = ROOT / "tmp/eth_size_axis_recheck_20260907"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
SEED, N_EST, BOOT = 20260907, 4, 1200


def day_ci(y, p, d, rng, B=BOOT):
    u = np.unique(d)
    if len(u) < 5: return (np.nan, np.nan)
    idx = {x: np.flatnonzero(d == x) for x in u}; o = []
    for _ in range(B):
        i = np.concatenate([idx[x] for x in rng.choice(u, len(u), replace=True)])
        if len(np.unique(y[i])) > 1: o.append(roc_auc_score(y[i], p[i]))
    return (float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))) if len(o) > B // 3 else (np.nan, np.nan)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    from tabpfn import TabPFNClassifier
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    D = pd.read_parquet(SRC / "features154.parquet")
    meta = json.loads((SRC / "meta.json").read_text())
    cols = [c for c in meta["feature_cols"] if c in D.columns]
    sp = D["split"].to_numpy(); day = D["timestamp"].dt.floor("D").to_numpy()
    rp = D["range_pct"].to_numpy(float); atr = D["atr_pct"].to_numpy(float)
    rn = rp / np.maximum(atr, 1e-12)
    tr = sp == "TRAIN"
    LABELS = {
        "S_raw  (range_pct)": (rp > np.nanmedian(rp[tr])).astype(float),
        "S_norm (range/atr)": (rn > np.nanmedian(rn[tr])).astype(float),
    }
    print(f"[입력] 앵커 {len(D):,} · 피쳐 {len(cols)} · device {dev}", flush=True)
    print(f"       range_pct 와 atr_pct 의 상관 {np.corrcoef(rp[np.isfinite(rp)&np.isfinite(atr)], atr[np.isfinite(rp)&np.isfinite(atr)])[0,1]:.4f}", flush=True)
    print(f"       range/atr 과 atr_pct 의 상관 {np.corrcoef(rn[np.isfinite(rn)&np.isfinite(atr)], atr[np.isfinite(rn)&np.isfinite(atr)])[0,1]:.4f}", flush=True)

    X = D[cols].to_numpy(np.float64)
    rows = []
    print("\n" + "=" * 100, flush=True)
    print(f"{'라벨':<22}{'모델':<18}" + "".join(f"{w[:3]:>16}" for w in WINS) + "   mean3", flush=True)
    print("=" * 100, flush=True)
    for lname, y in LABELS.items():
        for mname in ("atr_pct 단독", f"전체{len(cols)}피쳐+TabPFN"):
            ok = np.isfinite(y) & np.isfinite(atr)
            rec = {"label": lname, "model": mname}
            if mname.startswith("atr"):
                pred_all = atr
            else:
                t = ok & tr
                clf = TabPFNClassifier(device=dev, n_estimators=N_EST, random_state=SEED,
                                       ignore_pretraining_limits=True, memory_saving_mode=True)
                clf.fit(np.nan_to_num(X[t]).astype(np.float32), y[t].astype(int))
                pred_all = np.full(len(D), np.nan)
                for w in WINS:
                    te = ok & (sp == w)
                    if te.sum() >= 40:
                        pred_all[te] = clf.predict_proba(np.nan_to_num(X[te]).astype(np.float32))[:, 1]
            line = f"{lname:<22}{mname:<18}"
            for w in WINS:
                te = ok & (sp == w) & np.isfinite(pred_all)
                if te.sum() < 40 or len(np.unique(y[te])) < 2:
                    line += f"{'-':>16}"; continue
                a = roc_auc_score(y[te].astype(int), pred_all[te])
                lo, _ = day_ci(y[te].astype(int), pred_all[te], day[te], rng)
                rec[f"{w}_auc"], rec[f"{w}_lo"] = a, lo
                line += f"  {a:.4f}[{lo:.3f}]"
            rec["mean3"] = np.nanmean([rec.get(f"{w}_auc", np.nan) for w in WINS])
            rows.append(rec)
            print(line + f"  {rec['mean3']:.4f}", flush=True)

    # ── V자 S 에 atr 단독을 걸어 대조
    print("\n" + "=" * 100, flush=True)
    print("대조: V자 S 라벨(ATR 정규화된 크기)에 atr 단독", flush=True)
    print("=" * 100, flush=True)
    f = VRD / "decomp.csv"
    if f.exists():
        A = pd.read_csv(f)
        for _, r in A.iterrows():
            print(f"   {r['label']:<4}{str(r['desc'])[:38]:<40} mean3 {r['mean3']:.4f}", flush=True)
    print("   ⇒ V자 S(0.7708)는 ATR 로 나눈 라벨이라 변동성 지속성으로 설명되지 않는다.", flush=True)

    R = pd.DataFrame(rows); R.to_csv(OUT / "size_recheck.csv", index=False)
    g = lambda l, m: float(R[(R.label == l) & (R.model.str.startswith(m))].mean3.iloc[0])
    print("\n" + "=" * 100, flush=True)
    print(f"  S_raw : atr단독 {g('S_raw  (range_pct)','atr'):.4f} → 전체피쳐 {g('S_raw  (range_pct)','전체'):.4f}", flush=True)
    print(f"  S_norm: atr단독 {g('S_norm (range/atr)','atr'):.4f} → 전체피쳐 {g('S_norm (range/atr)','전체'):.4f}", flush=True)
    sn = g('S_norm (range/atr)', '전체')
    print(f"\n  ⇒ {'🔴크기 축은 변동성 지속성이었다 -- ATR 로 나누면 사라진다' if sn < 0.56 else '✅변동성 수준 너머의 예측이 남는다'}", flush=True)
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
