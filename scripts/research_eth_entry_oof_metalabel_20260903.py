#!/usr/bin/env python3
"""진입 필터용 OOF 메타라벨 생성 -- 누수 차단 (2026-09-03).

WHY
---
재료 `_pct`를 진입 필터의 입력으로 쓰면 누수가 생긴다. 그것도 100%다 --
메타라벨 TRAIN(2024-01~2025-08)이 필터 TRAIN과 완전히 같은 창이라, 필터 TRAIN의 **모든 행**에서
`_pct`가 in-sample이다. 훈련 중엔 그 피쳐가 실제보다 유능해 보이니 모델이 과의존하고, 실전에선
그 의존이 어긋난다 -- **entry 7전7패를 만든 "TRAIN 관계가 VALIDATION에서 안 살아남는" 패턴**이다.

⚠️평가가 부풀지는 않는다(VAL/OOS의 `_pct`는 이미 깨끗하다). 망가지는 건 **학습**이다.

WHAT (스태킹 표준 처방: OOF 교차적합)
------------------------------------
  워밍업  2024-01-01 ~ 2024-04-30  메타라벨 최초 학습용. 필터 TRAIN에서 제외한다.
  fold    2024-05 ~ 2025-08 을 4등분(각 4개월). fold k의 `_pct`는 **fold k 시작 이전 데이터만**
          본 모델이 만든다(확장창).
  최종    2025-09 이후(VAL/OOS/HOLDOUT)는 필터 TRAIN 전체(< 2025-09-01)로 학습한 모델.

백분위 매핑도 각 단계의 **자기 학습분포** 기준으로 만든다(고정 매핑, 인과적).
OOF fold는 시드 1개, 최종만 4시드 -- 비용 현실화.

실행: 서버 quant_ai (GPU + .env의 TABPFN_TOKEN)
"""
from __future__ import annotations

import json
import os
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
from sklearn.metrics import roc_auc_score  # noqa: E402

SRC = ROOT / "tmp/eth_causal_population_metalabel_20260902"
OUT = ROOT / "tmp/eth_entry_oof_metalabel_20260903"
WARMUP_END = pd.Timestamp("2024-05-01")
FOLDS = [pd.Timestamp(x) for x in
         ("2024-05-01", "2024-09-01", "2025-01-01", "2025-05-01", "2025-09-01")]
FILTER_TRAIN_END = pd.Timestamp("2025-09-01")
SEEDS_FINAL = [20260829, 141592, 271828, 577215]
SEED_OOF = 20260829


def log(m): print(f"[oof] {m}", flush=True)


def pct_map(train_proba, x):
    t = np.sort(np.asarray(train_proba, float))
    return np.searchsorted(t, np.asarray(x, float), side="right") / max(len(t), 1)


def main() -> int:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TABPFN_TOKEN="):
                os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip('"')
    from tabpfn import TabPFNClassifier

    cfg = json.loads((SRC / "config.json").read_text())
    feats = cfg["features"]
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []

    for name in cfg["cfg"]:
        f = SRC / f"{name}_causal_fires.csv"
        if not f.exists():
            log(f"⚠️ {name} 없음"); continue
        d = pd.read_csv(f, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        X = d[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        y = d["hit"].to_numpy()
        med = X[d.timestamp < FILTER_TRAIN_END].median()
        X = X.fillna(med)
        d["proba_oof"] = np.nan
        d["pct_oof"] = np.nan
        d["oof_source"] = ""

        # ---- OOF folds (확장창) ----
        for k in range(len(FOLDS) - 1):
            lo, hi = FOLDS[k], FOLDS[k + 1]
            trm = (d.timestamp < lo).to_numpy()
            fdm = ((d.timestamp >= lo) & (d.timestamp < hi)).to_numpy()
            if trm.sum() < 200 or fdm.sum() == 0:
                log(f"  {name} fold{k+1} 건너뜀 (train {int(trm.sum())} / fold {int(fdm.sum())})")
                continue
            clf = TabPFNClassifier(device="cuda", random_state=SEED_OOF)
            clf.fit(X[trm].to_numpy(), y[trm])
            ptr = clf.predict_proba(X[trm].to_numpy())[:, 1]
            pfd = clf.predict_proba(X[fdm].to_numpy())[:, 1]
            d.loc[fdm, "proba_oof"] = pfd
            d.loc[fdm, "pct_oof"] = pct_map(ptr, pfd)
            d.loc[fdm, "oof_source"] = f"fold{k+1}(train<{lo.date()},n={int(trm.sum())})"
            try:
                a = roc_auc_score(y[fdm], pfd) if len(np.unique(y[fdm])) > 1 else float("nan")
            except Exception:
                a = float("nan")
            log(f"  {name:24s} fold{k+1} {lo.date()}~{hi.date()} train {int(trm.sum()):5,} "
                f"→ {int(fdm.sum()):5,}행 AUC {a:.4f}")

        # ---- 최종: 필터 TRAIN 전체로 학습, 그 이후 구간 예측 (4시드) ----
        trm = (d.timestamp < FILTER_TRAIN_END).to_numpy()
        post = ~trm
        probas = []
        for sd in SEEDS_FINAL:
            clf = TabPFNClassifier(device="cuda", random_state=sd)
            clf.fit(X[trm].to_numpy(), y[trm])
            probas.append(clf.predict_proba(X.to_numpy())[:, 1])
        P = np.vstack(probas); pm = P.mean(axis=0)
        d.loc[post, "proba_oof"] = pm[post]
        d.loc[post, "pct_oof"] = pct_map(pm[trm], pm[post])
        d.loc[post, "oof_source"] = f"final(train<{FILTER_TRAIN_END.date()},n={int(trm.sum())})"

        rec = {"signal": name, "n": len(d),
               "n_oof": int(d.proba_oof.notna().sum() - post.sum()),
               "n_final": int(post.sum()),
               "n_dropped_warmup": int((d.timestamp < WARMUP_END).sum())}
        for wn, lo, hi in (("VAL", pd.Timestamp("2025-09-01"), pd.Timestamp("2026-01-01")),
                           ("OOS", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01"))):
            m = ((d.timestamp >= lo) & (d.timestamp < hi)).to_numpy()
            if m.sum() > 50 and len(np.unique(y[m])) > 1:
                rec[f"auc_{wn.lower()}"] = round(float(roc_auc_score(y[m], d.proba_oof.to_numpy()[m])), 4)
        summary.append(rec)
        d[["pos", "timestamp", "side", "is_bottom", "hit", "split",
           "proba_oof", "pct_oof", "oof_source"]].to_csv(OUT / f"{name}_oof.csv", index=False)
        log(f"{name:26s} OOF {rec['n_oof']:,} · 최종 {rec['n_final']:,} · 워밍업제외 {rec['n_dropped_warmup']:,}")

    s = pd.DataFrame(summary)
    s.to_csv(OUT / "oof_summary.csv", index=False)
    json.dump({"warmup_end": str(WARMUP_END), "folds": [str(x) for x in FOLDS],
               "filter_train_end": str(FILTER_TRAIN_END),
               "seed_oof": SEED_OOF, "seeds_final": SEEDS_FINAL,
               "scheme": "expanding-window time-series OOF; fold k trained on data strictly before fold k"},
              open(OUT / "oof_config.json", "w"), indent=2)
    log("\n=== 요약 ===")
    print(s.to_string(index=False))
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
