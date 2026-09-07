#!/usr/bin/env python3
"""**피쳐 카테고리별 스윕** — 어느 축이 신호를 갖는가 (2026-09-07).

사용자: *"피쳐 카테고리별로 나눠서 스윕하는건가? 아니면 전체를 다 넣어서 테스트 중인가?"*
=> 그때까지는 **전체 + 순열중요도 상위 K** 만 돌고 있었다. 카테고리 분해는 안 했다.

## 왜 카테고리인가
① **어느 축이 신호를 갖는지 분해**된다 -- 전체를 넣으면 "없다"만 알고 어디에 없는지는 모른다
② 카테고리가 작아(10~44개) **과적합이 덜하다**. 실제로 199피쳐 실험에서 TRAIN CV 가
   피쳐 수에 **단조 감소**했다(atr 1개 0.5441 > perm20 0.5381 > ... > all199 0.5046)
③ 순열중요도는 상관된 피쳐 사이에서 불안정한데, 카테고리는 그 영향을 안 받는다

## 카테고리 (150 감사통과 피쳐를 이름 규칙으로 분할, 겹침 없음)
  변동성/레인지 · 스프레드/유동성 · 추세/모멘텀 · 포지셔닝/강제흐름 ·
  분수차분/통계 · 구조/레벨 · 교차자산 · 기타
  + 대조: atr 단독 · all150 전체

절차는 부록 M 과 동일 -- TRAIN 내부 purged K-fold 에서만 HP 선택, VAL/OOS 최종 1회, 날 블록 귀무.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import research_eth_anchor_direction_tuned_20260907 as TU  # noqa: E402

SRCD = ROOT / "tmp/eth_anchor_features154_20260907"
OUT = ROOT / "tmp/eth_anchor_category_sweep_20260907"
TU.HP_GRID = TU.HP_GRID[:6]
TU.SEEDS = [11, 23]
NULL_B = 12

PATTERNS = [
    ("변동성/레인지", r"vol|atr|parkinson|garman|range|semivar|kurt"),
    ("스프레드/유동성", r"spread|corwin|roll_|amihud|kyle|vpin|illiq|depth|obi"),
    ("추세/모멘텀", r"trend|mom|ema|sma|macd|adx|slope|mtf"),
    ("포지셔닝/강제흐름", r"oi|lsr|funding|whale|retail|taker|long_short"),
    ("분수차분/통계", r"ffd|entropy|variance_ratio|hurst|autocorr|zscore|_z$"),
    ("구조/레벨", r"fvg|gap|pivot|level|sweep|liq|vwap|bb_|rsi|stoch"),
    ("교차자산", r"btc|basis|spot"),
]


def categories(cols):
    seen, out = set(), {}
    for name, pat in PATTERNS:
        g = [c for c in cols if re.search(pat, c, re.I) and c not in seen]
        seen |= set(g)
        if len(g) >= 5:
            out[name] = g
    rest = [c for c in cols if c not in seen]
    if len(rest) >= 5:
        out["기타"] = rest
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260907)
    D = pd.read_parquet(SRCD / "features154.parquet")
    meta = json.loads((SRCD / "meta.json").read_text())
    cols = [c for c in meta["feature_cols"] if c in D.columns]
    if "atr_pct" in D.columns and "atr_pct" not in cols:
        cols.append("atr_pct")
    F = D[cols].to_numpy(np.float64)
    cats = categories(cols)
    cats["all150"] = cols
    cats["atr(대조)"] = [c for c in cols if c == "atr_pct"] or [cols[0]]
    idx = {c: i for i, c in enumerate(cols)}
    print(f"[입력] {D.shape} · 피쳐 {len(cols)} · split {D.split.value_counts().to_dict()}")
    for k, v in cats.items():
        print(f"   {k:<16}{len(v):>4}개")
    rows = []
    for arm in ("hard", "three", "wbin"):
        print(f"\n[{arm}]", flush=True)
        for cname, cfeat in cats.items():
            fc = [idx[c] for c in cfeat]
            cv, _ = TU.cv_search.__wrapped__(D, F, cols, arm, rng) if hasattr(TU.cv_search, "__wrapped__") else (None, None)
            # HP 는 카테고리 안에서 TRAIN CV 로 고른다
            best_hp, best_cv = 0, -9
            m, y, w, multi = TU.arm_spec(D, arm)
            tr = m & (D["split"].to_numpy() == "TRAIN")
            kp = D["bar_idx"].to_numpy()[tr]
            folds = TU.purged_folds(kp, None)
            for hi, hp in enumerate(TU.HP_GRID):
                s = []
                for a, b in folds:
                    try:
                        p = np.mean([TU.fit_pred(F[tr][a][:, fc], y[tr][a], w[tr][a], F[tr][b][:, fc], hp, sd, multi)
                                     for sd in TU.SEEDS], axis=0)
                        s.append(TU.score(y[tr][b], p, multi))
                    except Exception:                       # noqa: BLE001
                        s.append(np.nan)
                v = float(np.nanmean(s))
                if np.isfinite(v) and v > best_cv:
                    best_cv, best_hp = v, hi
            r = TU.final_eval(D, F, arm, fc, TU.HP_GRID[best_hp], rng)
            nv, no = [], []
            for _ in range(NULL_B):
                rr = TU.final_eval(D, F, arm, fc, TU.HP_GRID[best_hp], rng, shuffle=True, seeds=TU.SEEDS[:1])
                nv.append(rr["VAL_auc"]); no.append(rr["OOS_auc"])
            r.update({"cat": cname, "n_feat": len(fc), "hp": best_hp, "cv": best_cv,
                      "null_V": float(np.nanpercentile(nv, 95)), "null_O": float(np.nanpercentile(no, 95))})
            r["PASS"] = bool(r.get("VAL_lo", 0) > 0.5 and r.get("OOS_lo", 0) > 0.5
                             and r["VAL_auc"] > r["null_V"] and r["OOS_auc"] > r["null_O"])
            rows.append(r)
            print(f"   {cname:<16}{len(fc):>4}개 CV {best_cv:.4f} · VAL {r['VAL_auc']:.4f}"
                  f"[{r.get('VAL_lo',np.nan):.3f},{r.get('VAL_hi',np.nan):.3f}] · OOS {r['OOS_auc']:.4f}"
                  f"[{r.get('OOS_lo',np.nan):.3f},{r.get('OOS_hi',np.nan):.3f}] · 귀무 {r['null_V']:.3f}/{r['null_O']:.3f}"
                  f" {'✅' if r['PASS'] else ''}", flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / "category_sweep.csv", index=False)
    print(f"\n통과 {int(R.PASS.sum())}/{len(R)} · 저장 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
