#!/usr/bin/env python3
"""B8: 확장 표본에서 피쳐 재선별 + 대조군 재통과 (2026-09-03).

B2의 선별은 TRAIN 16,826행(행/피쳐 104)에서 했고 봉우리가 안 잡혔다. B6 확장으로
TRAIN이 81,168행(행/피쳐 **504**)이 됐으므로 다시 잰다.

⚠️절차 (선별 자체가 새 선택 축이므로 대조군을 다시 통과해야 한다):
  1. **TRAIN 내부 분할**로 순열중요도 (VAL 미사용). 상관 |r|>0.95 중복 제거 후 그리디.
  2. k 스윕은 **동결 설정**(d3.0/w6/τ40bp/4슬롯)에서 -- 161본이 대조군을 통과한 바로 그 설정.
  3. 선별본에 **무작위 필터 대조군 · 5시드 · 시간블록 부트스트랩**을 다시 건다.

⭐사전등록 판정:
  **선별본이 161개본을 VAL·OOS 양 창에서 못 이기면 161개본을 그대로 동결한다.**
  (선별의 목적은 단순화이지 성능 향상이 아니다. 동등하면 단순한 쪽.)
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

from research_eth_entry_b1_cumulative_arms_20260903 import SEEDS, HP, stat  # noqa: E402
from research_eth_entry_b6_expand_20260903 import slotN  # noqa: E402

B6 = ROOT / "tmp/eth_entry_b6_expand_20260903"
OUT = ROOT / "tmp/eth_entry_b8_featsel_20260903"
DEPTH, WAIT, TAU, NSLOT = 3.0, 6, 0.0040, 4      # 161본이 대조군 통과한 동결 설정
KS = [15, 25, 40, 60, 90, 161]
B_RND = 150
CORR_CUT = 0.95
RNG = np.random.default_rng(20260903)


def log(m): print(f"[b8] {m}", flush=True)


def main() -> int:
    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    cfg = json.loads((ROOT / "tmp/eth_causal_population_metalabel_20260902/config.json").read_text())
    base = [c for c in cfg["features"] if c != "is_bottom"]
    excl = set(base + ["arm", "sig_id", "atr_pct", "depth", "y", "split", "timestamp", "i",
                       "side", "signal", "fi", "ei", "btf", "lim", "sd", "pred"])
    R = [c for c in D.columns if c.endswith("_r136")] + \
        [c for c in D.columns if c not in excl and not c.endswith("_r136")]
    R = list(dict.fromkeys([c for c in R if D[c].dtype.kind in "fiub"]))
    FEATS = list(dict.fromkeys(base + ["arm", "sig_id", "atr_pct", "depth"] + R))
    X = D[FEATS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    tr = (D.split == "TRAIN").to_numpy()
    X = X.fillna(X[tr].median())
    y = D["y"].to_numpy()
    log(f"행 {len(D):,} · TRAIN {int(tr.sum()):,} · 피쳐 {len(FEATS)} (행/피쳐 {int(tr.sum())/len(FEATS):.0f})")

    # ---- 1. TRAIN 내부 순열중요도 ----
    idx = np.flatnonzero(tr); cut = int(len(idx) * 0.8)
    itr, iva = idx[:cut], idx[cut:]
    sub = iva[:8000]
    imps = []
    for s in SEEDS[:3]:
        m = HistGradientBoostingRegressor(random_state=s, **HP).fit(X.iloc[itr], y[itr])
        imps.append(permutation_importance(m, X.iloc[sub], y[sub], n_repeats=3,
                                           random_state=s, n_jobs=-1).importances_mean)
    imp = np.mean(imps, axis=0)
    log(f"순열중요도 완료 (내부 {len(itr):,}/{len(sub):,}, 3시드)")

    order = np.argsort(-imp)
    C = X.iloc[itr[:20000]].corr().abs().to_numpy()
    sel, drop = [], 0
    for i in order:
        if len(sel) >= max(KS): break
        if any(C[i, j] > CORR_CUT for j in sel):
            drop += 1; continue
        sel.append(int(i))
    log(f"중복 제거 {drop} → 후보 {len(sel)}")
    log(f"상위 15: {[FEATS[i] for i in sel[:15]]}")

    # ---- 2. k 스윕 (동결 설정에서) ----
    dsel = (D.depth == DEPTH) & (D.btf <= WAIT)
    log(f"\n=== k 스윕 @ d{DEPTH}/w{WAIT}/τ{TAU*1e4:.0f}bp/{NSLOT}슬롯 ===")
    print(f"{'k':>4s} | " + " | ".join(f"{w:>20s}" for w in ("VAL", "OOS", "HOLDOUT")))
    rows, preds_by_k = [], {}
    for k in KS:
        fs = [FEATS[i] for i in sel[:k]] if k < len(FEATS) else FEATS
        ps = {s: HistGradientBoostingRegressor(random_state=s, **HP)
              .fit(X[fs][tr], y[tr]).predict(X[fs]) for s in SEEDS}
        p = np.mean([ps[s] for s in SEEDS], axis=0)
        preds_by_k[k] = (p, ps, fs)
        row = {"k": k}
        cells = []
        for wn in ("VAL", "OOS", "HOLDOUT"):
            w = D[dsel & (D.split == wn) & (p > TAU)]
            v = slotN(w, NSLOT); nn, m, _ = stat(v)
            row[f"{wn}_bp"] = round(m, 2); row[f"{wn}_n"] = nn
            cells.append(f"{m:+7.2f}bp n={nn:4d}")
        rows.append(row)
        print(f"{k:4d} | " + " | ".join(f"{c:>20s}" for c in cells))
    r = pd.DataFrame(rows)
    full = r[r.k == 161].iloc[0]
    cand = r[r.k < 161]
    bk = int(cand.loc[cand.VAL_bp.idxmax(), "k"])
    br = r[r.k == bk].iloc[0]
    log(f"\n161본: VAL {full.VAL_bp:+.2f} / OOS {full.OOS_bp:+.2f} / HOLD {full.HOLDOUT_bp:+.2f}")
    log(f"선별본 최고 k={bk}: VAL {br.VAL_bp:+.2f} / OOS {br.OOS_bp:+.2f} / HOLD {br.HOLDOUT_bp:+.2f}")
    win = (br.VAL_bp > full.VAL_bp) and (br.OOS_bp > full.OOS_bp)
    log(f"⭐사전등록 판정: 선별본이 양 창에서 161본을 " + ("**이김 → 선별본 채택**" if win else "**못 이김 → 161본 동결**"))

    # ---- 3. 채택 후보에 대조군 재검정 ----
    kk = bk if win else 161
    p, ps, fs = preds_by_k[kk]
    log(f"\n=== 대조군 재검정 (k={kk}, {len(fs)}피쳐) ===")
    ctrl = {}
    for wn in ("VAL", "OOS", "HOLDOUT"):
        m_ = (dsel & (D.split == wn)).to_numpy()
        w = D[m_]; pk = p[m_]
        keep = pk > TAU; frac = float(keep.mean())
        real = stat(slotN(w[keep], NSLOT))[1]
        allm = stat(slotN(w, NSLOT))[1]
        rr = np.array([stat(slotN(w[RNG.random(len(w)) < frac], NSLOT))[1] for _ in range(B_RND)])
        sv = [stat(slotN(w[ps[s][m_] > TAU], NSLOT))[1] for s in SEEDS]
        sub2 = w[keep].sort_values("fi"); v = slotN(w[keep], NSLOT)
        s2 = sub2.iloc[:len(v)].copy(); s2["y2"] = v; s2["day"] = (s2.fi // 288).astype(int)
        days = s2.day.unique()
        bs = np.array([np.concatenate([s2.loc[s2.day == dd, "y2"].to_numpy()
                       for dd in RNG.choice(days, len(days), replace=True)]).mean() * 1e4
                       for _ in range(2000)])
        ctrl[wn] = {"real": real, "keep_all": allm, "rnd": float(rr.mean()),
                    "p_rnd": float((rr >= real).mean()), "seeds_beat": int(sum(x > allm for x in sv)),
                    "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                    "blocks": int(len(days))}
        log(f"  {wn:8s} 실제 {real:+6.2f} | 무필터 {allm:+6.2f} | 무작위필터 {rr.mean():+6.2f} "
            f"p={float((rr>=real).mean()):.3f} | 시드 {sum(x>allm for x in sv)}/5 | "
            f"CI [{np.quantile(bs,.025):+.2f},{np.quantile(bs,.975):+.2f}] 블록 {len(days)}일")

    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "k_sweep.csv", index=False)
    pd.DataFrame({"feature": [FEATS[i] for i in sel], "imp": imp[sel]}).to_csv(
        OUT / "ranked_features.csv", index=False)
    json.dump({"chosen_k": kk, "selected_beats_full": bool(win), "config":
               {"depth": DEPTH, "wait": WAIT, "tau_bp": TAU * 1e4, "slots": NSLOT},
               "features": fs, "controls": ctrl}, open(OUT / "b8_report.json", "w"), indent=2)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
