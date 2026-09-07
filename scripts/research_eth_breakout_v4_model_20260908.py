#!/usr/bin/env python3
"""v4 모델 -- 테이프/OBI 가 정확도를 올리는가 (2026-09-08).

⚠️테이프는 2024-04-20 부터라 결합률 78.8%. **기준 모델도 같은 행으로 제한**해 공정 비교한다.
판정: 세 창(VAL/OOS/HOLDOUT) 정확도 CI 하한 > 기저. 셔플 대조군·경계 트립와이어 동반.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MY = ROOT / "tmp/eth_breakout_atr_state_20260908_s1"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
EMB = pd.Timedelta(hours=4)
SEED = 20260908
BOOT = 3000


def day_ci(v, day, rng, B=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def wf(X, y, ts, months, uniq, shuffle=False, rng=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    pred = np.full(len(y), np.nan)
    for i, mo in enumerate(uniq):
        if i < 5: continue
        te = (months == mo).to_numpy(); tr = (ts < ts[te].min() - EMB).to_numpy()
        if tr.sum() < 1500 or te.sum() < 30: continue
        yy = y.copy()
        if shuffle: yy[tr] = rng.permutation(yy[tr])
        c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, early_stopping=True,
                                           validation_fraction=0.15, random_state=SEED)
        c.fit(X[tr], yy[tr]); pred[te] = c.predict_proba(X[te])[:, 1]
    return pred


def main() -> int:
    rng = np.random.default_rng(SEED)
    from sklearn.metrics import roc_auc_score
    d = pd.read_parquet(MY / "dataset_v4.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    tape = [c for c in d.columns if c.startswith(("t_", "mv_")) and not c.startswith("t_obi")]
    obi = [c for c in d.columns if c.startswith("t_obi")]
    base = [c for c in d.columns if c.startswith(("f_", "sig_", "v2_"))] + \
           [c for c in d.columns if c.startswith(("x_m_tkv", "x_m_ttp", "x_m_ttc", "x_m_retail"))] + \
           ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"]
    have = d[tape].notna().any(axis=1).to_numpy()
    d = d[have].sort_values("timestamp").reset_index(drop=True)
    print(f"테이프 보유 행만: {len(d):,} · 기준 {len(base)} · 테이프 {len(tape)} · OBI {len(obi)}", flush=True)
    y = d["y"].to_numpy(int); ts = d["timestamp"]; day = ts.dt.floor("D").to_numpy()
    sp = d["split"].to_numpy(); months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    print(f"창별 n: " + " ".join(f"{w} {(sp==w).sum():,}" for w in ("TRAIN",) + WINS) + "\n")
    print("=" * 116)
    res = []
    for nm, cols in (("기준(같은 행)", base), ("+테이프", base + tape), ("+OBI", base + obi),
                     ("+테이프+OBI", base + tape + obi), ("테이프만", tape), ("테이프+OBI만", tape + obi)):
        pred = wf(d[cols].to_numpy(np.float32), y, ts, months, uniq)
        line = f"{nm:>14}({len(cols):>3}) | "
        rec = dict(cfg=nm, n_feat=len(cols))
        for w in WINS:
            m = np.isfinite(pred) & (sp == w)
            if m.sum() < 100: line += f"{w[:4]} -- | "; continue
            acc = ((pred[m] > 0.5).astype(int) == y[m]).astype(float)
            lo, hi = day_ci(acc, day[m], rng)
            b = max(y[m].mean(), 1 - y[m].mean())
            line += (f"{w[:4]} {acc.mean():.4f}[{lo:.4f}] A{roc_auc_score(y[m], pred[m]):.3f} "
                     f"기저{b:.3f} | ")
            rec[f"{w}_acc"] = float(acc.mean()); rec[f"{w}_lo"] = lo; rec[f"{w}_base"] = float(b)
            rec[f"{w}_auc"] = float(roc_auc_score(y[m], pred[m]))
        print(line, flush=True)
        res.append(rec)
    R = pd.DataFrame(res)
    R["pass3"] = [all(R.loc[i, f"{w}_lo"] > R.loc[i, f"{w}_base"] for w in WINS) for i in R.index]
    R.to_csv(MY / "v4_accuracy.csv", index=False)
    print("\n" + "=" * 116)
    print(f"⭐세 창 모두 CI 하한 > 기저: {int(R.pass3.sum())}/{len(R)}  "
          f"{R[R.pass3]['cfg'].tolist() if R.pass3.any() else ''}")
    # 최고 구성 셔플 대조군 + 순열 중요도
    R["m"] = R[[f"{w}_acc" for w in WINS]].min(1)
    b = R.sort_values("m", ascending=False).iloc[0]
    cols = {"기준(같은 행)": base, "+테이프": base + tape, "+OBI": base + obi,
            "+테이프+OBI": base + tape + obi, "테이프만": tape, "테이프+OBI만": tape + obi}[b.cfg]
    pc = wf(d[cols].to_numpy(np.float32), y, ts, months, uniq, True, np.random.default_rng(7))
    print(f"\n최고 구성 [{b.cfg}] 셔플 대조군: " + " ".join(
        f"{w[:4]} {((pc[np.isfinite(pc)&(sp==w)]>0.5).astype(int)==y[np.isfinite(pc)&(sp==w)]).mean():.4f}"
        for w in WINS))
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = d[cols].to_numpy(np.float32)
    te = (months == uniq[-2]).to_numpy(); tr = (ts < ts[te].min() - EMB).to_numpy()
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, early_stopping=True,
                                         validation_fraction=0.15, random_state=SEED).fit(X[tr], y[tr])
    ev = np.isin(sp, WINS); Xe = X[ev].copy()
    a0 = ((clf.predict_proba(Xe)[:, 1] > 0.5).astype(int) == y[ev]).mean()
    imp = []
    for j, c in enumerate(cols):
        s_ = Xe[:, j].copy(); Xe[:, j] = rng.permutation(s_)
        imp.append((c, a0 - ((clf.predict_proba(Xe)[:, 1] > 0.5).astype(int) == y[ev]).mean()))
        Xe[:, j] = s_
    imp.sort(key=lambda t: -t[1])
    print("\n순열 중요도 상위 15:")
    for c, v in imp[:15]: print(f"   {c:>26} {v:+.4f}")
    print(json.dumps({"pass3": int(R.pass3.sum())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
