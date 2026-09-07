#!/usr/bin/env python3
"""v4 마무리 -- 사전지정 소수 테이프 피쳐 + 경계 감사 (2026-09-08).

전체 테이프 53피쳐를 넣으면 VAL 이 52.33 → 50.22 로 **떨어진다**(학습 11,954행에 53피쳐 추가 = 과적합).
그래서 **사전 지정한 소수 가설 피쳐만** 넣어 본다(피쳐 사냥 아님):
  · `t_kyle_lambda` 임팩트 계수 -- 얇은 호가를 때린 이동인가
  · `mv_absorption` 흡수 -- 거래대금 대비 가격이 덜 움직였나
  · `mv_cvd_frac` 방향정렬 CVD 비율 -- 실제 공격적 흐름이 그 방향이었나
  · `mv_xl_frac` 초대형 체결 비중 -- 고래 한 방인가
  · `t_obi_obi_tight_al` 방향정렬 근접 OBI -- 반대편 호가가 비었나
그리고 순열 중요도 1위였던 **BTC 동조**(`v2_mv_btc_ret_atr` +0.0585)를 별도 축으로 확인한다.
마지막으로 계약대로 **경계 트립와이어**를 v4 에 돌린다.
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
PICK = ["t_kyle_lambda", "mv_absorption", "mv_cvd_frac", "mv_xl_frac",
        "t_obi_obi_tight_al", "t_impact_per_vol", "mv_kyle_lambda_max"]
BTC = ["v2_mv_btc_ret_atr", "v2_mv_idio_atr", "v2_mv_same_sign"]


def day_ci(v, day, rng, B=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def wf(X, y, ts, months, uniq):
    from sklearn.ensemble import HistGradientBoostingClassifier
    pred = np.full(len(y), np.nan)
    for i, mo in enumerate(uniq):
        if i < 5: continue
        te = (months == mo).to_numpy(); tr = (ts < ts[te].min() - EMB).to_numpy()
        if tr.sum() < 1500 or te.sum() < 30: continue
        c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, early_stopping=True,
                                           validation_fraction=0.15, random_state=SEED)
        c.fit(X[tr], y[tr]); pred[te] = c.predict_proba(X[te])[:, 1]
    return pred


def main() -> int:
    rng = np.random.default_rng(SEED)
    from sklearn.metrics import roc_auc_score
    d = pd.read_parquet(MY / "dataset_v4.parquet"); d["timestamp"] = pd.to_datetime(d["timestamp"])
    tape_all = [c for c in d.columns if c.startswith(("t_", "mv_"))]
    have = d[tape_all].notna().any(axis=1).to_numpy()
    d = d[have].sort_values("timestamp").reset_index(drop=True)
    base = [c for c in d.columns if c.startswith(("f_", "sig_", "v2_"))] + \
           [c for c in d.columns if c.startswith(("x_m_tkv", "x_m_ttp", "x_m_ttc", "x_m_retail"))] + \
           ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"]
    pick = [c for c in PICK if c in d.columns]
    nobtc = [c for c in base if c not in BTC]
    y = d["y"].to_numpy(int); ts = d["timestamp"]; day = ts.dt.floor("D").to_numpy()
    sp = d["split"].to_numpy(); months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    print(f"행 {len(d):,} · 사전지정 테이프 {len(pick)}개 {pick}\n" + "=" * 116, flush=True)
    for nm, cols in (("기준", base), (f"+사전지정{len(pick)}", base + pick),
                     ("기준−BTC동조", nobtc), ("BTC동조만", BTC + ["dir_up", "T_atr"])):
        pred = wf(d[cols].to_numpy(np.float32), y, ts, months, uniq)
        line = f"{nm:>16}({len(cols):>3}) | "
        for w in WINS:
            m = np.isfinite(pred) & (sp == w)
            acc = ((pred[m] > 0.5).astype(int) == y[m]).astype(float)
            lo, hi = day_ci(acc, day[m], rng)
            b = max(y[m].mean(), 1 - y[m].mean())
            line += (f"{w[:4]} {acc.mean():.4f}[{lo:.4f}] A{roc_auc_score(y[m], pred[m]):.3f} "
                     f"기저{b:.3f} {'✅' if lo > b else '❌'} | ")
        print(line, flush=True)
    print("\n" + "=" * 116)
    print("경계 트립와이어 (계약 준수) -- 새 테이프/OBI 피쳐군 제거 시 하락폭")
    print("=" * 116)
    full = base + [c for c in tape_all if c not in base]
    for nm, cols in (("전체", full), ("테이프/OBI 제거", base)):
        pred = wf(d[cols].to_numpy(np.float32), y, ts, months, uniq)
        m = np.isfinite(pred) & np.isin(sp, WINS)
        print(f"   {nm:>16}({len(cols):>3}): 정확도 "
              f"{((pred[m]>0.5).astype(int)==y[m]).mean():.4f} AUC {roc_auc_score(y[m], pred[m]):.4f}")
    print(json.dumps({"done": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
