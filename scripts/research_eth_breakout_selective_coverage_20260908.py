#!/usr/bin/env python3
"""**선택적 예측** -- 커버리지를 줄여 정확도를 올릴 수 있는가 (H=1h·±0.30%, 2026-09-08).

사용자: *"60%이상으로 나오게 하려면 어떻게 해야하지?"*

## 두 경로 중 하나만 진짜다
❌ **기저 올리기**: 배리어를 좁히면 라벨이 되돌림 쪽으로 기울어 기저가 오른다(P=0.25% 에서 이미
   0.567~0.619). 아무것도 안 하는 모델이 그만큼 찍는다 -- 부록 AP 에서 두 번 잡은 착시.
✅ **정보량 한계 안에서 커버리지 축소**: 확률이 애매한 건 보류하고 확신하는 것만 판정한다.

## 정보량 상한 (이항정규 근사 acc = Φ(√2·Φ⁻¹(AUC)/2))
AUC 0.594→0.579 · **0.648→0.606** · 0.70→0.645 · 0.75→0.683.
현재 OOS 정확도 0.6010 은 AUC 0.648 의 상한 0.606 에 **이미 붙어 있다** -- 전건 판정으로는 한계다.

## 설계
HGB `기준(69)` 워크포워드 예측을 그대로 쓰고, |p−0.5| 상위 c% 만 채점한다.
⚠️**임계값은 TRAIN 예측 분포에서만** 정한다(표본외 분위를 쓰면 그 자체가 미래참조).
⚠️커버리지·건수·창별 기저를 반드시 병기한다. 선택된 부분집합의 기저도 따로 낸다
   ([[feedback_outcome_selected_subset_inflates_metrics_20260908]]).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

MY = ROOT / "tmp/eth_breakout_atr_state_20260908_s1"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H, P = 12, 0.0030
EMB = pd.Timedelta(hours=4)
CHUNK, SEED, BOOT = 4000, 20260908, 3000
COV = (1.0, 0.5, 0.3, 0.2, 0.1, 0.05)


def first_touch(hi1, lo1, start, up, dn, nmin):
    n = len(start)
    tu = np.full(n, -1, np.int32); td = np.full(n, -1, np.int32)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        idx = start[a:b, None] + np.arange(nmin)[None, :]
        hu = hi1[idx] >= up[a:b, None]; hd = lo1[idx] <= dn[a:b, None]
        au = hu.any(1); ad = hd.any(1)
        tu[a:b] = np.where(au, hu.argmax(1), -1); td[a:b] = np.where(ad, hd.argmax(1), -1)
    return tu, td


def day_ci(v, day, rng, Bt=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 6 or len(v) < 30: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (Bt, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def main() -> int:
    rng = np.random.default_rng(SEED)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    d = pd.read_parquet(MY / "dataset_v5_f154.parquet").sort_values("timestamp").reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    eth = B._load_kl(B.ETH_KL)
    ts5 = eth["timestamp"].to_numpy(); O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy(); hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float)
    sgn = np.where(d["dir_up"].to_numpy() > 0, 1.0, -1.0)
    bi = d["bar_idx"].to_numpy(); T = d["T_atr"].to_numpy(float)
    entry = O5[np.minimum(bi + 1, len(O5) - 1)] * (1 + sgn * T)
    s1 = np.searchsorted(ts1, d["timestamp"].to_numpy())
    bt = np.searchsorted(ts5, ts1[np.clip(s1, 0, len(ts1) - 1)], side="right") - 1
    okm = (s1 > 0) & (s1 + H * 5 < len(ts1)) & (bt + H < len(C5)) & (bt >= 1)
    tu, td = first_touch(hi1, lo1, np.where(okm, s1, 0), entry * (1 + P), entry * (1 - P), H * 5)
    big = 1 << 30
    uo, do_ = tu >= 0, td >= 0
    au, ad = np.where(uo, tu, big), np.where(do_, td, big)
    cont = np.where(sgn > 0, uo & (au < ad), do_ & (ad < au))
    rev = np.where(sgn > 0, do_ & (ad < au), uo & (au < ad))
    x5 = np.minimum(bt + H, len(C5) - 1)
    clo = (C5[x5] - entry) / entry * 1e4 * sgn
    y = np.where(cont, 1, np.where(rev, 0, (clo > 0).astype(int)))
    ts = d["timestamp"]; day = ts.dt.floor("D").to_numpy(); sp = d["split"].to_numpy()
    months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    base = [c for c in d.columns if c.startswith(("f_", "sig_", "v2_"))] + \
           [c for c in d.columns if c.startswith(("x_m_tkv", "x_m_ttp", "x_m_ttc", "x_m_retail"))] + \
           ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"]
    X = d[base].to_numpy(np.float32)
    pred = np.full(len(y), np.nan)
    for i, mo in enumerate(uniq):
        if i < 5: continue
        te = (months == mo).to_numpy(); tr = (ts < ts[te].min() - EMB).to_numpy()
        if tr.sum() < 1500 or te.sum() < 30: continue
        c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, early_stopping=True,
                                           validation_fraction=0.15, random_state=SEED)
        c.fit(X[tr], y[tr])
        pred[te] = c.predict_proba(X[te])[:, 1]
        # TRAIN 예측(같은 모델)도 임계값 산정용으로 저장
        if i == len(uniq) - 1: pass
    # ⚠️임계값 산정용 TRAIN 예측 = 마지막 폴드 모델의 in-sample 이 아니라
    #   TRAIN 구간에 대한 워크포워드 예측이 없으므로 **TRAIN 을 시간순 뒤 30% 로 재현**
    trm = sp == "TRAIN"
    cut = ts[trm].quantile(0.7)
    tr_fit = trm & (ts <= cut).to_numpy(); tr_hold = trm & (ts > cut).to_numpy()
    c0 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                        l2_regularization=1.0, early_stopping=True,
                                        validation_fraction=0.15, random_state=SEED)
    c0.fit(X[tr_fit], y[tr_fit])
    p_tr = c0.predict_proba(X[tr_hold])[:, 1]
    conf_tr = np.abs(p_tr - 0.5)
    print(f"임계값 산정 표본(TRAIN 뒤 30%): {tr_hold.sum():,}건\n" + "=" * 116, flush=True)
    print(f"{'커버':>6} {'임계|p-.5|':>10} | " + " | ".join(f"{w[:8]:>30}" for w in WINS))
    print("=" * 116)
    rows = []
    for cv in COV:
        thr = 0.0 if cv >= 1.0 else float(np.quantile(conf_tr, 1 - cv))
        line = f"{cv:>5.0%} {thr:>10.4f} | "; rec = dict(cov=cv, thr=thr)
        for w in WINS:
            m = np.isfinite(pred) & (sp == w) & okm & (np.abs(pred - 0.5) >= thr)
            mw = np.isfinite(pred) & (sp == w) & okm
            if m.sum() < 40: line += f"{'n부족':>30} | "; continue
            acc = ((pred[m] > 0.5).astype(int) == y[m]).astype(float)
            lo, hi = day_ci(acc, day[m], rng)
            b_ = max(y[m].mean(), 1 - y[m].mean())
            line += (f"{acc.mean():.4f}[{lo:.4f}] 기저{b_:.3f} "
                     f"커버{m.sum()/mw.sum():.0%} n{m.sum():>4}{'✅' if lo > b_ else '  '} | ")
            rec[f"{w}_acc"] = float(acc.mean()); rec[f"{w}_lo"] = lo
            rec[f"{w}_base"] = float(b_); rec[f"{w}_cov"] = float(m.sum() / mw.sum())
            rec[f"{w}_n"] = int(m.sum())
        print(line, flush=True)
        rows.append(rec)
    R = pd.DataFrame(rows); R.to_csv(MY / "selective_coverage.csv", index=False)
    R["pass3"] = [all(R.loc[i, f"{w}_lo"] > R.loc[i, f"{w}_base"] for w in WINS
                      if f"{w}_lo" in R.columns) for i in R.index]
    print("\n" + "=" * 116)
    print(f"⭐세 창 모두 CI 하한 > (그 부분집합의) 기저: {int(R.pass3.sum())}/{len(R)}")
    print(f"⭐세 창 모두 정확도 ≥ 60%: "
          f"{int(sum(all(R.loc[i, f'{w}_acc'] >= 0.60 for w in WINS) for i in R.index))}/{len(R)}")
    print("\n하루 표시 건수 환산 (전체 하루 ~19건 기준): " +
          " · ".join(f"커버{c:.0%}→{19*c:.1f}건" for c in COV))
    print(json.dumps({"rows": len(R)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
