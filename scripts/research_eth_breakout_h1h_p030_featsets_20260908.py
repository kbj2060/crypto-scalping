#!/usr/bin/env python3
"""H=1시간(12봉) · 배리어 ±0.30% 에서 **피쳐셋 절제** (HGB, 2026-09-08).

사용자: *"h 1시간 배리어 0.3% 로 딥러닝 모델 학습하고 테스트해줘"* → *"그 전에 피쳐셋도 테스트해줘"*

라벨: 트리거 레벨에서 ±0.30% 1분봉 첫터치(60분 안), 미터치는 12봉 뒤 종가 부호 → **커버리지 100%**.
표본외 실측: 배리어 해소 **88.7%**, 돌파율 **0.448**(되돌림 55.2%).
⚠️**기저가 50% 가 아니다.** P 를 좁히면 라벨이 되돌림 쪽으로 기운다(부록 AP). 창별 기저를 병기한다.
⚠️엠바고는 라벨 지평 이상: `max(4h, 60분)` = 4h.
피쳐는 전부 트리거 분 `s1-1` 기준이라 H·P 와 무관하다(경계 계약).

피쳐군: v1봉 · 경로 · 레벨 · BTC동조 · 포지셔닝(x_m_) · 횡단면(x_xr_) · 테이프(t_/mv_) · OBI(t_obi_)
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
CHUNK = 4000
SEED = 20260908
BOOT = 3000


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
        if shuffle: yy[np.flatnonzero(tr)] = rng.permutation(yy[np.flatnonzero(tr)])
        c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, early_stopping=True,
                                           validation_fraction=0.15, random_state=SEED)
        c.fit(X[tr], yy[tr]); pred[te] = c.predict_proba(X[te])[:, 1]
    return pred


def main() -> int:
    rng = np.random.default_rng(SEED)
    from sklearn.metrics import roc_auc_score
    d = pd.read_parquet(MY / "dataset_v4.parquet").sort_values("timestamp").reset_index(drop=True)
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
    okm = (s1 > 0) & (s1 + H * 5 < len(ts1)) & (bt + H < len(C5))
    tu, td = first_touch(hi1, lo1, np.where(okm, s1, 0), entry * (1 + P), entry * (1 - P), H * 5)
    big = 1 << 30
    uo = tu >= 0; do_ = td >= 0
    au = np.where(uo, tu, big); ad = np.where(do_, td, big)
    upf = uo & (au < ad); dnf = do_ & (ad < au)
    cont = np.where(sgn > 0, upf, dnf); rev = np.where(sgn > 0, dnf, upf)
    x5 = np.minimum(bt + H, len(C5) - 1)
    clo = (C5[x5] - entry) / entry * 1e4 * sgn
    y = np.where(cont, 1, np.where(rev, 0, (clo > 0).astype(int)))
    ts = d["timestamp"]; day = ts.dt.floor("D").to_numpy(); sp = d["split"].to_numpy()
    months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    ev = okm & np.isin(sp, WINS)
    print(f"사건 {okm.sum():,} · 해소율 {(cont|rev)[okm].mean():.3f} · 돌파율 {y[okm].mean():.4f}")
    print("창별 기저(다수결): " + " ".join(
        f"{w} {max(y[okm&(sp==w)].mean(), 1-y[okm&(sp==w)].mean()):.4f}(n{(okm&(sp==w)).sum():,})"
        for w in ("TRAIN",) + WINS) + "\n" + "=" * 122, flush=True)

    G = {
        "v1봉": [c for c in d.columns if c.startswith(("f_", "sig_"))] +
                ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"],
        "경로": [c for c in d.columns if c.startswith("v2_mv_") and not any(
            k in c for k in ("btc", "idio", "same_sign"))],
        "레벨": [c for c in d.columns if c.startswith(("v2_brk", "v2_pos"))],
        "BTC": ["v2_mv_btc_ret_atr", "v2_mv_idio_atr", "v2_mv_same_sign"],
        "포지셔닝": [c for c in d.columns if c.startswith(("x_m_tkv", "x_m_ttp", "x_m_ttc", "x_m_retail"))],
        "횡단면": [c for c in d.columns if c.startswith("x_xr_")],
        "테이프": [c for c in d.columns if c.startswith(("t_", "mv_")) and not c.startswith("t_obi")],
        "OBI": [c for c in d.columns if c.startswith("t_obi")],
    }
    SETS = [("BTC동조만", ["BTC"] ), ("v1봉만", ["v1봉"]), ("v1+경로", ["v1봉", "경로"]),
            ("v1+레벨", ["v1봉", "레벨"]), ("v1+BTC", ["v1봉", "BTC"]),
            ("v1+포지셔닝", ["v1봉", "포지셔닝"]), ("v1+횡단면", ["v1봉", "횡단면"]),
            ("기준", ["v1봉", "경로", "레벨", "BTC", "포지셔닝"]),
            ("기준+테이프", ["v1봉", "경로", "레벨", "BTC", "포지셔닝", "테이프"]),
            ("기준+OBI", ["v1봉", "경로", "레벨", "BTC", "포지셔닝", "OBI"]),
            ("기준+횡단면", ["v1봉", "경로", "레벨", "BTC", "포지셔닝", "횡단면"]),
            ("전체", list(G))]
    print(f"{'구성':>14}{'피쳐':>5} | " + " | ".join(f"{w[:8]:>30}" for w in WINS))
    print("=" * 122)
    rows = []
    for nm, ks in SETS:
        cols = sorted({c for k in ks for c in G[k]})
        if nm == "BTC동조만": cols = cols + ["dir_up", "T_atr"]
        pred = wf(d[cols].to_numpy(np.float32), y, ts, months, uniq)
        line = f"{nm:>14}{len(cols):>5} | "; rec = dict(cfg=nm, n=len(cols))
        for w in WINS:
            m = np.isfinite(pred) & (sp == w) & okm
            acc = ((pred[m] > 0.5).astype(int) == y[m]).astype(float)
            lo, hi = day_ci(acc, day[m], rng)
            b_ = max(y[m].mean(), 1 - y[m].mean())
            line += (f"{acc.mean():.4f}[{lo:.4f}] A{roc_auc_score(y[m], pred[m]):.3f} "
                     f"기저{b_:.3f}{'✅' if lo > b_ else '  '} | ")
            rec[f"{w}_acc"] = float(acc.mean()); rec[f"{w}_lo"] = lo; rec[f"{w}_base"] = float(b_)
            rec[f"{w}_auc"] = float(roc_auc_score(y[m], pred[m]))
        print(line, flush=True)
        rows.append(rec)
    R = pd.DataFrame(rows); R.to_csv(MY / "h1h_p030_featsets.csv", index=False)
    R["pass3"] = [all(R.loc[i, f"{w}_lo"] > R.loc[i, f"{w}_base"] for w in WINS) for i in R.index]
    R["mAUC"] = R[[f"{w}_auc" for w in WINS]].min(1)
    print("\n" + "=" * 122)
    print(f"⭐세 창 모두 CI 하한 > 기저: {int(R.pass3.sum())}/{len(R)}  "
          f"{R[R.pass3]['cfg'].tolist() if R.pass3.any() else ''}")
    print("\n=== 세 창 최소 AUC 상위 5 (딥러닝에 넘길 후보) ===")
    print(R.sort_values("mAUC", ascending=False).head(5)
          [["cfg", "n"] + [f"{w}_auc" for w in WINS] + [f"{w}_acc" for w in WINS] + ["mAUC"]]
          .round(4).to_string(index=False))
    best = R.sort_values("mAUC", ascending=False).iloc[0]["cfg"]
    cols = sorted({c for k in dict(SETS)[best] for c in G[k]})
    pc = wf(d[cols].to_numpy(np.float32), y, ts, months, uniq, True, np.random.default_rng(3))
    print(f"\n최고 AUC 구성 [{best}] 라벨셔플 대조군: " + " ".join(
        f"{w[:4]} {((pc[np.isfinite(pc)&(sp==w)&okm]>0.5).astype(int)==y[np.isfinite(pc)&(sp==w)&okm]).mean():.4f}"
        for w in WINS))
    json.dump({"best_auc_set": best, "H": H, "P": P}, open(MY / "h1h_p030_best.json", "w"))
    print(json.dumps({"cells": len(R), "pass3": int(R.pass3.sum()), "best": best}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
