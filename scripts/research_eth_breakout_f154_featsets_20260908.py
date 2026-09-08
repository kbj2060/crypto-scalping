#!/usr/bin/env python3
"""H=1시간·±0.30% 에 **기존 154피쳐 엔지니어링 세트**를 붙여 재검정 (2026-09-08).

사용자: *"이전에 쓰던 150여개 피쳐는 안쓰나?"*
→ 안 쓰고 있었다. v4 의 134피쳐는 이번에 직접 만든 것이고, 기존 154세트는 이 과제에 처음 붙인다.

소스 `tmp/ilias_eth_154feature_dataset_extended_20260907/ilias_eth_154feature_2024_2026H1_combined.csv`
(277,010행 × 155컬럼 = 154피쳐 + timestamp · 5분 격자 · 2024-01-01~2026-08-19).
⚠️앵커 키 파일(`features154.parquet`, 4,755행)은 `any3/Wc3` 전용이고 앵커 봉 기준이라 못 쓴다.
   **시계열 원본을 내 피쳐봉 `bt-1`(트리거 분 직전 완결 5분봉)에 샘플링**한다 -- 경계 계약 준수.

가드: 단일피쳐 AUC ≥0.95 이면 중단 · 경계 트립와이어(그룹 제거 시 하락폭) 병기 ·
      셔플 대조군 · 창별 기저(P=0.30% 는 라벨이 되돌림 쪽으로 기운다).
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
F154 = ROOT / "tmp/ilias_eth_154feature_dataset_extended_20260907/ilias_eth_154feature_2024_2026H1_combined.csv"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H, P = 12, 0.0030
EMB = pd.Timedelta(hours=4)
CHUNK, SEED, BOOT = 4000, 20260908, 3000


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
    if len(u) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (Bt, len(u)))
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
    print("[1/4] v4 + 라벨 ...", flush=True)
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
    fb = bt - 1                                          # ⭐피쳐봉
    okm = (s1 > 0) & (s1 + H * 5 < len(ts1)) & (bt + H < len(C5)) & (fb >= 0)
    tu, td = first_touch(hi1, lo1, np.where(okm, s1, 0), entry * (1 + P), entry * (1 - P), H * 5)
    big = 1 << 30
    uo, do_ = tu >= 0, td >= 0
    au, ad = np.where(uo, tu, big), np.where(do_, td, big)
    cont = np.where(sgn > 0, uo & (au < ad), do_ & (ad < au))
    rev = np.where(sgn > 0, do_ & (ad < au), uo & (au < ad))
    x5 = np.minimum(bt + H, len(C5) - 1)
    clo = (C5[x5] - entry) / entry * 1e4 * sgn
    y = np.where(cont, 1, np.where(rev, 0, (clo > 0).astype(int)))

    print("[2/4] 154피쳐 시계열 로드 · bt-1 샘플링 ...", flush=True)
    F = pd.read_csv(F154, parse_dates=["timestamp"])
    F = F.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    fcols = [c for c in F.columns if c != "timestamp"]
    pos = pd.Index(F["timestamp"]).get_indexer(pd.DatetimeIndex(ts5[np.clip(fb, 0, len(ts5) - 1)]))
    A = F[fcols].to_numpy(np.float32)
    Xf = np.where((pos >= 0)[:, None], A[np.clip(pos, 0, len(A) - 1)], np.nan)
    for j, c in enumerate(fcols): d[f"i_{c}"] = Xf[:, j]
    cov = (pos >= 0).mean()
    print(f"      154피쳐 {len(fcols)}개 · 결합률 {cov:.1%}", flush=True)

    print("[3/4] 누수 가드 (단일피쳐 AUC) ...", flush=True)
    sp = d["split"].to_numpy(); tr0 = (sp == "TRAIN") & okm
    bad = []
    for c in fcols:
        v = d[f"i_{c}"].to_numpy(float)[tr0]; yy = y[tr0]; m = np.isfinite(v)
        if m.sum() < 500 or len(np.unique(v[m])) < 3: continue
        a = roc_auc_score(yy[m], v[m])
        if max(a, 1 - a) >= 0.95: bad.append((c, round(max(a, 1 - a), 4)))
    if bad:
        print(f"🔴누수 의심 {bad} -- 중단"); return 1
    print("      통과 (단일피쳐 AUC 최대 "
          f"{max(max(roc_auc_score(y[tr0][np.isfinite(d[f'i_{c}'].to_numpy(float)[tr0])], d[f'i_{c}'].to_numpy(float)[tr0][np.isfinite(d[f'i_{c}'].to_numpy(float)[tr0])]), 1-roc_auc_score(y[tr0][np.isfinite(d[f'i_{c}'].to_numpy(float)[tr0])], d[f'i_{c}'].to_numpy(float)[tr0][np.isfinite(d[f'i_{c}'].to_numpy(float)[tr0])])) for c in fcols[:40]):.3f} 표본)", flush=True)

    ts = d["timestamp"]; day = ts.dt.floor("D").to_numpy()
    months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    G = {
        "v1봉": [c for c in d.columns if c.startswith(("f_", "sig_"))] +
                ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"],
        "경로": [c for c in d.columns if c.startswith("v2_mv_") and not any(
            k in c for k in ("btc", "idio", "same_sign"))],
        "레벨": [c for c in d.columns if c.startswith(("v2_brk", "v2_pos"))],
        "BTC": ["v2_mv_btc_ret_atr", "v2_mv_idio_atr", "v2_mv_same_sign"],
        "포지셔닝": [c for c in d.columns if c.startswith(("x_m_tkv", "x_m_ttp", "x_m_ttc", "x_m_retail"))],
        "테이프": [c for c in d.columns if c.startswith(("t_", "mv_")) and not c.startswith("t_obi")],
        "OBI": [c for c in d.columns if c.startswith("t_obi")],
        "f154": [f"i_{c}" for c in fcols],
    }
    print(f"\n[4/4] 절제 · f154 {len(G['f154'])}개\n" + "=" * 122, flush=True)
    print(f"{'구성':>16}{'피쳐':>5} | " + " | ".join(f"{w[:8]:>30}" for w in WINS))
    print("=" * 122)
    SETS = [("f154만", ["f154"]), ("v1봉만", ["v1봉"]), ("BTC동조만", ["BTC"]),
            ("f154+BTC", ["f154", "BTC"]),
            ("기준(내것)", ["v1봉", "경로", "레벨", "BTC", "포지셔닝"]),
            ("기준+f154", ["v1봉", "경로", "레벨", "BTC", "포지셔닝", "f154"]),
            ("기준+OBI", ["v1봉", "경로", "레벨", "BTC", "포지셔닝", "OBI"]),
            ("전체+f154", list(G))]
    rows = []
    for nm, ks in SETS:
        cols = sorted({c for k in ks for c in G[k]})
        if nm == "BTC동조만": cols = cols + ["dir_up", "T_atr"]
        pred = wf(d[cols].to_numpy(np.float32), y, ts, months, uniq)
        line = f"{nm:>16}{len(cols):>5} | "; rec = dict(cfg=nm, n=len(cols))
        for w in WINS:
            m = np.isfinite(pred) & (sp == w) & okm
            acc = ((pred[m] > 0.5).astype(int) == y[m]).astype(float)
            lo, hi = day_ci(acc, day[m], rng)
            b_ = max(y[m].mean(), 1 - y[m].mean())
            a_ = roc_auc_score(y[m], pred[m])
            line += f"{acc.mean():.4f}[{lo:.4f}] A{a_:.3f} 기저{b_:.3f}{'✅' if lo > b_ else '  '} | "
            rec[f"{w}_acc"] = float(acc.mean()); rec[f"{w}_lo"] = lo
            rec[f"{w}_base"] = float(b_); rec[f"{w}_auc"] = float(a_)
        print(line, flush=True)
        rows.append(rec)
    R = pd.DataFrame(rows); R.to_csv(MY / "f154_featsets.csv", index=False)
    R["pass3"] = [all(R.loc[i, f"{w}_lo"] > R.loc[i, f"{w}_base"] for w in WINS) for i in R.index]
    R["mAUC"] = R[[f"{w}_auc" for w in WINS]].min(1)
    print("\n" + "=" * 122)
    print(f"⭐세 창 모두 CI 하한 > 기저: {int(R.pass3.sum())}/{len(R)}")
    print("\n=== 세 창 최소 AUC 순위 ===")
    print(R.sort_values("mAUC", ascending=False)[["cfg", "n"] + [f"{w}_auc" for w in WINS] + ["mAUC"]]
          .round(4).to_string(index=False))
    best = R.sort_values("mAUC", ascending=False).iloc[0]["cfg"]
    cols = sorted({c for k in dict(SETS)[best] for c in G[k]})
    pc = wf(d[cols].to_numpy(np.float32), y, ts, months, uniq, True, np.random.default_rng(5))
    print(f"\n최고 [{best}] 셔플 대조군: " + " ".join(
        f"{w[:4]} {((pc[np.isfinite(pc)&(sp==w)&okm]>0.5).astype(int)==y[np.isfinite(pc)&(sp==w)&okm]).mean():.4f}"
        for w in WINS))
    d.to_parquet(MY / "dataset_v5_f154.parquet")
    json.dump({"best": best, "f154_n": len(fcols), "cov": float(cov)},
              open(MY / "f154_best.json", "w"), ensure_ascii=False)
    print(json.dumps({"best": best, "pass3": int(R.pass3.sum())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
