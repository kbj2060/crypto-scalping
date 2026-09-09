#!/usr/bin/env python3
"""배리어를 **ATR 상대**로 바꾸면 저변동성 구간도 살아나는가 (2026-09-08).

사용자: *"atr을 절대적 수치 말고 상대적 수치로 재학습하면 변동성이 없는 구간에서도
        정확도가 오를 수 있지 않을까?"*

## 진단이 가리키는 곳은 피쳐가 아니라 **라벨**이다
트리거는 이미 ATR 상대다(T = 0.75×ATR). 상대가 아닌 건 **배리어 ±0.25%(절대)** 하나뿐이고,
그래서 ATR 구간마다 질문의 난이도가 딴판이다:
    ATR 0.00~0.155%  배리어/ATR 2.33×  시간청산 25.3%  돌파율 0.515
    ATR ≥0.404%      배리어/ATR 0.46×  시간청산  0.0%  돌파율 **0.334**
고ATR 의 "정확도 69.1%"는 대부분 **클래스 불균형**이다(셔플 귀무 66.1%, 초과 +3.0pp).
저ATR 은 정확도 54.6% 로 초라하지만 귀무 48.5% 라 **초과 +6.1pp 로 최고**다.

## 실험
배리어를 P = k × ATR 로 바꿔 재라벨·재학습한다. k ∈ {0.6, 0.8, 1.0, 1.25}.
(현행 고정 0.25% 는 중앙 ATR 기준 약 0.93×ATR 에 해당한다.)
기대: 돌파율이 전 구간에서 균등해지고, 확신도가 불균형이 아니라 실력을 반영하게 된다.

판정: **ATR 구간별 초과분(관측−셔플귀무)**이 고르게 양수인가. 원시 정확도는 보지 않는다.
"""
from __future__ import annotations
import os, sys, json
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "8")
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

MY = ROOT / "tmp/eth_breakout_atr_state_20260908_s1"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H, TM, NB, CHUNK, SEED = 12, 0.75, 32, 4000, 20260908
EMB = pd.Timedelta(hours=4)
KGRID = (0.6, 0.8, 1.0, 1.25)
ABS_P = 0.0025          # 현행(대조군)


def first_touch(hi1, lo1, start, up, dn, nmin):
    n = len(start); tu = np.full(n, -1, np.int32); td = np.full(n, -1, np.int32)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        ix = start[a:b, None] + np.arange(nmin)[None, :]
        hu = hi1[ix] >= up[a:b, None]; hd = lo1[ix] <= dn[a:b, None]
        tu[a:b] = np.where(hu.any(axis=1), hu.argmax(axis=1), -1)
        td[a:b] = np.where(hd.any(axis=1), hd.argmax(axis=1), -1)
    return tu, td


def main() -> int:
    from sklearn.ensemble import HistGradientBoostingClassifier
    d = pd.read_parquet(MY / "dataset_v2.parquet")
    d = d[(d.anchor == "first_fire") & (d.T_mult == TM)].sort_values("timestamp").reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    eth = B._load_kl(B.ETH_KL)
    ts5 = eth["timestamp"].to_numpy(); O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy(); hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float)
    sgn = np.where(d["dir_up"].to_numpy() > 0, 1.0, -1.0)
    ba = d["bar_idx"].to_numpy(); T = d["T_atr"].to_numpy(float)
    atr = d["atr_at_anchor"].to_numpy(float)
    entry = O5[np.minimum(ba + 1, len(O5) - 1)] * (1 + sgn * T)
    s1 = np.searchsorted(ts1, d["timestamp"].to_numpy())
    bt = np.searchsorted(ts5, ts1[np.clip(s1, 0, len(ts1) - 1)], side="right") - 1
    okm = (s1 > 0) & (s1 + H * 5 < len(ts1)) & (bt + H < len(C5)) & (bt >= 1)
    sp = d["split"].to_numpy(); ts = d["timestamp"]
    months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    base = [c for c in d.columns if c.startswith(("f_", "sig_", "v2_"))] + \
           [c for c in d.columns if c.startswith(("x_m_tkv", "x_m_ttp", "x_m_ttc", "x_m_retail"))] + \
           ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"]
    X = np.nan_to_num(d[base].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ev = (sp != "TRAIN") & okm
    qs = np.nanquantile(atr[ev], [0.2, 0.4, 0.6, 0.8]); edges = [0.0, *qs, 1.0]
    big = 1 << 30

    def label(P):
        tu, td = first_touch(hi1, lo1, np.where(okm, s1, 0), entry * (1 + P), entry * (1 - P), H * 5)
        uo, do_ = tu >= 0, td >= 0
        au, ad = np.where(uo, tu, big), np.where(do_, td, big)
        cont = np.where(sgn > 0, uo & (au < ad), do_ & (ad < au))
        rev = np.where(sgn > 0, do_ & (ad < au), uo & (au < ad))
        clo = (C5[np.minimum(bt + H, len(C5) - 1)] - entry) / entry * 1e4 * sgn
        none = ~(cont | rev)
        return np.where(cont, 1, np.where(rev, 0, (clo > 0).astype(int))), none

    def wf(y, shuf=False, rs=None):
        pred = np.full(len(y), np.nan)
        for i, mo in enumerate(uniq):
            if i < 5: continue
            te = (months == mo).to_numpy(); tr = (ts < ts[te].min() - EMB).to_numpy()
            if tr.sum() < 1500 or te.sum() < 30: continue
            itr = np.flatnonzero(tr); yy = y.copy()
            if shuf: yy[itr] = rs.permutation(yy[itr])
            c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                               l2_regularization=1.0, early_stopping=True,
                                               validation_fraction=0.15, random_state=SEED)
            pred[te] = c.fit(X[itr], yy[itr]).predict_proba(X[te])[:, 1]
        return pred

    ARMS = [("현행 절대 0.25%", None)] + [(f"ATR상대 {k}×", k) for k in KGRID]
    out = []
    for tag, k in ARMS:
        P = np.full(len(d), ABS_P) if k is None else atr * k
        y, none = label(P)
        pred = wf(y)
        nulls = [wf(y, True, np.random.default_rng(6600 + b)) for b in range(NB)]
        print(f"\n### {tag} · 돌파율 {y[ev].mean():.3f} · 시간청산 {none[ev].mean()*100:.1f}%", flush=True)
        print(f"{'ATR 구간':>18}{'n':>7}{'돌파율':>8}{'시청':>7}{'확신중앙':>10}{'강/일':>7}"
              f"{'정확도':>8}{'귀무':>8}{'초과':>8}")
        rows = []
        for i in range(5):
            lo_, hi_ = edges[i], edges[i + 1]
            m = ev & (atr >= lo_) & (atr < hi_) & np.isfinite(pred)
            if m.sum() < 50: continue
            acc = ((pred[m] > 0.5).astype(int) == y[m]).mean()
            nb = np.mean([((n[m] > 0.5).astype(int) == y[m]).mean() for n in nulls])
            conf = np.abs(pred[m] - 0.5)
            rng = f"{lo_*100:.3f}~{hi_*100:.3f}%" if hi_ < 1 else f"≥{lo_*100:.3f}%"
            mk = " ⭐" if lo_ <= 0.00156 < hi_ else ""
            print(f"{rng:>18}{m.sum():>7}{y[m].mean():>8.3f}{none[m].mean()*100:>6.1f}%"
                  f"{np.median(conf):>10.4f}{(conf>=0.1104).sum()/334:>7.1f}"
                  f"{acc*100:>7.1f}%{nb*100:>7.1f}%{(acc-nb)*100:>+7.1f}{mk}", flush=True)
            rows.append(dict(arm=tag, k=k, lo=lo_, n=int(m.sum()), up=float(y[m].mean()),
                             acc=float(acc), null=float(nb), exc=float((acc-nb)*100),
                             conf_med=float(np.median(conf))))
        # 세 창 전체
        cells = []
        for w in WINS:
            m = ev & (sp == w) & np.isfinite(pred)
            acc = ((pred[m] > 0.5).astype(int) == y[m]).mean()
            nb = np.mean([((n[m] > 0.5).astype(int) == y[m]).mean() for n in nulls])
            cells.append((acc, nb, (acc-nb)*100))
        print("   전체: " + " · ".join(f"{w[:4]} {a:.4f}(귀무 {b:.4f}, {e:+.1f}pp)"
                                       for w, (a, b, e) in zip(WINS, cells)))
        print(f"   ⭐구간별 초과 최소 {min(r['exc'] for r in rows):+.1f}pp · "
              f"표준편차 {np.std([r['exc'] for r in rows]):.2f}pp (낮을수록 레짐 무관)")
        out += rows
    pd.DataFrame(out).to_csv(MY / "atr_relative_barrier.csv", index=False)
    print(json.dumps({"done": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
