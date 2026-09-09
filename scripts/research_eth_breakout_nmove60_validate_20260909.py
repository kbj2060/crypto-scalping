#!/usr/bin/env python3
"""발현창 60분 단독 검정 -- 셔플 귀무(B=32) + 무작위 시드 20개 (2026-09-09, 사용자 채택).

발현창을 15분 -> 60분으로 넓히면 앵커 커버리지가 88.6% -> 99.3% 로 오른다. 정확도는 짝비교에서
시드 폭 안이었지만(같은 사건 -0.04/+0.45/+0.79pp), **모집단이 바뀌었으므로 사전등록 기대치를
다시 잰다** -- 15분판 숫자를 그대로 물려쓰면 다른 질문의 답을 쓰는 것이다.

현행(15분)과 **같은 절차·같은 시드 리스트**를 써서 직접 비교한다.

판정(CLAUDE.md 시드 게이트):
  ① 부호  20개 시드가 전부 셔플 귀무 위인가 (하나라도 아래면 FAIL)
  ② 폭    시드간 폭이 초과분의 50% 를 넘으면 크기 주장 불가(FLAG)
  ③ 세 창 동시 통과 비율
추가: ATR 구간별 초과가 고르게 양수인가(레짐 무관성 -- 이 교체의 존재 이유).
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

MY = ROOT / "tmp/eth_breakout_nmove_20260909"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H, TM, KATR, NB, NSEED = 12, 0.75, 0.8, 32, 20
COV = (1.0, 0.5)
EMB = pd.Timedelta(hours=4); CHUNK, SEED = 4000, 20260908
SEEDS = [106645, 168524, 199020, 213181, 218414, 305610, 372903, 400560, 409299, 517758,
         555682, 638019, 642546, 648434, 684832, 761154, 874811, 912929, 913878, 961476]


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
    d = pd.read_parquet(MY / "dataset_nm60.parquet")     # 발현창 60분 모집단
    d = d.sort_values("timestamp").reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    eth = B._load_kl(B.ETH_KL)
    ts5 = eth["timestamp"].to_numpy(); O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy(); hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float)
    sgn = np.where(d["dir_up"].to_numpy() > 0, 1.0, -1.0)
    ba = d["bar_idx"].to_numpy(); T = d["T_atr"].to_numpy(float); atr = d["atr_at_anchor"].to_numpy(float)
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
    trm = sp == "TRAIN"; cut = ts[trm].quantile(0.7)
    tr_fit = np.flatnonzero(trm & (ts <= cut).to_numpy()); tr_hold = trm & (ts > cut).to_numpy()

    P = atr * KATR
    tu, td = first_touch(hi1, lo1, np.where(okm, s1, 0), entry * (1 + P), entry * (1 - P), H * 5)
    big = 1 << 30
    uo, do_ = tu >= 0, td >= 0
    au, ad = np.where(uo, tu, big), np.where(do_, td, big)
    cont = np.where(sgn > 0, uo & (au < ad), do_ & (ad < au))
    rev = np.where(sgn > 0, do_ & (ad < au), uo & (au < ad))
    clo = (C5[np.minimum(bt + H, len(C5) - 1)] - entry) / entry * 1e4 * sgn
    y = np.where(cont, 1, np.where(rev, 0, (clo > 0).astype(int)))
    ev = (sp != "TRAIN") & okm
    print(f"배리어 {KATR}×ATR (중앙 {np.median(P[ev])*100:.3f}%) · 사건 {int(ev.sum()):,} "
          f"· 돌파율 {y[ev].mean():.4f} · 시간청산 {(~(cont|rev))[ev].mean()*100:.1f}%\n", flush=True)

    def run(sd, shuf=False, rs=None):
        pred = np.full(len(y), np.nan)
        for i, mo in enumerate(uniq):
            if i < 5: continue
            te = (months == mo).to_numpy(); tr = (ts < ts[te].min() - EMB).to_numpy()
            if tr.sum() < 1500 or te.sum() < 30: continue
            itr = np.flatnonzero(tr); yy = y.copy()
            if shuf: yy[itr] = rs.permutation(yy[itr])
            c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                               l2_regularization=1.0, early_stopping=True,
                                               validation_fraction=0.15, random_state=sd)
            pred[te] = c.fit(X[itr], yy[itr]).predict_proba(X[te])[:, 1]
        yf = y.copy()
        if shuf: yf[tr_fit] = rs.permutation(yf[tr_fit])
        c0 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                            l2_regularization=1.0, early_stopping=True,
                                            validation_fraction=0.15, random_state=sd)
        return pred, np.abs(c0.fit(X[tr_fit], yf[tr_fit]).predict_proba(X[tr_hold])[:, 1] - 0.5)

    def score(pred, conf):
        o = {}
        for cv in COV:
            thr = 0.0 if cv >= 1.0 else float(np.quantile(conf, 1 - cv))
            for w in WINS:
                m = np.isfinite(pred) & (sp == w) & okm & (np.abs(pred - 0.5) >= thr)
                o[(cv, w)] = (float(((pred[m] > 0.5).astype(int) == y[m]).mean()) if m.sum() >= 30 else np.nan,
                              int(m.sum()))
        return o

    print("셔플 귀무 B=32 …", flush=True)
    O = score(*run(SEED))
    NUL = {k: [] for k in O}
    for b in range(NB):
        S = score(*run(SEED, True, np.random.default_rng(5100 + b)))
        for k in O: NUL[k].append(S[k][0])
    print("=" * 92)
    print(f"{'커버':>6}{'창':>16}{'관측':>9}{'귀무평균':>10}{'귀무max':>9}{'p':>7}{'σ':>7}{'n':>7}")
    NM = {}
    for cv in COV:
        for w in WINS:
            o, n = O[(cv, w)]; a = np.array([x for x in NUL[(cv, w)] if np.isfinite(x)])
            p_ = ((a >= o).sum() + 1) / (len(a) + 1); NM[(cv, w)] = float(a.mean())
            print(f"{int(cv*100):>5}%{w:>16}{o:>9.4f}{a.mean():>10.4f}{a.max():>9.4f}{p_:>7.3f}"
                  f"{(o-a.mean())/max(a.std(ddof=1),1e-9):>7.1f}{n:>7}")

    print(f"\n무작위 시드 {NSEED}개 …", flush=True)
    rows = []
    for sd in SEEDS:
        s = score(*run(sd)); rows.append({(cv, w): s[(cv, w)][0] for cv in COV for w in WINS})
    print("=" * 100)
    print(f"{'커버':>6}{'창':>16}{'평균':>9}{'sd':>8}{'min':>8}{'max':>8}{'귀무':>9}{'초과':>9}  판정")
    verdict = []
    for cv in COV:
        for w in WINS:
            a = np.array([r[(cv, w)] for r in rows]); nm = NM[(cv, w)]
            exc = a.mean() - nm; span = a.max() - a.min(); bad = int((a <= nm).sum())
            v = f"FAIL({bad}/{len(a)})" if bad else ("FLAG" if span / abs(exc) > 0.5 else "OK")
            verdict.append(v)
            print(f"{int(cv*100):>5}%{w:>16}{a.mean():>9.4f}{a.std(ddof=1):>8.4f}{a.min():>8.4f}"
                  f"{a.max():>8.4f}{nm:>9.4f}{exc*100:>+8.2f}pp  {v}")
        ok = np.ones(len(rows), bool)
        for w in WINS: ok &= np.array([r[(cv, w)] for r in rows]) > NM[(cv, w)]
        print(f"{'':>6}{'세 창 동시':>16} → {ok.sum()}/{len(rows)} 시드\n")
    print(f"⭐OK {sum(v=='OK' for v in verdict)} · FLAG {sum(v=='FLAG' for v in verdict)} · "
          f"FAIL {sum('FAIL' in v for v in verdict)}")
    json.dump({"k": KATR, "null": {f"{int(c*100)}_{w}": NM[(c, w)] for c in COV for w in WINS},
               "seeds": SEEDS, "fail": sum('FAIL' in v for v in verdict)},
              open(MY / "nmove60_validate.json", "w"))
    # 아티팩트에 그대로 넣을 PREREG 블록을 출력한다(손으로 옮겨 적지 않기 위해).
    obs = {cv: {w: float(np.mean([r[(cv, w)] for r in rows])) for w in WINS} for cv in COV}
    evn = int(((sp != "TRAIN") & okm).sum())
    m_ev = (sp != "TRAIN") & okm
    dd = (ts[m_ev].max() - ts[m_ev].min()).total_seconds() / 86400.0
    pre = {f"cov{int(cv*100)}": {"acc": {w: round(obs[cv][w], 4) for w in WINS},
                                 "null": {w: round(NM[(cv, w)], 4) for w in WINS},
                                 "per_day": round(int(m_ev.sum()) * cv / dd, 1)}
           for cv in COV}
    print("\nPREREG = " + json.dumps(pre, ensure_ascii=False, indent=4))
    json.dump(pre, open(MY / "nmove60_prereg.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({"done": True, "fail": sum('FAIL' in v for v in verdict)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
