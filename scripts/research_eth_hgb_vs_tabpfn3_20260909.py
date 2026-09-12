#!/usr/bin/env python3
"""현행 HGB vs TabPFN v3 -- 앵커 돌파/되돌림 · 극점 탐지기 (2026-09-09, 사용자 요청).

## 왜 지금 다시 재나
돌파/되돌림의 "모델 축 종결"(2026-09-08)은 **v4 데이터셋(T=1.0)** 에서 잰 것이고, 극점 탐지기는
어제 만들어져 비교 자체가 없다. 둘 다 **배포된 구성**에서 TabPFN v3 와 붙여본 적이 없다.

## 세 팔 -- 모델과 표본크기를 가른다
    (1) HGB(전체)      현행 운영. 학습셋 전부를 쓴다.
    (2) HGB(부분표집)   TabPFN 과 **같은 컨텍스트**만 준다.  ⭐(1)-(2) 는 표본크기 효과
    (3) TabPFN v3      in-context. 컨텍스트 상한 때문에 부분표집이 강제된다.
⭐(2)-(3) 이 모델 효과다. 이 분리를 안 하면 "TabPFN 이 졌다"가 "표본이 작아 졌다"와 구분이 안 된다
  (저장소 기록: GBM 프록시는 학습셋이 TabPFN 상한을 넘으면 개선폭이 소멸했다).

## 판정
시드 5개. 차이가 **시드 폭 안이면 무승부**로 읽는다 -- 오늘만 두 번(긴칸 f154, 지연 등급) 이
가드에서 갈렸다. 창별로 잰다(합산 기저는 부풀린다).

⚠️서버 GPU(RTX 3070 Ti)는 대시보드의 TabPFN 재적합과 공유된다. 이 스크립트가 도는 동안
   대시보드가 느려질 수 있다(기록: 3초 -> 43초).
"""
from __future__ import annotations
import os, sys, json, time, argparse
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "8")
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

NM = ROOT / "tmp/eth_breakout_nmove_20260909"
EX = ROOT / "tmp/eth_signal_map_20260909"
OUT = ROOT / "tmp/eth_model_compare_20260909"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H, K, CHUNK, big = 12, 0.8, 4000, 1 << 30
SEEDS = (11, 22, 33, 44, 55)
EMB = pd.Timedelta(hours=4)


def hgb(seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.15, random_state=seed)


def tabpfn(seed, dev):
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(device=dev, random_state=seed, n_estimators=1)


def sub_idx(idx, cap, rng):
    """컨텍스트 상한 부분표집. 시간 순서를 유지한다(최근 편향을 인위로 넣지 않는다)."""
    if len(idx) <= cap:
        return idx
    return np.sort(rng.choice(idx, cap, replace=False))


def day_ci(v, day, rng, Bn=2000):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (Bn, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


# ------------------------------------------------------------------ A. 돌파/되돌림
def problem_breakout(cap, dev, rng):
    d = pd.read_parquet(NM / "dataset_nm15.parquet")     # 배포 구성(발현창 15분)
    d["timestamp"] = pd.to_datetime(d["timestamp"]); d = d.sort_values("timestamp").reset_index(drop=True)
    eth = B._load_kl(B.ETH_KL); C5 = eth["close"].to_numpy(float)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy(); hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float)
    feats = [c for c in d.columns if c.startswith(("f_", "sig_", "v2_"))] + \
            [c for c in d.columns if c.startswith(("x_m_tkv", "x_m_ttp", "x_m_ttc", "x_m_retail"))] + \
            ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"]
    X = np.nan_to_num(d[feats].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ts = d["timestamp"]; sp = d["split"].to_numpy(); day = ts.dt.floor("D").to_numpy()
    months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    s1 = d["s1"].to_numpy(); bt = d["bt"].to_numpy(); entry = d["entry_px"].to_numpy()
    sgn = np.where(d["dir_up"].to_numpy() > 0, 1.0, -1.0); P = d["atr_at_anchor"].to_numpy() * K
    okm = (s1 + H * 5 < len(ts1)) & (bt + H < len(C5))
    n = len(s1); tu = np.full(n, -1, np.int32); td = np.full(n, -1, np.int32)
    st = np.where(okm, s1, 0); up = entry * (1 + P); dn = entry * (1 - P)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n); ix = st[a:b, None] + np.arange(H * 5)[None, :]
        hu = hi1[ix] >= up[a:b, None]; hd = lo1[ix] <= dn[a:b, None]
        tu[a:b] = np.where(hu.any(1), hu.argmax(1), -1); td[a:b] = np.where(hd.any(1), hd.argmax(1), -1)
    au = np.where(tu >= 0, tu, big); ad = np.where(td >= 0, td, big)
    cont = np.where(sgn > 0, (tu >= 0) & (au < ad), (td >= 0) & (ad < au))
    rev = np.where(sgn > 0, (td >= 0) & (ad < au), (tu >= 0) & (au < ad))
    clo = (C5[np.minimum(bt + H, len(C5) - 1)] - entry) / entry * 1e4 * sgn
    y = np.where(cont, 1, np.where(rev, 0, (clo > 0).astype(int)))
    print(f"[돌파/되돌림] 사건 {int(okm.sum()):,} · 피쳐 {len(feats)} · 돌파율 {y[okm].mean():.4f}", flush=True)

    def wf(kind, seed):
        pred = np.full(len(y), np.nan); ntr = []
        for i, mo in enumerate(uniq):
            if i < 5: continue
            te = (months == mo).to_numpy(); tr = (ts < ts[te].min() - EMB).to_numpy()
            if tr.sum() < 1500 or te.sum() < 30: continue
            itr = np.flatnonzero(tr)
            if kind != "hgb_full":
                itr = sub_idx(itr, cap, np.random.default_rng(seed * 1000 + i))
            ntr.append(len(itr))
            m = hgb(seed) if kind.startswith("hgb") else tabpfn(seed, dev)
            m.fit(X[itr], y[itr]); pred[te] = m.predict_proba(X[te])[:, 1]
        return pred, int(np.median(ntr)) if ntr else 0

    from sklearn.metrics import roc_auc_score
    rows = []
    for kind, lab in (("hgb_full", "HGB(전체)"), ("hgb_sub", f"HGB(부분표집 {cap:,})"),
                      ("tabpfn", f"TabPFN v3({cap:,})")):
        for seed in SEEDS:
            t0 = time.time(); pred, med = wf(kind, seed); dt = time.time() - t0
            for w in WINS:
                m = np.isfinite(pred) & (sp == w) & okm
                b_ = max(y[m].mean(), 1 - y[m].mean())
                acc = ((pred[m] > .5) == y[m]).astype(float)
                rows.append(dict(problem="돌파/되돌림", arm=lab, seed=seed, win=w,
                                 acc=float(acc.mean()), base=float(b_),
                                 exc=float(acc.mean() - b_) * 100,
                                 auc=float(roc_auc_score(y[m], pred[m])), n=int(m.sum()),
                                 n_train=med, sec=round(dt, 1)))
            print(f"   {lab:22} 시드 {seed} · 학습 중앙 {med:,}행 · {dt:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "compare.csv", index=False)
    return rows


# ------------------------------------------------------------------ B. 극점 탐지기
def problem_extreme(cap, dev, rng):
    A = pd.read_parquet(EX / "extreme_frame.parquet")
    meta = json.load(open(EX / "extreme_frame_meta.json"))
    feats = meta["feats"]; VAL0 = pd.Timestamp(meta["val0"])
    A = A[A["_y"] >= 0].sort_values("_ts").reset_index(drop=True)
    X = np.nan_to_num(A[feats].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = A["_y"].to_numpy(int); tsx = A["_ts"]
    tr = (tsx < VAL0).to_numpy(); te = ~tr
    print(f"[극점 탐지기] 학습 {tr.sum():,} · 표본외 {te.sum():,} · 피쳐 {len(feats)} "
          f"· 기저 {y[te].mean():.4f}", flush=True)
    from sklearn.metrics import roc_auc_score
    rows = []
    for kind, lab in (("hgb_full", "HGB(전체)"), ("hgb_sub", f"HGB(부분표집 {cap:,})"),
                      ("tabpfn", f"TabPFN v3({cap:,})")):
        for seed in SEEDS:
            itr = np.flatnonzero(tr)
            if kind != "hgb_full":
                itr = sub_idx(itr, cap, np.random.default_rng(seed))
            t0 = time.time()
            m = hgb(seed) if kind.startswith("hgb") else tabpfn(seed, dev)
            m.fit(X[itr], y[itr]); p = m.predict_proba(X[te])[:, 1]
            dt = time.time() - t0
            auc = roc_auc_score(y[te], p)
            # 배포 운영점: 표본외 상위 10% 정밀도
            thr = np.quantile(p, 0.90); sel = p >= thr
            rows.append(dict(problem="극점", arm=lab, seed=seed, win="OOS",
                             auc=float(auc), prec10=float(y[te][sel].mean()),
                             base=float(y[te].mean()), n=int(te.sum()),
                             n_train=len(itr), sec=round(dt, 1)))
            print(f"   {lab:22} 시드 {seed} · 학습 {len(itr):,}행 · AUC {auc:.4f} "
                  f"· 상위10% 정밀도 {y[te][sel].mean():.4f} · {dt:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "compare_extreme.csv", index=False)
    return rows


# ------------------------------------------------------------------ B'. 극점 -- 워크포워드
def problem_extreme_wf(cap, dev, rng):
    """단일 분할 결과를 **월별 확장 워크포워드**로 다시 확인한다.

    단일 분할은 창을 하나만 본다 -- 그 창이 유난히 TabPFN 에 유리했을 수 있다.
    같은 프레임을 월별로 재학습하며 전 구간을 훑으면 그 가능성이 걸러진다.
    엠바고 4시간(라벨이 앞으로 12봉=1시간을 보므로 충분).
    """
    from sklearn.metrics import roc_auc_score
    A = pd.read_parquet(EX / "extreme_frame.parquet")
    meta = json.load(open(EX / "extreme_frame_meta.json"))
    feats = meta["feats"]; VAL0 = pd.Timestamp(meta["val0"])
    A = A[A["_y"] >= 0].sort_values("_ts").reset_index(drop=True)
    X = np.nan_to_num(A[feats].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = A["_y"].to_numpy(int); ts = A["_ts"]
    months = ts.dt.to_period("M"); uniq = sorted(months.unique())
    print(f"[극점 워크포워드] {len(A):,}행 · {uniq[0]}~{uniq[-1]} ({len(uniq)}개월) "
          f"· 워밍업 5개월 · 기저 {y.mean():.4f}", flush=True)
    rows = []
    for kind, lab in (("hgb_full", "HGB(전체)"), ("hgb_sub", f"HGB(부분표집 {cap:,})"),
                      ("tabpfn", f"TabPFN v3({cap:,})")):
        for seed in SEEDS:
            t0 = time.time(); pred = np.full(len(y), np.nan); ntr = []
            for i, mo in enumerate(uniq):
                if i < 5: continue
                te = (months == mo).to_numpy(); tr = (ts < ts[te].min() - EMB).to_numpy()
                if tr.sum() < 1500 or te.sum() < 100: continue
                itr = np.flatnonzero(tr)
                if kind != "hgb_full":
                    itr = sub_idx(itr, cap, np.random.default_rng(seed * 1000 + i))
                ntr.append(len(itr))
                m = hgb(seed) if kind.startswith("hgb") else tabpfn(seed, dev)
                m.fit(X[itr], y[itr]); pred[te] = m.predict_proba(X[te])[:, 1]
            dt = time.time() - t0
            for wlab, msk in (("전체", np.isfinite(pred)),
                              ("~2026-03", np.isfinite(pred) & (ts < VAL0).to_numpy()),
                              ("2026-04~", np.isfinite(pred) & (ts >= VAL0).to_numpy())):
                if msk.sum() < 300: continue
                thr = np.quantile(pred[msk], 0.90)
                rows.append(dict(problem="극점WF", arm=lab, seed=seed, win=wlab,
                                 auc=float(roc_auc_score(y[msk], pred[msk])),
                                 prec10=float(y[msk][pred[msk] >= thr].mean()),
                                 base=float(y[msk].mean()), n=int(msk.sum()),
                                 n_train=int(np.median(ntr)) if ntr else 0, sec=round(dt, 1)))
            print(f"   {lab:22} 시드 {seed} · 학습 중앙 {int(np.median(ntr)):,}행 · {dt:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "compare_extreme_wf.csv", index=False)
    return rows


def summarize(df, key, label):
    print("\n" + "=" * 104); print(label); print("=" * 104)
    for w in sorted(df["win"].unique()):
        q = df[df.win == w]
        print(f"  [{w}]")
        for arm in q["arm"].unique():
            a = q[q.arm == arm][key].to_numpy()
            print(f"     {arm:24} 평균 {a.mean():.4f}  [{a.min():.4f}~{a.max():.4f}]  "
                  f"시드 폭 {np.ptp(a):.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=10000, help="TabPFN 컨텍스트 상한(부분표집 크기)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", choices=["breakout", "extreme", "extreme_wf"], default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260909)
    if a.only != "extreme":
        r = problem_breakout(a.cap, a.device, rng)
        summarize(pd.DataFrame(r), "exc", "A. 앵커 돌파/되돌림 -- 기저 대비 초과(pp)")
    if a.only == "extreme_wf":
        r = problem_extreme_wf(a.cap, a.device, rng)
        summarize(pd.DataFrame(r), "auc", "B'. 극점 워크포워드 -- AUC")
        summarize(pd.DataFrame(r), "prec10", "B'. 극점 워크포워드 -- 상위 10% 정밀도")
        print("\n저장:", OUT); return 0
    if a.only != "breakout":
        r = problem_extreme(a.cap, a.device, rng)
        summarize(pd.DataFrame(r), "auc", "B. 극점 탐지기 -- 표본외 AUC")
        summarize(pd.DataFrame(r), "prec10", "B. 극점 탐지기 -- 상위 10% 정밀도")
    print("\n저장:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
