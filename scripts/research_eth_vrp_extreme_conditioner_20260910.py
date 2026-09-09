#!/usr/bin/env python3
"""**새 신호** 후보 3 — VRP 를 극점 탐지기의 **조건화 변수**로 (2026-09-10).

앞선 두 검정에서:
  · VRP 방향 규칙 → 기각(일군집 CI 0 포함, 이득이 6월 한 달)
  · 변동성 확장 분류 → 건수 맞추면 증분 0 (OOS 정밀도 .600 vs .591)
  · 변동성 회귀 증분만 생존 (TRAIN t=2.46 · 0 배제, OOS 부호 같으나 검정력 부족)

남은 형태는 **조건화**다. 이 대시보드에서 비용선에 닿는 신호는 극점 탐지기 하나뿐인데,
그 손익은 변동성에 달려 있다(맞힘 +29.5 / 빗나감 −21.6bp, 둘 다 변동성에 비례).
DVOL 은 **미래 변동성의 가격**이고 ATR 은 **과거 변동성**이다 -- 두 정보가 어긋나는 순간
(VRP 극단)이 극점 콜의 품질을 가르는지 본다.

⭐이건 새 신호를 **더하는** 게 아니라 기존 신호를 **거르는** 축이라, 판정 기준도 다르다:
  같은 커버리지에서 정밀도/순bp 가 갈라지는가, 그리고 그 갈림이 두 창에서 같은 부호인가.
⚠️전건을 센다. 사후에 좋은 분위만 고르지 않는다(분류학 A).
"""
from __future__ import annotations
import argparse, glob, json, os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import research_eth_dvol_vrp_signal_20260910 as S  # noqa: E402

EX = ROOT / "tmp/eth_signal_map_20260909"
OUT = ROOT / "tmp/eth_dvol_vrp_20260910"
H_EVAL, COST, ROLL_Q = 48, 10.0, 90 * 24


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--tier", default="강중")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import joblib

    # --- 극점 탐지기 표본외 점수 (배포 아티팩트 p1) + 손실가중 헤드 p2 ---------------
    A = pd.read_parquet(EX / "extreme_frame.parquet")
    fm = json.load(open(EX / "extreme_frame_meta.json"))
    feats, VAL0 = fm["feats"], pd.Timestamp(fm["val0"])
    A = A[A["_y"] >= 0].sort_values("_ts").reset_index(drop=True)
    p1s = pd.read_csv(EX / "tabpfn_p1_scores_20260910.csv", parse_dates=["_ts"])
    A = A.merge(p1s, on="_ts", how="left").dropna(subset=["p1"]).reset_index(drop=True)
    X = np.nan_to_num(A[feats].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    cw = joblib.load(ROOT / "data/live/eth_extreme_detector_costw_artifact/model.joblib")
    A["p2"] = np.mean([m.predict_proba(X)[:, 1] for m in cw], axis=0)

    # --- 가격/VRP 시간봉 -----------------------------------------------------------
    dv = pd.read_csv(S.DVOL_CSV, parse_dates=["timestamp"])[["timestamp", "close"]]
    tl = S.fetch_dvol_tail(dv.timestamp.max())
    dv = pd.concat([dv, tl], ignore_index=True).drop_duplicates("timestamp", keep="first")
    dv = dv.sort_values("timestamp").reset_index(drop=True).rename(columns={"close": "dvol"})
    px5 = S.load_px5(); p = px5.set_index("timestamp")
    r5 = np.log(p["close"]).diff()
    rv = (r5.rolling(24 * 12, min_periods=24 * 6).std() * np.sqrt(288 * 365) * 100).resample("1h").last()
    hh = pd.DataFrame({"rv": rv}).join(dv.set_index("timestamp")["dvol"], how="inner").dropna()
    hh["vrp"] = hh["dvol"] - hh["rv"]
    # 인과 롤링 분위 -> 삼분위 상태
    qh = hh["vrp"].rolling(ROLL_Q, min_periods=ROLL_Q // 3).quantile(0.67)
    ql = hh["vrp"].rolling(ROLL_Q, min_periods=ROLL_Q // 3).quantile(0.33)
    hh["vrp_state"] = np.where(hh["vrp"] >= qh, "고", np.where(hh["vrp"] <= ql, "저", "중"))
    hh.loc[qh.isna() | ql.isna(), "vrp_state"] = "-"

    A["_hr"] = pd.to_datetime(A["_ts"]).dt.floor("h")
    A = A.merge(hh[["vrp", "vrp_state"]], left_on="_hr", right_index=True, how="left")
    A = A[A["vrp_state"].isin(["고", "중", "저"])].reset_index(drop=True)

    # --- 손익 -----------------------------------------------------------------------
    px = px5.set_index("timestamp")
    op = px["open"].to_numpy(float); cl = px["close"].to_numpy(float); n = len(px)
    pos = pd.Series(np.arange(n), index=px.index)
    A["_j"] = pos.reindex(pd.DatetimeIndex(A["_ts"])).to_numpy()
    A = A.dropna(subset=["_j"]).copy(); A["_j"] = A["_j"].astype(int)
    A = A[A["_j"] + H_EVAL + 1 < n].reset_index(drop=True)
    j = A["_j"].to_numpy(); lg = A["_long"].to_numpy(bool); e = op[j+1]
    A["ret"] = np.where(lg, (cl[j+H_EVAL]-e)/e, (e-cl[j+H_EVAL])/e) * 1e4
    y = A["_y"].to_numpy(int); ts = pd.to_datetime(A["_ts"])
    oos = (ts >= VAL0).to_numpy(); o = np.flatnonzero(oos); k = len(o)//2
    cm = np.zeros(len(A), bool); cm[o[:k]] = True
    ev = np.zeros(len(A), bool); ev[o[k:]] = True
    p1 = A["p1"].to_numpy(); p2 = A["p2"].to_numpy()
    c1 = {g: float(np.quantile(p1[cm], 1-q)) for g, q in (("강", .05), ("중", .10), ("약", .25))}
    c2 = {"강": float(np.quantile(p2[cm], 0.95))}
    grade = np.where((p1 >= c1["강"]) & (p2 >= c2["강"]), "강",
             np.where(p1 >= c1["중"], "중", np.where(p1 >= c1["약"], "약", "-")))
    sel_tier = np.isin(grade, ["강", "중"]) if a.tier == "강중" else (grade == a.tier)
    days = (ts[ev].max() - ts[ev].min()).total_seconds()/86400
    print(f"극점 콜 모집단(v2 등급) · 평가 {ts[ev].min():%m-%d}~{ts[ev].max():%m-%d} ({days:.0f}일)")
    print(f"VRP 삼분위는 인과 롤링(90일). 티어={a.tier}\n")
    print("=" * 92)
    print(f"{'창':<10}{'VRP':<5}{'n':>6}{'건/일':>7}{'정밀도':>9}{'순bp':>9}{'평균VRP':>9}")
    print("=" * 92)
    rows = []
    for wname, wm in (("컷창(전반)", cm), ("측정창(후반)", ev)):
        for st in ("저", "중", "고"):
            m = wm & sel_tier & (A["vrp_state"].to_numpy() == st)
            if m.sum() < 10: continue
            rows.append(dict(win=wname, state=st, n=int(m.sum()), per_day=m.sum()/days,
                             prec=float(y[m].mean()), net=float(A["ret"][m].mean()-COST)))
            r = rows[-1]
            print(f"{wname:<10}{st:<5}{r['n']:>6}{r['per_day']:>7.2f}{r['prec']:>9.3f}"
                  f"{r['net']:>9.2f}{A['vrp'][m].mean():>9.1f}")
        print("-" * 92)
    d = pd.DataFrame(rows); d.to_csv(OUT / "vrp_conditioner.csv", index=False)
    print("판정 -- 갈림이 두 창에서 같은 부호인가 (고 − 저)")
    for met in ("prec", "net"):
        vals = {}
        for w in ("컷창(전반)", "측정창(후반)"):
            g = d[(d.win == w)].set_index("state")
            if "고" in g.index and "저" in g.index:
                vals[w] = g.loc["고", met] - g.loc["저", met]
        if len(vals) == 2:
            same = (vals["컷창(전반)"] > 0) == (vals["측정창(후반)"] > 0)
            print(f"  {met:<6} 전반 {vals['컷창(전반)']:+.3f} · 후반 {vals['측정창(후반)']:+.3f}  "
                  f"→ {'같은 부호' if same else '🔴부호 뒤집힘'}")
    print(f"\n저장 {OUT/'vrp_conditioner.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
