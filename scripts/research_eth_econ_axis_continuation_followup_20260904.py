#!/usr/bin/env python3
"""경제 축 지속 스크린 v1 -- 후속 진단 (2026-09-04). 본 스크린(research_eth_econ_axis_continuation_screen_20260904.py) 산출을 읽어:
  (1) PASS 4축: R 과거12봉 겹침/비겹침 부분집합의 경제성, 방향이 직전 ret3 부호와 같은 비율(지속 vs 역행), 겹침 시 R 방향 일치율, 월별 exp
  (2) retail_shift: 메트릭 1봉 지연(안전 타이밍) 버전의 완전 평가(포트폴리오·귀무·R 합집합 짝비교), 발동 분(minute) 분포(데이터 갱신 아티팩트 점검)
  (3) 상태 축 WORKS 3종(btc_shock·market_beta_move·activity)을 필터가 아니라 **사이징**으로: 분위 가중 0.6/0.8/1.0/1.2/1.4(창 내 평균 1 정규화) → R 대비 일별 짝비교
  (4) PASS 4축 합집합(사후 진단)과 R 짝비교
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_econ_axis_continuation_screen_20260904 as S
OUT = S.OUT; PASS4 = ["oi_unwind_move", "btc_shock", "market_beta_move", "retail_shift"]


def main():
    F, _ = S.build_axes(); ts = F["timestamp"].to_numpy(); Fi = F.set_index("timestamp")
    train_mask = ((F["timestamp"] >= S.TRAIN_START) & (F["timestamp"] < S.TRAIN_END)).to_numpy()
    D = pd.read_parquet(S.FC.FRAME, columns=["pos", "is_downside", "timestamp", "split", "net_bp", "net_bp_flip", "exit_off", "atr"])
    D["timestamp"] = pd.to_datetime(D["timestamp"]); D["is_downside"] = D["is_downside"].astype(int); key = D.set_index(["timestamp", "is_downside"])
    Fr = S.FC.load_fires(); Fr = Fr.loc[Fr["first_fire"]].drop_duplicates(["pos", "is_downside"])
    Rk = pd.MultiIndex.from_arrays([Fr["pos"].to_numpy(), 1 - Fr["is_downside"].to_numpy().astype(int)], names=["pos", "is_downside"])
    rr = D.set_index(["pos", "is_downside"]).reindex(Rk).reset_index(); rr = rr[np.isfinite(rr["net_bp"].to_numpy())]
    R = pd.DataFrame({"timestamp": rr["timestamp"].to_numpy(), "pos": rr["pos"].astype(int).to_numpy(), "split": rr["split"].to_numpy(), "trade_long": (rr["is_downside"] == 1).to_numpy(),
                      "entry_bar": rr["pos"].astype(int).to_numpy() + 1, "exit_bar": rr["pos"].astype(int).to_numpy() + 1 + rr["exit_off"].astype(int).to_numpy(), "pnl_bp": rr["net_bp"].to_numpy(), "net_bp": rr["net_bp"].to_numpy()}).sort_values("pos").reset_index(drop=True)
    R_run = {}
    for w in S.WINDOWS:
        Rw, _ = S.dedupe(R[R.split == w]); _, R_run[w] = S.pf(Rw)
    rep = {"pass4": {}, "retail_shift_lag1": {}, "state_sizing": {}, "pass4_union": {}}
    ret3 = Fi["ret3"]
    # (1)
    for name in PASS4:
        A = pd.read_parquet(OUT / f"triggers_{name}.parquet"); A["timestamp"] = pd.to_datetime(A["timestamp"])
        r3 = ret3.reindex(A["timestamp"]).to_numpy(); A["with_move"] = np.sign(r3) == np.where(A["trade_long"], 1, -1)
        # 겹친 R 의 방향 일치 (같은 방향 R 존재 = R_past12 정의 자체가 같은 방향) → 반대 방향 R 존재 여부도
        rp = {tl: np.sort(R.loc[R["trade_long"] == tl, "pos"].to_numpy()) for tl in (True, False)}
        def near(pos_arr, tl):
            a = rp[tl]; j = np.searchsorted(a, pos_arr, side="right") - 1; return (j >= 0) & (pos_arr - a[np.clip(j, 0, len(a) - 1)] <= S.PAST_W)
        A["R_opp_past12"] = [near(np.array([p]), not tl)[0] for p, tl in zip(A["pos"], A["trade_long"])]
        o = {}
        for w in S.WINDOWS:
            Aw = A[A.split == w]; o[w] = {"with_move_share": round(float(Aw["with_move"].mean()), 3), "R_same_dir_past12": round(float(Aw["R_past12"].mean()), 3),
                                        "R_opp_dir_past12": round(float(Aw["R_opp_past12"].mean()), 3),
                                        "overlap_subset": S.econ(Aw[Aw.R_past12]), "nonoverlap_subset": S.econ(Aw[~Aw.R_past12]),
                                        "monthly_sig_bp": {str(k): round(float(v), 1) for k, v in Aw.groupby(Aw["timestamp"].dt.to_period("M"))["net_bp"].mean().items()}}
            Ad, _ = S.dedupe(Aw[~Aw.R_past12]) if (~Aw.R_past12).sum() >= 20 else (None, 0)
            if Ad is not None and len(Ad):
                s, _ = S.pf(Ad); o[w]["nonoverlap_portfolio"] = s
        o["minute_of_hour_hist"] = {str(k): int(v) for k, v in A["timestamp"].dt.minute.value_counts().sort_index().items()}
        rep["pass4"][name] = o
        print(f"\n== {name}"); [print(" ", w, {k: v for k, v in o[w].items() if k != "monthly_sig_bp"}) for w in S.WINDOWS]; print("  minute hist", o["minute_of_hour_hist"])
    # (2) retail_shift lag1 완전 평가
    F2 = F.copy(); F2["retail_shift"] = F["retail_shift_lag1"]; up, dn, thr = S.triggers(F2, "retail_shift", S.Q_PRIMARY, train_mask); A2 = S.rows_for(key, ts, up, dn); A2["R_past12"] = S.overlap_past(A2, R)
    for w in S.WINDOWS:
        Aw = A2[A2.split == w].reset_index(drop=True); Ad, _ = S.dedupe(Aw); s, r = S.pf(Ad)
        U, _ = S.dedupe(pd.concat([R[R.split == w], Aw[["timestamp", "pos", "split", "trade_long", "entry_bar", "exit_bar", "pnl_bp", "net_bp"]]], ignore_index=True)); su, ru = S.pf(U)
        rep["retail_shift_lag1"][w] = {"econ_both": S.econ(Aw), "econ_up": S.econ(Aw[Aw.side == "up"]), "econ_dn": S.econ(Aw[Aw.side == "dn"]), "portfolio": s,
                                       "side_matched_null": S.side_null(D[D.split == w], int(Ad["trade_long"].sum()), int((~Ad["trade_long"]).sum()), s["exp_bp"]) if s else None,
                                       "R_past12_share": round(float(Aw["R_past12"].mean()), 3), "union_portfolio": su, "union_minus_R_day_paired": S.day_paired(ru, R_run[w]) if ru else None}
        print(f"retail_shift_lag1 {w}: econ {rep['retail_shift_lag1'][w]['econ_both']} pf {s} null {rep['retail_shift_lag1'][w]['side_matched_null']} union−R {rep['retail_shift_lag1'][w]['union_minus_R_day_paired']}")
    # (3) 상태 사이징
    dir_sign = np.where(R["trade_long"], 1.0, -1.0); WQ = np.array([0.6, 0.8, 1.0, 1.2, 1.4])
    for ax, mode in (("btc_shock", "aligned"), ("market_beta_move", "aligned"), ("activity", "raw")):
        x = Fi[ax].reindex(R["timestamp"]).to_numpy(float); val = x * dir_sign if mode == "aligned" else x; okm = np.isfinite(val)
        edges = np.nanquantile(val[okm & (R.split == "TRAIN").to_numpy()], [0.2, 0.4, 0.6, 0.8]); qb = np.where(okm, np.searchsorted(edges, val, side="right"), 2)
        rep["state_sizing"][ax] = {}
        for w in ("VAL", "OOS"):
            mw = (R.split == w).to_numpy(); Rw, _ = S.dedupe(R[mw].assign(wq=WQ[qb[mw]])); wgt = Rw["wq"].to_numpy(); wgt = wgt / wgt.mean()
            Rw2 = Rw.assign(pnl_bp=Rw["pnl_bp"].to_numpy() * wgt); s, r = S.pf(Rw2); dp = S.day_paired(r, R_run[w])
            rep["state_sizing"][ax][w] = {"portfolio": s, "vs_R_day_paired": dp}; print(f"sizing {ax} {w}: exp {s['exp_bp']} CI {s['day_ci95']} | vs R {dp}")
    # (4) PASS4 합집합 (사후 진단)
    for w in S.WINDOWS:
        parts = [R[R.split == w]]
        for name in PASS4:
            A = pd.read_parquet(OUT / f"triggers_{name}.parquet"); A["timestamp"] = pd.to_datetime(A["timestamp"]); parts.append(A.loc[A.split == w, ["timestamp", "pos", "split", "trade_long", "entry_bar", "exit_bar", "pnl_bp", "net_bp"]])
        U, nconf = S.dedupe(pd.concat(parts, ignore_index=True)); su, ru = S.pf(U); dp = S.day_paired(ru, R_run[w])
        rep["pass4_union"][w] = {"portfolio": su, "conflict_bars": nconf, "vs_R_day_paired": dp}; print(f"pass4 union {w}: {su} conflicts {nconf} | vs R {dp}")
    (OUT / "followup.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str)); print("FOLLOWUP_DONE")


if __name__ == "__main__":
    main()
