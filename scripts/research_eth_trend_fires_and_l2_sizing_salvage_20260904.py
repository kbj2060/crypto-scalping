"""추세 신호·섀도우 결정 알고리즘에서 더 살릴 것이 있나 — 두 가지 미시험 축 (2026-09-04)

(A) 추세 신호 8종의 발동을 **칩(순위)이 아니라 추가 사건원**으로: 발동 방향 그대로(=지속) 거래. 단독 포트폴리오 + 반전 지속 규칙(R)과의 **합집합** 포트폴리오,
    R 대비 일별 짝비교. 사전 후보 집합 = 규칙 통과 3종(oi_confirmed_breakout, regime_pullback_resume, spot_led_move); 8종 전체는 진단.
(B) L2 확률을 **방향이 아니라 사이징**에: R의 지속 거래 크기를 신호별 인과 분위수로 가중(저p=지속 확신 → 크게). 평균 가중 1로 정규화해 R과 일별 짝비교.
(C) 진단: R 거래 중 직전 12봉 안에 같은 방향 추세 발동이 있었는지(컨플루언스)로 나눈 PnL.
손익 = F0 프레임(sim_exit 5.0/1.5/0.1, 200봉, open[i+1], 10bp). 동시 5 슬롯 순차 체결. 연구/개발 점수. HOLDOUT 미접촉.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_live_decision_algorithm_v1_20260904 as M
TRG = ROOT / "data/research/eth_trend_signals_v1_screen_20260904"; OUT = ROOT / "data/research/eth_live_decision_algorithm_v1_20260904"
TREND8 = ["quarter_hour_boundary_flow", "session_open_range_breakout", "oi_confirmed_breakout", "funding_squeeze", "regime_pullback_resume", "btc_leadlag", "spot_led_move", "liquidity_vacuum"]
TPASS = ["oi_confirmed_breakout", "regime_pullback_resume", "spot_led_move"]
GAP, CAP = 12, 5; rng = np.random.default_rng(20260904)


def log(m): print(f"[salvage] {m}", flush=True)


def first_fire(pos):
    keep = np.zeros(len(pos), bool); last = -10**9
    for j, p in enumerate(pos):
        if p - last > GAP: keep[j] = True
        last = p
    return keep


def build_reversal_frame():
    fr = []
    for s in M.REV:
        d = pd.read_parquet(M.RB / f"robustness_{s}.parquet"); d["signal"] = s; fr.append(d)
    X = pd.concat(fr, ignore_index=True); X["timestamp"] = pd.to_datetime(X["timestamp"]); X["is_downside"] = X["is_downside"].astype(int)
    f0 = pd.read_parquet(M.F0, columns=["pos", "is_downside", "timestamp", "split", "net_bp", "exit_off", "reg_eth_bull", "reg_eth_bear", "reg_eth_chop"]); f0["timestamp"] = pd.to_datetime(f0["timestamp"]); f0["is_downside"] = f0["is_downside"].astype(int)
    f0["reg_eth"] = np.select([f0.reg_eth_bull == 1, f0.reg_eth_bear == 1, f0.reg_eth_chop == 1], ["bull", "bear", "chop"], "none"); own = f0.set_index(["timestamp", "is_downside"])
    key = pd.MultiIndex.from_arrays([X.timestamp, X.is_downside]); keyf = pd.MultiIndex.from_arrays([X.timestamp, 1 - X.is_downside])
    X["pos"] = own["pos"].reindex(key).to_numpy(); X["exit_off"] = own["exit_off"].reindex(key).to_numpy(); X["exit_off_flip"] = own["exit_off"].reindex(keyf).to_numpy(); X["reg_eth"] = own["reg_eth"].reindex(key).to_numpy()
    X = X.dropna(subset=["pos", "exit_off", "exit_off_flip"]).reset_index(drop=True); X["pos"] = X["pos"].astype(int)
    return X, own, f0


def trend_candidates(own):
    rows = []
    for s in TREND8:
        t = pd.read_parquet(TRG / f"triggers_{s}.parquet"); t["timestamp"] = pd.to_datetime(t["timestamp"]); t["is_downside"] = t["is_downside"].astype(int)
        key = pd.MultiIndex.from_arrays([t.timestamp, t.is_downside]); t["pos"] = own["pos"].reindex(key).to_numpy(); t["exit_off"] = own["exit_off"].reindex(key).to_numpy()
        t["pnl_bp"] = own["net_bp"].reindex(key).to_numpy(); t["split"] = own["split"].reindex(key).to_numpy(); t = t.dropna(subset=["pos", "exit_off", "pnl_bp"]).sort_values("pos").reset_index(drop=True); t["pos"] = t["pos"].astype(int)
        ff = np.zeros(len(t), bool)
        for sd in (0, 1):
            ii = np.flatnonzero(t["is_downside"].to_numpy() == sd); ff[ii] = first_fire(t["pos"].to_numpy()[ii])
        t = t[ff].reset_index(drop=True); t["signal"] = s; t["trade_long"] = t["is_downside"] == 1; t["entry_bar"] = t["pos"] + 1; t["exit_bar"] = t["pos"] + 1 + t["exit_off"].astype(int); t["score"] = 0.0
        rows.append(t[["timestamp", "split", "pos", "entry_bar", "exit_bar", "pnl_bp", "trade_long", "score", "signal"]])
    return pd.concat(rows, ignore_index=True)


def dedupe(C):
    C = C.sort_values("score", ascending=False).drop_duplicates(["entry_bar", "trade_long"])
    conflict = C.groupby("entry_bar")["trade_long"].transform("nunique") > 1
    return C[~conflict].sort_values("entry_bar").reset_index(drop=True), int(C.loc[conflict, "entry_bar"].nunique())


def daily(r, days, w=None):
    pnl = r["pnl"] if w is None else r["pnl"] * w
    return pd.Series(pnl / CAP, index=pd.DatetimeIndex(r["ts"]).normalize()).groupby(level=0).sum().reindex(days, fill_value=0.0)


def paired(sa, sb):
    d = (sb - sa).to_numpy(); boots = [rng.choice(d, len(d), replace=True).mean() for _ in range(2000)]
    return {"mean_per_day": round(float(d.mean()), 2), "ci95": [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)], "days_better": round(float((d > 0).mean()), 2), "days_equal": round(float((d == 0).mean()), 2)}


def summarize(r, days, ndw):
    ci, _ = M.day_ci(r["pnl"], r["ts"]); s = daily(r, days)
    return {"n": r["n"], "per_day": round(r["n"] / ndw, 2), "exp_bp": round(r["exp_bp"], 2), "day_ci95": ci, "win": round(r["win_rate"], 3), "max_dd_bp": round(r["max_dd_bp"], 1), "daily_mean_bp": round(float(s.mean()), 2), "pos_day": round(float((s > 0).mean()), 3)}


def main():
    X, own, f0 = build_reversal_frame(); T = trend_candidates(own); rep = {"holdout_touched": False, "cap": CAP, "windows": {}}
    log(f"trend first-fires(GAP12): " + ", ".join(f"{s}:{int((T.signal==s).sum())}" for s in TREND8))
    for w in ("VAL", "OOS"):
        F = X[X.split == w].reset_index(drop=True); ndw = int(F["day"].nunique()); days = pd.date_range(F.timestamp.min().normalize(), F.timestamp.max().normalize(), freq="D"); o = {"n_days": ndw}
        R_c, _ = M.decide(F, {"kind": "cont_all"}); R_c = R_c.assign(src="R"); rR = M.portfolio(R_c, CAP); sR = daily(rR, days); o["R_cont_all"] = summarize(rR, days, ndw)
        Tw = T[T.split == w].reset_index(drop=True)
        # (A) 단독
        o["trend_standalone"] = {}
        for s in TREND8:
            c = Tw[Tw.signal == s]; r = M.portfolio(c, CAP) if len(c) >= 20 else None
            if r: o["trend_standalone"][s] = summarize(r, days, ndw); o["trend_standalone"][s]["cap1_exp"] = round(M.portfolio(c, 1)["exp_bp"], 2)
        for nm, sset in (("T_pass3", TPASS), ("T_all8", TREND8)):
            c, nconf = dedupe(Tw[Tw.signal.isin(sset)].copy()); r = M.portfolio(c, CAP); o[f"{nm}_standalone"] = {**summarize(r, days, ndw), "conflict_bars": nconf, "cap1_exp": round(M.portfolio(c, 1)["exp_bp"], 2)}
            # 합집합: R 후보 + 추세 후보 (같은 봉·방향 중복 → R 우선(score 1), 충돌 → 스킵)
            U = pd.concat([R_c.assign(score=1.0), Tw[Tw.signal.isin(sset)].assign(src="T", score=0.0)], ignore_index=True); U, nconf_u = dedupe(U); rU = M.portfolio(U, CAP); sU = daily(rU, days)
            taken = U.iloc[rU["idx"]]; o[f"R_union_{nm}"] = {**summarize(rU, days, ndw), "conflict_bars": nconf_u, "taken_from_T": int((taken["src"] == "T").sum()), "taken_from_R": int((taken["src"] == "R").sum()),
                                                            "T_taken_exp_bp": round(float(rU["pnl"][(taken["src"] == "T").to_numpy()].mean()), 2) if (taken["src"] == "T").any() else None,
                                                            "paired_vs_R": paired(sR, sU)}
            # 겹침: 추세 후보 중 ±12봉 안에 같은 방향 R 후보가 있는 비율
            rp = {k: np.sort(R_c.loc[R_c.trade_long == k, "entry_bar"].to_numpy()) for k in (True, False)}; ov = []
            for eb, tl in zip(Tw.loc[Tw.signal.isin(sset), "entry_bar"], Tw.loc[Tw.signal.isin(sset), "trade_long"]):
                a = rp[tl]; j = np.searchsorted(a, eb); near = (j < len(a) and a[j] - eb <= GAP) or (j > 0 and eb - a[j - 1] <= GAP); ov.append(near)
            o[f"R_union_{nm}"]["T_within12_of_sameside_R"] = round(float(np.mean(ov)), 3) if ov else None
        # (C) 컨플루언스 진단: R 거래 직전 12봉 안 같은 방향 추세 발동(PASS3 / ALL8)
        takenR = R_c.iloc[rR["idx"]].assign(pnl=rR["pnl"]); o["confluence"] = {}
        for nm, sset in (("pass3", TPASS), ("all8", TREND8)):
            tp = {k: np.sort(Tw.loc[Tw.signal.isin(sset) & (Tw.trade_long == k), "pos"].to_numpy()) for k in (True, False)}; flag = []
            for pos, tl in zip(takenR["entry_bar"] - 1, takenR["trade_long"]):
                a = tp[tl]; j = np.searchsorted(a, pos, side="right"); flag.append(j > 0 and pos - a[j - 1] <= GAP)
            flag = np.array(flag); o["confluence"][nm] = {"with": [int(flag.sum()), round(float(takenR["pnl"][flag].mean()), 2) if flag.any() else None], "without": [int((~flag).sum()), round(float(takenR["pnl"][~flag].mean()), 2)]}
        # (B) L2 사이징: 신호별 인과 분위수 → 가중
        hi, lo = M.causal_quantile_flags(F, 0.70, 0.30)
        w3 = np.where(lo, 1.5, np.where(hi, 0.5, 1.0))
        # 연속: 신호별 인과 백분위 rank_pct(p) → w = 2*(1-rank_pct), 번인 전 1.0
        rk = np.ones(len(F))
        for s, g in F.groupby("signal", sort=False):
            idx = g.index.to_numpy(); p = g["p_new"].to_numpy()
            for j in range(50, len(p)): rk[idx[j]] = (p[:j] < p[j]).mean()
        wc = np.where(rk == 1.0, 1.0, 2.0 * (1.0 - rk)); wc[:] = np.clip(wc, 0.2, 2.0)
        o["l2_sizing"] = {}
        for nm, wv in (("w3_1.5/1/0.5", w3), ("w_cont_2(1-rank)", wc)):
            # 후보 순서는 decide()가 fade 블록 뒤 cont 블록으로 만들었으므로 F 순서와 정렬 필요: cont_all은 F 전체가 cont → entry_bar로 재정렬된 R_c의 원본 인덱스 복원
            Cw = F.assign(w=wv, entry_bar=F["pos"] + 1, trade_long=F["is_downside"] != 1, score=np.abs(F["p_new"] - 0.5))   # 지속 거래 방향 = 페이드의 반대
            Cw = Cw.sort_values("score", ascending=False).drop_duplicates(["entry_bar", "trade_long"]).set_index(["entry_bar", "trade_long"])["w"]   # decide()와 같은 중복 규칙
            tk = R_c.iloc[rR["idx"]]; wt = Cw.reindex(pd.MultiIndex.from_arrays([tk["entry_bar"].to_numpy(), tk["trade_long"].to_numpy()])).to_numpy()
            wt = np.where(np.isfinite(wt), wt, 1.0); wt = wt / wt.mean()
            sW = daily(rR, days, wt); o["l2_sizing"][nm] = {"weighted_exp_bp": round(float((rR["pnl"] * wt).mean()), 2), "unweighted_exp_bp": round(rR["exp_bp"], 2), "paired_vs_R": paired(sR, sW), "w_mean_before_norm": round(float(wv.mean()), 3)}
        rep["windows"][w] = o
        log(f"== {w} ({ndw}d) R: n {o['R_cont_all']['n']} exp {o['R_cont_all']['exp_bp']} CI {o['R_cont_all']['day_ci95']} daily {o['R_cont_all']['daily_mean_bp']}")
        for s, v in o["trend_standalone"].items(): log(f"   T {s:28s} n {v['n']:4d} ({v['per_day']}/d) exp {v['exp_bp']:+6.2f} CI {v['day_ci95']} win {v['win']} cap1 {v['cap1_exp']}")
        for nm in ("T_pass3", "T_all8"):
            v = o[f"{nm}_standalone"]; u = o[f"R_union_{nm}"]
            log(f"   {nm} standalone n {v['n']} ({v['per_day']}/d) exp {v['exp_bp']:+.2f} CI {v['day_ci95']} cap1 {v['cap1_exp']} | union n {u['n']} (T taken {u['taken_from_T']}, T exp {u['T_taken_exp_bp']}) exp {u['exp_bp']:+.2f} CI {u['day_ci95']} daily {u['daily_mean_bp']} | union−R/day {u['paired_vs_R']} | T within12 of R {u['T_within12_of_sameside_R']}")
        log(f"   confluence {o['confluence']}")
        for nm, v in o["l2_sizing"].items(): log(f"   L2 sizing {nm}: exp w {v['weighted_exp_bp']} vs {v['unweighted_exp_bp']} | daily paired {v['paired_vs_R']}")
    (OUT / "trend_and_sizing_salvage.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str)); print("SALVAGE_DONE", flush=True)


if __name__ == "__main__":
    main()
