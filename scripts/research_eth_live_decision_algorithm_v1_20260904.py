"""라이브 매매 결정 알고리즘 v1 연구 — 증거신호 발동 × L2 경제방향 확률 × 레짐 → 방향 라우팅 + 동시포지션 상한 (2026-09-04)

질문: 살아남은 재료(8종 발동, L2 확률 p_new, 지속 규칙, 레짐 OOF)를 **인과적으로** 결합하면 어떤 결정 규칙이 VAL/OOS에서 가장 낫고 견고한가.
모집단: 반전 8종 라이브 결정 봉 발동(칩별 horizon 중복제거) VAL/OOS — robustness_<sig>.parquet (p_new = TRAIN 학습 TabPFN 5시드 평균, 표본외).
손익: F0 프레임(sim_exit 5.0/1.5/0.1 ATR, 200봉, open[i+1] 진입, 10bp 차감) — 페이드 = 신호 방향 net_bp, 지속 = 반대 방향 net_bp_flip(그 측면 행의 exit_off).
결정 팔(사전 정의, 임계값은 창 분포가 아니라 고정 확률 — 라이브에서 그대로 쓸 수 있는 형태):
  cont_all / fade_all / fade p≥τ / cont p≤λ / router(τ,λ) / router+레짐 비상충 / (대조) 배포 터치확률 라우터
같은 봉 처리: 같은 (진입봉,방향) 중복은 |p−0.5| 큰 것 하나, 같은 진입봉에서 방향 충돌이면 둘 다 건너뜀.
포트폴리오: 진입봉 순서 슬롯 체결(portfolio() — research_homer_entry_v2_20260904 원문 복사), 동시 5(민감도 1·3).
통계: 건당 exp_bp · 일군집 부트스트랩 CI · 일 PnL(자본 대비, 1/CAP 슬라이스) 샤프 · MDD · 무작위 라우팅 귀무(B=200, 비율 매칭).
선택 규약: VAL로 팔 선택(exp_bp 일CI 하한>0 중 최대), OOS는 전 팔 1회 보고(재선택 없음). 연구/개발 점수 — 승격은 전진 섀도우.
HOLDOUT 미접촉.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RB = ROOT / "data/research/eth_chip_relabel_econ_dir_20260904"; F0 = ROOT / "tmp/homer_entry_v2_20260904/frame.parquet"
OUT = ROOT / "data/research/eth_live_decision_algorithm_v1_20260904"; OUT.mkdir(parents=True, exist_ok=True)
REV = ["taker_delta_z_climax", "short_term_return_z", "liquidity_sweep", "orthogonal_combo", "smt_divergence", "fib_extension_exhaustion", "demarker_extreme", "kalman_deviation_meanrev"]
CAP, B_BOOT, B_NULL = 5, 1000, 200
rng = np.random.default_rng(20260904)


def log(m): print(f"[decide] {m}", flush=True)


def portfolio(cand, max_conc):
    """진입봉 순서대로 슬롯 제약 하에 체결. cand: entry_bar/exit_bar/pnl_bp 컬럼. (research_homer_entry_v2_20260904.portfolio 원문)"""
    cand = cand.sort_values("entry_bar")
    eb = cand["entry_bar"].to_numpy(); xb = cand["exit_bar"].to_numpy(); pn = cand["pnl_bp"].to_numpy(); ts = cand["timestamp"].to_numpy()
    open_until, taken = [], []
    for k in range(len(cand)):
        open_until = [u for u in open_until if u > eb[k]]
        if len(open_until) < max_conc:
            open_until.append(xb[k]); taken.append(k)
    if not taken: return None
    p = pn[np.array(taken)]; eq = np.cumsum(p); dd = eq - np.maximum.accumulate(eq); w = p > 0
    return {"n": int(len(p)), "exp_bp": float(p.mean()), "total_bp": float(p.sum()), "win_rate": float(w.mean()),
            "payoff": float(p[w].mean() / -p[~w].mean()) if w.any() and (~w).any() else None, "max_dd_bp": float(dd.min()), "idx": np.array(taken), "pnl": p, "ts": ts[np.array(taken)]}


def day_ci(pnl, ts, B=B_BOOT):
    days = pd.DatetimeIndex(ts).normalize().to_numpy(); u = np.unique(days); idx = {d: np.flatnonzero(days == d) for d in u}; v = []
    for _ in range(B):
        ii = np.concatenate([idx[d] for d in rng.choice(u, len(u), replace=True)]); v.append(pnl[ii].mean())
    return [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)], int(len(u))


def daily_stats(pnl, ts, cap, n_days_window):
    s = pd.Series(pnl / cap, index=pd.DatetimeIndex(ts).normalize()).groupby(level=0).sum()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"), fill_value=0.0)
    mu, sd = s.mean(), s.std(ddof=1)
    return {"daily_mean_bp": round(float(mu), 2), "daily_sharpe_ann": round(float(mu / sd * np.sqrt(365)), 2) if sd > 0 else None, "pos_day_frac": round(float((s > 0).mean()), 3),
            "n_days_traded": int((s != 0).sum()), "total_bp_capital": round(float(s.sum()), 1)}


def causal_quantile_flags(F, q_hi=0.70, q_lo=0.30, burn=50, fixed=None):
    """신호별 p_new의 **인과** 분위수 임계값: 각 발동에서 같은 신호의 이전 발동들(창 시작부터 확장) 분위수와 비교. 번인 50건 전엔 결정 없음.
    fixed={signal: (hi, lo)} 가 주어지면(예: VAL 분위수를 OOS에 적용) 그 고정값을 쓴다."""
    hi = np.zeros(len(F), bool); lo = np.zeros(len(F), bool)
    for s, g in F.groupby("signal", sort=False):
        idx = g.index.to_numpy(); p = g["p_new"].to_numpy()
        if fixed is not None:
            th, tl = fixed[s]; hi[idx] = p >= th; lo[idx] = p <= tl; continue
        for j in range(burn, len(p)):
            past = p[:j]; hi[idx[j]] = p[j] >= np.quantile(past, q_hi); lo[idx[j]] = p[j] <= np.quantile(past, q_lo)
    return hi, lo


def decide(F, arm):
    """F: 발동 프레임. 반환: 후보 거래 DataFrame(entry_bar, exit_bar, pnl_bp, timestamp, trade_side, score)."""
    kind, a = arm["kind"], arm
    p = F["p_dep"].to_numpy() if a.get("prob") == "dep" else F["p_new"].to_numpy()
    fade = np.zeros(len(F), bool); cont = np.zeros(len(F), bool)
    if kind == "cont_all": cont[:] = True
    elif kind == "fade_all": fade[:] = True
    elif kind == "fade_p": fade = p >= a["tau"]
    elif kind == "cont_p": cont = p <= a["lam"]
    elif kind in ("router", "router_regime"):
        fade = p >= a["tau"]; cont = p <= a["lam"]
    elif kind == "cont_regime": cont[:] = True
    elif kind in ("qfade", "qcont", "qrouter", "cont_unless_qfade"):
        hi, lo = causal_quantile_flags(F, a.get("q_hi", 0.70), a.get("q_lo", 0.30), fixed=a.get("fixed"))
        if kind == "qfade": fade = hi
        elif kind == "qcont": cont = lo
        elif kind == "qrouter": fade = hi; cont = lo
        else: fade = hi; cont = ~hi                       # 기본 지속, 모델 상위 30%에서만 페이드
    if kind in ("router_regime", "cont_regime"):  # 거래 방향과 레짐이 상충하면 건너뜀: bull은 숏을, bear는 롱을 막는다
        fade_long = F["is_downside"].to_numpy() == 1            # 바닥 발동의 페이드 = 롱
        reg = F["reg_eth"].to_numpy()
        fade &= ~(((reg == "bear") & fade_long) | ((reg == "bull") & ~fade_long))
        cont &= ~(((reg == "bull") & fade_long) | ((reg == "bear") & ~fade_long))   # 지속 = 페이드의 반대 방향
    rows = []
    for m, side_is_fade in ((fade, True), (cont, False)):
        G = F[m]
        if G.empty: continue
        rows.append(pd.DataFrame({"timestamp": G["timestamp"].to_numpy(), "entry_bar": G["pos"].to_numpy() + 1,
                                  "exit_bar": G["pos"].to_numpy() + 1 + (G["exit_off"] if side_is_fade else G["exit_off_flip"]).to_numpy(),
                                  "pnl_bp": (G["net_bp"] if side_is_fade else G["net_bp_flip"]).to_numpy(),
                                  "trade_long": (G["is_downside"].to_numpy() == 1) == side_is_fade, "score": np.abs(G["p_new"].to_numpy() - 0.5), "signal": G["signal"].to_numpy()}))
    if not rows: return None, {"n_dec": 0, "n_conflict_bars": 0}
    C = pd.concat(rows, ignore_index=True); n_dec = len(C)
    C = C.sort_values("score", ascending=False).drop_duplicates(["entry_bar", "trade_long"])       # 같은 봉·같은 방향 중복 → 하나
    conflict = C.groupby("entry_bar")["trade_long"].transform("nunique") > 1; n_conf = int(C.loc[conflict, "entry_bar"].nunique())
    C = C[~conflict].reset_index(drop=True)
    return C, {"n_dec": n_dec, "n_conflict_bars": n_conf}


def null_pct(F, arm, obs, cap):
    """무작위 라우팅 귀무: 같은 발동 집합에서 fade/cont/skip 비율만 맞춰 무작위 배정 → exp_bp 분포."""
    C, _ = decide(F, arm)
    if C is None or len(C) == 0: return None
    n_f = int(C["trade_long"].eq(F["is_downside"].iloc[0] == 1).sum())  # placeholder not used
    kind_counts = {"fade": 0, "cont": 0}
    # 배정 비율: 결정된 페이드/지속 개수(중복제거 전 기준 근사)
    p = F["p_dep"].to_numpy() if arm.get("prob") == "dep" else F["p_new"].to_numpy()
    if arm["kind"] in ("fade_p",): kind_counts["fade"] = int((p >= arm["tau"]).sum())
    elif arm["kind"] in ("cont_p",): kind_counts["cont"] = int((p <= arm["lam"]).sum())
    elif arm["kind"] in ("router", "router_regime"): kind_counts["fade"] = int((p >= arm["tau"]).sum()); kind_counts["cont"] = int((p <= arm["lam"]).sum())
    elif arm["kind"] in ("qfade", "qcont", "qrouter", "cont_unless_qfade"):
        hi, lo = causal_quantile_flags(F, arm.get("q_hi", 0.70), arm.get("q_lo", 0.30), fixed=arm.get("fixed") if isinstance(arm.get("fixed"), dict) else None)
        kind_counts["fade"] = int(hi.sum()) if arm["kind"] != "qcont" else 0
        kind_counts["cont"] = int(lo.sum()) if arm["kind"] in ("qcont", "qrouter") else (int((~hi).sum()) if arm["kind"] == "cont_unless_qfade" else 0)
    elif arm["kind"] == "cont_all": kind_counts["cont"] = len(F)
    elif arm["kind"] == "fade_all": kind_counts["fade"] = len(F)
    vals = []
    for _ in range(B_NULL):
        perm = rng.permutation(len(F)); lab = np.full(len(F), "", object); lab[perm[:kind_counts["fade"]]] = "fade"; lab[perm[kind_counts["fade"]:kind_counts["fade"] + kind_counts["cont"]]] = "cont"
        G = F.copy(); G["p_new"] = np.where(lab == "fade", 1.0, np.where(lab == "cont", 0.0, 0.5))
        Cn, _ = decide(G, {"kind": "router", "tau": 0.99, "lam": 0.01})
        r = portfolio(Cn, cap) if Cn is not None and len(Cn) else None
        vals.append(r["exp_bp"] if r else np.nan)
    vals = np.array(vals); vals = vals[np.isfinite(vals)]
    return round(float((vals < obs).mean() * 100), 1) if len(vals) else None


def main():
    t0 = time.time()
    fr = []
    for s in REV:
        d = pd.read_parquet(RB / f"robustness_{s}.parquet"); d["signal"] = s; fr.append(d)
    X = pd.concat(fr, ignore_index=True); X["timestamp"] = pd.to_datetime(X["timestamp"])
    f0 = pd.read_parquet(F0, columns=["pos", "is_downside", "timestamp", "split", "net_bp", "net_bp_flip", "exit_off", "reg_eth_bull", "reg_eth_bear", "reg_eth_chop"])
    f0["timestamp"] = pd.to_datetime(f0["timestamp"]); f0["is_downside"] = f0["is_downside"].astype(int)
    f0["reg_eth"] = np.select([f0["reg_eth_bull"] == 1, f0["reg_eth_bear"] == 1, f0["reg_eth_chop"] == 1], ["bull", "bear", "chop"], "none")
    own = f0.set_index(["timestamp", "is_downside"]); X["is_downside"] = X["is_downside"].astype(int)
    key = pd.MultiIndex.from_arrays([X["timestamp"], X["is_downside"]]); keyf = pd.MultiIndex.from_arrays([X["timestamp"], 1 - X["is_downside"]])
    X["pos"] = own["pos"].reindex(key).to_numpy(); X["exit_off"] = own["exit_off"].reindex(key).to_numpy(); X["exit_off_flip"] = own["exit_off"].reindex(keyf).to_numpy()
    X["reg_eth"] = own["reg_eth"].reindex(key).to_numpy(); chk_net = own["net_bp"].reindex(key).to_numpy(); chk_flip = own["net_bp_flip"].reindex(key).to_numpy()
    X = X.dropna(subset=["pos", "exit_off", "exit_off_flip"]).reset_index(drop=True); X["pos"] = X["pos"].astype(int)
    parity = float(np.nanmax(np.abs(chk_net - X["net_bp"].reindex(range(len(chk_net))).to_numpy()))) if len(X) == len(chk_net) else None
    log(f"fires VAL {int((X.split=='VAL').sum())} OOS {int((X.split=='OOS').sum())} | net_bp parity max|Δ| {parity} | regime share {X['reg_eth'].value_counts(normalize=True).round(3).to_dict()}")
    log(f"p_new share: >=0.50 {float((X.p_new>=0.5).mean()):.3f} >=0.55 {float((X.p_new>=0.55).mean()):.3f} >=0.60 {float((X.p_new>=0.6).mean()):.3f} <=0.45 {float((X.p_new<=0.45).mean()):.3f} <=0.40 {float((X.p_new<=0.4).mean()):.3f}")
    ARMS = [{"name": "cont_all", "kind": "cont_all"}, {"name": "fade_all", "kind": "fade_all"},
            {"name": "fade_p>=0.50", "kind": "fade_p", "tau": 0.50}, {"name": "fade_p>=0.55", "kind": "fade_p", "tau": 0.55}, {"name": "fade_p>=0.60", "kind": "fade_p", "tau": 0.60},
            {"name": "cont_p<=0.45", "kind": "cont_p", "lam": 0.45}, {"name": "cont_p<=0.40", "kind": "cont_p", "lam": 0.40},
            {"name": "router(0.50,0.45)", "kind": "router", "tau": 0.50, "lam": 0.45}, {"name": "router(0.55,0.45)", "kind": "router", "tau": 0.55, "lam": 0.45},
            {"name": "router(0.55,0.40)", "kind": "router", "tau": 0.55, "lam": 0.40}, {"name": "router(0.60,0.40)", "kind": "router", "tau": 0.60, "lam": 0.40},
            {"name": "router_regime(0.55,0.45)", "kind": "router_regime", "tau": 0.55, "lam": 0.45}, {"name": "router_regime(0.50,0.45)", "kind": "router_regime", "tau": 0.50, "lam": 0.45},
            {"name": "cont_regime", "kind": "cont_regime"},
            {"name": "qfade_top30_causal", "kind": "qfade"}, {"name": "qcont_bot30_causal", "kind": "qcont"}, {"name": "qrouter_30/30_causal", "kind": "qrouter"},
            {"name": "cont_unless_qfade_top30_causal", "kind": "cont_unless_qfade"}, {"name": "cont_unless_qfade_top20_causal", "kind": "cont_unless_qfade", "q_hi": 0.80},
            {"name": "qfade_top30_VALq", "kind": "qfade", "fixed": "VALQ"}, {"name": "cont_unless_qfade_top30_VALq", "kind": "cont_unless_qfade", "fixed": "VALQ"},
            {"name": "CTRL_dep_router(0.55,0.45)", "kind": "router", "tau": 0.55, "lam": 0.45, "prob": "dep"}]
    VQ = {s: (float(np.quantile(g["p_new"], 0.70)), float(np.quantile(g["p_new"], 0.30))) for s, g in X[X["split"] == "VAL"].groupby("signal")}
    rep = {"cap": CAP, "cost_bp": 10.0, "holdout_touched": False, "fresh_forward_bar_by_bar": False, "trade_ledgers_used_as_input": False, "note": "research/dev score: stored F0 labels, causal features/thresholds", "arms": {}}
    for arm in ARMS:
        rep["arms"][arm["name"]] = {}
        for w in ("VAL", "OOS"):
            F = X[X["split"] == w].reset_index(drop=True); ndw = int(F["day"].nunique())
            if arm.get("fixed") == "VALQ":
                if w == "VAL": rep["arms"][arm["name"]][w] = {"n": 0, "note": "threshold calibration window"}; continue
                arm = {**arm, "fixed": VQ}
            C, meta = decide(F, arm); r = portfolio(C, CAP) if C is not None and len(C) else None
            if r is None: rep["arms"][arm["name"]][w] = {"n": 0, **meta}; continue
            ci, nd = day_ci(r["pnl"], r["ts"]); ds = daily_stats(r["pnl"], r["ts"], CAP, ndw)
            o = {"n": r["n"], "trades_per_day": round(r["n"] / ndw, 2), "exp_bp": round(r["exp_bp"], 2), "day_ci95": ci, "win_rate": round(r["win_rate"], 3), "payoff": round(r["payoff"], 2) if r["payoff"] else None,
                 "max_dd_bp": round(r["max_dd_bp"], 1), "total_bp": round(r["total_bp"], 1), **ds, **meta, "n_candidates": int(len(C)), "long_share": round(float(C.iloc[r["idx"]]["trade_long"].mean()), 3)}
            if arm["kind"] not in ("cont_all", "fade_all"): o["null_pct_exp_bp"] = null_pct(F, arm, r["exp_bp"], CAP)
            for cap2 in (1, 3):
                r2 = portfolio(C, cap2); o[f"cap{cap2}_exp_bp"] = round(r2["exp_bp"], 2) if r2 else None; o[f"cap{cap2}_n"] = r2["n"] if r2 else 0
            rep["arms"][arm["name"]][w] = o
            T = C.iloc[r["idx"]].assign(pnl_bp=r["pnl"])
            o["by_side"] = {("long" if k else "short"): [int(len(g)), round(float(g["pnl_bp"].mean()), 2)] for k, g in T.groupby("trade_long")}
            o["by_signal"] = {k: [int(len(g)), round(float(g["pnl_bp"].mean()), 2)] for k, g in T.groupby("signal")}
            o["by_month"] = {str(k): round(float(v), 2) for k, v in T.groupby(pd.to_datetime(T["timestamp"]).dt.to_period("M"))["pnl_bp"].mean().items()}
            if arm["name"] in ("cont_all", "cont_regime", "cont_unless_qfade_top30_causal", "qrouter_30/30_causal"):
                T.to_csv(OUT / f"trades_{arm['name'].replace('(', '_').replace(')', '').replace(',', '_').replace('/', '-')}_{w}.csv", index=False)
        v, oo = rep["arms"][arm["name"]].get("VAL", {}), rep["arms"][arm["name"]].get("OOS", {})
        log(f"{arm['name']:>27s} | VAL n {v.get('n',0):4d} ({v.get('trades_per_day','-')}/d) exp {v.get('exp_bp','-')} CI {v.get('day_ci95','-')} sh {v.get('daily_sharpe_ann','-')} dd {v.get('max_dd_bp','-')} null {v.get('null_pct_exp_bp','-')} | OOS n {oo.get('n',0):4d} ({oo.get('trades_per_day','-')}/d) exp {oo.get('exp_bp','-')} CI {oo.get('day_ci95','-')} sh {oo.get('daily_sharpe_ann','-')} dd {oo.get('max_dd_bp','-')} null {oo.get('null_pct_exp_bp','-')} | cap1 {v.get('cap1_exp_bp','-')}/{oo.get('cap1_exp_bp','-')}")
    # 선택 규약: VAL 일CI 하한 > 0 인 팔 중 exp_bp 최대 (대조 팔 제외)
    elig = [(n, a["VAL"]["exp_bp"]) for n, a in rep["arms"].items() if not n.startswith("CTRL") and a.get("VAL", {}).get("n", 0) >= 50 and a["VAL"]["day_ci95"][0] > 0]
    rep["selected_by_VAL"] = max(elig, key=lambda x: x[1])[0] if elig else None; rep["n_arms"] = len([a for a in ARMS if not a["name"].startswith("CTRL")])
    log(f"selected_by_VAL: {rep['selected_by_VAL']} (eligible {len(elig)}/{rep['n_arms']}) ({time.time()-t0:.0f}s)")
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str)); print("DECIDE_DONE", flush=True)


if __name__ == "__main__":
    main()
