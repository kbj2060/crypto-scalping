#!/usr/bin/env python3
"""발동 봉 지속 규칙 + 레짐(방향 일치) + 청산맵(지속 방향 클러스터 근접) **복합 게이트** 1회 평가 (2026-09-04).

앞선 두 진단의 결과를 한 규칙으로 묶어 TRAIN/VAL/OOS를 각 1회 본다. 선택은 TRAIN 표에서만 했다:
  · 레짐 R2 = 방향 불일치 셀 제외(바닥발동∧ETH bull, 천장발동∧ETH bear -- TRAIN에서 유이하게 음수였던 두 셀). chop 유지.
    레짐 R1 = 방향 일치만(바닥∧bear→숏, 천장∧bull→롱). 참고용.
  · 청산맵 LQ = 지속 방향 최근접 클러스터 거리 TRAIN 상위 삼분위(far, >5.07 ATR) 제외. (사전 가설 부호는 반대였고
    데이터가 세 구간 모두 ρ≈−0.24로 '근접일수록 지속'을 보여 그대로 채택 -- 자석 해석.)
변형 6개(C0 기본 / LQ / R2 / R1 / LQ∧R2 / LQ∧R1) 전부 보고. **주 후보 = LQ∧R2** (사전 지정, 최소 개입).
같은 게이트 아래 페이드(신호 방향)를 대조군으로 같이 낸다.

부수 산출: 기본 지속 규칙의 층 게이트 입력(`tmp/eth_fire_cont_pipeline_20260904/{fills.csv,gate_config.json,controls.json}`).
⚠️HOLDOUT 미접촉. 셀·GAP·비용·한도 상속.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


CM = _load("cont_mod2", "scripts/research_eth_evidence_fire_continuation_econ_20260904.py")
portfolio, day_boot, stats_of = CM.portfolio, CM.day_boot, CM.stats_of
FRAME, MAX_CONC, B_BOOT, CELL, FWD, COST = CM.FRAME, CM.MAX_CONC, CM.B_BOOT, CM.CELL, CM.FWD, CM.COST
ROWS = ROOT / "tmp/eth_fire_cont_liqmap_20260904/fire_cont_rows_with_levels.parquet"
OUT = ROOT / "data/research/eth_fire_cont_composite_20260904"
PIPE = ROOT / "tmp/eth_fire_cont_pipeline_20260904"
WINDOWS = ("TRAIN", "VAL", "OOS")
B_NULL = 200


def log(m): print(f"[comp] {m}", flush=True)


def pf(s, rng, pnl_col="net_bp"):
    if len(s) < 30:
        return None
    cand = pd.DataFrame({"timestamp": s["timestamp"].to_numpy(), "pos": s["pos"].to_numpy(), "p": 1.0, "entry_bar": s["pos"].to_numpy() + 1,
                         "exit_bar": s["pos"].to_numpy() + 1 + s["exit_off"].to_numpy(), "pnl_bp": s[pnl_col].to_numpy()})
    r = portfolio(cand, MAX_CONC)
    if r is None:
        return None
    lo, hi = day_boot(r["trades"]["pnl_bp"], r["trades"]["timestamp"], B_BOOT, rng); o = stats_of(r)
    o["day_ci95"] = [round(lo, 2), round(hi, 2)]; o["days"] = int(pd.DatetimeIndex(r["trades"]["timestamp"]).normalize().nunique())
    o["per_day"] = round(o["n"] / max(o["days"], 1), 2); o["long_share"] = round(float((r["trades"]["pos"].map(dict(zip(s["pos"], s["is_downside"]))) == 1).mean()), 3)
    mo = pd.Series(r["trades"]["pnl_bp"].to_numpy(), index=pd.to_datetime(r["trades"]["timestamp"].to_numpy())).groupby(lambda x: x.to_period("M")).mean()
    o["monthly_exp_bp"] = {str(k): round(float(v), 2) for k, v in mo.items()}
    return o, r["trades"]


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); PIPE.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(20260904)
    M = pd.read_parquet(ROWS)
    reg = pd.read_parquet(FRAME, columns=["pos", "reg_eth_bull", "reg_eth_bear", "reg_eth_chop"]).drop_duplicates("pos")
    reg["reg_eth"] = np.select([reg["reg_eth_bull"] == 1, reg["reg_eth_bear"] == 1, reg["reg_eth_chop"] == 1], ["bull", "bear", "chop"], "none")
    M = M.merge(reg[["pos", "reg_eth"]], on="pos", how="left")
    incons = ((M["fire_side"] == "bottom") & (M["reg_eth"] == "bull")) | ((M["fire_side"] == "top") & (M["reg_eth"] == "bear"))
    cons = ((M["fire_side"] == "bottom") & (M["reg_eth"] == "bear")) | ((M["fire_side"] == "top") & (M["reg_eth"] == "bull"))
    lq = M["d_cont_atr_t"].astype(str).isin(["near", "mid"])
    gates = {"C0_base": np.ones(len(M), bool), "LQ": lq.to_numpy(), "R2_excl_inconsistent": (~incons).to_numpy(), "R1_consistent_only": cons.to_numpy(),
             "LQ_R2": (lq & ~incons).to_numpy(), "LQ_R1": (lq & cons).to_numpy()}
    rep = {"cell": CELL, "forward_bars": FWD, "cost_bp": COST, "max_concurrent": MAX_CONC, "holdout_touched": False, "primary": "LQ_R2",
           "liq_far_edge_atr_train": 5.07, "variants": {}}
    print(f"\n{'variant':>22s} {'win':>5s} {'n_rows':>6s} {'pf_n':>5s} {'/day':>5s} {'cont_exp':>8s} {'dayCI':>16s} {'wr':>5s} {'payoff':>6s} {'maxDD':>7s} {'fade_exp':>8s} {'long%':>5s}")
    for nm, g in gates.items():
        rep["variants"][nm] = {}
        for w in WINDOWS:
            s = M.loc[g & (M["split"] == w).to_numpy()]
            c = pf(s, rng); f = pf(s, rng, "net_bp_flip")
            if c is None:
                continue
            co, ct = c; fo, _ = f if f else ({}, None)
            R = {"n_rows": int(len(s)), "cont": co, "fade": {k: fo.get(k) for k in ("exp_bp", "n", "day_ci95")} if fo else None,
                 "row_cont_bp": round(float(s["net_bp"].mean()), 2), "row_fade_bp": round(float(s["net_bp_flip"].mean()), 2),
                 "by_fire_side_cont_bp": {fs: round(float(s.loc[s["fire_side"] == fs, "net_bp"].mean()), 2) for fs in ("bottom", "top")}}
            rep["variants"][nm][w] = R
            print(f"{nm:>22s} {w:>5s} {len(s):6d} {co['n']:5d} {co['per_day']:5.1f} {co['exp_bp']:8.2f} {str(co['day_ci95']):>16s} {co['win_rate']:5.3f} "
                  f"{(co['payoff'] or 0):6.2f} {co['max_dd_bp']:7.0f} {(fo.get('exp_bp') if fo else float('nan')):8.2f} {co['long_share']*100:5.1f}")
            if nm == "LQ_R2":
                ct.to_csv(OUT / f"trades_LQ_R2_{w}.csv", index=False)
    # 주 후보 측면별 · 신호별 (VAL+OOS)
    g = gates["LQ_R2"]; s = M.loc[g & (M["split"] != "TRAIN").to_numpy()]
    rep["primary_detail"] = {"val_oos_by_fire_side": {fs: {"n": int((s["fire_side"] == fs).sum()), "cont_bp": round(float(s.loc[s["fire_side"] == fs, "net_bp"].mean()), 2),
                                                          "fade_bp": round(float(s.loc[s["fire_side"] == fs, "net_bp_flip"].mean()), 2)} for fs in ("bottom", "top")},
                             "val_oos_by_regime": {r_: {"n": int((s["reg_eth"] == r_).sum()), "cont_bp": round(float(s.loc[s["reg_eth"] == r_, "net_bp"].mean()), 2)} for r_ in ("bull", "bear", "chop")}}
    (OUT / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    print("\n[primary LQ_R2 VAL+OOS detail]", json.dumps(rep["primary_detail"], ensure_ascii=False))

    # ---------------- 층 게이트 입력 (기본 지속 규칙 C0; 복합은 같은 층의 행 필터) ----------------
    F = M.sort_values("timestamp").reset_index(drop=True)
    sig_of = pd.concat([CM.load_fires()]).loc[lambda d: d["first_fire"]].drop_duplicates(["pos", "is_downside"])[["pos", "is_downside", "signal"]]
    sig_of["is_downside"] = 1 - sig_of["is_downside"].astype(int)          # 발동 측면 -> 지속 측면 키
    F = F.merge(sig_of, on=["pos", "is_downside"], how="left")
    fills = pd.DataFrame({"timestamp": F["timestamp"], "signal": F["signal"].fillna("evidence8"), "side": F["side"], "sd": np.where(F["is_downside"] == 1, 1, -1),
                          "fi": F["pos"] + 1, "ei": F["pos"] + 1 + FWD, "btf": 1, "lim": F["entry"], "atr_pct": F["atr"] / F["entry"], "atr_abs": F["atr"],
                          "y": F["net_bp"] / 1e4, "split": F["split"], "l2_sample": (rng.random(len(F)) < 0.15).astype(int),
                          "fire_side": F["fire_side"], "reg_eth": F["reg_eth"], "d_cont_atr": F["d_cont_atr"], "liq_tertile": F["d_cont_atr_t"].astype(str)})
    fills.to_csv(PIPE / "fills.csv", index=False)
    # 대조군 리포트 (T1 형식: base / random_subsample / extra_passed) -- 측면비율 매칭 무작위 귀무
    Dn = pd.read_parquet(FRAME, columns=["pos", "is_downside", "timestamp", "split", "net_bp", "exit_off"])
    controls = {"base": {}, "random_subsample": {}, "extra_passed": []}
    for w in ("VAL", "OOS"):
        s = M.loc[(M["split"] == w).to_numpy()]; base = rep["variants"]["C0_base"][w]["cont"]["exp_bp"]; controls["base"][f"cont/{w}"] = base
        n_l = int((s["is_downside"] == 1).sum()); n_s = int((s["is_downside"] == 0).sum()); Dw = Dn.loc[Dn["split"] == w]
        pl, ps = Dw.loc[Dw["is_downside"] == 1], Dw.loc[Dw["is_downside"] == 0]; nulls = []
        for _ in range(B_NULL):
            x = pd.concat([pl.iloc[rng.choice(len(pl), size=n_l, replace=False)], ps.iloc[rng.choice(len(ps), size=n_s, replace=False)]])
            r = portfolio(pd.DataFrame({"timestamp": x["timestamp"].to_numpy(), "pos": x["pos"].to_numpy(), "p": 1.0, "entry_bar": x["pos"].to_numpy() + 1,
                                        "exit_bar": x["pos"].to_numpy() + 1 + x["exit_off"].to_numpy(), "pnl_bp": x["net_bp"].to_numpy()}), MAX_CONC)
            nulls.append(float(r["exp_bp"]))
        controls["random_subsample"][f"cont/{w}"] = nulls
        if base > rep["variants"]["C0_base"][w]["fade"]["exp_bp"]:
            controls["extra_passed"].append(f"flip/cont/{w}")
    try:
        from core.selection_stats import deflated_sharpe_ratio
        t = pd.read_csv(OUT / "trades_LQ_R2_OOS.csv") if (OUT / "trades_LQ_R2_OOS.csv").exists() else None
        tb = pd.read_csv(CM.OUT / "trades_cont_OOS.csv")
        d = pd.Series(tb["pnl_bp"].to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(tb["timestamp"])).normalize()).groupby(level=0).sum().to_numpy()
        sr = float(np.mean(d) / np.std(d, ddof=1)); dsr = deflated_sharpe_ratio(d, np.array([sr]))
        controls["dsr"] = dsr.get("deflated_sharpe_ratio"); controls["pbo"] = None
    except Exception as ex:                                       # noqa: BLE001
        controls["dsr"] = None; controls["pbo"] = None; controls["dsr_error"] = f"{type(ex).__name__}: {ex}"
    (PIPE / "controls.json").write_text(json.dumps(controls, default=float))
    gc = {"pipeline": "eth_fire_cont_20260904 -- 증거신호 8종 raw 첫발동(GAP12) 봉의 **반대 방향(지속)**, 경제라벨 sim_exit 5.0/1.5/0.1 200봉 10bp, 다음 봉 시가 진입",
          "splits": {"VAL": "2025-09-01", "OOS": "2026-01-01", "HOLDOUT": "2026-04-01"},
          "known_ts": {"assumption": "raw 단일봉 발동은 봉 τ 마감에 계산됨(known_ts=timestamp); GAP12 첫발동 판정은 과거만 봄; 진입 open[τ+1] (fi=pos+1, btf=1)"},
          "label": {"fills": "tmp/eth_fire_cont_pipeline_20260904/fills.csv", "ts_col": "timestamp", "y_col": "y", "entry_col": "lim", "side_col": "sd",
                    "atr_col": "atr_abs", "atr_is_absolute": True, "fill_idx_col": "fi", "exit_idx_col": "ei", "bars_to_fill_col": "btf", "signal_col": "signal",
                    "row_filter": "l2_sample == 1", "exit": {"sl_atr": CELL[0], "arm_atr": CELL[1], "trail_atr": CELL[2], "trail_anchor": "entry"},
                    "cost_roundtrip": COST / 1e4, "notional": 1.0, "tol_mean_bp": 2.0, "tol_winrate_pp": 2.0},
          "trigger": {"module": "gate_eth_entry_triggers_v1_adapter_20260903", "fn": "build_fires", "warmup_bars": 4000, "sample_n": 120},
          "scoring_parity": {"backtest": {"module": "scripts/research_homer_entry_v2_20260904.py", "fn": "trail_single", "style": "from_fill_bar"},
                             "live": {"adapter": "eth_v_rebound_econ_shadow"}, "n_paths": 300, "horizon_bars": 24},
          "controls": {"report": "tmp/eth_fire_cont_pipeline_20260904/controls.json", "dsr": controls.get("dsr"), "pbo": None},
          "selection": {"keep_frac": 1.0, "labels": ["y", "@recon"]}, "seed": 20260904}
    (PIPE / "gate_config.json").write_text(json.dumps(gc, indent=2, ensure_ascii=False))
    log(f"pipeline -> {PIPE} (fills {len(fills):,}, l2_sample {int(fills['l2_sample'].sum()):,}) · controls dsr {controls.get('dsr')} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
