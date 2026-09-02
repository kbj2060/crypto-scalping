#!/usr/bin/env python3
"""BTC 경제라벨 -- **1시간봉** 재도전. 5분봉에서 소진된 마지막 축.

## 왜 1시간봉인가

5분봉 3차 시도가 전부 실패했고 마지막이 문제의 성격을 드러냈다:

    1차(ARM1.5/SL16)  손익비 0.089, OOS AUC 0.6525, OOS -22.18bp
    2차(손익비하한)    --            OOS AUC 0.5823, OOS  -8.14bp
    3차(ARM8.0)       손익비 1.025, OOS AUC 0.4912, OOS -21.14bp

**손익비를 고칠수록 AUC가 떨어졌다** -- 앞선 높은 AUC는 왜도 불균형이 만든 착시였고,
균형을 잡으면 예측력이 0이다. 근본 원인 후보 둘:

  (A) **비용 장벽**: BTC 5m ATR 16bp에 비용 10bp -- 비용/ATR 62%(ETH 43%).
      무작위 기준선이 -1.04bp(ETH +2.6bp)라 진짜 예측력이 필요한데 없다.
  (B) **피쳐가 이 시간축에서 방향 정보를 못 담는다**.

1시간봉이면 ATR이 봉 길이의 제곱근에 비례해 커진다(16bp x sqrt(12) ~ 55bp).
비용/ATR이 62% -> ~18%로 떨어져 **(A)를 직접 제거**한다. 그래도 실패하면 (B)가 원인이고,
같은 피쳐셋으로는 BTC 방향 예측이 안 된다는 결론이 확정된다.

⚠️**기대치는 낮게 잡는다.** 3차의 OOS AUC 0.4912는 비용과 무관한 예측력 부재를 시사한다.

## 설계

5분봉 CSV를 1시간봉으로 리샘플하고 **같은 지표 파이프라인**을 다시 태운다(재구현 금지 --
compute_indicators/add_creative/add_broad/add_causal_columns를 그대로 쓴다).
그 위에서 5분봉과 **동일한 절차**를 밟는다:

  1) TRAIN에서만 exit 셀 선정 (손익비 하한 0.25, 라벨률 0.55~0.85)
  2) 5시드 앙상블 -> VAL 선정(분위/셀/한도) -> OOS 1회
  3) 방향뒤집기 대조 + 측면균형 진단

⚠️HOLDOUT 미터치. 라이브 코드 변경 없음.
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
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_pf = _load("pf_1h", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
sim_exit, portfolio = _pf.sim_exit, _pf.portfolio
SEEDS, CONTEXT_N = _pf.SEEDS, _pf.CONTEXT_N

BTC_5M = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
COST_BP = 10.0
# 1시간봉이므로 forward 창을 봉 수로 재환산: 5m 200봉(16.7h) ~ 1h 17봉.
# 넉넉히 48봉(2일)까지 본다 -- 봉이 길어지면 추세 지속 여지도 커진다.
FORWARD_BARS = 48
SL_GRID = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
ARM_GRID = (1.0, 1.5, 2.5, 4.0)
TRAIL_GRID = (0.10, 0.15, 0.2, 0.3, 0.5)
PAYOFF_FLOOR = 0.25
TAIL_FRACS = [0.01, 0.02, 0.05, 0.10]
MAX_CONCURRENT = [1, 3, 5]
CHUNK = 20000
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
OUT = ROOT / "data/research/btc_v_rebound_econ_1h_20260902/report.json"

BAR_FEATURES = ["atr", "atr_percentile_864", "range_width_pct", "hour_utc", "weekday",
                "delta_z", "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
                "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14",
                "pdi", "ndi", "bb_width_pctile", "rsi"]
TIER0 = ["is_downside", "sweep_penetration_atr", "flow_aligned_delta_z"] + BAR_FEATURES


def log(m): print(f"[btc-1h] {m}", flush=True)


def build_1h_frame() -> pd.DataFrame:
    """5분봉 -> 1시간봉 리샘플 후 **동일 지표 파이프라인** 적용."""
    from analyze_eth_broad_evidence_signal_sweep_20260814 import add_broad_indicators
    from analyze_eth_creative_reversal_evidence_signals_20260814 import add_creative_indicators
    from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import compute_indicators

    d = pd.read_csv(BTC_5M)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "taker_buy_base" in d.columns:
        agg["taker_buy_base"] = "sum"
    h = d.resample("1h").agg(agg).dropna(subset=["open", "high", "low", "close"]).reset_index()
    log(f"1시간봉 {len(h):,}행 ({h.timestamp.min()} ~ {h.timestamp.max()})")

    f = compute_indicators(h)
    f = add_creative_indicators(f)
    f = add_broad_indicators(f)

    bspec = importlib.util.spec_from_file_location(
        "bcand", ROOT / "scripts/build_btc_5m_evidence_signal_candidates_tier0_20260901.py")
    bmod = importlib.util.module_from_spec(bspec); bspec.loader.exec_module(bmod)
    sweep_impl = bmod.load_sweep_impl()

    ret3 = f["close"] / f["close"].shift(3) - 1.0
    f["ret3_z"] = (ret3 - ret3.rolling(288, min_periods=288).mean()) \
        / ret3.rolling(288, min_periods=288).std().replace(0.0, np.nan)
    causal = sweep_impl.add_causal_columns(h[["timestamp", "open", "high", "low", "close"]].copy())
    f["sweep_level_low"] = causal["sweep_level_low"]
    f["sweep_level_high"] = causal["sweep_level_high"]
    f["atr"] = causal["atr"]
    f["atr_percentile_864"] = f["atr"].rolling(864, min_periods=864).rank(pct=True)
    f["range_width_pct"] = (f["sweep_level_high"] - f["sweep_level_low"]) / f["close"]
    f["hour_utc"] = f["timestamp"].dt.hour
    f["weekday"] = f["timestamp"].dt.weekday
    f["rsi"] = bmod.rsi_wilder(f["close"])
    return f


def to_long(f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    atr = f["atr"].to_numpy(dtype=float)
    dz = f["delta_z"].to_numpy(dtype=float)
    for side, is_down in (("bottom", True), ("top", False)):
        sub = pd.DataFrame({"timestamp": f["timestamp"], "side": side})
        sub["is_downside"] = np.int8(1 if is_down else 0)
        lvl = (f["sweep_level_low"] if is_down else f["sweep_level_high"]).to_numpy(dtype=float)
        pen = (lvl - f["low"].to_numpy()) if is_down else (f["high"].to_numpy() - lvl)
        with np.errstate(invalid="ignore", divide="ignore"):
            sub["sweep_penetration_atr"] = np.where(np.isfinite(atr) & (atr > 0), pen / atr, np.nan)
        sub["flow_aligned_delta_z"] = dz if is_down else -dz
        for c in BAR_FEATURES:
            sub[c] = f[c].to_numpy()
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    out["bar_idx"] = np.tile(np.arange(len(f)), 2)
    return out


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    from sklearn.metrics import roc_auc_score
    import torch
    log(f"cuda: {torch.cuda.is_available()}")

    f = build_1h_frame()
    o, h_, l_, c = (f[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    nb = len(f)
    long = to_long(f).dropna(subset=TIER0).reset_index(drop=True)
    long = long.loc[long["bar_idx"] + FORWARD_BARS + 1 < nb].reset_index(drop=True)
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                     np.where(long["timestamp"] < VAL_END, "VAL",
                      np.where(long["timestamp"] < OOS_END, "OOS", "HOLDOUT")))
    long = long.loc[long["split"] != "HOLDOUT"].reset_index(drop=True)
    assert long["timestamp"].max() < OOS_END, "HOLDOUT 누출"
    tr = long.loc[long["split"] == "TRAIN"]
    log(f"long {len(long):,}행 (TRAIN {len(tr):,} / VAL {int((long.split=='VAL').sum()):,} / "
        f"OOS {int((long.split=='OOS').sum()):,})")

    i_tr = tr["bar_idx"].to_numpy().astype(int)
    atr_bp = tr["atr"].to_numpy(dtype=float) / c[i_tr] * 1e4
    log(f"⭐1시간봉 ATR: 중앙 {np.median(atr_bp):.1f}bp  (5분봉 16.0bp, ETH 5분 ~23bp)")
    log(f"   비용/ATR = {COST_BP/np.median(atr_bp)*100:.1f}%  (5분봉 62%, ETH 43%)")

    def net_for(sub, cell):
        sl, arm, trv = cell
        idx = sub["bar_idx"].to_numpy().astype(int)
        sgn = np.where(sub["is_downside"].to_numpy() == 1, 1.0, -1.0)
        at = sub["atr"].to_numpy(dtype=float)
        out = np.full(len(sub), np.nan); ex = np.full(len(sub), 0)
        for s_ in range(0, len(sub), CHUNK):
            e_ = min(s_ + CHUNK, len(sub)); j = idx[s_:e_]
            H = np.stack([h_[x+1:x+1+FORWARD_BARS] for x in j])
            L = np.stack([l_[x+1:x+1+FORWARD_BARS] for x in j])
            C = np.stack([c[x+1:x+1+FORWARD_BARS] for x in j])
            pn, e2 = sim_exit(o[j+1], at[s_:e_], sgn[s_:e_], H, L, C, sl, arm, trv)
            out[s_:e_] = pn * 1e4 - COST_BP; ex[s_:e_] = e2
        return out, ex

    log("")
    log(f"=== 1) TRAIN exit 셀 격자 (손익비>={PAYOFF_FLOOR}, 라벨률 0.55~0.85) ===")
    cands = []
    for sl in SL_GRID:
        for arm in ARM_GRID:
            for trv in TRAIL_GRID:
                v, _ = net_for(tr, (sl, arm, trv))
                v = v[np.isfinite(v)]
                if len(v) < 500:
                    continue
                w = v > 0
                po = float(v[w].mean() / -v[~w].mean()) if w.any() and (~w).any() else None
                r = {"cell": [sl, arm, trv], "label_rate": float(w.mean()),
                     "mean_bp": float(v.mean()), "median_bp": float(np.median(v)), "payoff": po}
                cands.append(r)
    ok = [r for r in cands if 0.55 <= r["label_rate"] <= 0.85 and (r["payoff"] or 0) >= PAYOFF_FLOOR]
    if not ok:
        log(f"  ⚠️조건 만족 셀 없음 -- 라벨률 범위만 적용")
        ok = [r for r in cands if 0.55 <= r["label_rate"] <= 0.85] or cands
    best_lab = max(ok, key=lambda r: r["mean_bp"])
    log(f"  ⭐라벨 셀: {best_lab['cell']}  라벨률 {best_lab['label_rate']:.4f}  "
        f"평균 {best_lab['mean_bp']:+.2f}bp  손익비 {best_lab['payoff']:.3f}")
    base = float(np.mean([r["mean_bp"] for r in cands]))
    log(f"  (참고) 전 격자 평균 net {base:+.2f}bp -- 5분봉 무작위 기준선은 -1.04bp였다")

    v_all, _ = net_for(long, tuple(best_lab["cell"]))
    long["y"] = (v_all > 0).astype(float)
    log(f"  전체 라벨률 {long['y'].mean():.4f} "
        f"(bottom {long.loc[long.is_downside==1,'y'].mean():.4f} / top {long.loc[long.is_downside==0,'y'].mean():.4f})")

    log("")
    log("=== 2) 5시드 앙상블 ===")
    tr_set = long.loc[long["split"] == "TRAIN"]
    probs = {"VAL": [], "OOS": []}
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        ctx = tr_set.iloc[np.sort(rng.choice(len(tr_set), size=min(CONTEXT_N, len(tr_set)), replace=False))]
        clf = TabPFNClassifier(device="cuda", random_state=sd, ignore_pretraining_limits=True)
        clf.fit(ctx[TIER0], ctx["y"].to_numpy())
        for spn in ("VAL", "OOS"):
            s = long.loc[long["split"] == spn]
            probs[spn].append(np.concatenate([clf.predict_proba(s[TIER0].iloc[k:k+20000])[:, 1]
                                              for k in range(0, len(s), 20000)]))
        log(f"  seed {sd} 완료")
    scored = {}
    aucs = {}
    for spn in ("VAL", "OOS"):
        s = long.loc[long["split"] == spn].copy()
        P = np.vstack(probs[spn]); s["p"] = P.mean(axis=0); scored[spn] = s
        lb = s.loc[s["y"].notna()]
        aucs[spn] = float(roc_auc_score(lb["y"], lb["p"])) if lb["y"].nunique() == 2 else None
        log(f"  {spn}: {len(s):,}행  AUC {aucs[spn]:.4f}  시드std {np.median(P.std(axis=0)):.4f}")

    def cand_df(s, cut, cell, flip=False):
        sel = s.loc[s["p"] >= cut]
        if len(sel) < 20:
            return None
        sub = sel.copy()
        if flip:
            sub["is_downside"] = 1 - sub["is_downside"]
        v, ex = net_for(sub, cell)
        idx = sel["bar_idx"].to_numpy().astype(int)
        return pd.DataFrame({"timestamp": sel["timestamp"].to_numpy(), "entry_bar": idx + 1,
                             "exit_bar": idx + 1 + ex, "pnl_bp": v})

    log("")
    log("=== 3) VAL 순차 포트폴리오 -> OOS 1회 ===")
    best = None
    for frac in TAIL_FRACS:
        k = max(20, int(round(len(scored["VAL"]) * frac)))
        cut = float(scored["VAL"].nlargest(k, "p")["p"].min())
        for sl in SL_GRID:
            for arm in ARM_GRID:
                for trv in TRAIL_GRID:
                    cd = cand_df(scored["VAL"], cut, (sl, arm, trv))
                    if cd is None:
                        continue
                    for mc in MAX_CONCURRENT:
                        r = portfolio(cd, mc)
                        if r is None or r["n"] < 20 or (r.get("payoff") or 0) < PAYOFF_FLOOR:
                            continue
                        if best is None or r["total_bp"] > best["r"]["total_bp"]:
                            best = {"frac": frac, "cut": cut, "cell": (sl, arm, trv), "mc": mc, "r": r}
    if best is None:
        log("  ⚠️손익비 하한을 만족하는 VAL 조합이 없다 -- 5분봉 2차와 같은 결과")
        report = {"asset": "BTCUSDT", "bar": "1h", "auc": aucs, "label_cell": best_lab,
                  "val_selection": None, "passed": False,
                  "note": "VAL에서 손익비 하한 만족 조합 없음"}
    else:
        b = best
        log(f"  [VAL 선정] 상위{b['frac']*100:g}% (p>={b['cut']:.4f}) 셀{b['cell']} 한도{b['mc']}  "
            f"n={b['r']['n']:,} 기대값 {b['r']['exp_bp']:+.2f}bp 총 {b['r']['total_bp']:+.0f}bp "
            f"승률 {b['r']['win_rate']*100:.1f}% 손익비 {b['r']['payoff']:.3f}")
        oos_res = {}
        for flip, nm in ((False, "정방향"), (True, "뒤집기")):
            cd = cand_df(scored["OOS"], b["cut"], b["cell"], flip=flip)
            if cd is None:
                continue
            r = portfolio(cd, b["mc"])
            if r is None:
                continue
            mo = pd.Series(r["pnl"], index=pd.to_datetime(r["ts"])).groupby(
                pd.to_datetime(r["ts"]).to_period("M")).mean()
            log(f"  [OOS] {nm}: n={r['n']:,} 기대값 {r['exp_bp']:+.2f}bp 총 {r['total_bp']:+.0f}bp "
                f"승률 {r['win_rate']*100:.1f}% 손익비 {r['payoff']} DD {r['max_dd_bp']:+.0f}bp")
            log(f"        월별: " + "  ".join(f"{k} {v:+.2f}" for k, v in mo.items()))
            oos_res[nm] = {**{k2: (round(v, 3) if isinstance(v, float) else v)
                              for k2, v in r.items() if k2 not in ("idx", "pnl", "ts")},
                           "monthly_exp_bp": {str(k): round(float(v), 2) for k, v in mo.items()}}
        fwd, flp = oos_res.get("정방향"), oos_res.get("뒤집기")
        passed = bool(fwd and fwd["total_bp"] > 0 and flp and fwd["exp_bp"] > flp["exp_bp"]
                      and sum(1 for v in fwd["monthly_exp_bp"].values() if v > 0) >= 2)
        log("")
        log(f"=== 판정: {'✅통과' if passed else '❌미통과'} ===")
        for spn in ("VAL", "OOS"):
            sel = scored[spn].loc[scored[spn]["p"] >= b["cut"]]
            nl = int((sel["is_downside"] == 1).sum())
            log(f"  측면 {spn}: 롱 {nl}/{len(sel)} ({nl/max(len(sel),1)*100:.1f}%)")
        report = {"asset": "BTCUSDT", "bar": "1h", "forward_bars": FORWARD_BARS,
                  "auc": aucs, "label_cell": best_lab,
                  "atr_bp_median": round(float(np.median(atr_bp)), 2),
                  "cost_over_atr_pct": round(COST_BP / float(np.median(atr_bp)) * 100, 1),
                  "val_selection": {"frac": b["frac"], "cut": round(b["cut"], 4),
                                    "cell": list(b["cell"]), "max_concurrent": b["mc"],
                                    **{k: (round(v, 3) if isinstance(v, float) else v)
                                       for k, v in b["r"].items() if k not in ("idx", "pnl", "ts")}},
                  "oos": oos_res, "passed": passed}
    report["scope"] = {"holdout_touched": False, "live_code_changed": False,
                       "payoff_floor": PAYOFF_FLOOR, "seeds": SEEDS, "cost_bp": COST_BP}
    report["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
