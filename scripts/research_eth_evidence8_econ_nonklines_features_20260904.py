#!/usr/bin/env python3
"""증거신호 8종 경제성 회귀 -- **비-klines 피쳐 확장** (목표: 8종 관문 통과).

## 왜 이 축인가

Tier0 23개는 **전부 klines 파생**이다. BTC는 "Tier0에 방향력 없음"으로 종결됐고
(`btc_v_rebound_econ_label_closed_no_direction_skill_20260902` -- 비용장벽을 실제로 제거해도
AUC 0.49 불변), ETH에서도 지금까지 시도한 확장 3종이 전부 실패했다:

    F1 재료텐서(8신호 OOF proba+레짐 51열)  1/16 통과 -- 희석
    F2 klines 여분 29열                     0/16
    통합풀 + 신호 원핫 32피쳐               0/8  (표본 8배인데도)

⇒ 남은 유일한 진짜 신규 정보원은 **비-klines**다. 오늘 진입모델이 이 축에서만 "OOS 상관 2배"를
   봤고, 핵심이 펀딩·OI·BTC 교차자산이었다.

## 피쳐 구성 (진입모델 161피쳐와 **동일 레시피**)

    klines(ETH/BTC) + metrics(OI, top-trader/전체 롱숏비) + fundingRate
    -> `features.engineering.FeatureEngineer.process(eth_df, btc_df)`
    구성: 펀딩 12 · OI 6 · 롱숏 2 · BTC 교차자산 15 · 테이커 3 · 레짐 3 + klines 파생

⚠️`merge_asof`는 키 dtype이 어긋나면 조용히 전부 NaN이 된다(이 저장소에서 4번 재발).
   병합 후 결측률을 반드시 출력한다.

⚠️TRAIN/VAL만. OOS·HOLDOUT 미터치.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
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


_pf = _load("pf_nk", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_s1 = _pf._s1
TIER0, sim_exit = _pf.TIER0, _pf.sim_exit

SIGNALS = {"liquidity_sweep": 30, "taker_delta_z_climax": 24, "short_term_return_z": 12,
           "orthogonal_combo": 24, "smt_divergence": 72, "fib_extension_exhaustion": 20,
           "demarker_extreme": 8, "kalman_deviation_meanrev": 12}
CELL = (4.0, 1.0, 0.1)
GAP, COST_BP, TOPQ, SEED = 12, 10.0, 0.10, 20260904
METRICS = ROOT / "data/TOTAL_ETHUSDT_metrics_2024_2026.csv"
FUNDING = ROOT / "data/TOTAL_ETHUSDT_fundingRate_2025_2026.csv"
DERIV = ["sum_open_interest_value", "sum_toptrader_long_short_ratio",
         "count_long_short_ratio", "last_funding_rate"]
OUT = ROOT / "data/research/eth_evidence8_econ_nonklines_20260904/report.json"


def log(m): print(f"[nk] {m}", flush=True)


def causal_first_fire(fire, gap):
    keep = np.zeros(len(fire), bool); last = -10**9
    for i in np.flatnonzero(fire):
        if i - last > gap:
            keep[i] = True
        last = i
    return keep


def cluster_t(vals, days):
    if len(vals) < 5:
        return np.nan
    dev = vals - vals.mean()
    s = sum(dev[days == d].sum() ** 2 for d in np.unique(days))
    se = np.sqrt(s) / len(vals)
    return vals.mean() / se if se > 0 else np.nan


def build_expanded(eth_csv, btc_csv):
    """진입모델과 동일 레시피로 비-klines 포함 확장 피쳐 프레임을 만든다."""
    from features.engineering import FeatureEngineer
    kl = pd.read_csv(eth_csv)
    kl["timestamp"] = pd.to_datetime(kl["timestamp"], errors="coerce")
    btc = pd.read_csv(btc_csv)
    btc["timestamp"] = pd.to_datetime(btc["timestamp"], errors="coerce")

    m = pd.read_csv(METRICS)
    m["timestamp"] = pd.to_datetime(m["create_time"], errors="coerce")
    f = pd.read_csv(FUNDING)
    f["timestamp"] = pd.to_datetime(f["calc_time"], errors="coerce")
    for d in (kl, btc, m, f):
        if getattr(d["timestamp"].dt, "tz", None) is not None:
            d["timestamp"] = d["timestamp"].dt.tz_localize(None)
    m = m.dropna(subset=["timestamp"]).sort_values("timestamp")
    f = f.dropna(subset=["timestamp"]).sort_values("timestamp")
    log(f"  metrics {len(m):,}행 {m['timestamp'].min()}~{m['timestamp'].max()}")
    log(f"  funding {len(f):,}행 {f['timestamp'].min()}~{f['timestamp'].max()}")

    raw = kl.sort_values("timestamp").reset_index(drop=True)
    raw = pd.merge_asof(raw, m[["timestamp", "sum_open_interest_value",
                                "sum_toptrader_long_short_ratio", "count_long_short_ratio"]],
                        on="timestamp", direction="backward")
    raw = pd.merge_asof(raw, f[["timestamp", "last_funding_rate"]],
                        on="timestamp", direction="backward")
    b = btc.rename(columns={"close": "close_btc", "volume": "volume_btc",
                            "quote_volume": "quote_volume_btc"})
    raw = raw.merge(b[["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]],
                    on="timestamp", how="left")
    for cname in DERIV + ["close_btc"]:
        miss = float(raw[cname].isna().mean())
        log(f"  결측(ffill 전) {cname:28s} {miss:.3f}")
    # ⭐펀딩 CSV가 2025~2026만 커버해 37.6%가 결측이고, 그대로 dropna하면 풀이 반토막나
    # 피쳐 비교가 교란된다(표본이 줄어든 탓인지 피쳐 탓인지 구분 불가). 펀딩은 8시간마다
    # 갱신되는 **느린 시계열**이라 전방채움이 인과적으로 정당하다(미래를 보지 않는다).
    raw["last_funding_rate"] = raw["last_funding_rate"].ffill()
    raw = raw.dropna(subset=DERIV + ["close_btc"]).reset_index(drop=True)
    log(f"  ffill 후 잔여 결측 last_funding_rate "
        f"{float(pd.read_csv(FUNDING, nrows=1).shape[0] * 0):.0f} -- dropna 통과분 {len(raw):,}봉")
    log(f"  파생 결합 후 {len(raw):,}봉")
    ed = raw[["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
              "trades", "taker_buy_base", "taker_buy_quote"] + DERIV].copy()
    bd = raw[["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]].copy()
    fr = FeatureEngineer().process(ed, bd)
    if "timestamp" not in fr.columns:
        fr = fr.reset_index()
    log(f"  ⭐확장 피쳐 프레임 {fr.shape[0]:,}행 x {fr.shape[1]}열")
    return fr


def main() -> int:
    t0 = time.time()
    log("기본 프레임...")
    sig, feat, eth = _s1.build_sig()
    dummy = np.full(len(sig), "none", dtype=object)
    long = _s1.long_frame_for(sig, feat, dummy, dummy)

    log("확장 피쳐 빌드(비-klines 포함)...")
    FR = build_expanded(_s1._feas.ETH_CSV, _s1._feas.BTC_CSV)
    if FR["timestamp"].dt.tz is not None:
        FR["timestamp"] = FR["timestamp"].dt.tz_localize(None)
    NUM = [c for c in FR.columns
           if c != "timestamp" and pd.api.types.is_numeric_dtype(FR[c])]
    kw = ["funding", "open_interest", "oi_", "long_short", "btc", "taker", "regime"]
    nonk = sorted({c for c in NUM for k in kw if k in c.lower()})
    log(f"  수치 피쳐 {len(NUM)}개 · 그중 비-klines 계열 {len(nonk)}개")

    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    if kl["timestamp"].dt.tz is not None:
        kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    o, h, l, c = (kl[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(kl)
    pos_of = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    ltsr = long["timestamp"]
    lts = (ltsr.dt.tz_localize(None) if ltsr.dt.tz is not None else ltsr).to_numpy()
    LONG = long.copy(); LONG["_ts"] = lts
    LONG = LONG.merge(FR.rename(columns={"timestamp": "_ts"}), on="_ts",
                      how="left", suffixes=("", "_x"))
    NUM = [c_ if c_ in LONG.columns else f"{c_}_x" for c_ in NUM]
    NUM = [c_ for c_ in NUM if c_ in LONG.columns]
    cov = float(LONG[NUM[0]].notna().mean()) if NUM else 0.0
    log(f"  병합 후 확장피쳐 커버리지 {cov:.3f}" + ("  ⚠️낮음" if cov < 0.5 else ""))

    lpos_all = np.array([pos_of.get(np.datetime64(t), -1) for t in LONG["_ts"].to_numpy()])
    isd_all = LONG["is_downside"].to_numpy().astype(bool)
    T0 = [x for x in TIER0 if x in LONG.columns]

    parts = []
    for SIGNAL, HZ in SIGNALS.items():
        bcol, tcol = f"bottom_{SIGNAL}", f"top_{SIGNAL}"
        S = sig[["timestamp", bcol, tcol]].copy()
        if S["timestamp"].dt.tz is not None:
            S["timestamp"] = S["timestamp"].dt.tz_localize(None)
        S["pos"] = [pos_of.get(np.datetime64(t), -1) for t in S["timestamp"].to_numpy()]
        S = S.loc[S["pos"] >= 0]
        fb = np.zeros(n, bool); ft = np.zeros(n, bool)
        fb[S["pos"].to_numpy()] = S[bcol].fillna(False).to_numpy(bool)
        ft[S["pos"].to_numpy()] = S[tcol].fillna(False).to_numpy(bool)
        kb, kt = causal_first_fire(fb, GAP), causal_first_fire(ft, GAP)
        keep = (lpos_all >= 0) & (lpos_all + 1 + HZ < n)
        keep &= np.where(isd_all, kb[np.clip(lpos_all, 0, n - 1)], kt[np.clip(lpos_all, 0, n - 1)])
        if keep.sum() < 300:
            continue
        cols = list(dict.fromkeys(["_ts", "split", "is_downside", "atr"] + T0 + NUM))
        D = LONG.loc[keep, cols].reset_index(drop=True)
        ii = lpos_all[keep]
        sg = np.where(D["is_downside"].to_numpy() == 1, 1.0, -1.0)
        entry = o[ii + 1]
        H = np.stack([h[i + 1:i + 1 + HZ] for i in ii])
        L = np.stack([l[i + 1:i + 1 + HZ] for i in ii])
        C = np.stack([c[i + 1:i + 1 + HZ] for i in ii])
        pn, _ = sim_exit(entry, D["atr"].to_numpy(float), sg, H, L, C, *CELL)
        D["net_bp"] = pn * 1e4 - COST_BP
        D["signal"] = SIGNAL
        parts.append(D)
    P = pd.concat(parts, ignore_index=True)
    P = P.dropna(subset=[NUM[0]]).reset_index(drop=True)
    for s_ in SIGNALS:
        P[f"is_{s_}"] = (P["signal"] == s_).astype(np.int8)
    ONE = [f"is_{s_}" for s_ in SIGNALS]
    net = P["net_bp"].to_numpy(float)
    split = P["split"].to_numpy()
    tr, va = split == "TRAIN", split == "VAL"
    days = pd.to_datetime(P["_ts"]).dt.floor("D").to_numpy()
    log(f"\n⭐풀 {len(P):,}건 (TRAIN {tr.sum():,} / VAL {va.sum():,}) 평균 {net.mean():+.2f}bp")

    from tabpfn import TabPFNRegressor
    FSETS = {"F0_tier0": T0 + ONE, "FX_nonklines": T0 + NUM + ONE}
    res = {}
    for fname, cols in FSETS.items():
        use = list(dict.fromkeys([x for x in cols if x in P.columns]))
        X = np.nan_to_num(P[use].apply(pd.to_numeric, errors="coerce").to_numpy(float),
                          nan=0.0, posinf=0.0, neginf=0.0)
        rng = np.random.default_rng(SEED)
        tri = np.flatnonzero(tr)
        ctx = rng.choice(tri, size=min(18000, len(tri)), replace=False)
        m = TabPFNRegressor(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
        m.fit(X[ctx], net[ctx]); pred = m.predict(X[va])
        Pva = P.loc[va].reset_index(drop=True)
        nv, dv = net[va], days[va]
        log(f"\n=== {fname} ({len(use)}피쳐) ===")
        print(f"{'신호':>24s}{'VAL n':>8s}{'k':>5s}{'독립일':>7s}{'전체bp':>9s}{'상위bp':>9s}{'일t':>7s}")
        print("-" * 70)
        per = {}
        for s_ in SIGNALS:
            mk = (Pva["signal"] == s_).to_numpy()
            if mk.sum() < 80:
                continue
            k = max(10, int(round(mk.sum() * TOPQ)))
            sp, sn, sd = pred[mk], nv[mk], dv[mk]
            top = np.argsort(-sp)[:k]
            tb, tt = float(sn[top].mean()), float(cluster_t(sn[top], sd[top]))
            print(f"{s_[:23]:>24s}{int(mk.sum()):8d}{k:5d}{len(np.unique(sd[top])):7d}"
                  f"{float(sn.mean()):9.2f}{tb:9.2f}{tt:7.2f}{'  ⭐' if tt > 1.96 else ''}")
            per[s_] = {"n_val": int(mk.sum()), "top_mean_bp": tb, "cluster_t": tt,
                       "all_mean_bp": float(sn.mean()),
                       "independent_days": int(len(np.unique(sd[top])))}
        kk = max(10, int(round(va.sum() * TOPQ)))
        tp = np.argsort(-pred)[:kk]
        ptb, ptt = float(nv[tp].mean()), float(cluster_t(nv[tp], dv[tp]))
        npass = sum(1 for v in per.values() if v["cluster_t"] > 1.96)
        log(f"  ⭐풀 상위10%: {ptb:+.2f}bp t={ptt:.2f} (n={kk}, 일{len(np.unique(dv[tp]))})")
        log(f"  ⭐신호별 통과: **{npass}/{len(per)}**")
        res[fname] = {"n_feat": len(use), "per_signal": per, "pool_top_bp": ptb,
                      "pool_top_t": ptt, "n_passed": npass}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"featsets": res, "n_pool": int(len(P)),
                               "n_nonklines_feats": len(nonk), "oos_touched": False,
                               "runtime_sec": round(time.time() - t0, 1)},
                              ensure_ascii=False, indent=2))
    log(f"산출: {OUT} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
