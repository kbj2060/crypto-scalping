#!/usr/bin/env python3
"""ETH **경제 축 지속 신호 스크린 v1** -- 차트 모양이 아니라 경제 상태 변수 하나의 극단으로 정의한 지속 트리거 (2026-09-04).

사용자: "다른 세션의 지속 신호 연구 결과를 토대로 다른 축의 지속 신호를 테스트. 모양을 맞추기보다 경제성을 기준으로."

배경(호메로스 §5.23/§5.25/§5.27): 반전 8종 첫발동의 지속 방향 규칙(R)만 VAL/OOS 두 창 CI>0. 추세 8종 v1은 차트 모양(돌파·되돌림·
세션 레인지)으로 정의했고 발동의 70%+가 R과 같은 순간이라 아카이브. 여기서는 트리거를 **경제 상태 변수 하나의 극단**
(TRAIN 분위수 임계, 방향 = 변수 부호)으로 정의한다 -- 조합 조건·패턴 없음. 각 축은 한 가지 경제 메커니즘을 대표한다.

## B: 독립 사건원 (트리거 = TRAIN 95분위 극단, 첫발동 GAP12 측면별, 방향 규칙은 아래에 고정)
  B1  taker_flow        퍼프 테이커 불균형 3봉 합 z288                        방향 = 부호     (공격적 체결 흐름)
  B2  oi_build_move     ΔOI 30분 z288 상위 5%  ∧ 방향 = sign(ret3)                          (신규 포지션이 미는 움직임)
  B3  oi_unwind_move    ΔOI 30분 z288 하위 5%  ∧ 방향 = sign(ret3)                          (강제청산/포지션 해소가 미는 움직임)
  B4  basis_change      퍼프−현물 베이시스 3봉 변화 z288                      방향 = 부호     (파생 프리미엄 확장)
  B5  btc_shock         BTC ret3 z288                                         방향 = 부호     (교차자산 충격 전파)
  B6  market_beta_move  알트 16종 σ-정규화 ret3 횡단 평균 z288                방향 = 부호     (시장 전체 이동)
  B7  eth_idio_move     ETH σ-ret3 − 시장 평균 z288                           방향 = 부호     (ETH 고유 이동)
  B8  depth_imbalance   bookDepth ±1% (bid−ask)/(bid+ask) z288                방향 = 부호     (OFI 문헌: bid 깊이 우위 → 상승)
  B9  retail_shift      Δ count_long_short_ratio 30분 z288                    방향 = −부호    (개미 쏠림 변화의 반대)
  B10 toptrader_shift   Δ sum_toptrader_long_short_ratio 30분 z288            방향 = +부호    (상위 트레이더 편)
  B11 activity_burst    체결 건수 z288 상위 5% ∧ 방향 = sign(ret1)                            (진단: 반전 8종과 가장 가까운 축)

## 평가 (전부 F0 프레임 두 측면 경제라벨 상속: open[i+1] 진입, sim_exit 5.0/1.5/0.1 ATR, 200봉, 10bp 차감)
  1) 단독 경제성: n, /일, P(신호>반대), 순손익 차이 일군집 CI, 동시5 순차 포트폴리오 exp + 일CI, 측면비율 매칭 무작위귀무 백분위(B=200)
  2) R 독립성: 발동 직전 12봉(과거만, 같은 봉 포함) 안 같은 방향 R 첫발동 존재 비율 -- 미래 창 금지(§5.27 7-3 룩어헤드 함정)
  3) 증분: (R ∪ B) − R 일별 짝비교(자본 대비 bp/일, 일 부트스트랩 CI, 이긴 날 비율)

## 판정 (결과 보기 전 고정 -- 추세 v1 규칙 상속)
  TRAIN n ≥ 300 ∧ P ≥ 0.53 ∧ 차이 일CI 하한 > 0, VAL·OOS 차이 > 0 둘 다 → PASS / 한쪽 → WEAK / 그 외 REJECT
  PASS 축에 한해: 과거12봉 R 겹침 < 50% → "독립", 짝비교 두 창 > 0 → "ADD 후보". q90/q98 변형·셀·진입지연·비용은 보고 전용(선택 아님).

## A: 같은 축을 R 지속 거래의 **상태**로 (발동 봉 값 × 지속 방향 부호 = 정렬값, TRAIN 5분위)
  분위별 지속 net_bp, 상위−하위 분위 갭 일CI, TRAIN 최악 분위 제거 필터의 R 대비 일별 짝비교.
  판정: TRAIN 갭 CI 하한 > 0 ∧ VAL·OOS 갭 > 0 → 상태 축 "작동"; 필터는 짝비교 두 창 CI 하한 > 0 이어야 채택.

HOLDOUT(≥2026-04-01) 미접촉. 연구/개발 점수 -- 승격은 층 게이트·전진 섀도우.
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
sys.path.insert(0, str(ROOT / "scripts"))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


TS = _load("trend_v1_mod", "scripts/research_eth_trend_signals_v1_screen_20260904.py")   # 로더·roll_z·first_fire
FC = _load("fire_cont_mod", "scripts/research_eth_evidence_fire_continuation_econ_20260904.py")  # load_fires, V2.sim_exit
M = _load("decide_mod", "scripts/research_eth_live_decision_algorithm_v1_20260904.py")   # portfolio, day_ci
OUT = ROOT / "data/research/eth_econ_axis_continuation_screen_20260904"
BASKET = ["BTCUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT",
          "DOTUSDT", "TRXUSDT", "BCHUSDT", "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "UNIUSDT"]
Q_PRIMARY, Q_VARIANTS = 0.95, (0.90, 0.98)
GAP, CAP, B_NULL, B_BOOT, PAST_W = 12, 5, 200, 1000, 12
WINDOWS = ("TRAIN", "VAL", "OOS")
TRAIN_START, TRAIN_END = pd.Timestamp("2024-05-01"), pd.Timestamp("2025-09-01")
CELLS_ROBUST = [(5.0, 1.5, 0.1), (4.0, 1.0, 0.1), (3.0, 1.5, 0.1), (4.0, 2.0, 0.1), (5.0, 1.5, 0.05), (5.0, 1.0, 0.1)]
# 축 정의: kind = signed(방향=부호·sign_mult) | cond_hi(상위 분위 ∧ 방향=ret) | cond_lo(하위 분위 ∧ 방향=ret)
AXES = {
    "taker_flow": ("signed", +1, None), "oi_build_move": ("cond_hi", None, "ret3"), "oi_unwind_move": ("cond_lo", None, "ret3"),
    "basis_change": ("signed", +1, None), "btc_shock": ("signed", +1, None), "market_beta_move": ("signed", +1, None),
    "eth_idio_move": ("signed", +1, None), "depth_imbalance": ("signed", +1, None), "retail_shift": ("signed", -1, None),
    "toptrader_shift": ("signed", +1, None), "activity_burst": ("cond_hi", None, "ret1"),
}
AXIS_SRC = {"taker_flow": "taker_flow", "oi_build_move": "oi_chg6", "oi_unwind_move": "oi_chg6", "basis_change": "basis_change",
            "btc_shock": "btc_shock", "market_beta_move": "market_beta_move", "eth_idio_move": "eth_idio_move",
            "depth_imbalance": "depth_imbalance", "retail_shift": "retail_shift", "toptrader_shift": "toptrader_shift", "activity_burst": "activity"}
# A(상태) 축: 정렬값 = x × 지속 방향 부호 (signed) / raw (state)
STATE_AXES = {"taker_flow": "aligned", "oi_chg6": "raw", "basis_change": "aligned", "btc_shock": "aligned", "market_beta_move": "aligned",
              "eth_idio_move": "aligned", "depth_imbalance": "aligned", "retail_shift": "aligned", "toptrader_shift": "aligned",
              "funding_z": "aligned", "activity": "raw"}
rng = np.random.default_rng(20260904)


def log(m): print(f"[econ-axis] {m}", flush=True)


# ----------------------------------------------------------------------------- 축 피쳐 (전부 봉 마감 이전 값)
def build_axes():
    t0 = time.time()
    kl = TS.load_klines(TS.KL); n = len(kl)
    o, h, l, c, v, tb, ntr = (kl[x].to_numpy(float) for x in ("open", "high", "low", "close", "volume", "taker_buy_base", "trades"))
    ts = kl["timestamp"]; close_ts = ts + pd.Timedelta(minutes=5)
    prev = np.r_[np.nan, c[:-1]]; tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    ret1 = c / prev - 1.0; ret3 = c / np.r_[np.nan, np.nan, np.nan, c[:-3]] - 1.0
    X = {}
    imb3 = pd.Series(2 * tb - v).rolling(3).sum() / pd.Series(v).rolling(3).sum().replace(0, np.nan)
    X["taker_flow"] = TS.roll_z(imb3).to_numpy()
    X["activity"] = TS.roll_z(pd.Series(ntr)).to_numpy()
    # metrics 5m (close_ts 이전 스냅샷; 15분 허용) + 1봉 지연 변형
    met = TS.load_metrics()
    for tag, at in (("", close_ts), ("_lag1", close_ts - pd.Timedelta(minutes=5))):
        m = pd.merge_asof(pd.DataFrame({"ts": at}), met, on="ts", direction="backward", tolerance=pd.Timedelta(minutes=15))
        oi = m["sum_open_interest_value"].to_numpy(float); X["oi_chg6" + tag] = TS.roll_z(pd.Series(oi / np.r_[[np.nan] * 6, oi[:-6]] - 1.0)).to_numpy()
        lsr = m["count_long_short_ratio"].to_numpy(float); X["retail_shift" + tag] = TS.roll_z(pd.Series(lsr - np.r_[[np.nan] * 6, lsr[:-6]])).to_numpy()
        top = m["sum_toptrader_long_short_ratio"].to_numpy(float); X["toptrader_shift" + tag] = TS.roll_z(pd.Series(top - np.r_[[np.nan] * 6, top[:-6]])).to_numpy()
    fu = TS.load_funding(); f = pd.merge_asof(pd.DataFrame({"ts": close_ts}), fu, on="ts", direction="backward"); X["funding_z"] = f["funding_z"].to_numpy(float)
    # 베이시스 (퍼프/현물 − 1) 3봉 변화
    spot = TS.load_klines(TS.KL_SPOT, "spot_"); sc = kl[["timestamp"]].merge(spot[["timestamp", "spot_close"]], on="timestamp", how="left")["spot_close"].ffill().to_numpy(float)
    basis = c / sc - 1.0; X["basis_change"] = TS.roll_z(pd.Series(basis - np.r_[np.nan, np.nan, np.nan, basis[:-3]])).to_numpy()
    # BTC 충격 + 시장 바스켓
    S = []; used = []
    for sym in BASKET:
        p = ROOT / f"binance_data/klines/{sym}/{sym}-5m-api.csv"
        if not p.exists():
            continue
        d = TS.load_klines(p, f"{sym}_"); cc = kl[["timestamp"]].merge(d[["timestamp", f"{sym}_close"]], on="timestamp", how="left")[f"{sym}_close"].to_numpy(float)
        r3 = cc / np.r_[np.nan, np.nan, np.nan, cc[:-3]] - 1.0; sd = pd.Series(r3).rolling(288, min_periods=144).std().to_numpy()
        S.append(r3 / np.where(sd > 0, sd, np.nan)); used.append(sym)
        if sym == "BTCUSDT":
            X["btc_shock"] = TS.roll_z(pd.Series(r3)).to_numpy()
    S = np.column_stack(S); nfin = np.isfinite(S).sum(1); market = np.where(nfin >= 8, np.nanmean(np.where(np.isfinite(S), S, np.nan), axis=1), np.nan)
    sd_e = pd.Series(ret3).rolling(288, min_periods=144).std().to_numpy(); eth_s = ret3 / np.where(sd_e > 0, sd_e, np.nan)
    X["market_beta_move"] = TS.roll_z(pd.Series(market)).to_numpy(); X["eth_idio_move"] = TS.roll_z(pd.Series(eth_s - market)).to_numpy()
    # bookDepth ±1% 불균형 (10분 허용)
    bd = TS.load_bookdepth(); b = pd.merge_asof(pd.DataFrame({"ts": close_ts}), bd[["ts", "up1", "dn1"]], on="ts", direction="backward", tolerance=pd.Timedelta(minutes=10))
    up1, dn1 = b["up1"].to_numpy(float), b["dn1"].to_numpy(float); X["depth_imbalance"] = TS.roll_z(pd.Series((dn1 - up1) / (dn1 + up1))).to_numpy()
    cov = {k: round(float(np.isfinite(x).mean()), 3) for k, x in X.items()}
    log(f"klines {n:,} · 바스켓 {len(used)} {used} · 커버리지 {cov} ({time.time()-t0:.0f}s)")
    F = pd.DataFrame(X); F.insert(0, "timestamp", ts.to_numpy()); F["ret1"] = ret1; F["ret3"] = ret3; F["atr"] = atr
    return F, {"basket": used, "coverage": cov}


# ----------------------------------------------------------------------------- 트리거
def triggers(F, name, q, train_mask):
    kind, smult, dircol = AXES[name]; x = F[AXIS_SRC[name]].to_numpy(float)
    if kind == "signed":
        t = float(np.nanquantile(np.abs(x[train_mask]), q)); up = x * smult >= t; dn = x * smult <= -t
    else:
        r = F[dircol].to_numpy(float)
        if kind == "cond_hi":
            t = float(np.nanquantile(x[train_mask], q)); trig = x >= t
        else:
            t = float(np.nanquantile(x[train_mask], 1 - q)); trig = x <= t
        up = trig & (r > 0); dn = trig & (r < 0)
    up = np.nan_to_num(up.astype(float)).astype(bool); dn = np.nan_to_num(dn.astype(float)).astype(bool)
    return up, dn, t


def rows_for(key, ts, up, dn):
    parts = []
    for mask, isd, side in ((up, 1, "up"), (dn, 0, "dn")):
        ff = TS.first_fire(mask, GAP); idx = np.flatnonzero(ff)
        r = key.reindex(pd.MultiIndex.from_arrays([ts[idx], np.full(len(idx), isd)], names=["timestamp", "is_downside"]))
        r = r[np.isfinite(r["net_bp"].to_numpy())].reset_index(); r["side"] = side; r["raw_fires"] = int(mask.sum()); parts.append(r)
    A = pd.concat(parts, ignore_index=True); A["trade_long"] = A["is_downside"] == 1
    A["entry_bar"] = A["pos"].astype(int) + 1; A["exit_bar"] = A["entry_bar"] + A["exit_off"].astype(int); A["pnl_bp"] = A["net_bp"]
    return A.sort_values("pos").reset_index(drop=True)


# ----------------------------------------------------------------------------- 통계
def day_ci_of(x, t, B=B_BOOT):
    d = pd.Series(np.asarray(x, float), index=pd.DatetimeIndex(t).normalize()); g = d.groupby(level=0)
    sums = g.sum().to_numpy(); cnts = g.count().to_numpy(); nd = len(sums); out = np.empty(B)
    for k in range(B):
        j = rng.integers(0, nd, nd); out[k] = sums[j].sum() / max(cnts[j].sum(), 1)
    return [round(float(np.percentile(out, 2.5)), 2), round(float(np.percentile(out, 97.5)), 2)]


def gap_day_ci(x_hi, t_hi, x_lo, t_lo, B=B_BOOT):
    """두 부분집합 평균 차이의 일군집 부트스트랩 (날짜 합집합에서 재표집)."""
    dh = pd.DatetimeIndex(t_hi).normalize().to_numpy(); dl = pd.DatetimeIndex(t_lo).normalize().to_numpy(); days = np.unique(np.concatenate([dh, dl]))
    ih = {d: np.flatnonzero(dh == d) for d in days}; il = {d: np.flatnonzero(dl == d) for d in days}; out = []
    for _ in range(B):
        pick = rng.choice(days, len(days), replace=True)
        a = np.concatenate([ih[d] for d in pick]); b = np.concatenate([il[d] for d in pick])
        if len(a) and len(b):
            out.append(x_hi[a].mean() - x_lo[b].mean())
    return [round(float(np.percentile(out, 2.5)), 2), round(float(np.percentile(out, 97.5)), 2)]


def econ(A):
    if len(A) < 20:
        return {"n": int(len(A))}
    days = pd.DatetimeIndex(A["timestamp"]).normalize().nunique(); diff = (A["net_bp"] - A["net_bp_flip"]).to_numpy()
    return {"n": int(len(A)), "per_day": round(len(A) / max(days, 1), 2), "p_sig_gt_opp": round(float((A["net_bp"] > A["net_bp_flip"]).mean()), 3),
            "sig_bp": round(float(A["net_bp"].mean()), 2), "opp_bp": round(float(A["net_bp_flip"].mean()), 2), "diff_bp": round(float(diff.mean()), 2),
            "diff_day_ci95": day_ci_of(diff, A["timestamp"]), "long_share": round(float(A["trade_long"].mean()), 3)}


def dedupe(C):
    C = C.drop_duplicates(["entry_bar", "trade_long"]); conflict = C.groupby("entry_bar")["trade_long"].transform("nunique") > 1
    return C[~conflict].reset_index(drop=True), int(C.loc[conflict, "entry_bar"].nunique())


def pf(C):
    r = M.portfolio(C, CAP) if len(C) else None
    if r is None:
        return None, None
    ci, nd = M.day_ci(r["pnl"], r["ts"])
    return {"n": r["n"], "per_day": round(r["n"] / max(nd, 1), 2), "exp_bp": round(r["exp_bp"], 2), "day_ci95": ci, "win_rate": round(r["win_rate"], 3),
            "max_dd_bp": round(r["max_dd_bp"], 1), "total_bp": round(r["total_bp"], 1)}, r


def daily(pnl, ts):
    return pd.Series(np.asarray(pnl, float) / CAP, index=pd.DatetimeIndex(ts).normalize()).groupby(level=0).sum()


def day_paired(rA, rB, B=B_BOOT):
    """A − B 일별 짝비교 (자본 대비 bp/일; 일 부트스트랩)."""
    a = daily(rA["pnl"], rA["ts"]); b = daily(rB["pnl"], rB["ts"]); days = a.index.union(b.index)
    d = (a.reindex(days, fill_value=0.0) - b.reindex(days, fill_value=0.0)).to_numpy(); out = np.empty(B)
    for k in range(B):
        out[k] = d[rng.integers(0, len(d), len(d))].mean()
    nz = d[d != 0]
    return {"mean_bp_per_day": round(float(d.mean()), 2), "ci95": [round(float(np.percentile(out, 2.5)), 2), round(float(np.percentile(out, 97.5)), 2)],
            "win_day_frac": round(float((nz > 0).mean()), 3) if len(nz) else None, "n_days": int(len(d)), "n_days_differ": int(len(nz))}


def side_null(D_w, n_long, n_short, obs):
    pool_l = D_w.loc[D_w["is_downside"] == 1]; pool_s = D_w.loc[D_w["is_downside"] == 0]; vals = []
    for _ in range(B_NULL):
        a = pool_l.iloc[rng.choice(len(pool_l), size=min(n_long, len(pool_l)), replace=False)]
        b = pool_s.iloc[rng.choice(len(pool_s), size=min(n_short, len(pool_s)), replace=False)]
        x = pd.concat([a, b]); cand = pd.DataFrame({"timestamp": x["timestamp"].to_numpy(), "entry_bar": x["pos"].to_numpy() + 1,
                                                    "exit_bar": x["pos"].to_numpy() + 1 + x["exit_off"].to_numpy(), "pnl_bp": x["net_bp"].to_numpy()})
        r = M.portfolio(cand, CAP); vals.append(r["exp_bp"] if r else np.nan)
    vals = np.asarray(vals, float)
    return {"mean_bp": round(float(np.nanmean(vals)), 2), "p95_bp": round(float(np.nanpercentile(vals, 95)), 2), "percentile_of_obs": round(float((vals < obs).mean() * 100), 1)}


def overlap_past(A, R):
    """A 각 행: 같은 방향 R 지속 거래가 [pos−PAST_W, pos] 안에 있는가 (과거만)."""
    out = np.zeros(len(A), bool)
    for tl in (True, False):
        rp = np.sort(R.loc[R["trade_long"] == tl, "pos"].to_numpy()); m = (A["trade_long"] == tl).to_numpy(); ap = A.loc[m, "pos"].to_numpy()
        j = np.searchsorted(rp, ap, side="right") - 1; ok = (j >= 0) & (ap - rp[np.clip(j, 0, max(len(rp) - 1, 0))] <= PAST_W) if len(rp) else np.zeros(len(ap), bool)
        out[m] = ok
    return out


def robustness(A, D_pos_atr, kl_full, mask_w):
    """PASS 축 전용: 셀 6종 / 진입 1봉 지연 / 비용 15bp 의 포트폴리오 exp (F0 라벨과 같은 sim_exit)."""
    o, h, l, c = (kl_full[x].to_numpy(float) for x in ("open", "high", "low", "close")); FWD = 200
    kidx = pd.Series(np.arange(len(kl_full)), index=pd.DatetimeIndex(kl_full["timestamp"]))
    kp = kidx.reindex(A["timestamp"]).to_numpy(); ok = np.isfinite(kp); kp = kp[ok].astype(int); A = A[ok]
    kp_ok = kp + FWD + 2 < len(o); kp = kp[kp_ok]; A = A[kp_ok]
    atr = D_pos_atr.reindex(A["pos"]).to_numpy(float); sign = np.where(A["trade_long"].to_numpy(), 1.0, -1.0); ts = A["timestamp"].to_numpy()
    H = np.stack([h[j + 1:j + 1 + FWD] for j in kp]); L = np.stack([l[j + 1:j + 1 + FWD] for j in kp]); C = np.stack([c[j + 1:j + 1 + FWD] for j in kp]); entry = o[kp + 1]
    base, ex0 = FC.V2.sim_exit(entry, atr, sign, H, L, C, 5.0, 1.5, 0.1); base = base * 1e4 - 10.0
    par = float(np.nanmax(np.abs(base - A["net_bp"].to_numpy()))); rob = {"label_parity_max_abs_bp": round(par, 6)}
    mw = mask_w[ok][kp_ok]

    def _pf(pnl, ex, offset=1):
        cand = pd.DataFrame({"timestamp": ts[mw], "entry_bar": A["pos"].to_numpy()[mw] + offset, "exit_bar": A["pos"].to_numpy()[mw] + offset + ex[mw], "pnl_bp": pnl[mw]})
        s, _ = pf(cand); return {"row_mean_bp": round(float(pnl[mw].mean()), 2), "pf_exp_bp": s["exp_bp"] if s else None, "day_ci95": s["day_ci95"] if s else None}
    for cell in CELLS_ROBUST:
        pn, ex = FC.V2.sim_exit(entry, atr, sign, H, L, C, *cell); rob[f"cell_{cell[0]}_{cell[1]}_{cell[2]}"] = _pf(pn * 1e4 - 10.0, ex)
    H1 = np.stack([h[j + 2:j + 1 + FWD] for j in kp]); L1 = np.stack([l[j + 2:j + 1 + FWD] for j in kp]); C1 = np.stack([c[j + 2:j + 1 + FWD] for j in kp])
    pn, ex = FC.V2.sim_exit(o[kp + 2], atr, sign, H1, L1, C1, 5.0, 1.5, 0.1); rob["delay_1bar"] = _pf(pn * 1e4 - 10.0, ex, offset=2)
    rob["cost_15bp"] = _pf(base - 5.0, ex0)
    return rob


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    F, meta = build_axes(); ts = F["timestamp"].to_numpy()
    train_mask = ((F["timestamp"] >= TRAIN_START) & (F["timestamp"] < TRAIN_END)).to_numpy()
    D = pd.read_parquet(FC.FRAME, columns=["pos", "is_downside", "timestamp", "split", "net_bp", "net_bp_flip", "exit_off", "atr"])
    D["timestamp"] = pd.to_datetime(D["timestamp"]); D["is_downside"] = D["is_downside"].astype(int)
    key = D.set_index(["timestamp", "is_downside"]); D_pos_atr = D.drop_duplicates("pos").set_index("pos")["atr"]
    split_of = D.drop_duplicates("timestamp").set_index("timestamp")["split"]
    base = {w: {"long": round(float(D.loc[(D.split == w) & (D.is_downside == 1), "net_bp"].mean()), 2), "short": round(float(D.loc[(D.split == w) & (D.is_downside == 0), "net_bp"].mean()), 2)} for w in WINDOWS}
    kl_full = pd.read_csv(FC.KL, usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    # ---- R: 반전 8종 첫발동 지속 규칙 (§5.23 모집단 원문) ----
    Fr = FC.load_fires(); Fr = Fr.loc[Fr["first_fire"]].drop_duplicates(["pos", "is_downside"])
    Rk = pd.MultiIndex.from_arrays([Fr["pos"].to_numpy(), 1 - Fr["is_downside"].to_numpy().astype(int)], names=["pos", "is_downside"])   # 지속 = 반대 측면 행
    Dp = D.set_index(["pos", "is_downside"]); rr = Dp.reindex(Rk).reset_index(); rr = rr[np.isfinite(rr["net_bp"].to_numpy())]
    R = pd.DataFrame({"timestamp": rr["timestamp"].to_numpy(), "pos": rr["pos"].astype(int).to_numpy(), "split": rr["split"].to_numpy(), "trade_long": (rr["is_downside"] == 1).to_numpy(),
                      "entry_bar": rr["pos"].astype(int).to_numpy() + 1, "exit_bar": rr["pos"].astype(int).to_numpy() + 1 + rr["exit_off"].astype(int).to_numpy(), "pnl_bp": rr["net_bp"].to_numpy(), "net_bp": rr["net_bp"].to_numpy()})
    R = R.sort_values("pos").reset_index(drop=True); R_pf = {}; R_run = {}; R_conf = {}
    for w in WINDOWS:
        Rw, nconf = dedupe(R[R.split == w]); s, r = pf(Rw); R_pf[w] = s; R_run[w] = r; R_conf[w] = nconf
    log(f"R 지속 규칙(동시5, 양측 충돌 봉 스킵): " + " · ".join(f"{w} n{R_pf[w]['n']} exp {R_pf[w]['exp_bp']:+.2f} CI {R_pf[w]['day_ci95']}" for w in WINDOWS) + f" (기대 §5.23: VAL +4.44 / OOS +6.78) ({time.time()-t0:.0f}s)")

    rep = {"prereg": __doc__.split("## 판정")[1].split("## A")[0].strip(), "baseline_net_bp_every_bar": base, "R_baseline": R_pf, "R_conflict_bars": R_conf,
           "meta": meta, "cap": CAP, "cost_bp": 10.0, "holdout_touched": False, "fresh_forward_bar_by_bar": False, "trade_ledgers_used_as_input": False,
           "future_rows_used_for_entry": False, "axes": {}, "state_axes": {}}
    print(f"\n{'axis':>18s} {'w':>5s} {'n':>5s} {'/day':>5s} {'P':>6s} {'sig':>7s} {'opp':>7s} {'diff[CI]':>20s} {'pf_exp[CI]':>22s} {'null%':>5s} {'R12%':>5s} {'union-R/day[CI] win':>26s}")
    for name in AXES:
        up, dn, thr = triggers(F, name, Q_PRIMARY, train_mask); A = rows_for(key, ts, up, dn); A["R_past12"] = overlap_past(A, R)
        A.to_parquet(OUT / f"triggers_{name}.parquet", index=False)
        Rr = {"threshold": round(thr, 4), "raw_fires_up_dn": [int(up.sum()), int(dn.sum())], "windows": {}}
        for w in WINDOWS:
            Aw = A[A.split == w].reset_index(drop=True); E = econ(Aw); Rw_ = {"econ_both": E}
            for side in ("up", "dn"):
                Rw_[f"econ_{side}"] = econ(Aw[Aw.side == side])
            if len(Aw) >= 20:
                Ad, nconf = dedupe(Aw); s, r = pf(Ad); Rw_["portfolio"] = s; Rw_["conflict_bars"] = nconf
                if s is not None:
                    Rw_["side_matched_null"] = side_null(D[D.split == w], int(Ad["trade_long"].sum()), int((~Ad["trade_long"]).sum()), s["exp_bp"])
                Rw_["R_past12_share"] = round(float(Aw["R_past12"].mean()), 3)
                U, nconf_u = dedupe(pd.concat([R[R.split == w], Aw[["timestamp", "pos", "split", "trade_long", "entry_bar", "exit_bar", "pnl_bp", "net_bp"]]], ignore_index=True))
                su, ru = pf(U); Rw_["union_portfolio"] = su
                if ru is not None and R_run[w] is not None:
                    Rw_["union_minus_R_day_paired"] = day_paired(ru, R_run[w])
                    Ub = U.iloc[ru["idx"]]; Rw_["union_new_trades"] = {"n": int((~Ub["pos"].isin(R.loc[R.split == w, "pos"])).sum())}
            Rr["windows"][w] = Rw_
            if E.get("n", 0) >= 20:
                dp = Rw_.get("union_minus_R_day_paired", {}); s = Rw_.get("portfolio") or {}
                print(f"{name:>18s} {w:>5s} {E['n']:5d} {E['per_day']:5.1f} {E['p_sig_gt_opp']:6.3f} {E['sig_bp']:7.2f} {E['opp_bp']:7.2f} {str(E['diff_bp'])+' '+str(E['diff_day_ci95']):>20s} "
                      f"{str(s.get('exp_bp'))+' '+str(s.get('day_ci95')):>22s} {str((Rw_.get('side_matched_null') or {}).get('percentile_of_obs')):>5s} {str(Rw_.get('R_past12_share')):>5s} "
                      f"{str(dp.get('mean_bp_per_day'))+' '+str(dp.get('ci95'))+' '+str(dp.get('win_day_frac')):>26s}")
        # 변형 (보고 전용)
        Rr["variants"] = {}
        for q in Q_VARIANTS:
            u2, d2, t2 = triggers(F, name, q, train_mask); A2 = rows_for(key, ts, u2, d2)
            Rr["variants"][f"q{int(q*100)}"] = {"threshold": round(t2, 4), **{w: econ(A2[A2.split == w]) for w in WINDOWS}}
        if name in ("oi_build_move", "oi_unwind_move", "retail_shift", "toptrader_shift"):    # metrics 1봉 지연 변형
            F2 = F.copy(); F2[AXIS_SRC[name]] = F[AXIS_SRC[name] + "_lag1"]
            u2, d2, t2 = triggers(F2, name, Q_PRIMARY, train_mask); A2 = rows_for(key, ts, u2, d2)
            Rr["variants"]["metrics_lag1"] = {"threshold": round(t2, 4), **{w: econ(A2[A2.split == w]) for w in WINDOWS}}
        tr_, va_, oo_ = (Rr["windows"][w]["econ_both"] for w in WINDOWS)
        train_ok = tr_.get("n", 0) >= 300 and tr_.get("p_sig_gt_opp", 0) >= 0.53 and tr_.get("diff_day_ci95", [-1])[0] > 0
        vo = int(va_.get("diff_bp", -1) > 0) + int(oo_.get("diff_bp", -1) > 0)
        Rr["verdict"] = "PASS" if (train_ok and vo == 2) else ("WEAK" if (train_ok and vo == 1) else "REJECT")
        if Rr["verdict"] == "PASS":
            ov = max(Rr["windows"][w].get("R_past12_share", 1.0) for w in ("VAL", "OOS")); Rr["independent_of_R"] = bool(ov < 0.5)
            dps = [Rr["windows"][w].get("union_minus_R_day_paired", {}).get("mean_bp_per_day", -1) for w in ("VAL", "OOS")]
            Rr["add_candidate"] = bool(all(x > 0 for x in dps))
            Rr["robustness"] = {w: robustness(A, D_pos_atr, kl_full, (A.split == w).to_numpy()) for w in ("VAL", "OOS")}
        rep["axes"][name] = Rr
        log(f"{name}: {Rr['verdict']}  thr {thr:.3f}  TRAIN n{tr_.get('n')} P{tr_.get('p_sig_gt_opp')} diff {tr_.get('diff_bp')} {tr_.get('diff_day_ci95')} | VAL {va_.get('diff_bp')} OOS {oo_.get('diff_bp')} ({time.time()-t0:.0f}s)")

    # ---- A: 상태 축 (R 지속 거래의 발동 봉 값, TRAIN 5분위) ----
    Fi = F.set_index("timestamp"); Rs = R.copy(); dir_sign = np.where(Rs["trade_long"], 1.0, -1.0)
    print(f"\n{'state axis':>18s} {'w':>5s} " + " ".join(f"{'Q'+str(i+1):>7s}" for i in range(5)) + f" {'Q5-Q1[CI]':>22s} {'drop-worst − R /day[CI]':>28s}")
    for ax, mode in STATE_AXES.items():
        x = Fi[ax].reindex(Rs["timestamp"]).to_numpy(float); val = x * dir_sign if mode == "aligned" else x
        okm = np.isfinite(val); edges = np.nanquantile(val[okm & (Rs.split == "TRAIN").to_numpy()], [0.2, 0.4, 0.6, 0.8]); qb = np.where(okm, np.searchsorted(edges, val, side="right"), -1)
        S = {"mode": mode, "edges": [round(float(e), 3) for e in edges], "coverage": round(float(okm.mean()), 3), "windows": {}}
        tr_means = [float(Rs.loc[(qb == i) & (Rs.split == "TRAIN"), "pnl_bp"].mean()) for i in range(5)]; worst = int(np.nanargmin(tr_means))
        for w in WINDOWS:
            mw = (Rs.split == w).to_numpy(); means = [round(float(Rs.loc[(qb == i) & mw, "pnl_bp"].mean()), 2) if ((qb == i) & mw).sum() >= 10 else None for i in range(5)]
            hi = (qb == 4) & mw; lo = (qb == 0) & mw; gap = round(float(Rs.loc[hi, "pnl_bp"].mean() - Rs.loc[lo, "pnl_bp"].mean()), 2)
            gci = gap_day_ci(Rs.loc[hi, "pnl_bp"].to_numpy(), Rs.loc[hi, "timestamp"], Rs.loc[lo, "pnl_bp"].to_numpy(), Rs.loc[lo, "timestamp"])
            Sw = {"quintile_mean_bp": means, "q5_minus_q1": gap, "gap_day_ci95": gci, "n": int(mw.sum())}
            if w != "TRAIN":
                keep = mw & (qb != worst) & okm; Fd, _ = dedupe(Rs[keep]); s, r = pf(Fd)
                Sw["drop_worst_quintile"] = {"worst_q": worst + 1, "portfolio": s, "vs_R_day_paired": day_paired(r, R_run[w]) if r is not None else None}
            S["windows"][w] = Sw
            dp = Sw.get("drop_worst_quintile", {}).get("vs_R_day_paired") or {}
            print(f"{ax:>18s} {w:>5s} " + " ".join(f"{(m if m is not None else float('nan')):7.2f}" for m in means) + f" {str(gap)+' '+str(gci):>22s} {str(dp.get('mean_bp_per_day'))+' '+str(dp.get('ci95')):>28s}")
        g = [S["windows"][w]["q5_minus_q1"] for w in WINDOWS]
        S["verdict"] = "WORKS" if (S["windows"]["TRAIN"]["gap_day_ci95"][0] > 0 and g[1] > 0 and g[2] > 0) else ("INVERSE" if (S["windows"]["TRAIN"]["gap_day_ci95"][1] < 0 and g[1] < 0 and g[2] < 0) else "NONE")
        rep["state_axes"][ax] = S
    rep["verdicts"] = {k: v["verdict"] for k, v in rep["axes"].items()}; rep["state_verdicts"] = {k: v["verdict"] for k, v in rep["state_axes"].items()}
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    print("\nbaseline every-bar net (long/short):", base); print("verdicts:", rep["verdicts"]); print("state verdicts:", rep["state_verdicts"])
    log(f"완료 -> {OUT/'report.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
