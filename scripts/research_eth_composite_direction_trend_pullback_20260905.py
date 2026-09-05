#!/usr/bin/env python3
"""ETH **복합 방향·추세·되돌림 알고리즘 v1** — 섀도우 알고리즘 + 증거신호 + 정보칩 취합 (2026-09-05).

사전등록: docs/experiments/eth_composite_direction_trend_pullback_prereg_20260905.md (결과 전 고정).

기준 R = 반전 8종 첫발동(GAP12) 지속 방향, open[i+1] 시장가, sim_exit(5.0/1.5/0.1) 200봉, 10bp, 동시 5.
세 축을 **R 대비 증분**으로만 판정한다 — VAL·OOS 두 창 모두 일별 짝비교 CI 하한 > 0 (§5.27 표준).

  C1 방향   다중신호 합의: 과거 W봉(0/3/12, 미래 금지) 안 같은 측면 첫발동 **서로 다른 신호 개수** m_same,
            반대 측면 m_opp. 필터(m_same≥2/≥3, m_net≥1) · 사이징(w ∝ m_same) · B2/레짐 정렬.
  C2 추세   정보칩 상태 조건화: 기술 상태 14축 + 경제 축(5.28 재료) TRAIN 5분위 갭 · 최악분위 제거 필터.
  C3 되돌림 지정가 되돌림 진입: limit = open[i+1] − sign·k·ATR (k 0.1~0.5), 유효 N봉(1~3).
            **청산 관리는 체결 봉의 다음 봉부터**(체결 이전 고가/저가 크레딧 금지 — 09-03 철회 원인).
            1분봉 재구성 변형으로 교차 확인. R_all → R_filled(역선택) → P_filled(가격) 분해.

HOLDOUT(≥2026-04-01) 미접촉. 연구/개발 점수 — 승격은 층 게이트 + 전진 섀도우.
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


V2 = _load("hev2_comp", "scripts/research_homer_entry_v2_20260904.py")
FC = _load("firecont_comp", "scripts/research_eth_evidence_fire_continuation_econ_20260904.py")
sim_exit, portfolio, day_boot, stats_of = V2.sim_exit, V2.portfolio, V2.day_boot, V2.stats_of
SIGNALS, FRAME = V2.SIGNALS, V2.OUT / "frame.parquet"
KL5 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
OUT = ROOT / "data/research/eth_composite_direction_trend_pullback_20260905"
CELL, FWD, COST, COST_MAKER, GAP, CAP = (5.0, 1.5, 0.1), 200, 10.0, 7.8, 12, 5
B_BOOT, B_NULL = 1000, 200
WINDOWS = ("TRAIN", "VAL", "OOS")
CONS_W = (0, 3, 12)
PULL_K = (0.10, 0.20, 0.30, 0.50)
PULL_N = (1, 2, 3)
rng = np.random.default_rng(20260905)


def log(m): print(f"[comp] {m}", flush=True)


# ----------------------------------------------------------------------------- 통계
def day_paired(pnl_a, ts_a, pnl_b, ts_b, cap=CAP, B=B_BOOT):
    """자본 대비 일손익(일별 합 / cap) 차이의 일 부트스트랩 — §5.27 표준. 거래 없는 날 0."""
    def s(p, t):
        return pd.Series(np.asarray(p, float), index=pd.DatetimeIndex(pd.to_datetime(np.asarray(t))).normalize()).groupby(level=0).sum() / cap
    A, Bs = s(pnl_a, ts_a), s(pnl_b, ts_b)
    days = A.index.union(Bs.index)
    d = (A.reindex(days).fillna(0.0) - Bs.reindex(days).fillna(0.0)).to_numpy()
    bo = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return {"diff_bp_day": round(float(d.mean()), 2), "ci95": [round(float(np.percentile(bo, 2.5)), 2), round(float(np.percentile(bo, 97.5)), 2)],
            "win_day_frac": round(float((d > 0).mean()), 3), "n_days": int(len(d))}


def pf(cand, tag=""):
    """순차 포트폴리오 + 일군집 CI + 일손익 통계. cand: entry_bar/exit_bar/pnl_bp/timestamp/p."""
    if cand is None or len(cand) == 0:
        return None
    r = portfolio(cand, CAP)
    if r is None:
        return None
    t = r["trades"]; p = t["pnl_bp"].to_numpy(); ts = t["timestamp"].to_numpy()
    lo, hi = day_boot(p, ts, B_BOOT, rng)
    s = pd.Series(p / CAP, index=pd.DatetimeIndex(pd.to_datetime(ts)).normalize()).groupby(level=0).sum()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"), fill_value=0.0)
    mu, sdv = float(s.mean()), float(s.std(ddof=1))
    o = stats_of(r)
    o.update({"day_ci95": [round(lo, 2), round(hi, 2)], "n_days": int(pd.DatetimeIndex(pd.to_datetime(ts)).normalize().nunique()),
              "per_day": round(len(p) / max(pd.DatetimeIndex(pd.to_datetime(ts)).normalize().nunique(), 1), 2),
              "daily_mean_bp": round(mu, 2), "daily_sharpe_ann": round(mu / sdv * np.sqrt(365), 2) if sdv > 0 else None,
              "pos_day_frac": round(float((s > 0).mean()), 3)})
    if tag:
        o["tag"] = tag
    return {"stats": o, "pnl": p, "ts": ts, "trades": t}


def cand_of(ts, entry_bar, exit_bar, pnl_bp, p=None):
    return pd.DataFrame({"timestamp": ts, "entry_bar": entry_bar, "exit_bar": exit_bar, "pnl_bp": pnl_bp,
                         "p": np.ones(len(pnl_bp)) if p is None else p})


# ----------------------------------------------------------------------------- 재료
def build():
    t0 = time.time()
    D = pd.read_parquet(FRAME)
    bar = D.drop_duplicates("pos").sort_values("pos").reset_index(drop=True)
    step = np.diff(bar["timestamp"].to_numpy()).astype("timedelta64[m]").astype(int)
    assert np.all(step == 5 * np.diff(bar["pos"].to_numpy())), "프레임 pos↔timestamp 비아핀"
    kl = pd.read_csv(KL5, usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    p_first = int(bar["pos"].iloc[0]); t_first = bar["timestamp"].iloc[0]
    k0 = int(np.searchsorted(kl["timestamp"].to_numpy(), np.datetime64(t_first))); assert kl["timestamp"].iloc[k0] == t_first
    need = int(bar["pos"].iloc[-1]) - p_first + FWD + max(PULL_N) + 3
    seg = kl.iloc[k0:k0 + need].reset_index(drop=True)
    assert np.all(np.diff(seg["timestamp"].to_numpy()).astype("timedelta64[m]").astype(int) == 5), "klines 구간 비연속"
    o, h, l, c = (seg[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    chk = bar.sample(2000, random_state=1)
    assert np.allclose(chk["entry"].to_numpy(), o[chk["pos"].to_numpy() - p_first + 1]), "entry 파리티 실패"

    # ---- 발동 + 다중도 (전부 과거만)
    F = FC.load_fires()
    Fp = F.loc[F["first_fire"]].drop_duplicates(["pos", "is_downside"]).copy()
    Fp = Fp.loc[Fp["pos"].isin(set(bar["pos"]))].sort_values("pos").reset_index(drop=True)
    # 모든 팔이 **같은 모집단**을 쓰도록, 되돌림 최대 지연까지 전방 200봉을 확보 못하는 꼬리 후보는 미리 제외
    room = (Fp["pos"].to_numpy() - p_first) + max(PULL_N) + 1 + FWD
    n_drop = int((room >= len(h)).sum())
    Fp = Fp.loc[room < len(h)].reset_index(drop=True)
    if n_drop:
        log(f"전방 봉 부족으로 꼬리 후보 {n_drop}건 제외 (모든 팔 동일 모집단 유지)")
    n_bars = int(bar["pos"].iloc[-1]) - p_first + 1
    fmat = np.zeros((len(SIGNALS), 2, n_bars), bool)                    # 신호 × 측면(0=top,1=bottom) × 봉
    FF = F.loc[F["first_fire"]]
    for si, s in enumerate(SIGNALS):
        g = FF.loc[FF["signal"] == s]
        for sd in (0, 1):
            q = g.loc[g["is_downside"] == sd, "pos"].to_numpy()
            q = q[(q >= p_first) & (q < p_first + n_bars)]
            fmat[si, sd, q - p_first] = True
    ind = {}                                                             # W봉 과거창 안 발동 여부
    for W in CONS_W:
        z = np.zeros_like(fmat)
        for si in range(len(SIGNALS)):
            for sd in (0, 1):
                z[si, sd] = pd.Series(fmat[si, sd].astype(np.int8)).rolling(W + 1, min_periods=1).max().to_numpy() > 0
        ind[W] = z
    pos = Fp["pos"].to_numpy(); sd = Fp["is_downside"].to_numpy().astype(int); bidx = pos - p_first
    cons = {}
    for W in CONS_W:
        same = ind[W][:, sd, bidx].sum(0)                                # (신호, 후보) -> 후보별 합
        opp = ind[W][:, 1 - sd, bidx].sum(0)
        cons[f"m_same_w{W}"] = same.astype(np.int16); cons[f"m_opp_w{W}"] = opp.astype(np.int16)
        cons[f"m_net_w{W}"] = (same - opp).astype(np.int16)

    # ---- 라벨 (F0 상속, 파리티 검증)
    atr = bar.set_index("pos")["atr"].reindex(pos).to_numpy(float)
    split = bar.set_index("pos")["split"].reindex(pos).to_numpy()
    ts = bar.set_index("pos")["timestamp"].reindex(pos).to_numpy()
    fade_sign = np.where(sd == 1, 1.0, -1.0); cont_sign = -fade_sign      # 지속: 바닥 발동->숏(-1), 천장->롱(+1)
    st0 = bidx + 1
    H = h[st0[:, None] + np.arange(FWD)]; L = l[st0[:, None] + np.arange(FWD)]; C = c[st0[:, None] + np.arange(FWD)]
    entry = o[st0]
    cont_ret, cont_ex = sim_exit(entry, atr, cont_sign, H, L, C, *CELL)
    fade_ret, _ = sim_exit(entry, atr, fade_sign, H, L, C, *CELL)
    cont_bp = cont_ret * 1e4 - COST; fade_bp = fade_ret * 1e4 - COST
    key = D.set_index(["pos", "is_downside"]).reindex(pd.MultiIndex.from_arrays([pos, sd], names=["pos", "is_downside"]))
    par_c = float(np.nanmax(np.abs(key["net_bp_flip"].to_numpy() - cont_bp)))
    par_f = float(np.nanmax(np.abs(key["net_bp"].to_numpy() - fade_bp)))
    log(f"라벨 파리티 |Δ|max 지속 {par_c:.2e} · 페이드 {par_f:.2e} bp")
    assert par_c < 1e-6 and par_f < 1e-6, "F0 프레임 파리티 실패 — 중단"

    # ---- 정보칩 상태
    S = {}
    for col in ("adx14", "bb_width_pctile", "atr_percentile_864", "vol_z", "range_width_pct", "atr_pct"):
        S[col] = ("raw", key[col].to_numpy(float))
    S["di_spread"] = ("aligned", (key["pdi"].to_numpy(float) - key["ndi"].to_numpy(float)))
    S["rsi_c"] = ("aligned", key["rsi"].to_numpy(float) - 50.0)
    S["bb_pctb_c"] = ("aligned", key["bb_pctb"].to_numpy(float) - 0.5)
    for col in ("vwap_dev_z", "cvd_roll_roc_48", "delta_z", "flow_aligned_delta_z", "ret3_z"):
        S[col] = ("aligned", key[col].to_numpy(float))
    # ER(24) 효율비 — 추세 직진성 (인과)
    cl = pd.Series(c)
    er = (cl.diff(24).abs() / cl.diff().abs().rolling(24).sum()).to_numpy()
    S["er24"] = ("raw", er[bidx])
    ax_meta = {}
    try:
        EA = _load("econax_comp", "scripts/research_eth_econ_axis_continuation_screen_20260904.py")
        AX, ax_meta = EA.build_axes()
        AX = AX.set_index("timestamp")
        sel = AX.reindex(pd.DatetimeIndex(ts))
        ALIGNED = {"taker_flow", "basis_change", "btc_shock", "market_beta_move", "eth_idio_move", "depth_imbalance",
                   "retail_shift", "toptrader_shift", "funding_z", "retail_shift_lag1", "toptrader_shift_lag1"}
        for col in ("taker_flow", "activity", "oi_chg6", "basis_change", "btc_shock", "market_beta_move",
                    "eth_idio_move", "depth_imbalance", "retail_shift", "retail_shift_lag1", "toptrader_shift", "funding_z"):
            if col in sel.columns:
                S["ax_" + col] = ("aligned" if col in ALIGNED else "raw", sel[col].to_numpy(float))
        log(f"경제 축 병합 {sum(1 for k in S if k.startswith('ax_'))}개")
    except Exception as e:                                                # 경제 축은 선택적(외부 데이터 의존)
        log(f"⚠️경제 축 병합 실패(기술 상태만 사용): {type(e).__name__}: {e}")

    reg = np.where(key["reg_eth_bull"].to_numpy() > 0.5, "bull", np.where(key["reg_eth_bear"].to_numpy() > 0.5, "bear", "chop"))
    log(f"첫발동 {len(Fp):,} · TRAIN/VAL/OOS {[int((split == w).sum()) for w in WINDOWS]} · "
        f"바닥 {int((sd == 1).sum()):,}/천장 {int((sd == 0).sum()):,} · 다중도 m_same_w0 평균 {cons['m_same_w0'].mean():.2f} ({time.time()-t0:.0f}s)")
    return dict(D=D, bar=bar, seg=seg, o=o, h=h, l=l, c=c, p_first=p_first, pos=pos, sd=sd, bidx=bidx, atr=atr,
                split=split, ts=ts, cont_sign=cont_sign, fade_sign=fade_sign, entry=entry, cont_bp=cont_bp,
                fade_bp=fade_bp, cont_ex=cont_ex, cons=cons, S=S, reg=reg, Fp=Fp, ax_meta=ax_meta,
                parity={"cont": par_c, "fade": par_f})


def gap_day_ci(x_hi, t_hi, x_lo, t_lo, B=B_BOOT):
    """서로 다른 두 집합의 평균 차이 — 일 군집 부트스트랩(같은 날들을 함께 재표집)."""
    dh = pd.DatetimeIndex(pd.to_datetime(t_hi)).normalize().to_numpy(); dl = pd.DatetimeIndex(pd.to_datetime(t_lo)).normalize().to_numpy()
    days = np.union1d(np.unique(dh), np.unique(dl))
    ih = {d: np.flatnonzero(dh == d) for d in days}; il = {d: np.flatnonzero(dl == d) for d in days}
    out = []
    for _ in range(B):
        ds = days[rng.integers(0, len(days), len(days))]
        a = np.concatenate([ih[d] for d in ds]); b = np.concatenate([il[d] for d in ds])
        if len(a) and len(b):
            out.append(float(x_hi[a].mean() - x_lo[b].mean()))
    if not out:
        return [None, None]
    return [round(float(np.percentile(out, 2.5)), 2), round(float(np.percentile(out, 97.5)), 2)]


def side_matched_null(D, w, n_long, n_short, obs, B=B_NULL):
    """측면비율 매칭 무작위 진입 귀무(지속 방향과 같은 측면 구성)."""
    Dw = D.loc[D["split"] == w]
    pool_l = Dw.loc[Dw["is_downside"] == 1]; pool_s = Dw.loc[Dw["is_downside"] == 0]     # 지속 롱 = 천장(is_downside 0 행의 flip)…
    vals = []
    for _ in range(B):
        a = pool_l.iloc[rng.choice(len(pool_l), size=min(n_long, len(pool_l)), replace=False)]
        b = pool_s.iloc[rng.choice(len(pool_s), size=min(n_short, len(pool_s)), replace=False)]
        x = pd.concat([a, b])
        r = portfolio(cand_of(x["timestamp"].to_numpy(), x["pos"].to_numpy() + 1,
                              x["pos"].to_numpy() + 1 + x["exit_off"].to_numpy(), x["net_bp"].to_numpy()), CAP)
        vals.append(r["exp_bp"] if r else np.nan)
    v = np.asarray(vals, float)
    return {"mean_bp": round(float(np.nanmean(v)), 2), "p95_bp": round(float(np.nanpercentile(v, 95)), 2),
            "percentile_of_obs": round(float((v < obs).mean() * 100), 1)}


# ----------------------------------------------------------------------------- C1 방향: 다중신호 합의
def axis_C1(B):
    pos, sd, split, ts, cont_bp, cont_ex, cons = B["pos"], B["sd"], B["split"], B["ts"], B["cont_bp"], B["cont_ex"], B["cons"]
    reg = B["reg"]; S = B["S"]
    out = {"buckets": {}, "arms": {}}
    # B2(retail_shift) 정렬: 축값 × 지속방향 부호 > 0 이면 "정렬"
    b2 = S.get("ax_retail_shift_lag1", S.get("ax_retail_shift"))
    b2_al = None
    if b2 is not None:
        raw = b2[1]
        b2_al = np.where(np.isfinite(raw), (-raw) * B["cont_sign"], np.nan)   # retail_shift 방향 = −부호 → 지속과 정렬 = (−x)·cont_sign
    reg_ok = ~(((reg == "bull") & (B["cont_sign"] < 0)) | ((reg == "bear") & (B["cont_sign"] > 0)))
    for w in WINDOWS:
        m = split == w
        base = pf(cand_of(ts[m], pos[m] + 1, pos[m] + 1 + cont_ex[m], cont_bp[m]), "R")
        out.setdefault("baseline", {})[w] = base["stats"]
        # 버킷: 다중도별 행 평균/포트폴리오
        bk = {}
        for W in CONS_W:
            ms = cons[f"m_same_w{W}"]; d = {}
            for v in range(1, 6):
                sel = m & (ms == v) if v < 5 else m & (ms >= 5)
                if sel.sum() < 30:
                    continue
                r = pf(cand_of(ts[sel], pos[sel] + 1, pos[sel] + 1 + cont_ex[sel], cont_bp[sel]))
                d[f"m>={v}" if v == 5 else f"m={v}"] = {"n": int(sel.sum()), "row_bp": round(float(cont_bp[sel].mean()), 2),
                                                        "pf_exp_bp": r["stats"]["exp_bp"] if r else None, "pf_n": r["stats"]["n"] if r else 0}
            bk[f"W{W}"] = d
        out["buckets"][w] = bk
        # 팔
        arms = {}
        for W in CONS_W:
            ms = cons[f"m_same_w{W}"]; mn = cons[f"m_net_w{W}"]
            arms[f"filter_m_same>=2_W{W}"] = m & (ms >= 2)
            arms[f"filter_m_same>=3_W{W}"] = m & (ms >= 3)
            arms[f"filter_m_net>=1_W{W}"] = m & (mn >= 1)
        arms["filter_regime_ok"] = m & reg_ok
        if b2_al is not None:
            arms["filter_b2_aligned"] = m & (np.nan_to_num(b2_al, nan=-1.0) > 0)
            arms["filter_b2_not_opposed"] = m & ~(np.nan_to_num(b2_al, nan=0.0) < -1.0)
        res = {}
        for nm, sel in arms.items():
            if sel.sum() < 50:
                res[nm] = {"n_rows": int(sel.sum()), "skip": "n<50"}; continue
            r = pf(cand_of(ts[sel], pos[sel] + 1, pos[sel] + 1 + cont_ex[sel], cont_bp[sel]))
            if r is None:
                continue
            res[nm] = {"n_rows": int(sel.sum()), **{k: r["stats"][k] for k in ("n", "exp_bp", "win_rate", "day_ci95", "per_day", "daily_mean_bp", "daily_sharpe_ann")},
                       "vs_R": day_paired(r["pnl"], r["ts"], base["pnl"], base["ts"])}
        # 사이징 w ∝ m_same (창 내 평균 1 정규화) — 슬롯 점유는 동일
        for W in CONS_W:
            ms = cons[f"m_same_w{W}"][m].astype(float)
            wgt = ms / ms.mean()
            r = pf(cand_of(ts[m], pos[m] + 1, pos[m] + 1 + cont_ex[m], cont_bp[m] * wgt))
            res[f"size_prop_m_same_W{W}"] = {**{k: r["stats"][k] for k in ("n", "exp_bp", "day_ci95", "daily_mean_bp", "daily_sharpe_ann")},
                                             "vs_R": day_paired(r["pnl"], r["ts"], base["pnl"], base["ts"])}
        out["arms"][w] = res
    return out


# ----------------------------------------------------------------------------- C2 추세: 정보칩 상태
def axis_C2(B):
    pos, sd, split, ts, cont_bp, cont_ex, S = B["pos"], B["sd"], B["split"], B["ts"], B["cont_bp"], B["cont_ex"], B["S"]
    cs = B["cont_sign"]; tr = split == "TRAIN"
    out = {"axes": {}}
    base = {w: pf(cand_of(ts[split == w], pos[split == w] + 1, pos[split == w] + 1 + cont_ex[split == w], cont_bp[split == w])) for w in WINDOWS}
    for name, (kind, raw) in S.items():
        x = raw * cs if kind == "aligned" else raw.astype(float)
        fin = np.isfinite(x)
        if (fin & tr).sum() < 500:
            out["axes"][name] = {"skip": "TRAIN 유한값 < 500", "coverage": round(float(fin.mean()), 3)}
            continue
        qs = np.quantile(x[fin & tr], [0.2, 0.4, 0.6, 0.8])
        qi = np.where(fin, np.digitize(x, qs), -1)                       # 0..4, 결측 -1
        rec = {"kind": kind, "coverage": round(float(fin.mean()), 3), "train_quintile_edges": [round(float(v), 4) for v in qs], "windows": {}}
        worst = None
        for w in WINDOWS:
            m = (split == w) & fin
            byq = {}
            for q in range(5):
                sel = m & (qi == q)
                if sel.sum() < 30:
                    continue
                byq[f"Q{q+1}"] = {"n": int(sel.sum()), "row_bp": round(float(cont_bp[sel].mean()), 2)}
            hi = m & (qi == 4); lo = m & (qi == 0)
            gap = float(cont_bp[hi].mean() - cont_bp[lo].mean()) if hi.sum() > 10 and lo.sum() > 10 else None
            rec["windows"][w] = {"by_quintile": byq, "gap_Q5_Q1_bp": round(gap, 2) if gap is not None else None,
                                 "gap_day_ci95": gap_day_ci(cont_bp[hi], ts[hi], cont_bp[lo], ts[lo]) if gap is not None else None}
            if w == "TRAIN" and byq:
                worst = int(min(byq, key=lambda k: byq[k]["row_bp"])[1]) - 1
        # TRAIN 최악 분위 제거 필터 → R 대비 짝비교
        if worst is not None:
            rec["train_worst_quintile"] = f"Q{worst+1}"; rec["filter"] = {}
            for w in WINDOWS:
                m = (split == w) & ~((qi == worst) & fin)
                if m.sum() < 200:
                    continue
                r = pf(cand_of(ts[m], pos[m] + 1, pos[m] + 1 + cont_ex[m], cont_bp[m]))
                if r is None:
                    continue
                rec["filter"][w] = {"n": r["stats"]["n"], "exp_bp": r["stats"]["exp_bp"], "day_ci95": r["stats"]["day_ci95"],
                                    "vs_R": day_paired(r["pnl"], r["ts"], base[w]["pnl"], base[w]["ts"])}
        out["axes"][name] = rec
    return out


# ----------------------------------------------------------------------------- C3 되돌림: 지정가 진입
def pullback_fill(B, k, N):
    """되돌림 지정가 체결 시뮬. 반환: filled(bool), fill_n(1..N), fill_px."""
    bidx, atr, cs, o, h, l = B["bidx"], B["atr"], B["cont_sign"], B["o"], B["h"], B["l"]
    ref = o[bidx + 1]; lim = ref - cs * k * atr
    n = len(bidx); filled = np.zeros(n, bool); fill_n = np.zeros(n, int); fill_px = np.full(n, np.nan)
    for step in range(1, N + 1):
        b = bidx + step
        hit = np.where(cs > 0, l[b] <= lim, h[b] >= lim) & ~filled
        px = np.where(cs > 0, np.minimum(lim, o[b]), np.maximum(lim, o[b]))     # 갭 통과 시 시가 체결(유리)
        fill_px = np.where(hit, px, fill_px); fill_n = np.where(hit, step, fill_n); filled = filled | hit
    return filled, fill_n, fill_px


def axis_C3(B):
    pos, split, ts, cont_bp, cont_ex = B["pos"], B["split"], B["ts"], B["cont_bp"], B["cont_ex"]
    bidx, atr, cs, h, l, c = B["bidx"], B["atr"], B["cont_sign"], B["h"], B["l"], B["c"]
    out = {"cells": {}}
    base = {w: pf(cand_of(ts[split == w], pos[split == w] + 1, pos[split == w] + 1 + cont_ex[split == w], cont_bp[split == w])) for w in WINDOWS}
    for k in PULL_K:
        for N in PULL_N:
            filled, fill_n, fill_px = pullback_fill(B, k, N)
            stp = bidx + fill_n + 1                                             # 청산 관리는 체결 봉 **다음** 봉부터
            idx = stp[:, None] + np.arange(FWD)
            Hn, Ln, Cn = h[idx], l[idx], c[idx]
            ret, ex = sim_exit(np.nan_to_num(fill_px, nan=1.0), atr, cs, Hn, Ln, Cn, *CELL)
            rec = {"fill_rate_all": round(float(filled.mean()), 3), "windows": {}}
            for w in WINDOWS:
                mw = split == w; mf = mw & filled
                if mf.sum() < 50:
                    rec["windows"][w] = {"n_filled": int(mf.sum()), "skip": "n<50"}; continue
                r_all = base[w]
                r_fil = pf(cand_of(ts[mf], pos[mf] + 1, pos[mf] + 1 + cont_ex[mf], cont_bp[mf]))          # 같은 건, 시장가 (역선택)
                d = {}
                for cost, tag in ((COST, "cost10"), (COST_MAKER, "cost7.8")):
                    pnl = ret[mf] * 1e4 - cost
                    r_p = pf(cand_of(ts[mf], pos[mf] + fill_n[mf], pos[mf] + fill_n[mf] + 1 + ex[mf], pnl))
                    if r_p is None:
                        continue
                    d[tag] = {"n": r_p["stats"]["n"], "exp_bp": r_p["stats"]["exp_bp"], "win_rate": r_p["stats"]["win_rate"],
                              "day_ci95": r_p["stats"]["day_ci95"], "per_day": r_p["stats"]["per_day"],
                              "daily_mean_bp": r_p["stats"]["daily_mean_bp"], "daily_sharpe_ann": r_p["stats"]["daily_sharpe_ann"],
                              "vs_R_all": day_paired(r_p["pnl"], r_p["ts"], r_all["pnl"], r_all["ts"]),
                              "vs_R_filled": day_paired(r_p["pnl"], r_p["ts"], r_fil["pnl"], r_fil["ts"]) if r_fil else None}
                rec["windows"][w] = {"n_filled": int(mf.sum()), "fill_rate": round(float(filled[mw].mean()), 3),
                                     "mean_fill_delay_bars": round(float(fill_n[mf].mean()), 2),
                                     "mean_price_edge_bp": round(float((cs[mf] * (B["entry"][mf] - fill_px[mf]) / B["entry"][mf]).mean() * 1e4), 2),
                                     "R_filled_exp_bp": r_fil["stats"]["exp_bp"] if r_fil else None,
                                     "R_all_exp_bp": r_all["stats"]["exp_bp"],
                                     "adverse_selection_vs_R_all": day_paired(r_fil["pnl"], r_fil["ts"], r_all["pnl"], r_all["ts"]) if r_fil else None,
                                     "pullback": d}
            out["cells"][f"k{k}_N{N}"] = rec
    return out


# ----------------------------------------------------------------------------- 판정 · 보고
def verdicts(rep):
    """사전등록 판정: 어떤 팔도 VAL·OOS 두 창 모두 짝비교 CI 하한 > 0 이어야 한다."""
    v = {"passes": [], "rule": "VAL·OOS 두 창 모두 vs_R 일별 짝비교 CI 하한 > 0"}
    def chk(name, dv, do):
        if dv and do and dv["ci95"][0] > 0 and do["ci95"][0] > 0:
            v["passes"].append({"arm": name, "VAL": dv, "OOS": do})
    for nm in rep["C1"]["arms"].get("VAL", {}):
        a, b = rep["C1"]["arms"]["VAL"].get(nm, {}), rep["C1"]["arms"]["OOS"].get(nm, {})
        chk("C1:" + nm, a.get("vs_R"), b.get("vs_R"))
    for nm, rec in rep["C2"]["axes"].items():
        f = rec.get("filter", {})
        chk("C2:" + nm + "(최악분위제거)", f.get("VAL", {}).get("vs_R"), f.get("OOS", {}).get("vs_R"))
    for cell, rec in rep["C3"]["cells"].items():
        for tag in ("cost10", "cost7.8"):
            a = rec["windows"].get("VAL", {}).get("pullback", {}).get(tag, {})
            b = rec["windows"].get("OOS", {}).get("pullback", {}).get(tag, {})
            chk(f"C3:{cell}:{tag}", a.get("vs_R_all"), b.get("vs_R_all"))
    v["n_pass"] = len(v["passes"])
    v["verdict"] = "R 단독 유지" if not v["passes"] else "후보 존재 — 층 게이트·전진 섀도우 필요"
    return v


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    B = build()
    rep = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "prereg": "docs/experiments/eth_composite_direction_trend_pullback_prereg_20260905.md",
           "cell": CELL, "forward_bars": FWD, "cost_bp": COST, "cost_maker_bp": COST_MAKER, "gap": GAP, "max_concurrent": CAP,
           "n_first_fires": int(len(B["pos"])), "label_parity_max_abs_bp": B["parity"], "holdout_touched": False,
           "econ_axis_meta": B["ax_meta"],
           "n_by_window": {w: int((B["split"] == w).sum()) for w in WINDOWS}}
    log("C1 방향(다중신호 합의) …")
    rep["C1"] = axis_C1(B)
    log("C2 추세(정보칩 상태) …")
    rep["C2"] = axis_C2(B)
    log("C3 되돌림(지정가 진입) …")
    rep["C3"] = axis_C3(B)
    # R 기준 무작위 귀무
    rep["baseline_null"] = {}
    for w in WINDOWS:
        m = B["split"] == w
        obs = rep["C1"]["baseline"][w]["exp_bp"]
        n_long = int((B["cont_sign"][m] > 0).sum()); n_short = int((B["cont_sign"][m] < 0).sum())
        rep["baseline_null"][w] = side_matched_null(B["D"], w, n_long, n_short, obs)
    rep["verdict"] = verdicts(rep)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"완료 {time.time()-t0:.0f}s → {OUT/'report.json'}")
    for w in WINDOWS:
        b = rep["C1"]["baseline"][w]
        log(f"  R {w}: n={b['n']} exp={b['exp_bp']}bp CI={b['day_ci95']} /일={b['per_day']} 샤프={b['daily_sharpe_ann']} 귀무백분위={rep['baseline_null'][w]['percentile_of_obs']}")
    log(f"  판정: {rep['verdict']['verdict']} (통과 팔 {rep['verdict']['n_pass']}개)")
    for p in rep["verdict"]["passes"]:
        log(f"    ✔ {p['arm']}: VAL {p['VAL']['diff_bp_day']} {p['VAL']['ci95']} · OOS {p['OOS']['diff_bp_day']} {p['OOS']['ci95']}")


if __name__ == "__main__":
    main()
