#!/usr/bin/env python3
"""ETH 유동성 **스윕(sweep) vs 런(run)** 분류기 (2026-09-26).

사용자 *"liquidity sweep 과 liquidity run 신호를 구분할 수 있는 최고 정확도를 가진 모델이나
로직을 만들어줘"*.

## 질문을 이렇게 고정했다
ICT 용어로 가격이 직전 스윙 고/저점(유동성)을 **찌른 뒤**
  · SWEEP(스윕) = 스톱을 쓸어담고 **되돌아온다**(반전)
  · RUN(런)     = 스톱을 연료 삼아 **그 방향으로 계속 간다**(지속)
대시보드 증거신호 `liquidity_sweep` 은 "찌른 봉이 레벨 안쪽으로 마감"이라는 **규칙 하나로 이미
스윕이라고 판정**한다. 그 규칙 자체가 이 질문의 베이스라인(R1)이다.

  트리거   봉 i 의 고가 > 직전 48봉 스윙고가(천장) / 저가 < 직전 48봉 스윙저가(바닥).
           레벨·룩백은 라이브 `compute_signals` 와 같고, `찌름 & 안쪽마감` 이 라이브
           `top/bottom_liquidity_sweep` 과 **봉 단위로 동치임을 assert** 한다(식 두 벌 금지).
           런 구간에서는 매 봉이 새로 찌르므로 같은 측면 찌름이 직전 COOLDOWN 봉 안에 없을 때만
           사건으로 센다(과거만 보는 인과 dedup — 미래를 보는 cluster_dedup 아님).
  결정시점 봉 i 마감 후. 트리거는 봉 i *안에서* 일어나므로 i 마감 전에는 알 수 없다(OBS=0 금지).
  라벨     기준가 close[i], 폭 K_ATR*ATR[i] 의 **대칭** 배리어를 봉 i+1 부터 H 봉 탐색(first_touch).
           레벨 안쪽 방향이 먼저 → SWEEP(y=1), 찌른 방향이 먼저 → RUN(y=0).
           같은 봉 양쪽 터치(순서 모름)·H 봉 안 미결은 **제외**하고 건수를 보고한다.
           대칭이라 무작위보행이면 기저가 정확히 50% 다 — 정확도를 bp 로 환산하지 않고 라벨을 직접 센다.
  피쳐     g_* = 트리거 봉 i 자신(마감됨), f_* = 봉 i-1 까지 맥락. 라벨은 i+1 부터라 경계 계약 준수
           (피쳐 창 끝 i < 라벨 시작 i+1). f_* 는 ATR[i-1] 로 정규화해 트리거 봉 정보가 새지 않게 했다.

## 모델 사다리 (선택은 VAL 로그손실, OOS·HOLDOUT 은 보고만)
  R0 다수클래스 · R1 대시보드 규칙(안쪽마감=SWEEP) · R2 깊이2 트리(2~3피쳐 규칙)
  LR 로지스틱 · HGB 부스팅(g+f) · HGB_ctx(f 만 = 트리거 봉 없이 맥락만)

## 누수 방어 (CLAUDE.md 사건 라벨 경계 계약 · 탐지 신호)
  · selftest: 라벨이 봉 i 의 고/저/시가/거래량에 **불변**, 피쳐가 i+1 이후 봉에 **불변**
  · 절단 인과성: 사건 표본마다 데이터를 봉 i 에서 잘라 피쳐를 다시 만들어 전체판과 대조
  · 단일피쳐 AUC≥0.95 / 모델 VAL AUC≥0.99 → FAIL
  · 모델이 R2 보다 OOS 5pp 이상 앞서는데 최상위 순열중요도 피쳐의 단변량 분위 효과가 평평 → FLAG
  · ⭐`--synthetic-null`: 가격을 1분 무작위보행으로 바꿔 **같은 파이프라인**을 돌린다.
    예측가능성이 0 인 데이터라 어떤 모델이든 50% 를 유의하게 넘으면 그것이 곧 누수다.

## 이것은 분류 연구 점수다
정확도는 거래 성과가 아니다(09-14 에 이 축의 bp 는 1,741일에서 음수로 종결). 승격·라이브 근거로
쓰려면 진입 파이프라인으로 옮긴 뒤 `gate_eth_entry_layers_20260903.py` 와 fresh-forward 를 따로 통과해야 한다.

사용:
  python scripts/research_eth_liquidity_sweep_vs_run_classifier_20260926.py --selftest
  python scripts/research_eth_liquidity_sweep_vs_run_classifier_20260926.py
  python scripts/research_eth_liquidity_sweep_vs_run_classifier_20260926.py --synthetic-null
출력: tmp/eth_liquidity_sweep_vs_run_20260926[_null]/{events.parquet, metrics.csv, report.json}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import research_eth_liquidity_sweep_block_independence_20260914 as P0  # noqa: E402
import live_evidence_signal_dashboard_20260823 as EV  # noqa: E402

OUT = ROOT / "tmp/eth_liquidity_sweep_vs_run_20260926"
LOOKBACK = EV.SWEEP_LOOKBACK   # 48 — 라이브와 같은 상수를 그대로 쓴다
HTF = 288                      # 24h 상위 레벨
COOLDOWN = 12                  # 같은 측면 찌름이 직전 1시간 안에 없을 때만 새 사건
K_ATR = 1.5                    # 08-29 V반등 라벨 최종판의 이동량 기준(1.5 ATR)과 같게
H = 24                         # 2시간
SPLITS = {"VAL": "2025-09-01", "OOS": "2026-01-01", "HOLDOUT": "2026-04-01"}
WF_START = "2023-01-01"            # 월별 확장창 walk-forward 채점 시작
BAR = pd.Timedelta("5min")
SEED = 20260926


def first_touch(hi, lo, start, ref, width, horizon):
    """봉 start..start+horizon-1 에서 ref±width 중 먼저 닿는 쪽.
    +1 위 먼저 · -1 아래 먼저 · 2 같은 봉 양쪽(순서 모름) · 0 미결 · -9 미래 봉 부족."""
    n = len(hi)
    out = np.zeros(len(start), np.int8)
    up, dn = ref + width, ref - width
    for k in range(horizon):
        j = start + k
        live = (j < n) & (out == 0)
        jj = np.minimum(j, n - 1)
        u = live & (hi[jj] >= up)
        d = live & (lo[jj] <= dn)
        out[u & ~d] = 1
        out[d & ~u] = -1
        out[u & d] = 2
    out[(out == 0) & (start + horizon > n)] = -9
    return out


def _bars_since(flag: np.ndarray, cap: int) -> np.ndarray:
    """봉 i **이전**(i 제외) 마지막 True 로부터 몇 봉 지났나. 없으면 cap."""
    n = len(flag)
    last = np.maximum.accumulate(np.where(flag, np.arange(n), -10**9))
    prev = np.concatenate([[-10**9], last[:-1]])
    return np.minimum(np.arange(n) - prev, cap)


def build(kl: pd.DataFrame, btc: pd.DataFrame | None) -> pd.DataFrame:
    """사건 행 = 첫 찌름 봉 i. 피쳐·라벨을 함께 만든다."""
    from research_eth_liquidity_sweep_mechanism_filters_20260914 import rolling_argextreme

    sig = EV.compute_signals(kl, btc_df=btc, funding_df=None)
    ts = sig["timestamp"]
    o, h, l, c = (sig[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    v, tb = sig["volume"].to_numpy(float), sig["taker_buy_base"].to_numpy(float)
    atrp = sig["atr_pct"].to_numpy(float)
    a = atrp * c
    n = len(sig)
    S = pd.Series
    swH = S(h).rolling(LOOKBACK, min_periods=LOOKBACK).max().shift(1).to_numpy()
    swL = S(l).rolling(LOOKBACK, min_periods=LOOKBACK).min().shift(1).to_numpy()
    htfH = S(h).rolling(HTF, min_periods=HTF).max().shift(1).to_numpy()
    htfL = S(l).rolling(HTF, min_periods=HTF).min().shift(1).to_numpy()
    posH, posL = rolling_argextreme(h, LOOKBACK, False), rolling_argextreme(l, LOOKBACK, True)
    pierce = {1: h > swH, -1: l < swL}
    # 식 두 벌 금지 — 찌름 & 안쪽마감 이 라이브 신호와 봉 단위로 같아야 한다
    for s, col, lvl in ((1, "top_liquidity_sweep", swH), (-1, "bottom_liquidity_sweep", swL)):
        mine = pierce[s] & ((c < lvl) if s == 1 else (c > lvl))
        live = sig[col].fillna(False).to_numpy(bool)
        assert np.array_equal(mine, live), f"{col}: 라이브 정의와 {int((mine != live).sum())}봉 불일치"

    ema = S(c).ewm(span=HTF, adjust=False).mean().to_numpy()
    atr_ratio = atrp / S(atrp).rolling(HTF, min_periods=HTF).mean().to_numpy()
    vol_rel = np.log(np.maximum(v, 1e-12) / S(v).rolling(LOOKBACK, min_periods=LOOKBACK).mean().shift(1).to_numpy())
    dlt = 2.0 * tb - v
    dfrac = dlt / np.maximum(v, 1e-12)
    dfrac3 = S(dlt).rolling(3, min_periods=3).sum().to_numpy() / np.maximum(S(v).rolling(3, min_periods=3).sum().to_numpy(), 1e-12)
    hour = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy() / 60.0
    if btc is not None:
        bm = btc.drop_duplicates("timestamp").set_index("timestamp").reindex(ts)
        bh, bl, bc = (bm[k].to_numpy(float) for k in ("high", "low", "close"))
        b_pierce = {1: bh > S(bh).rolling(LOOKBACK, min_periods=LOOKBACK).max().shift(1).to_numpy(),
                    -1: bl < S(bl).rolling(LOOKBACK, min_periods=LOOKBACK).min().shift(1).to_numpy()}

    rows = []
    lo_i = HTF + LOOKBACK + 2
    for s in (1, -1):
        top = s == 1
        ev = pierce[s] & (_bars_since(pierce[s], 10**6) > COOLDOWN)
        idx = np.flatnonzero(ev)
        idx = idx[idx >= lo_i]
        p = idx - 1                                   # 맥락 피쳐의 마지막 봉
        L = (swH if top else swL)[idx]
        ai, ap = a[idx], a[p]
        ext = (h if top else l)[idx]
        rng = np.maximum(h[idx] - l[idx], 1e-12)
        win = np.lib.stride_tricks.sliding_window_view(h if top else l, LOOKBACK)[idx - LOOKBACK]
        opp_since = _bars_since(pierce[-s], HTF)
        cnt = S(pierce[s].astype(float)).rolling(HTF, min_periods=1).sum().shift(1).to_numpy()
        d = {
            "timestamp": ts.to_numpy()[idx], "i": idx, "side": "top" if top else "bottom",
            "level": L, "ref": c[idx], "atr_abs": ai,
            # g_* : 트리거 봉 i 자신 (마감 확정)
            "g_depth": s * (ext - L) / ai,
            "g_close_lvl": s * (c[idx] - L) / ai,
            "g_closed_back": (s * (c[idx] - L) < 0).astype(float),     # = 대시보드 liquidity_sweep
            "g_rej_wick": ((h[idx] - np.maximum(o[idx], c[idx])) if top else (np.minimum(o[idx], c[idx]) - l[idx])) / rng,
            "g_body": s * (c[idx] - o[idx]) / ai,
            "g_range": (h[idx] - l[idx]) / ai,
            "g_vol_rel": vol_rel[idx],
            "g_delta": s * dfrac[idx],
            "g_delta3": s * dfrac3[idx],
            # f_* : 봉 i-1 까지 맥락 (ATR[i-1] 로 정규화)
            "f_pre_dist": s * (L - c[p]) / ap,
            "f_mom3": s * (c[p] - c[p - 3]) / ap,
            "f_mom12": s * (c[p] - c[p - 12]) / ap,
            "f_mom48": s * (c[p] - c[p - 48]) / ap,
            "f_mom288": s * (c[p] - c[p - HTF]) / ap,
            "f_ema_dist": s * (c[p] - ema[p]) / ap,
            "f_range48": (swH[idx] - swL[idx]) / ap,
            "f_level_age": (idx - (posH if top else posL)[idx]).astype(float),
            "f_touch": (np.abs(win - L[:, None]) <= 0.15 * ap[:, None]).sum(axis=1).astype(float),
            "f_htf_gap": s * ((htfH if top else htfL)[idx] - L) / ap,
            "f_atr_pct": atrp[p],
            "f_atr_ratio": atr_ratio[p],
            "f_prior_pierces": cnt[idx],
            "f_since_opp": opp_since[idx].astype(float),
            "f_hour_sin": np.sin(2 * np.pi * hour[idx] / 24), "f_hour_cos": np.cos(2 * np.pi * hour[idx] / 24),
            "f_side": np.full(len(idx), float(top)),
        }
        if btc is not None:
            d["g_btc_pierce"] = b_pierce[s][idx].astype(float)
            d["f_btc_rel"] = s * (np.log(c[p] / c[p - 12]) - np.log(bc[p] / bc[p - 12])) / atrp[p]
        ft = first_touch(h, l, idx + 1, c[idx], K_ATR * ai, H)
        d["touch"] = ft
        d["y"] = np.where(ft == -s, 1.0, np.where(ft == s, 0.0, np.nan))   # 안쪽 먼저=SWEEP
        rows.append(pd.DataFrame(d))
    E = pd.concat(rows, ignore_index=True).sort_values(["timestamp", "side"]).reset_index(drop=True)
    E["timestamp"] = pd.to_datetime(E["timestamp"])
    return E


def feat_cols(E):
    return [k for k in E.columns if k.startswith(("f_", "g_"))]


def split_of(ts) -> np.ndarray:
    ts = pd.DatetimeIndex(ts)
    v, o, hd = (pd.Timestamp(SPLITS[k]) for k in ("VAL", "OOS", "HOLDOUT"))
    return np.where(ts < v, "TRAIN", np.where(ts < o, "VAL", np.where(ts < hd, "OOS", "HOLDOUT")))


def truncation_check(kl, btc, E, n_sample=30) -> dict:
    """사건 표본마다 kl 을 봉 i 에서 잘라(미래 없음) 피쳐를 다시 만들어 전체판과 대조한다."""
    rng = np.random.default_rng(SEED)
    cols = feat_cols(E)
    pick = E.iloc[rng.choice(len(E), min(n_sample, len(E)), replace=False)]
    bad = 0; worst = 0.0
    for r in pick.itertuples():
        t = r.timestamp
        k0 = kl[kl["timestamp"] <= t].tail(6000)
        b0 = btc[btc["timestamp"] <= t] if btc is not None else None
        T = build(k0.reset_index(drop=True), b0)
        T = T[(T["timestamp"] == t) & (T["side"] == r.side)]
        if len(T) != 1:
            bad += 1; continue                       # 잘랐더니 사건이 사라지거나 생겼다 = 유령 발동
        x_full = E.loc[r.Index, cols].to_numpy(float); x_cut = T[cols].to_numpy(float)[0]
        diff = np.nanmax(np.abs(x_full - x_cut) / np.maximum(np.abs(x_full), 1.0))
        worst = max(worst, float(diff))
        bad += int(diff > 1e-6 or not np.array_equal(np.isnan(x_full), np.isnan(x_cut)))
    return {"sampled": int(len(pick)), "mismatch": int(bad), "worst_rel_diff": worst}


def _metrics(y, p):
    from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
    y = np.asarray(y, int); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); yh = (p > 0.5).astype(int)
    two = len(np.unique(y)) == 2
    return dict(n=int(len(y)), sweep_rate=round(float(y.mean()), 4), acc=round(float((yh == y).mean()), 4),
                bacc=round(float(balanced_accuracy_score(y, yh)), 4) if two else float("nan"),
                auc=round(float(roc_auc_score(y, p)), 4) if two else float("nan"),
                logloss=round(float(log_loss(y, p, labels=[0, 1])), 4),
                brier=round(float(brier_score_loss(y, p)), 4))


def run(E: pd.DataFrame, out: Path) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier, export_text

    cols = feat_cols(E); ctx = [k for k in cols if k.startswith("f_")]
    E = E.copy()
    E["split"] = split_of(E["timestamp"])
    # 라벨 창이 다음 구간으로 넘어가는 사건은 버린다(엠바고)
    E = E[split_of(E["timestamp"] + (H + 1) * BAR) == E["split"]]
    outcome = E["touch"].map({2: "both_same_bar", 0: "timeout", -9: "no_future"}).fillna("resolved")
    counts = {sp: outcome[E["split"] == sp].value_counts().to_dict() for sp in ("TRAIN", "VAL", "OOS", "HOLDOUT")}
    D = E[E["y"].notna()].reset_index(drop=True)
    D[cols] = D[cols].replace([np.inf, -np.inf], np.nan)
    sp = D["split"].to_numpy(); y = D["y"].to_numpy(int)
    tr, va = sp == "TRAIN", sp == "VAL"
    trva = tr | va
    X = D[cols].to_numpy(float)
    Xf = np.nan_to_num(X, nan=0.0)                  # LR/트리용 (HGB 는 NaN 을 직접 다룬다)

    preds, notes = {}, {}
    preds["R0_majority"] = np.full(len(D), float(y[tr].mean() > 0.5))
    preds["R1_dashboard_closed_back"] = D["g_closed_back"].to_numpy(float)
    t2 = DecisionTreeClassifier(max_depth=2, min_samples_leaf=200, random_state=SEED).fit(Xf[tr], y[tr])
    preds["R2_depth2_tree"] = t2.predict_proba(Xf)[:, 1]
    notes["R2_rule"] = export_text(t2, feature_names=cols)

    def pick(make, grid, Xm, name):
        best = None
        for g in grid:
            m = make(**g).fit(Xm[tr], y[tr])
            ll = log_loss(y[va], np.clip(m.predict_proba(Xm[va])[:, 1], 1e-6, 1 - 1e-6))
            if best is None or ll < best[0]:
                best = (ll, g)
        notes[f"{name}_params"] = best[1]
        m_tr = make(**best[1]).fit(Xm[tr], y[tr])        # VAL 은 TRAIN 만으로 학습한 모델로 채점
        m_all = make(**best[1]).fit(Xm[trva], y[trva])   # OOS/HOLDOUT 은 TRAIN+VAL 로 다시 학습
        p = m_all.predict_proba(Xm)[:, 1]
        p[va] = m_tr.predict_proba(Xm[va])[:, 1]
        p[tr] = m_tr.predict_proba(Xm[tr])[:, 1]
        return p, m_tr

    lr = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    preds["LR"], _ = pick(lr, [dict(C=c) for c in (0.01, 0.1, 1.0)], Xf, "LR")
    hgb = lambda **g: HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, **g)
    grid = [dict(learning_rate=lr_, max_iter=it, max_leaf_nodes=ml, min_samples_leaf=msl, l2_regularization=1.0)
            for lr_ in (0.03, 0.1) for it in (150, 400) for ml in (15, 31) for msl in (100, 400)]
    preds["HGB"], hgb_tr = pick(hgb, grid, X, "HGB")
    Xc = D[ctx].to_numpy(float)
    preds["HGB_ctx_no_trigger_bar"], _ = pick(hgb, grid, Xc, "HGB_ctx")

    # 표 — 모든 모델 × 모든 구간
    rows = []
    for name, p in preds.items():
        for s_ in ("TRAIN", "VAL", "OOS", "HOLDOUT"):
            m = sp == s_
            if m.sum() >= 20:
                rows.append(dict(model=name, split=s_, **_metrics(y[m], p[m])))
    M = pd.DataFrame(rows)
    learned = ["LR", "HGB", "HGB_ctx_no_trigger_bar"]
    val_ll = {k: float(M[(M.model == k) & (M.split == "VAL")]["logloss"].iloc[0]) for k in learned}
    best = min(val_ll, key=val_ll.get)                # 선택은 VAL 만으로

    # OOS 정확도 일군집 부트 CI, 확신 상위 커버리지 (임계는 VAL 분위에서 정해 OOS 에 적용)
    days = D["timestamp"].dt.floor("D").astype("int64").to_numpy()
    detail = {}
    for s_ in ("OOS", "HOLDOUT"):
        m = sp == s_
        if m.sum() < 50:
            continue
        pb = preds[best]
        corr = ((pb[m] > 0.5).astype(int) == y[m]).astype(float)
        lo_, hi_ = P0.day_cluster_boot(corr, days[m], B=1000)
        conf_va, conf = np.abs(preds[best][va] - 0.5), np.abs(pb[m] - 0.5)
        cov = {}
        for q in (0.1, 0.2, 0.3, 0.5, 1.0):
            thr = np.quantile(conf_va, 1 - q) if q < 1 else -1.0
            k = conf >= thr
            if k.sum() >= 20:
                cov[f"top{int(q*100)}pct_by_VAL_thr"] = dict(coverage=round(float(k.mean()), 3), n=int(k.sum()),
                                                             acc=round(float(corr[k].mean()), 4))
        sub = {}
        for tag, mk in (("dashboard_sweep_bars", D["g_closed_back"].to_numpy() == 1),
                        ("closed_beyond_bars", D["g_closed_back"].to_numpy() == 0),
                        ("top", D["side"].to_numpy() == "top"), ("bottom", D["side"].to_numpy() == "bottom")):
            k = mk[m]
            if k.sum() >= 20:
                sub[tag] = dict(n=int(k.sum()), sweep_rate=round(float(y[m][k].mean()), 4),
                                acc_best=round(float(corr[k].mean()), 4),
                                acc_R1=round(float(((preds["R1_dashboard_closed_back"][m][k] > 0.5) == y[m][k]).mean()), 4))
        detail[s_] = dict(acc=round(float(corr.mean()), 4), acc_ci95_day_cluster=[round(lo_, 4), round(hi_, 4)],
                          selective=cov, subgroups=sub)

    # 누수 진단
    Dtr = D[tr]
    single = {k: abs(roc_auc_score(y[tr], Dtr[k].fillna(Dtr[k].median())) - 0.5) + 0.5 for k in cols
              if Dtr[k].nunique() > 1}
    pi = permutation_importance(hgb_tr, X[va], y[va], scoring="accuracy", n_repeats=5, random_state=SEED)
    imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
    quint = {}
    for k in imp.index[:3]:
        v = Dtr[k]
        try:
            q = pd.qcut(v.rank(method="first"), 5, labels=False)
            r = pd.Series(y[tr]).groupby(q.to_numpy()).mean()
            quint[k] = dict(rates=[round(float(x), 4) for x in r], spread=round(float(r.max() - r.min()), 4))
        except ValueError:
            continue
    get = lambda k, s_: float(M[(M.model == k) & (M.split == s_)]["acc"].iloc[0]) if ((M.model == k) & (M.split == s_)).any() else float("nan")
    gap = get(best, "OOS") - get("R2_depth2_tree", "OOS")
    top_flat = bool(quint) and quint[next(iter(quint))]["spread"] < 0.02
    leak = dict(
        single_feature_auc_max=dict(feature=max(single, key=single.get), auc=round(max(single.values()), 4)),
        single_feature_auc_fail=bool(max(single.values()) >= 0.95),
        model_val_auc_fail=bool(M[(M.split == "VAL") & M.model.isin(learned)]["auc"].max() >= 0.99),
        best_minus_R2_oos_acc=round(gap, 4),
        perm_importance_top8={k: round(float(x), 4) for k, x in imp.head(8).items()},
        top_feature_quintile_sweep_rate=quint,
        detection_signal_flag=bool(gap >= 0.05 and top_flat),
    )
    wf = walkforward(D, cols, ctx)
    rep = dict(
        walkforward=wf,
        holdout_status="spent (2026-04+ 는 08-29 V반등 연구에서 소진 선언) — 진단 전용",
        spec=dict(lookback=LOOKBACK, htf=HTF, cooldown=COOLDOWN, k_atr=K_ATR, horizon_bars=H, splits=SPLITS,
                  label="symmetric first_touch from close[i], start i+1; inside-first=SWEEP(1), beyond-first=RUN(0)",
                  features="g_*=bar i (closed), f_*=through i-1; label starts i+1"),
        events_by_outcome=counts, best_by_val_logloss=best, val_logloss=val_ll, params=notes,
        best_detail=detail, leak=leak,
        fresh_forward_bar_by_bar=False,   # 분류 연구 점수. 거래 fresh-forward 아님
        causal_features_verified_by_truncation=None,
        trade_ledgers_used_as_input=False, saved_parent_exit_timestamps_used=False, future_rows_used_for_entry=False,
        promotion_evidence=False,
    )
    out.mkdir(parents=True, exist_ok=True)
    M.to_csv(out / "metrics.csv", index=False)
    D.to_parquet(out / "events.parquet", index=False)
    return rep, M


def walkforward(D: pd.DataFrame, cols: list, ctx: list) -> dict:
    """월별 확장창: 매달, 그 달 시작 전에 라벨 창까지 끝난 사건만으로 학습해 그 달 사건을 채점한다.
    하이퍼파라미터는 고정(튜닝 없음). 3개월 OOS(~700건)의 우연 폭 ±4pp 로는 로직 간 차이를 못 가리므로
    표본외 사건을 ~1만 건으로 늘려 비교한다."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    y = D["y"].to_numpy(int); ts = D["timestamp"]
    X = D[cols].to_numpy(float); Xf = np.nan_to_num(X, nan=0.0); Xc = D[ctx].to_numpy(float)
    month = ts.dt.to_period("M"); end = ts + (H + 1) * BAR
    hgb = lambda: HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, learning_rate=0.05,
                                                 max_iter=200, max_leaf_nodes=15, min_samples_leaf=200,
                                                 l2_regularization=1.0)
    specs = {"R2_depth2_tree": (lambda: DecisionTreeClassifier(max_depth=2, min_samples_leaf=200, random_state=SEED), Xf),
             "LR": (lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)), Xf),
             "HGB": (hgb, X), "HGB_ctx_no_trigger_bar": (hgb, Xc)}
    P = {k: np.full(len(D), np.nan) for k in ["R1_dashboard_closed_back", *specs]}
    for mo in sorted(month[ts >= pd.Timestamp(WF_START)].unique()):
        te = (month == mo).to_numpy(); trn = (end < mo.start_time).to_numpy()
        if trn.sum() < 1000 or te.sum() == 0:
            continue
        P["R1_dashboard_closed_back"][te] = D["g_closed_back"].to_numpy(float)[te]
        for k, (make, Xm) in specs.items():
            P[k][te] = make().fit(Xm[trn], y[trn]).predict_proba(Xm[te])[:, 1]
    m = np.isfinite(P["LR"])
    days = ts.dt.floor("D").astype("int64").to_numpy()[m]; yr = ts.dt.year.to_numpy()[m]
    corr = {k: ((p[m] > 0.5).astype(int) == y[m]).astype(float) for k, p in P.items()}
    res = {"n": int(m.sum()), "start": WF_START, "sweep_rate": round(float(y[m].mean()), 4), "models": {}}
    for k, c_ in corr.items():
        lo_, hi_ = P0.day_cluster_boot(c_, days, B=500)
        res["models"][k] = dict(acc=round(float(c_.mean()), 4), ci95=[round(lo_, 4), round(hi_, 4)],
                                by_year={int(u): round(float(c_[yr == u].mean()), 4) for u in np.unique(yr)})
    best = max(specs, key=lambda k: corr[k].mean())
    for ref_ in ("R1_dashboard_closed_back", "R2_depth2_tree"):
        if ref_ != best:
            lo_, hi_ = P0.day_cluster_boot(corr[best] - corr[ref_], days, B=500)
            res[f"{best}_minus_{ref_}"] = dict(diff=round(float((corr[best] - corr[ref_]).mean()), 4),
                                               ci95=[round(lo_, 4), round(hi_, 4)])
    res["best_learned"] = best
    return res


def synthetic_randomwalk(kl: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """타임스탬프만 유지하고 가격을 1분 무작위보행(5분봉 집계)으로 바꾼다. 예측가능성 0 인 귀무."""
    rng = np.random.default_rng(seed)
    n = len(kl)
    lr = rng.standard_normal((n, 5)) * 0.0008                       # 1분 로그수익 σ≈8bp
    path = 2000.0 * np.exp(np.cumsum(lr.ravel())).reshape(n, 5)
    op = np.concatenate([[2000.0], path[:-1, -1]])
    vol = rng.lognormal(6.0, 0.5, n)
    return pd.DataFrame(dict(timestamp=kl["timestamp"].to_numpy(), open=op,
                             high=np.maximum(path.max(1), op), low=np.minimum(path.min(1), op), close=path[:, -1],
                             volume=vol, taker_buy_base=vol * rng.uniform(0.3, 0.7, n)))


def selftest() -> None:
    # first_touch: 위 먼저 / 아래 먼저 / 같은 봉 양쪽 / 미결 / 미래 부족
    hi = np.array([10., 11, 12.5, 10, 10]); lo = np.array([9., 9.5, 10, 7, 9.9])
    assert first_touch(hi, lo, np.array([1]), np.array([10.]), np.array([2.]), 3)[0] == 1
    assert first_touch(hi, lo, np.array([3]), np.array([10.]), np.array([2.]), 2)[0] == -1
    assert first_touch(np.array([13.]), np.array([7.]), np.array([0]), np.array([10.]), np.array([2.]), 1)[0] == 2
    assert first_touch(hi, lo, np.array([0]), np.array([10.]), np.array([5.]), 2)[0] == 0
    assert first_touch(hi, lo, np.array([4]), np.array([10.]), np.array([5.]), 3)[0] == -9
    assert list(_bars_since(np.array([True, False, False, True, False]), 99)) == [99, 1, 2, 3, 1]
    # 합성 경로에서: 라벨은 봉 i 의 고/저/시가/거래량에 불변, 피쳐는 i+1 이후 봉에 불변
    ts = pd.date_range("2025-01-01", periods=3000, freq="5min")
    kl = synthetic_randomwalk(pd.DataFrame({"timestamp": ts}), seed=7)
    E = build(kl, None)
    assert len(E) > 20 and E["y"].notna().sum() > 20, len(E)
    # build() 의 라벨 = 봉 i+1 부터의 경로 + close[i]·ATR[i] 뿐이어야 한다
    re_ = first_touch(kl["high"].to_numpy(float), kl["low"].to_numpy(float), E["i"].to_numpy() + 1,
                      E["ref"].to_numpy(), K_ATR * E["atr_abs"].to_numpy(), H)
    assert np.array_equal(re_, E["touch"].to_numpy()), "build() 라벨 시작이 i+1 이 아니다"
    r = E.iloc[len(E) // 2]; i = int(r.i)
    k2 = kl.copy()
    k2.loc[i, ["volume", "taker_buy_base"]] *= [3.0, 0.5]
    k2.loc[i, "open"] = (k2.loc[i, "high"] + k2.loc[i, "low"]) / 2       # 시가는 TR(=ATR 폭)에 안 들어간다
    E2 = build(k2, None)
    r2 = E2[(E2.timestamp == r.timestamp) & (E2.side == r.side)].iloc[0]
    assert r2.touch == r.touch, "라벨이 봉 i 시가/거래량에 의존"
    # 고/저는 ATR[i] 폭을 바꾸므로 폭을 고정하고 경로 불변성만 본다(라벨은 i+1 부터)
    hi_, lo_ = kl["high"].to_numpy(float).copy(), kl["low"].to_numpy(float).copy()
    base = first_touch(hi_, lo_, np.array([i + 1]), np.array([r.ref]), np.array([K_ATR * r.atr_abs]), H)[0]
    hi_[i] += 50 * r.atr_abs; lo_[i] -= 50 * r.atr_abs
    assert first_touch(hi_, lo_, np.array([i + 1]), np.array([r.ref]), np.array([K_ATR * r.atr_abs]), H)[0] == base
    k3 = kl.copy()
    k3.loc[i + 1:, ["open", "high", "low", "close"]] *= 1.05
    E3 = build(k3, None)
    r3 = E3[(E3.timestamp == r.timestamp) & (E3.side == r.side)].iloc[0]
    cols = feat_cols(E)
    assert np.allclose(r[cols].to_numpy(float), r3[cols].to_numpy(float), equal_nan=True), "피쳐가 미래 봉에 의존"
    print("selftest OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--synthetic-null", action="store_true", help="가격을 무작위보행으로 바꿔 누수 귀무 검정")
    ap.add_argument("--csv", default=str(P0.CSV)); ap.add_argument("--btc", default=str(P0.BTC))
    ap.add_argument("--truncation-n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=SEED, help="--synthetic-null 의 무작위보행 시드")
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return 0
    kl = P0.load(Path(a.csv))
    btc = P0.load(Path(a.btc)) if Path(a.btc).exists() else None
    out = OUT
    if a.synthetic_null:
        kl, btc, out = synthetic_randomwalk(kl, a.seed), None, OUT.with_name(f"{OUT.name}_null_s{a.seed}")
    print(f"[1/3] ETH 5분봉 {len(kl):,}  {kl.timestamp.iloc[0]} ~ {kl.timestamp.iloc[-1]} UTC  BTC={'있음' if btc is not None else '없음'}", flush=True)
    E = build(kl, btc)
    print(f"[2/3] 사건 {len(E):,}건 (천장 {int((E.side=='top').sum()):,} / 바닥 {int((E.side=='bottom').sum()):,}), "
          f"해결 {int(E.y.notna().sum()):,}건 · 절단 인과성 검사 {a.truncation_n}건 …", flush=True)
    tc = truncation_check(kl, btc, E, a.truncation_n)
    rep, M = run(E, out)
    rep["causal_features_verified_by_truncation"] = tc
    rep["synthetic_null"] = bool(a.synthetic_null)
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))

    print(f"[3/3] 결과 → {out.relative_to(ROOT)}\n")
    piv = M.pivot_table(index="model", columns="split", values="acc")[["TRAIN", "VAL", "OOS"] + (["HOLDOUT"] if "HOLDOUT" in set(M.split) else [])]
    print("정확도 (라벨 직접 계수, 기저=sweep_rate)"); print(piv.round(4).to_string())
    print("\nOOS 상세"); print(M[M.split == "OOS"].set_index("model").drop(columns="split").to_string())
    print(f"\nVAL 로그손실로 고른 모델: {rep['best_by_val_logloss']}")
    for s_, d in rep["best_detail"].items():
        print(f"  {s_}: acc {d['acc']}  CI95(일군집) {d['acc_ci95_day_cluster']}")
        for k, v in d["selective"].items(): print(f"     {k}: {v}")
        for k, v in d["subgroups"].items(): print(f"     {k}: {v}")
    wf = rep["walkforward"]
    print(f"\n월별 확장창 walk-forward ({wf['start']}~, n={wf['n']:,}, 기저 sweep_rate={wf['sweep_rate']})")
    for k, v in wf["models"].items(): print(f"  {k:<26} acc {v['acc']:.4f} CI95 {v['ci95']}  연도별 {v['by_year']}")
    for k, v in wf.items():
        if "_minus_" in k: print(f"  {k}: {v}")
    print(f"\nR2 규칙:\n{rep['params']['R2_rule']}")
    print("누수 진단:", json.dumps(rep["leak"], ensure_ascii=False))
    print("절단 인과성:", tc)
    fail = rep["leak"]["single_feature_auc_fail"] or rep["leak"]["model_val_auc_fail"] or tc["mismatch"] > 0
    print(json.dumps({"done": True, "fail": bool(fail), "flag": rep["leak"]["detection_signal_flag"]}, ensure_ascii=False))
    return int(fail)


if __name__ == "__main__":
    raise SystemExit(main())
