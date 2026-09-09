#!/usr/bin/env python3
"""증거신호 메타라벨의 **타깃을 '익절 터치'에서 '진짜 극점'으로 교체** (2026-09-09, 사용자 지시).

동기: 2026-09-09 귀속 분해에서 증거신호의 병목이 청산이 아니라 **진입 정밀도**로 확인됐다.
  · 발동 봉이 진짜 ±12봉 극점일 확률 21.1%(무작위 2.9%의 3.6~12.4배) — 감지 자체는 실력
  · 극점 맞힌 발동 +34~+62bp / 빗나간 발동 −12~−50bp → 본전에 필요한 정밀도 **40.3%**
  · 현행 메타라벨 타깃(익절 K×ATR 터치)은 **변동성만 커도 달성**되므로 이 정밀도를 못 올린다
따라서 타깃을 정밀도와 같은 축으로 바꾼다.

🔴사건 라벨 경계 계약 준수 (CLAUDE.md):
  트리거 = 발동 봉 i (봉 i 종가에 확정)   ·   피쳐 = **봉 i 까지만**
  라벨   = **i+1 부터** 탐색: 바닥이면 `min(low[i+1..i+12]) >= low[i]` (앞으로 더 안 내려간다)
           천장이면 `max(high[i+1..i+12]) <= high[i]`
  기준 레벨 low[i]/high[i] 는 봉 i 종가 시점에 이미 알려진 값이다 → 미래참조 없음.
  ⚠️ ±12봉 양방향 정의(과거 절반 포함)를 쓰지 않는 이유: 그 절반이 피쳐 창과 같은 봉을 공유한다.

평가: AUC 뿐 아니라 **상위 분위 정밀도**(목표 40%)와 **경제성**(순bp, 같은 측면 무작위 귀무)까지.
분할(명시): TRAIN ~2026-03-31 · VAL 2026-04-01~2026-06-15 · OOS 2026-06-16~마지막.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import research_eth_signal_hitrate_20260909 as HR
import live_evidence_signal_dashboard_20260823 as EV
import build_eth_anchor_label_dataset_20260907 as B

OUT = ROOT / "tmp/eth_signal_map_20260909"
W = 12                      # 라벨 전방 창(60분) — 귀속 분해와 같은 스케일
H_EVAL = 48                 # 경제성 평가 홀딩(4시간)
COST = 10.0
SEEDS = [20260909, 771233, 305610, 517758, 961476]
VAL0, OOS0 = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-06-16")
RNG = np.random.default_rng(20260909)


def build(days: float):
    now = int(time.time() * 1000); t0 = now - int(days * 86400 * 1000)
    kl = HR.page("ETHUSDT", t0 - 1100 * 300_000, now); btc = HR.page("BTCUSDT", t0 - 1100 * 300_000, now)
    try:
        fund = EV.fetch_funding_history(limit=1000)
        fund["calc_time"] = pd.to_datetime(fund["calc_time"]).dt.tz_localize(None)
    except Exception:
        fund = None
    sig = EV.compute_signals(kl, btc_df=btc, funding_df=fund)
    btc_cl = btc.set_index("timestamp")["close"].reindex(
        pd.DatetimeIndex(sig["timestamp"])).ffill().to_numpy(float)
    return sig, btc_cl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=730.0)
    a = ap.parse_args()
    sig, btc_cl = build(a.days)
    ts = pd.to_datetime(sig["timestamp"])
    op = sig["open"].to_numpy(float); hi_ = sig["high"].to_numpy(float)
    lo_ = sig["low"].to_numpy(float); cl = sig["close"].to_numpy(float)
    atr = sig["atr_pct"].to_numpy(float); n = len(sig)
    lo, hi = 900, n - H_EVAL - W - 3
    print(f"봉 {n:,} · 평가 {ts.iloc[lo]} ~ {ts.iloc[hi]} UTC", flush=True)

    # ── 피쳐(전부 봉 i 까지) ────────────────────────────────────────────────
    S = pd.DataFrame(index=range(n))
    # ⚠️funding_z 는 API 가 1000행(=333일)만 줘서 2년 창의 앞쪽이 통째로 NaN 이다 -> 피쳐에서 뺀다
    #   (compute_signals 안에서는 그대로 쓰인다 -- orthogonal_combo 바닥 팔의 OR 조건)
    for c in ("p_fast", "p_slow", "delta_z", "vol_z", "lower_wick_ratio", "upper_wick_ratio",
              "ret3_z", "atr_pct", "dem", "kalman_dev_z"):
        S[c] = sig[c].to_numpy(float)
    c_s = pd.Series(cl)
    S["atr_pctile"] = pd.Series(atr).rolling(2016, min_periods=500).rank(pct=True).to_numpy()
    for w in (12, 48, 144):
        S[f"ret{w}"] = (c_s / c_s.shift(w) - 1).to_numpy() / np.maximum(atr, 1e-9)
        S[f"pos_in_range{w}"] = ((cl - pd.Series(lo_).rolling(w).min().to_numpy())
                                 / np.maximum(pd.Series(hi_).rolling(w).max().to_numpy()
                                              - pd.Series(lo_).rolling(w).min().to_numpy(), 1e-9))
        S[f"dist_lo{w}_atr"] = ((cl - pd.Series(lo_).rolling(w).min().to_numpy())
                                / np.maximum(cl * atr, 1e-9))
        S[f"dist_hi{w}_atr"] = ((pd.Series(hi_).rolling(w).max().to_numpy() - cl)
                                / np.maximum(cl * atr, 1e-9))
    b_s = pd.Series(btc_cl)
    S["btc_ret12"] = (b_s / b_s.shift(12) - 1).to_numpy() / np.maximum(atr, 1e-9)
    S["btc_ret48"] = (b_s / b_s.shift(48) - 1).to_numpy() / np.maximum(atr, 1e-9)
    S["eth_btc_div"] = S["ret12"] - S["btc_ret12"]
    S["hour"] = ts.dt.hour.to_numpy(); S["weekday"] = ts.dt.weekday.to_numpy()
    for s in B.SIGNALS:                              # 어느 신호가 같이 켜졌나(측면 정렬)
        S[f"f_{s}"] = 0.0
    FEATS = list(S.columns) + ["n_signals", "is_bottom"]

    rows = []
    for sd, long in (("bottom", True), ("top", False)):
        fires = {s: sig[f"{sd}_{s}"].fillna(False).to_numpy(bool) for s in B.SIGNALS}
        any_f = np.zeros(n, bool); cnt = np.zeros(n, int)
        for s in B.SIGNALS:
            any_f |= fires[s]; cnt += fires[s].astype(int)
        idx = np.flatnonzero(any_f); idx = idx[(idx >= lo) & (idx <= hi)]
        # 라벨: i+1..i+W 에서 봉 i 의 극값을 깨지 않는가 (전방 전용)
        if long:
            fwd = np.array([lo_[i + 1:i + 1 + W].min() for i in idx]); y = (fwd >= lo_[idx]).astype(int)
        else:
            fwd = np.array([hi_[i + 1:i + 1 + W].max() for i in idx]); y = (fwd <= hi_[idx]).astype(int)
        X = S.iloc[idx].copy()
        for s in B.SIGNALS:
            X[f"f_{s}"] = fires[s][idx].astype(float)
        X["n_signals"] = cnt[idx].astype(float); X["is_bottom"] = 1.0 if long else 0.0
        X["_i"] = idx; X["_y"] = y; X["_ts"] = ts.iloc[idx].to_numpy(); X["_long"] = long
        rows.append(X)
    A = pd.concat(rows, ignore_index=True).sort_values("_ts").reset_index(drop=True)
    A[FEATS] = A[FEATS].replace([np.inf, -np.inf], np.nan)
    A = A.dropna(subset=FEATS).reset_index(drop=True)
    print(f"모집단 {len(A):,}건 · 기저(극점) {A._y.mean()*100:.1f}% "
          f"· 바닥 {int(A._long.sum()):,} / 천장 {int((~A._long).sum()):,}", flush=True)

    tr = A[A._ts < VAL0]; va = A[(A._ts >= VAL0) & (A._ts < OOS0)]; oo = A[A._ts >= OOS0]
    if min(len(tr), len(va), len(oo)) < 200:
        print(f"⚠️분할 부족 TRAIN {len(tr)} VAL {len(va)} OOS {len(oo)}"); return 1
    print(f"TRAIN {len(tr):,} ({tr._ts.min():%Y-%m-%d}~{tr._ts.max():%Y-%m-%d}, 기저 {tr._y.mean()*100:.1f}%) · "
          f"VAL {len(va):,} (기저 {va._y.mean()*100:.1f}%) · OOS {len(oo):,} (기저 {oo._y.mean()*100:.1f}%)\n",
          flush=True)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    P = {k: np.zeros(len(d)) for k, d in (("va", va), ("oo", oo))}
    for sd_ in SEEDS:
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_depth=6,
                                           l2_regularization=1.0, random_state=sd_,
                                           early_stopping=True, validation_fraction=0.15)
        m.fit(tr[FEATS], tr._y)
        P["va"] += m.predict_proba(va[FEATS])[:, 1] / len(SEEDS)
        P["oo"] += m.predict_proba(oo[FEATS])[:, 1] / len(SEEDS)
    va = va.assign(p=P["va"]); oo = oo.assign(p=P["oo"])
    print(f"AUC  VAL {roc_auc_score(va._y, va.p):.4f} · OOS {roc_auc_score(oo._y, oo.p):.4f}", flush=True)

    def netbp(d, H=H_EVAL):
        i = d._i.to_numpy(); e = op[i + 1]; x = cl[i + H]
        r = (x - e) / e * 1e4
        return np.where(d._long.to_numpy(), r, -r) - COST

    days_oo = (oo._ts.max() - oo._ts.min()).total_seconds() / 86400
    print(f"\n■ 상위 분위 정밀도와 경제성 (OOS {days_oo:.0f}일) — 목표 정밀도 40%")
    print(f"{'커버리지':>10}{'건수':>8}{'건/일':>7}{'정밀도':>9}{'기저대비':>10}{'순bp(H48)':>12}{'전체대비':>10}")
    base_prec = oo._y.mean(); base_bp = netbp(oo).mean()
    res = []
    for cov in (1.0, 0.5, 0.25, 0.10, 0.05):
        q = oo.nlargest(max(int(len(oo) * cov), 20), "p")
        pr = q._y.mean(); nb = netbp(q).mean()
        print(f"{cov*100:>9.0f}%{len(q):>8}{len(q)/days_oo:>7.1f}{pr*100:>8.1f}%"
              f"{(pr-base_prec)*100:>+9.1f}pp{nb:>12.2f}{nb-base_bp:>+10.2f}")
        res.append(dict(cov=cov, n=len(q), per_day=round(len(q)/days_oo, 2),
                        prec=round(float(pr), 4), net_bp=round(float(nb), 2)))
    # 같은 측면 무작위 진입 귀무 (상위 10% 커버리지 기준)
    q = oo.nlargest(max(int(len(oo) * 0.10), 20), "p")
    pool = np.arange(lo, hi)
    nl = []
    for _ in range(300):
        r = RNG.choice(pool, len(q), replace=False)
        e = op[r + 1]; x = cl[r + H_EVAL]; rr = (x - e) / e * 1e4
        nl.append(np.where(q._long.to_numpy(), rr, -rr).mean() - COST)
    nl = np.array(nl)
    print(f"\n  상위10% vs 같은측면 무작위 진입: {netbp(q).mean():+.2f} vs {nl.mean():+.2f}bp "
          f"→ 초과 {netbp(q).mean()-nl.mean():+.2f}bp · p={float((nl>=netbp(q).mean()).mean()):.4f}")
    h = oo._ts.median()
    for lab, m in (("OOS 전반", oo._ts < h), ("OOS 후반", oo._ts >= h)):
        g = oo[m]; gq = g.nlargest(max(int(len(g)*0.10), 10), "p")
        print(f"  {lab}: 상위10% 정밀도 {gq._y.mean()*100:.1f}% · 순 {netbp(gq).mean():+.2f}bp "
              f"(전체 {netbp(g).mean():+.2f})")
    pd.DataFrame(res).to_csv(OUT / "extreme_metalabel_oos.csv", index=False)
    oo[["_ts", "_i", "_y", "_long", "p"]].to_csv(OUT / "extreme_metalabel_preds.csv", index=False)
    print(json.dumps({"done": True, "auc_oos": round(float(roc_auc_score(oo._y, oo.p)), 4),
                      "base": round(float(base_prec), 4)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
