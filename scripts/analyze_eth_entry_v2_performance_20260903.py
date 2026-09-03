#!/usr/bin/env python3
"""진입 모델 v2(TabPFN) **객관적 성능 분석** (2026-09-03).

사용자 질문: "예상 수익은 얼마나 되고 MDD는 어떻게 되는지, 자동매매로 쓸만한지."

⚠️**연구 표의 bp를 그대로 계좌 수익으로 읽으면 안 된다.** `fills.csv`의 `y`는
`price_move * NOTIONAL - COST * NOTIONAL` (NOTIONAL=0.9)로, **트레이드 하나가 notional 0.9를
전부 쓴다**는 가정이다. 슬롯이 4개면 동시 노출이 최대 3.6배(마진 1.2배)로 **계좌를 초과한다**.
그래서 여기서는 두 가지로 나눠 계산한다:
  ① 연구 표기      -- 각 트레이드 notional 0.9 (동시성 무시). 비교용일 뿐 실행 불가.
  ② 자본 제약      -- notional 0.9를 슬롯 수로 나눠 배분(슬롯당 0.225). 실제 실행 가능한 형태.
동시 보유 분포를 실측해 ②가 과도하게 보수적인지도 함께 본다.

산출: 창별 equity/MDD/Calmar/Sharpe, 체결빈도, 노출률, 꼬리, **비용 민감도**,
      그리고 자동매매 적합성 판정에 필요한 미검증 항목 목록.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

B6 = ROOT / "tmp/eth_entry_b6_expand_20260903"
ART = ROOT / "tmp/eth_entry_limit_fade_v2_tabpfn_20260903"
OUT = ROOT / "tmp/eth_entry_v2_performance_20260903"
NOTIONAL, SUB = 0.9, 18000


def log(m): print(f"[perf] {m}", flush=True)


def take_slots(df, n_slots):
    """slotN과 같은 규칙이되 **채택된 행 자체**를 돌려준다(equity 구성용)."""
    d = df.sort_values("fi")
    rows, busy = [], []
    for idx, fi, ei in zip(d.index, d.fi.to_numpy(), d.ei.to_numpy()):
        busy = [b for b in busy if b > fi]
        if len(busy) < n_slots:
            rows.append(idx); busy.append(ei)
    return d.loc[rows]


def dd_stats(eq):
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    i = int(np.argmin(dd))
    j = int(np.argmax(eq[:i + 1])) if i > 0 else 0
    # 회복까지 걸린 트레이드 수
    rec = np.flatnonzero(eq[i:] >= peak[i])
    return {"mdd_bp": float(dd.min()), "mdd_at_trade": i, "peak_at_trade": j,
            "recovered_after": int(rec[0]) if len(rec) else None}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TABPFN_TOKEN="):
                os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip('"')
    from tabpfn import TabPFNClassifier

    Q = joblib.load(ART / "model.joblib")
    POL = Q["policy"]; FEATS = Q["feature_cols"]; p_thr = float(POL["p_threshold"])
    NS = int(POL["slots"])
    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    X = D[FEATS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(pd.Series(Q["feature_medians"]))
    dsel = ((D.depth == POL["depth_atr"]) & (D.btf <= POL["wait_bars"])).to_numpy()
    prow = np.flatnonzero(dsel)
    Xp = X.iloc[prow].to_numpy(np.float32)
    log(f"후보 {len(prow):,} · p_thr {p_thr:.6f} · 슬롯 {NS}")

    loc = {int(v): i for i, v in enumerate(Q["context_index"])}
    ps = []
    for sd in Q["seeds"]:
        rs = np.random.default_rng(sd).choice(Q["context_index"], size=SUB, replace=False)
        sel = np.array([loc[int(v)] for v in rs])
        m = TabPFNClassifier(device="cuda", random_state=sd)
        m.fit(Q["context_X"][sel], Q["context_y"][sel])
        ps.append(m.predict_proba(Xp)[:, 1])
    P = np.full(len(D), -np.inf); P[prow] = np.mean(ps, axis=0)
    log("채점 완료")

    rep, led = {}, []
    for w in ("VAL", "OOS", "HOLDOUT"):
        m = dsel & (D.split == w).to_numpy() & (P > p_thr)
        t = take_slots(D[m], NS)
        if not len(t):
            continue
        y = t["y"].to_numpy(float)
        days = (t.timestamp.max() - t.timestamp.min()).total_seconds() / 86400
        bp = y * 1e4
        eq_full = np.cumsum(bp)                  # ① 연구 표기 (notional 0.9 전액)
        eq_cap = np.cumsum(bp / NS)              # ② 자본 제약 (슬롯당 0.9/4)
        d1, d2 = dd_stats(eq_full), dd_stats(eq_cap)
        # 동시 보유 분포
        occ = []
        busy = []
        for fi, ei in zip(t.fi.to_numpy(), t.ei.to_numpy()):
            busy = [b for b in busy if b > fi]
            occ.append(len(busy) + 1); busy.append(ei)
        occ = np.array(occ)
        # 노출률: 보유 봉 수 / 기간 봉 수
        span = (t.ei.to_numpy() - t.fi.to_numpy()).sum()
        tot_bars = days * 288
        wins = bp > 0
        r = {
            "n": int(len(t)), "days": round(days, 1), "trades_per_day": round(len(t) / days, 2),
            "mean_bp": round(float(bp.mean()), 2), "median_bp": round(float(np.median(bp)), 2),
            "win_rate": round(float(wins.mean()), 4),
            "avg_win_bp": round(float(bp[wins].mean()), 1) if wins.any() else 0.0,
            "avg_loss_bp": round(float(bp[~wins].mean()), 1) if (~wins).any() else 0.0,
            "pf": round(float(bp[wins].sum() / -bp[~wins].sum()), 3) if (~wins).any() else np.inf,
            "worst_bp": round(float(bp.min()), 1), "best_bp": round(float(bp.max()), 1),
            "total_full_pct": round(float(eq_full[-1]) / 100, 2),
            "mdd_full_pct": round(d1["mdd_bp"] / 100, 2),
            "total_cap_pct": round(float(eq_cap[-1]) / 100, 2),
            "mdd_cap_pct": round(d2["mdd_bp"] / 100, 2),
            "calmar": round(float(eq_cap[-1]) / abs(d2["mdd_bp"]), 2) if d2["mdd_bp"] < 0 else np.inf,
            "sharpe_per_trade": round(float(bp.mean() / bp.std()), 3) if bp.std() > 0 else np.inf,
            "mdd_recover_trades": d2["recovered_after"],
            "occ_mean": round(float(occ.mean()), 2), "occ_max": int(occ.max()),
            "exposure_pct": round(float(span / tot_bars) * 100, 1) if tot_bars else 0.0,
        }
        # 비용 민감도 (y는 이미 10bp 비용 차감분. 추가 비용은 notional 곱해 차감)
        for c in (15, 20, 25, 30):
            extra = (c - 10) / 1e4 * NOTIONAL * 1e4
            b2 = bp - extra
            r[f"mean_bp_at_{c}bp"] = round(float(b2.mean()), 2)
        rep[w] = r
        led.append(t.assign(split=w, bp=bp, occ=occ))

    pd.set_option("display.width", 250)
    R = pd.DataFrame(rep).T
    print("\n=== 트레이드 단위 ===")
    print(R[["n", "days", "trades_per_day", "mean_bp", "median_bp", "win_rate",
             "avg_win_bp", "avg_loss_bp", "pf", "worst_bp", "sharpe_per_trade"]].to_string())
    print("\n=== 포트폴리오 (① 연구 표기 = 실행 불가 · ② 자본 제약 = 실행 가능) ===")
    print(R[["total_full_pct", "mdd_full_pct", "total_cap_pct", "mdd_cap_pct", "calmar",
             "mdd_recover_trades", "occ_mean", "occ_max", "exposure_pct"]].to_string())
    print("\n=== 비용 민감도 (트레이드당 평균 bp) ===")
    print(R[["mean_bp"] + [f"mean_bp_at_{c}bp" for c in (15, 20, 25, 30)]].to_string())
    L = pd.concat(led, ignore_index=True)
    L[["timestamp", "split", "signal", "side", "arm", "bp", "occ", "fi", "ei", "atr_pct"]] \
        .to_csv(OUT / "trades.csv", index=False)

    # ---- 꼬리/표본 민감도: 이 성과가 소수 관측에 얹혀 있는가 ----
    print("\n=== ⭐표본 민감도 (창별) ===")
    rng = np.random.default_rng(20260903)
    print(f"{'':9s}{'손실건수':>8s}{'평균bp':>9s}{'부트스트랩 95% CI':>22s}"
          f"{'최악3건 제거시':>14s}{'최고3건 제거시':>14s}")
    for w in ("VAL", "OOS", "HOLDOUT"):
        b = L.loc[L.split == w, "bp"].to_numpy()
        bs = np.array([rng.choice(b, len(b), replace=True).mean() for _ in range(5000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        srt = np.sort(b)
        print(f"{w:9s}{int((b<0).sum()):8d}{b.mean():+9.2f}"
              f"   [{lo:+7.2f}, {hi:+7.2f}]{srt[3:].mean():+14.2f}{srt[:-3].mean():+14.2f}")

    # ---- 월별 안정성 ----
    print("\n=== 월별 (트레이드수 · 평균bp) ===")
    L["ym"] = pd.to_datetime(L.timestamp).dt.to_period("M").astype(str)
    g = L.groupby("ym")["bp"].agg(["count", "mean"]).round(1)
    print(" · ".join(f"{i}: {int(r['count'])}건 {r['mean']:+.0f}bp" for i, r in g.iterrows()))
    neg = int((g["mean"] < 0).sum())
    print(f"⭐음수 월 {neg}/{len(g)}개")

    json.dump(rep, open(OUT / "report.json", "w"), ensure_ascii=False, indent=2, default=str)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
