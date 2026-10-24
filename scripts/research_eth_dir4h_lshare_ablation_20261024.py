"""ETH 4h 방향 HGB(배포 B2) + 롱 몫 피쳐 넣은 판 vs 뺀 판 — 1회 사전등록 비교 (2026-10-24, 로컬 CPU 만).

사용자: «배포 4h 방향 HGB에 롱 몫 피쳐를 넣은 판과 뺀 판을 같은 표본 밖 기간으로 비교하는 1회 시험을 사전 기준 정해서 돌려줘»
배경: research_long_short_dominance_trend_20261024 — 롱 몫은 같은 창 가격과 동시 .4~.5, 단독 예측·모멘텀 대비 증분 ETH 통과 0.
  배포 65피쳐에 이미 테이커 불균형(T_imb 5m~4h)·ΔOI(O_doi 5m~1d)·O_doi1h_x_r 가 있다. 남는 몫 = 봉 단위 부호 조합 + 7·30일 창.

PRE-REGISTRATION (결과 전 고정):
  Arms: B2 = 배포 그대로(FEATS 65 · HGB early_stopping=False · max_iter 300 · 모델 1개 · 결정적).
        B2L = B2 + 롱 몫 6열: LS_1h·LS_4h·LS_1d·LS_7d·LS_30d(= (신규 롱+숏 정리)$ / 네 칸 합, 12·48·288·2016·8640봉)
              + NEWL_1d(= 신규 롱 / (신규 롱+신규 숏), 288봉). 4분류 = 대시보드 QNAME(ΔOI 부호 × 테이커 델타 부호, |ΔOI|×종가).
  Walk-forward: 매달 2023-01..2026-09, cut = 달 − 4h, 학습 창 365일, 학습 행 [::2] (라이브 train() 과 같음).
  Signals: 각 판 자기 표본 밖 예측의 월 고정 20/80 분위(month_fixed_q). 채점 = 위상 8 · 테이커 4bp · 5일 블록 부트스트랩(reverify 도구).
  주 기간 2025-01-01~2026-09-14 · 보조 2023-02~2024-12.
  판정:
    채택 후보(ADD) iff 2025~ 일 손익 B2L−B2 CI 하한 > 0 AND 2023-24 평균 차 > 0 AND 2025~ IC 차 > 0.
    악화(WORSE) iff 2025~ CI 상한 < 0.
    그 밖 = 차이 없음(NO GAIN) → 피쳐 추가 안 함, 롱 몫 축 종료(대시보드 서술용만).
  점검(결과 전 고정): ① 기준판 재현 = tmp/dir4h_early_stop_20261019/fits.npz 의 pB2 와 상관 ≥ .9999
    ② 인과 = 패널을 T 에서 잘라 만든 롱 몫 피쳐가 T 이전 행에서 전체판과 같다 ③ 단일 피쳐 |IC| < .1 (누수 트립와이어)
    ④ OI·델타 짝: |ΔOI_known| 이 거래량과 가장 맞는 델타 지연 s∈{0,1} 를 SWITCH(2024-03-04) 전·후 따로 고르고 그 지연을 기록.

플래그: fresh_forward_bar_by_bar=true(월별 재학습 표본 밖 예측) · trade_ledgers_used_as_input=false · saved_parent_exit_timestamps_used=false · future_rows_used_for_entry=false.

    nice -n 15 python scripts/research_eth_dir4h_lshare_ablation_20261024.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from live_eth_dir4h_20261015 import FEATS, HGB, lag, rsum  # noqa: E402
from research_eth_dir4h_reverify_20261019 import (BD, PERIODS, diff, ic_diff_ci, ic_stats, month_fixed_q,  # noqa: E402
                                                  pearson, phase_daily, rates, summ)

OUT = ROOT / "tmp/dir4h_lshare_ablation_20261024"
VISION = Path("/home/kbj20/crypto-scalping/data/binance_vision/panel/ETHUSDT.parquet")
B2_CACHE = Path("/home/kbj20/crypto-scalping/.claude/worktrees/agent-skills-maintenance-refactor-8afcd4/tmp/dir4h_early_stop_20261019/fits.npz")
SWITCH = pd.Timestamp("2024-03-04")   # research_eth_best_direction_model_20261010: 이후 metrics 는 한 행 늦춰야 «봉 마감에 아는 값»
LCOLS = ["LS_1h", "LS_4h", "LS_1d", "LS_7d", "LS_30d", "NEWL_1d"]
WIN = {"LS_1h": 12, "LS_4h": 48, "LS_1d": 288, "LS_7d": 2016, "LS_30d": 8640}
H = 48
PURGE = np.timedelta64(5 * H, "m")


def lshare(p: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """p = vision 패널(5분 격자, timestamp 오름차순). 행 t 의 값 = 봉 t 마감에 아는 것만."""
    ts = p.timestamp
    v = p.sum_open_interest.to_numpy(float)
    oik = np.where((ts >= SWITCH).to_numpy(), lag(v, 1), v)          # 봉 마감에 아는 OI
    doi = oik - lag(oik, 1)
    vol, buy, c = p.volume.to_numpy(float), p.taker_buy_base.to_numpy(float), p.close.ffill().to_numpy(float)
    d = 2 * buy - vol
    pre = (ts < SWITCH).to_numpy()
    shift = {}
    for era, m in (("pre", pre), ("post", ~pre)):   # ④ ΔOI_known 과 같은 봉의 델타 지연
        if not m.any():
            shift[era] = {"best": 1, "corr": {}}
            continue
        cs = {s: pd.Series(np.abs(doi[m])).corr(pd.Series(lag(vol, s)[m]), method="spearman") for s in (0, 1)}
        shift[era] = {"best": int(max(cs, key=cs.get)), "corr": {str(k): round(float(x), 3) for k, x in cs.items()}}
    dd = np.where(pre, lag(d, shift["pre"]["best"]), lag(d, shift["post"]["best"]))
    cc = np.where(pre, lag(c, shift["pre"]["best"]), lag(c, shift["post"]["best"]))
    ok = np.isfinite(doi) & np.isfinite(dd)
    usd = np.where(ok, np.abs(doi) * cc, 0.0)
    up, b = doi > 0, dd >= 0
    L, S = np.where(b, usd, 0.0), np.where(~b, usd, 0.0)
    NL, NS = np.where(up & b, usd, 0.0), np.where(up & ~b, usd, 0.0)
    F = pd.DataFrame({"ts": ts.to_numpy()})
    for nm, w in WIN.items():
        l, s, n = rsum(L, w), rsum(S, w), rsum(ok.astype(float), w)
        F[nm] = np.where((n >= 0.9 * w) & (l + s > 0), l / np.where(l + s > 0, l + s, 1), np.nan)
    nl, ns, n = rsum(NL, 288), rsum(NS, 288), rsum(ok.astype(float), 288)
    F["NEWL_1d"] = np.where((n >= 0.9 * 288) & (nl + ns > 0), nl / np.where(nl + ns > 0, nl + ns, 1), np.nan)
    return F, shift


def load_vision() -> pd.DataFrame:
    p = pd.read_parquet(VISION).sort_values("timestamp").drop_duplicates("timestamp")
    full = pd.DataFrame({"timestamp": pd.date_range(p.timestamp.min(), p.timestamp.max(), freq="5min")})
    return full.merge(p, on="timestamp", how="left")


def walk(X, y, ts, months) -> np.ndarray:
    kw = {**HGB, "early_stopping": False}
    pred = np.full(len(ts), np.nan)
    ok = np.isfinite(y)
    for m in months:
        cut = np.datetime64(m) - PURGE
        tr = (ts >= cut - np.timedelta64(365, "D")) & (ts < cut) & ok
        te = (ts >= np.datetime64(m)) & (ts < np.datetime64(m + pd.offsets.MonthBegin(1)))
        pred[te] = HistGradientBoostingRegressor(**kw).fit(X[tr][::2], y[tr][::2]).predict(X[te])
        print(f"  {m:%Y-%m}", flush=True)
    return pred


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    P = pd.read_parquet(BD / "panel.parquet", columns=["ts", "y_4h"] + FEATS)
    vis = load_vision()
    Lf, shift = lshare(vis)
    assert len(Lf) == len(P) and (Lf.ts.to_numpy() == P.ts.to_numpy()).all(), "vision 격자 ≠ 연구 패널 격자"
    # ② 인과: T 에서 자른 패널로 만든 피쳐 = 전체판의 T 이전 행 (지연 s 는 전체판 값으로 고정해 비교)
    T = pd.Timestamp("2025-06-01")
    Lc, _ = lshare(vis[vis.timestamp < T].reset_index(drop=True))
    a, b = Lc[LCOLS].to_numpy(), Lf[LCOLS].to_numpy()[: len(Lc)]
    assert np.allclose(a, b, equal_nan=True), "자른 패널과 피쳐가 다르다 = 미래 행이 섞였다"
    for c in LCOLS:
        P[c] = Lf[c].to_numpy(np.float32)
    ts, y = P.ts.to_numpy(), P.y_4h.to_numpy()
    fin = np.isfinite(y)
    uni = {c: round(float(pd.Series(P[c][fin]).corr(pd.Series(y[fin]), method="spearman")), 4) for c in LCOLS}
    assert max(abs(v) for v in uni.values()) < 0.1, f"③ 단일 피쳐 IC 과대 = 누수 의심 {uni}"
    months = pd.date_range("2023-01-01", "2026-09-01", freq="MS")
    pB2 = walk(P[FEATS].to_numpy(np.float32), y, ts, months)
    cache = np.load(B2_CACHE)["pB2"]
    both = np.isfinite(cache) & np.isfinite(pB2)
    repro = float(np.corrcoef(cache[both], pB2[both])[0, 1])
    assert repro >= 0.9999, f"① 기준판 재현 실패 {repro}"
    pL = walk(P[FEATS + LCOLS].to_numpy(np.float32), y, ts, months)
    np.savez(OUT / "preds.npz", pB2=pB2, pL=pL)
    arms = {"B2": pB2, "B2L": pL}
    rep = {"checks": {"b2_repro_corr": round(repro, 6), "causal_cut": str(T.date()), "univariate_spearman_y4h": uni, "doi_delta_shift": shift,
                      "feature_nan_rate_2025": {c: round(float(np.isnan(P[c][ts >= np.datetime64("2025-01-01")]).mean()), 4) for c in LCOLS}},
           "pnl": {}, "rates": {}, "diff": {}, "ic": {}, "ic_diff": {}}
    D = {}
    for nm, p in arms.items():
        lo, hi = month_fixed_q(p, ts, months)
        rep["pnl"][nm], rep["rates"][nm] = {}, {}
        for lab, a_, b_ in PERIODS:
            D[nm, lab] = phase_daily(p, lo, hi, ts, y, a_, b_); rep["pnl"][nm][lab] = summ(D[nm, lab])
            rep["rates"][nm][lab] = {k: v for k, v in rates(p, lo, hi, ts, a_, b_).items() if k in ("long", "short")}
    rep["diff"]["B2L − B2"] = {lab: diff(D["B2L", lab], D["B2", lab]) for lab, _, _ in PERIODS}
    ok = fin & np.isfinite(pB2) & np.isfinite(pL)
    yr = pd.DatetimeIndex(ts).year
    spans = [(str(Y), yr == Y) for Y in (2023, 2024, 2025, 2026)] + [(lab, (ts >= np.datetime64(a_)) & (ts < np.datetime64(b_))) for lab, a_, b_ in PERIODS]
    for lab, w in spans:
        m = ok & w
        day = pd.to_datetime(ts[m]).floor("D")
        S = {nm: ic_stats(p[m], y[m], day) for nm, p in arms.items()}
        rep["ic"][lab] = {nm: round(float(pearson(s.sum(0))), 4) for nm, s in S.items()}
        if lab in ("2025~", "2023-24"):
            v = ic_diff_ci(S["B2L"], S["B2"]); rep["ic_diff"][lab] = [round(v[0], 4), v[1]]
    m25 = ok & (ts >= np.datetime64("2025-01-01"))
    rep["pred_corr_2025"] = round(float(np.corrcoef(pB2[m25], pL[m25])[0, 1]), 4)
    dm = rep["diff"]["B2L − B2"]
    if dm["2025~"]["ci"][0] > 0 and dm["2023-24"]["mean"] > 0 and rep["ic_diff"]["2025~"][0] > 0:
        rep["decision"] = "ADD"
    elif dm["2025~"]["ci"][1] < 0:
        rep["decision"] = "WORSE"
    else:
        rep["decision"] = "NO GAIN"
    rep["flags"] = {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False,
                    "future_rows_used_for_entry": False}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    for k in ("checks", "pnl", "diff", "ic", "ic_diff", "rates", "pred_corr_2025", "decision"):
        print(k, json.dumps(rep[k], ensure_ascii=False), flush=True)


def selfcheck() -> None:
    # 4분류: 매 봉 OI↑ + 매수 우위 → 전부 신규 롱 → LS = NEWL = 1, 매도 우위면 0
    n = 9000
    ts = pd.date_range("2024-06-01", periods=n, freq="5min")
    oi = 1000 + np.arange(n, dtype=float)
    p = pd.DataFrame({"timestamp": ts, "sum_open_interest": oi, "volume": 10.0, "taker_buy_base": 8.0, "close": 100.0})
    F, _ = lshare(p)
    assert np.allclose(F.LS_1h.dropna(), 1) and np.allclose(F.NEWL_1d.dropna(), 1)
    F2, _ = lshare(p.assign(taker_buy_base=2.0))
    assert np.allclose(F2.LS_1d.dropna(), 0) and np.allclose(F2.NEWL_1d.dropna(), 0)
    assert F.LS_30d.isna().sum() >= 8640 - 1
    print("selfcheck ok")


if __name__ == "__main__":
    selfcheck()
    if "--selfcheck" not in sys.argv:
        main()
