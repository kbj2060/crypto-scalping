#!/usr/bin/env python3
"""일 단위 방향 «온라인 전문가 가중»(동적 모델 평균) 4코인 — 사전 등록 docs/prereg/dir_online_experts_20261025.md (648a1351).

    python3 scripts/research_dir_online_experts_20261025.py --selfcheck   # 트립와이어 T1·T2·T3
    python3 scripts/research_dir_online_experts_20261025.py               # 개발 구간(2022~2025) CPCV·PBO·DSR 관문
    python3 scripts/research_dir_online_experts_20261025.py --holdout     # 관문 통과 때만 2026 잠금 1회 채점

산출 tmp/dir_online_experts_20261025/{dev_report.json, holdout_report.json}.
"""
import argparse, glob, io, itertools, json, math, pathlib, sys, zipfile
import numpy as np, pandas as pd
from scipy.stats import norm

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = pathlib.Path("/home/kbj20/crypto-scalping/data")   # 원천은 메인 체크아웃(워크트리엔 data 없음)
OUT = ROOT / "tmp" / "dir_online_experts_20261025"
COINS = ("ETH", "BTC", "SOL", "XRP")
KS = (1, 3, 7, 14, 28, 56)
GRID = [(H, m, sz) for H in (5, 10, 20, 40, 80, 160) for m in (1, 3, 5) for sz in ("vol", "unit")]
DEV = ("2022-01-01", "2025-12-31"); HOLD = ("2026-01-01", "2026-09-14")
COST = 4e-4; SIG_T = 0.5 / math.sqrt(365)


def daily_close(sym):
    """일 d 종가 = d 23:55 시작 5분봉 종가(= 00:00 UTC 다음 날 직전). 2021 은 1분 zip 의 23:59 봉 종가."""
    p = pd.read_parquet(D / f"binance_vision/panel/{sym}USDT.parquet", columns=["timestamp", "close"])
    a = p.set_index("timestamp").close.resample("1D").last()
    zs = []
    for f in sorted(glob.glob(str(D / f"binance_vision/klines1m_futures_pre2022/{sym}USDT-1m-*.zip"))):
        with zipfile.ZipFile(f) as z:
            raw = z.read(z.namelist()[0]).decode()
        x = pd.read_csv(io.StringIO(raw), header=None, usecols=[0, 4])
        if not str(x.iat[0, 0]).isdigit():
            x = x.iloc[1:].astype(float)
        zs.append(pd.Series(x[4].values, index=pd.to_datetime(x[0].astype("int64"), unit="ms")))
    b = pd.concat(zs).sort_index().resample("1D").last()
    s = pd.concat([b[b.index < a.index[0]], a]).dropna()
    assert s.index.is_monotonic_increasing and not s.index.duplicated().any()
    return s


def experts(c):
    """15 전문가 신호(일 d 종가까지만 사용): 모멘텀 6 · 되돌림 6 · 롱 · 숏 · 현금."""
    cols = {}
    for k in KS:
        sg = np.sign(c / c.shift(k) - 1)
        cols[f"mom{k}"] = sg; cols[f"rev{k}"] = -sg
    cols["long"] = pd.Series(1.0, c.index); cols["short"] = pd.Series(-1.0, c.index); cols["flat"] = pd.Series(0.0, c.index)
    return pd.DataFrame(cols).fillna(0.0)


def expert_perf(E, r, sig):
    """p_e(d) = [s_e(d−1)·r(d) − 비용·|Δs_e|] / σ20(d−1) — 일 d 종가에 알 수 있다."""
    sp = E.shift(1); turn = (E.shift(1) - E.shift(2)).abs()
    return ((sp.mul(r, axis=0) - COST * turn).div(sig.shift(1), axis=0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def signal_vec(P, E, H, m):
    S = P.ewm(halflife=H, adjust=False).mean().values
    order = np.argsort(-S, axis=1, kind="stable")[:, :m]          # 동점 = 전문가 번호순
    return pd.Series(np.take_along_axis(E.values, order, 1).mean(1), E.index)


def signal_loop(P, E, H, m):
    """T3 독립 재구현: 점수를 하루씩 갱신하고 정렬로 고른다."""
    a = 1 - 0.5 ** (1 / H); S = np.zeros(P.shape[1]); out = []
    for i in range(len(P)):
        S = P.values[i] if i == 0 else a * P.values[i] + (1 - a) * S
        idx = sorted(range(len(S)), key=lambda j: (-S[j], j))[:m]
        out.append(E.values[i, idx].mean())
    return pd.Series(out, E.index)


def strat_pnl(w, r):
    """w(d) 를 d 종가에 잡고 r(d+1) 을 받는다 · 회전 비용."""
    return (w.shift(1) * r - COST * (w - w.shift(1)).abs().shift(1)).fillna(0.0)


def coin_frame(sym):
    c = daily_close(sym); r = c.pct_change(); sig = r.rolling(20).std()
    return c, r, sig, experts(c)


def size(s, sig, sz):
    return s * np.minimum(2.0, SIG_T / sig).fillna(0.0) if sz == "vol" else s


def build(frames):
    """36 구성 × 4코인 → 포트폴리오 일 손익 행렬 + 기준(보유·추세 규칙)."""
    cols = {}
    for H, m, sz in GRID:
        legs = []
        for sym, (c, r, sig, E) in frames.items():
            P = expert_perf(E, r, sig)
            legs.append(strat_pnl(size(signal_vec(P, E, H, m), sig, sz), r))
        cols[f"H{H}_m{m}_{sz}"] = pd.concat(legs, axis=1).mean(1)
    hold, trend = [], []
    for sym, (c, r, sig, E) in frames.items():
        hold.append(strat_pnl(pd.Series(1.0, c.index), r))
        tr = np.sign(sum(np.sign(c / c.shift(k) - 1) for k in (7, 14, 21, 28)) / 4).fillna(0.0)   # 09-27 추세 규칙(평균 부호)
        trend.append(strat_pnl(size(tr, sig, "vol"), r))
    return pd.DataFrame(cols), pd.concat(hold, axis=1).mean(1), pd.concat(trend, axis=1).mean(1)


sharpe = lambda x: float(x.mean() / x.std() * math.sqrt(365)) if x.std() > 0 else 0.0


def mdd(x):
    eq = (1 + x).cumprod(); return float((eq / eq.cummax() - 1).min())


def scale_for_mdd(x, target=-0.20):
    lo, hi = 0.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mdd(mid * x) > target else (lo, mid)
    return lo


def cpcv(R, n=10, k=2, emb=5):
    T = len(R); g = np.array_split(np.arange(T), n); oos, logits = [], []
    for te in itertools.combinations(range(n), k):
        te_idx = np.concatenate([g[i] for i in te]); ban = set(te_idx)
        for i in te:
            ban |= set(range(g[i][-1] + 1, min(T, g[i][-1] + 1 + emb)))
        tr_idx = np.array([t for t in range(T) if t not in ban])
        is_sr = R.iloc[tr_idx].apply(sharpe); os_sr = R.iloc[te_idx].apply(sharpe)
        best = is_sr.idxmax(); oos.append(os_sr[best])
        rank = os_sr.rank()[best] / (len(os_sr) + 1)
        logits.append(math.log(rank / (1 - rank)))
    return float(np.median(oos)), float(np.mean(np.array(logits) <= 0)), oos


def dsr(x, trial_srs):
    """Bailey·López de Prado — 일 단위 샤프."""
    sr = x.mean() / x.std(); T = len(x); g3 = float(((x - x.mean()) ** 3).mean() / x.std() ** 3); g4 = float(((x - x.mean()) ** 4).mean() / x.std() ** 4)
    N = len(trial_srs); V = float(np.var(np.array(trial_srs) / math.sqrt(365), ddof=1)); em = 0.5772156649
    sr0 = math.sqrt(V) * ((1 - em) * norm.ppf(1 - 1 / N) + em * norm.ppf(1 - 1 / (N * math.e)))
    return float(norm.cdf((sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)))


def block_ci(d, b=20, n=2000, seed=7):
    rng = np.random.default_rng(seed); v = d.values; T = len(v); nb = T // b + 1; ms = []
    for _ in range(n):
        st = rng.integers(0, T - b, nb); ms.append(np.concatenate([v[s:s + b] for s in st])[:T].mean())
    return [float(np.percentile(ms, 2.5) * 1e4), float(np.percentile(ms, 97.5) * 1e4)]


def yearly(x):
    return {str(y): round(sharpe(v), 2) for y, v in x.groupby(x.index.year)}


def funding_daily(sym):
    fs = sorted(glob.glob(str(D / f"research/funding_extracted/{sym}USDT/*.csv")))
    f = pd.concat([pd.read_csv(p) for p in fs]); f.index = pd.to_datetime(f.calc_time, unit="ms")
    return f.last_funding_rate.resample("1D").sum()   # 일 d 동안 정산된 펀딩 합 = d−1 종가에 잡은 포지션이 낸다


def selfcheck(frames):
    c, r, sig, E = frames["ETH"]
    # T2 일봉 종가 독립 재구현(1분 월 parquet)
    k = pd.concat([pd.read_parquet(f, columns=["t", "c"]) for f in sorted(glob.glob(str(D / "binance_vision/klines1m/ETHUSDT-1m-2022-*.parquet")))])
    k.index = pd.to_datetime(k.t, unit="ms"); k1 = k.c.resample("1D").last()
    both = pd.concat([c, k1], axis=1, join="inner").dropna(); rel = float((both.iloc[:, 0] / both.iloc[:, 1] - 1).abs().max())
    print(f"T2 2022 ETH 일 종가 panel vs 1분 {len(both)}일 최대 상대 차 {rel:.2e}"); assert len(both) >= 360 and rel < 1e-6
    # T3 학습기 두 구현 일치
    P = expert_perf(E, r, sig)
    for H, m in ((5, 1), (40, 3), (160, 5)):
        a, b = signal_vec(P, E, H, m), signal_loop(P, E, H, m)
        assert np.allclose(a.values, b.values), (H, m)
    print("T3 벡터판 = 루프판 신호 일치 (H5m1·H40m3·H160m5)")
    # T1 누수 탐지: r(d+1) 부호를 섞은 가짜 신호 → 폭증 · 하루 더 늦춤 → 크게 안 오름
    dev = slice(*DEV); s = signal_vec(P, E, 20, 3)
    real = sharpe(strat_pnl(s, r)[dev]); leak = sharpe(strat_pnl(0.5 * s + 0.5 * np.sign(r.shift(-1)).fillna(0), r)[dev])
    lag = sharpe(strat_pnl(s.shift(1).fillna(0), r)[dev])
    print(f"T1 ETH H20m3 샤프 실제 {real:.2f} · 누수판 {leak:.2f} · 하루 지연 {lag:.2f}")
    assert leak > real + 3 and lag < real + 0.5
    print("SELFCHECK PASS")


def main():
    global COST
    ap = argparse.ArgumentParser(); ap.add_argument("--selfcheck", action="store_true"); ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    frames = {s: coin_frame(s) for s in COINS}
    for s, (c, *_) in frames.items():
        print(s, "일봉", c.index[0].date(), "~", c.index[-1].date(), len(c))
    if a.selfcheck:
        return selfcheck(frames)
    R, hold, trend = build(frames)
    dev = slice(*DEV); Rd = R[dev]
    if a.holdout:
        rep = json.loads((OUT / "dev_report.json").read_text())
        assert rep["gates_pass"], "관문 불통과 — 2026 을 열지 않는다(사전 등록)"
        best, k = rep["chosen"], rep["mdd20_scale"]; ho = slice(*HOLD); x = R[best][ho] * k
        out = {"chosen": best, "scale": k, "sharpe": sharpe(x), "mdd": mdd(x), "ret": float((1 + x).prod() - 1),
               "hold_sharpe": sharpe(hold[ho]), "trend_sharpe": sharpe(trend[ho]), "days": len(x)}
        out["pass"] = out["sharpe"] > 0 and out["mdd"] >= -0.20
        (OUT / "holdout_report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False)); print(json.dumps(out, indent=1, ensure_ascii=False)); return
    med, pbo, oos = cpcv(Rd)
    srs = Rd.apply(sharpe); best = srs.idxmax(); x = Rd[best]
    d = dsr(x, list(srs.values)); k = scale_for_mdd(x)
    diff = x - trend[dev]
    H, m, sz = best.split("_"); H = int(H[1:]); m = int(m[1:])
    legs = []   # 2024~2025 펀딩 반영 민감도(선택 구성)
    for s, (c, r, sig, E) in frames.items():
        w = size(signal_vec(expert_perf(E, r, sig), E, H, m), sig, sz)
        f = funding_daily(s).reindex(r.index).fillna(0.0)
        legs.append(strat_pnl(w, r) - w.shift(1).fillna(0) * f)
    xf = pd.concat(legs, axis=1).mean(1)["2024-01-01":DEV[1]]
    cost_sens = {}
    for cbp in (0.0, 2e-4):
        keep, COST = COST, cbp
        cost_sens[f"{cbp*1e4:.0f}bp"] = round(sharpe(build(frames)[0][best][dev]), 3); COST = keep
    rep = {"prereg": "docs/prereg/dir_online_experts_20261025.md@648a1351", "trials": len(GRID), "dev": DEV,
           "cpcv_oos_median_sharpe": med, "cpcv_oos_quantiles": [float(np.percentile(oos, q)) for q in (10, 25, 50, 75, 90)],
           "pbo": pbo, "chosen": best, "chosen_dev_sharpe": float(srs[best]), "dsr": d,
           "gates": {"G1_cpcv_median_gt0": med > 0, "G2_pbo_lt_0.2": pbo < 0.2, "G3_dsr_gt_0.95": d > 0.95},
           "grid_sharpe": {k_: round(v, 3) for k_, v in srs.sort_values(ascending=False).items()},
           "grid_sharpe_median": float(srs.median()),
           "hold_dev_sharpe": sharpe(hold[dev]), "trend_dev_sharpe": sharpe(trend[dev]),
           "chosen_minus_trend_bp_per_day": float(diff.mean() * 1e4), "chosen_minus_trend_ci_bp": block_ci(diff),
           "yearly_sharpe": {"chosen": yearly(x), "hold": yearly(hold[dev]), "trend": yearly(trend[dev])},
           "mdd20_scale": k, "mdd20_ann_ret": float((1 + k * x).prod() ** (365 / len(x)) - 1), "chosen_mdd_unscaled": mdd(x),
           "funding_2024_25_sharpe": {"no_funding": sharpe(x["2024-01-01":]), "with_funding": sharpe(xf)},
           "cost_sensitivity_sharpe": cost_sens}
    rep["gates_pass"] = all(rep["gates"].values())
    (OUT / "dev_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    R.to_parquet(OUT / "grid_daily_pnl.parquet")
    print(json.dumps({k_: v for k_, v in rep.items() if k_ != "grid_sharpe"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
