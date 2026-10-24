#!/usr/bin/env python3
"""다주 추세 포트폴리오 — 문헌 기반 개선 3회차 사전 등록 docs/prereg/trend_portfolio_lit_20261025.md.

    python3 scripts/research_trend_portfolio_lit_20261025.py --selfcheck        # S1 보유 일치 · S2 누수/지연 · S3 10-25 엔진 파리티
    python3 scripts/research_trend_portfolio_lit_20261025.py --round 1          # 개발 구간(2022~2025) CPCV·PBO·DSR·관문
    python3 scripts/research_trend_portfolio_lit_20261025.py --round 1 --holdout   # 개발 관문 통과 때만 2026 1회 채점

일 단위 엔진: 일 d 00:00 UTC 에 일 d−1 종가까지로 목표 비중 → 지정가 P0(전날 종가), 60분(5분봉 12개) 안 1bp 관통이면 체결·수수료 0,
아니면 12번째 봉 종가에 테이커 4bp(10-25 시스템 스크립트와 같은 체결 모형) · 실제 펀딩 · 순자산 공유 4코인.
"""
import argparse, glob, importlib.util, io, itertools, json, math, pathlib, sys, zipfile
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = pathlib.Path("/home/kbj20/crypto-scalping/data")
OUT = ROOT / "tmp" / "trend_portfolio_lit_20261025"


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fn); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


SYS = _load("sys1025", "research_system_trend_ruleb_20261025.py")      # bars5 · rest_funding · simulate(파리티)
OE = _load("oe1025", "research_dir_online_experts_20261025.py")        # cpcv · dsr · block_ci · mdd
COINS = SYS.COINS; N = len(COINS); THRU, TAKER, GROSS = 1e-4, 4e-4, 2.0
DEV = ("2022-01-01", "2025-12-31"); HOLD = ("2026-01-01", "2026-10-23"); PRE = ("2020-04-01", "2021-12-31")
TAU_RANK = 0.4; TAUS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8)
LOOK = {"A": (7, 14, 21, 28), "B": (7, 14, 28, 56, 90), "C": (7, 14, 21, 28, 42, 56, 70, 84), "L": (120, 180, 270, 365)}
sharpe = OE.sharpe


# ---------------- 데이터 ----------------
def bars_pre2022(sym):
    """2020~2021 1분 zip → 5분 h,l,c (예열·견고성 전용)."""
    xs = []
    for f in sorted(glob.glob(str(D / f"binance_vision/klines1m_futures_pre2022/{sym}USDT-1m-*.zip"))):
        with zipfile.ZipFile(f) as z:
            x = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), header=None, usecols=[0, 2, 3, 4])
        if not str(x.iat[0, 0]).isdigit():
            x = x.iloc[1:].astype(float)
        xs.append(x)
    x = pd.concat(xs); x.index = pd.to_datetime(x[0].astype("int64"), unit="ms"); x = x[~x.index.duplicated()].sort_index()
    return x.resample("5min").agg({2: "max", 3: "min", 4: "last"}).set_axis(["h", "l", "c"], axis=1)


def daily_frame(sym):
    """일 d 행: c(23:55 봉 종가)·h·l·rv(5분 로그수익 제곱합)·lo12/hi12/c11(첫 60분, 체결 모형) · 펀딩 pre(00시 정산)/post."""
    x, miss = SYS.bars5(sym)
    x = pd.concat([bars_pre2022(sym), x]); x = x[~x.index.duplicated(keep="last")].sort_index()
    idx = pd.date_range(x.index[0].normalize(), pd.Timestamp(SYS.END) + pd.Timedelta("23:55:00"), freq="5min"); x = x.reindex(idx)
    first = x.c.first_valid_index(); x = x[first.normalize():]
    x["c"] = x.c.ffill(); x["h"] = x.h.fillna(x.c); x["l"] = x.l.fillna(x.c)
    days = pd.date_range(x.index[0], periods=len(x) // 288, freq="D")
    H, L, C = (x[k].values.reshape(-1, 288) for k in ("h", "l", "c"))
    lr = np.diff(np.log(x.c.values), prepend=np.nan).reshape(-1, 288)
    f = pd.DataFrame({"c": C[:, -1], "h": H.max(1), "l": L.min(1), "rv": np.nansum(lr ** 2, 1),
                      "lo12": L[:, :12].min(1), "hi12": H[:, :12].max(1), "c11": C[:, 11]}, days)
    if sym == "SOL":
        f.loc[: "2020-09-14"] = np.nan   # SOL 선물 상장 초기(빈 봉 채움) 제외
    fu = SYS.rest_funding(sym, 0, 0); ft = pd.to_datetime(fu.t, unit="ms")
    fu = pd.DataFrame({"d": ft.dt.normalize(), "pre": np.where(ft.dt.hour < 1, fu.rate, 0.0), "post": np.where(ft.dt.hour >= 1, fu.rate, 0.0)})
    g = fu.groupby("d")[["pre", "post"]].sum()
    f[["fpre", "fpost"]] = g.reindex(f.index).fillna(0.0).values
    return f[: SYS.END], miss


# ---------------- 신호·변동성 (일 d 종가까지 → 다음 날 결정에 shift(1)) ----------------
def signal(c, kind, look):
    if look == "E":   # Hurst 식 지평 분산: A·B·C 세 묶음 신호의 평균(묶음 선택 안 함)
        return sum(signal(c, kind, l) for l in "ABC") / 3
    if look == "F":   # E + Hurst·MOP 4~12개월 묶음 L (3회차)
        return sum(signal(c, kind, l) for l in "ABCL") / 4
    lr = np.log(c).diff(); out = []
    for k in LOOK[look]:
        if kind == "sign":
            out.append(np.sign(c / c.shift(k) - 1))
        else:   # Baltas·Kosowski TREND: k일 일 로그수익 t 통계, ±1 절단
            t = lr.rolling(k).mean() / (lr.rolling(k).std(ddof=0) / math.sqrt(k))
            out.append(t.clip(-1, 1))
    return sum(out) / len(out)


def vol(f, kind):
    lr = np.log(f.c).diff()
    if kind == "cc20":
        v = lr.rolling(20).std() ** 2
    elif kind == "ewma60":   # MOP 2012: 질량 중심 60일 EWMA, 평균 제거
        m = lr.ewm(com=60).mean(); v = ((lr - m) ** 2).ewm(com=60).mean()
    elif kind == "pk20":     # Parkinson
        v = (np.log(f.h / f.l) ** 2 / (4 * math.log(2))).rolling(20).mean()
    elif kind == "rv20":     # 5분 실현분산
        v = f.rv.rolling(20).mean()
    return np.sqrt(v * 365)


def targets(F, cfg, tau):
    """cfg = dict(look, sig, vol, port, crash). 반환 = 일 × 코인 목표 비중(순자산 대비 명목) — 일 d 행은 d−1 종가까지만."""
    S = pd.DataFrame({s: signal(F[s].c, cfg["sig"], cfg["look"]) for s in COINS})
    V = pd.DataFrame({s: vol(F[s], cfg["vol"]) for s in COINS})
    R = pd.DataFrame({s: F[s].c.pct_change() for s in COINS})
    S = S.where(S >= 0, S * cfg.get("short", 1.0))  # 숏 쪽 배율(2회차)
    U = S / V / N                                   # τ=1 단위 책
    if cfg["port"] == "cf":                          # Baltas·Kosowski 상관 계수, 부호 상관 90일
        cf = []
        for i in range(len(R)):
            if i < 90:
                cf.append(np.nan); continue
            rho = R.iloc[i - 89:i + 1].corr().values; x = S.iloc[i].fillna(0).values
            rb = sum(x[a] * x[b] * rho[a, b] for a in range(N) for b in range(a + 1, N)) * 2 / (N * (N - 1))
            cf.append(math.sqrt(N / max(1 + (N - 1) * rb, 0.1)))
        U = U.mul(pd.Series(cf, R.index), axis=0)
    elif cfg["port"] == "pv":                        # Harvey 외 2018: 책 자체 실현 변동성(60일)으로 목표 맞춤
        book = (U.shift(1) * R).sum(1, min_count=1)
        U = U.div(book.rolling(60).std() * math.sqrt(365), axis=0)
    if cfg.get("crash") == "dm":                     # Daniel·Moskowitz 공황 상태: 지수 1년 수익 < 0 & 지수 σ20 > 직전 1년 중앙 → ×0.5
        ix = R.mean(1, skipna=True); lvl = (1 + ix.fillna(0)).cumprod()
        sv = ix.rolling(20).std(); panic = (lvl / lvl.shift(365) - 1 < 0) & (sv > sv.rolling(365).median())
        U = U.mul(np.where(panic, 0.5, 1.0), axis=0)
    W = (tau * U).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    g = W.abs().sum(1); W = W.mul(np.minimum(1.0, GROSS / g.where(g > 0, 1.0)), axis=0)
    return W.shift(1).fillna(0.0)


# ---------------- 엔진 ----------------
def run(F, W, days, band=0.0, fill="maker", fees=True, funding=True):
    """일별 포트폴리오 수익률. band = 상대 재조정 띠(같은 부호이고 |목표−현재| ≤ band·|목표| 면 안 바꿈, band=inf 면 부호 바뀔 때만)."""
    A = {s: F[s].reindex(days) for s in COINS}; Wd = W.reindex(days).fillna(0.0); Nn = np.zeros(N); out = []
    cols = {s: {k: A[s][k].values for k in ("c", "lo12", "hi12", "c11", "fpre", "fpost")} for s in COINS}
    P0s = {s: F[s].c.shift(1).reindex(days).values for s in COINS}
    for t in range(len(days)):
        pnl = 0.0; newv = np.zeros(N)
        for i, s in enumerate(COINS):
            a = cols[s]; P0 = P0s[s][t]; c = a["c"][t]; cur = Nn[i]; tgt = Wd.iat[t, i]
            if np.isnan(P0) or np.isnan(c):
                newv[i] = 0.0; continue
            if band > 0 and np.sign(cur) == np.sign(tgt) and abs(tgt - cur) <= band * abs(tgt):
                tgt = cur
            dq = tgt - cur                              # 결정 = P0 기준 명목 차 · 체결 후 수량 = 목표 명목 / 체결가
            if funding:
                pnl -= cur * a["fpre"][t]
            q0 = cur / P0
            if dq == 0:
                q1, px = q0, P0
            else:
                if fill == "ideal":
                    px, fee = P0, 0.0
                elif fill == "taker":
                    px, fee = P0, TAKER
                else:
                    hit = a["lo12"][t] <= P0 * (1 - THRU) if dq > 0 else a["hi12"][t] >= P0 * (1 + THRU)
                    px, fee = (P0, 0.0) if hit else (a["c11"][t], TAKER)
                q1 = tgt / px
                pnl -= fee * abs(q1 - q0) * px if fees else 0.0
            pnl += q0 * (px - P0) + q1 * (c - px); v = q1 * c
            if funding:
                pnl -= tgt * a["fpost"][t]
            newv[i] = v
        Nn = newv / (1 + pnl); out.append(pnl)
    return pd.Series(out, days)


def turnover(W, days):
    return float(W.loc[days].diff().abs().sum(1).mean())


def markov_signs(S, days, seed):
    """S 부호(−1/0/+1) 4코인 풀 전이 행렬로 무작위 부호 경로 — 같은 보유 길이, 시점만 무작위."""
    T = np.zeros((3, 3))
    for s in COINS:
        v = np.sign(S[s].loc[days]).astype(int).values
        for a_, b_ in zip(v[:-1], v[1:]):
            T[a_ + 1, b_ + 1] += 1
    empty = T.sum(1) == 0; T[empty] = T.sum(0)                 # 한 번도 안 나온 상태(연속 신호의 0)는 전체 상태 비율로 — 1회차 수치 영향 없음
    T = T / T.sum(1, keepdims=True); rng = np.random.default_rng(seed); out = {}
    for s in COINS:
        cur = rng.choice(3); v = []
        for _ in days:
            v.append(cur - 1); cur = rng.choice(3, p=T[cur])
        out[s] = pd.Series(v, days)
    return pd.DataFrame(out)


def null_targets(F, cfg, tau, days, seed):
    """신호 크기는 그대로 두고 부호만 무작위(크기 0 인 날은 그 코인 비0 크기 중앙값)."""
    W = targets(F, cfg, tau).loc[days]; rs = markov_signs(W, days, seed)
    mag = W.abs(); med = mag.where(mag > 0).median()
    return (rs * mag.where(mag > 0, med, axis=1)).reindex(W.index)


# ---------------- 격자(회차별, 사전 등록) ----------------
def grid(rnd):
    if rnd == 1:
        return [dict(look=l, sig=g, vol=v, port=p, crash="none", band=0.0)
                for l in "ABC" for g in ("sign", "trend") for v in ("cc20", "ewma60", "pk20", "rv20") for p in ("none", "cf", "pv")]
    if rnd == 2:
        return [dict(look=l, sig="sign", vol=v, port="pv", crash=cr, short=sh, band=b)
                for l in "BE" for v in ("pk20", "rv20") for cr in ("none", "dm") for sh in (1.0, 0.5) for b in (0.0, 0.25)]
    if rnd == 3:
        return [dict(look=l, sig="sign", vol="rv20", port="pv", crash="dm", short=sh, band=0.25) for l in "EF" for sh in (0.5, 0.0)]
    raise SystemExit(f"회차 {rnd} 격자는 아직 사전 등록되지 않았다")


name = lambda c: f"{c['look']}_{c['sig']}_{c['vol']}_{c['port']}_{c['crash']}_b{c['band']}" + (f"_s{c['short']}" if "short" in c else "")
CURRENT = dict(look="A", sig="sign", vol="cc20", port="none", crash="none", band=0.0)   # 검증된 규칙: 0.5/σ20 최대 2배 ÷4


def current_rule(F):
    S = pd.DataFrame({s: signal(F[s].c, "sign", "A") for s in COINS}); V = pd.DataFrame({s: vol(F[s], "cc20") for s in COINS})
    return (S * np.minimum(2.0, 0.5 / V) / N).fillna(0.0).shift(1).fillna(0.0)


def hold_ret(F, days):
    return pd.DataFrame({s: F[s].c.pct_change() for s in COINS}).loc[days].mean(1)


def yearly(r):
    return {str(y): float((1 + v).prod() - 1) for y, v in r.groupby(r.index.year)}


def pick_tau(F, cfg, days):
    best = None
    for tau in TAUS:
        if OE.mdd(run(F, targets(F, cfg, tau), days, band=cfg["band"])) >= -0.20:
            best = tau
    return best


# ---------------- 실행 ----------------
def selfcheck(F):
    days = pd.date_range("2022-02-01", SYS.END, freq="D")
    W1 = pd.DataFrame({s: 1.0 if s == "ETH" else 0.0 for s in COINS}, index=F["ETH"].index)
    r = run(F, W1, days, fill="ideal", fees=False, funding=False); c = F["ETH"].c
    hold = c[SYS.END] / c[days[0] - pd.Timedelta("1D")]
    print(f"S1 항상 ETH 롱 1배 {float((1 + r).prod()):.12f} vs 보유 {hold:.12f}"); assert abs((1 + r).prod() - hold) < 1e-9
    cfg = dict(CURRENT); W = targets(F, cfg, TAU_RANK); dv = pd.date_range(*DEV, freq="D")
    base = sharpe(run(F, W, dv))
    Rn = pd.DataFrame({s: np.sign(F[s].c.pct_change()) for s in COINS}).fillna(0.0) * W.abs().replace(0, 0.1)   # 일 d 자신의 수익 부호
    leak = sharpe(run(F, Rn, dv)); lag = sharpe(run(F, W.shift(1).fillna(0.0), dv))
    print(f"S2 개발 샤프 실제 {base:.2f} · 누수판 {leak:.2f} · 하루 지연 {lag:.2f}"); assert leak > base + 2 and lag < base + 0.5
    # S3 독립 엔진(10-25 5분 순차 simulate) 파리티: 사다리·손절·펀딩 끔, 1/4 고정 크기, 방향 = 4기간 부호 평균의 부호, 부호 바뀔 때만 거래
    data, _, inputs, fund = SYS.load_all()
    Dser = {s: inputs[s].D for s in COINS}
    r_sys = SYS.simulate(data, inputs, fund, days, N, Dover=Dser, ladder=False, stop=False, funding=False, L_fixed=1.0)
    Wd = pd.DataFrame({s: np.sign(signal(F[s].c, "sign", "A")).shift(1) / N for s in COINS}).fillna(0.0)
    r_me = run(F, Wd, days, band=np.inf, funding=False)
    gap = float(((1 + r_sys).cumprod() / (1 + r_me).cumprod() - 1).abs().max())
    print(f"S3 10-25 엔진 vs 일 엔진 누적 최대 상대 차 {gap:.2e}"); assert gap < 1e-9
    print("SELFCHECK PASS")


def dev_round(F, rnd):
    dv = pd.date_range(*DEV, freq="D"); OUT.mkdir(parents=True, exist_ok=True)
    G = grid(rnd); cols = {}
    for c in G:
        cols[name(c)] = run(F, targets(F, c, TAU_RANK), dv, band=c["band"])
    R = pd.DataFrame(cols); R.to_parquet(OUT / f"round{rnd}_dev_daily.parquet")
    trials_f = OUT / "trials.json"; trials = json.loads(trials_f.read_text()) if trials_f.exists() else {}
    trials[f"round{rnd}"] = {k: sharpe(v) for k, v in R.items()}; trials_f.write_text(json.dumps(trials, indent=1))
    all_srs = [v for r_ in trials.values() for v in r_.values()]
    med, pbo, oos = OE.cpcv(R, emb=10)
    Rall = pd.concat([pd.read_parquet(OUT / f"round{i}_dev_daily.parquet").add_prefix(f"r{i}:") for i in range(1, rnd + 1)], axis=1)
    med_cum, pbo_cum, _ = OE.cpcv(Rall, emb=10)
    srs = R.apply(sharpe); best = srs.idxmax(); cfg = next(c for c in G if name(c) == best)
    tau = pick_tau(F, cfg, dv) or TAUS[0]; W = targets(F, cfg, tau); x = run(F, W, dv, band=cfg["band"])
    d = OE.dsr(x, all_srs); hold = hold_ret(F, dv); cur = run(F, current_rule(F), dv)
    nulls = [sharpe(run(F, null_targets(F, cfg, tau, dv, sd), dv, band=cfg["band"])) for sd in range(200)]
    sysd = pd.read_csv(ROOT / "tmp/system_trend_ruleb_20261025/system_daily.csv", index_col=0, parse_dates=True).iloc[:, 0][: DEV[1]]
    yr = yearly(x)
    rep = {"prereg": "docs/prereg/trend_portfolio_lit_20261025.md", "round": rnd, "trials_round": len(G), "trials_total": len(all_srs),
           "cpcv_oos_median_sharpe": med, "cpcv_oos_quantiles": [float(np.percentile(oos, q)) for q in (10, 25, 50, 75, 90)], "pbo": pbo, "pbo_cumulative": pbo_cum, "cpcv_oos_median_cumulative": med_cum, "trials_cumulative_pbo": Rall.shape[1],
           "chosen": best, "chosen_rank_sharpe": float(srs[best]), "tau": tau, "dev_sharpe": sharpe(x), "dev_mdd": OE.mdd(x),
           "dev_ann_ret": float((1 + x).prod() ** (365 / len(x)) - 1), "dsr": d, "yearly_ret": yr, "turnover_per_day": turnover(W, dv),
           "null_p95": float(np.percentile(nulls, 95)), "null_median": float(np.median(nulls)),
           "hold_sharpe": sharpe(hold), "hold_mdd": OE.mdd(hold), "hold_yearly": yearly(hold),
           "current_rule_sharpe": sharpe(cur), "current_rule_mdd": OE.mdd(cur), "current_rule_yearly": yearly(cur),
           "sys1025_dev_sharpe": sharpe(sysd[DEV[0]:]), "sys1025_dev_mdd": OE.mdd(sysd[DEV[0]:]),
           "chosen_minus_current_bp_ci": OE.block_ci(x - cur), "chosen_minus_current_bp": float((x - cur).mean() * 1e4),
           "chosen_minus_current_volmatched_bp_ci": OE.block_ci(x - cur * x.std() / cur.std()),
           "grid_sharpe_median": float(srs.median()), "grid_sharpe": {k: round(v, 3) for k, v in srs.sort_values(ascending=False).items()}}
    rep["gates"] = {"D1_cpcv_med_gt0": med > 0, "D2_pbo_lt_0.2": (pbo_cum if rnd >= 3 else pbo) < 0.2, "D3_dsr_gt_0.95": d > 0.95,
                    "D4_gt_hold": rep["dev_sharpe"] > rep["hold_sharpe"], "D5_mdd": rep["dev_mdd"] >= -0.20,
                    "D6_null_p95": rep["dev_sharpe"] > rep["null_p95"], "D7_years": sum(v > 0 for v in yr.values()) >= 3}
    rep["dev_pass"] = all(rep["gates"].values())
    # 견고성(서술, 관문 아님): 2020-04~2021 · 테이커 · 펀딩 없음
    pre = pd.date_range(*PRE, freq="D")
    xp = run(F, targets(F, cfg, tau), pre, band=cfg["band"], fill="taker", funding=False); hp = hold_ret(F, pre).fillna(0)
    cp = run(F, current_rule(F), pre, fill="taker", funding=False)
    rep["robust_2020_21"] = {"sharpe": sharpe(xp), "mdd": OE.mdd(xp), "hold_sharpe": sharpe(hp), "current_rule_sharpe": sharpe(cp)}
    (OUT / f"round{rnd}_dev_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rep.items() if k != "grid_sharpe"}, indent=1, ensure_ascii=False))
    print("상위 10:", list(rep["grid_sharpe"].items())[:10])


def holdout(F, rnd):
    rep = json.loads((OUT / f"round{rnd}_dev_report.json").read_text())
    assert rep["dev_pass"], "개발 관문 불통과 — 2026 을 열지 않는다(사전 등록)"
    hf = OUT / f"round{rnd}_holdout_report.json"; assert not hf.exists(), "2026 은 회차마다 1회만 채점"
    cfg = next(c for c in grid(rnd) if name(c) == rep["chosen"]); hd = pd.date_range(*HOLD, freq="D")
    x = run(F, targets(F, cfg, rep["tau"]), hd, band=cfg["band"]); h = hold_ret(F, hd); cur = run(F, current_rule(F), hd)
    out = {"chosen": rep["chosen"], "tau": rep["tau"], "sharpe": sharpe(x), "ret": float((1 + x).prod() - 1), "mdd": OE.mdd(x),
           "hold_sharpe": sharpe(h), "hold_ret": float((1 + h).prod() - 1), "current_rule_sharpe": sharpe(cur), "current_rule_ret": float((1 + cur).prod() - 1)}
    out["gates"] = {"H1_gt_hold": out["sharpe"] > out["hold_sharpe"], "H2_mdd": out["mdd"] >= -0.20, "H3_ret_gt0": out["ret"] > 0}
    out["pass"] = all(out["gates"].values()); hf.write_text(json.dumps(out, indent=1, ensure_ascii=False)); print(json.dumps(out, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selfcheck", action="store_true"); ap.add_argument("--round", type=int); ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    F = {}
    for s in COINS:
        F[s], miss = daily_frame(s); print(s, F[s].index[0].date(), "~", F[s].index[-1].date(), "2022~ 빈 5분봉", miss)
    if a.selfcheck:
        return selfcheck(F)
    return holdout(F, a.round) if a.holdout else dev_round(F, a.round)


if __name__ == "__main__":
    sys.exit(main())
