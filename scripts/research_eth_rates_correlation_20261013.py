"""금리 ↔ ETH 가격 추이·상관 (2026-10-13). 사용자 «금리와 이더리움 가격 추이와 상관관계 분석해줘».
선행: 10-06 국채선물↔ETH 1h·일 동조(2년, 평균 무관 · 2026-03~ 국면 의존 +) — treasury_futures_collector 메모. 여기선 2017~ 긴 추이 + 변화량 상관 + 나스닥 통제 + 선행·지연 + 금리 국면.

    python3 scripts/research_eth_rates_correlation_20261013.py           # FRED·ETH 받기(캐시) → report.json · 차트 png
    python3 scripts/research_eth_rates_correlation_20261013.py --selftest

데이터: FRED 공개 CSV(DGS10 10년 · DGS2 2년 · DFII10 10년 실질(TIPS) · DFF 기준금리 · NASDAQCOM 나스닥 종합) · ETH = 바이낸스 현물 1d(2017-08~, data.binance.vision).
정렬: FRED 날짜 d = 미 장 마감 무렵 값 · ETH 일봉 d 종가 = 24:00 UTC d(=20:00 ET d) → 같은 날 d 끼리 짝. 금리 발표일 사이(주말 포함) ETH 수익은 다음 금리일에 합산.
🔴수준끼리의 상관은 추세가 있는 두 계열이면 거의 항상 크게 나온다(가짜 회귀) → 판단은 변화량(수익률 vs 금리 변화 bp)으로만. 수준 상관은 서술용.
부호: corr(ETH 수익, 금리 변화) < 0 = 금리 오르면 ETH 내림. 최근 우선(주 판정 2025~).
"""
import json, sys, urllib.request
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE.parent / "tmp/eth_rates_20261013"
FRED = ("DGS10", "DGS2", "DFII10", "DFF", "NASDAQCOM")
END = "2026-10-12"
RNG = np.random.default_rng(20261013)


def fred(sid):
    p = OUT / f"fred_{sid}.csv"
    if not p.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        p.write_bytes(urllib.request.urlopen(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=60).read())
    d = pd.read_csv(p)
    return pd.Series(pd.to_numeric(d.iloc[:, 1], errors="coerce").to_numpy(), index=pd.to_datetime(d.iloc[:, 0]), name=sid).dropna()


def eth_close():
    import research_trend4_poi1h_solxrp_20261004 as R
    c = R.eth_spot_closes()                                                   # 월별 2017-08~2026-08
    rows = []
    for d in pd.date_range(c.index[-1] + pd.Timedelta(days=1), END):
        f = R.fetch(f"spot/daily/klines/ETHUSDT/1d/ETHUSDT-1d-{d:%Y-%m-%d}.zip")
        if f is not None:
            k = pd.read_csv(f, header=None)
            t = int(k.iat[0, 0])
            rows.append((pd.to_datetime(t // 1000 if t > 1e14 else t, unit="ms"), float(k.iat[0, 4])))
    if rows:
        c = pd.concat([c, pd.Series(dict(rows))]).sort_index()
    return c[~c.index.duplicated()]


def frame():
    eth = eth_close()
    F = pd.concat([fred(s) for s in FRED], axis=1)
    F = F[F.DGS10.notna()]                                                    # 금리 발표일만
    F["eth"] = eth.reindex(F.index)                                           # 그날 UTC 종가
    F = F[F.index >= eth.index[0]].dropna(subset=["eth"])
    F["slope"] = F.DGS10 - F.DGS2
    return F


def changes(F, rule=None):
    """금리일(또는 주·월말) 간격 변화: ETH·나스닥 로그수익(%) · 금리 변화(bp)."""
    G = F if rule is None else F.resample(rule).last().dropna(subset=["eth", "DGS10"])
    out = pd.DataFrame(index=G.index)
    out["eth"] = np.log(G.eth).diff() * 100
    out["ndx"] = np.log(G.NASDAQCOM).diff() * 100
    for c in ("DGS10", "DGS2", "DFII10", "slope", "DFF"):
        out[f"d{c}"] = G[c].diff() * 100
    return out.dropna(subset=["eth", "dDGS10"])


def block_ci(x, y, block, nb=2000):
    """연속 블록 부트스트랩 상관 95% CI (자기상관·변동성 군집 보존)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]
    n = len(x)
    if n < 3 * block:
        return [np.nan, np.nan]
    k = int(np.ceil(n / block))
    st = RNG.integers(0, n - block + 1, (nb, k))
    idx = (st[:, :, None] + np.arange(block)).reshape(nb, -1)[:, :n]
    xs, ys = x[idx], y[idx]
    xs, ys = xs - xs.mean(1, keepdims=True), ys - ys.mean(1, keepdims=True)
    r = (xs * ys).sum(1) / np.sqrt((xs ** 2).sum(1) * (ys ** 2).sum(1))
    return [round(float(v), 3) for v in np.quantile(r, [.025, .975])]


def resid(a, b):
    """a 에서 b(나스닥) 로 설명되는 몫을 뺀 잔차."""
    m = a.notna() & b.notna()
    beta = np.polyfit(b[m], a[m], 1)[0]
    return a - beta * b


def corr_row(D, x, block):
    m = D[[x, "eth"]].dropna()
    beta = float(np.polyfit(m[x], m.eth, 1)[0]) * 10                          # 금리 +10bp 당 ETH %
    mm = pd.concat([resid(D.eth, D.ndx), resid(D[x], D.ndx)], axis=1).dropna()
    return dict(n=len(m), r=round(float(m.eth.corr(m[x])), 3), ci=block_ci(m[x], m.eth, block), beta_pct_per_10bp=round(beta, 2),
                partial_r_ex_ndx=round(float(mm.iloc[:, 0].corr(mm.iloc[:, 1])), 3), partial_ci=block_ci(mm.iloc[:, 1], mm.iloc[:, 0], block),
                r_ndx_rate=round(float(D[["ndx", x]].dropna().corr().iat[0, 1]), 3))


PERIODS = (("2017-08~", "2017-08-01", END), ("2022~", "2022-01-01", END), ("2025~(주)", "2025-01-01", END), ("2026~", "2026-01-01", END))


def analyses(F):
    rep = {"range": [str(F.index[0].date()), str(F.index[-1].date())], "n_days": len(F)}
    Dd, Dw, Dm = changes(F), changes(F, "W-FRI"), changes(F, "ME")
    win = lambda D, a, b: D[(D.index >= a) & (D.index <= b)]                 # noqa: E731
    # 1) 변화량 상관(일·주·월) × 금리 종류 × 기간
    rep["change_corr"] = {f"{nm}|{x}|{pn}": corr_row(win(D, a, b), x, blk)
                          for nm, D, blk in (("daily", Dd, 20), ("weekly", Dw, 8), ("monthly", Dm, 4))
                          for x in ("dDGS10", "dDGS2", "dDFII10", "dslope") for pn, a, b in PERIODS if len(win(D, a, b)) >= 3 * blk}
    # 2) 연도별(주간)
    rep["yearly_weekly"] = {f"{x}|{y}": corr_row(g, x, 4) for x in ("dDGS10", "dDFII10", "dDGS2") for y, g in Dw.groupby(Dw.index.year) if len(g) >= 30}
    # 3) 선행·지연(주간): corr(금리 변화_w, ETH_{w+k}) — k>0 이면 금리가 먼저
    rep["lead_lag_weekly"] = {pn: {x: {k: dict(r=round(float(m.iloc[:, 0].corr(m.iloc[:, 1])), 3), ci=block_ci(m.iloc[:, 0], m.iloc[:, 1], 8))
                                       for k in range(-4, 5) for m in [pd.concat([s[x], s.eth.shift(-k)], axis=1).dropna()]}
                                   for x in ("dDGS10", "dDFII10")}
                              for pn, a, b in PERIODS for s in [win(Dw, a, b)]}
    # 4) 월간 선행: 이번 달 금리 변화 → 다음 달 ETH
    nm_ = Dm.assign(eth_next=Dm.eth.shift(-1)).dropna(subset=["eth_next"])
    rep["monthly_next"] = {pn: dict(n=int(len(s)), r=round(float(s.dDGS10.corr(s.eth_next)), 3), ci=block_ci(s.dDGS10, s.eth_next, 3),
                                    eth_next_when_up=round(float(s[s.dDGS10 > 0].eth_next.mean()), 2), eth_next_when_down=round(float(s[s.dDGS10 < 0].eth_next.mean()), 2))
                           for pn, a, b in PERIODS for s in [win(nm_, a, b)] if len(s) >= 12}
    # 5) 큰 금리 날(|Δ10Y| ≥ 10bp) 같은 날 ETH
    big = {}
    for pn, a, b in PERIODS:
        s = win(Dd, a, b); up, dn = s[s.dDGS10 >= 10], s[s.dDGS10 <= -10]
        big[pn] = dict(n_up=len(up), eth_up=round(float(up.eth.mean()), 2) if len(up) else None, share_eth_down_on_up=round(float((up.eth < 0).mean()), 3) if len(up) else None,
                       n_dn=len(dn), eth_dn=round(float(dn.eth.mean()), 2) if len(dn) else None, share_eth_up_on_dn=round(float((dn.eth > 0).mean()), 3) if len(dn) else None,
                       eth_all_days=round(float(s.eth.mean()), 3))
    rep["big_rate_days"] = big
    # 6) 기준금리 국면(직전 63영업일 DFF 변화 ≥+25bp 인상기 · ≤−25bp 인하기 · 그 외 동결) — ETH 연율 수익·변동
    dff = F.DFF; ch = dff - dff.shift(63)
    ph = pd.Series(np.where(ch >= 0.25, "인상기", np.where(ch <= -0.25, "인하기", "동결기")), index=F.index)[ch.notna()]
    r = np.log(F.eth).diff().reindex(ph.index)
    grp = (ph != ph.shift()).cumsum()
    rep["fed_phase"] = {p: dict(days=int((ph == p).sum()), eth_ann_pct=round(float(r[ph == p].mean() * 252 * 100), 1),
                                eth_vol_ann_pct=round(float(r[ph == p].std() * np.sqrt(252) * 100), 1),
                                spans=[f"{g.index[0]:%Y-%m}~{g.index[-1]:%Y-%m}" for _, g in ph[ph == p].groupby(grp[ph == p]) if len(g) >= 40])
                        for p in ("인상기", "동결기", "인하기")}
    # 7) 수준 상관(서술용, 가짜 회귀 주의) — 연도별 log ETH vs 금리 수준
    rep["level_corr_by_year"] = {int(y): {x: round(float(np.log(g.eth).corr(g[x])), 2) for x in ("DGS10", "DFII10", "DFF")} for y, g in F.groupby(F.index.year)}
    # 8) 90 영업일 이동 상관(일간) — 그림용
    roll = pd.DataFrame({"ETH↔10년": Dd.eth.rolling(90, min_periods=60).corr(Dd.dDGS10), "ETH↔실질 10년": Dd.eth.rolling(90, min_periods=60).corr(Dd.dDFII10),
                         "나스닥↔10년": Dd.ndx.rolling(90, min_periods=60).corr(Dd.dDGS10)})
    rep["rolling_last"] = {k: round(float(v), 3) for k, v in roll.iloc[-1].items()}
    rep["rolling_share_negative_2025~"] = {k: round(float((roll[k][roll.index >= "2025-01-01"] < 0).mean()), 3) for k in roll}
    return rep, roll


def plot(F, roll, rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for path in ("/home/kbj20/crypto-scalping/tmp/fonts/NotoSansKR-VF.ttf", "/mnt/c/Windows/Fonts/malgun.ttf"):   # 한글 글꼴(WSL 에 시스템 글꼴 없음)
        if Path(path).exists():
            fm.fontManager.addfont(path); plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name(); break
    plt.rcParams["axes.unicode_minus"] = False
    S, T1, T2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
    C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, ax = plt.subplots(3, 1, figsize=(12, 11), sharex=True, gridspec_kw={"height_ratios": [1.1, 1, 1]}, facecolor=S)
    for a in ax:
        a.set_facecolor(S); a.grid(color=GRID, lw=.6); a.tick_params(colors=T2, labelsize=9)
        for sp in a.spines.values():
            sp.set_visible(False)
    ax[0].plot(F.index, F.eth, color=C[0], lw=1.6); ax[0].set_yscale("log")
    tk = [100, 200, 500, 1000, 2000, 5000]
    ax[0].set_yticks(tk, [f"${v:,}" for v in tk]); ax[0].minorticks_off()
    ax[0].set_title("ETH 가격 (달러, 로그 눈금)", loc="left", color=T1, fontsize=11)
    for c, (k, nm) in zip((C[1], "#4a3aa7", C[2], C[3]), (("DGS10", "10년 국채"), ("DGS2", "2년 국채"), ("DFII10", "10년 실질(TIPS)"), ("DFF", "기준금리"))):
        ax[1].plot(F.index, F[k], color=c, lw=1.6, label=nm)
    ax[1].set_title("미국 금리 (%)", loc="left", color=T1, fontsize=11); ax[1].legend(frameon=False, fontsize=9, loc="upper left", labelcolor=T2)
    for c, k in zip((C[1], C[2], "#008300"), roll.columns):                     # 금리 패널과 같은 색 = 같은 금리(10년 주황 · 실질 청록) · 나스닥 = 초록
        ax[2].plot(roll.index, roll[k], color=c, lw=1.6, label=k)
    ax[2].axhline(0, color=T2, lw=.8)
    ax[2].set_title("90영업일 이동 상관: 일간 수익 vs 금리 변화 (음수 = 금리 오를 때 하락)", loc="left", color=T1, fontsize=11)
    ax[2].legend(frameon=False, fontsize=9, loc="lower left", labelcolor=T2, ncol=3); ax[2].set_ylim(-0.8, 0.6)
    ax[2].axvspan(pd.Timestamp("2025-01-01"), F.index[-1], color=GRID, alpha=.5, lw=0)
    fig.text(0.01, 0.005, f"자료: FRED(DGS10·DGS2·DFII10·DFF·NASDAQCOM), 바이낸스 ETHUSDT 현물 일봉 · {rep['range'][0]}~{rep['range'][1]} · 회색 = 주 판정 구간(2025~)", color=T2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT / "eth_rates_trend_corr.png", dpi=130, facecolor=S)
    yrs = sorted({int(k.split("|")[1]) for k in rep["yearly_weekly"] if k.startswith("dDGS10")})
    Y = [rep["yearly_weekly"][f"dDGS10|{y}"] for y in yrs]
    raw, par = np.array([y["r"] for y in Y]), np.array([y["partial_r_ex_ndx"] for y in Y])
    lo, hi = np.array([y["ci"][0] for y in Y]), np.array([y["ci"][1] for y in Y])
    fig, a = plt.subplots(figsize=(12, 4.2), facecolor=S); a.set_facecolor(S)
    x = np.arange(len(yrs)); w = .38
    a.bar(x - w / 2 - .01, raw, w, color=C[0], label="원 상관", yerr=[raw - lo, hi - raw], ecolor=T2, capsize=2, error_kw={"lw": .8})
    a.bar(x + w / 2 + .01, par, w, color=C[1], label="나스닥 몫 뺀 상관")
    a.axhline(0, color=T2, lw=.8); a.set_xticks(x, [str(y) for y in yrs]); a.grid(axis="y", color=GRID, lw=.6)
    for sp in a.spines.values():
        sp.set_visible(False)
    a.tick_params(colors=T2, labelsize=9)
    a.set_title("연도별 주간 상관: ETH 주간 수익 vs 10년 금리 주간 변화 (선 = 95% 범위)", loc="left", color=T1, fontsize=11)
    a.legend(frameon=False, fontsize=9, labelcolor=T2)
    fig.tight_layout(); fig.savefig(OUT / "eth_rates_yearly_corr.png", dpi=130, facecolor=S)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    F = frame()
    rep, roll = analyses(F)
    json.dump(rep, open(OUT / "report.json", "w"), ensure_ascii=False, indent=1, default=float)
    plot(F, roll, rep)
    print("saved", OUT, rep["range"])


def selftest():
    x = RNG.normal(size=2000); y = -0.5 * x + RNG.normal(size=2000)
    lo, hi = block_ci(x, y, 20)
    assert lo < np.corrcoef(x, y)[0, 1] < hi and hi < 0
    assert np.allclose(resid(pd.Series(np.arange(10.0)), pd.Series(np.arange(10.0) * 2)), 0)
    F = pd.DataFrame({"eth": [100, 110, 99.0], "NASDAQCOM": [1, 1, 1.0], "DGS10": [4.0, 4.1, 4.0], "DGS2": [3.0] * 3, "DFII10": [2.0] * 3,
                      "slope": [1.0, 1.1, 1.0], "DFF": [5.0] * 3}, index=pd.date_range("2025-01-01", periods=3))
    D = changes(F)
    assert np.allclose(D.dDGS10, [10, -10]) and np.allclose(D.eth, np.log([1.1, 0.9]) * 100)
    print("selftest OK")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
