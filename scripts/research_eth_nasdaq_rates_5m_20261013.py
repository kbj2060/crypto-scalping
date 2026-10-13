"""나스닥 ↔ ETH 상관 + «금리 발작» 때 ETH 5분봉 반응 (2026-10-13).
사용자 «나스닥과 이더리움 상관관계도 분석해줘. 그리고 요즘에는 금리가 발작하면 이더리움 가격도 급변하던데 타임프레임을 일이 아니라 5분봉으로 봐줘».

    python3 scripts/research_eth_nasdaq_rates_5m_20261013.py --fetch      # Dukascopy 1분봉(나스닥100 CFD · 미 30년 국채 CFD) 2024-01~ · Yahoo 5분(NQ·ZN·ZB) 최근 60일
    python3 scripts/research_eth_nasdaq_rates_5m_20261013.py              # report.json · 그림
    python3 scripts/research_eth_nasdaq_rates_5m_20261013.py --selftest

자료
- Dukascopy datafeed `{INS}/{YYYY}/{MM-1}/{DD}/BID_candles_min_1.bi5`(LZMA, 24바이트 >i4×5+f4: 초·시가·종가·저가·고가·거래량, 가격 ÷1000, UTC).
  USATECHIDXUSD = 나스닥100 CFD(선물 추종, 거의 24h) · USTBONDTRUSD = 미 30년 국채 선물 CFD(가격 ↓ = 금리 ↑). 10년물 CFD 는 없다 → 최근 60일 Yahoo ZN(10년)·ZB(30년) 5분으로 교차 확인.
- ETH = 바이낸스 USDT-M 1분봉(→5분) 2022-01~10-11 · 일 장기 = 현물 일봉 2017~ · FRED NASDAQCOM.
사전 고정(결과 보기 전)
- 5분 수익 = 봉 끝 종가 로그 차(bp). 비교는 두 종목 다 «살아 있는» 봉만(CFD 거래량 > 0).
- 금리 발작 = 30년물 5분 |수익| ≥ 4 × 직전 20거래일 같은 종목 5분 수익 표준편차(현재 봉 제외). 방향 s = −sign(채권 수익) (+1 = 금리↑).
  기준선 = 같은 «요일·UTC 5분 칸»의 발작 아닌 봉(시간대 변동성 맞춤). 구간: 예전 2024-01~2026-06 · 요즘 2026-07~10-11.
  보는 것: ①같은 봉 |ETH| 배수(발작/기준선) ②s×ETH(금리↑ 에 ETH↓ 면 음수) 같은 봉·이후 1h/2h 누적 ③나스닥 통제 회귀 ETH ~ NQ + 채권(발작 봉).
- 나스닥↔ETH: 일·주(2017~ FRED, ETH = UTC 종가 · 2024~ 는 나스닥 마감 16:00 ET 시각의 ETH 가격) · 5분(2024~) 월별·세션별(정규장 09:30–16:00 ET / 그 외)·큰 하락/상승 봉·선행 지연(±3봉).
🔴Yahoo·Dukascopy 원본은 tmp/eth_macro5m_20261013/ 에만, 해석만 한다(실행 안 함).
플래그: fresh_forward_bar_by_bar=true · trade_ledgers_used_as_input=false · future_rows_used_for_entry=false
"""
import concurrent.futures as cf
import json, lzma, sys, time, urllib.request
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
OUT = ROOT / "tmp/eth_macro5m_20261013"
DUKA = ("USATECHIDXUSD", "USTBONDTRUSD")
D0, END = "2024-01-01", "2026-10-11"                   # Dukascopy 응답이 파일당 6~18초라 2024~ 로 줄임(10-13)
RECENT = "2026-07-01"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
M5 = 300_000


# ── 받기 ────────────────────────────────────────────────────────────────
def _duka_one(ins, d):
    p = OUT / "duka" / ins / f"{d:%Y-%m-%d}.bi5"
    if p.exists():
        return 0
    url = f"https://datafeed.dukascopy.com/datafeed/{ins}/{d.year}/{d.month - 1:02d}/{d.day:02d}/BID_candles_min_1.bi5"
    for k in range(6):
        try:
            b = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60).read()
            if b[:5] == b"<html":
                raise IOError("html")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b)
            return 1
        except Exception:  # noqa: BLE001 -- 503 · 끊김 → 쉬었다 다시
            time.sleep(2 + 3 * k)
    return -1


def fetch():
    jobs = [(i, d) for d in pd.date_range(D0, END)[::-1] if d.dayofweek != 5 for i in DUKA]   # 토요일 = 장 없음 · 최근부터
    with cf.ThreadPoolExecutor(24) as ex:                                  # 서버 응답이 느릴 뿐(파일당 6~18초) → 동시 요청으로 메움
        r = list(ex.map(lambda j: _duka_one(*j), jobs))
    print("Dukascopy 받음", r.count(1), "있음", r.count(0), "실패", r.count(-1))
    for s in ("NQ=F", "ZN=F", "ZB=F"):
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{s.replace('=', '%3D')}?interval=5m&range=60d"
        (OUT / f"yahoo_{s[:2]}_5m.json").write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60).read())
    print("Yahoo 3종 받음")


# ── 적재 ────────────────────────────────────────────────────────────────
def decode_bi5(b, t0_ms):
    r = lzma.decompress(b)
    a = np.frombuffer(r, dtype=">i4").reshape(-1, 6).astype(np.int64)
    v = np.frombuffer(r, dtype=">f4").reshape(-1, 6)[:, 5].astype(float)
    return pd.DataFrame({"t": t0_ms + a[:, 0] * 1000, "c": a[:, 2] / 1000.0, "v": v})


def duka_1m(ins):
    rows = [decode_bi5(p.read_bytes(), int(pd.Timestamp(p.stem).value // 10**6)) for p in sorted((OUT / "duka" / ins).glob("*.bi5")) if p.stat().st_size]
    return pd.concat(rows, ignore_index=True)


def to5(k):
    """1분 → 5분: 종가 = 그 5분 마지막 분 종가, 거래량 합 · 행 t = 5분봉 시작(ms)."""
    g = k.groupby(k.t // M5 * M5)
    return pd.DataFrame({"c": g.c.last(), "v": g.v.sum()})


def yahoo_5m(nm):
    d = json.load(open(OUT / f"yahoo_{nm}_5m.json"))["chart"]["result"][0]
    q = d["indicators"]["quote"][0]
    s = pd.DataFrame({"c": q["close"], "v": q["volume"]}, index=np.array(d["timestamp"], dtype=np.int64) * 1000).dropna(subset=["c"])
    return s[~s.index.duplicated(keep="last")]


def eth_5m():
    import research_eth_liq_wall_break_conditions_20261013 as R
    R.DL, R.END = str(ROOT / "tmp/interim_20261013/new/k1m"), "2026-10-12"   # 09-15~10-11 일별 zip(중간 열람 때 받은 것)
    k = R.load_1m()
    return to5(k[k.t >= pd.Timestamp("2022-01-01").value // 10**6])


def panel():
    E = eth_5m().rename(columns={"c": "eth", "v": "eth_v"})
    N = to5(duka_1m("USATECHIDXUSD")).rename(columns={"c": "nq", "v": "nq_v"})
    B = to5(duka_1m("USTBONDTRUSD")).rename(columns={"c": "bond", "v": "bond_v"})
    P = E.join(N, how="left").join(B, how="left")
    P = P[P.index >= pd.Timestamp(D0).value // 10**6]
    for c in ("eth", "nq", "bond"):
        P[f"r_{c}"] = np.log(P[c]).diff() * 1e4
    P["nq_live"], P["bond_live"] = P.nq_v > 0, P.bond_v > 0
    ts = pd.to_datetime(P.index, unit="ms", utc=True)
    et = ts.tz_convert("America/New_York")
    P["ts"], P["et_min"] = ts, et.hour * 60 + et.minute
    P["cash"] = (P.et_min >= 570) & (P.et_min < 960) & (et.dayofweek < 5)
    P["slot"] = ts.dayofweek * 288 + ts.hour * 12 + ts.minute // 5
    P["day"] = ts.floor("D")
    return P


# ── 통계 도구 ───────────────────────────────────────────────────────────
def day_boot_mean(x, day, nb=2000, seed=7):
    d = pd.DataFrame({"x": x, "d": day}).dropna().groupby("d").x.agg(["sum", "count"])
    if len(d) < 10:
        return [np.nan, np.nan]
    idx = np.random.default_rng(seed).integers(0, len(d), (nb, len(d)))
    s, c = d["sum"].to_numpy()[idx].sum(1), d["count"].to_numpy()[idx].sum(1)
    return [round(float(v), 2) for v in np.quantile(s / c, [.025, .975])]


def corr_ci(x, y, day, nb=1000, seed=7):
    """일(또는 주어진 묶음) 블록 부트스트랩 상관 CI."""
    D = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float), "d": np.asarray(day)}).dropna()
    if len(D) < 50:
        return np.nan, [np.nan, np.nan]
    g = D.assign(xy=D.x * D.y, xx=D.x ** 2, yy=D.y ** 2).groupby("d")[["x", "y", "xy", "xx", "yy"]].sum()
    A = np.c_[g.to_numpy(), D.groupby("d").size().to_numpy()]
    S = A[np.random.default_rng(seed).integers(0, len(A), (nb, len(A)))].sum(1)
    n = S[:, 5]
    r = (S[:, 2] / n - S[:, 0] / n * S[:, 1] / n) / np.sqrt((S[:, 3] / n - (S[:, 0] / n) ** 2) * (S[:, 4] / n - (S[:, 1] / n) ** 2))
    return round(float(D.x.corr(D.y)), 3), [round(float(v), 3) for v in np.quantile(r, [.025, .975])]


def slope(x, y):
    return round(float(np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)[0]), 2)


# ── 나스닥 ↔ ETH ─────────────────────────────────────────────────────────
def nasdaq_daily():
    import research_eth_rates_correlation_20261013 as RC
    F = RC.frame()
    F = F[F.NASDAQCOM.notna()]
    out = {}
    for rule, nm, blk in ((None, "daily", 20), ("W-FRI", "weekly", 8)):
        D = RC.changes(F, rule)
        out[nm] = {}
        for pn, a in (("2017-08~", "2017-08-01"), ("2022~", "2022-01-01"), ("2025~(주)", "2025-01-01"), ("2026~", "2026-01-01")):
            s = D[D.index >= a]
            dn, up = s[s.ndx < 0], s[s.ndx > 0]
            out[nm][pn] = dict(n=len(s), r=round(float(s.eth.corr(s.ndx)), 3), ci=RC.block_ci(s.ndx, s.eth, blk), beta=slope(s.ndx, s.eth),
                               beta_down=slope(dn.ndx, dn.eth), beta_up=slope(up.ndx, up.eth))
        out[nm]["by_year"] = {int(y): round(float(g.eth.corr(g.ndx)), 3) for y, g in D.groupby(D.index.year)}
    Dd = RC.changes(F)
    return out, Dd.eth.rolling(90, min_periods=60).corr(Dd.ndx)


def nasdaq_close_aligned(P):
    """나스닥 정규장 마감(16:00 ET) 시각의 ETH 가격끼리 일간 수익 — UTC 종가의 4시간 어긋남 제거."""
    c = P[(P.et_min == 955) & P.nq_live]                                     # 15:55 봉 종가 = 16:00
    d = pd.DataFrame({"eth": np.log(c.eth).diff() * 100, "nq": np.log(c.nq).diff() * 100, "day": c.day}).dropna()
    out = {}
    for pn, a in (("2024~", D0), ("2025~", "2025-01-01"), ("요즘 07~", RECENT)):
        s = d[d.day >= pd.Timestamp(a, tz="UTC")]
        r, ci = corr_ci(s.nq, s.eth, s.day.dt.strftime("%Y-%W"))
        out[pn] = dict(n=len(s), r=r, ci=ci, beta=slope(s.nq, s.eth))
    return out


def nasdaq_5m(P):
    L = P[P.nq_live & P.r_eth.notna() & P.r_nq.notna()]
    out = {"by_month": {}, "session": {}, "lead_lag": {}, "updown": {}}
    for m, g in L.groupby(L.ts.dt.strftime("%Y-%m")):
        out["by_month"][m] = dict(r=round(float(g.r_eth.corr(g.r_nq)), 3), r_cash=round(float(g[g.cash].r_eth.corr(g[g.cash].r_nq)), 3), beta=slope(g.r_nq, g.r_eth))
    for pn, a, b in (("2024~2026-06", D0, RECENT), ("요즘 2026-07~", RECENT, "2026-10-12")):
        s = L[(L.ts >= pd.Timestamp(a, tz="UTC")) & (L.ts < pd.Timestamp(b, tz="UTC"))]
        for nm, m in (("정규장", s.cash), ("그 외", ~s.cash)):
            r, ci = corr_ci(s[m].r_nq, s[m].r_eth, s[m].day)
            out["session"][f"{pn}|{nm}"] = dict(n=int(m.sum()), r=r, ci=ci, beta=slope(s[m].r_nq, s[m].r_eth), r2_pct=round(100 * r * r, 1))
        sd = s.r_nq.std()
        dn, up = s[s.r_nq < -2 * sd], s[s.r_nq > 2 * sd]
        out["updown"][pn] = dict(beta_bigdown=slope(dn.r_nq, dn.r_eth), beta_bigup=slope(up.r_nq, up.r_eth),
                                 eth_on_bigdown_bp=round(float(dn.r_eth.mean()), 1), eth_on_bigup_bp=round(float(up.r_eth.mean()), 1), n=[len(dn), len(up)])
        x, y = s.r_nq.to_numpy(), s.r_eth.to_numpy()
        out["lead_lag"][pn] = {k: round(float(np.corrcoef(x[max(0, -k):len(s) - max(0, k)], y[max(0, k):len(s) - max(0, -k)])[0, 1]), 3) for k in range(-3, 4)}   # k>0: 나스닥 k봉 먼저
    return out


# ── 금리 발작 ───────────────────────────────────────────────────────────
def spasms(P, mult=4.0):
    B = P[P.bond_live & P.r_bond.notna()].copy()
    sd = B.r_bond.shift(1).rolling(20 * 270, min_periods=5 * 270).std()        # 직전 ~20거래일(활동 봉 ~270/일)
    B["z"] = B.r_bond / sd
    B["spasm"] = B.z.abs() >= mult
    B["s"] = -np.sign(B.r_bond)                                               # +1 = 금리↑(채권 가격↓)
    return B


def spasm_study(P, B):
    pos = pd.Series(np.arange(len(P)), index=P.index)
    r_eth, r_nq = P.r_eth.to_numpy(), P.r_nq.to_numpy()
    base_abs = P[~P.index.isin(B.index[B.spasm])].groupby("slot").r_eth.apply(lambda x: x.abs().mean())
    ev = B[B.spasm].copy()
    ii = pos.reindex(ev.index).to_numpy()
    ev["eth0"], ev["nq0"] = r_eth[ii], r_nq[ii]
    for h, nm in ((12, "1h"), (24, "2h")):
        ev[f"eth_{nm}"] = [np.nansum(r_eth[i + 1:i + 1 + h]) for i in ii]
    ev["eth_pre30"] = [np.nansum(r_eth[max(0, i - 6):i]) for i in ii]
    ev["base_abs"] = P.slot.reindex(ev.index).map(base_abs).to_numpy()
    ev["et_hhmm"] = [f"{int(m) // 60:02d}:{int(m) % 60:02d}" for m in P.et_min.reindex(ev.index)]
    ev["recent"] = ev.index >= pd.Timestamp(RECENT).value // 10**6
    out = {}
    for nm, g, months in (("예전 2024-01~2026-06", ev[~ev.recent], 30.0), ("요즘 2026-07~10-11", ev[ev.recent], 3.35)):
        days = P.day.reindex(g.index).to_numpy()
        sg = g.s.to_numpy()
        row = dict(n=len(g), per_month=round(len(g) / months, 1), abs_eth_bp=round(float(g.eth0.abs().mean()), 1), base_abs_bp=round(float(g.base_abs.mean()), 1),
                   abs_ratio=round(float(g.eth0.abs().mean() / g.base_abs.mean()), 2),
                   signed_eth0_bp=round(float((sg * g.eth0).mean()), 1), signed_eth0_ci=day_boot_mean(sg * g.eth0, days),
                   share_eth_against=round(float((sg * g.eth0 < 0).mean()), 3), signed_nq0_bp=round(float((sg * g.nq0).mean()), 1),
                   signed_eth_1h=round(float((sg * g.eth_1h).mean()), 1), signed_eth_1h_ci=day_boot_mean(sg * g.eth_1h, days),
                   signed_eth_2h=round(float((sg * g.eth_2h).mean()), 1), signed_eth_2h_ci=day_boot_mean(sg * g.eth_2h, days),
                   signed_pre30=round(float((sg * g.eth_pre30).mean()), 1), top_times_et=g.et_hhmm.value_counts().head(6).to_dict())
        if len(g) > 5:
            bs = np.random.default_rng(7).integers(0, len(g), (2000, len(g)))
            row["abs_ratio_ci"] = [round(float(v), 2) for v in np.quantile(g.eth0.abs().to_numpy()[bs].mean(1) / g.base_abs.to_numpy()[bs].mean(1), [.025, .975])]
        if len(g) >= 15:                                                      # 나스닥 통제: ETH0 ~ NQ0 + 채권0 (OLS t)
            X = np.c_[np.ones(len(g)), g.nq0.fillna(0), g.r_bond]
            b = np.linalg.lstsq(X, g.eth0.to_numpy(), rcond=None)[0]
            se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (g.eth0.to_numpy() - X @ b).var(ddof=3))
            row["reg_eth_on_nq_bond"] = dict(b_nq=round(float(b[1]), 2), t_nq=round(float(b[1] / se[1]), 1), b_bond=round(float(b[2]), 2), t_bond=round(float(b[2] / se[2]), 1))
        out[nm] = row
    return out, ev


def yahoo_check(P):
    """최근 60일 원천 대조: Dukascopy 30년 CFD ↔ Yahoo ZB(30년)·ZN(10년), 나스닥 CFD ↔ NQ — 5분 수익 상관(같은 봉·±1봉). Yahoo 행 = 봉 시작."""
    out = {}
    for nm, col in (("ZB", "r_bond"), ("ZN", "r_bond"), ("NQ", "r_nq")):
        y = (np.log(yahoo_5m(nm).c).diff() * 1e4).rename("y")
        J = P[[col]].join(y, how="inner").dropna()
        J = J[J[col] != 0]
        out[nm] = {f"shift{k}": round(float(J[col].corr(J.y.shift(k))), 3) for k in (-1, 0, 1)} | {"n": len(J)}
    return out


def plot(P, ev, roll_daily, nq5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    fp = "/home/kbj20/crypto-scalping/tmp/fonts/NotoSansKR-VF.ttf"
    fm.fontManager.addfont(fp); plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name(); plt.rcParams["axes.unicode_minus"] = False
    S, T1, T2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
    C = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]

    def style(a):
        a.set_facecolor(S); a.grid(color=GRID, lw=.6); a.tick_params(colors=T2, labelsize=9)
        for sp in a.spines.values():
            sp.set_visible(False)
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), facecolor=S)
    for a in ax:
        style(a)
    ax[0].plot(roll_daily.index, roll_daily, color=C[0], lw=1.6)
    ax[0].axhline(0, color=T2, lw=.8)
    ax[0].set_title("ETH ↔ 나스닥 일간 수익, 90영업일 이동 상관 (2017~)", loc="left", color=T1, fontsize=11)
    m = pd.DataFrame(nq5["by_month"]).T
    m.index = pd.to_datetime(m.index)
    ax[1].plot(m.index, m.r, color=C[0], lw=1.6, marker="o", ms=4, label="5분 수익 상관(나스닥 거래 봉 전체)")
    ax[1].plot(m.index, m.r_cash, color=C[1], lw=1.6, marker="o", ms=4, label="미 정규장(09:30–16:00 ET)만")
    ax[1].axvspan(pd.Timestamp(RECENT), m.index.max() + pd.Timedelta(days=20), color=GRID, alpha=.6, lw=0)
    ax[1].set_title("ETH ↔ 나스닥100 5분봉 수익 상관, 월별 (회색 = 요즘 2026-07~)", loc="left", color=T1, fontsize=11)
    ax[1].legend(frameon=False, fontsize=9, labelcolor=T2, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "eth_nasdaq_corr.png", dpi=130, facecolor=S)
    pos = pd.Series(np.arange(len(P)), index=P.index)
    r = P.r_eth.to_numpy()
    fig, a = plt.subplots(figsize=(12, 4.6), facecolor=S); style(a)
    for c, (nm, g) in zip((C[3], C[1]), (("예전 2024-01~2026-06", ev[~ev.recent]), ("요즘 2026-07~10-11", ev[ev.recent]))):
        ii = pos.reindex(g.index).to_numpy()
        M = np.array([[r[i + k] if 0 <= i + k < len(r) else np.nan for k in range(-5, 25)] for i in ii]) * g.s.to_numpy()[:, None]
        cum = np.r_[0.0, np.nancumsum(np.nanmean(M, 0))]
        a.plot(np.arange(-6, 25) * 5, cum - cum[6], color=c, lw=2, marker="o", ms=3.5, label=f"{nm} ({len(g)}회)")
    a.axvline(0, color=T2, lw=.8); a.axhline(0, color=T2, lw=.8)
    a.set_xlabel("발작 봉 끝 기준 분 (−5 ~ 0 = 발작 봉)", color=T2, fontsize=9); a.set_ylabel("금리 방향 부호 × ETH 누적 bp", color=T2, fontsize=9)
    a.set_title("금리 발작(30년물 5분 |z|≥4) 전후 ETH 평균 경로 — 아래로 = 금리 오를 때 ETH 하락", loc="left", color=T1, fontsize=11)
    a.legend(frameon=False, fontsize=9, labelcolor=T2)
    fig.tight_layout(); fig.savefig(OUT / "eth_rate_spasm_path.png", dpi=130, facecolor=S)


def main():
    P = panel()
    rep = {"range": [str(P.ts.min()), str(P.ts.max())], "bars": len(P), "live_share": {"nq": round(float(P.nq_live.mean()), 3), "bond": round(float(P.bond_live.mean()), 3)}}
    rep["source_check_yahoo"] = yahoo_check(P)
    rep["nasdaq_daily"], roll = nasdaq_daily()
    rep["nasdaq_close_aligned_daily"] = nasdaq_close_aligned(P)
    rep["nasdaq_5m"] = nasdaq_5m(P)
    rep["spasm"], ev = spasm_study(P, spasms(P))
    rep["spasm_mult3"] = spasm_study(P, spasms(P, 3.0))[0]                   # 문턱 민감도
    ev.to_parquet(OUT / "spasm_events.parquet")
    json.dump(rep, open(OUT / "report.json", "w"), ensure_ascii=False, indent=1, default=str)
    plot(P, ev, roll, rep["nasdaq_5m"])
    print("saved", OUT)


def selftest():
    k = pd.DataFrame({"t": np.array([0, 60_000, 300_000]), "c": [1.0, 2.0, 3.0], "v": [1.0, 1.0, 0.0]})
    f = to5(k)
    assert f.c.tolist() == [2.0, 3.0] and f.v.tolist() == [2.0, 0.0]
    raw = np.zeros((1, 6), dtype=">i4"); raw[0, 0], raw[0, 2] = 60, 123456
    d = decode_bi5(lzma.compress(raw.tobytes(), format=lzma.FORMAT_ALONE), 1_000)
    assert d.t.iat[0] == 61_000 and d.c.iat[0] == 123.456
    x = np.random.default_rng(0).normal(size=5000); y = .5 * x + np.random.default_rng(1).normal(size=5000)
    rr, ci = corr_ci(x, y, np.arange(5000) // 50)
    assert ci[0] < rr < ci[1] and abs(rr - .447) < .05
    print("selftest OK")


if __name__ == "__main__":
    {"--fetch": fetch, "--selftest": selftest}.get(sys.argv[1] if len(sys.argv) > 1 else "", main)()
