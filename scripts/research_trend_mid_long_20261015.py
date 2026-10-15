"""중·장기(7~91일) 추세 방향 예측 연구 — 2026-10-15.

단계:
  --fetch   공개 원천 일 단위 수집 → tmp/trend_mid_long_20261015/raw/
            (Coin Metrics 커뮤니티 · 바이낸스 현물 일봉 · FRED · 공포탐욕 · DefiLlama 스테이블코인)
  --build   38코인 일 패널 + 66피쳐(모멘텀·변동성·폭·거시·심리·온체인) → panel.parquet
            (범용 HGB/로지스틱/MLP 워크포워드는 42~54%·연도별 요동으로 기각 — 스크래치에서 수행)
  --liq     최종 규칙 «달러·유동성 점수 Z» 평가 + 오늘 신호 → liq_report.json
            Z = z(−Δ91 ln 달러지수) + z(Δ91 ln 연준 순유동성), z = ÷ 직전 730일 표준편차
            Z ≥ k → 상승, Z ≤ −k → 하락, 그 사이 = 예측 보류(k = 1.0 기본, 1.5 고확신)
  --selftest  인과성(잘라서 다시 계산해도 같은 Z) 점검
"""
import argparse, io, json, time, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "tmp" / "trend_mid_long_20261015"
RAW = OUT / "raw"

CM_METRICS = ["PriceUSD", "CapMVRVCur", "AdrActCnt", "TxCnt", "CapMrktCurUSD", "SplyCur",
              "FlowInExNtv", "FlowOutExNtv", "SplyExNtv", "HashRate", "FeeTotNtv"]
CM_ASSETS = ["btc", "eth", "ltc", "xrp", "bch", "doge", "ada", "link", "bnb", "sol", "trx", "xlm", "etc"]
BN_SYMS = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "TRX", "LTC", "LINK", "DOT", "AVAX", "MATIC",
           "BCH", "ETC", "XLM", "ATOM", "UNI", "FIL", "NEAR", "ALGO", "VET", "ICP", "AAVE", "EOS",
           "XTZ", "THETA", "NEO", "IOTA", "ZEC", "DASH", "XMR", "SUI", "APT", "ARB", "OP", "TON", "HBAR"]
# ponytail: FRED 발표 지연은 build 단계에서 시리즈별 고정 지연으로 처리(빈티지 미반영 — ALFRED 로 업그레이드)
FRED = ["DGS10", "DGS2", "DFII10", "T10YIE", "DTWEXBGS", "NASDAQCOM", "SP500", "VIXCLS",
        "BAMLH0A0HYM2", "WALCL", "RRPONTSYD", "WTREGEN", "M2SL", "DCOILWTICO", "GOLDAMGBD228NLBM"]


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as r:
                return r.read()
        except Exception:  # 네트워크 일시 오류만 재시도
            if k == tries - 1:
                raise
            time.sleep(2 + 3 * k)


def fetch_cm():
    for a in CM_ASSETS:
        if (RAW / f"cm_{a}.csv").exists():
            continue
        rows, url = [], ("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=" + a
                         + "&metrics=" + ",".join(CM_METRICS) + "&frequency=1d&page_size=10000&start_time=2010-01-01"
                         + "&ignore_unsupported_errors=true")
        while url:
            try:
                j = json.loads(get(url))
            except Exception as e:  # 커뮤니티 범위 밖 자산(403)
                print("cm", a, "FAIL", e); break
            rows += j.get("data", [])
            url = j.get("next_page_url")
        df = pd.DataFrame(rows)
        if df.empty:
            print("cm", a, "empty"); continue
        df["date"] = pd.to_datetime(df.pop("time")).dt.tz_localize(None).dt.normalize()
        df = df.drop(columns="asset").set_index("date").apply(pd.to_numeric, errors="coerce").sort_index()
        df.to_csv(RAW / f"cm_{a}.csv")
        print("cm", a, len(df), df.index.min().date(), list(df.columns))


def fetch_binance():
    for s in BN_SYMS:
        if (RAW / f"bn_{s}.csv").exists():
            continue
        rows, start = [], 1500000000000
        while True:
            try:
                j = json.loads(get(f"https://api.binance.com/api/v3/klines?symbol={s}USDT&interval=1d&limit=1000&startTime={start}"))
            except Exception as e:  # 상장 폐지·없는 심볼
                print("bn", s, "FAIL", e); break
            if not j:
                break
            rows += j
            start = j[-1][0] + 86400000
            if len(j) < 1000:
                break
        if not rows:
            print("bn", s, "none"); continue
        df = pd.DataFrame(rows).iloc[:, :11]
        df.columns = ["t", "open", "high", "low", "close", "vol", "tc", "qvol", "n", "tbvol", "tbqvol"]
        df["date"] = pd.to_datetime(df.t, unit="ms")
        df = df.set_index("date").drop(columns=["t", "tc"]).astype(float)
        df.to_csv(RAW / f"bn_{s}.csv")
        print("bn", s, len(df), df.index.min().date())


FUT_SYMS = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "LTC", "LINK", "DOT", "AVAX", "BCH", "ETC", "XLM", "ATOM", "TRX"]


def fetch_futures():
    """USDT 무기한: 펀딩 이력(8h) + 프리미엄 지수 일봉(베이시스) — 2019-09~."""
    for s in FUT_SYMS:
        if (RAW / f"fut_{s}.csv").exists():
            continue
        fr, start = [], 1567296000000
        while True:
            j = json.loads(get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={s}USDT&limit=1000&startTime={start}"))
            if not j:
                break
            fr += j
            start = j[-1]["fundingTime"] + 1
            if len(j) < 1000:
                break
        pk, start = [], 1567296000000
        while True:
            j = json.loads(get(f"https://fapi.binance.com/fapi/v1/premiumIndexKlines?symbol={s}USDT&interval=1d&limit=1000&startTime={start}"))
            if not j:
                break
            pk += j
            start = j[-1][0] + 86400000
            if len(j) < 1000:
                break
        f = pd.DataFrame(fr)
        f["date"] = pd.to_datetime(f.fundingTime, unit="ms").dt.normalize()
        d = f.assign(fundingRate=f.fundingRate.astype(float)).groupby("date").fundingRate.sum().rename("funding").to_frame()
        p = pd.DataFrame(pk).iloc[:, [0, 4]]
        p.columns = ["t", "premium"]
        p.index = pd.to_datetime(p.t, unit="ms")
        d = d.join(p.premium.astype(float), how="outer")
        d.index.name = "date"
        d.to_csv(RAW / f"fut_{s}.csv")
        print("fut", s, len(d), d.index.min().date())


def fetch_misc():
    for sid in FRED:
        try:
            df = pd.read_csv(io.BytesIO(get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")), na_values=".")
            df.to_csv(RAW / f"fred_{sid}.csv", index=False)
            print("fred", sid, len(df), df.iloc[0, 0])
        except Exception as e:
            print("fred", sid, "FAIL", e)
    j = json.loads(get("https://api.alternative.me/fng/?limit=0&format=json"))["data"]
    pd.DataFrame({"date": pd.to_datetime([int(x["timestamp"]) for x in j], unit="s"),
                  "fng": [int(x["value"]) for x in j]}).sort_values("date").to_csv(RAW / "fng.csv", index=False)
    j = json.loads(get("https://stablecoins.llama.fi/stablecoincharts/all"))
    pd.DataFrame({"date": pd.to_datetime([int(x["date"]) for x in j], unit="s"),
                  "stable_usd": [x.get("totalCirculatingUSD", {}).get("peggedUSD") for x in j]}).to_csv(RAW / "stable.csv", index=False)
    print("fng/stable ok")


H_LIST = [7, 14, 28, 56, 91]
# 발표 지연(일): 값이 날짜 d 로 찍힌 시리즈를 d+lag 일 종가부터 쓴다
# DTWEXBGS(H.10)는 주 1회 월요일에 한 주치가 발표 → 10일
FRED_LAG = {"DGS10": 1, "DGS2": 1, "DFII10": 1, "T10YIE": 1, "DTWEXBGS": 10, "NASDAQCOM": 1, "SP500": 1,
            "VIXCLS": 1, "DCOILWTICO": 1, "RRPONTSYD": 1, "WALCL": 2, "WTREGEN": 2, "M2SL": 55}


def _asof(s, idx, lag):
    """날짜 d 값을 d+lag 부터 쓰도록 밀고 idx 에 앞채움."""
    s = s.dropna().copy()
    s.index = s.index + pd.Timedelta(days=lag)
    return s.reindex(s.index.union(idx)).ffill().reindex(idx)


def coin_feats(df):
    lp = np.log(df.close)
    r1 = lp.diff()
    f = pd.DataFrame(index=df.index)
    vol30 = r1.rolling(30).std()
    for L in [7, 14, 28, 56, 91, 182, 365]:
        f[f"mom{L}"] = (lp - lp.shift(L)) / (vol30 * np.sqrt(L))
    for L in [20, 50, 100, 200]:
        f[f"ma{L}"] = (lp - lp.rolling(L).mean()) / vol30
    f["ma50_200"] = (lp.rolling(50).mean() - lp.rolling(200).mean()) / vol30
    f["vol7_30"] = np.log(r1.rolling(7).std() / vol30)
    f["vol30_180"] = np.log(vol30 / r1.rolling(180).std())
    f["vol30"] = vol30
    f["dd365"] = lp - lp.rolling(365, min_periods=200).max()
    f["up365"] = lp - lp.rolling(365, min_periods=200).min()
    f["qv7_90"] = np.log(df.qvol.rolling(7).mean() / df.qvol.rolling(90).mean())
    f["tb7"] = (df.tbqvol / df.qvol).rolling(7).mean() - (df.tbqvol / df.qvol).rolling(90).mean()
    f["skew30"] = r1.rolling(30).skew()
    f["t5"] = sum(np.sign(lp - lp.shift(L)) for L in [7, 14, 28, 56, 90]) / 5
    return f


def build():
    coins = {}
    for s in BN_SYMS:
        p = RAW / f"bn_{s}.csv"
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            df = df[df.qvol > 0]
            if len(df) >= 400:
                coins[s] = df
    idx = pd.date_range("2017-08-17", max(d.index.max() for d in coins.values()), freq="D")
    lps = pd.DataFrame({s: np.log(d.close) for s, d in coins.items()}).reindex(idx)
    # 시장 폭·분산(그날 거래되는 코인만)
    mkt = pd.DataFrame(index=idx)
    mkt["breadth50"] = (lps > lps.rolling(50, min_periods=50).mean()).where(lps.notna()).mean(1)
    mkt["breadth200"] = (lps > lps.rolling(200, min_periods=200).mean()).where(lps.rolling(200).count() >= 200).mean(1)
    r28 = lps - lps.shift(28)
    mkt["disp28"] = r28.std(1)
    mkt["mkt28"] = r28.median(1)
    mkt["mkt91"] = (lps - lps.shift(91)).median(1)
    b = lps["BTC"]
    mkt["btc_mom28"] = b - b.shift(28)
    mkt["btc_mom91"] = b - b.shift(91)
    mkt["btc_ma200"] = b - b.rolling(200).mean()
    eb = lps["ETH"] - b
    mkt["ethbtc28"] = eb - eb.shift(28)
    mkt["ethbtc91"] = eb - eb.shift(91)
    # 거시
    fr = {}
    for sid, lag in FRED_LAG.items():
        d = pd.read_csv(RAW / f"fred_{sid}.csv", parse_dates=[0], index_col=0).iloc[:, 0]
        fr[sid] = _asof(pd.to_numeric(d, errors="coerce"), idx, lag)
    fr = pd.DataFrame(fr)
    mac = pd.DataFrame(index=idx)
    for L in [28, 91]:
        mac[f"d10y_{L}"] = fr.DGS10 - fr.DGS10.shift(L)
        mac[f"dreal_{L}"] = fr.DFII10 - fr.DFII10.shift(L)
        mac[f"dxy_{L}"] = np.log(fr.DTWEXBGS / fr.DTWEXBGS.shift(L))
        mac[f"ndx_{L}"] = np.log(fr.NASDAQCOM / fr.NASDAQCOM.shift(L))
        mac[f"oil_{L}"] = np.log(fr.DCOILWTICO.clip(lower=1) / fr.DCOILWTICO.clip(lower=1).shift(L))
        netliq = fr.WALCL / 1e3 - fr.RRPONTSYD - fr.WTREGEN / 1e3  # 십억 달러
        mac[f"netliq_{L}"] = np.log(netliq / netliq.shift(L))
    mac["curve"] = fr.DGS10 - fr.DGS2
    mac["breakeven"] = fr.T10YIE
    mac["vix"] = np.log(fr.VIXCLS)
    mac["ndx_ma200"] = np.log(fr.NASDAQCOM / fr.NASDAQCOM.rolling(200).mean())
    mac["m2_yoy"] = np.log(fr.M2SL / fr.M2SL.shift(365))
    fng = pd.read_csv(RAW / "fng.csv", parse_dates=["date"], index_col="date").fng
    mac["fng"] = _asof(fng, idx, 1)
    mac["fng_d28"] = mac.fng - mac.fng.shift(28)
    st = pd.read_csv(RAW / "stable.csv", parse_dates=["date"], index_col="date").stable_usd
    st = _asof(st.where(st > 1e8), idx, 1)
    mac["stable28"] = np.log(st / st.shift(28))
    mac["stable91"] = np.log(st / st.shift(91))
    # 온체인(BTC·ETH 만, 하루 늦춤)
    for a in ["btc", "eth"]:
        cm = pd.read_csv(RAW / f"cm_{a}.csv", index_col=0, parse_dates=True)
        mv = _asof(cm.CapMVRVCur, idx, 1)
        mac[f"{a}_mvrv"] = np.log(mv)
        mac[f"{a}_mvrv_d91"] = np.log(mv / mv.shift(91))
        aa = _asof(cm.AdrActCnt.rolling(7).mean(), idx, 1)
        mac[f"{a}_adr_91"] = np.log(aa / aa.shift(91))
        if "SplyExNtv" in cm:
            se = _asof(cm.SplyExNtv, idx, 1)
            mac[f"{a}_exsply_28"] = np.log(se / se.shift(28))
    rows = []
    for s, df in coins.items():
        f = coin_feats(df).reindex(idx)
        lp = lps[s]
        f["rs_btc28"] = (lp - lp.shift(28)) - mkt.btc_mom28
        f["rs_btc91"] = (lp - lp.shift(91)) - mkt.btc_mom91
        f["age"] = np.log1p(lp.notna().cumsum())
        for h in H_LIST:
            f[f"fwd{h}"] = lp.shift(-h) - lp
        f["sym"] = s
        rows.append(f.join(mkt).join(mac).dropna(subset=["mom28", "fwd7"], how="any").reset_index(names="date"))
    panel = pd.concat(rows, ignore_index=True)
    panel.to_parquet(OUT / "panel.parquet")
    print("panel", panel.shape, panel.date.min().date(), panel.date.max().date(), panel.sym.nunique())


FEAT_EXCL = {"sym", "date", "vol30"} | {f"fwd{h}" for h in H_LIST}


def load_prices():
    """긴 역사 가격: Coin Metrics(BTC 2010~·ETH 2015~ 등) + CM 에 없는 코인은 바이낸스 종가."""
    px = {}
    for f in sorted(RAW.glob("cm_*.csv")):
        s = pd.read_csv(f, index_col=0, parse_dates=True).PriceUSD.dropna()
        if (s > 0).sum() > 700:
            px[f.stem[3:].upper()] = s[s > 0]
    for f in sorted(RAW.glob("bn_*.csv")):
        if f.stem[3:] not in px:
            px[f.stem[3:]] = pd.read_csv(f, index_col=0, parse_dates=True).close
    lp = np.log(pd.DataFrame(px).sort_index())
    return lp[lp.index >= "2012-01-01"]


def fred(sid, idx):
    s = pd.read_csv(RAW / f"fred_{sid}.csv", parse_dates=[0], index_col=0).iloc[:, 0]
    return _asof(pd.to_numeric(s, errors="coerce"), idx, FRED_LAG[sid])


def liq_score(dxy, walcl, rrp, tga, W=91, sd_win=730):
    """달러·유동성 점수. 입력은 이미 발표 지연만큼 밀린 일 시리즈(같은 인덱스)."""
    nl = walcl / 1e3 - rrp.fillna(0) - tga / 1e3          # 연준 순유동성(십억 달러)
    a = -np.log(dxy / dxy.shift(W))                       # 달러 약세 = +
    b = np.log(nl / nl.shift(W))                          # 유동성 증가 = +
    return a / a.rolling(sd_win, min_periods=365).std() + b / b.rolling(sd_win, min_periods=365).std()


def liq_inputs(idx):
    return [fred(k, idx) for k in ("DTWEXBGS", "WALCL", "RRPONTSYD", "WTREGEN")]


def _hits(lp, sig, h, cols, a, b):
    fwd = (lp.shift(-h) - lp)[cols].loc[a:b]
    s = sig.loc[a:b]
    ok = fwd.notna() & s.notna().values[:, None]
    return fwd.gt(0).eq(s.gt(0), axis=0).astype(float).where(ok), fwd.gt(0).astype(float).where(ok)


def _block_ci(x, block, rng, n_boot=1000):
    x = x.dropna().values
    if len(x) < 3 * block:
        return [None, None]
    k = int(np.ceil(len(x) / block)); st = np.arange(len(x) - block + 1)
    m = [np.concatenate([x[i:i + block] for i in rng.choice(st, k)])[:len(x)].mean() for _ in range(n_boot)]
    return [round(100 * float(v), 1) for v in np.percentile(m, [2.5, 97.5])]


def liq_report():
    lp = load_prices()
    Z = liq_score(*liq_inputs(lp.index))
    alts = [c for c in lp.columns if c not in ("BTC", "ETH")]
    groups = {"BTC": ["BTC"], "ETH": ["ETH"], "ALT": alts}
    rng = np.random.default_rng(0)
    rep = {"spec": "Z=z(-d91 ln DTWEXBGS[lag10]) + z(d91 ln (WALCL-RRP-TGA)[lag2/1/2]), z=/std730; |Z|>=k 만 예측",
           "today": {"date": str(Z.dropna().index[-1].date()), "Z": round(float(Z.dropna().iloc[-1]), 3)}, "rows": []}
    for k in (0.0, 1.0, 1.5):
        sig = np.sign(Z).where(Z.abs() >= k).replace(0, np.nan)
        for h in (14, 28, 56, 91):
            for g, cols in groups.items():
                row = {"k": k, "h": h, "asset": g, "coverage_2018": round(100 * float(sig.loc["2018":].notna().mean()), 1)}
                for a, b in (("2013", "2017"), ("2018", "2026"), ("2022", "2024"), ("2025", "2026")):
                    hit, up = _hits(lp, sig, h, cols, a, b)
                    st = hit.stack()
                    row[f"acc_{a}_{b}"] = round(100 * float(st.mean()), 1) if len(st) else None
                    row[f"base_up_{a}_{b}"] = round(100 * float(up.stack().mean()), 1) if len(st) else None
                    if g != "ALT" and a == "2018":
                        row["ci_2018_2026"] = _block_ci(hit[cols[0]], h, rng)
                hit, _ = _hits(lp, sig, h, cols, "2018", "2026")
                st = hit.stack()
                row["by_year"] = {int(y): round(100 * float(v), 1) for y, v in st.groupby(st.index.get_level_values(0).year).mean().items()}
                rep["rows"].append(row)
    (OUT / "liq_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    t = pd.DataFrame(rep["rows"])
    print(t[["k", "h", "asset", "coverage_2018", "acc_2018_2026", "ci_2018_2026", "base_up_2018_2026",
             "acc_2025_2026", "acc_2022_2024", "acc_2013_2017"]].to_string(index=False))
    print("today", rep["today"])


def selftest():
    """인과성: t 시점까지만 남긴 입력으로 다시 계산한 Z(t) 가 전체 계산 Z(t) 와 같아야 한다."""
    lp = load_prices()
    ins = liq_inputs(lp.index)
    Z = liq_score(*ins)
    for t in ["2020-03-15", "2022-06-30", "2025-10-10"]:
        cut = [s.loc[:t] for s in ins]
        assert abs(liq_score(*cut).loc[t] - Z.loc[t]) < 1e-9, t
    # 발표 지연: 원시 날짜 d 의 달러지수는 d+10 이전엔 보이지 않아야 한다
    raw = pd.read_csv(RAW / "fred_DTWEXBGS.csv", parse_dates=[0], index_col=0).iloc[:, 0].dropna()
    d = raw.index[-30]
    seen = ins[0].loc[d + pd.Timedelta(days=9)]
    assert not np.isclose(seen, raw.loc[d]) or np.isclose(raw.loc[:d].iloc[-2], raw.loc[d]), "lag broken"
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--liq", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if a.fetch:
        fetch_cm(); fetch_binance(); fetch_misc(); fetch_futures()
    if a.build:
        build()
    if a.liq:
        liq_report()
    if a.selftest:
        selftest()
