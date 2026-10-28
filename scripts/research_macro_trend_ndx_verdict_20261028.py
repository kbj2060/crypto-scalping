"""나스닥100 13주 판정(2026-10-28 대시보드 세션 임시 계산, 사용자 «Macro Trend 카드 주식 칸에 나스닥 추가»).

scripts/research_macro_trend_p3_verdict_20261028.py 판정 루프 그대로 -- 정식 판정표의 패널(panel_daily.parquet, CC BY-NC 라 git 밖)이 없어 따로 계산했다.
S&P500 을 같은 코드로 대조해 판정표와 거의 같음을 확인했다(ETH 동행 r .628 vs .630 · BTC .403 vs .406 · 판정 동일).
결과 4줄(나스닥)을 L_p3_trend_verdict.csv 끝에 붙였다 -- 정식 판정표를 다시 만들 때 그 줄로 바뀐다.
  서버: python scripts/research_macro_trend_ndx_verdict_20261028.py   (FRED_API_KEY · 바이낸스 REST -- 로컬에서 돌리지 말 것)
"""
# 다른 점: 코인 = 바이낸스 선물 일봉 종가(연구는 Yahoo Open[D+1]), 주간 = 일별 ffill(10) → W-FRI last(연구 sample('W') 와 같은 규약)
import os, json, numpy as np, pandas as pd, requests
from pathlib import Path
from dotenv import load_dotenv; load_dotenv(Path(__file__).resolve().parents[1] / ".env")
key = os.environ["FRED_API_KEY"]
def fred(sid):
    js = requests.get("https://api.stlouisfed.org/fred/series/observations", params={"series_id": sid, "api_key": key, "file_type": "json", "observation_start": "2019-01-01"}, timeout=30).json()
    ob = [o for o in js["observations"] if o["value"] not in (".", "")]
    return pd.Series([float(o["value"]) for o in ob], index=pd.to_datetime([o["date"] for o in ob]))
def kl(sym):
    out, start = [], 1546300800000
    while True:
        k = requests.get("https://fapi.binance.com/fapi/v1/klines", params={"symbol": sym, "interval": "1d", "limit": 1500, "startTime": start}, timeout=20).json()
        if not k: break
        out += k; start = k[-1][0] + 86400000
        if len(k) < 1500: break
    k = [x for x in out if float(x[6]) < pd.Timestamp.now().timestamp() * 1000]
    return pd.Series([float(x[4]) for x in k], index=pd.to_datetime([int(x[0]) for x in k], unit="ms").normalize())
panel = pd.concat({"NDX": fred("NASDAQ100"), "SPX": fred("SP500"), "BTC": kl("BTCUSDT"), "ETH": kl("ETHUSDT")}, axis=1).sort_index()
panel = panel.reindex(pd.date_range(panel.index.min(), panel.index.max(), freq="D"))
w = panel.ffill(limit=10).resample("W-FRI").last()
P3, P3A, P3B, P2, EXCL = ("2024-01-12", "2026-12-31"), ("2024-01-12", "2025-05-30"), ("2025-06-06", "2026-12-31"), ("2020-01-03", "2024-01-05"), ("2024-06-01", "2024-11-30")
def corr(d, lo, hi):
    d = d.loc[lo:hi].dropna(); return float(d.corr().iloc[0, 1]) if len(d) > 20 else float("nan")
def perm_p(d, gap, rng, n=3000):
    d = d.loc[P3[0]:P3[1]].dropna(); x, y = d.iloc[:, 0].values, d.iloc[:, 1].values; obs = np.corrcoef(x, y)[0, 1]
    null = [np.corrcoef(np.roll(x, k), y)[0, 1] for k in rng.integers(gap + 4, len(x) - gap - 4, n)]; return float(np.mean(np.abs(null) >= abs(obs)))
rng = np.random.default_rng(11); rows = []
for asset in ("BTC", "ETH"):
    c = 100 * np.log(w[asset])
    for kind in ("동행13", "선행13"):
        for name, col in (("나스닥", "NDX"), ("S&P500(대조)", "SPX")):
            x = 100 * np.log(w[col])
            d, gap = (pd.concat([c.diff(13), x.diff(13)], axis=1), 13) if kind == "동행13" else (pd.concat([x.diff(13).shift(1), c.shift(-13) - c], axis=1), 26)
            r3, ra, rb, r2 = corr(d, *P3), corr(d, *P3A), corr(d, *P3B), corr(d, *P2)
            dx = d.loc[P3[0]:P3[1]]; rx = float(dx.drop(dx.loc[EXCL[0]:EXCL[1]].index).dropna().corr().iloc[0, 1])
            p = perm_p(d, gap, rng); sg = np.sign(r3); dd = d.loc[P3[0]:P3[1]].dropna()
            rc = dd.iloc[:, 0].rolling(52).corr(dd.iloc[:, 1]).dropna(); roll = float((np.sign(rc) == sg).mean()) if len(rc) else float("nan")
            halves = np.sign(ra) == sg and np.sign(rb) == sg
            if abs(r3) >= 0.3 and p < 0.05 and halves and min(abs(ra), abs(rb)) >= 0.15 and np.sign(rx) == sg and roll >= 0.75: v = "사용 가능"
            elif (p < 0.10 or abs(r3) >= 0.3) and halves and np.sign(rx) == sg: v = "관찰"
            else: v = "불가"
            rel = "비교 불가" if not np.isfinite(r2) else ("신규" if abs(r2) < 0.15 else ("유지" if np.sign(r2) == sg else "반전"))
            rows.append(dict(asset=asset, kind=kind, indicator=name, verdict=v, vs_2020_23=rel, r_p3=round(r3, 4), perm_p=round(p, 4), r_first_half=round(ra, 4),
                             r_second_half=round(rb, 4), r_excl_2024h2=round(rx, 4), roll52_same_sign=round(roll, 4), r_2020_23=round(r2, 4), n_weeks=int(len(dd))))
print(pd.DataFrame(rows).to_string(index=False)); print(json.dumps(rows, ensure_ascii=False))
