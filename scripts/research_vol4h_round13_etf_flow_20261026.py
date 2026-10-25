#!/usr/bin/env python3
"""4h 변동성 13라운드 — QX4 + ETF 유입 크기 잔차 보정. 사전 등록 docs/prereg/vol4h_round13_etf_flow_20261026.md.

    python3 scripts/research_vol4h_round13_etf_flow_20261026.py --fetch   # Farside ETH·BTC 일별 유입
    python3 scripts/research_vol4h_round13_etf_flow_20261026.py          # 한 번만
실주문 없음.
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_vol4h_forecast_horserace_20261025 as V   # noqa: E402

OUT = Path("/home/kbj20/crypto-scalping/tmp/vol4h_r13_etf_20261026")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
WINS = {"W2": "5W2", "W4": "4"}


def parse(html):
    """Farside 표 → {날짜: Total(백만 $)}. 괄호 = 음수, «-»·빈칸 = 결측."""
    out = {}
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        c = [re.sub(r"<[^>]+>", "", x).strip() for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)]
        if len(c) < 3 or not re.fullmatch(r"\d{1,2} [A-Z][a-z]{2} \d{4}", c[0]):
            continue
        v = c[-1].replace(",", "")
        if v in ("", "-"):
            continue
        out[pd.Timestamp(c[0])] = -float(v[1:-1]) if v.startswith("(") else float(v)
    return out


def fetch():
    OUT.mkdir(parents=True, exist_ok=True)
    s = {k: pd.Series(parse(requests.get(f"https://farside.co.uk/{k}-etf-flow-all-data/", headers=UA, timeout=60).text)) for k in ("ethereum", "bitcoin")}
    df = pd.DataFrame({"eth": s["ethereum"], "btc": s["bitcoin"]}).sort_index(); df.index.name = "date"
    df.reset_index().to_parquet(OUT / "etf_flows.parquet"); print(df.tail(3), len(df), int(df.eth.notna().sum()))


def feats(df, d):
    """결정 시각 d(정시) → (xE, xB): D+1 14:00 UTC ≤ d 인 가장 늦은 거래일 D 의 |유입| / 직전 40 거래일 |유입| 중앙값."""
    out = []
    for col in ("eth", "btc"):
        s = df[col].dropna().abs(); ref = s.rolling(40, min_periods=20).median().shift(1)
        x = np.log1p(s / ref)
        known = s.index + pd.Timedelta(days=1, hours=14)
        k = np.searchsorted(known.values, d.values, side="right") - 1
        out.append(np.where(k >= 0, x.to_numpy()[np.maximum(k, 0)], np.nan))
    return out


def selftest():
    h = "<tr><td>21 Oct 2026</td><td>1</td><td>(29.3)</td></tr><tr><td>22 Oct 2026</td><td>x</td><td>1,104.4</td></tr><tr><td>Total</td><td>1</td><td>9</td></tr>"
    p = parse(h); assert p == {pd.Timestamp("2026-10-21"): -29.3, pd.Timestamp("2026-10-22"): 1104.4}, p
    df = pd.DataFrame({"eth": [1.0] * 30 + [5.0], "btc": [1.0] * 31}, index=pd.bdate_range("2026-01-01", periods=31))
    D = df.index[-1]; d = pd.DatetimeIndex([D + pd.Timedelta(days=1, hours=13), D + pd.Timedelta(days=1, hours=14)])
    xE, _ = feats(df, d)
    assert np.isclose(xE[0], np.log1p(1.0)) and np.isclose(xE[1], np.log1p(5.0)), xE   # 14:00 전엔 전날 값, 14:00 부터 D 값


def block_q(s, q, b=5, n=4000, seed=7):
    rng = np.random.default_rng(seed); m = len(s); k = int(np.ceil(m / b))
    return float(np.quantile([s[(rng.integers(0, m - b + 1, k)[:, None] + np.arange(b)).ravel()[:m]].mean() for _ in range(n)], q))


def main():
    selftest()
    rp = OUT / "report.json"; assert not rp.exists(), "판정은 한 번만"
    df = pd.read_parquet(OUT / "etf_flows.parquet").set_index("date")
    rep, days = {}, []
    for w, tg in WINS.items():
        Pv, Pt = (pd.read_parquet(V.OUT / f"pred_{s}_r{tg}_ETH.parquet") for s in ("val", "test"))

        def X(P):
            xE, xB = feats(df, P.index + pd.Timedelta("5min"))
            return np.c_[np.ones(len(P)), np.nan_to_num(xE), np.nan_to_num(xB)], float(np.isnan(xE).mean())
        Xv, miss_v = X(Pv); Xt, miss_t = X(Pt); hv, ht, yv, yt = Pv.v_QX4.to_numpy(), Pt.t_QX4.to_numpy(), Pv.y.to_numpy(), Pt.y.to_numpy()
        beta = np.linalg.lstsq(Xv, yv - hv, rcond=None)[0]
        pv, pt = {"QX4": hv, "QX4F": hv + Xv @ beta}, {"QX4": ht, "QX4F": ht + Xt @ beta}
        c = {m: float(np.log(np.mean(np.exp(yv - pv[m])))) for m in pv}
        L = {m: V.qlike(yt, pt[m] + c[m]) for m in pt}
        dd = pd.Series(L["QX4F"] - L["QX4"], index=Pt.index).groupby(Pt.index.normalize()).mean(); days.append(dd)
        rep[w] = dict(beta=beta.tolist(), missing=dict(val=miss_v, test=miss_t), qlike={m: float(v.mean()) for m, v in L.items()},
                      dq=float(dd.mean()), dq_ci95=list(V.day_ci(Pt.index, L["QX4F"] - L["QX4"]))[1:])
        print(w, json.dumps(rep[w], ensure_ascii=False), flush=True)
    s = pd.concat(days).to_numpy()
    rep["pooled"] = dict(dq=float(s.mean()), upper_one_sided90=block_q(s, .9), days=len(s))
    rep["E1"] = rep["pooled"]["upper_one_sided90"] < 0
    rep["E2"] = all(rep[w]["beta"][1] > 0 for w in WINS)
    rep["E3"] = all(rep[w]["dq"] < 0 for w in WINS)
    rep["verdict"] = "조건부 채택 후보" if rep["E1"] and rep["E2"] and rep["E3"] else "불통과 — ETF 유입 → 변동성 축 닫음"
    rp.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float)); print(json.dumps({k: rep[k] for k in ("pooled", "E1", "E2", "E3", "verdict")}, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--fetch"]: fetch()
    elif sys.argv[1:2] == ["--selftest"]: selftest(); print("ok")
    else: main()
