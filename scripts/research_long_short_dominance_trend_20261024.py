#!/usr/bin/env python3
"""롱·숏 우세(테이커 × ΔOI 4분류) ↔ 차트 추세 상관 — 2026-10-24.

사용자: «포지션 우세에 따른 차트 추세와 어떤 상관관계가 있는지 연구해줘»
(앞 질문: «1일,7일,30일 신규 숏/숏 정리/신규 롱/롱 정리를 모두 계산해서 롱과 숏이 누가 더 우세한지»)

## 지표 (대시보드 app.js QNAME 와 같은 규칙)
5분봉마다 ΔOI = OI(t+5m) − OI(t), 델타 = 테이커 매수 − 매도.
ΔOI>0 & 델타≥0 신규 롱 · ΔOI<0 & 델타≥0 숏 정리 · ΔOI>0 & 델타<0 신규 숏 · ΔOI<0 & 델타<0 롱 정리 (|ΔOI|×종가 $).
롱 몫 L_W(t) = [t−W, t) 의 (신규 롱 + 숏 정리) / 네 칸 합. 창 안 유효 봉 90% 미만이면 비움.

## 사전 고정 (결과 보기 전)
표본점: 일 단위 = 매일 UTC 00시 · 장중 = 매시 정각. 피쳐는 t 이전 봉만, 가격 P(t) = t 직전 봉 종가, 라벨은 P(t)→P(t+H).
DEV = ≤2024-12 · TEST = 2025-01~2026-08-21. ETH 2021-12~ 본 검정 · BTC·SOL 2024~ 복제(DEV = 2024 한 해).
A 동시: Spearman(L_W, R_W) — 같은 창 수익. (서술: 우세가 가격의 재표현인지)
B 예측: Spearman(L_W, F_H).
C 증분: rank(L_W) 를 rank(R_W)·rank(R_H) 에 회귀한 잔차 ↔ F_H (모멘텀 빼고 남는 것).
D 다이버전스: 상승 추세(R_W>0)에서 숏 우세(L_W < DEV 중앙값) − 롱 우세의 F_H 평균 차, 하락 추세에서도 같은 차.
   «가격은 오르는데 롱이 안 밀면 약한 상승» 실무 주장.
일 단위 W,H ∈ {1,7,30}일 · 장중 W,H ∈ {1,4}시간.
추론: 이동 블록 부트스트랩(일 30일 · 장중 168시간 블록) 2000회 · TEST 에서 ETH 가족(B·C·D) Holm.
통과 = ETH TEST CI 가 0 제외 + DEV 와 같은 부호 + Holm p<.05, 그리고 BTC·SOL TEST 부호 일치.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("/home/kbj20/crypto-scalping/data")
OUT = Path(__file__).resolve().parents[1] / "tmp" / "long_short_dominance_trend_20261024"
H_MS, DAY_MS, BAR = 3_600_000, 86_400_000, 300_000
TEST_FROM = pd.Timestamp("2025-01-01", tz="UTC").value // 10**6
KLINES = {"ETH": ["eth_5m_2021_2023_archive.csv", "eth_5m_1year.csv", "eth_5m_2026_gap.csv"],
          "BTC": ["btc_5m_1year.csv", "btc_5m_2026_gap.csv"],
          "SOL": ["sol_5m_2024_2026.csv"]}
METRICS = {"ETH": ["TOTAL_ETHUSDT_metrics_2021_2023.csv", "TOTAL_ETHUSDT_metrics_2024_2026.csv"],
           "BTC": ["TOTAL_BTCUSDT_metrics_2024_2026.csv"], "SOL": ["TOTAL_SOLUSDT_metrics_2024_2026.csv"]}
RNG = np.random.default_rng(20261024)


def _ms(col: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(col):
        return col.astype("int64")
    return pd.to_datetime(col, utc=True).astype("int64") // 10**6  # 문자열 = UTC 봉 시작(close_time 로 확인)


def load(coin: str) -> pd.DataFrame:
    ks = []
    for f in KLINES[coin]:
        k = pd.read_csv(D / f)
        k = k.rename(columns={k.columns[0]: "ts"})
        ks.append(pd.DataFrame({"ts": _ms(k.ts), "close": k.close, "vol": k.volume, "buy": k.taker_buy_base}))
    k = pd.concat(ks).drop_duplicates("ts").set_index("ts").sort_index()
    m = pd.concat([pd.read_csv(D / f, usecols=["create_time", "sum_open_interest"]) for f in METRICS[coin]])
    oi = pd.Series(m.sum_open_interest.values, index=_ms(m.create_time).values).groupby(level=0).last().sort_index()
    k = k[(k.index >= oi.index.min()) & (k.index + BAR <= oi.index.max())]
    # 정렬 점검: |ΔOI| 가 그 봉 거래량과 가장 맞는 위치가 shift 0 이어야 한다(OI 스냅샷 = create_time 시각)
    al = {}
    for s in (-1, 0, 1):
        d = oi.reindex(k.index + BAR * (1 + s)).values - oi.reindex(k.index + BAR * s).values
        al[s] = pd.Series(np.abs(d)).corr(pd.Series(k.vol.values), method="spearman")
    k["doi"] = oi.reindex(k.index + BAR).values - oi.reindex(k.index).values
    k["d"] = 2 * k.buy - k.vol
    return k, al


def hourly(k: pd.DataFrame) -> pd.DataFrame:
    v = k.dropna(subset=["doi"])
    usd = v.doi.abs() * v.close
    up, buy = v.doi > 0, v.d >= 0
    f = pd.DataFrame({"L": usd.where(buy, 0.0), "S": usd.where(~buy, 0.0),  # L = 신규 롱 + 숏 정리, S = 신규 숏 + 롱 정리
                      "NL": usd.where(up & buy, 0.0), "LC": usd.where(~up & ~buy, 0.0), "n": 1.0}, index=v.index)
    h = f.groupby(f.index // H_MS * H_MS).sum()
    grid = np.arange(k.index.min() // H_MS * H_MS, k.index.max() // H_MS * H_MS + H_MS, H_MS)
    h = h.reindex(grid, fill_value=0.0)
    last = k.close.groupby(k.index // H_MS * H_MS).last().reindex(grid).ffill()
    h["P"] = last.shift(1)  # P(t) = t 직전 봉 종가 (t = 시 정각)
    return h


def features(h: pd.DataFrame, wins_h: list[int]) -> pd.DataFrame:
    c = h[["L", "S", "n"]].cumsum()
    out = pd.DataFrame(index=h.index)
    lp = np.log(h.P)
    for w in wins_h:
        s = c - c.shift(w)  # [t−w, t) : 행 t 의 cumsum 은 시 t 를 포함하므로 한 칸 당긴다
        s = s.shift(1)
        lsh = s.L / (s.L + s.S)
        out[f"L{w}"] = lsh.where(s.n >= 0.9 * w * 12)
        out[f"R{w}"] = lp - lp.shift(w)
        out[f"F{w}"] = lp.shift(-w) - lp
    return out


def rank(x):
    return pd.Series(x).rank().values


def resid(y, xs):
    X = np.column_stack([np.ones(len(y))] + list(xs))
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b


def boot(stat, n: int, block: int, reps: int = 2000) -> np.ndarray:
    nb = int(np.ceil(n / block))
    out = np.empty(reps)
    for r in range(reps):
        st = RNG.integers(0, max(1, n - block), nb)
        idx = (st[:, None] + np.arange(block)).ravel()[:n]
        out[r] = stat(idx)
    return out


def summarize(est, bs):
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return {"est": round(float(est), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4), "p": round(float(max(p, 1 / len(bs))), 4), "n": None}


def tests(df: pd.DataFrame, W: int, H: int, block: int, med: float) -> dict:
    x = df[list(dict.fromkeys([f"L{W}", f"R{W}", f"R{H}", f"F{H}"]))].dropna()  # W == H 면 R 열이 겹친다
    L, RW, RH, F = (x[c].values for c in (f"L{W}", f"R{W}", f"R{H}", f"F{H}"))
    rL, rRW, rRH, rF = rank(L), rank(RW), rank(RH), rank(F)
    res = {}
    corr = lambda a, b, i: np.corrcoef(a[i], b[i])[0, 1]
    all_i = np.arange(len(x))
    for name, a, b in (("A_same", rL, rRW), ("B_pred", rL, rF), ("mom", rRW, rF)):
        r = summarize(corr(a, b, all_i), boot(lambda i: corr(a, b, i), len(x), block))
        res[name] = r
    ctrl = [rRW] if W == H else [rRW, rRH]
    e = resid(rL, ctrl)
    res["C_incr"] = summarize(corr(e, rF, all_i), boot(lambda i: corr(resid(rL[i], [c[i] for c in ctrl]), rF[i], np.arange(len(i))), len(x), block))
    for tr, mask in (("D_up", RW > 0), ("D_dn", RW < 0)):
        sd = (L < med) & mask
        ld = (L >= med) & mask
        st = lambda i: F[i][sd[i]].mean() - F[i][ld[i]].mean() if sd[i].any() and ld[i].any() else 0.0
        r = summarize(st(all_i), boot(st, len(x), block))
        r["n_short_dom"], r["n_long_dom"] = int(sd.sum()), int(ld.sum())
        res[tr] = r
    for v in res.values():
        v["n"] = len(x)
    return res


def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for j, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - j) * p))
        out[k] = round(run, 4)
    return out


def run(coin: str) -> dict:
    k, al = load(coin)
    assert max(al, key=al.get) == 0, f"{coin} OI 정렬이 shift 0 이 아님: {al}"
    h = hourly(k)
    rep = {"align_spearman_absdoi_vol": {str(s): round(v, 3) for s, v in al.items()},
           "range": [str(pd.Timestamp(h.index.min(), unit="ms")), str(pd.Timestamp(h.index.max(), unit="ms"))], "daily": {}, "intraday": {}}
    fd = features(h, [24, 168, 720])
    fd = fd[(fd.index % DAY_MS) == 0]
    fh = features(h, [1, 4])
    for tag, f, ws, block in (("daily", fd, [24, 168, 720], 30), ("intraday", fh, [1, 4], 168)):
        dev, test = f[f.index < TEST_FROM], f[f.index >= TEST_FROM]
        for W in ws:
            med = float(dev[f"L{W}"].median())
            for Hh in ws:
                key = f"W{W}h_H{Hh}h"
                rep[tag][key] = {"dev": tests(dev, W, Hh, block, med), "test": tests(test, W, Hh, block, med), "dev_median_L": round(med, 4)}
                print(coin, tag, key, "A", rep[tag][key]["test"]["A_same"]["est"], "B", rep[tag][key]["test"]["B_pred"]["est"],
                      "C", rep[tag][key]["test"]["C_incr"]["est"], flush=True)
    return rep


if __name__ == "__main__":
    # 자체점검: 잔차는 통제와 직교, holm 단조
    y, x = RNG.normal(size=200), RNG.normal(size=200)
    assert abs(np.corrcoef(resid(y + 2 * x, [x]), x)[0, 1]) < 1e-9
    assert holm({"a": .01, "b": .04, "c": .03}) == {"a": .03, "c": .06, "b": .06}
    OUT.mkdir(parents=True, exist_ok=True)
    reps = {c: run(c) for c in (sys.argv[1:] or ["ETH", "BTC", "SOL"])}
    if "ETH" in reps:  # ETH TEST 가족 Holm (B·C·D)
        ps = {f"{tag}:{key}:{t}": reps["ETH"][tag][key]["test"][t]["p"]
              for tag in ("daily", "intraday") for key in reps["ETH"][tag] for t in ("B_pred", "C_incr", "D_up", "D_dn")}
        reps["ETH_holm_test"] = holm(ps)
    (OUT / "report.json").write_text(json.dumps(reps, ensure_ascii=False, indent=1))
    print("wrote", OUT / "report.json")
