#!/usr/bin/env python3
"""코인글래스식 가공 지표 3종 쓸모 검정 (2026-11-16, 사용자 «3개 바로 돌려»).

질문(용도 정확도): 지표가 **위험 레이더식 HAR(지난 변동성 + 시각×주말)** 이 이미 아는 것 말고 다음 변동을 더 알려 주는가.
  #5 CDRI식 위험 점수 = OI 수준·|펀딩|·레버리지 대리(OI ÷ 24h 거래대금)·롱숏 쏠림 |log 계정비|·IV(DVOL, ETH·BTC)·거래량 열기
       각 30일 분위(인과) 같은 비중 평균. 24h 청산은 긴 이력이 없어 뺐다(코인글래스 가중치 비공개 -> 같은 비중 재구성).
  #8 코인 횡단 = |1h OI 변화| · 1h 거래대금 ÷ 직전 24h 시간 평균 -> 패널 증분 + «다음 1h 가장 크게 움직일 3코인 고르기».
  #9 무기한/현물 거래대금 비 = log(선물 24h ÷ 현물 24h) 와 1h 판, 각 30일 평균 대비 -> 변동 증분 + 다음 4h 방향 IC.
분할: 학습 2024-01-08~2025-12-31 · 검증 2026-01~05 · 시험 2026-06-01~10-31. 20코인 바이낸스 USDT 무기한, 시간봉(봉 끝 t).
인과: 피쳐 = t 이전 5분봉·metrics(공개 시각 = create_time+5분, 아카이브는 버킷 시작 라벨)·펀딩(calc_time)·DVOL(봉 시작+1h).
판정: 시험 ΔQLIKE(HAR+X − HAR) 일 블록 부트스트랩 단측 p · 6칸(3세트×2지평) Holm(α .10) 통과 · 위약(피쳐 원형 이동 50회) 10분위보다 좋음
  · 상대 개선 ≤ −0.5% 면 «통과», 실익 하한만 못 넘으면 «형식 통과·실익 없음», 점추정만 음수면 «통과 후보».
  양성 대조(다음 1h 실현 변동 + 잡음)는 채점 산술만 확인한다(시점은 selftest 몫).
  성분 탐색(components) 은 사후 탐색 -- 12칸 Bonferroni p 를 함께 적는다.
  (11-16 독립 검토 반영: Holm·실익 하한 · 성분/IV 코드 수록 · metrics ffill 10분 한도 · 분할 경계 4h 정제.)
  python scripts/research_coinglass_indicators_20261116.py --selftest
  python scripts/research_coinglass_indicators_20261116.py   # -> tmp/coinglass_ind_20261116/result.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = Path("/home/kbj20/crypto-scalping/data/vision_cg_20261116")
OUT = ROOT / "tmp" / "coinglass_ind_20261116"
SYMS = "BTC ETH SOL XRP DOGE BNB ADA AVAX LINK LTC DOT BCH TRX SUI NEAR APT OP ARB FIL ATOM".split()
TRAIN = ("2024-01-08", "2025-12-31 19:00")      # 4h 목표가 다음 구간으로 넘어가지 않게(정제)
VAL = ("2026-01-01", "2026-05-31 19:00")
HOLM_ALPHA, MIN_REL_PCT = 0.10, -0.5
TEST = ("2026-06-01", "2026-10-31 19:00")          # 4h 목표가 10-31 안에 끝나게
H = pd.Timedelta("1h")
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qv", "n", "tbv", "tbqv", "ignore"]


def _zips(pattern: str, names: list[str] | None = None) -> pd.DataFrame:
    out = []
    for p in sorted(glob.glob(pattern)):
        with zipfile.ZipFile(p) as z:
            with z.open(z.namelist()[0]) as fh:
                d = pd.read_csv(fh, header=None, names=names) if names else pd.read_csv(fh)
        if names and not str(d.iloc[0, 0]).replace(".", "").isdigit():   # 새 파일은 머리줄이 있다
            d = d.iloc[1:]
        out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def klines(kind: str, s: str) -> pd.DataFrame:
    d = _zips(str(SRC / kind / f"{s}USDT-5m-*.zip"), KCOLS)
    t = d["open_time"].astype("int64")
    t = np.where(t > 10**14, t // 1000, t)                       # 현물 2025~ 는 마이크로초
    d = pd.DataFrame({"t": pd.to_datetime(t, unit="ms"), "close": d["close"].astype(float), "qv": d["qv"].astype(float)})
    return d.drop_duplicates("t").set_index("t").sort_index()


def hourly(s: str) -> pd.DataFrame:
    """시간봉 끝 t 행 = [t-1h, t) 의 5분봉 요약 + 그 시각까지 공개된 맥락."""
    f, sp = klines("fut", s), klines("spot", s)
    r2 = np.log(f["close"]).diff() ** 2
    g = lambda x: x.groupby(x.index.floor("1h")).sum(min_count=12)   # noqa: E731 -- 12칸 다 있어야
    hr = pd.DataFrame({"rv": g(r2), "fqv": g(f["qv"]), "sqv": g(sp["qv"]).reindex(g(f["qv"]).index),
                       "close": f["close"].groupby(f.index.floor("1h")).last()})
    hr.index = hr.index + H                                          # 라벨 = 봉 끝(공개 시각)
    m = _zips(str(SRC / "met" / f"{s}USDT-metrics-*.zip"))
    m["known"] = pd.to_datetime(m["create_time"]) + pd.Timedelta("5min")
    m = m.drop_duplicates("known").set_index("known").sort_index()
    tol = pd.Timedelta("10min")                                       # 빠진 날의 옛 값을 끌고 오지 않는다
    hr["oi"] = m["sum_open_interest_value"].astype(float).reindex(hr.index, method="ffill", tolerance=tol)
    hr["lsr"] = m["count_long_short_ratio"].astype(float).reindex(hr.index, method="ffill", tolerance=tol)
    fu = _zips(str(SRC / "fund" / f"{s}USDT-fundingRate-*.zip"))
    fu = fu.assign(t=pd.to_datetime(fu["calc_time"].astype("int64"), unit="ms")).drop_duplicates("t").set_index("t").sort_index()
    hr["fund"] = fu["last_funding_rate"].astype(float).reindex(hr.index, method="ffill")
    if s in ("ETH", "BTC"):
        dv = pd.read_csv(SRC / "dvol" / f"{s}_dvol_hourly.csv", parse_dates=["timestamp"])
        hr["iv"] = dv.set_index(dv["timestamp"] + H)["close"].reindex(hr.index, method="ffill")
    else:
        hr["iv"] = np.nan
    hr["sym"] = s
    return hr


def pct30(x: pd.Series) -> pd.Series:
    """직전 720시간(자기 포함, t 공개분) 중 지금 값의 분위 0~100."""
    return x.rolling(720, min_periods=240).rank(pct=True) * 100


def features(hr: pd.DataFrame) -> pd.DataFrame:
    d = hr.copy()
    lrv = lambda w: np.log(d["rv"].rolling(w).mean() + 1e-12)   # noqa: E731
    d["h1"], d["h4"], d["h24"], d["h168"] = lrv(1), lrv(4), lrv(24), lrv(168)
    fwd = lambda k: d["rv"][::-1].rolling(k).sum()[::-1].shift(-1)   # noqa: E731 -- (t, t+k h] = 다음 k 개 봉
    d["y1"], d["y4"] = fwd(1), fwd(4)
    d["ret4"] = np.log(d["close"].shift(-4) / d["close"])
    d["hw"] = d.index.hour + 24 * (d.index.dayofweek >= 5)        # 봉 끝 시각 × 주말 48칸
    q24 = d["fqv"].rolling(24).sum()
    comp = {"c_oi": pct30(d["oi"]), "c_fund": pct30(d["fund"].abs()), "c_lev": pct30(d["oi"] / q24),
            "c_ls": pct30(np.log(d["lsr"]).abs()), "c_iv": pct30(d["iv"]),
            "c_vol": pct30(q24 / d["fqv"].rolling(24 * 7).sum().shift(24) * 7)}
    for k, v in comp.items():
        d[k] = v
    d["cdri"] = pd.DataFrame(comp).mean(axis=1, skipna=True)
    d["oi_abs1h"] = np.log(d["oi"] / d["oi"].shift(1)).abs()
    d["vol_chg1h"] = np.log(d["fqv"] / d["fqv"].rolling(24).mean().shift(1))
    ratio24 = np.log(q24 / d["sqv"].rolling(24).sum())
    ratio1 = np.log(d["fqv"] / d["sqv"])
    d["ps24"] = ratio24 - ratio24.rolling(720, min_periods=240).mean()
    d["ps1"] = ratio1 - ratio1.rolling(720, min_periods=240).mean()
    return d


BASE = ["h1", "h4", "h24", "h168"]
SETS = {"cdri": ["cdri"], "xs": ["oi_abs1h", "vol_chg1h"], "ps": ["ps24", "ps1"]}


def design(d: pd.DataFrame, cols: list[str], syms: list[str]) -> np.ndarray:
    hw = np.eye(48)[d["hw"].to_numpy()][:, 1:]
    fe = (d["sym"].to_numpy()[:, None] == np.array(syms)[None, 1:]).astype(float)
    return np.column_stack([np.ones(len(d)), d[BASE + cols].to_numpy(float), hw, fe])


def qlike(rv: np.ndarray, pred_log: np.ndarray, s2: float) -> np.ndarray:
    h = np.exp(pred_log + s2 / 2)
    x = rv / h
    return x - np.log(x) - 1


def fit_eval(P: pd.DataFrame, cols: list[str], y: str, syms: list[str]) -> dict[str, pd.DataFrame]:
    """학습 구간 OLS -> 구간별 시간×코인 손실(QLIKE)."""
    tr = P.loc[TRAIN[0]:TRAIN[1]]
    X, Y = design(tr, cols, syms), np.log(tr[y].to_numpy())
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    s2 = float(np.var(Y - X @ b))
    out = {}
    for nm, (a, z) in {"val": VAL, "test": TEST}.items():
        q = P.loc[a:z]
        out[nm] = pd.DataFrame({"loss": qlike(q[y].to_numpy(), design(q, cols, syms) @ b, s2),
                                "pred": design(q, cols, syms) @ b, "sym": q["sym"].to_numpy()}, index=q.index)
    return out


def boot_upper(diff_by_day: np.ndarray, q: float = 0.90, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """일 단위 차이(이미 일 평균) -> (평균, 단측 q 상한)."""
    m, up, _ = boot(diff_by_day, q, n, seed)
    return m, up


def boot(diff_by_day: np.ndarray, q: float = 0.90, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(평균, 단측 q 상한, 단측 p = 부트스트랩 평균이 0 이상인 비율)."""
    rng = np.random.default_rng(seed)
    bs = np.array([diff_by_day[rng.integers(0, len(diff_by_day), len(diff_by_day))].mean() for _ in range(n)])
    return float(diff_by_day.mean()), float(np.quantile(bs, q)), float((bs >= 0).mean())


def holm(p: dict[str, float], alpha: float = HOLM_ALPHA) -> dict[str, bool]:
    """Holm 단계 하강: 작은 p 부터 alpha/(m-i) 와 비교, 처음 실패하면 그 뒤 전부 실패."""
    keys = sorted(p, key=p.get)
    out, ok = {}, True
    for i, k in enumerate(keys):
        ok = ok and p[k] <= alpha / (len(keys) - i)
        out[k] = ok
    return out


def panel(syms: list[str]) -> pd.DataFrame:
    fr = [features(hourly(s)) for s in syms]
    P = pd.concat(fr).sort_index().replace([np.inf, -np.inf], np.nan)   # 거래대금·OI 0 인 봉의 log
    print("inf->nan", {c: int(P[c].isna().sum()) for c in sum(SETS.values(), [])}, flush=True)
    need = BASE + ["y1", "y4", "hw"] + sum(SETS.values(), [])
    P = P.dropna(subset=need)
    return P[(P["y1"] > 0) & (P["y4"] > 0) & (P["rv"] > 0)]


def delta(P, cols, y, syms, base_res=None):
    b = base_res or fit_eval(P, [], y, syms)
    x = fit_eval(P, cols, y, syms)
    res = {}
    for nm in ("val", "test"):
        assert x[nm].index.equals(b[nm].index) and (x[nm]["sym"].to_numpy() == b[nm]["sym"].to_numpy()).all()
        d = (x[nm]["loss"] - b[nm]["loss"]).groupby(x[nm].index.floor("1D")).mean().to_numpy()
        m, up, p = boot(d)
        res[nm] = {"dqlike": m, "up90": up, "p1": p, "base": float(b[nm]["loss"].mean()), "rel_pct": 100 * m / float(b[nm]["loss"].mean()),
                   "n_rows": int(len(x[nm])), "n_days": int(len(d))}
    return res, x


def placebo(P, cols, y, syms, base_res, n=100, seed=1):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        Q = P.copy()
        for s in syms:
            msk = (Q["sym"] == s).to_numpy()
            k = int(rng.integers(24 * 30, msk.sum() - 24 * 30))
            Q.loc[msk, cols] = np.roll(Q.loc[msk, cols].to_numpy(), k, axis=0)
        r, _ = delta(Q, cols, y, syms, base_res)
        vals.append(r["test"]["dqlike"])
    return np.array(vals)


def pick3(base: pd.DataFrame, alt: pd.DataFrame, P: pd.DataFrame, y: str) -> dict:
    """매시 예측 상위 3코인의 실제 다음 변동(log) − 그 시각 20코인 평균(log). alt − base 를 일 블록으로."""
    q = P.loc[base.index.min():base.index.max(), [y, "sym"]]
    act = pd.Series(np.log(q[y].to_numpy()), index=pd.MultiIndex.from_arrays([q.index, q["sym"]]))
    def top(r):
        k = r.rename_axis("t").reset_index()
        k = k[k.groupby("t")["sym"].transform("size") >= 15]
        k["a"] = act.reindex(pd.MultiIndex.from_arrays([k["t"], k["sym"]])).to_numpy()
        k["a"] = k["a"] - k.groupby("t")["a"].transform("mean")
        return k.sort_values("pred", ascending=False).groupby("t").head(3).groupby("t")["a"].mean()
    tb, ta = top(base), top(alt)
    d = (ta - tb).groupby(ta.index.floor("1D")).mean().dropna().to_numpy()
    m, up = boot_upper(-d)                                         # 상한 < 0 이면 alt 가 낫다
    return {"base_excess_log": float(tb.mean()), "alt_excess_log": float(ta.mean()), "gain_log": -m, "gain_lo90": -up}


def ic_dir(P: pd.DataFrame, col: str) -> dict:
    out = {}
    for nm, (a, z) in {"train": TRAIN, "val": VAL, "test": TEST}.items():
        q = P.loc[a:z, [col, "ret4"]].dropna()
        q = q[q.index.hour % 4 == 0]                                # 4h 겹침 제거
        a_, b_ = q[col].rank().to_numpy(), q["ret4"].rank().to_numpy()
        day = q.index.floor("1D").to_numpy()
        _, inv = np.unique(day, return_inverse=True)
        rows = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(500):
            ix = np.concatenate([rows[k] for k in rng.integers(0, len(rows), len(rows))])
            bs.append(np.corrcoef(a_[ix], b_[ix])[0, 1])
        out[nm] = {"ic": float(np.corrcoef(a_, b_)[0, 1]), "lo5": float(np.quantile(bs, .05)),
                   "hi95": float(np.quantile(bs, .95)), "n": int(len(q))}
    return out


COMP = ["c_oi", "c_fund", "c_lev", "c_ls", "c_vol", "c_iv"]


def components(P: pd.DataFrame, syms: list[str]) -> dict:
    """사후 성분 탐색: CDRI 성분 하나씩 + 코인 횡단 둘. 같은 행(성분 결측 제거)에서 기준선을 다시 맞춘다.
    c_iv 는 ETH·BTC 만(기준선도 두 코인으로) + 위약 50회. 6성분×2지평 = 12칸 Bonferroni p 병기."""
    Q = P.dropna(subset=[c for c in COMP if c != "c_iv"])
    Qiv = P[P["sym"].isin(["ETH", "BTC"])].dropna(subset=COMP)
    out = {}
    for y in ("y1", "y4"):
        b, biv = fit_eval(Q, [], y, syms), fit_eval(Qiv, [], y, syms)
        for c in COMP + ["oi_abs1h", "vol_chg1h"]:
            D, bb = (Qiv, biv) if c == "c_iv" else (Q, b)
            r, _ = delta(D, [c], y, syms, bb)
            r["p_bonf12"] = min(1.0, r["test"]["p1"] * 12) if c in COMP else None
            if c == "c_iv":
                pl = placebo(D, [c], y, ["BTC", "ETH"], bb, n=50)
                r["placebo_test_p10"], r["placebo_rank"] = float(np.quantile(pl, .10)), float((pl <= r["test"]["dqlike"]).mean())
                r["note"] = "기준선 = ETH·BTC 두 코인 HAR(20코인 칸과 직접 비교 금지)"
            out[f"{y}_{c}"] = r
            print("comp", y, c, round(r["test"]["rel_pct"], 3), "p", round(r["test"]["p1"], 4), flush=True)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    syms = [s for s in SYMS if glob.glob(str(SRC / "met" / f"{s}USDT-metrics-*.zip"))]
    cache = OUT / "panel.parquet"
    P = pd.read_parquet(cache) if cache.exists() else panel(syms)
    if not cache.exists():
        P.to_parquet(cache)
    print("panel", P.shape, P.index.min(), P.index.max(), "coins", P["sym"].nunique(), flush=True)
    R: dict = {"syms": syms, "rows": int(len(P)), "split": {"train": TRAIN, "val": VAL, "test": TEST}}
    rng = np.random.default_rng(7)
    for y in ("y1", "y4"):
        base = fit_eval(P, [], y, syms)
        P["pos_ctrl"] = np.log(P[y]) + rng.normal(0, 1.0, len(P))   # 양성 대조: 미래 변동 + 큰 잡음
        pc, _ = delta(P, ["pos_ctrl"], y, syms, base)
        R[f"{y}_positive_control"] = pc["test"]
        for nm, cols in SETS.items():
            r, alt = delta(P, cols, y, syms, base)
            pl = placebo(P, cols, y, syms, base, n=50)
            r["placebo_test_p10"] = float(np.quantile(pl, .10))
            r["placebo_rank"] = float((pl <= r["test"]["dqlike"]).mean())   # 작을수록 진짜가 위약보다 좋다
            if nm == "xs" and y == "y1":
                r["pick3_test"] = pick3(base["test"], alt["test"], P, y)
                r["pick3_val"] = pick3(base["val"], alt["val"], P, y)
            R[f"{y}_{nm}"] = r
            print(y, nm, json.dumps(r["test"], ensure_ascii=False), "위약순위", r["placebo_rank"], flush=True)
    cells = [f"{y}_{nm}" for y in ("y1", "y4") for nm in SETS]
    hp = holm({c: R[c]["test"]["p1"] for c in cells})
    for c in cells:
        t = R[c]["test"]
        formal = hp[c] and t["dqlike"] < R[c]["placebo_test_p10"]
        R[c]["holm_pass"] = hp[c]
        R[c]["verdict"] = ("통과" if formal and t["rel_pct"] <= MIN_REL_PCT else "형식 통과·실익 없음" if formal else
                           "통과 후보" if t["dqlike"] < 0 else "불통과")
        print(c, R[c]["verdict"], "p", round(t["p1"], 4), "rel", round(t["rel_pct"], 3), flush=True)
    R["components"] = components(P, syms)
    tst = P.loc[TEST[0]:TEST[1]]
    R["cdri_component_ic_y4_test"] = {c: float(tst[c].corr(np.log(tst["y4"]) - tst["h24"], method="spearman"))
                                      for c in ["c_oi", "c_fund", "c_lev", "c_ls", "c_iv", "c_vol", "cdri"]}
    R["ps_dir"] = {c: ic_dir(P, c) for c in ("ps24", "ps1")}
    (OUT / "result.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(R["ps_dir"], ensure_ascii=False), flush=True)


def selftest() -> None:
    # QLIKE 는 정답 분산에서 최소
    rv = np.full(1000, 2.0)
    assert qlike(rv, np.log(rv), 0).mean() < 1e-12 < qlike(rv, np.log(rv * 1.3), 0).mean()
    # Holm: 6칸이면 p .093 은 셋째 단계 α/4 = .025 에서 실패(3칸이면 마지막 α/1 로 통과)
    assert holm({"a": .0005, "b": .001, "c": .093, "d": .18, "e": .36, "f": .40}) == \
        {"a": True, "b": True, "c": False, "d": False, "e": False, "f": False}
    assert holm({"a": .001, "b": .02, "c": .093})["c"]
    # 피쳐·목표 경계: 봉 끝 t 행의 h1 은 [t-1h,t) · y1 은 [t, t+1h)
    idx = pd.date_range("2025-01-01", periods=24 * 40, freq="1h")
    hr = pd.DataFrame({"rv": np.arange(1, len(idx) + 1, dtype=float), "fqv": 1.0, "sqv": 1.0, "close": 1.0,
                       "oi": 1.0, "lsr": 1.0, "fund": 0.0, "iv": np.nan, "sym": "X"}, index=idx)
    d = features(hr)
    i = 500
    assert np.isclose(np.exp(d["h1"].iloc[i]), hr["rv"].iloc[i]) and d["y1"].iloc[i] == hr["rv"].iloc[i + 1]
    assert d["y4"].iloc[i] == hr["rv"].iloc[i + 1:i + 5].sum()
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else main()
    sys.exit(0)
