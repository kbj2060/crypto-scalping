#!/usr/bin/env python3
"""추세 규칙 × 매크로 «합의 필터» 탐색 백테스트 -- 사전등록 docs/prereg/trend_macro_agree_filter_20261028.md 그대로(2026-10-28).

기준 = 화면 추세 규칙(dashboard/trend_rule.py) 일 단위 백테스트 · 필터 = 매크로 Z_B4 와 엇갈리면 크기 ×0.5.
🔴판정 아님 -- 2020–23 은 매크로 부호를 정한 구간, 2024~ 는 이미 본 구간, 2026-06~ 는 2라운드 시험에 쓴 구간. 판정은 전진(2028-10).
  python scripts/research_trend_macro_agree_20261028.py --panel <panel_daily.parquet>
  python scripts/research_trend_macro_agree_20261028.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs/experiments/trend_macro_agree_20261028"
LOOKBACKS, VOL_N, TARGET, CAP, COST = (7, 14, 28, 56, 90), 20, 0.5, 2.0, 4e-4
DISAGREE_MULT = 0.5
PERIODS = (("2020-23(매크로 부호 적합 구간)", "2020-01-01", "2023-12-31"), ("2024-01~2026-05(이미 본 구간)", "2024-01-01", "2026-05-31"),
           ("2026-06~(시험에 쓴 구간·2번째)", "2026-06-01", None))


def trend_size(c: pd.Series) -> tuple[pd.Series, pd.Series]:
    """화면 규칙과 같은 식: s = 평균 sign(C_t − C_{t−L}) · 크기 = clip(s·0.5/σ20(연율, 표본 sd), ±2)."""
    s = sum(np.sign(c - c.shift(L)) for L in LOOKBACKS) / len(LOOKBACKS)
    vol = np.log(c).diff().rolling(VOL_N).std() * np.sqrt(365)
    return s, (s * TARGET / vol).clip(-CAP, CAP)


def macro_daily(z_week: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """금요일 τ 의 Z 를 다음 날부터 다음 τ 까지(τ 에 알려진 값)."""
    z = z_week.dropna().copy()
    z.index = z.index + pd.Timedelta(days=1)
    return z.reindex(z.index.union(idx)).ffill().reindex(idx)


def backtest(c: pd.Series, z_week: pd.Series) -> pd.DataFrame:
    s, b = trend_size(c)
    zd = macro_daily(z_week, c.index)
    m = pd.Series(np.where(zd.isna() | (np.sign(zd) == np.sign(s)), 1.0, DISAGREE_MULT), index=c.index)
    r = c.pct_change().shift(-1)                                   # t 종가에 정한 크기 → t+1 일 수익
    out = {}
    for name, pos in (("base", b), ("filter", b * m)):
        pos = pos.fillna(0.0)
        out[name] = pos * r - COST * pos.diff().abs().fillna(pos.abs())
    out["agree"] = (m == 1.0).astype(float)
    return pd.DataFrame(out).dropna()


def stats(p: pd.Series) -> dict:
    eq = (1 + p).cumprod()
    wk = p.resample("W-FRI").sum()
    return {"sharpe": float(p.mean() / p.std() * np.sqrt(365)) if p.std() > 0 else None, "ann_ret": float(p.mean() * 365),
            "mdd": float((eq / eq.cummax() - 1).min()), "cvar5_week": float(wk[wk <= wk.quantile(.05)].mean())}


def diff_ci(d: pd.Series, rng, block: int = 20, reps: int = 2000) -> list[float]:
    v, n = d.values, len(d)
    k = int(np.ceil(n / block))
    bs = [v[np.concatenate([np.arange(s, s + block) for s in rng.integers(0, n - block + 1, k)])[:n]].mean() * 365 for _ in range(reps)]
    return [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def main(panel_path: str) -> dict:
    import scripts.research_macro_regime_bias_20261028 as rb
    p = pd.read_parquet(panel_path)
    fz = json.loads(rb.FROZEN.read_text())
    w = rb.sample_week(p)
    rng = np.random.default_rng(20261028)
    res = {"prereg": "docs/prereg/trend_macro_agree_filter_20261028.md", "note": "탐색 -- 판정 아님", "assets": {}}
    for asset in ("ETH", "BTC"):
        zb, _ = rb.z_combo(rb.feats_for(w, rb.selected(fz, asset, "B4")), rb.selected(fz, asset, "B4"))
        bt = backtest(p[asset].dropna(), zb)
        ra = {}
        for name, lo, hi in PERIODS:
            x = bt.loc[lo:hi]
            ra[name] = {"days": int(len(x)), "agree_share": float(x.agree.mean()), "base": stats(x.base), "filter": stats(x["filter"]),
                        "diff_ann": float((x["filter"] - x.base).mean() * 365), "diff_ann_ci95": diff_ci(x["filter"] - x.base, rng)}
        res["assets"][asset] = ra
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def selftest() -> None:
    from dashboard.trend_rule import trend_payload
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    c = pd.Series(np.exp(np.cumsum(rng.normal(0, .03, 200))) * 100, idx)
    s, b = trend_size(c)
    tp = trend_payload(c.tolist(), [int(t.value // 1e6) for t in idx + pd.Timedelta(days=1)])
    assert abs(tp["signal"] - round(s.iloc[-1], 1)) < 1e-12, (tp["signal"], s.iloc[-1])   # 화면 규칙과 같은 신호(화면은 소수 1자리)
    assert abs(tp["size"] - b.iloc[-1]) < 5e-4, (tp["size"], b.iloc[-1])                  # 같은 크기(화면은 소수 3자리)
    zw = pd.Series([1.0, -1.0], pd.to_datetime(["2024-03-01", "2024-03-08"]))
    zd = macro_daily(zw, idx)
    assert np.isnan(zd["2024-03-01"]) and zd["2024-03-02"] == 1.0 and zd["2024-03-08"] == 1.0 and zd["2024-03-09"] == -1.0   # τ 다음 날부터
    down = pd.Series(100 * np.exp(-0.01 * np.arange(200) + rng.normal(0, .002, 200)), idx)   # 꾸준한 하락 → 추세 −1
    bt = backtest(down, pd.Series(1.0, pd.date_range("2024-01-05", periods=30, freq="W-FRI")))   # 매크로 늘 +1 → 엇갈림
    x = bt.loc["2024-04-15":]
    assert (x.agree == 0).all() and np.allclose(x["filter"], 0.5 * x["base"]), x.head()     # 엇갈리면 손익(비용 포함)이 정확히 절반
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        r = main(a.panel)
        for asset, ra in r["assets"].items():
            for name, v in ra.items():
                f = lambda d: f"샤프 {d['sharpe']:.2f} 연 {d['ann_ret']*100:+.1f}% MDD {d['mdd']*100:.1f}% 주CVaR {d['cvar5_week']*100:.1f}%"   # noqa: E731
                print(f"{asset} {name} {v['days']}일 합의 {v['agree_share']:.0%} | 기준 {f(v['base'])} | 필터 {f(v['filter'])} | 차 {v['diff_ann']*100:+.1f}%p [{v['diff_ann_ci95'][0]*100:+.1f},{v['diff_ann_ci95'][1]*100:+.1f}]")
