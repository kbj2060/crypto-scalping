"""ETF 이후(2024-01-11~) 기준 매크로 추세 지표 판정 (2026-10-28).

사용자 «ETF 이후 기준으로 추세 지표로 쓸 수 있는 것만» → «반전·신규 관계라도 ETF 이후에 꾸준히 같은 부호면 쓸 수 있는 것으로».
입력: research_macro_crypto_panel_20261027.py 가 만든 panel_daily.parquet (주간 W-FRI 표본은 상관 스크립트의 sample() 그대로).

관계 두 가지 (BTC·ETH 각각):
  동행13  corr(코인 13주 변화, 지표 13주 변화)                     -- «지금 큰 흐름이 어느 쪽인가» 확인용
  선행13  corr(지표 13주 변화 [1주 공표 시차], 코인 다음 13주 수익)  -- «다음 13주 편향»
판정 (2020–23 부호는 판정에 쓰지 않는다 -- 참고 표시 rel 만):
  사용 가능 = ETF 이후 |r|≥0.3 · 순환이동 순열 p<0.05 · 전반(2024-01~2025-05)·후반(2025-06~) 같은 부호 각 |r|≥0.15
             · 2024-06~11(엔캐리 청산·대선 랠리) 빼도 같은 부호 · 52주 롤링 상관 75%↑ 같은 부호
  관찰     = (p<0.10 또는 |r|≥0.3) · 반쪽 같은 부호 · 제외해도 같은 부호
  불가     = 그 밖
🔴 표본내 상관이다(ETF 이후 ≈2.8년 · 13주 독립 관측 ≈11). 표본외 검증 전 -- 승격 근거 아님.

  python scripts/research_macro_trend_p3_verdict_20261028.py
  → docs/experiments/macro_crypto_correlation_20261027/L_p3_trend_verdict.csv
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P3 = ("2024-01-12", "2026-12-31")
P3A = ("2024-01-12", "2025-05-30")
P3B = ("2025-06-06", "2026-12-31")
P2 = ("2020-01-03", "2024-01-05")
EXCL = ("2024-06-01", "2024-11-30")

# 이름 -> (패널 열, 변환). logpct = 100·ln, bp = 100·값(퍼센트 포인트 → bp), raw = 값 그대로
SERIES = {
    "S&P500": ("SPX", "logpct"), "KOSPI": ("KOSPI", "logpct"), "금": ("GOLD", "logpct"), "브렌트": ("DCOILBRENTEU", "logpct"),
    "달러지수": ("DTWEXBGS", "logpct"), "USD/JPY": ("DEXJPUS", "logpct"), "USD/CNY": ("DEXCHUS", "logpct"),
    "EUR/USD": ("DEXUSEU", "logpct"), "원/달러": ("DEXKOUS", "logpct"), "2Y금리": ("DGS2", "bp"), "10Y금리": ("DGS10", "bp"),
    "실질금리": ("DFII10", "bp"), "기대인플레": ("T10YIE", "bp"), "2s10s": ("T10Y2Y", "bp"), "기간프리미엄": ("THREEFYTP10", "bp"),
    "연방기금": ("DFF", "bp"), "VIX": ("VIXCLS", "logpct"), "Baa스프레드": ("BAA10Y", "bp"), "HY스프레드": ("BAMLH0A0HYM2", "bp"),
    "STLFSI": ("STLFSI4", "raw"), "순유동성": ("NETLIQ", "logpct"), "연준BS": ("WALCL", "logpct"), "지준": ("WRESBAL", "logpct"),
    "TGA": ("WTREGEN", "logpct"), "RRP": ("RRPONTSYD", "raw"), "스테이블코인": ("STABLE_SUPPLY", "logpct"),
}


def load_sampler():
    spec = importlib.util.spec_from_file_location("corr", ROOT / "scripts/research_macro_crypto_correlation_20261027.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.sample


def level(w: pd.DataFrame, col: str, how: str) -> pd.Series:
    s = w[col]
    return 100 * np.log(s.where(s > 0)) if how == "logpct" else (100 * s if how == "bp" else s)


def corr(d: pd.DataFrame, lo: str, hi: str) -> float:
    d = d.loc[lo:hi].dropna()
    return float(d.corr().iloc[0, 1]) if len(d) > 20 else float("nan")


def perm_p(d: pd.DataFrame, gap: int, rng: np.random.Generator, n: int = 3000) -> float:
    """레짐 지속성을 보존한 순환이동 순열검정(양측). gap = 겹침 길이보다 크게 민다."""
    d = d.loc[P3[0]:P3[1]].dropna()
    x, y = d.iloc[:, 0].values, d.iloc[:, 1].values
    obs = np.corrcoef(x, y)[0, 1]
    null = [np.corrcoef(np.roll(x, k), y)[0, 1] for k in rng.integers(gap + 4, len(x) - gap - 4, n)]
    return float(np.mean(np.abs(null) >= abs(obs)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(ROOT / "docs/experiments/macro_crypto_correlation_20261027/panel_daily.parquet"))
    ap.add_argument("--out", default=str(ROOT / "docs/experiments/macro_crypto_correlation_20261027/L_p3_trend_verdict.csv"))
    a = ap.parse_args()
    w = load_sampler()(pd.read_parquet(a.panel), "W")
    rng = np.random.default_rng(11)
    rows = []
    for asset in ("BTC", "ETH"):
        c = 100 * np.log(w[asset])
        for kind in ("동행13", "선행13"):
            for name, (col, how) in SERIES.items():
                x = level(w, col, how)
                if kind == "동행13":
                    d, gap = pd.concat([c.diff(13), x.diff(13)], axis=1), 13
                else:
                    d, gap = pd.concat([x.diff(13).shift(1), c.shift(-13) - c], axis=1), 26
                r3, ra, rb, r2 = corr(d, *P3), corr(d, *P3A), corr(d, *P3B), corr(d, *P2)
                dx = d.loc[P3[0]:P3[1]]
                rx = float(dx.drop(dx.loc[EXCL[0]:EXCL[1]].index).dropna().corr().iloc[0, 1])
                p = perm_p(d, gap, rng)
                sg = np.sign(r3)
                dd = d.loc[P3[0]:P3[1]].dropna()
                rc = dd.iloc[:, 0].rolling(52).corr(dd.iloc[:, 1]).dropna()
                roll = float((np.sign(rc) == sg).mean()) if len(rc) else float("nan")
                halves = np.sign(ra) == sg and np.sign(rb) == sg
                if abs(r3) >= 0.3 and p < 0.05 and halves and min(abs(ra), abs(rb)) >= 0.15 and np.sign(rx) == sg and roll >= 0.75:
                    verdict = "사용 가능"
                elif (p < 0.10 or abs(r3) >= 0.3) and halves and np.sign(rx) == sg:
                    verdict = "관찰"
                else:
                    verdict = "불가"
                rel = "비교 불가" if not np.isfinite(r2) else ("신규" if abs(r2) < 0.15 else ("유지" if np.sign(r2) == sg else "반전"))
                rows.append(dict(asset=asset, kind=kind, indicator=name, series=col, transform=how, verdict=verdict, vs_2020_23=rel,
                                 r_p3=r3, perm_p=p, r_first_half=ra, r_second_half=rb, r_excl_2024h2=rx, roll52_same_sign=roll,
                                 r_2020_23=r2, n_weeks=int(len(dd))))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False, float_format="%.4f")
    print(df[df.verdict != "불가"][["asset", "kind", "indicator", "verdict", "vs_2020_23", "r_p3", "perm_p"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
