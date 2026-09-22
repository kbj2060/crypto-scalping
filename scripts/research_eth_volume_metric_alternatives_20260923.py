#!/usr/bin/env python3
"""«거래대금» 대안 — 거래량 계열 11종 정면 비교 (2026-09-23).

## 채점 과제를 대시보드의 실제 용도로 고정한다
거래대금(quote_volume)이 화면에서 하는 일은 «전환 탐지»다 —
`거래대금 z288 q90 AND 체결속도 z288 q90` 이면 「전환이 시작됐다」(app.js:2189).
그래서 같은 과제로 잰다: **앞 30분(6봉) 실현 레인지가 상위 25% 인가.**

## 🔴 공짜로 맞히는 축 두 개를 먼저 통제한다
  C1 `range_bp_now`  — 지금 봉 자신의 폭. (「가장 넓은 봉」 무모델 규칙이 모델을 이긴 전례)
  C2 `tod`           — 시각. 거래량은 시간대의 함수라 z288 은 «지금이 아시아장인가»를 상당 부분 맞힌다.
통제 후 수치 = C1 십분위 **안에서** 계산한 AUC 의 가중평균.

창은 **겹치지 않게** 센다(6봉 간격). 겹친 창으로 세면 검정력이 6배 부풀어 있다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parents[1]
KL = next(p for p in (HERE / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv",
                      Path("/home/kbj20/crypto-scalping/binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"))
          if p.exists())
OUT = HERE / "tmp/eth_band_vol_volume_alt_20260923"

H = 6            # 앞 30분
Z = 288          # 현행 화면과 같은 후행 창(24h)
TOD = 288 * 28   # 시간대 정규화 = 같은 시각의 28일 중앙값
BOOT, SEED = 400, 20260923


def zs(s: pd.Series, w: int = Z) -> pd.Series:
    return (s - s.rolling(w).mean()) / s.rolling(w).std().replace(0, np.nan)


def candidates(df: pd.DataFrame) -> dict[str, pd.Series]:
    qv, nt = df["quote_volume"], df["trades"]
    tbq, c = df["taker_buy_quote"], df["close"]
    ret = c.pct_change()
    signed = 2 * tbq - qv                      # 테이커 매수 − 매도 (거래대금 단위)
    # 시간대 정규화: 같은 시각(=같은 5분 슬롯)의 후행 28일 중앙값으로 나눈다.
    slot = df["timestamp"].dt.hour * 12 + df["timestamp"].dt.minute // 5
    tod_med = qv.groupby(slot).transform(lambda s: s.rolling(28, min_periods=7).median().shift())
    tod_med_n = nt.groupby(slot).transform(lambda s: s.rolling(28, min_periods=7).median().shift())
    return {
        "거래대금 z288(현행)":   zs(qv),
        "체결속도 z288(현행짝)": zs(nt),
        "거래대금/시간대중앙":   np.log(qv / tod_med.replace(0, np.nan)),
        "체결속도/시간대중앙":   np.log(nt / tod_med_n.replace(0, np.nan)),
        "평균체결크기 z288":     zs(qv / nt.replace(0, np.nan)),
        "테이커불균형|·|":       (signed.abs() / qv.replace(0, np.nan)),
        "CVD기울기 z288":        zs(signed.rolling(12).sum()),
        "VPIN(50봉)":            signed.abs().rolling(50).sum() / qv.rolling(50).sum().replace(0, np.nan),
        "Amihud z288":           zs(ret.abs() / qv.replace(0, np.nan)),
        "Kyle람다 z288":         zs((ret.abs() / np.sqrt(qv.replace(0, np.nan))).rolling(12).mean()),
        "거래대금 z2016":        zs(qv, 2016),
    }


def controls(df: pd.DataFrame) -> dict[str, pd.Series]:
    c, h, l = df["close"], df["high"], df["low"]
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    r2 = np.log(c / pc) ** 2
    return {"C1 현재봉폭 bp": (h - l) / c * 1e4,
            "C2 시각(코사인)": np.cos(2 * np.pi * (df["timestamp"].dt.hour * 60
                                                 + df["timestamp"].dt.minute) / 1440),
            "C3 ATR144%": tr.ewm(alpha=1 / 144, adjust=False).mean() / c,
            # C4 HAR-RV: 「변동성 확장 24h」 신호를 통째로 지운 기준선(09-21 컬링 §1)
            "C4 HAR-RV(1/12/288)": np.log(r2.rolling(12).mean().clip(lower=1e-12))}


def auc_ci(y, p, day, rng):
    u = np.unique(day); by = {d: np.flatnonzero(day == d) for d in u}; o = []
    for _ in range(BOOT):
        i = np.concatenate([by[d] for d in rng.choice(u, len(u), replace=True)])
        if y[i].min() != y[i].max():
            o.append(roc_auc_score(y[i], p[i]))
    return (round(float(np.percentile(o, 2.5)), 4), round(float(np.percentile(o, 97.5)), 4)) if o else (np.nan, np.nan)


def cond_auc(y, p, strat):
    """C1 십분위 «안에서» 잰 AUC 의 가중평균 = 봉폭을 통제한 변별력."""
    q = pd.qcut(strat, 10, labels=False, duplicates="drop")
    num = den = 0.0
    for b in np.unique(q[np.isfinite(q)]):
        m = q == b
        if y[m].min() != y[m].max() and m.sum() > 50:
            num += roc_auc_score(y[m], p[m]) * m.sum(); den += m.sum()
    return round(num / den, 4) if den else np.nan


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(KL, parse_dates=["timestamp"]).drop_duplicates("timestamp")
    df = df.sort_values("timestamp").reset_index(drop=True)
    c = df["close"]
    fwd = (df["high"].shift(-1).rolling(H).max().shift(-(H - 1))
           - df["low"].shift(-1).rolling(H).min().shift(-(H - 1))) / c * 1e4
    feats = {**candidates(df), **controls(df)}
    F = pd.DataFrame({k: pd.Series(np.asarray(v, float)) for k, v in feats.items()})
    F["y_fwd"] = fwd
    F["day"] = df["timestamp"].dt.floor("D").to_numpy()

    F = F.iloc[::H].copy()                       # 겹치지 않는 창만
    F = F.replace([np.inf, -np.inf], np.nan).dropna()
    y = (F["y_fwd"] >= F["y_fwd"].quantile(0.75)).to_numpy().astype(int)
    day = F["day"].to_numpy()
    print(f"[data] 겹치지 않는 창 {len(F):,}개 · {len(np.unique(day)):,}일 · "
          f"상위25% 경계 {F['y_fwd'].quantile(0.75):.0f}bp")

    rows = []
    for name in feats:
        p = F[name].to_numpy(float)
        a = roc_auc_score(y, p)
        if a < 0.5:                              # 부호는 정보가 아니다 — 방향만 맞춘다
            p, a = -p, 1 - a
        lo, hi = auc_ci(y, p, day, rng)
        ca = cond_auc(y, p, F["C1 현재봉폭 bp"].to_numpy(float))
        ca3 = cond_auc(y, p, F["C3 ATR144%"].to_numpy(float))
        # 게이트 배수: 상위 10% / 1% 에서 앞 30분 레인지가 전체 중앙값의 몇 배인가
        med = F["y_fwd"].median()
        g90 = F["y_fwd"][p >= np.quantile(p, 0.90)].median() / med
        g99 = F["y_fwd"][p >= np.quantile(p, 0.99)].median() / med
        rows.append({"지표": name, "AUC": round(a, 4), "CI": [lo, hi],
                     "C1통제후AUC": ca, "C3통제후AUC": ca3, "q90배수": round(float(g90), 2),
                     "q99배수": round(float(g99), 2)})
        print(f"  {name:22s} AUC {a:.4f} [{lo},{hi}]  C1통제 {ca}  C3통제 {ca3}  "
              f"q90 {g90:.2f}x  q99 {g99:.2f}x")

    # 현행 «2종 AND» 와 후보 대체본 비교
    def gate(p, q=0.90):
        return p >= np.quantile(p, q)
    med = F["y_fwd"].median()
    pairs = {
        "현행 AND(거래대금 z288 · 체결속도 z288)": ("거래대금 z288(현행)", "체결속도 z288(현행짝)"),
        "시간대정규화 AND(거래대금 · 체결속도)":   ("거래대금/시간대중앙", "체결속도/시간대중앙"),
        "거래대금 z288 · 평균체결크기":            ("거래대금 z288(현행)", "평균체결크기 z288"),
        "체결속도/시간대 · 테이커불균형":          ("체결속도/시간대중앙", "테이커불균형|·|"),
    }
    print("\n[2종 AND 게이트]")
    ands = []
    for label, (a_, b_) in pairs.items():
        m = gate(F[a_].to_numpy(float)) & gate(F[b_].to_numpy(float))
        r = {"게이트": label, "발동율%": round(100 * m.mean(), 2),
             "레인지배수": round(float(F["y_fwd"][m].median() / med), 2),
             "상위25%적중률": round(float(y[m].mean()), 4)}
        ands.append(r); print(f"  {label:42s} 발동 {r['발동율%']:5.2f}%  "
                              f"배수 {r['레인지배수']:.2f}x  적중 {r['상위25%적중률']:.3f}")
    # ── 결정타: 최고 통제군에 «더해서» 오르는가 (표본외) ──────────────────────
    # 통제군 = C1 현재봉폭 + C3 ATR144 + C4 HAR-RV. 여기에 후보 1개를 더해 AUC 증분을 본다.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    CTRL = ["C1 현재봉폭 bp", "C3 ATR144%", "C4 HAR-RV(1/12/288)"]
    tr = np.arange(len(F)) < int(len(F) * 0.30)
    fit = lambda cols: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)) \
        .fit(F[cols].to_numpy()[tr], y[tr]).predict_proba(F[cols].to_numpy()[~tr])[:, 1]
    base = roc_auc_score(y[~tr], fit(CTRL))
    print(f"\n[증분] 통제군(C1+C3+C4) 표본외 AUC = {base:.4f}")
    inc = []
    for name in candidates(df):
        a = roc_auc_score(y[~tr], fit(CTRL + [name]))
        inc.append({"지표": name, "통제군+지표": round(a, 4), "증분": round(a - base, 4)})
        print(f"  +{name:22s} {a:.4f}  증분 {a - base:+.4f}")

    (OUT / "part_b_volume_metrics.json").write_text(
        json.dumps({"single": rows, "and_gates": ands, "control_auc": round(base, 4),
                    "incremental": inc}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def _selfcheck():
    d = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=400, freq="5min"),
                      "close": 100.0, "high": 101.0, "low": 99.0,
                      "quote_volume": np.arange(1.0, 401), "trades": 10.0,
                      "taker_buy_quote": np.arange(1.0, 401) * 0.5})
    z = zs(d["quote_volume"])
    assert np.isnan(z.iloc[Z - 2]) and np.isfinite(z.iloc[-1]), "z288 창 경계"
    # 테이커 정확히 반반이면 불균형 0
    assert abs(candidates(d)["테이커불균형|·|"].iloc[-1]) < 1e-12
    # cond_auc: 층 안에서 완전 분리면 1.0, 층 «사이»로만 갈리면 0.5
    yy = np.array([0, 1, 0, 1] * 30); pp = np.tile([0.0, 1.0, 0.0, 1.0], 30)
    assert cond_auc(yy, pp, pd.Series(np.repeat([0, 1], 60))) == 1.0


if __name__ == "__main__":
    _selfcheck()
    raise SystemExit(main())
