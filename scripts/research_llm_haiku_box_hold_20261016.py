#!/usr/bin/env python3
"""**Haiku 로 박스 유지 확률을 높일 수 있나** — 단독(A) · 배포 모델 확률 보정(B) vs 로지스틱 6칸(배포) · HGB 개선판 (2026-10-16).

왜: 사용자 «hgb 뿐만 아니라 haiku로 업그레이드 해달라는거였어» — 박스 유지 확률(dashboard/box_hold.py)을 Haiku 로 올릴 수 있는지.
표본: 4h 결정 행(매시 59분) 중 2025-01-01~2026-10-14, 서로 ≥8h(4h 정답 창·묶음 안 겹침), 개선 실험 특징이 다 있는 행 360.
  한 시점에 30분·1h·4h 세 박스를 같이 묻는다(같은 분이라 30분·1h 결정 행도 있다).
입력(사람이 읽는 숫자, 날짜·절대가격 없음): 박스 반폭 bp · 반폭/(σ24√W) · 창 안 위치 · 추세성 · 압축 · 24h 대비 창폭 · 창 거래량 비 ·
  |테이커 불균형| · 다음 W분 요일×시각 계절 · 변동성 기간구조(5·15·60·240분/24h) · 파킨슨 · 1h 최대 점프 · 시각·요일 · 예정 발표까지 분 ·
  과거(2022~24) 평균 유지율. B 는 여기에 배포 모델 확률을 더하고 «놓친 것이 보이면 고쳐라».
비교: F 로지스틱(배포 계수와 같은 학습) · HGB 개선판(research_eth_box_hold_improve_20261016 의 «−X−7일», 4h 는 F 6칸) — 둘 다 2022~24 학습.
Haiku 호출 = Claude Code 하위 에이전트(model haiku), 묶음 파일 하나만 읽고 JSON 배열 반환 → out_{A,B}/batch_XX.json.
  python scripts/research_llm_haiku_box_hold_20261016.py build
  python scripts/research_llm_haiku_box_hold_20261016.py eval
  python scripts/research_llm_haiku_box_hold_20261016.py --selfcheck
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("imp", ROOT / "scripts/research_eth_box_hold_improve_20261016.py")
imp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(imp)
OUT = ROOT / "tmp/haiku_box_20261016"
BDIR = ROOT / "tmp/haiku_box_20261016_inputs"          # 🔴 에이전트가 읽는 폴더 — 정답(OUT/gold.csv)과 분리
N, BATCH, GAP_H = 360, 60, 8
SC = list(imp.SCALES)                                    # 30m 1h 4h


def frame() -> pd.DataFrame:
    """결정 행 + 개선 특징, 시점(ts) 하나에 세 박스를 옆으로 붙인 표."""
    D = pd.read_parquet(imp.DEC)
    D = D[D.scale.isin(SC)].reset_index(drop=True)
    X = pd.read_parquet(imp.OUT / "features.parquet").replace([np.inf, -np.inf], np.nan)
    A = pd.concat([D[["scale", "ts", "hold"] + imp.F_BASE], X], axis=1)
    return A


def model_probs(A: pd.DataFrame) -> pd.DataFrame:
    """F 로지스틱·HGB 개선판, 2022~24 학습 → 모든 행 예측(척도별)."""
    noX = imp.F_BASE + [c for c in A.columns if c[:2] in ("V_", "R_", "Q_", "S_", "M_")]
    lean = [c for c in noX if c not in ("V_lsig7d", "Q_volz")]
    out = pd.DataFrame(index=A.index, columns=["p_F", "p_HGB"], dtype=float)
    for s in SC:
        g = A[A.scale == s]
        cols = lean if s != "4h" else imp.F_BASE
        tr = g[g.ts < "2025-01-01"].dropna(subset=lean)
        f = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(tr[imp.F_BASE], tr.hold)
        h = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500,
                                           l2_regularization=1.0, random_state=0).fit(tr[cols], tr.hold)
        ok = g.dropna(subset=lean)
        out.loc[ok.index, "p_F"] = f.predict_proba(ok[imp.F_BASE])[:, 1]
        out.loc[ok.index, "p_HGB"] = h.predict_proba(ok[cols])[:, 1]
    return out


def wide(A: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    B = pd.concat([A, P], axis=1)
    parts = []
    for s in SC:
        g = B[B.scale == s].set_index("ts").drop(columns="scale").add_suffix(f"@{s}")
        parts.append(g)
    W = pd.concat(parts, axis=1, join="inner")
    return W.dropna()


def pick(W: pd.DataFrame, n=N, seed=20261016) -> pd.DataFrame:
    W = W[(W.index >= "2025-01-01")]
    t = W.index.to_numpy().astype("datetime64[m]").astype(np.int64)
    chosen = []
    for i in np.random.default_rng(seed).permutation(len(W)):
        if all(abs(t[i] - t[j]) >= GAP_H * 60 for j in chosen):
            chosen.append(i)
            if len(chosen) == n:
                break
    assert len(chosen) == n
    return W.iloc[sorted(chosen)]


def sig(x, n=3):
    return None if x is None or not np.isfinite(x) else float(f"{x:.{n}g}")


def render(row: pd.Series, ts: pd.Timestamp, with_model: bool) -> dict:
    tt = ts + pd.Timedelta("1min")
    e = np.exp
    d = {"hour_utc": round(tt.hour + tt.minute / 60, 2), "weekday": tt.day_name(),
         "sigma_1m_24h_bp": sig(e(row["V_lsig24@1h"]) * 1e4),
         "vol_ratio_5m_vs_24h": sig(e(row["V_lsig5m@1h"])), "vol_ratio_15m_vs_24h": sig(e(row["V_lsig15m@1h"])),
         "vol_ratio_1h_vs_24h": sig(e(row["V_lsig1h@1h"])), "vol_ratio_4h_vs_24h": sig(e(row["V_lsig4h@1h"])),
         "parkinson_vol_ratio_1h_vs_24h": sig(e(row["V_park1h@1h"] - row["V_park24@1h"])),
         "max_1m_move_last_1h_in_sigma": sig(row["V_jump1h@1h"]),
         "abs_taker_imbalance_1h": sig(row["Q_imb1h@1h"]),
         "seasonal_vol_now_vs_avg": sig(e(row["S_season_now@1h"])),
         "major_us_macro_release_last_60min": bool(row["M_major_recent@1h"])}
    for s in imp.SCALES:
        b = {"box_half_width_bp": sig(e(row[f"log_rng@{s}"]) / 2),
             "half_width_in_sigma_units": sig(e(row[f"log_ds@{s}"])),
             "dist_to_nearest_edge_of_past_range_0to0.5": sig(row[f"R_edge@{s}"]),
             "trendiness_abs_move_over_range": sig(row[f"R_trend@{s}"]),
             "range_vs_previous_window": sig(e(row[f"R_compress@{s}"])),
             "range_vs_median_24h": sig(e(row[f"R_squeeze@{s}"])),
             "volume_per_min_last5m_vs_window": sig(e(row[f"Q_vol5_vs_W@{s}"])),
             "abs_taker_imbalance_window": sig(row[f"Q_imbW@{s}"]),
             "seasonal_vol_next_window_vs_avg": sig(e(row[f"S_season_fut@{s}"])),
             "major_us_macro_release_inside_next_window": bool(row[f"M_major_in@{s}"]),
             "jobless_claims_release_inside_next_window": bool(row[f"M_claims_in@{s}"])}
        if with_model:
            b["model_p_hold"] = round(float(row[f"p_F@{s}"]) * 100)
        d[f"box_{s}"] = b
    return d


PROMPT = """You are estimating "box hold" probabilities for the ETH perpetual futures market (Binance ETHUSDT).
At each snapshot, three boxes are drawn around the current price: box_30m, box_1h, box_4h. For window W in (30m, 1h, 4h):
  upper = price + half of the high-low range of the last W, lower = price - the same half-width.
  "hold" = during the NEXT W, the price high never touches the upper edge AND the low never touches the lower edge.
Historical average hold rates (2022-2024): 30m 26%, 1h 27%, 4h 29%. Holding is mostly about volatility: a box that is wide
relative to expected volatility over the next W holds more often; rising short-term volatility, busy hours, and scheduled
macro releases make breaks more likely.
{model_note}
Snapshots are independent (different days, at least 8 hours apart), shuffled; do not compare them or guess dates.
For EACH snapshot output p_hold_30m, p_hold_1h, p_hold_4h as integers 0-100 (calibrated probabilities).

Field notes: sigma_1m_24h_bp = std of 1-minute returns over the last 24h (bp); vol_ratio_X_vs_24h = recent std of 1m returns / 24h std;
half_width_in_sigma_units = half-width / (sigma_1m_24h * sqrt(W minutes)); dist_to_nearest_edge_of_past_range = where price sits in the last-W range
(0 = at an edge, 0.5 = middle); trendiness = |price change over W| / range; seasonal_* = typical volatility for that hour-of-week vs overall average.

Output ONLY one JSON array, same order: [{{"id": "...", "p_hold_30m": int, "p_hold_1h": int, "p_hold_4h": int}}, ...]

Snapshots:
"""
NOTE_B = ("Each box also has model_p_hold: the probability from a calibrated statistical model (logistic regression on width-vs-volatility, "
          "1h/24h volatility ratio, width, hour and weekend; out-of-sample AUC about 0.77). Start from it and adjust ONLY when the other "
          "fields show something that model does not use (e.g. 5m/15m/4h volatility, jumps, macro releases, seasonality, volume).")


def build() -> None:
    A = frame()
    P = model_probs(A)
    W = pick(wide(A, P))
    OUT.mkdir(parents=True, exist_ok=True)
    gold = pd.DataFrame({"id": [f"b{j:03d}" for j in range(len(W))], "ts": W.index})
    for s in SC:
        gold[f"hold_{s}"] = W[f"hold@{s}"].to_numpy().astype(int)
        gold[f"p_F_{s}"] = W[f"p_F@{s}"].to_numpy(); gold[f"p_HGB_{s}"] = W[f"p_HGB@{s}"].to_numpy()
    gold.to_csv(OUT / "gold.csv", index=False)
    order = np.random.default_rng(2).permutation(len(W))
    for cond, wm in (("A", False), ("B", True)):
        d = BDIR / cond
        d.mkdir(parents=True, exist_ok=True)
        rows = [{"id": gold.id[j], **render(W.iloc[j], W.index[j], wm)} for j in order]
        for b in range(0, len(rows), BATCH):
            txt = PROMPT.format(model_note=NOTE_B if wm else "") + "\n".join(json.dumps(r, separators=(",", ":")) for r in rows[b:b + BATCH])
            (d / f"batch_{b // BATCH:02d}.txt").write_text(txt)
    print(f"{len(W)} 시점 · {gold.ts.min()} ~ {gold.ts.max()} · 유지율 " + " ".join(f"{s} {gold[f'hold_{s}'].mean():.3f}" for s in SC))
    print("같은 표본의 기준 AUC: " + " ".join(f"{s} F {roc_auc_score(gold[f'hold_{s}'], gold[f'p_F_{s}']):.3f} HGB {roc_auc_score(gold[f'hold_{s}'], gold[f'p_HGB_{s}']):.3f}" for s in SC))


def calib(y, p, k=5):
    q = pd.qcut(p, k, labels=False, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q").mean()
    return float((g.p - g.y).abs().max())


def evaluate() -> None:
    from scipy.stats import spearmanr
    gold = pd.read_csv(OUT / "gold.csv")
    for cond in ("A", "B"):
        ans = {}
        for f in sorted((OUT / f"out_{cond}").glob("batch_*.json")):
            for a in json.loads(f.read_text()):
                ans[a["id"]] = a
        for s in SC:
            gold[f"p_{cond}_{s}"] = [ans[i][f"p_hold_{s}"] / 100 if i in ans else np.nan for i in gold.id]
    g = gold.dropna()
    print(f"채점 {len(g)}/{len(gold)} 시점")
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(g), len(g)) for _ in range(2000)]
    for s in SC:
        y = g[f"hold_{s}"].to_numpy()
        print(f"\n── {s} 박스 (유지율 {y.mean():.3f}) ──  {'모델':18s} AUC     ΔAUC vs 배포 F [95% CI]   로그손실  보정최대(5분위)")
        pf = g[f"p_F_{s}"].to_numpy()
        for k, nm in (("F", "배포 로지스틱 6칸"), ("HGB", "HGB 개선판"), ("A", "Haiku 단독"), ("B", "Haiku + 모델 확률")):
            p = g[f"p_{k}_{s}"].to_numpy()
            a = roc_auc_score(y, p)
            d = [roc_auc_score(y[b], p[b]) - roc_auc_score(y[b], pf[b]) for b in boots] if k != "F" else [0]
            print(f"   {nm:18s} {a:.3f}  {a - roc_auc_score(y, pf):+.3f} [{np.quantile(d, .025):+.3f}, {np.quantile(d, .975):+.3f}]  "
                  f"{log_loss(y, np.clip(p, .01, .99)):.4f}  {calib(y, p) * 100:4.1f}pp")
        for k in ("A", "B"):
            p = g[f"p_{k}_{s}"]
            print(f"   Haiku {k}: 평균 {p.mean():.3f} · 표준편차 {p.std():.3f} · ρ(배포 F) {spearmanr(p, g[f'p_F_{s}']).correlation:+.2f} · "
                  f"ρ(HGB) {spearmanr(p, g[f'p_HGB_{s}']).correlation:+.2f}" + (f" · 모델과 다른 답 {np.mean(np.abs(p - g[f'p_F_{s}'].round(2)) > 0.015):.0%}" if k == "B" else ""))
    g.to_csv(OUT / "scored.csv", index=False)


def _selfcheck() -> None:
    A = frame()
    assert set(A.scale.unique()) == set(SC)
    r = A[(A.scale == "4h") & (A.ts.dt.year == 2025)].dropna().iloc[0]
    one = A[(A.ts == r.ts)]
    assert set(one.scale) == set(SC), one.scale.tolist()            # 4h 결정 분엔 세 박스가 다 있다
    W = wide(one, pd.DataFrame(index=one.index, data={"p_F": .3, "p_HGB": .3}))
    x = render(W.iloc[0], W.index[0], True)
    assert {"box_30m", "box_1h", "box_4h"} <= set(x) and '"hold' not in json.dumps(x) and "hold@" not in json.dumps(x) and "ts" not in x
    assert x["box_4h"]["model_p_hold"] == 30 and "model_p_hold" not in render(W.iloc[0], W.index[0], False)["box_4h"]
    print("selfcheck OK")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selfcheck" in a:
        _selfcheck()
    elif a[:1] == ["build"]:
        build()
    elif a[:1] == ["eval"]:
        evaluate()
