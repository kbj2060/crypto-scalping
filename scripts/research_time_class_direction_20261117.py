#!/usr/bin/env python3
"""시간대별(활발 / 나머지) 15분 방향 예측 모델 정확도 — ETH · SOXL (2026-11-17). 사전등록 docs/prereg/time_class_direction_model_20261117.md (계산 전 커밋 1a9d6b56).

  selftest   합성 자료: 피쳐가 미래를 안 봄 · 라벨 직접 대조 · 시간대 분류 · 심은 신호를 모델이 찾음
  build      ETH·BTC 1분봉 중 로컬에 없는 날(10-30~11-15)을 Vision 에서 받는다
  run        1회 채점 → tmp/time_class_direction_20261117/result.json

결정 τ = 5분 경계. 피쳐 = 닫힌 5분봉 i(마감 = τ)까지. 라벨 = sign(log(종가[i+h] ÷ 종가[i])), h = 3(15분) · 12(60분).
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = ROOT / "tmp/time_class_direction_20261117"
VIS = Path("/home/kbj20/crypto-scalping/data/binance_vision")
NY, SEED, NB = "America/New_York", 20261117, 2000
BAR = 300_000
SPLIT = {"ETH": {"train": ("2024-01-08", "2025-12-31"), "val": ("2026-01-01", "2026-05-31"), "test": ("2026-06-01", "2026-11-16")},
         "SOXL": {"train": ("2026-05-22", "2026-08-31"), "val": ("2026-09-01", "2026-09-30"), "test": ("2026-10-01", "2026-11-16")}}   # [시작, 끝) · 경계 1일 purge
FRESH = {"ETH": "2026-10-28", "SOXL": "2026-11-11"}
RK = (1, 3, 6, 12, 24, 48, 288)
HGB = dict(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False)
SUB = ("미국개장", "오전", "나머지평일", "주말")


def load_crypto(sym: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(VIS / f"klines1m/{sym}-1m-202[4-6]-*.parquet"))) + sorted(glob.glob(str(VIS / f"klines1m_daily/{sym}-1m-2026-*.parquet"))) \
        + sorted(glob.glob(str(OUT / f"klines/{sym}-1m-*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in fs]).drop_duplicates("t").sort_values("t").reset_index(drop=True)


def bars5(k: pd.DataFrame) -> pd.DataFrame:
    """1분봉 → 5분봉(1분봉 5개가 다 있어야 한 봉). 색인 = 봉 시작 ms 의 빈틈 없는 격자(없는 봉은 NaN)."""
    b = k.assign(b=k.t // BAR * BAR).groupby("b").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"), tb=("tb", "sum"), n=("c", "size"))
    b = b[b.n == 5].drop(columns="n")
    return b.reindex(range(int(b.index[0]), int(b.index[-1]) + BAR, BAR))


def clock(t_end_ms: np.ndarray) -> pd.DataFrame:
    """결정 시각 τ(봉 마감)의 시간대. 활발 = 미국개장(평일 동부 09:30~11:00) ∪ 오전(일~목 동부 18:00~21:00)."""
    et = pd.to_datetime(t_end_ms, unit="ms", utc=True).tz_convert(NY)
    m, w = (et.hour * 60 + et.minute).to_numpy(), et.dayofweek.to_numpy()
    wk = ((w == 4) & (m >= 1200)) | (w == 5) | ((w == 6) & (m < 1080))
    sub = np.where(wk, 3, np.where((w < 5) & (m >= 570) & (m < 660), 0, np.where(((w <= 3) | (w == 6)) & (m >= 1080) & (m < 1260), 1, 2)))
    return pd.DataFrame({"sub": sub, "active": (sub <= 1).astype(float), "h_sin": np.sin(2 * np.pi * m / 1440), "h_cos": np.cos(2 * np.pi * m / 1440), "dow": w.astype(float),
                         "since_open": np.clip((m - 570) % 1440, 0, 1440).astype(float)})


def _wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def features(b: pd.DataFrame, cross: dict | None = None) -> pd.DataFrame:
    """행 i = 닫힌 5분봉 i 까지 본 피쳐(과거만). cross = {이름: 다른 종목 features() 결과} — 같은 봉 시각으로 맞춘다."""
    c, h, l, v, tb = b.c, b.h, b.l, b.v, b.tb
    lc = np.log(c); r = lc.diff(); sig = r.rolling(288, min_periods=200).std()
    F = {f"rz{k}": (lc - lc.shift(k)) / (sig * np.sqrt(k)) for k in RK}
    up, dn = r.clip(lower=0), (-r).clip(lower=0)
    for n in (14, 168):
        F[f"rsi{n}"] = 100 - 100 / (1 + _wilder(up, n) / _wilder(dn, n))
    tp = (h + l + c) / 3; ma = tp.rolling(20).mean(); F["cci20"] = (tp - ma) / (0.015 * (tp - ma).abs().rolling(20).mean())
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean(); F["macd_h"] = (macd - macd.ewm(span=9, adjust=False).mean()) / (c * sig)
    pc = c.shift(1); tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    pdm = (h.diff()).where((h.diff() > -l.diff()) & (h.diff() > 0), 0.0); ndm = (-l.diff()).where((-l.diff() > h.diff()) & (-l.diff() > 0), 0.0)
    atr = _wilder(tr, 14); pdi, ndi = 100 * _wilder(pdm, 14) / atr, 100 * _wilder(ndm, 14) / atr
    F["adx14"] = _wilder(100 * (pdi - ndi).abs() / (pdi + ndi), 14); F["di_diff"] = pdi - ndi
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std(); F["bb_b"] = (c - (m20 - 2 * s20)) / (4 * s20); F["bb_w"] = 4 * s20 / m20 / sig
    F["stoch14"] = (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min())
    for n in (20, 50, 200):
        F[f"ema{n}"] = (c / c.ewm(span=n, adjust=False, min_periods=n).mean() - 1) / sig
    F["vwap24"] = (c / ((tp * v).rolling(288, min_periods=200).sum() / v.rolling(288, min_periods=200).sum()) - 1) / sig
    for n in (12, 48, 288):
        F[f"don{n}"] = (c - l.rolling(n).min()) / (h.rolling(n).max() - l.rolling(n).min())
    for n in (1, 3, 12):
        F[f"tbi{n}"] = tb.rolling(n).sum() / v.rolling(n).sum() - 0.5
    F["volr"] = np.log(v.rolling(3).mean() / v.rolling(288, min_periods=200).mean())
    F["rvr"] = np.log(r.abs().rolling(3).mean() / r.abs().rolling(288, min_periods=200).mean())
    F = pd.DataFrame(F, index=b.index).replace([np.inf, -np.inf], np.nan)
    ck = clock(b.index.to_numpy() + BAR); ck.index = b.index
    F = pd.concat([F, ck], axis=1)
    for nm, G in (cross or {}).items():
        for k in (3, 12) if len(cross) > 1 else (1, 3, 12):
            F[f"{nm}_rz{k}"] = G[f"rz{k}"].reindex(b.index)
    F["ok"] = (r.rolling(288).count() >= 280) & (r.rolling(12).count() == 12) & sig.notna()     # 구멍 뒤 1일은 버린다
    for hh in (3, 12):
        F[f"fwd{hh}"] = (lc.shift(-hh) - lc) * 1e4                                               # 라벨용(피쳐 아님)
    F["mom"] = np.sign(lc - lc.shift(3))                                                        # 직전 15분 방향(단순 규칙)
    return F


def feat_cols(F: pd.DataFrame) -> list:
    return [c for c in F.columns if c not in ("ok", "fwd3", "fwd12", "mom", "sub")]


def cut(F: pd.DataFrame, a: str, b: str, purge_days: int = 1) -> pd.DataFrame:
    t0 = pd.Timestamp(a, tz="UTC").value // 10**6; t1 = pd.Timestamp(b, tz="UTC").value // 10**6 - purge_days * 86_400_000
    return F[(F.index + BAR >= t0) & (F.index + BAR < t1)]


def fit(tr: pd.DataFrame, cols: list, h: int, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    d = tr[tr.ok & tr[f"fwd{h}"].notna() & (tr[f"fwd{h}"] != 0)]
    return HistGradientBoostingClassifier(**HGB, random_state=seed).fit(d[cols].to_numpy(), (d[f"fwd{h}"] > 0).astype(int).to_numpy())


def boot(day: np.ndarray, *cols) -> np.ndarray:
    """달력일 블록: 일별 합을 다시 뽑아 (합1, 합2, …) 를 돌려준다."""
    g = pd.DataFrame({"d": day, **{f"c{i}": c for i, c in enumerate(cols)}}).groupby("d").sum().to_numpy()
    rng = np.random.default_rng(SEED)
    return g[rng.integers(0, len(g), (NB, len(g)))].sum(1)


def evaluate(F: pd.DataFrame, sym: str, h: int, seed: int = 0, importance: bool = False) -> dict:
    from sklearn.metrics import roc_auc_score
    cols = feat_cols(F); S = SPLIT[sym]
    tr, va, te = (cut(F, *S[k]) for k in ("train", "val", "test"))
    use = lambda d: d[d.ok & d[f"fwd{h}"].notna() & (d[f"fwd{h}"] != 0)]
    va, te = use(va), use(te)
    pooled = fit(tr, cols, h, seed); res = {"행": {"학습": int(use(tr).shape[0]), "검증": len(va), "시험": len(te)}}
    base_fns = {"따라가기": lambda d: d.mom.replace(0, 1).fillna(1), "거스르기": lambda d: -d.mom.replace(0, 1).fillna(1),
                "늘오름": lambda d: pd.Series(1.0, index=d.index), "늘내림": lambda d: pd.Series(-1.0, index=d.index)}
    for cls, act in (("활발", 1.0), ("나머지", 0.0)):
        sep = fit(tr[tr.active == act], cols, h, seed)
        v, t = va[va.active == act], te[te.active == act]
        yv, yt = np.sign(v[f"fwd{h}"]).to_numpy(), np.sign(t[f"fwd{h}"]).to_numpy()
        arms = {}
        for nm, mdl in (("통합", pooled), ("분리", sep)):
            pv = mdl.predict_proba(v[cols].to_numpy())[:, 1]; lo, hi = np.quantile(pv, [.2, .8])
            sv = np.where(pv >= hi, 1, np.where(pv <= lo, -1, 0)); arms[nm] = (mdl, lo, hi, float((sv[sv != 0] == yv[sv != 0]).mean()), sv)
        pick = max(arms, key=lambda a: (arms[a][3], a == "통합")); mdl, lo, hi, acc_v, sv = arms[pick]
        bv = {b: float((f(v).to_numpy()[sv != 0] == yv[sv != 0]).mean()) for b, f in base_fns.items()}; bpick = max(bv, key=bv.get)
        pt = mdl.predict_proba(t[cols].to_numpy())[:, 1]; st = np.where(pt >= hi, 1, np.where(pt <= lo, -1, 0)); on = st != 0
        hit = (st[on] == yt[on]).astype(float); bhit = (base_fns[bpick](t).to_numpy()[on] == yt[on]).astype(float); day = ((t.index.to_numpy()[on] + BAR) // 86_400_000)
        B = boot(day, hit, bhit, np.ones(len(hit))); accb, difb = B[:, 0] / B[:, 2], (B[:, 0] - B[:, 1]) / B[:, 2]
        sub = t["sub"].to_numpy()[on]; fwd = t[f"fwd{h}"].to_numpy()[on]
        fresh = (t.index.to_numpy()[on] + BAR) >= pd.Timestamp(FRESH[sym], tz="UTC").value // 10**6
        r = {"고른_팔": pick, "검증_확신정확도": {a: round(arms[a][3], 4) for a in arms}, "고른_기준선": bpick, "검증_기준선정확도": {b: round(x, 4) for b, x in bv.items()},
             "시험_행": len(t), "확신_건수": f"{int(hit.sum())}/{len(hit)}", "확신_정확도": float(hit.mean()), "확신_하한90": float(np.percentile(accb, 10)),
             "발동": float(on.mean()), "기준선_정확도": float(bhit.mean()), "차": float(hit.mean() - bhit.mean()), "차_하한90": float(np.percentile(difb, 10)),
             "p_차": float((difb <= 0).mean()), "날": int(len(np.unique(day))),
             "롱신호": f"{int(hit[st[on] > 0].sum())}/{int((st[on] > 0).sum())}", "숏신호": f"{int(hit[st[on] < 0].sum())}/{int((st[on] < 0).sum())}",
             "전체행_정확도": float(((pt > 0.5) == (yt > 0)).mean()), "AUC": float(roc_auc_score(yt > 0, pt)), "실제_오름_비율": float((yt > 0).mean()),
             "확신행_방향맞춘_평균bp": float((st[on] * fwd).mean()), "세부": {SUB[s]: f"{int(hit[sub == s].sum())}/{int((sub == s).sum())}" for s in range(4) if (sub == s).any()},
             "새구간": f"{int(hit[fresh].sum())}/{int(fresh.sum())}",
             "단일피쳐_AUC_최대": max(((abs(roc_auc_score(yt[np.isfinite(x)] > 0, x[np.isfinite(x)]) - .5) + .5, c) for c in cols for x in [t[c].to_numpy()] if np.isfinite(x).sum() > 100), default=(None, None))}
        r["1_50초과"] = bool(r["확신_하한90"] > 0.5); r["후보"] = bool(acc_v > 0.5 and acc_v > bv[bpick] and r["확신_정확도"] > 0.5 and r["차"] > 0)
        r["누수_의심"] = bool(r["단일피쳐_AUC_최대"][0] >= 0.60 or r["전체행_정확도"] > 0.58)
        if importance:
            from sklearn.inspection import permutation_importance
            s = v.sample(min(len(v), 20000), random_state=0); pi = permutation_importance(mdl, s[cols].to_numpy(), (s[f"fwd{h}"] > 0).astype(int).to_numpy(), n_repeats=3, random_state=0, scoring="roc_auc")
            r["중요_피쳐(검증 순열, AUC 감소)"] = [(cols[i], round(float(pi.importances_mean[i]), 4)) for i in np.argsort(-pi.importances_mean)[:8]]
        res[cls] = r
    return res


def holm(cells: dict) -> None:
    """네 칸의 «단순 규칙 초과» p 에 Holm(.025 · .033 · .05 · .10). 통과 = 1 과 2."""
    order = sorted(cells, key=lambda k: cells[k]["p_차"]); alive = True
    for i, k in enumerate(order):
        alive = alive and cells[k]["p_차"] < 0.10 / (len(order) - i)
        cells[k]["2_기준선초과_Holm"] = bool(alive and cells[k]["차_하한90"] > 0); cells[k]["통과"] = bool(cells[k]["1_50초과"] and cells[k]["2_기준선초과_Holm"])


def frames() -> dict:
    import research_soxl_intraday_clock_rules_20261117 as C
    eth, btc = features(bars5(load_crypto("ETHUSDT"))), features(bars5(load_crypto("BTCUSDT")))
    return {"ETH": features(bars5(load_crypto("ETHUSDT")), {"btc": btc}), "SOXL": features(bars5(C.load()), {"eth": eth, "btc": btc})}


def build() -> None:
    (OUT / "klines").mkdir(parents=True, exist_ok=True)
    for sym in ("ETHUSDT", "BTCUSDT"):
        last = pd.to_datetime(load_crypto(sym).t.iloc[-1], unit="ms").normalize()
        for day in pd.date_range(last + pd.Timedelta(days=1), "2026-11-15").strftime("%Y-%m-%d"):
            try:                                                # Vision 정적 파일(REST 아님)
                raw = urllib.request.urlopen(f"https://data.binance.vision/data/futures/um/daily/klines/{sym}/1m/{sym}-1m-{day}.zip", timeout=60).read()
            except urllib.error.HTTPError:
                print(sym, day, "없음"); continue
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                d = pd.read_csv(z.open(z.namelist()[0]), header=None)
            if not str(d.iloc[0, 0]).isdigit():
                d = d.iloc[1:]
            d = d.astype(float)
            pd.DataFrame({"t": d[0].astype("int64"), "h": d[2], "l": d[3], "c": d[4], "v": d[5], "tb": d[9]}).to_parquet(OUT / "klines" / f"{sym}-1m-{day}.parquet")
        k = load_crypto(sym); print(sym, len(k), pd.to_datetime(k.t.iloc[[0, -1]], unit="ms").astype(str).tolist())


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FR = frames(); out = {"primary_15m": {}, "aux_60m": {}, "seed1_15m": {}}
    for sym, F in FR.items():
        out["primary_15m"][sym] = evaluate(F, sym, 3, importance=True)
        out["aux_60m"][sym] = evaluate(F, sym, 12)
        s1 = evaluate(F, sym, 3, seed=1); out["seed1_15m"][sym] = {c: {"확신_건수": s1[c]["확신_건수"], "확신_정확도": s1[c]["확신_정확도"], "고른_팔": s1[c]["고른_팔"]} for c in ("활발", "나머지")}
    cells = {f"{s}·{c}": out["primary_15m"][s][c] for s in out["primary_15m"] for c in ("활발", "나머지")}; holm(cells)
    out["판정"] = {k: {x: v[x] for x in ("1_50초과", "2_기준선초과_Holm", "통과", "후보", "누수_의심")} for k, v in cells.items()}
    (OUT / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str)); print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


def selftest() -> None:
    rng = np.random.default_rng(5); n = 40 * 1440
    t = pd.Timestamp("2026-10-12", tz="UTC").value // 10**6 + np.arange(n) * 60000
    c = 100 * np.exp(np.cumsum(rng.normal(0, 3e-4, n))); v = rng.uniform(1, 3, n)
    k = pd.DataFrame({"t": t, "h": c * 1.0002, "l": c * 0.9998, "c": c, "v": v, "tb": v * rng.uniform(.3, .7, n)})
    b = bars5(k); F = features(b); cols = feat_cols(F)
    assert len(b) == n // 5 and "sub" not in cols and "fwd3" not in cols and "mom" not in cols
    # ① 미래를 잘라도 같은 피쳐
    i = 6000; Fc = features(bars5(k[k.t < b.index[i] + BAR]))
    assert np.allclose(F[cols].iloc[i].to_numpy(float), Fc[cols].iloc[-1].to_numpy(float), equal_nan=True), "피쳐가 미래를 본다"
    # ② 라벨 직접 대조 · 피쳐 rz1 은 자기 봉 수익(다음 봉 아님)
    lc = np.log(b.c.to_numpy()); assert abs(F.fwd3.iloc[i] - (lc[i + 3] - lc[i]) * 1e4) < 1e-9 and abs(F.fwd12.iloc[i] - (lc[i + 12] - lc[i]) * 1e4) < 1e-9
    r = np.diff(lc, prepend=np.nan); ok = F.ok.to_numpy() & np.isfinite(F.rz1.to_numpy())
    assert np.corrcoef(F.rz1.to_numpy()[ok][:-1], r[ok][:-1])[0, 1] > 0.95 and abs(np.corrcoef(F.rz1.to_numpy()[ok][:-1], r[1:][ok[:-1]])[0, 1]) < 0.1
    fy = F.fwd3.to_numpy(); worst = max(abs(np.corrcoef(x[m], fy[m])[0, 1]) for cc in cols for x in [F[cc].to_numpy()] for m in [ok & np.isfinite(x) & np.isfinite(fy)] if np.std(x[m]) > 0)
    assert worst < 0.1, ("피쳐가 라벨과 겹친다", worst)                              # 무작위 보행에선 어느 피쳐도 다음 15분 수익과 상관이 없어야 한다
    # ③ 시간대: 한국 겨울 23:30 = 동부 09:30 → 미국개장 · 한국 08:00 = 동부 18:00 → 오전 · 토 → 주말 · 서머타임 전(10-20)도 동부 기준
    ms = lambda s: pd.Timestamp(s, tz=NY).tz_convert("UTC").value // 10**6
    ck = clock(np.array([ms("2026-11-10 09:30"), ms("2026-11-10 10:59"), ms("2026-11-10 11:00"), ms("2026-11-10 18:00"), ms("2026-11-08 18:00"), ms("2026-11-06 18:00"), ms("2026-11-07 12:00"), ms("2026-10-20 09:45"), ms("2026-11-10 14:00")]))
    assert ck["sub"].tolist() == [0, 0, 2, 1, 1, 2, 3, 0, 2] and ck.active.tolist() == [1, 1, 0, 1, 1, 0, 0, 1, 0], ck["sub"].tolist()
    # ④ 심은 신호: 다음 15분 수익 부호가 tbi3 을 따르게 → 모델이 찾는다(파이프라인 점검)
    b2 = b.copy(); lc2 = np.log(b2.c.to_numpy()).copy(); sgn = np.sign((b2.tb.rolling(3).sum() / b2.v.rolling(3).sum() - 0.5).to_numpy())
    step = np.zeros(len(b2)); step[1:] = np.nan_to_num(sgn[:-1]) * 4e-4 + rng.normal(0, 6e-4, len(b2) - 1)
    c2 = 100 * np.exp(np.cumsum(step)); b2 = b2.assign(c=c2, h=c2 * 1.0002, l=c2 * 0.9998); F2 = features(b2)
    g = globals(); keep = g["SPLIT"]["SOXL"], g["HGB"]
    g["SPLIT"]["SOXL"] = {"train": ("2026-10-12", "2026-11-03"), "val": ("2026-11-03", "2026-11-12"), "test": ("2026-11-12", "2026-11-21")}; g["HGB"] = dict(HGB, max_iter=60, min_samples_leaf=40)
    try:
        r2 = evaluate(F2, "SOXL", 3)
    finally:
        g["SPLIT"]["SOXL"], g["HGB"] = keep
    assert r2["나머지"]["확신_정확도"] > 0.58 and r2["나머지"]["전체행_정확도"] > 0.54, r2["나머지"]
    cells = {"a": {"p_차": .01, "차_하한90": .01, "1_50초과": True}, "b": {"p_차": .2, "차_하한90": -.01, "1_50초과": True}, "c": {"p_차": .03, "차_하한90": .01, "1_50초과": False}}
    holm(cells); assert cells["a"]["통과"] and not cells["b"]["통과"] and not cells["c"]["통과"] and cells["c"]["2_기준선초과_Holm"]
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "build", "run"])
    {"selftest": selftest, "build": build, "run": run}[ap.parse_args().cmd]()
