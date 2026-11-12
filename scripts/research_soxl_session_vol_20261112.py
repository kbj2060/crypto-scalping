#!/usr/bin/env python3
"""SOXL 선물 세션별 15분·1시간 변동성 예측 (2026-11-12). 사전등록 docs/prereg/soxl_session_vol_20261112.md (계산 전 커밋 3081c802).

  selftest   세션 시각표 · 미래를 잘라도 같은 피쳐 · 표적 직접 계산 대조 · QLIKE · HAR 계수 복원
  build      SOXLUSDT·SNDKUSDT 1분봉을 Vision 일 파일로 받는다(tmp/soxl_vol_20261112/klines)
  run        적합·판정 → result.json

시각 규약: 결정 τ = 5분 경계(UTC). 피쳐 = 시작 < τ 인 1분봉만. 표적 = [τ, τ+H) 1분 로그수익 제곱합의 제곱근.
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
OUT = ROOT / "tmp/soxl_vol_20261112"
FIRST, LAST = "2026-05-15", "2026-11-10"
SPLIT = {"train": ("2026-05-22", "2026-08-31"), "val": ("2026-09-01", "2026-09-30"), "test": ("2026-10-01", "2026-11-11"),
         "refit": ("2026-05-22", "2026-09-30")}     # [시작, 끝) UTC -- 끝을 하루 당겨 경계 1일 purge
# ponytail: NYSE 휴장일 손으로 적음(2026 남은 것까지). 반일장(11-27·12-24)은 정규장으로 본다 -- 해가 바뀌면 달력 라이브러리로.
HOLIDAYS = {"2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"}
SESS = ("정규장", "프리장", "애프터", "야간", "주말", "휴장")
RV_FLOOR = 1e-4
WIN = {"lrv5": 5, "lrv15": 15, "lrv60": 60, "lrv240": 240, "lrv1d": 1440, "lrv7d": 10080}
HGB_COLS = [*WIN, "lrng15", "lvol15", "jump5", "sess", "since_open", "to_open", "to_close", "dow", "tod"]
HAR_COLS = ["lrv15", "lrv60", "lrv1d", "lrv7d"]


def clock(tau: np.ndarray, H: int) -> pd.DataFrame:
    """τ(UTC 초)의 세션·개장/마감까지 분 -- 시계만으로 정해진다."""
    et = pd.to_datetime(tau, unit="s", utc=True).tz_convert("America/New_York")
    m, w = (et.hour * 60 + et.minute).to_numpy(), et.dayofweek.to_numpy()
    hol = np.isin(et.strftime("%Y-%m-%d"), list(HOLIDAYS))
    sess = np.select([(w == 5) | ((w == 4) & (m >= 1200)) | ((w == 6) & (m < 1200)), hol & (m >= 240) & (m < 1200),
                      (m >= 570) & (m < 960), (m >= 240) & (m < 570), (m >= 960) & (m < 1200)], [4, 5, 0, 1, 2], 3)
    days = pd.date_range(et.min().normalize() - pd.Timedelta(days=5), et.max().normalize() + pd.Timedelta(days=5), freq="D")
    days = days[(days.dayofweek < 5) & ~np.isin(days.strftime("%Y-%m-%d"), list(HOLIDAYS))]
    opens = ((days + pd.Timedelta(minutes=570)).tz_convert("UTC").asi8 // 10**9)   # 정규장 개장(동부 09:30) UTC 초
    closes = opens + 390 * 60
    i = np.searchsorted(opens, tau, "right") - 1
    since_open = np.clip((tau - opens[i.clip(0)]) / 60, 0, 1440)
    to_open = np.clip((opens[np.searchsorted(opens, tau, "left").clip(max=len(opens) - 1)] - tau) / 60, 0, 1440)
    to_close = np.clip((closes[np.searchsorted(closes, tau, "left").clip(max=len(closes) - 1)] - tau) / 60, 0, 1440)
    return pd.DataFrame({"sess": sess, "since_open": since_open, "to_open": to_open, "to_close": to_close, "dow": w, "tod": m,
                         "open_in": (to_open < H).astype(float), "first60": ((sess == 0) & (since_open < 60)).astype(float),
                         "last30": ((sess == 0) & (to_close <= 30)).astype(float)})


def frame(k: pd.DataFrame, H: int) -> pd.DataFrame:
    """1분봉(t ms·h·l·c·v) → 5분 경계 τ 마다 피쳐·표적. 서빙도 이 함수를 쓴다(표적 열만 NaN)."""
    t0, t1 = int(k.t.iloc[0]) // 60000, int(k.t.iloc[-1]) // 60000
    g = k.assign(i=k.t // 60000 - t0).set_index("i").reindex(range(t1 - t0 + 1))      # 빈 분은 NaN
    c, h, l, qv = g.c.to_numpy(), g.h.to_numpy(), g.l.to_numpy(), (g.v * g.c).to_numpy()
    r = np.diff(np.log(c), prepend=np.nan)
    r2 = pd.Series(r * r)
    f = {name: np.log(np.sqrt(r2.rolling(n, min_periods=int(n * 0.95)).mean()).clip(lower=1e-6)) for name, n in WIN.items()}   # 분당 변동성(로그)
    f["lrng15"] = np.log(((pd.Series(h).rolling(15, min_periods=14).max() - pd.Series(l).rolling(15, min_periods=14).min()) / c).clip(lower=1e-6))
    f["lvol15"] = np.log((pd.Series(qv).rolling(15, min_periods=14).mean() / pd.Series(qv).rolling(1440, min_periods=1300).mean()).clip(lower=1e-6))
    f["jump5"] = np.abs(np.log(c / np.roll(c, 5))) / (np.exp(f["lrv60"]) * np.sqrt(5))
    hr_end = (np.arange(len(c)) + t0 + 1) % 60 == 0                                    # 그 시간의 마지막 분
    hc = pd.Series(np.where(hr_end, c, np.nan)).dropna()
    f["b0"] = (np.log(hc).diff().rolling(168, min_periods=150).std(ddof=1) * np.sqrt(H / 60)).reindex(range(len(c))).ffill().to_numpy()   # 현행: 7일 1시간 수익 σ
    fut = r2.rolling(H, min_periods=H).sum().shift(-(H - 1))                           # 분 j 부터 H 개
    prev = pd.Series(c).shift(1)
    exc = np.maximum(np.log(pd.Series(h).rolling(H, min_periods=H).max().shift(-(H - 1)) / prev),
                     -np.log(pd.Series(l).rolling(H, min_periods=H).min().shift(-(H - 1)) / prev))
    F = pd.DataFrame(f)
    j = np.arange(len(c))
    j = j[((j + t0) % 5 == 0) & (j >= 1)]                                              # τ = 분 j 의 시작
    d = F.iloc[j - 1].reset_index(drop=True)                                           # 피쳐 = 직전 분까지
    d["tau"] = (j + t0) * 60
    d["rv"] = np.sqrt(fut.to_numpy()[j])
    d["exc"] = exc.to_numpy()[j]
    return pd.concat([d, clock(d.tau.to_numpy(), H)], axis=1)


def qlike(rv: np.ndarray, var: np.ndarray) -> np.ndarray:
    x = np.maximum(rv, RV_FLOOR) ** 2 / var
    return x - np.log(x) - 1


def har_x(d: pd.DataFrame) -> np.ndarray:
    S = np.eye(6)[d.sess.to_numpy()][:, 1:]                                            # 세션 더미(정규장 기준)
    return np.column_stack([np.ones(len(d)), d[HAR_COLS].to_numpy(), S, d[["open_in", "first60", "last30"]].to_numpy()])


def fit_predict(fit: pd.DataFrame, new: pd.DataFrame, seed: int = 0) -> dict:
    """로그 RV 예측 → 눈금 맞춘 분산 예측(팔마다 적합 구간에서 평균 RV²/예측² = 1)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    y = np.log(np.maximum(fit.rv.to_numpy(), RV_FLOOR))
    beta = np.linalg.lstsq(har_x(fit), y, rcond=None)[0]
    hgb = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400, min_samples_leaf=100, l2_regularization=1.0,
                                        early_stopping=False, random_state=seed).fit(fit[HGB_COLS].to_numpy(), y)
    P = lambda d: {"B0": np.log(d.b0.to_numpy()), "HAR": har_x(d) @ beta, "HGB": hgb.predict(d[HGB_COLS].to_numpy())}
    pf, pn = P(fit), P(new)
    for p in (pf, pn):
        p["AVG"] = (p["HAR"] + p["HGB"]) / 2
    rv2 = np.maximum(fit.rv.to_numpy(), RV_FLOOR) ** 2
    return {a: np.exp(2 * pn[a]) * np.mean(rv2 / np.exp(2 * pf[a])) for a in pn}


def cut(d: pd.DataFrame, name: str) -> pd.DataFrame:
    a, b = (pd.Timestamp(x, tz="UTC").timestamp() for x in SPLIT[name])
    return d[(d.tau >= a) & (d.tau < b)].reset_index(drop=True)


def boot_upper(diff: np.ndarray, tau: np.ndarray, rng) -> tuple[float, float]:
    """달력일 블록 부트스트랩 -- (평균 차, 단측 90% 상한)."""
    day = tau // 86400
    s = pd.DataFrame({"d": day, "x": diff}).groupby("d").x.agg(["sum", "size"]).to_numpy()
    b = s[rng.integers(0, len(s), (2000, len(s)))]
    return float(diff.mean()), float(np.percentile(b[:, :, 0].sum(1) / b[:, :, 1].sum(1), 90))


def evaluate(sym: str, H: int) -> dict:
    d = frame(load(sym), H)
    need = [*HGB_COLS, "b0", "rv", "open_in"]
    n_raw, n_floor = len(d), int((d.rv < RV_FLOOR).sum())
    d = d.dropna(subset=need).reset_index(drop=True)
    rng = np.random.default_rng(20261112)
    tr, va, te, rf = (cut(d, s) for s in ("train", "val", "test", "refit"))
    Q = lambda rows, V: {a: float(qlike(rows.rv.to_numpy(), v).mean()) for a, v in V.items()}
    Vv = fit_predict(tr, va)
    qv = Q(va, Vv)
    pick = min(("HGB", "AVG"), key=qv.get)
    Vt = fit_predict(rf, te)
    qt = Q(te, Vt)
    y, tau = te.rv.to_numpy(), te.tau.to_numpy()
    ql = {a: qlike(y, v) for a, v in Vt.items()}
    ly = np.log(np.maximum(y, RV_FLOOR))
    r2 = lambda v: float(1 - ((ly - np.log(v) / 2) ** 2).sum() / ((ly - ly.mean()) ** 2).sum())
    j1, j2 = boot_upper(ql["HAR"] - ql["B0"], tau, rng), boot_upper(ql[pick] - ql["HAR"], tau, rng)
    cand = pick if (j1[1] < 0 and j2[1] < 0) else "HAR"
    sig = {a: np.sqrt(v) for a, v in Vt.items()}
    by = {}
    for s in range(6):
        m = te.sess.to_numpy() == s
        if m.sum():
            by[SESS[s]] = {"n": int(m.sum()), "qlike": {a: round(float(ql[a][m].mean()), 4) for a in ql},
                           "실제 RV 중앙%": round(float(np.median(y[m]) * 100), 3), "예측 σ 중앙%": {a: round(float(np.median(sig[a][m]) * 100), 3) for a in ("B0", cand)},
                           "최대 이동 > 1σ": {a: f"{int((te.exc.to_numpy()[m] > sig[a][m]).sum())}/{int(m.sum())}" for a in ("B0", cand)},
                           "최대 이동 > 2σ": {a: f"{int((te.exc.to_numpy()[m] > 2 * sig[a][m]).sum())}/{int(m.sum())}" for a in ("B0", cand)}}
    dec = [[round(float(np.median(sig[cand][q]) * 100), 3), round(float(np.median(y[q]) * 100), 3), round(float(np.sqrt(np.mean(y[q] ** 2)) * 100), 3)]
           for q in np.array_split(np.argsort(sig[cand]), 10)]
    seed1 = fit_predict(rf, te, seed=1)["HGB"]
    return {"sym": sym, "H": H, "rows": {"all": n_raw, "used": len(d), "train": len(tr), "val": len(va), "test": len(te), "rv_floor": n_floor},
            "val_qlike": {a: round(v, 4) for a, v in qv.items()}, "pick": pick, "test_qlike": {a: round(v, 4) for a, v in qt.items()},
            "test_logR2": {a: round(r2(v), 3) for a, v in Vt.items()},
            "judge1_HAR_minus_B0": [round(x, 4) for x in j1], "judge2_pick_minus_HAR": [round(x, 4) for x in j2],
            "pass1": bool(j1[1] < 0), "pass2": bool(j2[1] < 0), "candidate": cand if j1[1] < 0 else None,
            "leak_flag": bool(max(r2(v) for v in Vt.values()) > 0.8 or qt["HGB"] < 0.7 * qt["HAR"]),
            "hgb_seed0_vs_1_qlike": [round(qt["HGB"], 4), round(float(qlike(y, seed1).mean()), 4)],
            "by_session": by, "decile_pred_vs_real_median_rms(%)": dec}


def load(sym: str) -> pd.DataFrame:
    ps = sorted((OUT / "klines").glob(f"{sym}-1m-*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in ps]).drop_duplicates("t").sort_values("t").reset_index(drop=True)


def build() -> None:
    from scripts.research_sr_line_book_20261112 import vision_day
    (OUT / "klines").mkdir(parents=True, exist_ok=True)
    for sym in ("SOXLUSDT", "SNDKUSDT"):
        for day in pd.date_range(FIRST, LAST).strftime("%Y-%m-%d"):
            p = OUT / "klines" / f"{sym}-1m-{day}.parquet"
            if not p.exists():
                k = vision_day(sym, day)
                if k is not None:
                    k.to_parquet(p)
        k = load(sym)
        print(sym, "분봉", len(k), pd.to_datetime(k.t.iloc[[0, -1]], unit="ms").astype(str).tolist(), "빈 분", int((k.t.iloc[-1] - k.t.iloc[0]) // 60000 + 1 - len(k)))


def run() -> None:
    res = {"main": evaluate("SOXLUSDT", 15), "aux_60m": evaluate("SOXLUSDT", 60), "aux_sndk": evaluate("SNDKUSDT", 15)}
    (OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


def selftest() -> None:
    # 세션 시각표(서머타임 전후)
    ts = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp())
    c = clock(np.array([ts("2026-07-01 13:30"), ts("2026-11-03 14:30"), ts("2026-11-03 13:30"), ts("2026-07-04 12:00"), ts("2026-07-03 15:00"),
                        ts("2026-07-05 23:55"), ts("2026-07-06 00:00"), ts("2026-07-01 20:00"), ts("2026-07-01 13:20"), ts("2026-07-02 00:00")]), 15)
    assert c.sess.tolist() == [0, 0, 1, 4, 5, 4, 3, 2, 1, 3], c.sess.tolist()
    assert c.since_open[0] == 0 and c.to_open[0] == 0 and c.open_in[0] == 1 and c.first60[0] == 1 and c.to_close[0] == 390
    assert c.to_open[8] == 10 and c.open_in[8] == 1 and c.open_in[2] == 0 and c.to_open[2] == 60
    assert c.to_open[4] == 1440 and c.since_open[4] == 1440                    # 휴장일(7-3 금): 다음 개장은 월요일
    # 합성 12일 1분봉: 미래를 잘라도 같은 피쳐 · 표적 직접 계산 대조
    rng = np.random.default_rng(0)
    n = 12 * 1440
    cc = 60 * np.exp(np.cumsum(rng.normal(0, 3e-4, n)))
    k = pd.DataFrame({"t": ts("2026-06-01 00:00") * 1000 + np.arange(n) * 60000, "h": cc * (1 + 2e-4), "l": cc * (1 - 2e-4), "c": cc, "v": rng.uniform(1, 2, n)})
    k = k.drop(index=range(9000, 9007)).reset_index(drop=True)                 # 빈 분 7개
    A, M = frame(k, 15), 15000
    B = frame(k.iloc[:M], 15)
    cols = [*HGB_COLS, "b0", "open_in", "first60", "last30"]
    last = int(k.t.iloc[M - 1]) // 1000 + 60
    a, b = A[A.tau <= last].reset_index(drop=True), B[B.tau <= last].reset_index(drop=True)
    assert len(a) == len(b) and np.allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), equal_nan=True), "피쳐가 미래를 본다"
    row = A[A.rv.notna() & A.lrv7d.notna()].iloc[100]
    i = int(np.searchsorted(k.t.to_numpy(), row.tau * 1000))
    assert k.t[i] == row.tau * 1000 and abs(row.rv - np.sqrt((np.diff(np.log(k.c.to_numpy()[i - 1:i + 15])) ** 2).sum())) < 1e-12
    assert abs(row.lrv15 - np.log(np.sqrt((np.diff(np.log(k.c.to_numpy()[i - 16:i])) ** 2).mean()))) < 1e-12
    assert abs(row.exc - max(np.log(k.h[i:i + 15].max() / k.c[i - 1]), -np.log(k.l[i:i + 15].min() / k.c[i - 1]))) < 1e-12
    gap = A[(A.tau > k.t[8999] // 1000 - 840) & (A.tau <= k.t[8999] // 1000 + 60 + 420)]
    assert gap.rv.isna().all(), "봉 구멍이 든 표적 창이 남았다"
    # QLIKE · HAR 계수 복원
    assert np.allclose(qlike(np.array([0.01, 0.02]), np.array([1e-4, 4e-4])), 0)
    d = A.dropna(subset=[*HGB_COLS, "b0"]).reset_index(drop=True)
    d["rv"] = np.exp(-1.0 + 0.5 * d.lrv60 + 0.7 * (d.sess == 4))
    beta = np.linalg.lstsq(har_x(d), np.log(d.rv), rcond=None)[0]
    assert abs(beta[2] - 0.5) < 1e-6 and abs(beta[8] - 0.7) < 1e-6, beta
    V = fit_predict(d, d)
    assert qlike(d.rv.to_numpy(), V["HAR"]).mean() < 1e-9 and qlike(d.rv.to_numpy(), V["B0"]).mean() > 0.01
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "build", "run"])
    {"selftest": selftest, "build": build, "run": run}[ap.parse_args().cmd]()
