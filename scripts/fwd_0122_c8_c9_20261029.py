#!/usr/bin/env python3
"""2027-01-22 전진 판정 — C8 박스 1h + P2 · C9 호가 깊이 후보 3팔(박스 ETH 30분·4시간 · card30 닿음). 2026-10-29, 전진 자료 보기 전.
사전등록: docs/prereg/liqlab_liquidity_arms_20261028.md §14·§15 · docs/prereg/depth_incr_dashboard_models_20261028.md §10.

서버 저장소 루트에서(scripts/ 에 두지 않음 — tmp/fwd_0122_20261029/ 에 이 파일과 depth_incr_20261028.py 사본):
  1) 패널(깊이·P2): LIQLAB_END=2027-01-21 LIQLAB_OUT=data/liqlab_fwd0122 python tmp/liqlab_20261028/liqlab_data_20261028.py 4
  2) python tmp/fwd_0122_20261029/fwd_0122_c8_c9_20261029.py build [--dry] [--events data/live/<2027 갱신 발표 일정>.json]
  3) python tmp/fwd_0122_20261029/fwd_0122_c8_c9_20261029.py c8 [--dry]   ·   ... c9 [--dry]
--dry = 같은 코드로 시험창(2026-06-01 ~ 10-23) → 동결 결과(검수 review_box2 · depth_incr JSON · incr 계수) 재현 assert.
--dry --rb = 시험창 행만 새 빌드로(빌더 잡음 측정, 보고만). build --dry 는 둘 다 만든다.
--dry 없으면 판정일(2027-01-22) 전 거절 · 판정은 한 번만.
"""
import concurrent.futures as cf
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path.cwd(); HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / "tmp/liqlab_20261028"), str(ROOT), str(ROOT / "scripts")]
OUT = ROOT / "tmp/fwd_0122_20261029"; OUT.mkdir(parents=True, exist_ok=True)
DRY = "--dry" in sys.argv; TAG = ("dryrb" if "--rb" in sys.argv else "dry") if DRY else "fwd"   # dryrb = 시험창을 새 빌드 행으로(빌더 잡음 측정 · 재현 assert 없음)
EV0, EV1 = (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-10-24")) if DRY else (pd.Timestamp("2026-10-24"), pd.Timestamp("2027-01-22"))  # [EV0, 라벨 끝 ≤ EV1)
VERDICT = pd.Timestamp("2027-01-22")
FROZEN = {"box": ROOT / "tmp/liqlab_20261028/box_rows.parquet", "card": ROOT / "tmp/card30_direction_20260926/dataset.parquet"}
WIN = {"30m": 30, "1h": 60, "4h": 240}
F_BASE = ["log_ds", "log_vr", "log_rng", "hs", "hc", "wkd"]
PLACEBO_C8 = range(7, 64, 3)                              # §14: 7~61일 3일 간격 19개


def guard(out):
    if not DRY:
        assert pd.Timestamp.now("UTC").tz_localize(None) >= VERDICT, "판정일(2027-01-22) 전"
        assert not out.exists(), "판정은 한 번만"


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


# ── 박스 행 빌더: 라벨·국면·기하 3함수는 box-calib 빌더(cb8335d6:scripts/research_eth_box_hold_calibration_drift_20261025.py)와 같은 식.
#    동결 box_rows.parquet(= 그 빌더 rows.parquet, md5 e3363763)와 겹침 전 행 일치를 build 가 assert 한다.
def labels(h, l, c, W):
    n = len(c)
    mx, mn = sliding_window_view(h, W).max(1), sliding_window_view(l, W).min(1)
    y = np.full(n, -1, np.int8)
    i = np.arange(W - 1, n - W)
    half = 0.5 * (mx[i - W + 1] - mn[i - W + 1])
    y[i] = ((mx[i + 1] < c[i] + half) & (mn[i + 1] > c[i] - half)).astype(np.int8)
    return y


def base_rate(y, W, days=30):
    ok = (y >= 0).astype(float)
    cs, cn = np.r_[0, np.cumsum(np.where(y >= 0, y, 0))], np.r_[0, np.cumsum(ok)]
    i = np.arange(len(y))
    lo = np.maximum(i - days * 1440, 0)
    hi = np.maximum(i - W + 1, lo)
    s, m = cs[hi] - cs[lo], cn[hi] - cn[lo]
    return np.where(m >= days * 1440 / 2, s / np.maximum(m, 1), np.nan)


def base_feats(K, D):
    h, l, c = (K[k].to_numpy(float) for k in ("h", "l", "c"))
    r2 = np.r_[0, np.diff(np.log(c))] ** 2
    s24 = np.sqrt(pd.Series(r2).rolling(1440).mean().to_numpy())
    s60 = np.sqrt(pd.Series(r2).rolling(60).mean().to_numpy())
    i = D.i.to_numpy(); W = D.scale.map(WIN).to_numpy(); rng = np.empty(len(D))
    for w in WIN.values():
        m = W == w
        rng[m] = (pd.Series(h).rolling(w).max().to_numpy() - pd.Series(l).rolling(w).min().to_numpy())[i[m]]
    ts = pd.DatetimeIndex(D.ts); hr = ts.hour + ts.minute / 60
    return pd.DataFrame({"log_ds": np.log(0.5 * rng / c[i] / (s24[i] * np.sqrt(W))), "log_vr": np.log(np.maximum(s60[i], 1e-6) / s24[i]),
                         "log_rng": np.log(rng / c[i] * 1e4), "hs": np.sin(2 * np.pi * hr / 24), "hc": np.cos(2 * np.pi * hr / 24),
                         "wkd": (ts.dayofweek >= 5).astype(int)}, index=D.index)


def k1m(end, tf="1m"):
    """ETH {tf}봉 2022-01 ~ end (binance.vision · liqlab 원천 캐시 data/liqlab/raw 공유). 열 t(ms) h l c v tb."""
    import liqlab_data_20261028 as LD
    paths = [f"monthly/klines/ETHUSDT/{tf}/ETHUSDT-{tf}-{m}.zip" for m in pd.period_range("2022-01", "2026-09", freq="M").strftime("%Y-%m")] + \
            [f"daily/klines/ETHUSDT/{tf}/ETHUSDT-{tf}-{d}.zip" for d in pd.date_range("2026-10-01", end).strftime("%Y-%m-%d")]
    with cf.ThreadPoolExecutor(8) as ex:
        parts = list(ex.map(LD.fetch, paths))
    assert all(p is not None for p in parts), [q for q, p in zip(paths, parts) if p is None]
    k = pd.concat([p.set_axis(LD.KCOLS, axis=1) for p in parts])
    K = pd.DataFrame({"t": k.open_time.astype("int64"), **{a: k[b].astype(float) for a, b in
                                                          (("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume"), ("tb", "taker_buy_volume"))}})
    K = K.drop_duplicates("t").sort_values("t")
    K = K[K.t < (pd.Timestamp(end) + pd.Timedelta("1D")).value // 10 ** 6].reset_index(drop=True)
    assert (np.diff(K.t) == pd.Timedelta(tf.replace("m", "min")).value // 10 ** 6).all(), f"{tf}봉 빈칸"
    return K


def box_rows(K, events):
    from dashboard.box_hold import B as BH
    spec = importlib.util.spec_from_file_location("imp", ROOT / "scripts/research_eth_box_hold_improve_20261016.py")
    imp = importlib.util.module_from_spec(spec); spec.loader.exec_module(imp)
    imp.OUT = OUT / f"events_{TAG}"; imp.OUT.mkdir(exist_ok=True); shutil.copy(events, imp.OUT / "events.json")
    # X_ 칸(5분 패널·프리미엄 — 서버에 없음)은 배포 세 척도가 안 쓴다 → 그 블록 앞에서 끊은 features() 로
    assert not any(c.startswith("X_") for s in BH["scales"].values() for c in s["cols"])
    src = inspect.getsource(imp.features); exec(src[:src.index("    # X: 5분 패널")] + "    return X\n", imp.__dict__)
    lean = BH["scales"]["1h"]["cols"]
    t = K.t.to_numpy(); tsK = pd.to_datetime(t, unit="ms"); mins = t // 60000
    h, l, c = (K[k].to_numpy(float) for k in ("h", "l", "c"))
    start = np.searchsorted(t, pd.Timestamp("2022-01-01 23:59").value // 10 ** 6)
    rows = []
    for sc, W in WIN.items():
        y, step = labels(h, l, c, W), 60 if sc == "4h" else 5
        rngW = pd.Series(h).rolling(W).max().to_numpy() - pd.Series(l).rolling(W).min().to_numpy()
        ii = np.arange(start, len(K)); ii = ii[(mins[ii] % step == step - 1) & (y[ii] >= 0) & (rngW[ii] > 0)]
        rows.append(pd.DataFrame({"scale": sc, "ts": tsK[ii], "i": ii, "y": y[ii], "B_rate30": base_rate(y, W)[ii]}))
    D = pd.concat(rows, ignore_index=True)
    X = imp.features(D.assign(ts=D.ts), K)
    assert np.allclose(imp.PROF, np.asarray(BH["prof"], float)), "요일×시각 표가 배포와 다름"
    A = pd.concat([D, base_feats(K, D), X[[x for x in lean if x not in F_BASE]]], axis=1).replace([np.inf, -np.inf], np.nan)
    return A.drop(columns=["i"])


def card_rows(K, end):
    """card30 데이터셋 build() 를 그대로 — 5분 패널 = binance.vision 5분봉(동결 패널과 같은 원천 · 1분봉 묶음은 고가·거래량이 다른 봉이 있음,
    K 피쳐는 OHLCV·테이커만 씀) · 라벨 1분봉 = K · 산출은 OUT 아래."""
    import research_eth_card30_direction_hgb_tabpfn_20260926 as C
    k = K.set_index(pd.to_datetime(K.t, unit="ms").rename("t"))[["h", "l", "c", "v", "tb"]]
    k5 = k1m(end, "5m")
    P = pd.DataFrame({"timestamp": pd.to_datetime(k5.t, unit="ms"), "high": k5.h, "low": k5.l, "close": k5.c, "volume": k5.v, "taker_buy_base": k5.tb})
    for col in ("sum_open_interest", "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"):
        P[col] = np.nan                                                     # M 칸 — 시험 팔(K)은 안 씀
    d = OUT / f"card_{TAG}"; d.mkdir(exist_ok=True); P.to_parquet(d / "panel.parquet")
    C.PANEL, C.OUT, C.load_1m = d / "panel.parquet", d, (lambda end: k)
    C.build()
    return pd.read_parquet(d / "dataset.parquet")


def same(new, old, keys, cols, lo, hi, rtol):
    """동결본의 [lo, hi] 행 = 새 빌드 행(행 집합 양방향 · 값)."""
    o, n = old[(old.ts >= lo) & (old.ts <= hi)], new[(new.ts >= lo) & (new.ts <= hi)]
    m = o.merge(n, on=keys, how="outer", suffixes=("", "_n"), indicator=True)
    assert (m._merge == "both").all(), m._merge.value_counts().to_dict()
    for col in cols:
        assert np.allclose(m[col].astype(float), m[col + "_n"].astype(float), rtol=rtol, atol=1e-9, equal_nan=True), col
    return len(m)


def build():
    end = "2026-10-23" if DRY else "2027-01-21"
    guard(OUT / f"build_{TAG}.json")
    ev = Path(sys.argv[sys.argv.index("--events") + 1]) if "--events" in sys.argv else ROOT / "data/live/box_hold_events_20261016.json"
    E = json.loads(ev.read_text())
    assert pd.Timestamp(max(E["claims"]), unit="ms") >= EV1, f"발표 일정이 창 끝({EV1})까지 없음 — 2027 갱신본을 --events 로"
    K = k1m(end); rep = {"k1m": [str(pd.Timestamp(K.t.iloc[0], unit="ms")), str(pd.Timestamp(K.t.iloc[-1], unit="ms")), len(K)],
                         "events": str(ev), "events_md5": md5(ev), "frozen_md5": {k: md5(v) for k, v in FROZEN.items()}}
    from dashboard.box_hold import B as BH
    nb, ob = box_rows(K, ev), pd.read_parquet(FROZEN["box"]); hb = ob.ts.max() - pd.Timedelta("4h")   # 동결본 끝 4h 는 라벨 미해결로 빠진 행
    rep["box_overlap_rows"] = same(nb, ob, ["scale", "ts"], ["y", "B_rate30"] + list(BH["scales"]["1h"]["cols"]), pd.Timestamp("2022-03-01"), hb, 1e-9)
    K_cols = json.loads((FROZEN["card"].parent / "feats.json").read_text())["K"]
    nc, oc = card_rows(K, end), pd.read_parquet(FROZEN["card"]); hc = oc.ts.max() - pd.Timedelta("1h")    # 동결본 끝 행은 1분봉이 끊겨 res −9
    rep["card_overlap_rows"] = same(nc, oc, ["ts"], ["res"] + K_cols, pd.Timestamp("2022-03-01"), hc, 1e-6)
    for nm, n, o, hi in (("box", nb, ob, hb), ("card", nc, oc, hc)):
        # 학습·검증 = 동결본 · 채점 창 = 새 빌드. 새 빌드는 이동 합 누적 차(상대 ~2e-9)로 HGB 분기 경계 행(박스 1h 0.9%) 예측이 바뀐다 →
        # dry = 동결본 그대로(채점 코드 정확 재현) · dryrb = 시험창만 새 빌드(겹침 끝 뒤는 동결본) = 빌더 잡음이 판정 수치를 얼마나 움직이는지
        s = pd.concat([o[o.ts < EV0], n[n.ts >= EV0]] if not DRY else [o[o.ts < EV0], n[(n.ts >= EV0) & (n.ts <= hi)], o[o.ts > hi]], ignore_index=True)
        for tag, x in ((TAG, s),) if not DRY else (("dry", o), ("dryrb", s)):
            x.to_parquet(OUT / f"{nm}_rows_{tag}.parquet"); rep[f"{nm}_rows_{tag}"] = [len(x), int((x.ts >= EV0).sum()), str(x.ts.max())]
    (OUT / f"build_{TAG}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(rep, ensure_ascii=False, default=str))


# ── C8 박스 1h + P2 (§14 관문): 배포 A1 그대로 대비 · 매월 굴림 재적합(직전 90일 해결 행) · 단측 90% 상한 < 0 · 지연 위약 19개 95분위
def c8():
    os.environ["LIQLAB_D"] = "data/liqlab" if DRY else "data/liqlab_fwd0122"
    import liqlab_run_20261028 as R
    from sklearn.linear_model import LogisticRegression
    from dashboard.box_hold import B as BH, PLATT90
    out = OUT / f"c8_{TAG}.json"; guard(out)
    data = {s: R.panel(s) for s in ("ETHUSDT", "BTCUSDT")}
    assert all(p.index[-1] >= EV1 - pd.Timedelta("5min") for p, _ in data.values()), "패널이 창 끝까지 없음"
    cf_ = BH["scales"]["1h"]; cols = list(cf_["cols"]); H = pd.Timedelta(minutes=61)
    g = pd.read_parquet(OUT / f"box_rows_{TAG}.parquet", columns=["scale", "ts", "y"] + cols)
    g = g[(g.scale == "1h") & (g.ts >= "2025-06-01")].dropna(subset=cols).reset_index(drop=True)
    ts, y = pd.DatetimeIndex(g.ts), g.y.to_numpy(float)
    pA, ab = R.rolling_platt90(ts, cf_["model"].predict_proba(g[cols])[:, 1], y)          # 배포 A1 그대로(월별 Platt90 재생)
    lz, X0, P2 = R._logit(pA)[:, None], R.pred_frame("ETHUSDT", data), R.P_ADD["P2"]
    ev = (ts >= EV0) & (ts + H <= EV1)
    assert np.isfinite(pA[ev]).all() and ev.sum() > 1000
    va = (ts >= "2026-01-01") & (ts + H <= pd.Timestamp("2026-06-01"))                   # §13 검증 고정 적합 계수 재현(incr.json 대조)
    Z, _ = R._z(X0.reindex(ts - pd.Timedelta("4min")), P2, va)
    coef = LogisticRegression(C=1e6, max_iter=4000).fit(np.c_[lz, Z][va], y[va]).coef_[0][1:]
    inc = json.loads((ROOT / "tmp/liqlab_20261028/incr.json").read_text())["box"]["P2"]["coef"]
    assert np.allclose(np.round(coef, 5), [inc[c] for c in P2], atol=2e-5), (coef, inc)
    months = pd.date_range(EV0.replace(day=1), EV1 - pd.Timedelta("1min"), freq="MS")

    def roll(F):
        pa, pb = np.full(len(y), np.nan), np.full(len(y), np.nan)
        for T in months:
            fit = (ts >= T - pd.Timedelta(days=90)) & (ts + H <= T); app = (ts >= T) & (ts < T + pd.offsets.MonthBegin(1))
            Z, _ = R._z(F, P2, fit); X = np.c_[lz, Z]
            pb[app] = LogisticRegression(C=1e6, max_iter=2000).fit(lz[fit], y[fit]).predict_proba(lz[app])[:, 1]
            pa[app] = LogisticRegression(C=1e6, max_iter=4000).fit(X[fit], y[fit]).predict_proba(X[app])[:, 1]
        return pa, pb

    def d(p, q):
        return R._upper90(((p - y) ** 2 - (q - y) ** 2)[ev], ts[ev])

    F = lambda k: X0.reindex(ts - pd.Timedelta("4min") - pd.Timedelta(days=k))  # noqa: E731
    pa, pb = roll(F(0)); main, vs_roll = d(pa, pA), d(pa, pb)
    pl = {k: roll(F(k))[0] for k in PLACEBO_C8}
    plA, plR = {k: d(p, pA)[0] for k, p in pl.items()}, {k: d(p, pb)[0] for k, p in pl.items()}
    pct = float(np.mean([main[0] < v for v in plA.values()]) * 100)
    rep = {"window": [str(EV0), str(EV1)], "rows": int(ev.sum()), "days": int(pd.Series(ts[ev].normalize()).nunique()),
           "brier_A1_asis": float(np.mean(((pA - y) ** 2)[ev])), "brier_cand": float(np.mean(((pa - y) ** 2)[ev])),
           "main_vs_A1": {"d_brier": main[0], "up90": main[1]}, "vs_rolling_base": {"d_brier": vs_roll[0], "up90": vs_roll[1]},
           "placebo_vs_A1": plA, "placebo_pct_worse": pct, "valfit_coef": dict(zip(P2, np.round(coef, 5))),
           "platt_last": {"replay": ab.get(str(months[-1].date())), "served": json.loads(PLATT90.read_text()) if PLATT90.exists() else None}}
    rep["gate"] = {"up90<0": bool(main[1] < 0), "placebo_pct>=95": bool(pct >= 95)}
    rep["verdict"] = "통과" if all(rep["gate"].values()) else "불통과"
    if TAG == "dry":                                                        # 검수 review_box2.json(시험창 굴림 재적합) 재현
        rv = json.loads((ROOT / "tmp/liqlab_review_pred/review_box2.json").read_text())
        assert np.isclose(vs_roll[0], rv["rolling_P2"]["db"], rtol=1e-6) and np.isclose(vs_roll[1], rv["rolling_P2"]["u90"], rtol=1e-6), (vs_roll, rv["rolling_P2"])
        assert np.isclose(rep["brier_A1_asis"], rv["rolling_P2"]["brier_A1"], rtol=1e-9)
        assert all(abs(plR[r["k"]] - r["db"]) < 1.5e-6 for r in rv["rolling_lag_placebo"]["rows"])
        rep["dry_repro"] = "review_box2 rolling_P2·위약 19개·A1 Brier · incr.json 계수 일치"
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({k: rep[k] for k in ("rows", "main_vs_A1", "vs_rolling_base", "placebo_pct_worse", "gate", "verdict")}, ensure_ascii=False, default=float))


# ── C9 깊이 후보 3팔: depth_incr 의 같은 코드(학습 2024~25 · 같은 시드 · 같은 위약 9개 · 같은 grade)를 채점 창만 바꿔서
def c9():
    out = OUT / f"c9_{TAG}"; guard(out / "summary.json")
    os.environ.update(DEPTH_OUT=str(out.relative_to(ROOT)), DEPTH_LIQ="data/liqlab" if DRY else "data/liqlab_fwd0122",
                      DEPTH_TEST0=str(EV0), DEPTH_TEST1=str(EV1),
                      DEPTH_BOX_ROWS=str((OUT / f"box_rows_{TAG}.parquet").relative_to(ROOT)), DEPTH_CARD_ROWS=str((OUT / f"card_rows_{TAG}.parquet").relative_to(ROOT)))
    import depth_incr_20261028 as DI
    assert DI.__file__.startswith(str(HERE)), DI.__file__
    if not DRY:                                                             # 전진 패널의 깊이 = 동결 패널 깊이(겹침) · 창 끝까지
        a, b = DI.depth("ETH"), DI.depth_from(pd.read_parquet(ROOT / "data/liqlab/ETHUSDT_5m.parquet"))
        assert np.allclose(a.reindex(b.index).to_numpy(float), b.to_numpy(float), equal_nan=True) and a.index[-1] >= EV1 - pd.Timedelta("5min")
    DI.box("30m", "4h"); DI.card30("reach")
    names = ("box_30m", "box_4h", "card30_reach")
    rep = {n: json.loads((out / f"{n}.json").read_text())["verdict"] for n in names}
    for v in rep.values():
        v["kept"] = v["grade"] in ("후보", "엄격 통과")
    if TAG == "dry":                                                        # 원 시험 JSON 재현(검증·시험 차 · 위약 9개)
        for n in names:
            a, b = json.loads((out / f"{n}.json").read_text()), json.loads((ROOT / f"tmp/depth_incr_20261028/{n}.json").read_text())
            for w in ("val", "test"):
                assert np.isclose(a["real"][w]["mean"], b["real"][w]["mean"], rtol=1e-9, atol=1e-12), (n, w)
            assert np.allclose([a["placebo"][k] for k in b["placebo"]], list(b["placebo"].values()), rtol=1e-9, atol=1e-12), n
        rep["dry_repro"] = "depth_incr box_30m·box_4h·card30_reach 검증·시험 차와 위약 9개 일치"
    (out / "summary.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
    print(json.dumps(rep, ensure_ascii=False, default=float))


if __name__ == "__main__":
    {"build": build, "c8": c8, "c9": c9}[sys.argv[1]]()
