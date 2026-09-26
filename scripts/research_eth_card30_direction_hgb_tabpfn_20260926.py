#!/usr/bin/env python3
"""30분 카드 «닿는다면 위 먼저 vs 아래 먼저» 방향 모델 — HGB vs TabPFN (2026-09-26, 사용자 요청).

왜: 카드 `flow_read.CARD_CELLS` 는 위·아래 격차가 가중 평균 2.1pp 라 «항상 비슷»했다. 5분봉 라벨 상한 GBM AUC .525.
    카드 개편(닿을 확률 ↔ 방향 분리) 전에 방향 모델을 **카드와 같은 1분봉 선착 라벨**로 다시 잰다.
라벨: 결정 = 5분봉 i 마감(t = open_i + 5분). 기준가 close_i, 배리어 ± 0.5 × (봉 i-5..i 고저폭). 1분봉 open ≥ t 30개에서
      먼저 닿는 쪽. 같은 1분봉 양쪽 = 제외, 미도달 = 방향 과제에서 제외(닿을 확률은 카드 표 g 가 맡는다).
피쳐(전부 open < t 만): K = 5분봉 OHLCV·테이커 · M1 = 마지막 1분봉들 · M = metrics(OI·롱숏비, 한 봉 더 지연 —
      2026-07-12 create_time 규약 전환 대비 보수). 라이브 파리티: K·M1 은 klines REST 로 그대로 재현 가능, M 은 별도 배관.
분할: TRAIN < 2025-01 ≤ TEST (카드 표 연구와 같음). CI = 일 블록 부트스트랩. 연구 점수이지 승격 근거가 아니다.
팔: logit(6피쳐 기준선) · hgb(전체) · hgb_sub(TabPFN 과 같은 컨텍스트) · tabpfn — (hgb_sub − tabpfn) 이 모델 효과.

    python scripts/research_eth_card30_direction_hgb_tabpfn_20260926.py build          # 로컬(네트워크)
    python scripts/... fit --model hgb --feats K,KM1,KM1M                              # 로컬 CPU
    python scripts/... fit --model tabpfn --feats KM1M --cap 10000                     # 서버 GPU
    python scripts/... report
    python scripts/... parity      # 라이브 피쳐 함수 ↔ 데이터셋 대조(배포 전 필수)
    python scripts/... export      # 보정 온도 + 전 구간 재학습 → tmp/card30_direction_20260926/model.joblib
    python scripts/... votes       # 융합 4표 재구성(2023~, 서버 정의) → KV/Kv 팔 · fire(발동 봉 결합)
결과(09-26): 표를 피쳐로 넣어도 KV .5248 = Kv .5251(같은 행) → 증분 0. 발동 봉에선 모델·발동빈도·평균이 동률(CI 0 포함)이고
      평균이 «융합 롱인데 모델 반대» 10% → 0.2% 로 화면 모순을 없앤다 → 라이브는 발동 중에만 평균.
"""
from __future__ import annotations
import argparse, io, json, zipfile, urllib.error, urllib.request
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/binance_vision/panel/ETHUSDT.parquet"
M1DIR = ROOT / "data/binance_vision/klines1m"
OUT = ROOT / "tmp/card30_direction_20260926"
VISION = "https://data.binance.vision/data/futures/um"
SPLIT = pd.Timestamp("2025-01-01")
FEATS = {"K": [], "M1": ["m1_r1", "m1_r2", "m1_rv30", "m1_imb1"],
         "M": ["oi6", "oi12", "oi48", "tt", "tt_d12", "gls", "gls_d12", "tkr"]}
LOGIT6 = ["ret6", "pos24", "sma144", "tt", "gls", "gls_d12"]


def _zip_csv(url: str) -> pd.DataFrame | None:
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    raw = z.read(z.namelist()[0]).decode()
    d = pd.read_csv(io.StringIO(raw), header=0 if raw.startswith("open_time") else None)
    d = d.iloc[:, [0, 2, 3, 4, 5, 9]]                      # open_time high low close volume taker_buy_volume
    d.columns = ["t", "h", "l", "c", "v", "tb"]
    return d


def load_1m(end: pd.Timestamp) -> pd.DataFrame:
    """data.binance.vision 1분봉(월 파일, 월 파일이 아직 없는 달은 일 파일). 캐시는 월별 parquet."""
    M1DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for m in pd.period_range("2022-01", end.to_period("M"), freq="M"):
        f = M1DIR / f"ETHUSDT-1m-{m}.parquet"
        if not f.exists():
            d = _zip_csv(f"{VISION}/monthly/klines/ETHUSDT/1m/ETHUSDT-1m-{m}.zip")
            if d is None:
                days = pd.date_range(m.start_time, min(m.end_time, end), freq="D")
                ds = [_zip_csv(f"{VISION}/daily/klines/ETHUSDT/1m/ETHUSDT-1m-{x:%Y-%m-%d}.zip") for x in days]
                d = pd.concat([x for x in ds if x is not None])
            d.to_parquet(f)
            print(f"  1m {m}: {len(d):,}행", flush=True)
        parts.append(pd.read_parquet(f))
    d = pd.concat(parts).drop_duplicates("t").sort_values("t")
    d["t"] = pd.to_datetime(d["t"], unit="ms")
    return d.set_index("t").astype(float)


def build() -> None:
    P = pd.read_parquet(PANEL)
    ts = pd.DatetimeIndex(pd.to_datetime(P.timestamp.values))
    h, l, c, v = (P[k].to_numpy(float) for k in ("high", "low", "close", "volume"))
    S = pd.Series
    F: dict[str, np.ndarray] = {}
    lr = np.log(c)
    for k in (1, 3, 6, 12, 24, 48, 144, 288):
        F[f"ret{k}"] = (lr - S(lr).shift(k).values) * 1e4
    for N in (6, 24, 144, 288):
        H, L = S(h).rolling(N).max().values, S(l).rolling(N).min().values
        F[f"pos{N}"] = (c - L) / np.maximum(H - L, 1e-9)
    atr = S(h - l).rolling(144).mean().values
    for N in (48, 144, 288):
        F[f"sma{N}"] = (c - S(c).rolling(N).mean().values) / atr
    imb = 2 * P.taker_buy_base.to_numpy(float) / np.maximum(v, 1e-9) - 1
    for k in (1, 6, 12, 48):
        F[f"imb{k}"] = S(imb * v).rolling(k).sum().values / np.maximum(S(v).rolling(k).sum().values, 1e-9)
    rg = (S(h).rolling(6).max().values - S(l).rolling(6).min().values) / c * 1e4
    F["rg30"] = rg
    F["rgq"] = S(rg).rolling(288).rank(pct=True).values           # 라이브 range30_pct 와 같은 정의(자기 포함)
    F["volz"] = (v - S(v).rolling(288).mean().values) / S(v).rolling(288).std().values
    pc = S(c).shift(1).values
    F["last_body"] = (c - pc) / np.maximum(h - l, 1e-9)
    F["wick"] = ((h - np.maximum(c, pc)) - (np.minimum(c, pc) - l)) / np.maximum(h - l, 1e-9)
    F["hour"], F["dow"] = ts.hour.values.astype(float), ts.dayofweek.values.astype(float)
    FEATS["K"][:] = list(F)
    Mx = P[["sum_open_interest", "sum_toptrader_long_short_ratio", "count_long_short_ratio",
            "sum_taker_long_short_vol_ratio"]].shift(1)
    oi = np.log(Mx.sum_open_interest.where(Mx.sum_open_interest > 0).to_numpy(float))
    for k in (6, 12, 48):
        F[f"oi{k}"] = (oi - S(oi).shift(k).values) * 1e4
    F["tt"] = Mx.sum_toptrader_long_short_ratio.to_numpy(float); F["tt_d12"] = F["tt"] - S(F["tt"]).shift(12).values
    F["gls"] = Mx.count_long_short_ratio.to_numpy(float); F["gls_d12"] = F["gls"] - S(F["gls"]).shift(12).values
    F["tkr"] = Mx.sum_taker_long_short_vol_ratio.to_numpy(float)
    X = pd.DataFrame(F)

    m1 = load_1m(ts[-1])
    grid = pd.date_range(m1.index[0], ts[-1] + pd.Timedelta(minutes=40), freq="1min")
    m1 = m1[~m1.index.duplicated()].reindex(grid)
    mh, ml, mc = (m1[k].to_numpy(float) for k in ("h", "l", "c"))
    mimb = 2 * m1.tb.to_numpy(float) / np.maximum(m1.v.to_numpy(float), 1e-9) - 1
    p = grid.get_indexer(ts + pd.Timedelta(minutes=5))           # 결정 시각 = 라벨 첫 1분봉
    ok = (p >= 31) & (p + 30 <= len(grid))
    pp = np.where(ok, p, 31)
    same = np.abs(mc[pp - 1] / c - 1) < 1e-6
    assert same[ok & np.isfinite(mc[pp - 1])].mean() > 0.995, "5분봉 종가 ≠ 직전 1분봉 종가 — 정렬 오류"
    W = np.lib.stride_tricks.sliding_window_view
    H30, L30 = W(mh, 30)[pp], W(ml, 30)[pp]
    up, dn = c * (1 + 0.5 * rg / 1e4), c * (1 - 0.5 * rg / 1e4)
    hu, hd = H30 >= up[:, None], L30 <= dn[:, None]
    fu = np.where(hu.any(1), hu.argmax(1), 99); fd = np.where(hd.any(1), hd.argmax(1), 99)
    gap = np.isnan(H30).any(1) | np.isnan(L30).any(1)
    res = np.where((fu == 99) & (fd == 99), 2, np.where(fu == fd, -1, (fu < fd).astype(int)))   # 1 위 0 아래 2 미도달 -1 모호
    res[~ok | gap | ~np.isfinite(rg) | (rg <= 0)] = -9
    lc = np.log(mc)
    X["m1_r1"] = (lc[pp - 1] - lc[pp - 2]) * 1e4
    X["m1_r2"] = (lc[pp - 1] - lc[pp - 3]) * 1e4
    X["m1_rv30"] = W(np.diff(lc, prepend=np.nan), 30)[pp - 30].std(1) * 1e4   # 1분 수익률 30개(마지막 = pp-1)
    X["m1_imb1"] = mimb[pp - 1]
    X.loc[~ok, FEATS["M1"]] = np.nan
    X = X.replace([np.inf, -np.inf], np.nan)
    X["ts"], X["res"] = ts, res
    X = X.iloc[300:]
    OUT.mkdir(parents=True, exist_ok=True)
    X.to_parquet(OUT / "dataset.parquet")
    print(f"결정 {len(X):,} · {X.ts.min()} ~ {X.ts.max()} · 결과 {X.res.value_counts(normalize=True).round(4).to_dict()}")
    json.dump(FEATS, open(OUT / "feats.json", "w"), ensure_ascii=False)


def _load(feats: str):
    """feats: L6 | K[M1][M] · 'V' = 융합 표 피쳐 추가 · 'v'/'V' 가 있으면 표가 알려진 행(2023~)만 — Kv 와 KV 는 같은 행."""
    X = pd.read_parquet(OUT / "dataset.parquet")
    FE = json.load(open(OUT / "feats.json"))
    cols = LOGIT6 if feats == "L6" else \
        FE["K"] + (FE["M1"] if "M1" in feats else []) + (FE["M"] if feats.endswith("M") else [])
    if "v" in feats.lower():
        X = X.merge(pd.read_parquet(OUT / "votes.parquet"), on="ts", how="left").set_axis(X.index)
        X = X[X.v_score.notna()]
        cols = cols + (VOTE_COLS if "V" in feats else [])
    X = X[X.res.isin((0, 1))]
    return X, cols, (X.ts < SPLIT).to_numpy(), (X.ts >= SPLIT).to_numpy()


def fit(model: str, feats: str, cap: int, seeds: list[int]) -> None:
    X, cols, tr, te = _load(feats)
    y = X.res.to_numpy(int)
    Xtr, Xte, ytr = X.loc[tr, cols].to_numpy(np.float32), X.loc[te, cols].to_numpy(np.float32), y[tr]
    preds = {}
    for s in seeds:
        idx = np.arange(len(ytr))
        if model in ("hgb_sub", "tabpfn") and len(idx) > cap:     # 같은 시드 = hgb_sub 와 tabpfn 이 같은 컨텍스트
            idx = np.sort(np.random.default_rng(s).choice(idx, cap, replace=False))
        if model == "logit":
            from sklearn.pipeline import make_pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            m = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=1000))
        elif model in ("hgb", "hgb_sub"):
            from sklearn.ensemble import HistGradientBoostingClassifier
            m = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=200,
                                               l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                                               n_iter_no_change=30, random_state=s)
        elif model == "tabpfn":
            import torch
            from tabpfn import TabPFNClassifier
            m = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu", random_state=s,
                                 n_estimators=4, ignore_pretraining_limits=True)
        else:
            raise SystemExit(f"모르는 모델 {model}")
        m.fit(Xtr[idx], ytr[idx])
        preds[f"s{s}"] = np.concatenate([m.predict_proba(Xte[i:i + 4000])[:, 1] for i in range(0, len(Xte), 4000)])
        print(f"{model} {feats} seed {s}: n_ctx {len(idx):,} · TEST 평균 p {preds[f's{s}'].mean():.4f}", flush=True)
        if model == "logit":
            break
    tag = f"{model}_{feats}" + (f"_c{cap}" if model in ("hgb_sub", "tabpfn") else "")
    pd.DataFrame(preds, index=X.index[te]).to_parquet(OUT / f"pred_{tag}.parquet")   # 행 = 데이터셋 인덱스(팔마다 행이 다를 수 있다)
    json.dump({"seeds": seeds, "cols": cols, "cap": cap}, open(OUT / f"pred_{tag}.json", "w"))


def report() -> None:
    from sklearn.metrics import roc_auc_score as auc
    X, _, _, te0 = _load("K")
    XT = X[te0]
    print(f"TEST 결정(닿은 것만) {len(XT):,} · 위 먼저 {XT.res.mean():.4f}")
    rng = np.random.default_rng(7)
    rows = []
    for f in sorted(OUT.glob("pred_*.parquet")):
        P = pd.read_parquet(f)
        if isinstance(P.index, pd.RangeIndex) and len(P) == len(XT):   # 행 인덱스 저장 전 파일 = K 전체 TEST 순서
            P.index = XT.index
        Xa = XT.loc[P.index]
        y = Xa.res.to_numpy(int)
        u, inv = np.unique(Xa.ts.dt.normalize().to_numpy(), return_inverse=True)
        B = [np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv] for _ in range(200)]
        yr = Xa.ts.dt.year.to_numpy()
        base = y.mean()
        ll0 = -np.mean(y * np.log(base) + (1 - y) * np.log(1 - base))
        aucs = [auc(y, P[c]) for c in P.columns]
        p = P.mean(1).to_numpy()                                   # 시드 평균
        ci = np.percentile([auc(y, p, sample_weight=w) for w in B], [2.5, 97.5])
        ll = -np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
        s5 = np.abs(p - .5) >= .05
        dec = pd.qcut(p, 10, labels=False, duplicates="drop")
        rows.append(dict(arm=f.stem[5:], n=len(y), auc=auc(y, p), ci=f"[{ci[0]:.4f},{ci[1]:.4f}]",
                         seeds=f"{min(aucs):.4f}~{max(aucs):.4f}" if len(aucs) > 1 else "-",
                         y2025=auc(y[yr == 2025], p[yr == 2025]), y2026=auc(y[yr == 2026], p[yr == 2026]),
                         dll_x1e3=1e3 * (ll0 - ll), ge5pp=s5.mean(), acc5=((p[s5] > .5) == y[s5]).mean() if s5.any() else np.nan,
                         d0=y[dec == 0].mean(), d9=y[dec == dec.max()].mean()))
    R = pd.DataFrame(rows).set_index("arm")
    pd.set_option("display.width", 220)
    print(R.round(4).to_string())
    R.to_csv(OUT / "report.csv")


VOTE_COLS = ["v_whale", "v_oi", "v_al", "v_rj", "v_score", "dir30"]


def votes() -> None:
    """융합 4표 + 카드 30분 방향을 과거 전체로 재구성 → votes.parquet. 서버와 같은 정의(FUSED_Z=24h, 고래 한 표 =
    리테일 갈림 우선 아니면 중형). 입력 tmp/whale/tape/ETHUSDT(scripts/build_aggtrades_size3_tape_1m_20260925.py, 2023~).
    연구↔라이브 flow_read.fuse 패리티는 09-25 에 21,151/21,151 봉으로 확인됐다(같은 정의를 그대로 쓴다)."""
    import os, sys
    os.environ["FUSED_Z"] = "24h"
    os.chdir(ROOT)
    src = (ROOT / "scripts/research_eth_fused_signal_votes_gate_20260925.py").read_text().split("for H in (")[0]
    g: dict = {}
    sys.argv = ["x", "ETHUSDT"]
    exec(src, g)
    V = g["V"]
    whale = np.where(V["wr"] != 0, V["wr"], V["wm"])
    d = pd.DataFrame({"ts": g["ts"].tz_convert(None), "v_whale": whale, "v_oi": V["oi"], "v_al": V["al"], "v_rj": V["rj"],
                      "dir30": g["dir30"]})
    d["v_score"] = d.v_whale + d.v_oi + d.v_al + d.v_rj
    zok = np.isfinite(g["Z"]["whl"]) & np.isfinite(g["Z"]["ret"])       # 고래 z 가 아직 없으면(워밍업) 표를 모름으로
    d.loc[~zok, VOTE_COLS] = np.nan
    d.to_parquet(OUT / "votes.parquet")
    print(f"표 {len(d):,}봉 · {d.ts.min()} ~ {d.ts.max()} · 켜진 비율 " +
          " ".join(f"{c} {(d[c].fillna(0) != 0).mean():.3f}" for c in ("v_whale", "v_oi", "v_al", "v_rj")))


def fire() -> None:
    """융합 발동 봉(|4표 합|≥2 & 30분 폭 24h 분위≥2/3)에서 방향: 모델 · 발동 실측 빈도 · 둘의 평균.
    발동 빈도 상수는 TRAIN(2023-02~2024)에서만 잰다 → TEST(2025~) 에서 로그손실 차를 일 블록 CI 로."""
    import joblib
    X = pd.read_parquet(OUT / "dataset.parquet").pipe(lambda d: d.merge(pd.read_parquet(OUT / "votes.parquet"), on="ts",
                                                                          how="left").set_axis(d.index))
    hit, te = X.res.isin((0, 1)), X.ts >= SPLIT
    on = (X.v_score.abs() >= 2) & (X.rgq >= 2 / 3)
    Zt = X[hit & on & (X.ts >= "2023-02-01") & ~te]
    wt = float(((Zt.res == 1) == (Zt.v_score > 0)).mean())
    pK = pd.read_parquet(OUT / "pred_hgb_K.parquet").mean(1)
    pK.index = X.index[hit & te]                                    # TRAIN<2025 모델의 TEST 예측
    M = joblib.load(OUT / "model.joblib")
    lg = lambda q: np.log(q / (1 - q))                              # noqa: E731
    pc = 1 / (1 + np.exp(-(M["k"] * lg(pK) + (1 - M["k"]) * lg(M["base"]))))
    Z = X[hit & on & te]
    sd = np.sign(Z.v_score).to_numpy()
    y = ((Z.res == 1).to_numpy() == (sd > 0)).astype(int)
    ps = np.where(sd > 0, pc.loc[Z.index], 1 - pc.loc[Z.index])
    L = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))       # noqa: E731
    u, inv = np.unique(Z.ts.dt.normalize().to_numpy(), return_inverse=True)
    rng = np.random.default_rng(2)
    print(f"TRAIN 발동 {len(Zt):,} · 융합 쪽 먼저 {wt:.3f} || TEST 발동 {len(Z):,}·{len(u)}일 · 실현 {y.mean():.3f} · 모델 평균 {ps.mean():.3f}")
    for nm, pp in (("모델만", ps), ("발동 빈도만", np.full(len(y), wt)), ("평균", (ps + wt) / 2)):
        d = L(pp) - L(ps)
        s_, c_ = np.bincount(inv, d), np.bincount(inv)
        bs = [s_[k].sum() / c_[k].sum() for k in (rng.integers(0, len(u), len(u)) for _ in range(2000))]
        print(f"  {nm:8s} Δ로그손실 {1e3 * d.mean():+.2f}e-3 [{1e3 * np.percentile(bs, 2.5):+.2f},{1e3 * np.percentile(bs, 97.5):+.2f}]"
              f" · 동전 {np.mean(np.abs(pp - .5) < .05):.3f} · 반대 {np.mean(pp < .5):.3f}")


def _hgb(s: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=200,
                                          l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                                          n_iter_no_change=30, random_state=s)


def export(seeds: list[int]) -> None:
    """배포 모델: HGB·K. 보정 온도 k 는 «2024-07 전 학습 → 2024 하반기에서 맞춤 → TEST 2025~ 확인» 으로 정하고,
    최종 모델은 전 구간(패널 끝까지)으로 다시 학습해 같은 k 를 쓴다. p_cal = σ(k·logit(p) + (1−k)·logit(base))."""
    import joblib
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import roc_auc_score as auc
    X, cols, _, te = _load("K")
    y = X.res.to_numpy(int)
    tr0, ca = (X.ts < "2024-07-01").to_numpy(), ((X.ts >= "2024-07-01") & (X.ts < SPLIT)).to_numpy()
    lg = lambda p: np.log(p / (1 - p))                                          # noqa: E731
    sg = lambda z: 1 / (1 + np.exp(-z))                                         # noqa: E731
    ll = lambda yy, p: -np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p))     # noqa: E731
    ms = [_hgb(s).fit(X.loc[tr0, cols], y[tr0]) for s in seeds]
    pc = np.mean([m.predict_proba(X.loc[ca, cols])[:, 1] for m in ms], 0)
    pt = np.mean([m.predict_proba(X.loc[te, cols])[:, 1] for m in ms], 0)
    base = float(y[ca].mean())
    k = float(minimize_scalar(lambda k: ll(y[ca], sg(k * lg(pc) + (1 - k) * lg(base))), bounds=(0, 2), method="bounded").x)
    yt, pk = y[te], sg(k * lg(pt) + (1 - k) * lg(base))
    s5 = np.abs(pk - .5) >= .05
    rep = dict(k=k, base=base, test_auc=auc(yt, pk), test_dll_x1e3=1e3 * (ll(yt, np.full(len(yt), yt.mean())) - ll(yt, pk)),
               ge5pp=float(s5.mean()), acc5=float(((pk[s5] > .5) == yt[s5]).mean()))
    print("보정 프로토콜(TEST 2025~):", {a: round(b, 4) for a, b in rep.items()})
    final = [_hgb(s).fit(X[cols], y) for s in seeds]
    joblib.dump({"models": final, "k": k, "base": base, "cols": cols, "seeds": seeds,
                 "trained_through": str(X.ts.max()), "report": rep}, OUT / "model.joblib")
    print(f"저장 {OUT / 'model.joblib'} · 학습 {len(X):,}결정 · ~{X.ts.max()}")


def parity(n: int = 400) -> None:
    """라이브 피쳐 함수(dashboard.flow_read.card30_features) ↔ 연구 데이터셋 — 무작위 결정 n 개에서 26피쳐 전부 대조."""
    import sys
    sys.path.insert(0, str(ROOT))
    from dashboard.flow_read import card30_features
    P = pd.read_parquet(PANEL)
    X = pd.read_parquet(OUT / "dataset.parquet")
    cols = json.load(open(OUT / "feats.json"))["K"]
    t = (pd.to_datetime(P.timestamp.values).asi8 // 10**9)
    bars = [dict(time=int(a), high=b, low=c, close=d, volume=e, taker=f) for a, b, c, d, e, f in
            zip(t, P.high, P.low, P.close, P.volume, P.taker_buy_base)]
    rng = np.random.default_rng(3)
    idx = rng.choice(X.index[X.res.isin((0, 1))].to_numpy(), n, replace=False)
    worst, skipped = 0.0, 0
    for i in idx:
        f = card30_features(bars[i - 292:i + 1])
        if f is None:                                   # 패널 구멍(5분 간격 끊김) — 라이브도 None 을 낸다
            skipped += 1
            continue
        a, b = np.array([f[c] for c in cols]), X.loc[i, cols].to_numpy(float)
        ok = np.isfinite(b)
        err = np.abs(a[ok] - b[ok]) / np.maximum(np.abs(b[ok]), 1.0)
        worst = max(worst, float(err.max()))
        assert err.max() < 1e-6, (i, [(c, x, z) for c, x, z in zip(np.array(cols)[ok], a[ok], b[ok]) if abs(x - z) > 1e-6 * max(abs(z), 1)])
    print(f"파리티 OK · {n - skipped}/{n} 결정 · 26피쳐 최대 상대오차 {worst:.2e} · 구멍 건너뜀 {skipped}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "votes", "fit", "report", "export", "parity", "fire"])
    ap.add_argument("--model", default="hgb")
    ap.add_argument("--feats", default="K,KM1,KM1M")
    ap.add_argument("--cap", type=int, default=10000)
    ap.add_argument("--seeds", default="")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "votes":
        votes()
    elif a.cmd in ("fit", "export"):
        # 시드는 고정 간격이 아니라 무작위 추출(시드 다양성 정책) — 기본값은 한 번 뽑아 결과 json 에 기록한다.
        seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else \
            [int(x) for x in np.random.default_rng(20260926).integers(1, 2**31 - 1, 5)]
        if a.cmd == "export":
            export(seeds)
        for fs in (a.feats.split(",") if a.cmd == "fit" else []):
            fit(a.model, fs, a.cap, seeds)
    elif a.cmd == "parity":
        parity()
    elif a.cmd == "fire":
        fire()
    else:
        report()
