"""대시보드 원소(장기 재현 가능분) + 문헌 아이디어로 «최고의 방향 예측 모델» — 걷기 앞으로 검증 (2026-10-10 05:20~09:30 KST 목표 세션).
사용자 «외부 문헌이나 논문들을 학습해서 너의 창의적인 아이디어로 위 대시보드에 있는 원소들을 가지고 최고의 방향성 예측 모델을 만들어줘».

데이터(로컬, 2022-01~2026-09-14 · 마이크로구조·호가 깊이는 2024-01~):
  panel/{20코인}.parquet 5분(시작 시각) OHLCV·테이커 매수·OI·롱숏 비율 · aggfeat ETH 5분(카일 람다·VPIN·고래/리테일 불균형 등, Easley 외 2024 계열)
  · ETHUSDT_bookDepth 일별(±1~5% 누적 깊이, 스냅샷 시각) · eth_premium_5m(베이시스) · eth_spot_5m(현물 OHLC·테이커 매수).
결정 시각 T = 5분봉 k 마감. 피쳐는 T 에 아는 값만:
  봉 k 까지의 가격·체결 · metrics OI/롱숏 = 2024-03-04 전 행 k, 이후 행 k−1(스탬프+5분 스냅샷 규약, 09-27 확인) · aggfeat 행 k(봉 시작 표기) ·
  bookDepth 스냅샷 ≤ T · 프리미엄·현물 봉 k · 일봉 추세 = T 이전에 닫힌 UTC 일.
목표 y_h = ln(close[k+h]/close[k]) (bp), h = 3(15분)·12(1h)·48(4h)·288(1일) 봉. 진입 = 봉 k 종가(봉 마감에 시장가), 경로는 k+1 부터.
트립와이어: aggfeat tf_cvd 가 같은 봉 수익과 상관(봉 시작 표기 확인)·다음 봉과는 ≈0 이어야 한다(아니면 빌드 중단).

검증(사전 고정): 걷기 앞으로 — 학습 = 시험 해 이전 전부(정화 간격 = 지평), 시험 해 2024·2025·2026. 주 판정 = 2025-01~2026-09(최근 우선).
  지표: IC · 비겹침(지평 간격) 매매 = 예측 부호 × |예측| 상위 q(학습 예측 분위로 고정) · 비용 테이커 8bp/메이커 1.4bp · 일 블록 CI · 월별 부호.
  기준선: 일봉 추세 5표(TSMOM, 09-27 통과 부품) · 항상 롱.
플래그: fresh_forward_bar_by_bar=true · trade_ledgers_used_as_input=false · future_rows_used_for_entry=false.
"""
import glob, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

BV = Path("/home/kbj20/crypto-scalping/data/binance_vision")
OUT = "tmp/best_dir_20261010"
HS = {"15m": 3, "1h": 12, "4h": 48, "1d": 288}
SWITCH = pd.Timestamp("2024-03-04")
COINS = ["ADA", "APT", "ARB", "ATOM", "AVAX", "BNB", "BTC", "DOGE", "DOT", "FIL", "INJ", "LINK", "LTC", "NEAR", "OP", "SOL", "SUI", "TRX", "XRP"]


def lag(x, k):
    x = np.asarray(x, dtype=float)
    return np.r_[np.full(k, np.nan), x[:-k]] if k > 0 else x


def rsum(x, k):
    c = np.r_[0.0, np.nancumsum(np.nan_to_num(np.asarray(x, dtype=float)))]
    o = np.full(len(x), np.nan); o[k - 1:] = c[k:] - c[:-k]
    return o


def rz(x, w=2016, mp=288):
    """과거 w 봉(현재 제외) 대비 견고 z."""
    s = pd.Series(x).shift(1)
    med, q1, q3 = s.rolling(w, min_periods=mp).median(), s.rolling(w, min_periods=mp).quantile(.25), s.rolling(w, min_periods=mp).quantile(.75)
    return ((pd.Series(x) - med) / ((q3 - q1) / 1.349).replace(0, np.nan)).to_numpy()


def metrics_known(p, col):
    """metrics 열을 T 에 아는 값으로: 2024-03-04 이후는 한 행 늦춤."""
    v = p[col].to_numpy(dtype=float)
    return np.where((p.timestamp >= SWITCH).to_numpy(), lag(v, 1), v)


def load_panel(sym):
    p = pd.read_parquet(BV / f"panel/{sym}USDT.parquet").sort_values("timestamp").drop_duplicates("timestamp")
    full = pd.DataFrame({"timestamp": pd.date_range(p.timestamp.min(), p.timestamp.max(), freq="5min")})
    return full.merge(p, on="timestamp", how="left")


def build(sym="ETH"):
    os.makedirs(OUT, exist_ok=True)
    p = load_panel(sym)
    ts = p.timestamp; c = p.close.ffill().to_numpy(); lc = np.log(c)
    F = pd.DataFrame({"ts": ts})
    for nm, h in HS.items():
        F[f"y_{nm}"] = np.r_[(lc[h:] - lc[:-h]) * 1e4, np.full(h, np.nan)]
    # ── 가격·기술 ──
    r1 = np.r_[np.nan, np.diff(lc)] * 1e4
    for k, nm in ((1, "5m"), (3, "15m"), (12, "1h"), (48, "4h"), (288, "1d"), (2016, "7d"), (8064, "28d")):
        F[f"P_r{nm}"] = (lc - lag(lc, k)) * 1e4
    F["P_rv1h"] = pd.Series(r1).rolling(12).std().to_numpy(); F["P_rv1d"] = pd.Series(r1).rolling(288).std().to_numpy()
    F["P_vratio"] = F.P_rv1h / F.P_rv1d
    d = pd.Series(c).diff()
    F["P_rsi14"] = (100 - 100 / (1 + d.clip(lower=0).ewm(alpha=1 / 14).mean() / (-d.clip(upper=0)).ewm(alpha=1 / 14).mean())).to_numpy()
    F["P_pctb"] = ((pd.Series(c) - pd.Series(c).rolling(20).mean()) / (2 * pd.Series(c).rolling(20).std())).to_numpy()
    hi, lo = p.high.to_numpy(), p.low.to_numpy()
    h288, l288 = pd.Series(hi).rolling(288).max().to_numpy(), pd.Series(lo).rolling(288).min().to_numpy()
    F["P_pos1d"] = (c - l288) / (h288 - l288)
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - lag(c, 1)), np.abs(lo - lag(c, 1))))
    F["P_veto_z"] = (c - pd.Series(c).rolling(144).mean().to_numpy()) / pd.Series(tr).ewm(alpha=1 / 144).mean().to_numpy()
    F["P_ac144"] = pd.Series(r1).rolling(144).apply(lambda v: np.corrcoef(v[:-1], v[1:])[0, 1], raw=True).to_numpy()
    day = ts.dt.floor("D")
    dclose = pd.Series(c, index=ts).groupby(day.to_numpy()).last()               # 그날 종가 → 다음날 00:00 에 앎
    dk = np.searchsorted(dclose.index.to_numpy() + np.timedelta64(1, "D"), (ts + pd.Timedelta("5min")).to_numpy(), "right") - 1
    dv = dclose.to_numpy()
    votes = [np.where(dk - L >= 0, np.sign(dv[np.maximum(dk, 0)] - dv[np.maximum(dk - L, 0)]), np.nan) for L in (7, 14, 28, 56, 90)]
    F["P_trend5"] = np.mean(votes, axis=0)
    # ── 체결 흐름 ──
    v, tb = p.volume.to_numpy(), p.taker_buy_base.to_numpy()
    dl = 2 * tb - v
    for k, nm in ((1, "5m"), (3, "15m"), (12, "1h"), (48, "4h")):
        F[f"T_imb{nm}"] = rsum(dl, k) / rsum(v, k)
    F["T_cvd1h_z"] = rsum(dl, 12) / pd.Series(rsum(dl, 12)).shift(1).rolling(2016, min_periods=288).std().to_numpy()
    F["T_vol_z"] = rz(np.log(rsum(v, 12)))
    F["T_trades_z"] = rz(np.log(rsum(p.trades.to_numpy(dtype=float), 12)))
    F["T_size"] = np.log(rsum(v, 12) / rsum(p.trades.to_numpy(dtype=float), 12))
    # ── OI·롱숏(metrics, 스탬프 규약) ──
    oi = metrics_known(p, "sum_open_interest")
    for k, nm in ((1, "5m"), (12, "1h"), (48, "4h"), (288, "1d")):
        F[f"O_doi{nm}"] = (np.log(oi) - np.log(lag(oi, k))) * 1e4
    F["O_doi1h_z"] = rz(F.O_doi1h.to_numpy())
    F["O_doi1h_x_r"] = F.O_doi1h * np.sign(F.P_r1h)
    for col, nm in (("sum_toptrader_long_short_ratio", "toppos"), ("count_toptrader_long_short_ratio", "topacct"),
                    ("count_long_short_ratio", "glob"), ("sum_taker_long_short_vol_ratio", "takerls")):
        x = np.log(metrics_known(p, col))
        F[f"O_{nm}"] = x; F[f"O_{nm}_d1h"] = x - lag(x, 12); F[f"O_{nm}_z"] = rz(x, 8064, 2016)
    F["O_top_minus_glob"] = F.O_toppos - F.O_glob                                   # 고래(상위) vs 군중 쏠림 차
    # ── 베이시스·현물 (ETH 만 원천 있음) ──
    if sym == "ETH":
        pr = pd.read_parquet(BV / "eth_premium_5m.parquet"); sp = pd.read_parquet(BV / "eth_spot_5m.parquet")
        key = (ts.astype("int64") // 10**6).to_numpy()
        prm = pd.Series(pr.close.to_numpy() * 1e4, index=pr.open_time.astype("int64")).reindex(key).to_numpy()
        F["B_prem"] = prm; F["B_prem_d1h"] = prm - lag(prm, 12); F["B_prem_8h"] = pd.Series(prm).rolling(96, min_periods=48).mean().to_numpy()
        F["B_prem_z"] = rz(prm, 8064, 2016)
        sc = pd.Series(sp.close.to_numpy(), index=sp.open_time.astype("int64")).reindex(key).ffill().to_numpy()
        sv = pd.Series(sp.volume.to_numpy(), index=sp.open_time.astype("int64")).reindex(key).to_numpy()
        stb = pd.Series(sp.tb.to_numpy(), index=sp.open_time.astype("int64")).reindex(key).to_numpy()
        lsc = np.log(sc)
        for k, nm in ((1, "5m"), (3, "15m"), (12, "1h")):
            F[f"S_lead{nm}"] = ((lsc - lag(lsc, k)) - (lc - lag(lc, k))) * 1e4                   # 현물이 선물보다 더 간 몫
            F[f"S_imb{nm}"] = rsum(2 * stb - sv, k) / rsum(sv, k)
        F["S_imb_minus_perp1h"] = F.S_imb1h - F.T_imb1h
        F["S_vol_share"] = rsum(sv, 12) / (rsum(sv, 12) + rsum(v, 12))
    # ── 마이크로구조 aggfeat (2024-01~) ──
    A = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(BV / f"aggfeat/{sym}USDT_*.parquet")))]).sort_index()
    A = A[~A.index.duplicated()].reindex(ts.to_numpy())
    for col in A.columns:
        F[f"M_{col[3:]}"] = A[col].to_numpy()
    F["M_meansz"] = np.log(F.M_meansz)
    for col in ("bigimb", "smlimb", "cntimb", "kyle", "vpin", "signac"):
        F[f"M_{col}_1h"] = pd.Series(F[f"M_{col}"]).rolling(12, min_periods=6).mean().to_numpy()
    F["M_whale_minus_retail_1h"] = F.M_bigimb_1h - F.M_smlimb_1h
    trip = (np.corrcoef(np.nan_to_num(F.M_cvd), np.nan_to_num(r1))[0, 1], np.corrcoef(np.nan_to_num(F.M_cvd[:-1]), np.nan_to_num(r1[1:]))[0, 1])
    print("트립와이어 aggfeat cvd ↔ 같은 봉 / 다음 봉 수익:", np.round(trip, 3))
    assert trip[0] > 0.2 and abs(trip[1]) < 0.1, "aggfeat 시각 규약이 봉 시작이 아님 — 중단"
    # ── 호가 깊이(bookDepth, 2024-01~) ──
    Bk = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(BV / f"{sym}USDT_bookDepth_*.parquet")))]).sort_index()
    Bk = Bk[~Bk.index.duplicated()]
    T_ = (ts + pd.Timedelta("5min")).to_numpy()
    bi = np.searchsorted(Bk.index.to_numpy(), T_, "right") - 1
    fresh = (bi >= 0) & (T_ - Bk.index.to_numpy()[np.maximum(bi, 0)] <= np.timedelta64(5, "m"))
    for k in (1, 2, 5):
        bid, ask = Bk[f"n-{k}"].to_numpy()[np.maximum(bi, 0)], Bk[f"n{k}"].to_numpy()[np.maximum(bi, 0)]
        F[f"D_imb{k}"] = np.where(fresh, (bid - ask) / (bid + ask), np.nan)
    F["D_imb1_d1h"] = F.D_imb1 - lag(F.D_imb1.to_numpy(), 12)
    F["D_depth1_z"] = rz(np.log(np.where(fresh, Bk["n-1"].to_numpy()[np.maximum(bi, 0)] + Bk["n1"].to_numpy()[np.maximum(bi, 0)], np.nan)))
    # ── 교차 자산(BTC·20코인 폭) ──
    rets12, rets288 = [], []
    ref = "BTC" if sym != "BTC" else "ETH"                                          # 교차 기준 자산(BTC 대상이면 ETH)
    for other in [c if c != sym else "ETH" for c in COINS]:
        q = load_panel(other).set_index("timestamp").reindex(ts)
        lq = np.log(q.close.ffill().to_numpy())
        rets12.append((lq - lag(lq, 12)) * 1e4); rets288.append((lq - lag(lq, 288)) * 1e4)
        if other == ref:
            for k, nm in ((1, "5m"), (3, "15m"), (12, "1h"), (48, "4h")):
                F[f"X_btc_r{nm}"] = (lq - lag(lq, k)) * 1e4
            F["X_btc_imb1h"] = rsum(2 * q.taker_buy_base.to_numpy() - q.volume.to_numpy(), 12) / rsum(q.volume.to_numpy(), 12)
            boi = metrics_known(q.reset_index().rename(columns={"index": "timestamp"}), "sum_open_interest")
            F["X_btc_doi1h"] = (np.log(boi) - np.log(lag(boi, 12))) * 1e4
    R12, R288 = np.vstack(rets12), np.vstack(rets288)
    F["X_breadth1h"] = np.nanmean(R12, 0); F["X_breadth1d"] = np.nanmean(R288, 0)
    F["X_eth_rel1h"] = F.P_r1h - F.X_breadth1h; F["X_eth_rel1d"] = F.P_r1d - F.X_breadth1d
    F["X_eth_rank1d"] = (R288 < F.P_r1d.to_numpy()).mean(0)
    F["X_disp1h"] = np.nanstd(R12, 0)
    # ── 시간 ──
    tt = ts + pd.Timedelta("5min")
    hr = tt.dt.hour + tt.dt.minute / 60
    F["C_hsin"], F["C_hcos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    F["C_dow"] = tt.dt.dayofweek; F["C_to_funding"] = (8 - hr % 8) % 8
    F["C_weekend"] = (tt.dt.dayofweek >= 5).astype(float)
    F.replace([np.inf, -np.inf], np.nan, inplace=True)
    F.to_parquet(panel_path(sym))
    fc = feat_cols(F)
    print(len(F), "rows ·", len(fc), "features ·", F.ts.min(), "~", F.ts.max())
    cov = F[fc].notna().mean().round(2); print("커버리지<0.9:", cov[cov < .9].to_dict())


def panel_path(sym="ETH"):
    return f"{OUT}/panel.parquet" if sym == "ETH" else f"{OUT}/panel_{sym}.parquet"


def feat_cols(F):
    return [x for x in F.columns if len(x) > 2 and x[1] == "_" and x[0] in "PTOBSMDXCU"]


def ic(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank())) if m.sum() > 100 else np.nan


def day_ci(day, v, n=500, seed=0):
    rng = np.random.default_rng(seed)
    s = pd.Series(v).groupby(day).sum(); cnt = pd.Series(np.ones(len(v))).groupby(day).sum()
    S, C = s.to_numpy(), cnt.to_numpy()
    b = [S[ix].sum() / C[ix].sum() for ix in (rng.integers(0, len(S), len(S)) for _ in range(n))]
    return [round(float(x), 2) for x in np.quantile(b, [.025, .975])]


def trade_eval(pred, y, ts, h, thr_lo, thr_hi, cost):
    """비겹침(h 봉 간격) 매매: pred ≥ thr_hi 롱 · ≤ thr_lo 숏. 반환 n·평균 bp·일 블록 CI·연 샤프·월 양수 비."""
    idx = np.arange(0, len(y), h)
    p, yy, tt = pred[idx], y[idx], ts[idx]
    lo_, hi_ = (thr_lo[idx], thr_hi[idx]) if np.ndim(thr_lo) else (thr_lo, thr_hi)
    s = np.where(p >= hi_, 1, np.where(p <= lo_, -1, 0))
    m = (s != 0) & np.isfinite(yy) & np.isfinite(p)
    g = s[m] * yy[m] - cost
    if m.sum() < 30:
        return {"n": int(m.sum())}
    dday = pd.to_datetime(tt[m]).floor("D").to_numpy()
    mon = pd.Series(g).groupby(pd.to_datetime(tt[m]).to_period("M")).sum()
    per_year = (365 * 288 / h) * m.sum() / max(np.isfinite(yy).sum(), 1)
    return {"n": int(m.sum()), "bp": round(float(g.mean()), 2), "ci": day_ci(dday, g), "sharpe": round(float(g.mean() / (g.std() + 1e-9) * np.sqrt(per_year)), 2),
            "month_pos": f"{int((mon > 0).sum())}/{len(mon)}"}


def walk_forward(F, cols, nm, h, q=0.2, start_train=None, params=None, years=(2024, 2025, 2026)):
    from sklearn.ensemble import HistGradientBoostingRegressor
    y = F[f"y_{nm}"].to_numpy(); ts = F.ts.to_numpy(); yr = F.ts.dt.year.to_numpy()
    pred = np.full(len(F), np.nan); lo = np.full(len(F), np.nan); hi = np.full(len(F), np.nan)
    for Y in years:
        cut = np.datetime64(f"{Y}-01-01")
        tr = (ts < cut - np.timedelta64(5 * h, "m")) & np.isfinite(y)
        if start_train is not None:
            tr &= ts >= np.datetime64(start_train)
        if tr.sum() < 20000:
            continue
        te = yr == Y
        kw = dict(max_iter=300, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=1000, l2_regularization=1.0, random_state=0)
        kw.update(params or {})
        m = HistGradientBoostingRegressor(**kw).fit(F.loc[tr, cols].to_numpy()[::2], y[tr][::2])
        pred[te] = m.predict(F.loc[te, cols].to_numpy())
        q_ = np.quantile(m.predict(F.loc[tr, cols].to_numpy()[::7]), [q, 1 - q])
        lo[te], hi[te] = q_
    return pred, lo, hi


def evaluate(F, pred, lo, hi, nm, h, label):
    y = F[f"y_{nm}"].to_numpy(); ts = F.ts.to_numpy(); yr = F.ts.dt.year.to_numpy()
    out = {"label": label, "h": nm}
    main = (ts >= np.datetime64("2025-01-01")) & np.isfinite(pred)
    out["ic_main"] = round(ic(pred[main], y[main]), 4)
    out["ic_by_year"] = {int(Y): round(ic(pred[yr == Y], y[yr == Y]), 4) for Y in (2024, 2025, 2026)}
    for cn, cost in (("taker", 8.0), ("maker", 1.4)):
        out[f"{cn}_main"] = trade_eval(np.where(main, pred, np.nan), np.where(main, y, np.nan), ts, h, lo, hi, cost)
    out["maker_2024"] = trade_eval(np.where(yr == 2024, pred, np.nan), np.where(yr == 2024, y, np.nan), ts, h, lo, hi, 1.4)
    return out


def wf():
    F = pd.read_parquet(f"{OUT}/panel.parquet")
    FC = feat_cols(F)
    res = []
    for nm, h in HS.items():
        tr5 = F.P_trend5.to_numpy(); n = len(F)
        res.append(evaluate(F, tr5, np.full(n, -0.5), np.full(n, 0.5), nm, h, "기준선 일봉추세(|표|≥.5)"))
        res.append(evaluate(F, np.ones(n), np.full(n, -1.0), np.full(n, 0.5), nm, h, "기준선 항상 롱"))
        for label, cols, st in (("전체 HGB(2022~학습)", FC, None),
                                ("전체 HGB(2024~학습)", FC, "2024-01-01"),
                                ("마이크로·깊이 제외 HGB", [c for c in FC if c[0] not in "MD"], None)):
            pred, lo, hi = walk_forward(F, cols, nm, h, start_train=st)
            res.append(evaluate(F, pred, lo, hi, nm, h, label))
        for r in res[-5:]:
            print(json.dumps(r, ensure_ascii=False), flush=True)
    json.dump(res, open(f"{OUT}/wf_report.json", "w"), ensure_ascii=False, indent=1, default=str)


def selftest():
    assert np.isnan(lag([1., 2.], 1)[0]) and lag([1., 2.], 1)[1] == 1.0
    assert np.allclose(rsum(np.array([1., 2, 3]), 2)[1:], [3, 5])
    y = np.tile([10., -10], 30); pred = np.tile([1., -1], 30)
    ts = pd.date_range("2025-01-01", periods=60, freq="5min").to_numpy()
    r = trade_eval(pred, y, ts, 1, -.5, .5, 1.0)
    assert r["bp"] == 9.0 and r["n"] == 60                                          # 매번 맞히고 비용 1bp
    print("selftest OK")


if __name__ == "__main__":
    {"--build": build, "--wf": wf, "--selftest": selftest}.get(sys.argv[1], lambda: None)()


def robust4h():
    """사후 강건성(4h 후보): 원인 분리(학습 창 vs 마이크로·깊이) · 시드 3 · 위상 8 · 진입 1봉 지연 · 순열 중요도."""
    F = pd.read_parquet(f"{OUT}/panel.parquet"); FC = feat_cols(F)
    nm, h = "4h", 48
    y = F[f"y_{nm}"].to_numpy(); ts = F.ts.to_numpy(); main = ts >= np.datetime64("2025-01-01")
    y_d1 = np.r_[y[1:], np.nan]                                                      # 한 봉 늦게 진입(봉 k+1 종가)
    variants = {"A 전체·2024~": (FC, "2024-01-01"), "B 마이크로·깊이 제외·2024~": ([c for c in FC if c[0] not in "MD"], "2024-01-01"),
                "C 전체·2023~": (FC, "2023-01-01"), "D 마이크로·깊이만+가격": ([c for c in FC if c[0] in "MDP"], "2024-01-01")}
    out = {}
    for vn, (cols, st) in variants.items():
        rows = []
        for seed in (0, 1, 2):
            pred, lo, hi = walk_forward(F, cols, nm, h, start_train=st, params={"random_state": seed}, years=(2025, 2026))
            ph = []
            for off in range(0, 48, 6):
                sl = slice(off, None)
                r = trade_eval(np.where(main, pred, np.nan)[sl], np.where(main, y, np.nan)[sl], ts[sl], h, lo[sl], hi[sl], 1.4)
                ph.append(r.get("bp", np.nan))
            r_d1 = trade_eval(np.where(main, pred, np.nan), np.where(main, y_d1, np.nan), ts, h, lo, hi, 1.4)
            rows.append({"seed": seed, "ic": round(ic(pred[main], y[main]), 4), "maker_bp_phases": [round(x, 1) for x in ph],
                         "maker_bp_mean": round(float(np.nanmean(ph)), 2), "maker_delay1": r_d1.get("bp"), "taker_bp_mean": round(float(np.nanmean(ph)) - 6.6, 2)})
        out[vn] = rows
        print(vn, json.dumps(rows, ensure_ascii=False), flush=True)
    # 순열 중요도(A, 시드 0, 2025~ IC 하락)
    from sklearn.ensemble import HistGradientBoostingRegressor
    tr = (ts >= np.datetime64("2024-01-01")) & (ts < np.datetime64("2025-01-01") - np.timedelta64(240, "m")) & np.isfinite(y)
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=1000, l2_regularization=1.0,
                                      random_state=0).fit(F.loc[tr, FC].to_numpy()[::2], y[tr][::2])
    te = main & np.isfinite(y)
    X = F.loc[te, FC].to_numpy()[::4]; yt = y[te][::4]
    base = ic(m.predict(X), yt); rng = np.random.default_rng(0); imp = {}
    for j, cname in enumerate(FC):
        Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
        imp[cname] = round(base - ic(m.predict(Xp), yt), 4)
    top = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:15])
    fam = pd.Series(imp).groupby(lambda c: c[0]).sum().round(4).to_dict()
    out["perm_importance_top15"] = top; out["perm_by_family"] = fam; out["perm_base_ic_2025_train2024only"] = round(base, 4)
    print("순열 중요도 상위:", top, "\n묶음 합:", fam, flush=True)
    json.dump(out, open(f"{OUT}/robust4h.json", "w"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--robust4h":
    robust4h()


def wf_monthly(F, cols, nm, h, window_days, seeds=(0, 1), q=0.2, start="2025-01-01"):
    """매달 재학습 · 학습 창 = 직전 window_days(정화 간격 = 지평) · 시드 평균. 임계 = 학습 예측 분위."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    y = F[f"y_{nm}"].to_numpy(); ts = F.ts.to_numpy()
    pred = np.full(len(F), np.nan); lo = np.full(len(F), np.nan); hi = np.full(len(F), np.nan)
    months = pd.date_range(start, pd.Timestamp(ts.max()).normalize(), freq="MS")
    X = F[cols].to_numpy(dtype=np.float32)
    for m0 in months:
        m1 = m0 + pd.offsets.MonthBegin(1)
        cut = np.datetime64(m0) - np.timedelta64(5 * h, "m")
        tr = (ts < cut) & (ts >= cut - np.timedelta64(window_days, "D")) & np.isfinite(y)
        te = (ts >= np.datetime64(m0)) & (ts < np.datetime64(m1))
        if tr.sum() < 20000 or te.sum() == 0:
            continue
        ps, pt = [], []
        for s in seeds:
            mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=1000, l2_regularization=1.0,
                                                random_state=s).fit(X[tr][::2], y[tr][::2])
            ps.append(mdl.predict(X[te])); pt.append(mdl.predict(X[tr][::7]))
        pred[te] = np.mean(ps, 0); q_ = np.quantile(np.mean(pt, 0), [q, 1 - q]); lo[te], hi[te] = q_
        print("  월", m0.strftime("%Y-%m"), "학습", int(tr.sum()), flush=True)
    return pred, lo, hi


def phase_eval(F, pred, lo, hi, nm, h, costs=(1.4, 4.0, 8.0)):
    y = F[f"y_{nm}"].to_numpy(); ts = F.ts.to_numpy(); main = (ts >= np.datetime64("2025-01-01")) & np.isfinite(pred)
    out = {"ic": round(ic(pred[main], y[main]), 4), "ic_2025": round(ic(pred[main & (ts < np.datetime64("2026-01-01"))], y[main & (ts < np.datetime64("2026-01-01"))]), 4),
           "ic_2026": round(ic(pred[ts >= np.datetime64("2026-01-01")], y[ts >= np.datetime64("2026-01-01")]), 4)}
    for cost in costs:
        bps = []
        for off in range(0, h, max(h // 8, 1)):
            sl = slice(off, None)
            r = trade_eval(np.where(main, pred, np.nan)[sl], np.where(main, y, np.nan)[sl], ts[sl], h, lo[sl], hi[sl], cost)
            bps.append(r.get("bp", np.nan))
            if off == 0:
                out[f"c{cost}_phase0"] = r
        out[f"c{cost}_phase_mean"] = round(float(np.nanmean(bps)), 2); out[f"c{cost}_phase_min"] = round(float(np.nanmin(bps)), 2)
    return out


def monthly_grid():
    F = pd.read_parquet(f"{OUT}/panel.parquet"); FC = feat_cols(F)
    res = {}
    # 05:52 축소(다른 세션 GA 작업과 CPU 공유): 4h · 핵심 3변형 · 시드 1
    for vn, cols, win in (("전체+개장·365일", FC, 365), ("달력·개장 제외·365일", [c for c in FC if c[0] not in "CU"], 365),
                          ("전체+개장·730일", FC, 730)):
        for nm, h in (("4h", 48),):
            pred, lo, hi = wf_monthly(F, cols, nm, h, win, seeds=(0,))
            r = phase_eval(F, pred, lo, hi, nm, h)
            res[f"{vn}|{nm}"] = r
            print(vn, nm, json.dumps({k: v for k, v in r.items() if "phase0" not in k}, ensure_ascii=False), flush=True)
            np.save(f"{OUT}/pred_{vn}_{nm}.npy", np.c_[pred, lo, hi])
    json.dump(res, open(f"{OUT}/monthly_grid.json", "w"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--monthly":
    monthly_grid()


"""UTC 00:00 개장 되돌림(사전등록 05:35 KST, 사후 탐색에서 출발 — 24시각 중 00시 최강·20코인 20/20 음수 확인 뒤 고정).
결정 = 매일 00:30 UTC(00:25 봉 마감). 피쳐(그 시각까지): f30 자기 첫 30분 · f30_mkt 20코인 평균 · f30_btc · imb30 테이커 불균형 ·
vol30_z 첫 30분 거래량 ÷ 직전 20일 같은 창 중앙(로그) · doi30 OI 변화(스탬프 규약) · r1d 전날 24h · rv1d 전날 5분 변동.
목표 n1·n4·n8 = 00:30 → 01:30·04:30·08:30 로그수익 bp. 학습 = 20코인 풀링, 시험 해 이전 전부. 시험 2024·2025·2026.
규칙 R = |f30| 학습 상위 1/3 일 때 반대로. 모델 = 로지스틱 대신 리지 회귀(소수 피쳐) · HGB. 주 판정 = ETH 2025~ n4, 비용 4bp(지정가 진입+시장가 청산 근사),
일 단위 CI 하한 > 0 · 2024 같은 부호 · 다른 19코인 2025~ 평균 같은 부호."""
OPEN_FEATS = ["f30", "f30_mkt", "f30_btc", "imb30", "vol30_z", "doi30", "r1d", "rv1d"]


def open_panel():
    syms = ["ETH", "BTC", "SOL", "XRP", "BNB", "DOGE", "ADA", "LINK", "AVAX", "LTC", "DOT", "TRX", "ATOM", "FIL", "NEAR", "APT", "ARB", "OP", "INJ", "SUI"]
    days = pd.date_range("2022-01-03", "2026-09-12", freq="D")
    rows = []
    for sym in syms:
        p = load_panel(sym).set_index("timestamp")
        c = np.log(p.close.ffill()); v = p.volume.fillna(0); dl = (2 * p.taker_buy_base - p.volume).fillna(0)
        oi = pd.Series(metrics_known(p.reset_index(), "sum_open_interest"), index=p.index)
        g = lambda s, dt: s.reindex(days + dt).to_numpy()
        cs_v, cs_d = v.cumsum(), dl.cumsum()
        f30 = (g(c, pd.Timedelta("25min")) - g(c, pd.Timedelta("-5min"))) * 1e4
        v30 = g(cs_v, pd.Timedelta("25min")) - g(cs_v, pd.Timedelta("-5min"))
        d30 = g(cs_d, pd.Timedelta("25min")) - g(cs_d, pd.Timedelta("-5min"))
        r5 = c.diff() * 1e4
        rv = r5.rolling(288).std()
        D = pd.DataFrame({"day": days, "sym": sym, "f30": f30, "imb30": d30 / v30, "lv30": np.log(v30),
                          "doi30": (np.log(g(oi, pd.Timedelta("25min"))) - np.log(g(oi, pd.Timedelta("-5min")))) * 1e4,
                          "r1d": (g(c, pd.Timedelta("-5min")) - g(c, pd.Timedelta("-1445min"))) * 1e4, "rv1d": g(rv, pd.Timedelta("-5min")),
                          **{f"n{H}": (g(c, pd.Timedelta(hours=H, minutes=25)) - g(c, pd.Timedelta("25min"))) * 1e4 for H in (1, 4, 8)}})
        D["vol30_z"] = D.lv30 - D.lv30.shift(1).rolling(20, min_periods=10).median()
        rows.append(D)
    P = pd.concat(rows, ignore_index=True)
    mk = P.groupby("day").f30.mean(); bt = P[P.sym == "BTC"].set_index("day").f30
    P["f30_mkt"] = P.day.map(mk); P["f30_btc"] = P.day.map(bt)
    P["yr"] = P.day.dt.year
    return P.replace([np.inf, -np.inf], np.nan)


def open_reversal():
    from sklearn.ensemble import HistGradientBoostingRegressor
    P = open_panel(); P.to_parquet(f"{OUT}/open_panel.parquet")
    out = {}
    for tgt in ("n1", "n4", "n8"):
        res = {}
        for Y in (2024, 2025, 2026):
            tr = (P.yr < Y) & P[tgt].notna() & P.f30.notna(); te = (P.yr == Y) & P[tgt].notna() & P.f30.notna()
            thr = P.loc[tr, "f30"].abs().quantile(2 / 3)
            Xtr, ytr = P.loc[tr, OPEN_FEATS].to_numpy(), P.loc[tr, tgt].to_numpy()
            mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0) + 1e-9
            Z = np.clip(np.nan_to_num((Xtr - mu) / sd), -4, 4)
            b = np.linalg.solve(Z.T @ Z + 50 * np.eye(Z.shape[1]), Z.T @ (ytr - ytr.mean()))
            Zte = np.clip(np.nan_to_num((P.loc[te, OPEN_FEATS].to_numpy() - mu) / sd), -4, 4)
            P.loc[te, "p_ridge"] = Zte @ b
            ptr_r = Z @ b
            hg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=300, l2_regularization=2.0,
                                               random_state=0).fit(Xtr, ytr)
            P.loc[te, "p_hgb"] = hg.predict(P.loc[te, OPEN_FEATS].to_numpy()); ptr_h = hg.predict(Xtr)
            P.loc[te, "s_rule"] = np.where(P.loc[te, "f30"].abs() >= thr, -np.sign(P.loc[te, "f30"]), 0)
            for nm, pt, ptrn in (("ridge", "p_ridge", ptr_r), ("hgb", "p_hgb", ptr_h)):
                lo, hi = np.quantile(ptrn, [1 / 3, 2 / 3])
                P.loc[te, f"s_{nm}"] = np.where(P.loc[te, pt] >= hi, 1, np.where(P.loc[te, pt] <= lo, -1, 0))
            res[f"ridge_coef_{Y}"] = dict(zip(OPEN_FEATS, np.round(b, 2)))
        for nm in ("rule", "ridge", "hgb"):
            for sym_set, cond in (("ETH", P.sym == "ETH"), ("나머지19", P.sym != "ETH")):
                for Y in (2024, 2025, 2026):
                    m = cond & (P.yr == Y) & (P[f"s_{nm}"] != 0) & P[tgt].notna()
                    g = (P.loc[m, f"s_{nm}"] * P.loc[m, tgt]).to_numpy()
                    if len(g) < 20:
                        continue
                    dd = pd.Series(g).groupby(P.loc[m, "day"].to_numpy()).mean()
                    bs = np.quantile([dd.sample(len(dd), replace=True, random_state=i).mean() for i in range(400)], [.025, .975])
                    res[f"{nm}|{sym_set}|{Y}"] = {"n": int(len(g)), "gross_bp": round(float(g.mean()), 1), "ci": [round(float(x), 1) for x in bs],
                                                  "net4": round(float(g.mean() - 4), 1), "hit": round(float((g > 0).mean()), 3)}
        out[tgt] = res
        print(tgt, json.dumps({k: v for k, v in res.items() if "coef" not in k}, ensure_ascii=False), "\n계수", res.get("ridge_coef_2026"), flush=True)
    json.dump(out, open(f"{OUT}/open_reversal.json", "w"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--open":
    open_reversal()


def add_open_feats(sym="ETH"):
    """UTC 개장 되돌림을 일반 모델 피쳐로: 개장 후 누적 수익 · 그날 첫 30분(00:30 전엔 NaN) — ETH·BTC. 결정 T = 봉 k 마감."""
    F = pd.read_parquet(panel_path(sym))
    T = F.ts + pd.Timedelta("5min"); d0 = T.dt.floor("D")
    for ref_sym, pref in (("ETH", "U_eth"), ("BTC", "U_btc")):                      # 🔴 sym 을 덮어쓰지 말 것(07:16 SOL 이 panel_BTC 에 저장된 버그)
        s = load_panel(ref_sym).set_index("timestamp").close.ffill()
        px = lambda t: s.reindex((t - pd.Timedelta("5min")).to_numpy()).to_numpy()    # 시각 t 의 가격 = t−5분 봉 종가
        now, o, o30 = px(T), px(d0), px(d0 + pd.Timedelta("30min"))
        F[f"{pref}_since_open"] = np.log(now / o) * 1e4
        F[f"{pref}_f30_day"] = np.where((T - d0) >= pd.Timedelta("30min"), np.log(o30 / o) * 1e4, np.nan)
    F["U_min_since_open"] = ((T - d0).dt.total_seconds() / 60).to_numpy()
    F.to_parquet(panel_path(sym))
    print("열 추가:", [c for c in F.columns if c.startswith("U_")], F[[c for c in F.columns if c.startswith("U_")]].describe().T[["count", "mean"]].round(2).to_dict())


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--openfeats":
    add_open_feats()


def final_checks(vn="전체+개장·365일", nm="4h", h=48, n_null=2):
    """최고 후보 검증: 진입 1봉 지연 · q 민감도 · 월별/연도별·최대 낙폭 · 블록 섞은 가짜 정답 학습(같은 절차) 대비."""
    F = pd.read_parquet(f"{OUT}/panel.parquet"); FC = feat_cols(F)
    y = F[f"y_{nm}"].to_numpy(); ts = F.ts.to_numpy(); main = (ts >= np.datetime64("2025-01-01"))
    P = np.load(f"{OUT}/pred_{vn}_{nm}.npy"); pred, lo, hi = P[:, 0], P[:, 1], P[:, 2]
    out = {}
    y_d1 = np.r_[y[1:], np.nan]
    for lab, yy in (("즉시", y), ("1봉 지연", y_d1)):
        out[lab] = {c: trade_eval(np.where(main, pred, np.nan), np.where(main, yy, np.nan), ts, h, lo, hi, c) for c in (1.4, 4.0, 8.0)}
    for q in (0.1, 0.3, 0.4):                                                         # 임계 = 같은 시험 달의 예측 분포 대신, 학습 분위 근사: 전 달까지 예측 분위
        s = pd.Series(pred); lo_q = s.shift(1).expanding(8640).quantile(q).to_numpy(); hi_q = s.shift(1).expanding(8640).quantile(1 - q).to_numpy()
        out[f"q{q}"] = trade_eval(np.where(main, pred, np.nan), np.where(main, y, np.nan), ts, h, lo_q, hi_q, 4.0)
    idx = np.arange(0, len(y), h); p_, y_, t_ = pred[idx], y[idx], ts[idx]
    s = np.where(p_ >= hi[idx], 1, np.where(p_ <= lo[idx], -1, 0)); m = main[idx] & (s != 0) & np.isfinite(y_)
    g = pd.Series(s[m] * y_[m] - 4.0, index=pd.to_datetime(t_[m]))
    eq = g.cumsum(); dd = (eq - eq.cummax()).min()
    out["월별_bp합(4bp)"] = {str(k): int(v) for k, v in g.groupby(g.index.to_period("M")).sum().round(0).items()}
    out["최대낙폭_bp(4bp, 단위 노출)"] = round(float(dd), 0); out["누적_bp"] = round(float(eq.iloc[-1]), 0)
    out["롱비율"] = round(float((s[m] > 0).mean()), 3)
    # 블록 섞은 가짜 정답: 하루 블록 단위로 y 를 섞어 같은 월별 절차로 학습·평가
    nulls = []
    cols = FC
    for k in range(n_null):
        rng = np.random.default_rng(100 + k)
        dayid = (ts.astype("datetime64[D]")).astype(np.int64); ud = np.unique(dayid); perm = dict(zip(ud, rng.permutation(ud)))
        yf = y.copy()
        for d in ud:
            a, b = np.searchsorted(dayid, d), np.searchsorted(dayid, d, "right"); a2 = np.searchsorted(dayid, perm[d]); n = min(b - a, np.searchsorted(dayid, perm[d], "right") - a2)
            yf[a:a + n] = y[a2:a2 + n]
        G = F.copy(); G[f"y_{nm}"] = yf
        pn, ln, hn = wf_monthly(G, cols, nm, h, 365, seeds=(0,))
        r = trade_eval(np.where(main, pn, np.nan), np.where(main, y, np.nan), ts, h, ln, hn, 4.0)
        nulls.append({"ic_vs_real": round(ic(pn[main], y[main]), 4), "bp4_vs_real": r.get("bp")})
        print("가짜정답", k, nulls[-1], flush=True)
    out["null_models"] = nulls
    json.dump(out, open(f"{OUT}/final_checks.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "월별_bp합(4bp)"}, ensure_ascii=False, default=str), "\n월별", out["월별_bp합(4bp)"], flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--final":
    final_checks(n_null=int(sys.argv[2]) if len(sys.argv) > 2 else 2)


def window_grid():
    """학습 창 길이 민감도(180·270일) + 창 앙상블(180·270·365 평균) — 사후, 06:50 추가."""
    F = pd.read_parquet(f"{OUT}/panel.parquet"); FC = feat_cols(F); nm, h = "4h", 48
    res = {}
    preds = {365: np.load(f"{OUT}/pred_전체+개장·365일_4h.npy")}
    for win in (180, 270):
        pred, lo, hi = wf_monthly(F, FC, nm, h, win, seeds=(0,))
        np.save(f"{OUT}/pred_전체+개장·{win}일_4h.npy", np.c_[pred, lo, hi]); preds[win] = np.c_[pred, lo, hi]
        r = phase_eval(F, pred, lo, hi, nm, h); res[f"{win}일"] = r
        print(win, "일", json.dumps({k: v for k, v in r.items() if "phase0" not in k}, ensure_ascii=False), "| 위상0 4bp", r["c4.0_phase0"], flush=True)
    E = np.mean([preds[w] for w in (180, 270, 365)], axis=0)
    np.save(f"{OUT}/pred_앙상블180_270_365_4h.npy", E)
    r = phase_eval(F, E[:, 0], E[:, 1], E[:, 2], nm, h); res["앙상블"] = r
    print("앙상블", json.dumps({k: v for k, v in r.items() if "phase0" not in k}, ensure_ascii=False), "| 위상0 4bp", r["c4.0_phase0"], flush=True)
    json.dump(res, open(f"{OUT}/window_grid.json", "w"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--windows":
    window_grid()



def replicate(sym="BTC"):
    """독립 재현(사전 고정 06:58): ETH 최고 후보와 같은 절차(전체+개장 피쳐 · 365일 창 · 매달 재학습 · 4h)를 다른 코인에. q 0.2(사전)·0.1(사후 선택값)."""
    build(sym); add_open_feats(sym)
    F = pd.read_parquet(panel_path(sym)); FC = feat_cols(F); nm, h = "4h", 48
    pred, lo, hi = wf_monthly(F, FC, nm, h, 365, seeds=(0,))
    np.save(f"{OUT}/pred_{sym}_365_4h.npy", np.c_[pred, lo, hi])
    r = phase_eval(F, pred, lo, hi, nm, h); y = F.y_4h.to_numpy(); ts = F.ts.to_numpy(); main = ts >= np.datetime64("2025-01-01")
    sq = pd.Series(pred); l1, h1 = sq.shift(1).expanding(8640).quantile(.1).to_numpy(), sq.shift(1).expanding(8640).quantile(.9).to_numpy()
    r["q0.1"] = {c: trade_eval(np.where(main, pred, np.nan), np.where(main, y, np.nan), ts, h, l1, h1, c) for c in (1.4, 4.0, 8.0)}
    print(sym, len(FC), "피쳐", json.dumps(r, ensure_ascii=False, default=str), flush=True)
    json.dump(r, open(f"{OUT}/replicate_{sym}.json", "w"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--replicate":
    replicate(sys.argv[2] if len(sys.argv) > 2 else "BTC")


def pooled(syms=("ETH", "BTC", "SOL"), window=365):
    """다코인 풀링(07:08 사전 고정): 공통 피쳐 · 365일 창 · 매달 재학습 · 4h. 코인별 채점(q0.2 사전 · q0.1 참고).
    + ETH 단독에서 B·S(ETH 전용) 뺀 변형으로 «피쳐 차이 vs 자산 차이» 분리."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    for s_ in syms:
        if not os.path.exists(panel_path(s_)):
            build(s_); add_open_feats(s_)
    Ps = {s_: pd.read_parquet(panel_path(s_)) for s_ in syms}
    common = sorted(set.intersection(*[set(feat_cols(P)) for P in Ps.values()]))
    nm, h = "4h", 48
    out = {"n_common": len(common)}
    # ETH 단독 B·S 제외
    pe, le, he = wf_monthly(Ps["ETH"], common, nm, h, window, seeds=(0,))
    out["ETH 단독·공통피쳐"] = phase_eval(Ps["ETH"], pe, le, he, nm, h)
    print("ETH 단독·공통피쳐", json.dumps({k: v for k, v in out["ETH 단독·공통피쳐"].items() if "phase0" not in k}), flush=True)
    # 풀링
    ts0 = Ps["ETH"].ts.to_numpy()
    X = {s_: P[common].to_numpy(dtype=np.float32) for s_, P in Ps.items()}
    Y = {s_: P[f"y_{nm}"].to_numpy() for s_, P in Ps.items()}
    TS = {s_: P.ts.to_numpy() for s_, P in Ps.items()}
    pred = {s_: np.full(len(Ps[s_]), np.nan) for s_ in syms}; lo = {s_: pred[s_].copy() for s_ in syms}; hi = {s_: pred[s_].copy() for s_ in syms}
    for m0 in pd.date_range("2025-01-01", pd.Timestamp(ts0.max()).normalize(), freq="MS"):
        m1 = m0 + pd.offsets.MonthBegin(1); cut = np.datetime64(m0) - np.timedelta64(5 * h, "m")
        Xs, ys = [], []
        for s_ in syms:
            tr = (TS[s_] < cut) & (TS[s_] >= cut - np.timedelta64(window, "D")) & np.isfinite(Y[s_])
            Xs.append(X[s_][tr][::3]); ys.append(Y[s_][tr][::3] / np.nanstd(Y[s_][tr]))       # 코인별 변동성으로 표준화
        mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=1000, l2_regularization=1.0,
                                            random_state=0).fit(np.vstack(Xs), np.concatenate(ys))
        for s_ in syms:
            te = (TS[s_] >= np.datetime64(m0)) & (TS[s_] < np.datetime64(m1)); tr = (TS[s_] < cut) & (TS[s_] >= cut - np.timedelta64(window, "D"))
            pred[s_][te] = mdl.predict(X[s_][te]); q_ = np.quantile(mdl.predict(X[s_][tr][::7]), [.2, .8]); lo[s_][te], hi[s_][te] = q_
        print("  월", m0.strftime("%Y-%m"), flush=True)
    for s_ in syms:
        r = phase_eval(Ps[s_], pred[s_], lo[s_], hi[s_], nm, h); out[f"풀링→{s_}"] = r
        np.save(f"{OUT}/pred_pooled_{s_}_4h.npy", np.c_[pred[s_], lo[s_], hi[s_]])
        print(f"풀링→{s_}", json.dumps({k: v for k, v in r.items() if "phase0" not in k}), "| 위상0 4bp", r["c4.0_phase0"], flush=True)
    json.dump(out, open(f"{OUT}/pooled.json", "w"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--pooled":
    pooled()


def plot_equity():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont("/mnt/c/Windows/Fonts/malgun.ttf"); plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for sym, path, panel in (("ETH", f"{OUT}/pred_전체+개장·365일_4h.npy", panel_path("ETH")), ("BTC 재현", f"{OUT}/pred_BTC_365_4h.npy", panel_path("BTC"))):
        if not (os.path.exists(path) and os.path.exists(panel)):
            continue
        F = pd.read_parquet(panel, columns=["ts", "y_4h"]); y = F.y_4h.to_numpy(); ts = F.ts.to_numpy(); P = np.load(path); pred = P[:, 0]
        sq = pd.Series(pred)
        for q, ls in ((0.1, "-"), (0.2, "--")):
            lo, hi = sq.shift(1).expanding(8640).quantile(q).to_numpy(), sq.shift(1).expanding(8640).quantile(1 - q).to_numpy()
            idx = np.arange(0, len(y), 48); s = np.where(pred[idx] >= hi[idx], 1, np.where(pred[idx] <= lo[idx], -1, 0))
            m = (ts[idx] >= np.datetime64("2025-01-01")) & (s != 0) & np.isfinite(y[idx])
            for cost, a in ((4.0, 1.0), (8.0, .45)):
                g = pd.Series(s[m] * y[idx][m] - cost, index=pd.to_datetime(ts[idx][m])).cumsum() / 100
                ax.plot(g.index, g.values, ls, alpha=a, color="C0" if sym == "ETH" else "C3", label=f"{sym} 상·하위 {int(q*100)}% · 비용 {cost:g}bp")
    ax.axhline(0, color="k", lw=.6); ax.set_ylabel("누적 수익(%, 1배 노출 · 4시간 보유 합)"); ax.set_title("4시간 방향 모델 — 매달 재학습(최근 365일) · 2025-01~2026-09 표본 밖")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(f"{OUT}/equity.png", dpi=130); print(f"{OUT}/equity.png")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--plot":
    plot_equity()


def seeds_and_importance():
    """최종 후보(ETH · 전체+개장 · 365일 · 월별) 시드 1·2 재현 + 순열 중요도(2025-07·2026-03 두 달 모델, 그 달 시험)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    F = pd.read_parquet(panel_path("ETH")); FC = feat_cols(F); nm, h = "4h", 48
    out = {}
    for sd in (1, 2):
        pred, lo, hi = wf_monthly(F, FC, nm, h, 365, seeds=(sd,))
        r = phase_eval(F, pred, lo, hi, nm, h); out[f"seed{sd}"] = {k: v for k, v in r.items() if "phase0" not in k}
        print("시드", sd, json.dumps(out[f"seed{sd}"]), flush=True)
    y = F[f"y_{nm}"].to_numpy(); ts = F.ts.to_numpy(); X = F[FC].to_numpy(dtype=np.float32); imp = []
    for m0 in ("2025-07-01", "2026-03-01"):
        m0 = pd.Timestamp(m0); cut = np.datetime64(m0) - np.timedelta64(5 * h, "m")
        tr = (ts < cut) & (ts >= cut - np.timedelta64(365, "D")) & np.isfinite(y)
        te = (ts >= np.datetime64(m0)) & (ts < np.datetime64(m0 + pd.DateOffset(months=3))) & np.isfinite(y)
        mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=1000, l2_regularization=1.0,
                                            random_state=0).fit(X[tr][::2], y[tr][::2])
        Xt, yt = X[te][::2], y[te][::2]; base = ic(mdl.predict(Xt), yt); rng = np.random.default_rng(0); d = {}
        for j, c in enumerate(FC):
            Xp = Xt.copy(); Xp[:, j] = rng.permutation(Xp[:, j]); d[c] = base - ic(mdl.predict(Xp), yt)
        imp.append(pd.Series(d)); print("기준 IC", m0.date(), round(base, 4), flush=True)
    I = pd.concat(imp, axis=1).mean(1)
    out["perm_top20"] = I.sort_values(ascending=False).head(20).round(4).to_dict()
    out["perm_family"] = I.groupby(lambda c: c[0]).sum().round(4).to_dict()
    print("중요도 상위:", out["perm_top20"], "\n묶음:", out["perm_family"], flush=True)
    json.dump(out, open(f"{OUT}/seeds_importance.json", "w"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--seedsimp":
    seeds_and_importance()


def final_model():
    """최종 모델(07:30 고정): ETH · 전체 피쳐(+개장) · 최근 365일 · 매달 재학습(정화 4h) · HGB 시드 0·1·2 평균 · 4h 보유.
    신호 = 예측이 과거 예측 분포(확장, 30일 예열)의 상위 q 면 롱 · 하위 q 면 숏. q = 0.2(사전) · 0.1(참고).
    채점 = 위상 8개를 1/8 씩 든 포트폴리오의 일 손익(일 블록 CI) · 비용 1.4/4/8bp."""
    F = pd.read_parquet(panel_path("ETH")); FC = feat_cols(F); nm, h = "4h", 48
    pred, lo, hi = wf_monthly(F, FC, nm, h, 365, seeds=(0, 1, 2))
    np.save(f"{OUT}/pred_FINAL_eth_4h.npy", np.c_[pred, lo, hi])
    y = F[f"y_{nm}"].to_numpy(); ts = F.ts.to_numpy(); sq = pd.Series(pred); out = {"features": FC, "ic": round(ic(pred[ts >= np.datetime64("2025-01-01")], y[ts >= np.datetime64("2025-01-01")]), 4)}
    for q in (0.2, 0.1):
        l_, h_ = sq.shift(1).expanding(8640).quantile(q).to_numpy(), sq.shift(1).expanding(8640).quantile(1 - q).to_numpy()
        for cost in (1.4, 4.0, 8.0):
            rows = []
            for off in range(0, 48, 6):
                idx = np.arange(off, len(y), 48); s = np.where(pred[idx] >= h_[idx], 1, np.where(pred[idx] <= l_[idx], -1, 0))
                m = (ts[idx] >= np.datetime64("2025-01-01")) & (s != 0) & np.isfinite(y[idx])
                rows.append(pd.Series(s[m] * y[idx][m] - cost, index=pd.to_datetime(ts[idx][m])))
            g = pd.concat(rows) / 8; d = g.groupby(g.index.floor("D")).sum()
            d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"), fill_value=0.0)
            bs = [d.sample(len(d), replace=True, random_state=i).mean() for i in range(1000)]
            yr = d.groupby(d.index.year).sum()
            out[f"q{q}_c{cost}"] = {"day_mean_bp": round(float(d.mean()), 2), "ci": [round(float(np.quantile(bs, .025)), 2), round(float(np.quantile(bs, .975)), 2)],
                                    "sharpe_ann": round(float(d.mean() / d.std() * np.sqrt(365)), 2), "by_year_bp": {int(k): int(v) for k, v in yr.items()},
                                    "max_dd_bp": int((d.cumsum() - d.cumsum().cummax()).min()), "trades_per_day": round(float(sum(len(r) for r in rows) / 8 / len(d)), 2)}
            print(f"q{q} 비용{cost}", json.dumps(out[f"q{q}_c{cost}"]), flush=True)
    print("IC", out["ic"])
    json.dump(out, open(f"{OUT}/final_model.json", "w"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--finalmodel":
    final_model()


def freeze():
    """전진 사전등록용 동결(07:58): 2026-09-14 까지 최근 365일로 학습한 시드 0·1·2 HGB + 피쳐 목록 + 문턱 + 판정 규칙.
    판정(사전 고정): 동결 이후 새로 쌓인 데이터(2026-09-15~)에서, 위상 8개 1/8 포트폴리오 · 상·하위 20% · 비용 4bp · 일 블록 95% CI 하한 > 0
    이고 6개월(≥180일) 이상 쌓였을 때만 «통과». 그 전엔 건수·날짜만 보고 결과를 들여다보지 않는다. 매달 재학습은 하지 않는다(동결 모델 그대로)."""
    import hashlib, joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    F = pd.read_parquet(panel_path("ETH")); FC = feat_cols(F); y = F.y_4h.to_numpy(); ts = F.ts.to_numpy()
    cut = ts.max() - np.timedelta64(240, "m"); tr = (ts < cut) & (ts >= cut - np.timedelta64(365, "D")) & np.isfinite(y)
    X = F[FC].to_numpy(dtype=np.float32)
    models = [HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=1000, l2_regularization=1.0,
                                            random_state=s).fit(X[tr][::2], y[tr][::2]) for s in (0, 1, 2)]
    oos = np.load(f"{OUT}/pred_FINAL_eth_4h.npy")[:, 0]; oos = oos[np.isfinite(oos)]
    spec = {"frozen_at_kst": "2026-10-10 07:58", "train_window": [str(ts[tr].min()), str(ts[tr].max())], "n_train": int(tr.sum()),
            "features": FC, "target": "y_4h = ln(close[k+48]/close[k]) bp, 결정 = 5분봉 k 마감", "thresholds_q20_q80": [float(np.quantile(oos, .2)), float(np.quantile(oos, .8))],
            "hold": "4h", "cost_bp": 4.0, "verdict_rule": freeze.__doc__.split("판정(사전 고정): ")[1].strip(),
            "panel_sha256": hashlib.sha256(open(panel_path("ETH"), "rb").read()).hexdigest()[:16],
            "backtest_2025_26": json.load(open(f"{OUT}/final_model.json"))["q0.2_c4.0"],
            "caveats": ["2023~24 같은 방법 불통과(4bp 일 −5.5)", "BTC 재현 매매 0", "수익 37% 가 급락 5일", "2026 4bp 합 ≈ 0"]}
    joblib.dump({"models": models, "features": FC}, f"{OUT}/FROZEN_eth_4h_models.joblib")
    json.dump(spec, open(f"{OUT}/FROZEN_eth_4h_spec.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: v for k, v in spec.items() if k != "features"}, ensure_ascii=False, default=str))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--freeze":
    freeze()


def portfolio3():
    """세 부품 위험 맞춤 결합(08:00 고정): ① 4h HGB(2023~24 = 방법 홀드아웃 시드 0, 2025~26 = 최종 시드 3) ② 일봉 추세 5표
    ③ UTC 개장 되돌림(|첫 30분| > 0.5σ 반대·4h). 비용 4bp. 가중 = 직전 90일 일 표준편차 역수(전날까지, 60일 예열 후 시작, 부품별 상한 3배)."""
    F = pd.read_parquet(panel_path("ETH")); y = F.y_4h.to_numpy(); ts = F.ts.to_numpy()
    ph = np.load(f"{OUT}/pred_holdout_2023_24_4h.npy"); pf = np.load(f"{OUT}/pred_FINAL_eth_4h.npy")[:, 0]
    pred = np.where(ts < np.datetime64("2025-01-01"), np.r_[ph, np.full(len(F) - len(ph), np.nan)], pf)
    sq = pd.Series(pred); lo, hi = sq.shift(1).expanding(8640).quantile(.2).to_numpy(), sq.shift(1).expanding(8640).quantile(.8).to_numpy()
    rows = []
    for off in range(0, 48, 6):
        idx = np.arange(off, len(y), 48); s = np.where(pred[idx] >= hi[idx], 1, np.where(pred[idx] <= lo[idx], -1, 0))
        m = (s != 0) & np.isfinite(y[idx]); rows.append(pd.Series(s[m] * y[idx][m] - 4, index=pd.to_datetime(ts[idx][m])))
    g = pd.concat(rows) / 8; hgb = g.groupby(g.index.floor("D")).sum()
    T = F.ts + pd.Timedelta("5min"); m0 = ((T.dt.hour == 0) & (T.dt.minute == 0)).to_numpy()
    pos = np.where(np.abs(F.P_trend5[m0]) >= .5, np.sign(F.P_trend5[m0]), 0.0)
    trend = pd.Series(pos * F.y_1d[m0].to_numpy() - np.abs(np.diff(np.r_[0, pos])) * 4, index=(F.ts[m0] + pd.Timedelta("5min")).dt.floor("D").to_numpy())
    O = pd.read_parquet(f"{OUT}/open_panel.parquet"); E = O[O.sym == "ETH"].dropna(subset=["f30", "n4", "rv1d"])
    so = np.where(E.f30.abs() > 0.5 * E.rv1d * np.sqrt(6), -np.sign(E.f30), 0)
    openf = pd.Series(np.where(so != 0, so * E.n4 - 4, 0.0), index=pd.to_datetime(E.day.to_numpy()))
    idx = pd.date_range("2023-01-01", "2026-09-12", freq="D")
    P = pd.DataFrame({"4h HGB": hgb, "일봉 추세": trend, "UTC 개장 되돌림": openf}).reindex(idx).fillna(0.0)
    sd = P.rolling(90, min_periods=60).std().shift(1)
    W = (sd.mean(1).to_numpy()[:, None] / sd).clip(upper=3.0)                       # 평균 변동 대비 역수, 상한 3
    P["결합(위험 맞춤)"] = (P.iloc[:, :3] * W).sum(1) / 3
    P = P[W.notna().all(1)]
    out = {"corr": P.iloc[:, :3].corr().round(3).to_dict()}
    for c in P.columns:
        x = P[c]; r = {}
        for Y in (2023, 2024, 2025, 2026):
            v = x[x.index.year == Y]; r[Y] = round(float(v.mean() / v.std() * np.sqrt(365)), 2)
        eq = x.cumsum(); bs = [x.sample(len(x), replace=True, random_state=i).mean() for i in range(1000)]
        out[c] = {"yearly_sharpe": r, "sharpe_all": round(float(x.mean() / x.std() * np.sqrt(365)), 2), "day_mean_bp": round(float(x.mean()), 2),
                  "ci": [round(float(np.quantile(bs, .025)), 2), round(float(np.quantile(bs, .975)), 2)], "mdd_bp": int((eq - eq.cummax()).min())}
        print(f"{c:14s}", json.dumps(out[c], ensure_ascii=False), flush=True)
    print("상관", P.iloc[:, :3].corr().round(3).to_dict())
    json.dump(out, open(f"{OUT}/portfolio3.json", "w"), ensure_ascii=False, indent=1, default=str)
    P.to_parquet(f"{OUT}/portfolio3_daily.parquet")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--portfolio":
    portfolio3()


def forward_status():
    """전진 사전등록 상태 — 결과는 보지 않고(판정 전 금지) 동결 이후 쌓인 일수와 판정 가능일만."""
    days = sorted(p.name[11:21] for p in (BV / "panel_daily").glob("ETHUSDT-5m-*.parquet"))
    fwd = [d for d in days if d >= "2026-09-15"]
    first = pd.Timestamp("2026-09-15") + pd.Timedelta(days=180)
    print(f"동결 이후 ETH 5분 일 파일 {len(fwd)}일({fwd[0] if fwd else '-'} ~ {fwd[-1] if fwd else '-'}) · 판정 조건 ≥180일 → 가장 이른 판정 {first.date()} · "
          f"🔴aggfeat·bookDepth(4h HGB 입력)는 로컬 원천 2026-08-22 에서 끊김 → HGB 부품 판정 전 갱신 필요 · 추세·개장 되돌림은 5분 패널만으로 판정 가능")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--forward-status":
    forward_status()


def coin_replication():
    """권고 규칙(일봉 추세 5표 + UTC 00:00 개장 되돌림 k=.5·4h, 변동성 역수 비중) 을 파라미터 그대로 20코인에. 2022~2026 연도별 샤프. 08:00 고정."""
    O = pd.read_parquet(f"{OUT}/open_panel.parquet").dropna(subset=["f30", "rv1d"])
    res, daily = {}, {}
    for sym in sorted(O.sym.unique()):
        p = load_panel(sym); c_ = p.close.ffill().to_numpy()
        d = pd.Series(c_, index=p.timestamp).resample("1D", label="left").last(); dv = d.to_numpy()
        votes = np.mean([np.sign(dv - np.r_[np.full(L, np.nan), dv[:-L]]) for L in (7, 14, 28, 56, 90)], axis=0)
        pos = np.where(np.abs(votes) >= .5, np.sign(votes), 0.0)
        ret = np.r_[np.log(dv[1:] / dv[:-1]) * 1e4, np.nan]
        tr = pd.Series(pos * ret - np.abs(np.diff(np.r_[0, pos])) * 4, index=d.index + pd.Timedelta("1D"))
        E = O[O.sym == sym]; so = np.where(E.f30.abs() > .5 * E.rv1d * np.sqrt(6), -np.sign(E.f30), 0)
        op = pd.Series(np.where(so != 0, so * E.n4 - 4, 0.0), index=pd.to_datetime(E.day.to_numpy()))
        B = pd.DataFrame({"tr": tr, "op": op}).reindex(pd.date_range("2022-04-01", "2026-09-12", freq="D")).fillna(0.0)
        sd = B.rolling(90, min_periods=60).std().shift(1); W = (sd.mean(1).to_numpy()[:, None] / sd).clip(upper=3.0)
        x = ((B * W).sum(1) / 2)[W.notna().all(1)]
        daily[sym] = x / x[x.index < "2023-01-01"].std()                               # 코인 간 동일 위험(2022 변동으로 고정)
        res[sym] = {Y: round(float(x[x.index.year == Y].mean() / x[x.index.year == Y].std() * np.sqrt(365)), 2) for Y in range(2022, 2027)}
        res[sym]["all"] = round(float(x.mean() / x.std() * np.sqrt(365)), 2)
        print(sym, res[sym], flush=True)
    D = pd.DataFrame(daily).fillna(0.0); M = D.mean(1)
    res["20코인 평균 포트폴리오"] = {Y: round(float(M[M.index.year == Y].mean() / M[M.index.year == Y].std() * np.sqrt(365)), 2) for Y in range(2022, 2027)}
    res["20코인 평균 포트폴리오"]["all"] = round(float(M.mean() / M.std() * np.sqrt(365)), 2)
    pos_years = {s: sum(v[Y] > 0 for Y in range(2022, 2027)) for s, v in res.items() if s != "20코인 평균 포트폴리오"}
    res["플러스 해 수 분포"] = pd.Series(pos_years).value_counts().sort_index().to_dict()
    print("20코인 평균", res["20코인 평균 포트폴리오"], "| 코인별 플러스 해 수 분포", res["플러스 해 수 분포"], flush=True)
    json.dump(res, open(f"{OUT}/coin_replication.json", "w"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--coins":
    coin_replication()


def execution_realism(W=5, fee_m=0.0, fee_t=4.0, vol_gate=False, hold="pred_holdout_2023_24_4h_seed3.npy", save=None):
    """4h HGB(2023~24 방법 홀드아웃 + 2025~26 최종) 현실 실행: 결정 직전 1분 종가에 지정가, W 분 안 관통해야 체결(메이커 fee_m),
    아니면 W 분 뒤 종가로 시장가(fee_t). 진입·청산 모두. 위상 8 분산. vol_gate = 직전 1일 변동 ≥ 과거 확장 67분위일 때만."""
    import research_eth_liqmap_5m_direction_20261010 as LQ5
    F = pd.read_parquet(panel_path("ETH"), columns=["ts", "P_rv1d"]); ts = F.ts.to_numpy()
    ph = np.load(f"{OUT}/{hold}"); pf = np.load(f"{OUT}/pred_FINAL_eth_4h.npy")[:, 0]                # 홀드아웃 기본 = 시드 3(최종과 같은 조건)
    pred = np.where(ts < np.datetime64("2025-01-01"), np.r_[ph, np.full(len(F) - len(ph), np.nan)], pf); sq = pd.Series(pred)
    lo, hi = sq.shift(1).expanding(8640).quantile(.2).to_numpy(), sq.shift(1).expanding(8640).quantile(.8).to_numpy()
    gate = F.P_rv1d.to_numpy() >= F.P_rv1d.shift(1).expanding(288 * 90).quantile(2 / 3).to_numpy()
    k = LQ5.load_1m(); t1 = k.t.to_numpy(); h1, l1, c1 = k.h.to_numpy(), k.l.to_numpy(), k.c.to_numpy()

    def leg(T, side):
        i = np.searchsorted(t1, T); P = c1[i - 1]
        f = (l1[i:i + W] < P).any() if side > 0 else (h1[i:i + W] > P).any()
        return (P, fee_m) if f else (c1[i + W - 1], fee_t)
    rows = []
    for off in range(0, 48, 6):
        idx = np.arange(off, len(ts), 48); idx = idx[(ts[idx] >= np.datetime64("2023-02-01")) & (idx + 48 < len(ts))]
        for j in idx:
            if not np.isfinite(pred[j]) or (vol_gate and not gate[j]):
                continue
            s = 1 if pred[j] >= hi[j] else (-1 if pred[j] <= lo[j] else 0)
            if s:
                T0 = int((ts[j] + np.timedelta64(5, "m")).astype("datetime64[ms]").astype(np.int64))
                pe, ce = leg(T0, s); px, cx = leg(T0 + 14_400_000, -s)
                rows.append((pd.Timestamp(T0, unit="ms").floor("D"), (s * np.log(px / pe) * 1e4 - ce - cx) / 8))
    d = pd.Series([r[1] for r in rows], index=[r[0] for r in rows]).groupby(level=0).sum()
    d = d.reindex(pd.date_range("2023-02-01", "2026-09-13", freq="D"), fill_value=0.0)
    bs = np.quantile([d.sample(len(d), replace=True, random_state=i).mean() for i in range(500)], [.025, .975])
    out = {"W": W, "fee_m": fee_m, "fee_t": fee_t, "vol_gate": vol_gate, "yearly_sharpe": {int(Y): round(float(x.mean() / x.std() * np.sqrt(365)), 2) for Y, x in d.groupby(d.index.year)},
           "sharpe": round(float(d.mean() / d.std() * np.sqrt(365)), 2), "day_bp": round(float(d.mean()), 2), "ci": [round(float(x), 2) for x in bs]}
    print(json.dumps(out, ensure_ascii=False), flush=True)
    if save:
        d.to_frame("pnl").to_parquet(f"{OUT}/{save}")
    return out


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--exec":
    res = [execution_realism(save="hgb_exec_daily_seed3.parquet"), execution_realism(fee_m=1.0), execution_realism(vol_gate=True),
           execution_realism(hold="pred_holdout_2023_24_4h.npy")]
    json.dump(res, open(f"{OUT}/execution_realism.json", "w"), ensure_ascii=False, indent=1)


def replicate_combo(sym="SOL"):
    """독립 복제(09:00 고정, 선택에 안 쓴 자산): 같은 방법 전부를 sym 에 — 4h HGB(전체 피쳐·365일·월별·시드 0, 2023~2026) 를
    5분봉 지정가 근사(다음 5분봉 안 관통해야 체결, 아니면 그 봉 종가 시장가 4bp)로 + 일봉 추세 5표 + UTC 개장 되돌림 → 위험 맞춤 결합."""
    if not os.path.exists(panel_path(sym)):
        build(sym); add_open_feats(sym)
    F = pd.read_parquet(panel_path(sym)); FC = feat_cols(F); ts = F.ts.to_numpy()
    pred, _, _ = wf_monthly(F, FC, "4h", 48, 365, seeds=(0,), start="2023-01-01")
    np.save(f"{OUT}/pred_{sym}_combo_4h.npy", pred)
    sq = pd.Series(pred); lo, hi = sq.shift(1).expanding(8640).quantile(.2).to_numpy(), sq.shift(1).expanding(8640).quantile(.8).to_numpy()
    b = load_panel(sym); t5 = (b.timestamp.astype("int64") // 10**6).to_numpy(); h5, l5, c5 = b.high.to_numpy(), b.low.to_numpy(), b.close.ffill().to_numpy()

    def leg(T, side):
        i = np.searchsorted(t5, T); P = c5[i - 1]; f = (l5[i] < P) if side > 0 else (h5[i] > P)
        return (P, 0.0) if f else (c5[i], 4.0)
    rows = []
    for off in range(0, 48, 6):
        idx = np.arange(off, len(ts), 48); idx = idx[(ts[idx] >= np.datetime64("2023-02-01")) & (idx + 48 < len(ts))]
        for j in idx:
            if not np.isfinite(pred[j]):
                continue
            s = 1 if pred[j] >= hi[j] else (-1 if pred[j] <= lo[j] else 0)
            if s:
                T0 = int((ts[j] + np.timedelta64(5, "m")).astype("datetime64[ms]").astype(np.int64)); pe, ce = leg(T0, s); px, cx = leg(T0 + 14_400_000, -s)
                rows.append((pd.Timestamp(T0, unit="ms").floor("D"), (s * np.log(px / pe) * 1e4 - ce - cx) / 8))
    hgb = pd.Series([r[1] for r in rows], index=[r[0] for r in rows]).groupby(level=0).sum()
    d = pd.Series(c5, index=b.timestamp).resample("1D", label="left").last(); dv = d.to_numpy()
    votes = np.mean([np.sign(dv - np.r_[np.full(L, np.nan), dv[:-L]]) for L in (7, 14, 28, 56, 90)], axis=0); pos = np.where(np.abs(votes) >= .5, np.sign(votes), 0.0)
    tr = pd.Series(pos * np.r_[np.log(dv[1:] / dv[:-1]) * 1e4, np.nan] - np.abs(np.diff(np.r_[0, pos])) * 4, index=d.index + pd.Timedelta("1D"))
    O = pd.read_parquet(f"{OUT}/open_panel.parquet").dropna(subset=["f30", "rv1d"]); E = O[O.sym == sym]
    so = np.where(E.f30.abs() > .5 * E.rv1d * np.sqrt(6), -np.sign(E.f30), 0); op = pd.Series(np.where(so != 0, so * E.n4 - 4, 0.0), index=pd.to_datetime(E.day.to_numpy()))
    B = pd.DataFrame({"4h HGB": hgb, "추세": tr, "개장": op}).reindex(pd.date_range("2023-03-02", "2026-09-12", freq="D")).fillna(0.0)
    sd = B.rolling(90, min_periods=60).std().shift(1); W = (sd.mean(1).to_numpy()[:, None] / sd).clip(upper=3.0)
    B["결합"] = (B.iloc[:, :3] * W).sum(1) / 3; B = B[W.notna().all(1)]
    out = {}
    for c in B.columns:
        x = B[c]; out[c] = {"yearly": {int(Y): round(float(v.mean() / v.std() * np.sqrt(365)), 2) for Y, v in x.groupby(x.index.year)},
                            "sharpe": round(float(x.mean() / x.std() * np.sqrt(365)), 2), "corr_hgb": round(float(x.corr(B["4h HGB"])), 3)}
        print(sym, c, json.dumps(out[c], ensure_ascii=False), flush=True)
    json.dump(out, open(f"{OUT}/replicate_combo_{sym}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--repcombo":
    replicate_combo(sys.argv[2])
