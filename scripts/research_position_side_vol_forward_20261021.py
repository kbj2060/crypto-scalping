"""1일 롱·숏 분리 증분 → 배포 4h 변동성 GBM 위 증분 — 전진(2026-09-15~) 사전등록 (2026-10-21, 사용자 «배포된 4h 변동성 GBM 위에서도
증분이 있는지, 2026-09-15 이후 전진 자료로 새 사전등록 검정»). 앞 단계: research_position_side_price_impact_20261021.py (사후 탐색 후보).

🔒사전등록(전진 자료를 보기 전에 커밋으로 동결):
- GBM = 배포 방식 그대로(live_eth_sizing_vol_model_20260912: build_features 22열 · HGB max_iter300 lr.06 depth6 · 시드 8 · 로그 rv 타깃).
  학습 자료 = 서버 vol_panel(빠진 날 vision 으로 채움) · 끝 = 타깃까지 다 끝난 봉(시각 ≤ 2026-09-14 19:55 UTC). ETH = ETH 만 · XRP = ETH+XRP 합쳐(배포 그대로).
- 타깃 z = log(다음 48봉 5분 로그수익 표준편차) − log(GBM 예측) — 배포 모델의 잔차.
- 피쳐 E = M1(1일 창 물린 롱 고통 + 물린 숏 고통, 롱·숏 분리 장부) 을 [Msym1 + 앞 단계 통제 19 + 변동성 통제 + |VWAP 1·7·30일 거리| +
  |r24h|·|r7d|·|r30d|·|테이커 1h·24h|·|ΔOI 24h|] 위에 선형 잔차(순위). 계수·순위 기준 = 2022-02-01~2026-09-14(과거만).
- 스냅샷 = 매시 정각 봉 마감(앞 단계 그대로). 전진 = 2026-09-15 00:00 UTC 부터 타깃이 다 찬 마지막 스냅샷까지.
- 주 검정: ETH·XRP 전진 스피어만 IC(E, z) 의 평균 > 0 (단측, 방향은 앞 단계 8/8 양). 일 블록 부트스트랩(두 코인 같은 날 함께) 2000.
  통과 = 단측 p < .05 그리고 두 코인 모두 IC > 0.
- 보조(판정 아님): 코인별 전진 IC · 과거 2025-01~2026-09-14 에서 GBM(≤2024-12-31 학습) 잔차와 IC · 실용: z ≈ b·E 를 과거 OOS 잔차로 맞춰
  전진에서 GBM×exp(b·E) 의 로그 오차제곱 변화(%).
- 🔴검정력: 기대 IC .03~.04 · 전진 ≈ 35일 × 2코인, 4h 겹침 → 유효 표본 작음 → 검정력 대략 20~35%. 불통과 = «판정 불가»(반증 아님), 통과만 의미.
- BTC·SOL 은 배포 GBM 이 없어 뺀다.
사용: --data → --fit · --books → --test
"""
import json, os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import research_position_side_price_impact_20261021 as R   # noqa: E402
import live_eth_sizing_vol_model_20260912 as V               # noqa: E402
import build_binance_vision_panel_20260915 as B              # noqa: E402

OUT = "tmp/position_side_vol_forward_20261021"
VOLP = "/home/kbj20/crypto-scalping/data/live/vol_panel"
FWD0, LAST_DAY = pd.Timestamp("2026-09-15"), "2026-10-20"
CUT = {"fwd": pd.Timestamp("2026-09-14 19:55"), "hist": pd.Timestamp("2024-12-31 19:55")}
SYMS = ("ETHUSDT", "XRPUSDT")


def data():
    """① vol 패널 빠진 날 채움(vision 정적 아카이브, 순차) ② 프로파일 패널 09-15~ 이어 붙임(klines+metrics)."""
    os.makedirs(f"{OUT}/panel", exist_ok=True)
    for sym in SYMS:
        d = pd.read_parquet(f"{VOLP}/{sym}_5m.parquet")
        n = d.groupby(d.timestamp.dt.normalize()).size(); have = set(n[n >= 288].index)
        days = [x.strftime("%Y-%m-%d") for x in pd.date_range(d.timestamp.min().normalize(), LAST_DAY) if x not in have]
        add = []
        for x in days:
            time.sleep(0.3); k = V._vision_day(x, sym)
            if k is not None:
                add.append(k)
        d = pd.concat([d] + add).drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
        d.to_parquet(f"{OUT}/{sym}_vol.parquet")
        print(sym, "vol", len(d), d.timestamp.max(), "+", len(add), "/", len(days), "일 · 최대 간격", d.timestamp.diff().max(), flush=True)
        p = pd.read_parquet(f"{R.BV}/{sym}.parquet"); p = p[p.timestamp < FWD0]
        new = []
        for x in pd.date_range(FWD0, LAST_DAY):
            time.sleep(0.3); k = B.day(sym, x.strftime("%Y-%m-%d"))
            if k is not None:
                new.append(k)
        p = pd.concat([p] + new).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        p.to_parquet(f"{OUT}/panel/{sym}.parquet")
        f = p[p.timestamp >= FWD0]
        print(sym, "panel", len(p), p.timestamp.max(), "· 전진 봉", len(f), "OI 결측", int(f.sum_open_interest.isna().sum()),
              "· 최대 간격", p.timestamp.diff().max(), flush=True)


def vol_frame(sym):
    d = pd.read_parquet(f"{OUT}/{sym}_vol.parquet")
    X, y, ok, ts = V._frame(d)
    return X, y, ok, pd.DatetimeIndex(ts)


def fit():
    """배포 방식 GBM 둘(전진용 ≤09-14 · 과거용 ≤2024) × 코인 → 전 행 예측 저장."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    fr = {s: vol_frame(s) for s in SYMS}
    for tag, cut in CUT.items():
        for sym in SYMS:
            parts = [sym] if sym == "ETHUSDT" else ["XRPUSDT", "ETHUSDT"]
            A = np.vstack([np.nan_to_num(fr[s][0].to_numpy(float)[fr[s][2] & (fr[s][3] <= cut)], nan=0.0, posinf=0.0, neginf=0.0) for s in parts])
            yt = np.concatenate([fr[s][1][fr[s][2] & (fr[s][3] <= cut)] for s in parts])
            models = [HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6, random_state=int(sd)).fit(A, np.log(yt))
                      for sd in V.SEEDS]
            X, y, ok, ts = fr[sym]
            pr = V.predict_vol(models, X)
            pd.DataFrame({"ts": ts, "pred": pr, "y": y, "ok": ok}).to_parquet(f"{OUT}/{sym}_gbm_{tag}.parquet")
            print(tag, sym, "학습", len(yt), "행 · 끝", cut, flush=True)


def build_books():
    R.BV = f"{os.getcwd()}/{OUT}/panel"; R.OUT = f"{OUT}"
    for sym in SYMS:
        R.build(sym)


XCTRL = ["avw1", "avw7", "avw30", "a_r24h", "a_r7d", "a_r30d", "a_tk1h", "a_tk24", "a_doi24"]


def hist_rank(col, hist_mask):
    h = np.sort(col[hist_mask & np.isfinite(col)])
    return np.searchsorted(h, col, side="right") / len(h)


def ic_boot(a, b, day, nb=2000, seed=20261021):
    ud = np.unique(day); g = {u: np.flatnonzero(day == u) for u in ud}; rng = np.random.default_rng(seed)
    est = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    bs = np.array([pd.Series(a[ix]).corr(pd.Series(b[ix]), method="spearman")
                   for ix in (np.concatenate([g[u] for u in rng.choice(ud, len(ud))]) for _ in range(nb))])
    return est, bs


def frame_sym(sym):
    R.BV = f"{os.getcwd()}/{OUT}/panel"; R.OUT = f"{OUT}"
    S = R.frame(sym)
    for c in ("r24h", "r7d", "r30d", "tk1h", "tk24", "doi24"):
        S["a_" + c] = S[c].abs()
    S["bar"] = S.t - pd.Timedelta(minutes=5)
    hist = ((S.t >= "2022-02-01") & (S.t < FWD0)).to_numpy()
    v = ["s1h", "s24", "s7d", "ar1h", "ar4h", "vol_ratio", "doi1h", "doi24"]
    ctrl = v + [c for c in R.CTRL if c not in v] + XCTRL + ["Msym1"]
    Rk = {c: hist_rank(S[c].to_numpy(float), hist) for c in ctrl + ["M1"]}
    X = np.column_stack([np.ones(len(S))] + [Rk[c] for c in ctrl]); y = Rk["M1"]
    m = hist & np.isfinite(X).all(1) & np.isfinite(y)
    b = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
    S["E"] = y - X @ b
    for tag in CUT:
        g = pd.read_parquet(f"{OUT}/{sym}_gbm_{tag}.parquet").set_index("ts")
        g = g.reindex(S.bar)
        S[f"z_{tag}"] = np.where(g.ok.fillna(False).to_numpy(), np.log(g.y.to_numpy()) - np.log(g.pred.to_numpy()), np.nan)
    return S


def test():
    rep, fw = {}, {}
    for sym in SYMS:
        S = frame_sym(sym)
        F = S[(S.t >= FWD0) & np.isfinite(S.E) & np.isfinite(S.z_fwd)].reset_index(drop=True)
        fw[sym] = F
        H = S[(S.t >= "2025-01-01") & (S.t < FWD0) & np.isfinite(S.E) & np.isfinite(S.z_hist)].reset_index(drop=True)
        est, bs = ic_boot(H.E.to_numpy(), H.z_hist.to_numpy(), H.t.dt.floor("D").to_numpy(), nb=500)
        rep[f"보조 과거 2025~ {sym} IC(E, GBM≤2024 잔차)"] = {"ic": round(est, 4), "ci": [round(float(np.quantile(bs, q)), 4) for q in (.025, .975)], "n": len(H)}
        bb = float(np.polyfit(H.E, H.z_hist, 1)[0])                     # 실용: 과거 OOS 잔차로 기울기 → 전진 적용
        msle0 = float(np.mean(F.z_fwd ** 2)); msle1 = float(np.mean((F.z_fwd - bb * F.E) ** 2))
        rep[f"보조 실용 {sym}"] = {"b": round(bb, 4), "전진 로그오차제곱 GBM": round(msle0, 5), "+b·E": round(msle1, 5), "변화%": round(100 * (msle1 / msle0 - 1), 3)}
        rep[f"전진 표본 {sym}"] = {"n": len(F), "첫": str(F.t.min()), "끝": str(F.t.max()), "일수": int(F.t.dt.floor("D").nunique())}
    days = np.unique(np.concatenate([fw[s].t.dt.floor("D").to_numpy() for s in SYMS]))   # 주 검정: 두 코인 같은 날 함께
    idx = {s: {u: np.flatnonzero(fw[s].t.dt.floor("D").to_numpy() == u) for u in days} for s in SYMS}
    ic = {s: float(fw[s].E.corr(fw[s].z_fwd, method="spearman")) for s in SYMS}
    rng = np.random.default_rng(20261021); bs = []
    for _ in range(2000):
        dd = rng.choice(days, len(days)); v = []
        for s in SYMS:
            ix = np.concatenate([idx[s][u] for u in dd])
            v.append(fw[s].E.iloc[ix].reset_index(drop=True).corr(fw[s].z_fwd.iloc[ix].reset_index(drop=True), method="spearman"))
        bs.append(np.mean(v))
    bs = np.array(bs); mean = float(np.mean(list(ic.values()))); p1 = float(max((bs <= 0).mean(), 1 / 2000))
    rep["주 검정"] = {"IC 평균": round(mean, 4), "코인별": {k: round(v, 4) for k, v in ic.items()},
                    "단측 p": round(p1, 4), "90% CI(단측 하한=5%)": [round(float(np.quantile(bs, .05)), 4), round(float(np.quantile(bs, .95)), 4)],
                    "통과": bool(p1 < .05 and all(v > 0 for v in ic.values())), "일수": len(days)}
    json.dump(rep, open(f"{OUT}/report.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


def selftest():
    r = hist_rank(np.array([1., 2., 3., 4., 10.]), np.array([1, 1, 1, 1, 0], bool))
    assert np.allclose(r, [.25, .5, .75, 1., 1.]), r
    print("selftest OK -- 과거 분포 순위")


if __name__ == "__main__":
    a = sys.argv[1]
    {"--data": data, "--fit": fit, "--books": build_books, "--test": test, "--selftest": selftest}[a]()
