"""딥피쳐 2024~ 다시 고르기 + 동결 + 전진 채점 (2026-10-16, 사용자 «2024년 이후로 다시 고르고 전진 검증»).

앞 판(research_eth_deep_features_20261016.py)과 같은 피쳐 3,337개·같은 기준·같은 가짜 정답 20벌, 구간만 바꾼다(결과 전 고정):
- 탐색 2024-01 ~ 2025-06(2024 · 2025H1 두 반쪽 부호 같음, 일 군집 |t| ≥ 3) → 검증 2025-07 ~ 12(같은 부호 |t| ≥ 2).
- 동결: 지평별 생존 |t| 상위 ≤ 50 → 2024-01 ~ 2025-12 리지(α=10) · 표준화 평균/표준편차 · 신호 문턱(학습 예측 20·80분위) ·
  코드 sha256 → docs/prereg/eth_deep_features_2024sel_frozen_20261016.json.
- 보조: 2026-01 ~ 09-14 («일부 본 구간» -- 앞 판 확인 구간이자 원재료 연도별 IC 를 이미 봤다).
- 전진(주 판정): 2026-09-15 이후. --forward = data.binance.vision 에서 그 뒤 날짜를 받아 패널을 이어 붙이고(로컬 tmp, 공유 data/ 안 건드림)
  동결 모델을 봉 단위로 적용. 1차 열람 = 2026-10-16(09-15~10-15). 판정 = 09-15 부터 ≥ 180일(2027-03-14 이후):
  지평별 IC 일 블록 CI 하한 > 0 · 상·하위 20% 신호 평균 − 메이커 1.41bp 의 CI 하한 > 0. 그 전엔 --status(건수만).
사용: python scripts/research_eth_deep_features_2024sel_20261016.py --select | --forward | --status
"""
import hashlib, json, sys
from pathlib import Path

import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_eth_deep_features_20261016 as D  # noqa: E402

SEARCH, VAL, AUX = ("2024-01-01", "2025-07-01"), ("2025-07-01", "2026-01-01"), ("2026-01-01", "2026-09-15")
HALF = "2025-01-01"
FWD0 = "2026-09-15"
FROZEN = Path("docs/prereg/eth_deep_features_2024sel_frozen_20261016.json")
OUT = Path("tmp/deep_features_2024sel_20261016")


def select():
    OUT.mkdir(parents=True, exist_ok=True)
    Z, lch = D.level1()
    t = Z.index
    day = t.floor("D").asi8; month = t.to_period("M").astype(str).to_numpy()
    masks = {"a": (t >= SEARCH[0]) & (t < HALF), "b": (t >= HALF) & (t < SEARCH[1]),
             "s": (t >= SEARCH[0]) & (t < SEARCH[1]), "v": (t >= VAL[0]) & (t < VAL[1])}
    bnd = {k: np.flatnonzero(np.diff(day[m], prepend=day[m][0] - 1)) for k, m in masks.items()}
    Y = {h: (lch.shift(-h) - lch).to_numpy() * 1e4 for h in D.H}
    specs = [(c, "raw", c, None) for c in Z.columns] + D.level2_specs(list(Z.columns))
    rng = np.random.default_rng(20261016)
    yr = {h: [{k: D.rank_vec(Y[h][m]) for k, m in masks.items()}] + [
              {k: D.rank_vec(yn[m]) for k, m in masks.items()} for yn in (D.shuffle_days(Y[h], day, month, rng) for _ in range(D.NNULL))] for h in D.H}
    acc = {h: {k: ([], []) for k in masks} for h in D.H}; null_ok = {h: [] for h in D.H}
    for k0 in range(0, len(specs), 300):
        M = np.column_stack([D.feat(Z, s) for s in specs[k0:k0 + 300]]).astype(np.float64)
        R = {k: D.rank_cols(M[m]) for k, m in masks.items()}
        for h in D.H:
            oks = []
            for j, ys in enumerate(yr[h]):
                st = {k: D.day_t(R[k], ys[k], bnd[k]) for k in masks}
                ok = (np.abs(st["s"][1]) >= 3) & (np.sign(st["a"][0]) == np.sign(st["b"][0])) & \
                     (np.sign(st["v"][0]) == np.sign(st["s"][1])) & (np.abs(st["v"][1]) >= 2)
                if j == 0:
                    for k in masks:
                        acc[h][k][0].extend(st[k][0]); acc[h][k][1].extend(st[k][1])
                else:
                    oks.append(ok)
            null_ok[h].append(np.array(oks))
        print(f"  {min(k0 + 300, len(specs))}/{len(specs)}", flush=True)
    from sklearn.linear_model import Ridge
    names = [s[0] for s in specs]; is_raw = np.array([s[1] == "raw" for s in specs])
    frozen = {"created": "2026-10-16", "periods": {"search": SEARCH, "val": VAL, "train": (SEARCH[0], VAL[1]), "aux": AUX, "forward_from": FWD0},
              "code_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                              ("scripts/research_eth_deep_features_20261016.py", "scripts/research_eth_deep_features_2024sel_20261016.py")},
              "judge": "2027-03-14 이후 · 지평별 IC 일블록 CI 하한 > 0 · 상하위20% 평균−1.41bp CI 하한 > 0", "horizons": {}}
    report = {}
    for h in D.H:
        ic = {k: np.array(acc[h][k][0]) for k in masks}; tt = {k: np.array(acc[h][k][1]) for k in masks}
        surv = (np.abs(tt["s"]) >= 3) & (np.sign(ic["a"]) == np.sign(ic["b"])) & (np.sign(ic["v"]) == np.sign(tt["s"])) & (np.abs(tt["v"]) >= 2)
        nk = np.concatenate(null_ok[h], 1)
        top = np.flatnonzero(surv); top = top[np.argsort(-np.abs(tt["s"][top]))][:50]
        rep = {"생존": int(surv.sum()), "가짜 중앙/95분위": [float(np.median(nk.sum(1))), float(np.quantile(nk.sum(1), .95))],
               "1단 생존": int((surv & is_raw).sum()),
               "상위(이름, IC24, IC25H1, IC25H2, t)": [(names[i], round(float(ic["a"][i]), 4), round(float(ic["b"][i]), 4), round(float(ic["v"][i]), 4), round(float(tt["s"][i]), 1)) for i in top[:15]]}
        if len(top):
            M = np.column_stack([D.feat(Z, specs[i]) for i in top])
            tr = (t >= SEARCH[0]) & (t < pd.Timestamp(VAL[1]) - pd.Timedelta(hours=h)) & np.isfinite(Y[h])
            mu, sd = np.nanmean(M[tr], 0), np.nanstd(M[tr], 0); sd[sd == 0] = 1
            Zm = np.clip(np.nan_to_num((M - mu) / sd), -5, 5)
            mdl = Ridge(alpha=10.0).fit(Zm[tr], Y[h][tr])
            lo, hi = np.quantile(mdl.predict(Zm[tr]), [.2, .8])
            frozen["horizons"][str(h)] = {"specs": [list(specs[i]) for i in top], "mu": mu.tolist(), "sd": sd.tolist(),
                                          "coef": mdl.coef_.tolist(), "intercept": float(mdl.intercept_), "thr": [float(lo), float(hi)]}
            rep["보조 2026-01~09(일부 본 구간)"] = score(Z, lch, frozen["horizons"][str(h)], h, AUX)
        report[f"{h}h"] = rep
        print(h, json.dumps(rep, ensure_ascii=False), flush=True)
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(frozen, ensure_ascii=False))
    (OUT / "select_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("동결", FROZEN, hashlib.sha256(FROZEN.read_bytes()).hexdigest()[:16])


def predict(Z, fz):
    M = np.column_stack([D.feat(Z, tuple(s)) for s in fz["specs"]])
    Zm = np.clip(np.nan_to_num((M - np.array(fz["mu"])) / np.array(fz["sd"])), -5, 5)
    return Zm @ np.array(fz["coef"]) + fz["intercept"]


def score(Z, lch, fz, h, per):
    t = Z.index; m = (t >= per[0]) & (t < per[1])
    y = ((lch.shift(-h) - lch) * 1e4).to_numpy()
    p = predict(Z, fz)
    ok = m & np.isfinite(y)
    if ok.sum() < 48:
        return {"시간": int(ok.sum())}
    day = t.floor("D").asi8
    sp = lambda s: float(pd.Series(p[s]).corr(pd.Series(y[s]), method="spearman"))
    lo, hi = fz["thr"]; sig = np.where(p >= hi, 1.0, np.where(p <= lo, -1.0, 0.0))
    g = sig * y; on = ok & (sig != 0)
    ud = np.unique(day[ok]); ix = {u: np.flatnonzero(ok & (day == u)) for u in ud}; rb = np.random.default_rng(1)
    draws = [np.concatenate([ix[u] for u in rb.choice(ud, len(ud))]) for _ in range(500)]
    gb = [np.nanmean(g[s][sig[s] != 0]) for s in draws]
    return {"시간": int(ok.sum()), "일": int(len(ud)), "IC": round(sp(ok), 4),
            "IC CI": [round(float(np.quantile([sp(s) for s in draws], q)), 4) for q in (.025, .975)],
            "신호 비율": round(float(on.sum() / ok.sum()), 3), "평균 bp": round(float(np.nanmean(g[on])), 2),
            "CI": [round(float(np.nanquantile(gb, q)), 2) for q in (.025, .975)], "메이커 뒤": round(float(np.nanmean(g[on])) - 1.41, 2)}


VISION = "https://data.binance.vision/data/futures/um"
BK = Path.home() / "backups/crypto-scalping-server/data/lake"
OBR = Path("/home/kbj20/crypto-scalping/.claude/worktrees/limit-order-price-calc-e630d4/scripts")


def _get(url, dest):
    import urllib.request, zipfile
    if not dest.exists():
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception:                                                   # 아직 안 올라온 날 = 없음
            return None
    z = zipfile.ZipFile(dest); return pd.read_csv(z.open(z.namelist()[0]))


def extend():
    """09-14 ~ 어제(vision 에 있는 날) 패널·프리미엄·펀딩을 이어 붙이고 프로파일을 다시 만든다 → OUT/bv, OUT/prof, OUT/funding.csv.
    규약(09-14 288행 100% 일치 확인): 패널 행 시각 T ← klines open_time T · metrics create_time T."""
    raw = OUT / "raw"; raw.mkdir(parents=True, exist_ok=True); bv = OUT / "bv"; (bv / "panel").mkdir(parents=True, exist_ok=True)
    days = pd.date_range("2026-09-14", pd.Timestamp.utcnow().tz_localize(None).floor("D") - pd.Timedelta("1D"))
    last = None
    for sym in ("ETHUSDT", "BTCUSDT"):
        old = pd.read_parquet(D.BV / f"panel/{sym}.parquet")
        rows = []
        for d in days:
            ds = d.strftime("%Y-%m-%d")
            k = _get(f"{VISION}/daily/klines/{sym}/5m/{sym}-5m-{ds}.zip", raw / f"{sym}-5m-{ds}.zip")
            m = _get(f"{VISION}/daily/metrics/{sym}/{sym}-metrics-{ds}.zip", raw / f"{sym}-metrics-{ds}.zip")
            if k is None or m is None:
                break
            k = pd.DataFrame({"timestamp": pd.to_datetime(k.open_time, unit="ms"), "high": k.high, "low": k.low, "close": k.close,
                              "volume": k.volume, "trades": k["count"], "taker_buy_base": k.taker_buy_volume})
            m = m.assign(timestamp=pd.to_datetime(m.create_time))[["timestamp"] + D.METRICS[:1] + D.METRICS[1:]]
            rows.append(k.merge(m, on="timestamp", how="left")); last = ds
        new = pd.concat(rows)
        ov = old.merge(new, on="timestamp", suffixes=("", "_n"))
        assert len(ov) >= 288 and (abs(ov.close - ov.close_n) < 1e-9).all() and (abs(ov.sum_open_interest - ov.sum_open_interest_n).fillna(0) < 1e-6).all(), "이어 붙이기 규약 어긋남"
        pd.concat([old, new[~new.timestamp.isin(old.timestamp)]]).sort_values("timestamp").to_parquet(bv / f"panel/{sym}.parquet")
    pr = pd.read_parquet(D.BV / "eth_premium_5m.parquet"); parts = [pr]
    for d in days:
        ds = d.strftime("%Y-%m-%d")
        x = _get(f"{VISION}/daily/premiumIndexKlines/ETHUSDT/5m/ETHUSDT-5m-{ds}.zip", raw / f"prem-{ds}.zip")
        if x is None:
            break
        parts.append(pd.DataFrame({"open_time": x.open_time.astype(float), "close": x.close, "volume": 0.0, "tb": 0.0}))
    pd.concat(parts).drop_duplicates("open_time", keep="first").sort_values("open_time").to_parquet(bv / "eth_premium_5m.parquet")
    f = pd.read_csv(D.FUND, parse_dates=["timestamp"])
    sep = _get(f"{VISION}/monthly/fundingRate/ETHUSDT/ETHUSDT-fundingRate-2026-09.zip", raw / "funding-2026-09.zip")
    add = [pd.DataFrame({"timestamp": pd.to_datetime(sep.calc_time, unit="ms"), "last_funding_rate": sep.last_funding_rate})] if sep is not None else []
    import duckdb                                                            # 10월 = 백업 미러 1초 마크의 정산 직전 비율(월간 파일이 아직 없다)
    mk = duckdb.sql(f"select ts_ms, funding_rate, next_funding_ms from read_parquet('{BK}/binance/mark_1s/coin=ETH/*/*.parquet', hive_partitioning=false) "
                    f"where ts_ms >= epoch_ms(timestamp '2026-10-01')").df()
    if len(mk):
        g = mk.sort_values("ts_ms").groupby("next_funding_ms").funding_rate.last()
        add.append(pd.DataFrame({"timestamp": pd.to_datetime(g.index, unit="ms"), "last_funding_rate": g.to_numpy()}))
    fx = pd.concat([f] + add).drop_duplicates("timestamp").sort_values("timestamp")
    fx = fx[fx.timestamp <= pd.Timestamp(last) + pd.Timedelta("1D")]
    fx.to_csv(OUT / "funding.csv", index=False)
    # 프로파일 다시(연구 루프 그대로, 4.7년 전체)
    sys.path.insert(0, str(OBR))
    import research_eth_position_cluster_breakthrough_20261010 as C
    import research_eth_position_profile_trading_tests_20261009 as T
    p = pd.read_parquet(bv / "panel/ETHUSDT.parquet").sort_values("timestamp").reset_index(drop=True)
    p["oi"] = p.sum_open_interest.shift(1).ffill().bfill()
    t5 = p.timestamp.astype("int64").to_numpy() // 10**6
    snap = np.flatnonzero(((t5 + 300_000) % 3_600_000 == 0) & (np.arange(len(p)) > T.M))
    hist, tot = C.build(p.oi.to_numpy(float), p.volume.to_numpy(float), p.high.to_numpy(float), p.low.to_numpy(float),
                        p.close.to_numpy(float), snap, float(T.TAU), T.RHO, T.K, T.M, C.NB, C.BW)
    (OUT / "prof").mkdir(exist_ok=True)
    pd.DataFrame({"i5": snap, "t": t5[snap] + 300_000, "c": p.close.to_numpy()[snap], "tot1": tot[0], "tot7": tot[1]}).to_parquet(OUT / "prof/snap.parquet")
    np.save(OUT / "prof/hist.npy", hist.astype(np.float32))
    return last


def forward(status_only=False):
    fz = json.loads(FROZEN.read_text())
    last = extend()
    D.BV, D.PROF, D.FUND = OUT / "bv", OUT / "prof", OUT / "funding.csv"
    D.END = str((pd.Timestamp(last) + pd.Timedelta("1D")).date())
    Z, lch = D.level1()
    out = {"데이터 끝": last, "동결 해시": hashlib.sha256(FROZEN.read_bytes()).hexdigest()[:16]}
    for h, f in fz["horizons"].items():
        h = int(h)
        if status_only:
            out[f"{h}h"] = {"전진 시간": int(((Z.index >= FWD0) & np.isfinite((lch.shift(-h) - lch).to_numpy())).sum())}
            continue
        # 이어 붙인 데이터로 보조 구간 예측을 다시 만들어 원본과 대조(이어 붙이기·프로파일 재계산 트립와이어)
        out[f"{h}h"] = {"보조 재현(이어 붙인 데이터)": score(Z, lch, f, h, AUX), "전진 09-15~": score(Z, lch, f, h, (FWD0, D.END))}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    looks = OUT / "looks.json"
    hist_ = json.loads(looks.read_text()) if looks.exists() else []
    if not status_only:
        hist_.append({"at": pd.Timestamp.utcnow().isoformat(), "data_end": last})
        looks.write_text(json.dumps(hist_, ensure_ascii=False))
    print("열람 횟수", len(hist_))


if __name__ == "__main__":
    if "--select" in sys.argv:
        select()
    elif "--forward" in sys.argv or "--status" in sys.argv:
        forward(status_only="--status" in sys.argv)
    else:
        print(__doc__)
