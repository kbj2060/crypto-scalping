"""딥피쳐 탐색 (2026-10-16, 사용자 «원시 데이터 그 자체로 보지 말고 다른 데이터와 섞어서 파라미터를 조절해가면서 딥피쳐»).

결과 전 고정(사전등록) -- 매시 정각 t 결정(ETH 2022-04 ~ 2026-09-14), 정답 y_h = ln c[t+h] − ln c[t], h = 1·4·24 시간.
- 1단(원시 → z): 수익·실현변동·거래량·델타/거래량·ΔOI·롱숏비 3종 수준+변화·테이커 비율·프리미엄 수준+변화·펀딩·BTC 수익/델타/ΔOI·
  BTC 대비 상대 수익(30일 β)·코인베이스 프리미엄 수준+변화·포지션 프로파일(1·7일 위쪽 비중·현재가 밀도). 창 w = 1·4·24·72·168h.
  전부 직전 720h(30일) 평균·표준편차 z. metrics(OI·롱숏비) = 스냅샷 시각 기준(2024-03-04 부터 스탬프 T = T+5분).
- 2단(섞기): 잔차 = 흐름 z − β·같은 창 수익 z (720h 이동 회귀) · 곱 z_a·z_b · 엇갈림 |z_a|·1[부호 다름] · 조건부 z_a·1[|z_b|>1] (a·b 순서 둘 다).
  쌍 재료 = 4·24h 창 1단 + 수준형 · 같은 원천 쌍 제외.
- 고르기: 탐색 2022-04~2023 · 일 군집 t(일별 순위곱 평균/표준오차) |t| ≥ 3 · 2022·2023 두 해 IC 부호 같음 → 검증 2024 같은 부호 |t| ≥ 2.
  우연 보정 = 정답을 같은 달 안에서 날짜 단위로 섞은 가짜 정답 20벌로 같은 고르기 → 살아남는 개수 분포.
- 확인 2025-01 ~ 2026-09-14 한 번: 살아남은 것(지평별 |t| 상위 ≤50)으로 2022-04~2024 리지(α=10) → 확인 IC(일 블록 CI) · 1h 늦은 진입 IC ·
  상·하위 20%(학습 분위) 신호 평균 bp(총 / 메이커 왕복 1.41 / 테이커 8 뺀 값) · 1단만 같은 과정 대조.
- 통과: 진짜 생존 수 > 가짜 95분위 · 확인 IC CI 하한 > 0 · 메이커 비용 뒤 평균 CI 하한 > 0.
사용: python scripts/research_eth_deep_features_20261016.py [--selftest]
"""
import itertools, json, sys
from pathlib import Path

import numpy as np, pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
BV = ROOT / "data/binance_vision"
PROF = ROOT / ".claude/worktrees/limit-order-price-calc-e630d4/tmp/position_cluster_breakthrough_20261010"
CB = ROOT / ".claude/worktrees/medium-long-term-forecast-model-669354/tmp/h1h4_deep_20261015/cb1h_ETH-USD.parquet"
FUND = ROOT / "tmp/omega461_longwindow_20260917/funding_2021_2026.csv"
OUT = Path("tmp/deep_features_20261016")
SWITCH = pd.Timestamp("2024-03-04")
METRICS = ["sum_open_interest", "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
           "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
W = (1, 4, 24, 72, 168)
H = (1, 4, 24)
SEARCH, VAL, CONF = ("2022-04-01", "2024-01-01"), ("2024-01-01", "2025-01-01"), ("2025-01-01", "2026-09-15")
NNULL = 20
END = "2026-09-15"          # 데이터 끝(전진 채점이 늘린다)


def panel(sym):
    """5분 패널 → 행 = 봉 끝 시각. metrics 열은 스냅샷 시각으로 다시 맞춘다(봉 끝 = 그 봉 끝 스냅샷)."""
    p = pd.read_parquet(BV / f"panel/{sym}.parquet").sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    p = p.reindex(pd.date_range(p.index[0], p.index[-1], freq="5min"))
    snap = p.index + np.where(p.index >= SWITCH, pd.Timedelta("5min"), pd.Timedelta(0))
    end = p.index + pd.Timedelta("5min")
    for c in METRICS:
        s = pd.Series(p[c].to_numpy(), index=snap); s = s[~s.index.duplicated()]
        p[c] = s.reindex(end).to_numpy()
    p["d"] = 2 * p.taker_buy_base - p.volume
    p.index = end
    return p


def zs(x, n=720):
    m = x.rolling(n, min_periods=n // 2).mean(); s = x.rolling(n, min_periods=n // 2).std()
    return (x - m) / s.replace(0, np.nan)


def level1():
    e, b = panel("ETHUSDT"), panel("BTCUSDT")
    hr = e.index[(e.index.minute == 0)]
    hr = hr[(hr >= "2022-01-01") & (hr < END)]
    S = lambda s, w: s.rolling(w * 12, min_periods=w * 12).sum().reindex(hr)
    lc = np.log(e.close); lbc = np.log(b.close)
    r5 = lc.diff()
    pr = pd.read_parquet(BV / "eth_premium_5m.parquet")
    prem = pd.Series(pr.close.to_numpy() * 1e4, index=pd.to_datetime(pr.open_time, unit="ms") + pd.Timedelta("5min"))
    prem = prem[~prem.index.duplicated()].reindex(e.index)
    f = pd.read_csv(FUND, parse_dates=["timestamp"]); f = f.set_index(f.timestamp.dt.floor("min")).last_funding_rate
    f = f[~f.index.duplicated()]
    fund = f.reindex(f.index.union(hr)).ffill().reindex(hr) * 1e4
    cb = pd.read_parquet(CB).sort_values("t"); cb = pd.Series(cb.close.to_numpy(), index=pd.to_datetime(cb.t + 3600, unit="s"))
    cb = cb[~cb.index.duplicated()].reindex(hr)
    lch = lc.reindex(hr); lbch = lbc.reindex(hr)
    cbp = (np.log(cb) - lch) * 1e4
    X = {}
    RAT = (("sum_toptrader_long_short_ratio", "tpl"), ("count_toptrader_long_short_ratio", "tal"),
           ("count_long_short_ratio", "gl"), ("sum_taker_long_short_vol_ratio", "tk"))
    for w in W:
        X[f"ret_{w}"] = lch - lch.shift(w)
        X[f"rv_{w}"] = np.sqrt(S(r5 ** 2, w))
        X[f"vol_{w}"] = np.log(S(e.volume, w))
        X[f"dlt_{w}"] = S(e.d, w) / S(e.volume, w)
        X[f"doi_{w}"] = np.log(e.sum_open_interest.reindex(hr)) - np.log(e.sum_open_interest.reindex(hr)).shift(w)
        X[f"prem_{w}"] = prem.rolling(w * 12, min_periods=w * 6).mean().reindex(hr)
        X[f"bret_{w}"] = lbch - lbch.shift(w)
        X[f"bdlt_{w}"] = S(b.d, w) / S(b.volume, w)
        X[f"bdoi_{w}"] = np.log(b.sum_open_interest.reindex(hr)) - np.log(b.sum_open_interest.reindex(hr)).shift(w)
        X[f"cbp_{w}"] = cbp.rolling(w, min_periods=max(1, w // 2)).mean()
        for c, nm in RAT:
            lv = np.log(e[c].reindex(hr))
            X[f"d{nm}_{w}"] = lv - lv.shift(w)
    for c, nm in RAT:
        X[f"L{nm}"] = np.log(e[c].reindex(hr))
    X["Lprem"] = prem.reindex(hr); X["Lfund"] = fund; X["Lcbp"] = cbp
    for w in W:
        beta = X[f"ret_{w}"].rolling(720, min_periods=360).cov(X[f"bret_{w}"]) / X[f"bret_{w}"].rolling(720, min_periods=360).var()
        X[f"rel_{w}"] = X[f"ret_{w}"] - beta.shift(1) * X[f"bret_{w}"]
    Sn = pd.read_parquet(PROF / "snap.parquet"); Hh = np.load(PROF / "hist.npy", mmap_mode="r")
    st = pd.to_datetime(Sn.t, unit="ms")
    for wi, nm in ((0, "p1"), (1, "p7")):
        h = np.asarray(Hh[wi]).astype(float); tot = h.sum(1)
        X[f"L{nm}up"] = pd.Series(h[:, 30:].sum(1) / np.where(tot > 0, tot, np.nan), index=st).reindex(hr)
        X[f"L{nm}den"] = pd.Series(h[:, 28:32].mean(1) / np.maximum(np.median(h, 1), 1e-9), index=st).reindex(hr)
    Z = pd.DataFrame({k: zs(v.astype(float)) for k, v in X.items()}, index=hr)
    return Z, lch


def src(name):
    return name.split("_")[0].lstrip("L").replace("up", "").replace("den", "")


FLOW = ("dlt", "doi", "prem", "cbp", "bdlt", "bdoi", "dtpl", "dtal", "dgl", "dtk", "vol")


def level2_specs(cols):
    """(이름, 종류, a, b) 목록."""
    specs = [(f"res[{c}]", "res", c, f"ret_{c.split('_')[1]}") for c in cols if "_" in c and c.split("_")[0] in FLOW]
    base = [c for c in cols if c.startswith("L") or c.endswith("_4") or c.endswith("_24")]
    for a, b in itertools.combinations(base, 2):
        if src(a) == src(b):
            continue
        specs += [(f"{a}*{b}", "mul", a, b), (f"{a}≠{b}", "div", a, b), (f"{a}|{b}", "cond", a, b), (f"{b}|{a}", "cond", b, a)]
    return specs


def make(Z, kind, a, b):
    x, y = Z[a].to_numpy(), Z[b].to_numpy()
    if kind == "mul":
        return x * y
    if kind == "div":
        return np.abs(x) * (np.sign(x) != np.sign(y))
    if kind == "cond":
        return np.where(np.abs(y) > 1, x, 0.0) * np.where(np.isfinite(y), 1.0, np.nan)
    xs, ys = pd.Series(x), pd.Series(y)                                      # 잔차: 720h 이동 β
    beta = xs.rolling(720, min_periods=360).cov(ys) / ys.rolling(720, min_periods=360).var()
    return (xs - beta * ys).to_numpy()


def feat(Z, spec):
    _, kind, a, b = spec
    return Z[a].to_numpy() if kind == "raw" else make(Z, kind, a, b)


def rank_cols(M):
    R = pd.DataFrame(M).rank().to_numpy()
    mu = np.nanmean(R, 0); sd = np.nanstd(R, 0)
    return np.nan_to_num((R - mu) / np.where(sd > 0, sd, np.nan), nan=0.0)


def rank_vec(v):
    r = pd.Series(v).rank().to_numpy()
    return np.nan_to_num((r - np.nanmean(r)) / np.nanstd(r), nan=0.0)


def day_t(R, ry, bounds):
    """일 군집 t(시간순 정렬 가정): 일별 평균 순위곱 → 평균/표준오차 · IC ≈ 전체 평균 순위곱."""
    P = R * ry[:, None]
    sums = np.add.reduceat(P, bounds, axis=0)
    cnt = np.diff(np.append(bounds, len(P)))[:, None]
    dm = sums / cnt
    return P.mean(0), dm.mean(0) / (dm.std(0, ddof=1) / np.sqrt(len(dm)))


def shuffle_days(y, day, month, rng):
    """같은 달 안에서 날짜 단위로 정답을 섞는다."""
    out = y.copy()
    for mth in np.unique(month):
        ds = np.unique(day[month == mth]); perm = rng.permutation(ds)
        for d, e in zip(ds, perm):
            a, b = np.flatnonzero(day == d), np.flatnonzero(day == e)
            k = min(len(a), len(b)); out[a[:k]] = y[b[:k]]
    return out


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    Z, lch = level1()
    t = Z.index
    day = t.floor("D").asi8; month = t.to_period("M").astype(str).to_numpy()
    masks = {"a": (t >= SEARCH[0]) & (t < "2023-01-01"), "b": (t >= "2023-01-01") & (t < SEARCH[1]),
             "s": (t >= SEARCH[0]) & (t < SEARCH[1]), "v": (t >= VAL[0]) & (t < VAL[1])}
    m_c = (t >= CONF[0]) & (t < CONF[1])
    bnd = {k: np.flatnonzero(np.diff(day[m], prepend=day[m][0] - 1)) for k, m in masks.items()}
    Y = {h: (lch.shift(-h) - lch).to_numpy() * 1e4 for h in H}
    specs = [(c, "raw", c, None) for c in Z.columns] + level2_specs(list(Z.columns))
    print("피쳐", len(specs), "(1단", Z.shape[1], ")", flush=True)
    rng = np.random.default_rng(20261016)
    yr = {h: [{k: rank_vec(Y[h][m]) for k, m in masks.items()}] + [
              {k: rank_vec(yn[m]) for k, m in masks.items()} for yn in (shuffle_days(Y[h], day, month, rng) for _ in range(NNULL))] for h in H}
    acc = {h: {"ic": {k: [] for k in masks}, "t": {k: [] for k in masks}} for h in H}
    null_ok = {h: [] for h in H}
    CH = 300
    for k0 in range(0, len(specs), CH):
        M = np.column_stack([feat(Z, s) for s in specs[k0:k0 + CH]]).astype(np.float64)
        R = {k: rank_cols(M[m]) for k, m in masks.items()}
        for h in H:
            oks = []
            for j, ys in enumerate(yr[h]):
                st = {k: day_t(R[k], ys[k], bnd[k]) for k in masks}
                ok = (np.abs(st["s"][1]) >= 3) & (np.sign(st["a"][0]) == np.sign(st["b"][0])) & \
                     (np.sign(st["v"][0]) == np.sign(st["s"][1])) & (np.abs(st["v"][1]) >= 2)
                if j == 0:
                    for k in masks:
                        acc[h]["ic"][k] += list(st[k][0]); acc[h]["t"][k] += list(st[k][1])
                else:
                    oks.append(ok)
            null_ok[h].append(np.array(oks))
        print(f"  {min(k0 + CH, len(specs))}/{len(specs)}", flush=True)
    names = [s[0] for s in specs]; is_raw = np.array([s[1] == "raw" for s in specs])
    res = {"n_features": len(specs), "n_level1": int(is_raw.sum())}
    from sklearn.linear_model import Ridge
    idx_c0 = int(np.argmax(m_c))
    for h in H:
        ic = {k: np.array(v) for k, v in acc[h]["ic"].items()}; tt = {k: np.array(v) for k, v in acc[h]["t"].items()}
        surv = (np.abs(tt["s"]) >= 3) & (np.sign(ic["a"]) == np.sign(ic["b"])) & (np.sign(ic["v"]) == np.sign(tt["s"])) & (np.abs(tt["v"]) >= 2)
        nk = np.concatenate(null_ok[h], 1)
        row = {"생존(전체)": int(surv.sum()), "가짜 생존 중앙/95분위": [float(np.median(nk.sum(1))), float(np.quantile(nk.sum(1), .95))],
               "생존(1단만)": int((surv & is_raw).sum()), "가짜 1단 중앙/95분위": [float(np.median(nk[:, is_raw].sum(1))), float(np.quantile(nk[:, is_raw].sum(1), .95))]}
        top = np.flatnonzero(surv); top = top[np.argsort(-np.abs(tt["s"][top]))][:50]
        row["상위 생존(이름, IC22, IC23, IC24, t탐색)"] = [(names[i], round(float(ic["a"][i]), 4), round(float(ic["b"][i]), 4), round(float(ic["v"][i]), 4), round(float(tt["s"][i]), 1)) for i in top[:15]]
        for lab, sel in (("딥(전체 생존)", top), ("1단만 생존", np.array([i for i in top if is_raw[i]], dtype=int))):
            if len(sel) == 0:
                row[lab] = "생존 없음"; continue
            M = np.column_stack([feat(Z, specs[i]) for i in sel])
            tr = (t >= SEARCH[0]) & (np.arange(len(t)) < idx_c0 - h) & np.isfinite(Y[h])
            mu, sd = np.nanmean(M[tr], 0), np.nanstd(M[tr], 0); sd[sd == 0] = 1
            Zm = np.clip(np.nan_to_num((M - mu) / sd), -5, 5)
            mdl = Ridge(alpha=10.0).fit(Zm[tr], Y[h][tr])
            ptr, pc = mdl.predict(Zm[tr]), mdl.predict(Zm[m_c])
            yc = Y[h][m_c]; ycd = pd.Series(Y[h]).shift(-1).to_numpy()[m_c]; dc = day[m_c]; okc = np.isfinite(yc)
            sp = lambda a, b, s: float(pd.Series(a[s]).corr(pd.Series(b[s]), method="spearman"))
            ud = np.unique(dc[okc]); ix = {u: np.flatnonzero((dc == u) & okc) for u in ud}; rb = np.random.default_rng(1)
            draws = [np.concatenate([ix[u] for u in rb.choice(ud, len(ud))]) for _ in range(500)]
            lo, hi = np.quantile(ptr, [.2, .8]); sig = np.where(pc >= hi, 1.0, np.where(pc <= lo, -1.0, 0.0))
            g = sig * yc; on = (sig != 0) & okc
            gb = [np.nanmean(g[s][on[s]]) for s in draws]
            row[lab] = {"피쳐 수": int(len(sel)), "확인 IC": round(sp(pc, yc, okc), 4),
                        "IC CI": [round(float(np.quantile([sp(pc, yc, s) for s in draws], q)), 4) for q in (.025, .975)],
                        "1h 늦은 IC": round(sp(pc, ycd, okc & np.isfinite(ycd)), 4),
                        "신호 비율": round(float(on.mean()), 3), "총 bp": round(float(np.nanmean(g[on])), 2),
                        "CI": [round(float(np.quantile(gb, q)), 2) for q in (.025, .975)],
                        "메이커 뒤": round(float(np.nanmean(g[on])) - 1.41, 2), "테이커 뒤": round(float(np.nanmean(g[on])) - 8, 2)}
        res[f"{h}h"] = row
        print(h, json.dumps(row, ensure_ascii=False), flush=True)
    (OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def selftest():
    Zt = pd.DataFrame({"a": [1., -1, 2, -2], "b": [1., 1, -2, 0.5]})
    assert list(make(Zt, "mul", "a", "b")) == [1, -1, -4, -1]
    assert list(make(Zt, "div", "a", "b")) == [0, 1, 2, 2]
    assert list(make(Zt, "cond", "a", "b")) == [0, 0, 2, 0]
    day = np.repeat([1, 2], 5); month = np.array(["m"] * 10); y = np.arange(10.)
    ys = shuffle_days(y, day, month, np.random.default_rng(0))
    assert sorted(ys) == list(y) and set(ys[:5]) in ({0., 1, 2, 3, 4}, {5., 6, 7, 8, 9})
    R = np.array([[1.], [-1], [1], [-1]]); ry = np.array([1., -1, 1, -1])
    ic, tv = day_t(R, ry, np.array([0, 2]))
    assert abs(ic[0] - 1) < 1e-12
    print("selftest ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else run()
