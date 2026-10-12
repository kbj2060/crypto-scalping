"""향하는 쪽 가까운 청산벽을 «뚫는» 조건 (2026-10-13).
사용자 «현재 가격이 향하는 방향으로 가까운 청산벽을 뚫는 조건들을 4년치 데이터로 분석 · 실시간 데이터와 정리된 피쳐들로 연구».

    python3 scripts/research_eth_liq_wall_break_conditions_20261013.py --build     # 사건·라벨·피쳐 → tmp/liq_wall_break_20261013/events.parquet
    python3 scripts/research_eth_liq_wall_break_conditions_20261013.py --test      # 조건표·모델·위약 → report.json
    python3 scripts/research_eth_liq_wall_break_conditions_20261013.py --rt        # 실측 청산(07-19~) 조건
    python3 scripts/research_eth_liq_wall_break_conditions_20261013.py --selftest

벽 = 대시보드 풋프린트 «보이는» 청산 띠(liqBandEdges 임계 0.25) -- 10-12 research_eth_liq_band_sweep_upgrade 의 edges_long 캐시
(브랜치 claude/footprint-liquidation-distance-indicator-4a9790, V1.lit_edges). 행 t = 5분봉 시작, 값은 그 봉 끝(종가 기준)에 앎.
벽 질량은 live_liquidation_map._window_at + _spliced_raw 를 사건 시각에만 다시 호출(식 두 벌 금지).

사전등록(결과 보기 전 고정)
- 사건(위, 아래는 거울): 5분봉 k 의 고가 ≥ E = 봉 k−1 끝 지도의 위 가장자리, 그리고 종가[k−1] < E(벽 밖에서 다가와 닿음).
  같은 쪽 다음 사건은 12봉(1시간) 뒤. 한 봉이 양쪽을 다 닿으면 버림. 방향 s = 닿은 쪽(+1 위 / −1 아래) = «가격이 향하는 쪽».
- 결정 시각 = 봉 k 끝. 피쳐는 봉 k 까지(OI·metrics 는 스탬프 규약 때문에 k−1), 라벨은 그 다음 1분부터(사건 라벨 경계 계약).
- 라벨(뚫음): 반폭 H = (띠 끝 F − E) + 0.1%·E. 뚫음 = 1분 고/저가가 E + s·H(= 띠 끝 + 0.1%) 먼저 · 튕김 = E − s·H 먼저 · 4시간.
  같은 분 둘 다 = 애매(버림) · 4시간 안 둘 다 없음 = 미결(버림, 비율 보고). 결정 시각 종가가 이미 배리어 밖이면 «봉 안에서 결정»(버림, 비율 보고).
- 기하 기준선 p0 = (종가[k] − 튕김선)/(뚫음선 − 튕김선) = 드리프트 없는 가격의 뚫음 확률. 모든 조건은 잔차 y − p0 로 읽는다
  (09-11 교훈: 두 배리어 first-touch 는 거리비가 AUC 를 채운다).
- 위약 벽 = 7일(2016봉) 전 같은 봉의 «가장자리·띠 끝까지 상대 거리»를 지금 가격에 얹은 선. 같은 사건·라벨·피쳐(벽 질량 제외).
  ⇒ 조건이 위약에서도 같으면 «청산벽»의 성질이 아니라 아무 선에서나 통하는 모멘텀/변동성 성질.
- 구간: DEV 2022-01~2024-12 · VAL 2025 · TEST 2026-01~10-10. 주 판정 = RECENT(VAL∪TEST, 최근 우선).
- 조건 판정(피쳐마다): DEV 5분위 경계로 나눈 Q5−Q1 잔차 차. 통과 = RECENT 일 블록 부트 95% CI 가 0 제외 ∧ DEV 같은 부호 ∧ |차| ≥ 2pp.
  피쳐 ~50개 → 우연 통과 ~2.5개 기대, 그래서 본페로니 |z| ≥ 3.3 도 같이 표시. 벽 특이성 = (실제 − 위약) 차의 CI.
  aggfeat·bookDepth(2024-01~2026-08-22) 피쳐는 DEV = 2024 만.
- 모델: HGB 고정 하이퍼(학습 DEV) GEO(기하 5) vs ALL(전 피쳐) → VAL·TEST AUC·로그손실·정확도, ΔAUC 일 블록 CI. 위약에도 같은 것.
  🔴목적 정확도 기준(10-10 사용자): 판정은 «뚫음 예측 정확도가 기하를 넘나». 손익은 참고 한 줄.
플래그: fresh_forward_bar_by_bar=true · trade_ledgers_used_as_input=false · saved_parent_exit_timestamps_used=false · future_rows_used_for_entry=false
"""
import glob, json, os, sys, zipfile
import numpy as np, pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_liquidation_map_20260824 as LM  # noqa: E402

BV = "/home/kbj20/crypto-scalping/data/binance_vision"
OUT = os.path.join(HERE, "..", "tmp", "liq_wall_break_20261013")
TAIL = ("/tmp/claude-1000/-home-kbj20-crypto-scalping--claude-worktrees-footprint-liquidation-distance-indicator-4a9790/"
        "ec85a659-949f-4e97-8b36-2b9664158c76/scratchpad/sweep/eth_tail_risk_1m.parquet")   # 서버 lake tail_risk_1m 사본(~10-10)
M1, M5 = 60_000, 300_000
DL = ("/tmp/claude-1000/-home-kbj20-crypto-scalping--claude-worktrees-footprint-liquidation-distance-indicator-4a9790/"
      "ec85a659-949f-4e97-8b36-2b9664158c76/scratchpad/sweep/dl")                                  # vision 일별 zip 09-15~10-10 (로컬 공백 메움)
END = "2026-10-11"
COOL, HOLD, MARGIN, PL_LAG = 12, 240, 0.001, 2016
SPLITS = (("DEV", "2022-01-01", "2025-01-01"), ("VAL", "2025-01-01", "2026-01-01"), ("TEST", "2026-01-01", END))
GEO = ["p0", "h_sig", "band_sig", "over_sig", "s"]
MASS = ["wall_share", "behind_share", "opp_vs_wall1"]


# ── 데이터 ─────────────────────────────────────────────────────────────
def load_1m():
    fs = sorted(glob.glob(f"{BV}/klines1m/ETHUSDT-1m-*.parquet")) + sorted(glob.glob(f"{BV}/klines1m_daily/ETHUSDT-1m-*.parquet"))
    parts = [pd.read_parquet(f, columns=["t", "h", "l", "c", "v", "tb"]) for f in fs]
    for z in sorted(glob.glob(f"{DL}/ETHUSDT-1m-*.zip")):
        with zipfile.ZipFile(z) as zf:
            d = pd.read_csv(zf.open(zf.namelist()[0])).iloc[:, [0, 2, 3, 4, 5, 9]]
        d.columns = ["t", "h", "l", "c", "v", "tb"]
        parts.append(d.astype({"t": "int64"}))
    k = pd.concat(parts).drop_duplicates("t").sort_values("t")
    k = k[(k.t >= pd.Timestamp("2021-12-29").value // 10**6) & (k.t < pd.Timestamp(END).value // 10**6)]
    full = pd.DataFrame({"t": np.arange(k.t.iloc[0], k.t.iloc[-1] + 1, M1)}).merge(k, on="t", how="left")
    full["c"] = full.c.ffill()
    for col in ("h", "l"):
        full[col] = full[col].fillna(full.c)
    full[["v", "tb"]] = full[["v", "tb"]].fillna(0.0)
    return full.reset_index(drop=True)


def bars(k, ms):
    g = k.groupby(k.t // ms * ms)
    b = pd.DataFrame({"high": g.h.max(), "low": g.l.min(), "close": g.c.last(), "volume": g.v.sum(), "tb": g.tb.sum()}).reset_index(names="t")
    b["timestamp"] = pd.to_datetime(b.t, unit="ms", utc=True)
    return b


def daily_parquets(pattern):
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(pattern))])


# ── 라벨 ───────────────────────────────────────────────────────────────
@njit(cache=True)
def first_touch(h, l, j0, up, dn, hold):
    """j0 = 결정 뒤 첫 1분 인덱스. 위선 up·아래선 dn 중 먼저 닿은 쪽: 1 위 · 0 아래 · −1 같은 분(애매) · −2 미결."""
    for j in range(j0, min(len(h), j0 + hold)):
        a, b = h[j] >= up, l[j] <= dn
        if a and b:
            return -1
        if a:
            return 1
        if b:
            return 0
    return -2


# ── 사건 ───────────────────────────────────────────────────────────────
def detect(b5, up_e, up_f, dn_e, dn_f):
    """반환 DataFrame(k, s, E, F) -- 인자 배열은 이미 봉 k−1 끝 지도로 민 값."""
    c_prev = b5.close.shift(1).to_numpy()
    hit_up = (b5.high.to_numpy() >= up_e) & (c_prev < up_e)
    hit_dn = (b5.low.to_numpy() <= dn_e) & (c_prev > dn_e)
    rows, last = [], {1: -10**9, -1: -10**9}
    for k in np.flatnonzero(hit_up ^ hit_dn):
        s = 1 if hit_up[k] else -1
        if k - last[s] < COOL:
            continue
        last[s] = k
        rows.append((k, s, up_e[k] if s > 0 else dn_e[k], up_f[k] if s > 0 else dn_f[k]))
    return pd.DataFrame(rows, columns=["k", "s", "E", "F"])


def label(ev, b5, k1):
    """1분 경로로 뚫음/튕김. 결정 = 봉 k 끝 → 첫 분 = 그 끝 시각의 1분."""
    t0 = int(k1.t.iloc[0])
    h, l = k1.h.to_numpy(), k1.l.to_numpy()
    c = b5.close.to_numpy()[ev.k]
    H = np.abs(ev.F - ev.E) + MARGIN * ev.E
    thr, back = ev.E + ev.s * H, ev.E - ev.s * H
    j0 = ((b5.t.to_numpy()[ev.k] + M5 - t0) // M1).astype(np.int64)
    up, dn = np.where(ev.s > 0, thr, back), np.where(ev.s > 0, back, thr)
    decided = (c >= up) | (c <= dn)
    res = np.array([-3 if dd else first_touch(h, l, int(j), float(u), float(d), HOLD) for j, u, d, dd in zip(j0, up, dn, decided)])
    y = np.where(res >= 0, np.where(ev.s > 0, res, 1 - res), np.nan)        # 1 = 뚫음(향하는 쪽 먼저)
    return ev.assign(c=c, H=H, thr=thr, back=back, res=res, y=y, p0=np.clip((c - back) / (thr - back), 0, 1),
                     gain_bp=np.abs(thr - c) / c * 1e4, loss_bp=np.abs(c - back) / c * 1e4)


# ── 피쳐 ───────────────────────────────────────────────────────────────
SIGNED = ["r15", "r30", "r1h", "r4h", "r24h", "r7d", "consist1h", "dlt", "dlt1h", "dlt4h", "pos24", "clv",
          "tt_chg1h", "crowd", "taker_ls", "prem", "prem_chg1h", "spot_dlt1h", "btc_r30", "btc_r1h", "rel1h",
          "cvd", "bigimb", "smlimb", "cntimb", "cvd30", "bigimb30"]


def base_frame(b5):
    """5분 격자 위 피쳐(값 = 봉 끝에 앎). SIGNED 는 사건에서 s 를 곱해 «향하는 쪽 = +»로 맞춘다."""
    f = pd.DataFrame(index=b5.index)
    lc = np.log(b5.close)
    r = lc.diff()
    f["sig"] = r.rolling(288, min_periods=200).std()
    for n, nm in ((3, "r15"), (6, "r30"), (12, "r1h"), (48, "r4h"), (288, "r24h"), (2016, "r7d")):
        f[nm] = (lc - lc.shift(n)) / (f.sig * np.sqrt(n))
    f["consist1h"] = np.sign(r).rolling(12).sum() / 12
    f["vratio"] = r.rolling(12).std() / f.sig
    f["rng_sig"] = np.log(b5.high / b5.low) / f.sig
    f["sig_bp"] = f.sig * 1e4
    v = b5.volume
    vmed = v.rolling(288, min_periods=200).median()
    f["rvol"] = np.log((v + 1) / (vmed + 1))
    f["rvol1h"] = np.log((v.rolling(12).mean() + 1) / (vmed + 1))
    d = 2 * b5.tb - v
    f["dlt"] = d / v.replace(0, np.nan)
    f["dlt1h"] = d.rolling(12).sum() / v.rolling(12).sum()
    f["dlt4h"] = d.rolling(48).sum() / v.rolling(48).sum()
    hi, lo = b5.high.rolling(288).max(), b5.low.rolling(288).min()
    f["pos24"] = (b5.close - lo) / (hi - lo) * 2 - 1                         # −1 24h 저점 ~ +1 고점
    f["clv"] = (b5.close - b5.low) / (b5.high - b5.low).replace(0, np.nan) * 2 - 1
    ts = pd.to_datetime(b5.t, unit="ms", utc=True)
    f["hour"], f["dow"] = ts.dt.hour, ts.dt.dayofweek
    return f


def join_panels(b5, f):
    p = pd.concat([pd.read_parquet(f"{BV}/panel/ETHUSDT.parquet"), daily_parquets(f"{BV}/panel_daily/ETHUSDT-5m-*.parquet")])
    p["t"] = pd.to_datetime(p.timestamp).astype("int64") // 10**6
    p = b5[["t"]].merge(p.drop_duplicates("t"), on="t", how="left")
    lag = lambda x: x.shift(1)                                               # OI·metrics 는 k−1 (스탬프 규약 2024-03-04 전환)
    oi = np.log(lag(p.sum_open_interest))
    for n, nm in ((3, "doi15"), (12, "doi1h"), (48, "doi4h"), (288, "doi24h")):
        f[nm] = oi - oi.shift(n)
    tt = np.log(lag(p.sum_toptrader_long_short_ratio))
    f["tt_chg1h"] = tt - tt.shift(12)
    f["crowd"] = np.log(lag(p.count_long_short_ratio))
    f["taker_ls"] = np.log(lag(p.sum_taker_long_short_vol_ratio))
    tr = p.trades
    f["rtrades"] = np.log((tr + 1) / (tr.rolling(288, min_periods=200).median() + 1))
    sz = b5.volume / tr.replace(0, np.nan)
    f["tsize"] = np.log(sz / sz.rolling(288, min_periods=200).median())
    pr = pd.read_parquet(f"{BV}/eth_premium_5m.parquet").assign(t=lambda x: x.open_time.astype("int64"))
    pr = b5[["t"]].merge(pr[["t", "close"]].drop_duplicates("t"), on="t", how="left").close
    f["prem"], f["prem_chg1h"] = pr * 1e4, (pr - pr.shift(12)) * 1e4
    sp = pd.read_parquet(f"{BV}/eth_spot_5m.parquet").assign(t=lambda x: x.open_time.astype("int64"))
    sp = b5[["t"]].merge(sp[["t", "volume", "tb"]].drop_duplicates("t"), on="t", how="left")
    f["spot_share1h"] = sp.volume.rolling(12).sum() / (sp.volume.rolling(12).sum() + b5.volume.rolling(12).sum())
    f["spot_dlt1h"] = (2 * sp.tb - sp.volume).rolling(12).sum() / sp.volume.rolling(12).sum()
    bt = pd.read_parquet(f"{BV}/panel/BTCUSDT.parquet")
    bt["t"] = pd.to_datetime(bt.timestamp).astype("int64") // 10**6
    bl = np.log(b5[["t"]].merge(bt[["t", "close"]].drop_duplicates("t"), on="t", how="left").close)
    bsig = bl.diff().rolling(288, min_periods=200).std()
    f["btc_r30"] = (bl - bl.shift(6)) / (bsig * np.sqrt(6))
    f["btc_r1h"] = (bl - bl.shift(12)) / (bsig * np.sqrt(12))
    f["rel1h"] = f.r1h - f.btc_r1h
    return f


def join_tape(b5, f):
    a = daily_parquets(f"{BV}/aggfeat/ETHUSDT_*.parquet").reset_index()
    a["t"] = pd.to_datetime(a.ts).astype("int64") // 10**6
    a = b5[["t"]].merge(a.drop(columns="ts").drop_duplicates("t"), on="t", how="left")
    f["cvd"], f["bigimb"], f["smlimb"], f["cntimb"] = a.tf_cvd, a.tf_bigimb, a.tf_smlimb, a.tf_cntimb
    f["cvd30"], f["bigimb30"] = a.tf_cvd.rolling(6).mean(), a.tf_bigimb.rolling(6).mean()
    f["bigshare"], f["kyle"], f["vpin"], f["signac"], f["runlen"] = a.tf_bigshare, np.log(a.tf_kyle.clip(lower=1e-6)), a.tf_vpin, a.tf_signac, a.tf_runlen
    return f


def join_book(b5, ev):
    """봉 k 끝 1초 전까지의 마지막 bookDepth 스냅샷(±1·2% 누적 명목). 위 사건이면 벽 쪽 = 매도호가(n1·n2)."""
    bk = daily_parquets(f"{BV}/ETHUSDT_bookDepth_*.parquet").sort_index()
    bt = bk.index.values.astype("datetime64[ms]").astype("int64")
    tq = b5.t.to_numpy()[ev.k] + M5 - 1000
    i = np.searchsorted(bt, tq, side="right") - 1
    i30 = np.searchsorted(bt, tq - 30 * M1, side="right") - 1
    ok = (i >= 0) & (i30 >= 0) & (tq - bt[np.clip(i, 0, None)] <= 2 * M1)
    g = lambda col, ii: np.where(ok, bk[col].to_numpy()[np.clip(ii, 0, None)], np.nan)
    up = ev.s.to_numpy() > 0
    a1, b1, a2, b2, a1o, b1o = g("n1", i), g("n-1", i), g("n2", i), g("n-2", i), g("n1", i30), g("n-1", i30)
    wall1 = np.where(up, a1, b1)
    ev["book_opp_wall1"] = np.log(np.where(up, b1, a1) / wall1)              # + = 벽 쪽 호가가 얇다
    ev["book_opp_wall2"] = np.log(np.where(up, b2, a2) / np.where(up, a2, b2))
    ev["book_wall_chg30"] = np.log(wall1 / np.where(up, a1o, b1o))          # − = 벽 쪽 호가가 30분 사이 빠짐
    ev["book_tot1"] = np.log(a1 + b1)
    return ev


def wall_mass(ev, b5, b1h):
    """봉 k−1 끝 지도(_window_at + _spliced_raw)에서 띠 몫·띠 뒤 1% 몫·반대쪽 1% 대비 (몫 = 그 쪽 5% 안 무게 대비)."""
    h, m = LM._bar_arrays(b1h), LM._bar_arrays(b5)
    out = np.full((len(ev), 3), np.nan)
    for n, (k, s, E, F) in enumerate(zip(ev.k, ev.s, ev.E, ev.F)):
        cp = float(b5.close.iat[k - 1])
        raw = LM._spliced_raw(LM._window_at(h, m, int(b5.t.iat[k]) // 1000, 24), cp)
        if raw is None:
            continue
        sup, res, bw = raw
        side, opp = (res, sup) if s > 0 else (sup, res)
        if not side:
            continue
        px, w = np.array(list(side.keys())) * bw, np.array(list(side.values()))
        tot = w[np.abs(px / cp - 1) <= 0.05].sum()
        if tot <= 0:
            continue
        out[n, 0] = w[(px >= min(E, F) - bw / 2) & (px <= max(E, F) + bw / 2)].sum() / tot
        beyond = (px - F) * s
        out[n, 1] = w[(beyond > bw / 2) & (beyond <= 0.01 * cp)].sum() / tot
        w1 = w[np.abs(px / cp - 1) <= 0.01].sum()
        o1 = sum(v for b, v in opp.items() if abs(b * bw / cp - 1) <= 0.01) if opp else 0.0
        out[n, 2] = np.log((o1 + 1e-6 * tot) / (w1 + 1e-6 * tot))
    ev["wall_share"], ev["behind_share"], ev["opp_vs_wall1"] = out.T
    return ev


def finish(ev, b5, f):
    ev = ev.reset_index(drop=True)
    k = ev.k.to_numpy()
    for col in f.columns:
        ev[col] = f[col].to_numpy()[k]
    for col in SIGNED:
        ev[col] = ev[col] * ev.s
    cs = ev.sig.to_numpy()
    ev["h_sig"] = np.log1p(ev.H / ev.E) / cs
    ev["band_sig"] = np.log1p(np.abs(ev.F - ev.E) / ev.E) / cs
    ev["over_sig"] = ev.s * np.log(ev.c / ev.E) / cs
    hi, lo, c, cp = b5.high.to_numpy()[k], b5.low.to_numpy()[k], b5.close.to_numpy()[k], b5.close.to_numpy()[k - 1]
    ev["wick"] = np.where(ev.s > 0, hi - np.maximum(c, cp), np.minimum(c, cp) - lo) / np.where(hi > lo, hi - lo, np.nan)
    out = np.zeros(len(ev))                                                  # 같은 쪽 직전 24h 사건 수(재터치)
    for s in (1, -1):
        idx = np.flatnonzero(ev.s.to_numpy() == s)
        kk = k[idx]
        out[idx] = np.arange(len(kk)) - np.searchsorted(kk, kk - 288, side="left")
    ev["touches24"] = out
    ev["t"] = b5.t.to_numpy()[k]
    ev["day"] = pd.to_datetime(ev.t, unit="ms").dt.floor("D")
    return ev


def build():
    os.makedirs(OUT, exist_ok=True)
    k1 = load_1m()
    b5, b1h = bars(k1, M5), bars(k1, 3_600_000)
    b5 = b5.merge(pd.read_parquet(f"{OUT}/edges_long.parquet"), on="t", how="left")
    f = join_tape(b5, join_panels(b5, base_frame(b5)))
    cl = b5.close.to_numpy()
    sh = lambda a, n=1: np.concatenate([np.full(n, np.nan), a[:-n]])
    real = {c: sh(b5[c].to_numpy()) for c in ("up_edge", "up_top", "dn_edge", "dn_bottom")}
    plac = {c: sh(b5[c].to_numpy() / cl, PL_LAG + 1) * sh(cl) for c in real}  # 7일 전 상대 거리 × 지금(k−1) 종가
    out = []
    for kind, ed in (("real", real), ("placebo", plac)):
        ev = detect(b5, ed["up_edge"], ed["up_top"], ed["dn_edge"], ed["dn_bottom"])
        ev = ev[(ev.k > PL_LAG + 100) & ev.E.notna() & ev.F.notna()]
        ev = finish(label(ev, b5, k1), b5, f)
        ev = join_book(b5, ev)
        if kind == "real":
            ev = wall_mass(ev, b5, b1h)
        ev["kind"] = kind
        out.append(ev)
        print(kind, len(ev), {int(a): round(b, 3) for a, b in ev.res.value_counts(normalize=True).items()}, flush=True)
    ev = pd.concat(out, ignore_index=True)
    ev.to_parquet(f"{OUT}/events.parquet")
    print("saved", ev.shape)


# ── 검정 ───────────────────────────────────────────────────────────────
NONFEAT = {"k", "s", "E", "F", "c", "H", "thr", "back", "res", "y", "p0", "gain_bp", "loss_bp", "t", "day", "kind", "sig", "split", "r"}


def split_of(t):
    ts = pd.to_datetime(t, unit="ms")
    out = np.full(len(ts), "", dtype=object)
    for nm, a, b in SPLITS:
        out[(ts >= a) & (ts < b)] = nm
    return out


def day_boot(x, day, nb=1000, seed=7):
    d = pd.DataFrame({"x": x, "d": day}).dropna().groupby("d").x.agg(["sum", "count"])
    if len(d) < 20:
        return (np.nan, np.nan)
    idx = np.random.default_rng(seed).integers(0, len(d), (nb, len(d)))
    s, c = d["sum"].to_numpy()[idx].sum(1), d["count"].to_numpy()[idx].sum(1)
    return tuple(np.quantile(s / c, [0.025, 0.975]))


def _qtab(d, col, cuts):
    q = np.searchsorted(cuts, d[col].to_numpy())
    q1, q5 = (q == 0) * 1.0, (q == 4) * 1.0
    r = d.r.to_numpy()
    return pd.DataFrame({"r1": r * q1, "q1": q1, "r5": r * q5, "q5": q5, "day": d.day.to_numpy()}).groupby("day").sum()


def q_effect(d, col, cuts, nb=1000, seed=7):
    """Q5 − Q1 잔차 차(DEV 경계) + 일 블록 부트 CI."""
    d = d[[col, "r", "day"]].dropna()
    if len(d) < 200:
        return dict(n=len(d), eff=np.nan, lo=np.nan, hi=np.nan, z=np.nan)
    A = _qtab(d, col, cuts).to_numpy()
    f = lambda S: S[..., 2] / np.maximum(S[..., 3], 1) - S[..., 0] / np.maximum(S[..., 1], 1)
    eff = float(f(A.sum(0)))
    bs = f(A[np.random.default_rng(seed).integers(0, len(A), (nb, len(A)))].sum(1))
    sd = bs.std()
    return dict(n=len(d), eff=eff, lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), z=eff / sd if sd > 0 else np.nan)


def conditions(ev, feats):
    rows = []
    for col in feats:
        dev = ev[(ev.split == "DEV") & (ev.kind == "real")][col].dropna()
        if len(dev) < 500:
            continue
        cuts = np.quantile(dev, [.2, .4, .6, .8])
        if len(np.unique(cuts)) < 4:
            continue
        row = {"feat": col}
        for kind in ("real", "placebo"):
            for per in ("DEV", "RECENT", "TEST"):
                sub = ev[(ev.kind == kind) & (ev.split.isin(["VAL", "TEST"]) if per == "RECENT" else ev.split == per)]
                for kk, vv in q_effect(sub, col, cuts).items():
                    row[f"{kind}_{per}_{kk}"] = vv
        rows.append(row)
    t = pd.DataFrame(rows)
    t["pass"] = ((np.sign(t.real_RECENT_lo) == np.sign(t.real_RECENT_hi)) & (np.sign(t.real_DEV_eff) == np.sign(t.real_RECENT_eff))
                 & (t.real_RECENT_eff.abs() >= .02))
    t["bonf"] = t["pass"] & (t.real_RECENT_z.abs() >= 3.3)
    return t.sort_values("real_RECENT_z", key=np.abs, ascending=False)


def wall_specific(ev, feats, nb=500):
    """같은 조건이 청산벽에서 위약 선보다 더 세게 작동하나: (실제 Q5−Q1) − (위약 Q5−Q1), RECENT 같은 날 묶음 부트."""
    rec = ev[ev.split.isin(["VAL", "TEST"])]
    days = np.sort(rec.day.unique())
    rows = []
    for col in feats:
        dev = ev[(ev.split == "DEV") & (ev.kind == "real")][col].dropna()
        cuts = np.quantile(dev, [.2, .4, .6, .8])
        tabs = [_qtab(rec[rec.kind == kd][[col, "r", "day"]].dropna(), col, cuts).reindex(days, fill_value=0).to_numpy() for kd in ("real", "placebo")]
        f = lambda S: S[..., 2] / np.maximum(S[..., 3], 1) - S[..., 0] / np.maximum(S[..., 1], 1)
        idx = np.random.default_rng(7).integers(0, len(days), (nb, len(days)))
        bs = f(tabs[0][idx].sum(1)) - f(tabs[1][idx].sum(1))
        rows.append(dict(feat=col, real=float(f(tabs[0].sum(0))), placebo=float(f(tabs[1].sum(0))),
                         diff=float(f(tabs[0].sum(0)) - f(tabs[1].sum(0))), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975))))
    return pd.DataFrame(rows)


def models(ev, feats, kind):
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from sklearn.metrics import roc_auc_score, log_loss
    d = ev[ev.kind == kind]
    tr = d[d.split == "DEV"]
    out, preds = {}, {}
    for nm, cols in (("GEO", GEO), ("ALL", GEO + [c for c in feats if c not in GEO])):
        m = HGB(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0, random_state=0)
        m.fit(tr[cols], tr.y)
        for per in ("VAL", "TEST"):
            te = d[d.split == per]
            p = m.predict_proba(te[cols])[:, 1]
            preds[(nm, per)] = p
            out[f"{nm}_{per}"] = dict(auc=roc_auc_score(te.y, p), ll=log_loss(te.y, p), acc=float(np.mean((p > .5) == te.y)),
                                      auc_p0=roc_auc_score(te.y, te.p0), ll_p0=log_loss(te.y, te.p0.clip(.01, .99)),
                                      acc_p0=float(np.mean((te.p0 > .5) == te.y)), n=len(te), y=float(te.y.mean()), p0=float(te.p0.mean()))
    for per in ("VAL", "TEST"):
        te = d[d.split == per].reset_index(drop=True)
        g, a = preds[("GEO", per)], preds[("ALL", per)]
        days = te.day.to_numpy()
        ud = np.unique(days)
        di = {x: np.flatnonzero(days == x) for x in ud}
        bs = []
        for seed in range(300):
            ii = np.concatenate([di[x] for x in np.random.default_rng(seed).choice(ud, len(ud))])
            bs.append(roc_auc_score(te.y.iloc[ii], a[ii]) - roc_auc_score(te.y.iloc[ii], g[ii]))
        out[f"dAUC_{per}"] = dict(d=out[f"ALL_{per}"]["auc"] - out[f"GEO_{per}"]["auc"], lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)))
        lift = a - g                                                          # 기하보다 «더 뚫린다» 고 본 상위 20%
        top = lift >= np.quantile(lift, .8)
        pnl = np.where(te.y == 1, te.gain_bp, -te.loss_bp) - 8               # 참고: 뚫음 쪽 따라가기, 배리어 청산, 테이커 왕복 8bp
        out[f"top20_{per}"] = dict(n=int(top.sum()), y=float(te.y[top].mean()), p0=float(te.p0[top].mean()), geo=float(g[top].mean()),
                                   all=float(a[top].mean()), follow_bp=float(pnl[top].mean()), follow_ci=day_boot(pnl[top], days[top]))
    return out


def test():
    raw = pd.read_parquet(f"{OUT}/events.parquet")
    ev = raw[raw.y.notna()].copy()
    ev["split"] = split_of(ev.t)
    ev = ev[ev.split != ""]
    ev["r"] = ev.y - ev.p0
    feats = [c for c in ev.columns if c not in NONFEAT]
    rep = {"n": {f"{a}_{b}": int(v) for (a, b), v in ev.groupby(["kind", "split"]).size().items()},
           "res_share": {kd: {str(int(a)): float(b) for a, b in raw[raw.kind == kd].res.value_counts(normalize=True).items()} for kd in ("real", "placebo")}}
    base = {}
    for kind in ("real", "placebo"):
        for per in ("DEV", "VAL", "TEST"):
            d = ev[(ev.kind == kind) & (ev.split == per)]
            base[f"{kind}_{per}"] = dict(n=len(d), y=d.y.mean(), p0=d.p0.mean(), resid=d.r.mean(), resid_ci=day_boot(d.r.to_numpy(), d.day.to_numpy()),
                                         up_y=d[d.s > 0].y.mean(), dn_y=d[d.s < 0].y.mean(), gain_bp=d.gain_bp.median(), loss_bp=d.loss_bp.median())
    rep["base"] = base
    cond = conditions(ev, feats)
    cond.to_csv(f"{OUT}/conditions.csv", index=False)
    rep["pass"], rep["bonf"] = cond[cond["pass"]].feat.tolist(), cond[cond.bonf].feat.tolist()
    ws = wall_specific(ev, [c for c in rep["pass"] if c not in MASS])
    ws.to_csv(f"{OUT}/wall_specific.csv", index=False)
    feats_pl = [c for c in feats if c not in MASS]
    rep["model_real"] = models(ev, feats, "real")
    rep["model_real_no_mass"] = models(ev, feats_pl, "real")
    rep["model_placebo"] = models(ev, feats_pl, "placebo")
    rep["flags"] = dict(fresh_forward_bar_by_bar=True, trade_ledgers_used_as_input=False, saved_parent_exit_timestamps_used=False, future_rows_used_for_entry=False)
    json.dump(rep, open(f"{OUT}/report.json", "w"), ensure_ascii=False, indent=1, default=lambda x: float(x) if np.isscalar(x) else list(x))
    pd.set_option("display.width", 250)
    print(json.dumps({k: rep[k] for k in ("n", "res_share", "base")}, ensure_ascii=False, indent=1, default=float))
    show = ["feat", "real_DEV_eff", "real_RECENT_eff", "real_RECENT_lo", "real_RECENT_hi", "real_RECENT_z", "real_TEST_eff",
            "placebo_DEV_eff", "placebo_RECENT_eff", "pass", "bonf"]
    print(cond[show].round(3).to_string())
    print(ws.round(3).to_string())
    for kk in ("model_real", "model_real_no_mass", "model_placebo"):
        print(kk, json.dumps(rep[kk], default=float, ensure_ascii=False))


# ── 실측 청산(실시간 데이터) ────────────────────────────────────────────────
def rt():
    """07-19~ 실측 청산(바이낸스 forceOrder 1분 합, 서버 lake tail_risk_1m 사본). 위 사건이면 «밀리는 쪽» = 숏 청산(short_usd)."""
    ev = pd.read_parquet(f"{OUT}/events.parquet")
    ev = ev[ev.y.notna() & (ev.t >= pd.Timestamp("2026-07-19").value // 10**6)].copy()
    tr = pd.read_parquet(TAIL)
    tr = tr[tr.valid_liq_stream & ~tr.ws_stale]
    g = tr.groupby(tr.t // M5 * M5)[["long_usd_1m", "short_usd_1m"]].sum()
    full = pd.DataFrame(index=np.arange(g.index.min(), g.index.max() + 1, M5)).join(g).fillna(0.0)
    L, S = full.long_usd_1m, full.short_usd_1m
    q90 = lambda x: x.shift(1).rolling(2016, min_periods=500).quantile(0.9)
    feat = pd.DataFrame({"t": full.index, "L": L.values, "S": S.values, "L1h": L.rolling(12).sum().values, "S1h": S.rolling(12).sum().values,
                         "Lq": (L / q90(L).replace(0, np.nan)).values, "Sq": (S / q90(S).replace(0, np.nan)).values})
    ev = ev.merge(feat, on="t", how="left")                                   # t = 봉 k 시작 → 봉 k 5분 합(봉 끝에 앎)
    up = ev.s > 0
    ev["fuel"] = np.log1p(np.where(up, ev.S, ev.L))                         # 향하는 쪽으로 밀리는 포지션의 청산
    ev["counter"] = np.log1p(np.where(up, ev.L, ev.S))
    ev["fuel1h"] = np.log1p(np.where(up, ev.S1h, ev.L1h))
    ev["fuel_hot"] = (np.where(up, ev.Sq, ev.Lq) >= 1).astype(float)
    ev["r"] = ev.y - ev.p0
    rep = {}
    for kind in ("real", "placebo"):
        d = ev[ev.kind == kind]
        rep[kind] = {"n": len(d), "days": int(d.day.nunique()), "y": d.y.mean(), "p0": d.p0.mean(), "hot_share": d.fuel_hot.mean()}
        for col in ("fuel", "fuel1h", "counter", "fuel_hot"):
            dd = d[[col, "r", "day"]].dropna()
            cuts = np.array([.5, .6, .7, .8]) if col == "fuel_hot" else np.quantile(dd[col], [.2, .4, .6, .8])   # 0/1 → Q1 = 0 · Q5 = 1
            rep[kind][col] = q_effect(dd, col, cuts) if len(np.unique(cuts)) == 4 else "분위 겹침"
        # 사후 통제(결과 본 뒤): 기하 오프셋 로지스틱에 닿은 봉 거래량·폭·델타·모멘텀을 같이 넣어도 hot 이 남나 · 기간 반분
        import statsmodels.api as sm
        dd = d.dropna(subset=["rvol", "rng_sig", "dlt", "r15", "r1h", "h_sig"])
        off = np.log(dd.p0.clip(.02, .98) / (1 - dd.p0.clip(.02, .98)))
        grp = {"groups": pd.factorize(dd.day)[0]}
        for nm, cols in (("hot_only", ["fuel_hot"]), ("hot_ctrl", ["fuel_hot", "rvol", "rng_sig", "dlt", "r15", "r1h", "h_sig"])):
            m = sm.GLM(dd.y, sm.add_constant(dd[cols]), family=sm.families.Binomial(), offset=off).fit(cov_type="cluster", cov_kwds=grp)
            rep[kind][nm] = dict(coef=float(m.params.fuel_hot), se=float(m.bse.fuel_hot), p=float(m.pvalues.fuel_hot))
        h1 = dd.t < pd.Timestamp("2026-09-01").value // 10**6
        rep[kind]["halves_hot_minus_cold"] = [float(x[x.fuel_hot == 1].r.mean() - x[x.fuel_hot == 0].r.mean()) for x in (dd[h1], dd[~h1])]
    json.dump(rep, open(f"{OUT}/report_rt.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=float))


def selftest():
    h = np.array([10.0, 10.2, 10.6, 10.1]); l = np.array([9.9, 9.95, 10.0, 9.5])
    assert first_touch(h, l, 0, 10.5, 9.8, 10) == 1
    assert first_touch(h, l, 3, 10.5, 9.8, 10) == 0
    assert first_touch(np.array([11.0]), np.array([9.0]), 0, 10.5, 9.8, 10) == -1
    assert first_touch(h, l, 0, 99.0, 1.0, 10) == -2
    b5 = pd.DataFrame({"high": [10, 10, 10.6, 10.7, 10], "low": [9.9] * 5, "close": [10, 10, 10.4, 10.4, 10]})
    e = np.array([np.nan, 10.5, 10.5, 10.5, 10.5]); nan = np.full(5, np.nan)
    ev = detect(b5, e, e + .2, nan, nan)
    assert ev.k.tolist() == [2] and ev.s.tolist() == [1], ev                  # 봉 3 은 쿨다운
    assert np.searchsorted(np.array([.5, .6, .7, .8]), [0.0, 1.0]).tolist() == [0, 4]   # fuel_hot 0/1 → Q1/Q5
    d = pd.DataFrame({"x": [0.0] * 150 + [1.0] * 150, "r": [0.0] * 150 + [0.1] * 150, "day": list(range(300))})
    assert abs(q_effect(d, "x", np.array([.5, .6, .7, .8]))["eff"] - 0.1) < 1e-12
    print("selftest OK")


if __name__ == "__main__":
    {"--build": build, "--test": test, "--rt": rt, "--selftest": selftest}[sys.argv[1]]()
