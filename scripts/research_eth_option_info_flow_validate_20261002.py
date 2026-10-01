"""옵션 카드 «옵션 정보 · 검정 전» -- 흐름·비율·VoV 지표가 화면이 말하는 «의미»대로 움직이는지(수익 아님) 2026 ETH 로 검정.

지표 정의는 라이브 그대로:
  vg·vgb   = live_deribit_block_trade_collector.hourly_flow (테이커 순베가 $/vol pt, 블록 몫) -- 정시 t 의 값 = [t-24h, t) 완결 시간 합(app.js optInfoRows)
  블록 순베가·순델타 = 같은 파일 block_shape 를 블록마다 그대로 호출해 [t-24h, t) 합(block_sum_by_coin). 선물 헤지 다리는 ETH 0건 → 옵션 다리만.
  O/S      = [t-24h, t) 옵션 체결 수량 × 지수(t, 바이낸스 종가 대리) ÷ 같은 창 바이낸스 ETHUSDT 선물 quote_volume
  vov24    = collect_deribit_option_gex.write_state: DVOL 시간 종가 25개의 로그 변화 표준편차(ddof 1) × 100
시점 계약: D(t) = 시작 t-1h 인 DVOL 봉의 종가 · P(t) = 시작 t-1분 인 1분봉 종가. 결과(목표)는 전부 t 이후 구간.

사전 판정 기준(결과 보기 전 고정, CRITERIA 로 JSON 에도 기록):
  주 검정 = 순위 OLS(모든 변수를 순위 → 표준화) 의 x 계수, 시간별 표본 + 일 블록 부트스트랩 95% CI(B=1000).
  방향 주장이 있는 것: 부호 일치 + CI 0 배제 = 지지 / 반대 부호 + CI 0 배제 = 반대 / CI 0 포함 = 근거 없음(검정력 추정 병기).
  «방향 없음» 주장(수익률 부호): CI 0 포함 = 주장과 부합 / 0 배제 = 주장과 어긋남.
  겹치지 않는 하루 1표본(00:00 UTC) 결과는 보조(부호가 주 검정과 반대면 «흔들림»으로 적는다).
  VoV 띠: 상위 5분위 − 하위 5분위의 크기비(mean|r|/(σ√(2/π))) 차 > 0 이고 CI 0 배제 = 지지.
실행: python scripts/research_eth_option_info_flow_validate_20261002.py [--selftest]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import collect_deribit_option_gex_20260815 as gex            # noqa: E402
import live_deribit_block_trade_collector_20260928 as col    # noqa: E402

D = ROOT / "tmp/opt_validate_20261002"
H = 3_600_000
B = 1000
CRITERIA = {
    "primary": "rank-OLS 표준화 계수, 시간별 표본, 일 블록 부트스트랩 95% CI (B=1000)",
    "directional": "주장 부호 + CI 0 배제 = 지지 · 반대 부호 + CI 0 배제 = 반대 · CI 0 포함 = 근거 없음(검정력 병기)",
    "null_claim": "방향 없음 주장: CI 0 포함 = 부합 · 0 배제 = 어긋남",
    "daily": "00:00 UTC 하루 1표본 = 보조 (주 검정과 부호 반대면 흔들림)",
    "vov_band": "vov 상위5분위 − 하위5분위 크기비 차 > 0, CI 0 배제 = 지지",
}


# ---------- 지표 ----------
def flow_hourly(tr: pd.DataFrame) -> pd.DataFrame:
    """hourly_flow 의 벡터판 -- 시간 버킷(시작 h)별 cb/cs/pb/ps·vg·vgb·liq. selftest 가 원본과 대조한다."""
    spec = {i: gex._parse_instrument(i) for i in tr["instrument_name"].unique()}
    ok = tr["instrument_name"].map(lambda i: spec[i] is not None) & tr["instrument_name"].str.startswith("ETH-")
    t = tr[ok].copy()
    t["call"] = t["instrument_name"].map(lambda i: spec[i]["option_type"] == "call")
    t["K"] = t["instrument_name"].map(lambda i: spec[i]["strike"])
    exp = t["instrument_name"].map(lambda i: spec[i]["expiration_ts"].timestamp())
    yrs = (exp - t["ts_ms"] / 1000) / (365.0 * 86400)
    S, sig = t["index_price"].fillna(0).astype(float), t["iv"].fillna(0).astype(float) / 100
    sgn = np.where(t["direction"] == "buy", 1.0, -1.0)
    good = (S > 0) & (sig > 0) & (yrs > 0)
    with np.errstate(all="ignore"):
        d1 = (np.log(S / t["K"]) + 0.5 * sig * sig * yrs) / (sig * np.sqrt(yrs))
        vg = sgn * t["amount"] * S * np.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) * np.sqrt(yrs) / 100
    t["vg"] = np.where(good, vg, 0.0)
    t["vgb"] = np.where(t["block_trade_id"].notna(), t["vg"], 0.0)
    t["dlt"] = np.where(good & (t["K"] > 0), sgn * t["amount"] * np.where(t["call"], ndtr(d1), ndtr(d1) - 1), 0.0)
    buy = t["direction"] == "buy"
    for k, m in (("cb", t["call"] & buy), ("cs", t["call"] & ~buy), ("pb", ~t["call"] & buy), ("ps", ~t["call"] & ~buy)):
        t[k] = np.where(m, t["amount"], 0.0)
    t["liq"] = np.where(t["liquidation"].notna() & (t["liquidation"].astype(str) != "None"), t["amount"], 0.0)
    t["h"] = t["ts_ms"] // H * H
    return t.groupby("h")[["cb", "cs", "pb", "ps", "dlt", "vg", "vgb", "liq"]].sum()


def roll24_before(s: pd.Series, grid: np.ndarray) -> pd.Series:
    """시간 버킷 합(인덱스 = 버킷 시작) → 정시 t 의 [t-24h, t) 합. 진행 중 시간(t 자신)은 안 넣는다(app.js slice(0,-1))."""
    full = s.reindex(np.arange(grid.min() - 25 * H, grid.max() + H, H), fill_value=0.0)
    return full.rolling(24).sum().shift(1).reindex(grid)


def block_hourly(tr: pd.DataFrame) -> pd.DataFrame:
    """블록마다 라이브 block_shape(옵션 다리만) → 블록 첫 다리 시간 버킷별 순델타·순베가 합."""
    bl = tr[tr["block_trade_id"].notna() & tr["instrument_name"].str.startswith("ETH-")].sort_values(["ts_ms", "trade_id"])
    out = []
    cols = ["instrument_name", "direction", "amount", "iv", "index_price"]
    for bid, g in bl.groupby("block_trade_id", sort=False):
        try:
            s = col.block_shape(g[cols].to_dict("records"), int(g["ts_ms"].iloc[0]))
        except Exception:   # 라이브와 같다 -- 이름 규칙 밖이면 빠진다
            continue
        out.append((int(g["ts_ms"].iloc[0]) // H * H, s["net_delta"], s["net_vega_usd"]))
    b = pd.DataFrame(out, columns=["h", "bdelta", "bvega"])
    return b.groupby("h").agg(bdelta=("bdelta", "sum"), bvega=("bvega", "sum"), bn=("bvega", "size"))


def market_frame() -> pd.DataFrame:
    """정시 t 격자: DVOL·가격·실현변동성·선물 명목·vov24 와 t 이후 목표."""
    dv = pd.read_parquet(D / "eth_dvol_1h.parquet")
    Dt = pd.Series(dv["c"].values, index=dv["ts"].values + H)          # D(t) = 시작 t-1h 봉 종가
    k = pd.read_parquet(D / "ethusdt_1m.parquet").set_index("open_time")
    lr = np.log(k["close"]).diff()
    rv = (lr ** 2).rolling(1440).sum()                                  # 인덱스 m = [m-1439, m] 분
    qv = k["quote_volume"].rolling(1440).sum()
    grid = np.arange(int(k.index.min()) + 26 * H, int(k.index.max()) + 60_000 - 24 * H + 1, H)
    g = pd.DataFrame(index=grid)
    at = lambda s, off: s.reindex(grid + off).values                    # noqa: E731
    g["D"] = Dt.reindex(grid).values
    g["D_m24"] = Dt.reindex(grid - 24 * H).values
    g["D_p24"] = Dt.reindex(grid + 24 * H).values
    g["D_p1"] = Dt.reindex(grid + H).values
    g["D_m1"] = Dt.reindex(grid - H).values
    g["P"] = at(k["close"], -60_000)
    g["P_p1"] = at(k["close"], H - 60_000)
    g["P_p24"] = at(k["close"], 24 * H - 60_000)
    g["rv_past"] = np.sqrt(at(rv, -60_000))
    g["rv_fwd"] = np.sqrt(at(rv, 24 * H - 60_000))
    g["qv24"] = at(qv, -60_000)
    dl = np.log(Dt).diff()
    g["vov24"] = (dl.rolling(24).std() * 100).reindex(grid).values     # D(t-24h)..D(t) 25점 → 변화 24개
    g["logD"] = np.log(g["D"])
    g["dD_past"] = np.log(g["D"] / g["D_m24"])
    g["dD_fwd"] = np.log(g["D_p24"] / g["D"])
    g["absdD_fwd"] = g["dD_fwd"].abs()
    g["r1"] = np.log(g["P_p1"] / g["P"])
    g["r24"] = np.log(g["P_p24"] / g["P"])
    g["absr24"] = g["r24"].abs()
    g["sig24"] = g["D"] / 100 / math.sqrt(365)                          # 카드 «24시간 1σ» = IV/√365
    g["lrv_past"] = np.log(g["rv_past"])
    g["lrv_fwd"] = np.log(g["rv_fwd"])
    g["lrr_fwd"] = np.log(g["rv_fwd"] / g["sig24"])                     # 실현/내재 비
    g["lrr_past"] = np.log(g["rv_past"] / g["sig24"])
    g["day"] = g.index // (24 * H)
    return g


# ---------- 통계 ----------
def _z(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / (a.std() or 1)


def _beta_dr2(y, X, c):
    """y ~ [c, x](표준화) 의 x 계수와 증분 R²."""
    one = np.ones((len(y), 1))
    A = np.column_stack([one, X])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2f = 1 - ((y - A @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    if c:
        A0 = np.column_stack([one, X[:, 1:]])
        b0, *_ = np.linalg.lstsq(A0, y, rcond=None)
        r20 = 1 - ((y - A0 @ b0) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    else:
        r20 = 0.0
    return b[1], r2f - r20


def test(g, y, x, ctrl=(), rank=True, seed=0):
    """주 검정: 시간별 + 일 블록 부트스트랩. 보조: 00:00 UTC 하루 1표본(일 부트스트랩)."""
    cols = [y, x, *ctrl]
    d = g[cols + ["day"]].replace([np.inf, -np.inf], np.nan).dropna()
    T = (lambda a: rankdata(a)) if rank else (lambda a: a)
    out = {"n_hours": len(d), "n_days": int(d["day"].nunique())}
    for tag, dd in (("hourly", d), ("daily00", d[d.index % (24 * H) == 0])):
        Y = _z(T(dd[y].values))
        X = np.column_stack([_z(T(dd[c].values)) for c in [x, *ctrl]])
        b, r2 = _beta_dr2(Y, X, bool(ctrl))
        days = dd["day"].values
        ud = np.unique(days)
        idx = {u: np.flatnonzero(days == u) for u in ud}
        rng = np.random.default_rng(seed)
        bs, rs = [], []
        for _ in range(B):
            ii = np.concatenate([idx[u] for u in rng.choice(ud, len(ud))])
            bb, rr = _beta_dr2(Y[ii], X[ii], bool(ctrl))
            bs.append(bb); rs.append(rr)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        res = {"n": len(dd), "beta": round(float(b), 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
               "dR2": round(float(r2), 4), "dR2_ci": [round(float(v), 4) for v in np.percentile(rs, [2.5, 97.5])]}
        if lo <= 0 <= hi and b != 0:
            se = (hi - lo) / (2 * 1.96)
            res["days_for_80pct_power"] = int(math.ceil(len(ud) * ((1.96 + 0.84) * se / abs(b)) ** 2))
        out[tag] = res
    return out


def verdict(t, claim):
    """claim: +1 / −1 / 0(방향 없음)."""
    b, (lo, hi) = t["hourly"]["beta"], t["hourly"]["ci"]
    zero = lo <= 0 <= hi
    if claim == 0:
        v = "부합(0 포함)" if zero else "어긋남(0 배제)"
    elif zero:
        v = "근거 없음(CI 0 포함)"
    else:
        v = "지지" if np.sign(b) == claim else "반대"
    db = t["daily00"]["beta"]
    if claim and not zero and np.sign(db) != np.sign(b):
        v += " · 하루1표본 부호 흔들림"
    return v


def sign_agree(g, x, y, seed=0):
    d = g[[x, y, "day"]].dropna()
    d = d[(d[x] != 0) & (d[y] != 0)]
    a = (np.sign(d[x]) == np.sign(d[y])).astype(float).values
    days = d["day"].values
    ud = np.unique(days)
    idx = {u: np.flatnonzero(days == u) for u in ud}
    rng = np.random.default_rng(seed)
    bs = [a[np.concatenate([idx[u] for u in rng.choice(ud, len(ud))])].mean() for _ in range(B)]
    return {"n": len(a), "agree": round(float(a.mean()), 4), "ci": [round(float(v), 4) for v in np.percentile(bs, [2.5, 97.5])]}


def band_by_vov(g, seed=0):
    d = g[["vov24", "r24", "sig24", "day"]].dropna().copy()
    d["q"] = pd.qcut(d["vov24"], 5, labels=False)
    d["u"] = d["r24"].abs() / d["sig24"]
    d["hit"] = (d["u"] <= 1).astype(float)
    k = math.sqrt(2 / math.pi)
    rows = []
    for q, dd in d.groupby("q"):
        rows.append({"q": int(q) + 1, "n": len(dd), "vov_med": round(float(dd["vov24"].median()), 3),
                     "hit": round(float(dd["hit"].mean()), 4), "size_ratio": round(float(dd["u"].mean() / k), 4)})
    # 상위 − 하위 5분위 차, 일 블록 부트스트랩
    days = d["day"].values
    ud = np.unique(days)
    idx = {u: np.flatnonzero(days == u) for u in ud}
    rng = np.random.default_rng(seed)
    q, u, h = d["q"].values, d["u"].values, d["hit"].values
    def diff(ii):
        top, bot = ii[q[ii] == 4], ii[q[ii] == 0]
        return u[top].mean() / k - u[bot].mean() / k, h[top].mean() - h[bot].mean()
    all_i = np.arange(len(d))
    bs = np.array([diff(np.concatenate([idx[x] for x in rng.choice(ud, len(ud))])) for _ in range(B)])
    pt = diff(all_i)
    return {"bins": rows,
            "top_minus_bottom": {"size_ratio": round(pt[0], 4), "size_ratio_ci": [round(float(v), 4) for v in np.percentile(bs[:, 0], [2.5, 97.5])],
                                 "hit": round(pt[1], 4), "hit_ci": [round(float(v), 4) for v in np.percentile(bs[:, 1], [2.5, 97.5])]}}


def hedge_overlap(tr, g):
    """선물 헤지 다리 포함판 vs 옵션 다리만. 라이브 표(block_future_legs, 09-30 13:32~)엔 ETH 다리 0건 → 그 창에선 두 판이 같다.
    RFQ 표(09-24~)의 hedge 필드로 헤지가 붙은 ETH 패키지를 찾아 block_shape(fut=[hedge]) 로 순델타 차를 잰다."""
    fl = pd.read_parquet(D / "block_fut_legs.parquet")
    rfq = pd.read_parquet(D / "block_rfq.parquet")
    e = rfq[(rfq["api"] == "ETH") & rfq["hedge"].notna()]
    cols = ["instrument_name", "direction", "amount", "iv", "index_price"]
    out = {"live_fut_legs_window": [str(pd.to_datetime(fl["ts_ms"].min(), unit="ms")), str(pd.to_datetime(fl["ts_ms"].max(), unit="ms"))],
           "live_fut_legs_eth": int(fl["instrument_name"].str.startswith("ETH").sum()), "rfq_eth": int((rfq["api"] == "ETH").sum()),
           "rfq_eth_hedged": len(e), "cases": []}
    for _, r in e.iterrows():
        legs = tr[tr["block_rfq_id"] == float(r["rfq_id"])].sort_values("ts_ms")
        if legs.empty:
            continue
        hd = json.loads(r["hedge"])
        ts = int(legs["ts_ms"].iloc[0])
        a = col.block_shape(legs[cols].to_dict("records"), ts)["net_delta"]
        b = col.block_shape(legs[cols].to_dict("records"), ts, [hd])["net_delta"]
        t = ts // H * H + H
        w = g.loc[(g.index >= t - 7 * 24 * H) & (g.index <= t + 7 * 24 * H), "bdelta24"].abs()
        out["cases"].append({"t": str(pd.to_datetime(ts, unit="ms")), "opt_only": a, "with_hedge": b,
                             "bdelta24_at_next_hour": round(float(g["bdelta24"].get(t, np.nan)), 1),
                             "median_abs_bdelta24_pm7d": round(float(w.median()), 1)})
    return out


def vov_live_repro():
    """서버 option_summary(09-28~, 10분 표본)로 write_state 식 그대로(시간별 마지막 값 · 지난 25h · ≥12점) vov24 를 내고
    Deribit DVOL 1시간 이력판과 같은 정시에서 대조한다."""
    s = pd.read_parquet(D / "eth_summary.parquet")
    s = s[(s["currency"] == "ETH") & (s["dvol"] > 0)]
    ts = (pd.to_datetime(s["recorded_at_utc"], utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)
    s = s.assign(ms=ts.values, h=ts.values // H * H).sort_values("ms")
    last = s.groupby("h")["dvol"].last()
    dv = pd.read_parquet(D / "eth_dvol_1h.parquet")
    hist = np.log(pd.Series(dv["c"].values, index=dv["ts"].values + H)).diff().rolling(24).std() * 100
    rows = []
    for t in range(int(last.index.min()) + 25 * H, int(last.index.max()) + H, H):
        v = last[(last.index >= t - 25 * H) & (last.index < t)].values
        if len(v) >= 12 and t in hist.index:
            d = np.diff(np.log(v))
            rows.append((np.std(d, ddof=1) * 100, hist[t]))
    a = np.array(rows)
    return {"n_hours": len(a), "corr": round(float(np.corrcoef(a.T)[0, 1]), 4),
            "median_abs_diff": round(float(np.median(np.abs(a[:, 0] - a[:, 1]))), 4), "live_median": round(float(np.median(a[:, 0])), 4)}


# ---------- 실행 ----------
def run():
    g = market_frame()
    res = {"criteria": CRITERIA, "window": [str(pd.to_datetime(g.index.min(), unit="ms")), str(pd.to_datetime(g.index.max(), unit="ms"))]}
    C3 = ("logD", "lrv_past", "dD_past")
    # 4. VoV
    v = {"a_absdD_fwd_raw": test(g, "absdD_fwd", "vov24"),
         "c_absdD_fwd_ctrl": test(g, "absdD_fwd", "vov24", ("lrv_past", "logD", "dD_past")),
         "c_rv_over_sig_ctrl": test(g, "lrr_fwd", "vov24", ("lrr_past", "logD")),
         "b_band": band_by_vov(g), "live_repro": vov_live_repro()}
    for k_ in ("a_absdD_fwd_raw", "c_absdD_fwd_ctrl", "c_rv_over_sig_ctrl"):
        v[k_]["verdict"] = verdict(v[k_], +1)
    res["vov"] = v
    tp = D / "eth_opt_trades_2026.parquet"
    if not tp.exists():
        res["flow"] = "옵션 체결 파일 없음 -- 흐름·블록·O/S 생략"
        return res, g
    tr = pd.read_parquet(tp)
    fh = flow_hourly(tr)
    grid = g.index.values
    for c in ("vg", "vgb"):
        g[c + "24"] = roll24_before(fh[c], grid).values
    g["scr24"] = g["vg24"] - g["vgb24"]
    g["optq24"] = roll24_before(fh[["cb", "cs", "pb", "ps"]].sum(axis=1), grid).values
    g["os24"] = g["optq24"] * g["P"] / g["qv24"]
    g["os_pct30"] = g["os24"].rolling(720, min_periods=240).apply(lambda a: (a <= a[-1]).mean(), raw=True)
    bh = block_hourly(tr)
    for c in ("bdelta", "bvega", "bn"):
        g[c + "24"] = roll24_before(bh[c], grid).values
    first = int(tr["ts_ms"].min()) // H * H + 24 * H         # 24h 창이 꽉 찬 첫 정시부터
    g = g[g.index >= first]
    f = {}
    for x in ("vg24", "vgb24", "scr24"):
        f[x] = {"dD_fwd": test(g, "dD_fwd", x, C3), "lrv_fwd": test(g, "lrv_fwd", x, C3),
                "r24": test(g, "r24", x, C3), "r24_sign_agree": sign_agree(g, x, "r24")}
        f[x]["dD_fwd"]["verdict"] = verdict(f[x]["dD_fwd"], +1)
        f[x]["lrv_fwd"]["verdict"] = verdict(f[x]["lrv_fwd"], +1)
        f[x]["r24"]["verdict"] = verdict(f[x]["r24"], 0)
    # 보조: 문헌 효과가 짧은 지평에만 있는가(1시간 → 다음 1시간) · 동시각(지난 24h vg 와 지난 24h ΔDVOL) -- 판정엔 안 쓴다
    g["vg1"] = fh["vg"].reindex(g.index - H, fill_value=0.0).values
    g["dD1_fwd"] = np.log(g["D_p1"] / g["D"])
    g["dD1_past"] = np.log(g["D"] / g["D_m1"])
    f["aux_vg1_dD1_fwd"] = test(g, "dD1_fwd", "vg1", ("logD", "dD1_past"))
    f["aux_vg24_dD_past_same_window"] = test(g, "dD_past", "vg24")
    res["vg"] = f
    bk = {"bvega24_dD_fwd": test(g, "dD_fwd", "bvega24", C3),
          "bdelta24_r1": test(g, "r1", "bdelta24"), "bdelta24_r24": test(g, "r24", "bdelta24"),
          "bdelta24_r1_sign": sign_agree(g, "bdelta24", "r1"), "bdelta24_r24_sign": sign_agree(g, "bdelta24", "r24"),
          "blocks_per_24h_median": float(g["bn24"].median())}
    bk["bvega24_dD_fwd"]["verdict"] = verdict(bk["bvega24_dD_fwd"], +1)
    bk["bdelta24_r1"]["verdict"] = verdict(bk["bdelta24_r1"], 0)
    bk["bdelta24_r24"]["verdict"] = verdict(bk["bdelta24_r24"], 0)
    bk["same_as_vgb24"] = {"corr": round(float(g[["bvega24", "vgb24"]].corr().iloc[0, 1]), 6),
                           "max_abs_diff_usd": round(float((g["bvega24"] - g["vgb24"]).abs().max()), 1)}
    bk["hedge_overlap"] = hedge_overlap(tr, g)
    res["block"] = bk
    o = {}
    for x in ("os24", "os_pct30"):
        o[x] = {"lrv_fwd": test(g, "lrv_fwd", x, ("logD", "lrv_past")), "absr24": test(g, "absr24", x, ("logD", "lrv_past")),
                "lrv_fwd_raw": test(g, "lrv_fwd", x)}
        for k_ in o[x]:
            o[x][k_]["verdict"] = verdict(o[x][k_], +1)
    g["ln_opt24"] = np.log(g["optq24"] * g["P"])
    g["ln_fut24"] = np.log(g["qv24"])
    o["decomp"] = {x: test(g, "lrv_fwd", x, ("logD", "lrv_past")) for x in ("ln_opt24", "ln_fut24")}
    o["os24_summary"] = {k_: round(float(v_), 4) for k_, v_ in g["os24"].describe().items()}
    res["os"] = o
    res["n_trades"] = len(tr)
    res["n_blocks"] = int(tr["block_trade_id"].nunique())
    return res, g


def selftest():
    # ① 벡터판 vg/vgb/dlt/수량 == 라이브 hourly_flow (time.time 을 정시 직후로 고정)
    t0 = 1_767_225_600_000 + 5 * H     # 2026-01-01 05:00 UTC
    rows = [(t0 - 3 * H + 10, "ETH-27MAR26-3000-C", "buy", 10.0, 70.0, 2950.0, None, None, "1", "B1"),
            (t0 - 3 * H + 20, "ETH-27MAR26-2500-P", "sell", 4.0, 75.0, 2950.0, "M", None, "2", None),
            (t0 - 1 * H + 5, "ETH-30JAN26-3200-C", "sell", 7.0, 65.0, 2960.0, None, "BLK-1", "3", None),
            (t0 - 1 * H + 9, "ETH-2JAN26-2900-P", "buy", 2.0, 80.0, 2960.0, None, None, "4", None)]
    tr = pd.DataFrame(rows, columns=["ts_ms", "instrument_name", "direction", "amount", "iv", "index_price", "liquidation", "block_trade_id", "trade_id", "x"])
    with mock.patch.object(col.time, "time", return_value=t0 / 1000 + 30):
        live = col.hourly_flow([(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7] is not None) for r in rows])["ETH"]
    mine = flow_hourly(tr)
    for b in live:
        for k in ("cb", "cs", "pb", "ps", "dlt", "vg", "vgb", "liq"):
            m = float(mine[k].get(b["h"], 0.0))
            assert abs(m - b[k]) < 1e-3 + 1e-6 * abs(b[k]), (k, b["h"], m, b[k])
    assert mine["vgb"].abs().sum() > 0 and mine["liq"].sum() == 4.0
    # ② 시점 경계: 버킷 h=t 의 값은 정시 t 의 24h 합에 안 들어가고 t+1h 에 들어간다 · 24h 뒤에 빠진다
    s = pd.Series({t0: 5.0})
    grid = np.arange(t0 - 2 * H, t0 + 27 * H, H)
    r = roll24_before(s, grid)
    assert r[t0] == 0 and r[t0 + H] == 5 and r[t0 + 24 * H] == 5 and r[t0 + 25 * H] == 0
    print("selftest OK")


def fmt(t):
    h = t["hourly"]
    s = f"β {h['beta']:+.3f} [{h['ci'][0]:+.3f},{h['ci'][1]:+.3f}] ΔR² {h['dR2']:.4f} · 하루1표본 β {t['daily00']['beta']:+.3f} [{t['daily00']['ci'][0]:+.3f},{t['daily00']['ci'][1]:+.3f}] n={t['daily00']['n']}"
    if "days_for_80pct_power" in h:
        s += f" · 80%검정력 ~{h['days_for_80pct_power']}일"
    return s + (f" → {t['verdict']}" if "verdict" in t else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        return selftest()
    res, g = run()
    (D / "flow_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("창", res["window"])
    for sect in ("vov", "vg", "block", "os"):
        for k, t in (res.get(sect) or {}).items():
            if isinstance(t, dict) and "hourly" in t:
                print(f"[{sect}] {k}: {fmt(t)}")
            elif isinstance(t, dict) and any(isinstance(v, dict) and "hourly" in v for v in t.values()):
                for k2, t2 in t.items():
                    print(f"[{sect}] {k}→{k2}: {fmt(t2) if 'hourly' in t2 else t2}")
            else:
                print(f"[{sect}] {k}: {t}")


if __name__ == "__main__":
    main()
