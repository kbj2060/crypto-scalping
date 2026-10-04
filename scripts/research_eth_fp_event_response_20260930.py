"""흡수·소진·다이버를 «사건»으로: 위치 → 사건 → 반응 검정 (2026-09-30, 사용자가 붙인 전략 «신호가 아니라 사건 · 반전 예측이 아니라 반전 확인»).

앞 검정(research_eth_fp_pattern_lookback_20260930.py)은 «표식 봉 종가에서 바로 교과서 방향 진입»이었고 방향 0(다이버는 오히려 반대)이었다.
이번엔 전략대로: ① 위치(구조 레벨 근처) ② 사건(관심 등록) ③ 반응(다음 15분 안 반대 공격 + 직전 스윙 이탈 = 확인된 반전,
또는 사건 극값 돌파 = 실패 → 돌파 경고)이 나와야 진입한다.

사전 고정(결과 보기 전):
  시간틀   WHERE = 전일·24h 레벨 · WHAT = 5분봉 사건(대시보드 풋프린트와 같은 봉) · WHEN = 1분봉 확인.
  위치     사건 봉 고가(저가)가 다음 중 하나에서 10bp 안: 전일 고가(저가) · 전일 가치영역 VAH(VAL, 1분 거래량 프로파일 70%) ·
           직전 24h 고가(저가) · 세션 VWAP ±2σ(직전 봉 값). 전부 사건 전에 알려진 값.
  사건     다이버·소진·흡수? = 앞 검정의 L=12·C=288 정의. 효율 흡수 = |델타| z>2 · 거래량 z>2(직전 288봉) · |수익|/|델타| 가 직전 288봉 하위 20%
           (레포트의 Price Efficiency). 방향은 공격 반대쪽(매수 흡수 = 하락 사건).
  반응     사건 봉이 닫힌 뒤 15분(1분봉 15개) 안, 먼저 나온 쪽만:
           확인된 반전(B) = 1분 종가가 직전 15분 스윙 저가(고가)를 이탈 AND 사건 뒤 1분 델타 누적이 반대 방향 → 그 1분 종가에서 사건 방향 진입.
           실패(C)       = 1분 종가가 사건 봉 극값을 넘음 → 그 1분 종가에서 돌파 방향 진입(«흡수 실패 = 돌파 경고»).
           OI 변형(레포트 §14·§15): B_oi = 사건→진입 사이 OI 감소 · C_oi = OI 증가. 알려진 OI = 진입 분이 속한 5분봉의 직전 스탬프.
  참고(A)  사건 봉 종가에서 바로 사건 방향 진입(교과서) -- 앞 검정 재현.
  대조군   같은 위치·같은 반응 절차, 사건 없음: 다이버·소진·흡수? → 사건 없는 새 12봉 극값 봉 · 효율 흡수 → 가격을 «효율적으로» 민 강한 공격 봉.
  라벨     진입 1분 종가 → 30·60·120분 뒤 1분 종가(bp, 진입 방향 부호).
  판정     1순위 = 위치 있음 · B · 60분 · 대조군 초과가 TRAIN(2022~2024)·TEST(2025~) 둘 다 CI>0. 그 밖은 참고.
  CI       일 단위 포아송 블록 부트스트랩 B=1000.
  빠진 것  호가 재보충·청산·감마 -- 과거 이력이 몇 주뿐. 청산은 09-26부터 정확한 원본이 쌓인다.
  🆕레짐 분할(09-30 사용자 반박 자료 «평균이 조건부 효과를 희석한다», 결과 보기 전 고정):
    레짐 = 사건 봉 k 종가에 알려진 값.
      추세 = 5분 SMA144 ± ATR144(대시보드 veto K=1) 위/아래/안 → 사건 방향 기준 순추세 · 역추세 · 횡보.
      변동 = 직전 1시간 실현분산이 직전 30일 80분위 초과(급변)/이하(보통).
      OI   = 1시간 ΔOI(k−1 스탬프까지) z(직전 30일 표준편차) > 2 급증 · < −2 급감 · 그 외 보통.
      펀딩 = 마지막 정산 펀딩(8h5m 안)이 기본 0.01% 초과/기본/미만(로컬 파일 2025-01 ~ 2026-07 뿐).
      만기 = 금요일 04~12시 UTC(Deribit 08시 만기 ±4h).
      청산장은 과거 청산 이력이 없어 못 나눈다. 세션은 이미 위에 있다.
    판정 가족 = (다이버·소진·흡수: 위치 있음 / 효율: 전체) × (B, C) × 60분 × 레짐 13칸 = 104칸.
    통과 = 두 기간 모두 같은 부호로 CI 0 배제. 칸이 독립이면 우연 통과 기대 ≈ 104 × 2 × 0.025² ≈ 0.13칸.

실행: python scripts/research_eth_fp_event_response_20260930.py [--selftest] [--out DIR]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research_eth_fp_pattern_lookback_20260930 as R  # noqa: E402
from dashboard.market_ctx import session_vwap  # noqa: E402

L, C = 12, 288
NEAR_BP = 10.0
WIN_MIN = 15          # 반응 창(분)
SWING_MIN = 15        # 직전 스윙 창(분)
HS = (30, 60, 120)
FUND_CSV = "TOTAL_ETHUSDT_fundingRate_2025_2026.csv"   # data/ 바로 아래 · 2025-01 ~ 2026-07-31
REG_CUTS = ("추세:순추세", "추세:역추세", "추세:횡보", "변동:급변", "변동:보통", "OI:급증", "OI:급감", "OI:보통",
            "펀딩:기본 초과", "펀딩:기본", "펀딩:기본 미만", "만기:±4h", "만기:그밖")


def regimes(df5: pd.DataFrame, oi: np.ndarray, fund: pd.DataFrame) -> dict[str, np.ndarray]:
    """봉 k 종가에 알려진 레짐. 추세는 +1/−1/0(9 = 모름), 나머지는 버킷 문자열("" = 모름). OI 는 k−1 스탬프까지만."""
    c, h, lo = (df5[k] for k in ("close", "high", "low"))
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    sma, atr = c.rolling(144, min_periods=144).mean(), tr.rolling(144, min_periods=144).mean()
    trend = np.where(c > sma + atr, 1, np.where(c < sma - atr, -1, 0))
    trend[~np.isfinite((sma + atr).to_numpy())] = 9
    rv = np.log(c).diff().pow(2).rolling(12).sum()
    q80 = rv.rolling(8640, min_periods=2016).quantile(0.8).shift(1)
    vol = np.where(rv.isna() | q80.isna(), "", np.where(rv > q80, "급변", "보통"))
    o = pd.Series(oi, index=df5.index).shift(1)
    d = o - o.shift(12)
    z = d / d.rolling(8640, min_periods=2016).std().shift(1)
    oib = np.where(z.isna(), "", np.where(z > 2, "급증", np.where(z < -2, "급감", "보통")))
    f = pd.merge_asof(pd.DataFrame({"t": df5.index + pd.Timedelta(minutes=5)}),
                      fund.rename(columns={"calc_time": "ft"})[["ft", "last_funding_rate"]].sort_values("ft"),
                      left_on="t", right_on="ft", direction="backward")
    fr = f["last_funding_rate"].where((f["t"] - f["ft"]) <= pd.Timedelta(hours=8, minutes=5))
    fb = np.where(fr.isna(), "", np.where(fr > 1.0001e-4, "기본 초과", np.where(fr < 0.9999e-4, "기본 미만", "기본")))
    ts = df5.index
    exp = np.where((ts.weekday == 4) & (ts.hour >= 4) & (ts.hour < 12), "±4h", "그밖")
    return {"추세": trend, "변동": vol, "OI": oib, "펀딩": fb, "만기": exp}


def load_1m(data_dir: Path) -> pd.DataFrame:
    fs = sorted(glob.glob(str(data_dir / "ETHUSDT-1m-*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in fs]).drop_duplicates("t").sort_values("t").set_index("t")
    d = d.reindex(np.arange(d.index.min(), d.index.max() + 60_000, 60_000))
    d["delta"] = 2 * d["tb"] - d["v"]
    return d


def value_area(m1: pd.DataFrame) -> pd.DataFrame:
    """UTC 일별 1분 거래량 프로파일(칸 = 그날 중앙가의 5bp)의 VAH/VAL(70%). 인덱스 = 일(1970 기준 정수)."""
    tp = ((m1["h"] + m1["l"] + m1["c"]) / 3).to_numpy(); v = m1["v"].to_numpy()
    day = m1.index.to_numpy() // 86_400_000
    out = {}
    for dd in np.unique(day):
        s = (day == dd) & np.isfinite(tp) & np.isfinite(v)
        if s.sum() < 60:
            continue
        p, w = tp[s], v[s]
        bw = np.median(p) * 5e-4
        hist = np.bincount(np.floor((p - p.min()) / bw).astype(int), weights=w)
        poc = int(np.argmax(hist)); lo = hi = poc; tot = hist[poc]; target = 0.7 * hist.sum()
        while tot < target and (lo > 0 or hi < len(hist) - 1):
            up = hist[hi + 1] if hi < len(hist) - 1 else -1.0
            dn = hist[lo - 1] if lo > 0 else -1.0
            if up >= dn:
                hi += 1; tot += up
            else:
                lo -= 1; tot += dn
        out[int(dd)] = (p.min() + (hi + 1) * bw, p.min() + lo * bw)
    return pd.DataFrame(out, index=["vah", "val"]).T


def levels(df5: pd.DataFrame, m1: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """봉 k 의 고가가 저항 근처인가(near_res) · 저가가 지지 근처인가(near_sup). 레벨은 전부 봉 k 전에 알려진 값."""
    ts = df5.index.to_numpy().astype("datetime64[s]").astype(np.int64)
    day = pd.Series(ts // 86400)
    dmax, dmin = df5["high"].groupby(day.to_numpy()).max(), df5["low"].groupby(day.to_numpy()).min()
    pdh, pdl = (day - 1).map(dmax).to_numpy(), (day - 1).map(dmin).to_numpy()
    va = value_area(m1)
    vah, val = (day - 1).map(va["vah"]).to_numpy(), (day - 1).map(va["val"]).to_numpy()
    h24 = df5["high"].rolling(288, min_periods=200).max().shift(1).to_numpy()
    l24 = df5["low"].rolling(288, min_periods=200).min().shift(1).to_numpy()
    ok = df5["close"].notna().to_numpy()
    vw = np.full(len(df5), np.nan); vs = np.full(len(df5), np.nan)
    a, b = session_vwap(ts[ok].tolist(), df5["high"][ok].tolist(), df5["low"][ok].tolist(), df5["close"][ok].tolist(), df5["vol"][ok].tolist())
    vw[ok] = [np.nan if x is None else x for x in a]; vs[ok] = [np.nan if x is None else x for x in b]
    vwu, vwd = pd.Series(vw + 2 * vs).shift(1).to_numpy(), pd.Series(vw - 2 * vs).shift(1).to_numpy()
    hi, lo = df5["high"].to_numpy(), df5["low"].to_numpy()

    def near(x, lv):
        good = np.isfinite(lv) & np.isfinite(x)
        return good & (np.abs(x / np.where(good, lv, 1) - 1) * 1e4 <= NEAR_BP)
    return (near(hi, pdh) | near(hi, vah) | near(hi, h24) | near(hi, vwu),
            near(lo, pdl) | near(lo, val) | near(lo, l24) | near(lo, vwd))


def eff_events(df5: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """효율 흡수 사건(매수 흡수 −1 · 매도 흡수 +1)과 대조군(같은 강한 공격인데 가격을 «효율적으로» 민 봉, 같은 부호 규약)."""
    dl, v = df5["delta"], df5["vol"]
    zs = lambda s: (s - s.rolling(C, min_periods=C).mean().shift(1)) / s.rolling(C, min_periods=C).std().shift(1)   # noqa: E731
    dz, vz = zs(dl), zs(v)
    eff = ((df5["close"] / df5["close"].shift(1) - 1) * 1e4).abs() / dl.abs().clip(lower=1e-9)
    thr = eff.rolling(C, min_periods=C).quantile(0.2, interpolation="lower")
    strong = (dz.abs() > 2) & (vz > 2)
    side = -np.sign(dl).fillna(0).astype(int).to_numpy()
    return (np.where((strong & (eff <= thr)).to_numpy(), side, 0).astype(np.int8),
            np.where((strong & (eff > thr)).to_numpy(), side, 0).astype(np.int8))


def respond(m1: pd.DataFrame, df5: pd.DataFrame, oi: np.ndarray, bars: np.ndarray, sides: np.ndarray,
            delays: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """(5분봉, 방향, 지연)마다 반응 -- 지연 = 사건이 알려지기까지 더 기다리는 봉 수(흡수는 다음 봉이 닫혀야 안다 → 1).
    🔴09-30 첫 판은 흡수도 사건 봉 종가부터 셌다 = 한 봉 미래참조(A 칸 +9.6bp 가짜).
    반응: kind 0 없음 · 1 확인된 반전 · 2 실패. ent = 진입 1분 위치, dir = 진입 방향, doi = 사건→진입 OI 변화."""
    base = int(m1.index[0] // 60_000)
    t0 = df5.index.to_numpy().astype("datetime64[m]").astype(np.int64)
    c, h, lo, dl = (m1[k].to_numpy(float) for k in ("c", "h", "l", "delta"))
    hi5, lo5 = df5["high"].to_numpy(float), df5["low"].to_numpy(float)
    n = len(bars)
    kind = np.zeros(n, np.int8); ent = np.full(n, -1, np.int64); edir = np.zeros(n, np.int8); doi = np.full(n, np.nan)
    for q in range(n):
        k, s = int(bars[q]), int(sides[q]); dl_bars = 0 if delays is None else int(delays[q])
        j0 = int(t0[k] + 5 * (1 + dl_bars) - base)                      # 사건이 «알려진» 다음 분
        if j0 - SWING_MIN < 0 or j0 + WIN_MIN >= len(c):
            continue
        pre_l, pre_h = lo[j0 - SWING_MIN:j0], h[j0 - SWING_MIN:j0]
        cc, dd = c[j0:j0 + WIN_MIN], dl[j0:j0 + WIN_MIN]
        if not (np.isfinite(cc).all() and np.isfinite(dd).all() and np.isfinite(pre_l).all() and np.isfinite(pre_h).all()):
            continue
        cd = np.cumsum(dd)
        if s < 0:   # 하락 사건: 반전 = 스윙 저가 이탈 + 매도 우위 · 실패 = 사건 고가 돌파
            rev, fail = (cc < pre_l.min()) & (cd < 0), cc > hi5[k]
        else:
            rev, fail = (cc > pre_h.max()) & (cd > 0), cc < lo5[k]
        ir = int(np.argmax(rev)) if rev.any() else WIN_MIN
        jf = int(np.argmax(fail)) if fail.any() else WIN_MIN
        if ir == WIN_MIN and jf == WIN_MIN:
            continue
        kind[q], ent[q], edir[q] = (1, j0 + ir, s) if ir < jf else (2, j0 + jf, -s)
        kb = int((ent[q] + base) // 5 - t0[0] // 5)                      # 진입 분이 속한 5분봉 위치
        if 1 <= kb < len(oi) and k + dl_bars >= 1:
            doi[q] = oi[kb - 1] - oi[k + dl_bars - 1]                               # 둘 다 «그 시점에 알려진» 직전 스탬프
    return {"kind": kind, "ent": ent, "dir": edir, "doi": doi}


def fwd(c: np.ndarray, ent: np.ndarray, H: int) -> np.ndarray:
    out = np.full(len(ent), np.nan)
    ok = (ent >= 0) & (ent + H < len(c))
    out[ok] = (c[ent[ok] + H] / c[ent[ok]] - 1) * 1e4
    return out


def mean_ci(val: np.ndarray, day: np.ndarray, B: int = 1000, seed: int = 11) -> tuple[float, float, float]:
    days, inv = np.unique(day, return_inverse=True)
    s = np.bincount(inv, weights=val); n = np.bincount(inv)
    w = np.random.default_rng(seed).poisson(1.0, (B, len(days)))
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (w @ s) / (w @ n)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(s.sum() / n.sum()), float(lo), float(hi)


def run(data_dir: Path, out_dir: Path) -> None:
    df5 = R.load_5m(data_dir)
    m1 = load_1m(data_dir)
    panel = pd.read_parquet(Path(data_dir).parent / "panel/ETHUSDT.parquet", columns=["timestamp", "sum_open_interest"]).set_index("timestamp")
    oi = panel["sum_open_interest"].reindex(df5.index).to_numpy(float)
    near_res, near_sup = levels(df5, m1)
    mk = R.marks(df5, L, C)
    groups = {k: mk[k] for k in ("div", "exh", "abs")}
    groups["eff"] = eff_events(df5)
    t = df5.index
    split = np.where(np.asarray(t < R.START), "OLD", np.where(np.asarray(t < R.SPLIT), "TRAIN", "TEST"))   # OLD = 예열만
    hr = (t.hour + t.minute / 60.0).to_numpy()
    sess = np.full(len(t), "그밖", dtype=object)
    for name, a, b in R.SESSIONS:
        sess[(hr >= a) & (hr < b)] = name
    reg = regimes(df5, oi, pd.read_csv(Path(data_dir).parents[1] / FUND_CSV, parse_dates=["calc_time"]))
    sess_names = {s[0] for s in R.SESSIONS}

    def cut_mask(cn, ks, sd):
        if cn == "전체":
            return np.ones(len(ks), bool)
        if cn in sess_names:
            return sess[ks] == cn
        dim, b = cn.split(":")
        if dim == "추세":                                                # 사건 방향 기준
            tk = reg["추세"][ks]
            return (tk != 9) & (np.where(tk == 0, "횡보", np.where(tk == sd, "순추세", "역추세")) == b)
        return reg[dim][ks] == b
    DELAY = {"div": 0, "exh": 0, "abs": 1, "eff": 0}                     # 흡수는 다음 봉이 닫혀야 안다
    need = {}
    for gname, (ev, ct) in groups.items():                                # 반응은 (봉, 방향, 지연)에만 달려 있다 -- 한 번씩만 센다
        for arr in (ev, ct):
            for k in np.flatnonzero(arr):
                need[(int(k), int(arr[k]), DELAY[gname])] = None
    keys = np.array(list(need), dtype=np.int64)
    print(f"5분봉 {len(df5):,} · 1분봉 {len(m1):,} · 반응 후보 {len(keys):,} · 위치 저항 {near_res.mean():.3f} 지지 {near_sup.mean():.3f}")
    rsp = respond(m1, df5, oi, keys[:, 0], keys[:, 1], keys[:, 2])
    look = {(int(k), int(s), int(d)): q for q, (k, s, d) in enumerate(keys)}
    base = int(m1.index[0] // 60_000)
    c1 = m1["c"].to_numpy(float)
    fw = {H: fwd(c1, rsp["ent"], H) for H in HS}
    ent_day = (rsp["ent"] + base) // 1440
    t0 = t.to_numpy().astype("datetime64[m]").astype(np.int64)
    rows = []
    for name, (ev, ct) in groups.items():
        for locname in ("위치 있음", "전체"):
            def pick(arr):
                ks = np.flatnonzero(arr); sd = arr[ks].astype(int)
                if locname == "위치 있음":
                    keep = np.where(sd < 0, near_res[ks], near_sup[ks]); ks, sd = ks[keep], sd[keep]
                return ks, sd, np.array([look[(int(k), int(s), DELAY[name])] for k, s in zip(ks, sd)], dtype=np.int64)
            E, Ct = pick(ev), pick(ct)
            for stage in ("A", "B", "B_oi", "C", "C_oi"):
                for H in HS:
                    for sp in ("TRAIN", "TEST"):
                        for sn in ("전체",) + tuple(s[0] for s in R.SESSIONS) + REG_CUTS:
                            if sn != "전체" and (H != 60 or stage not in ("B", "C")):
                                continue

                            def vals(ks, sd, qq):
                                m = (split[ks] == sp) & cut_mask(sn, ks, sd)
                                ks, sd, qq = ks[m], sd[m], qq[m]
                                if stage == "A":                          # 사건이 알려진 봉의 종가(마지막 1분 종가)에서 바로, 사건 방향
                                    j = (t0[ks] + 4 + 5 * DELAY[name] - base).astype(np.int64)
                                    r = fwd(c1, j, H) * sd; d = (j + base) // 1440
                                else:
                                    sel = rsp["kind"][qq] == (1 if stage.startswith("B") else 2)
                                    if stage == "B_oi":
                                        sel &= rsp["doi"][qq] < 0
                                    if stage == "C_oi":
                                        sel &= rsp["doi"][qq] > 0
                                    r = fw[H][qq][sel] * rsp["dir"][qq][sel]; d = ent_day[qq][sel]
                                ok = np.isfinite(r)
                                return r[ok], d[ok]
                            ve, de = vals(*E); vc, dc = vals(*Ct)
                            row = {"event": name, "loc": locname, "stage": stage, "H": H, "split": sp, "session": sn, "n": len(ve), "n_ctl": len(vc)}
                            if len(ve) >= 30 and len(vc) >= 30:
                                row["hit"] = float((ve > 0).mean())
                                row["mean"], row["m_lo"], row["m_hi"] = mean_ci(ve, de)
                                row["ctl"] = float(vc.mean())
                                row["ex"], row["lo"], row["hi"] = R.block_ci(np.r_[ve, vc], np.r_[np.ones(len(ve), int), np.zeros(len(vc), int)],
                                                                             np.r_[de, dc].astype("datetime64[D]"))
                            rows.append(row)
        print(f"{name} 끝", flush=True)
    res = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "grid.csv", index=False)
    prim = res[(res["loc"] == "위치 있음") & (res.stage == "B") & (res.H == 60) & (res.session == "전체")]
    print(prim[["event", "split", "n", "n_ctl", "hit", "mean", "ctl", "ex", "lo", "hi"]].to_string(index=False))


def selftest() -> None:
    # 1분 격자 60분, 5분봉 12개. 하락 사건 봉 k=5(25~30분). 반응 창 30~44분, 직전 스윙 15~29분.
    t0 = 1_700_000_000_000 // 3_600_000 * 3_600_000
    idx = t0 + np.arange(60) * 60_000
    m1 = pd.DataFrame({"h": 101.0, "l": 99.0, "c": 100.0, "v": 10.0, "tb": 5.0, "delta": 0.0}, index=idx)
    df5 = pd.DataFrame({"high": 101.0, "low": 99.0, "close": 100.0}, index=pd.to_datetime(t0 + np.arange(12) * 300_000, unit="ms"))
    df5.iloc[5, df5.columns.get_loc("high")] = 103.0
    oi = np.full(12, 1000.0)
    m1.iloc[33, m1.columns.get_loc("c")] = 98.5; m1.iloc[30:34, m1.columns.get_loc("delta")] = -5.0     # 33분: 스윙 저가(99) 이탈 + 매도 우위
    r = respond(m1, df5, oi, np.array([5]), np.array([-1]))
    assert r["kind"][0] == 1 and r["ent"][0] == 33 and r["dir"][0] == -1, r
    m1b = m1.copy(); m1b.iloc[31, m1b.columns.get_loc("c")] = 103.5                                  # 31분: 사건 고가 돌파가 먼저 → 실패
    r = respond(m1b, df5, oi, np.array([5]), np.array([-1]))
    assert r["kind"][0] == 2 and r["ent"][0] == 31 and r["dir"][0] == 1, r
    m1c = m1.copy(); m1c.iloc[30:34, m1c.columns.get_loc("delta")] = 5.0                             # 이탈은 했는데 매수 우위 → 반전 아님
    assert respond(m1c, df5, oi, np.array([5]), np.array([-1]))["kind"][0] == 0
    oi2 = oi.copy(); oi2[5] = 900.0                                                                 # 진입(33분 → 봉 6)의 직전 스탬프 5 = 900 < 사건 직전 스탬프 4
    assert respond(m1, df5, oi2, np.array([5]), np.array([-1]))["doi"][0] == -100.0
    r = respond(m1, df5, oi, np.array([5]), np.array([-1]), np.array([1]))                           # 지연 1 → 35분부터: 33분 이탈은 못 본다
    assert r["kind"][0] == 0 or r["ent"][0] >= 35, r
    # 레짐: 미래 봉을 안 본다 · 펀딩은 정산 시각부터, 8h5m 넘으면 모름 · 만기 창
    n = 3000; ix = pd.date_range("2025-01-03", periods=n, freq="5min")                             # 금요일 00:00 시작
    rng = np.random.default_rng(0); cc = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    d5 = pd.DataFrame({"high": cc * 1.001, "low": cc * 0.999, "close": cc}, index=ix)
    fund = pd.DataFrame({"calc_time": pd.to_datetime(["2025-01-03 00:00", "2025-01-03 08:00"]), "last_funding_rate": [1e-4, 3e-4]})
    ois = 1000 + np.cumsum(rng.normal(0, 1, n))
    g = regimes(d5, ois, fund)
    d5b = d5.copy(); d5b.iloc[2500:] *= 1.5; oib = ois.copy(); oib[2500:] += 500
    g2 = regimes(d5b, oib, fund)
    assert all((np.asarray(g[k])[:2500] == np.asarray(g2[k])[:2500]).all() for k in g), "레짐이 미래 봉을 봤다"
    assert g["펀딩"][94] == "기본" and g["펀딩"][95] == "기본 초과" and g["펀딩"][192] == "기본 초과" and g["펀딩"][193] == ""
    assert g["만기"][47] == "그밖" and g["만기"][48] == "±4h" and g["만기"][143] == "±4h" and g["만기"][144] == "그밖"
    assert set(np.unique(g["추세"][200:])) <= {-1, 0, 1} and (g["추세"][:143] == 9).all()
    c = np.arange(200, dtype=float) + 100
    assert abs(fwd(c, np.array([10]), 30)[0] - (140 / 110 - 1) * 1e4) < 1e-9 and np.isnan(fwd(c, np.array([190]), 30)[0])
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("tmp/fp_event_response_20260930"))
    ap.add_argument("--start", default=None)
    ap.add_argument("--split", default=None)
    a = ap.parse_args()
    R.START = pd.Timestamp(a.start) if a.start else R.START
    R.SPLIT = pd.Timestamp(a.split) if a.split else R.SPLIT
    if a.selftest:
        selftest()
    else:
        run(a.data or R.default_data_dir(), a.out)
