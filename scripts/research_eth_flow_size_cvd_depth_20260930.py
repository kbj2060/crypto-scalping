"""대형 체결·CVD 기울기/가속·깊이/거래량 비 — 대시보드에 넣을 만한가 (2026-09-30, 사용자 «트레이딩에 도움이 되는가»).

모델 학습이 아니라 조건식/단순 지표를 과거에 적용해 그 뒤 가격을 센다. 사용자는 2024년 이전 데이터를 신뢰하지 않는다
⇒ 평가는 2025(앞)·2026(뒤) 두 해만. 2024-10~12 는 롤링 창 예열에만 쓴다.

사전 고정(결과 보기 전):
  데이터   1분봉 klines1m(델타 = 2·tb − v) · 5분 aggfeat(ts = 봉 시작, tf_cvd↔봉 CVD +1.0000 을 실행 때 다시 잰다) ·
           bookDepth(≈30초 스냅샷, n−k/nk = −k%/+k% 안 누적 명목 $, UTC).
  결정     5분봉 종가(봉 시작+5분). 지표는 그 봉까지(1분 지표는 그 5분봉의 마지막 1분까지). 깊이는 결정 시각 «이전» 마지막 스냅샷
           (같은 시각 제외, 120초보다 묵으면 결측).
  라벨     방향 = 결정 종가 → 6·12봉 뒤 종가(30·60분, bp) = 다음 봉부터. 크기 = 다음 60개 1분 수익의 실현변동성 log RV60(bp) ·
           보조 log(|60분 수익| + 1bp).
  지표 1   대형 체결(W = 5·30·60분): big명목 = bigshare·meansz·ntrade.
           1a bigimb_W = Σ(bigimb·big명목)/Σbig명목 (방향, 주장 +: 대형 매수 우위 → 상승 «큰손을 따라가라»)
           1b bigshare_W = Σbig명목/Σ명목 (크기, 주장 +)
           1c meansz_W = log(Σ명목/Σ건수 ÷ 뒤로 7일 같은 값) (크기, 주장 +)
  지표 2   CVD 기울기_N = 직전 N분(15·30·60) 1분 누적 델타의 OLS 기울기 ÷ 창의 분당 평균 거래량. 가속_N = 기울기 − 직전 N분 창의 기울기.
           가격 기울기_N = 같은 창 log 종가 OLS 기울기.
           2a 기울기(방향 +) · 2b 가속(방향 +) · 2c «가속 붙은 추세»: x = sign(기울기)·가속, 결과 = sign(기울기)·수익(+) ·
           2d «확인»: sign(CVD 기울기) = sign(가격 기울기) 봉 vs 다른 봉, 결과 = sign(가격 기울기)·수익(확인 − 불일치 > 0).
  지표 3   3a 깊이/거래량 = log((n−1 + n1)/직전 60분 거래대금), ±2% 는 n−2 + n2 (크기, 주장 −: 얇을수록 다음 움직임 큼)
           3b 깊이 비 = (n−1 − n1)/(n−1 + n1), ±2% 도 (방향, 주장 +: 매수 깊이 우위 → 상승, 부차)
  방향 효과 상위 20% − 하위 20% 평균 수익(bp). 문턱 = 뒤로 30일(자기 봉 제외) 20·80분위 = 그 시점에 쓸 수 있는 규칙. IC = 그 해 스피어만(보조).
  통제      같은 상위·하위 행에서 수익 ~ 1 + 상위표시 + [과거수익 5·30·60분, 전체 CVD(Σ델타/Σ거래량) 5·30·60분] OLS 의 상위표시 계수.
           2c·2d 는 통제변수에도 같은 부호를 곱한다. 사라지면 «가격(또는 전체 CVD)의 그림자».
  크기 효과 log RV60 ~ 1 + [log RV 1h·1d·1w(HAR), log 직전봉 폭] + z(지표) 의 z 계수(1σ 당 log 단위 ≈ %)와 ΔR². 통제 없는 스피어만은 참고.
  CI       일 단위 포아송 블록 부트스트랩 B=1000(95%) — 겹치는 라벨로 표본이 부풀지 않게.
  판정     통제 후 효과가 2025·2026 둘 다 같은 부호로 CI 0 배제 → 통과 · 한 해만 → 한 해만 · 그 밖 불합격.
           원판(통제 전) 통과인데 통제 후 아니면 «가격의 그림자». 부호가 주장과 반대면 «(주장 반대)».
           방향 통과여도 통제 후 효과/2(한쪽 건당) < 1.4bp → «정보는 있으나 매매 불가», 1.4~5bp → «메이커로만».
           크기 통과여도 ΔR² < 0.005(어느 해든) → «통계적으로만 — 사이징 실익 미미».
  누수 점검 ① 지표를 한 봉 더 늦춘 원판 효과 ② 1분 종가 = 5분 종가(매핑) ③ 깊이 스냅샷 < 결정 시각 전수 assert
           ④ aggfeat 대형 문턱은 빌더(build_aggtrades_orderflow_features_20260915.py)가 «그날 전체 체결의 99분위»로 잡아 장중 미래참조가
           섞일 수 있다 ⇒ 1a·1b 는 UTC 0–6시(그날 미래가 대부분) vs 18–24시(거의 과거) 계수를 나란히 본다.

사후 추가(첫 실행 뒤, 보조 — 판정 기준은 위 그대로): 원판 IC 가 두 해 CI 0 배제인 지표가 있어 «그림자» 여부를 가르려고
  통제 IC(지표·수익·통제변수를 순위화해 통제변수로 잔차화한 상관)와 통제 bp/σ(수익 ~ z(지표) + z(통제), 전 행 OLS 계수)를 붙였다.

실행: python scripts/research_eth_flow_size_cvd_depth_20260930.py [--selftest] [--data DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research_eth_fp_pattern_lookback_20260930 as R  # noqa: E402
import research_eth_fp_event_response_20260930 as E  # noqa: E402

WARM = pd.Timestamp("2024-10-01")
YEARS = (("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")),
         ("2026", pd.Timestamp("2026-01-01"), pd.Timestamp("2027-01-01")))
HS = (6, 12)                      # 5분봉 수 = 30·60분
QWIN = 8640                       # 뒤로 30일(5분봉)
DEPTH_TOL = pd.Timedelta("120s")
MAKER, TAKER = 1.4, 5.0
CTRL_DIR = ["ret5", "ret30", "ret60", "cvd5", "cvd30", "cvd60"]
CTRL_MAG = ["lrv1h", "lrv1d", "lrv1w", "lrange"]


# ───────────────────────── 계산 조각(자체점검 대상) ─────────────────────────
def win_slope(y: np.ndarray, N: int) -> np.ndarray:
    """끝이 t 인 직전 N점의 OLS 기울기(점당). 창에 NaN 이 있으면 NaN."""
    x = np.arange(N) - (N - 1) / 2
    out = np.full(len(y), np.nan)
    out[N - 1:] = np.convolve(y, (x / (x @ x))[::-1], "valid")
    return out


def rv_back(logc: np.ndarray, W: int) -> np.ndarray:
    """분 j 까지(포함) 직전 W개 1분 수익의 실현변동성(bp)."""
    r2 = pd.Series(np.diff(logc, prepend=np.nan) ** 2)
    return np.sqrt(r2.rolling(W, min_periods=int(W * .9)).sum().to_numpy()) * 1e4


def rv_fwd(logc: np.ndarray, W: int) -> np.ndarray:
    """분 j 의 다음 W분 실현변동성(bp) = r[j+1..j+W]. 결정 분 자신의 수익 r[j] 는 안 들어간다."""
    r2 = pd.Series(np.diff(logc, prepend=np.nan) ** 2)
    return np.sqrt(r2.rolling(W, min_periods=int(W * .9)).sum().shift(-W).to_numpy()) * 1e4


def last_min_pos(b5: pd.DatetimeIndex, t0_ms: int) -> np.ndarray:
    """5분봉(시작 b)의 마지막 1분(b+4분)의 1분 격자 위치."""
    return ((b5.asi8 // 10**6) + 240_000 - t0_ms) // 60_000


def depth_at(dec: pd.DatetimeIndex, snaps: pd.DataFrame) -> pd.DataFrame:
    """결정 시각 «이전»(같은 시각 제외) 마지막 스냅샷. DEPTH_TOL 보다 묵으면 NaN."""
    right = snaps.rename_axis("snap_ts").reset_index()
    return pd.merge_asof(pd.DataFrame({"dec": dec}), right, left_on="dec", right_on="snap_ts",
                         direction="backward", allow_exact_matches=False, tolerance=DEPTH_TOL)


def qgroups(x: pd.Series) -> np.ndarray:
    """뒤로 30일(자기 봉 제외) 20·80분위 → +1 상위 · −1 하위 · 0 중간 · NaN 결측."""
    r = x.rolling(QWIN, min_periods=QWIN // 2)
    lo, hi = r.quantile(.2).shift(1).to_numpy(), r.quantile(.8).shift(1).to_numpy()
    xv = x.to_numpy(float)
    g = np.where(xv >= hi, 1.0, np.where(xv <= lo, -1.0, 0.0))
    g[~(np.isfinite(xv) & np.isfinite(lo) & np.isfinite(hi))] = np.nan
    return g


def ols_boot(X: np.ndarray, y: np.ndarray, day: np.ndarray, k: int, B: int = 1000, seed: int = 5) -> dict:
    """y ~ X(절편 포함) OLS. 열 k 계수와 «k 를 뺀 모형 대비 ΔR²»의 일 블록 포아송 부트스트랩(95%). 행은 시간순이어야 한다."""
    _, inv = np.unique(day, return_inverse=True)
    st = np.r_[0, np.flatnonzero(np.diff(inv)) + 1]
    XX = np.add.reduceat(X[:, :, None] * X[:, None, :], st)
    Xy = np.add.reduceat(X * y[:, None], st)
    yy, sy, n = np.add.reduceat(y * y, st), np.add.reduceat(y, st), np.diff(np.r_[st, len(y)])
    W = np.vstack([np.ones(len(st)), np.random.default_rng(seed).poisson(1.0, (B, len(st)))])
    A, c = np.einsum("bd,dij->bij", W, XX), W @ Xy
    keep = [i for i in range(X.shape[1]) if i != k]

    def sse(A, c):
        b = np.linalg.solve(A, c[..., None])[..., 0]
        return b, W @ yy - (b * c).sum(1)
    b, s_full = sse(A, c)
    _, s_r = sse(A[:, keep][:, :, keep], c[:, keep])
    sst = W @ yy - (W @ sy) ** 2 / (W @ n)
    dr2 = (s_r - s_full) / sst
    blo, bhi = np.nanpercentile(b[1:, k], [2.5, 97.5])
    dlo, dhi = np.nanpercentile(dr2[1:], [2.5, 97.5])
    return {"b": float(b[0, k]), "lo": float(blo), "hi": float(bhi),
            "dr2": float(dr2[0]), "dr2_lo": float(dlo), "dr2_hi": float(dhi), "r2_ctrl": float(1 - s_r[0] / sst[0])}


def corr_ci(a: np.ndarray, b: np.ndarray, day: np.ndarray, B: int = 1000, seed: int = 3) -> tuple[float, float, float]:
    """피어슨 상관(순위를 넣으면 스피어만)의 일 블록 포아송 부트스트랩(95%)."""
    _, inv = np.unique(day, return_inverse=True)
    S = np.stack([np.bincount(inv, w) for w in (np.ones_like(a), a, b, a * a, b * b, a * b)], 1)
    W = np.vstack([np.ones(S.shape[0]), np.random.default_rng(seed).poisson(1.0, (B, S.shape[0]))])
    n, sa, sb, saa, sbb, sab = (W @ S).T
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (sab - sa * sb / n) / np.sqrt((saa - sa * sa / n) * (sbb - sb * sb / n))
    lo, hi = np.nanpercentile(r[1:], [2.5, 97.5])
    return float(r[0]), float(lo), float(hi)


def z(a: np.ndarray) -> np.ndarray:
    return np.clip((a - np.nanmean(a)) / np.nanstd(a), -5, 5)


def call(res: dict[str, tuple], sign: int) -> str:
    """두 해 (점, 하한, 상한) → 통과/한 해만/불합격(+주장 반대)."""
    sig = [np.sign(p) if (lo > 0 or hi < 0) else 0 for p, lo, hi in res.values()]
    if sig[0] != 0 and sig[0] == sig[1]:
        return "통과" + ("" if sig[0] == sign else "(주장 반대)")
    return "한 해만" if any(sig) else "불합격"


# ───────────────────────── 데이터 ─────────────────────────
def build(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    root = data_dir.parent
    df5 = R.load_5m(data_dir)
    m1 = E.load_1m(data_dir)
    t0 = int(m1.index[0])
    c1 = m1["c"].to_numpy(float); lc1 = np.log(c1); v1 = m1["v"].to_numpy(float); d1 = m1["delta"].to_numpy(float)
    cum = np.nancumsum(d1); cum[np.isnan(d1)] = np.nan                                  # 빠진 분이 든 창은 NaN
    f1 = {}
    for N in (15, 30, 60):
        cs = win_slope(cum, N) / pd.Series(v1).rolling(N).mean().to_numpy()
        f1[f"cvdslope{N}"] = cs
        f1[f"accel{N}"] = cs - np.r_[np.full(N, np.nan), cs[:-N]]
        f1[f"pxslope{N}"] = win_slope(lc1, N)
    f1["qv60"] = pd.Series(v1 * c1).rolling(60).sum().to_numpy()
    f1["lrv1h"], f1["lrv1d"], f1["lrv1w"] = (np.log(rv_back(lc1, W)) for W in (60, 1440, 10080))
    f1["lrvnext60"] = np.log(rv_fwd(lc1, 60))
    f1["c1m"] = c1
    pos = last_min_pos(df5.index, t0)
    okp = (pos >= 0) & (pos < len(c1))
    F = pd.DataFrame({k: np.where(okp, v[np.clip(pos, 0, len(c1) - 1)], np.nan) for k, v in f1.items()}, index=df5.index)
    c5 = df5["close"].to_numpy(float)
    both = np.isfinite(c5) & np.isfinite(F["c1m"].to_numpy())
    assert np.allclose(c5[both], F["c1m"].to_numpy()[both]), "1분 → 5분 매핑이 어긋났다"
    lc5 = np.log(c5)
    for k, nm in ((1, "5"), (6, "30"), (12, "60")):
        F[f"ret{nm}"] = (lc5 - np.r_[np.full(k, np.nan), lc5[:-k]]) * 1e4
        F[f"cvd{nm}"] = df5["delta"].rolling(k).sum() / df5["vol"].rolling(k).sum()
    F["lrange"] = np.log(np.maximum((df5["high"] - df5["low"]) / df5["close"] * 1e4, 0.5))
    for H in HS:
        F[f"fwd{H}"] = R.fwd_bp(c5, H, 0)
    F["labsfwd12"] = np.log(np.abs(F["fwd12"]) + 1)

    # 체결 집계
    ag = pd.concat(pd.read_parquet(f) for f in sorted((root / "aggfeat").glob("ETHUSDT_*.parquet"))).sort_index()
    ag = ag[~ag.index.duplicated()].reindex(df5.index)
    av = ag["tf_meansz"] * ag["tf_ntrade"]
    bigav = ag["tf_bigshare"] * av
    bigsv = pd.Series(np.where(bigav == 0, 0.0, ag["tf_bigimb"] * bigav), index=df5.index)
    sz7 = av.rolling(2016, min_periods=1500).sum() / ag["tf_ntrade"].rolling(2016, min_periods=1500).sum()
    for k, nm in ((1, "5"), (6, "30"), (12, "60")):
        sa = lambda s: s.rolling(k).sum()  # noqa: E731
        F[f"bigimb{nm}"] = sa(bigsv) / sa(bigav)
        F[f"bigshare{nm}"] = sa(bigav) / sa(av)
        F[f"meansz{nm}"] = np.log(sa(av) / sa(ag["tf_ntrade"]) / sz7)
    ev = df5.index >= pd.Timestamp("2025-01-01")
    agal = {int(o): float(pd.Series(ag["tf_cvd"].to_numpy()[ev]).corr(pd.Series(F["cvd5"].shift(o).to_numpy()[ev])))
            for o in (-1, 0, 1)}
    assert agal[0] > 0.99 and agal[0] > max(agal[-1], agal[1]), f"aggfeat 정렬 이상 {agal}"

    # 호가 깊이
    fs = [f for f in sorted(root.glob("ETHUSDT_bookDepth_*.parquet")) if f.stem[-10:] >= "2024-09-15"]
    dp = pd.concat(pd.read_parquet(f, columns=["n-2", "n-1", "n1", "n2"]) for f in fs).sort_index()
    dp = dp[~dp.index.duplicated()]
    dec = df5.index + pd.Timedelta("5min")
    m = depth_at(dec, dp)
    got = m["snap_ts"].notna().to_numpy()
    assert (m["snap_ts"][got] < m["dec"][got]).all(), "깊이 스냅샷이 결정 시각 이후"
    age = (m["dec"] - m["snap_ts"]).dt.total_seconds().to_numpy()
    b1, a1, b2, a2 = (m[k].to_numpy(float) for k in ("n-1", "n1", "n-2", "n2"))
    F["depthratio1"] = np.log((b1 + a1) / F["qv60"].to_numpy())
    F["depthratio2"] = np.log((b2 + a2) / F["qv60"].to_numpy())
    F["depthimb1"] = (b1 - a1) / (b1 + a1)
    F["depthimb2"] = (b2 - a2) / (b2 + a2)
    # 시간대 점검: 분별 깊이 비 변화 ↔ 같은 분 1분 수익(시차 0 이 최대여야 UTC 정렬)
    di = np.log(dp["n-1"] / dp["n1"]).resample("1min").last().diff()
    r1 = pd.Series(np.diff(lc1, prepend=np.nan), index=pd.to_datetime(m1.index, unit="ms"))
    j = r1.index >= pd.Timestamp("2025-01-01")
    tz = {int(L): float(di.corr(r1[j].shift(L))) for L in (-480, -1, 0, 1, 480)}

    F["dec"] = dec
    F = F[F.index >= WARM].replace([np.inf, -np.inf], np.nan)    # 거래량 0 분(분모 0)·변동 0 창(log 0)
    ev2 = F["dec"] >= pd.Timestamp("2025-01-01")
    meta = {"aggfeat_offset_corr(ts=봉시작이면 0 최대)": agal,
            "depth_tz_lag_corr(1분, 0 최대면 UTC)": tz,
            "depth_age_s_median": float(np.nanmedian(age[df5.index >= pd.Timestamp("2025-01-01")])),
            "depth_coverage_2025+": float(np.isfinite(F["depthimb1"][ev2]).mean()),
            "depth_n-1_mean_usd(2025+)": float(np.nanmean(b1[df5.index >= pd.Timestamp("2025-01-01")])),
            "aggfeat_coverage_2025+": float(np.isfinite(F["bigshare5"][ev2]).mean()),
            "eval_end": str(F["dec"][np.isfinite(F["fwd12"])].max())}
    return F, meta


# ───────────────────────── 검정 ─────────────────────────
def yr_masks(F: pd.DataFrame):
    for nm, a, b in YEARS:
        yield nm, ((F["dec"] >= a) & (F["dec"] < b)).to_numpy()


def dir_rows(F: pd.DataFrame, item: str, claim: str, xcol: str, x: pd.Series | None = None,
             m: np.ndarray | None = None) -> list[dict]:
    """방향: 상위 20% − 하위 20%(뒤로 30일 문턱) 원판 · 통제 후 · 한 봉 늦춤 · IC."""
    x = F[xcol] if x is None else x
    m = np.ones(len(F)) if m is None else m
    g = qgroups(x)
    glag = np.r_[np.nan, g[:-1]]
    mlag = np.r_[np.nan, m[:-1]]
    day = F["dec"].to_numpy().astype("datetime64[D]")
    C = F[CTRL_DIR].to_numpy(float) * m[:, None]
    out = []
    for H in HS:
        y = m * F[f"fwd{H}"].to_numpy(float)
        ylag = mlag * F[f"fwd{H}"].to_numpy(float)
        row = {"항목": item, "주장": claim, "지표": xcol, "H분": H * 5}
        raw, ctl, lag, ic = {}, {}, {}, {}
        for yn, ys in yr_masks(F):
            ok = ys & np.isfinite(y) & np.isfinite(g) & np.isfinite(C).all(1) & (m != 0)
            sel = ok & (g != 0)
            row[f"n{yn}"] = int(sel.sum())
            top = (g[sel] == 1).astype(int)
            raw[yn] = R.block_ci(y[sel], top, day[sel])
            X = np.column_stack([np.ones(sel.sum()), top, np.apply_along_axis(z, 0, C[sel])])
            o = ols_boot(X, y[sel], day[sel], 1)
            ctl[yn] = (o["b"], o["lo"], o["hi"])
            sl = ys & np.isfinite(ylag) & np.isfinite(glag) & (glag != 0) & (mlag != 0)
            lag[yn] = R.block_ci(ylag[sl], (glag[sl] == 1).astype(int), day[sl])
            xv = x.to_numpy(float)
            xi = ys & np.isfinite(y) & np.isfinite(xv) & np.isfinite(C).all(1) & (m != 0)
            rk = lambda a: pd.Series(a).rank().to_numpy()  # noqa: E731
            rx, ry = rk(xv[xi]), rk(y[xi])
            Rc = np.column_stack([np.ones(xi.sum())] + [rk(col) for col in C[xi].T])
            res = lambda v: v - Rc @ np.linalg.lstsq(Rc, v, rcond=None)[0]  # noqa: E731
            o = ols_boot(np.column_stack([np.ones(xi.sum()), z(xv[xi]), np.apply_along_axis(z, 0, C[xi])]), y[xi], day[xi], 1)
            ic[yn] = (corr_ci(rx, ry, day[xi]), corr_ci(res(rx), res(ry), day[xi]), (o["b"], o["lo"], o["hi"]))
        out.append(finish_dir(row, raw, ctl, lag, ic))
    return out


def finish_dir(row: dict, raw: dict, ctl: dict, lag: dict, ic: dict | None) -> dict:
    f = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f},{t[2]:+.2f}]"  # noqa: E731
    for yn in raw:
        row[f"원판{yn}"] = f(raw[yn]); row[f"통제후{yn}"] = f(ctl[yn]); row[f"한봉늦춤{yn}"] = f"{lag[yn][0]:+.2f}"
        if ic:
            for nm, t in zip(("IC", "통제IC", "통제bp/σ"), ic[yn]):
                row[f"{nm}{yn}"] = f"{t[0]:+.4f} [{t[1]:+.4f},{t[2]:+.4f}]" if nm != "통제bp/σ" else f(t)
    if ic:
        row["IC판정"] = call({y: ic[y][0] for y in ic}, 1)
        row["통제IC판정"] = call({y: ic[y][1] for y in ic}, 1)
    rj, cj = call(raw, 1), call(ctl, 1)
    row["판정원판"], row["판정통제"] = rj, cj
    edge = min(abs(ctl[y][0]) for y in ctl) / 2
    if cj.startswith("통과"):
        row["최종"] = cj + ("" if edge >= TAKER else (" — 메이커로만" if edge >= MAKER else " — 정보는 있으나 매매 불가"))
    elif rj.startswith("통과") and not cj.startswith("통과"):
        row["최종"] = "가격의 그림자"
    else:
        row["최종"] = cj
    row["건당(통제/2,bp)"] = round(edge, 2)
    return row


def confirm_rows(F: pd.DataFrame) -> list[dict]:
    """2d: CVD 기울기와 가격 기울기가 같은 방향(확인) vs 다름 — 결과 = sign(가격 기울기)·수익."""
    day = F["dec"].to_numpy().astype("datetime64[D]")
    out = []
    for N in (15, 30, 60):
        s = np.sign(F[f"pxslope{N}"].to_numpy(float))
        agree = (np.sign(F[f"cvdslope{N}"].to_numpy(float)) == s).astype(int)
        valid = np.isfinite(F[f"pxslope{N}"].to_numpy(float)) & np.isfinite(F[f"cvdslope{N}"].to_numpy(float)) & (s != 0)
        C = F[CTRL_DIR].to_numpy(float) * s[:, None]
        slag, alag, vlag = np.r_[0, s[:-1]], np.r_[0, agree[:-1]], np.r_[False, valid[:-1]]
        for H in HS:
            fw = F[f"fwd{H}"].to_numpy(float)
            y, ylag = s * fw, slag * fw
            row = {"항목": "2d 확인", "주장": "가격·CVD 기울기 같은 방향이면 더 이어짐", "지표": f"agree{N}", "H분": H * 5}
            raw, ctl, lag = {}, {}, {}
            for yn, ys in yr_masks(F):
                sel = ys & valid & np.isfinite(y) & np.isfinite(C).all(1)
                row[f"n{yn}"] = int(sel.sum())
                raw[yn] = R.block_ci(y[sel], agree[sel], day[sel])
                X = np.column_stack([np.ones(sel.sum()), agree[sel], np.apply_along_axis(z, 0, C[sel])])
                o = ols_boot(X, y[sel], day[sel], 1)
                ctl[yn] = (o["b"], o["lo"], o["hi"])
                sl = ys & vlag & np.isfinite(ylag)
                lag[yn] = R.block_ci(ylag[sl], alag[sl], day[sl])
                a = sel & (agree == 1)
                mu = E.mean_ci(y[a], day[a])
                row[f"확인봉평균{yn}"] = f"{mu[0]:+.2f} [{mu[1]:+.2f},{mu[2]:+.2f}]"
            out.append(finish_dir(row, raw, ctl, lag, None))
    return out


def mag_rows(F: pd.DataFrame, item: str, claim: str, xcol: str, sign: int, tod: bool = False) -> list[dict]:
    """크기: log RV60(주) · log|r60|(보조) ~ HAR + 직전봉 폭 + z(지표). 계수·ΔR²·원판 스피어만·한 봉 늦춤."""
    day = F["dec"].to_numpy().astype("datetime64[D]")
    hour = F["dec"].dt.hour.to_numpy()
    xs = F[xcol].to_numpy(float)
    xlag = np.r_[np.nan, xs[:-1]]
    C = F[CTRL_MAG].to_numpy(float)
    out = []
    for tgt in ("lrvnext60", "labsfwd12"):
        y = F[tgt].to_numpy(float)
        row = {"항목": item, "주장": claim, "지표": xcol, "목표": "logRV60" if tgt == "lrvnext60" else "log|r60|"}
        res, sp = {}, {}
        dr2 = []
        for yn, ys in yr_masks(F):
            ok = ys & np.isfinite(xs) & np.isfinite(y) & np.isfinite(C).all(1)
            row[f"n{yn}"] = int(ok.sum())
            X = np.column_stack([np.ones(ok.sum()), np.apply_along_axis(z, 0, C[ok]), z(xs[ok])])
            o = ols_boot(X, y[ok], day[ok], X.shape[1] - 1)
            res[yn] = (o["b"], o["lo"], o["hi"])
            dr2.append(o["dr2"])
            row[f"계수{yn}"] = f"{o['b']:+.4f} [{o['lo']:+.4f},{o['hi']:+.4f}]"
            row[f"ΔR2_{yn}"] = f"{o['dr2']:.4f} [{o['dr2_lo']:.4f},{o['dr2_hi']:.4f}]"
            row[f"R2통제{yn}"] = round(o["r2_ctrl"], 4)
            sp[yn] = corr_ci(pd.Series(xs[ok]).rank().to_numpy(), pd.Series(y[ok]).rank().to_numpy(), day[ok])
            row[f"원판ρ{yn}"] = f"{sp[yn][0]:+.4f} [{sp[yn][1]:+.4f},{sp[yn][2]:+.4f}]"
            ol = ys & np.isfinite(xlag) & np.isfinite(y) & np.isfinite(C).all(1)
            Xl = np.column_stack([np.ones(ol.sum()), np.apply_along_axis(z, 0, C[ol]), z(xlag[ol])])
            row[f"한봉늦춤{yn}"] = f"{ols_boot(Xl, y[ol], day[ol], Xl.shape[1] - 1, B=50)['b']:+.4f}"
            if tod:
                for lab, hm in (("0-6시", hour < 6), ("18-24시", hour >= 18)):
                    oo = ok & hm
                    Xt = np.column_stack([np.ones(oo.sum()), np.apply_along_axis(z, 0, C[oo]), z(xs[oo])])
                    row[f"계수{lab}_{yn}"] = f"{ols_boot(Xt, y[oo], day[oo], Xt.shape[1] - 1, B=50)['b']:+.4f}"
        row["판정원판"] = call(sp, sign)
        cj = call(res, sign)
        row["판정통제"] = cj
        if cj.startswith("통과"):
            row["최종"] = cj + ("" if min(dr2) >= 0.005 else " — 통계적으로만, 사이징 실익 미미")
        elif row["판정원판"].startswith("통과") and not cj.startswith("통과"):
            row["최종"] = "HAR-RV 의 그림자"
        else:
            row["최종"] = cj
        out.append(row)
    return out


def tod_dir(F: pd.DataFrame, xcol: str) -> dict:
    """대형 문턱 장중 미래참조 점검: 0–6시 vs 18–24시 원판 효과(60분)."""
    g = qgroups(F[xcol])
    y = F["fwd12"].to_numpy(float)
    day = F["dec"].to_numpy().astype("datetime64[D]")
    hour = F["dec"].dt.hour.to_numpy()
    out = {}
    for yn, ys in yr_masks(F):
        for lab, hm in (("0-6시", hour < 6), ("18-24시", hour >= 18)):
            sel = ys & hm & np.isfinite(y) & np.isfinite(g) & (g != 0)
            p, lo, hi = R.block_ci(y[sel], (g[sel] == 1).astype(int), day[sel])
            out[f"{yn} {lab}"] = f"{p:+.2f} [{lo:+.2f},{hi:+.2f}] n={int(sel.sum())}"
    return out


def run(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    F, meta = build(data_dir)
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    d = []
    for W in ("5", "30", "60"):
        d += dir_rows(F, "1a 대형 체결 방향", "대형 매수 우위 → 상승", f"bigimb{W}")
    for N in (15, 30, 60):
        d += dir_rows(F, "2a CVD 기울기", "기울기 방향으로 이어짐", f"cvdslope{N}")
    for N in (15, 30, 60):
        d += dir_rows(F, "2b 델타 가속", "가속 방향으로 이어짐", f"accel{N}")
    for N in (15, 30, 60):
        s = np.sign(F[f"cvdslope{N}"].to_numpy(float))
        d += dir_rows(F, "2c 가속 붙은 기울기", "기울기 쪽 가속이면 더 이어짐", f"sgnaccel{N}",
                      x=pd.Series(s * F[f"accel{N}"].to_numpy(float), index=F.index), m=s)
    d += confirm_rows(F)
    for k in ("1", "2"):
        d += dir_rows(F, "3b 깊이 비(부차)", "매수 깊이 우위 → 상승", f"depthimb{k}")
    mg = []
    for W in ("5", "30", "60"):
        mg += mag_rows(F, "1b 대형 비중", "대형 비중 크면 다음 움직임 큼", f"bigshare{W}", +1, tod=True)
    for W in ("5", "30", "60"):
        mg += mag_rows(F, "1c 평균 체결 크기", "평균 체결 크면 다음 움직임 큼", f"meansz{W}", +1)
    for k in ("1", "2"):
        mg += mag_rows(F, "3a 깊이/거래량", "비율 낮으면 다음 움직임 큼", f"depthratio{k}", -1)
    meta["tod_bigimb60(원판 60분)"] = tod_dir(F, "bigimb60")
    D, M = pd.DataFrame(d), pd.DataFrame(mg)
    D.to_csv(out_dir / "direction.csv", index=False)
    M.to_csv(out_dir / "magnitude.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    pd.set_option("display.width", 400, "display.max_columns", 60, "display.max_colwidth", 40)
    print(D.to_string()); print(M.to_string())
    print(json.dumps(meta["tod_bigimb60(원판 60분)"], ensure_ascii=False, indent=1))


# ───────────────────────── 자체점검 ─────────────────────────
def selftest() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=300).cumsum()
    s = win_slope(y, 15)
    assert np.isnan(s[13]) and abs(s[100] - np.polyfit(np.arange(15), y[86:101], 1)[0]) < 1e-9
    y2 = y.copy(); y2[101] += 50
    assert win_slope(y2, 15)[100] == s[100], "기울기가 미래 점을 봤다"
    # 방향 라벨은 결정 다음 봉부터: 결정 봉 i 의 자기 움직임은 안 들어가고 i+1 의 움직임은 들어간다
    c = 100 * np.exp(rng.normal(0, 1e-3, 500).cumsum())
    i, H = 200, 12
    f0 = R.fwd_bp(c, H, 0)[i]
    ca = c.copy(); ca[:i] *= 1.01                     # 봉 i 자신의 수익만 바뀜
    cb = c.copy(); cb[i + 1:] *= 1.01                 # 봉 i+1 에 점프
    cc = c.copy(); cc[i + H + 1:] *= 1.01             # 지평 밖
    assert abs(R.fwd_bp(ca, H, 0)[i] - f0) < 1e-9 and abs(R.fwd_bp(cc, H, 0)[i] - f0) < 1e-9
    assert abs(R.fwd_bp(cb, H, 0)[i] - f0) > 50
    # 크기 라벨(RV60): 결정 분 j 의 수익은 빼고 j+1..j+60 만
    lc = np.log(c); j = 200
    r0 = rv_fwd(lc, 60)[j]
    for shift, changes in ((j, False), (j + 1, True), (j + 60, True), (j + 61, False)):
        l2 = lc.copy(); l2[shift:] += 0.05
        assert (abs(rv_fwd(l2, 60)[j] - r0) > 1) == changes, f"RV 라벨 경계 {shift}"
    l2 = lc.copy(); l2[j:] += 0.05
    assert rv_back(l2, 60)[j] > rv_back(lc, 60)[j] + 1, "HAR 통제가 결정 분을 빼먹었다"
    # 1분 → 5분: 봉 시작 b 의 결정 분 = b+4분
    t0 = int(pd.Timestamp("2025-01-01").value // 10**6)
    b5 = pd.DatetimeIndex(["2025-01-01 00:00", "2025-01-01 00:05"])
    assert list(last_min_pos(b5, t0)) == [4, 9]
    # 깊이: 결정 시각 정각 스냅샷은 버리고, 120초보다 묵으면 결측
    dec = pd.DatetimeIndex(["2025-01-01 00:05", "2025-01-01 00:10", "2025-01-01 00:20"])
    snaps = pd.DataFrame({"n-1": [1.0, 2.0, 3.0, 4.0]}, index=pd.DatetimeIndex(
        ["2025-01-01 00:04:50", "2025-01-01 00:05:00", "2025-01-01 00:09:59", "2025-01-01 00:15:00"]))
    m = depth_at(dec, snaps)
    assert m["n-1"].iloc[0] == 1.0 and m["n-1"].iloc[1] == 3.0 and np.isnan(m["n-1"].iloc[2])
    # 뒤로 분위 문턱은 자기 봉을 안 본다
    x = pd.Series(rng.normal(size=QWIN + 10))
    g0 = qgroups(x); x2 = x.copy(); x2.iloc[-1] = 99
    assert g0[-2] == qgroups(x2)[-2] and qgroups(x2)[-1] == 1
    # OLS 부트스트랩 · 상관
    n = 4000; day = np.repeat(np.arange(200), 20).astype("datetime64[D]")
    xx = rng.normal(size=n); cc_ = rng.normal(size=n); yy = 2 * xx + cc_ + rng.normal(size=n)
    o = ols_boot(np.column_stack([np.ones(n), cc_, xx]), yy, day, 2)
    assert o["lo"] < 2 < o["hi"] and o["dr2"] > 0.3
    o0 = ols_boot(np.column_stack([np.ones(n), cc_, rng.normal(size=n)]), yy, day, 2)
    assert abs(o0["b"]) < 0.15 and o0["hi"] - o0["lo"] < 0.3   # 무관 변수(기각률 실측 2/40 ≈ 5%라 CI 포함 여부는 단언 안 함)
    assert abs(corr_ci(xx, xx, day)[0] - 1) < 1e-9
    assert call({"a": (1, .5, 2), "b": (1, .1, 3)}, 1) == "통과" and call({"a": (1, .5, 2), "b": (1, -1, 3)}, 1) == "한 해만"
    assert call({"a": (-1, -2, -.5), "b": (-1, -3, -.1)}, 1) == "통과(주장 반대)"
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data", type=Path, default=R.default_data_dir())
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "tmp/flow_size_cvd_depth_20260930")
    a = ap.parse_args()
    selftest() if a.selftest else run(a.data, a.out)
