#!/usr/bin/env python3
"""ETH 선물 트레이더에게 옵션 지표 4종(IV 랭크 · 10/15Δ 스큐 · vanna 노출 · vega/theta 노출)이 쓸모 있는가 (2026-09-30).

지표를 과거 시점에 그대로 계산해 두고 그 **다음**의 ETH 수익·실현 변동성을 센다(모델 학습 없음).
🔴사용자 규칙: 옵션 데이터 검증은 2026년만. 2025 DVOL 은 IV 랭크 365일 창 예열에만 쓰고 집계는 2026만.

── 사전등록 (결과 보기 전에 고정, 2026-09-30) ──────────────────────────────────────────
데이터
- 가격: ETHUSDT 선물 5분봉(로컬 5m CSV ~09-15 + data.binance.vision 일봉 파일 09-15~09-27 — 09-29 재검정이 받아둔
  사본을 복사, 네트워크 호출 없음). 라벨 끝 = 2026-09-27 23:55 UTC → 체인의 09-27 이후 스냅샷은 24h 라벨이 없다.
- DVOL: Deribit 공개 API 시간봉(2025-10~) + 로컬 CSV(그 전, 랭크 예열 전용). 봉 ts 는 시작 시각 → 종가는
  known_ts = ts + 1h 에 알려진 것으로 둔다(보수적).
- 체인: 서버 data/live/deribit_options.duckdb option_chain_snapshot(ETH, 08-15~, 시간봉 · 09-28~ 10분)을 read_only COPY.
  저장된 그릭스는 gamma 뿐 → mark_iv 로 블랙숄즈(r=0, 선도 F = 그 행의 underlying_price)를 직접 계산.

시점 계약 (Event-Label Boundary Contract)
- 피쳐 시각 k (DVOL known_ts · 체인 recorded_at). HAR-RV 피쳐는 k 이전에 **닫힌** 5분봉까지.
- 라벨 시작 봉 i0 = open ≥ k 인 첫 5분봉. 기준가 = close[i0] (k 뒤 최소 5분), 수익·RV 는 i0+1 봉부터.
  → 피쳐 마지막 봉과 라벨 첫 수익 사이에 최소 한 봉(i0)이 비어 있다.
- r_H = ln(close[i0+H]/close[i0]) bp · RV_H = 그 사이 5분 수익 제곱합의 연율화 %   (H = 1h · 4h · 24h)

독립 단위 = 일. 주 판정은 **24h 한 지평**, 표본은 하루 한 점(UTC 00:00 에 알려진 DVOL / 그날 03:00 전 첫 체인 스냅샷).
1h·4h 는 모든 시간 관측 + 일 블록 부트스트랩으로 **참고만**(판정에 안 씀).
CI = 일 블록 부트스트랩 2000회(시드 20260930) 95%. 효과 = 피쳐 표준화 1SD 당 계수.
변동성 검정 통제 = HAR(ln RV 직전 1일·7일·30일) + ln DVOL(k 에 알려진 최신).

항목 · 정의 · 주장(부호)
1 IV 랭크 = DVOL 의 직전 365일(시간봉) 백분위 0~100. 표본 2026-01-01~09-26, 반기 H1 = 1~5월 · H2 = 6~9월.
  1a 변동성: ln RV24_fwd ~ HAR + ln DVOL + z(랭크).  주장 β<0 (랭크 높으면 이후 실현이 낮다 = 평균회귀).
      추가 조건: 표본외(H1 적합 → H2 예측) MSE 개선 > 0.
  1b 방향: 24h 수익 평균(랭크 ≥80) − 평균(랭크 ≤20).  주장 > 0 (공포 뒤 반등).
      추가 조건: 랭크 ≥80 평균 > 전 표본 평균(항상 롱).
2 스큐 = 30일 보간 [IV(풋 |Δ|=0.15) − IV(콜 Δ=0.15)] vol pt (15Δ 가 판정, 10Δ 와 «7일 이상 가장 가까운 만기»는 참고).
  만기마다 OTM 옵션의 BS 선도 델타 위에서 선형 보간(외삽 = NaN), 30일을 사이에 둔 만기로 DTE 선형 보간.
  2a 방향: r24 ~ z(스큐). 주장 β>0 (풋이 비싸면 역발상 반등).
  2b 변동성: ln RV24_fwd ~ HAR + ln DVOL + z(스큐). 주장 β>0 (하방 변동성 확대).
3 vanna 노출 VEX = Σ w·vanna·0.01·F [USD, IV +1pt 당 딜러 델타 변화]. w = 딜러 가정 = 수집기 w_asm
  (collect_deribit_option_gex_20260815: 콜 +OI 딜러 매수 · 풋 −OI 딜러 매도, GEX 부호와 같음). DTE ≤ 30일, 1시간 미만 남은 종목 제외.
  3a 방향(매매 가능): 헤지 흐름 F = −VEX × ΔDVOL(직전 24h). r24_fwd ~ z(F). 주장 β>0.
  3b 기전(사후 · 같은 24h 의 ΔDVOL 을 쓰므로 매매 불가 · 판정 안 씀): r24 ~ ΔDVOL + z(−VEX × ΔDVOL). 주장 계수>0.
4 같은 w, DTE ≤ 30: VEGA = Σ w·vega·0.01 [USD / IV +1pt] · THETA = Σ w·theta/365 [USD/일].
  4a ln RV24_fwd ~ HAR + ln DVOL + z(VEGA). 주장 β<0 (딜러 vega 숏이 클수록 변동 확대).
  4b ln RV24_fwd ~ HAR + ln DVOL + z(THETA). 주장 β>0 (딜러 theta 가 음 = 딜러 롱감마일수록 붙잡힘 → RV 낮음).
  체인 항목(2~4) 반기: H1 = 08-15~09-05 · H2 = 09-06~09-26.

판정
- 통과: 전체 CI 가 주장 방향으로 0 배제 ∧ H1·H2 부호가 둘 다 주장 방향 ∧ (항목별 추가 조건).
- 불합격: 전체 CI 가 주장 반대로 0 배제 · CI 는 주장 방향인데 반기 불일치/추가 조건 실패 · 독립일 ≥ 60 인데 CI 0 포함.
- 검정력 부족: 독립일 < 60 이고 CI 가 0 포함(체인 항목). MDE(80% 검정력) = 2.8 × 부트스트랩 SE 를 같이 적는다.

사용: python3 scripts/research_eth_options_ivrank_skew_greeks_20260930.py [--selftest]
출력: tmp/options_ivrank_skew_greeks_20260930/ (report.txt · daily_*.csv). 체인 입력 chain_eth.parquet 은 서버에서:
  COPY (SELECT recorded_at_utc, option_type, strike, expiration_ts, days_to_expiry, open_interest, mark_iv, underlying_price
        FROM option_chain_snapshot WHERE currency='ETH' AND mark_iv > 0) TO '.../chain_eth.parquet'   (read_only, 0.35초)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]
OUT = ROOT / "tmp/options_ivrank_skew_greeks_20260930"
DVOL_CSV = Path("/home/kbj20/crypto-scalping/data/derivatives/deribit_dvol/ETH_dvol_hourly.csv")
WARM, T_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2026-09-27 23:55")
Y26, IV_H2, CH_H2 = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-01"), pd.Timestamp("2026-09-06")
BAR, HZ = pd.Timedelta("5min"), {"1h": 12, "4h": 48, "24h": 288}
HAR = ["lrv1d", "lrv7d", "lrv30d", "ldvol"]
B, SEED = 2000, 20260930


# ── 입력 ────────────────────────────────────────────────────────────────────
def load_px() -> pd.Series:
    """5분봉 종가(봉 시작 시각, naive UTC). 2026 재검정의 로더를 그대로 쓴다(입력 폴더만 이 출력 폴더)."""
    import research_eth_options_2026only_revalidation_20260929 as R
    R.W, R.T0, R.T_END = OUT, WARM, T_END
    return R.load_px5().set_index("timestamp")["close"].astype(float)


def load_dvol() -> pd.Series:
    """known_ts(= 봉 시작 + 1h) → DVOL 종가. API(2025-10~)가 원천, CSV 는 그 전 예열만."""
    import research_eth_options_2026only_revalidation_20260929 as R
    R.T0 = WARM
    api = R.load_dvol().set_index("timestamp")["close"]
    csv = pd.read_csv(DVOL_CSV, parse_dates=["timestamp"]).set_index("timestamp")["close"]
    ov = csv.index[:-1].intersection(api.index)     # CSV 마지막 행은 받을 때 형성 중이던 봉(08-04 10:00, 0.48pt 차)
    assert len(ov) > 1000 and (csv[ov] - api[ov]).abs().max() < 0.05, "CSV·API 시각 규약 불일치"
    s = pd.concat([csv[csv.index < api.index.min()], api]).sort_index()
    s.index = s.index + pd.Timedelta("1h")
    return s


# ── 그릭스 · 체인 피쳐 ───────────────────────────────────────────────────────
def bs(F, K, T, s) -> dict:
    """블랙(선도) r=0. vega·vanna 는 σ 1.00 당, theta 는 1년당. 풋·콜 공통(델타만 다름)."""
    sq = s * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s * s * T) / sq
    d2, pdf = d1 - sq, norm.pdf(d1)
    return {"dc": norm.cdf(d1), "gamma": pdf / (F * sq), "vega": F * pdf * np.sqrt(T),
            "vanna": -pdf * d2 / s, "theta": -F * pdf * s / (2 * np.sqrt(T))}


def smile_skew(e: pd.DataFrame, d: float) -> float:
    """한 만기 OTM 행(xd = |델타|) → IV(풋 d) − IV(콜 d). 보간만, 범위 밖이면 NaN."""
    iv = []
    for typ in ("put", "call"):
        x = e[e.option_type == typ].sort_values("xd")
        if len(x) < 2 or not x.xd.iloc[0] <= d <= x.xd.iloc[-1]:
            return np.nan
        iv.append(np.interp(d, x.xd, x.mark_iv))
    return iv[0] - iv[1]


def skew_at(snap: pd.DataFrame) -> list[float]:
    """[30일 15Δ, 30일 10Δ, 7일+최근접 15Δ, 7일+최근접 10Δ]"""
    per = pd.DataFrame([(e.days_to_expiry.iloc[0], smile_skew(e, 0.15), smile_skew(e, 0.10))
                        for _, e in snap.groupby("expiration_ts")], columns=["dte", "s15", "s10"]).sort_values("dte")
    out = []
    for col in ("s15", "s10"):
        v = per.dropna(subset=[col])
        out.append(float(np.interp(30, v.dte, v[col])) if len(v) >= 2 and v.dte.min() <= 30 <= v.dte.max() else np.nan)
    near = per[per.dte >= 7].head(1)
    return out + ([float(near.s15.iloc[0]), float(near.s10.iloc[0])] if len(near) else [np.nan, np.nan])


def chain_features(c: pd.DataFrame) -> pd.DataFrame:
    c = c[(c.days_to_expiry >= 1 / 24) & (c.mark_iv > 0) & (c.underlying_price > 0)].copy()
    F, K = c.underlying_price.to_numpy(), c.strike.to_numpy()
    g = bs(F, K, c.days_to_expiry.to_numpy() / 365, c.mark_iv.to_numpy() / 100)
    call = (c.option_type == "call").to_numpy()
    w = np.where(call, 1.0, -1.0) * c.open_interest.to_numpy() * (c.days_to_expiry.to_numpy() <= 30)
    c["vex"], c["vega_x"] = w * g["vanna"] * 0.01 * F, w * g["vega"] * 0.01
    c["theta_x"], c["gex"] = w * g["theta"] / 365, w * g["gamma"] * F * F * 0.01
    c["xd"] = np.where(call, g["dc"], 1 - g["dc"])
    agg = c.groupby("recorded_at_utc")[["vex", "vega_x", "theta_x", "gex"]].sum()
    otm = c[np.where(call, K >= F, K <= F)]
    sk = pd.DataFrame([[ts, *skew_at(s)] for ts, s in otm.groupby("recorded_at_utc")],
                      columns=["recorded_at_utc", "s15", "s10", "s15n", "s10n"]).set_index("recorded_at_utc")
    out = agg.join(sk)
    out.index = out.index.tz_convert(None)
    return out


# ── 라벨 · HAR ──────────────────────────────────────────────────────────────
class Px:
    def __init__(self, close: pd.Series):
        self.t, self.c = close.index.values, close.to_numpy(float)
        self.cs = np.concatenate([[0.0], np.cumsum(np.diff(np.log(self.c)) ** 2)])   # cs[j] = Σ lr[1..j]

    def attach(self, df: pd.DataFrame) -> pd.DataFrame:
        k, n = df.index.values, len(self.c)
        i0 = np.searchsorted(self.t, k, "left")                         # open ≥ k
        jl = np.searchsorted(self.t + BAR, k, "right") - 1              # 마지막으로 k 이전에 닫힌 봉
        assert (jl < i0).all()
        df = df.copy()
        for nm, h in (("1d", 288), ("7d", 2016), ("30d", 8640)):
            ok = jl - h >= 0
            v = np.where(ok, self.cs[np.clip(jl, 0, n - 1)] - self.cs[np.clip(jl - h, 0, n - 1)], np.nan)
            df[f"lrv{nm}"] = np.log(np.sqrt(v * 288 * 365 / h) * 100)
        ok0 = i0 < n
        df["t_ref"] = pd.to_datetime(np.where(ok0, self.t[np.clip(i0, 0, n - 1)], np.datetime64("NaT"))) + BAR
        for nm, h in HZ.items():
            j = i0 + h
            ok = (j < n) & ok0
            jj, ii = np.clip(j, 0, n - 1), np.clip(i0, 0, n - 1)
            ok &= (self.t[jj] - self.t[ii]) == np.timedelta64(h * 5, "m")   # 구멍 가로지르면 버린다
            df[f"r{nm}"] = np.where(ok, np.log(self.c[jj] / self.c[ii]) * 1e4, np.nan)
            df[f"lrvf{nm}"] = np.where(ok, np.log(np.sqrt((self.cs[jj] - self.cs[ii]) * 288 * 365 / h) * 100), np.nan)
        return df


def dvol_asof(dv: pd.Series, t) -> np.ndarray:
    i = np.searchsorted(dv.index.values, np.asarray(t, dtype="datetime64[ns]"), "right") - 1
    return np.where(i >= 0, dv.to_numpy()[np.clip(i, 0, None)], np.nan)


# ── 통계 ────────────────────────────────────────────────────────────────────
def fit(y, X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b, 1 - np.var(y - X @ b) / np.var(y)


def X_of(d, cols):
    return np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in cols])


def reg_stat(y, ctrl, x="z"):
    return lambda d: fit(d[y].to_numpy(float), X_of(d, ctrl + [x]))[0][-1] if len(d) > len(ctrl) + 3 else np.nan


def boot(d: pd.DataFrame, stat) -> np.ndarray:
    code = pd.factorize(d.index.normalize())[0]
    groups = [np.flatnonzero(code == i) for i in range(code.max() + 1)]
    rng = np.random.default_rng(SEED)
    return np.array([stat(d.iloc[np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])])
                     for _ in range(B)])


def judge(d, stat, s, h2_start, extra_ok=True) -> dict:
    est, bt = stat(d), boot(d, stat)
    lo, hi = np.nanpercentile(bt, [2.5, 97.5])
    h1, h2 = stat(d[d.index < h2_start]), stat(d[d.index >= h2_start])
    nd = d.index.normalize().nunique()
    ok, opp = ((lo > 0), (hi < 0)) if s > 0 else ((hi < 0), (lo > 0))
    same = np.sign(h1) == s and np.sign(h2) == s
    v = "통과" if ok and same and extra_ok else "불합격" if (opp or ok or nd >= 60) else "검정력 부족"
    return dict(est=est, lo=lo, hi=hi, mde=2.8 * np.nanstd(bt), h1=h1, h2=h2, n=len(d), days=nd, v=v)


def fmt(name, r, unit="") -> str:
    return (f"  {name:44} {r['est']:+8.3f}{unit} [{r['lo']:+.3f}, {r['hi']:+.3f}] · MDE {r['mde']:.3f} · "
            f"H1 {r['h1']:+.3f} / H2 {r['h2']:+.3f} · n={r['n']} 독립일 {r['days']} → {r['v']}")


def zcol(d, x):
    d = d.dropna(subset=[x]).copy()
    d["z"] = (d[x] - d[x].mean()) / d[x].std()
    return d


def secondary(d_hourly, x, s_ret, s_rv, lab) -> list[str]:
    """1h·4h 참고(모든 시간 관측, 일 블록 CI). 판정 안 씀."""
    out = []
    for h in ("1h", "4h"):
        cells = []
        for y, ctrl, s in ((f"r{h}", [], s_ret), (f"lrvf{h}", HAR, s_rv)):
            if s is None:
                continue
            d = zcol(d_hourly, x).dropna(subset=[y, *ctrl])
            st = reg_stat(y, ctrl)
            bt = boot(d, st)
            lo, hi = np.nanpercentile(bt, [2.5, 97.5])
            cells.append(f"{'수익bp' if y[0] == 'r' else 'lnRV'} {st(d):+.3f} [{lo:+.3f},{hi:+.3f}]")
        out.append(f"    참고 {lab} {h}: " + " · ".join(cells) + f" (n={len(d)})")
    return out


# ── 본체 ────────────────────────────────────────────────────────────────────
def main() -> int:
    px, dv = Px(load_px()), load_dvol()
    rank = dv.rolling("365D").rank(pct=True) * 100
    say = print
    rep = [f"# 옵션 지표 4종 → ETH 선물 검정 (2026년만) · 실행 {pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC",
           f"가격 5분봉 {pd.Timestamp(px.t[0])} ~ {pd.Timestamp(px.t[-1])} · DVOL known {dv.index.min()} ~ {dv.index.max()}"]

    # 1 IV 랭크 ------------------------------------------------------------------
    H = pd.DataFrame({"dvol": dv, "rank_": rank})
    H = H[(H.index >= Y26) & (H.index <= T_END)]
    H["ldvol"] = np.log(H.dvol)
    H = px.attach(H)
    D = H[H.index.hour == 0].dropna(subset=["r24h", "lrvf24h", *HAR])
    D.to_csv(OUT / "daily_ivrank.csv")
    rep.append(f"\n## 1 IV 랭크 (DVOL 365일 백분위) — 일 표본 {len(D)} · {D.index.min():%m-%d} ~ {D.index.max():%m-%d}")
    for lab, m in (("H1 1~5월", D.index < IV_H2), ("H2 6~9월", D.index >= IV_H2)):
        x = D[m]
        rep.append(f"  {lab}: 랭크 중앙 {x.rank_.median():.0f} · ≥80 {int((x.rank_ >= 80).sum())}일 · ≤20 {int((x.rank_ <= 20).sum())}일 · "
                   f"DVOL {x.dvol.min():.0f}~{x.dvol.max():.0f}")
    d = zcol(D, "rank_")
    b1, f1 = X_of(d[d.index < IV_H2], HAR), X_of(d[d.index < IV_H2], HAR + ["z"])
    y1 = d.loc[d.index < IV_H2, "lrvf24h"].to_numpy()
    y2, h2 = d.loc[d.index >= IV_H2, "lrvf24h"].to_numpy(), d[d.index >= IV_H2]
    mse = lambda X1, y1_, X2, y2_: np.mean((y2_ - X2 @ fit(y1_, X1)[0]) ** 2)
    dmse = mse(b1, y1, X_of(h2, HAR), y2) - mse(f1, y1, X_of(h2, HAR + ["z"]), y2)
    dr2 = fit(d.lrvf24h.to_numpy(), X_of(d, HAR + ["z"]))[1] - fit(d.lrvf24h.to_numpy(), X_of(d, HAR))[1]
    r = judge(d, reg_stat("lrvf24h", HAR), -1, IV_H2, extra_ok=dmse > 0)
    rep += [fmt("1a lnRV24 ~ HAR+lnDVOL+z(랭크)  주장 β<0", r),
            f"      ΔR² {dr2:+.4f} · 표본외 H1→H2 ΔMSE {dmse:+.5f} (>0 이면 랭크가 예측을 개선)"]
    ratio = np.exp(d.lrvf24h - d.ldvol)
    rep.append("      실현/내재(RV24_fwd/DVOL) 평균 — 랭크 ≤20 {:.2f} · 20~80 {:.2f} · ≥80 {:.2f}".format(
        ratio[d.rank_ <= 20].mean(), ratio[(d.rank_ > 20) & (d.rank_ < 80)].mean(), ratio[d.rank_ >= 80].mean()))
    rep += secondary(H, "rank_", None, -1, "랭크 lnRV")
    hi_, lo_ = d[d.rank_ >= 80].r24h, d[d.rank_ <= 20].r24h
    spread = lambda x: x.loc[x.rank_ >= 80, "r24h"].mean() - x.loc[x.rank_ <= 20, "r24h"].mean()
    r = judge(d, spread, +1, IV_H2, extra_ok=hi_.mean() > d.r24h.mean())
    rep += [fmt("1b r24 평균(랭크≥80) − (≤20)  주장 >0", r, "bp"),
            f"      ≥80 {hi_.mean():+.1f}bp (n={len(hi_)}) · ≤20 {lo_.mean():+.1f}bp (n={len(lo_)}) · 전체 {d.r24h.mean():+.1f}bp · "
            f"기울기 참고 {reg_stat('r24h', [])(d):+.1f}bp/SD"]
    rep += secondary(H, "rank_", +1, None, "랭크 수익")

    # 2~4 체인 -------------------------------------------------------------------
    say("체인 피쳐 계산 ...")
    C = chain_features(pd.read_parquet(OUT / "chain_eth.parquet"))
    C["ldvol"] = np.log(dvol_asof(dv, C.index))
    C["ddvol_past"] = C.ldvol.pipe(np.exp) - dvol_asof(dv, C.index - pd.Timedelta("24h"))
    C["flow_past"] = -C.vex * C.ddvol_past
    C = px.attach(C[C.index <= T_END])
    C["ddvol_fwd"] = dvol_asof(dv, C.t_ref + pd.Timedelta("24h")) - dvol_asof(dv, C.t_ref)
    C["flow_fwd"] = -C.vex * C.ddvol_fwd
    C.to_csv(OUT / "chain_features_hourly.csv")
    first = C[C.index.hour < 3].groupby(C[C.index.hour < 3].index.normalize()).head(1)
    E = first.dropna(subset=["r24h", "lrvf24h", *HAR])
    E.to_csv(OUT / "daily_chain.csv")
    alld = pd.date_range(C.index.min().normalize(), E.index.max().normalize(), freq="D")
    rep.append(f"\n## 체인 항목 — 스냅샷 {len(C)} · 일 표본 {len(E)} ({E.index.min():%m-%d} ~ {E.index.max():%m-%d}) · "
               f"빠진 날 {sorted(set(alld.strftime('%m-%d')) - set(E.index.strftime('%m-%d')))}")
    rep.append(f"  스큐 결측: 30d15Δ {C.s15.isna().mean():.1%} · 30d10Δ {C.s10.isna().mean():.1%} · "
               f"스큐15 중앙 {C.s15.median():+.2f}pt (범위 {C.s15.min():+.1f}~{C.s15.max():+.1f}) · 10Δ 중앙 {C.s10.median():+.2f}")
    rep.append("  노출 중앙(USD): VEX {:+,.0f}/pt · VEGA {:+,.0f}/pt · THETA {:+,.0f}/일 · GEX {:+,.0f}/1%".format(
        *(C[k].median() for k in ("vex", "vega_x", "theta_x", "gex"))))
    cr = C[["vex", "vega_x", "theta_x", "gex", "s15"]].corr()
    rep.append(f"  상관: THETA·GEX {cr.loc['theta_x', 'gex']:+.2f} · VEGA·GEX {cr.loc['vega_x', 'gex']:+.2f} · "
               f"VEX·GEX {cr.loc['vex', 'gex']:+.2f} · VEX·VEGA {cr.loc['vex', 'vega_x']:+.2f} · 부호 비율 VEX<0 {(C.vex < 0).mean():.0%} "
               f"VEGA<0 {(C.vega_x < 0).mean():.0%} THETA<0 {(C.theta_x < 0).mean():.0%}")

    tests = [("2a r24 ~ z(스큐15 30d)  주장 β>0", "s15", "r24h", [], +1, "bp"),
             ("2b lnRV24 ~ HAR+lnDVOL+z(스큐15)  주장 β>0", "s15", "lrvf24h", HAR, +1, ""),
             ("3a r24 ~ z(−VEX×ΔDVOL 직전24h)  주장 β>0", "flow_past", "r24h", [], +1, "bp"),
             ("4a lnRV24 ~ HAR+lnDVOL+z(VEGA)  주장 β<0", "vega_x", "lrvf24h", HAR, -1, ""),
             ("4b lnRV24 ~ HAR+lnDVOL+z(THETA)  주장 β>0", "theta_x", "lrvf24h", HAR, +1, "")]
    for name, x, y, ctrl, s, unit in tests:
        d = zcol(E, x).dropna(subset=[y, *ctrl])
        rep.append(fmt(name, judge(d, reg_stat(y, ctrl), s, CH_H2), unit))
        rep += secondary(C, x, s if y[0] == "r" else None, s if y[0] == "l" else None, x)
    rep.append("  참고(판정 안 씀) — 10Δ · 7일+최근접 만기:")
    for x in ("s10", "s15n", "s10n"):
        for y, ctrl in (("r24h", []), ("lrvf24h", HAR)):
            d = zcol(E, x).dropna(subset=[y, *ctrl])
            rep.append(fmt(f"    {x} → {y}", judge(d, reg_stat(y, ctrl), +1, CH_H2)))
    d = zcol(E, "flow_fwd").dropna(subset=["r24h", "ddvol_fwd"])
    r = judge(d, reg_stat("r24h", ["ddvol_fwd"]), +1, CH_H2)
    rep.append(fmt("3b 사후 기전 r24 ~ ΔDVOL + z(−VEX×ΔDVOL 같은24h)", r, "bp") + "  ※매매 불가·판정 안 씀")
    rep.append(f"      같은 24h ΔDVOL 단독 기울기 {reg_stat('r24h', [], 'ddvol_fwd')(d):+.1f}bp/pt (현물-변동성 상관)")
    rep.append(f"      공선성: corr(3a 흐름, −ΔDVOL 직전24h) {np.corrcoef(E.flow_past, -E.ddvol_past)[0, 1]:+.2f} · "
               f"corr(3b 흐름, ΔDVOL 같은24h) {np.corrcoef(E.flow_fwd, E.ddvol_fwd)[0, 1]:+.2f} · VEX 변동계수 {E.vex.std() / E.vex.mean():.2f}")
    rep.append(f"      규모: 일 표본 |ΔDVOL 24h| 중앙 {E.ddvol_past.abs().median():.1f}pt × VEX 중앙 → 딜러 재헤지 "
               f"${(E.ddvol_past.abs() * E.vex).median() / 1e6:.1f}M/일(딜러 가정 하)")

    rep.append("\n읽는 법: 계수 = 피쳐 1SD 당(수익은 bp, 변동성은 ln RV). [CI]=일 블록 부트스트랩 95%. "
               "MDE = 이 표본으로 80% 확률로 잡을 수 있는 최소 효과.")
    (OUT / "report.txt").write_text("\n".join(rep), encoding="utf-8")
    say("\n".join(rep))
    return 0


def _selftest() -> None:
    # 그릭스: 유한차분과 대조 + 부호
    F, K, T, s = 2500.0, 2700.0, 20 / 365, 0.6
    call = lambda F_, T_, s_: F_ * norm.cdf((np.log(F_ / K) + .5 * s_ * s_ * T_) / (s_ * np.sqrt(T_))) - \
        K * norm.cdf((np.log(F_ / K) - .5 * s_ * s_ * T_) / (s_ * np.sqrt(T_)))
    g, e = bs(F, K, T, s), 1e-4
    dc = lambda F_, s_: bs(F_, K, T, s_)["dc"]
    assert abs(g["vega"] - (call(F, T, s + e) - call(F, T, s - e)) / (2 * e)) < 1e-4 * g["vega"]
    assert abs(g["theta"] + (call(F, T + e, s) - call(F, T - e, s)) / (2 * e)) < 1e-3 * abs(g["theta"])
    assert abs(g["gamma"] - (dc(F + .01, s) - dc(F - .01, s)) / .02) < 1e-4 * g["gamma"]
    assert abs(g["vanna"] - (dc(F, s + e) - dc(F, s - e)) / (2 * e)) < 1e-4 * abs(g["vanna"])
    assert g["vega"] > 0 and g["theta"] < 0 and g["gamma"] > 0
    assert bs(F, 2900.0, T, s)["vanna"] > 0 and bs(F, 2100.0, T, s)["vanna"] < 0    # 위 OTM +, 아래 −
    # 딜러 부호: 콜 1계약 → +vanna, 풋 1계약 → −vanna (수집기 w_asm)
    ch = pd.DataFrame({"recorded_at_utc": pd.Timestamp("2026-09-01", tz="UTC"), "option_type": ["call", "put"],
                       "strike": [2900.0, 2900.0], "expiration_ts": pd.Timestamp("2026-09-21 08:00", tz="UTC"),
                       "days_to_expiry": [20.0, 20.0], "open_interest": [1.0, 1.0], "mark_iv": [60.0, 60.0],
                       "underlying_price": [F, F]})
    assert abs(chain_features(ch).vex.iloc[0]) < 1e-12                              # 같은 행사가 콜+풋 → 상쇄
    assert chain_features(ch.iloc[:1]).vex.iloc[0] > 0 and chain_features(ch.iloc[1:]).vex.iloc[0] < 0
    # 스큐 보간 · 외삽 금지 · 30일 보간
    mk = lambda put, call_: pd.DataFrame({"option_type": ["put"] * 4 + ["call"] * 4, "xd": [.05, .1, .2, .3] * 2,
                                          "mark_iv": put + call_})
    e1 = mk([70, 65, 60, 55], [60, 57, 54, 51])
    assert abs(smile_skew(e1, .15) - 7.0) < 1e-9 and np.isnan(smile_skew(e1, .4))
    two = pd.concat([mk([70, 66, 62, 58], [60, 60, 60, 60]).assign(days_to_expiry=20.0, expiration_ts=1),
                     mk([70, 70, 70, 70], [60, 60, 60, 60]).assign(days_to_expiry=40.0, expiration_ts=2)])
    sk = skew_at(two)                                # 20일 64−60=4pt, 40일 10pt → 30일 7pt ; 7일+최근접 = 20일 4pt
    assert abs(sk[0] - 7.0) < 1e-9 and abs(sk[2] - 4.0) < 1e-9
    # 라벨 시작: 06:00 봉에서 일어난 급등은 피쳐에도 라벨에도 안 들어간다
    t = pd.date_range("2026-01-01", periods=12000, freq="5min")
    c = pd.Series(100.0, index=t)
    c[c.index >= pd.Timestamp("2026-01-30 06:00")] = 200.0
    p = Px(c)
    for k in ("2026-01-30 06:03:40", "2026-01-30 06:00:00"):
        a = p.attach(pd.DataFrame(index=pd.DatetimeIndex([pd.Timestamp(k)])))
        assert a.t_ref.iloc[0] >= pd.Timestamp(k) + BAR, a.t_ref.iloc[0]
        assert a.r1h.iloc[0] == 0 and np.isneginf(a.lrvf1h.iloc[0])            # 라벨 창에 급등 없음
        assert np.isneginf(a.lrv1d.iloc[0])                                     # 피쳐 창에 급등 없음(RV=0)
    a = p.attach(pd.DataFrame(index=pd.DatetimeIndex([pd.Timestamp("2026-01-30 05:54:00")])))
    assert a.r1h.iloc[0] > 6000                                                 # 05:55 봉 이전 시점이면 라벨에 들어간다(양성 대조)
    # IV 랭크: 단조 증가면 마지막 = 100, 최저점 새로 찍으면 ~0
    s_ = pd.Series(np.r_[np.arange(9000.), -1.0], index=pd.date_range("2024-01-01", periods=9001, freq="h"))
    rk = s_.rolling("365D").rank(pct=True) * 100
    assert rk.iloc[-2] == 100 and rk.iloc[-1] < 0.02
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    _selftest()
    if not a.selftest:
        raise SystemExit(main())
