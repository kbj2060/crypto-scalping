#!/usr/bin/env python3
"""감마 재헤지 흔적 검정 (2026-10-01, 사용자 «다른 축 — 감마 재헤지 흔적») -- 어느 GEX 가정이 딜러 재헤지 행동과 맞물리나.

이론: 딜러 양감마면 가격이 오른 뒤 선물을 팔고 내린 뒤 산다(움직임을 누름), 음감마면 따라 산다.
  ⇒ 다음 선물 테이커 순매수 Y_{t+1..t+k} ≈ −c · GEX_t · Δp_t (c > 0). 물리 단위로는 딜러 델타 변화 = GEX$·r·100/S [ETH]
     (GEX$ = γ·F²·1% · w, 재구성 스크립트와 같은 식) 이고 딜러 거래 = −그 값 → 한 거래소가 헤지를 전부 받으면 계수 −1.

사전등록 (결과 보기 전 고정, 2026-10-01):
  분 t = 1분봉(바이낸스 ETHUSDT 선물 kline open 분). 결정 = 분 t 종가 시점. Δp_h = 1e4·(log c_t − log c_{t−h}) [bp], h = 5·15.
  라벨 Y_k = Σ_{j=1..k} Y_{t+j}, k = 5·15·60. Y = 테이커 순매수 ETH: 바이낸스(2·taker_buy − vol) · Deribit ETH-PERPETUAL(블록 다리 제외)
    · 합. 가격 = 바이낸스 1분 종가.
  GEX_t = 타임스탬프 ≤ 분 t 종가 시각인 가장 최근 값(2시간 넘게 묵으면 버림). 창(반기) 안에서 분 가중 z 표준화 → z_t.
  모형 (OLS): Y_k ~ 1 + z·Δp_h + Δp_h + z + Y_t + ΣY_{t−1..t−4} + ΣY_{t−5..t−14} + UTC 시 더미 23. 핵심 = z·Δp_h 계수 β.
    (z 의 중심화는 Δp 주효과만 옮기고 β 는 안 바뀐다 — 창 전체로 표준화해도 시점 누수 아님.) 단위 = 1σ GEX × 1bp 당 k 분 ETH.
  가정: dealer = 체결 기반 시간별 딜러 GEX(tmp/dealer_gex_reconstruct_2026_20261001/hourly_dealer_gex.parquet), 2026-01-01~09-30,
          반기 H_a 01-01~05-31 · H_b 06-01~09-30.
        conv = 관행(콜 +OI · 풋 −OI, mark_iv·underlying_price, 체인 스냅샷 시간별 첫 것), conv_flip = −conv,
        dealer_sw = dealer 를 같은 스냅샷 창으로 자른 것(나란히 보기). 창 08-15~09-30, H_a ~09-06 · H_b 09-07~.
  범위 front·week·all.
  주 판정 칸 = week · 합(바이낸스+Deribit) · h=15 · k=15. 통과 = β < 0 이고 두 반기 모두 일 블록 부트스트랩(400회) 95% CI 가 0 배제.
    (conv_flip 의 β 는 정의상 −conv 의 β — 둘 중 하나만 이론 부호일 수 있다.) 나머지 칸은 보조(같은 기준으로 표시만).
  보조(가격): 다음 수익 1e4·(log c_{t+k} − log c_t), k=15·60 ~ 1 + z·Δp_h + Δp_h + z + 시 더미. 이론: β < 0(양감마면 되돌림 강함).
  강건성(주 칸 설정, 판정 밖): ① GEX 1시간 늦추기(누수 점검) ② Δp·(직전 24h 실현변동성 z) 통제 추가(GEX↔변동성 수준 교란)
    ③ 물리 단위 회귀: 핵심 = GEX$·Δp·100/(1e4·S) [ETH], 기대 −(Δp 창의 움직임 중 t+1..t+k 에 헤지되는 몫 × 그 거래소 몫)
    — 즉시 헤지면 0 에 가깝고, 전부 늦게 이 거래소에서 하면 −1(겹치는 Δp 창은 selftest 4 참고).
  옵션 데이터는 2026 만(사용자 규칙). 바이낸스 API 호출 없음(로컬 kline · data.binance.vision 캐시).
출력 tmp/gamma_rehedge_footprint_2026_20261001/ : conv_gex_hourly.parquet · results.json.

  python scripts/research_eth_gamma_rehedge_footprint_2026_20261001.py
  python scripts/research_eth_gamma_rehedge_footprint_2026_20261001.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_dealer_gex_reconstruct_2026_20261001 as R  # noqa: E402  exposures · Bars
import research_eth_dealer_hedge_footprint_2026_20261001 as HF  # noqa: E402  binance_minutes · ols_boot · 캐시

OUT = ROOT / "tmp/gamma_rehedge_footprint_2026_20261001"
H_MS, M_MS = 3_600_000, 60_000
ms = lambda s: pd.Timestamp(s, tz="UTC").value // 10**6
D0, D1, SPLIT = ms("2026-01-01"), ms("2026-10-01"), ms("2026-06-01")
SW0, SW_SPLIT = ms("2026-08-15"), ms("2026-09-07")
SCOPES, HS, KS, KR = ("front", "week", "all"), (5, 15), (5, 15, 60), (15, 60)
VEN = {"binance": "Yb", "deribit": "Yd", "sum": "Ys"}


# ───────────────────────── 데이터 ─────────────────────────
def conv_gex() -> pd.DataFrame:
    """관행 GEX 시간별(재구성 스크립트 H3 와 같은 계산): 시간마다 첫 체인 스냅샷, w = ±OI."""
    p = OUT / "conv_gex_hourly.parquet"
    if p.exists():
        return pd.read_parquet(p)
    snap = pd.read_parquet(ROOT / "tmp/dpa20260930/snap.parquet")
    snap["ts"] = snap["recorded_at_utc"].astype("int64") // 1000
    st = np.sort(snap["ts"].unique()); s_eval = pd.Series(st).groupby(st // H_MS).min().to_numpy()
    B = R.Bars(pd.read_parquet(R.OUT / "perp_5m.parquet"), pd.read_parquet(R.OUT / "dvol_1h.parquet"))
    S = B.at(s_eval)["S"].to_numpy()
    sn = snap[snap["ts"].isin(s_eval)].copy(); sn["k"] = np.searchsorted(s_eval, sn["ts"].to_numpy())
    sn["exp_ms"] = sn["expiration_ts"].astype("int64") // 1000
    sn = sn[sn["exp_ms"] > sn["ts"]].rename(columns={"instrument_name": "inst"})
    sn["K"] = sn["strike"]; sn["sg"] = np.where(sn["option_type"] == "call", 1.0, -1.0); sn["w"] = sn["sg"] * sn["open_interest"]
    E = R.exposures(sn, s_eval, S, "mark_iv", fwd_col="underlying_price")
    out = pd.concat([pd.DataFrame({"ts": s_eval}), E[[f"gex_{sc}" for sc in SCOPES]]], axis=1)
    OUT.mkdir(parents=True, exist_ok=True); out.to_parquet(p)
    return out


def minutes() -> pd.DataFrame:
    """분 패널: Yb(바이낸스 테이커 순 ETH) · c(바이낸스 종가) · Yd(Deribit 무기한, 블록 다리 제외) · Ys = 합."""
    df = pd.DataFrame(index=np.arange(D0 // M_MS, D1 // M_MS)).join(HF.binance_minutes())
    fs = sorted((HF.OUT / "perp").glob("*.parquet"))
    P = pd.concat([pd.read_parquet(f) for f in fs])
    df["Yd"] = P["y_nb"].reindex(df.index)
    got = pd.to_datetime(df.index * 60, unit="s").strftime("%Y-%m-%d").isin([f.stem for f in fs])
    df.loc[got, "Yd"] = df.loc[got, "Yd"].fillna(0.0)             # 받은 날의 빈 분 = 체결 0
    df["Ys"] = df["Yb"] + df["Yd"]
    return df


def attach(df: pd.DataFrame, G: pd.DataFrame, cols: list, lag_ms: int = 0) -> pd.DataFrame:
    """분 t 에 ts ≤ (t+1)·60s − lag 인 가장 최근 G 행을 붙인다(2시간 넘게 묵으면 NaN)."""
    key = pd.DataFrame({"key": (df.index.to_numpy() + 1) * M_MS - lag_ms})
    m = pd.merge_asof(key, G[["ts"] + cols].sort_values("ts"), left_on="key", right_on="ts", direction="backward", tolerance=2 * H_MS)
    return pd.DataFrame(m[cols].to_numpy(), index=df.index, columns=cols)


def prep(df: pd.DataFrame) -> pd.DataFrame:
    """Δp·라벨·Y 시차를 미리 계산(라벨은 t+1 부터)."""
    lc = np.log(df["c"])
    for h in HS:
        df[f"dp{h}"] = 1e4 * (lc - lc.shift(h))
    for k in KR:
        df[f"fr{k}"] = 1e4 * (lc.shift(-k) - lc)
    for y in VEN.values():
        if y not in df:
            continue
        Y = df[y]; cs = Y.fillna(0.0).cumsum(); nn = Y.isna().cumsum()
        for k in KS:
            df[f"{y}_f{k}"] = (cs.shift(-k) - cs).where(nn.shift(-k) == nn)
        df[f"{y}_l4"] = Y.shift(1).rolling(4).sum(); df[f"{y}_l14"] = Y.shift(5).rolling(10).sum()
    m = df.index.to_numpy()
    df["day"] = m // 1440; df["hod"] = (m // 60) % 24
    return df


# ───────────────────────── 회귀 ─────────────────────────
def design(d: pd.DataFrame, g: str, h: int, tgt: str, y: str | None, extra: str | None = None, phys: bool = False):
    """열 순서: 상수, 핵심(z·Δp 또는 물리), Δp, z(또는 GEX$/1e6), [Y_t, Y 시차 둘], [추가], 시 더미. 핵심이 1번 열(HF.ols_boot 규약)."""
    G, dp = d[g].to_numpy(float), d[f"dp{h}"].to_numpy(float)
    ok0 = np.isfinite(G)
    z = (G - G[ok0].mean()) / G[ok0].std()
    cols = [np.ones(len(d))]
    cols += [G * dp / 1e4 * 100 / d["c"].to_numpy(), dp, G / 1e6] if phys else [z * dp, dp, z]
    if y:
        cols += [d[y].to_numpy(), d[f"{y}_l4"].to_numpy(), d[f"{y}_l14"].to_numpy()]
    if extra:
        v = d[extra].to_numpy(float); v = (v - np.nanmean(v)) / np.nanstd(v); cols.append(v * dp)
    Z = np.column_stack(cols + [pd.get_dummies(d["hod"], drop_first=True).to_numpy(float)])
    t = d[tgt].to_numpy(float)
    ok = np.isfinite(Z).all(1) & np.isfinite(t)
    return Z[ok], t[ok], d["day"].to_numpy()[ok]


def fit(d, *a, **kw):
    Z, t, day = design(d, *a, **kw)
    return HF.ols_boot(Z, t, day) if len(np.unique(day)) >= 5 else None


def halves(df, lo, split, hi, *a, **kw) -> dict:
    ts = df.index.to_numpy() * M_MS
    ra = fit(df[(ts >= lo) & (ts < split)], *a, **kw); rb = fit(df[(ts >= split) & (ts < hi)], *a, **kw)
    return {"H_a": ra, "H_b": rb, "pass_neg": bool(ra and rb and ra[2] < 0 and rb[2] < 0), "pass_pos": bool(ra and rb and ra[1] > 0 and rb[1] > 0)}


# ───────────────────────── 메인 ─────────────────────────
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = minutes()
    Hd = pd.read_parquet(R.OUT / "hourly_dealer_gex.parquet"); C = conv_gex()
    Cf = C.copy(); Cf[[f"gex_{sc}" for sc in SCOPES]] *= -1
    gc = [f"gex_{sc}" for sc in SCOPES]
    for nm, G in (("dealer", Hd), ("conv", C), ("conv_flip", Cf)):
        df[[f"{nm}_{c}" for c in gc]] = attach(df, G, gc).to_numpy()
        df[f"{nm}_lag_gex_week"] = attach(df, G, ["gex_week"], H_MS).to_numpy()[:, 0]
    df["rv24"] = attach(df, Hd, ["p24"]).to_numpy()[:, 0]       # 직전 24h log RV(시간별, 그 정시까지)
    df = prep(df)
    res = {"range": {"binance_last": str(pd.Timestamp(int(df["Yb"].last_valid_index()) * M_MS, unit="ms")),
                     "deribit_last": str(pd.Timestamp(int(df["Yd"].last_valid_index()) * M_MS, unit="ms")),
                     "conv_hours": len(C), "conv_first": str(pd.Timestamp(int(C["ts"].min()), unit="ms")),
                     "conv_last": str(pd.Timestamp(int(C["ts"].max()), unit="ms")),
                     "spearman_conv_vs_dealer_week_hourly": round(float(df[["conv_gex_week", "dealer_gex_week"]].dropna().iloc[::60].corr("spearman").iloc[0, 1]), 3)},
           "scale": {f"std_{y}_f{k}": round(float(df[f"{y}_f{k}"].std()), 1) for y in VEN.values() for k in KS}}
    res["scale"] |= {f"std_dp{h}_bp": round(float(df[f"dp{h}"].std()), 2) for h in HS}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    A = {"dealer": ("dealer", D0, SPLIT, D1), "dealer_sw": ("dealer", SW0, SW_SPLIT, D1),
         "conv": ("conv", SW0, SW_SPLIT, D1), "conv_flip": ("conv_flip", SW0, SW_SPLIT, D1)}
    flow, price = {}, {}
    for an, (src, lo, sp, hi) in A.items():
        for sc in SCOPES:
            g = f"{src}_gex_{sc}"
            for vn, y in VEN.items():
                for h in HS:
                    for k in KS:
                        r = halves(df, lo, sp, hi, g, h, f"{y}_f{k}", y)
                        flow[f"{an}|{sc}|{vn}|h{h}|k{k}"] = r
                        print("flow", an, sc, vn, h, k, r, flush=True)
            for h in HS:
                for k in KR:
                    price[f"{an}|{sc}|h{h}|k{k}"] = r = halves(df, lo, sp, hi, g, h, f"fr{k}", None)
                    print("price", an, sc, h, k, r, flush=True)
    res["flow"], res["price"] = flow, price
    res["primary"] = {an: flow[f"{an}|week|sum|h15|k15"] for an in A}
    rob = {}
    for an, (src, lo, sp, hi) in A.items():
        for vn, y in VEN.items():
            for k in KS:
                rob[f"lag1h|{an}|{vn}|k{k}"] = halves(df, lo, sp, hi, f"{src}_lag_gex_week", 15, f"{y}_f{k}", y)
                rob[f"volctl|{an}|{vn}|k{k}"] = halves(df, lo, sp, hi, f"{src}_gex_week", 15, f"{y}_f{k}", y, extra="rv24")
                rob[f"phys|{an}|{vn}|k{k}"] = halves(df, lo, sp, hi, f"{src}_gex_week", 15, f"{y}_f{k}", y, phys=True)
                print("rob", an, vn, k, {x: rob[f"{x}|{an}|{vn}|k{k}"] for x in ("lag1h", "volctl", "phys")}, flush=True)
    res["robust"] = rob
    # 격자 요약: 칸마다 두 반기 CI 가 음수로 0 배제 / 양수로 0 배제 된 수
    res["grid_count"] = {an: {"flow_neg": sum(v["pass_neg"] for kk, v in flow.items() if kk.startswith(an + "|")),
                              "flow_pos": sum(v["pass_pos"] for kk, v in flow.items() if kk.startswith(an + "|")),
                              "flow_n": sum(kk.startswith(an + "|") for kk in flow),
                              "price_neg": sum(v["pass_neg"] for kk, v in price.items() if kk.startswith(an + "|")),
                              "price_pos": sum(v["pass_pos"] for kk, v in price.items() if kk.startswith(an + "|")),
                              "price_n": sum(kk.startswith(an + "|") for kk in price)} for an in A}
    print("primary", json.dumps(res["primary"]), "\ngrid", json.dumps(res["grid_count"]), flush=True)
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("저장", OUT / "results.json")


def selftest() -> None:
    rng = np.random.default_rng(1)
    # 1) 시차 정렬: ts = 정시 h 인 값은 종가가 정확히 h 인 분(h·60−1)부터 쓰인다. 1시간 늦추면 그 분은 h−1 값.
    G = pd.DataFrame({"ts": np.arange(0, 10) * H_MS, "gex_week": np.arange(10.0)})
    d = pd.DataFrame(index=np.arange(0, 600))
    a = attach(d, G, ["gex_week"])["gex_week"]; b = attach(d, G, ["gex_week"], H_MS)["gex_week"]
    assert a[179] == 3 and a[178] == 2 and a[180] == 3 and b[179] == 2 and np.isnan(b[58]) and b[59] == 0
    # 2) 부호 규약: Y_{t+1} = −0.02·GEX_t·Δp15_t (양감마면 오른 뒤 판다) → β < 0, CI 0 배제. GEX 뒤집으면 β > 0.
    n = 1440 * 30
    c = 2000 * np.exp(np.cumsum(rng.standard_normal(n) * 5e-4))
    gh = rng.standard_normal(n // 60 + 1)
    df = pd.DataFrame({"c": c}, index=np.arange(n) + 29_000_000)
    df["g"] = gh[np.arange(n) // 60]; df["gf"] = -df["g"]
    dp = 1e4 * (np.log(c) - np.log(np.roll(c, 15))); dp[:15] = 0
    Y = rng.standard_normal(n) * 5; Y[1:] += -0.02 * df["g"].to_numpy()[:-1] * dp[:-1]
    df["Yb"] = Y
    df = prep(df)
    r = fit(df, "g", 15, "Yb_f5", "Yb"); rf = fit(df, "gf", 15, "Yb_f5", "Yb")
    assert r[0] < 0 and r[2] < 0 and rf[0] > 0 and rf[1] > 0, (r, rf)
    # 3) 같은 분(t) 의 1분 수익에만 심은 반응은 라벨(t+1..)에 안 잡힌다(Δp15 는 겹치는 창이라 1분 수익으로 심는다)
    r1 = 1e4 * np.diff(np.log(c), prepend=np.log(c[0]))
    df2 = df[["c", "g"]].copy(); df2["Yb"] = rng.standard_normal(n) * 5 - 0.3 * df["g"].to_numpy() * r1
    r2 = fit(prep(df2), "g", 15, "Yb_f5", "Yb")
    assert r2[1] < 0 < r2[2], r2
    # 4) 물리 단위: Y_{s} = −GEX$·Δp15_{s−1}·100/(1e4·S) 를 심으면 k=5 합은 겹치는 Δp 창 때문에 −(15+14+13+12+11)/15 = −4.33
    df3 = df[["c"]].copy(); df3["g"] = df["g"] * 1e6
    xp = df3["g"].to_numpy() * dp / 1e4 * 100 / c
    Y3 = rng.standard_normal(n) * 5; Y3[1:] += -xp[:-1]; df3["Yb"] = Y3
    r3 = fit(prep(df3), "g", 15, "Yb_f5", "Yb", phys=True)
    assert -4.45 < r3[0] < -4.2, r3
    # 5) 라벨 합 = Y_{t+1..t+k}
    p = prep(pd.DataFrame({"c": np.ones(20), "Yb": np.arange(20.0)}, index=np.arange(20)))
    assert p["Yb_f5"][3] == 4 + 5 + 6 + 7 + 8 and np.isnan(p["Yb_f5"][15])
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    selftest() if ap.parse_args().selftest else main()
