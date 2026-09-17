#!/usr/bin/env python3
"""A2단계 — 펀딩을 진짜 값으로 되돌린 136 피쳐 프레임. (2026-09-17)

A단계의 결함을 고친 판이다. 옛 판은 펀딩 원천이 `TOTAL_ETHUSDT_fundingRate_2025_2026.csv`
(2025~) 뿐이라 **2022·2023·2024 의 `last_funding_rate` 가 전량 중앙값**이었고, 거기서 파생된
9열(`funding_z_score` `ou_funding_z` `funding_pressure` `funding_price_divergence`
`long_squeeze_risk` `squeeze_power` `kel` **`ou_halflife`** …)이 같이 상수가 됐다.
그 상태로 학습창 깊이를 재니 「깊을수록 나쁘다」가 시드 3/3 부호 일관으로 나왔고, 그건
시장이 아니라 내 채움이었다(D단계 §3).

여기서는 `/fapi/v1/fundingRate` 로 2021-11~2026-09 전량(5,343건)을 받아 쓴다.

⭐**연도별 최빈값 점유율 보고를 1급 산출물로 만든다.** 「NaN 0」 은 「값이 있다」지 「정보가
있다」가 아니다 -- A단계의 assert 는 전부 통과했는데도 한 해가 통째로 상수였다.

원천: eth_5m_2021_2023_archive.csv(2021-12~2023-12 완전 OHLCV) + BV 패널 · OI/LSR·BTC 는
BV 패널 · 펀딩은 위 복구본. 2021-12 는 **웜업**으로 쓰고 산출은 2022-01-01 부터다.
"""
import os, sys, numpy as np, pandas as pd, joblib
from pathlib import Path
ROOT = Path(os.environ.get("ZEUS_ROOT") or ("/home/llewyn/crypto-scalping" if Path("/home/llewyn/crypto-scalping").exists() else Path.home()/"crypto-scalping")); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
from features.engineering import FeatureEngineer
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12

OUT = ROOT/"tmp/omega461_longwindow_20260917"; OUT.mkdir(parents=True, exist_ok=True)
OHLC = ["timestamp","open","high","low","close","volume","quote_volume","trades","taker_buy_base","taker_buy_quote"]

def panel(sym):
    d = pd.read_parquet(ROOT/f"data/binance_vision/panel/{sym}USDT.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_localize(None)
    d = d.drop_duplicates("timestamp").sort_values("timestamp")
    typ = (d.high + d.low + d.close) / 3.0
    d["open"] = d.close.shift(1).fillna(d.close)
    d["quote_volume"] = d.volume * typ
    d["taker_buy_quote"] = d.get("taker_buy_base", pd.Series(np.nan, index=d.index)) * typ
    return d.reset_index(drop=True)

def archive(rel):
    d = pd.read_csv(ROOT/rel)
    tc = "open_time" if "open_time" in d.columns else "timestamp"
    d["timestamp"] = pd.to_datetime(d[tc], unit="ms") if d[tc].dtype.kind in "iu" else pd.to_datetime(d[tc])
    return d[OHLC].drop_duplicates("timestamp").sort_values("timestamp")

pe = panel("ETH")
eth = pd.concat([archive("data/eth_5m_2021_2023_archive.csv"), pe[OHLC]], ignore_index=True)
eth = eth.drop_duplicates("timestamp", keep="first").sort_values("timestamp").reset_index(drop=True)
eth = eth.merge(pe[["timestamp","sum_open_interest","sum_toptrader_long_short_ratio","count_long_short_ratio"]],
                on="timestamp", how="left")
eth["sum_open_interest_value"] = eth.sum_open_interest * eth.close
eth = eth.merge(panel("BTC")[["timestamp","close","volume","quote_volume"]].rename(
    columns={"close":"close_btc","volume":"volume_btc","quote_volume":"quote_volume_btc"}), on="timestamp", how="left")
FR = OUT/"funding_2021_2026.csv"
assert FR.exists(), f"펀딩 복구본이 없다: {FR} -- fetch_funding_history 를 먼저 돌린다"
fr = pd.read_csv(FR); fr["timestamp"] = pd.to_datetime(fr["timestamp"])
assert fr.timestamp.min() <= pd.Timestamp("2021-12-01"), f"펀딩이 2021-12 를 못 덮는다: {fr.timestamp.min()}"
eth = pd.merge_asof(eth.sort_values("timestamp"), fr[["timestamp","last_funding_rate"]].sort_values("timestamp"),
                    on="timestamp", direction="backward")
print(f"펀딩 {len(fr):,}건 병합 · 5분봉 결측 {int(eth.last_funding_rate.isna().sum()):,}", flush=True)
eth = eth[eth.timestamp >= "2021-12-01"].reset_index(drop=True)
print(f"원시 {len(eth):,}봉 {eth.timestamp.min()} ~ {eth.timestamp.max()}", flush=True)

fe = FeatureEngineer()
F = fe.process(eth.drop(columns=["close_btc","volume_btc","quote_volume_btc"]).copy(),
               eth[["timestamp","close_btc","volume_btc","quote_volume_btc"]].copy())
if "timestamp" not in F.columns: F["timestamp"] = eth["timestamp"].to_numpy()
F = _with_raw_state12(F)            # state7_*/state12_* 8개 — 기존 열의 결정적 변환
print(f"엔지니어링+state12 후 {F.shape[1]}열", flush=True)

art = joblib.load(ROOT/"tmp/eth_regime_balnobb_20260910/model.joblib")
fcols, med = art["feature_cols"], art["feature_medians"]
miss = [c for c in fcols if c not in F.columns]
print(f"balnobb 136 중 생성 {len(fcols)-len(miss)} · 없음 {len(miss)}: {miss}")
assert not miss, "136 열이 다 안 만들어졌다 — 여기서 멈춘다"

# 🔴펀딩 3개 등 결측을 아티팩트 중앙값으로 채운다(사용자 결정). 채운 양을 반드시 기록한다.
filled = {}
for c in fcols:
    n = int(F[c].isna().sum())
    if n: F[c] = F[c].fillna(med[c]); filled[c] = n
F = F[F.timestamp >= "2022-01-01"].reset_index(drop=True)
F.to_parquet(OUT/"features_136_2022_2026_realfunding.parquet", index=False)
print(f"\n저장 {OUT/'features_136_2022_2026_realfunding.parquet'} · {len(F):,}행 × {F.shape[1]}열")
print(f"중앙값으로 채운 열 {len(filled)}개 (상위 6):")
for c,n in sorted(filled.items(), key=lambda x:-x[1])[:6]:
    print(f"   {c:34s} {n:7,}행 ({n/len(F)*100:5.1f}%) → {med[c]:.6g}")
print(f"\n연도별 봉수: {F.groupby(F.timestamp.dt.year).size().to_dict()}")
print(f"136열 NaN 잔존: {int(F[fcols].isna().sum().sum())}")


# ── ⭐연도별 상수성 보고 (D단계에서 나를 틀리게 만든 그 검사) ──
yr = F.timestamp.dt.year
years = sorted(yr.unique())
bad = []
for c in fcols:
    v = pd.to_numeric(F[c], errors="coerce")
    occ = {}
    for y in years:
        s = v[yr == y]
        occ[y] = float((s == s.mode().iloc[0]).mean()) if not s.mode().empty else float("nan")
    if max(occ.values()) >= 0.50 and min(occ.values()) <= 0.10:
        bad.append((c, occ))
print(f"\n=== 연도간 상수성 격차가 큰 열: {len(bad)}개 (연도 {years}) ===")
for c, o in bad:
    print(f"   {c:34s} " + " ".join(f"{y}:{o[y]:.3f}" for y in years))

# 알려진 잔존 결손: BV 패널의 롱숏비가 2022 를 못 덮는다(바이낸스 자체 이력 한계).
KNOWN = {"sum_toptrader_long_short_ratio", "whale_conviction", "ofti", "whale_retail_ratio"}
surprise = [c for c, _ in bad if c not in KNOWN]
assert not surprise, f"예상 못 한 상수열: {surprise} -- 여기서 멈춘다"
print(f"\n잔존 {len(bad)}개는 전부 알려진 롱숏비 계열(2022 한정). 펀딩 계열은 0개. ✅")
