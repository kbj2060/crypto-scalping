#!/usr/bin/env python3
"""ETH **추세(지속) 증거신호 8종 v1** -- 사전등록 스크린 (2026-09-04).

사용자: "추세신호도 반전 감지기처럼 연구·조사해서 8개 만들어줘 (최신 논문·외부 문헌·창의적 아이디어 반영)".
설계 문서: docs/experiments/eth_trend_signals_v1_design_20260904.md

## 반전 8종과 다른 점 (이번 세션의 교훈을 반영)
- 08-31 추세 셋업 10종·돌파 지속·지속 헤드가 전부 죽은 자리는 **라벨**(K×ATR 터치 = 변동성 라벨)과 **대조군**(무작위 봉 lift)이었다.
  여기서는 처음부터 **두 측면 경제라벨**(같은 봉의 신호 방향 vs 반대 방향, sim_exit 5.0/1.5/0.1·200봉·10bp, F0 프레임 원문)로
  방향 정보를 직접 잰다: P(신호 방향 > 반대), 순손익 차이의 일군집 CI. 반대 방향이 곧 뒤집기 대조군이다.
- 모집단은 **raw 인과 트리거의 첫발동(GAP=12)**. 클러스터 앵커 없음(5.16절).

## 8종 (방향 = 추세 지속 방향; 각 신호의 문헌/메커니즘은 설계 문서)
  T1 quarter_hour_boundary_flow  15분 경계 봉의 테이커 불균형 극단 + 레인지 확장 (Quarter-Hour Effect, arXiv 2607.09426: 경계 주문불균형이 4~12h 수익 예측)
  T2 session_open_range_breakout 세션(00:00/08:00/13:30 UTC) 첫 30분 레인지의 첫 종가 돌파 + 거래량 (intraday momentum, Wen et al. 2022 late-informed investors)
  T3 oi_confirmed_breakout       48봉 고/저 종가 돌파 + OI 증가(신규 포지션) + 거래량 (파생 '새 돈' 돌파; 08-31 유일 생존 조합의 재설계)
  T4 funding_squeeze             펀딩 극단(쏠림) 상태에서 쏠린 쪽 불리 방향 24봉 돌파 → 청산 연료 지속 (funding 피드백·청산 캐스케이드 문헌)
  T5 regime_pullback_resume      레짐(S12_K3 OOF) 방향 안에서 되돌림 후 재개(신저/신고) (오늘 확인: 방향 일치 레짐이 지속을 강화)
  T6 btc_leadlag                 BTC 15분 충격(z≥2)에 ETH가 절반도 못 따라간 봉 → ETH가 BTC 방향으로 추격 (교차자산 lead-lag; SMT의 지속 해석)
  T7 spot_led_move               15분 이동 ≥1 ATR이면서 현물이 선도(현물 수익률 ≥ 선물) → 실수요 이동 지속 (현물/선물 가격발견 문헌)
  T8 liquidity_vacuum            15분 이동 ≥0.7 ATR 방향 앞쪽 ±1% 호가 깊이가 1시간 전 대비 40% 이상 증발 (유동성 이탈·캐스케이드 문헌; bookDepth 공개 30초)

## 판정 (결과 보기 전 고정)
  TRAIN(발견): 양측 합산 n ≥ 300, P(신호>반대) ≥ 0.53, 차이 일군집 CI 하한 > 0
  VAL·OOS(각 1회): 차이 > 0 둘 다  → PASS(메타라벨·게이트 단계로)   / 한쪽만 → WEAK / 그 외 → REJECT
  부수: 12봉 1.0×ATR 터치 적중률 vs 매봉 기저(반전 8종과 같은 척도), 발동 빈도/일. HOLDOUT(≥2026-04-01) 미접촉.
"""
from __future__ import annotations

import glob
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KL = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
KL_BTC = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
KL_SPOT = ROOT / "binance_data/klines_spot/ETHUSDT/ETHUSDT-5m-spot.csv"
FRAME = ROOT / "tmp/homer_entry_v2_20260904/frame.parquet"
REG = ROOT / "tmp/eth_entry_oof_regime_20260903/regime_oof_eth.parquet"
BD_DIR = ROOT / "binance_data/bookDepth/ETHUSDT"
OUT = ROOT / "data/research/eth_trend_signals_v1_screen_20260904"
GAP, H_TOUCH, K_TOUCH, B_BOOT = 12, 12, 1.0, 500
START, END = pd.Timestamp("2024-04-01"), pd.Timestamp("2026-04-01")
SIGNALS = ["quarter_hour_boundary_flow", "session_open_range_breakout", "oi_confirmed_breakout", "funding_squeeze",
           "regime_pullback_resume", "btc_leadlag", "spot_led_move", "liquidity_vacuum"]


def log(m): print(f"[trend-v1] {m}", flush=True)


def first_fire(mask, gap=GAP):
    keep = np.zeros(len(mask), bool); last = -10**9
    for i in np.flatnonzero(mask):
        if i - last > gap:
            keep[i] = True
        last = i
    return keep


def roll_z(x: pd.Series, w=288, minp=144):
    m = x.rolling(w, min_periods=minp).mean(); s = x.rolling(w, min_periods=minp).std()
    return (x - m) / s.replace(0, np.nan)


def load_klines(path, prefix=""):
    d = pd.read_csv(path, parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")
    if d["timestamp"].dt.tz is not None:
        d["timestamp"] = d["timestamp"].dt.tz_localize(None)
    d = d.loc[(d["timestamp"] >= START - pd.Timedelta(days=10)) & (d["timestamp"] < END)].reset_index(drop=True)
    return d if not prefix else d.rename(columns={c: f"{prefix}{c}" for c in d.columns if c != "timestamp"})


def load_metrics():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "binance_data/metrics/ETHUSDT-metrics-*.zip"))):
        day = f[-14:-4]
        if not ("2024-03-20" <= day <= "2026-03-31"):
            continue
        z = zipfile.ZipFile(f); rows.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0]))))
    m = pd.concat(rows, ignore_index=True); m["ts"] = pd.to_datetime(m["create_time"])
    return m.sort_values("ts")[["ts", "sum_open_interest_value", "count_long_short_ratio", "sum_toptrader_long_short_ratio", "sum_taker_long_short_vol_ratio"]]


def load_funding():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "binance_data/funding_rate/ETHUSDT-fundingRate-*.zip"))):
        z = zipfile.ZipFile(f); rows.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0]))))
    fu = pd.concat(rows, ignore_index=True); fu["ts"] = pd.to_datetime(fu["calc_time"], unit="ms"); fu = fu.sort_values("ts")
    fu["funding_z"] = roll_z(fu["last_funding_rate"], w=90, minp=45)          # 30일(8h×90)
    return fu[["ts", "last_funding_rate", "funding_z"]]


def load_bookdepth():
    cache = OUT / "bookdepth_wide.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    parts = []
    for f in sorted(glob.glob(str(BD_DIR / "ETHUSDT-bookDepth-*.zip"))):
        try:
            z = zipfile.ZipFile(f); d = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
        except Exception:                                          # noqa: BLE001
            continue
        w = d.pivot_table(index="timestamp", columns="percentage", values="notional", aggfunc="last")
        w.columns = [f"{'up' if c > 0 else 'dn'}{abs(int(c))}" for c in w.columns]
        parts.append(w.reset_index())
    bd = pd.concat(parts, ignore_index=True); bd["ts"] = pd.to_datetime(bd["timestamp"]); bd = bd.drop(columns=["timestamp"]).sort_values("ts")
    OUT.mkdir(parents=True, exist_ok=True); bd.to_parquet(cache, index=False); return bd


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    kl = load_klines(KL); n = len(kl)
    o, h, l, c, v, tb = (kl[x].to_numpy(float) for x in ("open", "high", "low", "close", "volume", "taker_buy_base"))
    ts = kl["timestamp"]; close_ts = ts + pd.Timedelta(minutes=5)
    prev = np.r_[np.nan, c[:-1]]; tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    vol_z = roll_z(kl["volume"]).to_numpy(); imb = (2 * tb - v) / np.where(v > 0, v, np.nan); imb_z = roll_z(pd.Series(imb)).to_numpy()
    rng_atr = (h - l) / atr
    ret3 = c / np.r_[np.nan, np.nan, np.nan, c[:-3]] - 1.0
    hi48 = pd.Series(h).shift(1).rolling(48).max().to_numpy(); lo48 = pd.Series(l).shift(1).rolling(48).min().to_numpy()
    hi24 = pd.Series(h).shift(1).rolling(24).max().to_numpy(); lo24 = pd.Series(l).shift(1).rolling(24).min().to_numpy()
    log(f"klines {n:,} {ts.iloc[0]} ~ {ts.iloc[-1]} ({time.time()-t0:.0f}s)")

    # ---- 보조 데이터 (전부 봉 마감 이전 값만 merge_asof backward) ----
    btc = load_klines(KL_BTC, "btc_"); kl2 = kl[["timestamp"]].merge(btc[["timestamp", "btc_close"]], on="timestamp", how="left")
    bc = kl2["btc_close"].ffill().to_numpy(); bret3 = bc / np.r_[np.nan, np.nan, np.nan, bc[:-3]] - 1.0
    z_e = roll_z(pd.Series(ret3)).to_numpy(); z_b = roll_z(pd.Series(bret3)).to_numpy()
    spot = load_klines(KL_SPOT, "spot_"); kl3 = kl[["timestamp"]].merge(spot[["timestamp", "spot_close"]], on="timestamp", how="left")
    sc = kl3["spot_close"].ffill().to_numpy(); sret3 = sc / np.r_[np.nan, np.nan, np.nan, sc[:-3]] - 1.0
    met = load_metrics(); m = pd.merge_asof(pd.DataFrame({"ts": close_ts}), met, on="ts", direction="backward")
    oi = m["sum_open_interest_value"].to_numpy(float); oi_chg6 = oi / np.r_[[np.nan] * 6, oi[:-6]] - 1.0
    fu = load_funding(); f = pd.merge_asof(pd.DataFrame({"ts": close_ts}), fu, on="ts", direction="backward"); fz = f["funding_z"].to_numpy(float)
    reg = pd.read_parquet(REG); reg["timestamp"] = pd.to_datetime(reg["timestamp"])
    rg = kl[["timestamp"]].merge(reg[["timestamp", "regime_oof"]], on="timestamp", how="left")["regime_oof"].fillna(-1).astype(int).to_numpy()
    bd = load_bookdepth(); b = pd.merge_asof(pd.DataFrame({"ts": close_ts}), bd, on="ts", direction="backward")
    b1h = pd.merge_asof(pd.DataFrame({"ts": close_ts - pd.Timedelta(hours=1)}), bd, on="ts", direction="backward")
    up1, dn1 = b["up1"].to_numpy(float), b["dn1"].to_numpy(float); up1h, dn1h = b1h["up1"].to_numpy(float), b1h["dn1"].to_numpy(float)
    log(f"aux: btc {np.isfinite(bc).mean():.3f} spot {np.isfinite(sc).mean():.3f} oi {np.isfinite(oi).mean():.3f} funding {np.isfinite(fz).mean():.3f} regime {(rg>=0).mean():.3f} bookdepth {np.isfinite(up1).mean():.3f} ({time.time()-t0:.0f}s)")

    # ---- 트리거 (up = 롱 지속, dn = 숏 지속) ----
    T = {}
    minute = close_ts.dt.minute.to_numpy(); boundary = np.isin(minute, [0, 15, 30, 45])
    T["quarter_hour_boundary_flow"] = (boundary & (imb_z >= 2.0) & (rng_atr >= 1.0), boundary & (imb_z <= -2.0) & (rng_atr >= 1.0))
    up = np.zeros(n, bool); dn = np.zeros(n, bool)
    tod = ts.dt.hour * 60 + ts.dt.minute
    for start_min in (0, 8 * 60, 13 * 60 + 30):
        sess_start = np.flatnonzero((tod == start_min).to_numpy())
        for s0 in sess_start:
            if s0 + 30 >= n:
                continue
            rh, rl = h[s0:s0 + 6].max(), l[s0:s0 + 6].min()
            for i in range(s0 + 6, min(s0 + 30, n)):
                if vol_z[i] >= 1.0 and c[i] > rh:
                    up[i] = True; break
                if vol_z[i] >= 1.0 and c[i] < rl:
                    dn[i] = True; break
    T["session_open_range_breakout"] = (up, dn)
    T["oi_confirmed_breakout"] = ((c > hi48) & (oi_chg6 >= 0.003) & (vol_z >= 1.0), (c < lo48) & (oi_chg6 >= 0.003) & (vol_z >= 1.0))
    T["funding_squeeze"] = ((fz <= -1.5) & (c > hi24), (fz >= 1.5) & (c < lo24))
    lo_ref = pd.Series(l).shift(6).rolling(30).min().to_numpy(); hi_ref = pd.Series(h).shift(6).rolling(30).max().to_numpy()
    bounce_up = pd.Series(h).shift(1).rolling(5).max().to_numpy() - lo_ref; bounce_dn = hi_ref - pd.Series(l).shift(1).rolling(5).min().to_numpy()
    T["regime_pullback_resume"] = ((rg == 0) & (bounce_dn >= 0.75 * atr) & (c > hi_ref), (rg == 1) & (bounce_up >= 0.75 * atr) & (c < lo_ref))
    T["btc_leadlag"] = ((z_b >= 2.0) & (z_e < 0.5 * z_b), (z_b <= -2.0) & (z_e > 0.5 * z_b))
    mv3 = (c - np.r_[np.nan, np.nan, np.nan, c[:-3]]) / atr
    T["spot_led_move"] = ((mv3 >= 1.0) & (sret3 >= ret3), (mv3 <= -1.0) & (sret3 <= ret3))
    T["liquidity_vacuum"] = ((mv3 >= 0.7) & (up1 <= 0.6 * up1h), (mv3 <= -0.7) & (dn1 <= 0.6 * dn1h))

    # ---- 라벨: F0 프레임(두 측면 경제라벨) + 터치 라벨 ----
    D = pd.read_parquet(FRAME, columns=["pos", "is_downside", "timestamp", "split", "net_bp", "net_bp_flip", "exit_off"])
    D["timestamp"] = pd.to_datetime(D["timestamp"])
    key = D.set_index(["timestamp", "is_downside"])
    base = {w: {"long": round(float(D.loc[(D.split == w) & (D.is_downside == 1), "net_bp"].mean()), 2), "short": round(float(D.loc[(D.split == w) & (D.is_downside == 0), "net_bp"].mean()), 2)} for w in ("TRAIN", "VAL", "OOS")}
    fmax = pd.Series(h).shift(-1)[::-1].rolling(H_TOUCH, min_periods=1).max()[::-1].to_numpy(); fmin = pd.Series(l).shift(-1)[::-1].rolling(H_TOUCH, min_periods=1).min()[::-1].to_numpy()
    hit_up = (fmax - c) >= K_TOUCH * atr; hit_dn = (c - fmin) >= K_TOUCH * atr
    split_of = D.drop_duplicates("timestamp").set_index("timestamp")["split"]; sp = kl[["timestamp"]].merge(split_of.rename("split"), left_on="timestamp", right_index=True, how="left")["split"].to_numpy()
    touch_base = {w: {"up": round(float(np.nanmean(hit_up[sp == w])), 4), "dn": round(float(np.nanmean(hit_dn[sp == w])), 4)} for w in ("TRAIN", "VAL", "OOS")}
    rng = np.random.default_rng(20260904)

    def day_ci(x, t):
        d = pd.Series(np.asarray(x, float), index=pd.DatetimeIndex(t).normalize()); days = d.index.unique().to_numpy(); g = d.groupby(level=0)
        sums = g.sum().reindex(days).to_numpy(); cnts = g.count().reindex(days).to_numpy(); out = np.empty(B_BOOT)
        for k in range(B_BOOT):
            j = rng.integers(0, len(days), len(days)); out[k] = sums[j].sum() / max(cnts[j].sum(), 1)
        return round(float(np.percentile(out, 2.5)), 2), round(float(np.percentile(out, 97.5)), 2)

    rep = {"prereg": "TRAIN n>=300 & P(sig>opp)>=0.53 & diff dayCI low>0 ; VAL&OOS diff>0 -> PASS / one -> WEAK / else REJECT", "baseline_net_bp": base,
           "touch_base": touch_base, "signals": {}}
    print(f"\n{'signal':>28s} {'side':>4s} {'w':>5s} {'n':>5s} {'/day':>5s} {'P(s>o)':>7s} {'sig_bp':>7s} {'opp_bp':>7s} {'diffCI':>16s} {'touch':>6s}")
    for name in SIGNALS:
        up_m, dn_m = T[name]; R = {"sides": {}}
        rows_all = []
        for side, mask, isd, hit in (("up", up_m, 1, hit_up), ("dn", dn_m, 0, hit_dn)):
            ff = first_fire(np.nan_to_num(mask.astype(float)).astype(bool) if mask.dtype != bool else mask)
            idx = np.flatnonzero(ff); idx = idx[(ts.iloc[idx] >= START).to_numpy()]
            r = key.reindex(pd.MultiIndex.from_arrays([ts.iloc[idx].to_numpy(), np.full(len(idx), isd)], names=["timestamp", "is_downside"]))
            ok = np.isfinite(r["net_bp"].to_numpy()); r = r[ok].reset_index(); r["side"] = side; r["touch"] = hit[idx][ok]; rows_all.append(r)
            R["sides"][side] = {"raw_fires": int(mask.sum()), "first_fires": int(len(r))}
        A = pd.concat(rows_all, ignore_index=True) if rows_all else pd.DataFrame()
        for w in ("TRAIN", "VAL", "OOS"):
            for side in ("up", "dn", "both"):
                s = A[(A.split == w)] if side == "both" else A[(A.split == w) & (A.side == side)]
                if len(s) < 20:
                    R.setdefault(w, {})[side] = {"n": int(len(s))}; continue
                days = pd.DatetimeIndex(s["timestamp"]).normalize().nunique(); diff = (s["net_bp"] - s["net_bp_flip"]).to_numpy()
                lo, hi = day_ci(diff, s["timestamp"])
                R.setdefault(w, {})[side] = {"n": int(len(s)), "per_day": round(len(s) / max(days, 1), 2), "p_sig_gt_opp": round(float((s.net_bp > s.net_bp_flip).mean()), 3),
                                              "sig_bp": round(float(s.net_bp.mean()), 2), "opp_bp": round(float(s.net_bp_flip.mean()), 2), "diff_bp": round(float(diff.mean()), 2),
                                              "diff_day_ci95": [lo, hi], "touch_hit": round(float(s["touch"].mean()), 3)}
                print(f"{name:>28s} {side:>4s} {w:>5s} {len(s):5d} {len(s)/max(days,1):5.1f} {(s.net_bp > s.net_bp_flip).mean():7.3f} {s.net_bp.mean():7.2f} {s.net_bp_flip.mean():7.2f} {str([lo, hi]):>16s} {s['touch'].mean():6.3f}")
        tr_ = R.get("TRAIN", {}).get("both", {}); va_ = R.get("VAL", {}).get("both", {}); oo_ = R.get("OOS", {}).get("both", {})
        train_ok = tr_.get("n", 0) >= 300 and tr_.get("p_sig_gt_opp", 0) >= 0.53 and tr_.get("diff_day_ci95", [-1])[0] > 0
        vo = int(va_.get("diff_bp", -1) > 0) + int(oo_.get("diff_bp", -1) > 0)
        R["verdict"] = "PASS" if (train_ok and vo == 2) else ("WEAK" if (train_ok and vo == 1) else "REJECT")
        rep["signals"][name] = R
        log(f"{name}: {R['verdict']}  TRAIN {tr_}  VAL diff {va_.get('diff_bp')} OOS diff {oo_.get('diff_bp')}")
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    print("\nbaseline net (long/short):", base, "\ntouch base:", touch_base)
    print("verdicts:", {k: v["verdict"] for k, v in rep["signals"].items()})
    log(f"완료 -> {OUT/'report.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
