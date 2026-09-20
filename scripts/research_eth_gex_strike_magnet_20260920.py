"""«딜러가 가격을 스트라이크에 자석처럼 붙인다» — 실제 OI·감마 분포로 검정한다 (2026-09-20).

사용자: *"GEX 는 가격선을 시장 관리자가 자석처럼 붙이는 과정이 필요하다는데 그 엣지를 먹고 싶다"*.

🔴**09-16 에 핀닝을 이미 기각했지만 그 검정에는 구멍이 있다.** 그때는 옵션 데이터가 47스냅샷에서
   멈춰 있어 **라운드 $50/$25/$100 격자**로 대리 검정했다(f=0.49~0.52, 균등 0.500). 그건
   «가격이 라운드 숫자에 붙나»를 물은 것이지 **«실제 OI·감마가 쌓인 스트라이크에 붙나»가 아니다.**
   지금은 체인이 854스냅샷·만기 49개(ETH 는 **매일 08:00 UTC 만기**)라 그 구멍을 막을 수 있다.

사전등록(실행 전 고정):
  H1 자석위치  만기 정산가가 «고감마 스트라이크 K*» 에 **플라시보보다 가깝다**
  H2 끌림      만기 전 1h/4h 수익률이 d=(K*−S)/S 와 **양의 상관**(가격이 K* 쪽으로 간다)
  H3 경제성    d 방향으로 07:00 진입 → 08:00 청산, 순 > 1.4bp (USDC 메이커 왕복)
  판정 = 일군집 부트 CI95 0 배제 AND 플라시보 초과 AND 전·후반 부호 일치

🔴**결정적 통제 두 개**(없으면 이 검정은 반드시 거짓양성을 낸다):
  ①**Γ 는 ATM 에서 최대다** ⇒ «최대 감마 스트라이크»는 기계적으로 스팟을 따라다닌다.
    그러면 d≈0 이라 검정이 공허해진다. 그래서 |d| 분포를 **먼저** 보고, 감마가중(스팟 추종)과
    OI가중(격자에 고정)을 **갈라서** 낸다. 실측 확인: 최근 스냅에서 최대 OI 는 $2,000(스팟 2,575)인데
    최대 감마·OI 는 $2,600 이었다 — 두 «자석»은 다른 곳을 가리킨다.
  ②**플라시보**: 같은 스트라이크 격자 위에서 OI/감마를 **섞는다**. 격자·거리 분포를 보존하고
    «어느 스트라이크인가»만 깬다. (09-20 GEX 방향이 spot² 사본이었던 것과 같은 계열의 방어)

원천: deribit_gex.duckdb::option_chain_snapshot(≤3일 만기만 접음) + ETHUSDT 5분봉.
정산가는 Deribit 지수평균이 아니라 **ETHUSDT 무기한 08:00 UTC 종가**로 대리한다(체결 가능한 값).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp/rt_probe_20260920"
OUT = TMP / "strike_magnet.json"
RNG = np.random.default_rng(20260920)
COST_BP = 1.4          # USDC 메이커 왕복(실측, 2026-09-18)
ENTRY_LEAD_H = 1       # 만기 몇 시간 전에 진입하는가(H3)
R: dict = {}


def say(*a) -> None:
    print(" ".join(str(x) for x in a), flush=True)


def day_boot(vals: np.ndarray, days: np.ndarray, n: int = 2000) -> tuple[float, float]:
    """날 군집 부트스트랩 CI95. 같은 날 관측은 독립이 아니다."""
    uniq = np.unique(days)
    if len(uniq) < 4:
        return (np.nan, np.nan)
    out = []
    for _ in range(n):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        v = np.concatenate([vals[days == d] for d in pick])
        if len(v):
            out.append(v.mean())
    return tuple(np.percentile(out, [2.5, 97.5]))


# ── 입력 ────────────────────────────────────────────────────────────────────
c = pd.read_parquet(TMP / "chain_near.parquet")
c["ts"] = pd.to_datetime(c.recorded_at_utc, utc=True)
c["exp"] = pd.to_datetime(c.expiration_ts, utc=True)
# 🔴duckdb 는 datetime64[us] -- 해상도 무관 변환만 쓴다
c["sec"] = ((c.ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).astype("int64")
c["exp_sec"] = ((c.exp - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).astype("int64")

k = pd.read_parquet(TMP / "klines_5m.parquet")
for col in ("open", "high", "low", "close"):
    k[col] = k[col].astype(float)
k["sec"] = k.open_time // 1000
k = k.sort_values("sec").reset_index(drop=True)
ks, kc = k.sec.to_numpy(), k.close.to_numpy()


def px_at(sec: int) -> float:
    """그 시각 이전에 **닫힌** 5분봉 종가(체결 가능한 값)."""
    i = np.searchsorted(ks + 300, sec, side="right") - 1
    return float(kc[i]) if 0 <= i < len(kc) else np.nan


spacing = np.median(np.diff(np.sort(c.strike.unique())))
say(f"체인 {len(c):,}행 · 스냅샷 {c.ts.nunique()} · 만기 {c.exp.nunique()} · 스트라이크 간격 중앙 ${spacing:.0f}")

# ── 자석 후보 세 가지 ────────────────────────────────────────────────────────
def magnets(grp: pd.DataFrame) -> dict:
    """한 (스냅샷, 만기)의 스트라이크 프로파일에서 «자석» 후보를 뽑는다."""
    g = grp.groupby("strike").agg(oi=("oi", "sum"), goi=("goi", "sum"), goi_signed=("goi_signed", "sum")).reset_index()
    if len(g) < 3:
        return {}
    out = {"K_oi": float(g.strike[g.oi.idxmax()]),          # OI 최대 -- 격자에 고정(스팟 안 따라감)
           "K_goi": float(g.strike[g.goi.idxmax()]),        # 감마·OI 최대 -- ATM 추종(기계적)
           "K_cent": float((g.strike * g.oi).sum() / g.oi.sum())}   # OI 무게중심
    # 플라시보: 같은 격자 위에서 **값만** 섞는다(격자·거리 분포 보존, «어느 스트라이크인가»만 깸).
    # 🔴처음에 두 배열을 같은 순열로 섞었다가 짝이 그대로 보존돼 플라시보가 실제와 소수점까지
    #   같아졌다. 섞는 건 값 하나뿐이고 격자는 고정이다 -- «플라시보가 실제와 똑같다»가 그 지문이었다.
    strikes = g.strike.to_numpy()
    out["K_oi_pl"] = float(strikes[RNG.permutation(g.oi.to_numpy()).argmax()])
    out["K_goi_pl"] = float(strikes[RNG.permutation(g.goi.to_numpy()).argmax()])
    # 🔴대조군 하나 더: **OI 를 아예 안 쓰는** 순수 격자 최근접 스트라이크. 09-16 이 4.8년으로
    #   기각한 바로 그 측도다. 이것과 같은 답이 나오면 «OI·감마 자석»은 새 정보가 아니다.
    spot = float(grp.spot.iloc[0])
    out["K_grid"] = float(strikes[np.abs(strikes - spot).argmin()])
    return out


rows = []
for (snap_sec, exp_sec), grp in c.groupby(["sec", "exp_sec"]):
    m = magnets(grp)
    if not m:
        continue
    s0 = px_at(snap_sec)
    if not np.isfinite(s0):
        continue
    rows.append({"sec": snap_sec, "exp_sec": exp_sec, "spot": s0,
                 "hours_to_exp": (exp_sec - snap_sec) / 3600.0, **m})
M = pd.DataFrame(rows)
M["day"] = pd.to_datetime(M.sec, unit="s").dt.date.astype(str)
say(f"(스냅샷×만기) {len(M):,} · 고유일 {M.day.nunique()}")

# ⭐통제① — 두 자석이 스팟을 얼마나 따라다니나
say("\n## 통제① 자석이 스팟을 따라다니는가 (|K−S|/S, bp)")
for key, lab in (("K_oi", "최대 OI"), ("K_goi", "최대 감마·OI"), ("K_cent", "OI 무게중심")):
    d = (M[key] - M.spot) / M.spot * 1e4
    say(f"  {lab:14} |d| 중앙 {d.abs().median():7.0f}bp · p10 {d.abs().quantile(.1):6.0f} · p90 {d.abs().quantile(.9):7.0f}"
        f"  · corr(K, S) = {M[key].corr(M.spot):+.3f}")
    R.setdefault("control_track", {})[key] = {"abs_med_bp": float(d.abs().median()), "corr_spot": float(M[key].corr(M.spot))}
say(f"  ⇒ corr(K,S) 가 1 에 가까우면 그 «자석»은 스팟의 사본이다(Γ 가 ATM 최대라 기계적).")
say(f"  ⇒ 스트라이크 간격 ${spacing:.0f} = {spacing / M.spot.median() * 1e4:.0f}bp. |d| 가 이보다 작으면 붙일 여지 자체가 없다.")

# ── H1/H2: 만기 직전 창 ──────────────────────────────────────────────────────
say("\n## H1·H2 만기 직전 — 정산가가 자석에 붙나 · 그 방향으로 가나")
E = M[(M.hours_to_exp >= ENTRY_LEAD_H - 0.6) & (M.hours_to_exp <= ENTRY_LEAD_H + 0.6)].copy()
E = E.sort_values("hours_to_exp").groupby("exp_sec", as_index=False).first()      # 만기당 한 관측
E["s_exp"] = [px_at(int(s)) for s in E.exp_sec]
E = E[np.isfinite(E.s_exp)].copy()
E["ret_bp"] = (E.s_exp - E.spot) / E.spot * 1e4
say(f"만기 이벤트 {len(E)} 개 (진입 {ENTRY_LEAD_H}h 전) · 고유일 {E.day.nunique()}")
R["h1h2"] = {"n_expiries": int(len(E))}
say(f"\n{'자석':16} {'진입거리(bp)':>12} {'정산거리(bp)':>12} {'가까워짐':>10} {'끌림 corr':>10} {'방향맞춘 수익(bp)':>18}")
for key, lab in (("K_oi", "최대 OI"), ("K_goi", "최대 감마·OI"), ("K_cent", "OI 무게중심"),
                 ("K_grid", "순수격자 최근접"), ("K_oi_pl", "└플라시보 OI"), ("K_goi_pl", "└플라시보 감마")):
    d0 = (E[key] - E.spot) / E.spot * 1e4
    d1 = (E[key] - E.s_exp) / E.s_exp * 1e4
    closer = float((d1.abs() < d0.abs()).mean())
    corr = float(pd.Series(d0).corr(E.ret_bp, method="spearman"))
    signed = (E.ret_bp * np.sign(d0)).to_numpy()
    lo, hi = day_boot(signed, E.day.to_numpy())
    R["h1h2"][key] = {"d0_med": float(d0.abs().median()), "d1_med": float(d1.abs().median()),
                      "closer": closer, "corr": corr, "signed_mean": float(signed.mean()), "ci": [lo, hi]}
    say(f"{lab:16} {d0.abs().median():12.0f} {d1.abs().median():12.0f} {closer:10.1%} {corr:+10.3f}"
        f" {signed.mean():+9.2f} [{lo:+.1f},{hi:+.1f}]")
say("  ⭐«가까워짐»이 50% 면 동전이다. 플라시보와 갈라지지 않으면 자석이 아니다.")
say(f"  ⭐«방향맞춘 수익» CI 가 0 을 배제하고 비용 {COST_BP}bp 를 넘어야 먹을 수 있다.")

# ── H2 연속판: 매시 스냅샷 전부 ──────────────────────────────────────────────
say("\n## H2 연속 — 만기까지 남은 시간대별 끌림 (방향맞춘 앞 1h 수익, bp)")
M["s1h"] = [px_at(int(s) + 3600) for s in M.sec]
M["ret1h"] = (M.s1h - M.spot) / M.spot * 1e4
bins = [(0, 2), (2, 6), (6, 12), (12, 24), (24, 72)]
say(f"{'남은시간':10} {'n':>6} " + " ".join(f"{lab:>22}" for lab in ("최대 OI", "최대 감마·OI", "순수격자")))
for lo_h, hi_h in bins:
    m = (M.hours_to_exp >= lo_h) & (M.hours_to_exp < hi_h) & M.ret1h.notna()
    if m.sum() < 30:
        continue
    cells = []
    for key in ("K_oi", "K_goi", "K_grid"):
        d = ((M.loc[m, key] - M.loc[m, "spot"]) / M.loc[m, "spot"]).to_numpy()
        sig = (M.loc[m, "ret1h"].to_numpy() * np.sign(d))
        lo_, hi_ = day_boot(sig, M.loc[m, "day"].to_numpy())
        cells.append(f"{sig.mean():+7.2f} [{lo_:+.1f},{hi_:+.1f}]")
        R.setdefault("h2_cont", {}).setdefault(f"{lo_h}-{hi_h}h", {})[key] = {"mean": float(sig.mean()), "ci": [lo_, hi_], "n": int(m.sum())}
    say(f"{lo_h}~{hi_h}h{'':4} {int(m.sum()):6,} " + " ".join(f"{x:>22}" for x in cells))

# ── H3 경제성 ───────────────────────────────────────────────────────────────
say("\n## H3 경제성 — 자석 방향으로 1시간 보유 (만기 직전)")
for key, lab in (("K_oi", "최대 OI"), ("K_goi", "최대 감마·OI")):
    d0 = ((E[key] - E.spot) / E.spot).to_numpy()
    gross = (E.ret_bp.to_numpy() * np.sign(d0))
    net = gross - COST_BP
    lo, hi = day_boot(net, E.day.to_numpy())
    R.setdefault("h3", {})[key] = {"gross": float(gross.mean()), "net": float(net.mean()), "ci": [lo, hi],
                                   "hit": float((gross > 0).mean()), "n": int(len(gross))}
    say(f"  {lab:14} gross {gross.mean():+6.2f} · 순 {net.mean():+6.2f}bp [{lo:+.1f},{hi:+.1f}] · 적중 {(gross > 0).mean():.1%} · n={len(gross)}")
say(f"  (비용 {COST_BP}bp = USDC 메이커 왕복 실측. 테이커면 훨씬 크다.)")

# ── 통제② 🔴«자석»이 사실은 평균회귀인가 ────────────────────────────────────
# sign(d) = «스팟이 최근접 스트라이크 아래인가 위인가»는 **직전에 어느 쪽으로 움직였나**와 상관된다.
# 이 창은 박스권이라(5분 자기상관 −0.16) 평균회귀만으로 «방향맞춘 수익»이 양수가 될 수 있다.
say("\n## 통제② 자석인가 평균회귀인가 — 직전 1시간 수익을 통제한다 (만기 0~2h 창)")
M["s_prev1h"] = [px_at(int(s) - 3600) for s in M.sec]
M["ret_prev"] = (M.spot - M.s_prev1h) / M.s_prev1h * 1e4
near = (M.hours_to_exp >= 0) & (M.hours_to_exp < 2) & M.ret1h.notna() & M.ret_prev.notna()
sub = M[near].copy()
say(f"  n={len(sub)} · 고유일 {sub.day.nunique()}")
for key, lab in (("K_goi", "최대 감마·OI"), ("K_grid", "순수격자"), ("K_oi", "최대 OI")):
    d = ((sub[key] - sub.spot) / sub.spot).to_numpy()
    sig = sub.ret1h.to_numpy() * np.sign(d)
    # 직전 수익 5분위 안에서 같은 통계를 다시 낸다(평균회귀를 셀로 고정)
    # 🔴등분할 5분위의 «칸 평균의 평균»은 전체 평균과 같아진다 -- 그건 통제가 아니다.
    #   칸별 값을 그대로 내서 **부호가 칸마다 유지되는지**를 본다.
    qs = pd.qcut(sub.ret_prev.rank(method="first"), 5, labels=False).to_numpy()
    cells = [round(float(sig[qs == q].mean()), 1) for q in range(5) if (qs == q).sum() >= 8]
    lo, hi = day_boot(sig, sub.day.to_numpy())
    say(f"  {lab:14} 원본 {sig.mean():+6.2f} [{lo:+.1f},{hi:+.1f}] · 직전수익 5분위별 {cells}"
        f" · corr(sign(d), 직전수익) = {np.corrcoef(np.sign(d), sub.ret_prev)[0, 1]:+.3f}")
    R.setdefault("ctrl_meanrev", {})[key] = {"raw": float(sig.mean()), "cells": cells,
                                             "corr_sign_prev": float(np.corrcoef(np.sign(d), sub.ret_prev)[0, 1])}
say("  ⭐corr(sign(d), 직전수익) 이 크게 음수면 «자석 방향»은 «방금 반대로 갔다»와 같은 말이다.")

# ── 통제③ 🔴표본외 — 순수격자 측도는 옵션 데이터가 필요 없다 ────────────────
# 위에서 «감마 자석»과 «순수격자 최근접»이 거의 같은 답을 냈다(corr(K_goi,S)=0.975, |d| 23 vs 18bp).
# 순수격자는 스트라이크 격자와 가격만 있으면 되므로 **GEX 수집 이전 1년**에 같은 통계를 낼 수 있다.
# 37일 양수가 진짜면 1년에서도 살아야 한다. 09-16 은 4.8년 f 측도로 이미 기각했다.
say("\n## 통제③ 표본외 — 순수격자 자석을 GEX 이전 1년에 (2025-09-01~2026-08-15)")
kh = pd.read_parquet(TMP / "klines_5m_hist.parquet")
for col in ("open", "high", "low", "close"):
    kh[col] = kh[col].astype(float)
kh["sec"] = kh.open_time // 1000
kh = kh.sort_values("sec").reset_index(drop=True)
hs, hc = kh.sec.to_numpy(), kh.close.to_numpy()
# 만기 08:00 UTC. 진입은 만기 0~2시간 전(위 창과 같다), 보유 1시간.
utc_h = pd.to_datetime(kh.sec, unit="s").dt.hour.to_numpy()
utc_m = pd.to_datetime(kh.sec, unit="s").dt.minute.to_numpy()
entry = np.flatnonzero((utc_h == 6) & (utc_m == 0) | (utc_h == 7) & (utc_m == 0))    # 만기 2h·1h 전
say(f"  진입 시점 {len(entry)} 개 · 고유일 {pd.to_datetime(kh.sec.to_numpy()[entry], unit='s').date.__len__()}")
for grid in (25.0, 50.0, 100.0):
    sig, days = [], []
    for i in entry:
        j = i + 12                                     # 1시간 뒤
        if j >= len(hc):
            continue
        s0, s1 = hc[i], hc[j]
        k_near = round(s0 / grid) * grid               # 순수격자 최근접 스트라이크
        d = (k_near - s0) / s0
        if d == 0:
            continue
        sig.append((s1 - s0) / s0 * 1e4 * np.sign(d))
        days.append(str(pd.to_datetime(hs[i], unit="s").date()))
    sig, days = np.array(sig), np.array(days)
    lo, hi = day_boot(sig, days)
    R.setdefault("oos_grid", {})[f"${grid:.0f}"] = {"mean": float(sig.mean()), "ci": [lo, hi], "n": int(len(sig)),
                                                    "days": int(len(set(days))), "hit": float((sig > 0).mean())}
    say(f"  격자 ${grid:>5.0f}  방향맞춘 1h 수익 {sig.mean():+6.2f}bp [{lo:+.1f},{hi:+.1f}] · 적중 {(sig > 0).mean():.1%}"
        f" · n={len(sig)} · 날 {len(set(days))}")
say("  ⭐37일 창의 +6.0bp 가 1년에서도 살아 있으면 진짜다. 0 근처로 내려오면 그 +6.0 은 표본 노이즈다.")

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
say("\n->" + str(OUT))


def _selftest() -> None:
    """자석 추출과 부호 규약 -- 값이 아니라 «방향»이 맞는지만 고정한다."""
    df = pd.DataFrame({"strike": [2000.0, 2100.0, 2200.0], "oi": [10.0, 90.0, 5.0],
                       "goi": [1.0, 2.0, 50.0], "goi_signed": [1.0, 2.0, 50.0]})
    df["spot"] = 2010.0
    m = magnets(df)
    assert m["K_oi"] == 2100.0 and m["K_goi"] == 2200.0, m          # 두 자석은 다른 곳을 가리킬 수 있다
    assert m["K_grid"] == 2000.0, m                                  # 순수격자는 OI 를 안 본다
    # 🔴플라시보는 실제와 «달라야» 한다. 값이 한쪽으로 몰린 프로파일에서 여러 번 뽑으면 갈린다.
    picks = {magnets(df)["K_oi_pl"] for _ in range(40)}
    assert len(picks) > 1, picks
    assert 2000.0 < m["K_cent"] < 2200.0
    # 자석이 위에 있으면 d>0 이고, 가격이 오르면 «방향맞춘 수익»이 양수여야 한다
    d = (2100.0 - 2000.0) / 2000.0
    assert np.sign(d) * 10.0 > 0 and np.sign(-d) * 10.0 < 0


_selftest()
say("selftest ok")
