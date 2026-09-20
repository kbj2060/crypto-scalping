"""GEX(옵션 감마 노출)를 다른 원천과 같은 두 프레임으로 — 상태 서술 + 예측, 그리고 카드 통합 질문.

⚠️**판정하지 않는다.** GEX 를 «변동성 예측 증분»으로 승격하는 건 2026-09-16 에 사전등록됐고
   판정일이 **1h 09-28 · 4h 10-17**(그때 독립일 44/63)이다. 오늘(09-20) 독립일 37 에서 그 검정을
   다시 돌려 좋게 나왔다고 쓰면 훔쳐보기다. 여기서 하는 것은 셋뿐:
     ①상태 서술(분포·유효시간·에피소드 — 판정이 아니다)
     ②기존 결론의 재현 확인(ρ(GEX, 전방RV) 부호가 이론과 반대인가)
     ③**새 질문**: GEX 레짐이 «미시 참고» 카드의 다른 칩(60초 거래량 분위, QI×OFI 방아쇠)이
       말하는 것을 바꾸는가. 이건 사전등록된 vol-forecast 증분과 다른 질문이라 지금 물어도 된다.

원천: data/live/deribit_gex.duckdb::gex_summary (매시 cron, ETH 854 스냅샷 · 08-15~09-20 · 고유일 37)
     + ETHUSDT 5분봉(같은 구간) + 1초 패널(4.6일 겹침).
통계 단위는 **날**이다 — 시간별 스냅샷은 서로 독립이 아니다(ACF 아래에서 잰다).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp/rt_probe_20260920"
OUT = TMP / "gex_frames.json"
R: dict = {}
pd.set_option("display.width", 200, "display.max_columns", 30)


def say(*a) -> None:
    print(" ".join(str(x) for x in a), flush=True)


# ── 입력 ────────────────────────────────────────────────────────────────────
g = pd.read_parquet(TMP / "gex_summary.parquet")
g = g[g.currency == "ETH"].copy().sort_values("recorded_at_utc").reset_index(drop=True)
g["ts"] = pd.to_datetime(g.recorded_at_utc, utc=True)
# 🔴duckdb 는 datetime64[**us**] 로 준다 -- astype("int64")//10**9 는 마이크로초를 나눠 조용히 1970년이 된다.
#   해상도 무관 변환만 쓴다(이 함정으로 조인이 0행이 됐다).
g["sec"] = ((g.ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).astype("int64")
g["day"] = g.ts.dt.tz_convert("UTC").dt.date.astype(str)
g["front_ratio"] = g.front_month_gex_usd / g.total_gex_usd.replace(0, np.nan)

k = pd.read_parquet(TMP / "klines_5m.parquet")
for c in ("open", "high", "low", "close", "volume"):
    k[c] = k[c].astype(float)
k["sec"] = k.open_time // 1000
k = k.sort_values("sec").reset_index(drop=True)
k["ret"] = np.log(k.close).diff()
BARS = {"1h": 12, "4h": 48, "24h": 288}
for name, n in BARS.items():
    fwd_hi = k.high[::-1].rolling(n, min_periods=n // 2).max()[::-1].shift(-1)
    fwd_lo = k.low[::-1].rolling(n, min_periods=n // 2).min()[::-1].shift(-1)
    k[f"frng_{name}"] = (fwd_hi - fwd_lo) / k.close * 1e4                      # 앞 H 고저폭(bp)
    k[f"fret_{name}"] = (np.log(k.close.shift(-n)) - np.log(k.close)) * 1e4    # 앞 H 수익(bp)
    k[f"frv_{name}"] = k.ret[::-1].rolling(n, min_periods=n // 2).std()[::-1].shift(-1) * np.sqrt(n) * 1e4
    k[f"trv_{name}"] = k.ret.rolling(n, min_periods=n // 2).std() * np.sqrt(n) * 1e4   # 후행 RV(대조군)

# GEX 스냅샷 -> 그 시각 이전에 **닫힌** 5분봉 (엄격히 인과적)
idx = np.searchsorted(k.sec.to_numpy() + 300, g.sec.to_numpy(), side="right") - 1
ok = idx >= 0
G = g[ok].reset_index(drop=True)
kk = k.iloc[idx[ok]].reset_index(drop=True)
for c in [c for c in k.columns if c.startswith(("frng_", "fret_", "frv_", "trv_"))] + ["close", "sec"]:
    G[f"k_{c}"] = kk[c].to_numpy()
G["lag_s"] = G.sec - G.k_sec                       # 스냅샷이 그 봉 마감보다 얼마나 뒤인가
say(f"조인 {len(G)} 스냅샷 · 고유일 {G.day.nunique()} · 봉 마감 후 지연 중앙 {G.lag_s.median():.0f}초")


def day_stat(df: pd.DataFrame, fn) -> tuple[float, float, int]:
    """날 블록 평균±SE. 시간별 스냅샷은 독립이 아니므로 날 단위로 접는다."""
    vals = []
    for _, gg in df.groupby("day"):
        v = fn(gg)
        if v is not None and np.isfinite(v):
            vals.append(v)
    a = np.array(vals)
    return (a.mean(), a.std(ddof=1) / np.sqrt(len(a)), len(a)) if len(a) > 2 else (np.nan, np.nan, len(a))


def ic(df: pd.DataFrame, x: str, y: str, min_n: int = 8) -> tuple[float, float, int]:
    return day_stat(df, lambda gg: gg[[x, y]].dropna().corr(method="spearman").iloc[0, 1]
                    if len(gg[[x, y]].dropna()) >= min_n else None)


def f3(t) -> str:
    return f"{t[0]:+.3f}±{t[1]:.3f}" if np.isfinite(t[0]) else "n/a"


# ── A. 상태 프레임 ───────────────────────────────────────────────────────────
say("\n" + "=" * 96)
say("## A. 상태 서술 — GEX 를 «지금 상태»로 어떻게 읽나")
gap = np.diff(G.sec)
say(f"갱신 간격(초): 중앙 {np.median(gap):.0f} · p10 {np.percentile(gap, 10):.0f} · p90 {np.percentile(gap, 90):.0f} · 최대 {gap.max():.0f}")
say(f"  ⇒ 매시 cron. 화면이 보는 값은 **최대 1시간 묵은 것**이고, 그게 이 지표의 해상도다.")
q = G.total_gex_usd.quantile([0.05, 0.25, 0.5, 0.75, 0.95]) / 1e6
say(f"total GEX($M): p5 {q.iloc[0]:.1f} · p25 {q.iloc[1]:.1f} · p50 {q.iloc[2]:.1f} · p75 {q.iloc[3]:.1f} · p95 {q.iloc[4]:.1f}"
    f"  · 음(total) {(G.total_gex_usd < 0).mean():.1%}")
qf = G.front_ratio.quantile([0.05, 0.5, 0.95])
say(f"front/total: p5 {qf.iloc[0]:+.2f} · p50 {qf.iloc[1]:+.2f} · p95 {qf.iloc[2]:+.2f} · 음(front) {(G.front_month_gex_usd < 0).mean():.1%}")
say("  🔴이 표본에서 **total 감마는 한 번도 음수가 아니었다**(0.0%). 음감마는 front 에서만 6.7%.")
# ACF: 스냅샷 간격이 1시간이므로 lag 단위가 곧 시간
lv = np.log(G.total_gex_usd.clip(lower=1))
acf = {f"{h}h": float(pd.Series(lv).autocorr(h)) for h in (1, 4, 12, 24, 48)}
say("log(total) ACF: " + " · ".join(f"{k_}={v:+.2f}" for k_, v in acf.items()))
say(f"  ⇒ 24시간 뒤에도 {acf['24h']:+.2f} — **하루 단위로 움직이는 느린 지표**다. 1초 카드의 다른 칩과 시간축이 다르다.")
# 에피소드: 상·하위 20% 상태가 얼마나 이어지나(시간 단위)
rk = G.total_gex_usd.rank(pct=True)
st = pd.Series(np.where(rk >= 0.8, "높음", np.where(rk <= 0.2, "낮음", "보통")), index=G.index)
chg = (st != st.shift()).cumsum()
runs = st.groupby(chg).agg(["first", "size"])
ep = {s: {"n": int(len(gg)), "median_h": float(gg["size"].median()), "p90_h": float(gg["size"].quantile(0.9))}
      for s, gg in runs.groupby("first")}
say("에피소드(시간): " + " · ".join(f"{s} n={v['n']} 중앙 {v['median_h']:.0f}h p90 {v['p90_h']:.0f}h" for s, v in ep.items()))
R["state"] = {"gap_median_s": float(np.median(gap)), "acf": acf, "episodes": ep,
              "neg_total_share": float((G.total_gex_usd < 0).mean()),
              "neg_front_share": float((G.front_month_gex_usd < 0).mean()),
              "total_q": {str(kq): float(v) for kq, v in (G.total_gex_usd.quantile([0.05, 0.5, 0.95])).items()}}

# ── B. 예측 프레임 (기존 결론 재현 확인 — 판정 아님) ─────────────────────────
say("\n" + "=" * 96)
say("## B. 예측 — 기존 결론(부호가 이론과 반대)이 4일 늘어난 표본에서도 같은가")
say(f"{'지평':6} {'IC(GEX, 앞 고저폭)':>22} {'IC(GEX, 앞 RV)':>20} {'IC(GEX, 앞 수익)':>20} {'IC(후행RV, 앞 RV)':>20}")
R["pred"] = {}
for name in BARS:
    a, b, c, d = (ic(G, "total_gex_usd", f"k_frng_{name}"), ic(G, "total_gex_usd", f"k_frv_{name}"),
                  ic(G, "total_gex_usd", f"k_fret_{name}"), ic(G, f"k_trv_{name}", f"k_frv_{name}"))
    R["pred"][name] = {"frng": a[:2], "frv": b[:2], "fret": c[:2], "trv_frv": d[:2], "days": a[2]}
    say(f"{name:6} {f3(a):>22} {f3(b):>20} {f3(c):>20} {f3(d):>20}")
say("  ⭐부호가 **양수**면 «GEX 높을수록 앞으로 더 움직인다» = 이론(딜러 감마가 변동성을 누른다)과 반대.")
say("  ⭐대조군: 후행 RV 는 GEX 없이도 앞 RV 를 안다. GEX 가 그보다 강하지 않으면 새 정보가 아니다.")
# 후행 RV 를 통제한 뒤에도 남는가 (순위 잔차)
from numpy.linalg import lstsq


def partial(df: pd.DataFrame, x: str, y: str, ctrl: list[str]) -> tuple[float, float, int]:
    def f(gg):
        d = gg[[x, y] + ctrl].dropna()
        if len(d) < 10:
            return None
        Rk = d.rank()
        X = np.column_stack([np.ones(len(Rk))] + [Rk[c].values for c in ctrl])
        rx = Rk[x].values - X @ lstsq(X, Rk[x].values, rcond=None)[0]
        ry = Rk[y].values - X @ lstsq(X, Rk[y].values, rcond=None)[0]
        return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else None
    return day_stat(df, f)


say("\n후행 RV 통제 후 부분상관 (GEX → 앞 RV):")
for name in BARS:
    p = partial(G, "total_gex_usd", f"k_frv_{name}", [f"k_trv_{name}"])
    R["pred"][name]["partial_frv"] = p[:2]
    say(f"  {name:6} {f3(p):>20}  (날 {p[2]})")
say("\n일별 부호 일치(1h, 앞 고저폭):")
by_day = {d: round(float(gg[["total_gex_usd", "k_frng_1h"]].dropna().corr(method='spearman').iloc[0, 1]), 2)
          for d, gg in G.groupby("day") if len(gg[["total_gex_usd", "k_frng_1h"]].dropna()) >= 8}
pos = sum(1 for v in by_day.values() if v > 0)
say(f"  양수 {pos}/{len(by_day)}일 · 중앙 {np.median(list(by_day.values())):+.2f}")
R["pred"]["by_day_1h"] = by_day
say("⏰**판정은 하지 않는다** — «변동성 예측 증분» 승격은 09-16 사전등록, 판정일 1h 09-28 / 4h 10-17.")

# ── C. 새 질문: GEX 레짐이 카드의 다른 칩을 바꾸는가 (1초 패널 겹침) ──────────
say("\n" + "=" * 96)
say("## C. 카드 통합 — GEX 레짐이 다른 칩이 말하는 것을 바꾸는가 (1초 패널 4.6일 겹침)")
P = pd.read_parquet(TMP / "panel_1s.parquet")
mid = P.bt_mid.ffill(limit=5)
P["ret1"] = np.log(mid).diff() * 1e4
for h in (15, 300):
    P[f"fwd{h}"] = (np.log(mid.shift(-h)) - np.log(mid)) * 1e4
hi = mid[::-1].rolling(300, min_periods=1).max()[::-1].shift(-1)
lo = mid[::-1].rolling(300, min_periods=1).min()[::-1].shift(-1)
P["frng300"] = (hi - lo) / mid * 1e4
P["tr_vol"] = P.tr_buy_qty + P.tr_sell_qty
P["tr_vol60"] = P.tr_vol.rolling(60, min_periods=30).sum()
P["dd_ofi"] = ((P.dd_add_b - P.dd_rem_b) - (P.dd_add_a - P.dd_rem_a)).where(P.dd_valid == 1)
P["dd_ofi10"] = P.dd_ofi.rolling(10, min_periods=5).sum()
P["ofi_thr"] = P.dd_ofi10.abs().rolling(600, min_periods=60).median()
qi_s = np.where(P.bt_qi_last >= 0.56, 1, np.where(P.bt_qi_last <= -0.56, -1, 0))
ofi_s = np.where(P.dd_ofi10 >= P.ofi_thr, 1, np.where(P.dd_ofi10 <= -P.ofi_thr, -1, 0))
P["trig"] = np.where((qi_s != 0) & (qi_s == ofi_s), qi_s, 0)
# GEX 를 초 격자로 (마지막 스냅샷을 최대 2시간까지 끌고 온다 -- 그게 화면이 보는 값이다)
gsec = G.set_index("sec")[["total_gex_usd", "front_ratio"]]
P["gex"] = gsec.total_gex_usd.reindex(P.index, method="ffill", tolerance=7200)
P["gex_fr"] = gsec.front_ratio.reindex(P.index, method="ffill", tolerance=7200)
P["day"] = pd.to_datetime(P.index, unit="s").date.astype(str)
cov = float(P.gex.notna().mean())
say(f"겹침 커버리지 {cov:.1%} · 겹치는 날 {P.loc[P.gex.notna(), 'day'].nunique()}일 (판정엔 턱없이 적다 -- 방향 힌트만)")
if cov > 0.3:
    ghi = P.gex >= P.gex.quantile(0.66)
    glo = P.gex <= P.gex.quantile(0.34)
    vq = P.tr_vol60.rank(pct=True)
    say("\n### C-1. 앞 5분 고저폭(bp): GEX 레짐 × 60초 거래량 분위")
    say(f"{'':16} {'거래량 하위1/3':>16} {'중간':>12} {'상위1/3':>12}")
    R["combo"] = {}
    for gname, gm in (("GEX 높음", ghi), ("GEX 낮음", glo)):
        row = []
        for vname, vm in (("lo", vq <= 1 / 3), ("mid", (vq > 1 / 3) & (vq < 2 / 3)), ("hi", vq >= 2 / 3)):
            m = (gm & vm).fillna(False)
            s = day_stat(P[m], lambda gg: gg.frng300.mean() if len(gg) >= 300 else None)
            row.append(f3(s) if np.isfinite(s[0]) else "n/a")
            R["combo"][f"{gname}|{vname}"] = s[:2]
        say(f"{gname:16} {row[0]:>16} {row[1]:>12} {row[2]:>12}")
    say("  ⇒ 거래량 분위를 고정했을 때 GEX 행끼리 갈라지면 GEX 가 «새 정보»다. 안 갈라지면 거래량의 사본이다.")
    say("\n### C-2. QI×OFI 방아쇠의 edge15 가 GEX 레짐에 따라 달라지나")
    for gname, gm in (("GEX 높음", ghi), ("GEX 낮음", glo), ("전체", P.gex.notna())):
        s = day_stat(P[gm.fillna(False)],
                     lambda gg: (gg.loc[gg.trig == 1, "fwd15"].mean() - gg.loc[gg.trig == -1, "fwd15"].mean())
                     if (gg.trig == 1).sum() >= 100 and (gg.trig == -1).sum() >= 100 else None)
        R.setdefault("trig_by_gex", {})[gname] = s[:2]
        say(f"  {gname:10} edge15 {f3(s):>16}  (날 {s[2]})")
    say("\n### C-3. GEX 가 이미 아는 것의 사본인가 -- 같은 시각 후행 변동성과의 상관")
    for name in ("1h", "4h"):
        c = ic(G, "total_gex_usd", f"k_trv_{name}")
        R.setdefault("gex_vs_trailing", {})[name] = c[:2]
        say(f"  ρ(GEX, 후행RV {name}) = {f3(c)}  (날 {c[2]})")

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else str(o)), encoding="utf-8")
say("\n->" + str(OUT))

# ── D. 🔴가격수준 인공물 검사 — GEX 공식에 spot² 이 들어 있다 ─────────────────
# GEX = Σ ±Γ·OI·spot²·0.01 이므로 **가격이 오르면 기계적으로 커진다**. 하루 안에서 GEX 가 높다는 건
# «그 날 안에서 가격이 높다»와 거의 같은 말일 수 있고, 이 창이 박스권이면 앞 수익이 음수로 나온다.
# 이 저장소가 같은 함정을 두 번 겪었다(스프레드 bp = 가격수준 · S/R 거리 = 레인지 위치).
say("\n" + "=" * 96)
say("## D. 가격수준 인공물 검사 (GEX 는 spot² 을 품고 있다)")
G["spot"] = G.spot_price
G["gex_ex_spot"] = G.total_gex_usd / (G.spot ** 2)     # 척도무관: 감마가중 OI 자체
say(f"{'측도':22} {'IC vs 앞 1h 수익':>20} {'IC vs 앞 4h 수익':>20} {'IC vs 앞 4h RV':>20}")
for label, col in (("GEX (원본)", "total_gex_usd"), ("스팟 가격만", "spot"), ("GEX/스팟² (척도무관)", "gex_ex_spot")):
    say(f"{label:22} {f3(ic(G, col, 'k_fret_1h')):>20} {f3(ic(G, col, 'k_fret_4h')):>20} {f3(ic(G, col, 'k_frv_4h')):>20}")
say("\n스팟 가격 통제 후 부분상관:")
R["spot_ctrl"] = {}
for y, lab in (("k_fret_1h", "앞 1h 수익"), ("k_fret_4h", "앞 4h 수익"), ("k_frv_4h", "앞 4h RV"), ("k_frng_4h", "앞 4h 고저폭")):
    raw, ps = ic(G, "total_gex_usd", y), partial(G, "total_gex_usd", y, ["spot"])
    R["spot_ctrl"][y] = {"raw": raw[:2], "ctrl_spot": ps[:2]}
    say(f"  GEX → {lab:12} 원본 {f3(raw):>18}   스팟 통제 후 {f3(ps):>18}")
say("\n⭐판정: 스팟을 통제해 사라지면 «GEX 가 방향을 안다»가 아니라 «GEX 는 가격의 사본»이다.")
