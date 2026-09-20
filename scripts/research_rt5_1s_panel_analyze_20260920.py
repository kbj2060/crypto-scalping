"""1초 패널 분석: (1) 원천별 단독 «상태 읽기» (2) 원천 간 상관·선행지연·결합 상태.

통계: 관측이 초 단위로 강하게 자기상관이라 **1시간 블록**으로 나눠 블록별 통계의 평균±SE 로 낸다.
수익률은 bookTicker mid 기준 bp. 청산(1분)은 «직전 완결 분» 값만 선행 피쳐로 쓴다(같은 분은 미래참조).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = pd.read_parquet(ROOT / "tmp/rt_probe_20260920/panel_1s.parquet")
OUT = ROOT / "tmp/rt_probe_20260920/report.txt"
H = (1, 5, 15, 60, 300)
pd.set_option("display.width", 200, "display.max_columns", 40, "display.float_format", "{:.3f}".format)
lines: list[str] = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)


# ── 피쳐 ───────────────────────────────────────────────────────────────────
mid = P.bt_mid.ffill(limit=5)
P["ret1"] = np.log(mid).diff() * 1e4
for h in H:
    P[f"fwd{h}"] = (np.log(mid.shift(-h)) - np.log(mid)) * 1e4
    hi = mid[::-1].rolling(h, min_periods=1).max()[::-1].shift(-1)
    lo = mid[::-1].rolling(h, min_periods=1).min()[::-1].shift(-1)
    P[f"frng{h}"] = (hi - lo) / mid * 1e4          # 앞으로 h초 동안의 고저폭(bp) = 활동성
P["dmid60"] = (np.log(mid) - np.log(mid.shift(60))) * 1e4
P["dmid300"] = (np.log(mid) - np.log(mid.shift(300))) * 1e4
P["dmid10"] = (np.log(mid) - np.log(mid.shift(10))) * 1e4
P["rv60"] = P.ret1.rolling(60).std()

def rs(s, w): return s.rolling(w, min_periods=max(1, w // 2)).sum()

# 체결
P["tr_net"] = P.tr_buy_qty - P.tr_sell_qty
P["tr_vol"] = P.tr_buy_qty + P.tr_sell_qty
P["tr_nn"] = P.tr_buy_n + P.tr_sell_n
for w in (10, 60, 300):
    P[f"tr_net{w}"] = rs(P.tr_net, w); P[f"tr_vol{w}"] = rs(P.tr_vol, w)
    P[f"tr_imb{w}"] = P[f"tr_net{w}"] / P[f"tr_vol{w}"].replace(0, np.nan)
P["tr_big"] = np.where(P.tr_buy_max >= P.tr_sell_max, P.tr_buy_max, -P.tr_sell_max)
# 수급
P["wh_net"] = P.tr_whale_buy_qty - P.tr_whale_sell_qty
P["rt_net"] = P.tr_retail_buy_qty - P.tr_retail_sell_qty
P["wh_vol"] = P.tr_whale_buy_qty + P.tr_whale_sell_qty
for w in (60, 300):
    P[f"wh_net{w}"] = rs(P.wh_net, w); P[f"rt_net{w}"] = rs(P.rt_net, w)
    P[f"wh_share{w}"] = rs(P.wh_vol, w) / rs(P.tr_vol.where(P.wh_vol.notna()), w).replace(0, np.nan)
# 풋프린트(가격축 구조): 흡수 = 거래량 대비 가격이동, 델타 괴리 = 가격방향과 순매수 반대
P["fp_rng"] = (P.tr_px_max - P.tr_px_min) / mid * 1e4
P["fp_absorb60"] = P.tr_vol60 / (P.dmid60.abs() + 1.0)
P["fp_div60"] = -np.sign(P.dmid60) * P.tr_imb60          # >0: 가격은 올랐는데 순매도(괴리)
P["fp_eff60"] = P.dmid60 / (P.tr_net60.abs() + 1.0) * np.sign(P.tr_net60)  # 순매수 1ETH당 이동(bp)
# OI
P["oi_d"] = P.oi_d.fillna(0.0).where(P.oi_last.notna())
for w in (10, 60, 300):
    P[f"oi_d{w}"] = rs(P.oi_d, w)
P["oi_pct60"] = P.oi_d60 / P.oi_last * 1e4  # bp of OI
# 청산: 직전 완결 분(선행), 현재 분(동시)
liq_prev = P[["liq_long", "liq_short"]].shift(60)  # 분 m 의 값은 m+60초부터 사용 가능(+삽입지연 ≤15초)
P["lq_prev_long"] = liq_prev.liq_long; P["lq_prev_short"] = liq_prev.liq_short
P["lq_prev_net"] = P.lq_prev_long - P.lq_prev_short
P["lq_prev_tot"] = P.lq_prev_long + P.lq_prev_short
P["lq_now_net"] = P.liq_long - P.liq_short; P["lq_now_tot"] = P.liq_long + P.liq_short
# bookTicker
P["bt_qi10"] = P.bt_qi_mean.rolling(10, min_periods=5).mean()
P["bt_qi60"] = P.bt_qi_mean.rolling(60, min_periods=30).mean()
# depth
for b in (5, 10, 25, 50):
    P[f"dd_imb{b}"] = (P[f"dd_bid{b}"] - P[f"dd_ask{b}"]) / (P[f"dd_bid{b}"] + P[f"dd_ask{b}"])
P["dd_ofi"] = (P.dd_add_b - P.dd_rem_b) - (P.dd_add_a - P.dd_rem_a)
P["dd_ofi10"] = rs(P.dd_ofi, 10); P["dd_ofi60"] = rs(P.dd_ofi, 60)
P["dd_churn"] = P.dd_add_b + P.dd_rem_b + P.dd_add_a + P.dd_rem_a
P["dd_wall_asym"] = (P.dd_wall_b - P.dd_wall_a) / (P.dd_wall_b + P.dd_wall_a)
P["dd_depth10"] = P.dd_bid10 + P.dd_ask10
P.loc[P.dd_valid != 1, [c for c in P.columns if c.startswith("dd_")]] = np.nan
P["blk"] = P.index // 3600

STREAMS = {
    "체결(테이프)": ["tr_net", "tr_imb10", "tr_imb60", "tr_imb300", "tr_vol60", "tr_nn", "tr_big"],
    "수급(고래/리테일)": ["wh_net60", "wh_net300", "rt_net60", "rt_net300", "wh_share60"],
    "풋프린트(가격축)": ["fp_rng", "fp_absorb60", "fp_div60", "fp_eff60"],
    "OI": ["oi_d10", "oi_d60", "oi_d300", "oi_pct60"],
    "청산(1분·직전분)": ["lq_prev_net", "lq_prev_tot", "lq_prev_long", "lq_prev_short"],
    "호가 bookTicker": ["bt_qi_last", "bt_qi_mean", "bt_qi10", "bt_qi60", "bt_micro_bp", "bt_spread_bp", "bt_n"],
    "호가 depth": ["dd_imb5", "dd_imb10", "dd_imb25", "dd_imb50", "dd_ofi", "dd_ofi10", "dd_ofi60",
                  "dd_wall_asym", "dd_depth10", "dd_churn", "dd_n_lvl"],
}


def block_stat(fn) -> tuple[float, float, int]:
    vals = []
    for _, g in P.groupby("blk"):
        v = fn(g)
        if v is not None and np.isfinite(v):
            vals.append(v)
    vals = np.array(vals)
    return (vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals)), len(vals)) if len(vals) > 2 else (np.nan, np.nan, len(vals))


def ic(x: str, y: str, min_n=600) -> tuple[float, float, int]:
    def f(g):
        d = g[[x, y]].dropna()
        return d[x].corr(d[y], method="spearman") if len(d) >= min_n else None
    return block_stat(f)


def acf(x: str, lag: int) -> float:
    s = P[x]; return s.corr(s.shift(lag))


def fmt(m, se): return f"{m:+.3f}±{se:.3f}" if np.isfinite(m) else "  n/a  "


# ── 1. 원천별 단독 ──────────────────────────────────────────────────────────
say("=" * 100)
say(f"패널 {len(P):,}초  {pd.to_datetime(P.index.min(), unit='s')} ~ {pd.to_datetime(P.index.max(), unit='s')} UTC")
say("=" * 100)
for name, feats in STREAMS.items():
    say(f"\n## {name}")
    say(f"{'feature':14} {'cover':>5} {'p50':>9} {'p95':>9} {'p99':>9} | ACF 1s   10s   60s  300s | IC(spearman) vs fwd ret 1s/5s/15s/60s/300s | IC vs fwd range60 | top5%-bot5% fwd60 (bp)")
    for f in feats:
        s = P[f]; cov = s.notna().mean()
        q = s.abs().quantile([0.5, 0.95, 0.99]).values
        a = [acf(f, l) for l in (1, 10, 60, 300)]
        ics = [ic(f, f"fwd{h}") for h in H]
        icr = ic(f, "frng60")
        d = P[[f, "fwd60"]].dropna()
        lo, hi = d[f].quantile([0.05, 0.95])
        spread = d.fwd60[d[f] >= hi].mean() - d.fwd60[d[f] <= lo].mean()
        say(f"{f:14} {cov:5.2f} {q[0]:9.3g} {q[1]:9.3g} {q[2]:9.3g} | {a[0]:5.2f} {a[1]:5.2f} {a[2]:5.2f} {a[3]:5.2f} | "
            + " ".join(fmt(m, se) for m, se, _ in ics) + f" | {fmt(*icr[:2])} | {spread:+.2f}")

# 시간대 계절성(활동 지표) -- 「정상」의 기준선이 시간대마다 다르다
say("\n## 시간대(UTC) 계절성: 시간별 중앙값 / 전체 중앙값")
hr = pd.to_datetime(P.index, unit="s").hour
seas = P.groupby(hr)[["tr_vol60", "tr_nn", "bt_n", "dd_churn", "dd_depth10", "bt_spread_bp", "rv60", "lq_now_tot"]].median()
say((seas / seas.median()).round(2).to_string())

# ── 2. 원천 간 ──────────────────────────────────────────────────────────────
say("\n\n" + "=" * 100); say("## 2-A. 동시 상관(스피어만, 1시간 블록 평균) — 대표 피쳐")
REP = ["ret1", "dmid10", "dmid60", "tr_net", "tr_imb10", "tr_imb60", "wh_net60", "rt_net60", "oi_d10", "oi_d60",
       "lq_now_net", "lq_now_tot", "bt_qi_mean", "bt_qi10", "bt_micro_bp", "dd_imb5", "dd_imb10", "dd_imb50", "dd_ofi", "dd_ofi10",
       "dd_wall_asym", "fp_absorb60", "rv60", "dd_churn", "tr_vol60"]
M = pd.DataFrame(index=REP, columns=REP, dtype=float)
for i, a in enumerate(REP):
    for b in REP[i:]:
        m, _, _ = ic(a, b) if a != b else (1.0, 0, 0)
        M.loc[a, b] = M.loc[b, a] = m
say(M.round(2).to_string())

say("\n## 2-B. 선행·지연(CCF): corr(x_t, y_{t+lag}) — lag>0 이면 x 가 y 를 앞선다. 1초 격자, |lag|≤60")
PAIRS = [("tr_net", "ret1"), ("dd_ofi", "ret1"), ("bt_qi_mean", "ret1"), ("dd_imb5", "ret1"),
         ("tr_net", "oi_d"), ("ret1", "oi_d"), ("tr_net", "dd_ofi"), ("dd_imb5", "tr_net"), ("bt_qi_mean", "tr_net"),
         ("wh_net", "rt_net"), ("wh_net", "ret1"), ("rt_net", "ret1"), ("tr_vol", "dd_churn"), ("ret1", "dd_churn")]
LAGS = [-60, -30, -10, -5, -2, -1, 0, 1, 2, 5, 10, 30, 60]
say(f"{'x -> y':26} " + " ".join(f"{l:>6}" for l in LAGS) + "   | peak lag, corr")
for x, y in PAIRS:
    d = P[[x, y]]
    row = []
    for l in LAGS:
        c = d[x].corr(d[y].shift(-l))
        row.append(c)
    fine = {l: d[x].corr(d[y].shift(-l)) for l in range(-60, 61)}
    pk = max(fine, key=lambda k: abs(fine[k]) if np.isfinite(fine[k]) else -1)
    say(f"{x + ' -> ' + y:26} " + " ".join(f"{c:+6.3f}" for c in row) + f"   | {pk:+d}s {fine[pk]:+.3f}")

# 청산은 1분이라 분 격자에서 따로
say("\n## 2-C. 청산(1분)과 다른 원천(분 합계)의 선행·지연 — corr(x_m, liq_{m+lag})")
Mn = P.groupby(P.index // 60 * 60).agg(ret=("ret1", "sum"), tr_net=("tr_net", "sum"), oi_d=("oi_d", "sum"),
                                        liq_long=("liq_long", "last"), liq_short=("liq_short", "last"),
                                        wh_net=("wh_net", "sum"), rng=("fp_rng", "sum"), churn=("dd_churn", "sum"))
Mn["liq_net"] = Mn.liq_long - Mn.liq_short; Mn["liq_tot"] = Mn.liq_long + Mn.liq_short
Mn["absret"] = Mn.ret.abs()
for x, y in [("ret", "liq_net"), ("ret", "liq_tot"), ("absret", "liq_tot"), ("tr_net", "liq_net"), ("oi_d", "liq_tot"),
             ("liq_net", "ret"), ("liq_tot", "absret"), ("liq_long", "oi_d"), ("liq_tot", "oi_d"), ("wh_net", "liq_net")]:
    row = {l: Mn[x].corr(Mn[y].shift(-l), method="spearman") for l in (-5, -2, -1, 0, 1, 2, 5)}
    say(f"{x + ' -> ' + y:22} " + " ".join(f"{l:+d}m:{c:+.3f}" for l, c in row.items()))

# ── 2-D. 결합 상태표 ─────────────────────────────────────────────────────────
say("\n## 2-D. 결합 상태 → 그 뒤 60초/300초 수익(bp)·앞 300초 고저폭(bp)·다음 분 청산$ (1시간 블록 평균±SE, n초)")


def cond_table(mask_dict: dict[str, pd.Series], cols=("fwd60", "fwd300", "frng300", "liq_next")):
    P["liq_next"] = P.lq_now_tot.shift(-60)
    say(f"{'state':44} " + " ".join(f"{c:>18}" for c in cols) + "      n")
    for label, m in mask_dict.items():
        row = []
        for c in cols:
            mm, se, _ = block_stat(lambda g, c=c, m=m: g.loc[m.reindex(g.index).fillna(False), c].mean() if m.reindex(g.index).fillna(False).sum() >= 30 else None)
            row.append(fmt(mm, se))
        say(f"{label:44} " + " ".join(f"{r:>18}" for r in row) + f" {int(m.sum()):7d}")


def tert(s, lo=0.2, hi=0.8):
    a, b = s.quantile([lo, hi]); return s <= a, (s > a) & (s < b), s >= b


say("\n### (1) OI×가격 사분면 (60초 ΔOI 부호 × 60초 Δmid 부호, |Δmid60|>p50, |ΔOI60|>p50)")
big_p = P.dmid60.abs() > P.dmid60.abs().quantile(0.5); big_o = P.oi_d60.abs() > P.oi_d60.abs().quantile(0.5)
cond_table({
    "가격↑ OI↑ (신규 롱 유입)": big_p & big_o & (P.dmid60 > 0) & (P.oi_d60 > 0),
    "가격↑ OI↓ (숏 커버)": big_p & big_o & (P.dmid60 > 0) & (P.oi_d60 < 0),
    "가격↓ OI↑ (신규 숏 유입)": big_p & big_o & (P.dmid60 < 0) & (P.oi_d60 > 0),
    "가격↓ OI↓ (롱 청산/이탈)": big_p & big_o & (P.dmid60 < 0) & (P.oi_d60 < 0),
    "기준: 전체": P.oi_d60.notna(),
})

say("\n### (2) 테이커 순매수(60초) 5분위 × depth 10bp 불균형 5분위")
tl, tm, th = tert(P.tr_imb60); dl, dm, dh = tert(P.dd_imb10)
cond_table({
    "테이커 매수↑ & 호가 매수벽 두꺼움(동조)": th & dh,
    "테이커 매수↑ & 호가 매도벽 두꺼움(역행)": th & dl,
    "테이커 매도↑ & 호가 매도벽 두꺼움(동조)": tl & dl,
    "테이커 매도↑ & 호가 매수벽 두꺼움(역행)": tl & dh,
    "테이커 중립 & 호가 중립": tm & dm,
})

say("\n### (3) 테이커 순매수(60초) × ΔOI(60초)")
ol, om, oh = tert(P.oi_d60)
cond_table({
    "매수↑ OI↑ (공격적 신규 롱)": th & oh, "매수↑ OI↓ (숏 커버 매수)": th & ol,
    "매도↑ OI↑ (공격적 신규 숏)": tl & oh, "매도↑ OI↓ (롱 투매 청산)": tl & ol,
})

say("\n### (4) 직전 분 청산 규모 × 방향 (p95 이상 = 버스트)")
lt = P.lq_prev_tot; thr = lt[lt > 0].quantile(0.95)
cond_table({
    "롱 청산 버스트(직전분, ≥p95)": (P.lq_prev_long >= thr), "숏 청산 버스트(직전분, ≥p95)": (P.lq_prev_short >= thr),
    "롱 청산 버스트 & OI60 계속↓": (P.lq_prev_long >= thr) & ol, "롱 청산 버스트 & OI60 ↑(재진입)": (P.lq_prev_long >= thr) & oh,
    "숏 청산 버스트 & OI60 계속↓": (P.lq_prev_short >= thr) & ol, "숏 청산 버스트 & OI60 ↑(재진입)": (P.lq_prev_short >= thr) & oh,
    "직전분 청산 0": (P.lq_prev_tot == 0),
})

say("\n### (5) 고래 vs 리테일 (60초 순매수, 상하위 20%)")
wl, wm, wh = tert(P.wh_net60); rl, rm, rh = tert(P.rt_net60)
cond_table({
    "고래 매수 & 리테일 매수 (동조 상승)": wh & rh, "고래 매수 & 리테일 매도 (고래가 리테일 물량 흡수)": wh & rl,
    "고래 매도 & 리테일 매수 (리테일이 고래 물량 받음)": wl & rh, "고래 매도 & 리테일 매도 (동조 하락)": wl & rl,
})

say("\n### (6) 풋프린트 흡수/괴리 × 호가")
al, am, ah = tert(P.fp_absorb60); dv = P.fp_div60
cond_table({
    "흡수↑(거래량 많은데 안 움직임) & 매수벽": ah & dh, "흡수↑ & 매도벽": ah & dl,
    "가격↑인데 순매도(괴리, 상위20%) ": (dv >= dv.quantile(0.8)) & (P.dmid60 > 0),
    "가격↓인데 순매수(괴리, 상위20%)": (dv >= dv.quantile(0.8)) & (P.dmid60 < 0),
    "가격↑ & 순매수 (정합)": (dv <= dv.quantile(0.2)) & (P.dmid60 > 0),
    "가격↓ & 순매도 (정합)": (dv <= dv.quantile(0.2)) & (P.dmid60 < 0),
})

say("\n### (7) bookTicker 큐불균형 × depth OFI(10초) — 초단기(1~15초)")
ql, qm, qh = tert(P.bt_qi10); fl, fm, fh = tert(P.dd_ofi10)
P["liq_next"] = P.lq_now_tot.shift(-60)
say(f"{'state':44} {'fwd1':>18} {'fwd5':>18} {'fwd15':>18} {'fwd60':>18}      n")
for label, m in {"QI 매수 & OFI 매수": qh & fh, "QI 매수 & OFI 매도": qh & fl, "QI 매도 & OFI 매수": ql & fh, "QI 매도 & OFI 매도": ql & fl}.items():
    row = [fmt(*block_stat(lambda g, c=c, m=m: g.loc[m.reindex(g.index).fillna(False), c].mean() if m.reindex(g.index).fillna(False).sum() >= 30 else None)[:2]) for c in ("fwd1", "fwd5", "fwd15", "fwd60")]
    say(f"{label:44} " + " ".join(f"{r:>18}" for r in row) + f" {int(m.sum()):7d}")

say("\n### (8) 활동성 예측: 앞 300초 고저폭(bp)을 무엇이 미리 말하나 (IC)")
for f in ["rv60", "tr_vol60", "tr_nn", "bt_n", "dd_churn", "dd_depth10", "bt_spread_bp", "lq_prev_tot", "oi_d60", "fp_absorb60", "dd_n_lvl"]:
    m, se, n = ic(f, "frng300"); say(f"{f:14} IC vs frng300 {fmt(m, se)}  (blocks {n})")

OUT.write_text("\n".join(lines), encoding="utf-8")
print("->", OUT)
