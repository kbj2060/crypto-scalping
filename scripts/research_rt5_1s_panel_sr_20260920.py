"""청산맵 지지·저항(라이브 `compute_spliced_levels` 그대로, 24h 룩백)을 1초 패널에 붙여 두 프레임으로 본다.

레벨은 매 분 «그 분 시작 전에 닫힌 1h 봉 24개 + 그 순간 mid» 로 다시 계산한다(대시보드 60초 캐시와 같은 갱신).
플라시보 = 같은 분의 실제 «거리» 분포를 분 단위로 무작위 재배정한 가짜 레벨(기하는 같고 위치만 틀림).
상태 프레임: 레벨 근접 상태의 점유·지속·동시 리프트 · 실제 청산이 레벨 근처에 몰리는가.
예측 프레임: 거리 IC · 근접 후 앞 수익률 · 터치 후 돌파/반등률 · 다음 분 청산 — 전부 플라시보 대조.
출력: tmp/rt_probe_20260920/sr.json + 표 출력."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.live_liquidation_map_20260824 import compute_spliced_levels  # noqa: E402
exec(open(ROOT / "scripts/research_rt5_1s_panel_analyze_20260920.py").read().split("# ── 1. 원천별 단독")[0])  # noqa: 피쳐·block_stat·ic
OUT = ROOT / "tmp/rt_probe_20260920/sr.json"
LOOKBACK = 24
NEAR_BP = 15.0      # «근접» 판정
TOUCH_BP = 10.0     # 터치 = 거리가 10bp 안으로 처음 들어옴
BREAK_BP = 10.0     # 돌파 = 레벨 너머 10bp
rng = np.random.default_rng(20260920)

# ── 1h 봉 → 분마다 레벨 ────────────────────────────────────────────────────
kl = pd.read_parquet(ROOT / "tmp/rt_probe_20260920/klines_1h.parquet")
for c in ("open", "high", "low", "close", "volume"):
    kl[c] = kl[c].astype(float)
kl["timestamp"] = pd.to_datetime(kl.open_time, unit="ms", utc=True)
kl["close_sec"] = kl.open_time // 1000 + 3600
kl = kl.sort_values("open_time").reset_index(drop=True)
mid = P.bt_mid.ffill(limit=5)
minutes = np.arange(P.index.min() // 60 * 60, P.index.max() + 1, 60)
rows = []
for m in minutes:
    px = mid.get(m, np.nan)
    if not np.isfinite(px):
        continue
    closed = kl[kl.close_sec <= m].tail(LOOKBACK)
    if len(closed) < 20:
        continue
    lv = compute_spliced_levels(closed[["timestamp", "close", "high", "low", "volume"]].reset_index(drop=True), float(px))
    if not lv.get("warmed_up"):
        continue
    s = lv["support_levels"]; r = lv["resistance_levels"]
    rows.append(dict(ts_min=int(m), px=float(px),
                     sup1=s[0]["price"] if s else np.nan, sup1_w=s[0]["weight_pct"] if s else np.nan,
                     sup2=s[1]["price"] if len(s) > 1 else np.nan,
                     res1=r[0]["price"] if r else np.nan, res1_w=r[0]["weight_pct"] if r else np.nan,
                     res2=r[1]["price"] if len(r) > 1 else np.nan, n_sup=len(s), n_res=len(r)))
L = pd.DataFrame(rows).set_index("ts_min")
print(f"레벨 분 {len(L):,} · 지지 존재 {L.sup1.notna().mean():.3f} · 저항 존재 {L.res1.notna().mean():.3f} · "
      f"지지 거리 중앙 {((L.px - L.sup1) / L.px * 1e4).median():.1f}bp · 저항 {((L.res1 - L.px) / L.px * 1e4).median():.1f}bp")
# 플라시보: 거리(bp)를 분 단위로 섞어 같은 분의 mid 에 다시 붙인다
d_sup = ((L.px - L.sup1) / L.px).to_numpy(); d_res = ((L.res1 - L.px) / L.px).to_numpy()
perm = rng.permutation(len(L))
L["psup1"] = L.px * (1 - d_sup[perm]); L["pres1"] = L.px * (1 + d_res[perm])

# ── 초 단위 피쳐 ─────────────────────────────────────────────────────────────
P["ts_min"] = P.index // 60 * 60
for c in ("sup1", "sup1_w", "sup2", "res1", "res1_w", "res2", "n_sup", "n_res", "psup1", "pres1"):
    P[c] = P.ts_min.map(L[c])
P["sr_sup_bp"] = (mid - P.sup1) / mid * 1e4      # >0: 지지가 아래에 있음. <0: 지지를 뚫고 내려옴
P["sr_res_bp"] = (P.res1 - mid) / mid * 1e4      # >0: 저항이 위에 있음
P["sr_psup_bp"] = (mid - P.psup1) / mid * 1e4
P["sr_pres_bp"] = (P.pres1 - mid) / mid * 1e4
P["sr_pos"] = P.sr_sup_bp / (P.sr_sup_bp + P.sr_res_bp)   # 0=지지, 1=저항 (두 레벨 사이 위치)
P["sr_ppos"] = P.sr_psup_bp / (P.sr_psup_bp + P.sr_pres_bp)
P["sr_gap_bp"] = P.sr_sup_bp + P.sr_res_bp
P["sr_wdiff"] = P.sup1_w - P.res1_w
P["liq_next"] = P.lq_now_tot.shift(-60)
R: dict = {"meta": dict(minutes=int(len(L)), sup_exist=round(float(L.sup1.notna().mean()), 3), res_exist=round(float(L.res1.notna().mean()), 3),
                        sup_dist_med_bp=round(float(((L.px - L.sup1) / L.px * 1e4).median()), 1), res_dist_med_bp=round(float(((L.res1 - L.px) / L.px * 1e4).median()), 1),
                        near_bp=NEAR_BP, touch_bp=TOUCH_BP, break_bp=BREAK_BP)}


def cond(mask: pd.Series, col: str):
    mask = mask.fillna(False).astype(bool)
    return block_stat(lambda g, c=col, m=mask: g.loc[m.reindex(g.index).fillna(False), c].mean() if m.reindex(g.index).fillna(False).sum() >= 30 else None)


def fmt3(t): return f"{t[0]:+.3f}±{t[1]:.3f}" if np.isfinite(t[0]) else "n/a"


# ── 상태 프레임 ──────────────────────────────────────────────────────────────
near_s = P.sr_sup_bp.abs() <= NEAR_BP; near_r = P.sr_res_bp.abs() <= NEAR_BP
pnear_s = P.sr_psup_bp.abs() <= NEAR_BP; pnear_r = P.sr_pres_bp.abs() <= NEAR_BP
below_s = P.sr_sup_bp < -BREAK_BP; above_r = P.sr_res_bp < -BREAK_BP
state = pd.Series(np.select([below_s, above_r, near_s, near_r], ["지지 아래(뚫림)", "저항 위(뚫림)", "지지 근접", "저항 근접"], "레벨 사이"), index=P.index, dtype=object)
state[P.sr_sup_bp.isna() & P.sr_res_bp.isna()] = None


def episodes(s: pd.Series) -> dict:
    v = s.dropna(); chg = (v != v.shift()) | (v.index.to_series().diff() != 1); grp = chg.cumsum()
    ep = v.groupby(grp).agg(["first", "size"]); out = {}
    for st, g in ep.groupby("first"):
        out[str(st)] = dict(share=round(float((v == st).mean()), 4), episodes=int(len(g)), median_sec=float(g["size"].median()), p90_sec=float(g["size"].quantile(.9)))
    return out


R["state_occupancy"] = episodes(state)
print("\n## 레벨 대비 위치 상태 점유·지속"); print(json.dumps(R["state_occupancy"], ensure_ascii=False))

# 동시 리프트: 근접 상태에서 다른 원천 상태(분위 20/60/20)
def tri(s, lo, mid_, hi):
    r = s.rank(pct=True); o = pd.Series(np.where(r <= .2, lo, np.where(r >= .8, hi, mid_)), index=s.index, dtype=object); o[s.isna()] = None; return o
S = pd.DataFrame({"가격60": tri(P.dmid60, "하락", "횡보", "상승"), "테이커60": tri(P.tr_imb60, "매도우위", "중립", "매수우위"),
                  "호가±10bp": tri(P.dd_imb10, "매도벽", "중립", "매수벽"), "호가±50bp": tri(P.dd_imb50, "깊은매도벽", "중립", "깊은매수벽"),
                  "활동60": tri(P.tr_vol60, "조용", "보통", "활발"), "고래60": tri(P.wh_net60, "고래매도", "중립", "고래매수"), "OI60": tri(P.oi_d60, "감소", "보합", "증가")}, index=P.index)
lq = P.lq_now_tot.fillna(-1)
S["청산(현재분)"] = np.select([lq < 0, lq == 0, (P.liq_long > 0) & (P.liq_short == 0), (P.liq_short > 0) & (P.liq_long == 0), (P.liq_long > 0) & (P.liq_short > 0)], [None, "없음", "롱청산", "숏청산", "양쪽"], None)
unc = {c: S[c].value_counts(normalize=True).to_dict() for c in S.columns}
R["lifts"] = {}
for name, m in {"지지 근접(±15bp)": near_s, "저항 근접(±15bp)": near_r, "지지 아래(뚫림)": below_s, "저항 위(뚫림)": above_r,
                "플라시보 지지 근접": pnear_s, "플라시보 저항 근접": pnear_r}.items():
    m = m.fillna(False)
    R["lifts"][name] = {c: {str(k): dict(p=round(float(v), 3), lift=round(float(v / unc[c][k]), 2)) for k, v in S.loc[m, c].value_counts(normalize=True).items()} for c in S.columns}
    big = {c: {k: x["lift"] for k, x in d.items() if x["p"] >= .05 and abs(np.log2(x["lift"])) >= .3} for c, d in R["lifts"][name].items()}
    print(f"\n  {name} (n={int(m.sum()):,}, 점유 {m.mean():.3f}):", {c: v for c, v in big.items() if v})

# 실제 청산이 레벨 근처에 몰리는가 (분 단위)
Mn = P.groupby("ts_min").agg(liq_l=("liq_long", "last"), liq_s=("liq_short", "last"), sup_bp=("sr_sup_bp", "first"), res_bp=("sr_res_bp", "first"),
                              psup_bp=("sr_psup_bp", "first"), pres_bp=("sr_pres_bp", "first"), lo=("bt_mid_lo", "min"), hi=("bt_mid_hi", "max"), px=("bt_mid", "first"))
Mn["tot"] = Mn.liq_l + Mn.liq_s
burst = Mn.tot >= Mn.tot[Mn.tot > 0].quantile(.95)
R["liq_at_levels"] = {}
for lab, col in [("지지", "sup_bp"), ("저항", "res_bp"), ("플라시보 지지", "psup_bp"), ("플라시보 저항", "pres_bp")]:
    nr = Mn[col].abs() <= NEAR_BP
    side = Mn.liq_l if "지지" in lab else Mn.liq_s
    R["liq_at_levels"][lab] = dict(near_share=round(float(nr.mean()), 3), burst_near_share=round(float(nr[burst].mean()), 3),
                                  lift=round(float(nr[burst].mean() / nr.mean()), 2), side_liq_near_med=round(float(side[nr].median()), 0),
                                  side_liq_near_mean=round(float(side[nr].mean()), 0), side_liq_far_mean=round(float(side[~nr].mean()), 0),
                                  p_any_near=round(float((Mn.tot[nr] > 0).mean()), 3), p_any_far=round(float((Mn.tot[~nr] > 0).mean()), 3))
print("\n## 청산이 레벨 근처(±15bp)에 몰리는가 (분 단위)"); print(json.dumps(R["liq_at_levels"], ensure_ascii=False, indent=0))

# ── 예측 프레임 ──────────────────────────────────────────────────────────────
R["ic"] = {}
print("\n## 거리 피쳐 IC (앞 수익률·앞 5분 고저폭)")
for f in ["sr_sup_bp", "sr_res_bp", "sr_pos", "sr_ppos", "sr_gap_bp", "sr_wdiff", "sup1_w", "res1_w"]:
    R["ic"][f] = {h: fmt3(ic(f, f"fwd{h}")) for h in (15, 60, 300)} | {"frng300": fmt3(ic(f, "frng300"))}
    print(f"  {f:10}", R["ic"][f])

print("\n## 근접 상태 → 앞 수익률(bp)·앞 5분 고저폭·다음 분 청산$ (블록 평균±SE)")
R["cond"] = {}
for name, m in {"지지 근접": near_s, "저항 근접": near_r, "지지 아래(뚫림)": below_s, "저항 위(뚫림)": above_r,
                "플라시보 지지 근접": pnear_s, "플라시보 저항 근접": pnear_r, "레벨 사이": (state == "레벨 사이")}.items():
    R["cond"][name] = {c: fmt3(cond(m, c)) for c in ("fwd60", "fwd300", "frng300", "liq_next")} | {"n": int(m.fillna(False).sum())}
    print(f"  {name:14}", R["cond"][name])

# 터치 → 300초 안 돌파/반등 (첫 진입 초 기준, 60초 이상 떨어져 있던 뒤 처음 TOUCH_BP 안으로)
def touches(dist: pd.Series, level: pd.Series, side: int):
    """side=+1 지지(아래), −1 저항(위). 반환: 터치 초 인덱스, 300초 안 돌파(레벨 너머 BREAK_BP) 여부, 반등(터치가에서 반대쪽 BREAK_BP) 여부"""
    inside = (dist <= TOUCH_BP) & (dist >= -TOUCH_BP)
    was_far = (dist.shift(1).rolling(60, min_periods=60).min() > TOUCH_BP)
    t_idx = P.index[(inside & was_far).fillna(False)]
    out = []
    for t in t_idx:
        lv = level.get(t); m0 = mid.get(t)
        if not (np.isfinite(lv) and np.isfinite(m0)):
            continue
        seg = mid.loc[t + 1:t + 300]
        if len(seg) < 200:
            continue
        brk = (seg <= lv * (1 - BREAK_BP / 1e4)).any() if side > 0 else (seg >= lv * (1 + BREAK_BP / 1e4)).any()
        bnc = (seg >= m0 * (1 + BREAK_BP / 1e4)).any() if side > 0 else (seg <= m0 * (1 - BREAK_BP / 1e4)).any()
        out.append((t, bool(brk), bool(bnc), float((seg.iloc[-1] / m0 - 1) * 1e4)))
    return pd.DataFrame(out, columns=["t", "brk", "bnc", "ret300"])


print("\n## 터치(60초 이상 떨어져 있다가 ±10bp 안으로) 뒤 300초: 돌파율·반등률·평균 수익(bp)")
R["touch"] = {}
for lab, dist, lvl, side in [("지지", P.sr_sup_bp, P.sup1, 1), ("저항", P.sr_res_bp, P.res1, -1), ("플라시보 지지", P.sr_psup_bp, P.psup1, 1), ("플라시보 저항", P.sr_pres_bp, P.pres1, -1)]:
    T = touches(dist, lvl, side)
    if len(T) == 0:
        continue
    hr = T.t // 3600
    def bstat(col):
        v = T.groupby(hr)[col].mean(); return (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 2 else np.nan)
    R["touch"][lab] = dict(n=int(len(T)), brk=round(float(T.brk.mean()), 3), bnc=round(float(T.bnc.mean()), 3), ret300=fmt3(bstat("ret300")),
                           brk_se=round(bstat("brk")[1], 3), bnc_se=round(bstat("bnc")[1], 3))
    print(f"  {lab:10}", R["touch"][lab])
# 강도 분위별 돌파율 (실제 지지/저항)
for lab, dist, lvl, w, side in [("지지", P.sr_sup_bp, P.sup1, P.sup1_w, 1), ("저항", P.sr_res_bp, P.res1, P.res1_w, -1)]:
    T = touches(dist, lvl, side)
    if len(T) < 40:
        continue
    T["w"] = w.reindex(T.t).to_numpy()
    T["wq"] = pd.qcut(T.w.rank(method="first"), 3, labels=["약", "중", "강"])
    g = T.groupby("wq", observed=True).agg(n=("brk", "size"), brk=("brk", "mean"), bnc=("bnc", "mean"), ret=("ret300", "mean")).round(3)
    R["touch"][f"{lab}_by_weight"] = g.to_dict(orient="index"); print(f"  {lab} 강도별:\n{g.to_string()}")

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print("->", OUT)

# ── 대조 2: «레벨»인가 «거기까지 움직인 것»인가 ─────────────────────────────
print("\n## 대조 A. 레벨을 ±40bp 옮긴 플라시보 (같은 이동 맥락, 다른 위치) — 분 단위 청산")
R["shift_placebo"] = {}
for lab, lvl, sign in [("지지", "sup1", +1), ("저항", "res1", -1)]:
    for sh in (-40, 0, 40):
        lv = P[lvl] * (1 + sign * sh / 1e4)            # 지지: +40 = 실제 지지보다 40bp 위(먼저 닿음), −40 = 아래(뚫어야 닿음)
        d = (mid - lv) / mid * 1e4 * sign
        nr_min = d.abs().groupby(P.ts_min).min() <= NEAR_BP
        side = Mn.liq_l if sign > 0 else Mn.liq_s
        key = f"{lab} {sh:+d}bp"
        R["shift_placebo"][key] = dict(n_min=int(nr_min.sum()), p_any=round(float((Mn.tot[nr_min] > 0).mean()), 3), side_mean=round(float(side[nr_min].mean()), 0),
                                       burst_lift=round(float(nr_min[burst].mean() / nr_min.mean()), 2))
        print(f"  {key:10}", R["shift_placebo"][key])

print("\n## 대조 B. 활동(60초 거래량 시간대 분위)·직전 300초 이동을 맞춘 뒤 청산 확률 — 분 단위")
act_q = pd.qcut(P.tr_vol60.groupby(pd.to_datetime(P.index, unit='s').hour).rank(pct=True).groupby(P.ts_min).first(), 5, labels=False, duplicates="drop")
mv_q = pd.qcut(P.dmid300.abs().groupby(P.ts_min).first().rank(method="first"), 5, labels=False)
near_min_s = (P.sr_sup_bp.abs() <= NEAR_BP).groupby(P.ts_min).any(); near_min_r = (P.sr_res_bp.abs() <= NEAR_BP).groupby(P.ts_min).any()
pnear_min_s = (P.sr_psup_bp.abs() <= NEAR_BP).groupby(P.ts_min).any()
anyliq = (Mn.tot > 0)
R["matched"] = {}
for lab, nm in [("지지 근접", near_min_s), ("저항 근접", near_min_r), ("플라시보 지지 근접", pnear_min_s)]:
    df_ = pd.DataFrame({"near": nm, "liq": anyliq, "aq": act_q, "mq": mv_q}).dropna()
    # 셀별 (near 확률 − far 확률) 을 near 셀 가중으로 평균
    diffs = []; ws = []
    for (a, m), g in df_.groupby(["aq", "mq"]):
        if g.near.sum() >= 5 and (~g.near).sum() >= 5:
            diffs.append(g.liq[g.near].mean() - g.liq[~g.near].mean()); ws.append(g.near.sum())
    d = float(np.average(diffs, weights=ws)) if diffs else np.nan
    R["matched"][lab] = dict(raw_near=round(float(df_.liq[df_.near].mean()), 3), raw_far=round(float(df_.liq[~df_.near].mean()), 3), matched_diff=round(d, 3), cells=len(diffs))
    print(f"  {lab:12}", R["matched"][lab])

print("\n## 대조 C. 거리 IC 의 정체 — 24h 레인지 위치와 비교, 부분상관")
hi24 = mid.rolling(86400, min_periods=3600).max(); lo24 = mid.rolling(86400, min_periods=3600).min()
P["range_pos"] = (mid - lo24) / (hi24 - lo24)
from numpy.linalg import lstsq
def partial(x, y, ctrls):
    d = P[[x, y] + ctrls].dropna(); Rk = d.rank()
    X = np.column_stack([np.ones(len(Rk))] + [Rk[c].values for c in ctrls])
    rx = Rk[x].values - X @ lstsq(X, Rk[x].values, rcond=None)[0]; ry = Rk[y].values - X @ lstsq(X, Rk[y].values, rcond=None)[0]
    return round(float(np.corrcoef(rx, ry)[0, 1]), 3)
R["partial"] = {}
for x in ["sr_pos", "range_pos", "bt_mid"]:
    R["partial"][x] = dict(raw=fmt3(ic(x, "fwd300")), ctrl_dmid300=partial(x, "fwd300", ["dmid300"]),
                           ctrl_range=partial(x, "fwd300", ["dmid300", "range_pos"]) if x != "range_pos" else None,
                           ctrl_level=partial(x, "fwd300", ["dmid300", "bt_mid"]) if x != "bt_mid" else None,
                           ctrl_all=partial(x, "fwd300", ["dmid300", "range_pos", "bt_mid"]) if x == "sr_pos" else None)
    print(f"  {x:10}", R["partial"][x])
print("  sr_pos ↔ range_pos 스피어만", round(float(P[["sr_pos", "range_pos"]].dropna().corr(method="spearman").iloc[0, 1]), 3))
by_day = {}
dayi = pd.to_datetime(P.index, unit="s").date
for d_, g in P.groupby(dayi):
    dd = g[["sr_pos", "fwd300"]].dropna()
    if len(dd) > 3600: by_day[str(d_)] = round(float(dd.sr_pos.corr(dd.fwd300, method="spearman")), 3)
R["partial"]["sr_pos_by_day"] = by_day; print("  sr_pos→fwd300 일별", by_day)
OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
