"""전 지표 상관 완결 + 조합 전수 스캔.
(1) 전 원천 대표 피쳐(S/R·풋프린트 포함) 동시 스피어만 — 1초 격자와 1분 평균 격자
(2) 전 순서쌍 CCF 피크(±60초): 누가 누구를 몇 초 앞서나
(3) 12개 상태 원천의 모든 쌍(66) × 9칸 전수 스캔: 점유·앞 60/300초 수익(블록 SE)·활동 리프트·다음 분 청산 리프트.
    |fwd300|≥2.5SE & n≥3000초 & 전후반 부호 일치 인 칸만 «후보»로. 활동 리프트 ≥1.5 는 «활동 칸».
(4) 미시 3중 조합 QI×OFI×테이커.
출력 tmp/rt_probe_20260920/combos.json"""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = open(ROOT / "scripts/research_rt5_1s_panel_sr_20260920.py").read()
exec(src.split("# ── 상태 프레임")[0])   # P + 모든 피쳐 + S/R 거리 + ic/block_stat
OUT = ROOT / "tmp/rt_probe_20260920/combos.json"
J: dict = {}
hr = pd.to_datetime(P.index, unit="s").hour

# ── (1) 동시 상관 완결 ────────────────────────────────────────────────────────
REP = {"가격 Δ60s": "dmid60", "테이커 60s": "tr_imb60", "고래 60s": "wh_net60", "리테일 60s": "rt_net60", "흡수 60s": "fp_absorb60", "괴리 60s": "fp_div60",
       "ΔOI 60s": "oi_d60", "청산 순(현재분)": "lq_now_net", "청산 합(현재분)": "lq_now_tot", "QI 10s": "bt_qi10", "depth ±10bp": "dd_imb10", "depth ±50bp": "dd_imb50",
       "OFI 10s": "dd_ofi10", "S/R 위치": "sr_pos", "S/R 간격": "sr_gap_bp", "거래량 60s": "tr_vol60", "실현변동 60s": "rv60"}
names = list(REP)
M1 = pd.DataFrame(index=names, columns=names, dtype=float)
for i, a in enumerate(names):
    for b in names[i:]:
        m = 1.0 if a == b else ic(REP[a], REP[b])[0]
        M1.loc[a, b] = M1.loc[b, a] = m
J["corr_1s"] = M1.round(3).to_dict()
Pm = P[list(REP.values())].groupby(P.index // 60).mean()
M2 = Pm.corr(method="spearman"); M2.index = names; M2.columns = names
J["corr_1m"] = M2.round(3).to_dict()
print("## (1) 동시 스피어만 — 1초 격자 (블록 평균)"); print(M1.round(2).to_string())
print("\n## (1b) 동시 스피어만 — 1분 평균 격자"); print(M2.round(2).to_string())

# ── (2) CCF 피크 전수 ──────────────────────────────────────────────────────────
FAST = {"수익률 1s": "ret1", "테이커 1s": "tr_net", "고래 1s": "wh_net", "리테일 1s": "rt_net", "ΔOI 1s": "oi_d", "QI 1s": "bt_qi_mean", "depth ±5bp": "dd_imb5",
        "OFI 1s": "dd_ofi", "거래량 1s": "tr_vol", "churn 1s": "dd_churn", "S/R 위치": "sr_pos"}
fn = list(FAST); LAGS = range(-60, 61)
PK = pd.DataFrame(index=fn, columns=fn, dtype=object)
for a in fn:
    for b in fn:
        if a == b: PK.loc[a, b] = "—"; continue
        x = P[FAST[a]]; y = P[FAST[b]]
        cc = {l: x.corr(y.shift(-l)) for l in LAGS}
        pk = max(cc, key=lambda k: abs(cc[k]) if np.isfinite(cc[k]) else -1)
        PK.loc[a, b] = f"{pk:+d}s {cc[pk]:+.2f}" if abs(cc[pk]) >= 0.03 else "0"
J["ccf_peak"] = PK.to_dict()
print("\n## (2) CCF 피크: 행 x 가 열 y 를 «몇 초 앞서» 최대 상관 (lag>0 = x 선행, |r|<0.03 은 0)"); print(PK.to_string())

# ── (3) 상태 쌍 전수 스캔 ─────────────────────────────────────────────────────
def tri(s, lo, mid_, hi, by_hour=False):
    r = s.groupby(hr).rank(pct=True) if by_hour else s.rank(pct=True)
    o = pd.Series(np.where(r <= .2, lo, np.where(r >= .8, hi, mid_)), index=s.index, dtype=object); o[s.isna()] = None; return o
S = pd.DataFrame(index=P.index)
S["가격60"] = tri(P.dmid60, "하락", "횡보", "상승"); S["테이커60"] = tri(P.tr_imb60, "매도", "중립", "매수")
S["고래60"] = tri(P.wh_net60, "고래매도", "고래중립", "고래매수"); S["리테일60"] = tri(P.rt_net60, "리테일매도", "리테일중립", "리테일매수")
S["흡수60"] = tri(P.fp_absorb60, "효율", "보통", "흡수"); S["OI60"] = tri(P.oi_d60, "OI감소", "OI보합", "OI증가")
lp = P.lq_prev_net.fillna(np.nan)
S["청산(직전분)"] = pd.Series(np.select([P.lq_prev_tot.isna(), P.lq_prev_tot == 0, lp > 0, lp < 0, lp == 0], [None, "청산없음", "롱청산우세", "숏청산우세", "청산균형"], None), index=P.index, dtype=object)
S["QI10"] = tri(P.bt_qi10, "QI매도", "QI중립", "QI매수"); S["depth10"] = tri(P.dd_imb10, "매도벽", "벽중립", "매수벽"); S["depth50"] = tri(P.dd_imb50, "깊은매도벽", "깊은중립", "깊은매수벽")
S["OFI10"] = tri(P.dd_ofi10, "OFI매도", "OFI중립", "OFI매수")
S["S/R"] = pd.Series(np.select([P.sr_sup_bp.abs() <= 15, P.sr_res_bp.abs() <= 15], ["지지근접", "저항근접"], "레벨사이"), index=P.index, dtype=object); S.loc[P.sr_pos.isna(), "S/R"] = None
S["활동60"] = tri(P.tr_vol60, "조용", "보통", "활발", by_hour=True)
P["liq_next"] = P.lq_now_tot.shift(-60)
base = dict(fwd300=float(P.fwd300.mean()), frng300=float(P.frng300.mean()), liq_next=float(P.liq_next.mean()))
half = P.index < (P.index.min() + P.index.max()) // 2
cells = []
for a, b in itertools.combinations(S.columns, 2):
    for va in S[a].dropna().unique():
        for vb in S[b].dropna().unique():
            if "중립" in va or "보합" in va or "횡보" in va or "보통" in va or "사이" in va or "균형" in va: continue
            if "중립" in vb or "보합" in vb or "횡보" in vb or "보통" in vb or "사이" in vb or "균형" in vb: continue
            m = (S[a] == va) & (S[b] == vb)
            n = int(m.sum())
            if n < 3000: continue
            f60 = cond(m, "fwd60"); f300 = cond(m, "fwd300"); rng = cond(m, "frng300"); lq = cond(m, "liq_next")
            h1 = P.fwd300[m & half].mean(); h2 = P.fwd300[m & ~half].mean()
            cells.append(dict(a=a, va=va, b=b, vb=vb, n=n, share=round(n / len(P), 4), fwd60=round(f60[0], 2), fwd60_se=round(f60[1], 2), fwd300=round(f300[0], 2), fwd300_se=round(f300[1], 2),
                              z300=round(f300[0] / f300[1], 2) if f300[1] > 0 else 0, rng_lift=round(rng[0] / base["frng300"], 2), liq_lift=round(lq[0] / base["liq_next"], 2) if np.isfinite(lq[0]) else None,
                              half_sign_agree=bool(np.sign(h1) == np.sign(h2)), h1=round(float(h1), 2), h2=round(float(h2), 2)))
C = pd.DataFrame(cells)
J["cells"] = C.to_dict(orient="records")
cand = C[(C.z300.abs() >= 2.5) & C.half_sign_agree].sort_values("z300", key=abs, ascending=False)
act = C[C.rng_lift >= 1.5].sort_values("rng_lift", ascending=False)
J["candidates"] = cand.to_dict(orient="records"); J["activity_cells"] = act.head(25).to_dict(orient="records")
print(f"\n## (3) 상태 쌍 전수 스캔: 칸 {len(C)} (n≥3000초). 방향 후보(|z|≥2.5·전후반 부호 일치) {len(cand)} · 활동 칸(리프트≥1.5) {len(act)}")
print(cand[["a", "va", "b", "vb", "n", "fwd60", "fwd300", "fwd300_se", "z300", "h1", "h2", "rng_lift", "liq_lift"]].head(30).to_string(index=False))
print("\n  활동 상위:"); print(act[["a", "va", "b", "vb", "n", "rng_lift", "liq_lift", "fwd300", "fwd300_se"]].head(20).to_string(index=False))
# 다중비교 기준: 칸 수 대비 |z|≥2.5 기대 개수
print(f"  다중비교 참고: 칸 {len(C)}개에서 귀무 하 |z|≥2.5 기대 ≈ {len(C) * 0.0124:.1f}개, 관측 {int((C.z300.abs() >= 2.5).sum())}개 (전후반 일치 조건 전 {int((C.z300.abs() >= 2.5).sum())} → 후 {len(cand)})")

# ── (4) 미시 3중 ──────────────────────────────────────────────────────────────
print("\n## (4) QI10 × OFI10 × 테이커60 → fwd 5/15/60초 (bp)")
J["triple"] = []
for q in ("QI매수", "QI매도"):
    for o in ("OFI매수", "OFI매도"):
        for t in ("매수", "매도", "중립"):
            m = (S.QI10 == q) & (S.OFI10 == o) & (S["테이커60"] == t)
            if m.sum() < 1500: continue
            r = {h: cond(m, f"fwd{h}") for h in (5, 15, 60)}
            row = dict(qi=q, ofi=o, taker=t, n=int(m.sum()), **{f"fwd{h}": f"{r[h][0]:+.2f}±{r[h][1]:.2f}" for h in (5, 15, 60)})
            J["triple"].append(row); print("  ", row)
OUT.write_text(json.dumps(J, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print("->", OUT)
