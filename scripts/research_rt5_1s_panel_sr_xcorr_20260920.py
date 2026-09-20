"""청산맵 S/R × 다른 원천 상관 분석 (sr 스크립트 위에 얹는다).
(1) 동시 스피어만: S/R 거리·강도 vs 호가·테이커·수급·OI·청산·활동 (1시간 블록 평균)
(2) 접근 프로파일: 지지/저항까지 거리 구간별로 다른 원천 중앙값 — 실제 vs 플라시보
(3) 터치 이벤트 스터디: 터치 −120s…+120s 동안 다른 원천 평균 경로 — 실제 vs 플라시보
(4) 상호작용(예측): 근접 상태 × 다른 원천 상태 → 돌파율·앞 300초 수익 (플라시보 대조)
출력 tmp/rt_probe_20260920/sr_xcorr.json"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = open(ROOT / "scripts/research_rt5_1s_panel_sr_20260920.py").read()
exec(src.split("# ── 상태 프레임")[0])          # 레벨·플라시보·거리 피쳐까지
exec(src[src.index("def touches("):src.index('print("\\n## 터치')])  # touches() 정의만
OUT2 = ROOT / "tmp/rt_probe_20260920/sr_xcorr.json"
X: dict = {}
OTHER = {"QI 10s": "bt_qi10", "depth ±10bp": "dd_imb10", "depth ±50bp": "dd_imb50", "OFI 10s": "dd_ofi10", "벽 비대칭": "dd_wall_asym",
         "테이커 60s": "tr_imb60", "테이커 300s": "tr_imb300", "고래 60s": "wh_net60", "리테일 60s": "rt_net60", "ΔOI 60s": "oi_d60", "ΔOI 300s": "oi_d300",
         "청산 순(현재분)": "lq_now_net", "청산 합(현재분)": "lq_now_tot", "거래량 60s": "tr_vol60", "실현변동 60s": "rv60", "depth churn": "dd_churn", "깊이 ±10bp 합": "dd_depth10"}
SRF = {"지지 거리 bp": "sr_sup_bp", "저항 거리 bp": "sr_res_bp", "사이 위치(0지지…1저항)": "sr_pos", "지지 강도": "sup1_w", "저항 강도": "res1_w", "강도차": "sr_wdiff", "레벨 간격": "sr_gap_bp",
       "플라시보 위치": "sr_ppos"}

# (1) 동시 상관
print("## (1) 동시 스피어만 (1시간 블록 평균, |값|≥0.05 만 굵게)")
X["corr"] = {}
hdr = f"{'':22}" + "".join(f"{k[:9]:>10}" for k in OTHER)
print(hdr)
for sn, sc in SRF.items():
    row = {}
    for on, oc in OTHER.items():
        m, se, n = ic(sc, oc); row[on] = (round(float(m), 3) if np.isfinite(m) else None, round(float(se), 3) if np.isfinite(se) else None)
    X["corr"][sn] = row
    print(f"{sn:22}" + "".join(f"{(v[0] if v[0] is not None else float('nan')):>+10.3f}" for v in row.values()))

# (2) 접근 프로파일 — 거리 구간별 중앙값 (실제 vs 플라시보), 부호는 «레벨 쪽» 기준으로 맞춘다
print("\n## (2) 접근 프로파일: 거리 구간별 다른 원천 중앙값 (실제 / 플라시보)")
BINS = [(-1e9, -10), (-10, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 1e9)]
LBL = ["뚫림(<−10)", "±10", "10~25", "25~50", "50~100", "100~200", ">200"]
PROF_COLS = {"테이커60 (레벨쪽+)": "tr_imb60", "호가±10bp (레벨쪽+)": "dd_imb10", "호가±50bp (레벨쪽+)": "dd_imb50", "고래60 (레벨쪽+)": "wh_net60", "ΔOI60": "oi_d60",
             "청산 해당측 $": None, "P(청산>0)": None, "거래량60": "tr_vol60", "QI10 (레벨쪽+)": "bt_qi10"}
X["approach"] = {}
for lab, dist, pdist, sign in [("지지", P.sr_sup_bp, P.sr_psup_bp, -1), ("저항", P.sr_res_bp, P.sr_pres_bp, +1)]:
    # 레벨쪽+ : 지지에 다가갈 때 «지지를 향한 압력» = 매도(−), 그래서 sign=−1 을 곱해 «레벨 쪽 압력»을 양수로
    side_liq = P.liq_long if lab == "지지" else P.liq_short
    X["approach"][lab] = {}
    print(f"\n  [{lab}] 거리구간:      " + "".join(f"{l:>14}" for l in LBL))
    for pn, pc in PROF_COLS.items():
        vals_r = []; vals_p = []
        for (lo, hi) in BINS:
            mr = (dist > lo) & (dist <= hi); mp = (pdist > lo) & (pdist <= hi)
            if pc is None and pn.startswith("청산"):
                vr, vp = side_liq[mr].mean(), side_liq[mp].mean()
            elif pc is None:
                vr, vp = (P.lq_now_tot[mr] > 0).mean(), (P.lq_now_tot[mp] > 0).mean()
            else:
                s_ = P[pc] * (sign if "(레벨쪽+)" in pn else 1)
                vr, vp = s_[mr].median(), s_[mp].median()
            vals_r.append(float(vr) if np.isfinite(vr) else None); vals_p.append(float(vp) if np.isfinite(vp) else None)
        X["approach"][lab][pn] = dict(real=vals_r, placebo=vals_p)
        fmtv = lambda v: (f"{v:>+7.3g}" if v is not None and abs(v) < 100 else (f"{v:>7,.0f}" if v is not None else "    n/a"))
        print(f"    {pn:18}" + "".join(f"{fmtv(r)}/{fmtv(p)}" for r, p in zip(vals_r, vals_p)))

# (3) 터치 이벤트 스터디 (−120…+120초), 실제 vs 플라시보. 부호는 레벨쪽+
print("\n## (3) 터치 이벤트 스터디: 터치 시각 기준 창별 평균 (실제 / 플라시보)")
WIN = [(-120, -61), (-60, -31), (-30, -11), (-10, -1), (0, 10), (11, 30), (31, 60), (61, 120), (121, 300)]
WL = ["−120…−61", "−60…−31", "−30…−11", "−10…−1", "0…10", "11…30", "31…60", "61…120", "121…300"]
EV_COLS = {"테이커 순매수(레벨쪽+) ETH/s": ("tr_net", True), "OFI(레벨쪽+) ETH/s": ("dd_ofi", True), "호가±10bp(레벨쪽+)": ("dd_imb10", True), "QI(레벨쪽+)": ("bt_qi_mean", True),
           "ΔOI ETH/s": ("oi_d", False), "청산 합 $/분(현재분)": ("lq_now_tot", False), "거래량 ETH/s": ("tr_vol", False), "고래 순매수(레벨쪽+)": ("wh_net", True)}
X["event"] = {}
for lab, dist, lvl, pdist, plvl, side, sign in [("지지", P.sr_sup_bp, P.sup1, P.sr_psup_bp, P.psup1, 1, -1), ("저항", P.sr_res_bp, P.res1, P.sr_pres_bp, P.pres1, -1, +1)]:
    Tr = touches(dist, lvl, side); Tp = touches(pdist, plvl, side)
    X["event"][lab] = {"n_real": int(len(Tr)), "n_placebo": int(len(Tp))}
    print(f"\n  [{lab}] 터치 n 실제 {len(Tr)} / 플라시보 {len(Tp)}      " + "".join(f"{w:>13}" for w in WL))
    for en, (col, signed) in EV_COLS.items():
        s_ = P[col] * (sign if signed else 1)
        def path(T):
            out = []
            for lo, hi in WIN:
                vals = [s_.loc[t + lo:t + hi].mean() for t in T.t]
                vals = [v for v in vals if np.isfinite(v)]
                out.append(float(np.mean(vals)) if vals else None)
            return out
        pr_, pp_ = path(Tr), path(Tp)
        X["event"][lab][en] = dict(real=pr_, placebo=pp_)
        f_ = lambda v: f"{v:>+6.3g}" if v is not None and abs(v) < 1000 else (f"{v:>6,.0f}" if v is not None else "   n/a")
        print(f"    {en:26}" + "".join(f"{f_(r)}/{f_(p)}" for r, p in zip(pr_, pp_)))

# (4) 상호작용: 터치 시점의 다른 원천 상태 → 돌파/반등/수익 (실제 레벨만, n 작음)
print("\n## (4) 터치 시점 다른 원천 상태 → 300초 돌파율·반등률·수익 (실제 레벨; n 작아 방향 힌트만)")
X["interact"] = {}
COND = {"테이커60 레벨쪽 압력": ("tr_imb60", True), "호가±10bp 레벨쪽 벽": ("dd_imb10", True), "호가±50bp 레벨쪽 벽": ("dd_imb50", True), "OFI10 레벨쪽": ("dd_ofi10", True),
        "ΔOI60": ("oi_d60", False), "직전분 청산 합": ("lq_prev_tot", False), "고래60 레벨쪽": ("wh_net60", True), "거래량60": ("tr_vol60", False)}
for lab, dist, lvl, side, sign in [("지지", P.sr_sup_bp, P.sup1, 1, -1), ("저항", P.sr_res_bp, P.res1, -1, +1)]:
    T = touches(dist, lvl, side); X["interact"][lab] = {}
    print(f"\n  [{lab}] n={len(T)}")
    for cn, (col, signed) in COND.items():
        v = (P[col] * (sign if signed else 1)).reindex(T.t).to_numpy()
        ok = np.isfinite(v)
        if ok.sum() < 30:
            continue
        med = np.nanmedian(v); hi = v > med; lo = v <= med
        r = dict(hi_brk=round(float(T.brk[hi].mean()), 3), lo_brk=round(float(T.brk[lo].mean()), 3), hi_bnc=round(float(T.bnc[hi].mean()), 3), lo_bnc=round(float(T.bnc[lo].mean()), 3),
                 hi_ret=round(float(T.ret300[hi].mean()), 2), lo_ret=round(float(T.ret300[lo].mean()), 2), n=int(ok.sum()))
        X["interact"][lab][cn] = r
        print(f"    {cn:16} 상위반: 돌파 {r['hi_brk']:.2f} 반등 {r['hi_bnc']:.2f} 수익 {r['hi_ret']:+.1f} | 하위반: 돌파 {r['lo_brk']:.2f} 반등 {r['lo_bnc']:.2f} 수익 {r['lo_ret']:+.1f}")

OUT2.write_text(json.dumps(X, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print("->", OUT2)
