#!/usr/bin/env python3
"""**초반 방향 정확도를 어디에 쓸 수 있나** — 세 가지 용도 검정 (2026-09-05).

사용자: *"초반 방향을 맞춘다면 그것만으로 큰 장점인데 이걸 못 살리는 게 아쉽다."*

부록 §15~16이 확정한 것: 페이드 승률 0.54~0.56(기저 0.50)은 **진짜**인데 gross 기대값이 0이고
(손익비 0.85), 청산 구조·분할 매수로는 못 고친다. ⇒ **"베팅"으로는 쓸 수 없다.**
그럼 0 gross 신호가 값을 갖는 다른 용도가 있는가. 세 축.

  E1 **승률 강화**  다중도(같은 측면 동시 발동 신호 수)·투표수로 단기 승률을 올릴 수 있나.
                   ⭐승률은 꼬리에 안 흔들리는 통계라 PnL보다 훨씬 안정적으로 측정된다
                   (C1이 PnL로 재서 실패한 것과 **다른 측정**이다). gross도 같이 본다.
  E2 **손익비 조건화** 문제는 승률이 아니라 손익비(0.85)다. 인과 상태축 27종 × 분위로
                   **손익비 > 1**인 부분집합이 있는지 찾는다. 승률이 아니라 손익비를 타깃으로 한 첫 시도.
  E3 ⭐**청산 타이밍** 이미 보유 중인 지속 포지션에 대해, 새 칩 발동을 **조기 익절 트리거**로 쓴다.
                   진입 비용을 새로 내지 않으므로 0 gross 신호도 값을 가질 수 있다.
                   트리거 = 보유 방향에 **불리한** 페이드 방향의 새 첫발동(지속 숏 보유 중 새 바닥 발동).
                   대조 3종: (a) 같은 방향 발동에서 청산(플라시보) (b) 보유 중 무작위 봉 청산(귀무,
                   같은 청산 빈도 매칭) (c) 무조건 유지(현행 R).
                   ⚠️학습 청산모델 Phase 0 킬게이트(FAIL)와 다른 것 — 모델이 아니라 규칙이다.

규격: 8종 raw 첫발동(GAP12) 합집합 · 진입 open[i+1] · 지속 방향 · sim_exit(5.0/1.5/0.1) 200봉 ·
비용 10bp · 동시 5 슬롯 · 조기 청산은 **트리거 봉 다음 봉 시가**(인과). HOLDOUT 미접촉.
판정: E3는 VAL·OOS 두 창 모두 R 대비 일별 짝비교 CI 하한 > 0.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


C1 = _load("c1_use", "scripts/research_eth_composite_direction_trend_pullback_20260905.py")
FC = _load("fc_use", "scripts/research_eth_evidence_fire_continuation_econ_20260904.py")
OUT = ROOT / "data/research/eth_use_short_horizon_direction_20260905"
COST, WINDOWS, HS = 10.0, ("TRAIN", "VAL", "OOS"), (1, 2, 3, 6)
B_NULL = 200
rng = np.random.default_rng(20260905)


def log(m): print(f"[use] {m}", flush=True)


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    B = C1.build()
    pos, sd, split, ts, bidx = B["pos"], B["sd"], B["split"], B["ts"], B["bidx"]
    cont_bp, cont_ex, cons = B["cont_bp"], B["cont_ex"], B["cons"]
    o, c, h_, l_ = B["o"], B["c"], B["h"], B["l"]; n = len(c)
    entry = B["entry"]; cs = B["cont_sign"]; fs = -cs                      # 페이드 부호
    p_first = B["p_first"]
    rep = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "cost_bp": COST, "holdout_touched": False}

    # ---------------- E1 승률 강화
    log("E1 다중도별 단기 승률 …")
    e1 = {}
    for hh in HS:
        ok = (bidx + hh) < n
        r = fs * (c[np.where(ok, bidx + hh, 0)] - entry) / entry * 1e4      # gross 페이드 수익
        d = {}
        for w in WINDOWS:
            m = ok & (split == w); row = {}
            for W in (0, 3):
                ms = cons[f"m_same_w{W}"]
                for v in (1, 2, 3):
                    sel = m & (ms == v) if v < 3 else m & (ms >= 3)
                    if sel.sum() < 100:
                        continue
                    x = r[sel]; wn = x > 0
                    row[f"W{W}_m{'>=3' if v == 3 else v}"] = {
                        "n": int(sel.sum()), "win_rate": round(float(wn.mean()), 3), "gross_mean_bp": round(float(x.mean()), 2),
                        "payoff": round(float(x[wn].mean() / -x[~wn].mean()), 3) if wn.any() and (~wn).any() else None}
            d[w] = row
        e1[f"h{hh}"] = d
    rep["E1_multiplicity_win_rate"] = e1
    for hh in (1, 3):
        for w in WINDOWS:
            log(f"  h={hh} {w:5s} " + " · ".join(f"{k} 승률 {v['win_rate']:.3f} gross {v['gross_mean_bp']:>6} 손익비 {v['payoff']}"
                                                 for k, v in e1[f"h{hh}"][w].items() if k.startswith("W3")))

    # ---------------- E2 손익비 조건화
    log("E2 손익비 > 1 조건 탐색 …")
    e2 = {}; tr = split == "TRAIN"
    for hh in (1, 3):
        ok = (bidx + hh) < n
        r = fs * (c[np.where(ok, bidx + hh, 0)] - entry) / entry * 1e4
        for name, (kind, raw) in B["S"].items():
            x = raw * fs if kind == "aligned" else raw.astype(float)        # 페이드 방향 정렬
            fin = np.isfinite(x) & ok
            if (fin & tr).sum() < 500:
                continue
            qs = np.quantile(x[fin & tr], [0.2, 0.4, 0.6, 0.8]); qi = np.where(fin, np.digitize(x, qs), -1)
            rows = {}
            for q in range(5):
                cell = {}
                for w in WINDOWS:
                    sel = fin & (split == w) & (qi == q)
                    if sel.sum() < 80:
                        continue
                    y = r[sel]; wn = y > 0
                    cell[w] = {"n": int(sel.sum()), "win_rate": round(float(wn.mean()), 3),
                               "payoff": round(float(y[wn].mean() / -y[~wn].mean()), 3) if wn.any() and (~wn).any() else None,
                               "gross_mean_bp": round(float(y.mean()), 2), "net_bp": round(float(y.mean() - COST), 2)}
                if cell:
                    rows[f"Q{q+1}"] = cell
            e2[f"h{hh}_{name}"] = rows
    # 세 창 모두 손익비 > 1 이고 gross > 0 인 (축, 분위) 찾기
    hits = []
    for k, rows in e2.items():
        for q, cell in rows.items():
            if len(cell) == 3 and all((cell[w]["payoff"] or 0) > 1.0 and cell[w]["gross_mean_bp"] > 0 for w in WINDOWS):
                hits.append({"axis_quintile": f"{k}:{q}", **{w: cell[w] for w in WINDOWS}})
    rep["E2_payoff_conditioning"] = {"cells": e2, "three_window_payoff_gt1_and_gross_gt0": hits, "n_hits": len(hits),
                                     "n_cells_tested": sum(len(v) for v in e2.values())}
    log(f"  검정 셀 {rep['E2_payoff_conditioning']['n_cells_tested']} · 세 창 모두 손익비>1 ∧ gross>0 인 셀 **{len(hits)}개**")
    for x in hits[:8]:
        log(f"    {x['axis_quintile']}: " + " | ".join(f"{w} n={x[w]['n']} 승률 {x[w]['win_rate']} 손익비 {x[w]['payoff']} gross {x[w]['gross_mean_bp']} net {x[w]['net_bp']}" for w in WINDOWS))

    # ---------------- E3 청산 타이밍
    log("E3 칩 발동을 지속 포지션 조기 익절 트리거로 …")
    F = FC.load_fires(); FF = F.loc[F["first_fire"]]
    nb = int(B["bar"]["pos"].iloc[-1]) - p_first + 1
    fire_side = np.zeros((2, nb), bool)                                     # [0]=top, [1]=bottom
    for sdv in (0, 1):
        q = FF.loc[FF["is_downside"] == sdv, "pos"].to_numpy()
        q = q[(q >= p_first) & (q < p_first + nb)]
        fire_side[sdv, q - p_first] = True
    # 지속 숏(cs<0)은 원 발동이 bottom → 불리한 새 발동 = bottom(칩이 "위" 예상)
    # 지속 롱(cs>0)은 원 발동이 top    → 불리한 새 발동 = top
    adverse_row = np.where(cs < 0, 1, 0)                                    # bottom=1, top=0
    same_row = 1 - adverse_row
    nat_exit = bidx + 1 + cont_ex                                           # 자연 청산 봉(seg 인덱스)
    N = len(bidx)

    def first_trigger(rows):
        """보유 구간 [bidx+1, nat_exit) 안 첫 트리거 봉. 없으면 -1."""
        out = np.full(N, -1)
        for k in range(N):
            a, b = bidx[k] + 1, min(nat_exit[k], nb - 2)
            if b <= a:
                continue
            seg = fire_side[rows[k], a:b]
            j = int(np.argmax(seg)) if seg.any() else -1
            out[k] = (a + j) if j >= 0 or seg.any() else -1
        return out

    trig_adv = first_trigger(adverse_row); trig_same = first_trigger(same_row)

    def pnl_with_exit(trig):
        """트리거 봉 **다음 봉 시가**에 청산(인과). 트리거 없으면 자연 청산."""
        has = (trig >= 0) & ((trig + 1) < nb)
        ex_bar = np.where(has, trig + 1, nat_exit)
        p = np.where(has, cs * (o[np.minimum(ex_bar, nb - 1)] - entry) / entry * 1e4 - COST, cont_bp)
        return p, ex_bar, has

    base = {w: C1.pf(C1.cand_of(ts[split == w], pos[split == w] + 1, pos[split == w] + 1 + cont_ex[split == w], cont_bp[split == w])) for w in WINDOWS}
    e3 = {"baseline_R": {w: base[w]["stats"] for w in WINDOWS}, "arms": {}}
    for nm, trig in (("exit_on_adverse_fire", trig_adv), ("placebo_exit_on_same_fire", trig_same)):
        p, ex_bar, has = pnl_with_exit(trig)
        rec = {"trigger_rate_all": round(float(has.mean()), 3)}
        for w in WINDOWS:
            m = split == w
            r = C1.pf(C1.cand_of(ts[m], pos[m] + 1, ex_bar[m] + p_first, p[m]))     # ex_bar는 seg 인덱스 → 프레임 pos로
            if r is None:
                continue
            rec[w] = {**{k: r["stats"][k] for k in ("n", "exp_bp", "win_rate", "day_ci95", "daily_mean_bp", "daily_sharpe_ann")},
                      "trigger_rate": round(float(has[m].mean()), 3),
                      "mean_hold_bars": round(float((np.minimum(ex_bar[m], nat_exit[m]) - bidx[m] - 1).mean()), 1),
                      "vs_R": C1.day_paired(r["pnl"], r["ts"], base[w]["pnl"], base[w]["ts"])}
        e3["arms"][nm] = rec
        log(f"  {nm}: 트리거율 {rec['trigger_rate_all']} · " + " | ".join(
            f"{w} exp={rec[w]['exp_bp']:>6} 보유 {rec[w]['mean_hold_bars']:>5}봉 ΔR={rec[w]['vs_R']['diff_bp_day']:>7}{str(rec[w]['vs_R']['ci95']):>18}" for w in WINDOWS if w in rec))
    # 귀무: 같은 청산 시점 분포를 무작위 봉으로
    hold_adv = np.where(trig_adv >= 0, trig_adv - bidx - 1, -1)
    pool = hold_adv[hold_adv >= 0]
    nulls = {w: [] for w in WINDOWS}
    for _ in range(B_NULL):
        draw = rng.choice(pool, N, replace=True)
        fire = rng.random(N) < (hold_adv >= 0).mean()
        t2 = np.where(fire, bidx + 1 + draw, -1)
        t2 = np.where(t2 >= nat_exit, -1, t2)
        p2, ex2, has2 = pnl_with_exit(t2)
        for w in WINDOWS:
            m = split == w
            r = C1.pf(C1.cand_of(ts[m], pos[m] + 1, ex2[m] + p_first, p2[m]))
            if r is not None:
                nulls[w].append(C1.day_paired(r["pnl"], r["ts"], base[w]["pnl"], base[w]["ts"], B=1)["diff_bp_day"])
    e3["null_random_exit"] = {w: {"mean": round(float(np.mean(v)), 2), "p95": round(float(np.percentile(v, 95)), 2),
                                  "obs": e3["arms"]["exit_on_adverse_fire"][w]["vs_R"]["diff_bp_day"],
                                  "percentile_of_obs": round(float((np.asarray(v) < e3["arms"]["exit_on_adverse_fire"][w]["vs_R"]["diff_bp_day"]).mean() * 100), 1)}
                              for w, v in nulls.items() if v}
    log(f"  무작위 청산 귀무: {json.dumps(e3['null_random_exit'], ensure_ascii=False)}")
    rep["E3_exit_timing"] = e3
    a = e3["arms"]["exit_on_adverse_fire"]
    rep["verdict_E3"] = {"rule": "VAL·OOS 두 창 모두 vs_R CI 하한 > 0",
                         "pass": bool(a.get("VAL", {}).get("vs_R", {}).get("ci95", [-9])[0] > 0 and a.get("OOS", {}).get("vs_R", {}).get("ci95", [-9])[0] > 0)}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"완료 {time.time()-t0:.0f}s → {OUT/'report.json'} · E3 통과 {rep['verdict_E3']['pass']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
