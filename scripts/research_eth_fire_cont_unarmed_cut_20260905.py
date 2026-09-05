#!/usr/bin/env python3
"""지속 규칙(R) **무장 실패 조기 청산** 검정 (2026-09-05).

사용자: *"무장 실패한 4건 지금 손절하는 게 나은지 확인해줘"* — 라이브 4건으로는 판정 불가라 전 구간에서 규칙으로 검정한다.

질문: 진입 후 N봉까지 이익이 1.5×ATR(무장 문턱)에 닿지 못한 포지션을 **그 봉 종가로 즉시 청산**하면
      지속 규칙의 기대값이 개선되는가? (현행은 5×ATR 손절 / 200봉 시간청산까지 방치)

모집단·라벨은 §5.23 원문 상속: 반전 8종 raw 첫발동(GAP12) → 다음 봉 시가 진입 → 신호 반대 방향 →
sim_exit(SL 5.0 / ARM 1.5 / Trail 0.1 ATR, 200봉, 비관 순서, 봉 고가/저가) − 10bp.

  변형   cut_n ∈ {24, 48, 72, 96, 120, 144}봉에서 미무장이면 종가 청산
  평가   행 평균 · 동시5 순차 포트폴리오 exp · 기준 대비 **행별 짝** 차이의 일군집 CI(같은 트레이드 쌍이라 짝비교가 정확)
  선택   TRAIN에서만 보고 고르고, VAL/OOS는 1회 조회. 대조군: 무장 여부와 무관하게 N봉에서 자르는 팔(=시간청산 단축)
         — 이게 없으면 "무장 조건"의 기여와 "짧게 자르기"의 기여를 못 가른다.
  진단   무장 실패 조건부 잔여 손익 분포(현재 라이브 4건이 있는 구간 N=115~165 포함)

HOLDOUT 미접촉. 연구/개발 점수 — 규칙 변경은 사전등록·섀도우를 거쳐야 한다.
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
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


FC = _load("fc_cut", "scripts/research_eth_evidence_fire_continuation_econ_20260904.py")
M = _load("dec_cut", "scripts/research_eth_live_decision_algorithm_v1_20260904.py")
OUT = ROOT / "data/research/eth_fire_cont_unarmed_cut_20260905"
CELL = (5.0, 1.5, 0.1)
FWD, COST, CAP, B = 200, 10.0, 5, 2000
CUTS = [24, 48, 72, 96, 120, 144]
WINDOWS = ("TRAIN", "VAL", "OOS")
rng = np.random.default_rng(20260905)


def log(m): print(f"[cut] {m}", flush=True)


def sim(entry, atr, sign, H, L, C, sl, arm, trail, cut_n=None, cut_needs_unarmed=True):
    """sim_exit 원문 + (선택) cut_n봉에서 조기 청산. cut_needs_unarmed=False면 무장 여부 무관(대조군).
    반환: (수익률, 청산봉오프셋, 무장봉오프셋 or -1)."""
    n = len(entry)
    stop = entry - sign * sl * atr
    armed = np.zeros(n, bool); arm_at = np.full(n, -1)
    best = entry.copy(); done = np.zeros(n, bool)
    out = np.zeros(n); ex = np.full(n, H.shape[1] - 1)
    fav = np.where(sign[:, None] > 0, H, L)
    adv = np.where(sign[:, None] > 0, L, H)
    for t in range(H.shape[1]):
        if done.all():
            break
        a_ = adv[:, t]; live = ~done
        hit = live & np.where(sign > 0, a_ <= stop, a_ >= stop)
        out = np.where(hit, sign * (stop - entry) / entry, out)
        ex = np.where(hit, t, ex); done = done | hit
        f_ = fav[:, t]; live = ~done
        imp = live & (sign * (f_ - best) > 0)
        best = np.where(imp, f_, best)
        newly = live & ~armed & (sign * (best - entry) >= arm * atr)
        arm_at = np.where(newly, t, arm_at); armed = armed | newly
        ns = best - sign * trail * atr
        u = live & armed & (sign * (ns - stop) > 0)
        stop = np.where(u, ns, stop)
        if cut_n is not None and t == cut_n:                    # 봉 처리 끝에서 종가 청산
            cut = ~done & (~armed if cut_needs_unarmed else np.ones(n, bool))
            out = np.where(cut, sign * (C[:, t] - entry) / entry, out)
            ex = np.where(cut, t, ex); done = done | cut
    out = np.where(done, out, sign * (C[:, -1] - entry) / entry)
    return out * 1e4 - COST, ex, arm_at


def pf(pnl, ts, pos, ex):
    cand = pd.DataFrame({"timestamp": ts, "entry_bar": pos + 1, "exit_bar": pos + 1 + ex, "pnl_bp": pnl})
    r = M.portfolio(cand, CAP)
    if r is None:
        return None
    ci, nd = M.day_ci(r["pnl"], r["ts"])
    return {"n": r["n"], "exp_bp": round(r["exp_bp"], 2), "day_ci95": ci, "win_rate": round(r["win_rate"], 3), "max_dd_bp": round(r["max_dd_bp"], 0)}


def paired_ci(diff, ts):
    """같은 트레이드 쌍의 차이 → 일군집 부트스트랩."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(ts))).normalize()
    d = pd.Series(np.asarray(diff, float), index=idx); g = d.groupby(level=0)
    s = g.sum().to_numpy(); c = g.count().to_numpy(); o = np.empty(B)
    for k in range(B):
        j = rng.integers(0, len(s), len(s)); o[k] = s[j].sum() / max(c[j].sum(), 1)
    return [round(float(np.percentile(o, 2.5)), 2), round(float(np.percentile(o, 97.5)), 2)]


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_parquet(FC.FRAME, columns=["pos", "is_downside", "timestamp", "split", "entry", "atr"])
    D["timestamp"] = pd.to_datetime(D["timestamp"]); D["is_downside"] = D["is_downside"].astype(int)
    bar = D.drop_duplicates("pos").sort_values("pos").reset_index(drop=True)
    kl = pd.read_csv(FC.KL, usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    k0 = int(np.searchsorted(kl["timestamp"].to_numpy(), np.datetime64(bar["timestamp"].iloc[0])))
    p_first = int(bar["pos"].iloc[0]); need = int(bar["pos"].iloc[-1]) - p_first + FWD + 2
    seg = kl.iloc[k0:k0 + need]
    o, h, l, c = (seg[x].to_numpy(float) for x in ("open", "high", "low", "close"))

    F = FC.load_fires(); Fp = F.loc[F["first_fire"]].drop_duplicates(["pos", "is_downside"])
    atr_of = bar.set_index("pos")["atr"]; Fp = Fp.loc[Fp["pos"].isin(atr_of.index)].reset_index(drop=True)
    pos = Fp["pos"].to_numpy(); sd = Fp["is_downside"].to_numpy().astype(int)
    atr = atr_of.reindex(pos).to_numpy(float)
    split = bar.set_index("pos")["split"].reindex(pos).to_numpy()
    ts = bar.set_index("pos")["timestamp"].reindex(pos).to_numpy()
    cont_sign = np.where(sd == 1, -1.0, 1.0)                    # 지속 = 페이드의 반대
    kp = pos - p_first
    H = np.stack([h[j + 1:j + 1 + FWD] for j in kp]); L = np.stack([l[j + 1:j + 1 + FWD] for j in kp]); C = np.stack([c[j + 1:j + 1 + FWD] for j in kp])
    entry = o[kp + 1]
    log(f"지속 트레이드 {len(pos):,} (TRAIN/VAL/OOS {[int((split==w).sum()) for w in WINDOWS]}) ({time.time()-t0:.0f}s)")

    base, ex0, arm_at = sim(entry, atr, cont_sign, H, L, C, *CELL)
    rep = {"cell": CELL, "cuts": CUTS, "holdout_touched": False, "n": int(len(pos)), "windows": {}}

    # 진단: 무장 성공률·무장 시점, 무장 실패 조건부 잔여
    unarmed = arm_at < 0
    log(f"무장 성공 {(~unarmed).mean():.3f} · 무장 시점 중앙 {np.median(arm_at[~unarmed]):.0f}봉 · 무장 실패 {unarmed.sum():,}건 평균 {base[unarmed].mean():+.2f}bp vs 무장 성공 {base[~unarmed].mean():+.2f}bp")
    rep["arm_diag"] = {"arm_rate": round(float((~unarmed).mean()), 4), "arm_bar_median": float(np.median(arm_at[~unarmed])),
                       "unarmed_final_bp": round(float(base[unarmed].mean()), 2), "armed_final_bp": round(float(base[~unarmed].mean()), 2)}

    # 진단: N봉 시점 미무장 조건부 — 그때 자르면 vs 끝까지 들면
    diag = []
    for N in CUTS + [165]:
        alive_unarmed = (arm_at < 0) | (arm_at > N)
        alive = (ex0 > N) & alive_unarmed                       # N봉에 아직 살아있고 미무장
        if alive.sum() < 30:
            continue
        cut_pnl = cont_sign[alive] * (C[alive, N] - entry[alive]) / entry[alive] * 1e4 - COST
        row = {"N": N, "n_alive_unarmed": int(alive.sum()), "share_of_all": round(float(alive.mean()), 4),
               "cut_now_bp": round(float(cut_pnl.mean()), 2), "hold_to_end_bp": round(float(base[alive].mean()), 2),
               "hold_minus_cut": round(float(base[alive].mean() - cut_pnl.mean()), 2),
               "hold_win_rate": round(float((base[alive] > 0).mean()), 3), "eventual_arm_rate": round(float((arm_at[alive] > N).mean()), 3)}
        diag.append(row)
    rep["conditional_diag"] = diag
    print("\n[진단] N봉에 아직 미무장인 포지션: 그때 자르면 vs 그대로 두면 (전 구간)")
    print(f"{'N':>5s} {'n':>6s} {'전체비':>6s} {'즉시청산':>9s} {'보유지속':>9s} {'차이':>7s} {'보유승률':>7s} {'이후무장률':>8s}")
    for r in diag:
        print(f"{r['N']:5d} {r['n_alive_unarmed']:6d} {r['share_of_all']:6.3f} {r['cut_now_bp']:9.2f} {r['hold_to_end_bp']:9.2f} {r['hold_minus_cut']:7.2f} {r['hold_win_rate']:7.3f} {r['eventual_arm_rate']:8.3f}")

    # 팔 비교
    arms = {"baseline": (None, True)}
    for N in CUTS:
        arms[f"cut_unarmed@{N}"] = (N, True)
        arms[f"CTRL_cut_all@{N}"] = (N, False)
    res = {}
    for name, (cn, need_un) in arms.items():
        if cn is None:
            pnl, ex = base, ex0
        else:
            pnl, ex, _ = sim(entry, atr, cont_sign, H, L, C, *CELL, cut_n=cn, cut_needs_unarmed=need_un)
        res[name] = (pnl, ex)

    print(f"\n[팔 비교] 행 평균 / 포트폴리오 exp [일CI] / 기준 대비 행별짝 차이 [일CI]")
    for w in WINDOWS:
        m = split == w
        print(f"\n--- {w} (n={int(m.sum())}) ---")
        rw = {}
        b_pnl = res["baseline"][0]
        for name, (pnl, ex) in res.items():
            s = pf(pnl[m], ts[m], pos[m], ex[m])
            d = pnl[m] - b_pnl[m]
            ci = paired_ci(d, ts[m]) if name != "baseline" else [0.0, 0.0]
            n_ch = int((d != 0).sum())
            rw[name] = {"row_bp": round(float(pnl[m].mean()), 2), "portfolio": s, "paired_diff_bp": round(float(d.mean()), 2),
                        "paired_ci95": ci, "n_changed": n_ch}
            print(f"  {name:20s} row {pnl[m].mean():+7.2f} | pf {s['exp_bp']:+7.2f} {str(s['day_ci95']):>16s} | 짝차이 {d.mean():+7.2f} {str(ci):>16s} (변경 {n_ch})")
        rep["windows"][w] = rw
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    log(f"완료 -> {OUT/'report.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
