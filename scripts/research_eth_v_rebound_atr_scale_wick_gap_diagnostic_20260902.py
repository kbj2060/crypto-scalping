#!/usr/bin/env python3
"""왜 경제성이 안 나오는가 -- ATR 규모 / wick앵커 갭 / 비용지배 여부 직접 측정.

## 사용자 관찰 (2026-09-02)

"ATR이 너무 작은 거 아니야? 신호 뜨는 거 보니까 5분봉 1개 지나니까 익절하고 바로 나오고 있어."

대시보드 익절 판정(`live_..._20260829.py::_call_end_pos`)은 **그 봉의 저가**를 기준점으로 쓴다:
`target = low[pos] + 1.5 * pre_atr` (롱). 그런데 경제성 백테스트의 **진입가는 `open[pos+1]`**이다.
V자반등 봉은 정의상 저가에서 이미 튀어오른 봉이므로 **진입 시점에 목표의 상당 부분이 이미
지나가 있을 수 있다.** 9-10의 미착수 항목("라벨 앵커가 low[i] wick인데 실제 진입가는
open[i+1]이라는 갭은 별도 확인 필요")이 한 번도 측정된 적이 없다.

## 무엇을 재나 -- 두 가설을 가르는 것이 목적

  **가설 A (ATR 규모)**: 1.5*ATR 자체가 왕복 10bp에 비해 작다 -> 잡을 게 원래 없다.
  **가설 B (wick 갭)**: 목표는 충분히 큰데 **진입 시점에 이미 소진**돼 남는 몫이 작다.

  Q1 콜 시점 ATR(bp)과 1.5*ATR을 왕복비용 10bp와 직접 비교           -> A 검정
  Q2 **wick 갭**: (open[i+1]-low[i]) / (1.5*atr) = 진입 전 소진 비율   -> B 검정
  Q3 진입 후 **남은 목표**(bp)가 10bp를 넘는 콜의 비율                 -> A/B 공통 귀결
  Q4 대시보드 익절 도달까지 봉수 분포 (사용자가 본 "1봉")
  Q5 트레일링 실제 보유봉수 + 청산사유(스톱/만기)
  Q6 ⭐**무비용 경제성** -- 비용 0으로 같은 격자를 돌린다. 이게 두 설명을 가른다:
       gross가 VAL/OOS 양쪽 양수면 **비용지배**(신호는 맞는데 마진이 얇다)
       gross부터 OOS에서 뒤집히면 **방향 문제**(비용과 무관)
  Q7 ATR 3분위 층화 경제성 -- 고ATR 구간(1.5*ATR >> 비용)만 통과하는가

⚠️Q7의 근거는 "모델이 ATR을 못 봤다"가 아니다(Tier0에 이미 있다). **라벨이 ATR 정규화라
모델은 움직임이 거래 가능한 크기인지에 구조적으로 무관심하다** -- 예측 피쳐가 아니라
거래가능성 게이트로서의 ATR은 별개 축이다.

베이스는 배포판 그대로(동결 컨텍스트+TabPFN seed 20260829+thr 0.60). HOLDOUT 미터치.

Run on the server (GPU) via handoff.
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
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_s1 = _load("s1_diag", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
_bt = _s1._bt
SL_GRID, ARM_GRID, TRAIL_GRID = _bt.SL_GRID, _bt.ARM_GRID, _bt.TRAIL_GRID
FORWARD_BARS = _s1.FORWARD_BARS

CTX_CSV = ROOT / "data/labels/eth_5m_v_rebound_every_bar_20260901/tabpfn_train_context_frozen_every_bar_20260901.csv"
DEPLOYED_LABEL = {"atr_mult": 1.50, "t_sustain": 0.20, "full_bars": 12}
THR, LIVE_SEED = 0.60, 20260829
BADGE_ATR_MULT, BADGE_HORIZON = 1.5, 12
COST_BP, ARTIFACT_FREE_MIN = 10.0, 1.0
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
OUT_JSON = ROOT / "data/research/eth_v_rebound_atr_scale_diagnostic_20260902/report.json"


def log(m): print(f"[diag] {m}", flush=True)


def q(a, name=""):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a):
        return None
    return {"n": int(len(a)), "p25": round(float(np.percentile(a, 25)), 2),
            "median": round(float(np.median(a)), 2), "p75": round(float(np.percentile(a, 75)), 2),
            "mean": round(float(a.mean()), 2)}


def sim_with_exit(entry, atr, sign, H, L, C, sl, arm, trail, pessimistic):
    """simulate_trailing_vec와 **동일 로직**에 청산봉 인덱스만 추가로 반환한다."""
    n = len(entry)
    stop = entry - sign * sl * atr
    armed = np.zeros(n, bool); best = entry.copy()
    done = np.zeros(n, bool); out = np.zeros(n); exit_bar = np.full(n, H.shape[1] - 1)
    fav_all = np.where(sign[:, None] > 0, H, L)
    adv_all = np.where(sign[:, None] > 0, L, H)

    def upd(fav):
        nonlocal best, armed, stop
        live = ~done
        imp = live & (sign * (fav - best) > 0)
        best = np.where(imp, fav, best)
        armed_new = live & ~armed & (sign * (best - entry) >= arm * atr)
        armed[:] = armed | armed_new
        ns = best - sign * trail * atr
        u = live & armed & (sign * (ns - stop) > 0)
        stop[:] = np.where(u, ns, stop)

    def stp(adv, t):
        nonlocal done, out, exit_bar
        live = ~done
        hit = live & np.where(sign > 0, adv <= stop, adv >= stop)
        out[:] = np.where(hit, sign * (stop - entry) / entry, out)
        exit_bar[:] = np.where(hit, t, exit_bar)
        done[:] = done | hit

    for t in range(H.shape[1]):
        if done.all():
            break
        if pessimistic:
            stp(adv_all[:, t], t); upd(fav_all[:, t])
        else:
            upd(fav_all[:, t]); stp(adv_all[:, t], t)
    out = np.where(done, out, sign * (C[:, -1] - entry) / entry)
    return out, exit_bar, done


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    from sklearn.metrics import roc_auc_score
    import torch
    log(f"cuda: {torch.cuda.is_available()}")

    ctx = pd.read_csv(CTX_CSV)
    FEATURES = [c for c in ctx.columns if c not in ("timestamp", "label")]
    clf = TabPFNClassifier(device="cuda", random_state=LIVE_SEED, ignore_pretraining_limits=True)
    clf.fit(ctx[FEATURES], ctx["label"].to_numpy())

    _s1.VAL_END = OOS_END
    log("building every-bar frame ...")
    sig, feat, eth = _s1.build_sig()
    sb = _s1.label_param(sig, True, ambig="drop", anchor="wick", **DEPLOYED_LABEL)
    st = _s1.label_param(sig, False, ambig="drop", anchor="wick", **DEPLOYED_LABEL)
    long = _s1.long_frame_for(sig, feat, sb, st)
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                     np.where(long["timestamp"] < VAL_END, "VAL", "OOS"))
    assert long["timestamp"].max() < OOS_END, "HOLDOUT 누출"
    long = long.loc[long["split"] != "TRAIN"].dropna(subset=FEATURES).reset_index(drop=True)
    CH = 20000
    long["p"] = np.concatenate([clf.predict_proba(long[FEATURES].iloc[i:i + CH])[:, 1]
                                for i in range(0, len(long), CH)])
    for spn in ("VAL", "OOS"):
        s = long.loc[(long["split"] == spn) & long["label"].notna()]
        log(f"  자체검증 {spn} AUC {roc_auc_score(s['label'], s['p']):.4f}")

    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    pos_of = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    nk = len(kl)
    long["pos"] = [pos_of.get(np.datetime64(t.tz_localize(None)), -1) for t in long["timestamp"]]
    long = long.loc[long["pos"] >= 0].reset_index(drop=True)

    report = {"signal": "v_rebound_atr_scale_wick_gap_diagnostic", "asset": "ETHUSDT",
              "scope": {"base": "배포판(동결컨텍스트+TabPFN seed 20260829+thr 0.60)",
                        "cost_bp": COST_BP, "holdout_touched": False,
                        "live_code_changed": False}, "splits": {}}

    for spn in ("VAL", "OOS"):
        s = long.loc[(long["split"] == spn) & (long["p"] >= THR)].copy()
        s = s.loc[s["pos"] + FORWARD_BARS + 1 < nk].reset_index(drop=True)
        i = s["pos"].to_numpy().astype(int)
        dn = s["is_downside"].to_numpy() == 1
        sign = np.where(dn, 1.0, -1.0)
        atr = s["atr"].to_numpy(dtype=float)
        anchor = np.where(dn, l[i], h[i])          # 라벨/대시보드 기준점 = 그 봉의 wick
        entry = o[i + 1]                            # 백테스트 실제 진입가
        pre_atr = np.where(i >= 1, atr, np.nan)     # atr 컬럼은 이미 causal
        target = anchor + sign * BADGE_ATR_MULT * pre_atr

        log("")
        log(f"===== {spn}  호출 {len(s):,}건 =====")

        # Q1 ATR 규모
        atr_bp = atr / c[i] * 1e4
        tgt_bp = BADGE_ATR_MULT * atr_bp
        log(f"  Q1 ATR(bp)         {q(atr_bp)}")
        log(f"     1.5*ATR(bp)     {q(tgt_bp)}   <- 왕복비용 {COST_BP}bp와 비교")
        log(f"     1.5*ATR < 비용인 콜 비율: {float((tgt_bp < COST_BP).mean())*100:.1f}%")

        # Q2 wick 갭 -- 진입 전 이미 소진된 목표 비율
        consumed = sign * (entry - anchor)
        frac = consumed / (BADGE_ATR_MULT * pre_atr)
        log(f"  Q2 ⭐진입전 소진비율 (open[i+1]-wick)/(1.5*ATR)  {q(frac)}")
        log(f"     이미 100% 소진(진입시 목표 도달/초과): {float((frac >= 1).mean())*100:.1f}%")
        log(f"     50% 이상 소진:                          {float((frac >= 0.5).mean())*100:.1f}%")

        # Q3 진입 후 남은 목표
        remain_bp = (sign * (target - entry)) / entry * 1e4
        log(f"  Q3 진입후 남은 목표(bp) {q(remain_bp)}")
        log(f"     남은 목표 < 왕복비용({COST_BP}bp): {float((remain_bp < COST_BP).mean())*100:.1f}%")
        log(f"     남은 목표 <= 0:                    {float((remain_bp <= 0).mean())*100:.1f}%")

        # Q4 대시보드 익절 도달 봉수
        bars_tp = np.full(len(s), np.nan)
        for k in range(len(s)):
            if not np.isfinite(pre_atr[k]) or pre_atr[k] <= 0:
                continue
            end = min(i[k] + BADGE_HORIZON, nk - 1)
            seg = c[i[k] + 1:end + 1]
            hit = np.flatnonzero(seg >= target[k]) if dn[k] else np.flatnonzero(seg <= target[k])
            bars_tp[k] = (hit[0] + 1) if len(hit) else np.nan
        reached = np.isfinite(bars_tp)
        log(f"  Q4 대시보드 익절 도달률 {reached.mean()*100:.1f}%  "
            f"도달까지 봉수 {q(bars_tp)}")
        log(f"     **1봉만에 도달**: {float((bars_tp == 1).sum())/max(reached.sum(),1)*100:.1f}% "
            f"(도달 콜 중)")

        H = np.stack([h[j + 1:j + 1 + FORWARD_BARS] for j in i])
        L = np.stack([l[j + 1:j + 1 + FORWARD_BARS] for j in i])
        C = np.stack([c[j + 1:j + 1 + FORWARD_BARS] for j in i])

        # Q5/Q6/Q7: 격자 (비용 10bp vs 0bp)
        def grid(mask=None, cost=COST_BP):
            m = np.ones(len(s), bool) if mask is None else mask
            if m.sum() < 30:
                return None
            e_, a_, sg = entry[m], atr[m], sign[m]
            fwd = flip = 0; best = None
            for sl in SL_GRID:
                for arm in ARM_GRID:
                    if arm < ARTIFACT_FREE_MIN:
                        continue
                    for tr in TRAIL_GRID:
                        ov, _, _ = sim_with_exit(e_.copy(), a_, sg, H[m], L[m], C[m], sl, arm, tr, False)
                        pv, _, _ = sim_with_exit(e_.copy(), a_, sg, H[m], L[m], C[m], sl, arm, tr, True)
                        fo, _, _ = sim_with_exit(e_.copy(), a_, -sg, H[m], L[m], C[m], sl, arm, tr, False)
                        fp, _, _ = sim_with_exit(e_.copy(), a_, -sg, H[m], L[m], C[m], sl, arm, tr, True)
                        ob, pb = float(ov.mean()*1e4-cost), float(pv.mean()*1e4-cost)
                        fwd += int(ob > 0 and pb > 0)
                        flip += int(float(fo.mean()*1e4-cost) > 0 and float(fp.mean()*1e4-cost) > 0)
                        if best is None or pb > best["pess_bp"]:
                            best = {"sl": sl, "arm": arm, "trail": tr,
                                    "opt_bp": round(ob, 2), "pess_bp": round(pb, 2)}
            return {"n": int(m.sum()), "fwd_pass": fwd, "flip_pass": flip,
                    "margin": fwd - flip, "best": best}

        g_cost = grid()
        g_free = grid(cost=0.0)
        log(f"  Q6 ⭐비용 {COST_BP}bp: 정{g_cost['fwd_pass']}/뒤{g_cost['flip_pass']} "
            f"(차 {g_cost['margin']:+d})  최고pess {g_cost['best']['pess_bp']:+.2f}bp")
        log(f"     ⭐비용   0bp: 정{g_free['fwd_pass']}/뒤{g_free['flip_pass']} "
            f"(차 {g_free['margin']:+d})  최고pess {g_free['best']['pess_bp']:+.2f}bp")

        # Q5 보유봉수 (비용 10bp 최적셀 기준)
        b = g_cost["best"]
        _, ex, done = sim_with_exit(entry.copy(), atr, sign, H, L, C, b["sl"], b["arm"], b["trail"], True)
        log(f"  Q5 최적셀 SL/ARM/Tr={b['sl']}/{b['arm']}/{b['trail']}  "
            f"보유봉수 {q(ex + 1)}  스톱청산 {done.mean()*100:.1f}% / 만기 {(1-done.mean())*100:.1f}%")

        # Q7 ATR 3분위
        ter = pd.qcut(atr_bp, 3, labels=False, duplicates="drop")
        q7 = {}
        for t_ in sorted(pd.unique(ter[~pd.isna(ter)])):
            m = (ter == t_)
            gg = grid(m)
            if gg:
                q7[int(t_)] = {**gg, "atr_bp_median": round(float(np.median(atr_bp[m])), 2)}
                log(f"  Q7 ATR {int(t_)+1}/3분위 (중앙 {q7[int(t_)]['atr_bp_median']:.1f}bp) "
                    f"n={gg['n']:,}  정{gg['fwd_pass']}/뒤{gg['flip_pass']} (차 {gg['margin']:+d})  "
                    f"최고pess {gg['best']['pess_bp']:+.2f}bp")

        report["splits"][spn] = {
            "n_calls": int(len(s)),
            "Q1_atr_bp": q(atr_bp), "Q1_target_bp": q(tgt_bp),
            "Q1_target_below_cost_pct": round(float((tgt_bp < COST_BP).mean()) * 100, 1),
            "Q2_consumed_fraction": q(frac),
            "Q2_fully_consumed_pct": round(float((frac >= 1).mean()) * 100, 1),
            "Q2_half_consumed_pct": round(float((frac >= 0.5).mean()) * 100, 1),
            "Q3_remaining_bp": q(remain_bp),
            "Q3_remaining_below_cost_pct": round(float((remain_bp < COST_BP).mean()) * 100, 1),
            "Q3_remaining_nonpositive_pct": round(float((remain_bp <= 0).mean()) * 100, 1),
            "Q4_tp_reach_pct": round(float(reached.mean()) * 100, 1), "Q4_bars_to_tp": q(bars_tp),
            "Q4_one_bar_pct_of_reached": round(float((bars_tp == 1).sum()) / max(reached.sum(), 1) * 100, 1),
            "Q5_holding_bars": q(ex + 1), "Q5_stop_exit_pct": round(float(done.mean()) * 100, 1),
            "Q6_grid_cost10": g_cost, "Q6_grid_cost0": g_free, "Q7_atr_terciles": q7}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    report["runtime_sec"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT_JSON}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
