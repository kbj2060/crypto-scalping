#!/usr/bin/env python3
"""E0 경제라벨 후보를 **실제 자동매매 시스템 형태**로 재평가 -- 5시드 앙상블 + 순차 포트폴리오.

## 왜 (승격 게이트 미통과의 원인 교정)

`..._econ_label_promotion_gate_20260902.py`가 5시드 중 4/5만 양수로 미통과(평균 +7.05bp,
**std 5.34**). 실패 패턴이 진단을 준다:

  상위1% 선정 시드 → OOS +15.08 / +10.45
  상위5% 선정 시드 → OOS +5.10 / +5.30 / **-0.69**

**신호가 아니라 선정 절차가 흔들린다.** VAL에서 분위를 고르는 자유도가 시드마다 다른 답을
내고 그게 OOS와 역상관이다. ⚠️"1%로 고정하자"는 OOS를 보고 고르는 것이므로 금지 --
대신 **원인(시드 분산)을 제거**한다.

## 세 가지 교정

  ① **5시드 확률 앙상블** -- 컨텍스트 재추출(348,474행 중 18,000)이 분산의 원인이므로
     시드 하나를 고르지 말고 5시드 확률을 평균낸다. 저장소 시드-다양성 정책과 정합.
  ② ⭐**순차 포트폴리오 시뮬레이션** -- 현행 평가는 하루 최대 39건을 호출하는데 보유가
     최대 200봉(16.7h)이라 수십 포지션이 겹친다. 자동매매로 불가능하고 통계적으로도 n이
     부풀려진다(트레이드가 독립이 아님). 봉을 순서대로 진행하며 **동시보유 한도**를 걸고
     슬롯이 빌 때만 진입하는 실제 시스템으로 다시 잰다.
  ③ **최대낙폭 / 최대 연속손실 / 자본곡선** -- 승률 78.9%에 손익비 0.425는 음의 왜도라
     사이징을 정하려면 필수. 기대값만으로는 운용 못 한다.

## 판정 (사전 등록)

  · OOS 순차 포트폴리오 **누적 수익 > 0** 이고 **뒤집기보다 우위**
  · OOS 3개월 중 **2개월 이상 양수**
  · 앙상블이므로 시드별 부호일치는 앙상블 자체의 안정성으로 대체 --
    대신 **VAL 선정 조합을 OOS에 1회만** 적용(재최적화 금지)
  · 비용은 **10.0bp(테이커)** 로 판정. 8.11bp(메이커진입 실측)는 참고 병기.

⚠️HOLDOUT 미터치. 라이브 코드 변경 없음.

Run on the server via handoff.
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


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_s1 = _load("s1_pf", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
_bt = _s1._bt
TIER0 = _s1.FEATURE_COLUMNS
FORWARD_BARS = _s1.FORWARD_BARS
SL_GRID, ARM_GRID, TRAIL_GRID = _bt.SL_GRID, _bt.ARM_GRID, _bt.TRAIL_GRID

LABEL_CELL = (5.0, 1.5, 0.1)
COST_TAKER, COST_MAKER = 10.0, 8.11
ARTIFACT_FREE_MIN = 1.0
CONTEXT_N = 18000
SEEDS = [20260829, 141592, 271828, 577215, 20260902]
TAIL_FRACS = [0.005, 0.01, 0.02, 0.05]
MAX_CONCURRENT = [1, 3, 5]
CHUNK = 40000
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
OUT = ROOT / "data/research/eth_v_rebound_ensemble_portfolio_20260902/report.json"


def log(m): print(f"[pf] {m}", flush=True)


def sim_exit(entry, atr, sign, H, L, C, sl, arm, trail):
    """비관 기준 트레일링. (수익률, 청산봉오프셋) 반환."""
    n = len(entry)
    stop = entry - sign * sl * atr
    armed = np.zeros(n, bool); best = entry.copy()
    done = np.zeros(n, bool); out = np.zeros(n); ex = np.full(n, H.shape[1] - 1)
    fav = np.where(sign[:, None] > 0, H, L)
    adv = np.where(sign[:, None] > 0, L, H)
    for t in range(H.shape[1]):
        if done.all():
            break
        a_ = adv[:, t]
        live = ~done
        hit = live & np.where(sign > 0, a_ <= stop, a_ >= stop)
        out = np.where(hit, sign * (stop - entry) / entry, out)
        ex = np.where(hit, t, ex); done = done | hit
        f_ = fav[:, t]
        live = ~done
        imp = live & (sign * (f_ - best) > 0)
        best = np.where(imp, f_, best)
        newly = live & ~armed & (sign * (best - entry) >= arm * atr)
        armed = armed | newly
        ns = best - sign * trail * atr
        u = live & armed & (sign * (ns - stop) > 0)
        stop = np.where(u, ns, stop)
    out = np.where(done, out, sign * (C[:, -1] - entry) / entry)
    return out, ex


def portfolio(cand, max_conc):
    """진입봉 순서대로 슬롯 제약 하에 체결. cand: entry_bar/exit_bar/pnl_bp 컬럼."""
    cand = cand.sort_values("entry_bar")
    eb = cand["entry_bar"].to_numpy(); xb = cand["exit_bar"].to_numpy()
    pn = cand["pnl_bp"].to_numpy(); ts = cand["timestamp"].to_numpy()
    open_until, taken = [], []
    for k in range(len(cand)):
        open_until = [u for u in open_until if u > eb[k]]
        if len(open_until) < max_conc:
            open_until.append(xb[k]); taken.append(k)
    if not taken:
        return None
    p = pn[np.array(taken)]
    eq = np.cumsum(p)
    dd = eq - np.maximum.accumulate(eq)
    losses = (p <= 0).astype(int)
    mcl = cur = 0
    for x in losses:
        cur = cur + 1 if x else 0
        mcl = max(mcl, cur)
    w = p > 0
    return {"n": int(len(p)), "exp_bp": float(p.mean()), "total_bp": float(p.sum()),
            "win_rate": float(w.mean()),
            "payoff": float(p[w].mean() / -p[~w].mean()) if w.any() and (~w).any() else None,
            "max_dd_bp": float(dd.min()), "max_consec_loss": int(mcl),
            "idx": np.array(taken), "pnl": p, "ts": ts[np.array(taken)]}


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    import torch
    log(f"cuda: {torch.cuda.is_available()}")

    _s1.VAL_END = OOS_END
    log("building frame ...")
    sig, feat, eth = _s1.build_sig()
    sb = _s1.label_param(sig, True, ambig="drop", anchor="wick",
                         atr_mult=1.50, t_sustain=0.20, full_bars=12)
    st = _s1.label_param(sig, False, ambig="drop", anchor="wick",
                         atr_mult=1.50, t_sustain=0.20, full_bars=12)
    long = _s1.long_frame_for(sig, feat, sb, st)
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                     np.where(long["timestamp"] < VAL_END, "VAL", "OOS"))
    assert long["timestamp"].max() < OOS_END, "HOLDOUT 누출"

    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    pos_of = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    nk = len(kl)
    long["pos"] = [pos_of.get(np.datetime64(t.tz_localize(None)), -1) for t in long["timestamp"]]
    long = long.loc[(long["pos"] >= 0) & (long["pos"] + FORWARD_BARS + 1 < nk)].reset_index(drop=True)

    sl0, arm0, tr0 = LABEL_CELL
    i_all = long["pos"].to_numpy().astype(int)
    sgn_all = np.where(long["is_downside"].to_numpy() == 1, 1.0, -1.0)
    atr_all = long["atr"].to_numpy(dtype=float)
    net = np.full(len(long), np.nan)
    for s_ in range(0, len(long), CHUNK):
        e_ = min(s_ + CHUNK, len(long))
        idx = i_all[s_:e_]
        H = np.stack([h[j+1:j+1+FORWARD_BARS] for j in idx])
        L = np.stack([l[j+1:j+1+FORWARD_BARS] for j in idx])
        C = np.stack([c[j+1:j+1+FORWARD_BARS] for j in idx])
        pn, _ = sim_exit(o[idx+1], atr_all[s_:e_], sgn_all[s_:e_], H, L, C, sl0, arm0, tr0)
        net[s_:e_] = pn * 1e4 - COST_TAKER
    long["y"] = (net > 0).astype(float)
    log(f"E0 라벨률 {long['y'].mean():.4f}")

    # ---- ① 5시드 앙상블 확률 ----
    tr_set = long.loc[long["split"] == "TRAIN"]
    probs = {"VAL": [], "OOS": []}
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        ctx = tr_set.iloc[np.sort(rng.choice(len(tr_set), size=min(CONTEXT_N, len(tr_set)), replace=False))]
        clf = TabPFNClassifier(device="cuda", random_state=sd, ignore_pretraining_limits=True)
        clf.fit(ctx[TIER0], ctx["y"].to_numpy())
        for spn in ("VAL", "OOS"):
            s = long.loc[long["split"] == spn]
            probs[spn].append(np.concatenate([clf.predict_proba(s[TIER0].iloc[k:k+20000])[:, 1]
                                              for k in range(0, len(s), 20000)]))
        log(f"  seed {sd} 완료")
    scored = {}
    for spn in ("VAL", "OOS"):
        s = long.loc[long["split"] == spn].copy()
        P = np.vstack(probs[spn])
        s["p"] = P.mean(axis=0)
        s["p_std"] = P.std(axis=0)
        scored[spn] = s
        log(f"  {spn} 앙상블 확률: 시드간 std 중앙 {np.median(P.std(axis=0)):.4f}")

    def candidates(s, cut, cell, cost, flip=False):
        sel = s.loc[s["p"] >= cut]
        if len(sel) < 30:
            return None
        idx = sel["pos"].to_numpy().astype(int)
        sgn = np.where(sel["is_downside"].to_numpy() == 1, 1.0, -1.0)
        if flip:
            sgn = -sgn
        H = np.stack([h[j+1:j+1+FORWARD_BARS] for j in idx])
        L = np.stack([l[j+1:j+1+FORWARD_BARS] for j in idx])
        C = np.stack([c[j+1:j+1+FORWARD_BARS] for j in idx])
        sl, arm, trv = cell
        pn, ex = sim_exit(o[idx+1], sel["atr"].to_numpy(dtype=float), sgn, H, L, C, sl, arm, trv)
        return pd.DataFrame({"timestamp": sel["timestamp"].to_numpy(), "entry_bar": idx + 1,
                             "exit_bar": idx + 1 + ex, "pnl_bp": pn * 1e4 - cost})

    report = {"signal": "v_rebound_ensemble_portfolio_sim", "asset": "ETHUSDT",
              "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
              "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
              "scope": {"fix": "5시드 확률 앙상블 + 순차 포트폴리오(동시보유 한도)",
                        "seeds": SEEDS, "max_concurrent": MAX_CONCURRENT,
                        "cost_taker_bp": COST_TAKER, "cost_maker_bp": COST_MAKER,
                        "selection": "VAL에서 (분위, 셀, 한도) 선정 -> OOS 1회",
                        "holdout_touched": False, "live_code_changed": False},
              "val_grid": [], "oos": {}}

    log("")
    log("=== VAL 순차 포트폴리오 격자 (비용 10.0bp) ===")
    log(f"  {'분위':>6s} {'한도':>4s} {'셀':>14s} {'n':>5s} {'기대값':>9s} {'총bp':>9s} "
        f"{'승률':>6s} {'최대DD':>9s} {'연속손실':>6s}")
    best = None
    for frac in TAIL_FRACS:
        k = max(30, int(round(len(scored["VAL"]) * frac)))
        cut = float(scored["VAL"].nlargest(k, "p")["p"].min())
        for sl in SL_GRID:
            for arm in ARM_GRID:
                if arm < ARTIFACT_FREE_MIN:
                    continue
                for trv in TRAIL_GRID:
                    cand = candidates(scored["VAL"], cut, (sl, arm, trv), COST_TAKER)
                    if cand is None:
                        continue
                    for mc in MAX_CONCURRENT:
                        r = portfolio(cand, mc)
                        if r is None or r["n"] < 30:
                            continue
                        if best is None or r["total_bp"] > best["r"]["total_bp"]:
                            best = {"frac": frac, "cut": cut, "cell": (sl, arm, trv),
                                    "mc": mc, "r": r}
        # 분위별 대표(최적) 로그
        if best:
            b = best
            log(f"  {frac*100:>5.1f}% {b['mc']:>4d} {str(b['cell']):>14s} {b['r']['n']:>5,} "
                f"{b['r']['exp_bp']:>+8.2f}bp {b['r']['total_bp']:>+8.0f}bp "
                f"{b['r']['win_rate']*100:>5.1f}% {b['r']['max_dd_bp']:>+8.0f}bp "
                f"{b['r']['max_consec_loss']:>6d}")

    b = best
    log("")
    log(f"[VAL 선정] 상위{b['frac']*100:g}% (p>={b['cut']:.4f})  셀 {b['cell']}  동시보유 {b['mc']}  "
        f"n={b['r']['n']:,}  기대값 {b['r']['exp_bp']:+.2f}bp  총 {b['r']['total_bp']:+.0f}bp  "
        f"최대DD {b['r']['max_dd_bp']:+.0f}bp  연속손실 {b['r']['max_consec_loss']}")
    report["val_selection"] = {"frac": b["frac"], "cut": round(b["cut"], 4), "cell": list(b["cell"]),
                               "max_concurrent": b["mc"],
                               **{k2: (round(v, 3) if isinstance(v, float) else v)
                                  for k2, v in b["r"].items() if k2 not in ("idx", "pnl", "ts")}}

    log("")
    log("=== OOS 1회 (재최적화 없음) ===")
    for cost, cnm in ((COST_TAKER, "테이커10.0bp(판정)"), (COST_MAKER, "메이커진입8.11bp(참고)")):
        for flip, fnm in ((False, "정방향"), (True, "뒤집기")):
            cand = candidates(scored["OOS"], b["cut"], b["cell"], cost, flip=flip)
            if cand is None:
                log(f"  [{cnm}/{fnm}] 후보 부족"); continue
            r = portfolio(cand, b["mc"])
            if r is None:
                continue
            days = (pd.Timestamp(r["ts"].max()) - pd.Timestamp(r["ts"].min())).total_seconds()/86400
            mo = pd.Series(r["pnl"], index=pd.to_datetime(r["ts"])).groupby(
                pd.to_datetime(r["ts"]).to_period("M")).mean()
            log(f"  [{cnm}] {fnm}  n={r['n']:,} ({r['n']/max(days,1):.2f}건/일)  "
                f"기대값 {r['exp_bp']:+.2f}bp  총 {r['total_bp']:+.0f}bp  승률 {r['win_rate']*100:.1f}%  "
                f"손익비 {r['payoff']}  최대DD {r['max_dd_bp']:+.0f}bp  연속손실 {r['max_consec_loss']}")
            log(f"       월별: " + "  ".join(f"{k} {v:+.2f}bp" for k, v in mo.items()))
            report["oos"][f"{cnm}|{fnm}"] = {
                **{k2: (round(v, 3) if isinstance(v, float) else v)
                   for k2, v in r.items() if k2 not in ("idx", "pnl", "ts")},
                "per_day": round(r["n"]/max(days, 1), 3),
                "monthly_exp_bp": {str(k): round(float(v), 2) for k, v in mo.items()}}

    fwd = report["oos"].get("테이커10.0bp(판정)|정방향")
    flp = report["oos"].get("테이커10.0bp(판정)|뒤집기")
    ok = bool(fwd and fwd["total_bp"] > 0 and flp and fwd["exp_bp"] > flp["exp_bp"]
              and sum(1 for v in fwd["monthly_exp_bp"].values() if v > 0) >= 2)
    log("")
    log(f"=== 판정: {'✅통과 -- HOLDOUT 노출 검토' if ok else '❌미통과'} ===")
    report["passed"] = ok
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report["runtime_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
