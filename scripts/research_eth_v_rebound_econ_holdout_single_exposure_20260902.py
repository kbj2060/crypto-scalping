#!/usr/bin/env python3
"""⚠️⚠️ E0 경제라벨 후보의 **HOLDOUT 단일 노출** -- 실행 전 반드시 아래를 읽을 것.

## 이 스크립트의 성격

HOLDOUT(2026-04-01~)은 **한 번만 쓸 수 있는 자원**이다. 이 저장소는 이미
[[feedback_holdout_survival_not_predictable_from_val_oos_20260830]]에서 "VAL/OOS를 통과해도
HOLDOUT은 실패할 수 있다"를 확인했고, 2026-09-01에는 stop 명령이 늦게 도착해 **의도치 않게
HOLDOUT이 소모된 사고**도 있었다(9트리거 풀 정의).

따라서 이 스크립트는 **어떤 탐색도 하지 않는다**:

  · 격자 없음. 임계값·셀·동시보유한도를 **앙상블 포트폴리오 리포트 JSON에서 그대로 읽는다.**
    코드 안에 상수로 적지 않는다 -- 손으로 옮기다 바꾸는 사고를 구조적으로 막는다.
  · 리포트에 `passed: true`가 없으면 **즉시 중단**한다.
  · 결과가 나쁘게 나와도 재실행/재조정하지 않는다. 1회가 전부다.

## 전제 조건 (스크립트가 직접 확인)

  1. `..._ensemble_portfolio_sim_20260902.py` 리포트 존재 + `passed == true`
  2. 리포트에 val_selection(frac/cut/cell/max_concurrent)이 모두 있을 것
  3. HOLDOUT 구간에 최소 거래 수 확보

## 판정 (사전 등록 -- 실행 전 고정)

  · HOLDOUT 순차 포트폴리오 **누적 > 0**
  · **뒤집기보다 기대값 우위**
  · 비용 **10.0bp(테이커)** 기준. 8.11bp는 참고 병기.

통과 시에만 라이브 배선을 검토한다. 미통과면 그것으로 이 후보는 종료다.

Run on the server via handoff:
  handoff.sh launch server <job> -- python scripts/research_eth_v_rebound_econ_holdout_single_exposure_20260902.py
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


_pf = _load("pf_hold", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_s1 = _pf._s1
_bt = _pf._bt
TIER0, FORWARD_BARS = _pf.TIER0, _pf.FORWARD_BARS
sim_exit, portfolio = _pf.sim_exit, _pf.portfolio
LABEL_CELL, CONTEXT_N, SEEDS = _pf.LABEL_CELL, _pf.CONTEXT_N, _pf.SEEDS
COST_TAKER, COST_MAKER, CHUNK = _pf.COST_TAKER, _pf.COST_MAKER, _pf.CHUNK

PF_REPORT = ROOT / "data/research/eth_v_rebound_ensemble_portfolio_20260902/report.json"
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")
OUT = ROOT / "data/research/eth_v_rebound_econ_holdout_20260902/report.json"


def log(m): print(f"[holdout] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    # ---------- 전제조건 확인 (실패 시 HOLDOUT 미터치로 중단) ----------
    if not PF_REPORT.exists():
        log(f"❌앙상블 포트폴리오 리포트 없음: {PF_REPORT} -- 중단(HOLDOUT 미터치)")
        return 2
    pf = json.loads(PF_REPORT.read_text())
    if not pf.get("passed"):
        log("❌포트폴리오 게이트 미통과(passed != true) -- 중단(HOLDOUT 미터치)")
        return 2
    vs = pf.get("val_selection") or {}
    need = ("cut", "cell", "max_concurrent")
    if any(k not in vs for k in need):
        log(f"❌val_selection 불완전: {vs} -- 중단(HOLDOUT 미터치)")
        return 2
    CUT = float(vs["cut"]); CELL = tuple(vs["cell"]); MC = int(vs["max_concurrent"])
    log(f"✅전제조건 통과. 고정 설정: p>={CUT:.4f}  셀 {CELL}  동시보유 {MC}")
    log("⚠️이 스크립트는 격자를 돌리지 않는다 -- 위 설정으로 HOLDOUT 1회만 평가한다.")

    from tabpfn import TabPFNClassifier
    import torch
    log(f"cuda: {torch.cuda.is_available()}")

    _s1.VAL_END = pd.Timestamp("2099-01-01", tz="UTC")   # 전 구간 허용(HOLDOUT 포함)
    log("building frame (HOLDOUT 포함) ...")
    sig, feat, eth = _s1.build_sig()
    sb = _s1.label_param(sig, True, ambig="drop", anchor="wick",
                         atr_mult=1.50, t_sustain=0.20, full_bars=12)
    st = _s1.label_param(sig, False, ambig="drop", anchor="wick",
                         atr_mult=1.50, t_sustain=0.20, full_bars=12)
    long = _s1.long_frame_for(sig, feat, sb, st)

    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    pos_of = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    nk = len(kl)
    long["pos"] = [pos_of.get(np.datetime64(t.tz_localize(None)), -1) for t in long["timestamp"]]
    long = long.loc[(long["pos"] >= 0) & (long["pos"] + FORWARD_BARS + 1 < nk)].reset_index(drop=True)

    # 학습셋은 TRAIN 그대로 (HOLDOUT은 학습에 절대 안 들어간다)
    tr_set = long.loc[long["timestamp"] < TRAIN_END].copy()
    sl0, arm0, tr0 = LABEL_CELL
    idx_tr = tr_set["pos"].to_numpy().astype(int)
    sgn_tr = np.where(tr_set["is_downside"].to_numpy() == 1, 1.0, -1.0)
    atr_tr = tr_set["atr"].to_numpy(dtype=float)
    net = np.full(len(tr_set), np.nan)
    for s_ in range(0, len(tr_set), CHUNK):
        e_ = min(s_ + CHUNK, len(tr_set))
        ii = idx_tr[s_:e_]
        H = np.stack([h[j+1:j+1+FORWARD_BARS] for j in ii])
        L = np.stack([l[j+1:j+1+FORWARD_BARS] for j in ii])
        C = np.stack([c[j+1:j+1+FORWARD_BARS] for j in ii])
        pn, _ = sim_exit(o[ii+1], atr_tr[s_:e_], sgn_tr[s_:e_], H, L, C, sl0, arm0, tr0)
        net[s_:e_] = pn * 1e4 - COST_TAKER
    tr_set["y"] = (net > 0).astype(float)
    log(f"TRAIN {len(tr_set):,}행 라벨률 {tr_set['y'].mean():.4f}")

    hd = long.loc[long["timestamp"] >= HOLDOUT_START].copy()
    log(f"HOLDOUT {len(hd):,}행  ({hd['timestamp'].min()} ~ {hd['timestamp'].max()})")
    if len(hd) < 1000:
        log("❌HOLDOUT 표본 부족 -- 중단"); return 2

    P = []
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        ctx = tr_set.iloc[np.sort(rng.choice(len(tr_set), size=min(CONTEXT_N, len(tr_set)), replace=False))]
        clf = TabPFNClassifier(device="cuda", random_state=sd, ignore_pretraining_limits=True)
        clf.fit(ctx[TIER0], ctx["y"].to_numpy())
        P.append(np.concatenate([clf.predict_proba(hd[TIER0].iloc[k:k+20000])[:, 1]
                                 for k in range(0, len(hd), 20000)]))
        log(f"  seed {sd} 완료")
    hd["p"] = np.vstack(P).mean(axis=0)

    def run(cost, flip):
        sel = hd.loc[hd["p"] >= CUT]
        if len(sel) < 30:
            return None, len(sel)
        idx = sel["pos"].to_numpy().astype(int)
        sgn = np.where(sel["is_downside"].to_numpy() == 1, 1.0, -1.0)
        if flip:
            sgn = -sgn
        H = np.stack([h[j+1:j+1+FORWARD_BARS] for j in idx])
        L = np.stack([l[j+1:j+1+FORWARD_BARS] for j in idx])
        C = np.stack([c[j+1:j+1+FORWARD_BARS] for j in idx])
        pn, ex = sim_exit(o[idx+1], sel["atr"].to_numpy(dtype=float), sgn, H, L, C, *CELL)
        cand = pd.DataFrame({"timestamp": sel["timestamp"].to_numpy(), "entry_bar": idx+1,
                             "exit_bar": idx+1+ex, "pnl_bp": pn*1e4-cost})
        return portfolio(cand, MC), len(sel)

    log("")
    log("=== ⚠️HOLDOUT 1회 노출 결과 ===")
    report = {"signal": "v_rebound_econ_holdout_single_exposure", "asset": "ETHUSDT",
              "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
              "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
              "scope": {"config_source": str(PF_REPORT.relative_to(ROOT)),
                        "cut": CUT, "cell": list(CELL), "max_concurrent": MC,
                        "no_grid_search": True, "single_exposure": True,
                        "holdout_start": str(HOLDOUT_START),
                        "period": [str(hd["timestamp"].min()), str(hd["timestamp"].max())]},
              "results": {}}
    fwd_r = None
    for cost, cnm in ((COST_TAKER, "테이커10.0bp(판정)"), (COST_MAKER, "메이커진입8.11bp(참고)")):
        for flip, fnm in ((False, "정방향"), (True, "뒤집기")):
            r, nsel = run(cost, flip)
            if r is None:
                log(f"  [{cnm}/{fnm}] 후보 {nsel} -- 표본 부족"); continue
            days = (pd.Timestamp(r["ts"].max()) - pd.Timestamp(r["ts"].min())).total_seconds()/86400
            mo = pd.Series(r["pnl"], index=pd.to_datetime(r["ts"])).groupby(
                pd.to_datetime(r["ts"]).to_period("M")).mean()
            log(f"  [{cnm}] {fnm}  n={r['n']:,} ({r['n']/max(days,1):.2f}건/일)  "
                f"기대값 {r['exp_bp']:+.2f}bp  총 {r['total_bp']:+.0f}bp  승률 {r['win_rate']*100:.1f}%  "
                f"손익비 {r['payoff']}  최대DD {r['max_dd_bp']:+.0f}bp  연속손실 {r['max_consec_loss']}")
            log(f"       월별: " + "  ".join(f"{k} {v:+.2f}bp" for k, v in mo.items()))
            report["results"][f"{cnm}|{fnm}"] = {
                **{k2: (round(v, 3) if isinstance(v, float) else v)
                   for k2, v in r.items() if k2 not in ("idx", "pnl", "ts")},
                "per_day": round(r["n"]/max(days, 1), 3),
                "monthly_exp_bp": {str(k): round(float(v), 2) for k, v in mo.items()}}
            if cost == COST_TAKER and not flip:
                fwd_r = report["results"][f"{cnm}|{fnm}"]

    flp = report["results"].get("테이커10.0bp(판정)|뒤집기")
    ok = bool(fwd_r and fwd_r["total_bp"] > 0 and flp and fwd_r["exp_bp"] > flp["exp_bp"])
    log("")
    log(f"=== 판정: {'✅HOLDOUT 통과 -- 라이브 배선 검토 가능' if ok else '❌HOLDOUT 실패 -- 이 후보 종료'} ===")
    log("⚠️결과와 무관하게 재실행하지 않는다. HOLDOUT은 1회가 전부다.")
    report["passed"] = ok
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report["runtime_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
