#!/usr/bin/env python3
"""HOLDOUT 노출 **전** 실행 강건성 검정 -- 죽일 수 있는 것을 먼저 죽인다.

## 왜 지금

`..._ensemble_portfolio_sim_20260902.py`가 사전등록 기준 3개를 전부 통과했다
(OOS 기대값 +7.98bp / 누적 +11,031bp / 3개월 전부 양수 / 뒤집기 +0.32bp).
다음은 HOLDOUT인데 **1회뿐**이므로, VAL/OOS만으로 확인 가능한 실행 리스크를 먼저 소진한다.

## 검정 (전부 VAL 선정 설정 그대로, OOS에서 평가)

  R1 **진입 1봉 지연** -- 현행은 `open[i+1]` 진입. 라이브 사이클이 6.56초라 현실적이지만,
     신호 계산·주문 지연으로 한 봉 밀리면 어떻게 되는지. **가장 죽기 쉬운 축.**
  R2 **비용 상향** 10 -> 12 -> 15 -> 20bp -- 스프레드 확대/펀딩/슬리피지 여유
  R3 **슬리피지** 진입·청산 각 1/2/3bp 추가 (왕복 2/4/6bp)
  R4 **동시보유 한도 축소** 5 -> 3 -> 1 -- 자본 제약이 더 빡빡할 때

기준: R1(1봉 지연)에서 기대값이 양수로 남고 뒤집기보다 우위여야 한다. 여기서 죽으면
실행 민감도가 너무 높아 라이브로 못 간다.

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


_pf = _load("pf_rb", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_s1, _bt = _pf._s1, _pf._bt
TIER0, FORWARD_BARS = _pf.TIER0, _pf.FORWARD_BARS
sim_exit, portfolio = _pf.sim_exit, _pf.portfolio
LABEL_CELL, CONTEXT_N, SEEDS, CHUNK = _pf.LABEL_CELL, _pf.CONTEXT_N, _pf.SEEDS, _pf.CHUNK

PF_REPORT = ROOT / "data/research/eth_v_rebound_ensemble_portfolio_20260902/report.json"
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
OUT = ROOT / "data/research/eth_v_rebound_execution_robustness_20260902/report.json"


def log(m): print(f"[robust] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    pf = json.loads(PF_REPORT.read_text())
    vs = pf["val_selection"]
    CUT, CELL, MC0 = float(vs["cut"]), tuple(vs["cell"]), int(vs["max_concurrent"])
    log(f"VAL 선정 설정 고정: p>={CUT:.4f}  셀 {CELL}  동시보유 {MC0}")

    from tabpfn import TabPFNClassifier
    import torch
    log(f"cuda: {torch.cuda.is_available()}")

    _s1.VAL_END = OOS_END
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
    long = long.loc[(long["pos"] >= 0) & (long["pos"] + FORWARD_BARS + 4 < nk)].reset_index(drop=True)

    sl0, arm0, tr0 = LABEL_CELL
    i_all = long["pos"].to_numpy().astype(int)
    sgn_all = np.where(long["is_downside"].to_numpy() == 1, 1.0, -1.0)
    atr_all = long["atr"].to_numpy(dtype=float)
    net = np.full(len(long), np.nan)
    for s_ in range(0, len(long), CHUNK):
        e_ = min(s_ + CHUNK, len(long))
        ii = i_all[s_:e_]
        H = np.stack([h[j+1:j+1+FORWARD_BARS] for j in ii])
        L = np.stack([l[j+1:j+1+FORWARD_BARS] for j in ii])
        C = np.stack([c[j+1:j+1+FORWARD_BARS] for j in ii])
        pn, _ = sim_exit(o[ii+1], atr_all[s_:e_], sgn_all[s_:e_], H, L, C, sl0, arm0, tr0)
        net[s_:e_] = pn * 1e4 - 10.0
    long["y"] = (net > 0).astype(float)

    tr_set = long.loc[long["split"] == "TRAIN"]
    oos = long.loc[long["split"] == "OOS"].copy()
    P = []
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        ctx = tr_set.iloc[np.sort(rng.choice(len(tr_set), size=min(CONTEXT_N, len(tr_set)), replace=False))]
        clf = TabPFNClassifier(device="cuda", random_state=sd, ignore_pretraining_limits=True)
        clf.fit(ctx[TIER0], ctx["y"].to_numpy())
        P.append(np.concatenate([clf.predict_proba(oos[TIER0].iloc[k:k+20000])[:, 1]
                                 for k in range(0, len(oos), 20000)]))
    oos["p"] = np.vstack(P).mean(axis=0)
    log(f"OOS 앙상블 완료. 호출 {int((oos['p']>=CUT).sum()):,}건")

    def run(delay=1, cost=10.0, slip=0.0, mc=None, flip=False):
        """delay: 진입 봉 오프셋(1=open[i+1]). slip: 편도 bp(왕복 2*slip 적용)."""
        mc = MC0 if mc is None else mc
        sel = oos.loc[oos["p"] >= CUT]
        idx = sel["pos"].to_numpy().astype(int)
        if (idx + delay + FORWARD_BARS >= nk).any():
            keep = idx + delay + FORWARD_BARS < nk
            sel, idx = sel.loc[keep], idx[keep]
        if len(sel) < 30:
            return None
        sgn = np.where(sel["is_downside"].to_numpy() == 1, 1.0, -1.0)
        if flip:
            sgn = -sgn
        H = np.stack([h[j+delay:j+delay+FORWARD_BARS] for j in idx])
        L = np.stack([l[j+delay:j+delay+FORWARD_BARS] for j in idx])
        C = np.stack([c[j+delay:j+delay+FORWARD_BARS] for j in idx])
        pn, ex = sim_exit(o[idx+delay], sel["atr"].to_numpy(dtype=float), sgn, H, L, C, sl0, arm0, tr0)
        cand = pd.DataFrame({"timestamp": sel["timestamp"].to_numpy(), "entry_bar": idx+delay,
                             "exit_bar": idx+delay+ex, "pnl_bp": pn*1e4 - cost - 2*slip})
        return portfolio(cand, mc)

    def line(tag, r, rf=None):
        if r is None:
            log(f"  {tag:34s} 표본 부족"); return None
        fx = f"  뒤집기 {rf['exp_bp']:+.2f}bp" if rf else ""
        ok = r["exp_bp"] > 0 and (rf is None or r["exp_bp"] > rf["exp_bp"])
        log(f"  {tag:34s} n={r['n']:>5,}  기대값 {r['exp_bp']:>+7.2f}bp  총 {r['total_bp']:>+8.0f}bp  "
            f"승률 {r['win_rate']*100:4.1f}%  최대DD {r['max_dd_bp']:>+7.0f}bp{fx}"
            f"{'  ✅' if ok else '  ❌'}")
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in r.items() if k not in ("idx", "pnl", "ts")}

    report = {"signal": "v_rebound_execution_robustness", "asset": "ETHUSDT",
              "scope": {"config": {"cut": CUT, "cell": list(CELL), "max_concurrent": MC0},
                        "split": "OOS only", "holdout_touched": False,
                        "live_code_changed": False}, "tests": {}}

    log("")
    log("=== 기준선 (진입 open[i+1], 비용 10bp, 한도 5) ===")
    base = run(); basef = run(flip=True)
    report["tests"]["base"] = line("base", base, basef)

    log("")
    log("=== R1 ⭐진입 지연 (가장 죽기 쉬운 축) ===")
    for d in (1, 2, 3):
        r, rf = run(delay=d), run(delay=d, flip=True)
        report["tests"][f"R1_delay_{d}"] = line(f"진입 open[i+{d}]", r, rf)

    log("")
    log("=== R2 비용 상향 ===")
    for cost in (10.0, 12.0, 15.0, 20.0):
        r, rf = run(cost=cost), run(cost=cost, flip=True)
        report["tests"][f"R2_cost_{cost:g}"] = line(f"비용 {cost:g}bp", r, rf)

    log("")
    log("=== R3 슬리피지 추가 (편도) ===")
    for sp in (1.0, 2.0, 3.0):
        r, rf = run(slip=sp), run(slip=sp, flip=True)
        report["tests"][f"R3_slip_{sp:g}"] = line(f"슬리피지 편도 {sp:g}bp (왕복 {2*sp:g})", r, rf)

    log("")
    log("=== R4 동시보유 한도 축소 ===")
    for mc in (5, 3, 1):
        r, rf = run(mc=mc), run(mc=mc, flip=True)
        report["tests"][f"R4_mc_{mc}"] = line(f"동시보유 {mc}", r, rf)

    log("")
    log("=== R1+R2 결합 (1봉 지연 + 15bp) ===")
    r, rf = run(delay=2, cost=15.0), run(delay=2, cost=15.0, flip=True)
    report["tests"]["R1R2_combo"] = line("지연1봉 + 비용15bp", r, rf)

    t = report["tests"]
    d1 = t.get("R1_delay_2")
    passed = bool(d1 and d1["exp_bp"] > 0)
    log("")
    log(f"=== 판정: {'✅1봉 지연에서도 생존 -- HOLDOUT 노출 진행 가능' if passed else '❌1봉 지연에서 붕괴 -- 실행 민감도 과다'} ===")
    report["passed_delay_test"] = passed
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report["runtime_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
