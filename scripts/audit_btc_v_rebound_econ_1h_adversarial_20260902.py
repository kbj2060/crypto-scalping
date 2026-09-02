#!/usr/bin/env python3
"""BTC 1시간봉 경제라벨 통과 건의 **적대적 검증** -- 진짜 방향 스킬인가.

## 왜 의심하는가

`research_btc_v_rebound_econ_1h_20260902.py`가 사전등록 3기준을 통과했다
(OOS +20.36bp / 3개월 전부 양수 / 뒤집기 -36.79bp). 비용 장벽이 제거된 것도 확인됐다
(비용/ATR 62% -> 15.1%, 무작위 기준선 -1.04 -> +5.17bp). 그런데 세 가지가 걸린다:

  1. ⭐**AUC가 0.489/0.493으로 무작위**다. 5분봉 3차와 같다. 분류력 없이 경제성이 나오는
     패턴은 ETH에서 겪었고, 그때는 **방향 쏠림 아티팩트**를 배제해야 했다.
  2. **표본이 작다** -- OOS 133건, VAL 181건(5분봉은 수백~수천). 월별 뒤집기가
     -130.88/+66.98/-38.15로 요동치는 것도 소표본 신호다.
  3. **시드 std 0.048** -- 5분봉(0.034)보다 크고 ETH 앙상블(0.021)의 2.3배.
     TRAIN이 26,950행뿐이라 컨텍스트 18,000이 거의 전량이다.

## 검정 (ETH에서 결정적이었던 것 그대로)

  A. **측면별 갭** -- 롱 호출만/숏 호출만 따로. 진짜 방향 스킬이면 **양쪽 다** 양수여야 한다.
     한쪽만 크면 드리프트 편승이다.
  B. ⭐**측면비율 매칭 귀무분포**(B=200) -- 모델의 롱/숏 비율을 그대로 유지한 채 무작위 추출.
     "모델이 골랐는가"만 남긴다. 관측이 귀무분포 안이면 아티팩트다.
  C. **랜덤 부분표집 귀무분포** -- 같은 n을 무작위로. 소표본에서 통과수/기대값이 얼마나
     불안정한지 직접 잰다.
  D. **시드별 OOS 기대값** -- 앙상블이 아니라 시드 하나씩. 부호가 갈리면 불안정 확정.

⚠️읽기 전용. HOLDOUT 미터치.
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


_h = _load("h1", "scripts/research_btc_v_rebound_econ_1h_20260902.py")
_pf = _h._pf
sim_exit, portfolio = _pf.sim_exit, _pf.portfolio
TIER0, FORWARD_BARS, COST_BP = _h.TIER0, _h.FORWARD_BARS, _h.COST_BP
SEEDS, CONTEXT_N = _h.SEEDS, _h.CONTEXT_N
TRAIN_END, VAL_END, OOS_END = _h.TRAIN_END, _h.VAL_END, _h.OOS_END

REPORT = ROOT / "data/research/btc_v_rebound_econ_1h_20260902/report.json"
NULL_B, NULL_SEED = 200, 20260902
OUT = ROOT / "data/research/btc_v_rebound_econ_1h_20260902/adversarial.json"


def log(m): print(f"[btc-1h-adv] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    rep = json.loads(REPORT.read_text())
    vs = rep["val_selection"]
    CUT, CELL, MC = float(vs["cut"]), tuple(vs["cell"]), int(vs["max_concurrent"])
    LAB = tuple(rep["label_cell"]["cell"])
    log(f"검증 대상: p>={CUT:.4f} 셀{CELL} 한도{MC} (라벨셀 {LAB})")

    f = _h.build_1h_frame()
    o, h_, l_, c = (f[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    nb = len(f)
    long = _h.to_long(f).dropna(subset=TIER0).reset_index(drop=True)
    long = long.loc[long["bar_idx"] + FORWARD_BARS + 1 < nb].reset_index(drop=True)
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                     np.where(long["timestamp"] < VAL_END, "VAL",
                      np.where(long["timestamp"] < OOS_END, "OOS", "HOLDOUT")))
    long = long.loc[long["split"] != "HOLDOUT"].reset_index(drop=True)

    def net_for(sub, cell):
        sl, arm, trv = cell
        idx = sub["bar_idx"].to_numpy().astype(int)
        sgn = np.where(sub["is_downside"].to_numpy() == 1, 1.0, -1.0)
        at = sub["atr"].to_numpy(dtype=float)
        H = np.stack([h_[x+1:x+1+FORWARD_BARS] for x in idx])
        L = np.stack([l_[x+1:x+1+FORWARD_BARS] for x in idx])
        C = np.stack([c[x+1:x+1+FORWARD_BARS] for x in idx])
        pn, ex = sim_exit(o[idx+1], at, sgn, H, L, C, sl, arm, trv)
        return pn * 1e4 - COST_BP, ex

    v_all, _ = net_for(long, LAB)
    long["y"] = (v_all > 0).astype(float)
    tr_set = long.loc[long["split"] == "TRAIN"]

    # 시드별 확률 (D 검정용으로 개별 보존)
    per_seed = {"VAL": [], "OOS": []}
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        ctx = tr_set.iloc[np.sort(rng.choice(len(tr_set), size=min(CONTEXT_N, len(tr_set)), replace=False))]
        clf = TabPFNClassifier(device="cuda", random_state=sd, ignore_pretraining_limits=True)
        clf.fit(ctx[TIER0], ctx["y"].to_numpy())
        for spn in ("VAL", "OOS"):
            s = long.loc[long["split"] == spn]
            per_seed[spn].append(np.concatenate(
                [clf.predict_proba(s[TIER0].iloc[k:k+20000])[:, 1] for k in range(0, len(s), 20000)]))
        log(f"  seed {sd} 완료")
    scored = {}
    for spn in ("VAL", "OOS"):
        s = long.loc[long["split"] == spn].copy()
        s["p"] = np.vstack(per_seed[spn]).mean(axis=0)
        scored[spn] = s.reset_index(drop=True)

    def pf_of(sub, flip=False):
        if len(sub) < 15:
            return None
        s2 = sub.copy()
        if flip:
            s2["is_downside"] = 1 - s2["is_downside"]
        v, ex = net_for(s2, CELL)
        idx = sub["bar_idx"].to_numpy().astype(int)
        cd = pd.DataFrame({"timestamp": sub["timestamp"].to_numpy(), "entry_bar": idx + 1,
                           "exit_bar": idx + 1 + ex, "pnl_bp": v})
        return portfolio(cd, MC)

    report = {"asset": "BTCUSDT", "bar": "1h", "config": {"cut": CUT, "cell": list(CELL),
              "max_concurrent": MC, "label_cell": list(LAB)}, "tests": {}}

    # ---- A) 측면별 갭 ----
    log("")
    log("=== A) 측면별 갭 (진짜 스킬이면 양쪽 다 정방향 우위) ===")
    side_res = {}
    for spn in ("VAL", "OOS"):
        sel = scored[spn].loc[scored[spn]["p"] >= CUT]
        nl = int((sel["is_downside"] == 1).sum())
        log(f"  {spn}: 호출 {len(sel)}  롱 {nl} ({nl/max(len(sel),1)*100:.1f}%) 숏 {len(sel)-nl}")
        d = {}
        for side, nm in ((1, "롱"), (0, "숏")):
            sub = sel.loc[sel["is_downside"] == side]
            fw, fl = pf_of(sub), pf_of(sub, flip=True)
            if fw and fl:
                gap = fw["exp_bp"] - fl["exp_bp"]
                d[nm] = {"n": fw["n"], "fwd": round(fw["exp_bp"], 2),
                         "flip": round(fl["exp_bp"], 2), "gap": round(gap, 2)}
                log(f"     {nm} n={fw['n']:>4d}  정 {fw['exp_bp']:+8.2f}  뒤 {fl['exp_bp']:+8.2f}  "
                    f"갭 {gap:+8.2f}bp {'✅' if gap > 0 else '❌'}")
            else:
                log(f"     {nm} 표본 부족(n={len(sub)})")
        side_res[spn] = {"n_calls": int(len(sel)), "long_pct": round(nl/max(len(sel),1)*100, 1),
                         "per_side": d}
    report["tests"]["A_side_gap"] = side_res

    # ---- B/C) 귀무분포 ----
    log("")
    log(f"=== B/C) 귀무분포 (B={NULL_B}) ===")
    nrng = np.random.default_rng(NULL_SEED)
    null_res = {}
    for spn in ("VAL", "OOS"):
        s = scored[spn]
        sel = s.loc[s["p"] >= CUT]
        n = len(sel); nl = int((sel["is_downside"] == 1).sum()); ns = n - nl
        obs_f, obs_r = pf_of(sel), pf_of(sel, flip=True)
        obs_gap = obs_f["exp_bp"] - obs_r["exp_bp"]
        pl, ps = s.loc[s["is_downside"] == 1], s.loc[s["is_downside"] == 0]
        gaps_m, gaps_r = [], []
        for _ in range(NULL_B):
            a = pl.iloc[nrng.choice(len(pl), size=min(nl, len(pl)), replace=False)]
            b = ps.iloc[nrng.choice(len(ps), size=min(ns, len(ps)), replace=False)]
            m = pd.concat([a, b])
            fw, fl = pf_of(m), pf_of(m, flip=True)
            if fw and fl:
                gaps_m.append(fw["exp_bp"] - fl["exp_bp"])
            r = s.iloc[nrng.choice(len(s), size=min(n, len(s)), replace=False)]
            fw2, fl2 = pf_of(r), pf_of(r, flip=True)
            if fw2 and fl2:
                gaps_r.append(fw2["exp_bp"] - fl2["exp_bp"])
        pm = round(float((np.array(gaps_m) < obs_gap).mean()*100), 1) if len(gaps_m) >= 20 else None
        pr = round(float((np.array(gaps_r) < obs_gap).mean()*100), 1) if len(gaps_r) >= 20 else None
        log(f"  {spn}: 관측 갭 {obs_gap:+.2f}bp")
        log(f"     측면매칭 귀무 평균 {np.mean(gaps_m):+.2f}bp  백분위 {pm}% "
            f"{'✅스킬' if (pm or 0) >= 95 else '❌아티팩트'}")
        log(f"     무작위   귀무 평균 {np.mean(gaps_r):+.2f}bp  백분위 {pr}%")
        null_res[spn] = {"obs_gap": round(obs_gap, 2),
                         "side_matched_mean": round(float(np.mean(gaps_m)), 2),
                         "side_matched_pctile": pm,
                         "random_mean": round(float(np.mean(gaps_r)), 2),
                         "random_pctile": pr, "B": len(gaps_m)}
    report["tests"]["BC_null"] = null_res

    # ---- D) 시드별 OOS ----
    log("")
    log("=== D) 시드별 OOS 기대값 (앙상블이 아니라 개별) ===")
    seed_exp = []
    s_oos = long.loc[long["split"] == "OOS"].reset_index(drop=True)
    for i, sd in enumerate(SEEDS):
        s2 = s_oos.copy(); s2["p"] = per_seed["OOS"][i]
        sel = s2.loc[s2["p"] >= CUT]
        r = pf_of(sel)
        e = r["exp_bp"] if r else float("nan")
        seed_exp.append(e)
        log(f"  seed {sd}: n={r['n'] if r else 0:>4d}  기대값 {e:+8.2f}bp")
    arr = np.array([x for x in seed_exp if x == x])
    log(f"  평균 {arr.mean():+.2f}bp  std {arr.std():.2f}  양수 {int((arr>0).sum())}/{len(arr)}")
    report["tests"]["D_per_seed"] = {"oos_exp_bp": [round(float(x), 2) for x in seed_exp],
                                     "mean": round(float(arr.mean()), 2),
                                     "std": round(float(arr.std()), 2),
                                     "n_positive": int((arr > 0).sum())}

    # ---- 종합 ----
    both_pos = all(all(v["gap"] > 0 for v in side_res[spn]["per_side"].values())
                   and len(side_res[spn]["per_side"]) == 2 for spn in ("VAL", "OOS"))
    null_ok = all((null_res[spn]["side_matched_pctile"] or 0) >= 95 for spn in ("VAL", "OOS"))
    seed_ok = bool((arr > 0).all())
    log("")
    log("=== 종합 ===")
    log(f"  {'✅' if both_pos else '❌'} A 측면별 양쪽 갭 양수")
    log(f"  {'✅' if null_ok else '❌'} B 측면매칭 귀무 >=95%")
    log(f"  {'✅' if seed_ok else '❌'} D 5시드 전부 OOS 양수")
    ok = both_pos and null_ok and seed_ok
    log(f"  ⇒ {'✅진짜 방향 스킬 -- HOLDOUT 검토 가능' if ok else '❌검증 미통과'}")
    report["passed"] = ok
    report["runtime_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
