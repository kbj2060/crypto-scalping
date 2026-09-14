"""**제로샷 표형 쌍둥이** — TabICL(in-context 학습기)로 gym 라벨 위의 방향 정보를 다시 잰다 (2026-09-14).

사용자 *"tabpfn 모델로 정확도를 많이 올린거 같은데 다시 제로샷 모델 찾아봐"*.
GBM 쌍둥이(§5)는 표본내 IC +0.20 → 표본외 음수였다. in-context 모델은 학습 없이 문맥만 보므로
과적합 모양이 다를 수 있다 -- 같은 라벨·같은 창·같은 판정(IC · 적중률 vs 손익분기 · 상위10% 순bp ·
gym 안 순bp)으로 잰다. 로컬 설치본은 TabICL 2.2(회귀 지원). TabPFN 은 로컬 미설치(서버 GPU 포화).

ponytail: 문맥은 학습창 라벨에서 무작위 10,000 행(TabPFN v3 관행과 같은 크기). 평가 행은 창별 전부.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_rl_gym_direction_ppo_20260914 as P  # noqa: E402
import rl_gym_direction_env_20260914 as G  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", type=str, default=",".join(G.DEFAULT_FAMILIES))
    ap.add_argument("--context", type=int, default=10000)
    ap.add_argument("--tag", type=str, default="tabicl43")
    a = ap.parse_args()
    from tabicl import TabICLRegressor
    fam = tuple(a.families.split(","))
    d, sm, win, S, cols, _ = P.prepare(fam)
    lab = P.load_labels(d)   # 🔴타임스탬프로 정렬 -- 머신마다 klines 길이가 다르다
    tr_idx, tr_y = lab["TRAIN"]
    rng = np.random.default_rng(P.SEEDS[0])
    sub = rng.choice(len(tr_idx), size=min(a.context, len(tr_idx)), replace=False)
    Xc, yc = S[tr_idx[sub]], tr_y[sub]
    print(f"[{a.tag}] 상태 {S.shape[1]}D · 문맥 {len(sub):,} · 평가 {[ (w, len(lab[w][1])) for w in ('VAL','OOS','TEST') ]}", flush=True)
    t0 = time.time()
    m = TabICLRegressor(device="cpu"); m.fit(Xc, yc)
    out = {"tag": a.tag, "dim": int(S.shape[1]), "context": int(len(sub)), "windows": {}}
    preds = {}
    for w in ("VAL", "OOS", "TEST"):
        idx, y = lab[w]; t1 = time.time()
        pred = m.predict(S[idx]); preds[w] = pred
        ic = spearmanr(pred, y); mm = np.abs(y) > 1e-9
        hit = float((np.sign(pred[mm]) == np.sign(y[mm])).mean())
        k = int(0.1 * len(pred)); top = np.argsort(-np.abs(pred))[:k]
        sel = float(np.mean(np.sign(pred[top]) * y[top]))
        al = P.eval_policy(None, d, sm, S, win, w, A_override=np.ones(win[w][1] - win[w][0], int))[0]["net_bp"]
        ash = P.eval_policy(None, d, sm, S, win, w, A_override=np.full(win[w][1] - win[w][0], 2))[0]["net_bp"]
        cost = -(al + ash) / 2
        out["windows"][w] = {"n": int(len(y)), "ic": float(ic.statistic), "p": float(ic.pvalue), "hit_rate": hit,
                             "top_decile_dir_bp": sel, "top_decile_net_bp": sel - cost, "cost_bp": cost,
                             "sec": time.time() - t1}
        v = out["windows"][w]
        print(f"  {w:<5} n {v['n']:,} · IC {v['ic']:+.4f} (p {v['p']:.3f}) · 적중 {100*hit:.1f}% · 상위10% 방향 {sel:+.2f} → 순 {sel-cost:+.2f}bp · {v['sec']:.0f}s", flush=True)
    # gym 안에서의 쌍둥이 정책 (임계 c 는 VAL 에서 선택, 다른 창엔 그대로)
    best_c, best = None, None
    for c in (0.0, 5.0, 10.0, 20.0, 40.0):
        idx = lab["VAL"][0]; A = np.zeros(win["VAL"][1] - win["VAL"][0], int)
        A[idx - win["VAL"][0]] = np.where(preds["VAL"] > c, 1, np.where(preds["VAL"] < -c, 2, 0))
        r = P.eval_policy(None, d, sm, S, win, "VAL", A_override=A)[0]
        if best is None or r["net_bp"] > best["net_bp"]:
            best_c, best = c, r
    gymres = {}
    for w in ("VAL", "OOS", "TEST"):
        idx = lab[w][0]; A = np.zeros(win[w][1] - win[w][0], int)
        A[idx - win[w][0]] = np.where(preds[w] > best_c, 1, np.where(preds[w] < -best_c, 2, 0))
        r = P.eval_policy(None, d, sm, S, win, w, A_override=A)[0]
        gymres[w] = {"trades": int(r["trades"]), "net_bp": float(r["net_bp"]), "hit_rate": float(r["hit_rate"]), "t_day": float(r["t_day"])}
        print(f"  gym[{w}] c={best_c:g}bp · 거래 {r['trades']} · 순 {r['net_bp']:+.2f}bp · 적중 {100*r['hit_rate']:.1f}% · 일군집 t {r['t_day']:+.2f}", flush=True)
    out["gym_policy"] = {"threshold_bp": best_c, **gymres}
    out["fit_sec"] = time.time() - t0
    p = P.OUT / f"report_zeroshot_{a.tag}.json"
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"저장 {p} · 총 {out['fit_sec']:.0f}s")
    ics = [out["windows"][w]["ic"] for w in ("VAL", "OOS", "TEST")]
    assert all(np.isfinite(ics)), "IC 계산 실패"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
