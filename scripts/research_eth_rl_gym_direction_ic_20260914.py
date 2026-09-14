"""**방향 정보가 있는가** — gym 라벨로 재는 표본외 IC 와 손익분기 정확도 (2026-09-14).

PPO 세 팔이 전부 «관망」으로 수렴했고 지도학습 쌍둥이도 세 창 음수였다. 그 둘은 **정책의 결론**이라
«정보가 없다»를 직접 말하지는 않는다. 이 스크립트가 그 자리를 잰다 -- 모델을 거치지 않는 세 숫자:

  ① **비용 전 방향 수익**: 항상롱·항상숏의 평균은 정확히 −비용이고, 그 차이의 절반이 표류다.
  ② **쌍둥이 예측의 표본외 IC**: 학습창에서 적합한 회귀가 평가창에서 실현치와 얼마나 상관되나.
  ③ **손익분기 정확도 대비 실현 적중률**: 이 지평에서 몇 %를 맞혀야 본전인가, 실제로 몇 %인가.

라벨 `y(i) = ½·(r_long − r_short)·1e4` 는 **비용이 상쇄된 순수 가격 변동 bp** 다(같은 봉에서
양측면을 각각 한 번 넣어 배포 스택이 청산할 때까지 굴린 결과). 즉 y 의 부호를 맞히는 것이
이 gym 에서 «방향을 맞힌다」의 정의다.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_rl_gym_direction_ppo_20260914 as P  # noqa: E402
import rl_gym_direction_env_20260914 as G  # noqa: E402


def labels_window(d, sm, lo: int, hi: int, stride: int, cost=None):
    idx, ys = [], []
    for i in range(lo, hi, stride):
        if not sm["ok"][i]:
            continue
        a = P.trade_outcome(d, sm, int(i), 1, cost); b = P.trade_outcome(d, sm, int(i), 2, cost)
        if a is None or b is None:
            continue
        idx.append(i); ys.append(0.5e4 * (a - b))
    return np.array(idx), np.array(ys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--seed", type=int, default=P.SEEDS[0])
    a = ap.parse_args()
    d, sm, win, S, cols, _ = P.prepare(G.DEFAULT_FAMILIES)
    out = {"stride": a.stride, "seed": a.seed, "windows": {}}

    print(f"학습창에서 쌍둥이 적합 (stride 4) …", flush=True)
    tr_idx, tr_y = P.twin_labels(d, sm, win, 4)
    model = P.twin_fit_predict(S, tr_idx, tr_y, a.seed)
    print(f"  라벨 {len(tr_y):,} · 평균 {tr_y.mean():+.2f}bp · SD {tr_y.std():.1f}bp"
          f" · 표본내 IC {spearmanr(model.predict(S[tr_idx]), tr_y).statistic:+.4f}", flush=True)
    out["train"] = {"n": int(len(tr_y)), "mean_bp": float(tr_y.mean()), "sd_bp": float(tr_y.std()),
                    "ic_in_sample": float(spearmanr(model.predict(S[tr_idx]), tr_y).statistic)}

    for w in ("VAL", "OOS", "TEST", "BACK22_23"):
        lo, hi = win[w]
        idx, y = labels_window(d, sm, lo, hi, a.stride)
        pred = model.predict(S[idx])
        ic = spearmanr(pred, y)
        # 실현 적중률: 예측 부호가 실제 부호와 같은 비율 (|y|>0 인 것만)
        m = np.abs(y) > 1e-9
        hit = float((np.sign(pred[m]) == np.sign(y[m])).mean())
        # 이 창의 평균 왕복 비용 -- 항상롱·항상숏 평균의 부호를 뒤집은 값
        al = P.eval_policy(None, d, sm, S, win, w, A_override=np.ones(hi - lo, int))[0]
        ash = P.eval_policy(None, d, sm, S, win, w, A_override=np.full(hi - lo, 2))[0]
        cost_bp = -(al["net_bp"] + ash["net_bp"]) / 2
        drift_bp = (al["net_bp"] - ash["net_bp"]) / 2
        gross_sd = float(np.std(y))
        # 손익분기 정확도: (2a−1)·E|y| = 비용  ⇒  a* = ½ + 비용/(2·E|y|)
        e_abs = float(np.mean(np.abs(y)))
        breakeven = 0.5 + cost_bp / (2 * e_abs) if e_abs > 0 else float("nan")
        # 상위 십분위 |예측| 에서만 베팅했을 때의 평균 방향수익(부호 맞춘 쪽)
        k = int(0.1 * len(pred))
        top = np.argsort(-np.abs(pred))[:k]
        sel_bp = float(np.mean(np.sign(pred[top]) * y[top]))
        out["windows"][w] = {"n": int(len(y)), "ic": float(ic.statistic), "ic_p": float(ic.pvalue),
                             "hit_rate": hit, "breakeven_acc": breakeven, "cost_bp": float(cost_bp),
                             "drift_bp": float(drift_bp), "mean_abs_move_bp": e_abs,
                             "gross_sd_bp": gross_sd, "top_decile_dir_bp": sel_bp,
                             "top_decile_net_bp": sel_bp - float(cost_bp)}
        print(f"[{w}] n {len(y):,} · IC {ic.statistic:+.4f} (p {ic.pvalue:.3f}) · 적중 {100*hit:.1f}%"
              f" vs 손익분기 {100*breakeven:.1f}% · 비용 {cost_bp:.2f}bp · 표류 {drift_bp:+.2f}bp"
              f" · E|이동| {e_abs:.1f}bp · 상위10% 방향수익 {sel_bp:+.2f} → 순 {sel_bp-cost_bp:+.2f}bp",
              flush=True)

    p = P.OUT / "report_direction_ic.json"
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"저장 {p}")
    # 자체점검: 비용 추정이 배포 상수와 같은 자리인가(왕복 5.88 + 손절추가 + 펀딩)
    cs = [out["windows"][w]["cost_bp"] for w in ("VAL", "OOS", "TEST")]
    assert 4.0 < np.mean(cs) < 12.0, f"비용 추정 {np.mean(cs):.2f}bp 가 배포 비용 대역 밖이다"
    assert out["train"]["ic_in_sample"] > 0.02, "표본내 IC 조차 0 이면 회귀가 안 돌았다"
    print(f"확인: 창 평균 비용 {np.mean(cs):.2f}bp · 표본내 IC {out['train']['ic_in_sample']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
