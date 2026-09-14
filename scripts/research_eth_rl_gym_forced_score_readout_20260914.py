"""**강제 거래 정책의 점수 순위 판독** — 측면 붕괴를 걷어내고 «선호 순서」에 시점 정보가 있는지 (2026-09-14).

강제 거래(홀드 제거) 팔은 씨드마다 «항상 숏」/«항상 롱」으로 붕괴했다(측면 1.00). argmax 가
상수 편향에 지배돼 상태 의존 부분이 안 보인다. 재훈련 대신 **판독기**를 바꾼다:
  점수 s_i = logit롱 − logit숏 을 평가창 안에서 순위화 → 상위 q 롱 · 하위 q 숏 · 나머지 홀드.
측면이 구조적으로 균형되므로 남는 것은 «어느 봉을 롱/숏으로 골랐나」뿐이다.

판정은 사전등록과 같다: Δ(점수 규칙 − 같은 측면 무작위) 씨드 짝지음 ± SE · 적중률 vs 손익분기 · 역방향.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_rl_gym_direction_ppo_20260914 as P  # noqa: E402
import rl_gym_direction_env_20260914 as G  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", type=str, default="rm_forced,rm_forced_bal,rm_forced_ent")
    ap.add_argument("--q", type=float, default=0.10)
    a = ap.parse_args()
    d, sm, win, S, cols, _ = P.prepare(G.DEFAULT_FAMILIES)
    rng = np.random.default_rng(20260914)
    out = {"q": a.q, "tags": {}}
    for tag in a.tags.split(","):
        files = sorted(glob.glob(str(P.OUT / f"policy_{tag}_ent*_seed*.pt")))
        if not files:
            print(f"[{tag}] 저장된 정책 없음"); continue
        print(f"\n[{tag}] 정책 {len(files)}개 · 상위/하위 {a.q:.0%}")
        res = {w: [] for w in ("VAL", "OOS", "TEST", "BACK22_23")}
        for f in files:
            m = P.ActorCritic(S.shape[1]); m.load_state_dict(torch.load(f)); m.eval()
            for w in res:
                lo, hi = win[w]
                with torch.no_grad():
                    logits, _ = m(torch.as_tensor(S[lo:hi]))
                s = (logits[:, 1] - logits[:, 2]).numpy()
                q_lo, q_hi = np.quantile(s, [a.q, 1 - a.q])
                A = np.where(s >= q_hi, 1, np.where(s <= q_lo, 2, 0))
                pol = P.eval_policy(None, d, sm, S, win, w, A_override=A)[0]
                ssr = P.eval_policy(None, d, sm, S, win, w, A_override=rng.permutation(A))[0]
                rev = P.eval_policy(None, d, sm, S, win, w, A_override=np.where(A == 1, 2, np.where(A == 2, 1, 0)))[0]
                res[w].append({"pol": pol, "ssr": ssr, "rev": rev})
        summ = {}
        for w, rows in res.items():
            if not rows:
                continue
            dl = [r["pol"]["net_bp"] - r["ssr"]["net_bp"] for r in rows]
            se = float(np.std(dl, ddof=1) / math.sqrt(len(dl))) if len(dl) > 1 else float("nan")
            summ[w] = {"n_seeds": len(rows),
                       "trades": float(np.mean([r["pol"]["trades"] for r in rows])),
                       "net_bp": float(np.mean([r["pol"]["net_bp"] for r in rows])),
                       "hit_rate": float(np.mean([r["pol"]["hit_rate"] for r in rows])),
                       "side_share": float(np.mean([r["pol"]["side_share"] for r in rows])),
                       "delta_vs_ssr": float(np.mean(dl)), "delta_se": se,
                       "reversed_net_bp": float(np.mean([r["rev"]["net_bp"] for r in rows]))}
            v = summ[w]
            print(f"  {w:<10} 씨드 {v['n_seeds']} · 거래 {v['trades']:.0f} · 순 {v['net_bp']:+.2f}bp · 적중 {100*v['hit_rate']:.1f}%"
                  f" · 측면 {v['side_share']:.2f} · Δ(vs 같은측면무작위) {v['delta_vs_ssr']:+.2f} ± {v['delta_se']:.2f} · 역방향 {v['reversed_net_bp']:+.2f}")
        out["tags"][tag] = summ
    p = P.OUT / "report_forced_score_readout.json"
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"저장 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
