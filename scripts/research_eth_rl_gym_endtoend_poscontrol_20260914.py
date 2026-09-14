"""**끝까지 통과하는 양성대조** — «아키텍처가 배울 수 있는가」를 gym 안에서 직접 잰다 (2026-09-14).

사용자 *"지금 학습이 아예 안되는 아키텍쳐를 가진거 같아."*

## 🔴기존 양성대조(G1)의 구멍
`research_eth_rl_gym_direction_ppo_20260914.train(positive=...)` 은 **gym 을 통과하지 않는다** —
자체 `_decisions` 를 만들어 매 봉 즉시 보상(±0.01)을 주고 다음 봉으로 넘어간다. `gym.run()` 을
한 번도 안 부른다. 그래서 그 .800 은 다음만 검증했다:
  ✅ 상태 → 신경망 → PPO 갱신 → 정책이 바뀐다
검증하지 **않은** 것(= 실제 방향 학습이 지나는 경로 전부):
  ❌ semi-MDP 전이(빈 슬롯에서만 결정) · ❌ **청산 시점의 희소 보상 한 방** ·
  ❌ 가변 길이 옵션 위의 GAE·할인 · ❌ 실제 보상 눈금(로그 자산비 ×100) · ❌ 비용·손절·사다리

## 이 스크립트가 하는 일
**가격을 합성**해 «정답이 상태 안에 있는」 세계를 만들고, **나머지는 전부 진짜 gym** 으로 돌린다.
  · 외생 신호 `synth`(AR(1), 가격과 무관하게 생성) 를 상태에 **한 열 추가**한다.
  · 합성 로그수익 = `mu * sign(synth) + 잡음`. mu 는 「48봉 이동폭 = EDGE bp」가 되게 잡는다.
  · 고가/저가는 실제 봉의 (고가−종가)/(종가−저가) 비율을 보존해 다시 만든다 — 손절·배리어가 산다.
  · 라벨·비용·사이징·손절·사다리·semi-MDP·희소 보상은 **손대지 않는다**.
⇒ 이 세계에서 PPO 가 못 배우면 **배관이 깨진 것**이고, 배우면 아키텍처는 멀쩡하고 정보가 없는 것이다.

## EDGE 격자 = 「이 아키텍처의 탐지 하한」
EDGE 를 100 → 30 → 10bp 로 낮춰 가며 어디서 못 배우는지 본다. 그 값이 **«아키텍처가 감지할 수
있는 최소 엣지»** 이고, 실제 데이터의 엣지(≈0, 손익분기 6bp)와 직접 비교할 수 있는 숫자다.
⚠️합성 세계의 난이도는 실제와 다르다 — 이 숫자는 **상한 진단**이지 실제 성능 예측이 아니다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_rl_gym_direction_ppo_20260914 as P  # noqa: E402
import rl_gym_direction_env_20260914 as G  # noqa: E402

ENT, EPOCHS, MB = 0.001, 4, 8192


def synth_world(d: pd.DataFrame, edge_bp: float, horizon: int, seed: int):
    """정답이 상태에 있는 합성 가격. 반환: (close, high, low, synth 열)."""
    rng = np.random.default_rng(seed)
    n = len(d)
    # 외생 AR(1) 신호 -- 가격에서 유도되지 않는다(순환 없음). 지평 동안 대체로 유지되게 느리게.
    rho = float(np.exp(-1.0 / horizon))
    e = rng.normal(size=n)
    s = np.empty(n); s[0] = e[0]
    for i in range(1, n):
        s[i] = rho * s[i - 1] + np.sqrt(1 - rho ** 2) * e[i]
    sign = np.sign(s); sign[sign == 0] = 1.0
    # 봉당 표류: horizon 봉 누적이 edge_bp 가 되게
    mu = (edge_bp / 1e4) / horizon
    # 잡음은 실제 봉의 변동성을 그대로 쓴다(난이도를 실제와 비슷하게)
    c0 = d.close.to_numpy(float)
    lr = np.diff(np.log(c0), prepend=np.log(c0[0]))
    lr[0] = 0.0
    noise = rng.permutation(lr[1:])                       # 실제 수익률을 섞어 자기상관만 제거
    noise = np.concatenate([[0.0], noise])
    lr_s = mu * sign + noise
    close = np.exp(np.cumsum(lr_s)) * c0[0]
    # 봉 내부 구조 보존: 실제 (고가/종가, 저가/종가) 비율을 그대로 곱한다
    hi_r = np.clip(d.high.to_numpy(float) / c0, 1.0, None)
    lo_r = np.clip(d.low.to_numpy(float) / c0, None, 1.0)
    return close, close * hi_r, close * lo_r, s


def run(edge_bp: float, *, iters: int, n_env: int, seed: int, log=print) -> dict:
    d, sm, win, S43, cols, _ = P.prepare(G.DEFAULT_FAMILIES)
    if P.ADV_SCALE == "vol":
        P.SIGMA_HAT = P.sigma_hat(d, win["TRAIN"])
    tr_lo, tr_hi = win["TRAIN"]
    close, high, low, s = synth_world(d, edge_bp, horizon=48, seed=seed)
    # 상태 = 실제 43열 + 합성 신호 1열(정답). 정규화는 학습창에서 동결(실제 경로와 같은 규약).
    sc = pd.DataFrame({"timestamp": d.timestamp, "synth": s})
    nx = G.fit_normalizer(sc, ["synth"], tr_lo, tr_hi)
    S = np.hstack([S43, G.build_state(sc, nx)]).astype(np.float32)
    prices = (close, high, low)

    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = P.ActorCritic(S.shape[1]); opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    hist = []
    t0 = time.time()
    for it in range(iters):
        S_list, batches, stat = [], [], []
        for _ in range(n_env):
            lo = int(rng.integers(tr_lo, tr_hi - P.EP_BARS)); hi = lo + P.EP_BARS
            S_win = S[lo:hi]
            logp, v = P.policy_table(model, S_win)
            gym = G.DirectionGym(d, sm, lo, hi, prices=prices)     # 🔴진짜 gym
            res = P.rollout(gym, logp, rng, greedy=False)
            b = P.transitions(res, logp, v, lo, hi)
            if b is not None:
                S_list.append(S_win); batches.append(b)
            # 정답 대비 적중률 -- 정책이 «신호를 따라가는가」를 직접 본다
            tr = res["_trades"]
            if tr:
                acc = float(np.mean([np.sign(t["side"]) == np.sign(s[t["i"]]) for t in tr]))
                stat.append((len(tr), res["net_bp_mean"], acc))
        if not batches:
            hist.append({"it": it, "trades": 0}); continue
        P.ANTI_FLAT_EFF = 0.0
        st = P.ppo_update(model, opt, S_list, batches, ent_coef=ENT, epochs=EPOCHS, mb=MB)
        ev = float(st[3])
        n = float(np.mean([x[0] for x in stat])) if stat else 0.0
        bp = float(np.mean([x[1] for x in stat])) if stat else 0.0
        ac = float(np.mean([x[2] for x in stat])) if stat else 0.0
        hist.append({"it": it, "trades": n, "net_bp": bp, "acc": ac, "ent": float(st[2]), "ev": ev})
        if it % 10 == 0 or it == iters - 1:
            log(f"    it {it:3d} 거래/창 {n:6.0f} · 순 {bp:+7.2f}bp · **신호적중 {100*ac:5.1f}%** "
                f"· ent {st[2]:.3f} · ev {ev:+.3f} · {time.time()-t0:.0f}s", flush=True)
    # 평가: 학습창 밖 창에서 탐욕 정책
    out = {"edge_bp": edge_bp, "seed": seed, "hist": hist}
    for w in ("VAL", "OOS", "TEST"):
        lo, hi = win[w]
        logp, _ = P.policy_table(model, S[lo:hi]); A = logp.argmax(1)
        gym = G.DirectionGym(d, sm, lo, hi, prices=prices)
        res = gym.run(lambda i: (int(A[i - lo]), None))
        tr = res["_trades"]
        acc = float(np.mean([np.sign(t["side"]) == np.sign(s[t["i"]]) for t in tr])) if tr else 0.0
        out[w] = {"trades": len(tr), "net_bp": float(P.metrics(res, d)["net_bp"]),
                  "signal_acc": acc, "mult": float(res["mult"])}
        log(f"  [{w}] 거래 {len(tr):5d} · 순 {out[w]['net_bp']:+7.2f}bp · 신호적중 {100*acc:5.1f}% "
            f"· 계좌 {res['mult']:.3f}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", type=str, default="100,30,10")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--n-env", type=int, default=8)
    ap.add_argument("--seed", type=int, default=P.SEEDS[0])
    ap.add_argument("--credit", choices=["gae", "bandit"], default="gae")
    ap.add_argument("--adv-scale", choices=["none", "vol"], default="none")
    ap.add_argument("--adv-norm", choices=["batch", "std_only", "center_only", "none"], default="batch")
    ap.add_argument("--ortho-init", action="store_true")
    ap.add_argument("--critic-lambda", type=float, default=None)
    ap.add_argument("--ent", type=float, default=0.001)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--mb", type=int, default=8192)
    a = ap.parse_args()
    P.CREDIT = a.credit; P.ADV_SCALE = a.adv_scale
    P.ADV_NORM = a.adv_norm; P.ORTHO_INIT = a.ortho_init; P.CRITIC_LAMBDA = a.critic_lambda
    global ENT, EPOCHS, MB
    ENT, EPOCHS, MB = a.ent, a.epochs, a.mb
    P.OUT.mkdir(parents=True, exist_ok=True)
    logf = open(P.OUT / "log_endtoend_poscontrol.txt", "a")
    def log(s, **kw):
        print(s, **kw); logf.write(str(s) + "\n"); logf.flush()
    log(f"\n=== 끝까지 통과하는 양성대조 · edges {a.edges} · {a.iters}반복 × {a.n_env}창 · 씨드 {a.seed}"
        f" · 신용 {a.credit} · 이득눈금 {a.adv_scale} · 이득정규화 {a.adv_norm}"
        f"{' · 직교초기화' if a.ortho_init else ''} · critic_λ {a.critic_lambda} · ent {a.ent}"
        f" · epochs {a.epochs} · mb {a.mb}")
    log("합성 세계: 외생 AR(1) 신호가 48봉 표류를 정한다. 나머지(gym·비용·손절·희소보상)는 전부 진짜.")
    res = []
    for e in [float(x) for x in a.edges.split(",")]:
        log(f"\n[EDGE {e:g}bp] (왕복 비용 ≈ 6bp)")
        res.append(run(e, iters=a.iters, n_env=a.n_env, seed=a.seed, log=log))
    p = P.OUT / f"report_endtoend_poscontrol_{a.tag}.json" if getattr(a, "tag", None) else P.OUT / (
        f"report_endtoend_poscontrol_{a.credit}_{a.adv_scale}_{a.adv_norm}"
        f"{'_ortho' if a.ortho_init else ''}{'_cl%g' % a.critic_lambda if a.critic_lambda else ''}"
        f"{'_ent%g' % a.ent if a.ent != 0.001 else ''}{'_ep%d' % a.epochs if a.epochs != 4 else ''}.json")
    json.dump(res, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    log(f"\n저장 {p}")
    log("\n판정: EDGE 가 비용보다 충분히 큰데도 신호적중이 50% 근처에 머물면 **배관이 깨진 것**이다.")
    # 🔴판정은 **신호적중**으로 한다. 거래 수로 거르면 «선별적으로 정확한» 정책이 미학습으로 찍힌다
    # (2026-09-14 실제로 그랬다: 적중 100% 인데 거래 7건이라 «미학습」 출력).
    # 거래가 너무 적으면 판정 불가로 따로 표시한다 -- «학습 실패」와 «표본 부족」은 다른 말이다.
    for r in res:
        for w in ("VAL", "OOS", "TEST"):
            v = r[w]; n = v["trades"]
            tag = ("판정불가(표본 부족)" if n < 20 else
                   "✅학습됨" if v["signal_acc"] > 0.60 else
                   "🔴미학습" if v["signal_acc"] < 0.55 else "△애매")
            log(f"  EDGE {r['edge_bp']:>5g}bp [{w:<4}] 신호적중 {100*v['signal_acc']:5.1f}% · "
                f"거래 {n:5d} · 순 {v['net_bp']:+8.2f}bp · 계좌 {v['mult']:.3f}  {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
