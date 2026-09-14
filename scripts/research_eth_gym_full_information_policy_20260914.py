"""**완전정보 문맥 최적화** — RL 을 버리고 세 팔의 보상을 전부 아는 문제로 다시 짠다 (2026-09-14).

사용자 *"밴딧 대신 완전정보 문맥 회귀로 다시 짜줘"*. 근거: arXiv:2402.14740(보상이 단일 종단
스칼라면 가치망·GAE·클리핑은 불필요한 장치) + 2026-09-14 문헌조사 §C.

## 왜 이게 밴딧보다 쉬운가
가격 수용자 가정에서 **매 봉의 세 팔 보상을 전부 계산할 수 있다** — 배포 스택을 그 봉에서 롱으로
한 번, 숏으로 한 번 굴리면 된다. 그러면:
  · 표본이 «진입한 800~1,600건」이 아니라 **결정 전부**가 된다.
  · 탐색·가치망·GAE·클리핑·베이스라인이 전부 사라진다. **그래디언트 분산이 0** 이다.
  · PPO 가 지목당한 실패 경로(배치 이득 정규화·상수 크리틱·홀드 신용 누수·엔트로피 붕괴)가
    **구조적으로 존재하지 않는다**.

## 🔴정직하게 — 결정 규칙은 이미 본 것과 같다
`r_롱 = 이동 − 비용` · `r_숏 = −이동 − 비용` · `r_홀드 = 0` 이므로
argmax 는 «|예측 이동| > 비용 이면 그 부호로 베팅」으로 환원된다 = 임계 쌍둥이(§5)와 **같은 규칙**.
**새로운 것은 목적함수다**: 이동폭 제곱오차가 아니라 **기대 보상 −Σ_a π(a)·r_a** 를 직접 최대화한다.
틀렸을 때의 경제적 손실이 곧 가중치가 된다(cost-sensitive). 그 차이가 이 판이 재는 전부다.

## 세 팔 (전부 같은 gym·같은 배포 스택·같은 비용)
  A0 홀드 = 0 · A1 롱 = trade_outcome(i, LONG) · A2 숏 = trade_outcome(i, SHORT)
보상은 **크기 없는 건당 수익 r**(= fill_move − cost)로 둔다. 레버리지를 곱하면 사이징 모델이
어느 봉을 키울지까지 목적함수에 섞여 «방향을 배웠나」가 흐려진다. `--reward account` 로 바꿀 수 있다.

## 판정 (사전등록 — §12·§13 과 같은 규칙)
gym 안에서 탐욕 정책으로 돌려 **거래당 순bp · 적중률 vs 손익분기 · Δ(vs 같은측면 무작위)** 를 낸다.
통과선: 3창 중 ≥2 에서 순bp > 0 **그리고** 일군집 t ≥ 2. 대조군: 홀드 · 항상롱/숏 · 같은측면 무작위 ·
역방향 · 임계 쌍둥이(제곱오차 목적) — **쌍둥이를 못 이기면 목적함수 교체의 값이 0** 이다.
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
import torch
import torch.nn as nn
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_rl_gym_direction_ppo_20260914 as P  # noqa: E402
import rl_gym_direction_env_20260914 as G  # noqa: E402

LAB = P.OUT / "full_information_labels.npz"
EV = ("VAL", "OOS", "TEST")


# ── 세 팔 보상 ──────────────────────────────────────────────────────────────
def arm_rewards(d, sm, idx: np.ndarray, cost=None, log=print) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """각 봉에서 롱/숏을 각각 한 번씩 배포 스택으로 굴린 건당 수익 r. (idx, r_long, r_short)."""
    keep, rl, rs = [], [], []
    t0 = time.time()
    for k, i in enumerate(idx):
        if not sm["ok"][i]:
            continue
        a = P.trade_outcome(d, sm, int(i), 1, cost)
        b = P.trade_outcome(d, sm, int(i), 2, cost)
        if a is None or b is None:
            continue
        keep.append(i); rl.append(a); rs.append(b)
        if k % 20000 == 0 and k:
            log(f"    {k:,}/{len(idx):,} · {time.time()-t0:.0f}s", flush=True)
    return np.array(keep), np.array(rl), np.array(rs)


def build_labels(d, sm, win, stride_train: int, stride_eval: int, log=print) -> dict:
    out = {}
    for w in ("TRAIN",) + EV:
        lo, hi = win[w]
        s = stride_train if w == "TRAIN" else stride_eval
        log(f"  [{w}] 세 팔 보상 …", flush=True)
        i, rl, rs = arm_rewards(d, sm, np.arange(lo, hi, s), log=log)
        out[w] = (i, rl, rs)
        mv = 0.5e4 * (rl - rs); cst = -0.5e4 * (rl + rs)
        log(f"    n {len(i):,} · 이동 평균 {mv.mean():+.2f}bp SD {mv.std():.1f} · 비용 평균 {cst.mean():.2f}bp"
            f" · 베팅가치 있는 봉 {100*np.mean(np.maximum(rl, rs) > 0):.1f}%", flush=True)
    ts = d.timestamp.to_numpy().astype("datetime64[ns]").astype(np.int64)
    np.savez(LAB, names=np.array(list(out)), frame_len=np.int64(len(d)),
             **{f"{k}_idx": v[0] for k, v in out.items()},
             **{f"{k}_ts": ts[v[0]] for k, v in out.items()},
             **{f"{k}_rl": v[1] for k, v in out.items()},
             **{f"{k}_rs": v[2] for k, v in out.items()})
    return out


def load_labels(d) -> dict:
    """타임스탬프로 현재 프레임에 맞춘다(머신마다 klines 길이가 다르다)."""
    z = np.load(LAB, allow_pickle=True)
    ts = d.timestamp.to_numpy().astype("datetime64[ns]").astype(np.int64)
    order = np.argsort(ts); out = {}
    for k in [str(x) for x in z["names"]]:
        want = z[f"{k}_ts"]
        j = order[np.searchsorted(ts[order], want)]
        ok = (j < len(ts)) & (ts[np.clip(j, 0, len(ts) - 1)] == want)
        out[k] = (j[ok], z[f"{k}_rl"][ok], z[f"{k}_rs"][ok])
    return out


# ── 정책 ────────────────────────────────────────────────────────────────────
class Policy(nn.Module):
    """RL 팔과 **같은 몸통**(2×64 tanh) — 비교가 목적함수 차이만 남게."""

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 3))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.net[-1].weight, 0.01)

    def forward(self, x):
        return self.net(x)


def fit_policy(X: np.ndarray, R: np.ndarray, *, seed: int, epochs: int, lr: float,
               ent: float, wd: float, log=print) -> Policy:
    """**기대 보상 최대화**: loss = −mean(Σ_a π(a|s)·r_a). 표본추출 없음 ⇒ 분산 0."""
    torch.manual_seed(seed)
    m = Policy(X.shape[1]); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.as_tensor(X, dtype=torch.float32); Rt = torch.as_tensor(R, dtype=torch.float32)
    n = len(Xt); mb = 4096
    for e in range(epochs):
        perm = torch.randperm(n); tot = 0.0
        for s in range(0, n, mb):
            j = perm[s:s + mb]
            lsm = torch.log_softmax(m(Xt[j]), -1); p = lsm.exp()
            exp_r = (p * Rt[j]).sum(1).mean()
            ent_t = -(p * lsm).sum(1).mean()
            loss = -exp_r - ent * ent_t
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            tot += float(exp_r) * len(j)
        if e % 5 == 0 or e == epochs - 1:
            with torch.no_grad():
                pp = torch.softmax(m(Xt), -1)
                share = pp.mean(0).numpy()
            log(f"    ep {e:3d} 기대보상 {1e4*tot/n:+7.3f}bp · 행동비중 홀드 {share[0]:.2f}/롱 {share[1]:.2f}/숏 {share[2]:.2f}",
                flush=True)
    return m


@torch.no_grad()
def act(m: Policy, X: np.ndarray) -> np.ndarray:
    return torch.softmax(m(torch.as_tensor(X, dtype=torch.float32)), -1).numpy().argmax(1)


# ── 평가 ────────────────────────────────────────────────────────────────────
def gym_eval(d, sm, win, w, A_at_idx, idx, cost=None) -> dict:
    """결정 봉에만 행동을 심고 **진짜 gym** 을 돌린다(원시 결과를 그대로 돌려준다)."""
    lo, hi = win[w]
    A = np.zeros(hi - lo, int); A[idx - lo] = A_at_idx
    gym = G.DirectionGym(d, sm, lo, hi, **(cost or {}))
    return gym.run(lambda i: (int(A[i - lo]), None))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride-train", type=int, default=4)
    ap.add_argument("--stride-eval", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--families", type=str, default=",".join(G.DEFAULT_FAMILIES))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--tag", type=str, default="fullinfo")
    a = ap.parse_args()
    P.OUT.mkdir(parents=True, exist_ok=True)
    logf = open(P.OUT / f"log_{a.tag}.txt", "a")
    def log(s, **kw):
        print(s, **kw); logf.write(str(s) + "\n"); logf.flush()

    d, sm, win, S, cols, _ = P.prepare(tuple(a.families.split(",")))
    log(f"\n=== {a.tag} · 완전정보 문맥 최적화 · 상태 {S.shape[1]}D · 씨드 {P.SEEDS[:a.seeds]}")
    if LAB.exists() and not a.rebuild:
        lab = load_labels(d)
    else:
        log("세 팔 보상 생성 …")
        build_labels(d, sm, win, a.stride_train, a.stride_eval, log=log); lab = load_labels(d)
    for w in ("TRAIN",) + EV:
        i, rl, rs = lab[w]
        log(f"  {w:<5} n {len(i):,} · 베팅가치 {100*np.mean(np.maximum(rl, rs) > 0):.1f}%")

    ti, trl, trs = lab["TRAIN"]
    Rtr = np.stack([np.zeros_like(trl), trl, trs], 1)
    # ⭐오라클 상한 -- 세 팔을 **알고** 고르면 얼마인가. 학습 가능한 최대치의 절대 천장이다.
    for w in EV:
        i, rl, rs = lab[w]
        R = np.stack([np.zeros_like(rl), rl, rs], 1)
        orc = gym_eval(d, sm, win, w, R.argmax(1), i)
        mo = P.metrics(orc, d)
        log(f"  [오라클 {w}] 거래 {mo['trades']:5d} · 순 {mo['net_bp']:+8.2f}bp · 적중 {100*mo['hit_rate']:.1f}%"
            f" · 계좌 {orc['mult']:.3f}")

    rep = {"tag": a.tag, "dim": int(S.shape[1]), "arms": ["hold", "long", "short"], "seeds": P.SEEDS[:a.seeds],
           "n_train": int(len(ti)), "runs": [], "controls": {}}
    per_seed = {w: [] for w in EV}
    for sd in P.SEEDS[:a.seeds]:
        log(f"\n[씨드 {sd}]")
        m = fit_policy(S[ti], Rtr, seed=sd, epochs=a.epochs, lr=a.lr, ent=a.ent, wd=a.wd, log=log)
        row = {"seed": sd}
        for w in EV:
            i, rl, rs = lab[w]
            A = act(m, S[i])
            r = gym_eval(d, sm, win, w, A, i)
            mt = P.metrics(r, d)
            # 같은 측면 무작위: 행동 빈도는 그대로, 시점만 뒤섞는다
            ssr = gym_eval(d, sm, win, w, np.random.default_rng(sd).permutation(A), i)
            rev = gym_eval(d, sm, win, w, np.where(A == 1, 2, np.where(A == 2, 1, 0)), i)
            mssr, mrev = P.metrics(ssr, d), P.metrics(rev, d)
            row[w] = {"trades": mt["trades"], "net_bp": mt["net_bp"], "hit_rate": mt["hit_rate"],
                      "t_day": mt["t_day"], "side_share": mt["side_share"], "mult": float(r["mult"]),
                      "hold_share": float(np.mean(A == 0)),
                      "ssr_net_bp": mssr["net_bp"], "rev_net_bp": mrev["net_bp"],
                      "delta_ssr": mt["net_bp"] - mssr["net_bp"]}
            per_seed[w].append(row[w])
            v = row[w]
            log(f"  [{w}] 거래 {v['trades']:5d}(홀드 {100*v['hold_share']:.0f}%) · 순 {v['net_bp']:+7.2f}bp"
                f" · 적중 {100*v['hit_rate']:.1f}% · 일군집 t {v['t_day']:+.2f} · 측면 {v['side_share']:.2f}"
                f" · Δ(vs 같은측면무작위) {v['delta_ssr']:+.2f} · 역방향 {v['rev_net_bp']:+.2f}")
        rep["runs"].append(row)

    log("\n=== 씨드 평균 ± SE")
    gates = {}
    for w in EV:
        rows = per_seed[w]
        agg = {k: float(np.mean([r[k] for r in rows])) for k in
               ("trades", "net_bp", "hit_rate", "t_day", "side_share", "delta_ssr", "rev_net_bp", "hold_share")}
        se = lambda k: float(np.std([r[k] for r in rows], ddof=1) / math.sqrt(len(rows))) if len(rows) > 1 else float("nan")
        agg["net_bp_se"] = se("net_bp"); agg["delta_ssr_se"] = se("delta_ssr")
        rep["controls"][w] = agg
        gates[w] = bool(agg["net_bp"] > 0 and agg["t_day"] >= 2)
        log(f"  [{w}] 거래 {agg['trades']:.0f}(홀드 {100*agg['hold_share']:.0f}%) · 순 {agg['net_bp']:+7.2f} ± {agg['net_bp_se']:.2f}bp"
            f" · 적중 {100*agg['hit_rate']:.1f}% · t {agg['t_day']:+.2f} · Δ(ssr) {agg['delta_ssr']:+.2f} ± {agg['delta_ssr_se']:.2f}"
            f" · 역방향 {agg['rev_net_bp']:+.2f}")
    rep["gates"] = gates
    npass = sum(gates.values())
    log(f"\n판정(3창 중 ≥2 에서 순bp>0 & 일군집 t≥2): {gates} → {'⭐통과' if npass >= 2 else '탈락'}")
    p = P.OUT / f"report_{a.tag}.json"
    json.dump(rep, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    log(f"저장 {p}")
    # 자체점검
    i, rl, rs = lab["VAL"]
    mv = 0.5e4 * (rl - rs); cst = -0.5e4 * (rl + rs)
    assert 2.0 < cst.mean() < 20.0, f"두 팔 평균이 −비용이어야 한다: {cst.mean():.2f}bp"
    assert abs(mv.mean()) < 20.0, f"이동 평균이 편향됐다: {mv.mean():.2f}bp"
    log(f"확인: 두 팔 평균 = −비용 {cst.mean():.2f}bp · 이동 평균 {mv.mean():+.2f}bp (편향 없음)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
