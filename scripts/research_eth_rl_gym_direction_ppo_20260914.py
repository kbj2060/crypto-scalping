"""**방향 RL** — 호메로스 gym 위에서 {홀드, 롱, 숏}을 PPO 로 학습하고 사전등록 게이트로 판정 (2026-09-14).

설계 `docs/eth_rl_gym_control_agent_design_20260914.md` v2 §4~§7. 환경은 `rl_gym_direction_env_20260914`.

## 무엇이 09-12 DSAC 와 다른가
빈 슬롯에서만 결정(semi-MDP) · 보상 = 청산 시 ln(순자산비) 한 항 · 연속 재료 입력 · 홀드 벌점 0 ·
학습 탈드리프트/평가 실제 가격 · 드리프트 방어는 액터 손실 한 곳(direction_reg).

## 게이트 (사전등록, 설계 §7)
G1 양성대조(변동성확장 타깃) OOS AUC ≥ .65 · G2 3창 중 ≥2 에서 거래당 순bp>0 & 일군집 t≥2 &
로그성장>0 · G3 지도학습 쌍둥이 대비 +2SE · G4 역방향<0 & 같은측면 무작위 대비 +2SE ·
G5 2022~23 부호 일치 · G6 씨드 5 부호일치 + DSR · G7 붕괴(거래<10 · 측면>0.85) · G8 누수 신호 ·
G9 테이커 10bp 유지.

ponytail: PPO 를 직접 짠다(~80줄). 롤아웃은 정책 확률을 창 전체에 한 번 계산해 두고 gym 의
numpy 루프로 돈다(정책이 롤아웃 중 고정 · 관측에 포지션 상태 없음). GPU 안 씀.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm as _norm

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import rl_gym_direction_env_20260914 as G  # noqa: E402
import research_fresh_forward_random_entry_stack_20260914 as H  # noqa: E402

OUT = ROOT / "data/research/eth_rl_gym_direction_20260914"
SM_CACHE = ROOT / "data/research/eth_rl_gym_safe_mae_cache_20260914.pkl"
WINDOWS = {"TRAIN": ("2024-03-01", "2025-08-31"), "VAL": ("2025-09-01", "2025-12-31"),
           "OOS": ("2026-01-01", "2026-03-31"), "TEST": ("2026-04-01", "2026-08-20"),
           "BACK22_23": ("2022-01-01", "2023-12-31")}
EVAL_WINDOWS = ("VAL", "OOS", "TEST")
EP_BARS = 288 * 90                      # 90일 창
GAMMA_BAR = 0.9995                      # 봉당 할인(반감기 ≈ 4.8일). 설계 γ=1 에서 한 발 물러난 값 -- 분산 때문
LAMBDA = 0.95
# 🔴보상 눈금(2026-09-14 1차 실행에서 발견): 보상이 ln(순자산비)라 거래당 ±0.005 인데 엔트로피
# 계수 0.01·0.003 이 결정마다 붙어 «무작위로 거래하는 값»이 «거래 손실»보다 컸다 -- 씨드 10개 전부
# 엔트로피 0.6~1.0 나트에 머물며 창당 수백~천 건을 거래했다. 보상을 100 배(퍼센트 로그) 하고
# 엔트로피 격자를 {0.001, 0} 으로 내린다. 양성대조는 보상 0.01 눈금에서도 .80 이 나왔었다.
REWARD_SCALE = 100.0
DIRECTION_REG = 0.20
# 진짜 다양한 씨드(고정 간격 증가 아님) -- CLAUDE.md 시드 다양성 게이트
SEEDS = [int(x) for x in np.random.default_rng(20260914).integers(1, 1_000_000, size=5)]
torch.set_num_threads(4)


# ── 데이터 준비 ─────────────────────────────────────────────────────────────
def prepare(families):
    d = G.load_frame()
    sm = pickle.load(open(SM_CACHE, "rb"))
    ts = d.timestamp.to_numpy()
    win = {}
    for k, (a, b) in WINDOWS.items():
        lo = int(np.searchsorted(ts, np.datetime64(a))); hi = int(np.searchsorted(ts, np.datetime64(b + "T23:59:59")))
        win[k] = (lo, hi)
    cols = G.state_columns(families)
    norm = G.fit_normalizer(d, cols, *win["TRAIN"])
    S = G.build_state(d, norm)
    # 양성대조 라벨: 앞으로 48봉 실현변동성이 학습창 상위 1/3 인가 (t+1..t+48, 봉 t 이후만)
    lr = np.diff(np.log(d.close.to_numpy(float)), prepend=np.nan)
    fwd = pd.Series(lr[::-1]).rolling(48).std().to_numpy()[::-1]     # std of lr[t..t+47]
    fwd = np.roll(fwd, -1)                                             # lr[t+1..t+48]
    thr = np.nanpercentile(fwd[win["TRAIN"][0]:win["TRAIN"][1]], 66.7)
    volexp = (fwd >= thr).astype(np.float32)
    return d, sm, win, S, cols, volexp


def sigma_hat(d, train_win) -> np.ndarray:
    """봉별 예측 변동성(학습창 평균 1로 정규화, [0.3,3] 클립).

    r ≈ L·(움직임 − 비용) 이고 |움직임| 의 눈금은 실현변동성이 거의 다 설명한다. 이 저장소가
    방향은 못 맞혀도 **크기는 맞힌다**(AUC .82). 그 하나를 기울기 가중에 쓴다."""
    v = d["rv48"].to_numpy(float) if "rv48" in d.columns else d["atr_pct"].to_numpy(float)
    lo, hi = train_win
    base = float(np.nanmedian(v[lo:hi]))
    s = np.nan_to_num(v / max(base, 1e-12), nan=1.0, posinf=1.0, neginf=1.0)
    return np.clip(s, 0.3, 3.0).astype(np.float64)


def load_labels(d):
    """라벨을 **타임스탬프로** 프레임에 맞춘다.

    🔴`direction_labels.npz` 는 원래 정수 위치(idx)만 담았다. 위치는 **그 프레임에만** 유효한데,
    klines 파일은 머신마다 길이가 다르다(2026-09-14 서버 이관: 로컬 62MB / 서버 33MB).
    위치를 그대로 쓰면 **다른 봉을 가리킨 채 조용히** 틀린 결과가 나온다 -- 오류도 안 난다.
    그래서 `*_ts`(int64 ns)가 있으면 그걸로 현재 프레임의 위치를 다시 찾는다.
    없으면 프레임 길이가 저장 당시와 같은지 단언한다.
    """
    z = np.load(OUT / "direction_labels.npz", allow_pickle=True)
    ts = d.timestamp.to_numpy().astype("datetime64[ns]").astype(np.int64)
    order = np.argsort(ts)
    out = {}
    for k in [str(x) for x in z["names"]]:
        y = z[f"{k}_y"]
        if f"{k}_ts" in z.files:
            want = z[f"{k}_ts"]
            j = order[np.searchsorted(ts[order], want)]
            ok = (j < len(ts)) & (ts[j] == want)
            out[k] = (j[ok], y[ok])
            if ok.sum() < len(want):
                print(f"  ⚠️{k}: 라벨 {len(want):,} 중 {int(ok.sum()):,} 만 이 프레임에 있다", flush=True)
        else:
            n = int(z["frame_len"]) if "frame_len" in z.files else len(ts)
            assert len(ts) == n, (f"라벨이 위치 기반인데 프레임 길이가 다르다({len(ts)} != {n}) -- "
                                  "add_label_timestamps 로 *_ts 를 넣어라")
            out[k] = (z[f"{k}_idx"], y)
    return out


# ── 모델 ────────────────────────────────────────────────────────────────────
def _ortho(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain); nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, 3)
        self.v = nn.Linear(hidden, 1)
        if ORTHO_INIT:
            # 몸통 gain=√2(tanh 관행) · 정책 머리 **0.01**(초기 로짓을 거의 균일하게) · 가치 머리 1.0
            for m_ in self.body:
                if isinstance(m_, nn.Linear):
                    _ortho(m_, np.sqrt(2))
            _ortho(self.pi, 0.01); _ortho(self.v, 1.0)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


FORCED = False          # --reward-mode forced: 홀드를 행동에서 뺀다(빈 슬롯마다 롱/숏 강제)
HOLD_PENALTY = 0.0      # --reward-mode holdpen: 홀드 결정마다 −λ (보상 눈금 기준) = DSAC 의 r5_idle
# --reward-mode antiflat: 사용자 DSAC(`ensemble/train_rl_dsac_agent.py` anti_flat_lambda=0.08,
# min_abs=0.18, anneal) 의 **액터 손실** 항을 이식. 거래확률 평균이 바닥 아래면 λ·relu(바닥−P거래),
# λ 는 반복에 따라 0 으로 선형 감쇠(DSAC anti_flat_anneal_updates). 보상이 아니라 액터에 걸리므로
# 크리틱이 배워야 하는 간접 경로가 없다(09-12 정정 원칙과 같은 자리).
ANTI_FLAT_LAMBDA = 0.0
ANTI_FLAT_MIN = 0.05
ANTI_FLAT_EFF = 0.0     # 학습 루프가 반복마다 갱신
# ── 2026-09-14 구조 수정 2건 (사용자: "홀드 신용 문제랑 상수 가치함수 둘 다 고쳐서") ──────
# CREDIT="bandit": **GAE 사슬을 끊는다.** 이 문제는 «결정 → 한 번 관측 → 다시 플랫」이라 결정 간
#   결합이 슬롯 점유와 복리뿐이다. GAE 를 걸면 보상 0 인 홀드가 λ 가중으로 **자기가 만들지 않은**
#   뒤따르는 거래의 공과를 받는다(유효 신용 지평 ≈20결정, 진입 간격 ≈17결정 -- 거의 모든 홀드가
#   다음 거래를 떠안는다). 밴딧이면 이득 = r − V(s) 로 그 경로가 사라진다.
# ADV_SCALE="vol": **이득을 상태별 예측 변동성으로 나눈다.** 가치함수가 «상수가 최적」인 이유는
#   결과가 예측 불가라서인데, **분산의 출처(움직임 크기)는 이 저장소가 유일하게 잘 맞히는 것**이다
#   (크기 AUC .82). 고변동 봉의 거대한 |r| 이 기울기를 지배하던 것을 제거한다.
#   ⚠️이건 보상을 바꾸는 게 아니라 **기울기 가중**만 바꾼다 -- 판정 지표(순bp)는 원래대로다.
# 🔴ADV_NORM -- 2026-09-14 문헌조사가 **1순위 원인**으로 지목한 자리(arXiv:2508.08221 · 2601.08521).
#   기존: ADV = (ADV − 배치평균) / 배치표준편차.
#   우리 분포는 결정의 95% 가 홀드(이득 ≈ 0)다. 비용 때문에 배치평균이 **음수**면, 평균을 빼는
#   순간 **모든 홀드가 상태와 무관하게 균일한 양의 이득**을 받는다 → 상태 의존성 없는 홀드 붕괴.
#   같은 논문: 「보상 분포가 한 점에 몰리면 std 로 나누지 말라」(작은 std 가 기울기를 과증폭).
#   batch = 기존 · std_only = 평균 안 뺌 · center_only = std 로 안 나눔 · none = 원값 그대로
ADV_NORM = "batch"
# 🔴CRITIC_LAMBDA -- VC-PPO(arXiv:2503.01491): 희소 **종단** 보상에서 λ<1 은 그 한 개의 보상을
#   결정 시퀀스 뒤로 감쇠시켜 크리틱에 편향을 남긴다. 액터는 λ=0.95 로 두고 **크리틱만 λ=1**.
#   우리 실측 설명분산이 +0.001 이라(크리틱이 상수를 학습 중) 정확히 겨냥되는 자리다.
CRITIC_LAMBDA = None    # None = 액터와 같은 LAMBDA
# 🔴직교 초기화 + 정책 최종층 gain 0.01 (arXiv:2006.05990 · ICLR Blog Track 2022). 기본 PyTorch
#   초기화는 정책 로짓을 크게 시작시켜 초기에 한 행동으로 쏠리게 만든다.
ORTHO_INIT = False
CREDIT = "gae"          # gae | bandit
ADV_SCALE = "none"      # none | vol
SIGMA_HAT = None        # 봉별 예측 변동성(평균 1로 정규화). ADV_SCALE="vol" 일 때만 쓴다.


@torch.no_grad()
def policy_table(model, S_win: np.ndarray):
    logits, v = model(torch.as_tensor(S_win))
    if FORCED:
        logits = logits.clone(); logits[:, 0] = -1e9
    return torch.log_softmax(logits, -1).numpy(), v.numpy()


def rollout(gym: G.DirectionGym, logp: np.ndarray, rng, *, greedy: bool) -> dict:
    """logp 는 gym 창 [lo,hi) 에 대한 (N,3) 로그확률. 봉마다 행동을 미리 뽑아 둔다."""
    p = np.exp(logp)
    if greedy:
        A = p.argmax(1)
    else:
        u = rng.random(len(p))
        A = (u > p[:, 0]).astype(int) + (u > p[:, 0] + p[:, 1]).astype(int)
    lo = gym.lo
    out = gym.run(lambda i: (int(A[i - lo]), None))
    out["_A"] = A
    return out


# ── PPO ─────────────────────────────────────────────────────────────────────
def transitions(res: dict, logp: np.ndarray, v: np.ndarray, lo: int, hi: int):
    dec = res["_decisions"]
    if len(dec) == 0:
        return None
    i = dec[:, 0].astype(int); a = dec[:, 1].astype(int); r = dec[:, 2] * REWARD_SCALE; nx = dec[:, 3].astype(int)
    if HOLD_PENALTY > 0:
        r = r - HOLD_PENALTY * (a == 0)
    vi = v[i - lo]
    vnext = np.where((nx >= 0) & (nx < hi), v[np.clip(nx, lo, hi - 1) - lo], 0.0)
    dt = np.where(nx >= 0, nx - i, 1).astype(float)
    disc = GAMMA_BAR ** dt
    def _gae(lam: float) -> np.ndarray:
        delta = r + disc * vnext - vi
        out = np.zeros_like(delta); g = 0.0
        for k in range(len(delta) - 1, -1, -1):
            g = delta[k] + disc[k] * lam * g
            out[k] = g
        return out
    if CREDIT == "bandit":
        # 밴딧: 각 결정은 자기 보상만 책임진다. 홀드는 r=0 이므로 이득 = −V(s).
        adv = r - vi
        ret = r
    else:
        adv = _gae(LAMBDA)
        # 크리틱 목표만 다른 λ 로 따로 만든다(액터의 이득은 그대로) -- VC-PPO
        ret = (_gae(CRITIC_LAMBDA) + vi) if CRITIC_LAMBDA is not None else (adv + vi)
    if ADV_SCALE == "vol" and SIGMA_HAT is not None:
        adv = adv / SIGMA_HAT[i]
    return {"idx": i - lo, "a": a, "logp": logp[i - lo, a], "adv": adv, "ret": ret}


def ppo_update(model, opt, S_win_list, batches, *, ent_coef: float, epochs: int = 4, mb: int = 8192,
               clip: float = 0.2):
    X = torch.cat([torch.as_tensor(S[b["idx"]]) for S, b in zip(S_win_list, batches)])
    A = torch.as_tensor(np.concatenate([b["a"] for b in batches]))
    LP = torch.as_tensor(np.concatenate([b["logp"] for b in batches]), dtype=torch.float32)
    ADV = torch.as_tensor(np.concatenate([b["adv"] for b in batches]), dtype=torch.float32)
    RET = torch.as_tensor(np.concatenate([b["ret"] for b in batches]), dtype=torch.float32)
    if ADV_NORM == "batch":
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
    elif ADV_NORM == "std_only":
        ADV = ADV / (ADV.std() + 1e-8)
    elif ADV_NORM == "center_only":
        ADV = ADV - ADV.mean()
    n = len(X); stats = []; ev_ = []
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, mb):
            j = perm[s:s + mb]
            logits, v = model(X[j])
            lsm = torch.log_softmax(logits, -1)
            lp = lsm.gather(1, A[j, None]).squeeze(1)
            ratio = torch.exp(lp - LP[j])
            pg = -torch.min(ratio * ADV[j], torch.clamp(ratio, 1 - clip, 1 + clip) * ADV[j]).mean()
            vl = 0.5 * ((v - RET[j]) ** 2).mean()
            p = lsm.exp()
            ent = -(p * lsm).sum(1).mean()
            # 드리프트 방어 -- **액터 손실 한 곳**: 배치 평균 (P롱 − P숏)² (DSAC 정정 원칙)
            dreg = ((p[:, 1] - p[:, 2]).mean()) ** 2
            loss = pg + 0.5 * vl - ent_coef * ent + DIRECTION_REG * dreg
            with torch.no_grad():
                vr = RET[j].var()
                ev_.append(float(1.0 - (RET[j] - v).var() / vr) if vr > 1e-12 else 0.0)
            if ANTI_FLAT_EFF > 0:
                loss = loss + ANTI_FLAT_EFF * torch.relu(torch.tensor(ANTI_FLAT_MIN) - p[:, 1:].sum(1).mean())
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
            stats.append((pg.item(), vl.item(), ent.item()))
    m = np.mean(stats, axis=0)
    # ⭐**설명분산** -- 크리틱이 실제로 배우는지 보이는 유일한 숫자. 0 근처면 «상수를 학습 중」이다.
    return np.append(m, float(np.mean(ev_)) if ev_ else 0.0)


def train(d, sm, S, win, *, seed: int, iters: int, ent_coef: float, lr: float, n_env: int,
          positive: np.ndarray | None = None, cost=None, log=print, ent_anneal: float | None = None,
          train_cost=None):
    """train_cost: 학습 환경에만 쓰는 비용(gross 모드 = 수수료 0). 평가는 항상 cost(실제)."""
    """positive 가 주어지면 양성대조: 보상 = 롱이면 (2·volexp−1), 숏이면 그 반대, 홀드 0."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = ActorCritic(S.shape[1]); opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr_lo, tr_hi = win["TRAIN"]
    prices = G.dedrift(d, tr_lo, tr_hi)
    t0 = time.time()
    for it in range(iters):
        S_list, batches, tr_n, tr_bp = [], [], [], []
        for _ in range(n_env):
            lo = int(rng.integers(tr_lo, tr_hi - EP_BARS)); hi = lo + EP_BARS
            S_win = S[lo:hi]; logp, v = policy_table(model, S_win)
            if positive is not None:
                p = np.exp(logp); u = rng.random(len(p))
                A = (u > p[:, 0]).astype(int) + (u > p[:, 0] + p[:, 1]).astype(int)
                y = 2 * positive[lo:hi] - 1
                r = np.where(A == 1, y, np.where(A == 2, -y, 0.0)) * 0.01 / REWARD_SCALE
                i = np.arange(len(A)); nx = np.append(i[1:] + lo, -1)
                res = {"_decisions": np.column_stack([i + lo, A, r, nx]).astype(float)}
            else:
                gym = G.DirectionGym(d, sm, lo, hi, prices=prices, **((train_cost if train_cost is not None else cost) or {}))
                res = rollout(gym, logp, rng, greedy=False)
                tr_n.append(res["trades"]); tr_bp.append(res["net_bp_mean"])
            b = transitions(res, logp, v, lo, hi)
            if b is not None:
                S_list.append(S_win); batches.append(b)
        # 🔴엔트로피 감쇠(2026-09-14 2차에서 필요해짐): 고정 0.001 은 it 50 에서 엔트로피가
        # 0.008 로 죽어 창당 1~3 거래로 **조기 수렴**했다. «엣지가 없어서 관망」과 «탐색이 먼저
        # 죽어서 관망」이 구분이 안 된다. ent_anneal 을 주면 그 값에서 ent_coef 로 선형 감쇠한다.
        ec = ent_coef if ent_anneal is None else ent_anneal + (ent_coef - ent_anneal) * (it / max(iters - 1, 1))
        global ANTI_FLAT_EFF
        ANTI_FLAT_EFF = ANTI_FLAT_LAMBDA * max(0.0, 1.0 - it / max(iters - 1, 1))
        st = ppo_update(model, opt, S_list, batches, ent_coef=ec)
        if it % 10 == 0 or it == iters - 1:
            log(f"    it {it:3d} pg {st[0]:+.4f} v {st[1]:.5f} ent {st[2]:.3f} **ev {st[3]:+.3f}**"
                + (f" · 거래/창 {np.mean(tr_n):.0f} 순bp {np.mean(tr_bp):+.2f}" if tr_n else "")
                + f" · {time.time()-t0:.0f}s")
    return model


# ── 평가 ─────────────────────────────────────────────────────────────────────
def metrics(res: dict, d: pd.DataFrame) -> dict:
    tr = res["_trades"]; n = len(tr)
    r = np.array([t["r"] for t in tr]) * 1e4 if n else np.zeros(0)
    if n >= 2:
        days = pd.to_datetime(d.timestamp.to_numpy()[[t["i"] for t in tr]]).normalize()
        dm = pd.Series(r).groupby(days.to_numpy()).mean()
        t_day = float(dm.mean() / (dm.std(ddof=1) / math.sqrt(len(dm)))) if len(dm) > 2 and dm.std(ddof=1) > 0 else 0.0
        t_tr = float(r.mean() / (r.std(ddof=1) / math.sqrt(n))) if r.std(ddof=1) > 0 else 0.0
    else:
        t_day = t_tr = 0.0
    longs = sum(1 for t in tr if t["side"] > 0)
    logm = math.log(max(res["mult"], 1e-12))
    gross = np.array([t["r"] + t["cost_bp"] / 1e4 for t in tr if "cost_bp" in t]) * 1e4
    hit = float((gross > 0).mean()) if len(gross) else 0.0
    return {"trades": n, "net_bp": float(r.mean()) if n else 0.0, "t_trade": t_tr, "t_day": t_day,
            "gross_bp": float(gross.mean()) if len(gross) else 0.0, "hit_rate": hit,
            "log_mult": logm, "expo_time_x": res["expo_time_x"],
            "growth_per_expo": logm / res["expo_time_x"] if res["expo_time_x"] > 0 else 0.0,
            "mdd": res["mdd"], "stops": res["stops"], "side_share": max(longs, n - longs) / n if n else 0.0,
            "ruin": bool(res["ruin"])}


def eval_policy(model, d, sm, S, win, wname, *, A_override=None, cost=None) -> tuple[dict, np.ndarray]:
    lo, hi = win[wname]
    gym = G.DirectionGym(d, sm, lo, hi, **(cost or {}))
    if A_override is None:
        logp, _ = policy_table(model, S[lo:hi]); A = logp.argmax(1)
    else:
        A = A_override
    res = gym.run(lambda i: (int(A[i - lo]), None))
    return metrics(res, d), A


def controls(model, d, sm, S, win, wname, A_policy, rng, cost=None) -> dict:
    lo, hi = win[wname]; N = hi - lo; out = {}
    out["hold"] = eval_policy(model, d, sm, S, win, wname, A_override=np.zeros(N, int), cost=cost)[0]
    out["always_long"] = eval_policy(model, d, sm, S, win, wname, A_override=np.ones(N, int), cost=cost)[0]
    out["always_short"] = eval_policy(model, d, sm, S, win, wname, A_override=np.full(N, 2), cost=cost)[0]
    rev = np.where(A_policy == 1, 2, np.where(A_policy == 2, 1, 0))
    out["reversed"] = eval_policy(model, d, sm, S, win, wname, A_override=rev, cost=cost)[0]
    # 같은 측면 무작위: 정책의 행동 빈도는 그대로, 시점만 뒤섞는다
    out["same_side_random"] = eval_policy(model, d, sm, S, win, wname, A_override=rng.permutation(A_policy), cost=cost)[0]
    return out


def one_feature_rule(d, win, wname, q_lo, q_hi) -> np.ndarray:
    """직전 1h 수익(ret12) 하위 십분위 → 롱, 상위 십분위 → 숏. 분위 경계는 학습창."""
    lo, hi = win[wname]; x = d["ret12"].to_numpy(float)[lo:hi]
    return np.where(x <= q_lo, 1, np.where(x >= q_hi, 2, 0))


# ── 지도학습 쌍둥이 ───────────────────────────────────────────────────────────
def trade_outcome(d, sm, i: int, side: int, cost=None) -> float | None:
    """봉 i 에서 측면 side 로 한 번만 들어갔을 때의 size-free 순수익(r). 배포 스택 그대로."""
    gym = G.DirectionGym(d, sm, i, i + 1, **(cost or {}))
    hb = gym.hold_bars(i, "LONG" if side == 1 else "SHORT")
    if hb is None:
        return None
    gym.hi = i + hb + 1
    res = gym.run(lambda j: ((side if j == i else 0), None))
    return res["_trades"][0]["r"] if res["_trades"] else None


def twin_labels(d, sm, win, stride: int, cost=None) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = win["TRAIN"]; idx = np.arange(lo, hi, stride); ys = []
    for i in idx:
        if not sm["ok"][i]:
            ys.append(np.nan); continue
        a = trade_outcome(d, sm, int(i), 1, cost); b = trade_outcome(d, sm, int(i), 2, cost)
        ys.append(np.nan if a is None or b is None else 0.5e4 * (a - b))
    y = np.array(ys); m = np.isfinite(y)
    return idx[m], y[m]


def twin_fit_predict(S, idx, y, seed: int):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                      min_samples_leaf=200, l2_regularization=1.0, random_state=seed)
    m.fit(S[idx], y)
    return m


# ── DSR ──────────────────────────────────────────────────────────────────────
def deflated_sharpe(sr_list: list[float], sr_best: float, T: int, skew: float, kurt: float) -> float:
    """Bailey & López de Prado(2014). sr 는 거래 단위 샤프, T 는 거래 수."""
    N = len(sr_list)
    if N < 2 or T < 3:
        return float("nan")
    v = float(np.var(sr_list, ddof=1)); em = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - em) * _norm.ppf(1 - 1 / N) + em * _norm.ppf(1 - 1 / (N * math.e)))
    den = math.sqrt(max(1e-12, 1 - skew * sr_best + (kurt - 1) / 4 * sr_best ** 2))
    return float(_norm.cdf((sr_best - sr0) * math.sqrt(T - 1) / den))


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--n-env", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ent", type=str, default="0.001,0.0")
    ap.add_argument("--ent-anneal", type=float, default=None,
                    help="이 값에서 --ent 로 선형 감쇠. 조기 관망 붕괴와 «엣지 없음» 을 가른다")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--families", type=str, default=",".join(G.DEFAULT_FAMILIES))
    ap.add_argument("--positive-control", action="store_true")
    ap.add_argument("--twin", action="store_true")
    ap.add_argument("--twin-stride", type=int, default=4)
    ap.add_argument("--taker", action="store_true", help="G9: 진입·청산 테이커 5+5bp")
    ap.add_argument("--tag", type=str, default="main")
    ap.add_argument("--reward-mode", choices=["pnl", "forced", "holdpen", "gross", "antiflat"], default="pnl",
                    help="pnl=순손익 그대로 · forced=홀드 제거 · holdpen=홀드마다 −λ · gross=학습만 수수료 0")
    ap.add_argument("--hold-penalty", type=float, default=0.02)
    ap.add_argument("--anti-flat-lambda", type=float, default=0.08)
    ap.add_argument("--direction-reg", type=float, default=0.20,
                    help="액터 손실의 측면 균형 항 계수. forced 모드에서 0.2 는 «항상 숏」 붕괴를 못 막았다(2026-09-14)")
    ap.add_argument("--anti-flat-min", type=float, default=0.05)
    ap.add_argument("--credit", choices=["gae", "bandit"], default="gae",
                    help="bandit = GAE 사슬을 끊는다(홀드가 남의 거래 공과를 안 받는다)")
    ap.add_argument("--adv-scale", choices=["none", "vol"], default="none",
                    help="vol = 이득을 상태별 예측 변동성으로 나눈다(기울기 가중만 바뀐다)")
    ap.add_argument("--adv-norm", choices=["batch", "std_only", "center_only", "none"], default="batch",
                    help="batch(기존) 는 홀드 95%% 분포에서 상태 무관 홀드 편향을 만든다(arXiv:2508.08221)")
    ap.add_argument("--ortho-init", action="store_true", help="직교 초기화 + 정책머리 gain 0.01")
    ap.add_argument("--critic-lambda", type=float, default=None,
                    help="크리틱 GAE 의 λ 를 액터와 분리(1.0 = VC-PPO, arXiv:2503.01491)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--mb", type=int, default=8192)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fam = tuple(a.families.split(","))
    d, sm, win, S, cols, volexp = prepare(fam)
    cost = {"entry_bp": 5.0, "peg_exit_bp": 5.0} if a.taker else None
    global FORCED, HOLD_PENALTY, ANTI_FLAT_LAMBDA, ANTI_FLAT_MIN, DIRECTION_REG, CREDIT, ADV_SCALE, SIGMA_HAT, ADV_NORM, ORTHO_INIT, CRITIC_LAMBDA
    DIRECTION_REG = a.direction_reg
    CREDIT = a.credit; ADV_SCALE = a.adv_scale; ADV_NORM = a.adv_norm; ORTHO_INIT = a.ortho_init
    CRITIC_LAMBDA = a.critic_lambda
    if ADV_SCALE == "vol":
        SIGMA_HAT = sigma_hat(d, win["TRAIN"])
    FORCED = a.reward_mode == "forced"
    ANTI_FLAT_LAMBDA = a.anti_flat_lambda if a.reward_mode == "antiflat" else 0.0
    ANTI_FLAT_MIN = a.anti_flat_min
    HOLD_PENALTY = a.hold_penalty if a.reward_mode == "holdpen" else 0.0
    train_cost = {"entry_bp": 0.0, "peg_exit_bp": 0.0, "taker_exit_bp": 0.0} if a.reward_mode == "gross" else None
    logf = open(OUT / f"log_{a.tag}.txt", "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    log(f"\n=== {a.tag} · 신용 {a.credit} · 이득눈금 {a.adv_scale} · 이득정규화 {a.adv_norm}"
        f"{' · 직교초기화' if a.ortho_init else ''} · 보상 {a.reward_mode}{'(λ=%g)' % a.hold_penalty if a.reward_mode == 'holdpen' else ''} · 군 {fam} · 상태 {S.shape[1]}D · 창 {[(k, v[1]-v[0]) for k, v in win.items()]}"
        f" · 씨드 {SEEDS[:a.seeds]} · 비용 {'테이커 10bp' if a.taker else 'peg 5.88bp'}")
    report = {"tag": a.tag, "families": fam, "dim": int(S.shape[1]), "seeds": SEEDS[:a.seeds], "taker": a.taker,
              "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
              "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False}

    # ── G1 양성대조 ──────────────────────────────────────────────────────────
    if a.positive_control:
        from sklearn.metrics import roc_auc_score
        log("[G1] 양성대조 -- 같은 상태·같은 PPO 루프, 타깃 = 앞으로 48봉 변동성확장")
        m = train(d, sm, S, win, seed=SEEDS[0], iters=a.iters, ent_coef=0.003, lr=a.lr, n_env=a.n_env,
                  positive=volexp, log=log)
        g1 = {}
        for w in ("VAL", "OOS", "TEST"):
            lo, hi = win[w]; logp, _ = policy_table(m, S[lo:hi])
            g1[w] = float(roc_auc_score(volexp[lo:hi], np.exp(logp[:, 1]) - np.exp(logp[:, 2])))
        log(f"    AUC(P롱−P숏 vs volexp) VAL {g1['VAL']:.3f} OOS {g1['OOS']:.3f} TEST {g1['TEST']:.3f}"
            f" → G1 {'PASS' if g1['OOS'] >= 0.65 else 'FAIL'}")
        report["G1"] = g1

    # ── 학습: HP(엔트로피) × 씨드 ────────────────────────────────────────────
    ents = [float(x) for x in a.ent.split(",")]
    runs = []
    for ent in ents:
        for seed in SEEDS[:a.seeds]:
            log(f"[train] ent {ent} seed {seed}")
            m = train(d, sm, S, win, seed=seed, iters=a.iters, ent_coef=ent, lr=a.lr, n_env=a.n_env,
                      cost=cost, log=log, ent_anneal=a.ent_anneal, train_cost=train_cost)
            ev = {}; acts = {}
            for w in ("TRAIN", "VAL", "OOS", "TEST", "BACK22_23"):
                ev[w], acts[w] = eval_policy(m, d, sm, S, win, w, cost=cost)
            log("    " + " | ".join(f"{w} n{ev[w]['trades']} {ev[w]['net_bp']:+.2f}bp t{ev[w]['t_day']:+.1f} "
                                    f"g{ev[w]['log_mult']:+.3f} side{ev[w]['side_share']:.2f} hit{100*ev[w]['hit_rate']:.1f}"
                                    for w in ("VAL", "OOS", "TEST", "BACK22_23")))
            torch.save(m.state_dict(), OUT / f"policy_{a.tag}_ent{ent}_seed{seed}.pt")
            runs.append({"ent": ent, "seed": seed, "eval": ev, "_model": m, "_acts": acts})
    # HP 선택은 VAL 만 본다(선택 표면 = len(ents)·씨드 평균)
    by_ent = {e: float(np.mean([r["eval"]["VAL"]["net_bp"] for r in runs if r["ent"] == e])) for e in ents}
    best_ent = max(by_ent, key=by_ent.get)
    sel = [r for r in runs if r["ent"] == best_ent]
    log(f"[select] VAL 순bp 씨드평균 {by_ent} → ent {best_ent}")

    # ── 대조군·게이트 (선택된 HP 의 씨드마다) ───────────────────────────────
    rng = np.random.default_rng(20260914)
    q_lo, q_hi = np.nanpercentile(d["ret12"].to_numpy(float)[win["TRAIN"][0]:win["TRAIN"][1]], [10, 90])
    gates = {"G2": {}, "G4": {}, "G5": {}, "G7": {}}
    ctrl_all = {}
    for w in EVAL_WINDOWS + ("BACK22_23",):
        rows = []
        for r in sel:
            c = controls(r["_model"], d, sm, S, win, w, r["_acts"][w], rng, cost=cost)
            c["policy"] = r["eval"][w]
            rows.append(c)
        c1 = eval_policy(sel[0]["_model"], d, sm, S, win, w, A_override=one_feature_rule(d, win, w, q_lo, q_hi), cost=cost)[0]
        KEYS = ("trades", "net_bp", "gross_bp", "hit_rate", "t_day", "log_mult", "growth_per_expo", "side_share", "mdd")
        agg = {k: {m_: float(np.mean([row[k][m_] for row in rows])) for m_ in KEYS} for k in rows[0]}
        agg["one_feature_rule"] = {k: float(c1[k]) for k in KEYS}
        ctrl_all[w] = agg
        log(f"  [{w}] " + "  ".join(f"{k}: n{v['trades']:.0f} {v['net_bp']:+.2f}bp hit{100*v['hit_rate']:.1f} t{v['t_day']:+.1f}"
                                    for k, v in agg.items()))
        # ⭐강제/유도 거래의 판정 통계: 같은 측면 무작위 대비 Δ(씨드별 짝지음) ± SE
        dl = [row["policy"]["net_bp"] - row["same_side_random"]["net_bp"] for row in rows]
        agg["_delta_vs_ssr"] = {"mean": float(np.mean(dl)), "se": float(np.std(dl, ddof=1) / math.sqrt(len(dl))) if len(dl) > 1 else float("nan")}
        log(f"        Δ(정책−같은측면무작위) {agg['_delta_vs_ssr']['mean']:+.2f} ± {agg['_delta_vs_ssr']['se']:.2f}bp"
            f" · 정책 적중 {100*agg['policy']['hit_rate']:.1f}% · 역방향 {agg['reversed']['net_bp']:+.2f}bp")
        pol = [row["policy"] for row in rows]
        se = lambda key, arr: float(np.std([x[key] for x in arr], ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else float("nan")
        if w in EVAL_WINDOWS:
            gates["G2"][w] = bool(agg["policy"]["net_bp"] > 0 and agg["policy"]["t_day"] >= 2 and agg["policy"]["log_mult"] > 0)
            d_rev = agg["reversed"]["net_bp"]; d_ssr = agg["policy"]["net_bp"] - agg["same_side_random"]["net_bp"]
            se_ssr = se("net_bp", [{"net_bp": row["policy"]["net_bp"] - row["same_side_random"]["net_bp"]} for row in rows])
            gates["G4"][w] = bool(d_rev < 0 and (d_ssr > 2 * se_ssr if np.isfinite(se_ssr) else d_ssr > 0))
            gates["G7"][w] = bool(agg["policy"]["trades"] < 10 or agg["policy"]["side_share"] > 0.85)
    signs = {w: np.sign(ctrl_all[w]["policy"]["net_bp"]) for w in ("VAL", "OOS", "TEST", "BACK22_23")}
    gates["G5"] = bool(signs["BACK22_23"] == np.sign(np.mean([ctrl_all[w]["policy"]["net_bp"] for w in EVAL_WINDOWS])))
    seed_signs = [np.sign(np.mean([r["eval"][w]["net_bp"] for w in EVAL_WINDOWS])) for r in sel]
    gates["G6_seed_sign_agree"] = bool(len(set(seed_signs)) == 1)
    # DSR: HP 조합(ents × seeds) 전부를 시행으로 보고, 선택된 팔의 VAL 거래단위 샤프
    srs = [r["eval"]["VAL"]["t_trade"] / math.sqrt(max(r["eval"]["VAL"]["trades"], 1)) for r in runs]
    best = max(sel, key=lambda r: r["eval"]["VAL"]["net_bp"])
    tr = best["_model"]; lo, hi = win["VAL"]
    res = G.DirectionGym(d, sm, lo, hi, **(cost or {})).run(lambda i: (int(best["_acts"]["VAL"][i - lo]), None))
    rr = np.array([t["r"] for t in res["_trades"]])
    from scipy.stats import kurtosis, skew
    gates["G6_DSR"] = deflated_sharpe(srs, max(srs), len(rr), float(skew(rr)) if len(rr) > 3 else 0.0,
                                      float(kurtosis(rr, fisher=False)) if len(rr) > 3 else 3.0)
    g2_pass = sum(gates["G2"].values()) >= 2
    log(f"[gates] G2 {gates['G2']} → {'PASS' if g2_pass else 'FAIL'} · G4 {gates['G4']} · G5 {gates['G5']}"
        f" · G6 씨드부호 {gates['G6_seed_sign_agree']} DSR {gates['G6_DSR']:.3f} · G7 붕괴 {gates['G7']}")

    # ── G3 지도학습 쌍둥이 ────────────────────────────────────────────────────
    if a.twin:
        log(f"[G3] 쌍둥이 라벨 (stride {a.twin_stride}) …")
        t0 = time.time(); idx, y = twin_labels(d, sm, win, a.twin_stride, cost)
        log(f"    라벨 {len(y):,} · 평균 {y.mean():+.2f}bp · {time.time()-t0:.0f}s")
        twin = {}
        for w in EVAL_WINDOWS + ("BACK22_23",):
            lo, hi = win[w]; rows = []
            for r in sel:
                m = twin_fit_predict(S, idx, y, r["seed"]); pred = m.predict(S[lo:hi])
                best_c, best_m = None, None
                # 임계 c 는 VAL 에서 고른다(정책의 HP 선택과 같은 잣대) -- 다른 창엔 그 c 를 그대로
                cs = (0.0, 5.0, 10.0, 20.0, 40.0)
                if w == "VAL":
                    for c_ in cs:
                        A = np.where(pred > c_, 1, np.where(pred < -c_, 2, 0))
                        mm = eval_policy(None, d, sm, S, win, w, A_override=A, cost=cost)[0]
                        if best_m is None or mm["net_bp"] > best_m["net_bp"]:
                            best_c, best_m = c_, mm
                    r["_twin_c"] = best_c
                else:
                    c_ = r.get("_twin_c", 10.0)
                    A = np.where(pred > c_, 1, np.where(pred < -c_, 2, 0))
                    best_m = eval_policy(None, d, sm, S, win, w, A_override=A, cost=cost)[0]
                rows.append(best_m)
            twin[w] = {k: float(np.mean([x[k] for x in rows])) for k in ("trades", "net_bp", "t_day", "log_mult")}
            dlt = [r["eval"][w]["net_bp"] - x["net_bp"] for r, x in zip(sel, rows)]
            twin[w]["delta_policy_minus_twin"] = float(np.mean(dlt))
            twin[w]["delta_se"] = float(np.std(dlt, ddof=1) / math.sqrt(len(dlt))) if len(dlt) > 1 else float("nan")
            log(f"    [{w}] 쌍둥이 n{twin[w]['trades']:.0f} {twin[w]['net_bp']:+.2f}bp t{twin[w]['t_day']:+.1f}"
                f" · Δ(정책−쌍둥이) {twin[w]['delta_policy_minus_twin']:+.2f} ± {twin[w]['delta_se']:.2f}")
        gates["G3"] = {w: bool(twin[w]["delta_policy_minus_twin"] > 2 * twin[w]["delta_se"]) for w in EVAL_WINDOWS}
        report["twin"] = twin
        # G8 재료: 단변량 분위 효과 (쌍둥이 라벨 기준, 학습창)
        uni = {}
        for j, cname in enumerate(cols):
            x = S[idx, j]; q = pd.qcut(pd.Series(x), 10, labels=False, duplicates="drop")
            g_ = pd.Series(y).groupby(q.to_numpy()).mean()
            uni[cname] = float(g_.max() - g_.min()) if len(g_) > 1 else 0.0
        report["G8_univariate_decile_spread_bp"] = dict(sorted(uni.items(), key=lambda kv: -kv[1])[:10])
        log("    [G8] 단변량 십분위 스프레드 상위: " + ", ".join(f"{k} {v:.1f}" for k, v in list(report["G8_univariate_decile_spread_bp"].items())[:6]))

    # ── G8 순열 중요도 (선택 팔 첫 씨드, OOS) ─────────────────────────────────
    m = sel[0]["_model"]; lo, hi = win["OOS"]; base = sel[0]["eval"]["OOS"]["net_bp"]; imp = {}
    prng = np.random.default_rng(1)
    for j, cname in enumerate(cols):
        Sp = S[lo:hi].copy(); Sp[:, j] = prng.permutation(Sp[:, j])
        logp, _ = policy_table(m, Sp); A = logp.argmax(1)
        mm = eval_policy(None, d, sm, S, win, "OOS", A_override=A, cost=cost)[0]
        imp[cname] = base - mm["net_bp"]
    top = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:15])
    report["G8_permutation_importance_bp"] = top
    log("    [G8] 순열 중요도 상위: " + ", ".join(f"{k} {v:+.2f}" for k, v in list(top.items())[:8]))

    report.update({"runs": [{"ent": r["ent"], "seed": r["seed"], "eval": r["eval"]} for r in runs],
                   "selected_ent": best_ent, "controls": ctrl_all, "gates": gates})
    json.dump(report, open(OUT / f"report_{a.tag}.json", "w"), ensure_ascii=False, indent=1, default=float)
    log(f"저장 {OUT / f'report_{a.tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
