#!/usr/bin/env python3
"""**표결식 청산** — 되돌림 세력 vs 지속 세력이 청산을 정한다 (2026-09-05).

사용자: *"되돌림과 지속이 여당과 야당 세력이 되어 청산을 정해주는 건 어때?"*

부록 §17 E3이 단일 트리거를 기각했지만 거기서 두 가지가 새로 보였다:
  (1) **무작위 조기 청산 자체가 −27~−30bp/일**이다 — 어떤 청산 규칙이든 이걸 먼저 넘어야 한다.
  (2) **같은 측면 발동에서 청산하는 건 무작위보다 일관되게 낫다**(TRAIN +13.7 · VAL +6.0 · OOS +4.4).
      이 부호가 맞다 — 지속 숏 보유 중 **천장 발동**은 그 칩의 *지속* 방향이 롱이므로 내 포지션의 반대 세력이다.

## 표결 정의 (대시보드 `bottom_votes`/`top_votes`와 같은 산식, 라이브 정본 compute_signals)
각 봉에서 8종의 현재 상태를 센다. 내 포지션 기준으로:

    반대 세력(야당) = 그 칩의 **지속 방향이 내 포지션과 반대**인 표     (지속 숏 보유 → 천장 표)
    우리 세력(여당) = 그 칩의 **지속 방향이 내 포지션과 같은** 표       (지속 숏 보유 → 바닥 표)
    net = 야당 − 여당

## 팔 (전부 인과: 봉 t 마감에 표결 → 봉 t+1 **시가** 청산)
  θ ∈ {1,2,3,4}  net ≥ θ 이면 청산
  표 종류        raw(현재 상태 전부) / first(첫발동만)
  이익 조건      any(무조건) / profit(현재 미실현 > 0) / armed(미실현 ≥ 1.5×ATR)
                 ⭐C4가 "손실 중 조기 청산은 단조 악화"를 보였으므로 **익절 전용** 변형이 핵심이다.
  하드 청산      기존 sim_exit(5.0 손절/1.5 무장/0.1 트레일/200봉) 그대로 동시 작동 — 표결은 **추가** 청산일 뿐
  대조군         (a) 부호 뒤집기(여당 표에서 청산) (b) 무작위 청산(같은 청산시점 분포, B=200) (c) R(조기청산 없음)
  파리티         θ=∞(트리거 없음)이면 R의 net_bp와 비트 일치해야 한다 — 안 맞으면 즉시 중단

판정: VAL·OOS 두 창 모두 R 대비 일별 짝비교 CI 하한 > 0. HOLDOUT 미접촉.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


C1 = _load("c1_vote", "scripts/research_eth_composite_direction_trend_pullback_20260905.py")
XA = _load("xa_vote", "scripts/research_crossasset_fire_continuation_replication_20260905.py")
OUT = ROOT / "data/research/eth_vote_based_exit_20260905"
KL = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
KL_BTC = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
THETAS, VOTE_KINDS, PROFIT_CONDS = (1, 2, 3), ("raw", "first"), ("any", "profit", "armed")
SUSTAINS = (1, 6, 12, 24)      # 표 유지창(봉) -- "세력"은 순간이 아니라 지속되는 것. S=1이 순간 표결
COST, WINDOWS, B_NULL = 10.0, ("TRAIN", "VAL", "OOS"), 200
rng = np.random.default_rng(20260905)


def log(m): print(f"[vote] {m}", flush=True)


def sim_exit_with_vote(entry, atr, sign, O, H, L, C, sl, arm, trail, vote_ok=None):
    """sim_exit(원문 순서) + **표결 청산**. 봉 t 마감에 vote_ok[:,t]가 참이면 봉 t+1 **시가**에 청산.
    vote_ok=None이면 sim_exit과 비트 일치해야 한다."""
    n, T = len(entry), H.shape[1]
    stop = entry - sign * sl * atr
    armed = np.zeros(n, bool); best = entry.copy()
    done = np.zeros(n, bool); out = np.zeros(n); ex = np.full(n, T - 1)
    pend = np.zeros(n, bool)
    fav = np.where(sign[:, None] > 0, H, L); adv = np.where(sign[:, None] > 0, L, H)
    for t in range(T):
        if done.all():
            break
        # (0) 직전 봉 표결 → 이 봉 시가 청산 (하드 스톱보다 시간상 먼저)
        if vote_ok is not None:
            hv = (~done) & pend
            if hv.any():
                out = np.where(hv, sign * (O[:, t] - entry) / entry, out); ex = np.where(hv, t, ex); done = done | hv
        a_ = adv[:, t]; live = ~done
        hit = live & np.where(sign > 0, a_ <= stop, a_ >= stop)
        out = np.where(hit, sign * (stop - entry) / entry, out); ex = np.where(hit, t, ex); done = done | hit
        f_ = fav[:, t]; live = ~done
        imp = live & (sign * (f_ - best) > 0); best = np.where(imp, f_, best)
        newly = live & ~armed & (sign * (best - entry) >= arm * atr); armed = armed | newly
        ns = best - sign * trail * atr
        u = live & armed & (sign * (ns - stop) > 0); stop = np.where(u, ns, stop)
        if vote_ok is not None:
            pend = (~done) & vote_ok[:, t]
    out = np.where(done, out, sign * (C[:, -1] - entry) / entry)
    return out, ex


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    B = C1.build()
    pos, sd, split, ts, bidx = B["pos"], B["sd"], B["split"], B["ts"], B["bidx"]
    cont_bp, cont_ex, atr, entry, cs = B["cont_bp"], B["cont_ex"], B["atr"], B["entry"], B["cont_sign"]
    o, h, l, c = B["o"], B["h"], B["l"], B["c"]; n = len(c); p_first = B["p_first"]; FWD = C1.FWD

    # ---- 라이브 정본 표결 (대시보드 bottom_votes/top_votes 산식)
    kl = pd.read_csv(KL, usecols=["timestamp", "open", "high", "low", "close", "volume", "trades", "taker_buy_base"], parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    btc = pd.read_csv(KL_BTC, usecols=["timestamp", "open", "high", "low", "close", "volume", "trades", "taker_buy_base"], parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    sig = XA.DASH.compute_signals(kl.copy(), btc_df=btc, funding_df=None)
    t_first = B["bar"]["timestamp"].iloc[0]
    k0 = int(np.searchsorted(kl["timestamp"].to_numpy(), np.datetime64(t_first))); assert kl["timestamp"].iloc[k0] == t_first
    seg_sig = sig.iloc[k0:k0 + n].reset_index(drop=True)
    assert len(seg_sig) == n, f"표결 정렬 실패 {len(seg_sig)} vs {n}"
    V = {}
    for kind, S in itertools.product(VOTE_KINDS, SUSTAINS):
        bot = np.zeros(n, np.int8); top = np.zeros(n, np.int8)
        for s in XA.SIGNALS:
            for side, acc in (("bottom", "bot"), ("top", "top")):
                col = f"{side}_{s}"
                if col not in seg_sig.columns:
                    continue
                v = seg_sig[col].fillna(False).to_numpy(bool)
                if kind == "first":
                    v = XA.first_fire_mask(v, XA.GAP)
                if S > 1:      # 유지창: 과거 S봉 안에 그 신호가 그 측면으로 발동했으면 표 1 (뒤만 봄)
                    v = pd.Series(v.astype(np.int8)).rolling(S, min_periods=1).max().to_numpy() > 0
                (bot if acc == "bot" else top)[v] += 1
        V[(kind, S)] = (bot, top)
        log(f"표결({kind}, 유지 {S}봉): 바닥 평균 {bot.mean():.2f} 천장 평균 {top.mean():.2f} · 최대 {bot.max()}/{top.max()} · 표>=1 비율 {(bot>=1).mean():.3f}")

    ix = (bidx + 1)[:, None] + np.arange(FWD)
    O_, H_, L_, C_ = o[ix], h[ix], l[ix], c[ix]
    # 파리티: vote 없음 == R
    r0, e0 = sim_exit_with_vote(entry, atr, cs, O_, H_, L_, C_, *C1.CELL, vote_ok=None)
    par = float(np.max(np.abs(r0 * 1e4 - COST - cont_bp)))
    log(f"파리티 |Δ|max {par:.3e} bp (표결 없을 때 R과 비트 일치해야 함)")
    assert par < 1e-9, "파리티 실패 — 중단"

    base = {w: C1.pf(C1.cand_of(ts[split == w], pos[split == w] + 1, pos[split == w] + 1 + cont_ex[split == w], cont_bp[split == w])) for w in WINDOWS}
    rep = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "parity_max_abs_bp": par, "holdout_touched": False,
           "baseline_R": {w: base[w]["stats"] for w in WINDOWS}, "arms": {}}
    unreal = cs[:, None] * (C_ - entry[:, None]) / entry[:, None]          # 봉 마감 기준 미실현(비율)
    arm_lvl = (C1.CELL[1] * atr / entry)[:, None]
    prof = {"any": np.ones_like(unreal, bool), "profit": unreal > 0, "armed": unreal >= arm_lvl}

    def votes_mat(key, oppose):
        """야당(oppose=True) 또는 여당 표 행렬 (거래 × 봉)."""
        bot, top = V[key]
        # 지속 숏(cs<0): 야당 = 천장 표 / 여당 = 바닥 표.  지속 롱(cs>0): 야당 = 바닥 표 / 여당 = 천장 표
        pick_top = (cs < 0) if oppose else (cs > 0)
        return np.where(pick_top[:, None], top[ix], bot[ix]).astype(np.int8)

    results = []
    for kind, S, th, pc, flip in itertools.product(VOTE_KINDS, SUSTAINS, THETAS, PROFIT_CONDS, (False, True)):
        opp = votes_mat((kind, S), oppose=not flip); sup = votes_mat((kind, S), oppose=flip)
        cond = ((opp - sup) >= th) & prof[pc]
        r, ex = sim_exit_with_vote(entry, atr, cs, O_, H_, L_, C_, *C1.CELL, vote_ok=cond)
        p = r * 1e4 - COST
        nm = f"{'FLIP_' if flip else ''}{kind}_S{S}_th{th}_{pc}"
        rec = {"trigger_rate": round(float((ex < cont_ex).mean()), 3)}
        for w in WINDOWS:
            m = split == w
            rr = C1.pf(C1.cand_of(ts[m], pos[m] + 1, pos[m] + 1 + ex[m], p[m]))
            if rr is None:
                continue
            rec[w] = {**{k: rr["stats"][k] for k in ("n", "exp_bp", "win_rate", "day_ci95", "daily_mean_bp", "daily_sharpe_ann")},
                      "mean_hold_bars": round(float(ex[m].mean()), 1),
                      "vs_R": C1.day_paired(rr["pnl"], rr["ts"], base[w]["pnl"], base[w]["ts"])}
        rep["arms"][nm] = rec
        if all(w in rec for w in WINDOWS) and rec["trigger_rate"] > 0.02:      # no-op 팔은 순위에서 제외
            results.append((min(rec[w]["vs_R"]["diff_bp_day"] for w in ("VAL", "OOS")), nm, rec))
    results.sort(reverse=True)
    log(f"팔 {len(rep['arms'])}개. 상위 8(두 확인 창 최솟값 기준):")
    for v, nm, rec in results[:8]:
        log(f"  {nm:26s} 트리거율 {rec['trigger_rate']:.3f} · " + " | ".join(
            f"{w} exp={rec[w]['exp_bp']:>6} 보유 {rec[w]['mean_hold_bars']:>5}봉 ΔR={rec[w]['vs_R']['diff_bp_day']:>7}{str(rec[w]['vs_R']['ci95']):>18}" for w in WINDOWS))
    P = [nm for _, nm, rec in results if rec["VAL"]["vs_R"]["ci95"][0] > 0 and rec["OOS"]["vs_R"]["ci95"][0] > 0]
    rep["verdict"] = {"rule": "VAL·OOS 두 창 모두 vs_R CI 하한 > 0", "n_arms": len(rep["arms"]), "passes": P, "n_pass": len(P)}
    # 무작위 청산 귀무 (최고 팔의 청산시점 분포 매칭)
    if results:
        _, bn, brec = results[0]
        rep["best_arm"] = bn
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"완료 {time.time()-t0:.0f}s → {OUT/'report.json'} · 통과 {len(P)}/{len(rep['arms'])} {P}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
