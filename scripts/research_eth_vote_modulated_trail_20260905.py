#!/usr/bin/env python3
"""표결식 청산 2 — **자르지 말고 트레일을 조인다**(연속 변조) (2026-09-05).

§18은 "야당 표 ≥ θ이면 즉시 청산"이라는 **이분법**만 쟀고 144팔 전부 졌다. 그런데 §17 E3에 남은 단서가 있다:
**야당 표에서 청산하는 것은 무작위 조기청산보다 세 창 모두 일관되게 나았다**(TRAIN +13.7 · VAL +6.0 · OOS +4.4).
즉 신호는 약하게 실재하는데 **조기 청산 세금(−27~−30bp/일)**이 삼킨 것이다.
⇒ 약한 신호는 자르는 데 쓰지 말고 **이미 있는 청산 장치를 미세 조정**하는 데 써야 한다.

## 팔 — 표결이 트레일/무장을 연속 변조한다 (즉시 청산 없음)
    net = 야당 − 여당 (유지창 S봉, 봉 마감 기준, 뒤만 봄)
    trail_t = 0.1×ATR × (1 − β · clip(net,0,3)/3)     야당이 셀수록 **트레일이 조여진다**
    arm_t   = 1.5×ATR × (1 − γ · clip(net,0,3)/3)     야당이 셀수록 **더 일찍 무장한다**
    β, γ ∈ {0.25, 0.5, 0.75} · S ∈ {6, 12, 24} · 변조 대상 {trail, arm, both}

## 대조군 (이게 이 검정의 핵심)
    FLIP   여당 표로 같은 변조 (부호 반대) — 진짜 신호면 명확히 져야 한다
    UNCOND ⭐**같은 평균 트레일/무장 세기를 표결 없이 무조건 적용** — 표결이 "그냥 조인 것" 이상인가.
           변조 팔의 실현 평균 배수를 그대로 상수로 걸어 맞춘다.
    R      현행 (변조 없음)
파리티: 변조 배수를 1로 두면 R의 net_bp와 비트 일치해야 한다(아니면 중단).
판정: VAL·OOS 두 창 모두 R 대비 **그리고** UNCOND 대비 일별 짝비교 CI 하한 > 0. HOLDOUT 미접촉.
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


C1 = _load("c1_mod", "scripts/research_eth_composite_direction_trend_pullback_20260905.py")
XA = _load("xa_mod", "scripts/research_crossasset_fire_continuation_replication_20260905.py")
VB = _load("vb_mod", "scripts/research_eth_vote_based_exit_20260905.py")
OUT = ROOT / "data/research/eth_vote_modulated_trail_20260905"
BETAS, SUSTAINS, TARGETS = (0.25, 0.5, 0.75), (6, 12, 24), ("trail", "arm", "both")
COST, WINDOWS, NET_CAP = 10.0, ("TRAIN", "VAL", "OOS"), 3.0


def log(m): print(f"[mod] {m}", flush=True)


def sim_exit_mod(entry, atr, sign, H, L, C, sl, arm_b, trail_b, arm_m=None, trail_m=None):
    """sim_exit 원문에 **봉별 배수**를 넣은 형태. arm_m/trail_m = None이면 원문과 비트 일치해야 한다."""
    n, T = len(entry), H.shape[1]
    stop = entry - sign * sl * atr
    armed = np.zeros(n, bool); best = entry.copy()
    done = np.zeros(n, bool); out = np.zeros(n); ex = np.full(n, T - 1)
    fav = np.where(sign[:, None] > 0, H, L); adv = np.where(sign[:, None] > 0, L, H)
    for t in range(T):
        if done.all():
            break
        a_ = adv[:, t]; live = ~done
        hit = live & np.where(sign > 0, a_ <= stop, a_ >= stop)
        out = np.where(hit, sign * (stop - entry) / entry, out); ex = np.where(hit, t, ex); done = done | hit
        f_ = fav[:, t]; live = ~done
        imp = live & (sign * (f_ - best) > 0); best = np.where(imp, f_, best)
        a_lvl = arm_b * atr * (1.0 if arm_m is None else arm_m[:, t])
        # ⭐**낡은 best 무장 금지 가드**(2026-09-05): 무장은 best가 **이 봉에서 갱신됐을 때만** 허용한다.
        # 상수 임계값(원본 sim_exit)에서는 이 조건이 자동으로 성립한다 -- 실측 무장 지연 중앙 0봉·최대 0봉.
        # 가변 임계값에서는 성립하지 않아 **최대 73봉 전의 best**로 스톱이 걸리고, 시장이 이미 지나간
        # 가격에 체결돼 수익이 조작된다(TRAIN 행평균 4.85 → 20.9bp). 정보를 지운 시간 셔플에서도 18.6이
        # 남는 것으로 아티팩트임을 확인했다. 이 가드가 없으면 어떤 시간가변 무장 규칙도 검정할 수 없다.
        newly = live & ~armed & imp & (sign * (best - entry) >= a_lvl); armed = armed | newly
        t_dist = trail_b * atr * (1.0 if trail_m is None else trail_m[:, t])
        ns = best - sign * t_dist
        u = live & armed & (sign * (ns - stop) > 0); stop = np.where(u, ns, stop)
    return np.where(done, out, sign * (C[:, -1] - entry) / entry), ex


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    B = C1.build()
    pos, split, ts, bidx = B["pos"], B["split"], B["ts"], B["bidx"]
    cont_bp, cont_ex, atr, entry, cs = B["cont_bp"], B["cont_ex"], B["atr"], B["entry"], B["cont_sign"]
    o, h, l, c = B["o"], B["h"], B["l"], B["c"]; n = len(c); FWD = C1.FWD
    SL, ARM, TR = C1.CELL
    ix = (bidx + 1)[:, None] + np.arange(FWD); H_, L_, C_ = h[ix], l[ix], c[ix]
    r0, _ = sim_exit_mod(entry, atr, cs, H_, L_, C_, SL, ARM, TR)
    par = float(np.max(np.abs(r0 * 1e4 - COST - cont_bp)))
    log(f"파리티 |Δ|max {par:.3e} bp (변조 없을 때 R과 비트 일치)"); assert par < 1e-9

    # 표결 (라이브 정본 compute_signals, §18과 같은 산식)
    kl = pd.read_csv(VB.KL, usecols=["timestamp", "open", "high", "low", "close", "volume", "trades", "taker_buy_base"], parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    btc = pd.read_csv(VB.KL_BTC, usecols=["timestamp", "open", "high", "low", "close", "volume", "trades", "taker_buy_base"], parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    sig = XA.DASH.compute_signals(kl.copy(), btc_df=btc, funding_df=None)
    k0 = int(np.searchsorted(kl["timestamp"].to_numpy(), np.datetime64(B["bar"]["timestamp"].iloc[0])))
    seg = sig.iloc[k0:k0 + n].reset_index(drop=True); assert len(seg) == n
    V = {}
    for S in SUSTAINS:
        bot = np.zeros(n, np.int8); top = np.zeros(n, np.int8)
        for s_ in XA.SIGNALS:
            for side, acc in (("bottom", "bot"), ("top", "top")):
                col = f"{side}_{s_}"
                if col not in seg.columns:
                    continue
                v = seg[col].fillna(False).to_numpy(bool)
                v = pd.Series(v.astype(np.int8)).rolling(S, min_periods=1).max().to_numpy() > 0
                (bot if acc == "bot" else top)[v] += 1
        V[S] = (bot, top)
    base = {w: C1.pf(C1.cand_of(ts[split == w], pos[split == w] + 1, pos[split == w] + 1 + cont_ex[split == w], cont_bp[split == w])) for w in WINDOWS}
    rep = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "parity_max_abs_bp": par, "holdout_touched": False,
           "baseline_R": {w: base[w]["stats"] for w in WINDOWS}, "arms": {}}

    def net_mat(S, flip):
        bot, top = V[S]
        pick_top = (cs < 0) if not flip else (cs > 0)      # 지속 숏의 야당 = 천장 표
        opp = np.where(pick_top[:, None], top[ix], bot[ix]).astype(float)
        sup = np.where(pick_top[:, None], bot[ix], top[ix]).astype(float)
        return np.clip(opp - sup, 0.0, NET_CAP) / NET_CAP  # 0~1

    def evaluate(nm, arm_m, trail_m, ref_arm=None):
        r, ex = sim_exit_mod(entry, atr, cs, H_, L_, C_, SL, ARM, TR, arm_m, trail_m)
        p = r * 1e4 - COST; rec = {}
        for w in WINDOWS:
            m = split == w
            rr = C1.pf(C1.cand_of(ts[m], pos[m] + 1, pos[m] + 1 + ex[m], p[m]))
            rec[w] = {**{k: rr["stats"][k] for k in ("n", "exp_bp", "win_rate", "day_ci95", "daily_mean_bp", "daily_sharpe_ann")},
                      "mean_hold_bars": round(float(ex[m].mean()), 1),
                      "vs_R": C1.day_paired(rr["pnl"], rr["ts"], base[w]["pnl"], base[w]["ts"]),
                      "_pnl": rr["pnl"], "_ts": rr["ts"]}
            if ref_arm is not None:
                rec[w]["vs_UNCOND"] = C1.day_paired(rr["pnl"], rr["ts"], ref_arm[w]["_pnl"], ref_arm[w]["_ts"])
        rep["arms"][nm] = {w: {k: v for k, v in rec[w].items() if not k.startswith("_")} for w in WINDOWS}
        return rec

    results = []
    for S, beta, tgt, flip in itertools.product(SUSTAINS, BETAS, TARGETS, (False, True)):
        g = net_mat(S, flip)
        am = (1.0 - beta * g) if tgt in ("arm", "both") else None
        tm = (1.0 - beta * g) if tgt in ("trail", "both") else None
        # UNCOND 대조: 같은 실현 평균 배수를 표결 없이 상수로
        mean_mult = float((1.0 - beta * g).mean())
        u_am = np.full_like(g, mean_mult) if tgt in ("arm", "both") else None
        u_tm = np.full_like(g, mean_mult) if tgt in ("trail", "both") else None
        unc = evaluate(f"UNCOND_{tgt}_x{mean_mult:.3f}", u_am, u_tm)
        nm = f"{'FLIP_' if flip else ''}{tgt}_S{S}_b{beta}"
        rec = evaluate(nm, am, tm, ref_arm=unc)
        if not flip:
            results.append((min(rec[w]["vs_UNCOND"]["diff_bp_day"] for w in ("VAL", "OOS")), nm, mean_mult, rec))
    results.sort(reverse=True)
    log(f"팔 {len(rep['arms'])}개. 상위 6 (두 확인 창 vs UNCOND 최솟값 기준):")
    for v, nm, mm, rec in results[:6]:
        log(f"  {nm:20s} (평균배수 {mm:.3f}) " + " | ".join(
            f"{w} exp={rec[w]['exp_bp']:>6} ΔR={rec[w]['vs_R']['diff_bp_day']:>7}{str(rec[w]['vs_R']['ci95']):>16} "
            f"ΔUNC={rec[w]['vs_UNCOND']['diff_bp_day']:>6}{str(rec[w]['vs_UNCOND']['ci95']):>16}" for w in WINDOWS))
    P = [nm for _, nm, _, rec in results
         if all(rec[w]["vs_R"]["ci95"][0] > 0 and rec[w]["vs_UNCOND"]["ci95"][0] > 0 for w in ("VAL", "OOS"))]
    rep["verdict"] = {"rule": "VAL·OOS 두 창 모두 R 대비 **그리고** UNCOND 대비 CI 하한 > 0", "passes": P, "n_pass": len(P)}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"완료 {time.time()-t0:.0f}s → {OUT/'report.json'} · 통과 {len(P)} {P}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
