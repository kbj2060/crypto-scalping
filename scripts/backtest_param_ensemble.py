#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np

try:
    from scripts.optimize_duckdb_quant_formula import load_merged, run_sim, sample_params, sigmoid
except ModuleNotFoundError:
    from optimize_duckdb_quant_formula import load_merged, run_sim, sample_params, sigmoid

FEATURE_TEMPLATES = {
    "T_A_WHALE_RISK": ("w_nif", "w_tox", "w_aft"),
    "T_B_BOOK_ABS": ("w_obi", "w_abs", "w_flow"),
    "T_C_ENERGY_LIQ": ("w_eai", "w_liq", "w_flow"),
    "T_D_ORDERFLOW_RISK": ("w_vpin", "w_flow", "w_tox"),
    "T_E_WHALE_ABS_ENERGY": ("w_nif", "w_abs", "w_eai"),
}
WEIGHT_KEYS = ("w_nif", "w_flow", "w_obi", "w_abs", "w_liq", "w_eai", "w_tox", "w_aft", "w_vpin")


def _scores_for_param(m, p):
    obi = m["obi"].to_numpy(np.float64)
    flow = (2.0 * m["taker_buy_ratio"].clip(0, 1) - 1.0).to_numpy(np.float64)
    nif = m["nif_whale"].to_numpy(np.float64)
    absb = m["shadow_absorption_score"].to_numpy(np.float64)
    tox = m["shadow_toxicity_score"].to_numpy(np.float64)
    qcol = m["shadow_queue_collapse"].to_numpy(np.float64)
    aft = m["shadow_aftershock_prob"].to_numpy(np.float64)
    eai = m["eai"].to_numpy(np.float64)
    liq = ((m["short_usd_1m"] - m["long_usd_1m"]) / (m["short_usd_1m"].abs() + m["long_usd_1m"].abs() + 1e-8)).to_numpy(np.float64)

    imb = (m["quote_volume"].fillna(0.0) * np.abs(flow))
    vpin = (imb.rolling(60, min_periods=12).sum() / np.maximum(m["quote_volume"].rolling(60, min_periods=12).sum(), 1e-8)).fillna(0.0).clip(0, 1).to_numpy(np.float64)

    # overheat (rolling z)
    oi = m["oi_delta_pct"]
    fd = m["funding_rate"]
    oi_mu = oi.rolling(96, min_periods=24).mean()
    oi_sd = oi.rolling(96, min_periods=24).std().replace(0, np.nan)
    fd_mu = fd.rolling(96, min_periods=24).mean()
    fd_sd = fd.rolling(96, min_periods=24).std().replace(0, np.nan)
    over = (((oi - oi_mu) / oi_sd).replace([np.inf, -np.inf], np.nan).fillna(0.0) + ((fd - fd_mu) / fd_sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)).to_numpy(np.float64)

    # regime filters
    close = m["close"]
    prev_close = close.shift(1).fillna(close)
    trr = np.maximum((m["high"] - m["low"]).abs(), np.maximum((m["high"] - prev_close).abs(), (m["low"] - prev_close).abs()))
    atr = (trr.rolling(14, min_periods=5).mean() / np.maximum(close, 1e-8)).fillna(0.0).to_numpy(np.float64)
    v1h = m["volume"].rolling(12, min_periods=6).mean().fillna(0.0)
    v24h = m["volume"].rolling(288, min_periods=24).mean().ffill().fillna(1.0)
    volr = (v1h / np.maximum(v24h, 1e-8)).fillna(0.0).to_numpy(np.float64)

    raw_base = (
        p["w_nif"] * nif
        + p["w_flow"] * flow
        + p["w_obi"] * (-obi)
        + p["w_abs"] * absb
        + p["w_liq"] * liq
        + p["w_eai"] * np.tanh(eai / 2.0)
        - p["w_tox"] * tox
        - p["w_aft"] * aft
        - p["w_vpin"] * np.clip(vpin - 0.7, 0, 1)
    )

    # v2: regime-aware blend (trend formula + stress formula)
    formula_v2 = bool(p.get("_formula_v2", False))
    if formula_v2:
        stress_gate = sigmoid(
            8.0 * (vpin - 0.68)
            + 6.0 * (tox - 0.50)
            + 6.0 * (qcol - 0.45)
            + 4.0 * (aft - 0.45)
        )
        raw_trend = raw_base
        raw_stress = (
            -0.7 * p["w_nif"] * nif
            -0.5 * p["w_flow"] * flow
            +0.9 * p["w_obi"] * obi
            -0.4 * p["w_abs"] * absb
            -0.2 * p["w_eai"] * np.tanh(eai / 2.0)
            -0.1 * p["w_liq"] * liq
            -0.6 * p["w_tox"] * tox
            -0.5 * p["w_aft"] * aft
            -0.4 * p["w_vpin"] * np.clip(vpin - 0.65, 0, 1)
        )
        raw = (1.0 - stress_gate) * raw_trend + stress_gate * raw_stress
    else:
        raw = raw_base

    long_gate = (over < p["overheat_long_max"]).astype(float)
    short_boost = np.where(over > p["overheat_short_min"], p["short_boost"], 1.0)
    base_long = sigmoid((raw - p["bias"]) / max(p["temp"], 1e-4))
    base_short = sigmoid((-raw - p["bias"]) / max(p["temp"], 1e-4))
    tail_pen = np.clip(1.0 - (p["tail_tox"] * tox + p["tail_qc"] * qcol + p["tail_aft"] * aft), 0.0, 1.0)
    long_score = base_long * long_gate * tail_pen
    short_score = base_short * short_boost * tail_pen
    tradable = (atr >= p["atr_min"]) & (volr >= p["volr_min"]) & (vpin <= p["vpin_max"])

    return long_score, short_score, tradable, raw, tox, qcol


def _param_signal(m, p):
    long_score, short_score, tradable, raw, _, _ = _scores_for_param(m, p)
    hh = m["ts"].dt.hour
    us = ((hh >= 22) | (hh <= 4)).to_numpy(np.float64)
    eu = ((hh >= 15) & (hh < 22)).to_numpy(np.float64)

    N = len(m)
    sig = np.zeros(N, dtype=np.int8)
    strength = np.zeros(N, dtype=np.float64)
    long_h = False
    short_h = False

    for i in range(1, N):
        sess_mult = 0.93 if us[i] > 0.5 else (0.97 if eu[i] > 0.5 else 1.05)
        entry_th = p["entry"] * sess_mult
        exit_th = p["exit"] * (0.98 if us[i] > 0.5 else 1.0)

        if not long_h and long_score[i] >= entry_th:
            long_h = True
        elif long_h and long_score[i] <= exit_th:
            long_h = False
        if not short_h and short_score[i] >= entry_th:
            short_h = True
        elif short_h and short_score[i] <= exit_th:
            short_h = False

        s = 0
        sc = 0.0
        if long_h and (not short_h or long_score[i] >= short_score[i]):
            s = 1
            sc = long_score[i]
        elif short_h and (not long_h or short_score[i] > long_score[i]):
            s = -1
            sc = short_score[i]
        if not tradable[i]:
            s = 0
            sc = 0.0
        sig[i] = s
        strength[i] = sc
    return sig, strength


def _select_decorrelated_params(
    m,
    sorted_candidates,
    top_k: int,
    corr_th: float = 0.7,
    pool_scan_limit: int = 300,
):
    """
    Greedy top-k selection with pairwise signal-correlation cap.
    """
    if not sorted_candidates:
        return []

    k = int(max(1, top_k))
    lim = int(max(k, pool_scan_limit))
    th = float(max(0.0, min(0.9999, corr_th)))

    pool = sorted_candidates[:lim]
    selected = []
    selected_sig = []

    for _, _, p in pool:
        sig, _ = _param_signal(m, p)
        x = sig.astype(np.float64)
        if np.std(x) < 1e-12:
            continue
        ok = True
        for y in selected_sig:
            ys = y.astype(np.float64)
            if np.std(ys) < 1e-12:
                continue
            c = float(np.corrcoef(x, ys)[0, 1])
            if np.isnan(c):
                c = 1.0
            if abs(c) > th:
                ok = False
                break
        if ok:
            selected.append(p)
            selected_sig.append(sig)
            if len(selected) >= k:
                break

    # fallback fill if strict decorrelation couldn't fill top_k
    if len(selected) < k:
        selected_ids = {id(p) for p in selected}
        for _, _, p in sorted_candidates:
            if id(p) in selected_ids:
                continue
            selected.append(p)
            selected_ids.add(id(p))
            if len(selected) >= k:
                break

    return selected[:k]

def _apply_feature_template(p: dict, template_name: str) -> dict:
    q = copy.deepcopy(p)
    keep = set(FEATURE_TEMPLATES.get(template_name, ()))
    for k in WEIGHT_KEYS:
        if k not in keep:
            q[k] = 0.0
    q["_template"] = template_name
    return q

def _select_template_diverse_params(
    sorted_candidates,
    top_k: int,
    min_distinct: int,
    max_share: float,
):
    """
    Enforce template diversity in final top-k params.
    """
    if not sorted_candidates:
        return []
    k = int(max(1, top_k))
    need = int(max(1, min(min_distinct, k)))
    max_share = float(max(0.2, min(1.0, max_share)))
    cap = max(1, int(np.floor(k * max_share)))

    # Build template buckets preserving score order.
    buckets = {}
    for row in sorted_candidates:
        p = row[2]
        t = str(p.get("_template", "NONE"))
        buckets.setdefault(t, []).append(row)

    # Seed with top templates by best score.
    tmpl_rank = sorted(
        [(rows[0][0], t) for t, rows in buckets.items() if len(rows) > 0],
        key=lambda x: x[0],
        reverse=True,
    )
    seed_templates = [t for _, t in tmpl_rank[:need]]

    selected_rows = []
    tcount = {}
    used_ids = set()

    for t in seed_templates:
        row = buckets[t][0]
        selected_rows.append(row)
        tcount[t] = 1
        used_ids.add(id(row[2]))

    # Fill remainder by global rank under per-template cap.
    for row in sorted_candidates:
        if len(selected_rows) >= k:
            break
        p = row[2]
        pid = id(p)
        if pid in used_ids:
            continue
        t = str(p.get("_template", "NONE"))
        if tcount.get(t, 0) >= cap:
            continue
        selected_rows.append(row)
        tcount[t] = tcount.get(t, 0) + 1
        used_ids.add(pid)

    # If strict cap blocked fill, relax cap.
    if len(selected_rows) < k:
        for row in sorted_candidates:
            if len(selected_rows) >= k:
                break
            p = row[2]
            pid = id(p)
            if pid in used_ids:
                continue
            selected_rows.append(row)
            used_ids.add(pid)

    params = [r[2] for r in selected_rows[:k]]

    # Final safety: if diversity still not met, force-swap from missing templates.
    cur_templates = [str(p.get("_template", "NONE")) for p in params]
    cur_set = set(cur_templates)
    if len(cur_set) < need:
        missing = [t for _, t in tmpl_rank if t not in cur_set]
        for mt in missing:
            if len(set(cur_templates)) >= need:
                break
            # replace the most frequent template tail
            freq = {}
            for t in cur_templates:
                freq[t] = freq.get(t, 0) + 1
            replace_t = max(freq.items(), key=lambda x: x[1])[0]
            rep_idx = max(i for i, t in enumerate(cur_templates) if t == replace_t)
            candidate = buckets.get(mt, [None])[0]
            if candidate is None:
                continue
            params[rep_idx] = candidate[2]
            cur_templates[rep_idx] = mt

    return params


def _ensemble_backtest(
    m,
    params,
    min_votes,
    *,
    exit_on_hold: bool = True,
    state_cfg: dict | None = None,
    vote_cfg: dict | None = None,
):
    close = m["close"].to_numpy(np.float64)
    high = m["high"].to_numpy(np.float64)
    low = m["low"].to_numpy(np.float64)

    sigs = []
    strengths = []
    for p in params:
        s, st = _param_signal(m, p)
        sigs.append(s)
        strengths.append(st)
    sigs = np.stack(sigs)
    strengths = np.stack(strengths)

    # execution defaults from robust median
    def med(k):
        return float(np.median([p[k] for p in params]))

    lev = med("lev")
    maker_fee = 0.0002
    taker_fee = 0.0005
    slip = med("slip")
    tp = med("tp")
    sl = med("sl")
    trail = med("trail")
    trail_tox_a = med("trail_tox_a")
    trail_qc_b = med("trail_qc_b")
    trail_max_mult = med("trail_max_mult")
    cooldown_set = int(round(med("cooldown")))

    tox = m["shadow_toxicity_score"].to_numpy(np.float64)
    qcol = m["shadow_queue_collapse"].to_numpy(np.float64)

    cfg = dict(state_cfg or {})
    vcfg = dict(vote_cfg or {})
    soft_vote = bool(vcfg.get("soft_vote", False))
    soft_entry = float(vcfg.get("soft_entry", 0.62))
    soft_exit = float(vcfg.get("soft_exit", 0.45))
    soft_margin = float(vcfg.get("soft_margin", 0.05))
    ntz_margin = float(vcfg.get("ntz_margin", 0.0))  # no-trade zone on score gap
    protective_exit_v2 = bool(vcfg.get("protective_exit_v2", False))
    weak_support_bars = int(cfg.get("weak_support_bars", 0))    # 0이면 비활성
    keep_votes_min = int(cfg.get("keep_votes_min", max(1, min_votes - 2)))
    hold_loss_cut = float(cfg.get("hold_loss_cut", 0.0))        # 0이면 비활성

    eq = 1.0
    eq_curve = [eq]
    pos = 0
    size = 0.0
    entry = 0.0
    peak_px = 0.0
    trough_px = 0.0
    trades = 0
    wins = 0
    cooldown = 0
    bars_in_pos = 0
    weak_streak = 0

    for i in range(1, len(close)):
        if cooldown > 0:
            cooldown -= 1

        long_votes = int((sigs[:, i] == 1).sum())
        short_votes = int((sigs[:, i] == -1).sum())

        l_idx = np.where(sigs[:, i] == 1)[0]
        s_idx = np.where(sigs[:, i] == -1)[0]
        ls_avg = float(np.mean(strengths[l_idx, i])) if len(l_idx) else 0.0
        ss_avg = float(np.mean(strengths[s_idx, i])) if len(s_idx) else 0.0

        sig = 0
        if soft_vote:
            if pos == 0:
                if ls_avg >= soft_entry and (ls_avg - ss_avg) >= soft_margin:
                    sig = 1
                elif ss_avg >= soft_entry and (ss_avg - ls_avg) >= soft_margin:
                    sig = -1
            elif pos == 1:
                if ss_avg >= soft_entry and (ss_avg - ls_avg) >= soft_margin:
                    sig = -1
                elif ls_avg >= soft_exit and (ls_avg - ss_avg) >= soft_margin * 0.5:
                    sig = 1
                else:
                    sig = 0
            else:
                if ls_avg >= soft_entry and (ls_avg - ss_avg) >= soft_margin:
                    sig = 1
                elif ss_avg >= soft_exit and (ss_avg - ls_avg) >= soft_margin * 0.5:
                    sig = -1
                else:
                    sig = 0
        else:
            if long_votes >= min_votes and long_votes > short_votes:
                sig = 1
            elif short_votes >= min_votes and short_votes > long_votes:
                sig = -1

        # no-trade zone: if directional edge is too small, force HOLD
        if abs(ls_avg - ss_avg) < ntz_margin:
            sig = 0

        if sig == 1:
            st = ls_avg
        elif sig == -1:
            st = ss_avg
        else:
            st = 0.0

        target_size = max(0.0, min(1.0, st))

        if pos == 0 and sig != 0 and target_size > 0 and cooldown == 0:
            pos = sig
            size = target_size
            entry = close[i] * (1 + slip if pos == 1 else 1 - slip)
            eq *= (1.0 - taker_fee * size * lev)
            trades += 1
            peak_px = entry
            trough_px = entry
            bars_in_pos = 0
            weak_streak = 0
        elif pos != 0:
            bars_in_pos += 1
            if pos == 1:
                peak_px = max(peak_px, high[i])
            else:
                trough_px = min(trough_px, low[i])

            rr_m = (close[i] - entry) / max(entry, 1e-12)
            if pos == -1:
                rr_m = -rr_m

            dyn_gap = trail * min(1.0 + trail_tox_a * tox[i], trail_max_mult) * min(1.0 + trail_qc_b * qcol[i], trail_max_mult)
            hit_trail = (close[i] <= peak_px * (1 - dyn_gap)) if pos == 1 else (close[i] >= trough_px * (1 + dyn_gap))
            hit_tp = rr_m >= tp
            hit_sl = rr_m <= -sl

            # 상태머신 강화: HOLD에서 무한 보유 방지용 보조 종료 규칙
            side_votes = long_votes if pos == 1 else short_votes
            if sig == 0 and side_votes < keep_votes_min:
                weak_streak += 1
            else:
                weak_streak = 0

            weak_support_exit = (not exit_on_hold) and (weak_support_bars > 0) and (sig == 0) and (weak_streak >= weak_support_bars)
            hold_loss_exit = (not exit_on_hold) and (hold_loss_cut > 0.0) and (sig == 0) and (rr_m <= -abs(hold_loss_cut))
            stress_exit = False
            if protective_exit_v2:
                stress_exit = bool(
                    rr_m < 0.0
                    and (
                        (tox[i] > 0.72)
                        or (qcol[i] > 0.68)
                    )
                )

            should_exit = (
                hit_trail
                or hit_tp
                or hit_sl
                or sig == -pos
                or (exit_on_hold and sig == 0)
                or weak_support_exit
                or hold_loss_exit
                or stress_exit
            )
            if should_exit:
                exit_px = close[i] * (1 - slip if pos == 1 else 1 + slip)
                rr = (exit_px - entry) / max(entry, 1e-12)
                if pos == -1:
                    rr = -rr
                pnl = rr * size * lev
                eq *= (1.0 + pnl)
                eq *= (1.0 - taker_fee * size * lev)
                wins += int(pnl > 0)
                pos = 0
                size = 0.0
                entry = 0.0
                cooldown = cooldown_set
                bars_in_pos = 0
                weak_streak = 0
        eq_curve.append(eq)

    eqa = np.array(eq_curve, dtype=np.float64)
    pnl_pct = float((eqa[-1] - 1.0) * 100.0)
    peak = np.maximum.accumulate(eqa)
    mdd = float((eqa / np.maximum(peak, 1e-12) - 1.0).min() * 100.0)
    r = np.diff(eqa) / np.maximum(eqa[:-1], 1e-12)
    sharpe = float((r.mean() / (r.std() + 1e-12)) * np.sqrt(365 * 24 * 12)) if len(r) else 0.0
    wr = float(wins / trades * 100.0) if trades > 0 else 0.0

    return {
        "pnl_pct": pnl_pct,
        "mdd_pct": mdd,
        "trades": int(trades),
        "win_rate": wr,
        "sharpe": sharpe,
        "equity": float(eqa[-1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=36500)
    ap.add_argument("--search-iters", type=int, default=40000)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--min-votes", type=int, default=6)
    ap.add_argument("--decorrelate-topk", action="store_true")
    ap.add_argument("--corr-th", type=float, default=0.7)
    ap.add_argument("--corr-pool-limit", type=int, default=300)
    ap.add_argument("--feature-prune-3", action="store_true")
    ap.add_argument("--template-min-distinct", type=int, default=0, help="minimum distinct templates in selected top-k")
    ap.add_argument("--template-max-share", type=float, default=0.6, help="max share per template in selected top-k")
    ap.add_argument("--soft-vote", action="store_true")
    ap.add_argument("--soft-entry", type=float, default=0.62)
    ap.add_argument("--soft-exit", type=float, default=0.45)
    ap.add_argument("--soft-margin", type=float, default=0.05)
    ap.add_argument("--ntz-margin", type=float, default=0.0)
    ap.add_argument("--formula-v2", action="store_true")
    ap.add_argument("--protective-exit-v2", action="store_true")
    ap.add_argument("--use-valid-select", action="store_true")
    ap.add_argument("--valid-ratio", type=float, default=0.2)
    ap.add_argument("--valid-score-mdd-penalty", type=float, default=0.5)
    ap.add_argument("--regime-split", action="store_true")
    ap.add_argument("--compare-state-machine", action="store_true")
    ap.add_argument("--exit-on-hold", action="store_true")
    ap.add_argument("--tune-state-machine", action="store_true")
    ap.add_argument("--sm-iters", type=int, default=200)
    ap.add_argument("--sm-min-trades", type=int, default=0)
    ap.add_argument("--sm-max-mdd", type=float, default=0.0, help="e.g. 6.0 means require MDD >= -6.0%")
    ap.add_argument("--sm-trade-target", type=int, default=30)
    ap.add_argument("--sm-mdd-penalty", type=float, default=0.12)
    ap.add_argument("--sm-trade-reward", type=float, default=0.20)
    ap.add_argument("--sm-overtrade-penalty", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=20260414)
    ap.add_argument("--out", default="data/ensemble/metrics/param_ensemble_result.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    m = load_merged("binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv", args.days)
    vratio = float(max(0.05, min(0.45, args.valid_ratio)))
    split_idx = int(len(m) * (1.0 - vratio))
    m_train = m.iloc[:split_idx].reset_index(drop=True)
    m_valid = m.iloc[split_idx:].reset_index(drop=True)

    cands = []
    templates = list(FEATURE_TEMPLATES.keys())
    for _ in range(args.search_iters):
        p = sample_params(rng)
        # activity bias for ensemble pool
        p["cooldown"] = rng.randint(2, 30)
        p["entry"] = rng.uniform(0.50, 0.84)
        p["exit"] = rng.uniform(0.22, 0.60)
        p["volr_min"] = rng.uniform(0.15, 0.85)
        p["vpin_max"] = rng.uniform(0.80, 0.999)
        p["lev"] = rng.uniform(2.5, 10.0)
        if p["exit"] >= p["entry"]:
            p["exit"] = max(0.2, p["entry"] - 0.06)
        if args.formula_v2:
            p["_formula_v2"] = True
        if args.feature_prune_3:
            tname = templates[rng.randint(0, len(templates) - 1)]
            p = _apply_feature_template(p, tname)

        base_df = m_train if args.use_valid_select else m
        r = run_sim(base_df, p)
        # keep feasible pool only
        if r.mdd_pct < -8.0 or r.trades < 8:
            continue
        if args.use_valid_select:
            rv = run_sim(m_valid, p)
            score = rv.pnl_pct + 0.20 * rv.sharpe - float(args.valid_score_mdd_penalty) * abs(rv.mdd_pct)
        else:
            score = r.pnl_pct + 0.25 * r.sharpe - 0.15 * abs(r.mdd_pct)
        cands.append((score, r, p))

    cands.sort(key=lambda x: x[0], reverse=True)
    top = cands[: max(args.top_k, 10)]
    if args.decorrelate_topk:
        params = _select_decorrelated_params(
            m=m,
            sorted_candidates=cands,
            top_k=args.top_k,
            corr_th=float(args.corr_th),
            pool_scan_limit=int(args.corr_pool_limit),
        )
    else:
        params = [x[2] for x in top[: args.top_k]]

    # optional regime split: mix low-vol and high-vol parameter styles
    if args.regime_split and len(params) >= 4:
        low = [p for p in params if float(p.get("atr_min", 0.0)) <= 0.006]
        high = [p for p in params if float(p.get("atr_min", 0.0)) > 0.006]
        k = int(args.top_k)
        mixed = []
        li = hi = 0
        while len(mixed) < k and (li < len(low) or hi < len(high)):
            if li < len(low):
                mixed.append(low[li]); li += 1
                if len(mixed) >= k:
                    break
            if hi < len(high):
                mixed.append(high[hi]); hi += 1
        if len(mixed) >= max(4, k // 2):
            params = mixed[:k]

    # strict template diversity on final top-k (for feature-pruned mode)
    if args.feature_prune_3 and (int(args.template_min_distinct) > 0):
        diverse_params = _select_template_diverse_params(
            sorted_candidates=cands,
            top_k=int(args.top_k),
            min_distinct=int(args.template_min_distinct),
            max_share=float(args.template_max_share),
        )
        # keep decorrelated set only if it also satisfies minimum diversity
        if args.decorrelate_topk:
            dset = set(str(x.get("_template", "NONE")) for x in params)
            if len(dset) < int(args.template_min_distinct):
                params = diverse_params
        else:
            params = diverse_params

    if args.compare_state_machine:
        ens_exit_on_hold = _ensemble_backtest(
            m, params, min_votes=args.min_votes, exit_on_hold=True,
            vote_cfg={
                "soft_vote": bool(args.soft_vote),
                "soft_entry": float(args.soft_entry),
                "soft_exit": float(args.soft_exit),
                "soft_margin": float(args.soft_margin),
                "ntz_margin": float(args.ntz_margin),
                "protective_exit_v2": bool(args.protective_exit_v2),
            },
        )
        ens_hold_state = _ensemble_backtest(
            m, params, min_votes=args.min_votes, exit_on_hold=False,
            vote_cfg={
                "soft_vote": bool(args.soft_vote),
                "soft_entry": float(args.soft_entry),
                "soft_exit": float(args.soft_exit),
                "soft_margin": float(args.soft_margin),
                "ntz_margin": float(args.ntz_margin),
                "protective_exit_v2": bool(args.protective_exit_v2),
            },
        )
        ens = dict(ens_hold_state)
    else:
        ens = _ensemble_backtest(
            m,
            params,
            min_votes=args.min_votes,
            exit_on_hold=bool(args.exit_on_hold),
            vote_cfg={
                "soft_vote": bool(args.soft_vote),
                "soft_entry": float(args.soft_entry),
                "soft_exit": float(args.soft_exit),
                "soft_margin": float(args.soft_margin),
                "ntz_margin": float(args.ntz_margin),
                "protective_exit_v2": bool(args.protective_exit_v2),
            },
        )

    sm_tuned = None
    if args.tune_state_machine:
        best = None
        for _ in range(max(1, int(args.sm_iters))):
            cfg = {
                "weak_support_bars": rng.randint(2, 18),
                "keep_votes_min": rng.randint(1, max(1, args.min_votes)),
                "hold_loss_cut": rng.uniform(0.002, 0.02),
            }
            r_sm = _ensemble_backtest(
                m,
                params,
                min_votes=args.min_votes,
                exit_on_hold=False,
                state_cfg=cfg,
                vote_cfg={
                    "soft_vote": bool(args.soft_vote),
                    "soft_entry": float(args.soft_entry),
                    "soft_exit": float(args.soft_exit),
                    "soft_margin": float(args.soft_margin),
                    "ntz_margin": float(args.ntz_margin),
                    "protective_exit_v2": bool(args.protective_exit_v2),
                },
            )
            if int(args.sm_min_trades) > 0 and int(r_sm["trades"]) < int(args.sm_min_trades):
                continue
            if float(args.sm_max_mdd) > 0 and float(r_sm["mdd_pct"]) < -abs(float(args.sm_max_mdd)):
                continue

            # 사용자 정의 목적함수: MDD 축소 + 거래수 증가 + PnL 유지
            score = (
                r_sm["pnl_pct"]
                - float(args.sm_mdd_penalty) * abs(r_sm["mdd_pct"])
                + float(args.sm_trade_reward) * min(r_sm["trades"], int(args.sm_trade_target))
                - float(args.sm_overtrade_penalty) * max(0, r_sm["trades"] - 80)
            )
            if (best is None) or (score > best["score"]):
                best = {"score": float(score), "cfg": cfg, "result": r_sm}
        if best is not None:
            sm_tuned = best
            ens = dict(best["result"])

    out = {
        "dataset": {
            "rows": int(len(m)),
            "start": str(m["ts"].min()),
            "end": str(m["ts"].max()),
        },
        "search": {
            "iters": args.search_iters,
            "pool_size": len(cands),
            "top_k": args.top_k,
            "min_votes": args.min_votes,
            "decorrelate_topk": bool(args.decorrelate_topk),
            "corr_th": float(args.corr_th),
            "corr_pool_limit": int(args.corr_pool_limit),
            "effective_top_k": int(len(params)),
            "feature_prune_3": bool(args.feature_prune_3),
            "template_min_distinct": int(args.template_min_distinct),
            "template_max_share": float(args.template_max_share),
            "use_valid_select": bool(args.use_valid_select),
            "valid_ratio": float(vratio),
            "valid_score_mdd_penalty": float(args.valid_score_mdd_penalty),
            "regime_split": bool(args.regime_split),
            "soft_vote": bool(args.soft_vote),
            "soft_entry": float(args.soft_entry),
            "soft_exit": float(args.soft_exit),
            "soft_margin": float(args.soft_margin),
            "ntz_margin": float(args.ntz_margin),
            "formula_v2": bool(args.formula_v2),
            "protective_exit_v2": bool(args.protective_exit_v2),
        },
        "ensemble_result": ens,
        "execution_mode": {
            "exit_on_hold": (None if args.compare_state_machine else bool(args.exit_on_hold)),
            "compared": bool(args.compare_state_machine),
        },
        "comparison": (
            {
                "exit_on_hold": ens_exit_on_hold,
                "hold_state_machine": ens_hold_state,
                "delta_hold_minus_exit": {
                    "pnl_pct": float(ens_hold_state["pnl_pct"] - ens_exit_on_hold["pnl_pct"]),
                    "trades": int(ens_hold_state["trades"] - ens_exit_on_hold["trades"]),
                    "mdd_pct": float(ens_hold_state["mdd_pct"] - ens_exit_on_hold["mdd_pct"]),
                    "sharpe": float(ens_hold_state["sharpe"] - ens_exit_on_hold["sharpe"]),
                },
            }
            if args.compare_state_machine
            else None
        ),
        "state_machine_tuning": sm_tuned,
        "top_params": [
            {
                "rank": i + 1,
                "score": float(top[i][0]),
                "single": {
                    "pnl_pct": top[i][1].pnl_pct,
                    "mdd_pct": top[i][1].mdd_pct,
                    "trades": top[i][1].trades,
                    "win_rate": top[i][1].win_rate,
                    "sharpe": top[i][1].sharpe,
                },
                "params": top[i][2],
            }
            for i in range(min(args.top_k, len(top)))
        ],
        "selected_params": [
            {
                "rank": i + 1,
                "params": p,
            }
            for i, p in enumerate(params)
        ],
    }

    Path("data/ensemble/metrics").mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    printed = {
        "pool_size": len(cands),
        "ensemble": ens,
        "out": args.out,
    }
    if args.compare_state_machine:
        printed["comparison"] = out["comparison"]
    if args.tune_state_machine:
        printed["state_machine_tuning"] = sm_tuned
    print(json.dumps(printed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
