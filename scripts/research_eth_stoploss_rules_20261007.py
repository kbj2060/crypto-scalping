"""물타기 전략의 손절선 — 어디에 두나 (2026-10-07, 사용자 «물타기하다가 청산 당하게 생겼어. 손절라인을 어떻게 잡는게 좋을지»).

계기: 10-07 13:15 UTC ETHUSDC 롱 15.28 @2655.9, 순자산 ≈$1,175, 명목 33배, 청산가까지 1.04%.
이미 닫힌 것(재실행 안 함): 원장 고정 bp 스톱 −50~−300 은 전부 손해(09-27) · 물타기 veto 게이트 = 무작위(10-04) ·
  레짐으로 물타기 꼬리 고르기 0(10-07) · 꼬리의 정체 = 물타기 명목 팽창.
새로 묻는 것: (1) 선을 «무엇 기준»으로 긋나 — 첫 진입가 / 평균단가 / 변동성 배수 / 계좌 손실 % / 청산 거리
              (2) 선이 정해지면 크기를 어떻게 묶어야 청산이 원리적으로 안 나나.

── 결과 보기 전 고정 (2026-10-07) ────────────────────────────────────────────────────────────────
T 시장(주): pingpong_levers V0 와 같은 기계 — 핑퐁 페이드 후보(2022-01~2026-08) · 1단위 진입 · 첫 진입가 대비 44bp 마다 1단위 물타기
   (최대 NADD) · 평균단가 +20bp 익절 · 24h 시간 청산 · 순차 한 포지션. 여기에 손절을 더한다:
   none · F{100,150,200,300,500}(첫 진입가 대비 bp) · A{100,200}(평균단가 대비 bp) · V{0.5,0.75,1,1.5}(× 직전 24h 실현변동성, 첫 진입가 대비)
   × NADD{0,1,3}. 같은 봉에서 물타기 레벨과 손절이 같이 닿으면 물타기 먼저(보수 = 손실 큼) · 손절 체결 = 선 가격(봉 전체가 선 밖이면
   그 봉 종가) + 테이커 5bp/단위. 익절·물타기 = 지정가 0bp. 단위-bp = 첫 진입 1단위 명목 대비.
   주 판정 = TEST 2025-01~ · 보조 = 2022-01~2024-12(엇갈리면 최근 우선, 사용자 10-04).
   «손절이 돕는다» = 일 샤프 차(규칙 − none, 같은 NADD) 일 블록 부트스트랩 95% CI 가 0 배제 양수. «위험만 준다» = CVaR1% 개선 + 샤프 차 CI 하한 > −0.3.
   크기 환산(보고): 각 규칙의 TEST 최악 손실(단위-bp)로 «1단위 명목 ≤ 순자산 × r / 최악» (r = 10·20%) — 청산 없는 최대 크기.
L 원장(보조, 체결 단위 반사실): 같은 손절군 + E{5,10,20,30}(왕복 손실 ≥ 진입 시 지갑 x%) + Q{3,5}(청산가까지 거리 ≤ d%, MMR 0.5% 근사·
   다른 동시 포지션 무시) · 발동하면 그 왕복 남은 체결 무시하고 선 가격에 테이커 5bp 로 전량 청산 · 다른 왕복은 실제 그대로(복리 없음).
   열린 왕복은 마지막 1분 종가로 평가(생존 편향 방지). 일(진입일) 클러스터 부트스트랩 4,000회로 «규칙 − 실제» 총손익 CI.
C 조합(같은 날 2차, 사용자 «그 조합으로 돌려줘» -- 결과 보기 전 고정): 물타기 3회 · 크기는 판마다 «그 구간 선택 거래 평균 = 1»로 정규화.
   B0 고정 간격·손절 없음 · B1 고정 간격 + F300 · S0 크기 ∝ σ · S1 크기 ∝ σ + F300 · G0 간격·익절 ∝ σ(pingpong V3: k = σ/TRAIN 중앙,
   D = max(15, 44k)·TP = max(5, 20k)) · G1 G0 + 손절 300k bp(같은 비율) · SG1 크기·간격·손절 모두 ∝ σ.
   크기 ∝ σ = 사전등록(prereg_eth_pingpong_vol_sizing_fwd) 식 clip((σ24/√6)/157.394, .25, 3)/.76576 (σ24 = 위 일 척도).
   판정: TEST 일 샤프 차(판 − B1) 일 블록 95% CI 가 0 배제 양수 = «B1 보다 낫다» · AUX 같은 부호 = 보조. 함께: 판 − B0.
   보고: 건당·최악·CVaR1%·MDD(정규화 단위) · «최악 손실 ≤ 순자산 10%» 가 되는 평균 1단위 명목(순자산 배수).
   원장(보조): S1 = 왕복 F300 반사실 × 진입 시 크기 배수(왕복 평균 1 정규화) vs F300 · 실제 -- 총액·최악·왕복순 누적 MDD · 일 클러스터 CI.
실행: python scripts/research_eth_stoploss_rules_20261007.py --tape | --ledger | --combo | --selftest   산출 tmp/stoploss_rules_20261007/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_eth_trend30_60_hold_policy_20261007 import PREV, TR_END, VA_END, load_1m, ms, to_5m  # noqa: E402
import research_eth_avgdown_gate_notional_cap_20261004 as AG  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/stoploss_rules_20261007"
D0, TP0, HOLD, STOP_FEE = 44.0, 20.0, 1440, 5.0
SEED, N_BOOT = 20261007, 2000
RULES_T = [("none", 0, 0.0)] + [(f"F{k}", 1, k) for k in (100, 150, 200, 300, 500)] + [(f"A{k}", 2, k) for k in (100, 200)] \
    + [(f"V{m}", 3, m) for m in (0.5, 0.75, 1.0, 1.5)]


@numba.njit(cache=False)
def sim_stop(h, l, c, ei, side, nadd_max, H, mode, S, D, TP):
    """V0 + 손절(물타기 간격 D[k]·익절 TP[k] bp). mode 0 없음 · 1 첫 진입가 대비 S[k]bp · 2 평균단가 대비 S[k]bp. (V 는 S 에 σ 배수를 넣어 mode 1).
    반환: 단위-bp 손익(비용 전) · 단위 수 · 청산 인덱스 · 종류(1 익절 2 시간 3 손절)."""
    n = len(ei)
    pnl = np.zeros(n); q_out = np.zeros(n); xi = np.zeros(n, np.int64); kind = np.zeros(n, np.int64)
    for k in range(n):
        i = ei[k]; s = side[k]; p0 = c[i]
        q = 1.0; cost = p0; nadd = 0
        end = min(i + H, len(c) - 1)
        exit_px = c[end]; kind[k] = 2; xi[k] = end
        for j in range(i + 1, end + 1):
            added = False
            while nadd < nadd_max:
                lvl = p0 * (1 - s * D[k] * (nadd + 1) / 1e4)
                if (s > 0 and l[j] <= lvl) or (s < 0 and h[j] >= lvl):
                    q += 1.0; cost += lvl; nadd += 1; added = True
                else:
                    break
            if mode > 0:
                ref = p0 if mode == 1 else cost / q
                stop = ref * (1 - s * S[k] / 1e4)
                if (s > 0 and l[j] <= stop) or (s < 0 and h[j] >= stop):
                    gap = (s > 0 and h[j] < stop) or (s < 0 and l[j] > stop)
                    exit_px = c[j] if gap else stop; kind[k] = 3; xi[k] = j
                    break
            if added:
                continue
            tgt = cost / q * (1 + s * TP[k] / 1e4)
            if (s > 0 and h[j] >= tgt) or (s < 0 and l[j] <= tgt):
                exit_px = tgt; kind[k] = 1; xi[k] = j
                break
        pnl[k] = s * (exit_px * q - cost) / p0 * 1e4
        q_out[k] = q
    return pnl, q_out, xi, kind


def sequential(ei, xi):
    out = np.zeros(len(ei), bool); last = -1
    for k in np.argsort(ei, kind="stable"):
        if ei[k] > last:
            out[k] = True; last = xi[k]
    return out


def day_sharpe(d: pd.Series) -> float:
    return float(d.mean() / d.std() * np.sqrt(365)) if d.std() > 0 else float("nan")


def boot_diff(a: pd.Series, b: pd.Series, rng) -> list[float]:
    D = pd.concat([a, b], axis=1).fillna(0.0).to_numpy()
    S = D[rng.integers(0, len(D), (N_BOOT, len(D)))]
    sh = S.mean(1) / S.std(1) * np.sqrt(365)
    return [float(np.quantile(sh[:, 0] - sh[:, 1], q)) for q in (0.025, 0.975)]


def _reached(h, l, j0, j1, s, p0, nadd):
    """손절 뒤 24h 안에 «물타기 다 찬 평균단가 + 20bp»에 닿았나(손절 안 했으면 익절됐을 가능성의 상한)."""
    n = 1 + nadd
    avg = sum(p0 * (1 - s * D0 * i / 1e4) for i in range(n)) / n
    tgt = avg * (1 + s * TP0 / 1e4)
    return bool((h[j0 + 1:j1 + 1] >= tgt).any()) if s > 0 else bool((l[j0 + 1:j1 + 1] <= tgt).any())


def tape_inputs():
    k1 = load_1m(); b = to_5m(k1)
    t5, c5 = b.t.to_numpy(), b.c.to_numpy()
    r5 = np.diff(np.log(c5), prepend=np.nan) * 1e4
    sig24 = np.sqrt(pd.Series(r5 ** 2).rolling(288, min_periods=280).sum().to_numpy())     # 직전 24h 실현변동성(bp, 일 척도)
    C = pd.read_parquet(PREV / "tmp/pingpong_extract_20261007/candidates.parquet").sort_values("t")
    t1, c1, h1, l1 = (k1[x].to_numpy() for x in ("t", "c", "h", "l"))
    tdec = C.t.to_numpy("int64") + 60_000
    ei = np.searchsorted(t1, tdec)
    bi = np.searchsorted(t5, tdec // 300_000 * 300_000 - 300_000)          # 결정 전에 닫힌 마지막 5분봉
    ok = (ei + HOLD < len(c1)) & (bi < len(t5)) & np.isfinite(sig24[np.minimum(bi, len(t5) - 1)])
    C, ei, bi, tdec = C[ok].reset_index(drop=True), ei[ok], bi[ok], tdec[ok]
    side = -C.d.to_numpy(np.int64); sg = sig24[bi]; day = tdec // 86_400_000
    per = np.where(tdec >= VA_END, "TEST", np.where(tdec >= ms("2022-01-01"), "AUX", "-"))
    print(f"후보 {len(C)} · TEST(2025~) {(per == 'TEST').sum()} · σ24 중앙 {np.median(sg):.0f}bp", flush=True)
    return h1, l1, c1, ei, side, sg, day, per, tdec


def run_tape() -> dict:
    rng = np.random.default_rng(SEED)
    h1, l1, c1, ei, side, sg, day, per, tdec = tape_inputs()
    DD, TT = np.full(len(ei), D0), np.full(len(ei), TP0)
    res, daily = {"sig24_median_test": float(np.median(sg[per == "TEST"]))}, {}
    for nadd in (0, 1, 3):
        for name, mode, v in RULES_T:
            S = sg * v if mode == 3 else np.full(len(ei), float(v))
            pnl, q, xi, kind = sim_stop(h1, l1, c1, ei, side, nadd, HOLD, 1 if mode == 3 else mode, S, DD, TT)
            sel = sequential(ei, xi)
            net = pnl - np.where(kind == 3, STOP_FEE * q, 0.0)
            key = f"N{nadd}·{name}"
            res[key] = {}
            for p in ("TEST", "AUX"):
                m = sel & (per == p)
                x = net[m]; d = pd.Series(x).groupby(day[m]).sum()
                daily[(key, p)] = d
                eq = np.cumsum(x); qq = np.quantile(x, 0.01)
                res[key][p] = dict(n=int(m.sum()), per_trade=float(x.mean()), win=float((x > 0).mean()), stop_rate=float((kind[m] == 3).mean()),
                                   timeout_rate=float((kind[m] == 2).mean()), worst=float(x.min()), cvar1=float(x[x <= qq].mean()),
                                   sharpe=day_sharpe(d), mdd=float((eq - np.maximum.accumulate(eq)).min()), total=float(eq[-1]))
            if mode in (1, 3):
                st = sel & (kind == 3) & (per == "TEST")
                res[key]["TEST"]["regret_reach_tp_24h"] = float(np.mean(
                    [_reached(h1, l1, xi[k], ei[k] + HOLD, side[k], c1[ei[k]], nadd) for k in np.flatnonzero(st)])) if st.any() else None
    for key in [k for k in res if "·" in k]:
        base = f"{key.split('·')[0]}·none"
        for p in ("TEST", "AUX"):
            if key != base:
                res[key][p]["sharpe_diff_ci"] = boot_diff(daily[(key, p)], daily[(base, p)], rng)
    return res


def vol_mult(sig24):
    """전진 사전등록과 같은 크기 배수(σ24 = 일 척도 → 4h 척도로 나눠 넣는다)."""
    return np.clip(sig24 / np.sqrt(6) / 157.394, 0.25, 3.0) / 0.76576


def run_combo() -> dict:
    rng = np.random.default_rng(SEED)
    h1, l1, c1, ei, side, sg, day, per, tdec = tape_inputs()
    n = len(ei)
    k = sg / np.median(sg[tdec < TR_END])
    Dk, TPk = np.maximum(15.0, D0 * k), np.maximum(5.0, TP0 * k)
    one, D_, T_ = np.ones(n), np.full(n, D0), np.full(n, TP0)
    vm = vol_mult(sg)
    arms = {"B0": (D_, T_, 0, np.zeros(n), one), "B1": (D_, T_, 1, np.full(n, 300.0), one),
            "S0": (D_, T_, 0, np.zeros(n), vm), "S1": (D_, T_, 1, np.full(n, 300.0), vm),
            "G0": (Dk, TPk, 0, np.zeros(n), one), "G1": (Dk, TPk, 1, 300.0 * k, one), "SG1": (Dk, TPk, 1, 300.0 * k, vm)}
    res, daily = {"k_median_test": float(np.median(k[per == "TEST"])), "k_p90_test": float(np.quantile(k[per == "TEST"], .9))}, {}
    for name, (D, TP, mode, S, w) in arms.items():
        pnl, q, xi, kind = sim_stop(h1, l1, c1, ei, side, 3, HOLD, mode, S, D, TP)
        sel = sequential(ei, xi)
        net = pnl - np.where(kind == 3, STOP_FEE * q, 0.0)
        res[name] = {}
        for p in ("TEST", "AUX"):
            m = sel & (per == p)
            x = net[m] * w[m] / w[m].mean()
            d = pd.Series(x).groupby(day[m]).sum(); daily[(name, p)] = d
            eq = np.cumsum(x); qq = np.quantile(x, 0.01)
            res[name][p] = dict(n=int(m.sum()), per_trade=float(x.mean()), win=float((x > 0).mean()), stop_rate=float((kind[m] == 3).mean()),
                                worst=float(x.min()), cvar1=float(x[x <= qq].mean()), sharpe=day_sharpe(d),
                                mdd=float((eq - np.maximum.accumulate(eq)).min()), total=float(eq[-1]),
                                unit_x_for_10pct=float(0.10 / (-x.min() / 1e4)) if x.min() < 0 else None)
    for name in arms:
        for p in ("TEST", "AUX"):
            for base in ("B1", "B0"):
                if name != base:
                    res[name][p][f"sharpe_diff_vs_{base}"] = boot_diff(daily[(name, p)], daily[(base, p)], rng)
    return res


def ledger_combo() -> dict:
    rng = np.random.default_rng(SEED)
    T = pd.read_csv(OUT / "ledger_trips.csv").sort_values("start").reset_index(drop=True)
    w = vol_mult(T.sig24.to_numpy()); w = w / np.nanmean(w)
    n_nan = int(np.isnan(w).sum()); w = np.where(np.isnan(w), 1.0, w)    # 첫날 왕복은 24h 봉이 모자라 σ 없음 → 중립 1
    arms = {"actual": T.actual.to_numpy(), "F300": T.F300.to_numpy(), "S1": T.F300.to_numpy() * w, "S0": T.actual.to_numpy() * w}
    days = T.day.to_numpy(); ud = np.unique(days)
    out = {"n": int(len(T)), "n_no_sigma": n_nan, "w_min_max": [float(w.min()), float(w.max())]}
    for k, x in arms.items():
        eq = np.cumsum(x)
        out[k] = dict(total=float(x.sum()), worst=float(x.min()), mdd=float((eq - np.maximum.accumulate(eq)).min()))
        for base in ("F300", "actual"):
            if k != base:
                d = pd.Series(x - arms[base]).groupby(days).sum().reindex(ud).to_numpy()
                bs = d[rng.integers(0, len(ud), (4000, len(ud)))].sum(1)
                out[k][f"diff_vs_{base}"] = [float(d.sum()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
    return out


# ── 원장 ─────────────────────────────────────────────────────────────────────────
def trip_cf(g, s, kt, kh, kl, kc, kind, v, E0, sg, is_open):
    """한 왕복을 분 단위로: 그 분의 체결을 반영 → 그 분 불리 극값으로 손절 판정. 반환 (손익$, 손절가, 손절 시각)."""
    T, P, Q, INC, CM = g.time.to_numpy(), g.price.to_numpy(), g.qty.to_numpy(), g.inc.to_numpy(), g.commission.to_numpy()
    q = cash = cost_q = 0.0; first = P[0]; fi = 0
    for j in range(len(kt)):
        while fi < len(T) and T[fi] < kt[j] + 60_000:
            if INC[fi]:
                cost_q += P[fi] * Q[fi]; dq = Q[fi]
            else:
                cost_q -= (cost_q / q) * Q[fi] if q > 1e-12 else 0.0; dq = -Q[fi]
            cash -= s * dq * P[fi] + CM[fi]; q += dq; fi += 1
            if q < 1e-9:
                q = cost_q = 0.0
        if kind is None or q < 1e-9:
            continue
        avg = cost_q / q
        if kind == "F":
            stop = first * (1 - s * v / 1e4)
        elif kind == "A":
            stop = avg * (1 - s * v / 1e4)
        elif kind == "V":
            stop = first * (1 - s * v * sg / 1e4)
        elif kind == "E":                            # 왕복 손익 cash + s·q·p = −x%·E0 인 가격
            stop = (-v / 100 * E0 - cash) / (s * q)
        else:                                        # Q: (E0 + cash + s·q·p − mmr·q·p)/(q·p) = d  →  p = (E0+cash)/(q(d+mmr) − s·q)
            denom = q * (v / 100 + AG.MMR) - s * q
            stop = (E0 + cash) / denom if abs(denom) > 1e-12 else np.nan
        if np.isfinite(stop) and ((s > 0 and kl[j] <= stop) or (s < 0 and kh[j] >= stop)):
            gap = (s > 0 and kh[j] < stop) or (s < 0 and kl[j] > stop)
            px = kc[j] if gap else stop
            return cash + s * q * px - STOP_FEE / 1e4 * q * px, float(px), int(kt[j])
    for k in range(fi, len(T)):                      # 분봉 끝 뒤 체결
        dq = Q[k] if INC[k] else -Q[k]; cash -= s * dq * P[k] + CM[k]; q += dq
    return cash + (s * q * kc[-1] if is_open and q > 1e-9 else 0.0), None, None


def run_ledger() -> dict:
    rng = np.random.default_rng(SEED)
    f = AG.tag_events(AG.load_fills())
    inc = pd.DataFrame([json.loads(x) for x in open(AG.OUT / "income_full.jsonl")])
    wt, wc, _ = AG.flows_wallet(inc)
    t0 = int(f.time.min())
    k = AG.load_k1m(pd.Timestamp(t0, unit="ms").strftime("%Y-%m-%d"), pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    kt, kh, kl, kc = (k[x].to_numpy() for x in ("t", "h", "l", "c"))
    b5 = k.assign(b=k.t // 300_000 * 300_000).groupby("b").c.last()
    r5 = np.diff(np.log(b5.to_numpy()), prepend=np.nan) * 1e4
    sig_t, sig_v = b5.index.to_numpy(), np.sqrt(pd.Series(r5 ** 2).rolling(288, min_periods=280).sum().to_numpy())
    rules = [("actual", None, 0)] + [(f"F{v}", "F", v) for v in (100, 150, 200, 300, 500)] + [(f"A{v}", "A", v) for v in (100, 200)] \
        + [(f"V{v}", "V", v) for v in (0.5, 0.75, 1.0, 1.5)] + [(f"E{v}", "E", v) for v in (5, 10, 20, 30)] + [(f"Q{v}", "Q", v) for v in (3, 5)]
    trips = []
    for tr, g in f.groupby("trip"):
        g = g.sort_values(["time", "id"])
        s = int(g.sgn.iloc[0]); ts = int(g.time.iloc[0])
        qs = np.cumsum(np.where(g.inc, g.qty, -g.qty))
        is_open = qs[-1] > 1e-6
        te = int(kt[-1]) if is_open else int(g.time.iloc[-1])
        E0 = float(wc[np.searchsorted(wt, ts, side="right") - 1])
        si = np.searchsorted(sig_t, ts // 300_000 * 300_000 - 300_000)
        sg = float(sig_v[min(si, len(sig_v) - 1)])
        a, bix = np.searchsorted(kt, ts // 60_000 * 60_000), np.searchsorted(kt, te, side="right")
        row = {"trip": int(tr), "symbol": g.symbol.iloc[0], "side": "LONG" if s > 0 else "SHORT", "start": pd.Timestamp(ts, unit="ms").isoformat(),
               "open": bool(is_open), "E0": E0, "sig24": sg, "first_px": float(g.price.iloc[0]), "n_fills": int(len(g)),
               "n_adds": int(g.ev.max()), "max_notional": float((qs * g.price.to_numpy()).max()), "day": ts // 86_400_000}
        row["max_lev"] = row["max_notional"] / E0 if E0 > 0 else None
        for name, kind, v in rules:
            pnl, px, ht = trip_cf(g, s, kt[a:bix], kh[a:bix], kl[a:bix], kc[a:bix], kind, v, E0, sg, is_open)
            row[name] = pnl
            if kind:
                row[f"{name}_px"] = px; row[f"{name}_hit"] = None if ht is None else pd.Timestamp(ht, unit="ms").isoformat()
        trips.append(row)
    T = pd.DataFrame(trips)
    T.to_csv(OUT / "ledger_trips.csv", index=False)
    out = {"n_trips": int(len(T)), "first": T.start.min(), "actual_total": float(T.actual.sum()), "actual_worst": float(T.actual.min()),
           "open": T[T.open].to_dict("records"), "rules": {}}
    days = T.day.to_numpy(); ud = np.unique(days)
    for name, kind, v in rules[1:]:
        d = (T[name] - T.actual).to_numpy()
        per_day = pd.Series(d).groupby(days).sum().reindex(ud).to_numpy()
        bs = per_day[rng.integers(0, len(ud), (4000, len(ud)))].sum(1)
        hit = T[f"{name}_hit"].notna()
        out["rules"][name] = dict(total=float(T[name].sum()), diff=float(d.sum()), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                                  worst=float(T[name].min()), n_hit=int(hit.sum()), hit_actual=float(T.actual[hit].sum()),
                                  hit_cf=float(T[name][hit].sum()))
    return out


def selftest() -> None:
    # 기계: 롱 진입 100 → 44bp 마다 물타기 → F100 이면 3단위 들고 −100bp 에서 손절
    c = np.r_[100.0, np.linspace(100, 98.5, 200), np.full(50, 98.5)]
    h, l = c + 0.01, c - 0.01                                     # 봉 폭 > 봉 간 간격(0.0075) -- 갭 아님
    one = lambda v: np.array([v])  # noqa: E731
    pnl, q, xi, kind = sim_stop(h, l, c, np.array([0]), np.array([1]), 3, 240, 1, one(100.0), one(D0), one(TP0))
    assert kind[0] == 3 and q[0] == 3.0, (kind, q)
    exp = ((99.0 - 100) + (99.0 - 99.56) + (99.0 - 99.12)) / 100 * 1e4
    assert abs(pnl[0] - exp) < 1e-6, (pnl, exp)
    _, _, _, kind0 = sim_stop(h, l, c, np.array([0]), np.array([1]), 3, 240, 0, one(0.0), one(D0), one(TP0))
    # 간격·익절을 두 배로 → 88bp 마다 물타기: 1.5 내려간 경로에서 1단위 추가(88bp)뿐
    _, q2, _, _ = sim_stop(h, l, c, np.array([0]), np.array([1]), 3, 240, 0, one(0.0), one(88.0), one(40.0))
    assert q2[0] == 2.0, q2
    assert abs(vol_mult(np.array([157.394 * np.sqrt(6)]))[0] - 1 / 0.76576) < 1e-12
    assert kind0[0] == 2
    # 원장 반사실: 롱 1개 @100, F200 → 98 에서 청산
    g = pd.DataFrame({"time": [0], "price": [100.0], "qty": [1.0], "inc": [True], "commission": [0.0]})
    kt = np.arange(0, 2000) * 60_000; kc = np.linspace(100, 97, 2000)        # 촘촘한 봉(갭 없음)
    p, px, _ = trip_cf(g, 1, kt, kc + 0.01, kc - 0.01, kc, "F", 200, 1000.0, 100.0, True)
    assert abs(px - 98.0) < 1e-9 and abs(p - (-2 - STOP_FEE / 1e4 * 98)) < 1e-9, (p, px)
    # E10: 지갑 1000, 10개 → 손실 100$ 되는 가격 = 90
    g2 = g.assign(qty=[10.0]); kc2 = np.linspace(100, 85, 2000)
    _, px2, _ = trip_cf(g2, 1, kt, kc2 + 0.01, kc2 - 0.01, kc2, "E", 10, 1000.0, 100.0, True)
    assert abs(px2 - 90.0) < 1e-9, px2
    # Q3: 지갑 1000, 롱 100개 @100 → (1000 + 100(p−100) − .005·100p)/(100p) = .03 인 p
    g3 = g.assign(qty=[100.0]); kc3 = np.linspace(100, 90, 2000)
    _, px3, _ = trip_cf(g3, 1, kt, kc3 + 0.01, kc3 - 0.01, kc3, "Q", 3, 1000.0, 100.0, True)
    assert abs((1000 + 100 * (px3 - 100) - AG.MMR * 100 * px3) / (100 * px3) - 0.03) < 1e-9, px3
    # 숏 대칭: 숏 1개 @100, F200 → 102
    _, px4, _ = trip_cf(g, -1, kt, np.linspace(100, 103, 2000) + .01, np.linspace(100, 103, 2000) - .01, np.linspace(100, 103, 2000), "F", 200, 1000., 100., True)
    assert abs(px4 - 102.0) < 1e-9, px4
    print("selftest ok")


def main() -> int:
    ap = argparse.ArgumentParser()
    for m in ("--tape", "--ledger", "--combo", "--selftest"):
        ap.add_argument(m, action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return 0
    OUT.mkdir(parents=True, exist_ok=True)
    if a.tape:
        r = run_tape(); json.dump(r, open(OUT / "tape_result.json", "w"), ensure_ascii=False, indent=1)
    if a.ledger:
        r = run_ledger(); json.dump(r, open(OUT / "ledger_result.json", "w"), ensure_ascii=False, indent=1, default=str)
    if a.combo:
        r = {"tape": run_combo(), "ledger": ledger_combo()}
        json.dump(r, open(OUT / "combo_result.json", "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
