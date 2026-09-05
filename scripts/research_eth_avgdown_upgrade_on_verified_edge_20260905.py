#!/usr/bin/env python3
"""사용자 구조(30배 · 20% 트랜치 · 물타기 · 짧은 익절)를 **검증된 엣지 위에** 올려 업그레이드 (2026-09-05).

사용자: *"30배 · 20% 현금 · 물타기 · 짧게 먹고 나오는 전략을 지금 재료로 완벽한 경제성 전략으로
업그레이드할 수 있을지 연구해줘."*

## 선행이 확정한 것 (반복 안 함)
  §5.30 1편  칩 표시 규약(익절-또는-지평선, 손절 없음) -> 방향 무관 gross ~= 0. 손실 = 수수료.
  §5.30 2편  30배 전액증거금 고정 TP/SL 90셀 -> 양수 0. 승률 90.6%인데 EV -3.19%/건, 청산 9~12%.
  §5.30 3편  20% 트랜치 + 물타기(손절 없음) -> 전손률은 9~12%에서 1.2~2.4%로 내려가나 부호 불변.
  §5.30 4-1  ⭐칩의 **무작위 초과는 수수료에 불변**(+0.23~0.43%p/건 = 가격 3~4bp). 개선축은 통계가 아니라 실행.
  §5.29 C4   손절을 조이면 단조 악화. **넓은 손절은 필수 부품**.
  §5.29 C3   되돌림 지정가 진입은 역선택(-1~-13bp) -- 물타기와 같은 모집단.

## 이 스크립트가 새로 재는 것 -- "컨테이너를 바꾸면 사용자 구조가 사나"
검증된 엣지는 **방향(지속) + 트레일링 브래킷**이다(라이브 섀도우 가동 중, VAL +4.44 / OOS +6.78 bp/건).
사용자 구조의 세 부품을 **그 엣지 위에** 하나씩 얹어 각각이 엣지를 살리는지 죽이는지 본다.

  U1  레버리지 상한   전략 자신의 손절(5.0xATR)보다 **청산선(100%/L)이 먼저** 오는 비율.
                      30배 = 역행 333bp. ETH 5xATR이 그 안쪽이면 30배는 전략을 다른 전략으로 바꾼다.
  U2  컨테이너 교체   {트레일(검증본) · 부분익절+트레일(신규) · 고정TP 16.7 · 고정TP 33.3}
  U3  물타기          역행 {1.0, 2.0}xATR에서 트랜치 추가(최대 2/4회) + **신호기반 추가**(같은 방향
                      새 첫발동에서 추가) -- 가격 기반 물타기와 나란히 비교.
  U4  자본구조        트랜치 {20%, 10%} x 레버리지 {3(배포), 10, 30(사용자)}

판정: TRAIN에서만 선택, VAL/OOS 1회 조회. 같은 측면 무작위 귀무 동반. 일군집 부트스트랩 CI.
비용은 경로에 영향이 없으므로 한 번 시뮬 후 {4, 7.8, 10}bp를 해석적으로 차감한다(명목 대비).
HOLDOUT(>=2026-04-01) 미접촉.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TRAIN_END, VAL_END, OOS_END = (pd.Timestamp(x) for x in ("2025-09-01", "2026-01-01", "2026-04-01"))
FWD, GAP, ATR_N = 200, 12, 14
SL_ATR, ARM_ATR, TRAIL_ATR = 5.0, 1.5, 0.1          # F0 셀 상속 (자유도 0)
COSTS = [4.0, 7.8, 10.0]                             # 명목 대비 왕복 bp
B_BOOT, SEED = 400, 20260905
OUT = ROOT / "data/research/eth_avgdown_upgrade_verified_edge_20260905"
WINDOWS = ("TRAIN", "VAL", "OOS")

SIGNALS = ["taker_delta_z_climax", "short_term_return_z", "liquidity_sweep", "orthogonal_combo",
           "smt_divergence", "fib_extension_exhaustion", "demarker_extreme", "kalman_deviation_meanrev"]


def log(m): print(f"[upg] {m}", flush=True)


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


# ----------------------------------------------------------------------------- 경로 시뮬레이션
def simulate(o, h, l, c, atr, fire_up, fire_dn, pos, sgn, container, tp_x,
             add_mode, add_x, max_adds, lev, n):
    """트랜치 단위 경로 시뮬. 반환 dict.

    단위: notional은 **트랜치 수**(u). 계좌% 환산은 호출부에서 tranche x lev x 100.
    ⭐봉내 순서는 **가격 거리순**이다(2026-09-05 정정). 가격은 먼 레벨에 닿기 전에 가까운 레벨을 지난다:
      물타기(1~2xATR) -> 손절(5xATR)과 청산(100%/L) 중 **평균가에 가까운 쪽** -> 익절.
      따라서 손절선이 청산선보다 안쪽이면 **청산은 구조적으로 불가능**하다(갭/미체결 제외).
      역으로 손절이 없는 컨테이너(고정TP)에서는 청산이 유일한 하방 배리어다.
      같은 거리일 때만 비관(청산 우선). 이전 판(청산 무조건 우선)은 손절 있는 셀의 전손률을 과대평가한다.
    물타기 시 브래킷은 새 평균가 기준으로 **재설정**(stop/best/armed) -- 평균가가 곧 새 진입이다.
    """
    if pos + 1 >= n:
        return None
    entry = o[pos + 1]
    a = atr[pos]                                     # 발동 봉 ATR(절대가). 트레이드 내내 고정(sim_exit 규약)
    if not np.isfinite(a) or a <= 0:
        return None
    avg, u, k = entry, 1.0, 1
    realized = 0.0                                   # 부분청산 누적(평균가 기준 수익률 x 청산 유닛)
    stop = avg - sgn * SL_ATR * a
    best, armed = avg, False
    part_done = False
    adds_in_profit = 0                               # 추가 시점에 평가이익이었던 횟수(피라미딩 여부 진단)
    mae = 0.0                                        # 평균가 대비 최대 역행(비율) -- U1용
    liq_frac = np.inf if lev <= 0 else 1.0 / lev   # lev<=0 = 청산 없음(U1 기준선)
    end = min(pos + 1 + FWD, n - 1)
    for j in range(pos + 1, end + 1):
        hi, lo = h[j], l[j]
        fav_px = hi if sgn > 0 else lo
        adv_px = lo if sgn > 0 else hi
        adv = sgn * (avg - adv_px) / avg             # 평균가 대비 역행(양수 = 손해)
        if adv > mae:
            mae = adv
        # (1) 물타기 (가장 가까운 레벨: 1~2xATR)
        if max_adds and k <= max_adds:
            do_add = False
            if add_mode == "price":
                do_add = adv >= add_x * a / avg
            elif add_mode == "signal":
                do_add = bool(fire_up[j] if sgn > 0 else fire_dn[j])
            while do_add and k <= max_adds:
                # ⭐발동은 봉 j가 **마감돼야** 알려진다 -> 신호 기반 추가는 open[j+1] 체결(L4 known_ts 계약).
                #   close[j] 체결은 한 봉 미래참조다(2026-09-05 수정).
                if add_mode == "signal" and j + 1 >= n:
                    break
                add_px = (avg * (1 - sgn * add_x * a / avg)) if add_mode == "price" else o[j + 1]
                if sgn * (add_px - avg) > 0:
                    adds_in_profit += 1
                avg = (avg * u + add_px) / (u + 1.0)
                u += 1.0; k += 1
                stop = avg - sgn * SL_ATR * a; best, armed = avg, False
                adv = sgn * (avg - adv_px) / avg
                if adv > mae:
                    mae = adv
                do_add = (add_mode == "price") and (adv >= add_x * a / avg)
        # (2) 하방 배리어 -- 손절과 청산 중 **평균가에 가까운 쪽**이 먼저 체결된다
        liq_px = avg * (1 - sgn * liq_frac)
        has_stop = container in ("trail", "hybrid")
        stop_first = has_stop and (sgn * (stop - liq_px) >= 0)   # 손절선이 청산선보다 안쪽(유리한 쪽)
        if stop_first:
            if sgn * (adv_px - stop) <= 0:
                realized += sgn * (stop - avg) / avg * u
                return {"acct_units": realized, "cost_units": k, "outcome": "stop", "k": k,
                        "mae": mae, "exit_off": j - pos - 1, "pnl_units": realized, "adds_in_profit": adds_in_profit}
        else:
            if adv >= liq_frac:
                return {"acct_units": -k, "cost_units": k, "outcome": "liq", "k": k,
                        "mae": mae, "exit_off": j - pos - 1, "pnl_units": None, "adds_in_profit": adds_in_profit}
            if has_stop and sgn * (adv_px - stop) <= 0:
                realized += sgn * (stop - avg) / avg * u
                return {"acct_units": realized, "cost_units": k, "outcome": "stop", "k": k,
                        "mae": mae, "exit_off": j - pos - 1, "pnl_units": realized, "adds_in_profit": adds_in_profit}
        # (4) 익절 / 부분익절
        if container == "hybrid" and not part_done and sgn * (fav_px - avg) / avg >= tp_x * a / avg:
            px = avg + sgn * tp_x * a
            realized += sgn * (px - avg) / avg * (u * 0.5)
            u *= 0.5; part_done = True
        elif container == "tp" and sgn * (fav_px - avg) / avg >= tp_x / 1e4:
            px = avg * (1 + sgn * tp_x / 1e4)
            realized += sgn * (px - avg) / avg * u
            return {"acct_units": realized, "cost_units": k, "outcome": "tp", "k": k,
                    "mae": mae, "exit_off": j - pos - 1, "pnl_units": realized, "adds_in_profit": adds_in_profit}
        # 트레일 갱신
        if container in ("trail", "hybrid"):
            if sgn * (fav_px - best) > 0:
                best = fav_px
            if not armed and sgn * (best - avg) >= ARM_ATR * a:
                armed = True
            if armed:
                ns = best - sgn * TRAIL_ATR * a
                if sgn * (ns - stop) > 0:
                    stop = ns
    realized += sgn * (c[end] - avg) / avg * u
    return {"acct_units": realized, "cost_units": k, "outcome": "timeout", "k": k,
            "mae": mae, "exit_off": end - pos - 1, "pnl_units": realized, "adds_in_profit": adds_in_profit}


def day_ci(vals, days, rng, b=B_BOOT):
    u = np.unique(days)
    if len(u) < 2:
        return np.nan, np.nan
    by = {d: vals[days == d] for d in u}
    m = np.empty(b)
    for i in range(b):
        m[i] = np.concatenate([by[d] for d in rng.choice(u, size=len(u), replace=True)]).mean()
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    t0 = time.time(); rng = np.random.default_rng(SEED)
    log("신호 프레임 재구성...")
    _s1 = _load("s1_upg", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
    _s1.VAL_END = OOS_END
    sig, _f, _e = _s1.build_sig()
    ts = pd.to_datetime(sig["timestamp"]).dt.tz_localize(None)
    m = (ts < OOS_END).to_numpy()
    sig, ts = sig.loc[m].reset_index(drop=True), ts.loc[m].reset_index(drop=True)
    o, h, l, c = (sig[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(sig)
    day = ts.dt.floor("D").to_numpy()
    split = np.where(ts < TRAIN_END, "TRAIN", np.where(ts < VAL_END, "VAL", "OOS"))

    # ATR = 트루레인지 14봉 단순이동평균 (라이브 러너 정의)
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], axis=1)
    atr = pd.Series(tr).rolling(ATR_N).mean().to_numpy()

    # ── 모집단: 8종 raw 첫발동(GAP=12, 신호·측면별) 합집합, 같은 봉 양측면 발동은 스킵 ──
    up = np.zeros(n, bool); dn = np.zeros(n, bool)          # 신호 측면별 첫발동 봉
    for name in SIGNALS:
        for side, arr in (("bottom", dn), ("top", up)):     # bottom 발동 -> 지속=숏, top 발동 -> 지속=롱
            col = f"{side}_{name}"
            if col not in sig.columns:
                continue
            f = sig[col].fillna(False).to_numpy(bool)
            last = -10**9
            for i in np.flatnonzero(f):
                if i - last > GAP:
                    arr[i] = True
                last = i
    both = up & dn
    fire_up, fire_dn = up & ~both, dn & ~both               # 지속 롱 발동 / 지속 숏 발동
    P = np.flatnonzero(fire_up | fire_dn)
    S = np.where(fire_up[P], 1.0, -1.0)                     # 지속 방향
    ok = (P + 1 + FWD < n) & np.isfinite(atr[P]) & (atr[P] > 0)
    P, S = P[ok], S[ok]
    spP, dayP = split[P], day[P]
    log(f"  지속 모집단 {len(P):,}건 (양측면 동시발동 스킵 {int(both.sum())}봉) · "
        f"TRAIN {int((spP=='TRAIN').sum()):,} VAL {int((spP=='VAL').sum()):,} OOS {int((spP=='OOS').sum()):,}")
    atr_pct = atr[P] / o[P + 1]
    log(f"  ATR/가격 중앙 {np.median(atr_pct)*1e4:.1f}bp · 5xATR 중앙 {np.median(atr_pct)*5*1e4:.0f}bp "
        f"· 90분위 {np.percentile(atr_pct,90)*5*1e4:.0f}bp   (30배 청산선 = 333bp)")

    # ── 같은 측면 무작위 귀무 (창별 건수·측면비율 매칭) ──
    pool = np.flatnonzero((np.arange(n) + 1 + FWD < n) & np.isfinite(atr) & (atr > 0))
    Pn, Sn = [], []
    for sp in WINDOWS:
        cnt = int((spP == sp).sum()); cand = pool[split[pool] == sp]
        Pn.append(rng.choice(cand, size=cnt, replace=True))
        Sn.append(rng.permutation(S[spP == sp]))
    Pn, Sn = np.concatenate(Pn), np.concatenate(Sn)
    spN = split[Pn]

    # ── 격자 ──
    CONTAINERS = [("trail", 0.0), ("hybrid", 1.5), ("tp", 16.7), ("tp", 33.3)]
    ADDS = [("none", 0.0, 0), ("price", 1.0, 2), ("price", 1.0, 4),
            ("price", 2.0, 2), ("price", 2.0, 4), ("signal", 0.0, 2)]
    TRANCHES = [0.20, 0.10]
    LEVS = [3.0, 10.0, 30.0]
    combos = [(ct, tx, am, ax, ma, tr, lv)
              for (ct, tx) in CONTAINERS for (am, ax, ma) in ADDS
              for tr in TRANCHES for lv in LEVS]
    if os.environ.get("SMOKE"):
        combos = combos[:1]
        log("  ⚠️SMOKE: 격자 1셀만 -- U1 파리티 확인용")
    log(f"  격자 {len(combos)}셀 x (실측+귀무) 시뮬레이션...")

    days_all = np.unique(dayP)
    day_ix = {d: i for i, d in enumerate(days_all)}
    dayP_ix = np.array([day_ix[d] for d in dayP])
    daily = {f"{c:g}": np.full((len(combos), len(days_all)), np.nan) for c in COSTS}

    cells = []
    for ci, (ct, tx, am, ax, ma, tr, lv) in enumerate(combos, 1):
        res = [simulate(o, h, l, c, atr, fire_up, fire_dn, p, s, ct, tx, am, ax, ma, lv, n)
               for p, s in zip(P, S)]
        nres = [simulate(o, h, l, c, atr, fire_up, fire_dn, p, s, ct, tx, am, ax, ma, lv, n)
                for p, s in zip(Pn, Sn)]
        au = np.array([r["acct_units"] for r in res], float)
        cu = np.array([r["cost_units"] for r in res], float)
        oc = np.array([r["outcome"] for r in res], object)
        kk = np.array([r["k"] for r in res], float)
        aip = np.array([r["adds_in_profit"] for r in res], float)
        mae = np.array([r["mae"] for r in res], float)
        nau = np.array([r["acct_units"] for r in nres], float)
        ncu = np.array([r["cost_units"] for r in nres], float)
        scale = tr * lv * 100.0
        row = {"container": ct, "tp_x": tx, "add_mode": am, "add_x": ax, "max_adds": ma,
               "tranche": tr, "leverage": lv, "splits": {}}
        for cost in COSTS:
            acct = np.where(oc == "liq", au * tr * 100.0, (au - cost / 1e4 * cu) * scale)
            nacct = np.where(np.array([r["outcome"] for r in nres], object) == "liq",
                             nau * tr * 100.0, (nau - cost / 1e4 * ncu) * scale)
            dm = np.bincount(dayP_ix, weights=acct, minlength=len(days_all))
            dn_ = np.bincount(dayP_ix, minlength=len(days_all))
            daily[f"{cost:g}"][ci - 1] = np.where(dn_ > 0, dm / np.maximum(dn_, 1), np.nan)
            for sp in WINDOWS:
                msk = spP == sp
                if msk.sum() < 30:
                    continue
                key = f"{sp}@{cost:g}"
                lo, hi = day_ci(acct[msk], dayP[msk], rng)
                nm = float(nacct[spN == sp].mean())
                row["splits"][key] = {
                    "n": int(msk.sum()), "account_pct": round(float(acct[msk].mean()), 4),
                    "ci": [round(lo, 4), round(hi, 4)],
                    "excess_vs_null": round(float(acct[msk].mean() - nm), 4),
                    "null_account_pct": round(nm, 4)}
        for sp in WINDOWS:
            msk = spP == sp
            if msk.sum() < 30:
                continue
            row["splits"][f"{sp}@meta"] = {
                "win_rate": round(float((au[msk] > 0).mean()), 4),
                "ruin_rate": round(float((oc[msk] == "liq").mean()), 4),
                "tp_rate": round(float((oc[msk] == "tp").mean()), 4),
                "mean_tranches": round(float(kk[msk].mean()), 3),
                "adds_in_profit_frac": round(float(aip[msk].sum() / max((kk[msk] - 1).sum(), 1e-9)), 4),
                "mae_p50_bp": round(float(np.median(mae[msk]) * 1e4), 1),
                "mae_p95_bp": round(float(np.percentile(mae[msk], 95) * 1e4), 1),
                "gross_bp_equiv": round(float(au[msk].mean() * 1e4), 2)}
        cells.append(row)
        if ci % 10 == 0:
            log(f"    {ci}/{len(combos)} ({time.time()-t0:.0f}s)")

    # ── U1: 레버리지 상한 (기준 컨테이너 = 트레일 · 물타기 없음 · 무한 레버리지 근사) ──
    log("\n=== U1 레버리지 상한 ===")
    base = [simulate(o, h, l, c, atr, fire_up, fire_dn, p, s, "trail", 0.0, "none", 0.0, 0, 0.0, n)
            for p, s in zip(P, S)]
    bm = np.array([r["mae"] for r in base], float)
    bnet = np.array([r["acct_units"] for r in base], float) * 1e4 - 10.0     # 10bp 차감 = 배포 규약
    sl_frac = SL_ATR * atr[P] / o[P + 1]                 # 손절선까지 거리(비율)
    u1 = {}
    for sp in WINDOWS:
        msk = spP == sp
        u1[sp] = {"n": int(msk.sum()), "net_bp_10cost": round(float(bnet[msk].mean()), 3),
                  "mae_p50_bp": round(float(np.median(bm[msk]) * 1e4), 1),
                  "mae_p90_bp": round(float(np.percentile(bm[msk], 90) * 1e4), 1),
                  "mae_p99_bp": round(float(np.percentile(bm[msk], 99) * 1e4), 1),
                  "p_mae_exceeds_liq_by_lev": {str(int(L)): round(float((bm[msk] >= 1.0 / L).mean()), 5)
                                               for L in (3, 5, 10, 15, 20, 25, 30, 50)},
                  # ⭐실제로 청산 가능한 건 = **손절선이 청산선 바깥**인 트레이드뿐이다(5xATR >= 100%/L).
                  #   그 안쪽이면 가격이 손절을 먼저 지나므로 청산에 닿을 수 없다.
                  "p_stop_outside_liq_by_lev": {str(int(L)): round(float((sl_frac[msk] >= 1.0 / L).mean()), 5)
                                                for L in (3, 5, 10, 15, 20, 25, 30, 50)},
                  "sl_frac_p50_bp": round(float(np.median(sl_frac[msk]) * 1e4), 1),
                  "sl_frac_p99_bp": round(float(np.percentile(sl_frac[msk], 99) * 1e4), 1)}
        log(f"  {sp}: 파리티 net {u1[sp]['net_bp_10cost']:+.2f}bp/건 (기준 VAL +4.44 / OOS +6.78) · "
            f"MAE p50 {u1[sp]['mae_p50_bp']:.0f} p90 {u1[sp]['mae_p90_bp']:.0f} p99 {u1[sp]['mae_p99_bp']:.0f}bp")
        log(f"       MAE가 청산선 초과 " + " ".join(f"{L}x:{u1[sp]['p_mae_exceeds_liq_by_lev'][L]:.3%}"
                                              for L in ("10", "20", "30", "50")))
        log(f"       ⭐손절선이 청산선 바깥(=청산 가능) " + " ".join(
            f"{L}x:{u1[sp]['p_stop_outside_liq_by_lev'][L]:.3%}" for L in ("10", "20", "30", "50"))
            + f"  | 5xATR p50 {u1[sp]['sl_frac_p50_bp']:.0f}bp p99 {u1[sp]['sl_frac_p99_bp']:.0f}bp")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "daily.npz", days=days_all.astype("datetime64[D]"),
                        split=np.array([("TRAIN" if d < np.datetime64("2025-09-01") else
                                         "VAL" if d < np.datetime64("2026-01-01") else "OOS")
                                        for d in days_all.astype("datetime64[D]")]),
                        combos=np.array([f"{a}|{b}|{c_}|{d_}|{e_}|{f_}|{g_}" for (a, b, c_, d_, e_, f_, g_) in combos]),
                        **{f"cost{k}": v for k, v in daily.items()})
    (OUT / "report.json").write_text(json.dumps(
        {"holdout_touched": False, "n_pop": int(len(P)), "fwd": FWD, "gap": GAP,
         "cell": [SL_ATR, ARM_ATR, TRAIL_ATR], "costs_bp": COSTS,
         "u1_leverage_ceiling": u1, "cells": cells}, ensure_ascii=False, indent=2))
    log(f"\n산출: {OUT}/report.json ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
