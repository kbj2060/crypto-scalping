#!/usr/bin/env python3
"""배포 R의 청산: **트레일 재조정 주기 R분**과 **스톱 체결 가정**을 분리해 재측정 (2026-09-07).

## 정정된 문제 설정

앞선 `research_eth_exit_resolution_1m_20260907.py`에서 1분 감시가 5분 대비 −6~−7.6bp였다.
처음엔 "5분봉 시뮬이 봉 안의 트레일 발동을 놓치는 허구"로 읽었으나 **틀렸다**. `sim_exit`은 인과적이고
구현 가능하다 — 봉 t 동안 상주하는 스톱은 t−1 종가에 정한 값이고(틱 단위로 계속 발동), 봉 t 종가에
새 고가로 재조정한다. 즉 **거래소 상주 스톱을 5분마다 재조정하는 정책**의 정확한 모델이다.

⇒ 5분 대 1분은 "허구 대 현실"이 아니라 **두 개의 실행 가능한 정책**이고, 재조정이 느릴수록 벌었다.
   느린 재조정 = 상승 중엔 스톱이 뒤처져 헐겁고(흔들려 나가지 않음), 상승이 멎으면 한 주기 만에
   peak−trail로 따라붙는다. 균일하게 trail을 넓히는 것과는 **모양이 다른** 정책이다.

## 이 스크립트가 재는 것

1분봉을 걷되 **불리 판정은 매 분**(상주 스톱은 틱 단위로 발동), **best/무장/스톱 재조정은 R분마다**.
   R=5는 배포 `sim_exit`을 비트 재현해야 한다(assert). R=1은 최속 재조정.
   R ∈ {1,2,5,10,15,30,60} × trail ∈ {0.1,0.2} · sl 5.0 · arm 1.5 고정 · 200봉 · 10bp

체결 가정 브래킷 (상주 STOP_MARKET은 스톱가가 아니라 발동 후 시장가로 체결된다):
   stop  = 스톱가 정확 체결 (낙관 상한, 배포 시뮬 가정)
   c1    = 발동한 **1분봉 종가**
   l1    = 발동한 1분봉의 **불리 극값** (비관 하한)

대조군: 방향뒤집기(페이드) 동일 셀 — "재조정 주기 효과"가 방향 무관 셀 효과인지 분리.
평가: 순차 포트폴리오(동시 5) · 일군집 CI · TRAIN 발견 / VAL·OOS 1회 확인. HOLDOUT 미접촉.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


XR = _load("xr1", "scripts/research_eth_exit_resolution_1m_20260907.py")
V2 = _load("hev2_rp", "scripts/research_homer_entry_v2_20260904.py")
portfolio, day_boot, delta_day_boot, stats_of = V2.portfolio, V2.day_boot, V2.delta_day_boot, V2.stats_of
OUT = ROOT / "data/research/eth_exit_reprice_interval_20260907"
SL, ARM = 5.0, 1.5
REPRICES = (1, 2, 5, 10, 15, 30, 60)
TRAILS = (0.1, 0.2)
FILLS = ("stop", "feasible", "c1", "l1")
FWD5, FWD1, COST, MAX_CONC, B_BOOT, CHUNK = 200, 1000, 10.0, 5, 1000, 2000
WINDOWS = ("TRAIN", "VAL", "OOS")


def log(m): print(f"[rp] {m}", flush=True)


def sim_reprice(entry, atr, sign, H, L, C, sl, arm, trail, R, fill="stop"):
    """1분봉을 걷는다. **불리 판정은 매 분**(상주 스톱), **best/무장/재조정은 R분마다**.
    R=5·fill='stop'이면 5분봉 sim_exit과 동일해야 한다. 반환 (수익률, 청산분오프셋, 사유, 무장, 진단).

    fill='feasible' = **거래소에 실제로 걸 수 있는 상주 스톱만 인정한다.** 재조정 시 새 스톱이 그 시점
    시장가(그 분의 종가)보다 불리한 쪽에 있지 않으면(롱인데 스톱>종가) 그건 이미 발동됐어야 할 자리다 --
    거래소는 그런 스톱 주문을 거부한다("Order would immediately trigger"). 그 경우 **즉시 종가에 청산**한다.
    'stop'은 그 자리에 걸린 것으로 치고 **그 가격에 체결까지 시켜준다** -- 시장이 이미 떠난 가격이다."""
    n = len(entry); T = H.shape[1]
    stop = entry - sign * sl * atr
    armed = np.zeros(n, bool); best = entry.copy()
    done = np.zeros(n, bool); out = np.zeros(n); ex = np.full(n, T - 1)
    reason = np.full(n, 2, np.int8)
    fav = np.where(sign[:, None] > 0, H, L)
    adv = np.where(sign[:, None] > 0, L, H)
    run_best = entry.copy()                                   # 주기 안에서 누적되는 관측 고가
    n_infeas = np.zeros(n, int); gap_bp = np.zeros(n)         # 진단: 시장 반대편에 놓인 스톱
    for t in range(T):
        if done.all():
            break
        a_ = adv[:, t]; live = ~done
        hit = live & np.where(sign > 0, a_ <= stop, a_ >= stop)
        px = stop if fill == "stop" else (C[:, t] if fill == "c1" else a_)
        out = np.where(hit, sign * (px - entry) / entry, out)
        ex = np.where(hit, t, ex)
        reason = np.where(hit, np.where(armed, 1, 0).astype(np.int8), reason)
        done = done | hit
        f_ = fav[:, t]; live = ~done
        run_best = np.where(live & (sign * (f_ - run_best) > 0), f_, run_best)
        if (t + 1) % R == 0:                                  # 주기 종료 -> 재조정
            imp = live & (sign * (run_best - best) > 0)
            best = np.where(imp, run_best, best)
            newly = live & ~armed & (sign * (best - entry) >= arm * atr)
            armed = armed | newly
            ns = best - sign * trail * atr
            u = live & armed & (sign * (ns - stop) > 0)
            bad = u & (sign * (ns - C[:, t]) > 0)             # 스톱이 현재가보다 유리한 쪽 = 걸 수 없는 자리
            n_infeas += bad.astype(int)
            gap_bp = np.where(bad & (n_infeas == 1), sign * (ns - C[:, t]) / entry * 1e4, gap_bp)
            if fill == "feasible":
                out = np.where(bad, sign * (C[:, t] - entry) / entry, out)   # 즉시 종가 청산
                ex = np.where(bad, t, ex)
                reason = np.where(bad, np.where(armed, 1, 0).astype(np.int8), reason)
                done = done | bad
                u = u & ~bad
            stop = np.where(u, ns, stop)
    out = np.where(done, out, sign * (C[:, -1] - entry) / entry)
    return out, ex, reason, armed, n_infeas, gap_bp


def run(entry, atr, sign, h1, l1, c1, m0, sl, arm, trail, R, fill):
    n = len(entry); pnl = np.empty(n); ex = np.empty(n, int); rs = np.empty(n, np.int8); am = np.empty(n, bool)
    nif = np.empty(n, int); gp = np.empty(n)
    for i in range(0, n, CHUNK):
        s = slice(i, min(i + CHUNK, n)); st = m0[s]
        H = np.stack([h1[j:j + FWD1] for j in st]); L = np.stack([l1[j:j + FWD1] for j in st])
        Cc = np.stack([c1[j:j + FWD1] for j in st])
        pnl[s], ex[s], rs[s], am[s], nif[s], gp[s] = sim_reprice(entry[s], atr[s], sign[s], H, L, Cc, sl, arm, trail, R, fill)
    return pnl * 1e4 - COST, ex, rs, am, nif, gp


def pf(pnl, ts, pos, ex_min, rng):
    eb5 = pos + 1 + np.ceil((ex_min + 1) / 5).astype(int)
    cand = pd.DataFrame({"timestamp": ts, "pos": pos, "p": np.ones(len(pos)), "entry_bar": pos + 1,
                         "exit_bar": eb5, "pnl_bp": pnl})
    r = portfolio(cand, MAX_CONC)
    if r is None:
        return None, None
    lo, hi = day_boot(r["trades"]["pnl_bp"], r["trades"]["timestamp"], B_BOOT, rng)
    o = stats_of(r); o["day_ci95"] = [round(lo, 2), round(hi, 2)]
    return o, r["trades"]


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(20260907)
    D = pd.read_parquet(XR.FRAME, columns=["pos", "is_downside", "timestamp", "split", "entry", "atr", "net_bp_flip"])
    bar = D.drop_duplicates("pos").sort_values("pos")[["pos", "timestamp", "split", "entry", "atr"]].reset_index(drop=True)
    kl5 = pd.read_csv(XR.KL5, usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"])
    kl5 = kl5.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    t_first = bar["timestamp"].iloc[0]; p_first = int(bar["pos"].iloc[0])
    k0 = int(np.searchsorted(kl5["timestamp"].to_numpy(), np.datetime64(t_first)))
    need = int(bar["pos"].iloc[-1]) - p_first + FWD5 + 2
    seg = kl5.iloc[k0:k0 + need].reset_index(drop=True)
    o5 = seg["open"].to_numpy(float)
    F = XR.load_fires(); Fp = F.loc[F["first_fire"]].drop_duplicates(["pos", "is_downside"]).copy()
    atr_of = bar.set_index("pos")["atr"]; split_of = bar.set_index("pos")["split"]; ts_of = bar.set_index("pos")["timestamp"]
    Fp = Fp.loc[Fp["pos"].isin(atr_of.index)].reset_index(drop=True)
    pos = Fp["pos"].to_numpy(); sd = Fp["is_downside"].to_numpy().astype(int)
    atr = atr_of.reindex(pos).to_numpy(float); split = split_of.reindex(pos).to_numpy(); ts = ts_of.reindex(pos).to_numpy()
    kp = pos - p_first; entry = o5[kp + 1]
    cont = -np.where(sd == 1, 1.0, -1.0)

    t_lo = pd.Timestamp(seg["timestamp"].iloc[0]); t_hi = pd.Timestamp(seg["timestamp"].iloc[-1]) + pd.Timedelta(minutes=5)
    d1, raw_n, miss = XR.load_1m(t_lo, t_hi)
    h1, l1, c1, o1 = (d1[x].to_numpy(float) for x in ("high", "low", "close", "open"))
    m_of = {t: i for i, t in enumerate(d1["timestamp"].to_numpy())}
    m0 = np.array([m_of[np.datetime64(seg["timestamp"].iloc[j + 1])] for j in kp])
    assert np.allclose(o1[m0], entry) and int(miss.sum()) == 0
    log(f"발동 {len(Fp):,} · 1분 {len(d1):,} · TRAIN/VAL/OOS {[int((split==w).sum()) for w in WINDOWS]} ({time.time()-t0:.0f}s)")

    # 파리티: R=5 · fill=stop 이 배포 프레임 net_bp_flip과 비트 일치해야 한다
    p5, _, _, _, _, _ = run(entry, atr, cont, h1, l1, c1, m0, SL, ARM, 0.1, 5, "stop")
    fr = D.set_index(["pos", "is_downside"]).reindex(pd.MultiIndex.from_arrays([pos, sd], names=["pos", "is_downside"]))
    par = float(np.nanmax(np.abs(fr["net_bp_flip"].to_numpy() - p5)))
    log(f"파리티 R=5·stop vs 배포 프레임 |Δ|max {par:.3e}bp"); assert par < 1e-6, par

    grid = []
    for trail in TRAILS:
        for R in REPRICES:
            for fill in FILLS:
                pnl, ex, rs, am, nif, gp = run(entry, atr, cont, h1, l1, c1, m0, SL, ARM, trail, R, fill)
                row = {"trail": trail, "reprice_min": R, "fill": fill,
                       "trail_bp_at_median_atr": round(trail * float(np.median(atr / entry)) * 1e4, 2)}
                for w in WINDOWS:
                    m = split == w
                    st, _t = pf(pnl[m], ts[m], pos[m], ex[m], rng)
                    row[w] = st["exp_bp"]; row[f"{w}_ci"] = st["day_ci95"]; row[f"{w}_win"] = st["win_rate"]
                    row[f"{w}_hold_med_min"] = float(np.median((ex[m] + 1)))
                if fill == "stop":                              # 방향뒤집기 대조군 (같은 셀·같은 주기)
                    row["infeasible_stop_frac"] = round(float((nif > 0).mean()), 4)
                    row["infeasible_gap_bp_median"] = round(float(np.median(gp[nif > 0])), 2) if (nif > 0).any() else None
                    pf_, exf, _, _, _, _ = run(entry, atr, -cont, h1, l1, c1, m0, SL, ARM, trail, R, fill)
                    for w in WINDOWS:
                        m = split == w
                        stf, _t = pf(pf_[m], ts[m], pos[m], exf[m], rng)
                        row[f"{w}_flip"] = stf["exp_bp"]
                grid.append(row)
            r0 = [g for g in grid if g["trail"] == trail and g["reprice_min"] == R]
            log("  trail %.1f R=%2dmin (걸수없는스톱 %.1f%% 중앙갭 %s bp) · " % (
                trail, R, r0[0]["infeasible_stop_frac"] * 100, r0[0]["infeasible_gap_bp_median"]) + " · ".join(
                f"{w}[" + "/".join(f"{g['fill']} {g[w]:+.2f}" for g in r0) + f" flip {r0[0][w+'_flip']:+.2f}]" for w in WINDOWS))

    rep = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "sl": SL, "arm": ARM, "trails": TRAILS,
           "reprice_min": REPRICES, "fills": FILLS, "cost_bp": COST, "fwd_5m_bars": FWD5, "max_concurrent": MAX_CONC,
           "holdout_touched": False, "n_fires": int(len(Fp)), "parity_R5_stop_vs_frame_maxabs_bp": par,
           "median_atr_pct": round(float(np.median(atr / entry)), 5), "grid": grid}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    log(f"저장 {OUT/'report.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
