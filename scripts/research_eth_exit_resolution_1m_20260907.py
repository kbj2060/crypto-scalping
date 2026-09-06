#!/usr/bin/env python3
"""배포 R의 **트레일링 청산을 1분봉 해상도로 재판정** + 청산 해부 (2026-09-07).

## 왜 이걸 재는가

2026-09-06 결론: 발동 이후 페이드/지속 레짐은 **모든 지평에서 공정한 동전**(H200 0.5054 [0.4981,0.5127])
인데 배포 규칙 R(지속)은 +4.96bp/건을 번다. 차이를 만드는 건 예측이 아니라 **트레일링 청산 구조**다.
그렇다면 그 구조가 **시뮬레이션 해상도의 산물이 아닌지**부터 확인해야 한다.

`sim_exit`은 5분봉을 걷는다. 봉마다 (1)직전 스톱으로 불리한 극값 판정 → (2)유리한 극값으로 best/무장/트레일.
즉 **봉 t의 고가에서 끌어올린 스톱은 봉 t+1에 가서야 검사된다.** 그런데 배포 셀의 trail은 0.1×ATR ≈ **2.8bp**로
5분봉 하나의 평균 레인지(≈28bp)의 **1/10**이다. 봉 안에서 그 폭을 몇 번이고 왕복할 수 있는 크기다.
⇒ 5분봉 시뮬은 봉 내부의 트레일 발동을 **구조적으로 놓친다.** 방향이 어느 쪽인지는 논증이 아니라 측정 대상이다.

## 팔 (모집단·방향·진입·셀·비용·한도 전부 배포값 상속 — 자유도 0)

  A  5분봉 200봉  (배포 시뮬 원문, 프레임 `net_bp_flip`과 비트 일치 검증)
  B  **1분봉 1000분** (같은 entry/atr/sl/arm/trail, 같은 종료 시각 — 바뀌는 건 감시 해상도뿐)
  C  5분봉 낙관 순서 (유리→불리) — 5분봉 자체의 순서 모호성 폭을 재는 대조군

해부(E1): 청산 사유(무장 전 SL / 트레일 / 타임아웃)별 건수·기여 bp, 보유 시간, 무장률, MFE/MAE.
평가: 순차 포트폴리오(동시 5) · 일군집 CI · B−A 일별 짝비교 CI. HOLDOUT 미접촉.
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


V2 = _load("hev2_xr", "scripts/research_homer_entry_v2_20260904.py")
sim_exit, portfolio, day_boot, delta_day_boot, stats_of = V2.sim_exit, V2.portfolio, V2.day_boot, V2.delta_day_boot, V2.stats_of
SIGNALS, OOFD, FRAME = V2.SIGNALS, V2.OOFD_MAT, V2.OUT / "frame.parquet"
KL5 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
OUT = ROOT / "data/research/eth_exit_resolution_1m_20260907"
CELL = (5.0, 1.5, 0.1)
FWD5, MIN_PER_BAR, COST, GAP, MAX_CONC, B_BOOT = 200, 5, 10.0, 12, 5, 1000
FWD1 = FWD5 * MIN_PER_BAR
WINDOWS = ("TRAIN", "VAL", "OOS")
CHUNK = 2000


def log(m): print(f"[xr] {m}", flush=True)


# ---------------------------------------------------------------- 진단용 sim_exit (A 원문과 비트 일치 검증)
def sim_exit_diag(entry, atr, sign, H, L, C, sl, arm, trail, order="pess", fill="stop"):
    """원문 sim_exit + 청산 사유/무장/MFE/MAE. order='opt'이면 유리→불리 순서(낙관 대조군).
    반환: pnl(비율), 청산봉오프셋, 사유(0=무장전SL 1=트레일 2=타임아웃), 무장여부, mfe_atr, mae_atr."""
    n = len(entry); T = H.shape[1]
    stop = entry - sign * sl * atr
    armed = np.zeros(n, bool); best = entry.copy()
    done = np.zeros(n, bool); out = np.zeros(n); ex = np.full(n, T - 1)
    reason = np.full(n, 2, np.int8); armed_at_exit = np.zeros(n, bool)
    mfe = np.zeros(n); mae = np.zeros(n)
    fav = np.where(sign[:, None] > 0, H, L)
    adv = np.where(sign[:, None] > 0, L, H)

    def _adverse(t):
        nonlocal out, ex, done, reason, armed_at_exit
        a_ = adv[:, t]; live = ~done
        hit = live & np.where(sign > 0, a_ <= stop, a_ >= stop)
        px = stop if fill == "stop" else C[:, t]
        out = np.where(hit, sign * (px - entry) / entry, out)
        ex = np.where(hit, t, ex)
        reason = np.where(hit, np.where(armed, 1, 0).astype(np.int8), reason)
        armed_at_exit = armed_at_exit | (hit & armed)
        done = done | hit

    def _favorable(t):
        nonlocal best, armed, stop
        f_ = fav[:, t]; live = ~done
        imp = live & (sign * (f_ - best) > 0)
        best = np.where(imp, f_, best)
        newly = live & ~armed & (sign * (best - entry) >= arm * atr)
        armed = armed | newly
        ns = best - sign * trail * atr
        u = live & armed & (sign * (ns - stop) > 0)
        stop = np.where(u, ns, stop)

    for t in range(T):
        if done.all():
            break
        live = ~done
        mfe = np.where(live, np.maximum(mfe, sign * (fav[:, t] - entry) / atr), mfe)
        mae = np.where(live, np.maximum(mae, -sign * (adv[:, t] - entry) / atr), mae)
        if order == "pess":
            _adverse(t); _favorable(t)
        else:
            _favorable(t); _adverse(t)
    out = np.where(done, out, sign * (C[:, -1] - entry) / entry)
    return out, ex, reason, armed, mfe, mae


# ---------------------------------------------------------------- 데이터
def load_fires():
    out = []
    for s in SIGNALS:
        d = pd.read_csv(OOFD / f"{s}_oof.csv", usecols=["pos", "side"]).drop_duplicates(["pos", "side"]).sort_values("pos")
        d["is_downside"] = (d["side"] == "bottom").astype(np.int8); ff = np.zeros(len(d), bool)
        for sd in (0, 1):
            idx = np.flatnonzero(d["is_downside"].to_numpy() == sd); pos = d["pos"].to_numpy()[idx]
            keep = np.zeros(len(pos), bool); last = -10**9
            for j, p in enumerate(pos):
                if p - last > GAP:
                    keep[j] = True
                last = p
            ff[idx] = keep
        d["first_fire"] = ff; d["signal"] = s; out.append(d)
    return pd.concat(out, ignore_index=True)


def load_1m(t_lo, t_hi):
    """1분봉을 완전한 분 격자로 재색인. 결측 분은 직전 종가로 평탄 채움(무거래 분)."""
    d = pd.read_csv(KL1, usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"])
    d = d.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    d = d.loc[(d["timestamp"] >= t_lo) & (d["timestamp"] <= t_hi)].reset_index(drop=True)
    grid = pd.date_range(t_lo, t_hi, freq="1min")
    raw_n = len(d)
    d = d.set_index("timestamp").reindex(grid)
    miss = d["close"].isna().to_numpy()
    d["close"] = d["close"].ffill()
    for c in ("open", "high", "low"):
        d[c] = d[c].fillna(d["close"])
    d = d.reset_index().rename(columns={"index": "timestamp"})
    return d, raw_n, miss


def parity_1m_vs_5m(d1, miss, kl5):
    """결측이 하나도 없는 5분 그룹만 골라 1분→5분 재집계 대조. 최대 절대오차 반환."""
    g = pd.DataFrame({"timestamp": d1["timestamp"].dt.floor("5min"), "o": d1["open"], "h": d1["high"],
                      "l": d1["low"], "c": d1["close"], "miss": miss.astype(int)})
    agg = g.groupby("timestamp").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"),
                                     miss=("miss", "sum"), n=("c", "size"))
    agg = agg.loc[(agg["miss"] == 0) & (agg["n"] == 5)]
    m = kl5.set_index("timestamp").reindex(agg.index).dropna(subset=["open"])
    agg = agg.loc[m.index]
    bad = np.zeros(len(agg), bool); e = 0.0
    for a, b in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close")):
        d = np.abs(agg[a].to_numpy() - m[b].to_numpy())
        bad |= d > 1e-9; e = max(e, float(np.nanmax(d)))
    return e, int(len(agg)), int(bad.sum()), [str(t) for t in agg.index[bad][:5]]


def windows_from(arr, starts, width):
    """starts(정수 인덱스) 기준 width 길이 창을 청크로 잘라 stack."""
    for i in range(0, len(starts), CHUNK):
        s = starts[i:i + CHUNK]
        yield i, np.stack([arr[j:j + width] for j in s])


def run_arm(entry, atr, sign, arrH, arrL, arrC, starts, width, order="pess", fill="stop", cell=CELL):
    n = len(entry)
    pnl = np.empty(n); ex = np.empty(n, int); rs = np.empty(n, np.int8)
    am = np.empty(n, bool); mf = np.empty(n); ma = np.empty(n)
    for i, Hh in windows_from(arrH, starts, width):
        sl_ = slice(i, i + len(Hh))
        Ll = np.stack([arrL[j:j + width] for j in starts[sl_]])
        Cc = np.stack([arrC[j:j + width] for j in starts[sl_]])
        o = sim_exit_diag(entry[sl_], atr[sl_], sign[sl_], Hh, Ll, Cc, *cell, order=order, fill=fill)
        pnl[sl_], ex[sl_], rs[sl_], am[sl_], mf[sl_], ma[sl_] = o
    return pnl * 1e4 - COST, ex, rs, am, mf, ma


def pf(pnl, ts, pos, exit_bar5, rng):
    cand = pd.DataFrame({"timestamp": ts, "pos": pos, "p": np.ones(len(pos)), "entry_bar": pos + 1,
                         "exit_bar": exit_bar5, "pnl_bp": pnl})
    r = portfolio(cand, MAX_CONC)
    if r is None:
        return None, None
    lo, hi = day_boot(r["trades"]["pnl_bp"], r["trades"]["timestamp"], B_BOOT, rng)
    o = stats_of(r); o["day_ci95"] = [round(lo, 2), round(hi, 2)]
    o["n_days"] = int(pd.DatetimeIndex(r["trades"]["timestamp"]).normalize().nunique())
    return o, r["trades"]


def anatomy(pnl, ex, rs, am, mf, ma, unit_min):
    lab = {0: "sl_before_arm", 1: "trail", 2: "timeout"}
    o = {"armed_frac": round(float(am.mean()), 4),
         "hold_min_median": float(np.median((ex + 1) * unit_min)),
         "mfe_atr_median": round(float(np.median(mf)), 3), "mae_atr_median": round(float(np.median(ma)), 3),
         "by_reason": {}}
    tot = float(pnl.sum())
    for k, name in lab.items():
        m = rs == k
        o["by_reason"][name] = {"n": int(m.sum()), "frac": round(float(m.mean()), 4),
                                "mean_bp": round(float(pnl[m].mean()), 2) if m.any() else None,
                                "share_of_total_bp": round(float(pnl[m].sum() / tot), 4) if tot else None,
                                "hold_min_median": float(np.median((ex[m] + 1) * unit_min)) if m.any() else None}
    return o


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(20260907)
    D = pd.read_parquet(FRAME, columns=["pos", "is_downside", "timestamp", "split", "entry", "atr", "net_bp_flip"])
    bar = D.drop_duplicates("pos").sort_values("pos")[["pos", "timestamp", "split", "entry", "atr"]].reset_index(drop=True)

    kl5 = pd.read_csv(KL5, usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"])
    kl5 = kl5.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    t_first = bar["timestamp"].iloc[0]; p_first = int(bar["pos"].iloc[0])
    k0 = int(np.searchsorted(kl5["timestamp"].to_numpy(), np.datetime64(t_first))); assert kl5["timestamp"].iloc[k0] == t_first
    need = int(bar["pos"].iloc[-1]) - p_first + FWD5 + 2
    seg = kl5.iloc[k0:k0 + need].reset_index(drop=True)
    assert np.all(np.diff(seg["timestamp"].to_numpy()).astype("timedelta64[m]").astype(int) == 5), "5분봉 비연속"
    o5, h5, l5, c5 = (seg[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    kpos = lambda p: p - p_first

    F = load_fires(); Fp = F.loc[F["first_fire"]].drop_duplicates(["pos", "is_downside"]).copy()
    atr_of = bar.set_index("pos")["atr"]; split_of = bar.set_index("pos")["split"]; ts_of = bar.set_index("pos")["timestamp"]
    Fp = Fp.loc[Fp["pos"].isin(atr_of.index)].reset_index(drop=True)
    pos = Fp["pos"].to_numpy(); sd = Fp["is_downside"].to_numpy().astype(int)
    atr = atr_of.reindex(pos).to_numpy(float); split = split_of.reindex(pos).to_numpy()
    ts = ts_of.reindex(pos).to_numpy(); kp = kpos(pos)
    fade_sign = np.where(sd == 1, 1.0, -1.0); cont_sign = -fade_sign          # 배포 R = 지속
    entry = o5[kp + 1]
    log(f"첫발동 {len(Fp):,} · TRAIN/VAL/OOS {[int((split==w).sum()) for w in WINDOWS]} ({time.time()-t0:.0f}s)")

    # ---- 1분봉
    t_lo = pd.Timestamp(seg["timestamp"].iloc[0])
    t_hi = pd.Timestamp(seg["timestamp"].iloc[-1]) + pd.Timedelta(minutes=5)
    d1, raw_n, miss = load_1m(t_lo, t_hi)
    par_e, par_n, par_bad, par_ts = parity_1m_vs_5m(d1, miss, kl5)
    log(f"1분봉 {len(d1):,}분 (원본 {raw_n:,} · 결측채움 {int(miss.sum()):,} = {miss.mean()*100:.3f}%) · "
        f"5분 재집계 파리티 불일치 {par_bad}/{par_n:,}봉 ({par_bad/par_n*100:.5f}%) |Δ|max {par_e:.3g} "
        f"{par_ts} ({time.time()-t0:.0f}s)")
    # 두 CSV는 각각 다운로드된 별개 파일이라 극히 드문 단일봉 불일치는 데이터 아티팩트다.
    # 계통 오차(정렬/시간대)라면 비율이 크게 나온다 -- 비율로 막는다.
    assert par_bad / par_n < 1e-4, f"1분->5분 재집계 계통 불일치 {par_bad}/{par_n}"
    o1, h1, l1, c1 = (d1[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    m_of = {t: i for i, t in enumerate(d1["timestamp"].to_numpy())}
    m0 = np.array([m_of[np.datetime64(seg["timestamp"].iloc[j + 1])] for j in kp])
    assert np.allclose(o1[m0], entry), "1분 시가 != 5분 진입가"
    assert m0.max() + FWD1 <= len(o1), "1분 창 부족"

    # ---- 팔
    res, tr_out = {}, {}
    G5 = dict(arrH=h5, arrL=l5, arrC=c5, starts=kp + 1, width=FWD5)
    G1 = dict(arrH=h1, arrL=l1, arrC=c1, starts=m0, width=FWD1)
    for name, kw in (("A_5m_stop", dict(G5, order="pess", fill="stop")),      # 배포/섀도우 원문
                     ("C_5m_opt",  dict(G5, order="opt",  fill="stop")),      # 5분봉 순서 모호성 폭
                     ("D_5m_close", dict(G5, order="pess", fill="close")),    # 5분 폴링 + 관측봉 종가 시장가
                     ("B_1m_stop", dict(G1, order="pess", fill="stop")),      # 1분 감시 + 스톱가 체결
                     ("E_1m_close", dict(G1, order="pess", fill="close"))):   # 1분 폴링 + 관측봉 종가
        pnl, ex, rs, am, mf, ma = run_arm(entry, atr, cont_sign, **kw)
        unit = 5 if name.startswith(("A", "C")) else 1
        eb5 = pos + 1 + (ex + 1 if unit == 5 else np.ceil((ex + 1) / 5).astype(int))
        res[name] = {"unit_min": unit, "windows": {}}
        tr_out[name] = {}
        for w in WINDOWS:
            m = split == w
            st, tr = pf(pnl[m], ts[m], pos[m], eb5[m], rng)
            res[name]["windows"][w] = {"portfolio": st, "anatomy": anatomy(pnl[m], ex[m], rs[m], am[m], mf[m], ma[m], unit),
                                       "raw_mean_bp": round(float(pnl[m].mean()), 3), "n_rows": int(m.sum())}
            tr_out[name][w] = tr
        log(f"{name}: " + " · ".join(f"{w} {res[name]['windows'][w]['portfolio']['exp_bp']:+.2f}bp "
                                     f"{res[name]['windows'][w]['portfolio']['day_ci95']}" for w in WINDOWS)
            + f" ({time.time()-t0:.0f}s)")
        if name == "A_5m_stop":
            key = D.set_index(["pos", "is_downside"])
            fr = key.reindex(pd.MultiIndex.from_arrays([pos, sd], names=["pos", "is_downside"]))
            par = float(np.nanmax(np.abs(fr["net_bp_flip"].to_numpy() - pnl)))
            log(f"  라벨 파리티(프레임 net_bp_flip) |Δ|max {par:.2e}bp"); assert par < 1e-6

    # ---- B − A 일별 짝비교
    cmp_ = {}
    for other in ("B_1m_stop", "D_5m_close", "E_1m_close", "C_5m_opt"):
        cmp_[other] = {}
        for w in WINDOWS:
            ta, tb = tr_out["A_5m_stop"][w], tr_out[other][w]
            d, lo, hi = delta_day_boot(tb["pnl_bp"], tb["timestamp"], ta["pnl_bp"], ta["timestamp"], B_BOOT, rng)
            cmp_[other][w] = {"delta_vs_A_bp": round(d, 3), "ci95": [round(lo, 3), round(hi, 3)]}
        log(f"{other} − A: " + " · ".join(f"{w} {cmp_[other][w]['delta_vs_A_bp']:+.2f} {cmp_[other][w]['ci95']}" for w in WINDOWS))

    # ---- 트레일 폭 × 해상도 스윕 (선택이 아니라 **전 셀 보고** -- TRAIN이 발견집합, VAL/OOS는 1회 확인)
    log("스윕: 5분 vs 1분 해상도 × 트레일 폭 (전 셀 보고, 선택 없음)")
    sweep = []
    for arm_ in (1.0, 1.5, 2.0):
        for tr_ in (0.1, 0.2, 0.3, 0.5, 1.0):
            cell = (5.0, arm_, tr_); row = {"cell": cell, "trail_bp_at_median_atr": round(tr_ * float(np.median(atr / entry)) * 1e4, 2)}
            for tag, kw in (("A5", dict(G5, order="pess", fill="stop")), ("B1", dict(G1, order="pess", fill="stop"))):
                pnl_, ex_, _, _, _, _ = run_arm(entry, atr, cont_sign, cell=cell, **kw)
                unit = 5 if tag == "A5" else 1
                eb5_ = pos + 1 + (ex_ + 1 if unit == 5 else np.ceil((ex_ + 1) / 5).astype(int))
                for w in WINDOWS:
                    m = split == w
                    st_, _t = pf(pnl_[m], ts[m], pos[m], eb5_[m], rng)
                    row[f"{tag}_{w}"] = st_["exp_bp"]; row[f"{tag}_{w}_ci"] = st_["day_ci95"]
            for w in WINDOWS:
                row[f"gap_{w}"] = round(row[f"B1_{w}"] - row[f"A5_{w}"], 2)
            sweep.append(row)
            log(f"  arm {arm_} trail {tr_} ({row['trail_bp_at_median_atr']:.1f}bp) · "
                + " · ".join(f"{w} A5 {row[f'A5_{w}']:+.2f} B1 {row[f'B1_{w}']:+.2f} {row[f'B1_{w}_ci']}" for w in WINDOWS))

    rep = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "cell": CELL, "fwd_5m_bars": FWD5, "cost_bp": COST,
           "gap": GAP, "max_concurrent": MAX_CONC, "holdout_touched": False, "n_fires": int(len(Fp)),
           "median_atr_pct": round(float(np.median(atr / entry)), 5),
           "trail_bp_at_median_atr": round(float(CELL[2] * np.median(atr / entry) * 1e4), 2),
           "bar_range_bp_median_5m": round(float(np.median((h5[kp + 1] - l5[kp + 1]) / entry) * 1e4), 2),
           "onem_missing_frac": round(float(miss.mean()), 6),
           "parity_1m_5m": {"bars": par_n, "mismatch_bars": par_bad, "maxabs": par_e, "mismatch_ts": par_ts},
           "arms": res, "paired_vs_A": cmp_, "trail_resolution_sweep": sweep}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    log(f"저장 {OUT/'report.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
