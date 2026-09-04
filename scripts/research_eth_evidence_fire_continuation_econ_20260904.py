#!/usr/bin/env python3
"""증거신호 8종 **발동 봉의 방향** -- 경제라벨(F0 셀) 아래서 페이드 vs 지속(continuation) (2026-09-04).

## 발견 경위 (TRAIN이 발견 집합, VAL/OOS는 1회 확인)

"왜 V자반등 경제라벨(F0)만 살아남고 8종 발동 재라벨링은 전부 실패했나"를 진단하다가, F0 프레임
(매 봉 × 양방향, sim_exit 5.0/1.5/0.1, 200봉, 10bp)에서 **발동 봉 자체의 두 측면 라벨**을 나란히 봤다.
TRAIN 첫발동(GAP=12) 12,987건에서 신호 방향(페이드) 평균 −3.0bp, **반대 방향(지속) +4.7bp**,
P(페이드>지속)=0.446, 지속−페이드 일군집 CI [+4.5, +11.0]. 8종 전부·양 측면 모두 같은 부호였다.
⇒ 발동 봉은 이 라벨 척도에서 **반전 시점이 아니라 지속 시점**이다(§5.20 "트리거 후 2~4 ATR 더 진행"의
라벨 쪽 표현). 8종 재라벨링(§5.22)은 **방향을 페이드로 못 박은 채** 발동 모집단만 재라벨했으므로 실패가
구조적이었다. §5.22 축 11(방향반전)은 신호별 n 221~899·자기 horizon(8~72봉)·셀(4.0/1.0/0.1)에서 t>1.96을
요구해 0/8이었지만 8종 중 6종이 양수였다 -- 검정력 문제였다.

## 이 스크립트가 재는 것 (규칙에 자유도 없음 -- 셀·GAP·비용·한도 전부 상속)

  모집단   8종 raw 인과 단일봉 발동의 첫발동(GAP=12, 뒤만 봄) 합집합, (봉,측면) 중복 제거
  진입     open[i+1] 시장가 (F0 프레임과 동일)
  방향     페이드(신호 방향) vs 지속(반대 방향) -- 둘 다 보고, 판정 대상은 **지속**
  라벨     sim_exit(5.0/1.5/0.1) 200봉 − 10bp  ← F0에서 상속. 프레임 net_bp와 비트 일치 검증
  평가     순차 포트폴리오(동시 5) 기대값 · 일군집 CI · 측면별 갭(A2) · 측면비율 매칭 무작위귀무(B=200)
           · 월별 · 신호별 기여 · **견고성**: 셀 6종 / 진입 1봉 지연 / 비용 15bp / raw 발동 (선택이 아니라 보고)

⚠️HOLDOUT 미접촉. OOS는 고정 규칙 1회 조회. 이 결과는 진단이며, 승격 주장은 층 게이트(L4/L1/L2/L2P/T1/T2)
와 전진 섀도우를 거쳐야 한다.
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


V2 = _load("hev2_cont", "scripts/research_homer_entry_v2_20260904.py")
sim_exit, portfolio, day_boot, stats_of = V2.sim_exit, V2.portfolio, V2.day_boot, V2.stats_of
SIGNALS, OOFD, FRAME = V2.SIGNALS, V2.OOFD_MAT, V2.OUT / "frame.parquet"
KL = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT = ROOT / "data/research/eth_evidence_fire_continuation_econ_20260904"
CELL = (5.0, 1.5, 0.1)                                   # F0 상속 (선택 아님)
CELLS_ROBUST = [(5.0, 1.5, 0.1), (4.0, 1.0, 0.1), (3.0, 1.5, 0.1), (4.0, 2.0, 0.1), (5.0, 1.5, 0.05), (5.0, 1.0, 0.1)]
FWD, COST, GAP, MAX_CONC, B_NULL, B_BOOT = 200, 10.0, 12, 5, 200, 1000
WINDOWS = ("TRAIN", "VAL", "OOS")


def log(m): print(f"[cont] {m}", flush=True)


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


def run_cell(entry, atr, sign, H, L, C, cell, delay=0):
    pn, ex = sim_exit(entry, atr, sign, H, L, C, *cell)
    return pn * 1e4 - COST, ex


def pf_stats(pnl, ts, pos, ex, rng, p=None):
    cand = pd.DataFrame({"timestamp": ts, "pos": pos, "p": np.ones(len(pos)) if p is None else p, "entry_bar": pos + 1,
                         "exit_bar": pos + 1 + ex, "pnl_bp": pnl})
    r = portfolio(cand, MAX_CONC)
    if r is None:
        return None
    lo, hi = day_boot(r["trades"]["pnl_bp"], r["trades"]["timestamp"], B_BOOT, rng)
    o = stats_of(r); o["day_ci95"] = [round(lo, 2), round(hi, 2)]
    o["days"] = int(pd.DatetimeIndex(r["trades"]["timestamp"]).normalize().nunique())
    mo = pd.Series(r["trades"]["pnl_bp"].to_numpy(), index=pd.to_datetime(r["trades"]["timestamp"].to_numpy())).groupby(lambda x: x.to_period("M")).mean()
    o["monthly_exp_bp"] = {str(k): round(float(v), 2) for k, v in mo.items()}
    return o, r["trades"]


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(20260904)
    D = pd.read_parquet(FRAME, columns=["pos", "is_downside", "side", "timestamp", "split", "entry", "atr", "net_bp", "net_bp_flip", "exit_off"])
    bar = D.drop_duplicates("pos").sort_values("pos")[["pos", "timestamp", "split", "entry", "atr"]].reset_index(drop=True)
    step = np.diff(bar["timestamp"].to_numpy()).astype("timedelta64[m]").astype(int); dpos = np.diff(bar["pos"].to_numpy())
    assert np.all(step == 5 * dpos), "프레임 pos↔timestamp 비아핀"
    kl = pd.read_csv(KL, usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    t_first = bar["timestamp"].iloc[0]; p_first = int(bar["pos"].iloc[0])
    k0 = int(np.searchsorted(kl["timestamp"].to_numpy(), np.datetime64(t_first))); assert kl["timestamp"].iloc[k0] == t_first
    need = int(bar["pos"].iloc[-1]) - p_first + FWD + 2
    seg = kl.iloc[k0:k0 + need]
    assert np.all(np.diff(seg["timestamp"].to_numpy()).astype("timedelta64[m]").astype(int) == 5), "klines 구간 비연속"
    o, h, l, c = (seg[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    kpos = lambda p: p - p_first                      # 프레임 pos -> seg 인덱스
    atr_of = bar.set_index("pos")["atr"]; split_of = bar.set_index("pos")["split"]; ts_of = bar.set_index("pos")["timestamp"]
    # 파리티: 프레임 open[pos+1] == seg open
    chk = bar.sample(2000, random_state=1); assert np.allclose(chk["entry"].to_numpy(), o[kpos(chk["pos"].to_numpy()) + 1]), "entry 불일치"

    F = load_fires(); Fp = F.loc[F["first_fire"]].drop_duplicates(["pos", "is_downside"]).copy()
    Fp = Fp.loc[Fp["pos"].isin(atr_of.index)].reset_index(drop=True)
    pos = Fp["pos"].to_numpy(); sd = Fp["is_downside"].to_numpy().astype(int)
    atr = atr_of.reindex(pos).to_numpy(float); split = split_of.reindex(pos).to_numpy(); ts = ts_of.reindex(pos).to_numpy()
    fade_sign = np.where(sd == 1, 1.0, -1.0)          # 바닥 발동 -> 페이드 = 롱
    kp = kpos(pos)
    H = np.stack([h[j + 1:j + 1 + FWD] for j in kp]); L = np.stack([l[j + 1:j + 1 + FWD] for j in kp]); C = np.stack([c[j + 1:j + 1 + FWD] for j in kp])
    entry = o[kp + 1]
    log(f"첫발동 합집합 {len(Fp):,} (TRAIN/VAL/OOS {[(split == w).sum() for w in WINDOWS]}) · 바닥 {int((sd == 1).sum()):,} 천장 {int((sd == 0).sum()):,} ({time.time()-t0:.0f}s)")

    # 파리티: 페이드 라벨 == 프레임 net_bp (비트 일치), 지속 라벨 == 프레임 net_bp_flip
    fade0, ex_f0 = run_cell(entry, atr, fade_sign, H, L, C, CELL); cont0, ex_c0 = run_cell(entry, atr, -fade_sign, H, L, C, CELL)
    key = D.set_index(["pos", "is_downside"])
    fr = key.reindex(pd.MultiIndex.from_arrays([pos, sd], names=["pos", "is_downside"]))
    par_fade = float(np.nanmax(np.abs(fr["net_bp"].to_numpy() - fade0))); par_cont = float(np.nanmax(np.abs(fr["net_bp_flip"].to_numpy() - cont0)))
    log(f"라벨 파리티 |Δ|max 페이드 {par_fade:.2e}bp · 지속 {par_cont:.2e}bp (프레임 원문과 동일해야 함)")
    assert par_fade < 1e-6 and par_cont < 1e-6

    rep = {"cell": CELL, "forward_bars": FWD, "cost_bp": COST, "gap": GAP, "max_concurrent": MAX_CONC, "holdout_touched": False,
           "n_first_fires": int(len(Fp)), "label_parity_max_abs_bp": {"fade": par_fade, "cont": par_cont}, "windows": {}}
    trades_out = {}
    for w in WINDOWS:
        m = split == w; R = {"n_rows": int(m.sum())}
        for nm, arr, ex in (("fade", fade0, ex_f0), ("cont", cont0, ex_c0)):
            st, tr = pf_stats(arr[m], ts[m], pos[m], ex[m], rng)
            R[nm] = {"row_mean_bp": round(float(arr[m].mean()), 3), "row_wr": round(float((arr[m] > 0).mean()), 3), "portfolio": st}
            if nm == "cont":
                trades_out[w] = tr
        R["p_fade_gt_cont"] = round(float((fade0[m] > cont0[m]).mean()), 4)
        lo, hi = day_boot(cont0[m] - fade0[m], ts[m], B_BOOT, rng); R["cont_minus_fade_day_ci95"] = [round(lo, 2), round(hi, 2)]
        # A2 측면별: 바닥 발동(지속=숏) / 천장 발동(지속=롱)
        R["by_side"] = {}
        for sv, nm in ((1, "bottom_fire(cont=short)"), (0, "top_fire(cont=long)")):
            mm = m & (sd == sv); lo, hi = day_boot(cont0[mm] - fade0[mm], ts[mm], B_BOOT, rng)
            R["by_side"][nm] = {"n": int(mm.sum()), "fade_bp": round(float(fade0[mm].mean()), 2), "cont_bp": round(float(cont0[mm].mean()), 2),
                                "gap_bp": round(float((cont0[mm] - fade0[mm]).mean()), 2), "gap_day_ci95": [round(lo, 2), round(hi, 2)],
                                "p_fade_gt_cont": round(float((fade0[mm] > cont0[mm]).mean()), 3)}
        # 측면비율 매칭 무작위 귀무: 같은 창의 매봉 프레임에서 같은 측면 개수만큼 무작위 (봉,측면) 추출 -> 포트폴리오
        Dw = D.loc[D["split"] == w]; n_l = int((sd[m] == 0).sum()); n_s = int((sd[m] == 1).sum())   # 지속: 천장->롱(is_downside 1), 바닥->숏(0)
        pool_l = Dw.loc[Dw["is_downside"] == 1]; pool_s = Dw.loc[Dw["is_downside"] == 0]
        nulls = []
        for _ in range(B_NULL):
            a = pool_l.iloc[rng.choice(len(pool_l), size=min(n_l, len(pool_l)), replace=False)]
            b = pool_s.iloc[rng.choice(len(pool_s), size=min(n_s, len(pool_s)), replace=False)]
            x = pd.concat([a, b]); cand = pd.DataFrame({"timestamp": x["timestamp"].to_numpy(), "pos": x["pos"].to_numpy(), "p": 1.0,
                                                        "entry_bar": x["pos"].to_numpy() + 1, "exit_bar": x["pos"].to_numpy() + 1 + x["exit_off"].to_numpy(),
                                                        "pnl_bp": x["net_bp"].to_numpy()})
            r = portfolio(cand, MAX_CONC); nulls.append(r["exp_bp"] if r else np.nan)
        nulls = np.asarray(nulls, float); obs = R["cont"]["portfolio"]["exp_bp"]
        R["side_matched_null"] = {"mean_bp": round(float(np.nanmean(nulls)), 3), "p95_bp": round(float(np.nanpercentile(nulls, 95)), 3),
                                  "percentile_of_cont": round(float((nulls < obs).mean() * 100), 1)}
        # 신호별 기여 (지속, 행 평균)
        R["per_signal_cont"] = {}
        for s in SIGNALS:
            ms = m & Fp["signal"].to_numpy().__eq__(s)       # 합집합 dedup 후 signal은 첫 등장 신호 -- 참고치
            if ms.sum() >= 30:
                R["per_signal_cont"][s] = {"n": int(ms.sum()), "cont_bp": round(float(cont0[ms].mean()), 2), "fade_bp": round(float(fade0[ms].mean()), 2)}
        # 견고성 (보고 전용)
        rob = {}
        for cell in CELLS_ROBUST:
            cc, exc = run_cell(entry, atr, -fade_sign, H, L, C, cell); st, _ = pf_stats(cc[m], ts[m], pos[m], exc[m], rng)
            rob[f"cell_{cell[0]}_{cell[1]}_{cell[2]}"] = {"row_mean_bp": round(float(cc[m].mean()), 2), "pf_exp_bp": st["exp_bp"], "day_ci95": st["day_ci95"]}
        # 진입 1봉 지연: open[i+2], 창 한 칸 뒤 (마지막 봉 하나 짧음)
        H1 = np.stack([h[j + 2:j + 1 + FWD] for j in kp]); L1 = np.stack([l[j + 2:j + 1 + FWD] for j in kp]); C1 = np.stack([c[j + 2:j + 1 + FWD] for j in kp])
        c1, ex1 = run_cell(o[kp + 2], atr, -fade_sign, H1, L1, C1, CELL); st, _ = pf_stats(c1[m], ts[m], pos[m] + 1, ex1[m], rng)
        rob["delay_1bar"] = {"row_mean_bp": round(float(c1[m].mean()), 2), "pf_exp_bp": st["exp_bp"], "day_ci95": st["day_ci95"]}
        st, _ = pf_stats(cont0[m] - 5.0, ts[m], pos[m], ex_c0[m], rng)
        rob["cost_15bp"] = {"row_mean_bp": round(float(cont0[m].mean() - 5.0), 2), "pf_exp_bp": st["exp_bp"], "day_ci95": st["day_ci95"]}
        R["robustness_cont"] = rob
        rep["windows"][w] = R
        log(f"{w}: n {m.sum():,} · 페이드 {R['fade']['portfolio']['exp_bp']:+.2f} · 지속 {R['cont']['portfolio']['exp_bp']:+.2f} "
            f"CI {R['cont']['portfolio']['day_ci95']} · P(페이드>지속) {R['p_fade_gt_cont']} · 귀무백분위 {R['side_matched_null']['percentile_of_cont']} ({time.time()-t0:.0f}s)")
    # raw 발동(첫발동 아님) 참고
    Fr = F.drop_duplicates(["pos", "is_downside"]); Fr = Fr.loc[Fr["pos"].isin(atr_of.index)]
    key2 = key.reindex(pd.MultiIndex.from_arrays([Fr["pos"].to_numpy(), 1 - Fr["is_downside"].to_numpy().astype(int)], names=["pos", "is_downside"]))
    rep["raw_fires_cont_row_mean_bp"] = {w: round(float(key2.loc[key2["split"] == w, "net_bp"].mean()), 2) for w in WINDOWS}
    rep["raw_fires_n"] = {w: int((key2["split"] == w).sum()) for w in WINDOWS}
    (OUT / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    for w, tr in trades_out.items():
        tr.to_csv(OUT / f"trades_cont_{w}.csv", index=False)
    print(f"\n{'win':>5s} {'n':>6s} {'fade_pf':>8s} {'cont_pf':>8s} {'cont_CI':>18s} {'P(f>c)':>7s} {'null%':>6s} {'bot gap(CI)':>26s} {'top gap(CI)':>26s}")
    for w in WINDOWS:
        R = rep["windows"][w]; b = R["by_side"]["bottom_fire(cont=short)"]; t = R["by_side"]["top_fire(cont=long)"]
        print(f"{w:>5s} {R['n_rows']:6d} {R['fade']['portfolio']['exp_bp']:8.2f} {R['cont']['portfolio']['exp_bp']:8.2f} {str(R['cont']['portfolio']['day_ci95']):>18s} "
              f"{R['p_fade_gt_cont']:7.3f} {R['side_matched_null']['percentile_of_cont']:6.1f} {str(b['gap_bp'])+' '+str(b['gap_day_ci95']):>26s} {str(t['gap_bp'])+' '+str(t['gap_day_ci95']):>26s}")
    print("\n[robustness cont pf_exp_bp]")
    for k in rep["windows"]["VAL"]["robustness_cont"]:
        print(f"  {k:>18s} " + "  ".join(f"{w} {rep['windows'][w]['robustness_cont'][k]['pf_exp_bp']:+6.2f} {rep['windows'][w]['robustness_cont'][k]['day_ci95']}" for w in WINDOWS))
    print("\n[monthly cont]", {w: rep["windows"][w]["cont"]["portfolio"]["monthly_exp_bp"] for w in ("VAL", "OOS")})
    print("[per-signal cont VAL/OOS]", {w: rep["windows"][w]["per_signal_cont"] for w in ("VAL", "OOS")})
    print("[raw fires cont row mean]", rep["raw_fires_cont_row_mean_bp"], rep["raw_fires_n"])
    log(f"완료 -> {OUT/'report.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
