#!/usr/bin/env python3
"""수정된 `sim_exit`으로 **F0 프레임 라벨 전체 재계산** + 그 위의 규칙 판정 재평가 (2026-09-07).

`tmp/homer_entry_v2_20260904/frame.parquet`(매 봉 × 양측면 403,190행)의 `net_bp`/`net_bp_flip`은
결함 `sim_exit`(걸 수 없는 자리의 스톱을 그 가격에 체결)으로 만들어졌다. 이 프레임 위에 F0 경제라벨 ·
지속 규칙 R · 호메로스 진입 v2 · 증거신호8 경제성 · 09-06 페이드/지속 라벨이 전부 얹혀 있다.

세 가지 체결 가정으로 같은 봉·같은 셀(5.0/1.5/0.1, 200봉, 10bp)을 다시 계산한다:
  legacy  원문(결함) -- 프레임 저장값과 **비트 일치**해야 한다(파리티 게이트)
  fixed   `infeasible='exit'` -- 걸 수 없는 자리면 즉시 그 봉 종가 청산 (기본 수정)
  hold    `infeasible='hold'` -- 스톱을 올리지 않고 유지 (느슨한 트레일 변형 정책)

판정 재평가(선택 규칙에 모델이 없는 것만 -- 모델 기반 팔은 라벨이 바뀌면 재학습이 필요하므로 여기선
모집단 기준선 이동폭만 보고한다):
  V1 전체 봉 양측면 (F0 모집단 기준선)   V2 지속 규칙 R(첫발동·지속)  V3 R 페이드(대조)
  V4 R 신호별 기여                        V5 09-06 페이드/지속 라벨(`y_dec`)과 '동전' 주장
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


V2 = _load("hev2_rc", "scripts/research_homer_entry_v2_20260904.py")
sim_exit, sim_exit_legacy = V2.sim_exit, V2.sim_exit_legacy
portfolio, day_boot, delta_day_boot, stats_of = V2.portfolio, V2.day_boot, V2.delta_day_boot, V2.stats_of
SIGNALS, OOFD, FRAME = V2.SIGNALS, V2.OOFD_MAT, V2.OUT / "frame.parquet"
KL5 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT = ROOT / "data/research/eth_fixed_exit_recompute_20260907"
CELL, FWD, COST, GAP, MAX_CONC, B_BOOT, CHUNK = (5.0, 1.5, 0.1), 200, 10.0, 12, 5, 1000, 4000
WINDOWS = ("TRAIN", "VAL", "OOS")
MODES = ("legacy", "fixed", "hold")


def log(m): print(f"[rc] {m}", flush=True)


def run_mode(entry, atr, sign, h, l, c, kp, mode):
    n = len(entry); pnl = np.empty(n); ex = np.empty(n, int)
    for i in range(0, n, CHUNK):
        s = slice(i, min(i + CHUNK, n)); st = kp[s] + 1
        H = np.stack([h[j:j + FWD] for j in st]); L = np.stack([l[j:j + FWD] for j in st])
        C = np.stack([c[j:j + FWD] for j in st])
        f = sim_exit_legacy if mode == "legacy" else sim_exit
        kw = {} if mode == "legacy" else {"infeasible": "exit" if mode == "fixed" else "hold"}
        o, e = f(entry[s], atr[s], sign[s], H, L, C, *CELL, **kw)
        pnl[s], ex[s] = o, e
    return pnl * 1e4 - COST, ex


def pf(pnl, ts, pos, ex, rng):
    cand = pd.DataFrame({"timestamp": ts, "pos": pos, "p": np.ones(len(pos)), "entry_bar": pos + 1,
                         "exit_bar": pos + 1 + ex + 1, "pnl_bp": pnl})
    r = portfolio(cand, MAX_CONC)
    if r is None:
        return None, None
    lo, hi = day_boot(r["trades"]["pnl_bp"], r["trades"]["timestamp"], B_BOOT, rng)
    o = stats_of(r); o["day_ci95"] = [round(lo, 2), round(hi, 2)]
    return o, r["trades"]


def verdict(name, sel, sign_of, ts, pos, P, E, rng, out):
    """sel: 행 마스크. sign_of: +1=페이드 -1=지속(부호는 fade_sign 대비)."""
    o = {"n_rows": int(sel.sum()), "windows": {}}
    tr = {}
    for w in WINDOWS:
        m = sel & (out["split"] == w)
        o["windows"][w] = {}
        for mo in MODES:
            st, t = pf(P[mo][sign_of][m], ts[m], pos[m], E[mo][sign_of][m], rng)
            o["windows"][w][mo] = st; tr[(w, mo)] = t
        for mo in ("fixed", "hold"):
            d, lo, hi = delta_day_boot(tr[(w, mo)]["pnl_bp"], tr[(w, mo)]["timestamp"],
                                       tr[(w, "legacy")]["pnl_bp"], tr[(w, "legacy")]["timestamp"], B_BOOT, rng)
            o["windows"][w][f"delta_{mo}_vs_legacy"] = {"bp": round(d, 2), "ci95": [round(lo, 2), round(hi, 2)]}
    log(f"{name} (n={o['n_rows']:,}) · " + " · ".join(
        f"{w} legacy {o['windows'][w]['legacy']['exp_bp']:+.2f} → fixed {o['windows'][w]['fixed']['exp_bp']:+.2f} "
        f"{o['windows'][w]['fixed']['day_ci95']} / hold {o['windows'][w]['hold']['exp_bp']:+.2f}" for w in WINDOWS))
    return o


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(20260907)
    D = pd.read_parquet(FRAME, columns=["pos", "is_downside", "timestamp", "split", "entry", "atr", "net_bp", "net_bp_flip", "exit_off"])
    bar = D.drop_duplicates("pos").sort_values("pos").reset_index(drop=True)
    kl = pd.read_csv(KL5, usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"])
    kl = kl.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    p_first = int(bar["pos"].iloc[0])
    k0 = int(np.searchsorted(kl["timestamp"].to_numpy(), np.datetime64(bar["timestamp"].iloc[0])))
    seg = kl.iloc[k0:k0 + int(bar["pos"].iloc[-1]) - p_first + FWD + 2].reset_index(drop=True)
    assert np.all(np.diff(seg["timestamp"].to_numpy()).astype("timedelta64[m]").astype(int) == 5)
    o_, h, l, c = (seg[x].to_numpy(float) for x in ("open", "high", "low", "close"))

    pos = bar["pos"].to_numpy(); kp = pos - p_first
    entry = o_[kp + 1]; atr = bar["atr"].to_numpy(float); ts = bar["timestamp"].to_numpy()
    split = bar["split"].to_numpy()
    assert np.allclose(entry, bar["entry"].to_numpy()), "entry 불일치"
    log(f"봉 {len(bar):,} (프레임 행 {len(D):,}) · TRAIN/VAL/OOS {[int((split==w).sum()) for w in WINDOWS]} ({time.time()-t0:.0f}s)")

    # 봉마다 두 부호 (+1 = 바닥발동 페이드 = 롱 쪽 기준. 프레임 net_bp/net_bp_flip 정의와 맞춘다)
    P, E = {m: {} for m in MODES}, {m: {} for m in MODES}
    for mo in MODES:
        for sgn_tag, s in (("down_fade", np.ones(len(pos))), ("down_cont", -np.ones(len(pos)))):
            P[mo][sgn_tag], E[mo][sgn_tag] = run_mode(entry, atr, s, h, l, c, kp, mo)
        log(f"  {mo} 계산 완료 ({time.time()-t0:.0f}s)")

    # 파리티: legacy가 프레임 저장값과 비트 일치해야 한다 (is_downside=1 행 기준)
    dn = D.loc[D["is_downside"] == 1].set_index("pos").reindex(pos)
    par_f = float(np.nanmax(np.abs(dn["net_bp"].to_numpy() - P["legacy"]["down_fade"])))
    par_c = float(np.nanmax(np.abs(dn["net_bp_flip"].to_numpy() - P["legacy"]["down_cont"])))
    log(f"파리티 legacy vs 프레임: net_bp |Δ|max {par_f:.3e} · net_bp_flip |Δ|max {par_c:.3e}")
    assert par_f < 1e-9 and par_c < 1e-9, "legacy 재계산이 프레임과 불일치"

    outdf = pd.DataFrame({"pos": pos, "timestamp": ts, "split": split, "entry": entry, "atr": atr})
    for mo in MODES:
        for tag in ("down_fade", "down_cont"):
            outdf[f"{tag}_{mo}_bp"] = P[mo][tag]; outdf[f"{tag}_{mo}_ex"] = E[mo][tag]
    outdf.to_parquet(OUT / "bar_labels.parquet", index=False)

    rep = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "cell": CELL, "fwd": FWD, "cost_bp": COST,
           "gap": GAP, "max_concurrent": MAX_CONC, "holdout_touched": False,
           "parity_legacy_vs_frame_maxabs_bp": {"net_bp": par_f, "net_bp_flip": par_c},
           "n_bars": int(len(bar)), "verdicts": {}}

    allm = np.ones(len(pos), bool); ctx = {"split": split}
    rep["verdicts"]["V1_all_bars_cont"] = verdict("V1 전체봉 지속", allm, "down_cont", ts, pos, P, E, rng, ctx)
    rep["verdicts"]["V1b_all_bars_fade"] = verdict("V1b 전체봉 페이드", allm, "down_fade", ts, pos, P, E, rng, ctx)

    # 첫발동 (GAP=12, 신호·측면별) 합집합
    F = []
    for s in SIGNALS:
        d = pd.read_csv(OOFD / f"{s}_oof.csv", usecols=["pos", "side"]).drop_duplicates(["pos", "side"]).sort_values("pos")
        d["is_downside"] = (d["side"] == "bottom").astype(np.int8); ff = np.zeros(len(d), bool)
        for sd in (0, 1):
            idx = np.flatnonzero(d["is_downside"].to_numpy() == sd); pp = d["pos"].to_numpy()[idx]
            keep = np.zeros(len(pp), bool); last = -10**9
            for j, x in enumerate(pp):
                if x - last > GAP:
                    keep[j] = True
                last = x
            ff[idx] = keep
        d["first_fire"] = ff; d["signal"] = s; F.append(d)
    F = pd.concat(F, ignore_index=True)
    Fp = F.loc[F["first_fire"]].drop_duplicates(["pos", "is_downside"])
    idx_of = pd.Series(np.arange(len(pos)), index=pos)

    # R 규칙: 바닥발동은 지속=숏(down_cont), 천장발동은 지속=롱(=down_fade 부호와 같음)
    for tag, name in (("cont", "V2 지속 규칙 R"), ("fade", "V3 R 페이드(대조)")):
        rows_dn = Fp.loc[Fp["is_downside"] == 1, "pos"]; rows_up = Fp.loc[Fp["is_downside"] == 0, "pos"]
        i_dn = idx_of.reindex(rows_dn).dropna().astype(int).to_numpy()
        i_up = idx_of.reindex(rows_up).dropna().astype(int).to_numpy()
        sub = {}
        for w in WINDOWS:
            sub[w] = {}
            trs = {}
            for mo in MODES:
                a = "down_cont" if tag == "cont" else "down_fade"
                b = "down_fade" if tag == "cont" else "down_cont"
                ii = np.concatenate([i_dn, i_up]); pn = np.concatenate([P[mo][a][i_dn], P[mo][b][i_up]])
                ee = np.concatenate([E[mo][a][i_dn], E[mo][b][i_up]])
                m = split[ii] == w
                st, t = pf(pn[m], ts[ii][m], pos[ii][m], ee[m], rng)
                sub[w][mo] = st; trs[mo] = t
            for mo in ("fixed", "hold"):
                d, lo, hi = delta_day_boot(trs[mo]["pnl_bp"], trs[mo]["timestamp"], trs["legacy"]["pnl_bp"],
                                           trs["legacy"]["timestamp"], B_BOOT, rng)
                sub[w][f"delta_{mo}_vs_legacy"] = {"bp": round(d, 2), "ci95": [round(lo, 2), round(hi, 2)]}
        rep["verdicts"][f"V2_R_{tag}"] = {"n_fires": int(len(Fp)), "windows": sub}
        log(f"{name} (n={len(Fp):,}) · " + " · ".join(
            f"{w} legacy {sub[w]['legacy']['exp_bp']:+.2f} → fixed {sub[w]['fixed']['exp_bp']:+.2f} "
            f"{sub[w]['fixed']['day_ci95']} / hold {sub[w]['hold']['exp_bp']:+.2f}" for w in WINDOWS))

    # V5 09-06 페이드/지속 라벨: 지속이 페이드를 이기는 비율 (첫발동, 모드별)
    i_dn = idx_of.reindex(Fp.loc[Fp["is_downside"] == 1, "pos"]).dropna().astype(int).to_numpy()
    i_up = idx_of.reindex(Fp.loc[Fp["is_downside"] == 0, "pos"]).dropna().astype(int).to_numpy()
    v5 = {}
    for mo in MODES:
        cont = np.concatenate([P[mo]["down_cont"][i_dn], P[mo]["down_fade"][i_up]])
        fade = np.concatenate([P[mo]["down_fade"][i_dn], P[mo]["down_cont"][i_up]])
        v5[mo] = {"P_cont_beats_fade": round(float((cont > fade).mean()), 4),
                  "mean_cont_bp": round(float(cont.mean()), 2), "mean_fade_bp": round(float(fade.mean()), 2)}
    rep["verdicts"]["V5_fade_vs_cont_label"] = v5
    log("V5 09-06 라벨 P(지속>페이드) · " + " · ".join(
        f"{mo} {v5[mo]['P_cont_beats_fade']:.4f} (지속 {v5[mo]['mean_cont_bp']:+.2f} 페이드 {v5[mo]['mean_fade_bp']:+.2f})" for mo in MODES))

    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    log(f"저장 {OUT/'report.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
