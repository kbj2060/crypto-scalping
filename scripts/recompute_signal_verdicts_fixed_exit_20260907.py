#!/usr/bin/env python3
"""신호별·측면별 트레일링 판정 재계산 (2026-09-07) — 08-30 비용게이트 계열 포함.

`recompute_frame_labels_fixed_exit_20260907.py`가 저장한 봉별 라벨(legacy/fixed/hold)을 써서,
증거신호 8종의 **첫발동(GAP=12) 모집단**을 신호별·측면별로 갈라 두 방향(페이드/지속) 판정을 다시 낸다.
08-30 "트레일링 비용게이트 돌파/확인"(taker_delta_climax · short_term_return_z)과
"한계"(v_rebound 계열)가 여기 신호 단위 판정에 해당한다 -- 그 판정들은 **신호 자기 방향(페이드)**을
트레일링으로 청산한 것이다.
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
sys.path.insert(0, str(ROOT / "scripts"))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


V2 = _load("hev2_sv", "scripts/research_homer_entry_v2_20260904.py")
portfolio, day_boot, stats_of = V2.portfolio, V2.day_boot, V2.stats_of
SIGNALS, OOFD = V2.SIGNALS, V2.OOFD_MAT
OUT = ROOT / "data/research/eth_fixed_exit_recompute_20260907"
GAP, MAX_CONC, B_BOOT = 12, 5, 1000
WINDOWS = ("VAL", "OOS")
MODES = ("legacy", "fixed")


def log(m): print(f"[sv] {m}", flush=True)


def pf(pnl, ts, pos, ex, rng):
    cand = pd.DataFrame({"timestamp": ts, "pos": pos, "p": np.ones(len(pos)), "entry_bar": pos + 1,
                         "exit_bar": pos + 1 + ex + 1, "pnl_bp": pnl})
    r = portfolio(cand, MAX_CONC)
    if r is None or len(r["trades"]) < 30:
        return None
    lo, hi = day_boot(r["trades"]["pnl_bp"], r["trades"]["timestamp"], B_BOOT, rng)
    return {"n": r["n"], "exp_bp": round(r["exp_bp"], 2), "win_rate": round(r["win_rate"], 3),
            "day_ci95": [round(lo, 2), round(hi, 2)]}


def main():
    t0 = time.time(); rng = np.random.default_rng(20260907)
    B = pd.read_parquet(OUT / "bar_labels.parquet")
    idx = pd.Series(np.arange(len(B)), index=B["pos"].to_numpy())
    ts = B["timestamp"].to_numpy(); pos = B["pos"].to_numpy(); split = B["split"].to_numpy()

    res = {"windows": WINDOWS, "gap": GAP, "max_concurrent": MAX_CONC, "signals": {}}
    for sname in SIGNALS:
        d = pd.read_csv(OOFD / f"{sname}_oof.csv", usecols=["pos", "side"]).drop_duplicates(["pos", "side"]).sort_values("pos")
        d["is_downside"] = (d["side"] == "bottom").astype(np.int8)
        keep_all = np.zeros(len(d), bool)
        for sd in (0, 1):
            ii = np.flatnonzero(d["is_downside"].to_numpy() == sd); pp = d["pos"].to_numpy()[ii]
            k = np.zeros(len(pp), bool); last = -10**9
            for j, x in enumerate(pp):
                if x - last > GAP:
                    k[j] = True
                last = x
            keep_all[ii] = k
        d = d.loc[keep_all]
        res["signals"][sname] = {}
        for side_tag, sd in (("bottom", 1), ("top", 0)):
            rows = d.loc[d["is_downside"] == sd, "pos"]
            i = idx.reindex(rows).dropna().astype(int).to_numpy()
            if len(i) < 60:
                continue
            cell = {"n_fires": int(len(i))}
            for dir_tag in ("fade", "cont"):
                # 바닥발동: 페이드=롱(down_fade) · 천장발동: 페이드=숏(down_cont)
                col = ("down_fade" if sd == 1 else "down_cont") if dir_tag == "fade" else ("down_cont" if sd == 1 else "down_fade")
                for w in WINDOWS:
                    m = split[i] == w
                    for mo in MODES:
                        st = pf(B[f"{col}_{mo}_bp"].to_numpy()[i][m], ts[i][m], pos[i][m],
                                B[f"{col}_{mo}_ex"].to_numpy()[i][m], rng)
                        cell[f"{dir_tag}_{w}_{mo}"] = st
            res["signals"][sname][side_tag] = cell
            def g(k):
                v = cell.get(k); return f"{v['exp_bp']:+6.2f}" if v else "   n/a"
            log(f"{sname:28s} {side_tag:6s} n={cell['n_fires']:5,} · "
                f"페이드 VAL {g('fade_VAL_legacy')}→{g('fade_VAL_fixed')} OOS {g('fade_OOS_legacy')}→{g('fade_OOS_fixed')} · "
                f"지속 VAL {g('cont_VAL_legacy')}→{g('cont_VAL_fixed')} OOS {g('cont_OOS_legacy')}→{g('cont_OOS_fixed')}")

    # 요약: legacy에서 두 창 양수였던 (신호,측면,방향) 조합이 fixed에서 몇 개 살아남는가
    surv = {"legacy_two_window_positive": [], "fixed_two_window_positive": []}
    for sname, sides in res["signals"].items():
        for side_tag, cell in sides.items():
            for dir_tag in ("fade", "cont"):
                for mo in MODES:
                    v = [cell.get(f"{dir_tag}_{w}_{mo}") for w in WINDOWS]
                    if all(x and x["exp_bp"] > 0 for x in v):
                        surv[f"{mo}_two_window_positive"].append(f"{sname}/{side_tag}/{dir_tag}")
    res["survivors"] = surv
    log(f"\n두 창 모두 양수: legacy {len(surv['legacy_two_window_positive'])}개 → fixed {len(surv['fixed_two_window_positive'])}개")
    if surv["legacy_two_window_positive"]:
        log("  legacy: " + ", ".join(surv["legacy_two_window_positive"]))
    if surv["fixed_two_window_positive"]:
        log("  fixed:  " + ", ".join(surv["fixed_two_window_positive"]))
    (OUT / "signal_verdicts.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    log(f"저장 {OUT/'signal_verdicts.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
