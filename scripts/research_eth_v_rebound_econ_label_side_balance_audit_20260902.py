#!/usr/bin/env python3
"""E0 경제라벨 결과의 **방향 쏠림 아티팩트** 검증 -- 진짜 스킬인가 드리프트인가.

## 왜 의심하는가

`..._direct_economic_label_20260902.py`의 E0_binary|F0가 VAL 갭 +10.30bp(귀무 100%),
OOS 갭 +9.71bp(귀무 100%)로 이 프로그램 최초로 양쪽 관문을 통과했다. 그런데 앞뒤가 안 맞는다:

  · **자기라벨 AUC 0.5131 / 0.5194** -- 사실상 무작위. 분류력 없는 모델이 왜 +10bp 갭을?
  · 라벨률 0.7572, 아무 봉이나 진입해도 셀(5.0/1.5/0.1) 중앙 net이 **+25.50bp**.
    라벨이 거의 상수라 배울 게 별로 없다.

**가장 그럴듯한 대안 설명: 방향 쏠림.** 호출이 롱으로 몰려 있고 그 구간에 상승 드리프트가
있었다면, 뒤집기(=그 봉들을 전부 숏)는 **스킬과 무관하게** 손실이 난다. 기존 귀무분포는
전체에서 무작위 추출하므로 롱/숏이 50:50이 되어 이 효과를 잡지 못한다.

## 세 가지 검정

  1. 호출의 **롱/숏 비율** (원 실험이 기록하지 않음)
  2. ⭐**측면별 갭** -- 롱 호출만 / 숏 호출만 따로 duel. 진짜 방향 스킬이면 **양쪽 다** 갭이
     양수여야 한다. 드리프트 아티팩트면 한쪽만 크게 양수고 반대쪽은 음수다. **이게 결정적.**
  3. **측면비율 매칭 귀무분포**(B=200) -- 모델의 롱/숏 비율을 그대로 유지한 채 무작위 추출.
     갭이 사라지면 "모델이 고른 것"이 아니라 "그 비율로 고른 것"이 원인이다.

부수: 구간별 시장 드리프트(단순 보유수익)를 같이 찍어 맥락을 준다.

⚠️HOLDOUT 미터치. 라이브 코드 변경 없음.

Run on the server via handoff.
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
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_s1 = _load("s1_audit", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
_bt = _s1._bt
TIER0 = _s1.FEATURE_COLUMNS
FORWARD_BARS = _s1.FORWARD_BARS
SL_GRID, ARM_GRID, TRAIL_GRID = _bt.SL_GRID, _bt.ARM_GRID, _bt.TRAIL_GRID

LABEL_CELL = (5.0, 1.5, 0.1)
COST_BP, ARTIFACT_FREE_MIN = 10.0, 1.0
CONTEXT_N, SEED = 18000, 20260829
NULL_B, NULL_SEED = 200, 20260902
CHUNK = 40000
TARGET_N = {"VAL": 1693, "OOS": 1367}
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
OUT = ROOT / "data/research/eth_v_rebound_econ_side_audit_20260902/report.json"


def log(m): print(f"[sideaudit] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    import torch
    log(f"cuda: {torch.cuda.is_available()}")

    _s1.VAL_END = OOS_END
    log("building frame ...")
    sig, feat, eth = _s1.build_sig()
    sb = _s1.label_param(sig, True, ambig="drop", anchor="wick",
                         atr_mult=1.50, t_sustain=0.20, full_bars=12)
    st = _s1.label_param(sig, False, ambig="drop", anchor="wick",
                         atr_mult=1.50, t_sustain=0.20, full_bars=12)
    long = _s1.long_frame_for(sig, feat, sb, st)
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                     np.where(long["timestamp"] < VAL_END, "VAL", "OOS"))
    assert long["timestamp"].max() < OOS_END, "HOLDOUT 누출"

    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    pos_of = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    nk = len(kl)
    long["pos"] = [pos_of.get(np.datetime64(t.tz_localize(None)), -1) for t in long["timestamp"]]
    long = long.loc[(long["pos"] >= 0) & (long["pos"] + FORWARD_BARS + 1 < nk)].reset_index(drop=True)

    # ---- 시장 드리프트 (맥락) ----
    log("")
    log("=== 구간별 시장 드리프트 (단순 보유) ===")
    drift = {}
    for spn, (a, b) in (("TRAIN", (None, TRAIN_END)), ("VAL", (TRAIN_END, VAL_END)),
                        ("OOS", (VAL_END, OOS_END))):
        m = (eth["timestamp"] < b) if a is None else ((eth["timestamp"] >= a) & (eth["timestamp"] < b))
        px = eth.loc[m, "close"].to_numpy()
        if len(px) > 1:
            drift[spn] = round(float(px[-1] / px[0] - 1) * 100, 2)
            log(f"  {spn}: {drift[spn]:+.2f}%")

    # ---- E0 라벨 재현 ----
    sl_, arm_, tr_ = LABEL_CELL
    i_all = long["pos"].to_numpy().astype(int)
    sgn_all = np.where(long["is_downside"].to_numpy() == 1, 1.0, -1.0)
    atr_all = long["atr"].to_numpy(dtype=float)
    net = np.full(len(long), np.nan)
    for s_ in range(0, len(long), CHUNK):
        e_ = min(s_ + CHUNK, len(long))
        idx = i_all[s_:e_]
        H = np.stack([h[j+1:j+1+FORWARD_BARS] for j in idx])
        L = np.stack([l[j+1:j+1+FORWARD_BARS] for j in idx])
        C = np.stack([c[j+1:j+1+FORWARD_BARS] for j in idx])
        net[s_:e_] = _bt.simulate_trailing_vec(o[idx+1], atr_all[s_:e_], sgn_all[s_:e_],
                                               H, L, C, sl_, arm_, tr_, True) * 1e4 - COST_BP
    long["y"] = (net > 0).astype(float)
    log("")
    log(f"E0 라벨 재현: 라벨률 {long['y'].mean():.4f}  net 중앙 {np.nanmedian(net):+.2f}bp")
    for side, nm in ((1, "bottom(롱)"), (0, "top(숏)")):
        m = long["is_downside"] == side
        log(f"  {nm} 라벨률 {long.loc[m,'y'].mean():.4f}  net 중앙 {np.nanmedian(net[m.to_numpy()]):+.2f}bp")

    tr_set = long.loc[long["split"] == "TRAIN"]
    rng = np.random.default_rng(SEED)
    ctx = tr_set.iloc[np.sort(rng.choice(len(tr_set), size=min(CONTEXT_N, len(tr_set)), replace=False))]
    clf = TabPFNClassifier(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
    clf.fit(ctx[TIER0], ctx["y"].to_numpy())

    def build(s):
        rows = []
        for i_, isd, atr__ in zip(s["pos"].to_numpy(), s["is_downside"].to_numpy(), s["atr"].to_numpy()):
            i = int(i_)
            rows.append({"side": "long" if isd == 1 else "short", "atr": float(atr__),
                         "entry_price": float(o[i+1]),
                         "fwd_open": o[i+1:i+1+FORWARD_BARS], "fwd_high": h[i+1:i+1+FORWARD_BARS],
                         "fwd_low": l[i+1:i+1+FORWARD_BARS], "fwd_close": c[i+1:i+1+FORWARD_BARS]})
        return pd.DataFrame(rows)

    def duel(df):
        if len(df) < 30:
            return None
        e, a, s_, H, L, C = _bt.pack(df)
        ef, af, sf, Hf, Lf, Cf = _bt.pack(df, flip=True)
        fw, flp = [], []
        for sl in SL_GRID:
            for arm in ARM_GRID:
                if arm < ARTIFACT_FREE_MIN:
                    continue
                for trv in TRAIL_GRID:
                    fw.append(float((_bt.simulate_trailing_vec(e, a, s_, H, L, C, sl, arm, trv, True)*1e4-COST_BP).mean()))
                    flp.append(float((_bt.simulate_trailing_vec(ef, af, sf, Hf, Lf, Cf, sl, arm, trv, True)*1e4-COST_BP).mean()))
        fw, flp = np.array(fw), np.array(flp)
        return {"n": int(len(df)), "med_fwd": float(np.median(fw)), "med_flip": float(np.median(flp)),
                "gap_med": float(np.median(fw)-np.median(flp)), "best_fwd": float(fw.max())}

    report = {"signal": "v_rebound_econ_label_side_balance_audit", "asset": "ETHUSDT",
              "scope": {"question": "E0 갭 +10bp가 방향 스킬인가 방향 쏠림 드리프트인가",
                        "label_cell": list(LABEL_CELL), "drift_pct": drift,
                        "holdout_touched": False, "live_code_changed": False}, "splits": {}}

    nrng = np.random.default_rng(NULL_SEED)
    for spn in ("VAL", "OOS"):
        s = long.loc[long["split"] == spn].copy()
        s["p"] = np.concatenate([clf.predict_proba(s[TIER0].iloc[k:k+20000])[:, 1]
                                 for k in range(0, len(s), 20000)])
        sel = s.nlargest(min(TARGET_N[spn], len(s)), "p")
        n_long = int((sel["is_downside"] == 1).sum()); n_short = len(sel) - n_long
        pool_long_frac = float((s["is_downside"] == 1).mean())
        log("")
        log(f"===== {spn}  호출 {len(sel):,} =====")
        log(f"  1) 롱/숏 비율: 롱 {n_long:,} ({n_long/len(sel)*100:.1f}%) / 숏 {n_short:,} "
            f"({n_short/len(sel)*100:.1f}%)   [전체 풀은 롱 {pool_long_frac*100:.1f}%]")

        overall = duel(build(sel))
        log(f"  전체 갭 {overall['gap_med']:+.2f}bp  (정 {overall['med_fwd']:+.2f} / 뒤 {overall['med_flip']:+.2f})")

        log("  2) ⭐측면별 갭 (진짜 스킬이면 양쪽 다 양수)")
        per_side = {}
        for side, nm in ((1, "롱(bottom)"), (0, "숏(top)")):
            sub = sel.loc[sel["is_downside"] == side]
            d = duel(build(sub)) if len(sub) >= 30 else None
            per_side[nm] = d
            if d:
                flag = "✅" if d["gap_med"] > 0 else "❌"
                log(f"     {nm:12s} n={d['n']:>5,}  정 {d['med_fwd']:+7.2f}  뒤 {d['med_flip']:+7.2f}  "
                    f"갭 {d['gap_med']:+7.2f}bp {flag}")
            else:
                log(f"     {nm:12s} 표본 부족")

        log("  3) 측면비율 매칭 귀무분포")
        gaps = []
        pool_l = s.loc[s["is_downside"] == 1]; pool_s = s.loc[s["is_downside"] == 0]
        for _ in range(NULL_B):
            a_ = pool_l.iloc[nrng.choice(len(pool_l), size=min(n_long, len(pool_l)), replace=False)]
            b_ = pool_s.iloc[nrng.choice(len(pool_s), size=min(n_short, len(pool_s)), replace=False)]
            rd = duel(build(pd.concat([a_, b_])))
            if rd:
                gaps.append(rd["gap_med"])
        pct = round(float((np.array(gaps) < overall["gap_med"]).mean()*100), 1) if len(gaps) >= 20 else None
        log(f"     측면매칭 귀무 평균 갭 {np.mean(gaps):+.2f}bp   관측 {overall['gap_med']:+.2f}bp   "
            f"백분위 {pct}%{'  ✅스킬' if (pct or 0) >= 95 else '  ❌쏠림 아티팩트'}")

        report["splits"][spn] = {"n_calls": int(len(sel)), "n_long": n_long, "n_short": n_short,
                                 "pool_long_frac": round(pool_long_frac, 4),
                                 "overall": {k: round(v, 3) if isinstance(v, float) else v
                                             for k, v in overall.items()},
                                 "per_side": {k: ({kk: round(vv, 3) if isinstance(vv, float) else vv
                                                   for kk, vv in v.items()} if v else None)
                                              for k, v in per_side.items()},
                                 "side_matched_null_mean_gap": round(float(np.mean(gaps)), 3) if gaps else None,
                                 "side_matched_null_pctile": pct}

    log("")
    log("=== 판정 ===")
    ok = True
    for spn, v in report["splits"].items():
        sides = [d["gap_med"] for d in v["per_side"].values() if d]
        both_pos = all(x > 0 for x in sides) and len(sides) == 2
        nullok = (v["side_matched_null_pctile"] or 0) >= 95
        ok = ok and both_pos and nullok
        log(f"  {spn}: 측면별 양쪽 양수 {'✅' if both_pos else '❌'}   "
            f"측면매칭 귀무 {v['side_matched_null_pctile']}% {'✅' if nullok else '❌'}")
    log(f"  ⇒ {'✅진짜 방향 스킬' if ok else '❌방향 쏠림 아티팩트 -- E0 결과 무효'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    report["runtime_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
