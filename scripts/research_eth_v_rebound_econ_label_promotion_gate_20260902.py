#!/usr/bin/env python3
"""E0 경제라벨 후보의 **승격 게이트** -- 단일셀 OOS 실측 / 5시드 안정성 / 레짐·월별 분해.

## 배경

`..._direct_economic_label_20260902.py`의 `E0_binary|Tier0`가 VAL 갭 +10.30bp / OOS +9.71bp
(둘 다 귀무 100%)로 통과했고, `..._econ_label_side_balance_audit_20260902.py`가
**방향 쏠림 아티팩트가 아님**을 확인했다(양 측면 갭 전부 양수, 측면매칭 귀무 100%).

그러나 아직 승격 불가 -- 세 가지가 미확인이다:

  1. **단일 셀 OOS 실측**: 지금까지의 갭은 80셀 **중앙값**이다. 실제 거래는 셀 하나를 쓴다.
     VAL에서 (임계값, 셀)을 고르고 **OOS에서 1회** 평가해야 정직한 out-of-sample 숫자다.
     ⚠️라이브는 분위가 아니라 **고정 확률 임계값**으로 돈다 -- VAL에서 고른 확률컷을
     OOS에 그대로 적용한 값(현실)과 같은 분위를 적용한 값(진단) 둘 다 낸다.
  2. **N>=5 시드 안정성**: 이 저장소의 프로모션 게이트. 컨텍스트 재추출 시드 5개.
  3. **레짐 편중**: TRAIN +90.34% / VAL -32.09% / OOS -29.23% -- 검증 구간이 둘 다 하락장이다.
     월별로 쪼개 특정 구간에 몰려 있는지 본다.

## 통과 기준 (사전 등록)

  · VAL 선정 셀의 **OOS 기대값 > 0** (고정 확률컷 적용 기준)
  · 같은 셀 **뒤집기 < 정방향**
  · **5시드 전부 OOS 기대값 > 0** (부호 일치) -- 하나라도 음수면 미통과
  · OOS 월별 3개월 중 **2개월 이상 양수**

⚠️HOLDOUT 미터치(이 게이트를 통과해야만 1회 노출을 검토한다). 라이브 코드 변경 없음.

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


_s1 = _load("s1_gate", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
_bt = _s1._bt
TIER0 = _s1.FEATURE_COLUMNS
FORWARD_BARS = _s1.FORWARD_BARS
SL_GRID, ARM_GRID, TRAIL_GRID = _bt.SL_GRID, _bt.ARM_GRID, _bt.TRAIL_GRID

LABEL_CELL = (5.0, 1.5, 0.1)
COST_BP, ARTIFACT_FREE_MIN = 10.0, 1.0
CONTEXT_N = 18000
SEEDS = [20260829, 141592, 271828, 577215, 20260902]
TAIL_FRACS = [0.005, 0.01, 0.02, 0.05, 0.10]
CHUNK = 40000
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
OUT = ROOT / "data/research/eth_v_rebound_econ_promotion_gate_20260902/report.json"


def log(m): print(f"[gate] {m}", flush=True)


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

    sl0, arm0, tr0 = LABEL_CELL
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
                                               H, L, C, sl0, arm0, tr0, True) * 1e4 - COST_BP
    long["y"] = (net > 0).astype(float)
    log(f"E0 라벨: 라벨률 {long['y'].mean():.4f}")

    def build(s):
        rows = []
        for i_, isd, atr__ in zip(s["pos"].to_numpy(), s["is_downside"].to_numpy(), s["atr"].to_numpy()):
            i = int(i_)
            rows.append({"side": "long" if isd == 1 else "short", "atr": float(atr__),
                         "entry_price": float(o[i+1]),
                         "fwd_open": o[i+1:i+1+FORWARD_BARS], "fwd_high": h[i+1:i+1+FORWARD_BARS],
                         "fwd_low": l[i+1:i+1+FORWARD_BARS], "fwd_close": c[i+1:i+1+FORWARD_BARS]})
        return pd.DataFrame(rows)

    def cell_pnl(df, cell):
        """한 셀의 트레이드별 순손익(bp) 배열 -- 정방향/뒤집기."""
        sl, arm, trv = cell
        e, a, s_, H, L, C = _bt.pack(df)
        ef, af, sf, Hf, Lf, Cf = _bt.pack(df, flip=True)
        f = _bt.simulate_trailing_vec(e, a, s_, H, L, C, sl, arm, trv, True)*1e4-COST_BP
        r = _bt.simulate_trailing_vec(ef, af, sf, Hf, Lf, Cf, sl, arm, trv, True)*1e4-COST_BP
        return f, r

    def stats(pnl):
        w = pnl > 0
        return {"n": int(len(pnl)), "exp_bp": round(float(pnl.mean()), 3),
                "win_rate": round(float(w.mean()), 4),
                "payoff": round(float(pnl[w].mean()/-pnl[~w].mean()), 3) if w.any() and (~w).any() else None,
                "total_bp": round(float(pnl.sum()), 1)}

    report = {"signal": "v_rebound_econ_label_promotion_gate", "asset": "ETHUSDT",
              "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
              "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
              "scope": {"label": "E0_binary (트레일링 순손익>0), 셀 사전지정 " + str(LABEL_CELL),
                        "features": "Tier0 23", "seeds": SEEDS, "cost_bp": COST_BP,
                        "selection": "VAL에서 (분위, 셀) 선정 -> OOS 1회, 재최적화 없음",
                        "holdout_touched": False, "live_code_changed": False},
              "seeds": {}}

    per_seed_oos = []
    for si, sd in enumerate(SEEDS):
        tr_set = long.loc[long["split"] == "TRAIN"]
        rng = np.random.default_rng(sd)
        ctx = tr_set.iloc[np.sort(rng.choice(len(tr_set), size=min(CONTEXT_N, len(tr_set)), replace=False))]
        clf = TabPFNClassifier(device="cuda", random_state=sd, ignore_pretraining_limits=True)
        clf.fit(ctx[TIER0], ctx["y"].to_numpy())
        sc = {}
        for spn in ("VAL", "OOS"):
            s = long.loc[long["split"] == spn].copy()
            s["p"] = np.concatenate([clf.predict_proba(s[TIER0].iloc[k:k+20000])[:, 1]
                                     for k in range(0, len(s), 20000)])
            sc[spn] = s
        log("")
        log(f"########## seed {sd} ({si+1}/{len(SEEDS)}) ##########")

        # --- VAL에서 (분위, 셀) 선정 ---
        best = None
        for frac in TAIL_FRACS:
            k = max(30, int(round(len(sc["VAL"]) * frac)))
            sel = sc["VAL"].nlargest(k, "p")
            cut = float(sel["p"].min())
            df = build(sel)
            for sl in SL_GRID:
                for arm in ARM_GRID:
                    if arm < ARTIFACT_FREE_MIN:
                        continue
                    for trv in TRAIL_GRID:
                        f, r = cell_pnl(df, (sl, arm, trv))
                        if best is None or f.mean() > best["val_exp"]:
                            best = {"frac": frac, "cut": cut, "cell": (sl, arm, trv),
                                    "val_exp": float(f.mean()), "val_flip": float(r.mean()),
                                    "val_stats": stats(f), "val_n": int(len(df))}
        log(f"  [VAL 선정] 상위{best['frac']*100:g}% (p>={best['cut']:.4f})  "
            f"셀 {best['cell']}  기대값 {best['val_exp']:+.2f}bp  "
            f"승률 {best['val_stats']['win_rate']*100:.1f}%  손익비 {best['val_stats']['payoff']}  "
            f"뒤집기 {best['val_flip']:+.2f}bp")

        # --- OOS 1회: 고정 확률컷(현실) + 동일 분위(진단) ---
        oos = sc["OOS"]
        res = {}
        for mode, seln in (("고정확률컷", oos.loc[oos["p"] >= best["cut"]]),
                           ("동일분위", oos.nlargest(max(30, int(round(len(oos)*best["frac"]))), "p"))):
            if len(seln) < 30:
                log(f"  [OOS/{mode}] 호출 {len(seln)} -- 표본 부족"); continue
            f, r = cell_pnl(build(seln), best["cell"])
            stt = stats(f)
            days = (seln["timestamp"].max() - seln["timestamp"].min()).total_seconds()/86400
            log(f"  [OOS/{mode}] n={stt['n']:,} ({stt['n']/max(days,1):.1f}건/일)  "
                f"기대값 {stt['exp_bp']:+.2f}bp  승률 {stt['win_rate']*100:.1f}%  "
                f"손익비 {stt['payoff']}  누적 {stt['total_bp']:+.0f}bp  "
                f"뒤집기 {r.mean():+.2f}bp"
                f"{'  ✅' if stt['exp_bp'] > 0 and r.mean() < stt['exp_bp'] else '  ❌'}")
            # 월별
            mo = seln.assign(pnl=f).groupby(seln["timestamp"].dt.to_period("M"))["pnl"]
            monthly = {str(k): {"n": int(len(v)), "exp_bp": round(float(v.mean()), 2)}
                       for k, v in mo}
            log(f"      월별: " + "  ".join(f"{k} {v['exp_bp']:+.2f}bp(n={v['n']})"
                                            for k, v in monthly.items()))
            res[mode] = {**stt, "flip_exp_bp": round(float(r.mean()), 3), "monthly": monthly}
            if mode == "고정확률컷":
                per_seed_oos.append(stt["exp_bp"])
        report["seeds"][str(sd)] = {"val_selection": {**{k: v for k, v in best.items()
                                                          if k != "cell"}, "cell": list(best["cell"])},
                                    "oos": res}

    log("")
    log("=== ⭐5시드 안정성 (OOS 고정확률컷 기대값) ===")
    arr = np.array(per_seed_oos)
    log(f"  시드별: " + "  ".join(f"{x:+.2f}" for x in arr))
    log(f"  평균 {arr.mean():+.2f}bp  std {arr.std():.2f}  양수 {int((arr>0).sum())}/{len(arr)}")
    passed = len(arr) == len(SEEDS) and bool((arr > 0).all())
    log("")
    log(f"=== 판정: {'✅5시드 전부 OOS 양수 -- HOLDOUT 노출 검토 가능' if passed else '❌미통과'} ===")
    report["seed_stability"] = {"oos_exp_bp": [round(float(x), 3) for x in arr],
                                "mean": round(float(arr.mean()), 3), "std": round(float(arr.std()), 3),
                                "n_positive": int((arr > 0).sum()), "passed": passed}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report["runtime_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
