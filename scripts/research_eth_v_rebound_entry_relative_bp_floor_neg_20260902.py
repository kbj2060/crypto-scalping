#!/usr/bin/env python3
"""**진입 기준** bp 하한 미달 양성을 **음성(0)으로 학습** -- 모델의 '이미 끝난 봉' 선택 편향 교정 시도.

## 사용자 지적에서 나온 처방 (2026-09-02)

"익절가가 너무 가까워. 수수료도 못 벌고 나오고 있는 걸 라벨 1로 모델한테 가르치고 있잖아."

## 진단 요약 (이 처방이 겨누는 것)

  · 라벨 양성의 **진입 기준** 먹을수있는폭 중앙 42bp -- 라벨 자체는 대체로 거래 가능하다.
  · 그런데 **모델이 부르는 양성**의 소진율은 125%(라벨 양성 전체는 40%).
    → 모델은 양성 중에서도 **이미 끝난 것**을 골라낸다. Tier0 23피쳐가 전부 "이 봉에서 방금
      무슨 일이 있었나"를 재는 값이라(`sweep_penetration_atr`/`lower_wick_ratio`/`delta_z`/
      `ret3_z`) 소진이 큰 봉이 곧 고확률 봉이 된다.
  · 9-9의 bp 하한은 `fast_close_max - low[i]`(**앵커 기준**)라 이 갭을 전혀 못 막았다
    (비용 미만 비율: 앵커 기준 0.3~0.7% vs 진입 기준 4.3~6.5%).

## 이 실험

`capturable_bp = (fast_close_max - open[i+1]) / open[i+1]`(진입가 기준 30분 창 최대 이익)이
FLOOR 미만인 **양성을 음성(0)으로 재라벨**한다.

⚠️9-9는 같은 상황에서 **제외(ambiguous)**를 택했다("실제로 1.5×ATR 움직였으니 실패로 가르치면
틀린 학습"). 여기서는 반대로 간다 -- **거래 관점에서 못 먹은 반등은 실패**이고, 소진율은 결정
시점에 계산 가능한 값이므로(`(close[i]-low[i])/1.5ATR`) 모델이 배울 수 있는 구분이다.
**음성으로 가르쳐야 모델이 소진된 봉을 부르지 않게 된다** -- 제외하면 그냥 안 보여줄 뿐이다.

## ⭐메커니즘 검증 (헤드라인보다 먼저 볼 것)

이 처방이 작동한다면 **모델 호출의 소진율이 125%에서 내려가야** 한다. 안 내려가면 라벨을
바꿔도 선택 편향이 그대로라는 뜻이고, 경제성 숫자는 볼 필요가 없다. 그래서 호출의
소진율/먹을수있는폭 중앙값을 FLOOR별로 같이 찍는다.

## 판정

⚠️FLOOR마다 라벨 정의가 다르므로 자기라벨 AUC 직접비교 금지. **고정 타깃**(FLOOR=30 라벨)에서
같이 재고, **호출 빈도를 대조군(FLOOR=0, thr 0.60)에 일치**시킨 뒤 방향뒤집기 통제 경제성으로
판정한다. 셀은 VAL에서만 선정, ARM>=1.0.

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


_s1 = _load("s1_floor", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
_vs, _bt = _s1._vs, _s1._bt
FEATURE_COLUMNS = _s1.FEATURE_COLUMNS
FAST_BARS, FORWARD_BARS = _s1.FAST_BARS_FIXED, _s1.FORWARD_BARS
SL_GRID, ARM_GRID, TRAIL_GRID = _bt.SL_GRID, _bt.ARM_GRID, _bt.TRAIL_GRID

DEPLOYED = {"atr_mult": 1.50, "t_sustain": 0.20, "full_bars": 12}
FLOORS = [0, 10, 20, 30, 40]
REF_FLOOR = 30
BASE_THR = 0.60
CONTEXT_N, SEED = 18000, 20260829
COST_BP, ARTIFACT_FREE_MIN = 10.0, 1.0
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
OUT = ROOT / "data/research/eth_v_rebound_entry_bp_floor_neg_20260902/report.json"


def log(m): print(f"[floorneg] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    from sklearn.metrics import roc_auc_score
    import torch
    log(f"cuda: {torch.cuda.is_available()}")

    _s1.VAL_END = OOS_END
    log("building frame ...")
    sig, feat, eth = _s1.build_sig()
    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    pos_of = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    nk = len(kl)

    close, high, low = (sig[x].to_numpy() for x in ("close", "high", "low"))
    op, atr = sig["open"].to_numpy(), sig["atr"].to_numpy()
    pre_atr = _vs.shifted_at(atr, -1)
    fmax = _vs.fwd_window(close, 1, FAST_BARS, "max")
    fmin = _vs.fwd_window(close, 1, FAST_BARS, "min")

    base = {True: _s1.label_param(sig, True, ambig="drop", anchor="wick", **DEPLOYED),
            False: _s1.label_param(sig, False, ambig="drop", anchor="wick", **DEPLOYED)}

    def cap_bp(is_down):
        ent = _vs.shifted_at(op, 1)
        sgn = 1.0 if is_down else -1.0
        return sgn * ((fmax if is_down else fmin) - ent) / ent * 1e4

    def cons(is_down):
        anc = low if is_down else high
        ent = _vs.shifted_at(op, 1)
        sgn = 1.0 if is_down else -1.0
        return sgn * (ent - anc) / (1.5 * pre_atr)

    CAP = {d: cap_bp(d) for d in (True, False)}
    CONS = {d: cons(d) for d in (True, False)}

    def status_floor(is_down, floor):
        """하한 미달 양성 -> 음성(chop). 나머지는 배포판 라벨 그대로."""
        st = base[is_down].copy()
        if floor > 0:
            bad = (st == "v_rebound") & np.isfinite(CAP[is_down]) & (CAP[is_down] < floor)
            st[bad] = "chop"
        return st

    def frame_for(floor):
        lf = _s1.long_frame_for(sig, feat, status_floor(True, floor), status_floor(False, floor))
        lf["split"] = np.where(lf["timestamp"] < TRAIN_END, "TRAIN",
                        np.where(lf["timestamp"] < VAL_END, "VAL", "OOS"))
        assert lf["timestamp"].max() < OOS_END, "HOLDOUT 누출"
        lf["pos"] = [pos_of.get(np.datetime64(t.tz_localize(None)), -1) for t in lf["timestamp"]]
        lf = lf.loc[lf["pos"] >= 0].reset_index(drop=True)
        i = lf["pos"].to_numpy().astype(int)
        dn = lf["is_downside"].to_numpy() == 1
        lf["cap_bp"] = np.where(dn, CAP[True][i], CAP[False][i])
        lf["consumed"] = np.where(dn, CONS[True][i], CONS[False][i])
        return lf

    # 고정 평가 타깃 (FLOOR=REF 라벨) -- 순환성 caveat은 9-9와 동일하게 병기
    ref = frame_for(REF_FLOOR)
    ref_val = ref.loc[(ref["split"] == "VAL") & ref["label"].notna()][["timestamp", "is_downside", "label"]]
    ref_val = ref_val.rename(columns={"label": "ref_label"})
    log(f"고정 타깃: FLOOR={REF_FLOOR} 라벨, VAL n={len(ref_val):,} (base {ref_val['ref_label'].mean():.4f})")

    def build(s):
        rows = []
        for i_, isd, atr_ in zip(s["pos"].to_numpy(), s["is_downside"].to_numpy(), s["atr"].to_numpy()):
            i = int(i_)
            if i + FORWARD_BARS + 1 >= nk:
                continue
            rows.append({"side": "long" if isd == 1 else "short", "atr": float(atr_),
                         "entry_price": float(o[i+1]),
                         "fwd_open": o[i+1:i+1+FORWARD_BARS], "fwd_high": h[i+1:i+1+FORWARD_BARS],
                         "fwd_low": l[i+1:i+1+FORWARD_BARS], "fwd_close": c[i+1:i+1+FORWARD_BARS]})
        return pd.DataFrame(rows)

    def grid(df):
        if len(df) < 30:
            return None
        e, a, s_, H, L, C = _bt.pack(df)
        ef, af, sf, Hf, Lf, Cf = _bt.pack(df, flip=True)
        fwd = flip = 0; best = None
        for sl in SL_GRID:
            for arm in ARM_GRID:
                if arm < ARTIFACT_FREE_MIN:
                    continue
                for tr in TRAIL_GRID:
                    ov = _bt.simulate_trailing_vec(e, a, s_, H, L, C, sl, arm, tr, False)
                    pv = _bt.simulate_trailing_vec(e, a, s_, H, L, C, sl, arm, tr, True)
                    fo = _bt.simulate_trailing_vec(ef, af, sf, Hf, Lf, Cf, sl, arm, tr, False)
                    fp = _bt.simulate_trailing_vec(ef, af, sf, Hf, Lf, Cf, sl, arm, tr, True)
                    ob, pb = float(ov.mean()*1e4-COST_BP), float(pv.mean()*1e4-COST_BP)
                    fwd += int(ob > 0 and pb > 0)
                    flip += int(float(fo.mean()*1e4-COST_BP) > 0 and float(fp.mean()*1e4-COST_BP) > 0)
                    if best is None or pb > best["pess_bp"]:
                        best = {"sl": sl, "arm": arm, "trail": tr, "opt_bp": round(ob, 2),
                                "pess_bp": round(pb, 2),
                                "win_rate": round(float((pv*1e4 > COST_BP).mean()), 4)}
        return {"n": int(len(df)), "fwd_pass": fwd, "flip_pass": flip,
                "margin": fwd - flip, "best": best}

    report = {"signal": "v_rebound_entry_relative_bp_floor_negative", "asset": "ETHUSDT",
              "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
              "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
              "scope": {"treatment": "capturable_bp < FLOOR 인 양성을 **음성(0)**으로 재라벨",
                        "capturable_bp": "(fast_close_max - open[i+1]) / open[i+1]",
                        "contrast_with_9_9": "9-9는 앵커 기준 + 제외(ambiguous). 여기는 진입 기준 + 음성",
                        "floors": FLOORS, "ref_floor": REF_FLOOR, "base_thr": BASE_THR,
                        "cost_bp": COST_BP, "context_n": CONTEXT_N, "seed": SEED,
                        "cell_selected_on": "VAL only (ARM>=1.0)", "reoptimized_on_oos": False,
                        "circularity_caveat": "고정타깃이 FLOOR=30 라벨이라 FLOOR=30 학습이 유리 -- 9-9와 동일",
                        "holdout_touched": False, "live_code_changed": False},
              "variants": {}}

    target_n = {}
    log("")
    log(f"{'FLOOR':>6s} {'TRAIN라벨행':>10s} {'라벨률':>7s} | "
        f"{'호출 소진중앙':>12s} {'호출 먹을폭중앙':>14s} | {'정':>3s} {'뒤':>3s} {'차':>4s} "
        f"{'최고pess':>9s} {'승률':>6s} | {'고정타깃AUC':>10s}")
    for fl in FLOORS:
        lf = frame_for(fl)
        tr_ = lf.loc[(lf["split"] == "TRAIN") & lf["label"].notna()]
        rng = np.random.default_rng(SEED)
        ctx = tr_.iloc[np.sort(rng.choice(len(tr_), size=min(CONTEXT_N, len(tr_)), replace=False))]
        clf = TabPFNClassifier(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
        clf.fit(ctx[FEATURE_COLUMNS], ctx["label"].to_numpy())

        ent = {"train_n": int(len(tr_)), "label_rate": round(float(tr_["label"].mean()), 5)}
        for spn in ("VAL", "OOS"):
            s = lf.loc[lf["split"] == spn].copy()
            CH = 20000
            s["p"] = np.concatenate([clf.predict_proba(s[FEATURE_COLUMNS].iloc[k:k+CH])[:, 1]
                                     for k in range(0, len(s), CH)])
            if fl == 0:
                target_n[spn] = int((s["p"] >= BASE_THR).sum())
            sel = s.nlargest(min(target_n[spn], len(s)), "p")
            g = grid(build(sel))
            e2 = {"n_calls": int(len(sel)),
                  "call_consumed_median": round(float(np.nanmedian(sel["consumed"])) * 100, 1),
                  "call_cap_bp_median": round(float(np.nanmedian(sel["cap_bp"])), 2),
                  "grid": g}
            if spn == "VAL":
                m = s.merge(ref_val, on=["timestamp", "is_downside"], how="inner")
                e2["auc_fixed_target"] = round(float(roc_auc_score(m["ref_label"], m["p"])), 4) \
                    if m["ref_label"].nunique() == 2 else None
            sl_ = s.loc[s["label"].notna()]
            e2["auc_own_label"] = round(float(roc_auc_score(sl_["label"], sl_["p"])), 4) \
                if sl_["label"].nunique() == 2 else None
            ent[spn] = e2
        report["variants"][f"floor_{fl}"] = ent
        for spn in ("VAL", "OOS"):
            e2 = ent[spn]; g = e2["grid"]
            tagv = f"{fl:>4d}bp" if spn == "VAL" else "     "
            log(f"{tagv} {len(tr_) if spn=='VAL' else '':>10} "
                f"{ent['label_rate'] if spn=='VAL' else '':>7} | "
                f"{spn} {e2['call_consumed_median']:>7.1f}% {e2['call_cap_bp_median']:>12.2f}bp | "
                f"{g['fwd_pass']:>3d} {g['flip_pass']:>3d} {g['margin']:>+4d} "
                f"{g['best']['pess_bp']:>+8.2f}bp {g['best']['win_rate']*100:>5.1f}% | "
                f"{e2.get('auc_fixed_target', ''):>10}")

    log("")
    log("=== ⭐메커니즘 검증: 호출 소진율이 125%에서 내려갔는가 ===")
    for fl in FLOORS:
        v = report["variants"][f"floor_{fl}"]
        log(f"  FLOOR={fl:>2d}bp  VAL 호출소진 {v['VAL']['call_consumed_median']:>6.1f}%  "
            f"먹을폭 {v['VAL']['call_cap_bp_median']:>7.2f}bp   |   "
            f"OOS 호출소진 {v['OOS']['call_consumed_median']:>6.1f}%  "
            f"먹을폭 {v['OOS']['call_cap_bp_median']:>7.2f}bp")
    log("")
    log("=== 판정 (VAL/OOS 양쪽 정방향 우세 필요) ===")
    for fl in FLOORS:
        v = report["variants"][f"floor_{fl}"]
        ok = v["VAL"]["grid"]["margin"] > 0 and v["OOS"]["grid"]["margin"] > 0
        log(f"  {'✅' if ok else '  '}FLOOR={fl:>2d}bp  VAL 차{v['VAL']['grid']['margin']:>+4d}  "
            f"OOS 차{v['OOS']['grid']['margin']:>+4d}  "
            f"(OOS 최고pess {v['OOS']['grid']['best']['pess_bp']:+.2f}bp)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    report["runtime_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
