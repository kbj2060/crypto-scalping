#!/usr/bin/env python3
"""**지정가(메이커) 진입**으로 wick 앵커 갭을 회수할 수 있는가 -- 측정된 근본원인 직접 공격.

## 왜

2026-09-02 진단: 라벨 기준점은 `low[i]`(그 봉 저가)인데 진입가는 `open[i+1]`이라, 모델 호출의
목표 소진율이 121~128%다. **진입 시점에 움직임이 이미 끝나 있다.**

시장가로는 이 갭을 못 줄인다. 그러나 **앵커 쪽에 지정가를 걸어두면** 가격이 되돌아올 때만
체결되므로 소진분의 일부를 회수한다. 대가는 **미체결**(가격이 안 돌아오면 거래 자체가 없음).

이 저장소에는 peg-maker 섀도우 실측이 있다([[eth_maker_fill_shadow_realized_cost_checkpoint_20260824]]):
peg 100%체결·평균 **3.11bp**, static 88.9%체결(1.98bp)+11.1%타임아웃(14.82bp)=블렌디드 3.40bp.

## 비용 처리 (⚠️수수료 우대 가정 금지 규칙 준수)

  · **기준선 10.0bp** -- 테이커 왕복. **승격 판단은 이 값으로만 한다.**
  · 참고 **8.11bp** -- 메이커 진입(3.11bp 실측) + 테이커 청산(5.0bp). 트레일링 스톱은
    구조상 테이커이므로 왕복 전체를 메이커로 가정하지 않는다.
  ⚠️0% 프로모션 수수료는 쓰지 않는다("수수료를 못 이기면 트레이딩이 아니다" -- 사용자 결정).

## 격자

  · 진입 오프셋: 0(시장가) / -0.10 / -0.25 / -0.50 x ATR  (롱 기준, 숏은 대칭)
  · 체결 대기창: 1 / 3 / 6 봉. 창 안에 가격이 지정가에 닿으면 체결, 아니면 **거래 없음**.
  · 체결가는 지정가 그대로(메이커).

체결률과 기대값을 같이 낸다 -- **체결률이 떨어지면 기회손실**이므로 기대값만 보면 안 된다.
`총 bp = 기대값 x 체결건수`도 같이 찍어 실제 수익 규모를 비교한다.

라벨/모델은 승격 게이트와 동일(E0_binary, Tier0 23, 셀 사전지정 (5.0,1.5,0.1)).
셀·임계값은 VAL에서 선정, OOS 1회.

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


_s1 = _load("s1_maker", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
_bt = _s1._bt
TIER0 = _s1.FEATURE_COLUMNS
FORWARD_BARS = _s1.FORWARD_BARS
SL_GRID, ARM_GRID, TRAIL_GRID = _bt.SL_GRID, _bt.ARM_GRID, _bt.TRAIL_GRID

LABEL_CELL = (5.0, 1.5, 0.1)
COST_TAKER, COST_MAKER_ENTRY = 10.0, 8.11
ARTIFACT_FREE_MIN = 1.0
CONTEXT_N, SEED = 18000, 20260829
OFFSETS = [0.0, -0.10, -0.25, -0.50]        # x ATR, 롱 기준(유리한 쪽으로 물러남)
FILL_WINDOWS = [1, 3, 6]
TAIL_FRACS = [0.005, 0.01, 0.02, 0.05]
CHUNK = 40000
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
OUT = ROOT / "data/research/eth_v_rebound_maker_entry_20260902/report.json"


def log(m): print(f"[maker] {m}", flush=True)


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
    long = long.loc[(long["pos"] >= 0) & (long["pos"] + FORWARD_BARS + 8 < nk)].reset_index(drop=True)

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
                                               H, L, C, sl0, arm0, tr0, True) * 1e4 - COST_TAKER
    long["y"] = (net > 0).astype(float)

    tr_set = long.loc[long["split"] == "TRAIN"]
    rng = np.random.default_rng(SEED)
    ctx = tr_set.iloc[np.sort(rng.choice(len(tr_set), size=min(CONTEXT_N, len(tr_set)), replace=False))]
    clf = TabPFNClassifier(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
    clf.fit(ctx[TIER0], ctx["y"].to_numpy())
    log(f"모델 학습 완료 (라벨률 {long['y'].mean():.4f})")

    def exec_pnl(s, offset, win, cell, cost, flip=False):
        """지정가 진입 시뮬레이션 -> 체결분만 트레일링. (pnl배열, 체결률) 반환."""
        i = s["pos"].to_numpy().astype(int)
        sgn = np.where(s["is_downside"].to_numpy() == 1, 1.0, -1.0)
        if flip:
            sgn = -sgn
        atr = s["atr"].to_numpy(dtype=float)
        limit = o[i + 1] + sgn * offset * atr      # 롱이면 아래로, 숏이면 위로 물러남
        if offset == 0.0:
            filled = np.ones(len(s), bool); fill_bar = np.zeros(len(s), int)
        else:
            filled = np.zeros(len(s), bool); fill_bar = np.zeros(len(s), int)
            for w in range(win):
                touch = np.where(sgn > 0, l[i + 1 + w] <= limit, h[i + 1 + w] >= limit)
                new = (~filled) & touch
                fill_bar[new] = w; filled |= new
        if filled.sum() < 30:
            return None, float(filled.mean())
        fi = i[filled] + 1 + fill_bar[filled]      # 체결 봉
        e = limit[filled]
        H = np.stack([h[j:j+FORWARD_BARS] for j in fi])
        L = np.stack([l[j:j+FORWARD_BARS] for j in fi])
        C = np.stack([c[j:j+FORWARD_BARS] for j in fi])
        sl, arm, trv = cell
        pnl = _bt.simulate_trailing_vec(e, atr[filled], sgn[filled], H, L, C,
                                        sl, arm, trv, True) * 1e4 - cost
        return pnl, float(filled.mean())

    def stat(p):
        w = p > 0
        return {"n": int(len(p)), "exp_bp": round(float(p.mean()), 3),
                "win_rate": round(float(w.mean()), 4), "total_bp": round(float(p.sum()), 1),
                "payoff": round(float(p[w].mean()/-p[~w].mean()), 3) if w.any() and (~w).any() else None}

    scored = {}
    for spn in ("VAL", "OOS"):
        s = long.loc[long["split"] == spn].copy()
        s["p"] = np.concatenate([clf.predict_proba(s[TIER0].iloc[k:k+20000])[:, 1]
                                 for k in range(0, len(s), 20000)])
        scored[spn] = s

    report = {"signal": "v_rebound_maker_entry_recovery", "asset": "ETHUSDT",
              "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
              "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
              "scope": {"cost_taker_bp": COST_TAKER, "cost_maker_entry_bp": COST_MAKER_ENTRY,
                        "cost_note": "승격 판단은 10.0bp(테이커)로만. 8.11bp는 메이커진입 실측 참고치",
                        "offsets_atr": OFFSETS, "fill_windows": FILL_WINDOWS,
                        "label_cell": list(LABEL_CELL), "holdout_touched": False,
                        "live_code_changed": False},
              "val_grid": {}, "oos": {}}

    # ---- VAL에서 (분위, 셀, 오프셋, 창) 선정 ----
    log("")
    log("=== VAL 격자 (기준 비용 10.0bp) ===")
    log(f"  {'분위':>6s} {'오프셋':>7s} {'창':>3s} {'체결률':>7s} {'n':>6s} "
        f"{'기대값':>9s} {'승률':>6s} {'총bp':>9s}")
    best = None
    for frac in TAIL_FRACS:
        k = max(30, int(round(len(scored["VAL"]) * frac)))
        sel = scored["VAL"].nlargest(k, "p")
        cut = float(sel["p"].min())
        for off in OFFSETS:
            for win in (FILL_WINDOWS if off != 0.0 else [1]):
                for sl in SL_GRID:
                    for arm in ARM_GRID:
                        if arm < ARTIFACT_FREE_MIN:
                            continue
                        for trv in TRAIL_GRID:
                            pnl, fr = exec_pnl(sel, off, win, (sl, arm, trv), COST_TAKER)
                            if pnl is None:
                                continue
                            tot = float(pnl.sum())
                            if best is None or tot > best["val_total"]:
                                best = {"frac": frac, "cut": cut, "offset": off, "win": win,
                                        "cell": (sl, arm, trv), "val_total": tot,
                                        "val_exp": float(pnl.mean()), "fill_rate": fr,
                                        "val_stat": stat(pnl)}
                # 오프셋/창별 대표값(최적셀)만 로그
                rep = None
                for sl in SL_GRID:
                    for arm in ARM_GRID:
                        if arm < ARTIFACT_FREE_MIN:
                            continue
                        for trv in TRAIL_GRID:
                            pnl, fr = exec_pnl(sel, off, win, (sl, arm, trv), COST_TAKER)
                            if pnl is not None and (rep is None or pnl.sum() > rep[0]):
                                rep = (float(pnl.sum()), stat(pnl), fr)
                if rep:
                    log(f"  {frac*100:>5.1f}% {off:>7.2f} {win:>3d} {rep[2]*100:>6.1f}% "
                        f"{rep[1]['n']:>6,} {rep[1]['exp_bp']:>+8.2f}bp {rep[1]['win_rate']*100:>5.1f}% "
                        f"{rep[0]:>+8.0f}bp")

    log("")
    log(f"[VAL 선정] 상위{best['frac']*100:g}%(p>={best['cut']:.4f})  오프셋 {best['offset']:+.2f}ATR  "
        f"창 {best['win']}봉  셀 {best['cell']}  체결률 {best['fill_rate']*100:.1f}%  "
        f"기대값 {best['val_exp']:+.2f}bp  총 {best['val_total']:+.0f}bp")
    report["val_selection"] = {**{k: v for k, v in best.items() if k != "cell"},
                               "cell": list(best["cell"])}

    # ---- OOS 1회 (고정 확률컷) ----
    log("")
    log("=== OOS 1회 (VAL 선정 그대로, 재최적화 없음) ===")
    osel = scored["OOS"].loc[scored["OOS"]["p"] >= best["cut"]]
    for cost, nm in ((COST_TAKER, "테이커 10.0bp (승격기준)"), (COST_MAKER_ENTRY, "메이커진입 8.11bp (참고)")):
        for off, win, tag in ((0.0, 1, "시장가 대조군"), (best["offset"], best["win"], "선정 오프셋")):
            pnl, fr = exec_pnl(osel, off, win, best["cell"], cost)
            fl, _ = exec_pnl(osel, off, win, best["cell"], cost, flip=True)
            if pnl is None:
                log(f"  [{nm}/{tag}] 체결 부족(체결률 {fr*100:.1f}%)"); continue
            s_ = stat(pnl)
            log(f"  [{nm}] {tag:12s} off{off:+.2f} 창{win}  체결 {fr*100:5.1f}%  n={s_['n']:>5,}  "
                f"기대값 {s_['exp_bp']:>+7.2f}bp  승률 {s_['win_rate']*100:4.1f}%  "
                f"손익비 {s_['payoff']}  총 {s_['total_bp']:>+8.0f}bp  "
                f"뒤집기 {fl.mean():+.2f}bp"
                f"{'  ✅' if s_['exp_bp'] > 0 and fl.mean() < s_['exp_bp'] else '  ❌'}")
            report["oos"][f"{nm}|{tag}"] = {**s_, "fill_rate": round(fr, 4),
                                            "flip_exp_bp": round(float(fl.mean()), 3)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    report["runtime_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
