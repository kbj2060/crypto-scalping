"""배포 Omega4.6.1 의 실효 자유 파라미터 3개 스윕 — min_tp · min_sl · exit_threshold.

배포 구성에서 청산은 전부 하드코딩 배리어가 한다(exit_head 0/41, 0/262, 라이브 0/7).
ATR 적응은 2026 구간 98.5% 봉에서 floor 로 붕괴해 실효값이 TP 7.5% / SL 4.0% 상수다.
이 셋은 `retest.COMPONENTS` dict 값이므로 스윕은 dict 변형 + 기존 리플레이 재호출뿐이다.

부모 예측은 재계산하지 않고 live_gap/preds/ 의 배포번들 산출물을 재사용한다.
"""
import argparse
import itertools
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

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as sidecar  # noqa: E402
from replay_omega4_6_1_greedy_router_20260706 import greedy_replay, prepare_component  # noqa: E402

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
PRED = OUT / "preds"
_ORIG_EXIT = sidecar._predict_exit_prob_one
_PROBS: list[float] = []


def _probe_exit(*a, **kw):
    p = _ORIG_EXIT(*a, **kw)
    _PROBS.append(float(p))
    return p


def _off_exit(*_a, **_kw):
    return 0.0                      # 헤드 비활성 — 추론 자체를 건너뛴다


def _metrics(ledger: pd.DataFrame) -> dict:
    if ledger.empty:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0, "tp": 0, "sl": 0, "head": 0}
    r = ledger["trade_return"].to_numpy(float)
    eq = np.cumprod(1 + r)
    return {"pnl": float((eq[-1] - 1) * 100), "mdd": float((eq / np.maximum.accumulate(eq) - 1).min() * 100),
            "trades": int(len(r)), "wr": float((r > 0).mean() * 100),
            "tp": int((ledger["reason"] == "take_profit").sum()),
            "sl": int((ledger["reason"] == "stop_loss").sum()),
            "head": int((ledger["reason"] == "exit_head").sum())}


def _halves(ledger: pd.DataFrame) -> tuple[float, float]:
    """전반(1-5월) / 후반(6-8월) 복리 — 엣지 붕괴 이후 생존을 따로 본다."""
    if ledger.empty:
        return 0.0, 0.0
    t = pd.to_datetime(ledger["entry_timestamp"], errors="coerce")
    out = []
    for m in (t < pd.Timestamp("2026-06-01"), t >= pd.Timestamp("2026-06-01")):
        r = ledger.loc[m, "trade_return"].to_numpy(float)
        out.append(float((np.prod(1 + r) - 1) * 100) if len(r) else 0.0)
    return out[0], out[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-08-30")
    ap.add_argument("--mode", default="grid", choices=["probe", "grid"])
    ap.add_argument("--min-tp", default="0.075")
    ap.add_argument("--min-sl", default="0.040")
    ap.add_argument("--exit-threshold", default="0.95")
    ap.add_argument("--atr-window", default="192", help="ATR 창(봉). 여러 개면 콤마")
    ap.add_argument("--tp-mult", default="", help="비우면 cfg 기본(12). ATR 적응을 켜려면 floor 도 낮춘다")
    ap.add_argument("--sl-mult", default="")
    ap.add_argument("--exit-head", default="off", choices=["on", "off"],
                    help="off = 헤드 추론 생략(임계 0.95 에서 실측 0건이므로 배포 동작과 동일)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    aws = [int(v) for v in args.atr_window.split(",")]
    tms = [float(v) for v in args.tp_mult.split(",")] if args.tp_mult else [None]
    sms = [float(v) for v in args.sl_mult.split(",")] if args.sl_mult else [None]
    tps = [float(v) for v in args.min_tp.split(",")]
    sls = [float(v) for v in args.min_sl.split(",")]
    ths = [float(v) for v in args.exit_threshold.split(",")]
    device = parent._device("cpu")
    fee, slip = omega._load_fee_slip()
    frame = retest.load_frame_current(args.start, args.end)
    print(f"[프레임] {args.start} ~ {args.end}  {len(frame):,}봉  fee={fee} slip={slip}", flush=True)

    if args.mode == "probe":
        sidecar._predict_exit_prob_one = _probe_exit
        comps = {n: prepare_component(frame, PRED / n / f"predictions_{c['q_tag']}.csv", c, device)
                 for n, c in retest.COMPONENTS.items()}
        t0 = time.time()
        _s, ledger = greedy_replay(frame, comps, fee=fee, slip=slip,
                                   cost_mult=retest.COST_MULT, device=device)
        p = np.asarray(_PROBS)
        print(f"\n[exit 헤드 확률 실측] 호출 {len(p):,}회  {time.time()-t0:.0f}s")
        for q in (0.5, 0.9, 0.99, 0.999, 1.0):
            print(f"  q{q:<6} {np.quantile(p, q):.6f}")
        print(f"  임계 0.95 이상 {int((p >= 0.95).sum()):,}회 ({(p >= 0.95).mean()*100:.4f}%)")
        print(f"  원장 사유 {dict(ledger['reason'].value_counts())}  {_metrics(ledger)}")
        np.save(OUT / f"exit_probs{args.tag}.npy", p)
        return 0

    if args.exit_head == "off":
        sidecar._predict_exit_prob_one = _off_exit
    rows = []
    for tp, sl, th, aw, tm, sm in itertools.product(tps, sls, ths, aws, tms, sms):
        for c in retest.COMPONENTS.values():
            c["min_tp"], c["min_sl"], c["exit_threshold"] = tp, sl, th
            c["atr_window"] = aw
            if tm is not None:
                c["tp_mult"], c["sl_mult"] = tm, sm
        t0 = time.time()
        comps = {n: prepare_component(frame, PRED / n / f"predictions_{c['q_tag']}.csv", c, device)
                 for n, c in retest.COMPONENTS.items()}
        _s, ledger = greedy_replay(frame, comps, fee=fee, slip=slip,
                                   cost_mult=retest.COST_MULT, device=device)
        m = _metrics(ledger)
        m["h1"], m["h2"] = _halves(ledger)
        d0 = comps["zig075"]["dec"]
        act = omega._active(d0)
        mtp = float(pd.to_numeric(d0["take_profit"])[act].mean()) if act.any() else tp
        msl = float(pd.to_numeric(d0["stop_loss"])[act].mean()) if act.any() else sl
        adapt = float((pd.to_numeric(d0["take_profit"])[act] > tp + 1e-9).mean() * 100) if act.any() else 0.0
        m.update({"min_tp": tp, "min_sl": sl, "exit_threshold": th, "rr": round(tp / sl, 2),
                  "atr_window": aw, "tp_mult": tm, "sl_mult": sm,
                  "mean_tp": round(mtp * 100, 3), "mean_sl": round(msl * 100, 3),
                  "adapt_pct": round(adapt, 1),
                  "sec": round(time.time() - t0, 1), "exit_head": args.exit_head})
        rows.append(m)
        print(f"  floor {tp:.3f}/{sl:.3f} ATR{aw} ×{tm if tm else '기본'} "
              f"실효평균 TP {m['mean_tp']:.2f}% SL {m['mean_sl']:.2f}% (적응 {m['adapt_pct']:.0f}%) → "
              f"PnL {m['pnl']:+8.2f}%  MDD {m['mdd']:+7.2f}%  {m['trades']:3d}건  WR {m['wr']:4.1f}%  "
              f"전반 {m['h1']:+7.2f}% 후반 {m['h2']:+7.2f}%", flush=True)

    df = pd.DataFrame(rows).sort_values("pnl", ascending=False)
    f = OUT / f"barrier_sweep{args.tag}.csv"
    df.to_csv(f, index=False)
    print(f"\n[상위 10셀]\n{df.head(10).to_string(index=False)}\n산출물: {f}")
    print("\n주의: 단일 리플레이 구간 · 단일 시드. Seed-Diversity(N≥5) 미충족 → 승격 근거 아님.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
