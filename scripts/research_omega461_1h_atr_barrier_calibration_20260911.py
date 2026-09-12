"""평균 1시간 보유를 만드는 ATR 배수 실측 + 그 스케일의 비용 장벽.

핵심 항등식 — 필요 실력은 라벨이 아니라 비용/배리어 비율이 정한다:
    손익분기 승률 BE   = (SL + c) / (TP + SL)
    무작위 승률   base ≈  SL      / (TP + SL)
    필요 실력    gap  =  c / (TP + SL)          ← RR 이 소거된다
어떤 라벨 로직도 gap 을 낮추지 못한다. 낮추는 레버는 비용 c, 배리어 폭, 실제 실력뿐이다.

수직 배리어는 느슨하게(1152봉=4일) 둔다 — 하드 컷을 걸면 먼 쪽(TP) 도달이 잘려
결착 표본이 SL 로 치우친다(직전 실험에서 타임아웃 29%, 기저 승률 27~29% 로 왜곡됐다).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
from core.event_label_engine import (TripleBarrierConfig, apply_triple_barrier,  # noqa: E402
                                     atr_volatility, return_dispersion_volatility)

MAKER_COST = 0.0006      # 메이커 왕복 가정(수수료 0.02%x2 + 슬리피지 0.01%x2)


def main() -> int:
    fee, slip = omega._load_fee_slip()
    cost = 2 * (fee + slip)
    frame = retest.load_frame_current("2026-01-01", "2026-08-30")
    n = len(frame)
    print(f"[프레임] {n:,}봉 · 테이커 왕복 {cost*100:.3f}% · 메이커 가정 {MAKER_COST*100:.3f}%")

    vols = {"ATR(12)": atr_volatility(frame["high"], frame["low"], frame["close"], window=12),
            "ATR(96)": atr_volatility(frame["high"], frame["low"], frame["close"], window=96),
            "ret_disp(12,288)": return_dispersion_volatility(frame["close"], window=12, lookback=288)}
    for nm, v in vols.items():
        print(f"  {nm:18s} 중앙 {v.median()*100:.4f}%")

    ev = np.arange(0, n - 1200, 5)
    print(f"\n{'변동성':18s} {'배수':>6s} {'RR':>5s} {'TP%':>7s} {'SL%':>7s} {'보유평균':>8s} "
          f"{'보유중앙':>8s} {'타임아웃':>7s} {'손익분기':>8s} {'필요(테이커)':>11s} {'필요(메이커)':>11s}")
    rows = []
    for nm, v in vols.items():
        for mult in (0.5, 0.75, 1.0, 1.5, 2.0, 3.5, 5.0, 8.0, 14.0):
            for rr in (1.0, 1.88):
                cfg = TripleBarrierConfig(pt_mult=mult * rr, sl_mult=mult, max_hold=1152)
                tb = apply_triple_barrier(frame, ev, v, cfg)
                if tb.empty:
                    continue
                tp = mult * rr * v.median() * 100
                sl = mult * v.median() * 100
                hb = tb["bars_held"]
                to = float((tb["touch_type"] == "timeout").mean() * 100)
                be = (sl + cost * 100) / (tp + sl) * 100
                gt = cost * 100 / (tp + sl) * 100
                gm = MAKER_COST * 100 / (tp + sl) * 100
                rows.append({"vol": nm, "mult": mult, "rr": rr, "tp": tp, "sl": sl,
                             "hold_mean": float(hb.mean()), "hold_med": float(hb.median()),
                             "timeout": to, "be_wr": be, "gap_taker": gt, "gap_maker": gm})
                mark = " ←평균1시간" if 8 <= float(hb.mean()) <= 20 else ""
                print(f"{nm:18s} {mult:6.2f} {rr:5.2f} {tp:6.3f}% {sl:6.3f}% {hb.mean():7.1f}봉 "
                      f"{hb.median():7.1f}봉 {to:6.1f}% {be:7.1f}% {gt:10.2f}pp {gm:10.2f}pp{mark}",
                      flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap/atr_1h_calibration.csv", index=False)
    ok = df[(df.hold_mean >= 8) & (df.hold_mean <= 20)]
    print(f"\n[보유 **평균** 8~20봉(≈1시간) 셀 {len(ok)}개]")
    if len(ok):
        print(ok.sort_values("gap_taker").head(10).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print(f"\n최저 필요실력  테이커 {ok.gap_taker.min():.2f}pp · 메이커 {ok.gap_maker.min():.2f}pp")
    print("현재 2~3일 설계(7.5+4.0)는 테이커 1.22pp · 오늘 관측된 최대 신호 lift 는 +3.64pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
