"""MFE/MAE 정답지 — "여기 들어가면 얼마 먹고 얼마 각오하나".

방향 라벨(롱/숏/현금)은 배리어가 일중 변동보다 커서 하루 내내 같은 답을 준다(순수일 92%).
MFE/MAE 는 봉마다 연속값이라 같은 날 안에서도 갈릴 수 있다 — 그게 사실이면 타점 정보가
거기 있고, TP/SL 이 상수가 아니라 모델 출력이 된다(진입·청산 공동 학습).

먼저 재는 것: 일중 변동이 전체 변동의 몇 %인가. 작으면 방향 라벨과 같은 벽이다.
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

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
HORIZONS = [(6, "30분"), (12, "1시간"), (24, "2시간"), (72, "6시간"), (288, "1일")]


def _fwd_mfe_mae(close, opn, slip, H):
    """봉 i 에서 롱 진입(open[i+1])했을 때 앞 H봉의 최대 순행/역행. 숏은 부호 대칭."""
    n = len(close)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    for i in range(n - 2):
        E = opn[i + 1] * (1 + slip)
        seg = close[i + 1:min(i + 1 + H, n)]
        if len(seg) < 2:
            continue
        r = (seg * (1 - slip) - E) / E
        mfe[i], mae[i] = float(r.max()), float(r.min())
    return mfe, mae


def main() -> int:
    _fee, slip = omega._load_fee_slip()
    frame = retest.load_frame_current("2026-01-01", "2026-08-30")
    close, opn = frame["close"].to_numpy(float), frame["open"].to_numpy(float)
    t = pd.to_datetime(frame["timestamp"])
    day = t.dt.date.to_numpy()
    cols = {"timestamp": frame["timestamp"], "close": close}
    print(f"[프레임] {len(frame):,}봉  왕복비용 {2*(_fee+slip)*100:.3f}%\n")
    print(f"{'지평':7s} {'MFE 중앙':>9s} {'|MAE| 중앙':>10s} {'R=MFE/|MAE| 중앙':>16s} "
          f"{'일중 변동 비중':>13s} {'일중 R 범위 중앙':>15s}")
    for H, nm in HORIZONS:
        mfe, mae = _fwd_mfe_mae(close, opn, slip, H)
        cols[f"mfe_{H}"], cols[f"mae_{H}"] = mfe, mae
        ok = np.isfinite(mfe) & np.isfinite(mae) & (mae < -1e-9)
        R = np.full(len(mfe), np.nan)
        R[ok] = mfe[ok] / np.abs(mae[ok])
        cols[f"R_{H}"] = R
        d = pd.DataFrame({"day": day, "mfe": mfe, "mae": mae, "R": R}).dropna()
        # 분산 분해: 일내 분산 / 전체 분산
        gm = d.groupby("day")
        within = float(gm["R"].transform(lambda v: v - v.mean()).var())
        total = float(d["R"].var())
        rng = gm["R"].apply(lambda v: v.quantile(.9) - v.quantile(.1)).median()
        print(f"{nm:7s} {np.nanmedian(mfe)*100:8.3f}% {np.nanmedian(np.abs(mae))*100:9.3f}% "
              f"{np.nanmedian(R):15.3f} {within/max(total,1e-12)*100:12.1f}% {rng:14.3f}")
    df = pd.DataFrame(cols)
    df.to_csv(OUT / "mfe_mae_answerkey.csv", index=False)

    H = 12
    m, a, R = df[f"mfe_{H}"], df[f"mae_{H}"], df[f"R_{H}"]
    cost = 2 * (_fee + slip)
    MK = 0.00552          # 실측 메이커 peg+peg 왕복 5.52bp
    MKL = 0.00779         # 진입 peg + 청산 taker 7.79bp
    print(f"\n[1시간 지평 상세]  테이커 {cost*100:.3f}% · 메이커 {MK*100:.3f}~{MKL*100:.3f}%")
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        print(f"  q{q:<5} MFE {m.quantile(q)*100:7.3f}%  |MAE| {abs(a.quantile(1-q))*100:7.3f}%  "
              f"R {R.quantile(q):6.3f}")
    for cn, cv in (("테이커 0.140%", cost), ("메이커 0.552%→0.055%", MK), ("메이커(패) 0.078%", MKL)):
        print(f"  MFE > {cn:20s} 인 봉 {float((m > cv).mean())*100:5.1f}%")
    print(f"  R > 2 인 봉 {float((R > 2).mean())*100:5.1f}%  ·  R > 3 {float((R > 3).mean())*100:5.1f}%")
    net = m - cost
    print(f"  비용 차감 후 순 MFE 중앙 {net.median()*100:+.4f}%  ·  양수 봉 {float((net>0).mean())*100:.1f}%")
    netm = m - MK
    print(f"  메이커 차감 후 순 MFE 중앙 {netm.median()*100:+.4f}%  ·  양수 봉 {float((netm>0).mean())*100:.1f}%")
    print(f"\n산출물: {OUT / 'mfe_mae_answerkey.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
