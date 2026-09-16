#!/usr/bin/env python3
"""F2 — **보정 후** 포지션 편향. 캐시된 확률만 쓰므로 재학습 없다.

F 에서 편향을 q=0.75 에서 쟀는데, 권고안이 「온도보정한 base」라면 **그 임계값에서** 다시
재야 한다. 통과율을 절반으로 줄이면 어느 쪽 다리가 먼저 잘리는지가 바로 편향 문제다.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
import train_eval_omega461_parent_zig075_longwindow_20260917 as E  # noqa: E402
import research_omega461_parent_quality_calibration_and_bias_20260917 as F  # noqa: E402

TARGET = 0.035


def main() -> int:
    df, base_cols = E.load()
    vm = (df.timestamp >= E.VAL[0]) & (df.timestamp <= E.VAL[1] + " 23:59:59")
    val = df[vm].reset_index(drop=True)
    vdays = val.timestamp.dt.floor("D").to_numpy()
    fwd1 = val["fwd_1h_bp"].to_numpy(np.float64)
    yv = pd.to_numeric(val["zigzag_action"]).to_numpy(np.int64)

    z = np.load(E.OUT / "stageF_probs.npz", allow_pickle=True)
    store = {k: dict(v.item()) for k, v in z.items()}
    out = []
    print(f"{'모델':<14}{'q*':>7}{'통과':>7}{'롱%':>7}{'롱bp':>8}{'숏bp':>8}{'총bp':>8}"
          f"{'모델−숏':>10}{'CI(모델−숏)':>22}")
    for key, s in store.items():
        D, Q = s["D"], s["Q"]
        if s["tailD"] is None:                       # 배포본 -- 이미 3.5%, 그대로
            T, qstar, Qc = 1.0, E.Q_THRESH, Q
        else:
            T, _ = F.temp_scale(s["tailQ"], s["taily"])
            tQc = s["tailQ"] ** (1.0 / T); tQc /= tQc.sum(1, keepdims=True)
            tda = s["tailD"].argmax(1)
            tqf = np.where(tda > 0, tQc[np.arange(len(tQc)), tda], tQc[:, 0])
            qstar = float(np.clip(np.quantile(tqf[tda != 0],
                                              1.0 - TARGET / max((tda != 0).mean(), 1e-9)), 0.34, 0.999))
            Qc = Q ** (1.0 / T); Qc /= Qc.sum(1, keepdims=True)
        _, side = F.gate(D, Qc, qstar)
        r = F.bias_report(key, side, fwd1, yv, vdays)
        r["T"], r["q_star"] = T, qstar
        out.append(r)
        e = r["excess_vs_short"]
        print(f"{key:<14}{qstar:>7.3f}{r['n']:>7,}{r['long_share']*100:>6.1f}%"
              f"{r['per_side_bp']['롱']:>8.2f}{r['per_side_bp']['숏']:>8.2f}{r['gross_bp']:>8.2f}"
              f"{e['bp']:>+10.2f}  [{e['ci95'][0]:+7.2f},{e['ci95'][1]:+7.2f}]")
    (E.OUT / "stageF2_bias_after_calib.json").write_text(json.dumps(out, indent=2, default=float))
    print()
    print("숏 > 롱 인 모델:", sum(1 for r in out if r['per_side_bp']['숏'] > r['per_side_bp']['롱']), "/", len(out))
    print("모델−숏 CI 가 0 배제:", sum(1 for r in out if r['excess_vs_short']['ci95'][0] > 0), "/", len(out))
    ls = [r['long_share'] for r in out if r['name'].startswith('base')]
    print(f"base 롱비중 시드폭 {min(ls):.3f}~{max(ls):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
