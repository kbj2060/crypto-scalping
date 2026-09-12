"""'좋은 타점' 조건 스크린 — 어떤 사건 조건이 배리어 승률을 손익분기 위로 미는가.

라벨을 결과로만 정의하면 못 배운다(트리플배리어 quality 0.3663, MFE-|MAE| 요구정확도 0.58
> 달성 0.5633). 라벨은 **사건 조건 × 방향 타깃** 이어야 하고, 그러려면 먼저 조건 자체가
승률을 옮기는지 봐야 한다.

경계: 조건은 봉 i 에서 확정되고 라벨 탐색은 i+1 부터다(진입도 open[i+1]).
Event-Label Boundary Contract 를 만족한다 -- 조건 봉과 라벨 탐색 구간이 겹치지 않는다.
귀무는 순환이동이다(조건의 시간 군집을 보존한다). 측면별로 따로 잰다.
"""
import argparse
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
H = 4032
B_NULL, NULL_SEED = 400, 615372041
EVENT_RE = (r"sweep|reclaim|liq|wick|climax|spike|divergence|oi_up|oi_down|extreme|"
            r"breakout|squeeze|exhaust|trap|vacuum|contraction")


def _outcomes(close, opn, tp, sl, slip, horizon=None):
    """봉 i 에서 각 측면으로 진입했을 때 배리어 승패. 1=승 0=패 -1=시간청산.
    tp/sl 은 스칼라(고정 배리어) 또는 봉별 배열(변동성 적응) 둘 다 받는다."""
    n = len(close)
    tpa = np.full(n, float(tp)) if np.isscalar(tp) else np.asarray(tp, dtype=np.float64)
    sla = np.full(n, float(sl)) if np.isscalar(sl) else np.asarray(sl, dtype=np.float64)
    out = np.full((n, 2), -1, dtype=np.int8)
    for col, s in ((0, 1), (1, -1)):
        for i in range(n - 2):
            tp, sl = tpa[i], sla[i]
            if not (np.isfinite(tp) and np.isfinite(sl) and tp > 0 and sl > 0):
                continue
            E = opn[i + 1] * (1 + slip if s > 0 else 1 - slip)
            seg = close[i + 1:min(i + 1 + (H if horizon is None else horizon), n)]
            if s > 0:
                w, l = seg >= E * (1 + tp) / (1 - slip), seg <= E * (1 - sl) / (1 - slip)
            else:
                w, l = seg <= E * (1 - tp) / (1 + slip), seg >= E * (1 + sl) / (1 + slip)
            iw = int(w.argmax()) if w.any() else 1 << 30
            il = int(l.argmax()) if l.any() else 1 << 30
            if iw == il:
                continue
            out[i, col] = 1 if iw < il else 0
    return out


def _conditions(frame):
    """사건 조건 후보. 이진은 !=0, 연속은 양쪽 극단 10%."""
    import re
    cols = [c for c in frame.columns if re.search(EVENT_RE, c, re.I)]
    out = {}
    for c in cols:
        s = pd.to_numeric(frame[c], errors="coerce")
        if s.isna().all():
            continue
        u = s.dropna().unique()
        if len(u) <= 3:
            m = (s != 0) & s.notna()
            if 200 <= int(m.sum()) <= len(s) // 3:
                out[f"{c}!=0"] = m.to_numpy()
        else:
            q10, q90 = s.quantile(0.10), s.quantile(0.90)
            for nm, m in ((f"{c}≥q90", s >= q90), (f"{c}≤q10", s <= q10)):
                m = m & s.notna()
                if int(m.sum()) >= 200:
                    out[nm] = m.to_numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-08-30")
    ap.add_argument("--tp", type=float, default=0.075)
    ap.add_argument("--sl", type=float, default=0.040)
    args = ap.parse_args()
    fee, slip = omega._load_fee_slip()
    frame = retest.load_frame_current(args.start, args.end)
    close, opn = frame["close"].to_numpy(float), frame["open"].to_numpy(float)
    need = args.sl / (args.tp + args.sl) * 100
    print(f"[프레임] {len(frame):,}봉  TP {args.tp*100:.1f}% SL {args.sl*100:.1f}%  "
          f"손익분기 승률 {need:.1f}%", flush=True)

    oc = _outcomes(close, opn, args.tp, args.sl, slip)
    n = len(frame)
    base = {}
    for col, nm in ((0, "롱"), (1, "숏")):
        d = oc[:, col]
        base[nm] = float((d == 1).sum() / max((d >= 0).sum(), 1) * 100)
        print(f"  기저 {nm} 승률 {base[nm]:5.2f}%  (결착 {int((d>=0).sum()):,} · "
              f"시간청산 {int((d<0).sum()):,})", flush=True)

    conds = _conditions(frame)
    print(f"\n조건 후보 {len(conds)}개 · 순환이동 귀무 B={B_NULL}\n", flush=True)
    rng = np.random.default_rng(NULL_SEED)
    shifts = rng.integers(1, n, size=B_NULL)
    rows = []
    for cname, mask in conds.items():
        idx = np.flatnonzero(mask)
        for col, nm in ((0, "롱"), (1, "숏")):
            d = oc[:, col]
            sub = d[idx]
            k = sub >= 0
            if k.sum() < 100:
                continue
            wr = float((sub[k] == 1).mean() * 100)
            null = []
            for s in shifts:
                sd = d[(idx + s) % n]
                kk = sd >= 0
                if kk.sum():
                    null.append(float((sd[kk] == 1).mean() * 100))
            null = np.asarray(null)
            p = float((null >= wr).mean())
            rows.append({"조건": cname, "측면": nm, "n": int(k.sum()), "승률": wr,
                         "기저": base[nm], "lift": wr - base[nm],
                         "귀무중앙": float(np.median(null)),
                         "귀무q95": float(np.quantile(null, 0.95)), "p": p,
                         "손익분기초과": wr - need})
    df = pd.DataFrame(rows).sort_values("lift", ascending=False)
    f = OUT / "entry_condition_lift.csv"
    df.to_csv(f, index=False)
    show = df[(df.p <= 0.05) & (df.lift > 0)]
    print(f"[귀무 통과 조건 {len(show)}개 / 전체 {len(df)}셀]")
    if len(show):
        print(show.head(25).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    else:
        print("  없음 — 어떤 사건 조건도 승률을 유의하게 옮기지 못했다.")
    print(f"\n상위 lift 10셀(유의 무관):\n{df.head(10).to_string(index=False, float_format=lambda v: f'{v:.2f}')}")
    print(f"\n산출물: {f}")
    print(f"주의: 다중검정 {len(df)}셀 — p<0.05 는 우연히도 {len(df)*0.05:.1f}개 나온다. "
          "통과 개수가 그 기대치를 넘는지부터 본다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
