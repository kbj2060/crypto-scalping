"""배포 TabM 그대로 + 필터 두 개 — 저변동 배제 · 숏 편향 상한.

모델도 라벨도 안 건드린다. 배포 번들·사이드카·배리어·라우터 전부 그대로 두고 결정에
필터만 얹어 리플레이한다. exit 헤드는 실측 0/59,815 발동이라 끈 상태(동일 결과, 빠름).

편향 상한은 무작위 폐기가 아니라 **숏에만 quality 임계값을 더하는** 방식이다 — 인과적이고
모델이 자기 확신 낮은 숏부터 버린다.
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
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as sidecar  # noqa: E402
from core.event_label_engine import atr_volatility  # noqa: E402
from replay_omega4_6_1_greedy_router_20260706 import greedy_replay, prepare_component  # noqa: E402

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
PFX = "omega1_regime3_expertdq_"
ATR_CUTS = [0.0, 0.10, 0.30, 0.50]        # 저변동 하위 비율 배제
SHORT_EXTRA = [0.00, 0.05, 0.10, 0.20]    # 숏에만 더하는 quality 임계
sidecar._predict_exit_prob_one = lambda *a, **k: 0.0


def _metrics(lg, t):
    if lg.empty:
        return {"pnl": 0.0, "mdd": 0.0, "n": 0, "wr": 0.0, "h1": 0.0, "h2": 0.0, "long": 0.0}
    r = lg["trade_return"].to_numpy(float)
    eq = np.cumprod(1 + r)
    tt = pd.to_datetime(lg["entry_timestamp"], errors="coerce")
    h = [float((np.prod(1 + lg.loc[m, "trade_return"].to_numpy(float)) - 1) * 100) if m.any() else 0.0
         for m in (tt < pd.Timestamp("2026-06-01"), tt >= pd.Timestamp("2026-06-01"))]
    return {"pnl": float((eq[-1] - 1) * 100),
            "mdd": float((eq / np.maximum.accumulate(eq) - 1).min() * 100),
            "n": int(len(r)), "wr": float((r > 0).mean() * 100),
            "h1": h[0], "h2": h[1], "long": float((lg["side"] > 0).mean() * 100)}


def main() -> int:
    device = parent._device("cpu")
    fee, slip = omega._load_fee_slip()
    frame = retest.load_frame_current("2026-01-01", "2026-08-30")
    n = len(frame)
    t = pd.to_datetime(frame["timestamp"])
    atr = atr_volatility(frame["high"], frame["low"], frame["close"], window=96).to_numpy()

    comps0, preds = {}, {}
    for name, cfg in retest.COMPONENTS.items():
        pcsv = OUT / "preds" / name / f"predictions_{cfg['q_tag']}.csv"
        comps0[name] = prepare_component(frame, pcsv, cfg, device)
        preds[name] = pd.read_csv(pcsv)
        print(f"  [{name}] 활성 {int(omega._active(comps0[name]['dec']).sum()):,}봉 "
              f"· quality 임계 {cfg['quality_threshold']}", flush=True)

    rows = []
    for cut_q in ATR_CUTS:
        cut = float(np.nanquantile(atr[np.isfinite(atr)], cut_q)) if cut_q > 0 else -np.inf
        for extra in SHORT_EXTRA:
            comps = {}
            for name, cfg in retest.COMPONENTS.items():
                c = dict(comps0[name])
                d = c["dec"].copy()
                side = pd.to_numeric(d["side"], errors="raise").to_numpy(np.int64)
                q = pd.to_numeric(preds[name][PFX + "quality_for_action"], errors="coerce").to_numpy()
                drop = (~np.isfinite(atr)) | (atr < cut)
                if extra > 0:
                    drop |= (side < 0) & (q < float(cfg["quality_threshold"]) + extra)
                d.loc[drop, "action"] = omega.ACTION_CASH
                d.loc[drop, "side"] = 0
                d.loc[drop, "notional_exposure"] = 0.0
                c["dec"] = d
                comps[name] = c
            _s, lg = greedy_replay(frame, comps, fee=fee, slip=slip,
                                   cost_mult=retest.COST_MULT, device=device)
            m = _metrics(lg, t)
            m.update({"atr_cut": cut_q, "short_extra": extra,
                      "active": int(sum(omega._active(c["dec"]).sum() for c in comps.values()))})
            rows.append(m)
            tag = "배포 그대로" if (cut_q == 0 and extra == 0) else ""
            print(f"  ATR컷 {cut_q:4.0%} 숏임계+{extra:.2f}  활성 {m['active']:6,d}  "
                  f"PnL {m['pnl']:+8.2f}%  MDD {m['mdd']:+7.2f}%  {m['n']:3d}건  WR {m['wr']:4.1f}%  "
                  f"롱 {m['long']:3.0f}%  전반 {m['h1']:+8.2f}%  후반 {m['h2']:+8.2f}%  {tag}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "deployed_filters.csv", index=False)
    b = df[(df.atr_cut == 0) & (df.short_extra == 0)].iloc[0]
    print(f"\n[대조군 배포 그대로] PnL {b.pnl:+.2f}%  MDD {b.mdd:+.2f}%  {int(b.n)}건  "
          f"전반 {b.h1:+.2f}%  후반 {b.h2:+.2f}%")
    print(f"\n[전 구간 PnL 상위 5]\n"
          f"{df.nlargest(5,'pnl')[['atr_cut','short_extra','pnl','mdd','n','wr','long','h1','h2']].to_string(index=False, float_format=lambda v: f'{v:.2f}')}")
    print(f"\n[후반 방어 상위 5]\n"
          f"{df.nlargest(5,'h2')[['atr_cut','short_extra','pnl','mdd','n','wr','long','h1','h2']].to_string(index=False, float_format=lambda v: f'{v:.2f}')}")
    print("\n필터는 모델을 안 건드린다 — 개선이 있으면 즉시 적용 가능하다. "
          "단일 구간이라 승격 근거는 아니고, 후반 방어가 전반 손실을 얼마에 사는지가 판단 기준이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
