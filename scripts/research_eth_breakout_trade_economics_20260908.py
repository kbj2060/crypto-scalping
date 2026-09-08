#!/usr/bin/env python3
"""돌파/되돌림 신호를 **매매로 옮기면 얼마인가** -- 브라켓·커버리지 격자 (2026-09-08).

사용자: *"앵커돌파/되돌림 신호의 로직대로라면 트레이딩 전략을 어떻게 세우면 좋을까?"*

## 왜 계산해야 하는가
라벨은 ±0.25% 대칭 배리어 중 **먼저 닿는 쪽**이다. 그대로 매매하면 TP=SL=25bp 이고
왕복 비용 10bp(테이커) 를 얹으면 손익분기 정확도는
    p·25 − (1−p)·25 − 10 = 0  →  p = 35/50 = **70%**
이다. 실측 정확도는 전건 57.5%, 상위절반 63.6% -- **둘 다 70% 에 한참 못 미친다.**
그래서 "정확도가 귀무를 이긴다"와 "매매로 돈이 된다"는 전혀 다른 질문이고, 이 스크립트가
후자를 격자로 답한다. 지렛대는 세 개뿐이다: **비대칭 브라켓 · 커버리지 · 비용**.

## 규약
- 예측은 캐시된 워크포워드 5시드 평균(c1c3_preds_T0.75.npz, 라벨 P=0.25% 로 학습된 그 예측).
- 진입가 = 트리거 레벨(라벨 규약과 동일). 방향 = 모델 판정(돌파면 발현 방향, 되돌림이면 반대).
- 청산은 **1분봉 first-touch**, 같은 분에 양쪽이면 **비관적(SL 우선)**.
  1시간(12봉) 안에 어느 쪽도 안 닿으면 그 시점 종가로 시간청산.
- 비용은 왕복 bp 를 그대로 뺀다(테이커 10 / 메이커 7.8).
- 셔플 귀무를 같은 격자에 통과시킨다 -- 브라켓 기하학만으로 나오는 이익을 걷어내기 위해서다.

⚠️설계 단계 계산이다. 승격 주장이 아니다(CLAUDE.md fresh-forward 규약은 별도 검정을 요구한다).
"""
from __future__ import annotations
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "8")
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

MY = ROOT / "tmp/eth_breakout_atr_state_20260908_s1"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
WDAYS = {"VAL": 122, "OOS": 90, "HOLDOUT_SPENT": 122}
H, TM = 12, 0.75
TPS = (15, 20, 25, 30, 40, 50, 75)       # bp
SLS = (10, 15, 20, 25, 30, 40, 50)       # bp
COVS = (1.0, 0.5, 0.3, 0.2)
COST_TAKER, COST_MAKER = 10.0, 7.8


def main() -> int:
    d = pd.read_parquet(MY / "dataset_v2.parquet")
    d = d[(d.anchor == "first_fire") & (d.T_mult == TM)].sort_values("timestamp").reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    eth = B._load_kl(B.ETH_KL)
    ts5 = eth["timestamp"].to_numpy(); O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy(); hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float)

    sgn = np.where(d["dir_up"].to_numpy() > 0, 1.0, -1.0)      # 발현 방향
    bi = d["bar_idx"].to_numpy(); T = d["T_atr"].to_numpy(float)
    entry = O5[np.minimum(bi + 1, len(O5) - 1)] * (1 + sgn * T)
    s1 = np.searchsorted(ts1, d["timestamp"].to_numpy())
    bt = np.searchsorted(ts5, ts1[np.clip(s1, 0, len(ts1) - 1)], side="right") - 1
    okm = (s1 > 0) & (s1 + H * 5 < len(ts1)) & (bt + H < len(C5)) & (bt >= 1)
    sp = d["split"].to_numpy()

    z = np.load(MY / "c1c3_preds_T0.75.npz")
    pred = np.nanmean(z["obs"], axis=0)                        # 5시드 평균
    nulp = z["nul"]
    fin = np.isfinite(pred) & okm
    # 모델 판정 → 매매 방향. 돌파(p>0.5)면 발현 방향, 되돌림이면 반대.
    def side_of(p): return np.where(p > 0.5, sgn, -sgn)

    N = len(d); idx = np.flatnonzero(fin)
    span = np.arange(H * 5)
    J = s1[idx][:, None] + span[None, :]
    HI = hi1[np.clip(J, 0, len(hi1) - 1)]; LO = lo1[np.clip(J, 0, len(lo1) - 1)]
    x5 = np.minimum(bt + H, len(C5) - 1)
    CLO = C5[x5][idx]; ENT = entry[idx]

    def simulate(side, tp_bp, sl_bp):
        """1분 first-touch. 같은 분에 양쪽이면 비관적(SL). 미터치면 시간청산 종가."""
        up_is_tp = side > 0
        tp_px = ENT * (1 + side * tp_bp / 1e4)
        sl_px = ENT * (1 - side * sl_bp / 1e4)
        hit_tp = (HI >= tp_px[:, None]) if up_is_tp.all() else None
        # 롱/숏 혼재이므로 벡터로 분기
        tp_hit = np.where(side[:, None] > 0, HI >= tp_px[:, None], LO <= tp_px[:, None])
        sl_hit = np.where(side[:, None] > 0, LO <= sl_px[:, None], HI >= sl_px[:, None])
        big = 1 << 30
        a_tp = np.where(tp_hit.any(1), tp_hit.argmax(1), big)
        a_sl = np.where(sl_hit.any(1), sl_hit.argmax(1), big)
        won = a_tp < a_sl                     # 동시(같은 분)면 SL 우선 -- 비관적
        lost = a_sl <= a_tp
        none = (a_tp == big) & (a_sl == big)
        gross = np.where(none, (CLO - ENT) / ENT * 1e4 * side,
                         np.where(won, tp_bp, -sl_bp).astype(float))
        return gross, none

    rows = []
    for tag, P in (("관측", pred), *[(f"셔플{b}", nulp[b]) for b in range(8)]):
        pf = np.isfinite(P) & okm
        sd = side_of(P)[idx]
        conf = np.abs(P - 0.5)
        for tp in TPS:
            for sl in SLS:
                g, none = simulate(sd, tp, sl)
                for cv in COVS:
                    thr = 0.0 if cv >= 1.0 else float(np.nanquantile(conf[idx], 1 - cv))
                    m = (np.abs(P[idx] - 0.5) >= thr)
                    for w in WINS:
                        mw = m & (sp[idx] == w)
                        if mw.sum() < 30: continue
                        rows.append(dict(src=tag, tp=tp, sl=sl, cov=int(cv * 100), win=w,
                                         n=int(mw.sum()), per_day=mw.sum() / WDAYS[w],
                                         gross=float(g[mw].mean()),
                                         net_t=float(g[mw].mean() - COST_TAKER),
                                         net_m=float(g[mw].mean() - COST_MAKER),
                                         timeout=float(none[mw].mean())))
        print(f"  {tag} 완료", flush=True)
    R = pd.DataFrame(rows); R.to_csv(MY / "trade_economics.csv", index=False)

    OBS = R[R.src == "관측"]; NUL = R[R.src != "관측"]
    nul_mean = NUL.groupby(["tp", "sl", "cov", "win"])["net_t"].mean().rename("null_net_t")
    M = OBS.merge(nul_mean, on=["tp", "sl", "cov", "win"])
    M["excess"] = M["net_t"] - M["null_net_t"]
    # 세 창 동시 양수인 구성만
    piv = M.pivot_table(index=["tp", "sl", "cov"], columns="win",
                        values=["net_t", "excess", "per_day"])
    ok = piv[("net_t", "VAL")].notna()
    for w in WINS: ok &= (piv[("net_t", w)] > 0) & (piv[("excess", w)] > 0)
    print("\n" + "=" * 112)
    print(f"세 창 모두 **테이커 10bp 차감 후 양수 & 셔플 초과 양수**: {int(ok.sum())} / {len(piv)} 구성")
    print("=" * 112)
    if ok.sum():
        S = piv[ok].copy()
        S["minnet"] = S[[("net_t", w) for w in WINS]].min(axis=1)
        S = S.sort_values("minnet", ascending=False).head(15)
        print(f"{'TP':>4}{'SL':>4}{'커버':>6} | " + " ".join(f"{w[:4]:>18}" for w in WINS) + f"{'최소':>8}{'건/일':>7}")
        for (tp, sl, cv), r in S.iterrows():
            cells = " ".join(f"{r[('net_t', w)]:>7.2f}({r[('excess', w)]:+5.2f})" for w in WINS)
            print(f"{tp:>4}{sl:>4}{cv:>5}% | {cells}{r['minnet']:>8.2f}{r[('per_day','OOS')]:>7.1f}")
    else:
        print("  없음 -- 이 격자에서는 비용을 넘는 구성이 하나도 없다.")
        B2 = M.groupby(["tp", "sl", "cov"])["net_t"].min().sort_values(ascending=False).head(10)
        print("\n  (참고) 세 창 최소 net_t 상위 10:")
        for (tp, sl, cv), v in B2.items():
            print(f"     TP{tp} SL{sl} 커버{cv}% → 최소 {v:+.2f}bp")
    print(json.dumps({"done": True, "pass": int(ok.sum()), "total": len(piv)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
