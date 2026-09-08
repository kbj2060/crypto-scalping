#!/usr/bin/env python3
"""확신 붕괴가 **레짐 때문인가** (2026-09-08).

관측: 2026-09-05 부터 나흘 연속 사전등록 게이트 통과 0건. 확신 `강` 등급이 백테스트
      4.7건/일 → 라이브 1.2건/일 로 줄었고 확신 중앙값이 백테스트의 65% 수준이다.
원인 후보 셋(레짐 / 피쳐 분포 이동 / 모델 열화) 중 **레짐**을 먼저 배제하거나 확정한다.

가설: 지금 ATR 이 학습 구간의 59% 다(0.156% vs 0.265%). 변동성이 낮으면 ±0.25% 배리어가
      상대적으로 멀어져 결과가 더 무작위에 가까워지고, 모델이 0.5 근처로 수렴하는 게 자연스럽다.

검정: 백테스트 평가창을 **ATR 분위로 쪼개** 각 구간의 확신 분포·등급 비율·정확도를 본다.
      라이브와 같은 저ATR 구간에서도 확신이 낮으면 → **레짐이 설명한다**(모델 이상 아님).
      저ATR 구간에서도 확신이 정상이면 → 피쳐 이동 또는 열화다.

⚠️정확도는 셔플 귀무와 함께 읽는다 -- 구간마다 클래스 균형이 다르다.
"""
from __future__ import annotations
import os, sys, json
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "8")
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

MY = ROOT / "tmp/eth_breakout_atr_state_20260908_s1"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
H, TM, P, NB = 12, 0.75, 0.0025, 32
CHUNK = 4000


def main() -> int:
    meta = json.loads((ROOT / "data/live/breakout_reversal_shadow_artifact/meta.json").read_text())
    tiers = meta["confidence_tiers"]
    d = pd.read_parquet(MY / "dataset_v2.parquet")
    d = d[(d.anchor == "first_fire") & (d.T_mult == TM)].sort_values("timestamp").reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    eth = B._load_kl(B.ETH_KL)
    ts5 = eth["timestamp"].to_numpy(); O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy(); hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float)
    sgn = np.where(d["dir_up"].to_numpy() > 0, 1.0, -1.0)
    ba = d["bar_idx"].to_numpy(); T = d["T_atr"].to_numpy(float)
    entry = O5[np.minimum(ba + 1, len(O5) - 1)] * (1 + sgn * T)
    s1 = np.searchsorted(ts1, d["timestamp"].to_numpy())
    bt = np.searchsorted(ts5, ts1[np.clip(s1, 0, len(ts1) - 1)], side="right") - 1
    ok = (s1 > 0) & (s1 + H * 5 < len(ts1)) & (bt + H < len(C5)) & (bt >= 1)
    sp = d["split"].to_numpy()
    z = np.load(MY / "c1c3_preds_T0.75.npz"); obs = np.nanmean(z["obs"], axis=0); nul = z["nul"]
    idx = np.flatnonzero(np.isfinite(obs) & ok)
    big = 1 << 30
    span = np.arange(H * 5); J = s1[idx][:, None] + span[None, :]
    HI = hi1[np.clip(J, 0, len(hi1) - 1)]; LO = lo1[np.clip(J, 0, len(lo1) - 1)]
    ENT = entry[idx]; CLO = C5[np.minimum(bt + H, len(C5) - 1)][idx]; SGN = sgn[idx]
    u = HI >= (ENT * (1 + P))[:, None]; dn = LO <= (ENT * (1 - P))[:, None]
    a = np.where(u.any(1), u.argmax(1), big); b = np.where(dn.any(1), dn.argmax(1), big)
    y = np.where(np.where(SGN > 0, a < b, b < a), 1,
                 np.where(np.where(SGN > 0, b < a, a < b), 0,
                          ((CLO - ENT) / ENT * SGN > 0).astype(int)))
    conf = np.abs(obs[idx] - 0.5); acc = (obs[idx] > 0.5).astype(int) == y
    atr = d["atr_at_anchor"].to_numpy()[idx]; SPI = sp[idx]; ev = SPI != "TRAIN"
    day = d["timestamp"].dt.floor("D").to_numpy()[idx]

    LIVE_ATR = 0.00156
    print(f"라이브 ATR 중앙값 {LIVE_ATR*100:.3f}% · 백테스트 전체 중앙 {np.median(atr)*100:.3f}%")
    print(f"평가창 사건 {int(ev.sum()):,}\n")
    qs = np.nanquantile(atr[ev], [0.2, 0.4, 0.6, 0.8])
    edges = [0, *qs, 1]
    print("=" * 108)
    print(f"{'ATR 구간':>18}{'n':>7}{'건/일':>7}{'확신중앙':>10}{'확신90%':>9}"
          f"{'강':>7}{'중':>7}{'약':>7}{'미약':>7}{'정확도':>8}{'귀무':>8}{'초과':>8}")
    print("=" * 108)
    days = 334
    for i in range(5):
        lo_, hi_ = edges[i], edges[i + 1]
        m = ev & (atr >= lo_) & (atr < hi_)
        if m.sum() < 50: continue
        c = conf[m]
        nb = np.array([( (nul[k][idx][m] > 0.5).astype(int) == y[m]).mean() for k in range(nul.shape[0])])
        rng = f"{lo_*100:.3f}~{hi_*100:.3f}%" if hi_ < 1 else f"≥{lo_*100:.3f}%"
        mark = " ⭐라이브" if lo_ <= LIVE_ATR < hi_ else ""
        print(f"{rng:>18}{m.sum():>7}{m.sum()/days:>7.1f}{np.median(c):>10.4f}"
              f"{np.percentile(c,90):>9.4f}"
              f"{(c>=tiers['high']).sum()/days:>7.1f}"
              f"{((c>=tiers['mid'])&(c<tiers['high'])).sum()/days:>7.1f}"
              f"{((c>=tiers['low'])&(c<tiers['mid'])).sum()/days:>7.1f}"
              f"{(c<tiers['low']).sum()/days:>7.1f}"
              f"{acc[m].mean()*100:>7.1f}%{nb.mean()*100:>7.1f}%{(acc[m].mean()-nb.mean())*100:>+7.1f}{mark}")
    print("\n" + "=" * 108)
    # 라이브와 같은 ATR 대역만 따로
    band = (atr >= 0.0013) & (atr <= 0.0019)
    m = ev & band
    c = conf[m]
    nb = np.array([((nul[k][idx][m] > 0.5).astype(int) == y[m]).mean() for k in range(nul.shape[0])])
    print(f"⭐라이브와 같은 ATR 대역(0.130~0.190%)만: n={m.sum():,} ({m.sum()/days:.1f}건/일)")
    print(f"   확신 중앙 {np.median(c):.4f} · 90분위 {np.percentile(c,90):.4f}")
    print(f"   강 {(c>=tiers['high']).sum()/days:.1f}건/일 · 중 {((c>=tiers['mid'])&(c<tiers['high'])).sum()/days:.1f} "
          f"· 약 {((c>=tiers['low'])&(c<tiers['mid'])).sum()/days:.1f} · 미약 {(c<tiers['low']).sum()/days:.1f}")
    print(f"   정확도 {acc[m].mean()*100:.1f}% · 귀무 {nb.mean()*100:.1f}% · 초과 {(acc[m].mean()-nb.mean())*100:+.1f}pp")
    # 이 대역의 '강 비율'을 라이브 트리거 수에 곱하면 기대 강 건수
    rate = (c >= tiers["high"]).mean()
    print(f"\n   → 이 대역의 강 비율 {rate*100:.1f}% × 라이브 트리거 23.4건/일 = **기대 {rate*23.4:.1f}건/일**")
    print(f"      라이브 실측 강 = 1.2건/일")
    print("\n" + "=" * 108)
    print("ATR 시계열 (월별 중앙값, 평가창)")
    dd = pd.DataFrame({"m": pd.to_datetime(day[ev]).to_period("M").astype(str), "atr": atr[ev]})
    for k, v in dd.groupby("m")["atr"].median().items():
        bar = "█" * int(v * 100 * 40)
        print(f"   {k}  {v*100:.3f}%  {bar}")
    print(f"   2026-09 (라이브)  {LIVE_ATR*100:.3f}%  {'█'*int(LIVE_ATR*100*40)}")
    print(json.dumps({"done": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
