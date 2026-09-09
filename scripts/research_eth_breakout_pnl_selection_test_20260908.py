#!/usr/bin/env python3
"""돌파/되돌림 매매 격자의 **선택 편향 검정** -- 격자최대 귀무 · PBO · DSR (2026-09-08).

## 문제
브라켓 격자 196칸(TP 7 × SL 7 × 커버 4)에서 10칸이 세 창 모두 양수였고 최선이 OOS +2.53bp 였다.
그런데 **196칸에서 최선을 고른 것 자체**가 검정에 들어가야 한다 -- 칸별 p 값은 이걸 못 잡는다.

## 세 검정
A **격자최대 셔플 귀무**: 셔플 복제마다 격자 전체를 다시 돌려 **그 복제의 최선칸**을 기록한다.
  관측 최선(+2.53)을 그 분포와 비교한다. 다중검정을 절차 자체에 흡수시키는 방식이다.
  셔플 예측은 캐시(c1c3_preds_T0.75.npz, 라벨 셔플 학습본 32개)를 그대로 쓴다.
B **PBO** (Bailey/López de Prado CSCV): 평가구간을 S=10 블록으로 쪼개 C(10,5)=252 분할마다
  IS 최선칸을 골라 OOS 순위를 본다. PBO = IS최선이 OOS 중앙값 아래로 떨어지는 비율.
  0.5 면 완전 과적합, 낮을수록 좋다.
C **DSR** (Deflated Sharpe): 시행 횟수 N=196 을 반영해 샤프를 할인한다. 일수익 계열로 계산.

⚠️거래가 겹치므로(1시간 보유 × 하루 6~8건) 샤프는 **일 단위 합산** 계열로 낸다.
"""
from __future__ import annotations
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "8")
import sys, json, itertools
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm, skew, kurtosis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

MY = ROOT / "tmp/eth_breakout_atr_state_20260908_s1"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H, TM, COST = 12, 0.75, 10.0
TPS = (15, 20, 25, 30, 40, 50, 75)
SLS = (10, 15, 20, 25, 30, 40, 50)
COVS = (1.0, 0.5, 0.3, 0.2)
SBLK = 10


def main() -> int:
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
    okm = (s1 > 0) & (s1 + H * 5 < len(ts1)) & (bt + H < len(C5)) & (bt >= 1)
    sp = d["split"].to_numpy()

    z = np.load(MY / "c1c3_preds_T0.75.npz")
    obs = np.nanmean(z["obs"], axis=0); nul = z["nul"]
    idx = np.flatnonzero(np.isfinite(obs) & okm)
    span = np.arange(H * 5); J = s1[idx][:, None] + span[None, :]
    HI = hi1[np.clip(J, 0, len(hi1) - 1)]; LO = lo1[np.clip(J, 0, len(lo1) - 1)]
    ENT = entry[idx]; CLO = C5[np.minimum(bt + H, len(C5) - 1)][idx]
    SGN = sgn[idx]; SPI = sp[idx]
    day = d["timestamp"].dt.floor("D").to_numpy()[idx]
    ev = SPI != "TRAIN"                      # 평가 3창
    big = 1 << 30

    def pnl(P, tp, sl):
        """모델 P 로 방향을 정하고 TP/SL 브라켓을 돌린 **건당 순손익(bp)**."""
        side = np.where(P[idx] > 0.5, SGN, -SGN)
        tp_px = ENT * (1 + side * tp / 1e4); sl_px = ENT * (1 - side * sl / 1e4)
        tph = np.where(side[:, None] > 0, HI >= tp_px[:, None], LO <= tp_px[:, None])
        slh = np.where(side[:, None] > 0, LO <= sl_px[:, None], HI >= sl_px[:, None])
        a = np.where(tph.any(1), tph.argmax(1), big); b = np.where(slh.any(1), slh.argmax(1), big)
        none = (a == big) & (b == big)
        g = np.where(none, (CLO - ENT) / ENT * 1e4 * side, np.where(a < b, tp, -sl).astype(float))
        return g - COST

    CFG = [(tp, sl, cv) for tp in TPS for sl in SLS for cv in COVS]
    print(f"격자 {len(CFG)}칸 · 평가 사건 {int(ev.sum()):,}\n", flush=True)

    def grid_stats(P):
        """칸별 (세 창 최소 net, 선택마스크). 커버리지는 확신도 상위 k -- 라벨 안 씀."""
        conf = np.abs(P[idx] - 0.5)
        out = {}
        for tp in TPS:
            for sl in SLS:
                g = pnl(P, tp, sl)
                for cv in COVS:
                    thr = -np.inf if cv >= 1.0 else float(np.nanquantile(conf[ev], 1 - cv))
                    m = ev & (conf >= thr)
                    per = [g[m & (SPI == w)].mean() if (m & (SPI == w)).sum() >= 30 else np.nan
                           for w in WINS]
                    out[(tp, sl, cv)] = (float(np.min(per)) if np.all(np.isfinite(per)) else -np.inf,
                                         m, g)
        return out

    # ---------- A. 격자최대 셔플 귀무 ----------
    G = grid_stats(obs)
    best_cfg = max(CFG, key=lambda c: G[c][0]); best_obs = G[best_cfg][0]
    print(f"관측 최선칸: TP{best_cfg[0]} SL{best_cfg[1]} 커버{int(best_cfg[2]*100)}% → 세 창 최소 {best_obs:+.2f}bp")
    maxes = []
    for b in range(nul.shape[0]):
        Gb = grid_stats(nul[b])
        maxes.append(max(Gb[c][0] for c in CFG))
    maxes = np.array(maxes)
    p_grid = ((maxes >= best_obs).sum() + 1) / (len(maxes) + 1)
    print(f"\nA. 격자최대 귀무 (B={len(maxes)}): 셔플 최선 평균 {maxes.mean():+.2f} · "
          f"최대 {maxes.max():+.2f} · **p = {p_grid:.3f}**")
    print(f"   → 셔플도 196칸을 뒤지면 평균 {maxes.mean():+.2f}bp 짜리 칸을 찾아낸다. "
          f"{'관측이 그 위' if p_grid < 0.05 else '⚠️관측이 분포 안'}")

    # ---------- B. PBO (CSCV) ----------
    m_best, g_best = G[best_cfg][1], G[best_cfg][2]
    ord_ev = np.argsort(day[ev], kind="stable")
    ev_i = np.flatnonzero(ev)[ord_ev]
    blocks = np.array_split(ev_i, SBLK)
    PN = np.full((len(CFG), len(idx)), np.nan)
    SEL = np.zeros((len(CFG), len(idx)), bool)
    for ci, c in enumerate(CFG):
        _, m, g = G[c]; PN[ci][m] = g[m]; SEL[ci] = m
    lam = []
    for pick in itertools.combinations(range(SBLK), SBLK // 2):
        isb = np.concatenate([blocks[i] for i in pick])
        osb = np.concatenate([blocks[i] for i in range(SBLK) if i not in pick])
        mu_is = np.array([np.nanmean(PN[ci][isb]) if np.isfinite(PN[ci][isb]).sum() >= 20 else -np.inf
                          for ci in range(len(CFG))])
        mu_os = np.array([np.nanmean(PN[ci][osb]) if np.isfinite(PN[ci][osb]).sum() >= 20 else np.nan
                          for ci in range(len(CFG))])
        bi = int(np.argmax(mu_is))
        fin = np.isfinite(mu_os)
        r = (mu_os[fin] < mu_os[bi]).sum() / max(fin.sum() - 1, 1)   # 상대순위 (1=최고)
        lam.append(np.log(r / max(1 - r, 1e-9)) if 0 < r < 1 else (10 if r >= 1 else -10))
    lam = np.array(lam); pbo = float((lam <= 0).mean())
    print(f"\nB. PBO (CSCV S={SBLK}, 분할 {len(lam)}개): **PBO = {pbo:.3f}** "
          f"({'양호' if pbo < 0.3 else '주의' if pbo < 0.5 else '과적합'}) · 0.5=완전 과적합")

    # ---------- C. DSR ----------
    dfr = pd.DataFrame({"day": day[m_best], "bp": g_best[m_best]}).groupby("day")["bp"].sum()
    r = dfr.to_numpy(); sr = r.mean() / r.std(ddof=1)
    N = len(CFG)
    srs = np.array([G[c][0] for c in CFG]); srs = srs[np.isfinite(srs)]
    var_sr = srs.std(ddof=1) / max(abs(np.mean([np.nanstd(PN[ci]) for ci in range(len(CFG))])), 1e-9)
    g_ = 0.5772156649
    sr0 = (srs.std(ddof=1) / max(r.std(ddof=1), 1e-9)) * (
        (1 - g_) * norm.ppf(1 - 1 / N) + g_ * norm.ppf(1 - 1 / (N * np.e)))
    s3, s4 = skew(r), kurtosis(r, fisher=False)
    dsr = norm.cdf((sr - sr0) * np.sqrt(len(r) - 1) /
                   np.sqrt(max(1 - s3 * sr + (s4 - 1) / 4 * sr ** 2, 1e-9)))
    print(f"\nC. DSR: 일수익 {len(r)}일 · 평균 {r.mean():+.2f}bp/일 · 일샤프 {sr:.4f} "
          f"(연 {sr*np.sqrt(365):.2f})")
    print(f"   시행 N={N} 반영 기준선 SR0={sr0:.4f} · 왜도 {s3:+.2f} 첨도 {s4:.2f} → **DSR = {dsr:.3f}**"
          f" ({'통과' if dsr > 0.95 else '미달'}, 기준 >0.95)")

    json.dump({"best": [best_cfg[0], best_cfg[1], best_cfg[2]], "best_min_bp": best_obs,
               "p_grid_max": p_grid, "null_max_mean": float(maxes.mean()),
               "null_max_max": float(maxes.max()), "pbo": pbo, "dsr": float(dsr),
               "daily_sharpe": float(sr), "days": int(len(r))},
              open(MY / "pnl_selection_test.json", "w"))
    print("\n" + "=" * 96)
    ok = (p_grid < 0.05) and (pbo < 0.5) and (dsr > 0.95)
    print(f"종합: 격자최대 p{p_grid:.3f} · PBO {pbo:.3f} · DSR {dsr:.3f} → "
          f"{'✅세 검정 통과' if ok else '❌미통과'}")
    print(json.dumps({"done": True, "pass": bool(ok)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
