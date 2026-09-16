#!/usr/bin/env python3
"""H — **편향을 제거하고 다시 학습한다.** 학습 단계 + 결정 단계 둘 다.

## 편향의 출처 (학습 단계)
기존 가중치는
    w = compute_sample_weight("balanced", y_전체) * route_prob[:, expert]
로 **클래스 균형을 전체 라벨에서 잡고 라우팅 확률을 나중에 곱한다.** 그래서 bull 전문가가
실제로 보는 분포는 LONG 이 과대표집돼 있다(bull 레짐과 LONG 라벨이 상관). 전문가별 측면
사전확률이 여기서 생긴다.

**고침(NEW)**: 라우팅 가중을 **먼저** 적용하고 그 분포 위에서 클래스 균형을 잡는다.
    s_c = Σ_{i: y_i=c} route_w_i ;  w_i = route_w_i * (Σ_c s_c) / (3 * s_{y_i})
=> 전문가 안에서 세 클래스의 **총 가중치가 같아진다**. 라우팅은 그대로 살아 있다.

## 분산 (시드)
롱비중이 시드마다 16~71% 로 흔들렸다. **무작위 5시드 앙상블**(확률 평균)로 죽인다.
고정간격 증분이 아닌 진짜 무작위 추출 -- Seed-Diversity 계약.

## 결정 단계
 · q* : TRAIN 꼬리에서 통과율 3.5% 가 되게 잡는다(VAL 미사용)
 · **측면 대칭 게이트**: 롱/숏 임계값을 TRAIN 꼬리에서 **각각** 잡아 양쪽이 1.75% 씩
   통과하게 한다 -> 모델이 한쪽 다리를 계통적으로 더 쏠 수 없다.

## 평가
평균만 보면 착시가 난다(배포본은 상위 3일이 총이익의 67%). **중앙 건당bp · 양수일 비중 ·
최고1일 제거**를 같이 내고, **모델−롱과 모델−숏을 항상 같이** 낸다.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
import train_eval_omega461_parent_zig075_longwindow_20260917 as E  # noqa: E402
import research_omega461_parent_quality_calibration_and_bias_20260917 as F  # noqa: E402
import research_omega461_parent_pnl_concentration_20260917 as G  # noqa: E402

SEEDS5 = [613042, 27851, 904377, 155690, 488213]   # 무작위 추출, 고정간격 아님
TARGET = 0.035
OUTJ = E.OUT / "stageH_debias.json"


def log(*a, **k): print(*a, flush=True)


def w_old(y, route_w):
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight("balanced", y=y).astype(np.float32) * route_w


def w_new(y, route_w):
    """라우팅 가중 **후에** 클래스 균형. 전문가 안에서 세 클래스 총 가중치를 같게."""
    s = np.array([route_w[y == c].sum() for c in range(3)], dtype=np.float64)
    assert (s > 0).all(), f"전문가가 못 보는 클래스가 있다: {s}"
    scale = s.sum() / (3.0 * s)
    return (route_w * scale[y]).astype(np.float32)


def side_from(D, Q, q_long, q_short):
    da = D.argmax(1)
    qf = np.where(da > 0, Q[np.arange(len(Q)), da], Q[:, 0])
    thr = np.where(da == 1, q_long, q_short)
    final = np.where((da != 0) & (qf >= thr), da, 0)
    return np.where(final == 1, 1.0, np.where(final == 2, -1.0, 0.0))


def thresholds(tD, tQ, *, symmetric):
    """TRAIN 꼬리에서 임계값. symmetric 이면 롱/숏 각각 TARGET/2 통과하도록 따로."""
    da = tD.argmax(1)
    qf = np.where(da > 0, tQ[np.arange(len(tQ)), da], tQ[:, 0])
    def pick(mask, target):
        v = qf[mask]
        if len(v) < 50:
            return 0.999
        rate = target / max(mask.mean(), 1e-9)
        return float(np.clip(np.quantile(v, 1.0 - min(rate, 1.0)), 0.34, 0.999))
    if symmetric:
        return pick(da == 1, TARGET / 2), pick(da == 2, TARGET / 2)
    q = pick(da != 0, TARGET)
    return q, q


def evaluate(name, side, fwd, yv, days):
    r = F.bias_report(name, side, fwd, yv, days)
    r.update({k: v for k, v in G.day_stats(side, fwd, days).items() if not k.startswith("_")})
    return r


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} · seeds={SEEDS5}")
    df, base_cols = E.load()
    vm = (df.timestamp >= E.VAL[0]) & (df.timestamp <= E.VAL[1] + " 23:59:59")
    val = df[vm].reset_index(drop=True)
    yv = pd.to_numeric(val["zigzag_action"]).to_numpy(np.int64)
    xv_raw = tabm._base_input(val, base_cols)
    ev_expert = tabm._route_probs(val).argmax(1)
    vdays = val.timestamp.dt.floor("D").to_numpy()
    fwd1 = val["fwd_1h_bp"].to_numpy(np.float64)
    results = []

    for arm in ("base", "deep"):                      # base 먼저 (빠르다)
        t0, t1 = E.ARMS[arm]
        tm = (df.timestamp >= t0) & (df.timestamp <= t1 + " 23:59:59")
        tr = df[tm].reset_index(drop=True)
        yt = pd.to_numeric(tr["zigzag_action"]).to_numpy(np.int64)
        rt = tabm._route_probs(tr)
        n = len(tr); split = max(int(n * 0.85), min(n - 1, 512))
        xs, scaler = tabm._standardize_fit(tabm._base_input(tr, base_cols))
        xv_std = tabm._standardize_apply(xv_raw, scaler)
        tail_expert = rt[split:].argmax(1)
        log(f"\n{'='*90}\n=== {arm}  TRAIN {t0}~{t1} {n:,}행 · 꼬리 {n-split:,}\n{'='*90}")

        for wname, wfn in (("old", w_old), ("new", w_new)):
            Ds, Qs, tDs, tQs = [], [], [], []
            for seed in SEEDS5:
                D = np.zeros((len(val), 3)); Q = np.zeros((len(val), 3))
                tD = np.zeros((n - split, 3)); tQ = np.zeros((n - split, 3))
                for ei in range(3):
                    w = wfn(yt, rt[:, ei].astype(np.float32))
                    m, _ = E.fit_expert(xs[:split], yt[:split], w[:split],
                                        xs[split:], yt[split:], w[split:],
                                        seed=seed, ei=ei, device=device)
                    sel = ev_expert == ei
                    if sel.any():
                        D[sel], Q[sel] = E.heads(m, xv_std[sel], device)
                    ts = tail_expert == ei
                    if ts.any():
                        tD[ts], tQ[ts] = E.heads(m, xs[split:][ts], device)
                Ds.append(D); Qs.append(Q); tDs.append(tD); tQs.append(tQ)
                ls = (D.argmax(1) == 1).sum() / max((D.argmax(1) != 0).sum(), 1)
                log(f"  {wname}/seed {seed}: 방향 롱비중(게이트 전) {ls:.3f}")
            Dm, Qm = np.mean(Ds, 0), np.mean(Qs, 0)          # 5시드 앙상블
            tDm, tQm = np.mean(tDs, 0), np.mean(tQs, 0)

            for sym in (False, True):
                ql, qsh = thresholds(tDm, tQm, symmetric=sym)
                side = side_from(Dm, Qm, ql, qsh)
                tag = f"{arm}/{wname}/{'대칭게이트' if sym else '단일임계'}"
                r = evaluate(tag, side, fwd1, yv, vdays)
                r.update({"arm": arm, "weighting": wname, "symmetric": sym,
                          "q_long": ql, "q_short": qsh, "seeds": SEEDS5})
                results.append(r)
                el, es = r["excess_vs_long"], r["excess_vs_short"]
                log(f"    [{tag}] q(롱/숏)={ql:.3f}/{qsh:.3f} · n {r['n']:,} · 롱 {r['long_share']*100:.1f}% "
                    f"· 총 {r['gross_bp']:+.2f} · 중앙 {r['median_bp']:+.2f} · 양수일 {r['pos_day_share']*100:.1f}% "
                    f"· 최고1일제거 {r['drop_top1d_bp']:+.2f}")
                log(f"        모델−롱 {el['bp']:+.2f} [{el['ci95'][0]:+.2f},{el['ci95'][1]:+.2f}] · "
                    f"모델−숏 {es['bp']:+.2f} [{es['ci95'][0]:+.2f},{es['ci95'][1]:+.2f}]"
                    f"{'  🟢모델−숏 0배제' if es['ci95'][0] > 0 else ''}")
            OUTJ.write_text(json.dumps(results, indent=2, default=float))

    log(f"\n{'='*90}\n=== 최종 요약 ===")
    log(f"{'판':<28}{'롱%':>7}{'총bp':>8}{'중앙':>7}{'양수일':>8}{'모델−롱':>9}{'모델−숏':>9}{'숏CI0배제':>10}")
    for r in results:
        log(f"{r['name']:<28}{r['long_share']*100:>6.1f}%{r['gross_bp']:>8.2f}{r['median_bp']:>7.2f}"
            f"{r['pos_day_share']*100:>7.1f}%{r['excess_vs_long']['bp']:>+9.2f}"
            f"{r['excess_vs_short']['bp']:>+9.2f}"
            f"{('🟢' if r['excess_vs_short']['ci95'][0] > 0 else '  0포함'):>10}")
    log(f"\n저장: {OUTJ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
