#!/usr/bin/env python3
"""G — base 의 성과가 **몇 건/며칠에 몰려 있는가**. 캐시된 확률만 쓴다(재학습 없음).

의심: 「데이터가 많은 deep 이 시드 분산이 적어야 하는데 오히려 돈이 더 흔들린다.
      base 는 운으로 한두 번 크게 번 것 아닌가」 -- 09-15 에 실제로 걸렸던 함정이다
      ([[account_t_gap_is_one_trade_not_sizing_20260915]]: 상위 1건 빼니 t 1.02->4.63).

⚠️1h 전방수익은 5분봉에서 **12봉씩 겹친다**. 「상위 1건 제거」는 이웃 11봉이 같은 움직임을
들고 있어 반쪽짜리다. 그래서 **일 단위 제거를 1급**으로 보고 건 단위는 참고로 병기한다.

⭐**시드가 같은 날에서 벌었는가**도 잰다. 몇 건이 지배하면 모든 시드가 그 며칠을 잡으므로
**평균은 안정적으로 보이면서 실체는 불안정**하다 -- base 의 낮은 bp 시드폭(1.40)이 실력이
아니라 그 증상일 수 있다.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eval_omega461_parent_zig075_longwindow_20260917 as E  # noqa: E402
import research_omega461_parent_quality_calibration_and_bias_20260917 as F  # noqa: E402


def day_stats(side, fwd, days):
    ok = (side != 0) & np.isfinite(fwd)
    pnl, d = side[ok] * fwd[ok], days[ok]
    uniq = np.unique(d)
    dmean = np.array([pnl[d == u].mean() for u in uniq])      # 일평균 bp
    dsum = np.array([pnl[d == u].sum() for u in uniq])        # 일 기여(건수 가중)
    order = np.argsort(-dsum)
    tot = dsum.sum()
    out = {"n": int(ok.sum()), "days": len(uniq), "gross_bp": float(pnl.mean()),
           "median_bp": float(np.median(pnl)),
           "pos_day_share": float((dmean > 0).mean())}
    for k in (1, 2, 3, 5):
        keep = np.isin(d, uniq[order[k:]])
        out[f"drop_top{k}d_bp"] = float(pnl[keep].mean()) if keep.sum() > 50 else float("nan")
    out["top1d_pnl_share"] = float(dsum[order[0]] / tot) if tot != 0 else float("nan")
    out["top3d_pnl_share"] = float(dsum[order[:3]].sum() / tot) if tot != 0 else float("nan")
    srt = np.sort(pnl)[::-1]
    out["drop_top1_bp"] = float(srt[1:].mean())
    out["drop_top10_bp"] = float(srt[10:].mean())
    out["top1pct_pnl_share"] = float(srt[:max(1, len(srt) // 100)].sum() / max(srt.sum(), 1e-9))
    out["_top_days"] = [str(pd.Timestamp(u).date()) for u in uniq[order[:10]]]
    return out


def main() -> int:
    df, base_cols = E.load()
    vm = (df.timestamp >= E.VAL[0]) & (df.timestamp <= E.VAL[1] + " 23:59:59")
    val = df[vm].reset_index(drop=True)
    vdays = val.timestamp.dt.floor("D").to_numpy()
    fwd1 = val["fwd_1h_bp"].to_numpy(np.float64)
    store = {k: dict(v.item()) for k, v in np.load(E.OUT / "stageF_probs.npz", allow_pickle=True).items()}

    rows = {}
    print(f"{'모델':<14}{'건':>6}{'일':>5}{'총bp':>8}{'중앙bp':>8}{'양수일':>8}"
          f"{'최고1일제거':>11}{'2일':>8}{'3일':>8}{'최고1일지분':>11}{'상위3일지분':>11}")
    for key, s in store.items():
        _, side = F.gate(s["D"], s["Q"], E.Q_THRESH)
        r = day_stats(side, fwd1, vdays); rows[key] = r
        print(f"{key:<14}{r['n']:>6,}{r['days']:>5}{r['gross_bp']:>8.2f}{r['median_bp']:>8.2f}"
              f"{r['pos_day_share']*100:>7.1f}%{r['drop_top1d_bp']:>11.2f}{r['drop_top2d_bp']:>8.2f}"
              f"{r['drop_top3d_bp']:>8.2f}{r['top1d_pnl_share']*100:>10.1f}%{r['top3d_pnl_share']*100:>10.1f}%")

    print(f"\n--- 건 단위 (겹침 때문에 참고용) ---")
    print(f"{'모델':<14}{'총bp':>8}{'상위1건제거':>11}{'상위10건제거':>12}{'상위1% 지분':>12}")
    for key, r in rows.items():
        print(f"{key:<14}{r['gross_bp']:>8.2f}{r['drop_top1_bp']:>11.2f}{r['drop_top10_bp']:>12.2f}"
              f"{r['top1pct_pnl_share']*100:>11.1f}%")

    print(f"\n--- ⭐시드들이 «같은 날»에서 버는가 (상위10일 겹침) ---")
    for arm in ("deep", "base"):
        ks = [k for k in rows if k.startswith(arm)]
        sets = [set(rows[k]["_top_days"]) for k in ks]
        pair = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
        common = set.intersection(*sets)
        print(f"  {arm}: 쌍별 Jaccard {np.round(pair,3)} · 3시드 공통 상위일 {len(common)}/10 "
              f"{sorted(common)[:5]}")
    dep = set(rows["deployed/-"]["_top_days"])
    for arm in ("deep", "base"):
        u = set.union(*[set(rows[k]["_top_days"]) for k in rows if k.startswith(arm)])
        print(f"  배포본 상위10일 ∩ {arm} = {len(dep & u)}")

    (E.OUT / "stageG_concentration.json").write_text(json.dumps(rows, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
