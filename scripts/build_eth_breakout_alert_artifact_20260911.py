#!/usr/bin/env python3
"""경보 TabPFN 아티팩트 동결 + **라이브 경로 파리티 검증** (2026-09-11).

TabPFN 은 in-context 라 «모델» = 컨텍스트 행 + 사전학습 트랜스포머다. 그래서 컨텍스트를
그대로 얼린다(패널을 다시 만들어 재현하려 들지 않는다 — 재현 실패가 조용히 어긋난다).

🔴**라이브에는 «창 내 순위»가 없다.** 평가는 창 안에서 시드별 점수를 순위로 바꿔 평균했는데,
   워커는 한 번에 한 봉만 본다. 그래서 시드별 **기준 분포**를 같이 얼려 백분위로 바꾼다:
       score = mean_i  percentile(p_i(row), ref_dist_i)
   그리고 이 경로가 평가 수치를 실제로 내는지 **같은 창에서 재계산해 대조**한다.
   (기준 분포는 어느 시드의 컨텍스트에도 안 들어간 TRAIN 행에서 낸다)

🔴발동선은 **후행 분위로 적응**시킨다. 첫 판에서 기준분포 q99 로 고정했더니 실제 발동률이
   VAL 0.41% / OOS 1.72% / FWD **0.08%** 로 20배 갈렸다 — FWD 면 4일에 한 번 켜진다.
   확률 분포가 레짐 따라 이동하기 때문이다. (컨텍스트를 얼리는 것과 발동선을 적응시키는 것은
   **다른 축**이다 — 얼리는 게 나은 건 컨텍스트 쪽이다: FWD lift 4.10 vs 최근10k 0.84)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_breakout_alert_tabpfn_20260911 as R  # noqa: E402

OUT = ROOT / "data" / "live" / "eth_breakout_alert_artifact"
RULE_ID = "eth_breakout_alert_moveatr_tabpfn_20260911"
N_REF = 20000            # 기준 분포 표본(어느 컨텍스트에도 안 들어간 TRAIN 행)
CAP, N_EST = 10000, 8
FIRE = R.FIRE            # 0.01 — 배포 규칙 q99 와 같은 발동률


def pct_of(p: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
    """기준 분포에서의 백분위. 라이브와 오프라인이 **같은 함수**를 써야 파리티가 성립한다."""
    return np.searchsorted(ref_sorted, p, side="right") / len(ref_sorted)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--klines", default=str(R.KL5))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, default=3)   # 파리티에서 3개면 5개와 동급(18.6s vs 31.0s)
    ap.add_argument("--eval-stride", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    raw = R.build(Path(a.klines))
    gate = R.apply_gate(raw, "none")                       # 압축 게이트는 역효과 — 전 봉을 본다
    # ⭐타깃 확정: move_atr(이탈폭/ATR96). moveabs 는 최고 단변량이 atr_pct(lift 5.37)라
    #   «전환»이 아니라 «지금 변동성이 높다»의 다른 이름이었다(라벨 로직 연구 2026-09-11).
    p = R.label(raw, "move", cand=gate)
    feats = R.feature_cols(p)
    X = p[feats].to_numpy(np.float32)
    yall = p["y"].to_numpy()

    ok = pd.Series(gate, index=p.index) & np.isfinite(p.fwd_rv) & (p.win != "")
    bnd = np.zeros(len(p), bool)
    for nm, aa, bb in R.WINDOWS:
        idx = np.flatnonzero((p.timestamp >= aa) & (p.timestamp <= bb))
        if len(idx):
            bnd[idx[-R.EMBARGO:]] = True
    use = (ok & ~bnd & np.isfinite(p[feats]).all(axis=1)).to_numpy()
    ev_keep = (np.arange(len(p)) % a.eval_stride == 0) | (p.win == "TRAIN").to_numpy()
    use_ev = use & ev_keep
    tr = np.flatnonzero(use & (p.win == "TRAIN").to_numpy())
    print(f"[패널] 피쳐 {len(feats)}개 · TRAIN {len(tr):,}행 · 기저 {yall[tr].mean()*100:.2f}%")

    seeds = tuple(int(x) for x in np.random.default_rng(20260911).choice(10**6, a.seeds,
                                                                        replace=False))
    from tabpfn import TabPFNClassifier
    ctx_idx, models = [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        idx = tr[R.sub_idx(len(tr), CAP, rng)]
        m = TabPFNClassifier(device=a.device, random_state=sd, n_estimators=N_EST)
        m.fit(X[idx], yall[idx])
        ctx_idx.append(idx); models.append(m)
        print(f"  시드 {sd:7d} 컨텍스트 {len(idx):,} 적합 완료", flush=True)

    # 기준 분포 — 어느 시드의 컨텍스트에도 안 들어간 TRAIN 행에서만 낸다
    used = np.unique(np.concatenate(ctx_idx))
    pool = np.setdiff1d(tr, used)
    ref = np.sort(np.random.default_rng(7).choice(pool, min(N_REF, len(pool)), replace=False))
    print(f"[기준분포] 컨텍스트 밖 TRAIN {len(pool):,} 중 {len(ref):,}행")
    ref_sorted = []
    for sd, m in zip(seeds, models):
        pr = m.predict_proba(X[ref])[:, 1]
        ref_sorted.append(np.sort(pr).astype(np.float32))
        print(f"  시드 {sd:7d} 기준분포 완료 (중앙 {np.median(pr):.4f})", flush=True)

    def live_score(rows: np.ndarray, k: int) -> np.ndarray:
        """**라이브와 같은 경로**: 시드별 확률 → 기준분포 백분위 → 평균."""
        return np.mean([pct_of(models[i].predict_proba(X[rows])[:, 1], ref_sorted[i])
                        for i in range(k)], axis=0)

    cut = float(np.quantile(live_score(ref, len(seeds)), 1 - FIRE))
    print(f"[참고] 기준분포 고정 q{100*(1-FIRE):.0f} = {cut:.4f} — **쓰지 않는다**(창별 발동률이 20배 갈린다)")
    TRAIL = 672                 # 후행 창(솎은 행 기준). stride 3 이면 실봉 2016 = 배포 규칙과 같다

    # ── 파리티: 라이브 경로가 평가 수치를 내는가 + 시드 몇 개면 충분한가
    print(f"\n[라이브 경로 파리티 · 매칭 커버리지 {FIRE:.0%} lift]")
    print(f"  표기: 매칭커버리지lift / **후행분위lift** (후행 발동률)")
    print(f"{'시드수':>5s} " + " ".join(f"{w:>21s}" for w in R.EVAL_WINS) + "   봉당 추정")
    table, scores = {}, {}
    for k in range(1, len(seeds) + 1):
        cells, row = [], {}
        for w in R.EVAL_WINS:
            m = np.flatnonzero(use_ev & (p.win == w).to_numpy())
            if not len(m):
                cells.append(f"{'-':>20s}"); continue
            sc = live_score(m, k)
            scores.setdefault(k, {})[w] = (m, sc)
            lf, nf, pc = R.lift_at(sc, yall[m])
            # 후행 분위 발동선 — 라이브가 실제로 쓸 경로. 앞 TRAIL 행은 아직 못 켠다.
            thr = pd.Series(sc).rolling(TRAIL, min_periods=200).quantile(1 - FIRE).shift(1).to_numpy()
            fire = np.isfinite(thr) & (sc >= thr)
            lf_tr = (yall[m][fire].mean() / yall[m].mean()) if fire.sum() >= 20 else np.nan
            row[w] = {"lift_matched": lf, "prec": pc, "fires": nf,
                      "fire_rate_fixed": float((sc >= cut).mean()),
                      "lift_trailing": None if not np.isfinite(lf_tr) else float(lf_tr),
                      "fire_rate_trailing": float(fire.mean())}
            cells.append(f"{lf:6.2f}/{lf_tr:5.2f} ({fire.mean()*100:4.2f}%)")
        table[k] = row
        print(f"{k:5d} " + " ".join(cells) + f"   {6.2*k:5.1f}s", flush=True)

    best_rule = {"VAL": 5.06, "OOS": 2.36, "FWD": 2.31}   # move_atr·게이트없음 구성의 최고 규칙
    print(f"\n  최고 규칙: " + " · ".join(f"{w} {v:.2f}" for w, v in best_rule.items()))
    ok_k = [k for k, r in table.items()
            if all((r.get(w, {}).get("lift_trailing") or 0) > best_rule[w] for w in R.EVAL_WINS)]
    print(f"  후행분위 발동선으로 세 창 모두 규칙 초과하는 시드수: {ok_k or '없음'}")

    np.savez_compressed(OUT / "context.npz",
                        **{f"X_{i}": X[ci].astype(np.float32) for i, ci in enumerate(ctx_idx)},
                        **{f"y_{i}": yall[ci] for i, ci in enumerate(ctx_idx)},
                        **{f"ref_{i}": rs for i, rs in enumerate(ref_sorted)})
    meta = {
        "rule_id": RULE_ID, "created_utc": datetime.now(timezone.utc).isoformat(),
        "features": feats, "seeds": list(seeds), "n_estimators": N_EST, "context_rows": CAP,
        "label": {"kind": "move_atr", "horizon_bars": R.H,
                  "desc": "앞 24봉(2시간) 최대 이탈폭 / ATR96 이 창별 상위 5%인가",
                  "top_q": R.TOPQ},
        "gate": {"kind": "none", "why": "압축 게이트는 기저를 낮춘다(2.8~3.6% vs 무조건부 5.0%)"},
        "scoring": {"path": "seed proba -> percentile(ref_dist) -> mean",
                    "fire": {"kind": "trailing_quantile", "q": 1 - FIRE, "window_bars": 2016,
                             "min_periods": 200,
                             "why": "고정선은 창별 발동률이 0.08~1.72% 로 20배 갈렸다"},
                    "fixed_cut_reference_only": cut, "fire_target": FIRE},
        "train_span": [str(p.timestamp.iloc[tr[0]]), str(p.timestamp.iloc[tr[-1]])],
        "n_train": int(len(tr)), "base_rate": float(yall[tr].mean()),
        "parity": {str(k): {w: {kk: (None if not np.isfinite(vv) else round(float(vv), 4))
                                for kk, vv in d.items()} for w, d in r.items()}
                   for k, r in table.items()},
        "best_rule_same_config": best_rule,
        "note": "사람이 보는 경보다. 매매 트리거가 아니다. 방향은 말하지 않는다 — «크게 움직일 "
                "확률»만 말한다. 컨텍스트는 전 기간 무작위로 얼렸고 최근 데이터로 갱신하지 않는다"
                "(최근 10k 로 바꾸면 FWD lift 4.10 -> 0.84).",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"\n저장: {OUT}  (context.npz {(OUT/'context.npz').stat().st_size/1e6:.1f}MB)")
    print("⚠️라이브 모듈은 이 파일만 읽는다 — 패널을 다시 만들어 컨텍스트를 재현하지 않는다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
