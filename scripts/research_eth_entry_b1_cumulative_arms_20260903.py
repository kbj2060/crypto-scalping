#!/usr/bin/env python3
"""B1: 피쳐군 누적 arm 비교 -- 재료가 실제로 기여하는가 (2026-09-03).

피쳐군 (2026-09-03 사용자와 확정):
  A  Tier0 23 (트리거 봉, 규칙)          누수 없음
  B  팔·신호 메타 (arm/sig_id/side/atr)   누수 없음
  C  재료 `pct_oof` (OOF 교차적합)        ⭐이 프로젝트의 핵심 질문
  D  레짐 OOF (ETH/BTC, 확장창)           누수 차단됨
  (E 체결경로는 대기 중 취소 모델이 필요해 다음 단계)

⚠️워밍업(2024-01~04)은 OOF가 없으므로 필터 TRAIN에서 제외한다.

평가는 두 축으로 한다 -- 앞선 B0-a에서 **잣대가 틀렸었기 때문**이다:
  ① 무제한 자본 : 기준선 "양쪽 다 걸기". 여기선 거의 모든 팔이 양수라 필터가 원리상 불리하다.
  ② ⭐N슬롯 순차 : 자본 제약. **어느 팔을 잡을지 골라야 하므로 순위매김이 직접 수익이 된다.**
     실제 매매는 ②다.

모델은 로컬 HGB(빠른 선별). 승자만 서버 TabPFN 2단으로 확정한다 --
피쳐군 비교는 표본 크기가 동일하므로 GBM 프록시가 유효하다.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

SRC = ROOT / "tmp/eth_causal_population_metalabel_20260902"
OOFM = ROOT / "tmp/eth_entry_oof_metalabel_20260903"
OOFR = ROOT / "tmp/eth_entry_oof_regime_20260903"
ARMS = ROOT / "tmp/eth_entry_direction_oracle_v2_20260903/both_arms.csv"
OUT = ROOT / "tmp/eth_entry_b1_20260903"
WARMUP_END = pd.Timestamp("2024-05-01")
SEEDS = [76010, 130820, 194636, 331076, 703883]
HP = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=60,
          l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, n_iter_no_change=25)
TAUS = [-np.inf, 0.0, 0.0002, 0.0005, 0.0010, 0.0020]
SLOTS = [1, 2, 4]


def log(m): print(f"[b1] {m}", flush=True)


def stat(v):
    v = np.asarray(v, float)
    if len(v) == 0: return (0, 0.0, 0.0)
    w, l = v[v > 0].sum(), -v[v < 0].sum()
    return (len(v), float(v.mean() * 1e4), float(w / l) if l > 0 else float("inf"))


def slot_sim(df, keep, n_slots):
    """체결 시각순으로 진행. 슬롯이 비어 있으면 잡고, 아니면 건너뛴다.
    반환: 실제 잡은 팔들의 순수익 배열."""
    d = df[keep & df.filled.astype(bool)].sort_values("fill_i")
    taken, busy_until = [], []
    for fi, ei, y in zip(d.fill_i.to_numpy(), d.exit_i.to_numpy(), d.y.to_numpy()):
        busy_until = [b for b in busy_until if b > fi]
        if len(busy_until) < n_slots:
            taken.append(y); busy_until.append(ei)
    return np.asarray(taken, float)


def main() -> int:
    cfg = json.loads((SRC / "config.json").read_text())
    base = [c for c in cfg["features"] if c != "is_bottom"]

    a = pd.read_csv(ARMS, parse_dates=["ts"])
    rows = []
    for nm, retc, fillc, btf in (("sig", "p_sig", "sig_filled", "bars_to_fill_sig"),
                                 ("flip", "p_flip", "flip_filled", "bars_to_fill_flip")):
        t = a[["signal", "ts", "sig_dir", "atr", "split"]].copy()
        t["arm"] = 1 if nm == "sig" else 0
        t["y"] = a[retc].fillna(0.0).to_numpy()
        t["filled"] = a[fillc].to_numpy().astype(int)
        t["btf"] = a[btf].to_numpy()
        rows.append(t)
    d = pd.concat(rows, ignore_index=True)

    # 트리거 봉 Tier0 + OOF 재료
    F = []
    for name in cfg["cfg"]:
        x = pd.read_csv(SRC / f"{name}_causal_fires.csv", parse_dates=["timestamp"])[["timestamp"] + base]
        x["signal"] = name
        o = pd.read_csv(OOFM / f"{name}_oof.csv", parse_dates=["timestamp"])[["timestamp", "pct_oof"]]
        x = x.merge(o, on="timestamp", how="left")
        F.append(x)
    F = pd.concat(F, ignore_index=True).rename(columns={"timestamp": "ts"})
    d = d.merge(F, on=["signal", "ts"], how="left")

    for k in ("eth", "btc"):
        r = pd.read_parquet(OOFR / f"regime_oof_{k}.parquet").rename(
            columns={"timestamp": "ts", "regime_oof": f"regime_{k}"})[["ts", f"regime_{k}"]]
        d = d.merge(r, on="ts", how="left")
        d[f"regime_{k}"] = d[f"regime_{k}"].fillna(-1).astype(int)

    d["sig_id"] = pd.Categorical(d["signal"]).codes
    d = d[d.ts >= WARMUP_END].copy()                       # 워밍업 제외
    d = d.dropna(subset=["pct_oof"]).sort_values("ts").reset_index(drop=True)
    # 슬롯 시뮬용 인덱스 (체결봉 / 청산봉 근사 = 체결 + horizon)
    hz = {k: int(v["horizon"]) for k, v in cfg["cfg"].items()}
    d["hz"] = d.signal.map(hz)
    d["ts_i"] = (d.ts - d.ts.min()).dt.total_seconds() // 300
    d["fill_i"] = d.ts_i + d.btf.clip(lower=0)
    d["exit_i"] = d.fill_i + d.hz

    GA = base
    GB = ["arm", "sig_id", "sig_dir", "atr"]
    GC = ["pct_oof"]
    GD = ["regime_eth", "regime_btc"]
    ARMS_DEF = {"A+B": GA + GB, "A+B+C": GA + GB + GC, "A+B+C+D": GA + GB + GC + GD,
                "A+B+D": GA + GB + GD}
    tr = (d.split == "TRAIN").to_numpy()
    log(f"행 {len(d):,} | TRAIN {int(tr.sum()):,} " +
        " ".join(f"{k} {int(v):,}" for k, v in d.split.value_counts().items() if k != "TRAIN"))
    log(f"체결률 {float(d.filled.mean()):.1%} | 워밍업 제외 후 시작 {d.ts.min().date()}")

    log("\n=== 기준선 (모델 없음) ===")
    for wn in ("VAL", "OOS", "HOLDOUT"):
        w = d[d.split == wn]
        n, m, pf = stat(w.y)
        line = f"  {wn:8s} 무제한 팔당 {m:+6.2f}bp PF{pf:5.2f}"
        for ns in SLOTS:
            v = slot_sim(w, np.ones(len(w), bool), ns)
            _, sm, spf = stat(v)
            line += f" | {ns}슬롯 n={len(v):4d} {sm:+6.2f}bp PF{spf:5.2f}"
        log(line)

    res = []
    for aname, feats in ARMS_DEF.items():
        X = d[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X[tr].median())
        preds = [HistGradientBoostingRegressor(random_state=s, **HP)
                 .fit(X[tr], d.loc[tr, "y"]).predict(X) for s in SEEDS]
        d[f"pred_{aname}"] = np.mean(preds, axis=0)
        cors = {wn: float(np.corrcoef(d.loc[d.split == wn, f"pred_{aname}"], d.loc[d.split == wn, "y"])[0, 1])
                for wn in ("VAL", "OOS")}
        log(f"\n▶ {aname}  ({len(feats)}피쳐)  예측-실제 상관 VAL {cors['VAL']:+.4f} / OOS {cors['OOS']:+.4f}")
        for tau in TAUS:
            row = {"arm": aname, "tau_bp": (tau * 1e4 if np.isfinite(tau) else -999)}
            out = []
            for wn in ("VAL", "OOS"):
                w = d[d.split == wn]
                k = (w[f"pred_{aname}"] > tau).to_numpy()
                _, m, _ = stat(np.where(k, w.y.to_numpy(), 0.0))
                row[f"{wn}_unlim"] = round(m, 2)
                cell = f"무제한 {m:+6.2f}"
                for ns in SLOTS:
                    v = slot_sim(w, k, ns)
                    _, sm, _ = stat(v)
                    row[f"{wn}_slot{ns}"] = round(sm, 2)
                    row[f"{wn}_slot{ns}_n"] = len(v)
                    cell += f" · {ns}슬롯 {sm:+6.2f}(n{len(v)})"
                out.append(f"{wn} {cell}")
            res.append(row)
            print(f"   τ={row['tau_bp']:7.1f}bp | " + " | ".join(out))

    r = pd.DataFrame(res)
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "cumulative_arms.csv", index=False)
    d.to_csv(OUT / "arm_rows.csv", index=False)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
