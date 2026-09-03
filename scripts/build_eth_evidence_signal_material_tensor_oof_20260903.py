#!/usr/bin/env python3
"""증거신호 재료 텐서 **OOF본** (2026-09-03) -- 20260902 텐서의 누수 수정본.

WHY
---
20260902 텐서의 `<sig>_proba`/`_pct`는 **TRAIN 전체 구간에서 in-sample**이었다. 메타라벨이
2024-01~2025-08로 학습됐는데 그 예측을 같은 구간의 모든 봉에 채워 넣었기 때문이다.
그 결과 이 텐서로 학습하는 **모든 하류 모델이 지름길을 배운다**:

  내부검증(TRAIN 뒤 15%, 시간순) IC +0.4830  vs  진짜 VAL IC −0.0199   ← 간극 +0.50
  용량을 8배 줄여도 내부검증 불변(+0.4654~+0.4837) · 학습곡선은 교과서적으로 건강
  5시드가 중단 iter·내부검증 IC까지 소수점 4자리 동일  ← 결정론적 지름길

⚠️**라이브 서빙은 무관하다.** 오염은 **과거 TRAIN 행에만** 있고 **그 행으로 학습할 때만**
물린다. 대시보드/메타라벨 라이브 추론은 TRAIN-only 고정 컨텍스트로 현재 봉을 채점한다.

WHAT
----
진입 모델이 쓴 것과 **동일한 OOF 산출**(`tmp/eth_entry_oof_metalabel_20260903/`)을 쓴다:
  워밍업  2024-01-01 ~ 2024-04-30  메타라벨 최초 학습용 (텐서에서 **NaN**, 학습 제외 대상)
  fold    2024-05 ~ 2025-08 4등분, fold k는 그 시작 이전 데이터만 본 모델이 채운다(확장창)
  최종    2025-09 이후는 TRAIN 전체(<2025-09-01)로 학습한 모델 (4시드 평균)
백분위 매핑도 각 단계의 자기 학습분포 기준이다(인과적).

수정 효과 (`research_eth_material_tensor_oof_rebuild_test_20260903.py` 실측):
  누수지표(내부검증 IC − VAL IC)  +0.5029 → **−0.0430**  (12배 축소)
  하류 HGB VAL/HOLDOUT 부호        −0.0199/−0.0296 → **+0.0272/+0.0182**

⚠️단, 누수를 없애자 **학습이 원시 signed 합을 못 이긴다**(격자 16조합 0/16, Ridge 3α 전부
음수 -- 오염 상태에서 Ridge가 이겼던 건 선형으로 누수를 착취한 것이었다). 즉 이 텐서의 현재
사용 형태는 `signed` 합이며, RL/DL에 먹일 때는 **저용량 헤드를 전제**해야 한다.

열 구성은 20260902본과 동일하되 `_proba_cal`은 뺐다 -- OOF 산출에 캘리브레이션 열이 없고,
BSS가 어차피 ≤+0.057이라 이 신호들은 확률원이 아니라 랭커다.
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

SRC = ROOT / "tmp/eth_causal_population_metalabel_20260902"
OOFD = ROOT / "tmp/eth_entry_oof_metalabel_20260903"
OLD = ROOT / "data/materials/eth_evidence_signal_tensor_20260902"
OUT_DIR = ROOT / "data/materials/eth_evidence_signal_tensor_oof_20260903"
ETH_REGIME = ROOT / "tmp/eth_regime_s12k3_clean_20260902/predictions.parquet"
BTC_REGIME = ROOT / "tmp/btc_regime_s24k3_clean_20260902/predictions.parquet"
WARMUP_END = pd.Timestamp("2024-05-01")


def log(m): print(f"[oof-tensor] {m}", flush=True)


def main() -> int:
    from research_eth_kalman_demarker_gridscreen_20260831 import load_klines

    cfg = json.loads((SRC / "config.json").read_text())["cfg"]
    kl = load_klines()[["timestamp", "open", "high", "low", "close", "volume"]].sort_values(
        "timestamp").reset_index(drop=True)
    n = len(kl)
    pos = {t: i for i, t in enumerate(pd.DatetimeIndex(kl["timestamp"]))}
    out = kl[["timestamp"]].copy()
    log(f"기준 {n:,}봉 {kl.timestamp.min()} ~ {kl.timestamp.max()}")

    stats = []
    for name, cc in cfg.items():
        H = int(cc["horizon"])
        f = OOFD / f"{name}_oof.csv"
        if not f.exists():
            log(f"⚠️ {name}: OOF 없음 -- 중단"); return 1
        d = pd.read_csv(f, parse_dates=["timestamp"])
        n_all = len(d)
        d = d[np.isfinite(d["pct_oof"])].copy()          # 워밍업 구간은 OOF 값이 없다
        d = d[d.timestamp.isin(pos)].copy()
        d["i"] = [pos[t] for t in d.timestamp]
        d["dir"] = np.where(d["is_bottom"] == 1, 1.0, -1.0)
        d = d.sort_values(["i", "proba_oof"]).drop_duplicates("i", keep="last")

        fire = np.zeros(n); age = np.ones(n)
        P = np.zeros((n, 3))                              # proba, pct, dir
        pi = {int(i): (float(a), float(b), float(c)) for i, a, b, c in
              zip(d["i"], d["proba_oof"], d["pct_oof"], d["dir"])}
        fire[d["i"].to_numpy()] = d["dir"].to_numpy()
        last, li = (0.0, 0.0, 0.0), -10**9
        for i in range(n):
            if i in pi:
                last, li = pi[i], i
            el = i - li
            if el < H:
                P[i] = last; age[i] = el / H
        out[f"{name}_fire"] = fire
        out[f"{name}_proba"] = P[:, 0]
        out[f"{name}_pct"] = P[:, 1]
        out[f"{name}_signed"] = P[:, 1] * P[:, 2]
        out[f"{name}_age"] = age
        cov = float((P[:, 0] > 0).mean())
        stats.append({"signal": name, "horizon": H, "n_fires_oof": int(len(d)),
                      "n_fires_total": int(n_all), "warmup_dropped": int(n_all - len(d)),
                      "bar_coverage": round(cov, 4)})
        log(f"{name:26s} H={H:<3} 발동 {len(d):6,}/{n_all:6,} "
            f"(워밍업 제외 {n_all-len(d):,}) · 커버리지 {cov:.1%}")

    # ⚠️2026-09-03: 이전 버전은 `if p.exists()`로 **조용히 건너뛰었고**, 서버에 parquet이 없어
    # 레짐 2열이 통째로 빠진 41열 텐서가 만들어졌다(구본은 51열). 없으면 실패해야 한다 --
    # 오늘 파생거래소 500행 제약도 같은 "조용히 넘어가기"로 13개 피쳐를 망가뜨렸다.
    for tag, p in (("eth", ETH_REGIME), ("btc", BTC_REGIME)):
        if not p.exists():
            log(f"❌ 레짐 predictions 없음: {p}")
            return 1
        r = pd.read_parquet(p)
        out = out.merge(r.rename(columns={"regime": f"regime_{tag}"}), on="timestamp", how="left")
        out[f"regime_{tag}"] = out[f"regime_{tag}"].ffill().fillna(-1).astype(int)
        log(f"regime_{tag} 병합: chop 비중 {float((out[f'regime_{tag}']==2).mean()):.3f}")

    # ---- 검증 ----
    log("\n=== 검증 ===")
    bad = 0
    for name, cc in cfg.items():
        H = int(cc["horizon"])
        fc = out[f"{name}_fire"].to_numpy(); pc = out[f"{name}_proba"].to_numpy()
        act = np.flatnonzero(pc > 0)
        for i in act[:: max(1, len(act) // 2000)]:
            if not np.any(fc[max(0, i - H + 1):i + 1] != 0):
                bad += 1; break
    log(f"  유지창 밖 활성값 {bad}건 (0이어야 정상) · 결측 {int(out.isna().sum().sum())}개")
    n_exp = 1 + 5 * len(cfg) + 2                    # timestamp + 신호당 5열 + 레짐 2열
    if out.shape[1] != n_exp:
        log(f"❌ 열 수 {out.shape[1]} != 기대 {n_exp} -- 조용한 누락이 있다")
        return 1
    log(f"  ✅열 수 {out.shape[1]} = 기대값 (timestamp + {len(cfg)}×5 + 레짐 2)")
    warm = out.timestamp < WARMUP_END
    act_warm = sum(int((out[f"{c}_proba"][warm] > 0).sum()) for c in cfg)
    log(f"  ⭐워밍업(<{WARMUP_END.date()}) 구간 활성값 {act_warm}건 -- "
        f"{'정상(0이어야 함)' if act_warm == 0 else '⚠️OOF 없는 구간에 값이 있다'}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / "eth_evidence_material_5m.parquet", index=False)
    pd.DataFrame(stats).to_csv(OUT_DIR / "coverage.csv", index=False)
    json.dump({
        "rows": len(out), "cols": out.shape[1],
        "range": [str(out.timestamp.min()), str(out.timestamp.max())],
        "supersedes": "data/materials/eth_evidence_signal_tensor_20260902 (proba was in-sample over all of TRAIN)",
        "oof_source": "tmp/eth_entry_oof_metalabel_20260903 (warmup 2024-01~04 excluded; 4 expanding folds over 2024-05~2025-08; >=2025-09 from full-TRAIN model)",
        "columns_per_signal": ["<sig>_fire (+1 bottom / -1 top / 0)",
                               "<sig>_proba (OOF metalabel prob, held for the signal's own HORIZON)",
                               "<sig>_pct (percentile within that fold's own training distribution)",
                               "<sig>_signed (pct * direction)",
                               "<sig>_age (elapsed/HORIZON, 1.0 = expired)"],
        "extra": ["regime_eth", "regime_btc (0 bull / 1 bear / 2 chop)"],
        "causality": "value at index i uses information through bar i's CLOSE only",
        "warmup": f"before {WARMUP_END.date()} all signal columns are 0/age=1 -- no OOF value exists; EXCLUDE from downstream training",
        "leak_fix_measured": {"gap_innerVAL_minus_VAL_before": 0.5029, "after": -0.0430,
                              "downstream_hgb_VAL_before": -0.0199, "after": 0.0272},
        "known_limitation": "with the leak removed, no learned function beat the raw `signed` sum (grid 0/16, Ridge negative at all 3 alphas) -- use a low-capacity head",
        "signals": cfg,
    }, open(OUT_DIR / "README.json", "w"), ensure_ascii=False, indent=2)
    log(f"\n산출: {OUT_DIR}  ({len(out):,}행 × {out.shape[1]}열)")

    # 구본에 폐기 표시
    if (OLD / "README.json").exists():
        r = json.loads((OLD / "README.json").read_text())
        r["SUPERSEDED_BY"] = str(OUT_DIR.relative_to(ROOT))
        r["SUPERSEDED_REASON"] = ("`_proba`/`_pct` are in-sample over ALL of TRAIN -- any downstream "
                                  "model trained on these rows learns a shortcut (innerVAL +0.4830 "
                                  "vs true VAL -0.0199). Kept for reproducibility only. "
                                  "Live serving is unaffected.")
        r["SUPERSEDED_DATE"] = "2026-09-03"
        (OLD / "README.json").write_text(json.dumps(r, ensure_ascii=False, indent=2))
        log(f"구본 {OLD.name}/README.json 에 폐기 사유 기록")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
