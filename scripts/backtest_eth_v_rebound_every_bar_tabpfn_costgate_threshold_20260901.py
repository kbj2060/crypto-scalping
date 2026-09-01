#!/usr/bin/env python3
"""매 봉 스코어링 배포판 -- **실제 서빙 확률(TabPFN + 동결 컨텍스트)**로 경제성 게이트 재실행
   + 0.5 임계값이 여전히 옳은지 판단.

## 왜 이 스크립트가 필요한가 (두 개의 미결)

1. **경제성이 프록시 확률로만 측정됐다.** 매 봉 재설계의 트레일링 경제성 +8.01bp는
   `backtest_eth_v_rebound_every_bar_trailing_costgate_20260901.py`가 GBM(HistGradientBoosting,
   전체 TRAIN 182,969행 학습)으로 호출 population을 뽑아 잰 값이다. 그런데 라이브가 실제로
   서빙하는 건 **TabPFN + 동결 18,000행 컨텍스트**다. 분류 AUC는 둘이 거의 같았지만
   (TabPFN 0.6942 vs GBM 0.6953), 경제성은 AUC가 아니라 **상위 확률 꼬리의 순위**가 결정한다 --
   같은 AUC라도 어떤 봉을 호출하느냐가 다르면 bp가 달라질 수 있다. 이 저장소는 GBM과 TabPFN이
   갈린 전례가 있다(V_REBOUND 자신이 TabPFN에서 GBM을 +0.020/+0.014로 이겼음).

2. **라벨 발생률이 32.5% -> 14.6%로 바뀌었다.** 게이트 시절의 0.5 임계값을 그대로 물려받았는데,
   모집단이 바뀌었으므로 확률의 의미도 바뀌었다. 0.5가 여전히 합리적 운영점인지, 아니면 호출이
   너무 드물어지거나(화면에 거의 안 뜸) 너무 흔해지는지(신호 가치 희석) 실측이 필요하다.

## 무엇을 재는가

- **배포 설정 그대로** 재현: `live_eth_sweep_v_rebound_signal_20260829.py`의 TRAIN_CONTEXT_CSV를
  그대로 읽어 `TabPFNClassifier(random_state=20260829, ignore_pretraining_limits=True)`에 fit.
  전체 TRAIN 재학습이 아니다 -- 라이브가 매 사이클 하는 일과 문자 그대로 같은 fit이다.
- 임계값 스윕(0.40~0.70): 호출률 / precision / base 대비 lift, VAL·OOS 각각.
- 경제성: 각 임계값마다 트레일링 그리드(SL x ARM x Trail = 240조합) 전수. 선정은 **VAL만**,
  OOS는 선정 후 1회 평가. 낙관/비관 봉내순서 둘 다 보고
  (feedback_intrabar_ordering_optimistic_pessimistic_bracket).
- 방향 뒤집기 대조군: 같은 그리드를 부호 반전으로 재실행 --
  **그리드 전체에 적용**해야 한다(feedback_trailing_stop_low_arm_noise_harvest_artifact:
  단일 config만 뒤집으면 오판. fib_extension_exhaustion가 이걸로 클레임 철회됨).

⚠️ **HOLDOUT 미터치.** VAL/OOS만 쓴다. HOLDOUT은 단일노출 원칙이고 이 신호에서 이미 두 번 소모됐다.
⚠️ 라이브 코드 변경 없음. 이 스크립트는 읽기 전용 측정이다.

Run on the server (GPU) via handoff:
  handoff.sh launch server <job> -- python scripts/backtest_eth_v_rebound_every_bar_tabpfn_costgate_threshold_20260901.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# 경제성 시뮬레이터/피쳐/스플릿을 기존 검증된 스크립트에서 그대로 재사용한다(재구현 금지 --
# simulate_trailing_vec는 scalar 구현과의 self-check로 이미 검증돼 있다).
BT = ROOT / "scripts/backtest_eth_v_rebound_every_bar_trailing_costgate_20260901.py"
_spec = importlib.util.spec_from_file_location("everybar_costgate_20260901", BT)
_bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bt)

_feas = _bt._feas
_vs = _bt._vs
FEATURE_COLUMNS = _bt.FEATURE_COLUMNS
STANDARD_COST_BP = _bt.STANDARD_COST_BP
FORWARD_BARS = _bt.FORWARD_BARS
SL_GRID, ARM_GRID, TRAIL_GRID = _bt.SL_GRID, _bt.ARM_GRID, _bt.TRAIL_GRID

# 라이브 배포판이 실제로 쓰는 컨텍스트/시드를 라이브 모듈에서 직접 읽는다 -- 상수를 여기 복제하면
# 라이브가 바뀌었을 때 조용히 어긋난다.
import live_eth_sweep_v_rebound_signal_20260829 as _live  # noqa: E402
TRAIN_CONTEXT_CSV = _live.TRAIN_CONTEXT_CSV
LIVE_SEED = 20260829

OUT_JSON = ROOT / "data/research/eth_v_rebound_every_bar_tabpfn_costgate_20260901/report.json"
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# _feas.build_long_frame()은 `< VAL_END`(2026-01-01)에서 잘라 TRAIN/VAL 2분할만 만든다 -- 그
# 스크립트가 VAL까지만 필요했기 때문. 여기서는 OOS도 필요하므로 자르는 경계를 OOS 끝으로 밀고
# split을 직접 3분할한다. **feasibility 스크립트 자체는 건드리지 않는다**(다른 스크립트들이
# import해서 쓰는 검증된 산출물이라, 거기 상수를 바꾸면 그쪽 동작이 조용히 달라진다).
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")  # HOLDOUT은 이 이후 -- 여기서 잘라 미터치 보장


def log(msg: str) -> None:
    print(f"[tabpfn_costgate] {msg}", flush=True)


def build_called(scored: pd.DataFrame, thr: float, kl: pd.DataFrame) -> pd.DataFrame:
    """임계값 이상인 봉만 골라 전방 200봉 OHLC를 붙인다(경제성 시뮬레이터 입력 형식)."""
    called = scored.loc[scored["model_proba"] >= thr]
    ts_to_pos = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    rows = []
    for _, ev in called.iterrows():
        i = ts_to_pos.get(np.datetime64(ev["timestamp"].tz_localize(None)))
        if i is None or i + FORWARD_BARS + 1 >= len(kl):
            continue
        rows.append({
            "side": "long" if ev["is_downside"] == 1 else "short",
            "atr": float(ev["atr"]), "entry_price": float(o[i + 1]),
            "model_proba": float(ev["model_proba"]), "label": float(ev["label"]),
            "fwd_open": o[i + 1:i + 1 + FORWARD_BARS], "fwd_high": h[i + 1:i + 1 + FORWARD_BARS],
            "fwd_low": l[i + 1:i + 1 + FORWARD_BARS], "fwd_close": c[i + 1:i + 1 + FORWARD_BARS],
        })
    return pd.DataFrame(rows)


def main() -> int:
    t0 = time.time()
    from sklearn.metrics import roc_auc_score
    from tabpfn import TabPFNClassifier
    import torch
    log(f"cuda: {torch.cuda.is_available()}")

    log("building all-bar long frame (OOS 끝까지)...")
    _feas.VAL_END = OOS_END  # 자르는 경계만 민다 -- split은 아래에서 직접 3분할
    long = _feas.build_long_frame()
    long = long.loc[long["label"].notna()].dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                     np.where(long["timestamp"] < VAL_END, "VAL", "OOS"))
    assert long["timestamp"].max() < OOS_END, "HOLDOUT 누출"
    log(f"split 분포: {long['split'].value_counts().to_dict()}  "
        f"(마지막 봉 {long['timestamp'].max()}, HOLDOUT 경계 {OOS_END})")
    # ⚠️ tz_localize(None)이 필수다. build_called()의 ts_to_pos는 kl["timestamp"].to_numpy()를
    # 키로 쓰는데, tz-aware Series의 to_numpy()는 datetime64가 아니라 **Timestamp 객체 배열**을
    # 준다 -- 조회 키인 naive np.datetime64와 하나도 매칭되지 않아 호출이 조용히 0건이 된다
    # (2026-09-01 이 스크립트 첫 실행에서 실제로 전 임계값 0건으로 스킵됐다). 기존 검증된
    # _bt.main()도 같은 이유로 이 줄을 갖고 있다.
    kl = _vs.load_klines(_feas.ETH_CSV)[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)

    # === 배포 설정 그대로 fit ===
    ctx = pd.read_csv(TRAIN_CONTEXT_CSV)
    log(f"동결 컨텍스트: {TRAIN_CONTEXT_CSV.name}  n={len(ctx)}  라벨비율={ctx['label'].mean():.4f}")
    clf = TabPFNClassifier(device="cuda", random_state=LIVE_SEED, ignore_pretraining_limits=True)
    clf.fit(ctx[FEATURE_COLUMNS], ctx["label"].to_numpy())
    log("fit 완료 (라이브가 매 사이클 하는 것과 동일한 fit)")

    scored = {}
    for split in ("VAL", "OOS"):
        s = long.loc[long["split"] == split].copy()
        if s.empty:
            log(f"  ⚠️ {split} 비어 있음 -- 스킵")
            continue
        s["model_proba"] = clf.predict_proba(s[FEATURE_COLUMNS])[:, 1]
        auc = float(roc_auc_score(s["label"].to_numpy(), s["model_proba"].to_numpy()))
        log(f"  {split}: n={len(s)}  base={s['label'].mean():.4f}  AUC={auc:.4f}")
        scored[split] = s

    # === 임계값 스윕: 호출률/precision/lift ===
    log("")
    log("=== 임계값 스윕 (호출률 / precision / lift) ===")
    thr_table = {}
    for thr in THRESHOLDS:
        row = {}
        for split, s in scored.items():
            called = s.loc[s["model_proba"] >= thr]
            base = float(s["label"].mean())
            prec = float(called["label"].mean()) if len(called) else float("nan")
            row[split] = {
                "n_called": int(len(called)),
                "call_rate": round(float(len(called) / len(s)), 4),
                "precision": round(prec, 4) if len(called) else None,
                "lift_vs_base": round(prec / base, 3) if len(called) and base > 0 else None,
                "base": round(base, 4),
            }
        thr_table[f"{thr:.2f}"] = row
        parts = " | ".join(
            f"{sp} 호출 {r['n_called']:>5d}({r['call_rate']*100:5.2f}%) "
            f"prec={r['precision'] if r['precision'] is not None else float('nan'):.4f} "
            f"lift={r['lift_vs_base'] if r['lift_vs_base'] is not None else float('nan'):.2f}x"
            for sp, r in row.items())
        log(f"  thr={thr:.2f}  {parts}")

    # === 경제성: 임계값마다 그리드 전수, 선정은 VAL만 ===
    log("")
    log("=== 경제성 게이트 (선정=VAL만, OOS는 선정 후 1회 평가) ===")
    econ = {}
    for thr in THRESHOLDS:
        by_split = {}
        for split, s in scored.items():
            df = build_called(s, thr, kl)
            n_called_raw = int((s["model_proba"] >= thr).sum())
            if len(df) < n_called_raw * 0.5:
                # 전방 200봉이 모자라 잘리는 끝부분 말고는 이렇게 크게 줄 이유가 없다 --
                # 조용한 타임스탬프 미스매치를 다시 놓치지 않도록 크게 경고한다.
                log(f"  ⚠️ thr={thr:.2f} {split}: 임계값 통과 {n_called_raw}건 중 {len(df)}건만 "
                    f"전방봉 매칭 -- 타임스탬프 정렬 확인 필요")
            if len(df) < 50:
                log(f"  thr={thr:.2f} {split}: 호출 {len(df)}건 -- 표본 부족, 스킵")
                by_split[split] = {"n": int(len(df)), "insufficient": True}
                continue
            grid = _bt.run_grid(df, flip=False)
            flip = _bt.run_grid(df, flip=True)
            by_split[split] = {"n": int(len(df)), "grid": grid, "flip_grid": flip}
        if "VAL" not in by_split or by_split["VAL"].get("insufficient"):
            econ[f"{thr:.2f}"] = by_split
            continue

        # VAL에서 낙관/비관 둘 다 비용 넘는 것 중 비관 최고를 고른다(보수적 선정)
        vg = by_split["VAL"]["grid"]
        ok = [g for g in vg if g["opt_bp"] > 0 and g["pess_bp"] > 0]
        best = max(ok, key=lambda g: g["pess_bp"]) if ok else max(vg, key=lambda g: g["pess_bp"])
        key = (best["sl"], best["arm"], best["trail"])

        def find(grid):
            return next(g for g in grid if (g["sl"], g["arm"], g["trail"]) == key)

        sel = {"selected_on": "VAL", "config": {"sl": key[0], "arm": key[1], "trail": key[2]},
               "val_passes_both_orderings": bool(ok)}
        for split in by_split:
            if by_split[split].get("insufficient"):
                continue
            g, f = find(by_split[split]["grid"]), find(by_split[split]["flip_grid"])
            sel[split] = {"n": by_split[split]["n"], "opt_bp": round(g["opt_bp"], 2),
                          "pess_bp": round(g["pess_bp"], 2), "win_rate": round(g["win_rate"], 4),
                          "flip_opt_bp": round(f["opt_bp"], 2), "flip_pess_bp": round(f["pess_bp"], 2)}
        # 방향 뒤집기 노이즈수확 아티팩트 점검 -- 그리드 전체 기준
        for split in by_split:
            if by_split[split].get("insufficient"):
                continue
            g_ok = sum(1 for g in by_split[split]["grid"] if g["opt_bp"] > 0 and g["pess_bp"] > 0)
            f_ok = sum(1 for g in by_split[split]["flip_grid"] if g["opt_bp"] > 0 and g["pess_bp"] > 0)
            sel[f"{split}_grid_profitable"] = {"normal": g_ok, "flipped": f_ok,
                                               "total": len(by_split[split]["grid"])}
        # ⚠️뒤집기가 그리드에서 더 많이 수익나면 노이즈수확 아티팩트를 의심해야 한다
        # (feedback_trailing_stop_low_arm_noise_harvest_artifact_20260901). 판별 열쇠는 **ARM**:
        # 아티팩트는 ARM이 낮을 때(트레일이 거의 즉시 무장돼 잡음을 수확) 나타난다. 수익 조합의
        # ARM 분포를 정방향/뒤집기 각각 찍어, 뒤집기 우위가 저ARM에 몰려 있는지 확인한다.
        for split in by_split:
            if by_split[split].get("insufficient"):
                continue
            for tag, gname in (("정방향", "grid"), ("뒤집기", "flip_grid")):
                prof = [g for g in by_split[split][gname] if g["opt_bp"] > 0 and g["pess_bp"] > 0]
                if not prof:
                    continue
                by_arm = {}
                for g in prof:
                    by_arm[g["arm"]] = by_arm.get(g["arm"], 0) + 1
                sel[f"{split}_{tag}_arm_dist"] = {str(k): v for k, v in sorted(by_arm.items())}

        econ[f"{thr:.2f}"] = {"selection": sel,
                              "grids": {sp: {"n": v["n"], "grid": v.get("grid"),
                                             "flip_grid": v.get("flip_grid")}
                                        for sp, v in by_split.items()}}
        s = sel
        log(f"  thr={thr:.2f}  SL/ARM/Trail={key[0]}/{key[1]}/{key[2]}  "
            + "  ".join(f"{sp} n={s[sp]['n']} opt{s[sp]['opt_bp']:+.2f}bp pess{s[sp]['pess_bp']:+.2f}bp "
                        f"승률{s[sp]['win_rate']*100:.1f}% (뒤집기 opt{s[sp]['flip_opt_bp']:+.2f}bp)"
                        for sp in ("VAL", "OOS") if sp in s))
        for sp in ("VAL", "OOS"):
            k = f"{sp}_grid_profitable"
            if k in s:
                log(f"          {sp} 그리드 수익 조합: 정방향 {s[k]['normal']}/{s[k]['total']}  "
                    f"뒤집기 {s[k]['flipped']}/{s[k]['total']}")
                for tag in ("정방향", "뒤집기"):
                    dist = s.get(f"{sp}_{tag}_arm_dist")
                    if dist:
                        log(f"            {tag} 수익조합 ARM분포: "
                            + "  ".join(f"ARM={a}:{n}" for a, n in dist.items()))

    report = {
        "signal": "v_rebound_every_bar_tabpfn_costgate_threshold", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope": {"model": "TabPFN, 배포판과 동일 fit(동결 컨텍스트+동일 시드)",
                  "context_csv": str(TRAIN_CONTEXT_CSV.relative_to(ROOT)),
                  "context_n": int(len(ctx)), "seed": LIVE_SEED,
                  "holdout_touched": False, "live_code_changed": False,
                  "purpose": "GBM 프록시로만 측정됐던 경제성을 실제 서빙 확률로 재실행 + 0.5 임계값 재검토"},
        "cost_bp": STANDARD_COST_BP, "forward_bars": FORWARD_BARS,
        "grid": {"sl": list(SL_GRID), "arm": list(ARM_GRID), "trail": list(TRAIL_GRID)},
        "split_auc": {sp: round(float(roc_auc_score(s["label"], s["model_proba"])), 4)
                      for sp, s in scored.items()},
        "split_n": {sp: int(len(s)) for sp, s in scored.items()},
        "threshold_table": thr_table, "economics": econ,
        "gbm_proxy_reference_bp": 8.01,
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
