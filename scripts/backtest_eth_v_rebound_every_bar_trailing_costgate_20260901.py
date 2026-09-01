#!/usr/bin/env python3
"""매 봉 스코어링 재설계 -- 경제성 게이트 (트레일링스톱 그리드 + 방향뒤집기 대조군 + 무작위 기준선).

9-3~9-9로 분류 쪽 스크리닝을 끝낸 매 봉 설계에 대해, 이 저장소의 표준 경제성 게이트를 적용한다.
분류 AUC 통과가 경제성을 보장하지 않는다는 건 이 신호 자신이 증명한 바 있다 -- v7b는 분류
0.734/0.762/0.779로 통과하고도 트레일링스톱 205조합 전부에서 VAL·OOS 동시양수 0개로 전멸했다.

## 이 저장소 관례 준수 사항 (전부 기존 스크립트에서 verbatim)

- `simulate_trailing()` 로직: research_eth_v_rebound_multitrigger_giveback_costgate_flip_audit_
  20260901.py와 동일(재구현 아님). 다만 후보수 x 240조합 x 2(낙관/비관)이 스칼라 루프로는 너무
  느려 **봉단위 루프를 후보축으로 벡터화**했고, 원본 스칼라 구현과 무작위 200행 대조 self-check를
  통과해야만 진행한다(불일치 시 즉시 중단).
- STANDARD_COST_BP=10.0 (수수료 우대 가정 금지, feedback_no_fee_discount_assumptions...)
- entry_price = 발동 다음 봉의 시가. **매 봉 설계에서는 이 가정이 정직하다** -- local_extreme의
  30분 확정지연 문제(9-1/7번항목)가 게이트 제거로 사라졌기 때문.
- 낙관/비관 봉내순서 이중검증(feedback_intrabar_ordering_optimistic_pessimistic_bracket_20260830)
- **방향뒤집기 대조군을 VAL 양수 조합 전체에 적용**(단일 config만 검사하면 오판 --
  feedback_trailing_stop_low_arm_noise_harvest_artifact_20260901, fib_extension_exhaustion가
  이걸로 클레임 철회당한 전례)
- 무작위 진입 기준선(승률이 exit 구조 자체의 효과일 수 있음)

## 두 라벨을 함께 돌리는 이유 (9-9의 순환성 caveat 해소)

9-9는 절대 bp 하한 30bp가 분류에 도움된다고 했으나, 평가 타겟이 그 하한 라벨이라 순환성이 남았다.
**경제성은 라벨과 독립적인 실제 체결 시뮬레이션**이므로, FLOOR=0과 FLOOR=30 모델을 같은 그리드로
돌려 비교하면 그 순환성 없는 판정이 된다.

⚠️ VAL 전용 스크리닝이다. OOS/HOLDOUT 미터치 -- 최종 config 선정이 아니라 "경제성이 살아있는가"를
가리는 관문. 라이브 코드 변경 없음.

Run with the quant_ai conda env:
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/backtest_eth_v_rebound_every_bar_trailing_costgate_20260901.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path("/home/kbj20/crypto-scalping")
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FLOOR_SCRIPT = ROOT / "scripts/research_eth_v_rebound_absolute_bp_floor_sweep_20260901.py"
_spec = importlib.util.spec_from_file_location("bp_floor_sweep_costgate_20260901", FLOOR_SCRIPT)
_fl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fl)
_vs = _fl._vs
_feas = _fl._feas

from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402

OUT_JSON = ROOT / "data/research/eth_v_rebound_every_bar_costgate_20260901/costgate_report.json"

STANDARD_COST_BP = 10.0
FORWARD_BARS = 200
PROBA_THRESHOLD = 0.5
SEEDS = _fl.SEEDS
FEATURE_COLUMNS = _fl.FEATURE_COLUMNS
TRAIN_END = _fl.TRAIN_END
VAL_END = _fl.VAL_END
START = _fl.START

SL_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
ARM_GRID = (0.10, 0.25, 0.5, 0.75, 1.0, 1.5)
TRAIL_GRID = (0.10, 0.15, 0.2, 0.3, 0.5)
TEST_FLOORS = [0, 30]


def log(msg: str) -> None:
    print(f"[costgate] {msg}", flush=True)


def simulate_trailing_scalar(row, sl_mult, arm_mult, trail_mult, pessimistic) -> float:
    """Reference implementation, verbatim from research_eth_v_rebound_multitrigger_giveback_
    costgate_flip_audit_20260901.py -- used ONLY for the vectorised version's self-check."""
    atr, entry, side = row["atr"], row["entry_price"], row["side"]
    opens, highs, lows, closes = row["fwd_open"], row["fwd_high"], row["fwd_low"], row["fwd_close"]
    sign = 1.0 if side == "long" else -1.0
    stop = entry - sign * sl_mult * atr
    armed, best = False, entry
    for o, h, l, c in zip(opens, highs, lows, closes):
        fav_extreme = h if side == "long" else l
        adv_extreme = l if side == "long" else h

        def stop_hit() -> bool:
            return (adv_extreme <= stop) if side == "long" else (adv_extreme >= stop)

        def update_trailing() -> None:
            nonlocal armed, stop, best
            if sign * (fav_extreme - best) > 0:
                best = fav_extreme
            if not armed and sign * (best - entry) >= arm_mult * atr:
                armed = True
            if armed:
                new_stop = best - sign * trail_mult * atr
                if sign * (new_stop - stop) > 0:
                    stop = new_stop

        if pessimistic:
            if stop_hit():
                return sign * (stop - entry) / entry
            update_trailing()
        else:
            update_trailing()
            if stop_hit():
                return sign * (stop - entry) / entry
    return sign * (closes[-1] - entry) / entry


def simulate_trailing_vec(entry, atr, sign, H, L, C, sl, arm, trail, pessimistic) -> np.ndarray:
    """Vectorised over candidates (axis 0); the bar loop (axis 1) stays sequential, exactly
    mirroring the scalar reference's order of operations."""
    n = len(entry)
    stop = entry - sign * sl * atr
    armed = np.zeros(n, dtype=bool)
    best = entry.copy()
    done = np.zeros(n, dtype=bool)
    out = np.zeros(n, dtype=float)

    fav_all = np.where(sign[:, None] > 0, H, L)
    adv_all = np.where(sign[:, None] > 0, L, H)

    def do_update(fav):
        nonlocal best, armed, stop
        live = ~done
        improve = live & (sign * (fav - best) > 0)
        best = np.where(improve, fav, best)
        newly = live & ~armed & (sign * (best - entry) >= arm * atr)
        armed = armed | newly
        new_stop = best - sign * trail * atr
        upd = live & armed & (sign * (new_stop - stop) > 0)
        stop = np.where(upd, new_stop, stop)

    def do_stop(adv):
        nonlocal done, out
        live = ~done
        hit = live & np.where(sign > 0, adv <= stop, adv >= stop)
        out = np.where(hit, sign * (stop - entry) / entry, out)
        done = done | hit

    for t in range(H.shape[1]):
        if done.all():
            break
        fav, adv = fav_all[:, t], adv_all[:, t]
        if pessimistic:
            do_stop(adv)
            do_update(fav)
        else:
            do_update(fav)
            do_stop(adv)
    out = np.where(done, out, sign * (C[:, -1] - entry) / entry)
    return out


def build_called(long: pd.DataFrame, floor: int, kl: pd.DataFrame) -> pd.DataFrame:
    lab = f"label_{floor}"
    tr = long.loc[(long["split"] == "TRAIN") & long[lab].notna()]
    va = long.loc[(long["split"] == "VAL") & long[lab].notna()].copy()
    probs = []
    for sd in SEEDS:
        m = HistGradientBoostingClassifier(random_state=sd, max_iter=300, early_stopping=True,
                                           validation_fraction=0.15)
        m.fit(tr[FEATURE_COLUMNS], tr[lab].to_numpy())
        probs.append(m.predict_proba(va[FEATURE_COLUMNS])[:, 1])
    va["model_proba"] = np.mean(probs, axis=0)
    called = va.loc[va["model_proba"] >= PROBA_THRESHOLD].copy()
    log(f"  FLOOR={floor}: VAL {len(va)} 중 호출 {len(called)} ({len(called)/len(va)*100:.1f}%), "
        f"호출 precision={called[lab].mean():.4f}")

    ts_to_pos = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    rows = []
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    for _, ev in called.iterrows():
        i = ts_to_pos.get(np.datetime64(ev["timestamp"].tz_localize(None)))
        if i is None or i + FORWARD_BARS + 1 >= len(kl):
            continue
        rows.append({
            "side": "long" if ev["is_downside"] == 1 else "short",
            "atr": float(ev["atr"]), "entry_price": float(o[i + 1]),
            "model_proba": float(ev["model_proba"]), "label": float(ev[lab]),
            "fwd_open": o[i + 1:i + 1 + FORWARD_BARS], "fwd_high": h[i + 1:i + 1 + FORWARD_BARS],
            "fwd_low": l[i + 1:i + 1 + FORWARD_BARS], "fwd_close": c[i + 1:i + 1 + FORWARD_BARS],
        })
    return pd.DataFrame(rows)


def pack(df: pd.DataFrame, flip: bool = False):
    sign = np.where(df["side"].to_numpy() == "long", 1.0, -1.0)
    if flip:
        sign = -sign
    return (df["entry_price"].to_numpy(float), df["atr"].to_numpy(float), sign,
            np.stack(df["fwd_high"].to_numpy()), np.stack(df["fwd_low"].to_numpy()),
            np.stack(df["fwd_close"].to_numpy()))


def self_check(df: pd.DataFrame, n_sample: int = 200) -> dict:
    rng = np.random.default_rng(20260901)
    idx = rng.choice(len(df), size=min(n_sample, len(df)), replace=False)
    sub = df.iloc[idx].reset_index(drop=True)
    e, a, s, H, L, C = pack(sub)
    mism = 0
    for sl, arm, trail in ((4.0, 1.5, 0.1), (1.0, 0.25, 0.2), (5.0, 0.10, 0.5)):
        for pess in (False, True):
            vec = simulate_trailing_vec(e, a, s, H, L, C, sl, arm, trail, pess)
            sca = np.array([simulate_trailing_scalar(sub.iloc[i], sl, arm, trail, pess) for i in range(len(sub))])
            mism += int((~np.isclose(vec, sca, atol=1e-12)).sum())
    return {"n_rows": int(len(sub)), "n_configs": 6, "n_mismatches": mism}


def run_grid(df: pd.DataFrame, flip: bool = False) -> list[dict]:
    e, a, s, H, L, C = pack(df, flip=flip)
    out = []
    for sl in SL_GRID:
        for arm in ARM_GRID:
            for trail in TRAIL_GRID:
                opt = simulate_trailing_vec(e, a, s, H, L, C, sl, arm, trail, False)
                pes = simulate_trailing_vec(e, a, s, H, L, C, sl, arm, trail, True)
                out.append({
                    "sl": sl, "arm": arm, "trail": trail,
                    "opt_bp": float(opt.mean() * 1e4 - STANDARD_COST_BP),
                    "pess_bp": float(pes.mean() * 1e4 - STANDARD_COST_BP),
                    "win_rate": float((opt * 1e4 > STANDARD_COST_BP).mean()),
                })
    return out


def main() -> int:
    t0 = time.time()
    log("building long frame (features + floor labels)...")
    feat = _feas.build_all_bar_frame()
    eth = _vs.load_klines(_feas.ETH_CSV)
    btc = _vs.load_klines(_feas.BTC_CSV)
    impl = _vs.load_impl()
    causal = impl.add_causal_columns(eth[["timestamp", "open", "high", "low", "close"]].copy())
    sig = compute_signals(eth, btc_df=btc, funding_df=None)
    sig["atr"] = causal["atr"].to_numpy()

    fields = {"bottom": _fl.side_fields(sig, True), "top": _fl.side_fields(sig, False)}
    rows = []
    for side, is_down in (("bottom", True), ("top", False)):
        d = pd.DataFrame({"timestamp": sig["timestamp"]})
        for fl in TEST_FLOORS:
            d[f"status_{fl}"] = _fl.status_with_floor(fields[side], fl)
        merged = d.merge(feat, on="timestamp", how="inner", suffixes=("", "_f"))
        sub = pd.DataFrame({"timestamp": merged["timestamp"], "side": side})
        sub["is_downside"] = np.int8(1 if is_down else 0)
        level = merged["sweep_level_low"].to_numpy() if is_down else merged["sweep_level_high"].to_numpy()
        atr = merged["atr"].to_numpy(dtype=float)
        pen = (level - merged["low"].to_numpy()) if is_down else (merged["high"].to_numpy() - level)
        sub["sweep_penetration_atr"] = pen / atr
        sub["atr"] = atr
        sub["atr_percentile_864"] = merged["atr_percentile_864"].to_numpy()
        sub["range_width_pct"] = merged["range_width_pct"].to_numpy()
        sub["hour_utc"] = merged["hour_utc"].to_numpy()
        sub["weekday"] = merged["weekday"].to_numpy()
        dz = merged["delta_z"].to_numpy(dtype=float)
        sub["delta_z"] = dz
        sub["flow_aligned_delta_z"] = dz if is_down else -dz
        for col in ["p_fast", "p_slow", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
                    "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi",
                    "bb_width_pctile", "ret3_z", "rsi"]:
            sub[col] = merged[col].to_numpy()
        for fl in TEST_FLOORS:
            sub[f"status_{fl}"] = merged[f"status_{fl}"].to_numpy()
        rows.append(sub)

    long = pd.concat(rows, ignore_index=True)
    long = long.loc[(long["timestamp"] >= START) & (long["timestamp"] < VAL_END)].reset_index(drop=True)
    long = long.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN", "VAL")
    for fl in TEST_FLOORS:
        long[f"label_{fl}"] = np.where(long[f"status_{fl}"] == "v_rebound", 1.0,
                              np.where(long[f"status_{fl}"] == "chop", 0.0, np.nan))

    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)

    results = {}
    for fl in TEST_FLOORS:
        log(f"=== FLOOR={fl}bp 모델의 호출 후보 구축 ===")
        cand = build_called(long, fl, kl)
        log(f"  후보 {len(cand)}건 (fwd {FORWARD_BARS}봉)")
        if len(cand) < 50:
            log("  후보 부족 -- 스킵")
            continue

        sc = self_check(cand)
        log(f"  self-check(벡터화 vs 원본 스칼라): {sc['n_rows']}행 x {sc['n_configs']}config, "
            f"불일치 {sc['n_mismatches']}건")
        if sc["n_mismatches"]:
            log("  ⛔ 벡터화가 원본과 불일치 -- 중단")
            return 1

        grid = run_grid(cand)
        pos = [g for g in grid if g["opt_bp"] > 0 and g["pess_bp"] > 0]
        log(f"  VAL 양수(낙관·비관 동시) 조합: {len(pos)}/{len(grid)}")

        flip_grid = {(g["sl"], g["arm"], g["trail"]): g for g in run_grid(cand, flip=True)}
        genuine = []
        for g in pos:
            f = flip_grid[(g["sl"], g["arm"], g["trail"])]
            gap = g["opt_bp"] - f["opt_bp"]
            rec = {**g, "flip_opt_bp": f["opt_bp"], "gap_bp": gap, "genuine": gap > 0 and f["opt_bp"] < 0}
            genuine.append(rec)
        n_gen = sum(1 for g in genuine if g["genuine"])
        log(f"  방향뒤집기 통과(진짜): {n_gen}/{len(pos)}")

        rng = np.random.default_rng(20260901)
        ridx = rng.choice(len(cand), size=len(cand), replace=True)
        rand = cand.iloc[ridx].copy()
        rand["side"] = rng.choice(["long", "short"], size=len(rand))
        rand_grid = {(g["sl"], g["arm"], g["trail"]): g for g in run_grid(rand)}

        top = sorted(genuine, key=lambda g: -(g["gap_bp"])) if genuine else []
        for g in top[:6]:
            r = rand_grid[(g["sl"], g["arm"], g["trail"])]
            log(f"    SL={g['sl']:.1f}/ARM={g['arm']:.2f}/Tr={g['trail']:.2f}  "
                f"opt={g['opt_bp']:+7.2f} pess={g['pess_bp']:+7.2f} 승률={g['win_rate']:.3f} | "
                f"flip={g['flip_opt_bp']:+7.2f} gap={g['gap_bp']:+6.2f} | 무작위={r['opt_bp']:+7.2f} "
                f"| {'진짜' if g['genuine'] else '아티팩트'}")

        results[f"floor_{fl}"] = {
            "n_called": int(len(cand)), "called_precision": float(cand["label"].mean()),
            "self_check": sc, "n_grid": len(grid), "n_val_positive": len(pos),
            "n_genuine_after_flip": n_gen,
            "top_by_gap": [{**g, "random_baseline_opt_bp": rand_grid[(g["sl"], g["arm"], g["trail"])]["opt_bp"]}
                            for g in top[:15]],
        }

    report = {
        "signal": "v_rebound_every_bar_trailing_costgate", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope": {
            "screening_only": True, "model": "HistGradientBoostingClassifier proxy (not TabPFN)",
            "live_code_changed": False, "holdout_touched": False, "oos_touched": False,
            "population": "ALL bars (every-bar scoring design), VAL only",
            "entry_convention": "next bar open (+1 bar) -- honest here because the every-bar design "
                                 "removes local_extreme's 30-min confirmation delay",
            "cost_bp": STANDARD_COST_BP, "forward_bars": FORWARD_BARS,
            "grid": {"sl": list(SL_GRID), "arm": list(ARM_GRID), "trail": list(TRAIL_GRID)},
        },
        "results": results,
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT_JSON}")
    log(f"total runtime: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
