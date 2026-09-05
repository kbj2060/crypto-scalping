#!/usr/bin/env python3
"""증거신호 칩을 **화면에 표시되는 규약 그대로** 따라갔을 때의 성과 (2026-09-05, 사용자 요청).

사용자: *"내가 지금까지 계속 증거신호 칩과 지속 신호를 보고 트레이딩을 하고 있는데 증거신호칩과
V자반등칩의 신호를 따라가는게 늘 성과가 좋았어."* → *"증거신호칩 신호를 추적해서 성과 분석해줘"*.

⭐**왜 기존 측정과 다른가.** §5.23/§5.29가 잰 것은 "첫발동 봉에 칩 방향 진입 → **경제 브래킷**
(5.0 SL / 1.5 ARM / 0.1 trail ×ATR, 200봉 만기) 청산"이다. 그런데 **화면이 사람에게 말하는 것은
그게 아니다** -- 칩은 신호별 익절가(`_tp_price`: 발동 봉 종가 ± k×atr_pct)를 표시하고, 배지는
그 익절가에 닿거나 `horizon_bars`가 지날 때까지 유지된다. 손절선은 **표시하지 않는다**.
즉 화면을 보고 매매하는 사람의 규약은 "발동 → 표시된 익절가까지 보유, 안 닿으면 지평선에서 정리"다.
이 형태로는 아직 측정된 적이 없다. 이 스크립트가 그것을 그대로 재현한다.

규약(라이브 `scripts/live_evidence_signal_metalabel_20260829.py`와 1:1):
  · 모집단 = 신호별·측면별 raw 발동 중 **직전 `horizon_bars` 안에 같은 측면 raw 발동이 없는** 봉
    (`_find_recent_raw_fire_pos`의 라이브 결정 모집단 정의와 동일)
  · 익절가 = `close[i] * (1 ± k*atr_pct[i])`, atr_pct = atr[i]/close[i]  (`_tp_price` 원문)
  · 도달 판정 = 발동 **다음 봉부터** 고가/저가 기준 (`_tp_touched` 원문)
  · 진입 = `open[i+1]` (사람이 마감 봉 신호에 반응할 수 있는 가장 이른 시점)
  · 청산 = 익절가 도달 봉의 익절가, 아니면 `close[i+horizon]`
  · 방향 = 칩 방향(bottom→롱, top→숏) = **페이드**. 비교군으로 반대(지속)도 같이 잰다.
  · 비용 10bp 차감(표준 수수료; 수수료 우대 가정 금지)

⚠️**같은 측면 무작위 귀무**를 반드시 함께 낸다. VAL/OOS 구간이 둘 다 하락장(−32%/−29%)이라
숏 쪽 원시 bp는 드리프트만으로도 양수가 된다(§5.29 §7-1). 귀무는 같은 신호·같은 측면 비율·같은
horizon·같은 익절 규약으로 **무작위 봉**에 진입했을 때의 분포다.

일군집(하루) 부트스트랩 CI. 연구·개발 점수 -- HOLDOUT(≥2026-04-01) 미접촉.
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

TRAIN_END = pd.Timestamp("2025-09-01")
VAL_END = pd.Timestamp("2026-01-01")
OOS_END = pd.Timestamp("2026-04-01")          # HOLDOUT 경계 -- 이후는 로드 단계에서 잘라낸다
COST_BP = 10.0
B_BOOT = 2000
B_NULL = 200
SEED = 20260905
OUT = ROOT / "data/research/eth_chip_displayed_convention_pnl_20260905"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def log(m: str) -> None:
    print(f"[chip-pnl] {m}", flush=True)


# ── 칩 설정: 라이브 모듈에서 직접 읽는다(상수 복사 금지) ────────────────────────────
_META = _load("meta_live", "scripts/live_evidence_signal_metalabel_20260829.py")
CHIP = {name: {"horizon": int(cfg["horizon_bars"]), "k": float(cfg["k"])}
        for name, cfg in _META.METALABEL_SIGNALS.items()}


def first_fire_positions(fired: np.ndarray, horizon: int) -> np.ndarray:
    """직전 `horizon`봉 안에 같은 측면 발동이 없는 발동 봉(라이브 결정 모집단)."""
    idx = np.flatnonzero(fired)
    keep, last = [], -10**9
    for i in idx:
        if i - last > horizon:
            keep.append(i)
        last = i
    return np.asarray(keep, dtype=int)


def exit_at_tp_or_horizon(o, h, l, c, pos, side, k, atr_pct, horizon, n):
    """칩 규약 청산. 반환: (entry, exit, bars_held, tp_hit) -- 지평선이 프레임을 넘으면 NaN."""
    if pos + 1 >= n:
        return np.nan, np.nan, 0, False
    entry = o[pos + 1]
    tp = c[pos] * (1 + k * atr_pct[pos]) if side == "bottom" else c[pos] * (1 - k * atr_pct[pos])
    end = min(pos + horizon, n - 1)
    if end <= pos:
        return np.nan, np.nan, 0, False
    seg = slice(pos + 1, end + 1)
    hit = (h[seg] >= tp) if side == "bottom" else (l[seg] <= tp)
    w = np.flatnonzero(hit)
    if len(w):
        return entry, tp, int(w[0] + 1), True
    return entry, c[end], int(end - pos), False


def pnl_bp(entry, exit_, side_sign) -> float:
    return side_sign * (exit_ / entry - 1.0) * 1e4 - COST_BP


def day_cluster_ci(vals: np.ndarray, days: np.ndarray, rng, b: int = B_BOOT):
    """하루 단위 블록 부트스트랩 평균 CI(행 단위 t가 클러스터를 무시하는 문제 회피)."""
    uniq = np.unique(days)
    if len(uniq) < 2:
        return (np.nan, np.nan, len(uniq))
    by = {d: vals[days == d] for d in uniq}
    means = np.empty(b)
    for i in range(b):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        means[i] = np.concatenate([by[d] for d in pick]).mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(uniq))


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    log("신호 프레임 재구성(라이브 compute_signals 경로)...")
    _s1 = _load("s1_chip", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
    _s1.VAL_END = OOS_END
    sig, _feat, eth = _s1.build_sig()

    ts = pd.to_datetime(sig["timestamp"]).dt.tz_localize(None)
    keep = ts < OOS_END                                    # ⚠️HOLDOUT 미접촉
    sig = sig.loc[keep].reset_index(drop=True)
    ts = ts.loc[keep].reset_index(drop=True)
    o, h, l, c = (sig[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    atr = sig["atr"].to_numpy(dtype=float)
    atr_pct = atr / c
    n = len(sig)
    day = ts.dt.floor("D").to_numpy()
    split = np.where(ts < TRAIN_END, "TRAIN", np.where(ts < VAL_END, "VAL", "OOS"))
    log(f"  {n:,}봉 · {str(ts.iloc[0])[:10]} ~ {str(ts.iloc[-1])[:10]}")

    rows = []
    for name, cfg in CHIP.items():
        hz, k = cfg["horizon"], cfg["k"]
        for side in ("bottom", "top"):
            col = f"{side}_{name}"
            if col not in sig.columns:
                log(f"  ⚠️{col} 없음 -- 건너뜀")
                continue
            fired = sig[col].fillna(False).to_numpy(bool)
            sgn = 1.0 if side == "bottom" else -1.0           # 칩 방향(페이드)
            for pos in first_fire_positions(fired, hz):
                if not (np.isfinite(atr_pct[pos]) and atr_pct[pos] > 0):
                    continue
                e, x, bh, tp_hit = exit_at_tp_or_horizon(o, h, l, c, pos, side, k, atr_pct, hz, n)
                if not (np.isfinite(e) and np.isfinite(x)):
                    continue
                rows.append({"signal": name, "side": side, "pos": pos, "split": split[pos],
                             "day": day[pos], "bars_held": bh, "tp_hit": tp_hit,
                             "chip_bp": pnl_bp(e, x, sgn), "cont_bp": pnl_bp(e, x, -sgn),
                             "horizon": hz, "k": k})
    df = pd.DataFrame(rows)
    log(f"  칩 규약 거래 {len(df):,}건 (신호 {df['signal'].nunique()}종)")

    # ── 같은 측면 무작위 귀무: 같은 신호의 (측면, horizon, k) 조합을 무작위 봉에 걸어본다 ──
    log(f"  같은 측면 무작위 귀무 {B_NULL}회...")
    null_means = {s: [] for s in ("TRAIN", "VAL", "OOS")}
    for sp in ("TRAIN", "VAL", "OOS"):
        m = df["split"] == sp
        sub = df.loc[m]
        if sub.empty:
            continue
        pool = np.flatnonzero(split == sp)
        specs = sub[["side", "horizon", "k"]].to_numpy(object)
        for _ in range(B_NULL):
            picks = rng.choice(pool, size=len(specs), replace=True)
            vals = []
            for (side, hz, k), pos in zip(specs, picks):
                if not (np.isfinite(atr_pct[pos]) and atr_pct[pos] > 0):
                    continue
                e, x, _bh, _t = exit_at_tp_or_horizon(o, h, l, c, int(pos), side, float(k),
                                                      atr_pct, int(hz), n)
                if np.isfinite(e) and np.isfinite(x):
                    vals.append(pnl_bp(e, x, 1.0 if side == "bottom" else -1.0))
            if vals:
                null_means[sp].append(float(np.mean(vals)))

    report = {"convention": "chip_displayed: entry open[i+1], TP = close[i]*(1±k*atr_pct[i]), "
                            "exit at TP touch (high/low) or close[i+horizon]",
              "cost_bp": COST_BP, "holdout_touched": False, "chip_config": CHIP,
              "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
              "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
              "splits": {}, "per_signal": {}, "per_side": {}}

    log("\n=== 칩 표시 규약대로 따라갔을 때 (칩 방향 = 페이드) ===")
    log(f"{'창':6s} {'n':>6s} {'건/일':>6s} {'익절도달':>7s} {'칩 방향':>9s} {'일CI':>18s} "
        f"{'지속(반대)':>10s} {'무작위귀무':>10s} {'백분위':>6s}")
    for sp in ("TRAIN", "VAL", "OOS"):
        sub = df.loc[df["split"] == sp]
        if sub.empty:
            continue
        v = sub["chip_bp"].to_numpy()
        d = sub["day"].to_numpy()
        lo, hi, ndays = day_cluster_ci(v, d, rng)
        nm = np.array(null_means[sp]) if null_means[sp] else np.array([np.nan])
        pct = float((nm < v.mean()).mean() * 100) if np.isfinite(nm).all() else np.nan
        log(f"{sp:6s} {len(sub):6,d} {len(sub)/max(ndays,1):6.1f} {sub['tp_hit'].mean():7.1%} "
            f"{v.mean():+9.2f} [{lo:+7.2f},{hi:+7.2f}] {sub['cont_bp'].mean():+10.2f} "
            f"{np.nanmean(nm):+10.2f} {pct:6.1f}")
        report["splits"][sp] = {"n": int(len(sub)), "n_days": int(ndays),
                                "trades_per_day": round(len(sub) / max(ndays, 1), 2),
                                "tp_hit_rate": round(float(sub["tp_hit"].mean()), 4),
                                "chip_bp": round(float(v.mean()), 3),
                                "chip_ci": [round(lo, 3), round(hi, 3)],
                                "cont_bp": round(float(sub["cont_bp"].mean()), 3),
                                "null_mean_bp": round(float(np.nanmean(nm)), 3),
                                "null_percentile": (round(pct, 1) if np.isfinite(pct) else None),
                                "median_bars_held": int(sub["bars_held"].median())}

    log("\n=== 신호별 (칩 방향) ===")
    log(f"{'신호':26s} {'H':>3s} {'k':>5s} " + " ".join(f"{s:>18s}" for s in ("TRAIN", "VAL", "OOS")))
    for name in CHIP:
        sub_all = df.loc[df["signal"] == name]
        if sub_all.empty:
            continue
        cells, per = [], {}
        for sp in ("TRAIN", "VAL", "OOS"):
            s2 = sub_all.loc[sub_all["split"] == sp]
            if s2.empty:
                cells.append(f"{'-':>18s}"); continue
            lo, hi, _ = day_cluster_ci(s2["chip_bp"].to_numpy(), s2["day"].to_numpy(), rng, 800)
            cells.append(f"{s2['chip_bp'].mean():+7.1f}({len(s2):4d})".rjust(18))
            per[sp] = {"n": int(len(s2)), "chip_bp": round(float(s2["chip_bp"].mean()), 2),
                       "ci": [round(lo, 2), round(hi, 2)],
                       "tp_hit_rate": round(float(s2["tp_hit"].mean()), 3)}
        log(f"{name:26s} {CHIP[name]['horizon']:3d} {CHIP[name]['k']:5.2f} " + " ".join(cells))
        report["per_signal"][name] = per

    log("\n=== 측면별 (칩 방향 vs 같은 측면 무작위) ===")
    for sp in ("TRAIN", "VAL", "OOS"):
        for side in ("bottom", "top"):
            s2 = df.loc[(df["split"] == sp) & (df["side"] == side)]
            if s2.empty:
                continue
            lo, hi, _ = day_cluster_ci(s2["chip_bp"].to_numpy(), s2["day"].to_numpy(), rng, 800)
            ko = "바닥→롱" if side == "bottom" else "천장→숏"
            log(f"  {sp:6s} {ko}  n={len(s2):5d}  {s2['chip_bp'].mean():+7.2f}bp "
                f"[{lo:+6.2f},{hi:+6.2f}]  익절도달 {s2['tp_hit'].mean():.1%}")
            report["per_side"].setdefault(sp, {})[side] = {
                "n": int(len(s2)), "chip_bp": round(float(s2["chip_bp"].mean()), 2),
                "ci": [round(lo, 2), round(hi, 2)],
                "tp_hit_rate": round(float(s2["tp_hit"].mean()), 3)}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    df.to_parquet(OUT / "trades.parquet", index=False)
    log(f"\n산출: {OUT} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
