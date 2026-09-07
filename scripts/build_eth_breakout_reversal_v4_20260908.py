#!/usr/bin/env python3
"""돌파/되돌림 **v4** -- 1분 테이프(CVD/흡수/임팩트) + 30초 호가 OBI (2026-09-08).

사용자 제공 자원:
  · `data/research/eth_tape_1m_20260906.parquet` -- aggTrades 1분 집계 29컬럼, 2024-04-20~
  · `binance_data/bookDepth/ETHUSDT/` -- 869일 · 30초 · ±0.2~5% (→ `bookdepth_obi_1m.parquet`)

## 왜 이 데이터인가 (견고성 감사가 가리킨 곳)
ATR 상태 서술의 정체는 **ATR 이 아니라 발현 속도**였다:
속도 통제 후 ATR 효과 **−0.0008**(소멸) · ATR 통제 후 속도 효과 **−0.0608**(잔존), corr +0.462.
⇒ 다음 질문은 *"왜 빨랐나"* 이고 5분봉 가격으로는 답할 수 없다. 테이프가 정확히 그걸 답한다:
  얇은 호가를 때린 임팩트(kyle_lambda·impact_per_vol) / 고래인가(lg_·xl_) /
  한 방인가 지속 흐름인가(n_runs·max_runlen·switch_rate) / 반대편 호가가 비었나(OBI).

## 🔴경계 계약 준수 (`.claude/CLAUDE.md` Event-Label Boundary Contract)
결정은 분 `s1`(트리거 분) 안에서 일어나고 라벨은 `s1` 부터 탐색한다.
⇒ **테이프·호가 피쳐는 전부 분 `s1-1` 이하만 쓴다.** 이동구간 집계도 `s0 .. s1-1`.
   테이프가 1분 해상도라 5분봉 OI 처럼 "트리거 봉을 못 보는" 문제가 없다 -- 해상도가 정확히 맞는다.

⚠️테이프는 2024-04-20 부터다. 공정 비교를 위해 **기준 모델도 같은 행으로 제한**해 재측정한다.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

MY = ROOT / "tmp/eth_breakout_atr_state_20260908_s1"
TAPE = ROOT / "data/research/eth_tape_1m_20260906.parquet"
OBI = MY / "bookdepth_obi_1m.parquet"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
TCOL = ["n_trades", "volume", "notional", "signed_vol", "signed_notional", "q_max", "q_mean",
        "lg_vol", "lg_signed", "xl_vol", "xl_signed", "n_runs", "max_runlen", "med_gap_ms",
        "imbalance", "lg_share", "lg_imbalance", "xl_imbalance", "kyle_lambda",
        "impact_per_vol", "switch_rate", "avg_trade_size"]


def main() -> int:
    print("[1/4] 로드 ...", flush=True)
    d = pd.read_parquet(MY / "dataset_v2.parquet")
    d = d[(d.anchor == "first_fire") & (d.T_mult == 1.0)].reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    eth = B._load_kl(B.ETH_KL); ts5 = eth["timestamp"].to_numpy()
    m1 = pd.read_csv(KL1, usecols=["timestamp"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy()
    T = pd.read_parquet(TAPE)
    if "ts" in T.columns: T = T.set_index(pd.to_datetime(T["ts"]))
    T = T.sort_index()
    T = T[~T.index.duplicated(keep="last")]
    print(f"      테이프 {T.shape} {T.index[0]} ~ {T.index[-1]}", flush=True)
    O = pd.read_parquet(OBI) if OBI.exists() else None
    if O is not None:
        O = O.sort_index(); O = O[~O.index.duplicated(keep="last")]
        print(f"      OBI {O.shape} {O.index[0]} ~ {O.index[-1]}", flush=True)

    print("[2/4] 1분 격자 정렬 + 롤링 z ...", flush=True)
    grid = pd.DatetimeIndex(ts1)
    Tg = T.reindex(grid)
    V = {c: Tg[c].to_numpy(float) for c in TCOL if c in Tg.columns}
    for c in ("volume", "notional", "n_trades", "kyle_lambda", "avg_trade_size", "q_max"):
        if c in V:
            s = pd.Series(V[c])
            V[f"{c}_z"] = ((s - s.rolling(1440, min_periods=300).mean())
                           / s.rolling(1440, min_periods=300).std().replace(0, np.nan)).to_numpy()
    if O is not None:
        Og = O.reindex(grid)
        for c in O.columns: V[f"obi_{c}"] = Og[c].to_numpy(float)
        s = pd.Series(V["obi_depth_tot"])
        V["obi_depth_z"] = ((s - s.rolling(1440, min_periods=300).mean())
                            / s.rolling(1440, min_periods=300).std().replace(0, np.nan)).to_numpy()

    print("[3/4] 사건별 피쳐 (⭐s1-1 까지만) ...", flush=True)
    s1 = np.searchsorted(ts1, d["timestamp"].to_numpy())
    bi = d["bar_idx"].to_numpy()
    s0 = np.searchsorted(ts1, ts5[np.minimum(bi + 1, len(ts5) - 1)])
    sgn = np.where(d["dir_up"].to_numpy() > 0, 1.0, -1.0)
    last = s1 - 1                                            # ⭐완결된 마지막 분
    ok = (last >= 300) & (last < len(ts1)) & (s0 <= last)
    F = {}
    for c, arr in V.items():
        F[f"t_{c}"] = np.where(ok, arr[np.clip(last, 0, len(arr) - 1)], np.nan)
    # 방향 정렬 (부호 있는 피쳐는 발현 방향 기준으로 뒤집는다)
    for c in ("imbalance", "lg_imbalance", "xl_imbalance", "obi_obi_all", "obi_obi_tight"):
        k = f"t_{c}"
        if k in F: F[f"{k}_al"] = F[k] * sgn
    # 이동구간 집계 s0..s1-1
    MAXL = 20
    span = np.arange(MAXL)[None, :]
    j = np.clip(s0[:, None] + span, 0, len(ts1) - 1)
    msk = (span <= (last - s0)[:, None]) & ok[:, None]
    def agg(name, fn):
        a = V.get(name)
        if a is None: return
        x = np.where(msk, a[j], np.nan)
        F[f"mv_{name}_{fn}"] = (np.nanmean(x, 1) if fn == "mean" else
                                np.nanmax(x, 1) if fn == "max" else np.nansum(x, 1))
    for nm in ("volume", "notional", "signed_vol", "signed_notional", "lg_vol", "lg_signed",
               "xl_vol", "xl_signed", "n_trades"): agg(nm, "sum")
    for nm in ("kyle_lambda", "impact_per_vol", "switch_rate", "imbalance", "avg_trade_size"):
        agg(nm, "mean")
    for nm in ("kyle_lambda", "q_max", "max_runlen"): agg(nm, "max")
    with np.errstate(all="ignore"):
        F["mv_cvd_align"] = np.sign(F.get("mv_signed_vol_sum", np.zeros(len(d)))) * sgn
        F["mv_cvd_frac"] = F.get("mv_signed_vol_sum", np.nan) / np.maximum(
            F.get("mv_volume_sum", np.nan), 1e-9) * sgn
        F["mv_lg_frac"] = F.get("mv_lg_vol_sum", np.nan) / np.maximum(F.get("mv_volume_sum", np.nan), 1e-9)
        F["mv_xl_frac"] = F.get("mv_xl_vol_sum", np.nan) / np.maximum(F.get("mv_volume_sum", np.nan), 1e-9)
        # ⭐흡수: 큰 거래대금인데 가격이 덜 움직임 (= 반대편이 받아냄)
        move_abs = np.abs(d["T_atr"].to_numpy(float))
        F["mv_absorption"] = np.log1p(F.get("mv_notional_sum", np.nan)) / np.maximum(move_abs * 1e4, 1e-6)
    for k in list(F): F[k] = np.where(np.isfinite(F[k]), F[k], np.nan)
    X = pd.DataFrame(F)
    out = pd.concat([d.reset_index(drop=True), X], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    have = out[[c for c in out.columns if c.startswith("t_")]].notna().any(axis=1)
    print(f"      테이프 결합 {have.mean():.1%} ({have.sum():,}/{len(out):,})", flush=True)
    out.to_parquet(MY / "dataset_v4.parquet")
    print(f"[4/4] 저장 {out.shape} · 신규피쳐 {X.shape[1]}")
    print(json.dumps({"rows": len(out), "new_feats": int(X.shape[1]),
                      "tape_cov": float(have.mean())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
