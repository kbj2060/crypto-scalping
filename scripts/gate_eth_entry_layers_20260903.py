#!/usr/bin/env python3
"""ETH 진입 모델 **층별 재구성 게이트** (2026-09-03).

왜 -- 진입 모델이 네 번(A~D, docs/homer/README.md §5.21) 모델링 중간에 엎어졌다. 넷 다 표준
대조군 5종을 통과하고 무너졌고(대조군도 같은 오염된 기질 위에서 채점되므로), 잡은 조치는 예외
없이 "의심층을 다른 경로로 다시 만들어 대조"한 것이었다. 이 스크립트는 그 재구성을 **층 동결
시점에 강제**한다. 정적 스캐너가 아니다 -- 각 게이트는 숫자를 다시 계산해 파이프라인의 숫자와
대조하고 PASS/FAIL을 낸다.

  L4  known_ts 계약    진입(=라벨 시작)이 트리거가 *알려진* 봉보다 뒤인가 + 인덱스↔시각 정합   (A·D)
  L1  발동 인과성      데이터를 known_ts에서 잘라 다시 만들어도 같은 발동이 나오고,
                       known_ts 직전에서 잘랐을 때 "미래가 오면 사라질 발동"이 없는가            (A)
  L2  라벨 1분 재구성  체결 5분봉을 1분봉으로 갈라 체결 *이후*만 크레딧해도 라벨이 같은가         (B)
  L2P 채점 파리티      백테스트 라벨 함수와 라이브/섀도우 채점기가 같은 합성 경로에 같은 답인가   (B')
  L3  피쳐 누수        단일피쳐 AUC≥0.95 / 모델 VAL AUC≥0.99 / 스태킹 OOF 출처 / (선택) 한 봉 밀기 /
                       조인 시점(결정 시각 행이 마감봉인가 -- 다음 봉이면 한 봉 미래참조)                (C)
  T1  너무 완전한 통과 대조군 전부 통과 + DSR≈1 또는 PBO≈0 → 재구성(L1/L2) PASS 전 진행 금지 플래그
  T2  수치 취약성      float32/64·컬럼순서만 바꿔 재학습해도 VAL/OOS 선별 평균이 유지되는가
                       (부호 반전=FAIL · 변형간 폭이 |추정치|의 50% 초과=FLAG -- 그 크기에선 못 믿는다)

사용:
  python scripts/gate_eth_entry_layers_20260903.py --pipeline tmp/<dir> [--gates L4,L1,L2,L2P,L3,T1,T2]
  python scripts/gate_eth_entry_layers_20260903.py --selftest        # 합성 raw/dedup 트리거로 L1 정답 재현

<dir>/gate_config.json 을 읽고 <dir>/layer_gates.json 을 쓴다. FAIL이 하나라도 있으면 exit 1.
**게이트를 통과하지 못했거나 돌리지 않은 층 위에 만든 결과는 전부 provisional 이다.**

gate_config.json (경로는 저장소 루트 기준):
  splits        {"VAL": "2025-09-01", "OOS": "2026-01-01", "HOLDOUT": "2026-04-01"}
  known_ts      {"assumption": "<왜 known_ts = 트리거 봉인지 한 줄>"}                       # raw 트리거
                또는 {"fires": "path.csv", "ts_col":..., "signal_col":..., "known_ts_col":...}  # 확정 지연 트리거
  label         {"fills": "path.csv", "ts_col": "timestamp", "y_col": "y", "entry_col": "lim",
                 "side_col": "sd", "atr_col": "atr_pct", "fill_idx_col": "fi", "exit_idx_col": "ei",
                 "bars_to_fill_col": "btf", "signal_col": "signal",
                 "exit": {"sl_atr": 3.0, "arm_atr": 1.0, "trail_atr": 0.1, "trail_anchor": "peak"|"entry"},
                 "cost_roundtrip": 0.001, "notional": 0.9, "tol_mean_bp": 2.0, "tol_winrate_pp": 2.0,
                 "atr_is_absolute": false}   # true면 atr_col이 절대 ATR(가격 단위), trail_anchor='entry' 전용(sim_exit 산술 그대로)
  trigger       {"module": "scripts/x.py"|"x", "fn": "build_fires", "kwargs": {}, "warmup_bars": 4000,
                 "sample_n": 200}      # fn(kl) -> DataFrame[timestamp, ..., known_ts]; 없으면 L1 SKIP
  features      {"cols": [...] | "cols_from_model_card": "model_card.json",
                 "stacked": [{"col": "x_pct", "source_col": "x_pct_oof_source"}], "frame": "bar_features.parquet"}
  scoring_parity{"backtest": {"module":..., "fn":..., "style": "from_fill_bar"|"post_fill"},
                 "live": {"adapter": "eth_limit_fade_shadow_v1"} | {"module":..., "fn":...},
                 "n_paths": 300, "horizon_bars": 24}
  controls      {"report": "controls.json", "dsr": 0.99, "pbo": 0.04, "extra_passed": [...]}
  selection     {"keep_frac": 0.2037, "labels": ["y", "@recon"]}     # @recon = L2가 쓴 1분 재구성 라벨
  hp, seed      T2/L3 재학습용 HGB 하이퍼파라미터와 시드

정답 검증(철회 v1, tmp/eth_entry_limit_fade_v1_20260903): L2는 09-03 전수조사와 같은 방향으로
FAIL해야 한다(전체 후보 PF 3.66→0.99). L2P는 v1 `trail_out`(체결봉부터) vs 섀도우 `manage()`(L3
규약)가 갈려 FAIL, 같은 함수를 post_fill 스타일로 걸면 PASS. 합성 cluster_dedup 트리거는 L1
FAIL, raw는 PASS.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import re
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

KL5 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
BAR = pd.Timedelta("5min")
ALL_GATES = ["L4", "L1", "L2", "L2P", "L3", "T1", "T2"]
DEFAULT_HP = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=60,
                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, n_iter_no_change=25)
# 모델 출력으로 보이는 피쳐명 -- 스태킹 선언이 없으면 수동 확인 대상으로 올린다
SUSPICIOUS = re.compile(
    r"proba|oof|metalabel|^pred$|_pred$|"
    r"^(taker_delta_z_climax|short_term_return_z|liquidity_sweep|orthogonal_combo|smt_divergence|"
    r"fib_extension_exhaustion|demarker_extreme|kalman_deviation_meanrev)_pct$")


def log(m): print(f"[gate] {m}", flush=True)


def _res(gate, status, note="", **metrics):
    return {"gate": gate, "status": status, "note": note, "metrics": metrics}


def _load_klines(path):
    kl = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")
    return kl.drop_duplicates("timestamp").reset_index(drop=True)


def _import(module, fn):
    """module: 'scripts/foo.py' 같은 경로 또는 'foo' 같은 모듈명."""
    if module.endswith(".py"):
        p = ROOT / module
        spec = importlib.util.spec_from_file_location(p.stem, p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(module)
    return getattr(mod, fn)


def _split(ts, splits):
    ts = pd.DatetimeIndex(ts)
    v, o, h = (pd.Timestamp(splits[k]) for k in ("VAL", "OOS", "HOLDOUT"))
    return np.where(ts < v, "TRAIN", np.where(ts < o, "VAL", np.where(ts < h, "OOS", "HOLDOUT")))


def _stats(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if not len(v):
        return dict(n=0, mean_bp=float("nan"), winrate=float("nan"), pf=float("nan"))
    w = v > 0; loss = -v[~w].sum()
    return dict(n=int(len(v)), mean_bp=round(float(v.mean() * 1e4), 3), winrate=round(float(w.mean()), 4),
                pf=round(float(v[w].sum() / loss), 3) if loss > 0 else float("inf"))


def _feature_cols(cfg):
    F = cfg["features"]
    if F.get("cols"):
        return list(F["cols"])
    return list(json.loads((ROOT / F["cols_from_model_card"]).read_text())["feature_cols"])


def _load_fills(cfg):
    L = cfg["label"]
    return pd.read_csv(ROOT / L["fills"], parse_dates=[L.get("ts_col", "timestamp")], low_memory=False)


def trail(side, e, a, hi, lo, cl, sl, arm, tr, anchor="peak"):
    """저장소 표준 트레일링 청산(불리쪽 스톱 → best → 무장 → 트레일). 반환: price move(수수료 전).

    anchor: 트레일 폭의 기준. "peak" = peak×(1∓tr·a) (진입모델 v1 `trail_out` 규약),
    "entry" = peak ∓ tr·a·e (= best ∓ trail×ATR_abs, V자반등 경제라벨 `sim_exit`·섀도우 러너 규약).
    두 규약은 스톱 레벨이 tr·a·(peak−e)만큼 달라 청산 봉이 바뀔 수 있다 -- 파이프라인 것을 선언해야 재현된다.
    ⭐2026-09-07: 새 스톱이 그 봉 **종가보다 유리한 쪽**이면 거래소가 거부하는 자리이므로(걸 수 없는 스톱)
    그 봉 종가에 즉시 청산한다. 이전 판은 그 자리에 걸린 것으로 치고 그 가격에 체결까지 시켜줬다 --
    docs/experiments/eth_trailing_stop_infeasible_fill_invalidates_exit_edge_20260907.md"""
    off = (lambda pk: pk * tr * a) if anchor == "peak" else (lambda pk: tr * a * e)
    if side > 0:
        stop = e * (1 - sl * a); peak = e; armed = False
        for k in range(len(cl)):
            if lo[k] <= stop:
                return stop / e - 1.0
            if hi[k] > peak:
                peak = hi[k]
                if not armed and (peak - e) / e >= arm * a:
                    armed = True
            if armed:
                ns = peak - off(peak)
                if ns > stop:
                    if ns > cl[k]:                          # 걸 수 없는 스톱 (2026-09-07)
                        return cl[k] / e - 1.0
                    stop = ns
        return cl[-1] / e - 1.0
    stop = e * (1 + sl * a); peak = e; armed = False
    for k in range(len(cl)):
        if hi[k] >= stop:
            return 1.0 - stop / e
        if lo[k] < peak:
            peak = lo[k]
            if not armed and (e - peak) / e >= arm * a:
                armed = True
        if armed:
            ns = peak + off(peak)
            if ns < stop:
                if ns < cl[k]:                              # 걸 수 없는 스톱 (2026-09-07)
                    return 1.0 - cl[k] / e
                stop = ns
    return 1.0 - cl[-1] / e


def trail_abs(side, e, A, hi, lo, cl, sl, arm, tr):
    """`trail(anchor="entry")`의 **절대 ATR** 판: V자반등 경제라벨 `sim_exit`의 스칼라 산술을 연산 순서까지 그대로 옮겼다
    (stop = e ∓ sl·A, 무장 = ±(best−e) ≥ arm·A, 트레일 = best ∓ tr·A). 상대 ATR을 곱해서 되돌리면 1e-13 오차가
    무장/스톱 동률에서 청산 봉을 바꿀 수 있어(2026-09-04 L2 잔차 0.24bp) 파이프라인이 절대 ATR을 주면 이 경로를 쓴다."""
    sign = 1.0 if side > 0 else -1.0
    stop = e - sign * sl * A; best = e; armed = False
    for k in range(len(cl)):
        adv = lo[k] if sign > 0 else hi[k]
        if (adv <= stop) if sign > 0 else (adv >= stop):
            return sign * (stop - e) / e
        fav = hi[k] if sign > 0 else lo[k]
        if sign * (fav - best) > 0:
            best = fav
        if not armed and sign * (best - e) >= arm * A:
            armed = True
        if armed:
            ns = best - sign * tr * A
            if sign * (ns - stop) > 0:
                if sign * (ns - cl[k]) > 0:                 # 걸 수 없는 스톱 (2026-09-07)
                    return sign * (cl[k] - e) / e
                stop = ns
    return sign * (cl[-1] - e) / e


# ----------------------------------------------------------------------------- L4 known_ts 계약
def gate_L4(cfg, kl, **_):
    L = cfg["label"]; D = _load_fills(cfg)
    ts = pd.DatetimeIndex(D[L.get("ts_col", "timestamp")])
    klts = pd.DatetimeIndex(kl["timestamp"])
    fi = D[L["fill_idx_col"]].to_numpy(int)
    if fi.min() < 0 or fi.max() >= len(kl):
        return _res("L4", "FAIL", "체결 인덱스가 klines 범위를 벗어난다")
    fill_ts = klts[fi]
    notes = []
    # ① 저장된 행 인덱스 ↔ 타임스탬프 정합 (BTC 108봉 오프셋 사고 부류)
    if L.get("bars_to_fill_col"):
        expect = ts + pd.to_timedelta(D[L["bars_to_fill_col"]].to_numpy(int) * 5, unit="m")
        n_bad = int((expect != fill_ts).sum())
        if n_bad:
            return _res("L4", "FAIL", f"fill_idx가 timestamp+bars_to_fill과 {n_bad}행 불일치 -- 인덱스가 다른 프레임 것",
                        n_index_mismatch=n_bad)
        notes.append("fill_idx↔timestamp 정합")
    # ② known_ts -- 파이프라인이 명시적으로 선언해야 한다
    K = cfg.get("known_ts")
    if not K:
        return _res("L4", "FAIL", "known_ts 미선언 -- known_ts.assumption(raw 트리거) 또는 known_ts.fires(확정지연)를 써라")
    if "fires" in K:
        Fr = pd.read_csv(ROOT / K["fires"], parse_dates=[K["ts_col"], K["known_ts_col"]])
        key = [K["ts_col"], K["signal_col"]]
        m = D[[L.get("ts_col", "timestamp"), L["signal_col"]]].rename(
            columns={L.get("ts_col", "timestamp"): K["ts_col"], L["signal_col"]: K["signal_col"]})
        j = m.merge(Fr[key + [K["known_ts_col"]]].drop_duplicates(key), on=key, how="left")
        if j[K["known_ts_col"]].isna().any():
            return _res("L4", "FAIL", f"발동표에 없는 체결 {int(j[K['known_ts_col']].isna().sum())}행")
        known = pd.DatetimeIndex(j[K["known_ts_col"]])
        if int((known < pd.DatetimeIndex(j[K["ts_col"]])).sum()):
            return _res("L4", "FAIL", "known_ts가 앵커 봉보다 앞선 발동이 있다")
        src = f"fires:{K['fires']}"
    else:
        known = ts; src = f"assumption: {K['assumption']}"
    # ③ 진입(=라벨 시작) 봉은 known 봉 *이후*여야 한다 (known 봉 마감에 주문 → 다음 봉부터 체결)
    n_early = int((fill_ts <= known).sum())
    status = "PASS" if n_early == 0 else "FAIL"
    return _res("L4", status, "; ".join(notes + [f"known_ts {src}"]),
                n=int(len(D)), n_entry_not_after_known=n_early,
                min_bars_known_to_fill=int(((fill_ts - known) / BAR).min()))


# ----------------------------------------------------------------------------- L1 발동 인과성
def _fire_keys(df):
    cols = [c for c in df.columns if c != "known_ts"]
    return set(map(tuple, df[cols].astype(str).to_numpy().tolist())), cols


def gate_L1(cfg, kl, seed=0, **_):
    T = cfg.get("trigger")
    if not T:
        return _res("L1", "SKIP", "trigger.module/fn 미선언 -- 다음 파이프라인은 build_fires(kl)->[timestamp, ..., known_ts]를 노출할 것")
    fn = _import(T["module"], T["fn"]); kw = T.get("kwargs", {})
    warm, n_sample = int(T.get("warmup_bars", 4000)), int(T.get("sample_n", 200))
    full = fn(kl, **kw)
    if "known_ts" not in full.columns or "timestamp" not in full.columns:
        return _res("L1", "FAIL", "build_fires는 timestamp와 known_ts 열을 반환해야 한다")
    if int((pd.DatetimeIndex(full["known_ts"]) < pd.DatetimeIndex(full["timestamp"])).sum()):
        return _res("L1", "FAIL", "known_ts < timestamp 인 발동 존재")
    key_full, key_cols = _fire_keys(full)
    klts = kl["timestamp"].to_numpy()
    n_sample = min(n_sample, len(full))
    pick = full.sample(n_sample, random_state=int(seed)).reset_index(drop=True)
    n_missing = n_phantom = 0; ex_missing = []; ex_phantom = []; n_cuts = 0
    for r in pick.itertuples(index=False):
        k_ts = np.datetime64(pd.Timestamp(r.known_ts))
        cut0 = int(np.searchsorted(klts, k_ts, side="right"))            # known_ts 봉까지 포함
        me = tuple(str(getattr(r, c)) for c in key_cols)
        # (a) 자기 known_ts에서 잘라도 이 발동이 나와야 한다 -- 확인봉이 더 필요하면 known_ts가 거짓
        # (b) known_ts 직전(1·3봉 전)에서 잘랐을 때 나온 발동 중 전체집합에 없는 것 = 미래가 오면 바뀌는 발동
        for back in (0, 1, 3):
            cut = cut0 - back
            if cut <= warm // 2:
                continue
            sub = kl.iloc[max(0, cut - warm):cut].reset_index(drop=True)
            part = fn(sub, **kw); n_cuts += 1
            keys, _ = _fire_keys(part)
            if back == 0:
                if me not in keys:
                    n_missing += 1
                    if len(ex_missing) < 5: ex_missing.append(me)
            else:
                recent = part[pd.DatetimeIndex(part["timestamp"]) >= sub["timestamp"].iloc[max(0, len(sub) - 288)]]
                rk, _ = _fire_keys(recent)
                ph = rk - key_full
                n_phantom += len(ph)
                for x in list(ph)[: max(0, 5 - len(ex_phantom))]: ex_phantom.append(x)
    status = "PASS" if (n_missing == 0 and n_phantom == 0) else "FAIL"
    note = (f"표본 {n_sample}발동·절단 {n_cuts}회 · 자기 known_ts에서 사라진 발동 {n_missing} · "
            f"직전 절단에서만 존재하는 유령 발동 {n_phantom}")
    return _res("L1", status, note, n_fires_full=int(len(full)), n_sample=n_sample, n_cuts=n_cuts,
                n_missing_at_known_ts=n_missing, n_phantom_before_known_ts=n_phantom,
                examples_missing=ex_missing, examples_phantom=ex_phantom)


# ----------------------------------------------------------------------------- L2 라벨 1분 재구성
def gate_L2(cfg, kl, kl1=None, out_dir=None, **_):
    L = cfg["label"]; ex = L["exit"]
    sl, arm, tr = float(ex["sl_atr"]), float(ex["arm_atr"]), float(ex["trail_atr"])
    anchor = str(ex.get("trail_anchor", "peak")); atr_abs = bool(L.get("atr_is_absolute", False))
    if atr_abs and anchor != "entry":
        return _res("L2", "FAIL", "atr_is_absolute=true 는 trail_anchor='entry'(sim_exit 규약)에서만 지원")
    T = (lambda sd_, e_, a_, H_, L_, C_: trail_abs(sd_, e_, a_, H_, L_, C_, sl, arm, tr)) if atr_abs else \
        (lambda sd_, e_, a_, H_, L_, C_: trail(sd_, e_, a_, H_, L_, C_, sl, arm, tr, anchor))
    cost, N = float(L["cost_roundtrip"]), float(L["notional"])
    tol_mean, tol_win = float(L.get("tol_mean_bp", 2.0)), float(L.get("tol_winrate_pp", 2.0))
    D = _load_fills(cfg)
    if L.get("row_filter"):
        D = D.query(L["row_filter"])
    rows = D.index.to_numpy(); D = D.reset_index(drop=True)
    h5, l5, c5 = (kl[x].to_numpy(float) for x in ("high", "low", "close"))
    ts5 = pd.DatetimeIndex(kl["timestamp"])
    if kl1 is None:
        kl1 = _load_klines(KL1)
    b5 = kl1["timestamp"].dt.floor("5min")
    grp = kl1.groupby(b5).indices                                   # 5분봉 시작시각 -> 1분봉 위치들
    m1h, m1l = kl1["high"].to_numpy(float), kl1["low"].to_numpy(float)
    fi = D[L["fill_idx_col"]].to_numpy(int); ei = D[L["exit_idx_col"]].to_numpy(int)
    e = D[L["entry_col"]].to_numpy(float); a = D[L["atr_col"]].to_numpy(float)
    sd = D[L["side_col"]].to_numpy(int); y = D[L["y_col"]].to_numpy(float)
    n = len(D)
    y_rep_a = np.full(n, np.nan); y_rep_b = np.full(n, np.nan)
    y_rec = np.full(n, np.nan); fav_atr = np.full(n, np.nan); fill_min = np.full(n, np.nan)
    for i in range(n):
        f, hz = int(fi[i]), int(ei[i] - fi[i])
        if hz <= 0 or f + hz > len(kl):
            continue
        mv = T(sd[i], e[i], a[i], h5[f:f + hz], l5[f:f + hz], c5[f:f + hz])
        y_rep_a[i] = (mv - cost) * N; y_rep_b[i] = mv * N - cost
        a_rel = a[i] / e[i] if atr_abs else a[i]                     # 체결봉 유리폭 통계는 상대 ATR로
        fav = (h5[f] - e[i]) / e[i] if sd[i] > 0 else (e[i] - l5[f]) / e[i]
        fav_atr[i] = fav / a_rel if a_rel > 0 else np.nan
        idx = grp.get(ts5[f])
        if idx is None:
            continue
        sh, slo = m1h[idx], m1l[idx]
        hit = np.flatnonzero(slo <= e[i]) if sd[i] > 0 else np.flatnonzero(sh >= e[i])
        if not len(hit):
            continue
        k0 = int(hit[0]); fill_min[i] = k0
        ph, pl = sh[k0 + 1:], slo[k0 + 1:]                          # 체결 분 *다음* 분부터가 우리 것
        fh, fl = (float(ph.max()), float(pl.min())) if len(ph) else (e[i], e[i])
        H = np.concatenate([[fh], h5[f + 1:f + hz]]); Lw = np.concatenate([[fl], l5[f + 1:f + hz]])
        C = np.concatenate([[c5[f]], c5[f + 1:f + hz]])
        y_rec[i] = (T(sd[i], e[i], a[i], H, Lw, C) - cost) * N
    # ① 라벨 재현 -- 게이트가 파이프라인 라벨(청산·비용·명목)을 정확히 이해하는가
    ok_rep = np.isfinite(y_rep_a) & np.isfinite(y)
    err_a = float(np.max(np.abs(y_rep_a[ok_rep] - y[ok_rep]))); err_b = float(np.max(np.abs(y_rep_b[ok_rep] - y[ok_rep])))
    if min(err_a, err_b) > 1e-9:
        return _res("L2", "FAIL", f"라벨 재현 실패 max|Δ| {min(err_a, err_b)*1e4:.3f}bp -- exit/cost/notional 설정이 파이프라인과 다르다",
                    reproduction_err_bp=min(err_a, err_b) * 1e4)
    formula = "(move-cost)*notional" if err_a <= err_b else "move*notional-cost"
    if formula == "move*notional-cost":                             # 재구성도 같은 공식으로
        y_rec = np.where(np.isfinite(y_rec), (y_rec / N + cost) * N - cost, np.nan)
    # ② 1분 재구성 대조 (1분봉이 덮는 행만, 같은 행끼리)
    ok = np.isfinite(y_rec) & np.isfinite(y)
    s0, s1 = _stats(y[ok]), _stats(y_rec[ok])
    d_mean = s1["mean_bp"] - s0["mean_bp"]; d_win = (s1["winrate"] - s0["winrate"]) * 100
    status = "PASS" if (abs(d_mean) <= tol_mean and abs(d_win) <= tol_win) else "FAIL"
    fa = fav_atr[np.isfinite(fav_atr)]
    if out_dir is not None:
        pd.DataFrame({"row": rows, "y": y, "y_recon": y_rec, "fill_minute": fill_min, "fill_bar_fav_atr": fav_atr}
                     ).to_csv(Path(out_dir) / "label_reconstruction_1m.csv", index=False)
    note = (f"라벨공식 {formula} 재현 OK · 1분 커버 {ok.mean()*100:.1f}% · 평균 {s0['mean_bp']:+.2f}→{s1['mean_bp']:+.2f}bp "
            f"(Δ{d_mean:+.2f}) · 승률 {s0['winrate']*100:.1f}→{s1['winrate']*100:.1f}% · PF {s0['pf']}→{s1['pf']} · "
            f"체결봉 유리폭 중앙 {np.median(fa):.2f}ATR, ARM({arm}) 초과 {(fa >= arm).mean()*100:.1f}%")
    return _res("L2", status, note, n_total=n, n_covered=int(ok.sum()), label_formula=formula,
                pipeline=s0, reconstructed_1m=s1, delta_mean_bp=round(d_mean, 3), delta_winrate_pp=round(d_win, 3),
                fill_bar_fav_excursion_median_atr=round(float(np.median(fa)), 3),
                fill_bar_fav_excursion_frac_over_arm=round(float((fa >= arm).mean()), 4),
                tol_mean_bp=tol_mean, tol_winrate_pp=tol_win)


# ----------------------------------------------------------------------------- L2P 채점 파리티
def _live_eth_limit_fade_shadow_v1(side, e, a, H, L, C, post_hl, ex):
    """섀도우 러너 `manage()`를 합성 경로 하나에 대해 돌려 raw move를 돌려준다(비용 0·명목 1)."""
    mod = importlib.import_module("live_eth_entry_limit_fade_shadow_runner_20260903")
    mod.log = lambda m: None
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    p = {"sd": int(side), "atr_abs": a * e, "atr_pct": a, "entry": e, "limit": e,
         "stop": e - side * ex["sl_atr"] * a * e, "best": e, "armed": False, "bars_held": 0,
         "horizon": len(C) - 1, "cost_roundtrip": 0.0, "notional": 1.0,
         "entry_bar_utc": base.isoformat(), "placed_bar_utc": (base - timedelta(minutes=5)).isoformat(),
         "signal": "synthetic", "side": "bottom" if side > 0 else "top", "arm": 1,
         "post_fill_high": float(post_hl[0]), "post_fill_low": float(post_hl[1])}
    s = {"positions": [p], "ledger": [], "skipped": 0}
    bars = [{"timestamp_utc": (base + timedelta(minutes=5 * k)).isoformat(),
             "open": float(C[k]), "high": float(H[k]), "low": float(L[k]), "close": float(C[k])}
            for k in range(1, len(C))]
    mod.manage(s, bars, {"exit": {"sl_atr": ex["sl_atr"], "arm_atr": ex["arm_atr"], "trail_atr": ex["trail_atr"]}})
    if not s["ledger"]:
        raise RuntimeError("라이브 채점기가 포지션을 닫지 않았다")
    return s["ledger"][-1]["pnl_bp"] / 1e4


def _live_eth_v_rebound_econ_shadow(side, e, a, H, L, C, post_hl, ex):
    """경제라벨 섀도우 러너 `manage()`를 합성 경로 하나에 돌려 raw move를 돌려준다(비용 되돌림, 명목 1).
    시장가(다음 봉 시가) 진입이라 체결 봉 전체가 우리 것 -> 백테스트 style은 from_fill_bar 여야 한다.
    창 끝까지 안 닫히면 sim_exit과 같이 마지막 종가 청산으로 친다."""
    mod = importlib.import_module("live_eth_v_rebound_econ_shadow_runner_20260902")
    mod.log = lambda m: None
    mod.BRACKET = {"sl_atr": float(ex["sl_atr"]), "arm_atr": float(ex["arm_atr"]), "trail_atr": float(ex["trail_atr"])}
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    atr_abs = float(a) * float(e); sgn = 1.0 if side > 0 else -1.0
    p = {"entry_utc": base.isoformat(), "side": "long" if side > 0 else "short", "entry": float(e), "atr": atr_abs,
         "stop": float(e) - sgn * float(ex["sl_atr"]) * atr_abs, "best": float(e), "armed": False, "bars_held": 0,
         "last_bar_utc": None, "proba": 1.0, "opened_utc": base.isoformat()}
    s = {"positions": [p], "ledger": [], "consec_loss": 0}
    bars = [{"timestamp_utc": (base + timedelta(minutes=5 * k)).isoformat(),
             "open": float(C[k]), "high": float(H[k]), "low": float(L[k]), "close": float(C[k])} for k in range(len(C))]
    mod.manage(s, bars)
    if s["ledger"]:
        return (s["ledger"][-1]["pnl_bp"] + float(mod.COST_BP)) / 1e4
    return sgn * (float(C[-1]) - float(e)) / float(e)


LIVE_ADAPTERS = {"eth_limit_fade_shadow_v1": _live_eth_limit_fade_shadow_v1,
                 "eth_v_rebound_econ_shadow": _live_eth_v_rebound_econ_shadow}


def gate_L2P(cfg, seed=0, **_):
    P = cfg.get("scoring_parity")
    if not P:
        return _res("L2P", "SKIP", "scoring_parity 미선언 -- 라이브/섀도우 채점기가 생기면 반드시 건다")
    bt = _import(P["backtest"]["module"], P["backtest"]["fn"]); style = P["backtest"].get("style", "post_fill")
    lv = P["live"]
    live = LIVE_ADAPTERS[lv["adapter"]] if "adapter" in lv else _import(lv["module"], lv["fn"])
    ex = cfg["label"]["exit"]; n_paths = int(P.get("n_paths", 300)); hz = int(P.get("horizon_bars", 24))
    r = np.random.default_rng(int(seed)); diffs = []
    for _ in range(n_paths):
        side = 1 if r.random() < 0.5 else -1
        e = 100.0 * (1 + r.normal(0, 0.05)); a = float(r.uniform(0.002, 0.01))
        # 체결 봉: 체결 *이전* 유리한 극단(우리 것 아님)이 크고, 체결 이후 폭은 작다 -- B형 사고의 기하
        pre_fav = r.uniform(0.5, 3.0) * a * e; post_fav = r.uniform(0.0, 0.3) * a * e; post_adv = r.uniform(0.0, 0.5) * a * e
        if side > 0:
            H0, L0, ph, pl = e + max(pre_fav, post_fav), e - post_adv, e + post_fav, e - post_adv
        else:
            H0, L0, ph, pl = e + post_adv, e - max(pre_fav, post_fav), e + post_adv, e - post_fav
        mid = e + r.normal(0, a * e * 0.6, hz - 1).cumsum(); rg = np.abs(r.normal(0, a * e * 0.5, hz - 1))
        H = np.concatenate([[H0], mid + rg]); Lw = np.concatenate([[L0], mid - rg]); C = np.concatenate([[e], mid])
        if style == "from_fill_bar":
            mv_bt = bt(side, e, a, H, Lw, C)
        else:
            mv_bt = bt(side, e, a, np.concatenate([[ph], H[1:]]), np.concatenate([[pl], Lw[1:]]), C)
        diffs.append(float(mv_bt) - float(live(side, e, a, H, Lw, C, (ph, pl), ex)))
    d = np.abs(np.asarray(diffs)); tol = 1e-6                       # 섀도우 원장이 bp 소수 3자리로 반올림
    status = "PASS" if d.max() <= tol else "FAIL"
    return _res("L2P", status,
                f"백테스트({style}) vs 라이브 · 합성 {n_paths}경로 · 불일치 {(d > tol).mean()*100:.1f}% · "
                f"max|Δ| {d.max()*1e4:.2f}bp · 평균(백테스트−라이브) {np.mean(diffs)*1e4:+.2f}bp",
                n_paths=n_paths, backtest_style=style, frac_disagree=round(float((d > tol).mean()), 4),
                max_abs_diff_bp=round(float(d.max() * 1e4), 4), mean_diff_bp=round(float(np.mean(diffs) * 1e4), 4))


# ----------------------------------------------------------------------------- L3 피쳐 누수
def _shift_tripwire(cfg, D, cols, sp, y, seed):
    """피쳐를 한 봉 과거 것으로 바꿔 재학습. 밀었는데 VAL·OOS 둘 다 안 나빠지면 신선함의 이득이 노이즈."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    F = cfg["features"]; L = cfg["label"]
    fp = ROOT / F["frame"]
    fr = pd.read_parquet(fp) if fp.suffix == ".parquet" else pd.read_csv(fp, parse_dates=["timestamp"])
    use = [c for c in cols if c in fr.columns]
    ts = pd.DatetimeIndex(D[L.get("ts_col", "timestamp")])
    fr = fr.drop_duplicates("timestamp").set_index("timestamp")
    Xs = fr.reindex(ts - BAR)[use].to_numpy(float)
    X0 = D[use].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    keep = float(cfg["selection"]["keep_frac"]); hp = cfg.get("hp", DEFAULT_HP); tr = sp == "TRAIN"
    out = {}
    for tag, X in (("original", X0), ("shifted_1bar", Xs)):
        m = HistGradientBoostingRegressor(**hp, random_state=int(seed)).fit(X[tr], y[tr])
        pred = m.predict(X); thr = np.quantile(pred[tr], 1 - keep)
        out[tag] = {w: _stats(y[(sp == w) & (pred > thr)])["mean_bp"] for w in ("VAL", "OOS")}
    worse = sum(out["shifted_1bar"][w] < out["original"][w] for w in ("VAL", "OOS"))
    return {"status": "FLAG" if worse == 0 else "PASS", "n_frame_cols": len(use), **out,
            "note": "한 봉 밀어도 VAL·OOS 둘 다 안 나빠짐 -- 신선함 이득이 노이즈" if worse == 0 else "밀면 나빠짐(정상)"}


def _join_timing_check(D, ts_col, kl):
    """L3-⑤ 조인 시점: 결정 시각 행의 수익률 열이 **방금 마감한 봉**의 수익률인가, **다음 봉**(미래)의 것인가.
    이 저장소의 피쳐 프레임은 행 τ가 봉 τ 자신의 종가로 계산되므로, 결정 시각 τ에 '행 τ+1'을 조인하면 한 봉 미래참조다
    (2026-09-01 exit 스모크가 정확히 이렇게 조인해 AUC가 +0.10 부풀었다, 2026-09-04 실측)."""
    col = next((c for c in ("log_return", "ret_1") if c in D.columns), None)
    if col is None or kl is None:
        return {"status": "SKIP", "note": "수익률 열(log_return/ret_1) 또는 klines 없음"}
    k = kl[["timestamp", "close"]].copy()
    k["own"] = np.log(k["close"] / k["close"].shift(1)); k["nxt"] = k["own"].shift(-1)
    s = D[[ts_col, col]].dropna().sample(min(20000, len(D)), random_state=0)
    m = s.merge(k[["timestamp", "own", "nxt"]], left_on=ts_col, right_on="timestamp", how="inner").dropna()
    if len(m) < 500:
        return {"status": "SKIP", "note": f"대조 표본 부족 {len(m)}"}
    v = pd.to_numeric(m[col], errors="coerce").to_numpy(); ok = np.isfinite(v)
    r_own = float(np.corrcoef(v[ok], m["own"].to_numpy()[ok])[0, 1]); r_nxt = float(np.corrcoef(v[ok], m["nxt"].to_numpy()[ok])[0, 1])
    bad = r_nxt > r_own
    return {"status": "FAIL" if bad else "PASS", "col": col, "corr_closed_bar": round(r_own, 4), "corr_next_bar": round(r_nxt, 4),
            "note": "❌맥락이 다음 봉(미래)에서 조인됨" if bad else "마감봉 조인 확인"}


def gate_L3(cfg, seed=0, kl=None, **_):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    F = cfg["features"]; L = cfg["label"]; cols = _feature_cols(cfg)
    # 피쳐 표가 체결표와 다른 해상도(예: 청산 체크포인트)면 features.table/ts_col/y_col로 따로 준다
    if F.get("table"):
        tp = ROOT / F["table"]
        D = pd.read_parquet(tp) if tp.suffix == ".parquet" else pd.read_csv(tp, low_memory=False)
        ts_col, y_col = F.get("ts_col", "timestamp"), F.get("y_col", L["y_col"])
        D[ts_col] = pd.to_datetime(D[ts_col])
    else:
        D = _load_fills(cfg); ts_col, y_col = L.get("ts_col", "timestamp"), L["y_col"]
    missing = [c for c in cols if c not in D.columns]
    if missing:
        return _res("L3", "FAIL", f"피쳐 열 없음 {missing[:8]}")
    sp = _split(D[ts_col], cfg["splits"])
    y = D[y_col].to_numpy(float); yb = (y > 0).astype(int)
    tr, va = sp == "TRAIN", sp == "VAL"
    X = D[cols].apply(pd.to_numeric, errors="coerce")
    # ① 단일 피쳐 AUC (TRAIN) -- 라벨 구성요소가 피쳐로 새면 여기서 0.95+
    single = {}
    ytr = yb[tr]
    for c in cols:
        v = X[c].to_numpy(float)[tr]; m = np.isfinite(v)
        if m.sum() < 100 or ytr[m].min() == ytr[m].max():
            continue
        auc = roc_auc_score(ytr[m], v[m]); single[c] = max(auc, 1 - auc)
    leak_single = {c: round(v, 4) for c, v in single.items() if v >= 0.95}
    top5 = dict(sorted(single.items(), key=lambda kv: -kv[1])[:5])
    # ② 모델 VAL AUC -- 완벽 분리는 통과가 아니라 누수
    auc_val = float("nan")
    if va.sum() > 50 and yb[va].min() != yb[va].max():
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=int(seed)).fit(X[tr], ytr)
        auc_val = float(roc_auc_score(yb[va], clf.predict_proba(X[va])[:, 1]))
    # ③ 스태킹 OOF 출처 -- 선언된 것은 TRAIN 행이 전부 fold 밖이어야 한다
    stacked = F.get("stacked", []); oof_bad = {}
    for s in stacked:
        src = D[s["source_col"]].fillna("").astype(str)
        # 위반 = TRAIN 행의 값이 'final'(TRAIN 전체를 본 모델) 출처. 비활성('')은 해당 없음
        n_bad = int(src[tr].str.startswith("final").sum())
        if n_bad:
            oof_bad[s["col"]] = n_bad
    suspicious = [c for c in cols if SUSPICIOUS.search(c)] if not stacked else []
    # ④ 한 봉 밀기 (봉 인덱스 피쳐 프레임이 있을 때만)
    shift = _shift_tripwire(cfg, D, cols, sp, y, seed) if F.get("frame") else \
        {"status": "SKIP", "note": "features.frame(봉 인덱스 피쳐 프레임) 미제공 -- 다음 파이프라인은 저장할 것"}
    # ⑤ 조인 시점 (결정 시각 행 = 마감봉인가)
    jt = _join_timing_check(D, ts_col, kl) if F.get("table") else {"status": "SKIP", "note": "체결표 기반(체크포인트 표 아님)"}
    fail = bool(leak_single) or (np.isfinite(auc_val) and auc_val >= 0.99) or bool(oof_bad) or jt["status"] == "FAIL"
    status = "FAIL" if fail else "PASS"
    note = (f"단일피쳐 AUC≥0.95: {leak_single or '없음'} (최고 {list(top5.items())[0] if top5 else '-'}) · 모델 VAL AUC {auc_val:.4f} · "
            f"스태킹 선언 {len(stacked)}개 OOF 위반 {oof_bad or '없음'}"
            + (f" · ⚠️수동확인(모델 출력으로 보이는 이름) {suspicious}" if suspicious else "")
            + f" · 한봉밀기 {shift['status']} · 조인시점 {jt['status']}"
            + (f"(마감봉 r={jt['corr_closed_bar']} vs 다음봉 r={jt['corr_next_bar']})" if "corr_closed_bar" in jt else ""))
    return _res("L3", status, note, n_features=len(cols), single_feature_auc_top5=top5, single_feature_leaks=leak_single,
                model_val_auc=round(auc_val, 4) if np.isfinite(auc_val) else None, stacked_declared=len(stacked),
                oof_violations=oof_bad, suspicious_names=suspicious, shift_tripwire=shift, join_timing=jt)


# ----------------------------------------------------------------------------- T1 너무 완전한 통과
def gate_T1(cfg, **_):
    Cc = cfg.get("controls")
    if not Cc:
        return _res("T1", "SKIP", "controls 미선언")
    rep = json.loads((ROOT / Cc["report"]).read_text()); base = rep.get("base", {})
    results = {}
    for name, nulls in rep.items():
        if name == "base" or not isinstance(nulls, dict):
            continue
        for w, arr in nulls.items():
            if w in base and isinstance(arr, list) and arr:
                results[f"{name}/{w}"] = bool(base[w] > np.percentile(np.asarray(arr, float), 95))
    extra = list(Cc.get("extra_passed", []))
    n_pass, n = sum(results.values()) + len(extra), len(results) + len(extra)
    dsr, pbo = Cc.get("dsr"), Cc.get("pbo")
    all_pass = n > 0 and n_pass == n
    extreme = (dsr is not None and dsr >= 0.99) or (pbo is not None and pbo <= 0.05)
    status = "FLAG" if (all_pass and extreme) else "PASS"
    note = (f"대조군 {n_pass}/{n} 통과 · DSR {dsr} · PBO {pbo}"
            + (" → ⚠️전부 통과 + DSR/PBO 극단: 대조군은 같은 기질 위에서 채점된다. L1/L2 재구성 PASS 전 진행 금지"
               if status == "FLAG" else ""))
    return _res("T1", status, note, controls=results, extra_passed=extra, dsr=dsr, pbo=pbo)


# ----------------------------------------------------------------------------- T2 수치 취약성
def gate_T2(cfg, seed=0, out_dir=None, **_):
    from sklearn.ensemble import HistGradientBoostingRegressor
    L = cfg["label"]; S = cfg["selection"]; D = _load_fills(cfg); cols = _feature_cols(cfg)
    sp = _split(D[L.get("ts_col", "timestamp")], cfg["splits"])
    keep = float(S["keep_frac"]); hp = cfg.get("hp", DEFAULT_HP)
    X64 = D[cols].apply(pd.to_numeric, errors="coerce").astype("float64")
    rng = np.random.default_rng(int(seed)); shuf = list(rng.permutation(cols))
    variants = {"float64_df": X64, "float32_arr": X64.to_numpy(np.float32), "float64_shuffled_cols": X64[shuf]}
    labels = {}
    for tag in S.get("labels", ["y"]):
        if tag == "@recon":
            fp = Path(out_dir) / "label_reconstruction_1m.csv" if out_dir else None
            if not fp or not fp.exists():
                continue
            R = pd.read_csv(fp); yr = np.full(len(D), np.nan); yr[R["row"].to_numpy(int)] = R["y_recon"].to_numpy(float)
            labels["y_recon(1m)"] = yr
        else:
            labels[tag] = D[tag].to_numpy(float)
    report = {}; any_flip = False; any_wide = False
    for lname, y in labels.items():
        ok = np.isfinite(y); tr = (sp == "TRAIN") & ok
        per = {}
        for vname, X in variants.items():
            Xtr = X[tr] if isinstance(X, np.ndarray) else X[tr]
            m = HistGradientBoostingRegressor(**hp, random_state=int(seed)).fit(Xtr, y[tr])
            pred = m.predict(X); thr = np.quantile(pred[tr], 1 - keep)
            per[vname] = {w: _stats(y[(sp == w) & ok & (pred > thr)]) for w in ("VAL", "OOS", "HOLDOUT")}
        flips = {}
        for w in ("VAL", "OOS"):
            vals = [per[v][w]["mean_bp"] for v in variants]
            spread = float(np.nanmax(vals) - np.nanmin(vals)); med = float(np.nanmedian(vals))
            # 세 변형은 '같은 모델'이다 -- 차이는 전부 수치 노이즈. 그 노이즈가 추정치 자체만큼 크면
            # 부호가 안 뒤집혀도 그 크기에서는 믿을 수 없다 (09-03 실측: 정직 라벨 VAL +0.8/+4.7/+2.7)
            flips[w] = {"values_bp": vals, "sign_stable": bool(all(np.sign(vals) == np.sign(vals[0])) and all(np.isfinite(vals))),
                        "spread_bp": round(spread, 3), "spread_over_abs_median": round(spread / abs(med), 3) if med else float("inf"),
                        "holdout_diag_bp": [per[v]["HOLDOUT"]["mean_bp"] for v in variants]}
        any_flip |= not all(f["sign_stable"] for f in flips.values())
        any_wide |= any(f["spread_over_abs_median"] > 0.5 for f in flips.values())
        report[lname] = {"variants": per, "sign_check": flips}
    status = "FAIL" if any_flip else ("FLAG" if any_wide else "PASS")
    parts = []
    for lname, r in report.items():
        parts.append(lname + " " + " ".join(
            f"{w}[{'/'.join(f'{v:+.1f}' for v in r['sign_check'][w]['values_bp'])}]"
            + ("❌부호반전" if not r["sign_check"][w]["sign_stable"] else
               ("⚠️변형간 폭이 추정치의 %.0f%%" % (100 * r["sign_check"][w]["spread_over_abs_median"])
                if r["sign_check"][w]["spread_over_abs_median"] > 0.5 else "✅"))
            for w in ("VAL", "OOS")))
    return _res("T2", status, "dtype·컬럼순서 3변형 선별평균(bp): " + " | ".join(parts), keep_frac=keep, labels=report)


GATES = {"L4": gate_L4, "L1": gate_L1, "L2": gate_L2, "L2P": gate_L2P, "L3": gate_L3, "T1": gate_T1, "T2": gate_T2}


# ----------------------------------------------------------------------------- 러너 / 셀프테스트
def run_pipeline(pdir: Path, gates: list[str], seed: int | None) -> int:
    cfg = json.loads((pdir / "gate_config.json").read_text())
    seed = int(cfg.get("seed", 0) if seed is None else seed)
    log(f"파이프라인 {pdir.relative_to(ROOT)} · 게이트 {gates} · seed {seed}")
    kl = _load_klines(KL5)
    kl1 = _load_klines(KL1) if "L2" in gates else None
    results = []
    for g in gates:
        t0 = time.time()
        try:
            r = GATES[g](cfg, kl=kl, kl1=kl1, out_dir=pdir, seed=seed)
        except Exception as ex:                                     # noqa: BLE001 -- 게이트 자체 오류는 FAIL로 기록
            r = _res(g, "FAIL", f"게이트 실행 오류: {type(ex).__name__}: {ex}")
        r["seconds"] = round(time.time() - t0, 1); results.append(r)
        log(f"  {g:3s} {r['status']:5s} ({r['seconds']}s) {r['note']}")
    any_fail = any(r["status"] == "FAIL" for r in results); any_flag = any(r["status"] == "FLAG" for r in results)
    out = {"pipeline": str(pdir.relative_to(ROOT)), "run_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
           "gates": results, "any_fail": any_fail, "any_flag": any_flag,
           "verdict": "BLOCKED -- FAIL 층 위에 쌓지 말 것" if any_fail else
                      ("FLAG -- 재구성 게이트가 PASS했으면 진행 가능" if any_flag else "PASS")}
    (pdir / "layer_gates.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    log(f"→ {out['verdict']} · layer_gates.json 기록")
    return 1 if any_fail else 0


def _demo_zscore_fires(kl, dedup=False, z=-2.5, win=48, gap=12):
    """셀프테스트용 합성 트리거. dedup=True는 cluster_dedup(군집 최극단=앵커) -- A형 사고 그대로."""
    r = np.log(kl["close"].astype(float)).diff()
    zs = ((r - r.rolling(win).mean()) / r.rolling(win).std()).to_numpy()
    idx = np.flatnonzero(zs < z)
    if not dedup:
        ts = kl["timestamp"].to_numpy()[idx]
        return pd.DataFrame({"timestamp": ts, "signal": "demo_raw", "known_ts": ts})
    anchors, cur = [], []
    for j in idx:
        if cur and j - cur[-1] > gap:
            anchors.append(min(cur, key=lambda k: zs[k])); cur = []
        cur.append(j)
    if cur:
        anchors.append(min(cur, key=lambda k: zs[k]))
    ts = kl["timestamp"].to_numpy()[anchors]
    return pd.DataFrame({"timestamp": ts, "signal": "demo_dedup", "known_ts": ts})   # known_ts를 앵커 봉으로 '거짓' 선언


def selftest() -> int:
    kl = _load_klines(KL5).iloc[-60000:].reset_index(drop=True)      # 최근 ~7개월이면 충분
    ok = True
    for dedup, expect in ((False, "PASS"), (True, "FAIL")):
        cfg = {"trigger": {"module": "gate_eth_entry_layers_20260903", "fn": "_demo_zscore_fires",
                           "kwargs": {"dedup": dedup}, "warmup_bars": 1000, "sample_n": 80}}
        r = gate_L1(cfg, kl, seed=7)
        good = r["status"] == expect; ok &= good
        log(f"[selftest] L1 {'dedup' if dedup else 'raw  '} → {r['status']} (기대 {expect}) {'✅' if good else '❌'} · {r['note']}")
    log(f"[selftest] {'전부 통과' if ok else '실패'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", type=str, help="gate_config.json이 있는 디렉토리 (저장소 루트 기준)")
    ap.add_argument("--gates", type=str, default=",".join(ALL_GATES))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.pipeline:
        ap.error("--pipeline 또는 --selftest")
    gates = [g.strip() for g in a.gates.split(",") if g.strip()]
    bad = [g for g in gates if g not in GATES]
    if bad:
        ap.error(f"알 수 없는 게이트 {bad}")
    return run_pipeline(ROOT / a.pipeline, gates, a.seed)


if __name__ == "__main__":
    raise SystemExit(main())
