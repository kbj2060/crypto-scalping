#!/usr/bin/env python3
"""BTC 증거신호 7종 **라이브 메타라벨 스코어러** -- 대시보드 칩용 (판단 산출 전용).

## 이게 무엇인가

ETH의 `live_evidence_signal_metalabel_20260829.py`에 대응하는 BTC판. 2026-09-01에 BTC 7종의
그리드스크린 + TabPFN 메타라벨 검증이 끝났는데 **서빙 아티팩트가 없어** 라이브가 없었다.
2026-09-02에 동결 컨텍스트를 만들고(`build_btc_evidence_signal_frozen_contexts_20260902.py`)
이 스코어러를 붙인다.

## ⚠️ETH와 신호 정의는 같고, 라벨 파라미터는 BTC 전용이다

원시 신호 계산은 `compute_signals()`를 **ETH와 공유**한다(BTC 후보 빌드가 verbatim 재사용).
자산별로 다른 건 **메타라벨의 HIT정의/H/K/GAP**이고, 2026-09-01 그리드스크린이 BTC에서
독자 재선정했다:

    신호                       HIT정의        H     K      GAP    TRAIN hit률
    taker_delta_climax        종가기준       6     2.0     3      0.1388
    liquidity_sweep           터치+되돌림   20     2.0     6      0.1022
    kalman_deviation_meanrev  touch MFE     10     3.5     6      0.1425
    short_term_return_z       MAE 상한       6     2.0    12      0.3163
    orthogonal_combo          touch MFE      8     2.0     6      0.4271
    demarker_extreme          touch MFE      8     0.70    6      0.9003
    fib_extension_exhaustion  종가기준      10     2.75    6      0.1928

⚠️`demarker_extreme`의 hit률 0.9003은 K=0.70이 낮아 거의 다 맞는다는 뜻 -- 라이브 변별력이
낮으므로 표시할 때 감안할 것.

## ⚠️매매 신호가 아니다

ETH 증거신호와 같은 지위 -- **사람의 재량 판단을 위한 확률 이동 컨텍스트**다. 주문을 내지 않고
자동매매에 배선되지 않는다. BTC는 경제성 게이트를 통과한 모델이 아직 없다(2026-09-02
경제라벨 시도가 VAL에서 실패: 손익비 하한을 걸면 수익 조합이 없음).

Usage:
    from live_btc_evidence_signal_metalabel_20260902 import compute_btc_evidence_signals
    out = compute_btc_evidence_signals()
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CTX_DIR = ROOT / "data/labels/btc_5m_evidence_signal_live_contexts_20260902"
CTX_REPORT = CTX_DIR / "contexts_report.json"
SYMBOL = "BTCUSDT"
SCORE_TAIL_BARS = 3          # 최근 몇 봉을 채점할지
_MODELS: dict[str, Any] | None = None
_META: dict[str, Any] | None = None


def _meta() -> dict[str, Any]:
    global _META
    if _META is None:
        _META = json.loads(CTX_REPORT.read_text())
    return _META


def _load_models() -> dict[str, Any]:
    """신호별 동결 컨텍스트로 TabPFN을 적합해 캐시한다(첫 호출만 비용)."""
    global _MODELS
    if _MODELS is not None:
        return _MODELS
    from tabpfn import TabPFNClassifier
    out: dict[str, Any] = {}
    for name, info in _meta()["signals"].items():
        if "error" in info:
            continue
        csv = ROOT / info["artifact"]
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        feats = info["features"]
        if df["hit"].nunique() < 2:
            continue
        clf = TabPFNClassifier(device="cuda", random_state=20260902,
                               ignore_pretraining_limits=True)
        clf.fit(df[feats], df["hit"].to_numpy())
        out[name] = {"clf": clf, "features": feats,
                     "train_hit_rate": info["hit_rate"], "params": info["btc_params"]}
    _MODELS = out
    return out


def _build_frame() -> pd.DataFrame | None:
    """라이브 BTC 봉 -> 후보 빌드와 동일한 지표 프레임.

    ⚠️`build_btc_5m_evidence_signal_candidates_tier0_20260901.py`가 쓰는 것과 **같은 함수들**을
    재사용한다. 재구현하면 학습/추론 패리티가 조용히 깨진다.
    """
    from analyze_eth_broad_evidence_signal_sweep_20260814 import add_broad_indicators
    from analyze_eth_creative_reversal_evidence_signals_20260814 import add_creative_indicators
    from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import compute_indicators
    import requests

    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                         params={"symbol": SYMBOL, "interval": "5m", "limit": 1500}, timeout=15)
        r.raise_for_status()
        raw = r.json()
    except Exception:                                          # noqa: BLE001
        return None
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qv",
            "trades", "taker_buy_base", "tq", "ignore"]
    kl = pd.DataFrame(raw, columns=cols)
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        kl[c] = kl[c].astype(float)
    kl["timestamp"] = pd.to_datetime(kl["open_time"], unit="ms", utc=True)
    kl = kl.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    now_ms = int(time.time() * 1000)
    if len(kl) and int(kl.iloc[-1]["close_time"]) >= now_ms:
        kl = kl.iloc[:-1].reset_index(drop=True)               # 형성중 봉 제거
    if len(kl) < 900:
        return None
    f = compute_indicators(kl)
    f = add_creative_indicators(f)
    f = add_broad_indicators(f)

    # ⚠️후보 빌드(build_btc_5m_evidence_signal_candidates_tier0_20260901.py:159~173)와
    # **동일한 공식**으로 파생 피쳐를 만든다. 여기가 어긋나면 학습/추론 패리티가 깨진다.
    bspec = importlib.util.spec_from_file_location(
        "btc_cand_build",
        ROOT / "scripts/build_btc_5m_evidence_signal_candidates_tier0_20260901.py")
    bmod = importlib.util.module_from_spec(bspec)
    bspec.loader.exec_module(bmod)
    sweep_impl = bmod.load_sweep_impl()

    ret3 = f["close"] / f["close"].shift(3) - 1.0
    f["ret3_z"] = (ret3 - ret3.rolling(288, min_periods=288).mean()) \
        / ret3.rolling(288, min_periods=288).std().replace(0.0, np.nan)
    causal = sweep_impl.add_causal_columns(kl[["timestamp", "open", "high", "low", "close"]].copy())
    f["sweep_level_low"] = causal["sweep_level_low"]
    f["sweep_level_high"] = causal["sweep_level_high"]
    f["atr"] = causal["atr"]
    f["atr_percentile_864"] = f["atr"].rolling(864, min_periods=864).rank(pct=True)
    f["range_width_pct"] = (f["sweep_level_high"] - f["sweep_level_low"]) / f["close"]
    f["hour_utc"] = f["timestamp"].dt.hour
    f["weekday"] = f["timestamp"].dt.weekday
    f["rsi"] = bmod.rsi_wilder(f["close"])

    # 신호 전용 파생 2개 -- 각 연구 스크립트의 계산부를 그대로 import
    dspec = importlib.util.spec_from_file_location(
        "btc_dem", ROOT / "scripts/research_btc_demarker_extreme_metalabel_tabpfn_20260901.py")
    dmod = importlib.util.module_from_spec(dspec); dspec.loader.exec_module(dmod)
    f["dem"] = dmod.compute_demarker(f["high"], f["low"]).to_numpy()
    kspec = importlib.util.spec_from_file_location(
        "btc_kal", ROOT / "scripts/research_btc_kalman_deviation_meanrev_metalabel_tabpfn_20260901.py")
    kmod = importlib.util.module_from_spec(kspec); kspec.loader.exec_module(kmod)
    f["kalman_dev_z"] = kmod.compute_kalman_dev_z(f["close"].to_numpy())

    # ⚠️Tier0 CSV에 없어 각 연구 스크립트가 자체 추가하던 4개 파생 피쳐
    # (atr_pct 재계산 / nyse_open_flag / er_24 / realized_vol_ratio).
    # **재구현하지 않고 원본 함수를 import**한다 -- 여기서 다시 쓰면 학습/추론 패리티가
    # 조용히 깨진다(스모크 테스트에서 실제로 missing features로 드러났던 지점).
    spec = importlib.util.spec_from_file_location(
        "btc_ls_prep",
        ROOT / "scripts/research_btc_liquidity_sweep_metalabel_tabpfn_20260901.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    if "atr" not in f.columns and "atr_price" in f.columns:
        f["atr"] = f["atr_price"]
    if "atr_pct" not in f.columns:
        f["atr_pct"] = f["atr"] / f["close"].clip(lower=1e-12)
    f = prep.add_missing_features(f)

    return f


def compute_btc_evidence_signals() -> dict[str, Any]:
    """최근 봉들의 신호별 발동 여부 + 메타라벨 확률. 주문은 내지 않는다."""
    empty = {"warmed_up": False, "error": None, "asset": SYMBOL, "signals": {},
             "note": "재량 참고용 컨텍스트 -- 매매 신호 아님, 주문 없음"}
    try:
        if not CTX_REPORT.exists():
            return {**empty, "error": "frozen_contexts_missing"}
        frame = _build_frame()
        if frame is None:
            return {**empty, "error": "klines_unavailable"}
        from live_evidence_signal_dashboard_20260823 import compute_signals
        sig = compute_signals(frame, btc_df=None, funding_df=None)

        models = _load_models()
        n = len(frame)
        lo = max(0, n - SCORE_TAIL_BARS)
        out: dict[str, Any] = {}
        for name, m in models.items():
            fired = {}
            for side in ("bottom", "top"):
                col = f"{side}_{name}"
                if col not in sig.columns:
                    continue
                arr = sig[col].fillna(False).to_numpy()
                idx = [i for i in range(lo, n) if arr[i]]
                if not idx:
                    continue
                rows = frame.iloc[idx].copy()
                rows["is_bottom"] = 1 if side == "bottom" else 0
                miss = [c for c in m["features"] if c not in rows.columns]
                if miss:
                    fired[side] = {"error": f"missing features: {miss[:4]}"}
                    continue
                X = rows[m["features"]].apply(pd.to_numeric, errors="coerce")
                if X.isna().any(axis=1).all():
                    continue
                ok = ~X.isna().any(axis=1)
                p = m["clf"].predict_proba(X.loc[ok])[:, 1]
                fired[side] = {
                    "n_fires": int(ok.sum()),
                    "latest_proba": round(float(p[-1]), 4) if len(p) else None,
                    "latest_bar_utc": str(frame["timestamp"].iloc[idx[-1]]),
                }
            out[name] = {"fired": fired, "train_hit_rate": m["train_hit_rate"],
                         "btc_params": m["params"]}
        return {**empty, "warmed_up": True, "signals": out,
                "latest_bar_utc": str(frame["timestamp"].iloc[-1]),
                "n_models": len(models)}
    except Exception as e:                                     # noqa: BLE001
        return {**empty, "error": f"{type(e).__name__}: {e}"}


# ⚠️2026-09-02에 이 함수를 추가하며 발견: `_meta()`의 신호 키("taker_delta_climax")가
# `compute_signals()`의 실제 컬럼명("taker_delta_z_climax", ETH와 공유)과 다르다 --
# build_btc_5m_evidence_signal_candidates_tier0_20260901.py는 NAMED_TRIGGERS에 "z"를 포함한
# 이름을 쓰는데, contexts_report.json(그리고 이를 그대로 쓰는 HIT_SPEC/HOLDOUT_AUC, 섀도우
# 러너)만 축약된 이름을 쓴다. 위 compute_btc_evidence_signals()는 `col = f"{side}_{name}"`를
# 그대로 sig.columns에서 찾는데, "bottom_taker_delta_climax"는 존재하지 않아 항상 `continue`로
# 건너뛴다 -- 즉 **라이브 섀도우 러너에서 taker_delta_climax가 한 번도 발동을 기록하지 못하는
# 기존 버그**로 보인다(이 함수를 만들다 발견, 섀도우 러너 자체는 이번 변경 대상이 아니라
# 손대지 않음 -- 사용자에게 별도 보고). 아래 신규 함수는 이 별칭 매핑으로 올바른 컬럼을 찾는다.
RAW_COLUMN_ALIAS = {"taker_delta_climax": "taker_delta_z_climax"}

# 대시보드 메인 증거신호 패널용 fill-window 파라미터. ETH의 K_OVERRIDE/SUSTAIN_BARS_OVERRIDE
# (live_evidence_signal_dashboard_20260823.py)를 재사용하면 안 된다 -- compute_signals()는
# 자산과 무관하게 공유되지만, 그 안의 fill-window 계산은 ETH 전용 K/HORIZON을 참조하므로 BTC
# 프레임에 그대로 적용하면 조용히 다른 자산의 보정값이 들어간다. 아래 값은 BTC 자체 그리드스크린
# 결과(live_btc_evidence_signal_shadow_runner_20260902.py::HIT_SPEC와 동일 출처, 이 파일이
# import하는 대신 값만 복제 -- 순환 임포트 회피 + 러너 프로세스를 건드리지 않기 위함)다.
# ⚠️2026-09-03: `mode`를 추가했다. 그 전엔 7종 전부 터치로만 채웠는데, BTC 라벨은 자산별로
# 재스크리닝돼 ETH와 다르다 -- taker/fib는 **종가 기준**이라 중간 터치로 끝나지 않고,
# liquidity_sweep은 되돌림(giveback) 조건이, short_term_return_z는 MAE 상한이 더 붙는다.
# 각 신호의 라벨 스크립트에서 직접 확인했다(근거는 shadow_runner의 HIT_SPEC 주석 참조).
FILL_SPEC = {
    "taker_delta_climax":       {"k": 2.0,  "horizon": 6,  "mode": "close_at_h"},
    "liquidity_sweep":          {"k": 2.0,  "horizon": 20, "mode": "full_window",
                                 "full_window": 40},
    "kalman_deviation_meanrev": {"k": 3.5,  "horizon": 10, "mode": "touch"},
    "short_term_return_z":      {"k": 2.0,  "horizon": 6,  "mode": "touch_mae_capped",
                                 "k_loss_mult": 2.0},
    "orthogonal_combo":         {"k": 2.0,  "horizon": 8,  "mode": "touch"},
    "demarker_extreme":         {"k": 0.70, "horizon": 8,  "mode": "touch"},
    "fib_extension_exhaustion": {"k": 2.75, "horizon": 10, "mode": "close_at_h"},
}
PANEL_HISTORY_BARS = 48  # dashboard/server.py's EVIDENCE_SIGNAL_HISTORY_BARS(4h @ 5m)와 동일


def _fill_until_tp_or_horizon(raw: pd.Series, k: float, horizon_bars: int, side: str,
                               high: pd.Series, low: pd.Series, close: pd.Series,
                               atr_pct: pd.Series, mode: str = "touch",
                               k_loss_mult: float = 2.0) -> pd.Series:
    """live_evidence_signal_dashboard_20260823.py::compute_signals()의 동명 내부함수와 같은
    알고리즘(발동 시점부터 K*ATR% 터치 또는 HORIZON 경과 중 먼저 오는 시점까지 채움) --
    그쪽은 ETH의 high/low/close/atr_pct를 클로저로 참조하는 중첩함수라 그대로 import할 수
    없어(그리고 애초에 K/HORIZON도 자산별로 달라야 하므로) 순수함수로 복제했다."""
    n = len(raw)
    filled = np.zeros(n, dtype=bool)
    raw_arr = raw.fillna(False).to_numpy()
    high_a, low_a, close_a, atr_a = high.to_numpy(), low.to_numpy(), close.to_numpy(), atr_pct.to_numpy()
    # ⭐mode = 그 신호의 라벨이 **언제 확정되는가**.
    #   touch            터치 봉에서 확정 -> 거기서 끝
    #   touch_mae_capped 터치 봉까지의 MAE로 hit이 완전 확정 -> 터치 봉에서 끝
    #   close_at_h       H봉 종가로만 판정 -> 중간 터치로 못 끝냄, 전 구간 유지
    #   full_window      giveback이 close[i+FULL_WINDOW]를 필요로 함 -> 전 구간 유지
    for i in np.flatnonzero(raw_arr):
        end = min(i + horizon_bars, n - 1)
        if mode in ("touch", "touch_mae_capped") and not np.isnan(atr_a[i]):
            target = k * atr_a[i]
            level = close_a[i] * (1 - target) if side == "top" else close_a[i] * (1 + target)
            for b in range(i + 1, end + 1):
                touched = (low_a[b] <= level) if side == "top" else (high_a[b] >= level)
                if touched:
                    end = b               # MAE 상한 여부와 무관하게 **확정 시점**은 터치 봉이다
                    break
        filled[i:end + 1] = True
    return pd.Series(filled, index=raw.index)


def compute_btc_evidence_signals_panel(history_bars: int = PANEL_HISTORY_BARS) -> dict[str, Any]:
    """대시보드 메인 증거신호 패널(스냅샷 탭 '증거신호' 컴포넌트)용 payload -- ETH의
    dashboard/server.py::load_evidence_signals()와 같은 모양(signals 리스트, 신호마다
    bottom_history/top_history/bottom_raw_fire/top_raw_fire/bottom_fired/top_fired[/model_proba/
    model_side]) 이라 프론트엔드 renderEvidenceSignals()/evidenceStripSvg()를 그대로 재사용한다.
    위 compute_btc_evidence_signals()(섀도우 러너 전용, 그 모양에 의존하는 프로세스가 이미
    돌고 있어 변경하지 않음)와는 별개 함수지만 _build_frame()/_load_models() 캐시는 공유한다.
    ⚠️BTC는 ETH가 통과한 경제성 게이트를 통과하지 못했다(이 모듈 docstring +
    docs/experiments/btc_evidence_signal_economics_gate_20260902.md 참고). 그래도 노출하는 이유는
    이 대시보드의 증거신호 티어 자체가 애초에 손익 주장이 아니라 정보성(IC) 표시이기 때문 --
    ETH 신호들도 전부 그 지위이고, 여기 model_proba/train_hit_rate는 참고용 확률일 뿐 매매
    신호가 아니다(라벨 텍스트에도 명시)."""
    empty: dict[str, Any] = {"warmed_up": False, "error": None, "signals": []}
    try:
        if not CTX_REPORT.exists():
            return {**empty, "error": "frozen_contexts_missing"}
        frame = _build_frame()
        if frame is None:
            return {**empty, "error": "klines_unavailable"}
        from live_evidence_signal_dashboard_20260823 import compute_signals
        sig = compute_signals(frame, btc_df=None, funding_df=None)
        models = _load_models()

        n = len(sig)
        signals_payload: list[dict[str, Any]] = []
        bottom_votes = 0
        top_votes = 0
        for name, m in models.items():
            spec = FILL_SPEC.get(name)
            raw_name = RAW_COLUMN_ALIAS.get(name, name)
            bcol, tcol = f"bottom_{raw_name}", f"top_{raw_name}"
            if spec is None or bcol not in sig.columns or tcol not in sig.columns:
                continue
            fill_h = int(spec.get("full_window", spec["horizon"]))
            bottom_fill = _fill_until_tp_or_horizon(sig[bcol], spec["k"], fill_h, "bottom",
                                                     sig["high"], sig["low"], sig["close"],
                                                     sig["atr_pct"], spec.get("mode", "touch"))
            top_fill = _fill_until_tp_or_horizon(sig[tcol], spec["k"], fill_h, "top",
                                                  sig["high"], sig["low"], sig["close"],
                                                  sig["atr_pct"], spec.get("mode", "touch"))
            bottom_fired = bool(bottom_fill.iloc[-1])
            top_fired = bool(top_fill.iloc[-1])
            if bottom_fired:
                bottom_votes += 1
            if top_fired:
                top_votes += 1

            entry: dict[str, Any] = {
                "name": name,
                "bottom_fired": bottom_fired,
                "top_fired": top_fired,
                "bottom_history": bottom_fill.tail(history_bars).fillna(False).astype(bool).tolist(),
                "top_history": top_fill.tail(history_bars).fillna(False).astype(bool).tolist(),
                "bottom_raw_fire": sig[bcol].tail(history_bars).fillna(False).astype(bool).tolist(),
                "top_raw_fire": sig[tcol].tail(history_bars).fillna(False).astype(bool).tolist(),
            }
            # 배지에 쓸 현재봉 확률 -- 원시 발동 중인 쪽만(둘 다 발동이면 ETH와 동일하게
            # bottom 우선, dashboard/live/app.js::evidenceSideTone 참고).
            side = "bottom" if bool(sig[bcol].iloc[-1]) else ("top" if bool(sig[tcol].iloc[-1]) else None)
            if side is not None:
                row = sig.iloc[[n - 1]]
                miss = [c for c in m["features"] if c not in row.columns]
                if not miss:
                    X = row[m["features"]].apply(pd.to_numeric, errors="coerce")
                    if not X.isna().any(axis=1).all():
                        p = float(m["clf"].predict_proba(X)[:, 1][0])
                        entry["model_proba"] = round(p, 4)
                        entry["model_side"] = side
            signals_payload.append(entry)

        return {
            "warmed_up": True,
            "error": None,
            "latest_bar_utc": str(sig["timestamp"].iloc[-1]),
            "net_score": bottom_votes - top_votes,
            "bottom_votes": bottom_votes,
            "top_votes": top_votes,
            "signals": signals_payload,
        }
    except Exception as e:                                     # noqa: BLE001
        return {**empty, "error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    t0 = time.time()
    r = compute_btc_evidence_signals()
    r["cycle_sec"] = round(time.time() - t0, 2)
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
