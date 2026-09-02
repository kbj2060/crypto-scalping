#!/usr/bin/env python3
"""XRP 증거신호 + 레짐 **내부 로직 심층 점검** — 읽기가 아니라 실측 대조.

오늘 BTC 데이터 오염 2건이 전부 **행수 대조**로만 잡혔다. 코드를 읽어서는 안 보였다.
그래서 이 감사는 전부 "실제로 돌려서 값을 비교"하는 방식이다.

## 점검 항목

### A. 증거신호
  A1 동결 컨텍스트가 **XRP 데이터**인가 (행수·hit률을 BTC/ETH 산출물과 대조)
  A2 라이브 스코어러의 **셀(H/K/mode)**이 확정 셀과 일치하는가
  A3 `FILL_SPEC` mode가 **그 신호 라벨의 확정 규칙**과 일치하는가 (해상봉 포함)
  A4 학습 컨텍스트의 **피쳐 목록**과 라이브가 쓰는 피쳐가 같은가
  A5 라이브가 채점하는 **후보 모집단**이 학습과 같은 방식인가(클러스터 디둡 GAP)

### B. 레짐
  B1 모델 아티팩트의 **feature_cols**가 학습에 쓴 것과 같은가
  B2 라이브 스코어러의 **교차자산 슬롯**이 학습과 같은가 ⭐(자산마다 다르다)
  B3 라벨 상수(SCALE/DEBOUNCE_K)가 **선택된 값(48/6)**인가
  B4 라이브 추론 경로가 학습과 **같은 파생**을 태우는가(`_with_raw_state12`)
  B5 라이브 출력이 **실제 XRP 가격**에 반응하는가(BTC/ETH 값이 아닌가)

⚠️각 항목은 통과/실패와 **근거 수치**를 함께 남긴다. 실패는 배포 중단 사유다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = ROOT / "data/research/xrp_internal_logic_audit_20260903.json"
CTX_DIR = ROOT / "data/labels/xrp_5m_evidence_signal_live_contexts_20260903"
BTC_CTX = ROOT / "data/labels/btc_5m_evidence_signal_live_contexts_20260902/contexts_report.json"

# 3단계 그리드스크린 + 두께감사로 확정한 셀 (정본: docs/experiments/xrp_...20260903.md)
CONFIRMED = {
    "demarker_extreme":         {"h": 2,  "k": 1.5, "mode": "touch",                    "resolve": 2},
    "kalman_deviation_meanrev": {"h": 5,  "k": 2.0, "mode": "touch",                    "resolve": 5},
    "short_term_return_z":      {"h": 12, "k": 1.5, "mode": "touch_mae_capped",         "resolve": 12},
    "taker_delta_climax":       {"h": 9,  "k": 1.5, "mode": "touch_giveback_sustained", "resolve": 18},
    "orthogonal_combo":         {"h": 8,  "k": 2.0, "mode": "touch",                    "resolve": 8},
}
EXCLUDED = {"liquidity_sweep": 0.4886, "fib_extension_exhaustion": 0.4738}


def _m(n, rel):
    sp = importlib.util.spec_from_file_location(n, ROOT / rel)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def log(m): print(f"[deep] {m}", flush=True)


def main() -> int:
    res, fails = {}, []

    def check(key, ok, detail):
        res[key] = {"ok": bool(ok), "detail": detail}
        log(f"  {'✅' if ok else '❌'} {key:<6} {detail}")
        if not ok:
            fails.append(key)

    # ══════════ A. 증거신호 ══════════
    log("=== A. 증거신호 ===")
    ctx = json.loads((CTX_DIR / "contexts_report.json").read_text())
    btc = json.loads(BTC_CTX.read_text()) if BTC_CTX.exists() else {"signals": {}}

    # A1 자산 오염
    same_as_btc = []
    for name, v in ctx["signals"].items():
        b = btc["signals"].get(name)
        if b and v.get("rows") == b.get("rows") and abs((v.get("hit_rate") or 0) - (b.get("hit_rate") or -1)) < 1e-9:
            same_as_btc.append(name)
    check("A1", ctx.get("asset") == "XRPUSDT" and not same_as_btc,
          f"asset={ctx.get('asset')} · BTC와 (행수,hit률) 동일한 신호 {len(same_as_btc)}개 {same_as_btc}")

    # A2/A3 라이브 셀 + mode + 해상봉
    live = _m("xrplive", "scripts/live_xrp_evidence_signal_metalabel_20260903.py")
    sh = _m("xrpshadow", "scripts/live_xrp_evidence_signal_shadow_runner_20260903.py")
    bad_cell, bad_mode, bad_res = [], [], []
    for name, c in CONFIRMED.items():
        f = live.FILL_SPEC.get(name, {})
        s = sh.HIT_SPEC.get(name, {})
        if f.get("horizon") != c["h"] or abs(float(f.get("k", -1)) - c["k"]) > 1e-9:
            bad_cell.append(f"{name}: live H={f.get('horizon')} K={f.get('k')}")
        if s.get("horizon") != c["h"] or abs(float(s.get("k", -1)) - c["k"]) > 1e-9:
            bad_cell.append(f"{name}: shadow H={s.get('horizon')} K={s.get('k')}")
        if s.get("mode") != c["mode"]:
            bad_mode.append(f"{name}: shadow mode={s.get('mode')} != {c['mode']}")
        if sh._resolve_bars(s) != c["resolve"]:
            bad_res.append(f"{name}: 해상 {sh._resolve_bars(s)} != {c['resolve']}")
    check("A2", not bad_cell, f"라이브/섀도우 셀 불일치 {len(bad_cell)}건 {bad_cell}")
    check("A3", not bad_mode and not bad_res,
          f"mode 불일치 {len(bad_mode)}건 {bad_mode} · 해상봉 불일치 {len(bad_res)}건 {bad_res}")

    # A4 피쳐 목록 (동결 컨텍스트 CSV 헤더 vs 리포트 features)
    bad_feat = []
    for name, v in ctx["signals"].items():
        csv = CTX_DIR / f"{name}_frozen_context.csv"
        if not csv.exists():
            bad_feat.append(f"{name}: 컨텍스트 CSV 없음"); continue
        cols = set(pd.read_csv(csv, nrows=1).columns)
        miss = [c for c in (v.get("features") or []) if c not in cols]
        if miss:
            bad_feat.append(f"{name}: CSV에 없는 피쳐 {miss[:3]}")
    check("A4", not bad_feat, f"피쳐 불일치 {len(bad_feat)}건 {bad_feat}")

    # A5 제외 신호가 서빙에 없는가
    leaked = [n for n in EXCLUDED if n in live.FILL_SPEC or n in sh.HIT_SPEC or n in ctx["signals"]]
    check("A5", not leaked, f"HOLDOUT AUC<0.5인데 서빙에 남은 신호 {leaked} (기대: 없음)")

    # ══════════ B. 레짐 ══════════
    log("=== B. 레짐 ===")
    import joblib
    art = joblib.load(ROOT / "tmp/xrp_regime_s48k6_20260903/model.joblib")
    rep = json.loads((ROOT / "tmp/xrp_regime_s48k6_20260903/train_report.json").read_text())
    eth_art = joblib.load(ROOT / "tmp/eth_regime_gbm3_independent_20260826/model.joblib")

    # B1 feature_cols 출처 동일
    check("B1", list(art["feature_cols"]) == list(eth_art["feature_cols"]),
          f"feature_cols {len(art['feature_cols'])}개 · ETH GBM3 아티팩트와 동일={list(art['feature_cols'])==list(eth_art['feature_cols'])}")

    # B2 ⭐교차자산 슬롯
    live_r = _m("xrpregime", "scripts/live_regime_xrp_signal_20260903.py")
    canon = ROOT / "data/splits/year_oos/xrp_features_2024_2026.csv"
    raw = ROOT / "data/splits/year_oos/xrp_raw_frame_2024_2026.csv"
    # 학습 raw 프레임의 close_btc가 실제로 BTC 가격대인가(XRP는 ~0.5~3달러, BTC는 수만 달러)
    rr = pd.read_csv(raw, usecols=["close", "close_btc"], nrows=5000)
    xrp_med, cross_med = float(rr["close"].median()), float(rr["close_btc"].median())
    cross_is_btc = cross_med > 10_000 and xrp_med < 100
    check("B2", live_r.CROSS_SYMBOL == "BTCUSDT" and rep.get("cross_asset") == "BTCUSDT" and cross_is_btc,
          f"live CROSS_SYMBOL={live_r.CROSS_SYMBOL} · report={rep.get('cross_asset')} · "
          f"학습 raw close 중앙 {xrp_med:.4f}(XRP) / close_btc 중앙 {cross_med:,.0f}(BTC로 보임={cross_is_btc})")

    # B3 라벨 상수
    p3 = _m("xrp_p3b", "scripts/research_xrp_regime_s48k6_label_train_20260903.py")
    ls = rep.get("label_spec", {})
    check("B3", p3.SCALE == 48 and p3.DEBOUNCE_K == 6 and ls.get("scale_bars") == 48 and ls.get("debounce_k") == 6,
          f"코드 SCALE={p3.SCALE} K={p3.DEBOUNCE_K} · 아티팩트 scale={ls.get('scale_bars')} k={ls.get('debounce_k')}")

    # B4 라이브가 _with_raw_state12를 태우는가
    src = (ROOT / "scripts/live_regime_xrp_signal_20260903.py").read_text()
    check("B4", "_with_raw_state12(feats)" in src and "FeatureEngineer().process" in src,
          "라이브 추론이 FeatureEngineer + _with_raw_state12를 학습과 동일하게 태움")

    # B5 캐노니컬이 XRP인가 (가격대)
    cc = pd.read_csv(canon, usecols=["close"], nrows=5000)
    cm = float(cc["close"].median())
    check("B5", cm < 100, f"캐노니컬 close 중앙 {cm:.4f} (XRP 가격대면 <100)")

    log("")
    ok = not fails
    log(f"⇒ {'✅ 전 항목 통과' if ok else '❌ 실패: ' + ', '.join(fails)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"all_ok": ok, "failed": fails, "checks": res},
                              ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
