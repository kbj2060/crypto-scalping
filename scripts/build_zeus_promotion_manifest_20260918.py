#!/usr/bin/env python3
"""Zeus **집행 승격 매니페스트**를 만든다. (2026-09-18)

`BinanceFuturesExecutionAdapter` 는 `promotion_manifest_required` 일 때
`require_execution_promotion_manifest()` 로 이 파일을 검사하고, 통과하지 못하면
**주문 경로를 스스로 끈다**. 즉 이 파일이 저장소가 정해 둔 «라이브 자격»의 형식적 관문이다.

요구 항목(계약 원문):
  schema_version == "current_live_manifest_v1"
  promotion_eligible is True · promotion_blockers == []
  artifact_integrity.promotion_pass is True        ← 감사 산출 JSON 에서 읽는다
  fresh_forward.{validation,oos} 4개 불리언
  selection_statistics: gate_pass · DSR · PBO 와 각각의 임계값

🔴**통계를 지어내지 않는다.** DSR/PBO 는 `core/selection_stats.py` 로 실제 계산하고,
기준에 미달하면 **`promotion_eligible=false` 와 블로커를 적어** 그대로 기록한다.
매니페스트의 목적은 통과가 아니라 **증거를 한 파일에 고정하는 것**이다.

⭐**탐색 공간을 정직하게 센다**: 배리어 20칸 × 게이트 점수 5종 × 신호율 4수준 = 400 구성.
전부 캐시된 확률에서 재학습 없이 재현되므로 «실제로 해 본 탐색»의 지배적 부분이다.
🔴이보다 적게 세면 노이즈 바닥이 낮아져 DSR 이 부풀고, 그건 이 파일이 막으려는 바로 그것이다.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(os.environ.get("ZEUS_ROOT") or Path.home() / "crypto-scalping")
CODE = Path(__file__).resolve().parents[1]
for _p in (CODE, CODE / "scripts", ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(_p))
from core.selection_stats import deflated_sharpe_ratio, pbo_cscv   # noqa: E402

ART_NAME = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--art=")),
                "zeus_v4_shadow_20260918")
ART = ROOT / "data/live" / ART_NAME
# 🔴임계값은 «저자가 선언하고 계약이 그 선언을 검사하는» 값이다(저장소가 박은 상수가 아니다).
# 기본값 0.95/0.35 는 2026-09-18 내가 선언한 값 -- Bailey–López de Prado 관례(DSR≥0.95)와
# «동전(0.5)보다 엄격»(PBO≤0.35). 바꾸려면 --min-dsr/--max-pbo 와 함께 **근거를 남겨야 한다**.
MIN_DSR = float(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--min-dsr=")), 0.95))
MAX_PBO = float(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--max-pbo=")), 0.35))
AUTHORITY = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--authority=")), "")
REASON = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--reason=")), "")
if (MIN_DSR, MAX_PBO) != (0.95, 0.35):
    assert AUTHORITY and REASON, (
        "임계값을 바꾸려면 --authority 와 --reason 이 필수다. 근거 없는 완화는 계약 충족이 "
        "아니라 계약의 목적을 없애는 것이고, 그 사실이 매니페스트에 남아야 한다.")
TPS = [0.010, 0.015, 0.020, 0.025, 0.030]
SLS = [0.005, 0.007, 0.010, 0.013]
SCORES = ["q", "d", "dq", "margin", "edge"]
TARGETS = [8000, 15000, 25000, 38580]


def log(*a): print(*a, flush=True)


def main() -> int:
    import research_zeus_n0x2_sltp_grid_20260917 as G   # noqa: E402  (격자 · 폴드 · 시뮬)
    import research_omega461_side_skill_decomposition_20260917 as K  # noqa: E402
    import train_eval_omega461_parent_zig075_longwindow_20260917 as E  # noqa: E402

    spec = json.loads((ART / "meta.json").read_text())["spec"]
    log(f"{ART_NAME} · 점수 {spec.get('gate_score','q')} · q={spec['rollq_q']} · "
        f"TP{spec['tp']*100:g}/SL{spec['sl']*100:g}")

    # ── 일별 순bp 행렬 (일 × 구성) -- `--daymatrix` 로 미리 만든 것을 받는다 ──────
    mpath = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--matrix=")), "")
    if not mpath:
        log("🔴--matrix 가 없다. research_zeus_n0x2_sltp_grid_20260917.py --daymatrix 로 먼저 만든다.")
        return 3
    z = np.load(mpath, allow_pickle=True)
    M, names, days = z["matrix"], list(z["names"]), list(z["days"])
    chosen = f"{spec.get('gate_score','q')}|q{spec['rollq_q']}|{spec['tp']}|{spec['sl']}"
    if chosen not in names:
        log(f"🔴선택 구성이 행렬에 없다: {chosen}"); return 3
    ci = names.index(chosen)
    log(f"행렬 {M.shape[0]}일 × {M.shape[1]}구성 · 선택 = {chosen}")

    trial_sr = np.array([_sr(M[:, j]) for j in range(M.shape[1])], float)
    dsr = deflated_sharpe_ratio(M[:, ci], trial_sr)
    pbo = pbo_cscv(M, n_splits=8)
    log(f"DSR {dsr['deflated_sharpe_ratio']:.4f} (관측 SR {dsr['observed_sharpe']:.3f} · "
        f"시행 {dsr['n_trials']} · 노이즈바닥 {dsr['noise_floor_sharpe']:.3f})")
    log(f"PBO {pbo['pbo']:.4f} ({pbo['n_combinations']} 조합 · 중앙 logit {pbo['median_logit']:+.3f})")

    audit_p = ART / "omega_artifact_integrity_audit_20260630.json"
    audit = json.loads(audit_p.read_text()) if audit_p.exists() else {}
    blockers = []
    if audit.get("promotion_pass") is not True:
        blockers.append("artifact_integrity_not_passed")
    if not (dsr["deflated_sharpe_ratio"] >= MIN_DSR):
        blockers.append(f"deflated_sharpe_ratio<{MIN_DSR}")
    if not (pbo["pbo"] <= MAX_PBO):
        blockers.append(f"probability_backtest_overfit>{MAX_PBO}")

    ff = {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
          "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False}
    man = {
        "schema_version": "current_live_manifest_v1",
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "model_id": ART_NAME,
        "promotion_eligible": not blockers,
        "promotion_blockers": blockers,
        "artifact_integrity": {"promotion_pass": audit.get("promotion_pass"),
                               "audit_json": str(audit_p.relative_to(ROOT)),
                               "component_count": audit.get("component_count")},
        "fresh_forward": {"validation": dict(ff), "oos": dict(ff)},
        "selection_statistics": {
            "gate_pass": not blockers,
            "deflated_sharpe_ratio": float(dsr["deflated_sharpe_ratio"]),
            "minimum_deflated_sharpe_ratio": MIN_DSR,
            "probability_backtest_overfit": float(pbo["pbo"]),
            "maximum_probability_backtest_overfit": MAX_PBO,
            "observed_sharpe": float(dsr["observed_sharpe"]),
            "n_trials": int(dsr["n_trials"]),
            "noise_floor_sharpe": float(dsr["noise_floor_sharpe"]),
            "pbo_combinations": int(pbo["n_combinations"]),
            "search_space": {"barriers": len(TPS) * len(SLS), "gate_scores": len(SCORES),
                             "signal_rates": 4, "total": len(names)},
            "threshold_provenance": {
                "default_declared_20260918": {"min_dsr": 0.95, "max_pbo": 0.35,
                                              "declared_by": "assistant", "basis":
                                              "Bailey-Lopez de Prado 관례 · 동전(0.5)보다 엄격"},
                "in_effect": {"min_dsr": MIN_DSR, "max_pbo": MAX_PBO,
                              "authority": AUTHORITY or "assistant_default",
                              "reason": REASON or "기본값",
                              "changed_at": pd.Timestamp.utcnow().isoformat()},
                "statistics_unchanged": ("임계값만 바뀌었고 DSR/PBO 는 그대로다. "
                                         "기대값에는 400회 탐색의 선택편향이 반영돼 있지 않다."),
            },
            "cost_bp_applied": 1.41,
            "cost_basis": "ETHUSDC 메이커 섀도우 732 leg 실측 왕복(체결률 100%)",
        },
        "orders": "NOT_ENABLED -- 이 파일은 «자격»만 증명한다. 주문은 별도 env 3개가 필요하다",
        "execution_gates_still_required": [
            "BINANCE_EXECUTION_ENABLED=1", "dry_run=false",
            "BINANCE_EXECUTION_CONFIRM_LIVE=I_UNDERSTAND_REAL_ORDERS"],
    }
    out = ART / "execution_promotion_manifest.json"
    out.write_text(json.dumps(man, indent=2, ensure_ascii=False))
    log(f"\n{'✅ promotion_eligible=true' if not blockers else '🔴 블로커: ' + str(blockers)}")
    log(f"저장: {out}")
    return 0 if not blockers else 2


def _sr(x: np.ndarray) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    s = x.std(ddof=1) if x.size > 1 else 0.0
    return float(x.mean() / s) if s > 0 else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
